"""Bounded, measurement-only telemetry for the fixed G1 PPO worker."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from oracle_composition.harness.contract import decode_json_object

from .contracts import OBSERVATION_DIM
from .course_task import TASK_FEATURE_NAMES
from .training_contract import TRAINING_REWARD_INFO_ID, TRAINING_REWARD_SCALE
from .training_normalizer import (
    FIXED_NORMALIZER_STATE_SHA256,
    FixedNormalizerState,
    normalized_base_max_abs,
)

TELEMETRY_ID = "gmt_g1_ppo_training_telemetry/v1"
TELEMETRY_FILENAME = "training_telemetry_v1.jsonl"
SCALED_TELEMETRY_ID = "gmt_g1_ppo_training_telemetry/v2"
SCALED_TELEMETRY_FILENAME = "training_telemetry_v2.jsonl"
FIXED_NORMALIZER_TELEMETRY_ID = "gmt_g1_ppo_training_telemetry/v3"
FIXED_NORMALIZER_TELEMETRY_FILENAME = "training_telemetry_v3.jsonl"
ROLLOUT_STEPS = 512
MAX_TELEMETRY_BYTES = 4 * 1024**2

_GROUP_WIDTHS = {"base": OBSERVATION_DIM, "task": len(TASK_FEATURE_NAMES), "phase": 6}
_LOGGER_FIELDS = {
    "approx_kl": "train/approx_kl",
    "clip_fraction": "train/clip_fraction",
    "entropy_loss": "train/entropy_loss",
    "policy_gradient_loss": "train/policy_gradient_loss",
    "value_loss": "train/value_loss",
    "explained_variance": "train/explained_variance",
    "policy_std": "train/std",
}
_METRIC_FLOATS = (
    "progress_m",
    "lateral_error_m",
    "heading_error_rad",
    "root_height_m",
    "forward_speed_m_s",
    "speed_error_m_s",
    "posture_band_error_m",
    "joint_position_rmse_rad",
    "root_height_abs_error_m",
    "roll_pitch_rmse_rad",
)
_METRIC_BOOLS = (
    "inside_posture_region",
    "posture_success",
    "finish_condition_met",
    "horizon_reached",
    "fallen",
)
_DESCRIPTOR_FIELDS = {
    "schema_version",
    "telemetry_id",
    "path",
    "sha256",
    "size_bytes",
    "record_count",
    "rollout_boundary_count",
    "final_update_recorded",
    "rollout_boundary_semantics",
    "sb3_n_updates_semantics",
}
_ROLLOUT_SEMANTICS = "current_rollout_with_previous_update_then_final_flush"
_UPDATE_COUNT_SEMANTICS = "attempted_ppo_epochs_including_kl_stopped_partial_epochs"
_NORMALIZED_RANGE_SEMANTICS = (
    "maximum_absolute_value_after_fixed_prefix_transform_of_current_raw_rollout_buffer"
)
_UNAVAILABLE_REASONS = {"missing", "undefined_nonfinite"}


def _telemetry_identity(
    reward_scale: float | None, fixed_normalizer_sha256: str | None = None
) -> tuple[str, str, int]:
    if fixed_normalizer_sha256 is not None:
        if (
            type(reward_scale) is not float
            or reward_scale != TRAINING_REWARD_SCALE
            or fixed_normalizer_sha256 != FIXED_NORMALIZER_STATE_SHA256
        ):
            raise ValueError("fixed-normalizer telemetry requires the exact v3 trainer")
        return FIXED_NORMALIZER_TELEMETRY_ID, FIXED_NORMALIZER_TELEMETRY_FILENAME, 2
    if reward_scale is None:
        return TELEMETRY_ID, TELEMETRY_FILENAME, 1
    if type(reward_scale) is not float or reward_scale != TRAINING_REWARD_SCALE:
        raise ValueError("scaled telemetry requires the fixed 1/64 trainer factor")
    return SCALED_TELEMETRY_ID, SCALED_TELEMETRY_FILENAME, 1


def telemetry_filename(
    *, reward_scale: float | None, fixed_normalizer_sha256: str | None = None
) -> str:
    return _telemetry_identity(reward_scale, fixed_normalizer_sha256)[1]


def _float(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite numeric")
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must be finite numeric")
    result = float(array.item())
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric")
    return result


def _int(value: object, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a nonnegative integer")
    array = np.asarray(value)
    if array.shape != () or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must be a nonnegative integer")
    result = int(array.item())
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return result


def _moments(values: object, name: str) -> dict[str, float | int] | None:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not array.size:
        return None
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    mean = float(array.mean(dtype=np.float64))
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": mean,
        "standard_deviation": float(np.sqrt(np.mean((array - mean) ** 2))),
    }


def observation_group_moments(observations: object) -> dict[str, dict[str, float | int]]:
    """Summarize only values already present in one rollout buffer."""

    array = np.asarray(observations)
    width = sum(_GROUP_WIDTHS.values())
    if array.ndim < 2 or array.shape[-1] != width or not np.issubdtype(
        array.dtype, np.floating
    ):
        raise ValueError(f"rollout observations must be floating-point [..., {width}]")
    if not array.size or not np.isfinite(array).all():
        raise ValueError("rollout observations must be nonempty and finite")
    rows, start = array.reshape(-1, width).astype(np.float64, copy=False), 0
    result = {}
    for name, group_width in _GROUP_WIDTHS.items():
        summary = _moments(rows[:, start : start + group_width], f"{name} observations")
        assert summary is not None
        result[name] = {"observations": len(rows), "width": group_width, **summary}
        start += group_width
    return result


def _terminal(info: Mapping[str, object]) -> dict[str, object]:
    metrics = info.get("metrics")
    if not isinstance(metrics, Mapping) or type(metrics.get("control_step")) is not int:
        raise ValueError("training transition lacks terminal task metrics")
    result: dict[str, object] = {"control_step": metrics["control_step"]}
    for name in _METRIC_FLOATS:
        result[name] = _float(metrics.get(name), f"terminal {name}")
    for name in _METRIC_BOOLS:
        if type(metrics.get(name)) is not bool:
            raise ValueError(f"terminal {name} must be boolean")
        result[name] = metrics[name]
    return result


class EpisodeAccumulator:
    """Summarize complete episodes while carrying partial episodes between rollouts."""

    def __init__(self, reward_scale: float | None = None) -> None:
        _telemetry_identity(reward_scale)
        self.reward_scale = reward_scale
        self.returns: np.ndarray | None = None
        self.raw_returns: np.ndarray | None = None
        self.lengths: np.ndarray | None = None
        self.last: list[dict[str, object] | None] = []
        self.completed: list[dict[str, object]] = []

    def observe(
        self, rewards: object, dones: object, infos: Sequence[Mapping[str, object]]
    ) -> None:
        rewards, dones = np.asarray(rewards), np.asarray(dones)
        if (
            rewards.ndim != 1
            or dones.shape != rewards.shape
            or len(infos) != rewards.size
            or not np.issubdtype(rewards.dtype, np.number)
            or not np.isfinite(rewards).all()
        ):
            raise ValueError("training reward, done, or info vectors differ")
        if self.returns is None:
            self.returns = np.zeros(rewards.size, dtype=np.float64)
            if self.reward_scale is not None:
                self.raw_returns = np.zeros(rewards.size, dtype=np.float64)
            self.lengths = np.zeros(rewards.size, dtype=np.int64)
            self.last = [None] * rewards.size
        if rewards.shape != self.returns.shape:
            raise ValueError("training environment count changed")
        assert self.lengths is not None
        self.returns += rewards
        self.lengths += 1
        for index, (done, info) in enumerate(zip(dones, infos, strict=True)):
            raw_reward = None
            if self.reward_scale is not None:
                evidence = info.get("training_reward")
                if type(evidence) is not dict or set(evidence) != {
                    "schema_id",
                    "raw_total_reward",
                    "scaled_optimization_reward",
                    "total_training_reward_scale",
                }:
                    raise ValueError("scaled training transition lacks reward-unit evidence")
                raw_reward = _float(evidence["raw_total_reward"], "raw total reward")
                scaled_reward = _float(
                    evidence["scaled_optimization_reward"], "scaled optimization reward"
                )
                if (
                    evidence["schema_id"] != TRAINING_REWARD_INFO_ID
                    or evidence["total_training_reward_scale"] != self.reward_scale
                    or scaled_reward != float(rewards[index])
                    or scaled_reward != float(np.float32(raw_reward * self.reward_scale))
                ):
                    raise ValueError("scaled training reward evidence differs")
                assert self.raw_returns is not None
                self.raw_returns[index] += raw_reward
            terminal = _terminal(info)
            self.last[index] = terminal
            if bool(done):
                episode = {
                    "return": float(self.returns[index]),
                    "length": int(self.lengths[index]),
                    **terminal,
                }
                if raw_reward is not None:
                    assert self.raw_returns is not None
                    episode["raw_return"] = float(self.raw_returns[index])
                self.completed.append(episode)
                self.returns[index], self.lengths[index], self.last[index] = 0.0, 0, None
                if self.raw_returns is not None:
                    self.raw_returns[index] = 0.0

    def take_summary(self) -> dict[str, object]:
        episodes, self.completed = self.completed, []
        returns = (
            {
                "scaled_environment_returns": _moments(
                    [row["return"] for row in episodes], "scaled pre-bootstrap episode returns"
                ),
                "raw_environment_returns": _moments(
                    [row["raw_return"] for row in episodes], "raw environment episode returns"
                ),
            }
            if self.reward_scale is not None
            else {"returns": _moments([row["return"] for row in episodes], "episode returns")}
        )
        return {
            "completed": len(episodes),
            **returns,
            "lengths": _moments([row["length"] for row in episodes], "episode lengths"),
            "falls": sum(bool(row["fallen"]) for row in episodes),
            "horizons": sum(bool(row["horizon_reached"]) for row in episodes),
            "terminal_metrics": {
                name: _moments([row[name] for row in episodes], f"terminal {name}")
                for name in ("control_step", *_METRIC_FLOATS)
            },
            "terminal_true": {
                name: sum(bool(row[name]) for row in episodes) for name in _METRIC_BOOLS
            },
        }

    def incomplete(self) -> list[dict[str, object]]:
        if self.returns is None:
            return []
        assert self.lengths is not None
        result = []
        for index, length in enumerate(self.lengths):
            if not length:
                continue
            episode = {"length": int(length), "last_metrics": self.last[index]}
            if self.reward_scale is None:
                episode["return"] = float(self.returns[index])
            else:
                assert self.raw_returns is not None
                episode["scaled_environment_return"] = float(self.returns[index])
                episode["raw_environment_return"] = float(self.raw_returns[index])
            result.append(episode)
        return result


def ppo_update_record(
    through: int, logger_values: Mapping[str, object], model_n_updates: object
) -> dict[str, object]:
    if type(through) is not int or through < 1:
        raise ValueError("update transition boundary must be positive")
    metrics: dict[str, float | None] = {}
    unavailable: dict[str, str] = {}
    for name, key in _LOGGER_FIELDS.items():
        if key not in logger_values:
            metrics[name], unavailable[name] = None, "missing"
            continue
        value = logger_values[key]
        array = np.asarray(value)
        if (
            name == "explained_variance"
            and array.shape == ()
            and np.issubdtype(array.dtype, np.floating)
            and not np.isfinite(array).item()
        ):
            metrics[name], unavailable[name] = None, "undefined_nonfinite"
            continue
        metrics[name] = _float(value, key)
    logged = logger_values.get("train/n_updates")
    logged = None if logged is None else _int(logged, "train/n_updates")
    accessible = None if model_n_updates is None else _int(model_n_updates, "model._n_updates")
    if logged is not None and accessible is not None and logged != accessible:
        raise ValueError("SB3 update counters disagree")
    return {
        "trained_through_transitions": through,
        "metrics": metrics,
        "metric_unavailable_reasons": unavailable,
        "sb3_n_updates": accessible if accessible is not None else logged,
        "optimizer_minibatch_steps": None,
    }


class TrainingTelemetry:
    """Write one record per rollout and one post-train final-update record."""

    def __init__(
        self,
        path: Path,
        *,
        reward_scale: float | None = None,
        fixed_normalizer: FixedNormalizerState | None = None,
    ) -> None:
        self.path = Path(path)
        fixed_sha256 = fixed_normalizer.sha256 if fixed_normalizer is not None else None
        self.telemetry_id, self.filename, self.schema_version = _telemetry_identity(
            reward_scale, fixed_sha256
        )
        if self.path.name != self.filename:
            raise ValueError("training telemetry filename differs from its version")
        self.handle = self.path.open("xb")
        self.reward_scale = reward_scale
        self.fixed_normalizer = fixed_normalizer
        self.episodes, self.last_boundary = EpisodeAccumulator(reward_scale), None
        self.records = self.rollouts = self.bytes_written = 0
        self.finished = False

    def __enter__(self) -> TrainingTelemetry:
        return self

    def __exit__(self, *_args: object) -> None:
        self.handle.close()

    def observe_step(
        self, rewards: object, dones: object, infos: Sequence[Mapping[str, object]]
    ) -> None:
        self.episodes.observe(rewards, dones, infos)

    def _write(self, row: Mapping[str, object]) -> None:
        encoded = (
            json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
        ).encode()
        if self.bytes_written + len(encoded) > MAX_TELEMETRY_BYTES:
            raise ValueError("training telemetry exceeds its byte bound")
        self.handle.write(encoded)
        self.handle.flush()
        self.records += 1
        self.bytes_written += len(encoded)

    def rollout_boundary(
        self,
        through: int,
        observations: object,
        logger_values: Mapping[str, object],
        model_n_updates: object,
    ) -> None:
        if through % ROLLOUT_STEPS or through != (self.last_boundary or 0) + ROLLOUT_STEPS:
            raise ValueError("training rollout telemetry boundary is not contiguous")
        previous = (
            None
            if self.last_boundary is None
            else ppo_update_record(self.last_boundary, logger_values, model_n_updates)
        )
        self.rollouts += 1
        row = {
            "schema_version": self.schema_version,
            "telemetry_id": self.telemetry_id,
            "event": "rollout_boundary",
            "rollout_index": self.rollouts,
            "collected_through_transitions": through,
            "previous_update": previous,
            "episodes": self.episodes.take_summary(),
            "observation_groups": observation_group_moments(observations),
        }
        if self.fixed_normalizer is not None:
            row["fixed_normalized_base_range"] = {
                "source": "current_rollout_buffer_raw_observations",
                "normalizer_state_sha256": self.fixed_normalizer.sha256,
                "maximum_absolute_value": normalized_base_max_abs(
                    observations, self.fixed_normalizer
                ),
            }
        self._write(row)
        self.last_boundary = through

    def final_update(self, logger_values: Mapping[str, object], model_n_updates: object) -> None:
        if self.finished or self.last_boundary is None:
            raise ValueError("training telemetry lacks a unique final rollout")
        self._write(
            {
                "schema_version": self.schema_version,
                "telemetry_id": self.telemetry_id,
                "event": "final_update",
                "collected_through_transitions": self.last_boundary,
                "update": ppo_update_record(self.last_boundary, logger_values, model_n_updates),
                "incomplete_episodes": self.episodes.incomplete(),
            }
        )
        self.finished = True

    def descriptor(self, expected_transitions: int) -> dict[str, object]:
        if not self.finished:
            raise ValueError("training telemetry lacks its final update")
        self.handle.flush()
        encoded = self.path.read_bytes()
        counts = validate_training_telemetry(
            encoded,
            expected_transitions,
            reward_scale=self.reward_scale,
            fixed_normalizer_sha256=(
                self.fixed_normalizer.sha256 if self.fixed_normalizer is not None else None
            ),
        )
        if counts["record_count"] != self.records:
            raise ValueError("training telemetry writer and validator disagree")
        descriptor = {
            "schema_version": self.schema_version,
            "telemetry_id": self.telemetry_id,
            "path": self.filename,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "size_bytes": len(encoded),
            **counts,
            "rollout_boundary_semantics": _ROLLOUT_SEMANTICS,
            "sb3_n_updates_semantics": _UPDATE_COUNT_SEMANTICS,
        }
        if self.fixed_normalizer is not None:
            descriptor.update(
                fixed_normalizer_state_sha256=self.fixed_normalizer.sha256,
                normalized_base_range_semantics=_NORMALIZED_RANGE_SEMANTICS,
            )
        return descriptor


def _finite_json(value: object) -> None:
    if type(value) is float and not math.isfinite(value):
        raise ValueError("training telemetry contains non-finite JSON")
    if type(value) is list:
        for item in value:
            _finite_json(item)
    elif type(value) is dict:
        for item in value.values():
            _finite_json(item)


def _valid_update(value: object, through: int) -> bool:
    if (
        type(value) is not dict
        or set(value)
        != {
            "trained_through_transitions",
            "metrics",
            "metric_unavailable_reasons",
            "sb3_n_updates",
            "optimizer_minibatch_steps",
        }
        or type(value["trained_through_transitions"]) is not int
        or value["trained_through_transitions"] != through
        or value["optimizer_minibatch_steps"] is not None
        or not (
            value["sb3_n_updates"] is None
            or (type(value["sb3_n_updates"]) is int and value["sb3_n_updates"] >= 0)
        )
        or type(value["metrics"]) is not dict
        or set(value["metrics"]) != set(_LOGGER_FIELDS)
        or type(value["metric_unavailable_reasons"]) is not dict
    ):
        return False
    reasons = value["metric_unavailable_reasons"]
    for name, item in value["metrics"].items():
        if item is None:
            if reasons.get(name) not in _UNAVAILABLE_REASONS:
                return False
            if reasons[name] == "undefined_nonfinite" and name != "explained_variance":
                return False
        elif (
            type(item) not in {int, float}
            or not math.isfinite(item)
            or name in reasons
        ):
            return False
    return set(reasons) == {name for name, item in value["metrics"].items() if item is None}


def _valid_nonnegative_count(value: object) -> bool:
    return type(value) is int and value >= 0


def _valid_moments(value: object, expected_count: int) -> bool:
    if expected_count == 0:
        return value is None
    fields = {"count", "minimum", "maximum", "mean", "standard_deviation"}
    if type(value) is not dict or set(value) != fields:
        return False
    numbers = [value[name] for name in fields - {"count"}]
    return (
        type(value["count"]) is int
        and value["count"] == expected_count
        and all(type(number) in {int, float} and math.isfinite(number) for number in numbers)
        and value["minimum"] <= value["mean"] <= value["maximum"]
        and value["standard_deviation"] >= 0
    )


def _valid_episode_counts(value: object, *, scaled: bool) -> bool:
    if type(value) is not dict:
        return False
    required = {
        "completed",
        "lengths",
        "falls",
        "horizons",
        "terminal_metrics",
        "terminal_true",
    }
    required.update(
        {"scaled_environment_returns", "raw_environment_returns"} if scaled else {"returns"}
    )
    if set(value) != required or not all(
        _valid_nonnegative_count(value[name]) for name in ("completed", "falls", "horizons")
    ):
        return False
    completed = value["completed"]
    if (
        value["falls"] > completed
        or value["horizons"] > completed
        or not all(
            _valid_moments(value[name], completed)
            for name in (
                ("scaled_environment_returns", "raw_environment_returns")
                if scaled
                else ("returns",)
            )
        )
        or not _valid_moments(value["lengths"], completed)
        or type(value["terminal_metrics"]) is not dict
        or set(value["terminal_metrics"]) != {"control_step", *_METRIC_FLOATS}
        or not all(
            _valid_moments(summary, completed)
            for summary in value["terminal_metrics"].values()
        )
    ):
        return False
    terminal_true = value["terminal_true"]
    return (
        type(terminal_true) is dict
        and set(terminal_true) == set(_METRIC_BOOLS)
        and all(
            _valid_nonnegative_count(count) and count <= completed
            for count in terminal_true.values()
        )
    )


def _valid_observation_counts(value: object) -> bool:
    if type(value) is not dict or set(value) != set(_GROUP_WIDTHS):
        return False
    moment_fields = {"count", "minimum", "maximum", "mean", "standard_deviation"}
    for name, width in _GROUP_WIDTHS.items():
        summary = value[name]
        if (
            type(summary) is not dict
            or set(summary) != {"observations", "width", *moment_fields}
            or type(summary["observations"]) is not int
            or summary["observations"] < 1
            or type(summary["width"]) is not int
            or summary["width"] != width
            or not _valid_moments(
                {field: summary[field] for field in moment_fields},
                summary["observations"] * width,
            )
        ):
            return False
    return True


def _valid_normalized_range(value: object, expected_sha256: str) -> bool:
    return (
        type(value) is dict
        and set(value)
        == {"source", "normalizer_state_sha256", "maximum_absolute_value"}
        and value["source"] == "current_rollout_buffer_raw_observations"
        and value["normalizer_state_sha256"] == expected_sha256
        and type(value["maximum_absolute_value"]) in {int, float}
        and math.isfinite(value["maximum_absolute_value"])
        and value["maximum_absolute_value"] >= 0
    )


def _valid_incomplete(value: object, *, scaled: bool) -> bool:
    if type(value) is not list:
        return False
    expected = (
        {"length", "last_metrics", "scaled_environment_return", "raw_environment_return"}
        if scaled
        else {"length", "last_metrics", "return"}
    )
    return all(
        type(item) is dict
        and set(item) == expected
        and type(item["length"]) is int
        and item["length"] > 0
        and all(
            type(item[name]) in {int, float} and math.isfinite(item[name])
            for name in expected - {"length", "last_metrics"}
        )
        and type(item["last_metrics"]) is dict
        for item in value
    )


def validate_training_telemetry(
    encoded: bytes,
    expected_transitions: int,
    *,
    reward_scale: float | None = None,
    fixed_normalizer_sha256: str | None = None,
) -> dict[str, object]:
    """Validate the bounded identity and previous-update transition alignment."""

    if (
        type(encoded) is not bytes
        or not 0 < len(encoded) <= MAX_TELEMETRY_BYTES
        or type(expected_transitions) is not int
        or expected_transitions < ROLLOUT_STEPS
        or expected_transitions % ROLLOUT_STEPS
    ):
        raise ValueError("training telemetry bytes or budget is invalid")
    telemetry_id, _filename, schema_version = _telemetry_identity(
        reward_scale, fixed_normalizer_sha256
    )
    scaled = reward_scale is not None
    lines, rollouts = encoded.splitlines(), expected_transitions // ROLLOUT_STEPS
    if len(lines) != rollouts + 1 or any(not line for line in lines):
        raise ValueError("training telemetry record count differs")
    rows = [decode_json_object(line, source="training telemetry row") for line in lines]
    for row in rows:
        _finite_json(row)
        if (
            row.get("schema_version") != schema_version
            or row.get("telemetry_id") != telemetry_id
        ):
            raise ValueError("training telemetry identity differs")
    rollout_keys = {
        "schema_version",
        "telemetry_id",
        "event",
        "rollout_index",
        "collected_through_transitions",
        "previous_update",
        "episodes",
        "observation_groups",
    }
    if fixed_normalizer_sha256 is not None:
        rollout_keys.add("fixed_normalized_base_range")
    for index, row in enumerate(rows[:-1], start=1):
        through = index * ROLLOUT_STEPS
        if (
            set(row) != rollout_keys
            or row["event"] != "rollout_boundary"
            or type(row["rollout_index"]) is not int
            or row["rollout_index"] != index
            or type(row["collected_through_transitions"]) is not int
            or row["collected_through_transitions"] != through
            or (index == 1 and row["previous_update"] is not None)
            or (index > 1 and not _valid_update(row["previous_update"], through - ROLLOUT_STEPS))
            or not _valid_episode_counts(row["episodes"], scaled=scaled)
            or not _valid_observation_counts(row["observation_groups"])
            or (
                fixed_normalizer_sha256 is not None
                and not _valid_normalized_range(
                    row["fixed_normalized_base_range"], fixed_normalizer_sha256
                )
            )
        ):
            raise ValueError("training rollout telemetry fields or sequence differ")
    final = rows[-1]
    if (
        set(final)
        != {
            "schema_version",
            "telemetry_id",
            "event",
            "collected_through_transitions",
            "update",
            "incomplete_episodes",
        }
        or final["event"] != "final_update"
        or type(final["collected_through_transitions"]) is not int
        or final["collected_through_transitions"] != expected_transitions
        or not _valid_update(final["update"], expected_transitions)
        or not _valid_incomplete(final["incomplete_episodes"], scaled=scaled)
    ):
        raise ValueError("training final-update telemetry differs")
    return {
        "record_count": len(rows),
        "rollout_boundary_count": rollouts,
        "final_update_recorded": True,
    }


def validate_training_telemetry_descriptor(
    value: object,
    encoded: bytes,
    expected_transitions: int,
    *,
    reward_scale: float | None = None,
    fixed_normalizer_sha256: str | None = None,
) -> dict[str, object]:
    """Bind an exact optional descriptor to its retained JSONL bytes."""

    descriptor_fields = set(_DESCRIPTOR_FIELDS)
    if fixed_normalizer_sha256 is not None:
        descriptor_fields.update(
            {"fixed_normalizer_state_sha256", "normalized_base_range_semantics"}
        )
    if type(value) is not dict or set(value) != descriptor_fields:
        raise ValueError("training telemetry descriptor fields differ")
    if any(
        type(value[name]) is not int or value[name] < 1
        for name in ("size_bytes", "record_count", "rollout_boundary_count")
    ):
        raise ValueError("training telemetry descriptor counts differ")
    telemetry_id, filename, schema_version = _telemetry_identity(
        reward_scale, fixed_normalizer_sha256
    )
    counts = validate_training_telemetry(
        encoded,
        expected_transitions,
        reward_scale=reward_scale,
        fixed_normalizer_sha256=fixed_normalizer_sha256,
    )
    expected = {
        "schema_version": schema_version,
        "telemetry_id": telemetry_id,
        "path": filename,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
        **counts,
        "rollout_boundary_semantics": _ROLLOUT_SEMANTICS,
        "sb3_n_updates_semantics": _UPDATE_COUNT_SEMANTICS,
    }
    if fixed_normalizer_sha256 is not None:
        expected.update(
            fixed_normalizer_state_sha256=fixed_normalizer_sha256,
            normalized_base_range_semantics=_NORMALIZED_RANGE_SEMANTICS,
        )
    if value != expected:
        raise ValueError("training telemetry descriptor differs from retained bytes")
    return expected


__all__ = [
    "FIXED_NORMALIZER_TELEMETRY_FILENAME",
    "FIXED_NORMALIZER_TELEMETRY_ID",
    "MAX_TELEMETRY_BYTES",
    "SCALED_TELEMETRY_FILENAME",
    "SCALED_TELEMETRY_ID",
    "TELEMETRY_FILENAME",
    "TELEMETRY_ID",
    "EpisodeAccumulator",
    "TrainingTelemetry",
    "observation_group_moments",
    "ppo_update_record",
    "telemetry_filename",
    "validate_training_telemetry",
    "validate_training_telemetry_descriptor",
]
