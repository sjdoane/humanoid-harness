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

TELEMETRY_ID = "gmt_g1_ppo_training_telemetry/v1"
TELEMETRY_FILENAME = "training_telemetry_v1.jsonl"
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
_UPDATE_COUNT_SEMANTICS = "completed_ppo_epochs_not_optimizer_minibatches"


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

    def __init__(self) -> None:
        self.returns: np.ndarray | None = None
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
            self.lengths = np.zeros(rewards.size, dtype=np.int64)
            self.last = [None] * rewards.size
        if rewards.shape != self.returns.shape:
            raise ValueError("training environment count changed")
        assert self.lengths is not None
        self.returns += rewards
        self.lengths += 1
        for index, (done, info) in enumerate(zip(dones, infos, strict=True)):
            terminal = _terminal(info)
            self.last[index] = terminal
            if bool(done):
                self.completed.append(
                    {
                        "return": float(self.returns[index]),
                        "length": int(self.lengths[index]),
                        **terminal,
                    }
                )
                self.returns[index], self.lengths[index], self.last[index] = 0.0, 0, None

    def take_summary(self) -> dict[str, object]:
        episodes, self.completed = self.completed, []
        return {
            "completed": len(episodes),
            "returns": _moments([row["return"] for row in episodes], "episode returns"),
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
        return [
            {"return": float(self.returns[i]), "length": int(length), "last_metrics": self.last[i]}
            for i, length in enumerate(self.lengths)
            if length
        ]


def ppo_update_record(
    through: int, logger_values: Mapping[str, object], model_n_updates: object
) -> dict[str, object]:
    if type(through) is not int or through < 1:
        raise ValueError("update transition boundary must be positive")
    metrics = {
        name: None if key not in logger_values else _float(logger_values[key], key)
        for name, key in _LOGGER_FIELDS.items()
    }
    logged = logger_values.get("train/n_updates")
    logged = None if logged is None else _int(logged, "train/n_updates")
    accessible = None if model_n_updates is None else _int(model_n_updates, "model._n_updates")
    if logged is not None and accessible is not None and logged != accessible:
        raise ValueError("SB3 update counters disagree")
    return {
        "trained_through_transitions": through,
        "metrics": metrics,
        "sb3_n_updates": accessible if accessible is not None else logged,
        "optimizer_minibatch_steps": None,
    }


class TrainingTelemetry:
    """Write one record per rollout and one post-train final-update record."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.handle = self.path.open("xb")
        self.episodes, self.last_boundary = EpisodeAccumulator(), None
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
        self._write(
            {
                "schema_version": 1,
                "telemetry_id": TELEMETRY_ID,
                "event": "rollout_boundary",
                "rollout_index": self.rollouts,
                "collected_through_transitions": through,
                "previous_update": previous,
                "episodes": self.episodes.take_summary(),
                "observation_groups": observation_group_moments(observations),
            }
        )
        self.last_boundary = through

    def final_update(self, logger_values: Mapping[str, object], model_n_updates: object) -> None:
        if self.finished or self.last_boundary is None:
            raise ValueError("training telemetry lacks a unique final rollout")
        self._write(
            {
                "schema_version": 1,
                "telemetry_id": TELEMETRY_ID,
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
        counts = validate_training_telemetry(encoded, expected_transitions)
        if counts["record_count"] != self.records:
            raise ValueError("training telemetry writer and validator disagree")
        return {
            "schema_version": 1,
            "telemetry_id": TELEMETRY_ID,
            "path": TELEMETRY_FILENAME,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "size_bytes": len(encoded),
            **counts,
            "rollout_boundary_semantics": _ROLLOUT_SEMANTICS,
            "sb3_n_updates_semantics": _UPDATE_COUNT_SEMANTICS,
        }


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
    return (
        type(value) is dict
        and set(value)
        == {"trained_through_transitions", "metrics", "sb3_n_updates", "optimizer_minibatch_steps"}
        and value["trained_through_transitions"] == through
        and value["optimizer_minibatch_steps"] is None
        and (value["sb3_n_updates"] is None or type(value["sb3_n_updates"]) is int)
        and type(value["metrics"]) is dict
        and set(value["metrics"]) == set(_LOGGER_FIELDS)
        and all(item is None or type(item) in {int, float} for item in value["metrics"].values())
    )


def validate_training_telemetry(encoded: bytes, expected_transitions: int) -> dict[str, object]:
    """Validate the bounded identity and previous-update transition alignment."""

    if (
        type(encoded) is not bytes
        or not 0 < len(encoded) <= MAX_TELEMETRY_BYTES
        or type(expected_transitions) is not int
        or expected_transitions < ROLLOUT_STEPS
        or expected_transitions % ROLLOUT_STEPS
    ):
        raise ValueError("training telemetry bytes or budget is invalid")
    lines, rollouts = encoded.splitlines(), expected_transitions // ROLLOUT_STEPS
    if len(lines) != rollouts + 1 or any(not line for line in lines):
        raise ValueError("training telemetry record count differs")
    rows = [decode_json_object(line, source="training telemetry row") for line in lines]
    for row in rows:
        _finite_json(row)
        if row.get("schema_version") != 1 or row.get("telemetry_id") != TELEMETRY_ID:
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
    for index, row in enumerate(rows[:-1], start=1):
        through = index * ROLLOUT_STEPS
        if (
            set(row) != rollout_keys
            or row["event"] != "rollout_boundary"
            or row["rollout_index"] != index
            or row["collected_through_transitions"] != through
            or (index == 1 and row["previous_update"] is not None)
            or (index > 1 and not _valid_update(row["previous_update"], through - ROLLOUT_STEPS))
            or type(row["episodes"]) is not dict
            or type(row["observation_groups"]) is not dict
            or set(row["observation_groups"]) != set(_GROUP_WIDTHS)
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
        or final["collected_through_transitions"] != expected_transitions
        or not _valid_update(final["update"], expected_transitions)
        or type(final["incomplete_episodes"]) is not list
    ):
        raise ValueError("training final-update telemetry differs")
    return {
        "record_count": len(rows),
        "rollout_boundary_count": rollouts,
        "final_update_recorded": True,
    }


def validate_training_telemetry_descriptor(
    value: object, encoded: bytes, expected_transitions: int
) -> dict[str, object]:
    """Bind an exact optional descriptor to its retained JSONL bytes."""

    if type(value) is not dict or set(value) != _DESCRIPTOR_FIELDS:
        raise ValueError("training telemetry descriptor fields differ")
    counts = validate_training_telemetry(encoded, expected_transitions)
    expected = {
        "schema_version": 1,
        "telemetry_id": TELEMETRY_ID,
        "path": TELEMETRY_FILENAME,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "size_bytes": len(encoded),
        **counts,
        "rollout_boundary_semantics": _ROLLOUT_SEMANTICS,
        "sb3_n_updates_semantics": _UPDATE_COUNT_SEMANTICS,
    }
    if value != expected:
        raise ValueError("training telemetry descriptor differs from retained bytes")
    return expected


__all__ = [
    "MAX_TELEMETRY_BYTES",
    "TELEMETRY_FILENAME",
    "TELEMETRY_ID",
    "EpisodeAccumulator",
    "TrainingTelemetry",
    "observation_group_moments",
    "ppo_update_record",
    "validate_training_telemetry",
    "validate_training_telemetry_descriptor",
]
