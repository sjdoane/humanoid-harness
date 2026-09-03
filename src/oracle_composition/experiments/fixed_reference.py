"""Fail-closed contracts for the frozen Humanoid reference-tracker baseline.

These types define what may be compared.  They do not claim that a trained
checkpoint is competent, reference-conditioned, or suitable for oracle work.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from string import hexdigits
from typing import Any

import numpy as np
from numpy.typing import ArrayLike


class ExperimentContractError(ValueError):
    """Raised when an experiment cannot preserve its declared comparison."""


@dataclass(frozen=True, slots=True)
class BoundedJsonArtifact:
    """One parsed JSON object and the exact bytes that produced it."""

    value: dict[str, Any]
    encoded_bytes: bytes
    sha256: str


MAX_STUDY_DESIGN_BYTES = 256 * 1024
MATCHED_STATE_CAUSAL_INPUT_PROTOCOL_ID = "matched_state_numeric_reference_intervention/v1"
CAUSAL_INPUT_CONDITION_IDS = frozenset(
    {
        "C_exact",
        "C_zero_input",
        "C_shuffle_input",
        "C_shift_input",
    }
)
CANONICAL_EXECUTION_CALLABLE_AUTHORITY = "canonical_production_callables/v1"
NONAUTHORITATIVE_EXECUTION_CALLABLE_AUTHORITY = "noncanonical_callables_test_only/v1"
_HUMANOID_V5_ACTUATOR_GEAR_BY_JOINT = (
    100.0,
    100.0,
    100.0,
    100.0,
    100.0,
    300.0,
    200.0,
    100.0,
    100.0,
    300.0,
    200.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
)
_HUMANOID_V5_TORQUE_CAPACITY_N_M = (
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    120.0,
    80.0,
    40.0,
    40.0,
    120.0,
    80.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
)


class EvidencePurpose(StrEnum):
    """Non-interchangeable meanings of a run receipt."""

    RESOURCE_CALIBRATION = "resource_calibration"
    BEHAVIORAL_EVALUATION = "behavioral_evaluation"


class ClaimCeiling(StrEnum):
    """Highest claim a design can earn even if every declared check passes."""

    STATIC_TRACKING_FEASIBILITY = "static_tracking_feasibility"
    CAUSAL_REFERENCE_USE = "causal_reference_use"


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExperimentContractError(f"value is not canonical JSON: {exc}") from exc


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ExperimentContractError(f"JSON contains duplicate key {key!r}")
        value[key] = child
    return value


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ExperimentContractError(f"JSON contains non-finite number {value!r}")


def _assert_finite_json(value: object, *, field: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ExperimentContractError(f"{field} contains a non-finite number")
    if isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_json(child, field=f"{field}[{index}]")
    elif isinstance(value, dict):
        for key, child in value.items():
            _assert_finite_json(child, field=f"{field}.{key}")


def read_bounded_json_artifact(
    path: Path,
    *,
    maximum_bytes: int,
    artifact: str,
) -> BoundedJsonArtifact:
    """Read one regular JSON file once and bind its parsed value to those bytes."""

    if not isinstance(maximum_bytes, int) or isinstance(maximum_bytes, bool) or maximum_bytes < 1:
        raise ExperimentContractError("maximum_bytes must be a positive integer")
    resolved = Path(path)
    descriptor: int | None = None
    try:
        if stat.S_ISLNK(resolved.lstat().st_mode):
            raise ExperimentContractError(f"{artifact} must not be a symlink")
        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(resolved, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ExperimentContractError(f"{artifact} must be a regular file")
        if before.st_size > maximum_bytes:
            raise ExperimentContractError(f"{artifact} exceeds the bounded size limit")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        after = os.fstat(descriptor)
        if len(encoded) > maximum_bytes:
            raise ExperimentContractError(f"{artifact} exceeds the bounded size limit")
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if identity_before != identity_after or len(encoded) != before.st_size:
            raise ExperimentContractError(f"{artifact} changed while it was read")
        text = encoded.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_pairs,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except ExperimentContractError:
        raise
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise ExperimentContractError(f"cannot read {artifact}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if not isinstance(value, dict):
        raise ExperimentContractError(f"{artifact} root must be a JSON object")
    _assert_finite_json(value)
    return BoundedJsonArtifact(
        value=value,
        encoded_bytes=encoded,
        sha256=hashlib.sha256(encoded).hexdigest(),
    )


def read_bounded_json_object(
    path: Path,
    *,
    maximum_bytes: int,
    artifact: str,
) -> dict[str, Any]:
    """Read one strict bounded JSON object."""

    return read_bounded_json_artifact(
        path,
        maximum_bytes=maximum_bytes,
        artifact=artifact,
    ).value


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in hexdigits for character in value)
    )


def _require_sha256(value: object, *, field: str) -> str:
    if not _is_sha256(value):
        raise ExperimentContractError(f"{field} must be a lowercase SHA-256")
    return str(value)


def _require_exact_keys(
    value: dict[str, Any],
    *,
    expected: set[str],
    field: str,
) -> None:
    observed = set(value)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise ExperimentContractError(
            f"{field} keys mismatch: missing={missing!r}, extra={extra!r}"
        )


def _positive_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ExperimentContractError(f"{field} must be a positive integer")
    return value


def _probability(value: object, *, field: str, include_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentContractError(f"{field} must be numeric")
    resolved = float(value)
    lower_ok = resolved >= 0.0 if include_zero else resolved > 0.0
    if not math.isfinite(resolved) or not lower_ok or resolved > 1.0:
        boundary = "[0, 1]" if include_zero else "(0, 1]"
        raise ExperimentContractError(f"{field} must be in {boundary}")
    return resolved


@dataclass(frozen=True, slots=True)
class PPOTrainingDesign:
    """Predetermined Stable-Baselines3 PPO budget and hyperparameters."""

    total_timesteps_per_seed: int
    n_envs: int
    n_steps: int
    batch_size: int
    n_epochs: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_range: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    policy_hidden_layers: tuple[int, ...]
    device: str

    def __post_init__(self) -> None:
        integer_fields = (
            "total_timesteps_per_seed",
            "n_envs",
            "n_steps",
            "batch_size",
            "n_epochs",
        )
        for field in integer_fields:
            _positive_int(getattr(self, field), field=f"ppo.{field}")
        if isinstance(self.policy_hidden_layers, (str, bytes)):
            raise ExperimentContractError("ppo.policy_hidden_layers must be an ordered sequence")
        try:
            layers = tuple(self.policy_hidden_layers)
        except TypeError as exc:
            raise ExperimentContractError(
                "ppo.policy_hidden_layers must be an ordered sequence"
            ) from exc
        object.__setattr__(self, "policy_hidden_layers", layers)
        if not layers or any(
            not isinstance(width, int) or isinstance(width, bool) or width < 1 for width in layers
        ):
            raise ExperimentContractError("ppo.policy_hidden_layers must be positive integers")

        rollout_size = self.n_envs * self.n_steps
        if rollout_size % self.batch_size:
            raise ExperimentContractError("ppo.batch_size must divide n_envs * n_steps")
        if self.total_timesteps_per_seed % rollout_size:
            raise ExperimentContractError(
                "ppo.total_timesteps_per_seed must be an exact multiple of n_envs * n_steps"
            )
        _probability(self.gamma, field="ppo.gamma", include_zero=False)
        _probability(self.gae_lambda, field="ppo.gae_lambda")
        _probability(self.clip_range, field="ppo.clip_range", include_zero=False)
        for field in ("learning_rate", "max_grad_norm"):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value <= 0.0
            ):
                raise ExperimentContractError(f"ppo.{field} must be finite and positive")
        for field in ("ent_coef", "vf_coef"):
            value = getattr(self, field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ExperimentContractError(f"ppo.{field} must be finite and non-negative")
        if self.device not in {"cpu", "cuda", "mps"}:
            raise ExperimentContractError("ppo.device must be one of: cpu, cuda, mps")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PPOTrainingDesign:
        expected = {
            "total_timesteps_per_seed",
            "n_envs",
            "n_steps",
            "batch_size",
            "n_epochs",
            "learning_rate",
            "gamma",
            "gae_lambda",
            "clip_range",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
            "policy_hidden_layers",
            "device",
        }
        _require_exact_keys(value, expected=expected, field="ppo")
        layers = value["policy_hidden_layers"]
        if not isinstance(layers, list):
            raise ExperimentContractError("ppo.policy_hidden_layers must be a JSON array")
        return cls(
            total_timesteps_per_seed=value["total_timesteps_per_seed"],
            n_envs=value["n_envs"],
            n_steps=value["n_steps"],
            batch_size=value["batch_size"],
            n_epochs=value["n_epochs"],
            learning_rate=value["learning_rate"],
            gamma=value["gamma"],
            gae_lambda=value["gae_lambda"],
            clip_range=value["clip_range"],
            ent_coef=value["ent_coef"],
            vf_coef=value["vf_coef"],
            max_grad_norm=value["max_grad_norm"],
            policy_hidden_layers=tuple(layers),
            device=value["device"],
        )


@dataclass(frozen=True, slots=True)
class FixedReferenceStudyDesign:
    """Proposed or locked pre-data design; never an execution manifest."""

    schema_version: int
    experiment_id: str
    design_version: str
    design_status: str
    evidence_purpose: EvidencePurpose
    claim_ceiling: ClaimCeiling
    study_criteria_semantic_sha256: str
    study_criteria_file_sha256: str
    train_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    max_episode_steps: int
    horizon_steps: int
    static_reference_frames: int
    checkpoint_rule: str
    deterministic_evaluation: bool
    causal_use_evaluation_enabled: bool
    causal_use_block_reason: str
    ppo: PPOTrainingDesign

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != 1
        ):
            raise ExperimentContractError("schema_version must be 1")
        if self.experiment_id != "001_humanoid_fixed_reference":
            raise ExperimentContractError("unexpected experiment_id")
        if not isinstance(self.design_version, str) or not self.design_version.strip():
            raise ExperimentContractError("design_version must be a non-empty string")
        if self.design_status not in {"proposed_pre_calibration", "locked_pre_behavioral"}:
            raise ExperimentContractError(
                "design_status must be proposed_pre_calibration or locked_pre_behavioral"
            )
        if self.evidence_purpose is not EvidencePurpose.BEHAVIORAL_EVALUATION:
            raise ExperimentContractError("study design must be behavioral_evaluation")
        _require_sha256(
            self.study_criteria_semantic_sha256,
            field="study_criteria_semantic_sha256",
        )
        _require_sha256(
            self.study_criteria_file_sha256,
            field="study_criteria_file_sha256",
        )
        for field in ("train_seeds", "evaluation_seeds"):
            if isinstance(getattr(self, field), (str, bytes)):
                raise ExperimentContractError(f"{field} must be an ordered sequence")
        try:
            train_seeds = tuple(self.train_seeds)
            evaluation_seeds = tuple(self.evaluation_seeds)
        except TypeError as exc:
            raise ExperimentContractError("seed schedules must be ordered sequences") from exc
        object.__setattr__(self, "train_seeds", train_seeds)
        object.__setattr__(self, "evaluation_seeds", evaluation_seeds)
        for field, seeds in (
            ("train_seeds", train_seeds),
            ("evaluation_seeds", evaluation_seeds),
        ):
            if len(seeds) < 2:
                raise ExperimentContractError(f"{field} must contain at least two seeds")
            if any(
                not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds
            ):
                raise ExperimentContractError(f"{field} must contain non-negative integers")
            if len(set(seeds)) != len(seeds):
                raise ExperimentContractError(f"{field} must not contain duplicates")
        if set(train_seeds) & set(evaluation_seeds):
            raise ExperimentContractError("training and evaluation seeds must be disjoint")
        _positive_int(self.max_episode_steps, field="max_episode_steps")
        _positive_int(self.horizon_steps, field="horizon_steps")
        _positive_int(self.static_reference_frames, field="static_reference_frames")
        if self.static_reference_frames < self.max_episode_steps:
            raise ExperimentContractError(
                "static_reference_frames must cover the complete evaluation episode"
            )
        if self.checkpoint_rule != "final_timestep_only":
            raise ExperimentContractError("checkpoint_rule must be final_timestep_only")
        if not isinstance(self.deterministic_evaluation, bool) or not self.deterministic_evaluation:
            raise ExperimentContractError("evaluation actions must be deterministic")
        if self.claim_ceiling is not ClaimCeiling.STATIC_TRACKING_FEASIBILITY:
            raise ExperimentContractError(
                "fixed static-reference study ceiling must be static_tracking_feasibility"
            )
        if not isinstance(self.causal_use_evaluation_enabled, bool):
            raise ExperimentContractError("causal_use_evaluation_enabled must be boolean")
        if self.causal_use_evaluation_enabled:
            raise ExperimentContractError(
                "a static reference design cannot enable causal-use evaluation"
            )
        if (
            not isinstance(self.causal_use_block_reason, str)
            or not self.causal_use_block_reason.strip()
        ):
            raise ExperimentContractError("static design must record its causal-use blocker")

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_dict())

    def assert_locked_for_behavior(self) -> None:
        """Block training/evaluation until calibration choices are reviewed and locked."""

        if self.design_status != "locked_pre_behavioral":
            raise ExperimentContractError(
                "study design is proposed_pre_calibration, not locked for behavioral work"
            )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["evidence_purpose"] = self.evidence_purpose.value
        payload["claim_ceiling"] = self.claim_ceiling.value
        payload["train_seeds"] = list(self.train_seeds)
        payload["evaluation_seeds"] = list(self.evaluation_seeds)
        payload["ppo"]["policy_hidden_layers"] = list(self.ppo.policy_hidden_layers)
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FixedReferenceStudyDesign:
        expected = {
            "schema_version",
            "experiment_id",
            "design_version",
            "design_status",
            "evidence_purpose",
            "claim_ceiling",
            "study_criteria_semantic_sha256",
            "study_criteria_file_sha256",
            "train_seeds",
            "evaluation_seeds",
            "max_episode_steps",
            "horizon_steps",
            "static_reference_frames",
            "checkpoint_rule",
            "deterministic_evaluation",
            "causal_use_evaluation_enabled",
            "causal_use_block_reason",
            "ppo",
        }
        _require_exact_keys(value, expected=expected, field="study design")
        for field in ("train_seeds", "evaluation_seeds"):
            if not isinstance(value[field], list):
                raise ExperimentContractError(f"{field} must be a JSON array")
        if not isinstance(value["ppo"], dict):
            raise ExperimentContractError("ppo must be a JSON object")
        try:
            evidence_purpose = EvidencePurpose(value["evidence_purpose"])
            claim_ceiling = ClaimCeiling(value["claim_ceiling"])
        except ValueError as exc:
            raise ExperimentContractError(str(exc)) from exc
        return cls(
            schema_version=value["schema_version"],
            experiment_id=value["experiment_id"],
            design_version=value["design_version"],
            design_status=value["design_status"],
            evidence_purpose=evidence_purpose,
            claim_ceiling=claim_ceiling,
            study_criteria_semantic_sha256=value["study_criteria_semantic_sha256"],
            study_criteria_file_sha256=value["study_criteria_file_sha256"],
            train_seeds=tuple(value["train_seeds"]),
            evaluation_seeds=tuple(value["evaluation_seeds"]),
            max_episode_steps=value["max_episode_steps"],
            horizon_steps=value["horizon_steps"],
            static_reference_frames=value["static_reference_frames"],
            checkpoint_rule=value["checkpoint_rule"],
            deterministic_evaluation=value["deterministic_evaluation"],
            causal_use_evaluation_enabled=value["causal_use_evaluation_enabled"],
            causal_use_block_reason=value["causal_use_block_reason"],
            ppo=PPOTrainingDesign.from_dict(value["ppo"]),
        )


def load_study_design(path: Path) -> FixedReferenceStudyDesign:
    """Load an exact JSON study design, rejecting unknown or missing fields."""

    payload = read_bounded_json_object(
        path,
        maximum_bytes=MAX_STUDY_DESIGN_BYTES,
        artifact="study design",
    )
    return FixedReferenceStudyDesign.from_dict(payload)


@dataclass(frozen=True, slots=True)
class RuntimeFingerprint:
    """Exact runtime and authored artifacts consumed by one execution."""

    env_id: str
    gymnasium_version: str
    mujoco_version: str
    numpy_version: str
    stable_baselines3_version: str
    torch_version: str
    python_version: str
    platform_system: str
    platform_machine: str
    model_sha256: str
    dependency_lock_sha256: str
    observation_space_sha256: str
    action_space_sha256: str
    physical_action_space_sha256: str
    observation_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    qpos_shape: tuple[int, ...]
    qvel_shape: tuple[int, ...]
    actuator_gear_by_joint: tuple[float, ...]
    generalized_actuator_torque_capacity_n_m: tuple[float, ...]
    environment_max_episode_steps: int
    control_period_seconds: float
    reference_content_sha256: str
    reference_schema_sha256: str
    tracking_reward_sha256: str
    task_reward_sha256: str
    environment_source_sha256: str
    reference_abi_source_sha256: str
    tracking_reward_source_sha256: str
    wrapper_source_sha256: str
    experiment_contract_source_sha256: str
    evaluator_source_sha256: str
    runner_source_sha256: str
    execution_source_sha256: str
    study_summary_source_sha256: str
    policy_source_sha256: str
    source_tree_sha256: str
    policy_id: str
    action_transform_id: str
    observation_normalizer_id: str
    reward_normalizer_id: str

    def __post_init__(self) -> None:
        for field in (
            "model_sha256",
            "dependency_lock_sha256",
            "observation_space_sha256",
            "action_space_sha256",
            "physical_action_space_sha256",
            "reference_content_sha256",
            "reference_schema_sha256",
            "tracking_reward_sha256",
            "task_reward_sha256",
            "environment_source_sha256",
            "reference_abi_source_sha256",
            "tracking_reward_source_sha256",
            "wrapper_source_sha256",
            "experiment_contract_source_sha256",
            "evaluator_source_sha256",
            "runner_source_sha256",
            "execution_source_sha256",
            "study_summary_source_sha256",
            "policy_source_sha256",
            "source_tree_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        for field in ("observation_shape", "action_shape", "qpos_shape", "qvel_shape"):
            value = getattr(self, field)
            if isinstance(value, (str, bytes)):
                raise ExperimentContractError(f"{field} must be an ordered sequence")
            try:
                shape = tuple(value)
            except TypeError as exc:
                raise ExperimentContractError(f"{field} must be an ordered sequence") from exc
            object.__setattr__(self, field, shape)
            if not shape or any(
                not isinstance(width, int) or isinstance(width, bool) or width < 1
                for width in shape
            ):
                raise ExperimentContractError(f"{field} must contain positive integers")
        for field in (
            "actuator_gear_by_joint",
            "generalized_actuator_torque_capacity_n_m",
        ):
            value = getattr(self, field)
            if isinstance(value, (str, bytes)):
                raise ExperimentContractError(f"{field} must be an ordered numeric sequence")
            try:
                vector = tuple(value)
            except TypeError as exc:
                raise ExperimentContractError(
                    f"{field} must be an ordered numeric sequence"
                ) from exc
            if len(vector) != self.action_shape[0] or any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(float(item))
                or (field == "generalized_actuator_torque_capacity_n_m" and item <= 0.0)
                or (field == "actuator_gear_by_joint" and item == 0.0)
                for item in vector
            ):
                raise ExperimentContractError(
                    f"{field} must contain one finite valid value per actuator"
                )
            object.__setattr__(self, field, tuple(float(item) for item in vector))
        if self.actuator_gear_by_joint != _HUMANOID_V5_ACTUATOR_GEAR_BY_JOINT:
            raise ExperimentContractError(
                "actuator_gear_by_joint does not match the pinned Humanoid-v5 actuator ABI"
            )
        if self.generalized_actuator_torque_capacity_n_m != _HUMANOID_V5_TORQUE_CAPACITY_N_M:
            raise ExperimentContractError(
                "generalized_actuator_torque_capacity_n_m does not match the pinned "
                "Humanoid-v5 actuator ABI"
            )
        _positive_int(
            self.environment_max_episode_steps,
            field="environment_max_episode_steps",
        )
        if (
            not isinstance(self.control_period_seconds, (int, float))
            or isinstance(self.control_period_seconds, bool)
            or not math.isfinite(float(self.control_period_seconds))
            or self.control_period_seconds <= 0.0
        ):
            raise ExperimentContractError("control_period_seconds must be finite and positive")
        for field in (
            "env_id",
            "gymnasium_version",
            "mujoco_version",
            "numpy_version",
            "stable_baselines3_version",
            "torch_version",
            "python_version",
            "platform_system",
            "platform_machine",
            "policy_id",
            "action_transform_id",
            "observation_normalizer_id",
            "reward_normalizer_id",
        ):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise ExperimentContractError(f"{field} must be a non-empty string")

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for field in (
            "observation_shape",
            "action_shape",
            "qpos_shape",
            "qvel_shape",
            "actuator_gear_by_joint",
            "generalized_actuator_torque_capacity_n_m",
        ):
            payload[field] = list(payload[field])
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RuntimeFingerprint:
        expected = {
            "env_id",
            "gymnasium_version",
            "mujoco_version",
            "numpy_version",
            "stable_baselines3_version",
            "torch_version",
            "python_version",
            "platform_system",
            "platform_machine",
            "model_sha256",
            "dependency_lock_sha256",
            "observation_space_sha256",
            "action_space_sha256",
            "physical_action_space_sha256",
            "observation_shape",
            "action_shape",
            "qpos_shape",
            "qvel_shape",
            "actuator_gear_by_joint",
            "generalized_actuator_torque_capacity_n_m",
            "environment_max_episode_steps",
            "control_period_seconds",
            "reference_content_sha256",
            "reference_schema_sha256",
            "tracking_reward_sha256",
            "task_reward_sha256",
            "environment_source_sha256",
            "reference_abi_source_sha256",
            "tracking_reward_source_sha256",
            "wrapper_source_sha256",
            "experiment_contract_source_sha256",
            "evaluator_source_sha256",
            "runner_source_sha256",
            "execution_source_sha256",
            "study_summary_source_sha256",
            "policy_source_sha256",
            "source_tree_sha256",
            "policy_id",
            "action_transform_id",
            "observation_normalizer_id",
            "reward_normalizer_id",
        }
        _require_exact_keys(value, expected=expected, field="runtime fingerprint")
        for field in (
            "observation_shape",
            "action_shape",
            "qpos_shape",
            "qvel_shape",
            "actuator_gear_by_joint",
            "generalized_actuator_torque_capacity_n_m",
        ):
            if not isinstance(value[field], list):
                raise ExperimentContractError(f"{field} must be a JSON array")
        return cls(
            env_id=value["env_id"],
            gymnasium_version=value["gymnasium_version"],
            mujoco_version=value["mujoco_version"],
            numpy_version=value["numpy_version"],
            stable_baselines3_version=value["stable_baselines3_version"],
            torch_version=value["torch_version"],
            python_version=value["python_version"],
            platform_system=value["platform_system"],
            platform_machine=value["platform_machine"],
            model_sha256=value["model_sha256"],
            dependency_lock_sha256=value["dependency_lock_sha256"],
            observation_space_sha256=value["observation_space_sha256"],
            action_space_sha256=value["action_space_sha256"],
            physical_action_space_sha256=value["physical_action_space_sha256"],
            observation_shape=tuple(value["observation_shape"]),
            action_shape=tuple(value["action_shape"]),
            qpos_shape=tuple(value["qpos_shape"]),
            qvel_shape=tuple(value["qvel_shape"]),
            actuator_gear_by_joint=tuple(value["actuator_gear_by_joint"]),
            generalized_actuator_torque_capacity_n_m=tuple(
                value["generalized_actuator_torque_capacity_n_m"]
            ),
            environment_max_episode_steps=value["environment_max_episode_steps"],
            control_period_seconds=value["control_period_seconds"],
            reference_content_sha256=value["reference_content_sha256"],
            reference_schema_sha256=value["reference_schema_sha256"],
            tracking_reward_sha256=value["tracking_reward_sha256"],
            task_reward_sha256=value["task_reward_sha256"],
            environment_source_sha256=value["environment_source_sha256"],
            reference_abi_source_sha256=value["reference_abi_source_sha256"],
            tracking_reward_source_sha256=value["tracking_reward_source_sha256"],
            wrapper_source_sha256=value["wrapper_source_sha256"],
            experiment_contract_source_sha256=value["experiment_contract_source_sha256"],
            evaluator_source_sha256=value["evaluator_source_sha256"],
            runner_source_sha256=value["runner_source_sha256"],
            execution_source_sha256=value["execution_source_sha256"],
            study_summary_source_sha256=value["study_summary_source_sha256"],
            policy_source_sha256=value["policy_source_sha256"],
            source_tree_sha256=value["source_tree_sha256"],
            policy_id=value["policy_id"],
            action_transform_id=value["action_transform_id"],
            observation_normalizer_id=value["observation_normalizer_id"],
            reward_normalizer_id=value["reward_normalizer_id"],
        )


@dataclass(frozen=True, slots=True)
class FrozenExecutionManifest:
    """Reviewed runtime lock; training cannot infer or repair any field."""

    schema_version: int
    status: str
    execution_callable_authority: str
    design_sha256: str
    calibrated_design_sha256: str
    calibrated_design_file_sha256: str
    calibration_receipt_sha256: str
    calibration_receipt_file_sha256: str
    study_criteria_semantic_sha256: str
    study_criteria_file_sha256: str
    runtime: RuntimeFingerprint
    candidate_file_sha256: str
    reviewer_id: str
    review_note: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != 1
        ):
            raise ExperimentContractError("execution manifest schema_version must be 1")
        allowed_statuses = {
            CANONICAL_EXECUTION_CALLABLE_AUTHORITY: "frozen",
            NONAUTHORITATIVE_EXECUTION_CALLABLE_AUTHORITY: "test_only_non_authoritative",
        }
        if allowed_statuses.get(self.execution_callable_authority) != self.status:
            raise ExperimentContractError(
                "execution manifest status/callable authority combination is invalid"
            )
        _require_sha256(self.design_sha256, field="design_sha256")
        _require_sha256(
            self.calibrated_design_sha256,
            field="calibrated_design_sha256",
        )
        _require_sha256(
            self.calibrated_design_file_sha256,
            field="calibrated_design_file_sha256",
        )
        _require_sha256(
            self.calibration_receipt_sha256,
            field="calibration_receipt_sha256",
        )
        _require_sha256(
            self.calibration_receipt_file_sha256,
            field="calibration_receipt_file_sha256",
        )
        if self.calibrated_design_sha256 == self.design_sha256:
            raise ExperimentContractError(
                "calibrated proposed design and locked design must have distinct identities"
            )
        _require_sha256(
            self.study_criteria_semantic_sha256,
            field="study_criteria_semantic_sha256",
        )
        _require_sha256(
            self.study_criteria_file_sha256,
            field="study_criteria_file_sha256",
        )
        _require_sha256(self.candidate_file_sha256, field="candidate_file_sha256")
        if not isinstance(self.runtime, RuntimeFingerprint):
            raise ExperimentContractError("execution manifest runtime must be a fingerprint")
        for field in ("reviewer_id", "review_note"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip():
                raise ExperimentContractError(f"execution manifest {field} must be non-empty")

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "execution_callable_authority": self.execution_callable_authority,
            "design_sha256": self.design_sha256,
            "calibrated_design_sha256": self.calibrated_design_sha256,
            "calibrated_design_file_sha256": self.calibrated_design_file_sha256,
            "calibration_receipt_sha256": self.calibration_receipt_sha256,
            "calibration_receipt_file_sha256": self.calibration_receipt_file_sha256,
            "study_criteria_semantic_sha256": (self.study_criteria_semantic_sha256),
            "study_criteria_file_sha256": self.study_criteria_file_sha256,
            "runtime": self.runtime.to_dict(),
            "candidate_file_sha256": self.candidate_file_sha256,
            "reviewer_id": self.reviewer_id,
            "review_note": self.review_note,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FrozenExecutionManifest:
        _require_exact_keys(
            value,
            expected={
                "schema_version",
                "status",
                "execution_callable_authority",
                "design_sha256",
                "calibrated_design_sha256",
                "calibrated_design_file_sha256",
                "calibration_receipt_sha256",
                "calibration_receipt_file_sha256",
                "study_criteria_semantic_sha256",
                "study_criteria_file_sha256",
                "runtime",
                "candidate_file_sha256",
                "reviewer_id",
                "review_note",
            },
            field="execution manifest",
        )
        if not isinstance(value["runtime"], dict):
            raise ExperimentContractError("execution manifest runtime must be an object")
        return cls(
            schema_version=value["schema_version"],
            status=value["status"],
            execution_callable_authority=value["execution_callable_authority"],
            design_sha256=value["design_sha256"],
            calibrated_design_sha256=value["calibrated_design_sha256"],
            calibrated_design_file_sha256=value["calibrated_design_file_sha256"],
            calibration_receipt_sha256=value["calibration_receipt_sha256"],
            calibration_receipt_file_sha256=value["calibration_receipt_file_sha256"],
            study_criteria_semantic_sha256=value["study_criteria_semantic_sha256"],
            study_criteria_file_sha256=value["study_criteria_file_sha256"],
            runtime=RuntimeFingerprint.from_dict(value["runtime"]),
            candidate_file_sha256=value["candidate_file_sha256"],
            reviewer_id=value["reviewer_id"],
            review_note=value["review_note"],
        )

    def validate(self, *, design: FixedReferenceStudyDesign, observed: RuntimeFingerprint) -> None:
        design.assert_locked_for_behavior()
        if self.design_sha256 != design.sha256:
            raise ExperimentContractError("execution manifest does not bind this study design")
        if (
            self.study_criteria_semantic_sha256 != design.study_criteria_semantic_sha256
            or self.study_criteria_file_sha256 != design.study_criteria_file_sha256
        ):
            raise ExperimentContractError(
                "execution manifest does not bind the study design's criteria"
            )
        assert_runtime_matches(self.runtime, observed)


def assert_runtime_matches(
    expected: RuntimeFingerprint,
    observed: RuntimeFingerprint,
) -> None:
    """Reject every runtime drift; there are no ignored or compatible fields."""

    differences = [
        field
        for field in expected.__dataclass_fields__
        if getattr(expected, field) != getattr(observed, field)
    ]
    if differences:
        raise ExperimentContractError("runtime fingerprint mismatch: " + ", ".join(differences))


@dataclass(frozen=True, slots=True)
class ResourceCalibrationReceipt:
    """Throughput/memory timing that is prohibited from supporting behavior."""

    purpose: EvidencePurpose
    design_sha256: str
    runtime_sha256: str
    calibration_seed: int
    environment_steps: int
    wall_seconds: float
    peak_resident_bytes: int | None

    def __post_init__(self) -> None:
        if self.purpose is not EvidencePurpose.RESOURCE_CALIBRATION:
            raise ExperimentContractError("resource receipt purpose must be resource_calibration")
        for field in ("design_sha256", "runtime_sha256"):
            _require_sha256(getattr(self, field), field=field)
        if (
            not isinstance(self.calibration_seed, int)
            or isinstance(self.calibration_seed, bool)
            or self.calibration_seed < 0
        ):
            raise ExperimentContractError("calibration_seed must be a non-negative integer")
        _positive_int(self.environment_steps, field="environment_steps")
        if (
            isinstance(self.wall_seconds, bool)
            or not isinstance(self.wall_seconds, (int, float))
            or not math.isfinite(float(self.wall_seconds))
            or self.wall_seconds <= 0.0
        ):
            raise ExperimentContractError("wall_seconds must be finite and positive")
        if self.peak_resident_bytes is not None:
            _positive_int(self.peak_resident_bytes, field="peak_resident_bytes")


def sha256_file(path: Path, *, maximum_bytes: int | None = None) -> str:
    """Hash one bounded regular-file snapshot through a no-follow descriptor."""

    resolved = Path(path)
    descriptor: int | None = None
    digest = hashlib.sha256()
    if maximum_bytes is not None and (
        not isinstance(maximum_bytes, int) or isinstance(maximum_bytes, bool) or maximum_bytes < 1
    ):
        raise ExperimentContractError("maximum_bytes must be a positive integer")
    try:
        path_before = resolved.lstat()
        if stat.S_ISLNK(path_before.st_mode):
            raise ExperimentContractError(f"artifact must not be a symlink: {resolved}")
        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(resolved, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ExperimentContractError(f"artifact is not a regular file: {resolved}")
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        path_before_identity = (
            path_before.st_dev,
            path_before.st_ino,
            path_before.st_size,
            path_before.st_mtime_ns,
            path_before.st_ctime_ns,
        )
        if path_before_identity != before_identity:
            raise ExperimentContractError(f"artifact changed before it was hashed: {resolved}")
        if maximum_bytes is not None and before.st_size > maximum_bytes:
            raise ExperimentContractError(
                f"artifact exceeds bounded size limit {maximum_bytes}: {resolved}"
            )
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                raise ExperimentContractError(f"artifact became shorter while hashed: {resolved}")
            digest.update(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise ExperimentContractError(f"artifact grew while it was hashed: {resolved}")
        after = os.fstat(descriptor)
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        path_after = resolved.lstat()
        path_identity = (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        )
        if before_identity != after_identity or before_identity != path_identity:
            raise ExperimentContractError(f"artifact changed while it was hashed: {resolved}")
    except ExperimentContractError:
        raise
    except OSError as exc:
        raise ExperimentContractError(f"cannot hash artifact {resolved}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CheckpointReceipt:
    """Identity and budget receipt for a predetermined final checkpoint."""

    purpose: EvidencePurpose
    design_sha256: str
    runtime_sha256: str
    train_seed: int
    checkpoint_rule: str
    requested_timesteps: int
    observed_timesteps: int
    checkpoint_sha256: str
    checkpoint_size_bytes: int

    def __post_init__(self) -> None:
        if self.purpose is not EvidencePurpose.BEHAVIORAL_EVALUATION:
            raise ExperimentContractError("a checkpoint receipt must be behavioral_evaluation")
        for field in ("design_sha256", "runtime_sha256", "checkpoint_sha256"):
            _require_sha256(getattr(self, field), field=field)
        if (
            not isinstance(self.train_seed, int)
            or isinstance(self.train_seed, bool)
            or self.train_seed < 0
        ):
            raise ExperimentContractError("train_seed must be a non-negative integer")
        for field in ("requested_timesteps", "observed_timesteps", "checkpoint_size_bytes"):
            _positive_int(getattr(self, field), field=field)
        if self.checkpoint_rule != "final_timestep_only":
            raise ExperimentContractError("checkpoint_rule must be final_timestep_only")
        if self.requested_timesteps != self.observed_timesteps:
            raise ExperimentContractError("observed_timesteps must exactly equal the fixed budget")


def checkpoint_receipt(
    *,
    design: FixedReferenceStudyDesign,
    runtime: RuntimeFingerprint,
    train_seed: int,
    observed_timesteps: int,
    checkpoint_path: Path,
) -> CheckpointReceipt:
    """Bind exact checkpoint bytes to one seed, design, runtime, and budget."""

    design.assert_locked_for_behavior()
    if train_seed not in design.train_seeds:
        raise ExperimentContractError("train_seed is not in the predetermined schedule")
    try:
        size = checkpoint_path.stat().st_size
    except OSError as exc:
        raise ExperimentContractError(f"cannot stat checkpoint: {exc}") from exc
    return CheckpointReceipt(
        purpose=EvidencePurpose.BEHAVIORAL_EVALUATION,
        design_sha256=design.sha256,
        runtime_sha256=runtime.sha256,
        train_seed=train_seed,
        checkpoint_rule=design.checkpoint_rule,
        requested_timesteps=design.ppo.total_timesteps_per_seed,
        observed_timesteps=observed_timesteps,
        checkpoint_sha256=sha256_file(checkpoint_path),
        checkpoint_size_bytes=size,
    )


@dataclass(frozen=True, slots=True)
class BehavioralEvaluationManifest:
    """One arm of a matched-state numeric-reference intervention.

    Closed-loop oracle rollouts require a different manifest because their
    realized states and untransformed oracle-result traces may diverge after
    the first different action.
    """

    condition_id: str
    intervention_protocol_id: str
    design_sha256: str
    runtime_sha256: str
    evaluator_sha256: str
    ground_truth_reference_sha256: str
    matched_state_snapshot_set_sha256: str
    policy_input_reference_sha256: str
    policy_input_transform_receipt_sha256: str
    checkpoint_sha256_by_train_seed: tuple[tuple[int, str], ...]
    evaluation_seeds: tuple[int, ...]
    max_episode_steps: int
    deterministic_actions: bool
    perturbation_schedule_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.condition_id, str) or not self.condition_id.strip():
            raise ExperimentContractError("condition_id must be a non-empty string")
        if self.intervention_protocol_id != MATCHED_STATE_CAUSAL_INPUT_PROTOCOL_ID:
            raise ExperimentContractError(
                "behavioral evaluation manifest must use the matched-state "
                "numeric-reference intervention protocol"
            )
        for field in (
            "design_sha256",
            "runtime_sha256",
            "evaluator_sha256",
            "ground_truth_reference_sha256",
            "matched_state_snapshot_set_sha256",
            "policy_input_reference_sha256",
            "policy_input_transform_receipt_sha256",
            "perturbation_schedule_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if isinstance(self.checkpoint_sha256_by_train_seed, (str, bytes)):
            raise ExperimentContractError("checkpoint mapping must be an ordered sequence")
        try:
            raw_checkpoints = tuple(self.checkpoint_sha256_by_train_seed)
        except TypeError as exc:
            raise ExperimentContractError("checkpoint mapping must be an ordered sequence") from exc
        checkpoints: tuple[tuple[int, str], ...] = tuple()
        normalized: list[tuple[int, str]] = []
        for pair in raw_checkpoints:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ExperimentContractError(
                    "each checkpoint mapping must be [train_seed, sha256]"
                )
            normalized.append((pair[0], pair[1]))
        checkpoints = tuple(normalized)
        if isinstance(self.evaluation_seeds, (str, bytes)):
            raise ExperimentContractError("evaluation_seeds must be an ordered sequence")
        try:
            seeds = tuple(self.evaluation_seeds)
        except TypeError as exc:
            raise ExperimentContractError("evaluation_seeds must be an ordered sequence") from exc
        object.__setattr__(self, "checkpoint_sha256_by_train_seed", checkpoints)
        object.__setattr__(self, "evaluation_seeds", seeds)
        if not checkpoints:
            raise ExperimentContractError("evaluation must include checkpoints")
        train_seeds = [pair[0] for pair in checkpoints]
        if len(set(train_seeds)) != len(train_seeds):
            raise ExperimentContractError("checkpoint train seeds must be unique")
        for train_seed, digest in checkpoints:
            if not isinstance(train_seed, int) or isinstance(train_seed, bool) or train_seed < 0:
                raise ExperimentContractError(
                    "checkpoint train seeds must be non-negative integers"
                )
            _require_sha256(digest, field="checkpoint_sha256")
        if not seeds or len(set(seeds)) != len(seeds):
            raise ExperimentContractError("evaluation_seeds must be non-empty and unique")
        if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds):
            raise ExperimentContractError("evaluation_seeds must contain non-negative integers")
        _positive_int(self.max_episode_steps, field="max_episode_steps")
        if not isinstance(self.deterministic_actions, bool) or not self.deterministic_actions:
            raise ExperimentContractError("behavioral evaluation actions must be deterministic")


def assert_matched_evaluation_manifests(
    manifests: tuple[BehavioralEvaluationManifest, ...],
) -> None:
    """Validate the exact four-arm matched-state causal-input intervention."""

    if len(manifests) != len(CAUSAL_INPUT_CONDITION_IDS):
        raise ExperimentContractError("matched evaluation requires exactly four conditions")
    by_condition = {manifest.condition_id: manifest for manifest in manifests}
    if len(by_condition) != len(manifests):
        raise ExperimentContractError("condition_id values must be unique")
    if set(by_condition) != CAUSAL_INPUT_CONDITION_IDS:
        missing = sorted(CAUSAL_INPUT_CONDITION_IDS - set(by_condition))
        extra = sorted(set(by_condition) - CAUSAL_INPUT_CONDITION_IDS)
        raise ExperimentContractError(
            f"causal-input condition IDs differ from the exact contract; "
            f"missing={missing!r}, extra={extra!r}"
        )

    exact = by_condition["C_exact"]
    if exact.policy_input_reference_sha256 != exact.ground_truth_reference_sha256:
        raise ExperimentContractError(
            "C_exact policy input must equal the common ground-truth reference"
        )
    for condition_id, manifest in by_condition.items():
        if (
            condition_id != "C_exact"
            and manifest.policy_input_reference_sha256 == manifest.ground_truth_reference_sha256
        ):
            raise ExperimentContractError(
                f"condition {condition_id!r} must change the policy-input reference"
            )

    policy_inputs = [manifest.policy_input_reference_sha256 for manifest in manifests]
    if len(set(policy_inputs)) != len(policy_inputs):
        raise ExperimentContractError(
            "each causal-input condition must pin distinct policy-input bytes"
        )
    transform_receipts = [manifest.policy_input_transform_receipt_sha256 for manifest in manifests]
    if len(set(transform_receipts)) != len(transform_receipts):
        raise ExperimentContractError(
            "each causal-input condition must pin a distinct transform receipt"
        )

    reference = exact
    # Only the condition ID, policy-input bytes, and exact transformation receipt
    # may vary. The evaluator target and cached matched-state oracle snapshots are
    # common by construction.
    matched_fields = (
        "intervention_protocol_id",
        "design_sha256",
        "runtime_sha256",
        "evaluator_sha256",
        "ground_truth_reference_sha256",
        "matched_state_snapshot_set_sha256",
        "checkpoint_sha256_by_train_seed",
        "evaluation_seeds",
        "max_episode_steps",
        "deterministic_actions",
        "perturbation_schedule_sha256",
    )
    for manifest in manifests:
        if manifest is exact:
            continue
        mismatches = [
            field
            for field in matched_fields
            if getattr(reference, field) != getattr(manifest, field)
        ]
        if mismatches:
            raise ExperimentContractError(
                f"condition {manifest.condition_id!r} is not matched: " + ", ".join(mismatches)
            )


def assert_temporal_reference_variation(
    reference_values: ArrayLike,
    *,
    minimum_dynamic_range: float = 1e-8,
) -> None:
    """Reject references whose shuffle/shift arms would be numerically identical.

    Time-invariant commands can establish survival or tracking feasibility.  They
    cannot test temporal reference sensitivity because shuffling and shifting
    leave the policy input unchanged. Passing this structural precheck does not
    establish that variation is meaningful, that arms differ as intended, or
    that policy behavior causally depends on the reference.
    """

    values = np.asarray(reference_values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ExperimentContractError("causal-use reference must be a finite T x D matrix")
    if not np.isfinite(values).all():
        raise ExperimentContractError("causal-use reference contains non-finite values")
    if (
        not isinstance(minimum_dynamic_range, (int, float))
        or isinstance(minimum_dynamic_range, bool)
        or not math.isfinite(float(minimum_dynamic_range))
        or minimum_dynamic_range <= 0.0
    ):
        raise ExperimentContractError("minimum_dynamic_range must be finite and positive")
    dynamic_range = np.ptp(values, axis=0)
    if float(np.max(dynamic_range)) < float(minimum_dynamic_range):
        raise ExperimentContractError(
            "time-invariant reference cannot identify shuffled or time-shifted causal use"
        )


def assert_causal_reference_identifiable(
    reference_values: ArrayLike,
    *,
    minimum_dynamic_range: float = 1e-8,
) -> None:
    """Compatibility alias for the temporal-variation precheck.

    The historical function name is intentionally not evidence of causal
    identifiability. Callers must separately verify exact perturbation bytes,
    action sensitivity, and expected-direction behavioral effects.
    """

    assert_temporal_reference_variation(
        reference_values,
        minimum_dynamic_range=minimum_dynamic_range,
    )
