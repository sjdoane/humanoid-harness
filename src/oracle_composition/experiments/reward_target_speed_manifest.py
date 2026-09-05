"""Frozen design, runtime identity, and no-learning equivalence schemas."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import stat
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from oracle_composition.envs import humanoid as humanoid_module
from oracle_composition.envs.humanoid import (
    CONTACT_CAPTURE_ID,
    HumanoidExperimentConfig,
    make_humanoid_env,
)
from oracle_composition.rewards.contract import (
    ACTUATOR_QVEL_INDICES_BY_ACTION_V1,
    GENERALIZED_ACTUATOR_TORQUE_CAPACITY_N_M,
    RewardArtifactIdentityV1,
    canonical_json_bytes,
)
from oracle_composition.rewards.sandbox import sandbox_source_sha256, sandbox_worker_source_sha256
from oracle_composition.rewards.static_validation import validate_task_term_source

from .runtime_identity import (
    dependency_lock_path,
    module_sha256,
    source_tree_sha256,
    space_sha256,
)


def _deep_freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze_json(child) for key, child in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze_json(child) for child in value)
    return value


EXPECTED_PACKAGE_VERSIONS = MappingProxyType(
    {
        "gymnasium": "1.3.0",
        "mujoco": "3.12.0",
        "numpy": "2.5.2",
        "stable_baselines3": "2.9.0",
        "torch": "2.14.0",
    }
)
FAMILY_ID = "family-b-target-speed-v1"
DESIGN_STATUS = "proposed"
EVIDENCE_LABEL = "exploratory_reward_cycle"
TRAINING_SEEDS = (101, 202, 303, 404, 505)
EVALUATION_SEEDS = tuple(range(11001, 11021))
NO_LEARNING_SMOKE_SEED = 20260904
NO_LEARNING_SMOKE_STEPS = 8
MAX_CONFIG_BYTES = 128 * 1024
MAX_IDENTITY_FILE_BYTES = 64 * 1024 * 1024
RUNTIME_SOURCE_ROLES = (
    "compositor",
    "contract",
    "environment",
    "manifest",
    "policy",
    "protected_endpoint",
    "sandbox",
    "sandbox_worker",
    "scale_calibration",
    "trainer",
    "validator",
)
SMOKE_COMPARISON_FIELDS = (
    "observation",
    "stock_scalar",
    "terminated",
    "truncated",
    "integration_state",
    "qacc",
    "qfrc_actuator",
    "simulation_time",
    "time_limit_elapsed_steps",
)

FROZEN_STUDY_CONFIG_V1: Mapping[str, object] = _deep_freeze_json(
    {
        "schema_version": 1,
        "family_id": FAMILY_ID,
        "design_status": DESIGN_STATUS,
        "evidence_label": EVIDENCE_LABEL,
        "claim_ceiling": "software_contract_parity_sandbox_and_evaluator_only",
        "authorable_field": "candidate_task_term_source",
        "target_speeds_m_s": [0.5, 1.0, 1.5],
        "first_cycle_target_speed_m_s": 1.0,
        "environment": {
            "env_id": "Humanoid-v5",
            "default_kwargs": {
                "xml_file": "humanoid.xml",
                "frame_skip": 5,
                "forward_reward_weight": 1.25,
                "ctrl_cost_weight": 0.1,
                "contact_cost_weight": 5e-7,
                "contact_cost_range": ["-Infinity", 10.0],
                "healthy_reward": 5.0,
                "terminate_when_unhealthy": True,
                "healthy_z_range_m": [1.0, 2.0],
                "reset_noise_scale": 0.01,
                "exclude_current_positions_from_observation": True,
                "include_cinert_in_observation": True,
                "include_cvel_in_observation": True,
                "include_qfrc_actuator_in_observation": True,
                "include_cfrc_ext_in_observation": True,
            },
            "policy_observation_shape": [348],
            "policy_observation_dtype": "float64",
            "target_appended_to_observation": False,
            "physical_action_shape": [17],
            "physical_action_dtype": "float32",
            "physical_action_bounds": [-0.4, 0.4],
            "control_period_s": 0.015,
            "physics_timestep_s": 0.003,
            "physics_substeps": 5,
            "actuator_qvel_indices_by_action": list(ACTUATOR_QVEL_INDICES_BY_ACTION_V1),
            "max_episode_steps": 1000,
            "oracle_id": "null_oracle/v1",
            "reference_id": "none/v1",
            "tracker_id": "none/v1",
        },
        "candidate_interface": {
            "schema_id": "family-b-target-speed/reward-contract/v1",
            "read_set": ["com_x_velocity_m_s", "target_speed_m_s"],
            "maximum_source_bytes": 16384,
            "maximum_ast_nodes": 512,
            "functions": ["abs", "min", "max", "exp", "sqrt", "clip"],
            "task_term_abs_max": 1000.0,
            "total_abs_max": 1024.0,
        },
        "scale_calibration": {
            "applies_to": "candidate_rk",
            "target_speed_m_s": 1.0,
            "velocity_grid_m_s": [1.0 + 0.25 * index for index in range(-16, 17)],
            "stock_grid_mean": 1.25,
            "stock_grid_population_sd": 2.9755951785595207,
            "affine_alpha_closed_interval": [0.25, 4.0],
            "affine_beta_absolute_max": 10.0,
            "zero_or_non_finite_raw_sd": "reject",
            "baseline_terms": "exact_frozen_definitions_not_candidate_calibrations",
        },
        "endpoint": {
            "id": "protected_target_speed_hold/v1",
            "burn_in_steps": 200,
            "scored_steps": [201, 1000],
            "denominator_steps": 800,
            "sigma_m_s": 0.25,
            "post_termination_value": 0.0,
            "state_source": "body_mass_weighted_direct_mujoco_xipos",
        },
        "guardrails": {
            "survival_episodes_per_checkpoint": 19,
            "checkpoint_count": 4,
            "total_checkpoints": 5,
            "root_height_closed_interval_m": [1.0, 2.0],
            "minimum_torso_up_z": 0.5,
            "normalized_action_abs_max": 1.0,
            "normalized_action_rate_abs_max_per_s": 2.0 / 0.015,
            "torque_capacity_n_m": list(GENERALIZED_ACTUATOR_TORQUE_CAPACITY_N_M),
            "torque_capacity_relative_tolerance": 1e-9,
            "torque_action_absolute_tolerance": 1e-7,
            "non_foot_floor_contact_count": 0,
            "allowed_floor_geoms": ["left_foot", "right_foot"],
            "relative_non_inferiority_margin": 0.10,
            "relative_margin_label": "exploratory",
            "allowed_foot_impacts": "descriptive_only",
        },
        "trainer": {
            "algorithm": "stable_baselines3.PPO/v2.9.0",
            "policy": "local.SquashedGaussianActorCriticPolicy/tanh_jacobian/v1",
            "device": "cpu",
            "total_timesteps_per_arm_seed": 1048576,
            "n_envs": 4,
            "n_steps_per_env": 2048,
            "transitions_per_rollout": 8192,
            "rollout_count": 128,
            "batch_size": 512,
            "minibatches_per_epoch": 16,
            "n_epochs": 10,
            "learning_rate": 0.0003,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "value_clip": None,
            "entropy_coefficient": 0.0,
            "value_coefficient": 0.5,
            "max_gradient_norm": 0.5,
            "normalize_advantage": True,
            "g_sde": False,
            "sde_sample_frequency": -1,
            "initial_log_standard_deviation": 0.0,
            "full_standard_deviation": True,
            "use_expln": False,
            "actor_layers": [256, 256],
            "critic_layers": [256, 256],
            "activation": "Tanh",
            "orthogonal_initialization": True,
            "features_extractor": "FlattenExtractor",
            "rollout_buffer": "stable_baselines3.common.buffers.RolloutBuffer",
            "rollout_buffer_kwargs": {},
            "optimizer": "Adam",
            "optimizer_epsilon": 1e-5,
            "target_kl": None,
            "stats_window_size": 100,
            "tensorboard_log": None,
            "verbose": 0,
            "observation_normalization": "none",
            "return_normalization": "none",
            "action_transform": "policy_-1_1_to_physical_-0.4_0.4_once",
            "learn": {
                "reset_num_timesteps": True,
                "progress_bar": False,
                "pre_update_likelihood_audit_rollouts": 128,
            },
            "checkpoint_rule": "fresh_initialization_final_timestep_only",
            "deterministic_evaluation_actions": True,
        },
        "training_seeds": list(TRAINING_SEEDS),
        "evaluation_reset_seeds": list(EVALUATION_SEEDS),
        "compute_reservation": {"amount": 150, "unit": "CPU minute"},
        "cycle_arms": {
            "cycle_1": ["stock_r0", "candidate_rk"],
            "cycle_2_and_later": ["stock_r0", "manual_target_speed_v1", "candidate_rk"],
        },
        "manual_baseline": {
            "id": "manual_target_speed_v1",
            "source": "candidates/manual_target_speed_v1.py",
            "task_term": "1.25 * (1 - min(1, abs(v - v_star) / v_star))",
            "trusted_affine": {"alpha": 1.0, "beta": 0.0},
            "trained_in_b0": False,
        },
        "matched_baseline_plan": [
            "freeze_execution_manifest",
            "train_five_fresh_stock_r0_policies",
            "evaluate_all_final_checkpoints_on_twenty_reset_seeds",
            "reopen_and_rehash_all_artifacts",
            "apply_baseline_viability_stop_rule",
            "lock_candidate_validator_and_affine_receipts",
            "train_candidate_from_fresh_matched_initializations",
            "evaluate_with_identical_protected_endpoint",
            "preserve_all_failed_or_missing_seeds_without_retry",
            "reuse_r0_only_when_every_other_execution_identity_is_byte_identical",
        ],
        "b0": {"training_permitted": False, "behavioral_evaluation_permitted": False},
    }
)


class TargetSpeedManifestError(ValueError):
    """Raised on design, runtime, source, or result-lineage drift."""


def _require_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise TargetSpeedManifestError(f"{field} must be a lowercase SHA-256")
    return value


def _file_state(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_uid),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_regular_file_stable(
    path: Path, *, maximum_bytes: int = MAX_IDENTITY_FILE_BYTES
) -> tuple[bytes, tuple[int, ...]]:
    target = Path(os.path.abspath(path))
    descriptor: int | None = None
    try:
        before = target.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise TargetSpeedManifestError("identity input must be a regular non-symlink file")
        if not 0 < before.st_size <= maximum_bytes:
            raise TargetSpeedManifestError("identity input is empty or exceeds its size bound")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(target, flags)
        opened = os.fstat(descriptor)
        if _file_state(opened) != _file_state(before):
            raise TargetSpeedManifestError("identity input changed before it was opened")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                raise TargetSpeedManifestError("identity input shortened during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise TargetSpeedManifestError("identity input grew during read")
        after = os.fstat(descriptor)
        visible_after = target.lstat()
        if _file_state(after) != _file_state(opened) or _file_state(visible_after) != _file_state(
            opened
        ):
            raise TargetSpeedManifestError("identity input changed during read")
        return b"".join(chunks), _file_state(opened)
    except TargetSpeedManifestError:
        raise
    except OSError as exc:
        raise TargetSpeedManifestError(f"cannot read identity input: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _file_sha256(path: Path) -> str:
    payload, _state = _read_regular_file_stable(path)
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(value: np.ndarray, *, label: str) -> str:
    array = np.ascontiguousarray(value)
    if array.size < 1 or not np.isfinite(array).all():
        raise TargetSpeedManifestError(f"{label} must be finite and nonempty")
    metadata = canonical_json_bytes(
        {"label": label, "dtype": array.dtype.str, "shape": list(array.shape)}
    )
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(4, "big"))
    digest.update(metadata)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise TargetSpeedManifestError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    target = Path(os.path.abspath(path))
    try:
        encoded, _state = _read_regular_file_stable(target, maximum_bytes=MAX_CONFIG_BYTES)
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                TargetSpeedManifestError(f"non-finite JSON constant {item!r}")
            ),
        )
    except TargetSpeedManifestError:
        raise
    except (OSError, UnicodeError, ValueError, RecursionError) as exc:
        raise TargetSpeedManifestError("study config is invalid JSON") from exc
    if type(value) is not dict:
        raise TargetSpeedManifestError("study config root must be an object")
    return value, encoded


@dataclass(frozen=True, slots=True)
class TargetSpeedStudyDesignV1:
    payload: Mapping[str, object]
    file_sha256: str

    def __post_init__(self) -> None:
        if canonical_json_bytes(self.payload) != canonical_json_bytes(FROZEN_STUDY_CONFIG_V1):
            raise TargetSpeedManifestError("study config differs from the frozen v1 proposal")
        snapshot = json.loads(canonical_json_bytes(self.payload).decode("utf-8"))
        object.__setattr__(self, "payload", _deep_freeze_json(snapshot))
        _require_sha256(self.file_sha256, field="file_sha256")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.payload)).hexdigest()


def load_target_speed_study_design(path: Path) -> TargetSpeedStudyDesignV1:
    target = Path(os.path.abspath(path))
    payload, encoded = _load_json(target)
    return TargetSpeedStudyDesignV1(
        payload=payload,
        file_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _smoke_actions() -> np.ndarray:
    actions = np.vstack(
        (
            np.zeros(17, dtype=np.float32),
            np.full(17, np.float32(0.4), dtype=np.float32),
            np.full(17, np.float32(-0.4), dtype=np.float32),
            np.linspace(-0.4, 0.4, 17, dtype=np.float32),
            np.linspace(0.4, -0.4, 17, dtype=np.float32),
            np.full(17, np.float32(0.125), dtype=np.float32),
            np.full(17, np.float32(-0.125), dtype=np.float32),
            np.zeros(17, dtype=np.float32),
        )
    )
    if actions.shape != (NO_LEARNING_SMOKE_STEPS, 17):
        raise TargetSpeedManifestError("fixed control canary shape drifted")
    return np.ascontiguousarray(actions)


def _integration_state(environment: object) -> np.ndarray:
    import mujoco

    physical = environment.unwrapped
    flag = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(int(mujoco.mj_stateSize(physical.model, flag)), dtype=np.float64)
    mujoco.mj_getState(physical.model, physical.data, state, flag)
    return np.ascontiguousarray(state)


def _elapsed_steps(environment: object) -> int:
    value = getattr(environment, "_elapsed_steps", None)
    if type(value) is not int or value < 0:
        raise TargetSpeedManifestError("TimeLimit clock is unavailable")
    return value


def _trace_update(digest: Any, label: str, value: object) -> None:
    label_bytes = label.encode("utf-8")
    digest.update(len(label_bytes).to_bytes(4, "big"))
    digest.update(label_bytes)
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        metadata = canonical_json_bytes({"dtype": array.dtype.str, "shape": list(array.shape)})
        payload = len(metadata).to_bytes(4, "big") + metadata + array.tobytes(order="C")
    elif type(value) is float:
        payload = struct.pack(">d", value)
    elif type(value) is bool:
        payload = b"\x01" if value else b"\x00"
    elif type(value) is int:
        payload = value.to_bytes(8, "big", signed=True)
    else:
        raise TargetSpeedManifestError(f"unsupported smoke trace value {label}")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


@dataclass(frozen=True, slots=True)
class NoLearningInstrumentationSmokeV1:
    seed: int
    action_sha256: str
    compared_steps: int
    comparison_fields: tuple[str, ...]
    plain_trace_sha256: str
    instrumented_trace_sha256: str
    passed: bool
    behavioral_evidence: bool = False

    def __post_init__(self) -> None:
        for field in ("action_sha256", "plain_trace_sha256", "instrumented_trace_sha256"):
            _require_sha256(getattr(self, field), field=field)
        expected_action_sha256 = hashlib.sha256(_smoke_actions().tobytes(order="C")).hexdigest()
        if (
            self.seed != NO_LEARNING_SMOKE_SEED
            or self.action_sha256 != expected_action_sha256
            or self.compared_steps != NO_LEARNING_SMOKE_STEPS
            or self.comparison_fields != SMOKE_COMPARISON_FIELDS
        ):
            raise TargetSpeedManifestError("fixed-control smoke definition differs")
        if self.passed is not True or self.behavioral_evidence is not False:
            raise TargetSpeedManifestError("smoke receipt cannot authorize a behavioral claim")
        if self.plain_trace_sha256 != self.instrumented_trace_sha256:
            raise TargetSpeedManifestError("plain and instrumented traces diverged")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "seed": self.seed,
            "action_sha256": self.action_sha256,
            "compared_steps": self.compared_steps,
            "comparison_fields": list(self.comparison_fields),
            "plain_trace_sha256": self.plain_trace_sha256,
            "instrumented_trace_sha256": self.instrumented_trace_sha256,
            "passed": self.passed,
            "behavioral_evidence": self.behavioral_evidence,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


def run_no_learning_instrumentation_smoke() -> NoLearningInstrumentationSmokeV1:
    """Compare one fixed control trace; never construct or update a policy."""

    configuration = HumanoidExperimentConfig(terminate_when_unhealthy=True)
    plain = make_humanoid_env(configuration, capture_substep_contacts=False)
    instrumented = make_humanoid_env(configuration, capture_substep_contacts=True)
    fields = SMOKE_COMPARISON_FIELDS
    actions = _smoke_actions()
    action_sha = hashlib.sha256(actions.tobytes(order="C")).hexdigest()
    plain_digest = hashlib.sha256()
    instrumented_digest = hashlib.sha256()
    compared_steps = 0
    try:
        plain_observation, _plain_reset_data = plain.reset(seed=NO_LEARNING_SMOKE_SEED)
        instrumented_observation, _instrumented_reset_data = instrumented.reset(
            seed=NO_LEARNING_SMOKE_SEED
        )
        for digest, environment, observation in (
            (plain_digest, plain, plain_observation),
            (instrumented_digest, instrumented, instrumented_observation),
        ):
            _trace_update(digest, "reset/observation", observation)
            _trace_update(digest, "reset/integration_state", _integration_state(environment))
            _trace_update(digest, "reset/time_limit_elapsed_steps", _elapsed_steps(environment))
        if not np.array_equal(plain_observation, instrumented_observation):
            raise TargetSpeedManifestError("plain and instrumented reset observations differ")
        if not np.array_equal(_integration_state(plain), _integration_state(instrumented)):
            raise TargetSpeedManifestError("plain and instrumented reset states differ")
        for index, action in enumerate(actions, start=1):
            plain_step = plain.step(action.copy())
            instrumented_step = instrumented.step(action.copy())
            plain_values = (
                plain_step[0],
                float(plain_step[1]),
                bool(plain_step[2]),
                bool(plain_step[3]),
                _integration_state(plain),
                np.ascontiguousarray(plain.unwrapped.data.qacc.copy()),
                np.ascontiguousarray(plain.unwrapped.data.qfrc_actuator.copy()),
                float(plain.unwrapped.data.time),
                _elapsed_steps(plain),
            )
            instrumented_values = (
                instrumented_step[0],
                float(instrumented_step[1]),
                bool(instrumented_step[2]),
                bool(instrumented_step[3]),
                _integration_state(instrumented),
                np.ascontiguousarray(instrumented.unwrapped.data.qacc.copy()),
                np.ascontiguousarray(instrumented.unwrapped.data.qfrc_actuator.copy()),
                float(instrumented.unwrapped.data.time),
                _elapsed_steps(instrumented),
            )
            for field, left, right in zip(fields, plain_values, instrumented_values, strict=True):
                equal = (
                    isinstance(left, np.ndarray)
                    and isinstance(right, np.ndarray)
                    and left.dtype == right.dtype
                    and left.shape == right.shape
                    and np.array_equal(left, right)
                ) or (
                    not isinstance(left, np.ndarray) and type(left) is type(right) and left == right
                )
                if not equal:
                    raise TargetSpeedManifestError(
                        f"plain and instrumented {field} differ at control step {index}"
                    )
                _trace_update(plain_digest, f"step/{index}/{field}", left)
                _trace_update(instrumented_digest, f"step/{index}/{field}", right)
            if instrumented.unwrapped.last_control_step_substeps != 5:
                raise TargetSpeedManifestError("instrumentation omitted a physics substep")
            compared_steps = index
            if plain_values[2] or plain_values[3]:
                break
    finally:
        plain.close()
        instrumented.close()
    plain_sha = plain_digest.hexdigest()
    instrumented_sha = instrumented_digest.hexdigest()
    if compared_steps < 1 or plain_sha != instrumented_sha:
        raise TargetSpeedManifestError("plain and instrumented fixed-control traces diverged")
    return NoLearningInstrumentationSmokeV1(
        seed=NO_LEARNING_SMOKE_SEED,
        action_sha256=action_sha,
        compared_steps=compared_steps,
        comparison_fields=fields,
        plain_trace_sha256=plain_sha,
        instrumented_trace_sha256=instrumented_sha,
        passed=True,
    )


@dataclass(frozen=True, slots=True)
class TargetSpeedRuntimeFingerprintV1:
    package_versions: Mapping[str, str]
    python_version: str
    platform_system: str
    platform_machine: str
    env_id: str
    environment_kwargs: Mapping[str, object]
    observation_shape: tuple[int, ...]
    observation_dtype: str
    action_shape: tuple[int, ...]
    action_dtype: str
    qpos_shape: tuple[int, ...]
    qvel_shape: tuple[int, ...]
    control_period_s: float
    physics_timestep_s: float
    frame_skip: int
    model_sha256: str
    gymnasium_source_sha256: str
    dependency_lock_sha256: str
    source_tree_sha256: str
    observation_space_sha256: str
    action_space_sha256: str
    actuator_gear_sha256: str
    source_hashes: Mapping[str, str]
    actuator_qvel_indices_by_action: tuple[int, ...]
    torque_capacity_n_m: tuple[float, ...]
    contact_capture_id: str
    instrumentation_smoke_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.package_versions, Mapping) or dict(self.package_versions) != dict(
            EXPECTED_PACKAGE_VERSIONS
        ):
            raise TargetSpeedManifestError("installed dependency versions differ from the recipe")
        for field in ("python_version", "platform_system", "platform_machine"):
            if type(getattr(self, field)) is not str or not getattr(self, field):
                raise TargetSpeedManifestError(f"{field} must be a nonempty string")
        if (
            self.env_id != "Humanoid-v5"
            or self.observation_shape != (348,)
            or self.action_shape != (17,)
        ):
            raise TargetSpeedManifestError("stock Humanoid-v5 spaces differ")
        if self.observation_dtype != "float64" or self.action_dtype != "float32":
            raise TargetSpeedManifestError("stock Humanoid-v5 dtypes differ")
        if self.qpos_shape != (24,) or self.qvel_shape != (23,):
            raise TargetSpeedManifestError("stock Humanoid-v5 state shapes differ")
        if (
            self.control_period_s != 0.015
            or self.physics_timestep_s != 0.003
            or self.frame_skip != 5
        ):
            raise TargetSpeedManifestError("stock simulator cadence differs")
        if (
            self.actuator_qvel_indices_by_action != ACTUATOR_QVEL_INDICES_BY_ACTION_V1
            or type(self.torque_capacity_n_m) is not tuple
            or self.torque_capacity_n_m != GENERALIZED_ACTUATOR_TORQUE_CAPACITY_N_M
        ):
            raise TargetSpeedManifestError("stock actuator state mapping or capacities differ")
        expected_kwargs = HumanoidExperimentConfig(terminate_when_unhealthy=True).gym_kwargs()
        if (
            not isinstance(self.environment_kwargs, Mapping)
            or dict(self.environment_kwargs) != expected_kwargs
        ):
            raise TargetSpeedManifestError("effective Humanoid-v5 kwargs differ")
        if self.contact_capture_id != CONTACT_CAPTURE_ID:
            raise TargetSpeedManifestError("contact instrumentation identity differs")
        for field in (
            "model_sha256",
            "gymnasium_source_sha256",
            "dependency_lock_sha256",
            "source_tree_sha256",
            "observation_space_sha256",
            "action_space_sha256",
            "actuator_gear_sha256",
            "instrumentation_smoke_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if not isinstance(self.source_hashes, Mapping) or tuple(
            sorted(self.source_hashes)
        ) != tuple(sorted(RUNTIME_SOURCE_ROLES)):
            raise TargetSpeedManifestError("runtime source hash closure differs")
        for role, digest in self.source_hashes.items():
            if type(role) is not str or not role:
                raise TargetSpeedManifestError("runtime source role is invalid")
            _require_sha256(digest, field=f"source_hashes.{role}")
        object.__setattr__(self, "package_versions", MappingProxyType(dict(self.package_versions)))
        object.__setattr__(
            self, "environment_kwargs", MappingProxyType(dict(self.environment_kwargs))
        )
        object.__setattr__(
            self, "source_hashes", MappingProxyType(dict(sorted(self.source_hashes.items())))
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "package_versions": dict(self.package_versions),
            "python_version": self.python_version,
            "platform_system": self.platform_system,
            "platform_machine": self.platform_machine,
            "env_id": self.env_id,
            "environment_kwargs": dict(self.environment_kwargs),
            "observation_shape": list(self.observation_shape),
            "observation_dtype": self.observation_dtype,
            "action_shape": list(self.action_shape),
            "action_dtype": self.action_dtype,
            "qpos_shape": list(self.qpos_shape),
            "qvel_shape": list(self.qvel_shape),
            "control_period_s": self.control_period_s,
            "physics_timestep_s": self.physics_timestep_s,
            "frame_skip": self.frame_skip,
            "model_sha256": self.model_sha256,
            "gymnasium_source_sha256": self.gymnasium_source_sha256,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "observation_space_sha256": self.observation_space_sha256,
            "action_space_sha256": self.action_space_sha256,
            "actuator_gear_sha256": self.actuator_gear_sha256,
            "source_hashes": dict(self.source_hashes),
            "actuator_qvel_indices_by_action": list(self.actuator_qvel_indices_by_action),
            "torque_capacity_n_m": list(self.torque_capacity_n_m),
            "contact_capture_id": self.contact_capture_id,
            "instrumentation_smoke_sha256": self.instrumentation_smoke_sha256,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


def inspect_target_speed_runtime_v1() -> tuple[
    TargetSpeedRuntimeFingerprintV1, NoLearningInstrumentationSmokeV1
]:
    import gymnasium
    import mujoco
    import stable_baselines3
    import torch
    from gymnasium.envs.mujoco import humanoid_v5

    packages = {
        "gymnasium": str(gymnasium.__version__),
        "mujoco": str(mujoco.__version__),
        "numpy": str(np.__version__),
        "stable_baselines3": str(stable_baselines3.__version__),
        "torch": str(torch.__version__),
    }
    if packages != dict(EXPECTED_PACKAGE_VERSIONS):
        raise TargetSpeedManifestError("installed dependency versions differ from the recipe")
    smoke = run_no_learning_instrumentation_smoke()
    configuration = HumanoidExperimentConfig(terminate_when_unhealthy=True)
    environment = make_humanoid_env(configuration, capture_substep_contacts=False)
    try:
        observation, _reset_data = environment.reset(seed=NO_LEARNING_SMOKE_SEED)
        physical = environment.unwrapped
        model_path = Path(str(physical.fullpath))
        gym_source = Path(inspect.getsourcefile(humanoid_v5) or "")
        gears = np.abs(np.asarray(physical.model.actuator_gear)[:, 0])
        capacities = tuple(float(value) for value in gears * 0.4)
        qvel_indices = tuple(
            int(physical.model.jnt_dofadr[int(joint_id)])
            for joint_id in np.asarray(physical.model.actuator_trnid)[:, 0]
        )
        source_hashes = {
            "environment": module_sha256(humanoid_module),
            "contract": module_sha256(
                __import__("oracle_composition.rewards.contract", fromlist=["contract"])
            ),
            "compositor": module_sha256(
                __import__("oracle_composition.rewards.stock_humanoid", fromlist=["stock_humanoid"])
            ),
            "validator": hashlib.sha256(
                Path(validate_task_term_source.__code__.co_filename).read_bytes()
            ).hexdigest(),
            "scale_calibration": _file_sha256(
                Path(__file__).parents[1] / "rewards" / "scale_calibration.py"
            ),
            "sandbox": sandbox_source_sha256(),
            "sandbox_worker": sandbox_worker_source_sha256(),
            "protected_endpoint": hashlib.sha256(
                Path(__file__).with_name("reward_target_speed_evaluator.py").read_bytes()
            ).hexdigest(),
            "manifest": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "policy": _file_sha256(Path(__file__).with_name("squashed_policy.py")),
            "trainer": _file_sha256(Path(__file__).with_name("execution.py")),
        }
        return (
            TargetSpeedRuntimeFingerprintV1(
                package_versions=packages,
                python_version=platform.python_version(),
                platform_system=platform.system(),
                platform_machine=platform.machine(),
                env_id=str(physical.spec.id),
                environment_kwargs=configuration.gym_kwargs(),
                observation_shape=tuple(int(value) for value in observation.shape),
                observation_dtype=str(observation.dtype),
                action_shape=tuple(int(value) for value in environment.action_space.shape),
                action_dtype=str(environment.action_space.dtype),
                qpos_shape=tuple(int(value) for value in physical.data.qpos.shape),
                qvel_shape=tuple(int(value) for value in physical.data.qvel.shape),
                control_period_s=float(physical.dt),
                physics_timestep_s=float(physical.model.opt.timestep),
                frame_skip=int(physical.frame_skip),
                model_sha256=_file_sha256(model_path),
                gymnasium_source_sha256=_file_sha256(gym_source),
                dependency_lock_sha256=_file_sha256(dependency_lock_path()),
                source_tree_sha256=source_tree_sha256(),
                observation_space_sha256=space_sha256(environment.observation_space),
                action_space_sha256=space_sha256(environment.action_space),
                actuator_gear_sha256=_array_sha256(
                    np.asarray(physical.model.actuator_gear),
                    label="mujoco_actuator_gear",
                ),
                source_hashes=source_hashes,
                actuator_qvel_indices_by_action=qvel_indices,
                torque_capacity_n_m=capacities,
                contact_capture_id=CONTACT_CAPTURE_ID,
                instrumentation_smoke_sha256=smoke.sha256,
            ),
            smoke,
        )
    finally:
        environment.close()


@dataclass(frozen=True, slots=True)
class FrozenSourceBindingV1:
    role: str
    path: str
    sha256: str
    byte_count: int
    file_state: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            type(self.role) is not str
            or not self.role
            or self.role.startswith("_")
            or "__" in self.role
        ):
            raise TargetSpeedManifestError("manifest source role is invalid")
        if type(self.path) is not str or self.path != os.path.abspath(self.path):
            raise TargetSpeedManifestError("manifest source path must be absolute and normalized")
        _require_sha256(self.sha256, field=f"source_bindings.{self.role}.sha256")
        if type(self.byte_count) is not int or not 0 < self.byte_count <= MAX_IDENTITY_FILE_BYTES:
            raise TargetSpeedManifestError("manifest source byte count is invalid")
        if (
            type(self.file_state) is not tuple
            or len(self.file_state) != 8
            or any(type(item) is not int for item in self.file_state)
            or not stat.S_ISREG(self.file_state[2])
            or self.file_state[5] != self.byte_count
        ):
            raise TargetSpeedManifestError("manifest source file state is invalid")

    @classmethod
    def capture(cls, *, role: str, path: Path) -> FrozenSourceBindingV1:
        target = Path(os.path.abspath(path))
        payload, state = _read_regular_file_stable(target)
        return cls(
            role=role,
            path=str(target),
            sha256=hashlib.sha256(payload).hexdigest(),
            byte_count=len(payload),
            file_state=state,
        )

    def verify(self) -> None:
        target = Path(self.path)
        try:
            payload, current_state = _read_regular_file_stable(target)
        except TargetSpeedManifestError as exc:
            raise TargetSpeedManifestError(f"frozen source {self.role} is unavailable") from exc
        if (
            current_state != self.file_state
            or len(payload) != self.byte_count
            or hashlib.sha256(payload).hexdigest() != self.sha256
        ):
            raise TargetSpeedManifestError(f"frozen source {self.role} mutated after freeze")

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "path": self.path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "file_state": list(self.file_state),
        }


@dataclass(frozen=True, slots=True)
class RewardTargetSpeedExecutionManifestV1:
    status: str
    design_sha256: str
    runtime_sha256: str
    reward_artifacts: Mapping[str, RewardArtifactIdentityV1]
    source_bindings: tuple[FrozenSourceBindingV1, ...]
    training_seeds: tuple[int, ...]
    evaluation_seeds: tuple[int, ...]
    automatic_authorization: bool = False

    def __post_init__(self) -> None:
        if self.status != "frozen":
            raise TargetSpeedManifestError("execution manifest status must be frozen")
        _require_sha256(self.design_sha256, field="design_sha256")
        _require_sha256(self.runtime_sha256, field="runtime_sha256")
        if self.training_seeds != TRAINING_SEEDS or self.evaluation_seeds != EVALUATION_SEEDS:
            raise TargetSpeedManifestError("execution manifest seed schedule differs")
        if self.automatic_authorization is not False:
            raise TargetSpeedManifestError("execution manifest cannot automatically authorize work")
        if (
            not isinstance(self.reward_artifacts, Mapping)
            or not self.reward_artifacts
            or any(
                not isinstance(value, RewardArtifactIdentityV1)
                for value in self.reward_artifacts.values()
            )
        ):
            raise TargetSpeedManifestError("execution manifest reward artifacts are invalid")
        artifact_names = tuple(self.reward_artifacts)
        if any(
            type(name) is not str or not name or name.startswith("_") or "__" in name
            for name in artifact_names
        ):
            raise TargetSpeedManifestError("execution manifest artifact names are invalid")
        if type(self.source_bindings) is not tuple or any(
            not isinstance(binding, FrozenSourceBindingV1) for binding in self.source_bindings
        ):
            raise TargetSpeedManifestError("execution manifest source bindings are invalid")
        roles = [binding.role for binding in self.source_bindings]
        if not roles or len(roles) != len(set(roles)):
            raise TargetSpeedManifestError(
                "execution manifest source roles are empty or duplicated"
            )
        required_roles = {
            "study_config",
            "protocol",
            "decision_rule",
            *(f"candidate_source:{name}" for name in artifact_names),
        }
        if not required_roles.issubset(roles):
            raise TargetSpeedManifestError("execution manifest source closure is incomplete")
        bindings_by_role = {binding.role: binding for binding in self.source_bindings}
        for name, artifact in self.reward_artifacts.items():
            if (
                bindings_by_role[f"candidate_source:{name}"].sha256
                != artifact.candidate_source_sha256
            ):
                raise TargetSpeedManifestError(
                    f"candidate source binding for {name} differs from its artifact identity"
                )
        object.__setattr__(
            self,
            "reward_artifacts",
            MappingProxyType(dict(sorted(self.reward_artifacts.items()))),
        )

    def verify(
        self,
        *,
        design: TargetSpeedStudyDesignV1,
        runtime: TargetSpeedRuntimeFingerprintV1,
    ) -> None:
        if not isinstance(design, TargetSpeedStudyDesignV1) or not isinstance(
            runtime, TargetSpeedRuntimeFingerprintV1
        ):
            raise TargetSpeedManifestError(
                "manifest verification requires typed design and runtime"
            )
        if design.sha256 != self.design_sha256:
            raise TargetSpeedManifestError("study design drifted after manifest freeze")
        if runtime.sha256 != self.runtime_sha256:
            raise TargetSpeedManifestError("runtime drifted after manifest freeze")
        for binding in self.source_bindings:
            binding.verify()
        bindings_by_role = {binding.role: binding for binding in self.source_bindings}
        if bindings_by_role["study_config"].sha256 != design.file_sha256:
            raise TargetSpeedManifestError("study config file differs from its design receipt")
        for name, artifact in self.reward_artifacts.items():
            if (
                artifact.target_speed_m_s != 1.0
                or artifact.compositor_source_sha256 != runtime.source_hashes["compositor"]
                or artifact.gymnasium_source_sha256 != runtime.gymnasium_source_sha256
                or artifact.humanoid_xml_sha256 != runtime.model_sha256
                or artifact.dependency_hashes.get("uv.lock") != runtime.dependency_lock_sha256
                or bindings_by_role[f"candidate_source:{name}"].sha256
                != artifact.candidate_source_sha256
            ):
                raise TargetSpeedManifestError(
                    f"reward artifact {name} differs from the frozen first-cycle runtime"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": self.status,
            "design_sha256": self.design_sha256,
            "runtime_sha256": self.runtime_sha256,
            "reward_artifacts": {
                name: value.to_dict() for name, value in self.reward_artifacts.items()
            },
            "source_bindings": [value.to_dict() for value in self.source_bindings],
            "training_seeds": list(self.training_seeds),
            "evaluation_seeds": list(self.evaluation_seeds),
            "automatic_authorization": self.automatic_authorization,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


def validate_complete_result_schedule_v1(
    observed: Mapping[int, Sequence[int]],
) -> None:
    if set(observed) != set(TRAINING_SEEDS):
        raise TargetSpeedManifestError("result set is missing a predetermined training seed")
    for seed in TRAINING_SEEDS:
        if tuple(observed[seed]) != EVALUATION_SEEDS:
            raise TargetSpeedManifestError(
                f"result set for training seed {seed} is partial, duplicated, or reordered"
            )


__all__ = [
    "DESIGN_STATUS",
    "EVALUATION_SEEDS",
    "EVIDENCE_LABEL",
    "EXPECTED_PACKAGE_VERSIONS",
    "FAMILY_ID",
    "FROZEN_STUDY_CONFIG_V1",
    "NO_LEARNING_SMOKE_SEED",
    "NO_LEARNING_SMOKE_STEPS",
    "TRAINING_SEEDS",
    "FrozenSourceBindingV1",
    "NoLearningInstrumentationSmokeV1",
    "RewardTargetSpeedExecutionManifestV1",
    "TargetSpeedManifestError",
    "TargetSpeedRuntimeFingerprintV1",
    "TargetSpeedStudyDesignV1",
    "inspect_target_speed_runtime_v1",
    "load_target_speed_study_design",
    "run_no_learning_instrumentation_smoke",
    "validate_complete_result_schedule_v1",
]
