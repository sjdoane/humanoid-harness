"""Fail-closed execution paths for fixed-reference experiment 001.

The functions in this module separate disposable resource calibration, reviewed
manifest creation, one-seed training, and complete deterministic evaluation.
They never promote a checkpoint or turn measurements into a behavioral claim.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from oracle_composition.tracking import (
    make_static_stand_reference,
    tracking_state,
    validate_humanoid_actuator_abi,
)

from .artifact_io import publish_json_without_overwrite
from .fixed_reference import (
    CANONICAL_EXECUTION_CALLABLE_AUTHORITY,
    NONAUTHORITATIVE_EXECUTION_CALLABLE_AUTHORITY,
    EvidencePurpose,
    ExperimentContractError,
    FixedReferenceStudyDesign,
    FrozenExecutionManifest,
    ResourceCalibrationReceipt,
    RuntimeFingerprint,
    assert_runtime_matches,
    checkpoint_receipt,
    load_study_design,
    read_bounded_json_object,
    sha256_file,
)
from .fixed_reference_runner import inspect_runtime, make_static_tracking_env, validate_before_run
from .protected_evaluator import (
    FLOOR_GEOM_NAME,
    PERMITTED_STATIC_STAND_FLOOR_GEOMS,
    STATION_KEEPING_ORIGIN_SOURCE,
    ContactFact,
    EpisodeAccumulator,
    evaluate_tracking_step,
)
from .protected_runtime import (
    direct_contact_facts as _shared_direct_contact_facts,
)
from .protected_runtime import (
    finite_direct_vector as _shared_finite_direct_vector,
)
from .protected_runtime import (
    initial_protected_history as _shared_initial_protected_history,
)
from .protected_runtime import (
    normalized_policy_action as _shared_normalized_policy_action,
)
from .protected_runtime import (
    protected_step_inputs as _shared_protected_step_inputs,
)
from .squashed_policy import (
    ACTION_BOUNDARY_MARGIN,
    ACTION_LIKELIHOOD_AUDIT_ID,
    LOG_PROB_RECOMPUTE_ATOL,
    POLICY_ID,
    RolloutActionLikelihoodAudit,
    SquashedGaussianActorCriticPolicy,
    assert_squashed_policy_contract,
)

_SOURCE_CHECKOUT_CANDIDATE = Path(__file__).resolve().parents[3]
REPOSITORY_ROOT = (
    _SOURCE_CHECKOUT_CANDIDATE
    if (_SOURCE_CHECKOUT_CANDIDATE / "uv.lock").is_file()
    else Path.cwd().resolve()
)
DEFAULT_RUNS_DIR = REPOSITORY_ROOT / "experiments" / "001_humanoid_fixed_reference" / "runs"
CHECKPOINT_FILENAME = "checkpoint.zip"
CHECKPOINT_RECEIPT_FILENAME = "checkpoint_receipt.json"
EVALUATION_RECEIPT_FILENAME = "evaluation_receipt.json"
CALIBRATION_RECEIPT_FILENAME = "calibration_receipt.json"
PROTECTED_METRICS_STATE_SOURCE = (
    "direct_mujoco_state+seeded_reset_root_origin+all_physics_substep_contacts+policy_action/v4"
)
MAX_EXECUTION_ARTIFACT_BYTES = 4 * 1024 * 1024


def _execution_callable_authority(
    *callable_pairs: tuple[object, object],
) -> str:
    """Classify injected execution boundaries without granting them production authority."""

    if all(observed is canonical for observed, canonical in callable_pairs):
        return CANONICAL_EXECUTION_CALLABLE_AUTHORITY
    return NONAUTHORITATIVE_EXECUTION_CALLABLE_AUTHORITY


def _require_manifest_callable_authority(
    manifest: FrozenExecutionManifest,
    observed_authority: str,
) -> None:
    if manifest.execution_callable_authority != observed_authority:
        raise ExperimentContractError(
            "execution callable authority does not match the reviewed manifest"
        )


def _runtime_semantics_receipt_fields(
    runtime: RuntimeFingerprint,
) -> dict[str, object]:
    """Expose compiled actuator semantics and package-source closure in receipts."""

    return {
        "actuator_gear_by_joint": list(runtime.actuator_gear_by_joint),
        "generalized_actuator_torque_capacity_n_m": list(
            runtime.generalized_actuator_torque_capacity_n_m
        ),
        "source_tree_sha256": runtime.source_tree_sha256,
    }


def _audited_learn(
    model: object,
    *,
    total_timesteps: int,
    rollout_size: int,
) -> dict[str, object]:
    """Run learning with one mandatory pre-update action-likelihood audit per rollout."""

    if total_timesteps < 1 or rollout_size < 1 or total_timesteps % rollout_size:
        raise ExperimentContractError("audited learning requires an exact positive rollout count")
    if hasattr(model, "policy"):
        assert_squashed_policy_contract(model.policy)
    audit = RolloutActionLikelihoodAudit()
    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=True,
        progress_bar=False,
        callback=audit,
    )
    try:
        return audit.completion_receipt(expected_rollouts=total_timesteps // rollout_size)
    except (RuntimeError, ValueError) as exc:
        raise ExperimentContractError(f"action-likelihood audit incomplete: {exc}") from exc


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
        raise ExperimentContractError(f"receipt is not canonical JSON: {exc}") from exc


def _pretty_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExperimentContractError(f"receipt is not finite JSON: {exc}") from exc


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _read_json_object(path: Path, *, artifact: str) -> dict[str, Any]:
    return read_bounded_json_object(
        path,
        maximum_bytes=MAX_EXECUTION_ARTIFACT_BYTES,
        artifact=artifact,
    )


def load_execution_manifest(path: Path) -> FrozenExecutionManifest:
    """Load a strict frozen manifest without filling or repairing fields."""

    return FrozenExecutionManifest.from_dict(_read_json_object(path, artifact="execution manifest"))


def _write_complete_file(path: Path, payload: object) -> None:
    """Write and sync a file inside an unpublished staging directory."""

    data = _pretty_json(payload)
    try:
        with path.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        raise ExperimentContractError(f"cannot write receipt {path}: {exc}") from exc


def emit_json_without_overwrite(path: Path, payload: object) -> None:
    """Publish one descriptor-bound JSON file and refuse replacement."""

    publish_json_without_overwrite(path, payload)


def _validated_runs_dir(runs_dir: Path) -> Path:
    resolved = runs_dir.resolve()
    if resolved.name != "runs":
        raise ExperimentContractError("generated outputs must be placed in a directory named runs")
    resolved.mkdir(parents=True, exist_ok=True)
    if not resolved.is_dir():
        raise ExperimentContractError(f"runs path is not a directory: {resolved}")
    return resolved


@contextmanager
def _atomic_run_directory(
    runs_dir: Path,
    *,
    final_name: str,
) -> Iterator[tuple[Path, Path]]:
    """Publish a complete run directory with one same-filesystem rename."""

    parent = _validated_runs_dir(runs_dir)
    final = parent / final_name
    if final.exists():
        raise ExperimentContractError(f"refusing to overwrite existing run: {final}")
    staging = Path(tempfile.mkdtemp(prefix=f".{final_name}.", suffix=".pending", dir=parent))
    published = False
    try:
        yield staging, final
        if final.exists():
            raise ExperimentContractError(f"refusing to overwrite existing run: {final}")
        staging.rename(final)
        published = True
    finally:
        if not published and staging.is_dir():
            shutil.rmtree(staging)


def _make_vector_env(design: FixedReferenceStudyDesign, *, seed: int) -> object:
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("install the train extra before running PPO") from exc

    constructors = [lambda design=design: make_static_tracking_env(design)] * design.ppo.n_envs
    environment = DummyVecEnv(constructors)
    environment.seed(seed)
    return environment


def _make_ppo(design: FixedReferenceStudyDesign, environment: object, *, seed: int) -> object:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("install the train extra before running PPO") from exc

    hidden = list(design.ppo.policy_hidden_layers)
    model = PPO(
        SquashedGaussianActorCriticPolicy,
        environment,
        learning_rate=design.ppo.learning_rate,
        n_steps=design.ppo.n_steps,
        batch_size=design.ppo.batch_size,
        n_epochs=design.ppo.n_epochs,
        gamma=design.ppo.gamma,
        gae_lambda=design.ppo.gae_lambda,
        clip_range=design.ppo.clip_range,
        normalize_advantage=True,
        ent_coef=design.ppo.ent_coef,
        vf_coef=design.ppo.vf_coef,
        max_grad_norm=design.ppo.max_grad_norm,
        use_sde=False,
        sde_sample_freq=-1,
        rollout_buffer_class=None,
        target_kl=None,
        stats_window_size=100,
        tensorboard_log=None,
        policy_kwargs={"net_arch": {"pi": hidden, "vf": hidden}},
        verbose=0,
        seed=seed,
        device=design.ppo.device,
    )
    assert_squashed_policy_contract(model.policy)
    return model


def _load_ppo(checkpoint_path: Path, environment: object, *, device: str) -> object:
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("install the train extra before loading PPO") from exc
    model = PPO.load(
        checkpoint_path,
        env=environment,
        device=device,
        print_system_info=False,
    )
    assert_squashed_policy_contract(model.policy)
    return model


def _validated_evaluation_environment(
    environment: object,
    *,
    expected_reference_sha256: str,
    abi_factory: Callable[[object], object],
) -> object:
    """Resolve the audited wrapper attribute and physical Humanoid ABI."""

    get_wrapper_attr = getattr(environment, "get_wrapper_attr", None)
    if callable(get_wrapper_attr):
        try:
            observed_reference = get_wrapper_attr("reference_content_sha256")
        except AttributeError as exc:
            raise ExperimentContractError(
                "evaluation environment does not expose a reference identity"
            ) from exc
    else:
        observed_reference = getattr(environment, "reference_content_sha256", None)
    if observed_reference != expected_reference_sha256:
        raise ExperimentContractError("evaluation environment installed a different reference")

    # The outer normalized-action wrapper and reference tracker deliberately
    # alter their public spaces. ABI validation must inspect the physical base
    # Humanoid interface, not either authored wrapper.
    physical_environment = getattr(environment, "unwrapped", environment)
    return abi_factory(physical_environment)


def _finite_direct_vector(value: object, *, width: int, field: str) -> np.ndarray:
    """Validate a direct runtime vector before it reaches the evaluator."""

    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExperimentContractError(f"{field} is not a numeric vector") from exc
    if vector.shape != (width,) or not np.isfinite(vector).all():
        raise ExperimentContractError(f"{field} must be a finite ({width},) vector")
    return vector.copy()


def _normalized_policy_action(value: object) -> np.ndarray:
    action = _finite_direct_vector(value, width=17, field="normalized policy action")
    if np.any(action < -1.0) or np.any(action > 1.0):
        raise ExperimentContractError("normalized policy action must be in [-1, 1]")
    return action


def _initial_protected_history(environment: object, abi: object) -> tuple[np.ndarray, np.ndarray]:
    """Read reset-time control and acceleration for first-difference metrics."""

    physical = getattr(environment, "unwrapped", environment)
    data = getattr(physical, "data", None)
    action_space = getattr(physical, "action_space", None)
    qvel_indices = tuple(getattr(abi, "qvel_indices", ()))
    if data is None or action_space is None or len(qvel_indices) != 17:
        raise ExperimentContractError("physical reset history is unavailable")
    low = _finite_direct_vector(action_space.low, width=17, field="physical action lower bound")
    high = _finite_direct_vector(
        action_space.high,
        width=17,
        field="physical action upper bound",
    )
    if np.any(high <= low):
        raise ExperimentContractError("physical action bounds must be strictly ordered")
    control = _finite_direct_vector(data.ctrl, width=17, field="reset physical control")
    normalized_control = 2.0 * ((control - low) / (high - low)) - 1.0
    normalized_control = _normalized_policy_action(normalized_control)
    if not np.array_equal(normalized_control, np.zeros(17, dtype=np.float64)):
        raise ExperimentContractError("reset normalized physical control must be exactly zero")
    acceleration = _finite_direct_vector(
        np.asarray(data.qacc)[list(qvel_indices)],
        width=17,
        field="reset joint acceleration",
    )
    return normalized_control, acceleration


def _direct_contact_facts(environment: object) -> tuple[ContactFact, ...]:
    """Read every contact captured after every physics substep."""

    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("MuJoCo is required for protected contact measurements") from exc

    physical = getattr(environment, "unwrapped", environment)
    model = getattr(physical, "model", None)
    samples = getattr(physical, "last_control_step_contact_samples", None)
    captured_substeps = getattr(physical, "last_control_step_substeps", None)
    frame_skip = getattr(physical, "frame_skip", None)
    if model is None or not isinstance(samples, tuple):
        raise ExperimentContractError("physical MuJoCo contact state is unavailable")
    if (
        not isinstance(frame_skip, int)
        or isinstance(frame_skip, bool)
        or captured_substeps != frame_skip
    ):
        raise ExperimentContractError("contact capture did not cover every MuJoCo physics substep")
    indices_by_substep: dict[int, list[int]] = {index: [] for index in range(frame_skip)}
    contacts: list[ContactFact] = []
    for flattened_index, sample in enumerate(samples):
        substep_index = getattr(sample, "physics_substep_index", None)
        contact_index = getattr(sample, "contact_index_within_substep", None)
        if (
            not isinstance(substep_index, int)
            or isinstance(substep_index, bool)
            or substep_index not in indices_by_substep
            or not isinstance(contact_index, int)
            or isinstance(contact_index, bool)
            or contact_index < 0
        ):
            raise ExperimentContractError("substep contact index is invalid")
        indices_by_substep[substep_index].append(contact_index)
        geom1_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            int(getattr(sample, "geom1_id", -1)),
        )
        geom2_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            int(getattr(sample, "geom2_id", -1)),
        )
        if not geom1_name or not geom2_name:
            raise ExperimentContractError("active contact contains an unnamed geometry")
        normal_force = getattr(sample, "normal_force_n", None)
        if (
            isinstance(normal_force, bool)
            or not isinstance(normal_force, (int, float))
            or not math.isfinite(float(normal_force))
            or float(normal_force) < 0.0
        ):
            raise ExperimentContractError("substep contact normal force is invalid")
        involves_floor = FLOOR_GEOM_NAME in (geom1_name, geom2_name)
        if involves_floor:
            other_geom = geom2_name if geom1_name == FLOOR_GEOM_NAME else geom1_name
            forbidden = other_geom not in PERMITTED_STATIC_STAND_FLOOR_GEOMS
        else:
            forbidden = False
        contacts.append(
            ContactFact(
                contact_index=flattened_index,
                geom1_name=geom1_name,
                geom2_name=geom2_name,
                involves_floor=involves_floor,
                forbidden_floor_contact=forbidden,
                normal_force_n=float(normal_force),
            )
        )
    for substep_index, contact_indices in indices_by_substep.items():
        if contact_indices != list(range(len(contact_indices))):
            raise ExperimentContractError(
                f"contact capture is incomplete at physics substep {substep_index}"
            )
    return tuple(contacts)


def _protected_step_inputs(
    environment: object,
    abi: object,
    *,
    normalized_action: object,
    previous_normalized_action: object,
    previous_joint_acceleration: object,
) -> dict[str, object]:
    """Read reward-independent post-step control, effort, jerk, and contacts."""

    physical = getattr(environment, "unwrapped", environment)
    data = getattr(physical, "data", None)
    qvel_indices = tuple(getattr(abi, "qvel_indices", ()))
    control_period = getattr(abi, "control_period_seconds", None)
    if data is None or len(qvel_indices) != 17:
        raise ExperimentContractError("physical protected step state is unavailable")
    action = _normalized_policy_action(normalized_action)
    previous_action = _normalized_policy_action(previous_normalized_action)
    torque = _finite_direct_vector(
        np.asarray(data.qfrc_actuator)[list(qvel_indices)],
        width=17,
        field="post-transmission generalized actuator torque",
    )
    torque_capacity = _finite_direct_vector(
        getattr(abi, "generalized_actuator_torque_capacity_n_m", None),
        width=17,
        field="generalized actuator torque capacity",
    )
    if np.any(torque_capacity <= 0.0):
        raise ExperimentContractError("generalized actuator torque capacity must be positive")
    acceleration = _finite_direct_vector(
        np.asarray(data.qacc)[list(qvel_indices)],
        width=17,
        field="joint acceleration",
    )
    previous_acceleration = _finite_direct_vector(
        previous_joint_acceleration,
        width=17,
        field="previous joint acceleration",
    )
    if (
        isinstance(control_period, bool)
        or not isinstance(control_period, (int, float))
        or not math.isfinite(float(control_period))
        or float(control_period) <= 0.0
    ):
        raise ExperimentContractError("ABI control period must be finite and positive")
    return {
        "normalized_policy_action": action,
        "previous_normalized_policy_action": previous_action,
        "generalized_actuator_torque_n_m": torque,
        "generalized_actuator_torque_capacity_n_m": torque_capacity,
        "joint_accelerations_rad_s2": acceleration,
        "previous_joint_accelerations_rad_s2": previous_acceleration,
        "contact_facts": _direct_contact_facts(environment),
        "control_period_seconds": float(control_period),
    }


# Both experiment runners consume one protected runtime-reader implementation.
_finite_direct_vector = _shared_finite_direct_vector
_normalized_policy_action = _shared_normalized_policy_action
_initial_protected_history = _shared_initial_protected_history
_direct_contact_facts = _shared_direct_contact_facts
_protected_step_inputs = _shared_protected_step_inputs


_CALIBRATION_RECEIPT_KEYS = {
    "schema_version",
    "evidence_purpose",
    "completion_status",
    "design_status",
    "design_sha256",
    "design_file_sha256",
    "study_criteria_semantic_sha256",
    "study_criteria_file_sha256",
    "runtime_sha256",
    "execution_callable_authority",
    "actuator_gear_by_joint",
    "generalized_actuator_torque_capacity_n_m",
    "source_tree_sha256",
    "calibration_seed",
    "environment_steps",
    "wall_seconds",
    "environment_steps_per_second",
    "peak_resident_bytes",
    "memory_measurement",
    "action_likelihood_audit",
    "model_disposition",
    "checkpoint_emitted",
    "eligible_for_behavioral_evaluation",
    "automatic_promotion",
    "behavioral_claim",
}


def _matched_calibration_lineage(
    *,
    locked_design: FixedReferenceStudyDesign,
    calibrated_design_path: Path,
    calibration_receipt_path: Path,
    current_runtime: RuntimeFingerprint,
    execution_callable_authority: str,
) -> dict[str, str]:
    """Reverify exact disposable calibration evidence before manifest review."""

    calibrated_design_path = calibrated_design_path.resolve()
    calibration_receipt_path = calibration_receipt_path.resolve()
    calibrated_design = load_study_design(calibrated_design_path)
    if calibrated_design.design_status != "proposed_pre_calibration":
        raise ExperimentContractError("calibrated design must have proposed_pre_calibration status")
    if locked_design.design_status != "locked_pre_behavioral":
        raise ExperimentContractError("manifest design must be locked_pre_behavioral")
    if calibrated_design.design_version == locked_design.design_version:
        raise ExperimentContractError(
            "locked design must use a new design_version after calibration"
        )
    calibrated_payload = calibrated_design.to_dict()
    locked_payload = locked_design.to_dict()
    for field in ("design_version", "design_status"):
        calibrated_payload.pop(field)
        locked_payload.pop(field)
    if calibrated_payload != locked_payload:
        raise ExperimentContractError(
            "locked design differs from the calibrated design beyond version/status"
        )

    receipt = _read_json_object(
        calibration_receipt_path,
        artifact="calibration receipt",
    )
    missing = sorted(_CALIBRATION_RECEIPT_KEYS - set(receipt))
    extra = sorted(set(receipt) - _CALIBRATION_RECEIPT_KEYS)
    if missing or extra:
        raise ExperimentContractError(
            f"calibration receipt keys mismatch: missing={missing!r}, extra={extra!r}"
        )
    if not isinstance(receipt.get("schema_version"), int) or isinstance(
        receipt.get("schema_version"), bool
    ):
        raise ExperimentContractError("calibration receipt schema_version must be integer 1")
    exact = {
        "schema_version": 1,
        "evidence_purpose": EvidencePurpose.RESOURCE_CALIBRATION.value,
        "completion_status": "complete",
        "design_status": "proposed_pre_calibration",
        "design_sha256": calibrated_design.sha256,
        "design_file_sha256": sha256_file(calibrated_design_path),
        "study_criteria_semantic_sha256": (calibrated_design.study_criteria_semantic_sha256),
        "study_criteria_file_sha256": calibrated_design.study_criteria_file_sha256,
        "runtime_sha256": current_runtime.sha256,
        "execution_callable_authority": execution_callable_authority,
        **_runtime_semantics_receipt_fields(current_runtime),
        "memory_measurement": "not_measured",
        "model_disposition": "discarded_unserialized",
    }
    mismatches = [key for key, expected in exact.items() if receipt.get(key) != expected]
    if mismatches:
        raise ExperimentContractError(
            "calibration receipt does not match the calibrated design/current runtime: "
            + ", ".join(mismatches)
        )
    for field in (
        "checkpoint_emitted",
        "eligible_for_behavioral_evaluation",
        "automatic_promotion",
    ):
        if receipt.get(field) is not False:
            raise ExperimentContractError(f"calibration receipt {field} must be false")
    if receipt.get("behavioral_claim") is not None:
        raise ExperimentContractError("calibration receipt behavioral_claim must be null")
    if receipt.get("peak_resident_bytes") is not None:
        raise ExperimentContractError(
            "calibration receipt peak_resident_bytes must be null when memory is not measured"
        )

    calibration_seed = receipt.get("calibration_seed")
    if (
        not isinstance(calibration_seed, int)
        or isinstance(calibration_seed, bool)
        or calibration_seed < 0
    ):
        raise ExperimentContractError(
            "calibration receipt calibration_seed must be a non-negative integer"
        )
    environment_steps = receipt.get("environment_steps")
    rollout_size = calibrated_design.ppo.n_envs * calibrated_design.ppo.n_steps
    if (
        not isinstance(environment_steps, int)
        or isinstance(environment_steps, bool)
        or environment_steps < 1
        or environment_steps % rollout_size
    ):
        raise ExperimentContractError(
            "calibration receipt environment_steps must be a positive exact rollout multiple"
        )
    wall_seconds = receipt.get("wall_seconds")
    measured_rate = receipt.get("environment_steps_per_second")
    for field, value in (
        ("wall_seconds", wall_seconds),
        ("environment_steps_per_second", measured_rate),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ExperimentContractError(
                f"calibration receipt {field} must be finite and positive"
            )
    expected_rate = environment_steps / float(wall_seconds)
    if not math.isclose(float(measured_rate), expected_rate, rel_tol=1e-12, abs_tol=0.0):
        raise ExperimentContractError(
            "calibration receipt environment_steps_per_second is inconsistent"
        )
    _validate_action_likelihood_audit_receipt(
        receipt.get("action_likelihood_audit"),
        expected_rollouts=environment_steps // rollout_size,
        artifact="calibration receipt",
    )
    return {
        "calibrated_design_sha256": calibrated_design.sha256,
        "calibrated_design_file_sha256": sha256_file(calibrated_design_path),
        "calibration_receipt_sha256": _json_sha256(receipt),
        "calibration_receipt_file_sha256": sha256_file(calibration_receipt_path),
    }


def emit_manifest_candidate(
    *,
    design_path: Path,
    calibrated_design_path: Path,
    calibration_receipt_path: Path,
    output_path: Path,
    runtime_inspector: Callable[[FixedReferenceStudyDesign], RuntimeFingerprint] = inspect_runtime,
) -> dict[str, object]:
    """Inspect and publish a non-authorizing candidate for human review."""

    design_path = design_path.resolve()
    design = load_study_design(design_path)
    design.assert_locked_for_behavior()
    execution_callable_authority = _execution_callable_authority(
        (runtime_inspector, inspect_runtime),
    )
    runtime = runtime_inspector(design)
    calibration_lineage = _matched_calibration_lineage(
        locked_design=design,
        calibrated_design_path=calibrated_design_path,
        calibration_receipt_path=calibration_receipt_path,
        current_runtime=runtime,
        execution_callable_authority=execution_callable_authority,
    )
    candidate = {
        "schema_version": 1,
        "artifact_kind": "frozen_execution_manifest_candidate",
        "authorization_status": (
            "candidate_for_human_review"
            if execution_callable_authority == CANONICAL_EXECUTION_CALLABLE_AUTHORITY
            else "test_only_non_authoritative"
        ),
        "execution_callable_authority": execution_callable_authority,
        "design_sha256": design.sha256,
        "design_file_sha256": sha256_file(design_path),
        **calibration_lineage,
        "study_criteria_semantic_sha256": design.study_criteria_semantic_sha256,
        "study_criteria_file_sha256": design.study_criteria_file_sha256,
        "runtime": runtime.to_dict(),
        "automatic_authorization": False,
    }
    emit_json_without_overwrite(output_path, candidate)
    return {
        "artifact": "candidate_frozen_execution_manifest",
        "design_sha256": design.sha256,
        "design_file_sha256": sha256_file(design_path),
        **calibration_lineage,
        "candidate_semantic_sha256": _json_sha256(candidate),
        "candidate_file_sha256": sha256_file(output_path.resolve()),
        "output_path": str(output_path.resolve()),
        "execution_callable_authority": execution_callable_authority,
        "automatic_authorization": False,
    }


_MANIFEST_CANDIDATE_KEYS = {
    "schema_version",
    "artifact_kind",
    "authorization_status",
    "execution_callable_authority",
    "design_sha256",
    "design_file_sha256",
    "calibrated_design_sha256",
    "calibrated_design_file_sha256",
    "calibration_receipt_sha256",
    "calibration_receipt_file_sha256",
    "study_criteria_semantic_sha256",
    "study_criteria_file_sha256",
    "runtime",
    "automatic_authorization",
}


def finalize_reviewed_manifest(
    *,
    design_path: Path,
    calibrated_design_path: Path,
    calibration_receipt_path: Path,
    candidate_path: Path,
    output_path: Path,
    reviewer_id: str,
    review_note: str,
    runtime_inspector: Callable[[FixedReferenceStudyDesign], RuntimeFingerprint] = inspect_runtime,
) -> dict[str, object]:
    """Publish a frozen manifest only after an explicit human review record."""

    design_path = design_path.resolve()
    candidate_path = candidate_path.resolve()
    design = load_study_design(design_path)
    design.assert_locked_for_behavior()
    execution_callable_authority = _execution_callable_authority(
        (runtime_inspector, inspect_runtime),
    )
    current_runtime = runtime_inspector(design)
    calibration_lineage = _matched_calibration_lineage(
        locked_design=design,
        calibrated_design_path=calibrated_design_path,
        calibration_receipt_path=calibration_receipt_path,
        current_runtime=current_runtime,
        execution_callable_authority=execution_callable_authority,
    )
    candidate = _read_json_object(candidate_path, artifact="manifest candidate")
    missing = sorted(_MANIFEST_CANDIDATE_KEYS - set(candidate))
    extra = sorted(set(candidate) - _MANIFEST_CANDIDATE_KEYS)
    if missing or extra:
        raise ExperimentContractError(
            f"manifest candidate keys mismatch: missing={missing!r}, extra={extra!r}"
        )
    if not isinstance(candidate.get("schema_version"), int) or isinstance(
        candidate.get("schema_version"), bool
    ):
        raise ExperimentContractError("manifest candidate schema_version must be integer 1")
    expected = {
        "schema_version": 1,
        "artifact_kind": "frozen_execution_manifest_candidate",
        "authorization_status": (
            "candidate_for_human_review"
            if execution_callable_authority == CANONICAL_EXECUTION_CALLABLE_AUTHORITY
            else "test_only_non_authoritative"
        ),
        "execution_callable_authority": execution_callable_authority,
        "design_sha256": design.sha256,
        "design_file_sha256": sha256_file(design_path),
        **calibration_lineage,
        "study_criteria_semantic_sha256": design.study_criteria_semantic_sha256,
        "study_criteria_file_sha256": design.study_criteria_file_sha256,
        "automatic_authorization": False,
    }
    mismatches = [key for key, value in expected.items() if candidate.get(key) != value]
    if mismatches:
        raise ExperimentContractError(
            "manifest candidate does not match the locked design: " + ", ".join(mismatches)
        )
    if not isinstance(candidate.get("runtime"), dict):
        raise ExperimentContractError("manifest candidate runtime must be an object")
    runtime = RuntimeFingerprint.from_dict(candidate["runtime"])
    assert_runtime_matches(runtime, current_runtime)
    manifest = FrozenExecutionManifest(
        schema_version=1,
        status=(
            "frozen"
            if execution_callable_authority == CANONICAL_EXECUTION_CALLABLE_AUTHORITY
            else "test_only_non_authoritative"
        ),
        execution_callable_authority=execution_callable_authority,
        design_sha256=design.sha256,
        calibrated_design_sha256=calibration_lineage["calibrated_design_sha256"],
        calibrated_design_file_sha256=calibration_lineage["calibrated_design_file_sha256"],
        calibration_receipt_sha256=calibration_lineage["calibration_receipt_sha256"],
        calibration_receipt_file_sha256=calibration_lineage["calibration_receipt_file_sha256"],
        study_criteria_semantic_sha256=design.study_criteria_semantic_sha256,
        study_criteria_file_sha256=design.study_criteria_file_sha256,
        runtime=runtime,
        candidate_file_sha256=sha256_file(candidate_path),
        reviewer_id=reviewer_id,
        review_note=review_note,
    )
    manifest.validate(design=design, observed=current_runtime)
    emit_json_without_overwrite(output_path, manifest.to_dict())
    return {
        "artifact": "reviewed_frozen_execution_manifest",
        "design_sha256": design.sha256,
        **calibration_lineage,
        "candidate_file_sha256": manifest.candidate_file_sha256,
        "execution_manifest_sha256": manifest.sha256,
        "execution_manifest_file_sha256": sha256_file(output_path.resolve()),
        "output_path": str(output_path.resolve()),
        "execution_callable_authority": execution_callable_authority,
        "automatic_authorization": False,
    }


def calibrate_resources(
    *,
    design_path: Path,
    calibration_seed: int,
    environment_steps: int | None = None,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    runtime_inspector: Callable[[FixedReferenceStudyDesign], RuntimeFingerprint] = inspect_runtime,
    vector_env_factory: Callable[..., object] = _make_vector_env,
    model_factory: Callable[..., object] = _make_ppo,
) -> dict[str, object]:
    """Measure a proposed PPO workload, discard its model, and publish a receipt."""

    design_path = design_path.resolve()
    design = load_study_design(design_path)
    if design.design_status != "proposed_pre_calibration":
        raise ExperimentContractError(
            "resource calibration requires a proposed_pre_calibration design"
        )
    if (
        not isinstance(calibration_seed, int)
        or isinstance(calibration_seed, bool)
        or calibration_seed < 0
    ):
        raise ExperimentContractError("calibration_seed must be a non-negative integer")
    rollout_steps = design.ppo.n_envs * design.ppo.n_steps
    selected_steps = rollout_steps if environment_steps is None else environment_steps
    if (
        not isinstance(selected_steps, int)
        or isinstance(selected_steps, bool)
        or selected_steps < 1
        or selected_steps % rollout_steps
    ):
        raise ExperimentContractError(
            "calibration environment_steps must be a positive exact multiple of n_envs * n_steps"
        )

    execution_callable_authority = _execution_callable_authority(
        (runtime_inspector, inspect_runtime),
        (vector_env_factory, _make_vector_env),
        (model_factory, _make_ppo),
    )
    runtime = runtime_inspector(design)
    environment: object | None = None
    model: object | None = None
    start = time.perf_counter()
    observed_steps: int | None = None
    try:
        environment = vector_env_factory(design, seed=calibration_seed)
        model = model_factory(design, environment, seed=calibration_seed)
        action_likelihood_audit = _audited_learn(
            model,
            total_timesteps=selected_steps,
            rollout_size=rollout_steps,
        )
        observed_steps = int(model.num_timesteps)
        if observed_steps != selected_steps:
            raise ExperimentContractError(
                f"calibration collected {observed_steps} steps; expected exactly {selected_steps}"
            )
    finally:
        wall_seconds = time.perf_counter() - start
        if environment is not None:
            environment.close()
        model = None
        gc.collect()

    if observed_steps is None or not math.isfinite(wall_seconds) or wall_seconds <= 0.0:
        raise ExperimentContractError("resource calibration did not complete")
    base_receipt = ResourceCalibrationReceipt(
        purpose=EvidencePurpose.RESOURCE_CALIBRATION,
        design_sha256=design.sha256,
        runtime_sha256=runtime.sha256,
        calibration_seed=calibration_seed,
        environment_steps=observed_steps,
        wall_seconds=wall_seconds,
        peak_resident_bytes=None,
    )
    payload = {
        "schema_version": 1,
        "evidence_purpose": base_receipt.purpose.value,
        "completion_status": "complete",
        "design_status": design.design_status,
        "design_sha256": base_receipt.design_sha256,
        "design_file_sha256": sha256_file(design_path),
        "study_criteria_semantic_sha256": design.study_criteria_semantic_sha256,
        "study_criteria_file_sha256": design.study_criteria_file_sha256,
        "runtime_sha256": base_receipt.runtime_sha256,
        "execution_callable_authority": execution_callable_authority,
        **_runtime_semantics_receipt_fields(runtime),
        "calibration_seed": base_receipt.calibration_seed,
        "environment_steps": base_receipt.environment_steps,
        "wall_seconds": base_receipt.wall_seconds,
        "environment_steps_per_second": base_receipt.environment_steps / base_receipt.wall_seconds,
        "peak_resident_bytes": base_receipt.peak_resident_bytes,
        "memory_measurement": "not_measured",
        "action_likelihood_audit": action_likelihood_audit,
        "model_disposition": "discarded_unserialized",
        "checkpoint_emitted": False,
        "eligible_for_behavioral_evaluation": False,
        "automatic_promotion": False,
        "behavioral_claim": None,
    }
    final_name = f"calibration-seed-{calibration_seed:08d}-{design.sha256[:12]}"
    with _atomic_run_directory(runs_dir, final_name=final_name) as (staging, final):
        _write_complete_file(staging / CALIBRATION_RECEIPT_FILENAME, payload)
    return {
        "run_directory": str(final),
        "receipt_path": str(final / CALIBRATION_RECEIPT_FILENAME),
        "receipt_sha256": sha256_file(final / CALIBRATION_RECEIPT_FILENAME),
        **payload,
    }


def _training_receipt_payload(
    *,
    design: FixedReferenceStudyDesign,
    design_path: Path,
    manifest: FrozenExecutionManifest,
    manifest_path: Path,
    runtime: RuntimeFingerprint,
    train_seed: int,
    observed_timesteps: int,
    checkpoint_path: Path,
    action_likelihood_audit: dict[str, object],
    execution_callable_authority: str,
) -> dict[str, object]:
    identity = checkpoint_receipt(
        design=design,
        runtime=runtime,
        train_seed=train_seed,
        observed_timesteps=observed_timesteps,
        checkpoint_path=checkpoint_path,
    )
    return {
        "schema_version": 1,
        "evidence_purpose": identity.purpose.value,
        "completion_status": "complete",
        "claim_ceiling": design.claim_ceiling.value,
        "design_sha256": identity.design_sha256,
        "design_file_sha256": sha256_file(design_path),
        "study_criteria_semantic_sha256": design.study_criteria_semantic_sha256,
        "study_criteria_file_sha256": design.study_criteria_file_sha256,
        "execution_manifest_sha256": manifest.sha256,
        "execution_manifest_file_sha256": sha256_file(manifest_path),
        "runtime_sha256": identity.runtime_sha256,
        "execution_callable_authority": execution_callable_authority,
        **_runtime_semantics_receipt_fields(runtime),
        "train_seed": identity.train_seed,
        "algorithm_id": "stable_baselines3.PPO/v1",
        "policy_id": POLICY_ID,
        "action_transform_id": runtime.action_transform_id,
        "observation_normalizer_id": runtime.observation_normalizer_id,
        "reward_normalizer_id": runtime.reward_normalizer_id,
        "action_likelihood_audit": action_likelihood_audit,
        "checkpoint_rule": identity.checkpoint_rule,
        "requested_timesteps": identity.requested_timesteps,
        "observed_timesteps": identity.observed_timesteps,
        "checkpoint_file": CHECKPOINT_FILENAME,
        "checkpoint_sha256": identity.checkpoint_sha256,
        "checkpoint_size_bytes": identity.checkpoint_size_bytes,
        "checkpoint_eligible_for_declared_evaluation": (
            execution_callable_authority == CANONICAL_EXECUTION_CALLABLE_AUTHORITY
        ),
        "automatic_promotion": False,
        "behavioral_claim": None,
    }


def train_one_seed(
    *,
    design_path: Path,
    manifest_path: Path,
    train_seed: int,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    runtime_validator: Callable[..., RuntimeFingerprint] = validate_before_run,
    vector_env_factory: Callable[..., object] = _make_vector_env,
    model_factory: Callable[..., object] = _make_ppo,
) -> dict[str, object]:
    """Train one predetermined seed and atomically publish its final checkpoint."""

    design_path = design_path.resolve()
    manifest_path = manifest_path.resolve()
    design = load_study_design(design_path)
    manifest = load_execution_manifest(manifest_path)
    execution_callable_authority = _execution_callable_authority(
        (runtime_validator, validate_before_run),
        (vector_env_factory, _make_vector_env),
        (model_factory, _make_ppo),
    )
    _require_manifest_callable_authority(manifest, execution_callable_authority)
    if (
        not isinstance(train_seed, int)
        or isinstance(train_seed, bool)
        or train_seed not in design.train_seeds
    ):
        raise ExperimentContractError("train_seed is not in the predetermined schedule")

    # This is intentionally the last operation before constructing the training
    # environment. It re-inspects source, dependencies, platform, spaces, and
    # simulator facts rather than trusting the manifest's stored fingerprint.
    runtime = runtime_validator(design=design, manifest=manifest)
    final_name = f"train-seed-{train_seed:08d}-{design.sha256[:12]}-{manifest.sha256[:12]}"
    environment: object | None = None
    model: object | None = None
    with _atomic_run_directory(runs_dir, final_name=final_name) as (staging, final):
        try:
            environment = vector_env_factory(design, seed=train_seed)
            model = model_factory(design, environment, seed=train_seed)
            action_likelihood_audit = _audited_learn(
                model,
                total_timesteps=design.ppo.total_timesteps_per_seed,
                rollout_size=design.ppo.n_envs * design.ppo.n_steps,
            )
            observed_timesteps = int(model.num_timesteps)
            if observed_timesteps != design.ppo.total_timesteps_per_seed:
                raise ExperimentContractError(
                    f"training collected {observed_timesteps} steps; expected exactly "
                    f"{design.ppo.total_timesteps_per_seed}"
                )
            checkpoint_path = staging / CHECKPOINT_FILENAME
            model.save(checkpoint_path)
            if not checkpoint_path.is_file():
                raise ExperimentContractError("SB3 did not emit the declared final checkpoint")
            payload = _training_receipt_payload(
                design=design,
                design_path=design_path,
                manifest=manifest,
                manifest_path=manifest_path,
                runtime=runtime,
                train_seed=train_seed,
                observed_timesteps=observed_timesteps,
                checkpoint_path=checkpoint_path,
                action_likelihood_audit=action_likelihood_audit,
                execution_callable_authority=execution_callable_authority,
            )
            _write_complete_file(staging / CHECKPOINT_RECEIPT_FILENAME, payload)
        finally:
            if environment is not None:
                environment.close()
            model = None
            gc.collect()

    receipt_path = final / CHECKPOINT_RECEIPT_FILENAME
    return {
        "run_directory": str(final),
        "checkpoint_path": str(final / CHECKPOINT_FILENAME),
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        **payload,
    }


_TRAINING_RECEIPT_KEYS = {
    "schema_version",
    "evidence_purpose",
    "completion_status",
    "claim_ceiling",
    "design_sha256",
    "design_file_sha256",
    "study_criteria_semantic_sha256",
    "study_criteria_file_sha256",
    "execution_manifest_sha256",
    "execution_manifest_file_sha256",
    "runtime_sha256",
    "execution_callable_authority",
    "actuator_gear_by_joint",
    "generalized_actuator_torque_capacity_n_m",
    "source_tree_sha256",
    "train_seed",
    "algorithm_id",
    "policy_id",
    "action_transform_id",
    "observation_normalizer_id",
    "reward_normalizer_id",
    "action_likelihood_audit",
    "checkpoint_rule",
    "requested_timesteps",
    "observed_timesteps",
    "checkpoint_file",
    "checkpoint_sha256",
    "checkpoint_size_bytes",
    "checkpoint_eligible_for_declared_evaluation",
    "automatic_promotion",
    "behavioral_claim",
}

_ACTION_LIKELIHOOD_AUDIT_KEYS = {
    "audit_id",
    "passed",
    "expected_rollouts",
    "audited_rollouts",
    "action_boundary_margin",
    "max_abs_stored_action",
    "log_probability_atol",
    "max_log_probability_abs_error",
}


def _validate_action_likelihood_audit_receipt(
    value: object,
    *,
    expected_rollouts: int,
    artifact: str = "checkpoint",
) -> None:
    if not isinstance(value, dict):
        raise ExperimentContractError(f"{artifact} action-likelihood audit must be an object")
    missing = sorted(_ACTION_LIKELIHOOD_AUDIT_KEYS - set(value))
    extra = sorted(set(value) - _ACTION_LIKELIHOOD_AUDIT_KEYS)
    if missing or extra:
        raise ExperimentContractError(
            f"{artifact} action-likelihood audit keys mismatch: "
            f"missing={missing!r}, extra={extra!r}"
        )
    for field in ("expected_rollouts", "audited_rollouts"):
        observed = value.get(field)
        if not isinstance(observed, int) or isinstance(observed, bool):
            raise ExperimentContractError(
                f"{artifact} action-likelihood audit {field} must be an integer"
            )
    if value.get("passed") is not True:
        raise ExperimentContractError(f"{artifact} action-likelihood audit passed must be true")
    for field in ("action_boundary_margin", "log_probability_atol"):
        observed = value.get(field)
        if isinstance(observed, bool) or not isinstance(observed, (int, float)):
            raise ExperimentContractError(
                f"{artifact} action-likelihood audit {field} must be numeric"
            )
    exact = {
        "audit_id": ACTION_LIKELIHOOD_AUDIT_ID,
        "passed": True,
        "expected_rollouts": expected_rollouts,
        "audited_rollouts": expected_rollouts,
        "action_boundary_margin": ACTION_BOUNDARY_MARGIN,
        "log_probability_atol": LOG_PROB_RECOMPUTE_ATOL,
    }
    mismatches = [key for key, expected in exact.items() if value.get(key) != expected]
    if mismatches:
        raise ExperimentContractError(
            f"{artifact} action-likelihood audit mismatch: " + ", ".join(mismatches)
        )
    maximum = value.get("max_abs_stored_action")
    error = value.get("max_log_probability_abs_error")
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, (int, float))
        or not math.isfinite(float(maximum))
        or float(maximum) < 0.0
        or float(maximum) >= 1.0 - ACTION_BOUNDARY_MARGIN
    ):
        raise ExperimentContractError(f"{artifact} action-likelihood maximum is invalid")
    if (
        isinstance(error, bool)
        or not isinstance(error, (int, float))
        or not math.isfinite(float(error))
        or float(error) < 0.0
        or float(error) > LOG_PROB_RECOMPUTE_ATOL
    ):
        raise ExperimentContractError(f"{artifact} action-likelihood error is invalid")


def _validated_training_receipt(
    *,
    receipt_path: Path,
    design_path: Path,
    design: FixedReferenceStudyDesign,
    manifest_path: Path,
    manifest: FrozenExecutionManifest,
    runtime: RuntimeFingerprint,
    required_execution_callable_authority: str = CANONICAL_EXECUTION_CALLABLE_AUTHORITY,
) -> tuple[dict[str, Any], Path]:
    _require_manifest_callable_authority(
        manifest,
        required_execution_callable_authority,
    )
    receipt = _read_json_object(receipt_path, artifact="checkpoint receipt")
    missing = sorted(_TRAINING_RECEIPT_KEYS - set(receipt))
    extra = sorted(set(receipt) - _TRAINING_RECEIPT_KEYS)
    if missing or extra:
        raise ExperimentContractError(
            f"checkpoint receipt keys mismatch: missing={missing!r}, extra={extra!r}"
        )
    if not isinstance(receipt.get("schema_version"), int) or isinstance(
        receipt.get("schema_version"), bool
    ):
        raise ExperimentContractError("checkpoint receipt schema_version must be integer 1")
    for field in (
        "requested_timesteps",
        "observed_timesteps",
        "checkpoint_size_bytes",
    ):
        if not isinstance(receipt.get(field), int) or isinstance(receipt.get(field), bool):
            raise ExperimentContractError(f"checkpoint receipt {field} must be an integer")
    for field in (
        "checkpoint_eligible_for_declared_evaluation",
        "automatic_promotion",
    ):
        if not isinstance(receipt.get(field), bool):
            raise ExperimentContractError(f"checkpoint receipt {field} must be boolean")
    expected = {
        "schema_version": 1,
        "evidence_purpose": EvidencePurpose.BEHAVIORAL_EVALUATION.value,
        "completion_status": "complete",
        "claim_ceiling": design.claim_ceiling.value,
        "design_sha256": design.sha256,
        "design_file_sha256": sha256_file(design_path),
        "study_criteria_semantic_sha256": design.study_criteria_semantic_sha256,
        "study_criteria_file_sha256": design.study_criteria_file_sha256,
        "execution_manifest_sha256": manifest.sha256,
        "execution_manifest_file_sha256": sha256_file(manifest_path),
        "runtime_sha256": runtime.sha256,
        "execution_callable_authority": required_execution_callable_authority,
        **_runtime_semantics_receipt_fields(runtime),
        "algorithm_id": "stable_baselines3.PPO/v1",
        "policy_id": POLICY_ID,
        "action_transform_id": runtime.action_transform_id,
        "observation_normalizer_id": runtime.observation_normalizer_id,
        "reward_normalizer_id": runtime.reward_normalizer_id,
        "checkpoint_rule": design.checkpoint_rule,
        "requested_timesteps": design.ppo.total_timesteps_per_seed,
        "observed_timesteps": design.ppo.total_timesteps_per_seed,
        "checkpoint_file": CHECKPOINT_FILENAME,
        "checkpoint_eligible_for_declared_evaluation": (
            required_execution_callable_authority == CANONICAL_EXECUTION_CALLABLE_AUTHORITY
        ),
        "automatic_promotion": False,
        "behavioral_claim": None,
    }
    mismatches = [key for key, value in expected.items() if receipt.get(key) != value]
    if mismatches:
        raise ExperimentContractError(
            "checkpoint receipt does not match the frozen run: " + ", ".join(mismatches)
        )
    _validate_action_likelihood_audit_receipt(
        receipt["action_likelihood_audit"],
        expected_rollouts=(
            design.ppo.total_timesteps_per_seed // (design.ppo.n_envs * design.ppo.n_steps)
        ),
    )
    train_seed = receipt.get("train_seed")
    if (
        not isinstance(train_seed, int)
        or isinstance(train_seed, bool)
        or train_seed not in design.train_seeds
    ):
        raise ExperimentContractError("checkpoint receipt train_seed is not predetermined")
    checkpoint_path = receipt_path.parent / CHECKPOINT_FILENAME
    if not checkpoint_path.is_file():
        raise ExperimentContractError("checkpoint receipt's exact sibling checkpoint is missing")
    if receipt.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ExperimentContractError("checkpoint SHA-256 does not match its receipt")
    if receipt.get("checkpoint_size_bytes") != checkpoint_path.stat().st_size:
        raise ExperimentContractError("checkpoint size does not match its receipt")
    return receipt, checkpoint_path


def evaluate_checkpoint(
    *,
    design_path: Path,
    manifest_path: Path,
    checkpoint_receipt_path: Path,
    runs_dir: Path = DEFAULT_RUNS_DIR,
    runtime_validator: Callable[..., RuntimeFingerprint] = validate_before_run,
    environment_factory: Callable[[FixedReferenceStudyDesign], object] = make_static_tracking_env,
    model_loader: Callable[..., object] = _load_ppo,
    abi_factory: Callable[[object], object] = validate_humanoid_actuator_abi,
    state_reader: Callable[..., object] = tracking_state,
    history_reader: Callable[[object, object], tuple[np.ndarray, np.ndarray]] = (
        _initial_protected_history
    ),
    protected_step_reader: Callable[..., dict[str, object]] = _protected_step_inputs,
) -> dict[str, object]:
    """Evaluate one exact checkpoint on every declared seed or publish nothing."""

    design_path = design_path.resolve()
    manifest_path = manifest_path.resolve()
    checkpoint_receipt_path = checkpoint_receipt_path.resolve()
    design = load_study_design(design_path)
    manifest = load_execution_manifest(manifest_path)
    execution_callable_authority = _execution_callable_authority(
        (runtime_validator, validate_before_run),
        (environment_factory, make_static_tracking_env),
        (model_loader, _load_ppo),
        (abi_factory, validate_humanoid_actuator_abi),
        (state_reader, tracking_state),
        (history_reader, _initial_protected_history),
        (protected_step_reader, _protected_step_inputs),
    )
    _require_manifest_callable_authority(manifest, execution_callable_authority)
    runtime = runtime_validator(design=design, manifest=manifest)
    training_receipt, checkpoint_path = _validated_training_receipt(
        receipt_path=checkpoint_receipt_path,
        design_path=design_path,
        design=design,
        manifest_path=manifest_path,
        manifest=manifest,
        runtime=runtime,
        required_execution_callable_authority=execution_callable_authority,
    )

    reference = make_static_stand_reference(n_frames=design.static_reference_frames)
    reference.verify()
    if reference.identity.content_sha256 != runtime.reference_content_sha256:
        raise ExperimentContractError("evaluation reference differs from the frozen runtime")
    values = np.asarray(reference.values, dtype=np.float64)
    if not np.array_equal(values, np.broadcast_to(values[0], values.shape)):
        raise ExperimentContractError(
            "experiment 001 evaluator accepts only an exact static reference"
        )

    train_seed = int(training_receipt["train_seed"])
    final_name = (
        f"evaluation-train-seed-{train_seed:08d}-{training_receipt['checkpoint_sha256'][:12]}"
    )
    environment: object | None = None
    model: object | None = None
    episodes: list[dict[str, object]] = []
    with _atomic_run_directory(runs_dir, final_name=final_name) as (staging, final):
        try:
            environment = environment_factory(design)
            abi = _validated_evaluation_environment(
                environment,
                expected_reference_sha256=reference.identity.content_sha256,
                abi_factory=abi_factory,
            )
            model = model_loader(
                checkpoint_path,
                environment,
                device=design.ppo.device,
            )
            if int(model.num_timesteps) != design.ppo.total_timesteps_per_seed:
                raise ExperimentContractError(
                    "loaded SB3 checkpoint does not contain the exact budget"
                )

            for evaluation_seed in design.evaluation_seeds:
                observation, _ignored_reset_info = environment.reset(seed=evaluation_seed)
                previous_action, previous_acceleration = history_reader(environment, abi)
                initial_state = state_reader(environment, abi)
                accumulator = EpisodeAccumulator(
                    evaluation_seed=evaluation_seed,
                    requested_steps=design.max_episode_steps,
                    initial_root_position_world_m=(initial_state.root_position_world_m),
                    station_keeping_origin_source=STATION_KEEPING_ORIGIN_SOURCE,
                    initial_normalized_policy_action=previous_action,
                )
                for step_index in range(design.max_episode_steps):
                    action, _ignored_state = model.predict(observation, deterministic=True)
                    normalized_action = _normalized_policy_action(action)
                    (
                        observation,
                        tracking_reward,
                        terminated,
                        truncated,
                        _ignored_step_info,
                    ) = environment.step(normalized_action)
                    state = state_reader(environment, abi)
                    protected_inputs = protected_step_reader(
                        environment,
                        abi,
                        normalized_action=normalized_action,
                        previous_normalized_action=previous_action,
                        previous_joint_acceleration=previous_acceleration,
                    )
                    metrics = evaluate_tracking_step(
                        root_position_world_m=state.root_position_world_m,
                        root_height_m=state.root_height_m,
                        root_orientation_wxyz=state.root_orientation_wxyz,
                        root_linear_velocity_world_m_s=(state.root_linear_velocity_world_m_s),
                        root_angular_velocity_body_rad_s=(state.root_angular_velocity_body_rad_s),
                        joint_positions_rad=state.joint_positions_rad,
                        joint_velocities_rad_s=state.joint_velocities_rad_s,
                        reference_frame=values[step_index],
                        **protected_inputs,
                    )
                    accumulator.add(metrics, tracking_reward=float(tracking_reward))
                    previous_action = normalized_action
                    previous_acceleration = _finite_direct_vector(
                        protected_inputs["joint_accelerations_rad_s2"],
                        width=17,
                        field="joint acceleration history",
                    )
                    observed_step = step_index + 1
                    if (bool(terminated) or bool(truncated)) and (
                        observed_step < design.max_episode_steps
                    ):
                        raise ExperimentContractError(
                            f"evaluation seed {evaluation_seed} ended after {observed_step} "
                            f"steps; expected {design.max_episode_steps}"
                        )
                episodes.append(asdict(accumulator.finish(require_complete=True)))
        finally:
            if environment is not None:
                environment.close()
            model = None
            gc.collect()

        observed_seeds = [episode["evaluation_seed"] for episode in episodes]
        if observed_seeds != list(design.evaluation_seeds):
            raise ExperimentContractError(
                "evaluation did not complete every declared seed in order"
            )
        payload = {
            "schema_version": 1,
            "evidence_purpose": EvidencePurpose.BEHAVIORAL_EVALUATION.value,
            "completion_status": "complete_measurements_no_promotion",
            "claim_ceiling": design.claim_ceiling.value,
            "design_sha256": design.sha256,
            "design_file_sha256": sha256_file(design_path),
            "study_criteria_semantic_sha256": design.study_criteria_semantic_sha256,
            "study_criteria_file_sha256": design.study_criteria_file_sha256,
            "execution_manifest_sha256": manifest.sha256,
            "execution_manifest_file_sha256": sha256_file(manifest_path),
            "runtime_sha256": runtime.sha256,
            "execution_callable_authority": execution_callable_authority,
            **_runtime_semantics_receipt_fields(runtime),
            "checkpoint_receipt_file_sha256": sha256_file(checkpoint_receipt_path),
            "checkpoint_sha256": training_receipt["checkpoint_sha256"],
            "train_seed": train_seed,
            "evaluation_reference_sha256": reference.identity.content_sha256,
            "evaluation_reference_schema_sha256": reference.identity.schema_sha256,
            "evaluation_seeds_requested": list(design.evaluation_seeds),
            "evaluation_seeds_observed": observed_seeds,
            "max_episode_steps": design.max_episode_steps,
            "deterministic_actions": True,
            "protected_metrics_state_source": PROTECTED_METRICS_STATE_SOURCE,
            "station_keeping_origin_source": STATION_KEEPING_ORIGIN_SOURCE,
            "reward_telemetry_used_for_objective_metrics": False,
            "tracking_return_role": "environment_reward_diagnostic_only",
            "reference_timing_scope": "static_reference_only",
            "episodes": episodes,
            "complete": True,
            "automatic_promotion": False,
            "behavioral_claim": None,
        }
        _write_complete_file(staging / EVALUATION_RECEIPT_FILENAME, payload)

    receipt_path = final / EVALUATION_RECEIPT_FILENAME
    return {
        "run_directory": str(final),
        "receipt_path": str(receipt_path),
        "receipt_sha256": sha256_file(receipt_path),
        **payload,
    }


__all__ = [
    "CALIBRATION_RECEIPT_FILENAME",
    "CHECKPOINT_FILENAME",
    "CHECKPOINT_RECEIPT_FILENAME",
    "DEFAULT_RUNS_DIR",
    "EVALUATION_RECEIPT_FILENAME",
    "PROTECTED_METRICS_STATE_SOURCE",
    "calibrate_resources",
    "emit_json_without_overwrite",
    "emit_manifest_candidate",
    "evaluate_checkpoint",
    "finalize_reviewed_manifest",
    "load_execution_manifest",
    "train_one_seed",
]
