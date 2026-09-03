"""Reviewed, fail-closed runner for local phase-oracle exploration v1."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import gymnasium as gym
import numpy as np

from oracle_composition.envs import humanoid as humanoid_module
from oracle_composition.envs.humanoid import (
    CONTACT_CAPTURE_ID,
    HumanoidExperimentConfig,
    make_humanoid_env,
)
from oracle_composition.sources import minari_humanoid as minari_module
from oracle_composition.sources.minari_humanoid import (
    MinariHumanoidProjection,
    import_registered_minari_humanoid_episode,
)
from oracle_composition.traces import (
    ArtifactBinding,
    EvidenceClass,
    MissingReason,
    MissingSignal,
    NumericSignalSpec,
    TraceRecorder,
    TraceRole,
    load_trace,
)
from oracle_composition.traces import contract as trace_contract_module
from oracle_composition.tracking import (
    TrackingRewardConfig,
    compute_tracking_reward,
    tracking_state,
    validate_humanoid_actuator_abi,
)
from oracle_composition.tracking import humanoid_reference as humanoid_reference_module
from oracle_composition.tracking import reward as tracking_reward_module
from oracle_composition.tracking.humanoid_reference import (
    tracking_state_with_bounded_reset_orientation,
)

from . import exploratory_phase as phase_module
from . import oracle_exploration_design as design_module
from . import protected_evaluator as evaluator_module
from . import protected_runtime as protected_runtime_module
from .exploratory_phase import (
    BoundedPhaseConfig,
    BoundedPhaseMatcher,
    PhaseDecision,
    RecoveryDecision,
    RecoveryGate,
    RecoveryGateConfig,
    RecoveryMode,
)
from .fixed_reference import (
    ExperimentContractError,
    read_bounded_json_artifact,
    sha256_file,
)
from .local_controllers import (
    LocalControllerReceipt,
    load_local_behavior_cloning_controller,
    load_local_reference_residual_controller,
)
from .oracle_exploration_design import (
    EXPECTED_EXPERIMENT_ID,
    EXPECTED_PRE_EXECUTION_BINDINGS,
    OracleExplorationDesign,
    OracleExplorationRun,
    bounded_phase_config_from_design,
    collapse_definition_from_design,
    compose_physical_action_from_design,
    load_oracle_exploration_design,
    phase_scale_from_design,
    run_table_sha256,
)
from .protected_evaluator import EpisodeAccumulator, StepMetrics, evaluate_tracking_step
from .protected_runtime import initial_protected_history, protected_step_inputs

CLAIM_STATUS = "local_exploration_only"
MANIFEST_CANDIDATE_STATUS = "candidate_requires_human_review"
MANIFEST_REVIEWED_STATUS = "reviewed_frozen_pre_execution"
REVIEW_CONFIRMATION = "I reviewed the frozen v1 manifest"
MAX_EXECUTION_JSON_BYTES = 4 * 1024 * 1024
RUNTIME_RECEIPT_SCHEMA_VERSION = 1
EXECUTION_MANIFEST_SCHEMA_VERSION = 1
RUN_RECEIPT_SCHEMA_VERSION = 1
STUDY_RECEIPT_SCHEMA_VERSION = 1
STOCK_RETURN_SUMMATION_RULE = "python_math_fsum_in_action_order_over_exactly_1000_values/v1"


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExperimentContractError(f"value is not canonical JSON: {exc}") from exc


def _pretty_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExperimentContractError(f"value is not finite JSON: {exc}") from exc


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def runner_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _module_sha256(module: object) -> str:
    path = Path(str(getattr(module, "__file__", "")))
    if not path.is_file() or path.is_symlink():
        raise ExperimentContractError("bound module source must be one regular file")
    return sha256_file(path)


def _space_sha256(space: object) -> str:
    low = np.asarray(getattr(space, "low", None))
    high = np.asarray(getattr(space, "high", None))
    shape = tuple(int(value) for value in getattr(space, "shape", ()))
    dtype = np.dtype(getattr(space, "dtype", low.dtype))
    if not shape or low.shape != shape or high.shape != shape:
        raise ExperimentContractError("runtime space does not expose exact Box bounds")
    digest = hashlib.sha256()
    metadata = _canonical_json({"dtype": dtype.str, "shape": list(shape)})
    for label, payload in (
        (b"metadata", metadata),
        (b"low", np.ascontiguousarray(low, dtype=dtype).tobytes()),
        (b"high", np.ascontiguousarray(high, dtype=dtype).tobytes()),
    ):
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _dependency_lock_path() -> Path:
    package_lock = Path(__file__).resolve().parents[1] / "_runtime" / "uv.lock"
    checkout_lock = Path(__file__).resolve().parents[3] / "uv.lock"
    candidates = (package_lock, checkout_lock)
    if any(path.is_symlink() for path in candidates):
        raise ExperimentContractError("runtime dependency lock must not be a symlink")
    present = [path for path in candidates if path.is_file()]
    if not present:
        raise ExperimentContractError("runtime dependency lock is unavailable")
    if len(present) == 2 and sha256_file(present[0]) != sha256_file(present[1]):
        raise ExperimentContractError("packaged and checkout dependency locks disagree")
    return package_lock if package_lock in present else checkout_lock


def _time_limit_steps(environment: object) -> int:
    current = environment
    for _ in range(16):
        value = getattr(current, "_max_episode_steps", None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
        child = getattr(current, "env", None)
        if child is None or child is current:
            break
        current = child
    raise ExperimentContractError("runtime does not expose a bounded TimeLimit")


@dataclass(frozen=True, slots=True)
class OracleExplorationRuntimeReceipt:
    """Observed simulator, model, package, and controller-facing ABI facts."""

    schema_version: int
    claim_status: str
    environment_id: str
    gymnasium_version: str
    mujoco_version: str
    numpy_version: str
    model_xml_sha256: str
    max_actions: int
    terminate_when_unhealthy: bool
    reset_noise_scale: float
    exclude_current_positions_from_observation: bool
    frame_skip: int
    control_period_seconds: float
    observation_space_sha256: str
    action_space_sha256: str
    observation_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    qpos_shape: tuple[int, ...]
    qvel_shape: tuple[int, ...]
    actuator_joint_order: tuple[str, ...]
    actuator_qpos_indices: tuple[int, ...]
    actuator_qvel_indices: tuple[int, ...]
    actuator_gear_by_joint: tuple[float, ...]
    generalized_actuator_torque_capacity_n_m: tuple[float, ...]
    dependency_lock_sha256: str
    contact_capture_id: str
    environment_source_sha256: str
    reset_probe_seed: int

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        for field in (
            "observation_shape",
            "action_shape",
            "qpos_shape",
            "qvel_shape",
            "actuator_joint_order",
            "actuator_qpos_indices",
            "actuator_qvel_indices",
            "actuator_gear_by_joint",
            "generalized_actuator_torque_capacity_n_m",
        ):
            value[field] = list(value[field])
        return value

    @property
    def sha256(self) -> str:
        return _json_sha256(self.to_dict())


def inspect_oracle_exploration_runtime(
    design: OracleExplorationDesign,
    *,
    reset_probe_seed: int = 20260903,
) -> OracleExplorationRuntimeReceipt:
    """Inspect the runtime without advancing the simulator by one action."""

    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("install the gym extra before runtime inspection") from exc

    requested = design.to_dict()["requested_runtime"]
    config = HumanoidExperimentConfig(
        env_id=requested["environment_id"],
        terminate_when_unhealthy=requested["terminate_when_unhealthy"],
        reset_noise_scale=requested["reset_noise_scale"],
        exclude_current_positions_from_observation=(
            requested["exclude_current_positions_from_observation"]
        ),
        frame_skip=requested["frame_skip"],
    )
    environment = make_humanoid_env(config, capture_substep_contacts=True)
    try:
        observation, _ignored_info = environment.reset(seed=reset_probe_seed)
        physical = environment.unwrapped
        abi = validate_humanoid_actuator_abi(physical)
        model_path = Path(str(getattr(physical, "fullpath", "")))
        if not model_path.is_file() or model_path.is_symlink():
            raise ExperimentContractError("runtime model XML must be one regular file")
        receipt = OracleExplorationRuntimeReceipt(
            schema_version=RUNTIME_RECEIPT_SCHEMA_VERSION,
            claim_status=CLAIM_STATUS,
            environment_id=str(physical.spec.id),
            gymnasium_version=str(gym.__version__),
            mujoco_version=str(mujoco.__version__),
            numpy_version=str(np.__version__),
            model_xml_sha256=sha256_file(model_path),
            max_actions=_time_limit_steps(environment),
            terminate_when_unhealthy=bool(physical._terminate_when_unhealthy),
            reset_noise_scale=float(physical._reset_noise_scale),
            exclude_current_positions_from_observation=bool(
                physical._exclude_current_positions_from_observation
            ),
            frame_skip=int(physical.frame_skip),
            control_period_seconds=float(physical.dt),
            observation_space_sha256=_space_sha256(environment.observation_space),
            action_space_sha256=_space_sha256(environment.action_space),
            observation_shape=tuple(int(value) for value in observation.shape),
            action_shape=tuple(int(value) for value in environment.action_space.shape),
            qpos_shape=tuple(int(value) for value in physical.data.qpos.shape),
            qvel_shape=tuple(int(value) for value in physical.data.qvel.shape),
            actuator_joint_order=abi.joint_names,
            actuator_qpos_indices=abi.qpos_indices,
            actuator_qvel_indices=abi.qvel_indices,
            actuator_gear_by_joint=abi.actuator_gear_by_joint,
            generalized_actuator_torque_capacity_n_m=(abi.generalized_actuator_torque_capacity_n_m),
            dependency_lock_sha256=sha256_file(_dependency_lock_path()),
            contact_capture_id=str(getattr(physical, "contact_capture_id", "")),
            environment_source_sha256=_module_sha256(humanoid_module),
            reset_probe_seed=reset_probe_seed,
        )
        _validate_runtime_against_design(receipt, design)
        return receipt
    finally:
        environment.close()


def _validate_runtime_against_design(
    receipt: OracleExplorationRuntimeReceipt,
    design: OracleExplorationDesign,
) -> None:
    requested = design.to_dict()["requested_runtime"]
    comparisons = {
        "environment_id": requested["environment_id"],
        "gymnasium_version": requested["gymnasium_version"],
        "mujoco_version": requested["mujoco_version"],
        "model_xml_sha256": requested["mujoco_model_xml_sha256"],
        "max_actions": requested["max_actions"],
        "terminate_when_unhealthy": requested["terminate_when_unhealthy"],
        "reset_noise_scale": requested["reset_noise_scale"],
        "exclude_current_positions_from_observation": requested[
            "exclude_current_positions_from_observation"
        ],
        "frame_skip": requested["frame_skip"],
        "control_period_seconds": requested["control_period_seconds"],
    }
    mismatches = [
        field for field, expected in comparisons.items() if getattr(receipt, field) != expected
    ]
    if mismatches:
        raise ExperimentContractError(
            f"observed runtime differs from requested fields: {mismatches!r}"
        )
    if receipt.schema_version != RUNTIME_RECEIPT_SCHEMA_VERSION:
        raise ExperimentContractError("runtime receipt schema version is unsupported")
    if receipt.claim_status != CLAIM_STATUS:
        raise ExperimentContractError("runtime receipt exceeds the local claim boundary")
    if receipt.observation_shape != (348,) or receipt.action_shape != (17,):
        raise ExperimentContractError("runtime observation/action shape changed")
    action_rule = design.to_dict()["action_composition"]
    lower, upper = action_rule["final_clip_bounds_float32_as_float64"]
    expected_action_space = gym.spaces.Box(
        low=np.full(17, lower, dtype="<f4"),
        high=np.full(17, upper, dtype="<f4"),
        dtype=np.dtype("<f4"),
    )
    if receipt.action_space_sha256 != _space_sha256(expected_action_space):
        raise ExperimentContractError("runtime action space changed")
    if receipt.qpos_shape != (24,) or receipt.qvel_shape != (23,):
        raise ExperimentContractError("runtime qpos/qvel shape changed")
    if len(receipt.actuator_joint_order) != 17:
        raise ExperimentContractError("runtime actuator ABI is incomplete")
    if (
        list(receipt.actuator_joint_order)
        != design.to_dict()["action_composition"]["actuator_order"]
    ):
        raise ExperimentContractError("runtime actuator order differs from action composition")
    if receipt.contact_capture_id != CONTACT_CAPTURE_ID:
        raise ExperimentContractError("runtime does not expose all-substep contact capture")


@dataclass(frozen=True, slots=True)
class PreparedOracleExploration:
    """Validated in-memory inputs available before manifest review."""

    design_path: Path
    design_file_bytes: bytes
    design_file_sha256: str
    design: OracleExplorationDesign
    projection: MinariHumanoidProjection
    base_infer: Callable[[np.ndarray], np.ndarray]
    residual_infer: Callable[[np.ndarray, np.ndarray], np.ndarray]
    base_receipt: LocalControllerReceipt
    residual_receipt: LocalControllerReceipt
    runtime_receipt: OracleExplorationRuntimeReceipt
    bindings: Mapping[str, str]


def _require_design_pins(
    design: OracleExplorationDesign,
    projection: MinariHumanoidProjection,
    base_receipt: LocalControllerReceipt,
    residual_receipt: LocalControllerReceipt,
) -> None:
    pins = design.to_dict()["artifact_pins"]
    observed = {
        "source_record_sha256": projection.receipt.source_record_sha256,
        "hdf5_sha256": projection.receipt.hdf5_sha256,
        "metadata_sha256": projection.receipt.metadata_sha256,
        "projection_receipt_sha256": projection.receipt.sha256,
        "reference_content_sha256": projection.reference.identity.content_sha256,
        "reference_schema_sha256": projection.reference.identity.schema_sha256,
        "base_controller_sha256": base_receipt.content_sha256,
        "reference_residual_sha256": residual_receipt.content_sha256,
        "tracking_reward_sha256": TrackingRewardConfig().sha256,
    }
    observed["task_reward_sha256"] = _json_sha256(
        {"reward_id": "constant_zero_task_reward/v1", "value": 0.0}
    )
    mismatches = [field for field, value in observed.items() if pins[field] != value]
    if mismatches:
        raise ExperimentContractError(f"prepared artifacts differ from design pins: {mismatches!r}")
    if projection.receipt.source_origin_verified is not True:
        raise ExperimentContractError("Minari source must match the registered source record")
    if base_receipt.filename != pins["base_controller_filename"]:
        raise ExperimentContractError("base controller filename differs from the design pin")
    if residual_receipt.filename != pins["reference_residual_filename"]:
        raise ExperimentContractError("residual controller filename differs from the design pin")
    if not base_receipt.registered_content_match or not residual_receipt.registered_content_match:
        raise ExperimentContractError("controller bytes do not match the registered local hashes")
    action = design.to_dict()["action_composition"]
    if base_receipt.inference_output_contract != action["base_output_contract"]:
        raise ExperimentContractError("base controller receipt differs from the action contract")
    if not residual_receipt.inference_output_contract.startswith(
        action["residual_output_contract"] + ";"
    ):
        raise ExperimentContractError(
            "residual controller receipt differs from the action contract"
        )
    if residual_receipt.residual_scale != action["residual_scale_in_raw_control_units"]:
        raise ExperimentContractError("residual controller scale differs from the action contract")


def prepare_oracle_exploration(
    *,
    design_path: Path,
    hdf5_path: Path,
    metadata_path: Path,
    base_controller_path: Path,
    residual_controller_path: Path,
) -> PreparedOracleExploration:
    """Import and validate every pre-execution input without stepping Humanoid."""

    design_path = Path(design_path).absolute()
    design = load_oracle_exploration_design(design_path)
    return _prepare_oracle_exploration_from_design(
        design_path=design_path,
        design=design,
        hdf5_path=hdf5_path,
        metadata_path=metadata_path,
        base_controller_path=base_controller_path,
        residual_controller_path=residual_controller_path,
    )


def _prepare_oracle_exploration_from_design(
    *,
    design_path: Path,
    design: OracleExplorationDesign,
    hdf5_path: Path,
    metadata_path: Path,
    base_controller_path: Path,
    residual_controller_path: Path,
) -> PreparedOracleExploration:
    """Prepare assets against one already captured, canonical design object."""

    design_file_bytes = design.source_bytes
    design_file_sha256 = design.source_sha256
    projection = import_registered_minari_humanoid_episode(
        hdf5_path=Path(hdf5_path),
        metadata_path=Path(metadata_path),
        episode_id=0,
        expected_seed=123,
        expected_total_steps=1000,
        expected_final_terminated=False,
        expected_final_truncated=True,
    )
    base_infer, base_receipt = load_local_behavior_cloning_controller(Path(base_controller_path))
    residual_infer, residual_receipt = load_local_reference_residual_controller(
        Path(residual_controller_path)
    )
    _require_design_pins(design, projection, base_receipt, residual_receipt)
    runtime = inspect_oracle_exploration_runtime(design)
    if base_receipt.gymnasium_version != runtime.gymnasium_version:
        raise ExperimentContractError("base-controller load runtime differs from simulator runtime")
    if residual_receipt.gymnasium_version != runtime.gymnasium_version:
        raise ExperimentContractError(
            "residual-controller load runtime differs from simulator runtime"
        )

    bindings = {
        "oracle_exploration_design_sha256": design.sha256,
        "oracle_exploration_design_source_sha256": _module_sha256(design_module),
        "registered_source_record_sha256": str(projection.receipt.source_record_sha256),
        "registered_import_projection_receipt_sha256": projection.receipt.sha256,
        "minari_importer_source_sha256": _module_sha256(minari_module),
        "reference_schema_sha256": projection.reference.identity.schema_sha256,
        "humanoid_reference_source_sha256": _module_sha256(humanoid_reference_module),
        "observed_runtime_model_abi_receipt_sha256": runtime.sha256,
        "runner_source_sha256": runner_source_sha256(),
        "phase_oracle_source_sha256": _module_sha256(phase_module),
        "protected_evaluator_source_sha256": _module_sha256(evaluator_module),
        "protected_runtime_source_sha256": _module_sha256(protected_runtime_module),
        "tracking_reward_source_sha256": _module_sha256(tracking_reward_module),
        "trace_contract_source_sha256": _module_sha256(trace_contract_module),
        "base_controller_load_receipt_sha256": base_receipt.sha256(),
        "reference_residual_load_receipt_sha256": residual_receipt.sha256(),
    }
    if tuple(bindings) != EXPECTED_PRE_EXECUTION_BINDINGS:
        raise ExperimentContractError("pre-execution binding roles changed")
    return PreparedOracleExploration(
        design_path=design_path,
        design_file_bytes=design_file_bytes,
        design_file_sha256=design_file_sha256,
        design=design,
        projection=projection,
        base_infer=base_infer,
        residual_infer=residual_infer,
        base_receipt=base_receipt,
        residual_receipt=residual_receipt,
        runtime_receipt=runtime,
        bindings=bindings,
    )


@dataclass(frozen=True, slots=True)
class OracleExplorationExecutionManifest:
    schema_version: int
    experiment_id: str
    manifest_status: str
    claim_status: str
    evidence_class: str
    design_sha256: str
    design_file_sha256: str
    run_table_sha256: str
    expected_run_count: int
    bindings: Mapping[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "manifest_status": self.manifest_status,
            "claim_status": self.claim_status,
            "evidence_class": self.evidence_class,
            "design_sha256": self.design_sha256,
            "design_file_sha256": self.design_file_sha256,
            "run_table_sha256": self.run_table_sha256,
            "expected_run_count": self.expected_run_count,
            "bindings": dict(self.bindings),
        }

    @property
    def sha256(self) -> str:
        return _json_sha256(self.to_dict())


def manifest_from_prepared(
    prepared: PreparedOracleExploration,
) -> OracleExplorationExecutionManifest:
    """Create an unreviewed manifest candidate from observed facts."""

    return OracleExplorationExecutionManifest(
        schema_version=EXECUTION_MANIFEST_SCHEMA_VERSION,
        experiment_id=EXPECTED_EXPERIMENT_ID,
        manifest_status=MANIFEST_CANDIDATE_STATUS,
        claim_status=CLAIM_STATUS,
        evidence_class="exploratory",
        design_sha256=prepared.design.sha256,
        design_file_sha256=prepared.design_file_sha256,
        run_table_sha256=run_table_sha256(prepared.design.run_table),
        expected_run_count=prepared.design.expected_run_count,
        bindings=dict(prepared.bindings),
    )


def _manifest_from_dict(value: object) -> OracleExplorationExecutionManifest:
    if not isinstance(value, dict):
        raise ExperimentContractError("execution manifest must be an object")
    keys = {
        "schema_version",
        "experiment_id",
        "manifest_status",
        "claim_status",
        "evidence_class",
        "design_sha256",
        "design_file_sha256",
        "run_table_sha256",
        "expected_run_count",
        "bindings",
    }
    if set(value) != keys:
        raise ExperimentContractError("execution manifest keys differ from the v1 contract")
    bindings = value["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != set(EXPECTED_PRE_EXECUTION_BINDINGS):
        raise ExperimentContractError("execution manifest binding roles or order changed")
    manifest = OracleExplorationExecutionManifest(
        schema_version=value["schema_version"],
        experiment_id=value["experiment_id"],
        manifest_status=value["manifest_status"],
        claim_status=value["claim_status"],
        evidence_class=value["evidence_class"],
        design_sha256=value["design_sha256"],
        design_file_sha256=value["design_file_sha256"],
        run_table_sha256=value["run_table_sha256"],
        expected_run_count=value["expected_run_count"],
        bindings=dict(bindings),
    )
    if manifest.schema_version != EXECUTION_MANIFEST_SCHEMA_VERSION:
        raise ExperimentContractError("execution manifest schema version is unsupported")
    if manifest.experiment_id != EXPECTED_EXPERIMENT_ID:
        raise ExperimentContractError("execution manifest experiment id changed")
    if manifest.manifest_status not in {MANIFEST_CANDIDATE_STATUS, MANIFEST_REVIEWED_STATUS}:
        raise ExperimentContractError("execution manifest review status is invalid")
    if manifest.claim_status != CLAIM_STATUS or manifest.evidence_class != "exploratory":
        raise ExperimentContractError("execution manifest exceeds the exploration claim boundary")
    if manifest.expected_run_count != 240:
        raise ExperimentContractError("execution manifest must bind all 240 runs")
    for field in (
        manifest.design_sha256,
        manifest.design_file_sha256,
        manifest.run_table_sha256,
        *manifest.bindings.values(),
    ):
        if (
            not isinstance(field, str)
            or len(field) != 64
            or any(character not in "0123456789abcdef" for character in field)
        ):
            raise ExperimentContractError("execution manifest hashes must be lowercase SHA-256")
    return manifest


def _load_oracle_exploration_manifest_snapshot(
    path: Path,
) -> tuple[OracleExplorationExecutionManifest, bytes, str]:
    source = read_bounded_json_artifact(
        Path(path),
        maximum_bytes=MAX_EXECUTION_JSON_BYTES,
        artifact="oracle exploration execution manifest",
    )
    return _manifest_from_dict(source.value), source.encoded_bytes, source.sha256


def load_oracle_exploration_manifest(path: Path) -> OracleExplorationExecutionManifest:
    manifest, _encoded, _file_sha256 = _load_oracle_exploration_manifest_snapshot(path)
    return manifest


def emit_bytes_without_overwrite(path: Path, encoded: bytes) -> Path:
    """Publish exact bytes through an atomic no-overwrite link."""

    if not isinstance(encoded, bytes) or not encoded:
        raise ExperimentContractError("published artifact bytes must be nonempty")
    lexical = Path(path)
    if lexical.name in {"", ".", ".."}:
        raise ExperimentContractError("published artifact must have one filename")
    lexical.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = lexical.parent.resolve()
    if not resolved_parent.is_dir():
        raise ExperimentContractError("published artifact parent must be a directory")
    resolved = resolved_parent / lexical.name
    descriptor, pending_name = tempfile.mkstemp(
        prefix=f".{resolved.name}.", suffix=".pending", dir=resolved.parent
    )
    pending = Path(pending_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(pending, resolved)
        except FileExistsError as exc:
            raise ExperimentContractError(
                f"refusing to overwrite existing file: {resolved}"
            ) from exc
    finally:
        pending.unlink(missing_ok=True)
    return resolved


def emit_json_without_overwrite(path: Path, value: object) -> Path:
    """Publish a complete JSON artifact and refuse replacement."""

    return emit_bytes_without_overwrite(path, _pretty_json(value))


def emit_manifest_candidate(
    path: Path,
    prepared: PreparedOracleExploration,
) -> OracleExplorationExecutionManifest:
    manifest = manifest_from_prepared(prepared)
    emit_json_without_overwrite(path, manifest.to_dict())
    return manifest


def emit_pre_execution_receipts(
    output_dir: Path,
    prepared: PreparedOracleExploration,
) -> dict[str, Path]:
    """Persist the exact observed receipts reviewed with a manifest candidate."""

    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "observed_runtime_model_abi_receipt": prepared.runtime_receipt.to_dict(),
        "registered_import_projection_receipt": prepared.projection.receipt.to_dict(),
        "base_controller_load_receipt": prepared.base_receipt.to_dict(),
        "reference_residual_load_receipt": prepared.residual_receipt.to_dict(),
        "run_table": [row.to_dict() for row in prepared.design.run_table],
    }
    paths: dict[str, Path] = {}
    for name, payload in artifacts.items():
        paths[name] = emit_json_without_overwrite(directory / f"{name}.json", payload)
    return paths


def _materialize_study_lineage(
    *,
    staging: Path,
    prepared: PreparedOracleExploration,
    manifest: OracleExplorationExecutionManifest,
    manifest_file_bytes: bytes,
    manifest_file_sha256: str,
) -> tuple[Path, str]:
    """Copy the reviewed, nonsecret pre-execution lineage into the study."""

    if hashlib.sha256(prepared.design_file_bytes).hexdigest() != prepared.design_file_sha256:
        raise ExperimentContractError("carried design bytes changed in memory")
    if prepared.design_file_sha256 != manifest.design_file_sha256:
        raise ExperimentContractError("carried design file differs from the reviewed manifest")
    if hashlib.sha256(manifest_file_bytes).hexdigest() != manifest_file_sha256:
        raise ExperimentContractError("carried execution manifest bytes changed in memory")
    lineage = staging / "lineage"
    lineage.mkdir()
    files: dict[str, dict[str, object]] = {}

    def publish(
        filename: str,
        encoded: bytes,
        *,
        manifest_role: str,
        bound_sha256: str,
        semantic_role: str | None = None,
        semantic_sha256: str | None = None,
    ) -> None:
        path = emit_bytes_without_overwrite(lineage / filename, encoded)
        observed_sha256 = sha256_file(path)
        if observed_sha256 != hashlib.sha256(encoded).hexdigest():
            raise ExperimentContractError("published lineage bytes do not reverify")
        entry: dict[str, object] = {
            "bytes": len(encoded),
            "sha256": observed_sha256,
            "manifest_role": manifest_role,
            "bound_sha256": bound_sha256,
        }
        if semantic_role is not None and semantic_sha256 is not None:
            entry["semantic_manifest_role"] = semantic_role
            entry["semantic_sha256"] = semantic_sha256
        files[filename] = entry

    publish(
        "design.study.json",
        prepared.design_file_bytes,
        manifest_role="design_file_sha256",
        bound_sha256=manifest.design_file_sha256,
        semantic_role="oracle_exploration_design_sha256",
        semantic_sha256=prepared.design.sha256,
    )
    publish(
        "execution_manifest.reviewed.json",
        manifest_file_bytes,
        manifest_role="execution_manifest_file_sha256",
        bound_sha256=manifest_file_sha256,
        semantic_role="execution_manifest_sha256",
        semantic_sha256=manifest.sha256,
    )
    canonical_receipts = (
        (
            "observed_runtime_model_abi_receipt.json",
            prepared.runtime_receipt.to_dict(),
            "observed_runtime_model_abi_receipt_sha256",
        ),
        (
            "registered_import_projection_receipt.json",
            prepared.projection.receipt.to_dict(),
            "registered_import_projection_receipt_sha256",
        ),
        (
            "base_controller_load_receipt.json",
            prepared.base_receipt.to_dict(),
            "base_controller_load_receipt_sha256",
        ),
        (
            "reference_residual_load_receipt.json",
            prepared.residual_receipt.to_dict(),
            "reference_residual_load_receipt_sha256",
        ),
        (
            "run_table.json",
            [row.to_dict() for row in prepared.design.run_table],
            "run_table_sha256",
        ),
    )
    for filename, payload, role in canonical_receipts:
        encoded = _canonical_json(payload)
        bound_sha256 = (
            manifest.run_table_sha256 if role == "run_table_sha256" else manifest.bindings[role]
        )
        if hashlib.sha256(encoded).hexdigest() != bound_sha256:
            raise ExperimentContractError(f"{role} payload differs from its reviewed binding")
        publish(
            filename,
            encoded,
            manifest_role=role,
            bound_sha256=bound_sha256,
        )

    source_record_path = (
        Path(str(minari_module.__file__)).parent / "manifests" / "minari_humanoid_expert_v0.json"
    )
    source_record = read_bounded_json_artifact(
        source_record_path,
        maximum_bytes=MAX_EXECUTION_JSON_BYTES,
        artifact="registered Minari source record",
    )
    if source_record.sha256 != manifest.bindings["registered_source_record_sha256"]:
        raise ExperimentContractError("registered source record differs from its binding")
    publish(
        "registered_source_record.json",
        source_record.encoded_bytes,
        manifest_role="registered_source_record_sha256",
        bound_sha256=source_record.sha256,
    )

    included_roles = {
        str(role)
        for entry in files.values()
        for role in (
            entry["manifest_role"],
            entry.get("semantic_manifest_role"),
        )
        if isinstance(role, str) and role in manifest.bindings
    }
    lineage_manifest = {
        "schema_version": 1,
        "claim_status": CLAIM_STATUS,
        "execution_manifest_sha256": manifest.sha256,
        "files": files,
        "hash_only_pre_execution_bindings": {
            role: digest for role, digest in manifest.bindings.items() if role not in included_roles
        },
        "controller_and_source_code_bytes_embedded": False,
    }
    lineage_manifest_path = emit_bytes_without_overwrite(
        lineage / "lineage_manifest.json",
        _canonical_json(lineage_manifest),
    )
    return lineage_manifest_path, sha256_file(lineage_manifest_path)


def finalize_reviewed_manifest(
    *,
    candidate_path: Path,
    output_path: Path,
    review_confirmation: str,
) -> OracleExplorationExecutionManifest:
    """Freeze an exact candidate only after explicit human review."""

    if review_confirmation != REVIEW_CONFIRMATION:
        raise ExperimentContractError("exact human review confirmation is required")
    candidate = load_oracle_exploration_manifest(candidate_path)
    if candidate.manifest_status != MANIFEST_CANDIDATE_STATUS:
        raise ExperimentContractError("only an unmodified candidate can be finalized")
    reviewed = replace(candidate, manifest_status=MANIFEST_REVIEWED_STATUS)
    emit_json_without_overwrite(output_path, reviewed.to_dict())
    return reviewed


def validate_prepared_manifest(
    prepared: PreparedOracleExploration,
    manifest: OracleExplorationExecutionManifest,
) -> None:
    if manifest.manifest_status != MANIFEST_REVIEWED_STATUS:
        raise ExperimentContractError("execution requires a reviewed frozen manifest")
    candidate = manifest_from_prepared(prepared)
    expected = replace(candidate, manifest_status=MANIFEST_REVIEWED_STATUS)
    if manifest.to_dict() != expected.to_dict():
        raise ExperimentContractError("reviewed manifest differs from current pre-execution facts")


def _reference_state_vector(state: object) -> np.ndarray:
    value = np.concatenate(
        (
            np.asarray([state.root_height_m], dtype=np.float64),
            np.asarray(state.root_orientation_wxyz, dtype=np.float64),
            np.asarray(state.root_linear_velocity_world_m_s, dtype=np.float64),
            np.asarray(state.root_angular_velocity_body_rad_s, dtype=np.float64),
            np.asarray(state.joint_positions_rad, dtype=np.float64),
            np.asarray(state.joint_velocities_rad_s, dtype=np.float64),
        )
    )
    if value.shape != (45,) or not np.isfinite(value).all():
        raise ExperimentContractError("observable phase state must be one finite 45D row")
    return value


def _elapsed_matcher_config(config: BoundedPhaseConfig) -> BoundedPhaseConfig:
    return BoundedPhaseConfig(
        horizon_steps=config.horizon_steps,
        search_radius_frames=0,
        anchor_penalty=config.anchor_penalty,
        switch_margin=config.switch_margin,
        match_indices=config.match_indices,
        max_phase_advance_frames=config.max_phase_advance_frames,
        quaternion_indices=config.quaternion_indices,
        quaternion_angle_scale=config.quaternion_angle_scale,
        quaternion_unit_tolerance=config.quaternion_unit_tolerance,
    )


def _recovery_config(design: OracleExplorationDesign) -> RecoveryGateConfig:
    value = design.to_dict()["recovery_gate"]
    return RecoveryGateConfig(
        entry_height_min=value["entry_height_min_m"],
        entry_height_max=value["entry_height_max_m"],
        entry_up_z_min=value["entry_torso_up_z_min"],
        entry_pose_error_max=value["entry_pose_error_max"],
        exit_height_min=value["exit_height_min_m"],
        exit_height_max=value["exit_height_max_m"],
        exit_up_z_min=value["exit_torso_up_z_min"],
        exit_pose_error_max=value["exit_pose_error_max"],
        entry_pose_error_dwell_steps=value["entry_pose_error_dwell_steps"],
        exit_stable_steps=value["exit_stable_steps"],
    )


def _torso_up_z(quaternion_wxyz: np.ndarray) -> float:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ExperimentContractError("torso orientation must be one finite quaternion")
    _, x, y, _ = quaternion
    return float(1.0 - 2.0 * (x * x + y * y))


def diagnostic_stock_environment_return(
    rewards: Sequence[object],
    *,
    expected_count: int,
    summation_rule: str,
) -> float:
    """Validate and sum the diagnostic-only Gym rewards in action order."""

    if summation_rule != STOCK_RETURN_SUMMATION_RULE:
        raise ExperimentContractError("stock-return summation rule changed")
    if len(rewards) != expected_count:
        raise ExperimentContractError("stock-return diagnostic is missing step rewards")
    values: list[float] = []
    for reward in rewards:
        if isinstance(reward, bool):
            raise ExperimentContractError("stock environment reward must be finite")
        try:
            value = float(reward)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ExperimentContractError("stock environment reward must be finite") from exc
        if not math.isfinite(value):
            raise ExperimentContractError("stock environment reward must be finite")
        values.append(value)
    try:
        total = math.fsum(values)
    except OverflowError as exc:
        raise ExperimentContractError("stock environment return overflowed") from exc
    if not math.isfinite(total):
        raise ExperimentContractError("stock environment return must be finite")
    return total


def _ungated_recovery(decision: PhaseDecision) -> RecoveryDecision:
    return RecoveryDecision(
        mode=RecoveryMode.TRACK,
        residual_weight=1.0,
        selected_phase=decision.selected_phase,
        selected_pose_error=decision.selected_pose_error,
        stable_steps=0,
        pose_error_bad_steps=0,
        entered=False,
        exited=False,
        entry_reasons=(),
    )


def _recovery_transition_value(
    *,
    previous_mode: RecoveryMode,
    decision: RecoveryDecision,
    exit_stable_steps: int,
) -> str:
    if previous_mode is RecoveryMode.TRACK and decision.mode is RecoveryMode.RECOVER:
        if not decision.entered or not decision.entry_reasons:
            raise ExperimentContractError("recovery entry transition lacks guard reasons")
        return "track->recover;entry_reasons=" + ",".join(decision.entry_reasons)
    if previous_mode is RecoveryMode.RECOVER and decision.mode is RecoveryMode.TRACK:
        if not decision.exited:
            raise ExperimentContractError("recovery exit transition lacks exit evidence")
        return f"recover->track;exit_reason=all_exit_guards_stable_for_{exit_stable_steps}_steps"
    raise ExperimentContractError("recovery transition modes are not adjacent")


def compose_controller_action(
    *,
    design: OracleExplorationDesign,
    base_raw_control: object,
    residual_unscaled: object,
    gate_weight: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Compose the frozen float32 command and its effective normalized view."""

    physical = compose_physical_action_from_design(
        design,
        base_action=base_raw_control,
        residual_action=residual_unscaled,
        gate_weight=gate_weight,
    )
    normalized = np.clip(
        np.asarray(physical, dtype=np.float64) / 0.4,
        -1.0,
        1.0,
    )
    if physical.dtype != np.dtype("<f4") or not np.isfinite(physical).all():
        raise ExperimentContractError("composed controller action is invalid")
    return physical, normalized


def materialize_executed_action(
    *,
    design: OracleExplorationDesign,
    physical_control: object,
    action_space: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Materialize submitted bytes and the frozen effective normalized view."""

    rule = design.to_dict()["action_composition"]
    lower, upper = rule["final_clip_bounds_float32_as_float64"]
    shape = tuple(getattr(action_space, "shape", ()))
    dtype = np.dtype(getattr(action_space, "dtype", np.dtype("O")))
    if shape != (17,) or dtype.str != "<f4":
        raise ExperimentContractError("environment action space changed")
    if not np.array_equal(np.asarray(action_space.low), np.full(17, lower, dtype=dtype)):
        raise ExperimentContractError("environment action lower bound changed")
    if not np.array_equal(np.asarray(action_space.high), np.full(17, upper, dtype=dtype)):
        raise ExperimentContractError("environment action upper bound changed")
    try:
        typed = np.asarray(physical_control, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExperimentContractError("physical control cannot be cast to the action ABI") from exc
    if typed.shape != (17,) or not np.isfinite(typed).all():
        raise ExperimentContractError("executed physical control must be one finite 17D vector")
    executed_physical = np.asarray(typed, dtype=np.float64)
    if np.any(executed_physical < lower) or np.any(executed_physical > upper):
        raise ExperimentContractError("executed physical control exceeds the action ABI")
    normalized = np.clip(executed_physical / 0.4, -1.0, 1.0)
    return typed.copy(), executed_physical, normalized


def _apply_perturbation(environment: object, run: OracleExplorationRun) -> np.ndarray:
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("MuJoCo is required for perturbation execution") from exc
    if run.qvel_index is None or run.perturbation_action is None:
        raise ExperimentContractError("perturbation run is missing its frozen intervention")
    physical = environment.unwrapped
    qvel = np.asarray(physical.data.qvel)
    if qvel.shape != (23,) or not 0 <= run.qvel_index < 23:
        raise ExperimentContractError("perturbation qvel index is outside the runtime ABI")
    qvel[run.qvel_index] += run.signed_magnitude
    mujoco.mj_forward(physical.model, physical.data)
    observation = np.asarray(physical._get_obs(), dtype=np.float64)
    if observation.shape != (348,) or not np.isfinite(observation).all():
        raise ExperimentContractError("post-intervention observation is invalid")
    disturbance = np.zeros(23, dtype=np.float64)
    disturbance[run.qvel_index] = run.signed_magnitude
    return disturbance


def _numeric_signal_specs() -> tuple[NumericSignalSpec, ...]:
    return (
        NumericSignalSpec("robot.qpos", TraceRole.ROBOT, (24,), "mixed", "MuJoCo", "direct qpos"),
        NumericSignalSpec("robot.qvel", TraceRole.ROBOT, (23,), "mixed", "MuJoCo", "direct qvel"),
        NumericSignalSpec(
            "robot.root_position_world_m", TraceRole.ROBOT, (3,), "m", "world", "direct qpos[0:3]"
        ),
        NumericSignalSpec(
            "controller.action",
            TraceRole.CONTROLLER,
            (17,),
            "1",
            "actuator order",
            "executed normalized action from prior interval",
        ),
        NumericSignalSpec(
            "controller.physical_control",
            TraceRole.CONTROLLER,
            (17,),
            "MuJoCo control",
            "actuator order",
            "executed raw control from prior interval",
        ),
        NumericSignalSpec(
            "controller.applied_torque",
            TraceRole.CONTROLLER,
            (17,),
            "N*m",
            "actuator order",
            "direct qfrc_actuator",
        ),
        NumericSignalSpec(
            "controller.saturation",
            TraceRole.CONTROLLER,
            (17,),
            "1",
            "actuator order",
            "absolute normalized action equals one",
        ),
        NumericSignalSpec(
            "reference.frame",
            TraceRole.REFERENCE,
            (45,),
            "mixed",
            "reference ABI",
            "cached W[t][1] for prior interval",
        ),
        NumericSignalSpec(
            "reference.window_index",
            TraceRole.REFERENCE,
            (1,),
            "frame",
            "reference",
            "cached target frame index",
        ),
        NumericSignalSpec(
            "contact.floor_normal_force_n",
            TraceRole.CONTACT,
            (1,),
            "N",
            "world",
            "peak over all physics substeps",
        ),
        NumericSignalSpec(
            "oracle.mode",
            TraceRole.ORACLE,
            (1,),
            "1",
            "oracle",
            "RecoveryGate/v1 mode governing the prior interval",
        ),
        NumericSignalSpec(
            "oracle.phase",
            TraceRole.ORACLE,
            (1,),
            "frame",
            "reference",
            "window start governing prior interval",
        ),
        NumericSignalSpec(
            "oracle.transition_guard_margin",
            TraceRole.ORACLE,
            (1,),
            "scaled error",
            "oracle",
            "entry pose threshold minus selected pose error",
        ),
        NumericSignalSpec(
            "recovery.disturbance",
            TraceRole.RECOVERY,
            (23,),
            "mixed",
            "MuJoCo qvel",
            "exact qvel delta applied before the prior interval",
        ),
        NumericSignalSpec(
            "recovery.rejoin_state",
            TraceRole.RECOVERY,
            (3,),
            "mixed",
            "oracle",
            "residual weight; stable steps; bad pose steps",
        ),
        NumericSignalSpec(
            "task.progress",
            TraceRole.TASK,
            (1,),
            "m",
            "world x",
            "root x displacement from seeded reset",
        ),
        NumericSignalSpec(
            "evaluation.collapsed",
            TraceRole.EVALUATION,
            (1,),
            "1",
            "protected evaluator",
            "frozen physical collapse flag",
        ),
        NumericSignalSpec(
            "evaluation.tracking_errors",
            TraceRole.EVALUATION,
            (6,),
            "mixed",
            "protected evaluator",
            "height; orientation; root linear; root angular; joint position; joint velocity",
        ),
        NumericSignalSpec(
            "evaluation.stock_environment_reward",
            TraceRole.EVALUATION,
            (1,),
            "Gym reward",
            "environment",
            "finite diagnostic-only stock reward from prior interval",
        ),
    )


def _missing_signals() -> tuple[MissingSignal, ...]:
    return (
        MissingSignal(
            "controller.motor_target",
            MissingReason.NOT_APPLICABLE,
            "Humanoid-v5 uses direct torque-scaled actuator controls, not motor targets.",
        ),
        MissingSignal(
            "controller.energy_j",
            MissingReason.NOT_IMPLEMENTED,
            "The protected evaluator records action and post-transmission torque but has no audited energy integral.",
        ),
    )


def _trace_bindings(
    prepared: PreparedOracleExploration,
    manifest: OracleExplorationExecutionManifest,
) -> tuple[ArtifactBinding, ...]:
    pins = prepared.design.to_dict()["artifact_pins"]
    roles = (
        ("runtime", "observed_runtime_model_abi_receipt/v1", prepared.runtime_receipt.sha256),
        ("execution_manifest", "exploratory_phase_oracle_v1/manifest", manifest.sha256),
        (
            "evaluator",
            "protected_evaluator/source",
            manifest.bindings["protected_evaluator_source_sha256"],
        ),
        (
            "oracle",
            "bounded_phase_and_recovery/source",
            manifest.bindings["phase_oracle_source_sha256"],
        ),
        ("policy_checkpoint", "local_behavior_cloning/bytes", pins["base_controller_sha256"]),
        ("tracker_checkpoint", "local_reference_residual/bytes", pins["reference_residual_sha256"]),
        (
            "reference",
            prepared.projection.reference.identity.artifact_id,
            pins["reference_content_sha256"],
        ),
        ("task_reward", "constant_zero_task_reward/v1", pins["task_reward_sha256"]),
        ("tracking_reward", "humanoid_root_and_joint_tracking/v1", pins["tracking_reward_sha256"]),
        ("design", EXPECTED_EXPERIMENT_ID, prepared.design.sha256),
        ("run_table", "sha256_ranked_complete_blocks/v1", manifest.run_table_sha256),
        ("source_record", "minari_humanoid_expert_v0", pins["source_record_sha256"]),
        ("projection_receipt", "minari_episode_0_projection/v2", pins["projection_receipt_sha256"]),
        ("runner", "oracle_exploration_runner/source", manifest.bindings["runner_source_sha256"]),
        ("base_controller_load_receipt", "local_bc/load_receipt", prepared.base_receipt.sha256()),
        (
            "residual_controller_load_receipt",
            "local_residual/load_receipt",
            prepared.residual_receipt.sha256(),
        ),
    )
    return tuple(ArtifactBinding(*row) for row in roles)


def _reset_sample(
    *,
    physical: object,
    state: object,
    reference_frame: np.ndarray,
    phase: int,
    pose_margin: float,
    recovery: RecoveryDecision,
    initial_root_x: float,
) -> dict[str, object]:
    return {
        "robot.qpos": np.asarray(physical.data.qpos, dtype=np.float64).copy(),
        "robot.qvel": np.asarray(physical.data.qvel, dtype=np.float64).copy(),
        "robot.root_position_world_m": state.root_position_world_m,
        "controller.action": np.zeros(17, dtype=np.float64),
        "controller.physical_control": np.zeros(17, dtype=np.float64),
        "controller.applied_torque": np.zeros(17, dtype=np.float64),
        "controller.saturation": np.zeros(17, dtype=np.float64),
        "reference.frame": reference_frame,
        "reference.window_index": [phase],
        "contact.floor_normal_force_n": [0.0],
        "oracle.mode": [0.0 if recovery.mode is RecoveryMode.TRACK else 1.0],
        "oracle.phase": [phase],
        "oracle.transition_guard_margin": [pose_margin],
        "recovery.disturbance": np.zeros(23, dtype=np.float64),
        "recovery.rejoin_state": [
            recovery.residual_weight,
            recovery.stable_steps,
            recovery.pose_error_bad_steps,
        ],
        "task.progress": [float(state.root_position_world_m[0]) - initial_root_x],
        "evaluation.collapsed": [0.0],
        "evaluation.tracking_errors": np.zeros(6, dtype=np.float64),
        "evaluation.stock_environment_reward": [0.0],
    }


def _post_step_sample(
    *,
    physical: object,
    state: object,
    normalized_action: np.ndarray,
    physical_control: np.ndarray,
    target_frame: np.ndarray,
    target_index: int,
    phase: int,
    pose_margin: float,
    recovery: RecoveryDecision,
    metrics: StepMetrics,
    protected_inputs: Mapping[str, object],
    disturbance: np.ndarray,
    stock_environment_reward: float,
    initial_root_x: float,
) -> dict[str, object]:
    contacts = protected_inputs["contact_facts"]
    floor_peak = max(
        (contact.normal_force_n for contact in contacts if contact.involves_floor),
        default=0.0,
    )
    return {
        "robot.qpos": np.asarray(physical.data.qpos, dtype=np.float64).copy(),
        "robot.qvel": np.asarray(physical.data.qvel, dtype=np.float64).copy(),
        "robot.root_position_world_m": state.root_position_world_m,
        "controller.action": normalized_action,
        "controller.physical_control": physical_control,
        "controller.applied_torque": protected_inputs["generalized_actuator_torque_n_m"],
        "controller.saturation": np.equal(np.abs(normalized_action), 1.0).astype(np.float64),
        "reference.frame": target_frame,
        "reference.window_index": [target_index],
        "contact.floor_normal_force_n": [floor_peak],
        "oracle.mode": [0.0 if recovery.mode is RecoveryMode.TRACK else 1.0],
        "oracle.phase": [phase],
        "oracle.transition_guard_margin": [pose_margin],
        "recovery.disturbance": disturbance,
        "recovery.rejoin_state": [
            recovery.residual_weight,
            recovery.stable_steps,
            recovery.pose_error_bad_steps,
        ],
        "task.progress": [float(state.root_position_world_m[0]) - initial_root_x],
        "evaluation.collapsed": [float(metrics.collapsed)],
        "evaluation.tracking_errors": [
            metrics.root_height_abs_error_m,
            metrics.root_orientation_error_rad,
            metrics.root_linear_velocity_rmse_m_s,
            metrics.root_angular_velocity_rmse_rad_s,
            metrics.joint_position_rmse_rad,
            metrics.joint_velocity_rmse_rad_s,
        ],
        "evaluation.stock_environment_reward": [stock_environment_reward],
    }


def _run_one(
    *,
    prepared: PreparedOracleExploration,
    manifest: OracleExplorationExecutionManifest,
    run: OracleExplorationRun,
    output_dir: Path,
) -> dict[str, object]:
    requested = prepared.design.to_dict()["requested_runtime"]
    reference = np.asarray(prepared.projection.reference.values, dtype=np.float64)
    phase_config = bounded_phase_config_from_design(prepared.design)
    matcher_config = (
        phase_config if run.arm_id.startswith("T1_") else _elapsed_matcher_config(phase_config)
    )
    matcher = BoundedPhaseMatcher(
        reference=reference,
        scale=phase_scale_from_design(prepared.design, reference),
        config=matcher_config,
    )
    collapse_definition = collapse_definition_from_design(prepared.design)
    recovery_config = _recovery_config(prepared.design)
    recovery_gate = RecoveryGate(recovery_config)
    uses_gate = run.arm_id.endswith("_G1")
    environment = make_humanoid_env(
        HumanoidExperimentConfig(
            env_id=requested["environment_id"],
            terminate_when_unhealthy=requested["terminate_when_unhealthy"],
            reset_noise_scale=requested["reset_noise_scale"],
            exclude_current_positions_from_observation=(
                requested["exclude_current_positions_from_observation"]
            ),
            frame_skip=requested["frame_skip"],
        ),
        capture_substep_contacts=True,
    )
    samples: list[dict[str, object]] = []
    events: list[tuple[int, str, str, str]] = []
    stock_environment_rewards: list[float] = []
    try:
        observation, _ignored_info = environment.reset(seed=run.evaluation_seed)
        observation = np.asarray(observation, dtype=np.float64)
        if observation.shape != (348,) or not np.isfinite(observation).all():
            raise ExperimentContractError("seeded reset returned an invalid observation")
        physical = environment.unwrapped
        abi = validate_humanoid_actuator_abi(physical)
        previous_action, previous_acceleration = initial_protected_history(environment, abi)
        state = tracking_state_with_bounded_reset_orientation(environment, abi)
        initial_root_x = float(state.root_position_world_m[0])
        previous_phase: int | None = None
        accumulator = EpisodeAccumulator(
            evaluation_seed=run.evaluation_seed,
            requested_steps=requested["max_actions"],
            initial_root_position_world_m=state.root_position_world_m,
            station_keeping_origin_source=evaluator_module.STATION_KEEPING_ORIGIN_SOURCE,
            initial_normalized_policy_action=previous_action,
        )
        last_recovery_mode = RecoveryMode.TRACK
        for action_index in range(requested["max_actions"]):
            disturbance = np.zeros(23, dtype=np.float64)
            if run.perturbation_action == action_index:
                disturbance = _apply_perturbation(environment, run)
                observation = np.asarray(physical._get_obs(), dtype=np.float64)
                state = tracking_state(environment, abi)
                events.append(
                    (
                        action_index,
                        "recovery.disturbance_applied",
                        f"qvel[{run.qvel_index}]+={run.signed_magnitude:.17g} {run.unit}",
                        "frozen v1 perturbation schedule",
                    )
                )
            decision = matcher.select(
                _reference_state_vector(state),
                nominal_phase=action_index,
                previous_selected_phase=previous_phase,
            )
            previous_phase = decision.selected_phase
            recovery = (
                recovery_gate.step(
                    root_height_m=state.root_height_m,
                    torso_up_z=_torso_up_z(state.root_orientation_wxyz),
                    phase_decision=decision,
                )
                if uses_gate
                else _ungated_recovery(decision)
            )
            if recovery.mode is not last_recovery_mode:
                events.append(
                    (
                        action_index,
                        "oracle.transition",
                        _recovery_transition_value(
                            previous_mode=last_recovery_mode,
                            decision=recovery,
                            exit_stable_steps=recovery_config.exit_stable_steps,
                        ),
                        "RecoveryGate/v1",
                    )
                )
                last_recovery_mode = recovery.mode
            if decision.corrected:
                events.append(
                    (
                        action_index,
                        "oracle.phase_correction",
                        f"nominal={decision.nominal_phase};selected={decision.selected_phase}",
                        "BoundedPhaseMatcher/v1",
                    )
                )
            window = matcher.window(decision.selected_phase)
            pose_margin = (
                prepared.design.to_dict()["recovery_gate"]["entry_pose_error_max"]
                - decision.selected_pose_error
            )
            if action_index == 0:
                samples.append(
                    _reset_sample(
                        physical=physical,
                        state=state,
                        reference_frame=window[0],
                        phase=decision.selected_phase,
                        pose_margin=pose_margin,
                        recovery=recovery,
                        initial_root_x=initial_root_x,
                    )
                )
            base_action = np.asarray(prepared.base_infer(observation), dtype=np.float64)
            residual_action = np.asarray(
                prepared.residual_infer(observation, window), dtype=np.float64
            )
            physical_action, _pre_cast_normalized = compose_controller_action(
                design=prepared.design,
                base_raw_control=base_action,
                residual_unscaled=residual_action,
                gate_weight=recovery.residual_weight,
            )
            executed_action, executed_physical_action, normalized_action = (
                materialize_executed_action(
                    design=prepared.design,
                    physical_control=physical_action,
                    action_space=environment.action_space,
                )
            )
            observation, stock_reward, terminated, truncated, _ignored_step_info = environment.step(
                executed_action
            )
            if isinstance(stock_reward, bool):
                raise ExperimentContractError("stock environment reward must be finite")
            try:
                stock_reward_value = float(stock_reward)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ExperimentContractError("stock environment reward must be finite") from exc
            if not math.isfinite(stock_reward_value):
                raise ExperimentContractError("stock environment reward must be finite")
            stock_environment_rewards.append(stock_reward_value)
            if not np.array_equal(
                np.asarray(physical.data.ctrl, dtype=np.float64),
                executed_physical_action,
            ):
                raise ExperimentContractError(
                    "MuJoCo submitted control differs from the cast command bytes"
                )
            observation = np.asarray(observation, dtype=np.float64)
            if observation.shape != (348,) or not np.isfinite(observation).all():
                raise ExperimentContractError("Humanoid step returned an invalid observation")
            state = tracking_state(environment, abi)
            protected = protected_step_inputs(
                environment,
                abi,
                normalized_action=normalized_action,
                previous_normalized_action=previous_action,
                previous_joint_acceleration=previous_acceleration,
            )
            target_frame = window[1]
            metrics = evaluate_tracking_step(
                root_position_world_m=state.root_position_world_m,
                root_height_m=state.root_height_m,
                root_orientation_wxyz=state.root_orientation_wxyz,
                root_linear_velocity_world_m_s=state.root_linear_velocity_world_m_s,
                root_angular_velocity_body_rad_s=state.root_angular_velocity_body_rad_s,
                joint_positions_rad=state.joint_positions_rad,
                joint_velocities_rad_s=state.joint_velocities_rad_s,
                reference_frame=target_frame,
                collapse_definition=collapse_definition,
                **protected,
            )
            tracking_reward = compute_tracking_reward(
                state=state,
                reference_frame=target_frame,
            ).total
            accumulator.add(metrics, tracking_reward=tracking_reward)
            samples.append(
                _post_step_sample(
                    physical=physical,
                    state=state,
                    normalized_action=normalized_action,
                    physical_control=executed_physical_action,
                    target_frame=target_frame,
                    target_index=decision.selected_phase + 1,
                    phase=decision.selected_phase,
                    pose_margin=pose_margin,
                    recovery=recovery,
                    metrics=metrics,
                    protected_inputs=protected,
                    disturbance=disturbance,
                    stock_environment_reward=stock_reward_value,
                    initial_root_x=initial_root_x,
                )
            )
            previous_action = normalized_action
            previous_acceleration = np.asarray(
                protected["joint_accelerations_rad_s2"], dtype=np.float64
            ).copy()
            final_action = action_index + 1 == requested["max_actions"]
            if bool(terminated) or (bool(truncated) and not final_action):
                raise ExperimentContractError(
                    f"scheduled run ended after {action_index + 1} actions"
                )
        episode = accumulator.finish(require_complete=True)
        if len(samples) != requested["max_actions"] + 1:
            raise ExperimentContractError("run trace does not contain 1001 boundary samples")
        analysis = prepared.design.to_dict()["analysis"]
        stock_environment_return = diagnostic_stock_environment_return(
            stock_environment_rewards,
            expected_count=requested["max_actions"],
            summation_rule=analysis["stock_environment_return_summation_rule"],
        )
        trace_id = (
            f"v1-run-{run.run_order:03d}-seed-{run.evaluation_seed}-{run.condition_id}-{run.arm_id}"
        )
        recorder = TraceRecorder(
            trace_id=trace_id,
            evidence_class=EvidenceClass.EXPLORATORY,
            control_period_seconds=prepared.runtime_receipt.control_period_seconds,
            numeric_signals=_numeric_signal_specs(),
            missing_signals=_missing_signals(),
            artifact_bindings=_trace_bindings(prepared, manifest),
        )
        for sample in samples:
            recorder.add_sample(sample)
        for sample_index, event_type, value, source in events:
            recorder.add_event(
                sample_index=sample_index,
                event_type=event_type,
                value=value,
                source=source,
            )
        trace = recorder.finish()
        trace_path = output_dir / "trace.json"
        trace.write(trace_path)
        verified_trace = load_trace(trace_path)
        if verified_trace.sha256 != trace.sha256:
            raise ExperimentContractError("written trace failed content verification")
        run_receipt = {
            "schema_version": RUN_RECEIPT_SCHEMA_VERSION,
            "claim_status": CLAIM_STATUS,
            "evidence_class": "exploratory",
            "run": run.to_dict(),
            "actions_observed": requested["max_actions"],
            "trace_samples_observed": len(samples),
            "oracle_contract_faults": 0,
            "trace_filename": trace_path.name,
            "trace_sha256": trace.sha256,
            "diagnostic_coverage": trace.diagnostic_coverage,
            "episode_metrics": asdict(episode),
            "stock_environment_return_role": analysis["stock_environment_return_role"],
            "stock_environment_return_summation_rule": analysis[
                "stock_environment_return_summation_rule"
            ],
            "stock_environment_reward_count": len(stock_environment_rewards),
            "stock_environment_return": stock_environment_return,
            "automatic_promotion": False,
            "tier_d_claim": False,
            "causal_oracle_claim": False,
            "formal_oracle_evidence_claim": False,
        }
        emit_json_without_overwrite(output_dir / "run_receipt.json", run_receipt)
        return run_receipt
    finally:
        environment.close()


def _validate_completed_run_receipt(
    *,
    receipt: Mapping[str, object],
    expected_run: OracleExplorationRun,
    trace_path: Path,
    prepared: PreparedOracleExploration,
    manifest: OracleExplorationExecutionManifest,
) -> None:
    """Reverify one published-in-staging run against every completion gate."""

    if receipt.get("run") != expected_run.to_dict():
        raise ExperimentContractError("run receipt does not identify its scheduled row")
    expected_actions = prepared.design.to_dict()["evaluator"]["primary_outcome"]["required_actions"]
    if receipt.get("actions_observed") != expected_actions:
        raise ExperimentContractError("run receipt does not contain every required action")
    if receipt.get("trace_samples_observed") != expected_actions + 1:
        raise ExperimentContractError("run receipt does not contain every boundary sample")
    analysis = prepared.design.to_dict()["analysis"]
    if receipt.get("stock_environment_return_role") != "diagnostic_only":
        raise ExperimentContractError("stock environment return exceeded its diagnostic role")
    if (
        receipt.get("stock_environment_return_summation_rule")
        != analysis["stock_environment_return_summation_rule"]
    ):
        raise ExperimentContractError("stock environment return summation rule changed")
    if receipt.get("stock_environment_reward_count") != expected_actions:
        raise ExperimentContractError("stock environment return is missing step rewards")
    stock_return = receipt.get("stock_environment_return")
    if isinstance(stock_return, bool) or not isinstance(stock_return, (int, float)):
        raise ExperimentContractError("stock environment return must be finite")
    if not math.isfinite(float(stock_return)):
        raise ExperimentContractError("stock environment return must be finite")
    if receipt.get("oracle_contract_faults") != 0:
        raise ExperimentContractError("run receipt reports an oracle contract fault")
    if receipt.get("claim_status") != CLAIM_STATUS or any(
        receipt.get(field) is not False
        for field in (
            "automatic_promotion",
            "tier_d_claim",
            "causal_oracle_claim",
            "formal_oracle_evidence_claim",
        )
    ):
        raise ExperimentContractError("run receipt exceeds the local exploration boundary")
    trace = load_trace(trace_path)
    if len(trace.samples) != expected_actions + 1:
        raise ExperimentContractError("trace does not contain every required boundary sample")
    if trace.evidence_class is not EvidenceClass.EXPLORATORY:
        raise ExperimentContractError("trace evidence class exceeds exploratory")
    if trace.sha256 != receipt.get("trace_sha256"):
        raise ExperimentContractError("run trace hash does not reverify")
    if trace.artifact_bindings != _trace_bindings(prepared, manifest):
        raise ExperimentContractError("run trace artifact bindings differ from the manifest")


@contextmanager
def _atomic_study_directory(runs_dir: Path, *, final_name: str) -> Iterator[tuple[Path, Path]]:
    parent = Path(runs_dir).resolve()
    if parent.name != "runs":
        raise ExperimentContractError("study output parent must be named runs")
    parent.mkdir(parents=True, exist_ok=True)
    final = parent / final_name
    if final.exists():
        raise ExperimentContractError(f"refusing to overwrite existing study: {final}")
    staging = Path(tempfile.mkdtemp(prefix=f".{final_name}.", suffix=".pending", dir=parent))
    # Exceptions intentionally leave the private staging tree intact for failure retention.
    yield staging, final
    _rename_directory_without_replace(staging, final)


def _rename_directory_without_replace(source: Path, destination: Path) -> None:
    """Publish one directory with the host's atomic no-replace primitive."""

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        rename = libc.renamex_np
        rename.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename.restype = ctypes.c_int
        result = rename(source_bytes, destination_bytes, 0x00000004)
    elif hasattr(libc, "renameat2"):
        rename = libc.renameat2
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        result = rename(-100, source_bytes, -100, destination_bytes, 1)
    else:  # pragma: no cover - unsupported host safety boundary
        raise ExperimentContractError("host lacks an atomic no-replace directory rename")
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise ExperimentContractError(f"refusing to overwrite existing study: {destination}")
    raise ExperimentContractError(f"cannot publish study directory: {os.strerror(error_number)}")


def _contract_safe_failure_summary(error: BaseException) -> str:
    """Classify a failure without persisting paths, values, or a traceback."""

    if isinstance(error, KeyboardInterrupt):
        return "operator_interruption"
    if isinstance(error, SystemExit):
        return "system_exit_interruption"
    if isinstance(error, ExperimentContractError):
        return "experiment_contract_failure"
    if isinstance(error, OSError):
        return "host_io_failure"
    if isinstance(error, RuntimeError):
        return "runtime_failure"
    return "unexpected_execution_failure"


def _failed_study_name(*, success_name: str, staging: Path) -> str:
    attempt_id = hashlib.sha256(staging.name.encode("utf-8")).hexdigest()[:12]
    return f"{success_name}-failed-{attempt_id}"


def _hard_gate_status(
    design: OracleExplorationDesign,
    *,
    passed: Sequence[str],
    failed: Sequence[str],
) -> dict[str, list[str]]:
    declared = tuple(design.to_dict()["hard_gates"])
    passed_tuple = tuple(passed)
    failed_tuple = tuple(failed)
    if (
        len(set(passed_tuple)) != len(passed_tuple)
        or len(set(failed_tuple)) != len(failed_tuple)
        or set(passed_tuple) & set(failed_tuple)
        or not set(passed_tuple).issubset(declared)
        or not set(failed_tuple).issubset(declared)
    ):
        raise ExperimentContractError("hard-gate status is not a disjoint declared partition")
    resolved = set(passed_tuple) | set(failed_tuple)
    return {
        "declared_hard_gates": list(declared),
        "hard_gates_passed": list(passed_tuple),
        "hard_gates_failed": list(failed_tuple),
        "hard_gates_not_reached": [gate for gate in declared if gate not in resolved],
    }


def _failed_study_receipt(
    *,
    design: OracleExplorationDesign,
    manifest: OracleExplorationExecutionManifest,
    manifest_file_sha256: str,
    runs: Sequence[OracleExplorationRun],
    completed_count: int,
    completed_run_receipt_sha256: Sequence[str],
    active_run: OracleExplorationRun | None,
    failure_stage: str,
    lineage_manifest_sha256: str | None,
    hard_gates_passed: Sequence[str],
    hard_gates_failed: Sequence[str],
    error: BaseException,
) -> dict[str, object]:
    """Describe an incomplete attempt without interpreting partial outcomes."""

    if not 0 <= completed_count <= len(runs):
        raise ExperimentContractError("completed run count is outside the frozen table")
    if len(completed_run_receipt_sha256) != completed_count:
        raise ExperimentContractError("completed receipt hashes do not match completed runs")
    gate_status = _hard_gate_status(
        design,
        passed=hard_gates_passed,
        failed=hard_gates_failed,
    )
    return {
        "schema_version": STUDY_RECEIPT_SCHEMA_VERSION,
        "claim_status": CLAIM_STATUS,
        "evidence_class": "exploratory",
        "completion_status": "failed_attempt_retained_no_comparison",
        "design_sha256": design.sha256,
        "run_table_sha256": manifest.run_table_sha256,
        "execution_manifest_sha256": manifest.sha256,
        "execution_manifest_file_sha256": manifest_file_sha256,
        "expected_run_count": len(runs),
        "completed_run_count": completed_count,
        "completed_scheduled_runs": [run.to_dict() for run in runs[:completed_count]],
        "completed_run_receipt_file_sha256": list(completed_run_receipt_sha256),
        "failed_scheduled_run": active_run.to_dict() if active_run is not None else None,
        "missing_run_count": len(runs) - completed_count,
        "missing_scheduled_runs": [run.to_dict() for run in runs[completed_count:]],
        "failure_stage": failure_stage,
        "failure_class": type(error).__name__,
        "failure_summary": _contract_safe_failure_summary(error),
        "lineage_manifest_sha256": lineage_manifest_sha256,
        **gate_status,
        "pre_execution_hard_gates_complete": tuple(hard_gates_passed)
        == tuple(design.to_dict()["hard_gates"][:2]),
        "comparative_interpretation_allowed": False,
        "outcome_based_retry_allowed": False,
        "automatic_promotion": False,
        "tier_d_claim": False,
        "causal_oracle_claim": False,
        "formal_oracle_evidence_claim": False,
    }


def _claim_study_attempt(
    *,
    runs_dir: Path,
    manifest: OracleExplorationExecutionManifest,
    manifest_file_sha256: str,
) -> Path:
    """Claim this manifest once within one declared output root."""

    parent = Path(runs_dir).resolve()
    if parent.name != "runs":
        raise ExperimentContractError("study output parent must be named runs")
    parent.mkdir(parents=True, exist_ok=True)
    claim = parent / f".claim-local-v1-{manifest.sha256[:12]}.json"
    claim_receipt = {
        "schema_version": 1,
        "claim_status": CLAIM_STATUS,
        "execution_manifest_sha256": manifest.sha256,
        "execution_manifest_file_sha256": manifest_file_sha256,
        "outcome_based_retry_allowed": False,
        "enforcement_scope": "one_attempt_for_manifest_within_this_declared_output_root",
        "cross_root_duplicate_prevention_claim": False,
    }
    try:
        emit_json_without_overwrite(claim, claim_receipt)
    except ExperimentContractError as exc:
        if os.path.lexists(claim):
            raise ExperimentContractError(
                "this reviewed manifest already has an attempt in the declared output root"
            ) from exc
        raise
    return claim


def _emit_attempt_start_receipt(
    *,
    staging: Path,
    design: OracleExplorationDesign,
    manifest: OracleExplorationExecutionManifest,
    manifest_file_sha256: str,
) -> Path:
    gate_status = _hard_gate_status(design, passed=(), failed=())
    return emit_json_without_overwrite(
        staging / "attempt_start_receipt.json",
        {
            "schema_version": 1,
            "claim_status": CLAIM_STATUS,
            "attempt_status": "claimed_pre_execution",
            "design_sha256": design.sha256,
            "design_file_sha256": design.source_sha256,
            "run_table_sha256": manifest.run_table_sha256,
            "execution_manifest_sha256": manifest.sha256,
            "execution_manifest_file_sha256": manifest_file_sha256,
            "expected_run_count": len(design.run_table),
            "scheduled_runs": [run.to_dict() for run in design.run_table],
            **gate_status,
            "pre_execution_hard_gates_complete": False,
            "actions_executed": 0,
            "outcome_based_retry_allowed": False,
        },
    )


def run_oracle_exploration(
    *,
    design_path: Path,
    hdf5_path: Path,
    metadata_path: Path,
    base_controller_path: Path,
    residual_controller_path: Path,
    manifest_path: Path,
    runs_dir: Path,
) -> dict[str, object]:
    """Execute the complete frozen table once after every pre-action gate passes."""

    manifest, manifest_file_bytes, manifest_file_sha256 = (
        _load_oracle_exploration_manifest_snapshot(manifest_path)
    )
    if manifest.manifest_status != MANIFEST_REVIEWED_STATUS:
        raise ExperimentContractError("execution requires a reviewed frozen manifest")
    design_path = Path(design_path).absolute()
    design = load_oracle_exploration_design(design_path)
    if design.sha256 != manifest.design_sha256 or design.source_sha256 != (
        manifest.design_file_sha256
    ):
        raise ExperimentContractError("design file differs from the reviewed manifest")
    runs = design.run_table
    if len(runs) != 240 or run_table_sha256(runs) != manifest.run_table_sha256:
        raise ExperimentContractError("expanded run table differs from the reviewed manifest")
    claim_marker = _claim_study_attempt(
        runs_dir=runs_dir,
        manifest=manifest,
        manifest_file_sha256=manifest_file_sha256,
    )
    final_name = f"local-v1-{manifest.sha256[:12]}"
    receipts: list[dict[str, object]] = []
    run_receipt_file_sha256: list[str] = []
    active_run: OracleExplorationRun | None = None
    failure_stage = "study_output_initialization"
    staging_for_failure: Path | None = None
    lineage_manifest_sha256: str | None = None
    hard_gates_passed: tuple[str, ...] = ()
    hard_gates_failed: tuple[str, ...] = ()
    study_receipt: dict[str, object] | None = None
    try:
        with _atomic_study_directory(runs_dir, final_name=final_name) as (staging, final):
            staging_for_failure = staging
            _emit_attempt_start_receipt(
                staging=staging,
                design=design,
                manifest=manifest,
                manifest_file_sha256=manifest_file_sha256,
            )
            receipts_dir = staging / "runs"
            receipts_dir.mkdir()
            failure_stage = "pre_execution_validation"
            prepared = _prepare_oracle_exploration_from_design(
                design_path=design_path,
                design=design,
                hdf5_path=hdf5_path,
                metadata_path=metadata_path,
                base_controller_path=base_controller_path,
                residual_controller_path=residual_controller_path,
            )
            validate_prepared_manifest(prepared, manifest)
            hard_gates_passed = tuple(design.to_dict()["hard_gates"][:2])
            failure_stage = "lineage_materialization"
            _lineage_path, lineage_manifest_sha256 = _materialize_study_lineage(
                staging=staging,
                prepared=prepared,
                manifest=manifest,
                manifest_file_bytes=manifest_file_bytes,
                manifest_file_sha256=manifest_file_sha256,
            )
            failure_stage = "scheduled_run_execution_and_validation"
            for run in runs:
                active_run = run
                run_dir = receipts_dir / f"{run.run_order:03d}"
                run_dir.mkdir()
                receipt = _run_one(
                    prepared=prepared,
                    manifest=manifest,
                    run=run,
                    output_dir=run_dir,
                )
                persisted_source = read_bounded_json_artifact(
                    run_dir / "run_receipt.json",
                    maximum_bytes=MAX_EXECUTION_JSON_BYTES,
                    artifact="oracle exploration run receipt",
                )
                persisted_receipt = persisted_source.value
                if persisted_receipt != receipt:
                    raise ExperimentContractError("persisted run receipt differs from memory")
                _validate_completed_run_receipt(
                    receipt=persisted_receipt,
                    expected_run=run,
                    trace_path=run_dir / "trace.json",
                    prepared=prepared,
                    manifest=manifest,
                )
                receipts.append(receipt)
                run_receipt_file_sha256.append(persisted_source.sha256)
                active_run = None
            observed = [receipt["run"] for receipt in receipts]
            expected = [run.to_dict() for run in runs]
            if observed != expected:
                raise ExperimentContractError(
                    "study did not report every scheduled run exactly once"
                )
            hard_gates_passed = tuple(design.to_dict()["hard_gates"])
            failure_stage = "terminal_receipt_and_atomic_publication"
            success_gate_status = _hard_gate_status(
                design,
                passed=hard_gates_passed,
                failed=(),
            )
            study_receipt = {
                "schema_version": STUDY_RECEIPT_SCHEMA_VERSION,
                "claim_status": CLAIM_STATUS,
                "evidence_class": "exploratory",
                "completion_status": "complete_measurements_no_promotion",
                "design_sha256": prepared.design.sha256,
                "run_table_sha256": manifest.run_table_sha256,
                "execution_manifest_sha256": manifest.sha256,
                "execution_manifest_file_sha256": manifest_file_sha256,
                "expected_run_count": 240,
                "observed_run_count": len(receipts),
                "run_receipt_file_sha256": list(run_receipt_file_sha256),
                "lineage_manifest_sha256": lineage_manifest_sha256,
                **success_gate_status,
                "automatic_promotion": False,
                "tier_d_claim": False,
                "causal_oracle_claim": False,
                "formal_oracle_evidence_claim": False,
            }
            emit_json_without_overwrite(staging / "study_receipt.json", study_receipt)
    except BaseException as error:
        declared_hard_gates = tuple(design.to_dict()["hard_gates"])
        if failure_stage == "scheduled_run_execution_and_validation":
            hard_gates_failed = (declared_hard_gates[2],)
        failure_receipt = _failed_study_receipt(
            design=design,
            manifest=manifest,
            manifest_file_sha256=manifest_file_sha256,
            runs=runs,
            completed_count=len(receipts),
            completed_run_receipt_sha256=run_receipt_file_sha256,
            active_run=active_run,
            failure_stage=failure_stage,
            lineage_manifest_sha256=lineage_manifest_sha256,
            hard_gates_passed=hard_gates_passed,
            hard_gates_failed=hard_gates_failed,
            error=error,
        )
        if staging_for_failure is None or not staging_for_failure.is_dir():
            terminal_receipt_path = claim_marker.with_name(
                f"{claim_marker.stem}.terminal_failure.json"
            )
            try:
                emit_json_without_overwrite(
                    terminal_receipt_path,
                    failure_receipt,
                )
            except Exception as retention_error:
                raise ExperimentContractError(
                    f"v1 attempt failed; claim remains at {claim_marker}"
                ) from retention_error
            error.add_note(f"terminal v1 failure receipt retained at {terminal_receipt_path}")
            raise
        try:
            emit_json_without_overwrite(
                staging_for_failure / "failure_receipt.json",
                failure_receipt,
            )
            failed_final = Path(runs_dir).resolve() / _failed_study_name(
                success_name=final_name,
                staging=staging_for_failure,
            )
            _rename_directory_without_replace(staging_for_failure, failed_final)
        except Exception as retention_error:
            raise ExperimentContractError(
                f"v1 attempt failed; incomplete evidence remains at {staging_for_failure}"
            ) from retention_error
        error.add_note(f"immutable incomplete v1 evidence retained at {failed_final}")
        raise
    if study_receipt is None:
        raise ExperimentContractError("study completed without a terminal receipt")
    return {
        "study_directory": str(final),
        "study_receipt": study_receipt,
        "study_receipt_sha256": sha256_file(final / "study_receipt.json"),
    }


def _common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--hdf5", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--base-controller", type=Path, required=True)
    parser.add_argument("--residual-controller", type=Path, required=True)


def _prepared_from_args(args: argparse.Namespace) -> PreparedOracleExploration:
    return prepare_oracle_exploration(
        design_path=args.design,
        hdf5_path=args.hdf5,
        metadata_path=args.metadata,
        base_controller_path=args.base_controller,
        residual_controller_path=args.residual_controller,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="humanoid-oracle-exploration")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="validate inputs and emit a manifest candidate")
    _common_paths(prepare)
    prepare.add_argument("--output", type=Path, required=True)
    finalize = commands.add_parser("finalize", help="freeze an explicitly reviewed candidate")
    finalize.add_argument("--candidate", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--confirm-reviewed", action="store_true")
    run = commands.add_parser("run", help="execute the reviewed complete v1 table once")
    _common_paths(run)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--runs-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepared = _prepared_from_args(args)
        emit_pre_execution_receipts(args.output.parent, prepared)
        manifest = emit_manifest_candidate(args.output, prepared)
        print(json.dumps({"manifest_sha256": manifest.sha256, "status": manifest.manifest_status}))
        return 0
    if args.command == "finalize":
        confirmation = REVIEW_CONFIRMATION if args.confirm_reviewed else ""
        manifest = finalize_reviewed_manifest(
            candidate_path=args.candidate,
            output_path=args.output,
            review_confirmation=confirmation,
        )
        print(json.dumps({"manifest_sha256": manifest.sha256, "status": manifest.manifest_status}))
        return 0
    result = run_oracle_exploration(
        design_path=args.design,
        hdf5_path=args.hdf5,
        metadata_path=args.metadata,
        base_controller_path=args.base_controller,
        residual_controller_path=args.residual_controller,
        manifest_path=args.manifest,
        runs_dir=args.runs_dir,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - module CLI
    raise SystemExit(main())


__all__ = [
    "CLAIM_STATUS",
    "MANIFEST_CANDIDATE_STATUS",
    "MANIFEST_REVIEWED_STATUS",
    "REVIEW_CONFIRMATION",
    "OracleExplorationExecutionManifest",
    "OracleExplorationRuntimeReceipt",
    "PreparedOracleExploration",
    "compose_controller_action",
    "emit_manifest_candidate",
    "emit_pre_execution_receipts",
    "finalize_reviewed_manifest",
    "inspect_oracle_exploration_runtime",
    "load_oracle_exploration_manifest",
    "main",
    "manifest_from_prepared",
    "materialize_executed_action",
    "prepare_oracle_exploration",
    "run_oracle_exploration",
    "runner_source_sha256",
    "validate_prepared_manifest",
]
