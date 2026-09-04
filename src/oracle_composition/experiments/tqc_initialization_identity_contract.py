"""Pure contracts for TQC initialization-identity checks."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .fixed_reference import ExperimentContractError, read_bounded_json_artifact
from .tqc_calibration_contract import canonical_json, sha256_json
from .tqc_initialization_identity_v1 import (
    ACTION_SAMPLING_SEED,
    ACTOR_INITIALIZATION_SEED,
    DEPENDENCY_LOCK_SHA256,
    DESIGN_ARTIFACT_BYTE_COUNT,
    DESIGN_ARTIFACT_SHA256,
    E0_ACTION_SPACE_SHA256,
    E0_CALIBRATION_ID,
    E0_DESIGN_ARTIFACT_SHA256,
    E0_OBSERVATION_SPACE_SHA256,
    E0_RECEIPT_SHA256,
    E0_RUNTIME_SHA256,
    EXPANDED_OBSERVATION_SPACE_SHA256,
    EXPANDED_SCRATCH_INITIALIZATION_SEED,
    FIXTURE_ADAPTER_SOURCE_SHA256,
    FIXTURE_CONTRACT_SOURCE_SHA256,
    FIXTURE_RUNNER_SOURCE_SHA256,
    NUMPY_VERSION,
    OBSERVATION_SET_SHA256,
    REFERENCE_SET_SHA256,
    RESIDUAL_LOG_STD,
    RESIDUAL_SAMPLING_SEED,
    RESIDUAL_SCALE,
    SB3_BASE_POLICY_SOURCE_SHA256,
    SB3_CONTRIB_VERSION,
    SB3_DISTRIBUTION_SOURCE_SHA256,
    SB3_TORCH_LAYERS_SOURCE_SHA256,
    STABLE_BASELINES3_VERSION,
    TORCH_VERSION,
    TQC_POLICY_SOURCE_SHA256,
)

DESIGN_SCHEMA_VERSION = 1
FIXTURE_ID = "tqc_actor_expansion_transfer_fixture/v1"
ACTUAL_E1_GATE_ID = "tqc_initialization_identity/v1"
MAX_DESIGN_BYTES = 128 * 1024
MISSING_CONTROLLER_BLOCKER = "missing_hash_pinned_trained_tqc_actor_bytes"
UNEXECUTED_INITIALIZATIONS_BLOCKER = (
    "zero_residual_and_expanded_initializations_not_executed_on_trained_actor"
)
_FLOAT32 = np.dtype("<f4")


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ExperimentContractError(
            f"{field} keys differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ExperimentContractError(f"{field} must be an object")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ExperimentContractError(f"{field} must be nonempty bounded text")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ExperimentContractError(f"{field} must be an integer >= {minimum}")
    return value


def _finite_float(value: object, *, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ExperimentContractError(f"{field} must be a finite JSON float")
    return value


def _sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"{field} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class E0Requirement:
    calibration_id: str
    design_artifact_sha256: str
    receipt_sha256: str
    runtime_sha256: str
    seed: int
    seed_role: str
    environment_id: str
    observation_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    observation_space_sha256: str
    action_space_sha256: str
    expected_environment_steps: int
    expected_vector_steps: int
    expected_gradient_updates: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> E0Requirement:
        _require_exact_keys(raw, set(cls.__dataclass_fields__), field="required_e0")
        observation_shape = raw["observation_shape"]
        action_shape = raw["action_shape"]
        if not isinstance(observation_shape, list) or not isinstance(action_shape, list):
            raise ExperimentContractError("required_e0 shapes must be lists")
        value = cls(
            calibration_id=_text(raw["calibration_id"], field="required_e0.calibration_id"),
            design_artifact_sha256=_sha256(
                raw["design_artifact_sha256"], field="required_e0.design_artifact_sha256"
            ),
            receipt_sha256=_sha256(raw["receipt_sha256"], field="required_e0.receipt_sha256"),
            runtime_sha256=_sha256(raw["runtime_sha256"], field="required_e0.runtime_sha256"),
            seed=_integer(raw["seed"], field="required_e0.seed", minimum=1),
            seed_role=_text(raw["seed_role"], field="required_e0.seed_role"),
            environment_id=_text(raw["environment_id"], field="required_e0.environment_id"),
            observation_shape=tuple(
                _integer(item, field="required_e0.observation_shape[]", minimum=1)
                for item in observation_shape
            ),
            action_shape=tuple(
                _integer(item, field="required_e0.action_shape[]", minimum=1)
                for item in action_shape
            ),
            observation_space_sha256=_sha256(
                raw["observation_space_sha256"],
                field="required_e0.observation_space_sha256",
            ),
            action_space_sha256=_sha256(
                raw["action_space_sha256"], field="required_e0.action_space_sha256"
            ),
            expected_environment_steps=_integer(
                raw["expected_environment_steps"],
                field="required_e0.expected_environment_steps",
                minimum=1,
            ),
            expected_vector_steps=_integer(
                raw["expected_vector_steps"],
                field="required_e0.expected_vector_steps",
                minimum=1,
            ),
            expected_gradient_updates=_integer(
                raw["expected_gradient_updates"],
                field="required_e0.expected_gradient_updates",
                minimum=1,
            ),
        )
        fixed = {
            "calibration_id": E0_CALIBRATION_ID,
            "design_artifact_sha256": E0_DESIGN_ARTIFACT_SHA256,
            "receipt_sha256": E0_RECEIPT_SHA256,
            "runtime_sha256": E0_RUNTIME_SHA256,
            "seed": 92_001,
            "seed_role": "permanently_excluded_disposable_calibration",
            "environment_id": "Humanoid-v5",
            "observation_shape": (348,),
            "action_shape": (17,),
            "observation_space_sha256": E0_OBSERVATION_SPACE_SHA256,
            "action_space_sha256": E0_ACTION_SPACE_SHA256,
            "expected_environment_steps": 100_000,
            "expected_vector_steps": 20_000,
            "expected_gradient_updates": 19_980,
        }
        for field, expected in fixed.items():
            observed = getattr(value, field)
            if type(observed) is not type(expected) or observed != expected:
                raise ExperimentContractError(f"required_e0.{field} must equal {expected!r}")
        return value


@dataclass(frozen=True, slots=True)
class FixtureDesign:
    actor_initialization_seed: int
    expanded_scratch_initialization_seed: int
    action_sampling_seed: int
    residual_sampling_seed: int
    observation_batch_size: int
    base_observation_dim: int
    reference_window_shape: tuple[int, int]
    action_dim: int
    hidden_layers: tuple[int, ...]
    dtype: str
    activation: str
    use_sde: bool
    observation_construction: str
    reference_construction: str
    observation_set_sha256: str
    reference_set_sha256: str
    expanded_observation_space_sha256: str
    residual_distribution: str
    residual_log_std: float
    residual_scale: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> FixtureDesign:
        _require_exact_keys(raw, set(cls.__dataclass_fields__), field="fixture")
        window = raw["reference_window_shape"]
        hidden = raw["hidden_layers"]
        if not isinstance(window, list) or len(window) != 2:
            raise ExperimentContractError("fixture.reference_window_shape must contain H and D")
        if not isinstance(hidden, list) or not hidden or len(hidden) > 8:
            raise ExperimentContractError("fixture.hidden_layers must be a bounded list")
        use_sde = raw["use_sde"]
        if not isinstance(use_sde, bool):
            raise ExperimentContractError("fixture.use_sde must be boolean")
        value = cls(
            actor_initialization_seed=_integer(
                raw["actor_initialization_seed"],
                field="fixture.actor_initialization_seed",
                minimum=1,
            ),
            expanded_scratch_initialization_seed=_integer(
                raw["expanded_scratch_initialization_seed"],
                field="fixture.expanded_scratch_initialization_seed",
                minimum=1,
            ),
            action_sampling_seed=_integer(
                raw["action_sampling_seed"], field="fixture.action_sampling_seed", minimum=1
            ),
            residual_sampling_seed=_integer(
                raw["residual_sampling_seed"],
                field="fixture.residual_sampling_seed",
                minimum=1,
            ),
            observation_batch_size=_integer(
                raw["observation_batch_size"],
                field="fixture.observation_batch_size",
                minimum=2,
            ),
            base_observation_dim=_integer(
                raw["base_observation_dim"], field="fixture.base_observation_dim", minimum=1
            ),
            reference_window_shape=(
                _integer(window[0], field="fixture.reference_window_shape[0]", minimum=2),
                _integer(window[1], field="fixture.reference_window_shape[1]", minimum=1),
            ),
            action_dim=_integer(raw["action_dim"], field="fixture.action_dim", minimum=1),
            hidden_layers=tuple(
                _integer(item, field="fixture.hidden_layers[]", minimum=1) for item in hidden
            ),
            dtype=_text(raw["dtype"], field="fixture.dtype"),
            activation=_text(raw["activation"], field="fixture.activation"),
            use_sde=use_sde,
            observation_construction=_text(
                raw["observation_construction"], field="fixture.observation_construction"
            ),
            reference_construction=_text(
                raw["reference_construction"], field="fixture.reference_construction"
            ),
            observation_set_sha256=_sha256(
                raw["observation_set_sha256"], field="fixture.observation_set_sha256"
            ),
            reference_set_sha256=_sha256(
                raw["reference_set_sha256"], field="fixture.reference_set_sha256"
            ),
            expanded_observation_space_sha256=_sha256(
                raw["expanded_observation_space_sha256"],
                field="fixture.expanded_observation_space_sha256",
            ),
            residual_distribution=_text(
                raw["residual_distribution"], field="fixture.residual_distribution"
            ),
            residual_log_std=_finite_float(
                raw["residual_log_std"], field="fixture.residual_log_std"
            ),
            residual_scale=_finite_float(raw["residual_scale"], field="fixture.residual_scale"),
        )
        fixed = {
            "actor_initialization_seed": ACTOR_INITIALIZATION_SEED,
            "expanded_scratch_initialization_seed": EXPANDED_SCRATCH_INITIALIZATION_SEED,
            "action_sampling_seed": ACTION_SAMPLING_SEED,
            "residual_sampling_seed": RESIDUAL_SAMPLING_SEED,
            "observation_batch_size": 4,
            "base_observation_dim": 348,
            "reference_window_shape": (8, 45),
            "action_dim": 17,
            "hidden_layers": (256, 256),
            "dtype": "float32",
            "activation": "torch.nn.ReLU",
            "use_sde": False,
            "observation_construction": "integer_affine_grid_float32/v1",
            "reference_construction": "integer_affine_grid_float32/v1",
            "observation_set_sha256": OBSERVATION_SET_SHA256,
            "reference_set_sha256": REFERENCE_SET_SHA256,
            "expanded_observation_space_sha256": EXPANDED_OBSERVATION_SPACE_SHA256,
            "residual_distribution": "synthetic_zero_mean_squashed_diagonal_gaussian/v1",
            "residual_log_std": RESIDUAL_LOG_STD,
            "residual_scale": RESIDUAL_SCALE,
        }
        for field, expected in fixed.items():
            observed = getattr(value, field)
            if type(observed) is not type(expected) or observed != expected:
                raise ExperimentContractError(f"fixture.{field} must equal {expected!r}")
        return value

    @property
    def reference_input_dim(self) -> int:
        return self.reference_window_shape[0] * self.reference_window_shape[1]

    @property
    def expanded_observation_dim(self) -> int:
        return self.base_observation_dim + self.reference_input_dim


@dataclass(frozen=True, slots=True)
class ActionTransform:
    normalized_low: float
    normalized_high: float
    physical_low: float
    physical_high: float
    method: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ActionTransform:
        _require_exact_keys(raw, set(cls.__dataclass_fields__), field="action_transform")
        value = cls(
            normalized_low=_finite_float(
                raw["normalized_low"], field="action_transform.normalized_low"
            ),
            normalized_high=_finite_float(
                raw["normalized_high"], field="action_transform.normalized_high"
            ),
            physical_low=_finite_float(raw["physical_low"], field="action_transform.physical_low"),
            physical_high=_finite_float(
                raw["physical_high"], field="action_transform.physical_high"
            ),
            method=_text(raw["method"], field="action_transform.method"),
        )
        if value != cls(
            normalized_low=-1.0,
            normalized_high=1.0,
            physical_low=-0.4,
            physical_high=0.4,
            method="stable_baselines3.BasePolicy.unscale_action/v2.9",
        ):
            raise ExperimentContractError("action_transform differs from the frozen Humanoid ABI")
        return value


@dataclass(frozen=True, slots=True)
class IdentityTolerance:
    absolute: float
    relative: float

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> IdentityTolerance:
        _require_exact_keys(raw, set(cls.__dataclass_fields__), field="tolerance")
        value = cls(
            absolute=_finite_float(raw["absolute"], field="tolerance.absolute"),
            relative=_finite_float(raw["relative"], field="tolerance.relative"),
        )
        if value.absolute != 1e-6 or value.relative != 1e-6:
            raise ExperimentContractError("tolerance must equal the reviewed 1e-6/1e-6 bound")
        return value


@dataclass(frozen=True, slots=True)
class RuntimeRequirement:
    required_numpy_version: str
    required_torch_version: str
    required_stable_baselines3_version: str
    required_sb3_contrib_version: str
    dependency_lock_sha256: str
    tqc_policy_source_sha256: str
    sb3_distribution_source_sha256: str
    sb3_base_policy_source_sha256: str
    sb3_torch_layers_source_sha256: str
    fixture_adapter_source_sha256: str
    fixture_contract_source_sha256: str
    fixture_runner_source_sha256: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> RuntimeRequirement:
        _require_exact_keys(raw, set(cls.__dataclass_fields__), field="runtime")
        value = cls(
            required_numpy_version=_text(
                raw["required_numpy_version"], field="runtime.required_numpy_version"
            ),
            required_torch_version=_text(
                raw["required_torch_version"], field="runtime.required_torch_version"
            ),
            required_stable_baselines3_version=_text(
                raw["required_stable_baselines3_version"],
                field="runtime.required_stable_baselines3_version",
            ),
            required_sb3_contrib_version=_text(
                raw["required_sb3_contrib_version"],
                field="runtime.required_sb3_contrib_version",
            ),
            dependency_lock_sha256=_sha256(
                raw["dependency_lock_sha256"], field="runtime.dependency_lock_sha256"
            ),
            tqc_policy_source_sha256=_sha256(
                raw["tqc_policy_source_sha256"], field="runtime.tqc_policy_source_sha256"
            ),
            sb3_distribution_source_sha256=_sha256(
                raw["sb3_distribution_source_sha256"],
                field="runtime.sb3_distribution_source_sha256",
            ),
            sb3_base_policy_source_sha256=_sha256(
                raw["sb3_base_policy_source_sha256"],
                field="runtime.sb3_base_policy_source_sha256",
            ),
            sb3_torch_layers_source_sha256=_sha256(
                raw["sb3_torch_layers_source_sha256"],
                field="runtime.sb3_torch_layers_source_sha256",
            ),
            fixture_adapter_source_sha256=_sha256(
                raw["fixture_adapter_source_sha256"],
                field="runtime.fixture_adapter_source_sha256",
            ),
            fixture_contract_source_sha256=_sha256(
                raw["fixture_contract_source_sha256"],
                field="runtime.fixture_contract_source_sha256",
            ),
            fixture_runner_source_sha256=_sha256(
                raw["fixture_runner_source_sha256"],
                field="runtime.fixture_runner_source_sha256",
            ),
        )
        fixed = {
            "required_numpy_version": NUMPY_VERSION,
            "required_torch_version": TORCH_VERSION,
            "required_stable_baselines3_version": STABLE_BASELINES3_VERSION,
            "required_sb3_contrib_version": SB3_CONTRIB_VERSION,
            "dependency_lock_sha256": DEPENDENCY_LOCK_SHA256,
            "tqc_policy_source_sha256": TQC_POLICY_SOURCE_SHA256,
            "sb3_distribution_source_sha256": SB3_DISTRIBUTION_SOURCE_SHA256,
            "sb3_base_policy_source_sha256": SB3_BASE_POLICY_SOURCE_SHA256,
            "sb3_torch_layers_source_sha256": SB3_TORCH_LAYERS_SOURCE_SHA256,
            "fixture_adapter_source_sha256": FIXTURE_ADAPTER_SOURCE_SHA256,
            "fixture_contract_source_sha256": FIXTURE_CONTRACT_SOURCE_SHA256,
            "fixture_runner_source_sha256": FIXTURE_RUNNER_SOURCE_SHA256,
        }
        for field, expected in fixed.items():
            observed = getattr(value, field)
            if type(observed) is not type(expected) or observed != expected:
                raise ExperimentContractError(f"runtime.{field} must equal {expected!r}")
        return value


@dataclass(frozen=True, slots=True)
class InitializationIdentityDesign:
    schema_version: int
    fixture_id: str
    design_status: str
    evidence_class: str
    claim_ceiling: str
    required_e0: E0Requirement
    fixture: FixtureDesign
    action_transform: ActionTransform
    tolerance: IdentityTolerance
    runtime: RuntimeRequirement

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> InitializationIdentityDesign:
        _require_exact_keys(raw, set(cls.__dataclass_fields__), field="design")
        value = cls(
            schema_version=_integer(raw["schema_version"], field="schema_version", minimum=1),
            fixture_id=_text(raw["fixture_id"], field="fixture_id"),
            design_status=_text(raw["design_status"], field="design_status"),
            evidence_class=_text(raw["evidence_class"], field="evidence_class"),
            claim_ceiling=_text(raw["claim_ceiling"], field="claim_ceiling"),
            required_e0=E0Requirement.from_dict(_mapping(raw["required_e0"], field="required_e0")),
            fixture=FixtureDesign.from_dict(_mapping(raw["fixture"], field="fixture")),
            action_transform=ActionTransform.from_dict(
                _mapping(raw["action_transform"], field="action_transform")
            ),
            tolerance=IdentityTolerance.from_dict(_mapping(raw["tolerance"], field="tolerance")),
            runtime=RuntimeRequirement.from_dict(_mapping(raw["runtime"], field="runtime")),
        )
        fixed = {
            "schema_version": DESIGN_SCHEMA_VERSION,
            "fixture_id": FIXTURE_ID,
            "design_status": "synthetic_fixture_not_controller_evidence",
            "evidence_class": "software_identity_fixture",
            "claim_ceiling": "transfer_implementation_identity_only",
        }
        for field, expected in fixed.items():
            observed = getattr(value, field)
            if type(observed) is not type(expected) or observed != expected:
                raise ExperimentContractError(f"{field} must equal {expected!r}")
        if value.fixture.base_observation_dim != value.required_e0.observation_shape[0]:
            raise ExperimentContractError("fixture observation width differs from E0")
        if value.fixture.action_dim != value.required_e0.action_shape[0]:
            raise ExperimentContractError("fixture action width differs from E0")
        return value

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["required_e0"]["observation_shape"] = list(self.required_e0.observation_shape)
        value["required_e0"]["action_shape"] = list(self.required_e0.action_shape)
        value["fixture"]["reference_window_shape"] = list(self.fixture.reference_window_shape)
        value["fixture"]["hidden_layers"] = list(self.fixture.hidden_layers)
        return value

    @property
    def semantic_sha256(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class LoadedInitializationIdentityDesign:
    design: InitializationIdentityDesign
    artifact_sha256: str
    artifact_byte_count: int


def require_canonical_initialization_identity_design(
    design: InitializationIdentityDesign,
) -> None:
    if not isinstance(design, InitializationIdentityDesign):
        raise ExperimentContractError("initialization-identity design has the wrong type")
    validated = InitializationIdentityDesign.from_dict(design.to_dict())
    if validated != design:
        raise ExperimentContractError("initialization-identity design is not canonical v1")


def load_initialization_identity_design(path: Path) -> LoadedInitializationIdentityDesign:
    artifact = read_bounded_json_artifact(
        path,
        maximum_bytes=MAX_DESIGN_BYTES,
        artifact="TQC initialization-identity fixture design",
    )
    if artifact.sha256 != DESIGN_ARTIFACT_SHA256:
        raise ExperimentContractError(
            "TQC initialization-identity v1 design bytes differ from the compiled identity"
        )
    if len(artifact.encoded_bytes) != DESIGN_ARTIFACT_BYTE_COUNT:
        raise ExperimentContractError(
            "TQC initialization-identity v1 design byte count differs from the compiled identity"
        )
    return LoadedInitializationIdentityDesign(
        design=InitializationIdentityDesign.from_dict(artifact.value),
        artifact_sha256=artifact.sha256,
        artifact_byte_count=len(artifact.encoded_bytes),
    )


def float32_matrix(value: object, *, shape: tuple[int, int], field: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ExperimentContractError(f"{field} must be an ndarray")
    if value.dtype.str != _FLOAT32.str or value.shape != shape or not value.flags.c_contiguous:
        raise ExperimentContractError(f"{field} must be a C-order little-endian float32 matrix")
    if not np.isfinite(value).all():
        raise ExperimentContractError(f"{field} contains non-finite values")
    return value


def array_sha256(value: np.ndarray) -> str:
    if not isinstance(value, np.ndarray) or not value.flags.c_contiguous:
        raise ExperimentContractError("hashed array must use C-order storage")
    metadata = canonical_json({"dtype": value.dtype.str, "shape": list(value.shape)})
    raw = value.tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def array_receipt(value: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "sha256": array_sha256(value),
        "values": value.tolist(),
    }


def normalized_to_physical(value: np.ndarray, transform: ActionTransform) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype="<f4")
    if not np.isfinite(array).all():
        raise ExperimentContractError("normalized action contains non-finite values")
    if np.any(array < transform.normalized_low) or np.any(array > transform.normalized_high):
        raise ExperimentContractError("normalized action is outside [-1, 1]")
    low = np.float32(transform.physical_low)
    high = np.float32(transform.physical_high)
    physical = low + np.float32(0.5) * (array + np.float32(1.0)) * (high - low)
    return np.ascontiguousarray(physical, dtype="<f4")


def _comparison(
    left: np.ndarray,
    right: np.ndarray,
    *,
    tolerance: IdentityTolerance,
) -> dict[str, Any]:
    if left.shape != right.shape or left.dtype.str != right.dtype.str:
        raise ExperimentContractError("identity comparison arrays differ in shape or dtype")
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    scale = np.maximum(np.abs(left.astype(np.float64)), np.abs(right.astype(np.float64)))
    allowed = tolerance.absolute + tolerance.relative * scale
    return {
        "left_sha256": array_sha256(left),
        "right_sha256": array_sha256(right),
        "bitwise_equal": left.tobytes(order="C") == right.tobytes(order="C"),
        "within_tolerance": bool(np.all(delta <= allowed)),
        "maximum_absolute_difference": float(np.max(delta, initial=0.0)),
        "maximum_allowed_difference": float(np.max(allowed, initial=tolerance.absolute)),
    }


def verify_initialization_identity_arrays(
    *,
    design: InitializationIdentityDesign,
    base_mean: np.ndarray,
    base_log_std: np.ndarray,
    base_sampling_noise: np.ndarray,
    base_deterministic: np.ndarray,
    base_sampled: np.ndarray,
    base_physical: np.ndarray,
    base_sampled_physical: np.ndarray,
    expanded_mean: np.ndarray,
    expanded_log_std: np.ndarray,
    expanded_deterministic: np.ndarray,
    expanded_sampled: np.ndarray,
    expanded_physical: np.ndarray,
    expanded_sampled_physical: np.ndarray,
    alternate_reference_mean: np.ndarray,
    alternate_reference_log_std: np.ndarray,
    alternate_reference_deterministic: np.ndarray,
    alternate_reference_sampled: np.ndarray,
    alternate_reference_physical: np.ndarray,
    alternate_reference_sampled_physical: np.ndarray,
    residual_mean: np.ndarray,
    residual_log_std: np.ndarray,
    residual_sampling_noise: np.ndarray,
    residual_deterministic: np.ndarray,
    residual_sampled: np.ndarray,
    zero_residual_composed: np.ndarray,
    zero_residual_physical: np.ndarray,
    sampled_residual_composed: np.ndarray,
    sampled_residual_physical: np.ndarray,
) -> dict[str, Any]:
    """Verify one batch without granting the trained-controller E1 gate."""

    require_canonical_initialization_identity_design(design)
    shape = (design.fixture.observation_batch_size, design.fixture.action_dim)
    arrays = {
        name: float32_matrix(value, shape=shape, field=name)
        for name, value in {
            "base_mean": base_mean,
            "base_log_std": base_log_std,
            "base_sampling_noise": base_sampling_noise,
            "base_deterministic": base_deterministic,
            "base_sampled": base_sampled,
            "base_physical": base_physical,
            "base_sampled_physical": base_sampled_physical,
            "expanded_mean": expanded_mean,
            "expanded_log_std": expanded_log_std,
            "expanded_deterministic": expanded_deterministic,
            "expanded_sampled": expanded_sampled,
            "expanded_physical": expanded_physical,
            "expanded_sampled_physical": expanded_sampled_physical,
            "alternate_reference_mean": alternate_reference_mean,
            "alternate_reference_log_std": alternate_reference_log_std,
            "alternate_reference_deterministic": alternate_reference_deterministic,
            "alternate_reference_sampled": alternate_reference_sampled,
            "alternate_reference_physical": alternate_reference_physical,
            "alternate_reference_sampled_physical": alternate_reference_sampled_physical,
            "residual_mean": residual_mean,
            "residual_log_std": residual_log_std,
            "residual_sampling_noise": residual_sampling_noise,
            "residual_deterministic": residual_deterministic,
            "residual_sampled": residual_sampled,
            "zero_residual_composed": zero_residual_composed,
            "zero_residual_physical": zero_residual_physical,
            "sampled_residual_composed": sampled_residual_composed,
            "sampled_residual_physical": sampled_residual_physical,
        }.items()
    }
    tolerance = design.tolerance
    formula_checks = {
        "base_deterministic_from_mean": _comparison(
            arrays["base_deterministic"],
            np.ascontiguousarray(np.tanh(arrays["base_mean"]), dtype="<f4"),
            tolerance=tolerance,
        ),
        "base_seeded_sample_from_distribution": _comparison(
            arrays["base_sampled"],
            np.ascontiguousarray(
                np.tanh(
                    arrays["base_mean"]
                    + np.exp(arrays["base_log_std"]) * arrays["base_sampling_noise"]
                ),
                dtype="<f4",
            ),
            tolerance=tolerance,
        ),
        "expanded_deterministic_from_mean": _comparison(
            arrays["expanded_deterministic"],
            np.ascontiguousarray(np.tanh(arrays["expanded_mean"]), dtype="<f4"),
            tolerance=tolerance,
        ),
        "expanded_seeded_sample_from_distribution": _comparison(
            arrays["expanded_sampled"],
            np.ascontiguousarray(
                np.tanh(
                    arrays["expanded_mean"]
                    + np.exp(arrays["expanded_log_std"]) * arrays["base_sampling_noise"]
                ),
                dtype="<f4",
            ),
            tolerance=tolerance,
        ),
        "alternate_reference_deterministic_from_mean": _comparison(
            arrays["alternate_reference_deterministic"],
            np.ascontiguousarray(np.tanh(arrays["alternate_reference_mean"]), dtype="<f4"),
            tolerance=tolerance,
        ),
        "alternate_reference_seeded_sample_from_distribution": _comparison(
            arrays["alternate_reference_sampled"],
            np.ascontiguousarray(
                np.tanh(
                    arrays["alternate_reference_mean"]
                    + np.exp(arrays["alternate_reference_log_std"]) * arrays["base_sampling_noise"]
                ),
                dtype="<f4",
            ),
            tolerance=tolerance,
        ),
        "base_physical_transform": _comparison(
            arrays["base_physical"],
            normalized_to_physical(arrays["base_deterministic"], design.action_transform),
            tolerance=tolerance,
        ),
        "expanded_physical_transform": _comparison(
            arrays["expanded_physical"],
            normalized_to_physical(arrays["expanded_deterministic"], design.action_transform),
            tolerance=tolerance,
        ),
        "base_sampled_physical_transform": _comparison(
            arrays["base_sampled_physical"],
            normalized_to_physical(arrays["base_sampled"], design.action_transform),
            tolerance=tolerance,
        ),
        "expanded_sampled_physical_transform": _comparison(
            arrays["expanded_sampled_physical"],
            normalized_to_physical(arrays["expanded_sampled"], design.action_transform),
            tolerance=tolerance,
        ),
        "alternate_reference_physical_transform": _comparison(
            arrays["alternate_reference_physical"],
            normalized_to_physical(
                arrays["alternate_reference_deterministic"], design.action_transform
            ),
            tolerance=tolerance,
        ),
        "alternate_reference_sampled_physical_transform": _comparison(
            arrays["alternate_reference_sampled_physical"],
            normalized_to_physical(arrays["alternate_reference_sampled"], design.action_transform),
            tolerance=tolerance,
        ),
        "zero_residual_physical_transform": _comparison(
            arrays["zero_residual_physical"],
            normalized_to_physical(arrays["zero_residual_composed"], design.action_transform),
            tolerance=tolerance,
        ),
        "sampled_residual_physical_transform": _comparison(
            arrays["sampled_residual_physical"],
            normalized_to_physical(arrays["sampled_residual_composed"], design.action_transform),
            tolerance=tolerance,
        ),
    }
    zero_bytes = np.zeros(shape, dtype="<f4").tobytes(order="C")
    if arrays["residual_mean"].tobytes(order="C") != zero_bytes:
        raise ExperimentContractError("synthetic residual mean is not exact positive zero")
    if arrays["residual_deterministic"].tobytes(order="C") != zero_bytes:
        raise ExperimentContractError("synthetic deterministic residual is not exact positive zero")
    expected_log_std = np.full(shape, design.fixture.residual_log_std, dtype="<f4")
    if arrays["residual_log_std"].tobytes(order="C") != expected_log_std.tobytes(order="C"):
        raise ExperimentContractError("synthetic residual log standard deviation drifted")
    if not np.any(arrays["residual_sampled"] != 0.0):
        raise ExperimentContractError("seeded stochastic residual unexpectedly equals zero")
    formula_checks["residual_seeded_sample_from_distribution"] = _comparison(
        arrays["residual_sampled"],
        np.ascontiguousarray(
            np.tanh(
                arrays["residual_mean"]
                + np.exp(arrays["residual_log_std"]) * arrays["residual_sampling_noise"]
            ),
            dtype="<f4",
        ),
        tolerance=tolerance,
    )
    expected_sampled_composition = np.ascontiguousarray(
        np.clip(
            arrays["base_deterministic"]
            + np.float32(design.fixture.residual_scale) * arrays["residual_sampled"],
            -1.0,
            1.0,
        ),
        dtype="<f4",
    )
    formula_checks["sampled_residual_composition"] = _comparison(
        arrays["sampled_residual_composed"],
        expected_sampled_composition,
        tolerance=tolerance,
    )

    comparisons = {
        "expanded_mean_vs_base": _comparison(
            arrays["expanded_mean"], arrays["base_mean"], tolerance=tolerance
        ),
        "expanded_log_std_vs_base": _comparison(
            arrays["expanded_log_std"], arrays["base_log_std"], tolerance=tolerance
        ),
        "expanded_deterministic_vs_base": _comparison(
            arrays["expanded_deterministic"],
            arrays["base_deterministic"],
            tolerance=tolerance,
        ),
        "expanded_seeded_sample_vs_base": _comparison(
            arrays["expanded_sampled"], arrays["base_sampled"], tolerance=tolerance
        ),
        "expanded_physical_control_vs_base": _comparison(
            arrays["expanded_physical"], arrays["base_physical"], tolerance=tolerance
        ),
        "expanded_sampled_physical_control_vs_base": _comparison(
            arrays["expanded_sampled_physical"],
            arrays["base_sampled_physical"],
            tolerance=tolerance,
        ),
        "alternate_reference_mean_vs_base": _comparison(
            arrays["alternate_reference_mean"], arrays["base_mean"], tolerance=tolerance
        ),
        "alternate_reference_log_std_vs_base": _comparison(
            arrays["alternate_reference_log_std"],
            arrays["base_log_std"],
            tolerance=tolerance,
        ),
        "alternate_reference_deterministic_vs_base": _comparison(
            arrays["alternate_reference_deterministic"],
            arrays["base_deterministic"],
            tolerance=tolerance,
        ),
        "alternate_reference_seeded_sample_vs_base": _comparison(
            arrays["alternate_reference_sampled"],
            arrays["base_sampled"],
            tolerance=tolerance,
        ),
        "alternate_reference_physical_vs_base": _comparison(
            arrays["alternate_reference_physical"],
            arrays["base_physical"],
            tolerance=tolerance,
        ),
        "alternate_reference_sampled_physical_vs_base": _comparison(
            arrays["alternate_reference_sampled_physical"],
            arrays["base_sampled_physical"],
            tolerance=tolerance,
        ),
        "zero_residual_composition_vs_base": _comparison(
            arrays["zero_residual_composed"],
            arrays["base_deterministic"],
            tolerance=tolerance,
        ),
        "zero_residual_physical_vs_base": _comparison(
            arrays["zero_residual_physical"], arrays["base_physical"], tolerance=tolerance
        ),
    }
    if not all(item["within_tolerance"] for item in formula_checks.values()):
        raise ExperimentContractError("action/distribution formula check exceeded tolerance")
    if not all(item["within_tolerance"] for item in comparisons.values()):
        raise ExperimentContractError("initialization identity comparison exceeded tolerance")
    if not comparisons["zero_residual_composition_vs_base"]["bitwise_equal"]:
        raise ExperimentContractError(
            "zero-residual deterministic composition is not bitwise exact"
        )
    if not comparisons["zero_residual_physical_vs_base"]["bitwise_equal"]:
        raise ExperimentContractError("zero-residual physical control is not bitwise exact")

    return {
        "fixture_identity_checks_passed": True,
        "tolerances": asdict(tolerance),
        "formula_checks": formula_checks,
        "comparisons": comparisons,
        "stochastic_residual_is_nonzero": True,
        "claim_boundary": "synthetic_transfer_implementation_only_not_trained_controller_E1/v1",
    }


__all__ = [
    "ACTUAL_E1_GATE_ID",
    "DESIGN_SCHEMA_VERSION",
    "FIXTURE_ID",
    "MAX_DESIGN_BYTES",
    "MISSING_CONTROLLER_BLOCKER",
    "UNEXECUTED_INITIALIZATIONS_BLOCKER",
    "ActionTransform",
    "E0Requirement",
    "FixtureDesign",
    "IdentityTolerance",
    "InitializationIdentityDesign",
    "LoadedInitializationIdentityDesign",
    "RuntimeRequirement",
    "array_receipt",
    "array_sha256",
    "float32_matrix",
    "load_initialization_identity_design",
    "normalized_to_physical",
    "require_canonical_initialization_identity_design",
    "verify_initialization_identity_arrays",
]
