"""Pure fail-closed contracts for the Phase B fine-tuning runtime."""

from __future__ import annotations

import hashlib
import json
import math
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from types import MappingProxyType

from oracle_composition.contracts.reference_identity_v2 import (
    canonical_json_bytes,
    require_sha256,
    sha256_json,
)
from oracle_composition.harness.contract import (
    EVIDENCE_CLASS as PHASE_A_EVIDENCE_CLASS,
)
from oracle_composition.harness.contract import (
    OracleContractError,
    OracleProgram,
    oracle_program_from_dict,
    read_json_object,
)
from oracle_composition.rewards import target_speed_formula as target_speed_formula_module
from oracle_composition.rewards import target_speed_formula_t2 as target_speed_formula_t2_module
from oracle_composition.rewards import task_inputs_v2 as task_inputs_v2_module
from oracle_composition.rewards.target_speed_formula import FORMULA_ID as TARGET_SPEED_RECIPE_ID
from oracle_composition.rewards.target_speed_formula_t2 import (
    FORMULA_RUNTIME_ID as TARGET_SPEED_FORMULA_ID,
)
from oracle_composition.rewards.target_speed_formula_t2 import PARSER_ID as TARGET_SPEED_PARSER_ID
from oracle_composition.rewards.task_inputs_v2 import (
    TASK_INPUTS_V2_SCHEMA_ID,
    task_input_contract_v2,
)
from oracle_composition.tracking.reward import TrackingRewardConfig

ORACLE_SCHEMA_ID = "humanoid_reference_composition_oracle/v1"
REWARD_SCHEMA_ID = "reward_specification/tracking_only/v1"
TARGET_SPEED_REWARD_SCHEMA_ID = "reward_specification/target_speed_triangular_affine_t2/v1"
REPORT_V1_SCHEMA_ID = "humanoid_composition_cycle_report/v1"
REPORT_V2_SCHEMA_ID = "humanoid_composition_cycle_report/v2"
STARTING_CHECKPOINT_SCHEMA_ID = "humanoid_fine_tuning_starting_checkpoint/v1"
RUN_MANIFEST_SCHEMA_ID = "humanoid_fine_tuning_run_manifest/v2"
EVIDENCE_CLASS = "interface_check"
CLAIM_CEILING = (
    "exploratory_reference_conditioned_fine_tuning_utility_only_"
    "no_causal_reference_use_oracle_improvement_reward_improvement_"
    "generalization_naturalness_or_humanoid_competence_claim"
)
EXPERT_ACTOR_NPZ_SHA256 = "60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b"
TASK_INPUTS_V2_SOURCE_SHA256 = "9607f2d56a54922eac06ec7fcc740b79e68d4ed9e88b192602aae2900fb3f0d2"
TASK_INPUTS_V2_SCHEMA_SHA256 = "8f382dde13ee44c27cbbbc0b3a53a568338f6b7cebc8e660e4084425b7b494e9"
TASK_INPUT_ADMISSION_SOURCE_SHA256 = (
    "db63c68693232a95995a3867e230d6d83ee1d31ba7072405d9377a38ebbede44"
)
TASK_INPUT_ADMISSION_SCHEMA_SHA256 = (
    "c2d8908533a47e76e5ec469c80b6cf2d7061397d3716597ec7c66b4f8913b4b5"
)
_LEGACY_PHASE_B_ORACLE_SHA256 = "4d24f22780360d7632235572d97b3fccfe7c376162e69e15082f77bd7afcccf1"
REVIEWED_RUN_MANIFEST_SHA256 = "45ce5baa66b6d6dadba38c61bbdfc943a75c542ae7c355927f1031fc65661ebb"
SMOKE_SEEDS = (121901,)
COHORT_SEEDS = (121001, 121101, 121201, 121301, 121401)
SMOKE_TRANSITIONS = 196_608
COHORT_TRANSITIONS = 1_048_576
FINAL_CHECKPOINT_RULE = "final_transition_only"
T2_TRAINING_DESIGN_SCHEMA_ID = "humanoid_fine_tuning_training_design_t2/v1"
T2_RSI_FAMILY_ID = "expert_reference_nine_block_rsi/v1"
T2_REWARD_PAIRING_ID = "t2_reward_pairing/v1"
T2_REWARD_PAIRING_DERIVATION_ID = "t2_reward_study_rng_substream/v1"
NON_PAIRED_ID = "non_paired"
PAIRING_EXCLUDED_ARM_FIELDS = frozenset({"arm_label", "output_path", "reward", "timestamps"})
_T2_PAIRING_STUDY_FIELDS = frozenset(
    {
        "arms",
        "integrated_pairing_receipt_sha256",
        "schema_version",
        "status",
        "study_id",
        "study_manifest_schema_id",
        "study_pairing_sha256",
    }
)
_T2_PAIRING_COMMON_ARM_FIELDS = frozenset(
    {
        "calibration",
        "claim_ceiling",
        "evaluation",
        "evaluator",
        "evidence_class",
        "execution_manifest",
        "library",
        "oracle",
        "pairing",
        "pairing_adapter",
        "reference_corpus",
        "starting_checkpoint",
        "task",
        "training",
        "training_design",
    }
)
_T2_PAIRING_ARM_FIELDS = _T2_PAIRING_COMMON_ARM_FIELDS | PAIRING_EXCLUDED_ARM_FIELDS
T2_TRAINING_BLOCKS = (
    120001,
    120002,
    120003,
    120005,
    120007,
    120008,
    120009,
    120011,
    120012,
)

PHASE_POLICY = MappingProxyType(
    {
        "action_target_offset_rows": 1,
        "candidate_phase_bound": "0<=j<=task_step",
        "continuous_rematching": False,
        "error_scales": {
            "joint_position_rmse_rad": 0.35,
            "joint_velocity_rmse_rad_s": 2.0,
            "root_angular_velocity_rmse_rad_s": 2.0,
            "root_height_abs_error_m": 0.20,
            "root_linear_velocity_rmse_m_s": 1.0,
            "root_orientation_error_rad": 0.50,
        },
        "hold_advance_rows_per_step": 1,
        "lookahead": "none_before_live_guard",
        "phase_selection": "nearest_state_float64_lexicographic/v1",
        "score_order": [
            "maximum_normalized_error",
            "sum_squared_normalized_errors",
            "phase_index",
        ],
        "selection_events": [
            "behavior_change",
            "recovery_entry",
            "rejoin",
        ],
        "terminal_hold": "reference_end_only",
        "window_shape": [8, 45],
        "wrap": False,
    }
)

TRACKING_ONLY_FORMULA_ID = "tracking_only/v1"
TRACKING_ONLY_PARSER_ID = "no_candidate_inputs/v1"
REWARD_COMPOSITOR_ID = "tracking_plus_task_stock_telemetry/v1"
TRACKING_REWARD_ID = "humanoid_root_and_joint_tracking/v1"
FROZEN_TRACKING_REWARD_CONFIG = TrackingRewardConfig(
    root_height_scale_m=0.20,
    root_orientation_scale_rad=0.50,
    root_linear_velocity_scale_m_s=1.0,
    root_angular_velocity_scale_rad_s=2.0,
    joint_position_scale_rad=0.35,
    joint_velocity_scale_rad_s=2.0,
    root_linear_velocity_weight=0.10,
    root_angular_velocity_weight=0.10,
    joint_position_weight=0.50,
    joint_velocity_weight=0.30,
)
TRACKING_REWARD_CONFIG_SHA256 = FROZEN_TRACKING_REWARD_CONFIG.sha256
REWARD_SCHEMA_SHA256 = sha256_json(
    {
        "required_fields": [
            "compositor_id",
            "compositor_sha256",
            "evidence_class",
            "formula_id",
            "formula_sha256",
            "parameter_bounds_sha256",
            "parameters",
            "parser_id",
            "parser_sha256",
            "r_task",
            "reward_schema_id",
            "schema_sha256",
            "schema_version",
            "stock_reward",
            "task_inputs_schema_id",
            "task_inputs_schema_sha256",
            "task_inputs_source_sha256",
            "task_input_admission_id",
            "task_input_admission_contract",
            "task_input_admission_schema_sha256",
            "task_input_admission_source_sha256",
            "tracking_reward_config_sha256",
            "tracking_reward_id",
        ],
        "reward_schema_id": REWARD_SCHEMA_ID,
        "schema_version": 1,
    }
)
TRACKING_ONLY_FORMULA_SHA256 = sha256_json({"formula_id": TRACKING_ONLY_FORMULA_ID, "r_task": 0.0})
TRACKING_ONLY_PARSER_SHA256 = sha256_json(
    {"accepted_fields": [], "parser_id": TRACKING_ONLY_PARSER_ID}
)
TRACKING_ONLY_BOUNDS_SHA256 = sha256_json(
    {"parameters": {}, "r_task": {"maximum": 0.0, "minimum": 0.0}}
)
TARGET_SPEED_FORMULA_SHA256 = "064393887bb4a7157d614981cc2000940252e12c31ad0aabc13109737fd94301"
TARGET_SPEED_PARSER_SHA256 = "c60dea03e6f0e71b81875fea59c84bd8fe00ce39f94ac7c54ca6e2a57360dccb"
TARGET_SPEED_BOUNDS_BYTES = (
    b'{"parameters":{"alpha":{"maximum":4.0,"minimum":0.25},'
    b'"beta":{"maximum":10.0,"minimum":-10.0}},'
    b'"r_task":{"maximum":15.0,"minimum":-10.0}}'
)
TARGET_SPEED_BOUNDS_SHA256 = "2c6030264231a52320a44fe0f4d4ed519bc2e635b892f1b9c0d8929166106933"
TARGET_SPEED_REWARD_SCHEMA_SHA256 = sha256_json(
    {
        "required_fields": [
            "compositor_id",
            "compositor_sha256",
            "evidence_class",
            "formula_id",
            "formula_sha256",
            "internal_recipe_id",
            "output_envelope",
            "parameter_bounds",
            "parameter_bounds_sha256",
            "parameters",
            "parser_callable",
            "parser_id",
            "parser_sha256",
            "required_consumption_step",
            "reward_schema_id",
            "schema_sha256",
            "schema_version",
            "stock_reward",
            "task_inputs_schema_id",
            "task_inputs_schema_sha256",
            "task_inputs_source_sha256",
            "tracking_reward_config_sha256",
            "tracking_reward_id",
        ],
        "reward_schema_id": TARGET_SPEED_REWARD_SCHEMA_ID,
        "schema_version": 1,
    }
)
REWARD_COMPOSITOR_SHA256 = sha256_json(
    {
        "formula": "r_train=r_track+r_task",
        "id": REWARD_COMPOSITOR_ID,
        "stock_reward": "telemetry_only",
        "tracking_reward_config_sha256": TRACKING_REWARD_CONFIG_SHA256,
        "tracking_reward_id": TRACKING_REWARD_ID,
    }
)


class PhaseBContractError(ValueError):
    """Raised before model construction when a Phase B artifact differs."""


def _exact_mapping(value: object, expected: set[str], *, field: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected:
        raise PhaseBContractError(f"{field} keys differ from the contract")
    return value


def _nonempty_text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or len(value) > 512:
        raise PhaseBContractError(f"{field} must be nonempty bounded text")
    return value


def _integer(
    value: object,
    *,
    field: str,
    minimum: int = 0,
    maximum: int = 2_147_483_647,
) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise PhaseBContractError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def _finite(value: object, *, field: str) -> float:
    if type(value) not in {int, float}:
        raise PhaseBContractError(f"{field} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise PhaseBContractError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise PhaseBContractError(f"{field} must be finite")
    return result


def _sha(value: object, *, field: str) -> str:
    try:
        return require_sha256(value, field=field)
    except ValueError as exc:
        raise PhaseBContractError(str(exc)) from exc


def _require_canonical_file(path: Path) -> tuple[dict[str, object], bytes]:
    try:
        value, encoded = read_json_object(Path(path))
    except OracleContractError as exc:
        raise PhaseBContractError(str(exc)) from exc
    try:
        canonical = canonical_json_bytes(value)
    except ValueError as exc:
        raise PhaseBContractError(str(exc)) from exc
    if encoded != canonical:
        raise PhaseBContractError(f"{path} is not canonical JSON")
    return value, encoded


@dataclass(frozen=True, slots=True)
class PhaseBOracleProgram:
    """The Phase A state-machine grammar plus the immutable transfer policy."""

    program: OracleProgram

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def to_dict(self) -> dict[str, object]:
        result = self.program.to_dict()
        result["evidence_class"] = EVIDENCE_CLASS
        result["oracle_schema_id"] = ORACLE_SCHEMA_ID
        result["phase_policy"] = dict(PHASE_POLICY)
        return result

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
        *,
        available_behaviors: Sequence[str],
        allow_legacy_recovery: bool = False,
    ) -> PhaseBOracleProgram:
        required = {
            "behaviors",
            "evidence_class",
            "initial",
            "oracle_id",
            "oracle_schema_id",
            "phase_policy",
            "schema_version",
            "states",
            "transitions",
        }
        optional = {"recovery"}
        if (
            type(value) is not dict
            or not required.issubset(value)
            or not set(value).issubset(required | optional)
        ):
            raise PhaseBContractError("oracle keys differ from the Phase B contract")
        if value["oracle_schema_id"] != ORACLE_SCHEMA_ID or value["schema_version"] != 1:
            raise PhaseBContractError("oracle schema identity differs")
        if value["evidence_class"] != EVIDENCE_CLASS:
            raise PhaseBContractError(f"oracle evidence_class must be {EVIDENCE_CLASS}")
        if value["phase_policy"] != dict(PHASE_POLICY):
            raise PhaseBContractError("oracle phase policy differs from the frozen policy")
        phase_a = {
            key: item
            for key, item in value.items()
            if key not in {"oracle_schema_id", "phase_policy"}
        }
        phase_a["evidence_class"] = PHASE_A_EVIDENCE_CLASS
        try:
            program = oracle_program_from_dict(
                phase_a,
                available_behaviors=available_behaviors,
                allow_legacy_recovery=allow_legacy_recovery,
            )
        except OracleContractError as exc:
            raise PhaseBContractError(str(exc)) from exc
        return cls(program=program)


def load_phase_b_oracle(
    path: Path,
    *,
    available_behaviors: Sequence[str],
) -> tuple[PhaseBOracleProgram, str]:
    value, encoded = _require_canonical_file(path)
    raw_sha256 = hashlib.sha256(encoded).hexdigest()
    program = PhaseBOracleProgram.from_dict(
        value,
        available_behaviors=available_behaviors,
        allow_legacy_recovery=raw_sha256 == _LEGACY_PHASE_B_ORACLE_SHA256,
    )
    if program.canonical_bytes != encoded:
        raise PhaseBContractError("oracle bytes differ from canonical contract serialization")
    return program, raw_sha256


@dataclass(frozen=True, slots=True)
class TrackingOnlyRewardSpec:
    """The sole reward admitted before Astra supplies a reviewed V2."""

    @property
    def registry_key(self) -> tuple[str, ...]:
        return (
            REWARD_SCHEMA_SHA256,
            TRACKING_ONLY_FORMULA_SHA256,
            TRACKING_ONLY_PARSER_SHA256,
            TRACKING_ONLY_BOUNDS_SHA256,
            REWARD_COMPOSITOR_SHA256,
            TASK_INPUTS_V2_SOURCE_SHA256,
            TASK_INPUTS_V2_SCHEMA_SHA256,
            TASK_INPUT_ADMISSION_SOURCE_SHA256,
            TASK_INPUT_ADMISSION_SCHEMA_SHA256,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def to_dict(self) -> dict[str, object]:
        from .task_input_admission import task_input_admission_contract

        return {
            "compositor_id": REWARD_COMPOSITOR_ID,
            "compositor_sha256": REWARD_COMPOSITOR_SHA256,
            "evidence_class": EVIDENCE_CLASS,
            "formula_id": TRACKING_ONLY_FORMULA_ID,
            "formula_sha256": TRACKING_ONLY_FORMULA_SHA256,
            "parameter_bounds_sha256": TRACKING_ONLY_BOUNDS_SHA256,
            "parameters": {},
            "parser_id": TRACKING_ONLY_PARSER_ID,
            "parser_sha256": TRACKING_ONLY_PARSER_SHA256,
            "r_task": 0.0,
            "reward_schema_id": REWARD_SCHEMA_ID,
            "schema_sha256": REWARD_SCHEMA_SHA256,
            "schema_version": 1,
            "stock_reward": "telemetry_only",
            "task_inputs_schema_id": TASK_INPUTS_V2_SCHEMA_ID,
            "task_inputs_schema_sha256": TASK_INPUTS_V2_SCHEMA_SHA256,
            "task_inputs_source_sha256": TASK_INPUTS_V2_SOURCE_SHA256,
            "task_input_admission_id": "phase_b_stock_com_task_input_admission/v1",
            "task_input_admission_contract": task_input_admission_contract(),
            "task_input_admission_schema_sha256": TASK_INPUT_ADMISSION_SCHEMA_SHA256,
            "task_input_admission_source_sha256": TASK_INPUT_ADMISSION_SOURCE_SHA256,
            "tracking_reward_config_sha256": TRACKING_REWARD_CONFIG_SHA256,
            "tracking_reward_id": TRACKING_REWARD_ID,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TrackingOnlyRewardSpec:
        expected = cls().to_dict()
        if type(value) is not dict or value != expected:
            unknown = value.get("formula_id") if type(value) is dict else None
            if unknown != TRACKING_ONLY_FORMULA_ID:
                raise PhaseBContractError(f"unknown reward formula_id: {unknown!r}")
            raise PhaseBContractError("tracking-only reward specification differs")
        return cls()


@dataclass(frozen=True, slots=True)
class TargetSpeedRewardSpec:
    """Reviewed F2 formula binding with flat, bounded candidate parameters."""

    alpha: float
    beta: float

    def __post_init__(self) -> None:
        alpha = _finite(self.alpha, field="parameters.alpha")
        beta = _finite(self.beta, field="parameters.beta")
        if not 0.25 <= alpha <= 4.0:
            raise PhaseBContractError("target-speed alpha is outside [0.25, 4.0]")
        if not -10.0 <= beta <= 10.0:
            raise PhaseBContractError("target-speed beta is outside [-10.0, 10.0]")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "beta", beta)

    @property
    def registry_key(self) -> tuple[str, ...]:
        return (
            TARGET_SPEED_REWARD_SCHEMA_SHA256,
            TARGET_SPEED_FORMULA_SHA256,
            TARGET_SPEED_PARSER_SHA256,
            TARGET_SPEED_BOUNDS_SHA256,
            REWARD_COMPOSITOR_SHA256,
            TASK_INPUTS_V2_SOURCE_SHA256,
            TASK_INPUTS_V2_SCHEMA_SHA256,
            TASK_INPUT_ADMISSION_SOURCE_SHA256,
            TASK_INPUT_ADMISSION_SCHEMA_SHA256,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def to_dict(self) -> dict[str, object]:
        from .task_input_admission import task_input_admission_contract

        bounds = {
            "parameters": {
                "alpha": {"maximum": 4.0, "minimum": 0.25},
                "beta": {"maximum": 10.0, "minimum": -10.0},
            },
            "r_task": {"maximum": 15.0, "minimum": -10.0},
        }
        return {
            "compositor_id": REWARD_COMPOSITOR_ID,
            "compositor_sha256": REWARD_COMPOSITOR_SHA256,
            "evidence_class": "exploratory_fine_tuning_cycle",
            "formula_id": TARGET_SPEED_FORMULA_ID,
            "formula_sha256": TARGET_SPEED_FORMULA_SHA256,
            "internal_recipe_id": TARGET_SPEED_RECIPE_ID,
            "output_envelope": [-10.0, 15.0],
            "parameter_bounds": bounds,
            "parameter_bounds_sha256": TARGET_SPEED_BOUNDS_SHA256,
            "parameters": {"alpha": self.alpha, "beta": self.beta},
            "parser_callable": "parse_target_speed_formula_recipe",
            "parser_id": TARGET_SPEED_PARSER_ID,
            "parser_sha256": TARGET_SPEED_PARSER_SHA256,
            "required_consumption_step": {
                "contract": task_input_admission_contract(),
                "schema_sha256": TASK_INPUT_ADMISSION_SCHEMA_SHA256,
                "source_sha256": TASK_INPUT_ADMISSION_SOURCE_SHA256,
                "validator": "validate_admitted_task_inputs_v2",
            },
            "reward_schema_id": TARGET_SPEED_REWARD_SCHEMA_ID,
            "schema_sha256": TARGET_SPEED_REWARD_SCHEMA_SHA256,
            "schema_version": 1,
            "stock_reward": "telemetry_only",
            "task_inputs_schema_id": TASK_INPUTS_V2_SCHEMA_ID,
            "task_inputs_schema_sha256": TASK_INPUTS_V2_SCHEMA_SHA256,
            "task_inputs_source_sha256": TASK_INPUTS_V2_SOURCE_SHA256,
            "tracking_reward_config_sha256": TRACKING_REWARD_CONFIG_SHA256,
            "tracking_reward_id": TRACKING_REWARD_ID,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TargetSpeedRewardSpec:
        if type(value) is not dict or type(value.get("parameters")) is not dict:
            raise PhaseBContractError("target-speed reward specification differs")
        parameters = value["parameters"]
        if set(parameters) != {"alpha", "beta"}:
            raise PhaseBContractError("target-speed parameters must be flat alpha and beta")
        spec = cls(alpha=parameters["alpha"], beta=parameters["beta"])
        expected = spec.to_dict()
        if value != expected:
            if value.get("formula_sha256") != TARGET_SPEED_FORMULA_SHA256:
                raise PhaseBContractError("target-speed formula source identity differs")
            if canonical_json_bytes(value.get("parameter_bounds")) != TARGET_SPEED_BOUNDS_BYTES:
                raise PhaseBContractError("target-speed canonical bounds bytes differ")
            if "required_consumption_step" not in value:
                raise PhaseBContractError(
                    "target-speed task-input admission certificate is missing"
                )
            raise PhaseBContractError("target-speed reward specification differs")
        if canonical_json_bytes(expected["parameter_bounds"]) != TARGET_SPEED_BOUNDS_BYTES:
            raise PhaseBContractError("target-speed canonical bounds bytes differ")
        return spec


class RewardRegistry:
    """Immutable exact-key registry; unknown reward identities abort."""

    def __init__(self) -> None:
        source_path = Path(task_inputs_v2_module.__file__ or "")
        try:
            source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise PhaseBContractError("task-input V2 source is unavailable") from exc
        if source_sha256 != TASK_INPUTS_V2_SOURCE_SHA256:
            raise PhaseBContractError("task-input V2 source identity differs")
        if sha256_json(task_input_contract_v2()) != TASK_INPUTS_V2_SCHEMA_SHA256:
            raise PhaseBContractError("task-input V2 schema identity differs")
        from . import task_input_admission as admission_module

        admission_path = Path(admission_module.__file__ or "")
        try:
            admission_source_sha256 = hashlib.sha256(admission_path.read_bytes()).hexdigest()
        except OSError as exc:
            raise PhaseBContractError("task-input admission source is unavailable") from exc
        if admission_source_sha256 != TASK_INPUT_ADMISSION_SOURCE_SHA256:
            raise PhaseBContractError("task-input admission source identity differs")
        if (
            sha256_json(admission_module.task_input_admission_contract())
            != TASK_INPUT_ADMISSION_SCHEMA_SHA256
        ):
            raise PhaseBContractError("task-input admission schema identity differs")
        baseline = TrackingOnlyRewardSpec()
        for module, expected, field in (
            (target_speed_formula_t2_module, TARGET_SPEED_FORMULA_SHA256, "formula"),
            (target_speed_formula_module, TARGET_SPEED_PARSER_SHA256, "parser"),
        ):
            source = Path(module.__file__ or "")
            try:
                observed = hashlib.sha256(source.read_bytes()).hexdigest()
            except OSError as exc:
                raise PhaseBContractError(f"target-speed {field} source is unavailable") from exc
            if observed != expected:
                raise PhaseBContractError(f"target-speed {field} source identity differs")
        if (
            len(TARGET_SPEED_BOUNDS_BYTES) != 137
            or hashlib.sha256(TARGET_SPEED_BOUNDS_BYTES).hexdigest() != TARGET_SPEED_BOUNDS_SHA256
        ):
            raise PhaseBContractError("target-speed bounds authority differs")
        target = TargetSpeedRewardSpec(alpha=1.0, beta=0.0)
        self._entries = MappingProxyType(
            {
                baseline.registry_key: TrackingOnlyRewardSpec,
                target.registry_key: TargetSpeedRewardSpec,
            }
        )

    def resolve(
        self, value: Mapping[str, object]
    ) -> TrackingOnlyRewardSpec | TargetSpeedRewardSpec:
        formula_id = value.get("formula_id") if type(value) is dict else None
        if formula_id == TRACKING_ONLY_FORMULA_ID:
            spec: TrackingOnlyRewardSpec | TargetSpeedRewardSpec = TrackingOnlyRewardSpec.from_dict(
                value
            )
        elif formula_id == TARGET_SPEED_FORMULA_ID:
            spec = TargetSpeedRewardSpec.from_dict(value)
        else:
            raise PhaseBContractError(f"unknown reward formula_id: {formula_id!r}")
        try:
            expected_type = self._entries[spec.registry_key]
        except KeyError as exc:  # pragma: no cover - exact parser makes this defensive
            raise PhaseBContractError("reward registry key is unknown") from exc
        if type(spec) is not expected_type:
            raise PhaseBContractError("reward registry entry type differs")
        return spec

    def load(self, path: Path) -> tuple[TrackingOnlyRewardSpec | TargetSpeedRewardSpec, str]:
        value, encoded = _require_canonical_file(path)
        spec = self.resolve(value)
        if spec.canonical_bytes != encoded:
            raise PhaseBContractError("reward bytes differ from canonical serialization")
        return spec, hashlib.sha256(encoded).hexdigest()


def _artifact_binding(value: object, *, field: str) -> dict[str, object]:
    item = _exact_mapping(value, {"byte_count", "path", "sha256"}, field=field)
    _nonempty_text(item["path"], field=f"{field}.path")
    _integer(item["byte_count"], field=f"{field}.byte_count", minimum=1)
    _sha(item["sha256"], field=f"{field}.sha256")
    return item


@dataclass(frozen=True, slots=True)
class StartingCheckpointContract:
    source_expert: Mapping[str, object]
    strict_actor_export: Mapping[str, object]
    e1_receipt: Mapping[str, object]
    value_initialization_seed: int

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> StartingCheckpointContract:
        item = _exact_mapping(
            value,
            {
                "actor_input_layout",
                "e1_receipt",
                "evidence_class",
                "normalizer",
                "optimizer_constructed_after_copy",
                "ortho_init",
                "schema_version",
                "source_expert",
                "starting_checkpoint_schema_id",
                "strict_actor_export",
                "value_architecture",
                "value_initialization_seed",
            },
            field="starting checkpoint",
        )
        fixed = {
            "actor_input_layout": "state_float32_348_then_reference_row_major_float32_8x45/v1",
            "evidence_class": EVIDENCE_CLASS,
            "normalizer": None,
            "optimizer_constructed_after_copy": True,
            "ortho_init": False,
            "schema_version": 1,
            "starting_checkpoint_schema_id": STARTING_CHECKPOINT_SCHEMA_ID,
            "value_architecture": "independent_708_256_256_1_relu/v1",
        }
        for field, expected in fixed.items():
            if item[field] != expected:
                raise PhaseBContractError(f"starting checkpoint {field} differs")
        source = _artifact_binding(item["source_expert"], field="source_expert")
        export = _artifact_binding(item["strict_actor_export"], field="strict_actor_export")
        receipt = _artifact_binding(item["e1_receipt"], field="e1_receipt")
        seed = _integer(
            item["value_initialization_seed"],
            field="value_initialization_seed",
            minimum=1,
        )
        if source["sha256"] != EXPERT_ACTOR_NPZ_SHA256:
            raise PhaseBContractError("starting checkpoint expert identity differs")
        return cls(source, export, receipt, seed)


def training_design_contract_value() -> dict[str, object]:
    return {
        "actor_unfreeze": {
            "first_rollouts": 8,
            "initial_trainable": ["reference_columns", "value_network"],
            "then_trainable": "full_actor_and_value_network",
        },
        "checkpoint_selection": FINAL_CHECKPOINT_RULE,
        "cohort_seeds": list(COHORT_SEEDS),
        "environments": {
            "count": 4,
            "implementation": "DummyVecEnv",
            "stream_mix": {"composition": 2, "rehearsal": 2},
        },
        "evidence_class": "exploratory_fine_tuning_cycle",
        "normalization": {"observation": False, "reward": False},
        "ppo": {
            "batch_size": 512,
            "clip_range": 0.2,
            "clip_range_vf": None,
            "ent_coef": 0.0,
            "gae_lambda": 0.95,
            "gamma": 0.99,
            "learning_rate": 0.0003,
            "max_grad_norm": 0.5,
            "n_epochs": 10,
            "normalize_advantage": True,
            "target_kl": None,
            "vf_coef": 0.5,
        },
        "retries_or_seed_replacement": False,
        "rollout": {
            "rollout_count": 128,
            "steps_per_environment": 2048,
            "transitions_per_rollout": 8192,
        },
        "rsi": {
            "balanced_origin_actor_cells": 27,
            "schedule_classes": ["hold", "one_way", "round_trip"],
            "start_boundary_hash_modulus": 489,
        },
        "schema_version": 1,
        "training_blocks": [
            120001,
            120002,
            120003,
            120005,
            120007,
            120008,
            120009,
            120011,
            120012,
        ],
        "training_design_schema_id": "humanoid_fine_tuning_training_design/v1",
        "transitions_per_seed": COHORT_TRANSITIONS,
    }


def t2_training_design_contract_value() -> dict[str, object]:
    """Return the exact expert-only T2 design without changing the T1 contract."""

    value = training_design_contract_value()
    value["environments"] = {
        "count": 4,
        "implementation": "DummyVecEnv",
        "stream_by_environment": [
            "composition",
            "composition",
            "rehearsal",
            "rehearsal",
        ],
        "stream_contracts": {
            "composition": {
                "environment_indices": [0, 1],
                "oracle_program_id": "expert_hold_v1",
                "oracle_variant": "expert_hold/v1",
                "reset": "expert_start_boundary_0",
            },
            "rehearsal": {
                "environment_indices": [2, 3],
                "reference_behavior": "expert",
                "rsi_family": T2_RSI_FAMILY_ID,
            },
        },
        "stream_mix": {"composition": 2, "rehearsal": 2},
    }
    value["pairing"] = {
        "declared": True,
        "key_field": "study_pairing_sha256",
        "runtime_requirement": "reviewed_integrated_pairing_receipt",
        "stream_derivation_id": "t2_reward_study_rng_substream/v1",
    }
    value["rsi"] = {
        "environment_indices": [2, 3],
        "reference_behavior": "expert",
        "rsi_family": T2_RSI_FAMILY_ID,
        "schedule_class": "hold",
        "start_boundary_hash_modulus": 489,
        "training_blocks": list(T2_TRAINING_BLOCKS),
    }
    value["training_design_schema_id"] = T2_TRAINING_DESIGN_SCHEMA_ID
    return value


def validate_training_design(value: Mapping[str, object]) -> dict[str, object]:
    """Admit exactly the frozen T1 design or the expert-only T2 design."""

    if type(value) is not dict:
        raise PhaseBContractError("training design must be an object")
    observed = dict(value)
    if observed not in (training_design_contract_value(), t2_training_design_contract_value()):
        raise PhaseBContractError("training design semantics differ")
    return observed


def utility_evaluation_design_contract_value() -> dict[str, object]:
    return {
        "cell_episode_counts": {
            "fixed_round_trip": 20,
            "hold_expert": 20,
            "hold_medium": 20,
            "hold_simple": 20,
        },
        "cell_pass_minimum": 16,
        "claim_ceiling": "bounded utility only",
        "error_threshold": 1.0,
        "evaluation_blocks": list(range(120101, 120121)),
        "failure_denominator": "all_predeclared_episodes",
        "family_checkpoint_minimum": 4,
        "family_checkpoint_total": 5,
        "reference_resynchronization_steps": 64,
        "schema_version": 2,
        "step_zero_comparator_required": True,
        "task_success_calibration": {
            "calibration_block_ids": list(range(120201, 120221)),
            "calibration_policy_seed_ids": [122001, 122101, 122201, 122301, 122401],
            "censoring": {
                "fall_or_never_settled_latency_steps": 65,
                "maximum_admissible_latency_cap_steps": 64,
            },
            "empirical_quantile": 0.95,
            "quantile_method": "higher",
            "receipt_schema_id": "phase_b_task_success_calibration_receipt/v1",
            "segment_multiplicative_safety_margin": 1.1,
            "settled_state_multiplicative_safety_margin": 1.1,
            "transition_latency_additive_safety_margin_steps": 4,
        },
        "utility_evaluation_schema_id": "humanoid_fine_tuning_utility_evaluation/v2",
    }


def _canonical_pairing_object(encoded: object, *, field_name: str) -> dict[str, object]:
    if type(encoded) is not bytes or not encoded or len(encoded) > 256 * 1024:
        raise PhaseBContractError(f"{field_name} bytes are unavailable or oversized")
    try:
        value = json.loads(encoded)
    except (UnicodeError, ValueError) as exc:
        raise PhaseBContractError(f"{field_name} bytes are not JSON") from exc
    if type(value) is not dict or canonical_json_bytes(value) != encoded:
        raise PhaseBContractError(f"{field_name} bytes are not one canonical JSON object")
    return value


def _validate_t2_pairing_common_fields(common: Mapping[str, object]) -> None:
    for field in (
        "evaluator",
        "execution_manifest",
        "library",
        "oracle",
        "pairing_adapter",
        "reference_corpus",
        "starting_checkpoint",
        "training_design",
    ):
        _artifact_binding(common.get(field), field=f"pairing {field}")
    task = _exact_mapping(
        common.get("task"),
        {"environment_id", "expert_start", "horizon_steps", "target_com_forward_speed_m_s"},
        field="pairing task",
    )
    _nonempty_text(task["environment_id"], field="pairing task.environment_id")
    if type(task["expert_start"]) is not bool:
        raise PhaseBContractError("pairing task.expert_start must be a boolean")
    _integer(task["horizon_steps"], field="pairing task.horizon_steps", minimum=1)
    _finite(
        task["target_com_forward_speed_m_s"],
        field="pairing task.target_com_forward_speed_m_s",
    )
    training = _exact_mapping(
        common.get("training"),
        {
            "checkpoint_selection",
            "environment_count",
            "ppo_seeds",
            "retries_or_seed_replacement",
            "transitions_per_seed",
        },
        field="pairing training",
    )
    _nonempty_text(training["checkpoint_selection"], field="pairing checkpoint selection")
    _integer(training["environment_count"], field="pairing environment count", minimum=1)
    _integer(training["transitions_per_seed"], field="pairing transition budget", minimum=1)
    if type(training["retries_or_seed_replacement"]) is not bool:
        raise PhaseBContractError("pairing retries flag must be a boolean")
    ppo_seeds = training["ppo_seeds"]
    if type(ppo_seeds) is not list or not ppo_seeds:
        raise PhaseBContractError("pairing PPO seeds must be a nonempty list")
    for seed in ppo_seeds:
        _integer(seed, field="pairing PPO seed", minimum=1)
    if len(set(ppo_seeds)) != len(ppo_seeds):
        raise PhaseBContractError("pairing PPO seeds must be unique")
    evaluation = _exact_mapping(
        common.get("evaluation"),
        {
            "deterministic_actions",
            "evaluation_seeds",
            "evaluator_id",
            "expert_start",
            "horizon_steps",
            "report_schema_id",
        },
        field="pairing evaluation",
    )
    if (
        type(evaluation["deterministic_actions"]) is not bool
        or type(evaluation["expert_start"]) is not bool
    ):
        raise PhaseBContractError("pairing evaluation flags must be booleans")
    _integer(evaluation["horizon_steps"], field="pairing evaluation horizon", minimum=1)
    _nonempty_text(evaluation["evaluator_id"], field="pairing evaluator identity")
    _nonempty_text(evaluation["report_schema_id"], field="pairing report schema")
    evaluation_seeds = evaluation["evaluation_seeds"]
    if type(evaluation_seeds) is not list or not evaluation_seeds:
        raise PhaseBContractError("pairing evaluation seeds must be a nonempty list")
    for seed in evaluation_seeds:
        _integer(seed, field="pairing evaluation seed", minimum=1)
    if len(set(evaluation_seeds)) != len(evaluation_seeds):
        raise PhaseBContractError("pairing evaluation seeds must be unique")
    _nonempty_text(common.get("calibration"), field="pairing calibration binding")
    _nonempty_text(common.get("claim_ceiling"), field="pairing claim ceiling")
    _nonempty_text(common.get("evidence_class"), field="pairing evidence class")


def _validate_pairing_reward(value: object, *, field: str) -> str:
    reward = _exact_mapping(value, {"path", "reward_id", "sha256"}, field=field)
    path = _nonempty_text(reward["path"], field=f"{field}.path")
    _nonempty_text(reward["reward_id"], field=f"{field}.reward_id")
    digest = reward["sha256"]
    if (path == "TBD") is not (digest == "TBD"):
        raise PhaseBContractError(f"{field} path and SHA-256 readiness differ")
    if digest == "TBD":
        return "TBD"
    return _sha(digest, field=f"{field}.sha256")


@dataclass(frozen=True, slots=True)
class T2RewardPairing:
    """Verified exact-byte authority for one reward-only T2 arm pair."""

    study_manifest_bytes: bytes = dataclass_field(repr=False)
    baseline_arm_manifest_bytes: bytes = dataclass_field(repr=False)
    candidate_arm_manifest_bytes: bytes = dataclass_field(repr=False)
    study_pairing_sha256: str = dataclass_field(init=False)
    study_manifest_sha256: str = dataclass_field(init=False)
    baseline_arm_manifest_sha256: str = dataclass_field(init=False)
    candidate_arm_manifest_sha256: str = dataclass_field(init=False)
    baseline_reward_sha256: str = dataclass_field(init=False)
    candidate_reward_sha256: str = dataclass_field(init=False)

    def __post_init__(self) -> None:
        study = _canonical_pairing_object(
            self.study_manifest_bytes,
            field_name="pairing study manifest",
        )
        baseline = _canonical_pairing_object(
            self.baseline_arm_manifest_bytes,
            field_name="baseline arm manifest",
        )
        candidate = _canonical_pairing_object(
            self.candidate_arm_manifest_bytes,
            field_name="candidate arm manifest",
        )
        arms = study.get("arms")
        if (
            set(study) != _T2_PAIRING_STUDY_FIELDS
            or study.get("schema_version") != 1
            or study.get("study_manifest_schema_id") != "t2_reward_study_manifest/v1"
            or study.get("study_id") != "t2_reward_study_expert_hold/v1"
            or type(arms) is not list
            or len(arms) != 2
            or arms != [baseline, candidate]
            or canonical_json_bytes(arms[0]) != self.baseline_arm_manifest_bytes
            or canonical_json_bytes(arms[1]) != self.candidate_arm_manifest_bytes
        ):
            raise PhaseBContractError("pairing study and exact arm manifest bytes differ")
        _nonempty_text(study.get("status"), field="pairing study status")
        receipt_sha256 = study.get("integrated_pairing_receipt_sha256")
        if receipt_sha256 != "TBD":
            _sha(receipt_sha256, field="integrated_pairing_receipt_sha256")
        if set(baseline) != _T2_PAIRING_ARM_FIELDS or set(candidate) != _T2_PAIRING_ARM_FIELDS:
            raise PhaseBContractError("paired arm manifest fields differ")
        if baseline.get("arm_label") != "baseline" or candidate.get("arm_label") != "candidate":
            raise PhaseBContractError("paired arm labels differ")
        for arm, field in ((baseline, "baseline"), (candidate, "candidate")):
            _nonempty_text(arm.get("output_path"), field=f"{field} pairing output path")
            timestamps = _exact_mapping(
                arm.get("timestamps"),
                {"completed_utc", "started_utc"},
                field=f"{field} pairing timestamps",
            )
            for name, value in timestamps.items():
                _nonempty_text(value, field=f"{field} pairing timestamps.{name}")
        baseline_reward_sha256 = _validate_pairing_reward(
            baseline.get("reward"),
            field="baseline pairing reward",
        )
        candidate_reward_sha256 = _validate_pairing_reward(
            candidate.get("reward"),
            field="candidate pairing reward",
        )
        if baseline.get("reward") == candidate.get("reward") or (
            candidate_reward_sha256 != "TBD" and baseline_reward_sha256 == candidate_reward_sha256
        ):
            raise PhaseBContractError("reward-only pairing requires distinct reward specifications")
        expected_declaration = {
            "adapter_id": "t2_reward_study_pairing_adapter/v1",
            "declared": True,
            "derivation_id": T2_REWARD_PAIRING_DERIVATION_ID,
        }
        if (
            baseline.get("pairing") != expected_declaration
            or candidate.get("pairing") != expected_declaration
        ):
            raise PhaseBContractError("paired arm declaration differs")
        baseline_common = {
            name: value
            for name, value in baseline.items()
            if name not in PAIRING_EXCLUDED_ARM_FIELDS
        }
        candidate_common = {
            name: value
            for name, value in candidate.items()
            if name not in PAIRING_EXCLUDED_ARM_FIELDS
        }
        if baseline_common != candidate_common:
            raise PhaseBContractError("paired arms differ outside the reward-only exclusions")
        _validate_t2_pairing_common_fields(baseline_common)
        pairing_sha256 = hashlib.sha256(canonical_json_bytes(baseline_common)).hexdigest()
        _sha(study.get("study_pairing_sha256"), field="study_pairing_sha256")
        if study["study_pairing_sha256"] != pairing_sha256:
            raise PhaseBContractError("study pairing SHA-256 differs from exact arm bytes")
        object.__setattr__(self, "study_pairing_sha256", pairing_sha256)
        object.__setattr__(
            self,
            "study_manifest_sha256",
            hashlib.sha256(self.study_manifest_bytes).hexdigest(),
        )
        object.__setattr__(
            self,
            "baseline_arm_manifest_sha256",
            hashlib.sha256(self.baseline_arm_manifest_bytes).hexdigest(),
        )
        object.__setattr__(
            self,
            "candidate_arm_manifest_sha256",
            hashlib.sha256(self.candidate_arm_manifest_bytes).hexdigest(),
        )
        object.__setattr__(self, "baseline_reward_sha256", baseline_reward_sha256)
        object.__setattr__(self, "candidate_reward_sha256", candidate_reward_sha256)
        if self.baseline_arm_manifest_sha256 == self.candidate_arm_manifest_sha256:
            raise PhaseBContractError("paired arm execution identities must remain distinct")

    @classmethod
    def from_study_manifest_bytes(cls, encoded: bytes) -> T2RewardPairing:
        """Require the study bytes and both embedded arm byte identities."""

        study = _canonical_pairing_object(encoded, field_name="pairing study manifest")
        arms = study.get("arms")
        if type(arms) is not list or len(arms) != 2 or any(type(arm) is not dict for arm in arms):
            raise PhaseBContractError("pairing study requires exactly two arm manifests")
        return cls(
            study_manifest_bytes=encoded,
            baseline_arm_manifest_bytes=canonical_json_bytes(arms[0]),
            candidate_arm_manifest_bytes=canonical_json_bytes(arms[1]),
        )

    def arm_manifest_sha256(self, arm_label: str) -> str:
        if arm_label == "baseline":
            return self.baseline_arm_manifest_sha256
        if arm_label == "candidate":
            return self.candidate_arm_manifest_sha256
        raise PhaseBContractError("paired arm label must be baseline or candidate")

    def to_dict(self) -> dict[str, object]:
        return {
            "arm_manifest_sha256": {
                "baseline": self.baseline_arm_manifest_sha256,
                "candidate": self.candidate_arm_manifest_sha256,
            },
            "declared": True,
            "derivation_id": T2_REWARD_PAIRING_DERIVATION_ID,
            "pairing_id": T2_REWARD_PAIRING_ID,
            "reward_sha256": {
                "baseline": self.baseline_reward_sha256,
                "candidate": self.candidate_reward_sha256,
            },
            "study_manifest_sha256": self.study_manifest_sha256,
            "study_pairing_sha256": self.study_pairing_sha256,
        }


@dataclass(frozen=True, slots=True)
class FineTuningRunManifest:
    value: Mapping[str, object]

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FineTuningRunManifest:
        item = _exact_mapping(
            value,
            {
                "claim_ceiling",
                "evaluator",
                "evidence_class",
                "execution_profiles",
                "library",
                "oracle",
                "reference_corpus",
                "reward",
                "run_manifest_schema_id",
                "schema_version",
                "starting_checkpoint",
                "task",
                "training_design",
            },
            field="run manifest",
        )
        if (
            item["run_manifest_schema_id"] != RUN_MANIFEST_SCHEMA_ID
            or item["schema_version"] != 2
            or item["evidence_class"] != EVIDENCE_CLASS
            or item["claim_ceiling"] != CLAIM_CEILING
        ):
            raise PhaseBContractError("run manifest identity or claim boundary differs")
        expected_profiles = {
            "checkpoint_selection": FINAL_CHECKPOINT_RULE,
            "cohort": {
                "evidence_class": "exploratory_fine_tuning_cycle",
                "promotable": True,
                "seeds": list(COHORT_SEEDS),
                "transitions_per_seed": COHORT_TRANSITIONS,
            },
            "smoke": {
                "evidence_class": EVIDENCE_CLASS,
                "promotable": False,
                "seeds": list(SMOKE_SEEDS),
                "transitions_per_seed": SMOKE_TRANSITIONS,
            },
        }
        if item["execution_profiles"] != expected_profiles:
            raise PhaseBContractError("run manifest execution profiles differ")
        for field in (
            "evaluator",
            "library",
            "oracle",
            "reference_corpus",
            "reward",
            "starting_checkpoint",
            "task",
            "training_design",
        ):
            _artifact_binding(item[field], field=field)
        return cls(MappingProxyType(dict(item)))

    @property
    def sha256(self) -> str:
        return sha256_json(dict(self.value))


def _verify_bound_artifact(
    repository_root: Path,
    binding: Mapping[str, object],
    *,
    field: str,
) -> Path:
    root = Path(repository_root).resolve(strict=True)
    relative = Path(str(binding["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise PhaseBContractError(f"{field}.path must be repository-relative")
    candidate = root / relative
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise PhaseBContractError(f"{field} artifact is unavailable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise PhaseBContractError(f"{field} artifact must be a regular non-linked file")
    if before.st_size != binding["byte_count"]:
        raise PhaseBContractError(f"{field} artifact byte count differs")
    digest = hashlib.sha256()
    try:
        with candidate.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        after = candidate.lstat()
    except OSError as exc:
        raise PhaseBContractError(f"{field} artifact cannot be read") from exc
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in identity_fields):
        raise PhaseBContractError(f"{field} artifact changed while read")
    if digest.hexdigest() != binding["sha256"]:
        raise PhaseBContractError(f"{field} artifact SHA-256 differs")
    return candidate


def load_starting_checkpoint(
    path: Path,
    *,
    repository_root: Path,
) -> tuple[StartingCheckpointContract, str]:
    value, encoded = _require_canonical_file(path)
    contract = StartingCheckpointContract.from_dict(value)
    _verify_bound_artifact(
        repository_root,
        contract.source_expert,
        field="source_expert",
    )
    export_path = _verify_bound_artifact(
        repository_root,
        contract.strict_actor_export,
        field="strict_actor_export",
    )
    receipt_path = _verify_bound_artifact(
        repository_root,
        contract.e1_receipt,
        field="e1_receipt",
    )
    receipt, receipt_encoded = _require_canonical_file(receipt_path)
    source_receipt = receipt.get("source_expert")
    export_receipt = receipt.get("export")
    if (
        receipt.get("receipt_id") != "full_authority_warm_start_e1/v1"
        or receipt.get("passed") is not True
        or receipt.get("evidence_class") != EVIDENCE_CLASS
        or receipt.get("training_steps") != 0
        or type(source_receipt) is not dict
        or source_receipt.get("sha256") != EXPERT_ACTOR_NPZ_SHA256
        or type(export_receipt) is not dict
        or export_receipt.get("sha256") != contract.strict_actor_export["sha256"]
        or hashlib.sha256(receipt_encoded).hexdigest() != contract.e1_receipt["sha256"]
    ):
        raise PhaseBContractError("starting checkpoint E1 receipt differs")
    from oracle_composition.experiments.fixed_reference import ExperimentContractError

    from .receipts import validate_e1_receipt_at_admission

    try:
        validate_e1_receipt_at_admission(
            repository_root=repository_root,
            receipt_path=receipt_path,
            expected_receipt_sha256=str(contract.e1_receipt["sha256"]),
            export_path=export_path,
            expected_export_sha256=str(contract.strict_actor_export["sha256"]),
        )
    except ExperimentContractError as exc:
        raise PhaseBContractError(str(exc)) from exc
    return contract, hashlib.sha256(encoded).hexdigest()


def load_fine_tuning_run_manifest(
    path: Path,
    *,
    repository_root: Path,
) -> tuple[FineTuningRunManifest, str]:
    """Verify every authorable and frozen binding before model construction."""

    value, encoded = _require_canonical_file(path)
    manifest = FineTuningRunManifest.from_dict(value)
    observed_manifest_sha256 = hashlib.sha256(encoded).hexdigest()
    if observed_manifest_sha256 != REVIEWED_RUN_MANIFEST_SHA256:
        raise PhaseBContractError("run manifest digest differs from the reviewed admission seal")
    root = Path(repository_root)
    verified = {
        field: _verify_bound_artifact(root, manifest.value[field], field=field)
        for field in (
            "evaluator",
            "library",
            "oracle",
            "reference_corpus",
            "reward",
            "starting_checkpoint",
            "task",
            "training_design",
        )
    }
    from oracle_composition.contracts.reference_identity_v2 import validate_corpus_manifest
    from oracle_composition.harness.evidence import EvidenceChainError, verify_library_statistics
    from oracle_composition.harness.inputs import (
        load_library_manifest,
        load_task_spec,
        verify_library_artifacts,
    )

    try:
        library = load_library_manifest(verified["library"])
        task = load_task_spec(verified["task"], library=library)
        verify_library_artifacts(root, library)
        verify_library_statistics(root, library)
        corpus_value, corpus_encoded = _require_canonical_file(verified["reference_corpus"])
        validate_corpus_manifest(corpus_value)
    except (EvidenceChainError, OracleContractError, ValueError) as exc:
        raise PhaseBContractError(f"frozen task, library, or corpus is invalid: {exc}") from exc
    if (
        library.raw_sha256 != manifest.value["library"]["sha256"]
        or task.raw_sha256 != manifest.value["task"]["sha256"]
        or hashlib.sha256(corpus_encoded).hexdigest()
        != manifest.value["reference_corpus"]["sha256"]
        or library.source_evidence["corpus_manifest_v2"].sha256
        != manifest.value["reference_corpus"]["sha256"]
    ):
        raise PhaseBContractError("task, library, and reference-corpus cross-links differ")
    training_value, _training_encoded = _require_canonical_file(verified["training_design"])
    evaluator_value, _evaluator_encoded = _require_canonical_file(verified["evaluator"])
    if training_value != training_design_contract_value():
        raise PhaseBContractError("training design semantics differ")
    if evaluator_value != utility_evaluation_design_contract_value():
        raise PhaseBContractError("utility evaluator semantics differ")
    oracle, oracle_sha256 = load_phase_b_oracle(
        verified["oracle"],
        available_behaviors=("expert", "medium", "simple"),
    )
    if oracle_sha256 != manifest.value["oracle"]["sha256"] or not oracle.program.behaviors:
        raise PhaseBContractError("run manifest oracle binding differs")
    _reward, reward_sha256 = RewardRegistry().load(verified["reward"])
    if reward_sha256 != manifest.value["reward"]["sha256"]:
        raise PhaseBContractError("run manifest reward binding differs")
    _starting, starting_sha256 = load_starting_checkpoint(
        verified["starting_checkpoint"],
        repository_root=root,
    )
    if starting_sha256 != manifest.value["starting_checkpoint"]["sha256"]:
        raise PhaseBContractError("run manifest starting checkpoint differs")
    return manifest, observed_manifest_sha256


_REPORT_V1_REQUIRED = {
    "claim_ceiling",
    "cycle",
    "determinism_check",
    "evaluation",
    "evidence_class",
    "generated_utc",
    "library_manifest_sha256",
    "oracles",
    "per_episode",
    "report_schema_id",
    "runtime_fingerprint",
    "runtime_fingerprint_sha256",
    "schema_version",
    "summary",
    "task_spec_sha256",
    "trace_content_index",
    "wall_time_seconds",
}

_REPORT_V2_GROUPS = {
    "evaluation": {
        "evidence_statement",
        "step_zero_per_episode",
        "trained_per_episode",
    },
    "inputs": {
        "evaluator_sha256",
        "library_sha256",
        "oracle_canonical_sha256",
        "oracle_file_sha256",
        "reference_sha256",
        "reward_compositor_sha256",
        "reward_file_sha256",
        "reward_formula_id",
        "reward_formula_sha256",
        "reward_schema_id",
        "task_sha256",
        "training_design_sha256",
    },
    "policy": {
        "e1_receipt_sha256",
        "final_step",
        "full_checkpoint_sha256",
        "ppo_seed",
        "starting_expert_identity",
        "strict_export_sha256",
    },
    "training": {
        "disk_bytes",
        "losses",
        "observed_transitions",
        "peak_rss_bytes",
        "planned_transitions",
        "rollouts",
        "rsi_ledger_sha256",
        "stream_counts",
        "throughput_steps_s",
        "unfreeze_receipt_sha256",
        "updates",
        "wall_time_seconds",
    },
    "reference_runtime": {"records"},
    "reward_runtime": {
        "ignored_stock_reward",
        "parameters",
        "r_task",
        "r_track",
        "r_train",
    },
    "integrity": {
        "deterministic_reload",
        "explicit_missing_fields",
        "failure_receipt",
        "trace_index_sha256",
    },
}

_REPORT_V2_EPISODE_FIELDS = {
    "com_forward_speed_m_s",
    "contacts",
    "fall",
    "root_delta_forward_speed_m_s",
    "six_tracking_errors",
    "switch_records",
    "resynchronization_records",
}
_REPORT_V2_SUMMARY_FIELDS = {
    "arms",
    "distributions_by_arm",
    "distributions_by_cell",
    "episode_summaries",
    "hard_gates",
    "policy_seeds",
    "step_zero_comparator",
}
_REPORT_V2_REFERENCE_RECORD_FIELDS = {
    "behavior",
    "hidden_reward_target_sha256",
    "phase",
    "policy_window_indices",
    "policy_window_sha256",
    "reward_target_index",
    "task_step",
    "transfer",
}
_REPORT_V2_TRANSFER_FIELDS = {
    "candidate_range_inclusive",
    "candidates",
    "selected_normalized_errors",
    "selected_phase",
    "selected_score",
    "selection_reason",
    "source_behavior",
    "target_behavior",
    "target_window_indices",
    "task_step",
}


def validate_cycle_report(value: Mapping[str, object]) -> dict[str, object]:
    """Validate report v1 compatibility or the required v2 evidence groups."""

    if type(value) is not dict:
        raise PhaseBContractError("cycle report must be an object")
    schema = value.get("report_schema_id")
    if schema == REPORT_V1_SCHEMA_ID:
        if value.get("schema_version") != 1 or not _REPORT_V1_REQUIRED.issubset(value):
            raise PhaseBContractError("cycle report v1 is missing a required field")
        return dict(value)
    if schema != REPORT_V2_SCHEMA_ID or value.get("schema_version") != 2:
        raise PhaseBContractError("cycle report schema is unknown")
    required = _REPORT_V1_REQUIRED | set(_REPORT_V2_GROUPS)
    if not required.issubset(value):
        raise PhaseBContractError("cycle report v2 is missing a required top-level field")
    if value.get("claim_ceiling") != CLAIM_CEILING:
        raise PhaseBContractError("cycle report v2 claim ceiling differs")
    for group, fields in _REPORT_V2_GROUPS.items():
        section = value[group]
        if type(section) is not dict or not fields.issubset(section):
            raise PhaseBContractError(f"cycle report v2 {group} is missing a required field")
    episodes = value["per_episode"]
    if type(episodes) is not list or any(
        type(episode) is not dict or not _REPORT_V2_EPISODE_FIELDS.issubset(episode)
        for episode in episodes
    ):
        raise PhaseBContractError("cycle report v2 episode evidence is incomplete")
    summary = value["summary"]
    if type(summary) is not dict or not _REPORT_V2_SUMMARY_FIELDS.issubset(summary):
        raise PhaseBContractError("cycle report v2 summary evidence is incomplete")
    reference_records = value["reference_runtime"]["records"]
    if type(reference_records) is not list:
        raise PhaseBContractError("cycle report v2 reference records must be a list")
    for record in reference_records:
        if type(record) is not dict or not _REPORT_V2_REFERENCE_RECORD_FIELDS.issubset(record):
            raise PhaseBContractError("cycle report v2 reference record is incomplete")
        _sha(record["policy_window_sha256"], field="policy_window_sha256")
        _sha(record["hidden_reward_target_sha256"], field="hidden_reward_target_sha256")
        transfer = record["transfer"]
        if transfer is not None and (
            type(transfer) is not dict or not _REPORT_V2_TRANSFER_FIELDS.issubset(transfer)
        ):
            raise PhaseBContractError("cycle report v2 phase-transfer record is incomplete")
    integrity = value["integrity"]
    missing = integrity["explicit_missing_fields"]
    failure = integrity["failure_receipt"]
    if type(missing) is not list or any(type(item) is not str or not item for item in missing):
        raise PhaseBContractError("explicit_missing_fields must be a string list")
    if failure is not None and type(failure) is not dict:
        raise PhaseBContractError("report failure receipt must be an object when present")
    evaluation = value["evaluation"]
    if (
        type(evaluation["trained_per_episode"]) is not list
        or type(evaluation["step_zero_per_episode"]) is not list
        or evaluation["trained_per_episode"] != episodes
    ):
        raise PhaseBContractError("report trained and step-zero episode evidence differs")
    reward = value["reward_runtime"]
    for field in ("r_track", "r_task", "r_train", "ignored_stock_reward"):
        _finite(reward[field], field=f"reward_runtime.{field}")
    if not math.isclose(
        float(reward["r_train"]),
        float(reward["r_track"]) + float(reward["r_task"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise PhaseBContractError("reward streams do not compose as r_track + r_task")
    return dict(value)


__all__ = [
    "CLAIM_CEILING",
    "EVIDENCE_CLASS",
    "FROZEN_TRACKING_REWARD_CONFIG",
    "NON_PAIRED_ID",
    "ORACLE_SCHEMA_ID",
    "PAIRING_EXCLUDED_ARM_FIELDS",
    "PHASE_POLICY",
    "REPORT_V2_SCHEMA_ID",
    "REVIEWED_RUN_MANIFEST_SHA256",
    "REWARD_COMPOSITOR_SHA256",
    "REWARD_SCHEMA_ID",
    "REWARD_SCHEMA_SHA256",
    "RUN_MANIFEST_SCHEMA_ID",
    "STARTING_CHECKPOINT_SCHEMA_ID",
    "T2_REWARD_PAIRING_DERIVATION_ID",
    "T2_REWARD_PAIRING_ID",
    "T2_RSI_FAMILY_ID",
    "T2_TRAINING_BLOCKS",
    "T2_TRAINING_DESIGN_SCHEMA_ID",
    "TARGET_SPEED_BOUNDS_BYTES",
    "TARGET_SPEED_BOUNDS_SHA256",
    "TARGET_SPEED_FORMULA_ID",
    "TARGET_SPEED_FORMULA_SHA256",
    "TARGET_SPEED_PARSER_ID",
    "TARGET_SPEED_PARSER_SHA256",
    "TARGET_SPEED_REWARD_SCHEMA_ID",
    "TARGET_SPEED_REWARD_SCHEMA_SHA256",
    "TASK_INPUTS_V2_SCHEMA_SHA256",
    "TASK_INPUTS_V2_SOURCE_SHA256",
    "TASK_INPUT_ADMISSION_SCHEMA_SHA256",
    "TASK_INPUT_ADMISSION_SOURCE_SHA256",
    "TRACKING_ONLY_BOUNDS_SHA256",
    "TRACKING_ONLY_FORMULA_ID",
    "TRACKING_ONLY_FORMULA_SHA256",
    "TRACKING_ONLY_PARSER_ID",
    "TRACKING_ONLY_PARSER_SHA256",
    "TRACKING_REWARD_CONFIG_SHA256",
    "TRACKING_REWARD_ID",
    "FineTuningRunManifest",
    "PhaseBContractError",
    "PhaseBOracleProgram",
    "RewardRegistry",
    "StartingCheckpointContract",
    "T2RewardPairing",
    "TargetSpeedRewardSpec",
    "TrackingOnlyRewardSpec",
    "load_fine_tuning_run_manifest",
    "load_phase_b_oracle",
    "load_starting_checkpoint",
    "t2_training_design_contract_value",
    "training_design_contract_value",
    "utility_evaluation_design_contract_value",
    "validate_cycle_report",
    "validate_training_design",
]
