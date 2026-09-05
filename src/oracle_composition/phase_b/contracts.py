"""Pure fail-closed contracts for the Phase B fine-tuning runtime."""

from __future__ import annotations

import hashlib
import math
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
from oracle_composition.rewards import task_inputs_v2 as task_inputs_v2_module
from oracle_composition.rewards.task_inputs_v2 import (
    TASK_INPUTS_V2_SCHEMA_ID,
    task_input_contract_v2,
)
from oracle_composition.tracking.reward import TrackingRewardConfig

ORACLE_SCHEMA_ID = "humanoid_reference_composition_oracle/v1"
REWARD_SCHEMA_ID = "reward_specification/tracking_only/v1"
REPORT_V1_SCHEMA_ID = "humanoid_composition_cycle_report/v1"
REPORT_V2_SCHEMA_ID = "humanoid_composition_cycle_report/v2"
STARTING_CHECKPOINT_SCHEMA_ID = "humanoid_fine_tuning_starting_checkpoint/v1"
RUN_MANIFEST_SCHEMA_ID = "humanoid_fine_tuning_run_manifest/v1"
EVIDENCE_CLASS = "interface_check"
CLAIM_CEILING = (
    "exploratory_reference_conditioned_fine_tuning_utility_only_"
    "no_causal_reference_use_oracle_improvement_reward_improvement_"
    "generalization_naturalness_or_humanoid_competence_claim"
)
EXPERT_ACTOR_NPZ_SHA256 = "60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b"
TASK_INPUTS_V2_SOURCE_SHA256 = "9607f2d56a54922eac06ec7fcc740b79e68d4ed9e88b192602aae2900fb3f0d2"
TASK_INPUTS_V2_SCHEMA_SHA256 = "8f382dde13ee44c27cbbbc0b3a53a568338f6b7cebc8e660e4084425b7b494e9"
_LEGACY_PHASE_B_ORACLE_SHA256 = "4d24f22780360d7632235572d97b3fccfe7c376162e69e15082f77bd7afcccf1"

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


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PhaseBContractError(f"{field} must be an integer >= {minimum}")
    return value


def _finite(value: object, *, field: str) -> float:
    if type(value) not in {int, float}:
        raise PhaseBContractError(f"{field} must be numeric")
    result = float(value)
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
    def registry_key(self) -> tuple[str, str, str, str, str, str, str]:
        return (
            REWARD_SCHEMA_SHA256,
            TRACKING_ONLY_FORMULA_SHA256,
            TRACKING_ONLY_PARSER_SHA256,
            TRACKING_ONLY_BOUNDS_SHA256,
            REWARD_COMPOSITOR_SHA256,
            TASK_INPUTS_V2_SOURCE_SHA256,
            TASK_INPUTS_V2_SCHEMA_SHA256,
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def to_dict(self) -> dict[str, object]:
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
        baseline = TrackingOnlyRewardSpec()
        self._entries = MappingProxyType({baseline.registry_key: baseline})

    def resolve(self, value: Mapping[str, object]) -> TrackingOnlyRewardSpec:
        spec = TrackingOnlyRewardSpec.from_dict(value)
        try:
            return self._entries[spec.registry_key]
        except KeyError as exc:  # pragma: no cover - exact parser makes this defensive
            raise PhaseBContractError("reward registry key is unknown") from exc

    def load(self, path: Path) -> tuple[TrackingOnlyRewardSpec, str]:
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
                "library",
                "oracle",
                "ppo_seed",
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
            or item["schema_version"] != 1
            or item["evidence_class"] != EVIDENCE_CLASS
            or item["claim_ceiling"] != CLAIM_CEILING
        ):
            raise PhaseBContractError("run manifest identity or claim boundary differs")
        _integer(item["ppo_seed"], field="ppo_seed", minimum=1)
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
    _verify_bound_artifact(
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
    return contract, hashlib.sha256(encoded).hexdigest()


def load_fine_tuning_run_manifest(
    path: Path,
    *,
    repository_root: Path,
) -> tuple[FineTuningRunManifest, str]:
    """Verify every authorable and frozen binding before model construction."""

    value, encoded = _require_canonical_file(path)
    manifest = FineTuningRunManifest.from_dict(value)
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
    return manifest, hashlib.sha256(encoded).hexdigest()


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
    if bool(missing) != (failure is not None):
        raise PhaseBContractError("incomplete reports require exactly one failure receipt")
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
    "ORACLE_SCHEMA_ID",
    "PHASE_POLICY",
    "REPORT_V2_SCHEMA_ID",
    "REWARD_COMPOSITOR_SHA256",
    "REWARD_SCHEMA_ID",
    "REWARD_SCHEMA_SHA256",
    "RUN_MANIFEST_SCHEMA_ID",
    "STARTING_CHECKPOINT_SCHEMA_ID",
    "TASK_INPUTS_V2_SCHEMA_SHA256",
    "TASK_INPUTS_V2_SOURCE_SHA256",
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
    "TrackingOnlyRewardSpec",
    "load_fine_tuning_run_manifest",
    "load_phase_b_oracle",
    "load_starting_checkpoint",
    "validate_cycle_report",
]
