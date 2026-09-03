"""Strict design contract for the predeclared local phase-oracle v1 replay."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from string import hexdigits
from typing import Any

import numpy as np

from .fixed_reference import ExperimentContractError, read_bounded_json_artifact

MAX_DESIGN_BYTES = 256 * 1024
RUN_ORDER_ALGORITHM = "sha256_ranked_complete_blocks/v1"
EXPECTED_EXPERIMENT_ID = "exploratory_phase_oracle_v1"
EXPECTED_DESIGN_SHA256 = "8af8744fe4ded63181e9a50781de2cf3d9607cd954ac7c0601b9a04cd6902e9d"
EXPECTED_EVALUATION_SEEDS = tuple(range(4000, 4020))
EXCLUDED_PRIOR_EVALUATION_SEEDS = tuple(range(3000, 3020))
EXPECTED_PERTURBATION_STEPS = (200, 325, 450, 575, 700)
EXPECTED_ARM_IDS = ("T0_G0", "T0_G1", "T1_G0", "T1_G1")
EXPECTED_CONDITION_IDS = (
    "nominal",
    "lateral_velocity",
    "pitch_velocity_falsifier",
)
_EXPECTED_ARTIFACT_PINS = {
    "base_controller_filename": "minari_bc_v0.npz",
    "base_controller_sha256": ("ce2aa3a1358609f09509d7f352475a7b517c6d11858ff76419b18a187cb3adf3"),
    "reference_residual_filename": "reference_residual_ppo_v0.npz",
    "reference_residual_sha256": (
        "6916bf6778dd3044bca5feae22897b7d582e7389871a728549f791112d90fc22"
    ),
    "dataset_id": "mujoco/humanoid/expert-v0",
    "source_commit": "8e62dc7f7fcb4a19f8f869c65402d4bb60049117",
    "source_record_sha256": ("974700591304a4d3be576d55bd294558ae0cd10cbfce5bef7bd73ca21632f899"),
    "hdf5_sha256": "8253be693f06aeeac3cb62eeb349ad02d4ca0bcad02685b4b8ed390798d9aa1e",
    "metadata_sha256": ("2eb6e0ba388ceabef5eec1dea7e401e62391d856cf42b394c262db7c21366024"),
    "projection_receipt_sha256": (
        "433f1bb3e7a119cae49024ba714108cb378783232fd66bd1e8e3a797f5e3b754"
    ),
    "reference_content_sha256": (
        "ef7557643ec87a31e4feb98c30faad98a6139dd646babf2d9e3be57b18f6936c"
    ),
    "reference_schema_sha256": ("35d5e7cc86c0054d40362a40ff552ddf9252c3067d02a943cbc2f294c24b9771"),
    "tracking_reward_sha256": ("cc55731febc0c05a94174d4b99601fc8b3745273d21ee72236f4d9a75f81b10e"),
    "task_reward_sha256": ("1c008dabe0571a6cb76bb99bf3758f5097614baa711831c7d4e082292a240c8c"),
}
EXPECTED_PRE_EXECUTION_BINDINGS = (
    "oracle_exploration_design_sha256",
    "oracle_exploration_design_source_sha256",
    "registered_source_record_sha256",
    "registered_import_projection_receipt_sha256",
    "minari_importer_source_sha256",
    "reference_schema_sha256",
    "humanoid_reference_source_sha256",
    "observed_runtime_model_abi_receipt_sha256",
    "runner_source_sha256",
    "phase_oracle_source_sha256",
    "protected_evaluator_source_sha256",
    "protected_runtime_source_sha256",
    "tracking_reward_source_sha256",
    "trace_contract_source_sha256",
    "base_controller_load_receipt_sha256",
    "reference_residual_load_receipt_sha256",
)
EXPECTED_HARD_GATES = (
    "reviewed_execution_manifest_binds_all_required_pre_execution_roles",
    "observed_runtime_model_abi_receipt_matches_requested_runtime",
    "all_240_scheduled_runs_reported_exactly_once",
    "zero_oracle_contract_faults",
    "1000_actions_and_1001_trace_samples_per_completed_run",
    "finite_state_action_torque_and_metric_values",
    "all_five_physics_substeps_present_in_contact_capture",
    "trace_and_receipt_hashes_reverify",
)


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


def _mapping(value: object, *, field: str, keys: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ExperimentContractError(f"{field} must be an object")
    missing = sorted(keys - set(value))
    extra = sorted(set(value) - keys)
    if missing or extra:
        raise ExperimentContractError(
            f"{field} keys mismatch: missing={missing!r}, extra={extra!r}"
        )
    return value


def _sequence(value: object, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise ExperimentContractError(f"{field} must be an array")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise ExperimentContractError(f"{field} must be non-empty bounded text")
    return value


def _literal(value: object, expected: object, *, field: str) -> None:
    if isinstance(expected, dict):
        if not isinstance(value, dict):
            raise ExperimentContractError(f"{field} must be an object")
        if set(value) != set(expected):
            raise ExperimentContractError(f"{field} keys differ from the frozen contract")
        for key, expected_child in expected.items():
            _literal(value[key], expected_child, field=f"{field}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(value, list) or len(value) != len(expected):
            raise ExperimentContractError(f"{field} must match the frozen array")
        for index, (child, expected_child) in enumerate(zip(value, expected, strict=True)):
            _literal(child, expected_child, field=f"{field}[{index}]")
        return
    if type(value) is not type(expected) or value != expected:
        raise ExperimentContractError(f"{field} must equal {expected!r}")


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ExperimentContractError(f"{field} must be an integer >= {minimum}")
    return value


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExperimentContractError(f"{field} must be numeric")
    try:
        resolved = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ExperimentContractError(f"{field} must be finite") from exc
    if not math.isfinite(resolved):
        raise ExperimentContractError(f"{field} must be finite")
    return resolved


def _sha256(value: object, *, field: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in hexdigits for character in value)
    ):
        raise ExperimentContractError(f"{field} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class OracleArm:
    arm_id: str
    phase_level: str
    recovery_level: str


@dataclass(frozen=True, slots=True)
class PerturbationCondition:
    condition_id: str
    qvel_index: int | None
    component: str
    unit: str
    magnitude: float
    sign_field: str | None


@dataclass(frozen=True, slots=True)
class SeedPerturbationSchedule:
    evaluation_seed: int
    perturbation_action: int
    lateral_sign: int
    pitch_sign: int


@dataclass(frozen=True, slots=True)
class OracleExplorationRun:
    run_order: int
    block_order: int
    within_block_order: int
    evaluation_seed: int
    condition_id: str
    arm_id: str
    perturbation_action: int | None
    signed_magnitude: float
    qvel_index: int | None
    unit: str

    def to_dict(self) -> dict[str, object]:
        return {
            "run_order": self.run_order,
            "block_order": self.block_order,
            "within_block_order": self.within_block_order,
            "evaluation_seed": self.evaluation_seed,
            "condition_id": self.condition_id,
            "arm_id": self.arm_id,
            "perturbation_action": self.perturbation_action,
            "signed_magnitude": self.signed_magnitude,
            "qvel_index": self.qvel_index,
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class OracleExplorationDesign:
    design_version: str
    randomization_seed: int
    arms: tuple[OracleArm, ...]
    conditions: tuple[PerturbationCondition, ...]
    evaluation_seeds: tuple[int, ...]
    perturbation_schedule: tuple[SeedPerturbationSchedule, ...]
    expected_run_count: int
    expected_run_table_sha256: str
    canonical_bytes: bytes
    source_bytes: bytes
    source_sha256: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return a detached copy of the validated machine-readable design."""

        value = json.loads(self.canonical_bytes)
        if not isinstance(value, dict):  # construction already enforces this
            raise ExperimentContractError("validated design root changed type")
        return value

    @property
    def run_table(self) -> tuple[OracleExplorationRun, ...]:
        runs = _expand_run_table(self)
        observed_sha256 = run_table_sha256(runs)
        if len(runs) != self.expected_run_count:
            raise ExperimentContractError(
                f"expanded run count is {len(runs)}; expected {self.expected_run_count}"
            )
        if observed_sha256 != self.expected_run_table_sha256:
            raise ExperimentContractError("expanded run table differs from the frozen hash")
        return runs


def _parse_arms(value: object) -> tuple[OracleArm, ...]:
    raw = _sequence(value, field="factorial.arms")
    arms: list[OracleArm] = []
    for index, child in enumerate(raw):
        item = _mapping(
            child,
            field=f"factorial.arms[{index}]",
            keys=frozenset({"arm_id", "phase_level", "recovery_level"}),
        )
        arms.append(
            OracleArm(
                arm_id=_text(item["arm_id"], field=f"factorial.arms[{index}].arm_id"),
                phase_level=_text(
                    item["phase_level"], field=f"factorial.arms[{index}].phase_level"
                ),
                recovery_level=_text(
                    item["recovery_level"],
                    field=f"factorial.arms[{index}].recovery_level",
                ),
            )
        )
    expected = (
        OracleArm("T0_G0", "elapsed_time", "residual_always_active"),
        OracleArm("T0_G1", "elapsed_time", "explicit_recovery_fallback"),
        OracleArm("T1_G0", "bounded_state_conditioned", "residual_always_active"),
        OracleArm("T1_G1", "bounded_state_conditioned", "explicit_recovery_fallback"),
    )
    if tuple(arms) != expected:
        raise ExperimentContractError("factorial.arms must contain the exact ordered 2x2 design")
    return tuple(arms)


def _parse_conditions(value: object, *, max_actions: int) -> tuple[PerturbationCondition, ...]:
    raw = _sequence(value, field="conditions")
    conditions: list[PerturbationCondition] = []
    keys = frozenset({"condition_id", "qvel_index", "component", "unit", "magnitude", "sign_field"})
    for index, child in enumerate(raw):
        item = _mapping(child, field=f"conditions[{index}]", keys=keys)
        qvel_index = item["qvel_index"]
        if qvel_index is not None:
            qvel_index = _integer(qvel_index, field=f"conditions[{index}].qvel_index")
        sign_field = item["sign_field"]
        if sign_field is not None:
            sign_field = _text(sign_field, field=f"conditions[{index}].sign_field")
        conditions.append(
            PerturbationCondition(
                condition_id=_text(item["condition_id"], field=f"conditions[{index}].condition_id"),
                qvel_index=qvel_index,
                component=_text(item["component"], field=f"conditions[{index}].component"),
                unit=_text(item["unit"], field=f"conditions[{index}].unit"),
                magnitude=_finite(item["magnitude"], field=f"conditions[{index}].magnitude"),
                sign_field=sign_field,
            )
        )
    expected = (
        PerturbationCondition("nominal", None, "none", "1", 0.0, None),
        PerturbationCondition(
            "lateral_velocity",
            1,
            "root_linear_velocity_world_y",
            "m/s",
            1.0,
            "lateral_sign",
        ),
        PerturbationCondition(
            "pitch_velocity_falsifier",
            4,
            "root_angular_velocity_body_y",
            "rad/s",
            1.0,
            "pitch_sign",
        ),
    )
    if tuple(conditions) != expected:
        raise ExperimentContractError("conditions must match the exact v1 perturbation contract")
    if max(EXPECTED_PERTURBATION_STEPS) >= max_actions:
        raise ExperimentContractError("perturbation actions must precede the final action")
    return tuple(conditions)


def _parse_schedule(value: object) -> tuple[SeedPerturbationSchedule, ...]:
    raw = _sequence(value, field="perturbation_schedule")
    schedules: list[SeedPerturbationSchedule] = []
    keys = frozenset({"evaluation_seed", "perturbation_action", "lateral_sign", "pitch_sign"})
    for index, child in enumerate(raw):
        item = _mapping(child, field=f"perturbation_schedule[{index}]", keys=keys)
        schedule = SeedPerturbationSchedule(
            evaluation_seed=_integer(
                item["evaluation_seed"], field=f"perturbation_schedule[{index}].evaluation_seed"
            ),
            perturbation_action=_integer(
                item["perturbation_action"],
                field=f"perturbation_schedule[{index}].perturbation_action",
                minimum=1,
            ),
            lateral_sign=_integer(
                item["lateral_sign"],
                field=f"perturbation_schedule[{index}].lateral_sign",
                minimum=-1,
            ),
            pitch_sign=_integer(
                item["pitch_sign"],
                field=f"perturbation_schedule[{index}].pitch_sign",
                minimum=-1,
            ),
        )
        if schedule.lateral_sign not in {-1, 1} or schedule.pitch_sign not in {-1, 1}:
            raise ExperimentContractError("perturbation signs must be exactly -1 or 1")
        if schedule.perturbation_action not in EXPECTED_PERTURBATION_STEPS:
            raise ExperimentContractError("perturbation action is outside the frozen v1 set")
        schedules.append(schedule)
    if tuple(sorted(item.evaluation_seed for item in schedules)) != EXPECTED_EVALUATION_SEEDS:
        raise ExperimentContractError("perturbation schedule must cover each evaluation seed once")
    for step in EXPECTED_PERTURBATION_STEPS:
        rows = [item for item in schedules if item.perturbation_action == step]
        if len(rows) != 4:
            raise ExperimentContractError("each perturbation action must cover exactly four seeds")
        sign_pairs = {(item.lateral_sign, item.pitch_sign) for item in rows}
        if sign_pairs != {(-1, -1), (-1, 1), (1, -1), (1, 1)}:
            raise ExperimentContractError(
                "each perturbation action must contain all lateral/pitch sign pairs"
            )
    return tuple(sorted(schedules, key=lambda item: item.evaluation_seed))


def _validate_artifact_pins(value: object) -> None:
    item = _mapping(
        value,
        field="artifact_pins",
        keys=frozenset(_EXPECTED_ARTIFACT_PINS),
    )
    _literal(item, _EXPECTED_ARTIFACT_PINS, field="artifact_pins")


def _validate_phase_rule(value: object) -> None:
    item = _mapping(
        value,
        field="phase_rule",
        keys=frozenset(
            {
                "implementation_id",
                "horizon_steps",
                "search_radius_frames",
                "max_phase_advance_frames",
                "anchor_penalty",
                "switch_margin",
                "match_indices",
                "quaternion_indices",
                "quaternion_error",
                "quaternion_angle_scale_rad",
                "quaternion_unit_tolerance",
                "pose_error_aggregation",
                "scale_rule",
                "end_behavior",
                "recursive_phase_updates",
            }
        ),
    )
    _literal(
        item["implementation_id"], "bounded_phase_matcher/v1", field="phase_rule.implementation_id"
    )
    _literal(item["horizon_steps"], 8, field="phase_rule.horizon_steps")
    _literal(item["search_radius_frames"], 4, field="phase_rule.search_radius_frames")
    _literal(item["max_phase_advance_frames"], 2, field="phase_rule.max_phase_advance_frames")
    _literal(item["anchor_penalty"], 0.05, field="phase_rule.anchor_penalty")
    _literal(item["switch_margin"], 0.1, field="phase_rule.switch_margin")
    _literal(
        item["match_indices"], [*range(1, 5), *range(11, 28)], field="phase_rule.match_indices"
    )
    _literal(item["quaternion_indices"], [1, 2, 3, 4], field="phase_rule.quaternion_indices")
    _literal(
        item["quaternion_error"],
        "shortest_sign_invariant_geodesic_angle",
        field="phase_rule.quaternion_error",
    )
    _literal(
        item["quaternion_angle_scale_rad"],
        0.5,
        field="phase_rule.quaternion_angle_scale_rad",
    )
    _literal(
        item["quaternion_unit_tolerance"],
        1e-6,
        field="phase_rule.quaternion_unit_tolerance",
    )
    _literal(
        item["pose_error_aggregation"],
        "mean_of_17_squared_scaled_joint_errors_and_one_squared_scaled_geodesic_angle",
        field="phase_rule.pose_error_aggregation",
    )
    _literal(
        item["scale_rule"],
        {
            "implementation_id": "joint_std_floor_with_geodesic_quaternion/v2",
            "joint_position_floor_rad": 0.1,
            "standard_deviation_ddof": 0,
            "frame_domain": "all_1001_reference_frames",
            "scored_joint_indices": [*range(11, 28)],
            "unscored_scale_fill": 1.0,
            "quaternion_scale_array_role": "ignored_geodesic_angle_uses_explicit_scale",
        },
        field="phase_rule.scale_rule",
    )
    _literal(item["end_behavior"], "inclusive_terminal_hold", field="phase_rule.end_behavior")
    _literal(item["recursive_phase_updates"], False, field="phase_rule.recursive_phase_updates")


def _validate_recovery_gate(value: object) -> None:
    expected = {
        "implementation_id": "recovery_gate/v1",
        "entry_height_min_m": 1.05,
        "entry_height_max_m": 1.8,
        "entry_torso_up_z_min": 0.75,
        "entry_pose_error_max": 2.2,
        "entry_pose_error_dwell_steps": 3,
        "exit_height_min_m": 1.1,
        "exit_height_max_m": 1.7,
        "exit_torso_up_z_min": 0.8,
        "exit_pose_error_max": 1.5,
        "exit_stable_steps": 20,
        "recovery_action": "base_controller_only_residual_weight_zero",
    }
    item = _mapping(value, field="recovery_gate", keys=frozenset(expected))
    _literal(item, expected, field="recovery_gate")


def _validate_action_composition(value: object) -> None:
    expected = {
        "implementation_id": "raw_humanoid_control_additive_residual/v1",
        "actuator_order": [
            "abdomen_y",
            "abdomen_z",
            "abdomen_x",
            "right_hip_x",
            "right_hip_z",
            "right_hip_y",
            "right_knee",
            "left_hip_x",
            "left_hip_z",
            "left_hip_y",
            "left_knee",
            "right_shoulder1",
            "right_shoulder2",
            "right_elbow",
            "left_shoulder1",
            "left_shoulder2",
            "left_elbow",
        ],
        "composition_dtype": "little_endian_float32",
        "base_output_bounds_float32_as_float64": [
            -0.4000000059604645,
            0.4000000059604645,
        ],
        "base_output_contract": (
            "float32_raw_Humanoid-v5_control_with_action_space_endpoint_bytes"
        ),
        "residual_output_contract": "unscaled_dimensionless_action_in_closed_interval_-1_1",
        "residual_scale_in_raw_control_units": 0.08,
        "residual_scale_float32_as_float64": 0.07999999821186066,
        "gate_weight_values": [0.0, 1.0],
        "formula": (
            "float32_clip(float32(base_raw_control)+float32(gate_weight)*float32(0.08)*"
            "float32(residual_unscaled),float32(-0.4),float32(0.4))"
        ),
        "final_clip_bounds_float32_as_float64": [
            -0.4000000059604645,
            0.4000000059604645,
        ],
        "composition_timing": "after_pre_step_oracle_selection_before_environment_step",
        "environment_action": "post_clip_float32_raw_control_command",
        "trace_physical_action": "submitted_float32_command_promoted_exactly_to_float64",
        "effective_control_rule": (
            "MuJoCo_clamps_submitted_command_to_model_actuator_ctrlrange_-0.4_0.4"
        ),
        "trace_and_evaluator_normalized_action": (
            "clip(submitted_float32_command_promoted_to_float64_divided_by_exact_0.4,-1,1)"
        ),
    }
    item = _mapping(value, field="action_composition", keys=frozenset(expected))
    _literal(item, expected, field="action_composition")


def _validate_evaluator(value: object) -> None:
    expected = {
        "implementation_id": "protected_tracking_evaluator/v1",
        "target_frame_rule": ("cache_W_t_before_action_then_grade_x_t_plus_1_against_W_t_index_1"),
        "measurement_timing": "one_post_step_measurement_per_control_action",
        "collapse_definition": {
            "min_root_height_m": 1.0,
            "max_root_height_m": 2.0,
            "height_survival_bounds_inclusive": True,
            "min_torso_up_z": 0.5,
            "torso_up_survival_bound_inclusive": True,
        },
        "primary_outcome": {
            "name": "no_collapse_over_1000_actions",
            "required_actions": 1000,
            "required_post_step_samples": 1000,
            "true_iff": "all_required_post_step_samples_are_not_collapsed",
        },
        "metric_inputs": {
            "state": "direct_post_step_MuJoCo_state",
            "action": (
                "clip(submitted_float32_command_promoted_to_float64_divided_by_exact_0.4,-1,1)"
            ),
            "torque": "post_transmission_generalized_actuator_torque_at_17_actuated_DoFs",
            "contact": "every_active_contact_after_each_of_exactly_5_physics_substeps",
            "action_rate_baseline": "seeded_reset_zero_control_then_previous_executed_action",
            "jerk_baseline": (
                "seeded_reset_joint_acceleration_then_previous_post_step_acceleration"
            ),
        },
    }
    item = _mapping(value, field="evaluator", keys=frozenset(expected))
    _literal(item, expected, field="evaluator")


def _validate_attempt_policy(value: object) -> None:
    expected = {
        "identity": "reviewed_execution_manifest_sha256",
        "claim_scope": "resolved_runs_directory",
        "max_attempts_per_identity_in_scope": 1,
        "claim_timing": "atomic_before_first_environment_construction",
        "failure_retention": ("immutable_failed_receipt_with_completed_and_missing_schedule_rows"),
        "outcome_based_retry_allowed": False,
        "cross_output_root_enforcement": "not_enforceable_by_local_filesystem",
    }
    item = _mapping(value, field="attempt_policy", keys=frozenset(expected))
    _literal(item, expected, field="attempt_policy")


def _validate_reset_state_projection(value: object) -> None:
    expected = {
        "implementation_id": "bounded_reset_root_orientation_l2_projection/v1",
        "application": "reset_state_before_action_0_only",
        "raw_simulator_state": "unchanged_and_recorded_in_trace",
        "root_quaternion_qpos_indices": [3, 4, 5, 6],
        "component_bounds_inclusive": [
            [0.99, 1.01],
            [-0.01, 0.01],
            [-0.01, 0.01],
            [-0.01, 0.01],
        ],
        "l2_norm_bounds_inclusive": [0.98, 1.02],
        "projection": "divide_float64_quaternion_copy_by_its_float64_l2_norm",
        "projected_consumers": ["phase_matcher", "recovery_gate"],
        "controller_input": "raw_Gymnasium_reset_observation",
        "post_step_state_rule": "strict_unit_quaternion_without_projection",
    }
    item = _mapping(value, field="reset_state_projection", keys=frozenset(expected))
    _literal(item, expected, field="reset_state_projection")


def _validate_analysis(value: object) -> None:
    expected = {
        "analysis_scope": "paired_seed_diagnostics_conditioned_on_one_checkpoint_bundle",
        "independent_checkpoint_bundles": 1,
        "paired_unit": "evaluation_seed",
        "primary_outcome": "no_collapse_over_1000_actions",
        "report_phase_main_effect": True,
        "report_recovery_main_effect": True,
        "report_phase_by_recovery_interaction": True,
        "stock_environment_return_role": "diagnostic_only",
        "stock_environment_return_step_value": ("finite_float_returned_by_each_environment_step"),
        "stock_environment_return_summation_rule": (
            "python_math_fsum_in_action_order_over_exactly_1000_values/v1"
        ),
        "automatic_promotion": False,
        "causal_reference_use_claim_allowed": False,
        "missing_run_rule": "report_every_scheduled_run_no_outcome_based_retry",
    }
    item = _mapping(value, field="analysis", keys=frozenset(expected))
    _literal(item, expected, field="analysis")


def load_oracle_exploration_design(path: Path) -> OracleExplorationDesign:
    """Load the exact v1 design and reject schema or scientific-boundary drift."""

    source = read_bounded_json_artifact(
        Path(path), maximum_bytes=MAX_DESIGN_BYTES, artifact="oracle exploration design"
    )
    root = source.value
    keys = frozenset(
        {
            "schema_version",
            "experiment_id",
            "design_version",
            "design_status",
            "evidence_class",
            "distribution_status",
            "claim_ceiling",
            "redistributable",
            "reference_admission",
            "model_training_lineage",
            "fresh_seed_claim",
            "requested_runtime",
            "reset_state_projection",
            "observed_runtime_receipt_contract",
            "artifact_pins",
            "phase_rule",
            "recovery_gate",
            "action_composition",
            "evaluator",
            "attempt_policy",
            "factorial",
            "conditions",
            "evaluation_seeds",
            "excluded_prior_evaluation_seeds",
            "perturbation_schedule",
            "randomization",
            "analysis",
            "required_pre_execution_bindings",
            "hard_gates",
            "expected_run_count",
            "expected_run_table_sha256",
        }
    )
    _mapping(root, field="oracle exploration design", keys=keys)
    _literal(root["schema_version"], 1, field="schema_version")
    _literal(root["experiment_id"], EXPECTED_EXPERIMENT_ID, field="experiment_id")
    design_version = _text(root["design_version"], field="design_version")
    _literal(root["design_status"], "predeclared_not_executed", field="design_status")
    _literal(root["evidence_class"], "exploratory", field="evidence_class")
    _literal(root["distribution_status"], "local_exploration_only", field="distribution_status")
    _literal(
        root["claim_ceiling"],
        "exact_byte_local_replay_and_conditional_four_arm_diagnostics_only",
        field="claim_ceiling",
    )
    _literal(root["redistributable"], False, field="redistributable")
    _literal(root["reference_admission"], "Tier-K_not_admitted", field="reference_admission")
    _literal(root["model_training_lineage"], "not_available", field="model_training_lineage")
    _literal(
        root["fresh_seed_claim"],
        "predetermined_for_v1_not_proven_unseen_to_checkpoint_training",
        field="fresh_seed_claim",
    )
    _validate_reset_state_projection(root["reset_state_projection"])

    requested_runtime_expected = {
        "environment_id": "Humanoid-v5",
        "gymnasium_version": "1.3.0",
        "mujoco_version": "3.12.0",
        "mujoco_model_xml_sha256": (
            "85816f372c826d2094b4a598918233bd9c5843b2439119eece2733bdc2e0d073"
        ),
        "max_actions": 1000,
        "terminate_when_unhealthy": False,
        "reset_noise_scale": 0.01,
        "exclude_current_positions_from_observation": True,
        "frame_skip": 5,
        "control_period_seconds": 0.015,
        "deterministic_actions": True,
        "reference_horizon_steps": 8,
        "physical_residual_scale": 0.08,
    }
    requested_runtime = _mapping(
        root["requested_runtime"],
        field="requested_runtime",
        keys=frozenset(requested_runtime_expected),
    )
    _literal(requested_runtime, requested_runtime_expected, field="requested_runtime")
    observed_receipt_expected = {
        "status": "required_before_execution",
        "binding_role": "observed_runtime_model_abi_receipt_sha256",
        "required_match_fields": [
            "environment_id",
            "gymnasium_version",
            "mujoco_version",
            "mujoco_model_xml_sha256",
            "max_actions",
            "terminate_when_unhealthy",
            "reset_noise_scale",
            "exclude_current_positions_from_observation",
            "frame_skip",
            "control_period_seconds",
        ],
        "required_observed_fields": [
            "observation_space_sha256",
            "action_space_sha256",
            "qpos_shape",
            "qvel_shape",
            "actuator_joint_order",
            "actuator_qpos_indices",
            "actuator_qvel_indices",
            "actuator_gear_by_joint",
            "generalized_actuator_torque_capacity_n_m",
            "dependency_lock_sha256",
        ],
    }
    observed_receipt = _mapping(
        root["observed_runtime_receipt_contract"],
        field="observed_runtime_receipt_contract",
        keys=frozenset(observed_receipt_expected),
    )
    _literal(
        observed_receipt,
        observed_receipt_expected,
        field="observed_runtime_receipt_contract",
    )
    _validate_artifact_pins(root["artifact_pins"])
    _validate_phase_rule(root["phase_rule"])
    _validate_recovery_gate(root["recovery_gate"])
    _validate_action_composition(root["action_composition"])
    _validate_evaluator(root["evaluator"])
    _validate_attempt_policy(root["attempt_policy"])

    factorial = _mapping(
        root["factorial"],
        field="factorial",
        keys=frozenset({"design", "phase_factor", "recovery_factor", "arms"}),
    )
    _literal(factorial["design"], "full_2x2_repeated_within_seed", field="factorial.design")
    _literal(
        factorial["phase_factor"],
        ["elapsed_time", "bounded_state_conditioned"],
        field="factorial.phase_factor",
    )
    _literal(
        factorial["recovery_factor"],
        ["residual_always_active", "explicit_recovery_fallback"],
        field="factorial.recovery_factor",
    )
    arms = _parse_arms(factorial["arms"])
    conditions = _parse_conditions(
        root["conditions"], max_actions=requested_runtime_expected["max_actions"]
    )

    seeds = tuple(_sequence(root["evaluation_seeds"], field="evaluation_seeds"))
    if seeds != EXPECTED_EVALUATION_SEEDS:
        raise ExperimentContractError("evaluation_seeds must be exactly 4000 through 4019")
    excluded = tuple(
        _sequence(root["excluded_prior_evaluation_seeds"], field="excluded_prior_evaluation_seeds")
    )
    if excluded != EXCLUDED_PRIOR_EVALUATION_SEEDS:
        raise ExperimentContractError("excluded prior seeds must be exactly 3000 through 3019")
    if set(seeds) & set(excluded):
        raise ExperimentContractError("fresh and prior evaluation seeds overlap")
    schedule = _parse_schedule(root["perturbation_schedule"])

    randomization = _mapping(
        root["randomization"],
        field="randomization",
        keys=frozenset({"algorithm", "seed", "block_fields", "within_block_field"}),
    )
    _literal(randomization["algorithm"], RUN_ORDER_ALGORITHM, field="randomization.algorithm")
    randomization_seed = _integer(randomization["seed"], field="randomization.seed")
    _literal(
        randomization["block_fields"],
        ["evaluation_seed", "condition_id"],
        field="randomization.block_fields",
    )
    _literal(
        randomization["within_block_field"], "arm_id", field="randomization.within_block_field"
    )
    _validate_analysis(root["analysis"])
    _literal(
        root["required_pre_execution_bindings"],
        list(EXPECTED_PRE_EXECUTION_BINDINGS),
        field="required_pre_execution_bindings",
    )
    hard_gates = _sequence(root["hard_gates"], field="hard_gates")
    _literal(hard_gates, list(EXPECTED_HARD_GATES), field="hard_gates")

    expected_run_count = _integer(root["expected_run_count"], field="expected_run_count", minimum=1)
    expected_run_table_sha256 = _sha256(
        root["expected_run_table_sha256"], field="expected_run_table_sha256"
    )
    design = OracleExplorationDesign(
        design_version=design_version,
        randomization_seed=randomization_seed,
        arms=arms,
        conditions=conditions,
        evaluation_seeds=seeds,
        perturbation_schedule=schedule,
        expected_run_count=expected_run_count,
        expected_run_table_sha256=expected_run_table_sha256,
        canonical_bytes=_canonical_json(root),
        source_bytes=source.encoded_bytes,
        source_sha256=source.sha256,
    )
    _ = design.run_table
    if design.sha256 != EXPECTED_DESIGN_SHA256:
        raise ExperimentContractError(
            "canonical oracle exploration design differs from the frozen v1 digest"
        )
    return design


def _rank(seed: int, *parts: object) -> bytes:
    payload = "|".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return hashlib.sha256(payload).digest()


def _expand_run_table(design: OracleExplorationDesign) -> tuple[OracleExplorationRun, ...]:
    schedule_by_seed = {item.evaluation_seed: item for item in design.perturbation_schedule}
    conditions_by_id = {item.condition_id: item for item in design.conditions}
    blocks = [
        (seed, condition_id)
        for seed in design.evaluation_seeds
        for condition_id in EXPECTED_CONDITION_IDS
    ]
    blocks.sort(
        key=lambda block: (
            _rank(design.randomization_seed, "block", block[0], block[1]),
            block,
        )
    )
    runs: list[OracleExplorationRun] = []
    for block_order, (evaluation_seed, condition_id) in enumerate(blocks, start=1):
        arms = sorted(
            design.arms,
            key=lambda arm: (
                _rank(
                    design.randomization_seed,
                    "arm",
                    evaluation_seed,
                    condition_id,
                    arm.arm_id,
                ),
                arm.arm_id,
            ),
        )
        condition = conditions_by_id[condition_id]
        schedule = schedule_by_seed[evaluation_seed]
        sign = 0 if condition.sign_field is None else getattr(schedule, condition.sign_field)
        perturbation_action = (
            None if condition.condition_id == "nominal" else schedule.perturbation_action
        )
        for within_block_order, arm in enumerate(arms, start=1):
            runs.append(
                OracleExplorationRun(
                    run_order=len(runs) + 1,
                    block_order=block_order,
                    within_block_order=within_block_order,
                    evaluation_seed=evaluation_seed,
                    condition_id=condition.condition_id,
                    arm_id=arm.arm_id,
                    perturbation_action=perturbation_action,
                    signed_magnitude=sign * condition.magnitude,
                    qvel_index=condition.qvel_index,
                    unit=condition.unit,
                )
            )
    return tuple(runs)


def run_table_sha256(runs: tuple[OracleExplorationRun, ...]) -> str:
    """Hash the exact ordered run-table rows consumed by a future runner."""

    return hashlib.sha256(_canonical_json([run.to_dict() for run in runs])).hexdigest()


def bounded_phase_config_from_design(design: OracleExplorationDesign):
    """Construct the exact matcher config without quaternion-score fallback."""

    from .exploratory_phase import BoundedPhaseConfig

    phase = design.to_dict()["phase_rule"]
    return BoundedPhaseConfig(
        horizon_steps=phase["horizon_steps"],
        search_radius_frames=phase["search_radius_frames"],
        anchor_penalty=phase["anchor_penalty"],
        switch_margin=phase["switch_margin"],
        match_indices=tuple(phase["match_indices"]),
        max_phase_advance_frames=phase["max_phase_advance_frames"],
        quaternion_indices=tuple(phase["quaternion_indices"]),
        quaternion_angle_scale=phase["quaternion_angle_scale_rad"],
        quaternion_unit_tolerance=phase["quaternion_unit_tolerance"],
    )


def phase_scale_from_design(
    design: OracleExplorationDesign,
    reference: object,
) -> np.ndarray:
    """Compute the frozen all-frame population scale for phase matching."""

    try:
        values = np.asarray(reference, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ExperimentContractError("reference must be convertible to float64") from exc
    if values.shape != (1001, 45) or not np.isfinite(values).all():
        raise ExperimentContractError("reference must be one finite (1001, 45) array")
    rule = design.to_dict()["phase_rule"]["scale_rule"]
    indices = np.asarray(rule["scored_joint_indices"], dtype=np.int64)
    scale = np.full(45, rule["unscored_scale_fill"], dtype=np.float64)
    population_std = np.std(
        values[:, indices],
        axis=0,
        ddof=rule["standard_deviation_ddof"],
    )
    scale[indices] = np.maximum(population_std, rule["joint_position_floor_rad"])
    if not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ExperimentContractError("derived phase scale must be finite and positive")
    return scale


def collapse_definition_from_design(design: OracleExplorationDesign):
    """Construct the exact protected collapse rule declared by the design."""

    from .protected_evaluator import CollapseDefinition

    definition = design.to_dict()["evaluator"]["collapse_definition"]
    return CollapseDefinition(
        min_root_height_m=definition["min_root_height_m"],
        max_root_height_m=definition["max_root_height_m"],
        min_torso_up_z=definition["min_torso_up_z"],
    )


def compose_physical_action_from_design(
    design: OracleExplorationDesign,
    *,
    base_action: object,
    residual_action: object,
    gate_weight: object,
) -> np.ndarray:
    """Apply the frozen raw-control residual formula in actuator order."""

    try:
        base = np.asarray(base_action, dtype=np.float64)
        residual = np.asarray(residual_action, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ExperimentContractError("controller actions must be convertible to float64") from exc
    if base.shape != (17,) or residual.shape != (17,):
        raise ExperimentContractError("controller actions must have shape (17,)")
    if not np.isfinite(base).all() or not np.isfinite(residual).all():
        raise ExperimentContractError("controller actions must be finite")
    action = design.to_dict()["action_composition"]
    lower, upper = action["base_output_bounds_float32_as_float64"]
    if np.any(base < lower) or np.any(base > upper):
        raise ExperimentContractError("base action exceeds its frozen raw-control bounds")
    if np.any(residual < -1.0) or np.any(residual > 1.0):
        raise ExperimentContractError("residual action exceeds its frozen unscaled bounds")
    if isinstance(gate_weight, bool) or gate_weight not in action["gate_weight_values"]:
        raise ExperimentContractError("gate weight must be exactly 0.0 or 1.0")
    base_f32 = base.astype("<f4")
    residual_f32 = residual.astype("<f4")
    weight_f32 = np.float32(gate_weight)
    scale_f32 = np.float32(action["residual_scale_in_raw_control_units"])
    clip_low, clip_high = (
        np.float32(value) for value in action["final_clip_bounds_float32_as_float64"]
    )
    composed = np.add(
        base_f32,
        np.multiply(weight_f32 * scale_f32, residual_f32, dtype=np.float32),
        dtype=np.float32,
    )
    return np.clip(composed, clip_low, clip_high).astype("<f4", copy=False)


__all__ = [
    "EXCLUDED_PRIOR_EVALUATION_SEEDS",
    "EXPECTED_ARM_IDS",
    "EXPECTED_CONDITION_IDS",
    "EXPECTED_DESIGN_SHA256",
    "EXPECTED_EVALUATION_SEEDS",
    "EXPECTED_EXPERIMENT_ID",
    "EXPECTED_HARD_GATES",
    "EXPECTED_PERTURBATION_STEPS",
    "EXPECTED_PRE_EXECUTION_BINDINGS",
    "RUN_ORDER_ALGORITHM",
    "OracleExplorationDesign",
    "OracleExplorationRun",
    "PerturbationCondition",
    "SeedPerturbationSchedule",
    "bounded_phase_config_from_design",
    "collapse_definition_from_design",
    "compose_physical_action_from_design",
    "load_oracle_exploration_design",
    "phase_scale_from_design",
    "run_table_sha256",
]
