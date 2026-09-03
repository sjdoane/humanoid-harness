from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.oracle_exploration_design import (
    EXCLUDED_PRIOR_EVALUATION_SEEDS,
    EXPECTED_ARM_IDS,
    EXPECTED_CONDITION_IDS,
    EXPECTED_DESIGN_SHA256,
    EXPECTED_EVALUATION_SEEDS,
    EXPECTED_HARD_GATES,
    EXPECTED_PERTURBATION_STEPS,
    EXPECTED_PRE_EXECUTION_BINDINGS,
    bounded_phase_config_from_design,
    collapse_definition_from_design,
    compose_physical_action_from_design,
    load_oracle_exploration_design,
    phase_scale_from_design,
    run_table_sha256,
)

ROOT = Path(__file__).resolve().parents[2]
DESIGN_PATH = (
    ROOT / "experiments" / "exploratory_phase_oracle_v1" / "configs" / "local_v1.study.json"
)


def _payload() -> dict[str, object]:
    value = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_payload(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def test_v1_design_expands_to_the_frozen_complete_block_table() -> None:
    design = load_oracle_exploration_design(DESIGN_PATH)
    runs = design.run_table

    assert design.sha256 == EXPECTED_DESIGN_SHA256
    assert design.source_bytes == DESIGN_PATH.read_bytes()
    assert design.source_sha256 == hashlib.sha256(design.source_bytes).hexdigest()
    assert design.sha256 == ("8af8744fe4ded63181e9a50781de2cf3d9607cd954ac7c0601b9a04cd6902e9d")
    assert len(runs) == 240
    assert [run.run_order for run in runs] == list(range(1, 241))
    assert run_table_sha256(runs) == (
        "a683f2f555f621cc309e9a1c89f5221df1c7d7eb34e08356c4bed891cb4a938d"
    )
    assert set(design.evaluation_seeds) == set(EXPECTED_EVALUATION_SEEDS)
    assert not set(design.evaluation_seeds) & set(EXCLUDED_PRIOR_EVALUATION_SEEDS)

    combinations = Counter((run.evaluation_seed, run.condition_id, run.arm_id) for run in runs)
    assert len(combinations) == 240
    assert set(combinations.values()) == {1}
    assert Counter(run.arm_id for run in runs) == {arm: 60 for arm in EXPECTED_ARM_IDS}
    assert Counter(run.condition_id for run in runs) == {
        condition: 80 for condition in EXPECTED_CONDITION_IDS
    }

    blocks: dict[int, list[object]] = defaultdict(list)
    for run in runs:
        blocks[run.block_order].append(run)
    assert len(blocks) == 60
    for block_order, rows in blocks.items():
        assert [row.within_block_order for row in rows] == [1, 2, 3, 4]
        assert len({row.evaluation_seed for row in rows}) == 1, block_order
        assert len({row.condition_id for row in rows}) == 1, block_order
        assert {row.arm_id for row in rows} == set(EXPECTED_ARM_IDS)


def test_perturbation_schedule_balances_time_and_direction_before_execution() -> None:
    design = load_oracle_exploration_design(DESIGN_PATH)

    assert {item.perturbation_action for item in design.perturbation_schedule} == set(
        EXPECTED_PERTURBATION_STEPS
    )
    for step in EXPECTED_PERTURBATION_STEPS:
        rows = [item for item in design.perturbation_schedule if item.perturbation_action == step]
        assert len(rows) == 4
        assert Counter(item.lateral_sign for item in rows) == {-1: 2, 1: 2}
        assert Counter(item.pitch_sign for item in rows) == {-1: 2, 1: 2}
        assert {(item.lateral_sign, item.pitch_sign) for item in rows} == {
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        }

    perturbed = [run for run in design.run_table if run.condition_id != "nominal"]
    nominal = [run for run in design.run_table if run.condition_id == "nominal"]
    assert all(run.perturbation_action is None for run in nominal)
    assert all(run.signed_magnitude == 0.0 for run in nominal)
    assert Counter(run.signed_magnitude for run in perturbed) == {-1.0: 80, 1.0: 80}


def test_design_retains_exact_local_artifact_and_claim_boundaries() -> None:
    design = load_oracle_exploration_design(DESIGN_PATH)
    payload = design.to_dict()
    pins = payload["artifact_pins"]

    assert payload["design_status"] == "predeclared_not_executed"
    assert payload["evidence_class"] == "exploratory"
    assert payload["distribution_status"] == "local_exploration_only"
    assert payload["redistributable"] is False
    assert payload["reference_admission"] == "Tier-K_not_admitted"
    assert payload["model_training_lineage"] == "not_available"
    assert payload["analysis"]["causal_reference_use_claim_allowed"] is False
    assert payload["requested_runtime"]["gymnasium_version"] == "1.3.0"
    assert payload["requested_runtime"]["mujoco_version"] == "3.12.0"
    assert payload["requested_runtime"]["mujoco_model_xml_sha256"] == (
        "85816f372c826d2094b4a598918233bd9c5843b2439119eece2733bdc2e0d073"
    )
    assert payload["reset_state_projection"] == {
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
    assert payload["observed_runtime_receipt_contract"]["status"] == ("required_before_execution")
    assert pins["base_controller_sha256"] == (
        "ce2aa3a1358609f09509d7f352475a7b517c6d11858ff76419b18a187cb3adf3"
    )
    assert pins["reference_residual_sha256"] == (
        "6916bf6778dd3044bca5feae22897b7d582e7389871a728549f791112d90fc22"
    )
    assert pins["reference_content_sha256"] == (
        "ef7557643ec87a31e4feb98c30faad98a6139dd646babf2d9e3be57b18f6936c"
    )
    assert pins["projection_receipt_sha256"] == (
        "433f1bb3e7a119cae49024ba714108cb378783232fd66bd1e8e3a797f5e3b754"
    )
    assert payload["required_pre_execution_bindings"] == list(EXPECTED_PRE_EXECUTION_BINDINGS)
    assert payload["required_pre_execution_bindings"][0] == ("oracle_exploration_design_sha256")
    assert payload["hard_gates"] == list(EXPECTED_HARD_GATES)
    assert payload["evaluator"]["collapse_definition"] == {
        "min_root_height_m": 1.0,
        "max_root_height_m": 2.0,
        "height_survival_bounds_inclusive": True,
        "min_torso_up_z": 0.5,
        "torso_up_survival_bound_inclusive": True,
    }
    assert payload["action_composition"]["final_clip_bounds_float32_as_float64"] == [
        -0.4000000059604645,
        0.4000000059604645,
    ]
    assert payload["attempt_policy"] == {
        "identity": "reviewed_execution_manifest_sha256",
        "claim_scope": "resolved_runs_directory",
        "max_attempts_per_identity_in_scope": 1,
        "claim_timing": "atomic_before_first_environment_construction",
        "failure_retention": ("immutable_failed_receipt_with_completed_and_missing_schedule_rows"),
        "outcome_based_retry_allowed": False,
        "cross_output_root_enforcement": "not_enforceable_by_local_filesystem",
    }

    detached = design.to_dict()
    detached["artifact_pins"]["base_controller_sha256"] = "0" * 64
    assert design.to_dict()["artifact_pins"]["base_controller_sha256"] == (
        "ce2aa3a1358609f09509d7f352475a7b517c6d11858ff76419b18a187cb3adf3"
    )


def test_phase_config_constructs_exact_geodesic_quaternion_semantics() -> None:
    design = load_oracle_exploration_design(DESIGN_PATH)
    payload = design.to_dict()
    phase = bounded_phase_config_from_design(design)

    assert phase.horizon_steps == 8
    assert phase.search_radius_frames == 4
    assert phase.max_phase_advance_frames == 2
    assert phase.match_indices == (*range(1, 5), *range(11, 28))
    assert phase.quaternion_indices == (1, 2, 3, 4)
    assert phase.quaternion_angle_scale == 0.5
    assert phase.quaternion_unit_tolerance == 1e-6
    assert payload["phase_rule"]["quaternion_error"] == ("shortest_sign_invariant_geodesic_angle")
    assert "root_quaternion_floor" not in payload["phase_rule"]["scale_rule"]
    assert payload["phase_rule"]["scale_rule"]["quaternion_scale_array_role"] == (
        "ignored_geodesic_angle_uses_explicit_scale"
    )


def test_phase_scale_uses_all_frames_population_std_and_exact_joint_indices() -> None:
    design = load_oracle_exploration_design(DESIGN_PATH)
    reference = np.zeros((1001, 45), dtype=np.float64)
    reference[:, 11] = np.linspace(-2.0, 2.0, 1001)

    scale = phase_scale_from_design(design, reference)

    assert scale.shape == (45,)
    assert scale[11] == np.std(reference[:, 11], ddof=0)
    assert np.array_equal(scale[12:28], np.full(16, 0.1))
    assert np.array_equal(scale[:11], np.ones(11))
    assert np.array_equal(scale[28:], np.ones(17))


def test_design_constructs_exact_collapse_and_action_rules() -> None:
    design = load_oracle_exploration_design(DESIGN_PATH)
    collapse = collapse_definition_from_design(design)

    assert collapse.min_root_height_m == 1.0
    assert collapse.max_root_height_m == 2.0
    assert collapse.min_torso_up_z == 0.5
    physical = compose_physical_action_from_design(
        design,
        base_action=np.full(17, 0.39),
        residual_action=np.ones(17),
        gate_weight=1.0,
    )
    assert physical.dtype == np.dtype("<f4")
    assert np.array_equal(physical, np.full(17, np.float32(0.4), dtype="<f4"))

    with pytest.raises(ExperimentContractError, match="gate weight"):
        compose_physical_action_from_design(
            design,
            base_action=np.zeros(17),
            residual_action=np.zeros(17),
            gate_weight=0.5,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"unknown": True}), "keys mismatch"),
        (
            lambda value: value["evaluation_seeds"].__setitem__(0, 3000),
            "exactly 4000 through 4019",
        ),
        (
            lambda value: value["factorial"]["arms"].pop(),
            "exact ordered 2x2",
        ),
        (
            lambda value: value["perturbation_schedule"][0].update({"lateral_sign": 0}),
            "exactly -1 or 1",
        ),
        (
            lambda value: value.update({"expected_run_table_sha256": "f" * 64}),
            "differs from the frozen hash",
        ),
        (
            lambda value: value["analysis"].update({"stock_environment_return_role": "primary"}),
            "stock_environment_return_role",
        ),
        (
            lambda value: value["analysis"].update(
                {"stock_environment_return_summation_rule": "sum"}
            ),
            "stock_environment_return_summation_rule",
        ),
        (
            lambda value: value["reset_state_projection"].update({"application": "every_state"}),
            "reset_state_projection.application",
        ),
        (
            lambda value: value["requested_runtime"].update({"gymnasium_version": "1.3.1"}),
            "requested_runtime.gymnasium_version",
        ),
        (
            lambda value: value["hard_gates"].reverse(),
            "hard_gates",
        ),
        (
            lambda value: value["phase_rule"].update({"quaternion_indices": None}),
            "phase_rule.quaternion_indices",
        ),
        (
            lambda value: value["artifact_pins"].update({"base_controller_sha256": "f" * 64}),
            "artifact_pins.base_controller_sha256",
        ),
        (
            lambda value: value["required_pre_execution_bindings"].remove(
                "oracle_exploration_design_sha256"
            ),
            "required_pre_execution_bindings",
        ),
        (
            lambda value: value["phase_rule"]["scale_rule"].update({"standard_deviation_ddof": 1}),
            "standard_deviation_ddof",
        ),
        (
            lambda value: value["evaluator"]["collapse_definition"].update(
                {"min_root_height_m": 0.9}
            ),
            "min_root_height_m",
        ),
        (
            lambda value: value["action_composition"].update(
                {"residual_scale_in_raw_control_units": 0.09}
            ),
            "residual_scale_in_raw_control_units",
        ),
        (
            lambda value: value["attempt_policy"].update({"outcome_based_retry_allowed": True}),
            "outcome_based_retry_allowed",
        ),
    ],
)
def test_design_rejects_boundary_or_schedule_drift(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    payload = deepcopy(_payload())
    mutation(payload)
    path = tmp_path / "mutated.study.json"
    _write_payload(path, payload)

    with pytest.raises(ExperimentContractError, match=message):
        load_oracle_exploration_design(path)


def test_design_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.study.json"
    source = DESIGN_PATH.read_text(encoding="utf-8")
    path.write_text(
        source.replace(
            '{\n  "schema_version": 1,', '{\n  "schema_version": 1,\n  "schema_version": 1,'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ExperimentContractError, match="duplicate key"):
        load_oracle_exploration_design(path)
