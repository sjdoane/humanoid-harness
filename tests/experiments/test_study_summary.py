from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import oracle_composition.experiments.execution as execution_module
import oracle_composition.experiments.fixed_reference as fixed_reference_module
import oracle_composition.experiments.protected_evaluator as protected_evaluator_module
import oracle_composition.experiments.squashed_policy as squashed_policy_module
import oracle_composition.experiments.study_summary as study_summary_module
import oracle_composition.tracking.reward as tracking_reward_module
from oracle_composition.experiments.execution import (
    CHECKPOINT_FILENAME,
    PROTECTED_METRICS_STATE_SOURCE,
)
from oracle_composition.experiments.fixed_reference import (
    CANONICAL_EXECUTION_CALLABLE_AUTHORITY,
    NONAUTHORITATIVE_EXECUTION_CALLABLE_AUTHORITY,
    ExperimentContractError,
    FrozenExecutionManifest,
    RuntimeFingerprint,
    load_study_design,
    sha256_file,
)
from oracle_composition.experiments.protected_evaluator import STATION_KEEPING_ORIGIN_SOURCE
from oracle_composition.experiments.runtime_identity import source_tree_sha256
from oracle_composition.experiments.squashed_policy import (
    ACTION_BOUNDARY_MARGIN,
    ACTION_LIKELIHOOD_AUDIT_ID,
    LOG_PROB_RECOMPUTE_ATOL,
    POLICY_ID,
)
from oracle_composition.experiments.study_summary import (
    aggregate_static_study,
    exhaustive_seed_bootstrap,
    load_static_study_criteria,
)

ROOT = Path(__file__).resolve().parents[2]
PROPOSED_PATH = (
    ROOT
    / "experiments"
    / "001_humanoid_fixed_reference"
    / "configs"
    / "static_stand_precalibration_v0.study.json"
)
CRITERIA_PATH = (
    ROOT
    / "experiments"
    / "001_humanoid_fixed_reference"
    / "configs"
    / "static_positive_control_criteria_v1.json"
)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _runtime() -> RuntimeFingerprint:
    return RuntimeFingerprint(
        env_id="Humanoid-v5",
        gymnasium_version="1.3.0",
        mujoco_version="3.12.0",
        numpy_version="2.5.2",
        stable_baselines3_version="2.9.0",
        torch_version="2.14.0",
        python_version="3.13.15",
        platform_system="Darwin",
        platform_machine="arm64",
        model_sha256=SHA_A,
        dependency_lock_sha256=SHA_B,
        observation_space_sha256=SHA_C,
        action_space_sha256=SHA_A,
        physical_action_space_sha256=SHA_B,
        observation_shape=(528,),
        action_shape=(17,),
        qpos_shape=(24,),
        qvel_shape=(23,),
        actuator_gear_by_joint=(100.0,) * 5
        + (300.0, 200.0, 100.0, 100.0, 300.0, 200.0)
        + (25.0,) * 6,
        generalized_actuator_torque_capacity_n_m=(
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
        ),
        environment_max_episode_steps=1000,
        control_period_seconds=0.015,
        reference_content_sha256=SHA_A,
        reference_schema_sha256=SHA_B,
        tracking_reward_sha256=SHA_C,
        task_reward_sha256=SHA_A,
        environment_source_sha256=SHA_A,
        reference_abi_source_sha256=SHA_B,
        tracking_reward_source_sha256=sha256_file(Path(tracking_reward_module.__file__)),
        wrapper_source_sha256=SHA_B,
        experiment_contract_source_sha256=sha256_file(Path(fixed_reference_module.__file__)),
        evaluator_source_sha256=sha256_file(Path(protected_evaluator_module.__file__)),
        runner_source_sha256=SHA_A,
        execution_source_sha256=sha256_file(Path(execution_module.__file__)),
        study_summary_source_sha256=sha256_file(Path(study_summary_module.__file__)),
        policy_source_sha256=sha256_file(Path(squashed_policy_module.__file__)),
        source_tree_sha256=source_tree_sha256(),
        policy_id=POLICY_ID,
        action_transform_id="tanh_normalized_then_affine_physical_box/v1",
        observation_normalizer_id="none/v1",
        reward_normalizer_id="none/v1",
    )


def _study_files(
    tmp_path: Path,
    *,
    criteria_semantic_sha256: str | None = None,
) -> tuple[Path, Path]:
    proposed = load_study_design(PROPOSED_PATH)
    design = replace(
        proposed,
        design_version="locked-summary-test-v1",
        design_status="locked_pre_behavioral",
        study_criteria_semantic_sha256=(
            criteria_semantic_sha256
            if criteria_semantic_sha256 is not None
            else proposed.study_criteria_semantic_sha256
        ),
    )
    design_path = tmp_path / "locked.study.json"
    design_path.write_text(json.dumps(design.to_dict(), indent=2) + "\n", encoding="utf-8")
    manifest = FrozenExecutionManifest(
        schema_version=1,
        status="frozen",
        execution_callable_authority=CANONICAL_EXECUTION_CALLABLE_AUTHORITY,
        design_sha256=design.sha256,
        calibrated_design_sha256=SHA_A,
        calibrated_design_file_sha256=SHA_B,
        calibration_receipt_sha256=SHA_C,
        calibration_receipt_file_sha256=SHA_A,
        study_criteria_semantic_sha256=design.study_criteria_semantic_sha256,
        study_criteria_file_sha256=design.study_criteria_file_sha256,
        runtime=_runtime(),
        candidate_file_sha256=SHA_A,
        reviewer_id="test-reviewer",
        review_note="Reviewed for study-summary tests.",
    )
    manifest_path = tmp_path / "reviewed.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return design_path, manifest_path


def _episode(evaluation_seed: int, *, tracking_value: float = 0.1) -> dict[str, object]:
    return {
        "evaluation_seed": evaluation_seed,
        "observed_steps": 1000,
        "requested_steps": 1000,
        "survived_full_horizon": True,
        "first_collapse_step": None,
        "collapse_fraction": 0.0,
        "station_keeping_origin_source": STATION_KEEPING_ORIGIN_SOURCE,
        "initial_root_position_world_m": [0.0, 0.0, 1.4],
        "root_horizontal_displacement_max_m": 0.1,
        "root_linear_velocity_x_max_abs_m_s": 0.2,
        "root_linear_velocity_y_max_abs_m_s": 0.2,
        "root_linear_velocity_z_max_abs_m_s": 0.2,
        "control_period_seconds": 0.015,
        "root_height_rmse_m": tracking_value,
        "root_orientation_rmse_rad": tracking_value,
        "root_linear_velocity_rmse_m_s": tracking_value,
        "root_angular_velocity_rmse_rad_s": tracking_value,
        "joint_position_rmse_rad": tracking_value,
        "joint_velocity_rmse_rad_s": tracking_value,
        "joint_position_max_abs_rad": tracking_value,
        "joint_velocity_max_abs_rad_s": tracking_value,
        "normalized_policy_action_rms": 0.25,
        "normalized_policy_action_max_abs": 0.5,
        "normalized_policy_action_delta_rate_rms_per_s": 10.0,
        "normalized_policy_action_delta_rate_max_abs_per_s": 20.0,
        "generalized_actuator_torque_rms_n_m": 10.0,
        "generalized_actuator_torque_max_abs_n_m": 20.0,
        "generalized_actuator_torque_normalized_rms": 0.25,
        "generalized_actuator_torque_normalized_max_abs": 0.5,
        "joint_jerk_rms_rad_s3": 100.0,
        "joint_jerk_max_abs_rad_s3": 200.0,
        "simulator_contact_count": 2000,
        "simulator_contact_peak_normal_force_n": 500.0,
        "floor_contact_count": 2000,
        "floor_contact_peak_normal_force_n": 500.0,
        "forbidden_floor_contact_count": 0,
        "forbidden_floor_contact_fraction": 0.0,
        "forbidden_floor_contact_step_fraction": 0.0,
        "forbidden_floor_contact_peak_force_n": 0.0,
        "tracking_return": 900.0,
    }


def _checkpoint_receipt_payload(
    *,
    design_path: Path,
    manifest_path: Path,
    train_seed: int,
    checkpoint_path: Path,
) -> dict[str, object]:
    design = load_study_design(design_path)
    manifest = FrozenExecutionManifest.from_dict(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    return {
        "schema_version": 1,
        "evidence_purpose": "behavioral_evaluation",
        "completion_status": "complete",
        "claim_ceiling": design.claim_ceiling.value,
        "design_sha256": design.sha256,
        "design_file_sha256": sha256_file(design_path),
        "study_criteria_semantic_sha256": design.study_criteria_semantic_sha256,
        "study_criteria_file_sha256": design.study_criteria_file_sha256,
        "execution_manifest_sha256": manifest.sha256,
        "execution_manifest_file_sha256": sha256_file(manifest_path),
        "runtime_sha256": manifest.runtime.sha256,
        "execution_callable_authority": CANONICAL_EXECUTION_CALLABLE_AUTHORITY,
        "actuator_gear_by_joint": list(manifest.runtime.actuator_gear_by_joint),
        "generalized_actuator_torque_capacity_n_m": list(
            manifest.runtime.generalized_actuator_torque_capacity_n_m
        ),
        "source_tree_sha256": manifest.runtime.source_tree_sha256,
        "train_seed": train_seed,
        "algorithm_id": "stable_baselines3.PPO/v1",
        "policy_id": manifest.runtime.policy_id,
        "action_transform_id": manifest.runtime.action_transform_id,
        "observation_normalizer_id": manifest.runtime.observation_normalizer_id,
        "reward_normalizer_id": manifest.runtime.reward_normalizer_id,
        "action_likelihood_audit": {
            "audit_id": ACTION_LIKELIHOOD_AUDIT_ID,
            "passed": True,
            "expected_rollouts": (
                design.ppo.total_timesteps_per_seed // (design.ppo.n_envs * design.ppo.n_steps)
            ),
            "audited_rollouts": (
                design.ppo.total_timesteps_per_seed // (design.ppo.n_envs * design.ppo.n_steps)
            ),
            "action_boundary_margin": ACTION_BOUNDARY_MARGIN,
            "max_abs_stored_action": 0.5,
            "log_probability_atol": LOG_PROB_RECOMPUTE_ATOL,
            "max_log_probability_abs_error": 0.0,
        },
        "checkpoint_rule": design.checkpoint_rule,
        "requested_timesteps": design.ppo.total_timesteps_per_seed,
        "observed_timesteps": design.ppo.total_timesteps_per_seed,
        "checkpoint_file": CHECKPOINT_FILENAME,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_eligible_for_declared_evaluation": True,
        "automatic_promotion": False,
        "behavioral_claim": None,
    }


def _write_receipts(
    tmp_path: Path,
    *,
    design_path: Path,
    manifest_path: Path,
) -> tuple[list[Path], list[Path]]:
    design = load_study_design(design_path)
    manifest = FrozenExecutionManifest.from_dict(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    checkpoint_receipts: list[Path] = []
    evaluation_receipts: list[Path] = []
    for train_seed in design.train_seeds:
        run_directory = tmp_path / f"train-{train_seed}"
        run_directory.mkdir()
        checkpoint_path = run_directory / CHECKPOINT_FILENAME
        checkpoint_path.write_bytes(f"checkpoint-{train_seed}".encode())
        checkpoint_receipt = run_directory / "checkpoint_receipt.json"
        checkpoint_payload = _checkpoint_receipt_payload(
            design_path=design_path,
            manifest_path=manifest_path,
            train_seed=train_seed,
            checkpoint_path=checkpoint_path,
        )
        checkpoint_receipt.write_text(
            json.dumps(checkpoint_payload, indent=2) + "\n",
            encoding="utf-8",
        )

        evaluation_payload = {
            "schema_version": 1,
            "evidence_purpose": "behavioral_evaluation",
            "completion_status": "complete_measurements_no_promotion",
            "claim_ceiling": "static_tracking_feasibility",
            "design_sha256": design.sha256,
            "design_file_sha256": sha256_file(design_path),
            "study_criteria_semantic_sha256": design.study_criteria_semantic_sha256,
            "study_criteria_file_sha256": design.study_criteria_file_sha256,
            "execution_manifest_sha256": manifest.sha256,
            "execution_manifest_file_sha256": sha256_file(manifest_path),
            "runtime_sha256": manifest.runtime.sha256,
            "execution_callable_authority": CANONICAL_EXECUTION_CALLABLE_AUTHORITY,
            "actuator_gear_by_joint": list(manifest.runtime.actuator_gear_by_joint),
            "generalized_actuator_torque_capacity_n_m": list(
                manifest.runtime.generalized_actuator_torque_capacity_n_m
            ),
            "source_tree_sha256": manifest.runtime.source_tree_sha256,
            "checkpoint_receipt_file_sha256": sha256_file(checkpoint_receipt),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "train_seed": train_seed,
            "evaluation_reference_sha256": manifest.runtime.reference_content_sha256,
            "evaluation_reference_schema_sha256": manifest.runtime.reference_schema_sha256,
            "evaluation_seeds_requested": list(design.evaluation_seeds),
            "evaluation_seeds_observed": list(design.evaluation_seeds),
            "max_episode_steps": design.max_episode_steps,
            "deterministic_actions": True,
            "protected_metrics_state_source": PROTECTED_METRICS_STATE_SOURCE,
            "station_keeping_origin_source": STATION_KEEPING_ORIGIN_SOURCE,
            "reward_telemetry_used_for_objective_metrics": False,
            "tracking_return_role": "environment_reward_diagnostic_only",
            "reference_timing_scope": "static_reference_only",
            "episodes": [_episode(seed) for seed in design.evaluation_seeds],
            "complete": True,
            "automatic_promotion": False,
            "behavioral_claim": None,
            "future_receipt_extension": {"version": 1},
        }
        evaluation_receipt = tmp_path / f"evaluation-{train_seed}.json"
        evaluation_receipt.write_text(
            json.dumps(evaluation_payload, indent=2) + "\n",
            encoding="utf-8",
        )
        checkpoint_receipts.append(checkpoint_receipt)
        evaluation_receipts.append(evaluation_receipt)
    return checkpoint_receipts, evaluation_receipts


def _aggregate(
    tmp_path: Path,
    *,
    criteria_path: Path = CRITERIA_PATH,
) -> tuple[dict[str, object], list[Path], list[Path], Path, Path]:
    design_path, manifest_path = _study_files(tmp_path)
    checkpoint_receipts, evaluation_receipts = _write_receipts(
        tmp_path,
        design_path=design_path,
        manifest_path=manifest_path,
    )
    result = aggregate_static_study(
        design_path=design_path,
        manifest_path=manifest_path,
        criteria_path=criteria_path,
        checkpoint_receipt_paths=checkpoint_receipts,
        evaluation_receipt_paths=evaluation_receipts,
        output_path=tmp_path / "summary.json",
    )
    return (
        result,
        checkpoint_receipts,
        evaluation_receipts,
        design_path,
        manifest_path,
    )


def test_committed_criteria_are_exact_and_naturalness_is_not_assessed() -> None:
    criteria = load_static_study_criteria(CRITERIA_PATH)

    assert [threshold.maximum for threshold in criteria.reward_scale_occupancy_bounds] == [
        0.2,
        0.5,
        1.0,
        2.0,
        0.35,
        2.0,
        0.35,
        2.0,
    ]
    assert criteria.required_evaluation_episodes == 20
    assert criteria.minimum_conforming_episodes == 19
    assert criteria.required_independent_train_seeds == 5
    assert criteria.minimum_conforming_checkpoints == 4
    assert criteria.bounded_control_and_prohibited_floor_contact_conformance_complete is True
    assert criteria.naturalness_calibration_status.startswith("not_calibrated")
    assert criteria.automatic_promotion is False


def test_exact_structural_schema_rejects_arbitrary_guardrail(tmp_path: Path) -> None:
    payload = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
    payload["guardrail_thresholds"]["thresholds"] = [
        {
            "field": "future_guardrail_metric",
            "maximum": 1.0,
            "unit": "1",
            "basis": "arbitrary",
        }
    ]
    changed = tmp_path / "arbitrary.criteria.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExperimentContractError, match="exact frozen"):
        load_static_study_criteria(changed)


def test_criteria_id_cannot_hide_a_smaller_design(tmp_path: Path) -> None:
    payload = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
    payload["checkpoint_rule"]["required_evaluation_episodes"] = 2
    payload["checkpoint_rule"]["minimum_conforming_episodes"] = 1
    payload["study_rule"]["required_independent_train_seeds"] = 2
    payload["study_rule"]["minimum_conforming_checkpoints"] = 1
    payload["uncertainty"]["resample_size"] = 2
    payload["uncertainty"]["expected_resample_count"] = 4
    changed = tmp_path / "smaller.criteria.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExperimentContractError, match="exact 20/19"):
        load_static_study_criteria(changed)


def test_criteria_schema_version_rejects_boolean_alias(tmp_path: Path) -> None:
    payload = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
    payload["schema_version"] = True
    changed = tmp_path / "boolean-schema.criteria.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExperimentContractError, match="schema_version"):
        load_static_study_criteria(changed)


def test_complete_receipts_yield_only_named_nonclaim_conformance(
    tmp_path: Path,
) -> None:
    result, _, _, _, _ = _aggregate(tmp_path)

    assert result["independent_unit"] == "train_seed"
    assert result["independent_n"] == 5
    assert result["evaluation_seeds_per_checkpoint"] == 20
    assert (
        result[
            "coarse_episode_rule_reward_scale_occupancy_bounded_control_and_prohibited_"
            "floor_contact_conformance"
        ]["conforms"]
        is True
    )
    assert result["study_decision"] == {
        "decision_status": (
            "blocked_missing_independently_justified_station_keeping_and_naturalness_thresholds"
        ),
        "decision_name": "static_stand_feasibility_pass",
        "guardrail_threshold_status": (
            "complete_bounded_control_and_prohibited_floor_contact_conformance_prebehavioral"
        ),
        "passing_checkpoints": None,
        "required_passing_checkpoints": None,
        "static_stand_feasibility_pass": None,
        "coarse_episode_rule_reward_scale_occupancy_bounded_control_and_prohibited_"
        "floor_contact_conformance": True,
        "study_pass": None,
        "station_keeping_assessed": False,
        "station_keeping_measurements_complete": True,
        "station_keeping_thresholds_defined": False,
        "station_keeping_requirement_status": (
            "blocked_missing_independently_justified_prebehavioral_thresholds"
        ),
        "unthresholded_station_keeping_metrics": [
            "root_horizontal_displacement_max_m",
            "root_linear_velocity_x_max_abs_m_s",
            "root_linear_velocity_y_max_abs_m_s",
            "root_linear_velocity_z_max_abs_m_s",
        ],
        "naturalness_assessed": False,
        "naturalness_calibration_status": ("not_calibrated_no_independent_neutral_controller"),
        "interpretation": (
            "episode_rule_reward_scale_occupancy_bounded_control_and_prohibited_floor_"
            "contact_conformance_only_not_objective_tracking_or_"
            "thresholded_station_keeping_naturalness_causal_reference_use_or_"
            "oracle_superiority"
        ),
    }
    assert (
        result[
            "coarse_episode_rule_reward_scale_occupancy_bounded_control_and_prohibited_"
            "floor_contact_conformance_"
            "seed_level_uncertainty"
        ]["checkpoint_conformance_fraction"]["resample_count"]
        == 3125
    )
    assert result["automatic_promotion"] is False
    assert result["behavioral_claim"] is None
    assert "floor_contact_peak_normal_force_n" in result["additional_numeric_metrics"]
    assert "root_horizontal_displacement_max_m" in result["station_keeping_descriptive_metrics"]
    assert "future_receipt_extension" in result["additional_receipt_fields"]
    assert Path(result["output_path"]).is_file()


def test_episode_checkpoint_and_study_boundaries_are_exact(tmp_path: Path) -> None:
    design_path, manifest_path = _study_files(tmp_path)
    checkpoint_receipts, evaluation_receipts = _write_receipts(
        tmp_path,
        design_path=design_path,
        manifest_path=manifest_path,
    )

    # One episode above the maximum leaves exactly 19/20: checkpoint passes.
    first = json.loads(evaluation_receipts[0].read_text(encoding="utf-8"))
    first["episodes"][0]["root_height_rmse_m"] = 0.2000000001
    evaluation_receipts[0].write_text(json.dumps(first), encoding="utf-8")

    # Two episodes above the maximum leave 18/20: checkpoint fails. Do this
    # for two checkpoints so only 3/5 meet the coarse conformance rule.
    for receipt_path in evaluation_receipts[1:3]:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        payload["episodes"][0]["root_height_rmse_m"] = 0.2000000001
        payload["episodes"][1]["root_height_rmse_m"] = 0.2000000001
        receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    # Equality is inclusive and must remain a successful episode.
    fourth = json.loads(evaluation_receipts[3].read_text(encoding="utf-8"))
    fourth["episodes"][0]["root_height_rmse_m"] = 0.2
    evaluation_receipts[3].write_text(json.dumps(fourth), encoding="utf-8")

    result = aggregate_static_study(
        design_path=design_path,
        manifest_path=manifest_path,
        criteria_path=CRITERIA_PATH,
        checkpoint_receipt_paths=checkpoint_receipts,
        evaluation_receipt_paths=evaluation_receipts,
        output_path=tmp_path / "boundary-summary.json",
    )
    summaries = result["checkpoint_summaries"]
    assert summaries[0]["episode_rule_and_reward_scale_occupancy_episode_count"] == 19
    assert summaries[0]["episode_rule_and_reward_scale_occupancy_checkpoint_conformance"] is True
    assert summaries[1]["episode_rule_and_reward_scale_occupancy_episode_count"] == 18
    assert summaries[1]["episode_rule_and_reward_scale_occupancy_checkpoint_conformance"] is False
    assert summaries[3]["episode_rule_and_reward_scale_occupancy_episode_count"] == 20
    assert result[
        "coarse_episode_rule_reward_scale_occupancy_bounded_control_and_prohibited_"
        "floor_contact_conformance"
    ] == {
        "conforming_checkpoints": 3,
        "required_conforming_checkpoints": 4,
        "conforms": False,
        "episode_rule_reward_scale_only_conforming_checkpoints": 3,
        "episode_rule_reward_scale_only_conforms": False,
    }
    assert result["study_decision"]["static_stand_feasibility_pass"] is None
    assert result["study_decision"]["study_pass"] is None


def test_exhaustive_bootstrap_uses_only_five_seed_level_values() -> None:
    result = exhaustive_seed_bootstrap([0.0, 0.0, 0.0, 0.0, 1.0])

    assert result["independent_n"] == 5
    assert result["resample_count"] == 3125
    assert result["point_estimate_seed_mean"] == pytest.approx(0.2)
    assert result["interval_lower"] == pytest.approx(0.0)
    assert result["interval_upper"] == pytest.approx(0.6)
    with pytest.raises(ExperimentContractError, match="exactly five"):
        exhaustive_seed_bootstrap([0.0] * 100)


def test_criteria_semantic_and_file_hashes_are_bound_in_design(tmp_path: Path) -> None:
    semantic_directory = tmp_path / "semantic"
    semantic_directory.mkdir()
    design_path, manifest_path = _study_files(
        semantic_directory,
        criteria_semantic_sha256=SHA_A,
    )
    checkpoint_receipts, evaluation_receipts = _write_receipts(
        semantic_directory,
        design_path=design_path,
        manifest_path=manifest_path,
    )
    with pytest.raises(ExperimentContractError, match="semantic SHA-256"):
        aggregate_static_study(
            design_path=design_path,
            manifest_path=manifest_path,
            criteria_path=CRITERIA_PATH,
            checkpoint_receipt_paths=checkpoint_receipts,
            evaluation_receipt_paths=evaluation_receipts,
            output_path=semantic_directory / "semantic-rejected.json",
        )

    formatting_directory = tmp_path / "formatting"
    formatting_directory.mkdir()
    formatting_change = formatting_directory / "formatting-change.criteria.json"
    formatting_change.write_text(
        json.dumps(json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))),
        encoding="utf-8",
    )
    design_path, manifest_path = _study_files(formatting_directory)
    checkpoint_receipts, evaluation_receipts = _write_receipts(
        formatting_directory,
        design_path=design_path,
        manifest_path=manifest_path,
    )
    with pytest.raises(ExperimentContractError, match="file SHA-256"):
        aggregate_static_study(
            design_path=design_path,
            manifest_path=manifest_path,
            criteria_path=formatting_change,
            checkpoint_receipt_paths=checkpoint_receipts,
            evaluation_receipt_paths=evaluation_receipts,
            output_path=formatting_directory / "formatting-rejected.json",
        )


def test_aggregator_requires_frozen_prebehavioral_control_period_basis(tmp_path: Path) -> None:
    design_path, manifest_path = _study_files(tmp_path)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["runtime"]["control_period_seconds"] = 0.02
    manifest_path.write_text(json.dumps(manifest_payload, indent=2) + "\n", encoding="utf-8")
    checkpoint_receipts, evaluation_receipts = _write_receipts(
        tmp_path,
        design_path=design_path,
        manifest_path=manifest_path,
    )

    with pytest.raises(ExperimentContractError, match=r"0\.015 s rate basis"):
        aggregate_static_study(
            design_path=design_path,
            manifest_path=manifest_path,
            criteria_path=CRITERIA_PATH,
            checkpoint_receipt_paths=checkpoint_receipts,
            evaluation_receipt_paths=evaluation_receipts,
            output_path=tmp_path / "wrong-cadence.json",
        )


def test_criteria_rejects_post_hoc_change_to_reward_scale(tmp_path: Path) -> None:
    payload = json.loads(CRITERIA_PATH.read_text(encoding="utf-8"))
    payload["reward_scale_derived_occupancy_bounds"][0]["maximum"] = 0.21
    changed = tmp_path / "changed.criteria.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExperimentContractError, match="not tied to its scale"):
        load_static_study_criteria(changed)


def test_aggregator_rejects_duplicate_or_missing_training_seed(tmp_path: Path) -> None:
    design_path, manifest_path = _study_files(tmp_path)
    checkpoint_receipts, evaluation_receipts = _write_receipts(
        tmp_path,
        design_path=design_path,
        manifest_path=manifest_path,
    )
    with pytest.raises(ExperimentContractError, match="paths must not contain duplicates"):
        aggregate_static_study(
            design_path=design_path,
            manifest_path=manifest_path,
            criteria_path=CRITERIA_PATH,
            checkpoint_receipt_paths=checkpoint_receipts,
            evaluation_receipt_paths=[*evaluation_receipts[:4], evaluation_receipts[0]],
            output_path=tmp_path / "duplicate-path.json",
        )

    payload = json.loads(evaluation_receipts[-1].read_text(encoding="utf-8"))
    payload["train_seed"] = 101
    evaluation_receipts[-1].write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ExperimentContractError, match="missing, duplicated"):
        aggregate_static_study(
            design_path=design_path,
            manifest_path=manifest_path,
            criteria_path=CRITERIA_PATH,
            checkpoint_receipt_paths=checkpoint_receipts,
            evaluation_receipt_paths=evaluation_receipts,
            output_path=tmp_path / "duplicate-seed.json",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(schema_version=True), "schema_version"),
        (lambda payload: payload.update(complete=1), "complete must be boolean"),
        (
            lambda payload: payload.update(automatic_promotion=0),
            "automatic_promotion must be boolean",
        ),
        (
            lambda payload: payload.update(deterministic_actions=1),
            "deterministic_actions must be boolean",
        ),
        (
            lambda payload: payload.update(reward_telemetry_used_for_objective_metrics=0),
            "reward_telemetry_used_for_objective_metrics must be boolean",
        ),
        (
            lambda payload: payload.update(max_episode_steps=True),
            "max_episode_steps must be an integer",
        ),
        (lambda payload: payload.update(complete=False), "complete"),
        (
            lambda payload: payload["evaluation_seeds_observed"].reverse(),
            "evaluation_seeds_observed",
        ),
        (
            lambda payload: payload.update(
                evaluation_seeds_requested=[
                    float(seed) for seed in payload["evaluation_seeds_requested"]
                ]
            ),
            "evaluation_seeds_requested must contain only integer seeds",
        ),
        (
            lambda payload: payload["episodes"][0].update(observed_steps=999),
            "incomplete",
        ),
        (
            lambda payload: payload.update(runtime_sha256=SHA_B),
            "runtime_sha256",
        ),
        (
            lambda payload: payload.update(study_criteria_semantic_sha256=SHA_B),
            "study_criteria_semantic_sha256",
        ),
        (
            lambda payload: payload.update(study_criteria_file_sha256=SHA_B),
            "study_criteria_file_sha256",
        ),
        (
            lambda payload: payload.update(protected_metrics_state_source="reward_info/v0"),
            "protected_metrics_state_source",
        ),
    ],
)
def test_aggregator_rejects_incomplete_or_mismatched_evaluation_receipt(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    design_path, manifest_path = _study_files(tmp_path)
    checkpoint_receipts, evaluation_receipts = _write_receipts(
        tmp_path,
        design_path=design_path,
        manifest_path=manifest_path,
    )
    payload = json.loads(evaluation_receipts[0].read_text(encoding="utf-8"))
    mutation(payload)
    evaluation_receipts[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExperimentContractError, match=message):
        aggregate_static_study(
            design_path=design_path,
            manifest_path=manifest_path,
            criteria_path=CRITERIA_PATH,
            checkpoint_receipt_paths=checkpoint_receipts,
            evaluation_receipt_paths=evaluation_receipts,
            output_path=tmp_path / "rejected.json",
        )


@pytest.mark.parametrize("invalid_kind", ["duplicate", "nonfinite", "oversized"])
def test_aggregator_strictly_reads_evaluation_receipts(
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    design_path, manifest_path = _study_files(tmp_path)
    checkpoint_receipts, evaluation_receipts = _write_receipts(
        tmp_path,
        design_path=design_path,
        manifest_path=manifest_path,
    )
    receipt_path = evaluation_receipts[0]
    encoded = receipt_path.read_bytes()
    if invalid_kind == "duplicate":
        encoded = encoded.rstrip()[:-1] + b', "complete": true}\n'
    elif invalid_kind == "nonfinite":
        encoded = encoded.replace(
            b'"root_height_rmse_m": 0.1',
            b'"root_height_rmse_m": NaN',
            1,
        )
    else:
        encoded += b" " * study_summary_module.MAX_EVALUATION_RECEIPT_BYTES
    receipt_path.write_bytes(encoded)

    with pytest.raises(
        ExperimentContractError,
        match=r"duplicate key|non-finite|bounded size limit",
    ):
        aggregate_static_study(
            design_path=design_path,
            manifest_path=manifest_path,
            criteria_path=CRITERIA_PATH,
            checkpoint_receipt_paths=checkpoint_receipts,
            evaluation_receipt_paths=evaluation_receipts,
            output_path=tmp_path / "strict-receipt-rejected.json",
        )


@pytest.mark.parametrize(
    ("episode_mutation", "message"),
    [
        (
            lambda episode: episode.update(
                joint_position_rmse_rad=0.3,
                joint_position_max_abs_rad=0.0,
            ),
            "cannot exceed",
        ),
        (
            lambda episode: episode.update(
                normalized_policy_action_rms=1.0,
                normalized_policy_action_max_abs=0.0,
            ),
            "cannot exceed",
        ),
        (
            lambda episode: episode.update(
                normalized_policy_action_delta_rate_rms_per_s=0.0,
                normalized_policy_action_delta_rate_max_abs_per_s=0.0,
            ),
            "too small for the zero-reset",
        ),
        (
            lambda episode: episode.update(
                simulator_contact_count=0,
                floor_contact_count=10,
            ),
            "contact counts",
        ),
        (
            lambda episode: episode.update(tracking_return=1001.0),
            "tracking_return",
        ),
        (
            lambda episode: episode.update(simulator_contact_count=True),
            "must be an integer",
        ),
    ],
)
def test_aggregator_rejects_contradictory_episode_algebra(
    tmp_path: Path,
    episode_mutation,
    message: str,
) -> None:
    design_path, manifest_path = _study_files(tmp_path)
    checkpoint_receipts, evaluation_receipts = _write_receipts(
        tmp_path,
        design_path=design_path,
        manifest_path=manifest_path,
    )
    payload = json.loads(evaluation_receipts[0].read_text(encoding="utf-8"))
    episode_mutation(payload["episodes"][0])
    evaluation_receipts[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExperimentContractError, match=message):
        aggregate_static_study(
            design_path=design_path,
            manifest_path=manifest_path,
            criteria_path=CRITERIA_PATH,
            checkpoint_receipt_paths=checkpoint_receipts,
            evaluation_receipt_paths=evaluation_receipts,
            output_path=tmp_path / "contradictory-episode.json",
        )


@pytest.mark.parametrize(
    "runtime_field",
    [
        "execution_source_sha256",
        "experiment_contract_source_sha256",
        "tracking_reward_source_sha256",
        "evaluator_source_sha256",
        "policy_source_sha256",
        "study_summary_source_sha256",
        "source_tree_sha256",
    ],
)
def test_aggregator_rejects_decision_source_drift(
    tmp_path: Path,
    runtime_field: str,
) -> None:
    design_path, manifest_path = _study_files(tmp_path)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_payload["runtime"][runtime_field] = "0" * 64
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    checkpoint_receipts, evaluation_receipts = _write_receipts(
        tmp_path,
        design_path=design_path,
        manifest_path=manifest_path,
    )

    with pytest.raises(ExperimentContractError, match="source differs"):
        aggregate_static_study(
            design_path=design_path,
            manifest_path=manifest_path,
            criteria_path=CRITERIA_PATH,
            checkpoint_receipt_paths=checkpoint_receipts,
            evaluation_receipt_paths=evaluation_receipts,
            output_path=tmp_path / "source-drift.json",
        )


def test_aggregator_reopens_and_rehashes_sibling_checkpoint(tmp_path: Path) -> None:
    design_path, manifest_path = _study_files(tmp_path)
    checkpoint_receipts, evaluation_receipts = _write_receipts(
        tmp_path,
        design_path=design_path,
        manifest_path=manifest_path,
    )
    checkpoint_path = checkpoint_receipts[0].parent / CHECKPOINT_FILENAME
    checkpoint_path.write_bytes(b"tampered-after-evaluation")

    with pytest.raises(ExperimentContractError, match="checkpoint SHA-256"):
        aggregate_static_study(
            design_path=design_path,
            manifest_path=manifest_path,
            criteria_path=CRITERIA_PATH,
            checkpoint_receipt_paths=checkpoint_receipts,
            evaluation_receipt_paths=evaluation_receipts,
            output_path=tmp_path / "tampered-checkpoint.json",
        )


def test_aggregator_rejects_noncanonical_execution_authority(tmp_path: Path) -> None:
    design_path, manifest_path = _study_files(tmp_path)
    manifest = FrozenExecutionManifest.from_dict(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    test_only = replace(
        manifest,
        status="test_only_non_authoritative",
        execution_callable_authority=NONAUTHORITATIVE_EXECUTION_CALLABLE_AUTHORITY,
    )
    manifest_path.write_text(
        json.dumps(test_only.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    checkpoint_receipts, evaluation_receipts = _write_receipts(
        tmp_path,
        design_path=design_path,
        manifest_path=manifest_path,
    )

    with pytest.raises(ExperimentContractError, match="canonical production"):
        aggregate_static_study(
            design_path=design_path,
            manifest_path=manifest_path,
            criteria_path=CRITERIA_PATH,
            checkpoint_receipt_paths=checkpoint_receipts,
            evaluation_receipt_paths=evaluation_receipts,
            output_path=tmp_path / "noncanonical.json",
        )


def test_aggregator_rejects_arbitrary_checkpoint_hash_in_evaluation(tmp_path: Path) -> None:
    design_path, manifest_path = _study_files(tmp_path)
    checkpoint_receipts, evaluation_receipts = _write_receipts(
        tmp_path,
        design_path=design_path,
        manifest_path=manifest_path,
    )
    payload = json.loads(evaluation_receipts[0].read_text(encoding="utf-8"))
    payload["checkpoint_sha256"] = hashlib.sha256(b"arbitrary").hexdigest()
    evaluation_receipts[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ExperimentContractError, match="revalidated checkpoint"):
        aggregate_static_study(
            design_path=design_path,
            manifest_path=manifest_path,
            criteria_path=CRITERIA_PATH,
            checkpoint_receipt_paths=checkpoint_receipts,
            evaluation_receipt_paths=evaluation_receipts,
            output_path=tmp_path / "arbitrary-checkpoint.json",
        )


def test_aggregator_refuses_to_overwrite_summary(tmp_path: Path) -> None:
    result, checkpoint_receipts, evaluation_receipts, design_path, manifest_path = _aggregate(
        tmp_path
    )
    assert Path(result["output_path"]).is_file()

    with pytest.raises(ExperimentContractError, match="overwrite"):
        aggregate_static_study(
            design_path=design_path,
            manifest_path=manifest_path,
            criteria_path=CRITERIA_PATH,
            checkpoint_receipt_paths=checkpoint_receipts,
            evaluation_receipt_paths=evaluation_receipts,
            output_path=Path(result["output_path"]),
        )
