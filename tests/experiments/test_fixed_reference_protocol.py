from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.experiments import (
    BehavioralEvaluationManifest,
    EvidencePurpose,
    ExperimentContractError,
    FrozenExecutionManifest,
    ResourceCalibrationReceipt,
    RuntimeFingerprint,
    assert_matched_evaluation_manifests,
    assert_runtime_matches,
    assert_temporal_reference_variation,
    checkpoint_receipt,
    load_study_design,
)
from oracle_composition.experiments.fixed_reference import (
    CANONICAL_EXECUTION_CALLABLE_AUTHORITY,
    MAX_STUDY_DESIGN_BYTES,
)
from oracle_composition.experiments.fixed_reference_runner import (
    _source_tree_sha256,
    inspect_runtime,
    make_static_tracking_env,
)

ROOT = Path(__file__).resolve().parents[2]
DESIGN_PATH = (
    ROOT
    / "experiments"
    / "001_humanoid_fixed_reference"
    / "configs"
    / "static_stand_precalibration_v0.study.json"
)
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64
SHA_1 = "1" * 64
SHA_2 = "2" * 64


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
        tracking_reward_source_sha256=SHA_C,
        wrapper_source_sha256=SHA_B,
        experiment_contract_source_sha256=SHA_C,
        evaluator_source_sha256=SHA_C,
        runner_source_sha256=SHA_A,
        execution_source_sha256=SHA_B,
        study_summary_source_sha256=SHA_A,
        policy_source_sha256=SHA_C,
        source_tree_sha256=SHA_B,
        policy_id="local.SquashedGaussianActorCriticPolicy/tanh_jacobian/v1",
        action_transform_id="tanh_normalized_then_affine_physical_box/v1",
        observation_normalizer_id="none/v1",
        reward_normalizer_id="none/v1",
    )


def _condition(
    condition_id: str,
    policy_input_reference_sha256: str,
    policy_input_transform_receipt_sha256: str,
) -> BehavioralEvaluationManifest:
    return BehavioralEvaluationManifest(
        condition_id=condition_id,
        intervention_protocol_id="matched_state_numeric_reference_intervention/v1",
        design_sha256=SHA_A,
        runtime_sha256=SHA_B,
        evaluator_sha256=SHA_C,
        ground_truth_reference_sha256=SHA_D,
        matched_state_snapshot_set_sha256=SHA_E,
        policy_input_reference_sha256=policy_input_reference_sha256,
        policy_input_transform_receipt_sha256=policy_input_transform_receipt_sha256,
        checkpoint_sha256_by_train_seed=((101, SHA_A), (202, SHA_B)),
        evaluation_seeds=(11001, 11002),
        max_episode_steps=1000,
        deterministic_actions=True,
        perturbation_schedule_sha256=SHA_C,
    )


def test_committed_design_is_proposed_exact_budget_and_static_only() -> None:
    design = load_study_design(DESIGN_PATH)

    assert design.train_seeds == (101, 202, 303, 404, 505)
    assert len(design.evaluation_seeds) == 20
    assert design.ppo.total_timesteps_per_seed == 1_048_576
    assert design.ppo.total_timesteps_per_seed % (design.ppo.n_envs * design.ppo.n_steps) == 0
    assert design.claim_ceiling.value == "static_tracking_feasibility"
    assert design.design_status == "proposed_pre_calibration"
    assert design.causal_use_evaluation_enabled is False
    assert design.causal_use_block_reason
    assert len(design.sha256) == 64


def test_study_loader_rejects_unknown_fields(tmp_path: Path) -> None:
    payload = DESIGN_PATH.read_text(encoding="utf-8").rstrip()[:-1] + ', "fallback": true}'
    path = tmp_path / "changed.json"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(ExperimentContractError, match=r"extra=.*fallback"):
        load_study_design(path)


@pytest.mark.parametrize(
    ("invalid_kind", "message"),
    [
        ("duplicate", "duplicate key"),
        ("nonfinite", "non-finite"),
        ("oversized", "bounded size limit"),
        ("invalid_utf8", "cannot read study design"),
    ],
)
def test_study_loader_requires_bounded_strict_utf8_json(
    tmp_path: Path,
    invalid_kind: str,
    message: str,
) -> None:
    encoded = DESIGN_PATH.read_bytes()
    if invalid_kind == "duplicate":
        encoded = encoded.rstrip()[:-1] + b', "schema_version": 1}\n'
    elif invalid_kind == "nonfinite":
        encoded = encoded.replace(b'"learning_rate": 0.0003', b'"learning_rate": NaN')
    elif invalid_kind == "oversized":
        encoded += b" " * MAX_STUDY_DESIGN_BYTES
    else:
        encoded = b"\xff" + encoded
    path = tmp_path / "invalid.study.json"
    path.write_bytes(encoded)

    with pytest.raises(ExperimentContractError, match=message):
        load_study_design(path)


def test_runtime_and_manifest_fail_closed_on_any_drift() -> None:
    proposed = load_study_design(DESIGN_PATH)
    with pytest.raises(ExperimentContractError, match="schema_version"):
        replace(proposed, schema_version=True)
    design = replace(proposed, design_version="locked-v1", design_status="locked_pre_behavioral")
    runtime = _runtime()
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
        runtime=runtime,
        candidate_file_sha256=SHA_C,
        reviewer_id="test-reviewer",
        review_note="Reviewed for the contract unit test.",
    )

    manifest.validate(design=design, observed=runtime)
    with pytest.raises(ExperimentContractError, match="schema_version"):
        replace(manifest, schema_version=True)
    changed = replace(runtime, gymnasium_version="1.3.1")
    with pytest.raises(ExperimentContractError, match="gymnasium_version"):
        assert_runtime_matches(runtime, changed)
    with pytest.raises(ExperimentContractError, match="runtime fingerprint"):
        manifest.validate(design=design, observed=changed)
    changed_gears = list(runtime.actuator_gear_by_joint)
    changed_gears[0] = 99.0
    with pytest.raises(ExperimentContractError, match="pinned Humanoid-v5 actuator ABI"):
        replace(runtime, actuator_gear_by_joint=tuple(changed_gears))
    with pytest.raises(ExperimentContractError, match="does not bind"):
        replace(manifest, design_sha256=SHA_B).validate(design=design, observed=runtime)

    proposed = load_study_design(DESIGN_PATH)
    with pytest.raises(ExperimentContractError, match="not locked"):
        replace(manifest, design_sha256=proposed.sha256).validate(design=proposed, observed=runtime)


def test_checkpoint_receipt_hashes_exact_bytes_and_requires_exact_budget(tmp_path: Path) -> None:
    proposed = load_study_design(DESIGN_PATH)
    design = replace(proposed, design_version="locked-v1", design_status="locked_pre_behavioral")
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"exact-checkpoint-bytes")

    receipt = checkpoint_receipt(
        design=design,
        runtime=_runtime(),
        train_seed=101,
        observed_timesteps=design.ppo.total_timesteps_per_seed,
        checkpoint_path=checkpoint,
    )

    assert receipt.purpose is EvidencePurpose.BEHAVIORAL_EVALUATION
    assert receipt.checkpoint_size_bytes == len(b"exact-checkpoint-bytes")
    assert len(receipt.checkpoint_sha256) == 64
    checkpoint.write_bytes(b"changed-checkpoint-bytes")
    changed = checkpoint_receipt(
        design=design,
        runtime=_runtime(),
        train_seed=101,
        observed_timesteps=design.ppo.total_timesteps_per_seed,
        checkpoint_path=checkpoint,
    )
    assert changed.checkpoint_sha256 != receipt.checkpoint_sha256

    with pytest.raises(ExperimentContractError, match="exactly equal"):
        checkpoint_receipt(
            design=design,
            runtime=_runtime(),
            train_seed=101,
            observed_timesteps=design.ppo.total_timesteps_per_seed - 1,
            checkpoint_path=checkpoint,
        )


def test_resource_calibration_receipt_cannot_be_relabelled_as_behavior() -> None:
    receipt = ResourceCalibrationReceipt(
        purpose=EvidencePurpose.RESOURCE_CALIBRATION,
        design_sha256=SHA_A,
        runtime_sha256=SHA_B,
        calibration_seed=909,
        environment_steps=8192,
        wall_seconds=10.0,
        peak_resident_bytes=None,
    )
    assert receipt.purpose is EvidencePurpose.RESOURCE_CALIBRATION

    with pytest.raises(ExperimentContractError, match="resource_calibration"):
        replace(receipt, purpose=EvidencePurpose.BEHAVIORAL_EVALUATION)


def test_proposed_design_cannot_create_behavioral_checkpoint(tmp_path: Path) -> None:
    design = load_study_design(DESIGN_PATH)
    checkpoint = tmp_path / "policy.zip"
    checkpoint.write_bytes(b"not-eligible")

    with pytest.raises(ExperimentContractError, match="not locked"):
        checkpoint_receipt(
            design=design,
            runtime=_runtime(),
            train_seed=101,
            observed_timesteps=design.ppo.total_timesteps_per_seed,
            checkpoint_path=checkpoint,
        )


def test_static_reference_is_rejected_for_causal_use_but_dynamic_reference_passes() -> None:
    static = np.zeros((1000, 45), dtype=np.float64)
    with pytest.raises(ExperimentContractError, match="time-invariant"):
        assert_temporal_reference_variation(static)

    dynamic = static.copy()
    dynamic[500:, 11] = 0.2
    assert_temporal_reference_variation(dynamic)


def test_evaluation_conditions_hold_ground_truth_fixed_and_vary_only_policy_input() -> None:
    exact = _condition("C_exact", SHA_D, SHA_A)
    constant = _condition("C_constant_frame_input", SHA_E, SHA_B)
    shuffled = _condition("C_shuffle_input", SHA_F, SHA_C)
    shifted = _condition("C_shift_input", SHA_1, SHA_2)
    manifests = (exact, constant, shuffled, shifted)
    assert_matched_evaluation_manifests(manifests)

    with pytest.raises(ExperimentContractError, match="evaluation_seeds"):
        assert_matched_evaluation_manifests(
            (exact, constant, replace(shuffled, evaluation_seeds=(11001, 11003)), shifted)
        )
    with pytest.raises(ExperimentContractError, match="evaluation_seeds"):
        assert_matched_evaluation_manifests(
            (
                replace(constant, evaluation_seeds=(11001, 11003)),
                shuffled,
                shifted,
                exact,
            )
        )
    with pytest.raises(ExperimentContractError, match="non-negative integers"):
        replace(shuffled, evaluation_seeds=(11001, -1))
    with pytest.raises(ExperimentContractError, match="checkpoint mapping"):
        replace(shuffled, checkpoint_sha256_by_train_seed=((101, SHA_A, SHA_B),))
    with pytest.raises(ExperimentContractError, match="distinct policy-input bytes"):
        assert_matched_evaluation_manifests(
            (
                exact,
                constant,
                replace(
                    shuffled, policy_input_reference_sha256=constant.policy_input_reference_sha256
                ),
                shifted,
            )
        )
    with pytest.raises(ExperimentContractError, match="common ground-truth"):
        assert_matched_evaluation_manifests(
            (replace(exact, policy_input_reference_sha256=SHA_A), constant, shuffled, shifted)
        )
    with pytest.raises(ExperimentContractError, match="must change the policy-input"):
        assert_matched_evaluation_manifests(
            (
                exact,
                replace(constant, policy_input_reference_sha256=SHA_D),
                shuffled,
                shifted,
            )
        )
    with pytest.raises(ExperimentContractError, match="ground_truth_reference_sha256"):
        assert_matched_evaluation_manifests(
            (exact, constant, shuffled, replace(shifted, ground_truth_reference_sha256=SHA_2))
        )
    with pytest.raises(ExperimentContractError, match="distinct transform receipt"):
        assert_matched_evaluation_manifests(
            (
                exact,
                constant,
                shuffled,
                replace(shifted, policy_input_transform_receipt_sha256=SHA_C),
            )
        )
    with pytest.raises(ExperimentContractError, match="exactly four conditions"):
        assert_matched_evaluation_manifests((exact, shuffled))
    with pytest.raises(ExperimentContractError, match="condition IDs"):
        assert_matched_evaluation_manifests(
            (exact, constant, shuffled, replace(shifted, condition_id="C_unknown"))
        )


@pytest.mark.gym
def test_real_runtime_binds_time_limit_spaces_platform_and_execution_sources() -> None:
    design = load_study_design(DESIGN_PATH)
    runtime = inspect_runtime(design)

    assert runtime.environment_max_episode_steps == design.max_episode_steps == 1000
    assert runtime.observation_shape == (528,)
    assert runtime.action_shape == (17,)
    assert runtime.action_transform_id == "tanh_normalized_then_affine_physical_box/v1"
    assert runtime.policy_id == "local.SquashedGaussianActorCriticPolicy/tanh_jacobian/v1"
    assert runtime.physical_action_space_sha256 != runtime.action_space_sha256
    assert runtime.observation_normalizer_id == "none/v1"
    assert runtime.reward_normalizer_id == "none/v1"
    assert runtime.actuator_gear_by_joint == (
        (100.0,) * 5 + (300.0, 200.0, 100.0, 100.0, 300.0, 200.0) + (25.0,) * 6
    )
    assert runtime.generalized_actuator_torque_capacity_n_m == (
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
    assert all(
        len(getattr(runtime, field)) == 64
        for field in (
            "dependency_lock_sha256",
            "observation_space_sha256",
            "action_space_sha256",
            "environment_source_sha256",
            "reference_abi_source_sha256",
            "tracking_reward_source_sha256",
            "wrapper_source_sha256",
            "experiment_contract_source_sha256",
            "evaluator_source_sha256",
            "runner_source_sha256",
            "execution_source_sha256",
            "source_tree_sha256",
        )
    )


def test_source_tree_digest_changes_with_included_source_bytes(tmp_path: Path) -> None:
    package = tmp_path / "oracle_composition"
    package.mkdir()
    source = package / "reference.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    initial = _source_tree_sha256(package)

    source.write_text("VALUE = 2\n", encoding="utf-8")

    assert _source_tree_sha256(package) != initial


@pytest.mark.gym
def test_real_runner_rejects_design_horizon_that_differs_from_gym_time_limit() -> None:
    proposed = load_study_design(DESIGN_PATH)
    changed = replace(proposed, max_episode_steps=999)

    with pytest.raises(RuntimeError, match="does not match the installed Gym TimeLimit"):
        make_static_tracking_env(changed)
