from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.experiments.execution import (
    MAX_EXECUTION_ARTIFACT_BYTES,
    _initial_protected_history,
    _load_ppo,
    _make_ppo,
    _make_vector_env,
    _protected_step_inputs,
    _validated_evaluation_environment,
    calibrate_resources,
    emit_manifest_candidate,
    evaluate_checkpoint,
    finalize_reviewed_manifest,
    load_execution_manifest,
    train_one_seed,
)
from oracle_composition.experiments.fixed_reference import (
    CANONICAL_EXECUTION_CALLABLE_AUTHORITY,
    NONAUTHORITATIVE_EXECUTION_CALLABLE_AUTHORITY,
    ExperimentContractError,
    FrozenExecutionManifest,
    RuntimeFingerprint,
    load_study_design,
)
from oracle_composition.experiments.fixed_reference_runner import make_static_tracking_env
from oracle_composition.experiments.protected_evaluator import (
    STATION_KEEPING_ORIGIN_SOURCE,
)
from oracle_composition.tracking import (
    HumanoidTrackingState,
    make_static_stand_reference,
    validate_humanoid_actuator_abi,
)

ROOT = Path(__file__).resolve().parents[2]
PROPOSED_PATH = (
    ROOT
    / "experiments"
    / "001_humanoid_fixed_reference"
    / "configs"
    / "static_stand_precalibration_v0.study.json"
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
        observation_shape=(438,),
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
        environment_max_episode_steps=2,
        control_period_seconds=0.015,
        reference_content_sha256=make_static_stand_reference(n_frames=2).identity.content_sha256,
        reference_schema_sha256=make_static_stand_reference(n_frames=2).identity.schema_sha256,
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


def _design_path(tmp_path: Path, *, locked: bool) -> Path:
    proposed = load_study_design(PROPOSED_PATH)
    design = replace(
        proposed,
        design_version="locked-test-v1" if locked else "proposed-test-v1",
        design_status="locked_pre_behavioral" if locked else "proposed_pre_calibration",
        train_seeds=(101, 202),
        evaluation_seeds=(11001, 11002),
        max_episode_steps=2,
        horizon_steps=2,
        static_reference_frames=2,
        ppo=replace(
            proposed.ppo,
            total_timesteps_per_seed=4,
            n_envs=1,
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            policy_hidden_layers=(8, 8),
        ),
    )
    path = tmp_path / ("locked.study.json" if locked else "proposed.study.json")
    path.write_text(json.dumps(design.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def _manifest_path(tmp_path: Path, design_path: Path) -> Path:
    design = load_study_design(design_path)
    manifest = FrozenExecutionManifest(
        schema_version=1,
        status="test_only_non_authoritative",
        execution_callable_authority=NONAUTHORITATIVE_EXECUTION_CALLABLE_AUTHORITY,
        design_sha256=design.sha256,
        calibrated_design_sha256=SHA_A,
        calibrated_design_file_sha256=SHA_B,
        calibration_receipt_sha256=SHA_C,
        calibration_receipt_file_sha256=SHA_A,
        study_criteria_semantic_sha256=design.study_criteria_semantic_sha256,
        study_criteria_file_sha256=design.study_criteria_file_sha256,
        runtime=_runtime(),
        candidate_file_sha256=SHA_C,
        reviewer_id="test-reviewer",
        review_note="Reviewed for the execution unit test.",
    )
    path = tmp_path / "execution.manifest.json"
    path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def _validator(*, design, manifest):
    runtime = _runtime()
    manifest.validate(design=design, observed=runtime)
    return runtime


class _FakeVectorEnvironment:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeTrainingModel:
    def __init__(self, *, short_by: int = 0) -> None:
        self.num_timesteps = 0
        self.short_by = short_by
        self.save_calls = 0

    def learn(
        self,
        *,
        total_timesteps,
        reset_num_timesteps,
        progress_bar,
        callback=None,
    ):
        assert reset_num_timesteps is True
        assert progress_bar is False
        if callback is not None:
            callback.audited_rollouts = total_timesteps // 2
            callback.max_abs_stored_action = 0.5
            callback.max_log_probability_abs_error = 0.0
        self.num_timesteps = total_timesteps - self.short_by
        return self

    def save(self, path: Path) -> None:
        self.save_calls += 1
        path.write_bytes(b"one-final-sb3-checkpoint")


def _calibration_receipt(
    tmp_path: Path,
    design_path: Path,
    *,
    directory: str = "matched-calibration",
) -> Path:
    result = calibrate_resources(
        design_path=design_path,
        calibration_seed=909,
        environment_steps=2,
        runs_dir=tmp_path / directory / "runs",
        runtime_inspector=lambda _design: _runtime(),
        vector_env_factory=lambda _design, seed: _FakeVectorEnvironment(),
        model_factory=lambda _design, _environment, seed: _FakeTrainingModel(),
    )
    return Path(result["receipt_path"])


def test_manifest_candidate_requires_locked_design_and_never_authorizes(tmp_path: Path) -> None:
    proposed = _design_path(tmp_path, locked=False)
    calibration_receipt = _calibration_receipt(tmp_path, proposed)
    inspector_called = False

    def inspector(_design):
        nonlocal inspector_called
        inspector_called = True
        return _runtime()

    with pytest.raises(ExperimentContractError, match="not locked"):
        emit_manifest_candidate(
            design_path=proposed,
            calibrated_design_path=proposed,
            calibration_receipt_path=calibration_receipt,
            output_path=tmp_path / "rejected.json",
            runtime_inspector=inspector,
        )
    assert inspector_called is False

    locked = _design_path(tmp_path, locked=True)
    output = tmp_path / "candidate.json"
    result = emit_manifest_candidate(
        design_path=locked,
        calibrated_design_path=proposed,
        calibration_receipt_path=calibration_receipt,
        output_path=output,
        runtime_inspector=inspector,
    )
    with pytest.raises(ExperimentContractError, match=r"execution manifest.*keys mismatch"):
        load_execution_manifest(output)
    assert result["automatic_authorization"] is False
    assert result["execution_callable_authority"] == NONAUTHORITATIVE_EXECUTION_CALLABLE_AUTHORITY
    with pytest.raises(ExperimentContractError, match="overwrite"):
        emit_manifest_candidate(
            design_path=locked,
            calibrated_design_path=proposed,
            calibration_receipt_path=calibration_receipt,
            output_path=output,
            runtime_inspector=inspector,
        )

    frozen = tmp_path / "reviewed.manifest.json"
    finalized = finalize_reviewed_manifest(
        design_path=locked,
        calibrated_design_path=proposed,
        calibration_receipt_path=calibration_receipt,
        candidate_path=output,
        output_path=frozen,
        reviewer_id="human-reviewer",
        review_note="Checked the locked design and exact runtime fingerprint.",
        runtime_inspector=inspector,
    )
    manifest = load_execution_manifest(frozen)
    assert manifest.status == "test_only_non_authoritative"
    assert manifest.execution_callable_authority == NONAUTHORITATIVE_EXECUTION_CALLABLE_AUTHORITY
    assert manifest.runtime == _runtime()
    assert manifest.candidate_file_sha256 == result["candidate_file_sha256"]
    assert manifest.calibrated_design_sha256 == load_study_design(proposed).sha256
    assert manifest.calibration_receipt_file_sha256 == result["calibration_receipt_file_sha256"]
    assert manifest.reviewer_id == "human-reviewer"
    assert finalized["automatic_authorization"] is False


def test_manifest_candidate_rejects_post_calibration_design_change(tmp_path: Path) -> None:
    proposed = _design_path(tmp_path, locked=False)
    receipt = _calibration_receipt(tmp_path, proposed)
    locked_path = _design_path(tmp_path, locked=True)
    locked = load_study_design(locked_path)
    changed = replace(locked, ppo=replace(locked.ppo, n_epochs=2))
    locked_path.write_text(json.dumps(changed.to_dict(), indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ExperimentContractError, match="beyond version/status"):
        emit_manifest_candidate(
            design_path=locked_path,
            calibrated_design_path=proposed,
            calibration_receipt_path=receipt,
            output_path=tmp_path / "candidate.json",
            runtime_inspector=lambda _design: _runtime(),
        )


def test_manifest_candidate_rejects_calibration_action_audit_or_runtime_drift(
    tmp_path: Path,
) -> None:
    proposed = _design_path(tmp_path, locked=False)
    receipt_path = _calibration_receipt(tmp_path, proposed)
    locked = _design_path(tmp_path, locked=True)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["action_likelihood_audit"]["audited_rollouts"] = 0
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(ExperimentContractError, match="action-likelihood audit mismatch"):
        emit_manifest_candidate(
            design_path=locked,
            calibrated_design_path=proposed,
            calibration_receipt_path=receipt_path,
            output_path=tmp_path / "audit-rejected.json",
            runtime_inspector=lambda _design: _runtime(),
        )

    fresh_receipt = _calibration_receipt(tmp_path, proposed, directory="fresh-calibration")
    changed_runtime = replace(_runtime(), torch_version="2.14.1")
    with pytest.raises(ExperimentContractError, match=r"current runtime.*runtime_sha256"):
        emit_manifest_candidate(
            design_path=locked,
            calibrated_design_path=proposed,
            calibration_receipt_path=fresh_receipt,
            output_path=tmp_path / "runtime-rejected.json",
            runtime_inspector=lambda _design: changed_runtime,
        )


@pytest.mark.parametrize(
    "invalid_kind",
    ["duplicate", "nonfinite", "boolean", "oversized"],
)
def test_manifest_finalization_strictly_reads_candidate_json(
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    proposed = _design_path(tmp_path, locked=False)
    receipt = _calibration_receipt(tmp_path, proposed)
    locked = _design_path(tmp_path, locked=True)
    candidate = tmp_path / "candidate.json"
    emit_manifest_candidate(
        design_path=locked,
        calibrated_design_path=proposed,
        calibration_receipt_path=receipt,
        output_path=candidate,
        runtime_inspector=lambda _design: _runtime(),
    )
    if invalid_kind == "duplicate":
        encoded = candidate.read_bytes().rstrip()[:-1] + b', "schema_version": 1}\n'
    elif invalid_kind == "nonfinite":
        encoded = candidate.read_bytes().replace(b'"schema_version": 1', b'"schema_version": NaN')
    elif invalid_kind == "boolean":
        encoded = candidate.read_bytes().replace(b'"schema_version": 1', b'"schema_version": true')
    else:
        encoded = candidate.read_bytes() + b" " * MAX_EXECUTION_ARTIFACT_BYTES
    candidate.write_bytes(encoded)

    with pytest.raises(
        ExperimentContractError,
        match=r"duplicate key|non-finite|schema_version|bounded size limit",
    ):
        finalize_reviewed_manifest(
            design_path=locked,
            calibrated_design_path=proposed,
            calibration_receipt_path=receipt,
            candidate_path=candidate,
            output_path=tmp_path / "manifest.json",
            reviewer_id="human-reviewer",
            review_note="Reviewed exact candidate inputs.",
            runtime_inspector=lambda _design: _runtime(),
        )


def test_manifest_finalization_rehashes_calibration_receipt(tmp_path: Path) -> None:
    proposed = _design_path(tmp_path, locked=False)
    receipt = _calibration_receipt(tmp_path, proposed)
    locked = _design_path(tmp_path, locked=True)
    candidate = tmp_path / "candidate.json"
    emit_manifest_candidate(
        design_path=locked,
        calibrated_design_path=proposed,
        calibration_receipt_path=receipt,
        output_path=candidate,
        runtime_inspector=lambda _design: _runtime(),
    )
    receipt.write_bytes(receipt.read_bytes() + b" ")

    with pytest.raises(ExperimentContractError, match="calibration_receipt_file_sha256"):
        finalize_reviewed_manifest(
            design_path=locked,
            calibrated_design_path=proposed,
            calibration_receipt_path=receipt,
            candidate_path=candidate,
            output_path=tmp_path / "manifest.json",
            reviewer_id="human-reviewer",
            review_note="Reviewed exact candidate inputs.",
            runtime_inspector=lambda _design: _runtime(),
        )


def test_calibration_discards_model_and_cannot_emit_checkpoint(tmp_path: Path) -> None:
    design_path = _design_path(tmp_path, locked=False)
    environment = _FakeVectorEnvironment()
    model = _FakeTrainingModel()

    result = calibrate_resources(
        design_path=design_path,
        calibration_seed=909,
        environment_steps=2,
        runs_dir=tmp_path / "calibration" / "runs",
        runtime_inspector=lambda _design: _runtime(),
        vector_env_factory=lambda _design, seed: environment,
        model_factory=lambda _design, _environment, seed: model,
    )

    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["evidence_purpose"] == "resource_calibration"
    assert receipt["model_disposition"] == "discarded_unserialized"
    assert receipt["checkpoint_emitted"] is False
    assert receipt["eligible_for_behavioral_evaluation"] is False
    assert receipt["action_likelihood_audit"]["audited_rollouts"] == 1
    assert receipt["action_likelihood_audit"]["passed"] is True
    assert model.save_calls == 0
    assert environment.closed is True


def test_calibration_rejects_locked_design_and_fractional_rollout(tmp_path: Path) -> None:
    locked = _design_path(tmp_path, locked=True)
    with pytest.raises(ExperimentContractError, match="proposed_pre_calibration"):
        calibrate_resources(
            design_path=locked,
            calibration_seed=909,
            runs_dir=tmp_path / "runs",
            runtime_inspector=lambda _design: _runtime(),
        )

    proposed = _design_path(tmp_path, locked=False)
    with pytest.raises(ExperimentContractError, match="exact multiple"):
        calibrate_resources(
            design_path=proposed,
            calibration_seed=909,
            environment_steps=1,
            runs_dir=tmp_path / "runs",
            runtime_inspector=lambda _design: _runtime(),
        )


def _train_fake(tmp_path: Path) -> tuple[Path, Path, dict[str, object], _FakeTrainingModel]:
    design_path = _design_path(tmp_path, locked=True)
    manifest_path = _manifest_path(tmp_path, design_path)
    model = _FakeTrainingModel()
    result = train_one_seed(
        design_path=design_path,
        manifest_path=manifest_path,
        train_seed=101,
        runs_dir=tmp_path / "runs",
        runtime_validator=_validator,
        vector_env_factory=lambda _design, seed: _FakeVectorEnvironment(),
        model_factory=lambda _design, _environment, seed: model,
    )
    return design_path, manifest_path, result, model


def test_training_publishes_one_final_checkpoint_and_bound_receipt(tmp_path: Path) -> None:
    design_path, manifest_path, result, model = _train_fake(tmp_path)
    run_directory = Path(result["run_directory"])
    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))

    assert sorted(path.name for path in run_directory.iterdir()) == [
        "checkpoint.zip",
        "checkpoint_receipt.json",
    ]
    assert model.save_calls == 1
    assert receipt["design_file_sha256"]
    assert receipt["execution_manifest_file_sha256"]
    assert receipt["runtime_sha256"] == _runtime().sha256
    assert receipt["observed_timesteps"] == 4
    assert receipt["action_likelihood_audit"]["audited_rollouts"] == 2
    assert receipt["action_likelihood_audit"]["passed"] is True
    assert receipt["automatic_promotion"] is False
    assert receipt["execution_callable_authority"] == NONAUTHORITATIVE_EXECUTION_CALLABLE_AUTHORITY
    assert receipt["checkpoint_eligible_for_declared_evaluation"] is False
    assert load_study_design(design_path).design_status == "locked_pre_behavioral"
    assert load_execution_manifest(manifest_path).status == "test_only_non_authoritative"


def test_injected_training_boundary_cannot_mint_canonical_receipt(tmp_path: Path) -> None:
    design_path = _design_path(tmp_path, locked=True)
    manifest_path = _manifest_path(tmp_path, design_path)
    manifest = load_execution_manifest(manifest_path)
    canonical = replace(
        manifest,
        status="frozen",
        execution_callable_authority=CANONICAL_EXECUTION_CALLABLE_AUTHORITY,
    )
    manifest_path.write_text(
        json.dumps(canonical.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ExperimentContractError, match="callable authority"):
        train_one_seed(
            design_path=design_path,
            manifest_path=manifest_path,
            train_seed=101,
            runs_dir=tmp_path / "runs",
            runtime_validator=_validator,
            vector_env_factory=lambda _design, seed: _FakeVectorEnvironment(),
            model_factory=lambda _design, _environment, seed: _FakeTrainingModel(),
        )
    assert not (tmp_path / "runs").exists()


def test_training_rejects_budget_mismatch_without_publishing_partial_run(
    tmp_path: Path,
) -> None:
    design_path = _design_path(tmp_path, locked=True)
    manifest_path = _manifest_path(tmp_path, design_path)
    runs = tmp_path / "runs"
    with pytest.raises(ExperimentContractError, match="expected exactly"):
        train_one_seed(
            design_path=design_path,
            manifest_path=manifest_path,
            train_seed=101,
            runs_dir=runs,
            runtime_validator=_validator,
            vector_env_factory=lambda _design, seed: _FakeVectorEnvironment(),
            model_factory=lambda _design, _environment, seed: _FakeTrainingModel(short_by=1),
        )
    assert list(runs.iterdir()) == []


class _FakeEvaluationEnvironment:
    def __init__(self, design, *, end_early: bool = False) -> None:
        self.reference_content_sha256 = make_static_stand_reference(
            n_frames=design.static_reference_frames
        ).identity.content_sha256
        self.max_steps = design.max_episode_steps
        self.end_early = end_early
        self.step_count = 0
        self.reset_seeds: list[int] = []
        self.closed = False
        self.env = object()

    def reset(self, *, seed: int):
        self.reset_seeds.append(seed)
        self.step_count = 0
        return np.zeros(1), {"reference_tracking": {"tracking_error": "untrusted"}}

    def step(self, _action):
        self.step_count += 1
        ended = self.step_count == (1 if self.end_early else self.max_steps)
        return (
            np.zeros(1),
            1.0,
            False,
            ended,
            {"reference_tracking": {"tracking_error": "deliberately-wrong"}},
        )

    def close(self) -> None:
        self.closed = True


class _FakeEvaluationModel:
    def __init__(self, timesteps: int) -> None:
        self.num_timesteps = timesteps
        self.deterministic_flags: list[bool] = []

    def predict(self, _observation, *, deterministic: bool):
        self.deterministic_flags.append(deterministic)
        return np.zeros(17), None


def _exact_stand_state(_environment, _abi) -> HumanoidTrackingState:
    return HumanoidTrackingState(
        root_position_world_m=np.array([0.0, 0.0, 1.4]),
        root_height_m=1.4,
        root_orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        root_linear_velocity_world_m_s=np.zeros(3),
        root_angular_velocity_body_rad_s=np.zeros(3),
        joint_positions_rad=np.zeros(17),
        joint_velocities_rad_s=np.zeros(17),
    )


def _zero_history(_environment, _abi) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(17), np.zeros(17)


def _zero_protected_inputs(
    _environment,
    _abi,
    *,
    normalized_action,
    previous_normalized_action,
    previous_joint_acceleration,
):
    return {
        "normalized_policy_action": np.asarray(normalized_action),
        "previous_normalized_policy_action": np.asarray(previous_normalized_action),
        "generalized_actuator_torque_n_m": np.zeros(17),
        "generalized_actuator_torque_capacity_n_m": np.asarray(
            [40, 40, 40, 40, 40, 120, 80, 40, 40, 120, 80, 10, 10, 10, 10, 10, 10]
        ),
        "joint_accelerations_rad_s2": np.zeros(17),
        "previous_joint_accelerations_rad_s2": np.asarray(previous_joint_acceleration),
        "contact_facts": (),
        "control_period_seconds": 0.015,
    }


def test_evaluation_uses_direct_state_and_completes_every_declared_seed(
    tmp_path: Path,
) -> None:
    design_path, manifest_path, training, _ = _train_fake(tmp_path)
    design = load_study_design(design_path)
    environment = _FakeEvaluationEnvironment(design)
    model = _FakeEvaluationModel(design.ppo.total_timesteps_per_seed)
    state_calls = 0

    def station_state(_environment, _abi) -> HumanoidTrackingState:
        nonlocal state_calls
        episode_call = state_calls % (design.max_episode_steps + 1)
        state_calls += 1
        if episode_call == 0:
            root_position = np.array([10.0, -5.0, 1.4])
            root_velocity = np.zeros(3)
        else:
            root_position = np.array([10.6, -4.2, 1.4])
            root_velocity = np.array([-0.7, 0.3, -0.4])
        return HumanoidTrackingState(
            root_position_world_m=root_position,
            root_height_m=1.4,
            root_orientation_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            root_linear_velocity_world_m_s=root_velocity,
            root_angular_velocity_body_rad_s=np.zeros(3),
            joint_positions_rad=np.zeros(17),
            joint_velocities_rad_s=np.zeros(17),
        )

    result = evaluate_checkpoint(
        design_path=design_path,
        manifest_path=manifest_path,
        checkpoint_receipt_path=Path(training["receipt_path"]),
        runs_dir=tmp_path / "evaluation" / "runs",
        runtime_validator=_validator,
        environment_factory=lambda _design: environment,
        model_loader=lambda _path, _environment, device: model,
        abi_factory=lambda _environment: object(),
        state_reader=station_state,
        history_reader=_zero_history,
        protected_step_reader=_zero_protected_inputs,
    )

    receipt = json.loads(Path(result["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["evaluation_seeds_observed"] == [11001, 11002]
    assert len(receipt["episodes"]) == 2
    assert receipt["station_keeping_origin_source"] == STATION_KEEPING_ORIGIN_SOURCE
    assert all(
        episode["station_keeping_origin_source"] == STATION_KEEPING_ORIGIN_SOURCE
        for episode in receipt["episodes"]
    )
    assert all(
        episode["initial_root_position_world_m"] == [10.0, -5.0, 1.4]
        for episode in receipt["episodes"]
    )
    assert all(
        episode["root_horizontal_displacement_max_m"] == pytest.approx(1.0)
        for episode in receipt["episodes"]
    )
    assert all(
        episode["root_linear_velocity_x_max_abs_m_s"] == pytest.approx(0.7)
        and episode["root_linear_velocity_y_max_abs_m_s"] == pytest.approx(0.3)
        and episode["root_linear_velocity_z_max_abs_m_s"] == pytest.approx(0.4)
        and episode["control_period_seconds"] == pytest.approx(0.015)
        for episode in receipt["episodes"]
    )
    assert receipt["reward_telemetry_used_for_objective_metrics"] is False
    assert receipt["execution_callable_authority"] == NONAUTHORITATIVE_EXECUTION_CALLABLE_AUTHORITY
    assert receipt["automatic_promotion"] is False
    assert environment.reset_seeds == [11001, 11002]
    assert all(model.deterministic_flags)
    assert environment.closed is True


def test_evaluation_rejects_early_episode_and_publishes_nothing(tmp_path: Path) -> None:
    design_path, manifest_path, training, _ = _train_fake(tmp_path)
    design = load_study_design(design_path)
    output_runs = tmp_path / "early" / "runs"
    with pytest.raises(ExperimentContractError, match="ended after 1"):
        evaluate_checkpoint(
            design_path=design_path,
            manifest_path=manifest_path,
            checkpoint_receipt_path=Path(training["receipt_path"]),
            runs_dir=output_runs,
            runtime_validator=_validator,
            environment_factory=lambda _design: _FakeEvaluationEnvironment(
                design,
                end_early=True,
            ),
            model_loader=lambda _path, _environment, device: _FakeEvaluationModel(
                design.ppo.total_timesteps_per_seed
            ),
            abi_factory=lambda _environment: object(),
            state_reader=_exact_stand_state,
            history_reader=_zero_history,
            protected_step_reader=_zero_protected_inputs,
        )
    assert list(output_runs.iterdir()) == []


def test_evaluation_rehashes_checkpoint_before_loading(tmp_path: Path) -> None:
    design_path, manifest_path, training, _ = _train_fake(tmp_path)
    Path(training["checkpoint_path"]).write_bytes(b"tampered")
    loader_called = False

    def loader(_path, _environment, *, device):
        nonlocal loader_called
        loader_called = True
        raise AssertionError("must not load a changed checkpoint")

    with pytest.raises(ExperimentContractError, match="SHA-256"):
        evaluate_checkpoint(
            design_path=design_path,
            manifest_path=manifest_path,
            checkpoint_receipt_path=Path(training["receipt_path"]),
            runs_dir=tmp_path / "tamper" / "runs",
            runtime_validator=_validator,
            model_loader=loader,
        )
    assert loader_called is False


def test_evaluation_rejects_incomplete_action_likelihood_audit(tmp_path: Path) -> None:
    design_path, manifest_path, training, _ = _train_fake(tmp_path)
    receipt_path = Path(training["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["action_likelihood_audit"]["audited_rollouts"] -= 1
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    loader_called = False

    def loader(_path, _environment, *, device):
        nonlocal loader_called
        loader_called = True
        raise AssertionError("must not load a checkpoint whose rollout audit is incomplete")

    with pytest.raises(ExperimentContractError, match="action-likelihood audit mismatch"):
        evaluate_checkpoint(
            design_path=design_path,
            manifest_path=manifest_path,
            checkpoint_receipt_path=receipt_path,
            runs_dir=tmp_path / "audit-tamper" / "runs",
            runtime_validator=_validator,
            model_loader=loader,
        )
    assert loader_called is False


@pytest.mark.parametrize("invalid_kind", ["duplicate", "nonfinite", "boolean", "oversized"])
def test_evaluation_strictly_reads_checkpoint_receipt(
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    design_path, manifest_path, training, _ = _train_fake(tmp_path)
    receipt_path = Path(training["receipt_path"])
    encoded = receipt_path.read_bytes()
    if invalid_kind == "duplicate":
        encoded = encoded.rstrip()[:-1] + b', "schema_version": 1}\n'
    elif invalid_kind == "nonfinite":
        encoded = encoded.replace(b'"observed_timesteps": 4', b'"observed_timesteps": Infinity')
    elif invalid_kind == "boolean":
        encoded = encoded.replace(b'"schema_version": 1', b'"schema_version": true')
    else:
        encoded += b" " * MAX_EXECUTION_ARTIFACT_BYTES
    receipt_path.write_bytes(encoded)
    loader_called = False

    def loader(_path, _environment, *, device):
        nonlocal loader_called
        loader_called = True
        raise AssertionError("must not load from a non-strict checkpoint receipt")

    with pytest.raises(
        ExperimentContractError,
        match=r"duplicate key|non-finite|schema_version|bounded size limit",
    ):
        evaluate_checkpoint(
            design_path=design_path,
            manifest_path=manifest_path,
            checkpoint_receipt_path=receipt_path,
            runs_dir=tmp_path / "strict-receipt" / "runs",
            runtime_validator=_validator,
            model_loader=loader,
        )
    assert loader_called is False


@pytest.mark.parametrize(
    ("field", "alias"),
    [
        ("requested_timesteps", 4.0),
        ("observed_timesteps", 4.0),
        ("checkpoint_size_bytes", 15.0),
        ("checkpoint_eligible_for_declared_evaluation", 1),
        ("automatic_promotion", 0),
    ],
)
def test_evaluation_rejects_checkpoint_receipt_bool_and_numeric_aliases(
    tmp_path: Path,
    field: str,
    alias: object,
) -> None:
    design_path, manifest_path, training, _ = _train_fake(tmp_path)
    receipt_path = Path(training["receipt_path"])
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if field == "checkpoint_size_bytes":
        alias = float(payload[field])
    payload[field] = alias
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    loader_called = False

    def loader(_path, _environment, *, device):
        nonlocal loader_called
        loader_called = True
        raise AssertionError("must not load from a type-ambiguous checkpoint receipt")

    with pytest.raises(ExperimentContractError, match=field):
        evaluate_checkpoint(
            design_path=design_path,
            manifest_path=manifest_path,
            checkpoint_receipt_path=receipt_path,
            runs_dir=tmp_path / "typed-receipt" / "runs",
            runtime_validator=_validator,
            model_loader=loader,
        )
    assert loader_called is False


@pytest.mark.gym
def test_real_sb3_short_construct_save_and_load_path(tmp_path: Path) -> None:
    """Software smoke only; four steps cannot support a behavioral claim."""

    proposed = load_study_design(PROPOSED_PATH)
    design = replace(
        proposed,
        ppo=replace(
            proposed.ppo,
            total_timesteps_per_seed=4,
            n_envs=1,
            n_steps=2,
            batch_size=2,
            n_epochs=1,
            policy_hidden_layers=(8, 8),
        ),
    )
    environment = _make_vector_env(design, seed=909)
    try:
        model = _make_ppo(design, environment, seed=909)
        model.learn(total_timesteps=4, reset_num_timesteps=True, progress_bar=False)
        checkpoint = tmp_path / "short-smoke.zip"
        model.save(checkpoint)
        loaded = _load_ppo(checkpoint, environment, device="cpu")
        assert model.num_timesteps == loaded.num_timesteps == 4
        action, _ = loaded.predict(environment.reset(), deterministic=True)
        assert action.shape == (1, 17)
    finally:
        environment.close()


@pytest.mark.gym
def test_real_humanoid_rollout_stores_the_action_sent_through_the_affine_map() -> None:
    """Four-step execution invariant only; this is not behavioral evidence."""

    import gymnasium as gym
    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv

    proposed = load_study_design(PROPOSED_PATH)
    design = replace(
        proposed,
        ppo=replace(
            proposed.ppo,
            total_timesteps_per_seed=4,
            n_envs=1,
            n_steps=4,
            batch_size=4,
            n_epochs=1,
            policy_hidden_layers=(8, 8),
        ),
    )

    class RecordingActionWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.normalized_actions: list[np.ndarray] = []
            self.physical_controls: list[np.ndarray] = []

        def step(self, action):
            self.normalized_actions.append(np.asarray(action, dtype=np.float64).copy())
            transition = self.env.step(action)
            self.physical_controls.append(
                np.asarray(self.env.unwrapped.data.ctrl, dtype=np.float64).copy()
            )
            return transition

    recorder = RecordingActionWrapper(make_static_tracking_env(design))
    environment = DummyVecEnv([lambda: recorder])
    try:
        model = _make_ppo(design, environment, seed=909)
        # Preserve the policy that generated the rollout so its stored
        # likelihood can be recomputed exactly before any optimizer update.
        model.train = lambda: None
        model.learn(total_timesteps=4, reset_num_timesteps=True, progress_bar=False)

        stored_actions = np.asarray(model.rollout_buffer.actions[:, 0], dtype=np.float64)
        observed_actions = np.asarray(recorder.normalized_actions, dtype=np.float64)
        np.testing.assert_allclose(stored_actions, observed_actions, rtol=0.0, atol=0.0)
        assert np.all(stored_actions > -1.0)
        assert np.all(stored_actions < 1.0)

        inner_tracker = recorder.env.env
        physical_low = np.asarray(inner_tracker.action_space.low, dtype=np.float64)
        physical_high = np.asarray(inner_tracker.action_space.high, dtype=np.float64)
        expected_controls = physical_low + 0.5 * (stored_actions + 1.0) * (
            physical_high - physical_low
        )
        np.testing.assert_allclose(
            np.asarray(recorder.physical_controls),
            expected_controls,
            rtol=0.0,
            atol=1e-7,
        )

        observations = torch.as_tensor(
            model.rollout_buffer.observations.reshape((-1, *environment.observation_space.shape)),
            device=model.device,
        )
        actions = torch.as_tensor(
            model.rollout_buffer.actions.reshape((-1, *environment.action_space.shape)),
            device=model.device,
        )
        with torch.no_grad():
            _values, recomputed_log_probabilities, _entropy = model.policy.evaluate_actions(
                observations,
                actions,
            )
        np.testing.assert_allclose(
            recomputed_log_probabilities.cpu().numpy(),
            model.rollout_buffer.log_probs.reshape(-1),
            rtol=0.0,
            atol=1e-6,
        )
    finally:
        environment.close()


@pytest.mark.gym
def test_default_evaluation_factory_resolves_reference_and_physical_abi() -> None:
    design = load_study_design(PROPOSED_PATH)
    reference = make_static_stand_reference(n_frames=design.static_reference_frames)
    environment = make_static_tracking_env(design)
    try:
        abi = _validated_evaluation_environment(
            environment,
            expected_reference_sha256=reference.identity.content_sha256,
            abi_factory=validate_humanoid_actuator_abi,
        )
        assert abi.n_actuators == 17
    finally:
        environment.close()


@pytest.mark.gym
def test_real_protected_reader_uses_post_transmission_torque_and_all_contacts() -> None:
    design = load_study_design(PROPOSED_PATH)
    environment = make_static_tracking_env(design)
    try:
        environment.reset(seed=20260902)
        abi = validate_humanoid_actuator_abi(environment.unwrapped)
        previous_action, previous_acceleration = _initial_protected_history(environment, abi)
        normalized_action = np.ones(17, dtype=environment.action_space.dtype)
        environment.step(normalized_action)
        inputs = _protected_step_inputs(
            environment,
            abi,
            normalized_action=normalized_action,
            previous_normalized_action=previous_action,
            previous_joint_acceleration=previous_acceleration,
        )

        torque = np.asarray(inputs["generalized_actuator_torque_n_m"])
        capacity = np.asarray(inputs["generalized_actuator_torque_capacity_n_m"])
        assert torque[0] == pytest.approx(40.0)
        assert torque[5] == pytest.approx(120.0)
        assert torque[11] == pytest.approx(10.0)
        np.testing.assert_allclose(
            capacity,
            [40, 40, 40, 40, 40, 120, 80, 40, 40, 120, 80, 10, 10, 10, 10, 10, 10],
        )
        assert len(inputs["contact_facts"]) == len(
            environment.unwrapped.last_control_step_contact_samples
        )
        assert environment.unwrapped.last_control_step_substeps == 5
        assert np.asarray(inputs["joint_accelerations_rad_s2"]).shape == (17,)
        assert inputs["control_period_seconds"] == pytest.approx(0.015)
    finally:
        environment.close()
