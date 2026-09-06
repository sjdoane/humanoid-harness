from __future__ import annotations

import json
from pathlib import Path

import pytest

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.experiments.artifact_io import publish_bytes_without_overwrite
from oracle_composition.phase_b.evaluation import UtilityEvaluationDependencies
from oracle_composition.phase_b.evaluation_lineage import evaluator_source_identity
from oracle_composition.phase_b.evaluation_supervision import (
    EvaluationStatus,
    EvaluationWorkerRequest,
    supervise_policy_evaluation,
)
from oracle_composition.phase_b.persistence import publish_final_persistence
from oracle_composition.phase_b.report_v2 import ProtectedEpisodeMetrics
from oracle_composition.phase_b.runtime import fake_environment_factories, fake_policy_factory
from oracle_composition.phase_b.training import PPORecipe, TrainingPlan, run_ppo_training

ROOT = Path(__file__).resolve().parents[2]
ZERO_SPEEDS = (0.0,) * 1_000
ZERO_ERRORS = {
    "joint_position_rmse_rad": 0.0,
    "joint_velocity_rmse_rad_s": 0.0,
    "root_angular_velocity_rmse_rad_s": 0.0,
    "root_height_abs_error_m": 0.0,
    "root_linear_velocity_rmse_m_s": 0.0,
    "root_orientation_error_rad": 0.0,
}


def episode_runner(
    _policy: object,
    policy_seed: int,
    evaluation_seed: int,
    cell: str,
    _corpus_root: Path,
) -> ProtectedEpisodeMetrics:
    switches = (
        (
            {"boundary": 300, "from_behavior": "expert", "to_behavior": "medium"},
            {"boundary": 600, "from_behavior": "medium", "to_behavior": "expert"},
        )
        if cell == "fixed_round_trip"
        else ()
    )
    resynchronization = (
        (
            {
                "eight_consecutive_boundaries_at_or_below_one": True,
                "from_behavior": "expert",
                "settle_latency_steps": 7,
                "to_behavior": "medium",
            },
            {
                "eight_consecutive_boundaries_at_or_below_one": True,
                "from_behavior": "medium",
                "settle_latency_steps": 7,
                "to_behavior": "expert",
            },
        )
        if cell == "fixed_round_trip"
        else ()
    )
    return ProtectedEpisodeMetrics(
        policy_seed=policy_seed,
        evaluation_seed=evaluation_seed,
        cell=cell,
        checkpoint_sha256="0" * 64,
        observed_steps=1_000,
        root_delta_forward_speed_m_s=ZERO_SPEEDS,
        com_forward_speed_m_s=ZERO_SPEEDS,
        six_tracking_errors=ZERO_ERRORS,
        fall=False,
        forbidden_contacts=(),
        action_bounds_ok=True,
        switch_records=switches,
        resynchronization_records=resynchronization,
        segment_errors={"fast": 0.0, "return_fast": 0.0, "slow": 0.0},
        transition_window_error=0.0 if cell == "fixed_round_trip" else None,
        settle_latency_steps=7 if cell == "fixed_round_trip" else None,
        time_to_first_failure_steps=None,
        task_success=True,
        settled_state_normalized_error=0.0 if cell == "fixed_round_trip" else None,
    )


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    output = tmp_path_factory.mktemp("evaluation-checkpoint")
    plan = TrainingPlan(
        seed=11,
        transitions=16,
        manifest_sha256="d" * 64,
        evidence_class="interface_check",
        promotable=False,
        smoke=False,
        steps_per_environment=4,
        recipe=PPORecipe(batch_size=16, n_epochs=1),
        test_only=True,
    )
    result = run_ppo_training(
        plan=plan,
        policy_factory=fake_policy_factory,
        environment_factories=fake_environment_factories(plan=plan),
    )
    persistence = publish_final_persistence(output_directory=output, result=result, plan=plan)
    return persistence.checkpoint.path, persistence.checkpoint.sha256


def _run(
    tmp_path: Path,
    checkpoint: tuple[Path, str],
    dependencies: UtilityEvaluationDependencies,
):
    output = tmp_path.resolve()
    output.mkdir()
    manifest = publish_bytes_without_overwrite(
        output / "evaluation_manifest_v1.json",
        canonical_json_bytes({"schema_version": 1}),
    )
    source = evaluator_source_identity(ROOT)
    request = EvaluationWorkerRequest(
        checkpoint_path=str(checkpoint[0]),
        checkpoint_sha256=checkpoint[1],
        step_zero_actor_path=str(
            ROOT / "artifacts/experiments_003/phase_b/step_0_full_authority_actor_v1.npz"
        ),
        step_zero_actor_sha256="6ebc2b56be9a5f304b8b584157fd0141d449d75297366213e4976291cb2dcfe0",
        corpus_root=str(ROOT / "artifacts/reference_corpus_v2"),
        calibration_receipt_path=None,
        calibration_receipt_sha256=None,
        segment_targets_m_s=(3.0, 1.25, 3.0),
        dependencies=dependencies,
        output_directory=str(output),
        evaluator_source_sha256=source["sha256"],
        repository_root=str(ROOT),
    )
    return supervise_policy_evaluation(request=request, evaluation_manifest=manifest)


def test_evaluation_supervisor_accounts_for_all_planned_episodes(
    tmp_path: Path, checkpoint: tuple[Path, str]
) -> None:
    result = _run(
        tmp_path / "success",
        checkpoint,
        UtilityEvaluationDependencies(episode_runner=episode_runner, wall_seconds=30.0),
    )
    assert result.status is EvaluationStatus.SUCCEEDED
    assert result.terminal_value["completed_episodes"] == 160
    assert result.terminal_value["uncompleted_episodes"] == 0
    assert len(result.trained_episodes) == len(result.step_zero_episodes) == 80
    assert all(item.task_success is None for item in result.trained_episodes)


@pytest.mark.parametrize(
    ("dependencies", "status"),
    [
        (UtilityEvaluationDependencies(failure_mode="hang", wall_seconds=0.2), "timeout"),
        (UtilityEvaluationDependencies(failure_mode="crash", wall_seconds=10.0), "crash"),
        (UtilityEvaluationDependencies(failure_mode="non_finite", wall_seconds=10.0), "non_finite"),
        (
            UtilityEvaluationDependencies(
                episode_runner=episode_runner,
                fail_after_episodes=1,
                wall_seconds=10.0,
            ),
            "partial",
        ),
    ],
)
def test_evaluation_supervisor_writes_one_accounting_failure_receipt(
    tmp_path: Path,
    checkpoint: tuple[Path, str],
    dependencies: UtilityEvaluationDependencies,
    status: str,
) -> None:
    output = tmp_path / status
    result = _run(output, checkpoint, dependencies)
    assert result.status.value == status
    receipt = json.loads(result.terminal_receipt.path.read_bytes())
    assert receipt["planned_episodes"] == 160
    assert receipt["completed_episodes"] + receipt["uncompleted_episodes"] == 160
    assert len(list(output.glob("evaluation_*_receipt_v1.json"))) == 1
