from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.harness.cycle_cli import (
    CycleCliError,
    TrainingCliDependencies,
    main,
)
from oracle_composition.phase_b.evaluation import UtilityEvaluationDependencies
from oracle_composition.phase_b.persistence import (
    publish_checkpoint_index,
    publish_final_persistence,
)
from oracle_composition.phase_b.report_v2 import (
    ProtectedEpisodeMetrics,
    SeedReportFacts,
    exact_binomial_interval,
    task_success_endpoint,
    utility_gate,
)
from oracle_composition.phase_b.runtime import fake_environment_factories, fake_policy_factory
from oracle_composition.phase_b.supervision import ResourceLimits
from oracle_composition.phase_b.training import (
    COHORT_SEEDS,
    PPORecipe,
    TrainingPlan,
    run_ppo_training,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = Path("experiments/003_composition_speed_profile")
PHASE_B = EXPERIMENT / "phase_b"
ORACLE = PHASE_B / "oracle_cycle_1_reference_v1.json"
REWARD = PHASE_B / "tracking_only_v1.json"
ZERO_SPEEDS = (0.0,) * 1_000
ZERO_ERRORS = {
    "joint_position_rmse_rad": 0.0,
    "joint_velocity_rmse_rad_s": 0.0,
    "root_angular_velocity_rmse_rad_s": 0.0,
    "root_height_abs_error_m": 0.0,
    "root_linear_velocity_rmse_m_s": 0.0,
    "root_orientation_error_rad": 0.0,
}


def _limits() -> ResourceLimits:
    return ResourceLimits(
        per_seed_wall_seconds=20.0,
        cohort_wall_seconds=30.0,
        job_wall_seconds=30.0,
        rss_bytes=8 * 1024**3,
        free_disk_bytes=1,
        output_bytes=64 * 1024**2,
        throughput_floor_steps_s=1.0,
        throughput_warmup_transitions=65_536,
        throughput_window_transitions=65_536,
    )


def _dependencies(
    *, utility: UtilityEvaluationDependencies | None = None
) -> TrainingCliDependencies:
    return TrainingCliDependencies(
        runtime_kind="fake",
        test_only=True,
        allow_dirty=True,
        test_steps_per_environment=4,
        test_batch_size=16,
        test_n_epochs=1,
        resource_limits=_limits(),
        utility_dependencies=utility,
    )


def _train_args(output: Path) -> list[str]:
    return [
        "train",
        "--experiment",
        str(EXPERIMENT),
        "--cycle",
        "3",
        "--oracle",
        str(ORACLE),
        "--reward",
        str(REWARD),
        "--output",
        str(output),
        "--seeds",
        "11,13",
        "--transitions",
        "16",
    ]


def _episode(
    *,
    policy_seed: int,
    evaluation_seed: int,
    cell: str,
    checkpoint_sha256: str = "0" * 64,
    passed: bool = True,
    task_success: bool | None = None,
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
    errors = dict(ZERO_ERRORS)
    if not passed:
        errors["joint_position_rmse_rad"] = 0.36
    return ProtectedEpisodeMetrics(
        policy_seed=policy_seed,
        evaluation_seed=evaluation_seed,
        cell=cell,
        checkpoint_sha256=checkpoint_sha256,
        observed_steps=1_000,
        root_delta_forward_speed_m_s=ZERO_SPEEDS,
        com_forward_speed_m_s=ZERO_SPEEDS,
        six_tracking_errors=errors,
        fall=False,
        forbidden_contacts=(),
        action_bounds_ok=True,
        switch_records=switches,
        resynchronization_records=resynchronization,
        segment_errors={"fast": 0.1, "return_fast": 0.1, "slow": 0.1},
        transition_window_error=0.1 if cell == "fixed_round_trip" else None,
        settle_latency_steps=7 if cell == "fixed_round_trip" else None,
        time_to_first_failure_steps=None,
        task_success=task_success,
    )


def _fake_episode_runner(
    _policy: object,
    policy_seed: int,
    evaluation_seed: int,
    cell: str,
    _corpus_root: Path,
) -> ProtectedEpisodeMetrics:
    return _episode(
        policy_seed=policy_seed,
        evaluation_seed=evaluation_seed,
        cell=cell,
    )


def test_cli_fake_runtime_end_to_end_writes_deterministic_report_v2(tmp_path: Path) -> None:
    first = (tmp_path / "first").resolve()
    second = (tmp_path / "second").resolve()
    assert (
        main(
            _train_args(first),
            repository_root=ROOT,
            training_dependencies=_dependencies(),
        )
        == 0
    )
    assert (
        main(
            _train_args(second),
            repository_root=ROOT,
            training_dependencies=_dependencies(),
        )
        == 0
    )

    assert (first / "scientific_receipt_v2.json").read_bytes() == (
        second / "scientific_receipt_v2.json"
    ).read_bytes()
    report = json.loads((first / "scientific_receipt_v2.json").read_bytes())
    assert report["evidence_class"] == "interface_check"
    assert report["training"]["observed_transitions"] == 32
    assert report["training"]["stream_counts"] == {
        "composition": 16,
        "rehearsal": 16,
    }
    assert report["training"]["wall_time_seconds"] is None
    assert len(report["policy"]["checkpoints"]) == 2
    trace_index = json.loads((first / "trace_index_v2.json").read_bytes())
    assert trace_index["entry_count"] == 14
    assert not any("telemetry" in entry["role"] for entry in trace_index["entries"])
    assert (first / "telemetry_v1.json").read_bytes() != (second / "telemetry_v1.json").read_bytes()

    with pytest.raises(ExperimentContractError, match="fresh and no-overwrite"):
        main(
            _train_args(first),
            repository_root=ROOT,
            training_dependencies=_dependencies(),
        )


def test_smoke_mode_refuses_promotion_before_preflight(tmp_path: Path) -> None:
    args = _train_args((tmp_path / "never-created").resolve())
    args[args.index("11,13")] = "121901"
    args[args.index("16")] = "196608"
    args.extend(("--smoke", "--promote"))
    with pytest.raises(CycleCliError, match="refuses promotion"):
        main(args, repository_root=ROOT, training_dependencies=_dependencies())


def test_five_entry_checkpoint_index_binds_each_seed_artifact(tmp_path: Path) -> None:
    output = tmp_path.resolve()
    entries = []
    for seed in COHORT_SEEDS:
        plan = TrainingPlan(
            seed=seed,
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
        seed_directory = output / f"seed_{seed}"
        seed_directory.mkdir()
        entries.append(
            publish_final_persistence(
                output_directory=seed_directory,
                result=result,
                plan=plan,
            )
        )
        assert entries[-1].receipt_value["checkpoint_reload_bitwise_deterministic"] is True
        assert entries[-1].receipt_value["checkpoint_to_export_bitwise_equivalent"] is True
    index = publish_checkpoint_index(output_directory=output, entries=entries)
    value = json.loads(index.path.read_bytes())
    assert value["checkpoint_count"] == 5
    assert [row["ppo_seed"] for row in value["entries"]] == list(COHORT_SEEDS)
    assert all(row["checkpoint"]["path"].startswith("seed_") for row in value["entries"])


def test_cli_evaluate_policy_runs_hold_and_transition_cells_with_step_zero(
    tmp_path: Path,
) -> None:
    trained = (tmp_path / "trained").resolve()
    assert (
        main(
            _train_args(trained),
            repository_root=ROOT,
            training_dependencies=_dependencies(),
        )
        == 0
    )
    checkpoint = trained / "seed_11/checkpoint_seed_11_final.npz"
    output = (tmp_path / "evaluation").resolve()
    utility_dependencies = UtilityEvaluationDependencies(episode_runner=_fake_episode_runner)
    args = [
        "evaluate-policy",
        "--experiment",
        str(EXPERIMENT),
        "--cycle",
        "3",
        "--oracle",
        str(ORACLE),
        "--reward",
        str(REWARD),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
    ]
    assert (
        main(
            args,
            repository_root=ROOT,
            training_dependencies=_dependencies(utility=utility_dependencies),
        )
        == 0
    )
    report = json.loads((output / "scientific_receipt_v2.json").read_bytes())
    checkpoint_gate = report["summary"]["hard_gates"]["utility_gate"]["checkpoint_results"][0]
    assert all(cell["cell_passed"] for cell in checkpoint_gate["cells"].values())
    assert report["summary"]["hard_gates"]["utility_gate"]["family_passed"] is False
    assert len(json.loads((output / "trained_policy_metrics_v1.json").read_bytes())) == 80
    assert len(json.loads((output / "step_zero_metrics_v1.json").read_bytes())) == 80


def _seed_facts(seed: int) -> SeedReportFacts:
    return SeedReportFacts(
        ppo_seed=seed,
        checkpoint_sha256=f"{seed:064x}",
        strict_export_sha256=f"{seed + 1:064x}",
        training={
            "evidence_class": "exploratory_fine_tuning_cycle",
            "execution_manifest_sha256": "a" * 64,
            "losses": [],
            "observed_transitions": 16,
            "optimizer_updates": 1,
            "planned_transitions": 16,
            "ppo_seed": seed,
            "reward_totals": {
                "ignored_stock_reward": 0.0,
                "r_task": 0.0,
                "r_track": 0.0,
                "r_train": 0.0,
            },
            "rollouts": 1,
            "rsi_ledger_sha256": "b" * 64,
            "stream_counts": {"composition": 8, "rehearsal": 8},
            "unfreeze_receipt_sha256": "c" * 64,
        },
        step_zero_comparator={"bitwise_equal": True},
    )


def test_utility_cell_and_family_rules_use_fixed_denominators() -> None:
    checkpoints = [_seed_facts(seed) for seed in COHORT_SEEDS]
    episodes = []
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        for cell in ("hold_expert", "hold_medium", "hold_simple", "fixed_round_trip"):
            for evaluation_index, evaluation_seed in enumerate(range(120101, 120121)):
                episodes.append(
                    _episode(
                        policy_seed=checkpoint.ppo_seed,
                        evaluation_seed=evaluation_seed,
                        cell=cell,
                        checkpoint_sha256=checkpoint.checkpoint_sha256,
                        passed=not (
                            checkpoint_index == 4
                            and cell == "hold_expert"
                            and evaluation_index >= 15
                        ),
                    )
                )
    gate = utility_gate(checkpoints=checkpoints, episodes=episodes)
    assert gate["passed_checkpoint_count"] == 4
    assert gate["family_passed"] is True
    assert gate["checkpoint_results"][-1]["cells"]["hold_expert"] == {
        "cell_passed": False,
        "episode_count": 20,
        "episode_pass_count": 15,
        "fixed_denominator_without_replacement": True,
        "minimum_pass_count": 16,
    }


def test_task_success_endpoint_uses_exact_interval_and_uncalibrated_placeholder() -> None:
    episodes = [
        _episode(
            policy_seed=11,
            evaluation_seed=120101 + index,
            cell="fixed_round_trip",
            task_success=index < 16,
        )
        for index in range(20)
    ]
    endpoint = task_success_endpoint(episodes)
    assert endpoint["primary_endpoint"]["successes"] == 16
    assert endpoint["primary_endpoint"]["exact_binomial_95_interval"] == list(
        exact_binomial_interval(16, 20)
    )
    assert endpoint["segment_tolerances"]["calibration_status"] == ("placeholder_until_calibrated")
    assert endpoint["unsafe_arm_may_outrank_safe_arm"] is False


def test_resynchronization_limit_accepts_64_and_rejects_65() -> None:
    episode = _episode(
        policy_seed=11,
        evaluation_seed=120101,
        cell="fixed_round_trip",
    )
    at_limit = replace(
        episode,
        resynchronization_records=tuple(
            {**record, "settle_latency_steps": 64} for record in episode.resynchronization_records
        ),
    )
    over_limit = replace(
        episode,
        resynchronization_records=tuple(
            {**record, "settle_latency_steps": 65} for record in episode.resynchronization_records
        ),
    )
    assert at_limit.utility_passed is True
    assert over_limit.utility_passed is False
    wrong_boundary = replace(
        episode,
        switch_records=(
            {**episode.switch_records[0], "boundary": 299},
            episode.switch_records[1],
        ),
    )
    assert wrong_boundary.utility_passed is False
    hold_with_hidden_switch = replace(
        _episode(
            policy_seed=11,
            evaluation_seed=120101,
            cell="hold_expert",
        ),
        switch_records=({"boundary": 300, "from_behavior": "expert", "to_behavior": "medium"},),
    )
    assert hold_with_hidden_switch.utility_passed is False
