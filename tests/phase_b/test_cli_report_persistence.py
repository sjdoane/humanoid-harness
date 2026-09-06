from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.experiments.artifact_io import publish_bytes_without_overwrite
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.harness.cycle_cli import (
    CycleCliError,
    TrainingCliDependencies,
    main,
)
from oracle_composition.phase_b.calibration import (
    CALIBRATION_BLOCKS,
    CALIBRATION_POLICY_SEEDS,
    TaskSuccessCalibration,
    publish_calibration_receipt,
)
from oracle_composition.phase_b.evaluation import UtilityEvaluationDependencies, _score_task_success
from oracle_composition.phase_b.persistence import (
    publish_checkpoint_index,
    publish_final_persistence,
)
from oracle_composition.phase_b.report_v2 import (
    ProtectedEpisodeMetrics,
    SeedReportFacts,
    _derived_task_success,
    exact_binomial_interval,
    task_success_endpoint,
    utility_gate,
    validate_report_summary_recomputation,
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
        settled_state_normalized_error=0.1 if cell == "fixed_round_trip" else None,
    )


def _calibration(path: Path) -> tuple[Path, str, TaskSuccessCalibration]:
    samples = [
        {
            "block_id": block,
            "fall": False,
            "first_transition_latency_steps": 7,
            "policy_seed_id": seed,
            "second_transition_latency_steps": 8,
            "segment_speed_errors_m_s": {
                "fast": 0.2,
                "return_fast": 0.2,
                "slow": 0.2,
            },
            "settled_state_normalized_error": 0.2,
            "source_checkpoint_sha256": f"{seed:064x}",
        }
        for seed in CALIBRATION_POLICY_SEEDS
        for block in CALIBRATION_BLOCKS
    ]
    artifact = publish_calibration_receipt(path, samples)
    from oracle_composition.phase_b.calibration import load_calibration_receipt

    return (
        artifact.path,
        artifact.sha256,
        load_calibration_receipt(artifact.path, expected_sha256=artifact.sha256),
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
    assert report["evaluation"]["evidence_statement"] == (
        "interface/training receipt; no utility or behavioral evidence"
    )
    assert report["integrity"]["explicit_missing_fields"] == [
        "evaluation.trained_per_episode",
        "evaluation.step_zero_per_episode",
        "reference_runtime.records",
    ]
    changed = json.loads(json.dumps(report))
    changed["summary"]["hard_gates"]["utility_gate"]["family_passed"] = True
    with pytest.raises(ValueError, match="summary differs"):
        validate_report_summary_recomputation(changed)
    mixed_manifest = json.loads(json.dumps(report))
    mixed_manifest["training"]["seed_facts"][1]["execution_manifest_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="training and cohort authority"):
        validate_report_summary_recomputation(mixed_manifest)
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


def test_checkpoint_index_refuses_five_test_only_sixteen_transition_entries(
    tmp_path: Path,
) -> None:
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
    with pytest.raises(ExperimentContractError, match="success and job authority"):
        publish_checkpoint_index(output_directory=output, entries=entries)
    execution_manifest = publish_bytes_without_overwrite(
        output / "execution_manifest_v3.json",
        canonical_json_bytes(
            {
                "checkpoint_selection": "final_transition_only",
                "seeds": list(COHORT_SEEDS),
                "smoke": False,
                "test_only": False,
                "transitions_per_seed": 1_048_576,
            }
        ),
    )
    successes = []
    for seed in COHORT_SEEDS:
        successes.append(
            publish_bytes_without_overwrite(
                output / f"seed_{seed}/success_receipt_v2.json",
                canonical_json_bytes(
                    {
                        "evidence_class": "exploratory_fine_tuning_cycle",
                        "execution_manifest_sha256": execution_manifest.sha256,
                        "failure_receipt_present": False,
                        "outcome": "success",
                        "planned_transitions": 1_048_576,
                        "ppo_seed": seed,
                        "promotable": True,
                        "resource_controls": {
                            "cpu_time": {
                                "enforcement": "unsupported",
                                "limit_seconds": 1_200,
                                "resource": "RLIMIT_CPU",
                            },
                            "environment": {
                                "allowlist_enforced": True,
                                "environment_sha256": "d" * 64,
                                "keys": [],
                                "runtime_added_keys_removed": [],
                                "unexpected_keys": [],
                            },
                            "executed_modules": {
                                "enforcement": ("checkout_realpath_and_recorded_digest_verified"),
                                "final_sha256": "f" * 64,
                                "start_sha256": "e" * 64,
                            },
                            "filesystem": {"enforcement": "parent_observed_os_best_effort"},
                            "process_group_cleanup": {
                                "enforcement": "os_session_group_best_effort",
                                "succeeded": True,
                            },
                            "process_tree_rss": {"enforcement": "parent_observed_os_best_effort"},
                        },
                        "schema_version": 2,
                        "smoke": False,
                        "status": "succeeded",
                        "success_receipt_id": "humanoid_phase_b_seed_success/v2",
                        "test_only": False,
                        "worker_cleanup": {
                            "attempted": True,
                            "error": None,
                            "succeeded": True,
                        },
                    }
                ),
            )
        )
    job_result = publish_bytes_without_overwrite(
        output / "job_result_v1.json",
        canonical_json_bytes(
            {
                "execution_manifest_sha256": execution_manifest.sha256,
                "outcomes": [
                    {
                        "receipt": {
                            "byte_count": item.byte_count,
                            "filename": item.path.name,
                            "sha256": item.sha256,
                        },
                        "seed": seed,
                        "status": "succeeded",
                    }
                    for seed, item in zip(COHORT_SEEDS, successes, strict=True)
                ],
                "status": "succeeded",
            }
        ),
    )
    hidden_failure = json.loads(job_result.path.read_bytes())
    hidden_failure["outcomes"][2]["status"] = "counter_drift"
    hidden_job_result = publish_bytes_without_overwrite(
        output / "job_result_hidden_failure_v1.json",
        canonical_json_bytes(hidden_failure),
    )
    with pytest.raises(ExperimentContractError, match="omits a cohort success"):
        publish_checkpoint_index(
            output_directory=output,
            entries=entries,
            success_receipts=successes,
            execution_manifest=execution_manifest,
            job_result=hidden_job_result,
        )
    with pytest.raises(ExperimentContractError, match="smoke or test-only"):
        publish_checkpoint_index(
            output_directory=output,
            entries=entries,
            success_receipts=successes,
            execution_manifest=execution_manifest,
            job_result=job_result,
        )
    smoke_entries = [
        replace(
            entry,
            receipt_value=MappingProxyType(
                {
                    **dict(entry.receipt_value),
                    "evidence_class": "interface_check",
                    "promotable": False,
                    "smoke": True,
                    "test_only": False,
                }
            ),
        )
        for entry in entries
    ]
    with pytest.raises(ExperimentContractError, match="smoke or test-only"):
        publish_checkpoint_index(
            output_directory=output,
            entries=smoke_entries,
            success_receipts=successes,
            execution_manifest=execution_manifest,
            job_result=job_result,
        )


def test_cli_evaluate_policy_refuses_test_only_checkpoint_before_output_or_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    from oracle_composition.phase_b import evaluation_supervision

    spawned = False

    def spawn_sentinel(*_args: object, **_kwargs: object) -> object:
        nonlocal spawned
        spawned = True
        raise AssertionError("evaluation worker spawned before lineage admission")

    monkeypatch.setattr(evaluation_supervision, "_spawn", spawn_sentinel)
    calibration_path, calibration_sha256, _calibration_contract = _calibration(
        tmp_path / "calibration.json"
    )
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
        "--calibration-receipt",
        str(calibration_path),
        "--calibration-receipt-sha256",
        calibration_sha256,
        "--output",
        str(output),
    ]
    with pytest.raises(ExperimentContractError, match=r"lineage artifact|promotable full-budget"):
        main(
            args,
            repository_root=ROOT,
            training_dependencies=_dependencies(utility=utility_dependencies),
        )
    assert spawned is False
    assert not output.exists()


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


def test_utility_family_rejects_three_of_five_and_a_missing_checkpoint() -> None:
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
                        passed=not (checkpoint_index >= 3 and evaluation_index >= 15),
                    )
                )
    three_of_five = utility_gate(checkpoints=checkpoints, episodes=episodes)
    assert three_of_five["passed_checkpoint_count"] == 3
    assert three_of_five["family_passed"] is False

    missing = utility_gate(
        checkpoints=checkpoints[:-1],
        episodes=[episode for episode in episodes if episode.policy_seed != COHORT_SEEDS[-1]],
    )
    assert missing["required_checkpoint_count"] == 5
    assert len(missing["checkpoint_results"]) == 4
    assert missing["family_passed"] is False


def test_task_success_endpoint_uses_per_checkpoint_exact_interval_and_paired_effect(
    tmp_path: Path,
) -> None:
    episodes = [
        replace(
            _episode(
                policy_seed=11,
                evaluation_seed=120101 + index,
                cell="fixed_round_trip",
                task_success=index >= 16,
            ),
            segment_errors={
                "fast": 0.1 if index < 16 else 0.3,
                "return_fast": 0.1,
                "slow": 0.1,
            },
        )
        for index in range(20)
    ]
    baseline = [
        replace(
            episode,
            checkpoint_sha256="1" * 64,
            task_success=index >= 10,
            segment_errors={
                "fast": 0.1 if index < 10 else 0.3,
                "return_fast": 0.1,
                "slow": 0.1,
            },
        )
        for index, episode in enumerate(episodes)
    ]
    _path, _sha, calibration = _calibration(tmp_path / "calibration.json")
    endpoint = task_success_endpoint(episodes, step_zero_episodes=baseline, calibration=calibration)
    checkpoint = endpoint["primary_endpoint"]["checkpoint_results"][0]
    assert checkpoint["successes"] == 16
    assert checkpoint["exact_binomial_95_interval"] == list(exact_binomial_interval(16, 20))
    assert endpoint["primary_endpoint"]["pooled_episode_estimate"] is None
    assert endpoint["paired_seed_level_effects"][0][
        "effect_candidate_minus_step_zero"
    ] == pytest.approx(0.3)
    assert endpoint["unsafe_arm_may_outrank_safe_arm"] is False


def test_task_success_without_calibration_is_explicitly_non_scoring() -> None:
    episodes = [
        _episode(
            policy_seed=11,
            evaluation_seed=120101 + index,
            cell="fixed_round_trip",
            task_success=True,
        )
        for index in range(20)
    ]
    endpoint = task_success_endpoint(
        episodes,
        step_zero_episodes=[replace(item, checkpoint_sha256="1" * 64) for item in episodes],
        calibration=None,
    )
    row = endpoint["primary_endpoint"]["checkpoint_results"][0]
    assert endpoint["primary_endpoint"]["scoring_status"] == "non_scoring_missing_calibration"
    assert {row[name] for name in ("successes", "total", "proportion")} == {None}
    assert row["exact_binomial_95_interval"] is None


def test_task_success_endpoint_rejects_mixed_checkpoint_identity() -> None:
    episodes = [
        _episode(
            policy_seed=11,
            evaluation_seed=120101 + index,
            cell="fixed_round_trip",
            task_success=True,
        )
        for index in range(20)
    ]
    episodes[0] = replace(episodes[0], checkpoint_sha256="f" * 64)
    baseline = [replace(episode, checkpoint_sha256="1" * 64) for episode in episodes]
    calibration = TaskSuccessCalibration(
        sha256="2" * 64,
        segment_speed_error_bands_m_s={"fast": 1.0, "return_fast": 1.0, "slow": 1.0},
        transition_latency_caps_steps=(64, 64),
        settled_state_normalized_error_band=1.0,
        censoring_latency_steps=65,
    )
    with pytest.raises(ValueError, match="checkpoint identity"):
        task_success_endpoint(episodes, step_zero_episodes=baseline, calibration=calibration)


def test_task_success_endpoint_reports_paired_effects_across_five_ppo_seeds() -> None:
    episodes = [
        replace(
            _episode(
                policy_seed=policy_seed,
                evaluation_seed=120101 + index,
                cell="fixed_round_trip",
                checkpoint_sha256=f"{policy_seed:064x}",
                task_success=index >= 16,
            ),
            segment_errors={
                "fast": 0.1 if index < 16 else 1.1,
                "return_fast": 0.1,
                "slow": 0.1,
            },
        )
        for policy_seed in COHORT_SEEDS
        for index in range(20)
    ]
    baseline = [
        replace(
            episode,
            checkpoint_sha256="1" * 64,
            task_success=index % 20 >= 10,
            segment_errors={
                "fast": 0.1 if index % 20 < 10 else 1.1,
                "return_fast": 0.1,
                "slow": 0.1,
            },
        )
        for index, episode in enumerate(episodes)
    ]
    calibration = TaskSuccessCalibration(
        sha256="2" * 64,
        segment_speed_error_bands_m_s={"fast": 1.0, "return_fast": 1.0, "slow": 1.0},
        transition_latency_caps_steps=(64, 64),
        settled_state_normalized_error_band=1.0,
        censoring_latency_steps=65,
    )
    endpoint = task_success_endpoint(
        episodes,
        step_zero_episodes=baseline,
        calibration=calibration,
    )
    assert endpoint["primary_endpoint"]["five_seed_cohort_complete"] is True
    assert len(endpoint["primary_endpoint"]["checkpoint_results"]) == 5
    assert len(endpoint["paired_seed_level_effects"]) == 5
    assert endpoint["paired_seed_level_summary"][
        "mean_effect_across_policy_seeds"
    ] == pytest.approx(0.3)


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


def _settled_calibration(band: float = 0.25) -> TaskSuccessCalibration:
    return TaskSuccessCalibration(
        sha256="a" * 64,
        segment_speed_error_bands_m_s={"fast": 0.5, "slow": 0.5, "return_fast": 0.5},
        transition_latency_caps_steps=(16, 16),
        settled_state_normalized_error_band=band,
        censoring_latency_steps=65,
    )


@pytest.mark.parametrize("consumer", ["evaluator", "report"])
@pytest.mark.parametrize(
    ("error", "expected"),
    [(0.0, True), (0.25, True), (math.nextafter(0.25, math.inf), False), (None, False)],
)
def test_settled_band_is_required_by_both_task_success_consumers(
    consumer: str, error: float | None, expected: bool
) -> None:
    episode = replace(
        _episode(policy_seed=11, evaluation_seed=120101, cell="fixed_round_trip"),
        settled_state_normalized_error=error,
    )
    calibration = _settled_calibration()
    outcome = (
        _score_task_success(episode, calibration).task_success
        if consumer == "evaluator"
        else _derived_task_success(episode, calibration)
    )
    assert outcome is expected
    assert episode.utility_passed is True


@pytest.mark.parametrize("band", [0.0, 0.125, 0.25, 0.5])
def test_settled_scoring_consumes_the_calibrated_band(band: float) -> None:
    episode = replace(
        _episode(policy_seed=11, evaluation_seed=120101, cell="fixed_round_trip"),
        settled_state_normalized_error=0.25,
    )
    expected = band >= 0.25
    assert _score_task_success(episode, _settled_calibration(band)).task_success is expected
    assert _derived_task_success(episode, _settled_calibration(band)) is expected


@pytest.mark.parametrize("cell", ["fixed_round_trip", "hold_expert"])
def test_settled_scoring_without_calibration_stays_non_scoring(cell: str) -> None:
    episode = _episode(policy_seed=11, evaluation_seed=120101, cell=cell, task_success=True)
    assert _score_task_success(episode, None).task_success is None
    if cell == "hold_expert":
        assert _score_task_success(episode, _settled_calibration()).task_success is None


@pytest.mark.parametrize("error", [math.nextafter(0.25, math.inf), None])
def test_settled_report_recomputes_serialized_rows_instead_of_cached_success(
    error: float | None,
) -> None:
    baseline = [
        _episode(
            policy_seed=11,
            evaluation_seed=seed,
            cell="fixed_round_trip",
            checkpoint_sha256="b" * 64,
            task_success=True,
        )
        for seed in range(120101, 120121)
    ]
    candidate = [replace(item, checkpoint_sha256="c" * 64) for item in baseline]
    candidate[0] = replace(candidate[0], settled_state_normalized_error=error)
    encoded = canonical_json_bytes([item.to_dict() for item in candidate])
    reloaded = tuple(ProtectedEpisodeMetrics.from_dict(row) for row in json.loads(encoded))
    assert all(item.task_success is True for item in reloaded)
    endpoint = task_success_endpoint(
        reloaded, step_zero_episodes=baseline, calibration=_settled_calibration()
    )
    assert endpoint["primary_endpoint"]["checkpoint_results"][0]["successes"] == 19
    assert endpoint["paired_seed_level_effects"][0]["effect_candidate_minus_step_zero"] == (
        pytest.approx(-0.05)
    )
