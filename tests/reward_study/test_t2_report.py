from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from oracle_composition.phase_b.protected_metrics import ERROR_NAMES
from oracle_composition.reward_study.t2_evaluator import T2EpisodeMetrics, summarize_t2_speeds
from oracle_composition.reward_study.t2_report import (
    T2_POLICY_SEEDS,
    build_t2_study_report,
    paired_checkpoint_effect,
    publish_t2_study_report,
)


def _episode(policy_seed: int, evaluation_seed: int, *, speed: float) -> T2EpisodeMetrics:
    values = (speed,) * 1_000
    summaries = summarize_t2_speeds(values)
    return T2EpisodeMetrics(
        action_bounds_ok=True,
        checkpoint_sha256=hashlib.sha256(f"checkpoint:{policy_seed}".encode()).hexdigest(),
        com_forward_speed_m_s=values,
        complete_trace=True,
        descriptive_stock_return=10_000.0 + policy_seed,
        evaluation_seed=evaluation_seed,
        failure_reasons=(),
        first_fall_step=None,
        fraction_steps_in_target_band=summaries["fraction_steps_in_target_band"],
        fall_step_count=0,
        forbidden_contact_count=0,
        longest_out_of_band_run_steps=summaries["longest_out_of_band_run_steps"],
        mean_absolute_per_step_error_m_s=summaries["mean_absolute_per_step_error_m_s"],
        observed_steps=1_000,
        policy_seed=policy_seed,
        protected_task_return=summaries["protected_task_return"],
        six_tracking_rmse={name: 0.0 for name in ERROR_NAMES},
        speed_quantiles_m_s=summaries["speed_quantiles_m_s"],
        trace_sha256=hashlib.sha256(
            f"trace:{policy_seed}:{evaluation_seed}:{speed}".encode()
        ).hexdigest(),
    )


def _arms() -> tuple[list[T2EpisodeMetrics], list[T2EpisodeMetrics]]:
    baseline = []
    candidate = []
    for index, policy_seed in enumerate(T2_POLICY_SEEDS):
        for evaluation_seed in range(97001, 97021):
            baseline.append(_episode(policy_seed, evaluation_seed, speed=3.6 + index * 0.01))
            candidate.append(_episode(policy_seed, evaluation_seed, speed=3.1 + index * 0.01))
    return baseline, candidate


def _inputs() -> dict[str, object]:
    return {
        "baseline_reward_sha256": "a" * 64,
        "candidate_reward_sha256": "b" * 64,
        "evaluator_design_sha256": "c" * 64,
        "integrated_pairing_receipt_sha256": "3" * 64,
        "oracle_sha256": "d" * 64,
        "study_manifest_sha256": "e" * 64,
        "study_pairing_sha256": "f" * 64,
        "trace_index_sha256": "1" * 64,
        "training_design_sha256": "2" * 64,
    }


def test_paired_checkpoint_effect_uses_five_policies_and_exact_sign_result() -> None:
    baseline = {seed: 0.6 + index * 0.01 for index, seed in enumerate(T2_POLICY_SEEDS)}
    candidate = {seed: 0.1 + index * 0.01 for index, seed in enumerate(T2_POLICY_SEEDS)}
    result = paired_checkpoint_effect(baseline, candidate)
    assert result["policy_count"] == 5
    assert result["estimable"] is True
    assert result["mean_difference_m_s"] == pytest.approx(0.5)
    assert result["median_difference_m_s"] == pytest.approx(0.5)
    assert result["range_difference_m_s"] == pytest.approx([0.5, 0.5])
    assert result["paired_95_percent_t_interval_m_s"] == pytest.approx([0.5, 0.5])
    assert result["exact_one_sided_sign_result"]["p_value"] == pytest.approx(1 / 32)
    assert result["task_improvement_rule"]["passed"] is True


def test_paired_checkpoint_effect_refuses_pooled_n_100_interval() -> None:
    pooled = {index: 1.0 for index in range(100)}
    with pytest.raises(ValueError, match="pooled n = 100 is forbidden"):
        paired_checkpoint_effect(pooled, pooled)


def test_reward_diagnostic_corruption_does_not_change_protected_report_metrics() -> None:
    baseline, candidate = _arms()
    first = build_t2_study_report(
        inputs=_inputs(),
        baseline_episodes=baseline,
        candidate_episodes=candidate,
        reward_diagnostics={"r_task": 100.0, "r_train": 101.0},
    )
    corrupted = build_t2_study_report(
        inputs=_inputs(),
        baseline_episodes=baseline,
        candidate_episodes=candidate,
        reward_diagnostics={"r_task": "corrupted", "r_train": {"wrong": True}},
    )
    assert first["evaluation"] == corrupted["evaluation"]
    assert first["summary"] == corrupted["summary"]
    assert first["summary"]["gates"] == {
        "baseline_viable": True,
        "candidate_broke_safety": False,
        "candidate_broke_tracking": False,
        "candidate_safety_passed": True,
        "candidate_tracking_passed": True,
        "overall_candidate_passed": True,
        "task_improvement_rule_passed": True,
    }
    assert first["summary"]["weighted_aggregate"] is None
    assert len(first["summary"]["tradeoff_table"]) == 9


def test_incomplete_episode_is_reported_as_a_safety_failure_without_pooled_inference() -> None:
    baseline, candidate = _arms()
    candidate[0] = replace(
        candidate[0],
        com_forward_speed_m_s=candidate[0].com_forward_speed_m_s[:-1],
        complete_trace=False,
        descriptive_stock_return=None,
        failure_reasons=("incomplete_trace",),
        fraction_steps_in_target_band=None,
        longest_out_of_band_run_steps=None,
        mean_absolute_per_step_error_m_s=None,
        observed_steps=999,
        protected_task_return=None,
        six_tracking_rmse=None,
        speed_quantiles_m_s=None,
    )
    report = build_t2_study_report(
        inputs=_inputs(),
        baseline_episodes=baseline,
        candidate_episodes=candidate,
        reward_diagnostics={},
    )
    summary = report["summary"]
    assert summary["candidate"]["incomplete_trace_count"] == 1
    assert summary["candidate"]["safety_passed"] is False
    assert summary["paired_effect"]["estimable"] is False
    assert summary["paired_effect"]["task_improvement_rule"]["passed"] is False
    assert summary["gates"]["candidate_broke_safety"] is True
    assert summary["gates"]["overall_candidate_passed"] is False


def test_telemetry_changes_do_not_change_deterministic_scientific_receipt(
    tmp_path: Path,
) -> None:
    baseline, candidate = _arms()
    report = build_t2_study_report(
        inputs=_inputs(),
        baseline_episodes=baseline,
        candidate_episodes=candidate,
        reward_diagnostics={"status": "diagnostic_only"},
    )
    first = publish_t2_study_report(
        output_directory=tmp_path / "first",
        report=report,
        telemetry={"wall_time_seconds": 1.0},
    )
    second = publish_t2_study_report(
        output_directory=tmp_path / "second",
        report=report,
        telemetry={"wall_time_seconds": "corrupted", "host_note": "ignored by metrics"},
    )
    assert first.scientific_receipt.sha256 == second.scientific_receipt.sha256
    assert first.scientific_receipt.path.read_bytes() == second.scientific_receipt.path.read_bytes()
    assert first.telemetry.sha256 != second.telemetry.sha256
