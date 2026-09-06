"""Deterministic paired T2 report with telemetry kept in a separate artifact."""

from __future__ import annotations

import json
import math
import platform
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median, stdev

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.experiments.artifact_io import (
    PublishedArtifact,
    publish_bytes_without_overwrite,
)
from oracle_composition.phase_b.protected_metrics import ERROR_NAMES, ERROR_SCALES
from oracle_composition.phase_b.report_v2 import (
    SCIENTIFIC_RECEIPT_SCHEMA_ID,
    TELEMETRY_SCHEMA_ID,
)

from .study_manifest import STUDY_CLAIM_CEILING, STUDY_EVIDENCE_CLASS
from .t2_evaluator import (
    T2_EVALUATION_SEEDS,
    T2_HORIZON_STEPS,
    T2_REPORT_SCHEMA_ID,
    T2EpisodeMetrics,
)

T2_REPORT_WRITER_ID = "t2_reward_study_report_writer/v1"
T2_PAIRED_ENDPOINT_ID = "t2_paired_checkpoint_com_speed_mae/v1"
T2_MINIMUM_RELEVANT_EFFECT_M_S = 0.25
T2_PAIRED_T_CRITICAL_95_DF4 = 2.7764451051977987
T2_POLICY_SEEDS = (121001, 121101, 121201, 121301, 121401)


def _sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _optional_mean(values: Sequence[float | None]) -> float | None:
    if not values or any(value is None for value in values):
        return None
    return fmean(float(value) for value in values if value is not None)


def _validate_report_inputs(
    inputs: Mapping[str, object],
    *,
    required_inputs: set[str],
) -> None:
    if type(inputs) is not dict or set(inputs) != required_inputs:
        raise ValueError("T2 report input identities differ")
    for field in required_inputs:
        _sha256(inputs[field], field=field)


def checkpoint_summary(episodes: Sequence[T2EpisodeMetrics]) -> dict[str, object]:
    """Summarize one policy checkpoint over exactly 20 predeclared episodes."""

    if len(episodes) != len(T2_EVALUATION_SEEDS):
        raise ValueError("a T2 checkpoint requires exactly 20 evaluation episodes")
    ordered = sorted(episodes, key=lambda item: item.evaluation_seed)
    if tuple(item.evaluation_seed for item in ordered) != T2_EVALUATION_SEEDS:
        raise ValueError("a T2 checkpoint evaluation seed set differs")
    policy_seeds = {item.policy_seed for item in ordered}
    checkpoint_hashes = {item.checkpoint_sha256 for item in ordered}
    if len(policy_seeds) != 1 or len(checkpoint_hashes) != 1:
        raise ValueError("a T2 checkpoint summary mixes policy or checkpoint identities")
    component_gates = {}
    for name in ERROR_NAMES:
        passed = sum(
            item.complete_trace
            and item.six_tracking_rmse is not None
            and float(item.six_tracking_rmse[name]) <= ERROR_SCALES[name]
            for item in ordered
        )
        component_gates[name] = {
            "arm_rule_ready": passed >= 16,
            "episode_pass_count": passed,
            "episode_total": 20,
            "threshold": ERROR_SCALES[name],
        }
    tracking_pass_count = sum(item.tracking_passed for item in ordered)
    primary_values = [item.mean_absolute_per_step_error_m_s for item in ordered]
    task_returns = [item.protected_task_return for item in ordered]
    band_fractions = [item.fraction_steps_in_target_band for item in ordered]
    stock_returns = [item.descriptive_stock_return for item in ordered]
    return {
        "checkpoint_sha256": ordered[0].checkpoint_sha256,
        "complete_episode_count": sum(item.complete_trace for item in ordered),
        "descriptive_stock_return_mean": _optional_mean(stock_returns),
        "evaluation_episode_count": len(ordered),
        "fall_episode_count": sum(item.first_fall_step is not None for item in ordered),
        "fall_step_count": sum(item.fall_step_count for item in ordered),
        "forbidden_contact_count": sum(item.forbidden_contact_count for item in ordered),
        "fraction_steps_in_target_band_mean": _optional_mean(band_fractions),
        "incomplete_trace_count": sum(not item.complete_trace for item in ordered),
        "invalid_action_episode_count": sum(not item.action_bounds_ok for item in ordered),
        "mean_absolute_per_step_error_m_s": _optional_mean(primary_values),
        "per_episode_primary": [
            {
                "evaluation_seed": item.evaluation_seed,
                "mean_absolute_per_step_error_m_s": item.mean_absolute_per_step_error_m_s,
                "trace_sha256": item.trace_sha256,
            }
            for item in ordered
        ],
        "policy_seed": ordered[0].policy_seed,
        "protected_task_return_mean": _optional_mean(task_returns),
        "safety_passed": all(item.safety_passed for item in ordered),
        "six_tracking_component_gates": component_gates,
        "six_tracking_rmse_means": {
            name: _optional_mean(
                [
                    None if item.six_tracking_rmse is None else item.six_tracking_rmse[name]
                    for item in ordered
                ]
            )
            for name in ERROR_NAMES
        },
        "tracking_episode_pass_count": tracking_pass_count,
        "tracking_passed": tracking_pass_count >= 16,
    }


def arm_summary(
    *,
    arm_label: str,
    reward_sha256: str,
    episodes: Sequence[T2EpisodeMetrics],
) -> dict[str, object]:
    """Apply the 100-episode safety and 4-of-5 tracking gates."""

    if arm_label not in {"baseline", "candidate"}:
        raise ValueError("T2 arm label is unknown")
    _sha256(reward_sha256, field=f"{arm_label} reward SHA-256")
    if len(episodes) != len(T2_POLICY_SEEDS) * len(T2_EVALUATION_SEEDS):
        raise ValueError("a T2 arm requires exactly 100 predeclared episodes")
    grouped: dict[int, list[T2EpisodeMetrics]] = defaultdict(list)
    seen = set()
    for episode in episodes:
        identity = (episode.policy_seed, episode.evaluation_seed)
        if identity in seen:
            raise ValueError("a T2 arm contains a duplicate policy/evaluation seed pair")
        seen.add(identity)
        grouped[episode.policy_seed].append(episode)
    if set(grouped) != set(T2_POLICY_SEEDS):
        raise ValueError("a T2 arm policy seed set differs")
    checkpoints = [checkpoint_summary(grouped[seed]) for seed in T2_POLICY_SEEDS]
    if len({checkpoint["checkpoint_sha256"] for checkpoint in checkpoints}) != 5:
        raise ValueError("a T2 arm must bind one distinct final checkpoint per policy seed")
    component_arm_gates = {}
    for name in ERROR_NAMES:
        passed = sum(
            checkpoint["six_tracking_component_gates"][name]["arm_rule_ready"]
            for checkpoint in checkpoints
        )
        component_arm_gates[name] = {
            "checkpoint_pass_count": passed,
            "checkpoint_total": 5,
            "passed": passed >= 4,
            "threshold": ERROR_SCALES[name],
        }
    tracking_checkpoints = sum(checkpoint["tracking_passed"] for checkpoint in checkpoints)
    return {
        "arm_label": arm_label,
        "checkpoint_summaries": checkpoints,
        "complete_trace_count": sum(item.complete_trace for item in episodes),
        "fall_episode_count": sum(item.first_fall_step is not None for item in episodes),
        "forbidden_contact_count": sum(item.forbidden_contact_count for item in episodes),
        "incomplete_trace_count": sum(not item.complete_trace for item in episodes),
        "invalid_action_episode_count": sum(not item.action_bounds_ok for item in episodes),
        "primary_by_policy_seed": {
            str(checkpoint["policy_seed"]): checkpoint["mean_absolute_per_step_error_m_s"]
            for checkpoint in checkpoints
        },
        "reward_sha256": reward_sha256,
        "safety_passed": all(item.safety_passed for item in episodes),
        "six_tracking_component_arm_gates": component_arm_gates,
        "tracking_checkpoint_pass_count": tracking_checkpoints,
        "tracking_passed": tracking_checkpoints >= 4,
    }


def paired_checkpoint_effect(
    baseline_by_seed: Mapping[int, float],
    candidate_by_seed: Mapping[int, float],
) -> dict[str, object]:
    """Compute the preregistered five-policy paired estimate, never pooled episodes."""

    if len(baseline_by_seed) != 5 or len(candidate_by_seed) != 5:
        raise ValueError(
            "paired T2 inference requires five policy checkpoints; pooled n = 100 is forbidden"
        )
    if set(baseline_by_seed) != set(T2_POLICY_SEEDS) or set(candidate_by_seed) != set(
        T2_POLICY_SEEDS
    ):
        raise ValueError("paired T2 inference policy seed identities differ")
    for values in (baseline_by_seed, candidate_by_seed):
        if any(
            type(value) not in {int, float} or not math.isfinite(float(value))
            for value in values.values()
        ):
            raise ValueError("paired T2 checkpoint endpoint is non-finite")
    differences = [
        float(baseline_by_seed[seed]) - float(candidate_by_seed[seed]) for seed in T2_POLICY_SEEDS
    ]
    mean_difference = fmean(differences)
    standard_error = stdev(differences) / math.sqrt(5)
    half_width = T2_PAIRED_T_CRITICAL_95_DF4 * standard_error
    positive = sum(value > 0.0 for value in differences)
    negative = sum(value < 0.0 for value in differences)
    non_ties = positive + negative
    sign_p = math.fsum(math.comb(non_ties, count) for count in range(positive, non_ties + 1)) / (
        2**non_ties
    )
    all_positive = positive == 5
    task_rule = all_positive and mean_difference >= T2_MINIMUM_RELEVANT_EFFECT_M_S
    return {
        "difference_definition": "baseline_mae_minus_candidate_mae_positive_favors_candidate",
        "differences_by_policy_seed": [
            {"difference_m_s": value, "policy_seed": seed}
            for seed, value in zip(T2_POLICY_SEEDS, differences, strict=True)
        ],
        "exact_one_sided_sign_result": {
            "alternative": "baseline_mae_greater_than_candidate_mae",
            "negative_count": negative,
            "non_tied_policy_count": non_ties,
            "p_value": sign_p,
            "positive_count": positive,
            "tie_count": 5 - non_ties,
        },
        "estimable": True,
        "independent_unit": "final_policy_checkpoint",
        "mean_difference_m_s": mean_difference,
        "median_difference_m_s": median(differences),
        "non_estimable_reason": None,
        "paired_95_percent_t_interval_m_s": [
            mean_difference - half_width,
            mean_difference + half_width,
        ],
        "policy_count": 5,
        "range_difference_m_s": [min(differences), max(differences)],
        "task_improvement_rule": {
            "all_five_differences_positive": all_positive,
            "mean_reduction_at_least_0.25_m_s": (mean_difference >= T2_MINIMUM_RELEVANT_EFFECT_M_S),
            "passed": task_rule,
        },
    }


def _effect_from_arms(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    baseline_values = baseline["primary_by_policy_seed"]
    candidate_values = candidate["primary_by_policy_seed"]
    if any(value is None for value in baseline_values.values()) or any(
        value is None for value in candidate_values.values()
    ):
        return {
            "difference_definition": ("baseline_mae_minus_candidate_mae_positive_favors_candidate"),
            "differences_by_policy_seed": [
                {
                    "difference_m_s": (
                        None
                        if baseline_values[str(seed)] is None or candidate_values[str(seed)] is None
                        else float(baseline_values[str(seed)]) - float(candidate_values[str(seed)])
                    ),
                    "policy_seed": seed,
                }
                for seed in T2_POLICY_SEEDS
            ],
            "estimable": False,
            "exact_one_sided_sign_result": {
                "alternative": "baseline_mae_greater_than_candidate_mae",
                "negative_count": None,
                "non_tied_policy_count": None,
                "p_value": None,
                "positive_count": None,
                "tie_count": None,
            },
            "independent_unit": "final_policy_checkpoint",
            "mean_difference_m_s": None,
            "median_difference_m_s": None,
            "non_estimable_reason": "one_or_more_incomplete_checkpoint_endpoints",
            "paired_95_percent_t_interval_m_s": None,
            "policy_count": 5,
            "range_difference_m_s": None,
            "task_improvement_rule": {
                "all_five_differences_positive": False,
                "mean_reduction_at_least_0.25_m_s": False,
                "passed": False,
            },
        }
    return paired_checkpoint_effect(
        {int(seed): value for seed, value in baseline_values.items()},
        {int(seed): value for seed, value in candidate_values.items()},
    )


def _tradeoff_table(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    effect: Mapping[str, object],
) -> list[dict[str, object]]:
    rows = [
        {
            "baseline_passed": None,
            "candidate_passed": effect["task_improvement_rule"]["passed"],
            "criterion": "task_improvement_rule",
            "weight": None,
        },
        {
            "baseline_passed": baseline["safety_passed"],
            "candidate_passed": candidate["safety_passed"],
            "criterion": "safety",
            "weight": None,
        },
        {
            "baseline_passed": baseline["tracking_passed"],
            "candidate_passed": candidate["tracking_passed"],
            "criterion": "tracking_all_six_components",
            "weight": None,
        },
    ]
    for name in ERROR_NAMES:
        rows.append(
            {
                "baseline_passed": baseline["six_tracking_component_arm_gates"][name]["passed"],
                "candidate_passed": candidate["six_tracking_component_arm_gates"][name]["passed"],
                "criterion": name,
                "threshold": ERROR_SCALES[name],
                "weight": None,
            }
        )
    return rows


def _summary(
    *,
    baseline_reward_sha256: str,
    candidate_reward_sha256: str,
    baseline_episodes: Sequence[T2EpisodeMetrics],
    candidate_episodes: Sequence[T2EpisodeMetrics],
) -> dict[str, object]:
    baseline = arm_summary(
        arm_label="baseline",
        reward_sha256=baseline_reward_sha256,
        episodes=baseline_episodes,
    )
    candidate = arm_summary(
        arm_label="candidate",
        reward_sha256=candidate_reward_sha256,
        episodes=candidate_episodes,
    )
    effect = _effect_from_arms(baseline, candidate)
    baseline_viable = bool(baseline["safety_passed"] and baseline["tracking_passed"])
    task_passed = bool(effect["task_improvement_rule"]["passed"])
    candidate_safety = bool(candidate["safety_passed"])
    candidate_tracking = bool(candidate["tracking_passed"])
    return {
        "baseline": baseline,
        "candidate": candidate,
        "gates": {
            "baseline_viable": baseline_viable,
            "candidate_broke_safety": baseline_viable and not candidate_safety,
            "candidate_broke_tracking": baseline_viable and not candidate_tracking,
            "candidate_safety_passed": candidate_safety,
            "candidate_tracking_passed": candidate_tracking,
            "overall_candidate_passed": (
                baseline_viable and task_passed and candidate_safety and candidate_tracking
            ),
            "task_improvement_rule_passed": task_passed,
        },
        "paired_effect": effect,
        "tradeoff_table": _tradeoff_table(baseline, candidate, effect),
        "weighted_aggregate": None,
    }


def build_t2_study_report(
    *,
    inputs: Mapping[str, object],
    baseline_episodes: Sequence[T2EpisodeMetrics],
    candidate_episodes: Sequence[T2EpisodeMetrics],
    reward_diagnostics: Mapping[str, object],
) -> dict[str, object]:
    """Build report content whose protected summaries ignore reward diagnostics."""

    required_inputs = {
        "baseline_reward_sha256",
        "candidate_reward_sha256",
        "evaluator_design_sha256",
        "integrated_pairing_receipt_sha256",
        "oracle_sha256",
        "study_manifest_sha256",
        "study_pairing_sha256",
        "trace_index_sha256",
        "training_design_sha256",
    }
    _validate_report_inputs(inputs, required_inputs=required_inputs)
    if type(reward_diagnostics) is not dict:
        raise ValueError("T2 reward diagnostics must be a separate object")
    summary = _summary(
        baseline_reward_sha256=inputs["baseline_reward_sha256"],
        candidate_reward_sha256=inputs["candidate_reward_sha256"],
        baseline_episodes=baseline_episodes,
        candidate_episodes=candidate_episodes,
    )
    report = {
        "claim_ceiling": STUDY_CLAIM_CEILING,
        "diagnostics": {
            "candidate_reward_outputs_are_not_endpoints": True,
            "reward_outputs": dict(reward_diagnostics),
        },
        "evaluation": {
            "calibration": "none",
            "deterministic_actions": True,
            "evaluation_seeds": list(T2_EVALUATION_SEEDS),
            "expert_start": True,
            "full_sufficient_traces_required": True,
            "horizon_steps": T2_HORIZON_STEPS,
            "per_episode_by_arm": {
                "baseline": [item.to_dict() for item in baseline_episodes],
                "candidate": [item.to_dict() for item in candidate_episodes],
            },
            "protected_metrics_grade_candidate_reward": False,
        },
        "evidence_class": STUDY_EVIDENCE_CLASS,
        "generated_utc": "recorded_in_separate_telemetry",
        "inputs": dict(inputs),
        "integrity": {
            "deterministic_scientific_receipt": True,
            "telemetry_location": "telemetry_t2_v1.json",
            "wall_clock_and_host_excluded": True,
        },
        "paired_endpoint_id": T2_PAIRED_ENDPOINT_ID,
        "report_schema_id": T2_REPORT_SCHEMA_ID,
        "report_writer_id": T2_REPORT_WRITER_ID,
        "schema_version": 1,
        "scientific_receipt_schema_id": SCIENTIFIC_RECEIPT_SCHEMA_ID,
        "summary": summary,
        "telemetry_schema_id": TELEMETRY_SCHEMA_ID,
    }
    validate_t2_study_report(report)
    canonical_json_bytes(report)
    return report


def validate_t2_study_report(value: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "claim_ceiling",
        "diagnostics",
        "evaluation",
        "evidence_class",
        "generated_utc",
        "inputs",
        "integrity",
        "paired_endpoint_id",
        "report_schema_id",
        "report_writer_id",
        "schema_version",
        "scientific_receipt_schema_id",
        "summary",
        "telemetry_schema_id",
    }
    if type(value) is not dict or set(value) != expected:
        raise ValueError("T2 report fields differ")
    if (
        value["schema_version"] != 1
        or value["report_schema_id"] != T2_REPORT_SCHEMA_ID
        or value["report_writer_id"] != T2_REPORT_WRITER_ID
        or value["paired_endpoint_id"] != T2_PAIRED_ENDPOINT_ID
        or value["scientific_receipt_schema_id"] != SCIENTIFIC_RECEIPT_SCHEMA_ID
        or value["telemetry_schema_id"] != TELEMETRY_SCHEMA_ID
        or value["generated_utc"] != "recorded_in_separate_telemetry"
        or value["evidence_class"] != STUDY_EVIDENCE_CLASS
        or value["claim_ceiling"] != STUDY_CLAIM_CEILING
    ):
        raise ValueError("T2 report identity or claim boundary differs")
    integrity = value["integrity"]
    if integrity != {
        "deterministic_scientific_receipt": True,
        "telemetry_location": "telemetry_t2_v1.json",
        "wall_clock_and_host_excluded": True,
    }:
        raise ValueError("T2 report receipt and telemetry split differs")
    evaluation = value["evaluation"]
    required_evaluation = {
        "calibration",
        "deterministic_actions",
        "evaluation_seeds",
        "expert_start",
        "full_sufficient_traces_required",
        "horizon_steps",
        "per_episode_by_arm",
        "protected_metrics_grade_candidate_reward",
    }
    if type(evaluation) is not dict or set(evaluation) != required_evaluation:
        raise ValueError("T2 report evaluation fields differ")
    if any(
        (
            evaluation["calibration"] != "none",
            evaluation["deterministic_actions"] is not True,
            evaluation["evaluation_seeds"] != list(T2_EVALUATION_SEEDS),
            evaluation["expert_start"] is not True,
            evaluation["full_sufficient_traces_required"] is not True,
            evaluation["horizon_steps"] != T2_HORIZON_STEPS,
            evaluation["protected_metrics_grade_candidate_reward"] is not False,
        )
    ):
        raise ValueError("T2 report evaluation contract differs")
    per_arm = evaluation["per_episode_by_arm"]
    if type(per_arm) is not dict or set(per_arm) != {"baseline", "candidate"}:
        raise ValueError("T2 report arm episode fields differ")
    baseline = [T2EpisodeMetrics.from_dict(item) for item in per_arm["baseline"]]
    candidate = [T2EpisodeMetrics.from_dict(item) for item in per_arm["candidate"]]
    inputs = value["inputs"]
    _validate_report_inputs(
        inputs,
        required_inputs={
            "baseline_reward_sha256",
            "candidate_reward_sha256",
            "evaluator_design_sha256",
            "integrated_pairing_receipt_sha256",
            "oracle_sha256",
            "study_manifest_sha256",
            "study_pairing_sha256",
            "trace_index_sha256",
            "training_design_sha256",
        },
    )
    expected_summary = _summary(
        baseline_reward_sha256=inputs["baseline_reward_sha256"],
        candidate_reward_sha256=inputs["candidate_reward_sha256"],
        baseline_episodes=baseline,
        candidate_episodes=candidate,
    )
    if value["summary"] != expected_summary:
        raise ValueError("T2 report summary differs from recomputed episode rows")
    diagnostics = value["diagnostics"]
    if (
        type(diagnostics) is not dict
        or set(diagnostics)
        != {
            "candidate_reward_outputs_are_not_endpoints",
            "reward_outputs",
        }
        or diagnostics["candidate_reward_outputs_are_not_endpoints"] is not True
        or type(diagnostics["reward_outputs"]) is not dict
    ):
        raise ValueError("T2 report reward-diagnostic boundary differs")
    canonical_json_bytes(dict(value))
    return dict(value)


def load_t2_study_report(path: Path) -> dict[str, object]:
    candidate = Path(path)
    if (
        candidate.is_symlink()
        or not candidate.is_file()
        or candidate.stat().st_size > 512 * 1024**2
    ):
        raise ValueError("T2 report is unavailable or oversized")
    encoded = candidate.read_bytes()
    try:
        value = json.loads(encoded)
    except (UnicodeError, ValueError) as exc:
        raise ValueError("T2 report is not JSON") from exc
    if type(value) is not dict or canonical_json_bytes(value) != encoded:
        raise ValueError("T2 report is not canonical JSON")
    return validate_t2_study_report(value)


@dataclass(frozen=True, slots=True)
class T2ReportArtifacts:
    scientific_receipt: PublishedArtifact
    telemetry: PublishedArtifact


def publish_t2_study_report(
    *,
    output_directory: Path,
    report: Mapping[str, object],
    telemetry: Mapping[str, object],
) -> T2ReportArtifacts:
    """Publish deterministic science, then host/wall telemetry bound to its hash."""

    validated = validate_t2_study_report(report)
    scientific = publish_bytes_without_overwrite(
        Path(output_directory) / "scientific_receipt_t2_v1.json",
        canonical_json_bytes(validated),
    )
    telemetry_value = {
        "host": {"machine": platform.machine(), "system": platform.system()},
        "measurements": dict(telemetry),
        "schema_version": 1,
        "scientific_receipt_sha256": scientific.sha256,
        "telemetry_schema_id": TELEMETRY_SCHEMA_ID,
    }
    telemetry_artifact = publish_bytes_without_overwrite(
        Path(output_directory) / "telemetry_t2_v1.json",
        canonical_json_bytes(telemetry_value),
    )
    load_t2_study_report(scientific.path)
    return T2ReportArtifacts(scientific, telemetry_artifact)


__all__ = [
    "T2_MINIMUM_RELEVANT_EFFECT_M_S",
    "T2_PAIRED_ENDPOINT_ID",
    "T2_POLICY_SEEDS",
    "T2_REPORT_WRITER_ID",
    "T2ReportArtifacts",
    "arm_summary",
    "build_t2_study_report",
    "checkpoint_summary",
    "load_t2_study_report",
    "paired_checkpoint_effect",
    "publish_t2_study_report",
    "validate_t2_study_report",
]
