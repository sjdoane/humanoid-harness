"""T2 report admission.

Report hashes named files and replays traces; cross-manifest reconciliation is
enforced by the final-ready gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import stat
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
from oracle_composition.phase_b.contracts import (
    load_phase_b_oracle,
    load_starting_checkpoint,
    validate_training_design,
)
from oracle_composition.phase_b.protected_metrics import ERROR_NAMES, ERROR_SCALES
from oracle_composition.phase_b.report_v2 import (
    SCIENTIFIC_RECEIPT_SCHEMA_ID,
    TELEMETRY_SCHEMA_ID,
)

from .execution_manifest import validate_t2_execution_manifest
from .pairing import validate_pairing_receipt
from .study_manifest import (
    INTEGRATED_PAIRING_RECEIPT_BYTE_COUNT,
    INTEGRATED_PAIRING_RECEIPT_PATH,
    STUDY_CLAIM_CEILING,
    STUDY_EVIDENCE_CLASS,
    STUDY_FINAL_READY_STATUS,
    arm_common_fields,
    resolve_t2_reward_binding,
    validate_t2_study_manifest,
)
from .t2_evaluator import (
    T2_EVALUATION_SEEDS,
    T2_HORIZON_STEPS,
    T2_REPORT_SCHEMA_ID,
    T2EpisodeMetrics,
    _evaluate_t2_trace_against_reference,
    load_evaluator_design,
    load_t2_verified_reference,
)

T2_REPORT_WRITER_ID = "t2_reward_study_report_writer/v1"
T2_PAIRED_ENDPOINT_ID = "t2_paired_checkpoint_com_speed_mae/v1"
T2_TRACE_INDEX_SCHEMA_ID = "t2_protected_trace_index/v1"
T2_REWARD_DIAGNOSTICS_SCHEMA_ID = "t2_reward_diagnostics/v1"
T2_MINIMUM_RELEVANT_EFFECT_M_S = 0.25
T2_PAIRED_T_CRITICAL_95_DF4 = 2.7764451051977987
T2_POLICY_SEEDS = (121001, 121101, 121201, 121301, 121401)
T2_REPORT_ARTIFACT_INPUTS = frozenset(
    {
        "baseline_reward",
        "candidate_reward",
        "evaluator_design",
        "execution_manifest",
        "integrated_pairing_receipt",
        "library",
        "oracle",
        "pairing_adapter",
        "reference_corpus",
        "starting_checkpoint",
        "study_manifest",
        "trace_index",
        "training_design",
    }
)
T2_REPORT_REWARD_INPUTS = frozenset({"baseline_reward", "candidate_reward"})
T2_REPORT_COMMON_BINDINGS = {
    "evaluator_design": "evaluator",
    "execution_manifest": "execution_manifest",
    "library": "library",
    "oracle": "oracle",
    "pairing_adapter": "pairing_adapter",
    "reference_corpus": "reference_corpus",
    "starting_checkpoint": "starting_checkpoint",
    "training_design": "training_design",
}


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


def _artifact_binding(value: object, *, field: str) -> dict[str, object]:
    expected = {"byte_count", "path", "root", "sha256"}
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"T2 report {field} binding fields differ")
    if value["root"] not in {"repository", "run"}:
        raise ValueError(f"T2 report {field} root differs")
    if type(value["byte_count"]) is not int or value["byte_count"] <= 0:
        raise ValueError(f"T2 report {field} byte count differs")
    path = Path(value["path"]) if type(value["path"]) is str else Path("/")
    if (
        path.is_absolute()
        or not value["path"]
        or str(path) != value["path"]
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in value["path"]
    ):
        raise ValueError(f"T2 report {field} path is not canonical relative text")
    _sha256(value["sha256"], field=f"T2 report {field} SHA-256")
    return dict(value)


def _reward_artifact_binding(value: object, *, field: str) -> dict[str, object]:
    expected = {"byte_count", "path", "reward_id", "root", "sha256"}
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"T2 report {field} reward binding fields differ")
    checked = _artifact_binding(
        {key: value[key] for key in ("byte_count", "path", "root", "sha256")},
        field=field,
    )
    if type(value["reward_id"]) is not str or not value["reward_id"]:
        raise ValueError(f"T2 report {field} reward identity differs")
    return {**checked, "reward_id": value["reward_id"]}


def _validate_report_inputs(inputs: Mapping[str, object]) -> dict[str, object]:
    if type(inputs) is not dict or set(inputs) != {"artifacts", "study_pairing_sha256"}:
        raise ValueError("T2 report input identities differ")
    artifacts = inputs["artifacts"]
    if type(artifacts) is not dict or set(artifacts) != T2_REPORT_ARTIFACT_INPUTS:
        raise ValueError("T2 report artifact input identities differ")
    checked = {
        name: (
            _reward_artifact_binding(binding, field=name)
            if name in T2_REPORT_REWARD_INPUTS
            else _artifact_binding(binding, field=name)
        )
        for name, binding in artifacts.items()
    }
    return {
        "artifacts": checked,
        "study_pairing_sha256": _sha256(
            inputs["study_pairing_sha256"], field="study_pairing_sha256"
        ),
    }


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
    return {
        "checkpoint_sha256": ordered[0].checkpoint_sha256,
        "complete_episode_count": sum(item.complete_trace for item in ordered),
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
                "reference_lineage_sha256": item.reference_lineage_sha256,
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
        "difference_definition": ("baseline_mae_minus_candidate_mae_positive_favors_candidate"),
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
        "standard_error_m_s": standard_error,
        "task_improvement_rule": {
            "all_five_differences_positive": all_positive,
            "mean_reduction_at_least_0.25_m_s": (mean_difference >= T2_MINIMUM_RELEVANT_EFFECT_M_S),
            "passed": task_rule,
        },
    }


def _effect_from_arms(
    baseline: Mapping[str, object], candidate: Mapping[str, object]
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
            "standard_error_m_s": None,
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
) -> dict[str, object]:
    """Build scientific bytes containing no reward-output telemetry."""

    checked_inputs = _validate_report_inputs(inputs)
    artifacts = checked_inputs["artifacts"]
    report = {
        "claim_ceiling": STUDY_CLAIM_CEILING,
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
            "protected_metric_source": "direct_state_only",
            "reward_telemetry_in_scientific_receipt": False,
        },
        "evidence_class": STUDY_EVIDENCE_CLASS,
        "generated_utc": "recorded_in_separate_telemetry",
        "inputs": checked_inputs,
        "integrity": {
            "deterministic_scientific_receipt": True,
            "raw_traces_replayed_before_acceptance": True,
            "reward_diagnostics_location": "reward_diagnostics_t2_v1.json",
            "telemetry_location": "telemetry_t2_v1.json",
            "wall_clock_host_and_reward_telemetry_excluded": True,
        },
        "paired_endpoint_id": T2_PAIRED_ENDPOINT_ID,
        "report_schema_id": T2_REPORT_SCHEMA_ID,
        "report_writer_id": T2_REPORT_WRITER_ID,
        "schema_version": 1,
        "scientific_receipt_schema_id": SCIENTIFIC_RECEIPT_SCHEMA_ID,
        "summary": _summary(
            baseline_reward_sha256=artifacts["baseline_reward"]["sha256"],
            candidate_reward_sha256=artifacts["candidate_reward"]["sha256"],
            baseline_episodes=baseline_episodes,
            candidate_episodes=candidate_episodes,
        ),
    }
    _validate_t2_study_report_structure(report)
    canonical_json_bytes(report)
    return report


def _validate_t2_study_report_structure(
    value: Mapping[str, object],
) -> tuple[dict[str, object], list[T2EpisodeMetrics], list[T2EpisodeMetrics]]:
    expected = {
        "claim_ceiling",
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
    }
    if type(value) is not dict or set(value) != expected:
        raise ValueError("T2 report fields differ")
    if (
        value["schema_version"] != 1
        or value["report_schema_id"] != T2_REPORT_SCHEMA_ID
        or value["report_writer_id"] != T2_REPORT_WRITER_ID
        or value["paired_endpoint_id"] != T2_PAIRED_ENDPOINT_ID
        or value["scientific_receipt_schema_id"] != SCIENTIFIC_RECEIPT_SCHEMA_ID
        or value["generated_utc"] != "recorded_in_separate_telemetry"
        or value["evidence_class"] != STUDY_EVIDENCE_CLASS
        or value["claim_ceiling"] != STUDY_CLAIM_CEILING
    ):
        raise ValueError("T2 report identity or claim boundary differs")
    if value["integrity"] != {
        "deterministic_scientific_receipt": True,
        "raw_traces_replayed_before_acceptance": True,
        "reward_diagnostics_location": "reward_diagnostics_t2_v1.json",
        "telemetry_location": "telemetry_t2_v1.json",
        "wall_clock_host_and_reward_telemetry_excluded": True,
    }:
        raise ValueError("T2 report integrity boundary differs")
    evaluation = value["evaluation"]
    required_evaluation = {
        "calibration",
        "deterministic_actions",
        "evaluation_seeds",
        "expert_start",
        "full_sufficient_traces_required",
        "horizon_steps",
        "per_episode_by_arm",
        "protected_metric_source",
        "reward_telemetry_in_scientific_receipt",
    }
    if type(evaluation) is not dict or set(evaluation) != required_evaluation:
        raise ValueError("T2 report evaluation fields differ")
    if (
        evaluation["calibration"] != "none"
        or evaluation["deterministic_actions"] is not True
        or evaluation["evaluation_seeds"] != list(T2_EVALUATION_SEEDS)
        or evaluation["expert_start"] is not True
        or evaluation["full_sufficient_traces_required"] is not True
        or evaluation["horizon_steps"] != T2_HORIZON_STEPS
        or evaluation["protected_metric_source"] != "direct_state_only"
        or evaluation["reward_telemetry_in_scientific_receipt"] is not False
    ):
        raise ValueError("T2 report evaluation contract differs")
    per_arm = evaluation["per_episode_by_arm"]
    if type(per_arm) is not dict or set(per_arm) != {"baseline", "candidate"}:
        raise ValueError("T2 report arm episode fields differ")
    if type(per_arm["baseline"]) is not list or type(per_arm["candidate"]) is not list:
        raise ValueError("T2 report arm episodes must be arrays")
    baseline = [T2EpisodeMetrics.from_dict(item) for item in per_arm["baseline"]]
    candidate = [T2EpisodeMetrics.from_dict(item) for item in per_arm["candidate"]]
    inputs = _validate_report_inputs(value["inputs"])
    artifacts = inputs["artifacts"]
    expected_summary = _summary(
        baseline_reward_sha256=artifacts["baseline_reward"]["sha256"],
        candidate_reward_sha256=artifacts["candidate_reward"]["sha256"],
        baseline_episodes=baseline,
        candidate_episodes=candidate,
    )
    if value["summary"] != expected_summary:
        raise ValueError("T2 report summary differs from recomputed episode rows")
    canonical_json_bytes(dict(value))
    return inputs, baseline, candidate


def _regular_bytes(path: Path, *, field: str, maximum: int) -> bytes:
    candidate = Path(path)
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise ValueError(f"T2 report {field} is unavailable") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or not 0 < before.st_size <= maximum
    ):
        raise ValueError(f"T2 report {field} must be bounded regular data")
    encoded = candidate.read_bytes()
    after = candidate.lstat()
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in identity):
        raise ValueError(f"T2 report {field} changed while read")
    return encoded


def _resolve_binding(
    binding: Mapping[str, object],
    *,
    repository_root: Path,
    run_root: Path,
    field: str,
    maximum: int = 512 * 1024**2,
) -> tuple[Path, bytes]:
    checked_with_metadata = (
        _reward_artifact_binding(binding, field=field)
        if field in T2_REPORT_REWARD_INPUTS
        else _artifact_binding(binding, field=field)
    )
    checked = {key: checked_with_metadata[key] for key in ("byte_count", "path", "root", "sha256")}
    root = repository_root if checked["root"] == "repository" else run_root
    path = root / checked["path"]
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"T2 report {field} is unavailable") from exc
    if not resolved.is_relative_to(root) or resolved != path:
        raise ValueError(f"T2 report {field} path crosses a symlink or root boundary")
    encoded = _regular_bytes(path, field=field, maximum=maximum)
    if (
        len(encoded) != checked["byte_count"]
        or hashlib.sha256(encoded).hexdigest() != checked["sha256"]
    ):
        raise ValueError(f"T2 report {field} artifact identity differs")
    return path, encoded


def build_t2_trace_index(entries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    value = {
        "schema_version": 1,
        "trace_index_schema_id": T2_TRACE_INDEX_SCHEMA_ID,
        "traces": [dict(entry) for entry in entries],
    }
    validate_t2_trace_index(value)
    return value


def validate_t2_trace_index(value: Mapping[str, object]) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
        "schema_version",
        "trace_index_schema_id",
        "traces",
    }:
        raise ValueError("T2 trace index fields differ")
    if value["schema_version"] != 1 or value["trace_index_schema_id"] != T2_TRACE_INDEX_SCHEMA_ID:
        raise ValueError("T2 trace index identity differs")
    entries = value["traces"]
    if type(entries) is not list or len(entries) != 200:
        raise ValueError("T2 trace index must contain exactly 200 traces")
    expected_identities = {
        (arm, policy_seed, evaluation_seed)
        for arm in ("baseline", "candidate")
        for policy_seed in T2_POLICY_SEEDS
        for evaluation_seed in T2_EVALUATION_SEEDS
    }
    observed_identities = set()
    paths = set()
    for entry in entries:
        if type(entry) is not dict or set(entry) != {
            "arm_label",
            "byte_count",
            "evaluation_seed",
            "path",
            "policy_seed",
            "sha256",
        }:
            raise ValueError("T2 trace index entry fields differ")
        binding = _artifact_binding(
            {
                "byte_count": entry["byte_count"],
                "path": entry["path"],
                "root": "run",
                "sha256": entry["sha256"],
            },
            field="trace",
        )
        identity = (
            entry["arm_label"],
            entry["policy_seed"],
            entry["evaluation_seed"],
        )
        observed_identities.add(identity)
        paths.add(binding["path"])
    if observed_identities != expected_identities or len(paths) != 200:
        raise ValueError("T2 trace index coverage differs from the frozen grid")
    canonical_json_bytes(dict(value))
    return dict(value)


def _load_trace_index(encoded: bytes) -> dict[str, object]:
    try:
        value = json.loads(encoded)
    except (UnicodeError, ValueError) as exc:
        raise ValueError("T2 trace index is not JSON") from exc
    if type(value) is not dict or canonical_json_bytes(value) != encoded:
        raise ValueError("T2 trace index is not canonical JSON")
    return validate_t2_trace_index(value)


def _replay_reported_traces(
    *,
    trace_index: Mapping[str, object],
    report_episodes: Mapping[str, Sequence[T2EpisodeMetrics]],
    repository_root: Path,
    run_root: Path,
) -> None:
    reported = {
        (arm, item.policy_seed, item.evaluation_seed): item
        for arm, episodes in report_episodes.items()
        for item in episodes
    }
    if len(reported) != 200:
        raise ValueError("T2 report episode coverage differs from the frozen grid")
    reference_cache = {
        seed: load_t2_verified_reference(repository_root=repository_root, evaluation_seed=seed)
        for seed in T2_EVALUATION_SEEDS
    }
    replayed = set()
    for entry in trace_index["traces"]:
        identity = (
            entry["arm_label"],
            entry["policy_seed"],
            entry["evaluation_seed"],
        )
        if identity not in reported:
            raise ValueError("T2 trace index contains an unreported trace")
        path, encoded = _resolve_binding(
            {
                "byte_count": entry["byte_count"],
                "path": entry["path"],
                "root": "run",
                "sha256": entry["sha256"],
            },
            repository_root=repository_root,
            run_root=run_root,
            field="raw trace",
        )
        try:
            trace = json.loads(encoded)
        except (UnicodeError, ValueError) as exc:
            raise ValueError(f"T2 raw trace is not JSON: {path}") from exc
        if type(trace) is not dict or canonical_json_bytes(trace) != encoded:
            raise ValueError(f"T2 raw trace is not canonical JSON: {path}")
        metrics = _evaluate_t2_trace_against_reference(
            trace, reference=reference_cache[entry["evaluation_seed"]]
        )
        if (
            metrics.policy_seed != entry["policy_seed"]
            or metrics.evaluation_seed != entry["evaluation_seed"]
            or metrics.to_dict() != reported[identity].to_dict()
        ):
            raise ValueError("T2 reported episode metrics differ from raw-trace replay")
        replayed.add(identity)
    if replayed != set(reported):
        raise ValueError("T2 trace index does not cover exactly the reported traces")


def _canonical_json_object(encoded: bytes, *, field: str) -> dict[str, object]:
    try:
        value = json.loads(encoded)
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"T2 {field} is not JSON") from exc
    if type(value) is not dict or canonical_json_bytes(value) != encoded:
        raise ValueError(f"T2 {field} is not canonical JSON")
    return value


def _report_binding_without_root(binding: Mapping[str, object]) -> dict[str, object]:
    return {key: binding[key] for key in ("byte_count", "path", "sha256")}


def _report_reward_without_file_metadata(
    binding: Mapping[str, object],
) -> dict[str, object]:
    return {key: binding[key] for key in ("path", "reward_id", "sha256")}


def _reconcile_report_to_manifest(
    *,
    inputs: Mapping[str, object],
    study: Mapping[str, object],
) -> None:
    artifacts = inputs["artifacts"]
    common = arm_common_fields(study["arms"][0])
    for report_name, manifest_name in T2_REPORT_COMMON_BINDINGS.items():
        if _report_binding_without_root(artifacts[report_name]) != common[manifest_name]:
            raise ValueError(f"T2 report {report_name} binding differs from study manifest")
    for report_name, arm_index in (("baseline_reward", 0), ("candidate_reward", 1)):
        if (
            _report_reward_without_file_metadata(artifacts[report_name])
            != study["arms"][arm_index]["reward"]
        ):
            raise ValueError(f"T2 report {report_name} binding differs from study manifest")
    pairing = artifacts["integrated_pairing_receipt"]
    if (
        pairing["root"] != "repository"
        or pairing["path"] != INTEGRATED_PAIRING_RECEIPT_PATH
        or pairing["byte_count"] != INTEGRATED_PAIRING_RECEIPT_BYTE_COUNT
        or pairing["sha256"] != study["integrated_pairing_receipt_sha256"]
    ):
        raise ValueError("T2 report integrated pairing receipt differs from study manifest")
    if inputs["study_pairing_sha256"] != study["study_pairing_sha256"]:
        raise ValueError("T2 report and study pairing identities differ")


def _validate_final_ready_inputs(
    *,
    inputs: Mapping[str, object],
    resolved: Mapping[str, tuple[Path, bytes]],
    study: Mapping[str, object],
    repository_root: Path,
    run_root: Path,
) -> None:
    _reconcile_report_to_manifest(inputs=inputs, study=study)
    artifacts = inputs["artifacts"]

    pairing_receipt = _canonical_json_object(
        resolved["integrated_pairing_receipt"][1],
        field="integrated pairing receipt",
    )
    try:
        validate_pairing_receipt(pairing_receipt)
    except ValueError as exc:
        raise ValueError("T2 integrated pairing receipt failed validation") from exc

    training = _canonical_json_object(
        resolved["training_design"][1],
        field="training design",
    )
    execution = _canonical_json_object(
        resolved["execution_manifest"][1],
        field="execution manifest",
    )
    try:
        validate_training_design(training)
        load_phase_b_oracle(
            resolved["oracle"][0],
            available_behaviors=("expert", "medium", "simple"),
        )
        load_evaluator_design(
            resolved["evaluator_design"][0],
            evaluator_source_path=repository_root
            / "src/oracle_composition/reward_study/t2_evaluator.py",
            protected_metrics_source_path=repository_root
            / "src/oracle_composition/phase_b/protected_metrics.py",
            report_v2_source_path=repository_root / "src/oracle_composition/phase_b/report_v2.py",
            report_writer_source_path=repository_root
            / "src/oracle_composition/reward_study/t2_report.py",
        )
        validate_t2_execution_manifest(execution, repository_root=repository_root)
        starting_root = (
            repository_root
            if artifacts["starting_checkpoint"]["root"] == "repository"
            else run_root
        )
        load_starting_checkpoint(
            resolved["starting_checkpoint"][0],
            repository_root=starting_root,
        )
        resolve_t2_reward_binding(
            study["arms"][0]["reward"],
            artifact_root=(
                repository_root
                if artifacts["baseline_reward"]["root"] == "repository"
                else run_root
            ),
            candidate=False,
        )
        resolve_t2_reward_binding(
            study["arms"][1]["reward"],
            artifact_root=(
                repository_root
                if artifacts["candidate_reward"]["root"] == "repository"
                else run_root
            ),
            candidate=True,
        )
    except ValueError as exc:
        raise ValueError(f"T2 final-ready input contract is invalid: {exc}") from exc


def validate_t2_study_report(
    value: Mapping[str, object],
    *,
    repository_root: Path,
    run_root: Path,
) -> dict[str, object]:
    """Enforce the final-ready cross-manifest gate, then replay every raw trace."""

    repository = Path(repository_root).resolve(strict=True)
    run = Path(run_root).resolve(strict=True)
    raw_inputs = value.get("inputs") if type(value) is dict else None
    inputs = _validate_report_inputs(raw_inputs)
    study_path, study_bytes = _resolve_binding(
        inputs["artifacts"]["study_manifest"],
        repository_root=repository,
        run_root=run,
        field="study_manifest",
    )
    study = _canonical_json_object(study_bytes, field="bound study manifest")
    try:
        validated_study = validate_t2_study_manifest(study)
    except ValueError as exc:
        raise ValueError("T2 bound study manifest contract differs") from exc
    if validated_study["status"] != STUDY_FINAL_READY_STATUS:
        raise ValueError("study_not_final_ready")

    structured_inputs, baseline, candidate = _validate_t2_study_report_structure(value)
    if structured_inputs != inputs:
        raise ValueError("T2 report inputs changed during final-ready admission")
    resolved = {}
    for name, binding in inputs["artifacts"].items():
        resolved[name] = (
            (study_path, study_bytes)
            if name == "study_manifest"
            else _resolve_binding(
                binding,
                repository_root=repository,
                run_root=run,
                field=name,
                maximum=512 * 1024 if name == "trace_index" else 512 * 1024**2,
            )
        )
    _validate_final_ready_inputs(
        inputs=inputs,
        resolved=resolved,
        study=validated_study,
        repository_root=repository,
        run_root=run,
    )
    trace_index = _load_trace_index(resolved["trace_index"][1])
    _replay_reported_traces(
        trace_index=trace_index,
        report_episodes={"baseline": baseline, "candidate": candidate},
        repository_root=repository,
        run_root=run,
    )
    return dict(value)


def load_t2_study_report(path: Path, *, repository_root: Path, run_root: Path) -> dict[str, object]:
    encoded = _regular_bytes(path, field="scientific receipt", maximum=512 * 1024**2)
    try:
        value = json.loads(encoded)
    except (UnicodeError, ValueError) as exc:
        raise ValueError("T2 report is not JSON") from exc
    if type(value) is not dict or canonical_json_bytes(value) != encoded:
        raise ValueError("T2 report is not canonical JSON")
    return validate_t2_study_report(value, repository_root=repository_root, run_root=run_root)


def _reward_diagnostics_value(
    *, scientific_receipt_sha256: str, reward_telemetry: object | None
) -> dict[str, object]:
    status = "missing"
    telemetry = None
    if reward_telemetry is not None:
        if type(reward_telemetry) is dict:
            try:
                canonical_json_bytes(reward_telemetry)
            except (TypeError, ValueError, OverflowError):
                status = "malformed"
            else:
                status = "available"
                telemetry = dict(reward_telemetry)
        else:
            status = "malformed"
    return {
        "reward_diagnostics_schema_id": T2_REWARD_DIAGNOSTICS_SCHEMA_ID,
        "schema_version": 1,
        "scientific_receipt_sha256": _sha256(
            scientific_receipt_sha256, field="scientific_receipt_sha256"
        ),
        "status": status,
        "telemetry": telemetry,
    }


@dataclass(frozen=True, slots=True)
class T2ReportArtifacts:
    scientific_receipt: PublishedArtifact
    reward_diagnostics: PublishedArtifact
    telemetry: PublishedArtifact


def publish_t2_study_report(
    *,
    output_directory: Path,
    report: Mapping[str, object],
    telemetry: Mapping[str, object],
    reward_telemetry: object | None,
    repository_root: Path,
    run_root: Path,
) -> T2ReportArtifacts:
    """Publish verified science, separately keyed reward diagnostics, and telemetry."""

    validated = validate_t2_study_report(report, repository_root=repository_root, run_root=run_root)
    scientific = publish_bytes_without_overwrite(
        Path(output_directory) / "scientific_receipt_t2_v1.json",
        canonical_json_bytes(validated),
    )
    reward_diagnostics = publish_bytes_without_overwrite(
        Path(output_directory) / "reward_diagnostics_t2_v1.json",
        canonical_json_bytes(
            _reward_diagnostics_value(
                scientific_receipt_sha256=scientific.sha256,
                reward_telemetry=reward_telemetry,
            )
        ),
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
    load_t2_study_report(scientific.path, repository_root=repository_root, run_root=run_root)
    return T2ReportArtifacts(scientific, reward_diagnostics, telemetry_artifact)


__all__ = [
    "T2_MINIMUM_RELEVANT_EFFECT_M_S",
    "T2_PAIRED_ENDPOINT_ID",
    "T2_POLICY_SEEDS",
    "T2_REPORT_ARTIFACT_INPUTS",
    "T2_REPORT_WRITER_ID",
    "T2_REWARD_DIAGNOSTICS_SCHEMA_ID",
    "T2_TRACE_INDEX_SCHEMA_ID",
    "T2ReportArtifacts",
    "arm_summary",
    "build_t2_study_report",
    "build_t2_trace_index",
    "checkpoint_summary",
    "load_t2_study_report",
    "paired_checkpoint_effect",
    "publish_t2_study_report",
    "validate_t2_study_report",
    "validate_t2_trace_index",
]
