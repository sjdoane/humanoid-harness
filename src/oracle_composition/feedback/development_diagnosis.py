"""Route validated development measurements without authorizing a candidate."""

from __future__ import annotations

from pathlib import Path
from statistics import median

from oracle_composition.development.reference_ablation import (
    CLAIM_CEILING,
    DEVELOPMENT_BLOCKS,
    DevelopmentAblationError,
)
from oracle_composition.experiments.reference_input_transforms import CONDITION_IDS
from oracle_composition.phase_b.protected_metrics import ERROR_NAMES
from oracle_composition.phase_b.report_v2 import TRACKING_SCALES

from .development import ValidatedDevelopmentEvidence, validate_development_result
from .evidence import (
    FeedbackDiagnosis,
    FeedbackEvidenceError,
    ObservedFact,
    SteeringInput,
    load_steering,
)


def _condition_summary(
    evidence: ValidatedDevelopmentEvidence,
) -> tuple[
    dict[str, int],
    dict[str, int],
    dict[str, dict[str, float]],
    dict[str, float],
    dict[str, float],
]:
    safe_counts: dict[str, int] = {}
    tracking_counts: dict[str, int] = {}
    segment_medians: dict[str, dict[str, float]] = {}
    transition_medians: dict[str, float] = {}
    settled_medians: dict[str, float] = {}
    for condition in CONDITION_IDS:
        rows = [row for row in evidence.arms if row["condition_id"] == condition]
        safe_counts[condition] = sum(
            row["observed_steps"] == 1_000
            and row["fall"] is False
            and row["forbidden_contact_count"] == 0
            and row["action_bounds_ok"] is True
            for row in rows
        )
        tracking_counts[condition] = sum(
            all(
                float(row["six_tracking_errors"][name]) <= TRACKING_SCALES[name]
                for name in ERROR_NAMES
            )
            for row in rows
        )
        segment_medians[condition] = {
            segment: median(float(row["segment_errors"][segment]) for row in rows)
            for segment in ("fast", "slow", "return_fast")
        }
        transition_medians[condition] = median(
            float(row["transition_window_error"]) for row in rows
        )
        settled_medians[condition] = median(
            float(row["settled_state_normalized_error"]) for row in rows
        )
    return (
        safe_counts,
        tracking_counts,
        segment_medians,
        transition_medians,
        settled_medians,
    )


def _matched_summaries(
    evidence: ValidatedDevelopmentEvidence,
) -> tuple[dict[str, int], dict[str, int], dict[str, float]]:
    matched = evidence.result["matched_state_action_deltas"]
    changed = {
        condition: sum(
            int(matched[str(block)][condition]["bitwise_changed_steps"])
            for block in DEVELOPMENT_BLOCKS
        )
        for condition in CONDITION_IDS
    }
    evaluated = {
        condition: sum(
            int(matched[str(block)][condition]["evaluated_steps"]) for block in DEVELOPMENT_BLOCKS
        )
        for condition in CONDITION_IDS
    }
    maximum = {
        condition: max(
            float(matched[str(block)][condition]["maximum_max_abs_physical_action"])
            for block in DEVELOPMENT_BLOCKS
        )
        for condition in CONDITION_IDS
    }
    return changed, evaluated, maximum


def _route(
    *,
    safe: dict[str, int],
    tracking: dict[str, int],
    changed: dict[str, int],
    transition_concentrated: int,
) -> tuple[str, str, str, tuple[str, ...], str, str]:
    changed_interventions = sum(changed[name] > 0 for name in CONDITION_IDS[1:])
    if changed_interventions == 0:
        return (
            "adapter_repair",
            "high",
            "The validated actor produced no recorded action change under any non-exact reference intervention at matched observed states.",
            (
                "The actor may use the reference yet map these particular interventions to identical actions.",
            ),
            "A repaired reference-conditioning path should change at least one matched-state action.",
            "The bound actor remains unchanged after reference installation is independently verified.",
        )
    exact = CONDITION_IDS[0]
    if safe[exact] < len(DEVELOPMENT_BLOCKS) or tracking[exact] < len(DEVELOPMENT_BLOCKS):
        return (
            "adapter_repair",
            "moderate",
            "The exact-reference arm does not pass existing safety and basic tracking boundaries on every development block.",
            (
                "The smoke-trained actor may lack tracking capacity despite correct reference installation.",
                "The admitted reference may be dynamically infeasible for this policy family.",
            ),
            "Adapter repair should restore exact-reference safety and tracking before oracle changes are compared.",
            "The deficits persist under a separately verified competent tracker.",
        )
    if transition_concentrated:
        return (
            "oracle",
            "low",
            "Exact-reference tracking is safe, and transition-window normalized error exceeds later settled-window error on at least one development block.",
            (
                "The frozen actor, not authored oracle logic, may limit transition tracking.",
                "The contrast may be block-specific in-sample variation.",
            ),
            "A bounded transition change should reduce local error without weakening settled tracking.",
            "A matched development comparison leaves transition error unchanged or degrades settled tracking.",
        )
    return (
        "measurement",
        "moderate",
        "The ablation establishes reference-input sensitivity and exact-arm tracking but does not isolate a transition-local oracle defect.",
        ("No oracle patch may be warranted for this development case.",),
        "One preregistered transition stressor should separate phase transfer from tracker capacity.",
        "The stressor preserves the same non-localized pattern.",
    )


def diagnose_development_feedback(
    *,
    development_result_path: Path,
    smoke_run: Path,
    corpus_root: Path,
    protocol_path: Path,
    steering_path: Path | None = None,
) -> FeedbackDiagnosis:
    """Turn validated in-sample evidence into one non-authorizing route."""

    steering: SteeringInput | None = load_steering(steering_path)
    try:
        evidence = validate_development_result(
            development_result_path,
            smoke_run=smoke_run,
            corpus_root=corpus_root,
            protocol_path=protocol_path,
        )
    except FeedbackEvidenceError:
        raise
    except (DevelopmentAblationError, OSError, TypeError, ValueError) as exc:
        raise FeedbackEvidenceError(f"development evidence failed validation: {exc}") from exc
    safe, tracking, segments, transition, settled = _condition_summary(evidence)
    changed, evaluated, maximum = _matched_summaries(evidence)
    exact_rows = [row for row in evidence.arms if row["condition_id"] == CONDITION_IDS[0]]
    transition_concentrated = sum(
        float(row["transition_window_error"]) > float(row["settled_state_normalized_error"])
        for row in exact_rows
    )
    action, confidence, hypothesis, rivals, prediction, falsifier = _route(
        safe=safe,
        tracking=tracking,
        changed=changed,
        transition_concentrated=transition_concentrated,
    )
    source = evidence.trace_set_sha256
    facts = (
        ObservedFact(
            "safe_block_count_by_reference_condition",
            "development.in_sample.objective_recomputed",
            safe,
            source,
            candidate_visible=True,
        ),
        ObservedFact(
            "basic_tracking_pass_block_count_by_reference_condition",
            "development.in_sample.existing_tracking_boundaries",
            tracking,
            source,
            candidate_visible=True,
        ),
        ObservedFact(
            "median_segment_speed_error_m_s_by_reference_condition",
            "development.in_sample.objective_recomputed",
            segments,
            source,
            candidate_visible=True,
        ),
        ObservedFact(
            "median_transition_window_normalized_error_by_reference_condition",
            "development.in_sample.objective_recomputed",
            transition,
            source,
            candidate_visible=True,
        ),
        ObservedFact(
            "median_settled_window_normalized_error_by_reference_condition",
            "development.in_sample.objective_recomputed",
            settled,
            source,
            candidate_visible=True,
        ),
        ObservedFact(
            "recorded_matched_action_changed_steps_by_reference_condition",
            "development.in_sample.aggregate_recomputed_from_identity_bound_delta_rows",
            changed,
            source,
            candidate_visible=True,
        ),
        ObservedFact(
            "recorded_matched_action_evaluated_steps_by_reference_condition",
            "development.in_sample.aggregate_recomputed_from_identity_bound_delta_rows",
            evaluated,
            source,
            candidate_visible=True,
        ),
        ObservedFact(
            "recorded_max_abs_matched_action_delta_by_reference_condition",
            "development.in_sample.aggregate_recomputed_from_identity_bound_delta_rows",
            maximum,
            source,
            candidate_visible=True,
        ),
        ObservedFact(
            "exact_arm_transition_window_above_settled_block_count",
            "development.in_sample.objective_recomputed",
            transition_concentrated,
            source,
            candidate_visible=True,
        ),
    )
    manifest = evidence.manifest
    smoke_lineage = manifest["smoke_lineage"]
    protocol = manifest["protocol"]
    corpus_manifest = manifest["corpus_bindings"]["corpus_manifest_v2.json"]
    return FeedbackDiagnosis(
        action_surface=action,
        confidence=confidence,
        hypothesis=hypothesis,
        rivals=rivals,
        prediction=prediction,
        falsifier=falsifier,
        claim_ceilings=(CLAIM_CEILING,),
        source_identities={
            "development_manifest_sha256": evidence.manifest_sha256,
            "development_result_sha256": evidence.result_sha256,
            "development_trace_set_sha256": evidence.trace_set_sha256,
        },
        parent_identities={
            "actor_export_sha256": str(evidence.result["actor_export_sha256"]),
            "corpus_manifest_sha256": str(corpus_manifest["sha256"]),
            "development_protocol_sha256": str(protocol["sha256"]),
            "smoke_execution_manifest_sha256": str(smoke_lineage["execution_manifest_sha256"]),
        },
        observed_facts=facts,
        missing_evidence=(
            "reviewed_candidate_protocol_binding_this_diagnosis_to_one_oracle_delta",
            "raw_matched_action_pairs_for_independent_per_row_delta_recomputation",
            "raw_policy_observations_for_independent_actor_input_recomputation",
            "held_out_reference_ablation_after_candidate_lock",
            "calibrated_task_success_endpoint",
        ),
        steering=steering,
        proposal_ready=False,
        proposal_readiness_reason=(
            "The development measurements are candidate-safe, but no reviewed candidate "
            "protocol binds this diagnosis to one permitted oracle delta."
        ),
    )


__all__ = ["diagnose_development_feedback"]
