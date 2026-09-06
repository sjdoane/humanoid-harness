"""Validated evidence adapters and deterministic feedback routing.

The adapters in this module consume existing receipt authorities.  They do not
create a second evaluator, inspect held-out traces directly, or turn an
execution failure into scientific evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.harness.evidence import (
    ValidatedScientificReceipt,
    load_e003_execution,
    validate_scientific_receipt,
)
from oracle_composition.phase_b.contracts import REPORT_V2_SCHEMA_ID

MAX_FEEDBACK_EVIDENCE_BYTES = 512 * 1024**2
MAX_STEERING_BYTES = 16_384
MAX_STEERING_CHARACTERS = 4_000
SHA256_PATTERN_LENGTH = 64
SEED_FAILURE_RECEIPT_ID = "humanoid_phase_b_seed_failure/v2"
SEED_FAILURE_STATUSES = frozenset(
    {
        "action_bound_violation",
        "cleanup_failure",
        "counter_drift",
        "crash",
        "likelihood_failure",
        "malformed_frame",
        "non_finite",
        "not_started_due_to_cohort_failure",
        "phase_selection_failure",
        "precondition_failure",
        "resource_breach",
        "source_mutation",
        "timeout",
    }
)


class FeedbackEvidenceError(ValueError):
    """A feedback input does not satisfy an existing evidence contract."""


@dataclass(frozen=True, slots=True)
class ObservedFact:
    """One validated fact with an explicit candidate-context visibility boundary."""

    name: str
    scope: str
    value: object
    source_sha256: str
    candidate_visible: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "scope": self.scope,
            "source_sha256": self.source_sha256,
            "value": self.value,
            "visibility": (
                "candidate_context_development_only"
                if self.candidate_visible
                else "diagnosis_only_excluded_from_candidate_prompt"
            ),
        }


@dataclass(frozen=True, slots=True)
class SteeringInput:
    """Content-addressed user input; it is explicitly not a measurement."""

    text: str
    sha256: str
    byte_count: int

    def to_dict(self, *, include_text: bool) -> dict[str, object]:
        result: dict[str, object] = {
            "byte_count": self.byte_count,
            "epistemic_role": "user_steering_not_measurement",
            "sha256": self.sha256,
        }
        if include_text:
            result["text"] = self.text
        return result


@dataclass(frozen=True, slots=True)
class FeedbackDiagnosis:
    """Machine-readable decision output for a later persistent coordinator."""

    action_surface: str
    confidence: str
    hypothesis: str
    rivals: tuple[str, ...]
    prediction: str
    falsifier: str
    claim_ceilings: tuple[str, ...]
    source_identities: Mapping[str, str]
    parent_identities: Mapping[str, str]
    observed_facts: tuple[ObservedFact, ...]
    missing_evidence: tuple[str, ...]
    steering: SteeringInput | None
    proposal_ready: bool
    proposal_readiness_reason: str

    @property
    def candidate_kind(self) -> str:
        if self.action_surface == "oracle":
            return "oracle_json"
        if self.action_surface == "task_reward":
            return "target_speed_formula_recipe"
        return "none"

    def to_dict(self) -> dict[str, object]:
        return {
            "action_surface": self.action_surface,
            "candidate_kind": self.candidate_kind,
            "causal_status": (
                "bounded_hypothesis_not_causal_fact"
                if self.action_surface in {"oracle", "task_reward"}
                else "no_research_cause_assigned"
            ),
            "claim_ceilings": list(self.claim_ceilings),
            "confidence": self.confidence,
            "falsifier": self.falsifier,
            "hypothesis": self.hypothesis,
            "kind": "humanoid_feedback_diagnosis",
            "missing_evidence": list(self.missing_evidence),
            "observed_facts": [fact.to_dict() for fact in self.observed_facts],
            "parent_identities": dict(sorted(self.parent_identities.items())),
            "prediction": self.prediction,
            "readiness": {
                "diagnosis_ready": True,
                "evaluation_ready": False,
                "proposal_ready": self.proposal_ready,
                "revision_ready": False,
                "scientifically_informed_patch": self.proposal_ready,
                "training_ready": False,
            },
            "proposal_readiness_reason": self.proposal_readiness_reason,
            "protected_results_in_candidate_prompt": False,
            "rivals": list(self.rivals),
            "schema_version": 1,
            "source_identities": dict(sorted(self.source_identities.items())),
            "steering": None if self.steering is None else self.steering.to_dict(include_text=True),
        }


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == SHA256_PATTERN_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_canonical_object(
    path: Path, *, maximum: int, label: str
) -> tuple[dict[str, object], bytes]:
    def reject_constant(token: str) -> None:
        raise FeedbackEvidenceError(f"{label} contains non-finite JSON constant {token}")

    candidate = Path(path)
    try:
        state = candidate.stat(follow_symlinks=False)
    except OSError as exc:
        raise FeedbackEvidenceError(f"cannot inspect {label}: {exc}") from exc
    if (
        candidate.is_symlink()
        or not stat.S_ISREG(state.st_mode)
        or state.st_nlink != 1
        or not 1 <= state.st_size <= maximum
    ):
        raise FeedbackEvidenceError(f"{label} must be one bounded regular file")
    try:
        encoded = candidate.read_bytes()
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise FeedbackEvidenceError(f"{label} is not strict JSON: {exc}") from exc
    if type(value) is not dict or canonical_json_bytes(value) != encoded:
        raise FeedbackEvidenceError(f"{label} must be one canonical JSON object")
    return value, encoded


def load_steering(path: Path | None) -> SteeringInput | None:
    """Load exact user-authored bytes without treating their contents as evidence."""

    if path is None:
        return None
    candidate = Path(path)
    try:
        state = candidate.stat(follow_symlinks=False)
    except OSError as exc:
        raise FeedbackEvidenceError(f"cannot read steering input: {exc}") from exc
    if (
        candidate.is_symlink()
        or not stat.S_ISREG(state.st_mode)
        or state.st_nlink != 1
        or not 1 <= state.st_size <= MAX_STEERING_BYTES
    ):
        raise FeedbackEvidenceError("steering input must be one bounded regular file")
    try:
        with candidate.open("rb") as stream:
            encoded = stream.read(MAX_STEERING_BYTES + 1)
    except OSError as exc:
        raise FeedbackEvidenceError(f"cannot read steering input: {exc}") from exc
    if not 1 <= len(encoded) <= MAX_STEERING_BYTES:
        raise FeedbackEvidenceError("steering input exceeds its byte bound")
    try:
        text = encoded.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise FeedbackEvidenceError("steering input must be UTF-8") from exc
    if "\x00" in text or not text.strip() or len(text) > MAX_STEERING_CHARACTERS:
        raise FeedbackEvidenceError("steering input is empty, contains NUL, or exceeds its bound")
    return SteeringInput(text=text, sha256=_sha256(encoded), byte_count=len(encoded))


def validate_phase_a_receipt(
    *,
    receipt_path: Path,
    experiment: Path,
    repository_root: Path,
    expected_cycle: int,
) -> ValidatedScientificReceipt:
    """Validate the retained Phase A receipt and its full prior/trace chain."""

    sealed = load_e003_execution(Path(experiment), Path(repository_root))
    return validate_scientific_receipt(
        Path(receipt_path),
        experiment=Path(experiment),
        repository_root=Path(repository_root),
        library=sealed.library,
        task=sealed.task,
        expected_cycle=expected_cycle,
        expected_metric_core_sha256=None,
        verify_traces=True,
    )


def validate_seed_failure_receipt(path: Path) -> tuple[dict[str, object], str]:
    """Validate the exact current Phase B seed-failure receipt core."""

    value, encoded = _read_canonical_object(
        path, maximum=256 * 1024, label="Phase B seed failure receipt"
    )
    required = {
        "cleanup_outcome",
        "cleanup_succeeded",
        "evidence_class",
        "execution_manifest_sha256",
        "failure_receipt_id",
        "last_acknowledged_stage",
        "outcome",
        "planned_transitions",
        "ppo_seed",
        "primary_failure",
        "reason",
        "resource_controls",
        "schema_version",
        "smoke",
        "status",
        "success_receipt_present",
    }
    primary = value.get("primary_failure")
    if (
        set(value) != required
        or value.get("schema_version") != 2
        or value.get("failure_receipt_id") != SEED_FAILURE_RECEIPT_ID
        or value.get("outcome") != "failure"
        or value.get("success_receipt_present") is not False
        or value.get("status") not in SEED_FAILURE_STATUSES
        or not _is_sha256(value.get("execution_manifest_sha256"))
        or type(value.get("ppo_seed")) is not int
        or type(value.get("planned_transitions")) is not int
        or int(value["planned_transitions"]) <= 0
        or type(value.get("reason")) is not str
        or not value["reason"]
        or len(value["reason"]) > 1_000
        or type(value.get("last_acknowledged_stage")) is not str
        or not value["last_acknowledged_stage"]
        or type(value.get("cleanup_succeeded")) is not bool
        or type(value.get("cleanup_outcome")) is not dict
        or type(value.get("resource_controls")) is not dict
        or type(value.get("smoke")) is not bool
        or type(value.get("evidence_class")) is not str
        or not value["evidence_class"]
        or type(primary) is not dict
        or set(primary) != {"reason", "status"}
        or primary.get("reason") != value.get("reason")
        or primary.get("status") != value.get("status")
    ):
        raise FeedbackEvidenceError("Phase B seed failure receipt differs from its exact schema")
    return value, _sha256(encoded)


def load_phase_b_evidence(path: Path) -> tuple[str, dict[str, object], str]:
    """Detect then validate one supported current Phase B evidence artifact."""

    untrusted, encoded = _read_canonical_object(
        path, maximum=MAX_FEEDBACK_EVIDENCE_BYTES, label="Phase B evidence"
    )
    if untrusted.get("report_schema_id") == REPORT_V2_SCHEMA_ID:
        # Kept lazy so Phase A diagnosis does not import the optional Phase B
        # training stack.  This is the current report's canonical/recomputation
        # validator, not a generic JSON admission.
        from oracle_composition.phase_b.report_v2 import load_report_v2

        validated = load_report_v2(Path(path))
        return "validated_phase_b_report_v2", validated, _sha256(encoded)
    if untrusted.get("failure_receipt_id") == SEED_FAILURE_RECEIPT_ID:
        validated, digest = validate_seed_failure_receipt(Path(path))
        return "structurally_validated_phase_b_seed_failure_v2", validated, digest
    raise FeedbackEvidenceError(
        "Phase B evidence is not a supported report or seed failure receipt"
    )


def _phase_a_facts(receipt: ValidatedScientificReceipt) -> tuple[tuple[ObservedFact, ...], bool]:
    value = receipt.value
    current = receipt.arms[-1].value
    oracle_id = current["oracle_id"]
    rows = [row for row in value["per_episode"] if row["oracle_id"] == oracle_id]
    slow_simple = [float(row["slow_third_behavior_fractions"].get("simple", 0.0)) for row in rows]
    slow_absent = bool(rows) and all(math.isclose(item, 0.0, abs_tol=0.0) for item in slow_simple)
    facts = (
        ObservedFact("episode_count", f"phase_a.oracle:{oracle_id}", len(rows), receipt.sha256),
        ObservedFact(
            "fall_count",
            f"phase_a.oracle:{oracle_id}",
            sum(bool(row["fall"]) for row in rows),
            receipt.sha256,
        ),
        ObservedFact(
            "median_mean_absolute_speed_error_m_s",
            f"phase_a.oracle:{oracle_id}",
            float(current["median_mean_absolute_speed_error_m_s"]),
            receipt.sha256,
        ),
        ObservedFact(
            "median_switch_count",
            f"phase_a.oracle:{oracle_id}",
            float(current["median_switch_count"]),
            receipt.sha256,
        ),
        ObservedFact(
            "slow_segment_simple_behavior_absent_all_episodes",
            f"phase_a.oracle:{oracle_id}",
            slow_absent,
            receipt.sha256,
        ),
    )
    return facts, slow_absent


def _phase_b_report_facts(
    report: Mapping[str, object], source_sha256: str
) -> tuple[tuple[ObservedFact, ...], tuple[dict[str, object], ...]]:
    episodes = tuple(dict(row) for row in report["per_episode"])

    def safety_passed(row: Mapping[str, object]) -> bool:
        return (
            row.get("observed_steps") == 1_000
            and row.get("fall") is False
            and row.get("contacts") == []
            and row.get("action_bounds_ok") is True
        )

    def resynchronization_passed(row: Mapping[str, object]) -> bool:
        switches = row.get("switch_records")
        records = row.get("resynchronization_records")
        if row.get("cell") != "fixed_round_trip":
            return switches == [] and records == []
        if type(switches) is not list or type(records) is not list:
            return False
        return (
            len(switches) == 2
            and len(records) == 2
            and all(
                type(item) is dict
                and item.get("eight_consecutive_boundaries_at_or_below_one") is True
                and type(item.get("settle_latency_steps")) is int
                and 0 <= int(item["settle_latency_steps"]) <= 64
                for item in records
            )
        )

    facts = (
        ObservedFact("episode_count", "phase_b.protected_evaluation", len(episodes), source_sha256),
        ObservedFact(
            "safety_pass_count",
            "phase_b.protected_evaluation",
            sum(safety_passed(row) for row in episodes),
            source_sha256,
        ),
        ObservedFact(
            "utility_pass_count",
            "phase_b.protected_evaluation",
            sum(row["utility_passed"] is True for row in episodes),
            source_sha256,
        ),
        ObservedFact(
            "resynchronization_pass_count",
            "phase_b.protected_evaluation",
            sum(resynchronization_passed(row) for row in episodes),
            source_sha256,
        ),
        ObservedFact(
            "known_task_success_count",
            "phase_b.protected_evaluation",
            sum(row["task_success"] is True for row in episodes),
            source_sha256,
        ),
        ObservedFact(
            "known_task_failure_count",
            "phase_b.protected_evaluation",
            sum(row["task_success"] is False for row in episodes),
            source_sha256,
        ),
        ObservedFact(
            "median_slow_segment_error_m_s",
            "phase_b.protected_evaluation",
            median(float(row["segment_errors"]["slow"]) for row in episodes) if episodes else None,
            source_sha256,
        ),
    )
    return facts, episodes


def _phase_a_route(
    receipt: ValidatedScientificReceipt,
    facts: tuple[ObservedFact, ...],
    slow_absent: bool,
    steering: SteeringInput | None,
) -> FeedbackDiagnosis:
    current = receipt.arms[-1].value
    falls = int(current["fall_count"])
    missing = (
        "independent_reference_conditioned_policy_evidence",
        "protected_phase_b_task_and_tracking_evaluation",
        "causal_oracle_ablation",
    )
    if falls:
        action = "measurement"
        hypothesis = "The exploratory controller-switching arm fell, but the retained evidence does not isolate oracle logic from controller capacity or switching dynamics."
        rivals = (
            "The fixed controller library may be unable to execute the requested transition.",
            "The switch itself may destabilize the plant independently of the oracle guard.",
        )
        prediction = (
            "A protected matched evaluation should localize failures relative to switch boundaries."
        )
        falsifier = "The matched evaluation remains safe and shows no switch-localized degradation."
    elif slow_absent:
        action = "oracle"
        hypothesis = "The current exploratory controller-switching program did not express the declared slow subtask in the middle segment."
        rivals = (
            "The frozen controller library may not contain a policy that realizes the requested slow behavior.",
            "Controller switching is only a stand-in and may not predict reference-conditioned tracking.",
        )
        prediction = "A revised composition should create nonzero intended slow-behavior occupancy without increasing falls."
        falsifier = "The revised program changes guards but leaves behavior occupancy and the independent trajectory unchanged."
    else:
        action = "measurement"
        hypothesis = "Phase A is an exploratory interface result without enough evidence to select a research knob."
        rivals = (
            "Oracle, reward, controller, and reference limitations remain observationally coupled.",
        )
        prediction = "Protected Phase B evidence should separate transition, tracking, and task failure surfaces."
        falsifier = "The protected evidence remains incomplete or preserves the same ambiguity."
    return FeedbackDiagnosis(
        action_surface=action,
        confidence="low",
        hypothesis=hypothesis,
        rivals=rivals,
        prediction=prediction,
        falsifier=falsifier,
        claim_ceilings=(str(receipt.value["claim_ceiling"]),),
        source_identities={"phase_a_scientific_receipt_sha256": receipt.sha256},
        parent_identities={
            "oracle_sha256": str(current["oracle_sha256"]),
            "task_spec_sha256": str(receipt.value["task_spec_sha256"]),
        },
        observed_facts=facts,
        missing_evidence=missing,
        steering=steering,
        proposal_ready=False,
        proposal_readiness_reason=(
            "Phase A may suggest an oracle hypothesis but cannot authorize a scientifically informed patch."
        ),
    )


def _phase_b_failure_route(
    phase_a: ValidatedScientificReceipt,
    phase_a_facts: tuple[ObservedFact, ...],
    receipt: Mapping[str, object],
    digest: str,
    steering: SteeringInput | None,
) -> FeedbackDiagnosis:
    facts = (
        *phase_a_facts,
        ObservedFact("status", "phase_b.structural_failure_report", receipt["status"], digest),
        ObservedFact(
            "last_acknowledged_stage",
            "phase_b.structural_failure_report",
            receipt["last_acknowledged_stage"],
            digest,
        ),
        ObservedFact("reason", "phase_b.structural_failure_report", receipt["reason"], digest),
        ObservedFact(
            "claimed_execution_manifest_sha256",
            "phase_b.structural_failure_report",
            receipt["execution_manifest_sha256"],
            digest,
        ),
    )
    current = phase_a.arms[-1].value
    return FeedbackDiagnosis(
        action_surface="adapter_repair",
        confidence="high",
        hypothesis="A structurally valid local report declares an execution or admission failure before protected behavioral evidence; its runtime parent chain has not been reverified here.",
        rivals=("The declared failure may not bind to the intended manifest or job parent chain.",),
        prediction="Repairing the recorded execution stage should allow the same frozen request to reach a terminal scientific receipt.",
        falsifier="The identical failure recurs after the receipt-named execution condition is repaired.",
        claim_ceilings=(str(phase_a.value["claim_ceiling"]),),
        source_identities={
            "phase_a_scientific_receipt_sha256": phase_a.sha256,
            "phase_b_structural_failure_report_sha256": digest,
        },
        parent_identities={
            "oracle_sha256": str(current["oracle_sha256"]),
            "task_spec_sha256": str(phase_a.value["task_spec_sha256"]),
        },
        observed_facts=facts,
        missing_evidence=("protected_phase_b_terminal_evaluation",),
        steering=steering,
        proposal_ready=False,
        proposal_readiness_reason="A structurally validated failure report routes adapter inspection but cannot establish runtime fact or authorize a research patch.",
    )


def _phase_b_report_route(
    phase_a: ValidatedScientificReceipt,
    phase_a_facts: tuple[ObservedFact, ...],
    report: Mapping[str, object],
    digest: str,
    steering: SteeringInput | None,
) -> FeedbackDiagnosis:
    report_facts, episodes = _phase_b_report_facts(report, digest)
    facts = (*phase_a_facts, *report_facts)
    explicit_missing = tuple(str(item) for item in report["integrity"]["explicit_missing_fields"])
    task_unknown = bool(episodes) and all(row["task_success"] is None for row in episodes)
    missing = list(explicit_missing)
    if task_unknown:
        missing.append("calibrated_task_success_endpoint")

    def safety_passed(row: Mapping[str, object]) -> bool:
        return (
            row.get("observed_steps") == 1_000
            and row.get("fall") is False
            and row.get("contacts") == []
            and row.get("action_bounds_ok") is True
        )

    def resynchronization_passed(row: Mapping[str, object]) -> bool:
        switches = row.get("switch_records")
        records = row.get("resynchronization_records")
        if row.get("cell") != "fixed_round_trip":
            return switches == [] and records == []
        if type(switches) is not list or type(records) is not list:
            return False
        return (
            len(switches) == 2
            and len(records) == 2
            and all(
                type(item) is dict
                and item.get("eight_consecutive_boundaries_at_or_below_one") is True
                and type(item.get("settle_latency_steps")) is int
                and 0 <= int(item["settle_latency_steps"]) <= 64
                for item in records
            )
        )

    all_safe = bool(episodes) and all(safety_passed(row) for row in episodes)
    fixed = tuple(row for row in episodes if row["cell"] == "fixed_round_trip")
    holds = tuple(row for row in episodes if row["cell"] != "fixed_round_trip")
    boundary_only_failure = (
        all_safe
        and bool(fixed)
        and bool(holds)
        and all(row["utility_passed"] is True for row in holds)
        and any(row["utility_passed"] is False for row in fixed)
        and any(not resynchronization_passed(row) for row in fixed)
    )
    stable_substrate = all_safe and all(row["utility_passed"] is True for row in episodes)
    observed_task_failure = any(row["task_success"] is False for row in episodes)
    observed_task_success = bool(episodes) and all(row["task_success"] is True for row in episodes)

    if explicit_missing or not episodes or task_unknown:
        action = "measurement"
        confidence = "high"
        hypothesis = "The validated Phase B report explicitly lacks evidence required for a research-knob decision."
        rivals = (
            "No candidate should be selected by interpreting missing evidence as a negative result.",
        )
        prediction = (
            "Completing the named evidence fields should make the routing predicates decidable."
        )
        falsifier = (
            "The completed protected report still leaves oracle and reward explanations coupled."
        )
        ready = False
        reason = "Missing protected evidence blocks a scientifically informed patch."
    elif not all_safe:
        action = "measurement"
        confidence = "low"
        hypothesis = "Protected episodes contain a safety failure, but the report does not causally assign it to oracle or task reward."
        rivals = (
            "Controller or reference capacity may explain the fall or forbidden contact.",
            "Oracle transition and reward incentives remain competing explanations.",
        )
        prediction = (
            "A matched localization should determine whether failures concentrate at transitions."
        )
        falsifier = "Failures remain distributed independently of transition and task segments."
        ready = False
        reason = "An ambiguous safety failure cannot automatically blame either research knob."
    elif boundary_only_failure:
        action = "oracle"
        confidence = "moderate"
        hypothesis = (
            "Hold-cell utility passes while fixed round-trip transition or rejoin evidence fails."
        )
        rivals = (
            "Fixed-round-trip tracking error may contribute independently of authored transition logic.",
            "The frozen policy may lack transition robustness even under a correct oracle.",
            "Reference phase transfer may be the limiting frozen mechanism rather than the authored guard.",
        )
        prediction = "A bounded oracle change should improve rejoin evidence without degrading hold-cell tracking or safety."
        falsifier = "Matched protected evaluation shows unchanged rejoin evidence or new hold-cell degradation."
        ready = True
        reason = "Complete protected evidence supports an oracle hypothesis, not a causal claim."
    elif stable_substrate and observed_task_failure:
        action = "task_reward"
        confidence = "moderate"
        hypothesis = "Safety, tracking, and resynchronization are stable while the calibrated task endpoint still fails."
        rivals = (
            "The fixed policy may be insensitive to the current task-reward family.",
            "The task-reward family may be too narrow even if its parameters change.",
        )
        prediction = "A bounded task-reward candidate should improve the calibrated task endpoint without reducing utility or safety."
        falsifier = (
            "Matched protected evaluation shows no task improvement or any utility regression."
        )
        ready = True
        reason = (
            "Complete protected evidence supports a task-reward hypothesis, not a causal claim."
        )
    elif observed_task_success:
        action = "none"
        confidence = "high"
        hypothesis = "The validated protected endpoints do not identify a current failure requiring a candidate patch."
        rivals = ()
        prediction = (
            "Repeating the frozen evaluation should preserve the reported endpoint classification."
        )
        falsifier = "A matched repeat fails a protected endpoint."
        ready = False
        reason = "No observed failure warrants a patch."
    else:
        action = "measurement"
        confidence = "low"
        hypothesis = "The protected report is complete, but tracking, transition, and task signals do not isolate one authorable knob."
        rivals = (
            "The frozen controller or reference may dominate the observed error.",
            "Both oracle and task reward remain plausible research hypotheses.",
        )
        prediction = "A discriminating matched ablation should isolate one failure surface."
        falsifier = "The ablation preserves the same ambiguity."
        ready = False
        reason = "Ambiguous evidence blocks single-knob proposal readiness."

    inputs = report["inputs"]
    parents = {
        "oracle_sha256": str(report["oracles"][-1]["oracle_sha256"]),
        "task_spec_sha256": str(report["task_spec_sha256"]),
    }
    for name in ("reward_file_sha256", "reward_formula_sha256", "reward_schema_id"):
        if name in inputs:
            parents[name] = str(inputs[name])
    # This report is the protected evaluator.  Its outcomes may support a
    # human-readable diagnosis, but cannot select the next candidate or enter
    # candidate context.  A separate development probe must earn proposal
    # readiness in the next slice.
    if action in {"oracle", "task_reward"}:
        reason = (
            "Protected held-out evaluation is diagnostic-only; collect independent "
            "development evidence before selecting a candidate."
        )
    ready = False
    return FeedbackDiagnosis(
        action_surface=action,
        confidence=confidence,
        hypothesis=hypothesis,
        rivals=rivals,
        prediction=prediction,
        falsifier=falsifier,
        claim_ceilings=(
            str(phase_a.value["claim_ceiling"]),
            str(report["claim_ceiling"]),
        ),
        source_identities={
            "phase_a_scientific_receipt_sha256": phase_a.sha256,
            "phase_b_scientific_receipt_sha256": digest,
            "phase_b_trace_index_sha256": str(report["integrity"]["trace_index_sha256"]),
        },
        parent_identities=parents,
        observed_facts=facts,
        missing_evidence=tuple(dict.fromkeys(missing)),
        steering=steering,
        proposal_ready=ready,
        proposal_readiness_reason=reason,
    )


def diagnose_feedback(
    *,
    phase_a_receipt_path: Path,
    experiment: Path,
    repository_root: Path,
    phase_a_cycle: int,
    phase_b_evidence_path: Path | None = None,
    steering_path: Path | None = None,
) -> FeedbackDiagnosis:
    """Validate supplied evidence and emit one deterministic action surface."""

    steering = load_steering(steering_path)
    phase_a = validate_phase_a_receipt(
        receipt_path=phase_a_receipt_path,
        experiment=experiment,
        repository_root=repository_root,
        expected_cycle=phase_a_cycle,
    )
    phase_a_facts, slow_absent = _phase_a_facts(phase_a)
    if phase_b_evidence_path is None:
        return _phase_a_route(phase_a, phase_a_facts, slow_absent, steering)
    kind, value, digest = load_phase_b_evidence(phase_b_evidence_path)
    if kind == "structurally_validated_phase_b_seed_failure_v2":
        return _phase_b_failure_route(phase_a, phase_a_facts, value, digest, steering)
    return _phase_b_report_route(phase_a, phase_a_facts, value, digest, steering)


__all__ = [
    "FeedbackDiagnosis",
    "FeedbackEvidenceError",
    "ObservedFact",
    "SteeringInput",
    "diagnose_feedback",
    "load_phase_b_evidence",
    "load_steering",
    "validate_phase_a_receipt",
    "validate_seed_failure_receipt",
]
