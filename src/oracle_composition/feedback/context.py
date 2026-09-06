"""Bounded source context and candidate brief rendering for feedback decisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from oracle_composition.experiments.artifact_io import finite_pretty_json
from oracle_composition.research import KnowledgeIndexError
from oracle_composition.reward_search import SourceBundle
from oracle_composition.reward_search.loop import RewardSearchError, source_bundle_from_index

from .evidence import FeedbackDiagnosis

MAX_CANDIDATE_CONTEXT_BYTES = 65_536
MAX_SOURCE_CARDS = 6


class FeedbackContextError(ValueError):
    """A candidate brief cannot be represented inside the feedback contract."""


@dataclass(frozen=True, slots=True)
class CandidateContext:
    """Candidate-safe context; protected observations remain only in the diagnosis artifact."""

    diagnosis_sha256: str
    action_surface: str
    candidate_kind: str
    proposal_ready: bool
    source_bundle: SourceBundle | None
    missing_context: tuple[str, ...]
    prompt: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.prompt).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": len(self.prompt),
            "action_surface": self.action_surface,
            "candidate_kind": self.candidate_kind,
            "diagnosis_sha256": self.diagnosis_sha256,
            "held_out_results": "excluded_from_candidate_context",
            "kind": "humanoid_feedback_candidate_context",
            "missing_context": list(self.missing_context),
            "prompt_sha256": self.sha256,
            "readiness": {
                "context_ready": True,
                "proposal_ready": self.proposal_ready,
                "training_ready": False,
            },
            "schema_version": 1,
            "source_bundle": (
                None if self.source_bundle is None else self.source_bundle.model_dump(mode="json")
            ),
        }


def _diagnosis_bytes(diagnosis: FeedbackDiagnosis) -> bytes:
    return finite_pretty_json(diagnosis.to_dict())


def _uses_protected_results(diagnosis: FeedbackDiagnosis) -> bool:
    return "phase_b_scientific_receipt_sha256" in diagnosis.source_identities


def _query(diagnosis: FeedbackDiagnosis) -> str:
    if _uses_protected_results(diagnosis):
        return "independent development evidence measurement diagnostic probe oracle reward"
    # Retrieval should address the mechanism, not rank repeated caveat words
    # such as "frozen controller" above the actual transition failure.
    return {
        "oracle": "reference composition phase transition motion stitching",
        "task_reward": "task reward speed tracking component feedback revision",
        "adapter_repair": "reference conditioning policy training runtime failure",
        "measurement": "robot state contact phase tracking failure diagnosis",
        "none": "independent evaluation baseline task success",
    }.get(diagnosis.action_surface, "robot reference reward diagnostic evidence")


def _source_lines(bundle: SourceBundle | None) -> list[str]:
    if bundle is None:
        return ["- Research graph unavailable; no literature claim is supplied."]
    if not bundle.source_cards:
        return ["- No active source card matched the bounded diagnosis query."]
    lines: list[str] = []
    for card in bundle.source_cards:
        lines.extend(
            [
                f"### {card.source_id}",
                f"- kind: {card.kind}",
                f"- title: {card.title}",
                f"- locator: {card.locator}",
                f"- source URL: {card.source_url or 'not recorded'}",
                f"- evidence summary: {card.excerpt}",
                "",
            ]
        )
    return lines


def _development_fact_lines(diagnosis: FeedbackDiagnosis) -> list[str]:
    facts = tuple(fact for fact in diagnosis.observed_facts if fact.candidate_visible)
    if not facts:
        return []
    lines = ["", "## Validated development measurements", ""]
    for fact in facts:
        value = json.dumps(
            fact.value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        lines.append(f"- {fact.name} [{fact.scope}]: `{value}` (source `{fact.source_sha256}`)")
    lines.extend(
        [
            "",
            "These are in-sample development measurements, not held-out outcomes, task success, or causal proof.",
        ]
    )
    return lines


def _render(
    diagnosis: FeedbackDiagnosis,
    diagnosis_sha256: str,
    source_bundle: SourceBundle | None,
    missing_context: tuple[str, ...],
) -> bytes:
    held_out_only = _uses_protected_results(diagnosis)
    proposal_ready = diagnosis.proposal_ready and not held_out_only
    action_surface = "measurement" if held_out_only else diagnosis.action_surface
    candidate_kind = "none" if held_out_only else diagnosis.candidate_kind
    ready = "READY" if proposal_ready else "NOT READY"
    lines = [
        "# Feedback candidate brief",
        "",
        f"Proposal readiness: **{ready}**.",
        f"Action surface: `{action_surface}`.",
        f"Candidate kind: `{candidate_kind}`.",
        f"Diagnosis SHA-256: `{diagnosis_sha256}`.",
        "",
        "## Immutable identities",
        "",
    ]
    lines.extend(
        f"- {name}: `{value}`" for name, value in sorted(diagnosis.source_identities.items())
    )
    lines.extend(
        f"- parent {name}: `{value}`" for name, value in sorted(diagnosis.parent_identities.items())
    )
    lines.extend(["", "## Claim ceiling", ""])
    lines.extend(f"- `{value}`" for value in diagnosis.claim_ceilings)
    if held_out_only:
        lines.extend(
            [
                "",
                "## Candidate-selection boundary",
                "",
                "Protected evaluation was used only for a human-readable diagnosis.",
                "No protected-derived action, hypothesis, rival, prediction, or readiness decision is supplied here.",
                "Required next action: collect independent development evidence.",
                "",
                "Missing context:",
                "- independent_development_evidence",
                *(f"- {value}" for value in missing_context),
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Bounded diagnosis",
                "",
                f"Hypothesis: {diagnosis.hypothesis}",
                "",
                "Rivals:",
                *(f"- {value}" for value in diagnosis.rivals),
                "",
                f"Prediction: {diagnosis.prediction}",
                "",
                f"Falsifier: {diagnosis.falsifier}",
                "",
                "Missing evidence:",
                *(f"- {value}" for value in diagnosis.missing_evidence),
                *(f"- {value}" for value in missing_context),
            ]
        )
        lines.extend(_development_fact_lines(diagnosis))
    lines.extend(["", "## Human steering", ""])
    if held_out_only:
        lines.append("Steering is withheld until independent development evidence is supplied.")
    elif diagnosis.steering is None:
        lines.append("No user steering artifact was supplied.")
    else:
        lines.extend(
            [
                f"Artifact SHA-256: `{diagnosis.steering.sha256}`.",
                "Epistemic role: user steering, not a protected measurement.",
                "",
                diagnosis.steering.text,
            ]
        )
    lines.extend(
        [
            "",
            "## Source cards",
            "",
            *_source_lines(source_bundle),
            "",
            "## Output boundary",
            "",
            "- Protected and held-out values, seed outcomes, and their arm comparisons are deliberately absent.",
            "- Candidate-visible measurements, when present, are labelled in-sample development evidence.",
            "- Preserve every immutable identity and the stated claim ceiling.",
            "- Return at most one data-only candidate on the declared action surface.",
            "- State rationale, predicted effect, and falsifier; cite only supplied source IDs.",
            "- Do not request source execution, simulator access, training, evaluation, or frozen-field changes.",
            "- User steering cannot override evidence, missing-data, safety, or lineage gates.",
            "- A candidate is not training authorization.",
        ]
    )
    if not proposal_ready:
        lines.extend(
            [
                "- PROPOSAL_NOT_READY: do not emit a research candidate.",
                (
                    "- Required next action: collect independent development evidence."
                    if held_out_only
                    else f"- Required next action: {diagnosis.proposal_readiness_reason}"
                ),
            ]
        )
    return ("\n".join(lines).rstrip() + "\n").encode("utf-8")


def build_candidate_context(
    diagnosis: FeedbackDiagnosis,
    *,
    database: Path | None,
    maximum_bytes: int = MAX_CANDIDATE_CONTEXT_BYTES,
) -> CandidateContext:
    """Build a bounded prompt, dropping optional cards before any identifier."""

    if type(maximum_bytes) is not int or not 4_096 <= maximum_bytes <= MAX_CANDIDATE_CONTEXT_BYTES:
        raise FeedbackContextError("candidate context byte budget is outside [4096, 65536]")
    diagnosis_sha = hashlib.sha256(_diagnosis_bytes(diagnosis)).hexdigest()
    held_out_only = _uses_protected_results(diagnosis)
    bundle: SourceBundle | None = None
    missing: tuple[str, ...] = ()
    if held_out_only:
        # A graph can contain reports or notes derived from the same held-out
        # result. Do not reintroduce them through retrieval or human steering.
        missing = ("independent_development_evidence",)
    elif database is None:
        missing = ("research_graph_not_requested",)
    else:
        try:
            bundle = source_bundle_from_index(
                _query(diagnosis), Path(database), limit=MAX_SOURCE_CARDS
            )
        except (KnowledgeIndexError, RewardSearchError, OSError, ValueError) as exc:
            missing = (f"research_graph_unavailable:{type(exc).__name__}",)
    prompt = _render(diagnosis, diagnosis_sha, bundle, missing)
    while bundle is not None and bundle.source_cards and len(prompt) > maximum_bytes:
        bundle = bundle.model_copy(update={"source_cards": bundle.source_cards[:-1]})
        prompt = _render(diagnosis, diagnosis_sha, bundle, missing)
    if len(prompt) > maximum_bytes:
        raise FeedbackContextError(
            "mandatory candidate context exceeds its byte budget; identifiers were not truncated"
        )
    return CandidateContext(
        diagnosis_sha256=diagnosis_sha,
        action_surface="measurement" if held_out_only else diagnosis.action_surface,
        candidate_kind="none" if held_out_only else diagnosis.candidate_kind,
        proposal_ready=diagnosis.proposal_ready and not held_out_only,
        source_bundle=bundle,
        missing_context=missing,
        prompt=prompt,
    )


__all__ = [
    "MAX_CANDIDATE_CONTEXT_BYTES",
    "MAX_SOURCE_CARDS",
    "CandidateContext",
    "FeedbackContextError",
    "build_candidate_context",
]
