"""Fail-closed structural admission for oracle-training evidence.

This validator checks that required receipts are present. It does not establish
that the referenced artifacts are scientifically adequate; downstream code must
recompute each digest and evaluate the causal-use experiment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EvidenceLabel(StrEnum):
    INTERFACE_CHECK = "interface_check"
    STRUCTURAL_CANDIDATE = "structural_candidate"
    ORACLE_TRAINING = "oracle_training"


@dataclass(frozen=True, slots=True)
class OracleTrainingEvidence:
    environment_manifest_sha256: str | None = None
    reference_manifest_sha256: str | None = None
    tracker_sha256: str | None = None
    tracking_reward_sha256: str | None = None
    task_reward_sha256: str | None = None
    trainer_manifest_sha256: str | None = None
    evaluator_sha256: str | None = None
    actor_conditioned: bool = False
    critic_conditioned: bool = False
    train_eval_manifests_match: bool = False
    seeds_and_budget_matched: bool = False
    causal_reference_use_passed: bool = False


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    label: EvidenceLabel
    missing_or_invalid: tuple[str, ...]

    @property
    def admitted(self) -> bool:
        return self.label is EvidenceLabel.ORACLE_TRAINING

    @property
    def structurally_complete(self) -> bool:
        return not self.missing_or_invalid


def classify_oracle_training_evidence(evidence: OracleTrainingEvidence) -> AdmissionDecision:
    """Check receipt shape without promoting self-attestation to evidence.

    Even a complete record remains a ``structural_candidate`` until a separate
    trusted path recomputes every digest, reads the run ledger, and reruns the
    causal-reference evaluation. This module intentionally cannot issue an
    ``oracle_training`` label.
    """

    failures: list[str] = []
    digest_fields = (
        "environment_manifest_sha256",
        "reference_manifest_sha256",
        "tracker_sha256",
        "tracking_reward_sha256",
        "task_reward_sha256",
        "trainer_manifest_sha256",
        "evaluator_sha256",
    )
    for field in digest_fields:
        value = getattr(evidence, field)
        if not _is_sha256(value):
            failures.append(field)

    boolean_fields = (
        "actor_conditioned",
        "critic_conditioned",
        "train_eval_manifests_match",
        "seeds_and_budget_matched",
        "causal_reference_use_passed",
    )
    failures.extend(field for field in boolean_fields if getattr(evidence, field) is not True)

    return AdmissionDecision(
        label=(EvidenceLabel.INTERFACE_CHECK if failures else EvidenceLabel.STRUCTURAL_CANDIDATE),
        missing_or_invalid=tuple(failures),
    )


def _is_sha256(value: str | None) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None
