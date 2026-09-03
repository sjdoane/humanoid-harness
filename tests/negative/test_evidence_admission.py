from __future__ import annotations

import pytest

from oracle_composition.evaluation.admission import (
    EvidenceLabel,
    OracleTrainingEvidence,
    classify_oracle_training_evidence,
)


def test_empty_evidence_remains_an_interface_check() -> None:
    decision = classify_oracle_training_evidence(OracleTrainingEvidence())

    assert not decision.admitted
    assert decision.label is EvidenceLabel.INTERFACE_CHECK
    assert "tracker_sha256" in decision.missing_or_invalid
    assert "causal_reference_use_passed" in decision.missing_or_invalid


@pytest.mark.parametrize(
    "bad_digest",
    ["", "not-a-digest", "f" * 63, "g" * 64, "A" * 64],
)
def test_invalid_digest_fails_closed(bad_digest: str) -> None:
    evidence = _complete_evidence(tracker_sha256=bad_digest)

    decision = classify_oracle_training_evidence(evidence)

    assert not decision.admitted
    assert decision.missing_or_invalid == ("tracker_sha256",)


@pytest.mark.parametrize("truthy_non_boolean", [1, "true", object()])
def test_truthy_non_boolean_gate_fails_closed(truthy_non_boolean: object) -> None:
    evidence = _complete_evidence(actor_conditioned=truthy_non_boolean)

    decision = classify_oracle_training_evidence(evidence)

    assert not decision.admitted
    assert decision.missing_or_invalid == ("actor_conditioned",)


def test_complete_self_attested_receipt_is_only_a_structural_candidate() -> None:
    decision = classify_oracle_training_evidence(_complete_evidence())

    assert not decision.admitted
    assert decision.structurally_complete
    assert decision.label is EvidenceLabel.STRUCTURAL_CANDIDATE
    assert decision.missing_or_invalid == ()


def _complete_evidence(**overrides: object) -> OracleTrainingEvidence:
    values: dict[str, object] = {
        "environment_manifest_sha256": "0" * 64,
        "reference_manifest_sha256": "1" * 64,
        "tracker_sha256": "2" * 64,
        "tracking_reward_sha256": "3" * 64,
        "task_reward_sha256": "4" * 64,
        "trainer_manifest_sha256": "5" * 64,
        "evaluator_sha256": "6" * 64,
        "actor_conditioned": True,
        "critic_conditioned": True,
        "train_eval_manifests_match": True,
        "seeds_and_budget_matched": True,
        "causal_reference_use_passed": True,
    }
    values.update(overrides)
    return OracleTrainingEvidence(**values)  # type: ignore[arg-type]
