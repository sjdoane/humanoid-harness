from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from oracle_composition.cli import main
from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.feedback.context import _query, build_candidate_context
from oracle_composition.feedback.evidence import (
    FeedbackEvidenceError,
    _phase_b_report_route,
    load_steering,
    validate_seed_failure_receipt,
)
from oracle_composition.harness.evidence import ValidatedScientificReceipt

HASH_A = "a" * 64
HASH_B = "b" * 64
REPOSITORY = Path(__file__).resolve().parents[2]
EXPERIMENT = REPOSITORY / "experiments/003_composition_speed_profile"
PHASE_A_CYCLE_2 = EXPERIMENT / "cycles/cycle_2/scientific_receipt_v2.json"


def _phase_a() -> ValidatedScientificReceipt:
    return ValidatedScientificReceipt(
        path=Path("synthetic_phase_a.json"),
        value={"claim_ceiling": "synthetic_phase_a_only", "task_spec_sha256": HASH_A},
        encoded=b"synthetic",
        sha256=HASH_A,
        arms=(),
    )


def _row(
    cell: str,
    *,
    utility: bool,
    task_success: bool | None,
    fall: bool = False,
    resynchronized: bool = True,
) -> dict[str, object]:
    fixed = cell == "fixed_round_trip"
    return {
        "action_bounds_ok": True,
        "cell": cell,
        "contacts": [],
        "fall": fall,
        "observed_steps": 1_000,
        "resynchronization_records": (
            [
                {
                    "eight_consecutive_boundaries_at_or_below_one": True,
                    "settle_latency_steps": 8,
                },
                {
                    "eight_consecutive_boundaries_at_or_below_one": True,
                    "settle_latency_steps": 8,
                },
            ]
            if fixed and resynchronized
            else []
        ),
        "segment_errors": {"slow": 0.1},
        "switch_records": [{}, {}] if fixed else [],
        "task_success": task_success,
        "utility_passed": utility,
    }


def _report(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "claim_ceiling": "synthetic_protected_evaluation_only",
        "inputs": {},
        "integrity": {"explicit_missing_fields": [], "trace_index_sha256": HASH_A},
        "oracles": [{"oracle_sha256": HASH_A}],
        "per_episode": rows,
        "task_spec_sha256": HASH_A,
    }


def _diagnose_report(rows: list[dict[str, object]], digest: str):
    return _phase_b_report_route(_phase_a(), (), _report(rows), digest, None)


def test_real_cycle_two_cli_uses_only_current_twenty_episodes(tmp_path: Path, capsys) -> None:
    output = tmp_path / "feedback"
    steering = tmp_path / "steering.txt"
    steering.write_text("Prefer the smallest falsifiable oracle change.\n", encoding="utf-8")

    assert (
        main(
            [
                "--json",
                "diagnose",
                "--repository-root",
                str(REPOSITORY),
                "--experiment",
                str(EXPERIMENT),
                "--phase-a-receipt",
                str(PHASE_A_CYCLE_2),
                "--phase-a-cycle",
                "2",
                "--steering-file",
                str(steering),
                "--no-research-graph",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    diagnosis = json.loads((output / "diagnosis_v1.json").read_text(encoding="utf-8"))
    prompt = (output / "candidate_prompt_v1.md").read_text(encoding="utf-8")

    assert result["action_surface"] == "oracle"
    assert result["readiness"]["proposal_ready"] is False
    facts = {item["name"]: item["value"] for item in diagnosis["observed_facts"]}
    assert facts["episode_count"] == 20
    assert facts["fall_count"] == 0
    assert facts["slow_segment_simple_behavior_absent_all_episodes"] is True
    assert diagnosis["steering"]["sha256"] == hashlib.sha256(steering.read_bytes()).hexdigest()
    assert "controller_switching" in prompt
    assert "PROPOSAL_NOT_READY" in prompt
    assert "fall_count" not in prompt


def test_seed_failure_is_structural_only_and_rejects_tamper(tmp_path: Path) -> None:
    value = {
        "cleanup_outcome": {},
        "cleanup_succeeded": True,
        "evidence_class": "execution_failure_not_scientific_evidence",
        "execution_manifest_sha256": HASH_A,
        "failure_receipt_id": "humanoid_phase_b_seed_failure/v2",
        "last_acknowledged_stage": "admission",
        "outcome": "failure",
        "planned_transitions": 1_000,
        "ppo_seed": 7,
        "primary_failure": {
            "reason": "synthetic admission failure",
            "status": "precondition_failure",
        },
        "reason": "synthetic admission failure",
        "resource_controls": {},
        "schema_version": 2,
        "smoke": True,
        "status": "precondition_failure",
        "success_receipt_present": False,
    }
    path = tmp_path / "failure.json"
    path.write_bytes(canonical_json_bytes(value))
    admitted, _digest = validate_seed_failure_receipt(path)
    assert admitted["evidence_class"] == "execution_failure_not_scientific_evidence"

    value["reason"] = "tampered independently of primary_failure"
    path.write_bytes(canonical_json_bytes(value))
    with pytest.raises(FeedbackEvidenceError, match="exact schema"):
        validate_seed_failure_receipt(path)


def test_protected_opposite_outcomes_cannot_select_different_candidate_contexts() -> None:
    reward_diagnosis = _diagnose_report(
        [_row("hold_expert", utility=True, task_success=False)], HASH_A
    )
    oracle_diagnosis = _diagnose_report(
        [
            _row("hold_expert", utility=True, task_success=True),
            _row(
                "fixed_round_trip",
                utility=False,
                task_success=False,
                resynchronized=False,
            ),
        ],
        HASH_B,
    )
    assert reward_diagnosis.action_surface == "task_reward"
    assert oracle_diagnosis.action_surface == "oracle"
    assert reward_diagnosis.proposal_ready is False
    assert oracle_diagnosis.proposal_ready is False

    reward_context = build_candidate_context(reward_diagnosis, database=None)
    oracle_context = build_candidate_context(oracle_diagnosis, database=None)
    assert reward_context.action_surface == oracle_context.action_surface == "measurement"
    assert reward_context.candidate_kind == oracle_context.candidate_kind == "none"
    assert reward_context.proposal_ready is oracle_context.proposal_ready is False
    assert "task endpoint still fails" not in reward_context.prompt.decode()
    assert "round-trip" not in oracle_context.prompt.decode()

    def scrub(value: bytes) -> str:
        return re.sub(r"[0-9a-f]{64}", "HASH", value.decode())

    assert scrub(reward_context.prompt) == scrub(oracle_context.prompt)


def test_ambiguous_protected_fall_routes_measurement_without_reward_blame() -> None:
    diagnosis = _diagnose_report(
        [_row("hold_expert", utility=False, task_success=False, fall=True)], HASH_A
    )
    assert diagnosis.action_surface == "measurement"
    assert diagnosis.proposal_ready is False
    assert diagnosis.to_dict()["causal_status"] == "no_research_cause_assigned"


def test_heldout_context_never_queries_graph_or_copies_steering(
    tmp_path: Path, monkeypatch
) -> None:
    def forbidden_query(*args, **kwargs):
        pytest.fail("held-out-derived context must not query the research graph")

    monkeypatch.setattr(
        "oracle_composition.feedback.context.source_bundle_from_index", forbidden_query
    )
    steering_file = tmp_path / "steering.txt"
    steering_file.write_text("Leak canary: the held-out oracle failed.")
    diagnosis = replace(
        _diagnose_report([_row("hold_expert", utility=True, task_success=False)], HASH_A),
        steering=load_steering(steering_file),
    )
    context = build_candidate_context(diagnosis, database=tmp_path / "graph.sqlite")
    assert b"Leak canary" not in context.prompt
    assert context.source_bundle is None
    assert not context.proposal_ready


def test_steering_rejects_oversize_and_symlink(tmp_path: Path) -> None:
    oversized = tmp_path / "large.txt"
    oversized.write_bytes(b"x" * 16_385)
    with pytest.raises(FeedbackEvidenceError, match="bounded"):
        load_steering(oversized)
    target = tmp_path / "target.txt"
    target.write_text("A short instruction")
    link = tmp_path / "link.txt"
    link.symlink_to(target)
    with pytest.raises(FeedbackEvidenceError, match="bounded"):
        load_steering(link)


def test_retrieval_query_targets_mechanism_not_hypothesis_verbosity() -> None:
    diagnosis = _diagnose_report([_row("hold_expert", utility=True, task_success=False)], HASH_A)
    development = replace(
        diagnosis,
        action_surface="oracle",
        source_identities={"phase_a_scientific_receipt_sha256": HASH_A},
    )
    revised_text = replace(development, hypothesis="frozen controller " * 100, rivals=())
    assert _query(development) == _query(revised_text)
    assert _query(development) == "reference composition phase transition motion stitching"
