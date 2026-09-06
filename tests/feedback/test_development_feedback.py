from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from oracle_composition.cli import main
from oracle_composition.contracts.reference_identity_v2 import (
    array_sha256,
    canonical_json_bytes,
)
from oracle_composition.development.reference_ablation import (
    CLAIM_CEILING,
    DEVELOPMENT_BLOCKS,
    EVIDENCE_CLASS,
    PROTOCOL_ID,
    SmokeExportLineage,
    matched_action_delta,
    prepare_reference_transform,
    prepared_reference_window,
    summarize_matched_action_deltas,
)
from oracle_composition.experiments.reference_input_transforms import CONDITION_IDS
from oracle_composition.feedback.context import build_candidate_context
from oracle_composition.feedback.development import (
    MANIFEST_SCHEMA_ID,
    RESULT_SCHEMA_ID,
    TRACE_SCHEMA_ID,
    validate_development_result,
)
from oracle_composition.feedback.development_diagnosis import (
    _route,
    diagnose_development_feedback,
)
from oracle_composition.feedback.evidence import FeedbackEvidenceError
from oracle_composition.phase_b.protected_metrics import (
    protected_step_record,
    protected_trace_sha256,
    recompute_protected_episode,
    select_evaluator_nearest_phase,
)
from oracle_composition.phase_b.reference_runtime import (
    LoadedReferenceClip,
    tracking_state_from_reference_row,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = (
    ROOT / "experiments/003_composition_speed_profile/phase_b/"
    "development_reference_ablation_v1.json"
)
TARGETS = (5.520768616125457, 0.8853599908576963, 5.520768616125457)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _write(path: Path, value: object) -> dict[str, object]:
    encoded = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return {
        "byte_count": len(encoded),
        "filename": path.name,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _reference(behavior: str) -> np.ndarray:
    rows = np.zeros((1_001, 45), dtype="<f8")
    rows[:, 0] = 1.4
    rows[:, 1] = 1.0
    rows[:, 5] = TARGETS[1] if behavior == "simple" else TARGETS[0]
    rows[:, 11] = np.arange(1_001, dtype=np.float64) / 10_000.0
    return rows


def _identities(block: int) -> dict[str, dict[str, object]]:
    return {
        behavior: {
            "bundle_sha256": _sha(f"bundle-{block}-{behavior}"),
            "payload_sha256": _sha(f"payload-{block}-{behavior}"),
            "reference_identity_sha256": _sha(f"identity-{block}-{behavior}"),
        }
        for behavior in ("expert", "simple")
    }


def _clip(block: int, behavior: str) -> LoadedReferenceClip:
    identity = _identities(block)[behavior]
    rows = _reference(behavior)
    return LoadedReferenceClip(
        block=block,
        behavior=behavior,
        reference_rows=rows,
        boundary_observations=np.zeros((1_001, 1), dtype="<f8"),
        bundle_sha256=str(identity["bundle_sha256"]),
        payload_sha256=str(identity["payload_sha256"]),
        reference_identity_sha256=str(identity["reference_identity_sha256"]),
    )


def _switches(references: dict[str, np.ndarray]) -> list[dict[str, object]]:
    result = []
    for boundary, source, target in (
        (300, "expert", "simple"),
        (600, "simple", "expert"),
    ):
        selected = select_evaluator_nearest_phase(
            state={
                name: value
                for name, value in protected_step_record(
                    step=0,
                    state=tracking_state_from_reference_row(references[source][boundary]),
                    reference_behavior=source,
                    reference_index=boundary,
                    reference_row=references[source][boundary],
                    action=np.zeros(17, dtype="<f4"),
                    root_x_before_m=0.0,
                    body_mass=np.ones(1, dtype="<f8"),
                    body_xipos_before=np.zeros((1, 3), dtype="<f8"),
                    body_xipos_after=np.zeros((1, 3), dtype="<f8"),
                    forbidden_contacts=(),
                    fallen=False,
                    terminated=False,
                    truncated=False,
                )["state"].items()
            },
            target_rows=references[target],
            task_step=boundary,
            source_behavior=source,
            target_behavior=target,
            reason="development_preregistered_direct_switch",
        )
        result.append(
            {
                "boundary": boundary,
                "from_behavior": source,
                "selected_normalized_errors": selected["selected_normalized_errors"],
                "selected_phase": selected["selected_phase"],
                "selected_score": selected["selected_score"],
                "to_behavior": target,
            }
        )
    return result


def _series(
    condition: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    references = {behavior: _reference(behavior) for behavior in ("expert", "simple")}
    transforms = {
        behavior: prepare_reference_transform(references[behavior], condition_id=condition)
        for behavior in ("expert", "simple")
    }
    switches = _switches(references)
    switch_map = {int(row["boundary"]): row for row in switches}
    behavior = "expert"
    phase = 0
    policy = []
    objective = []
    action = np.zeros(17, dtype="<f4")
    action_sha = array_sha256(action)
    for step in range(1_000):
        if step in switch_map:
            behavior = str(switch_map[step]["to_behavior"])
            phase = int(switch_map[step]["selected_phase"])
        window = prepared_reference_window(transforms[behavior], current_frame=phase)
        window_f32 = np.ascontiguousarray(window.values, dtype="<f4")
        policy.append(
            {
                "action_sha256": action_sha,
                "actor_input_sha256": _sha(f"input-{condition}-{step}"),
                "behavior": behavior,
                "condition_id": condition,
                "observation_sha256": _sha(f"observation-{condition}-{step}"),
                "phase": phase,
                "policy_window_float32_sha256": array_sha256(window_f32),
                "policy_window_float64_sha256": window.sha256,
                "step": step,
                "window_source_indices": list(window.source_indices),
                "window_timeline_indices": list(window.timeline_indices),
            }
        )
        target_phase = min(phase + 1, 1_000)
        target = references[behavior][target_phase].copy()
        speed = float(target[5])
        state = tracking_state_from_reference_row(target)
        before = np.zeros((1, 3), dtype="<f8")
        after = np.zeros((1, 3), dtype="<f8")
        after[0, 0] = speed * 0.015
        objective.append(
            protected_step_record(
                step=step,
                state=state,
                reference_behavior=behavior,
                reference_index=target_phase,
                reference_row=target,
                action=action,
                root_x_before_m=0.0,
                body_mass=np.ones(1, dtype="<f8"),
                body_xipos_before=before,
                body_xipos_after=after,
                forbidden_contacts=(),
                fallen=False,
                terminated=False,
                truncated=step == 999,
            )
        )
        phase = target_phase
    return policy, objective, switches


def _matched(policy: list[dict[str, object]]) -> list[dict[str, object]]:
    exact = np.zeros(17, dtype="<f4")
    values = {
        CONDITION_IDS[0]: 0.0,
        CONDITION_IDS[1]: 0.1,
        CONDITION_IDS[2]: 0.2,
        CONDITION_IDS[3]: 0.3,
    }
    result = []
    for row in policy:
        for condition in CONDITION_IDS:
            compared = np.full(17, values[condition], dtype="<f4")
            result.append(
                {
                    "behavior": row["behavior"],
                    "condition_id": condition,
                    "phase": row["phase"],
                    "step": row["step"],
                    **matched_action_delta(exact, compared),
                }
            )
    return result


@dataclass(frozen=True)
class Fixture:
    result: Path
    smoke_run: Path
    corpus_root: Path
    smoke: SmokeExportLineage
    corpus_bindings: MappingProxyType


def _fixture(tmp_path: Path) -> Fixture:
    output = tmp_path / "development"
    output.mkdir()
    smoke_run = tmp_path / "smoke"
    smoke_run.mkdir()
    actor = smoke_run / "actor.npz"
    actor.write_bytes(b"actor")
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    actor_sha = hashlib.sha256(actor.read_bytes()).hexdigest()
    smoke = SmokeExportLineage(
        run_directory=smoke_run.resolve(),
        actor_path=actor,
        actor_sha256=actor_sha,
        actor_byte_count=actor.stat().st_size,
        checkpoint_sha256=_sha("checkpoint"),
        execution_manifest_sha256=_sha("execution"),
        training_facts_sha256=_sha("training"),
        report_inputs=MappingProxyType(
            {
                "reference_sha256": _sha("corpus-manifest"),
                "sealed_input_lineage_sha256": _sha("sealed"),
            }
        ),
        sealed_inputs=MappingProxyType({}),
        bindings=MappingProxyType(
            {
                "execution_manifest": MappingProxyType(
                    {"byte_count": 100, "sha256": _sha("execution")}
                )
            }
        ),
    )
    corpus_bindings = MappingProxyType(
        {
            "corpus_manifest_v2.json": MappingProxyType(
                {
                    "byte_count": 100,
                    "path": "artifacts/reference_corpus_v2/corpus_manifest_v2.json",
                    "roles": ("reference_corpus",),
                    "sha256": _sha("corpus-manifest"),
                }
            )
        }
    )
    protocol_bytes = PROTOCOL.read_bytes()
    snapshot = {
        "authority_identities": {},
        "device": "cpu",
        "git": {"clean": True, "commit": "a" * 40},
        "platform": {"machine": "fixture", "python": "3.13", "system": "fixture"},
        "source_sha256": {
            "src/oracle_composition/__init__.py": _sha("package"),
            "src/oracle_composition/feedback/development.py": _sha("source"),
        },
        "versions": {
            "gymnasium": "fixture",
            "mujoco": "fixture",
            "numpy": "fixture",
            "stable_baselines3": "fixture",
            "torch": "fixture",
        },
    }
    snapshot_sha = hashlib.sha256(canonical_json_bytes(snapshot)).hexdigest()
    manifest = {
        "actor_export": {"byte_count": actor.stat().st_size, "sha256": actor_sha},
        "actor_visible_factor": "reference_window_only",
        "calibration_inputs": None,
        "claim_ceiling": CLAIM_CEILING,
        "conditions": list(CONDITION_IDS),
        "corpus_bindings": {name: dict(value) for name, value in corpus_bindings.items()},
        "development_blocks": list(DEVELOPMENT_BLOCKS),
        "evidence_class": EVIDENCE_CLASS,
        "held_out_inputs_used": False,
        "manifest_schema_id": MANIFEST_SCHEMA_ID,
        "objective_reference": "untransformed_exact_reference_rows",
        "package_import_path": str(ROOT / "src/oracle_composition/__init__.py"),
        "policy_seed": 121901,
        "promotable": False,
        "protocol": {
            "byte_count": len(protocol_bytes),
            "path": str(PROTOCOL.resolve()),
            "sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        },
        "runtime_source_snapshot": snapshot,
        "runtime_source_snapshot_sha256": snapshot_sha,
        "schema_version": 1,
        "smoke_lineage": {
            "actor_sha256": actor_sha,
            "bindings": {name: dict(value) for name, value in smoke.bindings.items()},
            "checkpoint_sha256": smoke.checkpoint_sha256,
            "execution_manifest_sha256": smoke.execution_manifest_sha256,
            "reference_sha256": smoke.report_inputs["reference_sha256"],
            "run_directory": str(smoke.run_directory),
            "sealed_input_lineage_sha256": smoke.report_inputs["sealed_input_lineage_sha256"],
            "training_facts_sha256": smoke.training_facts_sha256,
        },
        "task_success_scoring": False,
    }
    manifest_binding = _write(output / "development_manifest_v1.json", manifest)
    series = {condition: _series(condition) for condition in CONDITION_IDS}
    exact_matched = _matched(series[CONDITION_IDS[0]][0])
    matched_summary = summarize_matched_action_deltas(exact_matched)
    arms = []
    traces = []
    matched_by_block = {}
    for block in DEVELOPMENT_BLOCKS:
        for condition in CONDITION_IDS:
            policy, objective, switches = series[condition]
            aggregates = recompute_protected_episode(
                steps=objective,
                cell="fixed_round_trip",
                switches=switches,
                segment_targets_m_s=TARGETS,
            )
            matched_rows = exact_matched if condition == CONDITION_IDS[0] else []
            transforms = {
                behavior: prepare_reference_transform(_reference(behavior), condition_id=condition)
                for behavior in ("expert", "simple")
            }
            trace = {
                "actor_export_sha256": actor_sha,
                "block": block,
                "claim_ceiling": CLAIM_CEILING,
                "condition_id": condition,
                "development_only": True,
                "evidence_class": EVIDENCE_CLASS,
                "matched_state_action_deltas": matched_rows,
                "objective_metric_kernel": (
                    "phase_b_protected_metric_math_reused_without_protected_split_or_gate"
                ),
                "objective_reference": "untransformed_exact_reference_rows",
                "objective_steps": objective,
                "objective_steps_sha256": protected_trace_sha256(objective),
                "policy_inputs": policy,
                "policy_inputs_sha256": hashlib.sha256(canonical_json_bytes(policy)).hexdigest(),
                "policy_reference_factor": "actor_visible_8x45_window_only",
                "policy_seed": 121901,
                "promotable": False,
                "protocol_id": PROTOCOL_ID,
                "reference_identities": _identities(block),
                "reference_transform_precomputations": {
                    behavior: dict(transforms[behavior].receipt)
                    for behavior in ("expert", "simple")
                },
                "schema_version": 1,
                "switch_records": switches,
                "task_success": None,
                "trace_schema_id": TRACE_SCHEMA_ID,
            }
            trace_binding = _write(
                output / f"trace_block_{block}_{condition}.json",
                trace,
            )
            arm = {
                "action_bounds_ok": aggregates["action_bounds_ok"],
                "block": block,
                "condition_id": condition,
                "fall": aggregates["fall"],
                "forbidden_contact_count": len(aggregates["forbidden_contacts"]),
                "observed_steps": aggregates["observed_steps"],
                "resynchronization_records": aggregates["resynchronization_records"],
                "segment_errors": aggregates["segment_errors"],
                "settled_state_normalized_error": aggregates["settled_state_normalized_error"],
                "settle_latency_steps": aggregates["settle_latency_steps"],
                "six_tracking_errors": aggregates["six_tracking_errors"],
                "task_success": None,
                "time_to_first_failure_steps": aggregates["time_to_first_failure_steps"],
                "trace": trace_binding,
                "transition_window_error": aggregates["transition_window_error"],
            }
            arms.append(arm)
            traces.append(trace_binding)
            if condition == CONDITION_IDS[0]:
                matched_by_block[str(block)] = {
                    name: dict(value) for name, value in matched_summary.items()
                }
    result = {
        "actor_export_sha256": actor_sha,
        "arms": arms,
        "claim_ceiling": CLAIM_CEILING,
        "completed_arm_count": 16,
        "development_only": True,
        "evidence_class": EVIDENCE_CLASS,
        "held_out_inputs_used": False,
        "manifest": manifest_binding,
        "matched_state_action_deltas": matched_by_block,
        "planned_arm_count": 16,
        "promotable": False,
        "result_schema_id": RESULT_SCHEMA_ID,
        "runtime_source_snapshot_sha256": snapshot_sha,
        "schema_version": 1,
        "status": "succeeded",
        "task_success": None,
        "trace_artifacts": traces,
    }
    result_path = output / "development_result_v1.json"
    _write(result_path, result)
    return Fixture(result_path, smoke_run, corpus_root, smoke, corpus_bindings)


def _patch_authorities(monkeypatch: pytest.MonkeyPatch, fixture: Fixture) -> None:
    monkeypatch.setattr(
        "oracle_composition.feedback.development.validate_smoke_export",
        lambda _path: fixture.smoke,
    )
    monkeypatch.setattr(
        "oracle_composition.feedback.development.validate_development_corpus",
        lambda _root, *, lineage, protocol: fixture.corpus_bindings,
    )
    monkeypatch.setattr(
        "oracle_composition.feedback.development.load_v2_reference_clip",
        lambda _root, *, block, behavior: _clip(block, behavior),
    )


def test_complete_development_chain_recomputes_candidate_safe_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _fixture(tmp_path)
    _patch_authorities(monkeypatch, fixture)

    validated = validate_development_result(
        fixture.result,
        smoke_run=fixture.smoke_run,
        corpus_root=fixture.corpus_root,
        protocol_path=PROTOCOL,
    )
    diagnosis = diagnose_development_feedback(
        development_result_path=fixture.result,
        smoke_run=fixture.smoke_run,
        corpus_root=fixture.corpus_root,
        protocol_path=PROTOCOL,
    )
    context = build_candidate_context(diagnosis, database=None)

    assert len(validated.arms) == 16
    assert diagnosis.action_surface == "measurement"
    assert diagnosis.proposal_ready is False
    assert all(fact.candidate_visible for fact in diagnosis.observed_facts)
    prompt = context.prompt.decode()
    assert "Validated development measurements" in prompt
    assert "recorded_matched_action_changed_steps_by_reference_condition" in prompt
    assert "in-sample development measurements" in prompt
    assert "PROPOSAL_NOT_READY" in prompt
    assert "protected_phase_b" not in prompt

    output = tmp_path / "feedback-output"
    assert (
        main(
            [
                "--json",
                "diagnose-development",
                "--repository-root",
                str(ROOT),
                "--development-result",
                str(fixture.result),
                "--smoke-run",
                str(fixture.smoke_run),
                "--corpus-root",
                str(fixture.corpus_root),
                "--protocol",
                str(PROTOCOL),
                "--no-research-graph",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    cli_result = json.loads(capsys.readouterr().out)
    assert cli_result["kind"] == "humanoid_development_feedback_run_result"
    assert cli_result["readiness"]["proposal_ready"] is False
    assert (output / "diagnosis_v1.json").is_file()
    assert (output / "candidate_prompt_v1.md").is_file()


def test_rebound_result_cannot_override_recomputed_arm_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _patch_authorities(monkeypatch, fixture)
    value = json.loads(fixture.result.read_bytes())
    value["arms"][0]["fall"] = True
    fixture.result.write_bytes(canonical_json_bytes(value))

    with pytest.raises(FeedbackEvidenceError, match="summary does not recompute"):
        validate_development_result(
            fixture.result,
            smoke_run=fixture.smoke_run,
            corpus_root=fixture.corpus_root,
            protocol_path=PROTOCOL,
        )


@pytest.mark.parametrize(
    ("safe_exact", "tracking_exact", "changed", "transition_count", "expected"),
    [
        (4, 4, 0, 0, "adapter_repair"),
        (4, 3, 1, 0, "adapter_repair"),
        (4, 4, 1, 1, "oracle"),
        (4, 4, 1, 0, "measurement"),
    ],
)
def test_development_routing_keeps_adapter_and_oracle_causes_separate(
    safe_exact: int,
    tracking_exact: int,
    changed: int,
    transition_count: int,
    expected: str,
) -> None:
    safe = {condition: 4 for condition in CONDITION_IDS}
    tracking = {condition: 4 for condition in CONDITION_IDS}
    changed_steps = {condition: changed for condition in CONDITION_IDS}
    safe[CONDITION_IDS[0]] = safe_exact
    tracking[CONDITION_IDS[0]] = tracking_exact
    changed_steps[CONDITION_IDS[0]] = 0

    assert (
        _route(
            safe=safe,
            tracking=tracking,
            changed=changed_steps,
            transition_concentrated=transition_count,
        )[0]
        == expected
    )
