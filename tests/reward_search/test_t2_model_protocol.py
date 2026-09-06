from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.reward_search import t2_model_protocol as protocol
from oracle_composition.reward_search.contracts import ModelCallReceipt, RewardProposal
from oracle_composition.reward_search.formula_contracts import (
    FormulaIngestionReceiptV2,
    FormulaProposalV2,
)
from oracle_composition.reward_search.loop import RewardSearchError, parse_model_bytes
from oracle_composition.reward_search.publication import finite_pretty_json
from oracle_composition.reward_search.t2_model_contracts import (
    CALL_IDENTITY_FILENAME,
    DISPATCH_INTENT_FILENAME,
    EXPECTED_BRANCH,
    EXPECTED_CHECKOUT_REALPATH,
    EXPECTED_IMPORT_ORIGIN,
    EXPECTED_PREPARATION_OWNER,
    EXPECTED_PREPARATION_ROLE,
    EXPECTED_PREPARATION_SCOPE,
    T2InitialDispatchIntent,
    T2InitialPacketRecord,
    T2ModelCallReceipt,
    T2ParameterProposal,
    T2PreDispatchSeal,
    T2ProtocolRefusalReceipt,
    T2RunArtifactBinding,
    derive_t2_call_id,
)
from oracle_composition.reward_search.t2_model_protocol import (
    BASELINE_BYTE_COUNT,
    BASELINE_GIT_OBJECT,
    BASELINE_GIT_PATH,
    BASELINE_SHA256,
    EVALUATOR_SOURCE_SHA256,
    PARAMETER_BOUNDS_SHA256,
    RECIPE_SOURCE_SHA256,
    TASK_INPUTS_SCHEMA_SHA256,
    TASK_INPUTS_SOURCE_SHA256,
    T2ProtocolError,
    ingest_initial_t2_sol_run,
    prepare_initial_t2_packet,
    publish_initial_t2_dispatch_intent,
    publish_initial_t2_packet,
    render_initial_t2_prompt,
)
from oracle_composition.reward_study.study_manifest import study_pairing_sha256_from_arm

OLD_BASELINE = b'{"compositor_id":"tracking_plus_task_stock_telemetry/v1","compositor_sha256":"f048a1d47280e88fd0bdf9dc207bed5c3c7bd3736feb95e59e298c6b9017b46b","evidence_class":"interface_check","formula_id":"tracking_only/v1","formula_sha256":"0e7514f39a4153e8baf308cfb217ea85314b7f088e5476cc5f0ba167801fe7fe","parameter_bounds_sha256":"42c0b5c273e19a3bb356c2d88c3ff4545ebaf1f78ac441af49521ff71bc38ca1","parameters":{},"parser_id":"no_candidate_inputs/v1","parser_sha256":"525ab6d15844f444c2eae522893e61cfe7c2bf5b3ec7d604e4a1fa10332ca83a","r_task":0.0,"reward_schema_id":"reward_specification/tracking_only/v1","schema_sha256":"fa7dc510f759cf0bec4038d0c82e6c49804756bbe2eae795b21203de8143c3ae","schema_version":1,"stock_reward":"telemetry_only","task_inputs_schema_id":"humanoid-fixed-com-speed/task-inputs/v2","task_inputs_schema_sha256":"8f382dde13ee44c27cbbbc0b3a53a568338f6b7cebc8e660e4084425b7b494e9","task_inputs_source_sha256":"9607f2d56a54922eac06ec7fcc740b79e68d4ed9e88b192602aae2900fb3f0d2","tracking_reward_config_sha256":"cc55731febc0c05a94174d4b99601fc8b3745273d21ee72236f4d9a75f81b10e","tracking_reward_id":"humanoid_root_and_joint_tracking/v1"}'
BASELINE = b'{"compositor_id":"tracking_plus_task_stock_telemetry/v1","compositor_sha256":"f048a1d47280e88fd0bdf9dc207bed5c3c7bd3736feb95e59e298c6b9017b46b","evidence_class":"interface_check","formula_id":"tracking_only/v1","formula_sha256":"0e7514f39a4153e8baf308cfb217ea85314b7f088e5476cc5f0ba167801fe7fe","parameter_bounds_sha256":"42c0b5c273e19a3bb356c2d88c3ff4545ebaf1f78ac441af49521ff71bc38ca1","parameters":{},"parser_id":"no_candidate_inputs/v1","parser_sha256":"525ab6d15844f444c2eae522893e61cfe7c2bf5b3ec7d604e4a1fa10332ca83a","r_task":0.0,"reward_schema_id":"reward_specification/tracking_only/v1","schema_sha256":"adb07a7bd474a0234b3025d84578e41925532e0c722510280daf64f49b512c84","schema_version":1,"stock_reward":"telemetry_only","task_input_admission_contract":{"cadence_seconds":0.015,"inclusive_velocity_range_m_s":[-25.0,25.0],"measurement_origin":"stock_body_mass_weighted_com_x_delta_over_control_period","numeric_representation":"builtin_float_from_little_endian_float64_measurement","schema_version":1,"task_input_admission_id":"phase_b_stock_com_task_input_admission/v1"},"task_input_admission_id":"phase_b_stock_com_task_input_admission/v1","task_input_admission_schema_sha256":"c2d8908533a47e76e5ec469c80b6cf2d7061397d3716597ec7c66b4f8913b4b5","task_input_admission_source_sha256":"db63c68693232a95995a3867e230d6d83ee1d31ba7072405d9377a38ebbede44","task_inputs_schema_id":"humanoid-fixed-com-speed/task-inputs/v2","task_inputs_schema_sha256":"8f382dde13ee44c27cbbbc0b3a53a568338f6b7cebc8e660e4084425b7b494e9","task_inputs_source_sha256":"9607f2d56a54922eac06ec7fcc740b79e68d4ed9e88b192602aae2900fb3f0d2","tracking_reward_config_sha256":"cc55731febc0c05a94174d4b99601fc8b3745273d21ee72236f4d9a75f81b10e","tracking_reward_id":"humanoid_root_and_joint_tracking/v1"}'
ROOT = Path(__file__).parents[2]
T2_SEAL_PATH = ROOT / "experiments/004_t2_reward_study/t2_seal_v1.json"
T2_SEAL = T2_SEAL_PATH.read_bytes()


@pytest.fixture(autouse=True)
def _isolated_canonical_call_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(protocol, "CALL_ROOT", tmp_path / "canonical-call-root")


def _record() -> tuple[bytes, T2InitialPacketRecord]:
    encoded = prepare_initial_t2_packet(
        baseline_reward_bytes=BASELINE,
        t2_seal_bytes=T2_SEAL,
    )
    parsed = parse_model_bytes(encoded, T2InitialPacketRecord, max_bytes=131_072)
    return encoded, parsed


def _t2pairr1_identity_table(document: str) -> dict[str, str]:
    marker = "### T2PAIRR1 identities\n"
    assert document.count(marker) == 1
    lines = document.split(marker, maxsplit=1)[1].splitlines()
    header_index = lines.index("| field | value |")
    assert lines[header_index + 1] == "|---|---|"
    rows: list[tuple[str, str]] = []
    for line in lines[header_index + 2 :]:
        if not line.startswith("|"):
            break
        match = re.fullmatch(r"\| ([^|]+) \| `([^`]+)` \|", line)
        assert match is not None
        rows.append((match.group(1), match.group(2)))
    return {field: value for field, value in rows}


def _assert_t2pairr1_identity_table(document: str, expected: dict[str, str]) -> None:
    assert _t2pairr1_identity_table(document) == expected


def _proposal(record: T2InitialPacketRecord, **updates: object) -> bytes:
    value: dict[str, object] = {
        "schema_version": 3,
        "kind": "t2_initial_parameter_proposal",
        "request_payload_sha256": record.request_payload_sha256,
        "baseline_sha256": record.baseline_sha256,
        "t2_contract_sha256": record.t2_contract_sha256,
        "evidence_dossier_sha256": record.evidence_dossier_sha256,
        "parameters": {"alpha": 1.5, "beta": -0.25},
        "rationale": "A bounded initial hypothesis.",
        "predicted_effect": "May sharpen reward around the fixed target.",
        "falsifier": "Protected measurements do not improve under the locked protocol.",
        "status": "hypothesis_only_not_admitted",
    }
    value.update(updates)
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=True) + "\n").encode()


def _envelope(
    tmp_path: Path,
    *,
    proposal: bytes | None = None,
    packet: bytes | None = None,
    request_updates: dict[str, object] | None = None,
    result_updates: dict[str, object] | None = None,
    publish_intent: bool = True,
    run_name: str = "fixture-run",
) -> tuple[bytes, Path]:
    record_bytes, record = _record()
    assert record.baseline.artifact_id.startswith(f"git:{BASELINE_GIT_OBJECT}:")
    prompt = render_initial_t2_prompt(record_bytes)
    intent_path = protocol.CALL_ROOT / DISPATCH_INTENT_FILENAME
    if publish_intent and not intent_path.exists():
        publish_initial_t2_dispatch_intent(record_bytes, intent_path)
    request: dict[str, object] = {
        "schema_version": 1,
        "mode": "read-only",
        "owner": "fable-f3-initial",
        "role": "candidate",
        "scope": record.expected_scope,
        "requested_model": "gpt-5.6-sol",
        "requested_reasoning_effort": "max",
        "runner_kind": "screen",
        "screen_name": "hh-sol-fixture",
        "codex_bin": "/fixture/codex",
        "codex_cli_version": "codex-cli fixture",
        "prompt_sha256": hashlib.sha256(prompt).hexdigest(),
        "prompt_bytes": len(prompt),
        "created_at_utc": "2026-09-05T23:00:00Z",
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "SUCCEEDED",
        "exit_code": 0,
        "thread_id": "fixture-thread",
        "finished_at_utc": "2026-09-05T23:10:00Z",
        "lease_release": "NOT_REQUIRED",
        "termination_escalated": False,
        "requested_model": "gpt-5.6-sol",
        "requested_reasoning_effort": "max",
        "runner_kind": "screen",
        "screen_name": "hh-sol-fixture",
        "codex_cli_version": "codex-cli fixture",
        "role": "candidate",
        "scope": record.expected_scope,
    }
    request.update(request_updates or {})
    result.update(result_updates or {})
    run = tmp_path / run_name
    run.mkdir()
    files = {
        "request.json": finite_pretty_json(request),
        "task-packet.md": prompt if packet is None else packet,
        "result.json": finite_pretty_json(result),
        "final.txt": _proposal(record) if proposal is None else proposal,
    }
    for name, encoded in files.items():
        path = run / name
        path.write_bytes(encoded)
        path.chmod(0o600)
    return record_bytes, run


def _load_receipt(path: Path) -> T2ModelCallReceipt:
    return parse_model_bytes(path.read_bytes(), T2ModelCallReceipt, max_bytes=65_536)


def _accepted_receipt_payload(tmp_path: Path) -> dict[str, object]:
    record_bytes, run = _envelope(tmp_path)
    outcome = ingest_initial_t2_sol_run(record_bytes, run)
    assert outcome.accepted
    return json.loads(outcome.receipt.path.read_bytes())


def test_true_baseline_packet_binds_exact_sources_and_visible_identities() -> None:
    assert len(BASELINE) == BASELINE_BYTE_COUNT
    assert hashlib.sha256(BASELINE).hexdigest() == BASELINE_SHA256
    assert not BASELINE.endswith(b"\n")
    assert (ROOT / BASELINE_GIT_PATH).read_bytes() == BASELINE
    baseline_payload = json.loads(BASELINE)
    assert baseline_payload["r_task"] == 0.0
    assert baseline_payload["task_input_admission_contract"] == {
        "cadence_seconds": 0.015,
        "inclusive_velocity_range_m_s": [-25.0, 25.0],
        "measurement_origin": "stock_body_mass_weighted_com_x_delta_over_control_period",
        "numeric_representation": "builtin_float_from_little_endian_float64_measurement",
        "schema_version": 1,
        "task_input_admission_id": "phase_b_stock_com_task_input_admission/v1",
    }
    record_bytes, record = _record()
    prompt = render_initial_t2_prompt(record_bytes)
    seal = parse_model_bytes(T2_SEAL, T2PreDispatchSeal, max_bytes=65_536)
    assert seal.pairing_receipt == (
        "1a2b7ece139974117fd5c75e040d9cc52cd51a4e42c9b5afd794a60b02232348"
    )
    assert seal.dispatch_state == "withheld_pending_dispatch_verdict"
    assert record.schema_version == 4
    assert record.t2_seal.sha256 == hashlib.sha256(T2_SEAL).hexdigest()
    assert record.study_manifest_sha256 == seal.study_manifest.sha256
    assert record.call_identity.t2_seal_sha256 == record.t2_seal.sha256
    assert (
        record.call_identity_sha256
        == hashlib.sha256(
            finite_pretty_json(record.call_identity.model_dump(mode="json"))
        ).hexdigest()
    )
    assert record.expected_scope.endswith(record.call_identity_sha256)
    assert record.contracts.recipe_parser_source.sha256 == RECIPE_SOURCE_SHA256
    assert record.contracts.evaluator_source.sha256 == EVALUATOR_SOURCE_SHA256
    assert record.contracts.task_inputs_source.sha256 == TASK_INPUTS_SOURCE_SHA256
    assert record.contracts.canonical_task_inputs_schema.sha256 == TASK_INPUTS_SCHEMA_SHA256
    assert record.contracts.canonical_parameter_bounds.sha256 == PARAMETER_BOUNDS_SHA256
    assert record.contracts.canonical_parameter_bounds.byte_count == 137
    assert record.evidence_dossier.aggregate_feedback == []
    assert record.semantic_request.task_text == (
        "Hold 3.0 m/s COM forward speed from the expert start."
    )
    assert record.semantic_request.target_speed_m_s == 3.0
    assert record.semantic_request.control_period_seconds == 0.015
    assert record.expected_checkout_realpath == EXPECTED_CHECKOUT_REALPATH
    assert record.expected_branch == EXPECTED_BRANCH
    assert record.expected_import_origin == EXPECTED_IMPORT_ORIGIN
    assert record.expected_preparation_owner == EXPECTED_PREPARATION_OWNER
    assert record.expected_preparation_role == EXPECTED_PREPARATION_ROLE
    assert record.expected_preparation_scope == EXPECTED_PREPARATION_SCOPE
    assert record.expected_owner == "fable-f3-initial"
    for digest in (
        record.request_payload_sha256,
        record.baseline_sha256,
        record.t2_contract_sha256,
        record.evidence_dossier_sha256,
    ):
        assert digest.encode() in prompt
    assert record.rendered_prompt_sha256.encode() not in prompt


def test_t2pairr1_document_identities_match_the_derived_committed_record() -> None:
    baseline = (ROOT / BASELINE_GIT_PATH).read_bytes()
    seal_bytes = T2_SEAL_PATH.read_bytes()
    record_bytes = prepare_initial_t2_packet(
        baseline_reward_bytes=baseline,
        t2_seal_bytes=seal_bytes,
    )
    record = parse_model_bytes(record_bytes, T2InitialPacketRecord, max_bytes=131_072)
    prompt = render_initial_t2_prompt(record_bytes)
    expected = {
        "study manifest": record.study_manifest_sha256,
        "F3 seal": record.t2_seal.sha256,
        "rendered prompt": hashlib.sha256(prompt).hexdigest(),
        "call ID": record.call_identity.call_id,
        "identity digest": record.call_identity_sha256,
        "launcher scope": record.expected_scope,
    }
    document_paths = (
        (ROOT / "docs/operations/dual-orchestration/F3_ONE_CALL_PROTOCOL.md", True),
        (ROOT / "docs/operations/dual-orchestration/F3_FABLE_REPIN.md", False),
    )
    for path, is_dispatch_protocol in document_paths:
        document = path.read_text(encoding="utf-8")
        _assert_t2pairr1_identity_table(document, expected)

        if is_dispatch_protocol:
            pinned_scopes = re.findall(
                r"`(t2-initial-parameter-hypothesis-only;call_identity_sha256=[0-9a-f]{64})`",
                document,
            )
            assert pinned_scopes == [record.expected_scope, record.expected_scope]
            assert (
                "eeb24a672fd61fbf91d668cd2ca62629decf9f9ed12124f03a27925c01ee4aef" not in document
            )
            assert (
                "1ae0e967207c1dc2a2d9535e98b7ea6782dabccdf6afff6f518bb07242bc892d" not in document
            )

        pinned_row = f"| call ID | `{expected['call ID']}` |"
        stale_document = document.replace(
            pinned_row,
            f"| call ID | `{'0' * 64}` |",
            1,
        )
        assert stale_document != document
        with pytest.raises(AssertionError):
            _assert_t2pairr1_identity_table(stale_document, expected)


def test_rendered_prompt_is_deterministic_hash_bound_and_minimal() -> None:
    first_record_bytes, first_record = _record()
    second_record_bytes, second_record = _record()
    first_prompt = render_initial_t2_prompt(first_record_bytes)
    second_prompt = render_initial_t2_prompt(second_record_bytes)

    assert first_record_bytes == second_record_bytes
    assert first_record == second_record
    assert first_prompt == second_prompt
    assert len(first_prompt) == 6_011
    assert hashlib.sha256(first_prompt).hexdigest() == (
        "4d1a29765d929472e2f2a346294f98ac08c0e35d3e18c147e761e46678fe2bbc"
    )
    assert first_record.rendered_prompt_sha256 == hashlib.sha256(first_prompt).hexdigest()
    assert first_record.rendered_prompt_byte_count == len(first_prompt)

    required = (
        b"Task: Hold 3.0 m/s COM forward speed from the expert start.",
        first_record.semantic_request.baseline_artifact_id.encode(),
        BASELINE_SHA256.encode(),
        b"target_speed_triangular_affine/v1",
        EVALUATOR_SOURCE_SHA256.encode(),
        RECIPE_SOURCE_SHA256.encode(),
        TASK_INPUTS_SOURCE_SHA256.encode(),
        PARAMETER_BOUNDS_SHA256.encode(),
        b"Only alpha and beta are authorable.",
        b"No measured feedback exists for this initial hypothesis call.",
    )
    assert all(value in first_prompt for value in required)
    assert BASELINE not in first_prompt
    assert b"qpos" not in first_prompt
    assert b"qvel" not in first_prompt
    assert b"worker_inputs" not in first_prompt
    assert b"45ce5baa" not in first_prompt
    retained_payloads = (
        first_record.baseline,
        first_record.contracts.recipe_parser_source,
        first_record.contracts.evaluator_source,
        first_record.contracts.task_inputs_source,
        first_record.contracts.canonical_task_inputs_schema,
        first_record.contracts.canonical_parameter_bounds,
    )
    assert all(item.content_base64.encode() not in first_prompt for item in retained_payloads)
    seal = parse_model_bytes(T2_SEAL, T2PreDispatchSeal, max_bytes=65_536)
    evaluator_bytes = (ROOT / seal.evaluator_design.path).read_bytes()
    study_manifest_bytes = (ROOT / seal.study_manifest.path).read_bytes()
    assert evaluator_bytes not in first_prompt
    assert study_manifest_bytes not in first_prompt
    assert seal.evaluator_design.sha256.encode() not in first_prompt
    assert seal.study_manifest.sha256.encode() not in first_prompt
    assert seal.study_pairing_sha256.encode() not in first_prompt
    assert first_record.t2_seal.sha256.encode() not in first_prompt
    assert b"integrated_pairing_receipt_sha256" not in first_prompt
    assert b"humanoid_phase_b_execution_manifest/v3" not in first_prompt


@pytest.mark.parametrize(
    "field",
    [
        "execution_manifest",
        "expert_hold_oracle",
        "training_design",
        "evaluator_design",
        "study_manifest",
        "tracking_only_baseline",
    ],
)
def test_preparation_refuses_every_changed_t2_seal_artifact(field: str) -> None:
    payload = json.loads(T2_SEAL)
    payload[field]["sha256"] = "0" * 64
    with pytest.raises(
        T2ProtocolError,
        match=r"(invalid T2 pre-dispatch seal|sealed T2 artifact identity differs)",
    ):
        prepare_initial_t2_packet(
            baseline_reward_bytes=BASELINE,
            t2_seal_bytes=finite_pretty_json(payload),
        )


def test_preparation_refuses_changed_pairing_key() -> None:
    payload = json.loads(T2_SEAL)
    payload["study_pairing_sha256"] = "0" * 64
    with pytest.raises(T2ProtocolError, match="manifest identity or pending state"):
        prepare_initial_t2_packet(
            baseline_reward_bytes=BASELINE,
            t2_seal_bytes=finite_pretty_json(payload),
        )


def test_preparation_refuses_changed_integrated_pairing_receipt() -> None:
    payload = json.loads(T2_SEAL)
    payload["pairing_receipt"] = "0" * 64
    with pytest.raises(T2ProtocolError, match="sealed T2 artifact identity differs"):
        prepare_initial_t2_packet(
            baseline_reward_bytes=BASELINE,
            t2_seal_bytes=finite_pretty_json(payload),
        )


def test_prompt_bytes_do_not_change_with_a_valid_non_model_facing_seal_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_record_bytes, original_record = _record()
    original_prompt = render_initial_t2_prompt(original_record_bytes)
    alternate_root = tmp_path / "alternate-repository"

    seal_payload = json.loads(T2_SEAL)
    manifest_payload = json.loads((ROOT / seal_payload["study_manifest"]["path"]).read_bytes())
    alternate_evaluator = canonical_json_bytes(
        {"design": "test-only-alternate-non-model-facing-evaluator"}
    )
    alternate_evaluator_binding = {
        "byte_count": len(alternate_evaluator),
        "path": seal_payload["evaluator_design"]["path"],
        "sha256": hashlib.sha256(alternate_evaluator).hexdigest(),
    }
    for arm in manifest_payload["arms"]:
        arm["evaluator"] = alternate_evaluator_binding
    manifest_payload["study_pairing_sha256"] = study_pairing_sha256_from_arm(
        manifest_payload["arms"][0]
    )
    alternate_manifest = canonical_json_bytes(manifest_payload)
    seal_payload["evaluator_design"] = alternate_evaluator_binding
    seal_payload["study_manifest"] = {
        "byte_count": len(alternate_manifest),
        "path": seal_payload["study_manifest"]["path"],
        "sha256": hashlib.sha256(alternate_manifest).hexdigest(),
    }
    seal_payload["study_pairing_sha256"] = manifest_payload["study_pairing_sha256"]
    alternate_seal = finite_pretty_json(seal_payload)

    copied_paths = [
        seal_payload["execution_manifest"]["path"],
        seal_payload["expert_hold_oracle"]["path"],
        seal_payload["training_design"]["path"],
        seal_payload["tracking_only_baseline"]["path"],
        "experiments/004_t2_reward_study/pairing_receipt_v1.json",
        "src/oracle_composition/rewards/target_speed_formula.py",
        "src/oracle_composition/rewards/target_speed_formula_t2.py",
        "src/oracle_composition/rewards/task_inputs_v2.py",
    ]
    for relative in copied_paths:
        destination = alternate_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    evaluator_path = alternate_root / alternate_evaluator_binding["path"]
    evaluator_path.parent.mkdir(parents=True, exist_ok=True)
    evaluator_path.write_bytes(alternate_evaluator)
    manifest_path = alternate_root / seal_payload["study_manifest"]["path"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(alternate_manifest)

    monkeypatch.setattr(protocol, "_ROOT", alternate_root)
    alternate_record_bytes = prepare_initial_t2_packet(
        baseline_reward_bytes=BASELINE,
        t2_seal_bytes=alternate_seal,
    )
    alternate_record = parse_model_bytes(
        alternate_record_bytes, T2InitialPacketRecord, max_bytes=131_072
    )
    assert alternate_record.t2_seal.sha256 != original_record.t2_seal.sha256
    assert alternate_record.call_identity_sha256 != original_record.call_identity_sha256
    assert render_initial_t2_prompt(alternate_record_bytes) == original_prompt


def test_valid_local_envelope_retains_raw_bytes_and_emits_honest_recipe(
    tmp_path: Path,
) -> None:
    record_bytes, run = _envelope(tmp_path)
    record = parse_model_bytes(record_bytes, T2InitialPacketRecord, max_bytes=131_072)
    original = {name: (run / name).read_bytes() for name in RUN_FILES}
    outcome = ingest_initial_t2_sol_run(record_bytes, run)
    receipt = _load_receipt(outcome.receipt.path)
    assert outcome.accepted and outcome.proposal is not None and outcome.recipe is not None
    assert len(outcome.retained_run_artifacts) == 4
    assert {item.path.read_bytes() for item in outcome.retained_run_artifacts} == set(
        original.values()
    )
    assert record.call_identity_sha256.encode() in original["request.json"]
    assert record.call_identity_sha256.encode() in original["result.json"]
    assert json.loads(outcome.recipe.path.read_bytes()) == {
        "formula_id": "target_speed_triangular_affine/v1",
        "alpha": 1.5,
        "beta": -0.25,
    }
    assert receipt.local_envelope_consistency == "verified"
    assert receipt.authenticated_model_origin == "not_attested"
    assert receipt.candidate_reward_admission == "missing"
    assert receipt.runtime_authorization == "not_authorized"
    assert receipt.training_authorization == "not_authorized"
    assert receipt.measured_improvement == "not_measured"
    assert receipt.expected_model == "gpt-5.6-sol"
    assert receipt.expected_reasoning_effort == "max"
    assert receipt.expected_runner_kind == "screen"
    assert receipt.expected_checkout_realpath == EXPECTED_CHECKOUT_REALPATH
    assert receipt.expected_branch == EXPECTED_BRANCH
    assert receipt.expected_import_origin == EXPECTED_IMPORT_ORIGIN
    assert receipt.expected_preparation_owner == EXPECTED_PREPARATION_OWNER
    assert receipt.expected_preparation_role == EXPECTED_PREPARATION_ROLE
    assert receipt.expected_preparation_scope == EXPECTED_PREPARATION_SCOPE
    assert receipt.expected_owner == "fable-f3-initial"
    assert receipt.expected_role == "candidate"
    assert receipt.expected_scope == record.expected_scope
    assert receipt.call_id == record.call_identity.call_id
    assert receipt.call_identity_sha256 == record.call_identity_sha256
    assert receipt.dispatch_intent_consistency == "verified"
    assert receipt.dispatch_intent_sha256 is not None
    assert receipt.ingestion_claim_sha256 is not None
    assert not receipt.replay_refusal
    assert receipt.metadata_semantics == "expected_configuration_not_served_model_attestation"
    receipt_payload = json.loads(outcome.receipt.path.read_bytes())
    assert "requested_model" not in receipt_payload
    assert "observed_model" not in receipt_payload


def test_ingestion_without_canonical_intent_is_refused_with_retained_receipt(
    tmp_path: Path,
) -> None:
    record_bytes, run = _envelope(tmp_path, publish_intent=False)
    outcome = ingest_initial_t2_sol_run(record_bytes, run)
    receipt = _load_receipt(outcome.receipt.path)
    assert not outcome.accepted
    assert len(outcome.retained_run_artifacts) == 4
    assert receipt.dispatch_intent_consistency == "missing"
    assert receipt.dispatch_intent_sha256 is None
    assert receipt.ingestion_claim_sha256 is None
    assert "dispatch intent is missing" in (outcome.rejection_reason or "")


def test_ingestion_with_mismatched_canonical_intent_is_refused_with_retained_receipt(
    tmp_path: Path,
) -> None:
    record_bytes, run = _envelope(tmp_path)
    intent_path = protocol.CALL_ROOT / DISPATCH_INTENT_FILENAME
    payload = json.loads(intent_path.read_bytes())
    payload["record_sha256"] = "0" * 64
    intent_path.write_bytes(finite_pretty_json(payload))
    intent_path.chmod(0o600)

    outcome = ingest_initial_t2_sol_run(record_bytes, run)
    receipt = _load_receipt(outcome.receipt.path)
    assert not outcome.accepted
    assert len(outcome.retained_run_artifacts) == 4
    assert receipt.dispatch_intent_consistency == "mismatched"
    assert receipt.dispatch_intent_sha256 == hashlib.sha256(intent_path.read_bytes()).hexdigest()
    assert receipt.ingestion_claim_sha256 is None
    assert "dispatch intent binding differs" in (outcome.rejection_reason or "")


def test_second_valid_run_is_refused_with_retained_receipt(tmp_path: Path) -> None:
    first_record, first_run = _envelope(tmp_path, run_name="first-valid-run")
    second_record, second_run = _envelope(tmp_path, run_name="second-valid-run")
    assert first_record == second_record

    first = ingest_initial_t2_sol_run(first_record, first_run)
    second = ingest_initial_t2_sol_run(second_record, second_run)
    first_receipt = _load_receipt(first.receipt.path)
    second_receipt = _load_receipt(second.receipt.path)
    assert first.accepted
    assert not second.accepted
    assert len(second.retained_run_artifacts) == 4
    assert second_receipt.replay_refusal
    assert second_receipt.ingestion_claim_sha256 == first_receipt.ingestion_claim_sha256
    assert "second run or provider replay refused" in (second.rejection_reason or "")


def test_provider_replay_of_same_run_is_refused_with_retained_receipt(tmp_path: Path) -> None:
    record_bytes, run = _envelope(tmp_path)
    first = ingest_initial_t2_sol_run(record_bytes, run)
    replay = ingest_initial_t2_sol_run(record_bytes, run)
    replay_receipt = _load_receipt(replay.receipt.path)
    assert first.accepted
    assert not replay.accepted
    assert len(replay.retained_run_artifacts) == 4
    assert replay_receipt.replay_refusal
    assert replay.receipt.path.is_file()


RUN_FILES = ("request.json", "task-packet.md", "result.json", "final.txt")


@pytest.mark.parametrize(
    ("request_updates", "result_updates", "packet"),
    [
        ({"prompt_sha256": "0" * 64}, {}, None),
        ({"owner": "astra-f3-initial"}, {}, None),
        ({"scope": "t2-initial-parameter-hypothesis-only"}, {}, None),
        ({}, {"scope": "t2-initial-parameter-hypothesis-only"}, None),
        ({}, {"status": "FAILED", "exit_code": 1}, None),
        ({}, {"termination_escalated": True}, None),
        ({}, {"thread_id": None}, None),
        ({}, {"finished_at_utc": "2026-09-05T23:20:01Z"}, None),
        ({}, {}, b"altered packet"),
    ],
)
def test_altered_envelopes_are_rejected_after_raw_retention(
    tmp_path: Path,
    request_updates: dict[str, object],
    result_updates: dict[str, object],
    packet: bytes | None,
) -> None:
    record_bytes, run = _envelope(
        tmp_path,
        request_updates=request_updates,
        result_updates=result_updates,
        packet=packet,
    )
    outcome = ingest_initial_t2_sol_run(record_bytes, run)
    receipt = _load_receipt(outcome.receipt.path)
    assert not outcome.accepted and outcome.proposal is None and outcome.recipe is None
    assert len(outcome.retained_run_artifacts) == 4
    assert all(item.state == "retained" for item in receipt.run_artifacts)
    assert receipt.local_envelope_consistency == "not_verified"


@pytest.mark.parametrize(
    ("started", "finished", "accepted"),
    [
        ("0001-01-01T00:00:00Z", "0001-01-01T00:00:00Z", True),
        ("9999-12-31T23:59:59Z", "9999-12-31T23:59:59Z", True),
        ("2026-12-31T23:59:50Z", "2027-01-01T00:00:10Z", True),
        ("2026-09-05T23:00:00Z", "2026-09-05T23:20:00Z", True),
        ("2026-09-05T23:00:01Z", "2026-09-05T23:00:00Z", False),
        ("2026-09-05T23:00:00Z", "2026-09-05T23:20:01Z", False),
    ],
)
def test_timestamp_duration_boundaries_emit_an_outcome_receipt(
    tmp_path: Path, started: str, finished: str, accepted: bool
) -> None:
    record_bytes, run = _envelope(
        tmp_path,
        request_updates={"created_at_utc": started},
        result_updates={"finished_at_utc": finished},
    )
    outcome = ingest_initial_t2_sol_run(record_bytes, run)
    receipt = _load_receipt(outcome.receipt.path)
    assert outcome.accepted is accepted
    assert len(outcome.retained_run_artifacts) == 4
    assert all(item.state == "retained" for item in receipt.run_artifacts)
    if accepted:
        assert outcome.rejection_reason is None
    else:
        assert "1,200-second deadline" in (outcome.rejection_reason or "")


@pytest.mark.parametrize(
    ("request_updates", "result_updates"),
    [
        ({"created_at_utc": "10000-01-01T00:00:00Z"}, {}),
        ({}, {"finished_at_utc": "not-a-timestamp"}),
    ],
)
def test_malformed_timestamps_are_normalized_after_raw_retention(
    tmp_path: Path,
    request_updates: dict[str, object],
    result_updates: dict[str, object],
) -> None:
    record_bytes, run = _envelope(
        tmp_path,
        request_updates=request_updates,
        result_updates=result_updates,
    )
    outcome = ingest_initial_t2_sol_run(record_bytes, run)
    receipt = _load_receipt(outcome.receipt.path)
    assert not outcome.accepted
    assert (outcome.rejection_reason or "").startswith("invalid ")
    assert len(outcome.retained_run_artifacts) == 4
    assert all(item.state == "retained" for item in receipt.run_artifacts)


def test_mismatched_model_and_malformed_request_do_not_invent_observations(
    tmp_path: Path,
) -> None:
    cases = tmp_path / "cases"
    cases.mkdir()
    (cases / "mismatch").mkdir()
    (cases / "malformed").mkdir()
    mismatch_record, mismatch_run = _envelope(
        cases / "mismatch", request_updates={"requested_model": "gpt-5.6-luna"}
    )
    malformed_record, malformed_run = _envelope(cases / "malformed")
    (malformed_run / "request.json").write_bytes(b"{")
    (malformed_run / "request.json").chmod(0o600)

    for record_bytes, run in ((mismatch_record, mismatch_run), (malformed_record, malformed_run)):
        outcome = ingest_initial_t2_sol_run(record_bytes, run)
        payload = json.loads(outcome.receipt.path.read_bytes())
        assert not outcome.accepted
        assert payload["expected_model"] == "gpt-5.6-sol"
        assert payload["metadata_semantics"] == (
            "expected_configuration_not_served_model_attestation"
        )
        assert "requested_model" not in payload
        assert "observed_model" not in payload
        request_binding = next(
            item for item in payload["run_artifacts"] if item["name"] == "request.json"
        )
        retained = protocol.CALL_ROOT / "records" / request_binding["retained_filename"]
        assert retained.read_bytes() == (run / "request.json").read_bytes()


@pytest.mark.parametrize(
    ("field", "astra_value"),
    [
        ("expected_owner", "astra-f3-initial"),
        (
            "expected_checkout_realpath",
            "/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra",
        ),
    ],
)
def test_astra_configuration_receipts_are_rejected(
    tmp_path: Path,
    field: str,
    astra_value: str,
) -> None:
    payload = _accepted_receipt_payload(tmp_path)
    payload[field] = astra_value
    with pytest.raises(RewardSearchError):
        parse_model_bytes(finite_pretty_json(payload), T2ModelCallReceipt, max_bytes=65_536)


def test_receipt_with_mismatched_baseline_hash_is_rejected(tmp_path: Path) -> None:
    payload = _accepted_receipt_payload(tmp_path)
    payload["baseline_sha256"] = "0" * 64
    with pytest.raises(RewardSearchError):
        parse_model_bytes(finite_pretty_json(payload), T2ModelCallReceipt, max_bytes=65_536)


def test_receipt_with_mismatched_call_identity_is_rejected(tmp_path: Path) -> None:
    payload = _accepted_receipt_payload(tmp_path)
    payload["call_identity_sha256"] = "0" * 64
    with pytest.raises(RewardSearchError):
        parse_model_bytes(finite_pretty_json(payload), T2ModelCallReceipt, max_bytes=65_536)


@pytest.mark.parametrize("accepted", [False, True])
@pytest.mark.parametrize(
    ("proposal_present", "recipe_present"),
    [(False, False), (False, True), (True, False), (True, True)],
)
def test_receipt_parser_requires_all_or_none_proposal_identities(
    tmp_path: Path,
    accepted: bool,
    proposal_present: bool,
    recipe_present: bool,
) -> None:
    payload = _accepted_receipt_payload(tmp_path)
    proposal_sha = payload["proposal_sha256"]
    recipe_sha = payload["candidate_recipe_sha256"]
    payload["disposition"] = (
        "parameter_proposal_format_accepted" if accepted else "parameter_proposal_rejected"
    )
    payload["proposal_sha256"] = proposal_sha if proposal_present else None
    payload["candidate_recipe_sha256"] = recipe_sha if recipe_present else None
    payload["rejection_reason"] = None if accepted else "forced rejection"
    payload["format_validation"] = "accepted_hypothesis_only" if accepted else "not_admitted"
    valid = (accepted and proposal_present and recipe_present) or (
        not accepted and not proposal_present and not recipe_present
    )
    if valid:
        parse_model_bytes(finite_pretty_json(payload), T2ModelCallReceipt, max_bytes=65_536)
    else:
        with pytest.raises(RewardSearchError):
            parse_model_bytes(finite_pretty_json(payload), T2ModelCallReceipt, max_bytes=65_536)


@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize(
    ("filename_present", "sha_present", "count_present"),
    [
        (False, False, False),
        (False, False, True),
        (False, True, False),
        (False, True, True),
        (True, False, False),
        (True, False, True),
        (True, True, False),
        (True, True, True),
    ],
)
def test_run_artifact_parser_requires_all_or_none_byte_identity(
    retained: bool,
    filename_present: bool,
    sha_present: bool,
    count_present: bool,
) -> None:
    payload = {
        "name": "request.json",
        "state": "retained" if retained else "missing",
        "retained_filename": "retained.bin" if filename_present else None,
        "sha256": "1" * 64 if sha_present else None,
        "byte_count": 1 if count_present else None,
        "rejection_detail": None if retained else "forced missing state",
    }
    valid = (retained and filename_present and sha_present and count_present) or (
        not retained and not filename_present and not sha_present and not count_present
    )
    if valid:
        parse_model_bytes(finite_pretty_json(payload), T2RunArtifactBinding, max_bytes=65_536)
    else:
        with pytest.raises(RewardSearchError):
            parse_model_bytes(finite_pretty_json(payload), T2RunArtifactBinding, max_bytes=65_536)


@pytest.mark.parametrize(
    ("name", "operation", "state"),
    [
        ("request.json", "missing", "missing"),
        ("task-packet.md", "empty", "empty"),
        ("result.json", "directory", "nonregular"),
        ("final.txt", "oversized", "oversized"),
    ],
)
def test_run_file_read_failures_have_explicit_unretained_states(
    tmp_path: Path, name: str, operation: str, state: str
) -> None:
    record_bytes, run = _envelope(tmp_path)
    target = run / name
    target.unlink()
    if operation == "empty":
        target.write_bytes(b"")
        target.chmod(0o600)
    elif operation == "directory":
        target.mkdir()
    elif operation == "oversized":
        target.write_bytes(b"x" * (131_073 if name == "task-packet.md" else 65_537))
        target.chmod(0o600)
    outcome = ingest_initial_t2_sol_run(record_bytes, run)
    receipt = _load_receipt(outcome.receipt.path)
    binding = next(item for item in receipt.run_artifacts if item.name == name)
    assert not outcome.accepted and binding.state == state
    assert binding.sha256 is None and binding.byte_count is None
    assert len(outcome.retained_run_artifacts) == 3


def test_wrong_baseline_is_refused() -> None:
    assert len(OLD_BASELINE) == 1_144
    assert hashlib.sha256(OLD_BASELINE).hexdigest() == (
        "0986d4fc907e94185e14f58c9ef6eabf8ec26a8f336c68ee450bbacd5d8224d4"
    )
    for baseline in (OLD_BASELINE, BASELINE + b"\n", b"x" * len(BASELINE), b""):
        with pytest.raises(T2ProtocolError, match="pinned tracking-only"):
            prepare_initial_t2_packet(
                baseline_reward_bytes=baseline,
                t2_seal_bytes=T2_SEAL,
            )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("semantic_request", "target_speed_m_s"), 2.0),
        (("semantic_request", "control_period_seconds"), 0.02),
        (("t2_contract_sha256",), "0" * 64),
        (("call_identity", "call_id"), "0" * 64),
        (("call_identity_sha256",), "0" * 64),
        (("contracts", "evaluator_source", "sha256"), "0" * 64),
        (("evidence_dossier", "aggregate_feedback"), [{"fabricated": True}]),
        (("parent_recipe",), {"alpha": 1.0, "beta": 0.0}),
    ],
)
def test_reserialized_record_mutations_are_not_trusted(
    path: tuple[str, ...], value: object
) -> None:
    record_bytes, _ = _record()
    payload = json.loads(record_bytes)
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(T2ProtocolError):
        render_initial_t2_prompt(finite_pretty_json(payload))


@pytest.mark.parametrize(
    "updates",
    [
        {"extra": True},
        {"parameters": {"alpha": True, "beta": 0.0}},
        {"parameters": {"alpha": "1", "beta": 0.0}},
        {"parameters": {"alpha": 4.1, "beta": 0.0}},
        {"parameters": {"alpha": 1.0, "beta": -10.1}},
        {"parameters": {"alpha": float("nan"), "beta": 0.0}},
        {"status": "admitted"},
        {"runtime_id": "candidate-controlled"},
    ],
)
def test_invalid_parameter_responses_are_retained_and_rejected(
    tmp_path: Path, updates: dict[str, object]
) -> None:
    record_bytes, record = _record()
    encoded = _proposal(record, **updates)
    run_record, run = _envelope(tmp_path, proposal=encoded)
    assert run_record == record_bytes
    outcome = ingest_initial_t2_sol_run(record_bytes, run)
    receipt = _load_receipt(outcome.receipt.path)
    final_binding = next(item for item in receipt.run_artifacts if item.name == "final.txt")
    assert not outcome.accepted and final_binding.state == "retained"
    assert receipt.local_envelope_consistency == "verified"
    assert receipt.result_thread_id == "fixture-thread"
    assert final_binding.sha256 == hashlib.sha256(encoded).hexdigest()
    retained_final = next(
        item for item in outcome.retained_run_artifacts if item.sha256 == final_binding.sha256
    )
    assert retained_final.path.read_bytes() == encoded


def test_duplicate_key_and_deep_json_responses_are_normalized_rejections(tmp_path: Path) -> None:
    _, record = _record()
    valid = json.loads(_proposal(record))
    duplicate = _proposal(record).replace(b'"status":', b'"status":"x","status":', 1)
    deep = json.dumps({"x": None})
    for _ in range(1_100):
        deep = '{"x":' + deep + "}"
    for index, encoded in enumerate((duplicate, deep.encode())):
        case = tmp_path / str(index)
        case.mkdir()
        run_record, run = _envelope(case, proposal=encoded)
        outcome = ingest_initial_t2_sol_run(run_record, run)
        assert not outcome.accepted
        assert len(outcome.rejection_reason or "") < 256
    assert valid["kind"] == "t2_initial_parameter_proposal"


def test_exact_bytes_and_path_boundaries_reject_every_argument_before_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BytesSubclass(bytes):
        pass

    class PathSubclass(type(Path())):
        pass

    record_bytes, record = _record()
    _, run = _envelope(tmp_path)
    packet_path = tmp_path / "packet.md"
    record_path = tmp_path / "record.json"
    intent_path = protocol.CALL_ROOT / DISPATCH_INTENT_FILENAME
    calls: list[str] = []

    def trap(*args: object, **kwargs: object) -> object:
        calls.append("called")
        raise AssertionError("callback reached")

    monkeypatch.setattr(
        "oracle_composition.reward_search.t2_model_protocol._read_bounded_regular_file", trap
    )
    monkeypatch.setattr(
        "oracle_composition.reward_search.t2_model_protocol.publish_bytes_without_overwrite", trap
    )

    invalid_calls = [
        lambda: prepare_initial_t2_packet(
            baseline_reward_bytes=BytesSubclass(BASELINE), t2_seal_bytes=T2_SEAL
        ),
        lambda: prepare_initial_t2_packet(
            baseline_reward_bytes=bytearray(BASELINE), t2_seal_bytes=T2_SEAL
        ),
        lambda: prepare_initial_t2_packet(
            baseline_reward_bytes=BASELINE, t2_seal_bytes=BytesSubclass(T2_SEAL)
        ),
        lambda: prepare_initial_t2_packet(
            baseline_reward_bytes=BASELINE, t2_seal_bytes=bytearray(T2_SEAL)
        ),
        lambda: render_initial_t2_prompt(BytesSubclass(record_bytes)),
        lambda: render_initial_t2_prompt(record),
        lambda: publish_initial_t2_packet(BytesSubclass(record_bytes), packet_path, record_path),
        lambda: publish_initial_t2_packet(record_bytes, PathSubclass(packet_path), record_path),
        lambda: publish_initial_t2_packet(record_bytes, "packet.md", record_path),
        lambda: publish_initial_t2_packet(record_bytes, packet_path, PathSubclass(record_path)),
        lambda: publish_initial_t2_packet(record_bytes, packet_path, "record.json"),
        lambda: publish_initial_t2_dispatch_intent(BytesSubclass(record_bytes), intent_path),
        lambda: publish_initial_t2_dispatch_intent(record_bytes, PathSubclass(intent_path)),
        lambda: publish_initial_t2_dispatch_intent(record_bytes, "dispatch-intent.json"),
        lambda: ingest_initial_t2_sol_run(BytesSubclass(record_bytes), run),
        lambda: ingest_initial_t2_sol_run(record, run),
        lambda: ingest_initial_t2_sol_run(record_bytes, PathSubclass(run)),
        lambda: ingest_initial_t2_sol_run(record_bytes, str(run)),
    ]
    for invalid_call in invalid_calls:
        with pytest.raises(T2ProtocolError, match=r"exact (builtin bytes|native Path)"):
            invalid_call()
    assert calls == []
    assert not packet_path.exists()
    assert not record_path.exists()


def test_invalid_record_fails_before_reader_or_publication_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def trap(*args: object, **kwargs: object) -> object:
        calls.append("called")
        raise AssertionError("callback reached")

    monkeypatch.setattr(
        "oracle_composition.reward_search.t2_model_protocol._read_bounded_regular_file", trap
    )
    monkeypatch.setattr(
        "oracle_composition.reward_search.t2_model_protocol.publish_bytes_without_overwrite", trap
    )
    with pytest.raises(T2ProtocolError):
        ingest_initial_t2_sol_run(b"{}", tmp_path / "run")
    assert calls == []


def test_packet_publication_uses_no_overwrite(tmp_path: Path) -> None:
    record_bytes, _ = _record()
    packet = tmp_path / "packet.md"
    record = tmp_path / "record.json"
    published = publish_initial_t2_packet(record_bytes, packet, record)
    assert published.packet.path.read_bytes() == render_initial_t2_prompt(record_bytes)
    assert published.record.path.read_bytes() == record_bytes
    with pytest.raises(ValueError, match="overwrite"):
        publish_initial_t2_packet(record_bytes, packet, tmp_path / "other.json")


def test_second_canonical_call_identity_claim_is_refused_with_receipt() -> None:
    record_bytes, record = _record()
    intent_path = protocol.CALL_ROOT / DISPATCH_INTENT_FILENAME
    published = publish_initial_t2_dispatch_intent(record_bytes, intent_path)
    original = intent_path.read_bytes()
    intent = parse_model_bytes(original, T2InitialDispatchIntent, max_bytes=65_536)

    assert published.path == intent_path
    assert intent.maximum_initial_calls == 1
    assert intent.attempts_remaining_after_intent == 0
    assert intent.revision_calls_authorized == 0
    assert intent.retries_authorized == 0
    assert intent.deadline_seconds_from_request_creation == 1_200
    assert intent.record_sha256 == hashlib.sha256(record_bytes).hexdigest()
    assert intent.rendered_prompt_sha256 == record.rendered_prompt_sha256
    assert intent.call_id == derive_t2_call_id(
        study_manifest_sha256=record.study_manifest_sha256,
        rendered_prompt_sha256=record.rendered_prompt_sha256,
    )
    assert intent.call_id != derive_t2_call_id(
        study_manifest_sha256="0" * 64,
        rendered_prompt_sha256=record.rendered_prompt_sha256,
    )
    assert intent.call_id != derive_t2_call_id(
        study_manifest_sha256=record.study_manifest_sha256,
        rendered_prompt_sha256="0" * 64,
    )
    assert intent.call_identity_sha256 == record.call_identity_sha256
    assert intent.expected_scope.endswith(record.call_identity_sha256)
    assert intent.expected_owner == "fable-f3-initial"
    assert intent.expected_preparation_owner == "fable-f3-prepare"
    identity_path = protocol.CALL_ROOT / CALL_IDENTITY_FILENAME
    assert hashlib.sha256(identity_path.read_bytes()).hexdigest() == record.call_identity_sha256

    with pytest.raises(T2ProtocolError, match="call identity claim was refused"):
        publish_initial_t2_dispatch_intent(record_bytes, intent_path)
    assert intent_path.read_bytes() == original
    refusals = list((protocol.CALL_ROOT / "refusals").glob("*.json"))
    assert len(refusals) == 1
    refusal = parse_model_bytes(
        refusals[0].read_bytes(), T2ProtocolRefusalReceipt, max_bytes=65_536
    )
    assert refusal.reason_code == "canonical_call_identity_already_claimed"
    assert refusal.repository_claim.endswith("external_call_absence_not_proven")


def test_sibling_dispatch_intent_path_is_refused_with_receipt() -> None:
    record_bytes, _record_value = _record()
    sibling = protocol.CALL_ROOT / "sibling-dispatch-intent.json"
    with pytest.raises(T2ProtocolError, match="not the canonical"):
        publish_initial_t2_dispatch_intent(record_bytes, sibling)
    assert not sibling.exists()
    assert not (protocol.CALL_ROOT / CALL_IDENTITY_FILENAME).exists()
    refusals = list((protocol.CALL_ROOT / "refusals").glob("*.json"))
    assert len(refusals) == 1
    refusal = parse_model_bytes(
        refusals[0].read_bytes(), T2ProtocolRefusalReceipt, max_bytes=65_536
    )
    assert refusal.reason_code == "noncanonical_dispatch_intent_path"


def test_conflicting_canonical_identity_is_refused_with_receipt() -> None:
    record_bytes, _record_value = _record()
    identity_path = protocol.CALL_ROOT / CALL_IDENTITY_FILENAME
    identity_path.parent.mkdir(parents=True)
    identity_path.write_bytes(b"{}")
    identity_path.chmod(0o600)

    with pytest.raises(T2ProtocolError, match="call identity claim was refused"):
        publish_initial_t2_dispatch_intent(
            record_bytes,
            protocol.CALL_ROOT / DISPATCH_INTENT_FILENAME,
        )
    refusals = list((protocol.CALL_ROOT / "refusals").glob("*.json"))
    refusal = parse_model_bytes(
        refusals[0].read_bytes(), T2ProtocolRefusalReceipt, max_bytes=65_536
    )
    assert refusal.reason_code == "canonical_call_identity_conflict"


def test_preexisting_canonical_intent_is_refused_with_receipt() -> None:
    record_bytes, _record_value = _record()
    intent_path = protocol.CALL_ROOT / DISPATCH_INTENT_FILENAME
    intent_path.parent.mkdir(parents=True)
    intent_path.write_bytes(b"{}")
    intent_path.chmod(0o600)

    with pytest.raises(T2ProtocolError, match="intent publication was refused"):
        publish_initial_t2_dispatch_intent(record_bytes, intent_path)
    refusals = list((protocol.CALL_ROOT / "refusals").glob("*.json"))
    refusal = parse_model_bytes(
        refusals[0].read_bytes(), T2ProtocolRefusalReceipt, max_bytes=65_536
    )
    assert refusal.reason_code == "canonical_dispatch_intent_already_exists"


def test_t2_response_and_receipt_do_not_cross_parse_a1_or_f1(tmp_path: Path) -> None:
    _, record = _record()
    response = _proposal(record)
    with pytest.raises(RewardSearchError):
        parse_model_bytes(response, RewardProposal, max_bytes=65_536)
    with pytest.raises(RewardSearchError):
        parse_model_bytes(response, FormulaProposalV2, max_bytes=65_536)

    run_record, run = _envelope(tmp_path, proposal=response)
    outcome = ingest_initial_t2_sol_run(run_record, run)
    receipt_bytes = outcome.receipt.path.read_bytes()
    with pytest.raises(RewardSearchError):
        parse_model_bytes(receipt_bytes, ModelCallReceipt, max_bytes=65_536)
    with pytest.raises(RewardSearchError):
        parse_model_bytes(receipt_bytes, FormulaIngestionReceiptV2, max_bytes=65_536)
    receipt = parse_model_bytes(receipt_bytes, T2ModelCallReceipt, max_bytes=65_536)
    assert receipt.kind == "t2_initial_model_call_receipt"


def test_accepted_source_files_remain_unchanged() -> None:
    expected = {
        "src/oracle_composition/rewards/target_speed_formula.py": RECIPE_SOURCE_SHA256,
        "src/oracle_composition/rewards/target_speed_formula_t2.py": EVALUATOR_SOURCE_SHA256,
        "src/oracle_composition/rewards/task_inputs_v2.py": TASK_INPUTS_SOURCE_SHA256,
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest


def test_t2_models_are_frozen() -> None:
    _, record = _record()
    proposal = parse_model_bytes(_proposal(record), T2ParameterProposal, max_bytes=65_536)
    with pytest.raises(Exception, match="frozen"):
        proposal.status = "changed"  # type: ignore[misc]
