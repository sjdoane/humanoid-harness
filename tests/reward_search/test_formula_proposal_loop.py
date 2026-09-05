from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from oracle_composition.reward_search.contracts import (
    AggregateMeasurement,
    EvidenceProvenance,
    MissingEvidence,
    RewardProposal,
    SourceBundle,
    SourceCard,
)
from oracle_composition.reward_search.formula_contracts import (
    RESPONSE_ORIGIN,
    FormulaEvidenceDossierV2,
    FormulaIngestionReceiptV2,
    FormulaIterationRecordV2,
    FormulaProposalV2,
)
from oracle_composition.reward_search.formula_loop import (
    FormulaSearchError,
    create_formula_bindings,
    formula_bindings_sha256,
    formula_model_sha256,
    frozen_configuration_sha256,
    ingest_formula_response,
    prepare_formula_revision,
    prepare_initial_formula_packet,
    render_formula_packet,
    replay_formula_response_bytes,
)
from oracle_composition.reward_search.loop import RewardSearchError, parse_model_bytes
from oracle_composition.reward_search.publication import finite_pretty_json
from oracle_composition.rewards.target_speed_formula import (
    FORMULA_ID,
    TargetSpeedFormulaRecipeV1,
)

ROOT = Path(__file__).parents[2]
HASH_A = "a" * 64


def _recipe(alpha: float = 1.0, beta: float = 0.0) -> TargetSpeedFormulaRecipeV1:
    return TargetSpeedFormulaRecipeV1(FORMULA_ID, alpha, beta)


def _bindings():
    return create_formula_bindings(
        config_bytes=(
            ROOT / "experiments/family_b_target_speed_formula_v1/config.json"
        ).read_bytes(),
        trusted_formula_implementation_bytes=(
            ROOT / "src/oracle_composition/rewards/target_speed_formula.py"
        ).read_bytes(),
        read_contract_bytes=(ROOT / "src/oracle_composition/rewards/contract.py").read_bytes(),
        independent_evaluator_id="synthetic-independent-evaluator/v1",
        independent_evaluator_bytes=b"synthetic evaluator identity; not executed\n",
        frozen_configuration_bytes={
            "frozen/trainer.json": b'{"trainer":"fixed","seeds":[101,202]}\n',
            "frozen/task.json": b'{"task":"target-speed"}\n',
        },
    )


def _sources() -> SourceBundle:
    return SourceBundle(
        schema_version=1,
        kind="reward_source_bundle",
        graph_schema_version=1,
        graph_corpus_sha256=HASH_A,
        normalized_query="target speed reward shaping",
        limit=2,
        source_cards=[
            SourceCard(
                source_id="source:1",
                paper_id="paper-1",
                kind="evidence",
                title="Bounded source",
                excerpt="A mechanism claim, not measured task evidence.",
                locator="synthetic-section",
                source_url=None,
            )
        ],
        synthetic=True,
    )


def _dossier(parent: TargetSpeedFormulaRecipeV1, *, measured: bool = False):
    bindings = _bindings()
    provenance = (
        [
            EvidenceProvenance(
                provenance_id="protected-eval-1",
                provenance_class="independent_protected_evaluator",
                locator="synthetic://aggregate",
                artifact_sha256=bindings.independent_evaluator.sha256,
                synthetic=True,
            )
        ]
        if measured
        else []
    )
    feedback = (
        [
            AggregateMeasurement(
                evidence_id="target-hold",
                value=0.25,
                unit="score",
                provenance_id="protected-eval-1",
            )
        ]
        if measured
        else []
    )
    missing = (
        [] if measured else [MissingEvidence(evidence_id="target-hold", reason="not measured")]
    )
    return FormulaEvidenceDossierV2(
        schema_version=2,
        kind="target_speed_formula_evidence_dossier",
        task_id="f1-target-speed",
        task_statement="Parameterize the frozen COM-x target-speed term.",
        parent_recipe_sha256=hashlib.sha256(parent.canonical_bytes).hexdigest(),
        independent_evaluator_sha256=bindings.independent_evaluator.sha256,
        frozen_configuration_sha256=frozen_configuration_sha256(bindings),
        evidence_provenance=provenance,
        aggregate_feedback=feedback,
        missing_evidence=missing,
        synthetic=True,
    )


def _packet():
    recipe = _recipe()
    return prepare_initial_formula_packet(recipe, _dossier(recipe), _sources(), _bindings())


def _response(record, *, recipe: dict[str, object] | None = None, **updates: object) -> bytes:
    value: dict[str, object] = {
        "schema_version": 2,
        "kind": "target_speed_formula_proposal",
        "request_payload_sha256": record.request_payload_sha256,
        "parent_recipe_sha256": record.manifest.parent_recipe_sha256,
        "dossier_sha256": record.manifest.dossier_sha256,
        "source_bundle_sha256": record.manifest.source_bundle_sha256,
        "bindings_sha256": record.manifest.bindings_sha256,
        "recipe": recipe or {"formula_id": FORMULA_ID, "alpha": 1.25, "beta": -0.5},
        "rationale": "A bounded synthetic fixture proposal.",
        "predicted_effect": "The protected score may change.",
        "falsifier": "No gain under the predetermined evaluator.",
        "cited_source_ids": ["source:1"],
    }
    value.update(updates)
    return finite_pretty_json(value)


def _prompt_section_bytes(prompt: bytes, heading: str) -> bytes:
    marker = f"## {heading}\n"
    remainder = prompt.decode("utf-8").split(marker, 1)[1]
    encoded = remainder.split("\n## ", 1)[0]
    return encoded.encode("utf-8")


def _prompt_json_section(prompt: bytes, heading: str) -> dict[str, object]:
    value = json.loads(_prompt_section_bytes(prompt, heading))
    assert type(value) is dict
    return value


def _response_from_prompt(prompt: bytes) -> bytes:
    payload = _prompt_json_section(prompt, "Canonical request payload")
    identity = _prompt_json_section(prompt, "Required response identity")
    manifest = payload["manifest"]
    assert type(manifest) is dict
    source_bundle = payload["bounded_source_bundle"]
    assert type(source_bundle) is dict
    source_cards = source_bundle["source_cards"]
    assert type(source_cards) is list and source_cards
    source_card = source_cards[0]
    assert type(source_card) is dict
    parent_recipe = payload["parent_recipe"]
    assert type(parent_recipe) is dict
    response_schema = payload["required_response_schema"]
    assert type(response_schema) is dict
    properties = response_schema["properties"]
    assert type(properties) is dict
    schema_version = properties["schema_version"]
    proposal_kind = properties["kind"]
    assert type(schema_version) is dict and type(proposal_kind) is dict
    return finite_pretty_json(
        {
            "schema_version": schema_version["const"],
            "kind": proposal_kind["const"],
            "request_payload_sha256": identity["request_payload_sha256"],
            "parent_recipe_sha256": manifest["parent_recipe_sha256"],
            "dossier_sha256": manifest["dossier_sha256"],
            "source_bundle_sha256": manifest["source_bundle_sha256"],
            "bindings_sha256": manifest["bindings_sha256"],
            "recipe": {
                "formula_id": parent_recipe["formula_id"],
                "alpha": 1.25,
                "beta": -0.5,
            },
            "rationale": "A bounded synthetic fixture proposal.",
            "predicted_effect": "The protected score may change.",
            "falsifier": "No gain under the predetermined evaluator.",
            "cited_source_ids": [source_card["source_id"]],
        }
    )


def _load(path: Path, model):
    return model.model_validate_json(path.read_bytes(), strict=True)


def _accepted(tmp_path: Path):
    record = _packet()
    outcome = ingest_formula_response(
        record,
        _response(record),
        tmp_path / "records",
        response_id="synthetic-initial",
    )
    assert outcome.proposal is not None
    return (
        record,
        outcome.response,
        _load(outcome.proposal.path, FormulaProposalV2),
        _load(outcome.receipt.path, FormulaIngestionReceiptV2),
        _load(outcome.iteration.path, FormulaIterationRecordV2),
    )


def test_prepare_ingest_and_revise_are_deterministic_and_retain_separate_lineage(
    tmp_path: Path,
) -> None:
    record = _packet()
    repeated = _packet()
    assert render_formula_packet(record) == render_formula_packet(repeated)
    assert record.request_payload_sha256 == repeated.request_payload_sha256
    assert record.rendered_prompt_sha256 == repeated.rendered_prompt_sha256
    rendered_prompt = render_formula_packet(record)
    assert hashlib.sha256(rendered_prompt).hexdigest() == record.rendered_prompt_sha256
    assert len(rendered_prompt) == record.rendered_prompt_byte_count
    prompt = rendered_prompt.decode()
    assert "only authorable data are formula_id, alpha, and beta" in prompt
    assert "authorizes no runtime, calibration, or training" in prompt
    assert "root_x_velocity_m_s" in prompt
    assert "explicitly_absent_pending_peer_resolution" in prompt

    outcome = ingest_formula_response(
        record,
        _response(record),
        tmp_path / "records",
        response_id="synthetic-initial",
    )
    assert outcome.accepted is True and outcome.proposal is not None
    proposal = _load(outcome.proposal.path, FormulaProposalV2)
    receipt = _load(outcome.receipt.path, FormulaIngestionReceiptV2)
    iteration = _load(outcome.iteration.path, FormulaIterationRecordV2)
    assert outcome.response.path.read_bytes() == _response(record)
    assert replay_formula_response_bytes(outcome.response, receipt) == _response(record)
    assert receipt.retained_response.artifact_id == outcome.response.path.name
    assert receipt.retained_response.sha256 == hashlib.sha256(_response(record)).hexdigest()
    assert receipt.retained_response.byte_count == len(_response(record))
    assert receipt.response_origin == RESPONSE_ORIGIN
    assert receipt.model_call_receipt == "not_created"
    assert receipt.formula_validation == "data_contract_only"
    assert receipt.dynamic_calibration == "not_performed"
    assert receipt.training_authorization == "not_authorized"
    assert receipt.measured_improvement == "not_measured"

    feedback_parent = proposal.recipe.to_trusted_recipe()
    revision = prepare_formula_revision(
        record,
        outcome.response,
        proposal,
        receipt,
        iteration,
        _dossier(feedback_parent, measured=True),
        _sources(),
        _bindings(),
    )
    assert revision.manifest.stage == "revision"
    assert revision.manifest.prior_iteration_sha256 == formula_model_sha256(iteration)
    assert revision.manifest.prior_receipt_sha256 == formula_model_sha256(receipt)
    assert revision.manifest.parent_recipe_sha256 == receipt.candidate_recipe_sha256


def test_responder_can_copy_every_required_identity_from_prompt_bytes(tmp_path: Path) -> None:
    record = _packet()
    prompt = render_formula_packet(record)
    request_payload = _prompt_section_bytes(prompt, "Canonical request payload")
    identity = _prompt_json_section(prompt, "Required response identity")
    assert hashlib.sha256(request_payload).hexdigest() == identity["request_payload_sha256"]
    assert identity["request_payload_sha256"] == record.request_payload_sha256

    outcome = ingest_formula_response(
        record,
        _response_from_prompt(prompt),
        tmp_path / "records",
        response_id="prompt-only-responder",
    )
    assert outcome.accepted is True and outcome.proposal is not None
    proposal = _load(outcome.proposal.path, FormulaProposalV2)
    assert proposal.request_payload_sha256 == record.request_payload_sha256


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("request_payload_sha256", "0" * 64),
        ("rendered_prompt_sha256", "0" * 64),
        ("rendered_prompt_byte_count", 1),
    ],
)
def test_packet_record_recomputes_semantic_and_prompt_identities(
    field: str,
    value: object,
) -> None:
    record = _packet()
    with pytest.raises(FormulaSearchError, match="packet identity differs"):
        render_formula_packet(record.model_copy(update={field: value}))


@pytest.mark.parametrize(
    ("binding_name", "expected"),
    [
        ("config", "invalid retained FormulaPacketRecordV2"),
        ("trusted_formula_implementation", "invalid retained FormulaPacketRecordV2"),
        ("read_contract", "invalid retained FormulaPacketRecordV2"),
        ("independent_evaluator", "invalid retained FormulaPacketRecordV2"),
        ("frozen_configuration", "invalid retained FormulaPacketRecordV2"),
        ("compositor_state", "invalid retained FormulaPacketRecordV2"),
    ],
)
def test_mutating_each_retained_binding_fails_closed(binding_name: str, expected: str) -> None:
    record = _packet()
    bindings = record.bindings
    if binding_name == "frozen_configuration":
        first = bindings.frozen_configuration[0].model_copy(update={"sha256": "0" * 64})
        mutated = bindings.model_copy(
            update={"frozen_configuration": [first, *bindings.frozen_configuration[1:]]}
        )
    elif binding_name == "compositor_state":
        mutated = bindings.model_copy(update={"compositor_state": "bound"})
    else:
        snapshot = getattr(bindings, binding_name).model_copy(update={"sha256": "0" * 64})
        mutated = bindings.model_copy(update={binding_name: snapshot})
    changed = record.model_copy(update={"bindings": mutated})
    with pytest.raises(FormulaSearchError, match=expected):
        render_formula_packet(changed)


@pytest.mark.parametrize(
    "field",
    [
        "parent_recipe_sha256",
        "config_sha256",
        "trusted_formula_implementation_sha256",
        "read_contract_sha256",
        "independent_evaluator_sha256",
        "frozen_configuration_sha256",
        "bindings_sha256",
        "dossier_sha256",
        "source_bundle_sha256",
        "response_schema_sha256",
    ],
)
def test_mutating_each_manifest_identity_fails_closed(field: str) -> None:
    record = _packet()
    manifest = record.manifest.model_copy(update={field: "0" * 64})
    with pytest.raises(FormulaSearchError, match="manifest differs"):
        render_formula_packet(record.model_copy(update={"manifest": manifest}))


def test_mutating_parent_dossier_or_source_bytes_fails_closed() -> None:
    record = _packet()
    changed_parent = record.parent_recipe.model_copy(update={"beta": 1.0})
    changed_dossier = record.dossier.model_copy(update={"task_statement": "changed task"})
    changed_source = record.source_bundle.model_copy(update={"normalized_query": "other query"})
    for changed in (
        record.model_copy(update={"parent_recipe": changed_parent}),
        record.model_copy(update={"dossier": changed_dossier}),
        record.model_copy(update={"source_bundle": changed_source}),
    ):
        with pytest.raises(FormulaSearchError):
            render_formula_packet(changed)


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("request_payload_sha256", "request_payload_sha256 differs"),
        ("parent_recipe_sha256", "parent_recipe_sha256 differs"),
        ("dossier_sha256", "dossier_sha256 differs"),
        ("source_bundle_sha256", "source_bundle_sha256 differs"),
        ("bindings_sha256", "bindings_sha256 differs"),
    ],
)
def test_response_identity_mutations_are_rejected_and_retained(
    tmp_path: Path, field: str, expected: str
) -> None:
    record = _packet()
    outcome = ingest_formula_response(
        record,
        _response(record, **{field: "0" * 64}),
        tmp_path / field,
        response_id=f"synthetic-{field}",
    )
    assert outcome.accepted is False and outcome.proposal is None
    assert expected in (outcome.rejection_reason or "")
    receipt = _load(outcome.receipt.path, FormulaIngestionReceiptV2)
    iteration = _load(outcome.iteration.path, FormulaIterationRecordV2)
    assert receipt.disposition == "formula_proposal_rejected"
    assert receipt.formula_validation == "not_admitted"
    assert iteration.disposition == "formula_proposal_rejected"


@pytest.mark.parametrize(
    ("updates", "expected"),
    [
        ({"surprise": True}, "invalid FormulaProposalV2"),
        ({"cited_source_ids": ["unknown"]}, "unknown source ID"),
        ({"recipe": {"formula_id": "unknown/v1", "alpha": 1.0, "beta": 0.0}}, "invalid"),
        ({"recipe": {"formula_id": FORMULA_ID, "alpha": True, "beta": 0.0}}, "invalid"),
    ],
)
def test_invalid_formula_responses_retain_rejection_receipts(
    tmp_path: Path, updates: dict[str, object], expected: str
) -> None:
    record = _packet()
    outcome = ingest_formula_response(
        record,
        _response(record, **updates),
        tmp_path / "records",
        response_id="synthetic-rejected",
    )
    assert outcome.accepted is False and outcome.proposal is None
    assert expected in (outcome.rejection_reason or "")
    assert outcome.receipt.path.exists() and outcome.iteration.path.exists()


def test_malformed_response_is_retained_byte_for_byte_and_replay_verified(
    tmp_path: Path,
) -> None:
    record = _packet()
    malformed = b'{"not":"valid"\xff'
    outcome = ingest_formula_response(
        record,
        malformed,
        tmp_path / "records",
        response_id="synthetic-malformed",
    )
    receipt = _load(outcome.receipt.path, FormulaIngestionReceiptV2)
    assert outcome.accepted is False and outcome.proposal is None
    assert outcome.response.path.read_bytes() == malformed
    assert replay_formula_response_bytes(outcome.response, receipt) == malformed
    assert receipt.retained_response.sha256 == hashlib.sha256(malformed).hexdigest()
    assert receipt.retained_response.byte_count == len(malformed)


def test_response_replay_recomputes_mutated_bytes_and_revision_refuses(
    tmp_path: Path,
) -> None:
    record, response, proposal, receipt, iteration = _accepted(tmp_path)
    original = response.path.read_bytes()
    response.path.write_bytes(b"x" * len(original))

    with pytest.raises(FormulaSearchError, match="exact bytes"):
        replay_formula_response_bytes(response, receipt)
    with pytest.raises(FormulaSearchError, match="exact bytes"):
        prepare_formula_revision(
            record,
            response,
            proposal,
            receipt,
            iteration,
            _dossier(proposal.recipe.to_trusted_recipe(), measured=True),
            _sources(),
            _bindings(),
        )


def test_raw_response_publication_collision_preserves_existing_bytes(tmp_path: Path) -> None:
    record = _packet()
    response = _response(record)
    response_sha256 = hashlib.sha256(response).hexdigest()
    output = tmp_path / "records"
    output.mkdir()
    collision = output / f"formula-response-collision-{response_sha256}.bin"
    collision.write_bytes(b"preexisting")

    with pytest.raises(ValueError, match="overwrite"):
        ingest_formula_response(record, response, output, response_id="collision")
    assert collision.read_bytes() == b"preexisting"
    assert list(output.iterdir()) == [collision]


def test_response_replay_refuses_substituted_fifo_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, response, _, receipt, _ = _accepted(tmp_path)
    response.path.unlink()
    os.mkfifo(response.path, 0o600)
    original_open = os.open

    def checked_open(path, flags, *args, **kwargs):
        # Fail before entering the OS if a regression could block on the fixture.
        assert flags & os.O_NONBLOCK, "replay must use nonblocking open"
        return original_open(path, flags, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr("oracle_composition.reward_search.formula_loop.os.open", checked_open)
        with pytest.raises(FormulaSearchError, match="metadata"):
            replay_formula_response_bytes(response, receipt)


def test_revision_rejects_wrong_prior_receipt_candidate_and_feedback(
    tmp_path: Path,
) -> None:
    record, response, proposal, receipt, iteration = _accepted(tmp_path)
    feedback_parent = proposal.recipe.to_trusted_recipe()
    feedback = _dossier(feedback_parent, measured=True)

    wrong_receipt = receipt.model_copy(update={"request_payload_sha256": "0" * 64})
    with pytest.raises(FormulaSearchError, match="request identity"):
        prepare_formula_revision(
            record,
            response,
            proposal,
            wrong_receipt,
            iteration,
            feedback,
            _sources(),
            _bindings(),
        )

    wrong_prompt = receipt.model_copy(update={"rendered_prompt_sha256": "0" * 64})
    with pytest.raises(FormulaSearchError, match="prompt identity"):
        prepare_formula_revision(
            record,
            response,
            proposal,
            wrong_prompt,
            iteration,
            feedback,
            _sources(),
            _bindings(),
        )

    wrong_candidate = proposal.model_copy(
        update={"recipe": proposal.recipe.model_copy(update={"beta": 1.0})}
    )
    with pytest.raises(FormulaSearchError, match="retained response differs"):
        prepare_formula_revision(
            record,
            response,
            wrong_candidate,
            receipt,
            iteration,
            feedback,
            _sources(),
            _bindings(),
        )

    wrong_parent = feedback.model_copy(update={"parent_recipe_sha256": "0" * 64})
    with pytest.raises(FormulaSearchError, match="feedback parent"):
        prepare_formula_revision(
            record,
            response,
            proposal,
            receipt,
            iteration,
            wrong_parent,
            _sources(),
            _bindings(),
        )

    missing_only = _dossier(feedback_parent)
    with pytest.raises(FormulaSearchError, match="protected aggregate feedback"):
        prepare_formula_revision(
            record,
            response,
            proposal,
            receipt,
            iteration,
            missing_only,
            _sources(),
            _bindings(),
        )

    wrong_evaluator = feedback.model_copy(update={"independent_evaluator_sha256": "0" * 64})
    with pytest.raises(FormulaSearchError, match="independent evaluator"):
        prepare_formula_revision(
            record,
            response,
            proposal,
            receipt,
            iteration,
            wrong_evaluator,
            _sources(),
            _bindings(),
        )

    wrong_frozen = feedback.model_copy(update={"frozen_configuration_sha256": "0" * 64})
    with pytest.raises(FormulaSearchError, match="frozen configuration"):
        prepare_formula_revision(
            record,
            response,
            proposal,
            receipt,
            iteration,
            wrong_frozen,
            _sources(),
            _bindings(),
        )


def test_public_boundaries_revalidate_top_level_and_nested_schema_labels() -> None:
    record = _packet()
    for forged in (
        record.model_copy(update={"kind": "forged_packet_kind"}),
        record.model_copy(update={"schema_version": 1}),
        record.model_copy(
            update={"bindings": record.bindings.model_copy(update={"kind": "forged_bindings"})}
        ),
        record.model_copy(
            update={"dossier": record.dossier.model_copy(update={"kind": "forged_dossier"})}
        ),
    ):
        with pytest.raises(FormulaSearchError, match="invalid retained FormulaPacketRecordV2"):
            render_formula_packet(forged)

    with pytest.raises(FormulaSearchError, match="invalid retained FormulaArtifactBindingsV2"):
        formula_bindings_sha256(record.bindings.model_copy(update={"kind": "forged_bindings"}))


def test_public_boundary_rejects_partial_lineage_before_revision(tmp_path: Path) -> None:
    record, response, proposal, receipt, iteration = _accepted(tmp_path)
    payload = receipt.model_dump(mode="python")
    del payload["bindings_sha256"]
    partial_receipt = FormulaIngestionReceiptV2.model_construct(**payload)

    with pytest.raises(FormulaSearchError, match="partial or forged field storage"):
        prepare_formula_revision(
            record,
            response,
            proposal,
            partial_receipt,
            iteration,
            _dossier(proposal.recipe.to_trusted_recipe(), measured=True),
            _sources(),
            _bindings(),
        )


def test_mutated_nested_lists_and_forged_entries_fail_full_schema_validation() -> None:
    duplicated_source = _packet()
    duplicated_source.source_bundle.source_cards.append(
        duplicated_source.source_bundle.source_cards[0]
    )
    with pytest.raises(FormulaSearchError, match="invalid retained FormulaPacketRecordV2"):
        render_formula_packet(duplicated_source)

    forged_binding = _packet()
    first = forged_binding.bindings.frozen_configuration[0].model_copy(update={"byte_count": 1})
    forged_binding.bindings.frozen_configuration[0] = first  # type: ignore[list-item]
    with pytest.raises(FormulaSearchError, match="invalid retained FormulaPacketRecordV2"):
        render_formula_packet(forged_binding)

    cyclic_source = _packet()
    cyclic_source.source_bundle.source_cards.append(  # type: ignore[arg-type]
        cyclic_source.source_bundle.source_cards
    )
    with pytest.raises(FormulaSearchError, match="cyclic nested value"):
        render_formula_packet(cyclic_source)


def test_nested_callback_object_is_rejected_before_callback_access() -> None:
    accessed: list[str] = []

    class CallbackTrap:
        def __getattribute__(self, name: str) -> object:
            accessed.append(name)
            raise AssertionError("callback-bearing value was accessed")

    record = _packet()
    record.source_bundle.source_cards.append(CallbackTrap())  # type: ignore[arg-type]
    with pytest.raises(FormulaSearchError, match="unsafe or forged nested value"):
        render_formula_packet(record)
    assert accessed == []


def test_dossier_requires_measured_or_explicitly_missing_evidence() -> None:
    parent = _recipe()
    payload = _dossier(parent).model_dump(mode="python")
    payload["missing_evidence"] = []
    with pytest.raises(ValidationError, match="measured or missing"):
        FormulaEvidenceDossierV2.model_validate(payload, strict=True)


def test_a1_code_and_formula_v2_artifacts_remain_schema_separated() -> None:
    record = _packet()
    formula_bytes = _response(record)
    with pytest.raises(RewardSearchError):
        parse_model_bytes(formula_bytes, RewardProposal, max_bytes=65_536)

    a1 = {
        "schema_version": 1,
        "kind": "reward_proposal",
        "parent_candidate_sha256": HASH_A,
        "dossier_sha256": HASH_A,
        "proposed_source": "def task_term(x):\n    return 0.0\n",
        "rationale": "A1 fixture",
        "predicted_effect": "Unknown",
        "falsifier": "No change",
        "cited_source_ids": [],
        "declared_read_surface": ["com_x_velocity_m_s", "target_speed_m_s"],
    }
    parsed_a1 = parse_model_bytes(finite_pretty_json(a1), RewardProposal, max_bytes=65_536)
    assert parsed_a1.kind == "reward_proposal"
    with pytest.raises(RewardSearchError):
        parse_model_bytes(finite_pretty_json(a1), FormulaProposalV2, max_bytes=65_536)


def test_receipts_are_frozen_and_publication_refuses_overwrite(tmp_path: Path) -> None:
    record = _packet()
    response = _response(record)
    outcome = ingest_formula_response(
        record, response, tmp_path / "records", response_id="synthetic-once"
    )
    receipt = _load(outcome.receipt.path, FormulaIngestionReceiptV2)
    with pytest.raises(ValidationError, match="frozen"):
        receipt.response_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="overwrite"):
        ingest_formula_response(
            record, response, tmp_path / "records", response_id="synthetic-once"
        )


def test_formula_loop_has_no_launch_dynamic_code_or_heavy_runtime_imports() -> None:
    paths = [
        ROOT / "src/oracle_composition/reward_search/formula_contracts.py",
        ROOT / "src/oracle_composition/reward_search/formula_loop.py",
        Path(__file__),
    ]
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_roots = {
            alias.name.split(".", 1)[0]
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        called_names = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert not imported_roots & {"gymnasium", "mujoco", "torch"}
        assert not imported_roots & {"argparse", "subprocess"}
        assert not called_names & {"compile", "exec", "eval", "__import__"}
