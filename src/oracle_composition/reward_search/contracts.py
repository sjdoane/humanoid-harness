"""Strict data contracts for the reward proposal loop."""

from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = 1
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class CandidateSnapshot(StrictModel):
    sha256: str = Field(pattern=SHA256_PATTERN)
    source_text: str = Field(min_length=1, max_length=16_384)

    @model_validator(mode="after")
    def verify_identity(self) -> CandidateSnapshot:
        observed = hashlib.sha256(self.source_text.encode("utf-8")).hexdigest()
        if observed != self.sha256:
            raise ValueError("candidate source SHA-256 differs")
        return self


class AuthorSurface(StrictModel):
    descriptor_id: str = Field(min_length=1, max_length=256)
    entrypoint: str = Field(min_length=1, max_length=256)
    readable_fields: list[str] = Field(min_length=1, max_length=32)
    output_unit: str = Field(min_length=1, max_length=64)
    max_source_bytes: int = Field(ge=1, le=16_384)

    @model_validator(mode="after")
    def validate_fields(self) -> AuthorSurface:
        if len(set(self.readable_fields)) != len(self.readable_fields):
            raise ValueError("author surface readable fields must be unique")
        if any(not field.strip() or field != field.strip() for field in self.readable_fields):
            raise ValueError("author surface readable fields must be trimmed text")
        return self


class FrozenConfigurationHash(StrictModel):
    component: str = Field(min_length=1, max_length=256)
    sha256: str = Field(pattern=SHA256_PATTERN)


class EvidenceProvenance(StrictModel):
    provenance_id: str = Field(min_length=1, max_length=256)
    provenance_class: str = Field(min_length=1, max_length=128)
    locator: str = Field(min_length=1, max_length=2_048)
    artifact_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    synthetic: bool


class AggregateMeasurement(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=256)
    value: float
    unit: str = Field(min_length=1, max_length=64)
    provenance_id: str = Field(min_length=1, max_length=256)


class MissingEvidence(StrictModel):
    evidence_id: str = Field(min_length=1, max_length=256)
    reason: str = Field(min_length=1, max_length=2_048)


class ProtectedEvidenceDossier(StrictModel):
    schema_version: Literal[1]
    kind: Literal["protected_evidence_dossier"]
    task_id: str = Field(min_length=1, max_length=256)
    task_statement: str = Field(min_length=1, max_length=8_192)
    adapter_id: str = Field(min_length=1, max_length=256)
    adapter_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    reward_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    author_surface: AuthorSurface
    frozen_configuration: list[FrozenConfigurationHash] = Field(min_length=1, max_length=64)
    parent_candidate: CandidateSnapshot
    evidence_provenance: list[EvidenceProvenance] = Field(max_length=256)
    aggregate_feedback: list[AggregateMeasurement] = Field(max_length=256)
    missing_evidence: list[MissingEvidence] = Field(max_length=256)
    synthetic: bool

    @model_validator(mode="after")
    def validate_evidence(self) -> ProtectedEvidenceDossier:
        components = [item.component for item in self.frozen_configuration]
        provenance = [item.provenance_id for item in self.evidence_provenance]
        measured = [item.evidence_id for item in self.aggregate_feedback]
        missing = [item.evidence_id for item in self.missing_evidence]
        for label, values in {
            "frozen configuration components": components,
            "evidence provenance IDs": provenance,
            "measurement IDs": measured,
            "missing-evidence IDs": missing,
        }.items():
            if len(set(values)) != len(values):
                raise ValueError(f"{label} must be unique")
        if set(measured) & set(missing):
            raise ValueError("evidence cannot be both measured and missing")
        unknown = {item.provenance_id for item in self.aggregate_feedback} - set(provenance)
        if unknown:
            raise ValueError(f"measurements reference unknown provenance: {sorted(unknown)!r}")
        if any(item.synthetic != self.synthetic for item in self.evidence_provenance):
            raise ValueError("dossier and evidence provenance synthetic labels differ")
        if not self.aggregate_feedback and not self.missing_evidence:
            raise ValueError("dossier must explicitly contain measured or missing evidence")
        if (
            len(self.parent_candidate.source_text.encode("utf-8"))
            > self.author_surface.max_source_bytes
        ):
            raise ValueError("parent candidate exceeds the author-surface byte limit")
        return self


class SourceCard(StrictModel):
    source_id: str = Field(min_length=1, max_length=512)
    paper_id: str | None = Field(default=None, max_length=512)
    kind: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=2_048)
    excerpt: str = Field(min_length=1, max_length=4_096)
    locator: str = Field(min_length=1, max_length=2_048)
    source_url: str | None = Field(default=None, max_length=4_096)


class SourceBundle(StrictModel):
    schema_version: Literal[1]
    kind: Literal["reward_source_bundle"]
    graph_schema_version: int = Field(ge=1)
    graph_corpus_sha256: str = Field(pattern=SHA256_PATTERN)
    normalized_query: str = Field(min_length=1, max_length=1_024)
    limit: int = Field(ge=1, le=20)
    source_cards: list[SourceCard] = Field(max_length=20)
    synthetic: bool

    @model_validator(mode="after")
    def validate_bundle(self) -> SourceBundle:
        if self.normalized_query != " ".join(self.normalized_query.lower().split()):
            raise ValueError("source query must be normalized lowercase text")
        if len(self.source_cards) > self.limit:
            raise ValueError("source card count exceeds the declared limit")
        ids = [card.source_id for card in self.source_cards]
        if len(set(ids)) != len(ids):
            raise ValueError("source card IDs must be unique")
        return self


class RewardProposal(StrictModel):
    schema_version: Literal[1]
    kind: Literal["reward_proposal"]
    parent_candidate_sha256: str = Field(pattern=SHA256_PATTERN)
    dossier_sha256: str = Field(pattern=SHA256_PATTERN)
    proposed_source: str = Field(min_length=1, max_length=16_384)
    rationale: str = Field(min_length=1, max_length=8_192)
    predicted_effect: str = Field(min_length=1, max_length=4_096)
    falsifier: str = Field(min_length=1, max_length=4_096)
    cited_source_ids: list[str] = Field(max_length=20)
    declared_read_surface: list[str] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_lists(self) -> RewardProposal:
        if len(set(self.cited_source_ids)) != len(self.cited_source_ids):
            raise ValueError("cited source IDs must be unique")
        if len(set(self.declared_read_surface)) != len(self.declared_read_surface):
            raise ValueError("declared read surface must be unique")
        return self


class PacketManifest(StrictModel):
    schema_version: Literal[1]
    kind: Literal["reward_task_packet_manifest"]
    stage: Literal["initial", "revision"]
    dossier_sha256: str = Field(pattern=SHA256_PATTERN)
    source_bundle_sha256: str = Field(pattern=SHA256_PATTERN)
    parent_candidate_sha256: str = Field(pattern=SHA256_PATTERN)
    reward_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    response_schema_sha256: str = Field(pattern=SHA256_PATTERN)
    prior_iteration_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_stage(self) -> PacketManifest:
        if (self.stage == "revision") != (self.prior_iteration_sha256 is not None):
            raise ValueError("only revision packets require a prior iteration SHA-256")
        return self


class PacketRecord(StrictModel):
    schema_version: Literal[1]
    kind: Literal["reward_task_packet_record"]
    manifest: PacketManifest
    dossier: ProtectedEvidenceDossier
    source_bundle: SourceBundle
    packet_sha256: str = Field(pattern=SHA256_PATTERN)
    packet_bytes: int = Field(ge=1, le=131_072)


class RunArtifactDigests(StrictModel):
    request_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    task_packet_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    result_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    final_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)


class ModelCallReceipt(StrictModel):
    schema_version: Literal[1]
    kind: Literal["reward_model_call_receipt"]
    disposition: Literal["proposal_format_accepted", "proposal_format_rejected"]
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    artifacts: RunArtifactDigests
    packet_sha256: str = Field(pattern=SHA256_PATTERN)
    packet_bytes: int = Field(ge=1, le=131_072)
    dossier_sha256: str = Field(pattern=SHA256_PATTERN)
    parent_candidate_sha256: str = Field(pattern=SHA256_PATTERN)
    requested_model: str | None = Field(default=None, max_length=256)
    requested_reasoning_effort: str | None = Field(default=None, max_length=64)
    runner_kind: str | None = Field(default=None, max_length=64)
    screen_name: str | None = Field(default=None, max_length=512)
    role: str | None = Field(default=None, max_length=256)
    scope: str | None = Field(default=None, max_length=2_048)
    result_thread_id: str | None = Field(default=None, max_length=512)
    proposal_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    candidate_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    rejection_reason: str | None = Field(default=None, max_length=4_096)
    metadata_semantics: Literal["requested_configuration_not_served_model_attestation"]
    candidate_state: Literal["awaiting_reward_validation_and_protected_evidence", "not_admitted"]
    generated_source_state: Literal["awaiting_B0_validation", "not_admitted"]
    reward_validation: Literal["not_performed"]
    training_authorization: Literal["not_authorized"]
    measured_improvement: Literal["not_measured"]

    @model_validator(mode="after")
    def validate_disposition(self) -> ModelCallReceipt:
        accepted = self.disposition == "proposal_format_accepted"
        if accepted != (self.proposal_sha256 is not None and self.candidate_sha256 is not None):
            raise ValueError("only accepted receipts identify a proposal and candidate")
        if accepted == (self.rejection_reason is not None):
            raise ValueError("only rejected receipts require a rejection reason")
        expected_candidate = (
            "awaiting_reward_validation_and_protected_evidence" if accepted else "not_admitted"
        )
        expected_source = "awaiting_B0_validation" if accepted else "not_admitted"
        if (
            self.candidate_state != expected_candidate
            or self.generated_source_state != expected_source
        ):
            raise ValueError("receipt semantic states do not match its disposition")
        if accepted and (
            None in self.artifacts.model_dump().values()
            or self.requested_model != "gpt-5.6-sol"
            or self.requested_reasoning_effort != "max"
            or self.runner_kind != "screen"
            or self.result_thread_id is None
        ):
            raise ValueError("accepted receipt lacks its complete Sol envelope identity")
        return self


class IterationRecord(StrictModel):
    schema_version: Literal[1]
    kind: Literal["reward_iteration_record"]
    stage: Literal["initial", "revision"]
    disposition: Literal["proposal_format_accepted", "proposal_format_rejected"]
    dossier_sha256: str = Field(pattern=SHA256_PATTERN)
    source_bundle_sha256: str = Field(pattern=SHA256_PATTERN)
    packet_sha256: str = Field(pattern=SHA256_PATTERN)
    parent_candidate_sha256: str = Field(pattern=SHA256_PATTERN)
    receipt_sha256: str = Field(pattern=SHA256_PATTERN)
    proposal_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    candidate_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    prior_iteration_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_links(self) -> IterationRecord:
        accepted = self.disposition == "proposal_format_accepted"
        if accepted != (self.proposal_sha256 is not None and self.candidate_sha256 is not None):
            raise ValueError("only accepted iterations identify a proposal and candidate")
        if (self.stage == "revision") != (self.prior_iteration_sha256 is not None):
            raise ValueError("only revision iterations identify a prior iteration")
        return self


class DetachedSolRequest(StrictModel):
    schema_version: Literal[1]
    mode: Literal["write", "read-only"]
    owner: str = Field(min_length=1, max_length=128)
    role: str = Field(min_length=1, max_length=128)
    scope: str = Field(min_length=1, max_length=2_048)
    requested_model: str = Field(min_length=1, max_length=128)
    requested_reasoning_effort: str = Field(min_length=1, max_length=64)
    runner_kind: str = Field(min_length=1, max_length=64)
    screen_name: str = Field(min_length=1, max_length=128)
    codex_bin: str = Field(min_length=1, max_length=4_096)
    codex_cli_version: str = Field(min_length=1, max_length=256)
    prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    prompt_bytes: int = Field(ge=1)
    created_at_utc: str = Field(min_length=1, max_length=64)


class DetachedSolResult(StrictModel):
    schema_version: Literal[1]
    status: str = Field(min_length=1, max_length=64)
    exit_code: int
    thread_id: str | None = Field(max_length=512)
    finished_at_utc: str = Field(min_length=1, max_length=64)
    lease_release: str = Field(min_length=1, max_length=64)
    termination_escalated: bool
    requested_model: str = Field(min_length=1, max_length=128)
    requested_reasoning_effort: str = Field(min_length=1, max_length=64)
    runner_kind: str = Field(min_length=1, max_length=64)
    screen_name: str = Field(min_length=1, max_length=128)
    codex_cli_version: str = Field(min_length=1, max_length=256)
    role: str = Field(min_length=1, max_length=128)
    scope: str = Field(min_length=1, max_length=2_048)
