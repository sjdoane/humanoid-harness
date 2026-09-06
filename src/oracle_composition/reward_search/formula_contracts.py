"""Schema-v2 contracts for data-only formula proposals."""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from oracle_composition.rewards.target_speed_formula import (
    FORMULA_ID,
    TargetSpeedFormulaRecipeV1,
)

from .contracts import AggregateMeasurement, EvidenceProvenance, MissingEvidence, SourceBundle

FORMULA_SCHEMA_VERSION = 2
SHA256_PATTERN = r"^[0-9a-f]{64}$"
RESPONSE_ORIGIN = "unverified_synthetic_supplied_bytes"


class FormulaStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class FormulaRecipePayloadV1(FormulaStrictModel):
    formula_id: Literal["target_speed_triangular_affine/v1"]
    alpha: float
    beta: float

    @field_validator("alpha", "beta", mode="before")
    @classmethod
    def validate_exact_number(cls, value: object) -> float:
        if type(value) not in (int, float):
            raise ValueError("formula parameters must be exact JSON numbers")
        try:
            result = float(value)
        except OverflowError as exc:
            raise ValueError("formula parameters must be finite") from exc
        if not math.isfinite(result):
            raise ValueError("formula parameters must be finite")
        return result

    @model_validator(mode="after")
    def validate_bounds(self) -> FormulaRecipePayloadV1:
        TargetSpeedFormulaRecipeV1(
            formula_id=self.formula_id,
            alpha=self.alpha,
            beta=self.beta,
        )
        return self

    def to_trusted_recipe(self) -> TargetSpeedFormulaRecipeV1:
        return TargetSpeedFormulaRecipeV1(
            formula_id=self.formula_id,
            alpha=self.alpha,
            beta=self.beta,
        )

    @classmethod
    def from_trusted_recipe(cls, recipe: TargetSpeedFormulaRecipeV1) -> FormulaRecipePayloadV1:
        if type(recipe) is not TargetSpeedFormulaRecipeV1:
            raise TypeError("recipe must be an exact TargetSpeedFormulaRecipeV1")
        return cls(formula_id=recipe.formula_id, alpha=recipe.alpha, beta=recipe.beta)


class RetainedArtifactV2(FormulaStrictModel):
    artifact_id: str = Field(min_length=1, max_length=512)
    sha256: str = Field(pattern=SHA256_PATTERN)
    byte_count: int = Field(ge=1, le=262_144)
    content_base64: str = Field(min_length=4, max_length=350_000)

    @model_validator(mode="after")
    def validate_retained_bytes(self) -> RetainedArtifactV2:
        if self.artifact_id != self.artifact_id.strip() or "\x00" in self.artifact_id:
            raise ValueError("artifact_id must be trimmed text without NUL")
        try:
            encoded = base64.b64decode(self.content_base64.encode("ascii"), validate=True)
        except (UnicodeError, binascii.Error) as exc:
            raise ValueError("artifact content is not strict base64") from exc
        if len(encoded) != self.byte_count:
            raise ValueError("artifact byte count differs from retained bytes")
        if hashlib.sha256(encoded).hexdigest() != self.sha256:
            raise ValueError("artifact SHA-256 differs from retained bytes")
        return self


class FormulaArtifactBindingsV2(FormulaStrictModel):
    schema_version: Literal[2]
    kind: Literal["target_speed_formula_artifact_bindings"]
    config: RetainedArtifactV2
    trusted_formula_implementation: RetainedArtifactV2
    read_contract: RetainedArtifactV2
    compositor_state: Literal["bound", "explicitly_absent_pending_peer_resolution"]
    compositor: RetainedArtifactV2 | None
    independent_evaluator: RetainedArtifactV2
    frozen_configuration: list[RetainedArtifactV2] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_artifact_set(self) -> FormulaArtifactBindingsV2:
        if (self.compositor_state == "bound") != (self.compositor is not None):
            raise ValueError("compositor state and retained artifact differ")
        artifacts = [
            self.config,
            self.trusted_formula_implementation,
            self.read_contract,
            self.independent_evaluator,
            *self.frozen_configuration,
        ]
        if self.compositor is not None:
            artifacts.append(self.compositor)
        identifiers = [artifact.artifact_id for artifact in artifacts]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("retained artifact IDs must be unique")
        return self


class FormulaEvidenceDossierV2(FormulaStrictModel):
    schema_version: Literal[2]
    kind: Literal["target_speed_formula_evidence_dossier"]
    task_id: str = Field(min_length=1, max_length=256)
    task_statement: str = Field(min_length=1, max_length=8_192)
    parent_recipe_sha256: str = Field(pattern=SHA256_PATTERN)
    independent_evaluator_sha256: str = Field(pattern=SHA256_PATTERN)
    frozen_configuration_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_provenance: list[EvidenceProvenance] = Field(max_length=256)
    aggregate_feedback: list[AggregateMeasurement] = Field(max_length=256)
    missing_evidence: list[MissingEvidence] = Field(max_length=256)
    synthetic: bool

    @model_validator(mode="after")
    def validate_evidence(self) -> FormulaEvidenceDossierV2:
        provenance = [item.provenance_id for item in self.evidence_provenance]
        measured = [item.evidence_id for item in self.aggregate_feedback]
        missing = [item.evidence_id for item in self.missing_evidence]
        for label, values in {
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
        return self


class FormulaProposalV2(FormulaStrictModel):
    schema_version: Literal[2]
    kind: Literal["target_speed_formula_proposal"]
    request_payload_sha256: str = Field(
        pattern=SHA256_PATTERN,
        description="SHA-256 of the canonical semantic request payload shown in the prompt.",
    )
    parent_recipe_sha256: str = Field(pattern=SHA256_PATTERN)
    dossier_sha256: str = Field(pattern=SHA256_PATTERN)
    source_bundle_sha256: str = Field(pattern=SHA256_PATTERN)
    bindings_sha256: str = Field(pattern=SHA256_PATTERN)
    recipe: FormulaRecipePayloadV1
    rationale: str = Field(min_length=1, max_length=8_192)
    predicted_effect: str = Field(min_length=1, max_length=4_096)
    falsifier: str = Field(min_length=1, max_length=4_096)
    cited_source_ids: list[str] = Field(max_length=20)

    @model_validator(mode="after")
    def validate_citations(self) -> FormulaProposalV2:
        if len(set(self.cited_source_ids)) != len(self.cited_source_ids):
            raise ValueError("cited source IDs must be unique")
        return self


class FormulaPacketManifestV2(FormulaStrictModel):
    schema_version: Literal[2]
    kind: Literal["target_speed_formula_packet_manifest"]
    stage: Literal["initial", "revision"]
    parent_recipe_sha256: str = Field(pattern=SHA256_PATTERN)
    config_sha256: str = Field(pattern=SHA256_PATTERN)
    trusted_formula_implementation_sha256: str = Field(pattern=SHA256_PATTERN)
    read_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    compositor_state: Literal["bound", "explicitly_absent_pending_peer_resolution"]
    compositor_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    independent_evaluator_sha256: str = Field(pattern=SHA256_PATTERN)
    frozen_configuration_sha256: str = Field(pattern=SHA256_PATTERN)
    bindings_sha256: str = Field(pattern=SHA256_PATTERN)
    dossier_sha256: str = Field(pattern=SHA256_PATTERN)
    source_bundle_sha256: str = Field(pattern=SHA256_PATTERN)
    response_schema_sha256: str = Field(pattern=SHA256_PATTERN)
    prior_iteration_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    prior_receipt_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_stage(self) -> FormulaPacketManifestV2:
        revision = self.stage == "revision"
        if revision != (
            self.prior_iteration_sha256 is not None and self.prior_receipt_sha256 is not None
        ):
            raise ValueError("only revision packets bind a prior iteration and receipt")
        if (self.compositor_state == "bound") != (self.compositor_sha256 is not None):
            raise ValueError("compositor state and identity differ")
        return self


class FormulaPacketRecordV2(FormulaStrictModel):
    schema_version: Literal[2]
    kind: Literal["target_speed_formula_packet_record"]
    manifest: FormulaPacketManifestV2
    parent_recipe: FormulaRecipePayloadV1
    bindings: FormulaArtifactBindingsV2
    dossier: FormulaEvidenceDossierV2
    source_bundle: SourceBundle
    request_payload_sha256: str = Field(
        pattern=SHA256_PATTERN,
        description="SHA-256 of the canonical semantic request payload.",
    )
    rendered_prompt_sha256: str = Field(
        pattern=SHA256_PATTERN,
        description="SHA-256 of the final UTF-8 prompt bytes.",
    )
    rendered_prompt_byte_count: int = Field(ge=1, le=131_072)


class FormulaRetainedResponseV2(FormulaStrictModel):
    artifact_id: str = Field(min_length=1, max_length=256)
    sha256: str = Field(pattern=SHA256_PATTERN)
    byte_count: int = Field(ge=1, le=65_536)

    @model_validator(mode="after")
    def validate_artifact_id(self) -> FormulaRetainedResponseV2:
        if (
            self.artifact_id != self.artifact_id.strip()
            or "\x00" in self.artifact_id
            or "/" in self.artifact_id
            or "\\" in self.artifact_id
        ):
            raise ValueError("response artifact_id must be one safe filename")
        return self


class FormulaIngestionReceiptV2(FormulaStrictModel):
    schema_version: Literal[2]
    kind: Literal["target_speed_formula_ingestion_receipt"]
    disposition: Literal["formula_proposal_accepted", "formula_proposal_rejected"]
    response_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    response_origin: Literal["unverified_synthetic_supplied_bytes"]
    retained_response: FormulaRetainedResponseV2
    request_payload_sha256: str = Field(pattern=SHA256_PATTERN)
    rendered_prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    parent_recipe_sha256: str = Field(pattern=SHA256_PATTERN)
    dossier_sha256: str = Field(pattern=SHA256_PATTERN)
    source_bundle_sha256: str = Field(pattern=SHA256_PATTERN)
    bindings_sha256: str = Field(pattern=SHA256_PATTERN)
    proposal_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    candidate_recipe_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    rejection_reason: str | None = Field(default=None, max_length=4_096)
    model_call_receipt: Literal["not_created"]
    formula_validation: Literal["data_contract_only", "not_admitted"]
    dynamic_calibration: Literal["not_performed"]
    training_authorization: Literal["not_authorized"]
    measured_improvement: Literal["not_measured"]

    @model_validator(mode="after")
    def validate_disposition(self) -> FormulaIngestionReceiptV2:
        expected_artifact_id = (
            f"formula-response-{self.response_id}-{self.retained_response.sha256}.bin"
        )
        if self.retained_response.artifact_id != expected_artifact_id:
            raise ValueError("retained response artifact ID differs from its receipt identity")
        accepted = self.disposition == "formula_proposal_accepted"
        identified = self.proposal_sha256 is not None and self.candidate_recipe_sha256 is not None
        if accepted != identified:
            raise ValueError("only accepted receipts identify a proposal and candidate recipe")
        if accepted == (self.rejection_reason is not None):
            raise ValueError("only rejected receipts require a rejection reason")
        expected = "data_contract_only" if accepted else "not_admitted"
        if self.formula_validation != expected:
            raise ValueError("receipt validation state differs from its disposition")
        return self


class FormulaIterationRecordV2(FormulaStrictModel):
    schema_version: Literal[2]
    kind: Literal["target_speed_formula_iteration_record"]
    stage: Literal["initial", "revision"]
    disposition: Literal["formula_proposal_accepted", "formula_proposal_rejected"]
    request_payload_sha256: str = Field(pattern=SHA256_PATTERN)
    rendered_prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    parent_recipe_sha256: str = Field(pattern=SHA256_PATTERN)
    dossier_sha256: str = Field(pattern=SHA256_PATTERN)
    source_bundle_sha256: str = Field(pattern=SHA256_PATTERN)
    bindings_sha256: str = Field(pattern=SHA256_PATTERN)
    receipt_sha256: str = Field(pattern=SHA256_PATTERN)
    proposal_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    candidate_recipe_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    prior_iteration_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    prior_receipt_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_links(self) -> FormulaIterationRecordV2:
        accepted = self.disposition == "formula_proposal_accepted"
        identified = self.proposal_sha256 is not None and self.candidate_recipe_sha256 is not None
        if accepted != identified:
            raise ValueError("only accepted iterations identify a proposal and candidate recipe")
        revision = self.stage == "revision"
        if revision != (
            self.prior_iteration_sha256 is not None and self.prior_receipt_sha256 is not None
        ):
            raise ValueError("only revision iterations identify prior lineage")
        return self


__all__ = [
    "FORMULA_ID",
    "FORMULA_SCHEMA_VERSION",
    "RESPONSE_ORIGIN",
    "FormulaArtifactBindingsV2",
    "FormulaEvidenceDossierV2",
    "FormulaIngestionReceiptV2",
    "FormulaIterationRecordV2",
    "FormulaPacketManifestV2",
    "FormulaPacketRecordV2",
    "FormulaProposalV2",
    "FormulaRecipePayloadV1",
    "FormulaRetainedResponseV2",
    "RetainedArtifactV2",
]
