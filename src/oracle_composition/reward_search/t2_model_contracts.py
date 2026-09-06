"""Strict records for the initial T2 parameter-proposal protocol."""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

T2_MODEL_SCHEMA_VERSION = 3
SHA256_PATTERN = r"^[0-9a-f]{64}$"

RUN_ARTIFACT_NAMES = ("request.json", "task-packet.md", "result.json", "final.txt")
MISSING_EVIDENCE = (
    "candidate_reward_admission",
    "verified_com_adapter_origin_cadence_gate",
    "execution_manifest",
    "protected_evaluator_results",
    "candidate_measurements",
)


class T2StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class T2RetainedArtifact(T2StrictModel):
    artifact_id: str = Field(min_length=1, max_length=512)
    sha256: str = Field(pattern=SHA256_PATTERN)
    byte_count: int = Field(ge=1, le=262_144)
    content_base64: str = Field(min_length=4, max_length=350_000)

    @model_validator(mode="after")
    def validate_exact_bytes(self) -> T2RetainedArtifact:
        if (
            type(self.artifact_id) is not str
            or self.artifact_id != self.artifact_id.strip()
            or "\x00" in self.artifact_id
        ):
            raise ValueError("artifact_id must be exact trimmed text without NUL")
        try:
            encoded = base64.b64decode(self.content_base64.encode("ascii"), validate=True)
        except (UnicodeError, binascii.Error) as exc:
            raise ValueError("artifact content is not strict base64") from exc
        if len(encoded) != self.byte_count:
            raise ValueError("artifact byte count differs from retained bytes")
        if hashlib.sha256(encoded).hexdigest() != self.sha256:
            raise ValueError("artifact SHA-256 differs from retained bytes")
        return self


class T2ContractBundle(T2StrictModel):
    schema_version: Literal[3]
    kind: Literal["t2_initial_contract_bundle"]
    recipe_parser_source: T2RetainedArtifact
    evaluator_source: T2RetainedArtifact
    task_inputs_source: T2RetainedArtifact
    canonical_task_inputs_schema: T2RetainedArtifact
    canonical_parameter_bounds: T2RetainedArtifact
    formula_id: Literal["target_speed_triangular_affine/v1"]
    parser_id: Literal["target_speed_triangular_affine_recipe/v1"]
    runtime_id: Literal["target_speed_triangular_affine_t2_adapter/v1"]
    response_schema_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_artifacts(self) -> T2ContractBundle:
        artifacts = (
            self.recipe_parser_source,
            self.evaluator_source,
            self.task_inputs_source,
            self.canonical_task_inputs_schema,
            self.canonical_parameter_bounds,
        )
        if any(type(item) is not T2RetainedArtifact for item in artifacts):
            raise ValueError("T2 contract artifacts must use the exact retained-artifact type")
        identifiers = [item.artifact_id for item in artifacts]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("T2 contract artifact IDs must be unique")
        return self


class T2InitialEvidenceDossier(T2StrictModel):
    schema_version: Literal[3]
    kind: Literal["t2_initial_missing_evidence_dossier"]
    evidence_stage: Literal["initial_only"]
    aggregate_feedback: list[object] = Field(max_length=0)
    missing_evidence: list[
        Literal[
            "candidate_reward_admission",
            "verified_com_adapter_origin_cadence_gate",
            "execution_manifest",
            "protected_evaluator_results",
            "candidate_measurements",
        ]
    ] = Field(min_length=5, max_length=5)
    baseline_evidence_class: Literal[
        "declared_tracking_only_configuration_not_observed_successful_load"
    ]
    candidate_reward_admission: Literal["missing"]
    verified_com_adapter_origin_cadence_gate: Literal["missing"]
    required_adapter_gate: Literal[
        "inclusive_-25.0_to_25.0_m_s_stock_com_origin_and_0.015_s_cadence_verification_missing"
    ]
    execution_manifest: Literal["missing"]
    protected_evaluator_results: Literal["missing"]
    candidate_measurements: Literal["missing"]
    current_physical_admission: Literal["not_claimed"]

    @model_validator(mode="after")
    def validate_initial_only_evidence(self) -> T2InitialEvidenceDossier:
        if type(self.aggregate_feedback) is not list or self.aggregate_feedback:
            raise ValueError("initial T2 dossier has no aggregate feedback")
        if (
            type(self.missing_evidence) is not list
            or tuple(self.missing_evidence) != MISSING_EVIDENCE
        ):
            raise ValueError("initial T2 dossier must preserve the fixed missing-evidence set")
        return self


class T2InitialRequest(T2StrictModel):
    schema_version: Literal[3]
    kind: Literal["t2_initial_parameter_request"]
    stage: Literal["initial_only"]
    baseline_sha256: str = Field(pattern=SHA256_PATTERN)
    baseline_byte_count: int = Field(ge=1, le=131_072)
    t2_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_dossier_sha256: str = Field(pattern=SHA256_PATTERN)
    target_speed_m_s: float
    control_period_seconds: float
    trusted_formula_id: Literal["target_speed_triangular_affine/v1"]
    trusted_parser_id: Literal["target_speed_triangular_affine_recipe/v1"]
    trusted_runtime_id: Literal["target_speed_triangular_affine_t2_adapter/v1"]
    formula_semantics: Literal["r_task=alpha*1.25*(1-min(1,abs(com_x_velocity_m_s-3.0)/3.0))+beta"]
    alpha_closed_interval: list[float] = Field(min_length=2, max_length=2)
    beta_closed_interval: list[float] = Field(min_length=2, max_length=2)
    authorable_parameters: list[Literal["alpha", "beta"]] = Field(min_length=2, max_length=2)
    baseline_role: Literal["tracking_only_configuration"]
    baseline_load_evidence: Literal["not_observed"]
    candidate_reward_admission: Literal["missing"]
    required_adapter_gate: Literal[
        "inclusive_-25.0_to_25.0_m_s_stock_com_origin_and_0.015_s_cadence_verification_missing"
    ]
    runtime_authorization: Literal["not_authorized"]
    training_authorization: Literal["not_authorized"]
    measured_improvement: Literal["not_measured"]

    @field_validator("target_speed_m_s", "control_period_seconds", mode="before")
    @classmethod
    def validate_exact_finite_float(cls, value: object) -> float:
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("T2 target and cadence must be exact finite floats")
        return value

    @model_validator(mode="after")
    def validate_fixed_request(self) -> T2InitialRequest:
        if self.target_speed_m_s != 3.0 or self.control_period_seconds != 0.015:
            raise ValueError("T2 target or cadence differs from the fixed initial request")
        if self.alpha_closed_interval != [0.25, 4.0]:
            raise ValueError("alpha bounds differ from the accepted T2 contract")
        if self.beta_closed_interval != [-10.0, 10.0]:
            raise ValueError("beta bounds differ from the accepted T2 contract")
        if type(self.authorable_parameters) is not list or self.authorable_parameters != [
            "alpha",
            "beta",
        ]:
            raise ValueError("only alpha and beta are authorable")
        return self


class T2InitialPacketRecord(T2StrictModel):
    schema_version: Literal[3]
    kind: Literal["t2_initial_parameter_packet_record"]
    baseline: T2RetainedArtifact
    contracts: T2ContractBundle
    evidence_dossier: T2InitialEvidenceDossier
    semantic_request: T2InitialRequest
    request_payload_sha256: str = Field(pattern=SHA256_PATTERN)
    baseline_sha256: str = Field(pattern=SHA256_PATTERN)
    t2_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_dossier_sha256: str = Field(pattern=SHA256_PATTERN)
    rendered_prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    rendered_prompt_byte_count: int = Field(ge=1, le=131_072)


class T2ParameterValues(T2StrictModel):
    alpha: float
    beta: float

    @field_validator("alpha", "beta", mode="before")
    @classmethod
    def validate_exact_number(cls, value: object) -> float:
        if type(value) not in (int, float):
            raise ValueError("T2 parameters must be exact JSON numbers")
        try:
            result = float(value)
        except OverflowError as exc:
            raise ValueError("T2 parameters must be finite") from exc
        if not math.isfinite(result):
            raise ValueError("T2 parameters must be finite")
        return result

    @model_validator(mode="after")
    def validate_bounds(self) -> T2ParameterValues:
        if not 0.25 <= self.alpha <= 4.0:
            raise ValueError("alpha is outside [0.25, 4.0]")
        if not -10.0 <= self.beta <= 10.0:
            raise ValueError("beta is outside [-10.0, 10.0]")
        return self


class T2ParameterProposal(T2StrictModel):
    schema_version: Literal[3]
    kind: Literal["t2_initial_parameter_proposal"]
    request_payload_sha256: str = Field(pattern=SHA256_PATTERN)
    baseline_sha256: str = Field(pattern=SHA256_PATTERN)
    t2_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_dossier_sha256: str = Field(pattern=SHA256_PATTERN)
    parameters: T2ParameterValues
    rationale: str = Field(min_length=1, max_length=8_192)
    predicted_effect: str = Field(min_length=1, max_length=4_096)
    falsifier: str = Field(min_length=1, max_length=4_096)
    status: Literal["hypothesis_only_not_admitted"]


class T2RunArtifactBinding(T2StrictModel):
    name: Literal["request.json", "task-packet.md", "result.json", "final.txt"]
    state: Literal["retained", "missing", "empty", "oversized", "nonregular", "unreadable"]
    retained_filename: str | None = Field(default=None, max_length=512)
    sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    byte_count: int | None = Field(default=None, ge=1, le=131_072)
    rejection_detail: str | None = Field(default=None, min_length=1, max_length=1_024)

    @model_validator(mode="after")
    def validate_state(self) -> T2RunArtifactBinding:
        retained = self.state == "retained"
        identity = (self.retained_filename, self.sha256, self.byte_count)
        if retained and any(value is None for value in identity):
            raise ValueError("only retained run artifacts carry byte identities")
        if not retained and any(value is not None for value in identity):
            raise ValueError("only retained run artifacts carry byte identities")
        if retained == (self.rejection_detail is not None):
            raise ValueError("only rejected reads carry rejection detail")
        if self.retained_filename is not None and (
            self.retained_filename != self.retained_filename.strip()
            or "/" in self.retained_filename
            or "\\" in self.retained_filename
            or "\x00" in self.retained_filename
        ):
            raise ValueError("retained filename must be one safe filename")
        return self


class T2ModelCallReceipt(T2StrictModel):
    schema_version: Literal[3]
    kind: Literal["t2_initial_model_call_receipt"]
    source_class: Literal["local_detached_sol_retained_run_v1"]
    disposition: Literal["parameter_proposal_format_accepted", "parameter_proposal_rejected"]
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    run_artifacts: list[T2RunArtifactBinding] = Field(min_length=4, max_length=4)
    original_packet_sha256: str = Field(pattern=SHA256_PATTERN)
    original_packet_byte_count: int = Field(ge=1, le=131_072)
    record_sha256: str = Field(pattern=SHA256_PATTERN)
    record_byte_count: int = Field(ge=1, le=131_072)
    request_payload_sha256: str = Field(pattern=SHA256_PATTERN)
    baseline_sha256: str = Field(pattern=SHA256_PATTERN)
    baseline_byte_count: int = Field(ge=1, le=131_072)
    t2_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_dossier_sha256: str = Field(pattern=SHA256_PATTERN)
    expected_model: Literal["gpt-5.6-sol"]
    expected_reasoning_effort: Literal["max"]
    expected_runner_kind: Literal["screen"]
    expected_owner: Literal["astra-f3-initial"]
    expected_role: Literal["candidate"]
    expected_scope: Literal["t2-initial-parameter-hypothesis-only"]
    result_thread_id: str | None = Field(default=None, max_length=512)
    proposal_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    candidate_recipe_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    rejection_reason: str | None = Field(default=None, min_length=1, max_length=4_096)
    metadata_semantics: Literal["expected_configuration_not_served_model_attestation"]
    local_envelope_consistency: Literal["verified", "not_verified"]
    authenticated_model_origin: Literal["not_attested"]
    candidate_reward_admission: Literal["missing"]
    adapter_admission: Literal["missing"]
    execution_manifest: Literal["missing"]
    protected_evaluator_results: Literal["missing"]
    candidate_measurements: Literal["missing"]
    format_validation: Literal["accepted_hypothesis_only", "not_admitted"]
    runtime_authorization: Literal["not_authorized"]
    training_authorization: Literal["not_authorized"]
    measured_improvement: Literal["not_measured"]

    @model_validator(mode="after")
    def validate_receipt(self) -> T2ModelCallReceipt:
        if type(self.run_artifacts) is not list:
            raise ValueError("run artifact bindings must be an exact list")
        if tuple(item.name for item in self.run_artifacts) != RUN_ARTIFACT_NAMES:
            raise ValueError("run artifact bindings must preserve the fixed artifact order")
        accepted = self.disposition == "parameter_proposal_format_accepted"
        identities = (self.proposal_sha256, self.candidate_recipe_sha256)
        if accepted and any(value is None for value in identities):
            raise ValueError("only format-accepted receipts identify proposal and recipe artifacts")
        if not accepted and any(value is not None for value in identities):
            raise ValueError("only format-accepted receipts identify proposal and recipe artifacts")
        if accepted == (self.rejection_reason is not None):
            raise ValueError("only rejected receipts require a rejection reason")
        envelope_verified = self.local_envelope_consistency == "verified"
        if accepted and not envelope_verified:
            raise ValueError("format acceptance requires a verified local envelope")
        expected_validation = "accepted_hypothesis_only" if accepted else "not_admitted"
        if self.format_validation != expected_validation:
            raise ValueError("format-validation state differs from receipt disposition")
        if accepted and any(item.state != "retained" for item in self.run_artifacts):
            raise ValueError("accepted receipt requires all four retained run artifacts")
        if envelope_verified != (
            self.result_thread_id is not None and bool(self.result_thread_id.strip())
        ):
            raise ValueError("only a verified local envelope carries a terminal thread ID")
        return self


__all__ = [
    "MISSING_EVIDENCE",
    "RUN_ARTIFACT_NAMES",
    "T2ContractBundle",
    "T2InitialEvidenceDossier",
    "T2InitialPacketRecord",
    "T2InitialRequest",
    "T2ModelCallReceipt",
    "T2ParameterProposal",
    "T2ParameterValues",
    "T2RetainedArtifact",
    "T2RunArtifactBinding",
]
