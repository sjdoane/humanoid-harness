"""Strict records for the initial T2 parameter-proposal protocol."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

T2_MODEL_SCHEMA_VERSION = 4
SHA256_PATTERN = r"^[0-9a-f]{64}$"

F3_CALL_PROTOCOL_ID = "f3_initial_t2_one_call/v1"
T2_SEAL_ARTIFACT_ID = "experiments/004_t2_reward_study/t2_seal_v1.json"
CALL_IDENTITY_FILENAME = "call-identity.json"
DISPATCH_INTENT_FILENAME = "dispatch-intent.json"
INGESTION_CLAIM_FILENAME = "ingestion-claim.json"

RUN_ARTIFACT_NAMES = ("request.json", "task-packet.md", "result.json", "final.txt")
MISSING_EVIDENCE = (
    "candidate_reward_admission",
    "verified_com_adapter_origin_cadence_gate",
    "execution_manifest",
    "protected_evaluator_results",
    "candidate_measurements",
)

EXPECTED_CHECKOUT_REALPATH = "/Users/samueldoane/Documents/ChatGPT/humanoid-harness"
EXPECTED_BRANCH = "main"
EXPECTED_IMPORT_ORIGIN = (
    "/Users/samueldoane/Documents/ChatGPT/humanoid-harness/src/oracle_composition/__init__.py"
)
EXPECTED_PREPARATION_OWNER = "fable-f3-prepare"
EXPECTED_PREPARATION_ROLE = "builder"
EXPECTED_PREPARATION_SCOPE = "t2-initial-packet-preparation-and-ingestion-only"
EXPECTED_CANDIDATE_OWNER = "fable-f3-initial"
EXPECTED_CANDIDATE_ROLE = "candidate"
EXPECTED_CANDIDATE_SCOPE = "t2-initial-parameter-hypothesis-only"
EXPECTED_MODEL = "gpt-5.6-sol"
EXPECTED_REASONING_EFFORT = "max"
EXPECTED_RUNNER_KIND = "screen"
EXPECTED_METADATA_SEMANTICS = "expected_configuration_not_served_model_attestation"
PINNED_BASELINE_SHA256 = "eea2b6a9893e6e4ca5aea5d6787580f12062084e758cc9c27db2f1376bcb1c5f"
PINNED_BASELINE_BYTE_COUNT = 1_773

_CALL_IDENTITY_SCOPE_MARKER = ";call_identity_sha256="
_T2_SEALED_PATHS = {
    "execution_manifest": "experiments/004_t2_reward_study/execution_manifest_t2_v1.json",
    "expert_hold_oracle": "experiments/004_t2_reward_study/oracle_expert_hold_v1.json",
    "training_design": "experiments/004_t2_reward_study/training_design_t2_v1.json",
    "evaluator_design": "experiments/004_t2_reward_study/evaluator_design_t2_v1.json",
    "study_manifest": ("experiments/004_t2_reward_study/t2_reward_study_expert_hold_v1.json"),
    "tracking_only_baseline": (
        "experiments/003_composition_speed_profile/phase_b/tracking_only_v1.json"
    ),
}


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def derive_t2_call_id(*, study_manifest_sha256: str, rendered_prompt_sha256: str) -> str:
    """Derive the protocol call ID from three unambiguous, versioned fields."""

    if not _is_sha256(study_manifest_sha256) or not _is_sha256(rendered_prompt_sha256):
        raise ValueError("call-ID inputs must be lowercase SHA-256 values")
    material = (
        F3_CALL_PROTOCOL_ID.encode("ascii")
        + b"\x00"
        + study_manifest_sha256.encode("ascii")
        + b"\x00"
        + rendered_prompt_sha256.encode("ascii")
    )
    return hashlib.sha256(material).hexdigest()


def expected_candidate_scope(call_identity_sha256: str) -> str:
    if not _is_sha256(call_identity_sha256):
        raise ValueError("call identity must be a lowercase SHA-256")
    return EXPECTED_CANDIDATE_SCOPE + _CALL_IDENTITY_SCOPE_MARKER + call_identity_sha256


def _canonical_model_sha256(model: BaseModel) -> str:
    encoded = (
        json.dumps(
            model.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class T2StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class T2SealedArtifact(T2StrictModel):
    path: str = Field(min_length=1, max_length=512)
    sha256: str = Field(pattern=SHA256_PATTERN)
    byte_count: int = Field(ge=1, le=262_144)

    @field_validator("path", mode="before")
    @classmethod
    def validate_canonical_path(cls, value: object) -> str:
        if type(value) is not str:
            raise ValueError("sealed artifact path must be exact text")
        parsed = PurePosixPath(value)
        if (
            parsed.is_absolute()
            or str(parsed) != value
            or any(part in {"", ".", ".."} for part in parsed.parts)
            or "\\" in value
            or "\x00" in value
        ):
            raise ValueError("sealed artifact path must be canonical and repository-relative")
        return value


class T2PreDispatchSeal(T2StrictModel):
    schema_version: Literal[1]
    kind: Literal["t2_pre_dispatch_seal"]
    seal_id: Literal["f3_t2_pre_dispatch_seal/v1"]
    study_id: Literal["t2_reward_study_expert_hold/v1"]
    execution_manifest: T2SealedArtifact
    expert_hold_oracle: T2SealedArtifact
    training_design: T2SealedArtifact
    evaluator_design: T2SealedArtifact
    study_manifest: T2SealedArtifact
    tracking_only_baseline: T2SealedArtifact
    study_pairing_sha256: str = Field(pattern=SHA256_PATTERN)
    pairing_receipt: str = Field(pattern=SHA256_PATTERN)
    dispatch_state: Literal[
        "withheld_pending_dispatch_verdict",
        "candidate_admitted_no_further_initial_dispatch",
    ]

    @model_validator(mode="after")
    def validate_frozen_paths(self) -> T2PreDispatchSeal:
        artifacts = {
            "execution_manifest": self.execution_manifest,
            "expert_hold_oracle": self.expert_hold_oracle,
            "training_design": self.training_design,
            "evaluator_design": self.evaluator_design,
            "study_manifest": self.study_manifest,
            "tracking_only_baseline": self.tracking_only_baseline,
        }
        if any(type(value) is not T2SealedArtifact for value in artifacts.values()):
            raise ValueError("T2 seal artifacts must use the exact sealed-artifact type")
        if any(value.path != _T2_SEALED_PATHS[key] for key, value in artifacts.items()):
            raise ValueError("T2 seal artifact path differs from the frozen protocol")
        if len({value.path for value in artifacts.values()}) != len(artifacts):
            raise ValueError("T2 seal artifact paths must be unique")
        if (
            self.tracking_only_baseline.sha256 != PINNED_BASELINE_SHA256
            or self.tracking_only_baseline.byte_count != PINNED_BASELINE_BYTE_COUNT
        ):
            raise ValueError("T2 seal baseline differs from the pinned tracking-only artifact")
        return self


class T2CallIdentity(T2StrictModel):
    schema_version: Literal[1]
    kind: Literal["t2_initial_call_identity"]
    protocol_id: Literal["f3_initial_t2_one_call/v1"]
    study_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    rendered_prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    t2_seal_sha256: str = Field(pattern=SHA256_PATTERN)
    call_id: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_call_id(self) -> T2CallIdentity:
        expected = derive_t2_call_id(
            study_manifest_sha256=self.study_manifest_sha256,
            rendered_prompt_sha256=self.rendered_prompt_sha256,
        )
        if self.call_id != expected:
            raise ValueError("call_id differs from the fixed F3 derivation")
        return self


class T2FableExpectedConfiguration(T2StrictModel):
    expected_checkout_realpath: Literal["/Users/samueldoane/Documents/ChatGPT/humanoid-harness"]
    expected_branch: Literal["main"]
    expected_import_origin: Literal[
        "/Users/samueldoane/Documents/ChatGPT/humanoid-harness/src/oracle_composition/__init__.py"
    ]
    expected_preparation_owner: Literal["fable-f3-prepare"]
    expected_preparation_role: Literal["builder"]
    expected_preparation_scope: Literal["t2-initial-packet-preparation-and-ingestion-only"]
    expected_model: Literal["gpt-5.6-sol"]
    expected_reasoning_effort: Literal["max"]
    expected_runner_kind: Literal["screen"]
    expected_owner: Literal["fable-f3-initial"]
    expected_role: Literal["candidate"]
    expected_scope: str = Field(min_length=1, max_length=2_048)
    metadata_semantics: Literal["expected_configuration_not_served_model_attestation"]

    @field_validator("expected_scope", mode="before")
    @classmethod
    def validate_call_bound_scope(cls, value: object) -> str:
        if type(value) is not str or not value.startswith(
            EXPECTED_CANDIDATE_SCOPE + _CALL_IDENTITY_SCOPE_MARKER
        ):
            raise ValueError("candidate scope lacks the call-identity binding")
        digest = value.removeprefix(EXPECTED_CANDIDATE_SCOPE + _CALL_IDENTITY_SCOPE_MARKER)
        if not _is_sha256(digest):
            raise ValueError("candidate scope has an invalid call-identity binding")
        return value


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
    task_text: Literal["Hold 3.0 m/s COM forward speed from the expert start."]
    baseline_artifact_id: Literal[
        "git:8d617e30dd239529a42e3a0211314d6173ffeafd:"
        "experiments/003_composition_speed_profile/phase_b/tracking_only_v1.json"
    ]
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


class T2InitialPacketRecord(T2FableExpectedConfiguration):
    schema_version: Literal[4]
    kind: Literal["t2_initial_parameter_packet_record"]
    baseline: T2RetainedArtifact
    t2_seal: T2RetainedArtifact
    contracts: T2ContractBundle
    evidence_dossier: T2InitialEvidenceDossier
    semantic_request: T2InitialRequest
    call_identity: T2CallIdentity
    call_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    study_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    request_payload_sha256: str = Field(pattern=SHA256_PATTERN)
    baseline_sha256: str = Field(pattern=SHA256_PATTERN)
    t2_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_dossier_sha256: str = Field(pattern=SHA256_PATTERN)
    rendered_prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    rendered_prompt_byte_count: int = Field(ge=1, le=131_072)

    @model_validator(mode="after")
    def validate_non_model_facing_bindings(self) -> T2InitialPacketRecord:
        if self.t2_seal.artifact_id != T2_SEAL_ARTIFACT_ID:
            raise ValueError("packet T2 seal identity differs")
        if (
            self.call_identity.t2_seal_sha256 != self.t2_seal.sha256
            or self.call_identity.study_manifest_sha256 != self.study_manifest_sha256
            or self.call_identity.rendered_prompt_sha256 != self.rendered_prompt_sha256
            or self.call_identity_sha256 != _canonical_model_sha256(self.call_identity)
            or self.expected_scope != expected_candidate_scope(self.call_identity_sha256)
        ):
            raise ValueError("packet call-identity binding differs")
        return self


class T2InitialDispatchIntent(T2FableExpectedConfiguration):
    schema_version: Literal[4]
    kind: Literal["t2_initial_dispatch_intent"]
    call_id: str = Field(pattern=SHA256_PATTERN)
    call_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    t2_seal_sha256: str = Field(pattern=SHA256_PATTERN)
    study_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    maximum_initial_calls: Literal[1]
    attempts_remaining_after_intent: Literal[0]
    revision_calls_authorized: Literal[0]
    retries_authorized: Literal[0]
    deadline_seconds_from_request_creation: Literal[1200]
    record_sha256: str = Field(pattern=SHA256_PATTERN)
    record_byte_count: int = Field(ge=1, le=131_072)
    rendered_prompt_sha256: str = Field(pattern=SHA256_PATTERN)
    rendered_prompt_byte_count: int = Field(ge=1, le=131_072)
    baseline_sha256: Literal["eea2b6a9893e6e4ca5aea5d6787580f12062084e758cc9c27db2f1376bcb1c5f"]
    baseline_byte_count: Literal[1773]
    request_payload_sha256: str = Field(pattern=SHA256_PATTERN)
    t2_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_dossier_sha256: str = Field(pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def validate_call_binding(self) -> T2InitialDispatchIntent:
        if self.call_id != derive_t2_call_id(
            study_manifest_sha256=self.study_manifest_sha256,
            rendered_prompt_sha256=self.rendered_prompt_sha256,
        ) or self.expected_scope != expected_candidate_scope(self.call_identity_sha256):
            raise ValueError("dispatch intent call-identity binding differs")
        return self


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


class T2IngestionClaim(T2StrictModel):
    schema_version: Literal[1]
    kind: Literal["t2_initial_ingestion_claim"]
    call_id: str = Field(pattern=SHA256_PATTERN)
    call_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    dispatch_intent_sha256: str = Field(pattern=SHA256_PATTERN)
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    run_artifacts_sha256: str = Field(pattern=SHA256_PATTERN)


class T2ProtocolRefusalReceipt(T2StrictModel):
    schema_version: Literal[1]
    kind: Literal["t2_initial_protocol_refusal_receipt"]
    stage: Literal["call_identity_claim", "dispatch_intent"]
    reason_code: Literal[
        "canonical_call_identity_already_claimed",
        "canonical_call_identity_conflict",
        "noncanonical_dispatch_intent_path",
        "canonical_dispatch_intent_already_exists",
    ]
    call_id: str = Field(pattern=SHA256_PATTERN)
    call_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    attempted_path_class: Literal["canonical", "noncanonical"]
    canonical_identity_path: Literal["call-identity.json"]
    canonical_dispatch_intent_path: Literal["dispatch-intent.json"]
    repository_claim: Literal[
        "canonical_call_identity_claim_refused_external_call_absence_not_proven"
    ]


class T2ModelCallReceipt(T2FableExpectedConfiguration):
    schema_version: Literal[4]
    kind: Literal["t2_initial_model_call_receipt"]
    source_class: Literal["local_detached_sol_retained_run_v1"]
    disposition: Literal["parameter_proposal_format_accepted", "parameter_proposal_rejected"]
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    run_artifacts: list[T2RunArtifactBinding] = Field(min_length=4, max_length=4)
    call_id: str = Field(pattern=SHA256_PATTERN)
    call_identity_sha256: str = Field(pattern=SHA256_PATTERN)
    t2_seal_sha256: str = Field(pattern=SHA256_PATTERN)
    study_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    dispatch_intent_consistency: Literal["verified", "missing", "mismatched"]
    dispatch_intent_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    ingestion_claim_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    replay_refusal: bool
    original_packet_sha256: str = Field(pattern=SHA256_PATTERN)
    original_packet_byte_count: int = Field(ge=1, le=131_072)
    record_sha256: str = Field(pattern=SHA256_PATTERN)
    record_byte_count: int = Field(ge=1, le=131_072)
    request_payload_sha256: str = Field(pattern=SHA256_PATTERN)
    baseline_sha256: Literal["eea2b6a9893e6e4ca5aea5d6787580f12062084e758cc9c27db2f1376bcb1c5f"]
    baseline_byte_count: Literal[1773]
    t2_contract_sha256: str = Field(pattern=SHA256_PATTERN)
    evidence_dossier_sha256: str = Field(pattern=SHA256_PATTERN)
    result_thread_id: str | None = Field(default=None, max_length=512)
    proposal_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    candidate_recipe_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)
    rejection_reason: str | None = Field(default=None, min_length=1, max_length=4_096)
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
        if self.call_id != derive_t2_call_id(
            study_manifest_sha256=self.study_manifest_sha256,
            rendered_prompt_sha256=self.original_packet_sha256,
        ) or self.expected_scope != expected_candidate_scope(self.call_identity_sha256):
            raise ValueError("call receipt call-identity binding differs")
        if self.dispatch_intent_consistency == "missing" and (
            self.dispatch_intent_sha256 is not None
        ):
            raise ValueError("dispatch-intent state and digest differ")
        if self.dispatch_intent_consistency == "verified" and (self.dispatch_intent_sha256 is None):
            raise ValueError("dispatch-intent state and digest differ")
        if self.ingestion_claim_sha256 is not None and (
            self.dispatch_intent_consistency != "verified"
        ):
            raise ValueError("only a verified dispatch intent can carry an ingestion claim")
        if accepted and (
            self.dispatch_intent_consistency != "verified"
            or self.ingestion_claim_sha256 is None
            or self.replay_refusal
        ):
            raise ValueError("format acceptance requires the first canonical ingestion claim")
        if self.replay_refusal and (
            accepted
            or self.dispatch_intent_consistency != "verified"
            or self.ingestion_claim_sha256 is None
        ):
            raise ValueError("replay refusal requires the existing canonical ingestion claim")
        return self


__all__ = [
    "CALL_IDENTITY_FILENAME",
    "DISPATCH_INTENT_FILENAME",
    "EXPECTED_BRANCH",
    "EXPECTED_CANDIDATE_OWNER",
    "EXPECTED_CANDIDATE_ROLE",
    "EXPECTED_CANDIDATE_SCOPE",
    "EXPECTED_CHECKOUT_REALPATH",
    "EXPECTED_IMPORT_ORIGIN",
    "EXPECTED_METADATA_SEMANTICS",
    "EXPECTED_MODEL",
    "EXPECTED_PREPARATION_OWNER",
    "EXPECTED_PREPARATION_ROLE",
    "EXPECTED_PREPARATION_SCOPE",
    "EXPECTED_REASONING_EFFORT",
    "EXPECTED_RUNNER_KIND",
    "F3_CALL_PROTOCOL_ID",
    "INGESTION_CLAIM_FILENAME",
    "MISSING_EVIDENCE",
    "PINNED_BASELINE_BYTE_COUNT",
    "PINNED_BASELINE_SHA256",
    "RUN_ARTIFACT_NAMES",
    "T2_SEAL_ARTIFACT_ID",
    "T2CallIdentity",
    "T2ContractBundle",
    "T2FableExpectedConfiguration",
    "T2IngestionClaim",
    "T2InitialDispatchIntent",
    "T2InitialEvidenceDossier",
    "T2InitialPacketRecord",
    "T2InitialRequest",
    "T2ModelCallReceipt",
    "T2ParameterProposal",
    "T2ParameterValues",
    "T2PreDispatchSeal",
    "T2ProtocolRefusalReceipt",
    "T2RetainedArtifact",
    "T2RunArtifactBinding",
    "T2SealedArtifact",
    "derive_t2_call_id",
    "expected_candidate_scope",
]
