"""Initial-only T2 packet preparation and retained detached-run ingestion."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel

import oracle_composition
from oracle_composition.rewards.target_speed_formula import (
    FORMULA_ID,
    parse_target_speed_formula_recipe,
)
from oracle_composition.rewards.target_speed_formula_t2 import (
    FORMULA_RUNTIME_ID,
    PARSER_ID,
    target_speed_formula_t2_bounds,
)
from oracle_composition.rewards.task_inputs_v2 import (
    CONTROL_PERIOD_SECONDS,
    TARGET_SPEED_M_S,
    task_input_contract_v2,
)

from .contracts import DetachedSolRequest, DetachedSolResult
from .loop import RewardSearchError, _read_bounded_regular_file, parse_model_bytes
from .publication import (
    PublishedArtifact,
    finite_pretty_json,
    publish_bytes_without_overwrite,
)
from .t2_model_contracts import (
    CALL_IDENTITY_FILENAME,
    DISPATCH_INTENT_FILENAME,
    EXPECTED_BRANCH,
    EXPECTED_CANDIDATE_OWNER,
    EXPECTED_CANDIDATE_ROLE,
    EXPECTED_CHECKOUT_REALPATH,
    EXPECTED_IMPORT_ORIGIN,
    EXPECTED_METADATA_SEMANTICS,
    EXPECTED_MODEL,
    EXPECTED_PREPARATION_OWNER,
    EXPECTED_PREPARATION_ROLE,
    EXPECTED_PREPARATION_SCOPE,
    EXPECTED_REASONING_EFFORT,
    EXPECTED_RUNNER_KIND,
    F3_CALL_PROTOCOL_ID,
    INGESTION_CLAIM_FILENAME,
    PINNED_BASELINE_BYTE_COUNT,
    PINNED_BASELINE_SHA256,
    RUN_ARTIFACT_NAMES,
    T2_SEAL_ARTIFACT_ID,
    T2CallIdentity,
    T2ContractBundle,
    T2FableExpectedConfiguration,
    T2IngestionClaim,
    T2InitialDispatchIntent,
    T2InitialEvidenceDossier,
    T2InitialPacketRecord,
    T2InitialRequest,
    T2ModelCallReceipt,
    T2ParameterProposal,
    T2PreDispatchSeal,
    T2ProtocolRefusalReceipt,
    T2RetainedArtifact,
    T2RunArtifactBinding,
    derive_t2_call_id,
    expected_candidate_scope,
)

MAX_RECORD_BYTES = 131_072
MAX_PACKET_BYTES = 131_072
MAX_JSON_BYTES = 65_536
MAX_FINAL_BYTES = 65_536
MAX_LOCAL_ARTIFACT_BYTES = 262_144
MAX_RUN_SECONDS = 1_200
MAX_PROTOCOL_RECEIPT_BYTES = 65_536

BASELINE_GIT_OBJECT = "8d617e30dd239529a42e3a0211314d6173ffeafd"
BASELINE_GIT_PATH = "experiments/003_composition_speed_profile/phase_b/tracking_only_v1.json"
BASELINE_ARTIFACT_ID = f"git:{BASELINE_GIT_OBJECT}:{BASELINE_GIT_PATH}"
BASELINE_SHA256 = PINNED_BASELINE_SHA256
BASELINE_BYTE_COUNT = PINNED_BASELINE_BYTE_COUNT

RECIPE_SOURCE_ID = "src/oracle_composition/rewards/target_speed_formula.py"
EVALUATOR_SOURCE_ID = "src/oracle_composition/rewards/target_speed_formula_t2.py"
TASK_INPUTS_SOURCE_ID = "src/oracle_composition/rewards/task_inputs_v2.py"
TASK_INPUTS_SCHEMA_ID = "canonical/task_input_contract_v2.json"
PARAMETER_BOUNDS_ID = "canonical/target_speed_formula_t2_bounds.json"

RECIPE_SOURCE_SHA256 = "c60dea03e6f0e71b81875fea59c84bd8fe00ce39f94ac7c54ca6e2a57360dccb"
EVALUATOR_SOURCE_SHA256 = "064393887bb4a7157d614981cc2000940252e12c31ad0aabc13109737fd94301"
TASK_INPUTS_SOURCE_SHA256 = "9607f2d56a54922eac06ec7fcc740b79e68d4ed9e88b192602aae2900fb3f0d2"
TASK_INPUTS_SCHEMA_SHA256 = "8f382dde13ee44c27cbbbc0b3a53a568338f6b7cebc8e660e4084425b7b494e9"
PARAMETER_BOUNDS_SHA256 = "2c6030264231a52320a44fe0f4d4ed519bc2e635b892f1b9c0d8929166106933"
PARAMETER_BOUNDS_BYTE_COUNT = 137

EXPECTED_OWNER = EXPECTED_CANDIDATE_OWNER
EXPECTED_ROLE = EXPECTED_CANDIDATE_ROLE
EXPECTED_EFFORT = EXPECTED_REASONING_EFFORT
EXPECTED_RUNNER = EXPECTED_RUNNER_KIND

_ROOT = Path(__file__).resolve().parents[3]
CALL_ROOT = Path(EXPECTED_CHECKOUT_REALPATH) / ".orchestration/f3-initial-call-20260906"
_NATIVE_PATH_TYPE = type(Path())
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RUN_FILE_LIMITS = {
    "request.json": MAX_JSON_BYTES,
    "task-packet.md": MAX_PACKET_BYTES,
    "result.json": MAX_JSON_BYTES,
    "final.txt": MAX_FINAL_BYTES,
}
_RUN_FILE_SLUGS = {
    "request.json": "request-json",
    "task-packet.md": "task-packet-md",
    "result.json": "result-json",
    "final.txt": "final-txt",
}
_EXPECTED_SOURCE_HASHES = {
    RECIPE_SOURCE_ID: RECIPE_SOURCE_SHA256,
    EVALUATOR_SOURCE_ID: EVALUATOR_SOURCE_SHA256,
    TASK_INPUTS_SOURCE_ID: TASK_INPUTS_SOURCE_SHA256,
}
_T2_SEAL_FIELDS = (
    "expert_hold_oracle",
    "training_design",
    "evaluator_design",
    "study_manifest",
    "tracking_only_baseline",
)


class T2ProtocolError(ValueError):
    """A T2 record, envelope, or proposal violates the initial-only boundary."""


@dataclass(frozen=True, slots=True)
class T2PublishedPacket:
    packet: PublishedArtifact
    record: PublishedArtifact


@dataclass(frozen=True, slots=True)
class T2IngestionOutcome:
    accepted: bool
    retained_run_artifacts: tuple[PublishedArtifact, ...]
    proposal: PublishedArtifact | None
    recipe: PublishedArtifact | None
    receipt: PublishedArtifact
    rejection_reason: str | None


def _compact_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise T2ProtocolError("invalid T2 protocol data") from exc


def _model_bytes(model: BaseModel) -> bytes:
    return finite_pretty_json(model.model_dump(mode="json"))


def _model_sha256(model: BaseModel) -> str:
    return hashlib.sha256(_model_bytes(model)).hexdigest()


def _expected_configuration(call_identity_sha256: str) -> T2FableExpectedConfiguration:
    return T2FableExpectedConfiguration(
        expected_checkout_realpath=EXPECTED_CHECKOUT_REALPATH,
        expected_branch=EXPECTED_BRANCH,
        expected_import_origin=EXPECTED_IMPORT_ORIGIN,
        expected_preparation_owner=EXPECTED_PREPARATION_OWNER,
        expected_preparation_role=EXPECTED_PREPARATION_ROLE,
        expected_preparation_scope=EXPECTED_PREPARATION_SCOPE,
        expected_model=EXPECTED_MODEL,
        expected_reasoning_effort=EXPECTED_EFFORT,
        expected_runner_kind=EXPECTED_RUNNER,
        expected_owner=EXPECTED_OWNER,
        expected_role=EXPECTED_ROLE,
        expected_scope=expected_candidate_scope(call_identity_sha256),
        metadata_semantics=EXPECTED_METADATA_SEMANTICS,
    )


def _validate_local_origin() -> None:
    expected_checkout = Path(EXPECTED_CHECKOUT_REALPATH)
    expected_import = Path(EXPECTED_IMPORT_ORIGIN)
    package_spec = getattr(oracle_composition, "__spec__", None)
    package_file = getattr(oracle_composition, "__file__", None)
    spec_origin = getattr(package_spec, "origin", None)
    if type(package_file) is not str or type(spec_origin) is not str:
        raise T2ProtocolError("F3 package import origin is unavailable")
    try:
        observed_checkout = Path(__file__).resolve(strict=True).parents[3]
        observed_import = Path(package_file).resolve(strict=True)
        observed_spec_origin = Path(spec_origin).resolve(strict=True)
    except OSError as exc:
        raise T2ProtocolError("F3 package import origin is unavailable") from exc
    if (
        observed_checkout != expected_checkout
        or observed_import != expected_import
        or observed_spec_origin != expected_import
    ):
        raise T2ProtocolError("F3 package import origin differs from the expected Fable checkout")


def _snapshot(artifact_id: str, encoded: bytes) -> T2RetainedArtifact:
    if type(artifact_id) is not str or not artifact_id or artifact_id != artifact_id.strip():
        raise T2ProtocolError("invalid T2 artifact identity")
    if type(encoded) is not bytes or not encoded or len(encoded) > MAX_LOCAL_ARTIFACT_BYTES:
        raise T2ProtocolError("invalid T2 artifact bytes")
    return T2RetainedArtifact(
        artifact_id=artifact_id,
        sha256=hashlib.sha256(encoded).hexdigest(),
        byte_count=len(encoded),
        content_base64=base64.b64encode(encoded).decode("ascii"),
    )


def _retained_bytes(artifact: T2RetainedArtifact) -> bytes:
    if type(artifact) is not T2RetainedArtifact:
        raise T2ProtocolError("invalid T2 retained artifact")
    try:
        encoded = base64.b64decode(artifact.content_base64.encode("ascii"), validate=True)
    except (UnicodeError, binascii.Error) as exc:
        raise T2ProtocolError("invalid T2 retained artifact") from exc
    if (
        not encoded
        or len(encoded) != artifact.byte_count
        or hashlib.sha256(encoded).hexdigest() != artifact.sha256
    ):
        raise T2ProtocolError("invalid T2 retained artifact")
    return encoded


def _read_local_source(artifact_id: str) -> bytes:
    try:
        encoded = _read_bounded_regular_file(
            _ROOT / artifact_id,
            max_bytes=MAX_LOCAL_ARTIFACT_BYTES,
            label=artifact_id,
        )
    except RewardSearchError as exc:
        raise T2ProtocolError("accepted local T2 source is unavailable") from exc
    if hashlib.sha256(encoded).hexdigest() != _EXPECTED_SOURCE_HASHES[artifact_id]:
        raise T2ProtocolError("accepted local T2 source identity differs")
    return encoded


def _read_sealed_t2_artifact(path: str, *, byte_count: int, sha256: str) -> bytes:
    try:
        encoded = _read_bounded_regular_file(
            _ROOT / path,
            max_bytes=MAX_LOCAL_ARTIFACT_BYTES,
            label=path,
        )
    except RewardSearchError as exc:
        raise T2ProtocolError("sealed T2 artifact is unavailable") from exc
    if len(encoded) != byte_count or hashlib.sha256(encoded).hexdigest() != sha256:
        raise T2ProtocolError("sealed T2 artifact identity differs")
    return encoded


def _validate_t2_seal(encoded: bytes, *, baseline_reward_bytes: bytes) -> T2PreDispatchSeal:
    if type(encoded) is not bytes:
        raise T2ProtocolError("t2_seal_bytes must be exact builtin bytes")
    try:
        seal = parse_model_bytes(encoded, T2PreDispatchSeal, max_bytes=MAX_JSON_BYTES)
    except (RewardSearchError, RecursionError, TypeError, ValueError) as exc:
        raise T2ProtocolError("invalid T2 pre-dispatch seal") from exc
    if encoded != _model_bytes(seal):
        raise T2ProtocolError("T2 pre-dispatch seal is not canonical JSON")

    contents: dict[str, bytes] = {}
    for field in _T2_SEAL_FIELDS:
        binding = getattr(seal, field)
        contents[field] = _read_sealed_t2_artifact(
            binding.path,
            byte_count=binding.byte_count,
            sha256=binding.sha256,
        )
    if contents["tracking_only_baseline"] != baseline_reward_bytes:
        raise T2ProtocolError("seal and supplied tracking-only baseline bytes differ")

    manifest_bytes = contents["study_manifest"]
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeError, ValueError) as exc:
        raise T2ProtocolError("sealed T2 study manifest is not JSON") from exc
    if type(manifest) is not dict or _compact_json(manifest) != manifest_bytes:
        raise T2ProtocolError("sealed T2 study manifest is not canonical JSON")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("study_id") != seal.study_id
        or manifest.get("study_manifest_schema_id") != "t2_reward_study_manifest/v1"
        or manifest.get("status")
        != "execution_no_go_until_candidate_and_integrated_pairing_receipt_exist"
        or manifest.get("integrated_pairing_receipt_sha256") != "TBD"
        or manifest.get("study_pairing_sha256") != seal.study_pairing_sha256
    ):
        raise T2ProtocolError("sealed T2 study manifest identity or pending state differs")
    arms = manifest.get("arms")
    if type(arms) is not list or len(arms) != 2 or any(type(arm) is not dict for arm in arms):
        raise T2ProtocolError("sealed T2 study manifest arm structure differs")
    excluded_pairing_fields = {"arm_label", "output_path", "reward", "timestamps"}
    common_arms = [
        {key: value for key, value in arm.items() if key not in excluded_pairing_fields}
        for arm in arms
    ]
    if (
        common_arms[0] != common_arms[1]
        or hashlib.sha256(_compact_json(common_arms[0])).hexdigest() != seal.study_pairing_sha256
    ):
        raise T2ProtocolError("sealed T2 study pairing key differs from common arm fields")
    expected_common = {
        "oracle": seal.expert_hold_oracle.model_dump(mode="json"),
        "training_design": seal.training_design.model_dump(mode="json"),
        "evaluator": seal.evaluator_design.model_dump(mode="json"),
    }
    if any(any(arm.get(key) != value for key, value in expected_common.items()) for arm in arms):
        raise T2ProtocolError("T2 seal and study manifest common artifact bindings differ")
    baseline_reward = arms[0].get("reward")
    if type(baseline_reward) is not dict or baseline_reward != {
        "path": seal.tracking_only_baseline.path,
        "reward_id": "tracking_only/v1",
        "sha256": seal.tracking_only_baseline.sha256,
    }:
        raise T2ProtocolError("T2 seal and study manifest baseline binding differ")
    return seal


def _response_schema_sha256() -> str:
    return hashlib.sha256(
        finite_pretty_json(T2ParameterProposal.model_json_schema(mode="validation"))
    ).hexdigest()


def _contract_bundle() -> T2ContractBundle:
    task_schema = _compact_json(task_input_contract_v2())
    bounds = _compact_json(target_speed_formula_t2_bounds())
    if (
        hashlib.sha256(task_schema).hexdigest() != TASK_INPUTS_SCHEMA_SHA256
        or hashlib.sha256(bounds).hexdigest() != PARAMETER_BOUNDS_SHA256
        or len(bounds) != PARAMETER_BOUNDS_BYTE_COUNT
    ):
        raise T2ProtocolError("canonical T2 contract identity differs")
    return T2ContractBundle(
        schema_version=3,
        kind="t2_initial_contract_bundle",
        recipe_parser_source=_snapshot(RECIPE_SOURCE_ID, _read_local_source(RECIPE_SOURCE_ID)),
        evaluator_source=_snapshot(EVALUATOR_SOURCE_ID, _read_local_source(EVALUATOR_SOURCE_ID)),
        task_inputs_source=_snapshot(
            TASK_INPUTS_SOURCE_ID, _read_local_source(TASK_INPUTS_SOURCE_ID)
        ),
        canonical_task_inputs_schema=_snapshot(TASK_INPUTS_SCHEMA_ID, task_schema),
        canonical_parameter_bounds=_snapshot(PARAMETER_BOUNDS_ID, bounds),
        formula_id=FORMULA_ID,
        parser_id=PARSER_ID,
        runtime_id=FORMULA_RUNTIME_ID,
        response_schema_sha256=_response_schema_sha256(),
    )


def _evidence_dossier() -> T2InitialEvidenceDossier:
    return T2InitialEvidenceDossier(
        schema_version=3,
        kind="t2_initial_missing_evidence_dossier",
        evidence_stage="initial_only",
        aggregate_feedback=[],
        missing_evidence=[
            "candidate_reward_admission",
            "verified_com_adapter_origin_cadence_gate",
            "execution_manifest",
            "protected_evaluator_results",
            "candidate_measurements",
        ],
        baseline_evidence_class=(
            "declared_tracking_only_configuration_not_observed_successful_load"
        ),
        candidate_reward_admission="missing",
        verified_com_adapter_origin_cadence_gate="missing",
        required_adapter_gate=(
            "inclusive_-25.0_to_25.0_m_s_stock_com_origin_and_0.015_s_cadence_verification_missing"
        ),
        execution_manifest="missing",
        protected_evaluator_results="missing",
        candidate_measurements="missing",
        current_physical_admission="not_claimed",
    )


def _validate_baseline(encoded: bytes) -> None:
    if (
        type(encoded) is not bytes
        or len(encoded) != BASELINE_BYTE_COUNT
        or hashlib.sha256(encoded).hexdigest() != BASELINE_SHA256
    ):
        raise T2ProtocolError("baseline differs from the pinned tracking-only Git blob")


def _semantic_request(
    *,
    baseline: T2RetainedArtifact,
    contracts_sha256: str,
    dossier_sha256: str,
) -> T2InitialRequest:
    return T2InitialRequest(
        schema_version=3,
        kind="t2_initial_parameter_request",
        stage="initial_only",
        baseline_sha256=baseline.sha256,
        baseline_byte_count=baseline.byte_count,
        t2_contract_sha256=contracts_sha256,
        evidence_dossier_sha256=dossier_sha256,
        task_text="Hold 3.0 m/s COM forward speed from the expert start.",
        baseline_artifact_id=BASELINE_ARTIFACT_ID,
        target_speed_m_s=TARGET_SPEED_M_S,
        control_period_seconds=CONTROL_PERIOD_SECONDS,
        trusted_formula_id=FORMULA_ID,
        trusted_parser_id=PARSER_ID,
        trusted_runtime_id=FORMULA_RUNTIME_ID,
        formula_semantics=("r_task=alpha*1.25*(1-min(1,abs(com_x_velocity_m_s-3.0)/3.0))+beta"),
        alpha_closed_interval=[0.25, 4.0],
        beta_closed_interval=[-10.0, 10.0],
        authorable_parameters=["alpha", "beta"],
        baseline_role="tracking_only_configuration",
        baseline_load_evidence="not_observed",
        candidate_reward_admission="missing",
        required_adapter_gate=(
            "inclusive_-25.0_to_25.0_m_s_stock_com_origin_and_0.015_s_cadence_verification_missing"
        ),
        runtime_authorization="not_authorized",
        training_authorization="not_authorized",
        measured_improvement="not_measured",
    )


def _render_prompt(
    request_payload: bytes,
    *,
    request_payload_sha256: str,
    baseline_sha256: str,
    t2_contract_sha256: str,
    evidence_dossier_sha256: str,
) -> bytes:
    visible_identities = finite_pretty_json(
        {
            "request_payload_sha256": request_payload_sha256,
            "baseline_sha256": baseline_sha256,
            "t2_contract_sha256": t2_contract_sha256,
            "evidence_dossier_sha256": evidence_dossier_sha256,
        }
    ).decode("utf-8")
    response_schema = finite_pretty_json(
        T2ParameterProposal.model_json_schema(mode="validation")
    ).decode("utf-8")
    task_dossier = "\n".join(
        (
            "Task: Hold 3.0 m/s COM forward speed from the expert start.",
            f"Baseline artifact: {BASELINE_ARTIFACT_ID}",
            f"Baseline bytes/SHA-256: {BASELINE_BYTE_COUNT} / {BASELINE_SHA256}",
            (
                "Accepted F2 family: "
                f"{FORMULA_ID}; parser {PARSER_ID}; runtime {FORMULA_RUNTIME_ID}."
            ),
            f"F2 evaluator source SHA-256: {EVALUATOR_SOURCE_SHA256}",
            f"F2 recipe/parser source SHA-256: {RECIPE_SOURCE_SHA256}",
            f"T2 task-input source SHA-256: {TASK_INPUTS_SOURCE_SHA256}",
            (
                "F2 authorable bounds: alpha [0.25, 4.0], beta [-10.0, 10.0]; "
                f"canonical bounds SHA-256 {PARAMETER_BOUNDS_SHA256}."
            ),
            "Only alpha and beta are authorable.",
            "No measured feedback exists for this initial hypothesis call.",
        )
    )
    sections = [
        "# F3 initial-only T2 parameter hypothesis packet\n",
        (
            "Return exactly one JSON object matching the supplied schema and no other text.\n"
            "Echo the four visible immutable digests exactly. Author only alpha and beta. "
            "Do not return Python, source code, a formula ID, parser ID, runtime ID, output path, "
            "execution switch, retry, or fabricated evidence. The tracking-only baseline is a "
            "declared configuration, not an observed successful load. Its admission-contract "
            "binding is not an observed adapter certificate. All candidate admission, adapter, "
            "manifest, protected-evaluator, measurement, runtime, and training evidence remains "
            "missing. The required stock-COM adapter gate is inclusive [-25,25] m/s with verified "
            "origin and 0.015 s cadence. There is no parent recipe: alpha=1,beta=0 is not the "
            "baseline, and alpha=0 is outside this formula family. Any response is hypothesis-only "
            "and not admitted.\n"
        ),
        "## Initial T2 evidence dossier\n" + task_dossier,
        "## Visible immutable identities\n" + visible_identities,
        "## Canonical semantic request payload\n" + request_payload.decode("utf-8"),
        "## Required response JSON schema\n" + response_schema,
    ]
    encoded = "\n".join(sections).encode("utf-8")
    if not encoded or len(encoded) > MAX_PACKET_BYTES:
        raise T2ProtocolError("rendered T2 packet exceeds its byte limit")
    return encoded


def _build_record(
    baseline_reward_bytes: bytes,
    t2_seal_bytes: bytes,
) -> tuple[T2InitialPacketRecord, bytes, bytes]:
    _validate_local_origin()
    _validate_baseline(baseline_reward_bytes)
    seal = _validate_t2_seal(t2_seal_bytes, baseline_reward_bytes=baseline_reward_bytes)
    baseline = _snapshot(BASELINE_ARTIFACT_ID, baseline_reward_bytes)
    retained_seal = _snapshot(T2_SEAL_ARTIFACT_ID, t2_seal_bytes)
    contracts = _contract_bundle()
    dossier = _evidence_dossier()
    contracts_sha = _model_sha256(contracts)
    dossier_sha = _model_sha256(dossier)
    request = _semantic_request(
        baseline=baseline,
        contracts_sha256=contracts_sha,
        dossier_sha256=dossier_sha,
    )
    request_payload = _model_bytes(request)
    request_sha = hashlib.sha256(request_payload).hexdigest()
    prompt = _render_prompt(
        request_payload,
        request_payload_sha256=request_sha,
        baseline_sha256=baseline.sha256,
        t2_contract_sha256=contracts_sha,
        evidence_dossier_sha256=dossier_sha,
    )
    prompt_sha256 = hashlib.sha256(prompt).hexdigest()
    call_identity = T2CallIdentity(
        schema_version=1,
        kind="t2_initial_call_identity",
        protocol_id=F3_CALL_PROTOCOL_ID,
        study_manifest_sha256=seal.study_manifest.sha256,
        rendered_prompt_sha256=prompt_sha256,
        t2_seal_sha256=retained_seal.sha256,
        call_id=derive_t2_call_id(
            study_manifest_sha256=seal.study_manifest.sha256,
            rendered_prompt_sha256=prompt_sha256,
        ),
    )
    call_identity_sha256 = _model_sha256(call_identity)
    record = T2InitialPacketRecord(
        **_expected_configuration(call_identity_sha256).model_dump(mode="python"),
        schema_version=4,
        kind="t2_initial_parameter_packet_record",
        baseline=baseline,
        t2_seal=retained_seal,
        contracts=contracts,
        evidence_dossier=dossier,
        semantic_request=request,
        call_identity=call_identity,
        call_identity_sha256=call_identity_sha256,
        study_manifest_sha256=seal.study_manifest.sha256,
        request_payload_sha256=request_sha,
        baseline_sha256=baseline.sha256,
        t2_contract_sha256=contracts_sha,
        evidence_dossier_sha256=dossier_sha,
        rendered_prompt_sha256=prompt_sha256,
        rendered_prompt_byte_count=len(prompt),
    )
    record_bytes = _model_bytes(record)
    if len(record_bytes) > MAX_RECORD_BYTES:
        raise T2ProtocolError("T2 packet record exceeds its byte limit")
    return record, prompt, record_bytes


def _parse_exact_model_bytes[ModelT: BaseModel](
    encoded: bytes,
    model: type[ModelT],
    *,
    max_bytes: int,
    label: str,
) -> ModelT:
    if type(encoded) is not bytes:
        raise T2ProtocolError(f"invalid {label}")
    try:
        return parse_model_bytes(encoded, model, max_bytes=max_bytes)
    except (RewardSearchError, RecursionError, TypeError, ValueError) as exc:
        raise T2ProtocolError(f"invalid {label}") from exc


def _validated_record(record_bytes: bytes) -> tuple[T2InitialPacketRecord, bytes, bytes]:
    record = _parse_exact_model_bytes(
        record_bytes,
        T2InitialPacketRecord,
        max_bytes=MAX_RECORD_BYTES,
        label="T2 initial packet record",
    )
    baseline_bytes = _retained_bytes(record.baseline)
    t2_seal_bytes = _retained_bytes(record.t2_seal)
    expected, prompt, canonical_record_bytes = _build_record(baseline_bytes, t2_seal_bytes)
    if record != expected or record_bytes != canonical_record_bytes:
        raise T2ProtocolError("T2 initial packet record differs from reconstructed semantics")
    return expected, prompt, canonical_record_bytes


def prepare_initial_t2_packet(*, baseline_reward_bytes: bytes, t2_seal_bytes: bytes) -> bytes:
    """Create one canonical initial-only record for the pinned tracking-only baseline."""

    if type(baseline_reward_bytes) is not bytes:
        raise T2ProtocolError("baseline_reward_bytes must be exact builtin bytes")
    if type(t2_seal_bytes) is not bytes:
        raise T2ProtocolError("t2_seal_bytes must be exact builtin bytes")
    _, _, record_bytes = _build_record(baseline_reward_bytes, t2_seal_bytes)
    return record_bytes


def render_initial_t2_prompt(record_bytes: bytes) -> bytes:
    """Reconstruct and return the exact model-facing prompt from canonical record bytes."""

    if type(record_bytes) is not bytes:
        raise T2ProtocolError("record_bytes must be exact builtin bytes")
    _, prompt, _ = _validated_record(record_bytes)
    return prompt


def _require_native_path(value: object, *, label: str) -> Path:
    if type(value) is not _NATIVE_PATH_TYPE:
        raise T2ProtocolError(f"{label} must be an exact native Path")
    return value


def publish_initial_t2_packet(
    record_bytes: bytes,
    packet_path: Path,
    record_path: Path,
) -> T2PublishedPacket:
    """Publish the reconstructed prompt and canonical record without overwrite."""

    if type(record_bytes) is not bytes:
        raise T2ProtocolError("record_bytes must be exact builtin bytes")
    packet_path = _require_native_path(packet_path, label="packet_path")
    record_path = _require_native_path(record_path, label="record_path")
    _, prompt, canonical_record = _validated_record(record_bytes)
    if packet_path == record_path:
        raise T2ProtocolError("packet_path and record_path must differ")
    packet = publish_bytes_without_overwrite(packet_path, prompt)
    record = publish_bytes_without_overwrite(record_path, canonical_record)
    return T2PublishedPacket(packet=packet, record=record)


def _canonical_call_path(filename: str) -> Path:
    if type(CALL_ROOT) is not _NATIVE_PATH_TYPE or not CALL_ROOT.is_absolute():
        raise T2ProtocolError("configured F3 call root must be an exact absolute native Path")
    return CALL_ROOT / filename


def _publish_or_reuse_exact(path: Path, encoded: bytes) -> PublishedArtifact:
    try:
        return publish_bytes_without_overwrite(path, encoded)
    except ValueError:
        absolute = Path(os.path.abspath(path))
        try:
            existing = _read_bounded_regular_file(
                absolute,
                max_bytes=max(len(encoded), MAX_PROTOCOL_RECEIPT_BYTES),
                label=absolute.name,
                required_mode=0o600,
            )
        except RewardSearchError:
            raise
        if existing != encoded:
            raise
        return PublishedArtifact(
            path=absolute,
            sha256=hashlib.sha256(existing).hexdigest(),
            byte_count=len(existing),
        )


def _retain_protocol_refusal(
    record: T2InitialPacketRecord,
    *,
    stage: str,
    reason_code: str,
    attempted_path_class: str,
) -> PublishedArtifact:
    receipt = T2ProtocolRefusalReceipt(
        schema_version=1,
        kind="t2_initial_protocol_refusal_receipt",
        stage=stage,
        reason_code=reason_code,
        call_id=record.call_identity.call_id,
        call_identity_sha256=record.call_identity_sha256,
        attempted_path_class=attempted_path_class,
        canonical_identity_path=CALL_IDENTITY_FILENAME,
        canonical_dispatch_intent_path=DISPATCH_INTENT_FILENAME,
        repository_claim=("canonical_call_identity_claim_refused_external_call_absence_not_proven"),
    )
    encoded = _model_bytes(receipt)
    digest = hashlib.sha256(encoded).hexdigest()
    path = _canonical_call_path("refusals") / (
        f"t2-protocol-refusal-{stage}-{reason_code}-{digest}.json"
    )
    return _publish_or_reuse_exact(path, encoded)


def _expected_dispatch_intent(
    record: T2InitialPacketRecord,
    prompt: bytes,
    canonical_record: bytes,
) -> tuple[T2InitialDispatchIntent, bytes]:
    intent = T2InitialDispatchIntent(
        **_expected_configuration(record.call_identity_sha256).model_dump(mode="python"),
        schema_version=4,
        kind="t2_initial_dispatch_intent",
        call_id=record.call_identity.call_id,
        call_identity_sha256=record.call_identity_sha256,
        t2_seal_sha256=record.t2_seal.sha256,
        study_manifest_sha256=record.study_manifest_sha256,
        maximum_initial_calls=1,
        attempts_remaining_after_intent=0,
        revision_calls_authorized=0,
        retries_authorized=0,
        deadline_seconds_from_request_creation=MAX_RUN_SECONDS,
        record_sha256=hashlib.sha256(canonical_record).hexdigest(),
        record_byte_count=len(canonical_record),
        rendered_prompt_sha256=hashlib.sha256(prompt).hexdigest(),
        rendered_prompt_byte_count=len(prompt),
        baseline_sha256=record.baseline_sha256,
        baseline_byte_count=record.baseline.byte_count,
        request_payload_sha256=record.request_payload_sha256,
        t2_contract_sha256=record.t2_contract_sha256,
        evidence_dossier_sha256=record.evidence_dossier_sha256,
    )
    return intent, _model_bytes(intent)


def publish_initial_t2_dispatch_intent(
    record_bytes: bytes,
    intent_path: Path,
) -> PublishedArtifact:
    """Atomically claim the canonical call identity, then publish its sole intent."""

    if type(record_bytes) is not bytes:
        raise T2ProtocolError("record_bytes must be exact builtin bytes")
    intent_path = _require_native_path(intent_path, label="intent_path")
    record, prompt, canonical_record = _validated_record(record_bytes)
    canonical_intent_path = _canonical_call_path(DISPATCH_INTENT_FILENAME)
    if intent_path != canonical_intent_path:
        _retain_protocol_refusal(
            record,
            stage="dispatch_intent",
            reason_code="noncanonical_dispatch_intent_path",
            attempted_path_class="noncanonical",
        )
        raise T2ProtocolError("dispatch intent path is not the canonical call-root path")

    identity_path = _canonical_call_path(CALL_IDENTITY_FILENAME)
    identity_bytes = _model_bytes(record.call_identity)
    try:
        publish_bytes_without_overwrite(identity_path, identity_bytes)
    except ValueError as exc:
        reason_code = "canonical_call_identity_conflict"
        try:
            existing = _read_bounded_regular_file(
                identity_path,
                max_bytes=MAX_JSON_BYTES,
                label=CALL_IDENTITY_FILENAME,
                required_mode=0o600,
            )
        except RewardSearchError:
            existing = None
        if existing == identity_bytes:
            reason_code = "canonical_call_identity_already_claimed"
        _retain_protocol_refusal(
            record,
            stage="call_identity_claim",
            reason_code=reason_code,
            attempted_path_class="canonical",
        )
        raise T2ProtocolError("canonical call identity claim was refused") from exc

    _intent, intent_bytes = _expected_dispatch_intent(record, prompt, canonical_record)
    try:
        return publish_bytes_without_overwrite(canonical_intent_path, intent_bytes)
    except ValueError as exc:
        _retain_protocol_refusal(
            record,
            stage="dispatch_intent",
            reason_code="canonical_dispatch_intent_already_exists",
            attempted_path_class="canonical",
        )
        raise T2ProtocolError("canonical dispatch intent publication was refused") from exc


def _validate_run_paths(run_directory: Path, records_directory: Path) -> tuple[Path, Path, str]:
    run_directory = _require_native_path(run_directory, label="run_directory")
    records_directory = _require_native_path(records_directory, label="records_directory")
    run_id = run_directory.name
    if type(run_id) is not str or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise T2ProtocolError("run directory name is not a fixed safe run ID")
    try:
        state = run_directory.lstat()
        if not stat.S_ISDIR(state.st_mode) or run_directory.is_symlink():
            raise T2ProtocolError("run_directory must be a real directory")
        resolved_run = run_directory.resolve(strict=True)
    except OSError as exc:
        raise T2ProtocolError("run_directory must be a readable real directory") from exc
    if records_directory.exists() and (
        records_directory.is_symlink() or not records_directory.is_dir()
    ):
        raise T2ProtocolError("records_directory must be a real directory when present")
    return resolved_run, records_directory, run_id


def _read_failure_state(path: Path, *, limit: int) -> str:
    try:
        state = path.lstat()
    except FileNotFoundError:
        return "missing"
    except OSError:
        return "unreadable"
    if not stat.S_ISREG(state.st_mode) or path.is_symlink():
        return "nonregular"
    if state.st_size == 0:
        return "empty"
    if state.st_size > limit:
        return "oversized"
    return "unreadable"


def _retain_run_file(
    *,
    run_dir: Path,
    output: Path,
    run_id: str,
    name: str,
    max_bytes: int,
) -> tuple[T2RunArtifactBinding, PublishedArtifact | None, bytes | None]:
    path = run_dir / name
    try:
        encoded = _read_bounded_regular_file(
            path,
            max_bytes=max_bytes,
            label=name,
            required_mode=0o600,
        )
    except RewardSearchError as exc:
        detail = str(exc)[:1_024]
        return (
            T2RunArtifactBinding(
                name=name,
                state=_read_failure_state(path, limit=max_bytes),
                retained_filename=None,
                sha256=None,
                byte_count=None,
                rejection_detail=detail,
            ),
            None,
            None,
        )
    digest = hashlib.sha256(encoded).hexdigest()
    retained_name = f"t2-run-{run_id}-{_RUN_FILE_SLUGS[name]}-{digest}.bin"
    published = _publish_or_reuse_exact(output / retained_name, encoded)
    if published.sha256 != digest or published.byte_count != len(encoded):
        raise T2ProtocolError("published run artifact differs from bounded-read bytes")
    return (
        T2RunArtifactBinding(
            name=name,
            state="retained",
            retained_filename=retained_name,
            sha256=digest,
            byte_count=len(encoded),
            rejection_detail=None,
        ),
        published,
        encoded,
    )


def _utc_timestamp(value: str, *, label: str) -> datetime:
    if type(value) is not str or not value.endswith("Z"):
        raise T2ProtocolError(f"invalid {label}")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except (OverflowError, ValueError) as exc:
        raise T2ProtocolError(f"invalid {label}") from exc
    if parsed.tzinfo != UTC:
        raise T2ProtocolError(f"invalid {label}")
    return parsed


def _validate_local_envelope(
    record: T2InitialPacketRecord,
    prompt: bytes,
    raw: dict[str, bytes],
) -> str:
    if raw["task-packet.md"] != prompt:
        raise T2ProtocolError("retained task packet differs from the reconstructed prompt")
    request = _parse_exact_model_bytes(
        raw["request.json"],
        DetachedSolRequest,
        max_bytes=MAX_JSON_BYTES,
        label="detached request",
    )
    result = _parse_exact_model_bytes(
        raw["result.json"],
        DetachedSolResult,
        max_bytes=MAX_JSON_BYTES,
        label="detached result",
    )
    if (
        request.mode != "read-only"
        or request.owner != EXPECTED_OWNER
        or request.role != EXPECTED_ROLE
        or request.scope != record.expected_scope
        or request.requested_model != EXPECTED_MODEL
        or request.requested_reasoning_effort != EXPECTED_EFFORT
        or request.runner_kind != EXPECTED_RUNNER
    ):
        raise T2ProtocolError("detached request differs from the fixed launcher configuration")
    if (
        request.prompt_sha256 != record.rendered_prompt_sha256
        or request.prompt_bytes != record.rendered_prompt_byte_count
    ):
        raise T2ProtocolError("detached request prompt identity differs")
    if result.status != "SUCCEEDED" or result.exit_code != 0 or result.termination_escalated:
        raise T2ProtocolError("terminal result is not an un-escalated success")
    if result.thread_id is None or not result.thread_id.strip():
        raise T2ProtocolError("successful terminal result requires a thread ID")
    for field in (
        "requested_model",
        "requested_reasoning_effort",
        "runner_kind",
        "screen_name",
        "codex_cli_version",
        "role",
        "scope",
    ):
        if getattr(request, field) != getattr(result, field):
            raise T2ProtocolError(f"detached request/result {field} differs")
    started = _utc_timestamp(request.created_at_utc, label="request created_at_utc")
    finished = _utc_timestamp(result.finished_at_utc, label="result finished_at_utc")
    elapsed = finished - started
    if elapsed < timedelta(0) or elapsed > timedelta(seconds=MAX_RUN_SECONDS):
        raise T2ProtocolError("terminal result is outside the fixed 1,200-second deadline")
    return result.thread_id


def _validate_proposal(
    record: T2InitialPacketRecord,
    encoded: bytes,
) -> tuple[T2ParameterProposal, bytes]:
    proposal = _parse_exact_model_bytes(
        encoded,
        T2ParameterProposal,
        max_bytes=MAX_FINAL_BYTES,
        label="T2 parameter proposal",
    )
    for field, expected in {
        "request_payload_sha256": record.request_payload_sha256,
        "baseline_sha256": record.baseline_sha256,
        "t2_contract_sha256": record.t2_contract_sha256,
        "evidence_dossier_sha256": record.evidence_dossier_sha256,
    }.items():
        if getattr(proposal, field) != expected:
            raise T2ProtocolError(f"proposal {field} differs from the initial packet")
    recipe_bytes = _compact_json(
        {
            "formula_id": FORMULA_ID,
            "alpha": proposal.parameters.alpha,
            "beta": proposal.parameters.beta,
        }
    )
    try:
        trusted_recipe = parse_target_speed_formula_recipe(recipe_bytes)
    except ValueError as exc:
        raise T2ProtocolError("trusted T2 recipe parser rejected the parameter proposal") from exc
    if trusted_recipe.canonical_bytes != recipe_bytes:
        raise T2ProtocolError("trusted T2 recipe parser changed canonical parameter bytes")
    return proposal, recipe_bytes


def _receipt(
    *,
    record: T2InitialPacketRecord,
    prompt: bytes,
    canonical_record: bytes,
    run_id: str,
    bindings: list[T2RunArtifactBinding],
    proposal: T2ParameterProposal | None,
    recipe_bytes: bytes | None,
    result_thread_id: str | None,
    envelope_verified: bool,
    rejection_reason: str | None,
    dispatch_intent_consistency: str,
    dispatch_intent_sha256: str | None,
    ingestion_claim_sha256: str | None,
    replay_refusal: bool,
) -> T2ModelCallReceipt:
    accepted = proposal is not None and recipe_bytes is not None and rejection_reason is None
    return T2ModelCallReceipt(
        **_expected_configuration(record.call_identity_sha256).model_dump(mode="python"),
        schema_version=4,
        kind="t2_initial_model_call_receipt",
        source_class="local_detached_sol_retained_run_v1",
        disposition=(
            "parameter_proposal_format_accepted" if accepted else "parameter_proposal_rejected"
        ),
        run_id=run_id,
        run_artifacts=bindings,
        call_id=record.call_identity.call_id,
        call_identity_sha256=record.call_identity_sha256,
        t2_seal_sha256=record.t2_seal.sha256,
        study_manifest_sha256=record.study_manifest_sha256,
        dispatch_intent_consistency=dispatch_intent_consistency,
        dispatch_intent_sha256=dispatch_intent_sha256,
        ingestion_claim_sha256=ingestion_claim_sha256,
        replay_refusal=replay_refusal,
        original_packet_sha256=hashlib.sha256(prompt).hexdigest(),
        original_packet_byte_count=len(prompt),
        record_sha256=hashlib.sha256(canonical_record).hexdigest(),
        record_byte_count=len(canonical_record),
        request_payload_sha256=record.request_payload_sha256,
        baseline_sha256=record.baseline_sha256,
        baseline_byte_count=record.baseline.byte_count,
        t2_contract_sha256=record.t2_contract_sha256,
        evidence_dossier_sha256=record.evidence_dossier_sha256,
        result_thread_id=result_thread_id if envelope_verified else None,
        proposal_sha256=None if proposal is None else _model_sha256(proposal),
        candidate_recipe_sha256=(
            None if recipe_bytes is None else hashlib.sha256(recipe_bytes).hexdigest()
        ),
        rejection_reason=rejection_reason,
        local_envelope_consistency="verified" if envelope_verified else "not_verified",
        authenticated_model_origin="not_attested",
        candidate_reward_admission="missing",
        adapter_admission="missing",
        execution_manifest="missing",
        protected_evaluator_results="missing",
        candidate_measurements="missing",
        format_validation="accepted_hypothesis_only" if accepted else "not_admitted",
        runtime_authorization="not_authorized",
        training_authorization="not_authorized",
        measured_improvement="not_measured",
    )


def _inspect_dispatch_intent(
    record: T2InitialPacketRecord,
    prompt: bytes,
    canonical_record: bytes,
) -> tuple[str, str | None, str | None]:
    _intent, expected_intent = _expected_dispatch_intent(record, prompt, canonical_record)
    identity_path = _canonical_call_path(CALL_IDENTITY_FILENAME)
    intent_path = _canonical_call_path(DISPATCH_INTENT_FILENAME)
    try:
        identity_bytes = _read_bounded_regular_file(
            identity_path,
            max_bytes=MAX_JSON_BYTES,
            label=CALL_IDENTITY_FILENAME,
            required_mode=0o600,
        )
    except RewardSearchError:
        identity_bytes = None
    try:
        intent_bytes = _read_bounded_regular_file(
            intent_path,
            max_bytes=MAX_JSON_BYTES,
            label=DISPATCH_INTENT_FILENAME,
            required_mode=0o600,
        )
    except RewardSearchError:
        intent_bytes = None
    if intent_bytes is None:
        try:
            intent_path.lstat()
        except FileNotFoundError:
            return "missing", None, "canonical dispatch intent is missing"
        except OSError:
            pass
        return "mismatched", None, "canonical dispatch intent is unreadable"
    observed_sha256 = hashlib.sha256(intent_bytes).hexdigest()
    if identity_bytes != _model_bytes(record.call_identity) or intent_bytes != expected_intent:
        return "mismatched", observed_sha256, "canonical dispatch intent binding differs"
    return "verified", observed_sha256, None


def _claim_ingestion(
    record: T2InitialPacketRecord,
    *,
    dispatch_intent_sha256: str,
    run_id: str,
    bindings: list[T2RunArtifactBinding],
) -> tuple[str, bool]:
    run_artifacts_sha256 = hashlib.sha256(
        _compact_json([item.model_dump(mode="json") for item in bindings])
    ).hexdigest()
    claim = T2IngestionClaim(
        schema_version=1,
        kind="t2_initial_ingestion_claim",
        call_id=record.call_identity.call_id,
        call_identity_sha256=record.call_identity_sha256,
        dispatch_intent_sha256=dispatch_intent_sha256,
        run_id=run_id,
        run_artifacts_sha256=run_artifacts_sha256,
    )
    encoded = _model_bytes(claim)
    claim_path = _canonical_call_path(INGESTION_CLAIM_FILENAME)
    try:
        published = publish_bytes_without_overwrite(claim_path, encoded)
    except ValueError:
        try:
            existing = _read_bounded_regular_file(
                claim_path,
                max_bytes=MAX_JSON_BYTES,
                label=INGESTION_CLAIM_FILENAME,
                required_mode=0o600,
            )
        except RewardSearchError as exc:
            raise T2ProtocolError("existing canonical ingestion claim is unreadable") from exc
        return hashlib.sha256(existing).hexdigest(), True
    return published.sha256, False


def ingest_initial_t2_sol_run(
    record_bytes: bytes,
    run_directory: Path,
) -> T2IngestionOutcome:
    """Retain one run and require the canonical identity, intent, and ingestion claim."""

    if type(record_bytes) is not bytes:
        raise T2ProtocolError("record_bytes must be exact builtin bytes")
    run_directory = _require_native_path(run_directory, label="run_directory")
    record, prompt, canonical_record = _validated_record(record_bytes)
    records_directory = _canonical_call_path("records")
    run_dir, output, run_id = _validate_run_paths(run_directory, records_directory)

    bindings: list[T2RunArtifactBinding] = []
    retained: list[PublishedArtifact] = []
    raw: dict[str, bytes] = {}
    for name in RUN_ARTIFACT_NAMES:
        binding, artifact, encoded = _retain_run_file(
            run_dir=run_dir,
            output=output,
            run_id=run_id,
            name=name,
            max_bytes=_RUN_FILE_LIMITS[name],
        )
        bindings.append(binding)
        if artifact is not None:
            assert encoded is not None
            retained.append(artifact)
            raw[name] = encoded

    proposal: T2ParameterProposal | None = None
    recipe_bytes: bytes | None = None
    result_thread_id: str | None = None
    envelope_verified = False
    dispatch_intent_consistency, dispatch_intent_sha256, rejection_reason = (
        _inspect_dispatch_intent(record, prompt, canonical_record)
    )
    ingestion_claim_sha256: str | None = None
    replay_refusal = False
    if dispatch_intent_consistency == "verified":
        assert dispatch_intent_sha256 is not None
        ingestion_claim_sha256, replay_refusal = _claim_ingestion(
            record,
            dispatch_intent_sha256=dispatch_intent_sha256,
            run_id=run_id,
            bindings=bindings,
        )
        if replay_refusal:
            rejection_reason = (
                "canonical call identity already has an ingestion claim; "
                "second run or provider replay refused"
            )
    failed_reads = [
        f"{binding.name}:{binding.state}" for binding in bindings if binding.state != "retained"
    ]
    if rejection_reason is None and failed_reads:
        rejection_reason = "run artifact read rejection: " + ", ".join(failed_reads)
    elif rejection_reason is None:
        try:
            result_thread_id = _validate_local_envelope(record, prompt, raw)
            envelope_verified = True
            proposal, recipe_bytes = _validate_proposal(record, raw["final.txt"])
        except T2ProtocolError as exc:
            rejection_reason = str(exc)[:4_096]

    proposal_artifact: PublishedArtifact | None = None
    recipe_artifact: PublishedArtifact | None = None
    if proposal is not None and recipe_bytes is not None:
        proposal_sha = _model_sha256(proposal)
        recipe_sha = hashlib.sha256(recipe_bytes).hexdigest()
        proposal_artifact = _publish_or_reuse_exact(
            output / f"t2-parameter-proposal-{run_id}-{proposal_sha}.json",
            _model_bytes(proposal),
        )
        recipe_artifact = _publish_or_reuse_exact(
            output / f"t2-recipe-{run_id}-{recipe_sha}.json",
            recipe_bytes,
        )

    receipt = _receipt(
        record=record,
        prompt=prompt,
        canonical_record=canonical_record,
        run_id=run_id,
        bindings=bindings,
        proposal=proposal,
        recipe_bytes=recipe_bytes,
        result_thread_id=result_thread_id,
        envelope_verified=envelope_verified,
        rejection_reason=rejection_reason,
        dispatch_intent_consistency=dispatch_intent_consistency,
        dispatch_intent_sha256=dispatch_intent_sha256,
        ingestion_claim_sha256=ingestion_claim_sha256,
        replay_refusal=replay_refusal,
    )
    receipt_sha = _model_sha256(receipt)
    receipt_artifact = _publish_or_reuse_exact(
        output / f"t2-model-call-receipt-{run_id}-{receipt_sha}.json",
        _model_bytes(receipt),
    )
    accepted = proposal is not None and recipe_bytes is not None and rejection_reason is None
    return T2IngestionOutcome(
        accepted=accepted,
        retained_run_artifacts=tuple(retained),
        proposal=proposal_artifact,
        recipe=recipe_artifact,
        receipt=receipt_artifact,
        rejection_reason=rejection_reason,
    )


__all__ = [
    "ingest_initial_t2_sol_run",
    "prepare_initial_t2_packet",
    "publish_initial_t2_dispatch_intent",
    "publish_initial_t2_packet",
    "render_initial_t2_prompt",
]
