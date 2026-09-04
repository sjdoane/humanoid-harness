"""Fail-closed admission for the frozen TQC v2 design review."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Sequence
from dataclasses import InitVar, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .fixed_reference import ExperimentContractError, read_bounded_json_artifact
from .tqc_calibration_contract import canonical_json
from .tqc_development_contract_v2 import (
    AUDIT_FILE_SHA256,
    AUDIT_SEMANTIC_SHA256,
    DESIGN_FILE_SHA256,
    DESIGN_SEMANTIC_SHA256,
    TRAINING_PROJECTION_SHA256,
    LoadedTQCDevelopmentDesignV2,
)

REVIEW_RECEIPT_ID = "tqc_dev_1m_v2_independent_design_review/v1"
REVIEWER_IDENTITY = "codex_read_only_agent:/root/tier_d_data_plan/v2_draft_audit"
REVIEW_COMPLETED_AT_UTC = "2026-09-04T02:59:58Z"
REVIEWED_GIT_COMMIT = "6fbe44c52ca0918a20fb68626d8e1fa11d67e214"
V2_CONTRACT_SOURCE_SHA256 = "85e9c0a395818d7296a96654427e7cd2177287510a16a1bde35345f01629f07b"
V2_CONTRACT_TEST_FILE_SHA256 = "676022c2d05d6e23d9ae8a52818c7140416eb93b3ee4c840548e41397415b0a9"
V2_PROTOCOL_DOC_SHA256 = "10441143f6f61bb4a4499303c9503496939f220eb61e9e6188f3ebc40f87e0fd"
V2_CLAIM_CEILING = "one_checkpoint_one_training_seed_twenty_fixed_reset_development_behavior"
IDENTITY_CHECK_CONTENT_SHA256 = "059fb42df32230394ac9581e59301e9d69823da63cbb940ecdb3acbeb41e1f49"
FOCUSED_CHECK_CONTENT_SHA256 = "7efb6b910e630c0820865b258fe28c362285ad65e6b5d36bd272280c18b2bb39"
FULL_CHECK_CONTENT_SHA256 = "fbd14475e841241740ba7f9694487d588d3da4eec38fbb1c59174dc1c738ab0a"
REVIEW_RECEIPT_CONTENT_SHA256 = "06989bcf424a6afd97bcd7fc41a230754df97313d5f52b0014ce90ba1a2549cf"
MAX_REVIEW_RECEIPT_BYTES = 64 * 1024
MAX_CHECK_RECEIPT_BYTES = 64 * 1024
MAX_REVIEWED_SOURCE_BYTES = 2 * 1024 * 1024

_AUTHORITY_ISSUER = object()
_RFC3339_UTC_SECONDS = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_exact(value: object, expected: object, *, field: str) -> None:
    if type(value) is not type(expected):
        raise ExperimentContractError(f"{field} has the wrong JSON type")
    if isinstance(expected, dict):
        if set(value) != set(expected):  # type: ignore[arg-type]
            raise ExperimentContractError(f"{field} keys differ")
        for key, child in expected.items():
            _require_exact(value[key], child, field=f"{field}.{key}")  # type: ignore[index]
        return
    if isinstance(expected, list):
        if len(value) != len(expected):  # type: ignore[arg-type]
            raise ExperimentContractError(f"{field} length differs")
        for index, child in enumerate(expected):
            _require_exact(value[index], child, field=f"{field}[{index}]")  # type: ignore[index]
        return
    if value != expected:
        raise ExperimentContractError(f"{field} differs")


def _read_canonical_receipt(
    path: Path,
    *,
    maximum_bytes: int,
    artifact: str,
) -> tuple[bytes, dict[str, Any]]:
    loaded = read_bounded_json_artifact(
        Path(path),
        maximum_bytes=maximum_bytes,
        artifact=artifact,
    )
    if loaded.encoded_bytes != canonical_json(loaded.value):
        raise ExperimentContractError(f"{artifact} must use canonical JSON UTF-8 bytes")
    return loaded.encoded_bytes, loaded.value


def _regular_file_sha256(path: Path, *, artifact: str) -> str:
    resolved = Path(path)
    descriptor: int | None = None
    try:
        if stat.S_ISLNK(resolved.lstat().st_mode):
            raise ExperimentContractError(f"{artifact} must not be a symlink")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(resolved, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ExperimentContractError(f"{artifact} must be a regular file")
        if before.st_size > MAX_REVIEWED_SOURCE_BYTES:
            raise ExperimentContractError(f"{artifact} exceeds the bounded size limit")
        digest = hashlib.sha256()
        observed_bytes = 0
        while observed_bytes <= MAX_REVIEWED_SOURCE_BYTES:
            chunk = os.read(
                descriptor, min(64 * 1024, MAX_REVIEWED_SOURCE_BYTES + 1 - observed_bytes)
            )
            if not chunk:
                break
            digest.update(chunk)
            observed_bytes += len(chunk)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if (
            observed_bytes > MAX_REVIEWED_SOURCE_BYTES
            or observed_bytes != before.st_size
            or before_identity != after_identity
        ):
            raise ExperimentContractError(f"{artifact} changed or exceeded its bound")
        return digest.hexdigest()
    except ExperimentContractError:
        raise
    except OSError as exc:
        raise ExperimentContractError(f"cannot read {artifact}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _identity_check() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "check_id": "tqc_dev_1m_v2_reviewed_identity/v1",
        "result": "PASS",
        "reviewed_git_commit": REVIEWED_GIT_COMMIT,
        "reviewed_design_file_sha256": DESIGN_FILE_SHA256,
        "reviewed_design_semantic_sha256": DESIGN_SEMANTIC_SHA256,
        "reviewed_e0_audit_file_sha256": AUDIT_FILE_SHA256,
        "reviewed_e0_audit_semantic_sha256": AUDIT_SEMANTIC_SHA256,
        "reviewed_training_projection_sha256": TRAINING_PROJECTION_SHA256,
        "reviewed_v2_contract_source_sha256": V2_CONTRACT_SOURCE_SHA256,
        "reviewed_v2_contract_test_file_sha256": V2_CONTRACT_TEST_FILE_SHA256,
        "reviewed_protocol_doc_sha256": V2_PROTOCOL_DOC_SHA256,
        "claim_ceiling": V2_CLAIM_CEILING,
        "authorizes_training": False,
        "behavioral_evidence": False,
    }


def _focused_check() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "check_id": "tqc_dev_1m_v2_focused_contract_tests/v1",
        "result": "PASS",
        "command": ".venv/bin/pytest -q tests/experiments/test_tqc_development_contract_v2.py",
        "observed_passed": 55,
        "observed_failed": 0,
        "scope": "frozen_v2_contract_slice_only",
        "covers_later_implementation": False,
        "authorizes_training": False,
        "behavioral_evidence": False,
    }


def _full_check() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "check_id": "tqc_dev_1m_v2_repository_tests/v1",
        "result": "PASS",
        "command": ".venv/bin/pytest -q",
        "observed_passed": 903,
        "observed_failed": 0,
        "scope": "repository_suite_at_frozen_v2_contract_review_boundary",
        "covers_later_implementation": False,
        "authorizes_training": False,
        "behavioral_evidence": False,
    }


def _expected_review_receipt() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "receipt_id": REVIEW_RECEIPT_ID,
        "reviewer_role": "independent_read_only",
        "reviewer_identity": REVIEWER_IDENTITY,
        "review_completed_at_utc": REVIEW_COMPLETED_AT_UTC,
        "verdict": "GO",
        "reviewed_design_file_sha256": DESIGN_FILE_SHA256,
        "reviewed_design_semantic_sha256": DESIGN_SEMANTIC_SHA256,
        "reviewed_e0_audit_file_sha256": AUDIT_FILE_SHA256,
        "reviewed_e0_audit_semantic_sha256": AUDIT_SEMANTIC_SHA256,
        "reviewed_v2_contract_source_sha256": V2_CONTRACT_SOURCE_SHA256,
        "reviewed_v2_contract_test_file_sha256": V2_CONTRACT_TEST_FILE_SHA256,
        "finding_count": 0,
        "resolved_finding_count": 0,
        "unresolved_p0_count": 0,
        "unresolved_p1_count": 0,
        "unresolved_p2_count": 0,
        "observed_check_receipts": [
            IDENTITY_CHECK_CONTENT_SHA256,
            FOCUSED_CHECK_CONTENT_SHA256,
            FULL_CHECK_CONTENT_SHA256,
        ],
    }


@dataclass(frozen=True, slots=True)
class ValidatedIndependentReviewReceiptV2:
    """Exact review GO; never training authorization or behavior evidence."""

    encoded_bytes: bytes
    content_sha256: str
    schema_version: int
    receipt_id: str
    reviewer_role: str
    reviewer_identity: str
    review_completed_at_utc: str
    verdict: str
    reviewed_design_file_sha256: str
    reviewed_design_semantic_sha256: str
    reviewed_e0_audit_file_sha256: str
    reviewed_e0_audit_semantic_sha256: str
    reviewed_v2_contract_source_sha256: str
    reviewed_v2_contract_test_file_sha256: str
    finding_count: int
    resolved_finding_count: int
    unresolved_p0_count: int
    unresolved_p1_count: int
    unresolved_p2_count: int
    observed_check_receipts: tuple[str, ...]
    reviewed_training_projection_sha256: str
    reviewed_protocol_doc_sha256: str
    claim_ceiling: str
    contract_review_go: bool
    authorizes_training: bool
    behavioral_evidence: bool
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _AUTHORITY_ISSUER:
            raise ExperimentContractError(
                "validated v2 review receipts may only be issued by strict validation"
            )
        receipt = json.loads(self.encoded_bytes)
        _require_exact(receipt, _expected_review_receipt(), field="validated review receipt")
        _require_exact(_sha256(self.encoded_bytes), self.content_sha256, field="receipt binding")
        _require_exact(
            self.content_sha256,
            REVIEW_RECEIPT_CONTENT_SHA256,
            field="compiled receipt SHA-256",
        )
        for field in _expected_review_receipt():
            expected = _expected_review_receipt()[field]
            observed: object = (
                self.observed_check_receipts
                if field == "observed_check_receipts"
                else getattr(self, field)
            )
            if field == "observed_check_receipts":
                expected = tuple(expected)
            _require_exact(observed, expected, field=f"authority.{field}")
        _require_exact(
            self.reviewed_training_projection_sha256,
            TRAINING_PROJECTION_SHA256,
            field="authority training projection",
        )
        _require_exact(
            self.reviewed_protocol_doc_sha256,
            V2_PROTOCOL_DOC_SHA256,
            field="authority protocol doc",
        )
        _require_exact(self.claim_ceiling, V2_CLAIM_CEILING, field="authority claim ceiling")
        _require_exact(self.contract_review_go, True, field="contract review GO")
        _require_exact(self.authorizes_training, False, field="training authorization")
        _require_exact(self.behavioral_evidence, False, field="behavior evidence")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.encoded_bytes)


def validate_independent_review_receipt_v2(
    loaded_design: LoadedTQCDevelopmentDesignV2,
    *,
    review_receipt_path: Path,
    observed_check_receipt_paths: Sequence[Path],
    contract_source_path: Path,
    contract_test_path: Path,
    protocol_doc_path: Path,
) -> ValidatedIndependentReviewReceiptV2:
    """Validate exact reviewed bytes and issue a non-training review capability."""

    if type(loaded_design) is not LoadedTQCDevelopmentDesignV2:
        raise ExperimentContractError("v2 review validation requires a strict-loaded design")
    design = loaded_design.to_dict()
    _require_exact(design["claim_ceiling"], V2_CLAIM_CEILING, field="design claim ceiling")
    contract = design["execution_manifest"]["independent_review_receipt_contract"]

    receipt_bytes, receipt = _read_canonical_receipt(
        review_receipt_path,
        maximum_bytes=MAX_REVIEW_RECEIPT_BYTES,
        artifact="v2 independent review receipt",
    )
    _require_exact(
        _sha256(receipt_bytes), REVIEW_RECEIPT_CONTENT_SHA256, field="review receipt SHA-256"
    )
    _require_exact(set(receipt), set(contract["exact_keys"]), field="review receipt exact keys")
    _require_exact(receipt, _expected_review_receipt(), field="review receipt")
    if not _RFC3339_UTC_SECONDS.fullmatch(receipt["review_completed_at_utc"]):
        raise ExperimentContractError("review completion time is not RFC3339 UTC seconds")
    try:
        datetime.strptime(receipt["review_completed_at_utc"], "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ExperimentContractError("review completion time is invalid") from exc

    if len(observed_check_receipt_paths) != 3:
        raise ExperimentContractError("exactly three ordered review check receipts are required")
    check_specs = (
        (IDENTITY_CHECK_CONTENT_SHA256, _identity_check(), "review identity check"),
        (FOCUSED_CHECK_CONTENT_SHA256, _focused_check(), "review focused-test check"),
        (FULL_CHECK_CONTENT_SHA256, _full_check(), "review full-test check"),
    )
    observed_hashes: list[str] = []
    for path, (expected_sha256, expected_value, artifact) in zip(
        observed_check_receipt_paths,
        check_specs,
        strict=True,
    ):
        check_bytes, check = _read_canonical_receipt(
            Path(path),
            maximum_bytes=MAX_CHECK_RECEIPT_BYTES,
            artifact=artifact,
        )
        check_sha256 = _sha256(check_bytes)
        _require_exact(check_sha256, expected_sha256, field=f"{artifact} SHA-256")
        _require_exact(check, expected_value, field=artifact)
        observed_hashes.append(check_sha256)
    _require_exact(
        receipt["observed_check_receipts"],
        observed_hashes,
        field="ordered observed check receipts",
    )

    source_hashes = (
        (
            _regular_file_sha256(contract_source_path, artifact="v2 contract source"),
            V2_CONTRACT_SOURCE_SHA256,
        ),
        (
            _regular_file_sha256(contract_test_path, artifact="v2 contract test"),
            V2_CONTRACT_TEST_FILE_SHA256,
        ),
        (
            _regular_file_sha256(protocol_doc_path, artifact="v2 protocol doc"),
            V2_PROTOCOL_DOC_SHA256,
        ),
    )
    for observed, expected in source_hashes:
        _require_exact(observed, expected, field="reviewed source SHA-256")

    return ValidatedIndependentReviewReceiptV2(
        encoded_bytes=receipt_bytes,
        content_sha256=REVIEW_RECEIPT_CONTENT_SHA256,
        schema_version=receipt["schema_version"],
        receipt_id=receipt["receipt_id"],
        reviewer_role=receipt["reviewer_role"],
        reviewer_identity=receipt["reviewer_identity"],
        review_completed_at_utc=receipt["review_completed_at_utc"],
        verdict=receipt["verdict"],
        reviewed_design_file_sha256=receipt["reviewed_design_file_sha256"],
        reviewed_design_semantic_sha256=receipt["reviewed_design_semantic_sha256"],
        reviewed_e0_audit_file_sha256=receipt["reviewed_e0_audit_file_sha256"],
        reviewed_e0_audit_semantic_sha256=receipt["reviewed_e0_audit_semantic_sha256"],
        reviewed_v2_contract_source_sha256=receipt["reviewed_v2_contract_source_sha256"],
        reviewed_v2_contract_test_file_sha256=receipt["reviewed_v2_contract_test_file_sha256"],
        finding_count=receipt["finding_count"],
        resolved_finding_count=receipt["resolved_finding_count"],
        unresolved_p0_count=receipt["unresolved_p0_count"],
        unresolved_p1_count=receipt["unresolved_p1_count"],
        unresolved_p2_count=receipt["unresolved_p2_count"],
        observed_check_receipts=tuple(receipt["observed_check_receipts"]),
        reviewed_training_projection_sha256=TRAINING_PROJECTION_SHA256,
        reviewed_protocol_doc_sha256=V2_PROTOCOL_DOC_SHA256,
        claim_ceiling=V2_CLAIM_CEILING,
        contract_review_go=True,
        authorizes_training=False,
        behavioral_evidence=False,
        _issuer=_AUTHORITY_ISSUER,
    )


__all__ = [
    "FOCUSED_CHECK_CONTENT_SHA256",
    "FULL_CHECK_CONTENT_SHA256",
    "IDENTITY_CHECK_CONTENT_SHA256",
    "REVIEW_RECEIPT_CONTENT_SHA256",
    "ValidatedIndependentReviewReceiptV2",
    "validate_independent_review_receipt_v2",
]
