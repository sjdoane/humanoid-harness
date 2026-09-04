from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from oracle_composition.experiments import tqc_development_review_v2 as review
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_calibration_contract import canonical_json
from oracle_composition.experiments.tqc_development_contract_v2 import (
    load_tqc_development_design_v2,
)
from oracle_composition.experiments.tqc_development_review_v2 import (
    ValidatedIndependentReviewReceiptV2,
    validate_independent_review_receipt_v2,
)

ROOT = Path(__file__).parents[2]
CONFIG_ROOT = ROOT / "experiments/bootstrap_tqc_humanoid/configs"
REVIEW_ROOT = ROOT / "experiments/bootstrap_tqc_humanoid/reviews"
DESIGN_PATH = CONFIG_ROOT / "tqc_base_controller_dev_1m_v2.study.json"
AUDIT_PATH = CONFIG_ROOT / "tqc_e0_reuse_dev_1m_v2.audit.json"
CONTRACT_SOURCE_PATH = ROOT / "src/oracle_composition/experiments/tqc_development_contract_v2.py"
CONTRACT_TEST_PATH = ROOT / "tests/experiments/test_tqc_development_contract_v2.py"
PROTOCOL_DOC_PATH = ROOT / "experiments/bootstrap_tqc_humanoid/DEV_1M_BASE_CONTROLLER_V2.md"
RECEIPT_PATH = REVIEW_ROOT / "tqc_dev_1m_v2_independent_design_review.review.json"
CHECK_PATHS = (
    REVIEW_ROOT / "tqc_dev_1m_v2_reviewed_identity.check.json",
    REVIEW_ROOT / "tqc_dev_1m_v2_focused_contract_tests.check.json",
    REVIEW_ROOT / "tqc_dev_1m_v2_repository_tests.check.json",
)


def _loaded_design():
    return load_tqc_development_design_v2(DESIGN_PATH, AUDIT_PATH)


def _validate(
    *,
    receipt_path: Path = RECEIPT_PATH,
    check_paths: tuple[Path, ...] = CHECK_PATHS,
    contract_source_path: Path = CONTRACT_SOURCE_PATH,
    contract_test_path: Path = CONTRACT_TEST_PATH,
    protocol_doc_path: Path = PROTOCOL_DOC_PATH,
) -> ValidatedIndependentReviewReceiptV2:
    return validate_independent_review_receipt_v2(
        _loaded_design(),
        review_receipt_path=receipt_path,
        observed_check_receipt_paths=check_paths,
        contract_source_path=contract_source_path,
        contract_test_path=contract_test_path,
        protocol_doc_path=protocol_doc_path,
    )


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_bytes())


def _write(path: Path, value: object) -> None:
    path.write_bytes(canonical_json(value))


def test_exact_independent_review_issues_non_training_authority() -> None:
    validated = _validate()

    assert validated.content_sha256 == hashlib.sha256(RECEIPT_PATH.read_bytes()).hexdigest()
    assert validated.to_dict() == _read(RECEIPT_PATH)
    assert validated.schema_version == 1
    assert validated.receipt_id == "tqc_dev_1m_v2_independent_design_review/v1"
    assert validated.reviewer_role == "independent_read_only"
    assert validated.verdict == "GO"
    assert validated.finding_count == 0
    assert validated.resolved_finding_count == 0
    assert validated.unresolved_p0_count == 0
    assert validated.unresolved_p1_count == 0
    assert validated.unresolved_p2_count == 0
    assert len(validated.observed_check_receipts) == 3
    assert validated.reviewed_training_projection_sha256 == (
        "1120f73fc458989a14a8f72f487fe4a9fa28922e1b55a9954744e271fb11425e"
    )
    assert validated.reviewed_protocol_doc_sha256 == (
        "10441143f6f61bb4a4499303c9503496939f220eb61e9e6188f3ebc40f87e0fd"
    )
    assert validated.contract_review_go is True
    assert validated.authorizes_training is False
    assert validated.behavioral_evidence is False


def test_review_authority_cannot_be_publicly_forged() -> None:
    fields = dataclasses.asdict(_validate())

    with pytest.raises(ExperimentContractError, match="strict validation"):
        ValidatedIndependentReviewReceiptV2(**fields)


def test_changed_verdict_fails_even_if_file_hash_is_rebound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _read(RECEIPT_PATH)
    value["verdict"] = "NO-GO"
    changed = tmp_path / "changed.review.json"
    _write(changed, value)
    monkeypatch.setattr(
        review,
        "REVIEW_RECEIPT_CONTENT_SHA256",
        hashlib.sha256(changed.read_bytes()).hexdigest(),
    )

    with pytest.raises(ExperimentContractError, match=r"review receipt\.verdict differs"):
        _validate(receipt_path=changed)


def test_receipt_requires_exact_canonical_json_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed = tmp_path / "noncanonical.review.json"
    changed.write_bytes(RECEIPT_PATH.read_bytes() + b"\n")
    monkeypatch.setattr(
        review,
        "REVIEW_RECEIPT_CONTENT_SHA256",
        hashlib.sha256(changed.read_bytes()).hexdigest(),
    )

    with pytest.raises(ExperimentContractError, match="canonical JSON UTF-8"):
        _validate(receipt_path=changed)


def test_changed_observed_test_count_fails_after_all_hashes_are_rebound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    changed_checks = []
    for path in CHECK_PATHS:
        copied = tmp_path / path.name
        copied.write_bytes(path.read_bytes())
        changed_checks.append(copied)
    focused = _read(changed_checks[1])
    focused["observed_passed"] = 54
    _write(changed_checks[1], focused)
    focused_sha256 = hashlib.sha256(changed_checks[1].read_bytes()).hexdigest()

    receipt = _read(RECEIPT_PATH)
    receipt["observed_check_receipts"][1] = focused_sha256  # type: ignore[index]
    changed_receipt = tmp_path / RECEIPT_PATH.name
    _write(changed_receipt, receipt)
    monkeypatch.setattr(review, "FOCUSED_CHECK_CONTENT_SHA256", focused_sha256)
    monkeypatch.setattr(
        review,
        "REVIEW_RECEIPT_CONTENT_SHA256",
        hashlib.sha256(changed_receipt.read_bytes()).hexdigest(),
    )

    with pytest.raises(ExperimentContractError, match="observed_passed differs"):
        _validate(receipt_path=changed_receipt, check_paths=tuple(changed_checks))


def test_check_receipts_are_ordered_and_content_bound() -> None:
    with pytest.raises(ExperimentContractError, match="identity check SHA-256"):
        _validate(check_paths=(CHECK_PATHS[1], CHECK_PATHS[0], CHECK_PATHS[2]))


@pytest.mark.parametrize(
    ("source", "argument", "error"),
    [
        (CONTRACT_SOURCE_PATH, "contract_source_path", "reviewed source SHA-256 differs"),
        (CONTRACT_TEST_PATH, "contract_test_path", "reviewed source SHA-256 differs"),
        (PROTOCOL_DOC_PATH, "protocol_doc_path", "reviewed source SHA-256 differs"),
    ],
)
def test_changed_reviewed_source_fails_closed(
    tmp_path: Path,
    source: Path,
    argument: str,
    error: str,
) -> None:
    changed = tmp_path / source.name
    changed.write_bytes(source.read_bytes() + b"\n")
    arguments = {argument: changed}

    with pytest.raises(ExperimentContractError, match=error):
        _validate(**arguments)  # type: ignore[arg-type]


def test_reviewed_source_symlink_is_rejected(tmp_path: Path) -> None:
    linked = tmp_path / "contract.py"
    linked.symlink_to(CONTRACT_SOURCE_PATH)

    with pytest.raises(ExperimentContractError, match="must not be a symlink"):
        _validate(contract_source_path=linked)
