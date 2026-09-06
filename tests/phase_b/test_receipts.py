from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes, sha256_file
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.phase_b.receipts import (
    ADMITTED_TRAINING_BLOCKS,
    validate_e1_receipt_at_admission,
)

ROOT = Path(__file__).resolve().parents[2]


def test_real_e1_receipt_is_canonical_complete_and_interface_only() -> None:
    path = (
        ROOT / "experiments/003_composition_speed_profile/phase_b/receipts/"
        "e1_full_authority_warm_start_v1.json"
    )
    encoded = path.read_bytes()
    value = json.loads(encoded)
    assert encoded == canonical_json_bytes(value)
    assert value["passed"] is True
    assert value["evidence_class"] == "interface_check"
    assert value["training_steps"] == 0
    assert value["fixture_batch"]["synthetic_count"] == 4
    assert value["fixture_batch"]["real_count"] == 64
    assert value["fixture_batch"]["count"] == 68
    assert {item["block"] for item in value["fixture_batch"]["real_fixtures"]} == set(
        ADMITTED_TRAINING_BLOCKS
    )
    assert value["source_hashes"] == {
        "policy": sha256_file(ROOT / "src/oracle_composition/phase_b/policy.py"),
        "receipt_generator": sha256_file(ROOT / "src/oracle_composition/phase_b/receipts.py"),
    }
    assert value["e1_sampling_epsilon_sha256"]
    assert len(value["reload_parameter_checks"]) == 8
    assert all(item["bitwise_equal"] for item in value["reload_parameter_checks"].values())
    assert sha256_file(ROOT / value["export"]["path"]) == value["export"]["sha256"]
    assert value["likelihood_audit"]["maximum_absolute_recomputation_difference"] <= 1e-5
    assert all(item["bitwise_equal"] for item in value["identity_checks"].values())
    assert all(item["bitwise_equal"] for item in value["reload_identity_checks"].values())


def test_e1_admission_rejects_rehashed_stale_source_receipt(tmp_path: Path) -> None:
    source = (
        ROOT / "experiments/003_composition_speed_profile/phase_b/receipts/"
        "e1_full_authority_warm_start_v1.json"
    )
    value = json.loads(source.read_bytes())
    value["source_hashes"]["policy"] = "0" * 64
    encoded = canonical_json_bytes(value)
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(encoded)
    export = ROOT / value["export"]["path"]
    with pytest.raises(ExperimentContractError, match="live modules"):
        validate_e1_receipt_at_admission(
            repository_root=ROOT,
            receipt_path=receipt,
            expected_receipt_sha256=hashlib.sha256(encoded).hexdigest(),
            export_path=export,
            expected_export_sha256=value["export"]["sha256"],
        )
