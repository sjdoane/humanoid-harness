from __future__ import annotations

import json
from pathlib import Path

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes, sha256_file
from oracle_composition.phase_b.receipts import ADMITTED_TRAINING_BLOCKS

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
