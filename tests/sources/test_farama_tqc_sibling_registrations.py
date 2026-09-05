from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.sources.farama_tqc_sibling_registrations import (
    SIBLING_REGISTRATION_SPECS,
    load_sibling_registration_receipt,
    validate_sibling_registration_receipt,
)

REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("variant", ["medium", "simple"])
def test_sibling_registration_is_exact_payload_free_and_api_labeled(variant: str) -> None:
    spec = SIBLING_REGISTRATION_SPECS[variant]
    receipt_path = REPOSITORY / spec.registration_receipt_logical_path
    receipt = load_sibling_registration_receipt(receipt_path, variant=variant)
    api_path = REPOSITORY / spec.captured_api_path
    if not api_path.is_file():
        pytest.skip("ignored captured sibling API record is not installed")
    api_bytes = api_path.read_bytes()
    api = json.loads(api_bytes)

    assert len(receipt_path.read_bytes()) == spec.registration_receipt_byte_count
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest() == (
        spec.registration_receipt_sha256
    )
    assert len(api_bytes) == spec.captured_api_byte_count
    assert hashlib.sha256(api_bytes).hexdigest() == spec.captured_api_sha256
    assert receipt["source"]["repository_commit"] == api["sha"]
    assert receipt["source"]["model_card_metric"]["verified"] is False
    assert receipt["rights"]["payload_bytes_enter_git"] is False

    remote = {item["rfilename"]: item for item in receipt["remote_inventory"]}
    local = {item["rfilename"]: item for item in receipt["local_inventory"]}
    assert set(remote) == {item["rfilename"] for item in api["siblings"]}
    assert set(local) == set(remote)
    for sibling in api["siblings"]:
        item = remote[sibling["rfilename"]]
        assert item["size"] == sibling["size"]
        if "lfs" in sibling:
            assert item["digest"] == {
                "algorithm": "lfs.sha256",
                "value": sibling["lfs"]["sha256"],
                "api_field": "lfs.sha256",
            }
        else:
            assert item["digest"] == {
                "algorithm": "git-blob-sha1",
                "value": sibling["blobId"],
                "api_field": "blobId",
            }
    assert {name for name, item in local.items() if item["status"] == "absent"} == (
        spec.absent_filenames
    )


@pytest.mark.parametrize("variant", ["medium", "simple"])
@pytest.mark.parametrize("mutation", ["digest_label", "missing_absence", "verified_metric"])
def test_sibling_registration_rejects_any_semantic_drift(
    variant: str,
    mutation: str,
) -> None:
    spec = SIBLING_REGISTRATION_SPECS[variant]
    receipt = load_sibling_registration_receipt(
        REPOSITORY / spec.registration_receipt_logical_path,
        variant=variant,
    )
    candidate = copy.deepcopy(receipt)
    if mutation == "digest_label":
        candidate["remote_inventory"][0]["digest"]["algorithm"] = "sha256"
    elif mutation == "missing_absence":
        candidate["local_inventory"].pop(0)
    else:
        candidate["source"]["model_card_metric"]["verified"] = True

    with pytest.raises(ExperimentContractError, match="Farama TQC sibling"):
        validate_sibling_registration_receipt(candidate, variant=variant)
