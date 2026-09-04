from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.sources.farama_tqc_registration import (
    HF_API_BYTE_COUNT,
    HF_API_SHA256,
    expected_registration_receipt,
    load_registration_receipt,
    validate_registration_receipt,
)

REPOSITORY = Path(__file__).resolve().parents[2]
REGISTRATION = (
    REPOSITORY / "research/source_controllers/farama_minari_humanoid_v5_tqc_expert/RECEIPT.json"
)
CAPTURED_API = (
    REPOSITORY / "artifacts/external/farama-minari-humanoid-v5-tqc-expert/hf_api_model_info.json"
)


def test_committed_registration_is_exact_and_payload_free() -> None:
    receipt = load_registration_receipt(REGISTRATION)

    assert receipt == expected_registration_receipt()
    assert len(receipt["remote_inventory"]) == 14
    assert len(receipt["local_inventory"]) == 14
    assert receipt["rights"] == {
        "hugging_face_license": "unspecified",
        "permitted_project_use": "local development only",
        "technical_import_approval": "Samuel approved 2026-09-04",
        "technical_approval_is_redistribution_grant": False,
        "payload_bytes_enter_git": False,
    }


def test_remote_digest_labels_match_the_captured_api_shape() -> None:
    if not CAPTURED_API.is_file():
        pytest.skip("ignored captured API record is not installed")
    payload = CAPTURED_API.read_bytes()
    assert len(payload) == HF_API_BYTE_COUNT
    assert hashlib.sha256(payload).hexdigest() == HF_API_SHA256
    api = json.loads(payload)
    receipt = expected_registration_receipt()
    observed = {item["rfilename"]: item for item in receipt["remote_inventory"]}

    assert set(observed) == {item["rfilename"] for item in api["siblings"]}
    for sibling in api["siblings"]:
        item = observed[sibling["rfilename"]]
        assert item["size"] == sibling["size"]
        assert item["api_record"]["blobId"] == sibling["blobId"]
        if "lfs" in sibling:
            assert item["digest"] == {
                "algorithm": "lfs.sha256",
                "value": sibling["lfs"]["sha256"],
                "api_field": "lfs.sha256",
            }
            assert item["api_record"]["lfs"] == sibling["lfs"]
        else:
            assert item["digest"] == {
                "algorithm": "git-blob-sha1",
                "value": sibling["blobId"],
                "api_field": "blobId",
            }
            assert item["api_record"] == {"blobId": sibling["blobId"]}


@pytest.mark.parametrize("mutation", ["mislabel_sha1", "omit_absent", "flatten_lfs"])
def test_registration_rejects_digest_or_presence_drift(mutation: str) -> None:
    receipt = copy.deepcopy(expected_registration_receipt())
    if mutation == "mislabel_sha1":
        receipt["remote_inventory"][0]["digest"]["algorithm"] = "sha256"
    elif mutation == "omit_absent":
        receipt["local_inventory"].pop(0)
    else:
        lfs_item = receipt["remote_inventory"][3]
        lfs_item["api_record"].pop("lfs")

    with pytest.raises(ExperimentContractError, match="registration receipt differs"):
        validate_registration_receipt(receipt)
