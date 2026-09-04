from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.sources.external_payload_policy import (
    DERIVED_ACTOR_NPZ_SHA256,
    DERIVED_ACTOR_STATE_SHA256,
    FORBIDDEN_EXTERNAL_PAYLOAD_SHA256,
    KNOWN_EXTERNAL_PAYLOAD_SHA256,
    TrackedBlob,
    forbidden_hashes_from_receipt,
    tracked_blobs_from_git_index,
    validate_tracked_blobs,
)

REPOSITORY = Path(__file__).resolve().parents[2]
IMPORT_RECEIPT = REPOSITORY / "artifacts/bootstrap_tqc_humanoid/external_actor_import_v1.json"
DERIVED_ACTOR = (
    REPOSITORY / "artifacts/bootstrap_tqc_humanoid/farama_minari_humanoid_v5_tqc_actor_v1.npz"
)
EXTERNAL_ROOT = REPOSITORY / "artifacts/external/farama-minari-humanoid-v5-tqc-expert"


def test_no_external_payload_bytes_are_present_in_the_git_index() -> None:
    validate_tracked_blobs(
        tracked_blobs_from_git_index(REPOSITORY),
        forbidden_sha256=FORBIDDEN_EXTERNAL_PAYLOAD_SHA256,
    )


def test_forbidden_set_binds_source_npz_and_actor_state_receipt_hashes() -> None:
    assert DERIVED_ACTOR_NPZ_SHA256 in FORBIDDEN_EXTERNAL_PAYLOAD_SHA256
    assert DERIVED_ACTOR_STATE_SHA256 in FORBIDDEN_EXTERNAL_PAYLOAD_SHA256
    assert KNOWN_EXTERNAL_PAYLOAD_SHA256 < FORBIDDEN_EXTERNAL_PAYLOAD_SHA256
    if IMPORT_RECEIPT.is_file():
        assert forbidden_hashes_from_receipt(IMPORT_RECEIPT) == (FORBIDDEN_EXTERNAL_PAYLOAD_SHA256)


def test_byte_keyed_policy_rejects_renamed_payload_and_hdf5_shard() -> None:
    payload = b"renamed external payload"
    with pytest.raises(ExperimentContractError, match="forbidden external payload bytes"):
        validate_tracked_blobs(
            [TrackedBlob(path="innocent-name.txt", content=payload)],
            forbidden_sha256=frozenset({hashlib.sha256(payload).hexdigest()}),
        )

    hdf5 = b"prefix" + b"\0" * (512 - len(b"prefix")) + b"\x89HDF\r\n\x1a\n"
    with pytest.raises(ExperimentContractError, match="dataset payload signature"):
        validate_tracked_blobs(
            [TrackedBlob(path="renamed.bin", content=hdf5)],
            forbidden_sha256=frozenset(),
        )


def test_source_controller_metadata_size_ceiling_is_enforced() -> None:
    with pytest.raises(ExperimentContractError, match="exceeds 1 MiB"):
        validate_tracked_blobs(
            [
                TrackedBlob(
                    path="research/source_controllers/oversized-metadata.txt",
                    content=b"x" * (1024 * 1024 + 1),
                )
            ],
            forbidden_sha256=frozenset(),
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "humanoid-v5-TQC-expert.zip",
        "humanoid-v5-TQC-expert/policy.pth",
        "humanoid-v5-TQC-expert/pytorch_variables.pth",
        "humanoid-v5-TQC-expert/data",
        "config.json",
    ],
)
def test_installed_source_payloads_are_rejected_under_renamed_paths(
    relative_path: str,
) -> None:
    path = EXTERNAL_ROOT / relative_path
    if not path.is_file():
        pytest.skip("ignored external payload is not installed")
    with pytest.raises(ExperimentContractError, match="forbidden external payload bytes"):
        validate_tracked_blobs(
            [TrackedBlob(path="renamed/source.dat", content=path.read_bytes())],
            forbidden_sha256=FORBIDDEN_EXTERNAL_PAYLOAD_SHA256,
        )


def test_installed_derived_actor_is_rejected_under_a_renamed_path() -> None:
    if not DERIVED_ACTOR.is_file():
        pytest.skip("ignored derived actor is not installed")
    assert hashlib.sha256(DERIVED_ACTOR.read_bytes()).hexdigest() == DERIVED_ACTOR_NPZ_SHA256
    with pytest.raises(ExperimentContractError, match="forbidden external payload bytes"):
        validate_tracked_blobs(
            [TrackedBlob(path="controller.bin", content=DERIVED_ACTOR.read_bytes())],
            forbidden_sha256=FORBIDDEN_EXTERNAL_PAYLOAD_SHA256,
        )
