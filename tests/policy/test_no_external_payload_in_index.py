from __future__ import annotations

import hashlib
import io
import subprocess
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import numpy as np
import pytest

from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.sources import external_payload_policy as policy
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
IMPORT_RECEIPT = (
    REPOSITORY / "artifacts/bootstrap_tqc_humanoid/external_actor_import_expert_03a3_v1.json"
)
DERIVED_ACTOR = (
    REPOSITORY / "artifacts/bootstrap_tqc_humanoid/farama_minari_humanoid_v5_tqc_actor_v1.npz"
)
EXTERNAL_ROOT = REPOSITORY / "artifacts/external/farama-minari-humanoid-v5-tqc-expert"


def _reencoded_actor(*, mutate: bool = False) -> bytes:
    with np.load(DERIVED_ACTOR, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    if mutate:
        arrays["mu.bias"][0] += np.float32(0.125)
    stream = io.BytesIO()
    with ZipFile(stream, "w", compression=ZIP_STORED) as archive:
        for name, value in arrays.items():
            payload = io.BytesIO()
            np.save(payload, value, allow_pickle=False)
            member = ZipInfo(f"{name}.npy", date_time=(2026, 9, 4, 12, 34, 56))
            member.compress_type = ZIP_STORED
            archive.writestr(member, payload.getvalue())
    return stream.getvalue()


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


def test_semantic_policy_rejects_reencoded_actor_but_allows_different_values() -> None:
    if not DERIVED_ACTOR.is_file():
        pytest.skip("ignored derived actor is not installed")
    equivalent = _reencoded_actor()
    different_values = _reencoded_actor(mutate=True)
    assert hashlib.sha256(equivalent).hexdigest() != DERIVED_ACTOR_NPZ_SHA256

    with pytest.raises(ExperimentContractError, match="forbidden external actor state"):
        validate_tracked_blobs(
            [TrackedBlob(path="reencoded.npz", content=equivalent)],
            forbidden_sha256=FORBIDDEN_EXTERNAL_PAYLOAD_SHA256,
        )
    with pytest.raises(ExperimentContractError, match="forbidden external actor state"):
        validate_tracked_blobs(
            [TrackedBlob(path="prefixed.npz", content=b"launcher-prefix" + equivalent)],
            forbidden_sha256=FORBIDDEN_EXTERNAL_PAYLOAD_SHA256,
        )

    validate_tracked_blobs(
        [TrackedBlob(path="different-values.npz", content=different_values)],
        forbidden_sha256=FORBIDDEN_EXTERNAL_PAYLOAD_SHA256,
    )


def test_semantic_policy_rejects_malformed_and_oversized_zip_without_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(policy.np, "load", lambda *_args, **_kwargs: pytest.fail("NPY parsed"))
    with pytest.raises(ExperimentContractError, match="candidate ZIP is malformed"):
        validate_tracked_blobs(
            [TrackedBlob(path="malformed.npz", content=b"PK\x03\x04truncated")],
            forbidden_sha256=frozenset(),
        )

    oversized = b"PK\x03\x04" + b"x" * policy.MAX_SEMANTIC_NPZ_BYTES
    monkeypatch.setattr(policy, "ZipFile", lambda *_args, **_kwargs: pytest.fail("ZIP parsed"))
    with pytest.raises(ExperimentContractError, match="semantic inspection bound"):
        validate_tracked_blobs(
            [TrackedBlob(path="oversized.npz", content=oversized)],
            forbidden_sha256=frozenset(),
        )


def test_git_warning_only_stderr_passes_for_both_read_only_commands(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    object_id = "a" * 40
    responses = iter(
        (
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=f"100644 {object_id} 0\ttracked.txt\0".encode(),
                stderr=b"warning: sandbox metadata lookup was denied\n",
            ),
            subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout=f"{object_id} blob 3\n".encode() + b"abc\n",
                stderr=b"warning: sandbox metadata lookup was denied\n",
            ),
        )
    )
    monkeypatch.setattr(policy.subprocess, "run", lambda *_args, **_kwargs: next(responses))

    assert tracked_blobs_from_git_index(tmp_path) == [
        TrackedBlob(path="tracked.txt", content=b"abc")
    ]


@pytest.mark.parametrize(
    ("stdout", "error"),
    [
        (b"not-a-header\n", "malformed"),
        (f"{'a' * 40} tree 3\nabc\n".encode(), "non-blob"),
        (f"{'a' * 40} blob 4\nabc\n".encode(), "length differs"),
    ],
)
def test_git_cat_file_refuses_malformed_wrong_type_and_incomplete_output(
    stdout: bytes,
    error: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr=b"")
    monkeypatch.setattr(policy.subprocess, "run", lambda *_args, **_kwargs: completed)

    with pytest.raises(ExperimentContractError, match=error):
        policy._read_git_objects(tmp_path, ["a" * 40])


@pytest.mark.parametrize("command", ["ls-files", "cat-file"])
def test_git_nonzero_status_refuses(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    object_id = "a" * 40
    success = subprocess.CompletedProcess(
        args=[],
        returncode=0,
        stdout=f"100644 {object_id} 0\ttracked.txt\0".encode(),
        stderr=b"",
    )
    failure = subprocess.CompletedProcess(args=[], returncode=7, stdout=b"", stderr=b"warning\n")
    responses = iter((failure,) if command == "ls-files" else (success, failure))
    monkeypatch.setattr(policy.subprocess, "run", lambda *_args, **_kwargs: next(responses))

    with pytest.raises(ExperimentContractError, match=f"git {command}"):
        tracked_blobs_from_git_index(tmp_path)
