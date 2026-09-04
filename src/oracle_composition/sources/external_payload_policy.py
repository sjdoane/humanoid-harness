"""Git-index policy for non-redistributable external controller payloads."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from oracle_composition.experiments.fixed_reference import ExperimentContractError

from .external_sb3_actor import validate_external_actor_import_receipt
from .farama_tqc_registration import LOCAL_FILES

MAX_SOURCE_CONTROLLER_BLOB_BYTES = 1024 * 1024
MAX_IMPORT_RECEIPT_BYTES = 1024 * 1024
SOURCE_CONTROLLER_PREFIX = PurePosixPath("research/source_controllers")
DERIVED_ACTOR_NPZ_SHA256 = "60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b"
DERIVED_ACTOR_STATE_SHA256 = "3fd39cc715a10126fd92b20f6ce213c380eb4d5df843a42315aac50cf116748a"

KNOWN_EXTERNAL_PAYLOAD_SHA256 = frozenset(
    digest
    for name in (
        "humanoid-v5-TQC-expert.zip",
        "humanoid-v5-TQC-expert/policy.pth",
        "humanoid-v5-TQC-expert/pytorch_variables.pth",
        "humanoid-v5-TQC-expert/data",
        "config.json",
    )
    if (entry := LOCAL_FILES[name]) is not None
    for digest in (entry[1],)
)
FORBIDDEN_EXTERNAL_PAYLOAD_SHA256 = KNOWN_EXTERNAL_PAYLOAD_SHA256 | {
    DERIVED_ACTOR_NPZ_SHA256,
    DERIVED_ACTOR_STATE_SHA256,
}


@dataclass(frozen=True, slots=True)
class TrackedBlob:
    path: str
    content: bytes


def _reject_duplicate_key(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ExperimentContractError("external actor import receipt has a duplicate key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> object:
    raise ExperimentContractError("external actor import receipt has a non-finite constant")


def _is_hdf5_payload(content: bytes) -> bool:
    signature = b"\x89HDF\r\n\x1a\n"
    offsets = [0]
    offset = 512
    while offset < len(content):
        offsets.append(offset)
        offset *= 2
    return any(content[position : position + len(signature)] == signature for position in offsets)


def forbidden_hashes_from_receipt(path: Path) -> frozenset[str]:
    """Bind source payloads plus the exact derived actor content and state hashes."""

    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise ExperimentContractError("external actor import receipt is unavailable") from exc
    if not 0 < len(payload) <= MAX_IMPORT_RECEIPT_BYTES:
        raise ExperimentContractError("external actor import receipt size is invalid")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_key,
            parse_constant=_reject_json_constant,
        )
    except ExperimentContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentContractError("external actor import receipt is invalid JSON") from exc
    receipt = validate_external_actor_import_receipt(value)
    actor = receipt["strict_actor_npz"]
    if (
        actor["sha256"] != DERIVED_ACTOR_NPZ_SHA256
        or actor["actor_state_sha256"] != DERIVED_ACTOR_STATE_SHA256
    ):
        raise ExperimentContractError("external actor payload policy binding differs")
    return FORBIDDEN_EXTERNAL_PAYLOAD_SHA256


def validate_tracked_blobs(
    blobs: Iterable[TrackedBlob],
    *,
    forbidden_sha256: frozenset[str],
) -> None:
    """Fail on byte identity, HDF5 data, or oversized source-controller blobs."""

    violations: list[str] = []
    for blob in blobs:
        if (
            type(blob) is not TrackedBlob
            or type(blob.path) is not str
            or type(blob.content) is not bytes
        ):
            raise ExperimentContractError("tracked blob record is invalid")
        digest = hashlib.sha256(blob.content).hexdigest()
        path = PurePosixPath(blob.path)
        if digest in forbidden_sha256:
            violations.append(f"{blob.path}: forbidden external payload bytes")
        if _is_hdf5_payload(blob.content):
            violations.append(f"{blob.path}: external dataset payload signature")
        try:
            path.relative_to(SOURCE_CONTROLLER_PREFIX)
        except ValueError:
            pass
        else:
            if len(blob.content) > MAX_SOURCE_CONTROLLER_BLOB_BYTES:
                violations.append(f"{blob.path}: source-controller blob exceeds 1 MiB")
    if violations:
        raise ExperimentContractError(
            "tracked external payload policy failed: " + "; ".join(violations)
        )


def _read_git_objects(repository: Path, object_ids: list[str]) -> dict[str, bytes]:
    request = "".join(f"{object_id}\n" for object_id in object_ids).encode("ascii")
    completed = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=repository,
        input=request,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 or completed.stderr:
        raise ExperimentContractError("git cat-file could not inspect the index")
    stream = io.BytesIO(completed.stdout)
    result: dict[str, bytes] = {}
    for expected_object_id in object_ids:
        header = stream.readline()
        try:
            observed_object_id, object_type, size_text = header.rstrip(b"\n").split(b" ")
            size = int(size_text)
        except (TypeError, ValueError) as exc:
            raise ExperimentContractError("git cat-file response is malformed") from exc
        if (
            observed_object_id.decode("ascii", errors="strict") != expected_object_id
            or object_type != b"blob"
            or size < 0
        ):
            raise ExperimentContractError("git cat-file returned a non-blob object")
        content = stream.read(size)
        if len(content) != size or stream.read(1) != b"\n":
            raise ExperimentContractError("git cat-file blob length differs")
        result[expected_object_id] = content
    if stream.read(1):
        raise ExperimentContractError("git cat-file returned trailing output")
    return result


def tracked_blobs_from_git_index(repository: Path) -> list[TrackedBlob]:
    """Read exact stage-zero blob bytes using only read-only Git commands."""

    repository = Path(repository)
    completed = subprocess.run(
        ["git", "ls-files", "--stage", "-z"],
        cwd=repository,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0 or completed.stderr:
        raise ExperimentContractError("git ls-files could not inspect the index")
    entries: list[tuple[str, str]] = []
    for raw_entry in completed.stdout.split(b"\x00"):
        if not raw_entry:
            continue
        try:
            metadata, raw_path = raw_entry.split(b"\t", 1)
            _mode, object_id, stage = metadata.decode("ascii").split(" ")
            path = raw_path.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ExperimentContractError("git index entry is malformed") from exc
        if stage != "0":
            raise ExperimentContractError("git index contains an unmerged entry")
        entries.append((path, object_id))
    unique_ids = list(dict.fromkeys(object_id for _, object_id in entries))
    objects = _read_git_objects(repository, unique_ids)
    return [TrackedBlob(path=path, content=objects[object_id]) for path, object_id in entries]


def enforce_external_payload_index_policy(
    repository: Path,
    *,
    import_receipt_path: Path,
) -> None:
    """Apply the payload policy to the repository's current Git index."""

    validate_tracked_blobs(
        tracked_blobs_from_git_index(Path(repository)),
        forbidden_sha256=forbidden_hashes_from_receipt(Path(import_receipt_path)),
    )


__all__ = [
    "DERIVED_ACTOR_NPZ_SHA256",
    "DERIVED_ACTOR_STATE_SHA256",
    "FORBIDDEN_EXTERNAL_PAYLOAD_SHA256",
    "KNOWN_EXTERNAL_PAYLOAD_SHA256",
    "MAX_SOURCE_CONTROLLER_BLOB_BYTES",
    "TrackedBlob",
    "enforce_external_payload_index_policy",
    "forbidden_hashes_from_receipt",
    "tracked_blobs_from_git_index",
    "validate_tracked_blobs",
]
