"""Git-index policy for non-redistributable external controller payloads."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile

import numpy as np

from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_actor_npz import actor_schema, actor_state_sha256

from .external_sb3_actor import validate_external_actor_import_receipt
from .farama_tqc_registration import LOCAL_FILES

MAX_SOURCE_CONTROLLER_BLOB_BYTES = 1024 * 1024
MAX_IMPORT_RECEIPT_BYTES = 1024 * 1024
MAX_SEMANTIC_NPZ_BYTES = 2 * 1024 * 1024
MAX_SEMANTIC_NPZ_MEMBERS = 16
MAX_SEMANTIC_NPY_MEMBER_BYTES = 512 * 1024
MAX_SEMANTIC_NPZ_UNCOMPRESSED_BYTES = 2 * 1024 * 1024
MAX_NPY_HEADER_BYTES = 4 * 1024
MAX_GIT_WARNING_BYTES = 64 * 1024
MAX_GIT_WARNING_LINES = 128
MAX_GIT_WARNING_LINE_BYTES = 2 * 1024
SOURCE_CONTROLLER_PREFIX = PurePosixPath("research/source_controllers")
DERIVED_ACTOR_NPZ_SHA256 = "60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b"
DERIVED_ACTOR_STATE_SHA256 = "3fd39cc715a10126fd92b20f6ce213c380eb4d5df843a42315aac50cf116748a"
DERIVED_SIBLING_ACTOR_IDENTITIES = {
    "medium": {
        "npz_sha256": "2677ebb70cd20e0ba8f8a591814fc853e325277db65184ebafb5bf5e21198b04",
        "state_sha256": "cffb5679e3ef10cc99980819013941cb55cdc9f2198ba4a91a306dfc5f73e00c",
    },
    "simple": {
        "npz_sha256": "b09aa921316640024e9703671d328d7ecbabe76f04cfed8c1d8fb86fbd95a917",
        "state_sha256": "3068049a0d18b7924d117a96aafee7127e411a70eba344ff985d31e97da3312b",
    },
}
SIBLING_SOURCE_PAYLOAD_SHA256 = frozenset(
    {
        "f47ae84f39c61416ddff1ce0a68dba446ea703a354469480515abd8570fa2779",
        "d54c93dd82d97cd931caf85fcba5b4722b9c86075e9c679576b6ba45b7823c66",
        "d00cd2dfdc0bc65964a9e4a47a7fef250d8ae3a006e01fe50345842dd945d41e",
        "63194995ab0605aacd68c6f160619c3cc5717233cee1a5e6ff8038a92fa644ad",
        "21d57bf16324533f6ed2275827a7f3b9e13b991a29927c3b131b2c9c98297ac7",
        "4814a61d04158a9535cec8d8173d4e2cb4f7be2e070b257b9d27215539d65735",
        "ca9aff0dc359d6011ddde33f196fc8b612781cba80247fb8a9889eb1df74e58e",
        "16f7205a840137b520ff4c800236c4e440d3168e362d68055dac78565ccd476c",
        "a689c4486127c8828a48ebc420493b36176b41f858cbffdc26517b052cfcbaea",
        "7b76d3edaee03ca9a9a24916013c3291cd4d053c3781bf285e5542f964894451",
    }
)

KNOWN_EXTERNAL_PAYLOAD_SHA256 = (
    frozenset(
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
    | SIBLING_SOURCE_PAYLOAD_SHA256
)
FORBIDDEN_EXTERNAL_PAYLOAD_SHA256 = KNOWN_EXTERNAL_PAYLOAD_SHA256 | {
    DERIVED_ACTOR_NPZ_SHA256,
    DERIVED_ACTOR_STATE_SHA256,
    *(
        digest
        for identity in DERIVED_SIBLING_ACTOR_IDENTITIES.values()
        for digest in identity.values()
    ),
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


def _read_npy_metadata(payload: bytes) -> tuple[tuple[int, ...], np.dtype[object], bool]:
    if not 0 < len(payload) <= MAX_SEMANTIC_NPY_MEMBER_BYTES:
        raise ExperimentContractError("NumPy ZIP member size is invalid")
    stream = io.BytesIO(payload)
    try:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(
                stream,
                max_header_size=MAX_NPY_HEADER_BYTES,
            )
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(
                stream,
                max_header_size=MAX_NPY_HEADER_BYTES,
            )
        else:
            raise ExperimentContractError("NumPy ZIP member format version is refused")
    except ExperimentContractError:
        raise
    except (EOFError, TypeError, ValueError, OverflowError) as exc:
        raise ExperimentContractError("NumPy ZIP member header is malformed") from exc
    if (
        type(shape) is not tuple
        or len(shape) > 4
        or any(type(width) is not int or width <= 0 for width in shape)
        or type(fortran_order) is not bool
        or not isinstance(dtype, np.dtype)
        or dtype.hasobject
        or dtype.fields is not None
        or dtype.subdtype is not None
        or dtype.itemsize <= 0
    ):
        raise ExperimentContractError("NumPy ZIP member dtype or shape is refused")
    element_count = 1
    for width in shape:
        element_count *= width
        if element_count * dtype.itemsize > MAX_SEMANTIC_NPY_MEMBER_BYTES:
            raise ExperimentContractError("NumPy ZIP member array exceeds its bound")
    if stream.tell() + element_count * dtype.itemsize != len(payload):
        raise ExperimentContractError("NumPy ZIP member byte count differs")
    return shape, dtype, fortran_order


def _semantic_actor_fingerprint(content: bytes) -> str | None:
    zip_prefix = content.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"))
    zip_eocd = b"PK\x05\x06" in content[-65_557:]
    if not zip_prefix and not zip_eocd:
        return None
    if not 0 < len(content) <= MAX_SEMANTIC_NPZ_BYTES:
        raise ExperimentContractError("candidate ZIP exceeds its semantic inspection bound")
    try:
        with ZipFile(io.BytesIO(content), "r") as archive:
            infos = archive.infolist()
            if not 0 < len(infos) <= MAX_SEMANTIC_NPZ_MEMBERS:
                raise ExperimentContractError("candidate ZIP member count is invalid")
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ExperimentContractError("candidate ZIP contains duplicate members")
            if archive.comment:
                raise ExperimentContractError("candidate ZIP comment is refused")
            if not all(name.endswith(".npy") for name in names):
                return None
            total = 0
            member_payloads: dict[str, bytes] = {}
            member_metadata: dict[str, tuple[tuple[int, ...], str, bool]] = {}
            for info in infos:
                name = info.filename
                path = PurePosixPath(name)
                if (
                    type(name) is not str
                    or path.is_absolute()
                    or len(path.parts) != 1
                    or any(part in {"", ".", ".."} for part in path.parts)
                    or info.is_dir()
                    or not 0 < info.file_size <= MAX_SEMANTIC_NPY_MEMBER_BYTES
                    or info.compress_size < 0
                    or info.compress_type not in {ZIP_STORED, ZIP_DEFLATED}
                    or bool(info.flag_bits & 0x1)
                ):
                    raise ExperimentContractError("candidate ZIP member metadata is unsafe")
                total += info.file_size
                if total > MAX_SEMANTIC_NPZ_UNCOMPRESSED_BYTES:
                    raise ExperimentContractError("candidate ZIP expansion exceeds its bound")
                with archive.open(info, "r") as stream:
                    payload = stream.read(MAX_SEMANTIC_NPY_MEMBER_BYTES + 1)
                    if stream.read(1) or len(payload) != info.file_size:
                        raise ExperimentContractError("candidate ZIP member length differs")
                shape, dtype, fortran_order = _read_npy_metadata(payload)
                key = name.removesuffix(".npy")
                member_payloads[key] = payload
                member_metadata[key] = (shape, dtype.str, fortran_order)
    except ExperimentContractError:
        raise
    except (BadZipFile, EOFError, OSError, RuntimeError, ValueError, zlib.error) as exc:
        raise ExperimentContractError("candidate ZIP is malformed") from exc

    expected = {
        item["name"]: (tuple(item["shape"]), item["dtype"], False)
        for item in actor_schema()["members_in_order"]
    }
    if member_metadata != expected:
        return None
    arrays: dict[str, np.ndarray] = {}
    try:
        for name, payload in member_payloads.items():
            value = np.load(
                io.BytesIO(payload),
                allow_pickle=False,
                max_header_size=MAX_NPY_HEADER_BYTES,
            )
            if type(value) is not np.ndarray:
                raise ExperimentContractError("NumPy ZIP member is not an array")
            arrays[name] = value
        return actor_state_sha256(arrays)
    except ExperimentContractError:
        raise
    except (EOFError, OSError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        raise ExperimentContractError("candidate actor ZIP array load refused") from exc


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
        try:
            semantic_fingerprint = _semantic_actor_fingerprint(blob.content)
        except ExperimentContractError as exc:
            violations.append(f"{blob.path}: semantic NumPy ZIP refused ({exc})")
            continue
        if semantic_fingerprint in forbidden_sha256:
            violations.append(f"{blob.path}: forbidden external actor state")
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


def _classify_git_warnings(stderr: bytes) -> tuple[str, ...]:
    if type(stderr) is not bytes or len(stderr) > MAX_GIT_WARNING_BYTES:
        raise ExperimentContractError("git warning output exceeds its bound")
    if not stderr:
        return ()
    try:
        text = stderr.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ExperimentContractError("git warning output is not UTF-8") from exc
    lines = text.splitlines()
    if (
        not lines
        or len(lines) > MAX_GIT_WARNING_LINES
        or any(
            not line
            or len(line.encode("utf-8")) > MAX_GIT_WARNING_LINE_BYTES
            or any(ord(character) < 32 and character != "\t" for character in line)
            for line in lines
        )
    ):
        raise ExperimentContractError("git warning output is malformed")
    return tuple(lines)


def _validate_git_object_id(value: str) -> str:
    if (
        type(value) is not str
        or len(value) not in {40, 64}
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError("git object ID is malformed")
    return value


def _read_git_objects(repository: Path, object_ids: list[str]) -> dict[str, bytes]:
    if type(object_ids) is not list or any(
        _validate_git_object_id(object_id) != object_id for object_id in object_ids
    ):
        raise ExperimentContractError("git object request is malformed")
    request = "".join(f"{object_id}\n" for object_id in object_ids).encode("ascii")
    completed = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=repository,
        input=request,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ExperimentContractError("git cat-file could not inspect the index")
    _classify_git_warnings(completed.stderr)
    stream = io.BytesIO(completed.stdout)
    result: dict[str, bytes] = {}
    for expected_object_id in object_ids:
        header = stream.readline()
        try:
            if not header.endswith(b"\n"):
                raise ValueError("header is incomplete")
            observed_object_id, object_type, size_text = header[:-1].split(b" ")
            observed_text = observed_object_id.decode("ascii", errors="strict")
            _validate_git_object_id(observed_text)
            if not size_text.isdigit():
                raise ValueError("size is not decimal")
            size = int(size_text)
        except (ExperimentContractError, TypeError, UnicodeDecodeError, ValueError) as exc:
            raise ExperimentContractError("git cat-file response is malformed") from exc
        if observed_text != expected_object_id or object_type != b"blob" or size < 0:
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
    if completed.returncode != 0:
        raise ExperimentContractError("git ls-files could not inspect the index")
    _classify_git_warnings(completed.stderr)
    entries: list[tuple[str, str]] = []
    for raw_entry in completed.stdout.split(b"\x00"):
        if not raw_entry:
            continue
        try:
            metadata, raw_path = raw_entry.split(b"\t", 1)
            mode, object_id, stage = metadata.decode("ascii").split(" ")
            path = raw_path.decode("utf-8", errors="strict")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ExperimentContractError("git index entry is malformed") from exc
        if (
            len(mode) != 6
            or any(character not in "01234567" for character in mode)
            or stage != "0"
            or not path
            or path in {entry_path for entry_path, _ in entries}
        ):
            raise ExperimentContractError("git index contains an unmerged entry")
        _validate_git_object_id(object_id)
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
    "DERIVED_SIBLING_ACTOR_IDENTITIES",
    "FORBIDDEN_EXTERNAL_PAYLOAD_SHA256",
    "KNOWN_EXTERNAL_PAYLOAD_SHA256",
    "MAX_SOURCE_CONTROLLER_BLOB_BYTES",
    "SIBLING_SOURCE_PAYLOAD_SHA256",
    "TrackedBlob",
    "enforce_external_payload_index_policy",
    "forbidden_hashes_from_receipt",
    "tracked_blobs_from_git_index",
    "validate_tracked_blobs",
]
