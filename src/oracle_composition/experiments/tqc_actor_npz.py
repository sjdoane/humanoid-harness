"""Strict data-only TQC actor archive."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Protocol
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

import numpy as np

from .artifact_io import publish_bytes_without_overwrite
from .fixed_reference import ExperimentContractError

FORMAT_ID = "strict_tqc_actor_npz_npy1_c_order_no_pickle/v1"
ARCHITECTURE_ID = "sb3_contrib_tqc_actor_348_256_256_17_relu_state_dependent_std/v1"
OBSERVATION_WIDTH = 348
ACTION_WIDTH = 17
MAX_ARCHIVE_BYTES = 2 * 1024 * 1024
MAX_MEMBER_BYTES = 512 * 1024
MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024

_FLOAT32 = np.dtype("<f4")
_INT64 = np.dtype("<i8")
_ARCHIVE_SCHEMA: Mapping[str, tuple[tuple[int, ...], np.dtype[object]]] = {
    "latent_pi.0.weight": ((256, OBSERVATION_WIDTH), _FLOAT32),
    "latent_pi.0.bias": ((256,), _FLOAT32),
    "latent_pi.2.weight": ((256, 256), _FLOAT32),
    "latent_pi.2.bias": ((256,), _FLOAT32),
    "mu.weight": ((ACTION_WIDTH, 256), _FLOAT32),
    "mu.bias": ((ACTION_WIDTH,), _FLOAT32),
    "log_std.weight": ((ACTION_WIDTH, 256), _FLOAT32),
    "log_std.bias": ((ACTION_WIDTH,), _FLOAT32),
    "action_low": ((ACTION_WIDTH,), _FLOAT32),
    "action_high": ((ACTION_WIDTH,), _FLOAT32),
    "format_version": ((1,), _INT64),
}

# Process-local capability: only the strict file loader may issue this record.
# It prevents accidental constructor forgery, not hostile in-process code.
_LOADED_ACTOR_ISSUER = object()


class _HashDigest(Protocol):
    def update(self, value: bytes) -> None: ...


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExperimentContractError(f"value is not canonical JSON: {exc}") from exc


def actor_schema() -> dict[str, object]:
    """Return the immutable archive layout and inference boundary."""

    return {
        "format_id": FORMAT_ID,
        "architecture_id": ARCHITECTURE_ID,
        "members_in_order": [
            {"name": name, "shape": list(shape), "dtype": dtype.str}
            for name, (shape, dtype) in _ARCHIVE_SCHEMA.items()
        ],
        "activation": "torch.nn.ReLU",
        "distribution": "squashed_diagonal_gaussian_state_dependent_log_std",
        "deterministic_normalized_action": "tanh(mu(latent_pi(float32_observation)))",
        "physical_action_transform": "low+(normalized+1)*(high-low)/2",
        "contains_executable_code": False,
        "contains_critic": False,
        "contains_optimizer": False,
        "contains_replay": False,
        "contains_rng_state": False,
        "exact_training_resume": False,
    }


def actor_schema_sha256() -> str:
    return hashlib.sha256(_canonical_json(actor_schema())).hexdigest()


def _canonical_sha256(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError("expected_sha256 must be a lowercase SHA-256")
    return value


def _coerce_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[object],
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ExperimentContractError(f"actor member {name} must be a NumPy array")
    if value.shape != shape:
        raise ExperimentContractError(f"actor member {name} has the wrong shape")
    if value.dtype != dtype or value.dtype.str != dtype.str:
        raise ExperimentContractError(f"actor member {name} has the wrong dtype")
    if not value.flags.c_contiguous:
        raise ExperimentContractError(f"actor member {name} must be C-contiguous")
    if not np.isfinite(value).all():
        raise ExperimentContractError(f"actor member {name} contains non-finite values")
    return value.copy(order="C")


def validate_actor_arrays(values: Mapping[str, object]) -> dict[str, np.ndarray]:
    """Validate the exact actor state and fixed Humanoid action transform."""

    if set(values) != set(_ARCHIVE_SCHEMA):
        raise ExperimentContractError(
            "TQC actor keys differ from the schema: "
            f"missing={sorted(set(_ARCHIVE_SCHEMA) - set(values))}, "
            f"extra={sorted(set(values) - set(_ARCHIVE_SCHEMA))}"
        )
    result = {
        name: _coerce_array(values[name], name=name, shape=shape, dtype=dtype)
        for name, (shape, dtype) in _ARCHIVE_SCHEMA.items()
    }
    if not np.array_equal(
        result["action_low"],
        np.full(ACTION_WIDTH, -0.4, dtype=_FLOAT32),
    ) or not np.array_equal(
        result["action_high"],
        np.full(ACTION_WIDTH, 0.4, dtype=_FLOAT32),
    ):
        raise ExperimentContractError("actor action bounds must equal the Humanoid-v5 ABI")
    if not np.array_equal(result["format_version"], np.asarray([1], dtype=_INT64)):
        raise ExperimentContractError("actor format_version must equal 1")
    return result


def actor_state_sha256(values: Mapping[str, object]) -> str:
    """Hash semantic array names, layouts, and bytes independent of ZIP metadata."""

    arrays = validate_actor_arrays(values)
    digest = hashlib.sha256(b"strict_tqc_actor_state/v1\0")
    for name, array in arrays.items():
        metadata = _canonical_json(
            {"dtype": array.dtype.str, "name": name, "shape": list(array.shape)}
        )
        raw = array.tobytes(order="C")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, value, version=(1, 0), allow_pickle=False)
    return stream.getvalue()


def encode_actor_npz(values: Mapping[str, object]) -> bytes:
    """Encode one deterministic, bounded, code-free NPZ payload."""

    arrays = validate_actor_arrays(values)
    stream = io.BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in arrays.items():
            member = ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o100600 << 16
            archive.writestr(member, _npy_bytes(value), compress_type=ZIP_DEFLATED, compresslevel=9)
    payload = stream.getvalue()
    if not 0 < len(payload) <= MAX_ARCHIVE_BYTES:
        raise ExperimentContractError("encoded TQC actor exceeds its archive bound")
    return payload


def write_actor_npz_exclusive(path: Path, values: Mapping[str, object]) -> str:
    """Publish one archive through the descriptor-bound artifact writer."""

    return publish_bytes_without_overwrite(Path(path), encode_actor_npz(values)).sha256


def _open_parent_without_links(path: Path) -> tuple[Path, int]:
    absolute = Path(os.path.abspath(path))
    if absolute.name in {"", ".", ".."} or "\0" in absolute.name:
        raise ExperimentContractError("TQC actor path must name one file")
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(absolute.anchor, flags)
        for component in absolute.parent.parts[1:]:
            if component in {"", ".", ".."}:
                raise ExperimentContractError("TQC actor path is unsafe")
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return absolute, descriptor
    except ExperimentContractError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise ExperimentContractError(
            "TQC actor ancestors must be real directories without symbolic links"
        ) from exc


def _read_bound_file(path: Path, *, expected_sha256: str) -> bytes:
    expected = _canonical_sha256(expected_sha256)
    absolute, parent_descriptor = _open_parent_without_links(path)
    descriptor: int | None = None
    try:
        entry_before = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(entry_before.st_mode):
            raise ExperimentContractError("TQC actor path must not be a symlink")
        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(absolute.name, flags, dir_fd=parent_descriptor)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= MAX_ARCHIVE_BYTES
        ):
            raise ExperimentContractError("TQC actor must be a bounded regular file")
        if (entry_before.st_dev, entry_before.st_ino) != (before.st_dev, before.st_ino):
            raise ExperimentContractError("TQC actor changed while it was opened")
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            payload = stream.read(MAX_ARCHIVE_BYTES + 1)
        after = os.fstat(descriptor)
        identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in identity):
            raise ExperimentContractError("TQC actor changed while it was read")
        if len(payload) != before.st_size or len(payload) > MAX_ARCHIVE_BYTES:
            raise ExperimentContractError("TQC actor length changed or exceeded its bound")
        entry_after = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        verification_absolute, verification_parent = _open_parent_without_links(absolute)
        try:
            verification_entry = os.stat(
                verification_absolute.name,
                dir_fd=verification_parent,
                follow_symlinks=False,
            )
            parent_state = os.fstat(parent_descriptor)
            verification_parent_state = os.fstat(verification_parent)
            parent_unchanged = (
                parent_state.st_dev,
                parent_state.st_ino,
            ) == (
                verification_parent_state.st_dev,
                verification_parent_state.st_ino,
            )
        finally:
            os.close(verification_parent)
        if not parent_unchanged or any(
            getattr(entry_before, field) != getattr(entry_after, field)
            or getattr(entry_before, field) != getattr(verification_entry, field)
            for field in identity
        ):
            raise ExperimentContractError("TQC actor path changed while it was read")
    except ExperimentContractError:
        raise
    except OSError as exc:
        raise ExperimentContractError(f"cannot open TQC actor archive: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_descriptor)
    observed = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(observed, expected):
        raise ExperimentContractError("TQC actor content SHA-256 does not match")
    return payload


def _parse_npy(
    payload: bytes,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[object],
) -> np.ndarray:
    stream = io.BytesIO(payload)
    try:
        version = np.lib.format.read_magic(stream)
        if version != (1, 0):
            raise ExperimentContractError(f"actor member {name} must use NPY 1.0")
        observed_shape, fortran_order, observed_dtype = np.lib.format.read_array_header_1_0(stream)
    except ExperimentContractError:
        raise
    except (EOFError, TypeError, ValueError) as exc:
        raise ExperimentContractError(f"actor member {name} has an invalid NPY header") from exc
    if observed_dtype.hasobject:
        raise ExperimentContractError(f"actor member {name} has an unsafe object dtype")
    if tuple(observed_shape) != shape or observed_dtype != dtype or observed_dtype.str != dtype.str:
        raise ExperimentContractError(f"actor member {name} differs from its array schema")
    if fortran_order:
        raise ExperimentContractError(f"actor member {name} must use C-order storage")
    count = int(np.prod(shape, dtype=np.int64))
    expected_bytes = count * dtype.itemsize
    if len(payload) - stream.tell() != expected_bytes:
        raise ExperimentContractError(f"actor member {name} payload length is invalid")
    value = np.frombuffer(payload, dtype=dtype, count=count, offset=stream.tell()).reshape(shape)
    if not np.isfinite(value).all():
        raise ExperimentContractError(f"actor member {name} contains non-finite values")
    return value.copy(order="C")


@dataclass(frozen=True, slots=True)
class LoadedTQCActor:
    """Canonical actor bytes admitted by :func:`load_actor_npz`."""

    content_sha256: str
    state_sha256: str
    schema_sha256: str
    byte_count: int
    arrays: Mapping[str, np.ndarray]
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _LOADED_ACTOR_ISSUER:
            raise ExperimentContractError("loaded TQC actors may only be issued by load_actor_npz")
        for value, field in (
            (self.content_sha256, "actor content SHA-256"),
            (self.state_sha256, "actor state SHA-256"),
            (self.schema_sha256, "actor schema SHA-256"),
        ):
            try:
                _canonical_sha256(value)
            except ExperimentContractError as exc:
                raise ExperimentContractError(f"{field} is invalid") from exc
        if type(self.byte_count) is not int or not 0 < self.byte_count <= MAX_ARCHIVE_BYTES:
            raise ExperimentContractError("loaded actor byte count is invalid")
        if self.schema_sha256 != actor_schema_sha256():
            raise ExperimentContractError("loaded actor schema SHA-256 differs")
        if actor_state_sha256(self.arrays) != self.state_sha256:
            raise ExperimentContractError("loaded actor state SHA-256 differs")
        canonical_payload = encode_actor_npz(self.arrays)
        if len(canonical_payload) != self.byte_count:
            raise ExperimentContractError("loaded actor byte count differs from canonical bytes")
        if hashlib.sha256(canonical_payload).hexdigest() != self.content_sha256:
            raise ExperimentContractError("loaded actor content differs from canonical bytes")


def load_actor_npz(path: Path, *, expected_sha256: str) -> LoadedTQCActor:
    """Load exact arrays without invoking NumPy pickle or Torch deserialization."""

    payload = _read_bound_file(Path(path), expected_sha256=expected_sha256)
    if not payload.startswith(b"PK\x03\x04"):
        raise ExperimentContractError("TQC actor is not an NPZ archive")
    if len(payload) < 22 or payload[-22:-18] != b"PK\x05\x06" or payload[-2:] != b"\x00\x00":
        raise ExperimentContractError("TQC actor must end at an un-commented ZIP record")
    expected_names = tuple(f"{name}.npy" for name in _ARCHIVE_SCHEMA)
    try:
        with ZipFile(io.BytesIO(payload), "r") as archive:
            infos = archive.infolist()
            names = tuple(info.filename for info in infos)
            if names != expected_names or len(set(names)) != len(names):
                raise ExperimentContractError("TQC actor member order or key set differs")
            if archive.comment or sum(info.file_size for info in infos) > MAX_UNCOMPRESSED_BYTES:
                raise ExperimentContractError("TQC actor expansion or comment is invalid")
            arrays: dict[str, np.ndarray] = {}
            for info, (name, (shape, dtype)) in zip(infos, _ARCHIVE_SCHEMA.items(), strict=True):
                member_mode = (info.external_attr >> 16) & 0xFFFF
                if (
                    info.is_dir()
                    or info.flag_bits != 0
                    or info.compress_type != ZIP_DEFLATED
                    or not 0 < info.file_size <= MAX_MEMBER_BYTES
                    or info.create_system != 3
                    or member_mode != 0o100600
                    or info.date_time != (1980, 1, 1, 0, 0, 0)
                    or bool(info.extra)
                    or bool(info.comment)
                ):
                    raise ExperimentContractError(
                        f"TQC actor member {name} has noncanonical ZIP metadata"
                    )
                with archive.open(info, "r") as member_stream:
                    member_payload = member_stream.read(info.file_size + 1)
                if len(member_payload) != info.file_size:
                    raise ExperimentContractError(f"TQC actor member {name} length is invalid")
                arrays[name] = _parse_npy(
                    member_payload,
                    name=name,
                    shape=shape,
                    dtype=dtype,
                )
    except ExperimentContractError:
        raise
    except (BadZipFile, EOFError, OSError, RuntimeError) as exc:
        raise ExperimentContractError(f"TQC actor archive is invalid: {exc}") from exc
    validated = validate_actor_arrays(arrays)
    if encode_actor_npz(validated) != payload:
        raise ExperimentContractError("TQC actor archive bytes are not canonical")
    immutable = {
        name: np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)
        for name, value in validated.items()
    }
    return LoadedTQCActor(
        content_sha256=hashlib.sha256(payload).hexdigest(),
        state_sha256=actor_state_sha256(immutable),
        schema_sha256=actor_schema_sha256(),
        byte_count=len(payload),
        arrays=MappingProxyType(immutable),
        _issuer=_LOADED_ACTOR_ISSUER,
    )


__all__ = [
    "ACTION_WIDTH",
    "ARCHITECTURE_ID",
    "FORMAT_ID",
    "MAX_ARCHIVE_BYTES",
    "OBSERVATION_WIDTH",
    "LoadedTQCActor",
    "actor_schema",
    "actor_schema_sha256",
    "actor_state_sha256",
    "encode_actor_npz",
    "load_actor_npz",
    "validate_actor_arrays",
    "write_actor_npz_exclusive",
]
