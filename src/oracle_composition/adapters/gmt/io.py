"""Bounded deterministic data-only archive helpers."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


class GMTAdmissionError(ValueError):
    """Raised when bytes do not match the pinned GMT admission contract."""


def sha256_file(path: Path, *, expected_size: int | None = None) -> str:
    observed_size = path.stat().st_size
    if expected_size is not None and observed_size != expected_size:
        raise GMTAdmissionError(
            f"size mismatch for {path.name}: expected {expected_size}, observed {observed_size}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_verified_bytes(
    path: Path,
    expected: str,
    *,
    expected_size: int | None = None,
    maximum_size: int = 16 * 1024 * 1024,
) -> bytes:
    observed_size = path.stat().st_size
    if expected_size is not None and observed_size != expected_size:
        raise GMTAdmissionError(
            f"size mismatch for {path.name}: expected {expected_size}, observed {observed_size}"
        )
    if observed_size > maximum_size:
        raise GMTAdmissionError(f"input exceeds {maximum_size} bytes: {path.name}")
    payload = path.read_bytes()
    if len(payload) != observed_size:
        raise GMTAdmissionError(f"input changed while being read: {path.name}")
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise GMTAdmissionError(
            f"SHA-256 mismatch for {path.name}: expected {expected}, observed {observed}"
        )
    return payload


def require_sha256(path: Path, expected: str, *, expected_size: int | None = None) -> None:
    read_verified_bytes(path, expected, expected_size=expected_size)


def _publish_without_overwrite(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise GMTAdmissionError(f"refusing to overwrite existing output: {destination}") from exc
    temporary.unlink()


def validate_zip_members(
    archive: zipfile.ZipFile,
    *,
    expected_count: int | None = None,
    expected_uncompressed_size: int | None = None,
    required_prefix: str | None = None,
    maximum_member_size: int = 8 * 1024 * 1024,
) -> dict[str, zipfile.ZipInfo]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise GMTAdmissionError("archive contains duplicate member names")
    if expected_count is not None and len(infos) != expected_count:
        raise GMTAdmissionError(
            f"archive member count mismatch: expected {expected_count}, observed {len(infos)}"
        )
    total = sum(info.file_size for info in infos)
    if expected_uncompressed_size is not None and total != expected_uncompressed_size:
        raise GMTAdmissionError(
            "archive uncompressed size mismatch: "
            f"expected {expected_uncompressed_size}, observed {total}"
        )
    for info in infos:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or ".." in path.parts:
            raise GMTAdmissionError(f"unsafe archive path: {info.filename}")
        if required_prefix is not None and not info.filename.startswith(required_prefix):
            raise GMTAdmissionError(f"member outside declared prefix: {info.filename}")
        if info.flag_bits & 0x1:
            raise GMTAdmissionError(f"encrypted archive member: {info.filename}")
        if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
            raise GMTAdmissionError(f"unsupported compression for {info.filename}")
        if info.file_size > maximum_member_size:
            raise GMTAdmissionError(f"oversized archive member: {info.filename}")
    return {info.filename: info for info in infos}


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.ascontiguousarray(array), allow_pickle=False)
    return stream.getvalue()


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    if path.exists():
        raise GMTAdmissionError(f"refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for key in sorted(arrays):
                if not key or "/" in key or "\\" in key:
                    raise GMTAdmissionError(f"unsafe NPZ key: {key!r}")
                info = zipfile.ZipInfo(f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o600 << 16
                archive.writestr(info, _npy_bytes(arrays[key]))
        _publish_without_overwrite(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_file(path)


def write_json_receipt(path: Path, payload: Mapping[str, Any]) -> str:
    if path.exists():
        raise GMTAdmissionError(f"refusing to overwrite existing receipt: {path}")
    serialized = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_without_overwrite(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(serialized).hexdigest()
