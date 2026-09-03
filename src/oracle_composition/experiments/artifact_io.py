"""Descriptor-bound publication for local research artifacts."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import secrets
import stat
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from .fixed_reference import ExperimentContractError


@dataclass(frozen=True, slots=True)
class PublishedArtifact:
    path: Path
    sha256: str
    byte_count: int


def finite_pretty_json(value: object) -> bytes:
    """Encode deterministic finite JSON with one terminal newline."""

    try:
        return (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExperimentContractError(f"value is not finite JSON: {exc}") from exc


def _open_parent_directory(path: Path, *, create_missing: bool = True) -> int:
    absolute = Path(os.path.abspath(path))
    if not absolute.is_absolute():
        raise ExperimentContractError("artifact output parent must be absolute")
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute.anchor, flags)
    except OSError as exc:
        raise ExperimentContractError("cannot open artifact output root") from exc
    try:
        for component in absolute.parts[1:]:
            if component in {"", ".", ".."}:
                raise ExperimentContractError("artifact output path is unsafe")
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create_missing:
                    raise
                with suppress(FileExistsError):
                    os.mkdir(component, mode=0o700, dir_fd=descriptor)
                child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except OSError as exc:
        os.close(descriptor)
        raise ExperimentContractError(
            "artifact output ancestors must be real directories without symbolic links"
        ) from exc
    except Exception:
        os.close(descriptor)
        raise


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _descriptor_sha256(descriptor: int, *, expected_size: int) -> str:
    os.lseek(descriptor, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    remaining = expected_size
    while remaining:
        chunk = os.read(descriptor, min(1024 * 1024, remaining))
        if not chunk:
            raise ExperimentContractError("published artifact became shorter during verification")
        digest.update(chunk)
        remaining -= len(chunk)
    if os.read(descriptor, 1):
        raise ExperimentContractError("published artifact grew during verification")
    return digest.hexdigest()


def _rename_without_overwrite(parent_descriptor: int, source: str, target: str) -> None:
    """Atomically move one entry without replacing an existing destination."""

    library = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    if sys.platform == "darwin":
        rename = library.renameatx_np
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        flags = 0x00000004 | 0x00000010 | 0x00000020
    elif sys.platform.startswith("linux"):
        try:
            rename = library.renameat2
        except AttributeError as exc:
            raise ExperimentContractError(
                "this platform lacks atomic no-overwrite artifact publication"
            ) from exc
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        flags = 1
    else:
        raise ExperimentContractError(
            "this platform lacks atomic no-overwrite artifact publication"
        )
    ctypes.set_errno(0)
    result = rename(
        parent_descriptor,
        source_bytes,
        parent_descriptor,
        target_bytes,
        flags,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(error_number, os.strerror(error_number), target)
    raise OSError(error_number, os.strerror(error_number), target)


def publish_bytes_without_overwrite(path: Path, encoded: bytes) -> PublishedArtifact:
    """Publish complete bytes atomically; retain, but never endorse, failed attempts."""

    if not isinstance(encoded, bytes) or not encoded:
        raise ExperimentContractError("published artifact bytes must be nonempty")
    requested = Path(path)
    if requested.name in {"", ".", ".."} or "\0" in requested.name:
        raise ExperimentContractError("published artifact must have one filename")
    absolute = Path(os.path.abspath(requested))
    parent_descriptor = _open_parent_directory(absolute.parent)
    try:
        os.stat(absolute.name, dir_fd=parent_descriptor, follow_symlinks=False)
    except FileNotFoundError:
        pass
    except OSError as exc:
        os.close(parent_descriptor)
        raise ExperimentContractError(f"cannot inspect artifact path {absolute}: {exc}") from exc
    else:
        os.close(parent_descriptor)
        raise ExperimentContractError(f"refusing to overwrite existing file: {absolute}")
    file_descriptor: int | None = None
    created: os.stat_result | None = None
    pending_name: str | None = None
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        for _attempt in range(100):
            pending_name = f".{absolute.name}.{secrets.token_hex(16)}.pending"
            try:
                file_descriptor = os.open(
                    pending_name,
                    flags,
                    0o600,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                continue
            break
        if file_descriptor is None or pending_name is None:
            raise ExperimentContractError("cannot allocate a unique pending artifact")
        created = os.fstat(file_descriptor)
        if not stat.S_ISREG(created.st_mode):
            raise ExperimentContractError("published artifact is not a regular file")
        os.fchmod(file_descriptor, 0o600)
        view = memoryview(encoded)
        written = 0
        while written < len(view):
            count = os.write(file_descriptor, view[written:])
            if count <= 0:
                raise ExperimentContractError("artifact write made no progress")
            written += count
        os.fsync(file_descriptor)
        expected_sha256 = hashlib.sha256(encoded).hexdigest()
        if _descriptor_sha256(file_descriptor, expected_size=len(encoded)) != expected_sha256:
            raise ExperimentContractError("published artifact bytes changed before publication")
        pending_state = os.stat(
            pending_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not _same_file(created, pending_state):
            raise ExperimentContractError("pending artifact path changed before publication")
        try:
            _rename_without_overwrite(parent_descriptor, pending_name, absolute.name)
        except FileExistsError as exc:
            raise ExperimentContractError(
                f"refusing to overwrite existing file: {absolute}"
            ) from exc
        pending_name = None
        final_entry_state = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if not _same_file(created, final_entry_state):
            raise ExperimentContractError("published artifact path changed during publication")
        os.fsync(parent_descriptor)
        final_descriptor_state = os.fstat(file_descriptor)
        final_entry_state = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not _same_file(created, final_descriptor_state)
            or not _same_file(created, final_entry_state)
            or final_descriptor_state.st_size != len(encoded)
            or final_entry_state.st_size != len(encoded)
            or final_descriptor_state.st_nlink != 1
            or stat.S_IMODE(final_descriptor_state.st_mode) != 0o600
        ):
            raise ExperimentContractError("published artifact changed during publication")
        if _descriptor_sha256(file_descriptor, expected_size=len(encoded)) != expected_sha256:
            raise ExperimentContractError("published artifact bytes changed during publication")
        verification_parent = _open_parent_directory(
            absolute.parent,
            create_missing=False,
        )
        try:
            if not _same_file(os.fstat(parent_descriptor), os.fstat(verification_parent)):
                raise ExperimentContractError(
                    "artifact output ancestors changed during publication"
                )
            visible_state = os.stat(
                absolute.name,
                dir_fd=verification_parent,
                follow_symlinks=False,
            )
            if not _same_file(created, visible_state):
                raise ExperimentContractError(
                    "published artifact is not visible at the requested path"
                )
        finally:
            os.close(verification_parent)
        return PublishedArtifact(
            path=absolute,
            sha256=expected_sha256,
            byte_count=len(encoded),
        )
    except OSError as exc:
        raise ExperimentContractError(f"cannot publish artifact {absolute}: {exc}") from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(parent_descriptor)


def publish_json_without_overwrite(path: Path, value: object) -> PublishedArtifact:
    """Publish deterministic JSON without a pathname re-read."""

    return publish_bytes_without_overwrite(path, finite_pretty_json(value))


__all__ = [
    "PublishedArtifact",
    "finite_pretty_json",
    "publish_bytes_without_overwrite",
    "publish_json_without_overwrite",
]
