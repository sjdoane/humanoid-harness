"""Descriptor-bound publication for local research artifacts."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import json
import os
import secrets
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from .fixed_reference import ExperimentContractError


@dataclass(frozen=True, slots=True)
class PublishedArtifact:
    path: Path
    sha256: str
    byte_count: int


class _BoundedSequentialRawWriter(io.RawIOBase):
    """Write-only descriptor adapter with an exact byte ceiling and rolling digest."""

    def __init__(self, descriptor: int, *, max_bytes: int) -> None:
        super().__init__()
        self._descriptor = descriptor
        self._max_bytes = max_bytes
        self._byte_count = 0
        self._digest = hashlib.sha256()

    @property
    def byte_count(self) -> int:
        return self._byte_count

    @property
    def sha256(self) -> str:
        return self._digest.hexdigest()

    def writable(self) -> bool:
        return True

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return False

    def fileno(self) -> int:
        self._checkClosed()
        return self._descriptor

    def write(self, value: bytes | bytearray | memoryview) -> int:
        self._checkClosed()
        view = memoryview(value).cast("B")
        if self._byte_count + len(view) > self._max_bytes:
            raise ExperimentContractError("streaming artifact exceeds its byte limit")
        written = 0
        while written < len(view):
            count = os.write(self._descriptor, view[written:])
            if count <= 0:
                raise ExperimentContractError("streaming artifact write made no progress")
            self._digest.update(view[written : written + count])
            self._byte_count += count
            written += count
        return written

    def flush(self) -> None:
        if not self.closed:
            os.fsync(self._descriptor)

    def close(self) -> None:
        if self.closed:
            return
        try:
            super().close()
        finally:
            os.close(self._descriptor)


@dataclass(slots=True)
class ReservedStreamingArtifact:
    """One bounded sequential stream retained on failure and published once."""

    path: Path
    partial_path: Path
    _parent_descriptor: int
    _artifact_descriptor: int
    _artifact_state: os.stat_result
    _raw_writer: _BoundedSequentialRawWriter
    writer: io.BufferedWriter
    _max_bytes: int
    _closed: bool = False

    def finalize(self) -> PublishedArtifact:
        """Seal, independently rehash, and atomically publish the stream."""

        if self._closed:
            raise ExperimentContractError("streaming artifact reservation is already closed")
        renamed = False
        expected_size = 0
        expected_sha256 = ""
        try:
            self.writer.flush()
            self.writer.close()
            expected_size = self._raw_writer.byte_count
            expected_sha256 = self._raw_writer.sha256
            if expected_size <= 0:
                raise ExperimentContractError("streaming artifact must be nonempty")
            if expected_size > self._max_bytes:
                raise ExperimentContractError("streaming artifact exceeds its byte limit")
            descriptor_state = os.fstat(self._artifact_descriptor)
            visible_partial = os.stat(
                self.partial_path.name,
                dir_fd=self._parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(descriptor_state.st_mode)
                or not _same_file(self._artifact_state, descriptor_state)
                or not _same_file(self._artifact_state, visible_partial)
                or descriptor_state.st_nlink != 1
                or visible_partial.st_nlink != 1
                or stat.S_IMODE(descriptor_state.st_mode) != 0o600
                or descriptor_state.st_size != expected_size
            ):
                raise ExperimentContractError("streaming artifact changed before publication")
            if (
                _descriptor_sha256(
                    self._artifact_descriptor,
                    expected_size=expected_size,
                )
                != expected_sha256
            ):
                raise ExperimentContractError(
                    "streaming artifact digest changed before publication"
                )
            verification_parent = _open_parent_directory(
                self.path.parent,
                create_missing=False,
            )
            try:
                if not _same_file(
                    os.fstat(self._parent_descriptor),
                    os.fstat(verification_parent),
                ):
                    raise ExperimentContractError(
                        "artifact output ancestors changed before publication"
                    )
            finally:
                os.close(verification_parent)
            try:
                _rename_without_overwrite(
                    self._parent_descriptor,
                    self.partial_path.name,
                    self.path.name,
                )
            except FileExistsError as exc:
                raise ExperimentContractError(
                    f"refusing to overwrite existing file: {self.path}"
                ) from exc
            renamed = True
            os.fsync(self._parent_descriptor)
            final_state = os.stat(
                self.path.name,
                dir_fd=self._parent_descriptor,
                follow_symlinks=False,
            )
            descriptor_state = os.fstat(self._artifact_descriptor)
            if (
                not _same_file(self._artifact_state, final_state)
                or not _same_file(self._artifact_state, descriptor_state)
                or descriptor_state.st_nlink != 1
                or stat.S_IMODE(descriptor_state.st_mode) != 0o600
                or descriptor_state.st_size != expected_size
                or _descriptor_sha256(
                    self._artifact_descriptor,
                    expected_size=expected_size,
                )
                != expected_sha256
            ):
                raise ExperimentContractError("streaming artifact changed during publication")
            verification_parent = _open_parent_directory(
                self.path.parent,
                create_missing=False,
            )
            try:
                if not _same_file(
                    os.fstat(self._parent_descriptor),
                    os.fstat(verification_parent),
                ):
                    raise ExperimentContractError(
                        "artifact output ancestors changed during publication"
                    )
                visible_final = os.stat(
                    self.path.name,
                    dir_fd=verification_parent,
                    follow_symlinks=False,
                )
                if not _same_file(self._artifact_state, visible_final):
                    raise ExperimentContractError(
                        "streaming artifact is not visible at the requested path"
                    )
            finally:
                os.close(verification_parent)
            return PublishedArtifact(
                path=self.path,
                sha256=expected_sha256,
                byte_count=expected_size,
            )
        except OSError as exc:
            retained = self.path if renamed else self.partial_path
            raise ExperimentContractError(
                f"cannot publish streaming artifact; retained at {retained}: {exc}"
            ) from exc
        finally:
            self.close()

    def close(self) -> None:
        """Close descriptors without deleting an incomplete artifact."""

        if self._closed:
            return
        with suppress(Exception):
            self.writer.close()
        os.close(self._artifact_descriptor)
        os.close(self._parent_descriptor)
        self._closed = True


@dataclass(slots=True)
class ReservedJsonArtifact:
    """One exclusively reserved output path held by open descriptors."""

    path: Path
    _parent_descriptor: int
    _reservation_descriptor: int
    _reservation_state: os.stat_result
    _closed: bool = False

    def finalize(self, value: object) -> PublishedArtifact:
        """Atomically exchange a complete JSON artifact into the reserved path."""

        if self._closed:
            raise ExperimentContractError("artifact reservation is already closed")
        try:
            encoded = finite_pretty_json(value)
        except Exception:
            self.close()
            raise
        expected_sha256 = hashlib.sha256(encoded).hexdigest()
        pending_descriptor: int | None = None
        pending_state: os.stat_result | None = None
        pending_name: str | None = None
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            visible_reservation = os.stat(
                self.path.name,
                dir_fd=self._parent_descriptor,
                follow_symlinks=False,
            )
            if not _same_file(self._reservation_state, visible_reservation):
                raise ExperimentContractError("reserved artifact path changed before finalization")
            for _attempt in range(100):
                pending_name = f".{self.path.name}.{secrets.token_hex(16)}.final"
                try:
                    pending_descriptor = os.open(
                        pending_name,
                        flags,
                        0o600,
                        dir_fd=self._parent_descriptor,
                    )
                except FileExistsError:
                    continue
                break
            if pending_descriptor is None or pending_name is None:
                raise ExperimentContractError("cannot allocate final artifact entry")
            pending_state = os.fstat(pending_descriptor)
            if not stat.S_ISREG(pending_state.st_mode):
                raise ExperimentContractError("final artifact entry is not a regular file")
            os.fchmod(pending_descriptor, 0o600)
            _write_descriptor_bytes(pending_descriptor, encoded)
            if (
                _descriptor_sha256(pending_descriptor, expected_size=len(encoded))
                != expected_sha256
            ):
                raise ExperimentContractError("final artifact bytes changed before exchange")
            visible_reservation = os.stat(
                self.path.name,
                dir_fd=self._parent_descriptor,
                follow_symlinks=False,
            )
            visible_pending = os.stat(
                pending_name,
                dir_fd=self._parent_descriptor,
                follow_symlinks=False,
            )
            if not _same_file(self._reservation_state, visible_reservation) or not _same_file(
                pending_state, visible_pending
            ):
                raise ExperimentContractError("artifact entries changed before final exchange")
            _exchange_entries(self._parent_descriptor, self.path.name, pending_name)
            final_state = os.stat(
                self.path.name,
                dir_fd=self._parent_descriptor,
                follow_symlinks=False,
            )
            retired_reservation = os.stat(
                pending_name,
                dir_fd=self._parent_descriptor,
                follow_symlinks=False,
            )
            if not _same_file(pending_state, final_state) or not _same_file(
                self._reservation_state, retired_reservation
            ):
                raise ExperimentContractError("artifact exchange produced unexpected identities")
            os.fsync(self._parent_descriptor)
            final_descriptor_state = os.fstat(pending_descriptor)
            if (
                final_descriptor_state.st_size != len(encoded)
                or final_descriptor_state.st_nlink != 1
                or stat.S_IMODE(final_descriptor_state.st_mode) != 0o600
                or _descriptor_sha256(pending_descriptor, expected_size=len(encoded))
                != expected_sha256
            ):
                raise ExperimentContractError("final artifact changed during finalization")
            verification_parent = _open_parent_directory(
                self.path.parent,
                create_missing=False,
            )
            try:
                if not _same_file(os.fstat(self._parent_descriptor), os.fstat(verification_parent)):
                    raise ExperimentContractError(
                        "artifact output ancestors changed during finalization"
                    )
                visible_final = os.stat(
                    self.path.name,
                    dir_fd=verification_parent,
                    follow_symlinks=False,
                )
                if not _same_file(pending_state, visible_final):
                    raise ExperimentContractError(
                        "final artifact is not visible at the requested path"
                    )
            finally:
                os.close(verification_parent)
            os.unlink(pending_name, dir_fd=self._parent_descriptor)
            pending_name = None
            return PublishedArtifact(
                path=self.path,
                sha256=expected_sha256,
                byte_count=len(encoded),
            )
        except OSError as exc:
            raise ExperimentContractError(
                f"cannot finalize reserved artifact {self.path}: {exc}"
            ) from exc
        finally:
            if pending_name is not None and pending_state is not None:
                try:
                    final_entry = os.stat(
                        self.path.name,
                        dir_fd=self._parent_descriptor,
                        follow_symlinks=False,
                    )
                    pending_entry = os.stat(
                        pending_name,
                        dir_fd=self._parent_descriptor,
                        follow_symlinks=False,
                    )
                    if _same_file(pending_state, final_entry) and _same_file(
                        self._reservation_state, pending_entry
                    ):
                        _exchange_entries(
                            self._parent_descriptor,
                            self.path.name,
                            pending_name,
                        )
                        final_entry = os.stat(
                            self.path.name,
                            dir_fd=self._parent_descriptor,
                            follow_symlinks=False,
                        )
                        pending_entry = os.stat(
                            pending_name,
                            dir_fd=self._parent_descriptor,
                            follow_symlinks=False,
                        )
                    if _same_file(self._reservation_state, final_entry) and _same_file(
                        pending_state, pending_entry
                    ):
                        os.unlink(pending_name, dir_fd=self._parent_descriptor)
                    elif _same_file(self._reservation_state, pending_entry):
                        recovery_name = f".{self.path.name}.in-progress-recovery"
                        with suppress(ExperimentContractError, OSError):
                            _rename_without_overwrite(
                                self._parent_descriptor,
                                pending_name,
                                recovery_name,
                            )
                except (ExperimentContractError, OSError):
                    pass
            if pending_descriptor is not None:
                os.close(pending_descriptor)
            self.close()

    def close(self) -> None:
        """Close reservation descriptors while leaving its visible receipt intact."""

        if not self._closed:
            os.close(self._reservation_descriptor)
            os.close(self._parent_descriptor)
            self._closed = True


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


def _write_descriptor_bytes(descriptor: int, encoded: bytes) -> None:
    os.ftruncate(descriptor, 0)
    os.lseek(descriptor, 0, os.SEEK_SET)
    view = memoryview(encoded)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise ExperimentContractError("artifact write made no progress")
        written += count
    os.fsync(descriptor)


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


def _exchange_entries(parent_descriptor: int, left: str, right: str) -> None:
    """Atomically swap two existing names in one descriptor-bound directory."""

    library = ctypes.CDLL(None, use_errno=True)
    left_bytes = os.fsencode(left)
    right_bytes = os.fsencode(right)
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
        flags = 0x00000002 | 0x00000010 | 0x00000020
    elif sys.platform.startswith("linux"):
        try:
            rename = library.renameat2
        except AttributeError as exc:
            raise ExperimentContractError("this platform lacks atomic artifact exchange") from exc
        rename.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        rename.restype = ctypes.c_int
        flags = 2
    else:
        raise ExperimentContractError("this platform lacks atomic artifact exchange")
    ctypes.set_errno(0)
    result = rename(
        parent_descriptor,
        left_bytes,
        parent_descriptor,
        right_bytes,
        flags,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    raise OSError(error_number, os.strerror(error_number), left, right)


def reserve_json_artifact(path: Path, provisional_value: object) -> ReservedJsonArtifact:
    """Reserve the final name and publish a valid provisional JSON receipt."""

    encoded = finite_pretty_json(provisional_value)
    requested = Path(path)
    if requested.name in {"", ".", ".."} or "\0" in requested.name:
        raise ExperimentContractError("reserved artifact must have one filename")
    absolute = Path(os.path.abspath(requested))
    parent_descriptor = _open_parent_directory(absolute.parent)
    file_descriptor: int | None = None
    created: os.stat_result | None = None
    reserved = False
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        file_descriptor = os.open(
            absolute.name,
            flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        created = os.fstat(file_descriptor)
        if not stat.S_ISREG(created.st_mode):
            raise ExperimentContractError("reserved artifact is not a regular file")
        os.fchmod(file_descriptor, 0o600)
        _write_descriptor_bytes(file_descriptor, encoded)
        expected_sha256 = hashlib.sha256(encoded).hexdigest()
        visible = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not _same_file(created, visible)
            or visible.st_size != len(encoded)
            or visible.st_nlink != 1
            or _descriptor_sha256(file_descriptor, expected_size=len(encoded)) != expected_sha256
        ):
            raise ExperimentContractError("artifact reservation changed during creation")
        os.fsync(parent_descriptor)
        reservation = ReservedJsonArtifact(
            path=absolute,
            _parent_descriptor=parent_descriptor,
            _reservation_descriptor=file_descriptor,
            _reservation_state=created,
        )
        reserved = True
        return reservation
    except FileExistsError as exc:
        raise ExperimentContractError(f"refusing to overwrite existing file: {absolute}") from exc
    except OSError as exc:
        raise ExperimentContractError(f"cannot reserve artifact {absolute}: {exc}") from exc
    finally:
        if not reserved:
            cleanup_state = created
            if cleanup_state is None and file_descriptor is not None:
                with suppress(OSError):
                    cleanup_state = os.fstat(file_descriptor)
            if cleanup_state is not None:
                try:
                    visible = os.stat(
                        absolute.name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                    if _same_file(cleanup_state, visible):
                        os.unlink(absolute.name, dir_fd=parent_descriptor)
                except OSError:
                    pass
            if file_descriptor is not None:
                os.close(file_descriptor)
            os.close(parent_descriptor)


def reserve_streaming_artifact(
    path: Path,
    *,
    max_bytes: int,
    buffer_size: int = 1024 * 1024,
) -> ReservedStreamingArtifact:
    """Reserve one hidden partial for bounded sequential publication."""

    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ExperimentContractError("streaming artifact max_bytes must be a positive integer")
    if isinstance(buffer_size, bool) or not isinstance(buffer_size, int) or buffer_size <= 0:
        raise ExperimentContractError("streaming artifact buffer_size must be a positive integer")
    requested = Path(path)
    if requested.name in {"", ".", ".."} or "\0" in requested.name:
        raise ExperimentContractError("streaming artifact must have one filename")
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

    artifact_descriptor: int | None = None
    stream_descriptor: int | None = None
    created: os.stat_result | None = None
    partial_name: str | None = None
    raw_writer: _BoundedSequentialRawWriter | None = None
    writer: io.BufferedWriter | None = None
    reserved = False
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        for _attempt in range(100):
            partial_name = f".{absolute.name}.{secrets.token_hex(16)}.partial"
            try:
                artifact_descriptor = os.open(
                    partial_name,
                    flags,
                    0o600,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                continue
            break
        if artifact_descriptor is None or partial_name is None:
            raise ExperimentContractError("cannot allocate a unique streaming partial")
        created = os.fstat(artifact_descriptor)
        if not stat.S_ISREG(created.st_mode):
            raise ExperimentContractError("streaming artifact is not a regular file")
        os.fchmod(artifact_descriptor, 0o600)
        visible = os.stat(partial_name, dir_fd=parent_descriptor, follow_symlinks=False)
        if (
            not _same_file(created, visible)
            or visible.st_nlink != 1
            or visible.st_size != 0
            or stat.S_IMODE(visible.st_mode) != 0o600
        ):
            raise ExperimentContractError("streaming artifact changed during reservation")
        stream_descriptor = os.dup(artifact_descriptor)
        raw_writer = _BoundedSequentialRawWriter(
            stream_descriptor,
            max_bytes=max_bytes,
        )
        stream_descriptor = None
        writer = io.BufferedWriter(raw_writer, buffer_size=buffer_size)
        os.fsync(parent_descriptor)
        reservation = ReservedStreamingArtifact(
            path=absolute,
            partial_path=absolute.parent / partial_name,
            _parent_descriptor=parent_descriptor,
            _artifact_descriptor=artifact_descriptor,
            _artifact_state=created,
            _raw_writer=raw_writer,
            writer=writer,
            _max_bytes=max_bytes,
        )
        reserved = True
        return reservation
    except OSError as exc:
        raise ExperimentContractError(
            f"cannot reserve streaming artifact {absolute}: {exc}"
        ) from exc
    finally:
        if not reserved:
            if writer is not None:
                with suppress(Exception):
                    writer.close()
            elif raw_writer is not None:
                with suppress(Exception):
                    raw_writer.close()
            elif stream_descriptor is not None:
                os.close(stream_descriptor)
            if created is not None and partial_name is not None:
                try:
                    visible = os.stat(
                        partial_name,
                        dir_fd=parent_descriptor,
                        follow_symlinks=False,
                    )
                    if _same_file(created, visible):
                        os.unlink(partial_name, dir_fd=parent_descriptor)
                except OSError:
                    pass
            if artifact_descriptor is not None:
                os.close(artifact_descriptor)
            os.close(parent_descriptor)


@contextmanager
def verified_artifact_reader(
    path: Path,
    *,
    expected_sha256: str,
    expected_size: int,
    max_bytes: int,
    required_mode: int = 0o600,
    buffer_size: int = 1024 * 1024,
) -> Iterator[io.BufferedReader]:
    """Yield a descriptor-bound reader and verify identity before and after use."""

    if (
        len(expected_sha256) != 64
        or expected_sha256.lower() != expected_sha256
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ExperimentContractError("expected artifact SHA-256 is invalid")
    for label, value in {
        "expected_size": expected_size,
        "max_bytes": max_bytes,
        "buffer_size": buffer_size,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ExperimentContractError(f"{label} must be a positive integer")
    if expected_size > max_bytes:
        raise ExperimentContractError("expected artifact size exceeds its byte limit")
    if (
        isinstance(required_mode, bool)
        or not isinstance(required_mode, int)
        or required_mode < 0
        or required_mode > 0o777
    ):
        raise ExperimentContractError("required_mode must be a permission mode")
    requested = Path(path)
    if requested.name in {"", ".", ".."} or "\0" in requested.name:
        raise ExperimentContractError("verified artifact must have one filename")
    absolute = Path(os.path.abspath(requested))
    parent_descriptor = _open_parent_directory(absolute.parent, create_missing=False)
    verification_descriptor: int | None = None
    reader_descriptor: int | None = None
    reader: io.BufferedReader | None = None
    try:
        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        verification_descriptor = os.open(
            absolute.name,
            flags,
            dir_fd=parent_descriptor,
        )
        reader_descriptor = os.open(
            absolute.name,
            flags,
            dir_fd=parent_descriptor,
        )
        descriptor_state = os.fstat(verification_descriptor)
        reader_state = os.fstat(reader_descriptor)
        visible_state = os.stat(
            absolute.name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISREG(descriptor_state.st_mode)
            or not _same_file(descriptor_state, reader_state)
            or not _same_file(descriptor_state, visible_state)
            or descriptor_state.st_nlink != 1
            or stat.S_IMODE(descriptor_state.st_mode) != required_mode
            or descriptor_state.st_size != expected_size
        ):
            raise ExperimentContractError("verified artifact metadata differs")
        if (
            _descriptor_sha256(
                verification_descriptor,
                expected_size=expected_size,
            )
            != expected_sha256
        ):
            raise ExperimentContractError("verified artifact SHA-256 differs")
        raw_reader = io.FileIO(reader_descriptor, mode="rb", closefd=True)
        reader_descriptor = None
        reader = io.BufferedReader(raw_reader, buffer_size=buffer_size)
        try:
            yield reader
        finally:
            reader.close()
            reader = None
            descriptor_state_after = os.fstat(verification_descriptor)
            visible_state_after = os.stat(
                absolute.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if (
                not _same_file(descriptor_state, descriptor_state_after)
                or not _same_file(descriptor_state, visible_state_after)
                or descriptor_state_after.st_nlink != 1
                or descriptor_state_after.st_size != expected_size
                or stat.S_IMODE(descriptor_state_after.st_mode) != required_mode
                or _descriptor_sha256(
                    verification_descriptor,
                    expected_size=expected_size,
                )
                != expected_sha256
            ):
                raise ExperimentContractError("verified artifact changed while in use")
            verification_parent = _open_parent_directory(
                absolute.parent,
                create_missing=False,
            )
            try:
                if not _same_file(
                    os.fstat(parent_descriptor),
                    os.fstat(verification_parent),
                ):
                    raise ExperimentContractError(
                        "verified artifact ancestors changed while in use"
                    )
            finally:
                os.close(verification_parent)
    except OSError as exc:
        raise ExperimentContractError(f"cannot read verified artifact {absolute}: {exc}") from exc
    finally:
        if reader is not None:
            reader.close()
        if reader_descriptor is not None:
            os.close(reader_descriptor)
        if verification_descriptor is not None:
            os.close(verification_descriptor)
        os.close(parent_descriptor)


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
    "ReservedJsonArtifact",
    "ReservedStreamingArtifact",
    "finite_pretty_json",
    "publish_bytes_without_overwrite",
    "publish_json_without_overwrite",
    "reserve_json_artifact",
    "reserve_streaming_artifact",
    "verified_artifact_reader",
]
