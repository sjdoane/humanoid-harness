"""Bounded canonical-JSON transport for the TQC v2 parent and worker."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import json
import math
import os
import select
import stat
import struct
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Literal

from .fixed_reference import ExperimentContractError
from .tqc_calibration_contract import canonical_json

MESSAGE_SCHEMA_VERSION = 1
MAX_MESSAGE_BYTES = 65_536
FRAME_HEADER_BYTES = 8
ATTEMPT_ID = "dev1m-v2-seed-95001-attempt-01"
STAGES = (
    "preflight",
    "model_constructed",
    "training",
    "training_complete",
    "persistence",
    "strict_reload",
    "evaluation",
    "visual_encoding",
    "finalized",
)
WORKER_TO_PARENT_TYPES = frozenset(
    {
        "worker_started",
        "stage_entered",
        "monitor_sample",
        "stage_completed",
        "preflight_receipts",
        "execution_manifest_acknowledged",
        "failure",
        "worker_complete",
    }
)
PARENT_TO_WORKER_TYPES = frozenset({"admit_execution_manifest", "request_shutdown"})
MESSAGE_KEYS = {
    "schema_version",
    "sequence_index",
    "attempt_id",
    "worker_pid",
    "monotonic_seconds",
    "message_type",
    "stage",
    "payload",
}

_DescriptorIdentity = tuple[int, int, int, int]


def _descriptor_identity(descriptor: int, *, expected_access: int) -> _DescriptorIdentity:
    try:
        status = os.fstat(descriptor)
        access = fcntl.fcntl(descriptor, fcntl.F_GETFL) & os.O_ACCMODE
    except OSError as exc:
        raise ExperimentContractError("channel descriptor is unavailable") from exc
    if not stat.S_ISFIFO(status.st_mode) or access != expected_access:
        raise ExperimentContractError(
            "channel descriptors must be direction-correct pipe endpoints"
        )
    return (int(status.st_dev), int(status.st_ino), stat.S_IFMT(status.st_mode), access)


def _darwin_pipe_pair_identity(descriptor: int, status: os.stat_result) -> tuple[int, int]:
    # Darwin's pipe_fdinfo stores vinfo_stat followed by the two linked handles.
    prefix_bytes = 24 + 136
    receipt_bytes = prefix_bytes + 16 + 8
    buffer = ctypes.create_string_buffer(receipt_bytes)
    try:
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pidfdinfo = libproc.proc_pidfdinfo
        proc_pidfdinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        proc_pidfdinfo.restype = ctypes.c_int
        observed_bytes = proc_pidfdinfo(
            os.getpid(), descriptor, 6, ctypes.byref(buffer), receipt_bytes
        )
    except (AttributeError, OSError) as exc:
        raise ExperimentContractError("cannot inspect Darwin pipe topology") from exc
    if observed_bytes != receipt_bytes:
        raise ExperimentContractError("Darwin pipe topology receipt is incomplete")
    handle, peer_handle = struct.unpack_from("=QQ", buffer.raw, prefix_bytes)
    if handle == 0 or peer_handle == 0 or handle != int(status.st_ino):
        raise ExperimentContractError("Darwin pipe topology identity differs")
    return tuple(sorted((handle, peer_handle)))


def _pipe_pair_identity(descriptor: int) -> tuple[int, ...]:
    try:
        status = os.fstat(descriptor)
    except OSError as exc:
        raise ExperimentContractError("channel descriptor is unavailable") from exc
    if sys.platform == "darwin":
        return _darwin_pipe_pair_identity(descriptor, status)
    if sys.platform.startswith("linux"):
        return (int(status.st_dev), int(status.st_ino))
    raise ExperimentContractError("pipe topology inspection is unsupported on this platform")


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExperimentContractError(f"worker message contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ExperimentContractError(f"worker message contains non-finite number {value!r}")


def _read_exact(
    descriptor: int,
    size: int,
    *,
    deadline: float,
    expected_identity: _DescriptorIdentity,
) -> bytes:
    chunks: list[bytes] = []
    observed = 0
    while observed < size:
        if _descriptor_identity(descriptor, expected_access=os.O_RDONLY) != expected_identity:
            raise ExperimentContractError("channel read descriptor identity changed")
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise ExperimentContractError("parent-worker message deadline expired")
        readable, _writable, _errors = select.select([descriptor], [], [], remaining)
        if not readable:
            raise ExperimentContractError("parent-worker message deadline expired")
        try:
            if _descriptor_identity(descriptor, expected_access=os.O_RDONLY) != expected_identity:
                raise ExperimentContractError("channel read descriptor identity changed")
            chunk = os.read(descriptor, min(size - observed, 64 * 1024))
        except BlockingIOError:
            continue
        except InterruptedError:
            continue
        if not chunk:
            raise ExperimentContractError("parent-worker message pipe closed early")
        chunks.append(chunk)
        observed += len(chunk)
    return b"".join(chunks)


def _write_exact(
    descriptor: int,
    payload: bytes,
    *,
    deadline: float,
    expected_identity: _DescriptorIdentity,
) -> None:
    view = memoryview(payload)
    written = 0
    while written < len(view):
        if _descriptor_identity(descriptor, expected_access=os.O_WRONLY) != expected_identity:
            raise ExperimentContractError("channel write descriptor identity changed")
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise ExperimentContractError("parent-worker message deadline expired")
        _readable, writable, _errors = select.select([], [descriptor], [], remaining)
        if not writable:
            raise ExperimentContractError("parent-worker message deadline expired")
        try:
            if _descriptor_identity(descriptor, expected_access=os.O_WRONLY) != expected_identity:
                raise ExperimentContractError("channel write descriptor identity changed")
            count = os.write(descriptor, view[written:])
        except BlockingIOError:
            continue
        except InterruptedError:
            continue
        except BrokenPipeError as exc:
            raise ExperimentContractError("parent-worker message pipe is closed") from exc
        if count <= 0:
            raise ExperimentContractError("parent-worker message write made no progress")
        written += count


def _validate_message(
    value: object,
    *,
    expected_sequence: int,
    expected_attempt_id: str,
    expected_worker_pid: int,
    allowed_types: frozenset[str],
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != MESSAGE_KEYS:
        raise ExperimentContractError("parent-worker message keys differ")
    if (
        value["schema_version"] != MESSAGE_SCHEMA_VERSION
        or type(value["schema_version"]) is not int
    ):
        raise ExperimentContractError("parent-worker message schema differs")
    if type(value["sequence_index"]) is not int or value["sequence_index"] != expected_sequence:
        raise ExperimentContractError("parent-worker message sequence is not contiguous")
    if type(value["attempt_id"]) is not str or value["attempt_id"] != expected_attempt_id:
        raise ExperimentContractError("parent-worker message attempt differs")
    if type(value["worker_pid"]) is not int or value["worker_pid"] != expected_worker_pid:
        raise ExperimentContractError("parent-worker message worker PID differs")
    monotonic = value["monotonic_seconds"]
    if type(monotonic) is not float or not math.isfinite(monotonic) or monotonic < 0:
        raise ExperimentContractError("parent-worker message monotonic time is invalid")
    message_type = value["message_type"]
    if type(message_type) is not str or message_type not in allowed_types:
        raise ExperimentContractError("parent-worker message type is not allowed")
    if type(value["stage"]) is not str or value["stage"] not in STAGES:
        raise ExperimentContractError("parent-worker message stage is invalid")
    if type(value["payload"]) is not dict:
        raise ExperimentContractError("parent-worker message payload must be an object")
    return value


@dataclass(slots=True)
class TQCWorkerChannelV2:
    """Own pipe duplicates with independent send and receive counters."""

    read_descriptor: int
    write_descriptor: int
    role: Literal["parent", "worker"]
    worker_pid: int
    attempt_id: str = ATTEMPT_ID
    _send_sequence: int = field(default=0, init=False)
    _receive_sequence: int = field(default=0, init=False)
    _closed: bool = field(default=False, init=False)
    _failed_reason: str | None = field(default=None, init=False)
    _transcript: Any = field(default_factory=hashlib.sha256, init=False, repr=False)
    _transcript_frame_count: int = field(default=0, init=False)
    _read_identity: _DescriptorIdentity = field(init=False, repr=False)
    _write_identity: _DescriptorIdentity = field(init=False, repr=False)
    _bound_role: Literal["parent", "worker"] = field(init=False, repr=False)
    _bound_worker_pid: int = field(init=False, repr=False)
    _bound_attempt_id: str = field(init=False, repr=False)
    _creator_pid: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.role not in {"parent", "worker"}:
            raise ExperimentContractError("channel role must be parent or worker")
        if type(self.worker_pid) is not int or self.worker_pid <= 1:
            raise ExperimentContractError("channel worker PID must be a positive process PID")
        if self.role == "worker" and self.worker_pid != os.getpid():
            raise ExperimentContractError("worker channel PID must equal the current process PID")
        if self.attempt_id != ATTEMPT_ID or type(self.attempt_id) is not str:
            raise ExperimentContractError("channel attempt id differs")
        if (
            type(self.read_descriptor) is not int
            or type(self.write_descriptor) is not int
            or self.read_descriptor < 0
            or self.write_descriptor < 0
            or self.read_descriptor == self.write_descriptor
        ):
            raise ExperimentContractError("channel descriptors are invalid")
        input_read_identity = _descriptor_identity(
            self.read_descriptor, expected_access=os.O_RDONLY
        )
        input_write_identity = _descriptor_identity(
            self.write_descriptor, expected_access=os.O_WRONLY
        )
        if _pipe_pair_identity(self.read_descriptor) == _pipe_pair_identity(self.write_descriptor):
            raise ExperimentContractError("channel requires two distinct dedicated pipes")

        owned: list[int] = []
        try:
            owned_read = os.dup(self.read_descriptor)
            owned.append(owned_read)
            owned_write = os.dup(self.write_descriptor)
            owned.append(owned_write)
            os.set_inheritable(owned_read, False)
            os.set_inheritable(owned_write, False)
            os.set_blocking(owned_read, False)
            os.set_blocking(owned_write, False)
            read_identity = _descriptor_identity(owned_read, expected_access=os.O_RDONLY)
            write_identity = _descriptor_identity(owned_write, expected_access=os.O_WRONLY)
            if (
                read_identity != input_read_identity
                or write_identity != input_write_identity
                or _pipe_pair_identity(owned_read) == _pipe_pair_identity(owned_write)
            ):
                raise ExperimentContractError(
                    "channel descriptor identity changed during ownership"
                )
        except BaseException:
            for descriptor in owned:
                with suppress(OSError):
                    os.close(descriptor)
            raise
        self.read_descriptor = owned_read
        self.write_descriptor = owned_write
        self._read_identity = read_identity
        self._write_identity = write_identity
        self._bound_role = self.role
        self._bound_worker_pid = self.worker_pid
        self._bound_attempt_id = self.attempt_id
        self._creator_pid = os.getpid()

    @property
    def send_sequence(self) -> int:
        return self._send_sequence

    @property
    def receive_sequence(self) -> int:
        return self._receive_sequence

    @property
    def failed_reason(self) -> str | None:
        return self._failed_reason

    @property
    def transcript_sha256(self) -> str:
        return self._transcript.hexdigest()

    @property
    def transcript_frame_count(self) -> int:
        return self._transcript_frame_count

    def _assert_active(self) -> None:
        if (
            os.getpid() != self._creator_pid
            or self.role != self._bound_role
            or self.worker_pid != self._bound_worker_pid
            or self.attempt_id != self._bound_attempt_id
        ):
            raise ExperimentContractError("parent-worker channel process or identity changed")
        if self._failed_reason is not None:
            raise ExperimentContractError(
                f"parent-worker channel is permanently failed: {self._failed_reason}"
            )
        if self._closed:
            raise ExperimentContractError("parent-worker channel is closed")

    def _fail_closed(self, exc: BaseException) -> None:
        if self._failed_reason is None:
            self._failed_reason = f"{type(exc).__name__}:{exc}"
        self._close_descriptors(suppress_errors=True)
        self._closed = True

    def _record_frame(self, *, flow: str, encoded: bytes) -> None:
        flow_bytes = flow.encode("ascii")
        self._transcript.update(len(flow_bytes).to_bytes(8, "big"))
        self._transcript.update(flow_bytes)
        self._transcript.update(len(encoded).to_bytes(8, "big"))
        self._transcript.update(encoded)
        self._transcript_frame_count += 1

    def send(
        self,
        message_type: str,
        stage: str,
        payload: dict[str, object],
        *,
        deadline: float,
    ) -> None:
        """Send one bounded canonical frame and advance only after full write."""

        self._assert_active()
        try:
            allowed = PARENT_TO_WORKER_TYPES if self.role == "parent" else WORKER_TO_PARENT_TYPES
            now = time.perf_counter()
            message = {
                "schema_version": MESSAGE_SCHEMA_VERSION,
                "sequence_index": self._send_sequence,
                "attempt_id": self.attempt_id,
                "worker_pid": self.worker_pid,
                "monotonic_seconds": float(now),
                "message_type": message_type,
                "stage": stage,
                "payload": payload,
            }
            _validate_message(
                message,
                expected_sequence=self._send_sequence,
                expected_attempt_id=self.attempt_id,
                expected_worker_pid=self.worker_pid,
                allowed_types=allowed,
            )
            encoded = canonical_json(message)
            if not encoded or len(encoded) > MAX_MESSAGE_BYTES:
                raise ExperimentContractError("parent-worker message exceeds its byte limit")
            frame = len(encoded).to_bytes(FRAME_HEADER_BYTES, "big") + encoded
            _write_exact(
                self.write_descriptor,
                frame,
                deadline=deadline,
                expected_identity=self._write_identity,
            )
            flow = "parent_to_worker" if self.role == "parent" else "worker_to_parent"
            self._record_frame(flow=flow, encoded=encoded)
            self._send_sequence += 1
        except BaseException as exc:
            self._fail_closed(exc)
            raise

    def receive(self, *, deadline: float) -> dict[str, Any]:
        """Receive one exact canonical frame and advance after full validation."""

        try:
            self._assert_active()
            header = _read_exact(
                self.read_descriptor,
                FRAME_HEADER_BYTES,
                deadline=deadline,
                expected_identity=self._read_identity,
            )
            size = int.from_bytes(header, "big")
            if size <= 0 or size > MAX_MESSAGE_BYTES:
                raise ExperimentContractError("parent-worker message length is invalid")
            encoded = _read_exact(
                self.read_descriptor,
                size,
                deadline=deadline,
                expected_identity=self._read_identity,
            )
            try:
                value = json.loads(
                    encoded.decode("utf-8", errors="strict"),
                    object_pairs_hook=_duplicate_rejecting_object,
                    parse_constant=_reject_nonfinite,
                )
            except ExperimentContractError:
                raise
            except (UnicodeError, ValueError, RecursionError) as exc:
                raise ExperimentContractError("parent-worker message is invalid JSON") from exc
            if canonical_json(value) != encoded:
                raise ExperimentContractError("parent-worker message is not canonical JSON")
            allowed = WORKER_TO_PARENT_TYPES if self.role == "parent" else PARENT_TO_WORKER_TYPES
            result = _validate_message(
                value,
                expected_sequence=self._receive_sequence,
                expected_attempt_id=self.attempt_id,
                expected_worker_pid=self.worker_pid,
                allowed_types=allowed,
            )
            flow = "worker_to_parent" if self.role == "parent" else "parent_to_worker"
            self._record_frame(flow=flow, encoded=encoded)
            self._receive_sequence += 1
            return result
        except BaseException as exc:
            self._fail_closed(exc)
            raise

    def _close_descriptors(self, *, suppress_errors: bool) -> None:
        first_error: BaseException | None = None
        for descriptor, expected_access, expected_identity in (
            (self.read_descriptor, os.O_RDONLY, self._read_identity),
            (self.write_descriptor, os.O_WRONLY, self._write_identity),
        ):
            try:
                if (
                    _descriptor_identity(descriptor, expected_access=expected_access)
                    != expected_identity
                ):
                    raise ExperimentContractError(
                        "channel descriptor identity changed before close"
                    )
                os.close(descriptor)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        if first_error is not None and not suppress_errors:
            raise first_error

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._close_descriptors(suppress_errors=False)
        finally:
            self._closed = True


__all__ = [
    "ATTEMPT_ID",
    "MAX_MESSAGE_BYTES",
    "MESSAGE_SCHEMA_VERSION",
    "PARENT_TO_WORKER_TYPES",
    "STAGES",
    "WORKER_TO_PARENT_TYPES",
    "TQCWorkerChannelV2",
]
