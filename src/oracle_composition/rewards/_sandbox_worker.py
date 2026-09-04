"""Private pipe worker for one immutable task-term snapshot."""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import math
import os
import resource
import struct
import sys

from .contract import CandidateTaskInputsV1
from .static_validation import validate_task_term_source

SOURCE_FRAME_MAX_BYTES = 16 * 1024 + 36
REQUEST_FRAME_MAX_BYTES = 1024
RESPONSE_FRAME_MAX_BYTES = 64
SOURCE_MAGIC = b"RWS1"
BATCH_OPCODE = 0x42
VALUES_OPCODE = 0x56
QUIT_OPCODE = 0x51
READY_OPCODE = 0x52
ERROR_OPCODE = 0x45
LIMIT_STATUS_OPCODE = 0x4C
LIMIT_STATUS_OK = 0
LIMIT_STATUS_ADDRESS_SPACE = 1
LIMIT_STATUS_CPU = 2
LIMIT_STATUS_FILE_SIZE = 3
LIMIT_STATUS_CORE_SIZE = 4
LIMIT_STATUS_OPEN_FILES = 5
LIMIT_STATUS_INTERNAL = 255
ADDRESS_SPACE_ENFORCED = 1
ADDRESS_SPACE_DARWIN_REFUSED = 2
ADDRESS_SPACE_FAILED = 3
ADDRESS_SPACE_LIMIT_BYTES = 1024**3
CPU_LIMIT_SECONDS = 900
FILE_SIZE_LIMIT_BYTES = 0
CORE_SIZE_LIMIT_BYTES = 0
OPEN_FILE_LIMIT = 16
MEMORY_CANARY_BLOCK_BYTES = 1024**2
LIBPROC_PATH = "/usr/lib/libproc.dylib"
PROC_PIDTBSDINFO = 3
PROC_PIDINFO_BUFFER_BYTES = 1024
EXIT_OK = 0
EXIT_ARGUMENT_PARSING_FAILED = 2
EXIT_CANARY_NOT_CONTAINED = 4
EXIT_CANARY_INVOCATION_INVALID = 5
EXIT_REQUIRED_DESCRIPTOR_MISSING = 64
EXIT_ENVIRONMENT_INVALID = 65
EXIT_LIMIT_APPLICATION_FAILED = 66
EXIT_LIMIT_STATUS_INTERNAL = 67
EXIT_SOURCE_BOOTSTRAP_FAILED = 70
EXIT_REQUEST_OR_EVALUATION_FAILED = 71
EXIT_INTENTIONAL_CRASH_CANARY = 86
WORKER_EXIT_CODE_MEANINGS = {
    EXIT_OK: "clean_exit_or_os_canary_operation_denied",
    EXIT_ARGUMENT_PARSING_FAILED: "command_line_argument_parsing_failed",
    EXIT_CANARY_NOT_CONTAINED: "os_canary_operation_was_not_denied",
    EXIT_CANARY_INVOCATION_INVALID: "os_canary_invocation_missing_context_or_unknown_kind",
    EXIT_REQUIRED_DESCRIPTOR_MISSING: "required_invocation_file_descriptor_missing",
    EXIT_ENVIRONMENT_INVALID: "environment_sanitization_failed",
    EXIT_LIMIT_APPLICATION_FAILED: "declared_resource_limit_application_failed",
    EXIT_LIMIT_STATUS_INTERNAL: "unexpected_internal_resource_limit_setup_failure",
    EXIT_SOURCE_BOOTSTRAP_FAILED: "source_bootstrap_or_static_validation_failed",
    EXIT_REQUEST_OR_EVALUATION_FAILED: "request_frame_or_task_evaluation_failed",
    EXIT_INTENTIONAL_CRASH_CANARY: "intentional_worker_crash_canary",
}
WORKER_ENVIRONMENT = {
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
}


class WorkerLimitError(RuntimeError):
    """A categorical failure applying one mandatory worker resource limit."""

    def __init__(self, limit_code: int, address_space_status: int) -> None:
        super().__init__(f"worker limit application failed with category {limit_code}")
        self.limit_code = limit_code
        self.address_space_status = address_space_status


def _apply_worker_limits() -> int:
    """Apply limits after exec and return the categorical address-space result."""

    os.umask(0o077)
    try:
        inherited_address_space = resource.getrlimit(resource.RLIMIT_AS)
    except (ValueError, OSError) as exc:
        raise WorkerLimitError(
            LIMIT_STATUS_ADDRESS_SPACE,
            ADDRESS_SPACE_FAILED,
        ) from exc
    try:
        resource.setrlimit(
            resource.RLIMIT_AS,
            (ADDRESS_SPACE_LIMIT_BYTES, ADDRESS_SPACE_LIMIT_BYTES),
        )
        address_space_status = ADDRESS_SPACE_ENFORCED
    except (ValueError, OSError) as exc:
        if sys.platform != "darwin":
            raise WorkerLimitError(
                LIMIT_STATUS_ADDRESS_SPACE,
                ADDRESS_SPACE_FAILED,
            ) from exc
        try:
            unchanged = resource.getrlimit(resource.RLIMIT_AS) == inherited_address_space
        except (ValueError, OSError) as inspection_error:
            raise WorkerLimitError(
                LIMIT_STATUS_ADDRESS_SPACE,
                ADDRESS_SPACE_FAILED,
            ) from inspection_error
        if not unchanged:
            raise WorkerLimitError(
                LIMIT_STATUS_ADDRESS_SPACE,
                ADDRESS_SPACE_FAILED,
            ) from exc
        address_space_status = ADDRESS_SPACE_DARWIN_REFUSED

    mandatory_limits = (
        (LIMIT_STATUS_CPU, resource.RLIMIT_CPU, CPU_LIMIT_SECONDS),
        (LIMIT_STATUS_FILE_SIZE, resource.RLIMIT_FSIZE, FILE_SIZE_LIMIT_BYTES),
        (LIMIT_STATUS_CORE_SIZE, resource.RLIMIT_CORE, CORE_SIZE_LIMIT_BYTES),
        (LIMIT_STATUS_OPEN_FILES, resource.RLIMIT_NOFILE, OPEN_FILE_LIMIT),
    )
    for limit_code, resource_id, value in mandatory_limits:
        try:
            resource.setrlimit(resource_id, (value, value))
        except (ValueError, OSError) as exc:
            raise WorkerLimitError(limit_code, address_space_status) from exc
    return address_space_status


def _sanitize_environment() -> None:
    for name in tuple(os.environ):
        if name not in WORKER_ENVIRONMENT:
            del os.environ[name]
    if dict(os.environ) != WORKER_ENVIRONMENT:
        raise RuntimeError("worker environment differs from its fixed allowlist")


def _read_exact(descriptor: int, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            raise EOFError("pipe closed inside a frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame(descriptor: int, *, maximum_bytes: int) -> bytes:
    header = _read_exact(descriptor, 4)
    length = int.from_bytes(header, "big")
    if length < 1 or length > maximum_bytes:
        raise ValueError("frame length is outside its fixed bound")
    return _read_exact(descriptor, length)


def _write_all(descriptor: int, value: bytes) -> None:
    view = memoryview(value)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise OSError("pipe write made no progress")
        written += count


def _write_frame(descriptor: int, payload: bytes) -> None:
    if not 0 < len(payload) <= RESPONSE_FRAME_MAX_BYTES:
        raise ValueError("response frame is outside its fixed bound")
    _write_all(descriptor, len(payload).to_bytes(4, "big") + payload)


def _error(descriptor: int, code: int) -> None:
    with contextlib.suppress(BaseException):
        _write_frame(descriptor, bytes((ERROR_OPCODE, code & 0xFF)))


def _write_limit_status(
    descriptor: int,
    *,
    address_space_status: int,
    limit_error: int = LIMIT_STATUS_OK,
) -> None:
    _write_frame(
        descriptor,
        bytes((LIMIT_STATUS_OPCODE, address_space_status, limit_error)),
    )


def _run_worker(source_descriptor: int, request_descriptor: int, response_descriptor: int) -> int:
    try:
        source_frame = _read_frame(source_descriptor, maximum_bytes=SOURCE_FRAME_MAX_BYTES)
        if len(source_frame) < 37 or source_frame[:4] != SOURCE_MAGIC:
            raise ValueError("source bootstrap frame is malformed")
        expected_digest = source_frame[4:36]
        source = source_frame[36:]
        if hashlib.sha256(source).digest() != expected_digest:
            raise ValueError("source bootstrap digest differs")
        program = validate_task_term_source(source)
        if bytes.fromhex(program.receipt.source_sha256) != expected_digest:
            raise ValueError("validated source digest differs")
        _write_frame(response_descriptor, bytes((READY_OPCODE,)) + expected_digest)
    except BaseException:
        _error(response_descriptor, 1)
        return EXIT_SOURCE_BOOTSTRAP_FAILED

    while True:
        try:
            payload = _read_frame(request_descriptor, maximum_bytes=REQUEST_FRAME_MAX_BYTES)
            if payload == bytes((QUIT_OPCODE,)):
                _write_frame(response_descriptor, bytes((QUIT_OPCODE,)))
                return 0
            if len(payload) != struct.calcsize(">BB8d"):
                raise ValueError("batch request size differs")
            unpacked = struct.unpack(">BB8d", payload)
            if unpacked[0] != BATCH_OPCODE or unpacked[1] != 4:
                raise ValueError("batch request opcode or count differs")
            values = unpacked[2:]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("batch request contains non-finite values")
            outputs: list[float] = []
            for index in range(0, 8, 2):
                inputs = CandidateTaskInputsV1(values[index], values[index + 1])
                outputs.append(program.task_term(inputs))
            response = bytes((VALUES_OPCODE,)) + struct.pack(">4d", *outputs)
            _write_frame(response_descriptor, response)
        except BaseException:
            _error(response_descriptor, 2)
            return EXIT_REQUEST_OR_EVALUATION_FAILED


def _operation_was_denied(operation: object) -> int:
    try:
        operation()  # type: ignore[operator]
    except BaseException:
        return EXIT_OK
    return EXIT_CANARY_NOT_CONTAINED


def _network_operation_was_denied(operation: str, address: tuple[str, int]) -> bool | None:
    """Return EPERM status for one governed operation on a fresh socket."""

    import socket

    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except BaseException:
        # Socket creation is not governed by Seatbelt's network operations, so
        # a creation failure cannot establish containment.
        return None
    try:
        getattr(probe, operation)(address)
    except PermissionError:
        return True
    except OSError as exc:
        return exc.errno == errno.EPERM
    except BaseException:
        return None
    else:
        return False
    finally:
        with contextlib.suppress(OSError):
            probe.close()


def _run_network_canary() -> int:
    results = (
        _network_operation_was_denied("connect", ("127.0.0.1", 9)),
        _network_operation_was_denied("bind", ("127.0.0.1", 0)),
    )
    if None in results:
        return EXIT_CANARY_INVOCATION_INVALID
    return EXIT_OK if all(results) else EXIT_CANARY_NOT_CONTAINED


def _run_process_info_canary(parent_pid: int) -> int:
    """Probe Seatbelt process-info through libproc, not unrelated getpriority."""

    import ctypes

    try:
        libproc = ctypes.CDLL(LIBPROC_PATH, use_errno=True)
        proc_pidinfo = libproc.proc_pidinfo
        proc_pidinfo.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
        proc_pidinfo.restype = ctypes.c_int
        buffer = ctypes.create_string_buffer(PROC_PIDINFO_BUFFER_BYTES)
        ctypes.set_errno(0)
        returned_bytes = proc_pidinfo(
            parent_pid,
            PROC_PIDTBSDINFO,
            0,
            buffer,
            len(buffer),
        )
        operation_errno = ctypes.get_errno()
    except BaseException:
        return EXIT_CANARY_INVOCATION_INVALID
    if returned_bytes == 0 and operation_errno == errno.EPERM:
        return EXIT_OK
    return EXIT_CANARY_NOT_CONTAINED


def _run_os_canary(
    kind: str,
    target: str | None,
    parent_pid: int | None,
    *,
    address_space_status: int,
) -> int:
    """Return zero only when the named hostile operation is contained."""

    if kind in {"file_read", "repository_read"}:
        return _operation_was_denied(lambda: open(str(target), "rb").read(1))
    if kind == "file_write":
        return _operation_was_denied(lambda: open("sandbox-escape", "wb").write(b"x"))
    if kind == "environment_secret":
        return EXIT_OK if dict(os.environ) == WORKER_ENVIRONMENT else EXIT_CANARY_NOT_CONTAINED
    if kind == "network":
        return _run_network_canary()
    if kind == "process_creation":
        import subprocess

        return _operation_was_denied(
            lambda: subprocess.run(["/usr/bin/true"], check=False, close_fds=True)
        )
    if kind == "fork":
        return _operation_was_denied(os.fork)
    if kind == "signal":
        import signal

        if parent_pid is None:
            return EXIT_CANARY_INVOCATION_INVALID
        return _operation_was_denied(lambda: os.kill(parent_pid, signal.SIGCONT))
    if kind == "tracing":
        if parent_pid is None:
            return EXIT_CANARY_INVOCATION_INVALID
        return _run_process_info_canary(parent_pid)
    if kind == "stdout_injection":
        os.write(1, b"contained-stdout-canary")
        return EXIT_OK
    if kind == "private_fd_injection":
        escaped = False
        for descriptor in range(3, 10):
            try:
                os.write(descriptor, b"x")
            except OSError:
                continue
            escaped = True
        return EXIT_CANARY_NOT_CONTAINED if escaped else EXIT_OK
    if kind == "timeout":
        while True:
            pass
    if kind == "memory_abuse":
        blocks: list[bytearray] = []
        while True:
            try:
                blocks.append(bytearray(MEMORY_CANARY_BLOCK_BYTES))
            except MemoryError:
                if address_space_status == ADDRESS_SPACE_ENFORCED:
                    return EXIT_OK
                # With no finite RLIMIT_AS, only the parent's fixed deadline
                # and process-group kill may establish containment.
                continue
    if kind == "worker_crash":
        os._exit(EXIT_INTENTIONAL_CRASH_CANARY)
    return EXIT_CANARY_INVOCATION_INVALID


def main() -> int:
    try:
        _sanitize_environment()
    except BaseException:
        return EXIT_ENVIRONMENT_INVALID
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-fd", type=int)
    parser.add_argument("--request-fd", type=int)
    parser.add_argument("--response-fd", type=int)
    parser.add_argument("--limit-status-fd", type=int)
    parser.add_argument("--limit-probe", action="store_true")
    parser.add_argument("--os-canary")
    parser.add_argument("--target")
    parser.add_argument("--parent-pid", type=int)
    arguments = parser.parse_args()
    if arguments.limit_status_fd is None:
        return EXIT_REQUIRED_DESCRIPTOR_MISSING
    try:
        address_space_status = _apply_worker_limits()
    except WorkerLimitError as exc:
        with contextlib.suppress(BaseException):
            _write_limit_status(
                arguments.limit_status_fd,
                address_space_status=exc.address_space_status,
                limit_error=exc.limit_code,
            )
        with contextlib.suppress(OSError):
            os.close(arguments.limit_status_fd)
        return EXIT_LIMIT_APPLICATION_FAILED
    except BaseException:
        with contextlib.suppress(BaseException):
            _write_limit_status(
                arguments.limit_status_fd,
                address_space_status=ADDRESS_SPACE_FAILED,
                limit_error=LIMIT_STATUS_INTERNAL,
            )
        with contextlib.suppress(OSError):
            os.close(arguments.limit_status_fd)
        return EXIT_LIMIT_STATUS_INTERNAL
    try:
        _write_limit_status(
            arguments.limit_status_fd,
            address_space_status=address_space_status,
        )
    finally:
        with contextlib.suppress(OSError):
            os.close(arguments.limit_status_fd)
    if arguments.limit_probe:
        return EXIT_OK
    if arguments.os_canary:
        return _run_os_canary(
            arguments.os_canary,
            arguments.target,
            arguments.parent_pid,
            address_space_status=address_space_status,
        )
    if None in (arguments.source_fd, arguments.request_fd, arguments.response_fd):
        return EXIT_REQUIRED_DESCRIPTOR_MISSING
    return _run_worker(arguments.source_fd, arguments.request_fd, arguments.response_fd)


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
