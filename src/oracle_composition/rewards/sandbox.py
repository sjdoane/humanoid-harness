"""Seatbelt-confined persistent reward worker using only framed pipes."""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
import select
import shutil
import signal
import stat
import struct
import subprocess
import sys
import sysconfig
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from . import _sandbox_worker
from .contract import CandidateTaskInputsV1, RewardContractError, canonical_json_bytes
from .static_validation import MAX_SOURCE_BYTES, statically_validate_task_term_source

SEATBELT_EXECUTABLE = Path("/usr/bin/sandbox-exec")
SEATBELT_PROFILE_ID = "reward-worker-seatbelt/v3"
STARTUP_TIMEOUT_SECONDS = 5.0
BATCH_TIMEOUT_SECONDS = 0.050
STDERR_FIRST_LINE_MAX_BYTES = 200
EXPECTED_BATCH_SIZE = 4
REQUEST_MAX_BYTES = 1024
RESPONSE_MAX_BYTES = 64
SOURCE_FRAME_MAX_BYTES = MAX_SOURCE_BYTES + 36
ADDRESS_SPACE_LIMIT_BYTES = _sandbox_worker.ADDRESS_SPACE_LIMIT_BYTES
CPU_LIMIT_SECONDS = _sandbox_worker.CPU_LIMIT_SECONDS
OPEN_FILE_LIMIT = _sandbox_worker.OPEN_FILE_LIMIT
FILE_SIZE_LIMIT_BYTES = _sandbox_worker.FILE_SIZE_LIMIT_BYTES
CORE_SIZE_LIMIT_BYTES = _sandbox_worker.CORE_SIZE_LIMIT_BYTES
DARWIN_ADDRESS_SPACE_LIMIT_REASON = "darwin_refuses_finite_rlimit_as"
RUNTIME_ADMISSION_REFUSAL = (
    "candidate runtime admission is unreviewed; R1 permits static source capture only"
)
ENVIRONMENT_ALLOWLIST = ("PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED", "PYTHONNOUSERSITE")
WORKER_ENVIRONMENT = MappingProxyType(
    {
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
)
CANARY_NAMES = (
    "file_read",
    "file_write",
    "environment_secret",
    "network",
    "process_creation",
    "fork",
    "signal",
    "tracing",
    "repository_read",
    "source_drift",
    "timeout",
    "memory_abuse",
    "stdout_injection",
    "private_fd_injection",
    "pickle",
    "object_dtype",
    "oversize_frame",
    "extra_frame",
    "worker_crash",
    "malformed_response",
)
OS_CANARY_NAMES = (
    "file_read",
    "file_write",
    "environment_secret",
    "network",
    "process_creation",
    "fork",
    "signal",
    "tracing",
    "repository_read",
    "timeout",
    "memory_abuse",
    "stdout_injection",
    "private_fd_injection",
    "worker_crash",
)
SELF_CANARY_NAMES = tuple(name for name in CANARY_NAMES if name not in OS_CANARY_NAMES)
_VALID_CANARY_VERDICTS = frozenset({"passed", "failed", "not_verified_in_builder_sandbox"})
_VALID_MEMORY_CONTAINMENT_MECHANISMS = frozenset({"rlimit_as", "timeout_kill"})
_VALID_WORKER_EXIT_KINDS = frozenset({"exit_code", "signal"})
_LIMIT_FAILURE_NAMES = {
    _sandbox_worker.LIMIT_STATUS_ADDRESS_SPACE: "address_space_limit",
    _sandbox_worker.LIMIT_STATUS_CPU: "cpu_limit",
    _sandbox_worker.LIMIT_STATUS_FILE_SIZE: "file_size_limit",
    _sandbox_worker.LIMIT_STATUS_CORE_SIZE: "core_size_limit",
    _sandbox_worker.LIMIT_STATUS_OPEN_FILES: "open_file_limit",
    _sandbox_worker.LIMIT_STATUS_INTERNAL: "internal_limit_status",
}


class RewardSandboxError(RewardContractError):
    """Raised when isolation, framing, identity, or worker health fails closed."""


def _require_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RewardSandboxError(f"{field} must be a lowercase SHA-256")
    return value


def sandbox_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def sandbox_worker_source_sha256() -> str:
    return hashlib.sha256(Path(_sandbox_worker.__file__).read_bytes()).hexdigest()


def _python_executable_path(python_executable: Path | None = None) -> Path:
    if python_executable is None:
        python_executable = Path(__file__).resolve().parents[3] / ".venv" / "bin" / "python"
    launch_path = Path(os.path.abspath(python_executable))
    if not launch_path.is_file():
        raise RewardSandboxError(f"Python launch path is not a file: {launch_path}")
    return launch_path


@dataclass(frozen=True, slots=True)
class PythonInterpreterIdentityV1:
    launch_path: str
    launch_file_sha256: str
    resolved_path: str
    resolved_file_sha256: str

    def __post_init__(self) -> None:
        if type(self.launch_path) is not str or not Path(self.launch_path).is_absolute():
            raise RewardSandboxError("interpreter launch path must be absolute")
        if type(self.resolved_path) is not str or not Path(self.resolved_path).is_absolute():
            raise RewardSandboxError("resolved interpreter path must be absolute")
        _require_sha256(self.launch_file_sha256, field="launch_file_sha256")
        _require_sha256(self.resolved_file_sha256, field="resolved_file_sha256")

    def to_dict(self) -> dict[str, str]:
        return {
            "launch_path": self.launch_path,
            "launch_file_sha256": self.launch_file_sha256,
            "resolved_path": self.resolved_path,
            "resolved_file_sha256": self.resolved_file_sha256,
        }


def capture_python_interpreter_identity(
    python_executable: Path | None = None,
) -> PythonInterpreterIdentityV1:
    """Hash the unresolved launch path and the binary it resolves to."""

    launch_path = _python_executable_path(python_executable)
    try:
        resolved_path = Path(os.path.realpath(launch_path))
        if not resolved_path.is_file():
            raise OSError(f"resolved interpreter is not a file: {resolved_path}")
        launch_sha256 = hashlib.sha256(launch_path.read_bytes()).hexdigest()
        resolved_sha256 = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise RewardSandboxError(f"cannot identify Python interpreter: {exc}") from exc
    return PythonInterpreterIdentityV1(
        launch_path=str(launch_path),
        launch_file_sha256=launch_sha256,
        resolved_path=str(resolved_path),
        resolved_file_sha256=resolved_sha256,
    )


def _worker_exit_code_receipt() -> dict[str, str]:
    return {
        str(code): meaning
        for code, meaning in sorted(_sandbox_worker.WORKER_EXIT_CODE_MEANINGS.items())
    }


def _file_state(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_uid),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


@dataclass(frozen=True, slots=True)
class CandidateSourceSnapshotV1:
    path: Path
    source_bytes: bytes
    source_sha256: str
    file_state: tuple[int, ...]


def capture_candidate_source(path: Path) -> CandidateSourceSnapshotV1:
    """Read one regular source file through a stable no-follow descriptor."""

    target = Path(os.path.abspath(path))
    descriptor: int | None = None
    try:
        before_path = target.lstat()
        if stat.S_ISLNK(before_path.st_mode) or not stat.S_ISREG(before_path.st_mode):
            raise RewardSandboxError("candidate source must be one regular non-symlink file")
        if not 0 < before_path.st_size <= MAX_SOURCE_BYTES:
            raise RewardSandboxError("candidate source is empty or exceeds 16 KiB")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(target, flags)
        opened = os.fstat(descriptor)
        if _file_state(opened) != _file_state(before_path):
            raise RewardSandboxError("candidate source changed before snapshot")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                raise RewardSandboxError("candidate source shortened during snapshot")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RewardSandboxError("candidate source grew during snapshot")
        after_fd = os.fstat(descriptor)
        after_path = target.lstat()
        if _file_state(opened) != _file_state(after_fd) or _file_state(opened) != _file_state(
            after_path
        ):
            raise RewardSandboxError("candidate source changed during snapshot")
        source = b"".join(chunks)
    except RewardSandboxError:
        raise
    except OSError as exc:
        raise RewardSandboxError(f"cannot snapshot candidate source: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    validated = statically_validate_task_term_source(source)
    return CandidateSourceSnapshotV1(
        path=target,
        source_bytes=source,
        source_sha256=validated.receipt.source_sha256,
        file_state=_file_state(opened),
    )


def assert_source_snapshot_unchanged(snapshot: CandidateSourceSnapshotV1) -> None:
    current = capture_candidate_source(snapshot.path)
    if (
        current.source_sha256 != snapshot.source_sha256
        or current.source_bytes != snapshot.source_bytes
        or current.file_state != snapshot.file_state
    ):
        raise RewardSandboxError("candidate source drifted after its immutable snapshot")


def _seatbelt_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _seatbelt_profile_v3(interpreter: PythonInterpreterIdentityV1) -> str:
    """Build the deny-default profile with only worker startup reads and pipes."""

    python = Path(interpreter.launch_path)
    resolved_python = Path(interpreter.resolved_path)
    project_package = Path(__file__).resolve().parents[1]
    package_directories = (
        project_package.parent,
        project_package,
        project_package / "rewards",
    )
    exact_sources = (
        project_package / "__init__.py",
        project_package / "rewards" / "__init__.py",
        project_package / "rewards" / "contract.py",
        project_package / "rewards" / "static_validation.py",
        project_package / "rewards" / "_sandbox_worker.py",
    )
    read_roots = {
        str(Path(sys.base_prefix).resolve()),
        str(Path(sys.prefix).resolve()),
        str(Path(sysconfig.get_paths()["stdlib"]).resolve()),
        "/System",
        "/usr/lib",
        "/Library/Apple",
        "/private/var/db/timezone",
    }
    rules = [
        "(version 1)",
        "(deny default)",
        f"(allow process-exec (literal {_seatbelt_string(str(python))}))",
        "(allow sysctl-read)",
        '(allow file-read* (literal "/dev/null"))',
        "(allow file-read-metadata)",
        '(allow file-read-data (literal "/"))',
        f"(allow file-read* (literal {_seatbelt_string(str(python))}))",
    ]
    if resolved_python != python:
        rules.append(f"(allow process-exec (literal {_seatbelt_string(str(resolved_python))}))")
    for root in sorted(read_roots):
        rules.append(f"(allow file-read* (subpath {_seatbelt_string(root)}))")
    for directory in package_directories:
        rules.append(f"(allow file-read-data (literal {_seatbelt_string(str(directory))}))")
    for source in exact_sources:
        rules.append(f"(allow file-read-data (literal {_seatbelt_string(str(source))}))")
    rules.append(f"(allow file-read-metadata (subpath {_seatbelt_string(str(project_package))}))")
    rules.extend(
        (
            "(deny network*)",
            "(deny file-write*)",
            "(deny process-fork)",
            "(deny signal)",
            "(deny process-info*)",
        )
    )
    return "\n".join(rules) + "\n"


def seatbelt_profile_v3(*, python_executable: Path | None = None) -> str:
    return _seatbelt_profile_v3(capture_python_interpreter_identity(python_executable))


def seatbelt_profile_sha256(*, python_executable: Path | None = None) -> str:
    return hashlib.sha256(
        seatbelt_profile_v3(python_executable=python_executable).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class AddressSpaceLimitReceiptV1:
    requested_bytes: int
    enforced: bool
    reason: str | None

    def __post_init__(self) -> None:
        if self.requested_bytes != ADDRESS_SPACE_LIMIT_BYTES:
            raise RewardSandboxError("address-space request differs from the frozen 1 GiB limit")
        if type(self.enforced) is not bool:
            raise RewardSandboxError("address-space enforced flag must be boolean")
        if self.enforced:
            if self.reason is not None:
                raise RewardSandboxError("an enforced address-space limit cannot carry a reason")
        elif self.reason not in {
            DARWIN_ADDRESS_SPACE_LIMIT_REASON,
            "rlimit_as_failed",
        }:
            raise RewardSandboxError("address-space non-enforcement reason is invalid")

    @property
    def fatal(self) -> bool:
        return not self.enforced and self.reason != DARWIN_ADDRESS_SPACE_LIMIT_REASON

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_bytes": self.requested_bytes,
            "enforced": self.enforced,
            "reason": self.reason,
        }


def _worker_exit_meaning(kind: str, value: int) -> str:
    if kind == "exit_code":
        return _sandbox_worker.WORKER_EXIT_CODE_MEANINGS.get(value, "unrecognized_worker_exit_code")
    try:
        return signal.Signals(value).name
    except ValueError:
        return "unrecognized_signal"


@dataclass(frozen=True, slots=True)
class WorkerExitStatusV1:
    kind: str
    value: int
    meaning: str

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind not in _VALID_WORKER_EXIT_KINDS:
            raise RewardSandboxError("worker exit kind is invalid")
        if type(self.value) is not int or self.value < 0:
            raise RewardSandboxError("worker exit value is invalid")
        if self.kind == "exit_code" and self.value > 255:
            raise RewardSandboxError("worker exit code is outside the POSIX byte range")
        if self.kind == "signal" and self.value == 0:
            raise RewardSandboxError("worker signal number must be positive")
        if self.meaning != _worker_exit_meaning(self.kind, self.value):
            raise RewardSandboxError("worker exit meaning differs from its categorical value")

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind, "value": self.value, "meaning": self.meaning}


def _worker_exit_status(returncode: int | None) -> WorkerExitStatusV1 | None:
    if returncode is None:
        return None
    kind = "signal" if returncode < 0 else "exit_code"
    value = -returncode if returncode < 0 else returncode
    return WorkerExitStatusV1(kind, value, _worker_exit_meaning(kind, value))


def _sanitized_stderr_first_line(stderr: bytes) -> str | None:
    if type(stderr) is not bytes or not stderr:
        return None
    raw_line = stderr.splitlines()[0][:STDERR_FIRST_LINE_MAX_BYTES]
    decoded = raw_line.decode("utf-8", errors="replace")
    result: list[str] = []
    byte_count = 0
    for character in decoded:
        sanitized = character if character.isprintable() else "?"
        encoded = sanitized.encode("utf-8")
        if byte_count + len(encoded) > STDERR_FIRST_LINE_MAX_BYTES:
            break
        result.append(sanitized)
        byte_count += len(encoded)
    return "".join(result).strip() or None


@dataclass(frozen=True, slots=True)
class CanaryVerdictV1:
    verdict: str
    error: str | None
    mechanism: str | None = None
    exit_status: WorkerExitStatusV1 | None = None
    stderr_first_line: str | None = None

    def __post_init__(self) -> None:
        if type(self.verdict) is not str or self.verdict not in _VALID_CANARY_VERDICTS:
            raise RewardSandboxError("unknown canary verdict")
        if self.verdict == "passed" and self.error is not None:
            raise RewardSandboxError("passed canary cannot carry an error")
        if self.verdict != "passed" and (type(self.error) is not str or not self.error):
            raise RewardSandboxError("non-passing canary requires the exact error")
        if self.mechanism is not None and self.mechanism not in (
            _VALID_MEMORY_CONTAINMENT_MECHANISMS
        ):
            raise RewardSandboxError("unknown canary containment mechanism")
        if self.verdict != "passed" and self.mechanism is not None:
            raise RewardSandboxError("a non-passing canary cannot claim a mechanism acted")
        if self.exit_status is not None and not isinstance(self.exit_status, WorkerExitStatusV1):
            raise RewardSandboxError("canary worker exit status is invalid")
        if self.stderr_first_line is not None and (
            type(self.stderr_first_line) is not str
            or not self.stderr_first_line
            or "\n" in self.stderr_first_line
            or "\r" in self.stderr_first_line
            or len(self.stderr_first_line.encode("utf-8")) > STDERR_FIRST_LINE_MAX_BYTES
        ):
            raise RewardSandboxError("canary stderr line is not sanitized and bounded")
        if self.verdict != "failed" and (
            self.exit_status is not None or self.stderr_first_line is not None
        ):
            raise RewardSandboxError("only a failed OS canary may carry process diagnostics")

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "verdict": self.verdict,
            "error": self.error,
            "mechanism": self.mechanism,
        }
        if self.exit_status is not None:
            value["exit_status"] = self.exit_status.to_dict()
        if self.stderr_first_line is not None:
            value["stderr_first_line"] = self.stderr_first_line
        return value


@dataclass(frozen=True, slots=True)
class SeatbeltCanaryReceiptV1:
    sandbox_executable: str
    profile_identity: str
    profile_sha256: str
    interpreter: PythonInterpreterIdentityV1
    address_space_limit: AddressSpaceLimitReceiptV1
    canaries: Mapping[str, CanaryVerdictV1]

    def __post_init__(self) -> None:
        if type(self.sandbox_executable) is not str or not self.sandbox_executable:
            raise RewardSandboxError("canary receipt sandbox executable is invalid")
        if self.profile_identity != SEATBELT_PROFILE_ID:
            raise RewardSandboxError("canary receipt profile identity is not v3")
        _require_sha256(self.profile_sha256, field="profile_sha256")
        if not isinstance(self.interpreter, PythonInterpreterIdentityV1):
            raise RewardSandboxError("canary receipt interpreter identity is invalid")
        if not isinstance(self.address_space_limit, AddressSpaceLimitReceiptV1):
            raise RewardSandboxError("canary receipt address-space limit is invalid")
        if not isinstance(self.canaries, Mapping) or set(self.canaries) != set(CANARY_NAMES):
            raise RewardSandboxError("canary receipt does not cover the complete frozen set")
        if any(not isinstance(value, CanaryVerdictV1) for value in self.canaries.values()):
            raise RewardSandboxError("canary receipt contains an invalid verdict")
        object.__setattr__(
            self,
            "canaries",
            MappingProxyType({name: self.canaries[name] for name in CANARY_NAMES}),
        )
        memory_canary = self.canaries["memory_abuse"]
        for name, result in self.canaries.items():
            if name != "memory_abuse" and result.mechanism is not None:
                raise RewardSandboxError("only memory_abuse may name a containment mechanism")
        if memory_canary.verdict == "passed":
            expected_mechanism = (
                "rlimit_as" if self.address_space_limit.enforced else "timeout_kill"
            )
            if memory_canary.mechanism != expected_mechanism:
                raise RewardSandboxError(
                    "memory_abuse did not name the containment mechanism that acted"
                )
        elif memory_canary.mechanism is not None:
            raise RewardSandboxError("unverified memory containment cannot name a mechanism")
        if self.address_space_limit.fatal and self.all_passed:
            raise RewardSandboxError("failed address-space limit cannot accompany passing canaries")

    @property
    def all_passed(self) -> bool:
        return all(item.verdict == "passed" for item in self.canaries.values())

    def assert_all_passed(self) -> None:
        failed = [name for name in CANARY_NAMES if self.canaries[name].verdict != "passed"]
        if failed:
            raise RewardSandboxError(
                "reward worker cannot start without every Seatbelt/runtime canary: "
                + ", ".join(failed)
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "sandbox_executable": self.sandbox_executable,
            "profile_identity": self.profile_identity,
            "profile_sha256": self.profile_sha256,
            "interpreter": self.interpreter.to_dict(),
            "worker_exit_codes": _worker_exit_code_receipt(),
            "address_space_limit": self.address_space_limit.to_dict(),
            "canaries": {name: self.canaries[name].to_dict() for name in CANARY_NAMES},
            "all_passed": self.all_passed,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


def _decode_worker_limit_status(
    payload: bytes,
) -> tuple[AddressSpaceLimitReceiptV1, str | None]:
    if type(payload) is not bytes or len(payload) != 3:
        raise RewardSandboxError("worker limit status does not match the fixed schema")
    if payload[0] != _sandbox_worker.LIMIT_STATUS_OPCODE:
        raise RewardSandboxError("worker limit status opcode differs")
    address_status = payload[1]
    if address_status == _sandbox_worker.ADDRESS_SPACE_ENFORCED:
        address_space_limit = AddressSpaceLimitReceiptV1(
            requested_bytes=ADDRESS_SPACE_LIMIT_BYTES,
            enforced=True,
            reason=None,
        )
    elif address_status == _sandbox_worker.ADDRESS_SPACE_DARWIN_REFUSED:
        address_space_limit = AddressSpaceLimitReceiptV1(
            requested_bytes=ADDRESS_SPACE_LIMIT_BYTES,
            enforced=False,
            reason=DARWIN_ADDRESS_SPACE_LIMIT_REASON,
        )
    elif address_status == _sandbox_worker.ADDRESS_SPACE_FAILED:
        address_space_limit = AddressSpaceLimitReceiptV1(
            requested_bytes=ADDRESS_SPACE_LIMIT_BYTES,
            enforced=False,
            reason="rlimit_as_failed",
        )
    else:
        raise RewardSandboxError("worker returned an unknown address-space status")

    failure_code = payload[2]
    if failure_code == _sandbox_worker.LIMIT_STATUS_OK:
        if address_space_limit.fatal:
            raise RewardSandboxError("fatal address-space status omitted its failure category")
        return address_space_limit, None
    failure = _LIMIT_FAILURE_NAMES.get(failure_code)
    if failure is None:
        raise RewardSandboxError("worker returned an unknown mandatory-limit failure")
    return address_space_limit, failure


def _force_stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        with contextlib.suppress(OSError):
            os.killpg(process.pid, signal.SIGKILL)
        if process.poll() is None:
            with contextlib.suppress(OSError):
                process.kill()
    with contextlib.suppress(OSError, subprocess.TimeoutExpired):
        process.wait(timeout=1.0)


def _collect_stderr_and_exit(
    process: subprocess.Popen[bytes],
) -> tuple[WorkerExitStatusV1 | None, str | None]:
    try:
        _stdout, stderr = process.communicate(timeout=1.0)
    except subprocess.TimeoutExpired:
        _force_stop_process(process)
        try:
            _stdout, stderr = process.communicate(timeout=1.0)
        except (OSError, subprocess.SubprocessError):
            stderr = b""
    return _worker_exit_status(process.returncode), _sanitized_stderr_first_line(stderr or b"")


def _failed_canary_verdict(
    error: str,
    *,
    process: subprocess.Popen[bytes] | None = None,
    returncode: int | None = None,
    stderr: bytes = b"",
) -> CanaryVerdictV1:
    if process is not None:
        exit_status, stderr_first_line = _collect_stderr_and_exit(process)
    else:
        exit_status = _worker_exit_status(returncode)
        stderr_first_line = _sanitized_stderr_first_line(stderr)
    return CanaryVerdictV1(
        "failed",
        error,
        exit_status=exit_status,
        stderr_first_line=stderr_first_line,
    )


def _kill_process_group(process: subprocess.Popen[bytes]) -> bool:
    """Return true only when this parent delivered and observed group SIGKILL."""

    if process.poll() is not None:
        return False
    try:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=1.0)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return process.returncode == -signal.SIGKILL


def _probe_worker_limits(
    *,
    python_executable: Path,
) -> tuple[AddressSpaceLimitReceiptV1, str | None]:
    """Apply limits in a trusted short-lived child without running candidate code."""

    private_cwd = Path(tempfile.mkdtemp(prefix="reward-limit-probe-"))
    os.chmod(private_cwd, 0o700)
    status_read, status_write = os.pipe()
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(
            [
                str(python_executable),
                "-I",
                "-m",
                "oracle_composition.rewards._sandbox_worker",
                "--limit-status-fd",
                str(status_write),
                "--limit-probe",
            ],
            cwd=private_cwd,
            env=dict(WORKER_ENVIRONMENT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=(status_write,),
            start_new_session=True,
        )
        os.close(status_write)
        status_write = -1
        payload = _read_frame_with_timeout(
            status_read,
            maximum_bytes=RESPONSE_MAX_BYTES,
            timeout_seconds=STARTUP_TIMEOUT_SECONDS,
        )
        address_space_limit, failure = _decode_worker_limit_status(payload)
        process.wait(timeout=STARTUP_TIMEOUT_SECONDS)
        expected_returncode = (
            _sandbox_worker.EXIT_OK
            if failure is None
            else _sandbox_worker.EXIT_LIMIT_APPLICATION_FAILED
        )
        if process.returncode != expected_returncode:
            raise RewardSandboxError(
                "worker limit probe exit status contradicts its categorical receipt"
            )
        return address_space_limit, failure
    except RewardSandboxError:
        raise
    except (OSError, subprocess.SubprocessError) as exc:
        raise RewardSandboxError(f"worker limit probe failed: {type(exc).__name__}: {exc}") from exc
    finally:
        for descriptor in (status_read, status_write):
            if descriptor >= 0:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
        if process is not None:
            _force_stop_process(process)
        shutil.rmtree(private_cwd)


def _boundary_unavailable_canary_receipt(
    *,
    verdict: str,
    error: str,
    address_space_limit: AddressSpaceLimitReceiptV1,
    sandbox_executable: Path,
    interpreter: PythonInterpreterIdentityV1,
) -> SeatbeltCanaryReceiptV1:
    results = {
        name: (
            CanaryVerdictV1(verdict, error)
            if name in OS_CANARY_NAMES
            else _protocol_or_identity_self_canary(name)
        )
        for name in CANARY_NAMES
    }
    return SeatbeltCanaryReceiptV1(
        sandbox_executable=str(sandbox_executable),
        profile_identity=SEATBELT_PROFILE_ID,
        profile_sha256=hashlib.sha256(
            _seatbelt_profile_v3(interpreter).encode("utf-8")
        ).hexdigest(),
        interpreter=interpreter,
        address_space_limit=address_space_limit,
        canaries=results,
    )


def _run_os_canary(
    name: str,
    *,
    profile: str,
    private_cwd: Path,
    repository_probe: Path,
    python_executable: Path,
    sandbox_executable: Path,
    expected_address_space_limit: AddressSpaceLimitReceiptV1,
) -> tuple[CanaryVerdictV1, AddressSpaceLimitReceiptV1]:
    probe = Path("/etc/hosts") if name == "file_read" else repository_probe
    status_read, status_write = os.pipe()
    command = [
        str(sandbox_executable),
        "-p",
        profile,
        str(python_executable),
        "-I",
        "-m",
        "oracle_composition.rewards._sandbox_worker",
        "--limit-status-fd",
        str(status_write),
        "--os-canary",
        name,
        "--target",
        str(probe),
        "--parent-pid",
        str(os.getpid()),
    ]
    process: subprocess.Popen[bytes] | None = None
    try:
        environment = dict(WORKER_ENVIRONMENT)
        if name == "environment_secret":
            environment["TASK_B0_CANARY_SECRET"] = "must-not-reach-candidate-code"
        process = subprocess.Popen(
            command,
            cwd=private_cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            close_fds=True,
            pass_fds=(status_write,),
            start_new_session=True,
        )
        os.close(status_write)
        status_write = -1
        status_payload = _read_frame_with_timeout(
            status_read,
            maximum_bytes=RESPONSE_MAX_BYTES,
            timeout_seconds=STARTUP_TIMEOUT_SECONDS,
        )
        address_space_limit, limit_failure = _decode_worker_limit_status(status_payload)
        if limit_failure is not None:
            _stdout, stderr = process.communicate(timeout=STARTUP_TIMEOUT_SECONDS)
            return (
                _failed_canary_verdict(
                    f"worker_limit_failure:{limit_failure}",
                    returncode=process.returncode,
                    stderr=stderr,
                ),
                address_space_limit,
            )
        timeout = (
            BATCH_TIMEOUT_SECONDS
            if name == "timeout" or (name == "memory_abuse" and not address_space_limit.enforced)
            else STARTUP_TIMEOUT_SECONDS
        )
        _stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        assert process is not None
        group_killed = _kill_process_group(process)
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            process.communicate(timeout=1.0)
        if name == "timeout" and group_killed:
            return CanaryVerdictV1("passed", None), address_space_limit
        if name == "memory_abuse" and not address_space_limit.enforced and group_killed:
            return (
                CanaryVerdictV1("passed", None, mechanism="timeout_kill"),
                address_space_limit,
            )
        reason = (
            "parent process-group kill did not terminate the canary"
            if not group_killed
            else "canary exceeded its fixed timeout without the declared mechanism"
        )
        if not group_killed:
            _force_stop_process(process)
        return _failed_canary_verdict(reason, process=process), address_space_limit
    except RewardSandboxError as exc:
        return (
            _failed_canary_verdict(str(exc), process=process),
            expected_address_space_limit,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return (
            _failed_canary_verdict(
                f"{type(exc).__name__}: {exc}",
                process=process,
            ),
            expected_address_space_limit,
        )
    finally:
        for descriptor in (status_read, status_write):
            if descriptor >= 0:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
    assert process is not None
    if name == "worker_crash":
        verdict = (
            CanaryVerdictV1("passed", None)
            if process.returncode == _sandbox_worker.EXIT_INTENTIONAL_CRASH_CANARY
            else _failed_canary_verdict(
                "intentional worker crash returned an unexpected status",
                returncode=process.returncode,
                stderr=stderr,
            )
        )
        return verdict, address_space_limit
    if name == "memory_abuse":
        if address_space_limit.enforced and process.returncode == _sandbox_worker.EXIT_OK:
            return (
                CanaryVerdictV1("passed", None, mechanism="rlimit_as"),
                address_space_limit,
            )
        return (
            _failed_canary_verdict(
                "memory canary exited before the declared containment mechanism acted",
                returncode=process.returncode,
                stderr=stderr,
            ),
            address_space_limit,
        )
    if process.returncode == _sandbox_worker.EXIT_OK:
        return CanaryVerdictV1("passed", None), address_space_limit
    return (
        _failed_canary_verdict(
            "OS canary operation was not contained or its invocation failed",
            returncode=process.returncode,
            stderr=stderr,
        ),
        address_space_limit,
    )


def run_seatbelt_canaries(
    *,
    repository_probe: Path,
    sandbox_executable: Path = SEATBELT_EXECUTABLE,
    python_executable: Path | None = None,
) -> SeatbeltCanaryReceiptV1:
    """Run the OS boundary; nested-denial is recorded, never promoted to pass."""

    if dict(WORKER_ENVIRONMENT) != _sandbox_worker.WORKER_ENVIRONMENT:
        raise RewardSandboxError("parent and worker environment allowlists differ")
    python = _python_executable_path(python_executable)
    interpreter = capture_python_interpreter_identity(python)
    sandbox = Path(sandbox_executable)
    address_space_limit, limit_failure = _probe_worker_limits(python_executable=python)
    if limit_failure is not None:
        return _boundary_unavailable_canary_receipt(
            verdict="failed",
            error=f"worker_limit_failure:{limit_failure}",
            address_space_limit=address_space_limit,
            sandbox_executable=sandbox,
            interpreter=interpreter,
        )
    if not sandbox.is_file():
        return _boundary_unavailable_canary_receipt(
            verdict="failed",
            error=f"Seatbelt executable is missing: {sandbox}",
            address_space_limit=address_space_limit,
            sandbox_executable=sandbox,
            interpreter=interpreter,
        )
    bootstrap_profile = (
        "(version 1)\n(deny default)\n"
        '(allow process-exec (literal "/usr/bin/true"))\n'
        "(allow file-read*)\n"
        "(allow sysctl-read)\n"
        "(allow mach-lookup)\n"
    )
    try:
        bootstrap = subprocess.run(
            [str(sandbox), "-p", bootstrap_profile, "/usr/bin/true"],
            env=dict(WORKER_ENVIRONMENT),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=5.0,
            check=False,
            close_fds=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _boundary_unavailable_canary_receipt(
            verdict="failed",
            error=f"{type(exc).__name__}: {exc}",
            address_space_limit=address_space_limit,
            sandbox_executable=sandbox,
            interpreter=interpreter,
        )
    if bootstrap.returncode != 0:
        exact_error = bootstrap.stderr.decode("utf-8", errors="replace").strip()
        verdict = (
            "not_verified_in_builder_sandbox"
            if "sandbox_apply: Operation not permitted" in exact_error
            else "failed"
        )
        return _boundary_unavailable_canary_receipt(
            verdict=verdict,
            error=exact_error or f"sandbox-exec exited {bootstrap.returncode}",
            address_space_limit=address_space_limit,
            sandbox_executable=sandbox,
            interpreter=interpreter,
        )

    profile = _seatbelt_profile_v3(interpreter)
    private_cwd = Path(tempfile.mkdtemp(prefix="reward-seatbelt-canaries-"))
    os.chmod(private_cwd, 0o700)
    try:
        observed = {
            name: _run_os_canary(
                name,
                profile=profile,
                private_cwd=private_cwd,
                repository_probe=repository_probe.resolve(),
                python_executable=python,
                sandbox_executable=sandbox,
                expected_address_space_limit=address_space_limit,
            )
            for name in OS_CANARY_NAMES
        }
        results: dict[str, CanaryVerdictV1] = {}
        for name in OS_CANARY_NAMES:
            verdict, observed_address_space_limit = observed[name]
            if observed_address_space_limit != address_space_limit:
                verdict = CanaryVerdictV1(
                    "failed",
                    "worker address-space status differs from the limit probe",
                )
            results[name] = verdict
        for name in SELF_CANARY_NAMES:
            results[name] = _protocol_or_identity_self_canary(name)
        return SeatbeltCanaryReceiptV1(
            sandbox_executable=str(sandbox),
            profile_identity=SEATBELT_PROFILE_ID,
            profile_sha256=hashlib.sha256(profile.encode("utf-8")).hexdigest(),
            interpreter=interpreter,
            address_space_limit=address_space_limit,
            canaries=results,
        )
    finally:
        shutil.rmtree(private_cwd)


def _protocol_or_identity_self_canary(name: str) -> CanaryVerdictV1:
    try:
        if name == "source_drift":
            with tempfile.TemporaryDirectory(prefix="reward-source-drift-canary-") as directory:
                path = Path(directory) / "candidate.py"
                source = b"def task_term(x):\n    return 0.0\n"
                path.write_bytes(source)
                snapshot = capture_candidate_source(path)
                path.write_bytes(source + b"\n")
                assert_source_snapshot_unchanged(snapshot)
        elif name == "pickle":
            _decode_batch_response(b"\x80\x05pickle")
        elif name == "object_dtype":
            _encode_batch_request([object(), object(), object(), object()])  # type: ignore[list-item]
        elif name == "oversize_frame":
            _validate_frame_length(REQUEST_MAX_BYTES + 1, maximum_bytes=REQUEST_MAX_BYTES)
        elif name == "extra_frame":
            read_descriptor, write_descriptor = os.pipe()
            try:
                payload = bytes((_sandbox_worker.VALUES_OPCODE,)) + struct.pack(">4d", 0, 0, 0, 0)
                framed = _encode_frame(payload, maximum_bytes=RESPONSE_MAX_BYTES)
                _write_all(write_descriptor, framed + framed)
                first = _read_frame_with_timeout(
                    read_descriptor,
                    maximum_bytes=RESPONSE_MAX_BYTES,
                    timeout_seconds=0.1,
                )
                _decode_batch_response(first)
                _assert_response_pipe_quiet(read_descriptor)
            finally:
                os.close(read_descriptor)
                os.close(write_descriptor)
        elif name == "malformed_response":
            _decode_batch_response(b"V")
        else:
            return CanaryVerdictV1("failed", "unknown protocol self-canary")
    except RewardSandboxError:
        return CanaryVerdictV1("passed", None)
    return CanaryVerdictV1("failed", f"{name} payload was accepted")


def _validate_frame_length(length: int, *, maximum_bytes: int) -> None:
    if type(length) is not int or not 0 < length <= maximum_bytes:
        raise RewardSandboxError("frame length is outside its fixed bound")


def _encode_frame(payload: bytes, *, maximum_bytes: int) -> bytes:
    if type(payload) is not bytes:
        raise RewardSandboxError("frame payload must be exact bytes")
    _validate_frame_length(len(payload), maximum_bytes=maximum_bytes)
    return len(payload).to_bytes(4, "big") + payload


def _encode_batch_request(inputs: Sequence[CandidateTaskInputsV1]) -> bytes:
    if (
        not isinstance(inputs, Sequence)
        or isinstance(inputs, (str, bytes))
        or len(inputs) != EXPECTED_BATCH_SIZE
        or any(not isinstance(item, CandidateTaskInputsV1) for item in inputs)
    ):
        raise RewardSandboxError("each worker call requires exactly four candidate inputs")
    scalars: list[float] = []
    for item in inputs:
        scalars.extend((item.com_x_velocity_m_s, item.target_speed_m_s))
    payload = struct.pack(">BB8d", _sandbox_worker.BATCH_OPCODE, EXPECTED_BATCH_SIZE, *scalars)
    return _encode_frame(payload, maximum_bytes=REQUEST_MAX_BYTES)


def _decode_batch_response(payload: bytes) -> tuple[float, float, float, float]:
    if type(payload) is not bytes or len(payload) != 1 + struct.calcsize(">4d"):
        raise RewardSandboxError("worker response does not match the fixed numeric schema")
    if payload[0] != _sandbox_worker.VALUES_OPCODE:
        raise RewardSandboxError("worker returned an error or unknown response opcode")
    values = struct.unpack(">4d", payload[1:])
    if not all(math.isfinite(value) and abs(value) <= 1000.0 for value in values):
        raise RewardSandboxError("worker returned non-finite or out-of-envelope output")
    return values


def _write_all(descriptor: int, value: bytes) -> None:
    view = memoryview(value)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise RewardSandboxError("pipe write made no progress")
        written += count


def _read_exact_with_timeout(descriptor: int, count: int, timeout_seconds: float) -> bytes:
    deadline = time.monotonic() + timeout_seconds
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        timeout = deadline - time.monotonic()
        if timeout <= 0.0:
            raise RewardSandboxError("worker response timed out")
        readable, _, _ = select.select([descriptor], [], [], timeout)
        if not readable:
            raise RewardSandboxError("worker response timed out")
        chunk = os.read(descriptor, remaining)
        if not chunk:
            raise RewardSandboxError("worker pipe closed before a complete response")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_frame_with_timeout(
    descriptor: int, *, maximum_bytes: int, timeout_seconds: float
) -> bytes:
    started = time.monotonic()
    header = _read_exact_with_timeout(descriptor, 4, timeout_seconds)
    length = int.from_bytes(header, "big")
    _validate_frame_length(length, maximum_bytes=maximum_bytes)
    remaining_timeout = timeout_seconds - (time.monotonic() - started)
    if remaining_timeout <= 0.0:
        raise RewardSandboxError("worker response timed out")
    return _read_exact_with_timeout(descriptor, length, remaining_timeout)


def _assert_response_pipe_quiet(descriptor: int) -> None:
    readable, _, _ = select.select([descriptor], [], [], 0.0)
    if readable:
        extra = os.read(descriptor, 1)
        if extra:
            raise RewardSandboxError("worker injected an extra response frame")
        raise RewardSandboxError("worker closed its response pipe")


@dataclass(frozen=True, slots=True)
class SandboxRuntimeReceiptV1:
    training_seed: int
    source_sha256: str
    validator_source_sha256: str
    sandbox_source_sha256: str
    worker_source_sha256: str
    seatbelt_profile_identity: str
    seatbelt_profile_sha256: str
    canary_receipt_sha256: str
    interpreter: PythonInterpreterIdentityV1
    address_space_limit: AddressSpaceLimitReceiptV1
    applied_limits: Mapping[str, int]
    environment_keys: tuple[str, ...]
    call_count: int
    maximum_call_latency_seconds: float
    exit_status: int

    def __post_init__(self) -> None:
        if type(self.training_seed) is not int or self.training_seed < 0:
            raise RewardSandboxError("runtime receipt training seed is invalid")
        for field in (
            "source_sha256",
            "validator_source_sha256",
            "sandbox_source_sha256",
            "worker_source_sha256",
            "seatbelt_profile_sha256",
            "canary_receipt_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if self.seatbelt_profile_identity != SEATBELT_PROFILE_ID:
            raise RewardSandboxError("runtime receipt profile identity is not v3")
        if not isinstance(self.interpreter, PythonInterpreterIdentityV1):
            raise RewardSandboxError("runtime receipt interpreter identity is invalid")
        if not isinstance(self.address_space_limit, AddressSpaceLimitReceiptV1):
            raise RewardSandboxError("runtime receipt address-space limit is invalid")
        if self.address_space_limit.fatal:
            raise RewardSandboxError("runtime receipt cannot admit a failed address-space limit")
        expected_limits = {
            "cpu_seconds": CPU_LIMIT_SECONDS,
            "file_size_bytes": FILE_SIZE_LIMIT_BYTES,
            "core_size_bytes": CORE_SIZE_LIMIT_BYTES,
            "open_files": OPEN_FILE_LIMIT,
            "startup_timeout_ms": int(STARTUP_TIMEOUT_SECONDS * 1000),
            "batch_timeout_ms": int(BATCH_TIMEOUT_SECONDS * 1000),
        }
        if (
            not isinstance(self.applied_limits, Mapping)
            or dict(self.applied_limits) != expected_limits
        ):
            raise RewardSandboxError("runtime receipt limit set differs")
        if self.environment_keys != ENVIRONMENT_ALLOWLIST:
            raise RewardSandboxError("runtime receipt environment allowlist differs")
        if type(self.call_count) is not int or self.call_count < 0:
            raise RewardSandboxError("runtime receipt call count is invalid")
        if (
            type(self.maximum_call_latency_seconds) is not float
            or not math.isfinite(self.maximum_call_latency_seconds)
            or self.maximum_call_latency_seconds < 0.0
            or self.maximum_call_latency_seconds > BATCH_TIMEOUT_SECONDS
            or (self.call_count == 0) is not (self.maximum_call_latency_seconds == 0.0)
        ):
            raise RewardSandboxError("runtime receipt call latency is invalid")
        if self.exit_status != _sandbox_worker.EXIT_OK:
            raise RewardSandboxError("runtime receipt requires a clean worker exit")
        object.__setattr__(self, "applied_limits", MappingProxyType(expected_limits))

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "training_seed": self.training_seed,
            "source_sha256": self.source_sha256,
            "validator_source_sha256": self.validator_source_sha256,
            "sandbox_source_sha256": self.sandbox_source_sha256,
            "worker_source_sha256": self.worker_source_sha256,
            "seatbelt_profile_identity": self.seatbelt_profile_identity,
            "seatbelt_profile_sha256": self.seatbelt_profile_sha256,
            "canary_receipt_sha256": self.canary_receipt_sha256,
            "interpreter": self.interpreter.to_dict(),
            "worker_exit_codes": _worker_exit_code_receipt(),
            "address_space_limit": self.address_space_limit.to_dict(),
            "applied_limits": dict(self.applied_limits),
            "environment_keys": list(self.environment_keys),
            "call_count": self.call_count,
            "maximum_call_latency_seconds": self.maximum_call_latency_seconds,
            "exit_status": self.exit_status,
        }


class RewardSandboxWorkerV1:
    """One-use, one-seed persistent worker; every fault invalidates the seed."""

    def __init__(
        self,
        *,
        training_seed: int,
        candidate_source_path: Path,
        canary_receipt: SeatbeltCanaryReceiptV1,
        python_executable: Path | None = None,
        sandbox_executable: Path = SEATBELT_EXECUTABLE,
    ) -> None:
        if type(training_seed) is not int or training_seed < 0:
            raise RewardSandboxError("training_seed must be a non-negative integer")
        if not isinstance(canary_receipt, SeatbeltCanaryReceiptV1):
            raise RewardSandboxError("a SeatbeltCanaryReceiptV1 is required")
        if dict(WORKER_ENVIRONMENT) != _sandbox_worker.WORKER_ENVIRONMENT:
            raise RewardSandboxError("parent and worker environment allowlists differ")
        canary_receipt.assert_all_passed()
        self.training_seed = training_seed
        self.snapshot = capture_candidate_source(candidate_source_path)
        self.canary_receipt = canary_receipt
        self.python_executable = _python_executable_path(python_executable)
        self.interpreter = capture_python_interpreter_identity(self.python_executable)
        self.sandbox_executable = Path(sandbox_executable)
        if str(self.sandbox_executable) != canary_receipt.sandbox_executable:
            raise RewardSandboxError("Seatbelt executable differs from the canary receipt")
        if self.interpreter != canary_receipt.interpreter:
            raise RewardSandboxError("Python interpreter differs from the canary receipt")
        if canary_receipt.profile_identity != SEATBELT_PROFILE_ID:
            raise RewardSandboxError("Seatbelt profile identity differs")
        expected_profile = hashlib.sha256(
            _seatbelt_profile_v3(self.interpreter).encode("utf-8")
        ).hexdigest()
        if expected_profile != canary_receipt.profile_sha256:
            raise RewardSandboxError("Seatbelt profile differs from the canary receipt")
        self._process: subprocess.Popen[bytes] | None = None
        self._request_write: int | None = None
        self._response_read: int | None = None
        self._private_cwd: Path | None = None
        self._address_space_limit: AddressSpaceLimitReceiptV1 | None = None
        self._started = False
        self._invalid = False
        self._closed = False
        self._call_count = 0
        self._maximum_call_latency = 0.0
        self._exit_status: int | None = None

    def start(self) -> None:
        raise RewardSandboxError(RUNTIME_ADMISSION_REFUSAL)

    def _start_after_runtime_admission(self) -> None:
        """R2 may call this only after adding a reviewed admission gate."""

        if self._started or self._closed or self._invalid:
            raise RewardSandboxError("worker instances are fresh and may be started only once")
        if capture_python_interpreter_identity(self.python_executable) != self.interpreter:
            raise RewardSandboxError("Python interpreter identity drifted before worker start")
        assert_source_snapshot_unchanged(self.snapshot)
        private_cwd = Path(tempfile.mkdtemp(prefix=f"reward-seed-{self.training_seed}-"))
        os.chmod(private_cwd, 0o700)
        source_read, source_write = os.pipe()
        request_read, request_write = os.pipe()
        response_read, response_write = os.pipe()
        limit_status_read, limit_status_write = os.pipe()
        profile = _seatbelt_profile_v3(self.interpreter)
        command = [
            str(self.sandbox_executable),
            "-p",
            profile,
            str(self.python_executable),
            "-I",
            "-m",
            "oracle_composition.rewards._sandbox_worker",
            "--limit-status-fd",
            str(limit_status_write),
            "--source-fd",
            str(source_read),
            "--request-fd",
            str(request_read),
            "--response-fd",
            str(response_write),
        ]
        process: subprocess.Popen[bytes] | None = None
        try:
            process = subprocess.Popen(
                command,
                cwd=private_cwd,
                env=dict(WORKER_ENVIRONMENT),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                pass_fds=(source_read, request_read, response_write, limit_status_write),
                start_new_session=True,
            )
            os.close(limit_status_write)
            limit_status_write = -1
            limit_payload = _read_frame_with_timeout(
                limit_status_read,
                maximum_bytes=RESPONSE_MAX_BYTES,
                timeout_seconds=STARTUP_TIMEOUT_SECONDS,
            )
            address_space_limit, limit_failure = _decode_worker_limit_status(limit_payload)
            os.close(limit_status_read)
            limit_status_read = -1
            if limit_failure is not None:
                raise RewardSandboxError(f"worker_limit_failure:{limit_failure}")
            if address_space_limit != self.canary_receipt.address_space_limit:
                raise RewardSandboxError(
                    "worker address-space status differs from its canary receipt"
                )
            os.close(source_read)
            source_read = -1
            os.close(request_read)
            request_read = -1
            os.close(response_write)
            response_write = -1
            source_payload = (
                _sandbox_worker.SOURCE_MAGIC
                + bytes.fromhex(self.snapshot.source_sha256)
                + self.snapshot.source_bytes
            )
            _write_all(
                source_write,
                _encode_frame(source_payload, maximum_bytes=SOURCE_FRAME_MAX_BYTES),
            )
            os.close(source_write)
            source_write = -1
            response = _read_frame_with_timeout(
                response_read,
                maximum_bytes=RESPONSE_MAX_BYTES,
                timeout_seconds=STARTUP_TIMEOUT_SECONDS,
            )
            expected = bytes((_sandbox_worker.READY_OPCODE,)) + bytes.fromhex(
                self.snapshot.source_sha256
            )
            if response != expected or process.poll() is not None:
                raise RewardSandboxError("worker did not return the exact source-bound ready frame")
            self._process = process
            self._request_write = request_write
            self._response_read = response_read
            self._private_cwd = private_cwd
            self._address_space_limit = address_space_limit
            self._started = True
            assert_source_snapshot_unchanged(self.snapshot)
        except BaseException as exc:
            for descriptor in (
                source_read,
                source_write,
                request_read,
                request_write,
                response_read,
                response_write,
                limit_status_read,
                limit_status_write,
            ):
                if descriptor >= 0:
                    with contextlib.suppress(OSError):
                        os.close(descriptor)
            if process is not None:
                _force_stop_process(process)
            shutil.rmtree(private_cwd)
            self._invalid = True
            if isinstance(exc, RewardSandboxError):
                raise
            raise RewardSandboxError(f"worker startup failed: {type(exc).__name__}: {exc}") from exc

    def _require_live(self) -> tuple[subprocess.Popen[bytes], int, int]:
        if not self._started or self._closed or self._invalid:
            raise RewardSandboxError("reward worker is not live")
        if self._process is None or self._request_write is None or self._response_read is None:
            raise RewardSandboxError("reward worker state is incomplete")
        if self._address_space_limit is None:
            raise RewardSandboxError("reward worker limit status is incomplete")
        if self._process.poll() is not None:
            self._invalidate()
            raise RewardSandboxError("reward worker crashed; the seed is invalid")
        return self._process, self._request_write, self._response_read

    def evaluate_batch(
        self, inputs: Sequence[CandidateTaskInputsV1]
    ) -> tuple[float, float, float, float]:
        process, request_write, response_read = self._require_live()
        try:
            assert_source_snapshot_unchanged(self.snapshot)
            request = _encode_batch_request(inputs)
            started = time.perf_counter()
            _write_all(request_write, request)
            payload = _read_frame_with_timeout(
                response_read,
                maximum_bytes=RESPONSE_MAX_BYTES,
                timeout_seconds=BATCH_TIMEOUT_SECONDS,
            )
            latency = time.perf_counter() - started
            if latency > BATCH_TIMEOUT_SECONDS:
                raise RewardSandboxError("worker batch exceeded the 50 ms deadline")
            outputs = _decode_batch_response(payload)
            if process.poll() is not None:
                raise RewardSandboxError("reward worker exited after its response")
            _assert_response_pipe_quiet(response_read)
            assert_source_snapshot_unchanged(self.snapshot)
            self._call_count += 1
            self._maximum_call_latency = max(self._maximum_call_latency, latency)
            return outputs
        except BaseException as exc:
            self._invalidate()
            if isinstance(exc, RewardSandboxError):
                raise
            raise RewardSandboxError(f"worker call failed: {type(exc).__name__}: {exc}") from exc

    def _invalidate(self) -> None:
        self._invalid = True
        process = self._process
        if process is not None and process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                process.kill()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process is not None:
            self._exit_status = process.returncode
        self._cleanup()

    def _cleanup(self) -> None:
        for field in ("_request_write", "_response_read"):
            descriptor = getattr(self, field)
            if descriptor is not None:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
                setattr(self, field, None)
        if self._private_cwd is not None:
            shutil.rmtree(self._private_cwd)
            self._private_cwd = None

    def close(self) -> SandboxRuntimeReceiptV1:
        process, request_write, response_read = self._require_live()
        try:
            assert_source_snapshot_unchanged(self.snapshot)
            _write_all(
                request_write,
                _encode_frame(
                    bytes((_sandbox_worker.QUIT_OPCODE,)), maximum_bytes=REQUEST_MAX_BYTES
                ),
            )
            response = _read_frame_with_timeout(
                response_read,
                maximum_bytes=RESPONSE_MAX_BYTES,
                timeout_seconds=STARTUP_TIMEOUT_SECONDS,
            )
            if response != bytes((_sandbox_worker.QUIT_OPCODE,)):
                raise RewardSandboxError("worker did not acknowledge shutdown")
            process.wait(timeout=STARTUP_TIMEOUT_SECONDS)
            if process.returncode != 0:
                raise RewardSandboxError("worker exited unsuccessfully")
            assert_source_snapshot_unchanged(self.snapshot)
            if capture_python_interpreter_identity(self.python_executable) != self.interpreter:
                raise RewardSandboxError("Python interpreter identity drifted during worker use")
            self._exit_status = process.returncode
            self._closed = True
            self._cleanup()
            validator_hash = hashlib.sha256(
                Path(statically_validate_task_term_source.__code__.co_filename).read_bytes()
            ).hexdigest()
            if self._address_space_limit != self.canary_receipt.address_space_limit:
                raise RewardSandboxError(
                    "worker address-space status drifted from its canary receipt"
                )
            return SandboxRuntimeReceiptV1(
                training_seed=self.training_seed,
                source_sha256=self.snapshot.source_sha256,
                validator_source_sha256=validator_hash,
                sandbox_source_sha256=sandbox_source_sha256(),
                worker_source_sha256=sandbox_worker_source_sha256(),
                seatbelt_profile_identity=SEATBELT_PROFILE_ID,
                seatbelt_profile_sha256=self.canary_receipt.profile_sha256,
                canary_receipt_sha256=self.canary_receipt.sha256,
                interpreter=self.interpreter,
                address_space_limit=self._address_space_limit,
                applied_limits=MappingProxyType(
                    {
                        "cpu_seconds": CPU_LIMIT_SECONDS,
                        "file_size_bytes": FILE_SIZE_LIMIT_BYTES,
                        "core_size_bytes": CORE_SIZE_LIMIT_BYTES,
                        "open_files": OPEN_FILE_LIMIT,
                        "startup_timeout_ms": int(STARTUP_TIMEOUT_SECONDS * 1000),
                        "batch_timeout_ms": int(BATCH_TIMEOUT_SECONDS * 1000),
                    }
                ),
                environment_keys=ENVIRONMENT_ALLOWLIST,
                call_count=self._call_count,
                maximum_call_latency_seconds=self._maximum_call_latency,
                exit_status=process.returncode,
            )
        except BaseException as exc:
            self._invalidate()
            if isinstance(exc, RewardSandboxError):
                raise
            raise RewardSandboxError(
                f"worker shutdown failed: {type(exc).__name__}: {exc}"
            ) from exc

    def __enter__(self) -> RewardSandboxWorkerV1:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None and not self._invalid:
            self.close()
        else:
            self._invalidate()


__all__ = [
    "ADDRESS_SPACE_LIMIT_BYTES",
    "BATCH_TIMEOUT_SECONDS",
    "CANARY_NAMES",
    "CPU_LIMIT_SECONDS",
    "DARWIN_ADDRESS_SPACE_LIMIT_REASON",
    "ENVIRONMENT_ALLOWLIST",
    "EXPECTED_BATCH_SIZE",
    "OPEN_FILE_LIMIT",
    "OS_CANARY_NAMES",
    "REQUEST_MAX_BYTES",
    "RESPONSE_MAX_BYTES",
    "RUNTIME_ADMISSION_REFUSAL",
    "SEATBELT_EXECUTABLE",
    "SEATBELT_PROFILE_ID",
    "SELF_CANARY_NAMES",
    "STARTUP_TIMEOUT_SECONDS",
    "STDERR_FIRST_LINE_MAX_BYTES",
    "AddressSpaceLimitReceiptV1",
    "CanaryVerdictV1",
    "CandidateSourceSnapshotV1",
    "PythonInterpreterIdentityV1",
    "RewardSandboxError",
    "RewardSandboxWorkerV1",
    "SandboxRuntimeReceiptV1",
    "SeatbeltCanaryReceiptV1",
    "WorkerExitStatusV1",
    "assert_source_snapshot_unchanged",
    "capture_candidate_source",
    "capture_python_interpreter_identity",
    "run_seatbelt_canaries",
    "sandbox_source_sha256",
    "sandbox_worker_source_sha256",
    "seatbelt_profile_sha256",
    "seatbelt_profile_v3",
]
