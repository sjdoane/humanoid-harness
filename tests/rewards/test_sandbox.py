from __future__ import annotations

import errno
import json
import os
import signal
import socket
import struct
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from oracle_composition.rewards import _sandbox_worker
from oracle_composition.rewards._sandbox_worker import (
    WORKER_ENVIRONMENT as WORKER_PROCESS_ENVIRONMENT,
)
from oracle_composition.rewards.contract import CandidateTaskInputsV1
from oracle_composition.rewards.sandbox import (
    CANARY_NAMES,
    DARWIN_ADDRESS_SPACE_LIMIT_REASON,
    OS_CANARY_NAMES,
    REQUEST_MAX_BYTES,
    SEATBELT_PROFILE_ID,
    SELF_CANARY_NAMES,
    STDERR_FIRST_LINE_MAX_BYTES,
    WORKER_ENVIRONMENT,
    AddressSpaceLimitReceiptV1,
    CanaryVerdictV1,
    RewardSandboxError,
    RewardSandboxWorkerV1,
    SandboxRuntimeReceiptV1,
    SeatbeltCanaryReceiptV1,
    _decode_batch_response,
    _decode_worker_limit_status,
    _encode_batch_request,
    _failed_canary_verdict,
    _protocol_or_identity_self_canary,
    _run_os_canary,
    _validate_frame_length,
    assert_source_snapshot_unchanged,
    capture_candidate_source,
    capture_python_interpreter_identity,
    run_seatbelt_canaries,
    seatbelt_profile_sha256,
    seatbelt_profile_v3,
)

ROOT = Path(__file__).parents[2]
SOURCE = b"def task_term(x):\n    return 1.0 - abs(x.com_x_velocity_m_s - x.target_speed_m_s)\n"
BUILDER_SANDBOX_REASON = "sandbox-exec: sandbox_apply: Operation not permitted"
requires_r2_host = pytest.mark.skip(
    reason="R2 host fixture required; R1 forbids subprocess reward workers and OS canaries"
)


@pytest.fixture(scope="module")
def canaries() -> SeatbeltCanaryReceiptV1:
    return run_seatbelt_canaries(repository_probe=ROOT / "AGENTS.md")


def _require_os_pass_or_skip(receipt: SeatbeltCanaryReceiptV1, *names: str) -> None:
    for name in names:
        result = receipt.canaries[name]
        if result.verdict == "failed":
            pytest.fail(f"OS canary {name} failed: {result.error}")
        if result.verdict == "not_verified_in_builder_sandbox":
            assert result.error == BUILDER_SANDBOX_REASON
            pytest.skip(BUILDER_SANDBOX_REASON)
        assert result.verdict == "passed"
        assert result.error is None
        if name != "memory_abuse":
            assert result.mechanism is None


def _read_limit_status_payload(descriptor: int) -> bytes:
    header = os.read(descriptor, 4)
    assert len(header) == 4
    size = int.from_bytes(header, "big")
    payload = os.read(descriptor, size)
    assert len(payload) == size
    return payload


@requires_r2_host
def test_unresolved_venv_interpreter_imports_package_under_isolated_mode(
    tmp_path: Path,
) -> None:
    identity = capture_python_interpreter_identity()
    expected_launch = ROOT / ".venv" / "bin" / "python"
    assert identity.launch_path == str(expected_launch)
    assert identity.resolved_path == os.path.realpath(expected_launch)
    assert identity.launch_path != identity.resolved_path
    assert identity.launch_file_sha256 == identity.resolved_file_sha256

    completed = subprocess.run(
        [
            identity.launch_path,
            "-I",
            "-c",
            "import oracle_composition; print(oracle_composition.__file__)",
        ],
        cwd=tmp_path,
        env=dict(WORKER_ENVIRONMENT),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=5.0,
        check=False,
        close_fds=True,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    assert str(ROOT / "src" / "oracle_composition") in completed.stdout.decode("utf-8")


def test_seatbelt_profile_v3_adds_only_three_package_directory_reads() -> None:
    identity = capture_python_interpreter_identity()
    project_package = (ROOT / "src" / "oracle_composition").resolve()
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
    expected_rules = [
        "(version 1)",
        "(deny default)",
        f'(allow process-exec (literal "{identity.launch_path}"))',
        "(allow sysctl-read)",
        '(allow file-read* (literal "/dev/null"))',
        "(allow file-read-metadata)",
        '(allow file-read-data (literal "/"))',
        f'(allow file-read* (literal "{identity.launch_path}"))',
    ]
    if identity.resolved_path != identity.launch_path:
        expected_rules.append(f'(allow process-exec (literal "{identity.resolved_path}"))')
    expected_rules.extend(f'(allow file-read* (subpath "{root}"))' for root in sorted(read_roots))
    expected_rules.extend(
        f'(allow file-read-data (literal "{directory}"))' for directory in package_directories
    )
    expected_rules.extend(
        f'(allow file-read-data (literal "{source}"))' for source in exact_sources
    )
    expected_rules.extend(
        (
            f'(allow file-read-metadata (subpath "{project_package}"))',
            "(deny network*)",
            "(deny file-write*)",
            "(deny process-fork)",
            "(deny signal)",
            "(deny process-info*)",
        )
    )

    assert SEATBELT_PROFILE_ID == "reward-worker-seatbelt/v3"
    assert seatbelt_profile_v3().splitlines() == expected_rules


def test_pipe_closed_failure_records_signal_or_worker_exit_code_and_bounded_stderr() -> None:
    crashed = _failed_canary_verdict(
        "worker pipe closed before a complete response",
        returncode=-signal.SIGABRT,
        stderr=b"seatbelt abort\nsecond line is not receipted",
    )
    assert crashed.exit_status is not None
    assert crashed.exit_status.to_dict() == {
        "kind": "signal",
        "value": signal.SIGABRT,
        "meaning": "SIGABRT",
    }
    assert crashed.stderr_first_line == "seatbelt abort"

    refused = _failed_canary_verdict(
        "worker pipe closed before a complete response",
        returncode=_sandbox_worker.EXIT_REQUIRED_DESCRIPTOR_MISSING,
        stderr=b"\x1b" + b"x" * 300 + b"\nignored",
    )
    assert refused.exit_status is not None
    assert refused.exit_status.to_dict() == {
        "kind": "exit_code",
        "value": 64,
        "meaning": "required_invocation_file_descriptor_missing",
    }
    assert refused.stderr_first_line is not None
    assert len(refused.stderr_first_line.encode("utf-8")) <= STDERR_FIRST_LINE_MAX_BYTES
    assert "\n" not in refused.stderr_first_line


def test_darwin_refused_address_limit_is_receipted_and_worker_bootstrap_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, tuple[int, int]]] = []
    worker_started: list[tuple[int, int, int]] = []

    def fake_setrlimit(resource_id: int, value: tuple[int, int]) -> None:
        calls.append((resource_id, value))
        if resource_id == _sandbox_worker.resource.RLIMIT_AS:
            raise ValueError("current limit exceeds maximum limit")

    monkeypatch.setattr(_sandbox_worker.resource, "setrlimit", fake_setrlimit)
    monkeypatch.setattr(_sandbox_worker.sys, "platform", "darwin")
    monkeypatch.setattr(_sandbox_worker, "_sanitize_environment", lambda: None)
    monkeypatch.setattr(
        _sandbox_worker,
        "_run_worker",
        lambda source, request, response: worker_started.append((source, request, response)) or 0,
    )
    status_read, status_write = os.pipe()
    original_umask = os.umask(0o077)
    os.umask(original_umask)
    try:
        monkeypatch.setattr(
            _sandbox_worker.sys,
            "argv",
            [
                "_sandbox_worker",
                "--limit-status-fd",
                str(status_write),
                "--source-fd",
                "100",
                "--request-fd",
                "101",
                "--response-fd",
                "102",
            ],
        )
        assert _sandbox_worker.main() == 0
        address_space_limit, failure = _decode_worker_limit_status(
            _read_limit_status_payload(status_read)
        )
    finally:
        os.umask(original_umask)
        os.close(status_read)
        with pytest.raises(OSError):
            os.close(status_write)

    assert worker_started == [(100, 101, 102)]
    assert failure is None
    assert address_space_limit.to_dict() == {
        "requested_bytes": _sandbox_worker.ADDRESS_SPACE_LIMIT_BYTES,
        "enforced": False,
        "reason": DARWIN_ADDRESS_SPACE_LIMIT_REASON,
    }
    runtime_receipt = SandboxRuntimeReceiptV1(
        training_seed=101,
        source_sha256="a" * 64,
        validator_source_sha256="b" * 64,
        sandbox_source_sha256="c" * 64,
        worker_source_sha256="d" * 64,
        seatbelt_profile_identity=SEATBELT_PROFILE_ID,
        seatbelt_profile_sha256="e" * 64,
        canary_receipt_sha256="f" * 64,
        interpreter=capture_python_interpreter_identity(),
        address_space_limit=address_space_limit,
        applied_limits={
            "cpu_seconds": 900,
            "file_size_bytes": 0,
            "core_size_bytes": 0,
            "open_files": 16,
            "startup_timeout_ms": 5000,
            "batch_timeout_ms": 50,
        },
        environment_keys=("PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED", "PYTHONNOUSERSITE"),
        call_count=0,
        maximum_call_latency_seconds=0.0,
        exit_status=0,
    )
    assert runtime_receipt.to_dict()["address_space_limit"] == {
        "requested_bytes": _sandbox_worker.ADDRESS_SPACE_LIMIT_BYTES,
        "enforced": False,
        "reason": DARWIN_ADDRESS_SPACE_LIMIT_REASON,
    }
    assert runtime_receipt.to_dict()["interpreter"] == (
        capture_python_interpreter_identity().to_dict()
    )
    assert runtime_receipt.to_dict()["worker_exit_codes"]["64"] == (
        "required_invocation_file_descriptor_missing"
    )
    assert calls == [
        (
            _sandbox_worker.resource.RLIMIT_AS,
            (
                _sandbox_worker.ADDRESS_SPACE_LIMIT_BYTES,
                _sandbox_worker.ADDRESS_SPACE_LIMIT_BYTES,
            ),
        ),
        (
            _sandbox_worker.resource.RLIMIT_CPU,
            (_sandbox_worker.CPU_LIMIT_SECONDS, _sandbox_worker.CPU_LIMIT_SECONDS),
        ),
        (
            _sandbox_worker.resource.RLIMIT_FSIZE,
            (_sandbox_worker.FILE_SIZE_LIMIT_BYTES, _sandbox_worker.FILE_SIZE_LIMIT_BYTES),
        ),
        (
            _sandbox_worker.resource.RLIMIT_CORE,
            (_sandbox_worker.CORE_SIZE_LIMIT_BYTES, _sandbox_worker.CORE_SIZE_LIMIT_BYTES),
        ),
        (
            _sandbox_worker.resource.RLIMIT_NOFILE,
            (_sandbox_worker.OPEN_FILE_LIMIT, _sandbox_worker.OPEN_FILE_LIMIT),
        ),
    ]


@pytest.mark.parametrize(
    ("failing_resource", "expected_failure"),
    (
        (_sandbox_worker.resource.RLIMIT_CPU, "cpu_limit"),
        (_sandbox_worker.resource.RLIMIT_FSIZE, "file_size_limit"),
        (_sandbox_worker.resource.RLIMIT_CORE, "core_size_limit"),
        (_sandbox_worker.resource.RLIMIT_NOFILE, "open_file_limit"),
    ),
)
def test_failure_of_any_non_address_worker_limit_is_fatal(
    monkeypatch: pytest.MonkeyPatch,
    failing_resource: int,
    expected_failure: str,
) -> None:
    calls: list[int] = []
    worker_started: list[bool] = []

    def fake_setrlimit(resource_id: int, value: tuple[int, int]) -> None:
        del value
        calls.append(resource_id)
        if resource_id == _sandbox_worker.resource.RLIMIT_AS:
            raise ValueError("current limit exceeds maximum limit")
        if resource_id == failing_resource:
            raise OSError("mandatory limit refused")

    monkeypatch.setattr(_sandbox_worker.resource, "setrlimit", fake_setrlimit)
    monkeypatch.setattr(_sandbox_worker.sys, "platform", "darwin")
    monkeypatch.setattr(_sandbox_worker, "_sanitize_environment", lambda: None)
    monkeypatch.setattr(
        _sandbox_worker,
        "_run_worker",
        lambda *_args: worker_started.append(True) or 0,
    )
    status_read, status_write = os.pipe()
    original_umask = os.umask(0o077)
    os.umask(original_umask)
    try:
        monkeypatch.setattr(
            _sandbox_worker.sys,
            "argv",
            [
                "_sandbox_worker",
                "--limit-status-fd",
                str(status_write),
                "--source-fd",
                "100",
                "--request-fd",
                "101",
                "--response-fd",
                "102",
            ],
        )
        assert _sandbox_worker.main() == 66
        address_space_limit, failure = _decode_worker_limit_status(
            _read_limit_status_payload(status_read)
        )
    finally:
        os.umask(original_umask)
        os.close(status_read)
        with pytest.raises(OSError):
            os.close(status_write)

    assert worker_started == []
    mandatory_order = [
        _sandbox_worker.resource.RLIMIT_AS,
        _sandbox_worker.resource.RLIMIT_CPU,
        _sandbox_worker.resource.RLIMIT_FSIZE,
        _sandbox_worker.resource.RLIMIT_CORE,
        _sandbox_worker.resource.RLIMIT_NOFILE,
    ]
    assert calls == mandatory_order[: mandatory_order.index(failing_resource) + 1]
    assert address_space_limit.to_dict() == {
        "requested_bytes": _sandbox_worker.ADDRESS_SPACE_LIMIT_BYTES,
        "enforced": False,
        "reason": DARWIN_ADDRESS_SPACE_LIMIT_REASON,
    }
    assert failure == expected_failure


def test_source_snapshot_toctou_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "candidate.py"
    path.write_bytes(SOURCE)
    snapshot = capture_candidate_source(path)
    path.write_bytes(SOURCE + b"\n")
    with pytest.raises(RewardSandboxError, match="drifted"):
        assert_source_snapshot_unchanged(snapshot)


def test_network_probe_requires_eperm_from_connect_and_bind_on_fresh_sockets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, tuple[str, int]]] = []
    created: list[object] = []

    class DeniedSocket:
        def __init__(self) -> None:
            self.index = len(created)
            created.append(self)

        def connect(self, address: tuple[str, int]) -> None:
            calls.append(("connect", self.index, address))
            raise PermissionError(errno.EPERM, "operation not permitted")

        def bind(self, address: tuple[str, int]) -> None:
            calls.append(("bind", self.index, address))
            raise OSError(errno.EPERM, "operation not permitted")

        def close(self) -> None:
            return None

    def denied_socket(family: int, kind: int) -> DeniedSocket:
        assert family == socket.AF_INET
        assert kind == socket.SOCK_STREAM
        return DeniedSocket()

    monkeypatch.setattr(socket, "socket", denied_socket)
    assert _sandbox_worker._run_network_canary() == _sandbox_worker.EXIT_OK
    assert len(created) == 2
    assert calls == [
        ("connect", 0, ("127.0.0.1", 9)),
        ("bind", 1, ("127.0.0.1", 0)),
    ]


def test_network_probe_rejects_connection_refusal_and_successful_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[str, int]]] = []

    class PermissiveSocket:
        def connect(self, address: tuple[str, int]) -> None:
            calls.append(("connect", address))
            raise ConnectionRefusedError(errno.ECONNREFUSED, "connection refused")

        def bind(self, address: tuple[str, int]) -> None:
            calls.append(("bind", address))

        def close(self) -> None:
            return None

    monkeypatch.setattr(socket, "socket", lambda *_args: PermissiveSocket())
    assert _sandbox_worker._run_network_canary() == _sandbox_worker.EXIT_CANARY_NOT_CONTAINED
    assert calls == [
        ("connect", ("127.0.0.1", 9)),
        ("bind", ("127.0.0.1", 0)),
    ]


@requires_r2_host
def test_worker_timeout_kills_infinite_loop_without_fallback(
    canaries: SeatbeltCanaryReceiptV1,
) -> None:
    _require_os_pass_or_skip(canaries, "timeout")


@requires_r2_host
def test_memory_bomb_hits_limit_without_parent_failure(canaries: SeatbeltCanaryReceiptV1) -> None:
    _require_os_pass_or_skip(canaries, "memory_abuse")
    expected_mechanism = "rlimit_as" if canaries.address_space_limit.enforced else "timeout_kill"
    assert canaries.canaries["memory_abuse"].mechanism == expected_mechanism


@requires_r2_host
def test_file_read_file_write_env_secret_socket_and_exec_are_denied(
    canaries: SeatbeltCanaryReceiptV1,
) -> None:
    _require_os_pass_or_skip(
        canaries,
        "file_read",
        "file_write",
        "environment_secret",
        "network",
        "process_creation",
    )


@requires_r2_host
def test_fork_signal_trace_and_repository_read_are_denied(
    canaries: SeatbeltCanaryReceiptV1,
) -> None:
    _require_os_pass_or_skip(canaries, "fork", "signal", "tracing", "repository_read")


@pytest.mark.parametrize(
    ("name", "deny_rule", "allow_rule"),
    (
        ("network", "(deny network*)", "(allow network*)"),
        ("tracing", "(deny process-info*)", "(allow process-info*)"),
    ),
)
@requires_r2_host
def test_governed_probe_reports_not_contained_under_permissive_profile(
    canaries: SeatbeltCanaryReceiptV1,
    tmp_path: Path,
    name: str,
    deny_rule: str,
    allow_rule: str,
) -> None:
    _require_os_pass_or_skip(canaries, name)
    restricted_profile = seatbelt_profile_v3()
    assert restricted_profile.count(deny_rule) == 1
    permissive_profile = restricted_profile.replace(deny_rule, allow_rule)

    verdict, address_space_limit = _run_os_canary(
        name,
        profile=permissive_profile,
        private_cwd=tmp_path,
        repository_probe=ROOT / "AGENTS.md",
        python_executable=Path(canaries.interpreter.launch_path),
        sandbox_executable=Path(canaries.sandbox_executable),
        expected_address_space_limit=canaries.address_space_limit,
    )

    assert address_space_limit == canaries.address_space_limit
    assert verdict.verdict == "failed"
    assert verdict.error == "OS canary operation was not contained or its invocation failed"
    assert verdict.exit_status is not None
    assert verdict.exit_status.to_dict() == {
        "kind": "exit_code",
        "value": _sandbox_worker.EXIT_CANARY_NOT_CONTAINED,
        "meaning": "os_canary_operation_was_not_denied",
    }


@requires_r2_host
def test_stdout_private_fd_and_extra_frame_injection_are_denied(
    canaries: SeatbeltCanaryReceiptV1,
) -> None:
    _require_os_pass_or_skip(canaries, "stdout_injection", "private_fd_injection", "extra_frame")


@requires_r2_host
def test_pickle_object_dtype_and_oversize_frame_are_rejected(
    canaries: SeatbeltCanaryReceiptV1,
) -> None:
    for name in ("pickle", "object_dtype", "oversize_frame"):
        assert canaries.canaries[name] == CanaryVerdictV1("passed", None)
    with pytest.raises(RewardSandboxError):
        _decode_batch_response(b"\x80\x05pickle")
    with pytest.raises(RewardSandboxError):
        _encode_batch_request([object()] * 4)  # type: ignore[list-item]
    with pytest.raises(RewardSandboxError):
        _validate_frame_length(REQUEST_MAX_BYTES + 1, maximum_bytes=REQUEST_MAX_BYTES)


@requires_r2_host
def test_worker_crash_and_malformed_response_invalidate_seed(
    canaries: SeatbeltCanaryReceiptV1,
) -> None:
    _require_os_pass_or_skip(canaries, "worker_crash")
    assert canaries.canaries["malformed_response"] == CanaryVerdictV1("passed", None)
    with pytest.raises(RewardSandboxError):
        _decode_batch_response(b"V")
    with pytest.raises(RewardSandboxError):
        _decode_batch_response(bytes((0x45, 2)))


@requires_r2_host
def test_missing_seatbelt_or_failed_canary_fails_closed(tmp_path: Path) -> None:
    receipt = run_seatbelt_canaries(
        repository_probe=ROOT / "AGENTS.md",
        sandbox_executable=tmp_path / "missing-sandbox-exec",
    )
    assert {name: receipt.canaries[name].verdict for name in OS_CANARY_NAMES} == dict.fromkeys(
        OS_CANARY_NAMES, "failed"
    )
    assert {name: receipt.canaries[name].verdict for name in SELF_CANARY_NAMES} == dict.fromkeys(
        SELF_CANARY_NAMES, "passed"
    )
    with pytest.raises(RewardSandboxError, match="cannot start"):
        receipt.assert_all_passed()

    source = tmp_path / "candidate.py"
    source.write_bytes(SOURCE)
    with pytest.raises(RewardSandboxError, match="cannot start"):
        RewardSandboxWorkerV1(
            training_seed=101,
            candidate_source_path=source,
            canary_receipt=receipt,
        )


def test_binary_batch_schema_is_exactly_four_finite_scalar_pairs() -> None:
    inputs = [CandidateTaskInputsV1(float(index), 1.0) for index in range(4)]
    encoded = _encode_batch_request(inputs)
    assert int.from_bytes(encoded[:4], "big") == struct.calcsize(">BB8d")
    assert len(encoded) < REQUEST_MAX_BYTES
    response = bytes((0x56,)) + struct.pack(">4d", 0.0, 1.0, 2.0, 3.0)
    assert _decode_batch_response(response) == (0.0, 1.0, 2.0, 3.0)
    with pytest.raises(RewardSandboxError, match="exactly four"):
        _encode_batch_request(inputs[:3])


@pytest.mark.parametrize(
    "name",
    (
        "source_drift",
        "pickle",
        "object_dtype",
        "oversize_frame",
        "extra_frame",
        "malformed_response",
    ),
)
def test_protocol_and_source_identity_canaries_exercise_real_rejection_paths(name: str) -> None:
    assert _protocol_or_identity_self_canary(name) == CanaryVerdictV1("passed", None)


@requires_r2_host
def test_builder_sandbox_records_every_canary_without_fabricating_pass(
    canaries: SeatbeltCanaryReceiptV1,
) -> None:
    _require_os_pass_or_skip(canaries, *OS_CANARY_NAMES)
    assert tuple(canaries.canaries) == CANARY_NAMES
    assert canaries.all_passed
    canaries.assert_all_passed()
    recorded = json.loads(
        (
            ROOT
            / "experiments"
            / "family_b_target_speed_v1"
            / "receipts"
            / "builder_sandbox_canaries.json"
        ).read_text(encoding="utf-8")
    )
    assert recorded["all_passed"] is False
    assert recorded["profile_identity"] == SEATBELT_PROFILE_ID
    assert recorded["profile_sha256"] == canaries.profile_sha256
    assert recorded["interpreter"] == canaries.interpreter.to_dict()
    assert recorded["worker_exit_codes"]["64"] == ("required_invocation_file_descriptor_missing")
    assert recorded["address_space_limit"] == canaries.address_space_limit.to_dict()


def test_recorded_builder_receipt_separates_os_nonverification_from_self_canaries() -> None:
    recorded = json.loads(
        (
            ROOT
            / "experiments"
            / "family_b_target_speed_v1"
            / "receipts"
            / "builder_sandbox_canaries.json"
        ).read_text(encoding="utf-8")
    )
    assert recorded["address_space_limit"] == {
        "requested_bytes": _sandbox_worker.ADDRESS_SPACE_LIMIT_BYTES,
        "enforced": False,
        "reason": DARWIN_ADDRESS_SPACE_LIMIT_REASON,
    }
    assert {name: recorded["canaries"][name] for name in OS_CANARY_NAMES} == {
        name: {
            "verdict": "not_verified_in_builder_sandbox",
            "error": BUILDER_SANDBOX_REASON,
            "mechanism": None,
        }
        for name in OS_CANARY_NAMES
    }
    assert {name: recorded["canaries"][name] for name in SELF_CANARY_NAMES} == {
        name: {
            "verdict": "passed",
            "error": None,
            "mechanism": None,
        }
        for name in SELF_CANARY_NAMES
    }
    assert recorded["all_passed"] is False


@requires_r2_host
def test_canary_receipt_binds_profile_and_has_no_inherited_home_or_path(
    canaries: SeatbeltCanaryReceiptV1,
) -> None:
    identity = capture_python_interpreter_identity()
    assert canaries.profile_identity == SEATBELT_PROFILE_ID
    assert canaries.profile_sha256 == seatbelt_profile_sha256()
    assert canaries.interpreter == identity
    assert canaries.to_dict()["interpreter"] == identity.to_dict()
    assert canaries.to_dict()["worker_exit_codes"] == {
        str(code): meaning
        for code, meaning in sorted(_sandbox_worker.WORKER_EXIT_CODE_MEANINGS.items())
    }
    assert dict(WORKER_ENVIRONMENT) == WORKER_PROCESS_ENVIRONMENT
    assert set(WORKER_ENVIRONMENT) == {
        "PYTHONNOUSERSITE",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
    }
    assert "HOME" not in WORKER_ENVIRONMENT
    assert "PATH" not in WORKER_ENVIRONMENT
    assert canaries.address_space_limit.to_dict() == {
        "requested_bytes": _sandbox_worker.ADDRESS_SPACE_LIMIT_BYTES,
        "enforced": False,
        "reason": DARWIN_ADDRESS_SPACE_LIMIT_REASON,
    }
    assert canaries.sha256 == canaries.sha256


def test_failed_canary_receipt_cannot_claim_all_passed() -> None:
    receipt = SeatbeltCanaryReceiptV1(
        sandbox_executable="/usr/bin/sandbox-exec",
        profile_identity=SEATBELT_PROFILE_ID,
        profile_sha256="a" * 64,
        interpreter=capture_python_interpreter_identity(),
        address_space_limit=AddressSpaceLimitReceiptV1(
            requested_bytes=_sandbox_worker.ADDRESS_SPACE_LIMIT_BYTES,
            enforced=False,
            reason=DARWIN_ADDRESS_SPACE_LIMIT_REASON,
        ),
        canaries={name: CanaryVerdictV1("failed", "denied") for name in CANARY_NAMES},
    )
    assert receipt.all_passed is False


def test_memory_canary_cannot_claim_pass_without_the_mechanism_that_acted() -> None:
    address_space_limit = AddressSpaceLimitReceiptV1(
        requested_bytes=_sandbox_worker.ADDRESS_SPACE_LIMIT_BYTES,
        enforced=False,
        reason=DARWIN_ADDRESS_SPACE_LIMIT_REASON,
    )
    without_mechanism = {name: CanaryVerdictV1("passed", None) for name in CANARY_NAMES}
    with pytest.raises(RewardSandboxError, match="mechanism that acted"):
        SeatbeltCanaryReceiptV1(
            sandbox_executable="/usr/bin/sandbox-exec",
            profile_identity=SEATBELT_PROFILE_ID,
            profile_sha256="a" * 64,
            interpreter=capture_python_interpreter_identity(),
            address_space_limit=address_space_limit,
            canaries=without_mechanism,
        )

    with_mechanism = dict(without_mechanism)
    with_mechanism["memory_abuse"] = CanaryVerdictV1(
        "passed",
        None,
        mechanism="timeout_kill",
    )
    receipt = SeatbeltCanaryReceiptV1(
        sandbox_executable="/usr/bin/sandbox-exec",
        profile_identity=SEATBELT_PROFILE_ID,
        profile_sha256="a" * 64,
        interpreter=capture_python_interpreter_identity(),
        address_space_limit=address_space_limit,
        canaries=with_mechanism,
    )
    assert receipt.canaries["memory_abuse"].mechanism == "timeout_kill"
