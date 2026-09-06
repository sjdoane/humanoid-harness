#!/usr/bin/env python3
"""Plan or supervise one admitted ten-second GMT reconstructed-actor probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import shutil
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from oracle_composition.adapters.gmt.checkpoint import verify_upstream_root
from oracle_composition.adapters.gmt.contracts import GMT_UPSTREAM_COMMIT, MOTION_SPECS
from oracle_composition.adapters.gmt.io import (
    GMTAdmissionError,
    sha256_file,
    write_json_receipt,
)
from oracle_composition.harness.resource_slot import (
    ResourceSlotError,
    canonical_json_bytes,
    load_validated_reservation,
    release_slot,
    reserve_slot,
    validate_slot_for_reservation,
)
from oracle_composition.phase_b.supervision import (
    _directory_bytes,
    _process_tree_rss_bytes,
    cleanup_worker_process,
)

ACCEPTED_SMOKE_WALL_SECONDS = 1_200
MAX_DEVELOPMENT_WALL_SECONDS = 1_200
MAX_DEVELOPMENT_CPU_SECONDS = 1_200
MAX_DEVELOPMENT_OUTPUT_BYTES = 1 * 1024**3
MAX_RSS_BYTES = 8 * 1024**3
MIN_FREE_DISK_BYTES = 20 * 1024**3
RECEIPT_RESERVE_BYTES = 512 * 1024
POLL_SECONDS = 0.05
GROUP_VALIDATION_SECONDS = 2.0
OFFICIAL_ACTOR_SHA256 = "bc444fbd56ba4a582d7c6367504f2093ccb081c6956fcee8f30f2e85ced28686"
BASELINE_ARTIFACT_LABEL = "gmt_reconstructed_baseline_probe_resource_receipt"
BASELINE_EVIDENCE_CLASS = "exploratory_baseline_no_learning_not_jit_equivalence_or_task_success"
TRACE_FILENAME = "gmt_reconstructed_baseline_10s.npz"
RECEIPT_FILENAME = "gmt_probe_resource_receipt_v1.json"
STDOUT_FILENAME = "child_stdout.json"
STDERR_FILENAME = "child_stderr.log"
SOURCE_PATHS = (
    "scripts/run_gmt_probe.py",
    "src/oracle_composition/__init__.py",
    "src/oracle_composition/adapters/gmt/__init__.py",
    "src/oracle_composition/adapters/gmt/__main__.py",
    "src/oracle_composition/adapters/gmt/actor.py",
    "src/oracle_composition/adapters/gmt/checkpoint.py",
    "src/oracle_composition/adapters/gmt/contracts.py",
    "src/oracle_composition/adapters/gmt/deployment.py",
    "src/oracle_composition/adapters/gmt/io.py",
    "src/oracle_composition/adapters/gmt/motions.py",
    "src/oracle_composition/adapters/gmt/reference_math.py",
    "src/oracle_composition/adapters/gmt/reference_runtime.py",
    "src/oracle_composition/adapters/gmt/replay.py",
    "src/oracle_composition/harness/resource_slot.py",
    "src/oracle_composition/phase_b/supervision.py",
    "uv.lock",
)
THREAD_ENVIRONMENT = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


class ProbeError(RuntimeError):
    """A probe precondition, resource gate, or terminal check failed."""


@dataclass(frozen=True)
class ProbeLimits:
    """Validated process ceilings for one immutable trusted plan."""

    wall_seconds: int = 120
    cpu_seconds: int = 120
    rss_bytes: int = 8 * 1024**3
    free_disk_bytes: int = 20 * 1024**3
    output_bytes: int = 128 * 1024**2

    def __post_init__(self) -> None:
        fields = {
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "rss_bytes": self.rss_bytes,
            "free_disk_bytes": self.free_disk_bytes,
            "output_bytes": self.output_bytes,
        }
        if any(type(value) is not int or value <= 0 for value in fields.values()):
            raise ValueError("probe limits must be positive integers")
        if self.wall_seconds > MAX_DEVELOPMENT_WALL_SECONDS:
            raise ValueError("probe wall limit exceeds the 1200-second development maximum")
        if self.cpu_seconds > MAX_DEVELOPMENT_CPU_SECONDS:
            raise ValueError("probe CPU limit exceeds the 1200-second development maximum")
        if self.rss_bytes > MAX_RSS_BYTES:
            raise ValueError("probe RSS limit exceeds 8 GiB")
        if self.free_disk_bytes < MIN_FREE_DISK_BYTES:
            raise ValueError("probe free-disk floor cannot be lower than 20 GiB")
        if not RECEIPT_RESERVE_BYTES < self.output_bytes <= MAX_DEVELOPMENT_OUTPUT_BYTES:
            raise ValueError("probe output limit must reserve a receipt and cannot exceed 1 GiB")

    @property
    def child_output_bytes(self) -> int:
        return self.output_bytes - RECEIPT_RESERVE_BYTES

    def as_dict(self) -> dict[str, int]:
        return {
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "rss_bytes": self.rss_bytes,
            "free_disk_bytes": self.free_disk_bytes,
            "output_bytes": self.output_bytes,
            "child_output_bytes": self.child_output_bytes,
        }


DEFAULT_PROBE_LIMITS = ProbeLimits()
EFFECTIVE_WALL_SECONDS = DEFAULT_PROBE_LIMITS.wall_seconds
CPU_TIME_SECONDS = DEFAULT_PROBE_LIMITS.cpu_seconds
RSS_BYTES = DEFAULT_PROBE_LIMITS.rss_bytes
FREE_DISK_BYTES = DEFAULT_PROBE_LIMITS.free_disk_bytes
OUTPUT_BYTES = DEFAULT_PROBE_LIMITS.output_bytes
CHILD_OUTPUT_BYTES = DEFAULT_PROBE_LIMITS.child_output_bytes


class ProcessLike(Protocol):
    pid: int
    returncode: int | None

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


class CleanupProcessAdapter:
    """Expose a Popen through the existing process-group cleanup protocol."""

    def __init__(self, process: ProcessLike) -> None:
        self._process = process
        self.pid = process.pid

    def is_alive(self) -> bool:
        return self._process.poll() is None

    def terminate(self) -> None:
        self._process.terminate()

    def kill(self) -> None:
        self._process.kill()

    def join(self, timeout: float = 0.0) -> None:
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return


@dataclass(frozen=True)
class ProbeConfig:
    repository_root: Path
    venv_python: Path
    upstream_root: Path
    weights_path: Path
    weights_sha256: str
    motion_path: Path
    motion_sha256: str
    motion_name: str
    output_directory: Path
    owner: str


@dataclass(frozen=True)
class ProbePlan:
    config: ProbeConfig
    commit: str
    canonical_argv: tuple[str, ...]
    inputs: dict[str, object]
    limits: ProbeLimits = DEFAULT_PROBE_LIMITS
    artifact_label: str = BASELINE_ARTIFACT_LABEL
    evidence_class: str = BASELINE_EVIDENCE_CLASS

    def __post_init__(self) -> None:
        if type(self.limits) is not ProbeLimits:
            raise ValueError("probe plan limits must be a validated ProbeLimits")
        for label, value in (
            ("artifact label", self.artifact_label),
            ("evidence class", self.evidence_class),
        ):
            if (
                type(value) is not str
                or value != value.strip()
                or not value
                or len(value.encode("utf-8")) > 256
                or any(ord(character) < 32 for character in value)
            ):
                raise ValueError(f"probe {label} is empty, unbounded, or contains controls")
        if "process_supervision" in self.inputs:
            raise ValueError("probe inputs reserve process_supervision for accepted binding")
        baseline_artifact = self.artifact_label == BASELINE_ARTIFACT_LABEL
        baseline_evidence = self.evidence_class == BASELINE_EVIDENCE_CLASS
        if baseline_artifact != baseline_evidence:
            raise ValueError("baseline artifact and evidence labels must change together")
        if self.limits != DEFAULT_PROBE_LIMITS and baseline_artifact:
            raise ValueError("nondefault limits require explicit nonbaseline evidence labels")

    def supervision_binding(self) -> dict[str, object]:
        return {
            "artifact_label": self.artifact_label,
            "evidence_class": self.evidence_class,
            "limits": self.limits.as_dict(),
        }

    def accepted_fields(self) -> dict[str, object]:
        baseline_binding = {
            "artifact_label": BASELINE_ARTIFACT_LABEL,
            "evidence_class": BASELINE_EVIDENCE_CLASS,
            "limits": DEFAULT_PROBE_LIMITS.as_dict(),
        }
        inputs = self.inputs
        if self.supervision_binding() != baseline_binding:
            inputs = {**self.inputs, "process_supervision": self.supervision_binding()}
        return {
            "canonical_argv": list(self.canonical_argv),
            "commit": self.commit,
            "conflict_check": "no_other_heavy_repository_job",
            "hard_wall_seconds": ACCEPTED_SMOKE_WALL_SECONDS,
            "inputs": inputs,
            "mode": "smoke",
            "output": str(self.config.output_directory),
            "owner": self.config.owner,
        }


@dataclass(frozen=True)
class ProbeDependencies:
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    rss_bytes: Callable[[int], int | None] = _process_tree_rss_bytes
    free_disk_bytes: Callable[[Path], int] = lambda path: shutil.disk_usage(path).free
    directory_bytes: Callable[[Path], int] = _directory_bytes
    spawn: Callable[..., ProcessLike] = subprocess.Popen
    cleanup: Callable[..., None] = cleanup_worker_process
    reserve: Callable[..., dict[str, object]] = reserve_slot
    validate_slot: Callable[..., dict[str, object]] = validate_slot_for_reservation
    release: Callable[..., dict[str, object]] = release_slot
    validate_plan: Callable[[ProbePlan], None] = lambda plan: _plan_unchanged(plan)
    verify_completed: Callable[[ProbePlan], dict[str, object]] = lambda plan: (
        _verify_completed_probe(plan)
    )
    validate_group: Callable[[int], bool] = lambda pid: (
        os.getpgid(pid) == pid and os.getsid(pid) == pid
    )


def _run_git(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ProbeError(f"Git preflight failed: {result.stderr.strip()[:500]}")
    return result.stdout.strip()


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _regular_file(path: Path, *, name: str) -> Path:
    absolute = _absolute(path)
    try:
        metadata = absolute.lstat()
    except OSError as exc:
        raise ProbeError(f"{name} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ProbeError(f"{name} must be a direct regular file")
    return absolute


def _venv_identity(path: Path) -> dict[str, object]:
    executable = _absolute(path)
    if executable.name not in {"python", "python3"} or executable.parent.name != "bin":
        raise ProbeError("Python must be the venv bin/python executable")
    venv = executable.parent.parent
    if venv.name != ".venv" or not os.access(executable, os.X_OK):
        raise ProbeError("Python must be an executable from a .venv")
    pyvenv_cfg = _regular_file(venv / "pyvenv.cfg", name="pyvenv.cfg")
    if _absolute(Path(sys.executable)) != executable or _absolute(Path(sys.prefix)) != venv:
        raise ProbeError("launcher itself must run under the selected .venv executable")
    resolved = executable.resolve(strict=True)
    return {
        "path": str(executable),
        "realpath": str(resolved),
        "executable_sha256": sha256_file(resolved),
        "pyvenv_cfg_sha256": sha256_file(pyvenv_cfg),
    }


def _repository_identity(root: Path) -> tuple[Path, str, dict[str, str]]:
    root = _absolute(root)
    if Path(_run_git(root, "rev-parse", "--show-toplevel")) != root:
        raise ProbeError("repository root differs from Git's top level")
    commit = _run_git(root, "rev-parse", "HEAD")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ProbeError("Git HEAD is not a full lowercase commit")
    if _run_git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ProbeError("GMT probe requires a clean source worktree")
    identities: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        source = _regular_file(root / relative, name=f"source {relative}")
        identities[relative] = sha256_file(source)
    return root, commit, identities


def _canonical_child_argv(config: ProbeConfig) -> tuple[str, ...]:
    trace = config.output_directory / TRACE_FILENAME
    return (
        str(config.venv_python),
        "-m",
        "oracle_composition.adapters.gmt",
        "replay",
        "--upstream-root",
        str(config.upstream_root),
        "--weights",
        str(config.weights_path),
        "--weights-sha256",
        config.weights_sha256,
        "--motion",
        str(config.motion_path),
        "--motion-sha256",
        config.motion_sha256,
        "--motion-name",
        config.motion_name,
        "--trace",
        str(trace),
        "--duration-seconds",
        "10",
    )


def build_probe_plan(config: ProbeConfig) -> ProbePlan:
    root, commit, source_files = _repository_identity(config.repository_root)
    output = _absolute(config.output_directory)
    if output.exists():
        raise ProbeError("probe output must not already exist")
    if output.is_relative_to(root):
        raise ProbeError("probe output must be outside the source worktree")
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        raise ProbeError("probe output parent must be a real existing directory")
    venv_python = _absolute(config.venv_python)
    upstream = _absolute(config.upstream_root)
    weights = _regular_file(config.weights_path, name="converted actor weights")
    motion = _regular_file(config.motion_path, name="converted motion")
    if config.weights_sha256 != OFFICIAL_ACTOR_SHA256:
        raise ProbeError("weights digest is not the admitted GMT baseline actor")
    if sha256_file(weights) != config.weights_sha256:
        raise ProbeError("converted actor weight bytes differ")
    if config.motion_name not in MOTION_SPECS:
        raise ProbeError("motion name is outside the pinned GMT catalog")
    if sha256_file(motion) != config.motion_sha256:
        raise ProbeError("converted motion bytes differ")
    normalized = ProbeConfig(
        repository_root=root,
        venv_python=venv_python,
        upstream_root=upstream,
        weights_path=weights,
        weights_sha256=config.weights_sha256,
        motion_path=motion,
        motion_sha256=config.motion_sha256,
        motion_name=config.motion_name,
        output_directory=output,
        owner=config.owner,
    )
    inputs: dict[str, object] = {
        "artifact_contract": "gmt_reconstructed_actor_baseline_no_learning/v1",
        "duration_seconds": 10,
        "motion": {
            "name": normalized.motion_name,
            "path": str(normalized.motion_path),
            "sha256": normalized.motion_sha256,
            "size": normalized.motion_path.stat().st_size,
        },
        "repository_sources": source_files,
        "support_files": verify_upstream_root(normalized.upstream_root),
        "upstream_commit": GMT_UPSTREAM_COMMIT,
        "venv": _venv_identity(normalized.venv_python),
        "weights": {
            "path": str(normalized.weights_path),
            "sha256": normalized.weights_sha256,
            "size": normalized.weights_path.stat().st_size,
        },
    }
    return ProbePlan(
        config=normalized,
        commit=commit,
        canonical_argv=_canonical_child_argv(normalized),
        inputs=inputs,
    )


def validate_reservation_binding(plan: ProbePlan, reservation: Mapping[str, object]) -> None:
    expected = plan.accepted_fields()
    observed = {key: reservation.get(key) for key in expected}
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        raise ProbeError("reservation does not bind the exact GMT probe plan")


def _child_environment(plan: ProbePlan) -> dict[str, str]:
    inherited = {
        name: os.environ[name]
        for name in ("HOME", "LANG", "LC_ALL", "SSL_CERT_DIR", "SSL_CERT_FILE", "TMPDIR")
        if name in os.environ
    }
    return {
        **inherited,
        **THREAD_ENVIRONMENT,
        "PATH": f"{plan.config.venv_python.parent}:/usr/bin:/bin:/usr/sbin",
        "PYTHONPATH": str(plan.config.repository_root / "src"),
        "VIRTUAL_ENV": str(plan.config.venv_python.parent.parent),
    }


def _apply_child_limits(limits: ProbeLimits = DEFAULT_PROBE_LIMITS) -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits.output_bytes, limits.output_bytes))


def _spawn_child(
    plan: ProbePlan,
    *,
    stdout: Any,
    stderr: Any,
    dependencies: ProbeDependencies,
) -> ProcessLike:
    def apply_child_limits() -> None:
        _apply_child_limits(plan.limits)

    return dependencies.spawn(
        list(plan.canonical_argv),
        cwd=plan.config.repository_root,
        env=_child_environment(plan),
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
        close_fds=True,
        preexec_fn=apply_child_limits,
    )


def _validate_process_group(
    process: ProcessLike,
    dependencies: ProbeDependencies,
) -> bool:
    deadline = dependencies.clock() + GROUP_VALIDATION_SECONDS
    while dependencies.clock() < deadline:
        try:
            if dependencies.validate_group(process.pid):
                return True
        except (OSError, ProcessLookupError):
            pass
        if process.poll() is not None:
            break
        dependencies.sleep(POLL_SECONDS)
    return False


def _read_bounded(path: Path, maximum: int, label: str) -> bytes:
    if path.is_symlink() or not path.is_file() or not 0 <= path.stat().st_size <= maximum:
        raise ProbeError(f"{label} is missing or exceeds its bound")
    payload = path.read_bytes()
    if len(payload) != path.stat().st_size:
        raise ProbeError(f"{label} changed while read")
    return payload


def _json_object(payload: bytes, label: str) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, child in items:
            if key in value:
                raise ProbeError(f"{label} has duplicate keys")
            value[key] = child
        return value

    def reject_constant(value: str) -> None:
        raise ProbeError(f"{label} contains non-finite JSON constant {value}")

    try:
        value = json.loads(payload, object_pairs_hook=pairs, parse_constant=reject_constant)
    except (UnicodeError, ValueError) as exc:
        raise ProbeError(f"{label} is malformed JSON") from exc
    if type(value) is not dict:
        raise ProbeError(f"{label} must be a JSON object")
    return value


def _verify_completed_probe(plan: ProbePlan) -> dict[str, object]:
    output = plan.config.output_directory
    trace = output / TRACE_FILENAME
    manifest_path = trace.with_suffix(".npz.manifest.json")
    stdout_path = output / STDOUT_FILENAME
    stderr_path = output / STDERR_FILENAME
    names = {path.name for path in output.iterdir()}
    expected_names = {trace.name, manifest_path.name, stdout_path.name, stderr_path.name}
    if names != expected_names:
        raise ProbeError("probe output file set differs")
    if trace.is_symlink() or not trace.is_file():
        raise ProbeError("replay trace is not a direct regular file")
    stdout = _json_object(_read_bounded(stdout_path, 128 * 1024, "child stdout"), "child stdout")
    stderr = _read_bounded(stderr_path, 1024 * 1024, "child stderr")
    manifest_bytes = _read_bounded(manifest_path, 128 * 1024, "replay manifest")
    manifest = _json_object(manifest_bytes, "replay manifest")
    trace_sha256 = sha256_file(trace)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    expected_inputs = {
        "upstream_commit": GMT_UPSTREAM_COMMIT,
        "weights_sha256": plan.config.weights_sha256,
        "motion_name": plan.config.motion_name,
        "motion_sha256": plan.config.motion_sha256,
        "support_files": plan.inputs["support_files"],
    }
    limits = manifest.get("limits")
    metrics = manifest.get("metrics")
    trace_record = manifest.get("trace")
    if (
        manifest.get("artifact") != "gmt_g1_reconstructed_actor_headless_replay"
        or manifest.get("claim_status") != "reconstructed_plain_actor_not_jit_equivalent"
        or manifest.get("inputs") != expected_inputs
        or type(limits) is not dict
        or limits.get("original_jit_executed") is not False
        or limits.get("jit_equivalence_tested") is not False
        or type(metrics) is not dict
        or metrics.get("simulated_seconds") != 10
        or type(trace_record) is not dict
        or trace_record.get("path") != TRACE_FILENAME
        or trace_record.get("sha256") != trace_sha256
        or trace_record.get("size") != trace.stat().st_size
        or stdout.get("trace_path") != str(trace)
        or stdout.get("trace_sha256") != trace_sha256
        or stdout.get("manifest_path") != str(manifest_path)
        or stdout.get("manifest_sha256") != manifest_sha256
        or stdout.get("claim_status") != "reconstructed_plain_actor_not_jit_equivalent"
    ):
        raise ProbeError("completed replay artifacts do not satisfy the GMT baseline contract")
    return {
        "child_stderr_bytes": len(stderr),
        "manifest": {
            "path": manifest_path.name,
            "sha256": manifest_sha256,
            "size": len(manifest_bytes),
        },
        "trace": {
            "path": trace.name,
            "sha256": trace_sha256,
            "size": trace.stat().st_size,
        },
    }


def _plan_unchanged(plan: ProbePlan) -> None:
    root, commit, source_files = _repository_identity(plan.config.repository_root)
    if root != plan.config.repository_root or commit != plan.commit:
        raise ProbeError("repository identity changed during the GMT probe")
    if source_files != plan.inputs["repository_sources"]:
        raise ProbeError("runtime source bytes changed during the GMT probe")
    if _venv_identity(plan.config.venv_python) != plan.inputs["venv"]:
        raise ProbeError("venv identity changed during the GMT probe")
    if sha256_file(plan.config.weights_path) != plan.config.weights_sha256:
        raise ProbeError("actor bytes changed during the GMT probe")
    if sha256_file(plan.config.motion_path) != plan.config.motion_sha256:
        raise ProbeError("motion bytes changed during the GMT probe")
    if verify_upstream_root(plan.config.upstream_root) != plan.inputs["support_files"]:
        raise ProbeError("model/support bytes changed during the GMT probe")


def _terminal_receipt(
    *,
    plan: ProbePlan,
    reservation: Mapping[str, object],
    token: Mapping[str, object],
    status: str,
    reason: str | None,
    wall_seconds: float,
    peak_rss_bytes: int,
    minimum_free_disk_bytes: int,
    maximum_output_bytes: int,
    group_validated: bool,
    cleanup_succeeded: bool,
    release: Mapping[str, object],
    artifacts: Mapping[str, object] | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact": plan.artifact_label,
        "evidence_class": plan.evidence_class,
        "status": status,
        "reason": reason,
        "commit": plan.commit,
        "canonical_argv": list(plan.canonical_argv),
        "inputs": plan.accepted_fields()["inputs"],
        "reservation_sha256": hashlib.sha256(canonical_json_bytes(dict(reservation))).hexdigest(),
        "slot": {
            "owner": token["owner"],
            "token_id": token["token_id"],
            **dict(release),
        },
        "resources": {
            "accepted_schema_wall_seconds": ACCEPTED_SMOKE_WALL_SECONDS,
            "effective_outer_wall_seconds": plan.limits.wall_seconds,
            "cpu_time_seconds": plan.limits.cpu_seconds,
            "rss_bytes": plan.limits.rss_bytes,
            "free_disk_bytes": plan.limits.free_disk_bytes,
            "output_bytes": plan.limits.output_bytes,
            "child_output_bytes": plan.limits.child_output_bytes,
            "one_thread_environment": THREAD_ENVIRONMENT,
            "observed_wall_seconds": wall_seconds,
            "observed_peak_rss_bytes": peak_rss_bytes,
            "observed_minimum_free_disk_bytes": minimum_free_disk_bytes,
            "observed_maximum_output_bytes": maximum_output_bytes,
        },
        "cleanup": {
            "process_group_validated": group_validated,
            "verified_before_release": cleanup_succeeded,
        },
        "artifacts": dict(artifacts) if artifacts is not None else None,
        "runtime": {"platform": platform.platform(), "supervisor_python": sys.version.split()[0]},
    }


def supervise_probe(
    plan: ProbePlan,
    *,
    coordination_root: Path,
    reservation: Mapping[str, object],
    reservation_path: Path,
    dependencies: ProbeDependencies | None = None,
) -> dict[str, object]:
    dependencies = dependencies or ProbeDependencies()
    validate_reservation_binding(plan, reservation)
    limits = plan.limits
    if dependencies.free_disk_bytes(plan.config.output_directory.parent) < limits.free_disk_bytes:
        raise ProbeError("free disk is below the plan's pre-spawn minimum")
    token = dependencies.reserve(
        coordination_root,
        owner=plan.config.owner,
        reservation_path=reservation_path,
        expected_wall_seconds=limits.wall_seconds,
    )
    process: ProcessLike | None = None
    group_validated = False
    cleanup_succeeded = True
    release_record: dict[str, object] = {"released": False, "retained": True}
    status = "precondition_failure"
    reason: str | None = None
    artifacts: dict[str, object] | None = None
    peak_rss = 0
    minimum_disk = dependencies.free_disk_bytes(plan.config.output_directory.parent)
    maximum_output = 0
    started = dependencies.clock()
    try:
        validated_token = dependencies.validate_slot(
            coordination_root,
            reservation,
            expected_wall_seconds=limits.wall_seconds,
        )
        if validated_token != token:
            raise ProbeError("pre-spawn slot validation returned a different token")
        dependencies.validate_plan(plan)
        plan.config.output_directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        stdout_path = plan.config.output_directory / STDOUT_FILENAME
        stderr_path = plan.config.output_directory / STDERR_FILENAME
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            process = _spawn_child(plan, stdout=stdout, stderr=stderr, dependencies=dependencies)
            group_validated = _validate_process_group(process, dependencies)
            if not group_validated:
                raise ProbeError("child process group identity could not be verified")
            status = "running"
            while process.poll() is None:
                wall = dependencies.clock() - started
                rss = dependencies.rss_bytes(process.pid)
                free = dependencies.free_disk_bytes(plan.config.output_directory)
                output = dependencies.directory_bytes(plan.config.output_directory)
                minimum_disk = min(minimum_disk, free)
                maximum_output = max(maximum_output, output)
                if wall >= limits.wall_seconds:
                    status, reason = (
                        "timeout",
                        f"effective {limits.wall_seconds}-second outer wall exceeded",
                    )
                    break
                if rss is None or type(rss) is not int or rss < 0:
                    status, reason = "resource_breach", "process-tree RSS observation unavailable"
                    break
                peak_rss = max(peak_rss, rss)
                if rss > limits.rss_bytes:
                    status, reason = "resource_breach", "process-tree RSS exceeded its plan limit"
                    break
                if free < limits.free_disk_bytes:
                    status, reason = "resource_breach", "free disk fell below its plan minimum"
                    break
                if output > limits.child_output_bytes:
                    status, reason = (
                        "resource_breach",
                        "child output exhausted its receipt-reserved plan allowance",
                    )
                    break
                dependencies.sleep(POLL_SECONDS)
            if process.poll() is not None and status == "running":
                status = "child_exited"
    except BaseException as exc:
        if reason is None:
            reason = f"{type(exc).__name__}: {exc}"[:1000]
        if status == "running":
            status = "precondition_failure"
    finally:
        if process is not None:
            try:
                dependencies.cleanup(
                    CleanupProcessAdapter(process),
                    group_validated=group_validated,
                )
            except BaseException as exc:
                cleanup_succeeded = False
                status = "cleanup_failure"
                reason = f"{type(exc).__name__}: {exc}"[:1000]
            if not group_validated:
                cleanup_succeeded = False
                status = "cleanup_failure"
                reason = "process group was not validated; exact descendant cleanup is unverified"
    wall_seconds = dependencies.clock() - started
    if plan.config.output_directory.exists():
        minimum_disk = min(
            minimum_disk,
            dependencies.free_disk_bytes(plan.config.output_directory),
        )
        maximum_output = max(
            maximum_output,
            dependencies.directory_bytes(plan.config.output_directory),
        )
    if cleanup_succeeded and maximum_output > limits.child_output_bytes:
        status, reason = "resource_breach", "final child output exceeded its bounded allowance"
    if cleanup_succeeded and minimum_disk < limits.free_disk_bytes:
        status, reason = "resource_breach", "final free disk fell below its plan minimum"
    if (
        cleanup_succeeded
        and process is not None
        and process.returncode == 0
        and status == "child_exited"
    ):
        try:
            dependencies.validate_plan(plan)
            verified = dependencies.verify_completed(plan)
            if type(verified) is not dict:
                raise ProbeError("terminal verifier must return an artifact object")
            canonical_json_bytes(verified)
            artifacts = verified
            status = "succeeded"
            reason = None
        except BaseException as exc:
            status = "artifact_failure"
            reason = f"{type(exc).__name__}: {exc}"[:1000]
    elif status == "child_exited":
        status = "child_failure"
        reason = f"child exited with code {process.returncode if process else None}"
    if cleanup_succeeded:
        try:
            released = dependencies.release(
                coordination_root,
                owner=plan.config.owner,
                token_id=str(token["token_id"]),
            )
            if released != token:
                raise ProbeError("released slot token differs")
            release_record = {"released": True, "retained": False}
        except BaseException as exc:
            status = "slot_release_failure"
            reason = f"{type(exc).__name__}: {exc}"[:1000]
            release_record = {"released": False, "retained": True}
    if plan.config.output_directory.exists():
        receipt = _terminal_receipt(
            plan=plan,
            reservation=reservation,
            token=token,
            status=status,
            reason=reason,
            wall_seconds=wall_seconds,
            peak_rss_bytes=peak_rss,
            minimum_free_disk_bytes=minimum_disk,
            maximum_output_bytes=maximum_output,
            group_validated=group_validated,
            cleanup_succeeded=cleanup_succeeded,
            release=release_record,
            artifacts=artifacts,
        )
        receipt_path = plan.config.output_directory / RECEIPT_FILENAME
        receipt_sha256 = write_json_receipt(receipt_path, receipt)
    else:
        receipt_path = None
        receipt_sha256 = None
    return {
        "ok": status == "succeeded",
        "status": status,
        "reason": reason,
        "receipt_path": str(receipt_path) if receipt_path else None,
        "receipt_sha256": receipt_sha256,
        "slot": release_record,
    }


def _shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--python", dest="venv_python", type=Path, required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--weights", dest="weights_path", type=Path, required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--motion", dest="motion_path", type=Path, required=True)
    parser.add_argument("--motion-sha256", required=True)
    parser.add_argument("--motion-name", required=True)
    parser.add_argument("--output", dest="output_directory", type=Path, required=True)
    parser.add_argument("--owner", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    request = commands.add_parser("request")
    _shared_arguments(request)
    run = commands.add_parser("run")
    _shared_arguments(run)
    run.add_argument("--coordination-root", type=Path, required=True)
    run.add_argument("--reservation", type=Path, required=True)
    return parser


def _config(arguments: argparse.Namespace) -> ProbeConfig:
    return ProbeConfig(
        repository_root=arguments.repository_root,
        venv_python=arguments.venv_python,
        upstream_root=arguments.upstream_root,
        weights_path=arguments.weights_path,
        weights_sha256=arguments.weights_sha256,
        motion_path=arguments.motion_path,
        motion_sha256=arguments.motion_sha256,
        motion_name=arguments.motion_name,
        output_directory=arguments.output_directory,
        owner=arguments.owner,
    )


def main() -> int:
    arguments = _parser().parse_args()
    try:
        plan = build_probe_plan(_config(arguments))
        if arguments.command == "request":
            result = {
                "schema_version": 1,
                "exact_probe_binding": plan.accepted_fields(),
                "effective_limits": {
                    "outer_wall_seconds": plan.limits.wall_seconds,
                    "cpu_seconds": plan.limits.cpu_seconds,
                    "rss_bytes": plan.limits.rss_bytes,
                    "free_disk_bytes": plan.limits.free_disk_bytes,
                    "output_bytes": plan.limits.output_bytes,
                    "child_output_bytes": plan.limits.child_output_bytes,
                },
                "reservation_v2_dynamic_fields_required": [
                    "accepted_until_utc",
                    "acceptance_message",
                    "acknowledgment",
                    "proposal_id",
                    "required_authorizer",
                ],
            }
        else:
            reservation = load_validated_reservation(
                arguments.coordination_root,
                arguments.reservation,
            )
            result = supervise_probe(
                plan,
                coordination_root=arguments.coordination_root,
                reservation=reservation,
                reservation_path=arguments.reservation,
            )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0 if result.get("ok", True) else 2
    except (GMTAdmissionError, ProbeError, ResourceSlotError, OSError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
