from __future__ import annotations

import hashlib
import json
import signal
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.harness.resource_slot import SLOT_FILENAME
from oracle_composition.harness.resource_slot import (
    canonical_json_bytes as slot_canonical_json_bytes,
)
from oracle_composition.phase_b import supervision as supervision_module
from oracle_composition.phase_b.isolation import (
    seal_artifact,
    sealed_input_lineage_sha256,
)
from oracle_composition.phase_b.supervision import (
    MAX_IPC_FRAME_BYTES,
    ResourceLimits,
    SupervisionDependencies,
    SupervisorStatus,
    _frame_bytes,
    cleanup_worker_process,
    limits_bound_by_reservation,
    supervise_training_job,
    validate_reservation,
    validate_training_preflight,
)

ROOT = Path(__file__).resolve().parents[2]
PHASE_B = ROOT / "experiments/003_composition_speed_profile/phase_b"


def test_directory_bytes_tolerates_checkpoint_publication_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pending = tmp_path / ".checkpoint.pending"
    pending.write_bytes(b"checkpoint")
    retained = tmp_path / "receipt.json"
    retained.write_bytes(b"{}")
    original_stat = Path.stat

    def stat_during_publish(path: Path, *args: object, **kwargs: object) -> object:
        if path == pending:
            pending.rename(tmp_path / "checkpoint.npz")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat_during_publish)
    assert supervision_module._directory_bytes(tmp_path) == 2
    assert supervision_module._directory_bytes(tmp_path) == 12


@pytest.mark.parametrize("dangling", (False, True))
def test_directory_bytes_still_rejects_symbolic_links(tmp_path: Path, dangling: bool) -> None:
    target = tmp_path / "target"
    if not dangling:
        target.write_bytes(b"checkpoint")
    (tmp_path / "link").symlink_to(target)
    with pytest.raises(ExperimentContractError, match="symbolic link"):
        supervision_module._directory_bytes(tmp_path)


def test_directory_bytes_does_not_hide_other_stat_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    denied = tmp_path / "checkpoint.npz"
    denied.write_bytes(b"checkpoint")
    original_stat = Path.stat

    def denied_stat(path: Path, *args: object, **kwargs: object) -> object:
        if path == denied:
            raise PermissionError("denied")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", denied_stat)
    with pytest.raises(PermissionError, match="denied"):
        supervision_module._directory_bytes(tmp_path)


@pytest.fixture(scope="module")
def preflight() -> object:
    return validate_training_preflight(
        repository_root=ROOT,
        experiment=ROOT / "experiments/003_composition_speed_profile",
        oracle_path=PHASE_B / "oracle_cycle_1_reference_v1.json",
        reward_path=PHASE_B / "tracking_only_v1.json",
        allow_dirty=True,
    )


def _limits(*, seed_wall: float = 20.0) -> ResourceLimits:
    return ResourceLimits(
        per_seed_wall_seconds=seed_wall,
        cohort_wall_seconds=30.0,
        job_wall_seconds=30.0,
        rss_bytes=8 * 1024**3,
        free_disk_bytes=1,
        output_bytes=64 * 1024**2,
        throughput_floor_steps_s=1.0,
        throughput_warmup_transitions=65_536,
        throughput_window_transitions=65_536,
    )


def _run(
    preflight: object,
    output: Path,
    *,
    failure_mode: str | None = None,
    seed_wall: float = 20.0,
    dependencies: SupervisionDependencies | None = None,
    limits: ResourceLimits | None = None,
) -> object:
    return supervise_training_job(
        preflight=preflight,
        output_directory=output.resolve(),
        seeds=(11, 13),
        transitions=16,
        smoke=False,
        reservation=None,
        limits=limits or _limits(seed_wall=seed_wall),
        runtime_kind="fake",
        test_only=True,
        test_steps_per_environment=4,
        test_batch_size=16,
        test_n_epochs=1,
        failure_mode=failure_mode,
        dependencies=dependencies,
    )


class _FakeProcess:
    pid = 424_242

    def __init__(self, *, alive: bool = True, exitcode: int | None = None) -> None:
        self.alive = alive
        self.exitcode = exitcode
        self.terminate_calls = 0
        self.kill_calls = 0
        self.survive_cleanup = False

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminate_calls += 1
        if not self.survive_cleanup:
            self.alive = False

    def kill(self) -> None:
        self.kill_calls += 1
        if not self.survive_cleanup:
            self.alive = False

    def join(self, timeout: float = 0.0) -> None:
        del timeout


class _ScriptedConnection:
    def __init__(self, frames: list[bytes]) -> None:
        self.frames = list(frames)
        self.sent: list[bytes] = []
        self.closed = False

    def poll(self, timeout: float) -> bool:
        del timeout
        return bool(self.frames)

    def recv_bytes(self, maxlength: int) -> bytes:
        encoded = self.frames.pop(0)
        if len(encoded) > maxlength:
            raise OSError("bad message length")
        return encoded

    def send_bytes(self, encoded: bytes) -> None:
        self.sent.append(encoded)

    def close(self) -> None:
        self.closed = True


def _observer_dependencies(
    spawn: object,
    *,
    clock: object = __import__("time").perf_counter,
    rss: object = lambda _pid: 0,
    free_disk: object = lambda _path: 10**15,
    output_bytes: object = lambda _path: 0,
    cleanup: object = cleanup_worker_process,
) -> SupervisionDependencies:
    return SupervisionDependencies(
        clock=clock,
        spawn_worker=spawn,
        process_tree_rss_bytes=rss,
        free_disk_bytes=free_disk,
        directory_bytes=output_bytes,
        cleanup_worker=cleanup,
    )


def test_fake_supervisor_two_seed_science_is_byte_deterministic(
    tmp_path: Path, preflight: object
) -> None:
    first = _run(preflight, tmp_path / "first")
    second = _run(preflight, tmp_path / "second")
    assert first.status == second.status == SupervisorStatus.SUCCEEDED
    assert [outcome.seed for outcome in first.outcomes] == [11, 13]
    assert all(outcome.persistence is not None for outcome in first.outcomes)

    deterministic_paths = ["execution_manifest_v3.json", "job_result_v1.json"]
    for seed in (11, 13):
        deterministic_paths.extend(
            (
                f"seed_{seed}/actor_seed_{seed}_final.npz",
                f"seed_{seed}/checkpoint_seed_{seed}_final.npz",
                f"seed_{seed}/persistence_seed_{seed}_v1.json",
                f"seed_{seed}/rsi_ledger_v1.json",
                f"seed_{seed}/success_receipt_v2.json",
                f"seed_{seed}/training_facts_v1.json",
            )
        )
    for relative in deterministic_paths:
        assert (tmp_path / "first" / relative).read_bytes() == (
            tmp_path / "second" / relative
        ).read_bytes()
    assert (tmp_path / "first/seed_11/telemetry_v1.json").read_bytes() != (
        tmp_path / "second/seed_11/telemetry_v1.json"
    ).read_bytes()
    telemetry = json.loads((tmp_path / "first/seed_11/telemetry_v1.json").read_bytes())
    assert telemetry["status"] == "succeeded"
    assert telemetry["minimum_free_disk_bytes"] > 0
    assert telemetry["maximum_output_bytes"] > 0
    assert telemetry["resource_limits"] == _limits().to_dict()
    controls = telemetry["resource_controls"]
    assert controls["executed_modules"]["enforcement"] == (
        "checkout_realpath_and_recorded_digest_verified"
    )
    assert len(controls["executed_modules"]["start_sha256"]) == 64
    assert len(controls["executed_modules"]["final_sha256"]) == 64
    success = json.loads((tmp_path / "first/seed_11/success_receipt_v2.json").read_bytes())
    assert success["resource_controls"] == controls
    manifest = json.loads((tmp_path / "first/execution_manifest_v3.json").read_bytes())
    assert manifest["execution_manifest_schema_id"] == "humanoid_phase_b_execution_manifest/v3"
    assert manifest["schema_version"] == 3
    assert len(manifest["sealed_input_lineage"]["artifacts"]) == 103


def test_preflight_seals_every_declared_input_family(preflight: object) -> None:
    artifacts = preflight.sealed_inputs
    roles = {role for artifact in artifacts for role in artifact.roles}
    assert {
        "evaluator",
        "library",
        "oracle",
        "reference_corpus",
        "reference_corpus.index",
        "reward",
        "starting_checkpoint",
        "starting_checkpoint.e1_receipt",
        "starting_checkpoint.e1_synthetic_design",
        "starting_checkpoint.source_expert",
        "starting_checkpoint.strict_actor_export",
        "task",
        "training_design",
    } <= roles
    bundle_roles = {role for role in roles if role.startswith("reference_corpus.bundle.corpus-")}
    assert len(bundle_roles) == 27
    assert len({artifact.relative_path for artifact in artifacts}) == len(artifacts)


def _envelope(message_type: str, payload: dict[str, object]) -> bytes:
    return canonical_json_bytes(
        {
            "frame_schema_id": "humanoid_phase_b_bounded_ipc_frame/v1",
            "message_type": message_type,
            "payload": payload,
            "schema_version": 1,
        }
    )


@pytest.mark.parametrize(
    "encoded",
    [
        b"x" * (MAX_IPC_FRAME_BYTES + 1),
        b'{"frame_schema_id":',
        _envelope("resource_breach", {}),
        _envelope("resource_breach", {"reason": "x", "extra": True}),
        _envelope("unknown", {}),
        (
            b'{"frame_schema_id":"humanoid_phase_b_bounded_ipc_frame/v1",'
            b'"message_type":"resource_breach","payload":{"reason":NaN},'
            b'"schema_version":1}\n'
        ),
    ],
    ids=("over_limit", "truncated", "missing_field", "extra_field", "unknown_type", "non_finite"),
)
def test_malformed_worker_frames_fail_once_before_construction_or_persistence(
    tmp_path: Path,
    preflight: object,
    encoded: bytes,
) -> None:
    process = _FakeProcess()
    connection = _ScriptedConnection([encoded])
    cleanup_calls = 0

    def spawn(_request: object) -> tuple[object, object]:
        return process, connection

    def cleanup(worker: object, *, group_validated: bool) -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        cleanup_worker_process(worker, group_validated=group_validated)

    output = tmp_path / hashlib.sha256(encoded).hexdigest()
    result = _run(
        preflight,
        output,
        dependencies=_observer_dependencies(spawn, cleanup=cleanup),
    )

    assert result.status is SupervisorStatus.MALFORMED_FRAME
    assert cleanup_calls == 1
    seed = output / "seed_11"
    assert len(list(seed.glob("*failure_receipt*"))) == 1
    assert not list(seed.glob("*success_receipt*"))
    assert not (seed / "training_facts_v1.json").exists()
    assert not list(seed.glob("checkpoint_*"))
    assert all(b"begin_construction" not in frame for frame in connection.sent)


def test_primary_non_finite_failure_preserves_close_failure_separately(
    tmp_path: Path,
    preflight: object,
) -> None:
    output = tmp_path / "primary-and-close"
    result = _run(preflight, output, failure_mode="non_finite_close_failure")

    assert result.status is SupervisorStatus.NON_FINITE
    receipt = json.loads((output / "seed_11/failure_receipt_v2.json").read_bytes())
    assert receipt["primary_failure"]["status"] == "non_finite"
    assert "non-finite" in receipt["primary_failure"]["reason"]
    worker_cleanup = receipt["cleanup_outcome"]["worker_environment"]
    assert worker_cleanup["attempted"] is True
    assert worker_cleanup["succeeded"] is False
    assert "close failure" in worker_cleanup["error"]
    assert not (output / "seed_11/success_receipt_v2.json").exists()


def _tight_limits(**overrides: object) -> ResourceLimits:
    values: dict[str, object] = {
        "per_seed_wall_seconds": 100.0,
        "cohort_wall_seconds": 200.0,
        "job_wall_seconds": 200.0,
        "cpu_time_seconds": 100.0,
        "rss_bytes": 1_000,
        "free_disk_bytes": 10,
        "output_bytes": 1_000,
        "throughput_floor_steps_s": 100.0,
        "throughput_warmup_transitions": 4,
        "throughput_window_transitions": 4,
    }
    values.update(overrides)
    return ResourceLimits(**values)


def _no_frame_spawn(
    *,
    alive: bool = True,
    exitcode: int | None = None,
) -> tuple[object, _FakeProcess, _ScriptedConnection]:
    process = _FakeProcess(alive=alive, exitcode=exitcode)
    connection = _ScriptedConnection([])

    def spawn(_request: object) -> tuple[object, object]:
        return process, connection

    return spawn, process, connection


@pytest.mark.parametrize("detector", ("rss", "disk", "aggregate_output"))
def test_parent_observers_cross_real_resource_detectors_and_cleanup(
    tmp_path: Path,
    preflight: object,
    detector: str,
) -> None:
    spawn, process, _connection = _no_frame_spawn()
    free_calls = 0
    observed_paths: list[Path] = []

    def free_disk(_path: Path) -> int:
        nonlocal free_calls
        free_calls += 1
        return 0 if detector == "disk" and free_calls > 1 else 10**15

    def directory_bytes(path: Path) -> int:
        observed_paths.append(path)
        return 1_001 if detector == "aggregate_output" and path.name == detector else 0

    dependencies = _observer_dependencies(
        spawn,
        rss=(lambda _pid: 1_001 if detector == "rss" else 0),
        free_disk=free_disk,
        output_bytes=directory_bytes,
    )
    output = tmp_path / detector
    result = _run(
        preflight,
        output,
        limits=_tight_limits(),
        dependencies=dependencies,
    )

    assert result.status is SupervisorStatus.RESOURCE_BREACH
    assert process.terminate_calls == 1
    receipt = json.loads((output / "seed_11/failure_receipt_v2.json").read_bytes())
    if detector == "aggregate_output":
        assert "aggregate job output" in receipt["reason"]
        assert output in observed_paths
    assert receipt["cleanup_outcome"]["supervisor_process_group"]["succeeded"] is True


def test_actual_low_throughput_progress_crosses_detector_and_cleans_process(
    tmp_path: Path,
    preflight: object,
) -> None:
    process = _FakeProcess()
    connection: _ScriptedConnection

    def spawn(request: object) -> tuple[object, object]:
        nonlocal connection
        started = {
            "cpu_time_control": {
                "enforcement": "os_enforced",
                "hard_limit_seconds": 101,
                "limit_seconds": 100,
                "resource": "RLIMIT_CPU",
            },
            "environment_control": {
                "allowlist_enforced": True,
                "environment_sha256": "1" * 64,
                "keys": [],
                "runtime_added_keys_removed": [],
                "unexpected_keys": [],
            },
            "executed_module_identity_sha256": "2" * 64,
            "pgid": process.pid,
            "pid": process.pid,
            "sealed_input_lineage_sha256": request.sealed_input_lineage_sha256,
            "sid": process.pid,
            "source_snapshot_sha256": request.source_snapshot_sha256,
        }
        acknowledged = {
            "executed_module_identity_sha256": "2" * 64,
            "manifest_sha256": request.execution_manifest_sha256,
            "model_or_environment_constructed": False,
            "sealed_input_lineage_sha256": request.sealed_input_lineage_sha256,
            "source_snapshot_sha256": request.source_snapshot_sha256,
        }

        def progress(steps: int) -> dict[str, object]:
            return {
                "observed_transitions": steps,
                "rollout_count": steps // 4,
                "stream_counts": {"composition": steps // 2, "rehearsal": steps // 2},
                "worker_peak_rss_bytes": 1,
            }

        connection = _ScriptedConnection(
            [
                _frame_bytes("worker_started", started),
                _frame_bytes("execution_acknowledged", acknowledged),
                _frame_bytes("progress", progress(4)),
                _frame_bytes("progress", progress(8)),
            ]
        )
        return process, connection

    clock_value = -0.1

    def clock() -> float:
        nonlocal clock_value
        clock_value += 0.1
        return clock_value

    output = tmp_path / "throughput"
    result = _run(
        preflight,
        output,
        limits=_tight_limits(),
        dependencies=_observer_dependencies(
            spawn,
            clock=clock,
            cleanup=lambda worker, *, group_validated: worker.terminate(),
        ),
    )
    assert result.status is SupervisorStatus.RESOURCE_BREACH
    assert process.terminate_calls == 1
    telemetry = json.loads((output / "seed_11/telemetry_v1.json").read_bytes())
    assert telemetry["throughput_windows"][0]["steps_per_second"] < 100.0


def test_cpu_limit_exit_is_resource_breach_and_process_is_reaped(
    tmp_path: Path,
    preflight: object,
) -> None:
    spawn, process, _connection = _no_frame_spawn(
        alive=False,
        exitcode=-int(signal.SIGXCPU),
    )
    output = tmp_path / "cpu"
    result = _run(
        preflight,
        output,
        limits=_tight_limits(),
        dependencies=_observer_dependencies(spawn),
    )
    assert result.status is SupervisorStatus.RESOURCE_BREACH
    assert process.terminate_calls == 0
    receipt = json.loads((output / "seed_11/failure_receipt_v2.json").read_bytes())
    assert "CPU-time" in receipt["reason"]


def test_surviving_worker_is_recorded_without_reclassifying_primary_failure(
    tmp_path: Path,
    preflight: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess()
    process.survive_cleanup = True
    failed = _frame_bytes(
        "failed",
        {
            "reason": "primary non-finite detector",
            "status": "non_finite",
            "worker_cleanup": {"attempted": True, "error": None, "succeeded": True},
        },
    )
    connection: _ScriptedConnection

    def spawn(request: object) -> tuple[object, object]:
        nonlocal connection
        started = _frame_bytes(
            "worker_started",
            {
                "cpu_time_control": {
                    "enforcement": "unsupported",
                    "limit_seconds": 100,
                    "resource": "RLIMIT_CPU",
                },
                "environment_control": {
                    "allowlist_enforced": True,
                    "environment_sha256": "1" * 64,
                    "keys": [],
                    "runtime_added_keys_removed": [],
                    "unexpected_keys": [],
                },
                "executed_module_identity_sha256": "2" * 64,
                "pgid": process.pid,
                "pid": process.pid,
                "sealed_input_lineage_sha256": request.sealed_input_lineage_sha256,
                "sid": process.pid,
                "source_snapshot_sha256": request.source_snapshot_sha256,
            },
        )
        connection = _ScriptedConnection([started, failed])
        return process, connection

    group_signals: list[int] = []

    def killpg(pid: int, selected_signal: int) -> None:
        assert pid == process.pid
        group_signals.append(selected_signal)

    monkeypatch.setattr(supervision_module.os, "killpg", killpg)

    output = tmp_path / "cleanup-survivor"
    result = _run(
        preflight,
        output,
        limits=_tight_limits(),
        dependencies=_observer_dependencies(spawn),
    )
    assert result.status is SupervisorStatus.NON_FINITE
    assert group_signals == [signal.SIGTERM, signal.SIGKILL]
    assert process.terminate_calls == process.kill_calls == 0
    receipt = json.loads((output / "seed_11/failure_receipt_v2.json").read_bytes())
    assert receipt["primary_failure"]["status"] == "non_finite"
    assert receipt["cleanup_outcome"]["supervisor_process_group"]["succeeded"] is False
    assert "survived cleanup" in receipt["cleanup_outcome"]["supervisor_process_group"]["error"]


@pytest.mark.parametrize("ack_kind", ("missing", "mismatched"))
def test_bad_worker_ack_never_admits_construction(
    tmp_path: Path,
    preflight: object,
    ack_kind: str,
) -> None:
    process = _FakeProcess()
    connection: _ScriptedConnection

    def spawn(request: object) -> tuple[object, object]:
        nonlocal connection
        started = _frame_bytes(
            "worker_started",
            {
                "cpu_time_control": {
                    "enforcement": "unsupported",
                    "limit_seconds": 100,
                    "resource": "RLIMIT_CPU",
                },
                "environment_control": {
                    "allowlist_enforced": True,
                    "environment_sha256": "1" * 64,
                    "keys": [],
                    "runtime_added_keys_removed": [],
                    "unexpected_keys": [],
                },
                "executed_module_identity_sha256": "2" * 64,
                "pgid": process.pid,
                "pid": process.pid,
                "sealed_input_lineage_sha256": request.sealed_input_lineage_sha256,
                "sid": process.pid,
                "source_snapshot_sha256": request.source_snapshot_sha256,
            },
        )
        frames = [started]
        if ack_kind == "mismatched":
            frames.append(
                _frame_bytes(
                    "execution_acknowledged",
                    {
                        "executed_module_identity_sha256": "2" * 64,
                        "manifest_sha256": "f" * 64,
                        "model_or_environment_constructed": False,
                        "sealed_input_lineage_sha256": request.sealed_input_lineage_sha256,
                        "source_snapshot_sha256": request.source_snapshot_sha256,
                    },
                )
            )
        connection = _ScriptedConnection(frames)
        return process, connection

    values = iter(
        (0.0, 0.0, 0.0, 0.0, 31.0) if ack_kind == "missing" else (0.0, 0.0, 0.0, 0.0, 0.0)
    )

    def clock() -> float:
        return next(values, 31.0)

    output = tmp_path / ack_kind
    result = _run(
        preflight,
        output,
        limits=_tight_limits(),
        dependencies=_observer_dependencies(
            spawn,
            clock=clock,
            cleanup=lambda worker, *, group_validated: worker.terminate(),
        ),
    )
    expected = (
        SupervisorStatus.TIMEOUT if ack_kind == "missing" else SupervisorStatus.PRECONDITION_FAILURE
    )
    assert result.status is expected
    assert all(b"begin_construction" not in frame for frame in connection.sent)
    assert not list((output / "seed_11").glob("*success_receipt*"))


@pytest.mark.parametrize("construction_admission", ("missing", "mismatched"))
def test_worker_never_constructs_without_exact_post_ack_admission(
    tmp_path: Path,
    preflight: object,
    construction_admission: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_value = {"construction_sentinel": True}
    manifest_bytes = canonical_json_bytes(manifest_value)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    plan = supervision_module.TrainingPlan(
        seed=11,
        transitions=8_192,
        manifest_sha256=manifest_sha256,
        evidence_class="interface_check",
        promotable=False,
        smoke=False,
        test_only=True,
    )
    request = supervision_module.WorkerRequest(
        plan=plan,
        output_directory=str(tmp_path),
        execution_manifest_bytes=manifest_bytes,
        execution_manifest_sha256=manifest_sha256,
        e1_receipt_sha256="1" * 64,
        source_snapshot_sha256=preflight.source_snapshot.sha256,
        sealed_inputs=preflight.sealed_inputs,
        sealed_input_lineage_sha256=preflight.sealed_input_lineage_sha256,
        repository_root=str(preflight.repository_root),
        runtime_kind="fake",
        runtime_config=None,
        failure_mode=None,
        rss_limit_bytes=1_000,
        cpu_time_limit_seconds=100.0,
        worker_environment=(),
    )
    frames = [
        _frame_bytes(
            "admit_execution",
            {"manifest": manifest_value, "manifest_sha256": manifest_sha256},
        )
    ]
    if construction_admission == "mismatched":
        frames.append(_frame_bytes("begin_construction", {"manifest_sha256": "f" * 64}))
    connection = _ScriptedConnection(frames)
    construction_calls = 0

    def construction_sentinel(_connection: object, _request: object) -> None:
        nonlocal construction_calls
        construction_calls += 1

    monkeypatch.setattr(supervision_module.os, "setsid", lambda: None)
    monkeypatch.setattr(
        supervision_module,
        "_environment_control",
        lambda _request: {
            "allowlist_enforced": True,
            "environment_sha256": "1" * 64,
            "keys": [],
            "runtime_added_keys_removed": [],
            "unexpected_keys": [],
        },
    )
    monkeypatch.setattr(
        supervision_module,
        "_apply_cpu_time_limit",
        lambda _seconds: {
            "enforcement": "unsupported",
            "limit_seconds": 100,
            "resource": "RLIMIT_CPU",
        },
    )
    monkeypatch.setattr(
        supervision_module,
        "_worker_authorities",
        lambda _request: (preflight.source_snapshot, {"sha256": "2" * 64}),
    )
    monkeypatch.setattr(supervision_module, "_worker_execute", construction_sentinel)

    with pytest.raises(ExperimentContractError, match="construction admission"):
        supervision_module._worker_session_entry(connection, request)

    assert construction_calls == 0
    assert connection.closed is True
    sent_types = [json.loads(frame)["message_type"] for frame in connection.sent]
    assert sent_types == ["worker_started", "execution_acknowledged", "failed"]


def test_twenty_minute_reservation_overrides_twenty_two_minute_seed_deadline_exactly(
    tmp_path: Path,
    preflight: object,
) -> None:
    spawn, process, _connection = _no_frame_spawn()
    values = iter((0.0, 0.0, 0.0, 20.0 * 60.0))

    def clock() -> float:
        return next(values, 20.0 * 60.0)

    output = tmp_path / "exact-deadline"
    bounded = limits_bound_by_reservation(
        _tight_limits(
            per_seed_wall_seconds=22.0 * 60.0,
            cohort_wall_seconds=95.0 * 60.0,
            job_wall_seconds=120.0 * 60.0,
            cpu_time_seconds=22.0 * 60.0,
        ),
        {"hard_wall_seconds": 20.0 * 60.0},
    )
    assert bounded.per_seed_wall_seconds == 20.0 * 60.0
    assert bounded.cpu_time_seconds == 20.0 * 60.0
    result = _run(
        preflight,
        output,
        limits=bounded,
        dependencies=_observer_dependencies(spawn, clock=clock),
    )
    assert result.status is SupervisorStatus.TIMEOUT
    assert process.terminate_calls == 1
    receipt = json.loads((output / "seed_11/failure_receipt_v2.json").read_bytes())
    assert receipt["reason"] == "per-seed wall limit exceeded"


def test_spawned_worker_strips_non_allowlisted_environment(
    tmp_path: Path,
    preflight: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FT2R3_MUST_NOT_REACH_WORKER", "secret")
    output = tmp_path / "clean-environment"
    result = _run(preflight, output)
    assert result.status is SupervisorStatus.SUCCEEDED
    telemetry = json.loads((output / "seed_11/telemetry_v1.json").read_bytes())
    environment = telemetry["resource_controls"]["environment"]
    assert environment["allowlist_enforced"] is True
    assert "FT2R3_MUST_NOT_REACH_WORKER" not in environment["keys"]
    assert environment["unexpected_keys"] == []
    cpu = telemetry["resource_controls"]["cpu_time"]
    assert cpu["enforcement"] in {"os_enforced", "unsupported"}


def test_worker_source_drift_crosses_authority_detector_before_construction(
    tmp_path: Path,
    preflight: object,
) -> None:
    def spawn(request: object) -> tuple[object, object]:
        return supervision_module._spawn_worker(replace(request, source_snapshot_sha256="f" * 64))

    output = tmp_path / "source-drift"
    result = _run(
        preflight,
        output,
        limits=_tight_limits(),
        dependencies=_observer_dependencies(spawn),
    )

    assert result.status is SupervisorStatus.SOURCE_MUTATION
    seed = output / "seed_11"
    receipt = json.loads((seed / "failure_receipt_v2.json").read_bytes())
    assert receipt["last_acknowledged_stage"] == "spawned"
    assert "source snapshot differs" in receipt["reason"]
    assert not list(seed.glob("checkpoint_*"))
    assert not list(seed.glob("*success_receipt*"))


def test_worker_rejects_input_replaced_between_parent_seal_and_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "checkout"
    root.mkdir()
    artifact = root / "oracle.json"
    artifact.write_bytes(b'{"oracle":1}\n')
    sealed_inputs = (seal_artifact(root, artifact, roles=("oracle",)),)
    lineage_sha256 = sealed_input_lineage_sha256(sealed_inputs)
    artifact.write_bytes(b'{"oracle":2}\n')
    manifest = {"construction_sentinel": True}
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    plan = supervision_module.TrainingPlan(
        seed=11,
        transitions=8_192,
        manifest_sha256=manifest_sha256,
        evidence_class="interface_check",
        promotable=False,
        smoke=False,
        test_only=True,
    )
    request = supervision_module.WorkerRequest(
        plan=plan,
        output_directory=str(tmp_path / "output"),
        execution_manifest_bytes=manifest_bytes,
        execution_manifest_sha256=manifest_sha256,
        e1_receipt_sha256="1" * 64,
        source_snapshot_sha256="2" * 64,
        sealed_inputs=sealed_inputs,
        sealed_input_lineage_sha256=lineage_sha256,
        repository_root=str(root),
        runtime_kind="fake",
        runtime_config=None,
        failure_mode=None,
        rss_limit_bytes=1_000,
        cpu_time_limit_seconds=100.0,
        worker_environment=(),
    )
    connection = _ScriptedConnection([])
    construction_calls = 0

    def construction_sentinel(_connection: object, _request: object) -> None:
        nonlocal construction_calls
        construction_calls += 1

    monkeypatch.setattr(supervision_module.os, "setsid", lambda: None)
    monkeypatch.setattr(supervision_module, "_environment_control", lambda _request: {})
    monkeypatch.setattr(supervision_module, "_apply_cpu_time_limit", lambda _seconds: {})
    monkeypatch.setattr(supervision_module, "_worker_execute", construction_sentinel)

    with pytest.raises(supervision_module.SourceMutationError, match="changed after preflight"):
        supervision_module._worker_session_entry(connection, request)

    assert construction_calls == 0
    assert connection.closed is True
    sent = [json.loads(frame) for frame in connection.sent]
    assert [frame["message_type"] for frame in sent] == ["failed"]
    assert sent[0]["payload"]["status"] == "source_mutation"
    assert not list((tmp_path / "output").glob("*success_receipt*"))


def test_worker_rejects_switched_import_root_before_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "checkout"
    source = root / "src/oracle_composition/example.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"VALUE = 1\n")
    switched = tmp_path / "switched/example.py"
    switched.parent.mkdir(parents=True)
    switched.write_bytes(source.read_bytes())
    sealed_inputs = (seal_artifact(root, source, roles=("source_fixture",)),)
    lineage_sha256 = sealed_input_lineage_sha256(sealed_inputs)
    relative = source.relative_to(root).as_posix()
    snapshot = supervision_module.RuntimeSourceSnapshot(
        {"source_sha256": {relative: hashlib.sha256(source.read_bytes()).hexdigest()}},
        "2" * 64,
    )
    manifest = {"construction_sentinel": True}
    manifest_bytes = canonical_json_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    plan = supervision_module.TrainingPlan(
        seed=11,
        transitions=8_192,
        manifest_sha256=manifest_sha256,
        evidence_class="interface_check",
        promotable=False,
        smoke=False,
        test_only=True,
    )
    request = supervision_module.WorkerRequest(
        plan=plan,
        output_directory=str(tmp_path / "output"),
        execution_manifest_bytes=manifest_bytes,
        execution_manifest_sha256=manifest_sha256,
        e1_receipt_sha256="1" * 64,
        source_snapshot_sha256=snapshot.sha256,
        sealed_inputs=sealed_inputs,
        sealed_input_lineage_sha256=lineage_sha256,
        repository_root=str(root),
        runtime_kind="fake",
        runtime_config=None,
        failure_mode=None,
        rss_limit_bytes=1_000,
        cpu_time_limit_seconds=100.0,
        worker_environment=(),
    )
    connection = _ScriptedConnection([])
    construction_calls = 0
    validate_modules = supervision_module.validate_executing_modules

    def construction_sentinel(_connection: object, _request: object) -> None:
        nonlocal construction_calls
        construction_calls += 1

    def switched_modules(
        repository_root: Path,
        source_sha256: object,
    ) -> dict[str, object]:
        return validate_modules(
            repository_root,
            source_sha256,
            module_files={"oracle_composition.example": switched},
        )

    monkeypatch.setattr(supervision_module.os, "setsid", lambda: None)
    monkeypatch.setattr(supervision_module, "_environment_control", lambda _request: {})
    monkeypatch.setattr(supervision_module, "_apply_cpu_time_limit", lambda _seconds: {})
    monkeypatch.setattr(
        supervision_module, "inspect_runtime_sources", lambda *_args, **_kw: snapshot
    )
    monkeypatch.setattr(supervision_module, "validate_executing_modules", switched_modules)
    monkeypatch.setattr(supervision_module, "_worker_execute", construction_sentinel)

    with pytest.raises(
        supervision_module.SourceMutationError, match="outside the declared checkout"
    ):
        supervision_module._worker_session_entry(connection, request)

    assert construction_calls == 0
    assert connection.closed is True
    sent = [json.loads(frame) for frame in connection.sent]
    assert [frame["message_type"] for frame in sent] == ["failed"]
    assert sent[0]["payload"]["status"] == "source_mutation"
    assert not list((tmp_path / "output").glob("*success_receipt*"))


@pytest.mark.parametrize(
    "failure_mode",
    ("rsi_get_attr_failure", "rsi_none", "rsi_omitted_reset", "rsi_empty"),
)
def test_rsi_evidence_failures_are_counter_drift_without_checkpoint_or_success(
    tmp_path: Path,
    preflight: object,
    failure_mode: str,
) -> None:
    output = tmp_path / failure_mode
    result = _run(preflight, output, failure_mode=failure_mode)
    assert result.status is SupervisorStatus.COUNTER_DRIFT
    seed = output / "seed_11"
    assert not list(seed.glob("checkpoint_*"))
    assert not list(seed.glob("*success_receipt*"))
    receipt = json.loads((seed / "failure_receipt_v2.json").read_bytes())
    assert receipt["primary_failure"]["status"] == "counter_drift"


@pytest.mark.parametrize(
    ("failure_mode", "status", "seed_wall"),
    [
        ("hang", SupervisorStatus.TIMEOUT, 0.25),
        ("crash", SupervisorStatus.CRASH, 20.0),
        ("non_finite", SupervisorStatus.NON_FINITE, 20.0),
        ("likelihood_failure", SupervisorStatus.LIKELIHOOD_FAILURE, 20.0),
        ("counter_drift", SupervisorStatus.COUNTER_DRIFT, 20.0),
        ("action_bound_violation", SupervisorStatus.ACTION_BOUND_VIOLATION, 20.0),
        ("phase_selection", SupervisorStatus.PHASE_SELECTION_FAILURE, 20.0),
    ],
)
def test_controlled_failure_status_has_no_success_and_accounts_for_all_seeds(
    tmp_path: Path,
    preflight: object,
    failure_mode: str,
    status: SupervisorStatus,
    seed_wall: float,
) -> None:
    output = tmp_path / failure_mode
    result = _run(preflight, output, failure_mode=failure_mode, seed_wall=seed_wall)
    assert result.status == status
    assert [(item.seed, item.status) for item in result.outcomes] == [
        (11, status),
        (13, SupervisorStatus.NOT_STARTED),
    ]
    for seed in (11, 13):
        seed_directory = output / f"seed_{seed}"
        assert not (seed_directory / "success_receipt_v2.json").exists()
        assert (seed_directory / "failure_receipt_v2.json").is_file()
        assert len(list(seed_directory.glob("*success_receipt*"))) == 0
        assert len(list(seed_directory.glob("*failure_receipt*"))) == 1


def test_output_is_fresh_and_no_overwrite(tmp_path: Path, preflight: object) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    with pytest.raises(ExperimentContractError, match="fresh and no-overwrite"):
        _run(preflight, output)


def test_mailbox_reservation_requires_the_complete_frozen_declaration(tmp_path: Path) -> None:
    output = (tmp_path / "reserved-output").resolve()
    mailbox = tmp_path / "mailbox"
    (mailbox / "messages").mkdir(parents=True)
    (mailbox / "acks").mkdir()
    proposal_id = "20260906T000000.000000Z-" + "1" * 32
    message_id = "20260906T000001.000000Z-" + "2" * 32
    accepted_until = (datetime.now(UTC) + timedelta(minutes=20)).isoformat().replace("+00:00", "Z")
    authoritative = {
        "accepted_until_utc": accepted_until,
        "canonical_argv": ["python", "-m", "oracle_composition.harness.cycle_cli", "train"],
        "commit": "1" * 40,
        "conflict_check": "no_other_heavy_repository_job",
        "hard_wall_seconds": 1_200,
        "inputs": {"oracle_file_sha256": "2" * 64},
        "mode": "smoke",
        "output": str(output),
        "owner": "phase-b-smoke-owner",
        "proposal_id": proposal_id,
        "required_authorizer": "fable",
    }
    message_path = mailbox / "messages" / f"{message_id}.json"
    message_path.write_text(
        json.dumps(
            {
                "body": canonical_json_bytes(authoritative).decode(),
                "from": "fable",
                "id": message_id,
                "kind": "acceptance",
                "reply_to": proposal_id,
                "schema_version": 1,
                "sent_at": "2026-09-06T00:00:01.000000Z",
                "subject": "Phase B smoke reservation",
                "to": "astra",
            },
            sort_keys=True,
        )
    )
    ack_path = mailbox / "acks" / f"{message_id}.json"
    ack_path.write_text(
        json.dumps(
            {
                "acknowledged_at": "2026-09-06T00:00:02.000000Z",
                "meaning": "read_not_agreement",
                "message_id": message_id,
                "recipient": "astra",
                "schema_version": 1,
            },
            sort_keys=True,
        )
    )
    reservation = {
        "accepted": True,
        "acceptance_message": {
            "path": f"messages/{message_id}.json",
            "sha256": hashlib.sha256(message_path.read_bytes()).hexdigest(),
        },
        "acknowledgment": {
            "path": f"acks/{message_id}.json",
            "sha256": hashlib.sha256(ack_path.read_bytes()).hexdigest(),
        },
        "schema_version": 2,
        **authoritative,
    }
    assert (
        validate_reservation(
            reservation,
            smoke=True,
            output_directory=output,
            test_only=False,
            mailbox_root=mailbox,
        )
        == reservation
    )
    incomplete = dict(reservation)
    incomplete.pop("acceptance_message")
    with pytest.raises(ExperimentContractError, match="missing or not accepted"):
        validate_reservation(
            incomplete,
            smoke=True,
            output_directory=output,
            test_only=False,
            mailbox_root=mailbox,
        )
    with pytest.raises(ExperimentContractError, match="command differs"):
        validate_reservation(
            reservation,
            smoke=True,
            output_directory=output,
            test_only=False,
            canonical_argv=["different"],
            mailbox_root=mailbox,
        )
    with pytest.raises(ExperimentContractError, match="stale"):
        validate_reservation(
            reservation,
            smoke=True,
            output_directory=output,
            test_only=False,
            now=datetime.now(UTC) + timedelta(hours=1),
            mailbox_root=mailbox,
        )
    wrong_authorizer = dict(reservation)
    wrong_authorizer["required_authorizer"] = "astra"
    with pytest.raises(ExperimentContractError, match=r"authorizer|acceptance"):
        validate_reservation(
            wrong_authorizer,
            smoke=True,
            output_directory=output,
            test_only=False,
            mailbox_root=mailbox,
        )
    unrecognized = dict(reservation)
    unrecognized["acceptance_message"] = {
        "path": "messages/20260906T000003.000000Z-" + "3" * 32 + ".json",
        "sha256": "3" * 64,
    }
    with pytest.raises(ExperimentContractError, match="unavailable"):
        validate_reservation(
            unrecognized,
            smoke=True,
            output_directory=output,
            test_only=False,
            mailbox_root=mailbox,
        )
    assert not output.exists()
    bounded = limits_bound_by_reservation(ResourceLimits(), reservation)
    assert bounded.per_seed_wall_seconds == 20 * 60
    assert bounded.cohort_wall_seconds == 20 * 60
    assert bounded.job_wall_seconds == 20 * 60
    assert bounded.cpu_time_seconds == 20 * 60


def test_invalid_production_reservation_precedes_output_and_worker_spawn(
    tmp_path: Path,
    preflight: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "never-created"

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("output creation or worker spawn preceded reservation admission")

    monkeypatch.setattr(supervision_module, "_fresh_output", forbidden)
    monkeypatch.setattr(supervision_module, "_spawn_worker", forbidden)
    with pytest.raises(ExperimentContractError, match="missing or not accepted"):
        supervise_training_job(
            preflight=preflight,
            output_directory=output,
            seeds=(121901,),
            transitions=196_608,
            smoke=True,
            reservation={},
            limits=ResourceLimits(),
            runtime_kind="real",
            test_only=False,
            canonical_argv=[
                "python",
                "-m",
                "oracle_composition.harness.cycle_cli",
                "train",
            ],
        )
    assert not output.exists()


def _write_slot_fixture(
    root: Path,
    *,
    output: Path,
    expired: bool = False,
    mismatched_command: bool = False,
) -> dict[str, object]:
    root.mkdir()
    accepted_until = datetime.now(UTC) + timedelta(minutes=(-1 if expired else 20))
    proposal_id = "20260906T000000.000000Z-" + "1" * 32
    acceptance_id = "20260906T000001.000000Z-" + "2" * 32
    reservation = {
        "accepted": True,
        "acceptance_message": {
            "path": f"messages/{acceptance_id}.json",
            "sha256": "3" * 64,
        },
        "accepted_until_utc": accepted_until.isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        ),
        "acknowledgment": {
            "path": f"acks/{acceptance_id}.json",
            "sha256": "4" * 64,
        },
        "canonical_argv": ["python", "-m", "oracle_composition.harness.cycle_cli", "train"],
        "commit": "5" * 40,
        "conflict_check": "no_other_heavy_repository_job",
        "hard_wall_seconds": 1_200,
        "inputs": {"runtime_source_snapshot_sha256": "6" * 64},
        "mode": "smoke",
        "output": str(output.resolve()),
        "owner": "phase-b-slot-owner",
        "proposal_id": proposal_id,
        "required_authorizer": "fable",
        "schema_version": 2,
    }
    token_argv = (
        ["different-command"] if mismatched_command else list(reservation["canonical_argv"])
    )
    token = {
        "acceptance_id": acceptance_id,
        "accepted_until_utc": reservation["accepted_until_utc"],
        "canonical_argv": token_argv,
        "commit": reservation["commit"],
        "expected_wall_seconds": 1_200,
        "hard_wall_seconds": 1_200,
        "owner": reservation["owner"],
        "proposal_id": proposal_id,
        "reservation_canonical_sha256": hashlib.sha256(
            slot_canonical_json_bytes(reservation)
        ).hexdigest(),
        "schema_version": 1,
        "token_id": "7" * 32,
    }
    (root / SLOT_FILENAME).write_bytes(slot_canonical_json_bytes(token))
    return reservation


@pytest.mark.parametrize(
    "slot_state", ("missing", "malformed", "expired", "mismatched", "expected_wall")
)
def test_production_slot_refuses_before_worker_spawn(
    tmp_path: Path,
    preflight: object,
    monkeypatch: pytest.MonkeyPatch,
    slot_state: str,
) -> None:
    output = tmp_path / "slot-refusal-output"
    slot_root = tmp_path / "coordination"
    if slot_state == "missing":
        slot_root.mkdir()
        reservation = _write_slot_fixture(
            tmp_path / "unused-coordination",
            output=output,
        )
    else:
        reservation = _write_slot_fixture(
            slot_root,
            output=output,
            expired=slot_state == "expired",
            mismatched_command=slot_state == "mismatched",
        )
        if slot_state == "malformed":
            (slot_root / SLOT_FILENAME).write_bytes(b"{}")
    spawn_calls = 0

    def forbidden_spawn(_request: object) -> tuple[object, object]:
        nonlocal spawn_calls
        spawn_calls += 1
        raise AssertionError("worker spawn must not be reached")

    monkeypatch.setattr(
        supervision_module,
        "validate_reservation",
        lambda *_args, **_kwargs: reservation,
    )
    monkeypatch.setattr(supervision_module, "_assert_no_conflicting_training_process", lambda: None)
    with pytest.raises(ExperimentContractError, match="heavy-job slot refused dispatch"):
        supervise_training_job(
            preflight=preflight,
            output_directory=output,
            seeds=(121901,),
            transitions=196_608,
            smoke=True,
            reservation=reservation,
            runtime_kind="real",
            canonical_argv=reservation["canonical_argv"],
            expected_wall_seconds=1_199 if slot_state == "expected_wall" else 1_200,
            dependencies=_observer_dependencies(forbidden_spawn),
            coordination_root=slot_root,
        )
    assert spawn_calls == 0
    if slot_state != "missing":
        assert (slot_root / SLOT_FILENAME).exists()


def test_validated_slot_is_released_after_spawn_failure(
    tmp_path: Path,
    preflight: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "spawn-failure-output"
    slot_root = tmp_path / "coordination"
    reservation = _write_slot_fixture(slot_root, output=output)

    def failing_spawn(_request: object) -> tuple[object, object]:
        assert (slot_root / SLOT_FILENAME).is_file()
        raise RuntimeError("controlled spawn failure")

    monkeypatch.setattr(
        supervision_module,
        "validate_reservation",
        lambda *_args, **_kwargs: reservation,
    )
    monkeypatch.setattr(supervision_module, "_assert_no_conflicting_training_process", lambda: None)
    with pytest.raises(RuntimeError, match="controlled spawn failure"):
        supervise_training_job(
            preflight=preflight,
            output_directory=output,
            seeds=(121901,),
            transitions=196_608,
            smoke=True,
            reservation=reservation,
            runtime_kind="real",
            canonical_argv=reservation["canonical_argv"],
            expected_wall_seconds=1_200,
            dependencies=_observer_dependencies(failing_spawn),
            coordination_root=slot_root,
        )
    assert not (slot_root / SLOT_FILENAME).exists()


def test_validated_slot_is_released_after_normal_supervision_return(
    tmp_path: Path,
    preflight: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "normal-return-output"
    slot_root = tmp_path / "coordination"
    reservation = _write_slot_fixture(slot_root, output=output)
    sentinel = object()

    def controlled_core(**kwargs: object) -> object:
        session = kwargs["slot_session"]
        session.configure(slot_root, reservation, expected_wall_seconds=1_200)
        session.validate_before_spawn()
        return sentinel

    monkeypatch.setattr(supervision_module, "_supervise_training_job", controlled_core)
    result = supervise_training_job(
        preflight=preflight,
        output_directory=output,
        seeds=(121901,),
        transitions=196_608,
        smoke=True,
        reservation=reservation,
        runtime_kind="real",
        canonical_argv=reservation["canonical_argv"],
        dependencies=_observer_dependencies(lambda _request: None),
        coordination_root=slot_root,
    )
    assert result is sentinel
    assert not (slot_root / SLOT_FILENAME).exists()


def _run_controlled_slot_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    *,
    slot_root: Path,
    reservation: dict[str, object],
    dependencies: SupervisionDependencies,
    operation: object,
) -> object:
    def controlled_core(**kwargs: object) -> object:
        session = kwargs["slot_session"]
        session.configure(slot_root, reservation, expected_wall_seconds=1_200)
        session.validate_before_spawn()
        return supervision_module._with_worker_cleanup(dependencies, session, operation)

    monkeypatch.setattr(supervision_module, "_supervise_training_job", controlled_core)
    return supervise_training_job(
        preflight=object(),
        output_directory=Path(reservation["output"]),
        seeds=(121901,),
        transitions=196_608,
        smoke=True,
        reservation=reservation,
        runtime_kind="real",
        canonical_argv=reservation["canonical_argv"],
        dependencies=dependencies,
        coordination_root=slot_root,
    )


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        (KeyboardInterrupt, "controlled post-spawn interrupt"),
        (RuntimeError, "controlled initializer exception"),
    ),
)
def test_production_core_cleans_worker_before_slot_release_on_post_spawn_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException],
    message: str,
) -> None:
    output = tmp_path / failure.__name__
    slot_root = tmp_path / f"coordination-{failure.__name__}"
    reservation = _write_slot_fixture(slot_root, output=output)
    source_snapshot = supervision_module.RuntimeSourceSnapshot(
        value={"git": {"commit": reservation["commit"]}, "source_sha256": {}},
        sha256="6" * 64,
    )
    preflight = supervision_module.TrainingPreflight(
        repository_root=tmp_path,
        experiment=tmp_path,
        oracle_path=tmp_path / "oracle.json",
        reward_path=tmp_path / "reward.json",
        template_manifest=object(),
        template_manifest_sha256="8" * 64,
        e003_execution_manifest_sha256="9" * 64,
        prior_scientific_receipt_sha256="a" * 64,
        source_snapshot=source_snapshot,
        sealed_inputs=(),
        sealed_input_lineage_sha256="b" * 64,
        report_inputs={},
        runtime_config=object(),
    )
    process = _FakeProcess()
    connection = _ScriptedConnection([])
    cleanup_calls = 0

    def cleanup(worker: _FakeProcess, *, group_validated: bool) -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        assert worker is process
        assert group_validated is False
        worker.alive = False

    dependencies = _observer_dependencies(lambda _request: None, cleanup=cleanup)

    def controlled_seed(**kwargs: object) -> object:
        guard = kwargs["cleanup_guard"]
        guard.attach(process, connection)
        raise failure(message)

    monkeypatch.setattr(supervision_module, "validate_reservation", lambda *_a, **_k: reservation)
    monkeypatch.setattr(supervision_module, "verify_sealed_inputs", lambda *_a, **_k: None)
    monkeypatch.setattr(supervision_module, "validate_executing_modules", lambda *_a, **_k: None)
    monkeypatch.setattr(supervision_module, "_assert_no_conflicting_training_process", lambda: None)
    monkeypatch.setattr(
        supervision_module,
        "_execution_manifest_value",
        lambda **_kwargs: {"schema_version": 3},
    )
    monkeypatch.setattr(supervision_module, "_supervise_seed", controlled_seed)

    with pytest.raises(failure, match=message):
        supervise_training_job(
            preflight=preflight,
            output_directory=output,
            seeds=(121901,),
            transitions=196_608,
            smoke=True,
            reservation=reservation,
            runtime_kind="real",
            canonical_argv=reservation["canonical_argv"],
            expected_wall_seconds=1_200,
            dependencies=dependencies,
            coordination_root=slot_root,
        )
    assert cleanup_calls == 1
    assert process.alive is False
    assert connection.closed is True
    assert not (slot_root / SLOT_FILENAME).exists()


def test_unverified_worker_cleanup_retains_exact_slot_for_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "cleanup-survives"
    slot_root = tmp_path / "coordination-cleanup-survives"
    reservation = _write_slot_fixture(slot_root, output=output)
    process = _FakeProcess()
    connection = _ScriptedConnection([])

    def cleanup(_worker: object, *, group_validated: bool) -> None:
        assert group_validated is False
        raise RuntimeError("controlled worker survived cleanup")

    dependencies = _observer_dependencies(lambda _request: None, cleanup=cleanup)

    def operation(guard: object) -> object:
        guard.attach(process, connection)
        raise KeyboardInterrupt("controlled interrupt")

    with pytest.raises(RuntimeError, match="survived cleanup"):
        _run_controlled_slot_lifecycle(
            monkeypatch,
            slot_root=slot_root,
            reservation=reservation,
            dependencies=dependencies,
            operation=operation,
        )
    assert process.alive is True
    assert connection.closed is True
    retained = json.loads((slot_root / SLOT_FILENAME).read_bytes())
    assert retained["owner"] == reservation["owner"]
    assert retained["token_id"] == "7" * 32


@pytest.mark.parametrize("cleanup_fails", (False, True))
def test_terminal_result_survives_cleanup_outcome_and_slot_release_tracks_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup_fails: bool,
) -> None:
    outcome = "fails" if cleanup_fails else "succeeds"
    output = tmp_path / f"cleanup-{outcome}"
    slot_root = tmp_path / f"coordination-cleanup-{outcome}"
    reservation = _write_slot_fixture(slot_root, output=output)
    process = _FakeProcess()
    connection = _ScriptedConnection([])
    sentinel = object()

    def cleanup(worker: _FakeProcess, *, group_validated: bool) -> None:
        assert worker is process
        assert group_validated is False
        if cleanup_fails:
            raise RuntimeError("controlled terminal cleanup failure")
        worker.alive = False

    dependencies = _observer_dependencies(lambda _request: None, cleanup=cleanup)

    def operation(guard: object) -> object:
        guard.attach(process, connection)
        try:
            guard.cleanup()
        except RuntimeError:
            if not cleanup_fails:
                raise
        return sentinel

    result = _run_controlled_slot_lifecycle(
        monkeypatch,
        slot_root=slot_root,
        reservation=reservation,
        dependencies=dependencies,
        operation=operation,
    )
    assert result is sentinel
    assert process.alive is cleanup_fails
    assert connection.closed is True
    assert (slot_root / SLOT_FILENAME).exists() is cleanup_fails


def test_spawn_worker_closes_pipes_and_worker_when_child_close_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Endpoint:
        def __init__(self, *, fail: bool = False) -> None:
            self.closed = False
            self.fail = fail

        def close(self) -> None:
            self.closed = True
            if self.fail:
                raise OSError("controlled child close failure")

    class Process:
        pid = 4242

        def start(self) -> None:
            return None

    parent = Endpoint()
    child = Endpoint(fail=True)
    process = Process()
    context = SimpleNamespace(
        Pipe=lambda **_kwargs: (parent, child),
        Process=lambda **_kwargs: process,
    )
    cleanup_calls = []
    monkeypatch.setattr(supervision_module.multiprocessing, "get_context", lambda _kind: context)
    monkeypatch.setattr(supervision_module, "_spawn_environment", lambda _env: nullcontext())
    monkeypatch.setattr(
        supervision_module,
        "cleanup_worker_process",
        lambda worker, *, group_validated: cleanup_calls.append((worker, group_validated)),
    )

    with pytest.raises(OSError, match="child close failure"):
        supervision_module._spawn_worker(SimpleNamespace(worker_environment=()))
    assert parent.closed is True
    assert child.closed is True
    assert cleanup_calls == [(process, False)]


def test_spawn_worker_closes_pipes_and_worker_when_process_start_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Endpoint:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class Process:
        pid = 4242

        def start(self) -> None:
            raise RuntimeError("controlled partial start failure")

    parent = Endpoint()
    child = Endpoint()
    process = Process()
    context = SimpleNamespace(
        Pipe=lambda **_kwargs: (parent, child),
        Process=lambda **_kwargs: process,
    )
    cleanup_calls = []
    monkeypatch.setattr(supervision_module.multiprocessing, "get_context", lambda _kind: context)
    monkeypatch.setattr(supervision_module, "_spawn_environment", lambda _env: nullcontext())
    monkeypatch.setattr(
        supervision_module,
        "cleanup_worker_process",
        lambda worker, *, group_validated: cleanup_calls.append((worker, group_validated)),
    )

    with pytest.raises(RuntimeError, match="partial start failure"):
        supervision_module._spawn_worker(SimpleNamespace(worker_environment=()))
    assert parent.closed is True
    assert child.closed is True
    assert cleanup_calls == [(process, False)]
