from __future__ import annotations

import multiprocessing
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from oracle_composition.experiments import tqc_development_manifest_v2 as manifest_module
from oracle_composition.experiments import tqc_development_resource_v2 as resource_module
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_development_contract_v2 import (
    EXPECTED_DISK_CHECKS,
    EXPECTED_VECTOR_STEPS,
)
from oracle_composition.experiments.tqc_development_resource_v2 import (
    EXPECTED_THROUGHPUT_WINDOWS,
    LIFECYCLE_STAGES,
    POST_TRAINING_DISK_STAGES,
    TQCResourceMonitorV2,
)


def _install_passing_measurements(monkeypatch: pytest.MonkeyPatch) -> None:
    clock = 0.0

    def perf_counter() -> float:
        nonlocal clock
        clock += 0.00001
        return clock

    monkeypatch.setattr(resource_module, "_PERF_COUNTER", perf_counter)
    monkeypatch.setattr(
        resource_module,
        "_GETRUSAGE",
        lambda _who: SimpleNamespace(ru_maxrss=100_000_000),
    )
    monkeypatch.setattr(
        resource_module,
        "_FSTATVFS",
        lambda _descriptor: SimpleNamespace(
            f_bavail=resource_module.FREE_DISK_MINIMUM_BYTES + 1_000_000,
            f_frsize=1,
        ),
    )


def _bound_monitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    install_measurements: bool = True,
) -> TQCResourceMonitorV2:
    if install_measurements:
        _install_passing_measurements(monkeypatch)
    work = tmp_path.resolve() / "attempt"
    work.mkdir(mode=0o700)
    work.chmod(0o700)
    monitor = TQCResourceMonitorV2(work)
    monitor.sample_lifecycle("preflight")

    class FakeManifest:
        def __init__(self) -> None:
            self.attempt_id = resource_module.ATTEMPT_ID
            self.claimed_work_directory_identity = monitor.claimed_work_directory_identity
            self.sha256 = "1" * 64

    monkeypatch.setattr(manifest_module, "ValidatedTQCExecutionManifestV2", FakeManifest)
    monitor.bind_execution_manifest(FakeManifest())
    return monitor


def _complete_training_prefix(monitor: TQCResourceMonitorV2) -> None:
    monitor.sample_lifecycle("post_model_construction")
    monitor.sample_lifecycle("pre_learn")
    for vector_step in range(1, EXPECTED_VECTOR_STEPS + 1):
        monitor.sample_vector_step(vector_step, vector_step * 5)
    monitor.sample_lifecycle("post_learn")


def test_complete_resource_schedule_issues_guarded_receipts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _bound_monitor(tmp_path, monkeypatch)
    _complete_training_prefix(monitor)

    training = monitor.validate_training_prefix()
    assert training.vector_sample_count == EXPECTED_VECTOR_STEPS
    assert training.throughput_window_count == EXPECTED_THROUGHPUT_WINDOWS
    assert training.disk_check_count == 2_001
    with pytest.raises(ExperimentContractError, match="only be issued"):
        replace(training)

    for stage in LIFECYCLE_STAGES[4:]:
        if stage == "pre_model_save":
            monitor.sample_post_training_disk_gate("pre_persistence")
        elif stage == "pre_evaluation":
            monitor.sample_post_training_disk_gate("pre_evaluation")
        elif stage == "pre_finalization":
            monitor.sample_post_training_disk_gate("finalization")
        monitor.sample_lifecycle(stage)

    receipt = monitor.finalize()
    assert receipt.lifecycle_stages == LIFECYCLE_STAGES
    assert receipt.post_training_disk_stages == POST_TRAINING_DISK_STAGES
    assert receipt.disk_check_count == EXPECTED_DISK_CHECKS
    assert receipt.all_resource_gates_passed is True
    assert len(receipt.lifecycle_samples) == 16
    assert len(receipt.throughput_windows) == 99
    assert len(receipt.post_training_disk_observations) == 3
    assert receipt.training_prefix_event_sha256 == training.prefix_event_sha256
    assert receipt.execution_manifest_sha256 == training.execution_manifest_sha256
    assert receipt.worker_pid == training.worker_pid
    assert receipt.claimed_work_directory_identity == training.claimed_work_directory_identity
    assert receipt.to_dict()["event_sha256"] == receipt.event_sha256
    with pytest.raises(ExperimentContractError, match="only be issued"):
        replace(receipt)
    with pytest.raises(ExperimentContractError, match="already finalized"):
        monitor.finalize()


def test_resource_monitor_rejects_reordered_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_passing_measurements(monkeypatch)
    work = tmp_path.resolve() / "attempt"
    work.mkdir(mode=0o700)
    monitor = TQCResourceMonitorV2(work)

    with pytest.raises(ExperimentContractError, match="reordered"):
        monitor.sample_lifecycle("pre_learn")


def test_resource_monitor_rejects_skipped_vector_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _bound_monitor(tmp_path, monkeypatch)
    monitor.sample_lifecycle("post_model_construction")
    monitor.sample_lifecycle("pre_learn")

    with pytest.raises(ExperimentContractError, match="vector counter"):
        monitor.sample_vector_step(2, 10)


def test_training_authority_requires_complete_schedule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _bound_monitor(tmp_path, monkeypatch)
    monitor.sample_lifecycle("post_model_construction")
    monitor.sample_lifecycle("pre_learn")

    with pytest.raises(ExperimentContractError, match="incomplete"):
        monitor.validate_training_prefix()


def test_post_training_disk_gates_are_ordered(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _bound_monitor(tmp_path, monkeypatch)

    with pytest.raises(ExperimentContractError, match="reordered"):
        monitor.sample_post_training_disk_gate("pre_evaluation")


@pytest.mark.parametrize("vector_step,environment_steps", [(True, 5), (1, True), (1, 4)])
def test_vector_counters_require_exact_integer_semantics(
    tmp_path: Path,
    vector_step: object,
    environment_steps: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _bound_monitor(tmp_path, monkeypatch)
    monitor.sample_lifecycle("post_model_construction")
    monitor.sample_lifecycle("pre_learn")

    with pytest.raises(ExperimentContractError, match="vector counter"):
        monitor.sample_vector_step(vector_step, environment_steps)


def test_resource_output_path_must_be_absolute() -> None:
    with pytest.raises(ExperimentContractError, match="absolute"):
        TQCResourceMonitorV2(Path("relative"))


def test_peak_rss_gate_fails_at_the_observed_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resource_module,
        "_GETRUSAGE",
        lambda _who: SimpleNamespace(ru_maxrss=resource_module.PEAK_RSS_LIMIT_BYTES + 1),
    )
    work = tmp_path.resolve() / "attempt"
    work.mkdir(mode=0o700)
    monitor = TQCResourceMonitorV2(work)

    with pytest.raises(ExperimentContractError, match="peak RSS"):
        monitor.sample_lifecycle("preflight")


def test_free_disk_gate_fails_at_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resource_module,
        "_FSTATVFS",
        lambda _descriptor: SimpleNamespace(
            f_bavail=resource_module.FREE_DISK_MINIMUM_BYTES - 1,
            f_frsize=1,
        ),
    )
    work = tmp_path.resolve() / "attempt"
    work.mkdir(mode=0o700)
    monitor = TQCResourceMonitorV2(work)

    with pytest.raises(ExperimentContractError, match="free disk"):
        monitor.sample_lifecycle("preflight")


def test_training_wall_gate_starts_at_pre_learn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = iter((0.0, 20_000.0, 30_000.0, 40_000.0, 50_801.0))
    monkeypatch.setattr(resource_module, "_PERF_COUNTER", lambda: next(observed))
    work = tmp_path.resolve() / "attempt"
    work.mkdir(mode=0o700)
    monitor = TQCResourceMonitorV2(work)
    monitor.sample_lifecycle("preflight")

    class FakeManifest:
        def __init__(self) -> None:
            self.attempt_id = resource_module.ATTEMPT_ID
            self.claimed_work_directory_identity = monitor.claimed_work_directory_identity
            self.sha256 = "1" * 64

    monkeypatch.setattr(manifest_module, "ValidatedTQCExecutionManifestV2", FakeManifest)
    monitor.bind_execution_manifest(FakeManifest())
    monitor.sample_lifecycle("post_model_construction")
    monitor.sample_lifecycle("pre_learn")

    with pytest.raises(ExperimentContractError, match="wall time"):
        monitor.sample_vector_step(1, 5)


def test_failed_disk_gate_permanently_poisoned_monitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _bound_monitor(tmp_path, monkeypatch)
    _complete_training_prefix(monitor)
    low_disk = SimpleNamespace(free=resource_module.FREE_DISK_MINIMUM_BYTES - 1)
    monkeypatch.setattr(
        resource_module,
        "_FSTATVFS",
        lambda _descriptor: SimpleNamespace(f_bavail=low_disk.free, f_frsize=1),
    )

    with pytest.raises(ExperimentContractError, match="free disk"):
        monitor.sample_post_training_disk_gate("pre_persistence")
    monkeypatch.setattr(
        resource_module,
        "_FSTATVFS",
        lambda _descriptor: SimpleNamespace(
            f_bavail=resource_module.FREE_DISK_MINIMUM_BYTES + 1,
            f_frsize=1,
        ),
    )
    with pytest.raises(ExperimentContractError, match="permanently failed"):
        monitor.sample_post_training_disk_gate("pre_persistence")
    with pytest.raises(ExperimentContractError, match="permanently failed"):
        monitor.finalize()


def test_disk_gate_cannot_be_taken_early_then_reused_at_later_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _bound_monitor(tmp_path, monkeypatch)
    _complete_training_prefix(monitor)
    monitor.sample_post_training_disk_gate("pre_persistence")

    with pytest.raises(ExperimentContractError, match="lifecycle point"):
        monitor.sample_post_training_disk_gate("pre_evaluation")


def test_work_directory_identity_change_poisoned_monitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _bound_monitor(tmp_path, monkeypatch)
    work = tmp_path.resolve() / "attempt"
    work.chmod(0o755)

    with pytest.raises(ExperimentContractError, match=r"owned 0700|identity changed"):
        monitor.sample_lifecycle("post_model_construction")
    with pytest.raises(ExperimentContractError, match="permanently failed"):
        monitor.sample_lifecycle("post_model_construction")


def test_directory_replace_and_restore_cannot_change_disk_measurement_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _bound_monitor(tmp_path, monkeypatch)
    work = tmp_path.resolve() / "attempt"
    moved = tmp_path.resolve() / "original"
    work.rename(moved)
    work.mkdir(mode=0o700)
    work.chmod(0o700)

    with pytest.raises(ExperimentContractError, match="identity changed"):
        monitor.sample_lifecycle("post_model_construction")

    work.rmdir()
    moved.rename(work)
    with pytest.raises(ExperimentContractError, match="permanently failed"):
        monitor.sample_lifecycle("post_model_construction")


def test_backward_clock_permanently_fails_monitor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = iter((5.0, 4.0))
    monkeypatch.setattr(resource_module, "_PERF_COUNTER", lambda: next(observed))
    monkeypatch.setattr(
        resource_module,
        "_FSTATVFS",
        lambda _descriptor: SimpleNamespace(
            f_bavail=resource_module.FREE_DISK_MINIMUM_BYTES + 1,
            f_frsize=1,
        ),
    )
    work = tmp_path.resolve() / "attempt"
    work.mkdir(mode=0o700)
    work.chmod(0o700)
    monitor = TQCResourceMonitorV2(work)

    with pytest.raises(ExperimentContractError, match="moved backward"):
        monitor.sample_lifecycle("preflight")
    with pytest.raises(ExperimentContractError, match="permanently failed"):
        monitor.sample_lifecycle("preflight")


def _use_monitor_in_fork(monitor: TQCResourceMonitorV2, sender: object) -> None:
    try:
        monitor.sample_lifecycle("post_model_construction")
    except BaseException as exc:
        sender.send((type(exc).__name__, str(exc)))
    else:
        sender.send(("none", ""))
    finally:
        sender.close()


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="fork unavailable"
)
def test_resource_monitor_cannot_cross_a_forked_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor = _bound_monitor(tmp_path, monkeypatch)
    receiver, sender = multiprocessing.get_context("fork").Pipe(duplex=False)
    process = multiprocessing.get_context("fork").Process(
        target=_use_monitor_in_fork,
        args=(monitor, sender),
    )
    process.start()
    sender.close()
    try:
        error_type, message = receiver.recv()
    finally:
        receiver.close()
        process.join(timeout=5.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=5.0)

    assert process.exitcode == 0
    assert error_type == "ExperimentContractError"
    assert "process boundary" in message


def test_throughput_floor_failure_is_observed_at_first_complete_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = 0.0

    def perf_counter() -> float:
        nonlocal clock
        clock += 1.0
        return clock

    monkeypatch.setattr(resource_module, "_PERF_COUNTER", perf_counter)
    monkeypatch.setattr(
        resource_module,
        "_GETRUSAGE",
        lambda _who: SimpleNamespace(ru_maxrss=100_000_000),
    )
    monkeypatch.setattr(
        resource_module,
        "_FSTATVFS",
        lambda _descriptor: SimpleNamespace(
            f_bavail=resource_module.FREE_DISK_MINIMUM_BYTES + 1,
            f_frsize=1,
        ),
    )
    monitor = _bound_monitor(tmp_path, monkeypatch, install_measurements=False)
    monitor.sample_lifecycle("post_model_construction")
    monitor.sample_lifecycle("pre_learn")
    for vector_step in range(1, 4_000):
        monitor.sample_vector_step(vector_step, vector_step * 5)

    with pytest.raises(ExperimentContractError, match="throughput"):
        monitor.sample_vector_step(4_000, 20_000)
