from __future__ import annotations

import inspect
import json
import multiprocessing
import os
import time
from contextlib import suppress
from multiprocessing import reduction

import pytest

from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_calibration_contract import canonical_json
from oracle_composition.experiments.tqc_development_channel_v2 import (
    ATTEMPT_ID,
    MAX_MESSAGE_BYTES,
    TQCWorkerChannelV2,
)


def _channels() -> tuple[TQCWorkerChannelV2, TQCWorkerChannelV2]:
    parent_read, worker_write = os.pipe()
    worker_read, parent_write = os.pipe()
    pid = max(os.getpid(), 2)
    parent: TQCWorkerChannelV2 | None = None
    worker: TQCWorkerChannelV2 | None = None
    try:
        parent = TQCWorkerChannelV2(parent_read, parent_write, "parent", pid)
        worker = TQCWorkerChannelV2(worker_read, worker_write, "worker", pid)
        return parent, worker
    except BaseException:
        if parent is not None:
            parent.close()
        if worker is not None:
            worker.close()
        raise
    finally:
        for descriptor in (parent_read, worker_write, worker_read, parent_write):
            os.close(descriptor)


def _spawn_channel_worker(read_handle: object, write_handle: object) -> None:
    read_descriptor = read_handle.detach()
    write_descriptor = write_handle.detach()
    channel: TQCWorkerChannelV2 | None = None
    try:
        channel = TQCWorkerChannelV2(
            read_descriptor,
            write_descriptor,
            "worker",
            os.getpid(),
        )
    finally:
        os.close(read_descriptor)
        os.close(write_descriptor)
    try:
        deadline = time.perf_counter() + 5.0
        channel.send("worker_started", "preflight", {"spawned": True}, deadline=deadline)
        received = channel.receive(deadline=deadline)
        if received["message_type"] != "request_shutdown":
            raise RuntimeError("spawned worker received the wrong parent message")
    finally:
        channel.close()


def test_two_pipe_channel_uses_independent_contiguous_sequences() -> None:
    parent, worker = _channels()
    try:
        deadline = time.perf_counter() + 1.0
        worker.send("worker_started", "preflight", {"ready": True}, deadline=deadline)
        first = parent.receive(deadline=deadline)
        parent.send("request_shutdown", "preflight", {"reason": "test"}, deadline=deadline)
        reverse = worker.receive(deadline=deadline)
        worker.send("stage_entered", "preflight", {}, deadline=deadline)
        second = parent.receive(deadline=deadline)

        assert first["sequence_index"] == 0
        assert reverse["sequence_index"] == 0
        assert second["sequence_index"] == 1
        assert parent.send_sequence == 1
        assert parent.receive_sequence == 2
        assert worker.send_sequence == 2
        assert worker.receive_sequence == 1
        assert parent.transcript_frame_count == 3
        assert worker.transcript_frame_count == 3
        assert parent.transcript_sha256 == worker.transcript_sha256
    finally:
        parent.close()
        worker.close()


def test_channel_sequence_and_failure_state_cannot_be_injected() -> None:
    parameters = inspect.signature(TQCWorkerChannelV2).parameters
    assert "_send_sequence" not in parameters
    assert "_receive_sequence" not in parameters
    assert "_closed" not in parameters
    assert "_failed_reason" not in parameters
    assert "_transcript_frame_count" not in parameters


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    (("role", "worker"), ("worker_pid", 99_999), ("attempt_id", "changed")),
)
def test_channel_rejects_post_construction_identity_relabeling(
    field_name: str,
    changed_value: object,
) -> None:
    parent, worker = _channels()
    try:
        setattr(parent, field_name, changed_value)
        with pytest.raises(ExperimentContractError, match="process or identity changed"):
            parent.send(
                "request_shutdown",
                "preflight",
                {"reason": "identity-test"},
                deadline=time.perf_counter() + 1.0,
            )
    finally:
        parent.close()
        worker.close()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_channel_cannot_be_used_by_a_forked_descendant() -> None:
    parent, worker = _channels()
    result_read, result_write = os.pipe()
    child_pid = os.fork()
    if child_pid == 0:  # pragma: no cover - asserted through the child receipt
        os.close(result_read)
        try:
            worker.send(
                "worker_started",
                "preflight",
                {},
                deadline=time.perf_counter() + 1.0,
            )
        except ExperimentContractError as exc:
            os.write(result_write, str(exc).encode())
        else:
            os.write(result_write, b"unexpected-success")
        finally:
            os.close(result_write)
            os._exit(0)
    os.close(result_write)
    try:
        result = os.read(result_read, 4096)
        waited_pid, status = os.waitpid(child_pid, 0)
        assert waited_pid == child_pid
        assert os.waitstatus_to_exitcode(status) == 0
        assert b"process or identity changed" in result
        assert b"unexpected-success" not in result
    finally:
        os.close(result_read)
        parent.close()
        worker.close()


def test_two_pipe_channel_operates_across_a_real_spawned_process() -> None:
    context = multiprocessing.get_context("spawn")
    parent_read, worker_write = os.pipe()
    worker_read, parent_write = os.pipe()
    process = context.Process(
        target=_spawn_channel_worker,
        args=(reduction.DupFd(worker_read), reduction.DupFd(worker_write)),
    )
    parent: TQCWorkerChannelV2 | None = None
    try:
        process.start()
        if process.pid is None:
            raise RuntimeError("spawned worker PID is absent")
        parent = TQCWorkerChannelV2(parent_read, parent_write, "parent", process.pid)
        for descriptor in (parent_read, worker_write, worker_read, parent_write):
            with suppress(OSError):
                os.close(descriptor)
        first = parent.receive(deadline=time.perf_counter() + 5.0)
        assert first["worker_pid"] == process.pid
        assert first["payload"] == {"spawned": True}
        parent.send(
            "request_shutdown",
            "preflight",
            {"reason": "spawn-test-complete"},
            deadline=time.perf_counter() + 5.0,
        )
        process.join(timeout=10.0)
        assert not process.is_alive()
        assert process.exitcode == 0
    finally:
        for descriptor in (parent_read, worker_write, worker_read, parent_write):
            with suppress(OSError):
                os.close(descriptor)
        if parent is not None:
            parent.close()
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)


def test_channel_rejects_wrong_direction_without_writing() -> None:
    parent, worker = _channels()
    try:
        with pytest.raises(ExperimentContractError, match="type is not allowed"):
            parent.send(
                "worker_started",
                "preflight",
                {},
                deadline=time.perf_counter() + 1.0,
            )
        assert parent.send_sequence == 0
    finally:
        parent.close()
        worker.close()


def test_receiver_rejects_noncanonical_json() -> None:
    parent, worker = _channels()
    try:
        value = {
            "schema_version": 1,
            "sequence_index": 0,
            "attempt_id": ATTEMPT_ID,
            "worker_pid": max(os.getpid(), 2),
            "monotonic_seconds": 1.0,
            "message_type": "worker_started",
            "stage": "preflight",
            "payload": {},
        }
        encoded = json.dumps(value, sort_keys=False, indent=1).encode()
        os.write(worker.write_descriptor, len(encoded).to_bytes(8, "big") + encoded)

        with pytest.raises(ExperimentContractError, match="not canonical"):
            parent.receive(deadline=time.perf_counter() + 1.0)
        assert parent.receive_sequence == 0
    finally:
        parent.close()
        worker.close()


def test_receiver_rejects_replayed_sequence() -> None:
    parent, worker = _channels()
    try:
        deadline = time.perf_counter() + 1.0
        worker.send("worker_started", "preflight", {}, deadline=deadline)
        parent.receive(deadline=deadline)
        value = {
            "schema_version": 1,
            "sequence_index": 0,
            "attempt_id": ATTEMPT_ID,
            "worker_pid": max(os.getpid(), 2),
            "monotonic_seconds": 1.0,
            "message_type": "stage_entered",
            "stage": "preflight",
            "payload": {},
        }
        encoded = canonical_json(value)
        os.write(worker.write_descriptor, len(encoded).to_bytes(8, "big") + encoded)

        with pytest.raises(ExperimentContractError, match="not contiguous"):
            parent.receive(deadline=deadline)
    finally:
        parent.close()
        worker.close()


def test_receiver_rejects_oversized_frame_before_payload_read() -> None:
    parent, worker = _channels()
    try:
        os.write(worker.write_descriptor, (MAX_MESSAGE_BYTES + 1).to_bytes(8, "big"))

        with pytest.raises(ExperimentContractError, match="length is invalid"):
            parent.receive(deadline=time.perf_counter() + 1.0)
        assert parent.failed_reason is not None
        with pytest.raises(ExperimentContractError, match="permanently failed"):
            parent.receive(deadline=time.perf_counter() + 1.0)
    finally:
        parent.close()
        worker.close()


def test_receiver_times_out_without_a_frame() -> None:
    parent, worker = _channels()
    try:
        with pytest.raises(ExperimentContractError, match="deadline expired"):
            parent.receive(deadline=time.perf_counter() + 0.01)
    finally:
        parent.close()
        worker.close()


def test_send_rejects_payload_over_limit() -> None:
    parent, worker = _channels()
    try:
        with pytest.raises(ExperimentContractError, match="byte limit"):
            worker.send(
                "monitor_sample",
                "training",
                {"value": "x" * MAX_MESSAGE_BYTES},
                deadline=time.perf_counter() + 1.0,
            )
        assert worker.failed_reason is not None
        with pytest.raises(ExperimentContractError, match="permanently failed"):
            worker.send(
                "worker_started",
                "preflight",
                {},
                deadline=time.perf_counter() + 1.0,
            )
    finally:
        parent.close()
        worker.close()


def test_channel_rejects_regular_file_descriptors(tmp_path) -> None:
    read_descriptor = os.open(tmp_path / "read", os.O_CREAT | os.O_RDONLY, 0o600)
    write_descriptor = os.open(tmp_path / "write", os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        with pytest.raises(ExperimentContractError, match="pipe endpoints"):
            TQCWorkerChannelV2(
                read_descriptor,
                write_descriptor,
                "parent",
                max(os.getpid(), 2),
            )
    finally:
        os.close(read_descriptor)
        os.close(write_descriptor)


def test_channel_rejects_read_and_write_ends_of_one_pipe() -> None:
    read_descriptor, write_descriptor = os.pipe()
    try:
        with pytest.raises(ExperimentContractError, match="two distinct dedicated pipes"):
            TQCWorkerChannelV2(
                read_descriptor,
                write_descriptor,
                "parent",
                max(os.getpid(), 2),
            )
    finally:
        os.close(read_descriptor)
        os.close(write_descriptor)


def test_worker_channel_rejects_a_nonself_worker_pid() -> None:
    parent_read, worker_write = os.pipe()
    worker_read, parent_write = os.pipe()
    try:
        with pytest.raises(ExperimentContractError, match="current process PID"):
            TQCWorkerChannelV2(
                worker_read,
                worker_write,
                "worker",
                os.getpid() + 100_000,
            )
    finally:
        for descriptor in (parent_read, worker_write, worker_read, parent_write):
            os.close(descriptor)


def test_channel_never_writes_to_or_closes_a_reused_descriptor() -> None:
    parent, worker = _channels()
    replacement_descriptors: set[int] = set()
    try:
        replaced_descriptor = parent.write_descriptor
        os.close(replaced_descriptor)
        replacement_read, replacement_write = os.pipe()
        replacement_descriptors.update((replacement_read, replacement_write))
        if replacement_write != replaced_descriptor:
            os.dup2(replacement_write, replaced_descriptor)
            replacement_descriptors.add(replaced_descriptor)

        with pytest.raises(ExperimentContractError, match="descriptor identity changed"):
            parent.send(
                "request_shutdown",
                "preflight",
                {},
                deadline=time.perf_counter() + 1.0,
            )
        os.fstat(replaced_descriptor)
        assert parent.failed_reason is not None
    finally:
        parent.close()
        worker.close()
        for descriptor in replacement_descriptors:
            with suppress(OSError):
                os.close(descriptor)
