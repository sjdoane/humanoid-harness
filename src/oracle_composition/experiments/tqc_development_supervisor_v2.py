"""One-worker supervision and two-phase preflight delivery for TQC v2."""

from __future__ import annotations

import hashlib
import math
import multiprocessing
import os
import signal
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import InitVar, dataclass, field
from multiprocessing import reduction
from pathlib import Path
from typing import Any

from .artifact_io import read_verified_artifact_bytes
from .fixed_reference import ExperimentContractError
from .tqc_calibration_contract import canonical_json
from .tqc_development_channel_v2 import TQCWorkerChannelV2
from .tqc_development_manifest_v2 import (
    ValidatedTQCExecutionManifestV2,
    ValidatedTQCPreflightContractV2,
)

ATTEMPT_ID = "dev1m-v2-seed-95001-attempt-01"
WORKER_STARTED_TIMEOUT_SECONDS = 30.0
TERMINATION_GRACE_SECONDS = 10.0
JOIN_TIMEOUT_SECONDS = 30.0
RECEIPT_FILENAMES = (
    "worker.bootstrap.json",
    "worker.runtime.json",
    "worker.instrumentation.json",
    "worker.preflight-set.json",
)
MAX_RECEIPT_BYTES = 512 * 1024

_DELIVERY_ISSUER = object()
_SPAWN_ISSUER = object()
_PREFLIGHT_BUNDLE_ISSUER = object()
_MANIFEST_ACK_ISSUER = object()
_MANIFEST_ACK_SEAL_ISSUER = object()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"{field} must be a lowercase SHA-256")
    return value


def _require_exact(value: object, expected: object, *, field: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ExperimentContractError(f"{field} differs")


@dataclass(frozen=True, slots=True)
class ValidatedTQCWorkerPreflightDeliveryV2:
    """Parent-observed receipt delivery; it cannot authorize construction."""

    attempt_id: str
    attempt_nonce: str
    preflight_contract_sha256: str
    claimed_work_directory_identity: str
    worker_pid: int
    worker_pgid: int
    worker_sid: int
    worker_process_start_monotonic_seconds: float
    ordered_receipt_sha256: tuple[str, str, str, str]
    preflight_receipts_message_sequence_index: int
    channel_transcript_frame_count: int
    channel_transcript_sha256: str
    channel_parent_send_sequence: int
    channel_parent_receive_sequence: int
    authorizes_model_construction: bool
    behavioral_evidence: bool
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _DELIVERY_ISSUER:
            raise ExperimentContractError(
                "worker preflight delivery may only be issued by the parent supervisor"
            )
        for name in (
            "attempt_nonce",
            "preflight_contract_sha256",
            "claimed_work_directory_identity",
            "channel_transcript_sha256",
        ):
            _require_sha256(getattr(self, name), field=name)
        if (
            self.attempt_id != ATTEMPT_ID
            or type(self.worker_pid) is not int
            or self.worker_pid <= 1
            or type(self.worker_pgid) is not int
            or type(self.worker_sid) is not int
            or not self.worker_pid == self.worker_pgid == self.worker_sid
            or type(self.worker_process_start_monotonic_seconds) is not float
            or not math.isfinite(self.worker_process_start_monotonic_seconds)
            or self.worker_process_start_monotonic_seconds < 0.0
            or type(self.ordered_receipt_sha256) is not tuple
            or len(self.ordered_receipt_sha256) != len(RECEIPT_FILENAMES)
            or self.preflight_receipts_message_sequence_index != 2
            or self.channel_parent_send_sequence != 0
            or self.channel_parent_receive_sequence != 3
            or self.channel_transcript_frame_count != 3
            or self.authorizes_model_construction is not False
            or self.behavioral_evidence is not False
        ):
            raise ExperimentContractError("worker preflight delivery authority is inconsistent")
        for index, value in enumerate(self.ordered_receipt_sha256):
            _require_sha256(value, field=f"ordered receipt SHA-256 {index}")


@dataclass(slots=True)
class SpawnedTQCWorkerV2:
    """Owned spawned process and its parent-side channel."""

    process: multiprocessing.Process = field(repr=False)
    channel: TQCWorkerChannelV2 = field(repr=False)
    parent_spawn_started_monotonic_seconds: float
    worker_pid: int
    _creator_pid: int
    _group_validated: bool = False
    _preflight_received: bool = False
    _manifest_acknowledged: bool = False
    _closed: bool = False
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _SPAWN_ISSUER:
            raise ExperimentContractError("spawned worker handles may only be issued by spawn")
        if (
            self._creator_pid != os.getpid()
            or self.process.pid != self.worker_pid
            or self.worker_pid <= 1
            or type(self.channel) is not TQCWorkerChannelV2
            or self.channel.role != "parent"
            or self.channel.worker_pid != self.worker_pid
            or type(self.parent_spawn_started_monotonic_seconds) is not float
            or not math.isfinite(self.parent_spawn_started_monotonic_seconds)
            or self.parent_spawn_started_monotonic_seconds < 0.0
        ):
            raise ExperimentContractError("spawned worker handle is inconsistent")

    def _assert_owner(self) -> None:
        if self._creator_pid != os.getpid():
            raise ExperimentContractError("spawned worker handle crossed its parent process")
        if self._closed:
            raise ExperimentContractError("spawned worker handle is closed")


@dataclass(frozen=True, slots=True)
class SupervisedTQCWorkerPreflightV2:
    """Exact four receipt byte strings and their parent delivery authority."""

    delivery: ValidatedTQCWorkerPreflightDeliveryV2
    worker_bootstrap_receipt_bytes: bytes = field(repr=False)
    host_runtime_receipt_bytes: bytes = field(repr=False)
    instrumentation_receipt_bytes: bytes = field(repr=False)
    worker_receipt_set_bytes: bytes = field(repr=False)
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _PREFLIGHT_BUNDLE_ISSUER:
            raise ExperimentContractError("supervised preflight bundles may only be issued once")
        values = (
            self.worker_bootstrap_receipt_bytes,
            self.host_runtime_receipt_bytes,
            self.instrumentation_receipt_bytes,
            self.worker_receipt_set_bytes,
        )
        if type(self.delivery) is not ValidatedTQCWorkerPreflightDeliveryV2:
            raise ExperimentContractError("supervised preflight delivery type differs")
        for index, (value, expected) in enumerate(
            zip(values, self.delivery.ordered_receipt_sha256, strict=True)
        ):
            if (
                type(value) is not bytes
                or not value
                or len(value) > MAX_RECEIPT_BYTES
                or hashlib.sha256(value).hexdigest() != expected
            ):
                raise ExperimentContractError(f"supervised preflight receipt {index} differs")


@dataclass(slots=True)
class _TQCWorkerManifestAckSealV2:
    """Bind a parent-observed acknowledgement to one live worker handle."""

    _payload: bytes = field(repr=False)
    _creator_pid: int
    _spawned_identity: int
    _manifest_identity: int
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _MANIFEST_ACK_SEAL_ISSUER:
            raise ExperimentContractError("worker manifest acknowledgement seals are private")
        if (
            not self._payload
            or self._creator_pid != os.getpid()
            or self._spawned_identity <= 0
            or self._manifest_identity <= 0
        ):
            raise ExperimentContractError("worker manifest acknowledgement seal is invalid")

    def validate(
        self,
        payload: dict[str, object],
        *,
        spawned: object,
        manifest: object,
    ) -> None:
        if (
            os.getpid() != self._creator_pid
            or id(spawned) != self._spawned_identity
            or id(manifest) != self._manifest_identity
            or canonical_json(payload) != self._payload
        ):
            raise ExperimentContractError("worker manifest acknowledgement seal differs")


@dataclass(frozen=True, slots=True)
class ValidatedTQCWorkerManifestAcknowledgementV2:
    """Parent-side proof that the worker re-admitted the exact final manifest."""

    attempt_id: str
    attempt_nonce: str
    preflight_contract_sha256: str
    claimed_work_directory_identity: str
    worker_pid: int
    manifest_absolute_path: str
    manifest_sha256: str
    manifest_byte_count: int
    acknowledgement_message_sequence_index: int
    channel_transcript_frame_count: int
    channel_transcript_sha256: str
    channel_parent_send_sequence: int
    channel_parent_receive_sequence: int
    worker_reverification_passed: bool
    authorizes_model_construction: bool
    behavioral_evidence: bool
    manifest: ValidatedTQCExecutionManifestV2 = field(repr=False, compare=False)
    spawned: SpawnedTQCWorkerV2 = field(repr=False, compare=False)
    _seal: _TQCWorkerManifestAckSealV2 = field(repr=False, compare=False)
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _MANIFEST_ACK_ISSUER:
            raise ExperimentContractError(
                "worker manifest acknowledgements may only be issued by supervision"
            )
        self._validate_sealed()

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"manifest", "spawned", "_seal", "_issuer"}
        }

    def _validate_sealed(self) -> None:
        if type(self._seal) is not _TQCWorkerManifestAckSealV2:
            raise ExperimentContractError("worker manifest acknowledgement seal is unavailable")
        self._seal.validate(self.to_dict(), spawned=self.spawned, manifest=self.manifest)
        if type(self.manifest) is not ValidatedTQCExecutionManifestV2:
            raise ExperimentContractError("acknowledged execution manifest type differs")
        if type(self.spawned) is not SpawnedTQCWorkerV2:
            raise ExperimentContractError("acknowledged spawned worker type differs")
        manifest_value = self.manifest.to_dict()
        if (
            self.attempt_id != ATTEMPT_ID
            or self.attempt_id != self.manifest.attempt_id
            or self.attempt_nonce != manifest_value["attempt_nonce"]
            or self.preflight_contract_sha256 != self.manifest.preflight_contract_sha256
            or self.claimed_work_directory_identity != self.manifest.claimed_work_directory_identity
            or self.worker_pid != self.spawned.worker_pid
            or self.manifest_absolute_path != manifest_value["manifest_absolute_path"]
            or self.manifest_sha256 != self.manifest.sha256
            or self.manifest_byte_count != self.manifest.byte_count
            or self.acknowledgement_message_sequence_index != 3
            or self.channel_parent_send_sequence != 1
            or self.channel_parent_receive_sequence != 4
            or self.channel_transcript_frame_count != 5
            or self.worker_reverification_passed is not True
            or self.authorizes_model_construction is not True
            or self.behavioral_evidence is not False
            or self.spawned._manifest_acknowledged is not True
        ):
            raise ExperimentContractError("worker manifest acknowledgement is inconsistent")
        for value, field_name in (
            (self.attempt_nonce, "attempt nonce"),
            (self.preflight_contract_sha256, "preflight contract SHA-256"),
            (self.claimed_work_directory_identity, "claimed work directory identity"),
            (self.manifest_sha256, "execution manifest SHA-256"),
            (self.channel_transcript_sha256, "channel transcript SHA-256"),
        ):
            _require_sha256(value, field=field_name)


def _session_worker_adapter(
    target: Callable[..., None],
    read_handle: Any,
    write_handle: Any,
    target_args: tuple[object, ...],
) -> None:
    """Enter a fresh session before exposing either channel endpoint."""

    os.setsid()
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
        target(channel, *target_args)
    finally:
        channel.close()


def _spawn_tqc_worker_process_v2(
    target: Callable[..., None],
    *,
    target_args: tuple[object, ...] = (),
) -> SpawnedTQCWorkerV2:
    """Spawn exactly one worker; the public path supplies the fixed entrypoint."""

    if not callable(target) or type(target_args) is not tuple:
        raise ExperimentContractError("worker target is invalid")
    context = multiprocessing.get_context("spawn")
    parent_read, worker_write = os.pipe()
    worker_read, parent_write = os.pipe()
    process: multiprocessing.Process | None = None
    parent_channel: TQCWorkerChannelV2 | None = None
    started = float(time.perf_counter())
    try:
        process = context.Process(
            target=_session_worker_adapter,
            args=(
                target,
                reduction.DupFd(worker_read),
                reduction.DupFd(worker_write),
                target_args,
            ),
            daemon=False,
        )
        process.start()
        if process.pid is None or process.pid <= 1:
            raise ExperimentContractError("spawned TQC worker PID is unavailable")
        parent_channel = TQCWorkerChannelV2(
            parent_read,
            parent_write,
            "parent",
            process.pid,
        )
        return SpawnedTQCWorkerV2(
            process=process,
            channel=parent_channel,
            parent_spawn_started_monotonic_seconds=started,
            worker_pid=process.pid,
            _creator_pid=os.getpid(),
            _issuer=_SPAWN_ISSUER,
        )
    except BaseException:
        if parent_channel is not None:
            parent_channel.close()
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=JOIN_TIMEOUT_SECONDS)
        raise
    finally:
        for descriptor in (parent_read, worker_write, worker_read, parent_write):
            with suppress(OSError):
                os.close(descriptor)


def spawn_tqc_worker_v2(
    preflight: ValidatedTQCPreflightContractV2,
) -> SpawnedTQCWorkerV2:
    """Spawn the one source-bound production worker after preflight publication."""

    if type(preflight) is not ValidatedTQCPreflightContractV2:
        raise ExperimentContractError("worker spawn requires exact preflight authority")
    from .tqc_development_worker_v2 import run_tqc_development_worker_v2

    return _spawn_tqc_worker_process_v2(
        run_tqc_development_worker_v2,
        target_args=(
            preflight.canonical_bytes,
            preflight.sha256,
            preflight.byte_count,
        ),
    )


def _receipt_record(value: object, *, expected_filename: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != {"filename", "sha256", "byte_count"}:
        raise ExperimentContractError("worker preflight receipt record keys differ")
    _require_exact(value["filename"], expected_filename, field="worker receipt filename")
    _require_sha256(value["sha256"], field=f"{expected_filename} SHA-256")
    if type(value["byte_count"]) is not int or not 0 < value["byte_count"] <= MAX_RECEIPT_BYTES:
        raise ExperimentContractError("worker preflight receipt byte count is invalid")
    return value


def receive_tqc_worker_preflight_v2(
    spawned: SpawnedTQCWorkerV2,
    preflight: ValidatedTQCPreflightContractV2,
    *,
    deadline: float | None = None,
) -> SupervisedTQCWorkerPreflightV2:
    """Receive sequence 0..2 and bind exact receipt bytes from the worker."""

    if type(spawned) is not SpawnedTQCWorkerV2:
        raise ExperimentContractError("worker preflight requires an exact spawned handle")
    spawned._assert_owner()
    if spawned._preflight_received:
        raise ExperimentContractError("worker preflight was already received")
    if type(preflight) is not ValidatedTQCPreflightContractV2:
        raise ExperimentContractError("worker preflight requires exact preflight authority")
    if preflight.attempt_id != ATTEMPT_ID:
        raise ExperimentContractError("worker preflight attempt differs")
    effective_deadline = (
        spawned.parent_spawn_started_monotonic_seconds + WORKER_STARTED_TIMEOUT_SECONDS
        if deadline is None
        else deadline
    )
    if type(effective_deadline) is not float or not math.isfinite(effective_deadline):
        raise ExperimentContractError("worker preflight deadline is invalid")

    started = spawned.channel.receive(deadline=effective_deadline)
    entered = spawned.channel.receive(deadline=effective_deadline)
    delivered = spawned.channel.receive(deadline=effective_deadline)
    started_payload = started["payload"]
    expected_started_keys = {
        "attempt_nonce",
        "preflight_contract_sha256",
        "claimed_work_directory_identity",
        "worker_pgid",
        "worker_sid",
        "worker_process_start_monotonic_seconds",
    }
    if (
        started["message_type"] != "worker_started"
        or started["stage"] != "preflight"
        or started["sequence_index"] != 0
        or type(started_payload) is not dict
        or set(started_payload) != expected_started_keys
    ):
        raise ExperimentContractError("worker-start handshake differs")
    if (
        entered["message_type"] != "stage_entered"
        or entered["stage"] != "preflight"
        or entered["sequence_index"] != 1
        or entered["payload"] != {"stage_index": 0}
    ):
        raise ExperimentContractError("worker preflight stage handshake differs")
    process_start = started_payload["worker_process_start_monotonic_seconds"]
    for field_name, expected in (
        ("attempt_nonce", preflight.attempt_nonce),
        ("preflight_contract_sha256", preflight.sha256),
        ("claimed_work_directory_identity", preflight.claimed_work_directory_identity),
        ("worker_pgid", spawned.worker_pid),
        ("worker_sid", spawned.worker_pid),
    ):
        _require_exact(started_payload[field_name], expected, field=f"worker start {field_name}")
    if (
        type(process_start) is not float
        or not math.isfinite(process_start)
        or process_start < spawned.parent_spawn_started_monotonic_seconds
        or process_start > started["monotonic_seconds"]
    ):
        raise ExperimentContractError("worker process start time differs")
    try:
        observed_pgid = os.getpgid(spawned.worker_pid)
        observed_sid = os.getsid(spawned.worker_pid)
    except OSError as exc:
        raise ExperimentContractError("spawned worker disappeared during preflight") from exc
    if observed_pgid != spawned.worker_pid or observed_sid != spawned.worker_pid:
        raise ExperimentContractError("spawned worker is not the observed session leader")
    spawned._group_validated = True

    delivered_payload = delivered["payload"]
    if (
        delivered["message_type"] != "preflight_receipts"
        or delivered["stage"] != "preflight"
        or delivered["sequence_index"] != 2
        or type(delivered_payload) is not dict
        or set(delivered_payload) != {"receipts"}
        or type(delivered_payload["receipts"]) is not list
        or len(delivered_payload["receipts"]) != len(RECEIPT_FILENAMES)
    ):
        raise ExperimentContractError("worker preflight receipt delivery differs")
    work_directory = Path(preflight.to_dict()["requested"]["claimed_work_directory_absolute_path"])
    records = tuple(
        _receipt_record(value, expected_filename=filename)
        for value, filename in zip(
            delivered_payload["receipts"],
            RECEIPT_FILENAMES,
            strict=True,
        )
    )
    receipt_bytes = tuple(
        read_verified_artifact_bytes(
            work_directory / str(record["filename"]),
            expected_sha256=str(record["sha256"]),
            expected_size=int(record["byte_count"]),
            max_bytes=MAX_RECEIPT_BYTES,
        )
        for record in records
    )
    ordered_sha256 = tuple(hashlib.sha256(value).hexdigest() for value in receipt_bytes)
    if ordered_sha256 != tuple(record["sha256"] for record in records):
        raise ExperimentContractError("worker preflight receipt order or bytes differ")
    delivery = ValidatedTQCWorkerPreflightDeliveryV2(
        attempt_id=preflight.attempt_id,
        attempt_nonce=preflight.attempt_nonce,
        preflight_contract_sha256=preflight.sha256,
        claimed_work_directory_identity=preflight.claimed_work_directory_identity,
        worker_pid=spawned.worker_pid,
        worker_pgid=observed_pgid,
        worker_sid=observed_sid,
        worker_process_start_monotonic_seconds=process_start,
        ordered_receipt_sha256=ordered_sha256,
        preflight_receipts_message_sequence_index=delivered["sequence_index"],
        channel_transcript_frame_count=spawned.channel.transcript_frame_count,
        channel_transcript_sha256=spawned.channel.transcript_sha256,
        channel_parent_send_sequence=spawned.channel.send_sequence,
        channel_parent_receive_sequence=spawned.channel.receive_sequence,
        authorizes_model_construction=False,
        behavioral_evidence=False,
        _issuer=_DELIVERY_ISSUER,
    )
    spawned._preflight_received = True
    return SupervisedTQCWorkerPreflightV2(
        delivery=delivery,
        worker_bootstrap_receipt_bytes=receipt_bytes[0],
        host_runtime_receipt_bytes=receipt_bytes[1],
        instrumentation_receipt_bytes=receipt_bytes[2],
        worker_receipt_set_bytes=receipt_bytes[3],
        _issuer=_PREFLIGHT_BUNDLE_ISSUER,
    )


def admit_tqc_execution_manifest_v2(
    spawned: SpawnedTQCWorkerV2,
    preflight: ValidatedTQCPreflightContractV2,
    manifest: ValidatedTQCExecutionManifestV2,
    *,
    deadline: float,
) -> ValidatedTQCWorkerManifestAcknowledgementV2:
    """Deliver one manifest and require the worker's exact acknowledgement."""

    if type(spawned) is not SpawnedTQCWorkerV2:
        raise ExperimentContractError("manifest delivery requires an exact spawned handle")
    spawned._assert_owner()
    if not spawned._preflight_received or not spawned._group_validated:
        raise ExperimentContractError("manifest delivery requires validated worker preflight")
    if spawned._manifest_acknowledged:
        raise ExperimentContractError("execution manifest was already acknowledged")
    if type(preflight) is not ValidatedTQCPreflightContractV2:
        raise ExperimentContractError("manifest delivery requires exact preflight authority")
    if type(manifest) is not ValidatedTQCExecutionManifestV2:
        raise ExperimentContractError("manifest delivery requires exact execution authority")
    if (
        manifest.attempt_id != preflight.attempt_id
        or manifest.preflight_contract_sha256 != preflight.sha256
        or manifest.claimed_work_directory_identity != preflight.claimed_work_directory_identity
    ):
        raise ExperimentContractError("execution manifest preflight binding differs")
    if type(deadline) is not float or not math.isfinite(deadline):
        raise ExperimentContractError("manifest acknowledgement deadline is invalid")
    manifest_value = manifest.to_dict()
    manifest_path = Path(manifest_value["manifest_absolute_path"])
    manifest_bytes = read_verified_artifact_bytes(
        manifest_path,
        expected_sha256=manifest.sha256,
        expected_size=manifest.byte_count,
        max_bytes=1024 * 1024,
    )
    if manifest_bytes != manifest.canonical_bytes:
        raise ExperimentContractError("execution manifest authority bytes differ from disk")
    spawned.channel.send(
        "admit_execution_manifest",
        "preflight",
        {
            "manifest_absolute_path": str(manifest_path),
            "manifest_byte_count": manifest.byte_count,
            "manifest_sha256": manifest.sha256,
        },
        deadline=deadline,
    )
    acknowledged = spawned.channel.receive(deadline=deadline)
    expected_payload = {
        "manifest_byte_count": manifest.byte_count,
        "manifest_sha256": manifest.sha256,
        "worker_reverification_passed": True,
    }
    if (
        acknowledged["message_type"] != "execution_manifest_acknowledged"
        or acknowledged["stage"] != "preflight"
        or acknowledged["sequence_index"] != 3
        or acknowledged["payload"] != expected_payload
    ):
        raise ExperimentContractError("worker execution-manifest acknowledgement differs")
    spawned._manifest_acknowledged = True
    payload: dict[str, object] = {
        "attempt_id": preflight.attempt_id,
        "attempt_nonce": preflight.attempt_nonce,
        "preflight_contract_sha256": preflight.sha256,
        "claimed_work_directory_identity": preflight.claimed_work_directory_identity,
        "worker_pid": spawned.worker_pid,
        "manifest_absolute_path": str(manifest_path),
        "manifest_sha256": manifest.sha256,
        "manifest_byte_count": manifest.byte_count,
        "acknowledgement_message_sequence_index": acknowledged["sequence_index"],
        "channel_transcript_frame_count": spawned.channel.transcript_frame_count,
        "channel_transcript_sha256": spawned.channel.transcript_sha256,
        "channel_parent_send_sequence": spawned.channel.send_sequence,
        "channel_parent_receive_sequence": spawned.channel.receive_sequence,
        "worker_reverification_passed": True,
        "authorizes_model_construction": True,
        "behavioral_evidence": False,
    }
    seal = _TQCWorkerManifestAckSealV2(
        _payload=canonical_json(payload),
        _creator_pid=os.getpid(),
        _spawned_identity=id(spawned),
        _manifest_identity=id(manifest),
        _issuer=_MANIFEST_ACK_SEAL_ISSUER,
    )
    return ValidatedTQCWorkerManifestAcknowledgementV2(
        **payload,
        manifest=manifest,
        spawned=spawned,
        _seal=seal,
        _issuer=_MANIFEST_ACK_ISSUER,
    )


def revalidate_tqc_worker_manifest_acknowledgement_v2(
    acknowledgement: ValidatedTQCWorkerManifestAcknowledgementV2,
) -> ValidatedTQCWorkerManifestAcknowledgementV2:
    """Revalidate one parent-observed manifest acknowledgement capability."""

    if type(acknowledgement) is not ValidatedTQCWorkerManifestAcknowledgementV2:
        raise ExperimentContractError("manifest acknowledgement must be the exact authority")
    acknowledgement._validate_sealed()
    return acknowledgement


def terminate_tqc_worker_v2(spawned: SpawnedTQCWorkerV2) -> None:
    """Bounded cleanup of the exact child or its validated process group."""

    if type(spawned) is not SpawnedTQCWorkerV2:
        raise ExperimentContractError("worker cleanup requires an exact spawned handle")
    spawned._assert_owner()
    process = spawned.process
    try:
        if process.is_alive():
            if spawned._group_validated:
                os.killpg(spawned.worker_pid, signal.SIGTERM)
            else:
                process.terminate()
            process.join(timeout=TERMINATION_GRACE_SECONDS)
        if process.is_alive():
            if spawned._group_validated:
                os.killpg(spawned.worker_pid, signal.SIGKILL)
            else:
                process.kill()
            process.join(timeout=JOIN_TIMEOUT_SECONDS)
        if process.is_alive():
            raise ExperimentContractError("spawned worker survived bounded cleanup")
    finally:
        spawned.channel.close()
        spawned._closed = True


__all__ = [
    "ATTEMPT_ID",
    "JOIN_TIMEOUT_SECONDS",
    "RECEIPT_FILENAMES",
    "TERMINATION_GRACE_SECONDS",
    "WORKER_STARTED_TIMEOUT_SECONDS",
    "SpawnedTQCWorkerV2",
    "SupervisedTQCWorkerPreflightV2",
    "ValidatedTQCWorkerManifestAcknowledgementV2",
    "ValidatedTQCWorkerPreflightDeliveryV2",
    "admit_tqc_execution_manifest_v2",
    "receive_tqc_worker_preflight_v2",
    "revalidate_tqc_worker_manifest_acknowledgement_v2",
    "spawn_tqc_worker_v2",
    "terminate_tqc_worker_v2",
]
