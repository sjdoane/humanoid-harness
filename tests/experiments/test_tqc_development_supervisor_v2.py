from __future__ import annotations

import dataclasses
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from oracle_composition.experiments import tqc_development_supervisor_v2 as supervisor
from oracle_composition.experiments.artifact_io import publish_bytes_without_overwrite
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_calibration_contract import canonical_json
from oracle_composition.experiments.tqc_development_channel_v2 import TQCWorkerChannelV2


@dataclass(frozen=True)
class _Preflight:
    canonical_bytes: bytes
    sha256: str
    byte_count: int
    attempt_id: str
    attempt_nonce: str
    claimed_work_directory_identity: str
    work_directory: str

    def to_dict(self) -> dict[str, object]:
        return {
            "requested": {
                "claimed_work_directory_absolute_path": self.work_directory,
            }
        }


@dataclass(frozen=True)
class _Manifest:
    canonical_bytes: bytes
    sha256: str
    byte_count: int
    attempt_id: str
    preflight_contract_sha256: str
    claimed_work_directory_identity: str
    path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt_nonce": "a" * 64,
            "manifest_absolute_path": str(self.path),
        }


def _fake_preflight(work_directory: Path) -> _Preflight:
    encoded = canonical_json({"test": "preflight"})
    return _Preflight(
        canonical_bytes=encoded,
        sha256=hashlib.sha256(encoded).hexdigest(),
        byte_count=len(encoded),
        attempt_id=supervisor.ATTEMPT_ID,
        attempt_nonce="a" * 64,
        claimed_work_directory_identity="b" * 64,
        work_directory=str(work_directory),
    )


def _write_receipts(work_directory: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index, filename in enumerate(supervisor.RECEIPT_FILENAMES):
        artifact = publish_bytes_without_overwrite(
            work_directory / filename,
            canonical_json({"index": index, "receipt": filename}),
        )
        records.append(
            {
                "filename": filename,
                "sha256": artifact.sha256,
                "byte_count": artifact.byte_count,
            }
        )
    return records


def _fake_manifest(work_directory: Path, preflight: _Preflight) -> _Manifest:
    encoded = canonical_json({"test": "execution-manifest"})
    artifact = publish_bytes_without_overwrite(
        work_directory / "execution.manifest.json",
        encoded,
    )
    return _Manifest(
        canonical_bytes=encoded,
        sha256=artifact.sha256,
        byte_count=artifact.byte_count,
        attempt_id=preflight.attempt_id,
        preflight_contract_sha256=preflight.sha256,
        claimed_work_directory_identity=preflight.claimed_work_directory_identity,
        path=artifact.path,
    )


def _preflight_worker(
    channel: TQCWorkerChannelV2,
    work_directory: str,
    nonce: str,
    preflight_sha256: str,
    work_identity: str,
    wrong_order: bool,
    wrong_receipt_hash: bool,
    wrong_manifest_ack: bool,
) -> None:
    started = float(time.perf_counter())
    deadline = started + 10.0
    if wrong_order:
        channel.send(
            "stage_entered",
            "preflight",
            {"stage_index": 0},
            deadline=deadline,
        )
        return
    channel.send(
        "worker_started",
        "preflight",
        {
            "attempt_nonce": nonce,
            "preflight_contract_sha256": preflight_sha256,
            "claimed_work_directory_identity": work_identity,
            "worker_pgid": os.getpgrp(),
            "worker_sid": os.getsid(0),
            "worker_process_start_monotonic_seconds": started,
        },
        deadline=deadline,
    )
    channel.send(
        "stage_entered",
        "preflight",
        {"stage_index": 0},
        deadline=deadline,
    )
    records = _write_receipts(Path(work_directory))
    if wrong_receipt_hash:
        records[2]["sha256"] = "f" * 64
    channel.send(
        "preflight_receipts",
        "preflight",
        {"receipts": records},
        deadline=deadline,
    )
    try:
        admitted = channel.receive(deadline=deadline)
    except ExperimentContractError:
        return
    if admitted["message_type"] != "admit_execution_manifest":
        return
    manifest_sha256 = admitted["payload"]["manifest_sha256"]
    if wrong_manifest_ack:
        manifest_sha256 = "f" * 64
    channel.send(
        "execution_manifest_acknowledged",
        "preflight",
        {
            "manifest_byte_count": admitted["payload"]["manifest_byte_count"],
            "manifest_sha256": manifest_sha256,
            "worker_reverification_passed": True,
        },
        deadline=deadline,
    )
    try:
        channel.receive(deadline=deadline)
    except ExperimentContractError:
        return


def _spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    wrong_order: bool = False,
    wrong_receipt_hash: bool = False,
    wrong_manifest_ack: bool = False,
) -> tuple[supervisor.SpawnedTQCWorkerV2, _Preflight]:
    tmp_path.chmod(0o700)
    preflight = _fake_preflight(tmp_path)
    monkeypatch.setattr(supervisor, "ValidatedTQCPreflightContractV2", _Preflight)
    spawned = supervisor._spawn_tqc_worker_process_v2(
        _preflight_worker,
        target_args=(
            str(tmp_path),
            preflight.attempt_nonce,
            preflight.sha256,
            preflight.claimed_work_directory_identity,
            wrong_order,
            wrong_receipt_hash,
            wrong_manifest_ack,
        ),
    )
    return spawned, preflight


def test_real_spawned_session_delivers_exact_receipt_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned, preflight = _spawn(tmp_path, monkeypatch)
    try:
        bundle = supervisor.receive_tqc_worker_preflight_v2(
            spawned,
            preflight,
            deadline=time.perf_counter() + 10.0,
        )

        assert spawned.process.pid == spawned.worker_pid
        assert os.getpgid(spawned.worker_pid) == spawned.worker_pid
        assert os.getsid(spawned.worker_pid) == spawned.worker_pid
        assert bundle.delivery.worker_pid == spawned.worker_pid
        assert bundle.delivery.channel_parent_send_sequence == 0
        assert bundle.delivery.channel_parent_receive_sequence == 3
        assert bundle.delivery.channel_transcript_frame_count == 3
        assert bundle.delivery.authorizes_model_construction is False
        assert bundle.delivery.behavioral_evidence is False
        assert (
            tuple(
                hashlib.sha256(value).hexdigest()
                for value in (
                    bundle.worker_bootstrap_receipt_bytes,
                    bundle.host_runtime_receipt_bytes,
                    bundle.instrumentation_receipt_bytes,
                    bundle.worker_receipt_set_bytes,
                )
            )
            == bundle.delivery.ordered_receipt_sha256
        )
        with pytest.raises(ExperimentContractError, match="already received"):
            supervisor.receive_tqc_worker_preflight_v2(spawned, preflight)
        with pytest.raises(ExperimentContractError, match="only be issued"):
            dataclasses.replace(bundle.delivery)
    finally:
        supervisor.terminate_tqc_worker_v2(spawned)
    assert not spawned.process.is_alive()


def test_parent_requires_exact_worker_manifest_acknowledgement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned, preflight = _spawn(tmp_path, monkeypatch)
    monkeypatch.setattr(supervisor, "ValidatedTQCExecutionManifestV2", _Manifest)
    try:
        supervisor.receive_tqc_worker_preflight_v2(
            spawned,
            preflight,
            deadline=time.perf_counter() + 10.0,
        )
        manifest = _fake_manifest(tmp_path, preflight)
        acknowledgement = supervisor.admit_tqc_execution_manifest_v2(
            spawned,
            preflight,
            manifest,
            deadline=time.perf_counter() + 10.0,
        )

        assert acknowledgement.manifest is manifest
        assert acknowledgement.acknowledgement_message_sequence_index == 3
        assert acknowledgement.channel_parent_send_sequence == 1
        assert acknowledgement.channel_parent_receive_sequence == 4
        assert acknowledgement.channel_transcript_frame_count == 5
        assert acknowledgement.worker_reverification_passed is True
        assert acknowledgement.authorizes_model_construction is True
        assert acknowledgement.behavioral_evidence is False
        assert (
            supervisor.revalidate_tqc_worker_manifest_acknowledgement_v2(acknowledgement)
            is acknowledgement
        )
        with pytest.raises(ExperimentContractError, match="already acknowledged"):
            supervisor.admit_tqc_execution_manifest_v2(
                spawned,
                preflight,
                manifest,
                deadline=time.perf_counter() + 10.0,
            )
        with pytest.raises(ExperimentContractError, match="only be issued"):
            dataclasses.replace(acknowledgement)
    finally:
        supervisor.terminate_tqc_worker_v2(spawned)


def test_parent_rejects_wrong_worker_manifest_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned, preflight = _spawn(tmp_path, monkeypatch, wrong_manifest_ack=True)
    monkeypatch.setattr(supervisor, "ValidatedTQCExecutionManifestV2", _Manifest)
    try:
        supervisor.receive_tqc_worker_preflight_v2(
            spawned,
            preflight,
            deadline=time.perf_counter() + 10.0,
        )
        manifest = _fake_manifest(tmp_path, preflight)
        with pytest.raises(ExperimentContractError, match="acknowledgement differs"):
            supervisor.admit_tqc_execution_manifest_v2(
                spawned,
                preflight,
                manifest,
                deadline=time.perf_counter() + 10.0,
            )
    finally:
        supervisor.terminate_tqc_worker_v2(spawned)


def test_supervisor_rejects_wrong_initial_message_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned, preflight = _spawn(tmp_path, monkeypatch, wrong_order=True)
    try:
        with pytest.raises(ExperimentContractError, match=r"pipe closed early|handshake differs"):
            supervisor.receive_tqc_worker_preflight_v2(
                spawned,
                preflight,
                deadline=time.perf_counter() + 5.0,
            )
    finally:
        supervisor.terminate_tqc_worker_v2(spawned)


def test_supervisor_rejects_receipt_hash_not_matching_retained_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawned, preflight = _spawn(tmp_path, monkeypatch, wrong_receipt_hash=True)
    try:
        with pytest.raises(ExperimentContractError, match="SHA-256 differs"):
            supervisor.receive_tqc_worker_preflight_v2(
                spawned,
                preflight,
                deadline=time.perf_counter() + 10.0,
            )
    finally:
        supervisor.terminate_tqc_worker_v2(spawned)


def test_supervisor_authorities_cannot_be_publicly_constructed() -> None:
    for authority in (
        supervisor.ValidatedTQCWorkerPreflightDeliveryV2,
        supervisor.SpawnedTQCWorkerV2,
        supervisor.SupervisedTQCWorkerPreflightV2,
        supervisor.ValidatedTQCWorkerManifestAcknowledgementV2,
    ):
        with pytest.raises(ExperimentContractError, match="only be issued"):
            authority.__new__(authority).__post_init__(None)
