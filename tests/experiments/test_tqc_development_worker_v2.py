from __future__ import annotations

import dataclasses
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from oracle_composition.experiments import tqc_development_manifest_v2 as manifest_module
from oracle_composition.experiments import tqc_development_worker_v2 as worker
from oracle_composition.experiments.artifact_io import publish_bytes_without_overwrite
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_calibration_contract import canonical_json


@dataclass(frozen=True)
class _Manifest:
    canonical_bytes: bytes
    sha256: str
    byte_count: int
    attempt_id: str
    claimed_work_directory_identity: str
    attempt_nonce: str

    def to_dict(self) -> dict[str, object]:
        return {"attempt_nonce": self.attempt_nonce}


@dataclass(frozen=True)
class _Preflight:
    manifest_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "observed": {
                "claimed_work_directory": {
                    "execution_manifest_absolute_path": str(self.manifest_path)
                }
            }
        }


class _Channel:
    def __init__(self, manifest: _Manifest, manifest_path: Path) -> None:
        self._manifest = manifest
        self._manifest_path = manifest_path
        self.send_sequence = 3
        self.receive_sequence = 0
        self.transcript_frame_count = 3
        self.transcript_sha256 = "b" * 64
        self.sent: tuple[str, str, dict[str, object], float] | None = None

    def receive(self, *, deadline: float) -> dict[str, object]:
        assert deadline > time.perf_counter()
        self.receive_sequence = 1
        self.transcript_frame_count = 4
        self.transcript_sha256 = "c" * 64
        return {
            "message_type": "admit_execution_manifest",
            "stage": "preflight",
            "sequence_index": 0,
            "payload": {
                "manifest_absolute_path": str(self._manifest_path),
                "manifest_byte_count": self._manifest.byte_count,
                "manifest_sha256": self._manifest.sha256,
            },
        }

    def send(
        self,
        message_type: str,
        stage: str,
        payload: dict[str, object],
        *,
        deadline: float,
    ) -> None:
        self.sent = (message_type, stage, payload, deadline)
        self.send_sequence = 4
        self.transcript_frame_count = 5
        self.transcript_sha256 = "d" * 64


def test_worker_acknowledges_only_after_exact_local_manifest_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = canonical_json({"test": "manifest"})
    artifact = publish_bytes_without_overwrite(tmp_path / "execution.json", encoded)
    manifest = _Manifest(
        canonical_bytes=encoded,
        sha256=artifact.sha256,
        byte_count=artifact.byte_count,
        attempt_id=worker.ATTEMPT_ID,
        claimed_work_directory_identity="e" * 64,
        attempt_nonce="a" * 64,
    )
    channel = _Channel(manifest, artifact.path)
    observed: dict[str, object] = {}

    def _revalidate(path: Path, **kwargs: object) -> _Manifest:
        observed["path"] = path
        observed.update(kwargs)
        return manifest

    monkeypatch.setattr(manifest_module, "ValidatedTQCExecutionManifestV2", _Manifest)
    monkeypatch.setattr(
        manifest_module,
        "revalidate_tqc_execution_manifest_for_worker_v2",
        _revalidate,
    )
    inputs = worker._WorkerInputsV2(
        design=object(),
        e0=object(),
        review=object(),
        preflight=_Preflight(artifact.path),
        work_directory=tmp_path,
        execution_source_paths={"test": tmp_path / "source.py"},
    )
    receipts = (b"bootstrap", b"runtime", b"instrumentation", b"receipt-set")

    acknowledgement = worker._receive_and_acknowledge_manifest(
        channel,
        inputs,
        receipts,
        deadline=time.perf_counter() + 10.0,
    )

    assert observed["path"] == artifact.path
    assert observed["channel_transcript_frame_count"] == 4
    assert observed["channel_transcript_sha256"] == "c" * 64
    assert channel.sent is not None
    assert channel.sent[:3] == (
        "execution_manifest_acknowledged",
        "preflight",
        {
            "manifest_byte_count": manifest.byte_count,
            "manifest_sha256": manifest.sha256,
            "worker_reverification_passed": True,
        },
    )
    assert acknowledgement.manifest is manifest
    assert acknowledgement.worker_pid == os.getpid()
    assert acknowledgement.acknowledgement_message_sequence_index == 3
    assert acknowledgement.channel_transcript_frame_count_after_acknowledgement == 5
    assert acknowledgement.authorizes_model_construction is True
    assert acknowledgement.behavioral_evidence is False
    assert (
        worker.revalidate_acknowledged_tqc_execution_manifest_v2(acknowledgement) is acknowledgement
    )
    with pytest.raises(ExperimentContractError, match="only be issued"):
        dataclasses.replace(acknowledgement)


def test_worker_acknowledgement_seal_rejects_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded = canonical_json({"test": "manifest"})
    artifact = publish_bytes_without_overwrite(tmp_path / "execution.json", encoded)
    manifest = _Manifest(
        canonical_bytes=encoded,
        sha256=artifact.sha256,
        byte_count=artifact.byte_count,
        attempt_id=worker.ATTEMPT_ID,
        claimed_work_directory_identity="e" * 64,
        attempt_nonce="a" * 64,
    )
    monkeypatch.setattr(manifest_module, "ValidatedTQCExecutionManifestV2", _Manifest)
    monkeypatch.setattr(
        manifest_module,
        "revalidate_tqc_execution_manifest_for_worker_v2",
        lambda *_args, **_kwargs: manifest,
    )
    acknowledgement = worker._receive_and_acknowledge_manifest(
        _Channel(manifest, artifact.path),
        worker._WorkerInputsV2(
            design=object(),
            e0=object(),
            review=object(),
            preflight=_Preflight(artifact.path),
            work_directory=tmp_path,
            execution_source_paths={},
        ),
        (b"bootstrap", b"runtime", b"instrumentation", b"receipt-set"),
        deadline=time.perf_counter() + 10.0,
    )

    object.__setattr__(acknowledgement, "behavioral_evidence", True)
    with pytest.raises(ExperimentContractError, match="seal differs"):
        worker.revalidate_acknowledged_tqc_execution_manifest_v2(acknowledgement)


def test_worker_acknowledgement_cannot_be_publicly_constructed() -> None:
    with pytest.raises(ExperimentContractError, match="only be issued"):
        worker.AcknowledgedTQCExecutionManifestV2.__new__(
            worker.AcknowledgedTQCExecutionManifestV2
        ).__post_init__(None)
