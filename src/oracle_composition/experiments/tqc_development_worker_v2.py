"""Fresh-process worker for the frozen TQC v2 development attempt."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any

from .artifact_io import publish_bytes_without_overwrite, read_verified_artifact_bytes
from .fixed_reference import ExperimentContractError
from .tqc_calibration_contract import canonical_json
from .tqc_development_channel_v2 import TQCWorkerChannelV2

WORKER_ID = "tqc_dev_1m_v2_worker/v1"
ATTEMPT_ID = "dev1m-v2-seed-95001-attempt-01"
MAX_PREFLIGHT_BYTES = 512 * 1024
MAX_EXECUTION_MANIFEST_BYTES = 1024 * 1024
PREFLIGHT_RECEIPT_FILENAMES = (
    "worker.bootstrap.json",
    "worker.runtime.json",
    "worker.instrumentation.json",
    "worker.preflight-set.json",
)

_ACK_ISSUER = object()
_ACK_SEAL_ISSUER = object()


def _require_sha256(value: object, *, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"{field_name} must be a lowercase SHA-256")
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExperimentContractError(f"worker input contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ExperimentContractError(f"worker input contains non-finite number {value!r}")


def _decode_canonical_object(
    encoded: bytes,
    *,
    maximum_bytes: int,
    artifact: str,
) -> dict[str, Any]:
    if type(encoded) is not bytes or not encoded or len(encoded) > maximum_bytes:
        raise ExperimentContractError(f"{artifact} bytes are absent or oversized")
    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except ExperimentContractError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ExperimentContractError(f"{artifact} is invalid JSON") from exc
    if type(value) is not dict or canonical_json(value) != encoded:
        raise ExperimentContractError(f"{artifact} is not one canonical object")
    return value


@dataclass(slots=True)
class _AcknowledgedManifestSealV2:
    """Bind an acknowledgement snapshot to one worker-local manifest."""

    _payload: bytes = field(repr=False)
    _creator_pid: int
    _manifest_identity: int
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _ACK_SEAL_ISSUER:
            raise ExperimentContractError("acknowledged manifest seals are private")
        if not self._payload or self._creator_pid != os.getpid() or self._manifest_identity <= 0:
            raise ExperimentContractError("acknowledged manifest seal is invalid")

    def validate(self, payload: Mapping[str, object], *, manifest: object) -> None:
        if (
            os.getpid() != self._creator_pid
            or id(manifest) != self._manifest_identity
            or canonical_json(dict(payload)) != self._payload
        ):
            raise ExperimentContractError("acknowledged manifest seal differs")


@dataclass(frozen=True, slots=True)
class AcknowledgedTQCExecutionManifestV2:
    """Worker-local authority created only after its acknowledgement was sent."""

    worker_id: str
    attempt_id: str
    attempt_nonce: str
    worker_pid: int
    execution_manifest_sha256: str
    execution_manifest_byte_count: int
    claimed_work_directory_identity: str
    acknowledgement_message_sequence_index: int
    channel_send_sequence_after_acknowledgement: int
    channel_receive_sequence_after_acknowledgement: int
    channel_transcript_frame_count_after_acknowledgement: int
    channel_transcript_sha256_after_acknowledgement: str
    worker_reverification_passed: bool
    authorizes_model_construction: bool
    behavioral_evidence: bool
    manifest: object = field(repr=False, compare=False)
    _seal: _AcknowledgedManifestSealV2 = field(repr=False, compare=False)
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _ACK_ISSUER:
            raise ExperimentContractError(
                "acknowledged manifests may only be issued by the worker handshake"
            )
        self._validate_sealed()

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"manifest", "_seal", "_issuer"}
        }

    def _validate_sealed(self) -> None:
        from .tqc_development_manifest_v2 import ValidatedTQCExecutionManifestV2

        if type(self._seal) is not _AcknowledgedManifestSealV2:
            raise ExperimentContractError("acknowledged manifest seal is unavailable")
        self._seal.validate(self.to_dict(), manifest=self.manifest)
        if type(self.manifest) is not ValidatedTQCExecutionManifestV2:
            raise ExperimentContractError("worker-local execution manifest type differs")
        manifest_value = self.manifest.to_dict()
        for value, field_name in (
            (self.attempt_nonce, "attempt nonce"),
            (self.execution_manifest_sha256, "execution manifest SHA-256"),
            (self.claimed_work_directory_identity, "claimed work directory identity"),
            (
                self.channel_transcript_sha256_after_acknowledgement,
                "channel transcript SHA-256",
            ),
        ):
            _require_sha256(value, field_name=field_name)
        if (
            self.worker_id != WORKER_ID
            or self.attempt_id != ATTEMPT_ID
            or self.attempt_id != self.manifest.attempt_id
            or self.attempt_nonce != manifest_value["attempt_nonce"]
            or self.worker_pid != os.getpid()
            or self.worker_pid <= 1
            or self.execution_manifest_sha256 != self.manifest.sha256
            or self.execution_manifest_byte_count != self.manifest.byte_count
            or self.claimed_work_directory_identity != self.manifest.claimed_work_directory_identity
            or self.acknowledgement_message_sequence_index != 3
            or self.channel_send_sequence_after_acknowledgement != 4
            or self.channel_receive_sequence_after_acknowledgement != 1
            or self.channel_transcript_frame_count_after_acknowledgement != 5
            or self.worker_reverification_passed is not True
            or self.authorizes_model_construction is not True
            or self.behavioral_evidence is not False
        ):
            raise ExperimentContractError("worker-local manifest acknowledgement is inconsistent")


def revalidate_acknowledged_tqc_execution_manifest_v2(
    acknowledgement: AcknowledgedTQCExecutionManifestV2,
) -> AcknowledgedTQCExecutionManifestV2:
    """Revalidate the worker-local acknowledgement without its private issuer."""

    if type(acknowledgement) is not AcknowledgedTQCExecutionManifestV2:
        raise ExperimentContractError("execution manifest acknowledgement type differs")
    acknowledgement._validate_sealed()
    return acknowledgement


@dataclass(frozen=True, slots=True)
class _WorkerInputsV2:
    design: object
    e0: object
    review: object
    preflight: object
    work_directory: Path
    execution_source_paths: Mapping[str, Path]


def _repository_root() -> Path:
    return Path(__file__).resolve(strict=True).parents[3]


def _load_worker_inputs(
    preflight_canonical_bytes: bytes,
    *,
    expected_preflight_sha256: str,
    expected_preflight_byte_count: int,
) -> _WorkerInputsV2:
    from .tqc_development_contract_v2 import (
        AUDIT_FILE_SHA256,
        DESIGN_FILE_SHA256,
        DESIGN_SEMANTIC_SHA256,
        TRAINING_PROJECTION_SHA256,
        load_tqc_development_design_v2,
        validate_v2_e0_reuse_artifacts,
    )
    from .tqc_development_manifest_v2 import (
        EXECUTION_SOURCE_ROLES,
        revalidate_tqc_preflight_contract_for_worker_v2,
    )
    from .tqc_development_review_v2 import validate_independent_review_receipt_v2

    _require_sha256(expected_preflight_sha256, field_name="expected preflight SHA-256")
    if (
        type(expected_preflight_byte_count) is not int
        or expected_preflight_byte_count != len(preflight_canonical_bytes)
        or hashlib.sha256(preflight_canonical_bytes).hexdigest() != expected_preflight_sha256
    ):
        raise ExperimentContractError("worker preflight transport bytes differ")
    raw = _decode_canonical_object(
        preflight_canonical_bytes,
        maximum_bytes=MAX_PREFLIGHT_BYTES,
        artifact="transported preflight contract",
    )
    if raw.get("attempt_id") != ATTEMPT_ID:
        raise ExperimentContractError("worker preflight attempt differs")

    root = _repository_root()
    experiment = root / "experiments/bootstrap_tqc_humanoid"
    config = experiment / "configs"
    review_root = experiment / "reviews"
    design = load_tqc_development_design_v2(
        config / "tqc_base_controller_dev_1m_v2.study.json",
        config / "tqc_e0_reuse_dev_1m_v2.audit.json",
    )
    e0 = validate_v2_e0_reuse_artifacts(
        design,
        calibration_design_path=config / "tqc_resource_calibration_v0.study.json",
        calibration_receipt_path=(
            root / "artifacts/bootstrap_tqc_humanoid/resource_calibration_seed_92001.json"
        ),
        superseded_v1_design_path=config / "tqc_base_controller_dev_1m_v0.study.json",
    )
    review = validate_independent_review_receipt_v2(
        design,
        review_receipt_path=(review_root / "tqc_dev_1m_v2_independent_design_review.review.json"),
        observed_check_receipt_paths=(
            review_root / "tqc_dev_1m_v2_reviewed_identity.check.json",
            review_root / "tqc_dev_1m_v2_focused_contract_tests.check.json",
            review_root / "tqc_dev_1m_v2_repository_tests.check.json",
        ),
        contract_source_path=(
            root / "src/oracle_composition/experiments/tqc_development_contract_v2.py"
        ),
        contract_test_path=root / "tests/experiments/test_tqc_development_contract_v2.py",
        protocol_doc_path=experiment / "DEV_1M_BASE_CONTROLLER_V2.md",
    )
    preflight_path = Path(str(raw.get("contract_absolute_path", "")))
    work_directory = Path(
        str(raw.get("requested", {}).get("claimed_work_directory_absolute_path", ""))
    )
    preflight = revalidate_tqc_preflight_contract_for_worker_v2(
        preflight_path,
        design,
        e0,
        review,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_design_file_sha256=DESIGN_FILE_SHA256,
        expected_design_semantic_sha256=DESIGN_SEMANTIC_SHA256,
        expected_e0_audit_file_sha256=AUDIT_FILE_SHA256,
        expected_training_projection_sha256=TRAINING_PROJECTION_SHA256,
        expected_claimed_work_directory_absolute_path=work_directory,
    )
    if preflight.canonical_bytes != preflight_canonical_bytes:
        raise ExperimentContractError("worker disk and transport preflight bytes differ")
    source_root = root / "src/oracle_composition"
    # The final paths are source-bound again when the manifest is received.
    execution_source_paths = {
        role: source_root
        / {
            "training_adapter_source_sha256": "experiments/tqc_development_training_v2.py",
            "attempt_supervisor_source_sha256": "experiments/tqc_development_supervisor_v2.py",
            "resource_monitor_source_sha256": "experiments/tqc_development_resource_v2.py",
            "protected_locomotion_evaluator_source_sha256": (
                "experiments/tqc_development_evaluator_v2.py"
            ),
            "metric_core_source_sha256": "experiments/tqc_development_metrics.py",
            "actor_export_source_sha256": "experiments/tqc_actor_equivalence_v2.py",
            "visual_capture_and_encoder_source_sha256": ("experiments/tqc_visual_evidence_v2.py"),
        }[role]
        for role in EXECUTION_SOURCE_ROLES
    }
    return _WorkerInputsV2(
        design=design,
        e0=e0,
        review=review,
        preflight=preflight,
        work_directory=work_directory,
        execution_source_paths=execution_source_paths,
    )


def _receipt_binding(inputs: _WorkerInputsV2, *, worker_start: float) -> dict[str, object]:
    return {
        "attempt_id": inputs.preflight.attempt_id,
        "attempt_nonce": inputs.preflight.attempt_nonce,
        "worker_pid": os.getpid(),
        "worker_process_start_monotonic_seconds": worker_start,
        "preflight_contract_sha256": inputs.preflight.sha256,
        "claimed_work_directory_identity": inputs.preflight.claimed_work_directory_identity,
        "project_python_source_tree_sha256": (inputs.preflight.project_python_source_tree_sha256),
    }


def _publish_preflight_receipts(
    inputs: _WorkerInputsV2,
    *,
    worker_start: float,
) -> tuple[tuple[dict[str, object], ...], tuple[bytes, bytes, bytes, bytes], object]:
    from .tqc_development_manifest_v2 import (
        worker_bootstrap_receipt_bytes_v2,
        worker_preflight_receipt_set_bytes_v2,
    )
    from .tqc_development_resource_v2 import TQCResourceMonitorV2
    from .tqc_development_runtime_v2 import inspect_tqc_development_runtime_v2
    from .tqc_instrumentation_equivalence_v2 import (
        run_tqc_instrumentation_equivalence_v2,
    )

    monitor = TQCResourceMonitorV2(inputs.work_directory)
    monitor.sample_lifecycle("preflight")
    binding = _receipt_binding(inputs, worker_start=worker_start)
    bootstrap = worker_bootstrap_receipt_bytes_v2(
        inputs.preflight,
        worker_process_start_monotonic_seconds=worker_start,
    )
    runtime = inspect_tqc_development_runtime_v2(
        inputs.design,
        receipt_binding=binding,
    )
    instrumentation = run_tqc_instrumentation_equivalence_v2(
        inputs.design,
        receipt_binding=binding,
    )
    receipt_set = worker_preflight_receipt_set_bytes_v2(
        inputs.preflight,
        worker_bootstrap_receipt_bytes=bootstrap,
        host_runtime_receipt_bytes=runtime.canonical_bytes,
        instrumentation_receipt_bytes=instrumentation.canonical_bytes,
    )
    encoded = (
        bootstrap,
        runtime.canonical_bytes,
        instrumentation.canonical_bytes,
        receipt_set,
    )
    artifacts = tuple(
        publish_bytes_without_overwrite(inputs.work_directory / filename, value)
        for filename, value in zip(PREFLIGHT_RECEIPT_FILENAMES, encoded, strict=True)
    )
    records = tuple(
        {
            "filename": artifact.path.name,
            "sha256": artifact.sha256,
            "byte_count": artifact.byte_count,
        }
        for artifact in artifacts
    )
    return records, encoded, monitor


def _receive_and_acknowledge_manifest(
    channel: TQCWorkerChannelV2,
    inputs: _WorkerInputsV2,
    receipt_bytes: tuple[bytes, bytes, bytes, bytes],
    *,
    deadline: float,
) -> AcknowledgedTQCExecutionManifestV2:
    from .tqc_development_manifest_v2 import (
        revalidate_tqc_execution_manifest_for_worker_v2,
    )

    message = channel.receive(deadline=deadline)
    payload = message["payload"]
    if (
        message["message_type"] != "admit_execution_manifest"
        or message["stage"] != "preflight"
        or message["sequence_index"] != 0
        or type(payload) is not dict
        or set(payload) != {"manifest_absolute_path", "manifest_byte_count", "manifest_sha256"}
    ):
        raise ExperimentContractError("worker execution-manifest delivery differs")
    manifest_path = Path(payload["manifest_absolute_path"])
    expected_path = Path(
        inputs.preflight.to_dict()["observed"]["claimed_work_directory"][
            "execution_manifest_absolute_path"
        ]
    )
    if manifest_path != expected_path:
        raise ExperimentContractError("worker execution-manifest path differs")
    manifest_bytes = read_verified_artifact_bytes(
        manifest_path,
        expected_sha256=payload["manifest_sha256"],
        expected_size=payload["manifest_byte_count"],
        max_bytes=MAX_EXECUTION_MANIFEST_BYTES,
    )
    _decode_canonical_object(
        manifest_bytes,
        maximum_bytes=MAX_EXECUTION_MANIFEST_BYTES,
        artifact="worker-read execution manifest",
    )
    manifest = revalidate_tqc_execution_manifest_for_worker_v2(
        manifest_path,
        expected_sha256=payload["manifest_sha256"],
        expected_byte_count=payload["manifest_byte_count"],
        design=inputs.design,
        e0=inputs.e0,
        review=inputs.review,
        preflight=inputs.preflight,
        worker_bootstrap_receipt_bytes=receipt_bytes[0],
        host_runtime_receipt_bytes=receipt_bytes[1],
        instrumentation_receipt_bytes=receipt_bytes[2],
        worker_receipt_set_bytes=receipt_bytes[3],
        execution_source_paths=inputs.execution_source_paths,
        preflight_receipts_message_sequence_index=2,
        channel_transcript_frame_count=channel.transcript_frame_count,
        channel_transcript_sha256=channel.transcript_sha256,
    )
    channel.send(
        "execution_manifest_acknowledged",
        "preflight",
        {
            "manifest_byte_count": manifest.byte_count,
            "manifest_sha256": manifest.sha256,
            "worker_reverification_passed": True,
        },
        deadline=deadline,
    )
    authority_payload: dict[str, object] = {
        "worker_id": WORKER_ID,
        "attempt_id": manifest.attempt_id,
        "attempt_nonce": manifest.to_dict()["attempt_nonce"],
        "worker_pid": os.getpid(),
        "execution_manifest_sha256": manifest.sha256,
        "execution_manifest_byte_count": manifest.byte_count,
        "claimed_work_directory_identity": manifest.claimed_work_directory_identity,
        "acknowledgement_message_sequence_index": 3,
        "channel_send_sequence_after_acknowledgement": channel.send_sequence,
        "channel_receive_sequence_after_acknowledgement": channel.receive_sequence,
        "channel_transcript_frame_count_after_acknowledgement": (channel.transcript_frame_count),
        "channel_transcript_sha256_after_acknowledgement": channel.transcript_sha256,
        "worker_reverification_passed": True,
        "authorizes_model_construction": True,
        "behavioral_evidence": False,
    }
    seal = _AcknowledgedManifestSealV2(
        _payload=canonical_json(authority_payload),
        _creator_pid=os.getpid(),
        _manifest_identity=id(manifest),
        _issuer=_ACK_SEAL_ISSUER,
    )
    return AcknowledgedTQCExecutionManifestV2(
        **authority_payload,
        manifest=manifest,
        _seal=seal,
        _issuer=_ACK_ISSUER,
    )


def _send_failure(
    channel: TQCWorkerChannelV2,
    *,
    stage: str,
    exc: BaseException,
) -> None:
    if channel.failed_reason is not None:
        return
    message = str(exc)
    if len(message) > 1024:
        message = message[:1024]
    try:
        channel.send(
            "failure",
            stage,
            {
                "exception_type": type(exc).__name__,
                "message": message,
                "observed_memory_error": isinstance(exc, MemoryError),
                "behavioral_evidence": False,
            },
            deadline=float(time.perf_counter() + 5.0),
        )
    except BaseException:
        return


def run_tqc_development_worker_v2(
    channel: TQCWorkerChannelV2,
    preflight_canonical_bytes: bytes,
    expected_preflight_sha256: str,
    expected_preflight_byte_count: int,
) -> None:
    """Reinspect preflight, acknowledge one manifest, then run the fixed attempt."""

    if type(channel) is not TQCWorkerChannelV2 or channel.role != "worker":
        raise ExperimentContractError("TQC v2 worker requires its exact worker channel")
    worker_start = float(time.perf_counter())
    stage = "preflight"
    monitor: object | None = None
    try:
        inputs = _load_worker_inputs(
            preflight_canonical_bytes,
            expected_preflight_sha256=expected_preflight_sha256,
            expected_preflight_byte_count=expected_preflight_byte_count,
        )
        deadline = float(time.perf_counter() + 300.0)
        channel.send(
            "worker_started",
            "preflight",
            {
                "attempt_nonce": inputs.preflight.attempt_nonce,
                "preflight_contract_sha256": inputs.preflight.sha256,
                "claimed_work_directory_identity": (
                    inputs.preflight.claimed_work_directory_identity
                ),
                "worker_pgid": os.getpgrp(),
                "worker_sid": os.getsid(0),
                "worker_process_start_monotonic_seconds": worker_start,
            },
            deadline=deadline,
        )
        channel.send(
            "stage_entered",
            "preflight",
            {"stage_index": 0},
            deadline=deadline,
        )
        records, receipts, monitor = _publish_preflight_receipts(
            inputs,
            worker_start=worker_start,
        )
        channel.send(
            "preflight_receipts",
            "preflight",
            {"receipts": list(records)},
            deadline=deadline,
        )
        acknowledgement = _receive_and_acknowledge_manifest(
            channel,
            inputs,
            receipts,
            deadline=deadline,
        )
        revalidate_acknowledged_tqc_execution_manifest_v2(acknowledgement)
        _run_acknowledged_attempt_v2(channel, inputs, acknowledgement, monitor)
    except BaseException as exc:
        if monitor is not None and hasattr(monitor, "_poison"):
            monitor._poison(exc)
        _send_failure(channel, stage=stage, exc=exc)
        raise


def _run_acknowledged_attempt_v2(
    channel: TQCWorkerChannelV2,
    inputs: _WorkerInputsV2,
    acknowledgement: AcknowledgedTQCExecutionManifestV2,
    resource_monitor: object,
) -> None:
    """Run stages after acknowledgement; completed by the integrated evidence modules."""

    del channel, inputs, acknowledgement, resource_monitor
    raise ExperimentContractError("TQC v2 post-acknowledgement pipeline is not integrated")


__all__ = [
    "ATTEMPT_ID",
    "WORKER_ID",
    "AcknowledgedTQCExecutionManifestV2",
    "revalidate_acknowledged_tqc_execution_manifest_v2",
    "run_tqc_development_worker_v2",
]
