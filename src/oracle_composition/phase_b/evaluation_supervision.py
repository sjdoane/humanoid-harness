"""Bounded parent/worker supervision for protected policy evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import time
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from multiprocessing.connection import Connection
from pathlib import Path
from types import MappingProxyType

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.experiments.artifact_io import (
    PublishedArtifact,
    publish_bytes_without_overwrite,
    read_verified_artifact_bytes,
)
from oracle_composition.experiments.fixed_reference import ExperimentContractError

from .calibration import TaskSuccessCalibration, load_calibration_receipt
from .evaluation import (
    EVALUATION_BLOCKS,
    UtilityEvaluationDependencies,
    evaluate_policy_checkpoint,
)
from .protected_metrics import (
    protected_trace_sha256,
    recompute_protected_episode,
)
from .report_v2 import ProtectedEpisodeMetrics
from .supervision import (
    HeavyJobSlotSession,
    SupervisionDependencies,
    cleanup_worker_process,
    shared_coordination_root,
)

EVALUATION_SUCCESS_RECEIPT_ID = "humanoid_phase_b_evaluation_success/v1"
EVALUATION_FAILURE_RECEIPT_ID = "humanoid_phase_b_evaluation_failure/v1"
EVALUATION_MANIFEST_SCHEMA_ID = "humanoid_phase_b_evaluation_manifest/v1"
MAX_EVALUATION_MANIFEST_BYTES = 2 * 1024 * 1024
PLANNED_EVALUATION_CELLS = (
    "hold_expert",
    "hold_medium",
    "hold_simple",
    "fixed_round_trip",
)
PLANNED_EVALUATION_EPISODES = 160
POLL_SECONDS = 0.05
MESSAGE_MAX_BYTES = 64 * 1024


class EvaluationStatus(StrEnum):
    SUCCEEDED = "succeeded"
    TIMEOUT = "timeout"
    CRASH = "crash"
    NON_FINITE = "non_finite"
    PARTIAL = "partial"
    SOURCE_MISMATCH = "source_mismatch"


@dataclass(frozen=True, slots=True)
class EvaluationWorkerRequest:
    checkpoint_path: str
    checkpoint_sha256: str
    step_zero_actor_path: str
    step_zero_actor_sha256: str
    corpus_root: str
    calibration_receipt_path: str | None
    calibration_receipt_sha256: str | None
    segment_targets_m_s: tuple[float, float, float]
    dependencies: UtilityEvaluationDependencies
    output_directory: str
    evaluator_source_sha256: str
    repository_root: str
    evaluation_manifest_path: str
    evaluation_manifest_sha256: str
    evaluation_manifest_byte_count: int


@dataclass(frozen=True, slots=True)
class EvaluationSupervisionResult:
    status: EvaluationStatus
    terminal_receipt: PublishedArtifact
    terminal_value: Mapping[str, object]
    trained_metrics: PublishedArtifact | None
    step_zero_metrics: PublishedArtifact | None
    trained_traces: PublishedArtifact | None
    step_zero_traces: PublishedArtifact | None
    trained_episodes: tuple[ProtectedEpisodeMetrics, ...]
    step_zero_episodes: tuple[ProtectedEpisodeMetrics, ...]
    trained_trace_rows: tuple[Mapping[str, object], ...]
    step_zero_trace_rows: tuple[Mapping[str, object], ...]
    calibration: TaskSuccessCalibration | None
    step_zero_comparator: Mapping[str, object]


def _artifact_record(artifact: PublishedArtifact) -> dict[str, object]:
    return {
        "byte_count": artifact.byte_count,
        "filename": artifact.path.name,
        "sha256": artifact.sha256,
    }


def _send(connection: Connection, value: Mapping[str, object]) -> None:
    payload = canonical_json_bytes(dict(value))
    if len(payload) > MESSAGE_MAX_BYTES:
        raise ExperimentContractError("evaluation supervisor message exceeds its bound")
    connection.send_bytes(payload)


def _receive(connection: Connection) -> dict[str, object]:
    try:
        payload = connection.recv_bytes(MESSAGE_MAX_BYTES)
        value = json.loads(payload)
    except (EOFError, OSError, UnicodeError, ValueError) as exc:
        raise ExperimentContractError("evaluation worker message is malformed") from exc
    if type(value) is not dict or canonical_json_bytes(value) != payload:
        raise ExperimentContractError("evaluation worker message is not canonical")
    return value


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_manifest_source(
    value: object,
    *,
    request_sha256: str,
    expected_source: Mapping[str, object] | None,
) -> None:
    if type(value) is not dict or set(value) != {
        "evaluator_source_id",
        "modules",
        "sha256",
    }:
        raise ExperimentContractError("evaluation manifest source identity is malformed")
    modules = value["modules"]
    if type(modules) is not list or not modules:
        raise ExperimentContractError("evaluation manifest source modules are malformed")
    module_names: list[str] = []
    for record in modules:
        if type(record) is not dict or set(record) != {
            "module",
            "realpath",
            "repository_relative_path",
            "sha256",
        }:
            raise ExperimentContractError("evaluation manifest source module is malformed")
        if (
            type(record["module"]) is not str
            or type(record["realpath"]) is not str
            or type(record["repository_relative_path"]) is not str
            or not _is_sha256(record["sha256"])
        ):
            raise ExperimentContractError("evaluation manifest source module fields differ")
        module_names.append(record["module"])
    if module_names != sorted(set(module_names)):
        raise ExperimentContractError("evaluation manifest source modules are not canonical")
    core = {
        "evaluator_source_id": "humanoid_phase_b_executed_evaluator_sources/v1",
        "modules": modules,
    }
    source_sha256 = hashlib.sha256(canonical_json_bytes(core)).hexdigest()
    if (
        value["evaluator_source_id"] != core["evaluator_source_id"]
        or value["sha256"] != source_sha256
        or source_sha256 != request_sha256
        or (expected_source is not None and value != dict(expected_source))
    ):
        raise ExperimentContractError("evaluation manifest source identity differs")


def _validate_evaluation_manifest_bytes(
    request: EvaluationWorkerRequest,
    encoded: bytes,
    *,
    expected_source: Mapping[str, object] | None,
) -> None:
    try:
        value = json.loads(encoded)
    except (UnicodeError, ValueError) as exc:
        raise ExperimentContractError("evaluation manifest is not JSON") from exc
    expected_keys = {
        "checkpoint_lineage",
        "evaluation_manifest_schema_id",
        "evaluator_sources",
        "planned_cells",
        "planned_episode_count",
        "planned_evaluation_seeds",
        "schema_version",
        "segment_targets_m_s",
        "step_zero_actor_sha256",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise ExperimentContractError("evaluation manifest schema differs")
    if canonical_json_bytes(value) != encoded:
        raise ExperimentContractError("evaluation manifest is not canonical JSON")
    lineage = value["checkpoint_lineage"]
    if (
        type(lineage) is not dict
        or set(lineage)
        != {
            "bindings",
            "checkpoint_metadata",
            "checkpoint_sha256",
            "execution_manifest_sha256",
            "lineage_id",
        }
        or type(lineage["bindings"]) is not dict
        or type(lineage["checkpoint_metadata"]) is not dict
        or lineage["lineage_id"] != "humanoid_phase_b_evaluation_lineage/v1"
        or not _is_sha256(request.checkpoint_sha256)
        or not _is_sha256(lineage["execution_manifest_sha256"])
        or lineage["checkpoint_sha256"] != request.checkpoint_sha256
    ):
        raise ExperimentContractError("evaluation manifest checkpoint selector differs")
    targets = value["segment_targets_m_s"]
    if (
        type(request.segment_targets_m_s) is not tuple
        or len(request.segment_targets_m_s) != 3
        or any(
            type(target) is not float or not math.isfinite(target)
            for target in request.segment_targets_m_s
        )
        or type(targets) is not list
        or len(targets) != 3
        or any(type(target) is not float or not math.isfinite(target) for target in targets)
        or targets != list(request.segment_targets_m_s)
    ):
        raise ExperimentContractError("evaluation manifest target selectors differ")
    if (
        not _is_sha256(request.step_zero_actor_sha256)
        or value["step_zero_actor_sha256"] != request.step_zero_actor_sha256
    ):
        raise ExperimentContractError("evaluation manifest starting actor selector differs")
    _validate_manifest_source(
        value["evaluator_sources"],
        request_sha256=request.evaluator_source_sha256,
        expected_source=expected_source,
    )
    if (
        value["evaluation_manifest_schema_id"] != EVALUATION_MANIFEST_SCHEMA_ID
        or type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["planned_cells"] != list(PLANNED_EVALUATION_CELLS)
        or value["planned_evaluation_seeds"] != list(EVALUATION_BLOCKS)
        or value["planned_episode_count"] != PLANNED_EVALUATION_EPISODES
    ):
        raise ExperimentContractError("evaluation manifest fixed schedule differs")


def _read_request_evaluation_manifest(
    request: EvaluationWorkerRequest,
    *,
    expected_source: Mapping[str, object] | None,
) -> bytes:
    encoded = read_verified_artifact_bytes(
        Path(request.evaluation_manifest_path),
        expected_sha256=request.evaluation_manifest_sha256,
        expected_size=request.evaluation_manifest_byte_count,
        max_bytes=MAX_EVALUATION_MANIFEST_BYTES,
    )
    _validate_evaluation_manifest_bytes(request, encoded, expected_source=expected_source)
    return encoded


def _verify_parent_manifest_binding(
    request: EvaluationWorkerRequest,
    artifact: PublishedArtifact,
) -> bytes:
    if (
        type(request.evaluation_manifest_path) is not str
        or Path(os.path.abspath(request.evaluation_manifest_path))
        != Path(os.path.abspath(artifact.path))
        or request.evaluation_manifest_sha256 != artifact.sha256
        or request.evaluation_manifest_byte_count != artifact.byte_count
    ):
        raise ExperimentContractError("evaluation manifest request binding differs")
    return _read_request_evaluation_manifest(request, expected_source=None)


def _worker(request: EvaluationWorkerRequest, connection: Connection) -> None:
    os.setsid()
    completed = 0
    try:
        from .evaluation_lineage import evaluator_source_identity

        source = evaluator_source_identity(Path(request.repository_root))
        manifest_bytes = _read_request_evaluation_manifest(request, expected_source=source)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest_byte_count = len(manifest_bytes)
        _send(
            connection,
            {
                "message_type": "started",
                "payload": {
                    "evaluator_source_sha256": source["sha256"],
                    "evaluation_manifest_byte_count": manifest_byte_count,
                    "evaluation_manifest_sha256": manifest_sha256,
                    "pgid": os.getpgrp(),
                    "pid": os.getpid(),
                    "sid": os.getsid(0),
                },
            },
        )
        admission = _receive(connection)
        if (
            admission.get("message_type") != "admit"
            or admission.get("evaluator_source_sha256") != request.evaluator_source_sha256
            or admission.get("evaluation_manifest_sha256") != manifest_sha256
            or admission.get("evaluation_manifest_byte_count") != manifest_byte_count
            or source["sha256"] != request.evaluator_source_sha256
        ):
            raise ExperimentContractError("evaluation source or manifest admission differs")
        failure_mode = request.dependencies.failure_mode
        if failure_mode == "crash":
            os._exit(19)
        if failure_mode == "hang":
            while True:
                time.sleep(0.1)
        if failure_mode == "non_finite":
            raise FloatingPointError("controlled non-finite evaluation outcome")

        def progress(value: int) -> None:
            nonlocal completed
            completed = value
            _send(
                connection,
                {"message_type": "progress", "payload": {"completed_episodes": value}},
            )
            if (
                request.dependencies.fail_after_episodes is not None
                and value >= request.dependencies.fail_after_episodes
            ):
                raise RuntimeError("controlled partial evaluation")

        result = evaluate_policy_checkpoint(
            checkpoint_path=Path(request.checkpoint_path),
            checkpoint_sha256=request.checkpoint_sha256,
            step_zero_actor_path=Path(request.step_zero_actor_path),
            step_zero_actor_sha256=request.step_zero_actor_sha256,
            corpus_root=Path(request.corpus_root),
            calibration_receipt_path=(
                Path(request.calibration_receipt_path)
                if request.calibration_receipt_path is not None
                else None
            ),
            calibration_receipt_sha256=request.calibration_receipt_sha256,
            segment_targets_m_s=request.segment_targets_m_s,
            dependencies=request.dependencies,
            progress_callback=progress,
        )
        if completed != PLANNED_EVALUATION_EPISODES:
            raise ExperimentContractError("evaluation worker completed a partial schedule")
        output = Path(request.output_directory)
        artifacts = {
            "trained_metrics": publish_bytes_without_overwrite(
                output / "trained_policy_metrics_v1.json",
                canonical_json_bytes([row.to_dict() for row in result.trained_episodes]),
            ),
            "step_zero_metrics": publish_bytes_without_overwrite(
                output / "step_zero_metrics_v1.json",
                canonical_json_bytes([row.to_dict() for row in result.step_zero_episodes]),
            ),
            "trained_traces": publish_bytes_without_overwrite(
                output / "trained_policy_traces_v1.json",
                canonical_json_bytes([dict(row) for row in result.trained_traces]),
            ),
            "step_zero_traces": publish_bytes_without_overwrite(
                output / "step_zero_traces_v1.json",
                canonical_json_bytes([dict(row) for row in result.step_zero_traces]),
            ),
        }
        _send(
            connection,
            {
                "message_type": "completed",
                "payload": {
                    "artifacts": {
                        name: _artifact_record(artifact) for name, artifact in artifacts.items()
                    },
                    "completed_episodes": completed,
                },
            },
        )
    except BaseException as exc:
        status = (
            "non_finite"
            if isinstance(exc, FloatingPointError)
            else ("partial" if completed else "crash")
        )
        with suppress(BrokenPipeError, EOFError, OSError):
            _send(
                connection,
                {
                    "message_type": "failed",
                    "payload": {
                        "completed_episodes": completed,
                        "reason": f"{type(exc).__name__}: {exc}"[:1000],
                        "status": status,
                    },
                },
            )
    finally:
        connection.close()


class _EvaluationSpawnCleanupUnverified(RuntimeError):
    pass


def _spawn(request: EvaluationWorkerRequest) -> tuple[multiprocessing.Process, Connection]:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    process = context.Process(target=_worker, args=(request, child), daemon=False)
    try:
        process.start()
        child.close()
    except BaseException as exc:
        with suppress(BaseException):
            child.close()
        with suppress(BaseException):
            parent.close()
        if process.pid is not None:
            try:
                _cleanup(process, group_validated=False)
            except BaseException as cleanup_exc:
                raise _EvaluationSpawnCleanupUnverified(
                    f"evaluation worker spawn cleanup failed: {cleanup_exc}"
                ) from exc
        raise
    return process, parent


def _cleanup(process: multiprocessing.Process, *, group_validated: bool) -> None:
    cleanup_worker_process(process, group_validated=group_validated)


class _EvaluationWorkerCleanupGuard:
    def __init__(self, slot_session: HeavyJobSlotSession) -> None:
        self._slot_session = slot_session
        self._process: object | None = None
        self._connection: object | None = None
        self._group_validated = False
        self._cleanup_attempted = False
        self._connection_closed = False

    def retain_for_unverified_spawn(self) -> None:
        self._slot_session.worker_spawned()

    def attach(self, process: object, connection: object) -> None:
        self._process = process
        self._connection = connection
        self._slot_session.worker_spawned()

    def validate_group(self) -> None:
        self._group_validated = True

    def cleanup(self) -> None:
        if self._process is None or self._cleanup_attempted:
            return
        self._cleanup_attempted = True
        try:
            _cleanup(self._process, group_validated=self._group_validated)
            self._slot_session.worker_cleanup_verified()
        finally:
            if self._connection is not None and not self._connection_closed:
                self._connection.close()
                self._connection_closed = True


def _read_list_artifact(
    output: Path,
    record: Mapping[str, object],
    *,
    maximum: int,
) -> tuple[PublishedArtifact, list[object]]:
    if type(record) is not dict or set(record) != {"byte_count", "filename", "sha256"}:
        raise ExperimentContractError("evaluation artifact binding is malformed")
    path = output / str(record["filename"])
    if path.parent != output or path.is_symlink() or not path.is_file():
        raise ExperimentContractError("evaluation artifact path differs")
    if type(record["byte_count"]) is not int or not 1 <= record["byte_count"] <= maximum:
        raise ExperimentContractError("evaluation artifact byte count is outside its bound")
    payload = path.read_bytes()
    if (
        len(payload) != record["byte_count"]
        or hashlib.sha256(payload).hexdigest() != record["sha256"]
    ):
        raise ExperimentContractError("evaluation artifact content binding differs")
    try:
        value = json.loads(payload)
    except (UnicodeError, ValueError) as exc:
        raise ExperimentContractError("evaluation artifact is not JSON") from exc
    if type(value) is not list or canonical_json_bytes(value) != payload:
        raise ExperimentContractError("evaluation artifact is not a canonical list")
    return PublishedArtifact(path, record["sha256"], record["byte_count"]), value


def _failure_result(
    *,
    output: Path,
    common: Mapping[str, object],
    status: EvaluationStatus,
    reason: str,
) -> EvaluationSupervisionResult:
    value = {
        **dict(common),
        "failure_receipt_id": EVALUATION_FAILURE_RECEIPT_ID,
        "outcome": "failure",
        "reason": reason[:1000],
        "status": status.value,
    }
    receipt = publish_bytes_without_overwrite(
        output / "evaluation_failure_receipt_v1.json", canonical_json_bytes(value)
    )
    return EvaluationSupervisionResult(
        status,
        receipt,
        MappingProxyType(value),
        None,
        None,
        None,
        None,
        (),
        (),
        (),
        (),
        None,
        MappingProxyType({}),
    )


def _validate_persisted_traces(
    *,
    trace_rows: list[object],
    episodes: tuple[ProtectedEpisodeMetrics, ...],
) -> tuple[Mapping[str, object], ...]:
    if len(trace_rows) != len(episodes):
        raise ExperimentContractError("evaluation trace count differs from metric episodes")
    checked_rows = []
    expected_fields = {
        "cell",
        "checkpoint_sha256",
        "evaluation_seed",
        "policy_seed",
        "schema_version",
        "segment_targets_m_s",
        "steps",
        "steps_sha256",
        "switch_records",
        "trace_schema_id",
    }
    for raw_trace, episode in zip(trace_rows, episodes, strict=True):
        if type(raw_trace) is not dict or set(raw_trace) != expected_fields:
            raise ExperimentContractError("evaluation protected trace fields differ")
        steps = raw_trace["steps"]
        switches = raw_trace["switch_records"]
        targets = raw_trace["segment_targets_m_s"]
        if (
            raw_trace["trace_schema_id"] != "humanoid_phase_b_protected_episode_trace/v1"
            or raw_trace["schema_version"] != 1
            or raw_trace["cell"] != episode.cell
            or raw_trace["policy_seed"] != episode.policy_seed
            or raw_trace["evaluation_seed"] != episode.evaluation_seed
            or raw_trace["checkpoint_sha256"] != episode.checkpoint_sha256
            or type(steps) is not list
            or type(switches) is not list
            or type(targets) is not list
            or len(targets) != 3
            or any(type(value) is not float or not math.isfinite(value) for value in targets)
            or raw_trace["steps_sha256"] != protected_trace_sha256(steps)
        ):
            raise ExperimentContractError("evaluation protected trace identity differs")
        aggregates = recompute_protected_episode(
            steps=steps,
            cell=episode.cell,
            switches=switches,
            segment_targets_m_s=tuple(targets),
        )
        recomputed = ProtectedEpisodeMetrics(
            policy_seed=episode.policy_seed,
            evaluation_seed=episode.evaluation_seed,
            cell=episode.cell,
            checkpoint_sha256=episode.checkpoint_sha256,
            observed_steps=int(aggregates["observed_steps"]),
            root_delta_forward_speed_m_s=tuple(aggregates["root_delta_forward_speed_m_s"]),
            com_forward_speed_m_s=tuple(aggregates["com_forward_speed_m_s"]),
            six_tracking_errors=dict(aggregates["six_tracking_errors"]),
            fall=bool(aggregates["fall"]),
            forbidden_contacts=tuple(aggregates["forbidden_contacts"]),
            action_bounds_ok=bool(aggregates["action_bounds_ok"]),
            switch_records=tuple(dict(item) for item in switches),
            resynchronization_records=tuple(aggregates["resynchronization_records"]),
            segment_errors=dict(aggregates["segment_errors"]),
            transition_window_error=aggregates["transition_window_error"],
            settle_latency_steps=aggregates["settle_latency_steps"],
            time_to_first_failure_steps=aggregates["time_to_first_failure_steps"],
            task_success=episode.task_success,
            settled_state_normalized_error=aggregates["settled_state_normalized_error"],
        )
        if recomputed.to_dict() != episode.to_dict():
            raise ExperimentContractError(
                "evaluation protected metrics differ from their sufficient trace"
            )
        checked_rows.append(MappingProxyType(dict(raw_trace)))
    return tuple(checked_rows)


def _supervise_policy_evaluation(
    *,
    request: EvaluationWorkerRequest,
    evaluation_manifest: PublishedArtifact,
    slot_session: HeavyJobSlotSession,
    cleanup_guard: _EvaluationWorkerCleanupGuard,
) -> EvaluationSupervisionResult:
    """Run all 160 episodes under one deadline and publish one terminal receipt."""

    dependencies = request.dependencies
    if (
        type(dependencies.wall_seconds) not in {int, float}
        or not 0 < float(dependencies.wall_seconds) <= 24 * 60 * 60
    ):
        raise ExperimentContractError("evaluation wall deadline is invalid")
    manifest_bytes = _verify_parent_manifest_binding(request, evaluation_manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_byte_count = len(manifest_bytes)
    output = Path(request.output_directory).resolve(strict=True)
    slot_session.validate_before_spawn()
    try:
        process, connection = _spawn(request)
    except _EvaluationSpawnCleanupUnverified:
        cleanup_guard.retain_for_unverified_spawn()
        raise
    cleanup_guard.attach(process, connection)
    started = time.perf_counter()
    deadline = started + float(dependencies.wall_seconds)
    completed = 0
    status: EvaluationStatus | None = None
    reason = ""
    artifacts: Mapping[str, object] | None = None
    started_seen = False
    cleanup_error = ""
    try:
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                status = EvaluationStatus.TIMEOUT
                reason = "evaluation hard wall deadline exceeded"
                break
            if connection.poll(min(POLL_SECONDS, remaining)):
                message = _receive(connection)
                payload = message.get("payload")
                if type(payload) is not dict:
                    raise ExperimentContractError("evaluation worker payload is malformed")
                if message.get("message_type") == "started":
                    if (
                        started_seen
                        or payload.get("pid") != process.pid
                        or payload.get("pgid") != process.pid
                        or payload.get("sid") != process.pid
                        or payload.get("evaluator_source_sha256") != request.evaluator_source_sha256
                        or payload.get("evaluation_manifest_sha256") != manifest_sha256
                        or payload.get("evaluation_manifest_byte_count") != manifest_byte_count
                    ):
                        status = EvaluationStatus.SOURCE_MISMATCH
                        reason = "evaluation worker source, manifest, or process group differs"
                        break
                    started_seen = True
                    cleanup_guard.validate_group()
                    _send(
                        connection,
                        {
                            "evaluator_source_sha256": request.evaluator_source_sha256,
                            "evaluation_manifest_byte_count": manifest_byte_count,
                            "evaluation_manifest_sha256": manifest_sha256,
                            "message_type": "admit",
                        },
                    )
                elif message.get("message_type") == "progress":
                    value = payload.get("completed_episodes")
                    if not started_seen or type(value) is not int or value != completed + 1:
                        status = EvaluationStatus.PARTIAL
                        reason = "evaluation episode accounting drifted"
                        break
                    completed = value
                elif message.get("message_type") == "completed":
                    if (
                        not started_seen
                        or payload.get("completed_episodes") != PLANNED_EVALUATION_EPISODES
                        or completed != PLANNED_EVALUATION_EPISODES
                    ):
                        status = EvaluationStatus.PARTIAL
                        reason = "evaluation completed without all planned episodes"
                    elif type(payload.get("artifacts")) is not dict:
                        status = EvaluationStatus.CRASH
                        reason = "evaluation completion omitted artifacts"
                    else:
                        artifacts = payload["artifacts"]
                        status = EvaluationStatus.SUCCEEDED
                    break
                elif message.get("message_type") == "failed":
                    reported_completed = payload.get("completed_episodes")
                    if (
                        type(reported_completed) is not int
                        or not completed <= reported_completed <= PLANNED_EVALUATION_EPISODES
                    ):
                        status = EvaluationStatus.CRASH
                        reason = "evaluation worker failure accounting is malformed"
                        break
                    completed = reported_completed
                    try:
                        status = EvaluationStatus(str(payload.get("status")))
                    except ValueError:
                        status = EvaluationStatus.CRASH
                    reason = str(payload.get("reason", "evaluation worker failed"))
                    break
                else:
                    status = EvaluationStatus.CRASH
                    reason = "evaluation worker message type is unknown"
                    break
            elif not process.is_alive():
                status = EvaluationStatus.CRASH
                reason = f"evaluation worker exited with code {process.exitcode}"
                break
    except (EOFError, OSError, ExperimentContractError) as exc:
        status = EvaluationStatus.CRASH
        reason = f"evaluation supervision failed: {exc}"
    finally:
        try:
            cleanup_guard.cleanup()
        except Exception as exc:
            cleanup_error = f"evaluation worker cleanup failed: {exc}"

    status = status or EvaluationStatus.CRASH
    if cleanup_error:
        status = EvaluationStatus.CRASH
        reason = cleanup_error
        artifacts = None
    common = {
        "completed_episodes": completed,
        "evaluation_manifest_sha256": manifest_sha256,
        "planned_episodes": PLANNED_EVALUATION_EPISODES,
        "schema_version": 1,
        "uncompleted_episodes": PLANNED_EVALUATION_EPISODES - completed,
        "wall_seconds": time.perf_counter() - started,
    }
    if status is not EvaluationStatus.SUCCEEDED or artifacts is None:
        return _failure_result(
            output=output,
            common=common,
            status=status,
            reason=reason or "evaluation worker produced no completion",
        )

    try:
        required = {"trained_metrics", "step_zero_metrics", "trained_traces", "step_zero_traces"}
        if set(artifacts) != required:
            raise ExperimentContractError("evaluation artifact set differs")
        trained_artifact, trained_rows = _read_list_artifact(
            output, artifacts["trained_metrics"], maximum=256 * 1024**2
        )
        baseline_artifact, baseline_rows = _read_list_artifact(
            output, artifacts["step_zero_metrics"], maximum=256 * 1024**2
        )
        trained_trace_artifact, trained_traces = _read_list_artifact(
            output, artifacts["trained_traces"], maximum=512 * 1024**2
        )
        baseline_trace_artifact, baseline_traces = _read_list_artifact(
            output, artifacts["step_zero_traces"], maximum=512 * 1024**2
        )
        trained_episodes = tuple(ProtectedEpisodeMetrics.from_dict(row) for row in trained_rows)
        baseline_episodes = tuple(ProtectedEpisodeMetrics.from_dict(row) for row in baseline_rows)
        if len(trained_episodes) != 80 or len(baseline_episodes) != 80:
            raise ExperimentContractError("evaluation metric artifacts omit planned episodes")
        calibration = (
            load_calibration_receipt(
                Path(request.calibration_receipt_path),
                expected_sha256=request.calibration_receipt_sha256,
            )
            if request.calibration_receipt_path is not None
            and request.calibration_receipt_sha256 is not None
            else None
        )
        if dependencies.episode_runner is None:
            checked_trained_traces = _validate_persisted_traces(
                trace_rows=trained_traces,
                episodes=trained_episodes,
            )
            checked_baseline_traces = _validate_persisted_traces(
                trace_rows=baseline_traces,
                episodes=baseline_episodes,
            )
        else:
            if trained_traces or baseline_traces:
                raise ExperimentContractError(
                    "injected episode runner returned unsupported trace evidence"
                )
            checked_trained_traces = ()
            checked_baseline_traces = ()
    except (ExperimentContractError, KeyError, TypeError, ValueError) as exc:
        return _failure_result(
            output=output,
            common=common,
            status=EvaluationStatus.CRASH,
            reason=f"evaluation artifact validation failed: {exc}",
        )
    trained_bytes = canonical_json_bytes(trained_rows)
    baseline_bytes = canonical_json_bytes(baseline_rows)
    comparator = {
        "bitwise_equal": True,
        "calibration_receipt_sha256": calibration.sha256 if calibration else None,
        "evaluation_schedule_identical": True,
        "step_zero_actor_sha256": request.step_zero_actor_sha256,
        "step_zero_metrics_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "trained_metrics_sha256": hashlib.sha256(trained_bytes).hexdigest(),
    }
    value = {
        **common,
        "artifacts": {name: dict(record) for name, record in artifacts.items()},
        "outcome": "success",
        "status": status.value,
        "success_receipt_id": EVALUATION_SUCCESS_RECEIPT_ID,
    }
    receipt = publish_bytes_without_overwrite(
        output / "evaluation_success_receipt_v1.json", canonical_json_bytes(value)
    )
    return EvaluationSupervisionResult(
        status,
        receipt,
        MappingProxyType(value),
        trained_artifact,
        baseline_artifact,
        trained_trace_artifact,
        baseline_trace_artifact,
        trained_episodes,
        baseline_episodes,
        checked_trained_traces,
        checked_baseline_traces,
        calibration,
        MappingProxyType(comparator),
    )


def supervise_policy_evaluation(
    *,
    request: EvaluationWorkerRequest,
    evaluation_manifest: PublishedArtifact,
    validated_reservation: Mapping[str, object] | None = None,
    coordination_root: Path | None = None,
    expected_wall_seconds: int | None = None,
    slot_dependencies: SupervisionDependencies | None = None,
    test_only: bool = False,
) -> EvaluationSupervisionResult:
    """Supervise evaluation while retaining its exact heavy-slot token through cleanup."""

    selected_dependencies = slot_dependencies or SupervisionDependencies()
    if type(selected_dependencies) is not SupervisionDependencies:
        raise ExperimentContractError("evaluation slot dependency authority differs")
    session = HeavyJobSlotSession(selected_dependencies, required=not test_only)
    if not test_only:
        if type(validated_reservation) is not dict:
            raise ExperimentContractError("evaluation requires a validated mailbox reservation")
        if type(expected_wall_seconds) is not int or expected_wall_seconds <= 0:
            raise ExperimentContractError("evaluation expected wall authority differs")
        root = shared_coordination_root(
            Path(request.repository_root),
            supplied=coordination_root,
        )
        session.configure(
            root,
            validated_reservation,
            expected_wall_seconds=expected_wall_seconds,
        )
        session.validate_before_spawn()
    cleanup_guard = _EvaluationWorkerCleanupGuard(session)
    try:
        try:
            return _supervise_policy_evaluation(
                request=request,
                evaluation_manifest=evaluation_manifest,
                slot_session=session,
                cleanup_guard=cleanup_guard,
            )
        finally:
            cleanup_guard.cleanup()
    finally:
        session.release()


__all__ = [
    "EVALUATION_FAILURE_RECEIPT_ID",
    "EVALUATION_SUCCESS_RECEIPT_ID",
    "MAX_EVALUATION_MANIFEST_BYTES",
    "PLANNED_EVALUATION_CELLS",
    "PLANNED_EVALUATION_EPISODES",
    "EvaluationStatus",
    "EvaluationSupervisionResult",
    "EvaluationWorkerRequest",
    "supervise_policy_evaluation",
]
