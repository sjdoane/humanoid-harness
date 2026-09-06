"""Bounded parent/worker supervision for protected policy evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import signal
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
)
from oracle_composition.experiments.fixed_reference import ExperimentContractError

from .calibration import TaskSuccessCalibration, load_calibration_receipt
from .evaluation import (
    UtilityEvaluationDependencies,
    evaluate_policy_checkpoint,
)
from .protected_metrics import (
    protected_trace_sha256,
    recompute_protected_episode,
)
from .report_v2 import ProtectedEpisodeMetrics

EVALUATION_SUCCESS_RECEIPT_ID = "humanoid_phase_b_evaluation_success/v1"
EVALUATION_FAILURE_RECEIPT_ID = "humanoid_phase_b_evaluation_failure/v1"
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


def _worker(request: EvaluationWorkerRequest, connection: Connection) -> None:
    os.setsid()
    completed = 0
    try:
        from .evaluation_lineage import evaluator_source_identity

        source = evaluator_source_identity(Path(request.repository_root))
        _send(
            connection,
            {
                "message_type": "started",
                "payload": {
                    "evaluator_source_sha256": source["sha256"],
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
            or source["sha256"] != request.evaluator_source_sha256
        ):
            raise ExperimentContractError("evaluation source admission differs")
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


def _spawn(request: EvaluationWorkerRequest) -> tuple[multiprocessing.Process, Connection]:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    process = context.Process(target=_worker, args=(request, child), daemon=False)
    process.start()
    child.close()
    return process, parent


def _cleanup(process: multiprocessing.Process, *, group_validated: bool) -> None:
    pid = process.pid
    if pid is None:
        return
    if process.is_alive():
        if group_validated:
            with suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGTERM)
        else:
            process.terminate()
        process.join(timeout=2.0)
    if process.is_alive():
        if group_validated:
            with suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGKILL)
        else:
            process.kill()
        process.join(timeout=2.0)
    if process.is_alive():
        raise ExperimentContractError("evaluation worker survived cleanup")
    process.join(timeout=0.0)


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


def supervise_policy_evaluation(
    *,
    request: EvaluationWorkerRequest,
    evaluation_manifest: PublishedArtifact,
) -> EvaluationSupervisionResult:
    """Run all 160 episodes under one deadline and publish one terminal receipt."""

    dependencies = request.dependencies
    if (
        type(dependencies.wall_seconds) not in {int, float}
        or not 0 < float(dependencies.wall_seconds) <= 24 * 60 * 60
    ):
        raise ExperimentContractError("evaluation wall deadline is invalid")
    output = Path(request.output_directory).resolve(strict=True)
    process, connection = _spawn(request)
    started = time.perf_counter()
    deadline = started + float(dependencies.wall_seconds)
    completed = 0
    group_validated = False
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
                    ):
                        status = EvaluationStatus.SOURCE_MISMATCH
                        reason = "evaluation worker source or process group differs"
                        break
                    started_seen = True
                    group_validated = True
                    _send(
                        connection,
                        {
                            "evaluator_source_sha256": request.evaluator_source_sha256,
                            "evaluation_manifest_sha256": evaluation_manifest.sha256,
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
            _cleanup(process, group_validated=group_validated)
        except Exception as exc:
            cleanup_error = f"evaluation worker cleanup failed: {exc}"
        connection.close()

    status = status or EvaluationStatus.CRASH
    if cleanup_error:
        status = EvaluationStatus.CRASH
        reason = cleanup_error
        artifacts = None
    common = {
        "completed_episodes": completed,
        "evaluation_manifest_sha256": evaluation_manifest.sha256,
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


__all__ = [
    "EVALUATION_FAILURE_RECEIPT_ID",
    "EVALUATION_SUCCESS_RECEIPT_ID",
    "PLANNED_EVALUATION_EPISODES",
    "EvaluationStatus",
    "EvaluationSupervisionResult",
    "EvaluationWorkerRequest",
    "supervise_policy_evaluation",
]
