from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.experiments.artifact_io import (
    PublishedArtifact,
    publish_bytes_without_overwrite,
)
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.phase_b import evaluation_supervision as supervision
from oracle_composition.phase_b.evaluation import UtilityEvaluationDependencies
from oracle_composition.phase_b.evaluation_lineage import evaluator_source_identity

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_SHA256 = "a" * 64
EXECUTION_MANIFEST_SHA256 = "b" * 64
STARTING_ACTOR_SHA256 = "c" * 64
TARGETS = (3.0, 1.25, 3.0)
PROCESS_ID = 731


class InjectedConnection:
    def __init__(self, inbound: dict[str, object] | None) -> None:
        self.inbound = canonical_json_bytes(inbound) if inbound is not None else None
        self.sent: list[bytes] = []
        self.closed = False

    def send_bytes(self, payload: bytes) -> None:
        self.sent.append(payload)

    def recv_bytes(self, _maximum: int) -> bytes:
        if self.inbound is None:
            raise AssertionError("worker attempted admission receive after manifest rejection")
        return self.inbound

    def close(self) -> None:
        self.closed = True

    def poll(self, _timeout: float) -> bool:
        return self.inbound is not None


class InjectedProcess:
    pid = PROCESS_ID

    def __init__(self, *, alive: bool) -> None:
        self.alive = alive
        self.exitcode = 19

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.alive = False

    def kill(self) -> None:
        self.alive = False

    def join(self, timeout: float = 0.0) -> None:
        del timeout


@pytest.fixture(scope="module")
def source_identity() -> dict[str, object]:
    return evaluator_source_identity(ROOT)


def _manifest_value(source: dict[str, object]) -> dict[str, object]:
    return {
        "checkpoint_lineage": {
            "bindings": {},
            "checkpoint_metadata": {"ppo_seed": 11},
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "execution_manifest_sha256": EXECUTION_MANIFEST_SHA256,
            "lineage_id": "humanoid_phase_b_evaluation_lineage/v1",
        },
        "evaluation_manifest_schema_id": "humanoid_phase_b_evaluation_manifest/v1",
        "evaluator_sources": copy.deepcopy(source),
        "planned_cells": [
            "hold_expert",
            "hold_medium",
            "hold_simple",
            "fixed_round_trip",
        ],
        "planned_episode_count": 160,
        "planned_evaluation_seeds": list(range(120101, 120121)),
        "schema_version": 1,
        "segment_targets_m_s": list(TARGETS),
        "step_zero_actor_sha256": STARTING_ACTOR_SHA256,
    }


def _publish_manifest(
    directory: Path,
    source: dict[str, object],
    *,
    value: dict[str, object] | None = None,
    encoded: bytes | None = None,
) -> PublishedArtifact:
    manifest_value = value if value is not None else _manifest_value(source)
    payload = encoded if encoded is not None else canonical_json_bytes(manifest_value)
    return publish_bytes_without_overwrite(directory / "evaluation_manifest_v1.json", payload)


def _request(
    directory: Path,
    artifact: PublishedArtifact,
    source: dict[str, object],
) -> supervision.EvaluationWorkerRequest:
    return supervision.EvaluationWorkerRequest(
        checkpoint_path=str(directory / "checkpoint.npz"),
        checkpoint_sha256=CHECKPOINT_SHA256,
        step_zero_actor_path=str(directory / "step_zero.npz"),
        step_zero_actor_sha256=STARTING_ACTOR_SHA256,
        corpus_root=str(directory / "corpus"),
        calibration_receipt_path=None,
        calibration_receipt_sha256=None,
        segment_targets_m_s=TARGETS,
        dependencies=UtilityEvaluationDependencies(),
        output_directory=str(directory),
        evaluator_source_sha256=str(source["sha256"]),
        repository_root=str(ROOT),
        evaluation_manifest_path=str(artifact.path),
        evaluation_manifest_sha256=artifact.sha256,
        evaluation_manifest_byte_count=artifact.byte_count,
    )


def _slot_dependencies(
    coordination_root: Path,
    reservation: dict[str, object],
) -> tuple[supervision.SupervisionDependencies, list[int], list[tuple[str, str]]]:
    validations: list[int] = []
    releases: list[tuple[str, str]] = []

    def validate_slot(
        root: Path,
        observed_reservation: object,
        *,
        expected_wall_seconds: int,
    ) -> dict[str, object]:
        assert root == coordination_root
        assert observed_reservation == reservation
        validations.append(expected_wall_seconds)
        return {"owner": "evaluation-owner", "token_id": "7" * 32}

    def release_slot(root: Path, *, owner: str, token_id: str) -> dict[str, object]:
        assert root == coordination_root
        releases.append((owner, token_id))
        return {"owner": owner, "token_id": token_id}

    dependencies = supervision.SupervisionDependencies(
        validate_heavy_job_slot=validate_slot,
        release_heavy_job_slot=release_slot,
    )
    return dependencies, validations, releases


def _admission(
    artifact: PublishedArtifact,
    source: dict[str, object],
) -> dict[str, object]:
    return {
        "evaluator_source_sha256": source["sha256"],
        "evaluation_manifest_byte_count": artifact.byte_count,
        "evaluation_manifest_sha256": artifact.sha256,
        "message_type": "admit",
    }


def _install_worker_sentinels(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    entered: list[dict[str, object]] = []

    def checkpoint_sentinel(**kwargs: object) -> None:
        entered.append(dict(kwargs))
        raise RuntimeError("checkpoint constructor sentinel")

    monkeypatch.setattr(supervision.os, "setsid", lambda: None)
    monkeypatch.setattr(supervision.os, "getpgrp", lambda: PROCESS_ID)
    monkeypatch.setattr(supervision.os, "getpid", lambda: PROCESS_ID)
    monkeypatch.setattr(supervision.os, "getsid", lambda _pid: PROCESS_ID)
    monkeypatch.setattr(supervision, "evaluate_policy_checkpoint", checkpoint_sentinel)
    return entered


def _messages(connection: InjectedConnection) -> list[dict[str, object]]:
    return [json.loads(payload) for payload in connection.sent]


def _assert_worker_rejected(
    request: supervision.EvaluationWorkerRequest,
    connection: InjectedConnection,
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, object]]:
    entered = _install_worker_sentinels(monkeypatch)
    supervision._worker(request, connection)  # type: ignore[arg-type]
    messages = _messages(connection)
    assert entered == []
    assert connection.closed is True
    assert messages[-1]["message_type"] == "failed"
    assert messages[-1]["payload"]["completed_episodes"] == 0
    assert all(message["message_type"] != "completed" for message in messages)
    return messages


def test_matching_manifest_and_admission_cross_checkpoint_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_identity: dict[str, object],
) -> None:
    artifact = _publish_manifest(tmp_path, source_identity)
    request = _request(tmp_path, artifact, source_identity)
    connection = InjectedConnection(_admission(artifact, source_identity))
    entered = _install_worker_sentinels(monkeypatch)

    supervision._worker(request, connection)  # type: ignore[arg-type]

    messages = _messages(connection)
    assert len(entered) == 1
    assert messages[0]["message_type"] == "started"
    assert messages[0]["payload"]["evaluation_manifest_sha256"] == artifact.sha256
    assert messages[0]["payload"]["evaluation_manifest_byte_count"] == artifact.byte_count


@pytest.mark.parametrize("case", ["missing_digest", "wrong_digest", "wrong_byte_count"])
def test_worker_rejects_manifest_admission_mismatch_before_checkpoint_entry(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_identity: dict[str, object],
) -> None:
    artifact = _publish_manifest(tmp_path, source_identity)
    request = _request(tmp_path, artifact, source_identity)
    admission = _admission(artifact, source_identity)
    if case == "missing_digest":
        del admission["evaluation_manifest_sha256"]
    elif case == "wrong_digest":
        admission["evaluation_manifest_sha256"] = "f" * 64
    else:
        admission["evaluation_manifest_byte_count"] = artifact.byte_count + 1

    messages = _assert_worker_rejected(
        request,
        InjectedConnection(admission),
        monkeypatch,
    )

    assert messages[0]["message_type"] == "started"


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "wrong_digest",
        "altered_bytes",
        "recorded_count",
        "noncanonical",
        "oversized",
    ],
)
def test_worker_rejects_invalid_manifest_bytes_before_checkpoint_entry(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_identity: dict[str, object],
) -> None:
    artifact = _publish_manifest(tmp_path, source_identity)
    request = _request(tmp_path, artifact, source_identity)
    if case == "missing":
        request = replace(request, evaluation_manifest_path=str(tmp_path / "missing.json"))
    elif case == "wrong_digest":
        request = replace(request, evaluation_manifest_sha256="f" * 64)
    elif case == "altered_bytes":
        altered = canonical_json_bytes({**_manifest_value(source_identity), "schema_version": 2})
        artifact.path.write_bytes(altered)
    elif case == "recorded_count":
        request = replace(
            request,
            evaluation_manifest_byte_count=request.evaluation_manifest_byte_count + 1,
        )
    elif case == "noncanonical":
        pretty = json.dumps(_manifest_value(source_identity), indent=2).encode("utf-8")
        noncanonical = publish_bytes_without_overwrite(tmp_path / "pretty.json", pretty)
        request = replace(
            request,
            evaluation_manifest_path=str(noncanonical.path),
            evaluation_manifest_sha256=noncanonical.sha256,
            evaluation_manifest_byte_count=noncanonical.byte_count,
        )
    else:
        oversized = publish_bytes_without_overwrite(
            tmp_path / "oversized.json",
            b"x" * (supervision.MAX_EVALUATION_MANIFEST_BYTES + 1),
        )
        request = replace(
            request,
            evaluation_manifest_path=str(oversized.path),
            evaluation_manifest_sha256=oversized.sha256,
            evaluation_manifest_byte_count=oversized.byte_count,
        )

    _assert_worker_rejected(request, InjectedConnection(None), monkeypatch)


@pytest.mark.parametrize(
    "case",
    [
        "checkpoint",
        "starting_actor",
        "targets",
        "source",
        "cells",
        "seeds",
        "episode_count",
    ],
)
def test_worker_rejects_manifest_selector_mismatch_before_checkpoint_entry(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_identity: dict[str, object],
) -> None:
    value = _manifest_value(source_identity)
    if case == "checkpoint":
        value["checkpoint_lineage"]["checkpoint_sha256"] = "d" * 64
    elif case == "starting_actor":
        value["step_zero_actor_sha256"] = "d" * 64
    elif case == "targets":
        value["segment_targets_m_s"] = [3.0, 1.5, 3.0]
    elif case == "source":
        manifest_source = value["evaluator_sources"]
        manifest_source["modules"][0]["sha256"] = "0" * 64
        core = {
            "evaluator_source_id": manifest_source["evaluator_source_id"],
            "modules": manifest_source["modules"],
        }
        manifest_source["sha256"] = hashlib.sha256(canonical_json_bytes(core)).hexdigest()
    elif case == "cells":
        value["planned_cells"] = list(reversed(value["planned_cells"]))
    elif case == "seeds":
        value["planned_evaluation_seeds"][0] = 120100
    else:
        value["planned_episode_count"] = 159
    artifact = _publish_manifest(tmp_path, source_identity, value=value)
    request = _request(tmp_path, artifact, source_identity)

    _assert_worker_rejected(request, InjectedConnection(None), monkeypatch)


@pytest.mark.parametrize("case", ["request_digest", "request_count", "altered_bytes"])
def test_parent_manifest_mismatch_refuses_before_spawn(
    case: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_identity: dict[str, object],
) -> None:
    artifact = _publish_manifest(tmp_path, source_identity)
    request = _request(tmp_path, artifact, source_identity)
    if case == "request_digest":
        request = replace(request, evaluation_manifest_sha256="f" * 64)
    elif case == "request_count":
        request = replace(
            request,
            evaluation_manifest_byte_count=request.evaluation_manifest_byte_count + 1,
        )
    else:
        artifact.path.write_bytes(
            canonical_json_bytes({**_manifest_value(source_identity), "schema_version": 2})
        )
    spawn_calls = 0

    def spawn_sentinel(_request: supervision.EvaluationWorkerRequest) -> None:
        nonlocal spawn_calls
        spawn_calls += 1
        raise AssertionError("spawn must not be reached")

    monkeypatch.setattr(supervision, "_spawn", spawn_sentinel)

    with pytest.raises(ExperimentContractError):
        supervision.supervise_policy_evaluation(
            request=request,
            evaluation_manifest=artifact,
            test_only=True,
        )

    assert spawn_calls == 0


def test_evaluation_missing_heavy_slot_refuses_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_identity: dict[str, object],
) -> None:
    artifact = _publish_manifest(tmp_path, source_identity)
    request = _request(tmp_path, artifact, source_identity)
    coordination_root = tmp_path / "coordination"
    coordination_root.mkdir()
    spawn_calls = 0

    def spawn_sentinel(_request: supervision.EvaluationWorkerRequest) -> None:
        nonlocal spawn_calls
        spawn_calls += 1
        raise AssertionError("evaluation spawn must not be reached")

    monkeypatch.setattr(supervision, "_spawn", spawn_sentinel)
    with pytest.raises(ExperimentContractError, match="heavy-job slot refused dispatch"):
        supervision.supervise_policy_evaluation(
            request=request,
            evaluation_manifest=artifact,
            validated_reservation={},
            coordination_root=coordination_root,
            expected_wall_seconds=1_200,
        )
    assert spawn_calls == 0


def test_evaluation_releases_exact_captured_slot_after_terminal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_identity: dict[str, object],
) -> None:
    artifact = _publish_manifest(tmp_path, source_identity)
    request = _request(tmp_path, artifact, source_identity)
    coordination_root = tmp_path / "coordination"
    coordination_root.mkdir()
    reservation = {"owner": "evaluation-owner"}
    validations = 0
    releases: list[tuple[str, str]] = []

    def validate_slot(
        root: Path,
        observed_reservation: object,
        *,
        expected_wall_seconds: int,
    ) -> dict[str, object]:
        nonlocal validations
        validations += 1
        assert root == coordination_root
        assert observed_reservation == reservation
        assert expected_wall_seconds == 1_800
        return {"owner": "evaluation-owner", "token_id": "7" * 32}

    def release_slot(root: Path, *, owner: str, token_id: str) -> dict[str, object]:
        assert root == coordination_root
        releases.append((owner, token_id))
        return {"owner": owner, "token_id": token_id}

    def terminal_failure(**kwargs: object) -> object:
        kwargs["slot_session"].validate_before_spawn()
        raise RuntimeError("controlled evaluation failure after pre-spawn validation")

    monkeypatch.setattr(supervision, "_supervise_policy_evaluation", terminal_failure)
    dependencies = supervision.SupervisionDependencies(
        validate_heavy_job_slot=validate_slot,
        release_heavy_job_slot=release_slot,
    )
    with pytest.raises(RuntimeError, match="controlled evaluation failure"):
        supervision.supervise_policy_evaluation(
            request=request,
            evaluation_manifest=artifact,
            validated_reservation=reservation,
            coordination_root=coordination_root,
            expected_wall_seconds=1_800,
            slot_dependencies=dependencies,
        )
    assert validations == 2
    assert releases == [("evaluation-owner", "7" * 32)]


@pytest.mark.parametrize(
    ("failure", "message"),
    (
        (KeyboardInterrupt, "controlled post-spawn interrupt"),
        (RuntimeError, "controlled initializer exception"),
    ),
)
def test_evaluation_post_spawn_failure_cleans_before_releasing_slot(
    failure: type[BaseException],
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_identity: dict[str, object],
) -> None:
    artifact = _publish_manifest(tmp_path, source_identity)
    request = _request(tmp_path, artifact, source_identity)
    coordination_root = tmp_path / "coordination"
    coordination_root.mkdir()
    reservation = {"owner": "evaluation-owner"}
    dependencies, validations, releases = _slot_dependencies(
        coordination_root,
        reservation,
    )
    process = InjectedProcess(alive=True)
    connection = InjectedConnection(None)
    cleanup_calls: list[bool] = []

    monkeypatch.setattr(supervision, "_spawn", lambda _request: (process, connection))

    def cleanup(worker: InjectedProcess, *, group_validated: bool) -> None:
        assert worker is process
        cleanup_calls.append(group_validated)
        worker.alive = False

    monkeypatch.setattr(supervision, "_cleanup", cleanup)

    def fail_after_spawn() -> float:
        raise failure(message)

    monkeypatch.setattr(supervision.time, "perf_counter", fail_after_spawn)
    with pytest.raises(failure, match=message):
        supervision.supervise_policy_evaluation(
            request=request,
            evaluation_manifest=artifact,
            validated_reservation=reservation,
            coordination_root=coordination_root,
            expected_wall_seconds=1_800,
            slot_dependencies=dependencies,
        )

    assert validations == [1_800, 1_800]
    assert cleanup_calls == [False]
    assert process.alive is False
    assert connection.closed is True
    assert releases == [("evaluation-owner", "7" * 32)]


@pytest.mark.parametrize("cleanup_fails", (False, True))
def test_evaluation_terminal_output_preserved_and_slot_tracks_cleanup(
    cleanup_fails: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_identity: dict[str, object],
) -> None:
    artifact = _publish_manifest(tmp_path, source_identity)
    request = _request(tmp_path, artifact, source_identity)
    coordination_root = tmp_path / "coordination"
    coordination_root.mkdir()
    reservation = {"owner": "evaluation-owner"}
    dependencies, validations, releases = _slot_dependencies(
        coordination_root,
        reservation,
    )
    process = InjectedProcess(alive=False)
    connection = InjectedConnection(None)
    cleanup_calls = 0

    monkeypatch.setattr(supervision, "_spawn", lambda _request: (process, connection))

    def cleanup(worker: InjectedProcess, *, group_validated: bool) -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1
        assert worker is process
        assert group_validated is False
        if cleanup_fails:
            raise ExperimentContractError("controlled evaluation worker survived cleanup")

    monkeypatch.setattr(supervision, "_cleanup", cleanup)
    result = supervision.supervise_policy_evaluation(
        request=request,
        evaluation_manifest=artifact,
        validated_reservation=reservation,
        coordination_root=coordination_root,
        expected_wall_seconds=1_800,
        slot_dependencies=dependencies,
    )

    assert result.status is supervision.EvaluationStatus.CRASH
    assert validations == [1_800, 1_800]
    assert cleanup_calls == 1
    assert connection.closed is True
    if cleanup_fails:
        assert "survived cleanup" in result.terminal_value["reason"]
        assert releases == []
    else:
        assert releases == [("evaluation-owner", "7" * 32)]


@pytest.mark.parametrize("failure_stage", ("start", "child_close"))
def test_evaluation_partial_spawn_closes_pipes_and_worker(
    failure_stage: str,
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
        pid = PROCESS_ID

        def start(self) -> None:
            if failure_stage == "start":
                raise RuntimeError("controlled partial start failure")

    parent = Endpoint()
    child = Endpoint(fail=failure_stage == "child_close")
    process = Process()
    context = SimpleNamespace(
        Pipe=lambda **_kwargs: (parent, child),
        Process=lambda **_kwargs: process,
    )
    cleanup_calls: list[bool] = []
    monkeypatch.setattr(supervision.multiprocessing, "get_context", lambda _kind: context)
    monkeypatch.setattr(
        supervision,
        "_cleanup",
        lambda worker, *, group_validated: cleanup_calls.append(group_validated),
    )

    with pytest.raises((RuntimeError, OSError), match="controlled"):
        supervision._spawn(SimpleNamespace())
    assert parent.closed is True
    assert child.closed is True
    assert cleanup_calls == [False]


def test_evaluation_unverified_partial_spawn_cleanup_retains_slot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_identity: dict[str, object],
) -> None:
    artifact = _publish_manifest(tmp_path, source_identity)
    request = _request(tmp_path, artifact, source_identity)
    coordination_root = tmp_path / "coordination"
    coordination_root.mkdir()
    reservation = {"owner": "evaluation-owner"}
    dependencies, validations, releases = _slot_dependencies(
        coordination_root,
        reservation,
    )

    def unverified_spawn(_request: object) -> object:
        raise supervision._EvaluationSpawnCleanupUnverified(
            "controlled partial-spawn cleanup failure"
        )

    monkeypatch.setattr(supervision, "_spawn", unverified_spawn)
    with pytest.raises(
        supervision._EvaluationSpawnCleanupUnverified,
        match="partial-spawn cleanup failure",
    ):
        supervision.supervise_policy_evaluation(
            request=request,
            evaluation_manifest=artifact,
            validated_reservation=reservation,
            coordination_root=coordination_root,
            expected_wall_seconds=1_800,
            slot_dependencies=dependencies,
        )

    assert validations == [1_800, 1_800]
    assert releases == []
