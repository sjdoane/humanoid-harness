from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

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
        )

    assert spawn_calls == 0
