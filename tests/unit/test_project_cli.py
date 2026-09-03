from __future__ import annotations

import json
from pathlib import Path

from oracle_composition.cli import main
from oracle_composition.traces import (
    REQUIRED_DIAGNOSTIC_SIGNALS,
    ArtifactBinding,
    EvidenceClass,
    MissingReason,
    MissingSignal,
    NumericSignalSpec,
    TraceRecorder,
    TraceRole,
)


def _write_record(directory: Path, paper_id: str, title: str) -> None:
    record = {
        "paper": {
            "id": paper_id,
            "title": title,
            "year": 2026,
            "primary_url": f"https://example.test/{paper_id}",
            "source_version": "test-v1",
        },
        "decision_relevance": {
            "tier": "core",
            "knobs": ["reference_composition"],
            "why_admitted": "Tests state-aware recovery decisions.",
        },
        "mechanisms": [],
        "parameters": [],
        "failure_modes": [],
        "evaluation": [],
        "relations": [],
        "evidence": [],
    }
    (directory / f"{paper_id}.json").write_text(json.dumps(record), encoding="utf-8")


def test_unified_cli_builds_and_queries_research_index(
    tmp_path: Path,
    capsys,
) -> None:
    extractions = tmp_path / "extractions"
    extractions.mkdir()
    _write_record(extractions, "2600.00001", "Phase recovery")
    database = tmp_path / "graph.db"

    build_code = main(
        [
            "--json",
            "research",
            "build",
            "--extractions",
            str(extractions),
            "--database",
            str(database),
        ]
    )
    build_output = json.loads(capsys.readouterr().out)
    query_code = main(
        [
            "--json",
            "research",
            "query",
            "phase recovery",
            "--database",
            str(database),
        ]
    )
    query_output = json.loads(capsys.readouterr().out)

    assert build_code == 0
    assert build_output["papers"] == 1
    assert query_code == 0
    assert query_output[0]["title"] == "Phase recovery"


def test_unified_cli_reports_missing_index_without_traceback(
    tmp_path: Path,
    capsys,
) -> None:
    code = main(
        [
            "research",
            "query",
            "phase",
            "--database",
            str(tmp_path / "missing.db"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "research index not found" in captured.err
    assert captured.out == ""


def test_unified_cli_reports_corrupt_index_without_traceback(
    tmp_path: Path,
    capsys,
) -> None:
    database = tmp_path / "corrupt.db"
    database.write_bytes(b"not a sqlite database")

    code = main(
        ["research", "query", "phase", "--database", str(database)]
    )

    captured = capsys.readouterr()
    assert code == 2
    assert "corrupt or incomplete" in captured.err
    assert captured.out == ""


def test_unified_cli_inspects_trace_identity(tmp_path: Path, capsys) -> None:
    signal = NumericSignalSpec(
        "robot.qpos",
        TraceRole.ROBOT,
        (1,),
        "mixed",
        "adapter",
        "fixture",
    )
    missing = tuple(
        MissingSignal(name, MissingReason.NOT_IMPLEMENTED, "Fixture omission.")
        for name in sorted(REQUIRED_DIAGNOSTIC_SIGNALS - {signal.name})
    )
    recorder = TraceRecorder(
        trace_id="trace/cli/v1",
        evidence_class=EvidenceClass.INTERFACE_CHECK,
        control_period_seconds=0.02,
        numeric_signals=(signal,),
        missing_signals=missing,
        artifact_bindings=(ArtifactBinding("runtime", "runtime/test", "a" * 64),),
    )
    recorder.add_sample({"robot.qpos": [0.0]})
    trace_path = recorder.finish().write(tmp_path / "trace.json")

    code = main(["--json", "trace", "inspect", str(trace_path)])
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["trace_id"] == "trace/cli/v1"
    assert output["sample_count"] == 1
    assert len(output["sha256"]) == 64
