from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

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


def test_unified_cli_build_accepts_optional_typed_source_records(tmp_path: Path, capsys) -> None:
    extractions = tmp_path / "extractions"
    extractions.mkdir()
    _write_record(extractions, "2600.00001", "Phase recovery")
    repository_root = Path(__file__).resolve().parents[2]
    supplemental = repository_root / "research/evidence/reference_audits/records"
    database = tmp_path / "graph.db"

    code = main(
        [
            "--json",
            "research",
            "build",
            "--extractions",
            str(extractions),
            "--supplemental-records",
            str(supplemental),
            "--database",
            str(database),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert code == 0
    assert output["papers"] == 1
    assert output["source_records"] == 2
    assert output["public_source_audits"] == 1
    assert output["local_numeric_measurements"] == 1


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

    code = main(["research", "query", "phase", "--database", str(database)])

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


def test_unified_cli_dispatches_reference_use_probe(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    from oracle_composition.experiments import reference_causal_probe

    observed: dict[str, Path] = {}

    def run_probe(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(to_dict=lambda: {"mechanistic_gate_passed": True})

    monkeypatch.setattr(reference_causal_probe, "run_reference_causal_probe", run_probe)
    output_path = tmp_path / "probe.json"
    code = main(
        [
            "--json",
            "tracker",
            "probe-reference-use",
            "--output",
            str(output_path),
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out)["mechanistic_gate_passed"] is True
    assert observed["output_path"] == output_path
    assert observed["hdf5_path"].name == "main_data.hdf5"


def test_unified_cli_dispatches_tqc_calibration(tmp_path: Path, capsys, monkeypatch) -> None:
    from oracle_composition.experiments import tqc_calibration

    observed: dict[str, Path] = {}

    def run_calibration(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            receipt={"calibration_gate_passed": True},
            to_dict=lambda: {
                "calibration_gate_passed": True,
                "claim_boundary": "resource_integrity_only_no_behavior_or_controller_claim/v1",
            },
        )

    monkeypatch.setattr(tqc_calibration, "run_tqc_calibration", run_calibration)
    design_path = tmp_path / "design.json"
    output_path = tmp_path / "receipt.json"
    code = main(
        [
            "--json",
            "tracker",
            "calibrate-tqc",
            "--design",
            str(design_path),
            "--output",
            str(output_path),
            "--confirm-disposable-resource-probe",
        ]
    )

    assert code == 0
    assert json.loads(capsys.readouterr().out)["calibration_gate_passed"] is True
    assert observed == {
        "design_path": design_path,
        "output_path": output_path,
        "confirm_disposable_resource_probe": True,
    }


def test_unified_cli_failed_tqc_calibration_preserves_json_mode(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    from oracle_composition.experiments import tqc_calibration

    monkeypatch.setattr(
        tqc_calibration,
        "run_tqc_calibration",
        lambda **_kwargs: SimpleNamespace(
            receipt={"calibration_gate_passed": False},
            to_dict=lambda: {
                "calibration_gate_passed": False,
                "claim_boundary": "resource_integrity_only_no_behavior_or_controller_claim/v1",
            },
        ),
    )

    code = main(
        [
            "--json",
            "tracker",
            "calibrate-tqc",
            "--design",
            str(tmp_path / "design.json"),
            "--output",
            str(tmp_path / "receipt.json"),
            "--confirm-disposable-resource-probe",
        ]
    )

    assert code == 2
    raw_output = capsys.readouterr().out
    assert raw_output.count("\n") == 1
    assert json.loads(raw_output)["calibration_gate_passed"] is False


def test_unified_cli_dispatches_tqc_transfer_fixture(
    tmp_path: Path,
    capsys,
    monkeypatch,
) -> None:
    from oracle_composition.experiments import tqc_initialization_identity

    observed: dict[str, Path] = {}

    def run_fixture(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(
            to_dict=lambda: {
                "fixture_identity_checks_passed": True,
                "actual_e1_gate_passed": False,
            }
        )

    monkeypatch.setattr(tqc_initialization_identity, "run_transfer_fixture", run_fixture)
    design_path = tmp_path / "design.json"
    e0_receipt_path = tmp_path / "e0.json"
    output_path = tmp_path / "fixture.json"
    code = main(
        [
            "--json",
            "tracker",
            "verify-tqc-transfer-fixture",
            "--design",
            str(design_path),
            "--e0-receipt",
            str(e0_receipt_path),
            "--output",
            str(output_path),
        ]
    )

    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["fixture_identity_checks_passed"] is True
    assert result["actual_e1_gate_passed"] is False
    assert observed == {
        "design_path": design_path,
        "e0_receipt_path": e0_receipt_path,
        "output_path": output_path,
    }


def test_unified_cli_requires_tqc_resource_acknowledgement(tmp_path: Path, capsys) -> None:
    design_path = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "bootstrap_tqc_humanoid"
        / "configs"
        / "tqc_resource_calibration_v0.study.json"
    )
    output = tmp_path / "receipt.json"

    code = main(
        [
            "--json",
            "tracker",
            "calibrate-tqc",
            "--design",
            str(design_path),
            "--output",
            str(output),
        ]
    )

    assert code == 2
    assert "--confirm-disposable-resource-probe" in capsys.readouterr().err
    assert not output.exists()
