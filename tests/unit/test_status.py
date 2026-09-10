from __future__ import annotations

import json
from pathlib import Path

from oracle_composition.cli import main
from oracle_composition.status import GMT_REPORT, SUBSTRATE_RECEIPT, program_status

ROOT = Path(__file__).parents[2]


def test_program_status_links_verified_repository_evidence() -> None:
    status = program_status(project_root=ROOT)

    assert status["evidence_class"] == "interface_check"
    assert status["measured_evidence"] == ["Humanoid-v5 reset/step interface check"]
    receipt = status["evidence_receipts"][0]
    assert receipt["receipt"] == str((ROOT / SUBSTRATE_RECEIPT).resolve())
    assert len(receipt["sha256"]) == 64
    assert "native Humanoid-v5 interface receipts only" in status["evidence_scope"]


def test_program_status_separates_vibe_target_from_gmt_implementation() -> None:
    status = program_status(project_root=ROOT)

    assert "post-training" in status["research_target"]
    assert "SONIC-based VIBE" in status["research_target"]
    assert any("GMT/G1 development proxy" in item for item in status["implemented_capability"])
    assert not any("VIBE" in item for item in status["implemented_capability"])
    assert "VIBE interface and dynamic-task assets are not admitted" in status["current_bottleneck"]
    assert "fixed/manual/harness" in status["next_gate"]
    assert "independent success criteria" in status["next_gate"]


def test_status_cli_exposes_scoped_status_without_local_evidence(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.chdir(tmp_path)

    assert main(["--json", "status"]) == 0
    status = json.loads(capsys.readouterr().out)

    assert "SONIC-based VIBE" in status["research_target"]
    assert status["reported_status"]["state"] == "missing"
    assert status["evidence_class"] == "none"
    assert status["evidence_receipts"] == []


def test_gmt_report_pointer_is_documentation_not_measured_evidence(tmp_path: Path) -> None:
    report = tmp_path / GMT_REPORT
    report.parent.mkdir(parents=True)
    report.write_text("# All tasks passed\nUnverified report text.\n", encoding="utf-8")

    status = program_status(project_root=tmp_path)

    assert status["reported_status"]["state"] == "available"
    assert status["reported_status"]["authority"] == "documentation_only"
    assert status["reported_status"]["report"] == GMT_REPORT.as_posix()
    assert "not VIBE/SONIC" in status["reported_status"]["adapter"]
    assert status["evidence_class"] == "none"
    assert status["measured_evidence"] == []
    assert status["evidence_receipts"] == []
    assert "All tasks passed" not in json.dumps(status)


def test_missing_gmt_report_stays_explicit_without_hiding_native_evidence(tmp_path: Path) -> None:
    receipt = tmp_path / SUBSTRATE_RECEIPT
    receipt.parent.mkdir(parents=True)
    receipt.write_bytes((ROOT / SUBSTRATE_RECEIPT).read_bytes())

    status = program_status(project_root=tmp_path)

    assert status["reported_status"]["state"] == "missing"
    assert status["evidence_class"] == "interface_check"
    assert status["measured_evidence"] == ["Humanoid-v5 reset/step interface check"]
    assert len(status["evidence_receipts"]) == 1


def test_program_status_does_not_invent_missing_or_malformed_evidence(
    tmp_path: Path,
) -> None:
    missing = program_status(project_root=tmp_path)
    assert missing["evidence_class"] == "none"
    assert missing["measured_evidence"] == []
    assert missing["evidence_receipts"] == []
    assert missing["reported_status"]["state"] == "missing"

    path = tmp_path / SUBSTRATE_RECEIPT
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"config": "not an object"}), encoding="utf-8")
    malformed = program_status(project_root=tmp_path)
    assert malformed["evidence_class"] == "none"
    assert malformed["evidence_receipts"] == []
