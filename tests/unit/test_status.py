from __future__ import annotations

import json
from pathlib import Path

from oracle_composition.status import SUBSTRATE_RECEIPT, program_status

ROOT = Path(__file__).parents[2]


def test_program_status_links_verified_repository_evidence() -> None:
    status = program_status(project_root=ROOT)

    assert status["evidence_class"] == "interface_check"
    assert status["measured_evidence"] == ["Humanoid-v5 reset/step interface check"]
    receipt = status["evidence_receipts"][0]
    assert receipt["receipt"] == str((ROOT / SUBSTRATE_RECEIPT).resolve())
    assert len(receipt["sha256"]) == 64


def test_program_status_does_not_invent_missing_or_malformed_evidence(
    tmp_path: Path,
) -> None:
    missing = program_status(project_root=tmp_path)
    assert missing["evidence_class"] == "none"
    assert missing["measured_evidence"] == []

    path = tmp_path / SUBSTRATE_RECEIPT
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"config": "not an object"}), encoding="utf-8")
    malformed = program_status(project_root=tmp_path)
    assert malformed["evidence_class"] == "none"
    assert malformed["evidence_receipts"] == []
