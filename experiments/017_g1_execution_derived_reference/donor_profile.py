"""Retain the pinned donor's physical speed profile; no simulation or conversion."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from oracle_composition.adapters.gmt.course_task import TaskFrame
from oracle_composition.adapters.gmt.io import write_json_receipt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from convert_study012_reference_candidate import DONOR_ROOT, validate_donor  # noqa: E402


def donor_profile() -> dict:
    qpos, receipt = validate_donor(DONOR_ROOT)
    frame = TaskFrame.initialize(qpos[0, :2], qpos[0, 3:7])
    rows = []
    for index in range(92, 197):
        before = frame.project(qpos[index, :2], qpos[index, 3:7])
        after = frame.project(qpos[index + 1, :2], qpos[index + 1, 3:7])
        rows.append({
            "donor_action_index": index,
            "relative_action_index": index - 92,
            "start_phase_seconds": (index - 92) * 0.02,
            "end_phase_seconds": (index - 91) * 0.02,
            "poststep_progress_m": after.progress_m,
            "poststep_root_height_m": float(qpos[index + 1, 2]),
            "task_frame_forward_speed_m_s": (after.progress_m - before.progress_m) / 0.02,
            "task_frame_lateral_speed_m_s": (after.lateral_m - before.lateral_m) / 0.02,
        })
    return {
        "schema_version": 1,
        "artifact": "study017_retained_donor_physical_speed_profile/v1",
        "source": receipt,
        "selection": "qpos[92:198]; 105 physical intervals, final boundary197",
        "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "donor_verifier_sha256": hashlib.sha256(
            (ROOT / "scripts/convert_study012_reference_candidate.py").read_bytes()
        ).hexdigest(),
        "rows": rows,
        "claim_scope": "descriptive_pinned_donor_execution_not_transformed_reference_velocity",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = donor_profile()
    digest = write_json_receipt(args.output, result)
    print(json.dumps({"path": str(args.output), "sha256": digest, "intervals": len(result["rows"])}))
