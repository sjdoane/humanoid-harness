"""Data-only effort accounting from selected, hash-pinned GMT run receipts."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from pathlib import Path

from oracle_composition.harness.contract import read_json_object

_SHA256 = re.compile(r"[0-9a-f]{64}")
MAX_RECEIPTS = 256


def _count(value: object, field: str) -> int:
    if type(value) is not int or not 0 <= value <= 2**53 - 1:
        raise ValueError(f"{field} must be a nonnegative exact integer")
    return value


def _duration(value: object) -> float:
    if type(value) not in {int, float}:
        raise ValueError("runtime.wall_seconds must be numeric")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("runtime.wall_seconds must be finite") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError("runtime.wall_seconds must be finite and nonnegative")
    return result


def summarize_effort(receipts: Sequence[tuple[Path, str]]) -> dict[str, object]:
    """Account for selected attempts without claiming complete search cost."""
    if not 1 <= len(receipts) <= MAX_RECEIPTS:
        raise ValueError(f"supply between 1 and {MAX_RECEIPTS} run receipts")
    rows = []
    seen_hashes: set[str] = set()
    seen_files: set[tuple[int, int]] = set()
    for path, expected in receipts:
        if type(expected) is not str or _SHA256.fullmatch(expected) is None:
            raise ValueError("manifest SHA-256 must be 64 lowercase hexadecimal characters")
        record, encoded = read_json_object(Path(path))
        digest = hashlib.sha256(encoded).hexdigest()
        if digest != expected:
            raise ValueError(f"manifest SHA-256 mismatch: {path}")
        file_stat = Path(path).stat()
        file_identity = (file_stat.st_dev, file_stat.st_ino)
        if digest in seen_hashes or file_identity in seen_files:
            raise ValueError("duplicate run receipt; an attempt cannot be counted twice")
        seen_hashes.add(digest)
        seen_files.add(file_identity)
        if (
            type(record.get("schema_version")) is not int
            or record["schema_version"] != 1
            or record.get("artifact") != "gmt_g1_course_development_run"
            or record.get("status") != "completed"
        ):
            raise ValueError("expected a completed GMT course development run receipt")
        config_hash = record.get("input_config_sha256")
        if type(config_hash) is not str or _SHA256.fullmatch(config_hash) is None:
            raise ValueError("run receipt lacks an exact input configuration identity")
        if "training" not in record:
            raise ValueError("run receipt is missing training status")
        training = record["training"]
        if training is None:
            transitions = 0
        elif type(training) is dict and "completed_transitions" in training:
            transitions = _count(training["completed_transitions"], "completed_transitions")
            if transitions == 0:
                raise ValueError("a training receipt must report positive completed transitions")
        else:
            raise ValueError("training receipt is missing completed transitions")
        claims = record.get("claims")
        if type(claims) is not dict or claims.get("training_performed") is not (
            training is not None
        ):
            raise ValueError("training status contradicts the run's declared claim")
        runtime = record.get("runtime")
        if type(runtime) is not dict or "wall_seconds" not in runtime:
            raise ValueError("run receipt is missing recorded wall time")
        rows.append(
            {
                "manifest_path": str(Path(path).absolute()),
                "manifest_sha256": digest,
                "input_config_sha256": config_hash,
                "training_performed": training is not None,
                "reported_training_transitions": transitions,
                "reported_run_wall_seconds": _duration(runtime["wall_seconds"]),
            }
        )

    try:
        wall_seconds = math.fsum(row["reported_run_wall_seconds"] for row in rows)
    except OverflowError as exc:
        raise ValueError("summed run wall time overflowed") from exc
    if not math.isfinite(wall_seconds):
        raise ValueError("summed run wall time overflowed")
    transitions = _count(
        sum(row["reported_training_transitions"] for row in rows),
        "summed training transitions",
    )
    return {
        "schema_version": 1,
        "artifact": "post_training_effort_summary/v1",
        "evidence_scope": "selected_gmt_receipt_reported_values",
        "receipt_count": len(rows),
        "runs": rows,
        "selected_receipts": {
            "training_run_count": sum(row["training_performed"] for row in rows),
            "reported_training_transitions": transitions,
            "summed_run_wall_seconds": wall_seconds,
        },
        "unmeasured": {
            "complete_search_training_transitions": None,
            "total_simulator_steps_including_evaluation": None,
            "revision_rounds": None,
            "human_edits_or_interventions": None,
            "active_human_seconds": None,
            "model_call_cost_usd": None,
            "steps_to_target_success": None,
        },
        "claim_limits": [
            "hashes_pin_receipt_bytes_not_truth_or_execution_authority",
            "no_policy_load_or_simulator_execution",
            "selected_receipts_do_not_prove_all_attempts_were_included",
            "failed_or_unreceipted_attempt_costs_are_not_known",
            "task_failures_are_not_filtered_from_completed_runs",
            "run_wall_time_includes_training_evaluation_and_artifact_work",
            "summed_run_durations_are_not_campaign_elapsed_time",
            "no_task_success_learning_efficiency_or_VIBE_claim",
        ],
    }
