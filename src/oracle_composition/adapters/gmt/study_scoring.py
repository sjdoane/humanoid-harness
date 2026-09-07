"""Data-only paired scoring for retained G1 course development runs."""

from __future__ import annotations

import hashlib
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from oracle_composition.feedback.g1_course import build_g1_course_feedback
from oracle_composition.harness.contract import decode_json_object, read_json_object

from .contracts import CONTROL_DT_SECONDS
from .course_config import load_run_config
from .course_runtime import (
    COURSE_RESIDUAL_RAW_SCALE,
    FINITE_HORIZON_RUNTIME,
    FOUR_STATE_FINITE_HORIZON_RUNTIME,
    LEGACY_RUNTIME,
    LOOP_RUNTIME,
    CourseRuntimeProfile,
    frozen_runtime_contract,
)
from .training_contract import CourseTrainerSpec, effective_training_contract
from .training_normalizer import FIXED_NORMALIZER_STATE_SHA256
from .training_telemetry import telemetry_filename

PAIR_SCORE_ID = "gmt_g1_course_study_pair_score/v1"
RESOURCE_RECEIPT_FILENAME = "gmt_probe_resource_receipt_v1.json"
_RESOURCE_RECEIPT_ARTIFACT = "gmt_g1_course_development_resource_receipt"
_HEX = frozenset("0123456789abcdef")


def _digest(value: object, *, length: int, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != length
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{field} must be one lowercase hexadecimal digest")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identities(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {"task", "oracle", "reward", "segments"}:
        raise ValueError("expected semantic identity fields differ")
    result = {
        name: _digest(value[name], length=64, field=f"expected {name} identity")
        for name in ("task", "oracle", "reward")
    }
    segments = value["segments"]
    if (
        type(segments) is not dict
        or not 1 <= len(segments) <= 8
        or any(type(name) is not str or not name for name in segments)
    ):
        raise ValueError("expected segment identities differ")
    result["segments"] = {
        name: _digest(digest, length=64, field=f"expected segment {name}")
        for name, digest in segments.items()
    }
    return result


@dataclass(frozen=True, slots=True)
class RunManifestBinding:
    """Content-address one retained run and its resource receipt."""

    path: Path
    sha256: str
    resource_receipt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise ValueError("run manifest path must be a Path")
        _digest(self.sha256, length=64, field="run manifest SHA-256")
        _digest(
            self.resource_receipt_sha256,
            length=64,
            field="resource receipt SHA-256",
        )


@dataclass(frozen=True, slots=True)
class CourseStudyExpectation:
    """Predeclared identity and runtime expectations for one study cell."""

    cell_id: str
    seed: int
    training_steps: int
    identities: Mapping[str, object]
    trainer: CourseTrainerSpec | None
    base_state_sha256: str
    source_commit: str
    runtime: CourseRuntimeProfile = LEGACY_RUNTIME

    def __post_init__(self) -> None:
        if (
            type(self.cell_id) is not str
            or not 1 <= len(self.cell_id) <= 64
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_.-"
                for character in self.cell_id
            )
        ):
            raise ValueError("study cell id must be one bounded lowercase identifier")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("expected seed must be a nonnegative signed 32-bit integer")
        if (
            type(self.training_steps) is not int
            or not 512 <= self.training_steps <= 262_144
            or self.training_steps % 512
        ):
            raise ValueError("expected training budget must be a bounded rollout multiple")
        object.__setattr__(self, "identities", _identities(dict(self.identities)))
        if self.trainer is not None and type(self.trainer) is not CourseTrainerSpec:
            raise ValueError("expected trainer must be the admitted profile or None")
        if self.runtime not in {
            LEGACY_RUNTIME,
            LOOP_RUNTIME,
            FINITE_HORIZON_RUNTIME,
            FOUR_STATE_FINITE_HORIZON_RUNTIME,
        }:
            raise ValueError("expected course runtime profile is not admitted")
        _digest(self.base_state_sha256, length=64, field="expected base state SHA-256")
        _digest(self.source_commit, length=40, field="expected source commit")


def _resource_linkage(
    *,
    run_root: Path,
    manifest: dict[str, object],
    binding: RunManifestBinding,
    expectation: CourseStudyExpectation,
) -> dict[str, object]:
    receipt_path = run_root / RESOURCE_RECEIPT_FILENAME
    receipt, encoded = read_json_object(receipt_path)
    observed_receipt_sha256 = hashlib.sha256(encoded).hexdigest()
    if observed_receipt_sha256 != binding.resource_receipt_sha256:
        raise ValueError("resource receipt bytes differ from their expected identity")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("artifact") != _RESOURCE_RECEIPT_ARTIFACT
        or receipt.get("status") != "succeeded"
        or receipt.get("commit") != expectation.source_commit
    ):
        raise ValueError("resource receipt identity or source commit differs")

    inputs = receipt.get("inputs")
    artifacts = receipt.get("artifacts")
    config_binding = inputs.get("config") if type(inputs) is dict else None
    repository = inputs.get("repository_sources") if type(inputs) is dict else None
    course_manifest = artifacts.get("course_manifest") if type(artifacts) is dict else None
    output_records = artifacts.get("outputs") if type(artifacts) is dict else None
    config_path = config_binding.get("path") if type(config_binding) is dict else None
    if (
        type(inputs) is not dict
        or inputs.get("workload") != "course"
        or inputs.get("course_mode") != "train"
        or inputs.get("artifact_contract") != "gmt_g1_fixed_development_launcher/v1"
        or type(config_binding) is not dict
        or type(config_path) is not str
        or config_binding.get("sha256") != manifest.get("input_config_sha256")
        or type(repository) is not dict
        or type(repository.get("file_count")) is not int
        or repository["file_count"] < 1
        or type(course_manifest) is not dict
        or course_manifest.get("path") != "course_run_manifest.json"
        or course_manifest.get("sha256") != binding.sha256
        or course_manifest.get("size") != binding.path.stat().st_size
        or type(output_records) is not dict
    ):
        raise ValueError("resource receipt does not bind the retained run")
    tree_sha256 = _digest(
        repository.get("canonical_tree_sha256"),
        length=64,
        field="resource source tree SHA-256",
    )

    expected_trainer = effective_training_contract(expectation.trainer)
    expected_course_runtime = expectation.runtime.manifest_contract()
    if expectation.trainer is None:
        if "trainer" in inputs:
            raise ValueError("raw trainer receipt must preserve the legacy input shape")
    elif inputs.get("trainer") != expected_trainer:
        raise ValueError("resource receipt trainer differs from the expected trainer")
    if expected_course_runtime is None:
        if "course_runtime" in inputs:
            raise ValueError("legacy resource receipt must preserve the runtime input shape")
    elif inputs.get("course_runtime") != expected_course_runtime:
        raise ValueError("resource receipt runtime differs from the expected course runtime")

    outputs = manifest.get("outputs")
    if type(outputs) is not dict or set(output_records) != set(outputs):
        raise ValueError("resource receipt output set differs from the run manifest")
    for name, expected_sha256 in outputs.items():
        record = output_records[name]
        path = run_root / name
        if (
            type(record) is not dict
            or set(record) != {"path", "sha256", "size"}
            or record["path"] != name
            or record["sha256"] != expected_sha256
            or type(record["size"]) is not int
            or record["size"] != path.stat().st_size
        ):
            raise ValueError(f"resource receipt output binding differs: {name}")

    argv = receipt.get("canonical_argv")
    if (
        type(argv) is not list
        or len(argv) != 7
        or argv[1:4] != ["-m", "oracle_composition.adapters.gmt.course_run", "--config"]
        or argv[4] != config_path
        or argv[5] != "--output"
        or type(argv[6]) is not str
        or Path(argv[6]).resolve() != run_root
    ):
        raise ValueError("resource receipt command does not bind the retained run")
    return {
        "path": str(receipt_path),
        "sha256": observed_receipt_sha256,
        "source_commit": expectation.source_commit,
        "source_tree_sha256": tree_sha256,
    }


def _updates(
    run_root: Path,
    manifest: dict[str, object],
    trainer: CourseTrainerSpec | None,
    *,
    runtime: CourseRuntimeProfile = LEGACY_RUNTIME,
) -> dict[str, object]:
    filename = telemetry_filename(
        reward_scale=(trainer.total_training_reward_scale if trainer is not None else None),
        fixed_normalizer_sha256=(
            FIXED_NORMALIZER_STATE_SHA256
            if trainer is not None and trainer.uses_fixed_observation_normalizer
            else None
        ),
        runtime=runtime,
    )
    outputs = manifest["outputs"]
    if type(outputs) is not dict or filename not in outputs:
        raise ValueError("study scoring requires version-matched training telemetry")
    rows = [
        decode_json_object(line, source=f"training telemetry row {index}")
        for index, line in enumerate((run_root / filename).read_bytes().splitlines(), start=1)
    ]
    updates = [
        row["previous_update"] for row in rows[:-1] if row["previous_update"] is not None
    ] + [rows[-1]["update"]]
    values = [update["metrics"]["explained_variance"] for update in updates]
    tail = values[-16:]
    available = len(tail) == 16 and all(type(value) in {int, float} for value in tail)
    completed_episodes = sum(row["episodes"]["completed"] for row in rows[:-1])
    falls = sum(row["episodes"]["falls"] for row in rows[:-1])
    return {
        "telemetry_path": filename,
        "update_count": len(updates),
        "completed_episode_count": completed_episodes,
        "fall_count": falls,
        "trained_through_transitions": [
            update["trained_through_transitions"] for update in updates
        ],
        "explained_variance_by_update": values,
        "explained_variance_last16_mean": fmean(tail) if available else None,
        "explained_variance_last16_available": available,
        "explained_variance_last16_at_least_0_5": available and fmean(tail) >= 0.5,
    }


def _checkpoint(frames: list[dict[str, Any]], seconds: int) -> dict[str, object]:
    step = round(seconds / CONTROL_DT_SECONDS)
    if step > len(frames):
        return {
            "control_step": step,
            "available": False,
            "reason": "trace_ended_before_checkpoint",
            "executed_mode": None,
            "lateral_error_m": None,
        }
    row = frames[step - 1]
    return {
        "control_step": step,
        "available": True,
        "reason": None,
        "executed_mode": row["executed_mode"],
        "lateral_error_m": row["metrics"]["lateral_error_m"],
    }


def _phase_measures(
    frames: list[dict[str, Any]], objective: dict[str, object]
) -> dict[str, object]:
    after = [row for row in frames if row["executed_mode"] == "after"]
    region = objective["region"]
    speed = objective["speed"]
    return {
        "actual_task_region": {
            "sample_count": region["inside_sample_count"],
            "entry_observed": region["entry_observed"],
            "exit_observed_after_entry": region["exit_observed_after_entry"],
            "first_entry_step": region["first_entry_step"],
            "first_exit_step": region["first_exit_step"],
            "posture_compliant_fraction": region["posture_compliant_fraction"],
            "minimum_root_height_m": region["minimum_root_height_m"],
            "mean_forward_speed_m_s": speed["inside_mean_forward_speed_m_s"],
            "mean_absolute_speed_error_m_s": speed["inside_mean_absolute_error_m_s"],
            "mean_speed_target_deviation_m_s": speed["inside_mean_speed_target_deviation_m_s"],
        },
        "executed_after_state": {
            "sample_count": len(after),
            "first_control_step": after[0]["metrics"]["control_step"] if after else None,
            "mean_absolute_speed_error_m_s": (
                fmean(row["metrics"]["speed_error_m_s"] for row in after) if after else None
            ),
            "maximum_lateral_error_m": (
                max(row["metrics"]["lateral_error_m"] for row in after) if after else None
            ),
            "final_progress_m": objective["final_progress_m"],
            "lateral_checkpoints": {
                "10_seconds": _checkpoint(frames, 10),
                "20_seconds": _checkpoint(frames, 20),
            },
        },
        "partition_note": (
            "actual_task_region_uses_position; executed_after_state_uses_oracle_diagnostics"
        ),
    }


def _score_cell(
    binding: RunManifestBinding, expectation: CourseStudyExpectation, temp_root: Path
) -> dict[str, object]:
    manifest_path = binding.path.resolve(strict=True)
    if _sha256_file(manifest_path) != binding.sha256:
        raise ValueError("run manifest bytes differ from their expected identity")
    feedback_dir = temp_root / expectation.cell_id
    feedback_result = build_g1_course_feedback(
        manifest_path=manifest_path,
        expected_manifest_sha256=binding.sha256,
        label="final_policy",
        output=feedback_dir,
    )
    manifest = decode_json_object(manifest_path.read_bytes(), source="course run manifest")
    run_root = manifest_path.parent
    config = load_run_config(run_root / "input_config.json")
    if (
        config.raw["mode"] != "train"
        or config.raw["seed"] != expectation.seed
        or config.raw["training_steps"] != expectation.training_steps
        or config.trainer != expectation.trainer
        or config.runtime != expectation.runtime
        or manifest.get("identities") != expectation.identities
        or manifest.get("input_config_sha256") != config.sha256
    ):
        raise ValueError("retained run differs from its predeclared study cell")
    frozen_runtime = manifest.get("frozen_runtime")
    expected_runtime = frozen_runtime_contract(
        expectation.runtime,
        trainer=effective_training_contract(expectation.trainer),
        residual_raw_scale=COURSE_RESIDUAL_RAW_SCALE,
    )
    if frozen_runtime != expected_runtime:
        raise ValueError("effective runtime differs from its predeclared study cell")
    training = manifest.get("training")
    if (
        type(training) is not dict
        or training.get("completed_transitions") != expectation.training_steps
        or training.get("policy_artifact") != "numeric_weights_not_optimizer_resume"
        or training.get("frozen_base_state_before_sha256") != expectation.base_state_sha256
        or training.get("frozen_base_state_after_sha256") != expectation.base_state_sha256
        or type(training.get("episodes")) is not int
        or type(training.get("falls")) is not int
        or not 0 <= training["falls"] <= training["episodes"]
    ):
        raise ValueError("training budget or frozen base state differs")
    resource = _resource_linkage(
        run_root=run_root,
        manifest=manifest,
        binding=binding,
        expectation=expectation,
    )
    feedback = decode_json_object(
        (feedback_dir / "feedback_v1.json").read_bytes(), source="verified course feedback"
    )
    frames = [
        decode_json_object(line, source=f"final policy frame {index}")
        for index, line in enumerate(
            (run_root / "final_policy_frames.jsonl").read_bytes().splitlines(), start=1
        )
    ]
    objective = manifest["final_policy"]["objective_evaluation"]
    updates = _updates(
        run_root,
        manifest,
        expectation.trainer,
        runtime=expectation.runtime,
    )
    expected_updates = expectation.training_steps // 512
    if updates["update_count"] != expected_updates:
        raise ValueError("training update count differs from the fixed budget")
    if (
        training["episodes"] != updates["completed_episode_count"]
        or training["falls"] != updates["fall_count"]
    ):
        raise ValueError("manifest training episode/fall totals differ from validated telemetry")
    result = {
        "cell_id": expectation.cell_id,
        "manifest": {"path": str(manifest_path), "sha256": binding.sha256},
        "resource": resource,
        "config_sha256": config.sha256,
        "identities": expectation.identities,
        "effective_trainer": effective_training_contract(expectation.trainer),
        "base_state_sha256": expectation.base_state_sha256,
        "feedback": {
            "sha256": feedback_result["feedback"]["sha256"],
            "receipt_sha256": feedback_result["receipt"]["sha256"],
            "protected_evaluation": feedback["protected_evaluation"],
        },
        "training": {
            "completed_transitions": training["completed_transitions"],
            "episodes": training["episodes"],
            "falls": training["falls"],
        },
        "objective": objective,
        "phase_measures": _phase_measures(frames, objective),
        "gates": {
            "development": objective["development_gate_results"],
            "development_all_passed": objective["development_gate_passed"],
            "explained_variance_last16_at_least_0_5": updates[
                "explained_variance_last16_at_least_0_5"
            ],
        },
        "learning": updates,
    }
    course_runtime = expectation.runtime.manifest_contract()
    if course_runtime is not None:
        result["course_runtime"] = course_runtime
    return result


def score_course_study_pair(
    *,
    first: RunManifestBinding,
    first_expected: CourseStudyExpectation,
    second: RunManifestBinding,
    second_expected: CourseStudyExpectation,
) -> dict[str, object]:
    """Validate and score two matched retained cells without altering their evidence."""

    if first_expected.cell_id == second_expected.cell_id:
        raise ValueError("paired study cell ids must be distinct")
    if (
        first_expected.seed != second_expected.seed
        or first_expected.training_steps != second_expected.training_steps
        or first_expected.base_state_sha256 != second_expected.base_state_sha256
        or first_expected.identities["task"] != second_expected.identities["task"]
    ):
        raise ValueError("paired study cells differ in seed, budget, task, or base state")
    with tempfile.TemporaryDirectory(prefix="humanoid-harness-study-score-") as raw:
        temp_root = Path(raw).resolve(strict=True)
        cells = {
            first_expected.cell_id: _score_cell(first, first_expected, temp_root),
            second_expected.cell_id: _score_cell(second, second_expected, temp_root),
        }
    if (
        first_expected.source_commit == second_expected.source_commit
        and cells[first_expected.cell_id]["resource"]["source_tree_sha256"]
        != cells[second_expected.cell_id]["resource"]["source_tree_sha256"]
    ):
        raise ValueError("same-commit study cells report different source trees")
    identity_differences = [
        name
        for name in ("oracle", "reward", "segments")
        if first_expected.identities[name] != second_expected.identities[name]
    ]
    pair = {
        "seed": first_expected.seed,
        "training_steps": first_expected.training_steps,
        "task_sha256": first_expected.identities["task"],
        "base_state_sha256": first_expected.base_state_sha256,
        "semantic_identity_differences": identity_differences,
    }
    if first_expected.runtime != LEGACY_RUNTIME or second_expected.runtime != LEGACY_RUNTIME:
        pair["course_runtime_profile_difference"] = (
            first_expected.runtime != second_expected.runtime
        )
    return {
        "schema_id": PAIR_SCORE_ID,
        "schema_version": 1,
        "claim_scope": "fixed_development_pair_not_heldout_or_universal",
        "pair": pair,
        "cells": cells,
    }


__all__ = [
    "PAIR_SCORE_ID",
    "CourseStudyExpectation",
    "RunManifestBinding",
    "score_course_study_pair",
]
