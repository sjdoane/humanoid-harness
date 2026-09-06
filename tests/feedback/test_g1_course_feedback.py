from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oracle_composition.adapters.gmt.course_evaluation import evaluate_episode
from oracle_composition.adapters.gmt.course_task import (
    CourseTaskSpec,
    TaskFrame,
    TaskRewardRecipe,
    evaluate_step,
)
from oracle_composition.adapters.gmt.io import sha256_file, write_deterministic_npz
from oracle_composition.feedback import g1_course as module


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qpos(x: float, height: float) -> np.ndarray:
    value = np.zeros(30, dtype=np.float64)
    value[:3] = (x, 0.0, height)
    value[3] = 1.0
    return value


def _reference(height: float = 0.50) -> np.ndarray:
    value = np.zeros(30, dtype=np.float64)
    value[0] = height
    return value


def _rows(spec: CourseTaskSpec) -> tuple[list[dict], dict[str, np.ndarray]]:
    task_frame = TaskFrame.initialize(
        np.zeros(2, dtype=np.float64), np.asarray([1.0, 0.0, 0.0, 0.0])
    )
    positions = (0.10, 0.30, 0.50, 0.80)
    heights = (0.70, 0.55, 0.48, 0.70)
    initial = _qpos(0.0, 0.70)
    before = initial
    frames = []
    qpos = [initial]
    reference = _reference()
    previous_inside = False
    for step, (position, height) in enumerate(zip(positions, heights, strict=True), start=1):
        after = _qpos(position, height)
        metrics = evaluate_step(
            spec=spec,
            frame=task_frame,
            before_qpos=before,
            after_qpos=after,
            ground_contact_bodies=(),
            current_reference=reference,
            control_step=step,
        )
        inside = metrics.inside_posture_region
        frames.append(
            {
                "metrics": metrics.to_dict(),
                "reward": {"task_reward": 0.0},
                "executed_mode": "inside" if inside else "travel",
                "executed_behavior": "crouch" if inside else "walk",
                "executed_phase_seconds": float(step * 0.02),
                "transition": (
                    {"control_step": step, "from": previous_inside, "to": inside}
                    if inside != previous_inside
                    else None
                ),
                "action_saturation_fraction": 0.0,
                "torque_saturation_fraction": 0.0,
                "trajectory": {
                    "qpos": after.tolist(),
                    "qvel": np.zeros(29, dtype=np.float64).tolist(),
                    "current_reference": reference.tolist(),
                    "composite_raw_action": np.zeros(23, dtype=np.float32).tolist(),
                    "contact_pairs": [],
                    "geom_body_names": ["world"],
                },
            }
        )
        qpos.append(after)
        before = after
        previous_inside = inside
    count = len(frames)
    arrays = {
        "qpos": np.asarray(qpos, dtype="<f8"),
        "qvel": np.zeros((count + 1, 29), dtype="<f8"),
        "residual_action": np.zeros((count, 23), dtype="<f4"),
        "current_reference": np.tile(reference.astype("<f4"), (count, 1)),
        "composite_raw_action": np.zeros((count, 23), dtype="<f4"),
    }
    return frames, arrays


def _write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    return sha256_file(path)


def _write_frames(path: Path, rows: list[dict]) -> str:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            for row in rows
        )
    )
    return sha256_file(path)


def _run_fixture(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, observe_region: bool = True
) -> tuple[Path, str, SimpleNamespace]:
    task = CourseTaskSpec(
        region_entry_distance_m=0.20 if observe_region else 2.0,
        region_exit_distance_m=0.60 if observe_region else 3.0,
        finish_distance_m=0.70 if observe_region else 4.0,
        target_speed_outside_m_s=1.0,
        target_speed_inside_m_s=0.65,
        posture_band_low_m=0.30,
        posture_band_high_m=0.60,
        horizon_steps=4,
    )
    recipe = TaskRewardRecipe(1.0, 2.0, 1.0, 0.5, 1.0)
    raw = {
        "schema_version": 1,
        "mode": "train",
        "assets": {},
        "task": task.to_dict(),
        "oracle": {},
        "segments": {},
        "reward": recipe.to_dict(),
        "seed": 7,
        "training_steps": 512,
    }
    config_bytes = (json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n").encode()
    config_sha = hashlib.sha256(config_bytes).hexdigest()
    config = SimpleNamespace(
        raw=raw,
        encoded=config_bytes,
        sha256=config_sha,
        task=task,
        recipe=recipe,
        program=SimpleNamespace(sha256="1" * 64),
        segments={"walk": SimpleNamespace(sha256="2" * 64)},
    )
    monkeypatch.setattr(module, "load_run_config", lambda path: config)
    root.mkdir()
    (root / "input_config.json").write_bytes(config_bytes)
    outputs = {"input_config.json": config_sha}
    summaries = {}
    rows, arrays = _rows(task)
    score = evaluate_episode(spec=task, frames=rows)
    for label, residual_rms in (("zero_residual", 0.0), ("final_policy", 0.1)):
        outputs[f"{label}_frames.jsonl"] = _write_frames(
            root / f"{label}_frames.jsonl", rows
        )
        outputs[f"{label}_trajectory.npz"] = write_deterministic_npz(
            root / f"{label}_trajectory.npz", arrays
        )
        summary = {
            "objective_evaluation": score,
            "training_reward_sum_not_success_metric": 0.0,
            "reset": {},
            "steps": len(rows),
            "residual_rms": residual_rms,
        }
        outputs[f"{label}_evaluation.json"] = _write_json(
            root / f"{label}_evaluation.json", summary
        )
        summaries[label] = summary
    for name in ("initial_residual_policy.npz", "final_residual_policy.npz"):
        (root / name).write_bytes(b"numeric policy fixture")
        outputs[name] = _sha(root / name)
    manifest = {
        "schema_version": 1,
        "artifact": "gmt_g1_course_development_run",
        "status": "completed",
        "input_config_sha256": config_sha,
        "outputs": outputs,
        "identities": {
            "task": task.sha256,
            "oracle": "1" * 64,
            "reward": recipe.sha256,
            "segments": {"walk": "2" * 64},
        },
        "frozen_runtime": {},
        "training": {"completed_transitions": 512},
        "zero_residual": summaries["zero_residual"],
        "final_policy": summaries["final_policy"],
        "runtime": {},
        "claims": {
            "development_only": True,
            "heldout_generalization_tested": False,
            "physical_obstacle_scene": False,
            "training_performed": True,
            "full_llm_revision_loop_demonstrated": False,
        },
    }
    manifest_path = root / "course_run_manifest.json"
    digest = _write_json(manifest_path, manifest)
    return manifest_path, digest, config


def _refresh_manifest(manifest_path: Path) -> str:
    manifest = json.loads(manifest_path.read_text())
    return _write_json(manifest_path, manifest)


def test_builds_exact_feedback_and_input_receipt(tmp_path, monkeypatch) -> None:
    manifest, digest, config = _run_fixture(tmp_path / "run", monkeypatch)

    result = module.build_g1_course_feedback(
        manifest_path=manifest,
        expected_manifest_sha256=digest,
        label="final_policy",
        output=tmp_path / "feedback",
    )

    feedback = json.loads((tmp_path / "feedback/feedback_v1.json").read_text())
    assert set(feedback) == {
        "evidence_class",
        "protected_evaluation",
        "source_manifest_sha256",
        "evaluation",
        "diagnosis",
    }
    assert set(feedback["evaluation"]) == set(module._EVALUATION_FIELDS)
    assert feedback["evaluation"]["task_sha256"] == config.task.sha256
    assert feedback["evaluation"]["episode_success"] is None
    assert "actual-minus-reference bias" in feedback["diagnosis"]
    assert "not held-out or task-success evidence" in feedback["diagnosis"]
    receipt = json.loads((tmp_path / "feedback/feedback_receipt_v1.json").read_text())
    assert receipt["inputs"]["source_manifest_sha256"] == digest
    assert receipt["inputs"]["label"] == "final_policy"
    assert receipt["output"]["sha256"] == result["feedback"]["sha256"]


def test_rejects_tampered_manifest_output(tmp_path, monkeypatch) -> None:
    manifest, digest, _ = _run_fixture(tmp_path / "run", monkeypatch)
    with (tmp_path / "run/final_policy_frames.jsonl").open("ab") as handle:
        handle.write(b"{}\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module.build_g1_course_feedback(
            manifest_path=manifest,
            expected_manifest_sha256=digest,
            label="final_policy",
            output=tmp_path / "feedback",
        )


def test_rejects_stored_objective_that_differs_from_recomputation(tmp_path, monkeypatch) -> None:
    manifest_path, _, _ = _run_fixture(tmp_path / "run", monkeypatch)
    manifest = json.loads(manifest_path.read_text())
    report_path = tmp_path / "run/final_policy_evaluation.json"
    report = json.loads(report_path.read_text())
    report["objective_evaluation"]["maximum_progress_m"] += 0.1
    manifest["final_policy"] = report
    manifest["outputs"][report_path.name] = _write_json(report_path, report)
    digest = _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="exact recomputation"):
        module.build_g1_course_feedback(
            manifest_path=manifest_path,
            expected_manifest_sha256=digest,
            label="final_policy",
            output=tmp_path / "feedback",
        )


def test_missing_region_is_explicit_missing_evidence(tmp_path, monkeypatch) -> None:
    manifest, digest, _ = _run_fixture(
        tmp_path / "run", monkeypatch, observe_region=False
    )

    module.build_g1_course_feedback(
        manifest_path=manifest,
        expected_manifest_sha256=digest,
        label="zero_residual",
        output=tmp_path / "feedback",
    )

    diagnosis = json.loads((tmp_path / "feedback/feedback_v1.json").read_text())["diagnosis"]
    assert "0 actual-region samples" in diagnosis
    assert "actual posture region was not observed" in diagnosis


def test_rejects_nonfresh_output_directory(tmp_path, monkeypatch) -> None:
    manifest, digest, _ = _run_fixture(tmp_path / "run", monkeypatch)
    output = tmp_path / "feedback"
    output.mkdir()

    with pytest.raises(FileExistsError):
        module.build_g1_course_feedback(
            manifest_path=manifest,
            expected_manifest_sha256=digest,
            label="final_policy",
            output=output,
        )


def test_rejects_frame_to_npz_crosslink_drift(tmp_path, monkeypatch) -> None:
    manifest_path, _, _ = _run_fixture(tmp_path / "run", monkeypatch)
    frames_path = tmp_path / "run/final_policy_frames.jsonl"
    rows = [json.loads(line) for line in frames_path.read_text().splitlines()]
    rows[0]["trajectory"]["qpos"][0] += 0.01
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"][frames_path.name] = _write_frames(frames_path, rows)
    digest = _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="frame and trajectory qpos differ"):
        module.build_g1_course_feedback(
            manifest_path=manifest_path,
            expected_manifest_sha256=digest,
            label="final_policy",
            output=tmp_path / "feedback",
        )
