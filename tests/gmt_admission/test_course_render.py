from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oracle_composition.adapters.gmt import course_render
from oracle_composition.adapters.gmt.course_render import (
    GROUND_OVERLAY_HALF_WIDTH_M,
    MAX_RENDER_FRAMES,
    course_ground_overlay,
    frame_annotation,
    hud_lines,
    load_course_render_inputs,
    select_frame_indices,
)
from oracle_composition.adapters.gmt.course_runtime import (
    AFTER_HEADING_FEEDBACK_RUNTIME,
    COURSE_RESIDUAL_RAW_SCALE,
    FINITE_HORIZON_RUNTIME,
    LEGACY_RUNTIME,
    LOOP_RUNTIME,
    frozen_runtime_contract,
)
from oracle_composition.adapters.gmt.course_task import CourseTaskSpec
from oracle_composition.adapters.gmt.io import GMTAdmissionError, write_deterministic_npz
from oracle_composition.adapters.gmt.training_contract import (
    TRAINING_REWARD_SCALE,
    CourseTrainerSpec,
    effective_training_contract,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task(steps: int) -> CourseTaskSpec:
    return CourseTaskSpec(
        region_entry_distance_m=1.0,
        region_exit_distance_m=2.0,
        finish_distance_m=3.5,
        target_speed_outside_m_s=1.0,
        target_speed_inside_m_s=0.5,
        posture_band_low_m=0.45,
        posture_band_high_m=0.85,
        horizon_steps=steps,
    )


def _metric(task: CourseTaskSpec, step: int, qpos: np.ndarray) -> dict[str, object]:
    return {
        "task_spec_sha256": task.sha256,
        "control_step": step,
        "progress_m": float(qpos[0]),
        "lateral_m": float(qpos[1]),
        "heading_error_signed_rad": 0.0,
        "root_height_m": float(qpos[2]),
        "torso_up": 1.0,
        "forward_speed_m_s": 1.0,
        "target_speed_m_s": 1.0,
        "speed_error_m_s": 0.0,
        "posture_band_error_m": 0.0,
        "lateral_error_m": 0.0,
        "heading_error_rad": 0.0,
        "inside_posture_region": False,
        "posture_success": False,
        "finish_condition_met": False,
        "horizon_reached": step == task.horizon_steps,
        "episode_success": None,
        "fallen": False,
        "failure_reasons": [],
        "joint_position_rmse_rad": 0.0,
        "root_height_abs_error_m": 0.0,
        "roll_pitch_rmse_rad": 0.0,
    }


def _reward(task: CourseTaskSpec) -> dict[str, object]:
    return {
        "task_spec_sha256": task.sha256,
        "task_reward_recipe_sha256": "c" * 64,
        "tracking_reward": 1.0,
        "task_reward": 1.0,
        "total_reward": 2.0,
        "speed_component_reward": 1.0,
        "posture_component_reward": 1.0,
        "lateral_component_reward": 1.0,
        "heading_component_reward": 1.0,
        "failure_penalty": 0.0,
    }


def _fixture(
    tmp_path: Path,
    *,
    steps: int = 3,
    mode: str = "probe",
    label: str = "zero_residual",
    nonzero_residual: bool = False,
    row_qpos_offset: float = 0.0,
    contact_substeps: int = 20,
    scaled: bool = False,
    low_rate: bool = False,
    fixed_normalizer: bool = False,
    loop_runtime: bool = False,
    finite_horizon_runtime: bool = False,
    heading_feedback_runtime: bool = False,
) -> tuple[Path, str, Path]:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    task = _task(steps)
    if sum((loop_runtime, finite_horizon_runtime, heading_feedback_runtime)) > 1:
        raise ValueError("fixture runtime must be unique")
    runtime = (
        AFTER_HEADING_FEEDBACK_RUNTIME
        if heading_feedback_runtime
        else LOOP_RUNTIME
        if loop_runtime
        else FINITE_HORIZON_RUNTIME
        if finite_horizon_runtime
        else LEGACY_RUNTIME
    )
    config = {
        "schema_version": runtime.config_schema_version,
        "mode": mode,
        "assets": {
            "upstream_root": str(upstream),
            "weights": {"path": "/unused/weights.npz", "sha256": "a" * 64},
            "motions": {},
        },
        "task": task.to_dict(),
        "oracle": {},
        "segments": {},
        "reward": {},
        "seed": 7,
        "training_steps": 0 if mode == "probe" else 512,
    }
    if runtime.config_value is not None:
        config["runtime"] = runtime.config_value
    trainer = (
        CourseTrainerSpec(
            TRAINING_REWARD_SCALE,
            profile_version=3 if fixed_normalizer else 2 if low_rate else 1,
        )
        if scaled or fixed_normalizer
        else None
    )
    if trainer is not None:
        config["trainer"] = trainer.to_dict()
    config_path = tmp_path / "input_config.json"
    config_path.write_bytes((json.dumps(config, sort_keys=True) + "\n").encode())
    config_sha = _sha(config_path)

    qpos = np.zeros((steps + 1, 30), dtype="<f8")
    qpos[:, 0] = np.arange(steps + 1) * 0.02
    qpos[:, 2] = 0.8
    qpos[:, 3] = 1.0
    qvel = np.zeros((steps + 1, 29), dtype="<f8")
    qvel[:, 0] = 1.0
    residual = np.zeros((steps, 23), dtype="<f4")
    if nonzero_residual:
        residual[0, 0] = np.float32(0.1)
    reference = np.zeros((steps, 30), dtype="<f4")
    action = np.zeros((steps, 23), dtype="<f4")
    trace_name = f"{label}_trajectory.npz"
    trace_path = tmp_path / trace_name
    write_deterministic_npz(
        trace_path,
        {
            "qpos": qpos,
            "qvel": qvel,
            "residual_action": residual,
            "current_reference": reference,
            "composite_raw_action": action,
        },
    )
    rows = []
    for index in range(steps):
        row_qpos = qpos[index + 1].copy()
        row_qpos[0] += row_qpos_offset
        rows.append(
            {
                "metrics": _metric(task, index + 1, qpos[index + 1]),
                "reward": _reward(task),
                "executed_mode": ("inside" if index == 1 else "before"),
                "executed_behavior": ("crouch" if index == 1 else "walk"),
                "executed_phase_seconds": index * 0.02,
                "transition": None,
                "action_saturation_fraction": 0.0,
                "torque_saturation_fraction": 0.0,
                "trajectory": {
                    "qpos": row_qpos.tolist(),
                    "qvel": qvel[index + 1].tolist(),
                    "current_reference": reference[index].tolist(),
                    "composite_raw_action": action[index].tolist(),
                    "contact_pairs": [[] for _ in range(contact_substeps)],
                    "geom_body_names": ["world", "left_ankle_roll_link"],
                },
            }
        )
    rows_name = f"{label}_frames.jsonl"
    rows_path = tmp_path / rows_name
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    )
    report = {
        "objective_evaluation": {},
        "training_reward_sum_not_success_metric": 0.0,
        "reset": {},
        "steps": steps,
        "residual_rms": 0.0 if label == "zero_residual" else 0.1,
    }
    outputs = {
        "input_config.json": config_sha,
        rows_name: _sha(rows_path),
        trace_name: _sha(trace_path),
    }
    manifest = {
        "schema_version": 1,
        "artifact": "gmt_g1_course_development_run",
        "status": "completed",
        "input_config_sha256": config_sha,
        "outputs": outputs,
        "identities": {"task": task.sha256, "oracle": "a" * 64, "reward": "b" * 64},
        "frozen_runtime": frozen_runtime_contract(
            runtime,
            trainer=effective_training_contract(trainer),
            residual_raw_scale=COURSE_RESIDUAL_RAW_SCALE,
        ),
        "training": {} if mode == "train" else None,
        "zero_residual": report if label == "zero_residual" else None,
        "final_policy": report if label == "final_policy" else None,
        "runtime": {},
        "claims": {
            "development_only": True,
            "training_performed": mode == "train",
        },
    }
    manifest_path = tmp_path / "course_run_manifest.json"
    manifest_path.write_bytes((json.dumps(manifest, sort_keys=True) + "\n").encode())
    return manifest_path, _sha(manifest_path), upstream


@pytest.fixture(autouse=True)
def _pin_fake_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(course_render, "verify_upstream_root", lambda _path: {"g1.xml": "f" * 64})


def test_admission_crosslinks_data_without_importing_mujoco(tmp_path: Path) -> None:
    manifest, digest, upstream = _fixture(tmp_path)
    before = "mujoco" in sys.modules
    admitted = load_course_render_inputs(
        manifest_path=manifest,
        manifest_sha256=digest,
        upstream_root=upstream,
        label="zero_residual",
    )

    assert admitted.arrays["qpos"].shape == (4, 30)
    assert admitted.arrays["qpos"].flags.writeable is False
    assert admitted.consumed_outputs["input_config.json"] == admitted.config_sha256
    assert ("mujoco" in sys.modules) is before
    annotation = frame_annotation(admitted, 2)
    assert annotation.executed_mode == "inside"
    assert annotation.progress_m == pytest.approx(0.04)
    assert annotation.forward_speed_m_s == pytest.approx(1.0)


def test_admission_accepts_exact_scaled_trainer_runtime(tmp_path: Path) -> None:
    manifest, digest, upstream = _fixture(
        tmp_path, mode="train", label="final_policy", scaled=True
    )

    admitted = load_course_render_inputs(
        manifest_path=manifest,
        manifest_sha256=digest,
        upstream_root=upstream,
        label="final_policy",
    )

    assert admitted.label == "final_policy"


def test_admission_accepts_exact_low_rate_trainer_runtime(tmp_path: Path) -> None:
    manifest, digest, upstream = _fixture(
        tmp_path,
        mode="train",
        label="final_policy",
        scaled=True,
        low_rate=True,
    )

    admitted = load_course_render_inputs(
        manifest_path=manifest,
        manifest_sha256=digest,
        upstream_root=upstream,
        label="final_policy",
    )

    assert admitted.label == "final_policy"


def test_admission_accepts_exact_fixed_normalizer_trainer_runtime(
    tmp_path: Path,
) -> None:
    manifest, digest, upstream = _fixture(
        tmp_path,
        mode="train",
        label="final_policy",
        fixed_normalizer=True,
    )

    admitted = load_course_render_inputs(
        manifest_path=manifest,
        manifest_sha256=digest,
        upstream_root=upstream,
        label="final_policy",
    )

    assert admitted.label == "final_policy"


def test_admission_accepts_exact_probe_only_loop_runtime(tmp_path: Path) -> None:
    manifest, digest, upstream = _fixture(tmp_path, loop_runtime=True)

    admitted = load_course_render_inputs(
        manifest_path=manifest,
        manifest_sha256=digest,
        upstream_root=upstream,
        label="zero_residual",
    )

    assert admitted.label == "zero_residual"
    retained = json.loads((tmp_path / "input_config.json").read_text())
    assert retained["runtime"] == LOOP_RUNTIME.config_value


def test_admission_reconstructs_after_heading_feedback_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, digest, upstream = _fixture(tmp_path, heading_feedback_runtime=True)
    encoded = (tmp_path / "input_config.json").read_bytes()
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        course_render,
        "load_run_config",
        lambda _path: SimpleNamespace(encoded=encoded),
    )

    def validate(**kwargs):
        observed.update(kwargs)

    monkeypatch.setattr(course_render, "validate_after_heading_feedback_trace", validate)

    admitted = load_course_render_inputs(
        manifest_path=manifest,
        manifest_sha256=digest,
        upstream_root=upstream,
        label="zero_residual",
    )

    assert admitted.label == "zero_residual"
    assert len(observed["frames"]) == 3
    assert observed["trajectory"]["qpos"].shape == (4, 30)


def test_admission_rejects_invalid_after_heading_feedback_trace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, digest, upstream = _fixture(tmp_path, heading_feedback_runtime=True)
    encoded = (tmp_path / "input_config.json").read_bytes()
    monkeypatch.setattr(
        course_render,
        "load_run_config",
        lambda _path: SimpleNamespace(encoded=encoded),
    )

    def reject(**_kwargs):
        raise ValueError("tampered trace")

    monkeypatch.setattr(course_render, "validate_after_heading_feedback_trace", reject)

    with pytest.raises(GMTAdmissionError, match="trace is invalid"):
        load_course_render_inputs(
            manifest_path=manifest,
            manifest_sha256=digest,
            upstream_root=upstream,
            label="zero_residual",
        )


def test_admission_accepts_exact_finite_horizon_training_runtime(tmp_path: Path) -> None:
    manifest, digest, upstream = _fixture(
        tmp_path,
        mode="train",
        label="final_policy",
        fixed_normalizer=True,
        finite_horizon_runtime=True,
    )

    admitted = load_course_render_inputs(
        manifest_path=manifest,
        manifest_sha256=digest,
        upstream_root=upstream,
        label="final_policy",
    )

    assert admitted.label == "final_policy"
    retained = json.loads((tmp_path / "input_config.json").read_text())
    assert retained["runtime"] == FINITE_HORIZON_RUNTIME.config_value


def test_admission_rejects_manifest_config_identity_disagreement(tmp_path: Path) -> None:
    manifest_path, _digest, upstream = _fixture(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    manifest["outputs"]["input_config.json"] = "e" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")

    with pytest.raises(GMTAdmissionError, match="config identities differ"):
        load_course_render_inputs(
            manifest_path=manifest_path,
            manifest_sha256=_sha(manifest_path),
            upstream_root=upstream,
            label="zero_residual",
        )


@pytest.mark.parametrize(
    ("fixture_kwargs", "message"),
    [
        ({"row_qpos_offset": 0.01}, "JSONL/NPZ mismatch"),
        ({"contact_substeps": 19}, "all 20 substeps"),
        ({"nonzero_residual": True}, "nonzero residual action"),
    ],
)
def test_admission_rejects_cross_artifact_divergence(
    tmp_path: Path, fixture_kwargs: dict[str, object], message: str
) -> None:
    manifest, digest, upstream = _fixture(tmp_path, **fixture_kwargs)
    with pytest.raises(GMTAdmissionError, match=message):
        load_course_render_inputs(
            manifest_path=manifest,
            manifest_sha256=digest,
            upstream_root=upstream,
            label="zero_residual",
        )


def test_admission_rejects_tampered_consumed_trace(tmp_path: Path) -> None:
    manifest, digest, upstream = _fixture(tmp_path)
    with (tmp_path / "zero_residual_trajectory.npz").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(GMTAdmissionError, match="SHA-256 mismatch"):
        load_course_render_inputs(
            manifest_path=manifest,
            manifest_sha256=digest,
            upstream_root=upstream,
            label="zero_residual",
        )


def test_final_policy_requires_train_mode_and_selected_outputs(tmp_path: Path) -> None:
    manifest, digest, upstream = _fixture(tmp_path)
    with pytest.raises(GMTAdmissionError, match="selected course render artifacts are absent"):
        load_course_render_inputs(
            manifest_path=manifest,
            manifest_sha256=digest,
            upstream_root=upstream,
            label="final_policy",
        )


def test_frame_selection_is_bounded_deterministic_and_keeps_endpoints() -> None:
    indices = select_frame_indices(2_001)
    assert len(indices) == MAX_RENDER_FRAMES
    assert indices[0] == 0
    assert indices[-1] == 2_000
    assert np.all(np.diff(indices) > 0)
    np.testing.assert_array_equal(indices, select_frame_indices(2_001))

    short = select_frame_indices(2)
    np.testing.assert_array_equal(short, [0, 1])


def test_course_overlay_uses_initial_heading_and_exact_task_distances() -> None:
    qpos = np.zeros(30, dtype="<f8")
    qpos[:2] = (10.0, -4.0)
    qpos[3:7] = (2**-0.5, 0.0, 0.0, 2**-0.5)  # 90-degree yaw, wxyz.
    overlay = course_ground_overlay(_task(10), qpos)

    np.testing.assert_allclose(overlay.region_center_xyz, (10.0, -2.5, 0.003), atol=1e-12)
    assert overlay.region_half_size_xyz == (0.5, GROUND_OVERLAY_HALF_WIDTH_M, 0.002)
    np.testing.assert_allclose(overlay.entry_line[0], (12.0, -3.0, 0.008), atol=1e-12)
    np.testing.assert_allclose(overlay.entry_line[1], (8.0, -3.0, 0.008), atol=1e-12)
    np.testing.assert_allclose(overlay.exit_line[0][1], -2.0, atol=1e-12)
    np.testing.assert_allclose(overlay.finish_line[0][1], -0.5, atol=1e-12)


def test_hud_directly_labels_targets_failures_and_nonphysical_markers(tmp_path: Path) -> None:
    manifest, digest, upstream = _fixture(tmp_path)
    admitted = load_course_render_inputs(
        manifest_path=manifest,
        manifest_sha256=digest,
        upstream_root=upstream,
        label="zero_residual",
    )
    annotation = replace(
        frame_annotation(admitted, 2),
        progress_m=1.5,
        posture_region_active=True,
        target_speed_m_s=0.5,
        recorded_failure_reasons=("non_foot_ground_contact",),
    )
    text = "\n".join(hud_lines(admitted, annotation))

    assert "Blue cue=[1.00,2.00)m" in text
    assert "orange line=finish 3.50m" in text
    assert "NOT physical obstacles" in text
    assert "speed current=1.000 | target=0.500 m/s" in text
    assert "target band=[0.450,0.850]m (ACTIVE)" in text
    assert "recorded failures: non_foot_ground_contact" in text
    assert "no policy/dynamics rerun" in text


def test_render_overlay_adds_display_geoms_only() -> None:
    class _GeomType:
        mjGEOM_BOX = 6
        mjGEOM_LINE = 100

    class _FakeMujoco:
        mjtGeom = _GeomType

        def __init__(self) -> None:
            self.initialized: list[tuple[object, int]] = []
            self.connected: list[tuple[object, int, float]] = []

        def mjv_initGeom(self, geom, geom_type, _size, _pos, _mat, _rgba) -> None:
            self.initialized.append((geom, geom_type))

        def mjv_connector(self, geom, geom_type, width, _start, _end) -> None:
            self.connected.append((geom, geom_type, width))

    class _Scene:
        def __init__(self) -> None:
            self.ngeom = 0
            self.maxgeom = 4
            self.geoms = [object() for _ in range(4)]

    qpos = np.zeros(30, dtype="<f8")
    qpos[3] = 1.0
    fake_mujoco, scene = _FakeMujoco(), _Scene()
    course_render._add_course_overlay(
        fake_mujoco,
        scene,
        course_ground_overlay(_task(10), qpos),
    )

    assert scene.ngeom == 4
    assert [geom_type for _, geom_type in fake_mujoco.initialized] == [6, 100, 100, 100]
    assert [width for _, _, width in fake_mujoco.connected] == [3.0, 3.0, 6.0]
