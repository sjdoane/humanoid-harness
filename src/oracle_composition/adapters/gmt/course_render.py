"""Render a verified GMT course trajectory without rerunning policy or dynamics."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import re
import stat
import tempfile
import textwrap
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oracle_composition.harness.contract import decode_json_object, read_json_object

from .checkpoint import verify_upstream_root
from .contracts import (
    ACTION_DIM,
    CONTROL_DT_SECONDS,
    GMT_UPSTREAM_COMMIT,
    REFERENCE_FRAME_DIM,
)
from .course_config import load_run_config
from .course_runtime import (
    COURSE_RESIDUAL_RAW_SCALE,
    frozen_runtime_contract,
    runtime_profile_from_config,
)
from .course_task import CourseTaskSpec, TaskFrame
from .heading_feedback import (
    AFTER_HEADING_FEEDBACK_TRACE_KEY,
    validate_after_heading_feedback_trace,
)
from .io import GMTAdmissionError, read_verified_bytes, sha256_file, write_json_receipt
from .replay import _extract_model_abi, validate_model_abi
from .training_contract import CourseTrainerSpec, effective_training_contract

FRAME_WIDTH = 480
FRAME_HEIGHT = 360
OUTPUT_FPS = 20
MAX_RENDER_FRAMES = 800
MAX_COURSE_STEPS = 2_000
GROUND_OVERLAY_HALF_WIDTH_M = 2.0

_SHA256 = re.compile(r"[0-9a-f]{64}")
_MANIFEST_KEYS = {
    "schema_version",
    "artifact",
    "status",
    "input_config_sha256",
    "outputs",
    "identities",
    "frozen_runtime",
    "training",
    "zero_residual",
    "final_policy",
    "runtime",
    "claims",
}
_CONFIG_KEYS = {
    "schema_version",
    "mode",
    "assets",
    "task",
    "oracle",
    "segments",
    "reward",
    "seed",
    "training_steps",
}
_CONFIG_KEYS_WITH_TRAINER = {*_CONFIG_KEYS, "trainer"}
_CONFIG_KEYS_WITH_RUNTIME = {*_CONFIG_KEYS, "runtime"}
_CONFIG_KEYS_WITH_RUNTIME_AND_TRAINER = {*_CONFIG_KEYS_WITH_RUNTIME, "trainer"}
_REPORT_KEYS = {
    "objective_evaluation",
    "training_reward_sum_not_success_metric",
    "reset",
    "steps",
    "residual_rms",
}
_ROW_KEYS = {
    "metrics",
    "reward",
    "executed_mode",
    "executed_behavior",
    "executed_phase_seconds",
    "transition",
    "action_saturation_fraction",
    "torque_saturation_fraction",
    "trajectory",
}
_METRIC_KEYS = {
    "task_spec_sha256",
    "control_step",
    "progress_m",
    "lateral_m",
    "heading_error_signed_rad",
    "root_height_m",
    "torso_up",
    "forward_speed_m_s",
    "target_speed_m_s",
    "speed_error_m_s",
    "posture_band_error_m",
    "lateral_error_m",
    "heading_error_rad",
    "inside_posture_region",
    "posture_success",
    "finish_condition_met",
    "horizon_reached",
    "episode_success",
    "fallen",
    "failure_reasons",
    "joint_position_rmse_rad",
    "root_height_abs_error_m",
    "roll_pitch_rmse_rad",
}
_REWARD_KEYS = {
    "task_spec_sha256",
    "task_reward_recipe_sha256",
    "tracking_reward",
    "task_reward",
    "total_reward",
    "speed_component_reward",
    "posture_component_reward",
    "lateral_component_reward",
    "heading_component_reward",
    "failure_penalty",
}
_TRAJECTORY_KEYS = {
    "qpos",
    "qvel",
    "current_reference",
    "composite_raw_action",
    "contact_pairs",
    "geom_body_names",
}
_ARRAY_DTYPES = {
    "qpos": "<f8",
    "qvel": "<f8",
    "residual_action": "<f4",
    "current_reference": "<f4",
    "composite_raw_action": "<f4",
}


@dataclass(frozen=True, slots=True)
class CourseRenderInputs:
    """Hash-verified, cross-linked data required for one recorded-state render."""

    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, object]
    config_sha256: str
    task: CourseTaskSpec
    label: str
    arrays: dict[str, np.ndarray]
    rows: tuple[dict[str, object], ...]
    consumed_outputs: dict[str, str]
    support_files: dict[str, str]
    upstream_root: Path


@dataclass(frozen=True, slots=True)
class FrameAnnotation:
    state_index: int
    simulation_time_seconds: float
    root_height_m: float
    forward_speed_m_s: float
    progress_m: float
    executed_mode: str
    target_speed_m_s: float
    posture_region_active: bool
    posture_band_low_m: float
    posture_band_high_m: float
    recorded_failure_reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CourseGroundOverlay:
    """Display-only initial-heading geometry; it has no simulator collision state."""

    region_center_xyz: tuple[float, float, float]
    region_half_size_xyz: tuple[float, float, float]
    rotation_matrix: tuple[float, ...]
    entry_line: tuple[tuple[float, float, float], tuple[float, float, float]]
    exit_line: tuple[tuple[float, float, float], tuple[float, float, float]]
    finish_line: tuple[tuple[float, float, float], tuple[float, float, float]]


def _exact_object(value: object, keys: set[str], field: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise GMTAdmissionError(f"{field} fields differ from the course-render contract")
    return value


def _digest(value: object, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise GMTAdmissionError(f"{field} must be one lowercase SHA-256 digest")
    return value


def _finite(value: object, field: str) -> float:
    if type(value) not in {int, float}:
        raise GMTAdmissionError(f"{field} must be finite numeric data")
    result = float(value)
    if not math.isfinite(result):
        raise GMTAdmissionError(f"{field} must be finite numeric data")
    return result


def _direct_regular_child(parent: Path, name: str) -> Path:
    if not name or Path(name).name != name or name in {".", ".."}:
        raise GMTAdmissionError(f"unsafe course output name: {name!r}")
    path = parent / name
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise GMTAdmissionError(f"course output is unavailable: {name}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise GMTAdmissionError(f"course output must be a regular non-linked file: {name}")
    return path


def _read_npz(path: Path, expected_digest: str, steps: int) -> dict[str, np.ndarray]:
    payload = read_verified_bytes(path, expected_digest, maximum_size=8 * 1024 * 1024)
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise GMTAdmissionError("course trajectory is not a valid NPZ archive") from exc
    expected_shapes = {
        "qpos": (steps + 1, 30),
        "qvel": (steps + 1, 29),
        "residual_action": (steps, ACTION_DIM),
        "current_reference": (steps, REFERENCE_FRAME_DIM),
        "composite_raw_action": (steps, ACTION_DIM),
    }
    arrays: dict[str, np.ndarray] = {}
    with archive:
        infos = archive.infolist()
        expected_members = {f"{name}.npy" for name in expected_shapes}
        if (
            len(infos) != len(expected_members)
            or {item.filename for item in infos} != expected_members
        ):
            raise GMTAdmissionError("course trajectory member set differs")
        if len({item.filename for item in infos}) != len(infos):
            raise GMTAdmissionError("course trajectory contains duplicate members")
        for info in infos:
            if info.compress_type != zipfile.ZIP_STORED or info.file_size > 2 * 1024 * 1024:
                raise GMTAdmissionError(f"unsafe course trajectory member: {info.filename}")
            name = info.filename.removesuffix(".npy")
            member = archive.read(info)
            stream = io.BytesIO(member)
            try:
                array = np.lib.format.read_array(stream, allow_pickle=False, max_header_size=512)
            except (EOFError, ValueError) as exc:
                raise GMTAdmissionError(f"invalid numeric course member: {name}") from exc
            if stream.tell() != len(member):
                raise GMTAdmissionError(f"trailing bytes in course member: {name}")
            if (
                array.shape != expected_shapes[name]
                or array.dtype.str != _ARRAY_DTYPES[name]
                or not array.flags.c_contiguous
                or not np.isfinite(array).all()
            ):
                raise GMTAdmissionError(f"course trajectory contract differs for {name}")
            array.flags.writeable = False
            arrays[name] = array
    if np.any(np.abs(arrays["residual_action"]) > 1.0):
        raise GMTAdmissionError("recorded residual action lies outside [-1, 1]")
    return arrays


def _numeric_vector(value: object, shape: tuple[int, ...], dtype: str, field: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.shape != shape
        or not np.issubdtype(array.dtype, np.number)
        or not np.isfinite(array).all()
    ):
        raise GMTAdmissionError(f"{field} differs from its finite numeric shape contract")
    return np.ascontiguousarray(array, dtype=dtype)


def _read_rows(
    path: Path,
    expected_digest: str,
    *,
    steps: int,
    arrays: dict[str, np.ndarray],
    task_sha256: str,
    after_heading_feedback: bool = False,
) -> tuple[dict[str, object], ...]:
    payload = read_verified_bytes(path, expected_digest, maximum_size=32 * 1024 * 1024)
    if not payload.endswith(b"\n"):
        raise GMTAdmissionError("course frames JSONL must end with one newline")
    encoded_rows = payload.split(b"\n")[:-1]
    if len(encoded_rows) != steps or any(not line for line in encoded_rows):
        raise GMTAdmissionError("course frames row count differs from the trajectory")
    rows: list[dict[str, object]] = []
    for index, encoded in enumerate(encoded_rows):
        decoded = decode_json_object(encoded, source=f"course frame {index}")
        allowed_keys = (
            (frozenset(_ROW_KEYS), frozenset({*_ROW_KEYS, AFTER_HEADING_FEEDBACK_TRACE_KEY}))
            if after_heading_feedback
            else (frozenset(_ROW_KEYS),)
        )
        if frozenset(decoded) not in allowed_keys:
            raise GMTAdmissionError("course frame fields differ from the course-render contract")
        row = decoded
        metrics = _exact_object(row["metrics"], _METRIC_KEYS, "course metrics")
        _exact_object(row["reward"], _REWARD_KEYS, "course reward")
        trajectory = _exact_object(row["trajectory"], _TRAJECTORY_KEYS, "course trajectory row")
        if metrics["task_spec_sha256"] != task_sha256:
            raise GMTAdmissionError("course frame task identity differs")
        if type(metrics["control_step"]) is not int or metrics["control_step"] != index + 1:
            raise GMTAdmissionError("course frame control-step sequence differs")
        for key in (
            "progress_m",
            "root_height_m",
            "forward_speed_m_s",
            "executed_phase_seconds",
            "action_saturation_fraction",
            "torque_saturation_fraction",
        ):
            owner = metrics if key in metrics else row
            _finite(owner[key], key)
        if any(
            type(row[key]) is not str or not row[key]
            for key in ("executed_mode", "executed_behavior")
        ):
            raise GMTAdmissionError("executed mode and behavior must be nonempty strings")
        if row["transition"] is not None and type(row["transition"]) is not dict:
            raise GMTAdmissionError("course transition must be null or an object")
        failure_reasons = metrics["failure_reasons"]
        if type(failure_reasons) is not list or any(
            type(reason) is not str or not reason for reason in failure_reasons
        ):
            raise GMTAdmissionError("recorded failure reasons must be a list of nonempty strings")
        comparisons = {
            "qpos": (arrays["qpos"][index + 1], (30,), "<f8"),
            "qvel": (arrays["qvel"][index + 1], (29,), "<f8"),
            "current_reference": (
                arrays["current_reference"][index],
                (REFERENCE_FRAME_DIM,),
                "<f4",
            ),
            "composite_raw_action": (
                arrays["composite_raw_action"][index],
                (ACTION_DIM,),
                "<f4",
            ),
        }
        for name, (expected, shape, dtype) in comparisons.items():
            observed = _numeric_vector(trajectory[name], shape, dtype, f"frame {index} {name}")
            if not np.array_equal(observed, expected):
                raise GMTAdmissionError(f"course JSONL/NPZ mismatch at frame {index}: {name}")
        geom_names = trajectory["geom_body_names"]
        contacts = trajectory["contact_pairs"]
        if type(geom_names) is not list or any(type(name) is not str for name in geom_names):
            raise GMTAdmissionError("recorded geometry names differ")
        if type(contacts) is not list or len(contacts) != 20:
            raise GMTAdmissionError("recorded contacts must cover all 20 substeps")
        for substep in contacts:
            if type(substep) is not list:
                raise GMTAdmissionError("recorded substep contacts must be lists")
            for pair in substep:
                if (
                    type(pair) is not list
                    or len(pair) != 2
                    or any(
                        type(item) is not int or not 0 <= item < len(geom_names) for item in pair
                    )
                ):
                    raise GMTAdmissionError("recorded contact pair differs")
        rows.append(row)
    return tuple(rows)


def load_course_render_inputs(
    *, manifest_path: Path, manifest_sha256: str, upstream_root: Path, label: str
) -> CourseRenderInputs:
    """Admit one exact completed course trace without importing MuJoCo or a policy."""

    if label not in {"zero_residual", "final_policy"}:
        raise GMTAdmissionError("course render label must be zero_residual or final_policy")
    manifest_path = Path(manifest_path)
    upstream_root = Path(upstream_root)
    expected_manifest = _digest(manifest_sha256, "manifest_sha256")
    manifest, manifest_encoded = read_json_object(manifest_path)
    observed_manifest = hashlib.sha256(manifest_encoded).hexdigest()
    if observed_manifest != expected_manifest:
        raise GMTAdmissionError("course manifest SHA-256 differs")
    manifest = _exact_object(manifest, _MANIFEST_KEYS, "course manifest")
    if (
        manifest["schema_version"] != 1
        or manifest["artifact"] != "gmt_g1_course_development_run"
        or manifest["status"] != "completed"
    ):
        raise GMTAdmissionError("course manifest identity or completion status differs")
    claims = manifest["claims"]
    if type(claims) is not dict or claims.get("development_only") is not True:
        raise GMTAdmissionError("course renderer accepts development-only artifacts")
    outputs = manifest["outputs"]
    if type(outputs) is not dict or not outputs:
        raise GMTAdmissionError("course manifest outputs must be a nonempty object")
    for name, digest in outputs.items():
        if type(name) is not str or Path(name).name != name:
            raise GMTAdmissionError("course manifest contains an unsafe output name")
        _digest(digest, f"outputs[{name}]")
    required_names = {
        "input_config.json",
        f"{label}_frames.jsonl",
        f"{label}_trajectory.npz",
    }
    if not required_names.issubset(outputs):
        raise GMTAdmissionError("selected course render artifacts are absent")

    config_digest = _digest(manifest["input_config_sha256"], "input_config_sha256")
    if outputs["input_config.json"] != config_digest:
        raise GMTAdmissionError("course config identities differ in the manifest")
    parent = manifest_path.parent
    config_path = _direct_regular_child(parent, "input_config.json")
    config, config_encoded = read_json_object(config_path)
    if hashlib.sha256(config_encoded).hexdigest() != config_digest:
        raise GMTAdmissionError("retained course config SHA-256 differs")
    if type(config) is not dict:
        raise GMTAdmissionError("course config fields differ from the course-render contract")
    try:
        runtime = runtime_profile_from_config(config)
    except ValueError as exc:
        raise GMTAdmissionError("course runtime profile is invalid") from exc
    valid_fields = (
        {
            frozenset(_CONFIG_KEYS_WITH_RUNTIME),
            frozenset(_CONFIG_KEYS_WITH_RUNTIME_AND_TRAINER),
        }
        if runtime.config_value is not None
        else {frozenset(_CONFIG_KEYS), frozenset(_CONFIG_KEYS_WITH_TRAINER)}
    )
    if frozenset(config) not in valid_fields:
        raise GMTAdmissionError("course config fields differ from the course-render contract")
    if config["mode"] not in {"probe", "train"}:
        raise GMTAdmissionError("course config identity or mode differs")
    if not runtime.training_admitted and config["mode"] != "probe":
        raise GMTAdmissionError("course runtime profile is probe-only")
    try:
        trainer = CourseTrainerSpec.from_dict(config["trainer"]) if "trainer" in config else None
    except ValueError as exc:
        raise GMTAdmissionError("course trainer preconditioning is invalid") from exc
    if trainer is not None and config["mode"] != "train":
        raise GMTAdmissionError("course trainer preconditioning requires train mode")
    frozen_runtime = manifest["frozen_runtime"]
    expected_runtime = frozen_runtime_contract(
        runtime,
        trainer=effective_training_contract(trainer),
        residual_raw_scale=COURSE_RESIDUAL_RAW_SCALE,
    )
    if frozen_runtime != expected_runtime:
        raise GMTAdmissionError("course runtime differs from the retained config")
    if label == "final_policy" and config["mode"] != "train":
        raise GMTAdmissionError("final_policy rendering requires a retained train-mode run")
    if type(config["assets"]) is not dict or type(config["assets"].get("upstream_root")) is not str:
        raise GMTAdmissionError("course config upstream-root field differs")
    if Path(config["assets"]["upstream_root"]).resolve() != upstream_root.resolve():
        raise GMTAdmissionError("render upstream root differs from the executed course config")
    try:
        task = CourseTaskSpec.from_dict(config["task"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GMTAdmissionError("retained course task is invalid") from exc
    if task.horizon_steps > MAX_COURSE_STEPS:
        raise GMTAdmissionError("course render exceeds the forty-second development bound")
    identities = manifest["identities"]
    if type(identities) is not dict or identities.get("task") != task.sha256:
        raise GMTAdmissionError("course task identity differs from the manifest")

    report = manifest[label]
    report = _exact_object(report, _REPORT_KEYS, f"{label} report")
    steps = report["steps"]
    if type(steps) is not int or not 1 <= steps <= task.horizon_steps:
        raise GMTAdmissionError("course report step count is outside the task horizon")
    _finite(report["residual_rms"], f"{label}.residual_rms")
    if label == "zero_residual" and report["residual_rms"] != 0.0:
        raise GMTAdmissionError("zero-residual report must retain exact zero residual RMS")

    trace_name = f"{label}_trajectory.npz"
    rows_name = f"{label}_frames.jsonl"
    trace_digest = _digest(outputs[trace_name], trace_name)
    rows_digest = _digest(outputs[rows_name], rows_name)
    arrays = _read_npz(_direct_regular_child(parent, trace_name), trace_digest, steps)
    if label == "zero_residual" and np.any(arrays["residual_action"] != 0):
        raise GMTAdmissionError("zero-residual trace contains a nonzero residual action")
    rows = _read_rows(
        _direct_regular_child(parent, rows_name),
        rows_digest,
        steps=steps,
        arrays=arrays,
        task_sha256=task.sha256,
        after_heading_feedback=runtime.after_heading_reference_feedback,
    )
    if runtime.after_heading_reference_feedback:
        try:
            admitted_config = load_run_config(config_path)
            if admitted_config.encoded != config_encoded:
                raise ValueError("retained course config changed during admission")
            validate_after_heading_feedback_trace(
                config=admitted_config,
                frames=rows,
                trajectory=arrays,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise GMTAdmissionError("after-heading feedback trace is invalid") from exc
    support_files = verify_upstream_root(upstream_root)
    consumed = {
        "input_config.json": config_digest,
        rows_name: rows_digest,
        trace_name: trace_digest,
    }
    return CourseRenderInputs(
        manifest_path=manifest_path,
        manifest_sha256=expected_manifest,
        manifest=manifest,
        config_sha256=config_digest,
        task=task,
        label=label,
        arrays=arrays,
        rows=rows,
        consumed_outputs=consumed,
        support_files=support_files,
        upstream_root=upstream_root,
    )


def select_frame_indices(state_rows: int) -> np.ndarray:
    """Select about 20 frames/s while retaining endpoints and at most 800 states."""

    if type(state_rows) is not int or state_rows < 2 or state_rows > MAX_COURSE_STEPS + 1:
        raise ValueError("state_rows must be an integer in [2, 2001]")
    duration = (state_rows - 1) * CONTROL_DT_SECONDS
    desired = min(state_rows, MAX_RENDER_FRAMES, max(2, math.floor(duration * OUTPUT_FPS) + 1))
    indices = np.rint(np.linspace(0, state_rows - 1, desired)).astype(np.int64)
    if len(np.unique(indices)) != desired or indices[0] != 0 or indices[-1] != state_rows - 1:
        raise RuntimeError("frame selection failed to retain unique endpoints")
    indices.flags.writeable = False
    return indices


def course_ground_overlay(task: CourseTaskSpec, initial_qpos: object) -> CourseGroundOverlay:
    """Place display-only course markers in the retained initial-heading frame."""

    qpos = np.asarray(initial_qpos)
    if (
        qpos.shape != (30,)
        or not np.issubdtype(qpos.dtype, np.floating)
        or not np.isfinite(qpos).all()
    ):
        raise ValueError("initial_qpos must contain 30 finite floating-point values")
    frame = TaskFrame.initialize(qpos[:2], qpos[3:7])
    yaw = frame.forward_yaw_rad
    forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    lateral = np.asarray([-forward[1], forward[0]], dtype=np.float64)
    origin = np.asarray(frame.origin_xy, dtype=np.float64)

    def point(progress: float, lateral_offset: float, height: float) -> tuple[float, float, float]:
        xy = origin + progress * forward + lateral_offset * lateral
        return (float(xy[0]), float(xy[1]), height)

    def line(progress: float) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return (
            point(progress, -GROUND_OVERLAY_HALF_WIDTH_M, 0.008),
            point(progress, GROUND_OVERLAY_HALF_WIDTH_M, 0.008),
        )

    midpoint = (task.region_entry_distance_m + task.region_exit_distance_m) / 2
    rotation = (
        float(forward[0]),
        float(lateral[0]),
        0.0,
        float(forward[1]),
        float(lateral[1]),
        0.0,
        0.0,
        0.0,
        1.0,
    )
    return CourseGroundOverlay(
        region_center_xyz=point(midpoint, 0.0, 0.003),
        region_half_size_xyz=(
            (task.region_exit_distance_m - task.region_entry_distance_m) / 2,
            GROUND_OVERLAY_HALF_WIDTH_M,
            0.002,
        ),
        rotation_matrix=rotation,
        entry_line=line(task.region_entry_distance_m),
        exit_line=line(task.region_exit_distance_m),
        finish_line=line(task.finish_distance_m),
    )


def frame_annotation(inputs: CourseRenderInputs, state_index: int) -> FrameAnnotation:
    """Compute HUD values from retained state only; never trust success labels."""

    qpos, qvel = inputs.arrays["qpos"], inputs.arrays["qvel"]
    if type(state_index) is not int or not 0 <= state_index < qpos.shape[0]:
        raise ValueError("state index is outside the retained trajectory")
    frame = TaskFrame.initialize(qpos[0, :2], qpos[0, 3:7])
    projection = frame.project(qpos[state_index, :2], qpos[state_index, 3:7])
    yaw = frame.forward_yaw_rad
    speed = float(qvel[state_index, 0] * math.cos(yaw) + qvel[state_index, 1] * math.sin(yaw))
    if state_index == 0:
        executed_mode, failures = "initial", ()
    else:
        row = inputs.rows[state_index - 1]
        executed_mode = str(row["executed_mode"])
        failures = tuple(row["metrics"]["failure_reasons"])
    active = (
        inputs.task.region_entry_distance_m
        <= projection.progress_m
        < inputs.task.region_exit_distance_m
    )
    target_speed = (
        inputs.task.target_speed_inside_m_s if active else inputs.task.target_speed_outside_m_s
    )
    return FrameAnnotation(
        state_index,
        state_index * CONTROL_DT_SECONDS,
        float(qpos[state_index, 2]),
        speed,
        projection.progress_m,
        executed_mode,
        target_speed,
        active,
        inputs.task.posture_band_low_m,
        inputs.task.posture_band_high_m,
        failures,
    )


def hud_lines(inputs: CourseRenderInputs, annotation: FrameAnnotation) -> tuple[str, ...]:
    """Return compact direct labels; color is never the only task-region cue."""

    task = inputs.task
    active = "ACTIVE" if annotation.posture_region_active else "inactive"
    failures = ", ".join(annotation.recorded_failure_reasons) or "none"
    lines = [
        "DEVELOPMENT ONLY - flat-ground posture task",
        f"Blue cue=[{task.region_entry_distance_m:0.2f},{task.region_exit_distance_m:0.2f})m; "
        f"orange line=finish {task.finish_distance_m:0.2f}m",
        "Visual markers only - NOT physical obstacles",
        f"Recorded states; no policy/dynamics rerun | arm={inputs.label.replace('_', ' ')}",
        f"t={annotation.simulation_time_seconds:0.2f}s | mode={annotation.executed_mode} | "
        f"progress={annotation.progress_m:0.3f}m",
        f"speed current={annotation.forward_speed_m_s:0.3f} | "
        f"target={annotation.target_speed_m_s:0.3f} m/s",
        f"root current={annotation.root_height_m:0.3f}m | target band="
        f"[{annotation.posture_band_low_m:0.3f},{annotation.posture_band_high_m:0.3f}]m "
        f"({active})",
    ]
    lines.extend(textwrap.wrap(f"recorded failures: {failures}", width=68))
    return tuple(lines)


def _add_course_overlay(mujoco: Any, scene: Any, overlay: CourseGroundOverlay) -> None:
    """Append four non-physical visualization geoms after scene construction."""

    if scene.ngeom + 4 > scene.maxgeom:
        raise RuntimeError("MuJoCo render scene has no capacity for course overlays")
    region = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        region,
        int(mujoco.mjtGeom.mjGEOM_BOX),
        np.asarray(overlay.region_half_size_xyz, dtype=np.float64),
        np.asarray(overlay.region_center_xyz, dtype=np.float64),
        np.asarray(overlay.rotation_matrix, dtype=np.float64),
        np.asarray([0.0, 0.45, 0.70, 0.18], dtype=np.float32),
    )
    scene.ngeom += 1
    for endpoints, width, color in (
        (overlay.entry_line, 3.0, [0.0, 0.45, 0.70, 1.0]),
        (overlay.exit_line, 3.0, [0.0, 0.45, 0.70, 1.0]),
        (overlay.finish_line, 6.0, [0.90, 0.45, 0.0, 1.0]),
    ):
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            int(mujoco.mjtGeom.mjGEOM_LINE),
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64).reshape(-1),
            np.asarray(color, dtype=np.float32),
        )
        mujoco.mjv_connector(
            geom,
            int(mujoco.mjtGeom.mjGEOM_LINE),
            width,
            np.asarray(endpoints[0], dtype=np.float64),
            np.asarray(endpoints[1], dtype=np.float64),
        )
        scene.ngeom += 1


def _publish_gif(images: list[Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.", suffix=".gif", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        images[0].save(
            temporary,
            save_all=True,
            append_images=images[1:],
            duration=round(1_000 / OUTPUT_FPS),
            loop=0,
            optimize=False,
            disposal=2,
        )
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise GMTAdmissionError(f"refusing to overwrite GIF: {output}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def render_course_trace(inputs: CourseRenderInputs, output: Path) -> dict[str, object]:
    """Render admitted qpos/qvel via mj_forward; this never steps or loads a policy."""

    output = Path(output)
    if output.suffix.lower() != ".gif":
        raise ValueError("course visualization output must use the .gif suffix")
    receipt_path = output.with_suffix(".gif.manifest.json")
    if output.exists() or receipt_path.exists():
        raise GMTAdmissionError("refusing to overwrite existing course visualization")
    support_before = verify_upstream_root(inputs.upstream_root)
    if support_before != inputs.support_files:
        raise GMTAdmissionError("upstream model/support identity changed after admission")
    try:
        import mujoco
        import PIL
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError("MuJoCo and Pillow are required only for approved rendering") from exc

    model = mujoco.MjModel.from_xml_path(str(inputs.upstream_root / "assets/robots/g1/g1.xml"))
    validate_model_abi(_extract_model_abi(mujoco, model))
    support_after = verify_upstream_root(inputs.upstream_root)
    if support_after != support_before:
        raise GMTAdmissionError("upstream model/support identity changed during model loading")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=FRAME_HEIGHT, width=FRAME_WIDTH)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 4.0
    camera.azimuth = 135.0
    camera.elevation = -15.0
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    indices = select_frame_indices(inputs.arrays["qpos"].shape[0])
    overlay = course_ground_overlay(inputs.task, inputs.arrays["qpos"][0])
    images: list[Any] = []
    try:
        for state_index in indices:
            index = int(state_index)
            data.qpos[:] = inputs.arrays["qpos"][index]
            data.qvel[:] = inputs.arrays["qvel"][index]
            data.time = index * CONTROL_DT_SECONDS
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.qpos[:3]
            renderer.update_scene(data, camera=camera)
            _add_course_overlay(mujoco, renderer.scene, overlay)
            image = Image.fromarray(renderer.render()).convert("RGB")
            draw = ImageDraw.Draw(image, "RGBA")
            annotation = frame_annotation(inputs, index)
            lines = hud_lines(inputs, annotation)
            panel_height = 7 + 17 * len(lines)
            draw.rectangle((0, 0, FRAME_WIDTH, panel_height), fill=(0, 0, 0, 200))
            for line_index, line in enumerate(lines):
                draw.text((8, 3 + 17 * line_index), line, fill=(255, 255, 255, 255), font=font)
            images.append(image)
    finally:
        renderer.close()
    if not images:
        raise RuntimeError("validated course trace produced no render frames")
    _publish_gif(images, output)
    gif_sha256 = sha256_file(output)
    receipt = {
        "schema_version": 1,
        "artifact": "gmt_g1_course_recorded_state_visualization",
        "status": "completed",
        "claims": {
            "development_only": True,
            "recorded_states": True,
            "policy_rerun": False,
            "dynamics_rerun": False,
            "training_inferred_from_render": False,
            "task_success_labeled": False,
            "physical_obstacle_scene": False,
            "display_only_course_markers": True,
        },
        "inputs": {
            "course_manifest_path": inputs.manifest_path.name,
            "course_manifest_sha256": inputs.manifest_sha256,
            "selected_label": inputs.label,
            "consumed_outputs": inputs.consumed_outputs,
            "upstream_commit": GMT_UPSTREAM_COMMIT,
            "support_files": inputs.support_files,
        },
        "render": {
            "format": "gif",
            "width": FRAME_WIDTH,
            "height": FRAME_HEIGHT,
            "fps": OUTPUT_FPS,
            "frames": len(images),
            "retained_state_rows": int(inputs.arrays["qpos"].shape[0]),
            "state_operation": "mujoco_forward_on_recorded_qpos_qvel",
            "maximum_frames": MAX_RENDER_FRAMES,
            "course_overlay": {
                "coordinate_frame": "retained_initial_root_heading",
                "posture_region_progress_interval_m": [
                    inputs.task.region_entry_distance_m,
                    inputs.task.region_exit_distance_m,
                ],
                "posture_region_exit_exclusive": True,
                "finish_progress_m": inputs.task.finish_distance_m,
                "display_half_width_m_not_a_task_lateral_boundary": GROUND_OVERLAY_HALF_WIDTH_M,
                "collision_or_dynamics_effect": False,
            },
        },
        "runtime": {
            "platform": platform.platform(),
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "pillow": PIL.__version__,
        },
        "output": {"path": output.name, "sha256": gif_sha256, "size": output.stat().st_size},
    }
    receipt_sha256 = write_json_receipt(receipt_path, receipt)
    return {
        "gif_path": str(output),
        "gif_sha256": gif_sha256,
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt_sha256,
        "frames": len(images),
        "label": inputs.label,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--label", choices=("zero_residual", "final_policy"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    inputs = load_course_render_inputs(
        manifest_path=args.manifest,
        manifest_sha256=args.manifest_sha256,
        upstream_root=args.upstream_root,
        label=args.label,
    )
    result = render_course_trace(inputs, args.output)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
