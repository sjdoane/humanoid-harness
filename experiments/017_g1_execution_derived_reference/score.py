"""Fail-closed Study017 scorer for the execution-derived reference probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from oracle_composition.adapters.gmt.contracts import (
    CONTROL_DT_SECONDS,
    REFERENCE_FRAME_DIM,
    REFERENCE_HORIZON,
    REFERENCE_OFFSETS,
)
from oracle_composition.adapters.gmt.course_evaluation import DEVELOPMENT_GATE_THRESHOLDS
from oracle_composition.adapters.gmt.course_runtime import (
    COURSE_RESIDUAL_RAW_SCALE,
    LOOP_RUNTIME,
    frozen_runtime_contract,
    runtime_profile_from_config,
)
from oracle_composition.adapters.gmt.course_task import (
    ALLOWED_GROUND_CONTACT_BODIES,
    TaskFrame,
)
from oracle_composition.adapters.gmt.io import write_json_receipt
from oracle_composition.adapters.gmt.training_contract import effective_training_contract
from oracle_composition.harness.contract import decode_json_object, read_json_object

_SCORER_PATH = Path(__file__).resolve()
_SHARED_PATH = _SCORER_PATH.parents[1] / "015_g1_task_aligned_after/score.py"
_SPEC = importlib.util.spec_from_file_location("study017_shared_probe_verifier", _SHARED_PATH)
shared = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = shared
_SPEC.loader.exec_module(shared)

RETAINED_MANIFEST_SHA256 = "92e2a72f743bded8b2694760c7492f8d35f0829489415a05c3afff6ec459944a"
RETAINED_RESOURCE_SHA256 = "ea00072569e159b8f04b1b7c801296b19297559b0ed91f04ec4077551cb429b8"
RETAINED_CONFIG_SHA256 = "ba45dba36ef88bdc522ed2115e46a4b3d876e00a627a089fd56ddf0de623e18a"
RETAINED_SOURCE_COMMIT = "3cb3102d5cb2e3a8603663df13ed50b5d6cad0cb"
RETAINED_SOURCE_TREE_SHA256 = "3ba7fe0ab72374e0667402848460c43915024fbcd2fe44f9ce5de791f0ab8b8e"
RETAINED_SOURCE_FILE_COUNT = 194

FRESH_SOURCE_TREE_SHA256 = "5e516691bfe5c090ff0625c6b13d67a66265a367c2b2000f6e41b61d98877738"
FRESH_SOURCE_FILE_COUNT = 196

CANDIDATE_CONFIG_SHA256 = "337b5f2c540ce364a45446431a22085c63ebf89b5284e5fa26bae606ffbb20db"
DERIVED_MOTION_NAME = "study012_inside_passage_v1"
DERIVED_ARCHIVE_SHA256 = "885e4c1b324a9b41a4d176ec5fee9e4bc634226d5ade46634111cb1c3204c259"
DERIVED_MANIFEST_SHA256 = "1c7edb409579d757ef6657378ab605e6aa013cf60b552630beb914728bec602c"
DERIVED_ARCHIVE_PATH = (
    "/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/"
    "artifacts/gmt/derived_references/study012_inside_passage_v1.npz"
)
DERIVED_MANIFEST_PATH = f"{DERIVED_ARCHIVE_PATH}.manifest.json"
DERIVED_CANDIDATE_ID = "gmt_g1_study012_executed_inside_passage/v1"
DERIVED_RESOURCE_BINDING = {
    "path": DERIVED_ARCHIVE_PATH,
    "sha256": DERIVED_ARCHIVE_SHA256,
    "size": 13_650,
    "manifest": {
        "path": DERIVED_MANIFEST_PATH,
        "sha256": DERIVED_MANIFEST_SHA256,
        "size": 10_965,
    },
    "provenance_class": "execution_derived_kinematic_candidate_not_dynamics_certificate",
    "candidate_id": DERIVED_CANDIDATE_ID,
    "training_admitted": False,
}
DERIVED_SEGMENT = {
    "motion_name": DERIVED_MOTION_NAME,
    "start_seconds": 0.0,
    "end_seconds": 2.0999999046325684,
    "entry_phase_end_seconds": 0.0,
    "boundary": "hold_last_pose_zero_velocity",
}

PREFIX_ACTION_COUNT = 92
PREFIX_STATE_COUNT = 93
DONOR_MINIMUM_HEIGHT_M = 0.504942203119977
DEPTH_TOLERANCE_M = 0.020
DONOR_ENDPOINT_PROGRESS_DELTA_M = 1.394466519355774
CONTROL_ENTRY_PROGRESS_M = 0.6549157893949207
KINEMATIC_ENDPOINT_PROGRESS_M = 2.049382308750695
REGION_EXIT_GUARD_M = 2.05


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _same_json(left: object, right: object) -> bool:
    try:
        return shared.canonical_json_bytes(left) == shared.canonical_json_bytes(right)
    except ValueError:
        return False


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    descriptor = shared.canonical_json_bytes({"dtype": array.dtype.str, "shape": list(array.shape)})
    return hashlib.sha256(descriptor + b"\0" + array.tobytes(order="C")).hexdigest()


def _sealed_source_tree() -> tuple[str, int]:
    tree = shared._digest(
        FRESH_SOURCE_TREE_SHA256,
        length=64,
        field="fresh executable source tree",
    )
    if type(FRESH_SOURCE_FILE_COUNT) is not int or FRESH_SOURCE_FILE_COUNT < 1:
        raise ValueError("fresh executable source file count must be sealed before data")
    return tree, FRESH_SOURCE_FILE_COUNT


def _pins(value: object, coordination_root: Path) -> Any:
    required = {"root", "manifest_sha256", "resource_sha256", "config_sha256", "source_commit"}
    optional = {"reservation_path", "reservation_sha256"}
    if type(value) is not dict or not required <= set(value) or set(value) - required - optional:
        raise ValueError("run pin fields differ")
    if ("reservation_path" in value) != ("reservation_sha256" in value):
        raise ValueError("reservation requires its path and canonical SHA-256")
    return shared.RunPins(
        root=Path(value["root"]).resolve(strict=True),
        manifest_sha256=value["manifest_sha256"],
        resource_sha256=value["resource_sha256"],
        config_sha256=value["config_sha256"],
        source_commit=value["source_commit"],
        reservation_path=(
            Path(value["reservation_path"]).resolve(strict=True)
            if "reservation_path" in value
            else None
        ),
        reservation_sha256=value.get("reservation_sha256"),
        coordination_root=coordination_root if "reservation_path" in value else None,
    )


def verify_config(control: dict, candidate: dict) -> None:
    if (
        runtime_profile_from_config(control) != LOOP_RUNTIME
        or LOOP_RUNTIME.observation_dim != 2_172
        or control.get("mode") != "probe"
        or control.get("training_steps") != 0
        or control.get("seed") != 20260906
    ):
        raise ValueError("control differs from the frozen 2172-D loop probe")
    expected = deepcopy(control)
    assets = expected.get("assets")
    motions = assets.get("motions") if type(assets) is dict else None
    segments = expected.get("segments")
    if (
        type(motions) is not dict
        or DERIVED_MOTION_NAME in motions
        or type(segments) is not dict
        or "crouch" not in segments
    ):
        raise ValueError("control reference bundle is malformed")
    motions[DERIVED_MOTION_NAME] = {
        "path": DERIVED_ARCHIVE_PATH,
        "sha256": DERIVED_ARCHIVE_SHA256,
        "manifest": {
            "path": DERIVED_MANIFEST_PATH,
            "sha256": DERIVED_MANIFEST_SHA256,
        },
    }
    segments["crouch"] = deepcopy(DERIVED_SEGMENT)
    if candidate != expected:
        raise ValueError("candidate changes more than the exact derived inside reference bundle")
    if runtime_profile_from_config(candidate) != LOOP_RUNTIME:
        raise ValueError("candidate must retain the frozen 2172-D loop runtime")


def verify_resource_inputs(control: dict, candidate: dict) -> dict[str, object]:
    expected_runtime = LOOP_RUNTIME.manifest_contract()
    normalized = []
    derived = None
    for name, run in (("control", control), ("candidate", candidate)):
        value = deepcopy(run.get("resource_fixed_inputs"))
        if type(value) is not dict or value.get("course_runtime") != expected_runtime:
            raise ValueError("resource receipt differs from the exact loop runtime")
        assets = value.get("gmt_assets")
        motions = assets.get("motions") if type(assets) is dict else None
        if type(motions) is not dict:
            raise ValueError("resource receipt lacks exact GMT motion bindings")
        if name == "control":
            if DERIVED_MOTION_NAME in motions:
                raise ValueError("fresh control must not consume the derived reference")
        else:
            derived = motions.pop(DERIVED_MOTION_NAME, None)
            if derived != DERIVED_RESOURCE_BINDING:
                raise ValueError("candidate resource receipt does not bind the derived archive")
        normalized.append(value)
    if normalized[0] != normalized[1]:
        raise ValueError("fresh arms differ beyond the declared derived reference resource")
    return {
        "course_runtime": expected_runtime,
        "observation_dim": LOOP_RUNTIME.observation_dim,
        "derived_reference": derived,
    }


def verify_loop_manifest(run: dict) -> dict[str, object]:
    pins = run["pins"]
    manifest, _ = shared._pinned_json(
        pins.root / "course_run_manifest.json",
        pins.manifest_sha256,
        source="course manifest",
    )
    expected = frozen_runtime_contract(
        LOOP_RUNTIME,
        trainer=effective_training_contract(None),
        residual_raw_scale=COURSE_RESIDUAL_RAW_SCALE,
    )
    if manifest.get("frozen_runtime") != expected or expected.get("observation_dim") != 2_172:
        raise ValueError("run manifest differs from the frozen 2172-D loop runtime")
    evaluation = decode_json_object(
        run["outputs"]["zero_residual_evaluation.json"],
        source="zero-residual evaluation",
    )
    reset = evaluation.get("reset")
    if (
        type(reset) is not dict
        or reset.get("runtime_id") != LOOP_RUNTIME.gym_runtime_id
        or reset.get("course_runtime") != LOOP_RUNTIME.manifest_contract()
    ):
        raise ValueError("run reset differs from the frozen loop runtime")
    return expected


def _first_mode(frames: list[dict], mode: str) -> int | None:
    return next(
        (index for index, row in enumerate(frames) if row.get("executed_mode") == mode), None
    )


def verify_prefix(control: dict, candidate: dict) -> dict[str, object]:
    control_frames = control["frames"]
    candidate_frames = candidate["frames"]
    if (
        _first_mode(control_frames, "inside") != PREFIX_ACTION_COUNT
        or _first_mode(candidate_frames, "inside") != PREFIX_ACTION_COUNT
    ):
        raise ValueError("first inside command is not the preregistered action index 92")
    if len(control_frames) <= PREFIX_ACTION_COUNT or len(candidate_frames) <= PREFIX_ACTION_COUNT:
        raise ValueError("probe ended before the first inside target")
    for index in range(PREFIX_ACTION_COUNT):
        if control_frames[index] != candidate_frames[index]:
            raise ValueError(f"complete trace prefix differs before inside command: {index}")
    for field, count in (
        ("composite_raw_action", PREFIX_ACTION_COUNT),
        ("qpos", PREFIX_STATE_COUNT),
        ("qvel", PREFIX_STATE_COUNT),
    ):
        if not np.array_equal(
            control["trajectory"][field][:count], candidate["trajectory"][field][:count]
        ):
            raise ValueError(f"numeric prefix differs before derived inside reference: {field}")
    first_control = control["trajectory"]["current_reference"][PREFIX_ACTION_COUNT]
    first_candidate = candidate["trajectory"]["current_reference"][PREFIX_ACTION_COUNT]
    if np.array_equal(first_control, first_candidate):
        raise ValueError("derived reference must change the first inside poststep target")
    return {
        "equal_complete_frame_count": PREFIX_ACTION_COUNT,
        "equal_action_count": PREFIX_ACTION_COUNT,
        "equal_qpos_qvel_boundary_count": PREFIX_STATE_COUNT,
        "first_inside_action_index": PREFIX_ACTION_COUNT,
        "first_inside_poststep_target_changed": True,
        "control_first_inside_target_sha256": _array_sha256(first_control),
        "candidate_first_inside_target_sha256": _array_sha256(first_candidate),
    }


def _contiguous_spans(indices: list[int]) -> list[dict[str, int]]:
    if not indices:
        return []
    spans = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            spans.append({"first_action_index": start, "last_action_index": previous})
            start = index
        previous = index
    spans.append({"first_action_index": start, "last_action_index": previous})
    return spans


def physical_region_visits(frames: list[dict], objective: dict, task: dict) -> dict[str, object]:
    entry = task.get("region_entry_distance_m")
    exit_ = task.get("region_exit_distance_m")
    if type(entry) is not float or type(exit_) is not float or not 0.0 < entry < exit_:
        raise ValueError("physical task region differs")
    progress = []
    for row in frames:
        metrics = row.get("metrics")
        value = metrics.get("progress_m") if type(metrics) is dict else None
        inside_flag = metrics.get("inside_posture_region") if type(metrics) is dict else None
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("frame progress must be finite")
        expected_inside = entry <= value < exit_
        if inside_flag is not expected_inside:
            raise ValueError("frame physical-region flag differs from progress")
        progress.append(value)
    inside = [entry <= value < exit_ for value in progress]
    all_indices = [index for index, value in enumerate(inside) if value]
    first_entry = all_indices[0] if all_indices else None
    first_exit = (
        next(
            (index for index in range(first_entry + 1, len(frames)) if progress[index] >= exit_),
            None,
        )
        if first_entry is not None
        else None
    )
    first_indices = (
        [index for index in all_indices if first_exit is None or index < first_exit]
        if first_entry is not None
        else []
    )
    later_indices = [
        index for index in all_indices if first_exit is not None and index >= first_exit
    ]
    after_indices = [
        index for index in all_indices if frames[index].get("executed_mode") == "after"
    ]
    region = objective.get("region")
    if (
        type(region) is not dict
        or region.get("inside_sample_count") != len(all_indices)
        or region.get("entry_observed") is not bool(all_indices)
        or region.get("first_entry_step") != (first_entry + 1 if first_entry is not None else None)
        or region.get("exit_observed_after_entry") is not (first_exit is not None)
        or region.get("first_exit_step") != (first_exit + 1 if first_exit is not None else None)
    ):
        raise ValueError("objective physical-region summary differs from all retained visits")
    return {
        "entry_distance_m": entry,
        "exit_distance_m": exit_,
        "first_entry_action_index": first_entry,
        "first_forward_exit_action_index": first_exit,
        "first_visit_action_indices": first_indices,
        "later_visit_action_indices": later_indices,
        "all_visit_action_indices": all_indices,
        "after_reentry_action_indices": after_indices,
        "first_visit_spans": _contiguous_spans(first_indices),
        "later_visit_spans": _contiguous_spans(later_indices),
        "first_visit_sample_count": len(first_indices),
        "later_visit_sample_count": len(later_indices),
        "all_visit_sample_count": len(all_indices),
        "after_reentry_sample_count": len(after_indices),
    }


def recorded_ground_contacts(frames: list[dict]) -> dict[str, object]:
    canonical_names = None
    nonfoot_actions = []
    nonfoot_substeps = 0
    foot_substeps = 0
    nonfoot_bodies: set[str] = set()
    ground_names = {"world", "body:0"}
    for action_index, row in enumerate(frames):
        retained = row.get("trajectory")
        if type(retained) is not dict:
            raise ValueError("frame lacks raw contact trajectory")
        names = retained.get("geom_body_names")
        contacts = retained.get("contact_pairs")
        if (
            type(names) is not list
            or not names
            or any(type(name) is not str or not name for name in names)
            or type(contacts) is not list
            or len(contacts) != 20
        ):
            raise ValueError("raw contact trace fields differ")
        if canonical_names is None:
            canonical_names = names
        elif names != canonical_names:
            raise ValueError("geom/body identity changed within one probe")
        action_nonfoot = False
        for substep in contacts:
            if type(substep) is not list:
                raise ValueError("contact substep must be a list")
            checked = []
            for pair in substep:
                if (
                    type(pair) is not list
                    or len(pair) != 2
                    or any(type(item) is not int or not 0 <= item < len(names) for item in pair)
                    or pair[0] > pair[1]
                ):
                    raise ValueError("contact geom pair differs")
                checked.append(tuple(pair))
            if checked != sorted(set(checked)):
                raise ValueError("contact geom pairs must be sorted and unique")
            ground_bodies = set()
            for first, second in checked:
                if names[first] in ground_names:
                    ground_bodies.add(names[second])
                if names[second] in ground_names:
                    ground_bodies.add(names[first])
            ground_bodies -= ground_names
            if ground_bodies & set(ALLOWED_GROUND_CONTACT_BODIES):
                foot_substeps += 1
            observed_nonfoot = ground_bodies - set(ALLOWED_GROUND_CONTACT_BODIES)
            if observed_nonfoot:
                action_nonfoot = True
                nonfoot_substeps += 1
                nonfoot_bodies.update(observed_nonfoot)
        if action_nonfoot:
            nonfoot_actions.append(action_index)
        reasons = row.get("metrics", {}).get("failure_reasons")
        if (
            type(reasons) is not list
            or ("non_foot_ground_contact" in reasons) is not action_nonfoot
        ):
            raise ValueError("non-foot contact reason differs from the raw substep trace")
    return {
        "raw_contact_action_count": len(frames),
        "raw_contact_substep_count": len(frames) * 20,
        "foot_ground_contact_substep_count": foot_substeps,
        "nonfoot_ground_contact_action_indices": nonfoot_actions,
        "nonfoot_ground_contact_action_count": len(nonfoot_actions),
        "nonfoot_ground_contact_substep_count": nonfoot_substeps,
        "nonfoot_ground_contact_bodies": sorted(nonfoot_bodies),
    }


def _body_local_velocity(orientation_wxyz: object, world_velocity: object) -> np.ndarray:
    quaternion = np.asarray(orientation_wxyz, dtype=np.float64)
    velocity = np.asarray(world_velocity, dtype=np.float64)
    if (
        quaternion.shape != (4,)
        or velocity.shape != (3,)
        or not np.isfinite(quaternion).all()
        or not np.isfinite(velocity).all()
        or not math.isclose(float(np.linalg.norm(quaternion)), 1.0, abs_tol=1e-6)
    ):
        raise ValueError("body-local velocity requires a finite unit WXYZ pose and XYZ velocity")
    w, x, y, z = quaternion
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return rotation.T @ velocity


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "minimum": None, "maximum": None, "max_abs": None}
    return {
        "count": len(values),
        "mean": fmean(values),
        "minimum": min(values),
        "maximum": max(values),
        "max_abs": max(abs(value) for value in values),
    }


def replay_composed_reference(config: Any, run: dict) -> dict[str, object]:
    """Rebuild every command from retained raw boundaries; do not rerun plant or actor."""

    import torch

    from oracle_composition.adapters.gmt.composition import ComposedReference
    from oracle_composition.adapters.gmt.heading_feedback import (
        AFTER_HEADING_FEEDBACK_TRACE_KEY,
        oracle_boundary_inputs,
    )

    if config.runtime != LOOP_RUNTIME or config.runtime.observation_dim != 2_172:
        raise ValueError("reference replay requires the frozen 2172-D loop runtime")
    frames = run["frames"]
    trajectory = run["trajectory"]
    qpos = trajectory.get("qpos")
    qvel = trajectory.get("qvel")
    references = trajectory.get("current_reference")
    if (
        type(qpos) is not np.ndarray
        or qpos.shape != (len(frames) + 1, 30)
        or qpos.dtype.str != "<f8"
        or type(qvel) is not np.ndarray
        or qvel.shape != (len(frames) + 1, 29)
        or qvel.dtype.str != "<f8"
        or type(references) is not np.ndarray
        or references.shape != (len(frames), REFERENCE_FRAME_DIM)
        or references.dtype.str != "<f4"
        or not all(value.flags.c_contiguous for value in (qpos, qvel, references))
        or not all(np.isfinite(value).all() for value in (qpos, qvel, references))
    ):
        raise ValueError("reference replay raw trajectory contract differs")
    task_frame = TaskFrame.initialize(qpos[0, :2], qpos[0, 3:7])
    oracle = ComposedReference(config.program, config.segments)
    offsets = torch.tensor(REFERENCE_OFFSETS, dtype=torch.float32) * CONTROL_DT_SECONDS
    windows = []
    endpoints = []
    transitions = []
    inside_rows = []
    first_lookahead_hold = None
    first_endpoint_hold = None
    lookahead_exposure_count = 0
    endpoint_exposure_count = 0
    endpoint_window_mismatches = 0
    inside_exit = None
    previous_inside = None
    for index, row in enumerate(frames):
        previous_state = oracle.machine.state
        previous_behavior = oracle.machine.behavior
        predecision_phase = oracle._phase(index).clone()
        inputs = oracle_boundary_inputs(
            task=config.task,
            frame=task_frame,
            qpos=qpos[index],
            qvel=qvel[index],
            control_step=index,
        )
        command = oracle.command(step=index, signals=inputs.signals, robot_pose=inputs.robot_pose)
        raw_phase = oracle._phase(index).clone()
        if (
            row.get("executed_mode") != command.state
            or row.get("executed_behavior") != command.behavior
            or row.get("executed_phase_seconds") != command.phase_seconds
            or not _same_json(row.get("transition"), command.transition)
            or AFTER_HEADING_FEEDBACK_TRACE_KEY in row
        ):
            raise ValueError("retained oracle phase or transition differs from raw-state replay")
        window = np.ascontiguousarray(command.window)
        current = np.ascontiguousarray(command.current)
        if (
            window.shape != (REFERENCE_HORIZON, REFERENCE_FRAME_DIM)
            or window.dtype.str != "<f4"
            or current.shape != (REFERENCE_FRAME_DIM,)
            or current.dtype.str != "<f4"
        ):
            raise ValueError("reconstructed composed-reference command differs")
        endpoint_phase = oracle._phase(index + 1).clone()
        endpoint = np.ascontiguousarray(oracle.current_after_step(index + 1))
        if not np.array_equal(endpoint, references[index]):
            raise ValueError("retained poststep reference differs from full endpoint replay")
        windows.append(window)
        endpoints.append(endpoint)
        endpoint_window_mismatches += not np.array_equal(window[0], endpoint)
        if command.transition is not None:
            record = {"action_index": index, **dict(command.transition)}
            if previous_state == "inside":
                previous_segment = config.segments[previous_behavior]
                record["source_raw_phase_seconds"] = float(predecision_phase)
                record["source_reported_phase_seconds"] = previous_segment.reported_phase(
                    predecision_phase
                )
                inside_exit = {
                    "transition_action_index": index,
                    "pre_action_progress_m": inputs.projection.progress_m,
                    "inside_raw_phase_seconds": float(predecision_phase),
                    "inside_reported_phase_seconds": previous_segment.reported_phase(
                        predecision_phase
                    ),
                    "last_inside_action_index": (
                        previous_inside["action_index"] if previous_inside is not None else None
                    ),
                    "last_inside_reported_phase_seconds": (
                        previous_inside["executed_phase_seconds"]
                        if previous_inside is not None
                        else None
                    ),
                }
            transitions.append(record)
        if command.state != "inside":
            continue
        segment = config.segments[command.behavior]
        if segment.boundary != "hold_last_pose_zero_velocity":
            raise ValueError("Study017 inside command must use terminal-hold semantics")
        terminal = np.ascontiguousarray(
            segment.features(torch.tensor([segment.duration], dtype=torch.float32))[0].numpy()
        )
        hold_mask = (raw_phase + offsets) >= segment.duration
        mask = hold_mask.numpy()
        if np.any(mask) and not np.array_equal(
            window[mask], np.repeat(terminal.reshape(1, -1), int(np.count_nonzero(mask)), axis=0)
        ):
            raise ValueError("terminal lookahead rows differ from the held endpoint")
        has_lookahead_hold = bool(np.any(mask))
        endpoint_hold = bool(endpoint_phase >= segment.duration)
        if endpoint_hold and not np.array_equal(endpoint, terminal):
            raise ValueError("terminal poststep target differs from the held endpoint")
        lookahead_exposure_count += int(has_lookahead_hold)
        endpoint_exposure_count += int(endpoint_hold)
        if has_lookahead_hold and first_lookahead_hold is None:
            first_row = int(np.flatnonzero(mask)[0])
            first_lookahead_hold = {
                "action_index": index,
                "executed_phase_seconds": command.phase_seconds,
                "raw_phase_seconds": float(raw_phase),
                "first_terminal_window_row": first_row,
                "first_terminal_lookahead_offset_seconds": float(offsets[first_row]),
                "terminal_window_row_count": int(np.count_nonzero(mask)),
            }
        if endpoint_hold and first_endpoint_hold is None:
            first_endpoint_hold = {
                "action_index": index,
                "poststep_control_step": index + 1,
                "executed_phase_seconds": command.phase_seconds,
                "raw_poststep_phase_seconds": float(endpoint_phase),
                "poststep_progress_m": row["metrics"]["progress_m"],
                "held_target_height_m": float(endpoint[0]),
            }
        body_velocity = _body_local_velocity(qpos[index + 1, 3:7], qvel[index + 1, :3])
        metrics = row.get("metrics")
        if type(metrics) is not dict or any(
            type(metrics.get(name)) is not float or not math.isfinite(metrics[name])
            for name in ("progress_m", "forward_speed_m_s")
        ):
            raise ValueError("inside speed profile lacks finite physical metrics")
        speed_row = {
            "action_index": index,
            "executed_phase_seconds": command.phase_seconds,
            "raw_phase_seconds": float(raw_phase),
            "physical_progress_m": metrics["progress_m"],
            "task_frame_forward_speed_m_s": metrics["forward_speed_m_s"],
            "candidate_body_local_forward_velocity_m_s": float(body_velocity[0]),
            "candidate_body_local_lateral_velocity_m_s": float(body_velocity[1]),
            "derived_reference_command_forward_velocity_m_s": float(current[3]),
            "derived_reference_command_lateral_velocity_m_s": float(current[4]),
            "issued_poststep_reference_forward_velocity_m_s": float(endpoint[3]),
            "issued_poststep_reference_lateral_velocity_m_s": float(endpoint[4]),
            "lookahead_terminal_hold_exposed": has_lookahead_hold,
            "poststep_endpoint_held": endpoint_hold,
        }
        inside_rows.append(speed_row)
        previous_inside = speed_row
    window_array = np.ascontiguousarray(np.stack(windows), dtype="<f4")
    endpoint_array = np.ascontiguousarray(np.stack(endpoints), dtype="<f4")
    summary_fields = (
        "task_frame_forward_speed_m_s",
        "candidate_body_local_forward_velocity_m_s",
        "candidate_body_local_lateral_velocity_m_s",
        "derived_reference_command_forward_velocity_m_s",
        "derived_reference_command_lateral_velocity_m_s",
        "issued_poststep_reference_forward_velocity_m_s",
        "issued_poststep_reference_lateral_velocity_m_s",
    )
    return {
        "scope": "raw_boundary_state_to_composed_reference_not_plant_or_actor_rerun",
        "all_retained_actions_replayed": True,
        "command_count": len(frames),
        "inside_command_count": len(inside_rows),
        "raw_qpos_sha256": _array_sha256(qpos),
        "raw_qvel_sha256": _array_sha256(qvel),
        "raw_composite_action_sha256": _array_sha256(trajectory["composite_raw_action"]),
        "reconstructed_window_shape": list(window_array.shape),
        "reconstructed_window_sha256": _array_sha256(window_array),
        "verified_poststep_endpoint_sha256": _array_sha256(endpoint_array),
        "window_first_row_poststep_endpoint_mismatch_count": endpoint_window_mismatches,
        "transitions": transitions,
        "inside_entry": (
            {
                "action_index": inside_rows[0]["action_index"],
                "selected_phase_seconds": frames[inside_rows[0]["action_index"]]["transition"][
                    "selected_phase_seconds"
                ],
            }
            if inside_rows
            else None
        ),
        "inside_exit": inside_exit,
        "terminal_hold": {
            "first_lookahead_exposure": first_lookahead_hold,
            "first_poststep_endpoint_hold": first_endpoint_hold,
            "lookahead_exposed_inside_action_count": lookahead_exposure_count,
            "endpoint_held_inside_action_count": endpoint_exposure_count,
            "endpoint_hold_exposure_seconds": endpoint_exposure_count * CONTROL_DT_SECONDS,
            "interpretation": (
                "terminal_mid_stride_crouch_pose_not_balanced_or_stationary_reference"
            ),
        },
        "inside_speed_profile": inside_rows,
        "inside_speed_summary": {
            field: _summary([row[field] for row in inside_rows]) for field in summary_fields
        },
        "lateral_velocity_scope": (
            "candidate_and_derived_reference_body_local_lateral_velocity_descriptive_only"
        ),
    }


def feasibility_screen(
    objective: dict,
    visits: dict[str, object],
    contacts: dict[str, object],
) -> dict[str, bool]:
    original = objective.get("development_gate_results")
    thresholds = objective.get("development_gate_thresholds")
    tracking = objective.get("tracking")
    region = objective.get("region")
    diagnostics = objective.get("oracle_diagnostics")
    if (
        type(original) is not dict
        or set(original) != shared._DEVELOPMENT_GATES
        or thresholds != DEVELOPMENT_GATE_THRESHOLDS
        or type(tracking) is not dict
        or type(region) is not dict
        or type(diagnostics) is not dict
    ):
        raise ValueError("objective evaluator or original eleven gates differ")
    minimum_height = region.get("minimum_root_height_m")
    joint_p95 = tracking.get("joint_position_rmse_rad_p95")
    roll_pitch_p95 = tracking.get("roll_pitch_rmse_rad_p95")
    executed = diagnostics.get("executed_mode_counts")
    return {
        "full_20s_without_fall": objective.get("duration_seconds") == 20.0
        and objective.get("fall_count") == 0,
        "no_recorded_nonfoot_ground_contact": contacts["nonfoot_ground_contact_action_count"] == 0,
        "three_switches": diagnostics.get("observed_switch_count") == 3,
        "all_four_modes_executed": type(executed) is dict
        and set(executed) == {"before", "inside", "rise", "after"}
        and all(type(count) is int and count > 0 for count in executed.values()),
        "physical_region_entry_and_exit_observed": region.get("entry_observed") is True
        and region.get("exit_observed_after_entry") is True,
        "finish_observed_at_3p5m": objective.get("finish_condition_observed") is True
        and type(objective.get("maximum_progress_m")) is float
        and objective["maximum_progress_m"] >= 3.5,
        "no_physical_region_reentry_in_after": visits["after_reentry_sample_count"] == 0,
        "minimum_25_inside_region_samples": visits["all_visit_sample_count"] >= 25,
        "inside_minimum_height_within_0p020m_of_donor": type(minimum_height) is float
        and abs(minimum_height - DONOR_MINIMUM_HEIGHT_M) <= DEPTH_TOLERANCE_M,
        "joint_tracking_p95_at_most_0p35rad": type(joint_p95) is float and joint_p95 <= 0.35,
        "roll_pitch_tracking_p95_at_most_0p25rad": type(roll_pitch_p95) is float
        and roll_pitch_p95 <= 0.25,
    }


def _loaded_config(run: dict) -> Any:
    from oracle_composition.adapters.gmt.course_config import load_run_config

    loaded = load_run_config(run["pins"].root / "input_config.json")
    if loaded.raw != run["config"] or loaded.sha256 != run["pins"].config_sha256:
        raise ValueError("loaded candidate config differs from retained pinned bytes")
    return loaded


def score_study(inputs: dict, *, run_verifier: Any = shared._verify_run) -> dict[str, object]:
    expected_fields = {
        "schema_version",
        "scorer_sha256",
        "baseline",
        "control",
        "candidate",
        "coordination_root",
    }
    if (
        type(inputs) is not dict
        or set(inputs) != expected_fields
        or type(inputs.get("schema_version")) is not int
        or inputs["schema_version"] != 1
    ):
        raise ValueError("Study017 input fields or schema differ")
    scorer_sha256 = shared._digest(
        inputs["scorer_sha256"], length=64, field="predata scorer SHA-256"
    )
    if _sha256(_SCORER_PATH.read_bytes()) != scorer_sha256:
        raise ValueError("Study017 scorer bytes differ from their predata pin")
    source_tree, source_count = _sealed_source_tree()
    coordination_root = Path(inputs["coordination_root"]).resolve(strict=True)
    pins = {
        name: _pins(inputs[name], coordination_root)
        for name in ("baseline", "control", "candidate")
    }
    baseline, control, candidate = (pins[name] for name in ("baseline", "control", "candidate"))
    if (
        baseline.manifest_sha256 != RETAINED_MANIFEST_SHA256
        or baseline.resource_sha256 != RETAINED_RESOURCE_SHA256
        or baseline.config_sha256 != RETAINED_CONFIG_SHA256
        or baseline.source_commit != RETAINED_SOURCE_COMMIT
        or baseline.reservation_path is not None
        or control.config_sha256 != RETAINED_CONFIG_SHA256
        or candidate.config_sha256 != CANDIDATE_CONFIG_SHA256
        or candidate.source_commit != control.source_commit
        or control.reservation_path is None
        or candidate.reservation_path is None
        or len({baseline.root, control.root, candidate.root}) != 3
    ):
        raise ValueError("Study017 retained baseline or fresh-pair pins differ")
    source_commit = shared._digest(
        control.source_commit, length=40, field="fresh common source commit"
    )
    with tempfile.TemporaryDirectory(prefix="study017-feedback-") as temporary:
        feedback_root = Path(temporary).resolve(strict=True)
        runs = {
            name: run_verifier(
                pins[name],
                feedback_root / name,
                expected_source_tree_sha256=(
                    RETAINED_SOURCE_TREE_SHA256 if name == "baseline" else source_tree
                ),
                expected_source_file_count=(
                    RETAINED_SOURCE_FILE_COUNT if name == "baseline" else source_count
                ),
            )
            for name in ("baseline", "control", "candidate")
        }
    baseline_run, control_run, candidate_run = (
        runs[name] for name in ("baseline", "control", "candidate")
    )
    if baseline_run["config_bytes"] != control_run["config_bytes"] or any(
        baseline_run["outputs"][name] != control_run["outputs"][name]
        for name in shared._COMPARABLE_OUTPUTS
    ):
        raise ValueError("fresh O7b control must byte-reproduce all three retained outputs")
    if any(run["source_tree_sha256"] != source_tree for run in (control_run, candidate_run)):
        raise ValueError("fresh arm executable source tree differs")
    verify_config(control_run["config"], candidate_run["config"])
    resource_contract = verify_resource_inputs(control_run, candidate_run)
    frozen_runtime = {name: verify_loop_manifest(run) for name, run in runs.items()}
    if len({_sha256(shared.canonical_json_bytes(value)) for value in frozen_runtime.values()}) != 1:
        raise ValueError("run frozen runtime manifests differ")
    prefix = verify_prefix(control_run, candidate_run)
    loaded = _loaded_config(candidate_run)
    replay = replay_composed_reference(loaded, candidate_run)
    visits = physical_region_visits(
        candidate_run["frames"],
        candidate_run["objective"],
        candidate_run["config"]["task"],
    )
    contacts = recorded_ground_contacts(candidate_run["frames"])
    screen = feasibility_screen(candidate_run["objective"], visits, contacts)
    return {
        "schema_version": 1,
        "study": "017",
        "artifact": "gmt_g1_execution_derived_reference_development_score/v1",
        "comparison": "retained_study015_candidate_vs_fresh_o7b_control_vs_derived_inside",
        "source_commit": source_commit,
        "fresh_source_tree_sha256": source_tree,
        "fresh_source_file_count": source_count,
        "candidate_config_sha256": CANDIDATE_CONFIG_SHA256,
        "scorer_sha256": scorer_sha256,
        "shared_verifier_sha256": _sha256(_SHARED_PATH.read_bytes()),
        "fresh_control_byte_reproduced_three_outputs": True,
        "runtime_contract": resource_contract,
        "prefix": prefix,
        "reference_replay": replay,
        "physical_region_visits": visits,
        "recorded_ground_contacts": contacts,
        "feasibility_screen": screen,
        "feasibility_screen_passed": all(screen.values()),
        "original_development_gate_results": candidate_run["objective"]["development_gate_results"],
        "original_objective": candidate_run["objective"],
        "kinematic_exit_margin_diagnostic": {
            "donor_boundary_half_open": [92, 198],
            "last_included_boundary": 197,
            "donor_endpoint_progress_delta_m": DONOR_ENDPOINT_PROGRESS_DELTA_M,
            "retained_control_entry_progress_m": CONTROL_ENTRY_PROGRESS_M,
            "kinematic_endpoint_progress_m": KINEMATIC_ENDPOINT_PROGRESS_M,
            "unchanged_region_exit_guard_m": REGION_EXIT_GUARD_M,
            "shortfall_m": REGION_EXIT_GUARD_M - KINEMATIC_ENDPOINT_PROGRESS_M,
            "interpretation": "kinematic_margin_risk_not_plant_impossibility_proof",
        },
        "run_bindings": {
            name: {
                "manifest_sha256": record.manifest_sha256,
                "resource_sha256": record.resource_sha256,
                "config_sha256": record.config_sha256,
                "reservation_sha256": record.reservation_sha256,
                "feedback": runs[name]["feedback"],
            }
            for name, record in pins.items()
        },
        "claim_scope": "one_seed_zero_residual_exact_bundle_feasibility_not_task_success",
        "claim_limits": [
            "original_eleven_development_gates_are_unchanged_and_reported_separately",
            "all_physical_region_visits_are_authoritative",
            "terminal_hold_is_mid_stride_and_not_a_balanced_stationary_reference",
            "stall_or_fall_cannot_isolate_crouch_fidelity_from_anticipation_and_handover",
            "candidate_and_derived_reference_body_local_lateral_velocity_are_descriptive_only",
            "reconstructed_windows_were_not_directly_retained_by_the_probe",
            "no_training_heldout_generalization_or_broad_dynamics_certificate",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs, encoded = read_json_object(args.inputs)
    if _sha256(encoded) != args.inputs_sha256:
        raise ValueError("Study017 inputs differ from their pinned bytes")
    result = score_study(inputs)
    result["inputs_sha256"] = args.inputs_sha256
    digest = write_json_receipt(args.output, result)
    print(
        json.dumps(
            {
                "path": str(args.output),
                "sha256": digest,
                "feasibility_screen": result["feasibility_screen"],
                "feasibility_screen_passed": result["feasibility_screen_passed"],
                "terminal_hold": result["reference_replay"]["terminal_hold"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
