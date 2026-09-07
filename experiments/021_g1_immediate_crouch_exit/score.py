"""Fail-closed Study021 scorer for one immediate crouch-exit comparison."""

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
from typing import Any

import numpy as np
import torch

import oracle_composition
from oracle_composition.adapters.gmt.composition import POSE_COLUMNS, POSE_SCALES
from oracle_composition.adapters.gmt.contracts import CONTROL_DT_SECONDS
from oracle_composition.adapters.gmt.course_proposal import apply_proposal
from oracle_composition.adapters.gmt.course_runtime import LOOP_RUNTIME, runtime_profile_from_config
from oracle_composition.adapters.gmt.course_task import TaskFrame
from oracle_composition.adapters.gmt.heading_feedback import oracle_boundary_inputs
from oracle_composition.adapters.gmt.io import write_json_receipt
from oracle_composition.harness.contract import read_json_object

_SCORER_PATH = Path(__file__).resolve()
_PROTOCOL_PATH = _SCORER_PATH.with_name("PROTOCOL.md")
_REPOSITORY_ROOT = _SCORER_PATH.parents[2]
_STUDY019_PATH = _SCORER_PATH.parents[1] / "019_g1_crouch_entry_alignment/score.py"


def _load_local(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Study021 dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


study019 = _load_local("study021_study019_helpers", _STUDY019_PATH)
study017 = study019.study017
study018_criteria = study019.study018_criteria
shared = study019.shared
development = study019.development
_feedback_parity = study019._feedback_parity
_objective = study019._objective
_replay_native_loop = study019.replay_native_loop
_reset_and_base = study019._reset_and_base
_retained = study019._retained
_verify_loop_manifest = study017.verify_loop_manifest
_verify_run = study019._verify_run

RETAINED_BINDING = study019.RETAINED_BINDING
RETAINED_SOURCE_TREE_SHA256 = study019.RETAINED_SOURCE_TREE_SHA256
RETAINED_SOURCE_FILE_COUNT = study019.RETAINED_SOURCE_FILE_COUNT
PARENT_FEEDBACK_SHA256 = study019.PARENT_FEEDBACK_SHA256
CONTROL_CONFIG_SHA256 = RETAINED_BINDING["config_sha256"]
CONTROL_CROUCH_SEGMENT_SHA256 = "187ab0f35f1532ac822d48f670047d3244d1de256cc6a0ad48a90d950dea96ca"
CANDIDATE_CROUCH_SEGMENT_SHA256 = "3584056f76f7c7582675cf365eb10cd4fb9eb5bc778cfa31edf1c456f3f54319"
PREFIX_ACTION_COUNT = 227
PREFIX_STATE_COUNT = 228
ENTRY_ACTION_INDEX = 92
GUARD_ACTION_INDEX = 227
CONTROL_RISE_ACTION_INDEX = 239
SEED = 20260906
_HEX = frozenset("0123456789abcdef")
_MODES = {"before", "inside", "rise", "after"}
_DEPENDENCY_PATHS = {
    "study019_scorer": _STUDY019_PATH,
    "study017_scorer": study019._STUDY017_PATH,
    "study018_criteria": study019._STUDY018_CRITERIA_PATH,
    "study015_scorer": study017._SHARED_PATH,
}


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: object, *, length: int, field: str) -> str:
    if type(value) is not str or len(value) != length or any(c not in _HEX for c in value):
        raise ValueError(f"{field} must be one lowercase hexadecimal digest")
    return value


def _file_pin(value: object, *, field: str) -> tuple[Path, str]:
    if type(value) is not dict or set(value) != {"path", "sha256"}:
        raise ValueError(f"{field} binding fields differ")
    raw = Path(value["path"])
    path = raw.resolve(strict=True)
    if not raw.is_absolute() or path != raw or not path.is_file():
        raise ValueError(f"{field} path must be one direct absolute file")
    return path, _digest(value["sha256"], length=64, field=f"{field} SHA-256")


def _pinned_json(path: Path, digest: str, *, source: str) -> tuple[dict, bytes]:
    value, encoded = read_json_object(path)
    if _sha256(encoded) != _digest(digest, length=64, field=f"{source} SHA-256"):
        raise ValueError(f"{source} bytes differ from their pin")
    return value, encoded


def _validate_local_source(common: dict[str, object]) -> dict[str, object]:
    root, commit, files = development._repository_sources(_REPOSITORY_ROOT)
    binding = development._source_tree_binding(files)
    package_root = Path(oracle_composition.__file__).resolve(strict=True).parents[2]
    if (
        root != _REPOSITORY_ROOT
        or package_root != root
        or commit != common["source_commit"]
        or binding
        != {
            "canonical_tree_sha256": common["source_tree_sha256"],
            "file_count": common["source_file_count"],
        }
    ):
        raise ValueError("local scoring source differs from the sealed run source")
    return {"root": str(root), "commit": commit, **binding}


def _validate_common(inputs: dict, *, extra: set[str]) -> dict[str, object]:
    fields = {
        "schema_version",
        "scorer_sha256",
        "protocol_sha256",
        "dependency_sha256",
        "source_commit",
        "source_tree_sha256",
        "source_file_count",
        "coordination_root",
    }
    if (
        type(inputs) is not dict
        or set(inputs) != fields | extra
        or type(inputs.get("schema_version")) is not int
        or inputs["schema_version"] != 1
    ):
        raise ValueError("Study021 input fields or schema differ")
    scorer = _digest(inputs["scorer_sha256"], length=64, field="scorer SHA-256")
    protocol = _digest(inputs["protocol_sha256"], length=64, field="protocol SHA-256")
    if (
        _sha256(_SCORER_PATH.read_bytes()) != scorer
        or _sha256(_PROTOCOL_PATH.read_bytes()) != protocol
    ):
        raise ValueError("Study021 scorer or protocol bytes differ from their predata pins")
    dependencies = inputs["dependency_sha256"]
    if (
        type(dependencies) is not dict
        or set(dependencies) != set(_DEPENDENCY_PATHS)
        or any(
            _digest(dependencies[name], length=64, field=f"{name} SHA-256")
            != _sha256(path.read_bytes())
            for name, path in _DEPENDENCY_PATHS.items()
        )
    ):
        raise ValueError("Study021 dependency bytes differ from their predata pins")
    count = inputs["source_file_count"]
    raw_root = Path(inputs["coordination_root"])
    root = raw_root.resolve(strict=True)
    if (
        type(count) is not int
        or count < 1
        or not raw_root.is_absolute()
        or root != raw_root
        or not root.is_dir()
    ):
        raise ValueError("source count or coordination root differs")
    common = {
        "scorer_sha256": scorer,
        "protocol_sha256": protocol,
        "dependency_sha256": dependencies,
        "source_commit": _digest(inputs["source_commit"], length=40, field="source commit"),
        "source_tree_sha256": _digest(inputs["source_tree_sha256"], length=64, field="source tree"),
        "source_file_count": count,
        "coordination_root": root,
    }
    _validate_local_source(common)
    return common


def _fresh(
    inputs: dict, common: dict[str, object], name: str, config_sha256: str
) -> tuple[Any, dict, Any]:
    return study019._fresh(inputs, common, name, config_sha256)


def verify_config(control: Any, candidate: Any) -> dict[str, object]:
    if (
        control.sha256 != CONTROL_CONFIG_SHA256
        or runtime_profile_from_config(control.raw) != LOOP_RUNTIME
        or control.raw.get("mode") != "probe"
        or control.raw.get("training_steps") != 0
        or control.raw.get("seed") != SEED
        or control.segments["crouch"].sha256 != CONTROL_CROUCH_SEGMENT_SHA256
    ):
        raise ValueError("control differs from the exact O7b loop probe")
    expected = deepcopy(control.raw)
    control_oracle_id = expected.get("oracle", {}).get("oracle_id")
    candidate_oracle_id = candidate.raw.get("oracle", {}).get("oracle_id")
    crouch = expected.get("segments", {}).get("crouch")
    if (
        type(control_oracle_id) is not str
        or type(candidate_oracle_id) is not str
        or not candidate_oracle_id
        or candidate_oracle_id == control_oracle_id
        or type(crouch) is not dict
        or crouch.get("exit_at_loop_boundary") is not True
    ):
        raise ValueError("Study021 oracle identity or control exit rule differs")
    expected["oracle"]["oracle_id"] = candidate_oracle_id
    del crouch["exit_at_loop_boundary"]
    if (
        candidate.raw != expected
        or candidate.runtime != LOOP_RUNTIME
        or candidate.segments["crouch"].exit_at_loop_boundary is not False
        or candidate.segments["crouch"].sha256 != CANDIDATE_CROUCH_SEGMENT_SHA256
        or candidate.program.sha256 == control.program.sha256
    ):
        raise ValueError("candidate changes more than oracle ID and immediate crouch exit")
    return {
        "control_oracle_id": control_oracle_id,
        "candidate_oracle_id": candidate_oracle_id,
        "control_oracle_sha256": control.program.sha256,
        "candidate_oracle_sha256": candidate.program.sha256,
        "control_crouch_segment_sha256": CONTROL_CROUCH_SEGMENT_SHA256,
        "candidate_crouch_segment_sha256": CANDIDATE_CROUCH_SEGMENT_SHA256,
        "control_exit_at_loop_boundary": True,
        "candidate_exit_at_loop_boundary": False,
    }


def _exit_dispatch(replay: dict) -> dict | None:
    value = replay.get("inside_exit_boundary_deferral")
    if value is not None and type(value) is not dict:
        raise ValueError("inside-exit dispatch is malformed")
    return value


def _reconstructed_prefix(config: Any, run: dict) -> dict[str, object]:
    """Rebuild the actor-facing numeric prefix from recorded state boundaries."""
    frames, trajectory = run["frames"], run["trajectory"]
    qpos, qvel, references = (trajectory[name] for name in ("qpos", "qvel", "current_reference"))
    if (
        len(frames) < PREFIX_ACTION_COUNT
        or len(qpos) < PREFIX_STATE_COUNT
        or len(qvel) < PREFIX_STATE_COUNT
        or len(references) < PREFIX_ACTION_COUNT
    ):
        raise ValueError("run is too short to reconstruct the exact prefix")
    frame = TaskFrame.initialize(qpos[0, :2], qpos[0, 3:7])
    oracle = study019.ComposedReference(config.program, config.segments)
    windows, currents, endpoints = [], [], []
    for index in range(PREFIX_ACTION_COUNT):
        inputs = oracle_boundary_inputs(
            task=config.task,
            frame=frame,
            qpos=qpos[index],
            qvel=qvel[index],
            control_step=index,
        )
        command = oracle.command(step=index, signals=inputs.signals, robot_pose=inputs.robot_pose)
        window = np.ascontiguousarray(command.window)
        current = np.ascontiguousarray(command.current)
        endpoint = np.ascontiguousarray(oracle.current_after_step(index + 1))
        if not np.array_equal(endpoint, references[index]):
            raise ValueError("recorded prefix target differs from native reconstruction")
        windows.append(window)
        currents.append(current)
        endpoints.append(endpoint)
    arrays = {
        "command_window": np.stack(windows),
        "command_current": np.stack(currents),
        "poststep_target": np.stack(endpoints),
    }
    return {
        "action_count": PREFIX_ACTION_COUNT,
        **{
            f"reconstructed_{name}_prefix_shape": list(value.shape)
            for name, value in arrays.items()
        },
        **{
            f"reconstructed_{name}_prefix_sha256": study017._array_sha256(value)
            for name, value in arrays.items()
        },
    }


def verify_prefix(
    control: dict,
    candidate: dict,
    control_replay: dict,
    candidate_replay: dict,
    control_config: Any,
    candidate_config: Any,
) -> dict[str, object]:
    control_dispatch, candidate_dispatch = (
        _exit_dispatch(control_replay),
        _exit_dispatch(candidate_replay),
    )
    if (
        type(control_dispatch) is not dict
        or type(candidate_dispatch) is not dict
        or control_dispatch["first_guard_satisfaction"]["action_index"] != GUARD_ACTION_INDEX
        or candidate_dispatch["first_guard_satisfaction"]["action_index"] != GUARD_ACTION_INDEX
        or control_dispatch["actual_switch"]["action_index"] != CONTROL_RISE_ACTION_INDEX
        or candidate_dispatch["actual_switch"]["action_index"] != GUARD_ACTION_INDEX
        or control_dispatch["deferred_action_count"] != 12
        or candidate_dispatch["deferred_action_count"] != 0
        or candidate_dispatch["actual_switch"].get("selected_phase_seconds") != 0.0
    ):
        raise ValueError("rise transition does not match the preregistered dispatch boundaries")
    reconstructed = (
        control_replay.get("prefix_reconstruction"),
        candidate_replay.get("prefix_reconstruction"),
    )
    if (
        any(type(value) is not dict for value in reconstructed)
        or reconstructed[0] != reconstructed[1]
    ):
        raise ValueError("reconstructed actor-facing numeric prefix differs")
    frames = (control["frames"], candidate["frames"])
    if any(len(rows) <= PREFIX_ACTION_COUNT for rows in frames):
        raise ValueError("probe ended before the expected changed command")
    for index in range(PREFIX_ACTION_COUNT):
        first, second = frames[0][index], frames[1][index]
        if index == ENTRY_ACTION_INDEX:
            first, second = deepcopy(first), deepcopy(second)
            transitions = (first.get("transition"), second.get("transition"))
            expected_hashes = (
                control_config.segments["crouch"].sha256,
                candidate_config.segments["crouch"].sha256,
            )
            if any(type(value) is not dict for value in transitions) or any(
                value.get("segment_sha256") != expected
                for value, expected in zip(transitions, expected_hashes, strict=True)
            ):
                raise ValueError("entry transition segment identity differs from its own config")
            for transition in transitions:
                transition["segment_sha256"] = "own_admitted_crouch_segment"
        if first != second:
            raise ValueError(f"frame prefix differs outside the action-92 identity: {index}")
    for name, count in (
        ("composite_raw_action", PREFIX_ACTION_COUNT),
        ("residual_action", PREFIX_ACTION_COUNT),
        ("current_reference", PREFIX_ACTION_COUNT),
        ("qpos", PREFIX_STATE_COUNT),
        ("qvel", PREFIX_STATE_COUNT),
    ):
        arrays = (control["trajectory"][name], candidate["trajectory"][name])
        if any(len(value) < count for value in arrays) or not np.array_equal(
            arrays[0][:count], arrays[1][:count]
        ):
            raise ValueError(f"numeric prefix differs before changed command 228: {name}")
    if (
        control["frames"][GUARD_ACTION_INDEX].get("executed_mode") != "inside"
        or control["frames"][GUARD_ACTION_INDEX].get("transition") is not None
        or candidate["frames"][GUARD_ACTION_INDEX].get("executed_mode") != "rise"
        or type(candidate["frames"][GUARD_ACTION_INDEX].get("transition")) is not dict
        or np.array_equal(
            control["trajectory"]["current_reference"][GUARD_ACTION_INDEX],
            candidate["trajectory"]["current_reference"][GUARD_ACTION_INDEX],
        )
    ):
        raise ValueError("action 227 is not the first changed inside-to-rise command")
    return {
        "equal_action_reference_rows": PREFIX_ACTION_COUNT,
        "equal_state_boundaries": PREFIX_STATE_COUNT,
        "allowed_frame_identity_exception": {
            "action_index": ENTRY_ACTION_INDEX,
            "field": "transition.segment_sha256",
            "control": CONTROL_CROUCH_SEGMENT_SHA256,
            "candidate": CANDIDATE_CROUCH_SEGMENT_SHA256,
        },
        "first_changed_action_index": GUARD_ACTION_INDEX,
        "first_changed_command_number_1_based": GUARD_ACTION_INDEX + 1,
        "reconstructed_numeric_prefix": reconstructed[0],
    }


def _boundary(config: Any, run: dict, index: int) -> dict[str, float | int]:
    qpos, qvel = run["trajectory"]["qpos"], run["trajectory"]["qvel"]
    frame = TaskFrame.initialize(qpos[0, :2], qpos[0, 3:7])
    projection = frame.project(qpos[index, :2], qpos[index, 3:7])
    forward = np.asarray([math.cos(frame.forward_yaw_rad), math.sin(frame.forward_yaw_rad)])
    return {
        "action_index": index,
        "command_number_1_based": index + 1,
        "progress_m": projection.progress_m,
        "lateral_m": projection.lateral_m,
        "heading_signed_rad": projection.heading_error_rad,
        "course_forward_speed_m_s": float(qvel[index, :2] @ forward),
        "robot_root_height_m": float(qpos[index, 2]),
    }


def _source_phase(config: Any, replay: dict, index: int) -> tuple[torch.Tensor, dict]:
    entry = next(
        (
            row
            for row in replay["transitions"]
            if row.get("from_state") == "before" and row.get("to_state") == "inside"
        ),
        None,
    )
    if type(entry) is not dict:
        raise ValueError("inside entry is absent before the exit measurement")

    def phase_at(action_index: int) -> torch.Tensor:
        return torch.tensor(
            entry["selected_phase_seconds"]
            + (action_index - entry["action_index"]) * CONTROL_DT_SECONDS,
            dtype=torch.float32,
        )

    raw = phase_at(index)
    previous = phase_at(index - 1)
    segment = config.segments["crouch"]
    return raw, {
        "source_clock_unwrapped_seconds": float(raw),
        "previous_source_clock_unwrapped_seconds": float(previous),
        "source_phase_reported_seconds": segment.reported_phase(raw),
        "previous_entry_loop_boundary_index": segment._entry_loop_boundary_index(previous),
        "current_entry_loop_boundary_index": segment._entry_loop_boundary_index(raw),
    }


def _dispatch_measurements(config: Any, run: dict, replay: dict) -> dict | None:
    dispatch = _exit_dispatch(replay)
    if dispatch is None:
        return None
    guard_index = dispatch["first_guard_satisfaction"]["action_index"]
    switch = dispatch.get("actual_switch")
    if type(switch) is not dict:
        guard_phase, guard_phases = _source_phase(config, replay, guard_index)
        guard_source = config.segments["crouch"].features(guard_phase.reshape(1))[0].numpy()
        return {
            "guard": {
                **_boundary(config, run, guard_index),
                **guard_phases,
                "source_target_height_m": float(guard_source[0]),
            },
            "switch": None,
            "deferred_action_count": None,
            "elapsed_deferral_seconds": None,
        }
    switch_index = switch["action_index"]
    raw_phase, phases = _source_phase(config, replay, switch_index)
    qpos, qvel = run["trajectory"]["qpos"], run["trajectory"]["qvel"]
    frame = TaskFrame.initialize(qpos[0, :2], qpos[0, 3:7])
    inputs = oracle_boundary_inputs(
        task=config.task,
        frame=frame,
        qpos=qpos[switch_index],
        qvel=qvel[switch_index],
        control_step=switch_index,
    )
    source = config.segments["crouch"].features(raw_phase.reshape(1))[0].numpy()
    selected = torch.tensor(switch["selected_phase_seconds"], dtype=torch.float32)
    destination = config.segments["rise"].features(selected.reshape(1))[0].numpy()
    destination_pose = destination[POSE_COLUMNS].astype(np.float64)
    source_pose = source[POSE_COLUMNS].astype(np.float64)
    robot_pose = np.asarray(inputs.robot_pose, dtype=np.float64)
    robot_distance = float(np.mean(((destination_pose - robot_pose) / POSE_SCALES) ** 2))
    if robot_distance != switch["normalized_pose_distance"]:
        raise ValueError("reported robot-to-destination normalized pose distance differs")
    guard_phase, guard_phases = _source_phase(config, replay, guard_index)
    guard_source = config.segments["crouch"].features(guard_phase.reshape(1))[0].numpy()
    return {
        "guard": {
            **_boundary(config, run, guard_index),
            **guard_phases,
            "source_target_height_m": float(guard_source[0]),
        },
        "switch": {
            **_boundary(config, run, switch_index),
            **phases,
            "source_target_height_m": float(source[0]),
            "selected_destination_phase_seconds": switch["selected_phase_seconds"],
            "destination_target_height_m": float(destination[0]),
            "robot_to_destination_normalized_pose_mse": robot_distance,
            "source_to_destination_normalized_pose_mse": float(
                np.mean(((destination_pose - source_pose) / POSE_SCALES) ** 2)
            ),
        },
        "deferred_action_count": dispatch["deferred_action_count"],
        "elapsed_deferral_seconds": dispatch["elapsed_deferral_seconds"],
    }


def _fall_event(frames: list[dict], indices: list[int], rise: int | None) -> dict | None:
    if not indices:
        return None
    index = indices[0]
    return {
        "action_index": index,
        "observed_boundary_index": index + 1,
        "executed_mode": frames[index]["executed_mode"],
        "observed_boundary_seconds_relative_to_rise": (
            None if rise is None else (index + 1 - rise) * CONTROL_DT_SECONDS
        ),
    }


def _contact_event(frames: list[dict], indices: list[int], rise: int | None) -> dict | None:
    if not indices:
        return None
    index = indices[0]
    return {
        "action_index": index,
        "executed_mode": frames[index]["executed_mode"],
        "action_interval_seconds_relative_to_rise": (
            None
            if rise is None
            else {
                "start": (index - rise) * CONTROL_DT_SECONDS,
                "end": (index + 1 - rise) * CONTROL_DT_SECONDS,
            }
        ),
    }


def _command_segment_identities(config: Any, run: dict, replay: dict) -> dict[str, object]:
    if replay.get("command_count") != len(run["frames"]):
        raise ValueError("native replay command count differs")
    rows = []
    crouch_indices = []
    for index, frame in enumerate(run["frames"]):
        behavior = frame.get("executed_behavior")
        segment = config.segments.get(behavior)
        if segment is None:
            raise ValueError("replayed command has no admitted segment identity")
        transition = frame.get("transition")
        if transition is not None and transition.get("segment_sha256") != segment.sha256:
            raise ValueError("transition segment identity differs from its own admitted command")
        rows.append([index, behavior, segment.sha256])
        if behavior == "crouch":
            crouch_indices.append(index)
    if not crouch_indices or any(
        rows[index][2] != config.segments["crouch"].sha256 for index in crouch_indices
    ):
        raise ValueError("crouch command identities are incomplete")
    return {
        "all_reconstructed_commands_checked": True,
        "command_identity_ledger_sha256": _sha256(json.dumps(rows, separators=(",", ":")).encode()),
        "crouch_segment_sha256": config.segments["crouch"].sha256,
        "crouch_action_count": len(crouch_indices),
        "first_crouch_action_index": crouch_indices[0],
        "last_crouch_action_index": crouch_indices[-1],
    }


def _diagnostics(run: dict, config: Any) -> dict[str, object]:
    objective = _objective(run)
    visits = study017.physical_region_visits(run["frames"], objective, config.raw["task"])
    contacts = study017.recorded_ground_contacts(run["frames"])
    replay = _replay_native_loop(config, run)
    replay["prefix_reconstruction"] = _reconstructed_prefix(config, run)
    heading = study018_criteria.heading_diagnostics(run["frames"])
    dispatch = _dispatch_measurements(config, run, replay)
    rise = (
        None
        if dispatch is None or dispatch["switch"] is None
        else dispatch["switch"]["action_index"]
    )
    after = next(
        (index for index, row in enumerate(run["frames"]) if row["executed_mode"] == "after"),
        None,
    )
    rise_rows = []
    if rise is not None:
        for index in range(rise, len(run["frames"])):
            if run["frames"][index]["executed_mode"] != "rise":
                break
            rise_rows.append(index)
    falls = [
        index
        for index, row in enumerate(run["frames"])
        if row.get("metrics", {}).get("fallen") is True
    ]
    contact_indices = contacts["nonfoot_ground_contact_action_indices"]
    region, speed, tracking = objective["region"], objective["speed"], objective["tracking"]
    measurements = {
        "inside_exit_dispatch": dispatch,
        "rise_dwell_action_count": None if rise is None else len(rise_rows),
        "rise_dwell_seconds": None if rise is None else len(rise_rows) * CONTROL_DT_SECONDS,
        "first_fall": _fall_event(run["frames"], falls, rise),
        "first_nonfoot_ground_contact": _contact_event(run["frames"], contact_indices, rise),
        "after_entry": None if after is None else _boundary(config, run, after),
        "ordered_transitions": [
            [row["from_state"], row["to_state"]] for row in replay["transitions"]
        ],
        "physical_region_minimum_root_height_m": region["minimum_root_height_m"],
        "physical_region_posture_compliant_fraction": region["posture_compliant_fraction"],
        "physical_region_sample_count": region["inside_sample_count"],
        "inside_mean_speed_target_deviation_m_s": speed["inside_mean_speed_target_deviation_m_s"],
        "overall_speed_mean_absolute_error_m_s": speed["mean_absolute_error_m_s"],
        "maximum_lateral_error_m": objective["maximum_lateral_error_m"],
        "joint_position_rmse_rad_p95": tracking["joint_position_rmse_rad_p95"],
        "roll_pitch_rmse_rad_p95": tracking["roll_pitch_rmse_rad_p95"],
        **heading,
    }
    return {
        "objective": objective,
        "visits": visits,
        "contacts": contacts,
        "replay": replay,
        "command_segment_identities": _command_segment_identities(config, run, replay),
        "measurements": measurements,
    }


def candidate_criteria(diagnostics: dict) -> tuple[dict[str, bool], dict[str, bool]]:
    objective, visits = diagnostics["objective"], diagnostics["visits"]
    contacts, measurements = diagnostics["contacts"], diagnostics["measurements"]
    dispatch = measurements["inside_exit_dispatch"]
    switch = dispatch.get("switch") if type(dispatch) is dict else None
    manipulation = {
        "guard_at_action_227": type(dispatch) is dict
        and dispatch.get("guard", {}).get("action_index") == GUARD_ACTION_INDEX,
        "first_rise_switch_at_action_227": type(switch) is dict
        and switch.get("action_index") == GUARD_ACTION_INDEX,
        "zero_action_deferral": type(dispatch) is dict
        and dispatch.get("deferred_action_count") == 0,
        "selected_rise_phase_zero": type(switch) is dict
        and switch.get("selected_destination_phase_seconds") == 0.0,
    }
    oracle, region, tracking = (
        objective["oracle_diagnostics"],
        objective["region"],
        objective["tracking"],
    )
    counts = oracle["executed_mode_counts"]
    if (
        type(counts) is not dict
        or set(counts) - _MODES
        or any(type(count) is not int or count < 0 for count in counts.values())
    ):
        raise ValueError("objective executed-mode counts differ")
    rows = {
        "manipulation": all(manipulation.values()),
        "survival": objective["duration_seconds"] == 20.0
        and objective["fall_count"] == 0
        and contacts["nonfoot_ground_contact_action_count"] == 0,
        "composition": oracle["observed_switch_count"] == 3
        and set(counts) == _MODES
        and all(counts.values())
        and measurements["ordered_transitions"]
        == [["before", "inside"], ["inside", "rise"], ["rise", "after"]],
        "exposure": region["entry_observed"] is True
        and region["exit_observed_after_entry"] is True
        and objective["finish_condition_observed"] is True
        and visits["all_visit_sample_count"] >= 25,
        "recovery_guardrail": visits["after_reentry_sample_count"] == 0,
        "tracking": tracking["joint_position_rmse_rad_p95"] <= 0.35
        and tracking["roll_pitch_rmse_rad_p95"] <= 0.25,
    }
    return rows, manipulation


def _manifest_identity(run: dict, config: Any) -> dict[str, object]:
    pins = run["pins"]
    manifest, _ = shared._pinned_json(
        pins.root / "course_run_manifest.json", pins.manifest_sha256, source="course manifest"
    )
    expected = {
        "task": config.task.sha256,
        "oracle": config.program.sha256,
        "reward": config.recipe.sha256,
        "segments": {name: segment.sha256 for name, segment in config.segments.items()},
    }
    if manifest.get("identities") != expected:
        raise ValueError("run manifest identities differ from its own admitted config")
    return {"identities": expected, "frozen_runtime": manifest.get("frozen_runtime")}


def _common_receipt(common: dict[str, object]) -> dict[str, object]:
    return {
        name: common[name]
        for name in (
            "scorer_sha256",
            "protocol_sha256",
            "dependency_sha256",
            "source_commit",
            "source_tree_sha256",
            "source_file_count",
        )
    }


def _require_control_boundary(replay: dict) -> dict:
    dispatch = _exit_dispatch(replay)
    if (
        type(dispatch) is not dict
        or dispatch["first_guard_satisfaction"]["action_index"] != GUARD_ACTION_INDEX
        or dispatch["actual_switch"]["action_index"] != CONTROL_RISE_ACTION_INDEX
        or dispatch["deferred_action_count"] != 12
        or dispatch["actual_switch"].get("selected_phase_seconds") != 0.0
    ):
        raise ValueError("fresh control does not reproduce the retained exit deferral")
    return dispatch


def score_control(inputs: dict) -> dict[str, object]:
    common = _validate_common(inputs, extra={"retained", "control"})
    retained_pins = _retained(inputs, common)
    _control_pins, control, control_config = _fresh(
        inputs, common, "control", CONTROL_CONFIG_SHA256
    )
    with tempfile.TemporaryDirectory(
        prefix="study021-retained-", dir=Path(tempfile.gettempdir()).resolve()
    ) as raw:
        retained = _verify_run(
            retained_pins,
            Path(raw) / "feedback",
            tree=RETAINED_SOURCE_TREE_SHA256,
            count=RETAINED_SOURCE_FILE_COUNT,
        )
    if retained["config_bytes"] != control["config_bytes"]:
        raise ValueError("fresh control config bytes differ from retained O7b")
    for name in shared._COMPARABLE_OUTPUTS:
        if retained["outputs"][name] != control["outputs"][name]:
            raise ValueError(f"fresh control does not byte-reproduce retained O7b: {name}")
    if retained["feedback"]["feedback"]["sha256"] != PARENT_FEEDBACK_SHA256:
        raise ValueError("retained proposal-parent feedback identity differs")
    feedback_parity = _feedback_parity(retained, control)
    loop = _verify_loop_manifest(control)
    replay = _replay_native_loop(control_config, control)
    _require_control_boundary(replay)
    _objective(control)
    manifest = _manifest_identity(control, control_config)
    return {
        "schema_version": 1,
        "study": "021",
        "artifact": "gmt_g1_study021_control_verification/v1",
        **_common_receipt(common),
        "retained_binding": inputs["retained"],
        "control_binding": inputs["control"],
        "verified_for_candidate_dispatch": True,
        "retained_output_byte_parity": {name: True for name in sorted(shared._COMPARABLE_OUTPUTS)},
        "proposal_parent_feedback_sha256": PARENT_FEEDBACK_SHA256,
        "fresh_feedback_parity": feedback_parity,
        "loop_runtime": loop,
        "manifest_identity": manifest,
        "inside_exit_boundary_deferral": replay["inside_exit_boundary_deferral"],
        "claim_scope": "fresh_control_byte_parity_only_not_candidate_evidence",
    }


def _control_receipt(binding: object, common: dict[str, object], inputs: dict) -> dict:
    path, digest = _file_pin(binding, field="control verification")
    receipt, _ = _pinned_json(path, digest, source="control verification")
    exact = all(receipt.get(name) == common[name] for name in _common_receipt(common))
    _digest(receipt.get("inputs_sha256"), length=64, field="control inputs SHA-256")
    feedback = receipt.get("fresh_feedback_parity")
    if (
        type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or receipt.get("artifact") != "gmt_g1_study021_control_verification/v1"
        or receipt.get("study") != "021"
        or not exact
        or receipt.get("retained_binding") != inputs["retained"]
        or receipt.get("control_binding") != inputs["control"]
        or receipt.get("verified_for_candidate_dispatch") is not True
        or set(receipt.get("retained_output_byte_parity", {})) != shared._COMPARABLE_OUTPUTS
        or not all(receipt["retained_output_byte_parity"].values())
        or receipt.get("proposal_parent_feedback_sha256") != PARENT_FEEDBACK_SHA256
        or type(feedback) is not dict
        or feedback.get("content_equal_except_source_manifest_sha256") is not True
        or feedback.get("retained_feedback_sha256") != PARENT_FEEDBACK_SHA256
        or feedback.get("fresh_control_feedback_sha256") == PARENT_FEEDBACK_SHA256
        or feedback.get("manifest_bound_feedback_sha256_different") is not True
        or receipt.get("inside_exit_boundary_deferral", {})
        .get("actual_switch", {})
        .get("action_index")
        != CONTROL_RISE_ACTION_INDEX
    ):
        raise ValueError("candidate scoring lacks exact successful control verification")
    return receipt


def _proposal(inputs: dict, control_config: Any, candidate_config: Any) -> dict[str, object]:
    proposal_path, proposal_sha = _file_pin(inputs["proposal"], field="proposal")
    feedback_path, feedback_sha = _file_pin(inputs["parent_feedback"], field="parent feedback")
    proposal, _ = _pinned_json(proposal_path, proposal_sha, source="proposal")
    feedback, feedback_bytes = _pinned_json(feedback_path, feedback_sha, source="parent feedback")
    hypothesis = proposal.get("hypothesis")
    if (
        proposal.get("factor") != "oracle"
        or feedback_sha != PARENT_FEEDBACK_SHA256
        or proposal.get("feedback_sha256") != feedback_sha
        or feedback.get("source_manifest_sha256") != RETAINED_BINDING["manifest_sha256"]
        or type(hypothesis) is not str
        or not all(label in hypothesis for label in ("Rationale:", "Prediction:", "Falsifier:"))
        or apply_proposal(control_config, proposal, feedback_bytes) != candidate_config.raw
    ):
        raise ValueError("candidate is not the exact labeled feedback-linked oracle proposal")
    return {
        "path": str(proposal_path),
        "sha256": proposal_sha,
        "proposal_id": proposal["proposal_id"],
        "feedback_path": str(feedback_path),
        "feedback_sha256": feedback_sha,
        "hypothesis": hypothesis,
    }


def score_pair(inputs: dict) -> dict[str, object]:
    extra = {
        "retained",
        "control",
        "candidate",
        "candidate_config_sha256",
        "control_verification",
        "proposal",
        "parent_feedback",
    }
    common = _validate_common(inputs, extra=extra)
    _retained(inputs, common)
    _control_receipt(inputs["control_verification"], common, inputs)
    candidate_hash = _digest(inputs["candidate_config_sha256"], length=64, field="candidate config")
    control_pins, control, control_config = _fresh(inputs, common, "control", CONTROL_CONFIG_SHA256)
    candidate_pins, candidate, candidate_config = _fresh(
        inputs, common, "candidate", candidate_hash
    )
    if (
        control_pins.reservation_path == candidate_pins.reservation_path
        or control_pins.reservation_sha256 == candidate_pins.reservation_sha256
        or control["resource_fixed_inputs"] != candidate["resource_fixed_inputs"]
    ):
        raise ValueError("paired runs lack separate reservations or exact fixed resources")
    config_change = verify_config(control_config, candidate_config)
    proposal = _proposal(inputs, control_config, candidate_config)
    loop = {
        "control": _verify_loop_manifest(control),
        "candidate": _verify_loop_manifest(candidate),
    }
    if loop["control"] != loop["candidate"]:
        raise ValueError("paired loop runtime manifests differ")
    manifest = {
        "control": _manifest_identity(control, control_config),
        "candidate": _manifest_identity(candidate, candidate_config),
    }
    if manifest["control"]["frozen_runtime"] != manifest["candidate"]["frozen_runtime"]:
        raise ValueError("paired frozen runtime components differ")
    reset = _reset_and_base(control, candidate, control_config, candidate_config)
    diagnostics = {
        "control": _diagnostics(control, control_config),
        "candidate": _diagnostics(candidate, candidate_config),
    }
    prefix = verify_prefix(
        control,
        candidate,
        diagnostics["control"]["replay"],
        diagnostics["candidate"]["replay"],
        control_config,
        candidate_config,
    )
    rows, manipulation = candidate_criteria(diagnostics["candidate"])
    return {
        "schema_version": 1,
        "study": "021",
        "artifact": "gmt_g1_study021_immediate_exit_pair_score/v1",
        **_common_receipt(common),
        "run_bindings": {"control": inputs["control"], "candidate": inputs["candidate"]},
        "candidate_config_sha256": candidate_hash,
        "control_verification_sha256": inputs["control_verification"]["sha256"],
        "proposal": proposal,
        "config_change": config_change,
        "fixed_resource_inputs_equal": True,
        "fixed_resource_inputs": control["resource_fixed_inputs"],
        "loop_runtime": loop,
        "manifest_identity": manifest,
        "reset_and_base_state": reset,
        "prefix_integrity": prefix,
        "candidate_criteria": rows,
        "manipulation_checks": manipulation,
        "study_screen_passed": all(rows.values()),
        "measurements": {name: value["measurements"] for name, value in diagnostics.items()},
        "native_reference_replay": {name: value["replay"] for name, value in diagnostics.items()},
        "reconstructed_command_segment_identities": {
            name: value["command_segment_identities"] for name, value in diagnostics.items()
        },
        "physical_region_visits": {name: value["visits"] for name, value in diagnostics.items()},
        "recorded_ground_contacts": {
            name: value["contacts"] for name, value in diagnostics.items()
        },
        "original_development_gate_results": {
            name: value["objective"]["development_gate_results"]
            for name, value in diagnostics.items()
        },
        "raw_objectives": {name: value["objective"] for name, value in diagnostics.items()},
        "claim_scope": "one_seed_zero_training_oracle_screen_not_causal_isolation_or_generalization",
        "claim_limits": [
            "windows_are_reconstructed_not_directly_retained",
            "no_training",
            "correlated_single_episode_pair_no_uncertainty_estimate",
        ],
    }


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("control", "pair"):
        command = commands.add_parser(name)
        command.add_argument("--inputs", type=Path, required=True)
        command.add_argument("--inputs-sha256", required=True)
        command.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    inputs, encoded = read_json_object(args.inputs)
    if _sha256(encoded) != _digest(args.inputs_sha256, length=64, field="Study021 inputs SHA-256"):
        raise ValueError("Study021 inputs differ from their pinned bytes")
    result = score_control(inputs) if args.command == "control" else score_pair(inputs)
    result["inputs_sha256"] = args.inputs_sha256
    digest = write_json_receipt(args.output, result)
    print(
        json.dumps(
            {
                "path": str(args.output),
                "sha256": digest,
                "artifact": result["artifact"],
                "verified_for_candidate_dispatch": result.get("verified_for_candidate_dispatch"),
                "study_screen_passed": result.get("study_screen_passed"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
