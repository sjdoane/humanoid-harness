"""Fail-closed Study019 scorer for one zero-training oracle comparison."""

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

import oracle_composition
from oracle_composition.adapters.gmt.composition import ComposedReference
from oracle_composition.adapters.gmt.contracts import REFERENCE_FRAME_DIM, REFERENCE_HORIZON
from oracle_composition.adapters.gmt.course_config import load_run_config
from oracle_composition.adapters.gmt.course_evaluation import DEVELOPMENT_GATE_THRESHOLDS
from oracle_composition.adapters.gmt.course_proposal import apply_proposal
from oracle_composition.adapters.gmt.course_runtime import LOOP_RUNTIME, runtime_profile_from_config
from oracle_composition.adapters.gmt.course_task import TaskFrame
from oracle_composition.adapters.gmt.heading_feedback import oracle_boundary_inputs
from oracle_composition.adapters.gmt.io import write_json_receipt
from oracle_composition.harness.contract import decode_json_object, read_json_object

_SCORER_PATH = Path(__file__).resolve()
_PROTOCOL_PATH = _SCORER_PATH.with_name("PROTOCOL.md")
_REPOSITORY_ROOT = _SCORER_PATH.parents[2]
_STUDY017_PATH = _SCORER_PATH.parents[1] / "017_g1_execution_derived_reference/score.py"
_STUDY018_CRITERIA_PATH = _SCORER_PATH.parents[1] / "018_g1_four_state_reward_loop/criteria.py"
_RUN_GMT_PROBE_PATH = _REPOSITORY_ROOT / "scripts/run_gmt_probe.py"
_RUN_GMT_DEVELOPMENT_PATH = _REPOSITORY_ROOT / "scripts/run_gmt_development.py"


def _load_local(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Study019 dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load_local("run_gmt_probe", _RUN_GMT_PROBE_PATH)
development = _load_local("study019_run_gmt_development", _RUN_GMT_DEVELOPMENT_PATH)
study017 = _load_local("study019_study017_helpers", _STUDY017_PATH)
study018_criteria = _load_local("study019_study018_criteria", _STUDY018_CRITERIA_PATH)
shared = study017.shared

RETAINED_ROOT = Path(
    "/Users/samueldoane/Documents/ChatGPT/humanoid-harness-probe-runs/"
    "gmt_course_o7b_study015_candidate_20260907"
)
RETAINED_BINDING = {
    "root": str(RETAINED_ROOT),
    "manifest_sha256": "92e2a72f743bded8b2694760c7492f8d35f0829489415a05c3afff6ec459944a",
    "resource_sha256": "ea00072569e159b8f04b1b7c801296b19297559b0ed91f04ec4077551cb429b8",
    "config_sha256": "ba45dba36ef88bdc522ed2115e46a4b3d876e00a627a089fd56ddf0de623e18a",
    "source_commit": "3cb3102d5cb2e3a8603663df13ed50b5d6cad0cb",
}
RETAINED_SOURCE_TREE_SHA256 = "3ba7fe0ab72374e0667402848460c43915024fbcd2fe44f9ce5de791f0ab8b8e"
RETAINED_SOURCE_FILE_COUNT = 194
PARENT_FEEDBACK_SHA256 = "6ea27ad8e8faa2aa0cfb6a5397d9d06aa58b55bd7913704a8cbbcd2d9837c795"
PREFIX_ACTION_COUNT = 71
PREFIX_STATE_COUNT = 72
SEED = 20260906
_HEX = frozenset("0123456789abcdef")
_MODES = {"before", "inside", "rise", "after"}
_DEVELOPMENT_GATES = {
    "finish_reached",
    "full_horizon_without_fall",
    "inside_mean_speed_target",
    "inside_posture_compliance",
    "inside_posture_dip",
    "joint_position_rmse_p95",
    "maximum_lateral_error",
    "mean_speed_error",
    "minimum_inside_samples",
    "region_entry_and_exit_observed",
    "roll_pitch_rmse_p95",
}


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: object, *, length: int, field: str) -> str:
    if type(value) is not str or len(value) != length or any(c not in _HEX for c in value):
        raise ValueError(f"{field} must be one lowercase hexadecimal digest")
    return value


def _pinned_json(path: Path, digest: str, *, source: str) -> tuple[dict, bytes]:
    value, encoded = read_json_object(path)
    if _sha256(encoded) != _digest(digest, length=64, field=f"{source} SHA-256"):
        raise ValueError(f"{source} bytes differ from their pin")
    return value, encoded


def _file_pin(value: object, *, field: str) -> tuple[Path, str]:
    if type(value) is not dict or set(value) != {"path", "sha256"}:
        raise ValueError(f"{field} binding fields differ")
    raw = Path(value["path"])
    path = raw.resolve(strict=True)
    if not raw.is_absolute() or path != raw or not path.is_file():
        raise ValueError(f"{field} path must be one direct absolute file")
    return path, _digest(value["sha256"], length=64, field=f"{field} SHA-256")


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
        or type(inputs["schema_version"]) is not int
        or inputs["schema_version"] != 1
    ):
        raise ValueError("Study019 input fields or schema differ")
    scorer = _digest(inputs["scorer_sha256"], length=64, field="scorer SHA-256")
    protocol = _digest(inputs["protocol_sha256"], length=64, field="protocol SHA-256")
    if (
        _sha256(_SCORER_PATH.read_bytes()) != scorer
        or _sha256(_PROTOCOL_PATH.read_bytes()) != protocol
    ):
        raise ValueError("Study019 scorer or protocol bytes differ from their predata pins")
    paths = {
        "study017_scorer": _STUDY017_PATH,
        "study015_scorer": study017._SHARED_PATH,
        "study018_criteria": _STUDY018_CRITERIA_PATH,
    }
    dependencies = inputs["dependency_sha256"]
    if (
        type(dependencies) is not dict
        or set(dependencies) != set(paths)
        or any(
            _digest(dependencies[name], length=64, field=f"{name} SHA-256")
            != _sha256(path.read_bytes())
            for name, path in paths.items()
        )
    ):
        raise ValueError("Study019 dependency bytes differ from their predata pins")
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


def _run_pins(value: object, common: dict[str, object], *, config_sha256: str, fresh: bool) -> Any:
    fields = {"root", "manifest_sha256", "resource_sha256", "config_sha256", "source_commit"}
    if fresh:
        fields |= {"reservation_path", "reservation_sha256"}
    if type(value) is not dict or set(value) != fields:
        raise ValueError("Study019 run pin fields differ")
    for name in fields - {"root", "reservation_path", "source_commit"}:
        _digest(value[name], length=64, field=f"run {name}")
    if value["config_sha256"] != config_sha256:
        raise ValueError("run config differs from its exact Study019 identity")
    expected_commit = common["source_commit"] if fresh else RETAINED_BINDING["source_commit"]
    if value["source_commit"] != expected_commit:
        raise ValueError("run source commit differs")
    for name in {"root", "reservation_path"} & fields:
        raw = Path(value[name])
        resolved = raw.resolve(strict=True)
        if not raw.is_absolute() or resolved != raw:
            raise ValueError(f"run {name} must be one direct absolute path")
    return study017._pins(value, common["coordination_root"])


def _retained(inputs: dict, common: dict[str, object]) -> Any:
    if inputs["retained"] != RETAINED_BINDING:
        raise ValueError("retained Study015 binding differs")
    return _run_pins(
        inputs["retained"], common, config_sha256=RETAINED_BINDING["config_sha256"], fresh=False
    )


def _verify_run(pins: Any, feedback_root: Path, *, tree: str, count: int) -> dict:
    run = shared._verify_run(
        pins,
        feedback_root,
        expected_source_tree_sha256=tree,
        expected_source_file_count=count,
    )
    feedback, encoded = read_json_object(feedback_root / "feedback_v1.json")
    if _sha256(encoded) != run["feedback"]["feedback"]["sha256"]:
        raise ValueError("rebuilt feedback bytes differ from their verifier receipt")
    run["rebuilt_feedback"] = feedback
    return run


def _feedback_parity(retained: dict, fresh: dict) -> dict[str, object]:
    first, second = deepcopy(retained["rebuilt_feedback"]), deepcopy(fresh["rebuilt_feedback"])
    first_manifest = first.pop("source_manifest_sha256", None)
    second_manifest = second.pop("source_manifest_sha256", None)
    first_hash = retained["feedback"]["feedback"]["sha256"]
    second_hash = fresh["feedback"]["feedback"]["sha256"]
    if (
        first_manifest != retained["pins"].manifest_sha256
        or second_manifest != fresh["pins"].manifest_sha256
        or first_manifest == second_manifest
        or first_hash == second_hash
        or first != second
    ):
        raise ValueError("fresh feedback differs beyond its new manifest provenance")
    return {
        "content_equal_except_source_manifest_sha256": True,
        "retained_feedback_sha256": first_hash,
        "fresh_control_feedback_sha256": second_hash,
        "manifest_bound_feedback_sha256_different": True,
    }


def verify_config(control: Any, candidate: Any) -> dict[str, object]:
    raw = control.raw
    if (
        control.sha256 != RETAINED_BINDING["config_sha256"]
        or runtime_profile_from_config(raw) != LOOP_RUNTIME
        or raw.get("mode") != "probe"
        or raw.get("training_steps") != 0
        or raw.get("seed") != SEED
    ):
        raise ValueError("control differs from exact O7b loop probe")
    expected = deepcopy(raw)
    first = expected.get("oracle", {}).get("transitions")
    second = candidate.raw.get("oracle", {}).get("transitions")
    candidate_id = candidate.raw.get("oracle", {}).get("oracle_id")
    if (
        type(first) is not list
        or type(second) is not list
        or len(first) != 3
        or len(second) != 3
        or first[0].get("from") != "before"
        or first[0].get("to") != "inside"
        or first[0].get("guard") != "x_travelled >= 0.65"
        or second[0].get("guard") != "x_travelled >= 0.40"
        or type(candidate_id) is not str
        or not candidate_id
        or candidate_id == expected["oracle"]["oracle_id"]
    ):
        raise ValueError("Study019 guard or immutable oracle identity differs")
    parent_id = expected["oracle"]["oracle_id"]
    expected["oracle"]["oracle_id"] = candidate_id
    expected["oracle"]["transitions"][0]["guard"] = "x_travelled >= 0.40"
    if candidate.raw != expected or candidate.runtime != LOOP_RUNTIME:
        raise ValueError("candidate changes more than oracle ID and the 0.40 m entry guard")
    return {
        "control_oracle_id": parent_id,
        "candidate_oracle_id": candidate_id,
        "control_guard": "x_travelled >= 0.65",
        "candidate_guard": "x_travelled >= 0.40",
    }


def replay_native_loop(
    config: Any, run: dict, *, diagnostic_indices: set[int] | None = None
) -> dict:
    """Reconstruct commands from retained boundaries; this is not a plant or actor rerun."""
    if config.runtime != LOOP_RUNTIME:
        raise ValueError("native replay requires the frozen loop runtime")
    frames, trajectory = run["frames"], run["trajectory"]
    qpos, qvel, references = (trajectory[name] for name in ("qpos", "qvel", "current_reference"))
    frame = TaskFrame.initialize(qpos[0, :2], qpos[0, 3:7])
    oracle = ComposedReference(config.program, config.segments)
    windows, endpoints, transitions = [], [], []
    observed, first_exit_guard, first_entry_guard = {}, None, None
    entry_rule = config.raw["oracle"]["transitions"][0].get("guard")
    entry_threshold = {"x_travelled >= 0.65": 0.65, "x_travelled >= 0.40": 0.40}.get(entry_rule)
    if entry_threshold is None:
        raise ValueError("native replay requires the exact Study019 entry guard")
    for index, row in enumerate(frames):
        previous_state, previous_behavior = oracle.machine.state, oracle.machine.behavior
        previous_segment, source_phase = (
            config.segments[previous_behavior],
            oracle._phase(index).clone(),
        )
        inputs = oracle_boundary_inputs(
            task=config.task, frame=frame, qpos=qpos[index], qvel=qvel[index], control_step=index
        )
        source_target = previous_segment.features(source_phase.reshape(1))[0].numpy()
        if (
            previous_state == "before"
            and inputs.projection.progress_m >= entry_threshold
            and first_entry_guard is None
        ):
            first_entry_guard = {
                "action_index": index,
                "command_number_1_based": index + 1,
                "progress_m": inputs.projection.progress_m,
                "source_phase_seconds": previous_segment.reported_phase(source_phase),
                "root_height_m": float(qpos[index, 2]),
                "source_target_height_m": float(source_target[0]),
            }
        if (
            previous_state == "inside"
            and inputs.projection.progress_m >= 2.05
            and first_exit_guard is None
        ):
            first_exit_guard = {
                "action_index": index,
                "command_number_1_based": index + 1,
                "progress_m": inputs.projection.progress_m,
                "source_phase_seconds": previous_segment.reported_phase(source_phase),
                "root_height_m": float(qpos[index, 2]),
                "source_target_height_m": float(source_target[0]),
            }
        command = oracle.command(step=index, signals=inputs.signals, robot_pose=inputs.robot_pose)
        if (
            row.get("executed_mode") != command.state
            or row.get("executed_behavior") != command.behavior
            or row.get("executed_phase_seconds") != command.phase_seconds
            or not study017._same_json(row.get("transition"), command.transition)
        ):
            raise ValueError("retained mode, phase, or transition differs from native replay")
        window = np.ascontiguousarray(command.window)
        endpoint = np.ascontiguousarray(oracle.current_after_step(index + 1))
        if window.shape != (REFERENCE_HORIZON, REFERENCE_FRAME_DIM) or not np.array_equal(
            endpoint, references[index]
        ):
            raise ValueError("retained post-step reference differs from native replay")
        windows.append(window)
        endpoints.append(endpoint)
        if diagnostic_indices and index in diagnostic_indices:
            segment = config.segments[command.behavior]
            observed[str(index)] = {
                "poststep_phase_seconds": segment.reported_phase(oracle._phase(index + 1)),
                "poststep_target_height_m": float(endpoint[0]),
            }
        if command.transition is not None:
            transitions.append(
                {
                    "action_index": index,
                    "command_number_1_based": index + 1,
                    "pre_action_progress_m": inputs.projection.progress_m,
                    "source_phase_seconds": previous_segment.reported_phase(source_phase),
                    "pre_action_root_height_m": float(qpos[index, 2]),
                    "post_action_root_height_m": float(qpos[index + 1, 2]),
                    "source_target_height_m": float(source_target[0]),
                    "poststep_target_height_m": float(endpoint[0]),
                    **dict(command.transition),
                }
            )
    stacked_windows, stacked_endpoints = np.stack(windows), np.stack(endpoints)
    entry = next(
        (
            row
            for row in transitions
            if row["from_state"] == "before" and row["to_state"] == "inside"
        ),
        None,
    )
    rise = next(
        (row for row in transitions if row["from_state"] == "inside" and row["to_state"] == "rise"),
        None,
    )
    entry_dispatch = (
        None
        if first_entry_guard is None or entry is None
        else {
            "guard": entry_rule,
            "first_guard_satisfaction": first_entry_guard,
            "actual_switch": entry,
            "deferred_action_count": entry["action_index"] - first_entry_guard["action_index"],
            "elapsed_deferral_seconds": (entry["action_index"] - first_entry_guard["action_index"])
            * 0.02,
        }
    )
    exit_deferral = (
        None
        if first_exit_guard is None or rise is None
        else {
            "first_guard_satisfaction": first_exit_guard,
            "actual_switch": rise,
            "deferred_action_count": rise["action_index"] - first_exit_guard["action_index"],
            "elapsed_deferral_seconds": (rise["action_index"] - first_exit_guard["action_index"])
            * 0.02,
        }
    )
    return {
        "scope": "raw_boundary_state_to_reconstructed_native_reference_not_plant_or_actor_rerun",
        "windows_reconstructed_not_directly_retained": True,
        "all_retained_actions_replayed": True,
        "command_count": len(frames),
        "reconstructed_window_shape": list(stacked_windows.shape),
        "reconstructed_window_sha256": study017._array_sha256(stacked_windows),
        "reconstructed_poststep_target_sha256": study017._array_sha256(stacked_endpoints),
        "transitions": transitions,
        "entry_boundary_dispatch": entry_dispatch,
        "inside_exit_boundary_deferral": exit_deferral,
        "diagnostic_poststep_targets": observed,
    }


def verify_prefix(
    control: dict, candidate: dict, control_replay: dict, candidate_replay: dict
) -> dict:
    def first_inside(replay: dict) -> dict | None:
        return next((row for row in replay["transitions"] if row["to_state"] == "inside"), None)

    control_entry, candidate_entry = first_inside(control_replay), first_inside(candidate_replay)
    control_dispatch = control_replay.get("entry_boundary_dispatch")
    candidate_dispatch = candidate_replay.get("entry_boundary_dispatch")
    if (
        control_entry is None
        or candidate_entry is None
        or control_entry["action_index"] != 92
        or candidate_entry["action_index"] != 71
        or control_entry.get("selected_phase_seconds") != 0.0
        or candidate_entry.get("selected_phase_seconds") != 0.0
        or type(control_dispatch) is not dict
        or type(candidate_dispatch) is not dict
        or control_dispatch["first_guard_satisfaction"]["action_index"] != 92
        or control_dispatch["actual_switch"]["action_index"] != 92
        or candidate_dispatch["first_guard_satisfaction"]["action_index"] != 71
        or candidate_dispatch["actual_switch"]["action_index"] != 71
        or control_dispatch["deferred_action_count"] != 0
        or candidate_dispatch["deferred_action_count"] != 0
    ):
        raise ValueError("entry transition does not match the preregistered dispatch boundary")
    if control["frames"][:PREFIX_ACTION_COUNT] != candidate["frames"][:PREFIX_ACTION_COUNT]:
        raise ValueError("complete frame prefix differs before changed command 72")
    for name, count in (
        ("composite_raw_action", 71),
        ("residual_action", 71),
        ("current_reference", 71),
        ("qpos", 72),
        ("qvel", 72),
    ):
        if not np.array_equal(
            control["trajectory"][name][:count], candidate["trajectory"][name][:count]
        ):
            raise ValueError(f"numeric prefix differs before changed command 72: {name}")
    if control["frames"][71] == candidate["frames"][71] or np.array_equal(
        control["trajectory"]["current_reference"][71],
        candidate["trajectory"]["current_reference"][71],
    ):
        raise ValueError("preregistered first changed command is not command 72")
    return {
        "equal_action_and_frame_rows": 71,
        "equal_state_boundaries": 72,
        "first_changed_action_index": 71,
        "first_changed_command_number_1_based": 72,
    }


def _objective(run: dict) -> dict:
    value = run["objective"]
    gates = value.get("development_gate_results") if type(value) is dict else None
    if (
        type(gates) is not dict
        or set(gates) != _DEVELOPMENT_GATES
        or any(type(result) is not bool for result in gates.values())
        or value.get("development_gate_thresholds") != DEVELOPMENT_GATE_THRESHOLDS
        or value.get("development_gate_passed") is not all(gates.values())
    ):
        raise ValueError("original eleven objective gates or thresholds differ")
    return value


def _reset_and_base(
    control: dict, candidate: dict, control_config: Any, candidate_config: Any
) -> dict:
    evaluations = [
        decode_json_object(
            run["outputs"]["zero_residual_evaluation.json"], source="zero evaluation"
        )
        for run in (control, candidate)
    ]
    resets = [value.get("reset") for value in evaluations]
    if any(type(value) is not dict for value in resets):
        raise ValueError("paired reset metadata is absent")
    if (
        resets[0].get("oracle_sha256") != control_config.program.sha256
        or resets[1].get("oracle_sha256") != candidate_config.program.sha256
    ):
        raise ValueError("paired reset oracle identities differ")
    if {k: v for k, v in resets[0].items() if k != "oracle_sha256"} != {
        k: v for k, v in resets[1].items() if k != "oracle_sha256"
    }:
        raise ValueError("paired reset metadata differs beyond oracle identity")
    hashes = {}
    for name in ("qpos", "qvel"):
        rows = [run["trajectory"][name][0] for run in (control, candidate)]
        if not np.array_equal(*rows):
            raise ValueError(f"paired initial base state differs: {name}")
        hashes[name] = study017._array_sha256(rows[0])
    return {
        "equal_beyond_oracle_identity": True,
        "initial_qpos_qvel_equal": True,
        "initial_state_array_sha256": hashes,
    }


def _diagnostics(run: dict, config: Any) -> dict[str, object]:
    objective = _objective(run)
    visits = study017.physical_region_visits(run["frames"], objective, config.raw["task"])
    entry = visits["first_entry_action_index"]
    replay = replay_native_loop(config, run, diagnostic_indices=set() if entry is None else {entry})
    contacts = study017.recorded_ground_contacts(run["frames"])
    heading = study018_criteria.heading_diagnostics(run["frames"])
    previous = unwrapped = 0.0
    signed_after = []
    for row in run["frames"]:
        signed = row["metrics"]["heading_error_signed_rad"]
        difference = signed - previous
        increment = (difference + math.pi) % (2 * math.pi) - math.pi
        if increment == -math.pi and difference > 0:
            increment = math.pi
        unwrapped += increment
        previous = signed
        if row["executed_mode"] == "after":
            signed_after.append(unwrapped)
    signed_heading = {
        "after_first_heading_signed_unwrapped_rad": signed_after[0] if signed_after else None,
        "after_final_heading_signed_unwrapped_rad": signed_after[-1] if signed_after else None,
        "after_peak_heading_signed_unwrapped_rad": (
            max(signed_after, key=abs) if signed_after else None
        ),
    }
    if (
        signed_after
        and abs(signed_heading["after_peak_heading_signed_unwrapped_rad"])
        != heading["after_heading_max_abs_unwrapped_rad"]
    ):
        raise ValueError("signed after-heading reconstruction differs")
    entry_target = (
        replay["diagnostic_poststep_targets"].get(str(entry)) if entry is not None else None
    )
    rise = next((row for row in replay["transitions"] if row["to_state"] == "rise"), None)
    return {
        "objective": objective,
        "visits": visits,
        "contacts": contacts,
        "replay": replay,
        "heading": heading,
        "measurements": {
            "minimum_physical_region_height_m": objective["region"]["minimum_root_height_m"],
            "physical_region_posture_compliant_fraction": objective["region"][
                "posture_compliant_fraction"
            ],
            "inside_mean_speed_target_deviation_m_s": objective["speed"][
                "inside_mean_speed_target_deviation_m_s"
            ],
            "overall_speed_mean_absolute_error_m_s": objective["speed"]["mean_absolute_error_m_s"],
            "maximum_lateral_error_m": objective["maximum_lateral_error_m"],
            "first_physical_entry_action_index": entry,
            "first_physical_entry_executed_mode": None
            if entry is None
            else run["frames"][entry]["executed_mode"],
            "first_physical_entry_pre_action_phase_seconds": None
            if entry is None
            else run["frames"][entry]["executed_phase_seconds"],
            "first_physical_entry_poststep_target": entry_target,
            "entry_boundary_dispatch": replay["entry_boundary_dispatch"],
            "rise_handover": rise,
            "inside_exit_boundary_deferral": replay["inside_exit_boundary_deferral"],
            **heading,
            **signed_heading,
        },
    }


def candidate_criteria(diagnostics: dict) -> tuple[dict[str, bool], dict[str, bool]]:
    objective, visits = diagnostics["objective"], diagnostics["visits"]
    measures, contacts = diagnostics["measurements"], diagnostics["contacts"]
    target = measures["first_physical_entry_poststep_target"]
    manipulation = {
        "first_physical_entry_executes_inside": measures["first_physical_entry_executed_mode"]
        == "inside",
        "entry_poststep_phase_in_1p45_to_1p80_s": target is not None
        and 1.45 <= target["poststep_phase_seconds"] <= 1.80,
        "entry_poststep_target_height_below_0p50_m": target is not None
        and target["poststep_target_height_m"] < 0.50,
    }
    oracle, tracking, region = (
        objective["oracle_diagnostics"],
        objective["tracking"],
        objective["region"],
    )
    counts = oracle["executed_mode_counts"]
    if (
        type(counts) is not dict
        or set(counts) - _MODES
        or any(type(v) is not int or v < 0 for v in counts.values())
    ):
        raise ValueError("objective executed-mode counts differ")
    rows = {
        "manipulation": all(manipulation.values()),
        "p1": region["posture_compliant_fraction"] >= 0.85,
        "survival": objective["duration_seconds"] == 20.0
        and objective["fall_count"] == 0
        and contacts["nonfoot_ground_contact_action_count"] == 0,
        "composition": oracle["observed_switch_count"] == 3
        and set(counts) == _MODES
        and all(counts.values()),
        "exposure": region["entry_observed"] is True
        and region["exit_observed_after_entry"] is True
        and objective["finish_condition_observed"] is True
        and visits["all_visit_sample_count"] >= 25,
        "recovery_guardrail": visits["after_reentry_sample_count"] == 0,
        "tracking": tracking["joint_position_rmse_rad_p95"] <= 0.35
        and tracking["roll_pitch_rmse_rad_p95"] <= 0.25,
    }
    return rows, manipulation


def _fresh(
    inputs: dict, common: dict[str, object], name: str, config_sha256: str
) -> tuple[Any, dict, Any]:
    pins = _run_pins(inputs[name], common, config_sha256=config_sha256, fresh=True)
    temp_parent = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix=f"study019-{name}-", dir=temp_parent) as raw:
        run = _verify_run(
            pins,
            Path(raw) / "feedback",
            tree=common["source_tree_sha256"],
            count=common["source_file_count"],
        )
    config = load_run_config(pins.root / "input_config.json")
    if config.sha256 != config_sha256:
        raise ValueError(f"{name} admitted config differs from its byte pin")
    return pins, run, config


def score_control(inputs: dict) -> dict[str, object]:
    common = _validate_common(inputs, extra={"retained", "control"})
    retained_pins = _retained(inputs, common)
    _control_pins, control, control_config = _fresh(
        inputs, common, "control", RETAINED_BINDING["config_sha256"]
    )
    temp_parent = Path(tempfile.gettempdir()).resolve()
    with tempfile.TemporaryDirectory(prefix="study019-retained-", dir=temp_parent) as raw:
        retained = _verify_run(
            retained_pins,
            Path(raw) / "feedback",
            tree=RETAINED_SOURCE_TREE_SHA256,
            count=RETAINED_SOURCE_FILE_COUNT,
        )
    if retained["config_bytes"] != control["config_bytes"]:
        raise ValueError("fresh control config bytes differ from retained Study015 O7b")
    for name in shared._COMPARABLE_OUTPUTS:
        if retained["outputs"][name] != control["outputs"][name]:
            raise ValueError(f"fresh control does not byte-reproduce retained O7b: {name}")
    if retained["feedback"]["feedback"]["sha256"] != PARENT_FEEDBACK_SHA256:
        raise ValueError("retained proposal-parent feedback identity differs")
    feedback_parity = _feedback_parity(retained, control)
    loop = study017.verify_loop_manifest(control)
    replay = replay_native_loop(control_config, control)
    _objective(control)
    return {
        "schema_version": 1,
        "study": "019",
        "artifact": "gmt_g1_study019_control_verification/v1",
        **{
            name: common[name]
            for name in (
                "scorer_sha256",
                "protocol_sha256",
                "dependency_sha256",
                "source_commit",
                "source_tree_sha256",
                "source_file_count",
            )
        },
        "retained_binding": inputs["retained"],
        "control_binding": inputs["control"],
        "verified_for_candidate_dispatch": True,
        "retained_output_byte_parity": {name: True for name in sorted(shared._COMPARABLE_OUTPUTS)},
        "proposal_parent_feedback_sha256": PARENT_FEEDBACK_SHA256,
        "fresh_feedback_parity": feedback_parity,
        "loop_runtime": loop,
        "native_reference_replay": replay,
        "claim_scope": "fresh_control_byte_parity_only_not_candidate_evidence",
    }


def _control_receipt(binding: object, common: dict[str, object], inputs: dict) -> dict:
    path, digest = _file_pin(binding, field="control verification")
    receipt, _ = _pinned_json(path, digest, source="control verification")
    exact = all(
        receipt.get(name) == common[name]
        for name in (
            "scorer_sha256",
            "protocol_sha256",
            "dependency_sha256",
            "source_commit",
            "source_tree_sha256",
            "source_file_count",
        )
    )
    _digest(receipt.get("inputs_sha256"), length=64, field="control inputs SHA-256")
    fresh_feedback = receipt.get("fresh_feedback_parity", {}).get("fresh_control_feedback_sha256")
    _digest(fresh_feedback, length=64, field="fresh control feedback SHA-256")
    if (
        receipt.get("artifact") != "gmt_g1_study019_control_verification/v1"
        or receipt.get("study") != "019"
        or not exact
        or receipt.get("retained_binding") != inputs["retained"]
        or receipt.get("control_binding") != inputs["control"]
        or receipt.get("verified_for_candidate_dispatch") is not True
        or set(receipt.get("retained_output_byte_parity", {})) != shared._COMPARABLE_OUTPUTS
        or not all(receipt["retained_output_byte_parity"].values())
        or receipt.get("proposal_parent_feedback_sha256") != PARENT_FEEDBACK_SHA256
        or receipt.get("fresh_feedback_parity", {}).get(
            "content_equal_except_source_manifest_sha256"
        )
        is not True
        or receipt["fresh_feedback_parity"].get("retained_feedback_sha256")
        != PARENT_FEEDBACK_SHA256
        or fresh_feedback == PARENT_FEEDBACK_SHA256
        or receipt["fresh_feedback_parity"].get("manifest_bound_feedback_sha256_different")
        is not True
    ):
        raise ValueError("candidate scoring lacks exact successful control verification")
    return receipt


def _proposal(inputs: dict, control_config: Any, candidate_config: Any, control: dict) -> dict:
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
    control_pins, control, control_config = _fresh(
        inputs, common, "control", RETAINED_BINDING["config_sha256"]
    )
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
    proposal = _proposal(inputs, control_config, candidate_config, control)
    loop = {
        "control": study017.verify_loop_manifest(control),
        "candidate": study017.verify_loop_manifest(candidate),
    }
    if loop["control"] != loop["candidate"]:
        raise ValueError("paired loop runtime manifests differ")
    reset = _reset_and_base(control, candidate, control_config, candidate_config)
    control_diagnostics = _diagnostics(control, control_config)
    candidate_diagnostics = _diagnostics(candidate, candidate_config)
    prefix = verify_prefix(
        control, candidate, control_diagnostics["replay"], candidate_diagnostics["replay"]
    )
    rows, manipulation = candidate_criteria(candidate_diagnostics)
    return {
        "schema_version": 1,
        "study": "019",
        "artifact": "gmt_g1_study019_oracle_alignment_pair_score/v1",
        **{
            name: common[name]
            for name in (
                "scorer_sha256",
                "protocol_sha256",
                "dependency_sha256",
                "source_commit",
                "source_tree_sha256",
                "source_file_count",
            )
        },
        "run_bindings": {"control": inputs["control"], "candidate": inputs["candidate"]},
        "candidate_config_sha256": candidate_hash,
        "control_verification_sha256": inputs["control_verification"]["sha256"],
        "proposal": proposal,
        "config_change": config_change,
        "fixed_resource_inputs_equal": True,
        "fixed_resource_inputs": control["resource_fixed_inputs"],
        "loop_runtime": loop,
        "reset_and_base_state": reset,
        "prefix_integrity": prefix,
        "candidate_criteria": rows,
        "manipulation_checks": manipulation,
        "study_screen_passed": all(rows.values()),
        "measurements": {
            name: value["measurements"]
            for name, value in (
                ("control", control_diagnostics),
                ("candidate", candidate_diagnostics),
            )
        },
        "native_reference_replay": {
            name: value["replay"]
            for name, value in (
                ("control", control_diagnostics),
                ("candidate", candidate_diagnostics),
            )
        },
        "physical_region_visits": {
            name: value["visits"]
            for name, value in (
                ("control", control_diagnostics),
                ("candidate", candidate_diagnostics),
            )
        },
        "recorded_ground_contacts": {
            name: value["contacts"]
            for name, value in (
                ("control", control_diagnostics),
                ("candidate", candidate_diagnostics),
            )
        },
        "original_development_gate_results": {
            name: value["objective"]["development_gate_results"]
            for name, value in (
                ("control", control_diagnostics),
                ("candidate", candidate_diagnostics),
            )
        },
        "raw_objectives": {
            name: value["objective"]
            for name, value in (
                ("control", control_diagnostics),
                ("candidate", candidate_diagnostics),
            )
        },
        "claim_scope": "one_seed_zero_training_oracle_screen_not_causal_isolation_or_generalization",
        "claim_limits": [
            "windows_are_reconstructed_not_directly_retained",
            "no_training",
            "correlated_single_episode_samples_no_uncertainty_estimate",
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
    if _sha256(encoded) != _digest(args.inputs_sha256, length=64, field="Study019 inputs SHA-256"):
        raise ValueError("Study019 inputs differ from their pinned bytes")
    result = score_control(inputs) if args.command == "control" else score_pair(inputs)
    result["inputs_sha256"] = args.inputs_sha256
    digest = write_json_receipt(args.output, result)
    summary = {"path": str(args.output), "sha256": digest, "artifact": result["artifact"]}
    summary.update(
        verified_for_candidate_dispatch=result.get("verified_for_candidate_dispatch"),
        study_screen_passed=result.get("study_screen_passed"),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
