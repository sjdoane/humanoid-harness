"""Focused integrity and diagnostic checks for the Study017 scorer."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oracle_composition.adapters.gmt.composition import ComposedReference, ReferenceSegment
from oracle_composition.adapters.gmt.course_runtime import LOOP_RUNTIME
from oracle_composition.adapters.gmt.course_task import CourseTaskSpec, TaskFrame
from oracle_composition.adapters.gmt.heading_feedback import oracle_boundary_inputs
from oracle_composition.adapters.gmt.reference_runtime import ReferenceMotion
from oracle_composition.harness.contract import oracle_program_from_dict


@pytest.fixture(scope="module")
def scorer():
    path = (
        Path(__file__).resolve().parents[2]
        / "experiments/017_g1_execution_derived_reference/score.py"
    )
    spec = importlib.util.spec_from_file_location("study017_test_score", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _control_config() -> dict:
    return {
        "schema_version": 2,
        "mode": "probe",
        "assets": {
            "upstream_root": "/fixed",
            "weights": {"fixed": True},
            "motions": {
                "basic_walk": {"fixed": "walk"},
                "crouchwalk_stand": {"fixed": "crouch"},
            },
        },
        "task": {"fixed": True},
        "reward": {"fixed": True},
        "oracle": {"fixed": True},
        "segments": {
            "walk": {"fixed": True},
            "crouch": {"old": True},
            "rise": {"fixed": True},
            "walk_after": {"fixed": True},
        },
        "runtime": {"schema_version": 1, "profile_id": "gmt_g1_four_state_loop_course/v1"},
        "seed": 20260906,
        "training_steps": 0,
    }


def _candidate_config(scorer) -> dict:
    value = copy.deepcopy(_control_config())
    value["assets"]["motions"][scorer.DERIVED_MOTION_NAME] = {
        "path": scorer.DERIVED_ARCHIVE_PATH,
        "sha256": scorer.DERIVED_ARCHIVE_SHA256,
        "manifest": {
            "path": scorer.DERIVED_MANIFEST_PATH,
            "sha256": scorer.DERIVED_MANIFEST_SHA256,
        },
    }
    value["segments"]["crouch"] = copy.deepcopy(scorer.DERIVED_SEGMENT)
    return value


def test_config_diff_is_only_exact_asset_and_inside_segment(scorer) -> None:
    control = _control_config()
    candidate = _candidate_config(scorer)
    scorer.verify_config(control, candidate)
    for mutation in ("reward", "oracle", "before_segment", "manifest"):
        wrong = copy.deepcopy(candidate)
        if mutation in {"reward", "oracle"}:
            wrong[mutation]["changed"] = True
        elif mutation == "before_segment":
            wrong["segments"]["walk"]["changed"] = True
        else:
            wrong["assets"]["motions"][scorer.DERIVED_MOTION_NAME]["manifest"]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="exact derived"):
            scorer.verify_config(control, wrong)


def test_exact_materialized_candidate_config_is_pinned_when_available(scorer) -> None:
    candidate_path = Path(
        "/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/"
        "artifacts/gmt/course_configs/study017_execution_derived_candidate_20260907.json"
    )
    control_path = Path(
        "/Users/samueldoane/Documents/ChatGPT/humanoid-harness-probe-runs/"
        "gmt_course_o7b_study015_candidate_20260907/input_config.json"
    )
    if not candidate_path.is_file() or not control_path.is_file():
        pytest.skip("exact local Study017 materials are unavailable")
    assert scorer._sha256(candidate_path.read_bytes()) == scorer.CANDIDATE_CONFIG_SHA256
    scorer.verify_config(
        json.loads(control_path.read_bytes()), json.loads(candidate_path.read_bytes())
    )


def _prefix_run() -> dict:
    frames = [
        {
            "executed_mode": "before" if index < 92 else "inside",
            "reward": {"total": 1.0},
        }
        for index in range(94)
    ]
    qpos = np.zeros((95, 30), dtype="<f8")
    qvel = np.zeros((95, 29), dtype="<f8")
    references = np.zeros((94, 30), dtype="<f4")
    return {
        "frames": frames,
        "trajectory": {
            "composite_raw_action": np.zeros((94, 23), dtype="<f4"),
            "qpos": qpos,
            "qvel": qvel,
            "current_reference": references,
        },
    }


def test_prefix_allows_first_change_at_action92_and_state93(scorer) -> None:
    control = _prefix_run()
    candidate = copy.deepcopy(control)
    candidate["trajectory"]["current_reference"][92, 0] = 0.5
    candidate["trajectory"]["composite_raw_action"][92, 0] = 0.1
    candidate["trajectory"]["qpos"][93, 0] = 0.1
    candidate["trajectory"]["qvel"][93, 0] = 0.1
    result = scorer.verify_prefix(control, candidate)
    assert result["equal_complete_frame_count"] == 92
    assert result["equal_qpos_qvel_boundary_count"] == 93
    assert result["first_inside_poststep_target_changed"] is True


@pytest.mark.parametrize(
    ("field", "index"),
    (("composite_raw_action", 91), ("qpos", 92), ("qvel", 92)),
)
def test_prefix_rejects_early_numeric_change(scorer, field: str, index: int) -> None:
    control = _prefix_run()
    candidate = copy.deepcopy(control)
    candidate["trajectory"]["current_reference"][92, 0] = 0.5
    candidate["trajectory"][field][index, 0] = 0.1
    with pytest.raises(ValueError, match="numeric prefix"):
        scorer.verify_prefix(control, candidate)


def test_prefix_requires_changed_first_inside_target_and_complete_frames(scorer) -> None:
    control = _prefix_run()
    candidate = copy.deepcopy(control)
    with pytest.raises(ValueError, match="must change"):
        scorer.verify_prefix(control, candidate)
    candidate["trajectory"]["current_reference"][92, 0] = 0.5
    candidate["frames"][12]["reward"]["total"] = 2.0
    with pytest.raises(ValueError, match="complete trace prefix"):
        scorer.verify_prefix(control, candidate)


def _objective(scorer, *, inside_count: int, first_entry: int | None, first_exit: int | None):
    return {
        "duration_seconds": 20.0,
        "fall_count": 0,
        "finish_condition_observed": True,
        "maximum_progress_m": 3.5,
        "region": {
            "entry_observed": first_entry is not None,
            "first_entry_step": first_entry + 1 if first_entry is not None else None,
            "exit_observed_after_entry": first_exit is not None,
            "first_exit_step": first_exit + 1 if first_exit is not None else None,
            "inside_sample_count": inside_count,
            "minimum_root_height_m": scorer.DONOR_MINIMUM_HEIGHT_M + 0.02,
        },
        "tracking": {
            "joint_position_rmse_rad_p95": 0.35,
            "roll_pitch_rmse_rad_p95": 0.25,
        },
        "oracle_diagnostics": {
            "observed_switch_count": 3,
            "executed_mode_counts": {"before": 1, "inside": 1, "rise": 1, "after": 1},
        },
        "development_gate_thresholds": copy.deepcopy(scorer.DEVELOPMENT_GATE_THRESHOLDS),
        "development_gate_results": {name: False for name in scorer.shared._DEVELOPMENT_GATES},
    }


def _region_frames(progress: list[float], modes: list[str]) -> list[dict]:
    return [
        {
            "executed_mode": mode,
            "metrics": {
                "progress_m": value,
                "inside_posture_region": 1.0 <= value < 2.0,
                "failure_reasons": [],
            },
        }
        for value, mode in zip(progress, modes, strict=True)
    ]


def test_all_first_and_late_physical_visits_remain_explicit(scorer) -> None:
    progress = [0.5, 1.2, 1.8, 2.1, 2.4, 1.7, 1.4, 2.2]
    modes = ["before", "inside", "inside", "rise", "after", "after", "after", "after"]
    frames = _region_frames(progress, modes)
    visits = scorer.physical_region_visits(
        frames,
        _objective(scorer, inside_count=4, first_entry=1, first_exit=3),
        {"region_entry_distance_m": 1.0, "region_exit_distance_m": 2.0},
    )
    assert visits["first_visit_action_indices"] == [1, 2]
    assert visits["later_visit_action_indices"] == [5, 6]
    assert visits["after_reentry_action_indices"] == [5, 6]


def _contact_frame(*, body: str | None = None, reason: bool = False) -> dict:
    names = ["world", "left_ankle_roll_link", "torso_link"]
    pairs = [] if body is None else [[0, names.index(body)]]
    return {
        "trajectory": {
            "geom_body_names": names,
            "contact_pairs": [pairs] + [[] for _ in range(19)],
        },
        "metrics": {"failure_reasons": ["non_foot_ground_contact"] if reason else []},
    }


def test_raw_nonfoot_contact_is_reconstructed_and_cross_checked(scorer) -> None:
    clean = scorer.recorded_ground_contacts([_contact_frame(body="left_ankle_roll_link")])
    assert clean["nonfoot_ground_contact_action_count"] == 0
    observed = scorer.recorded_ground_contacts([_contact_frame(body="torso_link", reason=True)])
    assert observed["nonfoot_ground_contact_bodies"] == ["torso_link"]
    with pytest.raises(ValueError, match="reason"):
        scorer.recorded_ground_contacts([_contact_frame(body="torso_link", reason=False)])


def _motion(height: float, *, frames: int = 6) -> ReferenceMotion:
    root = np.zeros((frames, 3), dtype="<f4")
    root[:, 0] = np.linspace(0.0, 0.1, frames, dtype=np.float32)
    root[:, 2] = height
    rotation = np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype="<f4"), (frames, 1))
    return ReferenceMotion(
        {
            "fps": np.asarray([50.0], dtype="<f8"),
            "root_pos": root,
            "root_rot": rotation,
            "dof_pos": np.zeros((frames, 23), dtype="<f4"),
        }
    )


def _replay_fixture():
    behaviors = ["walk", "crouch", "rise", "walk_after"]
    program = oracle_program_from_dict(
        {
            "schema_version": 1,
            "evidence_class": "exploratory_oracle_cycle",
            "oracle_id": "study017_replay_fixture",
            "behaviors": behaviors,
            "initial": "before",
            "states": {
                "before": {"behavior": "walk", "min_dwell": 1},
                "inside": {"behavior": "crouch", "min_dwell": 1},
                "rise": {"behavior": "rise", "min_dwell": 1},
                "after": {"behavior": "walk_after", "min_dwell": 1},
            },
            "transitions": [
                {"from": "before", "to": "inside", "priority": 0, "guard": "x_travelled >= 0.5"},
                {"from": "inside", "to": "rise", "priority": 0, "guard": "x_travelled >= 2.05"},
                {"from": "rise", "to": "after", "priority": 0, "guard": "dwell >= 1"},
            ],
        },
        available_behaviors=behaviors,
    )
    walk = _motion(0.8, frames=201)
    crouch = _motion(0.5, frames=6)
    rise = _motion(0.7, frames=51)
    segments = {
        "walk": ReferenceSegment(walk, "a" * 64, 0.0, float(walk.duration)),
        "crouch": ReferenceSegment(
            crouch,
            "b" * 64,
            0.0,
            float(crouch.duration),
            entry_phase_end_seconds=0.0,
            boundary="hold_last_pose_zero_velocity",
        ),
        "rise": ReferenceSegment(rise, "c" * 64, 0.0, float(rise.duration)),
        "walk_after": ReferenceSegment(walk, "d" * 64, 0.0, float(walk.duration)),
    }
    task = CourseTaskSpec(1.0, 2.0, 3.5, 0.7, 0.65, 0.3, 0.6, 9)
    config = SimpleNamespace(runtime=LOOP_RUNTIME, program=program, segments=segments, task=task)
    progress = [0.0, 0.6, 1.2, 1.5, 1.7, 1.9, 2.1, 2.2, 3.0, 3.6]
    qpos = np.zeros((len(progress), 30), dtype="<f8")
    qpos[:, 0] = progress
    qpos[:, 2] = 0.8
    qpos[:, 3] = 1.0
    qvel = np.zeros((len(progress), 29), dtype="<f8")
    qvel[1:, 0] = np.diff(progress) / 0.02
    frame = TaskFrame.initialize(qpos[0, :2], qpos[0, 3:7])
    oracle = ComposedReference(program, segments)
    frames = []
    endpoints = []
    for index in range(len(progress) - 1):
        inputs = oracle_boundary_inputs(
            task=task,
            frame=frame,
            qpos=qpos[index],
            qvel=qvel[index],
            control_step=index,
        )
        command = oracle.command(step=index, signals=inputs.signals, robot_pose=inputs.robot_pose)
        endpoint = oracle.current_after_step(index + 1)
        endpoints.append(endpoint)
        frames.append(
            {
                "executed_mode": command.state,
                "executed_behavior": command.behavior,
                "executed_phase_seconds": command.phase_seconds,
                "transition": command.transition,
                "metrics": {
                    "progress_m": progress[index + 1],
                    "forward_speed_m_s": (progress[index + 1] - progress[index]) / 0.02,
                },
            }
        )
    run = {
        "frames": frames,
        "trajectory": {
            "qpos": qpos,
            "qvel": qvel,
            "current_reference": np.ascontiguousarray(endpoints, dtype="<f4"),
            "composite_raw_action": np.zeros((len(frames), 23), dtype="<f4"),
        },
    }
    return config, run


def test_full_reference_replay_verifies_phase_transition_window_endpoint_and_hold(scorer) -> None:
    config, run = _replay_fixture()
    result = scorer.replay_composed_reference(config, run)
    assert result["all_retained_actions_replayed"] is True
    assert result["command_count"] == len(run["frames"])
    assert result["reconstructed_window_shape"] == [len(run["frames"]), 20, 30]
    assert result["inside_entry"]["selected_phase_seconds"] == 0.0
    assert result["inside_exit"]["transition_action_index"] == 6
    assert result["terminal_hold"]["first_lookahead_exposure"] is not None
    assert result["terminal_hold"]["first_poststep_endpoint_hold"] is not None
    assert (
        result["inside_speed_summary"]["candidate_body_local_lateral_velocity_m_s"]["max_abs"]
        == 0.0
    )
    assert result["lateral_velocity_scope"].endswith("descriptive_only")


@pytest.mark.parametrize("mutation", ["phase", "transition", "endpoint"])
def test_reference_replay_rejects_tampered_retained_evidence(scorer, mutation: str) -> None:
    config, run = _replay_fixture()
    if mutation == "phase":
        run["frames"][2]["executed_phase_seconds"] += 0.01
    elif mutation == "transition":
        run["frames"][1]["transition"]["selected_phase_seconds"] = 0.02
    else:
        run["trajectory"]["current_reference"][2, 0] += np.float32(0.01)
    with pytest.raises(ValueError, match=r"replay|phase|transition"):
        scorer.replay_composed_reference(config, run)


def test_body_local_lateral_velocity_uses_full_recorded_orientation(scorer) -> None:
    yaw = math.pi / 2
    orientation = np.asarray([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    local = scorer._body_local_velocity(orientation, np.asarray([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(local, [0.0, -1.0, 0.0], atol=1e-12)


def test_exact_feasibility_screen_does_not_replace_original_gates(scorer) -> None:
    objective = _objective(scorer, inside_count=25, first_entry=1, first_exit=3)
    objective["region"]["minimum_root_height_m"] = scorer.DONOR_MINIMUM_HEIGHT_M + 0.019
    visits = {"after_reentry_sample_count": 0, "all_visit_sample_count": 25}
    contacts = {"nonfoot_ground_contact_action_count": 0}
    screen = scorer.feasibility_screen(objective, visits, contacts)
    assert all(screen.values())
    assert not any(objective["development_gate_results"].values())
    objective["region"]["minimum_root_height_m"] = scorer.DONOR_MINIMUM_HEIGHT_M + 0.020001
    assert not scorer.feasibility_screen(objective, visits, contacts)[
        "inside_minimum_height_within_0p020m_of_donor"
    ]


def test_fresh_tree_is_fixed_but_common_full_commit_comes_from_run_pins(scorer) -> None:
    assert scorer._sealed_source_tree() == (
        "5e516691bfe5c090ff0625c6b13d67a66265a367c2b2000f6e41b61d98877738",
        196,
    )
    assert not hasattr(scorer, "FRESH_SOURCE_COMMIT")
