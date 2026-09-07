"""Negative-path checks for the fixed after-feedback comparison."""

from __future__ import annotations

import copy
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.adapters.gmt.course_task import TaskFrame


@pytest.fixture(scope="module")
def scorer():
    path = Path(__file__).resolve().parents[2] / "experiments/016_g1_after_feedback/score.py"
    spec = importlib.util.spec_from_file_location("study016_test_score", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config():
    return {
        "schema_version": 2,
        "runtime": {"schema_version": 1, "profile_id": "gmt_g1_four_state_loop_course/v1"},
        "task": {"unchanged": True},
        "reward": {"unchanged": True},
        "oracle": {"unchanged": True},
        "segments": {"unchanged": True},
    }


def test_only_declared_profile_change_is_allowed(scorer):
    control = _config()
    candidate = copy.deepcopy(control)
    candidate["schema_version"] = 4
    candidate["runtime"]["profile_id"] = scorer.PROFILE
    scorer.verify_config(control, candidate)
    for field in ("task", "reward", "oracle", "segments"):
        wrong = copy.deepcopy(candidate)
        wrong[field]["changed"] = True
        with pytest.raises(ValueError, match="more than"):
            scorer.verify_config(control, wrong)


def _run(scorer, *, initial_yaw=0.0):
    qpos = np.zeros((267, 30), dtype=np.float64)
    qpos[:, :2] = [2.0, -3.0]
    qpos[:, 2] = 0.75
    qpos[:, 3] = math.cos(initial_yaw / 2)
    qpos[:, 6] = math.sin(initial_yaw / 2)
    lateral_axis = np.asarray([-math.sin(initial_yaw), math.cos(initial_yaw)])
    qpos[263:, :2] += lateral_axis
    frame = TaskFrame.initialize(qpos[0, :2], qpos[0, 3:7])
    rows = [{"executed_mode": "before"} for _ in range(266)]
    references = np.zeros((266, 30), dtype=np.float32)
    for index in range(263, 266):
        projection = frame.project(qpos[index, :2], qpos[index, 3:7])
        target = max(-0.3, min(0.3, -0.3 * projection.lateral_m))
        error = target - projection.heading_error_rad
        correction = math.atan2(math.sin(error), math.cos(error))
        native = float(np.float32(0.125))
        issued = float(
            np.clip(np.float32(native) + np.float32(correction), np.float32(-0.3), np.float32(0.3))
        )
        references[index, 6] = issued
        rows[index] = {
            "executed_mode": "after",
            scorer.TRACE_KEY: {
                "pre_action_control_step": index,
                "observed_pre_action": {
                    "lateral_m": projection.lateral_m,
                    "heading_error_signed_rad": projection.heading_error_rad,
                },
                "target_heading_rad": target,
                "correction_yaw_rate_rad_s": correction,
                "applied_correction_yaw_rate_rad_s_float32": float(np.float32(correction)),
                "held_poststep_target": {
                    "native_yaw_rate_rad_s": native,
                    "issued_yaw_rate_rad_s": issued,
                    "matches_issued_window_first_row": True,
                },
                "window": {
                    "total_rate_saturation_fraction": 0.5,
                    "native_rate_above_limit_fraction": 0.1,
                },
            },
        }
    return {"frames": rows, "trajectory": {"qpos": qpos, "current_reference": references}}


@pytest.mark.parametrize("initial_yaw", [0.0, 0.7, -2.0])
def test_feedback_uses_rotated_reset_task_frame(scorer, initial_yaw):
    result = scorer.steering_measures(_run(scorer, initial_yaw=initial_yaw))
    assert result["after_action_count"] == 3
    assert result["first_correction_rad_s"] == pytest.approx(-0.3)
    assert result["first_two_seconds_lateral_abs_max_m"] == pytest.approx(1.0)
    assert result["first_two_seconds_boundary_count"] == 4


@pytest.mark.parametrize("mutation", ["sign", "future", "endpoint", "missing", "early"])
def test_steering_rejects_wrong_state_sign_target_or_activation(scorer, mutation):
    run = _run(scorer)
    row = run["frames"][263]
    trace = row[scorer.TRACE_KEY]
    if mutation == "sign":
        trace["correction_yaw_rate_rad_s"] *= -1
    elif mutation == "future":
        trace["observed_pre_action"]["lateral_m"] = 0.9
    elif mutation == "endpoint":
        run["trajectory"]["current_reference"][263, 6] += np.float32(0.01)
    elif mutation == "missing":
        del row[scorer.TRACE_KEY]
    elif mutation == "early":
        run["frames"][262][scorer.TRACE_KEY] = copy.deepcopy(trace)
    with pytest.raises(ValueError):
        scorer.steering_measures(run)


def test_prefix_still_rejects_reward_and_boundary_changes(scorer):
    run = _run(scorer)
    for row in run["frames"]:
        row["reward"] = {"total_reward": 1.0}
    run["trajectory"]["qvel"] = np.zeros((267, 29), dtype=np.float64)
    run["trajectory"]["composite_raw_action"] = np.zeros((266, 23), dtype=np.float32)
    candidate = copy.deepcopy(run)
    scorer.shared.verify_prefix(run, candidate)
    candidate["frames"][100]["reward"]["total_reward"] = 2.0
    with pytest.raises(ValueError, match="complete trace prefix"):
        scorer.shared.verify_prefix(run, candidate)
    candidate = copy.deepcopy(run)
    candidate["trajectory"]["qpos"][263, 0] += 0.01
    with pytest.raises(ValueError, match="numeric prefix"):
        scorer.shared.verify_prefix(run, candidate)


def _objective(scorer):
    return {
        "development_gate_results": {name: False for name in scorer.shared._DEVELOPMENT_GATES},
        "tracking": {"joint_position_rmse_rad_p95": 0.2, "roll_pitch_rmse_rad_p95": 0.2},
        "region": {"entry_observed": True},
        "oracle_diagnostics": {"observed_switch_count": 3},
        "duration_seconds": 20.0,
        "fall_count": 0,
        "maximum_progress_m": 3.5,
        "maximum_lateral_error_m": 2.49,
    }


@pytest.mark.parametrize("heading,passes", [(0.499, True), (0.5, False), (6.28, False)])
def test_heading_gate_is_strict_and_does_not_promote_original_task(scorer, heading, passes):
    objective = _objective(scorer)
    measures = {
        "after_unwrapped_heading_absolute_max_rad": heading,
        "post_after_region_reentry_count": 0,
        "first_rise_action_index": 239,
        "first_after_action_index": 263,
    }
    result = scorer.mechanism_screen(objective, measures)
    assert result["after_unwrapped_heading_below_0p5rad"] is passes
    assert all(result.values()) is passes
    assert not any(objective["development_gate_results"].values())


def test_partial_or_noncanonical_study_inputs_fail_closed(scorer):
    for value in ({}, {"schema_version": True}, [], None):
        with pytest.raises(ValueError, match="input fields"):
            scorer.score_study(value)
