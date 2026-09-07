from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np
import pytest

from oracle_composition.adapters.gmt.course_evaluation import evaluate_episode
from oracle_composition.adapters.gmt.course_runtime import (
    AFTER_HEADING_FEEDBACK_RUNTIME,
    FINITE_HORIZON_RUNTIME,
    LEGACY_RUNTIME,
    LOOP_RUNTIME,
)
from oracle_composition.adapters.gmt.course_task import CourseTaskSpec, TaskFrame, evaluate_step
from oracle_composition.adapters.gmt.heading_feedback import AFTER_HEADING_FEEDBACK_TRACE_KEY


def _spec() -> CourseTaskSpec:
    return CourseTaskSpec(
        region_entry_distance_m=0.20,
        region_exit_distance_m=0.55,
        finish_distance_m=1.00,
        target_speed_outside_m_s=1.0,
        target_speed_inside_m_s=0.65,
        posture_band_low_m=0.30,
        posture_band_high_m=0.60,
        horizon_steps=60,
    )


def _qpos(x: float, *, y: float = 0.0, height: float = 0.55) -> np.ndarray:
    value = np.zeros(30, dtype=np.float64)
    value[:3] = (x, y, height)
    value[3] = 1.0
    return value


def _reference() -> np.ndarray:
    value = np.zeros(30, dtype=np.float64)
    value[0] = 0.55
    return value


def _frames(
    spec: CourseTaskSpec | None = None,
    *,
    count: int | None = None,
    fall_step: int | None = None,
) -> list[dict]:
    task = spec or _spec()
    frame = TaskFrame.initialize(np.zeros(2, dtype=np.float64), np.asarray([1.0, 0.0, 0.0, 0.0]))
    rows = []
    progress = 0.0
    height = 0.55
    for step in range(1, (count or task.horizon_steps) + 1):
        before_progress = progress
        before_height = height
        speed = (
            task.target_speed_inside_m_s
            if task.region_entry_distance_m <= progress < task.region_exit_distance_m
            else task.target_speed_outside_m_s
        )
        progress += speed * 0.02
        inside = task.region_entry_distance_m <= progress < task.region_exit_distance_m
        height = 0.48 if inside else 0.55
        fallen = step == fall_step
        metrics = evaluate_step(
            spec=task,
            frame=frame,
            before_qpos=_qpos(before_progress, height=before_height),
            after_qpos=_qpos(progress, height=0.25 if fallen else height),
            ground_contact_bodies=("torso_link",) if fallen else (),
            current_reference=_reference(),
            control_step=step,
        )
        before_inside = (
            task.region_entry_distance_m <= before_progress < task.region_exit_distance_m
        )
        transition = None
        if inside != before_inside:
            transition = {
                "from_state": "posture" if before_inside else "travel",
                "to_state": "posture" if inside else "travel",
                "control_step": step,
            }
        rows.append(
            {
                "metrics": metrics.to_dict(),
                "reward": {"task_reward": float(step), "recipe_marker": 1.0},
                "executed_mode": "posture" if metrics.inside_posture_region else "travel",
                "executed_behavior": "crouch" if metrics.inside_posture_region else "walk",
                "executed_phase_seconds": step * 0.02,
                "transition": transition,
                "action_saturation_fraction": 0.0,
                "torque_saturation_fraction": 0.0,
                "trajectory": {"root_x": progress},
            }
        )
    return rows


def test_fixed_development_gates_pass_on_complete_compliant_trace() -> None:
    result = evaluate_episode(spec=_spec(), frames=_frames())

    assert result["claim_scope"] == "fixed_development_gate_only_not_heldout_or_universal"
    assert result["frame_count"] == 60
    assert result["duration_seconds"] == pytest.approx(1.2)
    assert result["finish_condition_observed"] is True
    assert result["fall_count"] == 0
    assert result["first_fall_step"] is None
    assert result["region"] == {
        "entry_observed": True,
        "first_entry_step": 11,
        "exit_observed_after_entry": True,
        "first_exit_step": 37,
        "inside_sample_count": 26,
        "posture_compliant_sample_count": 26,
        "posture_compliant_fraction": 1.0,
        "minimum_root_height_m": 0.48,
    }
    assert result["speed"]["mean_absolute_error_m_s"] < 0.02
    assert result["speed"]["inside_mean_forward_speed_m_s"] == pytest.approx(0.663, abs=0.001)
    assert result["speed"]["inside_mean_speed_target_deviation_m_s"] < 0.02
    assert result["transition_tracking"]["observed_switch_count"] == 2
    assert result["transition_tracking"]["transition_local"]["sample_count"] == 60
    assert result["transition_tracking"]["nontransition"]["sample_count"] == 0
    assert all(result["development_gate_results"].values())
    assert result["development_gate_passed"] is True
    assert result["episode_success"] is None


def test_heading_trace_schema_is_profile_scoped_and_never_changes_score() -> None:
    rows = _frames()
    rows[-1]["executed_mode"] = "after"
    baseline = evaluate_episode(spec=_spec(), frames=rows)
    rows[-1][AFTER_HEADING_FEEDBACK_TRACE_KEY] = {"diagnostic_only": 1.0}
    assert evaluate_episode(
        spec=_spec(), frames=rows, runtime=AFTER_HEADING_FEEDBACK_RUNTIME
    ) == baseline
    for runtime in (LEGACY_RUNTIME, LOOP_RUNTIME, FINITE_HORIZON_RUNTIME):
        with pytest.raises(ValueError, match="frame fields"):
            evaluate_episode(spec=_spec(), frames=rows, runtime=runtime)
    rows[0][AFTER_HEADING_FEEDBACK_TRACE_KEY] = {}
    with pytest.raises(ValueError, match="frame fields"):
        evaluate_episode(spec=_spec(), frames=rows, runtime=AFTER_HEADING_FEEDBACK_RUNTIME)
    del rows[0][AFTER_HEADING_FEEDBACK_TRACE_KEY]
    rows[-1][AFTER_HEADING_FEEDBACK_TRACE_KEY] = {"bad": float("nan")}
    with pytest.raises(ValueError, match="finite object"):
        evaluate_episode(spec=_spec(), frames=rows, runtime=AFTER_HEADING_FEEDBACK_RUNTIME)
    del rows[-1][AFTER_HEADING_FEEDBACK_TRACE_KEY]
    with pytest.raises(ValueError, match="frame fields"):
        evaluate_episode(spec=_spec(), frames=rows, runtime=AFTER_HEADING_FEEDBACK_RUNTIME)


def test_fall_is_reported_and_cannot_pass_the_development_gate() -> None:
    result = evaluate_episode(spec=_spec(), frames=_frames(count=12, fall_step=12))

    assert result["fall_count"] == 1
    assert result["first_fall_step"] == 12
    assert result["development_gate_results"]["full_horizon_without_fall"] is False
    assert result["development_gate_passed"] is False


def test_missing_region_exposure_is_missing_not_zero_compliance() -> None:
    spec = replace(
        _spec(),
        region_entry_distance_m=2.0,
        region_exit_distance_m=2.5,
        finish_distance_m=3.0,
    )
    result = evaluate_episode(spec=spec, frames=_frames(spec))

    assert result["region"]["inside_sample_count"] == 0
    assert result["region"]["posture_compliant_fraction"] is None
    assert result["speed"]["inside_mean_absolute_error_m_s"] is None
    assert result["development_gate_results"]["region_entry_and_exit_observed"] is False
    assert result["development_gate_results"]["inside_posture_compliance"] is False
    assert result["development_gate_passed"] is False


def test_short_nonterminal_trace_is_valid_evidence_but_fails_full_horizon() -> None:
    result = evaluate_episode(spec=_spec(), frames=_frames(count=30))

    assert result["frame_count"] == 30
    assert result["development_gate_results"]["full_horizon_without_fall"] is False
    assert result["development_gate_results"]["finish_reached"] is False
    assert result["development_gate_passed"] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("metrics", "speed_error_m_s"), math.nan),
        (("action_saturation_fraction",), math.inf),
        (("trajectory", "root_x"), math.nan),
    ],
)
def test_nonfinite_frame_evidence_fails_closed(path: tuple[str, ...], value: float) -> None:
    rows = _frames(count=1)
    target = rows[0]
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value

    with pytest.raises(ValueError, match=r"finite|lie in"):
        evaluate_episode(spec=_spec(), frames=rows)


def test_reward_recipe_payload_does_not_change_independent_measurement() -> None:
    original = _frames()
    changed = copy.deepcopy(original)
    for frame in changed:
        frame["reward"] = {
            "task_reward": -999.0,
            "invented_recipe_sha256": "not-measurement-authority",
        }

    assert evaluate_episode(spec=_spec(), frames=changed) == evaluate_episode(
        spec=_spec(), frames=original
    )


def test_oracle_labels_are_diagnostics_not_gate_authority() -> None:
    first = _frames()
    second = copy.deepcopy(first)
    for frame in second:
        frame["executed_mode"] = "unrelated-mode"
        frame["executed_behavior"] = "unrelated-behavior"
        frame["transition"] = None

    first_result = evaluate_episode(spec=_spec(), frames=first)
    second_result = evaluate_episode(spec=_spec(), frames=second)
    assert first_result["development_gate_results"] == second_result["development_gate_results"]
    assert first_result["oracle_diagnostics"] != second_result["oracle_diagnostics"]
    assert second_result["oracle_diagnostics"]["observed_switch_count"] == 0


def test_posture_dip_and_inside_mean_speed_are_fixed_gates() -> None:
    rows = _frames()
    for frame in rows:
        if frame["metrics"]["inside_posture_region"]:
            frame["metrics"]["root_height_m"] = 0.55
            frame["metrics"]["forward_speed_m_s"] = 0.85
            frame["metrics"]["speed_error_m_s"] = abs(0.85 - _spec().target_speed_inside_m_s)

    result = evaluate_episode(spec=_spec(), frames=rows)

    assert result["region"]["posture_compliant_fraction"] == 1.0
    assert result["development_gate_results"]["inside_posture_dip"] is False
    assert result["development_gate_results"]["inside_mean_speed_target"] is False


def test_wrong_task_identity_and_out_of_order_frames_fail_closed() -> None:
    rows = _frames(count=3)
    with pytest.raises(ValueError, match="task identity"):
        evaluate_episode(spec=replace(_spec(), finish_distance_m=1.1), frames=rows)

    rows[1]["metrics"]["control_step"] = 3
    with pytest.raises(ValueError, match="consecutive"):
        evaluate_episode(spec=_spec(), frames=rows)


def test_fabricated_step_episode_success_is_rejected() -> None:
    rows = _frames(count=1)
    rows[0]["metrics"]["episode_success"] = True

    with pytest.raises(ValueError, match="episode-success"):
        evaluate_episode(spec=_spec(), frames=rows)


def test_derived_task_metric_tampering_is_rejected() -> None:
    rows = _frames(count=1)
    rows[0]["metrics"]["lateral_error_m"] = 0.5

    with pytest.raises(ValueError, match="derived metrics"):
        evaluate_episode(spec=_spec(), frames=rows)


def test_frames_after_a_fall_are_rejected_as_non_gym_evidence() -> None:
    rows = _frames(count=3)
    fallen = _frames(count=2, fall_step=2)[-1]
    rows[1] = fallen

    with pytest.raises(ValueError, match="after a terminal"):
        evaluate_episode(spec=_spec(), frames=rows)


def test_tracking_error_twenty_ticks_after_switch_is_transition_local() -> None:
    rows = _frames()
    for frame in rows:
        frame["transition"] = None
    rows[29]["transition"] = {"control_step": 30}
    rows[49]["metrics"]["joint_position_rmse_rad"] = 0.20

    result = evaluate_episode(spec=_spec(), frames=rows)

    assert result["transition_tracking"]["window_semantics"] == (
        "union_clamped_25_ticks_before_through_50_ticks_after_marker"
    )
    assert result["transition_tracking"]["transition_local"]["sample_count"] == 56
    assert result["transition_tracking"]["nontransition"]["sample_count"] == 4
    assert result["transition_tracking"]["transition_local"]["joint_position_rmse_rad_mean"] > 0.0
    assert result["transition_tracking"]["nontransition"]["joint_position_rmse_rad_mean"] == 0.0
