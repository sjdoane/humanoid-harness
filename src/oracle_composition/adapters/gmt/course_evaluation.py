"""Independent full-episode development evaluator for the G1 posture course."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import fields
from statistics import fmean
from typing import Any

from .contracts import CONTROL_DT_SECONDS
from .course_runtime import LEGACY_RUNTIME, CourseRuntimeProfile
from .course_task import CourseStepMetrics, CourseTaskSpec
from .heading_feedback import AFTER_HEADING_FEEDBACK_TRACE_KEY

COURSE_EVALUATOR_ID = "gmt_g1_posture_course_development_evaluator/v1"
DEVELOPMENT_GATE_THRESHOLDS = {
    "minimum_inside_samples": 25,
    "minimum_inside_posture_compliant_fraction": 0.75,
    "maximum_inside_minimum_root_height_m": 0.50,
    "maximum_inside_mean_speed_target_deviation_m_s": 0.10,
    "maximum_mean_speed_error_m_s": 0.35,
    "maximum_lateral_error_m": 0.75,
    "maximum_joint_position_rmse_p95_rad": 0.35,
    "maximum_roll_pitch_rmse_p95_rad": 0.25,
}

_FRAME_FIELDS = {
    "metrics",
    "executed_mode",
    "executed_behavior",
    "executed_phase_seconds",
    "transition",
    "action_saturation_fraction",
    "torque_saturation_fraction",
}
_OPTIONAL_FRAME_FIELDS = {"reward", "trajectory"}
_METRIC_FIELDS = {field.name for field in fields(CourseStepMetrics)}
_METRIC_FLOAT_FIELDS = _METRIC_FIELDS - {
    "task_spec_sha256",
    "control_step",
    "inside_posture_region",
    "posture_success",
    "finish_condition_met",
    "horizon_reached",
    "episode_success",
    "fallen",
    "failure_reasons",
}
_NONNEGATIVE_METRICS = {
    "target_speed_m_s",
    "speed_error_m_s",
    "posture_band_error_m",
    "lateral_error_m",
    "heading_error_rad",
    "joint_position_rmse_rad",
    "root_height_abs_error_m",
    "roll_pitch_rmse_rad",
}
_TRACKING_FIELDS = (
    "joint_position_rmse_rad",
    "root_height_abs_error_m",
    "roll_pitch_rmse_rad",
)
_TRANSITION_PRE_STEPS = 25
_TRANSITION_POST_STEPS = 50


def _finite_tree(value: object) -> bool:
    if value is None or type(value) in {str, bool, int}:
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) in {list, tuple}:
        return all(_finite_tree(item) for item in value)
    if type(value) is dict:
        return all(type(key) is str and _finite_tree(item) for key, item in value.items())
    return False


def _validate_frame(
    frame: object, *, spec: CourseTaskSpec, expected_step: int, runtime: CourseRuntimeProfile
) -> dict[str, Any]:
    if type(frame) is not dict:
        raise ValueError("course evaluation frame fields differ")
    required_fields = _FRAME_FIELDS
    if runtime.after_heading_reference_feedback and frame.get("executed_mode") == "after":
        required_fields = _FRAME_FIELDS | {AFTER_HEADING_FEEDBACK_TRACE_KEY}
    if not required_fields <= set(frame) <= (required_fields | _OPTIONAL_FRAME_FIELDS):
        raise ValueError("course evaluation frame fields differ")
    if AFTER_HEADING_FEEDBACK_TRACE_KEY in frame and (
        type(frame[AFTER_HEADING_FEEDBACK_TRACE_KEY]) is not dict
        or not _finite_tree(frame[AFTER_HEADING_FEEDBACK_TRACE_KEY])
    ):
        raise ValueError("course evaluation heading evidence must be a finite object")
    metrics = frame.get("metrics")
    if type(metrics) is not dict or set(metrics) != _METRIC_FIELDS:
        raise ValueError("course evaluation metric fields differ")
    if metrics["task_spec_sha256"] != spec.sha256:
        raise ValueError("course evaluation task identity differs")
    if type(metrics["control_step"]) is not int or metrics["control_step"] != expected_step:
        raise ValueError("course evaluation steps must be consecutive from 1")
    if any(
        type(metrics[name]) is not float or not math.isfinite(metrics[name])
        for name in _METRIC_FLOAT_FIELDS
    ):
        raise ValueError("course evaluation metrics must be finite floats")
    if any(metrics[name] < 0.0 for name in _NONNEGATIVE_METRICS):
        raise ValueError("course evaluation error metrics must be nonnegative")
    bool_fields = (
        "inside_posture_region",
        "posture_success",
        "finish_condition_met",
        "horizon_reached",
        "fallen",
    )
    if any(type(metrics[name]) is not bool for name in bool_fields):
        raise ValueError("course evaluation flags must be booleans")
    reasons = metrics["failure_reasons"]
    if (
        type(reasons) not in {list, tuple}
        or any(type(reason) is not str or not reason for reason in reasons)
        or metrics["episode_success"] is not None
    ):
        raise ValueError("course evaluation failure or episode-success fields differ")
    inside = spec.region_entry_distance_m <= metrics["progress_m"] < spec.region_exit_distance_m
    target_speed = spec.target_speed_inside_m_s if inside else spec.target_speed_outside_m_s
    posture_error = (
        max(
            spec.posture_band_low_m - metrics["root_height_m"],
            0.0,
            metrics["root_height_m"] - spec.posture_band_high_m,
        )
        if inside
        else 0.0
    )
    if (
        metrics["target_speed_m_s"] != target_speed
        or metrics["speed_error_m_s"] != abs(metrics["forward_speed_m_s"] - target_speed)
        or metrics["posture_band_error_m"] != posture_error
        or metrics["lateral_error_m"] != abs(metrics["lateral_m"])
        or metrics["heading_error_rad"] != abs(metrics["heading_error_signed_rad"])
        or metrics["inside_posture_region"] is not inside
        or metrics["finish_condition_met"] is not (metrics["progress_m"] >= spec.finish_distance_m)
        or metrics["horizon_reached"] is not (expected_step >= spec.horizon_steps)
        or metrics["fallen"] is not bool(reasons)
        or metrics["posture_success"]
        is not (inside and metrics["posture_band_error_m"] == 0.0 and not metrics["fallen"])
    ):
        raise ValueError("course evaluation derived metrics differ")
    for name in ("executed_mode", "executed_behavior"):
        if type(frame[name]) is not str or not frame[name]:
            raise ValueError(f"course evaluation {name} must be nonempty")
    phase = frame["executed_phase_seconds"]
    if type(phase) is not float or not math.isfinite(phase) or phase < 0.0:
        raise ValueError("course evaluation phase must be one nonnegative finite float")
    for name in ("action_saturation_fraction", "torque_saturation_fraction"):
        value = frame[name]
        if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"course evaluation {name} must lie in [0, 1]")
    if frame["transition"] is not None and type(frame["transition"]) is not dict:
        raise ValueError("course evaluation transition must be an object or null")
    if any(name in frame and not _finite_tree(frame[name]) for name in _OPTIONAL_FRAME_FIELDS):
        raise ValueError("course evaluation optional evidence contains a non-finite value")
    if not _finite_tree(frame["transition"]):
        raise ValueError("course evaluation transition contains a non-finite value")
    return frame


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def _tracking_summary(rows: list[dict[str, Any]]) -> dict[str, object]:
    result: dict[str, object] = {"sample_count": len(rows)}
    for field in _TRACKING_FIELDS:
        values = [row["metrics"][field] for row in rows]
        result[f"{field}_mean"] = fmean(values) if values else None
        result[f"{field}_p95"] = _p95(values)
    return result


def evaluate_episode(
    *, spec: CourseTaskSpec, frames: list[dict], runtime: CourseRuntimeProfile = LEGACY_RUNTIME
) -> dict[str, object]:
    """Evaluate one retained development trace; reward values never enter scoring."""

    if type(frames) is not list or not frames or len(frames) > spec.horizon_steps:
        raise ValueError("course episode must contain 1..horizon successive frames")
    rows = [
        _validate_frame(frame, spec=spec, expected_step=index, runtime=runtime)
        for index, frame in enumerate(frames, start=1)
    ]
    if any(row["metrics"]["fallen"] or row["metrics"]["horizon_reached"] for row in rows[:-1]):
        raise ValueError("course evaluation contains frames after a terminal boundary")
    metrics = [row["metrics"] for row in rows]
    inside_rows = [row for row in rows if row["metrics"]["inside_posture_region"]]
    compliant_count = sum(row["metrics"]["posture_success"] for row in inside_rows)
    posture_fraction = compliant_count / len(inside_rows) if inside_rows else None
    entry_observed = bool(inside_rows)
    first_exit_step = None
    seen_inside = False
    for metric in metrics:
        seen_inside |= metric["inside_posture_region"]
        if (
            first_exit_step is None
            and seen_inside
            and metric["progress_m"] >= spec.region_exit_distance_m
        ):
            first_exit_step = metric["control_step"]
    exit_observed = first_exit_step is not None
    transition_markers = [index for index, row in enumerate(rows) if row["transition"] is not None]
    transition_local_indices = {
        index
        for marker in transition_markers
        for index in range(
            max(0, marker - _TRANSITION_PRE_STEPS),
            min(len(rows), marker + _TRANSITION_POST_STEPS + 1),
        )
    }
    transition_rows = [row for index, row in enumerate(rows) if index in transition_local_indices]
    nontransition_rows = [
        row for index, row in enumerate(rows) if index not in transition_local_indices
    ]
    speed_errors = [metric["speed_error_m_s"] for metric in metrics]
    inside_speed_errors = [row["metrics"]["speed_error_m_s"] for row in inside_rows]
    inside_speeds = [row["metrics"]["forward_speed_m_s"] for row in inside_rows]
    inside_mean_speed = fmean(inside_speeds) if inside_speeds else None
    inside_speed_deviation = (
        abs(inside_mean_speed - spec.target_speed_inside_m_s)
        if inside_mean_speed is not None
        else None
    )
    inside_minimum_height = (
        min(row["metrics"]["root_height_m"] for row in inside_rows) if inside_rows else None
    )
    joint_p95 = _p95([metric["joint_position_rmse_rad"] for metric in metrics])
    roll_pitch_p95 = _p95([metric["roll_pitch_rmse_rad"] for metric in metrics])
    fall_count = sum(metric["fallen"] for metric in metrics)
    finish_reached = any(metric["finish_condition_met"] for metric in metrics)
    max_lateral = max(metric["lateral_error_m"] for metric in metrics)
    gates = {
        "full_horizon_without_fall": len(rows) == spec.horizon_steps and fall_count == 0,
        "finish_reached": finish_reached,
        "region_entry_and_exit_observed": entry_observed and exit_observed,
        "minimum_inside_samples": len(inside_rows)
        >= DEVELOPMENT_GATE_THRESHOLDS["minimum_inside_samples"],
        "inside_posture_compliance": posture_fraction is not None
        and posture_fraction
        >= DEVELOPMENT_GATE_THRESHOLDS["minimum_inside_posture_compliant_fraction"],
        "inside_posture_dip": inside_minimum_height is not None
        and inside_minimum_height
        <= DEVELOPMENT_GATE_THRESHOLDS["maximum_inside_minimum_root_height_m"],
        "inside_mean_speed_target": inside_speed_deviation is not None
        and inside_speed_deviation
        <= DEVELOPMENT_GATE_THRESHOLDS["maximum_inside_mean_speed_target_deviation_m_s"],
        "mean_speed_error": fmean(speed_errors)
        <= DEVELOPMENT_GATE_THRESHOLDS["maximum_mean_speed_error_m_s"],
        "maximum_lateral_error": max_lateral
        <= DEVELOPMENT_GATE_THRESHOLDS["maximum_lateral_error_m"],
        "joint_position_rmse_p95": joint_p95
        <= DEVELOPMENT_GATE_THRESHOLDS["maximum_joint_position_rmse_p95_rad"],
        "roll_pitch_rmse_p95": roll_pitch_p95
        <= DEVELOPMENT_GATE_THRESHOLDS["maximum_roll_pitch_rmse_p95_rad"],
    }
    return {
        "evaluator_id": COURSE_EVALUATOR_ID,
        "claim_scope": "fixed_development_gate_only_not_heldout_or_universal",
        "task_sha256": spec.sha256,
        "frame_count": len(rows),
        "duration_seconds": len(rows) * CONTROL_DT_SECONDS,
        "final_progress_m": metrics[-1]["progress_m"],
        "maximum_progress_m": max(metric["progress_m"] for metric in metrics),
        "finish_condition_observed": finish_reached,
        "finish_condition_interpretation": "operational_traversal_not_discriminating_alone",
        "fall_count": fall_count,
        "first_fall_step": next(
            (metric["control_step"] for metric in metrics if metric["fallen"]), None
        ),
        "region": {
            "entry_observed": entry_observed,
            "first_entry_step": inside_rows[0]["metrics"]["control_step"] if inside_rows else None,
            "exit_observed_after_entry": exit_observed,
            "first_exit_step": first_exit_step,
            "inside_sample_count": len(inside_rows),
            "posture_compliant_sample_count": compliant_count,
            "posture_compliant_fraction": posture_fraction,
            "minimum_root_height_m": inside_minimum_height,
        },
        "speed": {
            "mean_absolute_error_m_s": fmean(speed_errors),
            "inside_mean_absolute_error_m_s": fmean(inside_speed_errors)
            if inside_speed_errors
            else None,
            "inside_mean_forward_speed_m_s": inside_mean_speed,
            "inside_target_speed_m_s": spec.target_speed_inside_m_s,
            "inside_mean_speed_target_deviation_m_s": inside_speed_deviation,
        },
        "maximum_lateral_error_m": max_lateral,
        "tracking": _tracking_summary(rows),
        "transition_tracking": {
            "window_semantics": "union_clamped_25_ticks_before_through_50_ticks_after_marker",
            "observed_switch_count": len(transition_markers),
            "transition_local": _tracking_summary(transition_rows),
            "nontransition": _tracking_summary(nontransition_rows),
        },
        "saturation": {
            "action_mean": fmean(row["action_saturation_fraction"] for row in rows),
            "action_max": max(row["action_saturation_fraction"] for row in rows),
            "torque_mean": fmean(row["torque_saturation_fraction"] for row in rows),
            "torque_max": max(row["torque_saturation_fraction"] for row in rows),
        },
        "oracle_diagnostics": {
            "executed_mode_counts": dict(
                sorted(Counter(row["executed_mode"] for row in rows).items())
            ),
            "executed_behavior_counts": dict(
                sorted(Counter(row["executed_behavior"] for row in rows).items())
            ),
            "phase_seconds_min": min(row["executed_phase_seconds"] for row in rows),
            "phase_seconds_max": max(row["executed_phase_seconds"] for row in rows),
            "observed_switch_count": len(transition_markers),
        },
        "trajectory_frame_count": sum("trajectory" in row for row in rows),
        "development_gate_thresholds": dict(DEVELOPMENT_GATE_THRESHOLDS),
        "development_gate_results": gates,
        "development_gate_passed": all(gates.values()),
        "episode_success": None,
    }


__all__ = [
    "COURSE_EVALUATOR_ID",
    "DEVELOPMENT_GATE_THRESHOLDS",
    "evaluate_episode",
]
