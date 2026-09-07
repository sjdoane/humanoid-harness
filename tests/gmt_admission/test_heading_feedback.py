from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from oracle_composition.adapters.gmt.composition import ReferenceSegment
from oracle_composition.adapters.gmt.contracts import (
    CONTROL_DT_SECONDS,
    DEFAULT_DOF_POSITION,
    REFERENCE_OFFSETS,
)
from oracle_composition.adapters.gmt.course_task import CourseTaskSpec, TaskFrame
from oracle_composition.adapters.gmt.heading_feedback import (
    ISSUED_YAW_RATE_LIMIT_RAD_S,
    YAW_RATE_COLUMN,
    after_heading_feedback_trace,
    issue_held_poststep_target,
    oracle_boundary_inputs,
    plan_after_heading_feedback,
)
from oracle_composition.adapters.gmt.reference_runtime import ReferenceMotion


def _task() -> CourseTaskSpec:
    return CourseTaskSpec(
        region_entry_distance_m=1.0,
        region_exit_distance_m=2.0,
        finish_distance_m=3.0,
        target_speed_outside_m_s=1.0,
        target_speed_inside_m_s=0.5,
        posture_band_low_m=0.45,
        posture_band_high_m=0.85,
        horizon_steps=1_000,
    )


def _window(yaw_rates: np.ndarray | None = None) -> np.ndarray:
    result = np.arange(20 * 30, dtype="<f4").reshape(20, 30) / np.float32(1_000)
    result[:, YAW_RATE_COLUMN] = (
        np.linspace(-0.36, 0.36, 20, dtype="<f4")
        if yaw_rates is None
        else yaw_rates
    )
    return np.ascontiguousarray(result, dtype="<f4")


def test_positive_lateral_error_commands_negative_heading_and_only_mutates_yaw() -> None:
    native = _window()
    plan = plan_after_heading_feedback(
        native,
        lateral_m=1.0,
        heading_error_signed_rad=0.0,
    )

    assert plan.target_heading_rad == -0.3
    assert plan.correction_yaw_rate_rad_s == pytest.approx(-0.3)
    assert plan.applied_correction_yaw_rate_rad_s == float(np.float32(-0.3))
    assert plan.issued_window.dtype == np.dtype("<f4")
    np.testing.assert_array_equal(
        plan.issued_window[:, :YAW_RATE_COLUMN], native[:, :YAW_RATE_COLUMN]
    )
    np.testing.assert_array_equal(
        plan.issued_window[:, YAW_RATE_COLUMN + 1 :], native[:, YAW_RATE_COLUMN + 1 :]
    )
    expected = np.clip(
        native[:, YAW_RATE_COLUMN] + np.float32(-0.3),
        -np.float32(ISSUED_YAW_RATE_LIMIT_RAD_S),
        np.float32(ISSUED_YAW_RATE_LIMIT_RAD_S),
    )
    np.testing.assert_array_equal(plan.issued_window[:, YAW_RATE_COLUMN], expected)
    assert plan.native_rate_above_limit_count == 4
    assert plan.total_rate_saturation_count > 0


def test_target_wrap_clamp_and_rotated_task_frame_keep_declared_units_and_sign() -> None:
    half = math.pi / 4
    initial_qpos = np.zeros(30, dtype="<f8")
    initial_qpos[2] = 0.8
    initial_qpos[3:7] = (math.cos(half), 0.0, 0.0, math.sin(half))
    frame = TaskFrame.initialize(initial_qpos[:2], initial_qpos[3:7])
    boundary = initial_qpos.copy()
    boundary[:2] = (-1.0, 2.0)
    heading = math.pi / 2 + 0.2
    boundary[3:7] = (math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2))
    inputs = oracle_boundary_inputs(
        task=_task(),
        frame=frame,
        qpos=boundary,
        qvel=np.zeros(29, dtype="<f8"),
        control_step=7,
    )
    plan = plan_after_heading_feedback(
        _window(),
        lateral_m=inputs.projection.lateral_m,
        heading_error_signed_rad=inputs.projection.heading_error_rad,
    )

    assert inputs.projection.lateral_m == pytest.approx(1.0)
    assert inputs.projection.heading_error_rad == pytest.approx(0.2)
    assert plan.target_heading_rad == -0.3
    assert plan.correction_yaw_rate_rad_s == pytest.approx(-0.5)

    wrapped = plan_after_heading_feedback(
        _window(),
        lateral_m=-100.0,
        heading_error_signed_rad=-math.pi + 0.1,
    )
    assert wrapped.target_heading_rad == 0.3
    assert wrapped.correction_yaw_rate_rad_s == pytest.approx(-math.pi + 0.2)


def test_held_poststep_target_uses_same_applied_correction_and_first_window_row() -> None:
    native = _window()
    plan = plan_after_heading_feedback(
        native,
        lateral_m=0.4,
        heading_error_signed_rad=0.1,
    )
    issued = issue_held_poststep_target(plan, native[0].copy())
    trace = after_heading_feedback_trace(
        control_step=263,
        plan=plan,
        source_current=native[0].copy(),
        issued_current=issued,
    )

    np.testing.assert_array_equal(issued, plan.issued_window[0])
    assert trace["pre_action_control_step"] == 263
    assert trace["held_poststep_target"]["matches_issued_window_first_row"] is True
    assert trace["window"]["native_rate_above_limit_count"] == 4
    assert trace["window"]["total_rate_saturation_count"] > 0

    changed = native[0].copy()
    changed[0] += np.float32(1.0e-3)
    changed_issued = issue_held_poststep_target(plan, changed)
    assert changed_issued[0] == changed[0]
    assert changed_issued[YAW_RATE_COLUMN] == issued[YAW_RATE_COLUMN]


def test_wrap_boundary_retains_native_endpoint_clock_and_holds_same_correction() -> None:
    root = np.zeros((31, 3), dtype="<f4")
    root[:, 0] = np.arange(31, dtype="<f4")
    root[:, 2] = np.float32(0.8)
    rotation = np.zeros((31, 4), dtype="<f4")
    rotation[:, 3] = np.float32(1.0)
    motion = ReferenceMotion(
        {
            "fps": np.asarray([30.0], dtype="<f8"),
            "root_pos": root,
            "root_rot": rotation,
            "dof_pos": np.broadcast_to(
                np.asarray(DEFAULT_DOF_POSITION, dtype="<f4"), (31, 23)
            ).copy(),
        }
    )
    segment = ReferenceSegment(motion, "a" * 64, 0.1, 0.2)
    phase = torch.tensor(4 * CONTROL_DT_SECONDS, dtype=torch.float32)
    offsets = torch.tensor(REFERENCE_OFFSETS, dtype=torch.float32) * CONTROL_DT_SECONDS
    native_window = segment.features(phase + offsets).numpy().copy()
    native_endpoint = segment.features(
        torch.tensor([5 * CONTROL_DT_SECONDS], dtype=torch.float32)
    )[0].numpy().copy()
    assert np.flatnonzero(native_endpoint != native_window[0]).tolist() == [3]
    assert native_endpoint[3] == np.float32(20.52631378173828)
    assert native_window[0, 3] == np.float32(23.684207916259766)

    plan = plan_after_heading_feedback(
        native_window,
        lateral_m=0.5,
        heading_error_signed_rad=0.1,
    )
    issued_endpoint = issue_held_poststep_target(plan, native_endpoint)
    trace = after_heading_feedback_trace(
        control_step=4,
        plan=plan,
        source_current=native_endpoint,
        issued_current=issued_endpoint,
    )

    assert trace["held_poststep_target"]["native_matches_window_first_row"] is False
    assert trace["held_poststep_target"]["matches_issued_window_first_row"] is False
    np.testing.assert_array_equal(
        issued_endpoint[np.arange(30) != YAW_RATE_COLUMN],
        native_endpoint[np.arange(30) != YAW_RATE_COLUMN],
    )
    expected_yaw = np.float32(
        np.clip(
            native_endpoint[YAW_RATE_COLUMN]
            + np.float32(plan.applied_correction_yaw_rate_rad_s),
            np.float32(-ISSUED_YAW_RATE_LIMIT_RAD_S),
            np.float32(ISSUED_YAW_RATE_LIMIT_RAD_S),
        )
    )
    assert issued_endpoint[YAW_RATE_COLUMN] == expected_yaw


@pytest.mark.parametrize(
    ("source", "lateral", "heading"),
    [
        (np.zeros((20, 30), dtype="<f8"), 0.0, 0.0),
        (np.zeros((19, 30), dtype="<f4"), 0.0, 0.0),
        (np.zeros((20, 30), dtype="<f4"), float("nan"), 0.0),
        (np.zeros((20, 30), dtype="<f4"), 0.0, True),
    ],
)
def test_feedback_law_rejects_noncanonical_inputs(
    source: np.ndarray, lateral: float, heading: float
) -> None:
    with pytest.raises(ValueError):
        plan_after_heading_feedback(
            source,
            lateral_m=lateral,
            heading_error_signed_rad=heading,
        )
