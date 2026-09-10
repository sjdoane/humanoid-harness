"""Fixed after-state heading feedback and retained-trace verification."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes

from .composition import ComposedReference
from .contracts import ACTION_DIM, CONTROL_DT_SECONDS, REFERENCE_FRAME_DIM, REFERENCE_HORIZON
from .course_task import CourseTaskSpec, FrameProjection, TaskFrame
from .reference_math import quaternion_to_euler_wxyz

if TYPE_CHECKING:
    from .course_config import CourseRunConfig

AFTER_HEADING_FEEDBACK_CONTRACT_ID = "gmt_g1_after_heading_reference_feedback/v1"
AFTER_HEADING_FEEDBACK_TRACE_KEY = "after_heading_reference_feedback"
YAW_RATE_COLUMN = 6
LATERAL_TO_HEADING_GAIN_RAD_PER_M = 0.3
TARGET_HEADING_LIMIT_RAD = 0.3
CORRECTION_TIME_CONSTANT_SECONDS = 1.0
ISSUED_YAW_RATE_LIMIT_RAD_S = 0.3
_ISSUED_YAW_RATE_MIN_F32 = np.float32(-ISSUED_YAW_RATE_LIMIT_RAD_S)
_ISSUED_YAW_RATE_MAX_F32 = np.float32(ISSUED_YAW_RATE_LIMIT_RAD_S)


def after_heading_feedback_contract() -> dict[str, object]:
    """Return the immutable law declared by the opt-in runtime profile."""

    return {
        "schema_version": 1,
        "contract_id": AFTER_HEADING_FEEDBACK_CONTRACT_ID,
        "active_executed_state": "after",
        "measurement_boundary": "pre_action_initial_heading_task_frame",
        "lateral_to_heading_gain_rad_per_m": LATERAL_TO_HEADING_GAIN_RAD_PER_M,
        "target_heading_limit_rad": TARGET_HEADING_LIMIT_RAD,
        "correction_time_constant_seconds": CORRECTION_TIME_CONSTANT_SECONDS,
        "issued_yaw_rate_limit_rad_s": ISSUED_YAW_RATE_LIMIT_RAD_S,
        "reference_yaw_rate_column": YAW_RATE_COLUMN,
        "window_rows": REFERENCE_HORIZON,
        "operation": "clip_total_native_plus_correction",
        "arithmetic": "float64_state_law_float32_applied_correction_and_output",
        "native_rate_above_limit_is_also_clipped": True,
        "poststep_target": "native_current_after_step_with_same_pre_action_correction_and_clip",
        "window_first_row_endpoint_equality": "measured_not_assumed_across_float32_wraps",
        "array_hash": "sha256_canonical_dtype_shape_nul_ordered_c_bytes",
    }


def _float_array(value: object, shape: tuple[int, ...], field: str) -> np.ndarray:
    result = np.asarray(value)
    if (
        result.shape != shape
        or not np.issubdtype(result.dtype, np.floating)
        or not np.isfinite(result).all()
    ):
        raise ValueError(f"{field} must be finite floating-point values with shape {shape}")
    return result


def _reference(value: object, shape: tuple[int, ...], field: str) -> np.ndarray:
    result = _float_array(value, shape, field)
    if result.dtype != np.dtype("<f4") or not result.flags.c_contiguous:
        raise ValueError(f"{field} must use contiguous little-endian float32 storage")
    return result


def _array_sha256(value: np.ndarray) -> str:
    descriptor = canonical_json_bytes({"dtype": value.dtype.str, "shape": list(value.shape)})
    return hashlib.sha256(descriptor + b"\0" + value.tobytes(order="C")).hexdigest()


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


@dataclass(frozen=True, slots=True)
class OracleBoundaryInputs:
    signals: dict[str, float]
    robot_pose: np.ndarray
    projection: FrameProjection


def oracle_boundary_inputs(
    *,
    task: CourseTaskSpec,
    frame: TaskFrame,
    qpos: object,
    qvel: object,
    control_step: int,
) -> OracleBoundaryInputs:
    """Build the oracle's existing boundary inputs from retained plant state."""

    position = _float_array(qpos, (7 + ACTION_DIM,), "boundary qpos")
    velocity = _float_array(qvel, (6 + ACTION_DIM,), "boundary qvel")
    if type(control_step) is not int or control_step < 0:
        raise ValueError("control_step must be a nonnegative integer")
    projection = frame.project(position[:2], position[3:7])
    yaw = frame.forward_yaw_rad
    forward_speed = float(velocity[0] * math.cos(yaw) + velocity[1] * math.sin(yaw))
    inside = (
        task.region_entry_distance_m
        <= projection.progress_m
        < task.region_exit_distance_m
    )
    quaternion = position[3:7]
    signals = {
        "t": control_step * CONTROL_DT_SECONDS,
        "v_x": forward_speed,
        "v_target": (
            task.target_speed_inside_m_s if inside else task.target_speed_outside_m_s
        ),
        "z_root": float(position[2]),
        "torso_up": float(1 - 2 * (quaternion[1] ** 2 + quaternion[2] ** 2)),
        "x_travelled": projection.progress_m,
    }
    robot_pose = np.concatenate(
        (
            position[2:3],
            quaternion_to_euler_wxyz(position[3:7])[:2],
            position[-ACTION_DIM:],
        )
    )
    return OracleBoundaryInputs(signals, robot_pose, projection)


@dataclass(frozen=True, slots=True)
class AfterHeadingFeedbackPlan:
    lateral_m: float
    heading_error_signed_rad: float
    target_heading_rad: float
    correction_yaw_rate_rad_s: float
    applied_correction_yaw_rate_rad_s: float
    native_window: np.ndarray
    issued_window: np.ndarray
    native_rate_above_limit_count: int
    total_rate_saturation_count: int

    @property
    def native_rate_above_limit_fraction(self) -> float:
        return self.native_rate_above_limit_count / REFERENCE_HORIZON

    @property
    def total_rate_saturation_fraction(self) -> float:
        return self.total_rate_saturation_count / REFERENCE_HORIZON

    @property
    def changed_window_row_count(self) -> int:
        return int(
            np.count_nonzero(
                self.native_window[:, YAW_RATE_COLUMN]
                != self.issued_window[:, YAW_RATE_COLUMN]
            )
        )


def plan_after_heading_feedback(
    source_window: object,
    *,
    lateral_m: float,
    heading_error_signed_rad: float,
) -> AfterHeadingFeedbackPlan:
    """Apply the fixed feedback law to only the native yaw-rate channel."""

    source = _reference(
        source_window,
        (REFERENCE_HORIZON, REFERENCE_FRAME_DIM),
        "native reference window",
    )
    if any(type(value) is not float or not math.isfinite(value) for value in (
        lateral_m,
        heading_error_signed_rad,
    )):
        raise ValueError("heading feedback inputs must be finite floats")
    target = float(
        np.clip(
            -LATERAL_TO_HEADING_GAIN_RAD_PER_M * lateral_m,
            -TARGET_HEADING_LIMIT_RAD,
            TARGET_HEADING_LIMIT_RAD,
        )
    )
    correction = _wrap_angle(target - heading_error_signed_rad) / (
        CORRECTION_TIME_CONSTANT_SECONDS
    )
    applied_correction = np.float32(correction)
    native_yaw = source[:, YAW_RATE_COLUMN]
    unclipped = native_yaw + applied_correction
    issued = source.copy()
    issued[:, YAW_RATE_COLUMN] = np.clip(
        unclipped,
        _ISSUED_YAW_RATE_MIN_F32,
        _ISSUED_YAW_RATE_MAX_F32,
    ).astype("<f4")
    issued = np.ascontiguousarray(issued, dtype="<f4")
    native = source.copy()
    native.setflags(write=False)
    issued.setflags(write=False)
    return AfterHeadingFeedbackPlan(
        lateral_m=lateral_m,
        heading_error_signed_rad=heading_error_signed_rad,
        target_heading_rad=target,
        correction_yaw_rate_rad_s=correction,
        applied_correction_yaw_rate_rad_s=float(applied_correction),
        native_window=native,
        issued_window=issued,
        native_rate_above_limit_count=int(
            np.count_nonzero(np.abs(native_yaw) > _ISSUED_YAW_RATE_MAX_F32)
        ),
        total_rate_saturation_count=int(
            np.count_nonzero(np.abs(unclipped) >= _ISSUED_YAW_RATE_MAX_F32)
        ),
    )


def issue_held_poststep_target(
    plan: AfterHeadingFeedbackPlan, source_current: object
) -> np.ndarray:
    """Apply the pre-action plan to the same action's poststep objective target."""

    source = _reference(source_current, (REFERENCE_FRAME_DIM,), "native poststep target")
    issued = source.copy()
    issued[YAW_RATE_COLUMN] = np.float32(
        np.clip(
            source[YAW_RATE_COLUMN] + np.float32(plan.applied_correction_yaw_rate_rad_s),
            _ISSUED_YAW_RATE_MIN_F32,
            _ISSUED_YAW_RATE_MAX_F32,
        )
    )
    return np.ascontiguousarray(issued, dtype="<f4")


def after_heading_feedback_trace(
    *,
    control_step: int,
    plan: AfterHeadingFeedbackPlan,
    source_current: np.ndarray,
    issued_current: np.ndarray,
) -> dict[str, object]:
    """Return compact evidence for one after-state action."""

    source = _reference(source_current, (REFERENCE_FRAME_DIM,), "native poststep target")
    issued = _reference(issued_current, (REFERENCE_FRAME_DIM,), "issued poststep target")
    return {
        "schema_version": 1,
        "contract_id": AFTER_HEADING_FEEDBACK_CONTRACT_ID,
        "pre_action_control_step": control_step,
        "observed_pre_action": {
            "lateral_m": plan.lateral_m,
            "heading_error_signed_rad": plan.heading_error_signed_rad,
        },
        "target_heading_rad": plan.target_heading_rad,
        "correction_yaw_rate_rad_s": plan.correction_yaw_rate_rad_s,
        "applied_correction_yaw_rate_rad_s_float32": (
            plan.applied_correction_yaw_rate_rad_s
        ),
        "window": {
            "native_float32_c_sha256": _array_sha256(plan.native_window),
            "issued_float32_c_sha256": _array_sha256(plan.issued_window),
            "native_rate_above_limit_count": plan.native_rate_above_limit_count,
            "native_rate_above_limit_fraction": plan.native_rate_above_limit_fraction,
            "total_rate_saturation_count": plan.total_rate_saturation_count,
            "total_rate_saturation_fraction": plan.total_rate_saturation_fraction,
            "changed_row_count": plan.changed_window_row_count,
            "changed_value_count": plan.changed_window_row_count,
            "unclipped_total_yaw_rate_rad_s": {
                "minimum": float(
                    np.min(
                        plan.native_window[:, YAW_RATE_COLUMN]
                        + np.float32(plan.applied_correction_yaw_rate_rad_s)
                    )
                ),
                "mean": float(
                    np.mean(
                        plan.native_window[:, YAW_RATE_COLUMN]
                        + np.float32(plan.applied_correction_yaw_rate_rad_s),
                        dtype=np.float64,
                    )
                ),
                "maximum": float(
                    np.max(
                        plan.native_window[:, YAW_RATE_COLUMN]
                        + np.float32(plan.applied_correction_yaw_rate_rad_s)
                    )
                ),
            },
        },
        "held_poststep_target": {
            "native_float32_c_sha256": _array_sha256(source),
            "issued_float32_c_sha256": _array_sha256(issued),
            "native_yaw_rate_rad_s": float(source[YAW_RATE_COLUMN]),
            "issued_yaw_rate_rad_s": float(issued[YAW_RATE_COLUMN]),
            "native_matches_window_first_row": bool(
                np.array_equal(source, plan.native_window[0])
            ),
            "matches_issued_window_first_row": bool(
                np.array_equal(issued, plan.issued_window[0])
            ),
        },
    }


def _same_json(observed: object, expected: object) -> bool:
    try:
        return canonical_json_bytes(observed) == canonical_json_bytes(expected)
    except ValueError:
        return False


def validate_after_heading_feedback_trace(
    *,
    config: CourseRunConfig,
    frames: Sequence[Mapping[str, object]],
    trajectory: Mapping[str, np.ndarray],
) -> dict[str, int]:
    """Rebuild the native oracle and validate every retained steering receipt."""

    enabled = config.runtime.after_heading_reference_feedback
    if not enabled:
        if any(AFTER_HEADING_FEEDBACK_TRACE_KEY in row for row in frames):
            raise ValueError("heading feedback trace is forbidden for this runtime")
        return {
            "after_actions": 0,
            "saturated_window_rows": 0,
            "native_rows_above_limit": 0,
            "changed_window_rows": 0,
        }
    qpos = _float_array(trajectory.get("qpos"), (len(frames) + 1, 7 + ACTION_DIM), "qpos")
    qvel = _float_array(trajectory.get("qvel"), (len(frames) + 1, 6 + ACTION_DIM), "qvel")
    references = _reference(
        trajectory.get("current_reference"),
        (len(frames), REFERENCE_FRAME_DIM),
        "current_reference",
    )
    task_frame = TaskFrame.initialize(qpos[0, :2], qpos[0, 3:7])
    oracle = ComposedReference(config.program, config.segments)
    after_actions = saturated_rows = native_rows = changed_rows = 0
    for index, row in enumerate(frames):
        inputs = oracle_boundary_inputs(
            task=config.task,
            frame=task_frame,
            qpos=qpos[index],
            qvel=qvel[index],
            control_step=index,
        )
        command = oracle.command(
            step=index,
            signals=inputs.signals,
            robot_pose=inputs.robot_pose,
        )
        source_current = oracle.current_after_step(index + 1)
        if (
            row.get("executed_mode") != command.state
            or row.get("executed_behavior") != command.behavior
            or row.get("executed_phase_seconds") != command.phase_seconds
            or not _same_json(row.get("transition"), command.transition)
        ):
            raise ValueError("retained executed oracle command differs from reconstruction")
        observed_trace = row.get(AFTER_HEADING_FEEDBACK_TRACE_KEY)
        if command.state != "after":
            if AFTER_HEADING_FEEDBACK_TRACE_KEY in row:
                raise ValueError("heading feedback trace is present before the after state")
            if not np.array_equal(references[index], source_current):
                raise ValueError("non-after reference differs from the native oracle target")
            continue
        plan = plan_after_heading_feedback(
            command.window,
            lateral_m=inputs.projection.lateral_m,
            heading_error_signed_rad=inputs.projection.heading_error_rad,
        )
        issued_current = issue_held_poststep_target(plan, source_current)
        expected_trace = after_heading_feedback_trace(
            control_step=index,
            plan=plan,
            source_current=source_current,
            issued_current=issued_current,
        )
        if not _same_json(observed_trace, expected_trace):
            raise ValueError("heading feedback trace differs from raw-state reconstruction")
        if not np.array_equal(references[index], issued_current):
            raise ValueError("retained reference differs from reconstructed heading feedback")
        after_actions += 1
        saturated_rows += plan.total_rate_saturation_count
        native_rows += plan.native_rate_above_limit_count
        changed_rows += plan.changed_window_row_count
    if after_actions and changed_rows == 0:
        raise ValueError("heading feedback profile executed after without a numeric manipulation")
    return {
        "after_actions": after_actions,
        "saturated_window_rows": saturated_rows,
        "native_rows_above_limit": native_rows,
        "changed_window_rows": changed_rows,
    }


__all__ = [
    "AFTER_HEADING_FEEDBACK_CONTRACT_ID",
    "AFTER_HEADING_FEEDBACK_TRACE_KEY",
    "AfterHeadingFeedbackPlan",
    "OracleBoundaryInputs",
    "after_heading_feedback_contract",
    "after_heading_feedback_trace",
    "issue_held_poststep_target",
    "oracle_boundary_inputs",
    "plan_after_heading_feedback",
    "validate_after_heading_feedback_trace",
]
