"""Pure observation, history, PD-control, and metric primitives for GMT G1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contracts import (
    ACTION_DIM,
    ACTION_SCALE,
    ANGULAR_VELOCITY_SCALE,
    DAMPING,
    DEFAULT_DOF_POSITION,
    DOF_POSITION_SCALE,
    DOF_VELOCITY_SCALE,
    HISTORY_LENGTH,
    OBSERVATION_DIM,
    PROPRIOCEPTION_DIM,
    RAW_ACTION_MAX,
    RAW_ACTION_MIN,
    REFERENCE_FRAME_DIM,
    REFERENCE_HORIZON,
    STIFFNESS,
    TORQUE_LIMITS,
    ZEROED_DOF_VELOCITY_INDICES,
)
from .reference_math import quaternion_to_euler_wxyz


def _array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, observed {value.shape}")
    if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
        raise ValueError(f"{name} must contain finite floating-point values")
    return value


def build_proprioception(
    *,
    dof_position: np.ndarray,
    dof_velocity: np.ndarray,
    orientation_wxyz: np.ndarray,
    angular_velocity: np.ndarray,
    last_raw_action: np.ndarray,
) -> np.ndarray:
    dof_position = _array(dof_position, (ACTION_DIM,), "dof_position")
    dof_velocity = _array(dof_velocity, (ACTION_DIM,), "dof_velocity")
    orientation_wxyz = _array(orientation_wxyz, (4,), "orientation_wxyz")
    angular_velocity = _array(angular_velocity, (3,), "angular_velocity")
    last_raw_action = _array(last_raw_action, (ACTION_DIM,), "last_raw_action")
    observed_velocity = dof_velocity.copy()
    observed_velocity[list(ZEROED_DOF_VELOCITY_INDICES)] = 0.0
    roll_pitch_yaw = quaternion_to_euler_wxyz(orientation_wxyz)
    proprioception = np.concatenate(
        (
            angular_velocity * ANGULAR_VELOCITY_SCALE,
            roll_pitch_yaw[:2],
            (dof_position - np.asarray(DEFAULT_DOF_POSITION)) * DOF_POSITION_SCALE,
            observed_velocity * DOF_VELOCITY_SCALE,
            last_raw_action,
        )
    )
    if proprioception.shape != (PROPRIOCEPTION_DIM,):
        raise RuntimeError(f"unexpected GMT proprioception shape: {proprioception.shape}")
    return proprioception


class ObservationHistory:
    """The upstream zero-initialized, pre-append 20-frame history contract."""

    def __init__(self) -> None:
        self._values = np.zeros((HISTORY_LENGTH, PROPRIOCEPTION_DIM), dtype=np.float64)

    @property
    def values(self) -> np.ndarray:
        return self._values.copy()

    def assemble(self, reference_window: np.ndarray, proprioception: np.ndarray) -> np.ndarray:
        reference_window = _array(
            reference_window,
            (REFERENCE_HORIZON, REFERENCE_FRAME_DIM),
            "reference_window",
        )
        proprioception = _array(
            proprioception,
            (PROPRIOCEPTION_DIM,),
            "proprioception",
        )
        observation = np.concatenate(
            (reference_window.reshape(-1), proprioception, self._values.reshape(-1))
        ).astype(np.float32)
        if observation.shape != (OBSERVATION_DIM,):
            raise RuntimeError(f"unexpected GMT observation shape: {observation.shape}")
        return observation

    def append(self, proprioception: np.ndarray) -> None:
        proprioception = _array(
            proprioception,
            (PROPRIOCEPTION_DIM,),
            "proprioception",
        )
        self._values[:-1] = self._values[1:]
        self._values[-1] = proprioception


@dataclass(frozen=True)
class ControlOutput:
    raw_history_action: np.ndarray
    clipped_action: np.ndarray
    pd_target: np.ndarray
    torque: np.ndarray


def pd_torque(
    *, pd_target: np.ndarray, dof_position: np.ndarray, dof_velocity: np.ndarray
) -> np.ndarray:
    pd_target = _array(pd_target, (ACTION_DIM,), "pd_target")
    dof_position = _array(dof_position, (ACTION_DIM,), "dof_position")
    dof_velocity = _array(dof_velocity, (ACTION_DIM,), "dof_velocity")
    torque = (pd_target - dof_position) * np.asarray(STIFFNESS) - dof_velocity * np.asarray(DAMPING)
    return np.clip(torque, -np.asarray(TORQUE_LIMITS), np.asarray(TORQUE_LIMITS))


def apply_pd_control(
    *, raw_action: np.ndarray, dof_position: np.ndarray, dof_velocity: np.ndarray
) -> ControlOutput:
    raw_action = _array(raw_action, (ACTION_DIM,), "raw_action")
    dof_position = _array(dof_position, (ACTION_DIM,), "dof_position")
    dof_velocity = _array(dof_velocity, (ACTION_DIM,), "dof_velocity")
    raw_history_action = raw_action.copy()
    clipped_action = np.clip(raw_action, RAW_ACTION_MIN, RAW_ACTION_MAX)
    pd_target = clipped_action * ACTION_SCALE + np.asarray(DEFAULT_DOF_POSITION)
    torque = pd_torque(
        pd_target=pd_target,
        dof_position=dof_position,
        dof_velocity=dof_velocity,
    )
    return ControlOutput(raw_history_action, clipped_action, pd_target, torque)


def reference_tracking_errors(
    *,
    dof_position: np.ndarray,
    root_height: float,
    roll_pitch: np.ndarray,
    current_reference: np.ndarray,
) -> tuple[float, float, float]:
    dof_position = _array(dof_position, (ACTION_DIM,), "dof_position")
    roll_pitch = _array(roll_pitch, (2,), "roll_pitch")
    current_reference = _array(
        current_reference,
        (REFERENCE_FRAME_DIM,),
        "current_reference",
    )
    joint_rmse = float(np.sqrt(np.mean((dof_position - current_reference[7:]) ** 2)))
    height_absolute_error = float(np.abs(root_height - current_reference[0]))
    roll_pitch_rmse = float(np.sqrt(np.mean((roll_pitch - current_reference[1:3]) ** 2)))
    return joint_rmse, height_absolute_error, roll_pitch_rmse
