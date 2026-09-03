"""Fixed componentized tracking reward with an immutable configuration hash."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import ArrayLike

from oracle_composition.contracts import OracleContractError
from oracle_composition.tracking.humanoid_reference import HumanoidTrackingState


@dataclass(frozen=True, slots=True)
class TrackingRewardConfig:
    """Frozen reward parameters shared by every oracle arm.

    Root pose is a multiplicative gate. This prevents a fallen humanoid with
    reference-matching limb angles from receiving a high tracking score.
    Cauchy root terms retain a bounded recovery signal far from the target;
    Gaussian detail terms preserve tight nominal tracking. Remaining component
    weights form a convex combination before gating.
    """

    root_height_scale_m: float = 0.20
    root_orientation_scale_rad: float = 0.50
    root_linear_velocity_scale_m_s: float = 1.0
    root_angular_velocity_scale_rad_s: float = 2.0
    joint_position_scale_rad: float = 0.35
    joint_velocity_scale_rad_s: float = 2.0
    root_linear_velocity_weight: float = 0.10
    root_angular_velocity_weight: float = 0.10
    joint_position_weight: float = 0.50
    joint_velocity_weight: float = 0.30

    def __post_init__(self) -> None:
        scale_fields = (
            "root_height_scale_m",
            "root_orientation_scale_rad",
            "root_linear_velocity_scale_m_s",
            "root_angular_velocity_scale_rad_s",
            "joint_position_scale_rad",
            "joint_velocity_scale_rad_s",
        )
        weight_fields = (
            "root_linear_velocity_weight",
            "root_angular_velocity_weight",
            "joint_position_weight",
            "joint_velocity_weight",
        )
        for field in scale_fields + weight_fields:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise OracleContractError(f"{field} must be a numeric scalar")
            if not math.isfinite(float(value)):
                raise OracleContractError(f"{field} must be finite")
        if any(getattr(self, field) <= 0.0 for field in scale_fields):
            raise OracleContractError("tracking reward scales must be positive")
        if any(getattr(self, field) < 0.0 for field in weight_fields):
            raise OracleContractError("tracking reward weights must be non-negative")
        if not math.isclose(
            sum(getattr(self, field) for field in weight_fields),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise OracleContractError("tracking reward weights must sum to one")

    def to_dict(self) -> dict[str, float | str]:
        return {"reward_id": "humanoid_root_and_joint_tracking/v1", **asdict(self)}

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class TrackingRewardResult:
    """Reward components and unit-bearing errors for one transition."""

    total: float
    root_pose_gate: float
    root_height_score: float
    root_orientation_score: float
    root_linear_velocity_score: float
    root_angular_velocity_score: float
    joint_position_score: float
    joint_velocity_score: float
    root_height_abs_error_m: float
    root_orientation_error_rad: float
    root_linear_velocity_rmse_m_s: float
    root_angular_velocity_rmse_rad_s: float
    joint_position_rmse_rad: float
    joint_velocity_rmse_rad_s: float
    joint_position_max_abs_rad: float
    joint_velocity_max_abs_rad_s: float

    def reward_components(self) -> dict[str, float]:
        return {
            "root_pose_gate": self.root_pose_gate,
            "root_height": self.root_height_score,
            "root_orientation": self.root_orientation_score,
            "root_linear_velocity": self.root_linear_velocity_score,
            "root_angular_velocity": self.root_angular_velocity_score,
            "joint_position": self.joint_position_score,
            "joint_velocity": self.joint_velocity_score,
            "total": self.total,
        }

    def error_components(self) -> dict[str, float]:
        return {
            "root_height_abs_error_m": self.root_height_abs_error_m,
            "root_orientation_error_rad": self.root_orientation_error_rad,
            "root_linear_velocity_rmse_m_s": self.root_linear_velocity_rmse_m_s,
            "root_angular_velocity_rmse_rad_s": self.root_angular_velocity_rmse_rad_s,
            "joint_position_rmse_rad": self.joint_position_rmse_rad,
            "joint_velocity_rmse_rad_s": self.joint_velocity_rmse_rad_s,
            "joint_position_max_abs_rad": self.joint_position_max_abs_rad,
            "joint_velocity_max_abs_rad_s": self.joint_velocity_max_abs_rad_s,
        }


def _vector(value: ArrayLike, *, field: str, width: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (width,):
        raise OracleContractError(f"{field} must have shape ({width},)")
    if not np.isfinite(array).all():
        raise OracleContractError(f"{field} must contain only finite values")
    return array


def _unit_quaternion(value: ArrayLike, *, field: str) -> np.ndarray:
    quaternion = _vector(value, field=field, width=4)
    if not math.isclose(
        float(np.linalg.norm(quaternion)),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise OracleContractError(f"{field} must be a unit quaternion")
    return quaternion


def _rmse(error: np.ndarray) -> float:
    return math.sqrt(float(np.mean(np.square(error))))


def _radial_score(error: float, scale: float) -> float:
    return math.exp(-((error / scale) ** 2))


def _cauchy_score(error: float, scale: float) -> float:
    ratio = error / scale
    return 1.0 / (1.0 + ratio * ratio)


def compute_tracking_reward(
    *,
    state: HumanoidTrackingState,
    reference_frame: ArrayLike,
    config: TrackingRewardConfig | None = None,
) -> TrackingRewardResult:
    """Score one root-and-joint state against one 45D reference frame."""

    selected = config or TrackingRewardConfig()
    reference = _vector(reference_frame, field="reference_frame", width=45)
    root_height = float(state.root_height_m)
    if not math.isfinite(root_height):
        raise OracleContractError("root_height_m must be finite")
    root_orientation = _unit_quaternion(
        state.root_orientation_wxyz,
        field="root_orientation_wxyz",
    )
    target_orientation = _unit_quaternion(
        reference[1:5],
        field="reference root orientation",
    )
    root_linear_velocity = _vector(
        state.root_linear_velocity_world_m_s,
        field="root_linear_velocity_world_m_s",
        width=3,
    )
    root_angular_velocity = _vector(
        state.root_angular_velocity_body_rad_s,
        field="root_angular_velocity_body_rad_s",
        width=3,
    )
    positions = _vector(
        state.joint_positions_rad,
        field="joint_positions_rad",
        width=17,
    )
    velocities = _vector(
        state.joint_velocities_rad_s,
        field="joint_velocities_rad_s",
        width=17,
    )

    height_error = abs(root_height - float(reference[0]))
    # q and -q encode the same rotation. The absolute dot product yields the
    # shortest sign-invariant angular distance on SO(3).
    orientation_dot = float(np.clip(abs(np.dot(root_orientation, target_orientation)), 0, 1))
    orientation_error = 2.0 * math.acos(orientation_dot)
    linear_velocity_error = root_linear_velocity - reference[5:8]
    angular_velocity_error = root_angular_velocity - reference[8:11]
    position_error = positions - reference[11:28]
    velocity_error = velocities - reference[28:45]

    linear_velocity_rmse = _rmse(linear_velocity_error)
    angular_velocity_rmse = _rmse(angular_velocity_error)
    position_rmse = _rmse(position_error)
    velocity_rmse = _rmse(velocity_error)
    height_score = _cauchy_score(height_error, selected.root_height_scale_m)
    orientation_score = _cauchy_score(
        orientation_error,
        selected.root_orientation_scale_rad,
    )
    linear_velocity_score = _radial_score(
        linear_velocity_rmse,
        selected.root_linear_velocity_scale_m_s,
    )
    angular_velocity_score = _radial_score(
        angular_velocity_rmse,
        selected.root_angular_velocity_scale_rad_s,
    )
    position_score = _radial_score(position_rmse, selected.joint_position_scale_rad)
    velocity_score = _radial_score(velocity_rmse, selected.joint_velocity_scale_rad_s)

    # Orientation is squared so a sideways torso cannot earn material reward
    # merely by matching joints at the target height. The Cauchy tail remains
    # nonzero for recovery ranking.
    root_pose_gate = height_score * orientation_score * orientation_score
    detail_score = (
        selected.root_linear_velocity_weight * linear_velocity_score
        + selected.root_angular_velocity_weight * angular_velocity_score
        + selected.joint_position_weight * position_score
        + selected.joint_velocity_weight * velocity_score
    )
    result = TrackingRewardResult(
        total=float(root_pose_gate * detail_score),
        root_pose_gate=float(root_pose_gate),
        root_height_score=float(height_score),
        root_orientation_score=float(orientation_score),
        root_linear_velocity_score=float(linear_velocity_score),
        root_angular_velocity_score=float(angular_velocity_score),
        joint_position_score=float(position_score),
        joint_velocity_score=float(velocity_score),
        root_height_abs_error_m=float(height_error),
        root_orientation_error_rad=float(orientation_error),
        root_linear_velocity_rmse_m_s=float(linear_velocity_rmse),
        root_angular_velocity_rmse_rad_s=float(angular_velocity_rmse),
        joint_position_rmse_rad=float(position_rmse),
        joint_velocity_rmse_rad_s=float(velocity_rmse),
        joint_position_max_abs_rad=float(np.max(np.abs(position_error))),
        joint_velocity_max_abs_rad_s=float(np.max(np.abs(velocity_error))),
    )
    if not all(math.isfinite(value) for value in asdict(result).values()):
        raise OracleContractError("tracking reward produced non-finite telemetry")
    return result
