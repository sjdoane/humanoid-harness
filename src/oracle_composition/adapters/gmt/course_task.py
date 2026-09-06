"""Pure task, metric, and reward contracts for the GMT G1 posture course."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes

from .contracts import ACTION_DIM, CONTROL_DT_SECONDS, REFERENCE_FRAME_DIM
from .deployment import reference_tracking_errors
from .reference_math import quaternion_to_euler_wxyz

COURSE_TASK_SPEC_ID = "gmt_g1_posture_course_task/v1"
TASK_REWARD_RECIPE_ID = "gmt_g1_posture_course_reward/v1"
TASK_REWARD_RECIPE_V2_ID = "gmt_g1_posture_course_reward/v2"
TASK_REWARD_DEPTH_SCALE_M = 0.10
# Exact foot collision-body names in the pinned GMT G1 XML.
ALLOWED_GROUND_CONTACT_BODIES = frozenset({"left_ankle_roll_link", "right_ankle_roll_link"})
ROOT_HEIGHT_FAILURE_M = 0.30
TORSO_UP_FAILURE_MIN = 0.5
TASK_FEATURE_NAMES = (
    "progress_m",
    "lateral_m",
    "heading_error_signed_rad",
    "root_height_m",
    "torso_up",
    "forward_speed_m_s",
    "target_speed_m_s",
    "inside_posture_region",
    "posture_band_low_m",
    "posture_band_high_m",
    "remaining_horizon_fraction",
)

_TASK_SCALES = (0.30, 0.20, 1.0, 0.50)
_TRACKING_SCALES = (0.35, 0.15, 0.35)
_TRACKING_WEIGHTS = (0.50, 0.25, 0.25)
_SPEC_FLOAT_FIELDS = (
    "region_entry_distance_m",
    "region_exit_distance_m",
    "finish_distance_m",
    "target_speed_outside_m_s",
    "target_speed_inside_m_s",
    "posture_band_low_m",
    "posture_band_high_m",
)
_RECIPE_WEIGHT_FIELDS = (
    "speed_weight",
    "posture_weight",
    "lateral_weight",
    "heading_weight",
    "failure_weight",
)


def _strict_float(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be one finite float")
    return value


def _record(
    value: Mapping[str, object],
    *,
    schema_id: str,
    payload_fields: tuple[str, ...],
    schema_version: int = 1,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != {"schema_id", "schema_version", *payload_fields}:
        raise ValueError(f"{schema_id} fields differ")
    if (
        value["schema_id"] != schema_id
        or type(value["schema_version"]) is not int
        or value["schema_version"] != schema_version
    ):
        raise ValueError(f"{schema_id} schema identity differs")
    return value


def _array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    result = np.asarray(value)
    if (
        result.shape != shape
        or not np.issubdtype(result.dtype, np.floating)
        or not np.isfinite(result).all()
    ):
        raise ValueError(f"{name} must be finite floating-point values with shape {shape}")
    return np.ascontiguousarray(result, dtype=np.float64)


def _orientation_euler(orientation_wxyz: object) -> np.ndarray:
    orientation = _array(orientation_wxyz, (4,), "orientation_wxyz")
    if not math.isclose(float(np.linalg.norm(orientation)), 1.0, abs_tol=1e-6):
        raise ValueError("orientation_wxyz must be a unit world quaternion in wxyz order")
    return quaternion_to_euler_wxyz(orientation)


@dataclass(frozen=True, slots=True)
class CourseTaskSpec:
    region_entry_distance_m: float
    region_exit_distance_m: float
    finish_distance_m: float
    target_speed_outside_m_s: float
    target_speed_inside_m_s: float
    posture_band_low_m: float
    posture_band_high_m: float
    horizon_steps: int

    def __post_init__(self) -> None:
        values = (getattr(self, name) for name in _SPEC_FLOAT_FIELDS)
        if any(type(value) is not float or not math.isfinite(value) for value in values):
            raise ValueError("course task continuous fields must be finite floats")
        if not 0.0 < self.region_entry_distance_m < self.region_exit_distance_m:
            raise ValueError("posture region distances must be positive and ordered")
        if not self.region_exit_distance_m < self.finish_distance_m <= 100.0:
            raise ValueError("finish distance must follow the posture region and be at most 100 m")
        if not (
            0.0 <= self.target_speed_outside_m_s <= 5.0
            and 0.0 <= self.target_speed_inside_m_s <= 5.0
        ):
            raise ValueError("target speeds must lie in [0, 5] m/s")
        if not (
            0.0 < self.posture_band_low_m < self.posture_band_high_m <= 2.0
            and self.posture_band_high_m > ROOT_HEIGHT_FAILURE_M
        ):
            raise ValueError("posture height band is invalid or wholly below the failure floor")
        if type(self.horizon_steps) is not int or not 1 <= self.horizon_steps <= 1_000_000:
            raise ValueError("horizon_steps must be an integer in [1, 1000000]")

    def to_dict(self) -> dict[str, object]:
        return {
            **{name: getattr(self, name) for name in _SPEC_FLOAT_FIELDS},
            "horizon_steps": self.horizon_steps,
            "schema_id": COURSE_TASK_SPEC_ID,
            "schema_version": 1,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CourseTaskSpec:
        value = _record(
            value,
            schema_id=COURSE_TASK_SPEC_ID,
            payload_fields=(*_SPEC_FLOAT_FIELDS, "horizon_steps"),
        )
        return cls(
            **{name: _strict_float(value[name], name) for name in _SPEC_FLOAT_FIELDS},
            horizon_steps=value["horizon_steps"],  # type: ignore[arg-type]
        )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


@dataclass(frozen=True, slots=True)
class TaskRewardRecipe:
    speed_weight: float
    posture_weight: float
    lateral_weight: float
    heading_weight: float
    failure_weight: float
    recipe_version: int = 1
    depth_strength: float = 0.0
    ceiling_fraction: float = 0.6

    def __post_init__(self) -> None:
        if any(
            type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 5.0
            for value in (getattr(self, name) for name in _RECIPE_WEIGHT_FIELDS)
        ):
            raise ValueError("task reward weights must be finite floats in [0, 5]")
        if type(self.recipe_version) is not int or self.recipe_version not in {1, 2}:
            raise ValueError("task reward recipe version must be 1 or 2")
        if any(
            type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in (self.depth_strength, self.ceiling_fraction)
        ):
            raise ValueError("task reward depth settings must be finite floats in [0, 1]")
        if self.recipe_version == 1 and (
            self.depth_strength != 0.0 or self.ceiling_fraction != 0.6
        ):
            raise ValueError("task reward v1 cannot carry v2 depth settings")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            **{name: getattr(self, name) for name in _RECIPE_WEIGHT_FIELDS},
            "schema_id": (
                TASK_REWARD_RECIPE_ID
                if self.recipe_version == 1
                else TASK_REWARD_RECIPE_V2_ID
            ),
            "schema_version": self.recipe_version,
        }
        if self.recipe_version == 2:
            result.update(
                depth_strength=self.depth_strength,
                ceiling_fraction=self.ceiling_fraction,
            )
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TaskRewardRecipe:
        if type(value) is not dict:
            raise ValueError("task reward recipe must be an object")
        if value.get("schema_id") == TASK_REWARD_RECIPE_ID:
            parsed = _record(
                value,
                schema_id=TASK_REWARD_RECIPE_ID,
                payload_fields=_RECIPE_WEIGHT_FIELDS,
            )
            return cls(
                **{name: _strict_float(parsed[name], name) for name in _RECIPE_WEIGHT_FIELDS}
            )
        if value.get("schema_id") == TASK_REWARD_RECIPE_V2_ID:
            parsed = _record(
                value,
                schema_id=TASK_REWARD_RECIPE_V2_ID,
                schema_version=2,
                payload_fields=(*_RECIPE_WEIGHT_FIELDS, "depth_strength", "ceiling_fraction"),
            )
            return cls(
                **{name: _strict_float(parsed[name], name) for name in _RECIPE_WEIGHT_FIELDS},
                recipe_version=2,
                depth_strength=_strict_float(parsed["depth_strength"], "depth_strength"),
                ceiling_fraction=_strict_float(parsed["ceiling_fraction"], "ceiling_fraction"),
            )
        raise ValueError("task reward recipe schema identity differs")

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


@dataclass(frozen=True, slots=True)
class FrameProjection:
    progress_m: float
    lateral_m: float
    heading_error_rad: float


@dataclass(frozen=True, slots=True)
class TaskFrame:
    origin_xy: tuple[float, float]
    forward_yaw_rad: float

    @classmethod
    def initialize(cls, root_xy: object, orientation_wxyz: object) -> TaskFrame:
        origin = _array(root_xy, (2,), "root_xy")
        yaw = float(_orientation_euler(orientation_wxyz)[2])
        return cls((float(origin[0]), float(origin[1])), yaw)

    def project(self, root_xy: object, orientation_wxyz: object) -> FrameProjection:
        position = _array(root_xy, (2,), "root_xy")
        yaw = float(_orientation_euler(orientation_wxyz)[2])
        delta = position - np.asarray(self.origin_xy)
        forward = np.asarray([math.cos(self.forward_yaw_rad), math.sin(self.forward_yaw_rad)])
        lateral = np.asarray([-forward[1], forward[0]])
        heading = math.atan2(
            math.sin(yaw - self.forward_yaw_rad), math.cos(yaw - self.forward_yaw_rad)
        )
        return FrameProjection(float(delta @ forward), float(delta @ lateral), heading)


@dataclass(frozen=True, slots=True)
class CourseStepMetrics:
    task_spec_sha256: str
    control_step: int
    progress_m: float
    lateral_m: float
    heading_error_signed_rad: float
    root_height_m: float
    torso_up: float
    forward_speed_m_s: float
    target_speed_m_s: float
    speed_error_m_s: float
    posture_band_error_m: float
    lateral_error_m: float
    heading_error_rad: float
    inside_posture_region: bool
    posture_success: bool
    finish_condition_met: bool
    horizon_reached: bool
    episode_success: None
    fallen: bool
    failure_reasons: tuple[str, ...]
    joint_position_rmse_rad: float
    root_height_abs_error_m: float
    roll_pitch_rmse_rad: float

    def task_features(self, spec: CourseTaskSpec) -> np.ndarray:
        """Return one layout; remaining time is 1 at reset and 0 at the horizon."""

        if spec.sha256 != self.task_spec_sha256:
            raise ValueError("task features require the metric's exact task specification")
        remaining = max(0.0, (spec.horizon_steps - self.control_step) / spec.horizon_steps)
        return np.asarray(
            [
                self.progress_m,
                self.lateral_m,
                self.heading_error_signed_rad,
                self.root_height_m,
                self.torso_up,
                self.forward_speed_m_s,
                self.target_speed_m_s,
                float(self.inside_posture_region),
                spec.posture_band_low_m,
                spec.posture_band_high_m,
                remaining,
            ],
            dtype="<f4",
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_step(
    *,
    spec: CourseTaskSpec,
    frame: TaskFrame,
    before_qpos: object,
    after_qpos: object,
    ground_contact_bodies: Sequence[str],
    current_reference: object,
    control_step: int,
) -> CourseStepMetrics:
    """Measure a boundary; control_step is the number of completed intervals."""

    before = _array(before_qpos, (7 + ACTION_DIM,), "before_qpos")
    after = _array(after_qpos, (7 + ACTION_DIM,), "after_qpos")
    reference = _array(current_reference, (REFERENCE_FRAME_DIM,), "current_reference")
    if type(control_step) is not int or not 0 <= control_step <= spec.horizon_steps:
        raise ValueError("control_step must be an integer boundary inside the task horizon")
    if type(ground_contact_bodies) not in {list, tuple} or any(
        type(name) is not str or not name for name in ground_contact_bodies
    ):
        raise ValueError("ground_contact_bodies must be a list or tuple of body names")
    before_projection = frame.project(before[:2], before[3:7])
    after_projection = frame.project(after[:2], after[3:7])
    roll_pitch = _orientation_euler(after[3:7])[:2]
    torso_up = float(math.cos(float(roll_pitch[0])) * math.cos(float(roll_pitch[1])))
    non_foot_contacts = tuple(
        sorted(set(ground_contact_bodies).difference(ALLOWED_GROUND_CONTACT_BODIES))
    )
    reasons = tuple(
        reason
        for condition, reason in (
            (after[2] < ROOT_HEIGHT_FAILURE_M, "root_height_below_0.30_m"),
            (torso_up < TORSO_UP_FAILURE_MIN, "torso_up_below_0.5"),
            (bool(non_foot_contacts), "non_foot_ground_contact"),
        )
        if condition
    )
    fallen = bool(reasons)
    inside = (
        spec.region_entry_distance_m <= after_projection.progress_m < spec.region_exit_distance_m
    )
    target_speed = spec.target_speed_inside_m_s if inside else spec.target_speed_outside_m_s
    forward_speed = (
        after_projection.progress_m - before_projection.progress_m
    ) / CONTROL_DT_SECONDS
    posture_error = (
        max(
            spec.posture_band_low_m - float(after[2]),
            0.0,
            float(after[2]) - spec.posture_band_high_m,
        )
        if inside
        else 0.0
    )
    joint_error, height_error, roll_pitch_error = reference_tracking_errors(
        dof_position=after[-ACTION_DIM:],
        root_height=float(after[2]),
        roll_pitch=roll_pitch,
        current_reference=reference,
    )
    finish_condition_met = after_projection.progress_m >= spec.finish_distance_m
    horizon_reached = control_step >= spec.horizon_steps
    return CourseStepMetrics(
        task_spec_sha256=spec.sha256,
        control_step=control_step,
        progress_m=after_projection.progress_m,
        lateral_m=after_projection.lateral_m,
        heading_error_signed_rad=after_projection.heading_error_rad,
        root_height_m=float(after[2]),
        torso_up=torso_up,
        forward_speed_m_s=forward_speed,
        target_speed_m_s=target_speed,
        speed_error_m_s=abs(forward_speed - target_speed),
        posture_band_error_m=posture_error,
        lateral_error_m=abs(after_projection.lateral_m),
        heading_error_rad=abs(after_projection.heading_error_rad),
        inside_posture_region=inside,
        posture_success=inside and posture_error == 0.0 and not fallen,
        finish_condition_met=finish_condition_met,
        horizon_reached=horizon_reached,
        episode_success=None,
        fallen=fallen,
        failure_reasons=reasons,
        joint_position_rmse_rad=joint_error,
        root_height_abs_error_m=height_error,
        roll_pitch_rmse_rad=roll_pitch_error,
    )


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    task_spec_sha256: str
    task_reward_recipe_sha256: str
    tracking_reward: float
    task_reward: float
    total_reward: float
    speed_component_reward: float
    posture_component_reward: float
    lateral_component_reward: float
    heading_component_reward: float
    failure_penalty: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _component_reward(error: float, scale: float) -> float:
    return math.exp(-((error / scale) ** 2))


def reward(
    *, spec: CourseTaskSpec, recipe: TaskRewardRecipe, metrics: CourseStepMetrics
) -> RewardBreakdown:
    if metrics.task_spec_sha256 != spec.sha256:
        raise ValueError("reward metrics and task specification identities differ")
    errors = (
        metrics.speed_error_m_s,
        metrics.posture_band_error_m,
        metrics.lateral_error_m,
        metrics.heading_error_rad,
        metrics.joint_position_rmse_rad,
        metrics.root_height_abs_error_m,
        metrics.roll_pitch_rmse_rad,
    )
    if any(type(value) is not float or not math.isfinite(value) or value < 0.0 for value in errors):
        raise ValueError("reward metrics must be finite nonnegative floats")
    if metrics.fallen:
        # A terminal failure cannot harvest positive reward by ending the episode early.
        task_components = (0.0, 0.0, 0.0, 0.0)
        tracking_reward = 0.0
        failure_penalty = -recipe.failure_weight
    else:
        task_components = tuple(
            _component_reward(error, scale)
            for error, scale in zip(errors[:4], _TASK_SCALES, strict=True)
        )
        if (
            recipe.recipe_version == 2
            and recipe.depth_strength > 0.0
            and metrics.inside_posture_region
        ):
            if type(metrics.root_height_m) is not float or not math.isfinite(
                metrics.root_height_m
            ):
                raise ValueError("task reward v2 requires one finite root height")
            ceiling = spec.posture_band_low_m + recipe.ceiling_fraction * (
                spec.posture_band_high_m - spec.posture_band_low_m
            )
            depth_error = max(0.0, metrics.root_height_m - ceiling)
            cauchy = 1.0 / (1.0 + (depth_error / TASK_REWARD_DEPTH_SCALE_M) ** 2)
            depth_multiplier = 1.0 - recipe.depth_strength * (1.0 - cauchy)
            task_components = (
                task_components[0],
                task_components[1] * depth_multiplier,
                task_components[2],
                task_components[3],
            )
        tracking_reward = sum(
            weight * _component_reward(error, scale)
            for error, scale, weight in zip(
                errors[4:], _TRACKING_SCALES, _TRACKING_WEIGHTS, strict=True
            )
        )
        failure_penalty = 0.0
    task_reward = (
        sum(
            weight * component
            for weight, component in zip(
                (
                    recipe.speed_weight,
                    recipe.posture_weight,
                    recipe.lateral_weight,
                    recipe.heading_weight,
                ),
                task_components,
                strict=True,
            )
        )
        + failure_penalty
    )
    return RewardBreakdown(
        task_spec_sha256=spec.sha256,
        task_reward_recipe_sha256=recipe.sha256,
        tracking_reward=tracking_reward,
        task_reward=task_reward,
        total_reward=tracking_reward + task_reward,
        speed_component_reward=task_components[0],
        posture_component_reward=task_components[1],
        lateral_component_reward=task_components[2],
        heading_component_reward=task_components[3],
        failure_penalty=failure_penalty,
    )


__all__ = [
    "ALLOWED_GROUND_CONTACT_BODIES",
    "COURSE_TASK_SPEC_ID",
    "ROOT_HEIGHT_FAILURE_M",
    "TASK_FEATURE_NAMES",
    "TASK_REWARD_DEPTH_SCALE_M",
    "TASK_REWARD_RECIPE_ID",
    "TASK_REWARD_RECIPE_V2_ID",
    "TORSO_UP_FAILURE_MIN",
    "CourseStepMetrics",
    "CourseTaskSpec",
    "FrameProjection",
    "RewardBreakdown",
    "TaskFrame",
    "TaskRewardRecipe",
    "evaluate_step",
    "reward",
]
