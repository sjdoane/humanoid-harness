"""Reward-independent metrics for the frozen Humanoid tracker baseline."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .fixed_reference import ExperimentContractError

REFERENCE_WIDTH = 45
FLOOR_GEOM_NAME = "floor"
PERMITTED_STATIC_STAND_FLOOR_GEOMS = frozenset({"left_foot", "right_foot"})
STATION_KEEPING_ORIGIN_SOURCE = "direct_mujoco_qpos_xyz_at_seeded_reset/v1"
ACTUATED_SCALAR_COUNT = 17
SUMMARY_ALGEBRA_REL_TOLERANCE = 1e-12
SUMMARY_ALGEBRA_ABS_TOLERANCE = 1e-12

# The policy emits float32 normalized actions while MuJoCo exposes float64
# post-transmission torque.  On the frozen Humanoid-v5 ABI, the largest
# observed conversion difference over direct endpoint and randomized checks is
# one float32 half-ULP (2.9802322387695312e-08).  Keep the admitted tolerance
# explicit and smaller than four such half-ULPs.
ACTION_TORQUE_NORMALIZATION_ABS_TOLERANCE = 1e-7
FROZEN_TORQUE_CAPACITY_MIN_N_M = 10.0
FROZEN_TORQUE_CAPACITY_MAX_N_M = 120.0
RAW_TORQUE_SUMMARY_ABS_TOLERANCE_N_M = (
    FROZEN_TORQUE_CAPACITY_MAX_N_M * ACTION_TORQUE_NORMALIZATION_ABS_TOLERANCE
)


def protected_evaluator_sha256() -> str:
    """Return the hash of the exact evaluator source consumed by a run."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _finite_vector(value: ArrayLike, *, width: int, field: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExperimentContractError(f"{field} must be a numeric vector") from exc
    if array.shape != (width,):
        raise ExperimentContractError(f"{field} must have shape ({width},)")
    if not np.isfinite(array).all():
        raise ExperimentContractError(f"{field} must contain only finite values")
    return array


def _finite_scalar(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ExperimentContractError(f"{field} must be a finite number")
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExperimentContractError(f"{field} must be a finite number") from exc
    if not math.isfinite(resolved):
        raise ExperimentContractError(f"{field} must be a finite number")
    return resolved


def _positive_scalar(value: object, *, field: str) -> float:
    resolved = _finite_scalar(value, field=field)
    if resolved <= 0.0:
        raise ExperimentContractError(f"{field} must be positive")
    return resolved


def _normalized_action(value: ArrayLike, *, field: str) -> np.ndarray:
    action = _finite_vector(value, width=17, field=field)
    if np.any(action < -1.0) or np.any(action > 1.0):
        raise ExperimentContractError(f"{field} must be in [-1, 1]")
    return action


def _positive_vector(value: ArrayLike, *, width: int, field: str) -> np.ndarray:
    vector = _finite_vector(value, width=width, field=field)
    if np.any(vector <= 0.0):
        raise ExperimentContractError(f"{field} must contain only positive values")
    return vector


def _unit_quaternion(value: ArrayLike, *, field: str) -> np.ndarray:
    quaternion = _finite_vector(value, width=4, field=field)
    if not math.isclose(float(np.linalg.norm(quaternion)), 1.0, abs_tol=1e-6, rel_tol=0.0):
        raise ExperimentContractError(f"{field} must be a unit quaternion")
    return quaternion


def _rms_and_max_abs(values: np.ndarray, *, field: str) -> tuple[float, float]:
    """Return finite RMS/max without overflowing on large finite inputs."""

    if values.size < 1 or not np.isfinite(values).all():
        raise ExperimentContractError(f"derived {field} must contain finite values")
    maximum = float(np.max(np.abs(values)))
    if maximum == 0.0:
        return 0.0, 0.0
    scaled = values / maximum
    rms = maximum * math.sqrt(float(np.mean(np.square(scaled))))
    if not math.isfinite(rms):
        raise ExperimentContractError(f"derived {field} must be finite")
    return rms, maximum


def _rmse(error: np.ndarray, *, field: str) -> float:
    return _rms_and_max_abs(error, field=field)[0]


def _validate_rms_max_summary(
    *,
    rms: float,
    maximum: float,
    sample_count: int,
    field: str,
) -> None:
    """Reject an RMS/max pair impossible for its declared scalar sample count."""

    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count < 1:
        raise ExperimentContractError(f"{field} sample_count must be positive")
    tolerance = SUMMARY_ALGEBRA_ABS_TOLERANCE + SUMMARY_ALGEBRA_REL_TOLERANCE * max(
        abs(rms),
        abs(maximum),
    )
    if rms > maximum + tolerance:
        raise ExperimentContractError(f"{field} RMS cannot exceed its maximum")
    if rms < maximum / math.sqrt(sample_count) - tolerance:
        raise ExperimentContractError(
            f"{field} RMS is too small for its maximum and scalar sample count"
        )


def _validate_frozen_torque_summary(
    *,
    normalized_action_rms: float,
    normalized_action_max_abs: float,
    raw_torque_rms_n_m: float,
    raw_torque_max_abs_n_m: float,
    normalized_torque_rms: float,
    normalized_torque_max_abs: float,
) -> None:
    """Reject summaries impossible under the frozen direct-actuator ABI."""

    for action_value, torque_value, label in (
        (normalized_action_rms, normalized_torque_rms, "RMS"),
        (normalized_action_max_abs, normalized_torque_max_abs, "maximum"),
    ):
        if not math.isclose(
            action_value,
            torque_value,
            rel_tol=0.0,
            abs_tol=ACTION_TORQUE_NORMALIZATION_ABS_TOLERANCE,
        ):
            raise ExperimentContractError(
                f"capacity-normalized torque {label} must match normalized policy "
                f"action {label} on the frozen actuator ABI"
            )

    for raw_value, normalized_value, label in (
        (raw_torque_rms_n_m, normalized_torque_rms, "RMS"),
        (raw_torque_max_abs_n_m, normalized_torque_max_abs, "maximum"),
    ):
        minimum = FROZEN_TORQUE_CAPACITY_MIN_N_M * normalized_value
        maximum = FROZEN_TORQUE_CAPACITY_MAX_N_M * normalized_value
        if raw_value < minimum - RAW_TORQUE_SUMMARY_ABS_TOLERANCE_N_M or raw_value > (
            maximum + RAW_TORQUE_SUMMARY_ABS_TOLERANCE_N_M
        ):
            raise ExperimentContractError(
                f"raw generalized actuator torque {label} is outside the frozen capacity envelope"
            )


@dataclass(frozen=True, slots=True)
class ContactFact:
    """One caller-supplied active MuJoCo contact at a simulator step.

    Callers must provide every active contact in simulator index order.
    ``normal_force_n`` is the nonnegative magnitude returned for that contact.
    Floor and forbidden classifications are checked against the exact static
    stand geom names rather than trusted as free booleans.
    """

    contact_index: int
    geom1_name: str
    geom2_name: str
    involves_floor: bool
    forbidden_floor_contact: bool
    normal_force_n: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.contact_index, int)
            or isinstance(self.contact_index, bool)
            or self.contact_index < 0
        ):
            raise ExperimentContractError("contact_index must be an integer >= 0")
        for field in ("geom1_name", "geom2_name"):
            value = getattr(self, field)
            if not isinstance(value, str) or not value.strip() or len(value) > 255:
                raise ExperimentContractError(f"{field} must be a nonempty bounded string")
        if not isinstance(self.involves_floor, bool):
            raise ExperimentContractError("involves_floor must be boolean")
        if not isinstance(self.forbidden_floor_contact, bool):
            raise ExperimentContractError("forbidden_floor_contact must be boolean")
        floor_occurrences = (self.geom1_name, self.geom2_name).count(FLOOR_GEOM_NAME)
        if floor_occurrences > 1:
            raise ExperimentContractError("a contact cannot contain the floor geom twice")
        observed_floor_contact = floor_occurrences == 1
        if self.involves_floor is not observed_floor_contact:
            raise ExperimentContractError("involves_floor must match the exact floor geom identity")
        other_geom = self.geom2_name if self.geom1_name == FLOOR_GEOM_NAME else self.geom1_name
        expected_forbidden = observed_floor_contact and (
            other_geom not in PERMITTED_STATIC_STAND_FLOOR_GEOMS
        )
        if self.forbidden_floor_contact is not expected_forbidden:
            raise ExperimentContractError(
                "forbidden_floor_contact must match the frozen static-stand geom set"
            )
        force = _finite_scalar(self.normal_force_n, field="normal_force_n")
        if force < 0.0:
            raise ExperimentContractError("normal_force_n cannot be negative")
        object.__setattr__(self, "normal_force_n", force)


@dataclass(frozen=True, slots=True)
class _CollapseDefinition:
    """Source-frozen physical guardrail, independent of the learned reward."""

    min_root_height_m: float = 1.0
    max_root_height_m: float = 2.0
    min_torso_up_z: float = 0.5

    def __post_init__(self) -> None:
        values = (
            self.min_root_height_m,
            self.max_root_height_m,
            self.min_torso_up_z,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ExperimentContractError("collapse thresholds must be finite numbers")
        if self.min_root_height_m >= self.max_root_height_m:
            raise ExperimentContractError("collapse height bounds must be ordered")
        if not -1.0 <= self.min_torso_up_z <= 1.0:
            raise ExperimentContractError("min_torso_up_z must be in [-1, 1]")


_FROZEN_COLLAPSE_DEFINITION = _CollapseDefinition()


@dataclass(frozen=True, slots=True)
class StepMetrics:
    """Independent measurements for one simulator transition.

    Contact counts are active contact samples at this transition, not inferred
    collision events. Contact peak fields retain allowed-foot impacts as well
    as forbidden-contact impacts.
    """

    root_position_world_m: tuple[float, float, float]
    root_linear_velocity_world_m_s: tuple[float, float, float]
    root_height_abs_error_m: float
    root_orientation_error_rad: float
    root_linear_velocity_rmse_m_s: float
    root_angular_velocity_rmse_rad_s: float
    joint_position_rmse_rad: float
    joint_velocity_rmse_rad_s: float
    joint_position_max_abs_rad: float
    joint_velocity_max_abs_rad_s: float
    normalized_policy_action_rms: float
    normalized_policy_action_max_abs: float
    normalized_policy_action_delta_rate_rms_per_s: float
    normalized_policy_action_delta_rate_max_abs_per_s: float
    generalized_actuator_torque_rms_n_m: float
    generalized_actuator_torque_max_abs_n_m: float
    generalized_actuator_torque_normalized_rms: float
    generalized_actuator_torque_normalized_max_abs: float
    joint_jerk_rms_rad_s3: float
    joint_jerk_max_abs_rad_s3: float
    simulator_contact_count: int
    simulator_contact_peak_normal_force_n: float
    floor_contact_count: int
    floor_contact_peak_normal_force_n: float
    forbidden_floor_contact_count: int
    forbidden_floor_contact_fraction: float
    forbidden_floor_contact_peak_force_n: float
    control_period_seconds: float
    root_height_m: float
    torso_up_z: float
    collapsed: bool

    def __post_init__(self) -> None:
        root_position = _finite_vector(
            self.root_position_world_m,
            width=3,
            field="root_position_world_m",
        )
        root_linear_velocity = _finite_vector(
            self.root_linear_velocity_world_m_s,
            width=3,
            field="root_linear_velocity_world_m_s",
        )
        object.__setattr__(
            self,
            "root_position_world_m",
            tuple(float(value) for value in root_position),
        )
        object.__setattr__(
            self,
            "root_linear_velocity_world_m_s",
            tuple(float(value) for value in root_linear_velocity),
        )
        nonnegative_fields = (
            "root_height_abs_error_m",
            "root_orientation_error_rad",
            "root_linear_velocity_rmse_m_s",
            "root_angular_velocity_rmse_rad_s",
            "joint_position_rmse_rad",
            "joint_velocity_rmse_rad_s",
            "joint_position_max_abs_rad",
            "joint_velocity_max_abs_rad_s",
            "normalized_policy_action_rms",
            "normalized_policy_action_max_abs",
            "normalized_policy_action_delta_rate_rms_per_s",
            "normalized_policy_action_delta_rate_max_abs_per_s",
            "generalized_actuator_torque_rms_n_m",
            "generalized_actuator_torque_max_abs_n_m",
            "generalized_actuator_torque_normalized_rms",
            "generalized_actuator_torque_normalized_max_abs",
            "joint_jerk_rms_rad_s3",
            "joint_jerk_max_abs_rad_s3",
            "simulator_contact_peak_normal_force_n",
            "floor_contact_peak_normal_force_n",
            "forbidden_floor_contact_fraction",
            "forbidden_floor_contact_peak_force_n",
        )
        for field in nonnegative_fields:
            value = _finite_scalar(getattr(self, field), field=field)
            if value < 0.0:
                raise ExperimentContractError(f"{field} cannot be negative")
        if self.normalized_policy_action_max_abs > 1.0:
            raise ExperimentContractError("normalized_policy_action_max_abs cannot exceed 1")
        rms_max_pairs = (
            ("joint_position_rmse_rad", "joint_position_max_abs_rad"),
            ("joint_velocity_rmse_rad_s", "joint_velocity_max_abs_rad_s"),
            ("normalized_policy_action_rms", "normalized_policy_action_max_abs"),
            (
                "normalized_policy_action_delta_rate_rms_per_s",
                "normalized_policy_action_delta_rate_max_abs_per_s",
            ),
            (
                "generalized_actuator_torque_rms_n_m",
                "generalized_actuator_torque_max_abs_n_m",
            ),
            (
                "generalized_actuator_torque_normalized_rms",
                "generalized_actuator_torque_normalized_max_abs",
            ),
            ("joint_jerk_rms_rad_s3", "joint_jerk_max_abs_rad_s3"),
        )
        for rms_field, max_field in rms_max_pairs:
            _validate_rms_max_summary(
                rms=getattr(self, rms_field),
                maximum=getattr(self, max_field),
                sample_count=ACTUATED_SCALAR_COUNT,
                field=rms_field,
            )
        if self.generalized_actuator_torque_normalized_max_abs > 1.0 + 1e-9:
            raise ExperimentContractError(
                "generalized_actuator_torque_normalized_max_abs cannot exceed 1"
            )
        _validate_frozen_torque_summary(
            normalized_action_rms=self.normalized_policy_action_rms,
            normalized_action_max_abs=self.normalized_policy_action_max_abs,
            raw_torque_rms_n_m=self.generalized_actuator_torque_rms_n_m,
            raw_torque_max_abs_n_m=self.generalized_actuator_torque_max_abs_n_m,
            normalized_torque_rms=self.generalized_actuator_torque_normalized_rms,
            normalized_torque_max_abs=(self.generalized_actuator_torque_normalized_max_abs),
        )
        _positive_scalar(self.control_period_seconds, field="control_period_seconds")
        root_height = _finite_scalar(self.root_height_m, field="root_height_m")
        if root_height != self.root_position_world_m[2]:
            raise ExperimentContractError("root_height_m must equal root_position_world_m z")
        torso_up_z = _finite_scalar(self.torso_up_z, field="torso_up_z")
        if not -1.0 <= torso_up_z <= 1.0:
            raise ExperimentContractError("torso_up_z must be in [-1, 1]")
        if not isinstance(self.collapsed, bool):
            raise ExperimentContractError("collapsed must be boolean")
        definition = _FROZEN_COLLAPSE_DEFINITION
        expected_collapsed = bool(
            root_height < definition.min_root_height_m
            or root_height > definition.max_root_height_m
            or torso_up_z < definition.min_torso_up_z
        )
        if self.collapsed is not expected_collapsed:
            raise ExperimentContractError(
                "collapsed must match the frozen root-height/torso-up definition"
            )
        for field in (
            "simulator_contact_count",
            "floor_contact_count",
            "forbidden_floor_contact_count",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ExperimentContractError(f"{field} must be an integer >= 0")
        if not (
            self.forbidden_floor_contact_count
            <= self.floor_contact_count
            <= self.simulator_contact_count
        ):
            raise ExperimentContractError(
                "contact counts must satisfy forbidden <= floor <= simulator"
            )
        expected_fraction = (
            self.forbidden_floor_contact_count / self.floor_contact_count
            if self.floor_contact_count
            else 0.0
        )
        if not math.isclose(
            self.forbidden_floor_contact_fraction,
            expected_fraction,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ExperimentContractError(
                "forbidden_floor_contact_fraction does not match contact counts"
            )
        if (
            self.forbidden_floor_contact_count == 0
            and self.forbidden_floor_contact_peak_force_n != 0.0
        ):
            raise ExperimentContractError(
                "forbidden contact peak force must be zero when count is zero"
            )
        if self.floor_contact_count == 0 and self.floor_contact_peak_normal_force_n != 0.0:
            raise ExperimentContractError(
                "floor contact peak force must be zero when count is zero"
            )
        if self.simulator_contact_count == 0 and self.simulator_contact_peak_normal_force_n != 0.0:
            raise ExperimentContractError(
                "simulator contact peak force must be zero when count is zero"
            )
        if self.floor_contact_peak_normal_force_n > (self.simulator_contact_peak_normal_force_n):
            raise ExperimentContractError(
                "floor contact peak force cannot exceed simulator contact peak force"
            )
        if self.forbidden_floor_contact_peak_force_n > (self.floor_contact_peak_normal_force_n):
            raise ExperimentContractError(
                "forbidden contact peak force cannot exceed floor contact peak force"
            )


def evaluate_tracking_step(
    *,
    root_position_world_m: ArrayLike,
    root_height_m: float,
    root_orientation_wxyz: ArrayLike,
    root_linear_velocity_world_m_s: ArrayLike,
    root_angular_velocity_body_rad_s: ArrayLike,
    joint_positions_rad: ArrayLike,
    joint_velocities_rad_s: ArrayLike,
    reference_frame: ArrayLike,
    normalized_policy_action: ArrayLike,
    previous_normalized_policy_action: ArrayLike,
    generalized_actuator_torque_n_m: ArrayLike,
    generalized_actuator_torque_capacity_n_m: ArrayLike,
    joint_accelerations_rad_s2: ArrayLike,
    previous_joint_accelerations_rad_s2: ArrayLike,
    contact_facts: Sequence[ContactFact],
    control_period_seconds: float,
) -> StepMetrics:
    """Recompute tracking, effort, contact, and smoothness measurements.

    Inputs are direct caller-supplied simulator/policy facts. This pure
    evaluator neither reads generated reward telemetry nor infers contact
    identity, action normalization, torque units, cadence, or acceleration
    history. For the validated hinge ABI, generalized actuator torque must come
    from ``qfrc_actuator`` at the actuator-driven DoF indices; MuJoCo
    ``actuator_force`` is pre-transmission input and is not an equivalent fact.
    """

    definition = _FROZEN_COLLAPSE_DEFINITION
    root_position = _finite_vector(
        root_position_world_m,
        width=3,
        field="root_position_world_m",
    )
    root_height = _finite_scalar(root_height_m, field="root_height_m")
    if root_height != float(root_position[2]):
        raise ExperimentContractError("root_height_m must equal root_position_world_m z")
    orientation = _unit_quaternion(root_orientation_wxyz, field="root_orientation_wxyz")
    linear_velocity = _finite_vector(
        root_linear_velocity_world_m_s,
        width=3,
        field="root_linear_velocity_world_m_s",
    )
    angular_velocity = _finite_vector(
        root_angular_velocity_body_rad_s,
        width=3,
        field="root_angular_velocity_body_rad_s",
    )
    joint_positions = _finite_vector(joint_positions_rad, width=17, field="joint_positions_rad")
    joint_velocities = _finite_vector(
        joint_velocities_rad_s,
        width=17,
        field="joint_velocities_rad_s",
    )
    action = _normalized_action(
        normalized_policy_action,
        field="normalized_policy_action",
    )
    previous_action = _normalized_action(
        previous_normalized_policy_action,
        field="previous_normalized_policy_action",
    )
    generalized_actuator_torque = _finite_vector(
        generalized_actuator_torque_n_m,
        width=17,
        field="generalized_actuator_torque_n_m",
    )
    generalized_actuator_torque_capacity = _positive_vector(
        generalized_actuator_torque_capacity_n_m,
        width=17,
        field="generalized_actuator_torque_capacity_n_m",
    )
    normalized_torque = generalized_actuator_torque / generalized_actuator_torque_capacity
    if np.any(np.abs(normalized_torque) > 1.0 + 1e-9):
        raise ExperimentContractError(
            "generalized actuator torque exceeds the declared per-joint capacity"
        )
    if not np.allclose(
        np.abs(normalized_torque),
        np.abs(action),
        rtol=0.0,
        atol=ACTION_TORQUE_NORMALIZATION_ABS_TOLERANCE,
    ):
        raise ExperimentContractError(
            "capacity-normalized generalized actuator torque must match the executed "
            "normalized policy action on the frozen actuator ABI"
        )
    joint_accelerations = _finite_vector(
        joint_accelerations_rad_s2,
        width=17,
        field="joint_accelerations_rad_s2",
    )
    previous_joint_accelerations = _finite_vector(
        previous_joint_accelerations_rad_s2,
        width=17,
        field="previous_joint_accelerations_rad_s2",
    )
    control_period = _positive_scalar(
        control_period_seconds,
        field="control_period_seconds",
    )
    if not isinstance(contact_facts, Sequence) or isinstance(contact_facts, (str, bytes)):
        raise ExperimentContractError("contact_facts must be a sequence")
    contacts = tuple(contact_facts)
    if any(not isinstance(contact, ContactFact) for contact in contacts):
        raise ExperimentContractError("contact_facts must contain only ContactFact values")
    if tuple(contact.contact_index for contact in contacts) != tuple(range(len(contacts))):
        raise ExperimentContractError(
            "contact_facts must cover every active contact in simulator index order"
        )
    reference = _finite_vector(reference_frame, width=REFERENCE_WIDTH, field="reference_frame")
    target_orientation = _unit_quaternion(
        reference[1:5],
        field="reference root orientation",
    )

    orientation_dot = float(np.clip(abs(np.dot(orientation, target_orientation)), 0.0, 1.0))
    orientation_error = 2.0 * math.acos(orientation_dot)
    root_linear_error = linear_velocity - reference[5:8]
    root_angular_error = angular_velocity - reference[8:11]
    joint_position_error = joint_positions - reference[11:28]
    joint_velocity_error = joint_velocities - reference[28:45]
    action_rms, action_max = _rms_and_max_abs(action, field="normalized policy action")
    action_delta_rate = (action - previous_action) / control_period
    action_delta_rate_rms, action_delta_rate_max = _rms_and_max_abs(
        action_delta_rate,
        field="normalized policy action delta rate",
    )
    generalized_actuator_torque_rms, generalized_actuator_torque_max = _rms_and_max_abs(
        generalized_actuator_torque,
        field="generalized actuator torque",
    )
    normalized_torque_rms, normalized_torque_max = _rms_and_max_abs(
        normalized_torque,
        field="capacity-normalized generalized actuator torque",
    )
    joint_jerk = (joint_accelerations - previous_joint_accelerations) / control_period
    joint_jerk_rms, joint_jerk_max = _rms_and_max_abs(
        joint_jerk,
        field="joint jerk",
    )
    floor_contacts = tuple(contact for contact in contacts if contact.involves_floor)
    forbidden_contacts = tuple(
        contact for contact in floor_contacts if contact.forbidden_floor_contact
    )
    forbidden_count = len(forbidden_contacts)
    floor_count = len(floor_contacts)
    forbidden_fraction = forbidden_count / floor_count if floor_count else 0.0
    simulator_peak_force = max(
        (contact.normal_force_n for contact in contacts),
        default=0.0,
    )
    floor_peak_force = max(
        (contact.normal_force_n for contact in floor_contacts),
        default=0.0,
    )
    forbidden_peak_force = max(
        (contact.normal_force_n for contact in forbidden_contacts),
        default=0.0,
    )

    # R[2, 2] for a wxyz quaternion: alignment of torso-local +z with world +z.
    _, x, y, _ = orientation
    torso_up_z = float(1.0 - 2.0 * (x * x + y * y))
    collapsed = bool(
        root_height < definition.min_root_height_m
        or root_height > definition.max_root_height_m
        or torso_up_z < definition.min_torso_up_z
    )
    return StepMetrics(
        root_position_world_m=tuple(float(value) for value in root_position),
        root_linear_velocity_world_m_s=tuple(float(value) for value in linear_velocity),
        root_height_abs_error_m=abs(root_height - float(reference[0])),
        root_orientation_error_rad=float(orientation_error),
        root_linear_velocity_rmse_m_s=_rmse(
            root_linear_error,
            field="root linear velocity error",
        ),
        root_angular_velocity_rmse_rad_s=_rmse(
            root_angular_error,
            field="root angular velocity error",
        ),
        joint_position_rmse_rad=_rmse(
            joint_position_error,
            field="joint position error",
        ),
        joint_velocity_rmse_rad_s=_rmse(
            joint_velocity_error,
            field="joint velocity error",
        ),
        joint_position_max_abs_rad=_rms_and_max_abs(
            joint_position_error,
            field="joint position error",
        )[1],
        joint_velocity_max_abs_rad_s=_rms_and_max_abs(
            joint_velocity_error,
            field="joint velocity error",
        )[1],
        normalized_policy_action_rms=action_rms,
        normalized_policy_action_max_abs=action_max,
        normalized_policy_action_delta_rate_rms_per_s=action_delta_rate_rms,
        normalized_policy_action_delta_rate_max_abs_per_s=action_delta_rate_max,
        generalized_actuator_torque_rms_n_m=generalized_actuator_torque_rms,
        generalized_actuator_torque_max_abs_n_m=(generalized_actuator_torque_max),
        generalized_actuator_torque_normalized_rms=normalized_torque_rms,
        generalized_actuator_torque_normalized_max_abs=normalized_torque_max,
        joint_jerk_rms_rad_s3=joint_jerk_rms,
        joint_jerk_max_abs_rad_s3=joint_jerk_max,
        simulator_contact_count=len(contacts),
        simulator_contact_peak_normal_force_n=simulator_peak_force,
        floor_contact_count=floor_count,
        floor_contact_peak_normal_force_n=floor_peak_force,
        forbidden_floor_contact_count=forbidden_count,
        forbidden_floor_contact_fraction=forbidden_fraction,
        forbidden_floor_contact_peak_force_n=forbidden_peak_force,
        control_period_seconds=control_period,
        root_height_m=root_height,
        torso_up_z=torso_up_z,
        collapsed=collapsed,
    )


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """One evaluation seed; episodes are repeated measures within a checkpoint.

    Contact counts sum active contact samples across simulator steps, not
    inferred collision events. ``forbidden_floor_contact_fraction`` uses all
    floor-contact samples as its denominator; the step fraction separately
    reports temporal prevalence.
    """

    evaluation_seed: int
    observed_steps: int
    requested_steps: int
    survived_full_horizon: bool
    first_collapse_step: int | None
    collapse_fraction: float
    station_keeping_origin_source: str
    initial_root_position_world_m: tuple[float, float, float]
    root_horizontal_displacement_max_m: float
    root_linear_velocity_x_max_abs_m_s: float
    root_linear_velocity_y_max_abs_m_s: float
    root_linear_velocity_z_max_abs_m_s: float
    control_period_seconds: float
    root_height_rmse_m: float
    root_orientation_rmse_rad: float
    root_linear_velocity_rmse_m_s: float
    root_angular_velocity_rmse_rad_s: float
    joint_position_rmse_rad: float
    joint_velocity_rmse_rad_s: float
    joint_position_max_abs_rad: float
    joint_velocity_max_abs_rad_s: float
    normalized_policy_action_rms: float
    normalized_policy_action_max_abs: float
    normalized_policy_action_delta_rate_rms_per_s: float
    normalized_policy_action_delta_rate_max_abs_per_s: float
    generalized_actuator_torque_rms_n_m: float
    generalized_actuator_torque_max_abs_n_m: float
    generalized_actuator_torque_normalized_rms: float
    generalized_actuator_torque_normalized_max_abs: float
    joint_jerk_rms_rad_s3: float
    joint_jerk_max_abs_rad_s3: float
    simulator_contact_count: int
    simulator_contact_peak_normal_force_n: float
    floor_contact_count: int
    floor_contact_peak_normal_force_n: float
    forbidden_floor_contact_count: int
    forbidden_floor_contact_fraction: float
    forbidden_floor_contact_step_fraction: float
    forbidden_floor_contact_peak_force_n: float
    tracking_return: float

    def __post_init__(self) -> None:
        for field in ("evaluation_seed", "observed_steps", "requested_steps"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ExperimentContractError(f"{field} must be an integer")
        if self.evaluation_seed < 0:
            raise ExperimentContractError("evaluation_seed cannot be negative")
        if self.requested_steps < 1 or not 0 < self.observed_steps <= self.requested_steps:
            raise ExperimentContractError(
                "episode steps must satisfy 0 < observed_steps <= requested_steps"
            )
        if not isinstance(self.survived_full_horizon, bool):
            raise ExperimentContractError("survived_full_horizon must be boolean")
        if self.station_keeping_origin_source != STATION_KEEPING_ORIGIN_SOURCE:
            raise ExperimentContractError(
                "station_keeping_origin_source does not identify seeded-reset MuJoCo qpos"
            )
        initial_root_position = _finite_vector(
            self.initial_root_position_world_m,
            width=3,
            field="initial_root_position_world_m",
        )
        object.__setattr__(
            self,
            "initial_root_position_world_m",
            tuple(float(value) for value in initial_root_position),
        )
        control_period = _positive_scalar(
            self.control_period_seconds,
            field="control_period_seconds",
        )
        object.__setattr__(self, "control_period_seconds", control_period)

        nonnegative_fields = (
            "collapse_fraction",
            "root_horizontal_displacement_max_m",
            "root_linear_velocity_x_max_abs_m_s",
            "root_linear_velocity_y_max_abs_m_s",
            "root_linear_velocity_z_max_abs_m_s",
            "root_height_rmse_m",
            "root_orientation_rmse_rad",
            "root_linear_velocity_rmse_m_s",
            "root_angular_velocity_rmse_rad_s",
            "joint_position_rmse_rad",
            "joint_velocity_rmse_rad_s",
            "joint_position_max_abs_rad",
            "joint_velocity_max_abs_rad_s",
            "normalized_policy_action_rms",
            "normalized_policy_action_max_abs",
            "normalized_policy_action_delta_rate_rms_per_s",
            "normalized_policy_action_delta_rate_max_abs_per_s",
            "generalized_actuator_torque_rms_n_m",
            "generalized_actuator_torque_max_abs_n_m",
            "generalized_actuator_torque_normalized_rms",
            "generalized_actuator_torque_normalized_max_abs",
            "joint_jerk_rms_rad_s3",
            "joint_jerk_max_abs_rad_s3",
            "simulator_contact_peak_normal_force_n",
            "floor_contact_peak_normal_force_n",
            "forbidden_floor_contact_fraction",
            "forbidden_floor_contact_step_fraction",
            "forbidden_floor_contact_peak_force_n",
        )
        for field in nonnegative_fields:
            if _finite_scalar(getattr(self, field), field=field) < 0.0:
                raise ExperimentContractError(f"{field} cannot be negative")
        tracking_return = _finite_scalar(self.tracking_return, field="tracking_return")
        if not 0.0 <= tracking_return <= self.observed_steps + 1e-9:
            raise ExperimentContractError(
                "tracking_return must be in [0, observed_steps] for the frozen reward"
            )
        if self.root_orientation_rmse_rad > math.pi + 1e-12:
            raise ExperimentContractError("root_orientation_rmse_rad cannot exceed pi")

        rms_max_pairs = (
            ("joint_position_rmse_rad", "joint_position_max_abs_rad"),
            ("joint_velocity_rmse_rad_s", "joint_velocity_max_abs_rad_s"),
            ("normalized_policy_action_rms", "normalized_policy_action_max_abs"),
            (
                "normalized_policy_action_delta_rate_rms_per_s",
                "normalized_policy_action_delta_rate_max_abs_per_s",
            ),
            (
                "generalized_actuator_torque_rms_n_m",
                "generalized_actuator_torque_max_abs_n_m",
            ),
            (
                "generalized_actuator_torque_normalized_rms",
                "generalized_actuator_torque_normalized_max_abs",
            ),
            ("joint_jerk_rms_rad_s3", "joint_jerk_max_abs_rad_s3"),
        )
        scalar_sample_count = ACTUATED_SCALAR_COUNT * self.observed_steps
        for rms_field, max_field in rms_max_pairs:
            _validate_rms_max_summary(
                rms=getattr(self, rms_field),
                maximum=getattr(self, max_field),
                sample_count=scalar_sample_count,
                field=rms_field,
            )
        root_velocity_max = max(
            self.root_linear_velocity_x_max_abs_m_s,
            self.root_linear_velocity_y_max_abs_m_s,
            self.root_linear_velocity_z_max_abs_m_s,
        )
        _validate_rms_max_summary(
            rms=self.root_linear_velocity_rmse_m_s,
            maximum=root_velocity_max,
            sample_count=3 * self.observed_steps,
            field="root_linear_velocity_rmse_m_s",
        )
        if self.normalized_policy_action_max_abs > 1.0:
            raise ExperimentContractError("normalized_policy_action_max_abs cannot exceed 1")
        if self.generalized_actuator_torque_normalized_max_abs > 1.0 + 1e-9:
            raise ExperimentContractError(
                "generalized_actuator_torque_normalized_max_abs cannot exceed 1"
            )
        _validate_frozen_torque_summary(
            normalized_action_rms=self.normalized_policy_action_rms,
            normalized_action_max_abs=self.normalized_policy_action_max_abs,
            raw_torque_rms_n_m=self.generalized_actuator_torque_rms_n_m,
            raw_torque_max_abs_n_m=self.generalized_actuator_torque_max_abs_n_m,
            normalized_torque_rms=self.generalized_actuator_torque_normalized_rms,
            normalized_torque_max_abs=(self.generalized_actuator_torque_normalized_max_abs),
        )
        if self.normalized_policy_action_max_abs == 0.0 and (
            self.normalized_policy_action_delta_rate_rms_per_s != 0.0
            or self.normalized_policy_action_delta_rate_max_abs_per_s != 0.0
        ):
            raise ExperimentContractError(
                "action-delta metrics must be zero when the episode action maximum is zero"
            )
        maximum_delta_rate_bound = 2.0 * self.normalized_policy_action_max_abs / control_period
        rms_delta_rate_bound = 2.0 * self.normalized_policy_action_rms / control_period
        minimum_delta_rate_max = self.normalized_policy_action_max_abs / (
            self.observed_steps * control_period
        )
        minimum_delta_rate_rms = (
            self.normalized_policy_action_rms
            * math.sqrt(2.0 / (self.observed_steps * (self.observed_steps + 1)))
            / control_period
        )
        action_rate_tolerance = SUMMARY_ALGEBRA_ABS_TOLERANCE + SUMMARY_ALGEBRA_REL_TOLERANCE * max(
            self.normalized_policy_action_delta_rate_rms_per_s,
            self.normalized_policy_action_delta_rate_max_abs_per_s,
            maximum_delta_rate_bound,
            rms_delta_rate_bound,
            minimum_delta_rate_max,
            minimum_delta_rate_rms,
        )
        if not math.isfinite(maximum_delta_rate_bound) or not math.isfinite(rms_delta_rate_bound):
            raise ExperimentContractError("action-delta rate bound must be finite")
        if (
            self.normalized_policy_action_delta_rate_max_abs_per_s
            > maximum_delta_rate_bound + action_rate_tolerance
        ):
            raise ExperimentContractError(
                "action-delta maximum exceeds the zero-reset action/cadence bound"
            )
        if (
            self.normalized_policy_action_delta_rate_rms_per_s
            > rms_delta_rate_bound + action_rate_tolerance
        ):
            raise ExperimentContractError(
                "action-delta RMS exceeds the zero-reset action/cadence bound"
            )
        if (
            self.normalized_policy_action_delta_rate_max_abs_per_s + action_rate_tolerance
            < minimum_delta_rate_max
        ):
            raise ExperimentContractError(
                "action-delta maximum is too small for the zero-reset episode action maximum"
            )
        if (
            self.normalized_policy_action_delta_rate_rms_per_s + action_rate_tolerance
            < minimum_delta_rate_rms
        ):
            raise ExperimentContractError(
                "action-delta RMS is too small for the zero-reset episode action RMS"
            )

        collapse_fraction = float(self.collapse_fraction)
        if collapse_fraction > 1.0:
            raise ExperimentContractError("collapse_fraction must be in [0, 1]")
        collapse_count_float = collapse_fraction * self.observed_steps
        collapse_count = round(collapse_count_float)
        if not math.isclose(
            collapse_count_float,
            collapse_count,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ExperimentContractError(
                "collapse_fraction must represent an integer collapsed-step count"
            )
        expected_survival = collapse_count == 0 and self.observed_steps == self.requested_steps
        if self.survived_full_horizon is not expected_survival:
            raise ExperimentContractError("survival fields are contradictory")
        if collapse_count == 0:
            if self.first_collapse_step is not None:
                raise ExperimentContractError("first_collapse_step requires a collapse")
        elif (
            not isinstance(self.first_collapse_step, int)
            or isinstance(self.first_collapse_step, bool)
            or not 1 <= self.first_collapse_step <= self.observed_steps
        ):
            raise ExperimentContractError(
                "first_collapse_step must identify an observed collapsed step"
            )
        if (
            collapse_count > 0
            and collapse_count > self.observed_steps - self.first_collapse_step + 1
        ):
            raise ExperimentContractError(
                "collapse count exceeds the steps at or after first_collapse_step"
            )

        for field in (
            "simulator_contact_count",
            "floor_contact_count",
            "forbidden_floor_contact_count",
        ):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ExperimentContractError(f"{field} must be an integer >= 0")
        if not (
            self.forbidden_floor_contact_count
            <= self.floor_contact_count
            <= self.simulator_contact_count
        ):
            raise ExperimentContractError(
                "contact counts must satisfy forbidden <= floor <= simulator"
            )
        expected_contact_fraction = (
            self.forbidden_floor_contact_count / self.floor_contact_count
            if self.floor_contact_count
            else 0.0
        )
        if not math.isclose(
            self.forbidden_floor_contact_fraction,
            expected_contact_fraction,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ExperimentContractError(
                "forbidden_floor_contact_fraction does not match contact counts"
            )
        step_fraction = float(self.forbidden_floor_contact_step_fraction)
        if step_fraction > 1.0:
            raise ExperimentContractError("forbidden_floor_contact_step_fraction must be in [0, 1]")
        forbidden_step_count_float = step_fraction * self.observed_steps
        forbidden_step_count = round(forbidden_step_count_float)
        if not math.isclose(
            forbidden_step_count_float,
            forbidden_step_count,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ExperimentContractError(
                "forbidden_floor_contact_step_fraction must represent an integer step count"
            )
        if forbidden_step_count > min(
            self.observed_steps,
            self.forbidden_floor_contact_count,
        ):
            raise ExperimentContractError(
                "forbidden-contact step count exceeds observed steps or contact samples"
            )
        if (self.forbidden_floor_contact_count == 0) != (forbidden_step_count == 0):
            raise ExperimentContractError(
                "forbidden-contact count and step fraction are contradictory"
            )

        zero_count_peak_pairs = (
            ("simulator_contact_count", "simulator_contact_peak_normal_force_n"),
            ("floor_contact_count", "floor_contact_peak_normal_force_n"),
            ("forbidden_floor_contact_count", "forbidden_floor_contact_peak_force_n"),
        )
        for count_field, peak_field in zero_count_peak_pairs:
            if getattr(self, count_field) == 0 and getattr(self, peak_field) != 0.0:
                raise ExperimentContractError(
                    f"{peak_field} must be zero when {count_field} is zero"
                )
        if self.floor_contact_peak_normal_force_n > (self.simulator_contact_peak_normal_force_n):
            raise ExperimentContractError(
                "floor contact peak force cannot exceed simulator contact peak force"
            )
        if self.forbidden_floor_contact_peak_force_n > (self.floor_contact_peak_normal_force_n):
            raise ExperimentContractError(
                "forbidden contact peak force cannot exceed floor contact peak force"
            )


class EpisodeAccumulator:
    """Accumulate a complete, non-cherry-picked evaluation episode."""

    _ERROR_FIELDS = (
        "root_height_abs_error_m",
        "root_orientation_error_rad",
        "root_linear_velocity_rmse_m_s",
        "root_angular_velocity_rmse_rad_s",
        "joint_position_rmse_rad",
        "joint_velocity_rmse_rad_s",
    )
    _CONTROL_RMS_FIELDS = (
        "normalized_policy_action_rms",
        "normalized_policy_action_delta_rate_rms_per_s",
        "generalized_actuator_torque_rms_n_m",
        "generalized_actuator_torque_normalized_rms",
        "joint_jerk_rms_rad_s3",
    )
    _RMS_FIELDS = (*_ERROR_FIELDS, *_CONTROL_RMS_FIELDS)

    def __init__(
        self,
        *,
        evaluation_seed: int,
        requested_steps: int,
        initial_root_position_world_m: ArrayLike,
        station_keeping_origin_source: str,
        initial_normalized_policy_action: ArrayLike,
    ) -> None:
        if not isinstance(evaluation_seed, int) or isinstance(evaluation_seed, bool):
            raise ExperimentContractError("evaluation_seed must be an integer")
        if not isinstance(requested_steps, int) or isinstance(requested_steps, bool):
            raise ExperimentContractError("requested_steps must be an integer")
        if requested_steps < 1:
            raise ExperimentContractError("requested_steps must be positive")
        if station_keeping_origin_source != STATION_KEEPING_ORIGIN_SOURCE:
            raise ExperimentContractError(
                "station_keeping_origin_source must identify seeded-reset MuJoCo qpos"
            )
        initial_root_position = _finite_vector(
            initial_root_position_world_m,
            width=3,
            field="initial_root_position_world_m",
        )
        initial_action = _normalized_action(
            initial_normalized_policy_action,
            field="initial_normalized_policy_action",
        )
        if not np.array_equal(initial_action, np.zeros(ACTUATED_SCALAR_COUNT)):
            raise ExperimentContractError(
                "initial_normalized_policy_action must be exactly zero for episode algebra"
            )
        self._evaluation_seed = evaluation_seed
        self._requested_steps = requested_steps
        self._station_keeping_origin_source = station_keeping_origin_source
        self._initial_root_position_world_m = tuple(float(value) for value in initial_root_position)
        self._steps = 0
        self._control_period_seconds: float | None = None
        self._root_horizontal_displacement_max = 0.0
        self._root_linear_velocity_axis_max = np.zeros(3, dtype=np.float64)
        self._rms_scales = {field: 0.0 for field in self._RMS_FIELDS}
        self._rms_scaled_squares = {field: 0.0 for field in self._RMS_FIELDS}
        self._joint_position_max = 0.0
        self._joint_velocity_max = 0.0
        self._normalized_action_max = 0.0
        self._normalized_action_delta_rate_max = 0.0
        self._generalized_actuator_torque_max = 0.0
        self._generalized_actuator_torque_normalized_max = 0.0
        self._joint_jerk_max = 0.0
        self._simulator_contact_count = 0
        self._simulator_contact_peak_normal_force = 0.0
        self._floor_contact_count = 0
        self._floor_contact_peak_normal_force = 0.0
        self._forbidden_floor_contact_count = 0
        self._forbidden_floor_contact_step_count = 0
        self._forbidden_floor_contact_peak_force = 0.0
        self._collapse_count = 0
        self._first_collapse_step: int | None = None
        self._tracking_return = 0.0

    def add(self, metrics: StepMetrics, *, tracking_reward: float) -> None:
        if self._steps >= self._requested_steps:
            raise ExperimentContractError("episode accumulator exceeded its fixed horizon")
        if not isinstance(metrics, StepMetrics):
            raise ExperimentContractError("metrics must be a StepMetrics value")
        if (
            isinstance(tracking_reward, bool)
            or not isinstance(tracking_reward, (int, float))
            or not math.isfinite(float(tracking_reward))
        ):
            raise ExperimentContractError("tracking_reward must be finite")
        resolved_tracking_reward = float(tracking_reward)
        if not 0.0 <= resolved_tracking_reward <= 1.0 + 1e-12:
            raise ExperimentContractError("tracking_reward must be in [0, 1] for the frozen reward")
        tracking_return = self._tracking_return + resolved_tracking_reward
        if not math.isfinite(tracking_return):
            raise ExperimentContractError("tracking reward sum must remain finite")
        if self._control_period_seconds is None:
            self._control_period_seconds = metrics.control_period_seconds
        elif metrics.control_period_seconds != self._control_period_seconds:
            raise ExperimentContractError(
                "episode control_period_seconds changed between simulator steps"
            )
        initial_root_position = np.asarray(
            self._initial_root_position_world_m,
            dtype=np.float64,
        )
        current_root_position = np.asarray(metrics.root_position_world_m, dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore"):
            horizontal_delta = current_root_position[:2] - initial_root_position[:2]
        horizontal_displacement = math.hypot(
            float(horizontal_delta[0]),
            float(horizontal_delta[1]),
        )
        if not math.isfinite(horizontal_displacement):
            raise ExperimentContractError("derived root horizontal displacement must be finite")
        self._root_horizontal_displacement_max = max(
            self._root_horizontal_displacement_max,
            horizontal_displacement,
        )
        root_linear_velocity = np.abs(
            np.asarray(metrics.root_linear_velocity_world_m_s, dtype=np.float64)
        )
        self._root_linear_velocity_axis_max = np.maximum(
            self._root_linear_velocity_axis_max,
            root_linear_velocity,
        )
        self._steps += 1
        for field in self._RMS_FIELDS:
            value = abs(getattr(metrics, field))
            scale = self._rms_scales[field]
            scaled_squares = self._rms_scaled_squares[field]
            if value != 0.0:
                if scale < value:
                    ratio = scale / value
                    scaled_squares = 1.0 + scaled_squares * ratio * ratio
                    scale = value
                else:
                    ratio = value / scale
                    scaled_squares += ratio * ratio
            self._rms_scales[field] = scale
            self._rms_scaled_squares[field] = scaled_squares
        self._joint_position_max = max(
            self._joint_position_max,
            metrics.joint_position_max_abs_rad,
        )
        self._joint_velocity_max = max(
            self._joint_velocity_max,
            metrics.joint_velocity_max_abs_rad_s,
        )
        self._normalized_action_max = max(
            self._normalized_action_max,
            metrics.normalized_policy_action_max_abs,
        )
        self._normalized_action_delta_rate_max = max(
            self._normalized_action_delta_rate_max,
            metrics.normalized_policy_action_delta_rate_max_abs_per_s,
        )
        self._generalized_actuator_torque_max = max(
            self._generalized_actuator_torque_max,
            metrics.generalized_actuator_torque_max_abs_n_m,
        )
        self._generalized_actuator_torque_normalized_max = max(
            self._generalized_actuator_torque_normalized_max,
            metrics.generalized_actuator_torque_normalized_max_abs,
        )
        self._joint_jerk_max = max(
            self._joint_jerk_max,
            metrics.joint_jerk_max_abs_rad_s3,
        )
        self._simulator_contact_count += metrics.simulator_contact_count
        self._simulator_contact_peak_normal_force = max(
            self._simulator_contact_peak_normal_force,
            metrics.simulator_contact_peak_normal_force_n,
        )
        self._floor_contact_count += metrics.floor_contact_count
        self._floor_contact_peak_normal_force = max(
            self._floor_contact_peak_normal_force,
            metrics.floor_contact_peak_normal_force_n,
        )
        self._forbidden_floor_contact_count += metrics.forbidden_floor_contact_count
        if metrics.forbidden_floor_contact_count:
            self._forbidden_floor_contact_step_count += 1
        self._forbidden_floor_contact_peak_force = max(
            self._forbidden_floor_contact_peak_force,
            metrics.forbidden_floor_contact_peak_force_n,
        )
        if metrics.collapsed:
            self._collapse_count += 1
            if self._first_collapse_step is None:
                self._first_collapse_step = self._steps
        self._tracking_return = tracking_return

    def finish(self, *, require_complete: bool = True) -> EpisodeMetrics:
        if self._steps < 1:
            raise ExperimentContractError("cannot summarize an empty episode")
        if require_complete and self._steps != self._requested_steps:
            raise ExperimentContractError(
                f"evaluation ended after {self._steps} steps; expected {self._requested_steps}"
            )
        if self._control_period_seconds is None:
            raise ExperimentContractError("episode control period was not observed")

        def aggregate(field: str) -> float:
            scale = self._rms_scales[field]
            if scale == 0.0:
                return 0.0
            return scale * math.sqrt(self._rms_scaled_squares[field] / self._steps)

        forbidden_contact_fraction = (
            self._forbidden_floor_contact_count / self._floor_contact_count
            if self._floor_contact_count
            else 0.0
        )

        return EpisodeMetrics(
            evaluation_seed=self._evaluation_seed,
            observed_steps=self._steps,
            requested_steps=self._requested_steps,
            survived_full_horizon=self._first_collapse_step is None
            and self._steps == self._requested_steps,
            first_collapse_step=self._first_collapse_step,
            collapse_fraction=self._collapse_count / self._steps,
            station_keeping_origin_source=self._station_keeping_origin_source,
            initial_root_position_world_m=self._initial_root_position_world_m,
            root_horizontal_displacement_max_m=(self._root_horizontal_displacement_max),
            root_linear_velocity_x_max_abs_m_s=float(self._root_linear_velocity_axis_max[0]),
            root_linear_velocity_y_max_abs_m_s=float(self._root_linear_velocity_axis_max[1]),
            root_linear_velocity_z_max_abs_m_s=float(self._root_linear_velocity_axis_max[2]),
            control_period_seconds=self._control_period_seconds,
            root_height_rmse_m=aggregate("root_height_abs_error_m"),
            root_orientation_rmse_rad=aggregate("root_orientation_error_rad"),
            root_linear_velocity_rmse_m_s=aggregate("root_linear_velocity_rmse_m_s"),
            root_angular_velocity_rmse_rad_s=aggregate("root_angular_velocity_rmse_rad_s"),
            joint_position_rmse_rad=aggregate("joint_position_rmse_rad"),
            joint_velocity_rmse_rad_s=aggregate("joint_velocity_rmse_rad_s"),
            joint_position_max_abs_rad=self._joint_position_max,
            joint_velocity_max_abs_rad_s=self._joint_velocity_max,
            normalized_policy_action_rms=aggregate("normalized_policy_action_rms"),
            normalized_policy_action_max_abs=self._normalized_action_max,
            normalized_policy_action_delta_rate_rms_per_s=aggregate(
                "normalized_policy_action_delta_rate_rms_per_s"
            ),
            normalized_policy_action_delta_rate_max_abs_per_s=(
                self._normalized_action_delta_rate_max
            ),
            generalized_actuator_torque_rms_n_m=aggregate("generalized_actuator_torque_rms_n_m"),
            generalized_actuator_torque_max_abs_n_m=(self._generalized_actuator_torque_max),
            generalized_actuator_torque_normalized_rms=aggregate(
                "generalized_actuator_torque_normalized_rms"
            ),
            generalized_actuator_torque_normalized_max_abs=(
                self._generalized_actuator_torque_normalized_max
            ),
            joint_jerk_rms_rad_s3=aggregate("joint_jerk_rms_rad_s3"),
            joint_jerk_max_abs_rad_s3=self._joint_jerk_max,
            simulator_contact_count=self._simulator_contact_count,
            simulator_contact_peak_normal_force_n=(self._simulator_contact_peak_normal_force),
            floor_contact_count=self._floor_contact_count,
            floor_contact_peak_normal_force_n=(self._floor_contact_peak_normal_force),
            forbidden_floor_contact_count=self._forbidden_floor_contact_count,
            forbidden_floor_contact_fraction=forbidden_contact_fraction,
            forbidden_floor_contact_step_fraction=(
                self._forbidden_floor_contact_step_count / self._steps
            ),
            forbidden_floor_contact_peak_force_n=(self._forbidden_floor_contact_peak_force),
            tracking_return=self._tracking_return,
        )
