"""Exact controller-facing reference ABI for Gymnasium ``Humanoid-v5``.

The policy command includes free-root posture and motion plus desired hinge
position and velocity for each of the 17 actuated joints in MuJoCo actuator
order.  Horizontal root position is deliberately omitted so a locomotion
reference does not pin the body to an absolute world ``x/y`` location.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from oracle_composition.contracts import (
    OracleContractError,
    ReferenceArtifact,
    ReferenceSchema,
    assert_exact_schema,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


HUMANOID_ACTUATOR_JOINT_ORDER = (
    "abdomen_y",
    "abdomen_z",
    "abdomen_x",
    "right_hip_x",
    "right_hip_z",
    "right_hip_y",
    "right_knee",
    "left_hip_x",
    "left_hip_z",
    "left_hip_y",
    "left_knee",
    "right_shoulder1",
    "right_shoulder2",
    "right_elbow",
    "left_shoulder1",
    "left_shoulder2",
    "left_elbow",
)
HUMANOID_QPOS_INDICES_BY_ACTUATOR = (
    8,
    7,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
)
HUMANOID_QVEL_INDICES_BY_ACTUATOR = (
    7,
    6,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
)
HUMANOID_ACTUATOR_GEAR_BY_JOINT = (
    100.0,
    100.0,
    100.0,
    100.0,
    100.0,
    300.0,
    200.0,
    100.0,
    100.0,
    300.0,
    200.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
)
HUMANOID_ACTUATOR_CONTROL_RANGE = (-0.4, 0.4)
HUMANOID_NOISY_ROOT_QUATERNION_NORM_BOUNDS = (0.98, 1.02)
HUMANOID_RESET_ROOT_QUATERNION_COMPONENT_BOUNDS = (
    (0.99, 1.01),
    (-0.01, 0.01),
    (-0.01, 0.01),
    (-0.01, 0.01),
)
HUMANOID_GENERALIZED_ACTUATOR_TORQUE_CAPACITY_N_M = (
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    120.0,
    80.0,
    40.0,
    40.0,
    120.0,
    80.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
)

# Gymnasium Humanoid-v5 uses a 0.003 s MuJoCo timestep and frame_skip=5.
HUMANOID_CONTROL_PERIOD_SECONDS = 0.015
HUMANOID_REFERENCE_CADENCE_HZ = 1.0 / HUMANOID_CONTROL_PERIOD_SECONDS
HUMANOID_REFERENCE_ROOT_SEMANTICS = (
    "mujoco_free_root:z+quaternion+linear_velocity_in_world;"
    "angular_velocity_in_torso_local;x+y_omitted"
)

# Broad numerical-safety envelope for authored candidates. These limits reject
# corrupt or unit-mismatched commands before reward evaluation. They are not a
# kinematic or dynamics-feasibility certificate.
MIN_REFERENCE_ROOT_HEIGHT_M = 0.0
MAX_REFERENCE_ROOT_HEIGHT_M = 5.0
MAX_REFERENCE_ROOT_LINEAR_SPEED_COMPONENT_M_S = 25.0
MAX_REFERENCE_ROOT_ANGULAR_SPEED_COMPONENT_RAD_S = 100.0
MAX_REFERENCE_JOINT_POSITION_ABS_RAD = 2.0 * math.pi
MAX_REFERENCE_JOINT_SPEED_ABS_RAD_S = 100.0

HUMANOID_REFERENCE_SCHEMA = ReferenceSchema(
    feature_names=(
        "root.position_z",
        "root.orientation_w",
        "root.orientation_x",
        "root.orientation_y",
        "root.orientation_z",
        "root.linear_velocity_x",
        "root.linear_velocity_y",
        "root.linear_velocity_z",
        "root.angular_velocity_x",
        "root.angular_velocity_y",
        "root.angular_velocity_z",
        *(f"joint_position.{name}" for name in HUMANOID_ACTUATOR_JOINT_ORDER),
        *(f"joint_velocity.{name}" for name in HUMANOID_ACTUATOR_JOINT_ORDER),
    ),
    feature_units=(
        "m",
        "1",
        "1",
        "1",
        "1",
        *("m/s" for _ in range(3)),
        *("rad/s" for _ in range(3)),
        *("rad" for _ in HUMANOID_ACTUATOR_JOINT_ORDER),
        *("rad/s" for _ in HUMANOID_ACTUATOR_JOINT_ORDER),
    ),
    root_frame=HUMANOID_REFERENCE_ROOT_SEMANTICS,
    cadence_hz=HUMANOID_REFERENCE_CADENCE_HZ,
)


@dataclass(frozen=True, slots=True)
class HumanoidActuatorABI:
    """Observed model addresses for the exact actuator-ordered state."""

    joint_names: tuple[str, ...]
    qpos_indices: tuple[int, ...]
    qvel_indices: tuple[int, ...]
    actuator_gear_by_joint: tuple[float, ...]
    generalized_actuator_torque_capacity_n_m: tuple[float, ...]
    control_period_seconds: float

    @property
    def n_actuators(self) -> int:
        return len(self.joint_names)


@dataclass(frozen=True, slots=True)
class HumanoidTrackingState:
    """Current simulator state in the exact fields and frames of the ABI."""

    root_position_world_m: NDArray[np.float64]
    root_height_m: float
    root_orientation_wxyz: NDArray[np.float64]
    root_linear_velocity_world_m_s: NDArray[np.float64]
    root_angular_velocity_body_rad_s: NDArray[np.float64]
    joint_positions_rad: NDArray[np.float64]
    joint_velocities_rad_s: NDArray[np.float64]


def _mujoco_name(model: object, object_type: object, index: int) -> str | None:
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - guarded by the Gym extra
        raise OracleContractError("MuJoCo is required to validate the Humanoid ABI") from exc
    return mujoco.mj_id2name(model, object_type, index)


def validate_humanoid_actuator_abi(env: object) -> HumanoidActuatorABI:
    """Inspect and reject any model that is not the pinned 17-actuator ABI."""

    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - guarded by the Gym extra
        raise OracleContractError("MuJoCo is required to validate the Humanoid ABI") from exc

    unwrapped = getattr(env, "unwrapped", env)
    model = getattr(unwrapped, "model", None)
    if model is None:
        raise OracleContractError("environment does not expose a MuJoCo model")
    if int(model.nu) != len(HUMANOID_ACTUATOR_JOINT_ORDER):
        raise OracleContractError(
            f"Humanoid actuator count is {int(model.nu)}; expected "
            f"{len(HUMANOID_ACTUATOR_JOINT_ORDER)}"
        )
    if (int(model.nq), int(model.nv)) != (24, 23):
        raise OracleContractError(
            f"Humanoid state dimensions are {(int(model.nq), int(model.nv))}; expected (24, 23)"
        )

    action_space = getattr(env, "action_space", None)
    if action_space is None or tuple(action_space.shape) != (len(HUMANOID_ACTUATOR_JOINT_ORDER),):
        raise OracleContractError("action space does not match the 17-actuator ABI")
    observation_space = getattr(env, "observation_space", None)
    if observation_space is None or tuple(observation_space.shape) != (348,):
        raise OracleContractError("observation space does not match Humanoid-v5 shape (348,)")

    names: list[str] = []
    qpos_indices: list[int] = []
    qvel_indices: list[int] = []
    actuator_gears: list[float] = []
    torque_capacities: list[float] = []
    errors: list[str] = []
    root_name = _mujoco_name(model, mujoco.mjtObj.mjOBJ_JOINT, 0)
    if root_name != "root":
        errors.append(f"free-root joint is named {root_name!r}; expected 'root'")
    if int(model.jnt_type[0]) != int(mujoco.mjtJoint.mjJNT_FREE):
        errors.append("joint[0] is not the expected free root")
    if (int(model.jnt_qposadr[0]), int(model.jnt_dofadr[0])) != (0, 0):
        errors.append("free-root qpos/qvel addresses must both start at zero")
    for actuator_index, expected_name in enumerate(HUMANOID_ACTUATOR_JOINT_ORDER):
        transmission_type = int(model.actuator_trntype[actuator_index])
        if transmission_type != int(mujoco.mjtTrn.mjTRN_JOINT):
            errors.append(f"actuator[{actuator_index}] is not a joint transmission")
            continue
        joint_index = int(model.actuator_trnid[actuator_index, 0])
        joint_name = _mujoco_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_index)
        actuator_name = _mujoco_name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_index,
        )
        names.append(joint_name or "")
        if joint_name != expected_name:
            errors.append(
                f"actuator[{actuator_index}] drives {joint_name!r}; expected {expected_name!r}"
            )
        if actuator_name != expected_name:
            errors.append(
                f"actuator[{actuator_index}] is named {actuator_name!r}; expected {expected_name!r}"
            )
        if int(model.jnt_type[joint_index]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            errors.append(f"joint {expected_name!r} is not a hinge")
        if int(model.actuator_dyntype[actuator_index]) != int(mujoco.mjtDyn.mjDYN_NONE):
            errors.append(f"actuator {expected_name!r} must have no activation dynamics")
        if int(model.actuator_gaintype[actuator_index]) != int(mujoco.mjtGain.mjGAIN_FIXED):
            errors.append(f"actuator {expected_name!r} must use fixed gain")
        gain_parameters = np.asarray(model.actuator_gainprm[actuator_index], dtype=np.float64)
        if gain_parameters[0] != 1.0 or np.any(gain_parameters[1:] != 0.0):
            errors.append(f"actuator {expected_name!r} must use unit fixed gain")
        if int(model.actuator_biastype[actuator_index]) != int(mujoco.mjtBias.mjBIAS_NONE):
            errors.append(f"actuator {expected_name!r} must have no actuator bias")
        if np.any(np.asarray(model.actuator_biasprm[actuator_index], dtype=np.float64) != 0.0):
            errors.append(f"actuator {expected_name!r} bias parameters must be zero")
        if bool(model.actuator_forcelimited[actuator_index]):
            errors.append(f"actuator {expected_name!r} must not have a force clamp")
        if not bool(model.actuator_ctrllimited[actuator_index]):
            errors.append(f"actuator {expected_name!r} must have a control clamp")
        gear = np.asarray(model.actuator_gear[actuator_index], dtype=np.float64)
        control_range = np.asarray(model.actuator_ctrlrange[actuator_index], dtype=np.float64)
        if (
            gear.shape != (6,)
            or not np.isfinite(gear).all()
            or gear[0] == 0.0
            or np.any(gear[1:] != 0.0)
        ):
            errors.append(f"actuator {expected_name!r} must use one finite scalar joint gear")
        elif (
            control_range.shape != (2,)
            or not np.isfinite(control_range).all()
            or control_range[1] <= control_range[0]
        ):
            errors.append(f"actuator {expected_name!r} control range must be finite and ordered")
        elif control_range[0] != -control_range[1]:
            errors.append(
                f"actuator {expected_name!r} control range must be exactly symmetric about zero"
            )
        else:
            observed_gear = float(gear[0])
            expected_gear = HUMANOID_ACTUATOR_GEAR_BY_JOINT[actuator_index]
            actuator_gears.append(observed_gear)
            if observed_gear != expected_gear:
                errors.append(
                    f"actuator {expected_name!r} signed gear is {observed_gear!r}; "
                    f"expected {expected_gear!r}"
                )
            expected_control_range = np.asarray(
                HUMANOID_ACTUATOR_CONTROL_RANGE,
                dtype=np.float64,
            )
            if not np.array_equal(control_range, expected_control_range):
                errors.append(
                    f"actuator {expected_name!r} control range is "
                    f"{tuple(float(value) for value in control_range)!r}; expected "
                    f"{HUMANOID_ACTUATOR_CONTROL_RANGE!r}"
                )
            observed_capacity = abs(observed_gear) * max(
                abs(float(control_range[0])),
                abs(float(control_range[1])),
            )
            torque_capacities.append(observed_capacity)
            expected_capacity = HUMANOID_GENERALIZED_ACTUATOR_TORQUE_CAPACITY_N_M[actuator_index]
            if observed_capacity != expected_capacity:
                errors.append(
                    f"actuator {expected_name!r} torque capacity is "
                    f"{observed_capacity!r} N*m; expected {expected_capacity!r} N*m"
                )
        qpos_indices.append(int(model.jnt_qposadr[joint_index]))
        qvel_indices.append(int(model.jnt_dofadr[joint_index]))

    if tuple(qpos_indices) != HUMANOID_QPOS_INDICES_BY_ACTUATOR:
        errors.append(
            f"actuator qpos addresses are {tuple(qpos_indices)!r}; "
            f"expected {HUMANOID_QPOS_INDICES_BY_ACTUATOR!r}"
        )
    if tuple(qvel_indices) != HUMANOID_QVEL_INDICES_BY_ACTUATOR:
        errors.append(
            f"actuator qvel addresses are {tuple(qvel_indices)!r}; "
            f"expected {HUMANOID_QVEL_INDICES_BY_ACTUATOR!r}"
        )
    if tuple(actuator_gears) != HUMANOID_ACTUATOR_GEAR_BY_JOINT:
        errors.append("signed actuator gear vector does not match the frozen Humanoid-v5 ABI")
    if tuple(torque_capacities) != HUMANOID_GENERALIZED_ACTUATOR_TORQUE_CAPACITY_N_M:
        errors.append("actuator torque-capacity vector does not match the frozen Humanoid-v5 ABI")
    action_dtype = np.dtype(action_space.dtype)
    if action_dtype.str != "<f4":
        errors.append("action Box dtype must be exact little-endian float32")
    expected_action_low = np.asarray(model.actuator_ctrlrange[:, 0], dtype=action_dtype)
    expected_action_high = np.asarray(model.actuator_ctrlrange[:, 1], dtype=action_dtype)
    if not np.array_equal(
        np.asarray(action_space.low, dtype=action_dtype), expected_action_low
    ) or not np.array_equal(
        np.asarray(action_space.high, dtype=action_dtype), expected_action_high
    ):
        errors.append("action Box bounds must match actuator control ranges")

    control_period = float(getattr(unwrapped, "dt", math.nan))
    if not math.isfinite(control_period) or not math.isclose(
        control_period,
        HUMANOID_CONTROL_PERIOD_SECONDS,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        errors.append(
            f"control period is {control_period!r}; expected {HUMANOID_CONTROL_PERIOD_SECONDS!r}"
        )
    if bool(getattr(unwrapped, "_terminate_when_unhealthy", True)):
        errors.append("terminate_when_unhealthy must be False for recovery-capable episodes")
    if bool(getattr(unwrapped, "_exclude_current_positions_from_observation", False)) is not True:
        errors.append("exclude_current_positions_from_observation must be True")
    if int(getattr(unwrapped, "frame_skip", -1)) != 5:
        errors.append("frame_skip must be 5")
    if errors:
        raise OracleContractError("Humanoid actuator ABI mismatch: " + "; ".join(errors))

    return HumanoidActuatorABI(
        joint_names=tuple(names),
        qpos_indices=tuple(qpos_indices),
        qvel_indices=tuple(qvel_indices),
        actuator_gear_by_joint=tuple(actuator_gears),
        generalized_actuator_torque_capacity_n_m=tuple(torque_capacities),
        control_period_seconds=control_period,
    )


def validate_humanoid_reference(reference: ReferenceArtifact) -> None:
    """Verify identity, schema, quaternion, and broad numerical safety.

    Passing this function does not establish joint-limit compatibility,
    contact consistency, kinematic realizability, or dynamics feasibility.
    Those require separate admission evidence tied to the exact simulator.
    """

    reference.verify()
    assert_exact_schema(HUMANOID_REFERENCE_SCHEMA, reference.schema)
    for frame_index, frame in enumerate(reference.values):
        quaternion = np.asarray(frame[1:5], dtype=np.float64)
        norm = float(np.linalg.norm(quaternion))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-8):
            raise OracleContractError(
                f"reference frame {frame_index} root quaternion must have unit norm"
            )
        first_nonzero = next((value for value in quaternion if abs(value) > 1e-12), 0.0)
        if first_nonzero < 0.0:
            raise OracleContractError(
                f"reference frame {frame_index} root quaternion must use canonical sign"
            )
        bounded_fields = (
            (
                "root height",
                np.asarray(frame[0:1], dtype=np.float64),
                MIN_REFERENCE_ROOT_HEIGHT_M,
                MAX_REFERENCE_ROOT_HEIGHT_M,
            ),
            (
                "root linear velocity component",
                np.asarray(frame[5:8], dtype=np.float64),
                -MAX_REFERENCE_ROOT_LINEAR_SPEED_COMPONENT_M_S,
                MAX_REFERENCE_ROOT_LINEAR_SPEED_COMPONENT_M_S,
            ),
            (
                "root angular velocity component",
                np.asarray(frame[8:11], dtype=np.float64),
                -MAX_REFERENCE_ROOT_ANGULAR_SPEED_COMPONENT_RAD_S,
                MAX_REFERENCE_ROOT_ANGULAR_SPEED_COMPONENT_RAD_S,
            ),
            (
                "joint position",
                np.asarray(frame[11:28], dtype=np.float64),
                -MAX_REFERENCE_JOINT_POSITION_ABS_RAD,
                MAX_REFERENCE_JOINT_POSITION_ABS_RAD,
            ),
            (
                "joint velocity",
                np.asarray(frame[28:45], dtype=np.float64),
                -MAX_REFERENCE_JOINT_SPEED_ABS_RAD_S,
                MAX_REFERENCE_JOINT_SPEED_ABS_RAD_S,
            ),
        )
        for field, values, lower, upper in bounded_fields:
            if np.any(values < lower) or np.any(values > upper):
                raise OracleContractError(
                    f"reference frame {frame_index} {field} exceeds numerical safety bounds"
                )


def make_static_stand_reference(*, n_frames: int = 1) -> ReferenceArtifact:
    """Create an immutable static default-pose stand candidate.

    This reproduces Gymnasium Humanoid-v5's free-root and actuated ``init_qpos``
    and ``init_qvel`` coordinates, except for intentionally omitted root
    ``x/y``. It is a positive-control *candidate*, not evidence that a learned
    controller can balance or recover.
    """

    if not isinstance(n_frames, int) or isinstance(n_frames, bool) or n_frames < 1:
        raise OracleContractError("n_frames must be a positive integer")
    frame = (
        1.4,  # world-frame root height from Humanoid-v5 init_qpos
        1.0,
        0.0,
        0.0,
        0.0,  # world-frame identity quaternion, wxyz
        0.0,
        0.0,
        0.0,  # world-frame root linear velocity
        0.0,
        0.0,
        0.0,  # torso-local root angular velocity
    ) + (0.0,) * (2 * len(HUMANOID_ACTUATOR_JOINT_ORDER))
    return ReferenceArtifact.create(
        artifact_id="gymnasium/Humanoid-v5/static_stand_init_pose/v1",
        schema=HUMANOID_REFERENCE_SCHEMA,
        values=(frame,) * n_frames,
    )


def actuated_state(
    env: object,
    abi: HumanoidActuatorABI,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Read finite joint positions and velocities in actuator order."""

    unwrapped = getattr(env, "unwrapped", env)
    data = getattr(unwrapped, "data", None)
    if data is None:
        raise OracleContractError("environment does not expose MuJoCo state data")
    positions = np.asarray(data.qpos[list(abi.qpos_indices)], dtype=np.float64)
    velocities = np.asarray(data.qvel[list(abi.qvel_indices)], dtype=np.float64)
    expected = (len(HUMANOID_ACTUATOR_JOINT_ORDER),)
    if positions.shape != expected or velocities.shape != expected:
        raise OracleContractError("actuated state does not match the 17-joint ABI")
    if not np.isfinite(positions).all() or not np.isfinite(velocities).all():
        raise OracleContractError("actuated state contains non-finite values")
    return positions.copy(), velocities.copy()


def _tracking_state(
    env: object,
    abi: HumanoidActuatorABI,
    *,
    normalize_bounded_root_orientation: bool,
) -> HumanoidTrackingState:
    unwrapped = getattr(env, "unwrapped", env)
    data = getattr(unwrapped, "data", None)
    if data is None:
        raise OracleContractError("environment does not expose MuJoCo state data")
    qpos = np.asarray(data.qpos, dtype=np.float64)
    qvel = np.asarray(data.qvel, dtype=np.float64)
    if qpos.shape != (24,) or qvel.shape != (23,):
        raise OracleContractError("Humanoid state vectors must have shapes (24,) and (23,)")
    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise OracleContractError("Humanoid state contains non-finite values")
    quaternion = qpos[3:7].copy()
    norm = float(np.linalg.norm(quaternion))
    if normalize_bounded_root_orientation:
        if any(
            not lower <= float(component) <= upper
            for component, (lower, upper) in zip(
                quaternion,
                HUMANOID_RESET_ROOT_QUATERNION_COMPONENT_BOUNDS,
                strict=True,
            )
        ):
            raise OracleContractError(
                "Humanoid reset root quaternion is outside the frozen noise envelope"
            )
        lower, upper = HUMANOID_NOISY_ROOT_QUATERNION_NORM_BOUNDS
        if not lower <= norm <= upper:
            raise OracleContractError(
                "Humanoid noisy root quaternion norm is outside the frozen bounds"
            )
        quaternion /= norm
    elif not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise OracleContractError("Humanoid root quaternion is not unit length")
    positions, velocities = actuated_state(env, abi)
    return HumanoidTrackingState(
        root_position_world_m=qpos[0:3].copy(),
        root_height_m=float(qpos[2]),
        root_orientation_wxyz=quaternion,
        root_linear_velocity_world_m_s=qvel[0:3].copy(),
        root_angular_velocity_body_rad_s=qvel[3:6].copy(),
        joint_positions_rad=positions,
        joint_velocities_rad_s=velocities,
    )


def tracking_state(env: object, abi: HumanoidActuatorABI) -> HumanoidTrackingState:
    """Read state while requiring an already unit-length root quaternion."""

    return _tracking_state(env, abi, normalize_bounded_root_orientation=False)


def tracking_state_with_bounded_reset_orientation(
    env: object,
    abi: HumanoidActuatorABI,
) -> HumanoidTrackingState:
    """Read reset state with the frozen Gym noise envelope and projection."""

    return _tracking_state(env, abi, normalize_bounded_root_orientation=True)
