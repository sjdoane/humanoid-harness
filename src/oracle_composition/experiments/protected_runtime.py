"""Direct simulator readers shared by protected evaluation runners."""

from __future__ import annotations

import math

import numpy as np

from .fixed_reference import ExperimentContractError
from .protected_evaluator import (
    FLOOR_GEOM_NAME,
    PERMITTED_STATIC_STAND_FLOOR_GEOMS,
    ContactFact,
)


def finite_direct_vector(value: object, *, width: int, field: str) -> np.ndarray:
    """Copy one finite fixed-width runtime vector."""

    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExperimentContractError(f"{field} is not a numeric vector") from exc
    if vector.shape != (width,) or not np.isfinite(vector).all():
        raise ExperimentContractError(f"{field} must be a finite ({width},) vector")
    return vector.copy()


def normalized_policy_action(value: object) -> np.ndarray:
    """Validate one normalized 17-actuator action."""

    action = finite_direct_vector(value, width=17, field="normalized policy action")
    if np.any(action < -1.0) or np.any(action > 1.0):
        raise ExperimentContractError("normalized policy action must be in [-1, 1]")
    return action


def initial_protected_history(environment: object, abi: object) -> tuple[np.ndarray, np.ndarray]:
    """Read reset-time control and acceleration for first-difference metrics."""

    physical = getattr(environment, "unwrapped", environment)
    data = getattr(physical, "data", None)
    action_space = getattr(physical, "action_space", None)
    qvel_indices = tuple(getattr(abi, "qvel_indices", ()))
    if data is None or action_space is None or len(qvel_indices) != 17:
        raise ExperimentContractError("physical reset history is unavailable")
    low = finite_direct_vector(
        action_space.low,
        width=17,
        field="physical action lower bound",
    )
    high = finite_direct_vector(
        action_space.high,
        width=17,
        field="physical action upper bound",
    )
    if np.any(high <= low):
        raise ExperimentContractError("physical action bounds must be strictly ordered")
    control = finite_direct_vector(data.ctrl, width=17, field="reset physical control")
    normalized_control = normalized_policy_action(2.0 * ((control - low) / (high - low)) - 1.0)
    if not np.array_equal(normalized_control, np.zeros(17, dtype=np.float64)):
        raise ExperimentContractError("reset normalized physical control must be exactly zero")
    acceleration = finite_direct_vector(
        np.asarray(data.qacc)[list(qvel_indices)],
        width=17,
        field="reset joint acceleration",
    )
    return normalized_control, acceleration


def direct_contact_facts(environment: object) -> tuple[ContactFact, ...]:
    """Read every contact captured after every physics substep."""

    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("MuJoCo is required for protected contact measurements") from exc

    physical = getattr(environment, "unwrapped", environment)
    model = getattr(physical, "model", None)
    samples = getattr(physical, "last_control_step_contact_samples", None)
    captured_substeps = getattr(physical, "last_control_step_substeps", None)
    frame_skip = getattr(physical, "frame_skip", None)
    if model is None or not isinstance(samples, tuple):
        raise ExperimentContractError("physical MuJoCo contact state is unavailable")
    if (
        not isinstance(frame_skip, int)
        or isinstance(frame_skip, bool)
        or captured_substeps != frame_skip
    ):
        raise ExperimentContractError("contact capture did not cover every MuJoCo physics substep")
    indices_by_substep: dict[int, list[int]] = {index: [] for index in range(frame_skip)}
    contacts: list[ContactFact] = []
    for flattened_index, sample in enumerate(samples):
        substep_index = getattr(sample, "physics_substep_index", None)
        contact_index = getattr(sample, "contact_index_within_substep", None)
        if (
            not isinstance(substep_index, int)
            or isinstance(substep_index, bool)
            or substep_index not in indices_by_substep
            or not isinstance(contact_index, int)
            or isinstance(contact_index, bool)
            or contact_index < 0
        ):
            raise ExperimentContractError("substep contact index is invalid")
        indices_by_substep[substep_index].append(contact_index)
        geom1_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            int(getattr(sample, "geom1_id", -1)),
        )
        geom2_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            int(getattr(sample, "geom2_id", -1)),
        )
        if not geom1_name or not geom2_name:
            raise ExperimentContractError("active contact contains an unnamed geometry")
        normal_force = getattr(sample, "normal_force_n", None)
        if (
            isinstance(normal_force, bool)
            or not isinstance(normal_force, (int, float))
            or not math.isfinite(float(normal_force))
            or float(normal_force) < 0.0
        ):
            raise ExperimentContractError("substep contact normal force is invalid")
        involves_floor = FLOOR_GEOM_NAME in (geom1_name, geom2_name)
        if involves_floor:
            other_geom = geom2_name if geom1_name == FLOOR_GEOM_NAME else geom1_name
            forbidden = other_geom not in PERMITTED_STATIC_STAND_FLOOR_GEOMS
        else:
            forbidden = False
        contacts.append(
            ContactFact(
                contact_index=flattened_index,
                geom1_name=geom1_name,
                geom2_name=geom2_name,
                involves_floor=involves_floor,
                forbidden_floor_contact=forbidden,
                normal_force_n=float(normal_force),
            )
        )
    for substep_index, contact_indices in indices_by_substep.items():
        if contact_indices != list(range(len(contact_indices))):
            raise ExperimentContractError(
                f"contact capture is incomplete at physics substep {substep_index}"
            )
    return tuple(contacts)


def protected_step_inputs(
    environment: object,
    abi: object,
    *,
    normalized_action: object,
    previous_normalized_action: object,
    previous_joint_acceleration: object,
) -> dict[str, object]:
    """Read reward-independent post-step control, effort, jerk, and contacts."""

    physical = getattr(environment, "unwrapped", environment)
    data = getattr(physical, "data", None)
    qvel_indices = tuple(getattr(abi, "qvel_indices", ()))
    control_period = getattr(abi, "control_period_seconds", None)
    if data is None or len(qvel_indices) != 17:
        raise ExperimentContractError("physical protected step state is unavailable")
    action = normalized_policy_action(normalized_action)
    previous_action = normalized_policy_action(previous_normalized_action)
    torque = finite_direct_vector(
        np.asarray(data.qfrc_actuator)[list(qvel_indices)],
        width=17,
        field="post-transmission generalized actuator torque",
    )
    torque_capacity = finite_direct_vector(
        getattr(abi, "generalized_actuator_torque_capacity_n_m", None),
        width=17,
        field="generalized actuator torque capacity",
    )
    if np.any(torque_capacity <= 0.0):
        raise ExperimentContractError("generalized actuator torque capacity must be positive")
    acceleration = finite_direct_vector(
        np.asarray(data.qacc)[list(qvel_indices)],
        width=17,
        field="joint acceleration",
    )
    previous_acceleration = finite_direct_vector(
        previous_joint_acceleration,
        width=17,
        field="previous joint acceleration",
    )
    if (
        isinstance(control_period, bool)
        or not isinstance(control_period, (int, float))
        or not math.isfinite(float(control_period))
        or float(control_period) <= 0.0
    ):
        raise ExperimentContractError("ABI control period must be finite and positive")
    return {
        "normalized_policy_action": action,
        "previous_normalized_policy_action": previous_action,
        "generalized_actuator_torque_n_m": torque,
        "generalized_actuator_torque_capacity_n_m": torque_capacity,
        "joint_accelerations_rad_s2": acceleration,
        "previous_joint_accelerations_rad_s2": previous_acceleration,
        "contact_facts": direct_contact_facts(environment),
        "control_period_seconds": float(control_period),
    }


__all__ = [
    "direct_contact_facts",
    "finite_direct_vector",
    "initial_protected_history",
    "normalized_policy_action",
    "protected_step_inputs",
]
