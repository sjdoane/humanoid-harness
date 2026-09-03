"""Frozen reference-tracking primitives independent of oracle selection."""

from .humanoid_reference import (
    HUMANOID_ACTUATOR_JOINT_ORDER,
    HUMANOID_QPOS_INDICES_BY_ACTUATOR,
    HUMANOID_QVEL_INDICES_BY_ACTUATOR,
    HUMANOID_REFERENCE_CADENCE_HZ,
    HUMANOID_REFERENCE_SCHEMA,
    HumanoidActuatorABI,
    HumanoidTrackingState,
    actuated_state,
    make_static_stand_reference,
    tracking_state,
    validate_humanoid_actuator_abi,
    validate_humanoid_reference,
)
from .reward import (
    TrackingRewardConfig,
    TrackingRewardResult,
    compute_tracking_reward,
)

__all__ = [
    "HUMANOID_ACTUATOR_JOINT_ORDER",
    "HUMANOID_QPOS_INDICES_BY_ACTUATOR",
    "HUMANOID_QVEL_INDICES_BY_ACTUATOR",
    "HUMANOID_REFERENCE_CADENCE_HZ",
    "HUMANOID_REFERENCE_SCHEMA",
    "HumanoidActuatorABI",
    "HumanoidTrackingState",
    "TrackingRewardConfig",
    "TrackingRewardResult",
    "actuated_state",
    "compute_tracking_reward",
    "make_static_stand_reference",
    "tracking_state",
    "validate_humanoid_actuator_abi",
    "validate_humanoid_reference",
]
