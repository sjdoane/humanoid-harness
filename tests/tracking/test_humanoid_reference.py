from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from oracle_composition.contracts import OracleContractError, ReferenceArtifact
from oracle_composition.envs.humanoid import make_humanoid_env
from oracle_composition.tracking import (
    HUMANOID_ACTUATOR_JOINT_ORDER,
    HUMANOID_REFERENCE_CADENCE_HZ,
    HUMANOID_REFERENCE_SCHEMA,
    actuated_state,
    make_static_stand_reference,
    validate_humanoid_actuator_abi,
    validate_humanoid_reference,
)


def test_reference_schema_is_actuator_ordered_and_root_explicitly_omitted() -> None:
    assert len(HUMANOID_ACTUATOR_JOINT_ORDER) == 17
    assert HUMANOID_REFERENCE_SCHEMA.width == 45
    assert HUMANOID_REFERENCE_SCHEMA.feature_names[:11] == (
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
    )
    assert HUMANOID_REFERENCE_SCHEMA.feature_names[11:28] == tuple(
        f"joint_position.{name}" for name in HUMANOID_ACTUATOR_JOINT_ORDER
    )
    assert HUMANOID_REFERENCE_SCHEMA.feature_names[28:] == tuple(
        f"joint_velocity.{name}" for name in HUMANOID_ACTUATOR_JOINT_ORDER
    )
    assert HUMANOID_REFERENCE_SCHEMA.feature_units == (
        ("m", "1", "1", "1", "1") + ("m/s",) * 3 + ("rad/s",) * 3 + ("rad",) * 17 + ("rad/s",) * 17
    )
    assert "quaternion" in HUMANOID_REFERENCE_SCHEMA.root_frame
    assert "x+y_omitted" in HUMANOID_REFERENCE_SCHEMA.root_frame
    assert HUMANOID_REFERENCE_SCHEMA.cadence_hz == pytest.approx(1.0 / 0.015)


def test_static_stand_reference_is_hashed_default_actuated_pose_candidate() -> None:
    reference = make_static_stand_reference(n_frames=3)

    assert reference.identity.n_frames == 3
    assert reference.identity.n_features == 45
    assert reference.identity.schema_sha256 == HUMANOID_REFERENCE_SCHEMA.sha256
    expected = (1.4, 1.0, 0.0, 0.0, 0.0) + (0.0,) * 40
    assert reference.values == (expected,) * 3
    validate_humanoid_reference(reference)


@pytest.mark.parametrize("n_frames", [0, -1, True, 1.5])
def test_static_stand_reference_rejects_invalid_length(n_frames: object) -> None:
    with pytest.raises(OracleContractError, match="n_frames"):
        make_static_stand_reference(n_frames=n_frames)  # type: ignore[arg-type]


def test_reference_validator_rejects_schema_semantic_change() -> None:
    changed_schema = replace(
        HUMANOID_REFERENCE_SCHEMA,
        cadence_hz=HUMANOID_REFERENCE_CADENCE_HZ / 2.0,
    )
    reference = ReferenceArtifact.create(
        artifact_id="gymnasium/Humanoid-v5/wrong_cadence/v1",
        schema=changed_schema,
        values=((0.0,) * changed_schema.width,),
    )

    with pytest.raises(OracleContractError, match="cadence_hz"):
        validate_humanoid_reference(reference)


def test_reference_validator_rejects_nonunit_or_noncanonical_quaternion() -> None:
    for quaternion in ((0.5, 0.0, 0.0, 0.0), (-1.0, 0.0, 0.0, 0.0)):
        frame = (1.4, *quaternion) + (0.0,) * 40
        reference = ReferenceArtifact.create(
            artifact_id="gymnasium/Humanoid-v5/bad_quaternion/v1",
            schema=HUMANOID_REFERENCE_SCHEMA,
            values=(frame,),
        )
        with pytest.raises(OracleContractError, match="quaternion"):
            validate_humanoid_reference(reference)


@pytest.mark.parametrize(
    ("feature_index", "unsafe_value"),
    [
        (0, -0.01),
        (0, 5.01),
        (5, 25.01),
        (8, -100.01),
        (11, 2.0 * np.pi + 0.01),
        (28, -100.01),
    ],
)
def test_reference_validator_rejects_unit_mismatch_or_unsafe_magnitude(
    feature_index: int,
    unsafe_value: float,
) -> None:
    frame = list(make_static_stand_reference().values[0])
    frame[feature_index] = unsafe_value
    reference = ReferenceArtifact.create(
        artifact_id="gymnasium/Humanoid-v5/unsafe_numeric_candidate/v1",
        schema=HUMANOID_REFERENCE_SCHEMA,
        values=(tuple(frame),),
    )

    with pytest.raises(OracleContractError, match="numerical safety"):
        validate_humanoid_reference(reference)


@pytest.mark.gym
def test_real_humanoid_actuator_mapping_matches_declared_order_and_addresses() -> None:
    env = make_humanoid_env()
    try:
        env.reset(seed=20260902)
        abi = validate_humanoid_actuator_abi(env)
        positions, velocities = actuated_state(env, abi)
        init_positions = env.unwrapped.init_qpos[list(abi.qpos_indices)]
        init_velocities = env.unwrapped.init_qvel[list(abi.qvel_indices)]
    finally:
        env.close()

    assert abi.joint_names == HUMANOID_ACTUATOR_JOINT_ORDER
    assert abi.qpos_indices == (8, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)
    assert abi.qvel_indices == (7, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
    assert abi.generalized_actuator_torque_capacity_n_m == (
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
    assert positions.shape == (17,)
    assert velocities.shape == (17,)
    assert np.array_equal(init_positions, np.zeros(17))
    assert np.array_equal(init_velocities, np.zeros(17))


@pytest.mark.gym
def test_actuator_validator_rejects_unhealthy_termination() -> None:
    import gymnasium as gym

    env = gym.make(
        "Humanoid-v5",
        terminate_when_unhealthy=True,
        reset_noise_scale=0.01,
        exclude_current_positions_from_observation=True,
        frame_skip=5,
    )
    try:
        with pytest.raises(OracleContractError, match="terminate_when_unhealthy"):
            validate_humanoid_actuator_abi(env)
    finally:
        env.close()


@pytest.mark.gym
def test_actuator_validator_rejects_one_float32_ulp_action_bound_drift() -> None:
    import gymnasium as gym

    env = make_humanoid_env()
    original = env.action_space
    low = original.low.copy()
    low[0] = np.nextafter(low[0], np.float32(0.0), dtype=np.float32)
    assert abs(float(low[0] - original.low[0])) < 1e-7
    env.action_space = gym.spaces.Box(
        low=low,
        high=original.high.copy(),
        dtype=original.dtype,
    )
    try:
        with pytest.raises(OracleContractError, match="action Box bounds"):
            validate_humanoid_actuator_abi(env)
    finally:
        env.close()


@pytest.mark.gym
def test_actuator_validator_requires_exactly_symmetric_control_ranges() -> None:
    import gymnasium as gym

    env = make_humanoid_env()
    original = env.action_space
    env.unwrapped.model.actuator_ctrlrange[0, 0] = -0.3
    low = np.asarray(env.unwrapped.model.actuator_ctrlrange[:, 0], dtype=original.dtype)
    high = np.asarray(env.unwrapped.model.actuator_ctrlrange[:, 1], dtype=original.dtype)
    env.action_space = gym.spaces.Box(low=low, high=high, dtype=original.dtype)
    try:
        with pytest.raises(OracleContractError, match="exactly symmetric"):
            validate_humanoid_actuator_abi(env)
    finally:
        env.close()


@pytest.mark.gym
def test_actuator_validator_rejects_in_memory_signed_gear_drift() -> None:
    env = make_humanoid_env()
    env.unwrapped.model.actuator_gear[0, 0] = 101.0
    try:
        with pytest.raises(OracleContractError, match="signed gear"):
            validate_humanoid_actuator_abi(env)
    finally:
        env.close()
