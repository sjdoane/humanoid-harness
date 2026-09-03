from __future__ import annotations

import math
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from oracle_composition.contracts import OracleContractError
from oracle_composition.envs.humanoid import make_humanoid_env
from oracle_composition.tracking import (
    HumanoidTrackingState,
    TrackingRewardConfig,
    compute_tracking_reward,
    make_static_stand_reference,
    tracking_state,
    validate_humanoid_actuator_abi,
)


def _target_state(**changes: object) -> HumanoidTrackingState:
    state = HumanoidTrackingState(
        root_position_world_m=np.asarray([0.0, 0.0, 1.4]),
        root_height_m=1.4,
        root_orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
        root_linear_velocity_world_m_s=np.zeros(3),
        root_angular_velocity_body_rad_s=np.zeros(3),
        joint_positions_rad=np.zeros(17),
        joint_velocities_rad_s=np.zeros(17),
    )
    return replace(state, **changes)


def _stand_frame() -> tuple[float, ...]:
    return make_static_stand_reference().values[0]


def test_exact_reference_receives_unit_tracking_scores() -> None:
    result = compute_tracking_reward(
        state=_target_state(),
        reference_frame=_stand_frame(),
    )

    assert result.total == pytest.approx(1.0)
    assert result.root_pose_gate == pytest.approx(1.0)
    assert all(score == pytest.approx(1.0) for score in result.reward_components().values())
    assert all(error == 0.0 for error in result.error_components().values())


def test_reward_components_and_unit_bearing_errors_are_independently_reported() -> None:
    config = TrackingRewardConfig()
    result = compute_tracking_reward(
        state=_target_state(joint_positions_rad=np.full(17, config.joint_position_scale_rad)),
        reference_frame=_stand_frame(),
        config=config,
    )

    assert result.joint_position_score == pytest.approx(math.exp(-1.0))
    assert result.root_pose_gate == pytest.approx(1.0)
    assert result.total == pytest.approx(0.5 * math.exp(-1.0) + 0.5)
    assert result.joint_position_rmse_rad == pytest.approx(0.35)
    assert result.joint_position_max_abs_rad == pytest.approx(0.35)
    assert result.reward_components()["total"] == result.total
    assert result.error_components()["joint_position_rmse_rad"] == pytest.approx(0.35)


def test_quaternion_score_is_sign_invariant() -> None:
    result = compute_tracking_reward(
        state=_target_state(root_orientation_wxyz=np.asarray([-1.0, 0.0, 0.0, 0.0])),
        reference_frame=_stand_frame(),
    )

    assert result.root_orientation_error_rad == 0.0
    assert result.total == pytest.approx(1.0)


def test_fallen_root_cannot_exploit_matching_joint_pose() -> None:
    fallen = compute_tracking_reward(
        state=_target_state(root_height_m=0.2),
        reference_frame=_stand_frame(),
    )
    moving_toward_target = compute_tracking_reward(
        state=_target_state(root_height_m=0.3),
        reference_frame=_stand_frame(),
    )

    assert fallen.joint_position_score == pytest.approx(1.0)
    assert fallen.joint_velocity_score == pytest.approx(1.0)
    assert fallen.root_pose_gate < 0.03
    assert fallen.total < 0.03
    assert moving_toward_target.total > fallen.total


def test_sideways_root_cannot_exploit_matching_joint_pose() -> None:
    half_angle = math.pi / 4.0
    result = compute_tracking_reward(
        state=_target_state(
            root_orientation_wxyz=np.asarray([math.cos(half_angle), math.sin(half_angle), 0.0, 0.0])
        ),
        reference_frame=_stand_frame(),
    )

    assert result.root_orientation_error_rad == pytest.approx(math.pi / 2.0)
    assert result.joint_position_score == pytest.approx(1.0)
    assert result.total < 0.01


@pytest.mark.gym
def test_real_humanoid_low_root_state_collapses_static_stand_score() -> None:
    env = make_humanoid_env()
    try:
        env.reset(seed=20260902)
        abi = validate_humanoid_actuator_abi(env)
        env.unwrapped.data.qpos[:] = env.unwrapped.init_qpos
        env.unwrapped.data.qvel[:] = env.unwrapped.init_qvel
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
        upright_state = tracking_state(env, abi)
        np.testing.assert_array_equal(
            upright_state.root_position_world_m,
            env.unwrapped.data.qpos[:3],
        )
        upright = compute_tracking_reward(
            state=upright_state,
            reference_frame=_stand_frame(),
        )

        env.unwrapped.data.qpos[2] = 0.2
        mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
        low = compute_tracking_reward(
            state=tracking_state(env, abi),
            reference_frame=_stand_frame(),
        )
    finally:
        env.close()

    assert upright.total == pytest.approx(1.0)
    assert low.joint_position_score == pytest.approx(1.0)
    assert low.total < 0.03


def test_reward_configuration_hash_binds_every_parameter() -> None:
    base = TrackingRewardConfig()
    changed = TrackingRewardConfig(root_height_scale_m=0.25)

    assert len(base.sha256) == 64
    assert base.sha256 != changed.sha256


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("joint_positions_rad", np.zeros(16)),
        ("joint_velocities_rad_s", np.zeros(18)),
        ("root_linear_velocity_world_m_s", np.zeros(2)),
        ("root_angular_velocity_body_rad_s", np.full(3, np.inf)),
        ("root_orientation_wxyz", np.asarray([0.5, 0.0, 0.0, 0.0])),
        ("joint_positions_rad", np.full(17, np.nan)),
    ],
)
def test_reward_rejects_wrong_shape_nonfinite_or_nonunit_state(
    field: str,
    value: np.ndarray,
) -> None:
    with pytest.raises(OracleContractError, match=field):
        compute_tracking_reward(
            state=_target_state(**{field: value}),
            reference_frame=_stand_frame(),
        )


def test_reward_rejects_wrong_reference_dimension() -> None:
    with pytest.raises(OracleContractError, match="reference_frame"):
        compute_tracking_reward(
            state=_target_state(),
            reference_frame=np.zeros(44),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"root_height_scale_m": 0.0},
        {"root_orientation_scale_rad": float("nan")},
        {"joint_velocity_scale_rad_s": float("inf")},
        {"joint_position_weight": -0.1, "joint_velocity_weight": 0.9},
        {
            "root_linear_velocity_weight": 0.1,
            "root_angular_velocity_weight": 0.1,
            "joint_position_weight": 0.4,
            "joint_velocity_weight": 0.3,
        },
    ],
)
def test_invalid_reward_configuration_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(OracleContractError):
        TrackingRewardConfig(**kwargs)
