from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest

from oracle_composition.contracts import OracleContractError, ReferenceArtifact
from oracle_composition.envs.humanoid import make_humanoid_env
from oracle_composition.envs.reference_tracking import FixedReferenceTrackingWrapper
from oracle_composition.tracking import (
    HUMANOID_REFERENCE_SCHEMA,
    make_static_stand_reference,
)


@pytest.mark.gym
def test_wrapper_exposes_one_finite_flat_box_to_actor_and_critic() -> None:
    env = FixedReferenceTrackingWrapper(
        make_humanoid_env(),
        reference=make_static_stand_reference(n_frames=2),
        horizon_steps=4,
    )
    try:
        observation, info = env.reset(seed=20260902)
    finally:
        env.close()

    assert isinstance(env.observation_space, gym.spaces.Box)
    assert env.observation_space.shape == (348 + 4 * 45,)
    assert observation.shape == env.observation_space.shape
    assert observation.dtype == env.observation_space.dtype
    assert np.isfinite(observation).all()
    reference_tail = observation[-4 * 45 :].reshape(4, 45)
    assert np.all(reference_tail[:, 0] == 1.4)
    assert np.all(reference_tail[:, 1] == 1.0)
    assert np.count_nonzero(reference_tail[:, 2:]) == 0
    assert info["reference_tracking"]["reference_window_shape"] == [4, 45]
    assert info["reference_tracking"]["policy_reference_window_frame_indices"] == [0, 1, 1, 1]


@pytest.mark.gym
def test_step_uses_only_fixed_tracking_reward_and_reports_zero_task_reward() -> None:
    env = FixedReferenceTrackingWrapper(
        make_humanoid_env(),
        reference=make_static_stand_reference(),
        horizon_steps=2,
    )
    try:
        env.reset(seed=20260902)
        observation, reward, terminated, truncated, info = env.step(
            np.zeros(17, dtype=env.action_space.dtype)
        )
    finally:
        env.close()

    telemetry = info["reference_tracking"]
    assert observation.shape == env.observation_space.shape
    assert np.isfinite(observation).all()
    assert reward == pytest.approx(telemetry["tracking_reward"]["total"])
    assert telemetry["task_reward"] == 0.0
    assert telemetry["ignored_environment_reward"] != reward
    assert telemetry["reward_target_frame_index"] == 0
    assert telemetry["reward_temporal_alignment"] == "post_step_state_against_next_reference_frame"
    assert set(telemetry["tracking_error"]) == {
        "root_height_abs_error_m",
        "root_orientation_error_rad",
        "root_linear_velocity_rmse_m_s",
        "root_angular_velocity_rmse_rad_s",
        "joint_position_rmse_rad",
        "joint_velocity_rmse_rad_s",
        "joint_position_max_abs_rad",
        "joint_velocity_max_abs_rad_s",
    }
    assert len(telemetry["reference_content_sha256"]) == 64
    assert len(telemetry["reference_schema_sha256"]) == 64
    assert len(telemetry["tracking_reward_sha256"]) == 64
    assert terminated is False
    assert truncated is False


@pytest.mark.gym
def test_post_step_state_is_graded_against_visible_next_reference_frame() -> None:
    first = make_static_stand_reference().values[0]
    second = first[:11] + (1.0,) * 17 + (0.0,) * 17
    reference = ReferenceArtifact.create(
        artifact_id="gymnasium/Humanoid-v5/two_frame_test/v1",
        schema=HUMANOID_REFERENCE_SCHEMA,
        values=(first, second),
    )
    env = FixedReferenceTrackingWrapper(
        make_humanoid_env(),
        reference=reference,
        horizon_steps=2,
    )
    try:
        initial, _ = env.reset(seed=20260902)
        following, _, _, _, info = env.step(np.zeros(17, dtype=env.action_space.dtype))
    finally:
        env.close()

    initial_window = initial[-90:].reshape(2, 45)
    following_window = following[-90:].reshape(2, 45)
    assert np.array_equal(initial_window[0], np.asarray(first))
    assert np.array_equal(initial_window[1], np.asarray(second))
    assert np.array_equal(following_window[0], np.asarray(second))
    assert np.array_equal(following_window[1], np.asarray(second))
    telemetry = info["reference_tracking"]
    assert telemetry["policy_reference_window_frame_indices"] == [0, 1]
    assert telemetry["reward_target_frame_index"] == 1
    assert telemetry["tracking_error"]["joint_position_rmse_rad"] > 0.9


@pytest.mark.gym
def test_wrapper_rejects_reference_with_wrong_schema() -> None:
    wrong_schema = replace(
        HUMANOID_REFERENCE_SCHEMA,
        root_frame="world",
    )
    reference = ReferenceArtifact.create(
        artifact_id="gymnasium/Humanoid-v5/wrong_root/v1",
        schema=wrong_schema,
        values=((0.0,) * wrong_schema.width,),
    )
    env = make_humanoid_env()
    try:
        with pytest.raises(OracleContractError, match="root_frame"):
            FixedReferenceTrackingWrapper(env, reference=reference, horizon_steps=1)
    finally:
        env.close()


@pytest.mark.gym
@pytest.mark.parametrize("horizon_steps", [0, 1])
def test_wrapper_rejects_invalid_horizon(horizon_steps: int) -> None:
    env = make_humanoid_env()
    try:
        with pytest.raises(OracleContractError, match="horizon_steps"):
            FixedReferenceTrackingWrapper(
                env,
                reference=make_static_stand_reference(),
                horizon_steps=horizon_steps,
            )
    finally:
        env.close()
