from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from oracle_composition.adapters.gmt.course_run import make_policy
from oracle_composition.adapters.gmt.training_contract import (
    TRAINING_REWARD_SCALE,
    CourseTrainerSpec,
)
from oracle_composition.adapters.gmt.training_wrapper import (
    TrainingRewardScale,
    training_env,
)


class NumericRewardFixture(gym.Env):
    def __init__(
        self,
        reward: float | np.float32 = 64.0,
        *,
        observation_dim: int = 2,
        action_dim: int = 1,
        truncate: bool = False,
    ) -> None:
        self.observation_space = gym.spaces.Box(
            -10.0, 10.0, (observation_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (action_dim,), dtype=np.float32)
        self.raw_reward = reward
        self.truncate = truncate
        self.reward_info = {"total_reward": float(reward), "tracking_reward": 1.0}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):
        assert self.action_space.contains(action)
        return (
            np.zeros(self.observation_space.shape, dtype=np.float32),
            self.raw_reward,
            False,
            self.truncate,
            {"reward": self.reward_info},
        )


class ContinueCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True


def test_legacy_path_is_same_object_and_factor_one_preserves_scalar_exactly() -> None:
    raw_reward = np.float32(1.25)
    env = NumericRewardFixture(raw_reward)
    assert training_env(env, None) is env

    wrapped = TrainingRewardScale(env, 1.0)
    wrapped.reset()
    _observation, reward, _terminated, _truncated, info = wrapped.step(
        np.zeros(1, dtype=np.float32)
    )

    assert reward is raw_reward
    assert info["reward"] is env.reward_info
    assert info["training_reward"]["raw_total_reward"] == float(raw_reward)
    assert info["training_reward"]["scaled_optimization_reward"] == float(raw_reward)


def test_fixed_scale_changes_only_training_scalar_and_adds_truthful_raw_info() -> None:
    env = NumericRewardFixture(64.0)
    wrapped = training_env(env, CourseTrainerSpec(TRAINING_REWARD_SCALE))
    wrapped.reset()

    observation, reward, terminated, truncated, info = wrapped.step(
        np.zeros(1, dtype=np.float32)
    )

    np.testing.assert_array_equal(observation, np.zeros(2, dtype=np.float32))
    assert reward == np.float32(1.0)
    assert terminated is False and truncated is False
    assert info["reward"] == {"total_reward": 64.0, "tracking_reward": 1.0}
    assert info["training_reward"] == {
        "schema_id": "gmt_g1_training_reward_observation/v1",
        "raw_total_reward": 64.0,
        "scaled_optimization_reward": 1.0,
        "total_training_reward_scale": 0.015625,
    }


@pytest.mark.parametrize("factor", [True, 0.0, 0.5, float("nan"), float("inf")])
def test_wrapper_rejects_unadmitted_factors(factor) -> None:
    with pytest.raises(ValueError, match="factor"):
        TrainingRewardScale(NumericRewardFixture(), factor)


def test_sb3_timeout_bootstrap_is_added_after_scaling_in_value_units(monkeypatch) -> None:
    vector_env = DummyVecEnv(
        [lambda: TrainingRewardScale(NumericRewardFixture(truncate=True), TRAINING_REWARD_SCALE)]
    )
    model = PPO(
        "MlpPolicy",
        vector_env,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        gamma=0.9,
        seed=11,
        device="cpu",
        verbose=0,
    )
    monkeypatch.setattr(
        model.policy,
        "predict_values",
        lambda observation: torch.full(
            (observation.shape[0],), 2.0, dtype=torch.float32, device=observation.device
        ),
    )
    _, callback = model._setup_learn(
        2,
        callback=ContinueCallback(),
        reset_num_timesteps=True,
        tb_log_name="timeout-units",
        progress_bar=False,
    )
    callback.on_training_start({}, {})

    assert model.collect_rollouts(
        vector_env, callback, model.rollout_buffer, n_rollout_steps=2
    )

    np.testing.assert_allclose(model.rollout_buffer.rewards[:, 0], [2.8, 2.8], rtol=0, atol=1e-6)


def test_training_wrapper_does_not_change_seeded_actor_initialization() -> None:
    torch.set_num_threads(1)
    raw = NumericRewardFixture(observation_dim=2171, action_dim=23)
    scaled = TrainingRewardScale(
        NumericRewardFixture(observation_dim=2171, action_dim=23),
        TRAINING_REWARD_SCALE,
    )

    raw_model = make_policy(raw, 23)
    scaled_model = make_policy(scaled, 23)

    for name, tensor in raw_model.policy.state_dict().items():
        assert torch.equal(tensor, scaled_model.policy.state_dict()[name])
