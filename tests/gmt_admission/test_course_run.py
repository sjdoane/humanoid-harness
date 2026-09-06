from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from oracle_composition.adapters.gmt.course_run import (
    _numeric_policy,
    _TrainingProgress,
    make_policy,
)
from oracle_composition.adapters.gmt.io import sha256_file


class NumericFixture(gym.Env):
    """No robot or dynamics: exercise the actual PPO adapter with bounded numbers."""

    def __init__(self):
        self.observation_space = gym.spaces.Box(-10.0, 10.0, (2171,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (23,), dtype=np.float32)
        self.tick = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.tick = 0
        return np.zeros(2171, dtype=np.float32), {}

    def step(self, action):
        assert self.action_space.contains(action)
        self.tick += 1
        obs = np.full(2171, self.tick / 100, dtype=np.float32)
        return (
            obs,
            float(1 - np.square(action).mean()),
            False,
            self.tick == 16,
            {"metrics": {"fallen": False}},
        )


def test_zero_initial_mean_is_exact_and_seeded_state_is_repeatable(tmp_path):
    torch.set_num_threads(1)
    first = make_policy(NumericFixture(), 19)
    observation = np.linspace(-1, 1, 2171, dtype=np.float32)
    action, _ = first.predict(observation, deterministic=True)
    np.testing.assert_array_equal(action, np.zeros(23, dtype=np.float32))
    second = make_policy(NumericFixture(), 19)
    for name, tensor in first.policy.state_dict().items():
        assert torch.equal(tensor, second.policy.state_dict()[name])
    path = tmp_path / "policy.npz"
    digest = _numeric_policy(first, path)
    assert sha256_file(path) == digest
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == set(first.policy.state_dict())
        for name, tensor in first.policy.state_dict().items():
            assert archive[name].tobytes() == tensor.numpy().tobytes()


def test_real_ppo_adapter_completes_exact_fixture_budget_and_logs_transitions(capsys):
    torch.set_num_threads(1)
    model = make_policy(NumericFixture(), 29)
    before = model.policy.action_net.weight.detach().clone()
    callback = _TrainingProgress()
    model.learn(total_timesteps=512, callback=callback)
    assert model.num_timesteps == 512
    assert callback.episodes == 32 and callback.falls == 0
    assert not torch.equal(before, model.policy.action_net.weight)
    assert '"transitions": 512' in capsys.readouterr().out
