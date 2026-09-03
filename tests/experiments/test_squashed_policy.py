from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution

from oracle_composition.experiments.squashed_policy import (
    RolloutActionLikelihoodAudit,
    SquashedGaussianActorCriticPolicy,
    squashed_policy_source_sha256,
)


def _model(environment):
    return PPO(
        SquashedGaussianActorCriticPolicy,
        environment,
        n_steps=8,
        batch_size=4,
        n_epochs=1,
        policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}},
        seed=7,
        device="cpu",
        verbose=0,
    )


def _normalized(environment):
    dtype = environment.action_space.dtype
    shape = environment.action_space.shape
    return gym.wrappers.RescaleAction(
        environment,
        min_action=-np.ones(shape, dtype=dtype),
        max_action=np.ones(shape, dtype=dtype),
    )


def test_policy_requires_exact_normalized_box() -> None:
    physical = gym.make("Pendulum-v1")
    try:
        with pytest.raises(ValueError, match="normalized"):
            _model(physical)
    finally:
        physical.close()


def test_sample_and_recomputed_log_probability_are_the_same_bounded_action() -> None:
    base = gym.make("Pendulum-v1")
    environment = _normalized(base)
    try:
        model = _model(environment)
        observation, _ = environment.reset(seed=11)
        observation_tensor, _ = model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            action, _value, old_log_probability = model.policy(observation_tensor)
            _new_value, recomputed_log_probability, entropy = model.policy.evaluate_actions(
                observation_tensor,
                action,
            )

        normalized = action.cpu().numpy()
        assert isinstance(model.policy.action_dist, SquashedDiagGaussianDistribution)
        assert model.policy.squash_output is True
        assert np.isfinite(normalized).all()
        assert np.all(normalized > -1.0)
        assert np.all(normalized < 1.0)
        torch.testing.assert_close(old_log_probability, recomputed_log_probability)
        assert entropy is None
    finally:
        environment.close()


def test_affine_wrapper_maps_normalized_endpoints_once() -> None:
    base = gym.make("Pendulum-v1")
    physical_low = base.action_space.low.copy()
    physical_high = base.action_space.high.copy()
    environment = _normalized(base)
    try:
        for normalized, expected in (
            (-1.0, physical_low),
            (0.0, (physical_low + physical_high) / 2.0),
            (1.0, physical_high),
        ):
            environment.reset(seed=13)
            environment.step(np.full(environment.action_space.shape, normalized, dtype=np.float32))
            np.testing.assert_allclose(base.unwrapped.last_u, expected, rtol=0.0, atol=0.0)
    finally:
        environment.close()


def test_short_learn_and_save_load_preserve_policy_contract(tmp_path) -> None:
    base = gym.make("Pendulum-v1")
    environment = _normalized(base)
    try:
        model = _model(environment)
        observation, _ = environment.reset(seed=17)
        audit = RolloutActionLikelihoodAudit()
        model.learn(total_timesteps=16, callback=audit)
        assert audit.audited_rollouts == 2
        assert audit.max_abs_stored_action < 1.0 - audit.action_boundary_margin
        assert audit.max_log_probability_abs_error <= audit.log_probability_atol
        expected, _ = model.predict(observation, deterministic=True)
        checkpoint = tmp_path / "policy.zip"
        model.save(checkpoint)
        loaded = PPO.load(checkpoint, env=environment, device="cpu")
        observed, _ = loaded.predict(observation, deterministic=True)

        assert isinstance(loaded.policy, SquashedGaussianActorCriticPolicy)
        assert isinstance(loaded.policy.action_dist, SquashedDiagGaussianDistribution)
        assert loaded.policy.squash_output is True
        assert np.isfinite(observed).all()
        assert np.all(observed >= -1.0)
        assert np.all(observed <= 1.0)
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
    finally:
        environment.close()


def test_policy_source_is_content_addressed() -> None:
    assert len(squashed_policy_source_sha256()) == 64


def test_boundary_rounded_actions_recompute_but_abort_before_ppo_update() -> None:
    base = gym.make("Pendulum-v1")
    environment = _normalized(base)
    try:
        model = _model(environment)
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.fill_(10.0)
            model.policy.log_std.fill_(0.0)
        observation, _ = environment.reset(seed=23)
        observation_tensor, _ = model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            action, _value, old_log_probability = model.policy(observation_tensor)
            _value_again, recomputed, _entropy = model.policy.evaluate_actions(
                observation_tensor,
                action,
            )
        assert abs(float(action.item())) == 1.0
        torch.testing.assert_close(old_log_probability, recomputed)

        with pytest.raises(RuntimeError, match="tanh-boundary"):
            model.learn(total_timesteps=8, callback=RolloutActionLikelihoodAudit())
    finally:
        environment.close()
