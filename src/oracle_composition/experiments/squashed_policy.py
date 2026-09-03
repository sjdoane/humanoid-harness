"""Bounded Gaussian policy used by the frozen PPO tracking baseline.

Stable-Baselines3 PPO normally samples an unbounded Gaussian, sends a clipped
copy to a bounded environment, and stores the unclipped sample for the policy
update.  That is not an acceptable likelihood contract for this experiment.
This policy samples through ``tanh`` and evaluates the same bounded sample with
the distribution's Jacobian correction.

The surrounding environment must expose a normalized ``[-1, 1]`` action Box.
An outer action-rescaling wrapper is responsible for the one affine conversion
to the physical actuator bounds.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.distributions import (
    SquashedDiagGaussianDistribution,
    get_action_dim,
)
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import obs_as_tensor

POLICY_ID = "local.SquashedGaussianActorCriticPolicy/tanh_jacobian/v1"
ACTION_TRANSFORM_ID = "tanh_normalized_then_affine_physical_box/v1"
ACTION_LIKELIHOOD_AUDIT_ID = "ppo_rollout_preupdate_action_likelihood/v1"
ACTION_BOUNDARY_MARGIN = 1e-6
LOG_PROB_RECOMPUTE_ATOL = 1e-5


class SquashedGaussianActorCriticPolicy(ActorCriticPolicy):
    """Actor-critic policy whose stored and executed normalized actions agree."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if bool(kwargs.get("use_sde", False)):
            raise ValueError("SquashedGaussianActorCriticPolicy does not permit gSDE")
        if bool(kwargs.pop("squash_output", False)):
            raise ValueError("squash_output is owned by the pinned policy implementation")

        # Build the standard diagonal-Gaussian network first.  The squashed
        # distribution has the same mean/log-std parameterization and adds no
        # trainable parameters, so the already-built optimizer remains exact.
        super().__init__(*args, squash_output=False, **kwargs)
        if not isinstance(self.action_space, spaces.Box):
            raise ValueError("squashed policy requires a continuous Box action space")
        if not np.all(np.isfinite(self.action_space.low)) or not np.all(
            np.isfinite(self.action_space.high)
        ):
            raise ValueError("squashed policy requires finite action bounds")
        if not np.array_equal(
            self.action_space.low,
            -np.ones(self.action_space.shape, dtype=self.action_space.dtype),
        ) or not np.array_equal(
            self.action_space.high,
            np.ones(self.action_space.shape, dtype=self.action_space.dtype),
        ):
            raise ValueError("squashed policy requires an exact normalized [-1, 1] Box")

        self.action_dist = SquashedDiagGaussianDistribution(get_action_dim(self.action_space))
        # SB3 uses this flag to affine-map policy-domain [-1, 1] samples to the
        # environment Box.  The normalized environment Box makes that step an
        # identity; the outer Gymnasium wrapper performs the sole physical map.
        self._squash_output = True

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        """Preserve the already-normalized sample without an identity remap.

        SB3 calls this hook because ``squash_output`` is true.  Its generic
        affine formula would map ``[-1, 1]`` back onto the same normalized Box
        and introduce avoidable float32 roundoff before the outer physical
        action wrapper.  This policy accepts only that exact Box, so identity
        is the complete and correct operation.
        """

        return scaled_action


def assert_squashed_policy_contract(policy: object) -> None:
    """Reject a model whose loaded policy no longer has the frozen semantics."""

    if not isinstance(policy, SquashedGaussianActorCriticPolicy):
        raise RuntimeError("model did not install the pinned squashed Gaussian policy")
    if not isinstance(policy.action_dist, SquashedDiagGaussianDistribution):
        raise RuntimeError("model action distribution is not the pinned squashed Gaussian")
    if policy.squash_output is not True:
        raise RuntimeError("model no longer declares bounded policy-domain actions")
    action_space = policy.action_space
    if not isinstance(action_space, spaces.Box):
        raise RuntimeError("model action space is not a Box")
    expected_low = -np.ones(action_space.shape, dtype=action_space.dtype)
    expected_high = np.ones(action_space.shape, dtype=action_space.dtype)
    if not np.array_equal(action_space.low, expected_low) or not np.array_equal(
        action_space.high,
        expected_high,
    ):
        raise RuntimeError("model action space is not the exact normalized [-1, 1] Box")


class RolloutActionLikelihoodAudit(BaseCallback):
    """Fail before each PPO update if stored action likelihoods diverge."""

    def __init__(
        self,
        *,
        action_boundary_margin: float = ACTION_BOUNDARY_MARGIN,
        log_probability_atol: float = LOG_PROB_RECOMPUTE_ATOL,
    ) -> None:
        super().__init__(verbose=0)
        if not 0.0 < action_boundary_margin < 1.0:
            raise ValueError("action_boundary_margin must be in (0, 1)")
        if not np.isfinite(log_probability_atol) or log_probability_atol <= 0.0:
            raise ValueError("log_probability_atol must be finite and positive")
        self.action_boundary_margin = float(action_boundary_margin)
        self.log_probability_atol = float(log_probability_atol)
        self.audited_rollouts = 0
        self.max_abs_stored_action = 0.0
        self.max_log_probability_abs_error = 0.0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        assert_squashed_policy_contract(self.model.policy)
        buffer = self.model.rollout_buffer
        if not buffer.full:
            raise RuntimeError("action-likelihood audit received an incomplete rollout")
        actions = np.asarray(buffer.actions)
        if actions.shape[-1] != get_action_dim(self.model.action_space):
            raise RuntimeError("rollout action width changed")
        if not np.isfinite(actions).all():
            raise RuntimeError("rollout contains non-finite stored actions")
        maximum = float(np.max(np.abs(actions)))
        self.max_abs_stored_action = max(self.max_abs_stored_action, maximum)
        if maximum >= 1.0 - self.action_boundary_margin:
            raise RuntimeError("rollout action reached the frozen tanh-boundary rejection margin")

        observations = np.asarray(buffer.observations).reshape(
            (-1, *self.model.observation_space.shape)
        )
        flat_actions = actions.reshape((-1, *self.model.action_space.shape))
        old_log_probabilities = np.asarray(buffer.log_probs).reshape(-1)
        previous_training_mode = self.model.policy.training
        self.model.policy.set_training_mode(False)
        try:
            with th.no_grad():
                observation_tensor = obs_as_tensor(observations, self.model.device)
                action_tensor = th.as_tensor(flat_actions, device=self.model.device)
                _values, recomputed_log_probabilities, _entropy = (
                    self.model.policy.evaluate_actions(observation_tensor, action_tensor)
                )
        finally:
            self.model.policy.set_training_mode(previous_training_mode)
        recomputed = recomputed_log_probabilities.detach().cpu().numpy().reshape(-1)
        if not np.isfinite(old_log_probabilities).all() or not np.isfinite(recomputed).all():
            raise RuntimeError("rollout contains non-finite action log probabilities")
        error = float(np.max(np.abs(recomputed - old_log_probabilities)))
        self.max_log_probability_abs_error = max(
            self.max_log_probability_abs_error,
            error,
        )
        if error > self.log_probability_atol:
            raise RuntimeError(
                "stored action log probability does not match pre-update recomputation"
            )
        self.audited_rollouts += 1

    def completion_receipt(self, *, expected_rollouts: int) -> dict[str, object]:
        """Return finite audit evidence only after every expected rollout passed."""

        if (
            not isinstance(expected_rollouts, int)
            or isinstance(expected_rollouts, bool)
            or expected_rollouts < 1
        ):
            raise ValueError("expected_rollouts must be a positive integer")
        if self.audited_rollouts != expected_rollouts:
            raise RuntimeError(
                f"audited {self.audited_rollouts} rollouts; expected {expected_rollouts}"
            )
        if not np.isfinite(self.max_abs_stored_action) or not np.isfinite(
            self.max_log_probability_abs_error
        ):
            raise RuntimeError("action-likelihood audit summary is non-finite")
        return {
            "audit_id": ACTION_LIKELIHOOD_AUDIT_ID,
            "passed": True,
            "expected_rollouts": expected_rollouts,
            "audited_rollouts": self.audited_rollouts,
            "action_boundary_margin": self.action_boundary_margin,
            "max_abs_stored_action": self.max_abs_stored_action,
            "log_probability_atol": self.log_probability_atol,
            "max_log_probability_abs_error": self.max_log_probability_abs_error,
        }


def squashed_policy_source_sha256() -> str:
    """Content address the exact local policy implementation."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


__all__ = [
    "ACTION_BOUNDARY_MARGIN",
    "ACTION_LIKELIHOOD_AUDIT_ID",
    "ACTION_TRANSFORM_ID",
    "LOG_PROB_RECOMPUTE_ATOL",
    "POLICY_ID",
    "RolloutActionLikelihoodAudit",
    "SquashedGaussianActorCriticPolicy",
    "assert_squashed_policy_contract",
    "squashed_policy_source_sha256",
]
