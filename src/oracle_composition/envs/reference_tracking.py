"""Gymnasium wrapper for a fixed, oracle-independent reference tracker task."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np

from oracle_composition.contracts import OracleContractError, ReferenceArtifact
from oracle_composition.tracking import (
    TrackingRewardConfig,
    compute_tracking_reward,
    tracking_state,
    validate_humanoid_actuator_abi,
    validate_humanoid_reference,
)


class FixedReferenceTrackingWrapper(gym.Wrapper):
    """Expose one immutable reference to policy and value function alike.

    The single flat observation returned by this wrapper is the only policy
    observation.  Stable-Baselines3 therefore supplies the same base state and
    exact flattened ``H x D`` command to both actor and critic.  The stock Gym
    reward is recorded but never added to the fixed tracking reward; declared
    task reward is exactly zero.  Each post-step state is graded against the
    next reference frame that was already visible in the pre-step window.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        reference: ReferenceArtifact,
        horizon_steps: int,
        reward_config: TrackingRewardConfig | None = None,
    ) -> None:
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise OracleContractError("base observation space must be a Box")
        if len(env.observation_space.shape) != 1:
            raise OracleContractError("base observation must be a flat vector")
        if not np.issubdtype(env.observation_space.dtype, np.floating):
            raise OracleContractError("base observation dtype must be floating point")

        validate_humanoid_reference(reference)
        # window() owns the type/resource-bound validation. Two frames are
        # required so the next-state reward target was visible to the policy.
        reference.window(start_frame=0, horizon_steps=horizon_steps)
        if horizon_steps < 2:
            raise OracleContractError(
                "horizon_steps must be at least 2 for next-frame reward alignment"
            )
        self._abi = validate_humanoid_actuator_abi(env)
        self._reference = reference
        self._horizon_steps = horizon_steps
        self._reward_config = reward_config or TrackingRewardConfig()
        self._frame_index = 0

        reference_width = horizon_steps * reference.schema.width
        dtype = env.observation_space.dtype
        reference_low = np.full(reference_width, -np.inf, dtype=dtype)
        reference_high = np.full(reference_width, np.inf, dtype=dtype)
        self.observation_space = gym.spaces.Box(
            low=np.concatenate((env.observation_space.low, reference_low)),
            high=np.concatenate((env.observation_space.high, reference_high)),
            dtype=dtype,
        )

    @property
    def reference_content_sha256(self) -> str:
        return self._reference.identity.content_sha256

    @property
    def tracking_reward_sha256(self) -> str:
        return self._reward_config.sha256

    def _window(self) -> tuple[np.ndarray, tuple[int, ...]]:
        window = self._reference.window(
            start_frame=min(self._frame_index, len(self._reference.values) - 1),
            horizon_steps=self._horizon_steps,
        )
        values = np.asarray(window.values, dtype=self.observation_space.dtype)
        if values.shape != (self._horizon_steps, self._reference.schema.width):
            raise OracleContractError("reference window shape changed after validation")
        if not np.isfinite(values).all():
            raise OracleContractError("reference window contains non-finite values")
        return values, window.frame_indices

    def _observation(self, base_observation: object, window: np.ndarray) -> np.ndarray:
        base = np.asarray(base_observation, dtype=self.observation_space.dtype)
        expected_base_shape = self.env.observation_space.shape
        if base.shape != expected_base_shape:
            raise OracleContractError(
                f"base observation has shape {base.shape}; expected {expected_base_shape}"
            )
        if not np.isfinite(base).all():
            raise OracleContractError("base observation contains non-finite values")
        combined = np.concatenate((base, window.reshape(-1)))
        if combined.shape != self.observation_space.shape:
            raise OracleContractError("combined observation does not match the declared Box")
        if not self.observation_space.contains(combined):
            raise OracleContractError("combined observation violates the declared Box")
        return combined

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        validate_humanoid_reference(self._reference)
        if validate_humanoid_actuator_abi(self.env) != self._abi:
            raise OracleContractError("Humanoid actuator ABI changed after wrapper construction")
        self._frame_index = 0
        base_observation, info = self.env.reset(seed=seed, options=options)
        window, indices = self._window()
        resolved_info = dict(info)
        resolved_info["reference_tracking"] = self._telemetry_header(indices)
        return self._observation(base_observation, window), resolved_info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # State x[t] observes frames r[t:t+H]. Action a[t] advances the plant
        # to x[t+1], which is graded against r[t+1]—the second frame that was
        # visible when the action was chosen. The final frame is held.
        _, policy_indices = self._window()
        base_observation, ignored_reward, terminated, truncated, info = self.env.step(action)
        if not math.isfinite(float(ignored_reward)):
            raise OracleContractError("base environment produced a non-finite reward")

        self._frame_index = min(self._frame_index + 1, len(self._reference.values) - 1)
        next_window, next_indices = self._window()
        reward = compute_tracking_reward(
            state=tracking_state(self.env, self._abi),
            reference_frame=next_window[0],
            config=self._reward_config,
        )

        telemetry = self._telemetry_header(policy_indices)
        telemetry.update(
            {
                "reward_target_frame_index": next_indices[0],
                "reward_temporal_alignment": "post_step_state_against_next_reference_frame",
                "tracking_reward": reward.reward_components(),
                "tracking_error": reward.error_components(),
                "task_reward": 0.0,
                "ignored_environment_reward": float(ignored_reward),
            }
        )
        resolved_info = dict(info)
        resolved_info["reference_tracking"] = telemetry
        return (
            self._observation(base_observation, next_window),
            reward.total,
            bool(terminated),
            bool(truncated),
            resolved_info,
        )

    def _telemetry_header(self, indices: tuple[int, ...]) -> dict[str, object]:
        return {
            "reference_artifact_id": self._reference.identity.artifact_id,
            "reference_content_sha256": self._reference.identity.content_sha256,
            "reference_schema_sha256": self._reference.identity.schema_sha256,
            "tracking_reward_sha256": self._reward_config.sha256,
            "reference_window_shape": [
                self._horizon_steps,
                self._reference.schema.width,
            ],
            "policy_reference_window_frame_indices": list(indices),
        }
