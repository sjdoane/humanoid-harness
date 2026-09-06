"""Training-only scalar reward wrapper for the fixed G1 preconditioning profile."""

from __future__ import annotations

import math
from numbers import Real

import gymnasium as gym
import numpy as np

from .training_contract import (
    TRAINING_REWARD_INFO_ID,
    TRAINING_REWARD_SCALE,
    CourseTrainerSpec,
)


def _factor(value: object) -> float:
    if type(value) is not float or not math.isfinite(value) or value not in {
        1.0,
        TRAINING_REWARD_SCALE,
    }:
        raise ValueError("training reward factor must be exactly 1 or the admitted 1/64")
    return value


class TrainingRewardScale(gym.Wrapper):
    """Scale only the scalar consumed by PPO while preserving raw reward evidence."""

    def __init__(self, env: gym.Env, factor: float) -> None:
        super().__init__(env)
        self.factor = _factor(factor)

    def step(self, action: object) -> tuple[object, object, bool, bool, dict]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        if type(info) is not dict or "training_reward" in info:
            raise ValueError("training reward wrapper requires unclaimed dictionary info")
        if isinstance(reward, (bool, np.bool_)) or not isinstance(reward, Real):
            raise ValueError("environment total reward must be finite numeric")
        raw = float(reward)
        if not math.isfinite(raw):
            raise ValueError("environment total reward must be finite numeric")
        reward_breakdown = info.get("reward")
        if (
            type(reward_breakdown) is not dict
            or not isinstance(reward_breakdown.get("total_reward"), Real)
            or isinstance(reward_breakdown["total_reward"], (bool, np.bool_))
            or not math.isfinite(reward_breakdown["total_reward"])
            or float(reward_breakdown["total_reward"]) != raw
        ):
            raise ValueError("raw reward info differs from the environment scalar")
        scaled = reward if self.factor == 1.0 else np.float32(raw * self.factor)
        if not math.isfinite(float(scaled)):
            raise ValueError("scaled training reward must be finite")
        wrapped_info = dict(info)
        wrapped_info["training_reward"] = {
            "schema_id": TRAINING_REWARD_INFO_ID,
            "raw_total_reward": raw,
            "scaled_optimization_reward": float(scaled),
            "total_training_reward_scale": self.factor,
        }
        return observation, scaled, terminated, truncated, wrapped_info


def training_env(env: gym.Env, trainer: CourseTrainerSpec | None) -> gym.Env:
    """Return the identical legacy object unless preconditioning is explicit."""

    if trainer is None:
        return env
    return TrainingRewardScale(env, trainer.total_training_reward_scale)


__all__ = ["TRAINING_REWARD_INFO_ID", "TrainingRewardScale", "training_env"]
