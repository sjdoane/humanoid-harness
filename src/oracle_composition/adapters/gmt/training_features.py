"""SB3 feature path for the opt-in fixed GMT observation normalizer."""

from __future__ import annotations

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .contracts import NORMALIZER_EPSILON, OBSERVATION_DIM
from .training_normalizer import FixedNormalizerState


class FixedActorNormalizerExtractor(BaseFeaturesExtractor):
    """Apply the fixed actor transform inside every SB3 train and predict call."""

    def __init__(
        self,
        observation_space: gym.Space,
        *,
        normalizer_mean: object,
        normalizer_std: object,
        normalizer_sha256: str,
    ) -> None:
        if (
            not isinstance(observation_space, gym.spaces.Box)
            or len(observation_space.shape) != 1
            or observation_space.shape[0] <= OBSERVATION_DIM
        ):
            raise ValueError("fixed normalizer requires a flat observation with a task tail")
        state = FixedNormalizerState.from_arrays(
            normalizer_mean,
            normalizer_std,
            expected_sha256=normalizer_sha256,
        )
        state.require_pinned()
        super().__init__(observation_space, features_dim=observation_space.shape[0])
        self.register_buffer("normalizer_mean", torch.from_numpy(state.mean.copy()))
        self.register_buffer(
            "normalizer_std", torch.from_numpy(state.standard_deviation.copy())
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        normalized = (observations[:, :OBSERVATION_DIM] - self.normalizer_mean) / (
            self.normalizer_std + NORMALIZER_EPSILON
        )
        return torch.cat((normalized, observations[:, OBSERVATION_DIM:]), dim=1)


def extractor_kwargs(state: FixedNormalizerState) -> dict[str, object]:
    state.require_pinned()
    return {
        "features_extractor_class": FixedActorNormalizerExtractor,
        "features_extractor_kwargs": {
            "normalizer_mean": state.mean.copy(),
            "normalizer_std": state.standard_deviation.copy(),
            "normalizer_sha256": state.sha256,
        },
    }


__all__ = ["FixedActorNormalizerExtractor", "extractor_kwargs"]
