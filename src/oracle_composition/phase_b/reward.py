"""Tracking-only Phase B reward composition with separate telemetry streams."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from oracle_composition.tracking.humanoid_reference import HumanoidTrackingState
from oracle_composition.tracking.reward import (
    TrackingRewardConfig,
    TrackingRewardResult,
    compute_tracking_reward,
)

from .contracts import (
    FROZEN_TRACKING_REWARD_CONFIG,
    PhaseBContractError,
    TrackingOnlyRewardSpec,
)


def require_frozen_tracking_reward_config(
    config: TrackingRewardConfig | None,
) -> TrackingRewardConfig:
    """Admit only the existing tracking reward's exact frozen configuration."""

    if config is None:
        return FROZEN_TRACKING_REWARD_CONFIG
    if type(config) is not TrackingRewardConfig or config != FROZEN_TRACKING_REWARD_CONFIG:
        raise PhaseBContractError("tracking reward configuration differs from the frozen scales")
    return config


@dataclass(frozen=True, slots=True)
class TrainingRewardStreams:
    """Named streams prevent stock, task, and tracking returns from collapsing."""

    r_track: float
    r_task: float
    r_train: float
    ignored_stock_reward: float
    tracking: TrackingRewardResult

    def to_dict(self) -> dict[str, object]:
        return {
            "ignored_stock_reward": self.ignored_stock_reward,
            "r_task": self.r_task,
            "r_track": self.r_track,
            "r_train": self.r_train,
            "tracking_errors": self.tracking.error_components(),
            "tracking_reward_components": self.tracking.reward_components(),
        }


def compose_tracking_only_reward(
    *,
    state: HumanoidTrackingState,
    hidden_reference_target: np.ndarray,
    ignored_stock_reward: float,
    specification: TrackingOnlyRewardSpec,
    config: TrackingRewardConfig | None = None,
) -> TrainingRewardStreams:
    """Return ``r_track + 0`` while keeping stock reward as telemetry only."""

    if type(specification) is not TrackingOnlyRewardSpec:
        raise PhaseBContractError("reward compositor requires the registered tracking-only spec")
    stock = float(ignored_stock_reward)
    if not math.isfinite(stock):
        raise PhaseBContractError("stock reward telemetry must be finite")
    target = np.asarray(hidden_reference_target)
    if (
        type(hidden_reference_target) is not np.ndarray
        or target.dtype.str != "<f8"
        or target.shape != (45,)
        or not target.flags.c_contiguous
        or not np.isfinite(target).all()
    ):
        raise PhaseBContractError("hidden tracking target must be finite float64[45]")
    tracking = compute_tracking_reward(
        state=state,
        reference_frame=target,
        config=require_frozen_tracking_reward_config(config),
    )
    r_track = float(tracking.total)
    r_task = 0.0
    r_train = r_track + r_task
    if not all(math.isfinite(value) for value in (r_track, r_task, r_train)):
        raise PhaseBContractError("training reward stream became non-finite")
    return TrainingRewardStreams(
        r_track=r_track,
        r_task=r_task,
        r_train=r_train,
        ignored_stock_reward=stock,
        tracking=tracking,
    )


__all__ = [
    "FROZEN_TRACKING_REWARD_CONFIG",
    "TrainingRewardStreams",
    "compose_tracking_only_reward",
    "require_frozen_tracking_reward_config",
]
