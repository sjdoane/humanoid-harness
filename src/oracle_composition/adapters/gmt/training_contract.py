"""Pure fixed contracts for the G1 PPO trainer and reward preconditioning."""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes

TRAINING_REWARD_SCALE = 1.0 / 64.0
TRAINER_SCHEMA_ID = "gmt_g1_total_training_reward_preconditioning/v1"
EFFECTIVE_TRAINING_CONTRACT_ID = "gmt_g1_ppo_training_contract/v2"
TRAINING_REWARD_METADATA_ID = "gmt_g1_training_reward_units/v1"
TRAINING_REWARD_INFO_ID = "gmt_g1_training_reward_observation/v1"

# This exact object is retained for configs that omit ``trainer``.
TRAINING_CONTRACT = {
    "algorithm": "stable_baselines3.PPO",
    "device": "cpu",
    "n_steps": 512,
    "batch_size": 128,
    "n_epochs": 4,
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "target_kl": 0.02,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "net_arch": [128, 128],
    "log_std_init": -1.5,
    "observation_normalization": "none",
    "reward_normalization": "none",
    "initial_mean_action": "zero_output_layer",
    "checkpoint_selection": "final_fixed_budget_only",
}


@dataclass(frozen=True, slots=True)
class CourseTrainerSpec:
    """The sole admitted opt-in trainer preconditioning profile."""

    total_training_reward_scale: float

    def __post_init__(self) -> None:
        value = self.total_training_reward_scale
        if type(value) is not float or not math.isfinite(value) or value != TRAINING_REWARD_SCALE:
            raise ValueError("trainer reward scale must be the fixed finite 1/64 profile")

    @classmethod
    def from_dict(cls, value: object) -> CourseTrainerSpec:
        expected = {"schema_id", "schema_version", "total_training_reward_scale"}
        if type(value) is not dict or set(value) != expected:
            raise ValueError("trainer fields differ from the fixed preconditioning profile")
        if (
            value["schema_id"] != TRAINER_SCHEMA_ID
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
        ):
            raise ValueError("trainer schema identity differs")
        return cls(value["total_training_reward_scale"])

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_id": TRAINER_SCHEMA_ID,
            "schema_version": 1,
            "total_training_reward_scale": self.total_training_reward_scale,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


def effective_training_contract(trainer: CourseTrainerSpec | None) -> dict[str, object]:
    """Return the exact legacy object or a content-addressed v2 effective contract."""

    if trainer is None:
        return copy.deepcopy(TRAINING_CONTRACT)
    payload: dict[str, object] = {
        "schema_id": EFFECTIVE_TRAINING_CONTRACT_ID,
        "schema_version": 2,
        "base_ppo_contract": copy.deepcopy(TRAINING_CONTRACT),
        "reward_preconditioning": trainer.to_dict(),
        "training_reward_units": "raw_environment_total_reward_times_static_factor",
        "evaluation_reward_units": "raw_environment_total_reward",
    }
    return {
        **payload,
        "identity_sha256": hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
    }


def training_reward_metadata(trainer: CourseTrainerSpec) -> dict[str, object]:
    """Describe scalar and telemetry units without treating scaling as task reward."""

    return {
        "schema_id": TRAINING_REWARD_METADATA_ID,
        "schema_version": 1,
        "trainer_sha256": trainer.sha256,
        "total_training_reward_scale": trainer.total_training_reward_scale,
        "raw_environment_reward_info_path": "reward.total_reward",
        "optimization_reward_units": "raw_environment_total_reward_times_static_factor",
        "telemetry_episode_return_units": (
            "sum_of_scaled_environment_rewards_before_timeout_bootstrap"
        ),
        "value_target_units": "scaled_optimization_reward",
        "scaled_to_raw_ppo_float32_unit_factor": 64.0,
        "raw_environment_returns_source": "unscaled_step_info_not_scaled_value_recovery",
        "cross_scale_loss_comparison_valid": False,
    }


__all__ = [
    "TRAINER_SCHEMA_ID",
    "TRAINING_CONTRACT",
    "TRAINING_REWARD_INFO_ID",
    "TRAINING_REWARD_SCALE",
    "CourseTrainerSpec",
    "effective_training_contract",
    "training_reward_metadata",
]
