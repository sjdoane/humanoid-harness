from __future__ import annotations

import pytest

from oracle_composition.envs.humanoid import HumanoidExperimentConfig


def test_humanoid_config_freezes_recovery_capable_termination() -> None:
    config = HumanoidExperimentConfig()

    assert config.env_id == "Humanoid-v5"
    assert config.terminate_when_unhealthy is False
    assert config.frame_skip == 5


@pytest.mark.parametrize("reset_noise_scale", [-0.1, float("nan"), float("inf")])
def test_invalid_reset_noise_is_rejected(reset_noise_scale: float) -> None:
    with pytest.raises(ValueError, match="reset_noise_scale"):
        HumanoidExperimentConfig(reset_noise_scale=reset_noise_scale)


def test_alternate_environment_is_rejected() -> None:
    with pytest.raises(ValueError, match="env_id"):
        HumanoidExperimentConfig(env_id="HumanoidStandup-v5")
