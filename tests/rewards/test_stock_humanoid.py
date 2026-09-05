from __future__ import annotations

import math

import numpy as np
import pytest

from oracle_composition.rewards.contract import RewardContractError, TrustedRewardStepV1
from oracle_composition.rewards.stock_humanoid import (
    CONTACT_COST_WEIGHT,
    body_mass_weighted_com_xy_f64,
    compose_stock_humanoid_reward,
    compose_trusted_humanoid_reward,
    trusted_step_from_post_step_humanoid,
)


def _step(
    *,
    z: float = 1.5,
    velocity: float = 0.0,
    control: float = 0.0,
    contact: np.ndarray | None = None,
) -> TrustedRewardStepV1:
    qpos = np.zeros(24, dtype=np.float64)
    qpos[2] = z
    qpos[3] = 1.0
    ctrl = np.full(17, control, dtype=np.float64)
    capacities = np.asarray(
        [40, 40, 40, 40, 40, 120, 80, 40, 40, 120, 80, 10, 10, 10, 10, 10, 10],
        dtype=np.float64,
    )
    normalized = ctrl / 0.4
    return TrustedRewardStepV1(
        qpos_after_f64=qpos,
        qvel_after_f64=np.zeros(23, dtype=np.float64),
        com_x_velocity_m_s=velocity,
        ctrl_f64=ctrl,
        generalized_actuator_torque_n_m_f64=normalized * capacities,
        external_contact_wrench_f64=(
            np.zeros((14, 6), dtype=np.float64) if contact is None else contact
        ),
        target_speed_m_s=1.0,
    )


def _expected(step: TrustedRewardStepV1) -> tuple[float, dict[str, float]]:
    forward = 1.25 * step.com_x_velocity_m_s
    healthy = 5.0 * (1.0 < step.qpos_after_f64[2] < 2.0)
    control_cost = 0.1 * np.sum(np.square(step.ctrl_f64))
    contact_cost = np.clip(
        CONTACT_COST_WEIGHT * np.sum(np.square(step.external_contact_wrench_f64)),
        -np.inf,
        10.0,
    )
    total = (forward + healthy) - (control_cost + contact_cost)
    return float(total), {
        "task_progress": float(forward),
        "healthy": float(healthy),
        "control": float(-control_cost),
        "contact": float(-contact_cost),
    }


@pytest.mark.parametrize(
    "z",
    [
        np.nextafter(1.0, -np.inf),
        1.0,
        np.nextafter(1.0, np.inf),
        np.nextafter(2.0, -np.inf),
        2.0,
        np.nextafter(2.0, np.inf),
    ],
)
@pytest.mark.parametrize("control", [-0.4, 0.0, 0.4])
def test_stock_parity_on_health_boundaries_and_action_endpoints(z: float, control: float) -> None:
    step = _step(z=float(z), velocity=-1.23456789, control=control)
    result = compose_stock_humanoid_reward(step)
    expected_total, expected_terms = _expected(step)
    assert result.total == expected_total
    assert dict(result.signed_terms) == expected_terms


@pytest.mark.parametrize("multiple", [0.0, 0.5, 1.0, 2.0])
def test_stock_parity_on_contact_clamp_cases(multiple: float) -> None:
    magnitude = math.sqrt((10.0 * multiple) / (CONTACT_COST_WEIGHT * 14 * 6))
    contact = np.full((14, 6), magnitude, dtype=np.float64)
    step = _step(z=1.5, velocity=2.0, contact=contact)
    result = compose_stock_humanoid_reward(step)
    expected_total, expected_terms = _expected(step)
    assert result.total == expected_total
    assert dict(result.signed_terms) == expected_terms
    assert result.signed_terms["contact"] >= -10.0


def test_world_contact_row_is_included_and_stock_envelope_aborts() -> None:
    contact = np.zeros((14, 6), dtype=np.float64)
    contact[0, 0] = 1000.0
    step = _step(contact=contact)
    result = compose_stock_humanoid_reward(step)
    assert result.signed_terms["contact"] == -(CONTACT_COST_WEIGHT * 1000.0**2)
    with pytest.raises(RewardContractError, match="task term"):
        compose_stock_humanoid_reward(_step(velocity=1000.0))


def test_compositor_aborts_nonfinite_wrong_type_and_envelope_without_fallback() -> None:
    step = _step()
    for candidate_value in (
        1,
        float("nan"),
        float("inf"),
        1000.0000001,
        lambda _x: 0.0,
    ):
        with pytest.raises(RewardContractError):
            compose_trusted_humanoid_reward(
                step,
                candidate_value,  # type: ignore[arg-type]
                affine_alpha=1.0,
                affine_beta=0.0,
            )


@pytest.mark.gym
def test_stock_parity_on_fixed_real_humanoid_reset_and_action_traces() -> None:
    import gymnasium as gym

    actions = (
        np.zeros(17, dtype=np.float32),
        np.full(17, 0.4, dtype=np.float32),
        np.full(17, -0.4, dtype=np.float32),
        np.linspace(-0.4, 0.4, 17, dtype=np.float32),
    )
    for seed in (1701, 1702):
        environment = gym.make("Humanoid-v5")
        try:
            environment.reset(seed=seed)
            for action in actions:
                physical = environment.unwrapped
                before = body_mass_weighted_com_xy_f64(
                    np.asarray(physical.model.body_mass),
                    np.asarray(physical.data.xipos),
                )
                _observation, stock_total, terminated, truncated, info = environment.step(action)
                step = trusted_step_from_post_step_humanoid(
                    environment,
                    com_xy_before_f64=before,
                    target_speed_m_s=1.0,
                )
                result = compose_stock_humanoid_reward(step)
                assert result.total == stock_total
                assert result.signed_terms["healthy"] == info["reward_survive"]
                assert result.signed_terms["task_progress"] == info["reward_forward"]
                assert result.signed_terms["control"] == info["reward_ctrl"]
                assert result.signed_terms["contact"] == info["reward_contact"]
                if terminated or truncated:
                    break
        finally:
            environment.close()
