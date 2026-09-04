"""Trusted Humanoid-v5 reward composition with stock operation parity."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from .contract import (
    ACTUATOR_QVEL_INDICES_BY_ACTION_V1,
    TASK_TERM_ABS_MAX,
    RewardContractError,
    RewardResultV1,
    TrustedRewardStepV1,
)

FORWARD_REWARD_WEIGHT = 1.25
HEALTHY_REWARD = 5.0
CTRL_COST_WEIGHT = 0.1
CONTACT_COST_WEIGHT = 5e-7
CONTACT_COST_MAX = 10.0

STOCK_TASK_TERM_SOURCE_V1 = b"def task_term(x):\n    return 1.25 * x.com_x_velocity_m_s\n"


def compositor_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def body_mass_weighted_com_xy_f64(
    body_mass_f64: np.ndarray,
    body_xipos_f64: np.ndarray,
) -> np.ndarray:
    """Reproduce Gymnasium's ``mass_center`` float64 operation order."""

    if not isinstance(body_mass_f64, np.ndarray) or body_mass_f64.dtype != np.dtype(np.float64):
        raise RewardContractError("body_mass_f64 must be a float64 numpy.ndarray")
    if not isinstance(body_xipos_f64, np.ndarray) or body_xipos_f64.dtype != np.dtype(np.float64):
        raise RewardContractError("body_xipos_f64 must be a float64 numpy.ndarray")
    if body_mass_f64.ndim != 1 or body_xipos_f64.shape != (body_mass_f64.size, 3):
        raise RewardContractError("body mass and xipos shapes differ")
    if body_mass_f64.size < 1 or not np.isfinite(body_mass_f64).all():
        raise RewardContractError("body masses must be finite and nonempty")
    if (
        np.any(body_mass_f64 < 0.0)
        or float(body_mass_f64.sum()) <= 0.0
        or not np.isfinite(body_xipos_f64).all()
    ):
        raise RewardContractError(
            "body masses must be non-negative with positive sum and xipos finite"
        )
    numerator = np.einsum("b,bj->j", body_mass_f64, body_xipos_f64)
    denominator = body_mass_f64.sum()
    result = (numerator / denominator)[0:2].copy()
    if (
        result.shape != (2,)
        or result.dtype != np.dtype(np.float64)
        or not np.isfinite(result).all()
    ):
        raise RewardContractError("derived center of mass is not a finite float64 pair")
    return result


def body_mass_weighted_com_x_velocity_m_s(
    *,
    body_mass_f64: np.ndarray,
    body_xipos_before_f64: np.ndarray,
    body_xipos_after_f64: np.ndarray,
    control_period_s: float,
) -> float:
    if type(control_period_s) is not float or control_period_s != 0.015:
        raise RewardContractError("control_period_s must be the exact float 0.015")
    before = body_mass_weighted_com_xy_f64(body_mass_f64, body_xipos_before_f64)
    after = body_mass_weighted_com_xy_f64(body_mass_f64, body_xipos_after_f64)
    velocity = (after - before) / control_period_s
    result = float(velocity[0])
    if not math.isfinite(result):
        raise RewardContractError("derived COM x velocity is non-finite")
    return result


def trusted_step_from_post_step_humanoid(
    environment: object,
    *,
    com_xy_before_f64: np.ndarray,
    target_speed_m_s: float,
) -> TrustedRewardStepV1:
    """Capture the direct post-step facts needed by the trusted compositor."""

    physical = getattr(environment, "unwrapped", environment)
    model = getattr(physical, "model", None)
    data = getattr(physical, "data", None)
    if model is None or data is None:
        raise RewardContractError("environment does not expose direct MuJoCo model/data")
    if (
        not isinstance(com_xy_before_f64, np.ndarray)
        or com_xy_before_f64.dtype != np.dtype(np.float64)
        or com_xy_before_f64.shape != (2,)
        or not np.isfinite(com_xy_before_f64).all()
    ):
        raise RewardContractError("com_xy_before_f64 must be a finite float64 pair")
    after = body_mass_weighted_com_xy_f64(
        np.asarray(model.body_mass),
        np.asarray(data.xipos),
    )
    control_period = float(getattr(physical, "dt", math.nan))
    if control_period != 0.015:
        raise RewardContractError("Humanoid control cadence differs from 0.015 s")
    velocity = (after - com_xy_before_f64) / control_period
    return TrustedRewardStepV1(
        qpos_after_f64=np.asarray(data.qpos),
        qvel_after_f64=np.asarray(data.qvel),
        com_x_velocity_m_s=float(velocity[0]),
        ctrl_f64=np.asarray(data.ctrl),
        generalized_actuator_torque_n_m_f64=np.asarray(data.qfrc_actuator)[
            list(ACTUATOR_QVEL_INDICES_BY_ACTION_V1)
        ],
        external_contact_wrench_f64=np.asarray(data.cfrc_ext),
        target_speed_m_s=target_speed_m_s,
        control_period_s=control_period,
    )


def _trusted_base_terms(step: TrustedRewardStepV1) -> tuple[float, float, float]:
    if not isinstance(step, TrustedRewardStepV1):
        raise RewardContractError("step must be a TrustedRewardStepV1")
    healthy = float(HEALTHY_REWARD * (1.0 < step.qpos_after_f64[2] < 2.0))
    control_cost = CTRL_COST_WEIGHT * np.sum(np.square(step.ctrl_f64))
    raw_contact_cost = CONTACT_COST_WEIGHT * np.sum(np.square(step.external_contact_wrench_f64))
    contact_cost = np.clip(raw_contact_cost, -np.inf, CONTACT_COST_MAX)
    return healthy, float(control_cost), float(contact_cost)


def _finalize(
    *,
    step: TrustedRewardStepV1,
    task_progress: float,
) -> RewardResultV1:
    if type(task_progress) is not float or not math.isfinite(task_progress):
        raise RewardContractError("task term must return one finite Python float")
    if abs(task_progress) > TASK_TERM_ABS_MAX:
        raise RewardContractError("task term exceeds the fail-closed envelope")
    healthy, control_cost, contact_cost = _trusted_base_terms(step)

    # This is intentionally the same grouping and subtraction order as
    # HumanoidEnv._get_rew in Gymnasium 1.3.0.
    rewards = task_progress + healthy
    costs = control_cost + contact_cost
    total = rewards - costs
    return RewardResultV1(
        total=float(total),
        signed_terms={
            "task_progress": task_progress,
            "healthy": healthy,
            "control": -control_cost,
            "contact": -contact_cost,
        },
    )


def compose_stock_humanoid_reward(step: TrustedRewardStepV1) -> RewardResultV1:
    """Return exact stock ``r_0`` terms through the common reward envelope."""

    task_progress = float(FORWARD_REWARD_WEIGHT * step.com_x_velocity_m_s)
    return _finalize(step=step, task_progress=task_progress)


def compose_trusted_humanoid_reward(
    step: TrustedRewardStepV1,
    raw_task_term: float,
    *,
    affine_alpha: float,
    affine_beta: float,
) -> RewardResultV1:
    """Scale one sandbox-returned scalar, then add immutable stock base terms."""

    if type(affine_alpha) is not float or not math.isfinite(affine_alpha):
        raise RewardContractError("affine_alpha must be one finite Python float")
    if type(affine_beta) is not float or not math.isfinite(affine_beta):
        raise RewardContractError("affine_beta must be one finite Python float")
    if not 0.25 <= affine_alpha <= 4.0:
        raise RewardContractError("affine_alpha is outside [0.25, 4]")
    if abs(affine_beta) > 10.0:
        raise RewardContractError("affine_beta exceeds 10 in magnitude")
    if type(raw_task_term) is not float or not math.isfinite(raw_task_term):
        raise RewardContractError("task term must return one finite Python float")
    if abs(raw_task_term) > TASK_TERM_ABS_MAX:
        raise RewardContractError("raw task term exceeds the fail-closed envelope")
    scaled = affine_alpha * raw_task_term + affine_beta
    if type(scaled) is not float or not math.isfinite(scaled):
        raise RewardContractError("scaled task term is non-finite")
    return _finalize(step=step, task_progress=scaled)


__all__ = [
    "CONTACT_COST_MAX",
    "CONTACT_COST_WEIGHT",
    "CTRL_COST_WEIGHT",
    "FORWARD_REWARD_WEIGHT",
    "HEALTHY_REWARD",
    "STOCK_TASK_TERM_SOURCE_V1",
    "body_mass_weighted_com_x_velocity_m_s",
    "body_mass_weighted_com_xy_f64",
    "compose_stock_humanoid_reward",
    "compose_trusted_humanoid_reward",
    "compositor_source_sha256",
    "trusted_step_from_post_step_humanoid",
]
