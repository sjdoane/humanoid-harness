from __future__ import annotations

import ast
import dataclasses
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.experiments import reward_target_speed_evaluator as evaluator_module
from oracle_composition.experiments.reward_target_speed_evaluator import (
    ACTUATOR_QVEL_INDICES_BY_ACTION,
    CONTROL_PERIOD_SECONDS,
    EVALUATION_SEEDS,
    HORIZON_STEPS,
    PHYSICS_SUBSTEPS,
    PHYSICS_TIMESTEP_SECONDS,
    TORQUE_CAPACITY_N_M,
    TRAINING_SEEDS,
    ProtectedTargetSpeedStepV1,
    TargetSpeedEpisodeAccumulatorV1,
    TargetSpeedEvaluationError,
    TargetSpeedSubstepContactV1,
    capture_body_xipos_f64,
    capture_protected_target_speed_step,
    compare_exploratory_guardrails,
    evaluate_target_speed_episode,
    target_speed_progress,
)


def _step(
    index: int,
    *,
    velocity: float = 1.0,
    before_x: float | None = None,
    action: float = 0.0,
    joint_velocity: float = 0.0,
    height: float = 1.5,
    contacts: tuple[TargetSpeedSubstepContactV1, ...] = (),
    terminated: bool = False,
    truncated: bool = False,
) -> ProtectedTargetSpeedStepV1:
    before = (index - 1) * velocity * CONTROL_PERIOD_SECONDS if before_x is None else before_x
    masses = np.zeros(14, dtype=np.float64)
    masses[1] = 1.0
    xipos_before = np.zeros((14, 3), dtype=np.float64)
    xipos_after = np.zeros((14, 3), dtype=np.float64)
    xipos_before[1, 0] = before
    xipos_after[1, 0] = before + velocity * CONTROL_PERIOD_SECONDS
    qpos = np.zeros(24, dtype=np.float64)
    qpos[2] = height
    qpos[3] = 1.0
    qvel = np.zeros(23, dtype=np.float64)
    qvel[6:] = joint_velocity
    normalized = np.full(17, action, dtype=np.float64)
    torque = normalized * np.asarray(TORQUE_CAPACITY_N_M, dtype=np.float64)
    return ProtectedTargetSpeedStepV1(
        transition_index=index,
        body_mass_f64=masses,
        body_xipos_before_f64=xipos_before,
        body_xipos_after_f64=xipos_after,
        qpos_after_f64=qpos,
        qvel_after_f64=qvel,
        normalized_action_f64=normalized,
        generalized_actuator_torque_n_m_f64=torque,
        contacts=contacts,
        captured_physics_substeps=PHYSICS_SUBSTEPS,
        control_period_s=CONTROL_PERIOD_SECONDS,
        physics_timestep_s=PHYSICS_TIMESTEP_SECONDS,
        terminated=terminated,
        truncated=truncated,
    )


def _episode(velocity: float, *, count: int = HORIZON_STEPS):
    steps = []
    before = 0.0
    for index in range(1, count + 1):
        step = _step(
            index,
            velocity=velocity,
            before_x=before,
            terminated=index == count and count < HORIZON_STEPS,
            truncated=index == HORIZON_STEPS,
        )
        steps.append(step)
        before = float(step.body_xipos_after_f64[1, 0])
    return evaluate_target_speed_episode(
        steps,
        evaluation_seed=11001,
        target_speed_m_s=1.0,
    )


def test_endpoint_is_one_on_target_and_exp_half_at_symmetric_quarter_error() -> None:
    on_target = _episode(1.0)
    below = _episode(0.75)
    above = _episode(1.25)
    assert on_target.endpoint == 1.0
    assert below.endpoint == pytest.approx(math.exp(-0.5), abs=1e-12)
    assert above.endpoint == pytest.approx(below.endpoint, abs=1e-12)
    assert target_speed_progress(0.75, 1.0) == target_speed_progress(1.25, 1.0)


def test_endpoint_zero_pads_every_missing_post_termination_step() -> None:
    early = _episode(1.0, count=500)
    assert early.endpoint == 300 / 800
    assert early.observed_steps == 500
    assert early.survived_full_horizon is False
    assert early.first_failure_step == 500


def test_guardrail_units_cadence_capacities_and_energy_action_rate() -> None:
    result = evaluate_target_speed_episode(
        [_step(1, action=0.5, joint_velocity=2.0, terminated=True)],
        evaluation_seed=11001,
        target_speed_m_s=1.0,
    )
    capacities = np.asarray(TORQUE_CAPACITY_N_M, dtype=np.float64)
    expected_energy = CONTROL_PERIOD_SECONDS * float(np.sum(0.5 * capacities * 2.0))
    assert result.energy_j == expected_energy
    assert result.mean_absolute_power_w == expected_energy / CONTROL_PERIOD_SECONDS
    assert result.normalized_action_rate_rms_per_s == pytest.approx(0.5 / 0.015)
    assert result.normalized_action_rate_max_abs_per_s == pytest.approx(0.5 / 0.015)
    assert result.torque_capacity_normalized_max_abs == 0.5


def test_all_five_substeps_contact_identity_and_allowed_impact_descriptives() -> None:
    contacts = tuple(
        TargetSpeedSubstepContactV1(
            physics_substep_index=substep,
            contact_index_within_substep=0,
            geom1_name="floor",
            geom2_name="left_foot" if substep < 4 else "torso",
            normal_force_n=float(substep + 1),
        )
        for substep in range(5)
    )
    result = evaluate_target_speed_episode(
        [_step(1, contacts=contacts, terminated=True)],
        evaluation_seed=11001,
        target_speed_m_s=1.0,
    )
    assert result.allowed_foot_floor_contact_count == 4
    assert result.allowed_foot_floor_peak_force_n == 4.0
    assert result.allowed_foot_floor_peak_impulse_n_s == 4.0 * 0.003
    assert result.allowed_foot_floor_total_impulse_n_s == pytest.approx((1 + 2 + 3 + 4) * 0.003)
    assert result.non_foot_floor_contact_count == 1
    assert result.non_foot_floor_contact_identities == ("torso",)
    assert result.non_foot_floor_conforms is False


def test_structural_and_continuity_failures_invalidate_the_episode() -> None:
    accumulator = TargetSpeedEpisodeAccumulatorV1(evaluation_seed=11001, target_speed_m_s=1.0)
    accumulator.add(_step(1))
    with pytest.raises(TargetSpeedEvaluationError, match="discontinuous"):
        accumulator.add(_step(2, before_x=999.0))
    with pytest.raises(TargetSpeedEvaluationError, match="invalidated"):
        accumulator.finish()

    base = _step(1, terminated=True)
    values = {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}
    values["captured_physics_substeps"] = 4
    with pytest.raises(TargetSpeedEvaluationError, match="five"):
        ProtectedTargetSpeedStepV1(**values)
    values = {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}
    values["control_period_s"] = 0.02
    with pytest.raises(TargetSpeedEvaluationError, match=r"0\.015"):
        ProtectedTargetSpeedStepV1(**values)
    values = {field.name: getattr(base, field.name) for field in dataclasses.fields(base)}
    values["generalized_actuator_torque_n_m_f64"] = np.ones(17, dtype=np.float64)
    with pytest.raises(TargetSpeedEvaluationError, match="differs"):
        ProtectedTargetSpeedStepV1(**values)


def test_height_boundaries_are_inclusive_for_protected_collapse() -> None:
    assert _step(1, height=1.0, terminated=True).collapsed is False
    assert _step(1, height=2.0, terminated=True).collapsed is False
    assert _step(1, height=np.nextafter(1.0, -np.inf), terminated=True).collapsed is True
    assert _step(1, height=np.nextafter(2.0, np.inf), terminated=True).collapsed is True


@pytest.mark.gym
def test_real_capture_uses_actuator_order_not_contiguous_dof_order() -> None:
    from oracle_composition.envs.humanoid import HumanoidExperimentConfig, make_humanoid_env

    environment = make_humanoid_env(
        HumanoidExperimentConfig(terminate_when_unhealthy=True),
        capture_substep_contacts=True,
    )
    normalized = np.linspace(-1.0, 1.0, 17, dtype=np.float64)
    try:
        environment.reset(seed=20260904)
        before = capture_body_xipos_f64(environment)
        _observation, _unused_scalar, terminated, truncated, _diagnostics = environment.step(
            np.asarray(0.4 * normalized, dtype=np.float32)
        )
        step = capture_protected_target_speed_step(
            environment,
            transition_index=1,
            body_xipos_before_f64=before,
            normalized_action_f64=normalized,
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
    finally:
        environment.close()
    capacities = np.asarray(TORQUE_CAPACITY_N_M, dtype=np.float64)
    assert np.allclose(
        step.generalized_actuator_torque_n_m_f64 / capacities,
        normalized,
        rtol=0.0,
        atol=1e-7,
    )
    assert ACTUATOR_QVEL_INDICES_BY_ACTION[:2] == (7, 6)


def _study_map(template):
    return {
        train_seed: tuple(
            dataclasses.replace(template, evaluation_seed=evaluation_seed)
            for evaluation_seed in EVALUATION_SEEDS
        )
        for train_seed in TRAINING_SEEDS
    }


def test_relative_noninferiority_uses_five_seed_medians_and_ten_percent_margin() -> None:
    template = _episode(1.0)
    baseline = _study_map(template)
    candidate = _study_map(
        dataclasses.replace(
            template,
            torque_rms_n_m=template.torque_rms_n_m,
            energy_j=template.energy_j,
            normalized_action_rate_rms_per_s=template.normalized_action_rate_rms_per_s,
        )
    )
    result = compare_exploratory_guardrails(baseline, candidate)
    assert result.passed is True
    assert result.margin == 0.10
    assert result.label == "exploratory_relative_non_inferiority"
    assert result.paired_endpoint_differences == (0.0,) * 5

    with pytest.raises(TargetSpeedEvaluationError, match="missing"):
        compare_exploratory_guardrails(
            {seed: episodes for seed, episodes in baseline.items() if seed != 505},
            candidate,
        )
    partial = dict(candidate)
    partial[101] = partial[101][:-1]
    with pytest.raises(TargetSpeedEvaluationError, match="partial"):
        compare_exploratory_guardrails(baseline, partial)

    failed_baseline = _study_map(_episode(1.0, count=500))
    with pytest.raises(TargetSpeedEvaluationError, match="baseline viability"):
        compare_exploratory_guardrails(failed_baseline, candidate)


def test_protected_endpoint_has_no_return_component_or_diagnostic_data_dependency() -> None:
    source_path = Path(inspect.getsourcefile(evaluator_module) or "")
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_fragments = ("reward", "task_term", "telemetry")
    for node in ast.walk(tree):
        values: list[str] = []
        if isinstance(node, ast.Name):
            values.append(node.id)
        elif isinstance(node, ast.arg):
            values.append(node.arg)
        elif isinstance(node, ast.Attribute):
            values.append(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            values.append(node.value)
        for value in values:
            lowered = value.lower()
            assert all(fragment not in lowered for fragment in forbidden_fragments)
    assert "info" not in {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
