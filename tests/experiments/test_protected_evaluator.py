from __future__ import annotations

import inspect
import math
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from oracle_composition.experiments import (
    EpisodeAccumulator,
    ExperimentContractError,
    evaluate_tracking_step,
    protected_evaluator_sha256,
)
from oracle_composition.experiments.protected_evaluator import (
    STATION_KEEPING_ORIGIN_SOURCE,
    CollapseDefinition,
    ContactFact,
    StepMetrics,
)

TORQUE_CAPACITY_N_M = np.asarray(
    [40, 40, 40, 40, 40, 120, 80, 40, 40, 120, 80, 10, 10, 10, 10, 10, 10],
    dtype=np.float64,
)


def _stand_reference() -> np.ndarray:
    reference = np.zeros(45, dtype=np.float64)
    reference[0] = 1.4
    reference[1] = 1.0
    return reference


def _evaluate(
    *,
    root_height_m: float = 1.4,
    root_position_world_m: object | None = None,
    root_linear_velocity_world_m_s: object | None = None,
    orientation: np.ndarray | None = None,
    normalized_policy_action: object | None = None,
    previous_normalized_policy_action: object | None = None,
    generalized_actuator_torque_n_m: object | None = None,
    generalized_actuator_torque_capacity_n_m: object | None = None,
    joint_accelerations_rad_s2: object | None = None,
    previous_joint_accelerations_rad_s2: object | None = None,
    contact_facts: object = (),
    control_period_seconds: object = 0.015,
    collapse_definition: CollapseDefinition | None = None,
) -> StepMetrics:
    return evaluate_tracking_step(
        root_position_world_m=(
            np.array([0.0, 0.0, root_height_m])
            if root_position_world_m is None
            else root_position_world_m
        ),
        root_height_m=root_height_m,
        root_orientation_wxyz=(
            np.array([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation
        ),
        root_linear_velocity_world_m_s=(
            np.zeros(3)
            if root_linear_velocity_world_m_s is None
            else root_linear_velocity_world_m_s
        ),
        root_angular_velocity_body_rad_s=np.zeros(3),
        joint_positions_rad=np.zeros(17),
        joint_velocities_rad_s=np.zeros(17),
        reference_frame=_stand_reference(),
        normalized_policy_action=(
            np.zeros(17) if normalized_policy_action is None else normalized_policy_action
        ),
        previous_normalized_policy_action=(
            np.zeros(17)
            if previous_normalized_policy_action is None
            else previous_normalized_policy_action
        ),
        generalized_actuator_torque_n_m=(
            np.zeros(17)
            if generalized_actuator_torque_n_m is None
            else generalized_actuator_torque_n_m
        ),
        generalized_actuator_torque_capacity_n_m=(
            TORQUE_CAPACITY_N_M
            if generalized_actuator_torque_capacity_n_m is None
            else generalized_actuator_torque_capacity_n_m
        ),
        joint_accelerations_rad_s2=(
            np.zeros(17) if joint_accelerations_rad_s2 is None else joint_accelerations_rad_s2
        ),
        previous_joint_accelerations_rad_s2=(
            np.zeros(17)
            if previous_joint_accelerations_rad_s2 is None
            else previous_joint_accelerations_rad_s2
        ),
        contact_facts=contact_facts,  # type: ignore[arg-type]
        control_period_seconds=control_period_seconds,  # type: ignore[arg-type]
        **({} if collapse_definition is None else {"collapse_definition": collapse_definition}),
    )


def _accumulator(*, evaluation_seed: int, requested_steps: int) -> EpisodeAccumulator:
    return EpisodeAccumulator(
        evaluation_seed=evaluation_seed,
        requested_steps=requested_steps,
        initial_root_position_world_m=np.array([0.0, 0.0, 1.4]),
        station_keeping_origin_source=STATION_KEEPING_ORIGIN_SOURCE,
        initial_normalized_policy_action=np.zeros(17),
    )


def _contact(
    contact_index: int,
    *,
    floor: bool,
    forbidden: bool,
    force_n: float,
) -> ContactFact:
    return ContactFact(
        contact_index=contact_index,
        geom1_name="floor" if floor else "torso",
        geom2_name=("torso1" if forbidden else "left_foot") if floor else "right_upper_arm",
        involves_floor=floor,
        forbidden_floor_contact=forbidden,
        normal_force_n=force_n,
    )


def test_exact_upright_state_has_zero_errors_and_is_not_collapsed() -> None:
    metrics = _evaluate()

    assert metrics.root_height_abs_error_m == 0.0
    assert metrics.root_orientation_error_rad == 0.0
    assert metrics.root_linear_velocity_rmse_m_s == 0.0
    assert metrics.root_angular_velocity_rmse_rad_s == 0.0
    assert metrics.joint_position_rmse_rad == 0.0
    assert metrics.joint_velocity_rmse_rad_s == 0.0
    assert metrics.normalized_policy_action_rms == 0.0
    assert metrics.normalized_policy_action_delta_rate_rms_per_s == 0.0
    assert metrics.generalized_actuator_torque_rms_n_m == 0.0
    assert metrics.generalized_actuator_torque_normalized_rms == 0.0
    assert metrics.joint_jerk_rms_rad_s3 == 0.0
    assert metrics.simulator_contact_count == 0
    assert metrics.simulator_contact_peak_normal_force_n == 0.0
    assert metrics.floor_contact_peak_normal_force_n == 0.0
    assert metrics.forbidden_floor_contact_fraction == 0.0
    assert metrics.control_period_seconds == pytest.approx(0.015)
    assert metrics.torso_up_z == pytest.approx(1.0)
    assert metrics.collapsed is False


def test_low_or_tilted_torso_is_collapsed_independently_of_joint_match() -> None:
    low = _evaluate(root_height_m=0.4)
    half_turn = math.pi / 4.0
    tilted = _evaluate(orientation=np.array([math.cos(half_turn), math.sin(half_turn), 0.0, 0.0]))

    assert low.joint_position_rmse_rad == 0.0
    assert low.collapsed is True
    assert tilted.root_orientation_error_rad == pytest.approx(math.pi / 2.0)
    assert tilted.torso_up_z == pytest.approx(0.0, abs=1e-12)
    assert tilted.collapsed is True


def test_collapse_definition_is_source_frozen_not_caller_configurable() -> None:
    assert "collapse" not in inspect.signature(evaluate_tracking_step).parameters


def test_explicit_collapse_definition_controls_the_primary_outcome() -> None:
    definition = CollapseDefinition(
        min_root_height_m=1.0,
        max_root_height_m=2.0,
        min_torso_up_z=0.5,
    )

    assert _evaluate(root_height_m=1.0, collapse_definition=definition).collapsed is False
    with pytest.raises(ExperimentContractError, match="source-frozen"):
        _evaluate(
            root_height_m=1.1,
            collapse_definition=CollapseDefinition(1.2, 2.0, 0.5),
        )

    with pytest.raises(ExperimentContractError, match="finite"):
        CollapseDefinition(min_root_height_m=10**400)


def test_control_effort_contact_and_smoothness_metrics_have_explicit_units() -> None:
    metrics = _evaluate(
        normalized_policy_action=np.full(17, 0.5),
        previous_normalized_policy_action=np.full(17, -0.5),
        generalized_actuator_torque_n_m=0.5 * TORQUE_CAPACITY_N_M,
        joint_accelerations_rad_s2=np.full(17, 3.0),
        previous_joint_accelerations_rad_s2=np.full(17, 1.0),
        contact_facts=(
            _contact(0, floor=True, forbidden=False, force_n=50.0),
            _contact(1, floor=True, forbidden=True, force_n=20.0),
            _contact(2, floor=False, forbidden=False, force_n=30.0),
        ),
        control_period_seconds=0.02,
    )

    assert metrics.normalized_policy_action_rms == pytest.approx(0.5)
    assert metrics.normalized_policy_action_max_abs == pytest.approx(0.5)
    assert metrics.normalized_policy_action_delta_rate_rms_per_s == pytest.approx(50.0)
    assert metrics.normalized_policy_action_delta_rate_max_abs_per_s == pytest.approx(50.0)
    assert metrics.generalized_actuator_torque_rms_n_m == pytest.approx(
        0.5 * np.sqrt(np.mean(np.square(TORQUE_CAPACITY_N_M)))
    )
    assert metrics.generalized_actuator_torque_max_abs_n_m == pytest.approx(60.0)
    assert metrics.generalized_actuator_torque_normalized_rms == pytest.approx(0.5)
    assert metrics.generalized_actuator_torque_normalized_max_abs == pytest.approx(0.5)
    assert metrics.joint_jerk_rms_rad_s3 == pytest.approx(100.0)
    assert metrics.joint_jerk_max_abs_rad_s3 == pytest.approx(100.0)
    assert metrics.simulator_contact_count == 3
    assert metrics.simulator_contact_peak_normal_force_n == pytest.approx(50.0)
    assert metrics.floor_contact_count == 2
    assert metrics.floor_contact_peak_normal_force_n == pytest.approx(50.0)
    assert metrics.forbidden_floor_contact_count == 1
    assert metrics.forbidden_floor_contact_fraction == pytest.approx(0.5)
    assert metrics.forbidden_floor_contact_peak_force_n == pytest.approx(20.0)


def test_effort_uses_post_transmission_generalized_hinge_torque() -> None:
    # Humanoid's motor gear can map a 0.4 pre-transmission actuator_force to
    # 40 N m in qfrc_actuator. The evaluator accepts only the latter fact.
    metrics = _evaluate(
        normalized_policy_action=np.ones(17),
        generalized_actuator_torque_n_m=np.full(17, 40.0),
        generalized_actuator_torque_capacity_n_m=np.full(17, 40.0),
    )

    assert metrics.generalized_actuator_torque_rms_n_m == 40.0
    assert metrics.generalized_actuator_torque_max_abs_n_m == 40.0
    assert metrics.generalized_actuator_torque_normalized_rms == 1.0
    assert metrics.generalized_actuator_torque_normalized_max_abs == 1.0


def test_executed_action_and_post_transmission_torque_must_match() -> None:
    with pytest.raises(ExperimentContractError, match="must match the executed"):
        _evaluate(
            normalized_policy_action=np.full(17, 0.5),
            generalized_actuator_torque_n_m=np.full(17, 0.5000002) * TORQUE_CAPACITY_N_M,
        )

    # Float32-to-float64 conversion noise within the frozen tolerance remains
    # admissible; this is not a behavioral or dynamics tolerance.
    _evaluate(
        normalized_policy_action=np.full(17, 0.5),
        generalized_actuator_torque_n_m=np.full(17, 0.50000005) * TORQUE_CAPACITY_N_M,
    )

    with pytest.raises(ExperimentContractError, match="must match the executed"):
        _evaluate(
            normalized_policy_action=np.full(17, 0.5),
            generalized_actuator_torque_n_m=-0.5 * TORQUE_CAPACITY_N_M,
        )


def test_episode_accumulator_requires_complete_horizon_and_reports_collapse() -> None:
    accumulator = _accumulator(evaluation_seed=11001, requested_steps=2)
    accumulator.add(_evaluate(), tracking_reward=1.0)
    with pytest.raises(ExperimentContractError, match="expected 2"):
        accumulator.finish()

    accumulator.add(_evaluate(root_height_m=0.4), tracking_reward=0.0)
    result = accumulator.finish()
    assert result.observed_steps == 2
    assert result.survived_full_horizon is False
    assert result.first_collapse_step == 2
    assert result.collapse_fraction == pytest.approx(0.5)
    assert result.tracking_return == pytest.approx(1.0)


def test_episode_accumulator_aggregates_control_and_contact_samples() -> None:
    accumulator = _accumulator(evaluation_seed=7, requested_steps=2)
    accumulator.add(_evaluate(control_period_seconds=0.5), tracking_reward=0.0)
    accumulator.add(
        _evaluate(
            normalized_policy_action=np.ones(17),
            previous_normalized_policy_action=np.zeros(17),
            generalized_actuator_torque_n_m=TORQUE_CAPACITY_N_M,
            joint_accelerations_rad_s2=np.ones(17),
            previous_joint_accelerations_rad_s2=np.zeros(17),
            contact_facts=(
                _contact(0, floor=True, forbidden=True, force_n=12.0),
                _contact(1, floor=True, forbidden=False, force_n=8.0),
            ),
            control_period_seconds=0.5,
        ),
        tracking_reward=0.0,
    )

    result = accumulator.finish()

    assert result.normalized_policy_action_rms == pytest.approx(math.sqrt(0.5))
    assert result.normalized_policy_action_max_abs == 1.0
    assert result.normalized_policy_action_delta_rate_rms_per_s == pytest.approx(math.sqrt(2.0))
    assert result.normalized_policy_action_delta_rate_max_abs_per_s == 2.0
    assert result.generalized_actuator_torque_rms_n_m == pytest.approx(
        np.sqrt(np.mean(np.square(TORQUE_CAPACITY_N_M)) / 2.0)
    )
    assert result.generalized_actuator_torque_max_abs_n_m == 120.0
    assert result.generalized_actuator_torque_normalized_rms == pytest.approx(math.sqrt(0.5))
    assert result.generalized_actuator_torque_normalized_max_abs == 1.0
    assert result.joint_jerk_rms_rad_s3 == pytest.approx(math.sqrt(2.0))
    assert result.joint_jerk_max_abs_rad_s3 == 2.0
    assert result.simulator_contact_count == 2
    assert result.simulator_contact_peak_normal_force_n == 12.0
    assert result.floor_contact_count == 2
    assert result.floor_contact_peak_normal_force_n == 12.0
    assert result.forbidden_floor_contact_count == 1
    assert result.forbidden_floor_contact_fraction == pytest.approx(0.5)
    assert result.forbidden_floor_contact_step_fraction == pytest.approx(0.5)
    assert result.forbidden_floor_contact_peak_force_n == 12.0


def test_episode_station_keeping_uses_seeded_reset_origin_and_direct_world_state() -> None:
    accumulator = EpisodeAccumulator(
        evaluation_seed=7,
        requested_steps=2,
        initial_root_position_world_m=np.array([1.0, -2.0, 1.4]),
        station_keeping_origin_source=STATION_KEEPING_ORIGIN_SOURCE,
        initial_normalized_policy_action=np.zeros(17),
    )
    accumulator.add(
        _evaluate(
            root_position_world_m=np.array([1.3, -1.6, 1.4]),
            root_linear_velocity_world_m_s=np.array([-0.2, 0.5, -0.1]),
        ),
        tracking_reward=0.5,
    )
    accumulator.add(
        _evaluate(
            root_position_world_m=np.array([1.6, -1.2, 1.4]),
            root_linear_velocity_world_m_s=np.array([-0.7, 0.1, 0.4]),
        ),
        tracking_reward=0.5,
    )

    result = accumulator.finish()

    assert result.station_keeping_origin_source == STATION_KEEPING_ORIGIN_SOURCE
    assert result.initial_root_position_world_m == (1.0, -2.0, 1.4)
    assert result.root_horizontal_displacement_max_m == pytest.approx(1.0)
    assert result.root_linear_velocity_x_max_abs_m_s == pytest.approx(0.7)
    assert result.root_linear_velocity_y_max_abs_m_s == pytest.approx(0.5)
    assert result.root_linear_velocity_z_max_abs_m_s == pytest.approx(0.4)
    assert result.control_period_seconds == pytest.approx(0.015)


def test_episode_origin_reset_action_and_cadence_provenance_fail_closed() -> None:
    with pytest.raises(ExperimentContractError, match="seeded-reset MuJoCo qpos"):
        EpisodeAccumulator(
            evaluation_seed=7,
            requested_steps=2,
            initial_root_position_world_m=np.array([0.0, 0.0, 1.4]),
            station_keeping_origin_source="reward_info/v0",
            initial_normalized_policy_action=np.zeros(17),
        )
    with pytest.raises(ExperimentContractError, match="exactly zero"):
        EpisodeAccumulator(
            evaluation_seed=7,
            requested_steps=2,
            initial_root_position_world_m=np.array([0.0, 0.0, 1.4]),
            station_keeping_origin_source=STATION_KEEPING_ORIGIN_SOURCE,
            initial_normalized_policy_action=np.full(17, 0.1),
        )

    accumulator = _accumulator(evaluation_seed=7, requested_steps=2)
    accumulator.add(_evaluate(control_period_seconds=0.015), tracking_reward=0.5)
    with pytest.raises(ExperimentContractError, match="changed between simulator steps"):
        accumulator.add(_evaluate(control_period_seconds=0.02), tracking_reward=0.5)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("normalized_policy_action", np.zeros(16)),
        ("normalized_policy_action", np.full(17, np.nan)),
        ("normalized_policy_action", np.full(17, 1.0001)),
        ("previous_normalized_policy_action", np.full(17, -1.0001)),
        ("generalized_actuator_torque_n_m", np.zeros(16)),
        ("generalized_actuator_torque_n_m", np.full(17, np.inf)),
        ("generalized_actuator_torque_capacity_n_m", np.zeros(16)),
        ("generalized_actuator_torque_capacity_n_m", np.zeros(17)),
        ("generalized_actuator_torque_capacity_n_m", np.full(17, np.inf)),
        ("joint_accelerations_rad_s2", np.zeros(16)),
        ("joint_accelerations_rad_s2", np.full(17, np.nan)),
        ("previous_joint_accelerations_rad_s2", np.zeros(16)),
        ("previous_joint_accelerations_rad_s2", np.full(17, np.inf)),
        ("control_period_seconds", 0.0),
        ("control_period_seconds", -0.01),
        ("control_period_seconds", float("nan")),
        ("control_period_seconds", True),
    ],
)
def test_new_control_inputs_fail_closed(field: str, bad: object) -> None:
    with pytest.raises(ExperimentContractError, match=field):
        _evaluate(**{field: bad})


def test_torque_exceeding_declared_joint_capacity_fails_closed() -> None:
    with pytest.raises(ExperimentContractError, match="exceeds"):
        _evaluate(
            generalized_actuator_torque_n_m=np.full(17, 10.1),
            generalized_actuator_torque_capacity_n_m=np.full(17, 10.0),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contact_index": -1},
        {"geom1_name": ""},
        {"geom2_name": "   "},
        {"involves_floor": 1},
        {"forbidden_floor_contact": "yes"},
        {"normal_force_n": -0.1},
        {"normal_force_n": float("nan")},
    ],
)
def test_contact_fact_fails_closed(kwargs: dict[str, Any]) -> None:
    values: dict[str, Any] = {
        "contact_index": 0,
        "geom1_name": "floor",
        "geom2_name": "torso",
        "involves_floor": True,
        "forbidden_floor_contact": True,
        "normal_force_n": 1.0,
    }
    values.update(kwargs)
    with pytest.raises(ExperimentContractError):
        ContactFact(**values)


def test_forbidden_classification_requires_exact_floor_and_permitted_geom_facts() -> None:
    with pytest.raises(ExperimentContractError, match="static-stand geom set"):
        _contact(0, floor=False, forbidden=True, force_n=1.0)
    with pytest.raises(ExperimentContractError, match="floor geom identity"):
        ContactFact(
            contact_index=0,
            geom1_name="floor",
            geom2_name="left_foot",
            involves_floor=False,
            forbidden_floor_contact=False,
            normal_force_n=1.0,
        )
    with pytest.raises(ExperimentContractError, match="static-stand geom set"):
        ContactFact(
            contact_index=0,
            geom1_name="floor",
            geom2_name="torso1",
            involves_floor=True,
            forbidden_floor_contact=False,
            normal_force_n=1.0,
        )


@pytest.mark.parametrize(
    "contacts",
    [
        "not-a-sequence",
        (object(),),
        (_contact(1, floor=True, forbidden=False, force_n=1.0),),
        (
            _contact(0, floor=True, forbidden=False, force_n=1.0),
            _contact(0, floor=True, forbidden=False, force_n=1.0),
        ),
    ],
)
def test_contact_coverage_and_types_fail_closed(contacts: object) -> None:
    with pytest.raises(ExperimentContractError, match="contact_facts"):
        _evaluate(contact_facts=contacts)


def test_nonfinite_derived_jerk_fails_closed() -> None:
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ExperimentContractError, match="joint jerk"),
    ):
        _evaluate(
            joint_accelerations_rad_s2=np.full(17, 1e308),
            previous_joint_accelerations_rad_s2=np.full(17, -1e308),
        )


def test_accumulator_rejects_untyped_metrics_and_out_of_range_reward() -> None:
    accumulator = _accumulator(evaluation_seed=7, requested_steps=2)
    with pytest.raises(ExperimentContractError, match="StepMetrics"):
        accumulator.add(object(), tracking_reward=0.0)  # type: ignore[arg-type]

    valid = _evaluate()
    with pytest.raises(ExperimentContractError, match=r"\[0, 1\]"):
        accumulator.add(valid, tracking_reward=1.1)
    accumulator.add(valid, tracking_reward=1.0)
    partial = accumulator.finish(require_complete=False)
    assert partial.observed_steps == 1
    assert partial.tracking_return == 1.0


def test_step_metric_invariants_fail_closed_when_constructed_directly() -> None:
    metrics = _evaluate(contact_facts=(_contact(0, floor=True, forbidden=True, force_n=1.0),))
    with pytest.raises(ExperimentContractError, match="cannot exceed 1"):
        replace(metrics, normalized_policy_action_max_abs=1.1)
    with pytest.raises(ExperimentContractError, match="cannot exceed"):
        replace(
            metrics,
            normalized_policy_action_rms=0.6,
            normalized_policy_action_max_abs=0.5,
        )
    with pytest.raises(ExperimentContractError, match="must match"):
        replace(
            metrics,
            generalized_actuator_torque_normalized_rms=0.1,
            generalized_actuator_torque_normalized_max_abs=0.1,
        )
    with pytest.raises(ExperimentContractError, match="contact counts"):
        replace(metrics, floor_contact_count=0)
    with pytest.raises(ExperimentContractError, match="does not match contact counts"):
        replace(metrics, forbidden_floor_contact_fraction=0.5)
    with pytest.raises(ExperimentContractError, match="floor contact peak"):
        replace(metrics, floor_contact_peak_normal_force_n=0.5)
    with pytest.raises(ExperimentContractError, match="must be boolean"):
        replace(metrics, collapsed=1)  # type: ignore[arg-type]
    with pytest.raises(ExperimentContractError, match="must match the frozen"):
        replace(metrics, collapsed=True)
    with pytest.raises(ExperimentContractError, match="too small"):
        replace(
            _evaluate(
                normalized_policy_action=np.full(17, 0.5),
                generalized_actuator_torque_n_m=0.5 * TORQUE_CAPACITY_N_M,
            ),
            normalized_policy_action_rms=0.0,
        )


def test_episode_metric_cross_field_invariants_fail_closed() -> None:
    accumulator = _accumulator(evaluation_seed=7, requested_steps=2)
    metrics = _evaluate()
    accumulator.add(metrics, tracking_reward=0.5)
    accumulator.add(metrics, tracking_reward=0.5)
    episode = accumulator.finish()

    with pytest.raises(ExperimentContractError, match="cannot exceed"):
        replace(
            episode,
            normalized_policy_action_rms=0.5,
            normalized_policy_action_max_abs=0.0,
        )
    with pytest.raises(ExperimentContractError, match="contact counts"):
        replace(episode, floor_contact_count=1)
    with pytest.raises(ExperimentContractError, match="integer collapsed-step"):
        replace(
            episode,
            survived_full_horizon=False,
            first_collapse_step=1,
            collapse_fraction=0.25,
        )
    with pytest.raises(ExperimentContractError, match="tracking_return"):
        replace(episode, tracking_return=2.1)
    with pytest.raises(ExperimentContractError, match="must match"):
        replace(
            episode,
            generalized_actuator_torque_normalized_rms=0.1,
            generalized_actuator_torque_normalized_max_abs=0.1,
        )
    with pytest.raises(ExperimentContractError, match="action-delta metrics must be zero"):
        replace(
            episode,
            normalized_policy_action_delta_rate_rms_per_s=1.0,
            normalized_policy_action_delta_rate_max_abs_per_s=1.0,
        )
    with pytest.raises(ExperimentContractError, match="capacity envelope"):
        replace(
            episode,
            normalized_policy_action_rms=0.5,
            normalized_policy_action_max_abs=0.5,
            generalized_actuator_torque_normalized_rms=0.5,
            generalized_actuator_torque_normalized_max_abs=0.5,
            generalized_actuator_torque_rms_n_m=0.0,
            generalized_actuator_torque_max_abs_n_m=0.0,
        )
    with pytest.raises(ExperimentContractError, match="too small"):
        replace(
            episode,
            normalized_policy_action_max_abs=0.5,
            generalized_actuator_torque_normalized_max_abs=0.5,
            generalized_actuator_torque_max_abs_n_m=5.0,
        )


def test_episode_action_delta_and_collapse_timing_algebra_fail_closed() -> None:
    accumulator = _accumulator(evaluation_seed=7, requested_steps=2)
    action = np.full(17, 0.1)
    accumulator.add(
        _evaluate(
            normalized_policy_action=action,
            previous_normalized_policy_action=np.zeros(17),
            generalized_actuator_torque_n_m=0.1 * TORQUE_CAPACITY_N_M,
        ),
        tracking_reward=0.5,
    )
    accumulator.add(
        _evaluate(
            normalized_policy_action=action,
            previous_normalized_policy_action=action,
            generalized_actuator_torque_n_m=0.1 * TORQUE_CAPACITY_N_M,
        ),
        tracking_reward=0.5,
    )
    episode = accumulator.finish()
    with pytest.raises(ExperimentContractError, match="zero-reset action/cadence bound"):
        replace(
            episode,
            normalized_policy_action_delta_rate_rms_per_s=100.0,
            normalized_policy_action_delta_rate_max_abs_per_s=100.0,
        )
    with pytest.raises(ExperimentContractError, match="too small for the zero-reset"):
        replace(
            episode,
            normalized_policy_action_delta_rate_rms_per_s=0.0,
            normalized_policy_action_delta_rate_max_abs_per_s=0.0,
        )

    collapse_accumulator = _accumulator(evaluation_seed=7, requested_steps=4)
    for _ in range(4):
        collapse_accumulator.add(_evaluate(), tracking_reward=0.5)
    complete = collapse_accumulator.finish()
    with pytest.raises(ExperimentContractError, match="steps at or after"):
        replace(
            complete,
            survived_full_horizon=False,
            first_collapse_step=4,
            collapse_fraction=0.5,
        )


def test_episode_forbidden_contact_count_fraction_and_peak_must_agree() -> None:
    accumulator = _accumulator(evaluation_seed=7, requested_steps=2)
    metrics = _evaluate(contact_facts=(_contact(0, floor=True, forbidden=True, force_n=3.0),))
    accumulator.add(metrics, tracking_reward=0.5)
    accumulator.add(_evaluate(), tracking_reward=0.5)
    episode = accumulator.finish()

    assert episode.forbidden_floor_contact_count == 1
    assert episode.forbidden_floor_contact_step_fraction == 0.5
    with pytest.raises(ExperimentContractError, match="does not match contact counts"):
        replace(episode, forbidden_floor_contact_fraction=0.0)
    with pytest.raises(ExperimentContractError, match="integer step count"):
        replace(episode, forbidden_floor_contact_step_fraction=0.25)
    with pytest.raises(ExperimentContractError, match="peak force cannot exceed"):
        replace(episode, forbidden_floor_contact_peak_force_n=4.0)


def test_evaluator_source_is_content_addressed() -> None:
    assert len(protected_evaluator_sha256()) == 64
