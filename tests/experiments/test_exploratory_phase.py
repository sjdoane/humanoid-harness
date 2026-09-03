from __future__ import annotations

import numpy as np
import pytest

from oracle_composition.experiments.exploratory_phase import (
    BoundedPhaseConfig,
    BoundedPhaseMatcher,
    PhaseDecision,
    RecoveryGate,
    RecoveryGateConfig,
    RecoveryMode,
)
from oracle_composition.experiments.fixed_reference import ExperimentContractError


def _matcher(
    *,
    switch_margin: float = 0.1,
    search_radius_frames: int = 1,
    max_phase_advance_frames: int = 2,
    reference_rows: int = 6,
) -> BoundedPhaseMatcher:
    reference = np.arange(reference_rows * 2, dtype=np.float64).reshape(reference_rows, 2)
    return BoundedPhaseMatcher(
        reference=reference,
        scale=np.ones(2, dtype=np.float64),
        config=BoundedPhaseConfig(
            horizon_steps=2,
            search_radius_frames=search_radius_frames,
            anchor_penalty=0.05,
            switch_margin=switch_margin,
            match_indices=(0, 1),
            max_phase_advance_frames=max_phase_advance_frames,
        ),
    )


def _gate(
    *,
    entry_pose_error_dwell_steps: int = 3,
    exit_stable_steps: int = 2,
) -> RecoveryGate:
    return RecoveryGate(
        RecoveryGateConfig(
            entry_height_min=1.05,
            entry_height_max=1.8,
            entry_up_z_min=0.75,
            entry_pose_error_max=2.2,
            exit_height_min=1.1,
            exit_height_max=1.7,
            exit_up_z_min=0.8,
            exit_pose_error_max=1.5,
            entry_pose_error_dwell_steps=entry_pose_error_dwell_steps,
            exit_stable_steps=exit_stable_steps,
        )
    )


def _phase_decision(pose_error: float, *, phase: int = 2) -> PhaseDecision:
    return PhaseDecision(
        nominal_phase=phase,
        previous_selected_phase=phase - 1,
        baseline_phase=phase,
        selected_phase=phase,
        best_candidate_phase=phase,
        selected_pose_error=pose_error,
        best_candidate_pose_error=pose_error,
        score_gain=0.0,
        candidate_phase_bounds=(phase, phase),
    )


def _quaternion_matcher(reference: np.ndarray) -> BoundedPhaseMatcher:
    return BoundedPhaseMatcher(
        reference=reference,
        scale=np.ones(reference.shape[1], dtype=np.float64),
        config=BoundedPhaseConfig(
            horizon_steps=1,
            search_radius_frames=1,
            anchor_penalty=0.0,
            switch_margin=0.0,
            match_indices=(0, 1, 2, 3),
            quaternion_indices=(0, 1, 2, 3),
            quaternion_angle_scale=0.5,
        ),
    )


def test_selected_pose_error_belongs_to_dispatched_nominal_phase() -> None:
    matcher = _matcher(switch_margin=100.0)

    decision = matcher.select(np.asarray([6.0, 7.0]), nominal_phase=2)

    assert decision.best_candidate_phase == 3
    assert decision.selected_phase == 2
    assert decision.selected_pose_error == pytest.approx(4.0)
    assert decision.best_candidate_pose_error == pytest.approx(0.0)
    assert not decision.corrected


def test_recovery_gate_consumes_the_dispatched_phase_error() -> None:
    matcher = _matcher(switch_margin=100.0)
    decision = matcher.select(np.asarray([6.0, 7.0]), nominal_phase=2)
    gate = _gate(entry_pose_error_dwell_steps=1)

    recovery = gate.step(
        root_height_m=1.3,
        torso_up_z=0.9,
        phase_decision=decision,
    )

    assert decision.best_candidate_pose_error == pytest.approx(0.0)
    assert recovery.selected_phase == decision.selected_phase == 2
    assert recovery.selected_pose_error == pytest.approx(4.0)
    assert recovery.mode is RecoveryMode.RECOVER
    assert recovery.entry_reasons == ("pose_error",)


def test_phase_switch_uses_penalized_gain_and_stays_inside_nominal_leash() -> None:
    matcher = _matcher(switch_margin=0.1)

    decision = matcher.select(np.asarray([6.0, 7.0]), nominal_phase=2)

    assert decision.selected_phase == 3
    assert decision.applied_offset_frames == 1
    assert decision.selected_pose_error == pytest.approx(0.0)
    assert decision.score_gain == pytest.approx(3.95)
    assert np.array_equal(matcher.window(decision.selected_phase), [[6.0, 7.0], [8.0, 9.0]])


def test_phase_selection_limits_an_adversarial_forward_jump() -> None:
    matcher = _matcher(search_radius_frames=4, reference_rows=12)

    decision = matcher.select(
        np.asarray([8.0, 9.0]),
        nominal_phase=4,
        previous_selected_phase=0,
    )

    assert decision.best_candidate_phase == 2
    assert decision.selected_phase == 2
    assert decision.phase_advance_frames == 2
    assert decision.candidate_phase_bounds == (0, 2)
    assert abs(decision.applied_offset_frames) <= matcher.config.search_radius_frames


def test_phase_selection_rejects_a_better_backward_frame_within_the_leash() -> None:
    matcher = _matcher(search_radius_frames=2, reference_rows=12)

    decision = matcher.select(
        np.asarray([4.0, 5.0]),
        nominal_phase=3,
        previous_selected_phase=3,
    )

    assert decision.candidate_phase_bounds == (3, 5)
    assert decision.selected_phase == 3
    assert decision.phase_advance_frames == 0


def test_phase_selection_fails_closed_instead_of_jumping_across_a_nominal_gap() -> None:
    matcher = _matcher(search_radius_frames=1, reference_rows=12)

    with pytest.raises(ExperimentContractError, match="do not overlap"):
        matcher.select(
            np.asarray([16.0, 17.0]),
            nominal_phase=8,
            previous_selected_phase=0,
        )


def test_previous_phase_only_limits_continuity_not_the_nominal_leash() -> None:
    matcher = _matcher(search_radius_frames=2, reference_rows=12)

    first = matcher.select(
        np.asarray([0.0, 1.0]),
        nominal_phase=2,
        previous_selected_phase=0,
    )
    second = matcher.select(
        np.asarray([0.0, 1.0]),
        nominal_phase=3,
        previous_selected_phase=first.selected_phase,
    )

    assert first.selected_phase == 0
    assert second.selected_phase >= 1
    assert second.selected_phase >= second.nominal_phase - matcher.config.search_radius_frames


def test_quaternion_signs_have_identical_geodesic_pose_error() -> None:
    half_angle = np.pi / 4.0
    matcher = _quaternion_matcher(
        np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [np.cos(half_angle), np.sin(half_angle), 0.0, 0.0],
            ]
        )
    )

    positive = matcher.select(np.asarray([1.0, 0.0, 0.0, 0.0]), nominal_phase=0)
    negative = matcher.select(np.asarray([-1.0, 0.0, 0.0, 0.0]), nominal_phase=0)

    assert positive.selected_phase == negative.selected_phase == 0
    assert positive.selected_pose_error == pytest.approx(0.0)
    assert negative.selected_pose_error == pytest.approx(0.0)


def test_quaternion_angle_counts_once_and_excludes_its_four_coordinates() -> None:
    matcher = BoundedPhaseMatcher(
        reference=np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0]]),
        scale=np.ones(5, dtype=np.float64),
        config=BoundedPhaseConfig(
            horizon_steps=1,
            search_radius_frames=0,
            anchor_penalty=0.0,
            switch_margin=0.0,
            match_indices=(0, 1, 2, 3, 4),
            quaternion_indices=(0, 1, 2, 3),
            quaternion_angle_scale=np.pi,
        ),
    )

    decision = matcher.select(np.asarray([0.0, 1.0, 0.0, 0.0, 1.0]), nominal_phase=0)

    # One normalized quaternion-angle component and one scalar component both equal one.
    assert decision.selected_pose_error == pytest.approx(1.0)


def test_omitted_quaternion_configuration_retains_componentwise_scoring() -> None:
    matcher = BoundedPhaseMatcher(
        reference=np.asarray([[1.0, 0.0, 0.0, 0.0]]),
        scale=np.ones(4, dtype=np.float64),
        config=BoundedPhaseConfig(
            horizon_steps=1,
            search_radius_frames=0,
            anchor_penalty=0.0,
            switch_margin=0.0,
            match_indices=(0, 1, 2, 3),
        ),
    )

    decision = matcher.select(np.asarray([-1.0, 0.0, 0.0, 0.0]), nominal_phase=0)

    assert decision.selected_pose_error == pytest.approx(1.0)


def test_quaternion_matcher_rejects_non_unit_reference_and_current_values() -> None:
    with pytest.raises(ExperimentContractError, match="reference quaternions must be unit"):
        _quaternion_matcher(
            np.asarray(
                [
                    [2.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
            )
        )

    matcher = _quaternion_matcher(
        np.asarray(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        )
    )
    with pytest.raises(ExperimentContractError, match="current quaternion must be unit"):
        matcher.select(np.asarray([0.5, 0.0, 0.0, 0.0]), nominal_phase=0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"quaternion_indices": (0, 1, 2)}, "four unique"),
        ({"quaternion_indices": (0, 1, 2, 4)}, "included in match_indices"),
        ({"quaternion_angle_scale": 0.0}, "angle_scale must be positive"),
        ({"quaternion_unit_tolerance": 1e-2}, "at most 1e-3"),
    ],
)
def test_quaternion_configuration_is_explicit_and_validated(
    overrides: dict[str, object],
    message: str,
) -> None:
    arguments: dict[str, object] = {
        "horizon_steps": 1,
        "search_radius_frames": 1,
        "anchor_penalty": 0.0,
        "switch_margin": 0.0,
        "match_indices": (0, 1, 2, 3),
        "quaternion_indices": (0, 1, 2, 3),
    }
    arguments.update(overrides)

    with pytest.raises(ExperimentContractError, match=message):
        BoundedPhaseConfig(**arguments)


def test_phase_config_normalizes_huge_numeric_overflow() -> None:
    with pytest.raises(ExperimentContractError, match="anchor_penalty must be finite"):
        BoundedPhaseConfig(
            horizon_steps=1,
            search_radius_frames=1,
            anchor_penalty=10**400,
            switch_margin=0.0,
            match_indices=(0,),
        )


def test_phase_selection_clamps_only_the_terminal_nominal_phase() -> None:
    matcher = _matcher()

    decision = matcher.select(np.asarray([8.0, 9.0]), nominal_phase=99)

    assert decision.nominal_phase == matcher.last_start == 4
    assert decision.selected_phase == 4


def test_terminal_window_advances_to_final_target_then_holds() -> None:
    reference = np.arange(40, dtype=np.float64).reshape(10, 4)
    matcher = BoundedPhaseMatcher(
        reference=reference,
        scale=np.ones(4),
        config=BoundedPhaseConfig(4, 0, 0.0, 0.0, (0, 1, 2, 3)),
    )

    decision = matcher.select(reference[8], nominal_phase=999)
    window = matcher.window(decision.selected_phase)

    assert matcher.last_start == decision.selected_phase == 8
    assert np.array_equal(window, reference[[8, 9, 9, 9]])
    assert np.array_equal(window[1], reference[-1])


def test_phase_matcher_rejects_nonfinite_derived_scores() -> None:
    matcher = BoundedPhaseMatcher(
        reference=np.asarray([[0.0], [1e308]]),
        scale=np.ones(1),
        config=BoundedPhaseConfig(1, 1, 0.0, 0.0, (0,)),
    )

    with pytest.raises(ExperimentContractError, match="derived phase errors"):
        matcher.select(np.asarray([5e307]), nominal_phase=0)


@pytest.mark.parametrize(
    ("reference", "scale", "message"),
    [
        (np.zeros((1, 2)), np.ones(2), "one horizon"),
        (np.zeros((4, 2)), np.ones(1), "match the reference width"),
        (np.zeros((4, 2)), np.asarray([1.0, 0.0]), "positive"),
    ],
)
def test_phase_matcher_rejects_invalid_arrays(
    reference: np.ndarray,
    scale: np.ndarray,
    message: str,
) -> None:
    config = BoundedPhaseConfig(2, 1, 0.05, 0.1, (0, 1))

    with pytest.raises(ExperimentContractError, match=message):
        BoundedPhaseMatcher(reference=reference, scale=scale, config=config)


@pytest.mark.parametrize(
    ("reference", "scale", "message"),
    [
        (["not-numeric"], np.ones(1), "reference must be convertible"),
        ([[10**400]], np.ones(1), "reference must be convertible"),
        (np.zeros((2, 1)), ["not-numeric"], "scale must be convertible"),
    ],
)
def test_phase_matcher_normalizes_array_conversion_errors(
    reference: object,
    scale: object,
    message: str,
) -> None:
    config = BoundedPhaseConfig(1, 0, 0.0, 0.0, (0,))

    with pytest.raises(ExperimentContractError, match=message):
        BoundedPhaseMatcher(reference=reference, scale=scale, config=config)


@pytest.mark.parametrize("current", [["not-numeric"], [10**400]])
def test_phase_matcher_normalizes_current_conversion_errors(current: object) -> None:
    matcher = BoundedPhaseMatcher(
        reference=np.zeros((1, 1)),
        scale=np.ones(1),
        config=BoundedPhaseConfig(1, 0, 0.0, 0.0, (0,)),
    )

    with pytest.raises(ExperimentContractError, match="current state must be convertible"):
        matcher.select(current, nominal_phase=0)


def test_recovery_gate_uses_hysteresis_and_exact_exit_dwell() -> None:
    gate = _gate(exit_stable_steps=2)

    entered = gate.step(root_height_m=1.3, torso_up_z=0.7, phase_decision=_phase_decision(1.0))
    first_stable = gate.step(
        root_height_m=1.3,
        torso_up_z=0.9,
        phase_decision=_phase_decision(1.0),
    )
    exited = gate.step(root_height_m=1.3, torso_up_z=0.9, phase_decision=_phase_decision(1.0))

    assert entered.mode is RecoveryMode.RECOVER
    assert entered.entry_reasons == ("torso_up_z",)
    assert entered.residual_weight == 0.0
    assert first_stable.mode is RecoveryMode.RECOVER
    assert first_stable.stable_steps == 1
    assert exited.mode is RecoveryMode.TRACK
    assert exited.exited
    assert exited.residual_weight == 1.0


def test_pose_error_requires_consecutive_entry_dwell() -> None:
    gate = _gate(entry_pose_error_dwell_steps=3)

    first = gate.step(root_height_m=1.3, torso_up_z=0.9, phase_decision=_phase_decision(3.0))
    second = gate.step(root_height_m=1.3, torso_up_z=0.9, phase_decision=_phase_decision(3.0))
    entered = gate.step(root_height_m=1.3, torso_up_z=0.9, phase_decision=_phase_decision(3.0))

    assert first.mode is RecoveryMode.TRACK
    assert first.pose_error_bad_steps == 1
    assert second.mode is RecoveryMode.TRACK
    assert second.pose_error_bad_steps == 2
    assert entered.mode is RecoveryMode.RECOVER
    assert entered.pose_error_bad_steps == 3
    assert entered.entry_reasons == ("pose_error",)


def test_pose_error_entry_dwell_resets_after_one_good_sample() -> None:
    gate = _gate(entry_pose_error_dwell_steps=2)
    gate.step(root_height_m=1.3, torso_up_z=0.9, phase_decision=_phase_decision(3.0))

    reset = gate.step(root_height_m=1.3, torso_up_z=0.9, phase_decision=_phase_decision(1.0))
    next_bad = gate.step(
        root_height_m=1.3,
        torso_up_z=0.9,
        phase_decision=_phase_decision(3.0),
    )

    assert reset.pose_error_bad_steps == 0
    assert next_bad.mode is RecoveryMode.TRACK
    assert next_bad.pose_error_bad_steps == 1


@pytest.mark.parametrize(
    ("root_height_m", "torso_up_z", "reason"),
    [(1.0, 0.9, "height"), (1.3, 0.7, "torso_up_z")],
)
def test_physical_violation_enters_immediately_without_pose_dwell(
    root_height_m: float,
    torso_up_z: float,
    reason: str,
) -> None:
    gate = _gate(entry_pose_error_dwell_steps=20)

    decision = gate.step(
        root_height_m=root_height_m,
        torso_up_z=torso_up_z,
        phase_decision=_phase_decision(1.0),
    )

    assert decision.mode is RecoveryMode.RECOVER
    assert decision.entered
    assert decision.entry_reasons == (reason,)


def test_recovery_exit_counter_resets_when_any_condition_fails() -> None:
    gate = _gate(exit_stable_steps=2)
    gate.step(root_height_m=1.0, torso_up_z=0.9, phase_decision=_phase_decision(1.0))
    gate.step(root_height_m=1.3, torso_up_z=0.9, phase_decision=_phase_decision(1.0))

    decision = gate.step(
        root_height_m=1.3,
        torso_up_z=0.79,
        phase_decision=_phase_decision(1.0),
    )

    assert decision.mode is RecoveryMode.RECOVER
    assert decision.stable_steps == 0
    assert not decision.exited


def test_recovery_config_rejects_exit_thresholds_without_hysteresis() -> None:
    with pytest.raises(ExperimentContractError, match="hysteresis"):
        RecoveryGateConfig(
            entry_height_min=1.05,
            entry_height_max=1.8,
            entry_up_z_min=0.75,
            entry_pose_error_max=2.2,
            exit_height_min=1.0,
            exit_height_max=1.7,
            exit_up_z_min=0.8,
            exit_pose_error_max=1.5,
            entry_pose_error_dwell_steps=3,
            exit_stable_steps=20,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("entry_height_min", -1.0, "non-negative"),
        ("exit_height_min", -1.0, "non-negative"),
        ("entry_pose_error_max", -0.1, "non-negative"),
        ("exit_pose_error_max", -0.1, "non-negative"),
        ("entry_up_z_min", 1.1, r"\[-1, 1\]"),
        ("exit_up_z_min", -1.1, r"\[-1, 1\]"),
    ],
)
def test_recovery_config_rejects_nonphysical_domains(
    field: str,
    value: float,
    message: str,
) -> None:
    arguments = {
        "entry_height_min": 1.05,
        "entry_height_max": 1.8,
        "entry_up_z_min": 0.75,
        "entry_pose_error_max": 2.2,
        "exit_height_min": 1.1,
        "exit_height_max": 1.7,
        "exit_up_z_min": 0.8,
        "exit_pose_error_max": 1.5,
        "entry_pose_error_dwell_steps": 3,
        "exit_stable_steps": 20,
    }
    arguments[field] = value

    with pytest.raises(ExperimentContractError, match=message):
        RecoveryGateConfig(**arguments)


def test_recovery_config_normalizes_huge_numeric_overflow() -> None:
    with pytest.raises(ExperimentContractError, match="entry_height_min must be finite"):
        RecoveryGateConfig(
            entry_height_min=10**400,
            entry_height_max=1.8,
            entry_up_z_min=0.75,
            entry_pose_error_max=2.2,
            exit_height_min=1.1,
            exit_height_max=1.7,
            exit_up_z_min=0.8,
            exit_pose_error_max=1.5,
            entry_pose_error_dwell_steps=3,
            exit_stable_steps=20,
        )


@pytest.mark.parametrize("torso_up_z", [-1.01, 1.01])
def test_recovery_gate_rejects_impossible_torso_up_values(torso_up_z: float) -> None:
    with pytest.raises(ExperimentContractError, match=r"\[-1, 1\]"):
        _gate().step(
            root_height_m=1.3,
            torso_up_z=torso_up_z,
            phase_decision=_phase_decision(1.0),
        )
