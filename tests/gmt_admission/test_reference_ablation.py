from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from oracle_composition.adapters.gmt.composition import ComposedReference, ReferenceSegment
from oracle_composition.adapters.gmt.contracts import (
    ACTION_DIM,
    OBSERVATION_DIM,
    PROPRIOCEPTION_DIM,
    REFERENCE_FRAME_DIM,
    REFERENCE_HORIZON,
    REFERENCE_OFFSETS,
)
from oracle_composition.adapters.gmt.control_runtime import PreparedControl
from oracle_composition.adapters.gmt.io import GMTAdmissionError
from oracle_composition.adapters.gmt.reference_ablation import (
    REFERENCE_ABLATION_ARMS,
    SHIFT_SECONDS,
    SHUFFLE_SEED,
    AblatedComposedReference,
    ReferenceInputAudit,
    actor_reference_window,
    fixed_shuffle_permutation,
    reference_ablation_contract,
    validate_reference_input_audit,
)
from oracle_composition.adapters.gmt.reference_ablation_evidence import (
    capture_zero_residual_actor_evidence,
    measure_practical_trajectory_divergence,
    validate_recomputed_base_action,
)
from oracle_composition.adapters.gmt.reference_runtime import ReferenceMotion
from oracle_composition.harness.contract import oracle_program_from_dict


def _motion(base_height: float) -> ReferenceMotion:
    frames = 601
    root = np.zeros((frames, 3), dtype="<f4")
    root[:, 0] = np.linspace(0.0, 4.0, frames, dtype="<f4")
    root[:, 2] = base_height + np.linspace(0.0, 0.1, frames, dtype="<f4")
    rotation = np.zeros((frames, 4), dtype="<f4")
    rotation[:, 3] = 1.0
    joints = np.zeros((frames, ACTION_DIM), dtype="<f4")
    joints[:, 0] = np.linspace(-0.5, 0.5, frames, dtype="<f4")
    return ReferenceMotion(
        {
            "fps": np.asarray([30.0], dtype="<f8"),
            "root_pos": root,
            "root_rot": rotation,
            "dof_pos": joints,
        }
    )


def _oracle() -> ComposedReference:
    program = oracle_program_from_dict(
        {
            "schema_version": 1,
            "evidence_class": "exploratory_oracle_cycle",
            "oracle_id": "reference_ablation_fixture",
            "behaviors": ["walk", "crouch"],
            "initial": "before",
            "states": {
                "before": {"behavior": "walk", "min_dwell": 1},
                "inside": {"behavior": "crouch", "min_dwell": 1},
                "after": {"behavior": "walk", "min_dwell": 1},
            },
            "transitions": [
                {"from": "before", "to": "inside", "priority": 0, "guard": "x_travelled >= 1"},
                {"from": "inside", "to": "after", "priority": 0, "guard": "x_travelled >= 2"},
            ],
        },
        available_behaviors=["walk", "crouch"],
    )
    return ComposedReference(
        program,
        {
            "walk": ReferenceSegment(_motion(0.8), "a" * 64, 0.0, 12.0),
            "crouch": ReferenceSegment(
                _motion(0.5),
                "b" * 64,
                0.0,
                12.0,
                boundary="hold_last_pose_zero_velocity",
            ),
        },
    )


def _command(oracle, step: int, x: float = 0.0):
    return oracle.command(
        step=step,
        signals={
            "t": step * 0.02,
            "v_x": 0.5,
            "v_target": 0.5,
            "z_root": 0.8,
            "torso_up": 1.0,
            "x_travelled": x,
        },
        robot_pose=np.asarray([0.8, *([0.0] * 25)], dtype="<f8"),
    )


def _assert_command_equal(left, right) -> None:
    assert (
        left.state,
        left.behavior,
        left.phase_seconds,
        left.phase_fraction,
        left.segment_sha256,
        None if left.transition is None else dict(left.transition),
    ) == (
        right.state,
        right.behavior,
        right.phase_seconds,
        right.phase_fraction,
        right.segment_sha256,
        None if right.transition is None else dict(right.transition),
    )
    np.testing.assert_array_equal(left.current, right.current)
    np.testing.assert_array_equal(left.window, right.window)
    assert left.current.tobytes() == right.current.tobytes()
    assert left.window.tobytes() == right.window.tobytes()


def test_exact_arm_is_a_bitwise_composed_runtime_noop() -> None:
    ordinary = _oracle()
    ablated = AblatedComposedReference(_oracle(), arm="exact")
    for step, x in enumerate((0.0, 1.5, 1.5, 2.5, 2.5)):
        expected = _command(ordinary, step, x)
        observed = _command(ablated, step, x)
        _assert_command_equal(observed, expected)
        _assert_command_equal(ablated.last_audit.original_command, expected)
        np.testing.assert_array_equal(ablated.last_audit.actor_reference_window, expected.window)
        np.testing.assert_array_equal(
            ablated.current_after_step(step + 1), ordinary.current_after_step(step + 1)
        )

    ablated.reset()
    with pytest.raises(GMTAdmissionError, match="no reference command"):
        _ = ablated.last_audit
    _assert_command_equal(_command(ablated, 0), _command(_oracle(), 0))


def test_fixed_interventions_preserve_original_command_and_have_exact_semantics() -> None:
    ordinary = _command(_oracle(), 0)
    observed: dict[str, ReferenceInputAudit] = {}
    for arm in REFERENCE_ABLATION_ARMS:
        ablated = AblatedComposedReference(_oracle(), arm=arm)
        _command(ablated, 0)
        observed[arm] = ablated.last_audit
        _assert_command_equal(ablated.last_audit.original_command, ordinary)
        validate_reference_input_audit(ablated.last_audit, ablated.segments)

    np.testing.assert_array_equal(observed["exact"].actor_reference_window, ordinary.window)
    assert observed["zero_reference"].actor_reference_window.tobytes() == np.zeros(
        (REFERENCE_HORIZON, REFERENCE_FRAME_DIM), dtype="<f4"
    ).tobytes()
    np.testing.assert_array_equal(
        observed["current_frame_repeated"].actor_reference_window,
        np.repeat(ordinary.current.reshape(1, -1), REFERENCE_HORIZON, axis=0),
    )
    np.testing.assert_array_equal(
        observed["shuffled_reference"].actor_reference_window,
        ordinary.window[fixed_shuffle_permutation()],
    )
    segment = _oracle().segments[ordinary.behavior]
    phase = torch.tensor(ordinary.phase_seconds, dtype=torch.float32)
    offsets = torch.tensor(REFERENCE_OFFSETS, dtype=torch.float32) * 0.02
    expected_shift = segment.features(phase + offsets + SHIFT_SECONDS).numpy()
    np.testing.assert_array_equal(
        observed["shifted_reference"].actor_reference_window,
        expected_shift,
    )


def test_shift_uses_the_active_segment_without_inventing_a_future_transition() -> None:
    ordinary = _oracle()
    ablated = AblatedComposedReference(_oracle(), arm="shifted_reference")
    _command(ordinary, 0, 0.0)
    _command(ablated, 0, 0.0)
    ordinary.current_after_step(1)
    ablated.current_after_step(1)
    _command(ordinary, 1, 1.5)
    switched = _command(ablated, 1, 1.5)

    assert switched.behavior == "crouch"
    assert ablated.last_audit.original_command.behavior == "crouch"
    assert np.max(ablated.last_audit.actor_reference_window[:, 0]) < 0.7
    np.testing.assert_array_equal(
        ablated.current_after_step(2), ordinary.current_after_step(2)
    )


def test_shift_uses_canonical_entry_loop_phase_mapping() -> None:
    segment = ReferenceSegment(
        _motion(0.8),
        "c" * 64,
        0.0,
        12.0,
        entry_phase_end_seconds=2.0,
        boundary="entry_once_then_loop",
        loop_start_seconds=4.0,
    )
    original = _command(_oracle(), 0)
    original = replace(
        original,
        behavior="loop",
        segment_sha256=segment.sha256,
        phase_seconds=10.0,
    )

    observed = actor_reference_window("shifted_reference", original, {"loop": segment})
    shifted_phases = (
        torch.tensor(10.0, dtype=torch.float32)
        + torch.tensor(REFERENCE_OFFSETS, dtype=torch.float32) * 0.02
        + SHIFT_SECONDS
    )

    np.testing.assert_array_equal(observed, segment.features(shifted_phases).numpy())
    assert not np.array_equal(observed, original.window)


def test_contract_freezes_arms_permutation_shift_and_claim_thresholds() -> None:
    contract = reference_ablation_contract()
    assert contract["artifact"] == "gmt_g1_closed_loop_reference_ablation"
    assert contract["evidence_class"] == "development_closed_loop_reference_input_effect"
    assert contract["arms"] == [
        "exact",
        "zero_reference",
        "current_frame_repeated",
        "shuffled_reference",
        "shifted_reference",
    ]
    assert contract["shuffle"] == {
        "rng": "numpy.random.PCG64",
        "seed": SHUFFLE_SEED,
        "permutation": [9, 7, 2, 5, 0, 11, 12, 13, 14, 3, 19, 15, 18, 10, 6, 1, 16, 4, 17, 8],
    }
    assert contract["shift_control_steps"] == 250
    assert contract["shift_seconds"] == 5.0
    assert contract["episode"] == {
        "runtime": "fresh_plant_actor_oracle_per_arm",
        "stop": "task_horizon_or_first_fall",
    }
    assert contract["reported_reference_errors"] == {
        "objective": "original_poststep_target",
        "actor_window_first_row": "supplemental_offset_or_permuted_target",
    }
    assert contract["practical_trajectory_divergence"] == {
        "root_translation_m": 0.001,
        "joint_angle_rad": 0.001,
        "root_orientation_geodesic_rad": 0.001,
        "report_windows_control_steps": [50, "full_common_observed_horizon"],
        "post_fall_comparison": "unavailable",
    }


def _qpos_trace(control_steps: int) -> np.ndarray:
    result = np.zeros((control_steps + 1, 30), dtype="<f8")
    result[:, 3] = 1.0
    return result


def test_practical_divergence_uses_sign_invariant_pose_and_common_horizon() -> None:
    exact = _qpos_trace(60)
    treated = _qpos_trace(30)
    treated[:, 3:7] *= -1.0
    treated[2, 0] += 0.0011
    treated[10, -1] += 0.0012
    angle = 0.0012
    treated[20, 3:7] = (
        np.cos(angle / 2.0),
        0.0,
        0.0,
        np.sin(angle / 2.0),
    )

    result = measure_practical_trajectory_divergence(exact, treated)

    assert result["first_50_control_steps"] == result["full_common_observed_horizon"]
    assert result["full_common_observed_horizon"]["compared_control_steps"] == 30
    assert result["full_common_observed_horizon"]["practical_divergence_observed"] is True
    assert result["full_common_observed_horizon"]["first_practical_divergence_control_step"] == 2
    assert result["full_common_observed_horizon"]["maximum_root_translation_m"] == pytest.approx(
        0.0011
    )
    assert result["full_common_observed_horizon"][
        "maximum_root_orientation_geodesic_rad"
    ] == pytest.approx(angle)
    assert result["post_common_observed_horizon"] == "unavailable"


def test_practical_divergence_rejects_malformed_pose_and_has_no_false_sign_effect() -> None:
    exact = _qpos_trace(2)
    sign_flipped = exact.copy()
    sign_flipped[:, 3:7] *= -1.0
    result = measure_practical_trajectory_divergence(exact, sign_flipped)
    assert result["full_common_observed_horizon"]["practical_divergence_observed"] is False

    malformed = exact.copy()
    malformed[1, 3:7] = 0.0
    with pytest.raises(GMTAdmissionError, match="zero quaternion"):
        measure_practical_trajectory_divergence(exact, malformed)


class _ReferenceActor(torch.nn.Module):
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return observation[:, :ACTION_DIM]


def _actor() -> _ReferenceActor:
    result = _ReferenceActor()
    result.eval()
    return result


def _actor_evidence():
    ablated = AblatedComposedReference(_oracle(), arm="shuffled_reference")
    _command(ablated, 0)
    actor_observation = np.zeros(OBSERVATION_DIM, dtype="<f4")
    actor_observation[:600] = ablated.last_audit.actor_reference_window.reshape(-1)
    base_raw = _actor()(torch.from_numpy(actor_observation).unsqueeze(0)).numpy()[0]
    prepared = PreparedControl(
        control_step=0,
        proprio=np.zeros(PROPRIOCEPTION_DIM, dtype="<f8"),
        obs=actor_observation,
        base_raw=np.ascontiguousarray(base_raw, dtype="<f4"),
    )
    zero = np.zeros(ACTION_DIM, dtype="<f4")
    evidence = capture_zero_residual_actor_evidence(
        audit=ablated.last_audit,
        prepared=prepared,
        residual_action=zero,
        composite_raw_action=prepared.base_raw.copy(),
    )
    return ablated, prepared, evidence


def test_zero_residual_actor_evidence_recomputes_and_is_immutable() -> None:
    ablated, _prepared, evidence = _actor_evidence()
    validate_reference_input_audit(evidence.audit, ablated.segments)
    validate_recomputed_base_action(evidence, _actor())
    assert not evidence.actor_observation.flags.writeable
    assert not evidence.audit.original_command.window.flags.writeable
    assert not evidence.audit.actor_reference_window.flags.writeable


def test_actor_evidence_rejects_reference_residual_action_and_recompute_tampering() -> None:
    ablated, prepared, evidence = _actor_evidence()
    mismatched_observation = prepared.obs.copy()
    mismatched_observation[0] += np.float32(0.1)
    with pytest.raises(GMTAdmissionError, match="audited reference window"):
        capture_zero_residual_actor_evidence(
            audit=ablated.last_audit,
            prepared=replace(prepared, obs=mismatched_observation),
            residual_action=np.zeros(ACTION_DIM, dtype="<f4"),
            composite_raw_action=prepared.base_raw.copy(),
        )
    nonzero = np.zeros(ACTION_DIM, dtype="<f4")
    nonzero[0] = np.float32(0.01)
    with pytest.raises(GMTAdmissionError, match="literal zero residual"):
        capture_zero_residual_actor_evidence(
            audit=ablated.last_audit,
            prepared=prepared,
            residual_action=nonzero,
            composite_raw_action=prepared.base_raw.copy(),
        )
    signed_zero = np.zeros(ACTION_DIM, dtype="<f4")
    signed_zero[0] = np.float32(-0.0)
    with pytest.raises(GMTAdmissionError, match="literal zero residual"):
        capture_zero_residual_actor_evidence(
            audit=ablated.last_audit,
            prepared=prepared,
            residual_action=signed_zero,
            composite_raw_action=prepared.base_raw.copy(),
        )
    with pytest.raises(GMTAdmissionError, match="preserve the base raw action"):
        capture_zero_residual_actor_evidence(
            audit=ablated.last_audit,
            prepared=prepared,
            residual_action=np.zeros(ACTION_DIM, dtype="<f4"),
            composite_raw_action=prepared.base_raw + np.float32(0.01),
        )
    forged_action = evidence.base_raw_action.copy()
    forged_action[0] += np.float32(0.01)
    with pytest.raises(GMTAdmissionError, match="recomputed base action"):
        validate_recomputed_base_action(
            replace(
                evidence,
                base_raw_action=forged_action,
                composite_raw_action=forged_action.copy(),
            ),
            _actor(),
        )


def test_audit_rejects_intervention_or_segment_identity_tampering() -> None:
    ablated = AblatedComposedReference(_oracle(), arm="shifted_reference")
    _command(ablated, 0)
    tampered = ablated.last_audit.actor_reference_window.copy()
    tampered[0, 0] += np.float32(0.01)
    forged = replace(ablated.last_audit, actor_reference_window=tampered)
    with pytest.raises(GMTAdmissionError, match="differs from its intervention"):
        validate_reference_input_audit(forged, ablated.segments)

    wrong_identity = replace(ablated.last_audit.original_command, segment_sha256="f" * 64)
    with pytest.raises(GMTAdmissionError, match="segment identity"):
        actor_reference_window("shifted_reference", wrong_identity, ablated.segments)
    with pytest.raises(GMTAdmissionError, match="fixed contract"):
        AblatedComposedReference(_oracle(), arm="caller_selected")
