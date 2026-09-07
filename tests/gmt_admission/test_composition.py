from __future__ import annotations

import numpy as np
import pytest
import torch

from oracle_composition.adapters.gmt.composition import ComposedReference, ReferenceSegment
from oracle_composition.adapters.gmt.contracts import CONTROL_DT_SECONDS, REFERENCE_OFFSETS
from oracle_composition.adapters.gmt.reference_runtime import ReferenceMotion
from oracle_composition.harness.contract import oracle_program_from_dict


def motion(height: float) -> ReferenceMotion:
    root = np.zeros((301, 3), dtype=np.float32)
    root[:, 2] = height
    rotations = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (301, 1))
    return ReferenceMotion(
        {
            "fps": np.array([30.0]),
            "root_pos": root,
            "root_rot": rotations,
            "dof_pos": np.zeros((301, 23), dtype=np.float32),
        }
    )


def composer() -> ComposedReference:
    program = oracle_program_from_dict(
        {
            "schema_version": 1,
            "evidence_class": "exploratory_oracle_cycle",
            "oracle_id": "fixture",
            "behaviors": ["walk", "crouch"],
            "initial": "before",
            "states": {
                "before": {"behavior": "walk", "min_dwell": 2},
                "inside": {"behavior": "crouch", "min_dwell": 2},
                "after": {"behavior": "walk", "min_dwell": 2},
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
            "walk": ReferenceSegment(motion(0.8), "a" * 64, 0.0, 10.0),
            "crouch": ReferenceSegment(motion(0.5), "b" * 64, 0.0, 10.0),
        },
    )


def command(oracle: ComposedReference, step: int, x: float = 0.0):
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
        robot_pose=np.array([0.8] + [0.0] * 25),
    )


def test_single_full_clip_preserves_exact_reference_windows():
    oracle = composer()
    for step in range(15):
        observed = command(oracle, step)
        np.testing.assert_array_equal(observed.window, oracle.segments["walk"].motion.window(step))
        np.testing.assert_array_equal(
            observed.current, oracle.segments["walk"].motion.current(step)
        )


def test_position_triggers_switch_not_elapsed_time_and_dwell_prevents_chatter():
    oracle = composer()
    assert command(oracle, 0, 1.5).behavior == "walk"
    assert command(oracle, 1, 1.5).behavior == "walk"
    switched = command(oracle, 2, 1.5)
    assert switched.behavior == "crouch"
    assert switched.transition["selected_phase_seconds"] == 0.0
    assert command(oracle, 3, 2.5).behavior == "crouch"
    assert command(oracle, 4, 2.5).behavior == "walk"
    oracle.reset()
    for step in range(20):
        assert command(oracle, step, 0.0).behavior == "walk"


def test_objective_uses_executed_mode_before_next_decision():
    oracle = composer()
    command(oracle, 0)
    command(oracle, 1)
    assert oracle.current_after_step(2)[0] == pytest.approx(0.8)
    assert command(oracle, 2, 1.5).current[0] == pytest.approx(0.5)


def test_repeated_or_skipped_decisions_and_invalid_segments_fail():
    oracle = composer()
    command(oracle, 0)
    with pytest.raises(ValueError, match="consecutive"):
        command(oracle, 0)
    with pytest.raises(ValueError, match="consecutive"):
        command(oracle, 2)
    with pytest.raises(ValueError, match="bounds"):
        ReferenceSegment(motion(0.8), "a" * 64, 2.0, 1.0)
    assert ReferenceSegment(motion(0.8), "a" * 64, 0.0, 5.0).sha256 != (
        ReferenceSegment(motion(0.8), "a" * 64, 0.0, 10.0).sha256
    )


def test_entry_phase_window_excludes_better_matching_terminal_pose():
    source = motion(0.8)
    source.root_position[:, 2] = torch.linspace(0.4, 0.8, source.frame_count)
    unrestricted = ReferenceSegment(source, "a" * 64, 0.0, 10.0)
    restricted = ReferenceSegment(source, "a" * 64, 0.0, 10.0, 0.5)
    pose = np.array([0.8] + [0.0] * 25)
    assert unrestricted.nearest_phase(pose)[0] > 9.0
    assert 0 <= restricted.nearest_phase(pose)[0] <= 0.5
    assert restricted.sha256 != unrestricted.sha256


def test_terminal_hold_preserves_last_pose_and_zeroes_velocity_without_wrap():
    source = motion(0.8)
    source.root_position[:, 2] = torch.linspace(0.4, 0.8, source.frame_count)
    source.root_velocity[:] = 1.0
    segment = ReferenceSegment(source, "a" * 64, 0.0, 10.0, boundary="hold_last_pose_zero_velocity")
    features = segment.features(torch.tensor([9.9, 10.0, 20.0]))
    assert features[1, 0] > 0.79
    np.testing.assert_array_equal(features[1].numpy(), features[2].numpy())
    np.testing.assert_array_equal(features[1:, 3:7].numpy(), np.zeros((2, 4)))
    assert segment.reported_phase(torch.tensor(20.0)) == 10.0


def test_invalid_entry_or_boundary_rejected():
    with pytest.raises(ValueError, match="entry phase"):
        ReferenceSegment(motion(0.8), "a" * 64, 0.0, 10.0, 10.0)
    with pytest.raises(ValueError, match="boundary"):
        ReferenceSegment(motion(0.8), "a" * 64, 0.0, 10.0, boundary="guess")


def test_entry_runs_once_then_only_the_declared_native_subwindow_repeats():
    source = motion(0.8)
    source.root_position[:, 2] = torch.arange(source.frame_count, dtype=torch.float32) / 30.0
    segment = ReferenceSegment(
        source,
        "a" * 64,
        2.7,
        4.86,
        entry_phase_end_seconds=0.15,
        boundary="entry_once_then_loop",
        loop_start_seconds=3.9,
        exit_at_loop_boundary=True,
    )
    phases = torch.tensor([0.0, 1.19, 2.159, 2.16, 2.26, 3.12], dtype=torch.float32)

    observed = segment.features(phases)
    expected_times = torch.tensor([2.7, 3.89, 4.859, 3.9, 4.0, 3.9], dtype=torch.float32)
    expected = source.features(expected_times)

    np.testing.assert_allclose(observed.numpy(), expected.numpy(), rtol=0, atol=1.0e-5)
    assert segment.reported_phase(torch.tensor(2.16)) == pytest.approx(1.2)
    assert segment.reported_phase(torch.tensor(3.12)) == pytest.approx(1.2)
    assert segment.identity["loop_start_seconds"] == 3.9
    assert segment.identity["exit_at_loop_boundary"] is True
    assert segment.sha256 != ReferenceSegment(source, "a" * 64, 2.7, 4.86, 0.15).sha256


def test_entry_loop_current_and_future_window_share_one_boundary_mapping():
    source = motion(0.8)
    source.root_position[:, 2] = torch.arange(source.frame_count, dtype=torch.float32) / 30.0
    segment = ReferenceSegment(
        source,
        "a" * 64,
        2.7,
        4.86,
        entry_phase_end_seconds=0.15,
        boundary="entry_once_then_loop",
        loop_start_seconds=3.9,
    )
    phase = torch.tensor(segment.duration - CONTROL_DT_SECONDS, dtype=torch.float32)
    offsets = torch.tensor(REFERENCE_OFFSETS, dtype=torch.float32) * CONTROL_DT_SECONDS

    current = segment.features(phase.reshape(1))[0]
    window = segment.features(phase + offsets)

    np.testing.assert_allclose(
        current.numpy(), source.features(torch.tensor([4.84]))[0].numpy(), rtol=0, atol=1.0e-5
    )
    np.testing.assert_allclose(
        window.numpy(),
        source.features(
            torch.tensor(
                [
                    3.9,
                    3.98,
                    4.08,
                    4.18,
                    4.28,
                    4.38,
                    4.48,
                    4.58,
                    4.68,
                    4.78,
                    3.92,
                    4.02,
                    4.12,
                    4.22,
                    4.32,
                    4.42,
                    4.52,
                    4.62,
                    4.72,
                    4.82,
                ]
            )
        ).numpy(),
        rtol=0,
        atol=1.0e-5,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"boundary": "entry_once_then_loop"},
        {"boundary": "entry_once_then_loop", "loop_start_seconds": 2.7},
        {"boundary": "entry_once_then_loop", "loop_start_seconds": 4.86},
        {
            "entry_phase_end_seconds": 1.3,
            "boundary": "entry_once_then_loop",
            "loop_start_seconds": 3.9,
        },
        {"boundary": "wrap_within_segment", "loop_start_seconds": 3.9},
    ],
)
def test_malformed_entry_loop_bounds_are_rejected(kwargs):
    with pytest.raises(ValueError, match="loop"):
        ReferenceSegment(motion(0.8), "a" * 64, 2.7, 4.86, **kwargs)


def test_spatial_guard_is_rechecked_only_at_each_entry_loop_boundary():
    program = oracle_program_from_dict(
        {
            "schema_version": 1,
            "evidence_class": "exploratory_oracle_cycle",
            "oracle_id": "boundary_fixture",
            "behaviors": ["crouch", "rise"],
            "initial": "inside",
            "states": {
                "inside": {"behavior": "crouch", "min_dwell": 0},
                "rise": {"behavior": "rise", "min_dwell": 1},
            },
            "transitions": [
                {
                    "from": "inside",
                    "to": "rise",
                    "priority": 0,
                    "guard": "x_travelled >= 1",
                }
            ],
        },
        available_behaviors=["crouch", "rise"],
    )
    oracle = ComposedReference(
        program,
        {
            "crouch": ReferenceSegment(
                motion(0.5),
                "a" * 64,
                0.0,
                0.2,
                entry_phase_end_seconds=0.02,
                boundary="entry_once_then_loop",
                loop_start_seconds=0.1,
                exit_at_loop_boundary=True,
            ),
            "rise": ReferenceSegment(motion(0.8), "b" * 64, 0.0, 1.0),
        },
    )

    def decide(step: int, x: float):
        return oracle.command(
            step=step,
            signals={
                "t": step * CONTROL_DT_SECONDS,
                "v_x": 0.5,
                "v_target": 0.5,
                "z_root": 0.5,
                "torso_up": 1.0,
                "x_travelled": x,
            },
            robot_pose=np.array([0.5] + [0.0] * 25),
        )

    for step in range(10):
        assert decide(step, 2.0).state == "inside"
    # The guard was true earlier, but is false at the first eligible boundary.
    assert decide(10, 0.5).state == "inside"
    for step in range(11, 15):
        assert decide(step, 2.0).state == "inside"
    switched = decide(15, 2.0)
    assert switched.state == "rise"
    assert switched.transition["control_step"] == 15


def test_loop_boundary_exit_requires_loop_semantics():
    with pytest.raises(ValueError, match="loop-boundary exit"):
        ReferenceSegment(
            motion(0.8),
            "a" * 64,
            0.0,
            10.0,
            exit_at_loop_boundary=True,
        )
