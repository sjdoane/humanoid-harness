from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from oracle_composition.phase_b.contracts import PHASE_POLICY, PhaseBOracleProgram
from oracle_composition.phase_b.receipts import ADMITTED_TRAINING_BLOCKS
from oracle_composition.phase_b.reference_runtime import (
    ComposedReferenceRuntime,
    select_nearest_phase,
    tracking_state_from_reference_row,
)

ROOT = Path(__file__).resolve().parents[2]


def _rows(offset: float = 0.0, *, count: int = 12) -> np.ndarray:
    value = np.zeros((count, 45), dtype="<f8")
    value[:, 0] = 1.4
    value[:, 1] = 1.0
    for index in range(count):
        value[index, 5] = offset + index
        value[index, 11:28] = (offset + index) / 10.0
    return value


def _program(*, same_behavior: bool = False) -> PhaseBOracleProgram:
    target_behavior = "expert" if same_behavior else "medium"
    raw = {
        "behaviors": ["expert", "medium"],
        "evidence_class": "interface_check",
        "initial": "start",
        "oracle_id": "phase_test",
        "oracle_schema_id": "humanoid_reference_composition_oracle/v1",
        "phase_policy": dict(PHASE_POLICY),
        "schema_version": 1,
        "states": {
            "finish": {"behavior": target_behavior, "min_dwell": 1},
            "start": {"behavior": "expert", "min_dwell": 1},
        },
        "transitions": [{"from": "start", "guard": "t >= 1", "priority": 0, "to": "finish"}],
    }
    return PhaseBOracleProgram.from_dict(raw, available_behaviors=("expert", "medium"))


def _signals(t: int, dwell: int) -> dict[str, object]:
    return {
        "dwell": dwell,
        "t": t,
        "torso_up": 1.0,
        "v_target": 3.0,
        "v_x": 0.0,
        "x_travelled": 0.0,
        "z_root": 1.4,
    }


def test_nearest_phase_selected_over_same_index_and_not_rematched_on_hold() -> None:
    expert = _rows()
    medium = _rows(100.0)
    medium[0] = expert[1]
    runtime = ComposedReferenceRuntime(
        _program(),
        {"expert": expert, "medium": medium},
    )
    initial = runtime.frame(
        state=tracking_state_from_reference_row(expert[0]),
        signals=_signals(0, 0),
    )
    assert initial.phase == 0
    runtime.advance()
    switched = runtime.frame(
        state=tracking_state_from_reference_row(expert[1]),
        signals=_signals(1, 1),
    )
    assert switched.transfer is not None
    assert switched.phase == 0
    assert switched.phase != 1
    assert switched.transfer.candidate_range == (0, 1)
    assert len(switched.transfer.candidates) == 2
    assert len(switched.transfer.selected_normalized_errors) == 6
    assert switched.transfer.target_window_indices == switched.policy_window_indices
    assert switched.hidden_reward_target_sha256
    runtime.advance()
    held = runtime.frame(
        state=tracking_state_from_reference_row(medium[0]),
        signals=_signals(2, 1),
    )
    assert held.phase == 1
    assert held.transfer is None
    assert len(runtime.transfer_logs) == 1


def test_state_transition_without_behavior_change_does_not_rematch() -> None:
    rows = _rows()
    runtime = ComposedReferenceRuntime(
        _program(same_behavior=True),
        {"expert": rows, "medium": _rows(20.0)},
    )
    runtime.frame(state=tracking_state_from_reference_row(rows[0]), signals=_signals(0, 0))
    runtime.advance()
    frame = runtime.frame(
        state=tracking_state_from_reference_row(rows[1]),
        signals=_signals(1, 1),
    )
    assert frame.phase == 1
    assert frame.transfer is None


def test_runtime_copies_references_and_emits_read_only_targets() -> None:
    expert = _rows()
    runtime = ComposedReferenceRuntime(
        _program(),
        {"expert": expert, "medium": _rows(20.0)},
    )
    expert[0, 0] = 99.0
    frame = runtime.frame(
        state=tracking_state_from_reference_row(_rows()[0]),
        signals=_signals(0, 0),
    )
    assert frame.policy_window[0, 0] == 1.4
    with np.testing.assert_raises(ValueError):
        frame.policy_window[0, 0] = 99.0
    with np.testing.assert_raises(ValueError):
        frame.hidden_reward_target[0] = 99.0


def test_selection_is_bounded_by_task_step() -> None:
    target = _rows()
    state = tracking_state_from_reference_row(target[8])
    decision = select_nearest_phase(
        state=state,
        target_rows=target,
        task_step=3,
        source_behavior="expert",
        target_behavior="medium",
        reason="test",
    )
    assert decision.selected_phase <= 3
    assert decision.candidate_range == (0, 3)


def test_no_wrap_and_terminal_hold_only_at_reference_end() -> None:
    expert = _rows(count=8)
    raw = {
        "behaviors": ["expert"],
        "evidence_class": "interface_check",
        "initial": "hold",
        "oracle_id": "hold_test",
        "oracle_schema_id": "humanoid_reference_composition_oracle/v1",
        "phase_policy": dict(PHASE_POLICY),
        "schema_version": 1,
        "states": {"hold": {"behavior": "expert", "min_dwell": 1}},
        "transitions": [],
    }
    program = PhaseBOracleProgram.from_dict(raw, available_behaviors=("expert",))
    runtime = ComposedReferenceRuntime(program, {"expert": expert})
    for step in range(10):
        frame = runtime.frame(
            state=tracking_state_from_reference_row(expert[min(step, 7)]),
            signals=_signals(step, step),
        )
        assert frame.terminal_hold is (step >= 7)
        assert frame.phase == min(step, 7)
        assert frame.policy_window_indices[-1] == 7
        runtime.advance()
    assert runtime.phase == 7


def test_real_static_receipt_matches_the_preregistered_ranges() -> None:
    path = (
        ROOT
        / "experiments/003_composition_speed_profile/phase_b/receipts/phase_transfer_static_v1.json"
    )
    value = json.loads(path.read_bytes())
    assert value["admitted_training_blocks"] == list(ADMITTED_TRAINING_BLOCKS)
    assert value["summary"]["expert_to_medium"]["selected_phase_range"] == [45, 281]
    assert value["summary"]["medium_after_300_to_expert"]["selected_phase_range"] == [
        23,
        427,
    ]
