from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from oracle_composition.contracts import (
    GuardOperator,
    ModeKind,
    ModeSpec,
    OracleContractError,
    OracleProgram,
    PhaseTransferKind,
    PhaseTransferSpec,
    ReferenceArtifact,
    ReferenceSchema,
    ScalarGuard,
    SignalSpec,
    TransitionKind,
    TransitionRule,
)
from oracle_composition.runtime import (
    ModePhase,
    OracleRuntime,
    OracleRuntimeError,
)


def _reference(*, offset: float = 0.0) -> ReferenceArtifact:
    return ReferenceArtifact.create(
        artifact_id="humanoid/composed/v1",
        schema=ReferenceSchema(("q0", "q1"), ("rad", "rad"), "pelvis", 50.0),
        values=tuple((offset + float(index), offset - float(index)) for index in range(8)),
    )


def _program(reference: ReferenceArtifact) -> OracleProgram:
    return OracleProgram(
        program_id="oracle/manual/v1",
        reference_identity=reference.identity,
        entry_mode="walk",
        modes=(
            ModeSpec("walk", (0, 3), max_dwell_steps=6),
            ModeSpec("recover", (3, 6), ModeKind.RECOVERY, max_dwell_steps=6),
            ModeSpec("done", (6, 8), ModeKind.TERMINAL),
        ),
        transitions=(
            TransitionRule(
                "fall",
                "walk",
                "recover",
                ScalarGuard("health", GuardOperator.LE, 0.4),
                priority=0,
                phase_transfer=PhaseTransferSpec(),
                min_dwell_steps=1,
                kind=TransitionKind.RECOVERY,
            ),
            TransitionRule(
                "complete",
                "walk",
                "done",
                ScalarGuard("progress", GuardOperator.GE, 1.0),
                priority=10,
                phase_transfer=PhaseTransferSpec(),
                min_dwell_steps=1,
            ),
            TransitionRule(
                "stable",
                "recover",
                "walk",
                ScalarGuard("stability", GuardOperator.GE, 0.8),
                priority=0,
                phase_transfer=PhaseTransferSpec(
                    PhaseTransferKind.NORMALIZED_PHASE_FROM_SIGNAL,
                    "resume_phase",
                ),
                kind=TransitionKind.REJOIN,
            ),
        ),
        signal_specs=(
            SignalSpec("health", "1", "observable", "observation:health"),
            SignalSpec("progress", "1", "observable", "observation:progress"),
            SignalSpec("stability", "1", "observable", "observation:stability"),
            SignalSpec("resume_phase", "1", "observable", "observation:phase"),
        ),
        horizon_steps=4,
        max_queries=20,
    )


def _runtime() -> OracleRuntime:
    reference = _reference()
    return OracleRuntime(program=_program(reference), reference=reference)


def test_result_is_typed_and_window_is_finite_h_by_d() -> None:
    result = _runtime().query({"health": 1.0, "progress": 0.0})

    assert isinstance(result.mode_phase, ModePhase)
    assert result.mode_phase.mode_name == "walk"
    assert result.mode_phase.local_phase == 0.0
    assert result.reference_window.frame_indices == (0, 1, 2, 2)
    assert result.reference_window.horizon == 4
    assert result.reference_window.width == 2
    assert len(result.program_sha256) == 64


def test_numpy_scalar_signals_are_admitted_as_numeric() -> None:
    result = _runtime().query({"health": np.float32(1.0), "progress": np.float64(0.0)})
    assert result.mode_phase.mode_name == "walk"


def test_lower_numeric_priority_wins_when_guards_overlap() -> None:
    runtime = _runtime()
    runtime.query({"health": 0.2, "progress": 1.0})
    result = runtime.query({"health": 0.2, "progress": 1.0})

    assert result.transition is not None
    assert result.transition.rule_id == "fall"
    assert result.transition.kind is TransitionKind.RECOVERY
    assert result.mode_phase.mode_name == "recover"


def test_recovery_and_rejoin_are_explicit_runtime_events() -> None:
    runtime = _runtime()
    runtime.query({"health": 0.2, "progress": 0.0})
    recovery = runtime.query({"health": 0.2, "progress": 0.0})
    rejoin = runtime.query({"stability": 0.9, "resume_phase": 0.25})

    assert recovery.transition is not None
    assert recovery.transition.kind is TransitionKind.RECOVERY
    assert recovery.transition.to_mode == "recover"
    assert rejoin.transition is not None
    assert rejoin.transition.kind is TransitionKind.REJOIN
    assert rejoin.transition.to_mode == "walk"
    assert rejoin.transition.phase_transfer_kind is (PhaseTransferKind.NORMALIZED_PHASE_FROM_SIGNAL)
    assert rejoin.transition.phase_signal == "resume_phase"
    assert rejoin.transition.phase_signal_value == 0.25
    assert rejoin.transition.target_entry_frame_index == 1
    assert rejoin.transition.target_entry_local_phase == 0.5
    assert rejoin.mode_phase.frame_index == 1
    assert rejoin.mode_phase.dwell_steps == 0
    assert rejoin.mode_phase.reference_offset_frames == 1

    continued = runtime.query({"health": 1.0, "progress": 0.0})
    assert continued.mode_phase.frame_index == 2
    assert continued.mode_phase.dwell_steps == 1
    assert continued.mode_phase.reference_offset_frames == 2


def test_reset_to_start_transition_reports_resolved_target_phase() -> None:
    runtime = _runtime()
    runtime.query({"health": 0.2, "progress": 0.0})
    recovery = runtime.query({"health": 0.2, "progress": 0.0})

    assert recovery.transition is not None
    assert recovery.transition.phase_transfer_kind is PhaseTransferKind.RESET_TO_START
    assert recovery.transition.phase_signal is None
    assert recovery.transition.phase_signal_value is None
    assert recovery.transition.target_entry_frame_index == 3
    assert recovery.transition.target_entry_local_phase == 0.0
    assert recovery.mode_phase.frame_index == 3
    assert recovery.mode_phase.reference_offset_frames == 0


@pytest.mark.parametrize(
    "bad_phase",
    [-0.001, 1.001, float("nan"), float("inf"), True, "0.5"],
)
def test_normalized_rejoin_phase_fails_closed_and_is_atomic(bad_phase: object) -> None:
    runtime = _runtime()
    runtime.query({"health": 0.2, "progress": 0.0})
    runtime.query({"health": 0.2, "progress": 0.0})
    before = runtime.state

    with pytest.raises(OracleRuntimeError, match="resume_phase"):
        runtime.query({"stability": 0.9, "resume_phase": bad_phase})

    assert runtime.state == before


def test_normalized_rejoin_phase_signal_is_required_before_dispatch() -> None:
    runtime = _runtime()
    runtime.query({"health": 0.2, "progress": 0.0})
    runtime.query({"health": 0.2, "progress": 0.0})

    with pytest.raises(OracleRuntimeError, match="resume_phase"):
        runtime.query({"stability": 0.9})


@pytest.mark.parametrize(
    ("normalized_phase", "expected_frame"),
    [
        (0.0, 0),
        (0.2499, 0),
        (0.25, 1),
        (0.7499, 1),
        (0.75, 2),
        (1.0, 2),
    ],
)
def test_normalized_phase_uses_deterministic_nearest_frame_mapping(
    normalized_phase: float,
    expected_frame: int,
) -> None:
    runtime = _runtime()
    runtime.query({"health": 0.2, "progress": 0.0})
    runtime.query({"health": 0.2, "progress": 0.0})

    result = runtime.query({"stability": 0.9, "resume_phase": normalized_phase})

    assert result.mode_phase.frame_index == expected_frame


def test_hysteresis_preserves_guard_until_minimum_dwell() -> None:
    reference = _reference()
    program = OracleProgram(
        program_id="oracle/hysteresis/v1",
        reference_identity=reference.identity,
        entry_mode="wait",
        modes=(
            ModeSpec("wait", (0, 3), max_dwell_steps=4),
            ModeSpec("done", (3, 4), ModeKind.TERMINAL),
        ),
        transitions=(
            TransitionRule(
                "ready",
                "wait",
                "done",
                ScalarGuard("ready", "ge", 0.8, hysteresis=0.1),
                priority=0,
                phase_transfer=PhaseTransferSpec(),
                min_dwell_steps=2,
            ),
        ),
        signal_specs=(SignalSpec("ready", "1", "observable", "observation:ready"),),
        horizon_steps=2,
        max_queries=10,
    )
    runtime = OracleRuntime(program=program, reference=reference)

    assert runtime.query({"ready": 0.81}).transition is None
    assert runtime.query({"ready": 0.75}).transition is None
    result = runtime.query({"ready": 0.75})
    assert result.transition is not None
    assert result.mode_phase.mode_name == "done"


def test_hysteresis_releases_after_crossing_exit_threshold() -> None:
    reference = _reference()
    base = OracleProgram(
        program_id="oracle/hysteresis/v1",
        reference_identity=reference.identity,
        entry_mode="wait",
        modes=(
            ModeSpec("wait", (0, 3), max_dwell_steps=4),
            ModeSpec("done", (3, 4), ModeKind.TERMINAL),
        ),
        transitions=(
            TransitionRule(
                "ready",
                "wait",
                "done",
                ScalarGuard("ready", "ge", 0.8, hysteresis=0.1),
                priority=0,
                phase_transfer=PhaseTransferSpec(),
                min_dwell_steps=2,
            ),
        ),
        signal_specs=(SignalSpec("ready", "1", "observable", "observation:ready"),),
        horizon_steps=2,
        max_queries=10,
    )
    runtime = OracleRuntime(program=base, reference=reference)

    runtime.query({"ready": 0.81})
    runtime.query({"ready": 0.69})
    result = runtime.query({"ready": 0.75})
    assert result.transition is None
    assert result.mode_phase.mode_name == "wait"


def test_missing_lower_priority_signal_fails_before_short_circuit() -> None:
    with pytest.raises(OracleRuntimeError, match="progress"):
        _runtime().query({"health": 0.2})


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), True, "0.2"])
def test_runtime_signal_must_be_a_finite_numeric_scalar(bad: object) -> None:
    with pytest.raises(OracleRuntimeError, match="health"):
        _runtime().query({"health": bad, "progress": 0.0})


def test_stalled_mode_fails_closed_at_maximum_dwell() -> None:
    reference = _reference()
    program = OracleProgram(
        program_id="oracle/liveness/v1",
        reference_identity=reference.identity,
        entry_mode="wait",
        modes=(
            ModeSpec("wait", (0, 2), max_dwell_steps=1),
            ModeSpec("done", (2, 3), ModeKind.TERMINAL),
        ),
        transitions=(
            TransitionRule(
                "never",
                "wait",
                "done",
                ScalarGuard("ready", "ge", 1.0),
                priority=0,
                phase_transfer=PhaseTransferSpec(),
            ),
        ),
        signal_specs=(SignalSpec("ready", "1", "observable", "observation:ready"),),
        horizon_steps=2,
        max_queries=10,
    )
    runtime = OracleRuntime(program=program, reference=reference)

    runtime.query({"ready": 0.0})
    with pytest.raises(OracleRuntimeError, match="max_dwell_steps"):
        runtime.query({"ready": 0.0})


def test_program_cannot_run_against_same_name_but_different_content() -> None:
    original = _reference()
    changed = _reference(offset=0.01)

    with pytest.raises(OracleContractError, match="exact reference identity"):
        OracleRuntime(program=_program(original), reference=changed)


def test_query_budget_fails_closed_until_explicit_reset() -> None:
    reference = _reference()
    program = replace(_program(reference), max_queries=1)
    runtime = OracleRuntime(program=program, reference=reference)

    runtime.query({"health": 1.0, "progress": 0.0})
    with pytest.raises(OracleRuntimeError, match="max_queries"):
        runtime.query({"health": 1.0, "progress": 0.0})
    assert runtime.reset().query_index == 0
