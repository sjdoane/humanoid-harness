from __future__ import annotations

from dataclasses import replace

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
    SignalVisibility,
    TransitionKind,
    TransitionRule,
)


def _reference() -> ReferenceArtifact:
    return ReferenceArtifact.create(
        artifact_id="humanoid/composed/v1",
        schema=ReferenceSchema(("q0", "q1"), ("rad", "rad"), "pelvis", 50.0),
        values=tuple((float(index), float(-index)) for index in range(8)),
    )


def _modes() -> tuple[ModeSpec, ...]:
    return (
        ModeSpec("walk", (0, 3), max_dwell_steps=6),
        ModeSpec("recover", (3, 6), ModeKind.RECOVERY, max_dwell_steps=6),
        ModeSpec("done", (6, 8), ModeKind.TERMINAL),
    )


def _signal_specs() -> tuple[SignalSpec, ...]:
    return (
        SignalSpec("health", "1", SignalVisibility.OBSERVABLE, "observation:health"),
        SignalSpec("progress", "1", "observable", "observation:progress"),
        SignalSpec("stability", "1", "privileged", "derived:stability", 1),
        SignalSpec("resume_phase", "1", "observable", "observation:phase"),
    )


def _rules() -> tuple[TransitionRule, ...]:
    return (
        TransitionRule(
            "fall",
            "walk",
            "recover",
            ScalarGuard("health", GuardOperator.LE, 0.4, hysteresis=0.1),
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
            ScalarGuard("stability", GuardOperator.GE, 0.8, hysteresis=0.05),
            priority=0,
            phase_transfer=PhaseTransferSpec(
                PhaseTransferKind.NORMALIZED_PHASE_FROM_SIGNAL,
                "resume_phase",
            ),
            kind=TransitionKind.REJOIN,
        ),
    )


def _program(
    *,
    modes: tuple[ModeSpec, ...] | None = None,
    rules: tuple[TransitionRule, ...] | None = None,
) -> OracleProgram:
    reference = _reference()
    return OracleProgram(
        program_id="oracle/manual/v1",
        reference_identity=reference.identity,
        entry_mode="walk",
        modes=_modes() if modes is None else modes,
        transitions=_rules() if rules is None else rules,
        signal_specs=_signal_specs(),
        horizon_steps=3,
        max_queries=100,
    )


def test_valid_program_has_stable_content_hash_and_sorted_outgoing_rules() -> None:
    program = _program()

    assert len(program.sha256) == 64
    assert [rule.rule_id for rule in program.outgoing("walk")] == [
        "fall",
        "complete",
    ]
    assert program.to_dict()["signal_specs"] == [
        {
            "name": "health",
            "unit": "1",
            "visibility": "observable",
            "source": "observation:health",
            "latency_steps": 0,
        },
        {
            "name": "progress",
            "unit": "1",
            "visibility": "observable",
            "source": "observation:progress",
            "latency_steps": 0,
        },
        {
            "name": "stability",
            "unit": "1",
            "visibility": "privileged",
            "source": "derived:stability",
            "latency_steps": 1,
        },
        {
            "name": "resume_phase",
            "unit": "1",
            "visibility": "observable",
            "source": "observation:phase",
            "latency_steps": 0,
        },
    ]
    assert program.to_dict()["transitions"][0]["phase_transfer"] == {
        "kind": "reset_to_start",
        "signal": None,
    }


def test_signal_contract_metadata_changes_program_content_identity() -> None:
    program = _program()
    delayed = replace(
        program,
        signal_specs=tuple(
            replace(signal, latency_steps=2) if signal.name == "health" else signal
            for signal in program.signal_specs
        ),
    )

    assert delayed.sha256 != program.sha256


def test_guard_must_bind_to_a_declared_signal() -> None:
    with pytest.raises(OracleContractError, match="guard uses undeclared signal"):
        replace(
            _program(),
            signal_specs=tuple(signal for signal in _signal_specs() if signal.name != "health"),
        )


def test_phase_transfer_must_bind_to_a_declared_dimensionless_signal() -> None:
    with pytest.raises(OracleContractError, match="phase transfer uses undeclared signal"):
        replace(
            _program(),
            signal_specs=tuple(
                signal for signal in _signal_specs() if signal.name != "resume_phase"
            ),
        )

    with pytest.raises(OracleContractError, match="dimensionless unit"):
        replace(
            _program(),
            signal_specs=tuple(
                replace(signal, unit="rad") if signal.name == "resume_phase" else signal
                for signal in _signal_specs()
            ),
        )


def test_duplicate_signal_specs_are_rejected() -> None:
    with pytest.raises(OracleContractError, match="signal spec names must be unique"):
        replace(_program(), signal_specs=(*_signal_specs(), _signal_specs()[0]))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"unit": ""},
        {"visibility": "sometimes"},
        {"source": "source with spaces"},
        {"latency_steps": -1},
        {"latency_steps": True},
    ],
)
def test_signal_spec_validation_fails_closed(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "name": "health",
        "unit": "1",
        "visibility": "observable",
        "source": "observation:health",
    }
    values.update(kwargs)
    with pytest.raises(OracleContractError):
        SignalSpec(**values)  # type: ignore[arg-type]


def test_phase_transfer_shape_is_typed_and_explicit() -> None:
    with pytest.raises(OracleContractError, match="cannot declare a signal"):
        PhaseTransferSpec("reset_to_start", "phase")
    with pytest.raises(OracleContractError, match="requires a stable signal"):
        PhaseTransferSpec("normalized_phase_from_signal")


def test_horizon_is_resource_bounded() -> None:
    with pytest.raises(OracleContractError, match="horizon_steps"):
        replace(_program(), horizon_steps=257)


def test_nonterminal_mode_requires_bounded_dwell() -> None:
    modes = (
        replace(_modes()[0], max_dwell_steps=None),
        *_modes()[1:],
    )
    with pytest.raises(OracleContractError, match="fail-closed liveness"):
        _program(modes=modes)


def test_entry_mode_must_be_a_task_mode() -> None:
    with pytest.raises(OracleContractError, match="entry_mode must be a task"):
        replace(_program(), entry_mode="recover")


def test_duplicate_outgoing_priority_is_rejected() -> None:
    rules = (
        *_rules(),
        TransitionRule(
            "duplicate-priority",
            "walk",
            "done",
            ScalarGuard("alternate", "ge", 1.0),
            priority=0,
            phase_transfer=PhaseTransferSpec(),
        ),
    )
    with pytest.raises(OracleContractError, match="priorities must be unique"):
        _program(rules=rules)


def test_edge_into_recovery_must_be_explicitly_typed() -> None:
    rules = (replace(_rules()[0], kind=TransitionKind.ADVANCE), *_rules()[1:])
    with pytest.raises(OracleContractError, match="explicit recovery"):
        _program(rules=rules)


def test_recovery_mode_requires_explicit_rejoin() -> None:
    with pytest.raises(OracleContractError, match="explicit rejoin"):
        _program(rules=_rules()[:2])


def test_rejoin_must_return_to_a_task_mode() -> None:
    rules = (*_rules()[:2], replace(_rules()[2], to_mode="done"))
    with pytest.raises(OracleContractError, match="rejoin must target a task"):
        _program(rules=rules)


def test_terminal_mode_cannot_have_outgoing_edges() -> None:
    rules = (
        *_rules(),
        TransitionRule(
            "leave-terminal",
            "done",
            "walk",
            ScalarGuard("restart", "ge", 1.0),
            priority=0,
            phase_transfer=PhaseTransferSpec(),
        ),
    )
    with pytest.raises(OracleContractError, match="cannot have outgoing"):
        _program(rules=rules)


def test_every_mode_must_have_a_topological_path_to_terminal() -> None:
    modes = (
        ModeSpec("a", (0, 2), max_dwell_steps=3),
        ModeSpec("b", (2, 4), max_dwell_steps=3),
        ModeSpec("done", (4, 5), ModeKind.TERMINAL),
    )
    rules = (
        TransitionRule(
            "a-to-b",
            "a",
            "b",
            ScalarGuard("x", "ge", 1.0),
            priority=0,
            phase_transfer=PhaseTransferSpec(),
        ),
        TransitionRule(
            "b-to-a",
            "b",
            "a",
            ScalarGuard("x", "lt", 1.0),
            priority=0,
            phase_transfer=PhaseTransferSpec(),
        ),
    )
    with pytest.raises(OracleContractError, match="path to a terminal"):
        _program(modes=modes, rules=rules)


def test_zero_dwell_cycle_is_rejected() -> None:
    modes = (
        ModeSpec("a", (0, 2), max_dwell_steps=3),
        ModeSpec("b", (2, 4), max_dwell_steps=3),
        ModeSpec("done", (4, 5), ModeKind.TERMINAL),
    )
    rules = (
        TransitionRule(
            "a-to-b",
            "a",
            "b",
            ScalarGuard("x", "ge", 1.0),
            priority=0,
            phase_transfer=PhaseTransferSpec(),
        ),
        TransitionRule(
            "b-to-a",
            "b",
            "a",
            ScalarGuard("x", "lt", 1.0),
            priority=0,
            phase_transfer=PhaseTransferSpec(),
        ),
        TransitionRule(
            "b-to-done",
            "b",
            "done",
            ScalarGuard("finished", "ge", 1.0),
            priority=1,
            phase_transfer=PhaseTransferSpec(),
        ),
    )
    with pytest.raises(OracleContractError, match="zero-dwell transition cycle"):
        _program(modes=modes, rules=rules)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"operator": "equals"},
        {"operator": "ge", "hysteresis": -0.1},
        {"operator": "ge", "threshold": float("nan")},
    ],
)
def test_guard_validation_fails_closed(kwargs: dict[str, object]) -> None:
    values = {"signal": "health", "operator": "ge", "threshold": 0.5}
    values.update(kwargs)
    with pytest.raises(OracleContractError):
        ScalarGuard(**values)  # type: ignore[arg-type]


def test_mode_span_cannot_exceed_bound_reference() -> None:
    modes = (replace(_modes()[0], reference_span=(0, 99)), *_modes()[1:])
    with pytest.raises(OracleContractError, match="exceeds bound reference"):
        _program(modes=modes)
