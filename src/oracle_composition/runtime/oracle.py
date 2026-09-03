"""Deterministic data-only runtime for a validated oracle program.

This runtime chooses phase/reference commands. Normalized phase is a declared
caller-supplied signal; the runtime does not estimate or learn phase. It does
not implement or imply a controller, tracker, reward, dynamics-feasibility
certificate, or successful robot behavior.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

from oracle_composition.contracts import (
    OracleContractError,
    OracleProgram,
    PhaseTransferKind,
    ReferenceArtifact,
    ReferenceWindow,
    TransitionKind,
    TransitionRule,
)


class OracleRuntimeError(RuntimeError):
    """Raised when runtime inputs or liveness fail closed."""


@dataclass(frozen=True)
class RuntimeState:
    """Next-query state with liveness dwell separate from reference offset."""

    mode_name: str
    dwell_steps: int
    reference_offset_frames: int
    query_index: int


@dataclass(frozen=True)
class ModePhase:
    """Typed mode/phase authority emitted with every reference window."""

    mode_name: str
    mode_index: int
    mode_kind: str
    local_phase: float
    frame_index: int
    dwell_steps: int
    reference_offset_frames: int


@dataclass(frozen=True)
class TransitionEvent:
    rule_id: str
    kind: TransitionKind
    from_mode: str
    to_mode: str
    priority: int
    pre_transition_dwell_steps: int
    phase_transfer_kind: PhaseTransferKind
    phase_signal: str | None
    phase_signal_value: float | None
    target_entry_frame_index: int
    target_entry_local_phase: float


@dataclass(frozen=True)
class OracleResult:
    """One immutable oracle decision and finite ``H x D`` window."""

    mode_phase: ModePhase
    reference_window: ReferenceWindow
    transition: TransitionEvent | None
    query_index: int
    program_sha256: str
    reference_sha256: str


def _runtime_scalar(value: object, *, signal: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise OracleRuntimeError(f"oracle signal {signal!r} must be a numeric scalar")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise OracleRuntimeError(f"oracle signal {signal!r} must be finite")
    return resolved


class OracleRuntime:
    """Episode-local executor; construct one instance per environment."""

    def __init__(
        self,
        *,
        program: OracleProgram,
        reference: ReferenceArtifact,
    ) -> None:
        reference.verify()
        if reference.identity != program.reference_identity:
            raise OracleContractError(
                "oracle program is not bound to this exact reference identity"
            )
        self._program = program
        self._reference = reference
        self._mode_index = {mode.name: index for index, mode in enumerate(program.modes)}
        self._guard_active = {rule.rule_id: False for rule in program.transitions}
        self._state = RuntimeState(
            mode_name=program.entry_mode,
            dwell_steps=0,
            reference_offset_frames=0,
            query_index=0,
        )

    @property
    def state(self) -> RuntimeState:
        return self._state

    @property
    def program(self) -> OracleProgram:
        return self._program

    def reset(self) -> RuntimeState:
        """Start a new episode at the validated entry mode."""

        self._guard_active = {rule.rule_id: False for rule in self.program.transitions}
        self._state = RuntimeState(
            mode_name=self.program.entry_mode,
            dwell_steps=0,
            reference_offset_frames=0,
            query_index=0,
        )
        return self._state

    def _validate_signals(
        self,
        outgoing: tuple[TransitionRule, ...],
        signals: Mapping[str, object],
    ) -> dict[str, float]:
        if not isinstance(signals, Mapping):
            raise OracleRuntimeError("oracle signals must be a mapping")
        required = sorted(
            {
                signal
                for rule in outgoing
                for signal in (rule.guard.signal, rule.phase_transfer.signal)
                if signal is not None
            }
        )
        missing = [name for name in required if name not in signals]
        if missing:
            raise OracleRuntimeError(f"oracle signals are missing required values: {missing!r}")
        observed = {name: _runtime_scalar(signals[name], signal=name) for name in required}
        normalized_phase_signals = {
            rule.phase_transfer.signal
            for rule in outgoing
            if rule.phase_transfer.kind is PhaseTransferKind.NORMALIZED_PHASE_FROM_SIGNAL
        }
        for name in normalized_phase_signals:
            if name is None:  # rejected by the immutable program contract
                raise OracleRuntimeError("normalized phase transfer has no bound signal")
            if not 0.0 <= observed[name] <= 1.0:
                raise OracleRuntimeError(f"normalized phase signal {name!r} must be in [0, 1]")
        return observed

    @staticmethod
    def _phase_transfer_offset(
        *,
        rule: TransitionRule,
        target_n_frames: int,
        observed: Mapping[str, float],
    ) -> tuple[int, float | None]:
        transfer = rule.phase_transfer
        if transfer.kind is PhaseTransferKind.RESET_TO_START:
            return 0, None
        if transfer.signal is None:  # rejected by the immutable program contract
            raise OracleRuntimeError("normalized phase transfer has no bound signal")
        normalized_phase = observed[transfer.signal]
        if target_n_frames == 1:
            return 0, normalized_phase
        # Nearest-frame mapping with explicit half-up behavior (not Python's
        # tie-to-even round), clamped only for floating-point endpoint safety.
        offset = math.floor(normalized_phase * (target_n_frames - 1) + 0.5)
        return min(max(offset, 0), target_n_frames - 1), normalized_phase

    def query(self, signals: Mapping[str, object]) -> OracleResult:
        """Evaluate one deterministic transition and emit one window.

        Every signal needed by every outgoing rule is checked before priority
        short-circuiting.  This prevents a high-priority match from hiding a
        broken lower-priority observation contract. Declared normalized-phase
        inputs are likewise checked for finiteness and membership in ``[0, 1]``
        before dispatch.
        """

        if self.state.query_index >= self.program.max_queries:
            raise OracleRuntimeError(f"oracle exceeded max_queries={self.program.max_queries}")
        current = self.program.mode(self.state.mode_name)
        outgoing = self.program.outgoing(current.name)
        observed = self._validate_signals(outgoing, signals)

        matching: list[TransitionRule] = []
        next_guard_active = dict(self._guard_active)
        for rule in outgoing:
            active = rule.guard.evaluate(
                observed[rule.guard.signal],
                active=self._guard_active[rule.rule_id],
            )
            next_guard_active[rule.rule_id] = active
            if active and self.state.dwell_steps >= rule.min_dwell_steps:
                matching.append(rule)

        transition: TransitionEvent | None = None
        dwell = self.state.dwell_steps
        reference_offset = self.state.reference_offset_frames
        if matching:
            chosen = matching[0]
            target = self.program.mode(chosen.to_mode)
            reference_offset, phase_signal_value = self._phase_transfer_offset(
                rule=chosen,
                target_n_frames=target.n_frames,
                observed=observed,
            )
            target_start, _ = target.reference_span
            target_entry_frame = target_start + reference_offset
            if target.n_frames == 1:
                target_entry_phase = 0.0
            else:
                target_entry_phase = reference_offset / (target.n_frames - 1)
            transition = TransitionEvent(
                rule_id=chosen.rule_id,
                kind=chosen.kind,
                from_mode=chosen.from_mode,
                to_mode=chosen.to_mode,
                priority=chosen.priority,
                pre_transition_dwell_steps=dwell,
                phase_transfer_kind=chosen.phase_transfer.kind,
                phase_signal=chosen.phase_transfer.signal,
                phase_signal_value=phase_signal_value,
                target_entry_frame_index=target_entry_frame,
                target_entry_local_phase=float(target_entry_phase),
            )
            for rule in outgoing:
                next_guard_active[rule.rule_id] = False
            current = target
            dwell = 0
        elif current.max_dwell_steps is not None and dwell >= current.max_dwell_steps:
            raise OracleRuntimeError(
                f"mode {current.name!r} exceeded max_dwell_steps="
                f"{current.max_dwell_steps} without an eligible transition"
            )

        start, stop = current.reference_span
        frame_index = min(start + reference_offset, stop - 1)
        window = self._reference.window(
            start_frame=frame_index,
            horizon_steps=self.program.horizon_steps,
            end_exclusive=stop,
        )
        if current.n_frames == 1:
            local_phase = 0.0
        else:
            local_phase = (frame_index - start) / (current.n_frames - 1)
        phase = ModePhase(
            mode_name=current.name,
            mode_index=self._mode_index[current.name],
            mode_kind=current.kind.value,
            local_phase=float(local_phase),
            frame_index=frame_index,
            dwell_steps=dwell,
            reference_offset_frames=reference_offset,
        )
        result = OracleResult(
            mode_phase=phase,
            reference_window=window,
            transition=transition,
            query_index=self.state.query_index,
            program_sha256=self.program.sha256,
            reference_sha256=self._reference.identity.content_sha256,
        )
        self._guard_active = next_guard_active
        self._state = RuntimeState(
            mode_name=current.name,
            dwell_steps=dwell + 1,
            reference_offset_frames=min(reference_offset + 1, current.n_frames - 1),
            query_index=self.state.query_index + 1,
        )
        return result
