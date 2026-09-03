"""Validated finite-state oracle-composition program."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, StrEnum
from numbers import Real

from .errors import OracleContractError
from .reference import MAX_HORIZON_STEPS, ReferenceIdentity

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_SIGNAL = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,127}$")
MAX_QUERIES = 10_000_000
MAX_SIGNAL_LATENCY_STEPS = 1_000_000


class ModeKind(StrEnum):
    TASK = "task"
    RECOVERY = "recovery"
    TERMINAL = "terminal"


class GuardOperator(StrEnum):
    GE = "ge"
    GT = "gt"
    LE = "le"
    LT = "lt"


class SignalVisibility(StrEnum):
    """Whether a signal is available to the policy or only to the oracle."""

    OBSERVABLE = "observable"
    PRIVILEGED = "privileged"


class TransitionKind(StrEnum):
    ADVANCE = "advance"
    RECOVERY = "recovery"
    REJOIN = "rejoin"


class PhaseTransferKind(StrEnum):
    """How a transition selects its first frame in the target mode."""

    RESET_TO_START = "reset_to_start"
    NORMALIZED_PHASE_FROM_SIGNAL = "normalized_phase_from_signal"


def _enum(value: object, enum_type: type[Enum], *, field: str) -> Enum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = [item.value for item in enum_type]
        raise OracleContractError(f"{field} must be one of {choices!r}") from exc


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise OracleContractError(f"{field} must be a numeric scalar")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise OracleContractError(f"{field} must be finite")
    return resolved


@dataclass(frozen=True)
class SignalSpec:
    """One scalar runtime input with explicit interface provenance.

    ``unit`` uses a stable machine-readable token (``"1"`` for a
    dimensionless value). ``source`` names the producing observation or
    deterministic derived feature. ``latency_steps`` is the declared age of
    that value at the oracle query boundary; the data-only runtime binds this
    declaration into the program hash but cannot independently measure it.
    """

    name: str
    unit: str
    visibility: SignalVisibility
    source: str
    latency_steps: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _SIGNAL.fullmatch(self.name):
            raise OracleContractError("signal name must be a stable identifier")
        if not isinstance(self.unit, str) or not _IDENTIFIER.fullmatch(self.unit):
            raise OracleContractError("signal unit must be a stable identifier")
        object.__setattr__(
            self,
            "visibility",
            _enum(self.visibility, SignalVisibility, field="signal visibility"),
        )
        if not isinstance(self.source, str) or not _IDENTIFIER.fullmatch(self.source):
            raise OracleContractError("signal source must be a stable identifier")
        if (
            not isinstance(self.latency_steps, int)
            or isinstance(self.latency_steps, bool)
            or not (0 <= self.latency_steps <= MAX_SIGNAL_LATENCY_STEPS)
        ):
            raise OracleContractError(
                f"signal latency_steps must be in [0, {MAX_SIGNAL_LATENCY_STEPS}]"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "unit": self.unit,
            "visibility": self.visibility.value,
            "source": self.source,
            "latency_steps": self.latency_steps,
        }


@dataclass(frozen=True)
class PhaseTransferSpec:
    """Content-addressed target-phase selection for one transition.

    Reset semantics select the target span's first frame. Normalized-phase
    semantics require a declared, dimensionless scalar signal in ``[0, 1]``;
    the runtime maps it to the nearest target frame with deterministic
    half-up rounding.
    """

    kind: PhaseTransferKind = PhaseTransferKind.RESET_TO_START
    signal: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "kind",
            _enum(self.kind, PhaseTransferKind, field="phase transfer kind"),
        )
        if self.kind is PhaseTransferKind.RESET_TO_START:
            if self.signal is not None:
                raise OracleContractError("reset-to-start phase transfer cannot declare a signal")
            return
        if not isinstance(self.signal, str) or not _SIGNAL.fullmatch(self.signal):
            raise OracleContractError(
                "normalized phase transfer requires a stable signal identifier"
            )

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.kind.value, "signal": self.signal}


@dataclass(frozen=True)
class ScalarGuard:
    """A safe scalar threshold with Schmitt-trigger hysteresis.

    Once active, a ``ge``/``gt`` guard remains active until the value crosses
    ``threshold - hysteresis``; ``le``/``lt`` uses
    ``threshold + hysteresis``.  This lets a condition persist through a
    minimum-dwell period without executing arbitrary expressions.
    """

    signal: str
    operator: GuardOperator
    threshold: float
    hysteresis: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.signal, str) or not _SIGNAL.fullmatch(self.signal):
            raise OracleContractError("guard signal must be a stable identifier")
        object.__setattr__(
            self,
            "operator",
            _enum(self.operator, GuardOperator, field="guard operator"),
        )
        object.__setattr__(
            self,
            "threshold",
            _finite(self.threshold, field="guard threshold"),
        )
        object.__setattr__(
            self,
            "hysteresis",
            _finite(self.hysteresis, field="guard hysteresis"),
        )
        if self.hysteresis < 0.0:
            raise OracleContractError("guard hysteresis cannot be negative")

    def evaluate(self, value: float, *, active: bool) -> bool:
        observed = _finite(value, field=f"signal {self.signal!r}")
        threshold = self.threshold
        if active:
            if self.operator in (GuardOperator.GE, GuardOperator.GT):
                threshold -= self.hysteresis
            else:
                threshold += self.hysteresis
        if self.operator is GuardOperator.GE:
            return observed >= threshold
        if self.operator is GuardOperator.GT:
            return observed > threshold
        if self.operator is GuardOperator.LE:
            return observed <= threshold
        return observed < threshold

    def to_dict(self) -> dict[str, object]:
        return {
            "signal": self.signal,
            "operator": self.operator.value,
            "threshold": self.threshold,
            "hysteresis": self.hysteresis,
        }


@dataclass(frozen=True)
class ModeSpec:
    """One reference span and its bounded runtime dwell."""

    name: str
    reference_span: tuple[int, int]
    kind: ModeKind = ModeKind.TASK
    max_dwell_steps: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _IDENTIFIER.fullmatch(self.name):
            raise OracleContractError("mode name must be a stable identifier")
        if (
            not isinstance(self.reference_span, Sequence)
            or isinstance(self.reference_span, (str, bytes))
            or len(self.reference_span) != 2
        ):
            raise OracleContractError("reference_span must contain [start, end)")
        span = tuple(self.reference_span)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in span):
            raise OracleContractError("reference_span bounds must be integers")
        object.__setattr__(self, "reference_span", span)
        object.__setattr__(self, "kind", _enum(self.kind, ModeKind, field="mode kind"))
        if span[0] < 0 or span[1] <= span[0]:
            raise OracleContractError("reference_span must be ordered and non-empty")
        if self.max_dwell_steps is not None and (
            not isinstance(self.max_dwell_steps, int)
            or isinstance(self.max_dwell_steps, bool)
            or self.max_dwell_steps < 1
        ):
            raise OracleContractError("max_dwell_steps must be a positive integer")

    @property
    def n_frames(self) -> int:
        return self.reference_span[1] - self.reference_span[0]

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "reference_span": list(self.reference_span),
            "kind": self.kind.value,
            "max_dwell_steps": self.max_dwell_steps,
        }


@dataclass(frozen=True)
class TransitionRule:
    """One deterministic edge; lower priority wins within a source mode."""

    rule_id: str
    from_mode: str
    to_mode: str
    guard: ScalarGuard
    priority: int
    phase_transfer: PhaseTransferSpec
    min_dwell_steps: int = 0
    kind: TransitionKind = TransitionKind.ADVANCE

    def __post_init__(self) -> None:
        for attribute in ("rule_id", "from_mode", "to_mode"):
            value = getattr(self, attribute)
            if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
                raise OracleContractError(f"{attribute} must be a stable identifier")
        if not isinstance(self.guard, ScalarGuard):
            raise OracleContractError("transition guard must be a ScalarGuard")
        for attribute in ("priority", "min_dwell_steps"):
            value = getattr(self, attribute)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise OracleContractError(f"{attribute} must be an integer >= 0")
        object.__setattr__(
            self,
            "kind",
            _enum(self.kind, TransitionKind, field="transition kind"),
        )
        if not isinstance(self.phase_transfer, PhaseTransferSpec):
            raise OracleContractError("transition phase_transfer must be a PhaseTransferSpec")

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "from_mode": self.from_mode,
            "to_mode": self.to_mode,
            "guard": self.guard.to_dict(),
            "priority": self.priority,
            "min_dwell_steps": self.min_dwell_steps,
            "kind": self.kind.value,
            "phase_transfer": self.phase_transfer.to_dict(),
        }


def _path_reachable(
    start: str,
    adjacency: dict[str, set[str]],
) -> set[str]:
    seen = {start}
    frontier = [start]
    while frontier:
        current = frontier.pop()
        for target in adjacency.get(current, set()):
            if target not in seen:
                seen.add(target)
                frontier.append(target)
    return seen


def _has_directed_cycle(adjacency: dict[str, set[str]]) -> bool:
    visited: set[str] = set()
    active: set[str] = set()

    def visit(node: str) -> bool:
        if node in active:
            return True
        if node in visited:
            return False
        visited.add(node)
        active.add(node)
        if any(visit(target) for target in adjacency.get(node, set())):
            return True
        active.remove(node)
        return False

    return any(visit(node) for node in adjacency)


@dataclass(frozen=True)
class OracleProgram:
    """An immutable program that is valid only for one reference identity."""

    program_id: str
    reference_identity: ReferenceIdentity
    entry_mode: str
    modes: tuple[ModeSpec, ...]
    transitions: tuple[TransitionRule, ...]
    signal_specs: tuple[SignalSpec, ...]
    horizon_steps: int
    max_queries: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "modes", tuple(self.modes))
        object.__setattr__(self, "transitions", tuple(self.transitions))
        object.__setattr__(self, "signal_specs", tuple(self.signal_specs))
        errors = validate_program(self)
        if errors:
            raise OracleContractError("invalid oracle program:\n  - " + "\n  - ".join(errors))

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def mode(self, name: str) -> ModeSpec:
        for mode in self.modes:
            if mode.name == name:
                return mode
        raise OracleContractError(f"unknown mode {name!r}")

    def outgoing(self, name: str) -> tuple[TransitionRule, ...]:
        return tuple(
            sorted(
                (rule for rule in self.transitions if rule.from_mode == name),
                key=lambda rule: rule.priority,
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "program_id": self.program_id,
            "reference_identity": {
                "artifact_id": self.reference_identity.artifact_id,
                "content_sha256": self.reference_identity.content_sha256,
                "schema_sha256": self.reference_identity.schema_sha256,
                "n_frames": self.reference_identity.n_frames,
                "n_features": self.reference_identity.n_features,
            },
            "entry_mode": self.entry_mode,
            "modes": [mode.to_dict() for mode in self.modes],
            "transitions": [rule.to_dict() for rule in self.transitions],
            "signal_specs": [signal.to_dict() for signal in self.signal_specs],
            "horizon_steps": self.horizon_steps,
            "max_queries": self.max_queries,
        }


def validate_program(program: OracleProgram) -> list[str]:
    """Return structural, guard, recovery, and liveness violations."""

    errors: list[str] = []
    if not isinstance(program.program_id, str) or not _IDENTIFIER.fullmatch(program.program_id):
        errors.append("program_id must be a stable identifier")
    if not isinstance(program.reference_identity, ReferenceIdentity):
        errors.append("reference_identity must be a ReferenceIdentity")
        n_reference_frames = 0
    else:
        n_reference_frames = program.reference_identity.n_frames
    if (
        not isinstance(program.horizon_steps, int)
        or isinstance(program.horizon_steps, bool)
        or not (1 <= program.horizon_steps <= MAX_HORIZON_STEPS)
    ):
        errors.append(f"horizon_steps must be in [1, {MAX_HORIZON_STEPS}]")
    if (
        not isinstance(program.max_queries, int)
        or isinstance(program.max_queries, bool)
        or not (1 <= program.max_queries <= MAX_QUERIES)
    ):
        errors.append(f"max_queries must be in [1, {MAX_QUERIES}]")
    if not program.modes:
        return [*errors, "program must contain at least one mode"]
    if any(not isinstance(mode, ModeSpec) for mode in program.modes):
        return [*errors, "all modes must be ModeSpec values"]
    if any(not isinstance(rule, TransitionRule) for rule in program.transitions):
        return [*errors, "all transitions must be TransitionRule values"]
    if any(not isinstance(signal, SignalSpec) for signal in program.signal_specs):
        return [*errors, "all signal_specs must be SignalSpec values"]

    signal_names = [signal.name for signal in program.signal_specs]
    signal_name_set = set(signal_names)
    if len(signal_names) != len(signal_name_set):
        errors.append("signal spec names must be unique")
    signal_by_name = {signal.name: signal for signal in program.signal_specs}

    names = [mode.name for mode in program.modes]
    name_set = set(names)
    if len(names) != len(name_set):
        errors.append("mode names must be unique")
    if program.entry_mode not in name_set:
        errors.append("entry_mode must name an existing mode")

    kinds = {mode.name: mode.kind for mode in program.modes}
    terminals = {name for name, kind in kinds.items() if kind is ModeKind.TERMINAL}
    recoveries = {name for name, kind in kinds.items() if kind is ModeKind.RECOVERY}
    if not terminals:
        errors.append("program must contain at least one terminal mode")
    if program.entry_mode in kinds and kinds[program.entry_mode] is not ModeKind.TASK:
        errors.append("entry_mode must be a task mode")

    for mode in program.modes:
        _, stop = mode.reference_span
        if stop > n_reference_frames:
            errors.append(f"mode {mode.name!r} reference_span exceeds bound reference")
        if mode.kind is ModeKind.TERMINAL:
            if mode.max_dwell_steps is not None:
                errors.append(f"terminal mode {mode.name!r} must not declare max_dwell_steps")
        elif mode.max_dwell_steps is None:
            errors.append(
                f"non-terminal mode {mode.name!r} requires max_dwell_steps for fail-closed liveness"
            )

    rule_ids: set[str] = set()
    priorities: set[tuple[str, int]] = set()
    adjacency: dict[str, set[str]] = {name: set() for name in names}
    zero_dwell_adjacency: dict[str, set[str]] = {name: set() for name in names}
    outgoing: dict[str, list[TransitionRule]] = {name: [] for name in names}
    for index, rule in enumerate(program.transitions):
        prefix = f"transition[{index}]"
        if rule.guard.signal not in signal_name_set:
            errors.append(f"{prefix}: guard uses undeclared signal {rule.guard.signal!r}")
        phase_signal = rule.phase_transfer.signal
        if phase_signal is not None:
            if phase_signal not in signal_name_set:
                errors.append(f"{prefix}: phase transfer uses undeclared signal {phase_signal!r}")
            elif signal_by_name[phase_signal].unit != "1":
                errors.append(
                    f"{prefix}: normalized phase signal {phase_signal!r} "
                    "must use dimensionless unit '1'"
                )
        if rule.rule_id in rule_ids:
            errors.append(f"{prefix}: duplicate rule_id {rule.rule_id!r}")
        rule_ids.add(rule.rule_id)
        if rule.from_mode not in name_set:
            errors.append(f"{prefix}: unknown from_mode {rule.from_mode!r}")
        if rule.to_mode not in name_set:
            errors.append(f"{prefix}: unknown to_mode {rule.to_mode!r}")
        if rule.from_mode == rule.to_mode:
            errors.append(f"{prefix}: self-transition is not allowed")
        priority_key = (rule.from_mode, rule.priority)
        if priority_key in priorities:
            errors.append(
                f"{prefix}: outgoing priorities must be unique for deterministic dispatch"
            )
        priorities.add(priority_key)
        if rule.from_mode in name_set and rule.to_mode in name_set:
            adjacency[rule.from_mode].add(rule.to_mode)
            if rule.min_dwell_steps == 0:
                zero_dwell_adjacency[rule.from_mode].add(rule.to_mode)
            outgoing[rule.from_mode].append(rule)
            source = program.mode(rule.from_mode)
            if source.max_dwell_steps is not None and rule.min_dwell_steps > source.max_dwell_steps:
                errors.append(f"{prefix}: min_dwell_steps exceeds source max_dwell_steps")
            source_kind = kinds[rule.from_mode]
            target_kind = kinds[rule.to_mode]
            if target_kind is ModeKind.RECOVERY:
                if rule.kind is not TransitionKind.RECOVERY:
                    errors.append(
                        f"{prefix}: edges into recovery modes must be explicit recovery edges"
                    )
            elif rule.kind is TransitionKind.RECOVERY:
                errors.append(f"{prefix}: recovery edge must target a recovery mode")
            if source_kind is ModeKind.RECOVERY:
                if rule.kind is not TransitionKind.REJOIN:
                    errors.append(
                        f"{prefix}: edges out of recovery modes must be explicit rejoin edges"
                    )
                if target_kind is not ModeKind.TASK:
                    errors.append(f"{prefix}: rejoin must target a task mode")
            elif rule.kind is TransitionKind.REJOIN:
                errors.append(f"{prefix}: rejoin edge must originate in a recovery mode")

    for mode in program.modes:
        mode_outgoing = outgoing[mode.name]
        if mode.kind is ModeKind.TERMINAL and mode_outgoing:
            errors.append(f"terminal mode {mode.name!r} cannot have outgoing edges")
        if mode.kind is not ModeKind.TERMINAL and not mode_outgoing:
            errors.append(f"non-terminal mode {mode.name!r} has no outgoing edge")
    for recovery in recoveries:
        incoming_recovery = any(
            rule.to_mode == recovery and rule.kind is TransitionKind.RECOVERY
            for rule in program.transitions
        )
        outgoing_rejoin = any(
            rule.from_mode == recovery and rule.kind is TransitionKind.REJOIN
            for rule in program.transitions
        )
        if not incoming_recovery:
            errors.append(f"recovery mode {recovery!r} has no explicit recovery edge")
        if not outgoing_rejoin:
            errors.append(f"recovery mode {recovery!r} has no explicit rejoin edge")

    if _has_directed_cycle(zero_dwell_adjacency):
        errors.append(
            "program contains a zero-dwell transition cycle; at least one edge "
            "in every cycle must require positive dwell"
        )

    if program.entry_mode in name_set:
        reachable = _path_reachable(program.entry_mode, adjacency)
        for name in names:
            if name not in reachable:
                errors.append(f"mode {name!r} is unreachable from entry_mode")
    reverse: dict[str, set[str]] = {name: set() for name in names}
    for source, targets in adjacency.items():
        for target in targets:
            reverse[target].add(source)
    can_reach_terminal: set[str] = set()
    for terminal in terminals:
        can_reach_terminal.update(_path_reachable(terminal, reverse))
    for name in names:
        if name not in can_reach_terminal:
            errors.append(f"mode {name!r} has no topological path to a terminal mode")
    return errors
