"""Fail-closed JSON oracle contract and deterministic state machine."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import stat
from collections import defaultdict, deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes

ORACLE_SCHEMA_ID = "humanoid_controller_switching_oracle/v1"
EVIDENCE_CLASS = "exploratory_oracle_cycle"
ALLOWED_SIGNALS = frozenset({"t", "v_x", "v_target", "z_root", "torso_up", "x_travelled", "dwell"})
MAX_JSON_BYTES = 256 * 1024
MAX_JSON_DEPTH = 64
MAX_JSON_NODES = 65_536
MAX_GUARD_CHARACTERS = 512
MAX_GUARD_NODES = 128
MAX_GUARD_DEPTH = 32
MAX_ORACLE_STATES = 64
MAX_ORACLE_TRANSITIONS = 256
LEGACY_RECOVERY_MIN_DWELL = 1
LEGACY_RECOVERY_MAX_DURATION = 64
LEGACY_RECOVERY_REENTRY_DWELL = 1
RECOVERY_REJOIN_SEMANTICS = "suspended_state_dwell_reset"
_LEGACY_RECOVERY_ORACLE_SHA256 = frozenset(
    {
        "32bfc555ffc578d4cf8f75823acc1ba98db6621b2bcd0204f0725b0bf70af029",
        "4368848952e43949ed5613352992d0db2dd61026cdbb78a7616c4b241e30d5a5",
        "4d24f22780360d7632235572d97b3fccfe7c376162e69e15082f77bd7afcccf1",
    }
)

_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class OracleContractError(ValueError):
    """Raised when an oracle or its guard language violates the contract."""


def _reject_constant(value: str) -> None:
    raise OracleContractError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise OracleContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def decode_json_object(encoded: bytes, *, source: str = "JSON") -> dict[str, object]:
    """Decode bounded UTF-8 JSON while rejecting duplicate keys and NaN/Infinity."""

    if type(encoded) is not bytes or not encoded or len(encoded) > MAX_JSON_BYTES:
        raise OracleContractError(f"{source} must be nonempty and at most {MAX_JSON_BYTES} bytes")
    try:
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except OracleContractError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        MemoryError,
        OverflowError,
        ValueError,
    ) as exc:
        raise OracleContractError(f"{source} is not valid bounded JSON: {exc}") from exc
    if type(value) is not dict:
        raise OracleContractError(f"{source} must contain one JSON object")
    pending: list[tuple[object, int]] = [(value, 1)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if depth > MAX_JSON_DEPTH:
            raise OracleContractError(f"{source} JSON structure exceeds the depth limit")
        if nodes > MAX_JSON_NODES:
            raise OracleContractError(f"{source} JSON structure exceeds the node limit")
        if type(item) is dict:
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
    return value


def read_json_object(path: Path) -> tuple[dict[str, object], bytes]:
    """Read one regular, non-linked JSON artifact exactly once."""

    candidate = Path(path)
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise OracleContractError(f"JSON artifact is unavailable: {candidate}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise OracleContractError(f"JSON artifact must be a regular non-linked file: {candidate}")
    if not 0 < before.st_size <= MAX_JSON_BYTES:
        raise OracleContractError(f"JSON artifact size is outside the bound: {candidate}")
    try:
        encoded = candidate.read_bytes()
        after = candidate.lstat()
    except OSError as exc:
        raise OracleContractError(f"JSON artifact cannot be read: {candidate}") from exc

    def identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)

    if identity(before) != identity(after) or len(encoded) != before.st_size:
        raise OracleContractError(f"JSON artifact changed while it was read: {candidate}")
    return decode_json_object(encoded, source=str(candidate)), encoded


def _exact_keys(
    value: object,
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    field: str,
) -> dict[str, object]:
    if type(value) is not dict:
        raise OracleContractError(f"{field} must be an object")
    keys = set(value)
    if not required.issubset(keys) or not keys.issubset(required | optional):
        raise OracleContractError(f"{field} keys differ from the contract")
    return value


def _name(value: object, *, field: str) -> str:
    if type(value) is not str or _NAME.fullmatch(value) is None:
        raise OracleContractError(f"{field} must be a lowercase snake-case identifier")
    return value


def _finite_number(value: object, *, field: str) -> float:
    if type(value) not in {int, float}:
        raise OracleContractError(f"{field} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise OracleContractError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise OracleContractError(f"{field} must be finite")
    return result


def _validate_guard_node(node: ast.AST) -> None:
    if isinstance(node, ast.Expression):
        _validate_guard_node(node.body)
        return
    if isinstance(node, ast.Name):
        if node.id not in ALLOWED_SIGNALS:
            raise OracleContractError(f"unknown guard signal: {node.id}")
        return
    if isinstance(node, ast.Constant):
        _finite_number(node.value, field="guard constant")
        return
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        if len(node.values) < 2:
            raise OracleContractError("boolean guard operation requires at least two operands")
        for value in node.values:
            _validate_guard_node(value)
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        _validate_guard_node(node.operand)
        return
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
    ):
        _validate_guard_node(node.operand)
        value = node.operand.value
        if type(value) not in {int, float}:
            raise OracleContractError("signed guard constants must be numeric")
        _finite_number(+value if isinstance(node.op, ast.UAdd) else -value, field="guard constant")
        return
    if isinstance(node, ast.Compare):
        if not node.ops or len(node.ops) != len(node.comparators):
            raise OracleContractError("guard comparison is malformed")
        if any(
            not isinstance(operator, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE))
            for operator in node.ops
        ):
            raise OracleContractError("guard comparison operator is forbidden")
        _validate_guard_node(node.left)
        for comparator in node.comparators:
            _validate_guard_node(comparator)
        return
    if isinstance(node, ast.Call):
        raise OracleContractError("calls are forbidden in guards")
    if isinstance(node, ast.Attribute):
        raise OracleContractError("attributes are forbidden in guards")
    raise OracleContractError(f"guard syntax is forbidden: {type(node).__name__}")


def _evaluate_guard_node(node: ast.AST, signals: Mapping[str, float]) -> float | bool:
    if isinstance(node, ast.Expression):
        return _evaluate_guard_node(node.body, signals)
    if isinstance(node, ast.Name):
        return signals[node.id]
    if isinstance(node, ast.Constant):
        return float(node.value)
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_guard_node(node.operand, signals)
        if isinstance(node.op, ast.Not):
            return not bool(operand)
        if isinstance(node.op, ast.UAdd):
            return +float(operand)
        if isinstance(node.op, ast.USub):
            return -float(operand)
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            return all(bool(_evaluate_guard_node(value, signals)) for value in node.values)
        return any(bool(_evaluate_guard_node(value, signals)) for value in node.values)
    if isinstance(node, ast.Compare):
        left = float(_evaluate_guard_node(node.left, signals))
        for operator, comparator in zip(node.ops, node.comparators, strict=True):
            right = float(_evaluate_guard_node(comparator, signals))
            if isinstance(operator, ast.Eq):
                passed = left == right
            elif isinstance(operator, ast.NotEq):
                passed = left != right
            elif isinstance(operator, ast.Lt):
                passed = left < right
            elif isinstance(operator, ast.LtE):
                passed = left <= right
            elif isinstance(operator, ast.Gt):
                passed = left > right
            else:
                passed = left >= right
            if not passed:
                return False
            left = right
        return True
    raise AssertionError("validated guard contains an unsupported node")


@dataclass(frozen=True, slots=True)
class GuardExpression:
    source: str
    _tree: ast.Expression

    @classmethod
    def parse(cls, source: object) -> GuardExpression:
        if type(source) is not str or not source.strip() or len(source) > MAX_GUARD_CHARACTERS:
            raise OracleContractError("guard must be nonempty bounded text")
        try:
            parsed = ast.parse(source, mode="eval")
        except (SyntaxError, RecursionError, MemoryError) as exc:
            raise OracleContractError(f"guard syntax is invalid: {exc}") from exc
        nodes = tuple(ast.walk(parsed))
        if len(nodes) > MAX_GUARD_NODES:
            raise OracleContractError("guard syntax tree exceeds the node limit")
        pending = [(parsed, 1)]
        while pending:
            node, depth = pending.pop()
            if depth > MAX_GUARD_DEPTH:
                raise OracleContractError("guard syntax tree exceeds the depth limit")
            pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
        _validate_guard_node(parsed)
        return cls(source=source, _tree=parsed)

    def evaluate(self, signals: Mapping[str, object]) -> bool:
        if set(signals) != ALLOWED_SIGNALS:
            raise OracleContractError("guard signals differ from the allowed signal set")
        checked = {
            name: _finite_number(signals[name], field=f"signal {name}")
            for name in sorted(ALLOWED_SIGNALS)
        }
        return self.evaluate_prevalidated(checked)

    def evaluate_prevalidated(self, signals: Mapping[str, float]) -> bool:
        """Evaluation seam for observing branch access after signal admission."""

        return bool(_evaluate_guard_node(self._tree, signals))


@dataclass(frozen=True, slots=True)
class OracleState:
    behavior: str
    min_dwell: int


@dataclass(frozen=True, slots=True)
class OracleTransition:
    source: str
    target: str
    guard: GuardExpression
    priority: int


@dataclass(frozen=True, slots=True)
class OracleRecovery:
    behavior: str
    guard: GuardExpression
    min_dwell: int
    max_duration: int
    reentry_dwell: int
    rejoin: str
    explicit_semantics: bool


@dataclass(frozen=True, slots=True)
class OracleProgram:
    oracle_id: str
    behaviors: tuple[str, ...]
    initial: str
    states: Mapping[str, OracleState]
    transitions: tuple[OracleTransition, ...]
    recovery: OracleRecovery | None

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "behaviors": list(self.behaviors),
            "evidence_class": EVIDENCE_CLASS,
            "initial": self.initial,
            "oracle_id": self.oracle_id,
            "schema_version": 1,
            "states": {
                name: {"behavior": state.behavior, "min_dwell": state.min_dwell}
                for name, state in self.states.items()
            },
            "transitions": [
                {
                    "from": transition.source,
                    "guard": transition.guard.source,
                    "priority": transition.priority,
                    "to": transition.target,
                }
                for transition in self.transitions
            ],
        }
        if self.recovery is not None:
            recovery: dict[str, object] = {
                "behavior": self.recovery.behavior,
                "guard": self.recovery.guard.source,
            }
            if self.recovery.explicit_semantics:
                recovery.update(
                    {
                        "max_duration": self.recovery.max_duration,
                        "min_dwell": self.recovery.min_dwell,
                        "reentry_dwell": self.recovery.reentry_dwell,
                        "rejoin": self.recovery.rejoin,
                    }
                )
            result["recovery"] = recovery
        return result


def _reject_zero_dwell_cycles(
    states: Mapping[str, OracleState], transitions: Sequence[OracleTransition]
) -> None:
    zero_states = {name for name, state in states.items() if state.min_dwell == 0}
    graph: dict[str, list[str]] = defaultdict(list)
    for transition in transitions:
        if transition.source in zero_states and transition.target in zero_states:
            graph[transition.source].append(transition.target)
    colors = {state: 0 for state in zero_states}
    for start in sorted(zero_states):
        if colors[start] != 0:
            continue
        colors[start] = 1
        stack: list[tuple[str, Iterator[str]]] = [(start, iter(graph[start]))]
        while stack:
            _node, targets = stack[-1]
            try:
                target = next(targets)
            except StopIteration:
                finished, _targets = stack.pop()
                colors[finished] = 2
                continue
            color = colors[target]
            if color == 1:
                raise OracleContractError("oracle contains a zero-dwell transition cycle")
            if color == 0:
                colors[target] = 1
                stack.append((target, iter(graph[target])))


def oracle_program_from_dict(
    value: Mapping[str, object],
    *,
    available_behaviors: Sequence[str],
    allow_legacy_recovery: bool = False,
) -> OracleProgram:
    root = _exact_keys(
        value,
        required={
            "schema_version",
            "evidence_class",
            "oracle_id",
            "behaviors",
            "initial",
            "states",
            "transitions",
        },
        optional={"recovery"},
        field="oracle",
    )
    if root["schema_version"] != 1:
        raise OracleContractError("oracle schema_version must be 1")
    if root["evidence_class"] != EVIDENCE_CLASS:
        raise OracleContractError(f"oracle evidence_class must be {EVIDENCE_CLASS}")
    oracle_id = _name(root["oracle_id"], field="oracle_id")
    available = tuple(_name(item, field="available behavior") for item in available_behaviors)
    if not available or len(set(available)) != len(available):
        raise OracleContractError("available behaviors must be unique and nonempty")
    raw_behaviors = root["behaviors"]
    if type(raw_behaviors) is not list or not raw_behaviors:
        raise OracleContractError("behaviors must be a nonempty array")
    behaviors = tuple(_name(item, field="behavior") for item in raw_behaviors)
    if len(set(behaviors)) != len(behaviors):
        raise OracleContractError("behaviors must be unique")
    unknown_behaviors = set(behaviors) - set(available)
    if unknown_behaviors:
        raise OracleContractError(f"undefined behaviors: {sorted(unknown_behaviors)}")

    raw_states = root["states"]
    if type(raw_states) is not dict or not raw_states:
        raise OracleContractError("states must be a nonempty object")
    if len(raw_states) > MAX_ORACLE_STATES:
        raise OracleContractError(f"oracle exceeds the {MAX_ORACLE_STATES}-state limit")
    states: dict[str, OracleState] = {}
    for raw_name, raw_state in raw_states.items():
        name = _name(raw_name, field="state name")
        item = _exact_keys(
            raw_state,
            required={"behavior", "min_dwell"},
            field=f"states.{name}",
        )
        behavior = _name(item["behavior"], field=f"states.{name}.behavior")
        if behavior not in behaviors:
            raise OracleContractError(f"state {name} uses undefined behavior {behavior}")
        min_dwell = item["min_dwell"]
        if type(min_dwell) is not int or not 0 <= min_dwell <= 1_000_000:
            raise OracleContractError(f"states.{name}.min_dwell must be a bounded integer")
        states[name] = OracleState(behavior=behavior, min_dwell=min_dwell)
    initial = _name(root["initial"], field="initial")
    if initial not in states:
        raise OracleContractError("initial state is missing from states")

    raw_transitions = root["transitions"]
    if type(raw_transitions) is not list:
        raise OracleContractError("transitions must be an array")
    if len(raw_transitions) > MAX_ORACLE_TRANSITIONS:
        raise OracleContractError(f"oracle exceeds the {MAX_ORACLE_TRANSITIONS}-transition limit")
    transitions: list[OracleTransition] = []
    priorities: dict[str, set[int]] = defaultdict(set)
    adjacency: dict[str, set[str]] = defaultdict(set)
    for index, raw_transition in enumerate(raw_transitions):
        item = _exact_keys(
            raw_transition,
            required={"from", "to", "guard", "priority"},
            field=f"transitions[{index}]",
        )
        source = _name(item["from"], field=f"transitions[{index}].from")
        target = _name(item["to"], field=f"transitions[{index}].to")
        if source not in states or target not in states:
            raise OracleContractError(f"transition {index} refers to an undefined state")
        priority = item["priority"]
        if type(priority) is not int or not 0 <= priority <= 1_000_000:
            raise OracleContractError(f"transitions[{index}].priority must be bounded integer")
        if priority in priorities[source]:
            raise OracleContractError(f"duplicate transition priority {priority} from {source}")
        priorities[source].add(priority)
        adjacency[source].add(target)
        transitions.append(
            OracleTransition(
                source=source,
                target=target,
                guard=GuardExpression.parse(item["guard"]),
                priority=priority,
            )
        )

    reachable = {initial}
    pending = deque([initial])
    while pending:
        source = pending.popleft()
        for target in sorted(adjacency[source]):
            if target not in reachable:
                reachable.add(target)
                pending.append(target)
    missing = set(states) - reachable
    if missing:
        raise OracleContractError(f"unreachable states: {sorted(missing)}")
    _reject_zero_dwell_cycles(states, transitions)

    recovery: OracleRecovery | None = None
    if "recovery" in root:
        raw_recovery = _exact_keys(
            root["recovery"],
            required={"behavior", "guard"},
            optional={"min_dwell", "max_duration", "reentry_dwell", "rejoin"},
            field="recovery",
        )
        recovery_behavior = _name(raw_recovery["behavior"], field="recovery.behavior")
        if recovery_behavior not in behaviors:
            raise OracleContractError("recovery uses an undefined behavior")
        semantic_fields = {"min_dwell", "max_duration", "reentry_dwell", "rejoin"}
        supplied_semantics = semantic_fields.intersection(raw_recovery)
        if supplied_semantics and supplied_semantics != semantic_fields:
            raise OracleContractError("recovery semantics must be declared together")
        explicit_semantics = supplied_semantics == semantic_fields
        if explicit_semantics:
            min_dwell = raw_recovery["min_dwell"]
            max_duration = raw_recovery["max_duration"]
            reentry_dwell = raw_recovery["reentry_dwell"]
            rejoin = raw_recovery["rejoin"]
            if type(min_dwell) is not int or not 1 <= min_dwell <= 1_000_000:
                raise OracleContractError("recovery.min_dwell must be a positive bounded integer")
            if type(max_duration) is not int or not min_dwell <= max_duration <= 1_000_000:
                raise OracleContractError(
                    "recovery.max_duration must be bounded and at least min_dwell"
                )
            if type(reentry_dwell) is not int or not 1 <= reentry_dwell <= 1_000_000:
                raise OracleContractError(
                    "recovery.reentry_dwell must be a positive bounded integer"
                )
            if rejoin != RECOVERY_REJOIN_SEMANTICS:
                raise OracleContractError(f"recovery.rejoin must be {RECOVERY_REJOIN_SEMANTICS}")
        elif allow_legacy_recovery:
            # Phase-A v1 files are immutable. Their two-field recovery form is
            # interpreted with these version-defined anti-chatter/liveness bounds.
            min_dwell = LEGACY_RECOVERY_MIN_DWELL
            max_duration = LEGACY_RECOVERY_MAX_DURATION
            reentry_dwell = LEGACY_RECOVERY_REENTRY_DWELL
            rejoin = RECOVERY_REJOIN_SEMANTICS
        else:
            raise OracleContractError(
                "recovery requires min_dwell, max_duration, reentry_dwell, and rejoin"
            )
        recovery = OracleRecovery(
            behavior=recovery_behavior,
            guard=GuardExpression.parse(raw_recovery["guard"]),
            min_dwell=min_dwell,
            max_duration=max_duration,
            reentry_dwell=reentry_dwell,
            rejoin=rejoin,
            explicit_semantics=explicit_semantics,
        )
    return OracleProgram(
        oracle_id=oracle_id,
        behaviors=behaviors,
        initial=initial,
        states=MappingProxyType(states),
        transitions=tuple(transitions),
        recovery=recovery,
    )


def load_oracle_program(
    path: Path, *, available_behaviors: Sequence[str]
) -> tuple[OracleProgram, str]:
    value, encoded = read_json_object(path)
    raw_sha256 = hashlib.sha256(encoded).hexdigest()
    program = oracle_program_from_dict(
        value,
        available_behaviors=available_behaviors,
        allow_legacy_recovery=raw_sha256 in _LEGACY_RECOVERY_ORACLE_SHA256,
    )
    return program, raw_sha256


@dataclass(frozen=True, slots=True)
class MachineDecision:
    state: str
    behavior: str
    controller_switched: bool
    state_transition: bool
    recovery_entered: bool
    recovery_exited: bool
    reason: str


class OracleMachine:
    """One-switch-per-step interpreter with recovery override and deterministic rejoin."""

    def __init__(self, program: OracleProgram) -> None:
        self.program = program
        self._state = program.initial
        self._behavior = program.states[program.initial].behavior
        self._dwell = 0
        self._recovery_active = False
        self._recovery_elapsed = 0
        self._recovery_reentry_remaining = 0
        outgoing: dict[str, list[OracleTransition]] = defaultdict(list)
        for transition in program.transitions:
            outgoing[transition.source].append(transition)
        self._outgoing = {
            state: tuple(sorted(items, key=lambda item: item.priority))
            for state, items in outgoing.items()
        }

    @property
    def state(self) -> str:
        return self._state

    @property
    def behavior(self) -> str:
        return self._behavior

    @property
    def dwell(self) -> int:
        return self._dwell

    def decide(
        self, signals: Mapping[str, object], *, allow_transitions: bool = True
    ) -> MachineDecision:
        if type(allow_transitions) is not bool:
            raise OracleContractError("allow_transitions must be boolean")
        if signals.get("dwell") != self._dwell:
            raise OracleContractError("dwell signal does not match machine state")
        checked = {
            name: _finite_number(signals[name], field=f"signal {name}")
            for name in sorted(ALLOWED_SIGNALS)
            if name in signals
        }
        if set(checked) != ALLOWED_SIGNALS:
            raise OracleContractError("machine signals differ from the allowed signal set")
        recovery = self.program.recovery
        if recovery is not None:
            recovery_guard = recovery.guard.evaluate(checked)
            if self._recovery_active:
                if recovery_guard and self._recovery_elapsed >= recovery.max_duration:
                    raise OracleContractError("recovery exceeded its maximum duration")
                if recovery_guard or self._recovery_elapsed < recovery.min_dwell:
                    return MachineDecision(
                        state=self._state,
                        behavior=self._behavior,
                        controller_switched=False,
                        state_transition=False,
                        recovery_entered=False,
                        recovery_exited=False,
                        reason="recovery_hold",
                    )
                self._recovery_active = False
                self._recovery_elapsed = 0
                # `advance()` follows every decision, including the exit decision.
                # The extra count preserves the declared number of future blocked entries.
                self._recovery_reentry_remaining = recovery.reentry_dwell + 1
                desired = self.program.states[self._state].behavior
                switched = desired != self._behavior
                self._behavior = desired
                self._dwell = 0
                return MachineDecision(
                    state=self._state,
                    behavior=self._behavior,
                    controller_switched=switched,
                    state_transition=False,
                    recovery_entered=False,
                    recovery_exited=True,
                    reason="recovery_rejoin",
                )
            if recovery_guard and self._recovery_reentry_remaining > 0:
                return MachineDecision(
                    state=self._state,
                    behavior=self._behavior,
                    controller_switched=False,
                    state_transition=False,
                    recovery_entered=False,
                    recovery_exited=False,
                    reason="recovery_reentry_dwell",
                )
            if recovery_guard:
                self._recovery_active = True
                self._recovery_elapsed = 0
                switched = recovery.behavior != self._behavior
                self._behavior = recovery.behavior
                self._dwell = 0
                return MachineDecision(
                    state=self._state,
                    behavior=self._behavior,
                    controller_switched=switched,
                    state_transition=False,
                    recovery_entered=True,
                    recovery_exited=False,
                    reason="recovery_guard",
                )

        state = self.program.states[self._state]
        if allow_transitions and self._dwell >= state.min_dwell:
            for transition in self._outgoing.get(self._state, ()):
                if transition.guard.evaluate(checked):
                    previous_behavior = self._behavior
                    self._state = transition.target
                    self._behavior = self.program.states[self._state].behavior
                    self._dwell = 0
                    return MachineDecision(
                        state=self._state,
                        behavior=self._behavior,
                        controller_switched=self._behavior != previous_behavior,
                        state_transition=True,
                        recovery_entered=False,
                        recovery_exited=False,
                        reason=f"transition_priority_{transition.priority}",
                    )
        return MachineDecision(
            state=self._state,
            behavior=self._behavior,
            controller_switched=False,
            state_transition=False,
            recovery_entered=False,
            recovery_exited=False,
            reason="hold" if allow_transitions else "ordinary_transition_deferred",
        )

    def advance(self) -> None:
        self._dwell += 1
        if self._recovery_active:
            self._recovery_elapsed += 1
        elif self._recovery_reentry_remaining > 0:
            self._recovery_reentry_remaining -= 1


__all__ = [
    "ALLOWED_SIGNALS",
    "EVIDENCE_CLASS",
    "MAX_GUARD_CHARACTERS",
    "MAX_GUARD_DEPTH",
    "MAX_GUARD_NODES",
    "MAX_JSON_BYTES",
    "MAX_JSON_DEPTH",
    "MAX_JSON_NODES",
    "MAX_ORACLE_STATES",
    "MAX_ORACLE_TRANSITIONS",
    "ORACLE_SCHEMA_ID",
    "RECOVERY_REJOIN_SEMANTICS",
    "MachineDecision",
    "OracleContractError",
    "OracleMachine",
    "OracleProgram",
    "decode_json_object",
    "load_oracle_program",
    "oracle_program_from_dict",
    "read_json_object",
]
