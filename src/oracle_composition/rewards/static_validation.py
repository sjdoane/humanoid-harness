"""Capability-minimal AST and deterministic scalar validation."""

from __future__ import annotations

import ast
import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .contract import (
    CANDIDATE_READ_SET_V1,
    TASK_TERM_ABS_MAX,
    CandidateTaskInputsV1,
    RewardContractError,
    canonical_json_bytes,
)

MAX_SOURCE_BYTES = 16 * 1024
MAX_AST_NODES = 512
SAFE_FUNCTION_NAMES = frozenset({"abs", "min", "max", "exp", "sqrt", "clip"})
_BINARY_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_UNARY_OPERATORS = (ast.UAdd, ast.USub, ast.Not)
_BOOLEAN_OPERATORS = (ast.And, ast.Or)
_COMPARISON_OPERATORS = (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)
_CALL_ARITIES = {
    "abs": (1, 1),
    "min": (2, 2),
    "max": (2, 2),
    "exp": (1, 1),
    "sqrt": (1, 1),
    "clip": (3, 3),
}


class StaticValidationError(RewardContractError):
    """Raised when candidate bytes exceed the v1 language or output contract."""


def _clip(value: float, lower: float, upper: float) -> float:
    if lower > upper:
        raise ValueError("clip lower bound exceeds upper bound")
    return min(max(value, lower), upper)


SAFE_GLOBALS: Mapping[str, object] = MappingProxyType(
    {
        "abs": abs,
        "min": min,
        "max": max,
        "exp": math.exp,
        "sqrt": math.sqrt,
        "clip": _clip,
    }
)


def _freeze_literal(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_literal(child) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_literal(child) for child in value)
    return value


def _public_name(value: str, *, field: str) -> None:
    if not value.isidentifier() or value.startswith("_") or "__" in value:
        raise StaticValidationError(f"{field} contains a private or dunder name")


def _numeric_literal(node: ast.Constant) -> None:
    value = node.value
    if isinstance(value, bool) or type(value) not in (int, float):
        raise StaticValidationError("task_term may contain only numeric literals")
    if isinstance(value, int) and value.bit_length() > 128:
        raise StaticValidationError("integer literal exceeds the bounded numeric language")
    if isinstance(value, float) and not math.isfinite(value):
        raise StaticValidationError("numeric literals must be finite")


class _FunctionValidator:
    def __init__(self, function: ast.FunctionDef) -> None:
        self.function = function
        self.locals: set[str] = set()
        self._collect_assignment_names(function.body)

    def _collect_assignment_names(self, statements: Sequence[ast.stmt]) -> None:
        for statement in statements:
            if isinstance(statement, ast.Assign):
                if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                    raise StaticValidationError("assignments require one plain local name")
                name = statement.targets[0].id
                _public_name(name, field="local assignment")
                if name in {"x", "task_term", *SAFE_FUNCTION_NAMES}:
                    raise StaticValidationError(
                        "candidate cannot replace an input or safe function"
                    )
                self.locals.add(name)
            elif isinstance(statement, ast.If):
                self._collect_assignment_names(statement.body)
                self._collect_assignment_names(statement.orelse)

    def validate(self) -> None:
        arguments = self.function.args
        if (
            self.function.name != "task_term"
            or len(arguments.args) != 1
            or arguments.args[0].arg != "x"
            or arguments.posonlyargs
            or arguments.kwonlyargs
            or arguments.vararg is not None
            or arguments.kwarg is not None
            or arguments.defaults
            or arguments.kw_defaults
            or self.function.decorator_list
            or self.function.returns is not None
            or arguments.args[0].annotation is not None
            or self.function.type_params
        ):
            raise StaticValidationError("source must define exactly task_term(x) without extras")
        self._statements(self.function.body, depth=0)
        if not _block_always_returns(self.function.body):
            raise StaticValidationError("every task_term control path must return a value")

    def _statements(self, statements: Sequence[ast.stmt], *, depth: int) -> None:
        if not statements:
            raise StaticValidationError("conditional blocks cannot be empty")
        if depth > 8:
            raise StaticValidationError("conditional nesting exceeds the bounded language")
        for statement in statements:
            if isinstance(statement, ast.Assign):
                self._expression(statement.value)
                continue
            if isinstance(statement, ast.Return):
                if statement.value is None:
                    raise StaticValidationError("task_term must return one scalar expression")
                self._expression(statement.value)
                continue
            if isinstance(statement, ast.If):
                self._expression(statement.test)
                self._statements(statement.body, depth=depth + 1)
                if statement.orelse:
                    self._statements(statement.orelse, depth=depth + 1)
                continue
            raise StaticValidationError(
                f"task_term statement {type(statement).__name__} is not allowed"
            )

    def _expression(self, node: ast.expr, *, direct_call_name: bool = False) -> None:
        if isinstance(node, ast.Constant):
            _numeric_literal(node)
            return
        if isinstance(node, ast.Name):
            _public_name(node.id, field="expression")
            if direct_call_name:
                if node.id not in SAFE_FUNCTION_NAMES:
                    raise StaticValidationError("only injected numeric functions may be called")
            elif node.id not in {"x", *self.locals}:
                raise StaticValidationError(f"unknown or global name {node.id!r}")
            return
        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name) or node.value.id != "x":
                raise StaticValidationError("attribute access is limited to the candidate input")
            _public_name(node.attr, field="attribute")
            if node.attr not in CANDIDATE_READ_SET_V1:
                raise StaticValidationError(f"candidate cannot read signal {node.attr!r}")
            return
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, _BINARY_OPERATORS):
                raise StaticValidationError("arithmetic operator is not allowed")
            if isinstance(node.op, ast.Pow) and (
                not isinstance(node.right, ast.Constant)
                or isinstance(node.right.value, bool)
                or type(node.right.value) not in (int, float)
                or not math.isfinite(float(node.right.value))
                or abs(float(node.right.value)) > 16.0
            ):
                raise StaticValidationError(
                    "power requires one finite literal exponent with magnitude at most 16"
                )
            self._expression(node.left)
            self._expression(node.right)
            return
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _UNARY_OPERATORS):
                raise StaticValidationError("unary operator is not allowed")
            self._expression(node.operand)
            return
        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, _BOOLEAN_OPERATORS) or len(node.values) < 2:
                raise StaticValidationError("boolean expression is not allowed")
            for value in node.values:
                self._expression(value)
            return
        if isinstance(node, ast.Compare):
            if not node.ops or len(node.ops) != len(node.comparators):
                raise StaticValidationError("comparison is malformed")
            if any(not isinstance(operator, _COMPARISON_OPERATORS) for operator in node.ops):
                raise StaticValidationError("comparison operator is not allowed")
            self._expression(node.left)
            for comparator in node.comparators:
                self._expression(comparator)
            return
        if isinstance(node, ast.IfExp):
            self._expression(node.test)
            self._expression(node.body)
            self._expression(node.orelse)
            return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise StaticValidationError("dynamic or attribute calls are not allowed")
            self._expression(node.func, direct_call_name=True)
            minimum, maximum = _CALL_ARITIES[node.func.id]
            if not minimum <= len(node.args) <= maximum or node.keywords:
                raise StaticValidationError(f"{node.func.id} call has the wrong fixed arity")
            for argument in node.args:
                if isinstance(argument, ast.Starred):
                    raise StaticValidationError("starred calls are not allowed")
                self._expression(argument)
            return
        raise StaticValidationError(f"expression {type(node).__name__} is not allowed")


def _block_always_returns(statements: Sequence[ast.stmt]) -> bool:
    for statement in statements:
        if isinstance(statement, ast.Return):
            return True
        if (
            isinstance(statement, ast.If)
            and statement.orelse
            and _block_always_returns(statement.body)
            and _block_always_returns(statement.orelse)
        ):
            return True
    return False


def _literal_metadata(module: ast.Module, function: ast.FunctionDef) -> dict[str, object]:
    metadata: dict[str, object] = {}
    for statement in module.body:
        if statement is function:
            continue
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                raise StaticValidationError("metadata requires one plain assignment name")
            name = statement.targets[0].id
            value_node = statement.value
        else:
            raise StaticValidationError("module may contain only literal metadata and task_term")
        _public_name(name, field="metadata")
        if name in {"x", "task_term", *SAFE_FUNCTION_NAMES} or name in metadata:
            raise StaticValidationError("metadata name is reserved or duplicated")
        try:
            value = ast.literal_eval(value_node)
            canonical_json_bytes(value)
        except (ValueError, TypeError, MemoryError, RecursionError, RewardContractError) as exc:
            raise StaticValidationError("metadata values must be finite JSON literals") from exc
        metadata[name] = value
    return metadata


def _source_bytes(value: bytes | str) -> bytes:
    if isinstance(value, str):
        try:
            encoded = value.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise StaticValidationError("candidate source must encode as UTF-8") from exc
    elif type(value) is bytes:
        encoded = value
    else:
        raise StaticValidationError("candidate source must be exact bytes or text")
    if not encoded or len(encoded) > MAX_SOURCE_BYTES:
        raise StaticValidationError("candidate source is empty or exceeds 16 KiB")
    try:
        encoded.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise StaticValidationError("candidate source is not strict UTF-8") from exc
    return encoded


def parse_and_validate_task_term_source(
    source: bytes | str,
) -> tuple[bytes, ast.Module, dict[str, object]]:
    encoded = _source_bytes(source)
    try:
        module = ast.parse(encoded.decode("utf-8"), filename="<candidate-task-term>", mode="exec")
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        raise StaticValidationError("candidate source is not valid bounded Python") from exc
    node_count = sum(1 for _node in ast.walk(module))
    if node_count > MAX_AST_NODES:
        raise StaticValidationError("candidate source exceeds 512 AST nodes")
    functions = [statement for statement in module.body if isinstance(statement, ast.FunctionDef)]
    async_functions = [
        statement for statement in module.body if isinstance(statement, ast.AsyncFunctionDef)
    ]
    if len(functions) != 1 or async_functions:
        raise StaticValidationError("source must contain exactly one non-async function")
    function = functions[0]
    _FunctionValidator(function).validate()
    metadata = _literal_metadata(module, function)
    return encoded, module, metadata


def _compile_task_term(module: ast.Module) -> Any:
    namespace: dict[str, object] = {"__builtins__": {}, **SAFE_GLOBALS}
    try:
        code = compile(module, "<candidate-task-term>", "exec", dont_inherit=True, optimize=2)
        exec(code, namespace, namespace)
    except BaseException as exc:
        raise StaticValidationError(
            f"candidate could not be installed: {type(exc).__name__}: {exc}"
        ) from exc
    function = namespace.get("task_term")
    if not callable(function):
        raise StaticValidationError("candidate did not install task_term")
    return function


DEFAULT_DETERMINISM_PROBES = (
    CandidateTaskInputsV1(-1.0, 1.0),
    CandidateTaskInputsV1(0.75, 1.0),
    CandidateTaskInputsV1(1.0, 1.0),
    CandidateTaskInputsV1(1.25, 1.0),
    CandidateTaskInputsV1(3.0, 1.0),
    CandidateTaskInputsV1(0.5, 0.5),
    CandidateTaskInputsV1(1.5, 1.5),
)


def _evaluate_one(function: Any, inputs: CandidateTaskInputsV1) -> float:
    try:
        value = function(inputs)
    except BaseException as exc:
        raise StaticValidationError(
            f"candidate failed a deterministic probe: {type(exc).__name__}: {exc}"
        ) from exc
    if type(value) is not float:
        raise StaticValidationError("task_term must return one Python float")
    if not math.isfinite(value):
        raise StaticValidationError("task_term returned NaN or infinity")
    if abs(value) > TASK_TERM_ABS_MAX:
        raise StaticValidationError("task_term breached the output envelope")
    return value


def _determinism_digest(function: Any, probes: Sequence[CandidateTaskInputsV1]) -> str:
    first = [_evaluate_one(function, item) for item in probes]
    for item in reversed(probes):
        _evaluate_one(function, item)
    second = [_evaluate_one(function, item) for item in probes]
    first_bits = [struct.pack(">d", value) for value in first]
    second_bits = [struct.pack(">d", value) for value in second]
    if first_bits != second_bits:
        raise StaticValidationError("task_term is not bitwise deterministic under interleaving")
    digest = hashlib.sha256()
    for inputs, encoded_value in zip(probes, first_bits, strict=True):
        encoded_input = inputs.canonical_bytes
        digest.update(len(encoded_input).to_bytes(4, "big"))
        digest.update(encoded_input)
        digest.update(encoded_value)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class StaticValidationReceiptV1:
    source_sha256: str
    source_bytes: int
    ast_nodes: int
    read_set: tuple[str, ...]
    metadata: Mapping[str, object]
    determinism_sha256: str
    passed: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "source_sha256": self.source_sha256,
            "source_bytes": self.source_bytes,
            "ast_nodes": self.ast_nodes,
            "read_set": list(self.read_set),
            "metadata": dict(self.metadata),
            "determinism_sha256": self.determinism_sha256,
            "passed": self.passed,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class ValidatedTaskTermV1:
    source_bytes: bytes
    metadata: Mapping[str, object]
    receipt: StaticValidationReceiptV1
    _function: Any

    def task_term(self, x: CandidateTaskInputsV1) -> float:
        if not isinstance(x, CandidateTaskInputsV1):
            raise StaticValidationError("task_term input must be CandidateTaskInputsV1")
        return _evaluate_one(self._function, x)


def validate_task_term_source(
    source: bytes | str,
    *,
    determinism_probes: Sequence[CandidateTaskInputsV1] = DEFAULT_DETERMINISM_PROBES,
) -> ValidatedTaskTermV1:
    encoded, module, metadata = parse_and_validate_task_term_source(source)
    if (
        not isinstance(determinism_probes, Sequence)
        or isinstance(determinism_probes, (str, bytes))
        or not determinism_probes
        or any(not isinstance(item, CandidateTaskInputsV1) for item in determinism_probes)
    ):
        raise StaticValidationError("determinism probes must be nonempty candidate inputs")
    function = _compile_task_term(module)
    determinism = _determinism_digest(function, tuple(determinism_probes))
    frozen_metadata = MappingProxyType(
        {name: _freeze_literal(value) for name, value in metadata.items()}
    )
    receipt = StaticValidationReceiptV1(
        source_sha256=hashlib.sha256(encoded).hexdigest(),
        source_bytes=len(encoded),
        ast_nodes=sum(1 for _node in ast.walk(module)),
        read_set=CANDIDATE_READ_SET_V1,
        metadata=frozen_metadata,
        determinism_sha256=determinism,
    )
    return ValidatedTaskTermV1(
        source_bytes=encoded,
        metadata=frozen_metadata,
        receipt=receipt,
        _function=function,
    )


__all__ = [
    "DEFAULT_DETERMINISM_PROBES",
    "MAX_AST_NODES",
    "MAX_SOURCE_BYTES",
    "SAFE_FUNCTION_NAMES",
    "StaticValidationError",
    "StaticValidationReceiptV1",
    "ValidatedTaskTermV1",
    "parse_and_validate_task_term_source",
    "validate_task_term_source",
]
