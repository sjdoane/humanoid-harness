"""Data-only validation for the bounded task-term source language."""

from __future__ import annotations

import ast
import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from .contract import (
    CANDIDATE_READ_SET_V1,
    RewardContractError,
    canonical_json_bytes,
)

MAX_SOURCE_BYTES = 16 * 1024
MAX_AST_NODES = 512
MAX_CONDITIONAL_DEPTH = 8
STATIC_VALIDATION_SCOPE_V1 = "source_bytes_and_ast_only"
DYNAMIC_VALIDATION_NOT_PERFORMED = "not_performed"
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
    """Raised when candidate bytes exceed the v1 static source language."""


def _copy_plain_json(value: object) -> object:
    """Copy exact JSON containers before any immutable view is created."""

    if type(value) is dict:
        copied: dict[str, object] = {}
        for key, child in value.items():  # type: ignore[union-attr]
            if type(key) is not str:
                raise StaticValidationError("metadata keys must be strings")
            copied[key] = _copy_plain_json(child)
        return copied
    if type(value) is list:
        return [_copy_plain_json(child) for child in value]  # type: ignore[union-attr]
    if value is None or type(value) in (bool, str, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise StaticValidationError("metadata must contain only finite JSON literal values")


def _freeze_plain_json(value: object) -> object:
    if type(value) is dict:
        return MappingProxyType(
            {
                key: _freeze_plain_json(child)
                for key, child in value.items()  # type: ignore[union-attr]
            }
        )
    if type(value) is list:
        return tuple(_freeze_plain_json(child) for child in value)  # type: ignore[union-attr]
    return value


def _thaw_internal_json(value: object) -> object:
    if type(value) is MappingProxyType:
        return {
            key: _thaw_internal_json(child)
            for key, child in value.items()  # type: ignore[union-attr]
        }
    if type(value) is tuple:
        return [_thaw_internal_json(child) for child in value]  # type: ignore[union-attr]
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
        for statement in statements:
            if isinstance(statement, ast.Assign):
                self._expression(statement.value, depth=depth)
                continue
            if isinstance(statement, ast.Return):
                if statement.value is None:
                    raise StaticValidationError("task_term must return one scalar expression")
                self._expression(statement.value, depth=depth)
                continue
            if isinstance(statement, ast.If):
                nested_depth = self._nested_conditional_depth(depth)
                self._expression(statement.test, depth=nested_depth)
                self._statements(statement.body, depth=nested_depth)
                if statement.orelse:
                    self._statements(statement.orelse, depth=nested_depth)
                continue
            raise StaticValidationError(
                f"task_term statement {type(statement).__name__} is not allowed"
            )

    @staticmethod
    def _nested_conditional_depth(depth: int) -> int:
        nested_depth = depth + 1
        if nested_depth > MAX_CONDITIONAL_DEPTH:
            raise StaticValidationError("conditional nesting exceeds the bounded language")
        return nested_depth

    def _expression(
        self,
        node: ast.expr,
        *,
        depth: int,
        direct_call_name: bool = False,
    ) -> None:
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
            self._expression(node.left, depth=depth)
            self._expression(node.right, depth=depth)
            return
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _UNARY_OPERATORS):
                raise StaticValidationError("unary operator is not allowed")
            self._expression(node.operand, depth=depth)
            return
        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, _BOOLEAN_OPERATORS) or len(node.values) < 2:
                raise StaticValidationError("boolean expression is not allowed")
            for value in node.values:
                self._expression(value, depth=depth)
            return
        if isinstance(node, ast.Compare):
            if not node.ops or len(node.ops) != len(node.comparators):
                raise StaticValidationError("comparison is malformed")
            if any(not isinstance(operator, _COMPARISON_OPERATORS) for operator in node.ops):
                raise StaticValidationError("comparison operator is not allowed")
            self._expression(node.left, depth=depth)
            for comparator in node.comparators:
                self._expression(comparator, depth=depth)
            return
        if isinstance(node, ast.IfExp):
            nested_depth = self._nested_conditional_depth(depth)
            self._expression(node.test, depth=nested_depth)
            self._expression(node.body, depth=nested_depth)
            self._expression(node.orelse, depth=nested_depth)
            return
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise StaticValidationError("dynamic or attribute calls are not allowed")
            self._expression(node.func, depth=depth, direct_call_name=True)
            minimum, maximum = _CALL_ARITIES[node.func.id]
            if not minimum <= len(node.args) <= maximum or node.keywords:
                raise StaticValidationError(f"{node.func.id} call has the wrong fixed arity")
            for argument in node.args:
                if isinstance(argument, ast.Starred):
                    raise StaticValidationError("starred calls are not allowed")
                self._expression(argument, depth=depth)
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
            value = _copy_plain_json(value)
            canonical_json_bytes(value)
        except (
            MemoryError,
            RecursionError,
            RewardContractError,
            TypeError,
            UnicodeError,
            ValueError,
        ) as exc:
            raise StaticValidationError("metadata values must be finite JSON literals") from exc
        metadata[name] = value
    return metadata


def _source_bytes(value: bytes | str) -> bytes:
    if type(value) is str:
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


@dataclass(frozen=True, slots=True)
class StaticAcceptanceReceiptV1:
    source_sha256: str
    source_bytes: int
    ast_nodes: int
    read_set: tuple[str, ...]
    metadata: Mapping[str, object]
    validation_scope: str = STATIC_VALIDATION_SCOPE_V1
    static_accepted: bool = True
    dynamic_validation_status: str = DYNAMIC_VALIDATION_NOT_PERFORMED

    def __post_init__(self) -> None:
        if (
            type(self.source_sha256) is not str
            or len(self.source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.source_sha256)
        ):
            raise StaticValidationError("static acceptance source_sha256 is invalid")
        if type(self.source_bytes) is not int or not 0 < self.source_bytes <= MAX_SOURCE_BYTES:
            raise StaticValidationError("static acceptance source byte count is invalid")
        if type(self.ast_nodes) is not int or not 0 < self.ast_nodes <= MAX_AST_NODES:
            raise StaticValidationError("static acceptance AST node count is invalid")
        if (
            type(self.read_set) not in (tuple, list)
            or any(type(item) is not str for item in self.read_set)
            or tuple(self.read_set) != CANDIDATE_READ_SET_V1
        ):
            raise StaticValidationError("static acceptance read set differs from v1")
        if type(self.metadata) is not dict:
            raise StaticValidationError("static acceptance metadata must be a plain JSON object")
        try:
            plain_metadata = _copy_plain_json(self.metadata)
            canonical_json_bytes(plain_metadata)
            frozen_metadata = _freeze_plain_json(plain_metadata)
        except (MemoryError, RewardContractError, RecursionError, UnicodeError) as exc:
            raise StaticValidationError("static acceptance metadata is not finite JSON") from exc
        if (
            type(self.validation_scope) is not str
            or self.validation_scope != STATIC_VALIDATION_SCOPE_V1
            or self.static_accepted is not True
            or type(self.dynamic_validation_status) is not str
            or self.dynamic_validation_status != DYNAMIC_VALIDATION_NOT_PERFORMED
        ):
            raise StaticValidationError("static acceptance status fields differ from v1")
        object.__setattr__(self, "read_set", CANDIDATE_READ_SET_V1)
        object.__setattr__(self, "metadata", frozen_metadata)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "source_sha256": self.source_sha256,
            "source_bytes": self.source_bytes,
            "ast_nodes": self.ast_nodes,
            "read_set": list(self.read_set),
            "metadata": _thaw_internal_json(self.metadata),
            "validation_scope": self.validation_scope,
            "static_accepted": self.static_accepted,
            "dynamic_validation_status": self.dynamic_validation_status,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> StaticAcceptanceReceiptV1:
        expected = {
            "schema_version",
            "source_sha256",
            "source_bytes",
            "ast_nodes",
            "read_set",
            "metadata",
            "validation_scope",
            "static_accepted",
            "dynamic_validation_status",
        }
        if (
            type(value) is not dict
            or any(type(key) is not str for key in value)
            or set(value) != expected
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
        ):
            raise StaticValidationError("static acceptance receipt keys or version differ")
        read_set = value["read_set"]
        metadata = value["metadata"]
        if type(read_set) is not list or any(type(item) is not str for item in read_set):
            raise StaticValidationError("static acceptance read_set must be a string list")
        if type(metadata) is not dict:
            raise StaticValidationError("static acceptance metadata must be an object")
        return cls(
            source_sha256=value["source_sha256"],  # type: ignore[arg-type]
            source_bytes=value["source_bytes"],  # type: ignore[arg-type]
            ast_nodes=value["ast_nodes"],  # type: ignore[arg-type]
            read_set=tuple(read_set),
            metadata=metadata,
            validation_scope=value["validation_scope"],  # type: ignore[arg-type]
            static_accepted=value["static_accepted"],  # type: ignore[arg-type]
            dynamic_validation_status=value["dynamic_validation_status"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class StaticallyAcceptedTaskTermSourceV1:
    source_bytes: bytes
    receipt: StaticAcceptanceReceiptV1

    def __post_init__(self) -> None:
        if type(self.source_bytes) is not bytes:
            raise StaticValidationError("statically accepted source must retain exact bytes")
        if type(self.receipt) is not StaticAcceptanceReceiptV1:
            raise StaticValidationError("statically accepted source requires its exact receipt")
        _encoded, current_receipt = _derive_static_acceptance(self.source_bytes)
        if current_receipt.canonical_bytes != self.receipt.canonical_bytes:
            raise StaticValidationError("static acceptance receipt is stale or differs from source")

    @property
    def metadata(self) -> Mapping[str, object]:
        return self.receipt.metadata


def _derive_static_acceptance(
    source: bytes | str,
) -> tuple[bytes, StaticAcceptanceReceiptV1]:
    encoded, module, metadata = parse_and_validate_task_term_source(source)
    receipt = StaticAcceptanceReceiptV1(
        source_sha256=hashlib.sha256(encoded).hexdigest(),
        source_bytes=len(encoded),
        ast_nodes=sum(1 for _node in ast.walk(module)),
        read_set=CANDIDATE_READ_SET_V1,
        metadata=metadata,
    )
    return encoded, receipt


def statically_validate_task_term_source(
    source: bytes | str,
) -> StaticallyAcceptedTaskTermSourceV1:
    encoded, receipt = _derive_static_acceptance(source)
    return StaticallyAcceptedTaskTermSourceV1(
        source_bytes=encoded,
        receipt=receipt,
    )


def assert_static_acceptance_current(
    accepted: StaticallyAcceptedTaskTermSourceV1,
) -> StaticallyAcceptedTaskTermSourceV1:
    """Reparse bytes and reject stale or forged static acceptance data."""

    if type(accepted) is not StaticallyAcceptedTaskTermSourceV1:
        raise StaticValidationError("a statically accepted task-term source is required")
    return StaticallyAcceptedTaskTermSourceV1(
        source_bytes=accepted.source_bytes,
        receipt=accepted.receipt,
    )


__all__ = [
    "DYNAMIC_VALIDATION_NOT_PERFORMED",
    "MAX_AST_NODES",
    "MAX_CONDITIONAL_DEPTH",
    "MAX_SOURCE_BYTES",
    "SAFE_FUNCTION_NAMES",
    "STATIC_VALIDATION_SCOPE_V1",
    "StaticAcceptanceReceiptV1",
    "StaticValidationError",
    "StaticallyAcceptedTaskTermSourceV1",
    "assert_static_acceptance_current",
    "parse_and_validate_task_term_source",
    "statically_validate_task_term_source",
]
