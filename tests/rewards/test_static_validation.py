from __future__ import annotations

import pytest

from oracle_composition.rewards.static_validation import (
    MAX_CONDITIONAL_DEPTH,
    StaticValidationError,
    statically_validate_task_term_source,
)

BENIGN = b"""NAME = "symmetric_target_speed"\nVERSION = 1\n\ndef task_term(x):\n    error = abs(x.com_x_velocity_m_s - x.target_speed_m_s)\n    return 1.0 - min(1.0, error)\n"""


def _reject(source: bytes | str, match: str | None = None) -> None:
    with pytest.raises(StaticValidationError, match=match):
        statically_validate_task_term_source(source)


def test_benign_candidate_is_statically_accepted_without_dynamic_claim() -> None:
    source = statically_validate_task_term_source(BENIGN)
    assert not hasattr(source, "task_term")
    assert source.receipt.static_accepted is True
    assert source.receipt.validation_scope == "source_bytes_and_ast_only"
    assert source.receipt.dynamic_validation_status == "not_performed"
    assert source.receipt.read_set == ("com_x_velocity_m_s", "target_speed_m_s")
    assert len(source.receipt.sha256) == 64
    assert "determinism_sha256" not in source.receipt.to_dict()
    assert "passed" not in source.receipt.to_dict()


def test_literal_metadata_is_deeply_immutable() -> None:
    source = b'METADATA = {"nested": [1, 2]}\n\ndef task_term(x):\n    return 0.0\n'
    program = statically_validate_task_term_source(source)
    nested = program.metadata["METADATA"]
    with pytest.raises(TypeError):
        nested["nested"] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        nested["nested"][0] = 3  # type: ignore[index]


@pytest.mark.parametrize(
    "metadata",
    [
        "METADATA = object()",
        "METADATA = {1: 'non-string-key'}",
        "_PRIVATE = 1",
        "VERSION = 1\nVERSION = 2",
    ],
)
def test_invalid_or_duplicated_metadata_is_rejected(metadata: str) -> None:
    _reject(f"{metadata}\n\ndef task_term(x):\n    return 0.0\n", "metadata")


def test_rejects_imports_dynamic_calls_and_dunders() -> None:
    attacks = (
        "import os\ndef task_term(x):\n return 0.0",
        "from math import sin\ndef task_term(x):\n return 0.0",
        "def task_term(x):\n return getattr(x, 'com_x_velocity_m_s')",
        "def task_term(x):\n return x.__class__",
        "def task_term(x):\n return (lambda y: y)(1.0)",
    )
    for attack in attacks:
        _reject(attack)


def test_rejects_unknown_signal_reward_info_and_evaluator_access() -> None:
    for name in ("qpos", "reward_info", "reward_forward", "protected_evaluator"):
        _reject(f"def task_term(x):\n return x.{name}", "cannot read signal")


def test_rejects_random_time_global_counter_and_mutable_state() -> None:
    attacks = (
        "def task_term(x):\n return random.random()",
        "def task_term(x):\n return time.time()",
        "counter = 0\ndef task_term(x):\n global counter\n counter += 1\n return float(counter)",
        "state = []\ndef task_term(x):\n state.append(1)\n return 0.0",
        "def task_term(x):\n nonlocal counter\n return 0.0",
    )
    for attack in attacks:
        _reject(attack)


def test_rejects_input_mutation_recursion_loop_and_allocation_ast() -> None:
    attacks = (
        "def task_term(x):\n x.com_x_velocity_m_s = 1.0\n return 0.0",
        "def task_term(x):\n return task_term(x)",
        "def task_term(x):\n while True:\n  pass\n return 0.0",
        "def task_term(x):\n for y in (1, 2):\n  pass\n return 0.0",
        "def task_term(x):\n return [x.com_x_velocity_m_s]",
        "def task_term(x):\n return sum(y for y in (1.0,))",
        "def task_term(x):\n return x[0]",
        "def task_term(x):\n try:\n  return 0.0\n except:\n  return 1.0",
        "def task_term(x):\n with x:\n  return 0.0",
        "def task_term(x):\n yield 0.0",
        "def task_term(x):\n return 2.0 ** x.com_x_velocity_m_s",
        "def task_term(x):\n return 2.0 ** 17.0",
        "class Bad:\n pass\ndef task_term(x):\n return 0.0",
    )
    for attack in attacks:
        _reject(attack)


def test_static_check_rejects_nonfinite_and_boolean_literals() -> None:
    for attack in (
        "def task_term(x):\n return 1e309",
        "def task_term(x):\n return -1e309",
        "def task_term(x):\n return True",
    ):
        _reject(attack)


def test_runtime_only_output_failures_are_not_claimed_by_static_acceptance() -> None:
    runtime_only_failures = (
        "def task_term(x):\n return 1",
        "def task_term(x):\n return 1000.0001",
        "def task_term(x):\n return sqrt(-1.0)",
    )
    for source in runtime_only_failures:
        accepted = statically_validate_task_term_source(source)
        assert accepted.receipt.dynamic_validation_status == "not_performed"


def test_rejects_wrong_signature_async_decorators_defaults_and_fallthrough() -> None:
    attacks = (
        "async def task_term(x):\n return 0.0",
        "def task_term(x=1.0):\n return 0.0",
        "def task_term(x, y):\n return 0.0",
        "@staticmethod\ndef task_term(x):\n return 0.0",
        "def task_term(x):\n if x.com_x_velocity_m_s > 0.0:\n  return 1.0",
        "def other(x):\n return 0.0",
    )
    for attack in attacks:
        _reject(attack)


def test_rejects_non_utf8_oversize_and_ast_budget() -> None:
    _reject(b"\xff")
    with pytest.raises(StaticValidationError, match="16 KiB"):
        statically_validate_task_term_source(b" " * (16 * 1024 + 1))
    assignments = "\n".join(f" a{i} = {i}.0" for i in range(200))
    _reject(f"def task_term(x):\n{assignments}\n return 0.0", "512 AST")


def _statement_conditionals(depth: int, *, leaf_expression: str = "0.0") -> str:
    lines = ["def task_term(x):"]
    for level in range(depth):
        lines.append(f"{'    ' * (level + 1)}if x.com_x_velocity_m_s >= {float(level)!r}:")
    lines.append(f"{'    ' * (depth + 1)}return {leaf_expression}")
    for level in reversed(range(depth)):
        lines.append(f"{'    ' * (level + 1)}else:")
        lines.append(f"{'    ' * (level + 2)}return 0.0")
    return "\n".join(lines) + "\n"


def _ternary_conditionals(depth: int) -> str:
    expression = "0.0"
    for _ in range(depth):
        expression = f"(0.0 if x.com_x_velocity_m_s >= x.target_speed_m_s else {expression})"
    return f"def task_term(x):\n    return {expression}\n"


@pytest.mark.parametrize(
    "source",
    [
        _statement_conditionals(MAX_CONDITIONAL_DEPTH),
        _ternary_conditionals(MAX_CONDITIONAL_DEPTH),
        _statement_conditionals(4, leaf_expression=_ternary_conditionals(4).split("return ", 1)[1]),
    ],
    ids=("statement-if", "ternary-ifexp", "mixed"),
)
def test_conditional_depth_eight_is_accepted(source: str) -> None:
    assert statically_validate_task_term_source(source).receipt.static_accepted is True


@pytest.mark.parametrize(
    "source",
    [
        _statement_conditionals(MAX_CONDITIONAL_DEPTH + 1),
        _ternary_conditionals(MAX_CONDITIONAL_DEPTH + 1),
        _statement_conditionals(4, leaf_expression=_ternary_conditionals(5).split("return ", 1)[1]),
    ],
    ids=("statement-if", "ternary-ifexp", "mixed"),
)
def test_conditional_depth_nine_is_rejected(source: str) -> None:
    _reject(source, "conditional nesting")
