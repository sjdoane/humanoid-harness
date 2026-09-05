from __future__ import annotations

import struct

import pytest

from oracle_composition.rewards.contract import CandidateTaskInputsV1
from oracle_composition.rewards.static_validation import (
    StaticValidationError,
    validate_task_term_source,
)

BENIGN = b"""NAME = "symmetric_target_speed"\nVERSION = 1\n\ndef task_term(x):\n    error = abs(x.com_x_velocity_m_s - x.target_speed_m_s)\n    return 1.0 - min(1.0, error)\n"""


def _reject(source: bytes | str, match: str | None = None) -> None:
    with pytest.raises(StaticValidationError, match=match):
        validate_task_term_source(source)


def test_benign_candidate_passes_with_bitwise_identical_repeated_results() -> None:
    program = validate_task_term_source(BENIGN)
    probe = CandidateTaskInputsV1(1.25, 1.0)
    first = struct.pack(">d", program.task_term(probe))
    program.task_term(CandidateTaskInputsV1(-2.0, 1.5))
    second = struct.pack(">d", program.task_term(probe))
    assert first == second
    assert program.receipt.passed is True
    assert program.receipt.read_set == ("com_x_velocity_m_s", "target_speed_m_s")
    assert len(program.receipt.sha256) == 64


def test_literal_metadata_is_deeply_immutable() -> None:
    source = b'METADATA = {"nested": [1, 2]}\n\ndef task_term(x):\n    return 0.0\n'
    program = validate_task_term_source(source)
    nested = program.metadata["METADATA"]
    with pytest.raises(TypeError):
        nested["nested"] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        nested["nested"][0] = 3  # type: ignore[index]


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


def test_rejects_nan_inf_wrong_type_and_output_envelope() -> None:
    attacks = (
        "def task_term(x):\n return 1e309",
        "def task_term(x):\n return -1e309",
        "def task_term(x):\n return 1",
        "def task_term(x):\n return True",
        "def task_term(x):\n return 1000.0001",
        "def task_term(x):\n return sqrt(-1.0)",
    )
    for attack in attacks:
        _reject(attack)


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
        validate_task_term_source(b" " * (16 * 1024 + 1))
    assignments = "\n".join(f" a{i} = {i}.0" for i in range(200))
    _reject(f"def task_term(x):\n{assignments}\n return 0.0", "512 AST")
