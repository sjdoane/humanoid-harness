"""Negative coverage for the composition-oracle contract."""

from __future__ import annotations

import copy

import pytest

from oracle_composition.harness import contract as contract_module
from oracle_composition.harness.contract import (
    ALLOWED_SIGNALS,
    MAX_GUARD_CHARACTERS,
    MAX_ORACLE_STATES,
    MAX_ORACLE_TRANSITIONS,
    GuardExpression,
    OracleContractError,
    OracleMachine,
    decode_json_object,
    oracle_program_from_dict,
)


def _oracle() -> dict[str, object]:
    return {
        "behaviors": ["expert", "simple"],
        "evidence_class": "exploratory_oracle_cycle",
        "initial": "start",
        "oracle_id": "test_oracle",
        "schema_version": 1,
        "states": {
            "start": {"behavior": "expert", "min_dwell": 1},
            "stop": {"behavior": "simple", "min_dwell": 0},
        },
        "transitions": [{"from": "start", "to": "stop", "guard": "t >= 1", "priority": 0}],
    }


@pytest.mark.parametrize(
    ("guard", "message"),
    [
        ("mystery > 0", "unknown guard signal"),
        ("abs(v_x) > 0", "calls are forbidden"),
        ("v_x.real > 0", "attributes are forbidden"),
        ("v_x < 1e309", "guard constant must be finite"),
    ],
)
def test_guard_contract_rejects_unknown_unsafe_and_nonfinite_expressions(
    guard: str, message: str
) -> None:
    value = _oracle()
    value["transitions"][0]["guard"] = guard
    with pytest.raises(OracleContractError, match=message):
        oracle_program_from_dict(value, available_behaviors=("expert", "simple"))


def test_oracle_rejects_unreachable_state() -> None:
    value = _oracle()
    value["states"]["orphan"] = {"behavior": "expert", "min_dwell": 1}
    with pytest.raises(OracleContractError, match="unreachable states"):
        oracle_program_from_dict(value, available_behaviors=("expert", "simple"))


def test_oracle_rejects_undefined_behavior() -> None:
    value = _oracle()
    value["states"]["stop"]["behavior"] = "medium"
    with pytest.raises(OracleContractError, match="undefined behavior"):
        oracle_program_from_dict(value, available_behaviors=("expert", "simple"))


def test_oracle_rejects_duplicate_outgoing_priority() -> None:
    value = _oracle()
    value["transitions"].append({"from": "start", "to": "stop", "guard": "v_x < 0", "priority": 0})
    with pytest.raises(OracleContractError, match="duplicate transition priority"):
        oracle_program_from_dict(value, available_behaviors=("expert", "simple"))


def test_oracle_rejects_missing_initial() -> None:
    value = _oracle()
    del value["initial"]
    with pytest.raises(OracleContractError, match="keys differ"):
        oracle_program_from_dict(value, available_behaviors=("expert", "simple"))


def test_oracle_rejects_zero_dwell_cycle() -> None:
    value = copy.deepcopy(_oracle())
    value["states"]["start"]["min_dwell"] = 0
    value["transitions"].append({"from": "stop", "to": "start", "guard": "t >= 2", "priority": 0})
    with pytest.raises(OracleContractError, match="zero-dwell transition cycle"):
        oracle_program_from_dict(value, available_behaviors=("expert", "simple"))


@pytest.mark.parametrize("constant", [b"NaN", b"Infinity", b"-Infinity"])
def test_json_decoder_rejects_raw_nonfinite_constants(constant: bytes) -> None:
    with pytest.raises(OracleContractError, match="non-finite JSON constant"):
        decode_json_object(b'{"value":' + constant + b"}")


def test_json_decoder_normalizes_duplicate_huge_and_oversize_failures() -> None:
    with pytest.raises(OracleContractError, match="duplicate JSON key"):
        decode_json_object(b'{"a":1,"a":2}')
    with pytest.raises(OracleContractError, match="not valid bounded JSON"):
        decode_json_object(b'{"value":' + b"9" * 5000 + b"}")
    with pytest.raises(OracleContractError, match="at most"):
        decode_json_object(b'{"value":"' + b"x" * (256 * 1024) + b'"}')
    deeply_nested = b'{"value":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}"
    with pytest.raises(OracleContractError, match="depth limit"):
        decode_json_object(deeply_nested)


@pytest.mark.parametrize(
    "error",
    [
        RecursionError("depth"),
        MemoryError("memory"),
        OverflowError("overflow"),
        ValueError("limit"),
    ],
)
def test_json_decoder_normalizes_runtime_decoder_limits(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fail(*_args: object, **_kwargs: object) -> object:
        raise error

    monkeypatch.setattr(contract_module.json, "loads", fail)
    with pytest.raises(OracleContractError, match="not valid bounded JSON"):
        decode_json_object(b"{}")


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("t > 0" + " " * MAX_GUARD_CHARACTERS, "bounded text"),
        (" or ".join(["t > 0"] * 30), "node limit"),
        ("not " * 40 + "t > 0", "depth limit"),
    ],
)
def test_guard_byte_node_and_depth_limits(source: str, message: str) -> None:
    with pytest.raises(OracleContractError, match=message):
        GuardExpression.parse(source)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("t == 0 or v_x > 100", True),
        ("t != 0 and v_x > 0", False),
    ],
)
def test_guard_boolean_operations_have_short_circuit_semantics(source: str, expected: bool) -> None:
    signals = {name: 0.0 for name in ALLOWED_SIGNALS}
    assert GuardExpression.parse(source).evaluate(signals) is expected


def test_oracle_rejects_unicode_identifier_and_structural_caps() -> None:
    value = _oracle()
    value["oracle_id"] = "tést"
    with pytest.raises(OracleContractError, match=r"ASCII|snake-case"):
        oracle_program_from_dict(value, available_behaviors=("expert", "simple"))

    value = _oracle()
    value["states"] = {
        f"s{index}": {"behavior": "expert", "min_dwell": 1}
        for index in range(MAX_ORACLE_STATES + 1)
    }
    value["initial"] = "s0"
    with pytest.raises(OracleContractError, match="state limit"):
        oracle_program_from_dict(value, available_behaviors=("expert", "simple"))

    value = _oracle()
    value["transitions"] = [
        {"from": "start", "to": "stop", "guard": "t >= 1", "priority": index}
        for index in range(MAX_ORACLE_TRANSITIONS + 1)
    ]
    with pytest.raises(OracleContractError, match="transition limit"):
        oracle_program_from_dict(value, available_behaviors=("expert", "simple"))


def _signals(machine: OracleMachine, *, z_root: float) -> dict[str, float | int]:
    return {
        "dwell": machine.dwell,
        "t": 0,
        "torso_up": 1.0,
        "v_target": 1.0,
        "v_x": 1.0,
        "x_travelled": 0.0,
        "z_root": z_root,
    }


def _recovery_oracle() -> dict[str, object]:
    value = _oracle()
    value["recovery"] = {
        "behavior": "simple",
        "guard": "z_root < 1.0",
        "max_duration": 2,
        "min_dwell": 2,
        "reentry_dwell": 2,
        "rejoin": "suspended_state_dwell_reset",
    }
    return value


def test_new_recovery_requires_complete_liveness_semantics() -> None:
    value = _oracle()
    value["recovery"] = {"behavior": "simple", "guard": "z_root < 1.0"}
    with pytest.raises(OracleContractError, match="recovery requires"):
        oracle_program_from_dict(value, available_behaviors=("expert", "simple"))


def test_recovery_minimum_dwell_reentry_hysteresis_and_maximum_duration() -> None:
    program = oracle_program_from_dict(_recovery_oracle(), available_behaviors=("expert", "simple"))
    machine = OracleMachine(program)
    assert machine.decide(_signals(machine, z_root=0.9)).recovery_entered
    machine.advance()
    assert machine.decide(_signals(machine, z_root=1.1)).reason == "recovery_hold"
    machine.advance()
    assert machine.decide(_signals(machine, z_root=1.1)).recovery_exited
    machine.advance()
    for _ in range(2):
        assert machine.decide(_signals(machine, z_root=0.9)).reason == "recovery_reentry_dwell"
        machine.advance()
    assert machine.decide(_signals(machine, z_root=0.9)).recovery_entered

    sink = OracleMachine(program)
    assert sink.decide(_signals(sink, z_root=0.9)).recovery_entered
    sink.advance()
    assert sink.decide(_signals(sink, z_root=0.9)).reason == "recovery_hold"
    sink.advance()
    with pytest.raises(OracleContractError, match="maximum duration"):
        sink.decide(_signals(sink, z_root=0.9))


@pytest.mark.parametrize("field", sorted(ALLOWED_SIGNALS))
def test_machine_rejects_every_nonfinite_runtime_signal(field: str) -> None:
    program = oracle_program_from_dict(_oracle(), available_behaviors=("expert", "simple"))
    machine = OracleMachine(program)
    signals = _signals(machine, z_root=1.4)
    signals[field] = float("nan")
    with pytest.raises(OracleContractError):
        machine.decide(signals)
