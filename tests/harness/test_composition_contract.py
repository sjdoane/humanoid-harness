"""Negative coverage for the composition-oracle contract."""

from __future__ import annotations

import copy

import pytest

from oracle_composition.harness.contract import OracleContractError, oracle_program_from_dict


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
