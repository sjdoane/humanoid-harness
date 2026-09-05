from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from oracle_composition.rewards.contract import CandidateTaskInputsV1, RewardContractError
from oracle_composition.rewards.task_inputs_v2 import (
    CandidateTaskInputsV2,
    TaskInputsV2Error,
    task_input_contract_v2,
    validate_task_inputs_v2,
)


def test_fixed_target_and_canonical_bytes() -> None:
    inputs = CandidateTaskInputsV2(2, 3)
    assert validate_task_inputs_v2(inputs) == (2.0, 3.0)
    assert inputs.canonical_bytes == b'{"com_x_velocity_m_s":2.0,"target_speed_m_s":3.0}'
    assert type(inputs.com_x_velocity_m_s) is float
    with pytest.raises(FrozenInstanceError):
        inputs.target_speed_m_s = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("value", [True, "3", None, math.nan, math.inf, -math.inf, 10**400])
def test_rejects_unsafe_or_nonfinite_values(value: object) -> None:
    with pytest.raises(TaskInputsV2Error):
        CandidateTaskInputsV2(value, 3.0)  # type: ignore[arg-type]
    with pytest.raises(TaskInputsV2Error):
        CandidateTaskInputsV2(1.0, value)  # type: ignore[arg-type]


@pytest.mark.parametrize("target", [0.0, 0.5, 1.0, 1.5, 0.8853599908576963, 5.520768616125457])
def test_old_targets_and_profile_targets_are_not_t2(target: float) -> None:
    with pytest.raises(TaskInputsV2Error, match=r"must equal 3\.0"):
        CandidateTaskInputsV2(1.0, target)


def test_v1_remains_separate() -> None:
    original = CandidateTaskInputsV1(1.0, 1.0)
    with pytest.raises(TaskInputsV2Error, match="exact CandidateTaskInputsV2"):
        validate_task_inputs_v2(original)
    with pytest.raises(RewardContractError):
        CandidateTaskInputsV1(1.0, 3.0)


def test_callback_objects_and_subclasses_are_rejected_without_callbacks() -> None:
    class HostileFloat(float):
        def __float__(self) -> float:
            raise AssertionError("conversion callback ran")

    class HostileInput(CandidateTaskInputsV2):
        def __getattribute__(self, name: str) -> object:
            raise AssertionError("attribute callback ran")

    with pytest.raises(TaskInputsV2Error):
        CandidateTaskInputsV2(HostileFloat(1), 3.0)
    with pytest.raises(TaskInputsV2Error):
        validate_task_inputs_v2(object.__new__(HostileInput))
    forged = object.__new__(CandidateTaskInputsV2)
    with pytest.raises(TaskInputsV2Error, match="incomplete"):
        validate_task_inputs_v2(forged)
    object.__setattr__(forged, "com_x_velocity_m_s", HostileFloat(1))
    object.__setattr__(forged, "target_speed_m_s", 3.0)
    with pytest.raises(TaskInputsV2Error):
        validate_task_inputs_v2(forged)


def test_contract_schema_is_fresh_and_preserves_stock_com_semantics() -> None:
    contract = task_input_contract_v2()
    assert contract["velocity_definition"] == "stock_mass_center_delta_x_over_control_period"
    assert contract["control_period_seconds"] == 0.015
    assert contract["target_speed_m_s"] == 3.0
    contract["fields"].append("root_x")  # type: ignore[union-attr]
    assert "root_x" not in task_input_contract_v2()["fields"]
