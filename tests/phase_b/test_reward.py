from __future__ import annotations

import numpy as np
import pytest

from oracle_composition.phase_b.contracts import (
    PhaseBContractError,
    TargetSpeedRewardSpec,
    TrackingOnlyRewardSpec,
)
from oracle_composition.phase_b.reference_runtime import tracking_state_from_reference_row
from oracle_composition.phase_b.reward import (
    compose_registered_reward,
    compose_tracking_only_reward,
)
from oracle_composition.phase_b.task_input_admission import (
    MEASUREMENT_ORIGIN,
    admit_task_inputs_v2,
)
from oracle_composition.tracking.reward import TrackingRewardConfig


def _row() -> np.ndarray:
    value = np.zeros(45, dtype="<f8")
    value[0] = 1.4
    value[1] = 1.0
    return value


def _task_inputs(velocity: float = 0.0) -> object:
    return admit_task_inputs_v2(
        com_x_velocity_m_s=velocity,
        measurement_origin=MEASUREMENT_ORIGIN,
        cadence_seconds=0.015,
    )


def test_reward_streams_remain_separate_and_stock_is_not_added() -> None:
    target = _row()
    result = compose_tracking_only_reward(
        state=tracking_state_from_reference_row(target),
        hidden_reference_target=target,
        ignored_stock_reward=1000.0,
        task_inputs=_task_inputs(),
        specification=TrackingOnlyRewardSpec(),
    )
    assert result.r_track == pytest.approx(1.0)
    assert result.r_task == 0.0
    assert result.r_train == result.r_track + result.r_task
    assert result.r_train != result.r_track + result.ignored_stock_reward
    assert set(result.to_dict()) == {
        "ignored_stock_reward",
        "r_task",
        "r_track",
        "r_train",
        "tracking_errors",
        "tracking_reward_components",
    }


def test_nonfinite_stock_telemetry_and_wrong_target_are_refused() -> None:
    target = _row()
    with pytest.raises(PhaseBContractError, match="stock reward"):
        compose_tracking_only_reward(
            state=tracking_state_from_reference_row(target),
            hidden_reference_target=target,
            ignored_stock_reward=float("nan"),
            task_inputs=_task_inputs(),
            specification=TrackingOnlyRewardSpec(),
        )
    with pytest.raises(PhaseBContractError, match=r"float64\[45\]"):
        compose_tracking_only_reward(
            state=tracking_state_from_reference_row(target),
            hidden_reference_target=target.astype("<f4"),
            ignored_stock_reward=0.0,
            task_inputs=_task_inputs(),
            specification=TrackingOnlyRewardSpec(),
        )


def test_tracking_reward_scale_drift_is_refused() -> None:
    target = _row()
    with pytest.raises(PhaseBContractError, match="frozen scales"):
        compose_tracking_only_reward(
            state=tracking_state_from_reference_row(target),
            hidden_reference_target=target,
            ignored_stock_reward=0.0,
            task_inputs=_task_inputs(),
            specification=TrackingOnlyRewardSpec(),
            config=TrackingRewardConfig(root_height_scale_m=0.21),
        )


@pytest.mark.parametrize("velocity", [-25.0, 25.0])
def test_task_input_admission_accepts_inclusive_boundaries(velocity: float) -> None:
    assert _task_inputs(velocity).inputs.com_x_velocity_m_s == velocity


@pytest.mark.parametrize("velocity", [-25.000001, 25.000001])
def test_task_input_admission_rejects_out_of_range_at_construction(velocity: float) -> None:
    with pytest.raises(PhaseBContractError, match="inclusive"):
        _task_inputs(velocity)


def test_task_input_admission_rejects_origin_cadence_and_consumption_forgery() -> None:
    with pytest.raises(PhaseBContractError, match="origin"):
        admit_task_inputs_v2(
            com_x_velocity_m_s=0.0,
            measurement_origin="root_x_delta",
            cadence_seconds=0.015,
        )
    with pytest.raises(PhaseBContractError, match=r"0\.015"):
        admit_task_inputs_v2(
            com_x_velocity_m_s=0.0,
            measurement_origin=MEASUREMENT_ORIGIN,
            cadence_seconds=0.02,
        )
    with pytest.raises(PhaseBContractError, match="numeric representation"):
        admit_task_inputs_v2(
            com_x_velocity_m_s=0.0,
            measurement_origin=MEASUREMENT_ORIGIN,
            cadence_seconds=0.015,
            numeric_representation="float32_from_info_velocity",
        )
    admitted = _task_inputs()
    object.__setattr__(admitted.inputs, "com_x_velocity_m_s", 25.000001)
    with pytest.raises(PhaseBContractError, match="inclusive"):
        compose_tracking_only_reward(
            state=tracking_state_from_reference_row(_row()),
            hidden_reference_target=_row(),
            ignored_stock_reward=0.0,
            task_inputs=admitted,
            specification=TrackingOnlyRewardSpec(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("measurement_origin", "root_x_delta"),
        ("numeric_representation", "float32"),
        ("cadence_seconds", 0.02),
        ("admission_id", "unreviewed/v1"),
    ],
)
def test_reward_consumption_rejects_forged_task_input_certificate(
    field: str, value: object
) -> None:
    admitted = _task_inputs()
    object.__setattr__(admitted, field, value)
    with pytest.raises(PhaseBContractError, match="certificate differs"):
        compose_tracking_only_reward(
            state=tracking_state_from_reference_row(_row()),
            hidden_reference_target=_row(),
            ignored_stock_reward=0.0,
            task_inputs=admitted,
            specification=TrackingOnlyRewardSpec(),
        )


def test_f2_registered_formula_consumes_certified_t2_input() -> None:
    target = _row()
    result = compose_registered_reward(
        state=tracking_state_from_reference_row(target),
        hidden_reference_target=target,
        ignored_stock_reward=500.0,
        task_inputs=_task_inputs(3.0),
        specification=TargetSpeedRewardSpec(alpha=4.0, beta=-10.0),
    )
    assert result.r_track == pytest.approx(1.0)
    assert result.r_task == pytest.approx(-5.0)
    assert result.r_train == pytest.approx(-4.0)
    assert result.ignored_stock_reward == 500.0


def test_f2_registered_formula_refuses_missing_consumption_certificate() -> None:
    target = _row()
    with pytest.raises(PhaseBContractError, match="admission"):
        compose_registered_reward(
            state=tracking_state_from_reference_row(target),
            hidden_reference_target=target,
            ignored_stock_reward=0.0,
            task_inputs=object(),
            specification=TargetSpeedRewardSpec(alpha=1.0, beta=0.0),
        )
