from __future__ import annotations

import numpy as np
import pytest

from oracle_composition.phase_b.contracts import PhaseBContractError, TrackingOnlyRewardSpec
from oracle_composition.phase_b.reference_runtime import tracking_state_from_reference_row
from oracle_composition.phase_b.reward import compose_tracking_only_reward
from oracle_composition.tracking.reward import TrackingRewardConfig


def _row() -> np.ndarray:
    value = np.zeros(45, dtype="<f8")
    value[0] = 1.4
    value[1] = 1.0
    return value


def test_reward_streams_remain_separate_and_stock_is_not_added() -> None:
    target = _row()
    result = compose_tracking_only_reward(
        state=tracking_state_from_reference_row(target),
        hidden_reference_target=target,
        ignored_stock_reward=1000.0,
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
            specification=TrackingOnlyRewardSpec(),
        )
    with pytest.raises(PhaseBContractError, match=r"float64\[45\]"):
        compose_tracking_only_reward(
            state=tracking_state_from_reference_row(target),
            hidden_reference_target=target.astype("<f4"),
            ignored_stock_reward=0.0,
            specification=TrackingOnlyRewardSpec(),
        )


def test_tracking_reward_scale_drift_is_refused() -> None:
    target = _row()
    with pytest.raises(PhaseBContractError, match="frozen scales"):
        compose_tracking_only_reward(
            state=tracking_state_from_reference_row(target),
            hidden_reference_target=target,
            ignored_stock_reward=0.0,
            specification=TrackingOnlyRewardSpec(),
            config=TrackingRewardConfig(root_height_scale_m=0.21),
        )
