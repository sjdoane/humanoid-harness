"""Keep the Study014 screen distinct from the unchanged task gates."""

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[2] / "experiments/014_g1_finite_depth_reward/score.py"
    spec = importlib.util.spec_from_file_location("study014_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _criteria():
    return _module().criteria


@pytest.fixture
def objective():
    return {
        "region": {
            "minimum_root_height_m": 0.504942203119977 - 0.02,
            "posture_compliant_fraction": 31 / 60,
            "entry_observed": True,
            "exit_observed_after_entry": True,
            "inside_sample_count": 25,
        },
        "duration_seconds": 20,
        "fall_count": 0,
        "oracle_diagnostics": {"observed_switch_count": 2},
        "speed": {"inside_mean_speed_target_deviation_m_s": 0.18357610804822067 + 0.05},
        "development_gate_results": {
            "joint_position_rmse_p95": True,
            "roll_pitch_rmse_p95": True,
            "inside_mean_speed_target": False,
        },
    }


def test_exact_screen_boundary_does_not_promote_task(objective):
    assert all(_criteria()(objective).values())
    assert not objective["development_gate_results"]["inside_mean_speed_target"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("minimum_root_height_m", None),
        ("minimum_root_height_m", 0.48495),
        ("posture_compliant_fraction", None),
        ("posture_compliant_fraction", 0.51666),
        ("inside_sample_count", 24),
        ("exit_observed_after_entry", False),
    ],
)
def test_missing_or_failing_region_rejects(objective, field, value):
    objective["region"][field] = value
    assert not all(_criteria()(objective).values())


@pytest.mark.parametrize("value", [None, 0.23358])
def test_missing_or_excess_speed_rejects(objective, value):
    objective["speed"]["inside_mean_speed_target_deviation_m_s"] = value
    assert not all(_criteria()(objective).values())


def test_deep_fall_is_not_success(objective):
    objective["fall_count"] = 1
    assert not all(_criteria()(objective).values())


def test_reward_only_reset_difference_is_declared():
    a = {
        "reset": {"reward_sha256": "r1", "runtime": "fixed"},
        "training_reward_sum_not_success_metric": 10,
        "objective": {"same": True},
    }
    b = {
        "reset": {"reward_sha256": "r4", "runtime": "fixed"},
        "training_reward_sum_not_success_metric": 8,
        "objective": {"same": True},
    }
    verify = _module().verify_zero_evaluation
    verify(a, b, "r1", "r4")
    with pytest.raises(AssertionError):
        verify(a, b, "r1", "wrong")
    b["reset"]["runtime"] = "changed"
    with pytest.raises(AssertionError):
        verify(a, b, "r1", "r4")
    b["reset"]["runtime"] = "fixed"
    b["objective"] = {"same": False}
    with pytest.raises(AssertionError):
        verify(a, b, "r1", "r4")
