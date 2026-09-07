from __future__ import annotations

import importlib.util
import math
from copy import deepcopy
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "experiments/018_g1_four_state_reward_loop/criteria.py"
SPEC = importlib.util.spec_from_file_location("study018_criteria_tests", PATH)
criteria = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(criteria)


def objective():
    return {
        "duration_seconds": 20.0,
        "fall_count": 0,
        "finish_condition_observed": True,
        "maximum_lateral_error_m": 2.0,
        "oracle_diagnostics": {
            "executed_mode_counts": {"before": 100, "inside": 100, "rise": 100, "after": 700},
            "observed_switch_count": 3,
        },
        "region": {
            "entry_observed": True,
            "exit_observed_after_entry": True,
            "inside_sample_count": 100,
            "posture_compliant_fraction": 0.80,
            "minimum_root_height_m": 0.54,
        },
        "tracking": {"joint_position_rmse_rad_p95": 0.35, "roll_pitch_rmse_rad_p95": 0.25},
        "speed": {"mean_absolute_error_m_s": 0.35, "inside_mean_speed_target_deviation_m_s": 0.10},
        "development_gate_results": {"inside_posture_dip": False, "maximum_lateral_error": False},
    }


def baseline(value):
    return criteria.baseline_criteria(value, nonfoot_contact_count=0, after_reentry_count=0)


def test_baseline_boundary_and_original_failed_gates_are_not_rewritten():
    value = objective()
    original = deepcopy(value)
    assert all(baseline(value).values())
    assert value == original
    value["maximum_lateral_error_m"] = 0.75
    assert baseline(value)["lateral_headroom_over_0p75m"] is False
    value["maximum_lateral_error_m"] = math.nextafter(0.75, math.inf)
    assert all(baseline(value).values())


@pytest.mark.parametrize(
    "parent,key,value,gate",
    [
        (None, "fall_count", 1, "full_20s_without_fall"),
        (None, "duration_seconds", 19.98, "full_20s_without_fall"),
        ("region", "posture_compliant_fraction", 0.74999, "compliance_at_least_0p75"),
        ("region", "inside_sample_count", 24, "minimum_25_inside_samples"),
        ("region", "posture_compliant_fraction", None, "compliance_at_least_0p75"),
        ("speed", "mean_absolute_error_m_s", 0.35001, "speed_mae_at_most_0p35"),
        ("oracle_diagnostics", "observed_switch_count", 2, "three_switches"),
    ],
)
def test_baseline_rejects_each_regression(parent, key, value, gate):
    record = objective()
    (record if parent is None else record[parent])[key] = value
    assert baseline(record)[gate] is False


def test_contact_and_physical_reentry_counts_are_separate_gates():
    rows = criteria.baseline_criteria(objective(), nonfoot_contact_count=1, after_reentry_count=1)
    assert not rows["no_nonfoot_ground_contact"]
    assert not rows["no_after_region_reentry"]
    value = objective()
    del value["oracle_diagnostics"]["executed_mode_counts"]["after"]
    assert baseline(value)["all_four_modes_executed"] is False
    with pytest.raises(ValueError, match="count"):
        criteria.baseline_criteria(value, nonfoot_contact_count=False, after_reentry_count=0)


def frames(angles, modes):
    return [
        {
            "executed_mode": mode,
            "executed_phase_seconds": 0.02 * index,
            "metrics": {
                "control_step": index + 1,
                "heading_error_signed_rad": angle,
                "heading_error_rad": abs(angle),
            },
        }
        for index, (angle, mode) in enumerate(zip(angles, modes, strict=True))
    ]


def test_signed_heading_unwraps_before_selecting_after_and_retains_winding():
    result = criteria.heading_diagnostics(
        frames([2.9, -2.9, -1.0, 1.0], ["before", "inside", "rise", "after"])
    )
    assert result["after_heading_max_abs_unwrapped_rad"] == pytest.approx(2 * math.pi + 1.0)
    assert result["after_first_heading_signed_rad"] == 1.0
    assert result["after_first_control_step"] == 4
    assert result["after_mode_sample_count"] == 1
    result = criteria.heading_diagnostics(frames([-2.9, 2.9], ["before", "after"]))
    assert result["after_heading_max_abs_unwrapped_rad"] == pytest.approx(2 * math.pi - 2.9)


def test_early_stop_has_unavailable_after_heading_not_fabricated_zero():
    result = criteria.heading_diagnostics(frames([0.1], ["before"]))
    assert result["after_heading_max_abs_unwrapped_rad"] is None
    assert result["after_mode_sample_count"] == 0


@pytest.mark.parametrize("bad", [None, math.nan, math.inf, True, 4.0])
def test_invalid_signed_heading_fails_closed(bad):
    rows = frames([0.1], ["after"])
    rows[0]["metrics"]["heading_error_signed_rad"] = bad
    with pytest.raises(ValueError):
        criteria.heading_diagnostics(rows)


def test_absolute_heading_cannot_substitute_for_missing_signed_signal():
    rows = frames([0.1], ["after"])
    del rows[0]["metrics"]["heading_error_signed_rad"]
    with pytest.raises(KeyError):
        criteria.heading_diagnostics(rows)


def test_pair_requires_relative_gain_without_compliance_or_heading_regression():
    first, second = objective(), objective()
    second["maximum_lateral_error_m"] = 1.4
    heading = {"after_heading_max_abs_unwrapped_rad": 1.0}

    def evaluate(other=heading):
        return criteria.pair_criteria(
            first, second, heading, other, nonfoot_contact_count=0, after_reentry_count=0
        )

    assert all(evaluate().values())
    second["region"]["posture_compliant_fraction"] = math.nextafter(0.8, 0)
    assert not evaluate()["compliance_no_regression"]
    second["region"]["posture_compliant_fraction"] = 0.8
    second["maximum_lateral_error_m"] = math.nextafter(1.4, math.inf)
    assert not evaluate()["lateral_reduction_30_percent"]
    assert not evaluate({"after_heading_max_abs_unwrapped_rad": None})[
        "after_heading_no_regression"
    ]
    assert not evaluate({"after_heading_max_abs_unwrapped_rad": 1.001})[
        "after_heading_no_regression"
    ]


def test_nonfinite_objective_is_invalid_not_an_ordinary_failed_gate():
    value = objective()
    value["speed"]["mean_absolute_error_m_s"] = math.nan
    with pytest.raises(ValueError, match="finite"):
        baseline(value)
