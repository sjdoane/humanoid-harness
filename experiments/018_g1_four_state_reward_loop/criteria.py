"""Predata Study018 gates; inputs must first pass independent run verification."""

from __future__ import annotations

import math

MODES = frozenset({"before", "inside", "rise", "after"})


def _number(value: object) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("measurement must be finite or explicitly unavailable")
    return float(value)


def _at_most(value: object, ceiling: float) -> bool:
    number = _number(value)
    return number is not None and number <= ceiling


def _at_least(value: object, floor: float) -> bool:
    number = _number(value)
    return number is not None and number >= floor


def _count(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("diagnostic count must be a nonnegative integer")
    return value


def heading_diagnostics(frames: list[dict]) -> dict:
    """Unwrap signed reset-relative heading before selecting after-mode samples."""
    if type(frames) is not list or not frames:
        raise ValueError("heading requires a nonempty recorded trace")
    previous = unwrapped = 0.0
    after = []
    for index, row in enumerate(frames):
        mode, metrics = row["executed_mode"], row["metrics"]
        if mode not in MODES or metrics["control_step"] != index + 1:
            raise ValueError("heading trace mode or post-step sequence differs")
        signed = _number(metrics["heading_error_signed_rad"])
        phase = _number(row["executed_phase_seconds"])
        if signed is None or phase is None or abs(signed) > math.pi:
            raise ValueError("signed wrapped heading and phase must be available")
        difference = signed - previous
        increment = (difference + math.pi) % (2 * math.pi) - math.pi
        if increment == -math.pi and difference > 0:
            increment = math.pi
        unwrapped += increment
        previous = signed
        if mode == "after":
            after.append((abs(unwrapped), signed, index + 1, phase))
    return {
        "after_heading_max_abs_unwrapped_rad": max(row[0] for row in after) if after else None,
        "after_mode_sample_count": len(after),
        "after_first_heading_signed_rad": after[0][1] if after else None,
        "after_first_control_step": after[0][2] if after else None,
        "after_first_phase_seconds": after[0][3] if after else None,
    }


def substrate_criteria(
    objective: dict, *, nonfoot_contact_count: int, after_reentry_count: int
) -> dict[str, bool]:
    region, speed = objective["region"], objective["speed"]
    oracle, tracking = objective["oracle_diagnostics"], objective["tracking"]
    counts = oracle["executed_mode_counts"]
    if type(counts) is not dict or set(counts) - MODES:
        raise ValueError("objective contains an unknown executed mode")
    for count in counts.values():
        _count(count)
    return {
        "full_20s_without_fall": _number(objective["duration_seconds"]) == 20.0
        and _count(objective["fall_count"]) == 0,
        "no_nonfoot_ground_contact": _count(nonfoot_contact_count) == 0,
        "three_switches": _count(oracle["observed_switch_count"]) == 3,
        "all_four_modes_executed": set(counts) == MODES and all(counts.values()),
        "region_entry_and_exit": region["entry_observed"] is True
        and region["exit_observed_after_entry"] is True,
        "finish_observed": objective["finish_condition_observed"] is True,
        "no_after_region_reentry": _count(after_reentry_count) == 0,
        "minimum_25_inside_samples": _count(region["inside_sample_count"]) >= 25,
        "compliance_at_least_0p75": _at_least(region["posture_compliant_fraction"], 0.75),
        "joint_p95_at_most_0p35": _at_most(tracking["joint_position_rmse_rad_p95"], 0.35),
        "roll_pitch_p95_at_most_0p25": _at_most(tracking["roll_pitch_rmse_rad_p95"], 0.25),
        "speed_mae_at_most_0p35": _at_most(speed["mean_absolute_error_m_s"], 0.35),
        "inside_speed_deviation_at_most_0p10": _at_most(
            speed["inside_mean_speed_target_deviation_m_s"], 0.10
        ),
    }


def baseline_criteria(
    objective: dict, *, nonfoot_contact_count: int, after_reentry_count: int
) -> dict[str, bool]:
    rows = substrate_criteria(
        objective,
        nonfoot_contact_count=nonfoot_contact_count,
        after_reentry_count=after_reentry_count,
    )
    lateral = _number(objective["maximum_lateral_error_m"])
    rows["lateral_headroom_over_0p75m"] = lateral is not None and lateral > 0.75
    return rows


def pair_criteria(
    baseline_objective: dict,
    candidate_objective: dict,
    baseline_heading: dict,
    candidate_heading: dict,
    *,
    nonfoot_contact_count: int,
    after_reentry_count: int,
) -> dict[str, bool]:
    rows = substrate_criteria(
        candidate_objective,
        nonfoot_contact_count=nonfoot_contact_count,
        after_reentry_count=after_reentry_count,
    )
    first_lateral = _number(baseline_objective["maximum_lateral_error_m"])
    first_heading = _number(baseline_heading["after_heading_max_abs_unwrapped_rad"])
    first_compliance = _number(baseline_objective["region"]["posture_compliant_fraction"])
    if first_lateral is None or first_heading is None or first_compliance is None:
        raise ValueError("baseline admission requires observed lateral, heading and compliance")
    rows.update(
        lateral_reduction_30_percent=_at_most(
            candidate_objective["maximum_lateral_error_m"], 0.70 * first_lateral
        ),
        after_heading_no_regression=_at_most(
            candidate_heading["after_heading_max_abs_unwrapped_rad"], first_heading
        ),
        compliance_no_regression=_at_least(
            candidate_objective["region"]["posture_compliant_fraction"], first_compliance
        ),
    )
    return rows
