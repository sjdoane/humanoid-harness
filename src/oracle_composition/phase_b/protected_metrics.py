"""Evaluator-owned protected metrics computed only from direct-state traces."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import (
    array_sha256,
    canonical_json_bytes,
)
from oracle_composition.experiments.fixed_reference import ExperimentContractError

CONTROL_PERIOD_SECONDS = 0.015
ERROR_NAMES = (
    "root_height_abs_error_m",
    "root_orientation_error_rad",
    "root_linear_velocity_rmse_m_s",
    "root_angular_velocity_rmse_rad_s",
    "joint_position_rmse_rad",
    "joint_velocity_rmse_rad_s",
)
ERROR_SCALES = {
    "root_height_abs_error_m": 0.20,
    "root_orientation_error_rad": 0.50,
    "root_linear_velocity_rmse_m_s": 1.0,
    "root_angular_velocity_rmse_rad_s": 2.0,
    "joint_position_rmse_rad": 0.35,
    "joint_velocity_rmse_rad_s": 2.0,
}
_STATE_WIDTHS = {
    "root_position_world_m": 3,
    "root_orientation_wxyz": 4,
    "root_linear_velocity_world_m_s": 3,
    "root_angular_velocity_body_rad_s": 3,
    "joint_positions_rad": 17,
    "joint_velocities_rad_s": 17,
}


def _finite(value: object, *, field: str) -> float:
    if type(value) not in {int, float}:
        raise ExperimentContractError(f"protected {field} must be an exact number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ExperimentContractError(f"protected {field} must be finite") from exc
    if not math.isfinite(result):
        raise ExperimentContractError(f"protected {field} must be finite")
    return result


def _vector(value: object, *, field: str, width: int) -> np.ndarray:
    if type(value) not in {list, tuple} or len(value) != width:
        raise ExperimentContractError(f"protected {field} must contain {width} values")
    return np.ascontiguousarray(
        [_finite(item, field=field) for item in value],
        dtype="<f8",
    )


def _mass_center_state(
    *,
    body_mass: object,
    body_xipos_before: object,
    body_xipos_after: object,
) -> dict[str, object]:
    if type(body_mass) not in {list, tuple} or not 1 <= len(body_mass) <= 128:
        raise ExperimentContractError("protected body-mass vector is malformed")
    count = len(body_mass)
    masses = _vector(body_mass, field="body_mass_kg", width=count)
    if np.any(masses < 0.0):
        raise ExperimentContractError("protected body masses must be non-negative")

    def positions(value: object, field: str) -> np.ndarray:
        if type(value) not in {list, tuple} or len(value) != count:
            raise ExperimentContractError(f"protected {field} is malformed")
        rows = [_vector(row, field=field, width=3) for row in value]
        return np.ascontiguousarray(rows, dtype="<f8")

    before = positions(body_xipos_before, "body_xipos_before_world_m")
    after = positions(body_xipos_after, "body_xipos_after_world_m")
    value = {
        "body_mass_kg": masses.tolist(),
        "body_xipos_after_world_m": after.tolist(),
        "body_xipos_before_world_m": before.tolist(),
    }
    value["sha256"] = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return value


def evaluator_state_record(state: object) -> dict[str, object]:
    """Copy the exact controller-facing state fields into canonical JSON data."""

    record = {
        name: np.ascontiguousarray(getattr(state, name), dtype="<f8").tolist()
        for name in _STATE_WIDTHS
    }
    record["root_height_m"] = float(state.root_height_m)
    validate_evaluator_state(record)
    return record


def validate_evaluator_state(value: object) -> dict[str, object]:
    expected = {*_STATE_WIDTHS, "root_height_m"}
    if type(value) is not dict or set(value) != expected:
        raise ExperimentContractError("protected evaluator state fields differ")
    for name, width in _STATE_WIDTHS.items():
        _vector(value[name], field=name, width=width)
    _finite(value["root_height_m"], field="root_height_m")
    return value


def evaluator_tracking_errors(
    state: Mapping[str, object],
    reference_row: Sequence[object] | np.ndarray,
) -> dict[str, float]:
    """Compute the six protected errors without importing reward code."""

    checked = validate_evaluator_state(dict(state))
    reference = _vector(
        reference_row.tolist() if type(reference_row) is np.ndarray else reference_row,
        field="reference_row",
        width=45,
    )
    orientation = _vector(
        checked["root_orientation_wxyz"],
        field="root_orientation_wxyz",
        width=4,
    )
    target_orientation = reference[1:5]
    if not math.isclose(float(np.linalg.norm(orientation)), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ExperimentContractError("protected state orientation must be a unit quaternion")
    if not math.isclose(
        float(np.linalg.norm(target_orientation)),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ExperimentContractError("protected reference orientation must be a unit quaternion")
    orientation_dot = float(np.clip(abs(np.dot(orientation, target_orientation)), 0.0, 1.0))

    def rmse(values: np.ndarray) -> float:
        return math.sqrt(float(np.mean(np.square(values), dtype=np.float64)))

    result = {
        "root_height_abs_error_m": abs(
            _finite(checked["root_height_m"], field="root_height_m") - float(reference[0])
        ),
        "root_orientation_error_rad": 2.0 * math.acos(orientation_dot),
        "root_linear_velocity_rmse_m_s": rmse(
            _vector(
                checked["root_linear_velocity_world_m_s"],
                field="root_linear_velocity_world_m_s",
                width=3,
            )
            - reference[5:8]
        ),
        "root_angular_velocity_rmse_rad_s": rmse(
            _vector(
                checked["root_angular_velocity_body_rad_s"],
                field="root_angular_velocity_body_rad_s",
                width=3,
            )
            - reference[8:11]
        ),
        "joint_position_rmse_rad": rmse(
            _vector(
                checked["joint_positions_rad"],
                field="joint_positions_rad",
                width=17,
            )
            - reference[11:28]
        ),
        "joint_velocity_rmse_rad_s": rmse(
            _vector(
                checked["joint_velocities_rad_s"],
                field="joint_velocities_rad_s",
                width=17,
            )
            - reference[28:45]
        ),
    }
    if set(result) != set(ERROR_NAMES) or not all(
        math.isfinite(value) and value >= 0.0 for value in result.values()
    ):
        raise ExperimentContractError("protected tracking errors are invalid")
    return result


def evaluator_mass_center_x_m(body_mass: np.ndarray, body_xipos: np.ndarray) -> float:
    """Compute world COM-x directly from MuJoCo mass-center state."""

    masses = np.asarray(body_mass)
    positions = np.asarray(body_xipos)
    if (
        masses.dtype.str != "<f8"
        or positions.dtype.str != "<f8"
        or masses.ndim != 1
        or positions.shape != (len(masses), 3)
        or not masses.flags.c_contiguous
        or not positions.flags.c_contiguous
        or not np.isfinite(masses).all()
        or not np.isfinite(positions).all()
        or np.any(masses < 0.0)
    ):
        raise ExperimentContractError("protected mass-center state is malformed")
    total_mass = float(np.sum(masses, dtype=np.float64))
    if not math.isfinite(total_mass) or total_mass <= 0.0:
        raise ExperimentContractError("protected total body mass is invalid")
    result = float(np.dot(masses, positions[:, 0]) / total_mass)
    if not math.isfinite(result):
        raise ExperimentContractError("protected mass-center position is non-finite")
    return result


def reference_row_record(*, behavior: str, index: int, row: np.ndarray) -> dict[str, object]:
    if (
        type(behavior) is not str
        or not behavior
        or type(index) is not int
        or index < 0
        or type(row) is not np.ndarray
        or row.dtype.str != "<f8"
        or row.shape != (45,)
        or not row.flags.c_contiguous
        or not np.isfinite(row).all()
    ):
        raise ExperimentContractError("protected reference-row identity is malformed")
    return {
        "behavior": behavior,
        "index": index,
        "sha256": array_sha256(row),
        "values": row.tolist(),
    }


def protected_step_record(
    *,
    step: int,
    state: object,
    reference_behavior: str,
    reference_index: int,
    reference_row: np.ndarray,
    action: np.ndarray,
    root_x_before_m: float,
    body_mass: np.ndarray,
    body_xipos_before: np.ndarray,
    body_xipos_after: np.ndarray,
    forbidden_contacts: Sequence[Mapping[str, object]],
    fallen: bool,
    terminated: bool,
    truncated: bool,
) -> dict[str, object]:
    mass_center_state = _mass_center_state(
        body_mass=np.asarray(body_mass).tolist(),
        body_xipos_before=np.asarray(body_xipos_before).tolist(),
        body_xipos_after=np.asarray(body_xipos_after).tolist(),
    )
    record = {
        "action": np.ascontiguousarray(action, dtype="<f4").tolist(),
        "control_period_s": CONTROL_PERIOD_SECONDS,
        "fallen": fallen,
        "forbidden_contacts": [dict(item) for item in forbidden_contacts],
        "mass_center_state": mass_center_state,
        "plant_terminated": terminated,
        "plant_truncated": truncated,
        "reference": reference_row_record(
            behavior=reference_behavior,
            index=reference_index,
            row=reference_row,
        ),
        "root_x_before_m": root_x_before_m,
        "state": evaluator_state_record(state),
        "step": step,
    }
    validate_protected_step(record)
    return record


def validate_protected_step(value: object) -> dict[str, object]:
    expected = {
        "action",
        "control_period_s",
        "fallen",
        "forbidden_contacts",
        "mass_center_state",
        "plant_terminated",
        "plant_truncated",
        "reference",
        "root_x_before_m",
        "state",
        "step",
    }
    if type(value) is not dict or set(value) != expected:
        raise ExperimentContractError("protected step trace fields differ")
    if type(value["step"]) is not int or value["step"] < 0:
        raise ExperimentContractError("protected trace step is invalid")
    if value["control_period_s"] != CONTROL_PERIOD_SECONDS:
        raise ExperimentContractError("protected trace cadence differs")
    for field in ("fallen", "plant_terminated", "plant_truncated"):
        if type(value[field]) is not bool:
            raise ExperimentContractError("protected trace flags must be booleans")
    action = _vector(value["action"], field="action", width=17)
    if np.any(action < -0.4) or np.any(action > 0.4):
        raise ExperimentContractError("protected trace action is out of bounds")
    _finite(value["root_x_before_m"], field="root_x_before_m")
    mass_center = value["mass_center_state"]
    if type(mass_center) is not dict or set(mass_center) != {
        "body_mass_kg",
        "body_xipos_after_world_m",
        "body_xipos_before_world_m",
        "sha256",
    }:
        raise ExperimentContractError("protected mass-center state fields differ")
    checked_mass_center = _mass_center_state(
        body_mass=mass_center["body_mass_kg"],
        body_xipos_before=mass_center["body_xipos_before_world_m"],
        body_xipos_after=mass_center["body_xipos_after_world_m"],
    )
    if mass_center != checked_mass_center:
        raise ExperimentContractError("protected mass-center state identity differs")
    reference = value["reference"]
    if type(reference) is not dict or set(reference) != {"behavior", "index", "sha256", "values"}:
        raise ExperimentContractError("protected trace reference identity differs")
    if (
        type(reference["behavior"]) is not str
        or not reference["behavior"]
        or type(reference["index"]) is not int
        or reference["index"] < 0
    ):
        raise ExperimentContractError("protected trace reference identity is malformed")
    row = _vector(reference["values"], field="reference values", width=45)
    if reference["sha256"] != array_sha256(row):
        raise ExperimentContractError("protected trace reference-row hash differs")
    validate_evaluator_state(value["state"])
    contacts = value["forbidden_contacts"]
    if type(contacts) is not list or any(type(item) is not dict for item in contacts):
        raise ExperimentContractError("protected trace contact records are malformed")
    return value


def select_evaluator_nearest_phase(
    *,
    state: Mapping[str, object],
    target_rows: np.ndarray,
    task_step: int,
    source_behavior: str,
    target_behavior: str,
    reason: str,
) -> dict[str, object]:
    """Apply the frozen phase score with evaluator-owned error calculations."""

    if (
        type(target_rows) is not np.ndarray
        or target_rows.dtype.str != "<f8"
        or target_rows.ndim != 2
        or target_rows.shape[1] != 45
        or not target_rows.flags.c_contiguous
        or not np.isfinite(target_rows).all()
        or type(task_step) is not int
        or not 0 <= task_step < len(target_rows)
    ):
        raise ExperimentContractError("protected phase-transfer inputs are malformed")
    candidates = []
    for phase in range(task_step + 1):
        raw = evaluator_tracking_errors(state, target_rows[phase])
        normalized = tuple(raw[name] / ERROR_SCALES[name] for name in ERROR_NAMES)
        score = (max(normalized), sum(value * value for value in normalized), phase)
        candidates.append((score, phase, normalized))
    score, phase, normalized = min(candidates, key=lambda item: item[0])
    return {
        "candidate_range_inclusive": [0, task_step],
        "selected_normalized_errors": {
            name: value for name, value in zip(ERROR_NAMES, normalized, strict=True)
        },
        "selected_phase": phase,
        "selected_score": [score[0], score[1], score[2]],
        "selection_reason": reason,
        "source_behavior": source_behavior,
        "target_behavior": target_behavior,
        "target_window_indices": [min(phase + offset, len(target_rows) - 1) for offset in range(8)],
        "task_step": task_step,
    }


def _resynchronization_records(
    normalized_errors: Sequence[float],
    switches: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    records = []
    for switch in switches:
        boundary = int(switch["boundary"])
        latest_start = min(len(normalized_errors) - 8, boundary + 56)
        latency = None
        for index in range(boundary, latest_start + 1):
            if all(value <= 1.0 for value in normalized_errors[index : index + 8]):
                latency = index + 8 - boundary
                break
        records.append(
            {
                "eight_consecutive_boundaries_at_or_below_one": latency is not None,
                "from_behavior": switch["from_behavior"],
                "settle_latency_steps": latency,
                "to_behavior": switch["to_behavior"],
            }
        )
    return records


def _fallen_from_state(state: Mapping[str, object]) -> bool:
    orientation = _vector(state["root_orientation_wxyz"], field="root_orientation_wxyz", width=4)
    w, x, y, z = (float(value) for value in orientation)
    norm = w * w + x * x + y * y + z * z
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=2e-6):
        raise ExperimentContractError("protected fall state orientation is not normalized")
    torso_up = (w * w - x * x - y * y + z * z) / norm
    height = _finite(state["root_height_m"], field="root_height_m")
    return not 1.0 <= height <= 2.0 or torso_up < 0.5


def recompute_protected_episode(
    *,
    steps: Sequence[Mapping[str, object]],
    cell: str,
    switches: Sequence[Mapping[str, object]],
    segment_targets_m_s: tuple[float, float, float],
) -> dict[str, object]:
    """Recompute every protected aggregate from one sufficient step trace."""

    if len(steps) != 1_000 or cell not in {
        "hold_expert",
        "hold_medium",
        "hold_simple",
        "fixed_round_trip",
    }:
        raise ExperimentContractError("protected episode trace identity or horizon differs")
    if (
        type(segment_targets_m_s) is not tuple
        or len(segment_targets_m_s) != 3
        or any(
            type(value) is not float or not math.isfinite(value) for value in segment_targets_m_s
        )
    ):
        raise ExperimentContractError("protected segment targets are malformed")
    raw_errors = {name: [] for name in ERROR_NAMES}
    normalized_max_errors = []
    root_speeds = []
    com_speeds = []
    contacts = []
    fall = False
    first_failure = None
    for expected_step, raw_step in enumerate(steps):
        step = validate_protected_step(dict(raw_step))
        if step["step"] != expected_step:
            raise ExperimentContractError("protected trace steps are not contiguous")
        reference = step["reference"]
        errors = evaluator_tracking_errors(step["state"], reference["values"])
        for name in ERROR_NAMES:
            raw_errors[name].append(errors[name])
        normalized_max_errors.append(max(errors[name] / ERROR_SCALES[name] for name in ERROR_NAMES))
        state = step["state"]
        root_speeds.append(
            (float(state["root_position_world_m"][0]) - float(step["root_x_before_m"]))
            / CONTROL_PERIOD_SECONDS
        )
        mass_center = step["mass_center_state"]
        body_mass = np.ascontiguousarray(mass_center["body_mass_kg"], dtype="<f8")
        before = np.ascontiguousarray(mass_center["body_xipos_before_world_m"], dtype="<f8")
        after = np.ascontiguousarray(mass_center["body_xipos_after_world_m"], dtype="<f8")
        com_speeds.append(
            (
                evaluator_mass_center_x_m(body_mass, after)
                - evaluator_mass_center_x_m(body_mass, before)
            )
            / CONTROL_PERIOD_SECONDS
        )
        contacts.extend(dict(item) for item in step["forbidden_contacts"])
        fallen = _fallen_from_state(state)
        if step["fallen"] is not fallen:
            raise ExperimentContractError("protected recorded fall flag differs from direct state")
        fall = fall or fallen
        if first_failure is None and (fallen or step["forbidden_contacts"]):
            first_failure = expected_step + 1
        if bool(step["plant_terminated"]) or (
            bool(step["plant_truncated"]) and expected_step + 1 != 1_000
        ):
            raise ExperimentContractError("protected trace ended before its fixed horizon")
    summaries = {
        name: math.sqrt(float(np.mean(np.square(values), dtype=np.float64)))
        for name, values in raw_errors.items()
    }
    resynchronization = _resynchronization_records(normalized_max_errors, switches)
    fast_target, slow_target, return_fast_target = segment_targets_m_s
    segment_errors = {
        "fast": float(np.mean(np.abs(np.asarray(com_speeds[:300]) - fast_target))),
        "slow": float(np.mean(np.abs(np.asarray(com_speeds[300:600]) - slow_target))),
        "return_fast": float(np.mean(np.abs(np.asarray(com_speeds[600:]) - return_fast_target))),
    }
    transition_window_error = None
    settled_state_error = None
    if cell == "fixed_round_trip":
        transition_window_error = float(
            np.mean([*normalized_max_errors[300:364], *normalized_max_errors[600:664]])
        )
        settled_state_error = float(
            max(
                np.mean(normalized_max_errors[332:364]),
                np.mean(normalized_max_errors[632:664]),
            )
        )
    latencies = [
        record["settle_latency_steps"]
        for record in resynchronization
        if record["settle_latency_steps"] is not None
    ]
    result = {
        "action_bounds_ok": True,
        "com_forward_speed_m_s": com_speeds,
        "fall": fall,
        "forbidden_contacts": contacts,
        "observed_steps": len(steps),
        "resynchronization_records": resynchronization,
        "root_delta_forward_speed_m_s": root_speeds,
        "segment_errors": segment_errors,
        "settled_state_normalized_error": settled_state_error,
        "settle_latency_steps": max(latencies) if latencies else None,
        "six_tracking_errors": summaries,
        "time_to_first_failure_steps": first_failure,
        "transition_window_error": transition_window_error,
    }
    canonical_json_bytes(result)
    return result


def protected_trace_sha256(steps: Sequence[Mapping[str, object]]) -> str:
    return hashlib.sha256(canonical_json_bytes([dict(step) for step in steps])).hexdigest()


__all__ = [
    "CONTROL_PERIOD_SECONDS",
    "ERROR_NAMES",
    "ERROR_SCALES",
    "evaluator_mass_center_x_m",
    "evaluator_state_record",
    "evaluator_tracking_errors",
    "protected_step_record",
    "protected_trace_sha256",
    "recompute_protected_episode",
    "reference_row_record",
    "select_evaluator_nearest_phase",
    "validate_evaluator_state",
    "validate_protected_step",
]
