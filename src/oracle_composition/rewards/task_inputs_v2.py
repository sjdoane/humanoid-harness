"""Data-only inputs for T2; simulator measurement remains the adapter's job."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

TASK_INPUTS_V2_SCHEMA_ID = "humanoid-fixed-com-speed/task-inputs/v2"
TARGET_SPEED_M_S = 3.0
CONTROL_PERIOD_SECONDS = 0.015


class TaskInputsV2Error(ValueError):
    """T2 inputs violate the fixed speed or numeric contract."""


def _finite_number(value: object, name: str) -> float:
    if type(value) not in (int, float):
        raise TaskInputsV2Error(f"{name} must be an exact builtin number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise TaskInputsV2Error(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise TaskInputsV2Error(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class CandidateTaskInputsV2:
    com_x_velocity_m_s: float
    target_speed_m_s: float

    def __post_init__(self) -> None:
        velocity, target = validate_task_inputs_v2(self)
        object.__setattr__(self, "com_x_velocity_m_s", velocity)
        object.__setattr__(self, "target_speed_m_s", target)

    def to_dict(self) -> dict[str, float]:
        velocity, target = validate_task_inputs_v2(self)
        return {"com_x_velocity_m_s": velocity, "target_speed_m_s": target}

    @property
    def canonical_bytes(self) -> bytes:
        return json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")


def validate_task_inputs_v2(value: object) -> tuple[float, float]:
    """Revalidate at consumption; frozen dataclasses alone are not an admission gate."""

    if type(value) is not CandidateTaskInputsV2:
        raise TaskInputsV2Error("expected exact CandidateTaskInputsV2")
    try:
        velocity = object.__getattribute__(value, "com_x_velocity_m_s")
        target = object.__getattribute__(value, "target_speed_m_s")
    except AttributeError as exc:
        raise TaskInputsV2Error("input slots are incomplete") from exc
    velocity = _finite_number(velocity, "com_x_velocity_m_s")
    target = _finite_number(target, "target_speed_m_s")
    if target != TARGET_SPEED_M_S:
        raise TaskInputsV2Error("T2 target_speed_m_s must equal 3.0")
    return velocity, target


def task_input_contract_v2() -> dict[str, object]:
    """Return fresh schema data, not a certificate that an adapter supplied these signals."""

    return {
        "schema_id": TASK_INPUTS_V2_SCHEMA_ID,
        "schema_version": 2,
        "fields": ["com_x_velocity_m_s", "target_speed_m_s"],
        "units": {"com_x_velocity_m_s": "m/s", "target_speed_m_s": "m/s"},
        "target_speed_m_s": TARGET_SPEED_M_S,
        "control_period_seconds": CONTROL_PERIOD_SECONDS,
        "velocity_definition": "stock_mass_center_delta_x_over_control_period",
        "stock_info_key": "x_velocity",
        "measurement_provenance": "must_be_verified_by_adapter",
    }
