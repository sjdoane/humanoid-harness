"""Adapter-owned admission certificate for the reviewed task-input V2 type."""

from __future__ import annotations

from dataclasses import dataclass

from oracle_composition.rewards.task_inputs_v2 import (
    CONTROL_PERIOD_SECONDS,
    CandidateTaskInputsV2,
    TaskInputsV2Error,
    validate_task_inputs_v2,
)

from .contracts import PhaseBContractError

TASK_INPUT_ADMISSION_ID = "phase_b_stock_com_task_input_admission/v1"
MEASUREMENT_ORIGIN = "stock_body_mass_weighted_com_x_delta_over_control_period"
NUMERIC_REPRESENTATION = "builtin_float_from_little_endian_float64_measurement"
VELOCITY_MIN_M_S = -25.0
VELOCITY_MAX_M_S = 25.0


def task_input_admission_contract() -> dict[str, object]:
    return {
        "cadence_seconds": CONTROL_PERIOD_SECONDS,
        "inclusive_velocity_range_m_s": [VELOCITY_MIN_M_S, VELOCITY_MAX_M_S],
        "measurement_origin": MEASUREMENT_ORIGIN,
        "numeric_representation": NUMERIC_REPRESENTATION,
        "schema_version": 1,
        "task_input_admission_id": TASK_INPUT_ADMISSION_ID,
    }


@dataclass(frozen=True, slots=True)
class AdmittedTaskInputsV2:
    inputs: CandidateTaskInputsV2
    measurement_origin: str
    numeric_representation: str
    cadence_seconds: float
    admission_id: str


def _validate_velocity(value: object) -> float:
    if type(value) is not float:
        raise PhaseBContractError(
            "COM velocity must use the certified builtin-float representation"
        )
    if not VELOCITY_MIN_M_S <= value <= VELOCITY_MAX_M_S:
        raise PhaseBContractError("COM velocity lies outside the inclusive [-25, 25] m/s range")
    return value


def admit_task_inputs_v2(
    *,
    com_x_velocity_m_s: float,
    measurement_origin: str,
    cadence_seconds: float,
    numeric_representation: str = NUMERIC_REPRESENTATION,
) -> AdmittedTaskInputsV2:
    """Construct the reviewed V2 payload only after adapter semantics are certified."""

    velocity = _validate_velocity(com_x_velocity_m_s)
    if measurement_origin != MEASUREMENT_ORIGIN:
        raise PhaseBContractError("task-input measurement origin is not the stock COM measurement")
    if type(cadence_seconds) is not float or cadence_seconds != CONTROL_PERIOD_SECONDS:
        raise PhaseBContractError("task-input cadence must equal exactly 0.015 s")
    if numeric_representation != NUMERIC_REPRESENTATION:
        raise PhaseBContractError("task-input numeric representation differs")
    try:
        inputs = CandidateTaskInputsV2(velocity, 3.0)
    except TaskInputsV2Error as exc:
        raise PhaseBContractError(str(exc)) from exc
    return AdmittedTaskInputsV2(
        inputs=inputs,
        measurement_origin=measurement_origin,
        numeric_representation=numeric_representation,
        cadence_seconds=cadence_seconds,
        admission_id=TASK_INPUT_ADMISSION_ID,
    )


def validate_admitted_task_inputs_v2(value: object) -> CandidateTaskInputsV2:
    """Revalidate the adapter certificate and range at reward consumption."""

    if type(value) is not AdmittedTaskInputsV2:
        raise PhaseBContractError("reward consumption requires an exact task-input admission")
    try:
        inputs = object.__getattribute__(value, "inputs")
        origin = object.__getattribute__(value, "measurement_origin")
        representation = object.__getattribute__(value, "numeric_representation")
        cadence = object.__getattribute__(value, "cadence_seconds")
        admission_id = object.__getattribute__(value, "admission_id")
    except AttributeError as exc:
        raise PhaseBContractError("task-input admission slots are incomplete") from exc
    if (
        admission_id != TASK_INPUT_ADMISSION_ID
        or origin != MEASUREMENT_ORIGIN
        or representation != NUMERIC_REPRESENTATION
        or type(cadence) is not float
        or cadence != CONTROL_PERIOD_SECONDS
    ):
        raise PhaseBContractError("task-input admission certificate differs")
    try:
        velocity, _target = validate_task_inputs_v2(inputs)
    except TaskInputsV2Error as exc:
        raise PhaseBContractError(str(exc)) from exc
    _validate_velocity(velocity)
    return inputs


__all__ = [
    "MEASUREMENT_ORIGIN",
    "NUMERIC_REPRESENTATION",
    "TASK_INPUT_ADMISSION_ID",
    "VELOCITY_MAX_M_S",
    "VELOCITY_MIN_M_S",
    "AdmittedTaskInputsV2",
    "admit_task_inputs_v2",
    "task_input_admission_contract",
    "validate_admitted_task_inputs_v2",
]
