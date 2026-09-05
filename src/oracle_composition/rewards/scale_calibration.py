"""Behavior-free scale matching and target-speed reward axioms."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from .contract import CandidateTaskInputsV1, RewardContractError, TaskTermV1, canonical_json_bytes

CALIBRATION_TARGET_SPEED_M_S = 1.0
CALIBRATION_GRID_STEP_M_S = 0.25
CALIBRATION_GRID_J = tuple(range(-16, 17))
CALIBRATION_GRID_M_S = tuple(
    CALIBRATION_TARGET_SPEED_M_S + CALIBRATION_GRID_STEP_M_S * index for index in CALIBRATION_GRID_J
)
STOCK_GRID_MEAN = 1.25
STOCK_GRID_POPULATION_SD = 2.9755951785595207
AFFINE_ALPHA_MIN = 0.25
AFFINE_ALPHA_MAX = 4.0
AFFINE_BETA_ABS_MAX = 10.0


class ScaleCalibrationError(RewardContractError):
    """Raised when scale matching or target-speed axioms fail closed."""


def _sha256_f64(values: Sequence[float], *, label: str) -> str:
    array = np.ascontiguousarray(values, dtype=np.dtype(">f8"))
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ScaleCalibrationError(f"{label} must be a finite one-dimensional sequence")
    header = canonical_json_bytes({"dtype": ">f8", "shape": [int(array.size)], "label": label})
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(4, "big"))
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _task_callable(task_term: object) -> object:
    function = getattr(task_term, "task_term", None)
    if function is None and callable(task_term):
        function = task_term
    if not callable(function):
        raise ScaleCalibrationError("task term does not expose task_term(x)")
    return function


def _evaluate(task_term: object, velocity: float, target: float) -> float:
    function = _task_callable(task_term)
    try:
        value = function(CandidateTaskInputsV1(float(velocity), float(target)))  # type: ignore[operator]
    except BaseException as exc:
        raise ScaleCalibrationError(
            f"task term failed a scale/axiom probe: {type(exc).__name__}: {exc}"
        ) from exc
    if type(value) is not float or not math.isfinite(value) or abs(value) > 1000.0:
        raise ScaleCalibrationError("task term probe is not a finite in-envelope Python float")
    return value


def frozen_stock_grid_statistics() -> tuple[float, float]:
    grid = np.asarray(CALIBRATION_GRID_M_S, dtype=np.float64)
    stock = 1.25 * grid
    mean = float(np.mean(stock))
    standard_deviation = float(np.std(stock, ddof=0))
    if mean != STOCK_GRID_MEAN or standard_deviation != STOCK_GRID_POPULATION_SD:
        raise ScaleCalibrationError("frozen stock grid statistics drifted")
    return mean, standard_deviation


@dataclass(frozen=True, slots=True)
class TargetSpeedAxiomReceiptV1:
    grid_sha256: str
    raw_values_sha256: str
    repeated_values_sha256: str
    unique_target_maximum: bool
    non_increasing_with_absolute_error: bool
    target_shift_response: bool
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "grid_sha256": self.grid_sha256,
            "raw_values_sha256": self.raw_values_sha256,
            "repeated_values_sha256": self.repeated_values_sha256,
            "unique_target_maximum": self.unique_target_maximum,
            "non_increasing_with_absolute_error": self.non_increasing_with_absolute_error,
            "target_shift_response": self.target_shift_response,
            "passed": self.passed,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()


def validate_target_speed_axioms(task_term: TaskTermV1 | object) -> TargetSpeedAxiomReceiptV1:
    first = [
        _evaluate(task_term, velocity, CALIBRATION_TARGET_SPEED_M_S)
        for velocity in CALIBRATION_GRID_M_S
    ]
    for velocity in reversed(CALIBRATION_GRID_M_S):
        _evaluate(task_term, velocity, CALIBRATION_TARGET_SPEED_M_S)
    repeated = [
        _evaluate(task_term, velocity, CALIBRATION_TARGET_SPEED_M_S)
        for velocity in CALIBRATION_GRID_M_S
    ]
    first_bits = [struct.pack(">d", value) for value in first]
    repeated_bits = [struct.pack(">d", value) for value in repeated]
    if first_bits != repeated_bits:
        raise ScaleCalibrationError("task term failed bitwise determinism on the frozen grid")

    center = CALIBRATION_GRID_J.index(0)
    maximum = max(first)
    unique_target_maximum = first[center] == maximum and first.count(maximum) == 1
    left_from_center = list(reversed(first[: center + 1]))
    right_from_center = first[center:]
    non_increasing = all(
        farther <= nearer
        for side in (left_from_center, right_from_center)
        for nearer, farther in pairwise(side)
    )
    target_shift_response = True
    for target in (0.5, 1.0, 1.5):
        centered = _evaluate(task_term, target, target)
        below = _evaluate(task_term, target - 0.25, target)
        above = _evaluate(task_term, target + 0.25, target)
        target_shift_response = target_shift_response and centered > below and centered > above
    receipt = TargetSpeedAxiomReceiptV1(
        grid_sha256=_sha256_f64(CALIBRATION_GRID_M_S, label="velocity_grid_m_s"),
        raw_values_sha256=_sha256_f64(first, label="raw_task_term"),
        repeated_values_sha256=_sha256_f64(repeated, label="raw_task_term"),
        unique_target_maximum=unique_target_maximum,
        non_increasing_with_absolute_error=non_increasing,
        target_shift_response=target_shift_response,
        passed=unique_target_maximum and non_increasing and target_shift_response,
    )
    if not receipt.passed:
        failures = [
            name
            for name, value in (
                ("unique grid maximum at target", unique_target_maximum),
                ("non-increasing absolute-error response", non_increasing),
                ("target-shift response", target_shift_response),
            )
            if not value
        ]
        raise ScaleCalibrationError("target-speed axioms failed: " + ", ".join(failures))
    return receipt


@dataclass(frozen=True, slots=True)
class ScaleCalibrationReceiptV1:
    candidate_source_sha256: str
    target_speed_m_s: float
    grid_sha256: str
    raw_values_sha256: str
    scaled_values_sha256: str
    raw_mean: float
    raw_population_sd: float
    stock_mean: float
    stock_population_sd: float
    affine_alpha: float
    affine_beta: float
    scaled_mean: float
    scaled_population_sd: float
    axiom_receipt_sha256: str
    passed: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "candidate_source_sha256": self.candidate_source_sha256,
            "target_speed_m_s": self.target_speed_m_s,
            "grid_sha256": self.grid_sha256,
            "raw_values_sha256": self.raw_values_sha256,
            "scaled_values_sha256": self.scaled_values_sha256,
            "raw_mean": self.raw_mean,
            "raw_population_sd": self.raw_population_sd,
            "stock_mean": self.stock_mean,
            "stock_population_sd": self.stock_population_sd,
            "affine_alpha": self.affine_alpha,
            "affine_beta": self.affine_beta,
            "scaled_mean": self.scaled_mean,
            "scaled_population_sd": self.scaled_population_sd,
            "axiom_receipt_sha256": self.axiom_receipt_sha256,
            "passed": self.passed,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


def calibrate_task_term_scale(
    task_term: TaskTermV1 | object,
    *,
    candidate_source_sha256: str,
) -> ScaleCalibrationReceiptV1:
    if (
        type(candidate_source_sha256) is not str
        or len(candidate_source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in candidate_source_sha256)
    ):
        raise ScaleCalibrationError("candidate_source_sha256 must be a lowercase SHA-256")
    validated_receipt = getattr(task_term, "receipt", None)
    if getattr(validated_receipt, "source_sha256", None) != candidate_source_sha256:
        raise ScaleCalibrationError(
            "candidate source SHA-256 differs from its static-validation receipt"
        )
    axiom_receipt = validate_target_speed_axioms(task_term)
    raw = np.asarray(
        [
            _evaluate(task_term, velocity, CALIBRATION_TARGET_SPEED_M_S)
            for velocity in CALIBRATION_GRID_M_S
        ],
        dtype=np.float64,
    )
    raw_mean = float(np.mean(raw))
    raw_sd = float(np.std(raw, ddof=0))
    if not math.isfinite(raw_sd) or raw_sd == 0.0:
        raise ScaleCalibrationError("authored grid population SD is zero or non-finite")
    stock_mean, stock_sd = frozen_stock_grid_statistics()
    alpha = stock_sd / raw_sd
    beta = stock_mean - alpha * raw_mean
    if not math.isfinite(alpha) or not AFFINE_ALPHA_MIN <= alpha <= AFFINE_ALPHA_MAX:
        raise ScaleCalibrationError("computed affine alpha is outside [0.25, 4]")
    if not math.isfinite(beta) or abs(beta) > AFFINE_BETA_ABS_MAX:
        raise ScaleCalibrationError("computed affine beta exceeds 10 in magnitude")
    scaled = alpha * raw + beta
    scaled_mean = float(np.mean(scaled))
    scaled_sd = float(np.std(scaled, ddof=0))
    if (
        not np.isfinite(scaled).all()
        or not math.isclose(scaled_mean, stock_mean, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(scaled_sd, stock_sd, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ScaleCalibrationError("affine scale does not reproduce the frozen stock moments")
    return ScaleCalibrationReceiptV1(
        candidate_source_sha256=candidate_source_sha256,
        target_speed_m_s=CALIBRATION_TARGET_SPEED_M_S,
        grid_sha256=_sha256_f64(CALIBRATION_GRID_M_S, label="velocity_grid_m_s"),
        raw_values_sha256=_sha256_f64(raw, label="raw_task_term"),
        scaled_values_sha256=_sha256_f64(scaled, label="scaled_task_term"),
        raw_mean=raw_mean,
        raw_population_sd=raw_sd,
        stock_mean=stock_mean,
        stock_population_sd=stock_sd,
        affine_alpha=float(alpha),
        affine_beta=float(beta),
        scaled_mean=scaled_mean,
        scaled_population_sd=scaled_sd,
        axiom_receipt_sha256=axiom_receipt.sha256,
    )


__all__ = [
    "AFFINE_ALPHA_MAX",
    "AFFINE_ALPHA_MIN",
    "AFFINE_BETA_ABS_MAX",
    "CALIBRATION_GRID_J",
    "CALIBRATION_GRID_M_S",
    "CALIBRATION_GRID_STEP_M_S",
    "CALIBRATION_TARGET_SPEED_M_S",
    "STOCK_GRID_MEAN",
    "STOCK_GRID_POPULATION_SD",
    "ScaleCalibrationError",
    "ScaleCalibrationReceiptV1",
    "TargetSpeedAxiomReceiptV1",
    "calibrate_task_term_scale",
    "frozen_stock_grid_statistics",
    "validate_target_speed_axioms",
]
