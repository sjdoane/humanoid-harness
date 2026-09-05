"""Static scale constants and the gated dynamic-evaluation entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING, NoReturn

import numpy as np

from .contract import RewardContractError
from .static_validation import (
    StaticallyAcceptedTaskTermSourceV1,
    StaticValidationError,
    assert_static_acceptance_current,
)

if TYPE_CHECKING:
    from .sandbox import RewardSandboxWorkerV1

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
DYNAMIC_ADMISSION_REFUSAL = (
    "dynamic task-term evaluation is disabled until sandbox runtime admission and its "
    "receipt path are independently reviewed; no local callable fallback is permitted"
)


class ScaleCalibrationError(RewardContractError):
    """Raised when scale/axiom evaluation is unbound, stale, or not admitted."""


def frozen_stock_grid_statistics() -> tuple[float, float]:
    grid = np.asarray(CALIBRATION_GRID_M_S, dtype=np.float64)
    stock = 1.25 * grid
    mean = float(np.mean(stock))
    standard_deviation = float(np.std(stock, ddof=0))
    if mean != STOCK_GRID_MEAN or standard_deviation != STOCK_GRID_POPULATION_SD:
        raise ScaleCalibrationError("frozen stock grid statistics drifted")
    return mean, standard_deviation


def _require_source_bound_admitted_worker(
    source: StaticallyAcceptedTaskTermSourceV1,
    *,
    worker: RewardSandboxWorkerV1 | None,
    candidate_source_sha256: str | None = None,
) -> NoReturn:
    try:
        current = assert_static_acceptance_current(source)
    except StaticValidationError as exc:
        raise ScaleCalibrationError(
            "axiom/scale evaluation requires a current statically accepted source; "
            "arbitrary callables are not admitted"
        ) from exc
    if candidate_source_sha256 is not None:
        if (
            type(candidate_source_sha256) is not str
            or len(candidate_source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in candidate_source_sha256)
        ):
            raise ScaleCalibrationError("candidate_source_sha256 must be a lowercase SHA-256")
        if candidate_source_sha256 != current.receipt.source_sha256:
            raise ScaleCalibrationError(
                "candidate source SHA-256 differs from its static-acceptance receipt"
            )
    if worker is None:
        raise ScaleCalibrationError(DYNAMIC_ADMISSION_REFUSAL)

    from .sandbox import RewardSandboxWorkerV1

    if type(worker) is not RewardSandboxWorkerV1:
        raise ScaleCalibrationError(
            "dynamic evaluation requires the exact source-bound RewardSandboxWorkerV1; "
            "arbitrary callables are not admitted"
        )
    if (
        worker.snapshot.source_sha256 != current.receipt.source_sha256
        or worker.snapshot.source_bytes != current.source_bytes
    ):
        raise ScaleCalibrationError(
            "sandbox worker source differs from the static-acceptance receipt"
        )
    raise ScaleCalibrationError(DYNAMIC_ADMISSION_REFUSAL)


def validate_target_speed_axioms(
    source: StaticallyAcceptedTaskTermSourceV1,
    *,
    worker: RewardSandboxWorkerV1 | None = None,
) -> NoReturn:
    """Refuse until reviewed dynamic admission can use the bound child worker."""

    _require_source_bound_admitted_worker(source, worker=worker)


def calibrate_task_term_scale(
    source: StaticallyAcceptedTaskTermSourceV1,
    *,
    candidate_source_sha256: str,
    worker: RewardSandboxWorkerV1 | None = None,
) -> NoReturn:
    """Refuse rather than evaluate authored code in the parent process."""

    _require_source_bound_admitted_worker(
        source,
        worker=worker,
        candidate_source_sha256=candidate_source_sha256,
    )


__all__ = [
    "AFFINE_ALPHA_MAX",
    "AFFINE_ALPHA_MIN",
    "AFFINE_BETA_ABS_MAX",
    "CALIBRATION_GRID_J",
    "CALIBRATION_GRID_M_S",
    "CALIBRATION_GRID_STEP_M_S",
    "CALIBRATION_TARGET_SPEED_M_S",
    "DYNAMIC_ADMISSION_REFUSAL",
    "STOCK_GRID_MEAN",
    "STOCK_GRID_POPULATION_SD",
    "ScaleCalibrationError",
    "calibrate_task_term_scale",
    "frozen_stock_grid_statistics",
    "validate_target_speed_axioms",
]
