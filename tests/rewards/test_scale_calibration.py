from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from oracle_composition.rewards.scale_calibration import (
    CALIBRATION_GRID_M_S,
    DYNAMIC_ADMISSION_REFUSAL,
    STOCK_GRID_MEAN,
    STOCK_GRID_POPULATION_SD,
    ScaleCalibrationError,
    calibrate_task_term_scale,
    frozen_stock_grid_statistics,
    validate_target_speed_axioms,
)
from oracle_composition.rewards.static_validation import (
    StaticAcceptanceReceiptV1,
    StaticValidationError,
    statically_validate_task_term_source,
)

ROOT = Path(__file__).parents[2]
SOURCE = b"def task_term(x):\n    return -abs(x.com_x_velocity_m_s - x.target_speed_m_s)\n"


class SentinelCallable:
    def __init__(self) -> None:
        self.touched = False

    def __call__(self, *_args: object, **_kwargs: object) -> float:
        self.touched = True
        return 0.0

    def task_term(self, *_args: object, **_kwargs: object) -> float:
        self.touched = True
        return 0.0


def test_frozen_stock_grid_statistics_remain_data_only() -> None:
    assert frozen_stock_grid_statistics() == (STOCK_GRID_MEAN, STOCK_GRID_POPULATION_SD)
    assert len(CALIBRATION_GRID_M_S) == 33
    assert CALIBRATION_GRID_M_S[0] == -3.0
    assert CALIBRATION_GRID_M_S[-1] == 5.0


def test_arbitrary_callable_is_refused_without_invocation() -> None:
    sentinel = SentinelCallable()
    with pytest.raises(ScaleCalibrationError, match="arbitrary callables"):
        validate_target_speed_axioms(sentinel)  # type: ignore[arg-type]
    with pytest.raises(ScaleCalibrationError, match="arbitrary callables"):
        calibrate_task_term_scale(  # type: ignore[arg-type]
            sentinel,
            candidate_source_sha256=hashlib.sha256(SOURCE).hexdigest(),
        )
    assert sentinel.touched is False


def test_fake_worker_callable_is_refused_without_invocation() -> None:
    accepted = statically_validate_task_term_source(SOURCE)
    sentinel = SentinelCallable()
    with pytest.raises(ScaleCalibrationError, match="exact source-bound"):
        validate_target_speed_axioms(accepted, worker=sentinel)  # type: ignore[arg-type]
    assert sentinel.touched is False


def test_static_source_cannot_manufacture_dynamic_acceptance_receipts() -> None:
    accepted = statically_validate_task_term_source(SOURCE)
    with pytest.raises(ScaleCalibrationError, match="dynamic task-term evaluation") as axiom:
        validate_target_speed_axioms(accepted)
    with pytest.raises(ScaleCalibrationError, match="dynamic task-term evaluation") as scale:
        calibrate_task_term_scale(
            accepted,
            candidate_source_sha256=accepted.receipt.source_sha256,
        )
    assert str(axiom.value) == DYNAMIC_ADMISSION_REFUSAL
    assert str(scale.value) == DYNAMIC_ADMISSION_REFUSAL


def test_candidate_hash_and_static_receipt_drift_are_rejected_before_runtime() -> None:
    accepted = statically_validate_task_term_source(SOURCE)
    with pytest.raises(ScaleCalibrationError, match="static-acceptance receipt"):
        calibrate_task_term_scale(accepted, candidate_source_sha256="0" * 64)

    stale_payload = accepted.receipt.to_dict()
    stale_payload["source_sha256"] = "0" * 64
    stale_receipt = StaticAcceptanceReceiptV1.from_dict(stale_payload)
    with pytest.raises(StaticValidationError, match="stale"):
        dataclasses.replace(accepted, receipt=stale_receipt)


def test_historical_dynamic_receipt_is_not_readmitted_as_static_acceptance() -> None:
    recorded = json.loads(
        (
            ROOT
            / "experiments"
            / "family_b_target_speed_v1"
            / "receipts"
            / "builder_synthetic_scale.json"
        ).read_text(encoding="utf-8")
    )
    with pytest.raises(StaticValidationError, match="keys or version"):
        StaticAcceptanceReceiptV1.from_dict(recorded["static_validation"])


@pytest.mark.skip(reason="R2 host fixture required: dynamic axiom/scale outcomes remain unreviewed")
@pytest.mark.parametrize(
    ("source", "expected_failure"),
    [
        (b"def task_term(x):\n    return 0.0\n", "zero population SD"),
        (
            b"def task_term(x):\n    return abs(x.com_x_velocity_m_s - x.target_speed_m_s)\n",
            "unique target maximum",
        ),
        (
            b"def task_term(x):\n    return 100.0 - abs(x.com_x_velocity_m_s - x.target_speed_m_s)\n",
            "affine beta bound",
        ),
    ],
)
def test_deferred_dynamic_negative_cases_require_admitted_worker(
    source: bytes,
    expected_failure: str,
) -> None:
    """Retain inherited negative cases without executing candidate source in R1."""

    pytest.fail(f"R2 must implement the worker-backed {expected_failure} assertion for {source!r}")
