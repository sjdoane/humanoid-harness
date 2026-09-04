from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from oracle_composition.rewards.scale_calibration import (
    CALIBRATION_GRID_M_S,
    STOCK_GRID_MEAN,
    STOCK_GRID_POPULATION_SD,
    ScaleCalibrationError,
    calibrate_task_term_scale,
    frozen_stock_grid_statistics,
    validate_target_speed_axioms,
)
from oracle_composition.rewards.static_validation import validate_task_term_source

ROOT = Path(__file__).parents[2]


def _program(expression: str):
    source = f"def task_term(x):\n    return {expression}\n".encode()
    return source, validate_task_term_source(source)


def test_scale_calibration_reproduces_frozen_grid_mean_sd_and_hashes_inputs() -> None:
    source, program = _program("-abs(x.com_x_velocity_m_s - x.target_speed_m_s)")
    receipt = calibrate_task_term_scale(
        program,
        candidate_source_sha256=hashlib.sha256(source).hexdigest(),
    )
    assert frozen_stock_grid_statistics() == (STOCK_GRID_MEAN, STOCK_GRID_POPULATION_SD)
    assert receipt.stock_mean == receipt.scaled_mean == 1.25
    assert receipt.stock_population_sd == STOCK_GRID_POPULATION_SD
    assert receipt.scaled_population_sd == pytest.approx(STOCK_GRID_POPULATION_SD, abs=1e-12)
    assert 0.25 <= receipt.affine_alpha <= 4.0
    assert abs(receipt.affine_beta) <= 10.0
    assert receipt.grid_sha256
    assert receipt.raw_values_sha256 != receipt.scaled_values_sha256
    assert len(CALIBRATION_GRID_M_S) == 33
    assert CALIBRATION_GRID_M_S[0] == -3.0
    assert CALIBRATION_GRID_M_S[-1] == 5.0


def test_axioms_require_unique_target_maximum_monotonicity_and_determinism() -> None:
    _source, program = _program("-abs(x.com_x_velocity_m_s - x.target_speed_m_s)")
    receipt = validate_target_speed_axioms(program)
    assert receipt.passed is True
    assert receipt.raw_values_sha256 == receipt.repeated_values_sha256

    _source, wrong_direction = _program("abs(x.com_x_velocity_m_s - x.target_speed_m_s)")
    with pytest.raises(ScaleCalibrationError, match="unique grid maximum"):
        validate_target_speed_axioms(wrong_direction)


def test_degenerate_or_out_of_range_affine_parameters_refuse() -> None:
    source, constant = _program("0.0")
    with pytest.raises(ScaleCalibrationError):
        calibrate_task_term_scale(
            constant,
            candidate_source_sha256=hashlib.sha256(source).hexdigest(),
        )

    narrow_source = b"def task_term(x):\n    return 1.25 * (1.0 - min(1.0, abs(x.com_x_velocity_m_s - x.target_speed_m_s) / x.target_speed_m_s))\n"
    narrow = validate_task_term_source(narrow_source)
    with pytest.raises(ScaleCalibrationError, match="alpha"):
        calibrate_task_term_scale(
            narrow,
            candidate_source_sha256=hashlib.sha256(narrow_source).hexdigest(),
        )

    shifted_source, shifted = _program("100.0 - abs(x.com_x_velocity_m_s - x.target_speed_m_s)")
    with pytest.raises(ScaleCalibrationError, match="beta"):
        calibrate_task_term_scale(
            shifted,
            candidate_source_sha256=hashlib.sha256(shifted_source).hexdigest(),
        )


def test_scale_receipt_refuses_a_mismatched_candidate_source_hash() -> None:
    _source, program = _program("-abs(x.com_x_velocity_m_s - x.target_speed_m_s)")
    with pytest.raises(ScaleCalibrationError, match="static-validation receipt"):
        calibrate_task_term_scale(program, candidate_source_sha256="0" * 64)


def test_non_monotonic_far_error_refuses() -> None:
    source = b"""def task_term(x):\n    error = abs(x.com_x_velocity_m_s - x.target_speed_m_s)\n    if error > 2.0:\n        return 0.0\n    else:\n        return -error\n"""
    program = validate_task_term_source(source)
    with pytest.raises(ScaleCalibrationError, match="non-increasing"):
        validate_target_speed_axioms(program)


def test_recorded_behavior_free_scale_receipt_replays_exactly() -> None:
    recorded = json.loads(
        (
            ROOT
            / "experiments"
            / "family_b_target_speed_v1"
            / "receipts"
            / "builder_synthetic_scale.json"
        ).read_text(encoding="utf-8")
    )
    assert recorded["training_run"] is False
    assert recorded["behavioral_evaluation"] is False
    source = recorded["source_utf8"].encode("utf-8")
    program = validate_task_term_source(source)
    axiom = validate_target_speed_axioms(program)
    scale = calibrate_task_term_scale(
        program,
        candidate_source_sha256=program.receipt.source_sha256,
    )
    assert recorded["static_validation"] == program.receipt.to_dict()
    assert recorded["static_validation_sha256"] == program.receipt.sha256
    assert recorded["target_speed_axioms"] == axiom.to_dict()
    assert recorded["target_speed_axioms_sha256"] == axiom.sha256
    assert recorded["scale_calibration"] == scale.to_dict()
    assert recorded["scale_calibration_sha256"] == scale.sha256
