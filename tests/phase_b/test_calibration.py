from __future__ import annotations

from pathlib import Path

import pytest

from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.phase_b import evaluation as evaluation_module
from oracle_composition.phase_b.calibration import (
    CALIBRATION_BLOCKS,
    CALIBRATION_POLICY_SEEDS,
    calibration_receipt_value,
)


def _samples() -> list[dict[str, object]]:
    return [
        {
            "block_id": block,
            "fall": block in CALIBRATION_BLOCKS[-2:],
            "first_transition_latency_steps": (None if block in CALIBRATION_BLOCKS[-2:] else 7),
            "policy_seed_id": seed,
            "second_transition_latency_steps": 8,
            "segment_speed_errors_m_s": {
                "fast": 0.2,
                "return_fast": 0.3,
                "slow": 0.4,
            },
            "settled_state_normalized_error": 0.5,
            "source_checkpoint_sha256": f"{seed:064x}",
        }
        for seed in CALIBRATION_POLICY_SEEDS
        for block in CALIBRATION_BLOCKS
    ]


def test_calibration_computes_frozen_margins_and_censored_caps() -> None:
    receipt = calibration_receipt_value(_samples())
    assert receipt["sample_count"] == 100
    assert receipt["split"]["disjoint_from_evaluation_and_training"] is True
    assert receipt["thresholds"] == {
        "censoring_latency_steps": 65,
        "segment_speed_error_bands_m_s": {
            "fast": pytest.approx(0.22),
            "return_fast": pytest.approx(0.33),
            "slow": pytest.approx(0.44),
        },
        "settled_state_normalized_error_band": pytest.approx(0.55),
        "transition_latency_caps_steps": [64, 64],
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("block_id", 120101, "outside the frozen split"),
        ("policy_seed_id", 121001, "outside the frozen split"),
        ("block_id", 10**1000, "integer bound"),
    ],
)
def test_calibration_rejects_overlap_and_out_of_representation_integer(
    field: str, value: int, message: str
) -> None:
    samples = _samples()
    samples[0][field] = value
    with pytest.raises(ExperimentContractError, match=message):
        calibration_receipt_value(samples)


def test_calibration_rejects_multiple_checkpoints_for_one_policy_seed() -> None:
    samples = _samples()
    samples[0]["source_checkpoint_sha256"] = "f" * 64
    with pytest.raises(ExperimentContractError, match="one source checkpoint"):
        calibration_receipt_value(samples)


def test_evaluator_refuses_missing_calibration_before_checkpoint_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    constructed = False

    def fail_if_called(*_args: object, **_kwargs: object) -> object:
        nonlocal constructed
        constructed = True
        raise AssertionError("checkpoint constructed before calibration admission")

    monkeypatch.setattr(evaluation_module, "load_full_checkpoint", fail_if_called)
    with pytest.raises(ExperimentContractError, match="calibration receipt is unavailable"):
        evaluation_module.evaluate_policy_checkpoint(
            checkpoint_path=tmp_path / "checkpoint.npz",
            checkpoint_sha256="0" * 64,
            step_zero_actor_path=tmp_path / "actor.npz",
            step_zero_actor_sha256="1" * 64,
            corpus_root=tmp_path,
            calibration_receipt_path=tmp_path / "absent.json",
            calibration_receipt_sha256="2" * 64,
        )
    assert constructed is False
