from __future__ import annotations

import json
import random

import numpy as np
import pytest
import torch

from oracle_composition.adapters.gmt.training_telemetry import (
    TELEMETRY_FILENAME,
    EpisodeAccumulator,
    TrainingTelemetry,
    observation_group_moments,
    ppo_update_record,
    validate_training_telemetry,
    validate_training_telemetry_descriptor,
)


def _info(step: int, *, fallen: bool = False, horizon: bool = False) -> dict:
    return {
        "metrics": {
            "control_step": step,
            "progress_m": float(step) / 10,
            "lateral_error_m": 0.1,
            "heading_error_rad": 0.2,
            "root_height_m": 0.6,
            "forward_speed_m_s": 0.7,
            "speed_error_m_s": 0.05,
            "posture_band_error_m": 0.0,
            "joint_position_rmse_rad": 0.1,
            "root_height_abs_error_m": 0.02,
            "roll_pitch_rmse_rad": 0.03,
            "inside_posture_region": True,
            "posture_success": not fallen,
            "finish_condition_met": False,
            "horizon_reached": horizon,
            "fallen": fallen,
        }
    }


def _observations(value: float = 0.0) -> np.ndarray:
    return np.full((2, 2171), value, dtype=np.float32)


def _numpy_state_equal(left: tuple, right: tuple) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_measurements_do_not_consume_python_numpy_or_torch_rng(tmp_path) -> None:
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.random.get_rng_state().clone()

    path = tmp_path / TELEMETRY_FILENAME
    with TrainingTelemetry(path) as telemetry:
        telemetry.observe_step([1.0], [False], [_info(1)])
        telemetry.rollout_boundary(512, _observations(), {}, None)
        telemetry.final_update({}, None)
        descriptor = telemetry.descriptor(512)
    validate_training_telemetry_descriptor(descriptor, path.read_bytes(), 512)

    assert random.getstate() == python_before
    assert _numpy_state_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.random.get_rng_state(), torch_before)


def test_rollouts_bind_previous_update_then_flush_final_update(tmp_path) -> None:
    path = tmp_path / TELEMETRY_FILENAME
    first_update = {
        "train/approx_kl": 0.012,
        "train/clip_fraction": 0.25,
        "train/n_updates": 4,
    }
    final_update = {"train/entropy_loss": -2.0, "train/n_updates": 8}
    with TrainingTelemetry(path) as telemetry:
        telemetry.rollout_boundary(512, _observations(), {}, 0)
        telemetry.rollout_boundary(1024, _observations(1.0), first_update, 4)
        telemetry.final_update(final_update, 8)
        descriptor = telemetry.descriptor(1024)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["previous_update"] is None
    assert rows[1]["previous_update"]["trained_through_transitions"] == 512
    assert rows[1]["previous_update"]["sb3_n_updates"] == 4
    assert rows[2]["update"]["trained_through_transitions"] == 1024
    assert rows[2]["update"]["sb3_n_updates"] == 8
    assert rows[2]["update"]["optimizer_minibatch_steps"] is None
    assert descriptor["sb3_n_updates_semantics"] == (
        "attempted_ppo_epochs_including_kl_stopped_partial_epochs"
    )
    assert descriptor["record_count"] == 3
    assert validate_training_telemetry(path.read_bytes(), 1024)["rollout_boundary_count"] == 2


def test_missing_logger_statistics_are_explicitly_null() -> None:
    record = ppo_update_record(512, {}, None)

    assert record["sb3_n_updates"] is None
    assert record["optimizer_minibatch_steps"] is None
    assert set(record["metrics"].values()) == {None}
    assert set(record["metric_unavailable_reasons"].values()) == {"missing"}


def test_undefined_explained_variance_is_null_but_other_stats_remain_strict() -> None:
    record = ppo_update_record(
        512,
        {"train/explained_variance": np.nan, "train/approx_kl": 0.01},
        4,
    )

    assert record["metrics"]["explained_variance"] is None
    assert record["metric_unavailable_reasons"]["explained_variance"] == (
        "undefined_nonfinite"
    )
    assert record["metrics"]["approx_kl"] == 0.01
    assert "approx_kl" not in record["metric_unavailable_reasons"]


def test_nonfinite_measurements_fail_closed() -> None:
    observations = _observations()
    observations[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonempty and finite"):
        observation_group_moments(observations)
    with pytest.raises(ValueError, match="finite numeric"):
        ppo_update_record(512, {"train/approx_kl": np.inf}, 4)
    with pytest.raises(ValueError, match="finite numeric"):
        ppo_update_record(512, {"train/value_loss": np.nan}, 4)
    accumulator = EpisodeAccumulator()
    with pytest.raises(ValueError, match="reward, done, or info"):
        accumulator.observe([np.nan], [False], [_info(1)])


def test_episode_summary_spans_rollout_boundaries_and_records_termination() -> None:
    accumulator = EpisodeAccumulator()
    accumulator.observe([1.0], [False], [_info(1)])
    assert accumulator.take_summary()["completed"] == 0
    accumulator.observe([2.0], [True], [_info(2, horizon=True)])

    summary = accumulator.take_summary()

    assert summary["completed"] == 1
    assert summary["returns"]["mean"] == 3.0
    assert summary["lengths"]["mean"] == 2.0
    assert summary["falls"] == 0
    assert summary["horizons"] == 1
    assert summary["terminal_true"]["horizon_reached"] == 1
    assert accumulator.incomplete() == []


@pytest.mark.parametrize(
    "mutation",
    ["hash", "size", "count", "boolean_count", "semantics", "extra"],
)
def test_descriptor_must_exactly_bind_retained_bytes(tmp_path, mutation: str) -> None:
    path = tmp_path / TELEMETRY_FILENAME
    with TrainingTelemetry(path) as telemetry:
        telemetry.rollout_boundary(512, _observations(), {}, None)
        telemetry.final_update({}, None)
        descriptor = telemetry.descriptor(512)
    descriptor = dict(descriptor)
    if mutation == "hash":
        descriptor["sha256"] = "0" * 64
    elif mutation == "size":
        descriptor["size_bytes"] += 1
    elif mutation == "count":
        descriptor["record_count"] += 1
    elif mutation == "boolean_count":
        descriptor["rollout_boundary_count"] = True
    elif mutation == "semantics":
        descriptor["sb3_n_updates_semantics"] = "optimizer_minibatches"
    else:
        descriptor["unbound"] = True

    with pytest.raises(ValueError, match="descriptor"):
        validate_training_telemetry_descriptor(descriptor, path.read_bytes(), 512)


@pytest.mark.parametrize("field,value", [("sb3_n_updates", True), ("sb3_n_updates", -1)])
def test_validator_rejects_boolean_or_negative_update_counts(
    tmp_path, field: str, value: object
) -> None:
    path = tmp_path / TELEMETRY_FILENAME
    with TrainingTelemetry(path) as telemetry:
        telemetry.rollout_boundary(512, _observations(), {}, None)
        telemetry.final_update({}, None)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[-1]["update"][field] = value
    encoded = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )

    with pytest.raises(ValueError, match="final-update"):
        validate_training_telemetry(encoded, 512)


@pytest.mark.parametrize("value", [True, -1])
def test_validator_rejects_boolean_or_negative_episode_counts(tmp_path, value: object) -> None:
    path = tmp_path / TELEMETRY_FILENAME
    with TrainingTelemetry(path) as telemetry:
        telemetry.rollout_boundary(512, _observations(), {}, None)
        telemetry.final_update({}, None)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["episodes"]["falls"] = value
    encoded = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )

    with pytest.raises(ValueError, match="rollout telemetry"):
        validate_training_telemetry(encoded, 512)
