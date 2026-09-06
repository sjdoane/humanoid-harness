from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from oracle_composition.adapters.gmt.contracts import (
    ACTION_DIM,
    HISTORY_LENGTH,
    OBSERVATION_DIM,
    PROPRIOCEPTION_DIM,
    REFERENCE_FRAME_DIM,
    REFERENCE_HORIZON,
)
from oracle_composition.adapters.gmt.io import GMTAdmissionError
from oracle_composition.adapters.gmt.reference_sensitivity import (
    SHIFT_CONTROL_STEPS,
    measure_reference_sensitivity,
    numeric_array_sha256,
    validate_probe_identities,
)


def _actor_values(observations: np.ndarray) -> np.ndarray:
    reference = observations[:, :600].reshape(-1, REFERENCE_HORIZON, REFERENCE_FRAME_DIM)
    actions = reference[:, 0, :ACTION_DIM] + 0.125 * reference[:, 1, :ACTION_DIM]
    actions += 0.01 * observations[:, 600 : 600 + ACTION_DIM]
    return np.ascontiguousarray(actions.astype("<f4", copy=False))


def _fixture(rows: int = 4) -> dict[str, np.ndarray | int]:
    reference = np.arange(
        rows * REFERENCE_HORIZON * REFERENCE_FRAME_DIM, dtype="<f4"
    ).reshape(rows, REFERENCE_HORIZON, REFERENCE_FRAME_DIM)
    reference /= np.float32(100.0)
    proprioception = np.arange(rows * PROPRIOCEPTION_DIM, dtype="<f8").reshape(
        rows, PROPRIOCEPTION_DIM
    )
    proprioception /= 1_000.0
    history = np.zeros((rows, HISTORY_LENGTH, PROPRIOCEPTION_DIM), dtype="<f4")
    for row in range(rows):
        previous = proprioception[max(0, row - HISTORY_LENGTH) : row].astype("<f4")
        history[row, HISTORY_LENGTH - previous.shape[0] :] = previous
    observations = np.zeros((rows, OBSERVATION_DIM), dtype="<f4")
    observations[:, :600] = reference.reshape(rows, -1)
    observations[:, 600:674] = proprioception.astype("<f4")
    observations[:, 674:] = history.reshape(rows, -1)
    return {
        "observations": observations,
        "recorded_actions": _actor_values(observations),
        "recorded_reference_windows": reference.copy(),
        "admitted_reference_windows": reference.copy(),
        "shifted_reference_windows": np.ascontiguousarray(reference + np.float32(0.5)),
        "recorded_proprioception": proprioception,
        "control_steps": np.arange(rows, dtype="<i8"),
        "shuffle_seed": 71,
    }


def _measure(
    values: dict[str, np.ndarray | int],
    actor: Callable[[np.ndarray], np.ndarray] = _actor_values,
):
    return measure_reference_sensitivity(
        observations=values["observations"],
        recorded_actions=values["recorded_actions"],
        recorded_reference_windows=values["recorded_reference_windows"],
        admitted_reference_windows=values["admitted_reference_windows"],
        shifted_reference_windows=values["shifted_reference_windows"],
        recorded_proprioception=values["recorded_proprioception"],
        control_steps=values["control_steps"],
        shuffle_seed=values["shuffle_seed"],
        actor=actor,
    )


def test_interventions_hold_all_non_reference_bytes_fixed() -> None:
    values = _fixture()
    calls: list[np.ndarray] = []

    def actor(observations: np.ndarray) -> np.ndarray:
        calls.append(observations.copy())
        return _actor_values(observations)

    measurement = _measure(values, actor)

    assert len(calls) == 4
    fixed = calls[0][:, 600:].tobytes()
    assert all(call[:, 600:].tobytes() == fixed for call in calls[1:])
    assert np.count_nonzero(calls[1][:, :600]) == 0
    expected_permutation = np.random.Generator(np.random.PCG64(71)).permutation(20)
    np.testing.assert_array_equal(measurement.arrays["shuffle_permutation"], expected_permutation)
    expected_shuffled = values["recorded_reference_windows"][:, expected_permutation, :]
    np.testing.assert_array_equal(calls[2][:, :600].reshape(-1, 20, 30), expected_shuffled)
    np.testing.assert_array_equal(
        calls[3][:, :600].reshape(-1, 20, 30), values["shifted_reference_windows"]
    )
    np.testing.assert_array_equal(
        measurement.arrays["shifted_source_control_step"],
        values["control_steps"] + SHIFT_CONTROL_STEPS,
    )


def test_actions_and_input_identities_remain_recomputable() -> None:
    values = _fixture()
    measurement = _measure(values)

    np.testing.assert_array_equal(
        measurement.arrays["recorded_raw_action"], values["recorded_actions"]
    )
    np.testing.assert_array_equal(
        measurement.arrays["recomputed_exact_action"], values["recorded_actions"]
    )
    for condition in ("zero_reference", "shuffled_reference", "shifted_reference"):
        summary = measurement.action_summaries[condition]
        assert summary["rows"] == 4
        assert summary["bitwise_changed_rows"] == 4
        assert summary["max_action_delta_l2"] > 0
    exact_identity = measurement.observation_identities["exact"]
    assert exact_identity["sha256"] == numeric_array_sha256(values["observations"])
    assert exact_identity["shape"] == [4, OBSERVATION_DIM]
    assert exact_identity["dtype"] == "<f4"


def test_exact_action_byte_mismatch_fails_closed() -> None:
    values = _fixture()
    mismatched = values["recorded_actions"].copy()
    mismatched.view(np.uint8)[0, 0] ^= 1
    values["recorded_actions"] = mismatched

    with pytest.raises(GMTAdmissionError, match="retained raw action bytes"):
        _measure(values)


def test_observation_lineage_mismatch_fails_closed() -> None:
    values = _fixture()
    windows = values["admitted_reference_windows"].copy()
    windows[0, 0, 0] += np.float32(1.0)
    values["admitted_reference_windows"] = windows

    with pytest.raises(GMTAdmissionError, match="admitted motion runtime"):
        _measure(values)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("trace_sha256", "trace SHA-256"),
        ("weights_sha256", "actor SHA-256"),
        ("motion_sha256", "motion identity"),
    ],
)
def test_external_trace_actor_and_motion_hashes_are_bound(
    field: str, message: str
) -> None:
    manifest = {
        "trace": {"sha256": "a" * 64},
        "inputs": {
            "weights_sha256": "b" * 64,
            "motion_name": "walk_stand",
            "motion_sha256": "c" * 64,
        },
    }
    selected = {
        "trace_sha256": "a" * 64,
        "weights_sha256": "b" * 64,
        "motion_name": "walk_stand",
        "motion_sha256": "c" * 64,
    }
    selected[field] = "0" * 64

    with pytest.raises(GMTAdmissionError, match=message):
        validate_probe_identities(manifest, **selected)
