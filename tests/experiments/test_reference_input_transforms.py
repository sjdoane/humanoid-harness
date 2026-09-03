from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from oracle_composition.experiments import reference_input_transforms as transforms
from oracle_composition.experiments.fixed_reference import (
    CAUSAL_INPUT_CONDITION_IDS,
    ExperimentContractError,
)


def _reference(n_frames: int = 1001) -> np.ndarray:
    reference = np.zeros((n_frames, 45), dtype="<f8", order="C")
    reference[:, 0] = 1.4 + np.arange(n_frames, dtype=np.float64) / 10_000.0
    reference[:, 1] = 1.0
    reference[:, 11] = np.arange(n_frames, dtype=np.float64) / 1000.0
    return reference


def test_ordered_conditions_match_the_authoritative_experiment_contract() -> None:
    assert frozenset(transforms.CONDITION_IDS) == CAUSAL_INPUT_CONDITION_IDS


def test_exact_transform_preserves_sequence_and_uses_terminal_hold() -> None:
    source = _reference()
    before = source.copy()

    result = transforms.transform_reference_input(
        source,
        condition_id=transforms.C_EXACT,
        current_frame=999,
    )

    np.testing.assert_array_equal(result.transformed_full_sequence, source)
    np.testing.assert_array_equal(result.window[0], source[999])
    np.testing.assert_array_equal(result.window[1:], np.repeat(source[1000][None, :], 7, axis=0))
    np.testing.assert_array_equal(source, before)
    assert result.receipt.index_map == tuple(range(1001))
    assert result.receipt.window_timeline_indices == (999, 1000, 1000, 1000, 1000, 1000, 1000, 1000)
    assert result.receipt.window_source_indices == result.receipt.window_timeline_indices
    assert result.receipt.source_sha256 == result.receipt.transformed_full_sequence_sha256
    assert result.receipt.output_window_sha256 == transforms.float64_array_sha256(result.window)
    assert result.transformed_full_sequence.flags.writeable is False
    assert result.window.flags.writeable is False
    with pytest.raises(ValueError):
        result.transformed_full_sequence.setflags(write=True)
    with pytest.raises(ValueError):
        result.window.setflags(write=True)


def test_zero_transform_emits_positive_float64_zero_bytes() -> None:
    result = transforms.transform_reference_input(
        _reference(),
        condition_id=transforms.C_ZERO_INPUT,
        current_frame=17,
    )

    assert result.transformed_full_sequence.dtype.str == "<f8"
    assert result.transformed_full_sequence.flags.c_contiguous
    assert np.count_nonzero(result.transformed_full_sequence.view("<u8")) == 0
    assert np.count_nonzero(result.window.view("<u8")) == 0
    assert result.receipt.index_map == (None,) * 1001
    assert result.receipt.window_source_indices == (None,) * 8
    assert result.receipt.parameters["fill_float64_bits_hex"] == "0000000000000000"


def test_sha256_ranked_shuffle_is_fixed_complete_and_rng_independent() -> None:
    source = _reference(11)
    np.random.seed(999)
    first = transforms.transform_reference_input(
        source,
        condition_id=transforms.C_SHUFFLE_INPUT,
        current_frame=2,
    )
    np.random.seed(1)
    second = transforms.transform_reference_input(
        source.copy(),
        condition_id=transforms.C_SHUFFLE_INPUT,
        current_frame=2,
    )

    expected_map = (8, 0, 9, 5, 6, 1, 10, 2, 4, 3, 7)
    assert first.receipt.index_map == expected_map
    assert second.receipt.index_map == expected_map
    assert tuple(sorted(first.receipt.index_map)) == tuple(range(11))
    np.testing.assert_array_equal(first.transformed_full_sequence, source[list(expected_map)])
    np.testing.assert_array_equal(first.window, second.window)
    assert first.receipt.to_dict() == second.receipt.to_dict()
    assert first.receipt.parameters["seed"] == 260907
    assert first.receipt.parameters["ranking_algorithm_id"] == (
        transforms.SHUFFLE_RANK_ALGORITHM_ID
    )


def test_shuffle_rejects_identity_and_value_equivalence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        transforms,
        "_shuffle_index_map",
        lambda *, n_frames, seed: tuple(range(n_frames)),
    )
    with pytest.raises(ExperimentContractError, match="must not be the identity"):
        transforms.transform_reference_input(
            _reference(5),
            condition_id=transforms.C_SHUFFLE_INPUT,
            current_frame=0,
        )

    monkeypatch.setattr(
        transforms,
        "_shuffle_index_map",
        lambda *, n_frames, seed: tuple(reversed(range(n_frames))),
    )
    symmetric = _reference(4)
    symmetric[3] = symmetric[0]
    symmetric[2] = symmetric[1]
    with pytest.raises(ExperimentContractError, match="value-equivalent"):
        transforms.transform_reference_input(
            symmetric,
            condition_id=transforms.C_SHUFFLE_INPUT,
            current_frame=0,
        )


def test_positive_250_frame_shift_has_exact_source_map() -> None:
    source = _reference()
    result = transforms.transform_reference_input(
        source,
        condition_id=transforms.C_SHIFT_INPUT,
        current_frame=800,
    )

    expected_map = tuple((index + 250) % 1001 for index in range(1001))
    assert result.receipt.index_map == expected_map
    np.testing.assert_array_equal(result.transformed_full_sequence, source[list(expected_map)])
    assert result.receipt.window_timeline_indices == tuple(range(800, 808))
    assert result.receipt.window_source_indices == tuple(
        (index + 250) % 1001 for index in range(800, 808)
    )
    assert result.receipt.parameters["shift_frames"] == 250
    assert result.receipt.parameters["operation"] == (
        "output_frame_i_reads_source_frame_i_plus_shift_mod_T"
    )


def test_shift_rejects_identity_modulo_length_and_sequence_symmetry() -> None:
    with pytest.raises(ExperimentContractError, match="must not be the identity"):
        transforms.transform_reference_input(
            _reference(250),
            condition_id=transforms.C_SHIFT_INPUT,
            current_frame=0,
        )

    half = _reference(250)
    periodic = np.concatenate((half, half), axis=0)
    with pytest.raises(ExperimentContractError, match="symmetry"):
        transforms.transform_reference_input(
            periodic,
            condition_id=transforms.C_SHIFT_INPUT,
            current_frame=0,
        )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ([[0.0] * 45, [1.0] * 45], "NumPy array"),
        (np.zeros((2, 45), dtype="<f4"), "little-endian float64"),
        (np.zeros((2, 45), dtype=">f8"), "little-endian float64"),
        (np.asfortranarray(np.zeros((2, 45), dtype="<f8")), "C-order"),
        (np.zeros((1, 45), dtype="<f8"), "at least two"),
        (np.zeros((2, 44), dtype="<f8"), r"shape \(T, 45\)"),
        (np.zeros((2, 45, 1), dtype="<f8"), r"shape \(T, 45\)"),
        (np.full((2, 45), np.nan, dtype="<f8"), "finite"),
        (np.zeros((2, 45), dtype="<f8"), "vary over time"),
    ],
)
def test_source_contract_fails_closed(source: object, message: str) -> None:
    with pytest.raises(ExperimentContractError, match=message):
        transforms.transform_reference_input(
            source,  # type: ignore[arg-type]
            condition_id=transforms.C_EXACT,
            current_frame=0,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"condition_id": "C_unknown"}, "condition_id"),
        ({"current_frame": -1}, "current_frame"),
        ({"current_frame": 1001}, "current_frame"),
        ({"current_frame": True}, "current_frame"),
        ({"horizon_steps": 7}, "horizon_steps must equal 8"),
        ({"horizon_steps": True}, "horizon_steps must equal 8"),
        ({"shuffle_seed": 260908}, "shuffle_seed must equal 260907"),
        ({"shuffle_seed": True}, "shuffle_seed must equal 260907"),
        ({"shift_frames": 249}, "shift_frames must equal 250"),
        ({"shift_frames": True}, "shift_frames must equal 250"),
    ],
)
def test_frozen_parameters_fail_closed(overrides: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {
        "condition_id": transforms.C_EXACT,
        "current_frame": 0,
    }
    arguments.update(overrides)
    with pytest.raises(ExperimentContractError, match=message):
        transforms.transform_reference_input(_reference(), **arguments)  # type: ignore[arg-type]


def test_receipt_is_canonical_content_addressed_and_binds_every_derivation() -> None:
    result = transforms.transform_reference_input(
        _reference(13),
        condition_id=transforms.C_SHIFT_INPUT,
        current_frame=10,
    )
    payload = result.receipt.to_dict()
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    canonical_index_map = json.dumps(
        {
            "algorithm_id": transforms.INDEX_MAP_HASH_ALGORITHM_ID,
            "index_map": payload["index_map"],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")

    assert payload["schema_version"] == 1
    assert payload["algorithm_id"] == "experiment_002a_numeric_reference_input/v1"
    assert payload["array_hash_algorithm_id"] == ("framed_c_order_little_endian_float64_array/v1")
    assert payload["index_map_hash_algorithm_id"] == "canonical_json_index_map/v1"
    assert payload["condition_id"] == transforms.C_SHIFT_INPUT
    assert payload["source_shape"] == [13, 45]
    assert payload["output_window_shape"] == [8, 45]
    assert len(payload["source_sha256"]) == 64
    assert len(payload["transformed_full_sequence_sha256"]) == 64
    assert len(payload["index_map_sha256"]) == 64
    assert len(payload["output_window_sha256"]) == 64
    assert payload["index_map_sha256"] == hashlib.sha256(canonical_index_map).hexdigest()
    assert payload["transformed_full_sequence_sha256"] == transforms.float64_array_sha256(
        result.transformed_full_sequence
    )
    assert payload["output_window_sha256"] == transforms.float64_array_sha256(result.window)
    assert result.receipt.sha256 == hashlib.sha256(canonical).hexdigest()
    assert result.receipt.sha256 == result.receipt.sha256
