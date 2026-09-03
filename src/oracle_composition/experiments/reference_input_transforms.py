"""Pure numeric-reference interventions for Experiment 002A."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import numpy as np

from .fixed_reference import CAUSAL_INPUT_CONDITION_IDS, ExperimentContractError

REFERENCE_WIDTH: Final = 45
HORIZON_STEPS: Final = 8
SHUFFLE_SEED: Final = 260_907
SHIFT_FRAMES: Final = 250

C_EXACT: Final = "C_exact"
C_ZERO_INPUT: Final = "C_zero_input"
C_SHUFFLE_INPUT: Final = "C_shuffle_input"
C_SHIFT_INPUT: Final = "C_shift_input"
CONDITION_IDS: Final = (
    C_EXACT,
    C_ZERO_INPUT,
    C_SHUFFLE_INPUT,
    C_SHIFT_INPUT,
)
if frozenset(CONDITION_IDS) != CAUSAL_INPUT_CONDITION_IDS:
    raise RuntimeError("ordered transform conditions differ from the experiment contract")

TRANSFORM_ALGORITHM_ID: Final = "experiment_002a_numeric_reference_input/v1"
SHUFFLE_RANK_ALGORITHM_ID: Final = "sha256_ranked_full_frame_indices/v1"
SHIFT_ALGORITHM_ID: Final = "cyclic_source_index_plus_250/v1"
ARRAY_HASH_ALGORITHM_ID: Final = "framed_c_order_little_endian_float64_array/v1"
INDEX_MAP_HASH_ALGORITHM_ID: Final = "canonical_json_index_map/v1"
RECEIPT_SCHEMA_VERSION: Final = 1

_FLOAT64_LE: Final = np.dtype("<f8")
_ZERO_BITS_HEX: Final = "0000000000000000"


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExperimentContractError(f"transform receipt is not canonical JSON: {exc}") from exc


def _framed_sha256(parts: tuple[tuple[bytes, bytes], ...]) -> str:
    digest = hashlib.sha256()
    for label, payload in parts:
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def float64_array_sha256(value: np.ndarray) -> str:
    """Hash one exact C-order little-endian float64 array with its shape."""

    array = _require_float64_array(value, field="hashed array", expected_width=None)
    metadata = _canonical_json(
        {
            "algorithm_id": ARRAY_HASH_ALGORITHM_ID,
            "dtype": _FLOAT64_LE.str,
            "order": "C",
            "shape": list(array.shape),
        }
    )
    return _framed_sha256(
        (
            (b"metadata", metadata),
            (b"values", array.tobytes(order="C")),
        )
    )


def _require_float64_array(
    value: object,
    *,
    field: str,
    expected_width: int | None,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ExperimentContractError(f"{field} must be a NumPy array")
    if value.dtype.str != _FLOAT64_LE.str:
        raise ExperimentContractError(f"{field} must use little-endian float64")
    if not value.flags.c_contiguous:
        raise ExperimentContractError(f"{field} must use C-order storage")
    if expected_width is not None and (value.ndim != 2 or value.shape[1] != expected_width):
        raise ExperimentContractError(f"{field} must have shape (T, {expected_width})")
    if value.ndim < 1:
        raise ExperimentContractError(f"{field} must have at least one dimension")
    if not np.isfinite(value).all():
        raise ExperimentContractError(f"{field} must contain only finite values")
    return value


def _validate_source(value: object) -> np.ndarray:
    source = _require_float64_array(
        value,
        field="source reference",
        expected_width=REFERENCE_WIDTH,
    )
    if source.shape[0] < 2:
        raise ExperimentContractError("source reference must contain at least two frames")
    if np.array_equal(source, np.broadcast_to(source[0], source.shape)):
        raise ExperimentContractError("source reference must vary over time")
    return source


def _require_exact_integer(value: object, *, expected: int, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        raise ExperimentContractError(f"{field} must equal {expected}")
    return value


def _index_map_sha256(index_map: tuple[int | None, ...]) -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "algorithm_id": INDEX_MAP_HASH_ALGORITHM_ID,
                "index_map": list(index_map),
            }
        )
    ).hexdigest()


def _shuffle_index_map(*, n_frames: int, seed: int) -> tuple[int, ...]:
    def rank(frame_index: int) -> tuple[bytes, int]:
        payload = _canonical_json(
            {
                "algorithm_id": SHUFFLE_RANK_ALGORITHM_ID,
                "frame_index": frame_index,
                "n_frames": n_frames,
                "seed": seed,
            }
        )
        return hashlib.sha256(payload).digest(), frame_index

    return tuple(sorted(range(n_frames), key=rank))


def _transform_parameters(
    condition_id: str,
    *,
    shuffle_seed: int,
    shift_frames: int,
) -> MappingProxyType[str, str | int]:
    common: dict[str, str | int] = {
        "horizon_steps": HORIZON_STEPS,
        "reference_width": REFERENCE_WIDTH,
    }
    if condition_id == C_EXACT:
        return MappingProxyType({**common, "operation": "identity"})
    if condition_id == C_ZERO_INPUT:
        return MappingProxyType(
            {
                **common,
                "fill_float64_bits_hex": _ZERO_BITS_HEX,
                "operation": "positive_zero_fill",
            }
        )
    if condition_id == C_SHUFFLE_INPUT:
        return MappingProxyType(
            {
                **common,
                "operation": "output_frame_i_reads_source_frame_at_rank_i",
                "ranking_algorithm_id": SHUFFLE_RANK_ALGORITHM_ID,
                "seed": shuffle_seed,
            }
        )
    return MappingProxyType(
        {
            **common,
            "operation": "output_frame_i_reads_source_frame_i_plus_shift_mod_T",
            "shift_algorithm_id": SHIFT_ALGORITHM_ID,
            "shift_frames": shift_frames,
        }
    )


@dataclass(frozen=True, slots=True)
class ReferenceInputTransformReceipt:
    """Content-addressed derivation of one policy-visible reference window."""

    condition_id: str
    source_shape: tuple[int, int]
    source_sha256: str
    transformed_full_sequence_sha256: str
    parameters: MappingProxyType[str, str | int]
    index_map: tuple[int | None, ...]
    index_map_sha256: str
    current_frame: int
    window_timeline_indices: tuple[int, ...]
    window_source_indices: tuple[int | None, ...]
    output_window_shape: tuple[int, int]
    output_window_sha256: str
    schema_version: int = RECEIPT_SCHEMA_VERSION
    algorithm_id: str = TRANSFORM_ALGORITHM_ID
    array_hash_algorithm_id: str = ARRAY_HASH_ALGORITHM_ID
    index_map_hash_algorithm_id: str = INDEX_MAP_HASH_ALGORITHM_ID
    dtype: str = "<f8"
    order: str = "C"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "algorithm_id": self.algorithm_id,
            "array_hash_algorithm_id": self.array_hash_algorithm_id,
            "index_map_hash_algorithm_id": self.index_map_hash_algorithm_id,
            "condition_id": self.condition_id,
            "dtype": self.dtype,
            "order": self.order,
            "source_shape": list(self.source_shape),
            "source_sha256": self.source_sha256,
            "transformed_full_sequence_sha256": self.transformed_full_sequence_sha256,
            "parameters": dict(self.parameters),
            "index_map": list(self.index_map),
            "index_map_sha256": self.index_map_sha256,
            "current_frame": self.current_frame,
            "window_timeline_indices": list(self.window_timeline_indices),
            "window_source_indices": list(self.window_source_indices),
            "output_window_shape": list(self.output_window_shape),
            "output_window_sha256": self.output_window_sha256,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()


@dataclass(frozen=True, slots=True)
class ReferenceInputTransformResult:
    """Immutable arrays and receipt for one intervention."""

    transformed_full_sequence: np.ndarray
    window: np.ndarray
    receipt: ReferenceInputTransformReceipt


def transform_reference_input(
    source_reference: np.ndarray,
    *,
    condition_id: str,
    current_frame: int,
    horizon_steps: int = HORIZON_STEPS,
    shuffle_seed: int = SHUFFLE_SEED,
    shift_frames: int = SHIFT_FRAMES,
) -> ReferenceInputTransformResult:
    """Apply one frozen intervention and return its terminal-hold window."""

    source = _validate_source(source_reference)
    if condition_id not in CONDITION_IDS:
        raise ExperimentContractError(f"condition_id must be one of {list(CONDITION_IDS)!r}")
    _require_exact_integer(horizon_steps, expected=HORIZON_STEPS, field="horizon_steps")
    _require_exact_integer(shuffle_seed, expected=SHUFFLE_SEED, field="shuffle_seed")
    _require_exact_integer(shift_frames, expected=SHIFT_FRAMES, field="shift_frames")
    if (
        not isinstance(current_frame, int)
        or isinstance(current_frame, bool)
        or not 0 <= current_frame < source.shape[0]
    ):
        raise ExperimentContractError("current_frame must index the source reference")

    n_frames = source.shape[0]
    if condition_id == C_EXACT:
        index_map: tuple[int | None, ...] = tuple(range(n_frames))
        transformed = source.copy(order="C")
    elif condition_id == C_ZERO_INPUT:
        index_map = (None,) * n_frames
        transformed = np.zeros(source.shape, dtype=_FLOAT64_LE, order="C")
    elif condition_id == C_SHUFFLE_INPUT:
        shuffled = _shuffle_index_map(n_frames=n_frames, seed=shuffle_seed)
        if tuple(sorted(shuffled)) != tuple(range(n_frames)):
            raise ExperimentContractError("shuffle index map must be a complete permutation")
        if shuffled == tuple(range(n_frames)):
            raise ExperimentContractError("shuffle index map must not be the identity")
        index_map = shuffled
        transformed = source[np.asarray(shuffled, dtype=np.int64)].copy(order="C")
        if np.array_equal(transformed, source):
            raise ExperimentContractError("shuffle is value-equivalent to the source reference")
    else:
        effective_shift = shift_frames % n_frames
        if effective_shift == 0:
            raise ExperimentContractError("cyclic shift must not be the identity")
        shifted = tuple((index + shift_frames) % n_frames for index in range(n_frames))
        index_map = shifted
        transformed = source[np.asarray(shifted, dtype=np.int64)].copy(order="C")
        if np.array_equal(transformed, source):
            raise ExperimentContractError("cyclic shift is a symmetry of the source reference")

    if transformed.dtype.str != _FLOAT64_LE.str or not transformed.flags.c_contiguous:
        raise ExperimentContractError("transformed reference violated the float64 C-order contract")
    timeline_indices = tuple(
        min(current_frame + offset, n_frames - 1) for offset in range(horizon_steps)
    )
    source_indices = tuple(index_map[index] for index in timeline_indices)
    window = transformed[np.asarray(timeline_indices, dtype=np.int64)].copy(order="C")
    transformed = np.frombuffer(transformed.tobytes(order="C"), dtype=_FLOAT64_LE).reshape(
        transformed.shape
    )
    window = np.frombuffer(window.tobytes(order="C"), dtype=_FLOAT64_LE).reshape(window.shape)

    receipt = ReferenceInputTransformReceipt(
        condition_id=condition_id,
        source_shape=(n_frames, REFERENCE_WIDTH),
        source_sha256=float64_array_sha256(source),
        transformed_full_sequence_sha256=float64_array_sha256(transformed),
        parameters=_transform_parameters(
            condition_id,
            shuffle_seed=shuffle_seed,
            shift_frames=shift_frames,
        ),
        index_map=index_map,
        index_map_sha256=_index_map_sha256(index_map),
        current_frame=current_frame,
        window_timeline_indices=timeline_indices,
        window_source_indices=source_indices,
        output_window_shape=(horizon_steps, REFERENCE_WIDTH),
        output_window_sha256=float64_array_sha256(window),
    )
    return ReferenceInputTransformResult(
        transformed_full_sequence=transformed,
        window=window,
        receipt=receipt,
    )


__all__ = [
    "ARRAY_HASH_ALGORITHM_ID",
    "CONDITION_IDS",
    "C_EXACT",
    "C_SHIFT_INPUT",
    "C_SHUFFLE_INPUT",
    "C_ZERO_INPUT",
    "HORIZON_STEPS",
    "INDEX_MAP_HASH_ALGORITHM_ID",
    "RECEIPT_SCHEMA_VERSION",
    "REFERENCE_WIDTH",
    "SHIFT_ALGORITHM_ID",
    "SHIFT_FRAMES",
    "SHUFFLE_RANK_ALGORITHM_ID",
    "SHUFFLE_SEED",
    "TRANSFORM_ALGORITHM_ID",
    "ReferenceInputTransformReceipt",
    "ReferenceInputTransformResult",
    "float64_array_sha256",
    "transform_reference_input",
]
