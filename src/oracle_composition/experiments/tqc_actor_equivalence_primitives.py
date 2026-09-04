"""Pure fixed-batch TQC actor equivalence primitives."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from .fixed_reference import ExperimentContractError
from .tqc_actor_npz import OBSERVATION_WIDTH

EQUIVALENCE_OBSERVATION_SHA256 = "0c6a81b06a88cab7eca0255e75f021008b60025c4ddc4d3719426e3647159ec6"
EQUIVALENCE_OBSERVATION_SHAPE = (4, OBSERVATION_WIDTH)
EQUIVALENCE_SAMPLING_SEED = 97_001
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExperimentContractError(f"value is not canonical JSON: {exc}") from exc


def canonical_array_sha256(value: np.ndarray) -> str:
    """Hash one C-order array with its exact dtype and shape."""

    if not isinstance(value, np.ndarray):
        raise ExperimentContractError("hashed value must be a NumPy array")
    if not value.flags.c_contiguous:
        raise ExperimentContractError("hashed array must use C-order storage")
    metadata = _canonical_json({"dtype": value.dtype.str, "shape": list(value.shape)})
    raw = value.tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def equivalence_observations() -> np.ndarray:
    """Build the frozen four-observation affine grid."""

    indices = np.arange(4 * OBSERVATION_WIDTH, dtype=np.int64).reshape(
        EQUIVALENCE_OBSERVATION_SHAPE
    )
    values = (((indices * 37 + 11) % 257) - 128).astype("<f4") / np.float32(64.0)
    result = np.ascontiguousarray(values, dtype="<f4")
    if canonical_array_sha256(result) != EQUIVALENCE_OBSERVATION_SHA256:
        raise ExperimentContractError("TQC equivalence observation batch differs")
    return result


def with_seeded_cpu_rng(computation: object, *, sampling_seed: int) -> object:
    """Run one Torch computation under a private CPU RNG state."""

    import torch

    if not callable(computation):
        raise ExperimentContractError("seeded computation must be callable")
    before = torch.random.get_rng_state().clone()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(sampling_seed)
    try:
        torch.random.set_rng_state(generator.get_state())
        return computation()
    finally:
        torch.random.set_rng_state(before)


def seeded_squashed_normal_sample(
    mean: object,
    log_std: object,
    *,
    sampling_seed: int,
) -> object:
    """Use the frozen reparameterized Normal sample and tanh transform."""

    import torch

    return with_seeded_cpu_rng(
        lambda: torch.tanh(torch.distributions.Normal(mean, torch.exp(log_std)).rsample()),
        sampling_seed=sampling_seed,
    )


__all__ = [
    "EQUIVALENCE_OBSERVATION_SHA256",
    "EQUIVALENCE_OBSERVATION_SHAPE",
    "EQUIVALENCE_SAMPLING_SEED",
    "LOG_STD_MAX",
    "LOG_STD_MIN",
    "canonical_array_sha256",
    "equivalence_observations",
    "seeded_squashed_normal_sample",
    "with_seeded_cpu_rng",
]
