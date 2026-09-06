from __future__ import annotations

import numpy as np
import pytest

from oracle_composition.adapters.gmt.contracts import TENSOR_SPECS


@pytest.fixture
def synthetic_actor_arrays() -> dict[str, np.ndarray]:
    arrays = {spec.key: np.zeros(spec.shape, dtype=spec.dtype) for spec in TENSOR_SPECS}
    arrays["normalizer_count"][0] = 491_520_000
    arrays["normalizer_std"].fill(1.0)
    arrays["actor_backbone.7.weight"].fill(1.0)
    arrays["actor_backbone.9.bias"][:] = np.linspace(-0.5, 0.5, 23, dtype="<f4")
    return arrays
