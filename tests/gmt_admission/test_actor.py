from __future__ import annotations

import hashlib
import zipfile

import numpy as np
import pytest
import torch

from oracle_composition.adapters.gmt.actor import (
    actor_from_arrays,
    apply_deployment_action,
    load_actor,
)
from oracle_composition.adapters.gmt.contracts import (
    DEFAULT_DOF_POSITION,
    OBSERVATION_DIM,
    PARAMETER_COUNT,
)
from oracle_composition.adapters.gmt.io import GMTAdmissionError, write_deterministic_npz


def test_plain_actor_is_finite_deterministic_and_frozen(
    synthetic_actor_arrays: dict[str, np.ndarray],
) -> None:
    actor = actor_from_arrays(synthetic_actor_arrays)
    observation = torch.zeros((2, OBSERVATION_DIM), dtype=torch.float32)

    with torch.no_grad():
        first = actor(observation)
        second = actor(observation)

    assert first.shape == (2, 23)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert torch.allclose(
        first[0], torch.from_numpy(synthetic_actor_arrays["actor_backbone.9.bias"])
    )
    assert all(not parameter.requires_grad for parameter in actor.parameters())
    assert sum(parameter.numel() for parameter in actor.parameters()) == PARAMETER_COUNT


def test_actor_loader_requires_converted_artifact_identity(
    tmp_path, synthetic_actor_arrays: dict[str, np.ndarray]
) -> None:
    path = tmp_path / "weights.npz"
    digest = write_deterministic_npz(path, synthetic_actor_arrays)
    actor = load_actor(path, expected_sha256=digest)

    assert isinstance(actor, torch.nn.Module)
    with pytest.raises(GMTAdmissionError, match="refusing to overwrite"):
        write_deterministic_npz(path, synthetic_actor_arrays)
    with pytest.raises(GMTAdmissionError, match="SHA-256 mismatch"):
        load_actor(path, expected_sha256="0" * 64)


def test_actor_loader_rejects_unexpected_npz_members(tmp_path) -> None:
    path = tmp_path / "malformed.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("unexpected.npy", b"not an NPY")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(GMTAdmissionError, match="member count mismatch"):
        load_actor(path, expected_sha256=digest)


def test_actor_rejects_malformed_arrays(
    synthetic_actor_arrays: dict[str, np.ndarray],
) -> None:
    missing = dict(synthetic_actor_arrays)
    missing.pop("actor_backbone.9.bias")
    with pytest.raises(GMTAdmissionError, match="tensor key mismatch"):
        actor_from_arrays(missing)

    wrong_shape = dict(synthetic_actor_arrays)
    wrong_shape["actor_backbone.9.bias"] = np.zeros((22,), dtype="<f4")
    with pytest.raises(GMTAdmissionError, match="shape mismatch"):
        actor_from_arrays(wrong_shape)

    non_finite = {key: value.copy() for key, value in synthetic_actor_arrays.items()}
    non_finite["normalizer_mean"][0] = np.nan
    with pytest.raises(GMTAdmissionError, match="non-finite"):
        actor_from_arrays(non_finite)


def test_actor_rejects_wrong_observation_width(
    synthetic_actor_arrays: dict[str, np.ndarray],
) -> None:
    actor = actor_from_arrays(synthetic_actor_arrays)
    with pytest.raises(ValueError, match="GMT observation"):
        actor(torch.zeros((1, OBSERVATION_DIM - 1)))


def test_action_adapter_preserves_preclip_history() -> None:
    raw = torch.tensor([[-12.0, *([0.0] * 21), 12.0]])
    history, target = apply_deployment_action(raw)
    expected = torch.tensor(DEFAULT_DOF_POSITION)
    expected[0] -= 5.0
    expected[-1] += 5.0

    assert torch.equal(history, raw)
    assert torch.equal(target[0], expected)
