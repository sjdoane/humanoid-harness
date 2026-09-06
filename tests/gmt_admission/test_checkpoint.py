from __future__ import annotations

import hashlib
import zipfile

import numpy as np
import pytest

from oracle_composition.adapters.gmt.checkpoint import (
    _ArchiveProfile,
    _extract_declared_storages,
)
from oracle_composition.adapters.gmt.contracts import (
    ACTION_DIM,
    DAMPING,
    DEFAULT_DOF_POSITION,
    JOINT_NAMES,
    PARAMETER_COUNT,
    REFERENCE_OFFSETS,
    STIFFNESS,
    TENSOR_SPECS,
    TORQUE_LIMITS,
    TensorSpec,
)
from oracle_composition.adapters.gmt.io import GMTAdmissionError


def _write_fixture_archive(path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def _profile(path, tensors: tuple[TensorSpec, ...]) -> _ArchiveProfile:
    return _ArchiveProfile(
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size=path.stat().st_size,
        prefix="fixture/",
        tensors=tensors,
    )


def test_declared_storage_extractor_never_needs_pickle(tmp_path) -> None:
    specs = (
        TensorSpec(0, "float_values", (2,)),
        TensorSpec(1, "integer_value", (1,), "<i8"),
    )
    path = tmp_path / "fixture.pt"
    _write_fixture_archive(
        path,
        {
            "fixture/data/0": np.asarray([1.5, -2.0], dtype="<f4").tobytes(),
            "fixture/data/1": np.asarray([7], dtype="<i8").tobytes(),
        },
    )

    arrays = _extract_declared_storages(path, _profile(path, specs))

    np.testing.assert_array_equal(arrays["float_values"], [1.5, -2.0])
    np.testing.assert_array_equal(arrays["integer_value"], [7])


def test_checkpoint_tampering_fails_before_zip_interpretation(tmp_path) -> None:
    spec = TensorSpec(0, "value", (1,))
    path = tmp_path / "fixture.pt"
    _write_fixture_archive(path, {"fixture/data/0": np.asarray([1], dtype="<f4").tobytes()})
    profile = _profile(path, (spec,))
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(GMTAdmissionError, match="size mismatch"):
        _extract_declared_storages(path, profile)


def test_checkpoint_rejects_missing_declared_storage(tmp_path) -> None:
    specs = (TensorSpec(0, "first", (1,)), TensorSpec(1, "second", (1,)))
    path = tmp_path / "fixture.pt"
    _write_fixture_archive(path, {"fixture/data/0": np.asarray([1], dtype="<f4").tobytes()})

    with pytest.raises(GMTAdmissionError, match="storage member set"):
        _extract_declared_storages(path, _profile(path, specs))


def test_checkpoint_and_controller_contract_is_complete() -> None:
    assert len(TENSOR_SPECS) == 31
    assert {spec.storage for spec in TENSOR_SPECS} == set(range(31))
    assert all(spec.strides[-1] == 1 for spec in TENSOR_SPECS)
    assert sum(np.prod(spec.shape) for spec in TENSOR_SPECS[:28]) == PARAMETER_COUNT
    assert len(JOINT_NAMES) == ACTION_DIM
    assert len(DEFAULT_DOF_POSITION) == ACTION_DIM
    assert len(STIFFNESS) == ACTION_DIM
    assert len(DAMPING) == ACTION_DIM
    assert len(TORQUE_LIMITS) == ACTION_DIM
    assert len(REFERENCE_OFFSETS) == 20
    assert REFERENCE_OFFSETS[0] == 1
    assert REFERENCE_OFFSETS[-1] == 95
