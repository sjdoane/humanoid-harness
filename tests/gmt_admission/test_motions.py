from __future__ import annotations

import hashlib
import struct
from dataclasses import replace

import numpy as np
import pytest

from oracle_composition.adapters.gmt.contracts import MOTION_SPECS, MotionArraySpan, MotionSpec
from oracle_composition.adapters.gmt.io import GMTAdmissionError, write_deterministic_npz
from oracle_composition.adapters.gmt.motions import _extract_motion_for_spec, load_converted_motion


def _motion_fixture() -> tuple[bytearray, MotionSpec]:
    payload = bytearray(220)
    payload[:2] = b"\x80\x04"
    payload[20] = 0x47
    payload[21:29] = struct.pack(">d", 30.0)
    spans = (
        MotionArraySpan("root_pos", 40, (1, 3)),
        MotionArraySpan("root_rot", 60, (1, 4)),
        MotionArraySpan("dof_pos", 100, (1, 23)),
    )
    values = {
        "root_pos": np.asarray([[1.0, 2.0, 3.0]], dtype="<f4"),
        "root_rot": np.asarray([[0.0, 0.0, 0.0, 0.95]], dtype="<f4"),
        "dof_pos": np.arange(23, dtype="<f4").reshape(1, 23),
    }
    for span in spans:
        payload[span.offset - 5] = 0x42
        payload[span.offset - 4 : span.offset] = span.nbytes.to_bytes(4, "little")
        payload[span.offset : span.offset + span.nbytes] = values[span.key].tobytes()
    spec = MotionSpec(
        name="fixture",
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
        frames=1,
        fps=30.0,
        fps_opcode="BINFLOAT",
        arrays=spans,
    )
    return payload, spec


def test_motion_literal_extraction_preserves_nonunit_quaternion(tmp_path) -> None:
    payload, spec = _motion_fixture()
    path = tmp_path / "fixture.pkl"
    path.write_bytes(payload)

    arrays = _extract_motion_for_spec(path, spec)

    assert arrays["fps"].item() == 30.0
    assert arrays["fps"].shape == (1,)
    assert arrays["root_rot"][0, 3] == pytest.approx(0.95)
    assert np.linalg.norm(arrays["root_rot"][0]) == pytest.approx(0.95)
    np.testing.assert_array_equal(arrays["dof_pos"], np.arange(23, dtype="<f4")[None, :])


def test_motion_rejects_malformed_declared_binbytes(tmp_path) -> None:
    payload, spec = _motion_fixture()
    payload[35] = 0x00
    path = tmp_path / "fixture.pkl"
    path.write_bytes(payload)
    tampered_spec = replace(spec, sha256=hashlib.sha256(payload).hexdigest())

    with pytest.raises(GMTAdmissionError, match="BINBYTES"):
        _extract_motion_for_spec(path, tampered_spec)


def test_motion_rejects_nonfinite_numeric_payload(tmp_path) -> None:
    payload, spec = _motion_fixture()
    payload[40:44] = np.asarray([np.nan], dtype="<f4").tobytes()
    path = tmp_path / "fixture.pkl"
    path.write_bytes(payload)
    tampered_spec = replace(spec, sha256=hashlib.sha256(payload).hexdigest())

    with pytest.raises(GMTAdmissionError, match="non-finite"):
        _extract_motion_for_spec(path, tampered_spec)


def test_converted_motion_loader_checks_stored_shape_and_exact_hash(tmp_path) -> None:
    spec = MOTION_SPECS["walk_stand"]
    arrays = {
        "fps": np.asarray([spec.fps], dtype="<f8"),
        "root_pos": np.zeros((spec.frames, 3), dtype="<f4"),
        "root_rot": np.tile(np.asarray([0.0, 0.0, 0.0, 0.98], dtype="<f4"), (spec.frames, 1)),
        "dof_pos": np.zeros((spec.frames, 23), dtype="<f4"),
    }
    path = tmp_path / "walk_stand.npz"
    digest = write_deterministic_npz(path, arrays)

    loaded = load_converted_motion(path, name="walk_stand", expected_sha256=digest)

    assert loaded["fps"].shape == (1,)
    assert loaded["fps"].item() == spec.fps
    with pytest.raises(GMTAdmissionError, match="SHA-256 mismatch"):
        load_converted_motion(path, name="walk_stand", expected_sha256="0" * 64)
