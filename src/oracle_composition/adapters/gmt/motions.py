"""Literal extraction of the eight hash-pinned GMT motion pickles."""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path

import numpy as np

from .contracts import GMT_UPSTREAM_COMMIT, MOTION_SPECS, MotionSpec
from .io import GMTAdmissionError, read_verified_bytes, write_deterministic_npz, write_json_receipt


def _decode_fps(payload: bytes, spec: MotionSpec) -> float:
    offset = 20
    if spec.fps_opcode == "BINFLOAT":
        if payload[offset] != 0x47:
            raise GMTAdmissionError(f"{spec.name}: expected BINFLOAT fps opcode")
        observed = struct.unpack(">d", payload[offset + 1 : offset + 9])[0]
    elif spec.fps_opcode == "BININT1":
        if payload[offset] != 0x4B:
            raise GMTAdmissionError(f"{spec.name}: expected BININT1 fps opcode")
        observed = float(payload[offset + 1])
    else:
        raise GMTAdmissionError(f"{spec.name}: unsupported declared fps opcode")
    if observed != spec.fps:
        raise GMTAdmissionError(
            f"{spec.name}: fps mismatch: expected {spec.fps}, observed {observed}"
        )
    return observed


def _extract_motion_for_spec(source: Path, spec: MotionSpec) -> dict[str, np.ndarray]:
    source = Path(source)
    payload = read_verified_bytes(source, spec.sha256, expected_size=spec.size)
    if payload[:2] != b"\x80\x04":
        raise GMTAdmissionError(f"{spec.name}: expected pickle protocol 4 marker")
    arrays: dict[str, np.ndarray] = {"fps": np.asarray(_decode_fps(payload, spec), dtype="<f8")}
    for span in spec.arrays:
        header_offset = span.offset - 5
        if header_offset < 0 or payload[header_offset] != 0x42:
            raise GMTAdmissionError(f"{spec.name}/{span.key}: expected BINBYTES envelope")
        declared_size = int.from_bytes(payload[header_offset + 1 : span.offset], "little")
        if declared_size != span.nbytes:
            raise GMTAdmissionError(
                f"{spec.name}/{span.key}: payload size mismatch: "
                f"expected {span.nbytes}, observed {declared_size}"
            )
        end = span.offset + span.nbytes
        if end > len(payload):
            raise GMTAdmissionError(f"{spec.name}/{span.key}: truncated payload")
        array = np.frombuffer(payload[span.offset : end], dtype="<f4").copy().reshape(span.shape)
        if not np.isfinite(array).all():
            raise GMTAdmissionError(f"{spec.name}/{span.key}: non-finite values")
        arrays[span.key] = array
    return arrays


def extract_motion_arrays(source: Path, *, name: str | None = None) -> dict[str, np.ndarray]:
    source = Path(source)
    motion_name = name or source.stem
    try:
        spec = MOTION_SPECS[motion_name]
    except KeyError as exc:
        raise GMTAdmissionError(f"motion is not in the pinned GMT catalog: {motion_name}") from exc
    return _extract_motion_for_spec(source, spec)


def _motion_receipt(
    spec: MotionSpec, output: Path, output_sha256: str, arrays: dict[str, np.ndarray]
) -> dict[str, object]:
    quaternion_norm = np.linalg.norm(arrays["root_rot"], axis=1)
    return {
        "schema_version": 1,
        "artifact": "gmt_g1_motion",
        "motion": spec.name,
        "source": {
            "upstream_commit": GMT_UPSTREAM_COMMIT,
            "sha256": spec.sha256,
            "size": spec.size,
            "format": "numpy_pickle_protocol_4_not_executed",
        },
        "output": {
            "format": "numeric_only_npz",
            "sha256": output_sha256,
            "size": output.stat().st_size,
        },
        "motion_contract": {
            "fps": spec.fps,
            "frames": spec.frames,
            "quaternion_order": "xyzw",
            "quaternions_normalized_by_converter": False,
            "quaternion_norm_min": float(quaternion_norm.min()),
            "quaternion_norm_max": float(quaternion_norm.max()),
            "arrays": {
                key: {"shape": list(value.shape), "dtype": value.dtype.str}
                for key, value in arrays.items()
            },
        },
    }


def convert_motion(source: Path, output: Path, *, name: str | None = None) -> dict[str, object]:
    source = Path(source)
    output = Path(output)
    motion_name = name or source.stem
    try:
        spec = MOTION_SPECS[motion_name]
    except KeyError as exc:
        raise GMTAdmissionError(f"motion is not in the pinned GMT catalog: {motion_name}") from exc
    arrays = _extract_motion_for_spec(source, spec)
    output_sha256 = write_deterministic_npz(output, arrays)
    receipt = _motion_receipt(spec, output, output_sha256, arrays)
    receipt_path = output.with_suffix(f"{output.suffix}.manifest.json")
    receipt_sha256 = write_json_receipt(receipt_path, receipt)
    receipt["receipt"] = {"path": receipt_path.name, "sha256": receipt_sha256}
    return receipt


def convert_motion_directory(source: Path, output: Path) -> list[dict[str, object]]:
    """Preflight all eight pinned inputs, then emit their numeric-only forms."""

    source = Path(source)
    output = Path(output)
    admitted = {
        name: _extract_motion_for_spec(source / f"{name}.pkl", spec)
        for name, spec in MOTION_SPECS.items()
    }
    destinations = [output / f"{name}.npz" for name in MOTION_SPECS]
    existing = [str(path) for path in destinations if path.exists()]
    existing.extend(
        str(path.with_suffix(f"{path.suffix}.manifest.json"))
        for path in destinations
        if path.with_suffix(f"{path.suffix}.manifest.json").exists()
    )
    if existing:
        raise GMTAdmissionError(f"refusing to overwrite motion outputs: {sorted(existing)}")
    receipts: list[dict[str, object]] = []
    for name, arrays in admitted.items():
        destination = output / f"{name}.npz"
        output_sha256 = write_deterministic_npz(destination, arrays)
        receipt = _motion_receipt(MOTION_SPECS[name], destination, output_sha256, arrays)
        receipt_path = destination.with_suffix(f"{destination.suffix}.manifest.json")
        receipt_sha256 = write_json_receipt(receipt_path, receipt)
        receipt["receipt"] = {"path": receipt_path.name, "sha256": receipt_sha256}
        receipts.append(receipt)
    return receipts


def motion_array_sha256(array: np.ndarray) -> str:
    """Digest exact C-order values for audit comparisons without model execution."""

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
