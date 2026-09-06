"""Safe conversion of the one hash-pinned GMT TorchScript checkpoint."""

from __future__ import annotations

import hashlib
import io
import struct
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import (
    ACTION_DIM,
    ACTION_SCALE,
    CURRENT_PROPRIOCEPTION_SLICE,
    CURRENT_REFERENCE_SLICE,
    GENERATED_MEMBER_SHA256,
    GENERATED_OPERATOR_ALLOWLIST,
    GMT_ARCHIVE_MEMBER_COUNT,
    GMT_ARCHIVE_PREFIX,
    GMT_ARCHIVE_UNCOMPRESSED_SIZE,
    GMT_CHECKPOINT_SHA256,
    GMT_CHECKPOINT_SIZE,
    GMT_G1_MESH_NAMES,
    GMT_G1_MESH_TREE_SHA256,
    GMT_SUPPORT_FILE_SHA256,
    GMT_UPSTREAM_COMMIT,
    LAYER_NORM_EPSILON,
    NORMALIZER_EPSILON,
    OBSERVATION_DIM,
    PARAMETER_COUNT,
    PROPRIOCEPTION_HISTORY_SLICE,
    RAW_ACTION_MAX,
    RAW_ACTION_MIN,
    REFERENCE_HORIZON,
    REFERENCE_OFFSETS,
    REFERENCE_SLICE,
    TENSOR_SPECS,
    TensorSpec,
)
from .io import (
    GMTAdmissionError,
    read_verified_bytes,
    require_sha256,
    sha256_file,
    validate_zip_members,
    write_deterministic_npz,
    write_json_receipt,
)


@dataclass(frozen=True)
class _ArchiveProfile:
    sha256: str
    size: int
    prefix: str
    tensors: Sequence[TensorSpec]
    member_count: int | None = None
    uncompressed_size: int | None = None
    generated_member_sha256: Mapping[str, str] | None = None


_GMT_PROFILE = _ArchiveProfile(
    sha256=GMT_CHECKPOINT_SHA256,
    size=GMT_CHECKPOINT_SIZE,
    prefix=GMT_ARCHIVE_PREFIX,
    tensors=TENSOR_SPECS,
    member_count=GMT_ARCHIVE_MEMBER_COUNT,
    uncompressed_size=GMT_ARCHIVE_UNCOMPRESSED_SIZE,
    generated_member_sha256=GENERATED_MEMBER_SHA256,
)


def _validate_numeric_arrays(arrays: Mapping[str, np.ndarray], specs: Sequence[TensorSpec]) -> None:
    expected_keys = {spec.key for spec in specs}
    if set(arrays) != expected_keys:
        missing = sorted(expected_keys - set(arrays))
        extra = sorted(set(arrays) - expected_keys)
        raise GMTAdmissionError(f"tensor key mismatch: missing={missing}, extra={extra}")
    for spec in specs:
        array = arrays[spec.key]
        if array.shape != spec.shape:
            raise GMTAdmissionError(
                f"shape mismatch for {spec.key}: expected {spec.shape}, observed {array.shape}"
            )
        if array.dtype.str != spec.dtype:
            raise GMTAdmissionError(
                f"dtype mismatch for {spec.key}: expected {spec.dtype}, observed {array.dtype.str}"
            )
        if not array.flags.c_contiguous:
            raise GMTAdmissionError(f"non-contiguous tensor: {spec.key}")
        if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
            raise GMTAdmissionError(f"non-finite tensor: {spec.key}")


def _extract_declared_storages(path: Path, profile: _ArchiveProfile) -> dict[str, np.ndarray]:
    payload = read_verified_bytes(path, profile.sha256, expected_size=profile.size)
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise GMTAdmissionError(f"invalid ZIP archive: {path.name}") from exc

    with archive:
        members = validate_zip_members(
            archive,
            expected_count=profile.member_count,
            expected_uncompressed_size=profile.uncompressed_size,
            required_prefix=profile.prefix,
        )
        storage_names = {f"{profile.prefix}data/{spec.storage}" for spec in profile.tensors}
        observed_storage_names = {
            name
            for name in members
            if name.startswith(f"{profile.prefix}data/")
            and name.removeprefix(f"{profile.prefix}data/").isdigit()
        }
        if observed_storage_names != storage_names:
            raise GMTAdmissionError("checkpoint storage member set does not match the declaration")

        for relative_name, expected_digest in (profile.generated_member_sha256 or {}).items():
            member_name = f"{profile.prefix}{relative_name}"
            if member_name not in members:
                raise GMTAdmissionError(f"missing declared graph member: {relative_name}")
            observed_digest = hashlib.sha256(archive.read(member_name)).hexdigest()
            if observed_digest != expected_digest:
                raise GMTAdmissionError(f"graph member digest mismatch: {relative_name}")

        if profile is _GMT_PROFILE:
            if archive.read(f"{profile.prefix}constants/0") != struct.pack("<d", 1.0e-4):
                raise GMTAdmissionError("unexpected normalization epsilon constant")
            if archive.read(f"{profile.prefix}constants/1") != struct.pack("<q", 20):
                raise GMTAdmissionError("unexpected temporal-horizon constant")

        arrays: dict[str, np.ndarray] = {}
        for spec in profile.tensors:
            member_name = f"{profile.prefix}data/{spec.storage}"
            info = members.get(member_name)
            if info is None:
                raise GMTAdmissionError(f"missing tensor storage {spec.storage}")
            if info.file_size != spec.nbytes:
                raise GMTAdmissionError(
                    f"storage size mismatch for {spec.key}: "
                    f"expected {spec.nbytes}, observed {info.file_size}"
                )
            payload = archive.read(member_name)
            if len(payload) != spec.nbytes:
                raise GMTAdmissionError(f"truncated tensor storage: {spec.key}")
            arrays[spec.key] = np.frombuffer(payload, dtype=spec.dtype).copy().reshape(spec.shape)

    _validate_numeric_arrays(arrays, profile.tensors)
    if profile is _GMT_PROFILE:
        count = int(arrays["normalizer_count"][0])
        if count != 491_520_000:
            raise GMTAdmissionError(f"unexpected normalizer count: {count}")
        standard_deviation = arrays["normalizer_std"]
        if np.any(standard_deviation <= 0):
            raise GMTAdmissionError("normalizer standard deviation must be strictly positive")
    return arrays


def extract_checkpoint_arrays(path: Path) -> dict[str, np.ndarray]:
    """Extract declared numeric storages without interpreting any pickle program."""

    return _extract_declared_storages(Path(path), _GMT_PROFILE)


def _tensor_receipts(arrays: Mapping[str, np.ndarray]) -> list[dict[str, object]]:
    return [
        {
            "key": spec.key,
            "storage": spec.storage,
            "shape": list(spec.shape),
            "dtype": spec.dtype,
            "strides": list(spec.strides),
            "sha256": hashlib.sha256(arrays[spec.key].tobytes(order="C")).hexdigest(),
        }
        for spec in TENSOR_SPECS
    ]


def convert_checkpoint(source: Path, output: Path) -> dict[str, object]:
    """Convert the exact upstream checkpoint to deterministic numeric-only NPZ."""

    source = Path(source)
    output = Path(output)
    arrays = extract_checkpoint_arrays(source)
    output_sha256 = write_deterministic_npz(output, arrays)
    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact": "gmt_g1_plain_actor_weights",
        "source": {
            "upstream_commit": GMT_UPSTREAM_COMMIT,
            "checkpoint_sha256": GMT_CHECKPOINT_SHA256,
            "checkpoint_size": GMT_CHECKPOINT_SIZE,
        },
        "output": {
            "format": "numeric_only_npz",
            "sha256": output_sha256,
            "size": output.stat().st_size,
        },
        "actor_contract": {
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            "parameter_count": PARAMETER_COUNT,
            "normalizer_epsilon": NORMALIZER_EPSILON,
            "layer_norm_epsilon": LAYER_NORM_EPSILON,
            "jit_equivalence": "not_tested",
            "generated_operator_allowlist": list(GENERATED_OPERATOR_ALLOWLIST),
            "observation_layout": {
                "reference": [REFERENCE_SLICE.start, REFERENCE_SLICE.stop],
                "current_reference": [
                    CURRENT_REFERENCE_SLICE.start,
                    CURRENT_REFERENCE_SLICE.stop,
                ],
                "current_proprioception": [
                    CURRENT_PROPRIOCEPTION_SLICE.start,
                    CURRENT_PROPRIOCEPTION_SLICE.stop,
                ],
                "proprioception_history": [
                    PROPRIOCEPTION_HISTORY_SLICE.start,
                    PROPRIOCEPTION_HISTORY_SLICE.stop,
                ],
                "reference_horizon": REFERENCE_HORIZON,
            },
            "reference_offsets_at_50hz": list(REFERENCE_OFFSETS),
            "action_sequence": {
                "history": "raw_preclip",
                "clip": [RAW_ACTION_MIN, RAW_ACTION_MAX],
                "scale": ACTION_SCALE,
                "target": "clipped_scaled_plus_default_pose",
            },
        },
        "tensors": _tensor_receipts(arrays),
    }
    receipt_path = output.with_suffix(f"{output.suffix}.manifest.json")
    receipt_sha256 = write_json_receipt(receipt_path, receipt)
    receipt["receipt"] = {"path": receipt_path.name, "sha256": receipt_sha256}
    return receipt


def load_converted_arrays(path: Path, *, expected_sha256: str) -> dict[str, np.ndarray]:
    """Load a converted NPZ only after exact identity and bounded-member checks."""

    path = Path(path)
    payload = read_verified_bytes(path, expected_sha256, maximum_size=9 * 1024 * 1024)
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise GMTAdmissionError(f"invalid converted NPZ: {path.name}") from exc

    expected_members = {f"{spec.key}.npy": spec for spec in TENSOR_SPECS}
    arrays: dict[str, np.ndarray] = {}
    with archive:
        members = validate_zip_members(
            archive,
            expected_count=len(TENSOR_SPECS),
            maximum_member_size=5 * 1024 * 1024,
        )
        if set(members) != set(expected_members):
            raise GMTAdmissionError("converted NPZ member set does not match the actor contract")
        for member_name, spec in expected_members.items():
            info = members[member_name]
            if info.compress_type != zipfile.ZIP_STORED:
                raise GMTAdmissionError(
                    f"converted tensor must be stored, not compressed: {spec.key}"
                )
            if not (spec.nbytes < info.file_size <= spec.nbytes + 512):
                raise GMTAdmissionError(f"invalid NPY envelope size for {spec.key}")
            payload = archive.read(member_name)
            stream = io.BytesIO(payload)
            try:
                array = np.lib.format.read_array(
                    stream,
                    allow_pickle=False,
                    max_header_size=512,
                )
            except (ValueError, EOFError) as exc:
                raise GMTAdmissionError(f"invalid numeric NPY member: {spec.key}") from exc
            if stream.tell() != len(payload):
                raise GMTAdmissionError(f"trailing bytes in NPY member: {spec.key}")
            arrays[spec.key] = array
    _validate_numeric_arrays(arrays, TENSOR_SPECS)
    return arrays


def converted_checkpoint_sha256(path: Path) -> str:
    """Return the identity a caller must bind when loading converted weights."""

    return sha256_file(Path(path))


def verify_upstream_root(path: Path) -> dict[str, str]:
    """Verify reviewed controller, robot, license, and documentation bytes."""

    root = Path(path)
    for relative_path, expected_digest in GMT_SUPPORT_FILE_SHA256.items():
        require_sha256(root / relative_path, expected_digest)
    mesh_root = root / "assets/robots/g1/meshes"
    mesh_tree = hashlib.sha256()
    mesh_receipts: dict[str, str] = {}
    for name in GMT_G1_MESH_NAMES:
        digest = sha256_file(mesh_root / name)
        mesh_tree.update(f"{name}\0{digest}\n".encode())
        mesh_receipts[f"assets/robots/g1/meshes/{name}"] = digest
    observed_mesh_tree = mesh_tree.hexdigest()
    if observed_mesh_tree != GMT_G1_MESH_TREE_SHA256:
        raise GMTAdmissionError(
            "G1 mesh-tree mismatch: "
            f"expected {GMT_G1_MESH_TREE_SHA256}, observed {observed_mesh_tree}"
        )
    return {
        **GMT_SUPPORT_FILE_SHA256,
        **mesh_receipts,
        "assets/robots/g1/meshes@tree": GMT_G1_MESH_TREE_SHA256,
    }
