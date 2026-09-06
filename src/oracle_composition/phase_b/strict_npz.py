"""Bounded data-only NPY members inside canonical NPZ archives."""

from __future__ import annotations

import io
from collections.abc import Mapping
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

import numpy as np

from oracle_composition.experiments.fixed_reference import ExperimentContractError

MAX_NPY_HEADER_BYTES = 1_024


def _parse_npy_member(
    payload: bytes,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype[object],
    archive_label: str,
) -> np.ndarray:
    stream = io.BytesIO(payload)
    try:
        version = np.lib.format.read_magic(stream)
        if version != (1, 0):
            raise ExperimentContractError(f"{archive_label} member {name} must use NPY 1.0")
        observed_shape, fortran_order, observed_dtype = np.lib.format.read_array_header_1_0(stream)
    except ExperimentContractError:
        raise
    except (EOFError, MemoryError, OSError, OverflowError, TypeError, ValueError) as exc:
        raise ExperimentContractError(
            f"{archive_label} member {name} has an invalid NPY header"
        ) from exc
    if observed_dtype.hasobject:
        raise ExperimentContractError(f"{archive_label} member {name} has an object dtype")
    if tuple(observed_shape) != shape or observed_dtype != dtype or observed_dtype.str != dtype.str:
        raise ExperimentContractError(
            f"{archive_label} member {name} differs from its array schema"
        )
    if fortran_order:
        raise ExperimentContractError(f"{archive_label} member {name} must use C order")
    data_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(payload) - stream.tell() != data_bytes:
        raise ExperimentContractError(f"{archive_label} member {name} payload byte count differs")
    try:
        value = np.frombuffer(
            payload,
            dtype=dtype,
            count=data_bytes // dtype.itemsize,
            offset=stream.tell(),
        ).reshape(shape)
        if dtype.kind == "f" and not np.isfinite(value).all():
            raise ExperimentContractError(
                f"{archive_label} member {name} contains non-finite values"
            )
        return value.copy(order="C")
    except ExperimentContractError:
        raise
    except (MemoryError, OverflowError, TypeError, ValueError) as exc:
        raise ExperimentContractError(
            f"{archive_label} member {name} cannot be materialized"
        ) from exc


def decode_strict_npz(
    payload: bytes,
    *,
    schema: Mapping[str, tuple[tuple[int, ...], np.dtype[object]]],
    archive_label: str,
    maximum_member_bytes: int,
    maximum_total_bytes: int,
) -> dict[str, np.ndarray]:
    """Validate expansion and every NPY header before allocating an array."""

    if type(payload) is not bytes or not payload.startswith(b"PK\x03\x04"):
        raise ExperimentContractError(f"{archive_label} is not an NPZ archive")
    if len(payload) < 22 or payload[-22:-18] != b"PK\x05\x06" or payload[-2:] != b"\x00\x00":
        raise ExperimentContractError(f"{archive_label} must end at an uncommented ZIP record")
    expected_names = tuple(f"{name}.npy" for name in schema)
    try:
        with ZipFile(io.BytesIO(payload), "r") as archive:
            infos = archive.infolist()
            names = tuple(info.filename for info in infos)
            if names != expected_names or len(names) != len(set(names)):
                raise ExperimentContractError(f"{archive_label} member order or member set differs")
            if archive.comment:
                raise ExperimentContractError(f"{archive_label} ZIP comment is forbidden")
            if sum(info.file_size for info in infos) > maximum_total_bytes:
                raise ExperimentContractError(f"{archive_label} total expansion exceeds its bound")
            arrays: dict[str, np.ndarray] = {}
            for info, (name, (shape, dtype)) in zip(infos, schema.items(), strict=True):
                expected_data_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                member_mode = (info.external_attr >> 16) & 0xFFFF
                if (
                    info.is_dir()
                    or info.flag_bits != 0
                    or info.compress_type != ZIP_DEFLATED
                    or not expected_data_bytes < info.file_size <= maximum_member_bytes
                    or info.file_size > expected_data_bytes + MAX_NPY_HEADER_BYTES
                    or info.create_system != 3
                    or member_mode != 0o100600
                    or info.date_time != (1980, 1, 1, 0, 0, 0)
                    or bool(info.extra)
                    or bool(info.comment)
                ):
                    raise ExperimentContractError(
                        f"{archive_label} member {name} has invalid bounded ZIP metadata"
                    )
                with archive.open(info, "r") as member:
                    member_payload = member.read(info.file_size + 1)
                if len(member_payload) != info.file_size:
                    raise ExperimentContractError(
                        f"{archive_label} member {name} bounded read differs"
                    )
                arrays[name] = _parse_npy_member(
                    member_payload,
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    archive_label=archive_label,
                )
            return arrays
    except ExperimentContractError:
        raise
    except (
        BadZipFile,
        EOFError,
        MemoryError,
        OSError,
        OverflowError,
        RuntimeError,
        ValueError,
    ) as exc:
        raise ExperimentContractError(f"{archive_label} is invalid") from exc


__all__ = ["decode_strict_npz"]
