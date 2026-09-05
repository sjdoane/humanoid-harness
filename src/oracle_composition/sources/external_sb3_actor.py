"""Bounded data-only import of the pinned external SB3 TQC actor."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import math
import os
import re
import selectors
import signal
import stat
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile

import numpy as np

from oracle_composition.experiments.artifact_io import publish_bytes_without_overwrite
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_actor_npz import (
    ACTION_WIDTH,
    OBSERVATION_WIDTH,
    LoadedTQCActor,
    actor_schema_sha256,
    actor_state_sha256,
    load_actor_npz,
    validate_actor_arrays,
    write_actor_npz_exclusive,
)

from ._external_sb3_actor_worker import (
    ACTOR_STATE_SCHEMA,
    ADDRESS_SPACE_LIMIT_BYTES,
    CPU_TIME_LIMIT_SECONDS,
    EXPECTED_TORCH_DISTRIBUTION,
    EXPECTED_TORCH_RUNTIME_VERSION,
    FRAME_LENGTHS,
    MAX_FRAME_BYTES,
    MAX_HEADER_BYTES,
    OUTCOME_CRASH_OR_SIGNAL,
    OUTCOME_MALFORMED_FRAME,
    OUTCOME_SUCCESS,
    OUTCOME_TIMEOUT,
    POLICY_STATE_SCHEMA,
    PROTOCOL_VERSION,
    RESPONSE_MAGIC,
    WORKER_OUTCOMES,
    WORKER_SOURCE_LOGICAL_PATH,
)
from .farama_tqc_registration import (
    FARAMA_SCRIPT_COMMIT,
    HF_API_BYTE_COUNT,
    HF_API_SHA256,
    REGISTRATION_ID,
    REGISTRATION_RECEIPT_BYTE_COUNT,
    REGISTRATION_RECEIPT_LOGICAL_PATH,
    REGISTRATION_RECEIPT_SHA256,
    SOURCE_COMMIT,
    SOURCE_REPOSITORY,
    load_registration_receipt,
)

AUTHORITY = "external_pretrained_artifact"
IMPORT_ID = "farama_minari_humanoid_v5_tqc_external_actor_import/v2"
POLICY_SCHEMA_ID = "sb3_2.4.1_tqc_policy_state_dict_348_17/v1"
POLICY_SHA256 = "1e64e56288155087089214548b6a634f332a41955a0d22629efd5ff2240e495c"
POLICY_BYTE_COUNT = 3_321_462
SOURCE_ZIP_SHA256 = "c0675e01b4efd26d9c773de3e9b4defb40301b5fbdac9f0f31517156fac59fe3"
SOURCE_ZIP_BYTE_COUNT = 7_377_061
METADATA_SHA256 = "7466a42ba68e54135ead527b00ee2d23f7c2e1a25138ee9f7ba876a692a29e1c"
METADATA_BYTE_COUNT = 68_638
MAX_POLICY_BYTES = 8 * 1024 * 1024
MAX_TORCH_ZIP_ENTRIES = 64
MAX_TORCH_ZIP_UNCOMPRESSED_BYTES = 32 * 1024 * 1024
MAX_METADATA_BYTES = 128 * 1024
MAX_METADATA_DEPTH = 16
MAX_METADATA_NODES = 4_096
MAX_METADATA_STRING_LENGTH = 32 * 1024
MAX_LOCK_BYTES = 4 * 1024 * 1024
MAX_WORKER_RESPONSE_BYTES = MAX_FRAME_BYTES
MAX_WORKER_STDERR_BYTES = 16 * 1024
MAX_SOURCE_CODE_BYTES = 1024 * 1024
WORKER_TIMEOUT_SECONDS = 45
IMPORTER_SOURCE_LOGICAL_PATH = "src/oracle_composition/sources/external_sb3_actor.py"

SOURCE_VERSIONS = {
    "stable_baselines3": "2.4.1",
    "pytorch": "2.5.1+cu124",
    "gymnasium": "1.0.0",
    "numpy": "1.26.4",
    "cloudpickle": "3.1.0",
    "sb3_contrib": "unknown",
}
EXPECTED_LOCAL_VERSIONS = {
    "stable_baselines3": "2.9.0",
    "torch": "2.14.0",
    "gymnasium": "1.3.0",
    "numpy": "2.5.2",
    "cloudpickle": "3.1.2",
    "sb3_contrib": "2.9.0",
}
METADATA_FIELDS = (
    "_total_timesteps",
    "num_timesteps",
    "_n_updates",
    "n_envs",
    "seed",
    "batch_size",
    "buffer_size",
    "gamma",
    "learning_rate",
    "tau",
    "target_entropy",
    "top_quantiles_to_drop_per_net",
    "ent_coef",
    "policy_kwargs",
)
EXPECTED_METADATA: dict[str, object] = {
    "_total_timesteps": 20_000_000,
    "num_timesteps": 19_965_000,
    "_n_updates": 3_992_979,
    "n_envs": 5,
    "seed": 0,
    "batch_size": 256,
    "buffer_size": 1_000_000,
    "gamma": 0.99,
    "learning_rate": 0.0003,
    "tau": 0.005,
    "target_entropy": -17.0,
    "top_quantiles_to_drop_per_net": 2,
    "ent_coef": "auto",
    "policy_kwargs": {"use_sde": False},
}
EXPECTED_SERIALIZED_PATHS = (
    "/policy_class/:serialized:",
    "/_last_obs/:serialized:",
    "/_last_episode_starts/:serialized:",
    "/_last_original_obs/:serialized:",
    "/ep_info_buffer/:serialized:",
    "/ep_success_buffer/:serialized:",
    "/replay_buffer_class/:serialized:",
    "/train_freq/:serialized:",
    "/observation_space/:serialized:",
    "/action_space/:serialized:",
    "/lr_schedule/:serialized:",
)

_ACTOR_PAYLOAD_BYTE_COUNT = sum(math.prod(shape) * 4 for _, shape in ACTOR_STATE_SCHEMA)
_AUTHORITY_ISSUER = object()
_SAFE_SERIALIZED_CHARACTERS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
)
_NESTED_ARCHIVE_PREFIXES = (
    b"PK\x03\x04",
    b"PK\x05\x06",
    b"\x1f\x8b",
    b"BZh",
    b"\xfd7zXZ\x00",
    b"7z\xbc\xaf\x27\x1c",
)


class ExternalActorWorkerError(ExperimentContractError):
    """Categorical refusal from the bounded external-actor worker boundary."""

    def __init__(self, outcome: str, detail: str) -> None:
        if type(outcome) is not str or outcome not in WORKER_OUTCOMES or outcome == OUTCOME_SUCCESS:
            outcome = OUTCOME_MALFORMED_FRAME
            detail = "invalid worker outcome"
        self.outcome = outcome
        super().__init__(f"external actor worker {outcome}: {detail}")


def _exact_json_match(value: object, expected: object) -> bool:
    try:
        return _exact_json_match_inner(value, expected)
    except (RecursionError, TypeError, ValueError):
        return False


def _exact_json_match_inner(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        if set(value) != set(expected):
            return False
        return all(_exact_json_match_inner(value[key], expected[key]) for key in expected)
    if type(expected) is list:
        return len(value) == len(expected) and all(
            _exact_json_match_inner(observed, wanted)
            for observed, wanted in zip(value, expected, strict=True)
        )
    return value == expected


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
        raise ExperimentContractError("external actor value is not canonical JSON") from exc


def _canonical_sha256(value: object, *, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"{field_name} must be a lowercase SHA-256")
    return value


def _validate_source_identity(
    value: object,
    *,
    logical_path: str,
    field_name: str,
) -> dict[str, object]:
    if (
        type(value) is not dict
        or set(value) != {"logical_path", "sha256", "byte_count"}
        or type(value.get("logical_path")) is not str
        or value["logical_path"] != logical_path
        or type(value.get("byte_count")) is not int
        or not 0 < value["byte_count"] <= MAX_SOURCE_CODE_BYTES
    ):
        raise ExperimentContractError(f"{field_name} source identity differs")
    _canonical_sha256(value.get("sha256"), field_name=f"{field_name} source SHA-256")
    return value


def _read_regular_file_once(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
    expected_sha256: str | None = None,
    expected_byte_count: int | None = None,
) -> bytes:
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ExperimentContractError("file read bound is invalid")
    if expected_sha256 is not None:
        _canonical_sha256(expected_sha256, field_name=f"expected {label} SHA-256")
    if expected_byte_count is not None and (
        type(expected_byte_count) is not int
        or expected_byte_count <= 0
        or expected_byte_count > maximum_bytes
    ):
        raise ExperimentContractError(f"expected {label} byte count is invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(os.fspath(path), flags)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or not 0 < before.st_size <= maximum_bytes
        ):
            raise ExperimentContractError(f"{label} must be a bounded regular file")
        if expected_byte_count is not None and before.st_size != expected_byte_count:
            raise ExperimentContractError(f"{label} byte count does not match")
        chunks: list[bytes] = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, name) != getattr(after, name) for name in identity_fields):
            raise ExperimentContractError(f"{label} changed while held open")
        if len(payload) != before.st_size or len(payload) > maximum_bytes:
            raise ExperimentContractError(f"{label} length changed or exceeded its bound")
    except ExperimentContractError:
        raise
    except OSError as exc:
        raise ExperimentContractError(f"cannot open {label} without following links") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if expected_sha256 is not None:
        observed = hashlib.sha256(payload).hexdigest()
        if not hmac.compare_digest(observed, expected_sha256):
            raise ExperimentContractError(f"{label} SHA-256 does not match")
    return payload


def _source_file_identity(path: Path, *, logical_path: str, label: str) -> dict[str, object]:
    payload = _read_regular_file_once(
        Path(path),
        maximum_bytes=MAX_SOURCE_CODE_BYTES,
        label=label,
    )
    return {
        "logical_path": logical_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_count": len(payload),
    }


def read_pinned_policy_bytes(path: Path) -> bytes:
    """Open the exact policy path once and retain only its verified bytes."""

    return _read_regular_file_once(
        Path(path),
        maximum_bytes=MAX_POLICY_BYTES,
        label="external policy.pth",
        expected_sha256=POLICY_SHA256,
        expected_byte_count=POLICY_BYTE_COUNT,
    )


def _safe_torch_member_name(name: object, *, root: str | None) -> tuple[str, str]:
    if type(name) is not str or not name or "\\" in name or name.startswith("/"):
        raise ExperimentContractError("Torch ZIP member path is unsafe")
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ExperimentContractError("Torch ZIP member path is unsafe")
    observed_root = parts[0]
    if root is not None and observed_root != root:
        raise ExperimentContractError("Torch ZIP members do not share one root")
    relative = "/".join(parts[1:])
    allowed = (
        relative in {"data.pkl", "version", "byteorder"}
        or re.fullmatch(r"data/[0-9]+", relative) is not None
        or (relative.startswith(".data/") and len(parts) >= 3)
    )
    if not allowed:
        raise ExperimentContractError("Torch ZIP member is outside the allowlisted layout")
    return observed_root, relative


def preflight_torch_zip(held_bytes: bytes) -> list[dict[str, object]]:
    """Validate the bounded Torch-ZIP container before deserialization."""

    if type(held_bytes) is not bytes or not 0 < len(held_bytes) <= MAX_POLICY_BYTES:
        raise ExperimentContractError("Torch ZIP held bytes are invalid")
    try:
        with ZipFile(io.BytesIO(held_bytes), "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if not 0 < len(infos) <= MAX_TORCH_ZIP_ENTRIES:
                raise ExperimentContractError("Torch ZIP entry count exceeds its bound")
            if len(names) != len(set(names)):
                raise ExperimentContractError("Torch ZIP contains duplicate member names")
            if archive.comment:
                raise ExperimentContractError("Torch ZIP comment is not allowed")
            total = sum(info.file_size for info in infos)
            if total > MAX_TORCH_ZIP_UNCOMPRESSED_BYTES:
                raise ExperimentContractError("Torch ZIP expansion exceeds its bound")
            root: str | None = None
            relative_names: set[str] = set()
            inventory: list[dict[str, object]] = []
            for info in infos:
                observed_root, relative = _safe_torch_member_name(info.filename, root=root)
                if root is None:
                    root = observed_root
                relative_names.add(relative)
                mode = (info.external_attr >> 16) & 0xFFFF
                if (
                    info.is_dir()
                    or info.file_size <= 0
                    or info.file_size > MAX_TORCH_ZIP_UNCOMPRESSED_BYTES
                    or info.compress_size < 0
                    or info.compress_type not in {ZIP_STORED, ZIP_DEFLATED}
                    or bool(info.flag_bits & 0x1)
                    or (info.create_system == 3 and stat.S_ISLNK(mode))
                ):
                    raise ExperimentContractError("Torch ZIP member metadata is unsafe")
                with archive.open(info, "r") as stream:
                    prefix = stream.read(512)
                if any(prefix.startswith(signature) for signature in _NESTED_ARCHIVE_PREFIXES):
                    raise ExperimentContractError("Torch ZIP contains a nested archive")
                if len(prefix) >= 262 and prefix[257:262] == b"ustar":
                    raise ExperimentContractError("Torch ZIP contains a nested archive")
                inventory.append(
                    {
                        "name": info.filename,
                        "compressed_byte_count": info.compress_size,
                        "uncompressed_byte_count": info.file_size,
                    }
                )
            if not {"data.pkl", "version", "byteorder"}.issubset(relative_names):
                raise ExperimentContractError("Torch ZIP required members are missing")
            if not any(re.fullmatch(r"data/[0-9]+", name) for name in relative_names):
                raise ExperimentContractError("Torch ZIP tensor storages are missing")
            return inventory
    except ExperimentContractError:
        raise
    except (BadZipFile, EOFError, OSError, RuntimeError, ValueError) as exc:
        raise ExperimentContractError("Torch ZIP preflight failed") from exc


def _reject_duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExperimentContractError("SB3 metadata contains a duplicate key")
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> object:
    raise ExperimentContractError("SB3 metadata contains a non-finite constant")


def _json_pointer(parts: tuple[str, ...]) -> str:
    return "/" + "/".join(part.replace("~", "~0").replace("/", "~1") for part in parts)


def _validate_json_tree(
    value: object,
    *,
    depth: int = 0,
    path: tuple[str, ...] = (),
    state: dict[str, object],
) -> None:
    if depth > MAX_METADATA_DEPTH:
        raise ExperimentContractError("SB3 metadata nesting exceeds its bound")
    state["nodes"] = int(state["nodes"]) + 1
    if int(state["nodes"]) > MAX_METADATA_NODES:
        raise ExperimentContractError("SB3 metadata node count exceeds its bound")
    if type(value) is dict:
        serialized = value.get(":serialized:")
        if ":serialized:" in value:
            kind = value.get(":type:")
            if (
                type(serialized) is not str
                or type(kind) is not str
                or len(serialized) < 32
                or len(serialized) > MAX_METADATA_STRING_LENGTH
                or any(character not in _SAFE_SERIALIZED_CHARACTERS for character in serialized)
            ):
                raise ExperimentContractError("SB3 serialized metadata envelope is invalid")
            serialized_paths = state["serialized_paths"]
            if type(serialized_paths) is not list:
                raise ExperimentContractError("SB3 metadata traversal state is invalid")
            serialized_paths.append(_json_pointer((*path, ":serialized:")))
        for key, child in value.items():
            if type(key) is not str or len(key) > MAX_METADATA_STRING_LENGTH:
                raise ExperimentContractError("SB3 metadata key is invalid")
            _validate_json_tree(child, depth=depth + 1, path=(*path, key), state=state)
        return
    if type(value) is list:
        for index, child in enumerate(value):
            _validate_json_tree(
                child,
                depth=depth + 1,
                path=(*path, str(index)),
                state=state,
            )
        return
    if type(value) is str:
        if len(value) > MAX_METADATA_STRING_LENGTH:
            raise ExperimentContractError("SB3 metadata string exceeds its bound")
        return
    if type(value) is float and not math.isfinite(value):
        raise ExperimentContractError("SB3 metadata number is non-finite")
    if value is None or type(value) in {bool, int, float}:
        return
    raise ExperimentContractError("SB3 metadata contains an unsupported JSON value")


def parse_sb3_metadata_bytes(payload: bytes) -> dict[str, object]:
    """Return only pinned scalar fields and serialized-field locations."""

    if type(payload) is not bytes or not 0 < len(payload) <= MAX_METADATA_BYTES:
        raise ExperimentContractError("SB3 metadata byte count is invalid")
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_key,
            parse_constant=_reject_json_constant,
        )
    except ExperimentContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ExperimentContractError("SB3 metadata is invalid strict JSON") from exc
    if type(value) is not dict:
        raise ExperimentContractError("SB3 metadata root must be an object")
    state: dict[str, object] = {"nodes": 0, "serialized_paths": []}
    _validate_json_tree(value, state=state)
    if any(field not in value for field in METADATA_FIELDS):
        raise ExperimentContractError("SB3 metadata is missing an allowlisted scalar")
    scalars = {field: value[field] for field in METADATA_FIELDS}
    if scalars != EXPECTED_METADATA:
        raise ExperimentContractError("SB3 metadata scalar values differ from the pinned source")
    serialized_paths = tuple(state["serialized_paths"])
    if serialized_paths != EXPECTED_SERIALIZED_PATHS:
        raise ExperimentContractError("SB3 serialized metadata paths differ")
    return {
        "scalars": scalars,
        "serialized_field_paths": list(serialized_paths),
        "serialized_field_count": len(serialized_paths),
    }


def read_pinned_metadata(path: Path) -> dict[str, object]:
    """Read the source ``data`` member once and expose no serialized values."""

    payload = _read_regular_file_once(
        Path(path),
        maximum_bytes=MAX_METADATA_BYTES,
        label="external SB3 data metadata",
        expected_sha256=METADATA_SHA256,
        expected_byte_count=METADATA_BYTE_COUNT,
    )
    return parse_sb3_metadata_bytes(payload)


def _read_local_versions(lock_path: Path) -> tuple[dict[str, str], str, int]:
    payload = _read_regular_file_once(
        Path(lock_path),
        maximum_bytes=MAX_LOCK_BYTES,
        label="dependency lock",
    )
    try:
        lock = tomllib.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ExperimentContractError("dependency lock is invalid") from exc
    packages = lock.get("package") if type(lock) is dict else None
    if type(packages) is not list:
        raise ExperimentContractError("dependency lock package table is unavailable")
    wanted = set(EXPECTED_LOCAL_VERSIONS)
    versions: dict[str, str] = {}
    for package in packages:
        if type(package) is not dict or type(package.get("name")) is not str:
            continue
        name = package["name"].replace("-", "_")
        if name in wanted:
            version = package.get("version")
            if type(version) is not str or name in versions:
                raise ExperimentContractError("dependency lock version entry is invalid")
            versions[name] = version
    if versions != EXPECTED_LOCAL_VERSIONS:
        raise ExperimentContractError("dependency lock versions differ from the import contract")
    return versions, hashlib.sha256(payload).hexdigest(), len(payload)


def _malformed_worker_frame(detail: str) -> ExternalActorWorkerError:
    return ExternalActorWorkerError(OUTCOME_MALFORMED_FRAME, detail)


def _validate_worker_resource_limits(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {"cpu", "address_space"}:
        raise _malformed_worker_frame("resource-limit frame differs")
    cpu = value["cpu"]
    address_space = value["address_space"]
    if not _exact_json_match(
        cpu,
        {
            "resource": "RLIMIT_CPU",
            "soft_seconds": CPU_TIME_LIMIT_SECONDS,
            "hard_seconds": CPU_TIME_LIMIT_SECONDS,
        },
    ):
        raise _malformed_worker_frame("CPU resource-limit frame differs")
    if (
        type(address_space) is not dict
        or set(address_space)
        != {
            "resource",
            "requested_bytes",
            "finite_enforced",
            "observed_soft",
            "observed_hard",
            "darwin_finite_limit_unavailable",
        }
        or type(address_space.get("resource")) is not str
        or address_space["resource"] != "RLIMIT_AS"
        or type(address_space.get("requested_bytes")) is not int
        or address_space["requested_bytes"] != ADDRESS_SPACE_LIMIT_BYTES
        or type(address_space.get("finite_enforced")) is not bool
        or type(address_space.get("observed_soft")) is not int
        or type(address_space.get("observed_hard")) is not int
        or type(address_space.get("darwin_finite_limit_unavailable")) is not bool
    ):
        raise _malformed_worker_frame("address-space resource-limit frame differs")
    return value


def _parse_worker_response(payload: bytes) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    prefix_bytes = len(RESPONSE_MAGIC) + FRAME_LENGTHS.size
    if type(payload) is not bytes or len(payload) > MAX_WORKER_RESPONSE_BYTES:
        raise _malformed_worker_frame("response size is invalid")
    if len(payload) < prefix_bytes or not payload.startswith(RESPONSE_MAGIC):
        raise _malformed_worker_frame("response prefix is truncated or invalid")
    try:
        header_length, raw_length = FRAME_LENGTHS.unpack(
            payload[len(RESPONSE_MAGIC) : prefix_bytes]
        )
    except (TypeError, ValueError) as exc:
        raise _malformed_worker_frame("response lengths are invalid") from exc
    header_start = prefix_bytes
    header_end = header_start + header_length
    frame_end = header_end + raw_length
    if (
        not 0 < header_length <= MAX_HEADER_BYTES
        or raw_length > MAX_WORKER_RESPONSE_BYTES
        or frame_end != len(payload)
    ):
        raise _malformed_worker_frame("response lengths differ")
    try:
        header = json.loads(
            payload[header_start:header_end].decode("ascii", errors="strict"),
            object_pairs_hook=_reject_duplicate_key,
            parse_constant=_reject_json_constant,
        )
    except (
        ExperimentContractError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
    ) as exc:
        raise _malformed_worker_frame("response header is invalid JSON") from exc
    if (
        type(header) is not dict
        or type(header.get("protocol_version")) is not int
        or header["protocol_version"] != PROTOCOL_VERSION
        or type(header.get("outcome")) is not str
        or header["outcome"] not in WORKER_OUTCOMES
    ):
        raise _malformed_worker_frame("response outcome contract differs")
    outcome = header["outcome"]
    if outcome != OUTCOME_SUCCESS:
        if (
            set(header) != {"protocol_version", "outcome", "detail"}
            or type(header.get("detail")) is not str
            or not header["detail"]
            or len(header["detail"]) > 160
            or raw_length != 0
        ):
            raise _malformed_worker_frame("error frame contract differs")
        raise ExternalActorWorkerError(outcome, header["detail"])

    expected_header_fields = {
        "protocol_version",
        "outcome",
        "runtime_identity",
        "worker_source_identity",
        "torch_load",
        "resource_limits",
        "actor_payload_byte_count",
        "policy_key_inventory",
        "mapping",
    }
    expected_inventory = [
        {"name": name, "shape": list(shape)} for name, shape in POLICY_STATE_SCHEMA
    ]
    allowed_mappings = (
        {"raw_type": "collections.OrderedDict", "validated_result_type": "dict"},
        {"raw_type": "dict", "validated_result_type": "dict"},
    )
    if (
        set(header) != expected_header_fields
        or type(header.get("actor_payload_byte_count")) is not int
        or header["actor_payload_byte_count"] != _ACTOR_PAYLOAD_BYTE_COUNT
        or not _exact_json_match(header.get("policy_key_inventory"), expected_inventory)
        or not any(_exact_json_match(header.get("mapping"), item) for item in allowed_mappings)
        or not _exact_json_match(
            header.get("runtime_identity"),
            {
                "runtime_version": EXPECTED_TORCH_RUNTIME_VERSION,
                "distribution": dict(EXPECTED_TORCH_DISTRIBUTION),
            },
        )
    ):
        raise _malformed_worker_frame("success frame contract differs")
    try:
        _validate_source_identity(
            header.get("worker_source_identity"),
            logical_path=WORKER_SOURCE_LOGICAL_PATH,
            field_name="worker",
        )
    except ExperimentContractError as exc:
        raise _malformed_worker_frame("worker source identity differs") from exc
    _validate_worker_resource_limits(header.get("resource_limits"))
    torch_load = header["torch_load"]
    if (
        type(torch_load) is not dict
        or set(torch_load)
        != {
            "call_count",
            "input",
            "map_location",
            "weights_only",
            "safe_globals_count",
            "safe_globals_sha256",
            "safe_globals_module_allowlist_passed",
            "ordered_names_unchanged",
        }
        or type(torch_load.get("call_count")) is not int
        or torch_load["call_count"] != 1
        or type(torch_load.get("input")) is not str
        or torch_load["input"] != "held_bytes_io.BytesIO"
        or type(torch_load.get("map_location")) is not str
        or torch_load["map_location"] != "cpu"
        or torch_load.get("weights_only") is not True
        or type(torch_load.get("safe_globals_count")) is not int
        or not 0 <= torch_load["safe_globals_count"] <= 1_024
        or torch_load.get("safe_globals_module_allowlist_passed") is not True
        or torch_load.get("ordered_names_unchanged") is not True
    ):
        raise _malformed_worker_frame("loader frame contract differs")
    try:
        _canonical_sha256(
            torch_load.get("safe_globals_sha256"),
            field_name="worker safe-globals SHA-256",
        )
    except ExperimentContractError as exc:
        raise _malformed_worker_frame("safe-globals hash differs") from exc
    raw = payload[header_end:frame_end]
    if raw_length != _ACTOR_PAYLOAD_BYTE_COUNT:
        raise _malformed_worker_frame("actor payload length differs")
    arrays: dict[str, np.ndarray] = {}
    offset = 0
    for source_name, shape in ACTOR_STATE_SCHEMA:
        size = math.prod(shape) * 4
        value = np.frombuffer(raw[offset : offset + size], dtype="<f4").reshape(shape).copy()
        arrays[source_name.removeprefix("actor.")] = np.ascontiguousarray(value, dtype="<f4")
        offset += size
    arrays.update(
        {
            "action_low": np.full(ACTION_WIDTH, -0.4, dtype="<f4"),
            "action_high": np.full(ACTION_WIDTH, 0.4, dtype="<f4"),
            "format_version": np.asarray([1], dtype="<i8"),
        }
    )
    try:
        validated = validate_actor_arrays(arrays)
    except ExperimentContractError as exc:
        raise _malformed_worker_frame("actor payload values differ") from exc
    return validated, header


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)


def _read_bounded_worker_output(
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float,
) -> tuple[bytes, bytes, int]:
    if process.stdout is None or process.stderr is None:
        _terminate_process_group(process)
        raise ExternalActorWorkerError(OUTCOME_CRASH_OR_SIGNAL, "worker pipes unavailable")
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ, ("stdout", MAX_WORKER_RESPONSE_BYTES))
    selector.register(process.stderr, selectors.EVENT_READ, ("stderr", MAX_WORKER_STDERR_BYTES))
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + timeout_seconds
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_process_group(process)
                raise ExternalActorWorkerError(OUTCOME_TIMEOUT, "worker deadline exceeded")
            events = selector.select(min(remaining, 0.25))
            if not events and process.poll() is not None:
                events = [(key, selectors.EVENT_READ) for key in selector.get_map().values()]
            for key, _mask in events:
                name, limit = key.data
                try:
                    chunk = os.read(key.fileobj.fileno(), 64 * 1024)
                except OSError as exc:
                    _terminate_process_group(process)
                    raise ExternalActorWorkerError(
                        OUTCOME_CRASH_OR_SIGNAL,
                        "worker pipe read failed",
                    ) from exc
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                buffers[name].extend(chunk)
                if len(buffers[name]) > limit:
                    _terminate_process_group(process)
                    raise _malformed_worker_frame(f"{name} exceeded its capture bound")
        returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        raise ExternalActorWorkerError(OUTCOME_TIMEOUT, "worker deadline exceeded") from exc
    finally:
        selector.close()
        process.stdout.close()
        process.stderr.close()
    return bytes(buffers["stdout"]), bytes(buffers["stderr"]), returncode


def _run_worker_process(
    command: tuple[str, ...],
    *,
    held_bytes: bytes,
    environment: Mapping[str, str],
    timeout_seconds: float,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    if (
        type(command) is not tuple
        or not command
        or any(type(part) is not str or not part for part in command)
        or type(held_bytes) is not bytes
        or not 0 < len(held_bytes) <= MAX_POLICY_BYTES
        or type(timeout_seconds) not in {int, float}
        or not 0 < timeout_seconds <= WORKER_TIMEOUT_SECONDS
    ):
        raise ExperimentContractError("external actor worker launch contract is invalid")
    with tempfile.TemporaryFile() as input_stream:
        input_stream.write(held_bytes)
        input_stream.seek(0)
        try:
            process = subprocess.Popen(
                command,
                stdin=input_stream,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
                start_new_session=True,
                env=dict(environment),
            )
        except OSError as exc:
            raise ExternalActorWorkerError(
                OUTCOME_CRASH_OR_SIGNAL,
                "worker launch failed",
            ) from exc
        stdout, stderr, returncode = _read_bounded_worker_output(
            process,
            timeout_seconds=float(timeout_seconds),
        )
    if returncode < 0:
        raise ExternalActorWorkerError(
            OUTCOME_CRASH_OR_SIGNAL,
            f"worker terminated by signal {-returncode}",
        )
    if stderr:
        raise _malformed_worker_frame("worker wrote to stderr")
    if not stdout and returncode != 0:
        raise ExternalActorWorkerError(
            OUTCOME_CRASH_OR_SIGNAL,
            f"worker exited with status {returncode}",
        )
    arrays, header = _parse_worker_response(stdout)
    if returncode != 0:
        raise ExternalActorWorkerError(
            OUTCOME_CRASH_OR_SIGNAL,
            f"worker emitted success then exited with status {returncode}",
        )
    return arrays, header


def _run_loader_subprocess(held_bytes: bytes) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    worker_path = Path(__file__).with_name("_external_sb3_actor_worker.py").resolve(strict=True)
    expected_source = _source_file_identity(
        worker_path,
        logical_path=WORKER_SOURCE_LOGICAL_PATH,
        label="external actor worker source",
    )
    environment = {
        "LANG": "C",
        "LC_ALL": "C",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    arrays, header = _run_worker_process(
        (sys.executable, "-I", os.fspath(worker_path)),
        held_bytes=held_bytes,
        environment=environment,
        timeout_seconds=WORKER_TIMEOUT_SECONDS,
    )
    if not _exact_json_match(header["worker_source_identity"], expected_source):
        raise ExternalActorWorkerError(OUTCOME_MALFORMED_FRAME, "worker source identity differs")
    return arrays, header


def _inventory_without_values(header: Mapping[str, object]) -> dict[str, object]:
    inventory = header["policy_key_inventory"]
    if type(inventory) is not list:
        raise ExperimentContractError("policy key inventory is unavailable")
    actor = [dict(item) for item in inventory if str(item["name"]).startswith("actor.")]
    critics = [dict(item) for item in inventory if not str(item["name"]).startswith("actor.")]
    return {
        "schema_id": POLICY_SCHEMA_ID,
        "tensor_contract": (
            "plain_cpu_float32_contiguous_offset0_exact_storage_finite_no_storage_overlap"
        ),
        "key_count": len(inventory),
        "key_inventory": [dict(item) for item in inventory],
        "actor_key_count": len(actor),
        "actor_key_inventory": actor,
        "transient_critic_key_count": len(critics),
        "transient_critic_key_inventory": critics,
        "transient_critic_handling": (
            "deserialized transiently because policy.pth is one mapping; names and shapes "
            "recorded; values discarded without retention, export, or hashing"
        ),
    }


def _build_import_receipt(
    *,
    registration: Mapping[str, object],
    metadata: Mapping[str, object],
    member_inventory: list[dict[str, object]],
    worker_header: Mapping[str, object],
    local_versions: Mapping[str, str],
    lock_sha256: str,
    lock_byte_count: int,
    actor_artifact_label: str,
    loaded_actor: LoadedTQCActor,
) -> dict[str, object]:
    importer_source_identity = _source_file_identity(
        Path(__file__).resolve(strict=True),
        logical_path=IMPORTER_SOURCE_LOGICAL_PATH,
        label="external actor importer source",
    )
    return {
        "schema_version": 2,
        "import_id": IMPORT_ID,
        "authority": AUTHORITY,
        "evidence_class": "external_base_import",
        "evidence_level": "interface_check",
        "source": {
            "registration_id": REGISTRATION_ID,
            "registration_receipt": {
                "logical_path": REGISTRATION_RECEIPT_LOGICAL_PATH,
                "sha256": REGISTRATION_RECEIPT_SHA256,
                "byte_count": REGISTRATION_RECEIPT_BYTE_COUNT,
                "captured_api": dict(registration["captured_api"]),
                "rights": dict(registration["rights"]),
            },
            "repository": SOURCE_REPOSITORY,
            "repository_commit": SOURCE_COMMIT,
            "farama_script_commit": FARAMA_SCRIPT_COMMIT,
            "policy_pth": {"sha256": POLICY_SHA256, "byte_count": POLICY_BYTE_COUNT},
            "data": {"sha256": METADATA_SHA256, "byte_count": METADATA_BYTE_COUNT},
            "source_zip": {
                "sha256": SOURCE_ZIP_SHA256,
                "byte_count": SOURCE_ZIP_BYTE_COUNT,
                "opened_by_importer": False,
            },
        },
        "source_versions": dict(SOURCE_VERSIONS),
        "local_versions": {
            "source": "uv.lock",
            "uv_lock_sha256": lock_sha256,
            "uv_lock_byte_count": lock_byte_count,
            "packages": dict(local_versions),
            "worker_torch_runtime": worker_header["runtime_identity"],
        },
        "implementation": {
            "importer_source": importer_source_identity,
            "worker_source": worker_header["worker_source_identity"],
        },
        "source_metadata": {
            **dict(metadata),
            "timestep_counters_reconciled": False,
            "timestep_counter_note": (
                "num_timesteps 19965000 and _total_timesteps 20000000 are retained "
                "without reconciliation"
            ),
            "config_json": "ignored and never opened by the importer",
        },
        "abi": {
            "observation_width": OBSERVATION_WIDTH,
            "action_width": ACTION_WIDTH,
            "action_low": [-0.4] * ACTION_WIDTH,
            "action_high": [0.4] * ACTION_WIDTH,
        },
        "environment_termination": {
            "source_training_terminate_when_unhealthy": True,
            "source_training_basis": "Farama default Humanoid-v5",
            "project_execution_terminate_when_unhealthy": False,
            "same_mdp_claimed": False,
        },
        "model_card": {
            "description": "SB3 TQC, `20 x 10^6` steps, runs without falling",
            "metric": "mean_reward 10370.61 +/- 1542.02",
            "deterministic_episode_count": 1_000,
            "verified": False,
        },
        "torch_zip_preflight": {
            "entry_limit": MAX_TORCH_ZIP_ENTRIES,
            "uncompressed_byte_limit": MAX_TORCH_ZIP_UNCOMPRESSED_BYTES,
            "member_count": len(member_inventory),
            "total_uncompressed_byte_count": sum(
                int(item["uncompressed_byte_count"]) for item in member_inventory
            ),
            "member_inventory": member_inventory,
            "passed": True,
        },
        "deserialization": {
            **dict(worker_header["torch_load"]),
            "mapping": worker_header["mapping"],
            "fresh_spawned_subprocess": True,
            "alternative_loader_fallback": False,
            "policy_state": _inventory_without_values(worker_header),
        },
        "isolation": {
            "resource_limits": worker_header["resource_limits"],
            "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
            "network_os_enforced": False,
            "network_note": (
                "macOS network isolation is documented but not OS-enforced; the worker "
                "receives only held checkpoint bytes and a minimal environment"
            ),
        },
        "strict_actor_npz": {
            "logical_path": actor_artifact_label,
            "sha256": loaded_actor.content_sha256,
            "byte_count": loaded_actor.byte_count,
            "actor_state_sha256": loaded_actor.state_sha256,
            "schema_sha256": loaded_actor.schema_sha256,
            "contains_external_payload_bytes": True,
            "eligible_for_git": False,
        },
        "claim": {
            "establishes": (
                "integrity-verified external actor bytes and strict actor NPZ identity"
            ),
            "does_not_establish": [
                "E1",
                "tracker admission",
                "reference use",
                "Humanoid behavior",
            ],
        },
    }


def validate_external_actor_import_receipt(value: object) -> dict[str, object]:
    """Validate every required external-import provenance and authority field."""

    required = {
        "schema_version",
        "import_id",
        "authority",
        "evidence_class",
        "evidence_level",
        "source",
        "source_versions",
        "local_versions",
        "implementation",
        "source_metadata",
        "abi",
        "environment_termination",
        "model_card",
        "torch_zip_preflight",
        "deserialization",
        "isolation",
        "strict_actor_npz",
        "claim",
    }
    if type(value) is not dict or set(value) != required:
        raise ExperimentContractError("external actor import receipt fields differ")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 2
        or type(value["import_id"]) is not str
        or value["import_id"] != IMPORT_ID
        or type(value["authority"]) is not str
        or value["authority"] != AUTHORITY
        or type(value["evidence_class"]) is not str
        or value["evidence_class"] != "external_base_import"
        or type(value["evidence_level"]) is not str
        or value["evidence_level"] != "interface_check"
        or not _exact_json_match(value["source_versions"], SOURCE_VERSIONS)
    ):
        raise ExperimentContractError("external actor import identity differs")
    source = value["source"]
    expected_source = {
        "registration_id": REGISTRATION_ID,
        "registration_receipt": {
            "logical_path": REGISTRATION_RECEIPT_LOGICAL_PATH,
            "sha256": REGISTRATION_RECEIPT_SHA256,
            "byte_count": REGISTRATION_RECEIPT_BYTE_COUNT,
            "captured_api": {
                "path": (
                    "artifacts/external/farama-minari-humanoid-v5-tqc-expert/hf_api_model_info.json"
                ),
                "sha256": HF_API_SHA256,
                "byte_count": HF_API_BYTE_COUNT,
            },
            "rights": {
                "hugging_face_license": "unspecified",
                "permitted_project_use": "local development only",
                "technical_import_approval": "Samuel approved 2026-09-04",
                "technical_approval_is_redistribution_grant": False,
                "payload_bytes_enter_git": False,
            },
        },
        "repository": SOURCE_REPOSITORY,
        "repository_commit": SOURCE_COMMIT,
        "farama_script_commit": FARAMA_SCRIPT_COMMIT,
        "policy_pth": {"sha256": POLICY_SHA256, "byte_count": POLICY_BYTE_COUNT},
        "data": {"sha256": METADATA_SHA256, "byte_count": METADATA_BYTE_COUNT},
        "source_zip": {
            "sha256": SOURCE_ZIP_SHA256,
            "byte_count": SOURCE_ZIP_BYTE_COUNT,
            "opened_by_importer": False,
        },
    }
    if not _exact_json_match(source, expected_source):
        raise ExperimentContractError("external actor source binding differs")
    local_versions = value["local_versions"]
    if (
        type(local_versions) is not dict
        or set(local_versions)
        != {
            "source",
            "uv_lock_sha256",
            "uv_lock_byte_count",
            "packages",
            "worker_torch_runtime",
        }
        or local_versions["source"] != "uv.lock"
        or type(local_versions["source"]) is not str
        or not _exact_json_match(local_versions["packages"], EXPECTED_LOCAL_VERSIONS)
        or not _exact_json_match(
            local_versions["worker_torch_runtime"],
            {
                "runtime_version": EXPECTED_LOCAL_VERSIONS["torch"],
                "distribution": {
                    "name": "torch",
                    "version": EXPECTED_LOCAL_VERSIONS["torch"],
                },
            },
        )
        or type(local_versions["uv_lock_byte_count"]) is not int
        or local_versions["uv_lock_byte_count"] <= 0
    ):
        raise ExperimentContractError("external actor local versions differ")
    _canonical_sha256(local_versions["uv_lock_sha256"], field_name="uv.lock SHA-256")
    implementation = value["implementation"]
    if type(implementation) is not dict or set(implementation) != {
        "importer_source",
        "worker_source",
    }:
        raise ExperimentContractError("external actor implementation identity differs")
    expected_importer_source = _source_file_identity(
        Path(__file__).resolve(strict=True),
        logical_path=IMPORTER_SOURCE_LOGICAL_PATH,
        label="external actor importer source",
    )
    expected_worker_source = _source_file_identity(
        Path(__file__).with_name("_external_sb3_actor_worker.py").resolve(strict=True),
        logical_path=WORKER_SOURCE_LOGICAL_PATH,
        label="external actor worker source",
    )
    for field_name, logical_path in (
        ("importer_source", IMPORTER_SOURCE_LOGICAL_PATH),
        ("worker_source", WORKER_SOURCE_LOGICAL_PATH),
    ):
        _validate_source_identity(
            implementation[field_name],
            logical_path=logical_path,
            field_name=field_name,
        )
    if not _exact_json_match(implementation["importer_source"], expected_importer_source) or not (
        _exact_json_match(implementation["worker_source"], expected_worker_source)
    ):
        raise ExperimentContractError("external actor implementation source differs")
    metadata = value["source_metadata"]
    expected_metadata = {
        "scalars": EXPECTED_METADATA,
        "serialized_field_paths": list(EXPECTED_SERIALIZED_PATHS),
        "serialized_field_count": len(EXPECTED_SERIALIZED_PATHS),
        "timestep_counters_reconciled": False,
        "timestep_counter_note": (
            "num_timesteps 19965000 and _total_timesteps 20000000 are retained "
            "without reconciliation"
        ),
        "config_json": "ignored and never opened by the importer",
    }
    if not _exact_json_match(metadata, expected_metadata):
        raise ExperimentContractError("external actor source metadata differs")
    abi = value["abi"]
    if not _exact_json_match(
        abi,
        {
            "observation_width": OBSERVATION_WIDTH,
            "action_width": ACTION_WIDTH,
            "action_low": [-0.4] * ACTION_WIDTH,
            "action_high": [0.4] * ACTION_WIDTH,
        },
    ):
        raise ExperimentContractError("external actor ABI differs")
    if not _exact_json_match(
        value["environment_termination"],
        {
            "source_training_terminate_when_unhealthy": True,
            "source_training_basis": "Farama default Humanoid-v5",
            "project_execution_terminate_when_unhealthy": False,
            "same_mdp_claimed": False,
        },
    ):
        raise ExperimentContractError("external actor termination provenance differs")
    if not _exact_json_match(
        value["model_card"],
        {
            "description": "SB3 TQC, `20 x 10^6` steps, runs without falling",
            "metric": "mean_reward 10370.61 +/- 1542.02",
            "deterministic_episode_count": 1_000,
            "verified": False,
        },
    ):
        raise ExperimentContractError("external actor model-card status differs")
    preflight = value["torch_zip_preflight"]
    if (
        type(preflight) is not dict
        or set(preflight)
        != {
            "entry_limit",
            "uncompressed_byte_limit",
            "member_count",
            "total_uncompressed_byte_count",
            "member_inventory",
            "passed",
        }
        or type(preflight["entry_limit"]) is not int
        or preflight["entry_limit"] != MAX_TORCH_ZIP_ENTRIES
        or type(preflight["uncompressed_byte_limit"]) is not int
        or preflight["uncompressed_byte_limit"] != MAX_TORCH_ZIP_UNCOMPRESSED_BYTES
        or preflight["passed"] is not True
        or type(preflight["member_inventory"]) is not list
        or type(preflight["member_count"]) is not int
        or preflight["member_count"] != len(preflight["member_inventory"])
        or preflight["member_count"] <= 0
        or preflight["member_count"] > MAX_TORCH_ZIP_ENTRIES
        or type(preflight["total_uncompressed_byte_count"]) is not int
        or not 0 < preflight["total_uncompressed_byte_count"] <= MAX_TORCH_ZIP_UNCOMPRESSED_BYTES
    ):
        raise ExperimentContractError("external actor Torch-ZIP preflight differs")
    inventory_names: set[str] = set()
    inventory_uncompressed_bytes = 0
    for item in preflight["member_inventory"]:
        if (
            type(item) is not dict
            or set(item) != {"name", "compressed_byte_count", "uncompressed_byte_count"}
            or type(item["name"]) is not str
            or not item["name"]
            or item["name"] in inventory_names
            or type(item["compressed_byte_count"]) is not int
            or item["compressed_byte_count"] < 0
            or type(item["uncompressed_byte_count"]) is not int
            or item["uncompressed_byte_count"] <= 0
        ):
            raise ExperimentContractError("external actor Torch-ZIP inventory differs")
        inventory_names.add(item["name"])
        inventory_uncompressed_bytes += item["uncompressed_byte_count"]
    if inventory_uncompressed_bytes != preflight["total_uncompressed_byte_count"]:
        raise ExperimentContractError("external actor Torch-ZIP inventory total differs")
    deserialization = value["deserialization"]
    expected_loader = {
        "call_count": 1,
        "input": "held_bytes_io.BytesIO",
        "map_location": "cpu",
        "weights_only": True,
        "safe_globals_module_allowlist_passed": True,
        "ordered_names_unchanged": True,
    }
    if (
        type(deserialization) is not dict
        or set(deserialization)
        != {
            *expected_loader,
            "safe_globals_count",
            "safe_globals_sha256",
            "mapping",
            "fresh_spawned_subprocess",
            "alternative_loader_fallback",
            "policy_state",
        }
        or not _exact_json_match(
            {key: deserialization.get(key) for key in expected_loader},
            expected_loader,
        )
        or deserialization.get("fresh_spawned_subprocess") is not True
        or deserialization.get("alternative_loader_fallback") is not False
        or not any(
            _exact_json_match(deserialization.get("mapping"), allowed)
            for allowed in (
                {"raw_type": "collections.OrderedDict", "validated_result_type": "dict"},
                {"raw_type": "dict", "validated_result_type": "dict"},
            )
        )
    ):
        raise ExperimentContractError("external actor deserialization contract differs")
    if (
        type(deserialization.get("safe_globals_count")) is not int
        or not 0 <= deserialization["safe_globals_count"] <= 1_024
    ):
        raise ExperimentContractError("external actor safe-globals count differs")
    _canonical_sha256(
        deserialization.get("safe_globals_sha256"),
        field_name="external actor safe-globals SHA-256",
    )
    policy_state = deserialization.get("policy_state")
    expected_inventory = [
        {"name": name, "shape": list(shape)} for name, shape in POLICY_STATE_SCHEMA
    ]
    expected_policy_state = {
        "schema_id",
        "tensor_contract",
        "key_count",
        "key_inventory",
        "actor_key_count",
        "actor_key_inventory",
        "transient_critic_key_count",
        "transient_critic_key_inventory",
        "transient_critic_handling",
    }
    if type(policy_state) is not dict or set(policy_state) != expected_policy_state:
        raise ExperimentContractError("external actor policy key schema differs")
    expected_policy_state_value = {
        "schema_id": POLICY_SCHEMA_ID,
        "tensor_contract": (
            "plain_cpu_float32_contiguous_offset0_exact_storage_finite_no_storage_overlap"
        ),
        "key_count": len(POLICY_STATE_SCHEMA),
        "key_inventory": expected_inventory,
        "actor_key_count": len(ACTOR_STATE_SCHEMA),
        "actor_key_inventory": expected_inventory[:8],
        "transient_critic_key_count": len(POLICY_STATE_SCHEMA) - len(ACTOR_STATE_SCHEMA),
        "transient_critic_key_inventory": expected_inventory[8:],
        "transient_critic_handling": (
            "deserialized transiently because policy.pth is one mapping; names and shapes "
            "recorded; values discarded without retention, export, or hashing"
        ),
    }
    if not _exact_json_match(policy_state, expected_policy_state_value):
        raise ExperimentContractError("external actor policy key schema differs")
    isolation = value["isolation"]
    if (
        type(isolation) is not dict
        or set(isolation)
        != {
            "resource_limits",
            "worker_timeout_seconds",
            "network_os_enforced",
            "network_note",
        }
        or type(isolation.get("worker_timeout_seconds")) is not int
        or isolation.get("worker_timeout_seconds") != WORKER_TIMEOUT_SECONDS
        or isolation.get("network_os_enforced") is not False
        or type(isolation.get("network_note")) is not str
        or "not OS-enforced" not in isolation.get("network_note", "")
        or type(isolation.get("resource_limits")) is not dict
    ):
        raise ExperimentContractError("external actor isolation record differs")
    limits = isolation["resource_limits"]
    if (
        set(limits) != {"cpu", "address_space"}
        or type(limits.get("cpu")) is not dict
        or not _exact_json_match(
            limits["cpu"],
            {
                "resource": "RLIMIT_CPU",
                "soft_seconds": CPU_TIME_LIMIT_SECONDS,
                "hard_seconds": CPU_TIME_LIMIT_SECONDS,
            },
        )
        or type(limits.get("address_space")) is not dict
        or set(limits["address_space"])
        != {
            "resource",
            "requested_bytes",
            "finite_enforced",
            "observed_soft",
            "observed_hard",
            "darwin_finite_limit_unavailable",
        }
        or type(limits["address_space"].get("resource")) is not str
        or limits["address_space"].get("resource") != "RLIMIT_AS"
        or type(limits["address_space"].get("requested_bytes")) is not int
        or limits["address_space"].get("requested_bytes") != ADDRESS_SPACE_LIMIT_BYTES
        or type(limits["address_space"].get("finite_enforced")) is not bool
        or type(limits["address_space"].get("observed_soft")) is not int
        or type(limits["address_space"].get("observed_hard")) is not int
        or type(limits["address_space"].get("darwin_finite_limit_unavailable")) is not bool
    ):
        raise ExperimentContractError("external actor resource limits differ")
    strict_actor = value["strict_actor_npz"]
    if (
        type(strict_actor) is not dict
        or set(strict_actor)
        != {
            "logical_path",
            "sha256",
            "byte_count",
            "actor_state_sha256",
            "schema_sha256",
            "contains_external_payload_bytes",
            "eligible_for_git",
        }
        or type(strict_actor["logical_path"]) is not str
        or not strict_actor["logical_path"]
        or type(strict_actor["byte_count"]) is not int
        or strict_actor["byte_count"] <= 0
        or strict_actor["contains_external_payload_bytes"] is not True
        or strict_actor["eligible_for_git"] is not False
        or strict_actor["schema_sha256"] != actor_schema_sha256()
    ):
        raise ExperimentContractError("external strict actor record differs")
    for field_name in ("sha256", "actor_state_sha256", "schema_sha256"):
        _canonical_sha256(strict_actor[field_name], field_name=f"strict actor {field_name}")
    if not _exact_json_match(
        value["claim"],
        {
            "establishes": (
                "integrity-verified external actor bytes and strict actor NPZ identity"
            ),
            "does_not_establish": [
                "E1",
                "tracker admission",
                "reference use",
                "Humanoid behavior",
            ],
        },
    ):
        raise ExperimentContractError("external actor claim ceiling differs")
    return value


@dataclass(frozen=True, slots=True)
class ExternalPretrainedActorAuthority:
    """Process-local authority for exact external source and strict NPZ bytes."""

    receipt_sha256: str
    receipt_byte_count: int
    receipt_bytes: bytes = field(repr=False, compare=False)
    source_arrays: Mapping[str, np.ndarray] = field(repr=False, compare=False)
    loaded_actor: LoadedTQCActor = field(repr=False, compare=False)
    _creator_pid: int = field(repr=False, compare=False)
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _AUTHORITY_ISSUER:
            raise ExperimentContractError(
                "external pretrained actor authority may only be issued by the importer"
            )
        self._validate_sealed()

    def to_receipt_dict(self) -> dict[str, object]:
        try:
            value = json.loads(
                self.receipt_bytes.decode("utf-8", errors="strict"),
                object_pairs_hook=_reject_duplicate_key,
                parse_constant=_reject_json_constant,
            )
        except ExperimentContractError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ExperimentContractError("sealed external import receipt is invalid") from exc
        return validate_external_actor_import_receipt(value)

    def _validate_sealed(self) -> None:
        _canonical_sha256(self.receipt_sha256, field_name="external import receipt SHA-256")
        if (
            self._creator_pid != os.getpid()
            or type(self.receipt_byte_count) is not int
            or self.receipt_byte_count != len(self.receipt_bytes)
            or hashlib.sha256(self.receipt_bytes).hexdigest() != self.receipt_sha256
            or type(self.loaded_actor) is not LoadedTQCActor
        ):
            raise ExperimentContractError("external pretrained actor authority seal differs")
        receipt = self.to_receipt_dict()
        strict_actor = receipt["strict_actor_npz"]
        if (
            actor_state_sha256(self.source_arrays) != self.loaded_actor.state_sha256
            or self.loaded_actor.content_sha256 != strict_actor["sha256"]
            or self.loaded_actor.byte_count != strict_actor["byte_count"]
            or self.loaded_actor.state_sha256 != strict_actor["actor_state_sha256"]
            or self.loaded_actor.schema_sha256 != strict_actor["schema_sha256"]
        ):
            raise ExperimentContractError("external actor arrays differ from sealed receipt")


def revalidate_external_pretrained_actor_authority(
    authority: ExternalPretrainedActorAuthority,
) -> ExternalPretrainedActorAuthority:
    """Revalidate only the exact process-local external import authority."""

    if type(authority) is not ExternalPretrainedActorAuthority:
        raise ExperimentContractError("external actor requires exact import authority")
    authority._validate_sealed()
    return authority


def import_external_sb3_actor(
    *,
    registration_receipt_path: Path,
    policy_path: Path,
    metadata_path: Path,
    dependency_lock_path: Path,
    actor_output_path: Path,
    receipt_output_path: Path,
    actor_artifact_label: str,
) -> ExternalPretrainedActorAuthority:
    """Import the pinned actor once; never open the source archive or retry loading."""

    if type(actor_artifact_label) is not str or not actor_artifact_label:
        raise ExperimentContractError("external actor artifact label is invalid")
    registration = load_registration_receipt(Path(registration_receipt_path))
    held_policy = read_pinned_policy_bytes(Path(policy_path))
    member_inventory = preflight_torch_zip(held_policy)
    metadata = read_pinned_metadata(Path(metadata_path))
    local_versions, lock_sha256, lock_byte_count = _read_local_versions(Path(dependency_lock_path))
    source_arrays, worker_header = _run_loader_subprocess(held_policy)
    if not _exact_json_match(
        worker_header["runtime_identity"],
        {
            "runtime_version": local_versions["torch"],
            "distribution": {"name": "torch", "version": local_versions["torch"]},
        },
    ):
        raise ExperimentContractError("loader Torch runtime differs from uv.lock")
    actor_output = Path(actor_output_path)
    actor_sha256 = write_actor_npz_exclusive(actor_output, source_arrays)
    loaded_actor = load_actor_npz(actor_output, expected_sha256=actor_sha256)
    receipt = _build_import_receipt(
        registration=registration,
        metadata=metadata,
        member_inventory=member_inventory,
        worker_header=worker_header,
        local_versions=local_versions,
        lock_sha256=lock_sha256,
        lock_byte_count=lock_byte_count,
        actor_artifact_label=actor_artifact_label,
        loaded_actor=loaded_actor,
    )
    validate_external_actor_import_receipt(receipt)
    receipt_bytes = _canonical_json(receipt) + b"\n"
    published = publish_bytes_without_overwrite(Path(receipt_output_path), receipt_bytes)
    immutable_source = {}
    for name, value in validate_actor_arrays(source_arrays).items():
        immutable = np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)
        immutable_source[name] = immutable
    return ExternalPretrainedActorAuthority(
        receipt_sha256=published.sha256,
        receipt_byte_count=published.byte_count,
        receipt_bytes=receipt_bytes,
        source_arrays=MappingProxyType(immutable_source),
        loaded_actor=loaded_actor,
        _creator_pid=os.getpid(),
        _issuer=_AUTHORITY_ISSUER,
    )


__all__ = [
    "ACTOR_STATE_SCHEMA",
    "AUTHORITY",
    "EXPECTED_METADATA",
    "EXPECTED_SERIALIZED_PATHS",
    "IMPORT_ID",
    "METADATA_BYTE_COUNT",
    "METADATA_SHA256",
    "POLICY_BYTE_COUNT",
    "POLICY_SHA256",
    "POLICY_STATE_SCHEMA",
    "ExternalActorWorkerError",
    "ExternalPretrainedActorAuthority",
    "import_external_sb3_actor",
    "parse_sb3_metadata_bytes",
    "preflight_torch_zip",
    "read_pinned_metadata",
    "read_pinned_policy_bytes",
    "revalidate_external_pretrained_actor_authority",
    "validate_external_actor_import_receipt",
]
