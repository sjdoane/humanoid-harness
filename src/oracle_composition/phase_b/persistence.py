"""Strict final-only checkpointing and actor persistence for Phase B."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import (
    array_sha256,
    canonical_json_bytes,
)
from oracle_composition.experiments.artifact_io import (
    PublishedArtifact,
    publish_bytes_without_overwrite,
)
from oracle_composition.experiments.fixed_reference import ExperimentContractError

from .policy import (
    ACTION_WIDTH,
    HIDDEN_WIDTH,
    LOG_STD_MAX,
    LOG_STD_MIN,
    OBSERVATION_WIDTH,
    POLICY_INPUT_WIDTH,
    REFERENCE_HORIZON,
    REFERENCE_WIDTH,
    FullAuthorityActor,
    FullAuthorityPolicy,
    LoadedFullAuthorityActor,
    compose_policy_input,
    encode_full_authority_actor,
)
from .strict_npz import MAX_NPY_HEADER_BYTES, decode_strict_npy_member, decode_strict_npz
from .training import COHORT_SEEDS, TrainingPlan, TrainingResult

CHECKPOINT_SCHEMA_ID = "humanoid_phase_b_full_checkpoint/v1"
CHECKPOINT_FORMAT_ID = "deterministic_npz_npy1_no_pickle/v1"
PERSISTENCE_RECEIPT_ID = "humanoid_phase_b_final_persistence/v1"
CHECKPOINT_INDEX_SCHEMA_ID = "humanoid_phase_b_checkpoint_index/v1"
MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024
MAX_ACTOR_EXPORT_BYTES = 4 * 1024 * 1024
MAX_CHECKPOINT_MANIFEST_BYTES = 256 * 1024
MAX_CHECKPOINT_MEMBER_BYTES = 4 * 1024 * 1024
MAX_CHECKPOINT_EXPANDED_BYTES = 32 * 1024 * 1024
MAX_CHECKPOINT_MEMBERS = 59


def _valid_success_resource_controls(value: object) -> bool:
    if type(value) is not dict or set(value) != {
        "cpu_time",
        "environment",
        "executed_modules",
        "filesystem",
        "process_group_cleanup",
        "process_tree_rss",
    }:
        return False
    modules = value["executed_modules"]
    cpu = value["cpu_time"]
    environment = value["environment"]
    cleanup = value["process_group_cleanup"]
    rss = value["process_tree_rss"]

    def sha256(candidate: object) -> bool:
        return (
            type(candidate) is str
            and len(candidate) == 64
            and all(character in "0123456789abcdef" for character in candidate)
        )

    cpu_keys = {"enforcement", "limit_seconds", "resource"}
    if type(cpu) is dict and cpu.get("enforcement") == "os_enforced":
        cpu_keys.add("hard_limit_seconds")
    elif type(cpu) is dict and cpu.get("enforcement") == "unsupported" and "error" in cpu:
        cpu_keys.add("error")
    environment_keys = environment.get("keys") if type(environment) is dict else None
    removed = environment.get("runtime_added_keys_removed") if type(environment) is dict else None
    return (
        type(cpu) is dict
        and set(cpu) == cpu_keys
        and cpu.get("enforcement") in {"os_enforced", "unsupported"}
        and cpu.get("resource") == "RLIMIT_CPU"
        and type(cpu.get("limit_seconds")) is int
        and cpu["limit_seconds"] > 0
        and (
            cpu.get("enforcement") != "unsupported"
            or "error" not in cpu
            or type(cpu["error"]) is str
        )
        and (
            cpu.get("enforcement") != "os_enforced"
            or (
                type(cpu.get("hard_limit_seconds")) is int
                and cpu["hard_limit_seconds"] >= cpu["limit_seconds"]
            )
        )
        and type(environment) is dict
        and set(environment)
        == {
            "allowlist_enforced",
            "environment_sha256",
            "keys",
            "runtime_added_keys_removed",
            "unexpected_keys",
        }
        and environment.get("allowlist_enforced") is True
        and environment.get("unexpected_keys") == []
        and sha256(environment.get("environment_sha256"))
        and type(environment_keys) is list
        and type(removed) is list
        and all(type(item) is str for item in (*environment_keys, *removed))
        and environment_keys == sorted(set(environment_keys))
        and removed == sorted(set(removed))
        and type(modules) is dict
        and set(modules) == {"enforcement", "final_sha256", "start_sha256"}
        and modules.get("enforcement") == "checkout_realpath_and_recorded_digest_verified"
        and sha256(modules.get("start_sha256"))
        and sha256(modules.get("final_sha256"))
        and value["filesystem"] == {"enforcement": "parent_observed_os_best_effort"}
        and type(cleanup) is dict
        and cleanup == {"enforcement": "os_session_group_best_effort", "succeeded": True}
        and type(rss) is dict
        and set(rss) == {"enforcement"}
        and rss.get("enforcement")
        in {"parent_observed_os_best_effort", "unsupported_in_current_os_sandbox"}
    )


_STRICT_EXPORT_SCHEMA: Mapping[str, tuple[tuple[int, ...], np.dtype[object]]] = {
    "latent_pi.0.weight": ((HIDDEN_WIDTH, POLICY_INPUT_WIDTH), np.dtype("<f4")),
    "latent_pi.0.bias": ((HIDDEN_WIDTH,), np.dtype("<f4")),
    "latent_pi.2.weight": ((HIDDEN_WIDTH, HIDDEN_WIDTH), np.dtype("<f4")),
    "latent_pi.2.bias": ((HIDDEN_WIDTH,), np.dtype("<f4")),
    "mu.weight": ((ACTION_WIDTH, HIDDEN_WIDTH), np.dtype("<f4")),
    "mu.bias": ((ACTION_WIDTH,), np.dtype("<f4")),
    "log_std.weight": ((ACTION_WIDTH, HIDDEN_WIDTH), np.dtype("<f4")),
    "log_std.bias": ((ACTION_WIDTH,), np.dtype("<f4")),
    "action_low": ((ACTION_WIDTH,), np.dtype("<f4")),
    "action_high": ((ACTION_WIDTH,), np.dtype("<f4")),
    "input_layout": ((3,), np.dtype("<i8")),
    "log_std_bounds": ((2,), np.dtype("<f4")),
    "format_version": ((1,), np.dtype("<i8")),
    "source_actor_sha256": ((1,), np.dtype("|S64")),
}

_ACTOR_CHECKPOINT_TO_EXPORT = {
    "actor.latent_0.weight": "latent_pi.0.weight",
    "actor.latent_0.bias": "latent_pi.0.bias",
    "actor.latent_2.weight": "latent_pi.2.weight",
    "actor.latent_2.bias": "latent_pi.2.bias",
    "actor.mu.weight": "mu.weight",
    "actor.mu.bias": "mu.bias",
    "actor.log_std.weight": "log_std.weight",
    "actor.log_std.bias": "log_std.bias",
}

_CHECKPOINT_POLICY_SCHEMA: Mapping[str, tuple[tuple[int, ...], np.dtype[object]]] = {
    "policy:actor.action_high": ((ACTION_WIDTH,), np.dtype("<f4")),
    "policy:actor.action_low": ((ACTION_WIDTH,), np.dtype("<f4")),
    "policy:actor.latent_0.bias": ((HIDDEN_WIDTH,), np.dtype("<f4")),
    "policy:actor.latent_0.weight": ((HIDDEN_WIDTH, POLICY_INPUT_WIDTH), np.dtype("<f4")),
    "policy:actor.latent_2.bias": ((HIDDEN_WIDTH,), np.dtype("<f4")),
    "policy:actor.latent_2.weight": ((HIDDEN_WIDTH, HIDDEN_WIDTH), np.dtype("<f4")),
    "policy:actor.log_std.bias": ((ACTION_WIDTH,), np.dtype("<f4")),
    "policy:actor.log_std.weight": ((ACTION_WIDTH, HIDDEN_WIDTH), np.dtype("<f4")),
    "policy:actor.mu.bias": ((ACTION_WIDTH,), np.dtype("<f4")),
    "policy:actor.mu.weight": ((ACTION_WIDTH, HIDDEN_WIDTH), np.dtype("<f4")),
    "policy:value.0.bias": ((HIDDEN_WIDTH,), np.dtype("<f4")),
    "policy:value.0.weight": ((HIDDEN_WIDTH, POLICY_INPUT_WIDTH), np.dtype("<f4")),
    "policy:value.2.bias": ((HIDDEN_WIDTH,), np.dtype("<f4")),
    "policy:value.2.weight": ((HIDDEN_WIDTH, HIDDEN_WIDTH), np.dtype("<f4")),
    "policy:value.4.bias": ((1,), np.dtype("<f4")),
    "policy:value.4.weight": ((1, HIDDEN_WIDTH), np.dtype("<f4")),
}
_OPTIMIZER_PARAMETER_SHAPES = (
    (HIDDEN_WIDTH, POLICY_INPUT_WIDTH),
    (HIDDEN_WIDTH,),
    (HIDDEN_WIDTH, HIDDEN_WIDTH),
    (HIDDEN_WIDTH,),
    (ACTION_WIDTH, HIDDEN_WIDTH),
    (ACTION_WIDTH,),
    (ACTION_WIDTH, HIDDEN_WIDTH),
    (ACTION_WIDTH,),
    (HIDDEN_WIDTH, POLICY_INPUT_WIDTH),
    (HIDDEN_WIDTH,),
    (HIDDEN_WIDTH, HIDDEN_WIDTH),
    (HIDDEN_WIDTH,),
    (1, HIDDEN_WIDTH),
    (1,),
)
_OPTIMIZER_ARRAY_KEY = re.compile(r"^optimizer:(\d+):(exp_avg|exp_avg_sq|step)$")


@dataclass(frozen=True, slots=True)
class LoadedFullCheckpoint:
    policy: FullAuthorityPolicy
    optimizer: object
    metadata: Mapping[str, object]
    sha256: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class PersistenceResult:
    checkpoint: PublishedArtifact
    strict_export: PublishedArtifact
    receipt: PublishedArtifact
    receipt_value: Mapping[str, object]


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, value, version=(1, 0), allow_pickle=False)
    return stream.getvalue()


def _zip_member(name: str) -> ZipInfo:
    member = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    member.compress_type = ZIP_DEFLATED
    member.create_system = 3
    member.external_attr = 0o100600 << 16
    return member


def _jsonable_optimizer_value(value: object) -> object:
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ExperimentContractError("optimizer metadata contains NaN or Inf")
        return value
    if isinstance(value, tuple | list):
        return [_jsonable_optimizer_value(item) for item in value]
    raise ExperimentContractError(f"optimizer metadata type is unsupported: {type(value).__name__}")


def _checkpoint_arrays(
    policy: FullAuthorityPolicy,
    optimizer: object,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    arrays: dict[str, np.ndarray] = {}
    for name, tensor in policy.state_dict().items():
        value = np.ascontiguousarray(tensor.detach().cpu().numpy())
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ExperimentContractError(f"checkpoint policy tensor is non-finite: {name}")
        arrays[f"policy:{name}"] = value
    state = optimizer.state_dict()
    groups = []
    for group in state["param_groups"]:
        groups.append(
            {
                name: (list(value) if name == "params" else _jsonable_optimizer_value(value))
                for name, value in sorted(group.items())
            }
        )
    scalar_state: dict[str, dict[str, object]] = {}
    for parameter_index, values in sorted(state["state"].items()):
        encoded_values: dict[str, object] = {}
        for name, value in sorted(values.items()):
            if isinstance(value, torch.Tensor):
                key = f"optimizer:{parameter_index}:{name}"
                array = np.ascontiguousarray(value.detach().cpu().numpy())
                if array.dtype.kind == "f" and not np.isfinite(array).all():
                    raise ExperimentContractError("checkpoint optimizer tensor is non-finite")
                arrays[key] = array
                encoded_values[name] = {"array_key": key}
            else:
                encoded_values[name] = _jsonable_optimizer_value(value)
        scalar_state[str(parameter_index)] = encoded_values
    return arrays, {"param_groups": groups, "state": scalar_state}


try:
    import torch
except ImportError as exc:  # pragma: no cover - guarded by the train extra
    raise ExperimentContractError("Torch is required for Phase B persistence") from exc


def _actor_from_trained_parameters(
    parameters: Mapping[str, np.ndarray],
    *,
    source_actor_sha256: str,
) -> FullAuthorityActor:
    """Construct through the sealed FT1 type, then install finite trained parameters."""

    seed_parameters = {
        name: np.array(value, dtype="<f4", order="C", copy=True)
        for name, value in parameters.items()
    }
    seed_parameters["latent_pi.0.weight"][:, OBSERVATION_WIDTH:] = np.float32(0.0)
    actor = FullAuthorityActor(seed_parameters, source_actor_sha256=source_actor_sha256)
    destinations = {
        "latent_pi.0.weight": actor.latent_0.weight,
        "latent_pi.0.bias": actor.latent_0.bias,
        "latent_pi.2.weight": actor.latent_2.weight,
        "latent_pi.2.bias": actor.latent_2.bias,
        "mu.weight": actor.mu.weight,
        "mu.bias": actor.mu.bias,
        "log_std.weight": actor.log_std.weight,
        "log_std.bias": actor.log_std.bias,
    }
    if set(parameters) != set(destinations):
        raise ExperimentContractError("trained actor parameter key set differs")
    with torch.no_grad():
        for name, destination in destinations.items():
            value = parameters[name]
            if (
                type(value) is not np.ndarray
                or value.dtype.str != "<f4"
                or tuple(value.shape) != tuple(destination.shape)
                or not value.flags.c_contiguous
                or not np.isfinite(value).all()
            ):
                raise ExperimentContractError(f"trained actor parameter differs: {name}")
            destination.copy_(torch.from_numpy(value))
    return actor


def _decode_strict_actor_export(payload: bytes) -> dict[str, np.ndarray]:
    arrays = decode_strict_npz(
        payload,
        schema=_STRICT_EXPORT_SCHEMA,
        archive_label="strict trained export",
        maximum_member_bytes=MAX_ACTOR_EXPORT_BYTES,
        maximum_total_bytes=MAX_ACTOR_EXPORT_BYTES * 2,
    )
    if (
        not np.array_equal(arrays["action_low"], np.full(ACTION_WIDTH, -0.4, dtype="<f4"))
        or not np.array_equal(arrays["action_high"], np.full(ACTION_WIDTH, 0.4, dtype="<f4"))
        or not np.array_equal(
            arrays["input_layout"],
            np.asarray([OBSERVATION_WIDTH, REFERENCE_HORIZON, REFERENCE_WIDTH], dtype="<i8"),
        )
        or not np.array_equal(
            arrays["log_std_bounds"], np.asarray([LOG_STD_MIN, LOG_STD_MAX], dtype="<f4")
        )
        or not np.array_equal(arrays["format_version"], np.asarray([1], dtype="<i8"))
    ):
        raise ExperimentContractError("strict trained export constants differ")
    return arrays


def load_trained_full_authority_actor(
    path: Path,
    *,
    expected_sha256: str,
) -> LoadedFullAuthorityActor:
    """Strictly reload a final actor while leaving the FT1 E1 loader unchanged."""

    payload = _read_bounded_regular(
        Path(path), expected_sha256=expected_sha256, maximum=MAX_ACTOR_EXPORT_BYTES
    )
    arrays = _decode_strict_actor_export(payload)
    try:
        source_actor_sha256 = bytes(arrays["source_actor_sha256"][0]).decode("ascii")
    except (UnicodeError, ValueError) as exc:
        raise ExperimentContractError("strict trained export source identity differs") from exc
    actor_parameters = {name: arrays[name] for name in _ACTOR_CHECKPOINT_TO_EXPORT.values()}
    actor = _actor_from_trained_parameters(
        actor_parameters,
        source_actor_sha256=source_actor_sha256,
    )
    if encode_full_authority_actor(actor) != payload:
        raise ExperimentContractError("strict trained actor export is not canonical")
    return LoadedFullAuthorityActor(
        actor=actor,
        content_sha256=expected_sha256,
        byte_count=len(payload),
        member_hashes=MappingProxyType(
            {name: array_sha256(value) for name, value in arrays.items()}
        ),
    )


def encode_full_checkpoint(
    *,
    result: TrainingResult,
    plan: TrainingPlan,
) -> bytes:
    """Encode model, value network, optimizer, and exact counters without pickle."""

    if type(result) is not TrainingResult or type(plan) is not TrainingPlan:
        raise ExperimentContractError("checkpoint requires exact training authority")
    facts = dict(result.scientific_facts)
    if (
        facts.get("observed_transitions") != plan.transitions
        or facts.get("planned_transitions") != plan.transitions
        or facts.get("ppo_seed") != plan.seed
    ):
        raise ExperimentContractError("only the exact final transition may be checkpointed")
    arrays, optimizer_value = _checkpoint_arrays(result.policy, result.optimizer)
    records: list[dict[str, object]] = []
    members: list[tuple[str, bytes]] = []
    for index, (key, value) in enumerate(sorted(arrays.items())):
        member_name = f"arrays/{index:04d}.npy"
        payload = _npy_bytes(value)
        records.append(
            {
                "array_sha256": array_sha256(value),
                "dtype": value.dtype.str,
                "key": key,
                "member": member_name,
                "npy_sha256": hashlib.sha256(payload).hexdigest(),
                "shape": list(value.shape),
            }
        )
        members.append((member_name, payload))
    metadata = {
        "actor_source_sha256": result.policy.actor.source_actor_sha256,
        "array_records": records,
        "checkpoint_format_id": CHECKPOINT_FORMAT_ID,
        "checkpoint_schema_id": CHECKPOINT_SCHEMA_ID,
        "evidence_class": plan.evidence_class,
        "execution_manifest_sha256": plan.manifest_sha256,
        "optimizer": optimizer_value,
        "planned_transitions": plan.transitions,
        "ppo_seed": plan.seed,
        "promotable": plan.promotable,
        "rollouts": plan.rollout_count,
        "schema_version": 1,
        "smoke": plan.smoke,
        "test_only": plan.test_only,
        "training_facts_sha256": hashlib.sha256(canonical_json_bytes(facts)).hexdigest(),
        "transitions": plan.transitions,
        "value_initialization_seed": result.policy.value_seed,
    }
    metadata_bytes = canonical_json_bytes(metadata)
    stream = io.BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr(_zip_member("manifest.json"), metadata_bytes, compresslevel=9)
        for name, payload in members:
            archive.writestr(_zip_member(name), payload, compresslevel=9)
    encoded = stream.getvalue()
    if not 0 < len(encoded) <= MAX_CHECKPOINT_BYTES:
        raise ExperimentContractError("full checkpoint exceeds its byte bound")
    return encoded


def _read_bounded_regular(path: Path, *, expected_sha256: str, maximum: int) -> bytes:
    candidate = Path(path)
    before = candidate.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ExperimentContractError("checkpoint must be a regular non-linked file")
    if not 0 < before.st_size <= maximum:
        raise ExperimentContractError("checkpoint byte count is outside its bound")
    encoded = candidate.read_bytes()
    after = candidate.lstat()
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in identity):
        raise ExperimentContractError("checkpoint changed while read")
    if hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise ExperimentContractError("checkpoint SHA-256 differs")
    return encoded


def _bounded_zip_read(archive: ZipFile, info: ZipInfo, *, maximum: int) -> bytes:
    if not 0 < info.file_size <= maximum:
        raise ExperimentContractError("checkpoint member expansion is outside its bound")
    with archive.open(info, "r") as member:
        payload = member.read(info.file_size + 1)
    if len(payload) != info.file_size:
        raise ExperimentContractError("checkpoint bounded member read differs")
    return payload


def _validate_checkpoint_manifest(manifest: object) -> dict[str, object]:
    if type(manifest) is not dict:
        raise ExperimentContractError("full checkpoint manifest must be an object")
    required = {
        "actor_source_sha256",
        "array_records",
        "checkpoint_format_id",
        "checkpoint_schema_id",
        "evidence_class",
        "execution_manifest_sha256",
        "optimizer",
        "planned_transitions",
        "ppo_seed",
        "promotable",
        "rollouts",
        "schema_version",
        "smoke",
        "test_only",
        "training_facts_sha256",
        "transitions",
        "value_initialization_seed",
    }
    if set(manifest) != required or (
        manifest["checkpoint_schema_id"] != CHECKPOINT_SCHEMA_ID
        or manifest["checkpoint_format_id"] != CHECKPOINT_FORMAT_ID
        or manifest["schema_version"] != 1
        or manifest["transitions"] != manifest["planned_transitions"]
    ):
        raise ExperimentContractError("full checkpoint contract differs")
    bounded_integers = {
        "planned_transitions": (1, 1_048_576),
        "ppo_seed": (1, 2_147_483_647),
        "rollouts": (1, 1_000_000),
        "transitions": (1, 1_048_576),
        "value_initialization_seed": (1, 2_147_483_647),
    }
    if any(
        type(manifest[name]) is not int or not minimum <= manifest[name] <= maximum
        for name, (minimum, maximum) in bounded_integers.items()
    ):
        raise ExperimentContractError("full checkpoint integer field lies outside its bound")
    for name in ("promotable", "smoke", "test_only"):
        if type(manifest[name]) is not bool:
            raise ExperimentContractError("full checkpoint boolean field differs")
    for name in (
        "actor_source_sha256",
        "execution_manifest_sha256",
        "training_facts_sha256",
    ):
        value = manifest[name]
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ExperimentContractError("full checkpoint identity field differs")
    return manifest


def _checkpoint_array_schema(
    record: Mapping[str, object],
) -> tuple[tuple[int, ...], np.dtype[object]]:
    key = record.get("key")
    if type(key) is not str:
        raise ExperimentContractError("checkpoint array key is invalid")
    if key in _CHECKPOINT_POLICY_SCHEMA:
        return _CHECKPOINT_POLICY_SCHEMA[key]
    match = _OPTIMIZER_ARRAY_KEY.fullmatch(key)
    if match is None:
        raise ExperimentContractError("checkpoint contains an unknown array key")
    parameter_index = int(match.group(1))
    if parameter_index >= len(_OPTIMIZER_PARAMETER_SHAPES):
        raise ExperimentContractError("checkpoint optimizer parameter index is invalid")
    shape = (1,) if match.group(2) == "step" else _OPTIMIZER_PARAMETER_SHAPES[parameter_index]
    return shape, np.dtype("<f4")


def _decode_full_checkpoint(encoded: bytes) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    if (
        type(encoded) is not bytes
        or not encoded.startswith(b"PK\x03\x04")
        or len(encoded) < 22
        or encoded[-22:-18] != b"PK\x05\x06"
        or encoded[-2:] != b"\x00\x00"
    ):
        raise ExperimentContractError("full checkpoint is not an uncommented NPZ archive")
    try:
        with ZipFile(io.BytesIO(encoded), "r") as archive:
            infos = archive.infolist()
            if (
                len(infos) not in {38, MAX_CHECKPOINT_MEMBERS}
                or len({info.filename for info in infos}) != len(infos)
                or infos[0].filename != "manifest.json"
            ):
                raise ExperimentContractError("checkpoint member count, set, or order differs")
            if (
                archive.comment
                or sum(info.file_size for info in infos) > MAX_CHECKPOINT_EXPANDED_BYTES
            ):
                raise ExperimentContractError("checkpoint total expansion exceeds its bound")
            for info in infos:
                if (
                    info.date_time != (1980, 1, 1, 0, 0, 0)
                    or info.compress_type != ZIP_DEFLATED
                    or info.create_system != 3
                    or ((info.external_attr >> 16) & 0xFFFF) != 0o100600
                    or info.flag_bits != 0
                    or info.extra
                    or info.comment
                    or info.is_dir()
                ):
                    raise ExperimentContractError("checkpoint ZIP metadata differs")
            manifest_bytes = _bounded_zip_read(
                archive,
                infos[0],
                maximum=MAX_CHECKPOINT_MANIFEST_BYTES,
            )
            manifest = _validate_checkpoint_manifest(json.loads(manifest_bytes.decode("utf-8")))
            if canonical_json_bytes(manifest) != manifest_bytes:
                raise ExperimentContractError("checkpoint manifest is not canonical JSON")
            records = manifest["array_records"]
            expected_optimizer_indices = (
                {0, 8, 9, 10, 11, 12, 13}
                if int(manifest["rollouts"]) <= 8
                else set(range(len(_OPTIMIZER_PARAMETER_SHAPES)))
            )
            expected_member_count = (
                1 + len(_CHECKPOINT_POLICY_SCHEMA) + 3 * len(expected_optimizer_indices)
            )
            if (
                type(records) is not list
                or len(infos) != expected_member_count
                or len(records) != expected_member_count - 1
                or tuple(info.filename for info in infos[1:])
                != tuple(record.get("member") for record in records if type(record) is dict)
            ):
                raise ExperimentContractError("checkpoint array member order differs")
            arrays: dict[str, np.ndarray] = {}
            for index, (info, record) in enumerate(zip(infos[1:], records, strict=True)):
                if type(record) is not dict or set(record) != {
                    "array_sha256",
                    "dtype",
                    "key",
                    "member",
                    "npy_sha256",
                    "shape",
                }:
                    raise ExperimentContractError("checkpoint array record differs")
                expected_member = f"arrays/{index:04d}.npy"
                shape, dtype = _checkpoint_array_schema(record)
                if (
                    record["member"] != expected_member
                    or info.filename != expected_member
                    or record["dtype"] != dtype.str
                    or record["shape"] != list(shape)
                    or type(record["array_sha256"]) is not str
                    or type(record["npy_sha256"]) is not str
                    or len(record["array_sha256"]) != 64
                    or len(record["npy_sha256"]) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for digest in (record["array_sha256"], record["npy_sha256"])
                        for character in digest
                    )
                ):
                    raise ExperimentContractError("checkpoint array schema differs")
                expected_data_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                if (
                    not expected_data_bytes < info.file_size <= MAX_CHECKPOINT_MEMBER_BYTES
                    or info.file_size > expected_data_bytes + MAX_NPY_HEADER_BYTES
                ):
                    raise ExperimentContractError(
                        "checkpoint member expansion differs from its array schema"
                    )
                payload = _bounded_zip_read(
                    archive,
                    info,
                    maximum=MAX_CHECKPOINT_MEMBER_BYTES,
                )
                if hashlib.sha256(payload).hexdigest() != record["npy_sha256"]:
                    raise ExperimentContractError("checkpoint NPY identity differs")
                value = decode_strict_npy_member(
                    payload,
                    name=str(record["key"]),
                    shape=shape,
                    dtype=dtype,
                    archive_label="full checkpoint",
                )
                if array_sha256(value) != record["array_sha256"]:
                    raise ExperimentContractError("checkpoint array content differs")
                key = str(record["key"])
                if key in arrays:
                    raise ExperimentContractError("checkpoint array key is duplicated")
                arrays[key] = value
            expected_array_keys = set(_CHECKPOINT_POLICY_SCHEMA) | {
                f"optimizer:{index}:{field}"
                for index in expected_optimizer_indices
                for field in ("exp_avg", "exp_avg_sq", "step")
            }
            if set(arrays) != expected_array_keys:
                raise ExperimentContractError("checkpoint array member set is incomplete")
    except ExperimentContractError:
        raise
    except (
        BadZipFile,
        EOFError,
        KeyError,
        MemoryError,
        OSError,
        OverflowError,
        RecursionError,
        UnicodeError,
        ValueError,
    ) as exc:
        raise ExperimentContractError(f"full checkpoint is invalid: {exc}") from exc
    return manifest, arrays


def load_full_checkpoint(path: Path, *, expected_sha256: str) -> LoadedFullCheckpoint:
    """Strictly load a code-free checkpoint and reconstruct its inference path."""

    encoded = _read_bounded_regular(
        Path(path), expected_sha256=expected_sha256, maximum=MAX_CHECKPOINT_BYTES
    )
    manifest, arrays = _decode_full_checkpoint(encoded)
    actor_arrays = {
        export_name: np.ascontiguousarray(arrays[f"policy:{checkpoint_name}"], dtype="<f4")
        for checkpoint_name, export_name in _ACTOR_CHECKPOINT_TO_EXPORT.items()
    }
    actor = _actor_from_trained_parameters(
        actor_arrays,
        source_actor_sha256=str(manifest["actor_source_sha256"]),
    )
    policy = FullAuthorityPolicy(actor, value_seed=int(manifest["value_initialization_seed"]))
    expected_state = policy.state_dict()
    state = {}
    for name, expected in expected_state.items():
        key = f"policy:{name}"
        if key not in arrays:
            raise ExperimentContractError(f"checkpoint policy tensor is missing: {name}")
        tensor = torch.from_numpy(arrays[key].copy(order="C"))
        if tensor.shape != expected.shape or tensor.dtype != expected.dtype:
            raise ExperimentContractError(f"checkpoint policy tensor schema differs: {name}")
        state[name] = tensor
    if set(key for key in arrays if key.startswith("policy:")) != {
        f"policy:{name}" for name in expected_state
    }:
        raise ExperimentContractError("checkpoint contains an unknown policy tensor")
    policy.load_state_dict(state, strict=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    raw_optimizer = manifest["optimizer"]
    if type(raw_optimizer) is not dict or set(raw_optimizer) != {"param_groups", "state"}:
        raise ExperimentContractError("checkpoint optimizer contract differs")
    saved_state: dict[int, dict[str, object]] = {}
    for index_text, values in raw_optimizer["state"].items():
        if type(index_text) is not str or not index_text.isdigit() or type(values) is not dict:
            raise ExperimentContractError("checkpoint optimizer state key differs")
        restored: dict[str, object] = {}
        for name, value in values.items():
            if type(value) is dict and set(value) == {"array_key"}:
                key = value["array_key"]
                if type(key) is not str or key not in arrays:
                    raise ExperimentContractError("checkpoint optimizer array is missing")
                restored[name] = torch.from_numpy(arrays[key].copy(order="C"))
            else:
                restored[name] = value
        saved_state[int(index_text)] = restored
    groups = raw_optimizer["param_groups"]
    if type(groups) is not list or len(groups) != 1:
        raise ExperimentContractError("checkpoint optimizer group count differs")
    group = dict(groups[0])
    if type(group.get("betas")) is list:
        group["betas"] = tuple(group["betas"])
    optimizer.load_state_dict({"param_groups": [group], "state": saved_state})
    if encode_full_checkpoint_from_loaded(policy, optimizer, manifest) != encoded:
        raise ExperimentContractError("full checkpoint is not canonical or reload-stable")
    return LoadedFullCheckpoint(
        policy=policy,
        optimizer=optimizer,
        metadata=MappingProxyType(manifest),
        sha256=expected_sha256,
        byte_count=len(encoded),
    )


def load_full_checkpoint_metadata(
    path: Path,
    *,
    expected_sha256: str,
) -> Mapping[str, object]:
    """Validate a bounded checkpoint and return metadata without constructing a model."""

    encoded = _read_bounded_regular(
        Path(path), expected_sha256=expected_sha256, maximum=MAX_CHECKPOINT_BYTES
    )
    manifest, _arrays = _decode_full_checkpoint(encoded)
    return MappingProxyType(dict(manifest))


def encode_full_checkpoint_from_loaded(
    policy: FullAuthorityPolicy,
    optimizer: object,
    manifest: Mapping[str, object],
) -> bytes:
    """Re-encode a strict-loaded checkpoint using its immutable scientific metadata."""

    arrays, optimizer_value = _checkpoint_arrays(policy, optimizer)
    records: list[dict[str, object]] = []
    members: list[tuple[str, bytes]] = []
    for index, (key, value) in enumerate(sorted(arrays.items())):
        name = f"arrays/{index:04d}.npy"
        payload = _npy_bytes(value)
        records.append(
            {
                "array_sha256": array_sha256(value),
                "dtype": value.dtype.str,
                "key": key,
                "member": name,
                "npy_sha256": hashlib.sha256(payload).hexdigest(),
                "shape": list(value.shape),
            }
        )
        members.append((name, payload))
    rebuilt = dict(manifest)
    rebuilt["array_records"] = records
    rebuilt["optimizer"] = optimizer_value
    stream = io.BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr(
            _zip_member("manifest.json"), canonical_json_bytes(rebuilt), compresslevel=9
        )
        for name, payload in members:
            archive.writestr(_zip_member(name), payload, compresslevel=9)
    return stream.getvalue()


def frozen_inference_fixtures() -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.Generator(np.random.PCG64(20260905))
    states = np.ascontiguousarray(
        generator.uniform(-0.25, 0.25, size=(4, OBSERVATION_WIDTH)).astype("<f4")
    )
    references = np.ascontiguousarray(
        generator.uniform(-0.25, 0.25, size=(4, REFERENCE_HORIZON, REFERENCE_WIDTH)).astype("<f4")
    )
    return states, references


def _deterministic_actions(policy: FullAuthorityPolicy) -> np.ndarray:
    states, references = frozen_inference_fixtures()
    return policy.actor.act(compose_policy_input(states, references)).physical


def publish_final_persistence(
    *,
    output_directory: Path,
    result: TrainingResult,
    plan: TrainingPlan,
) -> PersistenceResult:
    """Publish exactly one final checkpoint/export pair and its strict receipt."""

    output = Path(output_directory)
    if not output.is_dir() or output.is_symlink():
        raise ExperimentContractError("persistence output must be the fresh real run directory")
    checkpoint_bytes = encode_full_checkpoint(result=result, plan=plan)
    checkpoint = publish_bytes_without_overwrite(
        output / f"checkpoint_seed_{plan.seed}_final.npz", checkpoint_bytes
    )
    loaded = load_full_checkpoint(checkpoint.path, expected_sha256=checkpoint.sha256)
    before_actions = _deterministic_actions(result.policy)
    after_actions = _deterministic_actions(loaded.policy)
    if before_actions.tobytes(order="C") != after_actions.tobytes(order="C"):
        raise ExperimentContractError("checkpoint reload changed deterministic inference bits")
    export_bytes = encode_full_authority_actor(result.policy.actor)
    strict_export = publish_bytes_without_overwrite(
        output / f"actor_seed_{plan.seed}_final.npz", export_bytes
    )
    loaded_export = load_trained_full_authority_actor(
        strict_export.path,
        expected_sha256=strict_export.sha256,
    )
    export_actions = loaded_export.actor.act(
        compose_policy_input(*frozen_inference_fixtures())
    ).physical
    if export_actions.tobytes(order="C") != before_actions.tobytes(order="C"):
        raise ExperimentContractError("checkpoint-to-export inference differs")
    checkpoint_parameters = result.policy.actor.parameter_arrays()
    export_parameters = loaded_export.actor.parameter_arrays()
    if any(
        checkpoint_parameters[name].tobytes(order="C") != export_parameters[name].tobytes(order="C")
        for name in checkpoint_parameters
    ):
        raise ExperimentContractError("checkpoint-to-export actor parameters differ")
    receipt_value = {
        "checkpoint": {
            "byte_count": checkpoint.byte_count,
            "filename": checkpoint.path.name,
            "sha256": checkpoint.sha256,
        },
        "checkpoint_reload_bitwise_deterministic": True,
        "checkpoint_to_export_bitwise_equivalent": True,
        "evidence_class": plan.evidence_class,
        "execution_manifest_sha256": plan.manifest_sha256,
        "final_transition_only": True,
        "fixture_action_sha256": array_sha256(before_actions),
        "persistence_receipt_id": PERSISTENCE_RECEIPT_ID,
        "planned_transitions": plan.transitions,
        "ppo_seed": plan.seed,
        "promotable": plan.promotable,
        "schema_version": 1,
        "smoke": plan.smoke,
        "strict_export": {
            "byte_count": strict_export.byte_count,
            "filename": strict_export.path.name,
            "sha256": strict_export.sha256,
        },
        "test_only": plan.test_only,
        "transitions": plan.transitions,
        "training_facts_sha256": hashlib.sha256(
            canonical_json_bytes(dict(result.scientific_facts))
        ).hexdigest(),
    }
    receipt = publish_bytes_without_overwrite(
        output / f"persistence_seed_{plan.seed}_v1.json",
        canonical_json_bytes(receipt_value),
    )
    return PersistenceResult(
        checkpoint=checkpoint,
        strict_export=strict_export,
        receipt=receipt,
        receipt_value=MappingProxyType(receipt_value),
    )


def publish_checkpoint_index(
    *,
    output_directory: Path,
    entries: Sequence[PersistenceResult],
    success_receipts: Sequence[PublishedArtifact] | None = None,
    execution_manifest: PublishedArtifact | None = None,
    job_result: PublishedArtifact | None = None,
) -> PublishedArtifact:
    """Publish a complete five-checkpoint cohort index; partial cohorts are refused."""

    if len(entries) != 5:
        raise ExperimentContractError("checkpoint index requires exactly five entries")
    if (
        success_receipts is None
        or len(success_receipts) != 5
        or execution_manifest is None
        or job_result is None
    ):
        raise ExperimentContractError("checkpoint index requires success and job authority")
    output = Path(output_directory).resolve()
    manifest_payload = _read_bounded_regular(
        execution_manifest.path,
        expected_sha256=execution_manifest.sha256,
        maximum=2 * 1024 * 1024,
    )
    job_payload = _read_bounded_regular(
        job_result.path,
        expected_sha256=job_result.sha256,
        maximum=256 * 1024,
    )
    try:
        manifest_value = json.loads(manifest_payload)
        job_value = json.loads(job_payload)
    except (UnicodeError, ValueError) as exc:
        raise ExperimentContractError("checkpoint cohort authority is malformed") from exc
    if (
        type(manifest_value) is not dict
        or type(job_value) is not dict
        or canonical_json_bytes(manifest_value) != manifest_payload
        or canonical_json_bytes(job_value) != job_payload
        or manifest_value.get("seeds") != list(COHORT_SEEDS)
        or manifest_value.get("transitions_per_seed") != 1_048_576
        or manifest_value.get("checkpoint_selection") != "final_transition_only"
        or manifest_value.get("smoke") is not False
        or manifest_value.get("test_only") is not False
        or job_value.get("status") != "succeeded"
        or job_value.get("execution_manifest_sha256") != execution_manifest.sha256
    ):
        raise ExperimentContractError("checkpoint cohort execution authority differs")
    success_by_seed: dict[int, tuple[PublishedArtifact, dict[str, object]]] = {}
    for artifact in success_receipts:
        payload = _read_bounded_regular(
            artifact.path,
            expected_sha256=artifact.sha256,
            maximum=64 * 1024,
        )
        try:
            success = json.loads(payload)
        except (UnicodeError, ValueError) as exc:
            raise ExperimentContractError("checkpoint success receipt is malformed") from exc
        seed = success.get("ppo_seed") if type(success) is dict else None
        if (
            type(seed) is not int
            or seed in success_by_seed
            or canonical_json_bytes(success) != payload
            or success.get("success_receipt_id") != "humanoid_phase_b_seed_success/v2"
            or success.get("schema_version") != 2
            or success.get("outcome") != "success"
            or success.get("status") != "succeeded"
            or success.get("evidence_class") != "exploratory_fine_tuning_cycle"
            or success.get("planned_transitions") != 1_048_576
            or success.get("promotable") is not True
            or success.get("smoke") is not False
            or success.get("test_only") is not False
            or success.get("execution_manifest_sha256") != execution_manifest.sha256
            or not _valid_success_resource_controls(success.get("resource_controls"))
            or success.get("worker_cleanup")
            != {"attempted": True, "error": None, "succeeded": True}
        ):
            raise ExperimentContractError("checkpoint success receipt lacks cohort authority")
        job_binding = {
            "byte_count": artifact.byte_count,
            "filename": artifact.path.name,
            "sha256": artifact.sha256,
        }
        if not any(
            type(row) is dict
            and row.get("seed") == seed
            and row.get("status") == "succeeded"
            and row.get("receipt") == job_binding
            for row in job_value.get("outcomes", [])
        ):
            raise ExperimentContractError("job result omits a cohort success receipt")
        success_by_seed[seed] = (artifact, success)
    if tuple(sorted(success_by_seed)) != COHORT_SEEDS:
        raise ExperimentContractError("checkpoint success receipt seed set differs")
    rows = []
    for entry in entries:
        if type(entry) is not PersistenceResult:
            raise ExperimentContractError("checkpoint index entry authority differs")
        value = dict(entry.receipt_value)
        expected = {
            "evidence_class": "exploratory_fine_tuning_cycle",
            "execution_manifest_sha256": execution_manifest.sha256,
            "planned_transitions": 1_048_576,
            "promotable": True,
            "smoke": False,
            "test_only": False,
            "transitions": 1_048_576,
        }
        if any(value.get(field) != expected_value for field, expected_value in expected.items()):
            raise ExperimentContractError("checkpoint index refuses smoke or test-only outcomes")
        _read_bounded_regular(
            entry.checkpoint.path,
            expected_sha256=entry.checkpoint.sha256,
            maximum=MAX_CHECKPOINT_BYTES,
        )
        _read_bounded_regular(
            entry.strict_export.path,
            expected_sha256=entry.strict_export.sha256,
            maximum=MAX_ACTOR_EXPORT_BYTES,
        )
        receipt_payload = _read_bounded_regular(
            entry.receipt.path,
            expected_sha256=entry.receipt.sha256,
            maximum=64 * 1024,
        )
        loaded_checkpoint = load_full_checkpoint(
            entry.checkpoint.path,
            expected_sha256=entry.checkpoint.sha256,
        )
        loaded_export = load_trained_full_authority_actor(
            entry.strict_export.path,
            expected_sha256=entry.strict_export.sha256,
        )
        success_artifact, success_value = success_by_seed[int(value["ppo_seed"])]
        success_artifacts = success_value.get("artifacts")
        if (
            value.get("checkpoint")
            != {
                "byte_count": entry.checkpoint.byte_count,
                "filename": entry.checkpoint.path.name,
                "sha256": entry.checkpoint.sha256,
            }
            or value.get("strict_export")
            != {
                "byte_count": entry.strict_export.byte_count,
                "filename": entry.strict_export.path.name,
                "sha256": entry.strict_export.sha256,
            }
            or receipt_payload != canonical_json_bytes(value)
            or loaded_checkpoint.metadata.get("ppo_seed") != value.get("ppo_seed")
            or loaded_checkpoint.metadata.get("execution_manifest_sha256")
            != execution_manifest.sha256
            or loaded_checkpoint.metadata.get("planned_transitions") != 1_048_576
            or loaded_checkpoint.metadata.get("transitions") != 1_048_576
            or loaded_checkpoint.metadata.get("promotable") is not True
            or loaded_checkpoint.metadata.get("smoke") is not False
            or loaded_checkpoint.metadata.get("test_only") is not False
            or loaded_export.actor.source_actor_sha256
            != loaded_checkpoint.policy.actor.source_actor_sha256
            or type(success_artifacts) is not dict
            or type(success_artifacts.get("persistence")) is not dict
            or success_artifacts.get("persistence", {}).get("sha256") != entry.receipt.sha256
        ):
            raise ExperimentContractError("checkpoint index entry binding differs")

        def indexed(artifact: PublishedArtifact) -> dict[str, object]:
            try:
                relative = artifact.path.resolve().relative_to(output)
            except ValueError as exc:
                raise ExperimentContractError("checkpoint index entry escapes its cohort") from exc
            return {
                "byte_count": artifact.byte_count,
                "path": relative.as_posix(),
                "sha256": artifact.sha256,
            }

        rows.append(
            {
                "checkpoint": indexed(entry.checkpoint),
                "persistence_receipt": indexed(entry.receipt),
                "ppo_seed": value["ppo_seed"],
                "strict_export": indexed(entry.strict_export),
                "success_receipt": indexed(success_artifact),
            }
        )
    rows.sort(key=lambda item: int(item["ppo_seed"]))
    if tuple(int(row["ppo_seed"]) for row in rows) != COHORT_SEEDS:
        raise ExperimentContractError("checkpoint index seed set differs from the frozen cohort")
    value = {
        "cohort_seeds": list(COHORT_SEEDS),
        "checkpoint_count": 5,
        "checkpoint_index_schema_id": CHECKPOINT_INDEX_SCHEMA_ID,
        "entries": rows,
        "execution_manifest_sha256": execution_manifest.sha256,
        "job_result_sha256": job_result.sha256,
        "schema_version": 1,
        "selection_rule": "final_transition_only_no_replacement",
        "summary": {
            "full_budget_success_count": len(rows),
            "promotable_checkpoint_count": len(rows),
        },
    }
    return publish_bytes_without_overwrite(
        output / "checkpoint_index_v1.json",
        canonical_json_bytes(value),
    )


__all__ = [
    "CHECKPOINT_INDEX_SCHEMA_ID",
    "CHECKPOINT_SCHEMA_ID",
    "LoadedFullCheckpoint",
    "PersistenceResult",
    "encode_full_checkpoint",
    "frozen_inference_fixtures",
    "load_full_checkpoint",
    "load_full_checkpoint_metadata",
    "load_trained_full_authority_actor",
    "publish_checkpoint_index",
    "publish_final_persistence",
]
