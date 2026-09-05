"""Isolated weights-only worker for one pinned SB3 policy state mapping."""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import io
import json
import os
import resource
import struct
import sys
from collections import OrderedDict
from pathlib import Path

MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_FRAME_BYTES = 2 * 1024 * 1024
MAX_HEADER_BYTES = 64 * 1024
MAX_SOURCE_BYTES = 1024 * 1024
ADDRESS_SPACE_LIMIT_BYTES = 8 * 1024 * 1024 * 1024
CPU_TIME_LIMIT_SECONDS = 30
PROTOCOL_VERSION = 2
RESPONSE_MAGIC = b"HH_EXTERNAL_SB3_ACTOR_V2\x00"
FRAME_LENGTHS = struct.Struct(">II")
SAFE_GLOBAL_MODULES = ("builtins", "traceback", "collections", "torch")
EXPECTED_TORCH_RUNTIME_VERSION = "2.14.0"
EXPECTED_TORCH_DISTRIBUTION = {"name": "torch", "version": "2.14.0"}
WORKER_SOURCE_LOGICAL_PATH = "src/oracle_composition/sources/_external_sb3_actor_worker.py"

OUTCOME_SUCCESS = "SUCCESS"
OUTCOME_TIMEOUT = "TIMEOUT"
OUTCOME_CRASH_OR_SIGNAL = "CRASH_OR_SIGNAL"
OUTCOME_RESOURCE_REFUSAL = "RESOURCE_REFUSAL"
OUTCOME_SAFE_GLOBALS_DRIFT = "SAFE_GLOBALS_DRIFT"
OUTCOME_MALFORMED_FRAME = "MALFORMED_FRAME"
OUTCOME_SCHEMA_OR_TENSOR_REFUSAL = "SCHEMA_OR_TENSOR_REFUSAL"
OUTCOME_NON_FINITE_VALUES = "NON_FINITE_VALUES"
OUTCOME_RUNTIME_DRIFT = "RUNTIME_DRIFT"
OUTCOME_INPUT_REFUSAL = "INPUT_REFUSAL"
OUTCOME_INTERNAL_ERROR = "INTERNAL_ERROR"
WORKER_OUTCOMES = frozenset(
    {
        OUTCOME_SUCCESS,
        OUTCOME_TIMEOUT,
        OUTCOME_CRASH_OR_SIGNAL,
        OUTCOME_RESOURCE_REFUSAL,
        OUTCOME_SAFE_GLOBALS_DRIFT,
        OUTCOME_MALFORMED_FRAME,
        OUTCOME_SCHEMA_OR_TENSOR_REFUSAL,
        OUTCOME_NON_FINITE_VALUES,
        OUTCOME_RUNTIME_DRIFT,
        OUTCOME_INPUT_REFUSAL,
        OUTCOME_INTERNAL_ERROR,
    }
)

POLICY_STATE_SCHEMA = (
    ("actor.latent_pi.0.weight", (256, 348)),
    ("actor.latent_pi.0.bias", (256,)),
    ("actor.latent_pi.2.weight", (256, 256)),
    ("actor.latent_pi.2.bias", (256,)),
    ("actor.mu.weight", (17, 256)),
    ("actor.mu.bias", (17,)),
    ("actor.log_std.weight", (17, 256)),
    ("actor.log_std.bias", (17,)),
    ("critic.qf0.0.weight", (256, 365)),
    ("critic.qf0.0.bias", (256,)),
    ("critic.qf0.2.weight", (256, 256)),
    ("critic.qf0.2.bias", (256,)),
    ("critic.qf0.4.weight", (25, 256)),
    ("critic.qf0.4.bias", (25,)),
    ("critic.qf1.0.weight", (256, 365)),
    ("critic.qf1.0.bias", (256,)),
    ("critic.qf1.2.weight", (256, 256)),
    ("critic.qf1.2.bias", (256,)),
    ("critic.qf1.4.weight", (25, 256)),
    ("critic.qf1.4.bias", (25,)),
    ("critic_target.qf0.0.weight", (256, 365)),
    ("critic_target.qf0.0.bias", (256,)),
    ("critic_target.qf0.2.weight", (256, 256)),
    ("critic_target.qf0.2.bias", (256,)),
    ("critic_target.qf0.4.weight", (25, 256)),
    ("critic_target.qf0.4.bias", (25,)),
    ("critic_target.qf1.0.weight", (256, 365)),
    ("critic_target.qf1.0.bias", (256,)),
    ("critic_target.qf1.2.weight", (256, 256)),
    ("critic_target.qf1.2.bias", (256,)),
    ("critic_target.qf1.4.weight", (25, 256)),
    ("critic_target.qf1.4.bias", (25,)),
)
ACTOR_STATE_SCHEMA = POLICY_STATE_SCHEMA[:8]


class WorkerRefusal(Exception):
    def __init__(self, outcome: str, detail: str) -> None:
        if type(outcome) is not str or outcome not in WORKER_OUTCOMES or outcome == OUTCOME_SUCCESS:
            raise ValueError("worker refusal outcome is invalid")
        if type(detail) is not str or not detail or len(detail) > 160:
            raise ValueError("worker refusal detail is invalid")
        self.outcome = outcome
        self.detail = detail
        super().__init__(detail)


def _safe_globals_observation(
    entries: object,
    *,
    refusal_category: str,
) -> tuple[tuple[str, ...], str]:
    if type(entries) is not list:
        raise WorkerRefusal(OUTCOME_SAFE_GLOBALS_DRIFT, refusal_category)
    qualified_names: list[str] = []
    for entry in entries:
        module = getattr(entry, "__module__", None)
        qualname = getattr(entry, "__qualname__", None)
        if (
            type(module) is not str
            or type(qualname) is not str
            or not module
            or not qualname
            or not (module in SAFE_GLOBAL_MODULES or module.startswith("torch."))
        ):
            raise WorkerRefusal(OUTCOME_SAFE_GLOBALS_DRIFT, refusal_category)
        qualified_names.append(f"{module}.{qualname}")
    ordered_names = tuple(qualified_names)
    stable_names = tuple(sorted(ordered_names))
    encoded = json.dumps(
        stable_names,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return ordered_names, hashlib.sha256(encoded).hexdigest()


def _apply_resource_limits() -> dict[str, object]:
    try:
        resource.setrlimit(
            resource.RLIMIT_CPU,
            (CPU_TIME_LIMIT_SECONDS, CPU_TIME_LIMIT_SECONDS),
        )
    except (OSError, ValueError) as exc:
        raise WorkerRefusal(
            OUTCOME_RESOURCE_REFUSAL,
            "CPU resource limit refused",
        ) from exc
    finite_address_space = True
    try:
        resource.setrlimit(
            resource.RLIMIT_AS,
            (ADDRESS_SPACE_LIMIT_BYTES, ADDRESS_SPACE_LIMIT_BYTES),
        )
    except (OSError, ValueError):
        if sys.platform != "darwin":
            raise WorkerRefusal(
                OUTCOME_RESOURCE_REFUSAL,
                "address-space resource limit refused",
            ) from None
        inherited = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, inherited)
        finite_address_space = False
    observed_as = resource.getrlimit(resource.RLIMIT_AS)
    observed_cpu = resource.getrlimit(resource.RLIMIT_CPU)
    return {
        "cpu": {
            "resource": "RLIMIT_CPU",
            "soft_seconds": observed_cpu[0],
            "hard_seconds": observed_cpu[1],
        },
        "address_space": {
            "resource": "RLIMIT_AS",
            "requested_bytes": ADDRESS_SPACE_LIMIT_BYTES,
            "finite_enforced": finite_address_space,
            "observed_soft": observed_as[0],
            "observed_hard": observed_as[1],
            "darwin_finite_limit_unavailable": (
                sys.platform == "darwin" and not finite_address_space
            ),
        },
    }


def _read_input() -> bytes:
    payload = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    if not 0 < len(payload) <= MAX_INPUT_BYTES:
        raise WorkerRefusal(OUTCOME_INPUT_REFUSAL, "input size refused")
    return payload


def _worker_source_identity() -> dict[str, object]:
    try:
        payload = Path(__file__).resolve(strict=True).read_bytes()
    except OSError as exc:
        raise WorkerRefusal(OUTCOME_RUNTIME_DRIFT, "worker source unavailable") from exc
    if not 0 < len(payload) <= MAX_SOURCE_BYTES:
        raise WorkerRefusal(OUTCOME_RUNTIME_DRIFT, "worker source size refused")
    return {
        "logical_path": WORKER_SOURCE_LOGICAL_PATH,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_count": len(payload),
    }


def _installed_torch_distribution_identity() -> dict[str, str]:
    try:
        distribution = importlib_metadata.distribution("torch")
        name = distribution.metadata["Name"]
        version = distribution.version
    except (KeyError, OSError, ValueError, importlib_metadata.PackageNotFoundError) as exc:
        raise WorkerRefusal(OUTCOME_RUNTIME_DRIFT, "Torch distribution unavailable") from exc
    if type(name) is not str or type(version) is not str:
        raise WorkerRefusal(OUTCOME_RUNTIME_DRIFT, "Torch distribution identity invalid")
    return {"name": name, "version": version}


def _validate_torch_runtime(torch: object) -> dict[str, object]:
    raw_runtime_version = getattr(torch, "__version__", None)
    distribution = _installed_torch_distribution_identity()
    if (
        not isinstance(raw_runtime_version, str)
        or str(raw_runtime_version) != EXPECTED_TORCH_RUNTIME_VERSION
        or distribution != EXPECTED_TORCH_DISTRIBUTION
    ):
        raise WorkerRefusal(OUTCOME_RUNTIME_DRIFT, "Torch runtime identity differs")
    return {
        "runtime_version": str(raw_runtime_version),
        "distribution": distribution,
    }


def _validate_policy_state(state: object, torch: object) -> tuple[dict[str, object], bytes]:
    if type(state) is not dict:
        raise WorkerRefusal(OUTCOME_SCHEMA_OR_TENSOR_REFUSAL, "policy state type refused")
    expected_names = tuple(name for name, _ in POLICY_STATE_SCHEMA)
    if tuple(state) != expected_names or any(type(name) is not str for name in state):
        raise WorkerRefusal(OUTCOME_SCHEMA_OR_TENSOR_REFUSAL, "policy state schema refused")

    storage_intervals: list[tuple[int, int]] = []
    actor_payloads: list[bytes] = []
    inventory: list[dict[str, object]] = []
    for name, shape in POLICY_STATE_SCHEMA:
        tensor = state[name]
        if (
            type(tensor) is not torch.Tensor
            or tensor.dtype != torch.float32
            or tensor.device.type != "cpu"
            or tensor.device.index is not None
            or tensor.layout != torch.strided
            or tuple(tensor.shape) != shape
            or not tensor.is_contiguous()
            or tensor.storage_offset() != 0
            or tensor.requires_grad
        ):
            raise WorkerRefusal(
                OUTCOME_SCHEMA_OR_TENSOR_REFUSAL,
                "tensor contract refused",
            )
        storage = tensor.untyped_storage()
        expected_storage_bytes = tensor.numel() * tensor.element_size()
        if storage.nbytes() != expected_storage_bytes:
            raise WorkerRefusal(
                OUTCOME_SCHEMA_OR_TENSOR_REFUSAL,
                "tensor storage bound refused",
            )
        address = storage.data_ptr()
        end = address + expected_storage_bytes
        if (
            address <= 0
            or end <= address
            or any(
                address < other_end and other_start < end
                for other_start, other_end in storage_intervals
            )
        ):
            raise WorkerRefusal(
                OUTCOME_SCHEMA_OR_TENSOR_REFUSAL,
                "tensor storage alias refused",
            )
        storage_intervals.append((address, end))
        if not bool(torch.isfinite(tensor).all().item()):
            raise WorkerRefusal(OUTCOME_NON_FINITE_VALUES, "non-finite tensor refused")
        inventory.append({"name": name, "shape": list(shape)})
        if name.startswith("actor."):
            raw = tensor.detach().numpy().tobytes(order="C")
            if len(raw) != expected_storage_bytes:
                raise WorkerRefusal(
                    OUTCOME_SCHEMA_OR_TENSOR_REFUSAL,
                    "actor tensor byte count refused",
                )
            actor_payloads.append(raw)

    actor_bytes = b"".join(actor_payloads)
    expected_actor_bytes = sum(_numel(shape) * 4 for _, shape in ACTOR_STATE_SCHEMA)
    if len(actor_bytes) != expected_actor_bytes:
        raise WorkerRefusal(
            OUTCOME_SCHEMA_OR_TENSOR_REFUSAL,
            "actor payload length refused",
        )
    return {"policy_key_inventory": inventory}, actor_bytes


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for width in shape:
        result *= width
    return result


def _encode_frame(header: dict[str, object], payload: bytes = b"") -> bytes:
    header_bytes = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    if not 0 < len(header_bytes) <= MAX_HEADER_BYTES or len(payload) > MAX_FRAME_BYTES:
        raise WorkerRefusal(OUTCOME_INTERNAL_ERROR, "worker frame bound refused")
    frame = RESPONSE_MAGIC + FRAME_LENGTHS.pack(len(header_bytes), len(payload))
    frame += header_bytes + payload
    if len(frame) > MAX_FRAME_BYTES:
        raise WorkerRefusal(OUTCOME_INTERNAL_ERROR, "worker frame bound refused")
    return frame


def _error_frame(refusal: WorkerRefusal) -> bytes:
    return _encode_frame(
        {
            "protocol_version": PROTOCOL_VERSION,
            "outcome": refusal.outcome,
            "detail": refusal.detail,
        }
    )


def _load_and_encode(held_bytes: bytes, limits: dict[str, object]) -> bytes:
    if sys.byteorder != "little":
        raise WorkerRefusal(OUTCOME_RUNTIME_DRIFT, "host byte order refused")
    import torch

    runtime_identity = _validate_torch_runtime(torch)
    worker_source_identity = _worker_source_identity()
    before = torch.serialization.get_safe_globals()
    safe_global_names, safe_globals_sha256 = _safe_globals_observation(
        before,
        refusal_category="process safe globals refused",
    )
    torch_load_call_count = 1
    try:
        loaded = torch.load(io.BytesIO(held_bytes), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise WorkerRefusal(
            OUTCOME_SCHEMA_OR_TENSOR_REFUSAL,
            "weights-only deserialization refused",
        ) from exc
    after = torch.serialization.get_safe_globals()
    after_names, _ = _safe_globals_observation(
        after,
        refusal_category="process safe globals changed",
    )
    if after_names != safe_global_names:
        raise WorkerRefusal(OUTCOME_SAFE_GLOBALS_DRIFT, "process safe globals changed")
    if type(loaded) is OrderedDict:
        state = dict(loaded)
        raw_mapping_type = "collections.OrderedDict"
    elif type(loaded) is dict:
        state = loaded
        raw_mapping_type = "dict"
    else:
        raise WorkerRefusal(
            OUTCOME_SCHEMA_OR_TENSOR_REFUSAL,
            "policy state mapping type refused",
        )
    if type(state) is not dict:
        raise WorkerRefusal(
            OUTCOME_SCHEMA_OR_TENSOR_REFUSAL,
            "policy state normalization refused",
        )
    details, actor_bytes = _validate_policy_state(state, torch)
    header = {
        "protocol_version": PROTOCOL_VERSION,
        "outcome": OUTCOME_SUCCESS,
        "runtime_identity": runtime_identity,
        "worker_source_identity": worker_source_identity,
        "torch_load": {
            "call_count": torch_load_call_count,
            "input": "held_bytes_io.BytesIO",
            "map_location": "cpu",
            "weights_only": True,
            "safe_globals_count": len(before),
            "safe_globals_sha256": safe_globals_sha256,
            "safe_globals_module_allowlist_passed": True,
            "ordered_names_unchanged": True,
        },
        "mapping": {
            "raw_type": raw_mapping_type,
            "validated_result_type": "dict",
        },
        "resource_limits": limits,
        "actor_payload_byte_count": len(actor_bytes),
        **details,
    }
    return _encode_frame(header, actor_bytes)


def _write_frame(frame: bytes) -> None:
    view = memoryview(frame)
    while view:
        written = os.write(1, view)
        if written <= 0:
            raise OSError("worker output closed")
        view = view[written:]


def main() -> int:
    try:
        limits = _apply_resource_limits()
        frame = _load_and_encode(_read_input(), limits)
        _write_frame(frame)
        return 0
    except WorkerRefusal as exc:
        try:
            _write_frame(_error_frame(exc))
        except BaseException:
            return 3
        return 2
    except BaseException:
        try:
            refusal = WorkerRefusal(OUTCOME_INTERNAL_ERROR, "worker internal refusal")
            _write_frame(_error_frame(refusal))
        except BaseException:
            return 3
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
