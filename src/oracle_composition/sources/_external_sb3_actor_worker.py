"""Isolated weights-only worker for one pinned SB3 policy state mapping."""

from __future__ import annotations

import hashlib
import io
import json
import os
import resource
import struct
import sys
from collections import OrderedDict

MAX_INPUT_BYTES = 8 * 1024 * 1024
ADDRESS_SPACE_LIMIT_BYTES = 8 * 1024 * 1024 * 1024
CPU_TIME_LIMIT_SECONDS = 30
RESPONSE_MAGIC = b"HH_EXTERNAL_SB3_ACTOR_V1\x00"
SAFE_GLOBAL_MODULES = ("builtins", "traceback", "collections", "torch")

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
    pass


def _safe_globals_observation(
    entries: object,
    *,
    refusal_category: str,
) -> tuple[tuple[str, ...], str]:
    if type(entries) is not list:
        raise WorkerRefusal(refusal_category)
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
            raise WorkerRefusal(refusal_category)
        qualified_names.append(f"{module}.{qualname}")
    names = tuple(sorted(qualified_names))
    encoded = json.dumps(
        names,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return names, hashlib.sha256(encoded).hexdigest()


def _apply_resource_limits() -> dict[str, object]:
    resource.setrlimit(
        resource.RLIMIT_CPU,
        (CPU_TIME_LIMIT_SECONDS, CPU_TIME_LIMIT_SECONDS),
    )
    finite_address_space = True
    try:
        resource.setrlimit(
            resource.RLIMIT_AS,
            (ADDRESS_SPACE_LIMIT_BYTES, ADDRESS_SPACE_LIMIT_BYTES),
        )
    except (OSError, ValueError):
        if sys.platform != "darwin":
            raise WorkerRefusal("address-space resource limit refused") from None
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
        raise WorkerRefusal("input size refused")
    return payload


def _validate_policy_state(state: object, torch: object) -> tuple[dict[str, object], bytes]:
    if type(state) is not dict:
        raise WorkerRefusal("policy state type refused")
    expected_names = tuple(name for name, _ in POLICY_STATE_SCHEMA)
    if tuple(state) != expected_names or any(type(name) is not str for name in state):
        raise WorkerRefusal("policy state schema refused")

    storage_addresses: set[int] = set()
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
            raise WorkerRefusal("tensor contract refused")
        storage = tensor.untyped_storage()
        expected_storage_bytes = tensor.numel() * tensor.element_size()
        if storage.nbytes() != expected_storage_bytes:
            raise WorkerRefusal("tensor storage bound refused")
        address = storage.data_ptr()
        if address <= 0 or address in storage_addresses:
            raise WorkerRefusal("tensor storage alias refused")
        storage_addresses.add(address)
        if not bool(torch.isfinite(tensor).all().item()):
            raise WorkerRefusal("non-finite tensor refused")
        inventory.append({"name": name, "shape": list(shape)})
        if name.startswith("actor."):
            raw = tensor.detach().numpy().tobytes(order="C")
            if len(raw) != expected_storage_bytes:
                raise WorkerRefusal("actor tensor byte count refused")
            actor_payloads.append(raw)

    actor_bytes = b"".join(actor_payloads)
    expected_actor_bytes = sum(_numel(shape) * 4 for _, shape in ACTOR_STATE_SCHEMA)
    if len(actor_bytes) != expected_actor_bytes:
        raise WorkerRefusal("actor payload length refused")
    return {"policy_key_inventory": inventory}, actor_bytes


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for width in shape:
        result *= width
    return result


def _load_and_encode(held_bytes: bytes, limits: dict[str, object]) -> bytes:
    if sys.byteorder != "little":
        raise WorkerRefusal("host byte order refused")
    import torch

    before = torch.serialization.get_safe_globals()
    safe_global_names, safe_globals_sha256 = _safe_globals_observation(
        before,
        refusal_category="process safe globals refused",
    )
    torch_load_call_count = 0
    torch_load_call_count += 1
    loaded = torch.load(io.BytesIO(held_bytes), map_location="cpu", weights_only=True)
    if torch_load_call_count != 1:
        raise WorkerRefusal("torch load call count refused")
    after = torch.serialization.get_safe_globals()
    after_names, _ = _safe_globals_observation(
        after,
        refusal_category="process safe globals changed",
    )
    if len(after) != len(before) or after_names != safe_global_names:
        raise WorkerRefusal("process safe globals changed")
    if type(loaded) is OrderedDict:
        state = dict(loaded)
        raw_mapping_type = "collections.OrderedDict"
    elif type(loaded) is dict:
        state = loaded
        raw_mapping_type = "dict"
    else:
        raise WorkerRefusal("policy state mapping type refused")
    if type(state) is not dict:
        raise WorkerRefusal("policy state normalization refused")
    details, actor_bytes = _validate_policy_state(state, torch)
    header = {
        "protocol_version": 1,
        "torch_version": torch.__version__,
        "torch_load": {
            "call_count": torch_load_call_count,
            "input": "held_bytes_io.BytesIO",
            "map_location": "cpu",
            "weights_only": True,
            "safe_globals_count": len(before),
            "safe_globals_sha256": safe_globals_sha256,
            "safe_globals_module_allowlist_passed": True,
        },
        "mapping": {
            "raw_type": raw_mapping_type,
            "validated_result_type": "dict",
        },
        "resource_limits": limits,
        "actor_payload_byte_count": len(actor_bytes),
        **details,
    }
    header_bytes = json.dumps(
        header,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return RESPONSE_MAGIC + struct.pack(">Q", len(header_bytes)) + header_bytes + actor_bytes


def main() -> int:
    try:
        limits = _apply_resource_limits()
        response = _load_and_encode(_read_input(), limits)
        sys.stdout.buffer.write(response)
        sys.stdout.buffer.flush()
        return 0
    except BaseException:
        os.write(2, b"external SB3 actor worker refused input\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
