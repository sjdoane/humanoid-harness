"""Bounded persistence and strict reload for the frozen TQC v2 attempt."""

from __future__ import annotations

import gc
import hashlib
import io
import math
import os
import stat
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .artifact_io import (
    PublishedArtifact,
    ReservedJsonArtifact,
    ReservedStreamingArtifact,
    finite_pretty_json,
    read_verified_artifact_bytes,
    reserve_json_artifact,
    reserve_streaming_artifact,
    verified_artifact_reader,
)
from .fixed_reference import ExperimentContractError
from .runtime_identity import space_sha256
from .tqc_actor_npz import (
    LoadedTQCActor,
    actor_schema_sha256,
    actor_state_sha256,
    encode_actor_npz,
    load_actor_npz,
    validate_actor_arrays,
)
from .tqc_calibration_contract import canonical_json
from .tqc_development_contract_v2 import TOTAL_ENVIRONMENT_STEPS
from .tqc_development_resource_v2 import TQCResourceMonitorV2
from .tqc_development_training_v2 import (
    EXPECTED_GRADIENT_UPDATES,
    TQCTrainingCompletionAuthorityV2,
    _consume_tqc_training_model_v2,
    revalidate_tqc_training_completion_v2,
)

PERSISTENCE_ID = "tqc_dev_1m_v2_trusted_local_persistence/v1"
STRICT_RELOAD_ID = "tqc_dev_1m_v2_strict_reload/v1"
MODEL_FILENAME = "tqc_model_step_1000000.zip"
REPLAY_FILENAME = "tqc_replay_step_1000000.pkl"
ACTOR_FILENAME = "tqc_actor_step_1000000.npz"
ACTOR_MANIFEST_FILENAME = "tqc_actor_step_1000000.manifest.json"
MAX_MODEL_BYTES = 100 * 1024 * 1024
MAX_REPLAY_BYTES = 8 * 1024**3
MAX_ACTOR_BYTES = 2 * 1024 * 1024
MAX_ACTOR_MANIFEST_BYTES = 1024 * 1024
EXPECTED_REPLAY_ARRAY_PAYLOAD_BYTES = 5_648_000_000
ARRAY_CHUNK_BYTES = 1024 * 1024
REPLAY_ARRAY_ORDER = (
    "observations",
    "next_observations",
    "actions",
    "rewards",
    "dones",
    "timeouts",
)
OPTIMIZER_GROUP_FIELDS = (
    "lr",
    "betas",
    "eps",
    "weight_decay",
    "amsgrad",
    "maximize",
    "foreach",
    "capturable",
    "differentiable",
    "fused",
    "decoupled_weight_decay",
)
STRICT_RELOAD_HASH_PAIRS = (
    "policy_parameters_and_buffers_pre_save_vs_reloaded",
    "log_entropy_coefficient_pre_save_vs_reloaded",
    "actor_optimizer_state_and_param_groups_pre_save_vs_reloaded",
    "critic_optimizer_state_and_param_groups_pre_save_vs_reloaded",
    "entropy_optimizer_state_and_param_groups_pre_save_vs_reloaded",
    "replay_arrays_pre_save_vs_reloaded",
    "replay_metadata_pre_save_vs_reloaded",
)
MEMORY_SAFE_RELOAD_LIFECYCLE = (
    "persist_model_replay_and_actor_from_original_model",
    "close_original_training_env_release_original_model_and_replay_then_collect_garbage",
    "load_bound_model_and_validate_effective_model_and_empty_replay_structure",
    "set_loaded_replay_buffer_to_none_then_collect_garbage",
    "load_bound_same_attempt_replay_once",
    "validate_exact_replay_class_shapes_dtypes_position_full_and_counters",
    "release_strict_reload_model_and_replay_then_collect_garbage_before_protected_evaluation",
)
PERSISTENCE_CLAIM = "persistence_integrity_only_no_behavior_tracker_or_oracle_claim/v1"
STRICT_RELOAD_CLAIM = "strict_reload_integrity_only_no_behavior_tracker_or_oracle_claim/v1"

_PERSISTENCE_ISSUER = object()
_STRICT_RELOAD_ISSUER = object()
_PERSISTENCE_SEAL_ISSUER = object()
_STRICT_RELOAD_SEAL_ISSUER = object()


@dataclass(slots=True)
class _TQCPersistenceSealV2:
    """Bind one canonical authority payload to its process-local objects."""

    _payload: bytes = field(repr=False)
    _creator_pid: int
    _completion_identity: int
    _loaded_actor_identity: int
    _phase: str
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _PERSISTENCE_SEAL_ISSUER:
            raise ExperimentContractError("TQC persistence seals are private")
        if (
            not self._payload
            or self._creator_pid != os.getpid()
            or self._completion_identity <= 0
            or self._loaded_actor_identity <= 0
            or self._phase != "issued"
        ):
            raise ExperimentContractError("TQC persistence seal is invalid")

    def validate(
        self,
        payload: Mapping[str, object],
        *,
        completion: object,
        loaded_actor: object,
    ) -> None:
        if (
            os.getpid() != self._creator_pid
            or self._phase != "issued"
            or id(completion) != self._completion_identity
            or id(loaded_actor) != self._loaded_actor_identity
            or canonical_json(dict(payload)) != self._payload
        ):
            raise ExperimentContractError("TQC persistence authority seal differs")

    def begin_reload(
        self,
        payload: Mapping[str, object],
        *,
        completion: object,
        loaded_actor: object,
    ) -> None:
        self.validate(payload, completion=completion, loaded_actor=loaded_actor)
        self._phase = "reloading"

    def validate_consumed(
        self,
        payload: Mapping[str, object],
        *,
        completion: object,
        loaded_actor: object,
    ) -> None:
        if (
            os.getpid() != self._creator_pid
            or self._phase != "consumed"
            or id(completion) != self._completion_identity
            or id(loaded_actor) != self._loaded_actor_identity
            or canonical_json(dict(payload)) != self._payload
        ):
            raise ExperimentContractError("consumed TQC persistence authority seal differs")

    def finish_reload(self) -> None:
        if os.getpid() != self._creator_pid or self._phase != "reloading":
            raise ExperimentContractError("TQC persistence reload phase differs")
        self._phase = "consumed"

    def fail_reload(self) -> None:
        if self._phase == "reloading":
            self._phase = "failed"


@dataclass(slots=True)
class _TQCStrictReloadSealV2:
    """Bind one completed reload payload without replaying issuer validation."""

    _payload: bytes = field(repr=False)
    _creator_pid: int
    _persistence_identity: int
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _STRICT_RELOAD_SEAL_ISSUER:
            raise ExperimentContractError("TQC strict-reload seals are private")
        if not self._payload or self._creator_pid != os.getpid() or self._persistence_identity <= 0:
            raise ExperimentContractError("TQC strict-reload seal is invalid")

    def validate(
        self,
        payload: Mapping[str, object],
        *,
        persistence_authority: object,
    ) -> None:
        if (
            os.getpid() != self._creator_pid
            or id(persistence_authority) != self._persistence_identity
            or canonical_json(dict(payload)) != self._payload
        ):
            raise ExperimentContractError("TQC strict-reload authority seal differs")


def _require_sha256(value: object, *, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"{field_name} must be a lowercase SHA-256")
    return value


def _qualified_type(value: object) -> str:
    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def _projection(completion: TQCTrainingCompletionAuthorityV2) -> dict[str, Any]:
    design = completion.design.to_dict()
    projection = design.get("training_projection")
    if type(projection) is not dict:
        raise ExperimentContractError("TQC v2 training projection is unavailable")
    return projection


def _validate_persistence_projection(projection: Mapping[str, Any]) -> None:
    persistence = projection.get("persistence")
    gates = projection.get("resource_gates")
    replay = projection.get("replay_buffer")
    if not isinstance(persistence, Mapping) or not isinstance(gates, Mapping):
        raise ExperimentContractError("TQC v2 persistence design is unavailable")
    if not isinstance(replay, Mapping):
        raise ExperimentContractError("TQC v2 replay design is unavailable")
    trusted = persistence.get("trusted_local_state")
    actor = persistence.get("actor_export")
    if not isinstance(trusted, Mapping) or not isinstance(actor, Mapping):
        raise ExperimentContractError("TQC v2 persistence subcontracts are unavailable")
    expected = {
        "checkpoint_rule": "exact_final_1000000_environment_step_checkpoint_only",
        "artifact_order": [
            "trusted_local_model",
            "trusted_local_replay",
            "strict_actor_export",
            "strict_reload_receipts",
            "protected_evaluation",
        ],
        "exclusive_creation_required": True,
        "content_sha256_and_byte_count_required": True,
        "source_step_and_design_binding_required": True,
        "execution_manifest_binding_required": True,
    }
    if any(persistence.get(name) != value for name, value in expected.items()):
        raise ExperimentContractError("TQC v2 persistence contract differs")
    trusted_expected = {
        "model_filename": MODEL_FILENAME,
        "replay_filename": REPLAY_FILENAME,
        "replay_expected_array_payload_bytes": EXPECTED_REPLAY_ARRAY_PAYLOAD_BYTES,
        "replay_no_overwrite": True,
        "replay_partial_artifact_eligible": False,
        "replay_streaming_sha256_and_byte_count_required": True,
        "simultaneous_full_replay_buffers_allowed": 1,
        "strict_reload_byte_exact_state_hash_equality_required": True,
        "model_reload_required": True,
        "replay_reload_required": True,
        "reloaded_num_timesteps_required": TOTAL_ENVIRONMENT_STEPS,
        "reloaded_n_updates_required": EXPECTED_GRADIENT_UPDATES,
        "reloaded_replay_position_required": 0,
        "reloaded_replay_full_required": True,
        "public_distribution_allowed": False,
    }
    if any(trusted.get(name) != value for name, value in trusted_expected.items()):
        raise ExperimentContractError("TQC v2 trusted-local persistence contract differs")
    if trusted.get("replay_array_hash_order") != list(REPLAY_ARRAY_ORDER):
        raise ExperimentContractError("TQC v2 replay hash order differs")
    if trusted.get("optimizer_param_group_fields") != list(OPTIMIZER_GROUP_FIELDS):
        raise ExperimentContractError("TQC v2 optimizer hash fields differ")
    strict_expected = {
        "strict_reload_hash_pairs": list(STRICT_RELOAD_HASH_PAIRS),
        "memory_safe_reload_lifecycle": list(MEMORY_SAFE_RELOAD_LIFECYCLE),
        "strict_reload_state_hash_rule": (
            "ordered_canonical_JSON_entries_of_name_dtype_shape_and_length_prefixed_array_sha256/v1"
        ),
        "replay_metadata_hash_rule": ("canonical_JSON_exact_primitive_types_no_object_repr/v1"),
        "optimizer_state_hash_fields_per_owned_parameter": [
            "step",
            "exp_avg",
            "exp_avg_sq",
        ],
        "load_policy": "same_attempt_locally_created_exact_paths_only_never_uploaded",
        "strict_model_load_call": {
            "method": "sb3_contrib.TQC.load/v2.9",
            "path_argument": "exact_bound_exclusive_binary_reader",
            "env_argument": None,
            "device": "cpu",
            "print_system_info": False,
            "force_reset": True,
        },
        "strict_replay_load_call": {
            "method": "sb3_contrib.TQC.load_replay_buffer/v2.9",
            "path_argument": "exact_bound_exclusive_binary_reader",
            "truncate_last_traj": False,
        },
    }
    if any(trusted.get(name) != value for name, value in strict_expected.items()):
        raise ExperimentContractError("TQC v2 strict-reload contract differs")
    if (
        actor.get("filename") != ACTOR_FILENAME
        or actor.get("manifest_filename") != ACTOR_MANIFEST_FILENAME
    ):
        raise ExperimentContractError("TQC v2 actor persistence filenames differ")
    gate_expected = {
        "maximum_trusted_model_archive_bytes": MAX_MODEL_BYTES,
        "maximum_trusted_replay_archive_bytes": MAX_REPLAY_BYTES,
        "maximum_actor_export_bytes": MAX_ACTOR_BYTES,
        "maximum_actor_manifest_bytes": MAX_ACTOR_MANIFEST_BYTES,
        "replay_buffer_allocation_bytes": EXPECTED_REPLAY_ARRAY_PAYLOAD_BYTES,
    }
    if any(gates.get(name) != value for name, value in gate_expected.items()):
        raise ExperimentContractError("TQC v2 persistence byte gates differ")
    if replay.get("expected_allocation_bytes") != EXPECTED_REPLAY_ARRAY_PAYLOAD_BYTES:
        raise ExperimentContractError("TQC v2 replay allocation contract differs")


def _finite_primitive(value: object, *, field_name: str) -> object:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ExperimentContractError(f"{field_name} is non-finite")
        return value
    if type(value) in {tuple, list}:
        return [
            _finite_primitive(item, field_name=f"{field_name}[{index}]")
            for index, item in enumerate(value)
        ]
    raise ExperimentContractError(f"{field_name} is not an exact JSON primitive")


def _array_entry(name: str, value: np.ndarray) -> dict[str, object]:
    """Hash a finite C-order array without materializing its byte payload."""

    if (
        type(name) is not str
        or not name
        or not isinstance(value, np.ndarray)
        or not value.flags.c_contiguous
        or value.dtype.hasobject
    ):
        raise ExperimentContractError(f"state array {name!r} is not safe C-order data")
    flat = value.reshape(-1)
    elements_per_chunk = max(1, ARRAY_CHUNK_BYTES // max(1, value.dtype.itemsize))
    for offset in range(0, flat.size, elements_per_chunk):
        if not bool(np.isfinite(flat[offset : offset + elements_per_chunk]).all()):
            raise ExperimentContractError(f"state array {name} contains non-finite values")
    header = canonical_json({"dtype": value.dtype.str, "shape": list(value.shape)})
    byte_count = int(value.nbytes)
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(byte_count.to_bytes(8, "big"))
    byte_view = memoryview(value).cast("B")
    for offset in range(0, byte_count, ARRAY_CHUNK_BYTES):
        digest.update(byte_view[offset : offset + ARRAY_CHUNK_BYTES])
    return {
        "name": name,
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "byte_count": byte_count,
        "array_sha256": digest.hexdigest(),
    }


def _ordered_entry_sha256(entries: tuple[dict[str, object], ...]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        encoded = canonical_json(entry)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _tensor_entry(name: str, value: object) -> dict[str, object]:
    import torch

    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ExperimentContractError(f"state tensor {name} must be an exact CPU tensor")
    array = value.detach().numpy()
    if not array.flags.c_contiguous:
        raise ExperimentContractError(f"state tensor {name} must be C-contiguous")
    return _array_entry(name, array)


def _policy_state_sha256(model: object) -> str:
    state = getattr(getattr(model, "policy", None), "state_dict", None)
    if not callable(state):
        raise ExperimentContractError("TQC policy state is unavailable")
    entries = tuple(_tensor_entry(name, value) for name, value in state().items())
    if not entries:
        raise ExperimentContractError("TQC policy state must be nonempty")
    return _ordered_entry_sha256(entries)


def _entropy_state_sha256(model: object) -> str:
    return _ordered_entry_sha256(
        (_tensor_entry("algorithm.log_ent_coef", getattr(model, "log_ent_coef", None)),)
    )


def _optimizer_state_sha256(
    optimizer: object,
    named_parameters: tuple[tuple[str, object], ...],
    *,
    role: str,
) -> str:
    import torch

    if type(optimizer) is not torch.optim.Adam or len(optimizer.param_groups) != 1:
        raise ExperimentContractError(f"TQC {role} optimizer structure differs")
    group = optimizer.param_groups[0]
    if set(group) != {*OPTIMIZER_GROUP_FIELDS, "params"}:
        raise ExperimentContractError(f"TQC {role} optimizer fields differ")
    owned = tuple(group["params"])
    expected = tuple(parameter for _name, parameter in named_parameters)
    if len(owned) != len(expected) or any(
        left is not right for left, right in zip(owned, expected, strict=True)
    ):
        raise ExperimentContractError(f"TQC {role} optimizer ownership differs")
    if set(optimizer.state) != set(expected):
        raise ExperimentContractError(f"TQC {role} optimizer state ownership differs")
    group_payload = {
        name: _finite_primitive(group[name], field_name=f"{role}.{name}")
        for name in OPTIMIZER_GROUP_FIELDS
    }
    group_payload["ordered_parameter_names"] = [name for name, _value in named_parameters]
    state_entries: list[dict[str, object]] = []
    for parameter_name, parameter in named_parameters:
        values = optimizer.state[parameter]
        if type(values) is not dict or set(values) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ExperimentContractError(f"TQC {role} optimizer state fields differ")
        step = values["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.numel() != 1
            or float(step.item()) != float(EXPECTED_GRADIENT_UPDATES)
        ):
            raise ExperimentContractError(f"TQC {role} optimizer step differs")
        fields = tuple(
            _tensor_entry(f"{parameter_name}.{field_name}", values[field_name])
            for field_name in ("step", "exp_avg", "exp_avg_sq")
        )
        state_entries.append(
            {
                "parameter_name": parameter_name,
                "parameter_dtype": parameter.detach().numpy().dtype.str,
                "parameter_shape": list(parameter.shape),
                "fields_sha256": _ordered_entry_sha256(fields),
            }
        )
    return hashlib.sha256(
        canonical_json(
            {
                "param_group": group_payload,
                "role": role,
                "state": state_entries,
            }
        )
    ).hexdigest()


def _model_state_receipt(model: object) -> dict[str, str]:
    if (
        type(getattr(model, "num_timesteps", None)) is not int
        or model.num_timesteps != TOTAL_ENVIRONMENT_STEPS
        or type(getattr(model, "_n_updates", None)) is not int
        or model._n_updates != EXPECTED_GRADIENT_UPDATES
    ):
        raise ExperimentContractError("TQC persistence model counters differ")
    actor = getattr(model, "actor", None)
    critic = getattr(model, "critic", None)
    entropy = getattr(model, "log_ent_coef", None)
    if actor is None or critic is None or entropy is None:
        raise ExperimentContractError("TQC persistence model state is incomplete")
    actor_parameters = tuple(actor.named_parameters())
    critic_parameters = tuple(critic.named_parameters())
    return {
        "policy_state_sha256": _policy_state_sha256(model),
        "entropy_state_sha256": _entropy_state_sha256(model),
        "actor_optimizer_state_sha256": _optimizer_state_sha256(
            actor.optimizer,
            actor_parameters,
            role="actor",
        ),
        "critic_optimizer_state_sha256": _optimizer_state_sha256(
            critic.optimizer,
            critic_parameters,
            role="critic",
        ),
        "entropy_optimizer_state_sha256": _optimizer_state_sha256(
            getattr(model, "ent_coef_optimizer", None),
            (("algorithm.log_ent_coef", entropy),),
            role="entropy_coefficient",
        ),
    }


def _replay_state_receipt(
    model: object,
    projection: Mapping[str, Any],
    *,
    final: bool = True,
) -> dict[str, object]:
    from stable_baselines3.common.buffers import ReplayBuffer

    replay = getattr(model, "replay_buffer", None)
    replay_design = projection.get("replay_buffer")
    if type(replay) is not ReplayBuffer or not isinstance(replay_design, Mapping):
        raise ExperimentContractError("TQC replay state is unavailable")
    entries = tuple(_array_entry(name, getattr(replay, name, None)) for name in REPLAY_ARRAY_ORDER)
    payload_bytes = sum(int(entry["byte_count"]) for entry in entries)
    expected_metadata = {
        "class_id": "stable_baselines3.common.buffers.ReplayBuffer",
        "requested_transition_capacity": replay_design["requested_transition_capacity"],
        "internal_vector_slot_capacity": replay_design["internal_vector_slot_capacity"],
        "n_envs": replay_design["n_envs"],
        "position": replay_design["expected_final_position"] if final else 0,
        "full": replay_design["expected_final_full"] if final else False,
        "optimize_memory_usage": replay_design["optimize_memory_usage"],
        "handle_timeout_termination": replay_design["handle_timeout_termination"],
        "n_step_return": replay_design["n_step_return"],
        "device_type": "cpu",
        "observation_space_sha256": space_sha256(replay.observation_space),
        "action_space_sha256": space_sha256(replay.action_space),
    }
    observed_metadata = {
        "class_id": _qualified_type(replay),
        "requested_transition_capacity": replay.buffer_size * replay.n_envs,
        "internal_vector_slot_capacity": replay.buffer_size,
        "n_envs": replay.n_envs,
        "position": replay.pos,
        "full": replay.full,
        "optimize_memory_usage": replay.optimize_memory_usage,
        "handle_timeout_termination": replay.handle_timeout_termination,
        "n_step_return": getattr(model, "n_steps", None),
        "device_type": getattr(getattr(replay, "device", None), "type", None),
        "observation_space_sha256": space_sha256(replay.observation_space),
        "action_space_sha256": space_sha256(replay.action_space),
    }
    metadata_types = {
        "class_id": str,
        "requested_transition_capacity": int,
        "internal_vector_slot_capacity": int,
        "n_envs": int,
        "position": int,
        "full": bool,
        "optimize_memory_usage": bool,
        "handle_timeout_termination": bool,
        "n_step_return": int,
        "device_type": str,
        "observation_space_sha256": str,
        "action_space_sha256": str,
    }
    if any(
        type(observed_metadata[name]) is not expected_type
        for name, expected_type in metadata_types.items()
    ):
        raise ExperimentContractError("TQC replay metadata uses an inexact primitive type")
    if observed_metadata != expected_metadata:
        raise ExperimentContractError("TQC replay metadata differs from the frozen design")
    expected_arrays = {
        "observations": (
            (
                replay_design["internal_vector_slot_capacity"],
                replay_design["n_envs"],
                *replay_design["observation_shape"],
            ),
            replay_design["observation_dtype"],
        ),
        "next_observations": (
            (
                replay_design["internal_vector_slot_capacity"],
                replay_design["n_envs"],
                *replay_design["observation_shape"],
            ),
            replay_design["observation_dtype"],
        ),
        "actions": (
            (
                replay_design["internal_vector_slot_capacity"],
                replay_design["n_envs"],
                *replay_design["action_shape"],
            ),
            replay_design["action_dtype"],
        ),
        "rewards": (
            (replay_design["internal_vector_slot_capacity"], replay_design["n_envs"]),
            replay_design["reward_dtype"],
        ),
        "dones": (
            (replay_design["internal_vector_slot_capacity"], replay_design["n_envs"]),
            replay_design["done_dtype"],
        ),
        "timeouts": (
            (replay_design["internal_vector_slot_capacity"], replay_design["n_envs"]),
            replay_design["timeout_dtype"],
        ),
    }
    for entry in entries:
        shape, dtype = expected_arrays[str(entry["name"])]
        if entry["shape"] != list(shape) or entry["dtype"] != dtype:
            raise ExperimentContractError(f"TQC replay array {entry['name']} differs")
    if payload_bytes != replay_design["expected_allocation_bytes"]:
        raise ExperimentContractError("TQC replay array payload size differs")
    return {
        "array_entries": entries,
        "array_payload_bytes": payload_bytes,
        "arrays_sha256": _ordered_entry_sha256(entries),
        "metadata": observed_metadata,
        "metadata_sha256": hashlib.sha256(canonical_json(observed_metadata)).hexdigest(),
    }


def _actor_arrays(model: object) -> dict[str, np.ndarray]:
    actor = getattr(model, "actor", None)
    state_dict = getattr(actor, "state_dict", None)
    action_space = getattr(actor, "action_space", None)
    if not callable(state_dict) or action_space is None:
        raise ExperimentContractError("TQC actor state is unavailable")
    arrays: dict[str, np.ndarray] = {}
    for name, tensor in state_dict().items():
        value = tensor.detach()
        if value.device.type != "cpu":
            raise ExperimentContractError("TQC actor persistence requires CPU tensors")
        arrays[name] = np.ascontiguousarray(value.numpy(), dtype="<f4")
    arrays.update(
        {
            "action_low": np.ascontiguousarray(action_space.low, dtype="<f4"),
            "action_high": np.ascontiguousarray(action_space.high, dtype="<f4"),
            "format_version": np.asarray([1], dtype="<i8"),
        }
    )
    return validate_actor_arrays(arrays)


def _directory_signature(descriptor: int) -> tuple[int, int, int, int, int, int]:
    observed = os.fstat(descriptor)
    return (
        int(observed.st_dev),
        int(observed.st_ino),
        int(observed.st_mode),
        int(observed.st_nlink),
        int(observed.st_uid),
        int(observed.st_gid),
    )


def _assert_reservation_parent(
    reservation: ReservedStreamingArtifact | ReservedJsonArtifact,
    monitor: TQCResourceMonitorV2,
) -> None:
    duplicate = monitor.duplicate_work_directory_descriptor()
    try:
        if _directory_signature(duplicate) != _directory_signature(reservation._parent_descriptor):
            raise ExperimentContractError(
                "artifact reservation parent differs from the retained work directory"
            )
    finally:
        os.close(duplicate)


def _validate_work_directory(
    completion: TQCTrainingCompletionAuthorityV2,
    work_directory: Path,
) -> Path:
    requested = Path(work_directory)
    absolute = Path(os.path.abspath(requested))
    if not requested.is_absolute() or requested != absolute:
        raise ExperimentContractError("persistence work directory must be normalized and absolute")
    duplicate = completion.resource_monitor.duplicate_work_directory_descriptor()
    try:
        observed = os.fstat(duplicate)
        visible = os.stat(absolute, follow_symlinks=False)
        if (
            not stat.S_ISDIR(visible.st_mode)
            or (observed.st_dev, observed.st_ino) != (visible.st_dev, visible.st_ino)
            or stat.S_IMODE(visible.st_mode) != 0o700
        ):
            raise ExperimentContractError(
                "persistence path differs from the retained work directory"
            )
    except OSError as exc:
        raise ExperimentContractError("persistence work directory is unavailable") from exc
    finally:
        os.close(duplicate)
    return absolute


def _validate_artifact_record(
    artifact: PublishedArtifact,
    *,
    work_directory: Path,
    filename: str,
    maximum_bytes: int,
) -> None:
    if (
        type(artifact) is not PublishedArtifact
        or artifact.path != work_directory / filename
        or type(artifact.byte_count) is not int
        or not 0 < artifact.byte_count <= maximum_bytes
    ):
        raise ExperimentContractError(f"persisted artifact {filename} binding differs")
    _require_sha256(artifact.sha256, field_name=f"{filename} SHA-256")
    try:
        observed = os.stat(artifact.path, follow_symlinks=False)
    except OSError as exc:
        raise ExperimentContractError(f"persisted artifact {filename} is unavailable") from exc
    if (
        not stat.S_ISREG(observed.st_mode)
        or observed.st_nlink != 1
        or stat.S_IMODE(observed.st_mode) != 0o600
        or observed.st_size != artifact.byte_count
    ):
        raise ExperimentContractError(f"persisted artifact {filename} metadata differs")


def _artifact_dict(artifact: PublishedArtifact) -> dict[str, object]:
    return {
        "filename": artifact.path.name,
        "sha256": artifact.sha256,
        "byte_count": artifact.byte_count,
    }


@dataclass(frozen=True, slots=True)
class TQCPersistenceAuthorityV2:
    """Process-local proof that the exact final state was persisted once."""

    persistence_id: str
    attempt_id: str
    execution_manifest_sha256: str
    claimed_work_directory_identity: str
    work_directory: Path
    worker_pid: int
    design_file_sha256: str
    design_semantic_sha256: str
    training_projection_sha256: str
    training_integrity_sha256: str
    source_environment_steps: int
    source_gradient_updates: int
    source_policy_state_sha256: str
    source_entropy_state_sha256: str
    source_actor_optimizer_state_sha256: str
    source_critic_optimizer_state_sha256: str
    source_entropy_optimizer_state_sha256: str
    source_replay_arrays_sha256: str
    source_replay_metadata_sha256: str
    source_replay_array_payload_bytes: int
    source_actor_state_sha256: str
    model_artifact: PublishedArtifact
    replay_artifact: PublishedArtifact
    actor_artifact: PublishedArtifact
    actor_manifest_artifact: PublishedArtifact
    persistence_sha256: str
    all_artifacts_persisted: bool
    behavioral_evidence: bool
    claim_boundary: str
    loaded_actor: LoadedTQCActor = field(repr=False, compare=False)
    training_completion: TQCTrainingCompletionAuthorityV2 = field(repr=False, compare=False)
    _seal: _TQCPersistenceSealV2 = field(repr=False, compare=False)
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _PERSISTENCE_ISSUER:
            raise ExperimentContractError(
                "TQC persistence authority may only be issued by exact persistence"
            )
        self._validate_sealed()

    def _validate_payload_seal(self) -> None:
        if type(self._seal) is not _TQCPersistenceSealV2:
            raise ExperimentContractError("TQC persistence authority seal is unavailable")
        self._seal.validate(
            self.to_dict(),
            completion=self.training_completion,
            loaded_actor=self.loaded_actor,
        )

    def _validate_sealed(self) -> None:
        self._validate_payload_seal()
        for name in (
            "execution_manifest_sha256",
            "claimed_work_directory_identity",
            "design_file_sha256",
            "design_semantic_sha256",
            "training_projection_sha256",
            "training_integrity_sha256",
            "source_policy_state_sha256",
            "source_entropy_state_sha256",
            "source_actor_optimizer_state_sha256",
            "source_critic_optimizer_state_sha256",
            "source_entropy_optimizer_state_sha256",
            "source_replay_arrays_sha256",
            "source_replay_metadata_sha256",
            "source_actor_state_sha256",
            "persistence_sha256",
        ):
            _require_sha256(getattr(self, name), field_name=name)
        if type(self.training_completion) is not TQCTrainingCompletionAuthorityV2:
            raise ExperimentContractError("TQC persistence training authority differs")
        if type(self.loaded_actor) is not LoadedTQCActor:
            raise ExperimentContractError("TQC persistence strict actor load differs")
        completion = self.training_completion
        if (
            self.persistence_id != PERSISTENCE_ID
            or self.attempt_id != completion.attempt_id
            or self.execution_manifest_sha256 != completion.execution_manifest_sha256
            or self.claimed_work_directory_identity != completion.claimed_work_directory_identity
            or self.worker_pid != completion.worker_pid
            or self.worker_pid != os.getpid()
            or self.design_file_sha256 != completion.design_file_sha256
            or self.design_semantic_sha256 != completion.design_semantic_sha256
            or self.training_projection_sha256 != completion.training_projection_sha256
            or self.training_integrity_sha256 != completion.training_integrity_sha256
            or self.source_environment_steps != TOTAL_ENVIRONMENT_STEPS
            or self.source_gradient_updates != EXPECTED_GRADIENT_UPDATES
            or self.source_replay_array_payload_bytes != EXPECTED_REPLAY_ARRAY_PAYLOAD_BYTES
            or self.source_actor_state_sha256 != self.loaded_actor.state_sha256
            or self.actor_artifact.sha256 != self.loaded_actor.content_sha256
            or self.actor_artifact.byte_count != self.loaded_actor.byte_count
            or self.all_artifacts_persisted is not True
            or self.behavioral_evidence is not False
            or self.claim_boundary != PERSISTENCE_CLAIM
        ):
            raise ExperimentContractError("TQC persistence authority is inconsistent")
        for artifact, filename, maximum in (
            (self.model_artifact, MODEL_FILENAME, MAX_MODEL_BYTES),
            (self.replay_artifact, REPLAY_FILENAME, MAX_REPLAY_BYTES),
            (self.actor_artifact, ACTOR_FILENAME, MAX_ACTOR_BYTES),
            (
                self.actor_manifest_artifact,
                ACTOR_MANIFEST_FILENAME,
                MAX_ACTOR_MANIFEST_BYTES,
            ),
        ):
            _validate_artifact_record(
                artifact,
                work_directory=self.work_directory,
                filename=filename,
                maximum_bytes=maximum,
            )
        expected = hashlib.sha256(
            canonical_json(
                {key: value for key, value in self.to_dict().items() if key != "persistence_sha256"}
            )
        ).hexdigest()
        if self.persistence_sha256 != expected:
            raise ExperimentContractError("TQC persistence authority content hash differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "persistence_id": self.persistence_id,
            "attempt_id": self.attempt_id,
            "execution_manifest_sha256": self.execution_manifest_sha256,
            "claimed_work_directory_identity": self.claimed_work_directory_identity,
            "work_directory": str(self.work_directory),
            "worker_pid": self.worker_pid,
            "design_file_sha256": self.design_file_sha256,
            "design_semantic_sha256": self.design_semantic_sha256,
            "training_projection_sha256": self.training_projection_sha256,
            "training_integrity_sha256": self.training_integrity_sha256,
            "source_environment_steps": self.source_environment_steps,
            "source_gradient_updates": self.source_gradient_updates,
            "source_policy_state_sha256": self.source_policy_state_sha256,
            "source_entropy_state_sha256": self.source_entropy_state_sha256,
            "source_actor_optimizer_state_sha256": (self.source_actor_optimizer_state_sha256),
            "source_critic_optimizer_state_sha256": (self.source_critic_optimizer_state_sha256),
            "source_entropy_optimizer_state_sha256": (self.source_entropy_optimizer_state_sha256),
            "source_replay_arrays_sha256": self.source_replay_arrays_sha256,
            "source_replay_metadata_sha256": self.source_replay_metadata_sha256,
            "source_replay_array_payload_bytes": self.source_replay_array_payload_bytes,
            "source_actor_state_sha256": self.source_actor_state_sha256,
            "model_artifact": _artifact_dict(self.model_artifact),
            "replay_artifact": _artifact_dict(self.replay_artifact),
            "actor_artifact": _artifact_dict(self.actor_artifact),
            "actor_manifest_artifact": _artifact_dict(self.actor_manifest_artifact),
            "persistence_sha256": self.persistence_sha256,
            "all_artifacts_persisted": self.all_artifacts_persisted,
            "behavioral_evidence": self.behavioral_evidence,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class TQCStrictReloadAuthorityV2:
    """Process-local proof of all seven frozen strict-reload hash pairs."""

    strict_reload_id: str
    attempt_id: str
    execution_manifest_sha256: str
    claimed_work_directory_identity: str
    training_integrity_sha256: str
    persistence_sha256: str
    actor_equivalence_receipt_sha256: str
    source_environment_steps: int
    loaded_environment_steps: int
    source_gradient_updates: int
    loaded_gradient_updates: int
    source_policy_state_sha256: str
    loaded_policy_state_sha256: str
    source_entropy_state_sha256: str
    loaded_entropy_state_sha256: str
    source_actor_optimizer_state_sha256: str
    loaded_actor_optimizer_state_sha256: str
    source_critic_optimizer_state_sha256: str
    loaded_critic_optimizer_state_sha256: str
    source_entropy_optimizer_state_sha256: str
    loaded_entropy_optimizer_state_sha256: str
    source_replay_arrays_sha256: str
    loaded_replay_arrays_sha256: str
    source_replay_metadata_sha256: str
    loaded_replay_metadata_sha256: str
    model_artifact_sha256: str
    replay_artifact_sha256: str
    original_model_released: bool
    original_replay_released: bool
    reload_model_released: bool
    reload_replay_released: bool
    simultaneous_full_replay_buffers_allowed: int
    passed: bool
    behavioral_evidence: bool
    continuation_claim: str
    claim_boundary: str
    strict_reload_sha256: str
    persistence_authority: TQCPersistenceAuthorityV2 = field(repr=False, compare=False)
    _seal: _TQCStrictReloadSealV2 = field(repr=False, compare=False)
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _STRICT_RELOAD_ISSUER:
            raise ExperimentContractError(
                "TQC strict-reload authority may only be issued by exact reload"
            )
        self._validate_sealed()

    def _validate_sealed(self) -> None:
        if type(self._seal) is not _TQCStrictReloadSealV2:
            raise ExperimentContractError("TQC strict-reload authority seal is unavailable")
        self._seal.validate(
            self.to_dict(),
            persistence_authority=self.persistence_authority,
        )
        persistence = self.persistence_authority
        if type(persistence._seal) is not _TQCPersistenceSealV2:
            raise ExperimentContractError("TQC persistence authority seal is unavailable")
        persistence._seal.validate_consumed(
            persistence.to_dict(),
            completion=persistence.training_completion,
            loaded_actor=persistence.loaded_actor,
        )
        for name in (
            "execution_manifest_sha256",
            "claimed_work_directory_identity",
            "training_integrity_sha256",
            "persistence_sha256",
            "actor_equivalence_receipt_sha256",
            "source_policy_state_sha256",
            "loaded_policy_state_sha256",
            "source_entropy_state_sha256",
            "loaded_entropy_state_sha256",
            "source_actor_optimizer_state_sha256",
            "loaded_actor_optimizer_state_sha256",
            "source_critic_optimizer_state_sha256",
            "loaded_critic_optimizer_state_sha256",
            "source_entropy_optimizer_state_sha256",
            "loaded_entropy_optimizer_state_sha256",
            "source_replay_arrays_sha256",
            "loaded_replay_arrays_sha256",
            "source_replay_metadata_sha256",
            "loaded_replay_metadata_sha256",
            "model_artifact_sha256",
            "replay_artifact_sha256",
            "strict_reload_sha256",
        ):
            _require_sha256(getattr(self, name), field_name=name)
        authority = self.persistence_authority
        pairs_equal = all(
            left == right
            for left, right in (
                (self.source_policy_state_sha256, self.loaded_policy_state_sha256),
                (self.source_entropy_state_sha256, self.loaded_entropy_state_sha256),
                (
                    self.source_actor_optimizer_state_sha256,
                    self.loaded_actor_optimizer_state_sha256,
                ),
                (
                    self.source_critic_optimizer_state_sha256,
                    self.loaded_critic_optimizer_state_sha256,
                ),
                (
                    self.source_entropy_optimizer_state_sha256,
                    self.loaded_entropy_optimizer_state_sha256,
                ),
                (self.source_replay_arrays_sha256, self.loaded_replay_arrays_sha256),
                (self.source_replay_metadata_sha256, self.loaded_replay_metadata_sha256),
            )
        )
        if (
            type(authority) is not TQCPersistenceAuthorityV2
            or self.strict_reload_id != STRICT_RELOAD_ID
            or self.attempt_id != authority.attempt_id
            or self.execution_manifest_sha256 != authority.execution_manifest_sha256
            or self.claimed_work_directory_identity != authority.claimed_work_directory_identity
            or self.training_integrity_sha256 != authority.training_integrity_sha256
            or self.persistence_sha256 != authority.persistence_sha256
            or self.source_environment_steps != TOTAL_ENVIRONMENT_STEPS
            or self.loaded_environment_steps != TOTAL_ENVIRONMENT_STEPS
            or self.source_gradient_updates != EXPECTED_GRADIENT_UPDATES
            or self.loaded_gradient_updates != EXPECTED_GRADIENT_UPDATES
            or self.model_artifact_sha256 != authority.model_artifact.sha256
            or self.replay_artifact_sha256 != authority.replay_artifact.sha256
            or not pairs_equal
            or self.original_model_released is not True
            or self.original_replay_released is not True
            or self.reload_model_released is not True
            or self.reload_replay_released is not True
            or self.simultaneous_full_replay_buffers_allowed != 1
            or self.passed is not True
            or self.behavioral_evidence is not False
            or self.continuation_claim
            != "operational_continuation_only_not_bitwise_equivalent_resume"
            or self.claim_boundary != STRICT_RELOAD_CLAIM
        ):
            raise ExperimentContractError("TQC strict-reload authority is inconsistent")
        expected = hashlib.sha256(
            canonical_json(
                {
                    key: value
                    for key, value in self.to_dict().items()
                    if key != "strict_reload_sha256"
                }
            )
        ).hexdigest()
        if self.strict_reload_sha256 != expected:
            raise ExperimentContractError("TQC strict-reload content hash differs")

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"persistence_authority", "_seal", "_issuer"}
        }


def _new_persistence_authority(
    completion: TQCTrainingCompletionAuthorityV2,
    *,
    work_directory: Path,
    model_state: Mapping[str, str],
    replay_state: Mapping[str, object],
    actor_state: str,
    model_artifact: PublishedArtifact,
    replay_artifact: PublishedArtifact,
    actor_artifact: PublishedArtifact,
    actor_manifest_artifact: PublishedArtifact,
    loaded_actor: LoadedTQCActor,
) -> TQCPersistenceAuthorityV2:
    payload: dict[str, object] = {
        "persistence_id": PERSISTENCE_ID,
        "attempt_id": completion.attempt_id,
        "execution_manifest_sha256": completion.execution_manifest_sha256,
        "claimed_work_directory_identity": completion.claimed_work_directory_identity,
        "work_directory": str(work_directory),
        "worker_pid": completion.worker_pid,
        "design_file_sha256": completion.design_file_sha256,
        "design_semantic_sha256": completion.design_semantic_sha256,
        "training_projection_sha256": completion.training_projection_sha256,
        "training_integrity_sha256": completion.training_integrity_sha256,
        "source_environment_steps": TOTAL_ENVIRONMENT_STEPS,
        "source_gradient_updates": EXPECTED_GRADIENT_UPDATES,
        "source_policy_state_sha256": model_state["policy_state_sha256"],
        "source_entropy_state_sha256": model_state["entropy_state_sha256"],
        "source_actor_optimizer_state_sha256": model_state["actor_optimizer_state_sha256"],
        "source_critic_optimizer_state_sha256": model_state["critic_optimizer_state_sha256"],
        "source_entropy_optimizer_state_sha256": model_state["entropy_optimizer_state_sha256"],
        "source_replay_arrays_sha256": replay_state["arrays_sha256"],
        "source_replay_metadata_sha256": replay_state["metadata_sha256"],
        "source_replay_array_payload_bytes": replay_state["array_payload_bytes"],
        "source_actor_state_sha256": actor_state,
        "model_artifact": _artifact_dict(model_artifact),
        "replay_artifact": _artifact_dict(replay_artifact),
        "actor_artifact": _artifact_dict(actor_artifact),
        "actor_manifest_artifact": _artifact_dict(actor_manifest_artifact),
        "all_artifacts_persisted": True,
        "behavioral_evidence": False,
        "claim_boundary": PERSISTENCE_CLAIM,
    }
    persistence_sha256 = hashlib.sha256(canonical_json(payload)).hexdigest()
    sealed_payload = {**payload, "persistence_sha256": persistence_sha256}
    seal = _TQCPersistenceSealV2(
        _payload=canonical_json(sealed_payload),
        _creator_pid=os.getpid(),
        _completion_identity=id(completion),
        _loaded_actor_identity=id(loaded_actor),
        _phase="issued",
        _issuer=_PERSISTENCE_SEAL_ISSUER,
    )
    return TQCPersistenceAuthorityV2(
        persistence_id=PERSISTENCE_ID,
        attempt_id=completion.attempt_id,
        execution_manifest_sha256=completion.execution_manifest_sha256,
        claimed_work_directory_identity=completion.claimed_work_directory_identity,
        work_directory=work_directory,
        worker_pid=completion.worker_pid,
        design_file_sha256=completion.design_file_sha256,
        design_semantic_sha256=completion.design_semantic_sha256,
        training_projection_sha256=completion.training_projection_sha256,
        training_integrity_sha256=completion.training_integrity_sha256,
        source_environment_steps=TOTAL_ENVIRONMENT_STEPS,
        source_gradient_updates=EXPECTED_GRADIENT_UPDATES,
        source_policy_state_sha256=model_state["policy_state_sha256"],
        source_entropy_state_sha256=model_state["entropy_state_sha256"],
        source_actor_optimizer_state_sha256=model_state["actor_optimizer_state_sha256"],
        source_critic_optimizer_state_sha256=model_state["critic_optimizer_state_sha256"],
        source_entropy_optimizer_state_sha256=model_state["entropy_optimizer_state_sha256"],
        source_replay_arrays_sha256=str(replay_state["arrays_sha256"]),
        source_replay_metadata_sha256=str(replay_state["metadata_sha256"]),
        source_replay_array_payload_bytes=int(replay_state["array_payload_bytes"]),
        source_actor_state_sha256=actor_state,
        model_artifact=model_artifact,
        replay_artifact=replay_artifact,
        actor_artifact=actor_artifact,
        actor_manifest_artifact=actor_manifest_artifact,
        persistence_sha256=persistence_sha256,
        all_artifacts_persisted=True,
        behavioral_evidence=False,
        claim_boundary=PERSISTENCE_CLAIM,
        loaded_actor=loaded_actor,
        training_completion=completion,
        _seal=seal,
        _issuer=_PERSISTENCE_ISSUER,
    )


def persist_tqc_training_completion_v2(
    completion: TQCTrainingCompletionAuthorityV2,
    *,
    work_directory: Path,
) -> TQCPersistenceAuthorityV2:
    """Persist the exact final model, replay, and code-free actor once."""

    if type(completion) is not TQCTrainingCompletionAuthorityV2:
        raise ExperimentContractError("persistence requires exact TQC training completion")
    model = revalidate_tqc_training_completion_v2(completion)
    projection = _projection(completion)
    _validate_persistence_projection(projection)
    directory = _validate_work_directory(completion, Path(work_directory))
    monitor = completion.resource_monitor
    reservations: list[ReservedStreamingArtifact | ReservedJsonArtifact] = []
    try:
        monitor.sample_post_training_disk_gate("pre_persistence")
        model_reservation = reserve_streaming_artifact(
            directory / MODEL_FILENAME,
            max_bytes=MAX_MODEL_BYTES,
        )
        reservations.append(model_reservation)
        replay_reservation = reserve_streaming_artifact(
            directory / REPLAY_FILENAME,
            max_bytes=MAX_REPLAY_BYTES,
        )
        reservations.append(replay_reservation)
        actor_reservation = reserve_streaming_artifact(
            directory / ACTOR_FILENAME,
            max_bytes=MAX_ACTOR_BYTES,
        )
        reservations.append(actor_reservation)
        actor_manifest_reservation = reserve_json_artifact(
            directory / ACTOR_MANIFEST_FILENAME,
            {
                "attempt_id": completion.attempt_id,
                "behavioral_evidence": False,
                "claim_boundary": "incomplete_actor_export_not_admissible/v1",
                "status": "IN_PROGRESS",
            },
        )
        reservations.append(actor_manifest_reservation)
        for reservation in reservations:
            _assert_reservation_parent(reservation, monitor)

        monitor.sample_lifecycle("pre_model_save")
        source_model_state = _model_state_receipt(model)
        source_replay_state = _replay_state_receipt(model, projection)
        source_actor_arrays = _actor_arrays(model)
        source_actor_state = actor_state_sha256(source_actor_arrays)
        model.save(model_reservation.writer)
        model_artifact = model_reservation.finalize()
        monitor.sample_lifecycle("post_model_save")
        if _model_state_receipt(model) != source_model_state:
            raise ExperimentContractError("TQC model changed during model persistence")

        monitor.sample_lifecycle("pre_replay_save")
        model.save_replay_buffer(replay_reservation.writer)
        replay_artifact = replay_reservation.finalize()
        monitor.sample_lifecycle("post_replay_save")
        if _replay_state_receipt(model, projection) != source_replay_state:
            raise ExperimentContractError("TQC replay changed during replay persistence")

        monitor.sample_lifecycle("pre_actor_export")
        actor_reservation.writer.write(encode_actor_npz(source_actor_arrays))
        actor_artifact = actor_reservation.finalize()
        loaded_actor = load_actor_npz(
            actor_artifact.path,
            expected_sha256=actor_artifact.sha256,
        )
        if (
            loaded_actor.state_sha256 != source_actor_state
            or actor_state_sha256(_actor_arrays(model)) != source_actor_state
        ):
            raise ExperimentContractError("TQC actor changed during strict export")
        actor_manifest = {
            "schema_version": 1,
            "manifest_id": "tqc_dev_1m_v2_strict_actor_export/v1",
            "attempt_id": completion.attempt_id,
            "execution_manifest_sha256": completion.execution_manifest_sha256,
            "claimed_work_directory_identity": completion.claimed_work_directory_identity,
            "training_integrity_sha256": completion.training_integrity_sha256,
            "source_environment_steps": TOTAL_ENVIRONMENT_STEPS,
            "source_gradient_updates": EXPECTED_GRADIENT_UPDATES,
            "model_artifact": _artifact_dict(model_artifact),
            "replay_artifact": _artifact_dict(replay_artifact),
            "actor_artifact": _artifact_dict(actor_artifact),
            "actor_state_sha256": source_actor_state,
            "actor_schema_sha256": actor_schema_sha256(),
            "contains_executable_code": False,
            "publication_allowed": False,
            "behavioral_evidence": False,
            "claim_boundary": PERSISTENCE_CLAIM,
        }
        if len(finite_pretty_json(actor_manifest)) > MAX_ACTOR_MANIFEST_BYTES:
            raise ExperimentContractError("TQC actor manifest exceeds its byte limit")
        actor_manifest_artifact = actor_manifest_reservation.finalize(actor_manifest)
        if actor_manifest_artifact.byte_count > MAX_ACTOR_MANIFEST_BYTES:
            raise ExperimentContractError("TQC actor manifest exceeds its byte limit")
        monitor.sample_lifecycle("post_actor_export")
        if revalidate_tqc_training_completion_v2(completion) is not model:
            raise ExperimentContractError(
                "TQC completion model identity changed during persistence"
            )
        if (
            _model_state_receipt(model) != source_model_state
            or _replay_state_receipt(model, projection) != source_replay_state
            or actor_state_sha256(_actor_arrays(model)) != source_actor_state
        ):
            raise ExperimentContractError("TQC source state changed before persistence issuance")
        authority = _new_persistence_authority(
            completion,
            work_directory=directory,
            model_state=source_model_state,
            replay_state=source_replay_state,
            actor_state=source_actor_state,
            model_artifact=model_artifact,
            replay_artifact=replay_artifact,
            actor_artifact=actor_artifact,
            actor_manifest_artifact=actor_manifest_artifact,
            loaded_actor=loaded_actor,
        )
        return authority
    except BaseException as exc:
        monitor._poison(exc)
        if not isinstance(exc, Exception):
            raise
        if isinstance(exc, ExperimentContractError):
            raise
        raise ExperimentContractError(f"TQC v2 persistence failed: {exc}") from exc
    finally:
        for reservation in reservations:
            if isinstance(reservation, ReservedStreamingArtifact):
                reservation.abort()
            else:
                reservation.close()


def revalidate_tqc_persistence_authority_v2(
    authority: TQCPersistenceAuthorityV2,
) -> object:
    """Recheck the authority, live source state, and retained artifact metadata."""

    if type(authority) is not TQCPersistenceAuthorityV2:
        raise ExperimentContractError("strict reload requires exact persistence authority")
    authority._validate_payload_seal()
    _validate_work_directory(authority.training_completion, authority.work_directory)
    authority._validate_sealed()
    model = revalidate_tqc_training_completion_v2(authority.training_completion)
    projection = _projection(authority.training_completion)
    if _model_state_receipt(model) != {
        "policy_state_sha256": authority.source_policy_state_sha256,
        "entropy_state_sha256": authority.source_entropy_state_sha256,
        "actor_optimizer_state_sha256": authority.source_actor_optimizer_state_sha256,
        "critic_optimizer_state_sha256": authority.source_critic_optimizer_state_sha256,
        "entropy_optimizer_state_sha256": authority.source_entropy_optimizer_state_sha256,
    }:
        raise ExperimentContractError("TQC live model changed after persistence")
    replay = _replay_state_receipt(model, projection)
    if (
        replay["arrays_sha256"] != authority.source_replay_arrays_sha256
        or replay["metadata_sha256"] != authority.source_replay_metadata_sha256
        or replay["array_payload_bytes"] != authority.source_replay_array_payload_bytes
    ):
        raise ExperimentContractError("TQC live replay changed after persistence")
    if (
        authority.loaded_actor.content_sha256 != authority.actor_artifact.sha256
        or authority.loaded_actor.state_sha256 != authority.source_actor_state_sha256
    ):
        raise ExperimentContractError("TQC strict actor changed after persistence")
    return model


def _load_bound_model(artifact: PublishedArtifact) -> object:
    from sb3_contrib import TQC

    payload = read_verified_artifact_bytes(
        artifact.path,
        expected_sha256=artifact.sha256,
        expected_size=artifact.byte_count,
        max_bytes=MAX_MODEL_BYTES,
    )
    try:
        model = TQC.load(
            io.BytesIO(payload),
            env=None,
            device="cpu",
            print_system_info=False,
            force_reset=True,
        )
    finally:
        del payload
    if type(model) is not TQC:
        raise ExperimentContractError("strict-loaded TQC model structure differs")
    return model


def _new_strict_reload_authority(
    persistence: TQCPersistenceAuthorityV2,
    *,
    actor_equivalence_receipt_sha256: str,
    loaded_model_state: Mapping[str, str],
    loaded_replay_state: Mapping[str, object],
) -> TQCStrictReloadAuthorityV2:
    payload: dict[str, object] = {
        "strict_reload_id": STRICT_RELOAD_ID,
        "attempt_id": persistence.attempt_id,
        "execution_manifest_sha256": persistence.execution_manifest_sha256,
        "claimed_work_directory_identity": persistence.claimed_work_directory_identity,
        "training_integrity_sha256": persistence.training_integrity_sha256,
        "persistence_sha256": persistence.persistence_sha256,
        "actor_equivalence_receipt_sha256": actor_equivalence_receipt_sha256,
        "source_environment_steps": persistence.source_environment_steps,
        "loaded_environment_steps": TOTAL_ENVIRONMENT_STEPS,
        "source_gradient_updates": persistence.source_gradient_updates,
        "loaded_gradient_updates": EXPECTED_GRADIENT_UPDATES,
        "source_policy_state_sha256": persistence.source_policy_state_sha256,
        "loaded_policy_state_sha256": loaded_model_state["policy_state_sha256"],
        "source_entropy_state_sha256": persistence.source_entropy_state_sha256,
        "loaded_entropy_state_sha256": loaded_model_state["entropy_state_sha256"],
        "source_actor_optimizer_state_sha256": (persistence.source_actor_optimizer_state_sha256),
        "loaded_actor_optimizer_state_sha256": loaded_model_state["actor_optimizer_state_sha256"],
        "source_critic_optimizer_state_sha256": (persistence.source_critic_optimizer_state_sha256),
        "loaded_critic_optimizer_state_sha256": loaded_model_state["critic_optimizer_state_sha256"],
        "source_entropy_optimizer_state_sha256": (
            persistence.source_entropy_optimizer_state_sha256
        ),
        "loaded_entropy_optimizer_state_sha256": loaded_model_state[
            "entropy_optimizer_state_sha256"
        ],
        "source_replay_arrays_sha256": persistence.source_replay_arrays_sha256,
        "loaded_replay_arrays_sha256": loaded_replay_state["arrays_sha256"],
        "source_replay_metadata_sha256": persistence.source_replay_metadata_sha256,
        "loaded_replay_metadata_sha256": loaded_replay_state["metadata_sha256"],
        "model_artifact_sha256": persistence.model_artifact.sha256,
        "replay_artifact_sha256": persistence.replay_artifact.sha256,
        "original_model_released": True,
        "original_replay_released": True,
        "reload_model_released": True,
        "reload_replay_released": True,
        "simultaneous_full_replay_buffers_allowed": 1,
        "passed": True,
        "behavioral_evidence": False,
        "continuation_claim": "operational_continuation_only_not_bitwise_equivalent_resume",
        "claim_boundary": STRICT_RELOAD_CLAIM,
    }
    strict_reload_sha256 = hashlib.sha256(canonical_json(payload)).hexdigest()
    seal = _TQCStrictReloadSealV2(
        _payload=canonical_json({**payload, "strict_reload_sha256": strict_reload_sha256}),
        _creator_pid=os.getpid(),
        _persistence_identity=id(persistence),
        _issuer=_STRICT_RELOAD_SEAL_ISSUER,
    )
    return TQCStrictReloadAuthorityV2(
        **payload,
        strict_reload_sha256=strict_reload_sha256,
        persistence_authority=persistence,
        _seal=seal,
        _issuer=_STRICT_RELOAD_ISSUER,
    )


def _validate_actor_equivalence_receipt(
    persistence: TQCPersistenceAuthorityV2,
    receipt: object,
) -> str:
    from .tqc_actor_equivalence_v2 import (
        TQCActorEquivalenceReceipt,
        revalidate_tqc_actor_equivalence_receipt_v2,
    )

    if type(receipt) is not TQCActorEquivalenceReceipt:
        raise ExperimentContractError(
            "strict reload requires exact actor equivalence before source release"
        )
    receipt = revalidate_tqc_actor_equivalence_receipt_v2(receipt)
    if (
        receipt.actor_authority.persistence_authority is not persistence
        or receipt.attempt_id != persistence.attempt_id
        or receipt.execution_manifest_sha256 != persistence.execution_manifest_sha256
        or receipt.claimed_work_directory_identity != persistence.claimed_work_directory_identity
        or receipt.design_file_sha256 != persistence.design_file_sha256
        or receipt.design_semantic_sha256 != persistence.design_semantic_sha256
        or receipt.training_projection_sha256 != persistence.training_projection_sha256
        or receipt.training_integrity_sha256 != persistence.training_integrity_sha256
        or receipt.persistence_sha256 != persistence.persistence_sha256
        or receipt.trusted_model_artifact_sha256 != persistence.model_artifact.sha256
        or receipt.trusted_model_artifact_byte_count != persistence.model_artifact.byte_count
        or receipt.source_environment_step != TOTAL_ENVIRONMENT_STEPS
        or receipt.source_update_count != EXPECTED_GRADIENT_UPDATES
        or receipt.trusted_actor_state_sha256 != persistence.source_actor_state_sha256
        or receipt.loaded_actor_state_sha256 != persistence.loaded_actor.state_sha256
        or receipt.passed is not True
    ):
        raise ExperimentContractError("actor equivalence receipt persistence binding differs")
    return hashlib.sha256(canonical_json(receipt.to_dict())).hexdigest()


def strict_reload_tqc_persistence_v2(
    persistence: TQCPersistenceAuthorityV2,
    *,
    actor_equivalence_receipt: object,
) -> TQCStrictReloadAuthorityV2:
    """Release the source replay, then strictly reload and compare exact state."""

    if type(persistence) is not TQCPersistenceAuthorityV2:
        raise ExperimentContractError("strict reload requires exact persistence authority")
    monitor = persistence.training_completion.resource_monitor
    loaded: object | None = None
    reload_begun = False
    try:
        revalidate_tqc_persistence_authority_v2(persistence)
        equivalence_sha256 = _validate_actor_equivalence_receipt(
            persistence,
            actor_equivalence_receipt,
        )
        persistence._seal.begin_reload(
            persistence.to_dict(),
            completion=persistence.training_completion,
            loaded_actor=persistence.loaded_actor,
        )
        reload_begun = True
        original = _consume_tqc_training_model_v2(persistence.training_completion)
        environment = original.get_env()
        original.replay_buffer = None
        if environment is not None:
            environment.close()
        original.env = None
        if hasattr(original, "_vec_normalize_env"):
            original._vec_normalize_env = None
        del original
        del environment
        gc.collect()

        loaded = _load_bound_model(persistence.model_artifact)
        loaded_model_state = _model_state_receipt(loaded)
        if loaded_model_state != {
            "policy_state_sha256": persistence.source_policy_state_sha256,
            "entropy_state_sha256": persistence.source_entropy_state_sha256,
            "actor_optimizer_state_sha256": (persistence.source_actor_optimizer_state_sha256),
            "critic_optimizer_state_sha256": (persistence.source_critic_optimizer_state_sha256),
            "entropy_optimizer_state_sha256": (persistence.source_entropy_optimizer_state_sha256),
        }:
            raise ExperimentContractError("strict-loaded TQC model state differs")
        empty_replay_state = _replay_state_receipt(
            loaded,
            _projection(persistence.training_completion),
            final=False,
        )
        if empty_replay_state["array_payload_bytes"] != (
            persistence.source_replay_array_payload_bytes
        ):
            raise ExperimentContractError("strict-loaded TQC empty replay structure differs")
        loaded.replay_buffer = None
        gc.collect()
        with verified_artifact_reader(
            persistence.replay_artifact.path,
            expected_sha256=persistence.replay_artifact.sha256,
            expected_size=persistence.replay_artifact.byte_count,
            max_bytes=MAX_REPLAY_BYTES,
        ) as reader:
            loaded.load_replay_buffer(reader, truncate_last_traj=False)
        loaded_replay_state = _replay_state_receipt(
            loaded,
            _projection(persistence.training_completion),
        )
        if (
            loaded_replay_state["arrays_sha256"] != persistence.source_replay_arrays_sha256
            or loaded_replay_state["metadata_sha256"] != persistence.source_replay_metadata_sha256
            or loaded_replay_state["array_payload_bytes"]
            != persistence.source_replay_array_payload_bytes
        ):
            raise ExperimentContractError("strict-loaded TQC replay state differs")
        loaded.replay_buffer = None
        loaded.env = None
        if hasattr(loaded, "_vec_normalize_env"):
            loaded._vec_normalize_env = None
        del loaded
        loaded = None
        gc.collect()
        persistence._seal.finish_reload()
        authority = _new_strict_reload_authority(
            persistence,
            actor_equivalence_receipt_sha256=equivalence_sha256,
            loaded_model_state=loaded_model_state,
            loaded_replay_state=loaded_replay_state,
        )
        return authority
    except BaseException as exc:
        if reload_begun:
            persistence._seal.fail_reload()
        monitor._poison(exc)
        if loaded is not None:
            with np.errstate(all="ignore"):
                if hasattr(loaded, "replay_buffer"):
                    loaded.replay_buffer = None
        gc.collect()
        if not isinstance(exc, Exception):
            raise
        if isinstance(exc, ExperimentContractError):
            raise
        raise ExperimentContractError(f"TQC v2 strict reload failed: {exc}") from exc


def revalidate_tqc_strict_reload_authority_v2(
    authority: TQCStrictReloadAuthorityV2,
) -> TQCStrictReloadAuthorityV2:
    """Revalidate the sealed reload result without access to its private issuer."""

    if type(authority) is not TQCStrictReloadAuthorityV2:
        raise ExperimentContractError("strict-reload result must be the exact authority")
    authority._validate_sealed()
    return authority


__all__ = [
    "ACTOR_FILENAME",
    "ACTOR_MANIFEST_FILENAME",
    "MODEL_FILENAME",
    "PERSISTENCE_ID",
    "REPLAY_FILENAME",
    "STRICT_RELOAD_ID",
    "TQCPersistenceAuthorityV2",
    "TQCStrictReloadAuthorityV2",
    "persist_tqc_training_completion_v2",
    "revalidate_tqc_persistence_authority_v2",
    "revalidate_tqc_strict_reload_authority_v2",
    "strict_reload_tqc_persistence_v2",
]
