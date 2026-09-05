"""Pure full-clip replay and reference-corpus contracts."""

from __future__ import annotations

import io
import math
import struct
from collections.abc import Mapping
from typing import Any
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import (
    CONTROL_PERIOD_SECONDS,
    REFERENCE_HORIZON,
    REFERENCE_SCHEMA_ID,
    REFERENCE_WIDTH,
    ArrayBindingV2,
    ArtifactBindingV2,
    ReferenceIdentityV2,
    ReferenceIdentityV2Error,
    array_sha256,
    canonical_json_bytes,
    require_sha256,
    sha256_json,
)
from oracle_composition.envs.reference_corpus import (
    EXPECTED_REFERENCE_WRAPPER_TYPES,
    FULL_CONTACT_CAPTURE_ID,
)
from oracle_composition.tracking.humanoid_reference import (
    HUMANOID_ACTUATOR_JOINT_ORDER,
    HUMANOID_QPOS_INDICES_BY_ACTUATOR,
    HUMANOID_QVEL_INDICES_BY_ACTUATOR,
    HUMANOID_REFERENCE_SCHEMA,
)

CLIP_PAYLOAD_ID = "humanoid_full_clip_arrays_canonical_npz/v1"
RUNTIME_ID = "humanoid_reference_corpus_runtime/v1"
WRAPPER_STATE_ID = "gymnasium_exact_reference_corpus_wrapper_state/v1"
REFERENCE_CONSTRUCTION_ID = "humanoid_boundary_state_to_45d_sign_continuous/v1"
CONTACT_SEQUENCE_ID = "five_substeps_ordered_geom_name_force6/v1"
POLICY_INPUT_ID = "raw_observation_f32_only_no_branch_metadata/v1"
STATE_SPEC = "mjtState.mjSTATE_INTEGRATION"
STATE_FLAG = 16_383
STATE_SIZE = 195
STATE_DTYPE = np.dtype("<f8")
OBSERVATION_WIDTH = 348
ACTION_WIDTH = 17
CFRCE_SHAPE = (13, 6)
QPOS_SLICE = slice(1, 25)
QVEL_SLICE = slice(25, 48)
GEOM_NAME_BYTES = 32
HASH_BYTES = 64
MAX_CLIP_PAYLOAD_BYTES = 128 * 1024 * 1024
WRAPPER_FLAG_NAMES = (
    "order_enforcing_has_reset",
    "passive_checked_reset",
    "passive_checked_step",
    "passive_checked_render",
    "passive_close_called",
)
EXPECTED_STATE_COMPONENT_SIZES = (
    ("time", 1),
    ("qpos", 24),
    ("qvel", 23),
    ("act", 0),
    ("warmstart", 23),
    ("ctrl", 17),
    ("qfrc_applied", 23),
    ("xfrc_applied", 84),
    ("eq_active", 0),
    ("mocap_pos", 0),
    ("mocap_quat", 0),
    ("userdata", 0),
    ("plugin", 0),
)


class ReferenceCorpusContractError(ValueError):
    """A corpus payload, runtime, or replay bundle violates the frozen design."""


def _exact_array(
    value: object,
    *,
    name: str,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
    finite: bool = True,
) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise ReferenceCorpusContractError(f"{name} must be a NumPy array")
    if value.dtype != dtype or value.dtype.str != dtype.str:
        raise ReferenceCorpusContractError(f"{name} dtype differs")
    if value.shape != shape:
        raise ReferenceCorpusContractError(f"{name} shape differs")
    if not value.flags.c_contiguous:
        raise ReferenceCorpusContractError(f"{name} must use C-order storage")
    if finite and value.dtype.kind in {"f", "c"} and not np.isfinite(value).all():
        raise ReferenceCorpusContractError(f"{name} contains a non-finite value")
    return value


def _canonical_quaternion(
    value: np.ndarray,
    *,
    previous: np.ndarray | None,
) -> np.ndarray:
    quaternion = np.asarray(value, dtype="<f8").copy()
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ReferenceCorpusContractError("root quaternion is invalid")
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ReferenceCorpusContractError("root quaternion has zero or non-finite norm")
    quaternion /= norm
    if previous is None:
        first = next((float(component) for component in quaternion if abs(component) > 1e-15), 0.0)
        if first < 0.0:
            quaternion *= -1.0
    elif float(np.dot(previous, quaternion)) < 0.0:
        quaternion *= -1.0
    return np.ascontiguousarray(quaternion, dtype="<f8")


def derive_reference_rows(
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    joint_order: tuple[str, ...] = HUMANOID_ACTUATOR_JOINT_ORDER,
    velocity_frame: str = "torso_local",
    cadence_seconds: float = CONTROL_PERIOD_SECONDS,
) -> np.ndarray:
    """Derive the exact 45-D ABI rows with continuous quaternion signs."""

    if joint_order != HUMANOID_ACTUATOR_JOINT_ORDER:
        raise ReferenceCorpusContractError("reference joint order differs from the frozen ABI")
    if velocity_frame != "torso_local":
        raise ReferenceCorpusContractError("root angular velocity frame differs")
    if type(cadence_seconds) is not float or cadence_seconds != CONTROL_PERIOD_SECONDS:
        raise ReferenceCorpusContractError("reference cadence differs")
    positions = np.asarray(qpos)
    velocities = np.asarray(qvel)
    if (
        positions.dtype != np.dtype("<f8")
        or velocities.dtype != np.dtype("<f8")
        or positions.ndim != 2
        or velocities.ndim != 2
        or positions.shape[1:] != (24,)
        or velocities.shape != (positions.shape[0], 23)
        or not positions.flags.c_contiguous
        or not velocities.flags.c_contiguous
        or not np.isfinite(positions).all()
        or not np.isfinite(velocities).all()
    ):
        raise ReferenceCorpusContractError("qpos/qvel arrays differ from the Humanoid ABI")
    rows = np.empty((positions.shape[0], REFERENCE_WIDTH), dtype="<f8")
    previous: np.ndarray | None = None
    for index in range(positions.shape[0]):
        quaternion = _canonical_quaternion(positions[index, 3:7], previous=previous)
        previous = quaternion
        rows[index, 0] = positions[index, 2]
        rows[index, 1:5] = quaternion
        rows[index, 5:8] = velocities[index, 0:3]
        rows[index, 8:11] = velocities[index, 3:6]
        rows[index, 11:28] = positions[index, HUMANOID_QPOS_INDICES_BY_ACTUATOR]
        rows[index, 28:45] = velocities[index, HUMANOID_QVEL_INDICES_BY_ACTUATOR]
    return rows


def derive_reference_row(
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    previous_quaternion: np.ndarray | None,
) -> np.ndarray:
    positions = np.ascontiguousarray(np.asarray(qpos, dtype="<f8").reshape(1, 24))
    velocities = np.ascontiguousarray(np.asarray(qvel, dtype="<f8").reshape(1, 23))
    row = derive_reference_rows(positions, velocities)
    if previous_quaternion is not None:
        row[0, 1:5] = _canonical_quaternion(
            positions[0, 3:7],
            previous=np.asarray(previous_quaternion, dtype="<f8"),
        )
    return np.ascontiguousarray(row[0], dtype="<f8")


def reference_window_index_arrays(boundaries: int) -> tuple[np.ndarray, np.ndarray]:
    if type(boundaries) is not int or boundaries < 2:
        raise ReferenceCorpusContractError("reference needs at least two boundaries")
    final_index = boundaries - 1
    indices = np.empty((boundaries, REFERENCE_HORIZON), dtype="<i8")
    terminal_hold = np.empty((boundaries, REFERENCE_HORIZON), dtype="|u1")
    for boundary in range(boundaries):
        for offset in range(REFERENCE_HORIZON):
            requested = boundary + offset
            indices[boundary, offset] = min(requested, final_index)
            terminal_hold[boundary, offset] = int(requested > final_index)
    return indices, terminal_hold


def validate_reference_window_indices(
    indices: np.ndarray,
    terminal_hold: np.ndarray,
    *,
    boundaries: int,
) -> None:
    expected_indices, expected_hold = reference_window_index_arrays(boundaries)
    _exact_array(
        indices,
        name="reference_window_indices",
        dtype=np.dtype("<i8"),
        shape=expected_indices.shape,
    )
    _exact_array(
        terminal_hold,
        name="reference_window_terminal_hold",
        dtype=np.dtype("|u1"),
        shape=expected_hold.shape,
    )
    if not np.array_equal(indices, expected_indices):
        raise ReferenceCorpusContractError("reference windows crop, wrap, or skip rows")
    if not np.array_equal(terminal_hold, expected_hold):
        raise ReferenceCorpusContractError("terminal holding begins before the declared end")


def _array_schema(
    steps: int, contacts: int, *, screen_canary: bool
) -> dict[str, tuple[np.dtype[Any], tuple[int, ...]]]:
    boundaries = steps + 1
    schema: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {
        "boundary_integration_state": (np.dtype("<f8"), (boundaries, STATE_SIZE)),
        "boundary_observation": (np.dtype("<f8"), (boundaries, OBSERVATION_WIDTH)),
        "boundary_cfrc_ext": (np.dtype("<f8"), (boundaries, *CFRCE_SHAPE)),
        "boundary_root_xy": (np.dtype("<f8"), (boundaries, 2)),
        "boundary_simulation_time": (np.dtype("<f8"), (boundaries,)),
        "boundary_wrapper_elapsed": (np.dtype("<i8"), (boundaries,)),
        "boundary_wrapper_flags": (np.dtype("|u1"), (boundaries, len(WRAPPER_FLAG_NAMES))),
        "boundary_result_flags": (np.dtype("|u1"), (boundaries, 2)),
        "boundary_torso_up_z": (np.dtype("<f8"), (boundaries,)),
        "reference_rows": (np.dtype("<f8"), (boundaries, REFERENCE_WIDTH)),
        "reference_window_indices": (np.dtype("<i8"), (boundaries, REFERENCE_HORIZON)),
        "reference_window_terminal_hold": (np.dtype("|u1"), (boundaries, REFERENCE_HORIZON)),
        "boundary_integration_sha256": (np.dtype("|S64"), (boundaries,)),
        "boundary_observation_sha256": (np.dtype("|S64"), (boundaries,)),
        "boundary_cfrc_ext_sha256": (np.dtype("|S64"), (boundaries,)),
        "boundary_root_xy_sha256": (np.dtype("|S64"), (boundaries,)),
        "boundary_reference_sha256": (np.dtype("|S64"), (boundaries,)),
        "boundary_rng_state_sha256": (np.dtype("|S64"), (boundaries,)),
        "boundary_record_sha256": (np.dtype("|S64"), (boundaries,)),
        "transition_normalized_action": (np.dtype("<f4"), (steps, ACTION_WIDTH)),
        "transition_physical_action": (np.dtype("<f4"), (steps, ACTION_WIDTH)),
        "transition_returned_observation": (np.dtype("<f8"), (steps, OBSERVATION_WIDTH)),
        "transition_reward": (np.dtype("<f8"), (steps,)),
        "transition_flags": (np.dtype("|u1"), (steps, 2)),
        "transition_next_wrapper_elapsed": (np.dtype("<i8"), (steps,)),
        "transition_next_wrapper_flags": (np.dtype("|u1"), (steps, len(WRAPPER_FLAG_NAMES))),
        "transition_actor_input_sha256": (np.dtype("|S64"), (steps,)),
        "transition_actor_output_sha256": (np.dtype("|S64"), (steps,)),
        "transition_physical_action_sha256": (np.dtype("|S64"), (steps,)),
        "transition_returned_observation_sha256": (np.dtype("|S64"), (steps,)),
        "transition_contact_sequence_sha256": (np.dtype("|S64"), (steps,)),
        "transition_record_sha256": (np.dtype("|S64"), (steps,)),
        "transition_contact_substeps": (np.dtype("<i8"), (steps,)),
        "transition_nonfoot_floor_contact": (np.dtype("|u1"), (steps,)),
        "contact_offsets": (np.dtype("<i8"), (steps + 1,)),
        "contact_substep": (np.dtype("<i8"), (contacts,)),
        "contact_index_within_substep": (np.dtype("<i8"), (contacts,)),
        "contact_geom1_id": (np.dtype("<i8"), (contacts,)),
        "contact_geom2_id": (np.dtype("<i8"), (contacts,)),
        "contact_geom1_name": (np.dtype(f"|S{GEOM_NAME_BYTES}"), (contacts,)),
        "contact_geom2_name": (np.dtype(f"|S{GEOM_NAME_BYTES}"), (contacts,)),
        "contact_force_torque": (np.dtype("<f8"), (contacts, 6)),
    }
    if screen_canary:
        schema["canary_plain_reward"] = (np.dtype("<f8"), (steps,))
        schema["canary_plain_returned_observation_sha256"] = (
            np.dtype("|S64"),
            (steps,),
        )
    return schema


def _hash_ascii(values: np.ndarray, *, name: str) -> None:
    for raw in values:
        try:
            require_sha256(bytes(raw).decode("ascii"), field=name)
        except (UnicodeError, ReferenceIdentityV2Error) as exc:
            raise ReferenceCorpusContractError(f"{name} contains an invalid SHA-256") from exc


def contact_sequence_sha256(arrays: Mapping[str, np.ndarray], transition_index: int) -> str:
    offsets = arrays["contact_offsets"]
    start = int(offsets[transition_index])
    stop = int(offsets[transition_index + 1])
    payload = bytearray()
    payload.extend(struct.pack(">q", int(arrays["transition_contact_substeps"][transition_index])))
    names = (
        "contact_substep",
        "contact_index_within_substep",
        "contact_geom1_id",
        "contact_geom2_id",
    )
    for index in range(start, stop):
        for name in names:
            payload.extend(struct.pack(">q", int(arrays[name][index])))
        for name in ("contact_geom1_name", "contact_geom2_name"):
            encoded = bytes(arrays[name][index]).rstrip(b"\0")
            payload.extend(len(encoded).to_bytes(8, "big"))
            payload.extend(encoded)
        payload.extend(np.asarray(arrays["contact_force_torque"][index], dtype=">f8").tobytes())
    import hashlib

    return hashlib.sha256(bytes(payload)).hexdigest()


def boundary_record_sha256(arrays: Mapping[str, np.ndarray], index: int) -> str:
    return sha256_json(
        {
            "index": index,
            "integration_state_sha256": array_sha256(arrays["boundary_integration_state"][index]),
            "observation_sha256": array_sha256(arrays["boundary_observation"][index]),
            "cfrc_ext_sha256": array_sha256(arrays["boundary_cfrc_ext"][index]),
            "root_xy_sha256": array_sha256(arrays["boundary_root_xy"][index]),
            "simulation_time_ieee754": struct.pack(
                ">d", float(arrays["boundary_simulation_time"][index])
            ).hex(),
            "wrapper_elapsed": int(arrays["boundary_wrapper_elapsed"][index]),
            "wrapper_flags": [int(value) for value in arrays["boundary_wrapper_flags"][index]],
            "result_flags": [int(value) for value in arrays["boundary_result_flags"][index]],
            "torso_up_z_ieee754": struct.pack(
                ">d", float(arrays["boundary_torso_up_z"][index])
            ).hex(),
            "reference_sha256": array_sha256(arrays["reference_rows"][index]),
            "rng_state_sha256": bytes(arrays["boundary_rng_state_sha256"][index]).decode(),
            "window_indices_sha256": array_sha256(arrays["reference_window_indices"][index]),
            "window_terminal_hold_sha256": array_sha256(
                arrays["reference_window_terminal_hold"][index]
            ),
        }
    )


def transition_record_sha256(arrays: Mapping[str, np.ndarray], index: int) -> str:
    return sha256_json(
        {
            "index": index,
            "actor_input_sha256": bytes(arrays["transition_actor_input_sha256"][index]).decode(),
            "actor_output_sha256": bytes(arrays["transition_actor_output_sha256"][index]).decode(),
            "physical_action_sha256": bytes(
                arrays["transition_physical_action_sha256"][index]
            ).decode(),
            "returned_observation_sha256": bytes(
                arrays["transition_returned_observation_sha256"][index]
            ).decode(),
            "reward_ieee754": struct.pack(">d", float(arrays["transition_reward"][index])).hex(),
            "flags": [int(value) for value in arrays["transition_flags"][index]],
            "next_wrapper_elapsed": int(arrays["transition_next_wrapper_elapsed"][index]),
            "next_wrapper_flags": [
                int(value) for value in arrays["transition_next_wrapper_flags"][index]
            ],
            "contact_sequence_sha256": contact_sequence_sha256(arrays, index),
            "nonfoot_floor_contact": bool(arrays["transition_nonfoot_floor_contact"][index]),
        }
    )


def validate_clip_arrays(
    values: Mapping[str, np.ndarray],
    *,
    steps: int,
    screen_canary: bool,
) -> dict[str, np.ndarray]:
    """Validate every serialized array and all cross-array invariants."""

    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ReferenceCorpusContractError("steps must be in [1, 1000]")
    if type(screen_canary) is not bool:
        raise ReferenceCorpusContractError("screen_canary must be boolean")
    if type(values) is not dict:
        raise ReferenceCorpusContractError("clip arrays must be an exact mapping")
    offsets = values.get("contact_offsets")
    if not isinstance(offsets, np.ndarray) or offsets.shape != (steps + 1,):
        raise ReferenceCorpusContractError("contact offsets are missing or malformed")
    if offsets.dtype != np.dtype("<i8") or offsets.dtype.str != "<i8":
        raise ReferenceCorpusContractError("contact offset dtype differs")
    if offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ReferenceCorpusContractError("contact offsets must be contiguous and monotone")
    contacts = int(offsets[-1])
    schema = _array_schema(steps, contacts, screen_canary=screen_canary)
    if set(values) != set(schema):
        raise ReferenceCorpusContractError("clip array names differ from the exact schema")
    result: dict[str, np.ndarray] = {}
    for name, (dtype, shape) in schema.items():
        value = _exact_array(values[name], name=name, dtype=dtype, shape=shape)
        result[name] = value
    boolean_arrays = (
        "boundary_wrapper_flags",
        "boundary_result_flags",
        "reference_window_terminal_hold",
        "transition_flags",
        "transition_next_wrapper_flags",
        "transition_nonfoot_floor_contact",
    )
    if any(np.any(result[name] > 1) for name in boolean_arrays):
        raise ReferenceCorpusContractError("boolean byte arrays must contain only zero or one")
    if not np.array_equal(result["boundary_wrapper_elapsed"], np.arange(steps + 1)):
        raise ReferenceCorpusContractError("wrapper counters are missing or noncontiguous")
    if not np.array_equal(result["transition_next_wrapper_elapsed"], np.arange(1, steps + 1)):
        raise ReferenceCorpusContractError("transition wrapper counters differ")
    if not np.array_equal(
        result["transition_next_wrapper_flags"], result["boundary_wrapper_flags"][1:]
    ):
        raise ReferenceCorpusContractError("transition next-wrapper state differs")
    if not np.array_equal(result["transition_flags"], result["boundary_result_flags"][1:]):
        raise ReferenceCorpusContractError("transition flags differ from next boundaries")
    expected_wrapper_flags = np.zeros((steps + 1, len(WRAPPER_FLAG_NAMES)), dtype="|u1")
    expected_wrapper_flags[:, 0:2] = 1
    expected_wrapper_flags[1:, 2] = 1
    if not np.array_equal(result["boundary_wrapper_flags"], expected_wrapper_flags):
        raise ReferenceCorpusContractError("wrapper flags differ from the exact reset/step order")
    expected_result_flags = np.zeros((steps + 1, 2), dtype="|u1")
    if steps == 1000:
        expected_result_flags[-1, 1] = 1
    if not np.array_equal(result["boundary_result_flags"], expected_result_flags):
        raise ReferenceCorpusContractError("result flags differ from the declared horizon")
    times = result["boundary_simulation_time"]
    expected_times = times[0] + np.arange(steps + 1, dtype="<f8") * CONTROL_PERIOD_SECONDS
    if not np.allclose(times, expected_times, rtol=0.0, atol=1e-12):
        raise ReferenceCorpusContractError("simulation times differ from the 15 ms clock")
    if not np.array_equal(times, result["boundary_integration_state"][:, 0]):
        raise ReferenceCorpusContractError("simulation times differ from integration state")
    qpos = np.ascontiguousarray(result["boundary_integration_state"][:, QPOS_SLICE])
    qvel = np.ascontiguousarray(result["boundary_integration_state"][:, QVEL_SLICE])
    derived_rows = derive_reference_rows(qpos, qvel)
    if not np.array_equal(result["reference_rows"], derived_rows):
        raise ReferenceCorpusContractError(
            "reference row disagrees in quaternion sign, joint order, velocity frame, or cadence"
        )
    if not np.array_equal(result["boundary_root_xy"], qpos[:, 0:2]):
        raise ReferenceCorpusContractError("root x/y sidecar differs from qpos")
    if not np.array_equal(
        result["boundary_observation"][:, -78:].reshape(steps + 1, 13, 6),
        result["boundary_cfrc_ext"],
    ):
        raise ReferenceCorpusContractError("boundary force array differs from raw observation")
    validate_reference_window_indices(
        result["reference_window_indices"],
        result["reference_window_terminal_hold"],
        boundaries=steps + 1,
    )
    if not np.all(result["transition_contact_substeps"] == 5):
        raise ReferenceCorpusContractError("one or more transitions omitted a physics substep")
    if np.any(np.abs(result["transition_normalized_action"]) > 1.0):
        raise ReferenceCorpusContractError("normalized action lies outside [-1, 1]")
    if np.any(np.abs(result["transition_physical_action"]) > 0.4):
        raise ReferenceCorpusContractError("physical action lies outside Humanoid-v5 bounds")
    if np.any(result["contact_substep"] < 0) or np.any(result["contact_substep"] >= 5):
        raise ReferenceCorpusContractError("contact substep index is invalid")
    if np.any(result["contact_geom1_id"] < 0) or np.any(result["contact_geom2_id"] < 0):
        raise ReferenceCorpusContractError("contact geometry id is invalid")
    for name in ("contact_geom1_name", "contact_geom2_name"):
        for raw in result[name]:
            encoded = bytes(raw).rstrip(b"\0")
            try:
                text = encoded.decode("utf-8", errors="strict")
            except UnicodeError as exc:
                raise ReferenceCorpusContractError("contact geometry name is not UTF-8") from exc
            if not text or len(encoded) > GEOM_NAME_BYTES:
                raise ReferenceCorpusContractError("contact geometry name is invalid")
    for transition in range(steps):
        start = int(result["contact_offsets"][transition])
        stop = int(result["contact_offsets"][transition + 1])
        expected_contact_index = [0] * 5
        previous_key = (-1, -1)
        nonfoot_floor_contact = False
        for index in range(start, stop):
            substep = int(result["contact_substep"][index])
            within = int(result["contact_index_within_substep"][index])
            key = (substep, within)
            if key <= previous_key or within != expected_contact_index[substep]:
                raise ReferenceCorpusContractError("contact sequence order differs")
            previous_key = key
            expected_contact_index[substep] += 1
            names = {
                bytes(result["contact_geom1_name"][index]).rstrip(b"\0").decode(),
                bytes(result["contact_geom2_name"][index]).rstrip(b"\0").decode(),
            }
            if "floor" in names and not names.issubset({"floor", "left_foot", "right_foot"}):
                nonfoot_floor_contact = True
        if bool(result["transition_nonfoot_floor_contact"][transition]) != (nonfoot_floor_contact):
            raise ReferenceCorpusContractError("non-foot floor contact flag differs")
    hash_arrays = (
        "boundary_integration_sha256",
        "boundary_observation_sha256",
        "boundary_cfrc_ext_sha256",
        "boundary_root_xy_sha256",
        "boundary_reference_sha256",
        "boundary_rng_state_sha256",
        "boundary_record_sha256",
        "transition_actor_input_sha256",
        "transition_actor_output_sha256",
        "transition_physical_action_sha256",
        "transition_returned_observation_sha256",
        "transition_contact_sequence_sha256",
        "transition_record_sha256",
    )
    for name in hash_arrays:
        _hash_ascii(result[name], name=name)
    for index in range(steps + 1):
        expected = {
            "boundary_integration_sha256": array_sha256(
                result["boundary_integration_state"][index]
            ),
            "boundary_observation_sha256": array_sha256(result["boundary_observation"][index]),
            "boundary_cfrc_ext_sha256": array_sha256(result["boundary_cfrc_ext"][index]),
            "boundary_root_xy_sha256": array_sha256(result["boundary_root_xy"][index]),
            "boundary_reference_sha256": array_sha256(result["reference_rows"][index]),
            "boundary_record_sha256": boundary_record_sha256(result, index),
        }
        for name, digest in expected.items():
            if bytes(result[name][index]).decode() != digest:
                raise ReferenceCorpusContractError(f"{name} differs at boundary {index}")
    for index in range(steps):
        expected = {
            "transition_actor_input_sha256": array_sha256(
                np.ascontiguousarray(result["boundary_observation"][index], dtype="<f4")
            ),
            "transition_actor_output_sha256": array_sha256(
                result["transition_normalized_action"][index]
            ),
            "transition_physical_action_sha256": array_sha256(
                result["transition_physical_action"][index]
            ),
            "transition_returned_observation_sha256": array_sha256(
                result["transition_returned_observation"][index]
            ),
            "transition_contact_sequence_sha256": contact_sequence_sha256(result, index),
            "transition_record_sha256": transition_record_sha256(result, index),
        }
        for name, digest in expected.items():
            if bytes(result[name][index]).decode() != digest:
                raise ReferenceCorpusContractError(f"{name} differs at transition {index}")
    return result


def array_bindings(values: Mapping[str, np.ndarray]) -> list[dict[str, object]]:
    return [
        ArrayBindingV2(
            name=name,
            dtype=value.dtype.str,
            shape=tuple(int(size) for size in value.shape),
            sha256=array_sha256(value),
        ).to_dict()
        for name, value in sorted(values.items())
    ]


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, value, version=(1, 0), allow_pickle=False)
    return stream.getvalue()


def encode_clip_payload(
    values: dict[str, np.ndarray],
    *,
    steps: int,
    screen_canary: bool,
) -> bytes:
    arrays = validate_clip_arrays(values, steps=steps, screen_canary=screen_canary)
    stream = io.BytesIO()
    with ZipFile(stream, mode="w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(arrays):
            info = ZipInfo(filename=f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100600 << 16
            archive.writestr(
                info, _npy_bytes(arrays[name]), compress_type=ZIP_DEFLATED, compresslevel=6
            )
    payload = stream.getvalue()
    if not payload or len(payload) > MAX_CLIP_PAYLOAD_BYTES:
        raise ReferenceCorpusContractError("canonical clip payload size is invalid")
    return payload


def decode_clip_payload(
    payload: bytes,
    *,
    steps: int,
    screen_canary: bool,
) -> dict[str, np.ndarray]:
    if type(payload) is not bytes or not payload or len(payload) > MAX_CLIP_PAYLOAD_BYTES:
        raise ReferenceCorpusContractError("clip payload bytes are invalid")
    try:
        with ZipFile(io.BytesIO(payload), mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename.removesuffix(".npy") for info in infos]
            if any(not info.filename.endswith(".npy") for info in infos):
                raise ReferenceCorpusContractError("clip payload contains a non-array member")
            if names != sorted(names) or len(names) != len(set(names)):
                raise ReferenceCorpusContractError(
                    "clip payload member order or uniqueness differs"
                )
            arrays: dict[str, np.ndarray] = {}
            for info, name in zip(infos, names, strict=True):
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
                    raise ReferenceCorpusContractError("clip payload ZIP metadata differs")
                with archive.open(info, mode="r") as member:
                    value = np.lib.format.read_array(member, allow_pickle=False)
                arrays[name] = np.ascontiguousarray(value)
    except ReferenceCorpusContractError:
        raise
    except (BadZipFile, EOFError, OSError, ValueError) as exc:
        raise ReferenceCorpusContractError(f"clip payload is invalid: {exc}") from exc
    validated = validate_clip_arrays(arrays, steps=steps, screen_canary=screen_canary)
    if encode_clip_payload(validated, steps=steps, screen_canary=screen_canary) != payload:
        raise ReferenceCorpusContractError("clip payload bytes are not canonical")
    return validated


def reference_schema_payload() -> dict[str, object]:
    return {
        "schema_id": REFERENCE_SCHEMA_ID,
        "feature_names": list(HUMANOID_REFERENCE_SCHEMA.feature_names),
        "feature_units": list(HUMANOID_REFERENCE_SCHEMA.feature_units),
        "root_frame": HUMANOID_REFERENCE_SCHEMA.root_frame,
        "joint_order": list(HUMANOID_ACTUATOR_JOINT_ORDER),
        "width": REFERENCE_WIDTH,
        "cadence_seconds": CONTROL_PERIOD_SECONDS,
        "horizon": REFERENCE_HORIZON,
        "window_advance_rows_per_step": 1,
        "wrap": False,
        "root_xy_policy_visible": False,
        "quaternion_sign": "first_positive_then_adjacent_dot_nonnegative",
        "angular_velocity_frame": "torso_local",
    }


def validate_runtime_identity(value: Mapping[str, object]) -> dict[str, object]:
    expected_keys = {
        "runtime_id",
        "environment_id",
        "environment_kwargs",
        "environment_semantics",
        "wrapper_types",
        "time_limit_steps",
        "python_version",
        "platform_system",
        "platform_release",
        "platform_machine",
        "numpy_version",
        "torch_version",
        "gymnasium_version",
        "mujoco_version",
        "model_sha256",
        "model_byte_count",
        "uv_lock_sha256",
        "observation_space_sha256",
        "action_space_sha256",
        "integration_state_spec",
        "integration_state_flag",
        "integration_state_size",
        "integration_state_components",
        "timestep_seconds",
        "frame_skip",
        "control_period_seconds",
        "integrator",
        "solver",
        "solver_iterations",
        "contact_capture_id",
        "source_hashes",
        "thread_settings",
        "cpu_only",
        "deterministic_actor_mean",
        "same_host_determinism_scope",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise ReferenceCorpusContractError("runtime identity keys differ")
    frozen = {
        "runtime_id": RUNTIME_ID,
        "environment_id": "Humanoid-v5",
        "environment_kwargs": {
            "exclude_current_positions_from_observation": True,
            "frame_skip": 5,
            "render_mode": None,
            "reset_noise_scale": 0.01,
            "terminate_when_unhealthy": False,
        },
        "environment_semantics": {
            "termination": "TimeLimit_only_at_1000",
            "healthy_z_range_open_m": [1.0, 2.0],
            "observation": "raw_Humanoid-v5_348d_excluding_root_xy",
            "action": "physical_17d_float32_in_closed_minus_0.4_to_0.4",
        },
        "wrapper_types": list(EXPECTED_REFERENCE_WRAPPER_TYPES),
        "time_limit_steps": 1000,
        "numpy_version": "2.5.2",
        "torch_version": "2.14.0",
        "gymnasium_version": "1.3.0",
        "mujoco_version": "3.12.0",
        "integration_state_spec": STATE_SPEC,
        "integration_state_flag": STATE_FLAG,
        "integration_state_size": STATE_SIZE,
        "integration_state_components": [list(item) for item in EXPECTED_STATE_COMPONENT_SIZES],
        "timestep_seconds": 0.003,
        "frame_skip": 5,
        "control_period_seconds": CONTROL_PERIOD_SECONDS,
        "integrator": "mjINT_RK4",
        "solver": "mjSOL_PGS",
        "solver_iterations": 50,
        "contact_capture_id": FULL_CONTACT_CAPTURE_ID,
        "cpu_only": True,
        "deterministic_actor_mean": True,
        "same_host_determinism_scope": "same_host_same_process_settings_bitwise/v1",
    }
    for field, expected in frozen.items():
        if value[field] != expected or type(value[field]) is not type(expected):
            raise ReferenceCorpusContractError(f"runtime {field} differs")
    for field in ("python_version", "platform_system", "platform_release", "platform_machine"):
        if type(value[field]) is not str or not value[field]:
            raise ReferenceCorpusContractError(f"runtime {field} is invalid")
    for field in (
        "model_sha256",
        "uv_lock_sha256",
        "observation_space_sha256",
        "action_space_sha256",
    ):
        try:
            require_sha256(value[field], field=field)
        except ReferenceIdentityV2Error as exc:
            raise ReferenceCorpusContractError(str(exc)) from exc
    if type(value["model_byte_count"]) is not int or value["model_byte_count"] <= 0:
        raise ReferenceCorpusContractError("runtime model byte count is invalid")
    for mapping_field in ("source_hashes", "thread_settings"):
        if type(value[mapping_field]) is not dict or not value[mapping_field]:
            raise ReferenceCorpusContractError(f"runtime {mapping_field} is invalid")
    for source_hash in value["source_hashes"].values():
        try:
            require_sha256(source_hash, field="runtime source hash")
        except ReferenceIdentityV2Error as exc:
            raise ReferenceCorpusContractError(str(exc)) from exc
    canonical_json_bytes(value)
    return dict(value)


def require_same_runtime(recorded: Mapping[str, object], observed: Mapping[str, object]) -> None:
    validate_runtime_identity(recorded)
    validate_runtime_identity(observed)
    if canonical_json_bytes(recorded) != canonical_json_bytes(observed):
        raise ReferenceCorpusContractError("live runtime differs from the recorded runtime")


def validate_bundle_manifest(value: Mapping[str, object]) -> dict[str, object]:
    """Validate the non-circular bundle core plus its v2 reference identity."""

    expected_keys = {
        "bundle_id",
        "schema_version",
        "core",
        "core_sha256",
        "reference_identity",
        "reference_identity_sha256",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise ReferenceCorpusContractError("bundle manifest keys differ")
    if value["bundle_id"] != "humanoid_full_clip_replay_bundle/v1" or value["schema_version"] != 1:
        raise ReferenceCorpusContractError("bundle manifest identity differs")
    core = value["core"]
    if type(core) is not dict:
        raise ReferenceCorpusContractError("bundle core must be an exact mapping")
    required_core = {
        "artifact_type",
        "clip_id",
        "clip_kind",
        "actor_variant",
        "seed",
        "reset_order",
        "no_retry",
        "steps",
        "boundaries",
        "collector_pid",
        "runtime_identity",
        "runtime_identity_sha256",
        "source_actor",
        "bound_artifacts",
        "payload",
        "array_bindings",
        "reference_contract",
        "wrapper_state_contract",
        "contact_contract",
        "policy_visible_input",
        "controller_state",
        "rng_state",
        "reward_canary",
        "source_hashes",
        "same_host_determinism_scope",
    }
    if set(core) != required_core:
        raise ReferenceCorpusContractError("bundle core keys differ")
    if core["artifact_type"] != "same_runtime_full_integration_replay":
        raise ReferenceCorpusContractError("Minari-only or reduced-state artifact is not Tier D")
    if core["clip_kind"] not in {"corpus", "development_screen"}:
        raise ReferenceCorpusContractError("clip kind differs")
    if core["actor_variant"] not in {"expert", "medium", "simple"}:
        raise ReferenceCorpusContractError("actor variant differs")
    if core["clip_kind"] == "development_screen" and core["actor_variant"] != "expert":
        raise ReferenceCorpusContractError("development screen must use the expert actor")
    if type(core["clip_id"]) is not str or not core["clip_id"] or len(core["clip_id"]) > 128:
        raise ReferenceCorpusContractError("bundle clip id is invalid")
    if type(core["seed"]) is not int or isinstance(core["seed"], bool) or core["seed"] < 0:
        raise ReferenceCorpusContractError("bundle seed is invalid")
    if (
        type(core["reset_order"]) is not int
        or isinstance(core["reset_order"], bool)
        or core["reset_order"] < 0
    ):
        raise ReferenceCorpusContractError("bundle reset order is invalid")
    if type(core["steps"]) is not int or not 1 <= core["steps"] <= 1000:
        raise ReferenceCorpusContractError("bundle steps are invalid")
    if core["boundaries"] != core["steps"] + 1:
        raise ReferenceCorpusContractError("bundle is missing intermediate boundaries")
    if core["no_retry"] is not True:
        raise ReferenceCorpusContractError("bundle permits retry")
    if type(core["collector_pid"]) is not int or core["collector_pid"] <= 0:
        raise ReferenceCorpusContractError("collector pid is invalid")
    validate_runtime_identity(core["runtime_identity"])
    if core["runtime_identity_sha256"] != sha256_json(core["runtime_identity"]):
        raise ReferenceCorpusContractError("runtime identity hash differs")
    reference = core["reference_contract"]
    if reference != reference_schema_payload():
        raise ReferenceCorpusContractError("reference contract differs")
    if core["wrapper_state_contract"] != {
        "id": WRAPPER_STATE_ID,
        "elapsed_counter": "TimeLimit._elapsed_steps",
        "flags_in_order": list(WRAPPER_FLAG_NAMES),
        "rng_state_bound_at_every_boundary": True,
        "unknown_fields_permitted": False,
    }:
        raise ReferenceCorpusContractError("wrapper-state restoration is unknown or incomplete")
    if core["contact_contract"] != {
        "id": CONTACT_SEQUENCE_ID,
        "substeps": 5,
        "force_frame": "MuJoCo_contact_frame",
        "force_components": ["normal", "tangent1", "tangent2", "torque1", "torque2", "torsion"],
        "ordered_by": ["transition", "physics_substep", "contact_index_within_substep"],
    }:
        raise ReferenceCorpusContractError("contact contract differs")
    if core["policy_visible_input"] != {
        "id": POLICY_INPUT_ID,
        "fields": ["raw_observation_f32"],
        "shape": [OBSERVATION_WIDTH],
        "branch_metadata_permitted": False,
    }:
        raise ReferenceCorpusContractError("branch metadata entered the policy-visible input")
    if core["controller_state"] != {
        "actor_recurrent_state": None,
        "normalizer_state": None,
        "residual_controller_state": None,
    }:
        raise ReferenceCorpusContractError("controller continuation state differs")
    if core["same_host_determinism_scope"] != "same_host_same_process_settings_bitwise/v1":
        raise ReferenceCorpusContractError("bundle determinism scope differs")
    if type(core["source_actor"]) is not dict or set(core["source_actor"]) != {
        "variant",
        "npz_sha256",
        "actor_state_sha256",
        "actor_schema_sha256",
        "source_policy_sha256",
        "import_receipt_sha256",
        "equivalence_receipt_sha256",
        "inference_id",
    }:
        raise ReferenceCorpusContractError("source actor identity keys differ")
    if core["source_actor"]["variant"] != core["actor_variant"]:
        raise ReferenceCorpusContractError("source actor variant differs")
    for field, child in core["source_actor"].items():
        if field.endswith("sha256"):
            require_sha256(child, field=field)
    if type(core["bound_artifacts"]) is not list or not core["bound_artifacts"]:
        raise ReferenceCorpusContractError("bundle has no bound local artifacts")
    artifact_bindings: list[ArtifactBindingV2] = []
    for raw in core["bound_artifacts"]:
        if type(raw) is not dict or set(raw) != {
            "role",
            "logical_path",
            "object_path",
            "sha256",
            "byte_count",
        }:
            raise ReferenceCorpusContractError("bound artifact binding differs")
        try:
            artifact_bindings.append(ArtifactBindingV2(**raw))
        except (TypeError, ReferenceIdentityV2Error) as exc:
            raise ReferenceCorpusContractError(f"bound artifact is invalid: {exc}") from exc
    roles = {binding.role for binding in artifact_bindings}
    if len(roles) != len(artifact_bindings):
        raise ReferenceCorpusContractError("bound artifact roles are duplicated")
    required_roles = {
        "source_policy_bytes",
        "strict_actor_npz",
        "import_receipt",
        "equivalence_receipt",
        "mujoco_model_bytes",
        "dependency_lock",
        "collector_source",
        "certifier_source",
        "metric_source",
    }
    if not required_roles.issubset(roles):
        raise ReferenceCorpusContractError("bundle omits a required replay artifact")
    source_actor_to_role = {
        "npz_sha256": "strict_actor_npz",
        "source_policy_sha256": "source_policy_bytes",
        "import_receipt_sha256": "import_receipt",
        "equivalence_receipt_sha256": "equivalence_receipt",
    }
    for actor_field, role in source_actor_to_role.items():
        matches = [binding for binding in artifact_bindings if binding.role == role]
        if len(matches) != 1 or matches[0].sha256 != core["source_actor"][actor_field]:
            raise ReferenceCorpusContractError(
                f"source actor {actor_field} differs from its bound artifact"
            )
    payload = core["payload"]
    if type(payload) is not dict or set(payload) != {
        "role",
        "logical_path",
        "object_path",
        "sha256",
        "byte_count",
    }:
        raise ReferenceCorpusContractError("bundle payload binding differs")
    try:
        payload_binding = ArtifactBindingV2(**payload)
    except (TypeError, ReferenceIdentityV2Error) as exc:
        raise ReferenceCorpusContractError(f"bundle numeric payload is invalid: {exc}") from exc
    if payload_binding.role != "clip_numeric_payload":
        raise ReferenceCorpusContractError("bundle numeric payload role differs")
    if type(core["array_bindings"]) is not list or not core["array_bindings"]:
        raise ReferenceCorpusContractError("bundle array bindings are absent")
    array_names: set[str] = set()
    for binding in core["array_bindings"]:
        if type(binding) is not dict:
            raise ReferenceCorpusContractError("array binding is malformed")
        try:
            parsed = ArrayBindingV2(
                name=binding.get("name"),
                dtype=binding.get("dtype"),
                shape=tuple(binding.get("shape", ())),
                sha256=binding.get("sha256"),
            )
        except (TypeError, ReferenceIdentityV2Error) as exc:
            raise ReferenceCorpusContractError(f"array binding is invalid: {exc}") from exc
        if parsed.name in array_names:
            raise ReferenceCorpusContractError("array binding names are duplicated")
        array_names.add(parsed.name)
    rng_state = core["rng_state"]
    if (
        type(rng_state) is not dict
        or set(rng_state) != {"binding", "boundary_hash_array"}
        or rng_state["boundary_hash_array"] != "boundary_rng_state_sha256"
        or type(rng_state["binding"]) is not dict
        or rng_state["binding"].get("role") != "environment_rng_state"
    ):
        raise ReferenceCorpusContractError("bundle RNG restoration binding differs")
    try:
        rng_binding = ArtifactBindingV2(**rng_state["binding"])
    except (TypeError, ReferenceIdentityV2Error) as exc:
        raise ReferenceCorpusContractError(f"bundle RNG binding is invalid: {exc}") from exc
    rng_matches = [binding for binding in artifact_bindings if binding.role == rng_binding.role]
    if len(rng_matches) != 1 or rng_matches[0] != rng_binding:
        raise ReferenceCorpusContractError("bundle RNG object is not bound exactly once")
    if "boundary_rng_state_sha256" not in array_names:
        raise ReferenceCorpusContractError("bundle omits boundary RNG hashes")
    if type(core["source_hashes"]) is not dict or set(core["source_hashes"]) != {
        "collector_source",
        "certifier_source",
        "metric_source",
    }:
        raise ReferenceCorpusContractError("bundle source-hash roles differ")
    for field, digest in core["source_hashes"].items():
        try:
            require_sha256(digest, field=f"{field} SHA-256")
        except ReferenceIdentityV2Error as exc:
            raise ReferenceCorpusContractError(str(exc)) from exc
        matches = [binding for binding in artifact_bindings if binding.role == field]
        if len(matches) != 1 or matches[0].sha256 != digest:
            raise ReferenceCorpusContractError(f"bundle {field} differs from its source object")
    if (
        core["source_hashes"]["collector_source"]
        != core["runtime_identity"]["source_hashes"]["reference_corpus_collector"]
    ):
        raise ReferenceCorpusContractError("collector source differs from the runtime identity")
    runtime_roles = {
        "mujoco_model_bytes": ("model_sha256", "model_byte_count"),
        "dependency_lock": ("uv_lock_sha256", None),
    }
    for role, (hash_field, size_field) in runtime_roles.items():
        matches = [binding for binding in artifact_bindings if binding.role == role]
        if len(matches) != 1 or matches[0].sha256 != core["runtime_identity"][hash_field]:
            raise ReferenceCorpusContractError(f"runtime {role} differs from its bound object")
        if size_field is not None and matches[0].byte_count != core["runtime_identity"][size_field]:
            raise ReferenceCorpusContractError(f"runtime {role} size differs")
    reward_canary = core["reward_canary"]
    if type(reward_canary) is not dict:
        raise ReferenceCorpusContractError("reward canary must be an exact mapping")
    if core["clip_kind"] == "development_screen":
        if (
            set(reward_canary)
            != {
                "status",
                "role",
                "locomotion_metrics_may_read_reward",
                "instrumented_reward_sha256",
                "plain_reward_sha256",
                "visual_capture",
            }
            or reward_canary["status"] != "passed"
            or reward_canary["role"] != "plain_vs_instrumented_equivalence_only"
            or reward_canary["locomotion_metrics_may_read_reward"] is not False
            or reward_canary["instrumented_reward_sha256"] != reward_canary["plain_reward_sha256"]
            or reward_canary["visual_capture"] != "disabled_by_external_screen_design/v1"
        ):
            raise ReferenceCorpusContractError("development-screen reward canary differs")
        for field in ("instrumented_reward_sha256", "plain_reward_sha256"):
            try:
                require_sha256(reward_canary[field], field=field)
            except ReferenceIdentityV2Error as exc:
                raise ReferenceCorpusContractError(str(exc)) from exc
    elif reward_canary != {
        "status": "not_applicable",
        "role": "replay_canary_only",
        "locomotion_metrics_may_read_reward": False,
    }:
        raise ReferenceCorpusContractError("corpus reward canary differs")
    if core.get("core_sha256", False):
        raise ReferenceCorpusContractError("bundle core must not contain its own hash")
    if value["core_sha256"] != sha256_json(core):
        raise ReferenceCorpusContractError("bundle core hash differs")
    identity_raw = value["reference_identity"]
    identity_keys = {
        "identity_id",
        "schema_version",
        *ReferenceIdentityV2.__dataclass_fields__,
    }
    if (
        type(identity_raw) is not dict
        or set(identity_raw) != identity_keys
        or identity_raw.get("identity_id") != "humanoid_same_runtime_reference_identity/v2"
        or identity_raw.get("schema_version") != 2
    ):
        raise ReferenceCorpusContractError("reference identity payload differs")
    identity_fields = dict(identity_raw)
    identity_fields.pop("identity_id", None)
    identity_fields.pop("schema_version", None)
    identity_fields["parent_provenance"] = dict(identity_fields["parent_provenance"])
    try:
        identity = ReferenceIdentityV2(**identity_fields)
    except (TypeError, ReferenceIdentityV2Error) as exc:
        raise ReferenceCorpusContractError(f"reference identity is invalid: {exc}") from exc
    if identity.replay_bundle_core_sha256 != value["core_sha256"]:
        raise ReferenceCorpusContractError("reference identity does not bind the replay bundle")
    expected_identity_fields = {
        "reference_schema_sha256": sha256_json(reference_schema_payload()),
        "actor_npz_sha256": core["source_actor"]["npz_sha256"],
        "actor_state_sha256": core["source_actor"]["actor_state_sha256"],
        "import_receipt_sha256": core["source_actor"]["import_receipt_sha256"],
        "equivalence_receipt_sha256": core["source_actor"]["equivalence_receipt_sha256"],
        "generator_source_sha256": core["source_hashes"]["collector_source"],
        "certifier_source_sha256": core["source_hashes"]["certifier_source"],
        "runtime_identity_sha256": core["runtime_identity_sha256"],
        "seed": core["seed"],
        "n_boundaries": core["boundaries"],
        "cadence_seconds": CONTROL_PERIOD_SECONDS,
    }
    for field, expected in expected_identity_fields.items():
        if getattr(identity, field) != expected:
            raise ReferenceCorpusContractError(f"reference identity {field} differs")
    expected_parent = {
        "actor_variant": core["actor_variant"],
        "source_policy_sha256": core["source_actor"]["source_policy_sha256"],
        "import_receipt_sha256": core["source_actor"]["import_receipt_sha256"],
        "equivalence_receipt_sha256": core["source_actor"]["equivalence_receipt_sha256"],
    }
    if identity.derivation_kind != "original" or identity.parent_provenance != expected_parent:
        raise ReferenceCorpusContractError("reference identity parent provenance differs")
    if identity.reference_content_sha256 != array_sha256_from_binding(
        core["array_bindings"], "reference_rows"
    ):
        raise ReferenceCorpusContractError("reference identity content hash differs")
    if identity.root_xy_sidecar_sha256 != array_sha256_from_binding(
        core["array_bindings"], "boundary_root_xy"
    ):
        raise ReferenceCorpusContractError("reference identity root x/y hash differs")
    if value["reference_identity_sha256"] != identity.sha256:
        raise ReferenceCorpusContractError("reference identity hash differs")
    return dict(value)


def array_sha256_from_binding(bindings: object, name: str) -> str:
    if type(bindings) is not list:
        raise ReferenceCorpusContractError("array bindings must be a list")
    matches = [
        binding for binding in bindings if type(binding) is dict and binding.get("name") == name
    ]
    if len(matches) != 1:
        raise ReferenceCorpusContractError(f"array binding {name!r} is missing or duplicated")
    digest = matches[0].get("sha256")
    try:
        return require_sha256(digest, field=f"array binding {name} SHA-256")
    except ReferenceIdentityV2Error as exc:
        raise ReferenceCorpusContractError(str(exc)) from exc


__all__ = [
    "ACTION_WIDTH",
    "CLIP_PAYLOAD_ID",
    "CONTACT_SEQUENCE_ID",
    "EXPECTED_STATE_COMPONENT_SIZES",
    "GEOM_NAME_BYTES",
    "OBSERVATION_WIDTH",
    "POLICY_INPUT_ID",
    "QPOS_SLICE",
    "QVEL_SLICE",
    "REFERENCE_CONSTRUCTION_ID",
    "RUNTIME_ID",
    "STATE_DTYPE",
    "STATE_FLAG",
    "STATE_SIZE",
    "WRAPPER_FLAG_NAMES",
    "WRAPPER_STATE_ID",
    "ReferenceCorpusContractError",
    "array_bindings",
    "array_sha256_from_binding",
    "boundary_record_sha256",
    "contact_sequence_sha256",
    "decode_clip_payload",
    "derive_reference_row",
    "derive_reference_rows",
    "encode_clip_payload",
    "reference_schema_payload",
    "reference_window_index_arrays",
    "require_same_runtime",
    "transition_record_sha256",
    "validate_bundle_manifest",
    "validate_clip_arrays",
    "validate_reference_window_indices",
    "validate_runtime_identity",
]
