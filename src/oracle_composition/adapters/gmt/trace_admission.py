"""Data-only admission for retained GMT replay traces."""

from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import verify_upstream_root
from .contracts import (
    ACTION_DIM,
    ACTION_SCALE,
    CONTROL_DT_SECONDS,
    GMT_G1_MESH_NAMES,
    GMT_G1_MESH_TREE_SHA256,
    GMT_UPSTREAM_COMMIT,
    HISTORY_LENGTH,
    MOTION_SPECS,
    OBSERVATION_DIM,
    PROPRIOCEPTION_DIM,
    RAW_ACTION_MAX,
    RAW_ACTION_MIN,
    REFERENCE_FRAME_DIM,
    REFERENCE_HORIZON,
    REFERENCE_OFFSETS,
    SIMULATION_DECIMATION,
    SIMULATION_DT_SECONDS,
)
from .io import GMTAdmissionError, read_verified_bytes, validate_zip_members
from .replay import ModelABI, validate_model_abi

MAX_TRACE_BYTES = 32 * 1024 * 1024


def _reject_json_constant(value: str) -> None:
    raise GMTAdmissionError(f"non-finite JSON constant is forbidden: {value}")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GMTAdmissionError(f"duplicate manifest key: {key}")
        result[key] = value
    return result


def read_trace_manifest(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    payload = read_verified_bytes(Path(path), expected_sha256, maximum_size=128 * 1024)
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GMTAdmissionError("invalid GMT replay manifest JSON") from exc
    if not isinstance(decoded, dict):
        raise GMTAdmissionError("GMT replay manifest must be a JSON object")
    return decoded


def _model_abi(value: Any) -> ModelABI:
    if not isinstance(value, dict):
        raise GMTAdmissionError("model_abi must be an object")
    try:
        abi = ModelABI(
            nq=int(value["nq"]),
            nv=int(value["nv"]),
            nu=int(value["nu"]),
            joint_names=tuple(value["joint_names"]),
            joint_types=tuple(value["joint_types"]),
            actuator_names=tuple(value["actuator_names"]),
            actuator_joint_ids=tuple(int(item) for item in value["actuator_joint_ids"]),
            sensor_names=tuple(value["sensor_names"]),
            sensor_types=tuple(value["sensor_types"]),
            sensor_dimensions=tuple(int(item) for item in value["sensor_dimensions"]),
            keyframe_count=int(value["keyframe_count"]),
            home_qpos=tuple(float(item) for item in value["home_qpos"]),
            timestep=float(value["timestep"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise GMTAdmissionError("malformed model_abi") from exc
    validate_model_abi(abi)
    return abi


def expected_trace_contract(simulation_steps: int) -> dict[str, tuple[tuple[int, ...], str]]:
    if not 1 <= simulation_steps <= 10_000:
        raise GMTAdmissionError("trace must contain 1..10000 simulation steps")
    control_steps = (simulation_steps + SIMULATION_DECIMATION - 1) // SIMULATION_DECIMATION
    return {
        "sim_time": ((simulation_steps,), "<f8"),
        "sim_qpos": ((simulation_steps, 30), "<f8"),
        "sim_qvel": ((simulation_steps, 29), "<f8"),
        "sim_pd_target": ((simulation_steps, ACTION_DIM), "<f8"),
        "sim_torque": ((simulation_steps, ACTION_DIM), "<f8"),
        "sim_contact_count": ((simulation_steps,), "<i4"),
        "sim_torque_saturation_fraction": ((simulation_steps,), "<f4"),
        "control_sim_step": ((control_steps,), "<i8"),
        "control_time": ((control_steps,), "<f8"),
        "control_orientation_wxyz": ((control_steps, 4), "<f4"),
        "control_angular_velocity": ((control_steps, 3), "<f4"),
        "control_reference_window": (
            (control_steps, REFERENCE_HORIZON, REFERENCE_FRAME_DIM),
            "<f4",
        ),
        "control_reference_current": ((control_steps, REFERENCE_FRAME_DIM), "<f4"),
        "control_proprioception": ((control_steps, PROPRIOCEPTION_DIM), "<f8"),
        "control_observation": ((control_steps, OBSERVATION_DIM), "<f4"),
        "control_raw_action": ((control_steps, ACTION_DIM), "<f4"),
        "control_clipped_action": ((control_steps, ACTION_DIM), "<f4"),
        "control_pd_target": ((control_steps, ACTION_DIM), "<f8"),
        "control_joint_rmse": ((control_steps,), "<f8"),
        "control_root_height_abs_error": ((control_steps,), "<f8"),
        "control_roll_pitch_rmse": ((control_steps,), "<f8"),
        "control_action_saturation_fraction": ((control_steps,), "<f4"),
    }


def validate_trace_manifest(
    manifest: dict[str, Any],
    *,
    trace_path: Path,
    support_files: dict[str, str],
) -> tuple[ModelABI, dict[str, tuple[tuple[int, ...], str]]]:
    try:
        trace = manifest["trace"]
        inputs = manifest["inputs"]
        runtime = manifest["runtime"]
        controller = manifest["controller_abi"]
        metrics = manifest["metrics"]
        limits = manifest["limits"]
    except KeyError as exc:
        raise GMTAdmissionError(f"missing replay manifest section: {exc.args[0]}") from exc
    if manifest.get("schema_version") != 1:
        raise GMTAdmissionError("unsupported replay manifest schema")
    if manifest.get("artifact") != "gmt_g1_reconstructed_actor_headless_replay":
        raise GMTAdmissionError("manifest is not a GMT reconstructed-actor replay")
    if manifest.get("claim_status") != "reconstructed_plain_actor_not_jit_equivalent":
        raise GMTAdmissionError("unexpected replay claim status")
    if inputs.get("upstream_commit") != GMT_UPSTREAM_COMMIT:
        raise GMTAdmissionError("replay upstream commit mismatch")
    if inputs.get("motion_name") not in MOTION_SPECS:
        raise GMTAdmissionError("replay motion is outside the pinned GMT catalog")
    if inputs.get("support_files") != support_files:
        raise GMTAdmissionError("replay model/mesh identities do not match the verified files")
    mesh_keys = {f"assets/robots/g1/meshes/{name}" for name in GMT_G1_MESH_NAMES}
    if (
        not mesh_keys.issubset(support_files)
        or support_files.get("assets/robots/g1/meshes@tree") != GMT_G1_MESH_TREE_SHA256
    ):
        raise GMTAdmissionError("verified support receipt is missing the pinned mesh identities")
    if runtime.get("simulation_dt_seconds") != SIMULATION_DT_SECONDS:
        raise GMTAdmissionError("replay simulation timing mismatch")
    if runtime.get("control_dt_seconds") != CONTROL_DT_SECONDS:
        raise GMTAdmissionError("replay control timing mismatch")
    if runtime.get("state_sampling") != "pre_step":
        raise GMTAdmissionError("replay state sampling mismatch")
    expected_controller = {
        "reference_quaternion_order": "xyzw",
        "reference_quaternions_normalized": False,
        "sensor_quaternion_order": "wxyz",
        "reference_offsets_control_steps": list(REFERENCE_OFFSETS),
        "history_frames": HISTORY_LENGTH,
        "history_update": "append_current_proprioception_after_actor_call",
        "action_history": "raw_preclip_actor_output",
        "raw_action_clip": [RAW_ACTION_MIN, RAW_ACTION_MAX],
        "action_scale": ACTION_SCALE,
        "pd_torque_recomputed_each_simulation_step": True,
    }
    if controller != expected_controller:
        raise GMTAdmissionError("replay controller ABI mismatch")
    if (
        limits.get("original_jit_executed") is not False
        or limits.get("jit_equivalence_tested") is not False
    ):
        raise GMTAdmissionError("replay does not preserve the reconstructed-actor boundary")
    try:
        duration = float(metrics["simulated_seconds"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GMTAdmissionError("replay duration is missing or invalid") from exc
    if not math.isfinite(duration) or not 0 < duration <= 10.0:
        raise GMTAdmissionError("replay duration must be in (0, 10] seconds")
    simulation_steps = round(duration / SIMULATION_DT_SECONDS)
    if not math.isclose(
        simulation_steps * SIMULATION_DT_SECONDS,
        duration,
        rel_tol=0,
        abs_tol=1.0e-12,
    ):
        raise GMTAdmissionError("replay duration is not aligned to 1 ms")
    contract = expected_trace_contract(simulation_steps)
    expected_arrays = {
        key: {"shape": list(shape), "dtype": dtype} for key, (shape, dtype) in contract.items()
    }
    if not isinstance(trace, dict) or trace.get("arrays") != expected_arrays:
        raise GMTAdmissionError("replay trace-array contract mismatch")
    if trace.get("path") != trace_path.name:
        raise GMTAdmissionError("replay trace filename mismatch")
    if trace.get("size") != trace_path.stat().st_size:
        raise GMTAdmissionError("replay trace size mismatch")
    return _model_abi(manifest.get("model_abi")), contract


def read_trace_arrays(
    path: Path,
    *,
    expected_sha256: str,
    contract: dict[str, tuple[tuple[int, ...], str]],
) -> dict[str, np.ndarray]:
    payload = read_verified_bytes(Path(path), expected_sha256, maximum_size=MAX_TRACE_BYTES)
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise GMTAdmissionError("invalid GMT replay NPZ") from exc
    arrays: dict[str, np.ndarray] = {}
    with archive:
        members = validate_zip_members(
            archive,
            expected_count=len(contract),
            maximum_member_size=8 * 1024 * 1024,
        )
        if sum(info.file_size for info in members.values()) > MAX_TRACE_BYTES:
            raise GMTAdmissionError("GMT trace uncompressed data exceeds its bound")
        if set(members) != {f"{key}.npy" for key in contract}:
            raise GMTAdmissionError("GMT trace member set mismatch")
        for key, (shape, dtype) in contract.items():
            member_name = f"{key}.npy"
            info = members[member_name]
            if info.compress_type != zipfile.ZIP_STORED:
                raise GMTAdmissionError(f"trace member must be stored: {member_name}")
            stream = io.BytesIO(archive.read(member_name))
            try:
                array = np.lib.format.read_array(stream, allow_pickle=False, max_header_size=512)
            except (ValueError, EOFError) as exc:
                raise GMTAdmissionError(f"invalid numeric trace member: {member_name}") from exc
            if stream.tell() != info.file_size:
                raise GMTAdmissionError(f"trailing bytes in trace member: {member_name}")
            if array.shape != shape or array.dtype.str != dtype or not array.flags.c_contiguous:
                raise GMTAdmissionError(f"trace array contract mismatch: {member_name}")
            if not np.isfinite(array).all():
                raise GMTAdmissionError(f"non-finite trace member: {member_name}")
            arrays[key] = array
    expected_sim_time = np.arange(arrays["sim_time"].shape[0]) * SIMULATION_DT_SECONDS
    expected_control_step = np.arange(arrays["control_sim_step"].shape[0]) * SIMULATION_DECIMATION
    if not np.array_equal(arrays["sim_time"], expected_sim_time):
        raise GMTAdmissionError("trace simulation clock mismatch")
    if not np.array_equal(arrays["control_sim_step"], expected_control_step):
        raise GMTAdmissionError("trace control-step clock mismatch")
    if not np.array_equal(arrays["control_time"], expected_control_step * SIMULATION_DT_SECONDS):
        raise GMTAdmissionError("trace control-time clock mismatch")
    return arrays


def load_validated_replay(
    *,
    trace_path: Path,
    manifest_path: Path,
    manifest_sha256: str,
    upstream_root: Path,
) -> tuple[dict[str, Any], ModelABI, dict[str, np.ndarray], dict[str, str]]:
    """Load one replay chain without importing or executing MuJoCo."""

    manifest = read_trace_manifest(manifest_path, expected_sha256=manifest_sha256)
    support_files = verify_upstream_root(upstream_root)
    abi, contract = validate_trace_manifest(
        manifest,
        trace_path=trace_path,
        support_files=support_files,
    )
    arrays = read_trace_arrays(
        trace_path,
        expected_sha256=manifest["trace"]["sha256"],
        contract=contract,
    )
    return manifest, abi, arrays, support_files
