"""Bounded headless MuJoCo replay for the reconstructed GMT G1 actor."""

from __future__ import annotations

import math
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .actor import load_actor
from .checkpoint import verify_upstream_root
from .contracts import (
    ACTION_DIM,
    ACTION_SCALE,
    CONTROL_DT_SECONDS,
    DEFAULT_DOF_POSITION,
    GMT_UPSTREAM_COMMIT,
    HISTORY_LENGTH,
    JOINT_NAMES,
    OBSERVATION_DIM,
    PROPRIOCEPTION_DIM,
    RAW_ACTION_MAX,
    RAW_ACTION_MIN,
    REFERENCE_FRAME_DIM,
    REFERENCE_HORIZON,
    REFERENCE_OFFSETS,
    SIMULATION_DECIMATION,
    SIMULATION_DT_SECONDS,
    SURVIVAL_ROOT_HEIGHT_MIN,
    TORQUE_LIMITS,
)
from .deployment import (
    ObservationHistory,
    apply_pd_control,
    build_proprioception,
    pd_torque,
    reference_tracking_errors,
)
from .io import GMTAdmissionError, write_deterministic_npz, write_json_receipt
from .reference_math import quaternion_to_euler_wxyz
from .reference_runtime import ReferenceMotion


@dataclass(frozen=True)
class ReplayConfig:
    upstream_root: Path
    weights_path: Path
    weights_sha256: str
    motion_path: Path
    motion_sha256: str
    motion_name: str
    trace_path: Path
    duration_seconds: float = 10.0
    survival_root_height_min: float = SURVIVAL_ROOT_HEIGHT_MIN


@dataclass(frozen=True)
class ModelABI:
    nq: int
    nv: int
    nu: int
    joint_names: tuple[str, ...]
    joint_types: tuple[str, ...]
    actuator_names: tuple[str, ...]
    actuator_joint_ids: tuple[int, ...]
    sensor_names: tuple[str, ...]
    sensor_types: tuple[str, ...]
    sensor_dimensions: tuple[int, ...]
    keyframe_count: int
    home_qpos: tuple[float, ...]
    timestep: float


def validate_model_abi(abi: ModelABI) -> None:
    expected_joint_names = ("pelvis", *(f"{name}_joint" for name in JOINT_NAMES))
    expected_actuator_names = tuple(f"{name}_joint" for name in JOINT_NAMES)
    expected_home = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, *DEFAULT_DOF_POSITION)
    failures: list[str] = []
    if (abi.nq, abi.nv, abi.nu) != (30, 29, ACTION_DIM):
        failures.append(f"nq/nv/nu={abi.nq}/{abi.nv}/{abi.nu}")
    if abi.joint_names != expected_joint_names:
        failures.append("joint names/order")
    if abi.joint_types != ("free", *("hinge" for _ in JOINT_NAMES)):
        failures.append("joint types/order")
    if abi.actuator_names != expected_actuator_names:
        failures.append("actuator names/order")
    if abi.actuator_joint_ids != tuple(range(1, ACTION_DIM + 1)):
        failures.append("actuator-to-joint mapping")
    if abi.sensor_names != ("orientation", "position", "angular-velocity"):
        failures.append("sensor names/order")
    if abi.sensor_types != ("framequat", "framepos", "gyro"):
        failures.append("sensor types/order")
    if abi.sensor_dimensions != (4, 3, 3):
        failures.append("sensor dimensions/order")
    if abi.keyframe_count != 1:
        failures.append(f"keyframe count={abi.keyframe_count}")
    if len(abi.home_qpos) != 30 or not np.allclose(abi.home_qpos, expected_home, atol=0, rtol=0):
        failures.append("home keyframe qpos")
    if abi.timestep != SIMULATION_DT_SECONDS:
        failures.append(f"timestep={abi.timestep}")
    if failures:
        raise GMTAdmissionError(f"G1 MuJoCo ABI mismatch: {', '.join(failures)}")


def _extract_model_abi(mujoco: Any, model: Any) -> ModelABI:
    name = mujoco.mj_id2name
    joint_names = tuple(
        name(model, mujoco.mjtObj.mjOBJ_JOINT, index) for index in range(model.njnt)
    )
    actuator_names = tuple(
        name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) for index in range(model.nu)
    )
    sensor_names = tuple(
        name(model, mujoco.mjtObj.mjOBJ_SENSOR, index) for index in range(model.nsensor)
    )
    joint_type_names = {
        int(mujoco.mjtJoint.mjJNT_FREE): "free",
        int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
    }
    sensor_type_names = {
        int(mujoco.mjtSensor.mjSENS_FRAMEQUAT): "framequat",
        int(mujoco.mjtSensor.mjSENS_FRAMEPOS): "framepos",
        int(mujoco.mjtSensor.mjSENS_GYRO): "gyro",
    }
    return ModelABI(
        nq=int(model.nq),
        nv=int(model.nv),
        nu=int(model.nu),
        joint_names=joint_names,
        joint_types=tuple(
            joint_type_names.get(int(value), f"unknown:{value}") for value in model.jnt_type
        ),
        actuator_names=actuator_names,
        actuator_joint_ids=tuple(int(value) for value in model.actuator_trnid[:, 0]),
        sensor_names=sensor_names,
        sensor_types=tuple(
            sensor_type_names.get(int(value), f"unknown:{value}") for value in model.sensor_type
        ),
        sensor_dimensions=tuple(int(value) for value in model.sensor_dim),
        keyframe_count=int(model.nkey),
        home_qpos=tuple(float(value) for value in model.key_qpos[0]) if model.nkey else (),
        timestep=float(model.opt.timestep),
    )


def _validate_config(config: ReplayConfig) -> tuple[int, int]:
    if not math.isfinite(config.duration_seconds) or not 0 < config.duration_seconds <= 10.0:
        raise ValueError("GMT replay duration must be finite and in (0, 10] seconds")
    simulation_steps = round(config.duration_seconds / SIMULATION_DT_SECONDS)
    if not math.isclose(
        simulation_steps * SIMULATION_DT_SECONDS,
        config.duration_seconds,
        rel_tol=0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("GMT replay duration must be an integer number of 1 ms steps")
    if not math.isfinite(config.survival_root_height_min):
        raise ValueError("survival root-height threshold must be finite")
    if config.trace_path.suffix != ".npz":
        raise ValueError("GMT trace output must use the .npz suffix")
    manifest_path = config.trace_path.with_suffix(".npz.manifest.json")
    if config.trace_path.exists() or manifest_path.exists():
        raise GMTAdmissionError("refusing to overwrite existing replay output")
    control_steps = (simulation_steps + SIMULATION_DECIMATION - 1) // SIMULATION_DECIMATION
    return simulation_steps, control_steps


def _allocate_traces(simulation_steps: int, control_steps: int) -> dict[str, np.ndarray]:
    return {
        "sim_time": np.arange(simulation_steps, dtype=np.float64) * SIMULATION_DT_SECONDS,
        "sim_qpos": np.empty((simulation_steps, 30), dtype=np.float64),
        "sim_qvel": np.empty((simulation_steps, 29), dtype=np.float64),
        "sim_pd_target": np.empty((simulation_steps, ACTION_DIM), dtype=np.float64),
        "sim_torque": np.empty((simulation_steps, ACTION_DIM), dtype=np.float64),
        "sim_contact_count": np.empty((simulation_steps,), dtype=np.int32),
        "sim_torque_saturation_fraction": np.empty((simulation_steps,), dtype=np.float32),
        "control_sim_step": np.empty((control_steps,), dtype=np.int64),
        "control_time": np.empty((control_steps,), dtype=np.float64),
        "control_orientation_wxyz": np.empty((control_steps, 4), dtype=np.float32),
        "control_angular_velocity": np.empty((control_steps, 3), dtype=np.float32),
        "control_reference_window": np.empty(
            (control_steps, REFERENCE_HORIZON, REFERENCE_FRAME_DIM), dtype=np.float32
        ),
        "control_reference_current": np.empty(
            (control_steps, REFERENCE_FRAME_DIM), dtype=np.float32
        ),
        "control_proprioception": np.empty((control_steps, PROPRIOCEPTION_DIM), dtype=np.float64),
        "control_observation": np.empty((control_steps, OBSERVATION_DIM), dtype=np.float32),
        "control_raw_action": np.empty((control_steps, ACTION_DIM), dtype=np.float32),
        "control_clipped_action": np.empty((control_steps, ACTION_DIM), dtype=np.float32),
        "control_pd_target": np.empty((control_steps, ACTION_DIM), dtype=np.float64),
        "control_joint_rmse": np.empty((control_steps,), dtype=np.float64),
        "control_root_height_abs_error": np.empty((control_steps,), dtype=np.float64),
        "control_roll_pitch_rmse": np.empty((control_steps,), dtype=np.float64),
        "control_action_saturation_fraction": np.empty((control_steps,), dtype=np.float32),
    }


def _summary(
    traces: dict[str, np.ndarray],
    *,
    config: ReplayConfig,
    wall_seconds: float,
) -> dict[str, object]:
    root_height = traces["sim_qpos"][:, 2]
    survived = root_height >= config.survival_root_height_min
    first_failure = np.flatnonzero(~survived)
    survival_seconds = (
        config.duration_seconds
        if first_failure.size == 0
        else float(first_failure[0] * SIMULATION_DT_SECONDS)
    )
    return {
        "simulated_seconds": config.duration_seconds,
        "wall_seconds": wall_seconds,
        "realtime_factor": config.duration_seconds / wall_seconds,
        "survival_root_height_min": config.survival_root_height_min,
        "survived_full_duration": bool(survived.all()),
        "survival_seconds": survival_seconds,
        "root_height": {
            "minimum": float(root_height.min()),
            "mean": float(root_height.mean()),
            "final_pre_step": float(root_height[-1]),
        },
        "tracking": {
            "joint_rmse_mean": float(traces["control_joint_rmse"].mean()),
            "joint_rmse_final": float(traces["control_joint_rmse"][-1]),
            "root_height_abs_error_mean": float(traces["control_root_height_abs_error"].mean()),
            "roll_pitch_rmse_mean": float(traces["control_roll_pitch_rmse"].mean()),
        },
        "saturation": {
            "raw_action_fraction_mean": float(traces["control_action_saturation_fraction"].mean()),
            "torque_fraction_mean": float(traces["sim_torque_saturation_fraction"].mean()),
        },
    }


def run_headless_replay(config: ReplayConfig) -> dict[str, object]:
    """Run a maximum-ten-second local replay; importing this module does not start MuJoCo."""

    simulation_steps, control_steps = _validate_config(config)
    support_files = verify_upstream_root(config.upstream_root)
    actor = load_actor(
        config.weights_path,
        expected_sha256=config.weights_sha256,
        freeze=True,
    )
    motion = ReferenceMotion.from_converted(
        config.motion_path,
        name=config.motion_name,
        expected_sha256=config.motion_sha256,
    )
    try:
        import mujoco
    except ImportError as exc:
        raise RuntimeError("MuJoCo is required only for the authorized replay command") from exc

    xml_path = config.upstream_root / "assets/robots/g1/g1.xml"
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    abi = _extract_model_abi(mujoco, model)
    validate_model_abi(abi)
    model.opt.timestep = SIMULATION_DT_SECONDS
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_step(model, data)

    traces = _allocate_traces(simulation_steps, control_steps)
    history = ObservationHistory()
    last_raw_action = np.zeros(ACTION_DIM, dtype=np.float32)
    pd_target = np.asarray(DEFAULT_DOF_POSITION, dtype=np.float64)
    torque_limits = np.asarray(TORQUE_LIMITS)
    control_index = 0
    started = time.perf_counter()
    with torch.no_grad():
        for simulation_step in range(simulation_steps):
            qpos = np.asarray(data.qpos).copy()
            qvel = np.asarray(data.qvel).copy()
            dof_position = qpos[-ACTION_DIM:].astype(np.float32)
            dof_velocity = qvel[-ACTION_DIM:].astype(np.float32)
            traces["sim_qpos"][simulation_step] = qpos
            traces["sim_qvel"][simulation_step] = qvel
            traces["sim_contact_count"][simulation_step] = data.ncon

            if simulation_step % SIMULATION_DECIMATION == 0:
                orientation = np.asarray(data.sensor("orientation").data, dtype=np.float32).copy()
                angular_velocity = np.asarray(
                    data.sensor("angular-velocity").data, dtype=np.float32
                ).copy()
                reference_window = motion.window(control_index).numpy()
                reference_current = motion.current(control_index).numpy()
                proprioception = build_proprioception(
                    dof_position=dof_position,
                    dof_velocity=dof_velocity,
                    orientation_wxyz=orientation,
                    angular_velocity=angular_velocity,
                    last_raw_action=last_raw_action,
                )
                observation = history.assemble(reference_window, proprioception)
                raw_action = actor(torch.from_numpy(observation).unsqueeze(0)).numpy()[0]
                if not np.isfinite(raw_action).all():
                    raise RuntimeError(f"non-finite actor output at control step {control_index}")
                control = apply_pd_control(
                    raw_action=raw_action,
                    dof_position=dof_position,
                    dof_velocity=dof_velocity,
                )
                last_raw_action = control.raw_history_action
                pd_target = control.pd_target
                history.append(proprioception)
                roll_pitch = quaternion_to_euler_wxyz(orientation)[:2]
                tracking = reference_tracking_errors(
                    dof_position=dof_position,
                    root_height=float(qpos[2]),
                    roll_pitch=roll_pitch,
                    current_reference=reference_current,
                )
                traces["control_sim_step"][control_index] = simulation_step
                traces["control_time"][control_index] = simulation_step * SIMULATION_DT_SECONDS
                traces["control_orientation_wxyz"][control_index] = orientation
                traces["control_angular_velocity"][control_index] = angular_velocity
                traces["control_reference_window"][control_index] = reference_window
                traces["control_reference_current"][control_index] = reference_current
                traces["control_proprioception"][control_index] = proprioception
                traces["control_observation"][control_index] = observation
                traces["control_raw_action"][control_index] = raw_action
                traces["control_clipped_action"][control_index] = control.clipped_action
                traces["control_pd_target"][control_index] = pd_target
                traces["control_joint_rmse"][control_index] = tracking[0]
                traces["control_root_height_abs_error"][control_index] = tracking[1]
                traces["control_roll_pitch_rmse"][control_index] = tracking[2]
                traces["control_action_saturation_fraction"][control_index] = np.mean(
                    np.abs(raw_action) >= 10.0
                )
                control_index += 1

            torque = pd_torque(
                pd_target=pd_target,
                dof_position=dof_position,
                dof_velocity=dof_velocity,
            )
            traces["sim_pd_target"][simulation_step] = pd_target
            traces["sim_torque"][simulation_step] = torque
            traces["sim_torque_saturation_fraction"][simulation_step] = np.mean(
                np.abs(torque) >= torque_limits
            )
            data.ctrl[:] = torque
            mujoco.mj_step(model, data)
    wall_seconds = time.perf_counter() - started
    if control_index != control_steps:
        raise RuntimeError(f"control trace count mismatch: {control_index} != {control_steps}")

    trace_sha256 = write_deterministic_npz(config.trace_path, traces)
    summary = _summary(traces, config=config, wall_seconds=wall_seconds)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "artifact": "gmt_g1_reconstructed_actor_headless_replay",
        "claim_status": "reconstructed_plain_actor_not_jit_equivalent",
        "inputs": {
            "upstream_commit": GMT_UPSTREAM_COMMIT,
            "weights_sha256": config.weights_sha256,
            "motion_name": config.motion_name,
            "motion_sha256": config.motion_sha256,
            "support_files": support_files,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
            "mujoco": mujoco.__version__,
            "simulation_dt_seconds": SIMULATION_DT_SECONDS,
            "control_dt_seconds": CONTROL_DT_SECONDS,
            "state_sampling": "pre_step",
            "reference_metric_alignment": "current_time",
            "initial_mujoco_warmup_steps": 1,
        },
        "controller_abi": {
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
        },
        "model_abi": asdict(abi),
        "trace": {
            "path": config.trace_path.name,
            "sha256": trace_sha256,
            "size": config.trace_path.stat().st_size,
            "arrays": {
                key: {"shape": list(value.shape), "dtype": value.dtype.str}
                for key, value in traces.items()
            },
        },
        "metrics": summary,
        "limits": {
            "original_jit_executed": False,
            "jit_equivalence_tested": False,
            "preview_generated": False,
            "survival_is_root_height_threshold_proxy": True,
        },
    }
    manifest_path = config.trace_path.with_suffix(".npz.manifest.json")
    manifest_sha256 = write_json_receipt(manifest_path, manifest)
    return {
        "trace_path": str(config.trace_path),
        "trace_sha256": trace_sha256,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "metrics": summary,
        "claim_status": manifest["claim_status"],
    }
