"""Exact parity check between the shared GMT runtime and its retained replay."""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .contracts import ACTION_DIM, GMT_UPSTREAM_COMMIT, SIMULATION_DECIMATION
from .control_runtime import G1ControlRuntime, GMTActorSession, compose_residual_raw_action
from .io import GMTAdmissionError, write_json_receipt
from .reference_runtime import ReferenceMotion
from .reference_sensitivity import validate_probe_identities
from .trace_admission import load_validated_replay

CONTROL_STEPS = 500
SIMULATION_STEPS = CONTROL_STEPS * SIMULATION_DECIMATION
CLAIM_STATUS = "shared_runtime_exact_parity_to_retained_reconstructed_baseline"


@dataclass(frozen=True, slots=True)
class RuntimeParityConfig:
    trace_path: Path
    trace_sha256: str
    manifest_path: Path
    manifest_sha256: str
    upstream_root: Path
    weights_path: Path
    weights_sha256: str
    motion_path: Path
    motion_sha256: str
    motion_name: str
    output_path: Path


@dataclass(slots=True)
class _ExactComparison:
    chunks_compared: int = 0
    values_compared: int = 0
    exact_chunks: int = 0
    exact_values: int = 0
    max_abs_difference: float = 0.0

    def compare(self, observed: object, expected: object, *, name: str) -> None:
        left = np.asarray(observed)
        right = np.asarray(expected)
        if left.shape != right.shape or left.dtype.str != right.dtype.str:
            raise GMTAdmissionError(
                f"runtime parity {name} shape/dtype mismatch: "
                f"{left.shape}/{left.dtype.str} != {right.shape}/{right.dtype.str}"
            )
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            raise GMTAdmissionError(f"runtime parity {name} contains non-finite values")
        left = np.ascontiguousarray(left)
        right = np.ascontiguousarray(right)
        byte_width = left.dtype.itemsize
        equal_values = np.all(
            left.view(np.uint8).reshape(-1, byte_width)
            == right.view(np.uint8).reshape(-1, byte_width),
            axis=1,
        )
        exact = int(np.count_nonzero(equal_values))
        delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
        maximum = float(np.max(delta)) if delta.size else 0.0
        self.chunks_compared += 1
        self.values_compared += left.size
        self.exact_chunks += int(exact == left.size)
        self.exact_values += exact
        self.max_abs_difference = max(self.max_abs_difference, maximum)
        if exact != left.size:
            raise GMTAdmissionError(
                f"runtime parity divergence for {name}: {left.size - exact}/{left.size} "
                f"values differ exactly; max absolute difference={maximum:.17g}"
            )

    def receipt(self) -> dict[str, int | float]:
        return {
            "chunks_compared": self.chunks_compared,
            "values_compared": self.values_compared,
            "exact_chunks": self.exact_chunks,
            "exact_values": self.exact_values,
            "max_abs_difference": self.max_abs_difference,
        }


def _counter(value: object, expected: int, *, name: str) -> None:
    if type(value) is not int or value != expected:
        raise GMTAdmissionError(f"runtime parity {name} mismatch: {value!r} != {expected}")


def _measure_runtime_parity(
    *,
    trace: Mapping[str, np.ndarray],
    runtime: Any,
    session: Any,
    reference_window: Callable[[int], np.ndarray],
) -> dict[str, object]:
    """Run exact comparisons; test doubles may supply the three runtime interfaces."""

    control_steps = int(trace["control_raw_action"].shape[0])
    simulation_steps = int(trace["sim_qpos"].shape[0])
    if simulation_steps != control_steps * SIMULATION_DECIMATION:
        raise GMTAdmissionError("runtime parity requires complete 20-substep control intervals")
    comparisons: dict[str, _ExactComparison] = {
        name: _ExactComparison()
        for name in (
            "boundary_qpos",
            "boundary_qvel",
            "boundary_orientation_wxyz",
            "boundary_angular_velocity",
            "reference_window",
            "prepared_proprio",
            "prepared_obs",
            "base_raw_action",
            "zero_residual_composite",
            "interval_composite_raw_action",
            "clipped_action",
            "pd_target",
            "applied_torque",
            "post_substep_qpos",
            "post_substep_qvel",
        )
    }

    session.reset()
    boundary = runtime.reset()
    zero_residual = np.zeros(ACTION_DIM, dtype="<f4")
    matched_poststates = 0
    for control_step in range(control_steps):
        simulation_step = control_step * SIMULATION_DECIMATION
        _counter(boundary.control_step, control_step, name="boundary control_step")
        _counter(boundary.simulation_step, simulation_step, name="boundary simulation_step")
        comparisons["boundary_qpos"].compare(
            boundary.qpos, trace["sim_qpos"][simulation_step], name="boundary qpos"
        )
        comparisons["boundary_qvel"].compare(
            boundary.qvel, trace["sim_qvel"][simulation_step], name="boundary qvel"
        )
        comparisons["boundary_orientation_wxyz"].compare(
            boundary.orientation_wxyz,
            trace["control_orientation_wxyz"][control_step],
            name="boundary orientation",
        )
        comparisons["boundary_angular_velocity"].compare(
            boundary.angular_velocity,
            trace["control_angular_velocity"][control_step],
            name="boundary angular velocity",
        )

        window = np.ascontiguousarray(reference_window(control_step))
        comparisons["reference_window"].compare(
            window, trace["control_reference_window"][control_step], name="reference window"
        )
        prepared = session.prepare(boundary, window)
        _counter(prepared.control_step, control_step, name="prepared control_step")
        comparisons["prepared_proprio"].compare(
            prepared.proprio,
            trace["control_proprioception"][control_step],
            name="prepared proprio",
        )
        comparisons["prepared_obs"].compare(
            prepared.obs, trace["control_observation"][control_step], name="prepared obs"
        )
        comparisons["base_raw_action"].compare(
            prepared.base_raw,
            trace["control_raw_action"][control_step],
            name="base raw action",
        )
        composite = compose_residual_raw_action(prepared.base_raw, zero_residual)
        comparisons["zero_residual_composite"].compare(
            composite, prepared.base_raw, name="zero-residual composite"
        )

        next_boundary, interval = runtime.step(composite)
        _counter(interval.start_control_step, control_step, name="interval control_step")
        _counter(interval.start_simulation_step, simulation_step, name="interval simulation_step")
        comparisons["interval_composite_raw_action"].compare(
            interval.composite_raw_action,
            trace["control_raw_action"][control_step],
            name="interval composite raw action",
        )
        comparisons["clipped_action"].compare(
            interval.clipped_action,
            trace["control_clipped_action"][control_step],
            name="clipped action",
        )
        comparisons["pd_target"].compare(
            interval.pd_target,
            trace["control_pd_target"][control_step],
            name="PD target",
        )
        comparisons["applied_torque"].compare(
            interval.torque_by_substep,
            trace["sim_torque"][simulation_step : simulation_step + SIMULATION_DECIMATION],
            name="applied torque substeps",
        )

        available = min(SIMULATION_DECIMATION, simulation_steps - simulation_step - 1)
        shifted = slice(simulation_step + 1, simulation_step + 1 + available)
        comparisons["post_substep_qpos"].compare(
            interval.qpos_after_substep[:available],
            trace["sim_qpos"][shifted],
            name="post-substep qpos shifted by +1",
        )
        comparisons["post_substep_qvel"].compare(
            interval.qvel_after_substep[:available],
            trace["sim_qvel"][shifted],
            name="post-substep qvel shifted by +1",
        )
        matched_poststates += available
        session.commit(prepared, composite)
        boundary = next_boundary

    _counter(boundary.control_step, control_steps, name="final boundary control_step")
    _counter(boundary.simulation_step, simulation_steps, name="final boundary simulation_step")
    excluded = control_steps * SIMULATION_DECIMATION - matched_poststates
    if excluded != 1:
        raise GMTAdmissionError(
            f"runtime parity expected one unavailable final poststate, got {excluded}"
        )
    return {
        "control_steps": control_steps,
        "simulation_steps": simulation_steps,
        "matched_runtime_poststate_samples": matched_poststates,
        "unavailable_baseline_final_poststate_samples": excluded,
        "comparisons": {name: value.receipt() for name, value in comparisons.items()},
    }


def run_runtime_parity(config: RuntimeParityConfig) -> dict[str, object]:
    """Publish exact zero-residual runtime parity against one admitted replay."""

    output_path = Path(config.output_path)
    if output_path.suffix != ".json":
        raise ValueError("runtime-parity output must use the .json suffix")
    if output_path.exists():
        raise GMTAdmissionError("refusing to overwrite runtime-parity receipt")
    manifest, replay_abi, trace, support_files = load_validated_replay(
        trace_path=Path(config.trace_path),
        manifest_path=Path(config.manifest_path),
        manifest_sha256=config.manifest_sha256,
        upstream_root=Path(config.upstream_root),
    )
    validate_probe_identities(
        manifest,
        trace_sha256=config.trace_sha256,
        weights_sha256=config.weights_sha256,
        motion_name=config.motion_name,
        motion_sha256=config.motion_sha256,
    )
    if trace["control_raw_action"].shape[0] != CONTROL_STEPS:
        raise GMTAdmissionError("runtime parity requires the retained 500-control-step replay")
    if trace["sim_qpos"].shape[0] != SIMULATION_STEPS:
        raise GMTAdmissionError("runtime parity requires the retained 10000-substep replay")

    runtime = G1ControlRuntime(Path(config.upstream_root))
    if runtime.model_abi != replay_abi or runtime.support_files != support_files:
        raise GMTAdmissionError("shared runtime model/support identity differs from the replay")
    session = GMTActorSession(Path(config.weights_path), expected_sha256=config.weights_sha256)
    motion = ReferenceMotion.from_converted(
        Path(config.motion_path),
        name=config.motion_name,
        expected_sha256=config.motion_sha256,
    )
    measurement = _measure_runtime_parity(
        trace=trace,
        runtime=runtime,
        session=session,
        reference_window=lambda step: motion.window(step).detach().cpu().numpy(),
    )
    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact": "gmt_g1_shared_runtime_zero_residual_parity",
        "claim_status": CLAIM_STATUS,
        "inputs": {
            "upstream_commit": GMT_UPSTREAM_COMMIT,
            "trace": {"path": Path(config.trace_path).name, "sha256": config.trace_sha256},
            "replay_manifest": {
                "path": Path(config.manifest_path).name,
                "sha256": config.manifest_sha256,
            },
            "actor": {"path": Path(config.weights_path).name, "sha256": config.weights_sha256},
            "motion": {
                "path": Path(config.motion_path).name,
                "name": config.motion_name,
                "sha256": config.motion_sha256,
            },
            "support_files": support_files,
            "model_abi": asdict(replay_abi),
        },
        "comparison_contract": {
            "numeric_tolerance": 0,
            "equality": "shape, dtype, and each scalar's exact bytes",
            "boundary_mapping": "control k uses retained pre-step simulation sample 20*k",
            "torque_mapping": "control k uses retained torque samples 20*k through 20*k+19",
            "poststate_mapping": "runtime substep j uses retained pre-step sample 20*k+j+1",
            "final_poststate": "not compared because retained pre-step trace has no sample 10000",
            "zero_residual": "float32 base + float32(0.25) * float32 zeros",
        },
        "measurement": measurement,
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
        },
        "limits": {
            "original_jit_executed": False,
            "jit_equivalence_tested": False,
            "runtime_parity_only": True,
            "task_competence_tested": False,
            "reference_composition_tested": False,
            "learning_performed": False,
        },
    }
    artifact_sha256 = write_json_receipt(output_path, receipt)
    return {
        "artifact_path": str(output_path),
        "artifact_sha256": artifact_sha256,
        "claim_status": CLAIM_STATUS,
        "measurement": measurement,
    }


__all__ = ["RuntimeParityConfig", "run_runtime_parity"]
