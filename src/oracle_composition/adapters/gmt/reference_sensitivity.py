"""Matched-state GMT reference-input sensitivity measurement."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .actor import load_actor
from .contracts import (
    ACTION_DIM,
    CONTROL_DT_SECONDS,
    GMT_UPSTREAM_COMMIT,
    HISTORY_LENGTH,
    OBSERVATION_DIM,
    PROPRIOCEPTION_DIM,
    PROPRIOCEPTION_HISTORY_SLICE,
    REFERENCE_FRAME_DIM,
    REFERENCE_HORIZON,
    REFERENCE_SLICE,
    SIMULATION_DECIMATION,
)
from .io import GMTAdmissionError, write_deterministic_npz, write_json_receipt
from .reference_runtime import ReferenceMotion
from .trace_admission import load_validated_replay

SHIFT_CONTROL_STEPS = 250
CLAIM_STATUS = "matched_state_reference_input_influence_only"
HASH_ENCODING = "sha256(json_dtype_shape + newline + exact_c_order_bytes)"

ActorEvaluator = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ReferenceSensitivityConfig:
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
    shuffle_seed: int
    output_path: Path


@dataclass(frozen=True)
class SensitivityMeasurement:
    arrays: dict[str, np.ndarray]
    observation_identities: dict[str, dict[str, object]]
    action_summaries: dict[str, dict[str, float | int]]
    permutation: tuple[int, ...]


def numeric_array_sha256(value: np.ndarray) -> str:
    """Hash exact numeric bytes together with their shape and dtype."""

    array = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\n" + array.tobytes(order="C")).hexdigest()


def _float32_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.str != "<f4" or not array.flags.c_contiguous:
        raise GMTAdmissionError(f"{name} must be contiguous little-endian float32 {shape}")
    if not np.isfinite(array).all():
        raise GMTAdmissionError(f"{name} contains non-finite values")
    return array


def _same_bytes(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.shape == right.shape
        and left.dtype.str == right.dtype.str
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _validate_observation_lineage(
    *,
    observations: np.ndarray,
    recorded_reference_windows: np.ndarray,
    admitted_reference_windows: np.ndarray,
    recorded_proprioception: np.ndarray,
) -> None:
    rows = observations.shape[0]
    recorded_flat = recorded_reference_windows.reshape(rows, -1)
    admitted_flat = admitted_reference_windows.reshape(rows, -1)
    if not _same_bytes(observations[:, REFERENCE_SLICE], recorded_flat):
        raise GMTAdmissionError("trace observation/reference-window bytes differ")
    if not _same_bytes(recorded_flat, admitted_flat):
        raise GMTAdmissionError("trace reference windows differ from the admitted motion runtime")

    proprioception = np.ascontiguousarray(recorded_proprioception.astype("<f4"))
    current = observations[:, REFERENCE_SLICE.stop : PROPRIOCEPTION_HISTORY_SLICE.start]
    if not _same_bytes(current, proprioception):
        raise GMTAdmissionError("trace observation/current-proprioception bytes differ")
    expected_history = np.zeros((rows, HISTORY_LENGTH, PROPRIOCEPTION_DIM), dtype="<f4")
    for row in range(rows):
        previous = proprioception[max(0, row - HISTORY_LENGTH) : row]
        expected_history[row, HISTORY_LENGTH - previous.shape[0] :] = previous
    history = observations[:, PROPRIOCEPTION_HISTORY_SLICE].reshape(
        rows, HISTORY_LENGTH, PROPRIOCEPTION_DIM
    )
    if not _same_bytes(history, expected_history):
        raise GMTAdmissionError("trace observation/history bytes differ")


def _evaluate(actor: ActorEvaluator, observations: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(actor(observations.copy()))
    expected_shape = (observations.shape[0], ACTION_DIM)
    if result.shape != expected_shape or result.dtype.str != "<f4":
        raise GMTAdmissionError(f"{name} actor output must be little-endian float32 {expected_shape}")
    if not result.flags.c_contiguous or not np.isfinite(result).all():
        raise GMTAdmissionError(f"{name} actor output must be finite and contiguous")
    return result


def _action_summary(exact: np.ndarray, intervention: np.ndarray) -> dict[str, float | int]:
    delta = intervention.astype(np.float64) - exact.astype(np.float64)
    l2 = np.linalg.norm(delta, axis=1)
    linf = np.max(np.abs(delta), axis=1)
    changed = np.any(
        exact.view(np.uint8).reshape(exact.shape[0], -1)
        != intervention.view(np.uint8).reshape(intervention.shape[0], -1),
        axis=1,
    )
    return {
        "rows": int(exact.shape[0]),
        "bitwise_changed_rows": int(np.count_nonzero(changed)),
        "mean_action_delta_l2": float(np.mean(l2)),
        "max_action_delta_l2": float(np.max(l2)),
        "mean_action_delta_linf": float(np.mean(linf)),
        "max_action_delta_linf": float(np.max(linf)),
    }


def _identity(value: np.ndarray) -> dict[str, object]:
    return {
        "sha256": numeric_array_sha256(value),
        "shape": list(value.shape),
        "dtype": value.dtype.str,
    }


def measure_reference_sensitivity(
    *,
    observations: np.ndarray,
    recorded_actions: np.ndarray,
    recorded_reference_windows: np.ndarray,
    admitted_reference_windows: np.ndarray,
    shifted_reference_windows: np.ndarray,
    recorded_proprioception: np.ndarray,
    control_steps: np.ndarray,
    shuffle_seed: int,
    actor: ActorEvaluator,
) -> SensitivityMeasurement:
    """Change reference inputs while holding every recorded non-reference byte fixed."""

    observations = np.asarray(observations)
    if observations.ndim != 2 or observations.shape[1] != OBSERVATION_DIM:
        raise GMTAdmissionError(f"observations must have shape [rows, {OBSERVATION_DIM}]")
    rows = observations.shape[0]
    if rows < 1:
        raise GMTAdmissionError("reference sensitivity needs at least one recorded state")
    observations = _float32_array(observations, (rows, OBSERVATION_DIM), "observations")
    recorded_actions = _float32_array(
        recorded_actions, (rows, ACTION_DIM), "recorded actions"
    )
    reference_shape = (rows, REFERENCE_HORIZON, REFERENCE_FRAME_DIM)
    recorded_reference_windows = _float32_array(
        recorded_reference_windows, reference_shape, "recorded reference windows"
    )
    admitted_reference_windows = _float32_array(
        admitted_reference_windows, reference_shape, "admitted reference windows"
    )
    shifted_reference_windows = _float32_array(
        shifted_reference_windows, reference_shape, "shifted reference windows"
    )
    recorded_proprioception = np.asarray(recorded_proprioception)
    if (
        recorded_proprioception.shape != (rows, PROPRIOCEPTION_DIM)
        or recorded_proprioception.dtype.str != "<f8"
        or not recorded_proprioception.flags.c_contiguous
        or not np.isfinite(recorded_proprioception).all()
    ):
        raise GMTAdmissionError(
            f"recorded proprioception must be contiguous little-endian float64 "
            f"({rows}, {PROPRIOCEPTION_DIM})"
        )
    control_steps = np.asarray(control_steps)
    if (
        control_steps.shape != (rows,)
        or control_steps.dtype.str != "<i8"
        or not control_steps.flags.c_contiguous
        or np.any(control_steps < 0)
    ):
        raise GMTAdmissionError(f"control steps must be contiguous little-endian int64 ({rows},)")
    _validate_observation_lineage(
        observations=observations,
        recorded_reference_windows=recorded_reference_windows,
        admitted_reference_windows=admitted_reference_windows,
        recorded_proprioception=recorded_proprioception,
    )

    generator = np.random.Generator(np.random.PCG64(shuffle_seed))
    permutation = np.ascontiguousarray(
        generator.permutation(REFERENCE_HORIZON).astype("<i8", copy=False)
    )
    exact_input = observations.copy()
    zero_input = observations.copy()
    zero_input[:, REFERENCE_SLICE] = 0.0
    shuffled_input = observations.copy()
    shuffled_reference = recorded_reference_windows[:, permutation, :]
    shuffled_input[:, REFERENCE_SLICE] = shuffled_reference.reshape(rows, -1)
    shifted_input = observations.copy()
    shifted_input[:, REFERENCE_SLICE] = shifted_reference_windows.reshape(rows, -1)
    inputs = {
        "exact": exact_input,
        "zero_reference": zero_input,
        "shuffled_reference": shuffled_input,
        "shifted_reference": shifted_input,
    }
    suffix = observations[:, REFERENCE_SLICE.stop :]
    for name, value in inputs.items():
        if not _same_bytes(value[:, REFERENCE_SLICE.stop :], suffix):
            raise RuntimeError(f"{name} changed a non-reference observation byte")

    recomputed = _evaluate(actor, exact_input, name="exact")
    if not _same_bytes(recomputed, recorded_actions):
        raise GMTAdmissionError("recomputed exact actions differ from retained raw action bytes")
    action_arrays = {
        "recorded_raw_action": recorded_actions.copy(),
        "recomputed_exact_action": recomputed,
        "zero_reference_action": _evaluate(actor, zero_input, name="zero-reference"),
        "shuffled_reference_action": _evaluate(
            actor, shuffled_input, name="shuffled-reference"
        ),
        "shifted_reference_action": _evaluate(actor, shifted_input, name="shifted-reference"),
    }
    output_arrays = {
        "control_step": control_steps.copy(),
        "shifted_source_control_step": np.ascontiguousarray(
            (control_steps + SHIFT_CONTROL_STEPS).astype("<i8", copy=False)
        ),
        "shuffle_permutation": permutation,
        **action_arrays,
    }
    return SensitivityMeasurement(
        arrays=output_arrays,
        observation_identities={name: _identity(value) for name, value in inputs.items()},
        action_summaries={
            name: _action_summary(recomputed, action_arrays[f"{name}_action"])
            for name in ("zero_reference", "shuffled_reference", "shifted_reference")
        },
        permutation=tuple(int(value) for value in permutation),
    )


def validate_probe_identities(
    manifest: Mapping[str, Any],
    *,
    trace_sha256: str,
    weights_sha256: str,
    motion_name: str,
    motion_sha256: str,
) -> None:
    """Bind the probe to the externally selected trace, actor, and motion identities."""

    trace = manifest.get("trace")
    inputs = manifest.get("inputs")
    if not isinstance(trace, Mapping) or trace.get("sha256") != trace_sha256:
        raise GMTAdmissionError("selected trace SHA-256 differs from the replay manifest")
    if not isinstance(inputs, Mapping) or inputs.get("weights_sha256") != weights_sha256:
        raise GMTAdmissionError("selected actor SHA-256 differs from the replay manifest")
    if inputs.get("motion_name") != motion_name or inputs.get("motion_sha256") != motion_sha256:
        raise GMTAdmissionError("selected motion identity differs from the replay manifest")


def _actor_evaluator(actor: torch.nn.Module) -> ActorEvaluator:
    def evaluate(observations: np.ndarray) -> np.ndarray:
        actions: list[np.ndarray] = []
        with torch.inference_mode():
            for observation in observations:
                tensor = torch.from_numpy(np.ascontiguousarray(observation)).unsqueeze(0)
                actions.append(actor(tensor).detach().cpu().numpy()[0])
        return np.ascontiguousarray(np.stack(actions).astype("<f4", copy=False))

    return evaluate


def _motion_windows(motion: ReferenceMotion, control_steps: np.ndarray) -> np.ndarray:
    windows = [motion.window(int(step)).detach().cpu().numpy() for step in control_steps]
    return np.ascontiguousarray(np.stack(windows).astype("<f4", copy=False))


def run_reference_sensitivity(config: ReferenceSensitivityConfig) -> dict[str, object]:
    """Validate one retained replay and publish a numeric matched-state probe."""

    output_path = Path(config.output_path)
    receipt_path = output_path.with_suffix(f"{output_path.suffix}.manifest.json")
    if output_path.suffix != ".npz":
        raise ValueError("reference-sensitivity output must use the .npz suffix")
    if output_path.exists() or receipt_path.exists():
        raise GMTAdmissionError("refusing to overwrite reference-sensitivity output")
    manifest, _, trace, _ = load_validated_replay(
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
    actor = load_actor(Path(config.weights_path), expected_sha256=config.weights_sha256)
    motion = ReferenceMotion.from_converted(
        Path(config.motion_path),
        name=config.motion_name,
        expected_sha256=config.motion_sha256,
    )
    if np.any(trace["control_sim_step"] % SIMULATION_DECIMATION):
        raise GMTAdmissionError("trace control steps are not aligned to policy cadence")
    control_steps = np.ascontiguousarray(
        (trace["control_sim_step"] // SIMULATION_DECIMATION).astype("<i8", copy=False)
    )
    admitted_windows = _motion_windows(motion, control_steps)
    shifted_windows = _motion_windows(motion, control_steps + SHIFT_CONTROL_STEPS)
    measurement = measure_reference_sensitivity(
        observations=trace["control_observation"],
        recorded_actions=trace["control_raw_action"],
        recorded_reference_windows=trace["control_reference_window"],
        admitted_reference_windows=admitted_windows,
        shifted_reference_windows=shifted_windows,
        recorded_proprioception=trace["control_proprioception"],
        control_steps=control_steps,
        shuffle_seed=config.shuffle_seed,
        actor=_actor_evaluator(actor),
    )
    output_sha256 = write_deterministic_npz(output_path, measurement.arrays)
    output_arrays = {
        key: {"shape": list(value.shape), "dtype": value.dtype.str}
        for key, value in measurement.arrays.items()
    }
    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact": "gmt_g1_matched_state_reference_sensitivity",
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
        },
        "matched_state_contract": {
            "rows": int(control_steps.shape[0]),
            "reference_slice": [REFERENCE_SLICE.start, REFERENCE_SLICE.stop],
            "fixed_non_reference_slice": [REFERENCE_SLICE.stop, OBSERVATION_DIM],
            "fixed_non_reference_dimensions": OBSERVATION_DIM - REFERENCE_SLICE.stop,
            "hash_encoding": HASH_ENCODING,
            "observation_identities": measurement.observation_identities,
        },
        "interventions": {
            "zero_reference": {"operation": "set all 600 reference dimensions to +0.0"},
            "shuffled_reference": {
                "operation": "apply one permutation to each 20-frame reference window",
                "rng": "numpy.random.PCG64",
                "seed": config.shuffle_seed,
                "permutation": list(measurement.permutation),
            },
            "shifted_reference": {
                "operation": "replace only the reference window via admitted motion.window",
                "shift_control_steps": SHIFT_CONTROL_STEPS,
                "shift_seconds": SHIFT_CONTROL_STEPS * CONTROL_DT_SECONDS,
            },
        },
        "action_differences_from_recomputed_exact": measurement.action_summaries,
        "delta_definition": "intervention_raw_action - recomputed_exact_raw_action per state",
        "output": {
            "path": output_path.name,
            "sha256": output_sha256,
            "size": output_path.stat().st_size,
            "arrays": output_arrays,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": str(torch.__version__),
        },
        "limits": {
            "matched_recorded_states_only": True,
            "closed_loop_rollouts_run": False,
            "survival_or_tracking_effect_tested": False,
            "reference_composition_tested": False,
            "learning_performed": False,
            "task_success_tested": False,
            "action_differences_are_not_a_behavioral_causal_claim": True,
        },
    }
    receipt_sha256 = write_json_receipt(receipt_path, receipt)
    return {
        "output_path": str(output_path),
        "output_sha256": output_sha256,
        "manifest_path": str(receipt_path),
        "manifest_sha256": receipt_sha256,
        "claim_status": CLAIM_STATUS,
        "action_differences_from_recomputed_exact": measurement.action_summaries,
    }
