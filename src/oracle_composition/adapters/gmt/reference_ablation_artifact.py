"""Data-only validation for one closed-loop GMT reference ablation."""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from oracle_composition.feedback.g1_course import (
    _load_frames,
    _load_trajectory,
    _verified_bytes,
    _verify_boundary_metrics,
    _verify_frame_crosslinks,
)

from .actor import load_actor
from .composition import ComposedReference, ReferenceCommand
from .contracts import (
    ACTION_DIM,
    CONTROL_DT_SECONDS,
    OBSERVATION_DIM,
    PROPRIOCEPTION_DIM,
    REFERENCE_FRAME_DIM,
    REFERENCE_HORIZON,
)
from .course_config import CourseRunConfig, load_run_config
from .course_evaluation import evaluate_episode
from .course_task import TaskFrame
from .deployment import ObservationHistory, build_proprioception, reference_tracking_errors
from .io import GMTAdmissionError, sha256_file, validate_zip_members
from .reference_ablation import (
    REFERENCE_ABLATION_ARMS,
    REFERENCE_ABLATION_ARTIFACT,
    REFERENCE_ABLATION_CONTRACT_ID,
    REFERENCE_ABLATION_EVIDENCE_CLASS,
    REFERENCE_ABLATION_MANIFEST_FILENAME,
    REFERENCE_AUDIT_SCHEMA_ID,
    ReferenceInputAudit,
    reference_ablation_contract,
    require_reference_ablation_config,
    validate_reference_input_audit,
)
from .reference_ablation_evidence import (
    ZeroResidualActorEvidence,
    measure_practical_trajectory_divergence,
    validate_recomputed_base_action,
)
from .reference_math import quaternion_to_euler_wxyz

_MANIFEST_FIELDS = {
    "schema_version",
    "artifact",
    "evidence_class",
    "status",
    "input_config_sha256",
    "intervention_contract",
    "outputs",
    "identities",
    "arms",
    "treated_vs_exact",
    "runtime",
    "claims",
}
_SUMMARY_FIELDS = {
    "objective_evaluation",
    "training_reward_sum_not_success_metric",
    "reset",
    "steps",
    "residual_rms",
    "actor_input_frame_count",
}
_AUDIT_FIELDS = {
    "schema_id",
    "schema_version",
    "contract_id",
    "arm",
    "control_step",
    "state",
    "behavior",
    "phase_seconds",
    "phase_fraction",
    "segment_sha256",
    "transition",
    "array_sha256",
}
_AUDIT_ARRAYS = {
    "original_command_current",
    "original_command_window",
    "actor_reference_window",
    "actor_observation",
    "base_raw_action",
}
_EVIDENCE_DTYPES = {
    "control_step": "<i8",
    "original_command_current": "<f4",
    "original_command_window": "<f4",
    "actor_reference_window": "<f4",
    "actor_observation": "<f4",
    "prepared_proprio": "<f8",
    "boundary_orientation_wxyz": "<f4",
    "boundary_angular_velocity": "<f4",
    "base_raw_action": "<f4",
    "residual_action": "<f4",
    "composite_raw_action": "<f4",
    "original_poststep_target": "<f4",
    "original_poststep_tracking_errors": "<f8",
    "actor_window_first_row_tracking_errors": "<f8",
}
_CLAIMS = {
    "development_only": True,
    "closed_loop_rollouts_performed": True,
    "actor_reference_input_only_intervened": True,
    "original_objective_target_preserved": True,
    "zero_residual_all_arms": True,
    "learning_performed": False,
    "heldout_generalization_tested": False,
    "task_success_claimed": False,
    "reference_composition_quality_claimed": False,
    "universal_reference_use_claimed": False,
    "null_interpretation": "no_detected_effect_under_only_these_interventions",
}
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_CONFIG_BYTES = 256 * 1024
_MAX_FRAMES_BYTES = 128 * 1024 * 1024
_MAX_NUMERIC_BYTES = 256 * 1024 * 1024


def _sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GMTAdmissionError(f"{field} must be one lowercase SHA-256")
    return value


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _same_bytes(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _output_names(arm: str) -> set[str]:
    return {
        f"{arm}_frames.jsonl",
        f"{arm}_trajectory.npz",
        f"{arm}_evaluation.json",
        f"{arm}_actor_input.npz",
        f"{arm}_reference_audit.jsonl",
    }


def expected_reference_ablation_outputs() -> set[str]:
    result = {"input_config.json"}
    for arm in REFERENCE_ABLATION_ARMS:
        result.update(_output_names(arm))
    return result


def _load_json(encoded: bytes, *, source: str) -> dict[str, Any]:
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GMTAdmissionError(f"{source} is not valid JSON") from exc
    if type(value) is not dict:
        raise GMTAdmissionError(f"{source} must be one JSON object")
    return value


def _load_audit(encoded: bytes, frame_count: int) -> list[dict[str, Any]]:
    rows = []
    for index, line in enumerate(encoded.splitlines(), start=1):
        if not line:
            raise GMTAdmissionError("reference audit contains a blank JSONL record")
        rows.append(_load_json(line, source=f"reference audit row {index}"))
    if len(rows) != frame_count:
        raise GMTAdmissionError("reference audit count differs from the trajectory")
    return rows


def _load_evidence(encoded: bytes, frame_count: int) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(io.BytesIO(encoded), "r") as archive:
        members = validate_zip_members(
            archive,
            expected_count=len(_EVIDENCE_DTYPES),
            maximum_member_size=64 * 1024 * 1024,
        )
        if set(members) != {f"{name}.npy" for name in _EVIDENCE_DTYPES}:
            raise GMTAdmissionError("reference actor-input evidence arrays differ")
        if sum(member.file_size for member in members.values()) > _MAX_NUMERIC_BYTES:
            raise GMTAdmissionError("reference actor-input evidence expands beyond its bound")
    shapes = {
        "control_step": (frame_count,),
        "original_command_current": (frame_count, REFERENCE_FRAME_DIM),
        "original_command_window": (
            frame_count,
            REFERENCE_HORIZON,
            REFERENCE_FRAME_DIM,
        ),
        "actor_reference_window": (
            frame_count,
            REFERENCE_HORIZON,
            REFERENCE_FRAME_DIM,
        ),
        "actor_observation": (frame_count, OBSERVATION_DIM),
        "prepared_proprio": (frame_count, PROPRIOCEPTION_DIM),
        "boundary_orientation_wxyz": (frame_count, 4),
        "boundary_angular_velocity": (frame_count, 3),
        "base_raw_action": (frame_count, ACTION_DIM),
        "residual_action": (frame_count, ACTION_DIM),
        "composite_raw_action": (frame_count, ACTION_DIM),
        "original_poststep_target": (frame_count, REFERENCE_FRAME_DIM),
        "original_poststep_tracking_errors": (frame_count, 3),
        "actor_window_first_row_tracking_errors": (frame_count, 3),
    }
    result = {}
    with np.load(io.BytesIO(encoded), allow_pickle=False) as archive:
        if set(archive.files) != set(_EVIDENCE_DTYPES):
            raise GMTAdmissionError("reference actor-input evidence members differ")
        for name, dtype in _EVIDENCE_DTYPES.items():
            value = archive[name]
            if (
                value.shape != shapes[name]
                or value.dtype.str != dtype
                or not np.isfinite(value).all()
            ):
                raise GMTAdmissionError(f"reference actor-input evidence differs: {name}")
            result[name] = np.ascontiguousarray(value)
    return result


def _tracking_errors(qpos: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.asarray(
        reference_tracking_errors(
            dof_position=qpos[-ACTION_DIM:],
            root_height=float(qpos[2]),
            roll_pitch=quaternion_to_euler_wxyz(qpos[3:7])[:2],
            current_reference=np.ascontiguousarray(target, dtype="<f8"),
        ),
        dtype="<f8",
    )


def _signals(
    config: CourseRunConfig,
    frame: TaskFrame,
    qpos: np.ndarray,
    qvel: np.ndarray,
    control_step: int,
) -> tuple[dict[str, float], np.ndarray]:
    projection = frame.project(qpos[:2], qpos[3:7])
    yaw = frame.forward_yaw_rad
    speed = float(qvel[0] * math.cos(yaw) + qvel[1] * math.sin(yaw))
    inside = (
        config.task.region_entry_distance_m
        <= projection.progress_m
        < config.task.region_exit_distance_m
    )
    quaternion = qpos[3:7]
    signals = {
        "t": control_step * CONTROL_DT_SECONDS,
        "v_x": speed,
        "v_target": config.task.target_speed_inside_m_s
        if inside
        else config.task.target_speed_outside_m_s,
        "z_root": float(qpos[2]),
        "torso_up": float(1 - 2 * (quaternion[1] ** 2 + quaternion[2] ** 2)),
        "x_travelled": projection.progress_m,
    }
    pose = np.concatenate((qpos[2:3], quaternion_to_euler_wxyz(quaternion)[:2], qpos[-ACTION_DIM:]))
    return signals, pose


def _compare_command(observed: ReferenceCommand, expected: ReferenceCommand) -> None:
    observed_metadata = (
        observed.state,
        observed.behavior,
        observed.phase_seconds,
        observed.phase_fraction,
        observed.segment_sha256,
        None if observed.transition is None else dict(observed.transition),
    )
    expected_metadata = (
        expected.state,
        expected.behavior,
        expected.phase_seconds,
        expected.phase_fraction,
        expected.segment_sha256,
        None if expected.transition is None else dict(expected.transition),
    )
    if (
        observed_metadata != expected_metadata
        or not _same_bytes(observed.current, expected.current)
        or not _same_bytes(observed.window, expected.window)
    ):
        raise GMTAdmissionError("retained original command differs from composed runtime replay")


def _verify_arm(
    *,
    arm: str,
    config: CourseRunConfig,
    output: Path,
    outputs: Mapping[str, str],
    summary: object,
    actor: Any,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    frame_bytes = _verified_bytes(
        output / f"{arm}_frames.jsonl",
        outputs[f"{arm}_frames.jsonl"],
        _MAX_FRAMES_BYTES,
    )
    frames = _load_frames(frame_bytes)
    frame_count = len(frames)
    trajectory = _load_trajectory(
        _verified_bytes(
            output / f"{arm}_trajectory.npz",
            outputs[f"{arm}_trajectory.npz"],
            _MAX_NUMERIC_BYTES,
        ),
        frame_count,
    )
    _verify_frame_crosslinks(frames, trajectory)
    _verify_boundary_metrics(spec=config.task, frames=frames, trajectory=trajectory)
    objective = evaluate_episode(spec=config.task, frames=frames)
    terminal_metrics = frames[-1]["metrics"]
    if not (terminal_metrics["fallen"] or terminal_metrics["horizon_reached"]):
        raise GMTAdmissionError(f"{arm} ended before the horizon without a fall")
    report = _load_json(
        _verified_bytes(
            output / f"{arm}_evaluation.json",
            outputs[f"{arm}_evaluation.json"],
            _MAX_CONFIG_BYTES,
        ),
        source=f"{arm} evaluation",
    )
    if set(report) != _SUMMARY_FIELDS - {"actor_input_frame_count"}:
        raise GMTAdmissionError(f"{arm} standard evaluation fields differ")
    if (
        report["objective_evaluation"] != objective
        or report["steps"] != frame_count
        or report["residual_rms"] != 0
        or not isinstance(report["reset"], Mapping)
        or type(report["training_reward_sum_not_success_metric"]) not in {int, float}
        or not math.isfinite(report["training_reward_sum_not_success_metric"])
    ):
        raise GMTAdmissionError(f"{arm} standard evaluation differs from retained trajectory")
    if (
        not isinstance(summary, Mapping)
        or set(summary) != _SUMMARY_FIELDS
        or dict(summary) != {**report, "actor_input_frame_count": frame_count}
    ):
        raise GMTAdmissionError(f"{arm} manifest summary differs from retained evidence")

    evidence = _load_evidence(
        _verified_bytes(
            output / f"{arm}_actor_input.npz",
            outputs[f"{arm}_actor_input.npz"],
            _MAX_NUMERIC_BYTES,
        ),
        frame_count,
    )
    audit_rows = _load_audit(
        _verified_bytes(
            output / f"{arm}_reference_audit.jsonl",
            outputs[f"{arm}_reference_audit.jsonl"],
            _MAX_FRAMES_BYTES,
        ),
        frame_count,
    )
    oracle = ComposedReference(config.program, config.segments)
    task_frame = TaskFrame.initialize(trajectory["qpos"][0, :2], trajectory["qpos"][0, 3:7])
    history = ObservationHistory()
    last_raw_action = np.zeros(ACTION_DIM, dtype="<f4")
    for offset, row in enumerate(audit_rows):
        if set(row) != _AUDIT_FIELDS or row.get("array_sha256") is None:
            raise GMTAdmissionError("reference audit fields differ")
        if (
            row["schema_id"] != REFERENCE_AUDIT_SCHEMA_ID
            or row["schema_version"] != 1
            or row["contract_id"] != REFERENCE_ABLATION_CONTRACT_ID
            or row["arm"] != arm
            or row["control_step"] != offset
            or set(row["array_sha256"]) != _AUDIT_ARRAYS
            or evidence["control_step"][offset] != offset
        ):
            raise GMTAdmissionError("reference audit identity or sequence differs")
        command = ReferenceCommand(
            state=row["state"],
            behavior=row["behavior"],
            current=evidence["original_command_current"][offset],
            window=evidence["original_command_window"][offset],
            phase_seconds=row["phase_seconds"],
            phase_fraction=row["phase_fraction"],
            segment_sha256=row["segment_sha256"],
            transition=row["transition"],
        )
        audit = ReferenceInputAudit(
            arm=arm,
            control_step=offset,
            original_command=command,
            actor_reference_window=evidence["actor_reference_window"][offset],
        )
        validate_reference_input_audit(audit, config.segments)
        expected_hashes = {
            "original_command_current": _array_sha256(command.current),
            "original_command_window": _array_sha256(command.window),
            "actor_reference_window": _array_sha256(audit.actor_reference_window),
            "actor_observation": _array_sha256(evidence["actor_observation"][offset]),
            "base_raw_action": _array_sha256(evidence["base_raw_action"][offset]),
        }
        if row["array_sha256"] != expected_hashes:
            raise GMTAdmissionError("reference audit numeric hash differs")

        signals, pose = _signals(
            config,
            task_frame,
            trajectory["qpos"][offset],
            trajectory["qvel"][offset],
            offset,
        )
        expected_command = oracle.command(step=offset, signals=signals, robot_pose=pose)
        _compare_command(command, expected_command)
        expected_poststep = oracle.current_after_step(offset + 1)
        if not _same_bytes(
            np.ascontiguousarray(expected_poststep, dtype="<f4"),
            trajectory["current_reference"][offset],
        ) or not _same_bytes(
            trajectory["current_reference"][offset],
            evidence["original_poststep_target"][offset],
        ):
            raise GMTAdmissionError("original objective target differs from composed replay")
        frame = frames[offset]
        if (
            frame["executed_mode"] != expected_command.state
            or frame["executed_behavior"] != expected_command.behavior
            or frame["executed_phase_seconds"] != expected_command.phase_seconds
            or frame["transition"]
            != (None if expected_command.transition is None else dict(expected_command.transition))
        ):
            raise GMTAdmissionError("executed oracle metadata differs from composed replay")

        expected_proprio = build_proprioception(
            dof_position=np.ascontiguousarray(
                trajectory["qpos"][offset, -ACTION_DIM:], dtype="<f4"
            ),
            dof_velocity=np.ascontiguousarray(
                trajectory["qvel"][offset, -ACTION_DIM:], dtype="<f4"
            ),
            orientation_wxyz=evidence["boundary_orientation_wxyz"][offset],
            angular_velocity=evidence["boundary_angular_velocity"][offset],
            last_raw_action=last_raw_action,
        )
        if not _same_bytes(
            np.ascontiguousarray(expected_proprio, dtype="<f8"),
            evidence["prepared_proprio"][offset],
        ):
            raise GMTAdmissionError("prepared proprioception differs from retained boundaries")
        expected_observation = history.assemble(
            audit.actor_reference_window, evidence["prepared_proprio"][offset]
        )
        if not _same_bytes(
            np.ascontiguousarray(expected_observation, dtype="<f4"),
            evidence["actor_observation"][offset],
        ):
            raise GMTAdmissionError("actor observation differs from retained input history")
        actor_evidence = ZeroResidualActorEvidence(
            audit=audit,
            actor_observation=evidence["actor_observation"][offset],
            prepared_proprio=evidence["prepared_proprio"][offset],
            base_raw_action=evidence["base_raw_action"][offset],
            residual_action=evidence["residual_action"][offset],
            composite_raw_action=evidence["composite_raw_action"][offset],
        )
        validate_recomputed_base_action(actor_evidence, actor)
        if not _same_bytes(
            actor_evidence.residual_action, trajectory["residual_action"][offset]
        ) or not _same_bytes(
            actor_evidence.composite_raw_action,
            trajectory["composite_raw_action"][offset],
        ):
            raise GMTAdmissionError("actor evidence action differs from executed trajectory")
        original_errors = _tracking_errors(
            trajectory["qpos"][offset + 1], trajectory["current_reference"][offset]
        )
        actor_errors = _tracking_errors(
            trajectory["qpos"][offset + 1], audit.actor_reference_window[0]
        )
        if not _same_bytes(
            original_errors, evidence["original_poststep_tracking_errors"][offset]
        ) or not _same_bytes(
            actor_errors,
            evidence["actor_window_first_row_tracking_errors"][offset],
        ):
            raise GMTAdmissionError("reference tracking-error evidence differs")
        metrics = frame["metrics"]
        if not np.array_equal(
            original_errors,
            np.asarray(
                [
                    metrics["joint_position_rmse_rad"],
                    metrics["root_height_abs_error_m"],
                    metrics["roll_pitch_rmse_rad"],
                ],
                dtype="<f8",
            ),
        ):
            raise GMTAdmissionError("objective errors differ from the standard evaluator")
        history.append(evidence["prepared_proprio"][offset])
        last_raw_action = trajectory["composite_raw_action"][offset]
    return trajectory, report


def validate_reference_ablation_artifact(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    config_path: Path,
    expected_config_sha256: str,
) -> dict[str, object]:
    manifest_file = Path(manifest_path)
    if (
        not manifest_file.is_absolute()
        or manifest_file.name != REFERENCE_ABLATION_MANIFEST_FILENAME
        or manifest_file.is_symlink()
    ):
        raise GMTAdmissionError("reference ablation manifest path is not direct and absolute")
    output = manifest_file.parent
    if output.is_symlink() or not output.is_dir():
        raise GMTAdmissionError("reference ablation output directory is not direct")
    manifest_sha256 = _sha256(expected_manifest_sha256, field="manifest SHA-256")
    manifest = _load_json(
        _verified_bytes(manifest_file, manifest_sha256, _MAX_MANIFEST_BYTES),
        source="reference ablation manifest",
    )
    config_sha256 = _sha256(expected_config_sha256, field="config SHA-256")
    _verified_bytes(config_path, config_sha256, _MAX_CONFIG_BYTES)
    config = load_run_config(config_path)
    require_reference_ablation_config(config)
    if config.sha256 != config_sha256:
        raise GMTAdmissionError("reference ablation config bytes differ from the accepted input")
    if (
        set(manifest) != _MANIFEST_FIELDS
        or manifest["schema_version"] != 1
        or manifest["artifact"] != REFERENCE_ABLATION_ARTIFACT
        or manifest["evidence_class"] != REFERENCE_ABLATION_EVIDENCE_CLASS
        or manifest["status"] != "completed"
        or manifest["input_config_sha256"] != config_sha256
        or manifest["intervention_contract"] != reference_ablation_contract()
        or manifest["claims"] != _CLAIMS
    ):
        raise GMTAdmissionError("reference ablation manifest identity or contract differs")
    outputs = manifest["outputs"]
    if type(outputs) is not dict or set(outputs) != expected_reference_ablation_outputs():
        raise GMTAdmissionError("reference ablation output ledger differs")
    actual_names = {path.name for path in output.iterdir()}
    allowed = {
        *outputs,
        REFERENCE_ABLATION_MANIFEST_FILENAME,
        "child_stdout.json",
        "child_stderr.log",
    }
    if (
        not {REFERENCE_ABLATION_MANIFEST_FILENAME, *outputs} <= actual_names
        or not actual_names <= allowed
    ):
        raise GMTAdmissionError("reference ablation has unknown or missing output files")
    artifacts = {}
    for name, raw_digest in outputs.items():
        digest = _sha256(raw_digest, field=f"reference ablation output {name}")
        path = output / name
        if sha256_file(path) != digest:
            raise GMTAdmissionError(f"reference ablation output hash differs: {name}")
        artifacts[name] = {"path": name, "sha256": digest, "size": path.stat().st_size}
    retained_config = _verified_bytes(
        output / "input_config.json", outputs["input_config.json"], _MAX_CONFIG_BYTES
    )
    admitted_config = _verified_bytes(config_path, config_sha256, _MAX_CONFIG_BYTES)
    if retained_config != admitted_config:
        raise GMTAdmissionError("retained reference ablation config differs")
    expected_identities = {
        "actor": config.assets["weights"]["sha256"],
        "task": config.task.sha256,
        "oracle": config.program.sha256,
        "reward": config.recipe.sha256,
        "segments": {name: segment.sha256 for name, segment in config.segments.items()},
        "course_runtime": config.runtime.manifest_contract(),
    }
    if manifest["identities"] != expected_identities:
        raise GMTAdmissionError("reference ablation semantic identities differ")
    if (
        type(manifest["arms"]) is not dict
        or set(manifest["arms"]) != set(REFERENCE_ABLATION_ARMS)
        or type(manifest["treated_vs_exact"]) is not dict
        or set(manifest["treated_vs_exact"]) != set(REFERENCE_ABLATION_ARMS[1:])
    ):
        raise GMTAdmissionError("reference ablation arm set or order differs")
    runtime = manifest["runtime"]
    if (
        type(runtime) is not dict
        or set(runtime) != {"wall_seconds", "torch", "numpy", "platform"}
        or type(runtime["wall_seconds"]) not in {int, float}
        or not 0 <= runtime["wall_seconds"] < float("inf")
        or any(
            type(runtime[name]) is not str or not runtime[name]
            for name in ("torch", "numpy", "platform")
        )
    ):
        raise GMTAdmissionError("reference ablation runtime provenance differs")
    actor = load_actor(
        Path(config.assets["weights"]["path"]),
        expected_sha256=config.assets["weights"]["sha256"],
        freeze=True,
    )
    trajectories = {}
    reports = {}
    for arm in REFERENCE_ABLATION_ARMS:
        trajectory, report = _verify_arm(
            arm=arm,
            config=config,
            output=output,
            outputs=outputs,
            summary=manifest["arms"][arm],
            actor=actor,
        )
        trajectories[arm] = trajectory
        reports[arm] = report
    exact_reset = reports["exact"]["reset"]
    if any(report["reset"] != exact_reset for report in reports.values()):
        raise GMTAdmissionError("reference ablation arms did not use the same fixed reset")
    expected_comparisons = {
        arm: measure_practical_trajectory_divergence(
            trajectories["exact"]["qpos"], trajectories[arm]["qpos"]
        )
        for arm in REFERENCE_ABLATION_ARMS[1:]
    }
    if manifest["treated_vs_exact"] != expected_comparisons:
        raise GMTAdmissionError("reference ablation divergence summary differs")
    return {
        "manifest": {
            "path": REFERENCE_ABLATION_MANIFEST_FILENAME,
            "sha256": manifest_sha256,
            "size": manifest_file.stat().st_size,
        },
        "outputs": artifacts,
        "arms": {
            arm: {
                "steps": reports[arm]["steps"],
                "fall_count": reports[arm]["objective_evaluation"]["fall_count"],
            }
            for arm in REFERENCE_ABLATION_ARMS
        },
        "treated_vs_exact": expected_comparisons,
        "claim_ceiling": "closed_loop_reference_input_effect_under_fixed_tested_interventions_only",
    }


__all__ = [
    "expected_reference_ablation_outputs",
    "validate_reference_ablation_artifact",
]
