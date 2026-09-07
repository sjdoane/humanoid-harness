"""Run the fixed closed-loop GMT actor-reference ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .composition import ComposedReference
from .contracts import ACTION_DIM
from .control_runtime import PreparedControl
from .course_config import CourseRunConfig, load_run_config
from .course_run import _rollout, make_env
from .deployment import reference_tracking_errors
from .io import GMTAdmissionError, sha256_file, write_deterministic_npz, write_json_receipt
from .reference_ablation import (
    REFERENCE_ABLATION_ARMS,
    REFERENCE_ABLATION_ARTIFACT,
    REFERENCE_ABLATION_CONTRACT_ID,
    REFERENCE_ABLATION_EVIDENCE_CLASS,
    REFERENCE_ABLATION_MANIFEST_FILENAME,
    REFERENCE_AUDIT_SCHEMA_ID,
    AblatedComposedReference,
    ReferenceInputAudit,
    reference_ablation_contract,
    require_reference_ablation_config,
    validate_reference_input_audit,
)
from .reference_ablation_evidence import (
    ZeroResidualActorEvidence,
    capture_zero_residual_actor_evidence,
    measure_practical_trajectory_divergence,
    validate_recomputed_base_action,
)
from .reference_math import quaternion_to_euler_wxyz


@dataclass(frozen=True, slots=True)
class _CapturedControl:
    audit: ReferenceInputAudit
    prepared: PreparedControl
    residual: np.ndarray
    orientation_wxyz: np.ndarray
    angular_velocity: np.ndarray


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _tracking_errors(qpos: np.ndarray, target: np.ndarray) -> np.ndarray:
    roll_pitch = quaternion_to_euler_wxyz(qpos[3:7])[:2]
    return np.asarray(
        reference_tracking_errors(
            dof_position=qpos[-ACTION_DIM:],
            root_height=float(qpos[2]),
            roll_pitch=roll_pitch,
            current_reference=np.ascontiguousarray(target, dtype="<f8"),
        ),
        dtype="<f8",
    )


def _write_audit_rows(path: Path, arm: str, evidence: list[ZeroResidualActorEvidence]) -> str:
    with path.open("x", encoding="utf-8") as handle:
        for item in evidence:
            command = item.audit.original_command
            row = {
                "schema_id": REFERENCE_AUDIT_SCHEMA_ID,
                "schema_version": 1,
                "contract_id": REFERENCE_ABLATION_CONTRACT_ID,
                "arm": arm,
                "control_step": item.audit.control_step,
                "state": command.state,
                "behavior": command.behavior,
                "phase_seconds": command.phase_seconds,
                "phase_fraction": command.phase_fraction,
                "segment_sha256": command.segment_sha256,
                "transition": None if command.transition is None else dict(command.transition),
                "array_sha256": {
                    "original_command_current": _array_sha256(command.current),
                    "original_command_window": _array_sha256(command.window),
                    "actor_reference_window": _array_sha256(item.audit.actor_reference_window),
                    "actor_observation": _array_sha256(item.actor_observation),
                    "base_raw_action": _array_sha256(item.base_raw_action),
                },
            }
            handle.write(
                json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
            )
    return sha256_file(path)


def _retain_actor_evidence(
    *,
    output: Path,
    arm: str,
    captured: list[_CapturedControl],
    standard_trace: Path,
    standard_frames: Path,
    actor: torch.nn.Module,
    segments: dict,
) -> tuple[dict[str, str], int]:
    with np.load(standard_trace, allow_pickle=False) as archive:
        trajectory = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    frame_rows = [json.loads(line) for line in standard_frames.read_text().splitlines()]
    if len(captured) != len(frame_rows) or len(captured) != trajectory["residual_action"].shape[0]:
        raise GMTAdmissionError("actor evidence count differs from the executed trajectory")

    evidence: list[ZeroResidualActorEvidence] = []
    poststep_errors: list[np.ndarray] = []
    actor_target_errors: list[np.ndarray] = []
    for offset, captured_control in enumerate(captured):
        validate_reference_input_audit(captured_control.audit, segments)
        item = capture_zero_residual_actor_evidence(
            audit=captured_control.audit,
            prepared=captured_control.prepared,
            residual_action=captured_control.residual,
            composite_raw_action=trajectory["composite_raw_action"][offset],
        )
        validate_recomputed_base_action(item, actor)
        if not np.array_equal(item.residual_action, trajectory["residual_action"][offset]):
            raise GMTAdmissionError("retained residual differs from the executed trajectory")
        objective_error = _tracking_errors(
            trajectory["qpos"][offset + 1], trajectory["current_reference"][offset]
        )
        observed = frame_rows[offset]["metrics"]
        expected = np.asarray(
            [
                observed["joint_position_rmse_rad"],
                observed["root_height_abs_error_m"],
                observed["roll_pitch_rmse_rad"],
            ],
            dtype="<f8",
        )
        if not np.array_equal(objective_error, expected):
            raise GMTAdmissionError("objective tracking errors differ from the standard frame")
        evidence.append(item)
        poststep_errors.append(objective_error)
        actor_target_errors.append(
            _tracking_errors(trajectory["qpos"][offset + 1], item.audit.actor_reference_window[0])
        )

    evidence_path = output / f"{arm}_actor_input.npz"
    evidence_sha256 = write_deterministic_npz(
        evidence_path,
        {
            "control_step": np.asarray([item.audit.control_step for item in evidence], dtype="<i8"),
            "original_command_current": np.asarray(
                [item.audit.original_command.current for item in evidence], dtype="<f4"
            ),
            "original_command_window": np.asarray(
                [item.audit.original_command.window for item in evidence], dtype="<f4"
            ),
            "actor_reference_window": np.asarray(
                [item.audit.actor_reference_window for item in evidence], dtype="<f4"
            ),
            "actor_observation": np.asarray(
                [item.actor_observation for item in evidence], dtype="<f4"
            ),
            "prepared_proprio": np.asarray(
                [item.prepared_proprio for item in evidence], dtype="<f8"
            ),
            "boundary_orientation_wxyz": np.asarray(
                [item.orientation_wxyz for item in captured], dtype="<f4"
            ),
            "boundary_angular_velocity": np.asarray(
                [item.angular_velocity for item in captured], dtype="<f4"
            ),
            "base_raw_action": np.asarray([item.base_raw_action for item in evidence], dtype="<f4"),
            "residual_action": np.asarray([item.residual_action for item in evidence], dtype="<f4"),
            "composite_raw_action": np.asarray(
                [item.composite_raw_action for item in evidence], dtype="<f4"
            ),
            "original_poststep_target": np.asarray(trajectory["current_reference"], dtype="<f4"),
            "original_poststep_tracking_errors": np.asarray(poststep_errors, dtype="<f8"),
            "actor_window_first_row_tracking_errors": np.asarray(actor_target_errors, dtype="<f8"),
        },
    )
    audit_path = output / f"{arm}_reference_audit.jsonl"
    audit_sha256 = _write_audit_rows(audit_path, arm, evidence)
    return {
        evidence_path.name: evidence_sha256,
        audit_path.name: audit_sha256,
    }, len(evidence)


def _run_arm(
    config: CourseRunConfig, output: Path, arm: str
) -> tuple[dict[str, object], dict[str, str]]:
    env = make_env(config, record_trajectory=True)
    if type(env.oracle) is not ComposedReference:
        raise GMTAdmissionError("reference ablation requires the reviewed composed runtime")
    oracle = AblatedComposedReference(env.oracle, arm=arm)
    env.oracle = oracle  # Installed before reset; every arm gets a fresh environment.
    captured: list[_CapturedControl] = []

    def observe(prepared: PreparedControl, residual: np.ndarray) -> None:
        captured.append(
            _CapturedControl(
                audit=oracle.last_audit,
                prepared=prepared,
                residual=residual.copy(),
                orientation_wxyz=env._boundary.orientation_wxyz.copy(),
                angular_velocity=env._boundary.angular_velocity.copy(),
            )
        )

    try:
        report, outputs = _rollout(config, env, None, output, arm, step_observer=observe)
        actor_outputs, frame_count = _retain_actor_evidence(
            output=output,
            arm=arm,
            captured=captured,
            standard_trace=output / f"{arm}_trajectory.npz",
            standard_frames=output / f"{arm}_frames.jsonl",
            actor=env.actor._actor,
            segments=env.oracle.segments,
        )
        outputs.update(actor_outputs)
        return {**report, "actor_input_frame_count": frame_count}, outputs
    finally:
        env.close()


def run_reference_ablation(config_path: Path, output: Path) -> dict[str, object]:
    config = load_run_config(config_path)
    require_reference_ablation_config(config)
    output.mkdir(exist_ok=True)
    if output.is_symlink() or any(
        path.name not in {"child_stdout.json", "child_stderr.log"} for path in output.iterdir()
    ):
        raise GMTAdmissionError(
            "reference ablation output must be fresh except for supervisor logs"
        )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    with (output / "input_config.json").open("xb") as handle:
        handle.write(config.encoded)
    outputs = {"input_config.json": config.sha256}
    summaries: dict[str, object] = {}
    qpos: dict[str, np.ndarray] = {}
    for arm in REFERENCE_ABLATION_ARMS:
        summary, arm_outputs = _run_arm(config, output, arm)
        summaries[arm] = summary
        outputs.update(arm_outputs)
        with np.load(output / f"{arm}_trajectory.npz", allow_pickle=False) as archive:
            qpos[arm] = np.ascontiguousarray(archive["qpos"], dtype="<f8")
    comparisons = {
        arm: measure_practical_trajectory_divergence(qpos["exact"], qpos[arm])
        for arm in REFERENCE_ABLATION_ARMS
        if arm != "exact"
    }
    manifest = {
        "schema_version": 1,
        "artifact": REFERENCE_ABLATION_ARTIFACT,
        "evidence_class": REFERENCE_ABLATION_EVIDENCE_CLASS,
        "status": "completed",
        "input_config_sha256": config.sha256,
        "intervention_contract": reference_ablation_contract(),
        "outputs": outputs,
        "identities": {
            "actor": config.assets["weights"]["sha256"],
            "task": config.task.sha256,
            "oracle": config.program.sha256,
            "reward": config.recipe.sha256,
            "segments": {name: segment.sha256 for name, segment in config.segments.items()},
            "course_runtime": config.runtime.manifest_contract(),
        },
        "arms": summaries,
        "treated_vs_exact": comparisons,
        "runtime": {
            "wall_seconds": time.monotonic() - started,
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "claims": {
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
        },
    }
    manifest_sha256 = write_json_receipt(output / REFERENCE_ABLATION_MANIFEST_FILENAME, manifest)
    return {
        "manifest_sha256": manifest_sha256,
        "arms": summaries,
        "treated_vs_exact": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run_reference_ablation(arguments.config, arguments.output),
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
