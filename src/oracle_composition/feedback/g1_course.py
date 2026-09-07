"""Build proposal-safe feedback from one pinned G1 development rollout."""

from __future__ import annotations

import hashlib
import io
import stat
import zipfile
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from oracle_composition.adapters.gmt.contracts import (
    ACTION_DIM,
    CONTROL_DT_SECONDS,
    REFERENCE_FRAME_DIM,
)
from oracle_composition.adapters.gmt.course_config import load_run_config
from oracle_composition.adapters.gmt.course_evaluation import evaluate_episode
from oracle_composition.adapters.gmt.course_proposal import (
    COURSE_FEEDBACK_EVIDENCE_CLASS,
    MAX_DIAGNOSIS_CHARACTERS,
)
from oracle_composition.adapters.gmt.course_runtime import (
    COURSE_RESIDUAL_RAW_SCALE,
    frozen_runtime_contract,
)
from oracle_composition.adapters.gmt.course_task import CourseTaskSpec, TaskFrame, evaluate_step
from oracle_composition.adapters.gmt.io import validate_zip_members
from oracle_composition.adapters.gmt.training_contract import (
    effective_training_contract,
    training_reward_metadata,
)
from oracle_composition.adapters.gmt.training_telemetry import (
    MAX_TELEMETRY_BYTES,
    SCALED_TELEMETRY_FILENAME,
    TELEMETRY_FILENAME,
    validate_training_telemetry_descriptor,
)
from oracle_composition.experiments.artifact_io import publish_json_without_overwrite
from oracle_composition.harness.contract import decode_json_object

_MANIFEST_FIELDS = {
    "schema_version",
    "artifact",
    "status",
    "input_config_sha256",
    "outputs",
    "identities",
    "frozen_runtime",
    "training",
    "zero_residual",
    "final_policy",
    "runtime",
    "claims",
}
_PROBE_OUTPUTS = {
    "input_config.json",
    "zero_residual_frames.jsonl",
    "zero_residual_trajectory.npz",
    "zero_residual_evaluation.json",
}
_TRAIN_OUTPUTS = {
    *_PROBE_OUTPUTS,
    "initial_residual_policy.npz",
    "final_residual_policy.npz",
    "final_policy_frames.jsonl",
    "final_policy_trajectory.npz",
    "final_policy_evaluation.json",
}
_TRAIN_TELEMETRY_OUTPUTS = {*_TRAIN_OUTPUTS, TELEMETRY_FILENAME}
_TRAIN_SCALED_TELEMETRY_OUTPUTS = {*_TRAIN_OUTPUTS, SCALED_TELEMETRY_FILENAME}
_SUMMARY_FIELDS = {
    "objective_evaluation",
    "training_reward_sum_not_success_metric",
    "reset",
    "steps",
    "residual_rms",
}
_EVALUATION_FIELDS = (
    "evaluator_id",
    "claim_scope",
    "task_sha256",
    "development_gate_results",
    "development_gate_passed",
    "episode_success",
)
_TRAJECTORY_FIELDS = {
    "qpos",
    "qvel",
    "current_reference",
    "composite_raw_action",
    "contact_pairs",
    "geom_body_names",
}
_NPZ_DTYPES = {
    "qpos": "<f8",
    "qvel": "<f8",
    "residual_action": "<f4",
    "current_reference": "<f4",
    "composite_raw_action": "<f4",
}
_BOUNDARY_METRIC_FIELDS = (
    "progress_m",
    "lateral_m",
    "heading_error_signed_rad",
    "root_height_m",
    "torso_up",
    "forward_speed_m_s",
    "target_speed_m_s",
    "speed_error_m_s",
    "posture_band_error_m",
    "lateral_error_m",
    "heading_error_rad",
    "inside_posture_region",
    "finish_condition_met",
    "horizon_reached",
    "joint_position_rmse_rad",
    "root_height_abs_error_m",
    "roll_pitch_rmse_rad",
)
_MAX_MANIFEST_BYTES = 1024 * 1024
_MAX_JSON_BYTES = 256 * 1024
_MAX_FRAMES_BYTES = 128 * 1024 * 1024
_MAX_NUMERIC_BYTES = 128 * 1024 * 1024


def _sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be one lowercase SHA-256")
    return value


def _verified_bytes(path: Path, expected_sha256: str, maximum: int) -> bytes:
    candidate = Path(path)
    before = candidate.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"input must be a regular non-linked file: {candidate}")
    if not 0 < before.st_size <= maximum:
        raise ValueError(f"input size is outside its bound: {candidate.name}")
    encoded = candidate.read_bytes()
    after = candidate.lstat()

    def identity(value: Any) -> tuple[int, int, int, int]:
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)

    if identity(before) != identity(after) or len(encoded) != before.st_size:
        raise ValueError(f"input changed while being read: {candidate.name}")
    observed = hashlib.sha256(encoded).hexdigest()
    if observed != expected_sha256:
        raise ValueError(f"SHA-256 mismatch for {candidate.name}")
    return encoded


def _json(encoded: bytes, *, source: str) -> dict[str, object]:
    return decode_json_object(encoded, source=source)


def _load_frames(encoded: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(encoded.splitlines(), start=1):
        if not line:
            raise ValueError("course frames contain a blank JSONL record")
        rows.append(_json(line, source=f"course frame {index}"))
    if not rows:
        raise ValueError("course frames are empty")
    return rows


def _load_trajectory(encoded: bytes, frame_count: int) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(io.BytesIO(encoded), "r") as archive:
        members = validate_zip_members(
            archive,
            expected_count=len(_NPZ_DTYPES),
            maximum_member_size=32 * 1024 * 1024,
        )
        if set(members) != {f"{name}.npy" for name in _NPZ_DTYPES}:
            raise ValueError("course trajectory NPZ members differ")
        if sum(member.file_size for member in members.values()) > 32 * 1024 * 1024:
            raise ValueError("course trajectory expands beyond its numeric bound")
    expected_shapes = {
        "qpos": (frame_count + 1, 7 + ACTION_DIM),
        "qvel": (frame_count + 1, 6 + ACTION_DIM),
        "residual_action": (frame_count, ACTION_DIM),
        "current_reference": (frame_count, REFERENCE_FRAME_DIM),
        "composite_raw_action": (frame_count, ACTION_DIM),
    }
    result: dict[str, np.ndarray] = {}
    with np.load(io.BytesIO(encoded), allow_pickle=False) as archive:
        if set(archive.files) != set(_NPZ_DTYPES):
            raise ValueError("course trajectory arrays differ")
        for name, dtype in _NPZ_DTYPES.items():
            value = archive[name]
            if (
                value.shape != expected_shapes[name]
                or value.dtype.str != dtype
                or not np.isfinite(value).all()
            ):
                raise ValueError(f"course trajectory {name} contract differs")
            result[name] = np.ascontiguousarray(value)
    return result


def _verify_frame_crosslinks(
    frames: list[dict[str, Any]], trajectory: dict[str, np.ndarray]
) -> None:
    for offset, frame in enumerate(frames):
        retained = frame.get("trajectory")
        if type(retained) is not dict or set(retained) != _TRAJECTORY_FIELDS:
            raise ValueError("selected course frame lacks its exact trajectory record")
        comparisons = (
            ("qpos", "<f8", trajectory["qpos"][offset + 1]),
            ("qvel", "<f8", trajectory["qvel"][offset + 1]),
            ("current_reference", "<f4", trajectory["current_reference"][offset]),
            ("composite_raw_action", "<f4", trajectory["composite_raw_action"][offset]),
        )
        for name, dtype, expected in comparisons:
            observed = np.asarray(retained[name], dtype=dtype)
            if observed.shape != expected.shape or not np.array_equal(observed, expected):
                raise ValueError(f"course frame and trajectory {name} differ")


def _verify_boundary_metrics(
    *,
    spec: CourseTaskSpec,
    frames: list[dict[str, Any]],
    trajectory: dict[str, np.ndarray],
) -> None:
    task_frame = TaskFrame.initialize(trajectory["qpos"][0, :2], trajectory["qpos"][0, 3:7])
    for offset, row in enumerate(frames):
        recomputed = evaluate_step(
            spec=spec,
            frame=task_frame,
            before_qpos=trajectory["qpos"][offset],
            after_qpos=trajectory["qpos"][offset + 1],
            ground_contact_bodies=(),
            current_reference=np.asarray(
                row["trajectory"]["current_reference"], dtype=np.float64
            ),
            control_step=offset + 1,
        ).to_dict()
        observed = row["metrics"]
        for name in _BOUNDARY_METRIC_FIELDS:
            if observed[name] != recomputed[name]:
                raise ValueError(f"course metric {name} differs from retained trajectory")
        # A transient/contact failure can only add producer evidence. When no
        # such evidence exists, posture compliance is boundary-recomputable.
        if (
            not observed["fallen"]
            and observed["posture_success"] != recomputed["posture_success"]
        ):
            raise ValueError("course metric posture_success differs from retained trajectory")
        boundary_reasons = set(recomputed["failure_reasons"])
        if not boundary_reasons <= set(observed["failure_reasons"]):
            raise ValueError("course boundary failure evidence differs from retained trajectory")


def _diagnosis(
    *,
    label: str,
    frames: list[dict[str, Any]],
    trajectory: dict[str, np.ndarray],
    objective: dict[str, object],
    posture_low: float,
    posture_high: float,
) -> str:
    metrics = [frame["metrics"] for frame in frames]
    inside_indices = [
        index for index, metric in enumerate(metrics) if metric["inside_posture_region"]
    ]
    lines = [
        f"Measured development trace: {label}; {len(frames)} frames; "
        f"{len(frames) * CONTROL_DT_SECONDS:.3f} s."
    ]
    missing: list[str] = []
    if inside_indices:
        reference_heights = trajectory["current_reference"][inside_indices, 0].astype(float)
        actual_heights = np.asarray(
            [metrics[index]["root_height_m"] for index in inside_indices], dtype=np.float64
        )
        reference_compliance = float(
            np.mean((reference_heights >= posture_low) & (reference_heights <= posture_high))
        )
        actual_compliance = objective["region"]["posture_compliant_fraction"]
        minimum_offset = int(np.argmin(actual_heights))
        minimum_step = metrics[inside_indices[minimum_offset]]["control_step"]
        lines.extend(
            (
                "Posture region: "
                f"{len(inside_indices)} actual-region samples; actual compliance "
                f"{actual_compliance:.3f}; reference-height compliance at those samples "
                f"{reference_compliance:.3f}.",
                "Root height in actual region: mean actual-minus-reference bias "
                f"{fmean((actual_heights - reference_heights).tolist()):+.3f} m; "
                f"actual minimum {actual_heights[minimum_offset]:.3f} m at step "
                f"{minimum_step}; reference {reference_heights[minimum_offset]:.3f} m "
                "at that step.",
                "Inside-region speed: mean "
                f"{objective['speed']['inside_mean_forward_speed_m_s']:.3f} m/s; "
                f"target {objective['speed']['inside_target_speed_m_s']:.3f} m/s; "
                "mean absolute error "
                f"{objective['speed']['inside_mean_absolute_error_m_s']:.3f} m/s.",
            )
        )
    else:
        missing.append("actual posture region was not observed")
        lines.append(
            "Posture region: 0 actual-region samples; actual/reference compliance and "
            "height-bias comparisons are unavailable."
        )
        lines.append("Inside-region speed: mean and mean absolute error are unavailable.")
    region = objective["region"]
    entry_step = region["first_entry_step"]
    exit_step = region["first_exit_step"]
    entry_phase = (
        f"step {entry_step}, {frames[entry_step - 1]['executed_phase_seconds']:.3f} s"
        if entry_step is not None
        else "unavailable"
    )
    exit_phase = (
        f"step {exit_step}, {frames[exit_step - 1]['executed_phase_seconds']:.3f} s"
        if exit_step is not None
        else "unavailable"
    )
    lines.append(
        "Executed reference phase at actual region boundaries: "
        f"entry {entry_phase}; exit {exit_phase}."
    )
    mode_counts = Counter(frame["executed_mode"] for frame in frames)
    lines.append(
        "Executed-mode durations: "
        + ", ".join(
            f"{mode}={count * CONTROL_DT_SECONDS:.3f} s"
            for mode, count in sorted(mode_counts.items())
        )
        + "."
    )
    forward = [metric["forward_speed_m_s"] for metric in metrics]
    targets = [metric["target_speed_m_s"] for metric in metrics]
    speed_errors = [metric["speed_error_m_s"] for metric in metrics]
    heading = [metric["heading_error_rad"] for metric in metrics]
    lateral = [metric["lateral_error_m"] for metric in metrics]
    lines.append(
        "Motion: forward speed mean "
        f"{fmean(forward):.3f} m/s; target mean {fmean(targets):.3f} m/s; "
        f"absolute speed error mean {fmean(speed_errors):.3f} m/s; heading absolute "
        f"mean/max {fmean(heading):.3f}/{max(heading):.3f} rad; lateral absolute "
        f"mean/max {fmean(lateral):.3f}/{max(lateral):.3f} m."
    )
    if not objective["region"]["exit_observed_after_entry"]:
        missing.append("posture-region exit was not observed after entry")
    if not objective["finish_condition_observed"]:
        missing.append("finish condition was not observed")
    first_fall = objective["first_fall_step"]
    lines.append(
        f"Falls: {objective['fall_count']}; first fall step "
        f"{first_fall if first_fall is not None else 'none'}. Missing evidence: "
        f"{'; '.join(missing) if missing else 'none'}."
    )
    gates = objective["development_gate_results"]
    lines.append(
        f"Development gates: {sum(gates.values())}/{len(gates)} passed. "
        "These are fixed in-sample measurements, not held-out or task-success evidence. "
        "Transient substep failures remain producer-recorded evidence."
    )
    diagnosis = "\n".join(lines)
    if not diagnosis or len(diagnosis) > MAX_DIAGNOSIS_CHARACTERS:
        raise ValueError("course feedback diagnosis exceeds its contract")
    return diagnosis


def _training_telemetry_diagnosis(
    encoded: bytes, *, reward_scale: float | None = None
) -> str:
    rows = [
        decode_json_object(line, source="G1 training telemetry row")
        for line in encoded.splitlines()
    ]
    rollout_rows = rows[:-1]
    completed = [row for row in rollout_rows if row["episodes"]["completed"]]
    total_completed = sum(row["episodes"]["completed"] for row in rollout_rows)
    if completed:
        if reward_scale is None:
            first_return = completed[0]["episodes"]["returns"]["mean"]
            last_return = completed[-1]["episodes"]["returns"]["mean"]
            episode_text = (
                f"{total_completed} completed episodes; first/last available rollout "
                f"episode-return means {first_return:.6g}/{last_return:.6g}"
            )
        else:
            first_ppo = completed[0]["episodes"]["scaled_environment_returns"]["mean"]
            last_ppo = completed[-1]["episodes"]["scaled_environment_returns"]["mean"]
            first_raw = completed[0]["episodes"]["raw_environment_returns"]["mean"]
            last_raw = completed[-1]["episodes"]["raw_environment_returns"]["mean"]
            episode_text = (
                f"{total_completed} completed episodes; first/last available rollout "
                f"Scaled pre-bootstrap return means {first_ppo:.6g}/{last_ppo:.6g}; raw environment "
                f"return means {first_raw:.6g}/{last_raw:.6g}"
            )
    else:
        episode_text = "no completed episodes"
    update = rows[-1]["update"]
    metrics = update["metrics"]
    reasons = update["metric_unavailable_reasons"]

    def metric(name: str, label: str) -> str:
        value = metrics[name]
        return (
            f"{label}={value:.6g}"
            if value is not None
            else f"{label}=unavailable ({reasons[name]})"
        )

    attempted = update["sb3_n_updates"]
    attempted_text = (
        str(attempted) if attempted is not None else "unavailable (missing)"
    )
    summary = (
        f"Training telemetry: {episode_text}. Final update through "
        f"{update['trained_through_transitions']} transitions: "
        f"{metric('approx_kl', 'KL')}; {metric('clip_fraction', 'clip fraction')}; "
        f"{metric('explained_variance', 'explained variance')}; "
        f"{metric('value_loss', 'value loss')}; attempted PPO epochs including "
        f"KL-stopped partial epochs={attempted_text}. Descriptive only; these values do "
        "not establish convergence or a causal mechanism."
    )
    if reward_scale is not None:
        summary += (
            " PPO value loss is in scaled optimization-reward units and is not comparable "
            "to value loss from raw-reward runs."
        )
    return summary


def build_g1_course_feedback(
    *,
    manifest_path: Path,
    expected_manifest_sha256: str,
    label: str,
    output: Path,
) -> dict[str, object]:
    """Validate one completed development run and publish exact feedback/v1."""

    manifest_sha256 = _sha256(expected_manifest_sha256, field="manifest SHA-256")
    if label not in {"zero_residual", "final_policy"}:
        raise ValueError("feedback label must be zero_residual or final_policy")
    manifest_file = Path(manifest_path).resolve(strict=True)
    run_root = manifest_file.parent
    manifest = _json(
        _verified_bytes(manifest_file, manifest_sha256, _MAX_MANIFEST_BYTES),
        source="G1 course run manifest",
    )
    if (
        set(manifest) != _MANIFEST_FIELDS
        or manifest["schema_version"] != 1
        or manifest["artifact"] != "gmt_g1_course_development_run"
        or manifest["status"] != "completed"
    ):
        raise ValueError("course run manifest identity or fields differ")
    outputs = manifest["outputs"]
    if type(outputs) is not dict:
        raise ValueError("course run output ledger is malformed")
    config_sha256 = _sha256(manifest["input_config_sha256"], field="input config")
    output_names = set(outputs)
    train_output = output_names in (
        _TRAIN_OUTPUTS,
        _TRAIN_TELEMETRY_OUTPUTS,
        _TRAIN_SCALED_TELEMETRY_OUTPUTS,
    )
    mode_hint = "train" if train_output else "probe"
    if output_names not in (
        _PROBE_OUTPUTS,
        _TRAIN_OUTPUTS,
        _TRAIN_TELEMETRY_OUTPUTS,
        _TRAIN_SCALED_TELEMETRY_OUTPUTS,
    ):
        raise ValueError("course run output ledger differs")
    retained: dict[str, bytes] = {}
    telemetry_encoded = None
    selected = {
        "input_config.json",
        f"{label}_frames.jsonl",
        f"{label}_trajectory.npz",
        f"{label}_evaluation.json",
    }
    if not selected <= set(outputs):
        raise ValueError("selected course evidence is absent from this run")
    for name, digest in outputs.items():
        expected = _sha256(digest, field=f"course output {name}")
        maximum = (
            MAX_TELEMETRY_BYTES
            if name in {TELEMETRY_FILENAME, SCALED_TELEMETRY_FILENAME}
            else _MAX_FRAMES_BYTES
            if name.endswith("_frames.jsonl")
            else _MAX_NUMERIC_BYTES
            if name.endswith(".npz")
            else _MAX_JSON_BYTES
        )
        encoded = _verified_bytes(run_root / name, expected, maximum)
        if name in {TELEMETRY_FILENAME, SCALED_TELEMETRY_FILENAME}:
            telemetry_encoded = encoded
        if name in selected:
            retained[name] = encoded
    if outputs.get("input_config.json") != config_sha256:
        raise ValueError("retained config hash differs from manifest input identity")
    config_path = run_root / "input_config.json"
    config = load_run_config(config_path)
    if config.encoded != retained["input_config.json"] or config.sha256 != config_sha256:
        raise ValueError("admitted config differs from retained run bytes")
    if config.raw["mode"] != mode_hint:
        raise ValueError("course run mode and output ledger differ")
    if config.trainer is not None and output_names != _TRAIN_SCALED_TELEMETRY_OUTPUTS:
        raise ValueError("scaled trainer requires its exact v2 telemetry output")
    if config.trainer is None and output_names == _TRAIN_SCALED_TELEMETRY_OUTPUTS:
        raise ValueError("v2 telemetry requires the scaled trainer config")
    training = manifest["training"]
    has_descriptor = type(training) is dict and "telemetry" in training
    descriptor = training.get("telemetry") if has_descriptor else None
    if telemetry_encoded is None:
        if has_descriptor:
            raise ValueError("training telemetry descriptor lacks its output")
    else:
        validate_training_telemetry_descriptor(
            descriptor,
            telemetry_encoded,
            config.raw["training_steps"],
            reward_scale=(
                config.trainer.total_training_reward_scale
                if config.trainer is not None
                else None
            ),
        )
    has_preconditioning = type(training) is dict and "reward_preconditioning" in training
    if config.trainer is None:
        if has_preconditioning:
            raise ValueError("legacy training record contains preconditioning metadata")
    elif (
        not has_preconditioning
        or training["reward_preconditioning"] != training_reward_metadata(config.trainer)
    ):
        raise ValueError("scaled training reward metadata differs")
    frozen_runtime = manifest["frozen_runtime"]
    expected_runtime = frozen_runtime_contract(
        config.runtime,
        trainer=effective_training_contract(config.trainer),
        residual_raw_scale=COURSE_RESIDUAL_RAW_SCALE,
    )
    if frozen_runtime != expected_runtime:
        raise ValueError("course runtime differs from retained config")
    identities = {
        "task": config.task.sha256,
        "oracle": config.program.sha256,
        "reward": config.recipe.sha256,
        "segments": {name: segment.sha256 for name, segment in config.segments.items()},
    }
    if manifest["identities"] != identities:
        raise ValueError("course run semantic identities differ from retained config")
    if label == "final_policy" and mode_hint != "train":
        raise ValueError("final_policy feedback requires a completed train-mode run")
    frames = _load_frames(retained[f"{label}_frames.jsonl"])
    trajectory = _load_trajectory(retained[f"{label}_trajectory.npz"], len(frames))
    _verify_frame_crosslinks(frames, trajectory)
    _verify_boundary_metrics(spec=config.task, frames=frames, trajectory=trajectory)
    objective = evaluate_episode(spec=config.task, frames=frames)
    report = _json(
        retained[f"{label}_evaluation.json"], source=f"{label} course evaluation"
    )
    if set(report) != _SUMMARY_FIELDS or manifest[label] != report:
        raise ValueError("selected course evaluation and manifest summary differ")
    if report["steps"] != len(frames) or report["objective_evaluation"] != objective:
        raise ValueError("stored course objective differs from exact recomputation")
    if objective["trajectory_frame_count"] != len(frames):
        raise ValueError("selected objective does not bind every trajectory frame")
    evaluation = {name: objective[name] for name in _EVALUATION_FIELDS}
    if evaluation["episode_success"] is not None:
        raise ValueError("development feedback cannot promote episode success")
    diagnosis = _diagnosis(
        label=label,
        frames=frames,
        trajectory=trajectory,
        objective=objective,
        posture_low=config.task.posture_band_low_m,
        posture_high=config.task.posture_band_high_m,
    )
    if telemetry_encoded is not None:
        reward_scale = (
            config.trainer.total_training_reward_scale
            if config.trainer is not None
            else None
        )
        diagnosis = (
            f"{diagnosis}\n"
            f"{_training_telemetry_diagnosis(telemetry_encoded, reward_scale=reward_scale)}"
        )
        if len(diagnosis) > MAX_DIAGNOSIS_CHARACTERS:
            raise ValueError("course feedback diagnosis exceeds its contract")
    feedback = {
        "evidence_class": COURSE_FEEDBACK_EVIDENCE_CLASS,
        "protected_evaluation": False,
        "source_manifest_sha256": manifest_sha256,
        "evaluation": evaluation,
        "diagnosis": diagnosis,
    }
    destination = Path(output)
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    feedback_artifact = publish_json_without_overwrite(destination / "feedback_v1.json", feedback)
    receipt = {
        "schema_version": 1,
        "artifact": "gmt_g1_course_feedback_build_receipt/v1",
        "status": "completed",
        "inputs": {
            "source_manifest_path": str(manifest_file),
            "source_manifest_sha256": manifest_sha256,
            "input_config_sha256": config_sha256,
            "label": label,
            "selected_outputs": {
                name: outputs[name]
                for name in sorted(selected - {"input_config.json"})
            },
        },
        "output": {
            "path": feedback_artifact.path.name,
            "sha256": feedback_artifact.sha256,
            "byte_count": feedback_artifact.byte_count,
        },
        "claim_limits": [
            "development_only",
            "no_causal_inference",
            "no_heldout_or_task_success_promotion",
            "transient_substep_failures_remain_producer_evidence",
        ],
    }
    runtime = config.runtime.manifest_contract()
    if runtime is not None:
        receipt["inputs"]["course_runtime"] = runtime
    receipt_artifact = publish_json_without_overwrite(
        destination / "feedback_receipt_v1.json", receipt
    )
    return {
        "feedback": {
            "path": str(feedback_artifact.path),
            "sha256": feedback_artifact.sha256,
            "byte_count": feedback_artifact.byte_count,
        },
        "receipt": {
            "path": str(receipt_artifact.path),
            "sha256": receipt_artifact.sha256,
            "byte_count": receipt_artifact.byte_count,
        },
    }


__all__ = ["build_g1_course_feedback"]
