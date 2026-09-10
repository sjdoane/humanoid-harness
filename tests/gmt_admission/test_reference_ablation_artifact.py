from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from oracle_composition.adapters.gmt.composition import ComposedReference, ReferenceSegment
from oracle_composition.adapters.gmt.contracts import ACTION_DIM
from oracle_composition.adapters.gmt.control_runtime import PreparedControl
from oracle_composition.adapters.gmt.course_evaluation import evaluate_episode
from oracle_composition.adapters.gmt.course_runtime import LEGACY_RUNTIME
from oracle_composition.adapters.gmt.course_task import (
    CourseTaskSpec,
    TaskFrame,
    TaskRewardRecipe,
    evaluate_step,
)
from oracle_composition.adapters.gmt.deployment import ObservationHistory, build_proprioception
from oracle_composition.adapters.gmt.io import (
    sha256_file,
    write_deterministic_npz,
    write_json_receipt,
)
from oracle_composition.adapters.gmt.reference_ablation import (
    REFERENCE_ABLATION_ARMS,
    REFERENCE_ABLATION_ARTIFACT,
    REFERENCE_ABLATION_EVIDENCE_CLASS,
    REFERENCE_ABLATION_MANIFEST_FILENAME,
    AblatedComposedReference,
    reference_ablation_contract,
)
from oracle_composition.adapters.gmt.reference_ablation_artifact import (
    validate_reference_ablation_artifact,
)
from oracle_composition.adapters.gmt.reference_ablation_evidence import (
    measure_practical_trajectory_divergence,
)
from oracle_composition.adapters.gmt.reference_ablation_run import (
    _CapturedControl,
    _retain_actor_evidence,
)
from oracle_composition.adapters.gmt.reference_runtime import ReferenceMotion
from oracle_composition.feedback.g1_course import build_g1_course_feedback
from oracle_composition.harness.contract import oracle_program_from_dict


class _ReferenceActor(torch.nn.Module):
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return observation[:, :ACTION_DIM]


def _motion() -> ReferenceMotion:
    frames = 601
    root = np.zeros((frames, 3), dtype="<f4")
    root[:, 0] = np.linspace(0.0, 4.0, frames, dtype="<f4")
    root[:, 2] = np.float32(0.8)
    rotation = np.zeros((frames, 4), dtype="<f4")
    rotation[:, 3] = np.float32(1.0)
    joints = np.zeros((frames, ACTION_DIM), dtype="<f4")
    joints[:, 0] = np.linspace(-0.1, 0.1, frames, dtype="<f4")
    return ReferenceMotion(
        {
            "fps": np.asarray([30.0], dtype="<f8"),
            "root_pos": root,
            "root_rot": rotation,
            "dof_pos": joints,
        }
    )


def _program_and_segments():
    program = oracle_program_from_dict(
        {
            "schema_version": 1,
            "evidence_class": "exploratory_oracle_cycle",
            "oracle_id": "reference_ablation_fixture",
            "behaviors": ["walk", "crouch"],
            "initial": "before",
            "states": {
                "before": {"behavior": "walk", "min_dwell": 1},
                "inside": {"behavior": "crouch", "min_dwell": 1},
                "after": {"behavior": "walk", "min_dwell": 1},
            },
            "transitions": [
                {
                    "from": "before",
                    "to": "inside",
                    "priority": 0,
                    "guard": "x_travelled >= 1",
                },
                {
                    "from": "inside",
                    "to": "after",
                    "priority": 0,
                    "guard": "x_travelled >= 2",
                },
            ],
        },
        available_behaviors=["walk", "crouch"],
    )
    motion = _motion()
    return program, {
        "walk": ReferenceSegment(motion, "a" * 64, 0.0, 12.0),
        "crouch": ReferenceSegment(motion, "b" * 64, 0.0, 12.0),
    }


def _config(config_path: Path):
    task = CourseTaskSpec(
        region_entry_distance_m=1.0,
        region_exit_distance_m=2.0,
        finish_distance_m=3.0,
        target_speed_outside_m_s=0.5,
        target_speed_inside_m_s=0.3,
        posture_band_low_m=0.4,
        posture_band_high_m=0.7,
        horizon_steps=2,
    )
    recipe = TaskRewardRecipe(1.0, 1.0, 1.0, 1.0, 1.0)
    program, segments = _program_and_segments()
    encoded = b'{"fixture":"reference-ablation"}\n'
    config_path.write_bytes(encoded)
    return SimpleNamespace(
        raw={"mode": "probe", "training_steps": 0, "seed": 17},
        encoded=encoded,
        sha256=hashlib.sha256(encoded).hexdigest(),
        assets={"weights": {"path": str(config_path), "sha256": "c" * 64}},
        task=task,
        recipe=recipe,
        program=program,
        segments=segments,
        trainer=None,
        runtime=LEGACY_RUNTIME,
    )


def _qpos(step: int) -> np.ndarray:
    result = np.zeros(30, dtype="<f8")
    result[:3] = (0.01 * step, 0.0, 0.8)
    result[3] = 1.0
    return result


def _qvel() -> np.ndarray:
    result = np.zeros(29, dtype="<f8")
    result[0] = 0.5
    return result


def _write_frames(path: Path, frames: list[dict]) -> str:
    path.write_text(
        "".join(
            json.dumps(frame, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
            for frame in frames
        ),
        encoding="utf-8",
    )
    return sha256_file(path)


def _reset_metadata(config) -> dict[str, object]:
    result = {
        "runtime_id": config.runtime.gym_runtime_id,
        "signal_contract_id": "gmt_initial_heading_frame_boundary_signals/v1",
        "reset_distribution": "fixed_home_keyframe_one_warmup_step",
        "seed_effect": "policy_training_rng_only_no_reset_randomization",
        "task_sha256": config.task.sha256,
        "oracle_sha256": config.program.sha256,
        "reward_sha256": config.recipe.sha256,
    }
    runtime = config.runtime.manifest_contract()
    if runtime is not None:
        result["course_runtime"] = runtime
    return result


def _rewrite_receipt(path: Path, payload: dict) -> str:
    replacement = path.with_name(f"{path.name}.replacement")
    digest = write_json_receipt(replacement, payload)
    path.unlink()
    replacement.rename(path)
    return digest


def _rewrite_npz(path: Path, arrays: dict[str, np.ndarray]) -> str:
    replacement = path.with_name(f"{path.name}.replacement")
    digest = write_deterministic_npz(replacement, arrays)
    path.unlink()
    replacement.rename(path)
    return digest


def _inject_json_member(encoded: bytes, member: bytes) -> bytes:
    opening = encoded.index(b"{") + 1
    return encoded[:opening] + member + encoded[opening:]


def _arm_fixture(root: Path, config, actor: torch.nn.Module, arm: str):
    exact_oracle = ComposedReference(config.program, config.segments)
    oracle = AblatedComposedReference(exact_oracle, arm=arm)
    oracle.reset()
    qpos = np.asarray([_qpos(step) for step in range(3)], dtype="<f8")
    qvel = np.asarray([_qvel() for _ in range(3)], dtype="<f8")
    task_frame = TaskFrame.initialize(qpos[0, :2], qpos[0, 3:7])
    history = ObservationHistory()
    last_raw = np.zeros(ACTION_DIM, dtype="<f4")
    zero = np.zeros(ACTION_DIM, dtype="<f4")
    captured = []
    frames = []
    current_references = []
    composites = []
    for step in range(config.task.horizon_steps):
        command = oracle.command(
            step=step,
            signals={
                "t": step * 0.02,
                "v_x": 0.5,
                "v_target": 0.5,
                "z_root": 0.8,
                "torso_up": 1.0,
                "x_travelled": 0.01 * step,
            },
            robot_pose=np.asarray([0.8, *([0.0] * 25)], dtype="<f8"),
        )
        orientation = np.asarray([1.0, 0.0, 0.0, 0.0], dtype="<f4")
        angular_velocity = np.zeros(3, dtype="<f4")
        proprio = build_proprioception(
            dof_position=qpos[step, -ACTION_DIM:].astype("<f4"),
            dof_velocity=qvel[step, -ACTION_DIM:].astype("<f4"),
            orientation_wxyz=orientation,
            angular_velocity=angular_velocity,
            last_raw_action=last_raw,
        )
        observation = history.assemble(command.window, proprio)
        with torch.inference_mode():
            base = actor(torch.from_numpy(observation).unsqueeze(0))[0].numpy()
        prepared = PreparedControl(
            control_step=step,
            proprio=np.ascontiguousarray(proprio, dtype="<f8"),
            obs=np.ascontiguousarray(observation, dtype="<f4"),
            base_raw=np.ascontiguousarray(base, dtype="<f4"),
        )
        captured.append(
            _CapturedControl(
                audit=oracle.last_audit,
                prepared=prepared,
                residual=zero.copy(),
                orientation_wxyz=orientation,
                angular_velocity=angular_velocity,
            )
        )
        current_reference = oracle.current_after_step(step + 1).astype("<f4")
        current_references.append(current_reference)
        composites.append(prepared.base_raw)
        metrics = evaluate_step(
            spec=config.task,
            frame=task_frame,
            before_qpos=qpos[step],
            after_qpos=qpos[step + 1],
            ground_contact_bodies=(),
            current_reference=current_reference,
            control_step=step + 1,
        )
        frames.append(
            {
                "metrics": metrics.to_dict(),
                "executed_mode": command.state,
                "executed_behavior": command.behavior,
                "executed_phase_seconds": command.phase_seconds,
                "transition": None if command.transition is None else dict(command.transition),
                "action_saturation_fraction": 0.0,
                "torque_saturation_fraction": 0.0,
                "trajectory": {
                    "qpos": qpos[step + 1].tolist(),
                    "qvel": qvel[step + 1].tolist(),
                    "current_reference": current_reference.tolist(),
                    "composite_raw_action": prepared.base_raw.tolist(),
                    "contact_pairs": [],
                    "geom_body_names": ["world"],
                },
            }
        )
        history.append(proprio)
        last_raw = prepared.base_raw

    outputs = {}
    frames_path = root / f"{arm}_frames.jsonl"
    outputs[frames_path.name] = _write_frames(frames_path, frames)
    trajectory_path = root / f"{arm}_trajectory.npz"
    outputs[trajectory_path.name] = write_deterministic_npz(
        trajectory_path,
        {
            "qpos": qpos,
            "qvel": qvel,
            "residual_action": np.zeros((2, ACTION_DIM), dtype="<f4"),
            "current_reference": np.asarray(current_references, dtype="<f4"),
            "composite_raw_action": np.asarray(composites, dtype="<f4"),
        },
    )
    objective = evaluate_episode(spec=config.task, frames=frames)
    report = {
        "objective_evaluation": objective,
        "training_reward_sum_not_success_metric": 0.0,
        "reset": _reset_metadata(config),
        "steps": 2,
        "residual_rms": 0.0,
    }
    evaluation_path = root / f"{arm}_evaluation.json"
    outputs[evaluation_path.name] = write_json_receipt(evaluation_path, report)
    actor_outputs, count = _retain_actor_evidence(
        output=root,
        arm=arm,
        captured=captured,
        standard_trace=trajectory_path,
        standard_frames=frames_path,
        actor=actor,
        segments=config.segments,
    )
    outputs.update(actor_outputs)
    return {**report, "actor_input_frame_count": count}, outputs, qpos


def _artifact_fixture(root: Path, config, actor: torch.nn.Module) -> tuple[Path, str]:
    root.mkdir()
    (root / "input_config.json").write_bytes(config.encoded)
    outputs = {"input_config.json": config.sha256}
    summaries = {}
    qpos = {}
    for arm in REFERENCE_ABLATION_ARMS:
        summaries[arm], arm_outputs, qpos[arm] = _arm_fixture(root, config, actor, arm)
        outputs.update(arm_outputs)
    comparisons = {
        arm: measure_practical_trajectory_divergence(qpos["exact"], qpos[arm])
        for arm in REFERENCE_ABLATION_ARMS[1:]
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
            "actor": "c" * 64,
            "task": config.task.sha256,
            "oracle": config.program.sha256,
            "reward": config.recipe.sha256,
            "segments": {name: segment.sha256 for name, segment in config.segments.items()},
            "course_runtime": None,
        },
        "arms": summaries,
        "treated_vs_exact": comparisons,
        "runtime": {
            "wall_seconds": 1.0,
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "platform": "fixture",
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
    path = root / REFERENCE_ABLATION_MANIFEST_FILENAME
    return path, write_json_receipt(path, manifest)


def _validate(path: Path, digest: str, config_path: Path, config):
    return validate_reference_ablation_artifact(
        manifest_path=path,
        expected_manifest_sha256=digest,
        config_path=config_path,
        expected_config_sha256=config.sha256,
    )


def test_validates_fixed_five_arm_artifact_and_rejects_ordinary_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    config = _config(config_path)
    actor = _ReferenceActor().eval()
    manifest_path, digest = _artifact_fixture(tmp_path / "run", config, actor)
    monkeypatch.setattr(
        "oracle_composition.adapters.gmt.reference_ablation_artifact.load_run_config",
        lambda _: config,
    )
    monkeypatch.setattr(
        "oracle_composition.adapters.gmt.reference_ablation_artifact.load_actor",
        lambda *_, **__: actor,
    )

    result = _validate(manifest_path, digest, config_path, config)

    assert list(result["arms"]) == list(REFERENCE_ABLATION_ARMS)
    assert result["claim_ceiling"].endswith("fixed_tested_interventions_only")
    with pytest.raises(ValueError, match="manifest identity or fields differ"):
        build_g1_course_feedback(
            manifest_path=manifest_path,
            expected_manifest_sha256=digest,
            label="zero_residual",
            output=tmp_path / "feedback",
        )


def test_revalidates_after_exact_supervisor_receipt_publication_and_rejects_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    config = _config(config_path)
    actor = _ReferenceActor().eval()
    manifest_path, digest = _artifact_fixture(tmp_path / "run", config, actor)
    monkeypatch.setattr(
        "oracle_composition.adapters.gmt.reference_ablation_artifact.load_run_config",
        lambda _: config,
    )
    monkeypatch.setattr(
        "oracle_composition.adapters.gmt.reference_ablation_artifact.load_actor",
        lambda *_, **__: actor,
    )
    receipt = manifest_path.parent / "gmt_probe_resource_receipt_v1.json"
    write_json_receipt(manifest_path.parent / "child_stdout.json", {"ok": True})
    (manifest_path.parent / "child_stderr.log").write_text("", encoding="utf-8")
    write_json_receipt(receipt, {"outer_provenance": "parent-pins-these-bytes"})

    _validate(manifest_path, digest, config_path, config)

    receipt.unlink()
    receipt.symlink_to(config_path)
    with pytest.raises(ValueError, match="supervisor receipt must be a regular non-linked file"):
        _validate(manifest_path, digest, config_path, config)


@pytest.mark.parametrize(
    ("array_name", "index", "replacement", "message"),
    [
        ("qpos", (0, 3), -1.0, "initial qpos differs"),
        ("qvel", (0, 2), 0.125, "initial qvel differs"),
    ],
)
def test_rejects_one_arm_initial_state_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    array_name: str,
    index: tuple[int, int],
    replacement: float,
    message: str,
) -> None:
    config_path = tmp_path / "config.json"
    config = _config(config_path)
    actor = _ReferenceActor().eval()
    manifest_path, _ = _artifact_fixture(tmp_path / "run", config, actor)
    monkeypatch.setattr(
        "oracle_composition.adapters.gmt.reference_ablation_artifact.load_run_config",
        lambda _: config,
    )
    monkeypatch.setattr(
        "oracle_composition.adapters.gmt.reference_ablation_artifact.load_actor",
        lambda *_, **__: actor,
    )
    manifest = json.loads(manifest_path.read_text())
    trajectory_path = manifest_path.parent / "zero_reference_trajectory.npz"
    with np.load(trajectory_path, allow_pickle=False) as archive:
        arrays = {name: np.ascontiguousarray(archive[name]) for name in archive.files}
    arrays[array_name][index] = replacement
    manifest["outputs"][trajectory_path.name] = _rewrite_npz(trajectory_path, arrays)
    digest = _rewrite_receipt(manifest_path, manifest)

    with pytest.raises(ValueError, match=message):
        _validate(manifest_path, digest, config_path, config)


def test_rejects_shared_but_unexpected_reset_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "config.json"
    config = _config(config_path)
    actor = _ReferenceActor().eval()
    manifest_path, _ = _artifact_fixture(tmp_path / "run", config, actor)
    monkeypatch.setattr(
        "oracle_composition.adapters.gmt.reference_ablation_artifact.load_run_config",
        lambda _: config,
    )
    monkeypatch.setattr(
        "oracle_composition.adapters.gmt.reference_ablation_artifact.load_actor",
        lambda *_, **__: actor,
    )
    manifest = json.loads(manifest_path.read_text())
    for arm in REFERENCE_ABLATION_ARMS:
        report_path = manifest_path.parent / f"{arm}_evaluation.json"
        report = json.loads(report_path.read_text())
        report["reset"] = {"same_across_arms_but_not_runtime_derived": True}
        manifest["outputs"][report_path.name] = _rewrite_receipt(report_path, report)
        manifest["arms"][arm] = {**report, "actor_input_frame_count": 2}
    digest = _rewrite_receipt(manifest_path, manifest)

    with pytest.raises(ValueError, match="reset metadata differs from the fixed profile"):
        _validate(manifest_path, digest, config_path, config)


@pytest.mark.parametrize("source", ["manifest", "audit", "report"])
@pytest.mark.parametrize(
    ("malformation", "member", "message"),
    [
        ("duplicate", None, "duplicate JSON key"),
        ("nonfinite", b'"invalid":NaN,', "non-finite JSON constant"),
    ],
)
def test_rejects_non_strict_json_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source: str,
    malformation: str,
    member: bytes | None,
    message: str,
) -> None:
    config_path = tmp_path / "config.json"
    config = _config(config_path)
    actor = _ReferenceActor().eval()
    manifest_path, digest = _artifact_fixture(tmp_path / "run", config, actor)
    monkeypatch.setattr(
        "oracle_composition.adapters.gmt.reference_ablation_artifact.load_run_config",
        lambda _: config,
    )
    monkeypatch.setattr(
        "oracle_composition.adapters.gmt.reference_ablation_artifact.load_actor",
        lambda *_, **__: actor,
    )
    manifest = json.loads(manifest_path.read_text())
    duplicate_members = {
        "manifest": b'"status":"completed",',
        "audit": b'"arm":"exact",',
        "report": b'"steps":2,',
    }
    injected = duplicate_members[source] if malformation == "duplicate" else member
    assert injected is not None
    if source == "manifest":
        encoded = _inject_json_member(manifest_path.read_bytes(), injected)
        manifest_path.write_bytes(encoded)
        digest = hashlib.sha256(encoded).hexdigest()
    else:
        suffix = "reference_audit.jsonl" if source == "audit" else "evaluation.json"
        target = manifest_path.parent / f"exact_{suffix}"
        encoded = target.read_bytes()
        if source == "audit":
            lines = encoded.splitlines(keepends=True)
            first = lines[0].removesuffix(b"\n")
            lines[0] = _inject_json_member(first, injected) + b"\n"
            encoded = b"".join(lines)
        else:
            encoded = _inject_json_member(encoded, injected)
        target.write_bytes(encoded)
        manifest["outputs"][target.name] = hashlib.sha256(encoded).hexdigest()
        digest = _rewrite_receipt(manifest_path, manifest)

    with pytest.raises(ValueError, match=message):
        _validate(manifest_path, digest, config_path, config)


@pytest.mark.parametrize("mutation", ["hash", "forged_metrics", "extra_file"])
def test_rejects_tampered_or_unscoped_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    config_path = tmp_path / "config.json"
    config = _config(config_path)
    actor = _ReferenceActor().eval()
    manifest_path, digest = _artifact_fixture(tmp_path / "run", config, actor)
    monkeypatch.setattr(
        "oracle_composition.adapters.gmt.reference_ablation_artifact.load_run_config",
        lambda _: config,
    )
    monkeypatch.setattr(
        "oracle_composition.adapters.gmt.reference_ablation_artifact.load_actor",
        lambda *_, **__: actor,
    )
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if mutation == "hash":
        (root / "zero_reference_actor_input.npz").write_bytes(b"changed")
    elif mutation == "extra_file":
        (root / "unscoped.txt").write_text("not evidence")
    else:
        frames_path = root / "exact_frames.jsonl"
        frames = [json.loads(line) for line in frames_path.read_text().splitlines()]
        frames[0]["metrics"]["root_height_m"] += 0.01
        manifest["outputs"][frames_path.name] = _write_frames(frames_path, frames)
        objective = evaluate_episode(spec=config.task, frames=frames)
        evaluation_path = root / "exact_evaluation.json"
        report = json.loads(evaluation_path.read_text())
        report["objective_evaluation"] = objective
        manifest["outputs"][evaluation_path.name] = write_json_receipt(
            evaluation_path.with_suffix(".replacement"), report
        )
        evaluation_path.unlink()
        evaluation_path.with_suffix(".replacement").rename(evaluation_path)
        manifest["arms"]["exact"] = {**report, "actor_input_frame_count": 2}
        manifest_path.unlink()
        digest = write_json_receipt(manifest_path, manifest)

    with pytest.raises(
        (ValueError, OSError),
        match=r"hash differs|unknown or missing|metric root_height_m differs",
    ):
        _validate(manifest_path, digest, config_path, config)
