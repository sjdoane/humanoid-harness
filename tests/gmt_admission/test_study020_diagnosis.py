from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oracle_composition.adapters.gmt.course_evaluation import evaluate_episode
from oracle_composition.adapters.gmt.course_runtime import FOUR_STATE_FINITE_HORIZON_RUNTIME
from oracle_composition.adapters.gmt.course_task import (
    CourseTaskSpec,
    TaskFrame,
    TaskRewardRecipe,
    evaluate_step,
    reward,
)
from oracle_composition.adapters.gmt.io import write_json_receipt

MODULE = Path(__file__).parents[2] / "experiments/020_g1_saved_policy_diagnostic/diagnosis.py"
specification = importlib.util.spec_from_file_location("study020_diagnosis", MODULE)
diagnosis = importlib.util.module_from_spec(specification)
specification.loader.exec_module(diagnosis)


@pytest.mark.parametrize("mutation", [None, "noise", "action", "raw", "counter", "dtype", "std"])
def test_sampling_reconstructs_exact_actions_and_rejects_inconsistent_evidence(mutation):
    noise = np.full((1000, 23), 0.25, dtype="<f4")
    mean = np.tile(np.linspace(-1.4, 1.4, 23, dtype="<f4"), (5, 1))
    std = np.full_like(mean, 0.5)
    raw = mean + noise[:5] * std
    actions = np.clip(raw, -1, 1)
    arrays = {"policy_mean": mean, "policy_std": std, "unclipped_action": raw.copy()}
    receipt = {
        "clipped_component_count": int(np.count_nonzero(raw != actions)),
        "total_component_count": actions.size,
        "max_abs_unclipped_action": float(np.max(np.abs(raw))),
    }
    if mutation == "noise":
        noise[0, 10] += 0.1
    elif mutation == "action":
        actions[0, 10] += 0.1
    elif mutation == "raw":
        arrays["unclipped_action"][0, 10] += 0.1
    elif mutation == "counter":
        receipt["clipped_component_count"] += 1
    elif mutation == "dtype":
        arrays["policy_mean"] = mean.astype(np.float64)
    elif mutation == "std":
        arrays["policy_std"][0, 10] = -1
    stream = io.BytesIO()
    np.savez(stream, **arrays)

    def verify():
        evidence = diagnosis.sampled_action_evidence(stream.getvalue(), 5)
        return diagnosis.verify_sampled_actions(noise, evidence, actions, receipt)

    if mutation is None:
        assert verify() == receipt["clipped_component_count"] / actions.size
    else:
        with pytest.raises(ValueError):
            verify()


def test_cli_rejects_consistently_rehashed_missing_parity_before_episode_analysis(
    tmp_path, monkeypatch
):
    fixture_path = Path(__file__).with_name("test_saved_policy_diagnostic.py")
    fixture_spec = importlib.util.spec_from_file_location("study020_artifact_fixture", fixture_path)
    fixture = importlib.util.module_from_spec(fixture_spec)
    fixture_spec.loader.exec_module(fixture)
    _, config = fixture._write_config(tmp_path)
    output = fixture._complete_output(tmp_path, config)
    manifest_path = output / "saved_policy_diagnostic_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    missing = "initial_deterministic_parity.json"
    (output / missing).unlink()
    manifest["outputs"].pop(missing)
    manifest_path.unlink()
    manifest_sha = write_json_receipt(manifest_path, manifest)
    source = "a" * 40
    receipt = {
        "status": "succeeded",
        "commit": source,
        "inputs": {"config": {"sha256": config.sha256}},
        "artifacts": {
            "saved_policy_diagnostic_manifest": {
                "path": manifest_path.name,
                "sha256": manifest_sha,
                "size": manifest_path.stat().st_size,
            },
            "outputs": {
                name: {"path": name, "sha256": digest, "size": (output / name).stat().st_size}
                for name, digest in manifest["outputs"].items()
            },
        },
    }
    resource_sha = write_json_receipt(output / "gmt_probe_resource_receipt_v1.json", receipt)
    monkeypatch.setattr(
        diagnosis.subprocess,
        "check_output",
        lambda argv, **kwargs: source if "rev-parse" in argv else "",
    )
    with pytest.raises(ValueError, match="output ledger fields differ"):
        diagnosis.score_run(output, manifest_sha, resource_sha, source)
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == manifest_sha


def episode(*, fall: bool = False):
    task = CourseTaskSpec(0.2, 0.55, 1.0, 1.0, 0.65, 0.3, 0.6, 60)
    recipe = TaskRewardRecipe(1.0, 1.0, 1.0, 1.0, 5.0)
    config = SimpleNamespace(task=task, recipe=recipe, runtime=FOUR_STATE_FINITE_HORIZON_RUNTIME)
    count = 10 if fall else 60
    qpos = np.zeros((count + 1, 30), dtype="<f8")
    qpos[:, 0] = np.arange(count + 1) * 0.02
    qpos[:, 1] = np.arange(count + 1) * 0.001
    qpos[:, 2] = 0.55
    qpos[:, 3] = 1.0
    if fall:
        qpos[-1, 2] = 0.2
    qvel = np.zeros((count + 1, 29), dtype="<f8")
    qvel[:, 0] = 1.0
    references = np.zeros((count, 30), dtype="<f4")
    references[:, 0] = 0.55
    actions = np.zeros((count, 23), dtype="<f4")
    trajectory = dict(
        qpos=qpos,
        qvel=qvel,
        current_reference=references,
        residual_action=actions,
        composite_raw_action=actions.copy(),
    )
    frame = TaskFrame.initialize(qpos[0, :2], qpos[0, 3:7])
    frames = []
    for index in range(count):
        metrics = evaluate_step(
            spec=task,
            frame=frame,
            before_qpos=qpos[index],
            after_qpos=qpos[index + 1],
            ground_contact_bodies=(),
            current_reference=references[index].astype(float),
            control_step=index + 1,
        )
        frames.append(
            {
                "metrics": metrics.to_dict(),
                "reward": reward(spec=task, recipe=recipe, metrics=metrics).to_dict(),
                "executed_mode": "before" if index < 5 else "inside",
                "executed_behavior": "walk" if index < 5 else "crouch",
                "executed_phase_seconds": float(index * 0.02),
                "transition": {
                    "control_step": 5,
                    "from_state": "before",
                    "to_state": "inside",
                    "from_behavior": "walk",
                    "to_behavior": "crouch",
                    "selected_phase_seconds": 0.1,
                }
                if index == 5
                else None,
                "action_saturation_fraction": 0.0,
                "torque_saturation_fraction": 0.0,
                "trajectory": {
                    "qpos": qpos[index + 1].tolist(),
                    "qvel": qvel[index + 1].tolist(),
                    "current_reference": references[index].tolist(),
                    "composite_raw_action": actions[index].tolist(),
                    "geom_body_names": [],
                    "contact_pairs": [],
                },
            }
        )
    report = {
        "steps": count,
        "objective_evaluation": evaluate_episode(spec=task, frames=frames, runtime=config.runtime),
        "training_reward_sum_not_success_metric": sum(x["reward"]["total_reward"] for x in frames),
        "residual_rms": 0.0,
    }
    return config, frames, trajectory, report


@pytest.mark.parametrize("fall", [False, True])
def test_episode_recomputes_returns_and_pre_action_handover(fall):
    result = diagnosis.episode_summary(*episode(fall=fall))
    assert result["fall"] is fall
    assert result["early_termination_censors_later_exposure"] is fall
    assert result["handovers"][0]["action_index"] == 5
    assert result["handovers"][0]["pre_action_progress_m"] == 0.1
    assert result["handovers"][0]["pre_action_lateral_m"] == 0.005
    assert result["handovers"][0]["pre_action_course_forward_qvel_m_s"] == 1.0
    assert result["handovers"][0]["observed_seconds_until_episode_end"] == pytest.approx(
        0.1 if fall else 1.1
    )


@pytest.mark.parametrize("target", ["metric", "reward", "report", "trajectory", "rms", "action"])
def test_episode_rejects_contradictory_evidence(target):
    config, frames, trajectory, report = episode()
    if target == "metric":
        frames[0]["metrics"]["root_height_m"] += 0.1
    elif target == "reward":
        frames[0]["reward"]["total_reward"] += 0.1
    elif target == "report":
        report["training_reward_sum_not_success_metric"] += 0.1
    elif target == "trajectory":
        trajectory["qpos"][1, 2] += 0.1
    elif target == "rms":
        report["residual_rms"] += 0.1
    else:
        trajectory["residual_action"][0, 0] = 1.1
    with pytest.raises(ValueError):
        diagnosis.episode_summary(config, frames, trajectory, report)


def pairs():
    rows = []
    for seed in diagnosis.NOISE_SEEDS:
        for policy in ("initial", "final"):
            rows.append(
                {
                    "noise_seed": seed,
                    "policy": policy,
                    "raw_total_reward_sum": 10.0 + seed + (2 if policy == "final" else 0),
                    "fall": seed % 4 in ((2, 3) if policy == "initial" else (1, 3)),
                    "duration_seconds": 20.0,
                }
            )
    return rows


def test_pairing_retains_all_differences_and_four_fall_cells():
    result = diagnosis.paired_summary(list(reversed(pairs())), list(diagnosis.NOISE_SEEDS))
    assert list(x["noise_seed"] for x in result["pairs"]) == list(diagnosis.NOISE_SEEDS)
    assert set(result["fall_cells"].values()) == {4}
    assert result["initial_falls"] == result["final_falls"] == 8
    assert result["paired_return_difference"] == {
        "count": 16,
        "mean": 2.0,
        "median": 2.0,
        "minimum": 2.0,
        "maximum": 2.0,
    }
    assert result["adoption_decision"] == "none"


@pytest.mark.parametrize("failure", ["missing", "duplicate", "unknown", "nan", "boolean_fall"])
def test_pairing_rejects_missing_or_invalid_outcomes(failure):
    rows = copy.deepcopy(pairs())
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows.append(rows[0])
    elif failure == "unknown":
        rows[0]["policy"] = "best"
    elif failure == "nan":
        rows[0]["raw_total_reward_sum"] = float("nan")
    else:
        rows[0]["fall"] = 1
    with pytest.raises(ValueError):
        diagnosis.paired_summary(rows, list(diagnosis.NOISE_SEEDS))


@pytest.mark.parametrize("failure", ["missing", "wrong_index", "wrong_mode", "spurious"])
def test_transition_table_rejects_contradictory_markers(failure):
    config, frames, trajectory, report = episode()
    if failure == "missing":
        frames[5]["transition"] = None
    elif failure == "wrong_index":
        frames[5]["transition"]["control_step"] = 6
    elif failure == "wrong_mode":
        frames[5]["transition"]["to_state"] = "after"
    else:
        frames[6]["transition"] = frames[5]["transition"]
    with pytest.raises(ValueError):
        diagnosis.episode_summary(config, frames, trajectory, report)


@pytest.mark.parametrize(
    "schedule", [list(diagnosis.NOISE_SEEDS[:1]), list(reversed(diagnosis.NOISE_SEEDS))]
)
def test_caller_cannot_redefine_complete_schedule(schedule):
    with pytest.raises(ValueError, match="complete ordered"):
        diagnosis.paired_summary(pairs(), schedule)


def test_episode_rejects_consistent_partial_trace_without_terminal_fall():
    config, frames, trajectory, report = episode()
    frames = frames[:10]
    trajectory = {
        name: values[: 11 if name in {"qpos", "qvel"} else 10]
        for name, values in trajectory.items()
    }
    report["steps"] = 10
    report["objective_evaluation"] = evaluate_episode(
        spec=config.task, frames=frames, runtime=config.runtime
    )
    report["training_reward_sum_not_success_metric"] = sum(
        row["reward"]["total_reward"] for row in frames
    )
    with pytest.raises(ValueError, match="incomplete episode"):
        diagnosis.episode_summary(config, frames, trajectory, report)


@pytest.mark.parametrize("duration", [float("nan"), 0.0, 20.02, True])
def test_pairing_rejects_invalid_observed_duration(duration):
    rows = pairs()
    rows[0]["duration_seconds"] = duration
    with pytest.raises(ValueError, match="invalid episode outcome"):
        diagnosis.paired_summary(rows, list(diagnosis.NOISE_SEEDS))
