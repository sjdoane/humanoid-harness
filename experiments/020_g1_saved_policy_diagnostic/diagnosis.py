"""Data-only episode checks and paired summaries; never selects a checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import subprocess
import zipfile
from pathlib import Path
from statistics import fmean, median

import numpy as np

from oracle_composition.adapters.gmt.course_config import load_run_config
from oracle_composition.adapters.gmt.course_evaluation import evaluate_episode
from oracle_composition.adapters.gmt.course_task import CourseStepMetrics, TaskFrame, reward
from oracle_composition.adapters.gmt.io import validate_zip_members, write_json_receipt
from oracle_composition.feedback.g1_course import (
    _load_frames,
    _load_trajectory,
    _verified_bytes,
    _verify_boundary_metrics,
    _verify_frame_crosslinks,
)

NOISE_SEEDS = tuple(range(20260920, 20260936))


def sampled_action_evidence(encoded: bytes, steps: int) -> dict:
    with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
        members = validate_zip_members(archive, expected_count=3, maximum_member_size=128 * 1024)
        if set(members) != {
            f"{name}.npy" for name in ("policy_mean", "policy_std", "unclipped_action")
        }:
            raise ValueError("sampling evidence members differ")
    with np.load(io.BytesIO(encoded), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if any(
        array.shape != (steps, 23) or array.dtype != np.dtype("<f4") or not np.isfinite(array).all()
        for array in arrays.values()
    ):
        raise ValueError("sampling evidence tensor contract differs")
    if np.any(arrays["policy_std"] <= 0):
        raise ValueError("sampling standard deviations must be positive")
    return arrays


def verify_sampled_actions(
    noise: np.ndarray, evidence: dict, actions: np.ndarray, receipt: dict
) -> float:
    raw = np.add(
        evidence["policy_mean"],
        np.multiply(noise[: len(actions)], evidence["policy_std"], dtype=np.float32),
        dtype=np.float32,
    )
    clipped = np.clip(raw, -1.0, 1.0)
    count = int(np.count_nonzero(raw != clipped))
    if (
        not np.array_equal(raw, evidence["unclipped_action"])
        or not np.array_equal(clipped, actions)
        or receipt["clipped_component_count"] != count
        or receipt["total_component_count"] != actions.size
        or receipt["max_abs_unclipped_action"] != float(np.max(np.abs(raw)))
    ):
        raise ValueError("recorded sampling arithmetic or clipping counters differ")
    return count / actions.size


def episode_summary(config, frames: list[dict], trajectory: dict, report: dict) -> dict:
    """Recheck recorded evidence before deriving handover or return diagnostics."""
    _verify_frame_crosslinks(frames, trajectory)
    _verify_boundary_metrics(spec=config.task, frames=frames, trajectory=trajectory)
    objective = evaluate_episode(spec=config.task, frames=frames, runtime=config.runtime)
    if report["steps"] != len(frames) or report["objective_evaluation"] != objective:
        raise ValueError("episode report differs from the recorded objective")
    total = 0.0
    for row in frames:
        expected = reward(
            spec=config.task,
            recipe=config.recipe,
            metrics=CourseStepMetrics(**row["metrics"]),
        ).to_dict()
        if row["reward"] != expected:
            raise ValueError("recorded reward differs from the frozen recipe")
        total += expected["total_reward"]
    if total != report["training_reward_sum_not_success_metric"]:
        raise ValueError("raw episode reward differs from the report")
    actions = trajectory["residual_action"]
    if (
        actions.shape != (len(frames), 23)
        or not np.isfinite(actions).all()
        or np.any(np.abs(actions) > 1.0)
    ):
        raise ValueError("retained actions differ from the clipped residual contract")
    rms = float(np.sqrt(np.mean(np.asarray(actions, dtype=np.float64) ** 2)))
    if rms != report["residual_rms"]:
        raise ValueError("retained residual RMS differs from the report")

    qpos, qvel = trajectory["qpos"], trajectory["qvel"]
    frame = TaskFrame.initialize(qpos[0, :2], qpos[0, 3:7])
    forward = np.asarray([math.cos(frame.forward_yaw_rad), math.sin(frame.forward_yaw_rad)])
    handovers = []
    if frames[0]["transition"] is not None:
        raise ValueError("initial command cannot claim an observed mode handover")
    for index in range(1, len(frames)):
        previous, current = frames[index - 1], frames[index]
        if previous["executed_mode"] == current["executed_mode"]:
            if current["transition"] is not None:
                raise ValueError("transition marker without an executed mode change")
            continue
        transition = current["transition"]
        expected = {
            "control_step": index,
            "from_state": previous["executed_mode"],
            "to_state": current["executed_mode"],
            "from_behavior": previous["executed_behavior"],
            "to_behavior": current["executed_behavior"],
            "selected_phase_seconds": current["executed_phase_seconds"],
        }
        if type(transition) is not dict or any(transition.get(k) != v for k, v in expected.items()):
            raise ValueError("executed mode change contradicts its transition record")
        state = frame.project(qpos[index, :2], qpos[index, 3:7])
        handovers.append(
            {
                "action_index": index,
                "from_mode": previous["executed_mode"],
                "to_mode": current["executed_mode"],
                "pre_action_height_m": float(qpos[index, 2]),
                "pre_action_progress_m": state.progress_m,
                "pre_action_lateral_m": state.lateral_m,
                "pre_action_heading_signed_rad": state.heading_error_rad,
                "pre_action_course_forward_qvel_m_s": float(qvel[index, :2] @ forward),
                "incoming_interval_course_forward_speed_m_s": previous["metrics"][
                    "forward_speed_m_s"
                ],
                "native_executed_phase_seconds": current["executed_phase_seconds"],
                "observed_seconds_until_episode_end": (len(frames) - index) * 0.02,
                "episode_end_is_fall": bool(objective["fall_count"]),
            }
        )
    terminal = frames[-1]
    full_horizon = len(frames) == config.task.horizon_steps
    if not full_horizon and not terminal["metrics"]["fallen"]:
        raise ValueError("incomplete episode without a recorded terminal fall")
    return {
        "raw_total_reward_sum": total,
        "raw_task_reward_sum": sum(row["reward"]["task_reward"] for row in frames),
        "raw_tracking_reward_sum": sum(row["reward"]["tracking_reward"] for row in frames),
        "fall": bool(objective["fall_count"]),
        "duration_seconds": objective["duration_seconds"],
        "full_horizon": full_horizon,
        "early_termination_censors_later_exposure": not full_horizon,
        "terminal_mode": terminal["executed_mode"],
        "termination_reasons": terminal["metrics"]["failure_reasons"],
        "objective": objective,
        "handovers": handovers,
        "residual_rms": rms,
        "at_action_bound_component_fraction": float(np.mean(np.abs(actions) >= 1.0)),
        "claim_limit": "boundary_metrics_and_rewards_recomputed_substep_failures_remain_recorded",
    }


def paired_summary(episodes: list[dict], noise_seeds: list[int]) -> dict:
    """Summarize fixed checkpoints across paired noise, not training-seed replicates."""
    if tuple(noise_seeds) != NOISE_SEEDS or any(type(seed) is not int for seed in noise_seeds):
        raise ValueError("noise seeds must match the complete ordered Study020 schedule")
    expected = {(seed, policy) for seed in noise_seeds for policy in ("initial", "final")}
    indexed = {}
    for row in episodes:
        key = (row["noise_seed"], row["policy"])
        if key not in expected or key in indexed:
            raise ValueError("duplicate or unexpected paired episode")
        if (
            type(row["noise_seed"]) is not int
            or type(row["fall"]) is not bool
            or type(row["raw_total_reward_sum"]) not in {int, float}
            or not math.isfinite(row["raw_total_reward_sum"])
            or type(row["duration_seconds"]) not in {int, float}
            or not math.isfinite(row["duration_seconds"])
            or not 0 < row["duration_seconds"] <= 20.0
        ):
            raise ValueError("invalid episode outcome")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("missing paired episode; cannot report a complete diagnostic")
    cells = {
        "both_survive": 0,
        "initial_survives_final_falls": 0,
        "initial_falls_final_survives": 0,
        "both_fall": 0,
    }
    pairs = []
    for seed in noise_seeds:
        initial, final = (indexed[(seed, policy)] for policy in ("initial", "final"))
        cell = (
            "both_fall"
            if initial["fall"] and final["fall"]
            else "initial_falls_final_survives"
            if initial["fall"]
            else "initial_survives_final_falls"
            if final["fall"]
            else "both_survive"
        )
        cells[cell] += 1
        pairs.append(
            {
                "noise_seed": seed,
                "initial_raw_return": initial["raw_total_reward_sum"],
                "final_raw_return": final["raw_total_reward_sum"],
                "final_minus_initial_return": final["raw_total_reward_sum"]
                - initial["raw_total_reward_sum"],
                "fall_cell": cell,
                "initial_duration_seconds": initial["duration_seconds"],
                "final_duration_seconds": final["duration_seconds"],
            }
        )
    differences = [row["final_minus_initial_return"] for row in pairs]
    return {
        "pairs": pairs,
        "fall_cells": cells,
        "initial_falls": cells["initial_falls_final_survives"] + cells["both_fall"],
        "final_falls": cells["initial_survives_final_falls"] + cells["both_fall"],
        "paired_return_difference": {
            "count": len(differences),
            "mean": fmean(differences),
            "median": median(differences),
            "minimum": min(differences),
            "maximum": max(differences),
        },
        "uncertainty": "all_pairs_and_range_reported_no_confidence_interval_or_hypothesis_test",
        "claim_scope": "conditional_action_noise_diagnostic_one_training_seed_one_reset",
        "adoption_decision": "none",
    }


def score_run(run: Path, manifest_sha256: str, resource_sha256: str, source_commit: str) -> dict:
    """Bind a successful native run, then independently rebuild every episode summary."""
    from oracle_composition.adapters.gmt.saved_policy_diagnostic_config import (
        load_noise_archive,
        load_saved_policy_diagnostic_config,
        verify_saved_policy_diagnostic_outputs,
    )

    repository = Path(__file__).resolve().parents[2]
    head = subprocess.check_output(
        ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(repository), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    )
    if head != source_commit or dirty:
        raise ValueError("score at the pinned clean run source")
    manifest_name = "saved_policy_diagnostic_manifest.json"
    manifest_bytes = _verified_bytes(run / manifest_name, manifest_sha256, 1024**2)
    manifest = json.loads(manifest_bytes)
    resource = json.loads(
        _verified_bytes(run / "gmt_probe_resource_receipt_v1.json", resource_sha256, 1024**2)
    )
    if resource["status"] != "succeeded" or resource["commit"] != source_commit:
        raise ValueError("diagnostic lacks its successful native source receipt")
    if resource["artifacts"]["saved_policy_diagnostic_manifest"] != {
        "path": manifest_name,
        "sha256": manifest_sha256,
        "size": len(manifest_bytes),
    }:
        raise ValueError("native receipt does not bind the diagnostic manifest")
    ledger = manifest["outputs"]
    if set(ledger) != set(resource["artifacts"]["outputs"]):
        raise ValueError("native output ledger differs from the diagnostic manifest")
    for name, digest in ledger.items():
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("unsafe output filename")
        payload = _verified_bytes(run / name, digest, 128 * 1024**2)
        item = resource["artifacts"]["outputs"][name]
        if item["sha256"] != digest or item["size"] != len(payload):
            raise ValueError("native output identity differs from retained bytes")
    diagnostic_bytes = _verified_bytes(
        run / "input_diagnostic_config.json", ledger["input_diagnostic_config.json"], 256 * 1024
    )
    if resource["inputs"]["config"]["sha256"] != hashlib.sha256(diagnostic_bytes).hexdigest():
        raise ValueError("native request differs from the retained diagnostic config")
    admitted = load_saved_policy_diagnostic_config(run / "input_diagnostic_config.json")
    verify_saved_policy_diagnostic_outputs(admitted, run)
    noise = load_noise_archive(admitted.noise)
    diagnostic = json.loads(diagnostic_bytes)
    course_bytes = _verified_bytes(
        run / "input_course_config.json", ledger["input_course_config.json"], 256 * 1024
    )
    if hashlib.sha256(course_bytes).hexdigest() != diagnostic["retained"]["course_config_sha256"]:
        raise ValueError("course config differs from the pinned retained018 input")
    config = load_run_config(run / "input_course_config.json")
    seeds = diagnostic["sampling"]["noise_seeds"]
    if seeds != list(NOISE_SEEDS):
        raise ValueError("noise schedule differs from the Study020 declaration")
    episodes = []
    sampled_receipts = {row["label"]: row["sampling"] for row in manifest["episodes"]}
    for seed in seeds:
        for policy in ("initial", "final"):
            label = f"sample_{seed}_{policy}"
            frames = _load_frames(
                _verified_bytes(
                    run / f"{label}_frames.jsonl", ledger[f"{label}_frames.jsonl"], 128 * 1024**2
                )
            )
            trajectory = _load_trajectory(
                _verified_bytes(
                    run / f"{label}_trajectory.npz", ledger[f"{label}_trajectory.npz"], 32 * 1024**2
                ),
                len(frames),
            )
            report = json.loads(
                _verified_bytes(
                    run / f"{label}_evaluation.json", ledger[f"{label}_evaluation.json"], 256 * 1024
                )
            )
            evidence = sampled_action_evidence(
                _verified_bytes(
                    run / f"{label}_sampling.npz",
                    ledger[f"{label}_sampling.npz"],
                    512 * 1024,
                ),
                len(frames),
            )
            clipping_fraction = verify_sampled_actions(
                noise[seeds.index(seed)],
                evidence,
                trajectory["residual_action"],
                sampled_receipts[label],
            )
            episodes.append(
                {
                    "noise_seed": seed,
                    "policy": policy,
                    "sampling": sampled_receipts[label],
                    "clipped_component_fraction": clipping_fraction,
                    "sampled_action_arithmetic_verified": True,
                    **episode_summary(config, frames, trajectory, report),
                }
            )
    return {
        "artifact": "gmt_g1_study020_saved_policy_diagnosis/v1",
        "source_commit": source_commit,
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_manifest_sha256": manifest_sha256,
        "resource_receipt_sha256": resource_sha256,
        "episodes": episodes,
        "paired": paired_summary(episodes, seeds),
        "training_performed": False,
        "all_development_gates_passed_count": sum(
            e["objective"]["development_gate_passed"] for e in episodes
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--resource-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = score_run(args.run, args.manifest_sha256, args.resource_sha256, args.source_commit)
    digest = write_json_receipt(args.output, result)
    print(json.dumps({"output": str(args.output), "sha256": digest, "paired": result["paired"]}))


if __name__ == "__main__":
    main()
