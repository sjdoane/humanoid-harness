"""Verify the frozen reward-only contrast and its pre-data predictions."""

import argparse
import hashlib
import json
from pathlib import Path

from oracle_composition.adapters.gmt.course_config import load_run_config
from oracle_composition.adapters.gmt.course_runtime import FINITE_HORIZON_RUNTIME
from oracle_composition.adapters.gmt.course_task import CourseStepMetrics, reward
from oracle_composition.adapters.gmt.io import write_json_receipt
from oracle_composition.adapters.gmt.study_scoring import (
    CourseStudyExpectation,
    RunManifestBinding,
    score_course_study_pair,
)

BASE_COMMIT = "9bca23bf62ea386cd7f0492aa63626b6c2a219dc"
BASE_MANIFEST = "fa4fe2cf601eae318cce06b3704ae2206b92b1e99011e95d9dbd3a687a8ad502"
BASE_RESOURCE = "75e67574fb797a576c49886192bc0f88753d44142e873dbb500d3ad96bb06f89"
BASE_CONFIG = "0be730e48fc53c9e34e671138a49cb6cfd3f1a41b350cf2047d55c347a6600bf"
CANDIDATE_CONFIG = "39dd6a045fe680676389a8b46bdfe146bd231ccdb6fc9811499d56a33ab2cb13"
SOURCE_TREE = "3ba7fe0ab72374e0667402848460c43915024fbcd2fe44f9ce5de791f0ab8b8e"
BASE_STATE = "3b81bab42253cfe540f76aff06267eca14427d2ed8c3da25136c4e99eabc9a1c"


def criteria(objective):
    region = objective["region"]
    return {
        "depth_improves_002m": region["minimum_root_height_m"] is not None
        and region["minimum_root_height_m"] <= 0.504942203119977 - 0.02,
        "compliance_no_regression": region["posture_compliant_fraction"] is not None
        and region["posture_compliant_fraction"] >= 31 / 60,
        "full_horizon_without_fall": objective["duration_seconds"] == 20
        and objective["fall_count"] == 0,
        "both_switches": objective["oracle_diagnostics"]["observed_switch_count"] == 2,
        "entry_exit": region["entry_observed"] and region["exit_observed_after_entry"],
        "minimum_25_inside_samples": region["inside_sample_count"] >= 25,
        "inside_speed_regression_limit": objective["speed"][
            "inside_mean_speed_target_deviation_m_s"
        ]
        is not None
        and objective["speed"]["inside_mean_speed_target_deviation_m_s"]
        <= 0.18357610804822067 + 0.05,
        "joint_tracking": objective["development_gate_results"]["joint_position_rmse_p95"],
        "roll_pitch_tracking": objective["development_gate_results"]["roll_pitch_rmse_p95"],
    }


def rows(path):
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def verify_zero_evaluation(first, second, first_recipe, second_recipe):
    assert first["reset"]["reward_sha256"] == first_recipe
    assert second["reset"]["reward_sha256"] == second_recipe
    assert {k: v for k, v in first["reset"].items() if k != "reward_sha256"} == {
        k: v for k, v in second["reset"].items() if k != "reward_sha256"
    }
    allowed = {"reset", "training_reward_sum_not_success_metric"}
    assert {k: v for k, v in first.items() if k not in allowed} == {
        k: v for k, v in second.items() if k not in allowed
    }


def verify_rewards(path, cfg):
    active = 0
    for row in rows(path):
        metrics = CourseStepMetrics(**row["metrics"])
        observed = row["reward"]
        assert reward(spec=cfg.task, recipe=cfg.recipe, metrics=metrics).to_dict() == observed
        if metrics.inside_posture_region and not metrics.fallen:
            ceiling = cfg.task.posture_band_low_m + cfg.recipe.ceiling_fraction * (
                cfg.task.posture_band_high_m - cfg.task.posture_band_low_m
            )
            active += int(
                cfg.recipe.recipe_version == 2
                and cfg.recipe.depth_strength > 0
                and metrics.root_height_m > ceiling
            )
    return active


def main():
    if not __debug__:
        raise RuntimeError("Study014 verifier requires Python assertions enabled")
    parser = argparse.ArgumentParser()
    for field in ("baseline", "candidate", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    for field in ("source-commit", "manifest-sha256", "resource-sha256"):
        parser.add_argument(f"--{field}", required=True)
    args = parser.parse_args()
    old = load_run_config(args.baseline / "input_config.json")
    new = load_run_config(args.candidate / "input_config.json")
    assert old.sha256 == BASE_CONFIG and new.sha256 == CANDIDATE_CONFIG
    assert {k for k in old.raw.keys() | new.raw.keys() if old.raw.get(k) != new.raw.get(k)} == {
        "reward"
    }
    assert old.runtime == new.runtime == FINITE_HORIZON_RUNTIME
    assert old.trainer == new.trainer and new.trainer.profile_version == 3
    for name in ("zero_residual_trajectory.npz", "initial_residual_policy.npz"):
        assert (args.baseline / name).read_bytes() == (args.candidate / name).read_bytes(), name
    for a, b in zip(
        rows(args.baseline / "zero_residual_frames.jsonl"),
        rows(args.candidate / "zero_residual_frames.jsonl"),
        strict=True,
    ):
        assert {k: v for k, v in a.items() if k != "reward"} == {
            k: v for k, v in b.items() if k != "reward"
        }
        allowed = {
            "task_reward_recipe_sha256",
            "posture_component_reward",
            "task_reward",
            "total_reward",
        }
        assert {k: v for k, v in a["reward"].items() if k not in allowed} == {
            k: v for k, v in b["reward"].items() if k not in allowed
        }
    a = json.loads((args.baseline / "zero_residual_evaluation.json").read_bytes())
    b = json.loads((args.candidate / "zero_residual_evaluation.json").read_bytes())
    verify_zero_evaluation(a, b, old.recipe.sha256, new.recipe.sha256)

    def binding(root, cfg, cell, commit, manifest, resource):
        identities = {
            "task": cfg.task.sha256,
            "oracle": cfg.program.sha256,
            "reward": cfg.recipe.sha256,
            "segments": {k: v.sha256 for k, v in cfg.segments.items()},
        }
        return (
            RunManifestBinding(root / "course_run_manifest.json", manifest, resource),
            CourseStudyExpectation(
                cell,
                20260906,
                131072,
                identities,
                cfg.trainer,
                BASE_STATE,
                commit,
                runtime=cfg.runtime,
            ),
        )

    first, fe = binding(args.baseline, old, "r1", BASE_COMMIT, BASE_MANIFEST, BASE_RESOURCE)
    second, se = binding(
        args.candidate, new, "r4", args.source_commit, args.manifest_sha256, args.resource_sha256
    )
    pair = score_course_study_pair(
        first=first, first_expected=fe, second=second, second_expected=se
    )
    assert pair["pair"]["semantic_identity_differences"] == ["reward"]
    assert not pair["pair"]["course_runtime_profile_difference"]
    assert all(
        cell["resource"]["source_tree_sha256"] == SOURCE_TREE for cell in pair["cells"].values()
    )
    active_counts = {}
    for name, path, cfg in (("r1", args.baseline, old), ("r4", args.candidate, new)):
        for label in ("zero_residual", "final_policy"):
            active_counts[f"{name}_{label}"] = verify_rewards(path / f"{label}_frames.jsonl", cfg)
    assert active_counts["r4_zero_residual"] == 58
    objective = pair["cells"]["r4"]["objective"]
    gates = criteria(objective)
    result = {
        "study": "014",
        "pair": pair,
        "criteria": gates,
        "screen_passed": all(gates.values()),
        "active_depth_rows": active_counts,
        "active_depth_rows_scope": "saved_evaluation_traces_not_training_activation_telemetry",
        "source_tree_equal": True,
        "zero_nonreward_parity": True,
        "initial_policy_byte_equal": True,
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "claim_scope": "one_seed_reward_revision_not_heldout_or_general_success",
    }
    digest = write_json_receipt(args.output, result)
    print(
        json.dumps(
            {
                "path": str(args.output),
                "sha256": digest,
                "criteria": gates,
                "screen_passed": result["screen_passed"],
                "active_depth_rows": active_counts,
                "objective": objective,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
