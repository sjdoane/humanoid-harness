"""Publish the canonical, no-training Phase B input artifact set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes, sha256_file
from oracle_composition.experiments.artifact_io import (
    PublishedArtifact,
    publish_bytes_without_overwrite,
)
from oracle_composition.experiments.external_tqc_initialization_identity import (
    EXPERT_ACTOR_NPZ_SHA256,
)
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.harness.contract import oracle_program_from_dict

from .contracts import (
    CLAIM_CEILING,
    EVIDENCE_CLASS,
    FineTuningRunManifest,
    PhaseBOracleProgram,
    StartingCheckpointContract,
    TrackingOnlyRewardSpec,
)

PHASE_A_ORACLE_PATH = "experiments/003_composition_speed_profile/cycles/cycle_1/oracle_1.json"
EXPERT_ACTOR_PATH = "artifacts/bootstrap_tqc_humanoid/farama_minari_humanoid_v5_tqc_actor_v1.npz"
STEP_ZERO_EXPORT_PATH = "artifacts/experiments_003/phase_b/step_0_full_authority_actor_v1.npz"
E1_RECEIPT_PATH = (
    "experiments/003_composition_speed_profile/phase_b/receipts/"
    "e1_full_authority_warm_start_v1.json"
)
TASK_PATH = "experiments/003_composition_speed_profile/task_spec_v1.json"
LIBRARY_PATH = "experiments/003_composition_speed_profile/library_manifest_v1.json"
CORPUS_PATH = "artifacts/reference_corpus_v2/corpus_manifest_v2.json"


def _binding(root: Path, path: str) -> dict[str, object]:
    candidate = root / path
    if candidate.is_symlink() or not candidate.is_file():
        raise ExperimentContractError(f"bound Phase B artifact is unavailable: {path}")
    return {
        "byte_count": candidate.stat().st_size,
        "path": path,
        "sha256": sha256_file(candidate),
    }


def _publish(path: Path, value: object) -> PublishedArtifact:
    return publish_bytes_without_overwrite(path, canonical_json_bytes(value))


def _artifact_binding(published: PublishedArtifact, root: Path) -> dict[str, object]:
    return {
        "byte_count": published.byte_count,
        "path": published.path.relative_to(root).as_posix(),
        "sha256": published.sha256,
    }


def _training_design() -> dict[str, object]:
    return {
        "actor_unfreeze": {
            "first_rollouts": 8,
            "initial_trainable": ["reference_columns", "value_network"],
            "then_trainable": "full_actor_and_value_network",
        },
        "checkpoint_selection": "final_transition_only",
        "cohort_seeds": [121001, 121101, 121201, 121301, 121401],
        "environments": {
            "count": 4,
            "implementation": "DummyVecEnv",
            "stream_mix": {
                "composition": 2,
                "rehearsal": 2,
            },
        },
        "evidence_class": "exploratory_fine_tuning_cycle",
        "normalization": {"observation": False, "reward": False},
        "ppo": {
            "batch_size": 512,
            "clip_range": 0.2,
            "clip_range_vf": None,
            "ent_coef": 0.0,
            "gae_lambda": 0.95,
            "gamma": 0.99,
            "learning_rate": 0.0003,
            "max_grad_norm": 0.5,
            "n_epochs": 10,
            "normalize_advantage": True,
            "target_kl": None,
            "vf_coef": 0.5,
        },
        "retries_or_seed_replacement": False,
        "rollout": {
            "rollout_count": 128,
            "steps_per_environment": 2048,
            "transitions_per_rollout": 8192,
        },
        "rsi": {
            "balanced_origin_actor_cells": 27,
            "schedule_classes": ["hold", "one_way", "round_trip"],
            "start_boundary_hash_modulus": 489,
        },
        "schema_version": 1,
        "training_blocks": [
            120001,
            120002,
            120003,
            120005,
            120007,
            120008,
            120009,
            120011,
            120012,
        ],
        "training_design_schema_id": "humanoid_fine_tuning_training_design/v1",
        "transitions_per_seed": 1048576,
    }


def _evaluation_design() -> dict[str, object]:
    return {
        "cell_episode_counts": {
            "fixed_round_trip": 20,
            "hold_expert": 20,
            "hold_medium": 20,
            "hold_simple": 20,
        },
        "cell_pass_minimum": 16,
        "claim_ceiling": "bounded utility only",
        "error_threshold": 1.0,
        "evaluation_blocks": list(range(120101, 120121)),
        "failure_denominator": "all_predeclared_episodes",
        "family_checkpoint_minimum": 4,
        "family_checkpoint_total": 5,
        "reference_resynchronization_steps": 64,
        "schema_version": 1,
        "step_zero_comparator_required": True,
        "utility_evaluation_schema_id": "humanoid_fine_tuning_utility_evaluation/v1",
    }


def publish_contract_artifacts(
    *,
    repository_root: Path,
    output_directory: Path,
) -> dict[str, PublishedArtifact]:
    """Publish one dependency-ordered canonical contract set without overwrite."""

    root = Path(repository_root).resolve(strict=True)
    output = Path(output_directory)
    raw_oracle = json.loads((root / PHASE_A_ORACLE_PATH).read_text(encoding="utf-8"))
    phase_a = oracle_program_from_dict(
        raw_oracle,
        available_behaviors=("expert", "medium", "simple"),
    )
    oracle_value = PhaseBOracleProgram(phase_a)
    oracle = _publish(output / "oracle_cycle_1_reference_v1.json", oracle_value.to_dict())
    reward_value = TrackingOnlyRewardSpec()
    reward = _publish(output / "tracking_only_v1.json", reward_value.to_dict())
    training = _publish(output / "training_design_v1.json", _training_design())
    evaluator = _publish(output / "utility_evaluation_design_v1.json", _evaluation_design())
    expert_binding = _binding(root, EXPERT_ACTOR_PATH)
    if expert_binding["sha256"] != EXPERT_ACTOR_NPZ_SHA256:
        raise ExperimentContractError("expert NPZ hash mismatch")
    starting_value = {
        "actor_input_layout": ("state_float32_348_then_reference_row_major_float32_8x45/v1"),
        "e1_receipt": _binding(root, E1_RECEIPT_PATH),
        "evidence_class": EVIDENCE_CLASS,
        "normalizer": None,
        "optimizer_constructed_after_copy": True,
        "ortho_init": False,
        "schema_version": 1,
        "source_expert": expert_binding,
        "starting_checkpoint_schema_id": "humanoid_fine_tuning_starting_checkpoint/v1",
        "strict_actor_export": _binding(root, STEP_ZERO_EXPORT_PATH),
        "value_architecture": "independent_708_256_256_1_relu/v1",
        "value_initialization_seed": 20260905,
    }
    StartingCheckpointContract.from_dict(starting_value)
    starting = _publish(output / "starting_checkpoint_v1.json", starting_value)
    manifest_value = {
        "claim_ceiling": CLAIM_CEILING,
        "evaluator": _artifact_binding(evaluator, root),
        "evidence_class": EVIDENCE_CLASS,
        "library": _binding(root, LIBRARY_PATH),
        "oracle": _artifact_binding(oracle, root),
        "ppo_seed": 121901,
        "reference_corpus": _binding(root, CORPUS_PATH),
        "reward": _artifact_binding(reward, root),
        "run_manifest_schema_id": "humanoid_fine_tuning_run_manifest/v1",
        "schema_version": 1,
        "starting_checkpoint": _artifact_binding(starting, root),
        "task": _binding(root, TASK_PATH),
        "training_design": _artifact_binding(training, root),
    }
    FineTuningRunManifest.from_dict(manifest_value)
    manifest = _publish(output / "run_manifest_interface_check_v1.json", manifest_value)
    return {
        "evaluator": evaluator,
        "manifest": manifest,
        "oracle": oracle,
        "reward": reward,
        "starting_checkpoint": starting,
        "training_design": training,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    try:
        published = publish_contract_artifacts(
            repository_root=args.repository_root,
            output_directory=args.output_directory,
        )
    except (ExperimentContractError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = {
        name: {
            "byte_count": artifact.byte_count,
            "path": str(artifact.path),
            "sha256": artifact.sha256,
        }
        for name, artifact in published.items()
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


__all__ = ["publish_contract_artifacts"]


if __name__ == "__main__":
    raise SystemExit(main())
