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
from oracle_composition.harness.contract import load_oracle_program

from .contracts import (
    CLAIM_CEILING,
    EVIDENCE_CLASS,
    FineTuningRunManifest,
    PhaseBOracleProgram,
    StartingCheckpointContract,
    TrackingOnlyRewardSpec,
    training_design_contract_value,
    utility_evaluation_design_contract_value,
)

PHASE_A_ORACLE_PATH = "experiments/003_composition_speed_profile/cycles/cycle_1/oracle_1.json"
PHASE_A_ORACLE_SHA256 = "32bfc555ffc578d4cf8f75823acc1ba98db6621b2bcd0204f0725b0bf70af029"
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
    return training_design_contract_value()


def _evaluation_design() -> dict[str, object]:
    return utility_evaluation_design_contract_value()


def publish_contract_artifacts(
    *,
    repository_root: Path,
    output_directory: Path,
) -> dict[str, PublishedArtifact]:
    """Publish one dependency-ordered canonical contract set without overwrite."""

    root = Path(repository_root).resolve(strict=True)
    output = Path(output_directory)
    phase_a, phase_a_sha256 = load_oracle_program(
        root / PHASE_A_ORACLE_PATH,
        available_behaviors=("expert", "medium", "simple"),
    )
    if phase_a_sha256 != PHASE_A_ORACLE_SHA256:
        raise ExperimentContractError("Phase A oracle identity differs")
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
        "execution_profiles": {
            "checkpoint_selection": "final_transition_only",
            "cohort": {
                "evidence_class": "exploratory_fine_tuning_cycle",
                "promotable": True,
                "seeds": [121001, 121101, 121201, 121301, 121401],
                "transitions_per_seed": 1048576,
            },
            "smoke": {
                "evidence_class": "interface_check",
                "promotable": False,
                "seeds": [121901],
                "transitions_per_seed": 196608,
            },
        },
        "reference_corpus": _binding(root, CORPUS_PATH),
        "reward": _artifact_binding(reward, root),
        "run_manifest_schema_id": "humanoid_fine_tuning_run_manifest/v2",
        "schema_version": 2,
        "starting_checkpoint": _artifact_binding(starting, root),
        "task": _binding(root, TASK_PATH),
        "training_design": _artifact_binding(training, root),
    }
    FineTuningRunManifest.from_dict(manifest_value)
    manifest = _publish(output / "run_manifest_training_admission_v2.json", manifest_value)
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
