"""CLI adapter for validated feedback diagnosis and candidate-context publication."""

from __future__ import annotations

from pathlib import Path

from oracle_composition.experiments.artifact_io import (
    PublishedArtifact,
    finite_pretty_json,
    publish_bytes_without_overwrite,
)

from .context import build_candidate_context
from .development_diagnosis import diagnose_development_feedback
from .evidence import diagnose_feedback


def _artifact_value(artifact: PublishedArtifact) -> dict[str, object]:
    return {
        "byte_count": artifact.byte_count,
        "path": str(artifact.path),
        "sha256": artifact.sha256,
    }


def run_diagnose_command(
    *,
    repository_root: Path,
    experiment: Path,
    phase_a_receipt: Path,
    phase_a_cycle: int,
    phase_b_evidence: Path | None,
    steering_file: Path | None,
    database: Path | None,
    output: Path,
) -> dict[str, object]:
    """Run the data-only feedback slice and publish immutable coordinator inputs."""

    diagnosis = diagnose_feedback(
        phase_a_receipt_path=phase_a_receipt,
        experiment=experiment,
        repository_root=repository_root,
        phase_a_cycle=phase_a_cycle,
        phase_b_evidence_path=phase_b_evidence,
        steering_path=steering_file,
    )
    context = build_candidate_context(diagnosis, database=database)
    destination = Path(output)
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    diagnosis_artifact = publish_bytes_without_overwrite(
        destination / "diagnosis_v1.json", finite_pretty_json(diagnosis.to_dict())
    )
    context_artifact = publish_bytes_without_overwrite(
        destination / "candidate_context_v1.json", finite_pretty_json(context.to_dict())
    )
    prompt_artifact = publish_bytes_without_overwrite(
        destination / "candidate_prompt_v1.md", context.prompt
    )
    return {
        "action_surface": context.action_surface,
        "artifacts": {
            "candidate_context": _artifact_value(context_artifact),
            "candidate_prompt": _artifact_value(prompt_artifact),
            "diagnosis": _artifact_value(diagnosis_artifact),
        },
        "candidate_kind": context.candidate_kind,
        "claim_ceilings": list(diagnosis.claim_ceilings),
        "human_diagnostic_surface": diagnosis.action_surface,
        "kind": "humanoid_feedback_run_result",
        "limitations": [
            "no_model_call",
            "no_candidate_ingestion",
            "no_replay_screen",
            "no_training_or_evaluation_authorization",
            "protected_and_held_out_values_excluded_from_candidate_prompt",
        ],
        "readiness": diagnosis.to_dict()["readiness"],
        "schema_version": 1,
    }


def run_diagnose_development_command(
    *,
    development_result: Path,
    smoke_run: Path,
    corpus_root: Path,
    protocol: Path,
    steering_file: Path | None,
    database: Path | None,
    output: Path,
) -> dict[str, object]:
    """Publish candidate-safe facts from one strict development ablation."""

    diagnosis = diagnose_development_feedback(
        development_result_path=development_result,
        smoke_run=smoke_run,
        corpus_root=corpus_root,
        protocol_path=protocol,
        steering_path=steering_file,
    )
    context = build_candidate_context(diagnosis, database=database)
    destination = Path(output)
    destination.mkdir(mode=0o700, parents=True, exist_ok=False)
    diagnosis_artifact = publish_bytes_without_overwrite(
        destination / "diagnosis_v1.json", finite_pretty_json(diagnosis.to_dict())
    )
    context_artifact = publish_bytes_without_overwrite(
        destination / "candidate_context_v1.json", finite_pretty_json(context.to_dict())
    )
    prompt_artifact = publish_bytes_without_overwrite(
        destination / "candidate_prompt_v1.md", context.prompt
    )
    return {
        "action_surface": context.action_surface,
        "artifacts": {
            "candidate_context": _artifact_value(context_artifact),
            "candidate_prompt": _artifact_value(prompt_artifact),
            "diagnosis": _artifact_value(diagnosis_artifact),
        },
        "candidate_kind": context.candidate_kind,
        "claim_ceilings": list(diagnosis.claim_ceilings),
        "human_diagnostic_surface": diagnosis.action_surface,
        "kind": "humanoid_development_feedback_run_result",
        "limitations": [
            "in_sample_development_evidence_only",
            "no_model_call_or_candidate_ingestion",
            "no_training_or_evaluation_authorization",
            "no_task_success_or_causal_claim",
            "protected_and_held_out_values_excluded_from_candidate_prompt",
        ],
        "readiness": diagnosis.to_dict()["readiness"],
        "schema_version": 1,
    }


__all__ = ["run_diagnose_command", "run_diagnose_development_command"]
