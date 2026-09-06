"""Candidate admission and final-ready T2 re-sealing.

Action-bound failure is interface-reachable; no production T2 trace producer
has run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.reward_search.publication import finite_pretty_json
from oracle_composition.reward_search.t2_model_contracts import T2PreDispatchSeal

from .execution_manifest import final_ready_t2_execution_manifest_contract_value
from .study_manifest import (
    CANDIDATE_REWARD_ID,
    STUDY_FINAL_READY_STATUS,
    STUDY_STATUS,
    load_t2_study_manifest,
    resolve_t2_reward_binding,
    study_pairing_sha256_from_arm,
    validate_t2_study_manifest,
)


@dataclass(frozen=True, slots=True)
class T2FinalReadyArtifacts:
    """Exact bytes produced together by one successful candidate admission."""

    execution_manifest_bytes: bytes
    study_manifest_bytes: bytes
    t2_seal_bytes: bytes
    admission_commit: str


def _binding(path: str, encoded: bytes) -> dict[str, object]:
    return {
        "byte_count": len(encoded),
        "path": path,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _candidate_relative_path(candidate_root: Path, candidate_path: Path) -> str:
    root = Path(candidate_root).resolve(strict=True)
    raw = Path(candidate_path)
    try:
        before = raw.lstat()
    except OSError as exc:
        raise ExperimentContractError("T2 candidate reward is unavailable") from exc
    try:
        resolved = raw.resolve(strict=True)
        relative = resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ExperimentContractError("T2 candidate reward lies outside its artifact root") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or resolved != root / relative
    ):
        raise ExperimentContractError("T2 candidate reward must be a regular non-linked artifact")
    return relative.as_posix()


def _final_seal_value(
    *,
    study: dict[str, object],
    study_manifest_path: str,
    study_manifest_bytes: bytes,
    execution_manifest_bytes: bytes,
    repository_root: Path,
) -> dict[str, object]:
    baseline = study["arms"][0]
    common = baseline
    baseline_reward = baseline["reward"]
    baseline_path = Path(repository_root) / baseline_reward["path"]
    value = {
        "dispatch_state": "candidate_admitted_no_further_initial_dispatch",
        "evaluator_design": common["evaluator"],
        "execution_manifest": _binding(
            common["execution_manifest"]["path"], execution_manifest_bytes
        ),
        "expert_hold_oracle": common["oracle"],
        "kind": "t2_pre_dispatch_seal",
        "pairing_receipt": study["integrated_pairing_receipt_sha256"],
        "schema_version": 1,
        "seal_id": "f3_t2_pre_dispatch_seal/v1",
        "study_id": study["study_id"],
        "study_manifest": _binding(study_manifest_path, study_manifest_bytes),
        "study_pairing_sha256": study["study_pairing_sha256"],
        "tracking_only_baseline": {
            "byte_count": baseline_path.stat().st_size,
            "path": baseline_reward["path"],
            "sha256": baseline_reward["sha256"],
        },
        "training_design": common["training_design"],
    }
    T2PreDispatchSeal.model_validate(value)
    return value


def admit_t2_candidate_and_reseal(
    *,
    repository_root: Path,
    pending_study_manifest_path: Path,
    candidate_artifact_root: Path,
    candidate_reward_path: Path,
    expected_admission_commit: str,
    allow_final_ready_reseal: bool = False,
) -> T2FinalReadyArtifacts:
    """Resolve a candidate, verify clean HEAD, and regenerate every final seal."""

    root = Path(repository_root).resolve(strict=True)
    pending_path = Path(pending_study_manifest_path).resolve(strict=True)
    try:
        study_relative_path = pending_path.relative_to(root).as_posix()
    except ValueError as exc:
        raise ExperimentContractError("T2 study manifest lies outside the checkout") from exc
    pending, _pending_sha256 = load_t2_study_manifest(
        pending_path,
        repository_root=root,
        candidate_artifact_root=candidate_artifact_root,
        validate_execution_runtime=not allow_final_ready_reseal,
    )
    allowed_statuses = (
        {STUDY_STATUS, STUDY_FINAL_READY_STATUS} if allow_final_ready_reseal else {STUDY_STATUS}
    )
    if pending["status"] not in allowed_statuses:
        raise ExperimentContractError("T2 candidate admission requires the pending study state")

    candidate_relative_path = _candidate_relative_path(
        Path(candidate_artifact_root), Path(candidate_reward_path)
    )
    candidate_bytes = Path(candidate_reward_path).read_bytes()
    candidate_binding = {
        "path": candidate_relative_path,
        "reward_id": CANDIDATE_REWARD_ID,
        "sha256": hashlib.sha256(candidate_bytes).hexdigest(),
    }
    resolve_t2_reward_binding(
        candidate_binding,
        artifact_root=candidate_artifact_root,
        candidate=True,
    )
    if (
        pending["status"] == STUDY_FINAL_READY_STATUS
        and pending["arms"][1]["reward"] != candidate_binding
    ):
        raise ExperimentContractError("T2 re-seal candidate differs from the admitted candidate")

    execution = final_ready_t2_execution_manifest_contract_value(
        root,
        expected_admission_commit=expected_admission_commit,
    )
    execution_bytes = canonical_json_bytes(execution)
    final_study = copy.deepcopy(pending)
    execution_path = final_study["arms"][0]["execution_manifest"]["path"]
    execution_binding = _binding(execution_path, execution_bytes)
    for arm in final_study["arms"]:
        arm["execution_manifest"] = execution_binding
    final_study["arms"][1]["reward"] = candidate_binding
    final_study["status"] = STUDY_FINAL_READY_STATUS
    final_study["study_pairing_sha256"] = study_pairing_sha256_from_arm(final_study["arms"][0])
    validated_study = validate_t2_study_manifest(final_study)
    study_bytes = canonical_json_bytes(validated_study)
    seal_value = _final_seal_value(
        study=validated_study,
        study_manifest_path=study_relative_path,
        study_manifest_bytes=study_bytes,
        execution_manifest_bytes=execution_bytes,
        repository_root=root,
    )
    seal_bytes = finite_pretty_json(seal_value)
    T2PreDispatchSeal.model_validate_json(seal_bytes)
    return T2FinalReadyArtifacts(
        execution_manifest_bytes=execution_bytes,
        study_manifest_bytes=study_bytes,
        t2_seal_bytes=seal_bytes,
        admission_commit=expected_admission_commit,
    )


def _replace_seal_set(
    replacements: Sequence[tuple[Path, bytes]],
) -> None:
    staged: list[tuple[Path, Path]] = []
    try:
        for target, encoded in replacements:
            before = target.lstat()
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                raise ExperimentContractError("T2 re-seal target is not a regular file")
            descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
            temporary_path = Path(temporary)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fchmod(handle.fileno(), stat.S_IMODE(before.st_mode))
                os.fsync(handle.fileno())
            staged.append((target, temporary_path))
        for target, temporary in staged:
            os.replace(temporary, target)
    finally:
        for _target, temporary in staged:
            with suppress(FileNotFoundError):
                temporary.unlink()


def admit_t2_study(
    *,
    repository_root: Path,
    experiment: Path,
    expected_commit: str,
) -> T2FinalReadyArtifacts:
    """Re-seal the retained T2 candidate once from an expected clean commit."""

    root = Path(repository_root).resolve(strict=True)
    experiment_path = Path(experiment)
    if not experiment_path.is_absolute():
        experiment_path = root / experiment_path
    experiment_path = experiment_path.resolve(strict=True)
    expected_experiment = (root / "experiments/004_t2_reward_study").resolve(strict=True)
    if experiment_path != expected_experiment:
        raise ExperimentContractError("T2 re-seal requires the canonical experiment directory")
    artifacts = admit_t2_candidate_and_reseal(
        repository_root=root,
        pending_study_manifest_path=experiment_path / "t2_reward_study_expert_hold_v1.json",
        candidate_artifact_root=root,
        candidate_reward_path=experiment_path / "candidate_target_speed_t2_v1.json",
        expected_admission_commit=expected_commit,
        allow_final_ready_reseal=True,
    )
    _replace_seal_set(
        (
            (
                experiment_path / "execution_manifest_t2_v1.json",
                artifacts.execution_manifest_bytes,
            ),
            (
                experiment_path / "t2_reward_study_expert_hold_v1.json",
                artifacts.study_manifest_bytes,
            ),
            (experiment_path / "t2_seal_v1.json", artifacts.t2_seal_bytes),
        )
    )
    return artifacts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="admit_t2_study")
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    artifacts = admit_t2_study(
        repository_root=args.repository_root,
        experiment=args.experiment,
        expected_commit=args.expected_commit,
    )
    print(
        json.dumps(
            {
                "admission_commit": artifacts.admission_commit,
                "execution_manifest_sha256": hashlib.sha256(
                    artifacts.execution_manifest_bytes
                ).hexdigest(),
                "study_manifest_sha256": hashlib.sha256(artifacts.study_manifest_bytes).hexdigest(),
                "t2_seal_sha256": hashlib.sha256(artifacts.t2_seal_bytes).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "T2FinalReadyArtifacts",
    "admit_t2_candidate_and_reseal",
    "admit_t2_study",
    "main",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
