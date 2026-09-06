"""Fail-closed matched-arm manifest for the expert-hold T2 reward study."""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Mapping
from pathlib import Path

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.phase_b.contracts import (
    COHORT_SEEDS,
    COHORT_TRANSITIONS,
    FINAL_CHECKPOINT_RULE,
    RewardRegistry,
    TargetSpeedRewardSpec,
    TrackingOnlyRewardSpec,
    load_phase_b_oracle,
    load_starting_checkpoint,
    validate_training_design,
)

from .execution_manifest import load_t2_execution_manifest
from .pairing import PAIRING_ADAPTER_ID, PAIRING_DERIVATION_ID
from .t2_evaluator import (
    T2_EVALUATION_SEEDS,
    T2_EVALUATOR_ID,
    T2_HORIZON_STEPS,
    T2_REPORT_SCHEMA_ID,
    T2_TARGET_SPEED_M_S,
    load_evaluator_design,
)

STUDY_ID = "t2_reward_study_expert_hold/v1"
STUDY_MANIFEST_SCHEMA_ID = "t2_reward_study_manifest/v1"
STUDY_STATUS = "execution_no_go_until_candidate_and_integrated_pairing_receipt_exist"
STUDY_EVIDENCE_CLASS = "exploratory_fine_tuning_cycle"
STUDY_CLAIM_CEILING = (
    "candidate_met_or_did_not_meet_the_preregistered_t2_criterion_in_this_matched_"
    "five_seed_implementation_no_generalization_naturalness_causal_reference_use_or_"
    "humanoid_competence_claim"
)
BASELINE_REWARD_ID = "tracking_only/v1"
BASELINE_REWARD_SHA256 = "eea2b6a9893e6e4ca5aea5d6787580f12062084e758cc9c27db2f1376bcb1c5f"
CANDIDATE_REWARD_ID = "target_speed_triangular_affine_t2_adapter/v1"
PAIRING_EXCLUDED_ARM_FIELDS = frozenset({"arm_label", "output_path", "reward", "timestamps"})
_COMMON_ARM_FIELDS = {
    "calibration",
    "claim_ceiling",
    "evaluation",
    "evaluator",
    "evidence_class",
    "execution_manifest",
    "library",
    "oracle",
    "pairing",
    "pairing_adapter",
    "reference_corpus",
    "starting_checkpoint",
    "task",
    "training",
    "training_design",
}
_ARM_FIELDS = _COMMON_ARM_FIELDS | set(PAIRING_EXCLUDED_ARM_FIELDS)


class T2StudyManifestError(ValueError):
    """Raised when matched-arm identity or execution readiness is ambiguous."""


def _sha256(value: object, *, field: str, allow_tbd: bool = False) -> str:
    if allow_tbd and value == "TBD":
        return "TBD"
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise T2StudyManifestError(f"{field} must be a lowercase SHA-256")
    return value


def _artifact_binding(value: object, *, field: str) -> dict[str, object]:
    expected = {"byte_count", "path", "sha256"}
    if type(value) is not dict or set(value) != expected:
        raise T2StudyManifestError(f"{field} artifact binding fields differ")
    if type(value["byte_count"]) is not int or value["byte_count"] <= 0:
        raise T2StudyManifestError(f"{field}.byte_count must be positive")
    if type(value["path"]) is not str or not value["path"] or len(value["path"]) > 512:
        raise T2StudyManifestError(f"{field}.path is invalid")
    _sha256(value["sha256"], field=f"{field}.sha256")
    return dict(value)


def _reward_binding(value: object, *, field: str, candidate: bool) -> dict[str, object]:
    expected = {"path", "reward_id", "sha256"}
    if type(value) is not dict or set(value) != expected:
        raise T2StudyManifestError(f"{field} reward binding fields differ")
    expected_id = CANDIDATE_REWARD_ID if candidate else BASELINE_REWARD_ID
    if value["reward_id"] != expected_id:
        raise T2StudyManifestError(f"{field} reward identity differs")
    if candidate:
        _sha256(value["sha256"], field=f"{field}.sha256", allow_tbd=True)
        if (value["path"] == "TBD") is not (value["sha256"] == "TBD"):
            raise T2StudyManifestError("candidate reward path and digest readiness differ")
        if value["path"] != "TBD" and (
            type(value["path"]) is not str or not value["path"] or len(value["path"]) > 512
        ):
            raise T2StudyManifestError("candidate reward path is invalid")
    else:
        if (
            value["path"]
            != ("experiments/003_composition_speed_profile/phase_b/tracking_only_v1.json")
            or value["sha256"] != BASELINE_REWARD_SHA256
        ):
            raise T2StudyManifestError("baseline reward binding differs")
    return dict(value)


def arm_common_fields(arm: Mapping[str, object]) -> dict[str, object]:
    """Project exactly the fields allowed to determine paired stochastic streams."""

    if type(arm) is not dict or set(arm) != _ARM_FIELDS:
        raise T2StudyManifestError("study arm fields differ")
    return {key: value for key, value in arm.items() if key not in PAIRING_EXCLUDED_ARM_FIELDS}


def study_pairing_sha256_from_arm(arm: Mapping[str, object]) -> str:
    return hashlib.sha256(canonical_json_bytes(arm_common_fields(arm))).hexdigest()


def _validate_common_fields(common: Mapping[str, object]) -> None:
    if common.get("task") != {
        "environment_id": "Humanoid-v5",
        "expert_start": True,
        "horizon_steps": T2_HORIZON_STEPS,
        "target_com_forward_speed_m_s": T2_TARGET_SPEED_M_S,
    }:
        raise T2StudyManifestError("study task differs from frozen T2")
    if common.get("training") != {
        "checkpoint_selection": FINAL_CHECKPOINT_RULE,
        "environment_count": 4,
        "ppo_seeds": list(COHORT_SEEDS),
        "retries_or_seed_replacement": False,
        "transitions_per_seed": COHORT_TRANSITIONS,
    }:
        raise T2StudyManifestError("study training budget or seeds differ")
    if common.get("evaluation") != {
        "deterministic_actions": True,
        "evaluation_seeds": list(T2_EVALUATION_SEEDS),
        "evaluator_id": T2_EVALUATOR_ID,
        "expert_start": True,
        "horizon_steps": T2_HORIZON_STEPS,
        "report_schema_id": T2_REPORT_SCHEMA_ID,
    }:
        raise T2StudyManifestError("study evaluation contract differs")
    if common.get("pairing") != {
        "adapter_id": PAIRING_ADAPTER_ID,
        "declared": True,
        "derivation_id": PAIRING_DERIVATION_ID,
    }:
        raise T2StudyManifestError("study pairing declaration differs")
    if common.get("calibration") != "none":
        raise T2StudyManifestError("T2 primary calibration must be none")
    if common.get("evidence_class") != STUDY_EVIDENCE_CLASS:
        raise T2StudyManifestError("study evidence class differs")
    if common.get("claim_ceiling") != STUDY_CLAIM_CEILING:
        raise T2StudyManifestError("study claim ceiling differs")
    for field in (
        "evaluator",
        "execution_manifest",
        "library",
        "oracle",
        "pairing_adapter",
        "reference_corpus",
        "starting_checkpoint",
        "training_design",
    ):
        _artifact_binding(common.get(field), field=field)


def validate_t2_study_arm_pair(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    """Refuse any between-arm drift outside declared arm-specific fields."""

    baseline_common = arm_common_fields(baseline)
    candidate_common = arm_common_fields(candidate)
    if baseline_common != candidate_common:
        raise T2StudyManifestError("study arms differ in a common field")
    _validate_common_fields(baseline_common)
    if baseline.get("arm_label") != "baseline" or candidate.get("arm_label") != "candidate":
        raise T2StudyManifestError("study arm labels differ")
    for arm, field in ((baseline, "baseline"), (candidate, "candidate")):
        if arm.get("output_path") != "TBD" and (
            type(arm.get("output_path")) is not str or not arm["output_path"]
        ):
            raise T2StudyManifestError(f"{field} output path is invalid")
        timestamps = arm.get("timestamps")
        if type(timestamps) is not dict or set(timestamps) != {"completed_utc", "started_utc"}:
            raise T2StudyManifestError(f"{field} timestamps fields differ")
        if any(value != "TBD" for value in timestamps.values()):
            raise T2StudyManifestError(f"{field} timestamps must remain TBD before execution")
    _reward_binding(baseline.get("reward"), field="baseline", candidate=False)
    _reward_binding(candidate.get("reward"), field="candidate", candidate=True)
    return baseline_common


def validate_t2_study_manifest(value: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "arms",
        "integrated_pairing_receipt_sha256",
        "schema_version",
        "status",
        "study_id",
        "study_manifest_schema_id",
        "study_pairing_sha256",
    }
    if type(value) is not dict or set(value) != expected:
        raise T2StudyManifestError("study manifest fields differ")
    if (
        value["schema_version"] != 1
        or value["study_id"] != STUDY_ID
        or value["study_manifest_schema_id"] != STUDY_MANIFEST_SCHEMA_ID
        or value["status"] != STUDY_STATUS
        or value["integrated_pairing_receipt_sha256"] != "TBD"
    ):
        raise T2StudyManifestError("study manifest identity or status differs")
    arms = value["arms"]
    if type(arms) is not list or len(arms) != 2 or any(type(arm) is not dict for arm in arms):
        raise T2StudyManifestError("study manifest requires exactly two arm slots")
    common = validate_t2_study_arm_pair(arms[0], arms[1])
    expected_pairing = hashlib.sha256(canonical_json_bytes(common)).hexdigest()
    _sha256(value["study_pairing_sha256"], field="study_pairing_sha256")
    if value["study_pairing_sha256"] != expected_pairing:
        raise T2StudyManifestError("study pairing SHA-256 differs from common fields")
    canonical_json_bytes(dict(value))
    return dict(value)


def _verify_artifact(root: Path, binding: Mapping[str, object], *, field: str) -> Path:
    relative = Path(str(binding["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise T2StudyManifestError(f"{field} path must be repository-relative")
    path = root / relative
    try:
        before = path.lstat()
    except OSError as exc:
        raise T2StudyManifestError(f"{field} artifact is unavailable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise T2StudyManifestError(f"{field} artifact must be a regular file")
    encoded = path.read_bytes()
    after = path.lstat()

    def identity(item: object) -> tuple[int, int, int, int]:
        return (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)

    if identity(before) != identity(after):
        raise T2StudyManifestError(f"{field} artifact changed while read")
    if "byte_count" in binding and len(encoded) != binding["byte_count"]:
        raise T2StudyManifestError(f"{field} artifact byte count differs")
    if hashlib.sha256(encoded).hexdigest() != binding["sha256"]:
        raise T2StudyManifestError(f"{field} artifact SHA-256 differs")
    return path


def resolve_t2_reward_binding(
    binding: Mapping[str, object],
    *,
    artifact_root: Path,
    candidate: bool,
) -> tuple[object, str, Path]:
    """Resolve either T2 arm through the immutable Phase B reward registry."""

    checked = _reward_binding(
        binding,
        field="candidate" if candidate else "baseline",
        candidate=candidate,
    )
    if checked["sha256"] == "TBD":
        raise T2StudyManifestError("candidate reward is not ready for registry resolution")
    root = Path(artifact_root).resolve(strict=True)
    path = _verify_artifact(
        root, checked, field="candidate reward" if candidate else "baseline reward"
    )
    try:
        spec, digest = RewardRegistry().load(path)
    except ValueError as exc:
        raise T2StudyManifestError(f"T2 reward registry refused the bound artifact: {exc}") from exc
    expected_type = TargetSpeedRewardSpec if candidate else TrackingOnlyRewardSpec
    if digest != checked["sha256"] or type(spec) is not expected_type:
        raise T2StudyManifestError("T2 reward registry digest differs")
    return spec, digest, path


def load_t2_study_manifest(
    path: Path,
    *,
    repository_root: Path | None = None,
    candidate_artifact_root: Path | None = None,
) -> tuple[dict[str, object], str]:
    """Load canonical bytes and optionally verify every currently frozen file."""

    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_size > 256 * 1024:
        raise T2StudyManifestError("study manifest is unavailable or oversized")
    encoded = candidate.read_bytes()
    try:
        value = json.loads(encoded)
    except (UnicodeError, ValueError) as exc:
        raise T2StudyManifestError("study manifest is not JSON") from exc
    if type(value) is not dict or canonical_json_bytes(value) != encoded:
        raise T2StudyManifestError("study manifest is not canonical JSON")
    validated = validate_t2_study_manifest(value)
    if repository_root is not None:
        root = Path(repository_root).resolve(strict=True)
        common = arm_common_fields(validated["arms"][0])
        verified = {
            field: _verify_artifact(root, common[field], field=field)
            for field in (
                "evaluator",
                "execution_manifest",
                "library",
                "oracle",
                "pairing_adapter",
                "reference_corpus",
                "starting_checkpoint",
                "training_design",
            )
        }
        training_encoded = verified["training_design"].read_bytes()
        try:
            training_value = json.loads(training_encoded)
        except (UnicodeError, ValueError) as exc:
            raise T2StudyManifestError("T2 training design is not JSON") from exc
        if canonical_json_bytes(training_value) != training_encoded:
            raise T2StudyManifestError("T2 training design is not canonical JSON")
        try:
            validate_training_design(training_value)
            _oracle, oracle_sha256 = load_phase_b_oracle(
                verified["oracle"],
                available_behaviors=("expert", "medium", "simple"),
            )
            _evaluator, evaluator_sha256 = load_evaluator_design(
                verified["evaluator"],
                evaluator_source_path=root / "src/oracle_composition/reward_study/t2_evaluator.py",
                protected_metrics_source_path=(
                    root / "src/oracle_composition/phase_b/protected_metrics.py"
                ),
                report_v2_source_path=root / "src/oracle_composition/phase_b/report_v2.py",
                report_writer_source_path=root / "src/oracle_composition/reward_study/t2_report.py",
            )
            _execution, execution_sha256 = load_t2_execution_manifest(
                verified["execution_manifest"],
                repository_root=root,
            )
            _reward, reward_sha256, _baseline_reward_path = resolve_t2_reward_binding(
                validated["arms"][0]["reward"],
                artifact_root=root,
                candidate=False,
            )
            candidate_reward_sha256 = validated["arms"][1]["reward"]["sha256"]
            if candidate_reward_sha256 != "TBD":
                _candidate, resolved_candidate_sha256, _candidate_path = resolve_t2_reward_binding(
                    validated["arms"][1]["reward"],
                    artifact_root=(candidate_artifact_root or root),
                    candidate=True,
                )
                if resolved_candidate_sha256 != candidate_reward_sha256:
                    raise T2StudyManifestError("candidate reward registry digest differs")
            _starting, starting_sha256 = load_starting_checkpoint(
                verified["starting_checkpoint"],
                repository_root=root,
            )
        except ValueError as exc:
            raise T2StudyManifestError(f"T2 bound contract is invalid: {exc}") from exc
        expected_hashes = {
            "evaluator": evaluator_sha256,
            "execution_manifest": execution_sha256,
            "oracle": oracle_sha256,
            "reward": reward_sha256,
            "starting_checkpoint": starting_sha256,
        }
        observed_hashes = {
            "evaluator": common["evaluator"]["sha256"],
            "execution_manifest": common["execution_manifest"]["sha256"],
            "oracle": common["oracle"]["sha256"],
            "reward": validated["arms"][0]["reward"]["sha256"],
            "starting_checkpoint": common["starting_checkpoint"]["sha256"],
        }
        if expected_hashes != observed_hashes:
            raise T2StudyManifestError("T2 bound artifact contract digest differs")
    return validated, hashlib.sha256(encoded).hexdigest()


__all__ = [
    "BASELINE_REWARD_ID",
    "BASELINE_REWARD_SHA256",
    "CANDIDATE_REWARD_ID",
    "PAIRING_EXCLUDED_ARM_FIELDS",
    "STUDY_CLAIM_CEILING",
    "STUDY_EVIDENCE_CLASS",
    "STUDY_ID",
    "STUDY_MANIFEST_SCHEMA_ID",
    "STUDY_STATUS",
    "T2StudyManifestError",
    "arm_common_fields",
    "load_t2_study_manifest",
    "resolve_t2_reward_binding",
    "study_pairing_sha256_from_arm",
    "validate_t2_study_arm_pair",
    "validate_t2_study_manifest",
]
