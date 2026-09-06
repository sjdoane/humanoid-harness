"""Validate one data-only G1 course revision without treating it as evidence."""

from __future__ import annotations

import copy
import hashlib
import re
from typing import Any

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.harness.contract import decode_json_object, oracle_program_from_dict

from .composition import ReferenceSegment
from .course_config import CourseRunConfig
from .course_evaluation import COURSE_EVALUATOR_ID
from .course_task import TaskRewardRecipe

COURSE_FEEDBACK_EVIDENCE_CLASS = "gmt_g1_course_development_feedback/v1"
COURSE_PROPOSAL_SCHEMA_VERSION = 1
MAX_HYPOTHESIS_CHARACTERS = 2_048
MAX_DIAGNOSIS_CHARACTERS = 4_096

_PROPOSAL_FIELDS = {
    "schema_version",
    "proposal_id",
    "parent_config_sha256",
    "feedback_sha256",
    "factor",
    "hypothesis",
    "replacement",
}
_FEEDBACK_FIELDS = {
    "evidence_class",
    "protected_evaluation",
    "source_manifest_sha256",
    "evaluation",
    "diagnosis",
}
_EVALUATION_FIELDS = {
    "evaluator_id",
    "claim_scope",
    "task_sha256",
    "development_gate_results",
    "development_gate_passed",
    "episode_success",
}
_DEVELOPMENT_GATES = {
    "full_horizon_without_fall",
    "finish_reached",
    "region_entry_and_exit_observed",
    "minimum_inside_samples",
    "inside_posture_compliance",
    "inside_posture_dip",
    "inside_mean_speed_target",
    "mean_speed_error",
    "maximum_lateral_error",
    "joint_position_rmse_p95",
    "roll_pitch_rmse_p95",
}
_PROPOSAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DEVELOPMENT_SCOPE = "fixed_development_gate_only_not_heldout_or_universal"


def _sha256(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be one lowercase SHA-256")
    return value


def _text(value: object, field: str, maximum: int) -> str:
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise ValueError(f"{field} must be nonempty bounded text")
    return value


def _feedback(parent: CourseRunConfig, encoded: bytes, expected_sha256: str) -> dict:
    if hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise ValueError("proposal feedback bytes differ from feedback_sha256")
    value = decode_json_object(encoded, source="G1 course development feedback")
    if set(value) != _FEEDBACK_FIELDS:
        raise ValueError("course feedback fields differ")
    if (
        value["evidence_class"] != COURSE_FEEDBACK_EVIDENCE_CLASS
        or value["protected_evaluation"] is not False
    ):
        raise ValueError("course feedback evidence class or protection boundary differs")
    _sha256(value["source_manifest_sha256"], "source_manifest_sha256")
    _text(value["diagnosis"], "diagnosis", MAX_DIAGNOSIS_CHARACTERS)
    evaluation = value["evaluation"]
    if type(evaluation) is not dict or set(evaluation) != _EVALUATION_FIELDS:
        raise ValueError("course feedback evaluation must be one objective summary")
    results = evaluation.get("development_gate_results")
    if (
        evaluation.get("evaluator_id") != COURSE_EVALUATOR_ID
        or evaluation.get("claim_scope") != _DEVELOPMENT_SCOPE
        or evaluation.get("task_sha256") != parent.task.sha256
        or type(results) is not dict
        or set(results) != _DEVELOPMENT_GATES
        or any(type(result) is not bool for result in results.values())
        or type(evaluation.get("development_gate_passed")) is not bool
        or evaluation["development_gate_passed"] is not all(results.values())
        or evaluation.get("episode_success", object()) is not None
    ):
        raise ValueError("course feedback objective summary differs")
    return value


def _motion_objects(parent: CourseRunConfig) -> dict[str, Any]:
    motions = parent.assets["motions"]
    result = {}
    for name, asset in motions.items():
        matches = [
            segment.motion
            for segment in parent.segments.values()
            if segment.parent_sha256 == asset["sha256"]
        ]
        if not matches:
            raise ValueError("parent config lacks its admitted motion runtime")
        result[name] = matches[0]
    return result


def _oracle_replacement(parent: CourseRunConfig, replacement: object) -> dict[str, object]:
    if type(replacement) is not dict or set(replacement) != {"oracle", "segments"}:
        raise ValueError("oracle replacement must contain exactly oracle and segments")
    raw_segments = replacement["segments"]
    if type(raw_segments) is not dict or not 1 <= len(raw_segments) <= 8:
        raise ValueError("oracle replacement must contain one to eight segments")
    motions = _motion_objects(parent)
    segments = {}
    consumed = set()
    for behavior, raw in raw_segments.items():
        required = {"motion_name", "start_seconds", "end_seconds"}
        optional = {"entry_phase_end_seconds", "boundary"}
        if type(raw) is not dict or not required <= set(raw) <= required | optional:
            raise ValueError("proposal segment fields differ")
        motion_name = raw["motion_name"]
        if type(motion_name) is not str or motion_name not in motions:
            raise ValueError("proposal segment uses an unadmitted motion")
        if any(type(raw[name]) is not float for name in ("start_seconds", "end_seconds")):
            raise ValueError("proposal segment bounds must be floats")
        parent_sha256 = parent.assets["motions"][motion_name]["sha256"]
        options = {
            name: raw[name] for name in ("entry_phase_end_seconds", "boundary") if name in raw
        }
        segments[behavior] = ReferenceSegment(
            motions[motion_name],
            parent_sha256,
            raw["start_seconds"],
            raw["end_seconds"],
            **options,
        )
        consumed.add(motion_name)
    if consumed != set(motions):
        raise ValueError("proposal must consume exactly the parent's admitted motions")
    program = oracle_program_from_dict(replacement["oracle"], available_behaviors=list(segments))
    if set(program.states) != {"before", "inside", "after"}:
        raise ValueError("proposal must preserve the three course state slots")
    if set(program.behaviors) != set(segments):
        raise ValueError("proposal oracle and segment behaviors must agree exactly")
    return {"oracle": program.to_dict(), "segments": copy.deepcopy(raw_segments)}


def apply_proposal(parent: CourseRunConfig, proposal: dict, feedback: bytes) -> dict[str, object]:
    """Return a new config; validation is not truth, and inputs remain separate provenance."""

    if type(parent) is not CourseRunConfig:
        raise ValueError("proposal parent must be one admitted CourseRunConfig")
    if (
        hashlib.sha256(parent.encoded).hexdigest() != parent.sha256
        or decode_json_object(parent.encoded, source="parent course config") != parent.raw
    ):
        raise ValueError("proposal parent bytes and admitted config differ")
    if type(proposal) is not dict or set(proposal) != _PROPOSAL_FIELDS:
        raise ValueError("course proposal fields differ")
    if (
        type(proposal["schema_version"]) is not int
        or proposal["schema_version"] != COURSE_PROPOSAL_SCHEMA_VERSION
    ):
        raise ValueError("course proposal schema version differs")
    if type(proposal["proposal_id"]) is not str or not _PROPOSAL_ID.fullmatch(
        proposal["proposal_id"]
    ):
        raise ValueError("proposal_id is malformed")
    if _sha256(proposal["parent_config_sha256"], "parent_config_sha256") != parent.sha256:
        raise ValueError("proposal parent_config_sha256 differs")
    feedback_sha256 = _sha256(proposal["feedback_sha256"], "feedback_sha256")
    _feedback(parent, feedback, feedback_sha256)
    _text(proposal["hypothesis"], "hypothesis", MAX_HYPOTHESIS_CHARACTERS)
    factor = proposal["factor"]
    candidate = copy.deepcopy(parent.raw)
    if factor == "oracle":
        candidate.update(_oracle_replacement(parent, proposal["replacement"]))
        changed = {"oracle", "segments"}
    elif factor == "reward":
        replacement = proposal["replacement"]
        if type(replacement) is not dict or set(replacement) != {"reward"}:
            raise ValueError("reward replacement must contain exactly reward")
        candidate["reward"] = TaskRewardRecipe.from_dict(replacement["reward"]).to_dict()
        changed = {"reward"}
    else:
        raise ValueError("proposal factor must be oracle or reward")
    parent_fields = set(parent.raw)
    if set(candidate) != parent_fields or any(
        canonical_json_bytes(candidate[name]) != canonical_json_bytes(parent.raw[name])
        for name in parent_fields - changed
    ):
        raise ValueError("proposal changed a frozen parent config field")
    return candidate


__all__ = [
    "COURSE_FEEDBACK_EVIDENCE_CLASS",
    "COURSE_PROPOSAL_SCHEMA_VERSION",
    "apply_proposal",
]
