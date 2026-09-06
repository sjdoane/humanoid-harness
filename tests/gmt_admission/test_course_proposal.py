from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from oracle_composition.adapters.gmt.composition import ReferenceSegment
from oracle_composition.adapters.gmt.course_config import CONFIG_KEYS, CourseRunConfig
from oracle_composition.adapters.gmt.course_evaluation import COURSE_EVALUATOR_ID
from oracle_composition.adapters.gmt.course_proposal import (
    COURSE_FEEDBACK_EVIDENCE_CLASS,
    COURSE_PROPOSAL_SCHEMA_VERSION,
    apply_proposal,
)
from oracle_composition.adapters.gmt.course_task import CourseTaskSpec, TaskRewardRecipe
from oracle_composition.adapters.gmt.reference_runtime import ReferenceMotion
from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.harness.contract import oracle_program_from_dict

WALK_SHA256 = "908ba0e4f6acf1ecf829b0ddb73ed7e649ba6e7ca9fa1a96b43a95e0fdc2e3b0"
CROUCH_SHA256 = "a67c364e7d0013c24f43096e194f5d8099b2d1ebcaa3db42767402a0924be964"


def _motion(height: float) -> ReferenceMotion:
    root = np.zeros((301, 3), dtype=np.float32)
    root[:, 2] = height
    rotations = np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (301, 1))
    return ReferenceMotion(
        {
            "fps": np.array([30.0]),
            "root_pos": root,
            "root_rot": rotations,
            "dof_pos": np.zeros((301, 23), dtype=np.float32),
        }
    )


def _oracle(oracle_id: str = "parent") -> dict:
    return {
        "schema_version": 1,
        "evidence_class": "exploratory_oracle_cycle",
        "oracle_id": oracle_id,
        "behaviors": ["walk", "crouch"],
        "initial": "before",
        "states": {
            "before": {"behavior": "walk", "min_dwell": 25},
            "inside": {"behavior": "crouch", "min_dwell": 25},
            "after": {"behavior": "walk", "min_dwell": 25},
        },
        "transitions": [
            {"from": "before", "to": "inside", "priority": 0, "guard": "x_travelled >= 1"},
            {"from": "inside", "to": "after", "priority": 0, "guard": "x_travelled >= 2"},
        ],
    }


def _parent() -> CourseRunConfig:
    task = CourseTaskSpec(
        region_entry_distance_m=1.0,
        region_exit_distance_m=2.0,
        finish_distance_m=3.5,
        target_speed_outside_m_s=0.7,
        target_speed_inside_m_s=0.65,
        posture_band_low_m=0.30,
        posture_band_high_m=0.60,
        horizon_steps=1_000,
    )
    recipe = TaskRewardRecipe(1.0, 2.0, 1.0, 0.5, 1.0)
    oracle = _oracle()
    assets = {
        "upstream_root": "/frozen/upstream",
        "weights": {"path": "/frozen/weights.npz", "sha256": "b" * 64},
        "motions": {
            "walk_stand": {"path": "/frozen/walk.npz", "sha256": WALK_SHA256},
            "crouchwalk_stand": {
                "path": "/frozen/crouch.npz",
                "sha256": CROUCH_SHA256,
            },
        },
    }
    raw = {
        "schema_version": 1,
        "mode": "train",
        "assets": assets,
        "task": task.to_dict(),
        "oracle": oracle,
        "segments": {
            "walk": {
                "motion_name": "walk_stand",
                "start_seconds": 0.0,
                "end_seconds": 10.0,
            },
            "crouch": {
                "motion_name": "crouchwalk_stand",
                "start_seconds": 0.0,
                "end_seconds": 10.0,
            },
        },
        "reward": recipe.to_dict(),
        "seed": 17,
        "training_steps": 512,
    }
    motions = {"walk": _motion(0.8), "crouch": _motion(0.48)}
    segments = {
        name: ReferenceSegment(
            motions[name],
            WALK_SHA256 if name == "walk" else CROUCH_SHA256,
            0.0,
            10.0,
        )
        for name in motions
    }
    encoded = canonical_json_bytes(raw)
    return CourseRunConfig(
        raw=raw,
        encoded=encoded,
        sha256=hashlib.sha256(encoded).hexdigest(),
        assets=assets,
        task=task,
        recipe=recipe,
        program=oracle_program_from_dict(oracle, available_behaviors=list(segments)),
        segments=segments,
    )


def _feedback(parent: CourseRunConfig) -> bytes:
    return canonical_json_bytes(
        {
            "evidence_class": COURSE_FEEDBACK_EVIDENCE_CLASS,
            "protected_evaluation": False,
            "source_manifest_sha256": "f" * 64,
            "evaluation": {
                "evaluator_id": COURSE_EVALUATOR_ID,
                "claim_scope": "fixed_development_gate_only_not_heldout_or_universal",
                "task_sha256": parent.task.sha256,
                "development_gate_results": {
                    "full_horizon_without_fall": True,
                    "finish_reached": True,
                    "region_entry_and_exit_observed": True,
                    "minimum_inside_samples": True,
                    "inside_posture_compliance": True,
                    "inside_posture_dip": True,
                    "inside_mean_speed_target": False,
                    "mean_speed_error": True,
                    "maximum_lateral_error": True,
                    "joint_position_rmse_p95": True,
                    "roll_pitch_rmse_p95": True,
                },
                "development_gate_passed": False,
                "episode_success": None,
            },
            "diagnosis": "Inside-region speed remained below its preregistered target.",
        }
    )


def _reward_replacement() -> dict:
    return {"reward": TaskRewardRecipe(1.5, 2.0, 1.0, 0.5, 1.0).to_dict()}


def _oracle_replacement() -> dict:
    return {
        "oracle": _oracle("revised"),
        "segments": {
            "walk": {
                "motion_name": "walk_stand",
                "start_seconds": 0.0,
                "end_seconds": 10.0,
            },
            "crouch": {
                "motion_name": "crouchwalk_stand",
                "start_seconds": 0.1,
                "end_seconds": 9.9,
            },
        },
    }


def _proposal(
    parent: CourseRunConfig,
    feedback: bytes,
    *,
    factor: str = "reward",
) -> dict:
    return {
        "schema_version": COURSE_PROPOSAL_SCHEMA_VERSION,
        "proposal_id": "course_revision_001",
        "parent_config_sha256": parent.sha256,
        "feedback_sha256": hashlib.sha256(feedback).hexdigest(),
        "factor": factor,
        "hypothesis": "Increase the fixed speed component without changing the course.",
        "replacement": _reward_replacement() if factor == "reward" else _oracle_replacement(),
    }


def test_reward_proposal_changes_only_reward_and_does_not_mutate_parent() -> None:
    parent = _parent()
    original = copy.deepcopy(parent.raw)
    feedback = _feedback(parent)

    candidate = apply_proposal(parent, _proposal(parent, feedback), feedback)

    assert candidate["reward"] == _reward_replacement()["reward"]
    for field in CONFIG_KEYS - {"reward"}:
        assert canonical_json_bytes(candidate[field]) == canonical_json_bytes(parent.raw[field])
    assert set(candidate) == CONFIG_KEYS
    assert parent.raw == original


def test_oracle_proposal_changes_only_oracle_and_segments() -> None:
    parent = _parent()
    feedback = _feedback(parent)

    candidate = apply_proposal(
        parent,
        _proposal(parent, feedback, factor="oracle"),
        feedback,
    )

    assert candidate["oracle"]["oracle_id"] == "revised"
    assert candidate["segments"]["crouch"]["start_seconds"] == 0.1
    for field in CONFIG_KEYS - {"oracle", "segments"}:
        assert canonical_json_bytes(candidate[field]) == canonical_json_bytes(parent.raw[field])


@pytest.mark.parametrize("frozen_field", ["task", "assets", "training_steps", "mode", "seed"])
def test_proposal_cannot_mix_in_a_frozen_field(frozen_field: str) -> None:
    parent = _parent()
    feedback = _feedback(parent)
    proposal = _proposal(parent, feedback)
    proposal["replacement"][frozen_field] = parent.raw[frozen_field]

    with pytest.raises(ValueError, match="exactly reward"):
        apply_proposal(parent, proposal, feedback)


@pytest.mark.parametrize("field", ["task", "assets", "training_steps", "mode"])
def test_extra_top_level_proposal_fields_fail_closed(field: str) -> None:
    parent = _parent()
    feedback = _feedback(parent)
    proposal = _proposal(parent, feedback)
    proposal[field] = parent.raw[field]

    with pytest.raises(ValueError, match="proposal fields"):
        apply_proposal(parent, proposal, feedback)


def test_parent_and_feedback_hashes_are_exact() -> None:
    parent = _parent()
    feedback = _feedback(parent)
    proposal = _proposal(parent, feedback)
    proposal["parent_config_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="parent_config_sha256 differs"):
        apply_proposal(parent, proposal, feedback)

    proposal = _proposal(parent, feedback)
    proposal["feedback_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="feedback bytes differ"):
        apply_proposal(parent, proposal, feedback)


@pytest.mark.parametrize("value", [6.0, float("nan"), True])
def test_invalid_reward_values_fail_closed(value: object) -> None:
    parent = _parent()
    feedback = _feedback(parent)
    proposal = _proposal(parent, feedback)
    proposal["replacement"]["reward"]["speed_weight"] = value

    with pytest.raises(ValueError, match=r"weights|finite"):
        apply_proposal(parent, proposal, feedback)


def test_wrong_factor_payload_and_unadmitted_motion_fail_closed() -> None:
    parent = _parent()
    feedback = _feedback(parent)
    mixed = _proposal(parent, feedback)
    mixed["replacement"]["oracle"] = _oracle()
    with pytest.raises(ValueError, match="exactly reward"):
        apply_proposal(parent, mixed, feedback)

    proposal = _proposal(parent, feedback, factor="oracle")
    proposal["replacement"]["segments"]["crouch"]["motion_name"] = "dance"
    with pytest.raises(ValueError, match="unadmitted motion"):
        apply_proposal(parent, proposal, feedback)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("protected_evaluation",), True, "protection boundary"),
        (("evidence_class",), "protected_report/v1", "evidence class"),
        (("source_manifest_sha256",), "F" * 64, "source_manifest_sha256"),
        (("evaluation", "task_sha256"), "0" * 64, "objective summary"),
        (("evaluation", "development_gate_passed"), True, "objective summary"),
        (("evaluation", "episode_success"), True, "objective summary"),
        (("diagnosis",), "", "diagnosis"),
    ],
)
def test_feedback_boundary_rejects_wrong_or_protected_evidence(
    path: tuple[str, ...], value: object, message: str
) -> None:
    parent = _parent()
    raw = json.loads(_feedback(parent))
    target = raw
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    feedback = canonical_json_bytes(raw)

    with pytest.raises(ValueError, match=message):
        apply_proposal(parent, _proposal(parent, feedback), feedback)


def test_nonfinite_duplicate_or_extra_feedback_fails_closed() -> None:
    parent = _parent()
    raw = json.loads(_feedback(parent))
    raw["extra"] = "not admitted"
    extra = canonical_json_bytes(raw)
    with pytest.raises(ValueError, match="feedback fields"):
        apply_proposal(parent, _proposal(parent, extra), extra)

    nonfinite = _feedback(parent).replace(b'"episode_success":null', b'"episode_success":NaN')
    with pytest.raises(ValueError, match="non-finite"):
        apply_proposal(parent, _proposal(parent, nonfinite), nonfinite)

    duplicate = _feedback(parent).replace(
        b'"protected_evaluation":false',
        b'"protected_evaluation":false,"protected_evaluation":false',
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        apply_proposal(parent, _proposal(parent, duplicate), duplicate)


def test_malformed_metadata_and_mutated_parent_fail_closed() -> None:
    parent = _parent()
    feedback = _feedback(parent)
    proposal = _proposal(parent, feedback)
    proposal["schema_version"] = True
    with pytest.raises(ValueError, match="schema version"):
        apply_proposal(parent, proposal, feedback)

    proposal = _proposal(parent, feedback)
    proposal["hypothesis"] = " "
    with pytest.raises(ValueError, match="hypothesis"):
        apply_proposal(parent, proposal, feedback)

    changed = copy.deepcopy(parent.raw)
    changed["seed"] += 1
    forged = replace(parent, raw=changed)
    with pytest.raises(ValueError, match="parent bytes"):
        apply_proposal(forged, _proposal(forged, feedback), feedback)
