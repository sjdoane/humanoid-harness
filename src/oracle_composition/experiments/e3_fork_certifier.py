"""Pure certification of the frozen 36-block E3 same-state fork corpus."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import (
    CORPUS_ACTORS,
    CORPUS_SEEDS,
    E3_THRESHOLDS,
    canonical_json_bytes,
    e3_manifest_payload,
    sha256_file,
    sha256_json,
    validate_e3_manifest,
    validate_full_clip_certificate,
)

from .reference_corpus_contract import validate_bundle_manifest, validate_clip_arrays

E3_CERTIFIER_ID = "same_state_three_actor_e3_fork_certifier/v1"


class E3CertificationError(ValueError):
    """The corpus structure is not the frozen same-state experiment."""


def _bound_artifact_sha256(manifest: Mapping[str, object], role: str) -> str:
    matches = [
        item
        for item in manifest["core"]["bound_artifacts"]
        if type(item) is dict and item.get("role") == role
    ]
    if len(matches) != 1 or type(matches[0].get("sha256")) is not str:
        raise E3CertificationError(f"E3 source role {role!r} is missing or duplicated")
    return matches[0]["sha256"]


@dataclass(frozen=True, slots=True)
class E3Branch:
    actor_variant: str
    manifest_sha256: str
    manifest: Mapping[str, object]
    certificate_sha256: str
    certificate: Mapping[str, object]
    arrays: Mapping[str, np.ndarray]


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return math.sqrt(float(np.mean(np.square(left - right))))


def _quaternion_geodesic_max(left: np.ndarray, right: np.ndarray) -> float:
    values: list[float] = []
    for first, second in zip(left, right, strict=True):
        dot = float(np.clip(abs(float(np.dot(first, second))), 0.0, 1.0))
        values.append(2.0 * math.acos(dot))
    return max(values)


def _branch_status(branch: E3Branch) -> dict[str, object]:
    arrays = branch.arrays
    heights = arrays["reference_rows"][1:, 0]
    upright = arrays["boundary_torso_up_z"][1:] >= 0.5
    healthy = (heights > 1.0) & (heights < 2.0)
    flags = arrays["transition_flags"]
    expected_flags = np.zeros((1000, 2), dtype="|u1")
    expected_flags[-1, 1] = 1
    full_horizon = bool(np.array_equal(flags, expected_flags))
    no_collapse = bool(np.all(healthy) and np.all(upright))
    no_nonfoot = not bool(np.any(arrays["transition_nonfoot_floor_contact"]))
    tier_d = bool(branch.certificate["all_transitions_passed"])
    return {
        "actor_variant": branch.actor_variant,
        "bundle_manifest_sha256": branch.manifest_sha256,
        "reference_identity_sha256": branch.manifest["reference_identity_sha256"],
        "tier_d_certificate_sha256": branch.certificate_sha256,
        "tier_d_passed": tier_d,
        "full_horizon": full_horizon,
        "no_collapse": no_collapse,
        "healthy_boundary_count": int(np.count_nonzero(healthy)),
        "upright_boundary_count": int(np.count_nonzero(upright)),
        "no_nonfoot_floor_contact": no_nonfoot,
        "nonfoot_floor_contact_transition_count": int(
            np.count_nonzero(arrays["transition_nonfoot_floor_contact"])
        ),
        "branch_passed": tier_d and full_horizon and no_collapse and no_nonfoot,
    }


def evaluate_e3_pair(
    expert: E3Branch,
    other: E3Branch,
    statuses: Mapping[str, dict[str, object]],
) -> dict[str, object]:
    """Apply the locked action, future-row, and branch-status thresholds."""

    expert_actions = expert.arrays["transition_physical_action"]
    other_actions = other.arrays["transition_physical_action"]
    first_action = float(np.max(np.abs(expert_actions[0] - other_actions[0])))
    first_eight_rms = _rmse(expert_actions[:8], other_actions[:8])
    expert_rows = expert.arrays["reference_rows"][1:8]
    other_rows = other.arrays["reference_rows"][1:8]
    future = {
        "future_root_z_max_abs_m": float(np.max(np.abs(expert_rows[:, 0] - other_rows[:, 0]))),
        "future_quaternion_geodesic_max_rad": _quaternion_geodesic_max(
            expert_rows[:, 1:5], other_rows[:, 1:5]
        ),
        "future_root_linear_velocity_rmse_m_s": _rmse(expert_rows[:, 5:8], other_rows[:, 5:8]),
        "future_root_angular_velocity_rmse_rad_s": _rmse(expert_rows[:, 8:11], other_rows[:, 8:11]),
        "future_joint_position_rmse_rad": _rmse(expert_rows[:, 11:28], other_rows[:, 11:28]),
        "future_joint_velocity_rmse_rad_s": _rmse(expert_rows[:, 28:45], other_rows[:, 28:45]),
    }
    first_passed = first_action > E3_THRESHOLDS["first_action_max_abs_physical"]
    eight_passed = first_eight_rms > E3_THRESHOLDS["first_eight_actions_rms_physical"]
    future_passed_fields = [
        field for field, value in future.items() if value > E3_THRESHOLDS[field]
    ]
    future_passed = bool(future_passed_fields)
    branches_passed = bool(
        statuses[expert.actor_variant]["branch_passed"]
        and statuses[other.actor_variant]["branch_passed"]
    )
    reasons: list[str] = []
    if not first_passed:
        reasons.append("identical_or_weak_first_action")
    if not eight_passed:
        reasons.append("weak_first_eight_action_difference")
    if not future_passed:
        reasons.append("weak_future_reference_variation")
    if not branches_passed:
        reasons.append("branch_horizon_collapse_contact_or_replay_failure")
    return {
        "pair": f"expert_vs_{other.actor_variant}",
        "first_action_max_abs_physical": first_action,
        "first_action_passed": first_passed,
        "first_eight_actions_rms_physical": first_eight_rms,
        "first_eight_actions_passed": eight_passed,
        "future_metrics": future,
        "future_thresholds_passed": future_passed_fields,
        "future_variation_passed": future_passed,
        "both_branches_passed": branches_passed,
        "pair_passed": first_passed and eight_passed and future_passed and branches_passed,
        "failure_reasons": reasons,
    }


def require_distinct_actor_states(branches: tuple[E3Branch, ...]) -> None:
    """Reject a purported three-actor corpus backed by repeated actor state."""

    actor_states = [
        branch.manifest["core"]["source_actor"]["actor_state_sha256"] for branch in branches
    ]
    if len(set(actor_states)) != len(actor_states):
        raise E3CertificationError("self-reference-only or duplicated actor corpus is forbidden")


def require_same_state_anchors(branches: tuple[E3Branch, ...]) -> None:
    """Require identical environment, wrapper, RNG, and controller anchor state."""

    anchor_fields = (
        "boundary_integration_state",
        "boundary_observation",
        "boundary_cfrc_ext",
        "boundary_root_xy",
        "boundary_simulation_time",
        "boundary_wrapper_elapsed",
        "boundary_wrapper_flags",
        "boundary_result_flags",
        "boundary_torso_up_z",
        "reference_rows",
        "boundary_rng_state_sha256",
    )
    expert = branches[0]
    for branch in branches[1:]:
        for field in anchor_fields:
            if not np.array_equal(expert.arrays[field][0], branch.arrays[field][0]):
                raise E3CertificationError(
                    f"nonidentical same-state anchors: {branch.actor_variant} {field}"
                )
        if canonical_json_bytes(
            expert.manifest["core"]["controller_state"]
        ) != canonical_json_bytes(branch.manifest["core"]["controller_state"]):
            raise E3CertificationError("nonidentical controller continuation state")


def _validate_branch(branch: E3Branch, *, seed: int, actor_variant: str) -> None:
    if type(branch) is not E3Branch or branch.actor_variant != actor_variant:
        raise E3CertificationError("E3 branch actor order differs")
    manifest = validate_bundle_manifest(branch.manifest)
    core = manifest["core"]
    if (
        core["clip_kind"] != "corpus"
        or core["seed"] != seed
        or core["actor_variant"] != actor_variant
        or core["steps"] != 1000
    ):
        raise E3CertificationError("E3 branch identity differs")
    validate_clip_arrays(dict(branch.arrays), steps=1000, plain_comparison=True)
    if _bound_artifact_sha256(manifest, "e3_metric_source") != sha256_file(Path(__file__)):
        raise E3CertificationError("running E3 certifier source differs from the replay bundle")
    if _bound_artifact_sha256(manifest, "e3_manifest_source") != sha256_json(e3_manifest_payload()):
        raise E3CertificationError("frozen E3 manifest differs from the replay bundle")
    certificate = validate_full_clip_certificate(branch.certificate)
    if (
        certificate["clip_id"] != core["clip_id"]
        or certificate["bundle_manifest_sha256"] != branch.manifest_sha256
        or certificate["reference_identity_sha256"] != manifest["reference_identity_sha256"]
    ):
        raise E3CertificationError("E3 branch Tier-D certificate binding differs")


def certify_e3_block(seed: int, branches: tuple[E3Branch, ...]) -> dict[str, object]:
    """Certify one structural same-state block and both locked actor pairs."""

    if seed not in CORPUS_SEEDS or type(branches) is not tuple or len(branches) != 3:
        raise E3CertificationError("E3 block seed or branch count differs")
    for branch, actor in zip(branches, CORPUS_ACTORS, strict=True):
        _validate_branch(branch, seed=seed, actor_variant=actor)
    require_distinct_actor_states(branches)
    require_same_state_anchors(branches)
    expert = branches[0]
    statuses = {branch.actor_variant: _branch_status(branch) for branch in branches}
    pairs = [
        evaluate_e3_pair(expert, branches[1], statuses),
        evaluate_e3_pair(expert, branches[2], statuses),
    ]
    block_passed = all(pair["pair_passed"] for pair in pairs)
    return {
        "seed": seed,
        "same_state_anchor_passed": True,
        "branches": [statuses[actor] for actor in CORPUS_ACTORS],
        "pairs": pairs,
        "block_passed": block_passed,
    }


def certify_e3_corpus(
    blocks: tuple[tuple[int, tuple[E3Branch, ...]], ...],
    *,
    manifest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Report every predetermined block with no replacement or omission."""

    design = e3_manifest_payload() if manifest is None else dict(manifest)
    validate_e3_manifest(design)
    if type(blocks) is not tuple or tuple(seed for seed, _branches in blocks) != CORPUS_SEEDS:
        raise E3CertificationError("E3 blocks are missing, replaced, or out of order")
    results = [certify_e3_block(seed, branches) for seed, branches in blocks]
    return summarize_e3_block_results(tuple(results), manifest=design)


def summarize_e3_block_results(
    results: tuple[Mapping[str, object], ...],
    *,
    manifest: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Aggregate already-certified blocks without retaining clip arrays."""

    design = e3_manifest_payload() if manifest is None else dict(manifest)
    validate_e3_manifest(design)
    if (
        type(results) is not tuple
        or tuple(result.get("seed") for result in results) != CORPUS_SEEDS
        or any(type(result) is not dict for result in results)
    ):
        raise E3CertificationError("E3 block results are missing, replaced, or out of order")
    passed = sum(bool(result["block_passed"]) for result in results)
    pair_passes = sum(bool(pair["pair_passed"]) for result in results for pair in result["pairs"])
    payload = {
        "certifier_id": E3_CERTIFIER_ID,
        "schema_version": 1,
        "certifier_source_sha256": sha256_file(Path(__file__)),
        "e3_manifest_sha256": sha256_json(design),
        "block_count": len(results),
        "passed_block_count": passed,
        "failed_block_count": len(results) - passed,
        "qualifying_nonexpert_pair_count": pair_passes,
        "qualifying_corpus": passed > 0,
        "blocks": results,
        "claim_ceiling": "fork_identifiability_only_no_tracker_reference_use_or_robustness",
    }
    payload["result_sha256"] = sha256_json(payload)
    return payload


__all__ = [
    "E3_CERTIFIER_ID",
    "E3Branch",
    "E3CertificationError",
    "certify_e3_block",
    "certify_e3_corpus",
    "evaluate_e3_pair",
    "require_distinct_actor_states",
    "require_same_state_anchors",
    "summarize_e3_block_results",
]
