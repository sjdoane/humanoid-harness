from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.phase_b.contracts import PhaseBContractError, T2RewardPairing
from oracle_composition.phase_b.training import (
    PPORecipe,
    TrainingPlan,
    domain_separated_seed,
)
from oracle_composition.reward_study.pairing import (
    NON_PAIRED_ID,
    TRAINING_STREAM_DOMAINS,
    build_t2_training_plan,
    derive_evaluation_seed_identities,
    derive_t2_rsi_assignment,
    derive_training_stream_seeds,
    pairing_from_study_manifest_bytes,
    validate_pairing_receipt,
)
from oracle_composition.reward_study.study_manifest import (
    STUDY_FINAL_READY_STATUS,
    study_pairing_sha256_from_arm,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/004_t2_reward_study"
STUDY_PATH = EXPERIMENT / "t2_reward_study_expert_hold_v1.json"
SUPERSEDED_STUDY_PATH = (
    ROOT / "experiments/004_t2_reward_study/superseded/superseded_t2pairr1_study_manifest_v1.json"
)
RECEIPT_PATH = EXPERIMENT / "pairing_receipt_v1.json"


def _study() -> dict[str, object]:
    return json.loads(STUDY_PATH.read_bytes())


def _completed_fake_study() -> dict[str, object]:
    value = _study()
    value["arms"][1]["reward"] = {
        "path": "fake-runtime/candidate_reward.json",
        "reward_id": "target_speed_triangular_affine_t2_adapter/v1",
        "sha256": hashlib.sha256(b"fake candidate reward bytes").hexdigest(),
    }
    value["status"] = STUDY_FINAL_READY_STATUS
    return value


def _pairing(value: dict[str, object] | None = None) -> T2RewardPairing:
    return pairing_from_study_manifest_bytes(canonical_json_bytes(value or _completed_fake_study()))


def test_exact_study_and_arm_bytes_retain_distinct_execution_identities() -> None:
    pairing = _pairing()
    assert pairing.study_pairing_sha256 == _study()["study_pairing_sha256"]
    assert pairing.baseline_arm_manifest_sha256 != pairing.candidate_arm_manifest_sha256
    assert pairing.to_dict()["pairing_id"] == "t2_reward_pairing/v1"
    assert pairing.to_dict()["reward_sha256"] == {
        "baseline": pairing.baseline_reward_sha256,
        "candidate": pairing.candidate_reward_sha256,
    }


def test_exact_byte_authority_refuses_schema_shaped_common_field_addition() -> None:
    value = _completed_fake_study()
    for arm in value["arms"]:
        arm["unexpected_common_field"] = "same-in-both-arms"
    common = {
        name: field
        for name, field in value["arms"][0].items()
        if name not in {"arm_label", "output_path", "reward", "timestamps"}
    }
    value["study_pairing_sha256"] = hashlib.sha256(canonical_json_bytes(common)).hexdigest()
    with pytest.raises(PhaseBContractError, match="arm manifest fields differ"):
        T2RewardPairing.from_study_manifest_bytes(canonical_json_bytes(value))


def test_reward_only_arms_have_identical_integrated_fake_runtime_streams() -> None:
    receipt = json.loads(RECEIPT_PATH.read_bytes())
    assert validate_pairing_receipt(receipt) == receipt
    for binding in receipt["runtime_sources"].values():
        source = ROOT / binding["path"]
        assert source.stat().st_size == binding["byte_count"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == binding["sha256"]
    assert receipt["reward_sha256"]["baseline"] != receipt["reward_sha256"]["candidate"]
    assert all(row["identical"] is True for row in receipt["per_seed"])
    assert all(row["identical"] is True for row in receipt["per_seed_and_environment_slot"])
    assert {row["environment_slot"] for row in receipt["per_seed_and_environment_slot"]} == {
        0,
        1,
        2,
        3,
    }


@pytest.mark.parametrize("mutation", ["source_path", "stream_digest"])
def test_pairing_receipt_refuses_rebound_sources_and_non_digests(mutation: str) -> None:
    receipt = json.loads(RECEIPT_PATH.read_bytes())
    if mutation == "source_path":
        receipt["runtime_sources"]["phase_b_contracts"]["path"] = (
            "src/oracle_composition/phase_b/runtime.py"
        )
        message = "runtime-source path differs"
    else:
        row = receipt["per_seed_and_environment_slot"][0]
        row["baseline_sha256"] = row["candidate_sha256"] = "not-a-digest"
        message = "SHA-256"
    with pytest.raises(ValueError, match=message):
        validate_pairing_receipt(receipt)


def test_changed_frozen_field_changes_key_and_stale_key_is_refused() -> None:
    original = _completed_fake_study()
    changed = copy.deepcopy(original)
    for arm in changed["arms"]:
        arm["task"]["target_com_forward_speed_m_s"] = 3.1
    with pytest.raises(PhaseBContractError, match="pairing SHA-256"):
        T2RewardPairing.from_study_manifest_bytes(canonical_json_bytes(changed))
    changed["study_pairing_sha256"] = study_pairing_sha256_from_arm(changed["arms"][0])
    changed_pairing = T2RewardPairing.from_study_manifest_bytes(canonical_json_bytes(changed))
    assert changed_pairing.study_pairing_sha256 != _pairing(original).study_pairing_sha256


@pytest.mark.parametrize("mutation", ["missing", "forged"])
def test_missing_or_forged_key_is_refused_from_exact_bytes(mutation: str) -> None:
    value = _completed_fake_study()
    if mutation == "missing":
        del value["study_pairing_sha256"]
    else:
        value["study_pairing_sha256"] = "0" * 64
    with pytest.raises((PhaseBContractError, ValueError), match=r"manifest fields|pairing SHA-256"):
        _pairing(value)


def test_paired_plan_cannot_start_from_a_supplied_digest_without_study_bytes() -> None:
    with pytest.raises(ValueError, match="exact verified study and arm bytes"):
        TrainingPlan(
            seed=11,
            transitions=128,
            manifest_sha256="a" * 64,
            evidence_class="interface_check",
            promotable=False,
            smoke=False,
            pairing_declared=True,
            steps_per_environment=4,
            recipe=PPORecipe(batch_size=16, n_epochs=1),
            test_only=True,
        )
    with pytest.raises(ValueError, match="exact verified study and arm bytes"):
        derive_training_stream_seeds(
            ppo_seed=121001,
            pairing_declared=True,
            study_pairing="a" * 64,
        )


def test_paired_plan_refuses_study_bytes_with_a_missing_candidate_reward_hash() -> None:
    incomplete = T2RewardPairing.from_study_manifest_bytes(SUPERSEDED_STUDY_PATH.read_bytes())
    with pytest.raises(ValueError, match="both exact reward SHA-256 identities"):
        build_t2_training_plan(
            pairing=incomplete,
            arm_label="baseline",
            ppo_seed=121001,
            transitions=128,
            evidence_class="interface_check",
            promotable=False,
            smoke=False,
            steps_per_environment=4,
            recipe=PPORecipe(batch_size=16, n_epochs=1),
            test_only=True,
        )


def test_rsi_and_evaluation_schedules_are_indexed_per_environment_slot() -> None:
    pairing = _pairing()
    left = derive_t2_rsi_assignment(
        pairing=pairing,
        ppo_seed=121001,
        global_episode_index=0,
        environment_index=2,
    )
    right = derive_t2_rsi_assignment(
        pairing=pairing,
        ppo_seed=121001,
        global_episode_index=0,
        environment_index=3,
    )
    assert left.global_episode_index == right.global_episode_index == 0
    assert left.reference_behavior == right.reference_behavior == "expert"
    assert left.schedule_class == right.schedule_class == "hold"
    assert (left.block, left.start_boundary) != (right.block, right.start_boundary)
    identities = derive_evaluation_seed_identities(
        pairing=pairing,
        ppo_seed=121001,
        evaluation_seeds=[97001, 97002],
    )
    assert [row["declared_index"] for row in identities] == [0, 1]
    assert [row["environment_reset_seed"] for row in identities] == [97001, 97002]


def test_legacy_single_arm_derivation_is_unchanged_and_labeled_non_paired() -> None:
    manifest = "a" * 64
    observed = derive_training_stream_seeds(
        ppo_seed=121001,
        pairing_declared=False,
        manifest_sha256=manifest,
    )
    assert observed == {
        domain: domain_separated_seed(manifest, 121001, domain)
        for domain in TRAINING_STREAM_DOMAINS
    }
    plan = TrainingPlan(
        seed=11,
        transitions=128,
        manifest_sha256=manifest,
        evidence_class="interface_check",
        promotable=False,
        smoke=False,
        steps_per_environment=4,
        recipe=PPORecipe(batch_size=16, n_epochs=1),
        test_only=True,
    )
    assert plan.pairing_id == NON_PAIRED_ID
    assert plan.randomization_sha256 == manifest


def test_adapter_builds_both_plans_from_the_same_verified_study_bytes() -> None:
    pairing = _pairing()
    plans = [
        build_t2_training_plan(
            pairing=pairing,
            arm_label=label,
            ppo_seed=121001,
            transitions=128,
            evidence_class="interface_check",
            promotable=False,
            smoke=False,
            steps_per_environment=4,
            recipe=PPORecipe(batch_size=16, n_epochs=1),
            test_only=True,
        )
        for label in ("baseline", "candidate")
    ]
    assert plans[0].manifest_sha256 != plans[1].manifest_sha256
    assert plans[0].randomization_sha256 == plans[1].randomization_sha256


def test_committed_receipt_is_canonical_and_retains_predecessor_lineage() -> None:
    encoded = RECEIPT_PATH.read_bytes()
    value = json.loads(encoded)
    assert encoded == canonical_json_bytes(value)
    assert hashlib.sha256(encoded).hexdigest() == (
        "1a2b7ece139974117fd5c75e040d9cc52cd51a4e42c9b5afd794a60b02232348"
    )
    assert validate_pairing_receipt(value) == value
    assert value["source_study_manifest_sha256"] == (
        "8e81792ab6b4847776ce4cd352bb4eda6c0f705ceaf9ff644a908508c8982d10"
    )
    assert value["study_pairing_sha256"] == (
        "64529d781ae3fb5030ce6d018504c69e31e6e62307775c8e47cdb8c81996c1e7"
    )
    assert (
        hashlib.sha256(STUDY_PATH.read_bytes()).hexdigest() != value["source_study_manifest_sha256"]
    )
