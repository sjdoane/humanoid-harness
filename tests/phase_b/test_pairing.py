from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.contracts.reference_identity_v2 import array_sha256, canonical_json_bytes
from oracle_composition.phase_b import runtime as runtime_module
from oracle_composition.phase_b.contracts import PhaseBContractError, T2RewardPairing
from oracle_composition.phase_b.runtime import fake_policy_factory
from oracle_composition.phase_b.training import (
    BalancedRSIScheduler,
    PPORecipe,
    TrainingPlan,
    paired_action_noise,
    paired_minibatch_permutation,
)
from oracle_composition.reward_study.pairing import build_t2_training_plan

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "experiments/004_t2_reward_study/t2_reward_study_expert_hold_v1.json"


def _pairing() -> T2RewardPairing:
    value = json.loads(STUDY.read_bytes())
    value["arms"][1]["reward"] = {
        "path": "fake-runtime/candidate_reward.json",
        "reward_id": "target_speed_triangular_affine_t2_adapter/v1",
        "sha256": hashlib.sha256(b"candidate reward").hexdigest(),
    }
    return T2RewardPairing.from_study_manifest_bytes(canonical_json_bytes(value))


def _plans() -> tuple[TrainingPlan, TrainingPlan]:
    pairing = _pairing()
    values = []
    for label in ("baseline", "candidate"):
        values.append(
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
        )
    return values[0], values[1]


def _policy_sha256(plan: TrainingPlan) -> str:
    policy = fake_policy_factory(plan)
    identities = {
        name: array_sha256(np.ascontiguousarray(tensor.detach().cpu().numpy()))
        for name, tensor in policy.state_dict().items()
    }
    return hashlib.sha256(canonical_json_bytes(identities)).hexdigest()


def test_runtime_consumers_use_pairing_key_while_retaining_arm_execution_hashes() -> None:
    baseline, candidate = _plans()
    assert baseline.manifest_sha256 != candidate.manifest_sha256
    assert baseline.randomization_sha256 == candidate.randomization_sha256
    assert _policy_sha256(baseline) == _policy_sha256(candidate)
    for rollout_index in (0, 7):
        assert np.array_equal(
            paired_action_noise(
                baseline,
                rollout_index=rollout_index,
                steps_per_environment=11,
            ),
            paired_action_noise(
                candidate,
                rollout_index=rollout_index,
                steps_per_environment=11,
            ),
        )
    for update_index in (0, 13):
        assert np.array_equal(
            paired_minibatch_permutation(
                baseline,
                update_index=update_index,
                sample_count=128,
            ),
            paired_minibatch_permutation(
                candidate,
                update_index=update_index,
                sample_count=128,
            ),
        )


def test_paired_reset_counters_are_independent_per_environment_slot() -> None:
    baseline, _candidate = _plans()
    assignments = runtime_module._SharedResetAssignments(BalancedRSIScheduler.from_plan(baseline))
    assert assignments.rehearsal(2).global_episode_index == 0
    assert assignments.rehearsal(2).global_episode_index == 1
    assert assignments.rehearsal(3).global_episode_index == 0
    first_composition = assignments.composition_block(0)
    assignments.composition_block(0)
    first_other_slot = assignments.composition_block(1)
    scheduler = BalancedRSIScheduler.from_plan(baseline)
    assert first_composition == scheduler.composition_block(
        global_episode_index=0,
        environment_index=0,
    )
    assert first_other_slot == scheduler.composition_block(
        global_episode_index=0,
        environment_index=1,
    )


def test_declared_pairing_refuses_missing_bytes_before_any_factory_call() -> None:
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


def test_exact_arm_bytes_cannot_be_replaced_behind_the_study_manifest() -> None:
    encoded = STUDY.read_bytes()
    value = json.loads(encoded)
    baseline = canonical_json_bytes(value["arms"][0])
    candidate = json.loads(canonical_json_bytes(value["arms"][1]))
    candidate["reward"]["sha256"] = hashlib.sha256(b"other reward").hexdigest()
    with pytest.raises(PhaseBContractError, match="exact arm manifest bytes differ"):
        T2RewardPairing(
            study_manifest_bytes=encoded,
            baseline_arm_manifest_bytes=baseline,
            candidate_arm_manifest_bytes=canonical_json_bytes(candidate),
        )
