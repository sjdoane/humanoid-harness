from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.contracts.reference_identity_v2 import array_sha256, canonical_json_bytes
from oracle_composition.phase_b import runtime as runtime_module
from oracle_composition.phase_b import training as training_module
from oracle_composition.phase_b.contracts import PhaseBContractError, T2RewardPairing
from oracle_composition.phase_b.runtime import fake_environment_factories, fake_policy_factory
from oracle_composition.phase_b.training import (
    BalancedRSIScheduler,
    PPORecipe,
    TrainingPlan,
    paired_action_noise,
    paired_minibatch_permutation,
    run_ppo_training,
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
                transitions=32,
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


def _paired_production_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, dict[str, object]]:
    baseline, candidate = _plans()
    labels = {
        baseline.manifest_sha256: "baseline",
        candidate.manifest_sha256: "candidate",
    }
    captures: dict[str, dict[str, object]] = {
        label: {"action_primitives": [], "minibatch_permutations": []} for label in labels.values()
    }
    original_action_noise = training_module.paired_action_noise
    original_minibatch_permutation = training_module.paired_minibatch_permutation

    def capture_action_noise(
        plan: TrainingPlan,
        *,
        rollout_index: int,
        steps_per_environment: int,
    ) -> np.ndarray:
        value = original_action_noise(
            plan,
            rollout_index=rollout_index,
            steps_per_environment=steps_per_environment,
        )
        captures[labels[plan.manifest_sha256]]["action_primitives"].append(value.copy())
        return value

    def capture_minibatch_permutation(
        plan: TrainingPlan,
        *,
        update_index: int,
        sample_count: int,
    ) -> np.ndarray:
        value = original_minibatch_permutation(
            plan,
            update_index=update_index,
            sample_count=sample_count,
        )
        captures[labels[plan.manifest_sha256]]["minibatch_permutations"].append(value.copy())
        return value

    def observe_factory(
        factory: Callable[[], object], environments: list[object]
    ) -> Callable[[], object]:
        def build() -> object:
            environment = factory()
            environments.append(environment)
            return environment

        return build

    with monkeypatch.context() as observer:
        observer.setattr(training_module, "paired_action_noise", capture_action_noise)
        observer.setattr(
            training_module,
            "paired_minibatch_permutation",
            capture_minibatch_permutation,
        )
        for label, plan in (("baseline", baseline), ("candidate", candidate)):
            created = []
            factories = fake_environment_factories(plan=plan, episode_steps=2)
            try:
                result = run_ppo_training(
                    plan=plan,
                    policy_factory=fake_policy_factory,
                    environment_factories=tuple(
                        observe_factory(factory, created) for factory in factories
                    ),
                )
            except Exception as exc:
                raise AssertionError("paired production routing failed during execution") from exc
            captures[label]["arm_execution_sha256"] = plan.manifest_sha256
            captures[label]["reset_streams"] = {
                environment.environment_index: list(environment.reset_ledger)
                for environment in created
            }
            captures[label]["rsi_stream"] = [dict(item) for item in result.rsi_ledger]
    return captures


def _assert_paired_production_capture(captures: dict[str, dict[str, object]]) -> None:
    baseline = captures["baseline"]
    candidate = captures["candidate"]
    assert baseline["arm_execution_sha256"] != candidate["arm_execution_sha256"]
    for field in ("action_primitives", "minibatch_permutations"):
        left = baseline[field]
        right = candidate[field]
        assert left, f"paired production {field} branch was not observed"
        assert len(left) == len(right)
        assert all(
            np.array_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        ), f"paired production {field} branch was misrouted"
    assert baseline["reset_streams"] == candidate["reset_streams"], (
        "paired production per-slot reset streams were misrouted"
    )
    assert baseline["rsi_stream"] == candidate["rsi_stream"], (
        "paired production RSI streams were misrouted"
    )
    assert set(baseline["reset_streams"]) == {0, 1, 2, 3}
    assert {entry["environment_index"] for entry in baseline["rsi_stream"]} == {2, 3}


def test_paired_fake_arms_traverse_the_production_ppo_and_runtime_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_paired_production_capture(_paired_production_capture(monkeypatch))


def test_paired_production_regression_detects_a_disabled_routing_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with monkeypatch.context() as disabled:
        disabled.setattr(training_module, "_paired_routing_enabled", lambda _plan: False)
        with pytest.raises(AssertionError, match="paired production"):
            _assert_paired_production_capture(_paired_production_capture(disabled))


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
