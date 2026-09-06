from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.phase_b import training as training_module
from oracle_composition.phase_b.policy import FullAuthorityPolicy, load_full_authority_actor
from oracle_composition.phase_b.runtime import (
    fake_environment_factories,
    fake_policy_factory,
    worker_step_zero_e1_audit,
)
from oracle_composition.phase_b.training import (
    COHORT_SEEDS,
    FULL_TRANSITIONS_PER_SEED,
    REFERENCE_ONLY_ROLLOUTS,
    SMOKE_SEED,
    SMOKE_TRANSITIONS,
    BalancedRSIScheduler,
    PhaseBTrainingError,
    PPORecipe,
    RSIRestorationReceipt,
    TrainingPlan,
    audit_rollout_likelihood,
    compute_truncation_aware_gae,
    domain_separated_seed,
    run_ppo_training,
    validate_rsi_restoration_receipt,
)

ROOT = Path(__file__).resolve().parents[2]
PHASE_B = ROOT / "experiments/003_composition_speed_profile/phase_b"
MANIFEST_SHA256 = "a" * 64


def _plan(*, transitions: int = 128, seed: int = 11) -> TrainingPlan:
    return TrainingPlan(
        seed=seed,
        transitions=transitions,
        manifest_sha256=MANIFEST_SHA256,
        evidence_class="interface_check",
        promotable=False,
        smoke=False,
        steps_per_environment=4,
        recipe=PPORecipe(batch_size=16, n_epochs=1),
        test_only=True,
    )


def test_frozen_production_plan_has_exact_ppo_budget_and_recipe() -> None:
    plan = TrainingPlan(
        seed=COHORT_SEEDS[0],
        transitions=FULL_TRANSITIONS_PER_SEED,
        manifest_sha256=MANIFEST_SHA256,
        evidence_class="exploratory_fine_tuning_cycle",
        promotable=True,
        smoke=False,
    )
    assert plan.rollout_count == 128
    assert plan.transitions_per_rollout == 8_192
    assert plan.recipe == PPORecipe()

    smoke = TrainingPlan(
        seed=SMOKE_SEED,
        transitions=SMOKE_TRANSITIONS,
        manifest_sha256=MANIFEST_SHA256,
        evidence_class="interface_check",
        promotable=False,
        smoke=True,
    )
    assert smoke.rollout_count == 24


def test_first_eight_rollouts_change_only_reference_columns_and_value() -> None:
    plan = _plan()
    initial = fake_policy_factory(plan)
    initial_state = initial.actor.latent_0.weight.detach().numpy()[:, :348].copy()
    initial_reference = initial.actor.latent_0.weight.detach().numpy()[:, 348:].copy()
    initial_value = [parameter.detach().numpy().copy() for parameter in initial.value.parameters()]

    result = run_ppo_training(
        plan=plan,
        policy_factory=fake_policy_factory,
        environment_factories=fake_environment_factories(plan=plan),
    )

    trained_state = result.policy.actor.latent_0.weight.detach().numpy()[:, :348]
    trained_reference = result.policy.actor.latent_0.weight.detach().numpy()[:, 348:]
    assert np.array_equal(initial_state, trained_state)
    assert not np.array_equal(initial_reference, trained_reference)
    assert any(
        not np.array_equal(before, after.detach().numpy())
        for before, after in zip(initial_value, result.policy.value.parameters(), strict=True)
    )
    receipts = result.scientific_facts["unfreeze_rollouts"]
    assert len(receipts) == REFERENCE_ONLY_ROLLOUTS
    assert {receipt["actor_stage"] for receipt in receipts} == {"reference_columns_only"}
    assert all(receipt["state_columns_changed"] is False for receipt in receipts)
    initialization = result.scientific_facts["optimizer_initialization"]
    assert initialization["state_empty"] is True
    assert initialization["parameter_membership_unique"] is True
    assert initialization["membership_complete"] is True
    expected_membership = initialization["membership_sha256"]
    assert all(
        receipt[stage]["membership_sha256"] == expected_membership
        and receipt[stage]["parameter_membership_unique"] is True
        and receipt[stage]["membership_complete"] is True
        for receipt in receipts
        for stage in ("optimizer_authority_before", "optimizer_authority_after")
    )


def test_worker_counts_streams_exactly_half_and_orders_global_rsi_resets() -> None:
    plan = _plan(transitions=144)
    result = run_ppo_training(
        plan=plan,
        policy_factory=fake_policy_factory,
        environment_factories=fake_environment_factories(plan=plan),
    )
    assert result.scientific_facts["stream_counts"] == {
        "composition": 72,
        "rehearsal": 72,
    }
    assert [entry["global_episode_index"] for entry in result.rsi_ledger] == list(
        range(len(result.rsi_ledger))
    )
    assert result.scientific_facts["likelihood_audit"]["all_passed"] is True
    assert result.scientific_facts["unfreeze_rollouts"][-1]["actor_stage"] == "full_actor"
    final_receipt = result.scientific_facts["unfreeze_rollouts"][-1]
    assert final_receipt["state_columns_changed"] is True
    assert (
        final_receipt["optimizer_authority_before"]["membership_sha256"]
        == (final_receipt["optimizer_authority_after"]["membership_sha256"])
    )


def test_time_limit_transitions_use_value_bootstrap_without_changing_step_count() -> None:
    plan = _plan(transitions=16)
    result = run_ppo_training(
        plan=plan,
        policy_factory=fake_policy_factory,
        environment_factories=fake_environment_factories(plan=plan, episode_steps=2),
    )
    assert result.scientific_facts["observed_transitions"] == 16
    assert result.scientific_facts["time_limit_bootstrap_count"] == 8
    assert result.scientific_facts["stream_counts"] == {
        "composition": 8,
        "rehearsal": 8,
    }


def test_rollout_likelihood_audit_rejects_old_log_prob_only_mutation() -> None:
    observations = np.zeros((1, 2, 708), dtype="<f4")
    pre_tanh = np.zeros((1, 2, 17), dtype="<f4")
    means = np.zeros_like(pre_tanh)
    log_stds = np.zeros_like(pre_tanh)
    from oracle_composition.phase_b.policy import numpy_log_likelihood

    old = numpy_log_likelihood(
        pre_tanh.reshape(2, 17), means.reshape(2, 17), log_stds.reshape(2, 17)
    )
    rollout = training_module._Rollout(
        observations=observations,
        pre_tanh=pre_tanh,
        old_log_prob=np.ascontiguousarray(old.reshape(1, 2), dtype="<f4"),
        rollout_mean=means,
        rollout_log_std=log_stds,
        rewards=np.zeros((1, 2), dtype="<f4"),
        dones=np.zeros((1, 2), dtype=np.bool_),
        values=np.zeros((1, 2), dtype="<f4"),
        advantages=np.zeros((1, 2), dtype="<f4"),
        returns=np.zeros((1, 2), dtype="<f4"),
    )
    receipt = audit_rollout_likelihood(rollout, rollout_index=0).to_dict()
    assert receipt["passed"] is True
    assert receipt["audit_stage"] == "before_any_update_for_rollout"
    mutated = replace(rollout, old_log_prob=rollout.old_log_prob.copy())
    mutated.old_log_prob[0, 0] += np.float32(0.1)
    with pytest.raises(PhaseBTrainingError, match="likelihood audit"):
        audit_rollout_likelihood(mutated, rollout_index=0)


def test_truncation_gae_uses_terminal_value_only_and_never_reset_value() -> None:
    adjusted, advantages, returns = compute_truncation_aware_gae(
        rewards=np.asarray([[1.0, 1.0, 1.0]], dtype="<f4"),
        dones=np.asarray([[True, True, True]], dtype=np.bool_),
        truncations=np.asarray([[True, False, False]], dtype=np.bool_),
        terminal_values=np.asarray([[5.0, 9.0, 0.0]], dtype="<f4"),
        values=np.asarray([[2.0, 2.0, 2.0]], dtype="<f4"),
        last_values=np.asarray([99.0, 99.0, 99.0], dtype="<f4"),
        gamma=0.9,
        gae_lambda=0.95,
    )
    assert adjusted[0].tolist() == pytest.approx([5.5, 1.0, 1.0])
    assert advantages[0].tolist() == pytest.approx([3.5, -1.0, -1.0])
    assert returns[0].tolist() == pytest.approx([5.5, 1.0, 1.0])


def test_scheduler_balances_cells_classes_and_rng_domains() -> None:
    scheduler = BalancedRSIScheduler(manifest_sha256=MANIFEST_SHA256, ppo_seed=11)
    audit = scheduler.audit_prefix(81)
    assert audit["maximum_cell_count_delta"] == 0
    seeds = {
        domain_separated_seed(MANIFEST_SHA256, 11, domain, index)
        for index, domain in enumerate(("actions", "minibatches", "scheduler", "environment"))
    }
    assert len(seeds) == 4


def _valid_rsi_receipt() -> RSIRestorationReceipt:
    return RSIRestorationReceipt(
        start_boundary=8,
        predecessor_boundary=7,
        predecessor_executed=True,
        predecessor_counted=False,
        counted_transitions_before=12,
        counted_transitions_after=12,
        wrapper_elapsed_after=8,
        observation_sha256="1" * 64,
        integration_state_sha256="2" * 64,
        rng_state_sha256="3" * 64,
    )


@pytest.mark.parametrize(
    ("receipt", "match"),
    [
        (
            replace(
                _valid_rsi_receipt(),
                predecessor_boundary=None,
                predecessor_executed=False,
            ),
            "direct boundary restore",
        ),
        (replace(_valid_rsi_receipt(), predecessor_counted=True), "must not be counted"),
        (replace(_valid_rsi_receipt(), wrapper_elapsed_after=0), "counter was reset"),
    ],
)
def test_rsi_shortcuts_are_refused(receipt: RSIRestorationReceipt, match: str) -> None:
    with pytest.raises(PhaseBTrainingError, match=match):
        validate_rsi_restoration_receipt(receipt)


def test_real_expert_step_zero_uses_the_worker_action_path_on_ft1_fixtures() -> None:
    actor = load_full_authority_actor(
        ROOT / "artifacts/experiments_003/phase_b/step_0_full_authority_actor_v1.npz",
        expected_sha256=("6ebc2b56be9a5f304b8b584157fd0141d449d75297366213e4976291cb2dcfe0"),
    )
    audit = worker_step_zero_e1_audit(
        FullAuthorityPolicy(actor.actor, value_seed=20260905),
        repository_root=ROOT,
        receipt_path=PHASE_B / "receipts/e1_full_authority_warm_start_v1.json",
        expected_receipt_sha256=(
            "5754db8e8afdc7f05f8a67ab3fb6a69ae453968a6c14e9f1dca545d688834bc9"
        ),
    )
    assert audit == {
        "action_sha256": "5a70c79209d0830ef63d8d5cef59c53f6bf49f7b9c60cadf6de4e9e0b1ceb568",
        "bitwise_equal": True,
        "e1_receipt_sha256": ("5754db8e8afdc7f05f8a67ab3fb6a69ae453968a6c14e9f1dca545d688834bc9"),
        "fixture_count": 68,
        "reported_beside_checkpoint": True,
        "worker_path": (
            "compose_policy_input_then_full_authority_actor_then_exact_physical_action"
        ),
    }

    with pytest.raises(ExperimentContractError, match="receipt identity"):
        worker_step_zero_e1_audit(
            FullAuthorityPolicy(actor.actor, value_seed=20260905),
            repository_root=ROOT,
            receipt_path=PHASE_B / "receipts/e1_full_authority_warm_start_v1.json",
            expected_receipt_sha256="0" * 64,
        )
