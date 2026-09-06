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
from oracle_composition.phase_b.policy import FullAuthorityActor
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
    *,
    sink_mutation: str | None = None,
    disable_pairing: bool = False,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    baseline, candidate = _plans()
    plans = {"baseline": baseline, "candidate": candidate}
    labels = {
        baseline.manifest_sha256: "baseline",
        candidate.manifest_sha256: "candidate",
    }
    captures: dict[str, dict[str, object]] = {
        label: {
            "action_epsilon": [],
            "composition_schedule": {0: [], 1: []},
            "execution_error": None,
            "helper_calls": {"action": 0, "minibatch": 0},
            "ppo_update_inputs": [],
            "reset_routes": {0: [], 1: [], 2: [], 3: []},
            "rollout_inputs": [],
            "rsi_sequence": {2: [], 3: []},
        }
        for label in labels.values()
    }
    original_action_noise = training_module.paired_action_noise
    original_minibatch_permutation = training_module.paired_minibatch_permutation
    original_actor_act = FullAuthorityActor.act
    original_ppo_update = training_module._ppo_update
    original_torch_from_numpy = training_module.torch.from_numpy
    actor_labels: dict[int, str] = {}
    active_update: list[str] = []

    def count_action_noise(
        plan: TrainingPlan,
        *,
        rollout_index: int,
        steps_per_environment: int,
    ) -> np.ndarray:
        label = labels[plan.manifest_sha256]
        captures[label]["helper_calls"]["action"] += 1
        return original_action_noise(
            plan,
            rollout_index=rollout_index,
            steps_per_environment=steps_per_environment,
        )

    def count_minibatch_permutation(
        plan: TrainingPlan,
        *,
        update_index: int,
        sample_count: int,
    ) -> np.ndarray:
        label = labels[plan.manifest_sha256]
        captures[label]["helper_calls"]["minibatch"] += 1
        return original_minibatch_permutation(
            plan,
            update_index=update_index,
            sample_count=sample_count,
        )

    def observe_actor_act(
        actor: FullAuthorityActor,
        policy_input: object,
        *,
        epsilon: np.ndarray | None = None,
    ) -> object:
        if epsilon is None or id(actor) not in actor_labels:
            return original_actor_act(actor, policy_input, epsilon=epsilon)
        label = actor_labels[id(actor)]
        consumed = np.ascontiguousarray(epsilon, dtype="<f4")
        if sink_mutation == "action":
            consumed = np.ascontiguousarray(np.roll(consumed, 1, axis=0), dtype="<f4")
        captures[label]["action_epsilon"].append(consumed.copy())
        return original_actor_act(actor, policy_input, epsilon=consumed)

    def observe_torch_from_numpy(value: np.ndarray) -> object:
        if not active_update:
            return original_torch_from_numpy(value)
        consumed = value
        if sink_mutation == "minibatch":
            consumed = np.ascontiguousarray(np.roll(value, 1, axis=0), dtype=value.dtype)
        captures[active_update[-1]]["ppo_update_inputs"].append(
            np.array(consumed, order="C", copy=True)
        )
        return original_torch_from_numpy(consumed)

    def observe_ppo_update(**kwargs: object) -> object:
        plan = kwargs["plan"]
        rollout = kwargs["rollout"]
        assert isinstance(plan, TrainingPlan)
        label = labels[plan.manifest_sha256]
        captures[label]["rollout_inputs"].append(
            {
                field: getattr(rollout, field).copy()
                for field in ("observations", "pre_tanh", "old_log_prob", "advantages", "returns")
            }
        )
        active_update.append(label)
        try:
            return original_ppo_update(**kwargs)
        finally:
            assert active_update.pop() == label

    def observe_policy_factory(plan: TrainingPlan) -> object:
        policy = fake_policy_factory(plan)
        actor_labels[id(policy.actor)] = labels[plan.manifest_sha256]
        return policy

    class ObservedAssignments:
        def __init__(
            self,
            assignments: runtime_module._SharedResetAssignments,
            *,
            environment_index: int,
            label: str,
        ) -> None:
            self._assignments = assignments
            self._environment_index = environment_index
            self._label = label

        def composition_block(self, environment_index: int) -> int:
            assert environment_index == self._environment_index
            block = self._assignments.composition_block(environment_index)
            routed_index = environment_index
            if sink_mutation == "composition":
                blocks = training_module.TRAINING_BLOCKS
                block = blocks[(blocks.index(block) + 1) % len(blocks)]
            elif sink_mutation == "reset":
                routed_index = 1 - environment_index
                block = self._assignments.scheduler.composition_block(
                    global_episode_index=len(
                        captures[self._label]["composition_schedule"][environment_index]
                    ),
                    environment_index=routed_index,
                )
            captures[self._label]["reset_routes"][environment_index].append(routed_index)
            captures[self._label]["composition_schedule"][environment_index].append(block)
            return block

        def rehearsal(self, environment_index: int) -> object:
            assert environment_index == self._environment_index
            assignment = self._assignments.rehearsal(environment_index)
            routed_index = environment_index
            if sink_mutation == "rsi":
                routed_index = 5 - environment_index
                assignment = self._assignments.scheduler.assignment(
                    global_episode_index=assignment.global_episode_index,
                    environment_index=routed_index,
                )
            captures[self._label]["reset_routes"][environment_index].append(routed_index)
            captures[self._label]["rsi_sequence"][environment_index].append(assignment.to_dict())
            return assignment

    def observe_factory(
        factory: Callable[[], object], environments: list[object], *, label: str
    ) -> Callable[[], object]:
        def build() -> object:
            environment = factory()
            environments.append(environment)
            environment.assignments = ObservedAssignments(
                environment.assignments,
                environment_index=environment.environment_index,
                label=label,
            )
            return environment

        return build

    with monkeypatch.context() as observer:
        observer.setattr(training_module, "paired_action_noise", count_action_noise)
        observer.setattr(
            training_module,
            "paired_minibatch_permutation",
            count_minibatch_permutation,
        )
        observer.setattr(FullAuthorityActor, "act", observe_actor_act)
        observer.setattr(training_module, "_ppo_update", observe_ppo_update)
        observer.setattr(training_module.torch, "from_numpy", observe_torch_from_numpy)
        if disable_pairing:
            observer.setattr(training_module, "_paired_routing_enabled", lambda _plan: False)
        for label, plan in (("baseline", baseline), ("candidate", candidate)):
            created: list[object] = []
            factories = fake_environment_factories(plan=plan, episode_steps=2)
            try:
                run_ppo_training(
                    plan=plan,
                    policy_factory=observe_policy_factory,
                    environment_factories=tuple(
                        observe_factory(factory, created, label=label) for factory in factories
                    ),
                )
            except Exception as exc:
                captures[label]["execution_error"] = exc
            captures[label]["arm_execution_sha256"] = plan.manifest_sha256
            captures[label]["reset_streams"] = {
                environment.environment_index: list(environment.reset_ledger)
                for environment in created
            }
    expected: dict[str, dict[str, object]] = {}
    for label, plan in plans.items():
        if plan.study_pairing is None:
            raise AssertionError("paired test plan lost its exact pairing authority")
        action_epsilon = []
        for rollout_index in range(plan.rollout_count):
            noise = paired_action_noise(
                plan,
                rollout_index=rollout_index,
                steps_per_environment=plan.steps_per_environment,
            )
            action_epsilon.extend(item.copy() for item in noise)
        ppo_update_inputs = []
        for rollout_index, rollout_inputs in enumerate(captures[label]["rollout_inputs"]):
            flattened = {
                "observations": rollout_inputs["observations"].reshape(
                    plan.transitions_per_rollout,
                    training_module.POLICY_INPUT_WIDTH,
                ),
                "pre_tanh": rollout_inputs["pre_tanh"].reshape(
                    plan.transitions_per_rollout,
                    training_module.ACTION_WIDTH,
                ),
                "old_log_prob": rollout_inputs["old_log_prob"].reshape(
                    plan.transitions_per_rollout
                ),
                "advantages": rollout_inputs["advantages"].reshape(plan.transitions_per_rollout),
                "returns": rollout_inputs["returns"].reshape(plan.transitions_per_rollout),
            }
            for epoch_index in range(plan.recipe.n_epochs):
                ordering = paired_minibatch_permutation(
                    plan,
                    update_index=rollout_index * plan.recipe.n_epochs + epoch_index,
                    sample_count=plan.transitions_per_rollout,
                )
                for start in range(0, plan.transitions_per_rollout, plan.recipe.batch_size):
                    indices = ordering[start : start + plan.recipe.batch_size]
                    selected = {
                        field: np.ascontiguousarray(values[indices], dtype="<f4")
                        for field, values in flattened.items()
                    }
                    ppo_update_inputs.extend(
                        (
                            selected["observations"],
                            selected["pre_tanh"],
                            selected["observations"],
                            selected["pre_tanh"],
                            selected["old_log_prob"],
                            selected["advantages"],
                            selected["returns"],
                            selected["observations"],
                        )
                    )
        scheduler = BalancedRSIScheduler(
            manifest_sha256=plan.manifest_sha256,
            ppo_seed=plan.seed,
            study_pairing=plan.study_pairing,
        )
        reset_count = 1 + (plan.transitions // plan.n_envs // 2)
        composition_schedule = {
            environment_index: [
                scheduler.composition_block(
                    global_episode_index=episode_index,
                    environment_index=environment_index,
                )
                for episode_index in range(reset_count)
            ]
            for environment_index in (0, 1)
        }
        rsi_sequence = {
            environment_index: [
                scheduler.assignment(
                    global_episode_index=episode_index,
                    environment_index=environment_index,
                ).to_dict()
                for episode_index in range(reset_count)
            ]
            for environment_index in (2, 3)
        }
        reset_streams = {
            0: [
                {
                    "block": block,
                    "environment_index": 0,
                    "global_episode_index": episode_index,
                    "stream": "composition",
                }
                for episode_index, block in enumerate(composition_schedule[0])
            ],
            1: [
                {
                    "block": block,
                    "environment_index": 1,
                    "global_episode_index": episode_index,
                    "stream": "composition",
                }
                for episode_index, block in enumerate(composition_schedule[1])
            ],
            2: [{**assignment, "stream": "rehearsal"} for assignment in rsi_sequence[2]],
            3: [{**assignment, "stream": "rehearsal"} for assignment in rsi_sequence[3]],
        }
        expected[label] = {
            "action_epsilon": action_epsilon,
            "composition_schedule": composition_schedule,
            "ppo_update_inputs": ppo_update_inputs,
            "reset_routes": {
                environment_index: [environment_index] * reset_count
                for environment_index in range(4)
            },
            "reset_streams": reset_streams,
            "rsi_sequence": rsi_sequence,
        }
    return captures, expected


def _array_sequences_equal(left: object, right: object) -> bool:
    return (
        isinstance(left, list)
        and isinstance(right, list)
        and len(left) == len(right)
        and all(
            np.array_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    )


def _assert_paired_production_capture(
    captures: dict[str, dict[str, object]],
    expected: dict[str, dict[str, object]],
    *,
    route: str | None = None,
) -> None:
    baseline = captures["baseline"]
    candidate = captures["candidate"]
    assert baseline["arm_execution_sha256"] != candidate["arm_execution_sha256"]
    routes = (
        ("action", "action_epsilon", True),
        ("minibatch", "ppo_update_inputs", True),
        ("composition", "composition_schedule", False),
        ("reset", "reset_streams", False),
        ("rsi", "rsi_sequence", False),
    )
    selected = [item for item in routes if route is None or item[0] == route]
    assert selected, f"unknown paired production route: {route}"
    for route_name, field, array_sequence in selected:
        for label in ("baseline", "candidate"):
            actual_value = captures[label][field]
            expected_value = expected[label][field]
            matches = (
                _array_sequences_equal(actual_value, expected_value)
                if array_sequence
                else actual_value == expected_value
            )
            assert matches, f"paired production {route_name} sink was misrouted"
        cross_arm_match = (
            _array_sequences_equal(baseline[field], candidate[field])
            if array_sequence
            else baseline[field] == candidate[field]
        )
        assert cross_arm_match, f"paired production {route_name} arms differ"
    if route in {None, "reset"}:
        for label in ("baseline", "candidate"):
            assert captures[label]["reset_routes"] == expected[label]["reset_routes"], (
                "paired production reset sink was misrouted"
            )
    if route is None:
        assert baseline["execution_error"] is None
        assert candidate["execution_error"] is None


def _assert_complete_sink_capture(
    captures: dict[str, dict[str, object]],
    expected: dict[str, dict[str, object]],
) -> None:
    for label in ("baseline", "candidate"):
        for field in ("action_epsilon", "ppo_update_inputs"):
            assert len(captures[label][field]) == len(expected[label][field])
        for field in ("composition_schedule", "reset_routes", "reset_streams", "rsi_sequence"):
            assert captures[label][field].keys() == expected[label][field].keys()
            for environment_index, values in captures[label][field].items():
                assert len(values) == len(expected[label][field][environment_index])


def test_paired_fake_arms_traverse_the_production_ppo_and_runtime_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures, expected = _paired_production_capture(monkeypatch)
    _assert_paired_production_capture(captures, expected)


@pytest.mark.parametrize(
    ("route", "mutation"),
    [
        ("action", "action"),
        ("minibatch", "minibatch"),
        ("composition", "composition"),
        ("reset", "reset"),
        ("rsi", "rsi"),
        ("action", "disabled"),
        ("minibatch", "disabled"),
        ("composition", "disabled"),
        ("reset", "disabled"),
        ("rsi", "disabled"),
    ],
    ids=lambda value: value,
)
def test_paired_production_regression_detects_each_sink_and_disabled_router_mutation(
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    mutation: str,
) -> None:
    captures, expected = _paired_production_capture(
        monkeypatch,
        sink_mutation=None if mutation == "disabled" else mutation,
        disable_pairing=mutation == "disabled",
    )
    _assert_complete_sink_capture(captures, expected)
    plan, _candidate = _plans()
    for label in ("baseline", "candidate"):
        assert captures[label]["action_epsilon"], "actor.act epsilon sink was not observed"
        assert captures[label]["ppo_update_inputs"], "PPO update input sink was not observed"
        if mutation != "disabled":
            assert captures[label]["helper_calls"] == {
                "action": plan.rollout_count,
                "minibatch": plan.rollout_count * plan.recipe.n_epochs,
            }
    with pytest.raises(AssertionError, match=rf"paired production {route} sink was misrouted"):
        _assert_paired_production_capture(captures, expected, route=route)


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
