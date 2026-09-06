from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.envs.humanoid import make_humanoid_env
from oracle_composition.harness.inputs import load_frozen_inputs
from oracle_composition.phase_b.contracts import RewardRegistry, load_phase_b_oracle
from oracle_composition.phase_b.policy import build_full_authority_policy, compose_policy_input
from oracle_composition.phase_b.reference_runtime import (
    ComposedReferenceRuntime,
    load_v2_reference_clip,
)
from oracle_composition.phase_b.reward import compose_tracking_only_reward
from oracle_composition.phase_b.runtime import (
    RealRuntimeConfig,
    real_environment_factories,
    real_policy_factory,
)
from oracle_composition.phase_b.task_input_admission import (
    MEASUREMENT_ORIGIN,
    admit_task_inputs_v2,
)
from oracle_composition.phase_b.training import PPORecipe, TrainingPlan
from oracle_composition.sources.strict_tqc_actor_runtime import StrictTQCActorRuntime
from oracle_composition.tracking.humanoid_reference import (
    HumanoidTrackingState,
    tracking_state,
    tracking_state_with_bounded_reset_orientation,
    validate_humanoid_actuator_abi,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/003_composition_speed_profile"
EXPERT_PATH = ROOT / "artifacts/bootstrap_tqc_humanoid/farama_minari_humanoid_v5_tqc_actor_v1.npz"
EXPERT_SHA256 = "60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b"
STEP_ZERO_SHA256 = "6ebc2b56be9a5f304b8b584157fd0141d449d75297366213e4976291cb2dcfe0"


def _torso_up(state: HumanoidTrackingState) -> float:
    w, x, y, z = (float(value) for value in state.root_orientation_wxyz)
    return w * w - x * x - y * y + z * z


@pytest.mark.gym
def test_step_zero_actor_runs_200_real_steps_with_composed_window() -> None:
    _library, task = load_frozen_inputs(EXPERIMENT)
    oracle, _oracle_file_sha256 = load_phase_b_oracle(
        EXPERIMENT / "phase_b/oracle_cycle_1_reference_v1.json",
        available_behaviors=("expert", "medium", "simple"),
    )
    clips = {
        behavior: load_v2_reference_clip(
            ROOT / "artifacts/reference_corpus_v2",
            block=120001,
            behavior=behavior,
        )
        for behavior in oracle.program.behaviors
    }
    runtime = ComposedReferenceRuntime(
        oracle,
        {name: clip.reference_rows for name, clip in clips.items()},
    )
    policy = build_full_authority_policy(EXPERT_PATH, value_seed=20260905)
    expert = StrictTQCActorRuntime.from_npz(EXPERT_PATH, expected_sha256=EXPERT_SHA256)
    reward, _reward_sha256 = RewardRegistry().load(EXPERIMENT / "phase_b/tracking_only_v1.json")
    environment = make_humanoid_env()
    try:
        abi = validate_humanoid_actuator_abi(environment)
        observation, _info = environment.reset(seed=97001)
        current_state = tracking_state_with_bounded_reset_orientation(environment, abi)
        initial_x = float(current_state.root_position_world_m[0])
        last_speed = 0.0
        rewards: list[float] = []
        for step in range(200):
            signals = {
                "dwell": runtime.dwell,
                "t": step,
                "torso_up": _torso_up(current_state),
                "v_target": task.target_at(step),
                "v_x": last_speed,
                "x_travelled": float(current_state.root_position_world_m[0]) - initial_x,
                "z_root": current_state.root_height_m,
            }
            frame = runtime.frame(state=current_state, signals=signals)
            state_input = np.ascontiguousarray(observation, dtype="<f4")
            action = policy.actor.act(compose_policy_input(state_input, frame.policy_window))
            expert_action = expert.act(state_input)
            assert np.array_equal(action.normalized, expert_action.normalized_output)
            assert np.array_equal(action.physical, expert_action.physical_action)
            next_observation, stock_reward, terminated, truncated, _info = environment.step(
                action.physical
            )
            assert not terminated
            assert not truncated
            next_state = tracking_state(environment, abi)
            streams = compose_tracking_only_reward(
                state=next_state,
                hidden_reference_target=frame.hidden_reward_target,
                ignored_stock_reward=float(stock_reward),
                task_inputs=admit_task_inputs_v2(
                    com_x_velocity_m_s=0.0,
                    measurement_origin=MEASUREMENT_ORIGIN,
                    cadence_seconds=0.015,
                ),
                specification=reward,
            )
            assert streams.r_train == streams.r_track + streams.r_task
            assert streams.r_train != streams.r_track + streams.ignored_stock_reward
            rewards.append(streams.r_train)
            last_speed = (
                float(next_state.root_position_world_m[0])
                - float(current_state.root_position_world_m[0])
            ) / task.control_period_seconds
            runtime.advance()
            observation = next_observation
            current_state = next_state
        assert runtime.task_step == 200
        assert len(rewards) == 200
        assert all(math.isfinite(value) for value in rewards)
    finally:
        environment.close()


@pytest.mark.gym
def test_rehearsal_adapter_executes_uncounted_predecessor_then_one_counted_step() -> None:
    config = RealRuntimeConfig(
        repository_root=ROOT,
        experiment=EXPERIMENT,
        oracle_path=EXPERIMENT / "phase_b/oracle_cycle_1_reference_v1.json",
        reward_path=EXPERIMENT / "phase_b/tracking_only_v1.json",
        starting_actor_path=(
            ROOT / "artifacts/experiments_003/phase_b/step_0_full_authority_actor_v1.npz"
        ),
        starting_actor_sha256=STEP_ZERO_SHA256,
        value_seed=20260905,
    )
    plan = TrainingPlan(
        seed=11,
        transitions=16,
        manifest_sha256="a" * 64,
        evidence_class="interface_check",
        promotable=False,
        smoke=False,
        steps_per_environment=4,
        recipe=PPORecipe(batch_size=16, n_epochs=1),
        test_only=True,
    )
    environment = real_environment_factories(plan=plan, config=config)[2]()
    policy = real_policy_factory(config)(plan)
    try:
        observation, reset_info = environment.reset()
        action = policy.actor.act(
            compose_policy_input(
                observation[:348].copy(),
                observation[348:].reshape(8, 45).copy(),
            )
        ).physical
        _next, _reward, _terminated, _truncated, info = environment.step(action)
        predecessor = environment.rsi_ledger[0]["predecessor_receipt"]
        assert reset_info["stream"] == "rehearsal"
        assert predecessor["predecessor_executed"] is True
        assert predecessor["predecessor_counted"] is False
        assert predecessor["counted_transitions_before"] == 0
        assert predecessor["counted_transitions_after"] == 0
        assert environment.counted_transitions == 1
        assert info["phase_b"]["counted_transition_delta"] == 1
    finally:
        environment.close()
