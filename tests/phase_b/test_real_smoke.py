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
