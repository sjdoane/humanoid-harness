from __future__ import annotations

import copy
import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from oracle_composition.adapters.gmt.composition import ComposedReference, ReferenceSegment
from oracle_composition.adapters.gmt.contracts import (
    ACTION_DIM,
    DEFAULT_DOF_POSITION,
    OBSERVATION_DIM,
    REFERENCE_FRAME_DIM,
    REFERENCE_HORIZON,
    SIMULATION_DECIMATION,
)
from oracle_composition.adapters.gmt.control_runtime import (
    ControlBoundary,
    ControlInterval,
    PreparedControl,
)
from oracle_composition.adapters.gmt.course_config import CourseRunConfig
from oracle_composition.adapters.gmt.course_runtime import (
    AFTER_HEADING_FEEDBACK_RUNTIME,
    FINITE_HORIZON_RUNTIME,
    LEGACY_RUNTIME,
    LOOP_RUNTIME,
    CourseRuntimeProfile,
)
from oracle_composition.adapters.gmt.course_task import CourseTaskSpec, TaskRewardRecipe
from oracle_composition.adapters.gmt.gym_env import (
    AFTER_HEADING_FEEDBACK_OBSERVATION_DIM,
    FINITE_HORIZON_RESIDUAL_OBSERVATION_DIM,
    LOOP_RESIDUAL_OBSERVATION_DIM,
    RESIDUAL_OBSERVATION_DIM,
    GMTResidualEnv,
)
from oracle_composition.adapters.gmt.heading_feedback import (
    AFTER_HEADING_FEEDBACK_TRACE_KEY,
    YAW_RATE_COLUMN,
    validate_after_heading_feedback_trace,
)
from oracle_composition.adapters.gmt.reference_runtime import ReferenceMotion
from oracle_composition.harness.contract import oracle_program_from_dict


def _qpos(x: float, *, height: float = 0.8) -> np.ndarray:
    result = np.zeros(30, dtype="<f8")
    result[:3] = (x, 0.0, height)
    result[3] = 1.0
    result[-ACTION_DIM:] = np.asarray(DEFAULT_DOF_POSITION, dtype="<f8")
    return result


def _motion(height: float) -> ReferenceMotion:
    root = np.zeros((301, 3), dtype="<f4")
    root[:, 2] = np.float32(height)
    rotation = np.zeros((301, 4), dtype="<f4")
    rotation[:, 3] = 1.0  # ReferenceMotion uses xyzw.
    return ReferenceMotion(
        {
            "fps": np.asarray([30.0], dtype="<f8"),
            "root_pos": root,
            "root_rot": rotation,
            "dof_pos": np.broadcast_to(
                np.asarray(DEFAULT_DOF_POSITION, dtype="<f4"), (301, ACTION_DIM)
            ).copy(),
        }
    )


def _oracle() -> ComposedReference:
    program = oracle_program_from_dict(
        {
            "schema_version": 1,
            "evidence_class": "exploratory_oracle_cycle",
            "oracle_id": "gym_fixture",
            "behaviors": ["walk", "crouch"],
            "initial": "before",
            "states": {
                "before": {"behavior": "walk", "min_dwell": 1},
                "inside": {"behavior": "crouch", "min_dwell": 1},
                "after": {"behavior": "walk", "min_dwell": 1},
            },
            "transitions": [
                {"from": "before", "to": "inside", "priority": 0, "guard": "x_travelled >= 1"},
                {"from": "inside", "to": "after", "priority": 0, "guard": "x_travelled >= 2"},
            ],
        },
        available_behaviors=["walk", "crouch"],
    )
    return ComposedReference(
        program,
        {
            "walk": ReferenceSegment(_motion(0.8), "a" * 64, 0.0, 10.0),
            "crouch": ReferenceSegment(_motion(0.5), "b" * 64, 0.0, 10.0),
        },
    )


def _four_state_oracle() -> ComposedReference:
    behaviors = ["walk_before", "crouch", "rise", "walk_after"]
    program = oracle_program_from_dict(
        {
            "schema_version": 1,
            "evidence_class": "exploratory_oracle_cycle",
            "oracle_id": "four_state_heading_fixture",
            "behaviors": behaviors,
            "initial": "before",
            "states": {
                "before": {"behavior": "walk_before", "min_dwell": 1},
                "inside": {"behavior": "crouch", "min_dwell": 1},
                "rise": {"behavior": "rise", "min_dwell": 1},
                "after": {"behavior": "walk_after", "min_dwell": 1},
            },
            "transitions": [
                {"from": "before", "to": "inside", "priority": 0, "guard": "x_travelled >= 1"},
                {"from": "inside", "to": "rise", "priority": 0, "guard": "x_travelled >= 2"},
                {"from": "rise", "to": "after", "priority": 0, "guard": "x_travelled >= 2"},
            ],
        },
        available_behaviors=behaviors,
    )
    return ComposedReference(
        program,
        {
            "walk_before": ReferenceSegment(_motion(0.8), "a" * 64, 0.0, 10.0),
            "crouch": ReferenceSegment(_motion(0.5), "b" * 64, 0.0, 10.0),
            "rise": ReferenceSegment(_motion(0.7), "c" * 64, 0.0, 10.0),
            "walk_after": ReferenceSegment(_motion(0.8), "d" * 64, 0.0, 10.0),
        },
    )


def _pose(x: float, lateral: float, yaw: float, *, height: float = 0.8) -> np.ndarray:
    result = _qpos(x, height=height)
    result[1] = lateral
    result[3:7] = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    return result


def _heading_env(final_lateral: float = 1.2, *, horizon_steps: int = 10):
    plant = _FakePlant([0.0, 1.1, 2.1, 2.2, 2.3])
    plant.states = [
        _pose(0.0, 0.0, 0.0),
        _pose(1.1, 0.1, 0.02),
        _pose(2.1, 0.4, 0.05),
        _pose(2.2, 1.0, 0.10),
        _pose(2.3, final_lateral, 0.20),
    ]
    actor = _FakeActorSession()
    oracle = _four_state_oracle()
    task = _task(horizon_steps=horizon_steps)
    recipe = TaskRewardRecipe(1.0, 1.0, 1.0, 1.0, 1.0)
    env = GMTResidualEnv(
        plant=plant,
        actor=actor,
        oracle=oracle,
        task=task,
        recipe=recipe,
        record_trajectory=True,
        runtime=AFTER_HEADING_FEEDBACK_RUNTIME,
    )
    config = CourseRunConfig(
        raw={},
        encoded=b"{}",
        sha256="0" * 64,
        assets={},
        task=task,
        recipe=recipe,
        program=oracle.program,
        segments=oracle.segments,
        runtime=AFTER_HEADING_FEEDBACK_RUNTIME,
    )
    return env, plant, actor, config


def _task(*, horizon_steps: int = 10) -> CourseTaskSpec:
    return CourseTaskSpec(
        region_entry_distance_m=1.0,
        region_exit_distance_m=2.0,
        finish_distance_m=3.0,
        target_speed_outside_m_s=1.0,
        target_speed_inside_m_s=0.5,
        posture_band_low_m=0.45,
        posture_band_high_m=0.85,
        horizon_steps=horizon_steps,
    )


class _FakePlant:
    geom_body_names = ("world", "left_ankle_roll_link", "torso_link")

    def __init__(
        self,
        positions: list[float],
        *,
        contact_substep: int | None = None,
        fall_substep: int | None = None,
    ) -> None:
        self.states = [_qpos(position) for position in positions]
        self.contact_substep = contact_substep
        self.fall_substep = fall_substep
        self.index = 0
        self.reset_count = 0
        self.executed_actions: list[np.ndarray] = []

    def _boundary(self) -> ControlBoundary:
        velocity = np.zeros(29, dtype="<f8")
        if self.index:
            velocity[0] = (self.states[self.index][0] - self.states[self.index - 1][0]) / 0.02
        return ControlBoundary(
            qpos=self.states[self.index].copy(),
            qvel=velocity,
            orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype="<f4"),
            angular_velocity=np.zeros(3, dtype="<f4"),
            control_step=self.index,
            simulation_step=self.index * SIMULATION_DECIMATION,
        )

    def reset(self) -> ControlBoundary:
        self.index = 0
        self.reset_count += 1
        return self._boundary()

    def step(self, composite: np.ndarray) -> tuple[ControlBoundary, ControlInterval]:
        before = self.states[self.index]
        after = self.states[self.index + 1]
        qpos = np.stack(
            [
                before + (after - before) * ((substep + 1) / SIMULATION_DECIMATION)
                for substep in range(SIMULATION_DECIMATION)
            ]
        ).astype("<f8")
        if self.fall_substep is not None:
            qpos[self.fall_substep, 2] = 0.2
        contacts = [() for _ in range(SIMULATION_DECIMATION)]
        if self.contact_substep is not None:
            contacts[self.contact_substep] = ((0, 2),)
        self.executed_actions.append(composite.copy())
        start = self.index
        interval = ControlInterval(
            composite_raw_action=composite,
            clipped_action=np.clip(composite, -10.0, 10.0).astype("<f4"),
            pd_target=np.zeros(ACTION_DIM, dtype="<f8"),
            qpos_after_substep=qpos,
            qvel_after_substep=np.zeros((SIMULATION_DECIMATION, 29), dtype="<f8"),
            torque_by_substep=np.zeros((SIMULATION_DECIMATION, ACTION_DIM), dtype="<f8"),
            contact_pairs_after_substep=tuple(contacts),
            start_control_step=start,
            start_simulation_step=start * SIMULATION_DECIMATION,
        )
        self.index += 1
        return self._boundary(), interval


class _FakeActorSession:
    def __init__(self) -> None:
        self.pending: PreparedControl | None = None
        self.reset_count = 0
        self.stale_pending_clears = 0
        self.windows: list[np.ndarray] = []
        self.committed_actions: list[np.ndarray] = []
        self.base_raw = np.linspace(-0.2, 0.2, ACTION_DIM, dtype="<f4")

    def reset(self) -> None:
        if self.pending is not None:
            self.stale_pending_clears += 1
        self.pending = None
        self.reset_count += 1

    def prepare(self, boundary: ControlBoundary, window: np.ndarray) -> PreparedControl:
        if self.pending is not None:
            raise AssertionError("stale pending actor preparation")
        self.windows.append(window.copy())
        obs = np.zeros(OBSERVATION_DIM, dtype="<f4")
        obs[: REFERENCE_HORIZON * REFERENCE_FRAME_DIM] = window.reshape(-1)
        obs[600] = np.float32(boundary.qpos[0])
        obs[601] = np.float32(boundary.qpos[2])
        self.pending = PreparedControl(
            control_step=boundary.control_step,
            proprio=np.zeros(74, dtype="<f8"),
            obs=obs,
            base_raw=self.base_raw,
        )
        return self.pending

    def commit(self, prepared: PreparedControl, composite: np.ndarray) -> None:
        if prepared is not self.pending:
            raise AssertionError("commit did not use pending actor preparation")
        self.committed_actions.append(composite.copy())
        self.pending = None


class _ContinueCallback(BaseCallback):
    def _on_step(self) -> bool:
        return True


def _env(
    positions: list[float],
    *,
    task: CourseTaskSpec | None = None,
    contact_substep: int | None = None,
    fall_substep: int | None = None,
    runtime: CourseRuntimeProfile | None = None,
) -> tuple[GMTResidualEnv, _FakePlant, _FakeActorSession]:
    plant = _FakePlant(
        positions,
        contact_substep=contact_substep,
        fall_substep=fall_substep,
    )
    actor = _FakeActorSession()
    kwargs = {} if runtime is None else {"runtime": runtime}
    env = GMTResidualEnv(
        plant=plant,
        actor=actor,
        oracle=_oracle(),
        task=task or _task(),
        recipe=TaskRewardRecipe(1.0, 1.0, 1.0, 1.0, 1.0),
        **kwargs,
    )
    return env, plant, actor


def _zero_action() -> np.ndarray:
    return np.zeros(ACTION_DIM, dtype="<f4")


def test_fixed_observation_layout_and_one_reference_window_feed_all_modes() -> None:
    env, _plant, actor = _env([0.0, 1.5, 2.5])
    observations = [env.reset()[0]]
    observations.append(env.step(_zero_action())[0])
    observations.append(env.step(_zero_action())[0])

    assert RESIDUAL_OBSERVATION_DIM == 2_171
    assert [observation.shape for observation in observations] == [(2_171,)] * 3
    assert all(observation.dtype == np.dtype("<f4") for observation in observations)
    np.testing.assert_array_equal(observations[0][-4:-1], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(observations[1][-4:-1], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(observations[2][-4:-1], [0.0, 0.0, 1.0])
    for observation, window in zip(observations, actor.windows, strict=True):
        np.testing.assert_array_equal(observation[:600], window.reshape(-1))


def test_loop_profile_adds_fixed_rise_slot_for_three_state_matched_control() -> None:
    env, _plant, _actor = _env([0.0, 1.5, 2.5], runtime=LOOP_RUNTIME)
    observations = [env.reset()[0]]
    observations.append(env.step(_zero_action())[0])
    observations.append(env.step(_zero_action())[0])

    assert LOOP_RESIDUAL_OBSERVATION_DIM == 2_172
    assert [observation.shape for observation in observations] == [(2_172,)] * 3
    np.testing.assert_array_equal(observations[0][-5:-1], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(observations[1][-5:-1], [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(observations[2][-5:-1], [0.0, 0.0, 0.0, 1.0])
    assert all(observation[-3] == 0.0 for observation in observations)


def test_after_heading_profile_issues_actor_window_and_held_target_from_pre_action_state() -> None:
    env, _plant, actor, config = _heading_env()
    initial_qpos = env.plant.states[0].copy()
    initial_qvel = env.plant._boundary().qvel.copy()
    env.reset(seed=7)
    frames = []
    for _ in range(4):
        _observation, _reward, _terminated, _truncated, info = env.step(_zero_action())
        frames.append(info)

    assert AFTER_HEADING_FEEDBACK_OBSERVATION_DIM == 2_172
    assert AFTER_HEADING_FEEDBACK_TRACE_KEY not in frames[0]
    assert AFTER_HEADING_FEEDBACK_TRACE_KEY not in frames[1]
    assert AFTER_HEADING_FEEDBACK_TRACE_KEY not in frames[2]
    trace = frames[3][AFTER_HEADING_FEEDBACK_TRACE_KEY]
    assert trace["pre_action_control_step"] == 3
    assert trace["observed_pre_action"]["lateral_m"] == pytest.approx(1.0)
    assert trace["observed_pre_action"]["heading_error_signed_rad"] == pytest.approx(0.1)
    assert trace["target_heading_rad"] == -0.3
    assert trace["correction_yaw_rate_rad_s"] == pytest.approx(-0.4)
    assert np.all(actor.windows[3][:, YAW_RATE_COLUMN] == np.float32(-0.3))
    np.testing.assert_array_equal(
        frames[3]["trajectory"]["current_reference"], actor.windows[3][0]
    )

    trajectory = {
        "qpos": np.asarray(
            [initial_qpos, *[row["trajectory"]["qpos"] for row in frames]], dtype="<f8"
        ),
        "qvel": np.asarray(
            [initial_qvel, *[row["trajectory"]["qvel"] for row in frames]], dtype="<f8"
        ),
        "current_reference": np.asarray(
            [row["trajectory"]["current_reference"] for row in frames], dtype="<f4"
        ),
    }
    summary = validate_after_heading_feedback_trace(
        config=config,
        frames=frames,
        trajectory=trajectory,
    )
    assert summary["after_actions"] == 1
    assert summary["saturated_window_rows"] == 20

    forged = copy.deepcopy(frames)
    forged[3][AFTER_HEADING_FEEDBACK_TRACE_KEY]["target_heading_rad"] = -0.2
    with pytest.raises(ValueError, match="raw-state reconstruction"):
        validate_after_heading_feedback_trace(
            config=config,
            frames=forged,
            trajectory=trajectory,
        )
    trace_before_after = copy.deepcopy(frames)
    trace_before_after[0][AFTER_HEADING_FEEDBACK_TRACE_KEY] = copy.deepcopy(trace)
    with pytest.raises(ValueError, match="before the after state"):
        validate_after_heading_feedback_trace(
            config=config,
            frames=trace_before_after,
            trajectory=trajectory,
        )
    wrong_reference = {name: value.copy() for name, value in trajectory.items()}
    wrong_reference["current_reference"][3, 0] += np.float32(1.0e-3)
    with pytest.raises(ValueError, match="reconstructed heading feedback"):
        validate_after_heading_feedback_trace(
            config=config,
            frames=frames,
            trajectory=wrong_reference,
        )


def test_heading_rollout_publishes_through_real_evaluator(tmp_path) -> None:
    from oracle_composition.adapters.gmt.course_evaluation import evaluate_episode
    from oracle_composition.adapters.gmt.course_run import _rollout

    env, _plant, _actor, config = _heading_env(horizon_steps=4)
    config = replace(config, raw={"seed": 7})
    report, outputs = _rollout(config, env, None, tmp_path, "zero_residual")
    frames = [
        json.loads(line)
        for line in (tmp_path / "zero_residual_frames.jsonl").read_text().splitlines()
    ]
    with np.load(tmp_path / "zero_residual_trajectory.npz", allow_pickle=False) as archive:
        trajectory = {name: archive[name] for name in archive.files}
    assert report["steps"] == 4
    assert len(outputs) == 3
    assert report["objective_evaluation"] == evaluate_episode(
        spec=config.task, frames=frames, runtime=config.runtime
    )
    assert validate_after_heading_feedback_trace(
        config=config, frames=frames, trajectory=trajectory
    )["after_actions"] == 1
    with pytest.raises(ValueError, match="frame fields"):
        evaluate_episode(spec=config.task, frames=frames)


def test_after_heading_validator_rejects_executed_but_numeric_noop_profile() -> None:
    env, plant, _actor, config = _heading_env()
    plant.states[3] = _pose(2.2, 0.0, 0.0)
    initial_qpos = plant.states[0].copy()
    initial_qvel = plant._boundary().qvel.copy()
    env.reset(seed=7)
    frames = [env.step(_zero_action())[4] for _ in range(4)]
    trajectory = {
        "qpos": np.asarray(
            [initial_qpos, *[row["trajectory"]["qpos"] for row in frames]], dtype="<f8"
        ),
        "qvel": np.asarray(
            [initial_qvel, *[row["trajectory"]["qvel"] for row in frames]], dtype="<f8"
        ),
        "current_reference": np.asarray(
            [row["trajectory"]["current_reference"] for row in frames], dtype="<f4"
        ),
    }

    with pytest.raises(ValueError, match="without a numeric manipulation"):
        validate_after_heading_feedback_trace(
            config=config,
            frames=frames,
            trajectory=trajectory,
        )


def test_after_heading_plan_does_not_depend_on_same_action_future_state() -> None:
    first, _first_plant, first_actor, _config = _heading_env(final_lateral=1.2)
    second, _second_plant, second_actor, _config = _heading_env(final_lateral=-4.0)
    first.reset(seed=7)
    second.reset(seed=7)
    first_info = second_info = None
    for _ in range(4):
        first_info = first.step(_zero_action())[4]
        second_info = second.step(_zero_action())[4]

    np.testing.assert_array_equal(first_actor.windows[3], second_actor.windows[3])
    assert not np.array_equal(first_actor.windows[4], second_actor.windows[4])
    assert (
        first_info[AFTER_HEADING_FEEDBACK_TRACE_KEY]
        == second_info[AFTER_HEADING_FEEDBACK_TRACE_KEY]
    )
    np.testing.assert_array_equal(
        first_info["trajectory"]["current_reference"],
        second_info["trajectory"]["current_reference"],
    )


def test_failed_next_command_cannot_leave_a_stale_heading_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env, _plant, _actor, _config = _heading_env()
    env.reset(seed=7)
    for _ in range(3):
        env.step(_zero_action())
    assert env._heading_feedback_plan is not None

    def reject(**_kwargs):
        raise ValueError("command failure")

    monkeypatch.setattr(env.oracle, "command", reject)
    with pytest.raises(ValueError, match="command failure"):
        env._prepare()
    assert env._heading_feedback_plan is None


def test_heading_profile_is_byte_exact_with_loop_profile_before_after_entry() -> None:
    loop_plant = _FakePlant([0.0, 1.1, 2.1, 2.2])
    heading_plant = _FakePlant([0.0, 1.1, 2.1, 2.2])
    states = [
        _pose(0.0, 0.0, 0.0),
        _pose(1.1, 0.1, 0.02),
        _pose(2.1, 0.4, 0.05),
        _pose(2.2, 1.0, 0.10),
    ]
    loop_plant.states = [state.copy() for state in states]
    heading_plant.states = [state.copy() for state in states]
    loop_actor, heading_actor = _FakeActorSession(), _FakeActorSession()
    task, recipe = _task(), TaskRewardRecipe(1.0, 1.0, 1.0, 1.0, 1.0)
    loop_env = GMTResidualEnv(
        plant=loop_plant,
        actor=loop_actor,
        oracle=_four_state_oracle(),
        task=task,
        recipe=recipe,
        record_trajectory=True,
        runtime=LOOP_RUNTIME,
    )
    heading_env = GMTResidualEnv(
        plant=heading_plant,
        actor=heading_actor,
        oracle=_four_state_oracle(),
        task=task,
        recipe=recipe,
        record_trajectory=True,
        runtime=AFTER_HEADING_FEEDBACK_RUNTIME,
    )
    loop_observation, _ = loop_env.reset(seed=7)
    heading_observation, _ = heading_env.reset(seed=7)
    np.testing.assert_array_equal(loop_observation, heading_observation)
    for index in range(3):
        loop_step = loop_env.step(_zero_action())
        heading_step = heading_env.step(_zero_action())
        if index < 2:
            np.testing.assert_array_equal(loop_step[0], heading_step[0])
        assert loop_step[1:] == heading_step[1:]
        np.testing.assert_array_equal(loop_actor.windows[index], heading_actor.windows[index])


def test_finite_horizon_profile_preserves_legacy_observation_layout() -> None:
    legacy, _legacy_plant, _legacy_actor = _env([0.0, 0.25], runtime=LEGACY_RUNTIME)
    finite, _finite_plant, _finite_actor = _env(
        [0.0, 0.25], runtime=FINITE_HORIZON_RUNTIME
    )

    legacy_observation, _ = legacy.reset(seed=7)
    finite_observation, _ = finite.reset(seed=7)

    assert FINITE_HORIZON_RESIDUAL_OBSERVATION_DIM == RESIDUAL_OBSERVATION_DIM == 2_171
    np.testing.assert_array_equal(finite_observation, legacy_observation)
    assert finite_observation[OBSERVATION_DIM + 10] == np.float32(1.0)


def test_task_region_comes_from_actual_position_not_oracle_mode() -> None:
    env, _plant, _actor = _env([0.0, 1.5, 0.5])
    env.reset()
    env.step(_zero_action())
    observation, _reward, terminated, truncated, info = env.step(_zero_action())

    assert not terminated and not truncated
    assert info["executed_mode"] == "inside"
    assert info["metrics"]["inside_posture_region"] is False
    assert info["metrics"]["target_speed_m_s"] == _task().target_speed_outside_m_s
    assert observation[OBSERVATION_DIM + 6] == _task().target_speed_outside_m_s
    assert observation[OBSERVATION_DIM + 7] == 0.0
    np.testing.assert_array_equal(observation[-4:-1], [0.0, 1.0, 0.0])


def test_reward_uses_executed_pre_switch_reference() -> None:
    env, _plant, _actor = _env([0.0, 1.5])
    env.reset()
    _observation, _reward, _terminated, _truncated, info = env.step(_zero_action())

    assert info["executed_behavior"] == "walk"
    assert env._command.behavior == "crouch"
    assert info["metrics"]["root_height_abs_error_m"] < 1.0e-6
    assert info["metrics"]["root_height_abs_error_m"] != pytest.approx(0.3)


@pytest.mark.parametrize(
    ("plant_kwargs", "reason"),
    [
        ({"contact_substep": 19}, "non_foot_ground_contact"),
        ({"fall_substep": 11}, "substep_root_height_failure"),
    ],
)
def test_any_of_twenty_substeps_can_trigger_terminal_failure(plant_kwargs, reason) -> None:
    env, _plant, _actor = _env([0.0, 0.1], **plant_kwargs)
    env.reset()
    _observation, _reward, terminated, truncated, info = env.step(_zero_action())

    assert terminated is True and truncated is False
    assert reason in info["metrics"]["failure_reasons"]


def test_zero_residual_passes_exact_base_to_plant_and_actor_history() -> None:
    env, plant, actor = _env([0.0, 0.1])
    env.reset()
    env.step(_zero_action())

    assert plant.executed_actions[0].tobytes() == actor.base_raw.tobytes()
    assert actor.committed_actions[0].tobytes() == actor.base_raw.tobytes()


def test_finish_condition_alone_does_not_end_episode() -> None:
    env, _plant, _actor = _env([0.0, 3.1])
    env.reset()
    _observation, _reward, terminated, truncated, info = env.step(_zero_action())

    assert info["metrics"]["finish_condition_met"] is True
    assert terminated is False
    assert truncated is False


def test_truncation_returns_next_state_and_seeded_reset_remains_fixed() -> None:
    env, plant, actor = _env([0.0, 0.25], task=replace(_task(), horizon_steps=1))
    first, _ = env.reset(seed=7)
    stale = actor.pending
    second, _ = env.reset(seed=999)
    np.testing.assert_array_equal(first, second)
    assert plant.reset_count == 2
    assert actor.stale_pending_clears == 1
    assert actor.pending is not stale

    observation, _reward, terminated, truncated, _info = env.step(_zero_action())
    assert terminated is False and truncated is True
    assert observation[600] == np.float32(0.25)
    assert actor.pending is not None and actor.pending.control_step == 1
    terminal_pending = actor.pending
    with pytest.raises(RuntimeError, match="reset is required"):
        env.step(_zero_action())
    env.reset(seed=7)
    assert actor.stale_pending_clears == 2
    assert actor.pending is not terminal_pending and actor.pending.control_step == 0


@pytest.mark.parametrize(
    ("horizon_steps", "fall_substep", "expected"),
    [
        (10, None, (False, False)),
        (10, 11, (True, False)),
        (1, None, (True, False)),
        (1, 11, (True, False)),
    ],
    ids=("ordinary", "fall", "horizon", "fall-and-horizon"),
)
def test_finite_horizon_profile_uses_only_intrinsic_termination(
    horizon_steps: int,
    fall_substep: int | None,
    expected: tuple[bool, bool],
) -> None:
    env, _plant, _actor = _env(
        [0.0, 0.25],
        task=replace(_task(), horizon_steps=horizon_steps),
        fall_substep=fall_substep,
        runtime=FINITE_HORIZON_RUNTIME,
    )
    env.reset(seed=7)

    observation, _reward, terminated, truncated, info = env.step(_zero_action())

    assert (terminated, truncated) == expected
    assert info["metrics"]["horizon_reached"] is (horizon_steps == 1)
    assert bool(info["metrics"]["fallen"]) is (fall_substep is not None)
    assert observation[OBSERVATION_DIM + 10] == np.float32(
        0.0 if horizon_steps == 1 else (horizon_steps - 1) / horizon_steps
    )


def test_legacy_fall_at_timeout_keeps_historical_dual_flags() -> None:
    env, _plant, _actor = _env(
        [0.0, 0.25],
        task=replace(_task(), horizon_steps=1),
        fall_substep=11,
        runtime=LEGACY_RUNTIME,
    )
    env.reset(seed=7)

    _observation, _reward, terminated, truncated, _info = env.step(_zero_action())

    assert terminated is True and truncated is True


def _collect_horizon_rewards(runtime: CourseRuntimeProfile) -> np.ndarray:
    vector_env = DummyVecEnv(
        [
            lambda: _env(
                [0.0, 0.25],
                task=replace(_task(), horizon_steps=1),
                runtime=runtime,
            )[0]
        ]
    )
    model = PPO(
        "MlpPolicy",
        vector_env,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        gamma=0.9,
        seed=11,
        device="cpu",
        verbose=0,
    )
    model.policy.predict_values = lambda observation: torch.full(
        (observation.shape[0],), 2.0, dtype=torch.float32, device=observation.device
    )
    _, callback = model._setup_learn(
        2,
        callback=_ContinueCallback(),
        reset_num_timesteps=True,
        tb_log_name="finite-horizon-collector",
        progress_bar=False,
    )
    callback.on_training_start({}, {})
    assert model.collect_rollouts(vector_env, callback, model.rollout_buffer, n_rollout_steps=2)
    return model.rollout_buffer.rewards[:, 0].copy()


def test_sb3_collector_bootstraps_legacy_timeout_but_not_intrinsic_horizon() -> None:
    legacy_rewards = _collect_horizon_rewards(LEGACY_RUNTIME)
    finite_rewards = _collect_horizon_rewards(FINITE_HORIZON_RUNTIME)

    np.testing.assert_allclose(
        legacy_rewards,
        finite_rewards + np.float32(0.9 * 2.0),
        rtol=0,
        atol=1e-6,
    )
