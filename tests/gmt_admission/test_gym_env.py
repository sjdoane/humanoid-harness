from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

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
from oracle_composition.adapters.gmt.course_task import CourseTaskSpec, TaskRewardRecipe
from oracle_composition.adapters.gmt.gym_env import (
    RESIDUAL_OBSERVATION_DIM,
    GMTResidualEnv,
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


def _env(
    positions: list[float],
    *,
    task: CourseTaskSpec | None = None,
    contact_substep: int | None = None,
    fall_substep: int | None = None,
) -> tuple[GMTResidualEnv, _FakePlant, _FakeActorSession]:
    plant = _FakePlant(
        positions,
        contact_substep=contact_substep,
        fall_substep=fall_substep,
    )
    actor = _FakeActorSession()
    env = GMTResidualEnv(
        plant=plant,
        actor=actor,
        oracle=_oracle(),
        task=task or _task(),
        recipe=TaskRewardRecipe(1.0, 1.0, 1.0, 1.0, 1.0),
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
