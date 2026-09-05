from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from oracle_composition.harness.contract import (
    OracleMachine,
    load_oracle_program,
    oracle_program_from_dict,
)
from oracle_composition.harness.executor import compute_episode_metrics, execute_episode
from oracle_composition.harness.inputs import ScheduleSegment, TaskSpec

REPOSITORY = Path(__file__).resolve().parents[2]


class _FakeActor:
    def __init__(self, speed_m_s: float) -> None:
        self._delta = np.float32(speed_m_s * 0.015)

    def act(self, _observation: np.ndarray) -> SimpleNamespace:
        action = np.zeros(17, dtype="<f4")
        action[0] = self._delta
        return SimpleNamespace(physical_action=action)


class _FakeEnvironment:
    def __init__(self, *, stock_reward: float = 1.0, info_speed: float | None = None) -> None:
        self.unwrapped = self
        self.data = SimpleNamespace(qpos=np.zeros(24, dtype=np.float64))
        self._step = 0
        self._stock_reward = stock_reward
        self._info_speed = info_speed

    def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, object]]:
        self._step = 0
        self.data.qpos[:] = 0.0
        self.data.qpos[0] = seed / 1_000_000.0
        self.data.qpos[2] = 1.4
        self.data.qpos[3] = 1.0
        observation = np.zeros(348, dtype=np.float64)
        observation[0] = seed
        return observation, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        self._step += 1
        delta = float(action[0])
        self.data.qpos[0] += delta
        observation = np.zeros(348, dtype=np.float64)
        observation[0] = self.data.qpos[0]
        observation[1] = self._step
        info_speed = delta / 0.015 if self._info_speed is None else self._info_speed
        return observation, self._stock_reward, False, False, {"x_velocity": info_speed}


def _task() -> TaskSpec:
    return TaskSpec(
        task_text="test",
        horizon_steps=15,
        control_period_seconds=0.015,
        schedule=(
            ScheduleSegment(start=0, stop=5, target_m_s=5.0),
            ScheduleSegment(start=5, stop=10, target_m_s=1.0),
            ScheduleSegment(start=10, stop=15, target_m_s=5.0),
        ),
        seeds=(1, 2),
        cycle_zero_oracle_ids=("single_fast", "single_slow", "playback", "handwritten"),
        trace_artifact_directory="artifacts/test",
        slow_behavior="simple",
        library_manifest_sha256="1" * 64,
        raw_sha256="2" * 64,
    )


def _program():
    return oracle_program_from_dict(
        {
            "behaviors": ["expert"],
            "evidence_class": "exploratory_oracle_cycle",
            "initial": "fast",
            "oracle_id": "single_fast",
            "schema_version": 1,
            "states": {"fast": {"behavior": "expert", "min_dwell": 0}},
            "transitions": [],
        },
        available_behaviors=("expert", "simple"),
    )


def _execute(seed: int):
    return execute_episode(
        environment=_FakeEnvironment(),
        actors={"expert": _FakeActor(5.0)},
        program=_program(),
        task=_task(),
        seed=seed,
        runtime_fingerprint_sha256="3" * 64,
    )


def test_executor_trace_is_identical_for_same_seed_and_differs_for_another_seed() -> None:
    first = _execute(1)
    repeated = _execute(1)
    other_seed = _execute(2)
    assert first.trace_sha256 == repeated.trace_sha256
    assert first.trace_bytes == repeated.trace_bytes
    assert first.trace_sha256 != other_seed.trace_sha256


def test_metrics_do_not_read_the_trace_file(tmp_path: Path) -> None:
    execution = _execute(1)
    trace_path = tmp_path / "trace.json"
    trace_path.write_bytes(execution.trace_bytes)
    before = compute_episode_metrics(
        seed=1,
        initial_boundary=execution.initial_boundary,
        samples=execution.metric_samples,
        behavior_names=("expert",),
    )
    trace_path.write_bytes(b"corrupted after evaluation")
    after = compute_episode_metrics(
        seed=1,
        initial_boundary=execution.initial_boundary,
        samples=execution.metric_samples,
        behavior_names=("expert",),
    )
    assert before == after == execution.metrics


def test_protected_endpoints_ignore_adversarial_reward_and_info_velocity() -> None:
    first = execute_episode(
        environment=_FakeEnvironment(stock_reward=17.0, info_speed=999.0),
        actors={"expert": _FakeActor(5.0)},
        program=_program(),
        task=_task(),
        seed=1,
        runtime_fingerprint_sha256="3" * 64,
    )
    second = execute_episode(
        environment=_FakeEnvironment(stock_reward=-23.0, info_speed=-999.0),
        actors={"expert": _FakeActor(5.0)},
        program=_program(),
        task=_task(),
        seed=1,
        runtime_fingerprint_sha256="3" * 64,
    )
    assert first.metrics.mean_absolute_speed_error_m_s == (
        second.metrics.mean_absolute_speed_error_m_s
    )
    assert first.metrics.fall == second.metrics.fall
    assert first.metrics.first_fall_step == second.metrics.first_fall_step
    assert first.metrics.task_return != second.metrics.task_return


def test_playback_switches_exactly_before_actions_300_and_600() -> None:
    program, _raw_sha256 = load_oracle_program(
        REPOSITORY / "experiments/003_composition_speed_profile/arms/02_playback.json",
        available_behaviors=("expert", "medium", "simple"),
    )
    machine = OracleMachine(program)
    switch_steps: list[int] = []
    for step in range(1000):
        decision = machine.decide(
            {
                "dwell": machine.dwell,
                "t": step,
                "torso_up": 1.0,
                "v_target": 1.0,
                "v_x": 1.0,
                "x_travelled": float(step),
                "z_root": 1.4,
            }
        )
        if decision.controller_switched:
            switch_steps.append(step)
        machine.advance()
    assert switch_steps == [300, 600]
