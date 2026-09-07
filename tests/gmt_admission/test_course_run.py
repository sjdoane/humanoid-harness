from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from oracle_composition.adapters.gmt import course_run as module
from oracle_composition.adapters.gmt.contracts import OBSERVATION_DIM, PROPRIOCEPTION_DIM
from oracle_composition.adapters.gmt.control_runtime import PreparedControl
from oracle_composition.adapters.gmt.course_run import (
    _numeric_policy,
    _rollout,
    _TrainingProgress,
    make_policy,
)
from oracle_composition.adapters.gmt.io import sha256_file
from oracle_composition.adapters.gmt.training_contract import (
    BASE_LEARNING_RATE,
    LOW_LEARNING_RATE,
    TRAINING_REWARD_SCALE,
    CourseTrainerSpec,
)
from oracle_composition.adapters.gmt.training_telemetry import (
    TELEMETRY_FILENAME,
    TrainingTelemetry,
)


class NumericFixture(gym.Env):
    """No robot or dynamics: exercise the actual PPO adapter with bounded numbers."""

    def __init__(self):
        self.observation_space = gym.spaces.Box(-10.0, 10.0, (2171,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (23,), dtype=np.float32)
        self.tick = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.tick = 0
        return np.zeros(2171, dtype=np.float32), {}

    def step(self, action):
        assert self.action_space.contains(action)
        self.tick += 1
        obs = np.full(2171, self.tick / 100, dtype=np.float32)
        horizon = self.tick == 16
        metrics = {
            "control_step": self.tick,
            "progress_m": self.tick / 100,
            "lateral_error_m": 0.0,
            "heading_error_rad": 0.0,
            "root_height_m": 0.7,
            "forward_speed_m_s": 0.5,
            "speed_error_m_s": 0.0,
            "posture_band_error_m": 0.0,
            "joint_position_rmse_rad": 0.0,
            "root_height_abs_error_m": 0.0,
            "roll_pitch_rmse_rad": 0.0,
            "inside_posture_region": False,
            "posture_success": False,
            "finish_condition_met": False,
            "horizon_reached": horizon,
            "fallen": False,
        }
        return (
            obs,
            float(1 - np.square(action).mean()),
            False,
            horizon,
            {"metrics": metrics},
        )


def test_zero_initial_mean_is_exact_and_seeded_state_is_repeatable(tmp_path):
    torch.set_num_threads(1)
    first = make_policy(NumericFixture(), 19)
    observation = np.linspace(-1, 1, 2171, dtype=np.float32)
    action, _ = first.predict(observation, deterministic=True)
    np.testing.assert_array_equal(action, np.zeros(23, dtype=np.float32))
    second = make_policy(NumericFixture(), 19)
    for name, tensor in first.policy.state_dict().items():
        assert torch.equal(tensor, second.policy.state_dict()[name])
    path = tmp_path / "policy.npz"
    digest = _numeric_policy(first, path)
    assert sha256_file(path) == digest
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == set(first.policy.state_dict())
        for name, tensor in first.policy.state_dict().items():
            assert archive[name].tobytes() == tensor.numpy().tobytes()


def test_real_ppo_adapter_completes_exact_fixture_budget_and_logs_transitions(capsys, tmp_path):
    torch.set_num_threads(1)
    model = make_policy(NumericFixture(), 29)
    before = model.policy.action_net.weight.detach().clone()
    with TrainingTelemetry(tmp_path / TELEMETRY_FILENAME) as telemetry:
        callback = _TrainingProgress(telemetry)
        model.learn(total_timesteps=512, callback=callback)
        descriptor = telemetry.descriptor(512)
    assert model.num_timesteps == 512
    assert callback.episodes == 32 and callback.falls == 0
    assert not torch.equal(before, model.policy.action_net.weight)
    assert descriptor["rollout_boundary_count"] == 1
    assert '"transitions": 512' in capsys.readouterr().out


def test_low_rate_profile_changes_optimizer_rate_not_initial_policy() -> None:
    torch.set_num_threads(1)
    v1 = CourseTrainerSpec(TRAINING_REWARD_SCALE)
    v2 = CourseTrainerSpec(TRAINING_REWARD_SCALE, profile_version=2)

    raw = make_policy(NumericFixture(), 31)
    control = make_policy(NumericFixture(), 31, v1)
    low_rate = make_policy(NumericFixture(), 31, v2)

    assert raw.learning_rate == BASE_LEARNING_RATE
    assert control.learning_rate == BASE_LEARNING_RATE
    assert low_rate.learning_rate == LOW_LEARNING_RATE
    assert control.lr_schedule(1.0) == BASE_LEARNING_RATE
    assert low_rate.lr_schedule(1.0) == LOW_LEARNING_RATE
    assert control.policy.optimizer.param_groups[0]["lr"] == BASE_LEARNING_RATE
    assert low_rate.policy.optimizer.param_groups[0]["lr"] == LOW_LEARNING_RATE
    for name, tensor in control.policy.state_dict().items():
        assert torch.equal(tensor, raw.policy.state_dict()[name])
        assert torch.equal(tensor, low_rate.policy.state_dict()[name])


class _ZeroRolloutFixture:
    def __init__(self) -> None:
        self._boundary = SimpleNamespace(
            qpos=np.zeros(30, dtype=np.float64),
            qvel=np.zeros(29, dtype=np.float64),
        )
        self.actions: list[np.ndarray] = []

    def reset(self, *, seed: int):
        assert seed == 17
        return np.zeros(4, dtype=np.float32), {"seed": seed}

    def step(self, action: np.ndarray):
        self.actions.append(action.copy())
        step = len(self.actions)
        qpos = np.zeros(30, dtype=np.float64)
        qpos[0] = step * 0.01
        qvel = np.zeros(29, dtype=np.float64)
        self._boundary = SimpleNamespace(qpos=qpos, qvel=qvel)
        return (
            np.full(4, step, dtype=np.float32),
            1.0,
            False,
            step == 2,
            {
                "trajectory": {
                    "qpos": qpos.tolist(),
                    "qvel": qvel.tolist(),
                    "current_reference": np.zeros(30, dtype=np.float32).tolist(),
                    "composite_raw_action": np.zeros(23, dtype=np.float32).tolist(),
                }
            },
        )


class _ObservedZeroRolloutFixture(_ZeroRolloutFixture):
    def reset(self, *, seed: int):
        result = super().reset(seed=seed)
        self._prepared = self._prepared_at(0)
        return result

    def _prepared_at(self, step: int) -> PreparedControl:
        return PreparedControl(
            control_step=step,
            proprio=np.full(PROPRIOCEPTION_DIM, step, dtype="<f8"),
            obs=np.full(OBSERVATION_DIM, step, dtype="<f4"),
            base_raw=np.full(23, step, dtype="<f4"),
        )

    def step(self, action: np.ndarray):
        result = super().step(action)
        self._prepared = self._prepared_at(len(self.actions))
        return result


def test_zero_residual_artifacts_are_exact_across_trainer_profiles(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(module, "evaluate_episode", lambda **_: {"fixture": True})
    task = SimpleNamespace(horizon_steps=2)
    outputs = []
    for version in (None, 1, 2):
        output = tmp_path / ("raw" if version is None else f"v{version}")
        output.mkdir()
        config = SimpleNamespace(
            raw={"seed": 17},
            task=task,
            trainer=(
                None
                if version is None
                else CourseTrainerSpec(TRAINING_REWARD_SCALE, profile_version=version)
            ),
        )
        env = _ZeroRolloutFixture()
        report, artifacts = _rollout(config, env, None, output, "zero_residual")
        assert report["residual_rms"] == 0.0
        assert all(np.count_nonzero(action) == 0 for action in env.actions)
        outputs.append({name: (output / name).read_bytes() for name in artifacts})

    assert outputs[0] == outputs[1] == outputs[2]


def test_rollout_observer_sees_pre_step_input_without_changing_standard_artifacts(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(module, "evaluate_episode", lambda **_: {"fixture": True})
    config = SimpleNamespace(raw={"seed": 17}, task=SimpleNamespace(horizon_steps=2))
    baseline_dir = tmp_path / "baseline"
    observed_dir = tmp_path / "observed"
    baseline_dir.mkdir()
    observed_dir.mkdir()
    _, baseline = _rollout(
        config, _ObservedZeroRolloutFixture(), None, baseline_dir, "zero_residual"
    )
    seen: list[tuple[int, bytes]] = []

    def observe(prepared: PreparedControl, action: np.ndarray) -> None:
        seen.append((prepared.control_step, action.tobytes()))

    _, observed = _rollout(
        config,
        _ObservedZeroRolloutFixture(),
        None,
        observed_dir,
        "zero_residual",
        step_observer=observe,
    )

    assert seen == [
        (0, np.zeros(23, dtype="<f4").tobytes()),
        (1, np.zeros(23, dtype="<f4").tobytes()),
    ]
    assert {name: (baseline_dir / name).read_bytes() for name in baseline} == {
        name: (observed_dir / name).read_bytes() for name in observed
    }
