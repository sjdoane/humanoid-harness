from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
import torch

from oracle_composition.adapters.gmt import training_normalizer as normalizer_module
from oracle_composition.adapters.gmt import training_telemetry as telemetry_module
from oracle_composition.adapters.gmt.course_run import _TrainingProgress, make_policy
from oracle_composition.adapters.gmt.course_runtime import (
    FOUR_STATE_FINITE_HORIZON_RUNTIME,
    LEGACY_RUNTIME,
)
from oracle_composition.adapters.gmt.training_contract import (
    TRAINING_REWARD_SCALE,
    CourseTrainerSpec,
)
from oracle_composition.adapters.gmt.training_normalizer import (
    FixedNormalizerState,
    normalizer_state_sha256,
)
from oracle_composition.adapters.gmt.training_telemetry import (
    FOUR_STATE_FINITE_HORIZON_TELEMETRY_FILENAME,
    FOUR_STATE_FINITE_HORIZON_TELEMETRY_ID,
    TrainingTelemetry,
    observation_group_moments,
    telemetry_filename,
    validate_training_telemetry,
    validate_training_telemetry_descriptor,
)
from oracle_composition.adapters.gmt.training_wrapper import training_env


def _fixed_state(monkeypatch: pytest.MonkeyPatch) -> FixedNormalizerState:
    mean = np.linspace(-1.0, 1.0, 2154, dtype="<f4")
    standard_deviation = np.linspace(0.01, 1.0, 2154, dtype="<f4")
    digest = normalizer_state_sha256(mean, standard_deviation)
    monkeypatch.setattr(normalizer_module, "FIXED_NORMALIZER_STATE_SHA256", digest)
    monkeypatch.setattr(
        normalizer_module,
        "FIXED_NORMALIZER_MEAN_SHA256",
        hashlib.sha256(mean.tobytes()).hexdigest(),
    )
    monkeypatch.setattr(
        normalizer_module,
        "FIXED_NORMALIZER_STD_SHA256",
        hashlib.sha256(standard_deviation.tobytes()).hexdigest(),
    )
    monkeypatch.setattr(telemetry_module, "FIXED_NORMALIZER_STATE_SHA256", digest)
    return FixedNormalizerState.from_arrays(mean, standard_deviation, expected_sha256=digest)


def _observations(state: FixedNormalizerState) -> np.ndarray:
    observations = np.zeros((2, 2172), dtype=np.float32)
    observations[:, :2154] = state.mean + np.float32(2.0) * (
        state.standard_deviation + np.float32(1.0e-4)
    )
    observations[:, 2154:2165] = 1.0e6
    observations[0, 2165:] = (0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.25)
    observations[1, 2165:] = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.75)
    return observations


def _encode(rows: list[dict[str, object]]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows
    )


def test_v4_binds_exact_runtime_tail_and_fixed_prefix_normalizer(tmp_path, monkeypatch) -> None:
    state = _fixed_state(monkeypatch)
    path = tmp_path / FOUR_STATE_FINITE_HORIZON_TELEMETRY_FILENAME
    with TrainingTelemetry(
        path,
        reward_scale=TRAINING_REWARD_SCALE,
        fixed_normalizer=state,
        runtime=FOUR_STATE_FINITE_HORIZON_RUNTIME,
    ) as writer:
        writer.rollout_boundary(512, _observations(state), {}, None)
        writer.final_update({}, None)
        descriptor = writer.descriptor(512)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    runtime_contract = FOUR_STATE_FINITE_HORIZON_RUNTIME.manifest_contract()
    assert rows[0]["observation_groups"]["base"]["width"] == 2154
    assert rows[0]["observation_groups"]["task"]["width"] == 11
    assert rows[0]["observation_groups"]["phase"]["width"] == 7
    assert rows[0]["fixed_normalized_base_range"]["maximum_absolute_value"] == pytest.approx(
        2.0, abs=2.0e-6
    )
    assert rows[0]["course_runtime"] == runtime_contract
    assert rows[1]["course_runtime"] == runtime_contract
    assert descriptor["course_runtime"] == runtime_contract
    assert descriptor["telemetry_id"] == FOUR_STATE_FINITE_HORIZON_TELEMETRY_ID
    assert descriptor["schema_version"] == 3
    validate_training_telemetry_descriptor(
        descriptor,
        path.read_bytes(),
        512,
        reward_scale=TRAINING_REWARD_SCALE,
        fixed_normalizer_sha256=state.sha256,
        runtime=FOUR_STATE_FINITE_HORIZON_RUNTIME,
    )
    with pytest.raises(ValueError, match="identity"):
        validate_training_telemetry(
            path.read_bytes(),
            512,
            reward_scale=TRAINING_REWARD_SCALE,
            fixed_normalizer_sha256=state.sha256,
            runtime=LEGACY_RUNTIME,
        )


@pytest.mark.parametrize("location", ["rollout", "final", "descriptor"])
def test_v4_rejects_changed_runtime_contract(tmp_path, monkeypatch, location: str) -> None:
    state = _fixed_state(monkeypatch)
    path = tmp_path / FOUR_STATE_FINITE_HORIZON_TELEMETRY_FILENAME
    with TrainingTelemetry(
        path,
        reward_scale=TRAINING_REWARD_SCALE,
        fixed_normalizer=state,
        runtime=FOUR_STATE_FINITE_HORIZON_RUNTIME,
    ) as writer:
        writer.rollout_boundary(512, _observations(state), {}, None)
        writer.final_update({}, None)
        descriptor = writer.descriptor(512)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if location == "descriptor":
        descriptor["course_runtime"]["profile_id"] = "changed/v1"
        encoded = path.read_bytes()
    else:
        row = rows[0] if location == "rollout" else rows[-1]
        row["course_runtime"]["profile_id"] = "changed/v1"
        encoded = _encode(rows)

    with pytest.raises(ValueError, match="telemetry"):
        validate_training_telemetry_descriptor(
            descriptor,
            encoded,
            512,
            reward_scale=TRAINING_REWARD_SCALE,
            fixed_normalizer_sha256=state.sha256,
            runtime=FOUR_STATE_FINITE_HORIZON_RUNTIME,
        )


def test_runtime_selects_width_and_identity_instead_of_guessing_from_shape(
    tmp_path, monkeypatch
) -> None:
    state = _fixed_state(monkeypatch)
    malformed = replace(
        FOUR_STATE_FINITE_HORIZON_RUNTIME,
        profile_id="gmt_g1_unadmitted_four_state_course/v1",
    )

    assert (
        observation_group_moments(np.zeros((1, 2171), dtype=np.float32), runtime=LEGACY_RUNTIME)[
            "phase"
        ]["width"]
        == 6
    )
    with pytest.raises(ValueError, match=r"\[\.\.\., 2172\]"):
        observation_group_moments(
            np.zeros((1, 2171), dtype=np.float32),
            runtime=FOUR_STATE_FINITE_HORIZON_RUNTIME,
        )
    with pytest.raises(ValueError, match="not admitted"):
        observation_group_moments(np.zeros((1, 2172), dtype=np.float32), runtime=malformed)
    with pytest.raises(ValueError, match="not admitted"):
        telemetry_filename(
            reward_scale=TRAINING_REWARD_SCALE,
            fixed_normalizer_sha256=state.sha256,
            runtime=malformed,
        )


class _SoftwareOnlyFourStateEnv(gym.Env):
    """Bounded interface fixture; this is not a robot or scientific training run."""

    def __init__(self, state: FixedNormalizerState) -> None:
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (2172,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (23,), dtype=np.float32)
        self.state = state
        self.episode_step = 0

    def _observation(self) -> np.ndarray:
        observation = np.zeros(2172, dtype=np.float32)
        observation[:2154] = self.state.mean
        observation[2154:2165] = np.linspace(-0.5, 0.5, 11, dtype=np.float32)
        slot = min(self.episode_step // 4, 3)
        phase = np.zeros(7, dtype=np.float32)
        phase[:2] = (0.0, 1.0)
        phase[2 + slot] = 1.0
        phase[-1] = np.float32((self.episode_step % 4) / 4)
        observation[2165:] = phase
        return observation

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_step = 0
        return self._observation(), {}

    def step(self, action):
        assert self.action_space.contains(action)
        self.episode_step += 1
        terminated = self.episode_step == 16
        reward = float(np.float32(1.0 - 0.001 * np.square(action).sum()))
        metrics = {
            "control_step": self.episode_step,
            "progress_m": float(self.episode_step) / 16,
            "lateral_error_m": 0.0,
            "heading_error_rad": 0.0,
            "root_height_m": 0.75,
            "forward_speed_m_s": 1.0,
            "speed_error_m_s": 0.0,
            "posture_band_error_m": 0.0,
            "joint_position_rmse_rad": 0.0,
            "root_height_abs_error_m": 0.0,
            "roll_pitch_rmse_rad": 0.0,
            "inside_posture_region": 4 <= self.episode_step < 8,
            "posture_success": True,
            "finish_condition_met": terminated,
            "horizon_reached": terminated,
            "fallen": False,
        }
        return (
            self._observation(),
            reward,
            terminated,
            False,
            {"reward": {"total_reward": reward}, "metrics": metrics},
        )


def test_real_sb3_callback_accepts_first_512_step_v4_rollout(tmp_path, monkeypatch) -> None:
    """Exercise the production callback contract without MuJoCo or robot evidence."""

    torch.set_num_threads(1)
    state = _fixed_state(monkeypatch)
    trainer = CourseTrainerSpec(TRAINING_REWARD_SCALE, profile_version=3)
    env = training_env(_SoftwareOnlyFourStateEnv(state), trainer)
    model = make_policy(env, 47, trainer, fixed_normalizer=state)
    path = tmp_path / FOUR_STATE_FINITE_HORIZON_TELEMETRY_FILENAME
    with TrainingTelemetry(
        path,
        reward_scale=TRAINING_REWARD_SCALE,
        fixed_normalizer=state,
        runtime=FOUR_STATE_FINITE_HORIZON_RUNTIME,
    ) as writer:
        callback = _TrainingProgress(writer)
        model.learn(total_timesteps=512, callback=callback)
        descriptor = writer.descriptor(512)

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert model.num_timesteps == 512
    assert callback.episodes == 32
    assert callback.falls == 0
    assert rows[0]["collected_through_transitions"] == 512
    assert rows[0]["observation_groups"]["phase"]["width"] == 7
    assert rows[0]["episodes"]["completed"] == 32
    assert rows[0]["episodes"]["horizons"] == 32
    assert rows[-1]["update"]["trained_through_transitions"] == 512
    validate_training_telemetry_descriptor(
        descriptor,
        path.read_bytes(),
        512,
        reward_scale=TRAINING_REWARD_SCALE,
        fixed_normalizer_sha256=state.sha256,
        runtime=FOUR_STATE_FINITE_HORIZON_RUNTIME,
    )
