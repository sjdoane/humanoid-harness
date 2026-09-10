from __future__ import annotations

import io
import zipfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

from oracle_composition.adapters.gmt.actor import load_actor
from oracle_composition.adapters.gmt.contracts import NORMALIZER_EPSILON, OBSERVATION_DIM
from oracle_composition.adapters.gmt.course_config import ACTOR_SHA256
from oracle_composition.adapters.gmt.course_run import _numeric_policy, make_policy
from oracle_composition.adapters.gmt.training_contract import (
    TRAINING_REWARD_SCALE,
    CourseTrainerSpec,
)
from oracle_composition.adapters.gmt.training_features import (
    FixedActorNormalizerExtractor,
)
from oracle_composition.adapters.gmt.training_normalizer import (
    FIXED_NORMALIZER_MEAN_SHA256,
    FIXED_NORMALIZER_STATE_SHA256,
    FIXED_NORMALIZER_STD_SHA256,
    FixedNormalizerState,
    fixed_normalizer_contract,
    fixed_normalizer_policy_metadata,
    normalized_base_max_abs,
    normalizer_state_sha256,
    pinned_normalizer_from_actor,
    validate_policy_normalizer_archive,
)


class _NumericEnvironment(gym.Env):
    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (OBSERVATION_DIM + 17,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (23,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}

    def step(self, action):  # pragma: no cover - this factory test never steps
        raise AssertionError("factory test must not execute an environment")


def _actor_path() -> Path:
    path = (
        Path(__file__).parents[2]
        / "artifacts/gmt/2a590de25a1eb08e/numeric/gmt_g1_actor_weights.npz"
    )
    if not path.is_file():
        pytest.skip("exact local GMT numeric actor artifact is unavailable")
    return path


def _encoded(arrays: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def test_fixed_state_rejects_wrong_shape_nonfinite_nonpositive_and_digest() -> None:
    mean = np.zeros(OBSERVATION_DIM, dtype="<f4")
    std = np.ones(OBSERVATION_DIM, dtype="<f4")
    digest = normalizer_state_sha256(mean, std)

    FixedNormalizerState.from_arrays(mean, std, expected_sha256=digest)
    for changed_mean, changed_std, match in (
        (mean[:-1], std, "mean"),
        (mean.astype("<f8"), std, "mean"),
        (np.full_like(mean, np.nan), std, "mean"),
        (mean, np.zeros_like(std), "standard deviation"),
        (mean, np.full_like(std, np.inf), "standard deviation"),
    ):
        with pytest.raises(ValueError, match=match):
            FixedNormalizerState.from_arrays(
                changed_mean, changed_std, expected_sha256=digest
            )
    with pytest.raises(ValueError, match="exact identity"):
        FixedNormalizerState.from_arrays(mean, std, expected_sha256="0" * 64)


def test_exact_actor_buffers_and_transform_match_pinned_numeric_artifact() -> None:
    actor = load_actor(_actor_path(), expected_sha256=ACTOR_SHA256)
    state = pinned_normalizer_from_actor(actor)
    contract = fixed_normalizer_contract()

    assert state.sha256 == FIXED_NORMALIZER_STATE_SHA256
    assert contract["buffers"]["normalizer_mean"]["sha256"] == (
        FIXED_NORMALIZER_MEAN_SHA256
    )
    assert contract["buffers"]["normalizer_std"]["sha256"] == (
        FIXED_NORMALIZER_STD_SHA256
    )
    observation = np.linspace(
        -2.0, 2.0, OBSERVATION_DIM + 17, dtype=np.float32
    )[None, :]
    extractor = FixedActorNormalizerExtractor(
        _NumericEnvironment().observation_space,
        normalizer_mean=state.mean,
        normalizer_std=state.standard_deviation,
        normalizer_sha256=state.sha256,
    )
    observed = extractor(torch.from_numpy(observation)).detach().numpy()
    expected_base = (
        torch.from_numpy(observation[:, :OBSERVATION_DIM]) - actor.normalizer_mean
    ) / (actor.normalizer_std + NORMALIZER_EPSILON)

    assert torch.equal(torch.from_numpy(observed[:, :OBSERVATION_DIM]), expected_base)
    np.testing.assert_array_equal(observed[:, OBSERVATION_DIM:], observation[:, OBSERVATION_DIM:])
    assert normalized_base_max_abs(observation, state) == float(
        torch.max(torch.abs(expected_base))
    )


def test_policy_factory_uses_same_extractor_and_retains_exact_zero_initial_mean(
    tmp_path: Path,
) -> None:
    torch.set_num_threads(1)
    actor = load_actor(_actor_path(), expected_sha256=ACTOR_SHA256)
    state = pinned_normalizer_from_actor(actor)
    trainer = CourseTrainerSpec(TRAINING_REWARD_SCALE, profile_version=3)
    env = _NumericEnvironment()
    model = make_policy(env, 37, trainer, fixed_normalizer=state)

    assert isinstance(model.policy.features_extractor, FixedActorNormalizerExtractor)
    assert sum(
        parameter.numel() for parameter in model.policy.features_extractor.parameters()
    ) == 0
    assert model.policy.pi_features_extractor is model.policy.features_extractor
    assert model.policy.vf_features_extractor is model.policy.features_extractor
    observation = np.linspace(-1.0, 1.0, OBSERVATION_DIM + 17, dtype=np.float32)
    action, _ = model.predict(observation, deterministic=True)
    assert np.isfinite(action).all()
    np.testing.assert_array_equal(action, np.zeros(23, dtype=np.float32))
    values, log_prob, entropy = model.policy.evaluate_actions(
        torch.from_numpy(observation[None, :]),
        torch.zeros((1, 23), dtype=torch.float32),
    )
    assert all(torch.isfinite(value).all() for value in (values, log_prob, entropy))

    path = tmp_path / "policy.npz"
    _numeric_policy(model, path, fixed_normalizer=state)
    assert validate_policy_normalizer_archive(path.read_bytes()) == (
        fixed_normalizer_policy_metadata()
    )


def test_v2_and_v3_same_seed_trainable_tensors_are_initially_byte_equal() -> None:
    torch.set_num_threads(1)
    actor = load_actor(_actor_path(), expected_sha256=ACTOR_SHA256)
    state = pinned_normalizer_from_actor(actor)
    v2 = make_policy(
        _NumericEnvironment(),
        41,
        CourseTrainerSpec(TRAINING_REWARD_SCALE, profile_version=2),
    )
    v3 = make_policy(
        _NumericEnvironment(),
        41,
        CourseTrainerSpec(TRAINING_REWARD_SCALE, profile_version=3),
        fixed_normalizer=state,
    )

    v2_parameters = dict(v2.policy.named_parameters())
    v3_parameters = dict(v3.policy.named_parameters())
    assert set(v2_parameters) == set(v3_parameters)
    for name, parameter in v2_parameters.items():
        assert torch.equal(parameter, v3_parameters[name]), name


def test_policy_factory_requires_exact_profile_and_buffer_pairing() -> None:
    actor = load_actor(_actor_path(), expected_sha256=ACTOR_SHA256)
    state = pinned_normalizer_from_actor(actor)
    with pytest.raises(ValueError, match="supplied together"):
        make_policy(
            _NumericEnvironment(),
            43,
            CourseTrainerSpec(TRAINING_REWARD_SCALE, profile_version=3),
        )
    with pytest.raises(ValueError, match="supplied together"):
        make_policy(
            _NumericEnvironment(),
            43,
            CourseTrainerSpec(TRAINING_REWARD_SCALE, profile_version=2),
            fixed_normalizer=state,
        )


def test_numeric_policy_rejects_missing_tampered_and_extra_normalizer_buffers() -> None:
    actor = load_actor(_actor_path(), expected_sha256=ACTOR_SHA256)
    state = pinned_normalizer_from_actor(actor)
    arrays = {
        f"{prefix}.{suffix}": (
            state.mean.copy() if suffix == "normalizer_mean" else state.standard_deviation.copy()
        )
        for prefix in ("features_extractor", "pi_features_extractor", "vf_features_extractor")
        for suffix in ("normalizer_mean", "normalizer_std")
    }
    validate_policy_normalizer_archive(_encoded(arrays))

    missing = dict(arrays)
    missing.pop("vf_features_extractor.normalizer_std")
    with pytest.raises(ValueError, match="malformed"):
        validate_policy_normalizer_archive(_encoded(missing))
    tampered = {name: value.copy() for name, value in arrays.items()}
    tampered["features_extractor.normalizer_mean"][0] += np.float32(1.0)
    with pytest.raises(ValueError, match="malformed"):
        validate_policy_normalizer_archive(_encoded(tampered))
    extra = dict(arrays)
    extra["other.normalizer_mean"] = state.mean.copy()
    extra["other.normalizer_std"] = state.standard_deviation.copy()
    with pytest.raises(ValueError, match="malformed"):
        validate_policy_normalizer_archive(_encoded(extra))


def test_numeric_policy_rejects_small_compressed_archive_with_oversized_member() -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("oversized.npy", b"\0" * (16 * 1024**2 + 1))
    assert len(stream.getvalue()) < 64 * 1024

    with pytest.raises(ValueError, match="malformed"):
        validate_policy_normalizer_archive(stream.getvalue())
