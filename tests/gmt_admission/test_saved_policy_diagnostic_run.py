from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from oracle_composition.adapters.gmt import saved_policy_diagnostic_run as diagnostic
from oracle_composition.adapters.gmt.actor import load_actor
from oracle_composition.adapters.gmt.course_config import ACTOR_SHA256
from oracle_composition.adapters.gmt.course_run import make_policy
from oracle_composition.adapters.gmt.course_runtime import FOUR_STATE_FINITE_HORIZON_RUNTIME
from oracle_composition.adapters.gmt.io import write_deterministic_npz
from oracle_composition.adapters.gmt.saved_policy_diagnostic_config import (
    RETAINED_DETERMINISTIC_OUTPUTS,
    FileBinding,
    _validate_parity,
)
from oracle_composition.adapters.gmt.training_contract import (
    TRAINING_REWARD_SCALE,
    CourseTrainerSpec,
)
from oracle_composition.adapters.gmt.training_normalizer import pinned_normalizer_from_actor


class NumericEnvironment(gym.Env):
    def __init__(self, width=8):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (width,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (23,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        raise AssertionError("numeric policy tests must not reset or step a simulator")

    def step(self, action):
        raise AssertionError("numeric policy tests must not reset or step a simulator")


@pytest.fixture
def model():
    torch.set_num_threads(1)
    model = PPO("MlpPolicy", NumericEnvironment(), device="cpu", seed=37, n_steps=8, batch_size=8)
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.copy_(torch.linspace(-1.4, 1.4, 23))
        model.policy.log_std.copy_(torch.linspace(-1.5, -0.5, 23))
    return model


def test_noise_matches_rowwise_native_draws_and_does_not_change_global_rng():
    before = torch.get_rng_state().clone()
    noise = diagnostic.paired_noise()
    assert torch.equal(torch.get_rng_state(), before)
    assert noise.shape == (16, 1000, 23)
    assert noise.dtype == np.dtype("<f4")
    for index, seed in enumerate(diagnostic.NOISE_SEEDS):
        generator = torch.Generator().manual_seed(seed)
        expected = torch.stack(
            [
                torch.empty((1, 23), dtype=torch.float32).normal_(generator=generator)[0]
                for _ in range(1000)
            ]
        ).numpy()
        np.testing.assert_array_equal(noise[index], expected)
    bulk = (
        torch.empty((1000, 23))
        .normal_(generator=torch.Generator().manual_seed(diagnostic.NOISE_SEEDS[0]))
        .numpy()
    )
    assert not np.array_equal(noise[0], bulk)


def test_samples_match_sb3_native_normal_then_box_clipping_and_keep_policy_state(model):
    observation = np.linspace(-1, 1, 8, dtype=np.float32)
    noise = diagnostic.paired_noise()[0]
    sampler = diagnostic.RecordedNoisePredictor(model, noise)
    before = diagnostic.policy_state_digest(model)
    torch.manual_seed(diagnostic.NOISE_SEEDS[0])
    clipped, maximum = 0, 0.0
    expected_raw = []
    for _ in range(10):
        with torch.no_grad():
            tensor, _ = model.policy.obs_to_tensor(observation)
            raw = model.policy.get_distribution(tensor).sample().numpy()[0]
        expected = np.clip(raw, -1.0, 1.0)
        observed, _ = sampler.predict(observation, deterministic=True)
        np.testing.assert_array_equal(observed, expected)
        clipped += int(np.count_nonzero(raw != expected))
        expected_raw.append(raw)
        maximum = max(maximum, float(np.max(np.abs(raw))))
    assert clipped > 0
    assert diagnostic.policy_state_digest(model) == before
    evidence = sampler.sampled_action_evidence()
    np.testing.assert_array_equal(evidence["unclipped_action"], np.asarray(expected_raw))
    np.testing.assert_array_equal(
        evidence["unclipped_action"],
        evidence["policy_mean"] + noise[:10] * evidence["policy_std"],
    )
    assert sampler.sampling_receipt(diagnostic.NOISE_SEEDS[0]) == {
        "noise_seed": diagnostic.NOISE_SEEDS[0],
        "noise_rows_available": 1000,
        "noise_rows_consumed": 10,
        "action_dimension": 23,
        "clipped_component_count": clipped,
        "total_component_count": 230,
        "max_abs_unclipped_action": maximum,
    }


def test_noise_is_added_before_clipping_and_uses_each_policies_own_std(model):
    observation = np.zeros(8, dtype=np.float32)
    noise = np.full((1000, 23), -2.0, dtype=np.float32)
    sampler = diagnostic.RecordedNoisePredictor(model, noise)
    action, _ = sampler.predict(observation)
    with torch.no_grad():
        mean = model.policy.action_net.bias.numpy()
        std = model.policy.log_std.exp().numpy()
    np.testing.assert_array_equal(action, np.clip(mean + noise[0] * std, -1, 1))
    assert not np.array_equal(action, np.clip(np.clip(mean, -1, 1) + noise[0] * std, -1, 1))
    with torch.no_grad():
        model.policy.log_std.add_(0.3)
    changed, _ = diagnostic.RecordedNoisePredictor(model, noise).predict(observation)
    assert not np.array_equal(changed, action)


def test_zero_noise_matches_installed_deterministic_prediction(model, monkeypatch):
    observation = np.zeros(8, dtype=np.float32)
    expected, _ = model.predict(observation, deterministic=True)
    predictor = diagnostic.RecordedNoisePredictor(model, None)
    np.testing.assert_array_equal(predictor.predict(observation)[0], expected)
    monkeypatch.setattr(model, "predict", lambda *a, **k: (np.zeros(23, dtype=np.float32), None))
    with pytest.raises(ValueError, match="zero-noise action"):
        predictor.predict(observation)


@pytest.mark.parametrize(
    "noise",
    [
        np.zeros((1, 23), dtype=np.float32),
        np.zeros((1000, 23), dtype=np.float64),
        np.full((1000, 23), np.nan, dtype=np.float32),
    ],
)
def test_sampler_rejects_incompatible_noise(model, noise):
    with pytest.raises(ValueError, match="sampling requires"):
        diagnostic.RecordedNoisePredictor(model, noise)


def test_sampler_rejects_exhaustion_and_vectorized_or_wrong_action_contract(model):
    predictor = diagnostic.RecordedNoisePredictor(model, None)
    with pytest.raises(ValueError, match="unbatched"):
        predictor.predict(np.zeros((1, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="fixed schedule"):
        predictor.predict(np.zeros(8, dtype=np.float32), deterministic=False)
    predictor.index = 1000
    with pytest.raises(ValueError, match="fixed schedule"):
        predictor.predict(np.zeros(8, dtype=np.float32))
    model.action_space = gym.spaces.Box(-2.0, 2.0, (23,), dtype=np.float32)
    with pytest.raises(ValueError, match="Box action path"):
        diagnostic.RecordedNoisePredictor(model, None)


@pytest.fixture
def fixed_policy():
    actor_path = (
        Path(__file__).parents[2]
        / "artifacts/gmt/2a590de25a1eb08e/numeric/gmt_g1_actor_weights.npz"
    )
    if not actor_path.is_file():
        pytest.skip("exact local numeric actor artifact unavailable")
    actor = load_actor(actor_path, expected_sha256=ACTOR_SHA256)
    return make_policy(
        NumericEnvironment(FOUR_STATE_FINITE_HORIZON_RUNTIME.observation_dim),
        37,
        CourseTrainerSpec(TRAINING_REWARD_SCALE, profile_version=3),
        fixed_normalizer=pinned_normalizer_from_actor(actor),
    )


def archive_binding(path, arrays):
    digest = write_deterministic_npz(path, arrays)
    return FileBinding(path, digest, path.stat().st_size)


def test_strict_numeric_load_retains_own_full_state_and_buffers(fixed_policy, tmp_path):
    arrays = {
        name: value.detach().numpy().copy()
        for name, value in fixed_policy.policy.state_dict().items()
    }
    arrays["log_std"] = np.linspace(-1.51, -1.49, 23, dtype=np.float32)
    arrays["action_net.bias"] = np.linspace(-0.1, 0.1, 23, dtype=np.float32)
    binding = archive_binding(tmp_path / "policy.npz", arrays)
    receipt = diagnostic.strict_load_policy(fixed_policy, binding)
    schema = [[name, arrays[name].dtype.str, list(arrays[name].shape)] for name in sorted(arrays)]
    assert receipt["archive"] == binding.receipt()
    assert receipt["state_dict_key_count"] == 19
    assert (
        receipt["state_dict_schema_sha256"]
        == hashlib.sha256(json.dumps(schema, separators=(",", ":")).encode()).hexdigest()
    )
    assert receipt["log_std_sha256"] == hashlib.sha256(arrays["log_std"].tobytes()).hexdigest()
    for name, value in fixed_policy.policy.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), arrays[name])
    assert not fixed_policy.policy.training
    assert all(not value.requires_grad for value in fixed_policy.policy.parameters())


@pytest.mark.parametrize(
    "corruption", ["keys", "shape", "dtype", "nonfinite", "normalizer", "hash"]
)
def test_strict_load_rejects_changed_contract(fixed_policy, tmp_path, corruption):
    arrays = {
        name: value.detach().numpy().copy()
        for name, value in fixed_policy.policy.state_dict().items()
    }
    if corruption == "keys":
        arrays.pop("action_net.bias")
    elif corruption == "shape":
        arrays["action_net.bias"] = arrays["action_net.bias"][:-1]
    elif corruption == "dtype":
        arrays["log_std"] = arrays["log_std"].astype(np.float64)
    elif corruption == "nonfinite":
        arrays["log_std"][0] = np.nan
    elif corruption == "normalizer":
        arrays["features_extractor.normalizer_mean"][0] += 1.0
    binding = archive_binding(tmp_path / "policy.npz", arrays)
    if corruption == "hash":
        binding = FileBinding(binding.path, "0" * 64, binding.size)
    with pytest.raises(ValueError):
        diagnostic.strict_load_policy(fixed_policy, binding)


def test_evaluation_worker_has_no_training_call_or_optimizer_resume():
    tree = ast.parse(inspect.getsource(diagnostic))
    forbidden = {"learn", "backward", "run_course", "torch.load", "PPO.load"}
    names = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    assert not any(name in forbidden or name.rsplit(".", 1)[-1] in forbidden for name in names)


def test_canonical_noise_rejects_rehashed_changed_value(tmp_path):
    noise = diagnostic.paired_noise()
    original = archive_binding(
        tmp_path / "noise.npz",
        {
            "seeds": np.asarray(diagnostic.NOISE_SEEDS, dtype="<i8"),
            "standard_normal": noise,
        },
    )
    np.testing.assert_array_equal(diagnostic.verified_canonical_noise(original), noise)
    noise[-1, -1, -1] += np.float32(0.1)
    changed = archive_binding(
        tmp_path / "changed.npz",
        {
            "seeds": np.asarray(diagnostic.NOISE_SEEDS, dtype="<i8"),
            "standard_normal": noise,
        },
    )
    with pytest.raises(ValueError, match="declared rowwise Torch generator"):
        diagnostic.verified_canonical_noise(changed)


def test_runtime_parity_receipts_match_validator_and_failure_stops(tmp_path):
    ledger, receipts = {}, {}
    for policy, hashes in RETAINED_DETERMINISTIC_OUTPUTS.items():
        receipt = diagnostic.write_parity_receipt(
            tmp_path, policy, f"{policy}_deterministic", hashes, hashes
        )
        receipts[policy] = receipt
        ledger[receipt["path"]] = receipt["sha256"]
        for kind, extension in (("frames", "jsonl"), ("trajectory", "npz"), ("evaluation", "json")):
            ledger[f"{policy}_deterministic_{kind}.{extension}"] = hashes[kind]
    _validate_parity(
        manifest_value={"sampled_started_after_both_receipts": True, "receipts": receipts},
        outputs=ledger,
        output_root=tmp_path,
    )
    failed = tmp_path / "failed"
    with pytest.raises(ValueError, match="no sampled evaluation admitted"):
        diagnostic.write_parity_receipt(
            failed, "initial", "initial_deterministic", {"frames": "a" * 64}, {"frames": "b" * 64}
        )
    assert (
        json.loads((failed / "initial_deterministic_parity.json").read_bytes())["passed"] is False
    )
