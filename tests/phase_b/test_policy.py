from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from oracle_composition.experiments.external_tqc_initialization_identity import (
    EXPERT_ACTOR_NPZ_SHA256,
)
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_actor_npz import load_actor_npz
from oracle_composition.phase_b.policy import (
    FullAuthorityActor,
    StrictPolicyInput,
    build_full_authority_policy,
    compose_policy_input,
    exact_physical_action,
    export_full_authority_actor,
    load_full_authority_actor,
    verify_actor_warm_start,
    verify_optimizer_authority,
)

ROOT = Path(__file__).resolve().parents[2]
ACTOR_PATH = ROOT / "artifacts/bootstrap_tqc_humanoid/farama_minari_humanoid_v5_tqc_actor_v1.npz"


def _policy() -> object:
    return build_full_authority_policy(ACTOR_PATH, value_seed=20260905)


def _input() -> object:
    state = np.linspace(-1.0, 1.0, 348, dtype="<f4")
    window = np.linspace(1.0, -1.0, 360, dtype="<f4").reshape(8, 45)
    return compose_policy_input(state, window)


def test_exact_actor_export_reload_and_both_heads(tmp_path: Path) -> None:
    policy = _policy()
    before = policy.actor.act(_input(), epsilon=np.ones(17, dtype="<f4"))
    published = export_full_authority_actor(tmp_path / "actor.npz", policy.actor)
    loaded = load_full_authority_actor(published.path, expected_sha256=published.sha256)
    after = loaded.actor.act(_input(), epsilon=np.ones(17, dtype="<f4"))
    for field in ("mean", "log_std", "normalized", "physical"):
        assert np.array_equal(getattr(before, field), getattr(after, field))
    assert "log_std.weight" in loaded.member_hashes
    assert "mu.weight" in loaded.member_hashes


@pytest.mark.parametrize("name", ["mu.weight", "log_std.weight"])
def test_altered_copied_bit_is_refused(name: str) -> None:
    policy = _policy()
    expected = policy.actor.parameter_arrays()
    module = policy.actor.mu if name == "mu.weight" else policy.actor.log_std
    with torch.no_grad():
        bits = module.weight.detach().view(torch.int32)
        bits[0, 0] ^= 1
    with pytest.raises(ExperimentContractError, match="copied actor parameter"):
        verify_actor_warm_start(policy.actor, expected)


def test_nonzero_reference_weight_and_state_independent_log_std_are_refused() -> None:
    policy = _policy()
    expected = policy.actor.parameter_arrays()
    with torch.no_grad():
        policy.actor.latent_0.weight[0, 348] = torch.tensor(1e-6, dtype=torch.float32)
    with pytest.raises(ExperimentContractError, match="reference columns"):
        verify_actor_warm_start(policy.actor, expected)

    policy = _policy()
    expected = policy.actor.parameter_arrays()
    with torch.no_grad():
        policy.actor.log_std.weight.zero_()
    with pytest.raises(ExperimentContractError, match="copied actor parameter"):
        verify_actor_warm_start(policy.actor, expected)


def test_omitted_clamp_is_refused() -> None:
    policy = _policy()
    expected = policy.actor.parameter_arrays()
    policy.actor.log_std_max = None
    with pytest.raises(ExperimentContractError, match="log-std"):
        verify_actor_warm_start(policy.actor, expected)


def test_reordered_or_float64_inputs_are_refused() -> None:
    policy = _policy()
    with pytest.raises(ExperimentContractError, match="little-endian float32"):
        compose_policy_input(
            np.zeros(348, dtype="<f8"),
            np.zeros((8, 45), dtype="<f4"),
        )
    issued = _input()
    issued.layout_id = "reference_then_state/v1"
    with pytest.raises(ExperimentContractError, match="layout"):
        policy.actor.act(issued)
    with pytest.raises(ExperimentContractError, match="issued"):
        StrictPolicyInput(
            np.zeros(708, dtype="<f4"),
            layout_id="reference_then_state/v1",
        )


def test_normalizer_and_generic_rescale_wrapper_are_refused() -> None:
    with pytest.raises(ExperimentContractError, match="normalizers are forbidden"):
        build_full_authority_policy(
            ACTOR_PATH,
            value_seed=20260905,
            normalizer=object(),
        )
    with pytest.raises(ExperimentContractError, match="generic"):
        exact_physical_action(
            np.zeros(17, dtype="<f4"),
            np.full(17, -0.4, dtype="<f4"),
            np.full(17, 0.4, dtype="<f4"),
            adapter_id="gymnasium.RescaleAction",
        )


def test_optimizer_must_cover_replacement_actor_and_fresh_value_parameters() -> None:
    policy = _policy()
    complete = torch.optim.Adam(policy.parameters(), lr=3e-4)
    verify_optimizer_authority(policy, complete)
    actor_only = torch.optim.Adam(policy.actor.parameters(), lr=3e-4)
    with pytest.raises(ExperimentContractError, match="omits"):
        verify_optimizer_authority(policy, actor_only)


def test_source_hash_and_state_dependent_heads_are_exact() -> None:
    loaded = load_actor_npz(ACTOR_PATH, expected_sha256=EXPERT_ACTOR_NPZ_SHA256)
    actor = FullAuthorityActor.from_expert(loaded)
    assert actor.log_std.weight.shape == (17, 256)
    assert actor.mu.weight.shape == (17, 256)
    assert actor.latent_0.weight.shape == (256, 708)
