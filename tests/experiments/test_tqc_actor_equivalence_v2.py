from __future__ import annotations

import copy
import io
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from sb3_contrib import TQC
from sb3_contrib.tqc.policies import Actor
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from oracle_composition.experiments import tqc_actor_equivalence_v2 as equivalence_module
from oracle_composition.experiments.artifact_io import (
    PublishedArtifact,
    publish_bytes_without_overwrite,
)
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_actor_equivalence_v2 import (
    EQUIVALENCE_OBSERVATION_SHA256,
    EQUIVALENCE_SAMPLING_SEED,
    admit_final_tqc_checkpoint_actor,
    canonical_array_sha256,
    equivalence_observations,
    revalidate_tqc_actor_equivalence_receipt_v2,
    verify_actor_equivalence_v2,
)
from oracle_composition.experiments.tqc_actor_npz import (
    ACTION_WIDTH,
    OBSERVATION_WIDTH,
    load_actor_npz,
    write_actor_npz_exclusive,
)


class _FakePersistenceAuthority:
    def __init__(
        self,
        model: TQC,
        *,
        model_artifact: PublishedArtifact,
        actor_artifact: PublishedArtifact,
        loaded_actor: object,
    ) -> None:
        self.model = model
        self.attempt_id = "dev1m-v2-seed-95001-attempt-01"
        self.execution_manifest_sha256 = "1" * 64
        self.claimed_work_directory_identity = "2" * 64
        self.design_file_sha256 = "3" * 64
        self.design_semantic_sha256 = "4" * 64
        self.training_projection_sha256 = "5" * 64
        self.training_integrity_sha256 = "6" * 64
        self.persistence_sha256 = "7" * 64
        self.model_artifact = model_artifact
        self.actor_artifact = actor_artifact
        self.loaded_actor = loaded_actor
        self.source_actor_state_sha256 = loaded_actor.state_sha256

    def _validate_payload_seal(self) -> None:
        return None


@pytest.fixture(autouse=True)
def _scoped_persistence_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        equivalence_module,
        "TQCPersistenceAuthorityV2",
        _FakePersistenceAuthority,
    )
    monkeypatch.setattr(
        equivalence_module,
        "revalidate_tqc_persistence_authority_v2",
        lambda authority: authority.model,
    )


def _actor(seed: int) -> Actor:
    observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(OBSERVATION_WIDTH,),
        dtype=np.float64,
    )
    action_space = gym.spaces.Box(
        low=np.full(ACTION_WIDTH, -0.4, dtype=np.float32),
        high=np.full(ACTION_WIDTH, 0.4, dtype=np.float32),
        dtype=np.float32,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        extractor = FlattenExtractor(observation_space)
        actor = Actor(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=[256, 256],
            features_extractor=extractor,
            features_dim=extractor.features_dim,
            activation_fn=torch.nn.ReLU,
            use_sde=False,
        )
    actor.eval()
    return actor


def _arrays(actor: Actor) -> dict[str, np.ndarray]:
    result = {
        name: np.ascontiguousarray(value.detach().cpu().numpy(), dtype="<f4")
        for name, value in actor.state_dict().items()
    }
    result.update(
        {
            "action_low": np.full(ACTION_WIDTH, -0.4, dtype="<f4"),
            "action_high": np.full(ACTION_WIDTH, 0.4, dtype="<f4"),
            "format_version": np.asarray([1], dtype="<i8"),
        }
    )
    return result


def _strict_loaded(actor: Actor, tmp_path: Path):
    path = tmp_path / "actor.npz"
    digest = write_actor_npz_exclusive(path, _arrays(actor))
    return load_actor_npz(path, expected_sha256=digest)


def _model(
    seed: int,
    *,
    initialize_optimizer_state: bool = True,
    critic_arch: list[int] | None = None,
) -> tuple[TQC, DummyVecEnv]:
    environment = DummyVecEnv([lambda: gym.make("Humanoid-v5")])
    net_arch: list[int] | dict[str, list[int]] = [256, 256]
    if critic_arch is not None:
        net_arch = {"pi": [256, 256], "qf": critic_arch}
    model = TQC(
        "MlpPolicy",
        environment,
        buffer_size=100,
        learning_starts=1_000,
        policy_kwargs={
            "net_arch": net_arch,
            "n_quantiles": 25,
            "n_critics": 2,
            "optimizer_kwargs": {"eps": 1e-5},
        },
        seed=seed,
        device="cpu",
        verbose=0,
    )
    model.num_timesteps = 1_000_000
    model._n_updates = 199_980
    if initialize_optimizer_state:
        for optimizer, parameters in (
            (model.actor.optimizer, tuple(model.actor.parameters())),
            (model.critic.optimizer, tuple(model.critic.parameters())),
            (model.ent_coef_optimizer, (model.log_ent_coef,)),
        ):
            for parameter in parameters:
                optimizer.state[parameter] = {
                    "step": torch.tensor(199_980.0, dtype=torch.float32),
                    "exp_avg": torch.zeros_like(parameter),
                    "exp_avg_sq": torch.zeros_like(parameter),
                }
    return model, environment


def _persistence(
    model: TQC,
    tmp_path: Path,
    *,
    loaded_actor: object | None = None,
    model_artifact: PublishedArtifact | None = None,
) -> _FakePersistenceAuthority:
    stream = io.BytesIO()
    model.save(stream)
    if model_artifact is None:
        model_artifact = publish_bytes_without_overwrite(
            tmp_path / "model.zip",
            stream.getvalue(),
        )
    actor_path = tmp_path / "persisted_actor.npz"
    if loaded_actor is None:
        actor_digest = write_actor_npz_exclusive(actor_path, _arrays(model.actor))
        loaded_actor = load_actor_npz(actor_path, expected_sha256=actor_digest)
    actor_artifact = PublishedArtifact(
        path=actor_path,
        sha256=loaded_actor.content_sha256,
        byte_count=loaded_actor.byte_count,
    )
    return _FakePersistenceAuthority(
        model,
        model_artifact=model_artifact,
        actor_artifact=actor_artifact,
        loaded_actor=loaded_actor,
    )


def _authority(model: TQC, tmp_path: Path):
    return admit_final_tqc_checkpoint_actor(_persistence(model, tmp_path))


def test_equivalence_grid_matches_the_frozen_identity() -> None:
    observations = equivalence_observations()

    assert observations.shape == (4, OBSERVATION_WIDTH)
    assert observations.dtype == np.dtype("<f4")
    assert observations.flags.c_contiguous
    assert canonical_array_sha256(observations) == EQUIVALENCE_OBSERVATION_SHA256


def test_v2_equivalence_checks_all_outputs_and_preserves_rng(tmp_path: Path) -> None:
    model, environment = _model(95_001)
    try:
        authority = _authority(model, tmp_path)
        torch.manual_seed(5_123)
        rng_before = torch.random.get_rng_state().clone()

        receipt = verify_actor_equivalence_v2(authority)

        assert receipt.passed is True
        assert receipt.attempt_id == authority.attempt_id
        assert receipt.training_integrity_sha256 == authority.training_integrity_sha256
        assert receipt.sampling_seed == EQUIVALENCE_SAMPLING_SEED
        assert receipt.observation_sha256 == EQUIVALENCE_OBSERVATION_SHA256
        assert receipt.trusted_actor_state_sha256 == receipt.loaded_actor_state_sha256
        assert receipt.trusted_mean_sha256 == receipt.loaded_mean_sha256
        assert receipt.trusted_log_std_sha256 == receipt.loaded_log_std_sha256
        assert (
            receipt.trusted_deterministic_action_sha256
            == receipt.loaded_deterministic_action_sha256
        )
        assert receipt.trusted_seeded_sample_sha256 == receipt.loaded_seeded_sample_sha256
        assert receipt.cpu_rng_state_before_sha256 == receipt.cpu_rng_state_after_sha256
        assert torch.equal(torch.random.get_rng_state(), rng_before)
        assert receipt.to_dict()["passed"] is True
        assert revalidate_tqc_actor_equivalence_receipt_v2(receipt) is receipt
        with pytest.raises(ExperimentContractError, match="only be issued"):
            replace(receipt)
        with pytest.raises(ExperimentContractError, match="only be issued"):
            replace(authority)
    finally:
        environment.close()


def test_v2_equivalence_rejects_actor_state_mismatch(tmp_path: Path) -> None:
    model, environment = _model(95_001)
    loaded = _strict_loaded(_actor(95_002), tmp_path)

    try:
        with pytest.raises(ExperimentContractError, match="actor state binding differs"):
            admit_final_tqc_checkpoint_actor(_persistence(model, tmp_path, loaded_actor=loaded))
    finally:
        environment.close()


def test_v2_equivalence_rejects_actor_changed_after_admission(tmp_path: Path) -> None:
    model, environment = _model(95_001)
    authority = _authority(model, tmp_path)
    model.actor.action_space = gym.spaces.Box(
        low=np.full(ACTION_WIDTH, -1.0, dtype=np.float32),
        high=np.full(ACTION_WIDTH, 1.0, dtype=np.float32),
        dtype=np.float32,
    )

    try:
        with pytest.raises(ExperimentContractError, match="action bounds"):
            verify_actor_equivalence_v2(authority)
    finally:
        environment.close()


def test_v2_equivalence_rejects_cross_attempt_persistence_substitution(
    tmp_path: Path,
) -> None:
    model, environment = _model(95_001)
    other_directory = tmp_path / "other"
    other_directory.mkdir()
    authority = _authority(model, tmp_path)
    other = _persistence(model, other_directory)
    other.attempt_id = "dev1m-v2-seed-95001-attempt-02"
    forged = copy.copy(authority)
    object.__setattr__(forged, "persistence_authority", other)

    try:
        with pytest.raises(ExperimentContractError, match="authority seal differs"):
            verify_actor_equivalence_v2(forged)
    finally:
        environment.close()


def test_v2_equivalence_revalidation_rejects_altered_receipt(tmp_path: Path) -> None:
    model, environment = _model(95_001)
    receipt = verify_actor_equivalence_v2(_authority(model, tmp_path))
    forged = copy.copy(receipt)
    object.__setattr__(forged, "attempt_id", "dev1m-v2-seed-95001-attempt-02")

    try:
        with pytest.raises(ExperimentContractError, match="receipt seal differs"):
            revalidate_tqc_actor_equivalence_receipt_v2(forged)
    finally:
        environment.close()


@pytest.mark.parametrize("sampling_seed", [True, -1, 97_002, 1.5])
def test_v2_equivalence_rejects_any_other_sampling_seed(
    tmp_path: Path,
    sampling_seed: object,
) -> None:
    model, environment = _model(95_001)

    try:
        with pytest.raises(ExperimentContractError, match="sampling seed"):
            verify_actor_equivalence_v2(
                _authority(model, tmp_path),
                sampling_seed=sampling_seed,
            )
    finally:
        environment.close()


def test_checkpoint_admission_rejects_wrong_counters(tmp_path: Path) -> None:
    model, environment = _model(95_001)
    model.num_timesteps -= 1

    try:
        with pytest.raises(ExperimentContractError, match="not at the frozen final checkpoint"):
            _authority(model, tmp_path)
    finally:
        environment.close()


def test_checkpoint_admission_rejects_faked_counters_without_optimizer_state(
    tmp_path: Path,
) -> None:
    model, environment = _model(95_001, initialize_optimizer_state=False)

    try:
        with pytest.raises(ExperimentContractError, match="optimizer ownership"):
            _authority(model, tmp_path)
    finally:
        environment.close()


def test_checkpoint_admission_rejects_wrong_critic_architecture(tmp_path: Path) -> None:
    model, environment = _model(95_001, critic_arch=[128, 128])

    try:
        with pytest.raises(ExperimentContractError, match="policy state differs"):
            _authority(model, tmp_path)
    finally:
        environment.close()


def test_checkpoint_admission_rejects_wrong_adam_group(tmp_path: Path) -> None:
    model, environment = _model(95_001)
    model.actor.optimizer.param_groups[0]["lr"] = 0.123

    try:
        with pytest.raises(ExperimentContractError, match="parameter group differs"):
            _authority(model, tmp_path)
    finally:
        environment.close()


@pytest.mark.parametrize("byte_count", [0, True, 100 * 1024 * 1024 + 1])
def test_checkpoint_admission_rejects_model_archive_byte_bounds(
    tmp_path: Path,
    byte_count: object,
) -> None:
    model, environment = _model(95_001)
    invalid = PublishedArtifact(
        path=tmp_path / "model.zip",
        sha256="2" * 64,
        byte_count=byte_count,
    )

    try:
        with pytest.raises(ExperimentContractError, match="byte count"):
            admit_final_tqc_checkpoint_actor(_persistence(model, tmp_path, model_artifact=invalid))
    finally:
        environment.close()


def test_seeded_sample_restores_cpu_rng_on_failure() -> None:
    torch.manual_seed(5_123)
    before = torch.random.get_rng_state().clone()

    with pytest.raises(TypeError):
        equivalence_module._seeded_sample(
            torch.zeros((4, ACTION_WIDTH), dtype=torch.float32),
            object(),
            sampling_seed=EQUIVALENCE_SAMPLING_SEED,
        )

    assert torch.equal(torch.random.get_rng_state(), before)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_v2_equivalence_does_not_change_mps_rng(tmp_path: Path) -> None:
    model, environment = _model(95_001)
    before = torch.mps.get_rng_state().clone()

    try:
        verify_actor_equivalence_v2(_authority(model, tmp_path))
        assert torch.equal(torch.mps.get_rng_state(), before)
    finally:
        environment.close()


def test_array_hash_rejects_non_c_order() -> None:
    value = np.arange(24, dtype="<f4").reshape(4, 6).T
    assert not value.flags.c_contiguous

    with pytest.raises(ExperimentContractError, match="C-order"):
        canonical_array_sha256(value)


def test_checkpoint_admission_rejects_bare_model(tmp_path: Path) -> None:
    model, environment = _model(95_001)
    try:
        with pytest.raises(ExperimentContractError, match="persistence authority"):
            admit_final_tqc_checkpoint_actor(model)
    finally:
        environment.close()
