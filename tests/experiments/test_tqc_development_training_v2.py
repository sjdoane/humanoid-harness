from __future__ import annotations

import gc
import hashlib
import os
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sb3_contrib import TQC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.vec_env import DummyVecEnv

from oracle_composition.envs.humanoid import HumanoidExperimentConfig, make_humanoid_env
from oracle_composition.experiments import tqc_development_manifest_v2 as manifest_module
from oracle_composition.experiments import tqc_development_resource_v2 as resource_module
from oracle_composition.experiments import tqc_development_training_v2 as training_module
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_development_contract_v2 import (
    EXPECTED_GRADIENT_UPDATES,
    EXPECTED_PARAMETER_CHECKS,
    EXPECTED_VECTOR_STEPS,
    load_tqc_development_design_v2,
)
from oracle_composition.experiments.tqc_development_resource_v2 import TQCResourceMonitorV2
from oracle_composition.experiments.tqc_development_training_v2 import (
    EXPECTED_OPTIMIZER_CHECKS,
    EXPECTED_REPLAY_ALLOCATION_BYTES,
    EXPECTED_ROLLOUT_CHECKS,
    EXPECTED_TRAINING_SCALAR_CHECKS,
    TQCModelConstructionAuthorityV2,
    TQCTrainingCompletionAuthorityV2,
    TQCTrainingEnvironmentAuthorityV2,
)

ROOT = Path(__file__).parents[2]
CONFIG_ROOT = ROOT / "experiments/bootstrap_tqc_humanoid/configs"
DESIGN_PATH = CONFIG_ROOT / "tqc_base_controller_dev_1m_v2.study.json"
AUDIT_PATH = CONFIG_ROOT / "tqc_e0_reuse_dev_1m_v2.audit.json"


def _loaded():
    return load_tqc_development_design_v2(DESIGN_PATH, AUDIT_PATH)


def _projection():
    return _loaded().to_dict()["training_projection"]


def _small_model(*, learning_starts: int = 5) -> tuple[TQC, DummyVecEnv]:
    environment = DummyVecEnv(
        [lambda: make_humanoid_env(HumanoidExperimentConfig(), capture_substep_contacts=False)]
    )
    model = TQC(
        "MlpPolicy",
        environment,
        learning_rate=0.0003,
        buffer_size=64,
        learning_starts=learning_starts,
        batch_size=8,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=None,
        replay_buffer_class=ReplayBuffer,
        replay_buffer_kwargs={"handle_timeout_termination": True},
        optimize_memory_usage=False,
        n_steps=1,
        ent_coef="auto",
        target_update_interval=1,
        target_entropy="auto",
        top_quantiles_to_drop_per_net=2,
        use_sde=False,
        sde_sample_freq=-1,
        use_sde_at_warmup=False,
        stats_window_size=100,
        tensorboard_log=None,
        policy_kwargs={
            "net_arch": [256, 256],
            "activation_fn": torch.nn.ReLU,
            "use_sde": False,
            "log_std_init": -3.0,
            "use_expln": False,
            "clip_mean": 2.0,
            "features_extractor_class": FlattenExtractor,
            "features_extractor_kwargs": None,
            "normalize_images": True,
            "optimizer_class": torch.optim.Adam,
            "optimizer_kwargs": {"eps": 1e-5},
            "n_quantiles": 25,
            "n_critics": 2,
            "share_features_extractor": False,
        },
        verbose=0,
        seed=95_001,
        device="cpu",
        _init_setup_model=True,
    )
    model.set_logger(Logger(folder=None, output_formats=[]))
    return model, environment


class _MechanicalResourceMonitor:
    def __init__(self) -> None:
        self.lifecycle = ["preflight", "post_model_construction"]
        self.vector_samples: list[tuple[int, int]] = []

    def sample_lifecycle(self, stage: str) -> None:
        self.lifecycle.append(stage)

    def sample_vector_step(self, vector_step: int, environment_steps: int) -> None:
        self.vector_samples.append((vector_step, environment_steps))

    def validate_training_prefix(self) -> object:
        return object()


def _mechanical_plan() -> training_module._CallbackPlan:
    return training_module._CallbackPlan(
        n_envs=1,
        expected_vector_steps=30,
        expected_environment_steps=30,
        learning_starts=5,
        expected_updates=25,
        replay_slot_capacity=64,
        expected_final_replay_position=30,
        expected_final_replay_full=False,
        parameter_interval=10,
        expected_parameter_checks=5,
        expected_optimizer_checks=4,
        expected_scalar_checks=25,
        first_scalar_callback=7,
        observation_shape=(348,),
        action_shape=(17,),
    )


def _callback_locals() -> dict[str, object]:
    return {
        "new_obs": np.zeros((1, 348), dtype="<f8"),
        "rewards": np.zeros((1,), dtype="<f4"),
        "dones": np.zeros((1,), dtype=np.bool_),
        "actions": np.zeros((1, 17), dtype="<f4"),
        "buffer_actions": np.zeros((1, 17), dtype="<f4"),
        "infos": [{}],
        "num_collected_steps": 1,
    }


def _preflighted_monitor_and_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[TQCResourceMonitorV2, object]:
    clock = 0.0

    def perf_counter() -> float:
        nonlocal clock
        clock += 0.00001
        return clock

    monkeypatch.setattr(resource_module, "_PERF_COUNTER", perf_counter)
    monkeypatch.setattr(
        resource_module,
        "_GETRUSAGE",
        lambda _who: SimpleNamespace(ru_maxrss=100_000_000),
    )
    monkeypatch.setattr(
        resource_module,
        "_FSTATVFS",
        lambda _descriptor: SimpleNamespace(
            f_bavail=resource_module.FREE_DISK_MINIMUM_BYTES + 1,
            f_frsize=1,
        ),
    )
    work = (tmp_path / "attempt").resolve()
    work.mkdir(mode=0o700)
    work.chmod(0o700)
    monitor = TQCResourceMonitorV2(work)
    loaded = _loaded()

    class FakeManifest:
        attempt_id = resource_module.ATTEMPT_ID
        sha256 = "1" * 64
        design_file_sha256 = loaded.file_sha256
        design_semantic_sha256 = loaded.semantic_sha256
        training_projection_sha256 = loaded.training_projection_sha256
        claimed_work_directory_identity = monitor.claimed_work_directory_identity
        model_construction_allowed = True
        authorizes_training = True
        behavioral_evidence = False

    monkeypatch.setattr(manifest_module, "ValidatedTQCExecutionManifestV2", FakeManifest)
    return monitor, FakeManifest()


def _prime_callback_start(model: TQC, environment: DummyVecEnv) -> None:
    model._last_obs = environment.reset()
    model._last_episode_starts = np.ones((1,), dtype=np.bool_)
    model._total_timesteps = 30
    model._num_timesteps_at_start = 0


def test_canonical_plan_and_constructor_arguments_are_fully_explicit() -> None:
    projection = _projection()
    plan = training_module._canonical_callback_plan(projection)
    kwargs = training_module._constructor_kwargs(projection)

    assert plan.expected_vector_steps == EXPECTED_VECTOR_STEPS
    assert plan.expected_updates == EXPECTED_GRADIENT_UPDATES
    assert plan.expected_parameter_checks == EXPECTED_PARAMETER_CHECKS
    assert plan.expected_scalar_checks == EXPECTED_TRAINING_SCALAR_CHECKS
    assert EXPECTED_ROLLOUT_CHECKS == 200_000
    assert EXPECTED_OPTIMIZER_CHECKS == 201
    assert EXPECTED_TRAINING_SCALAR_CHECKS == 199_980
    assert EXPECTED_REPLAY_ALLOCATION_BYTES == 5_648_000_000
    assert kwargs["buffer_size"] == 1_000_000
    assert kwargs["learning_starts"] == 100
    assert kwargs["train_freq"] == 1
    assert kwargs["gradient_steps"] == 1
    assert kwargs["replay_buffer_class"] is ReplayBuffer
    assert kwargs["policy_kwargs"] == {
        "net_arch": [256, 256],
        "activation_fn": torch.nn.ReLU,
        "use_sde": False,
        "log_std_init": -3.0,
        "use_expln": False,
        "clip_mean": 2.0,
        "features_extractor_class": FlattenExtractor,
        "features_extractor_kwargs": None,
        "normalize_images": True,
        "optimizer_class": torch.optim.Adam,
        "optimizer_kwargs": {"eps": 1e-5},
        "n_quantiles": 25,
        "n_critics": 2,
        "share_features_extractor": False,
    }


@pytest.mark.gym
def test_real_plain_humanoid_vector_environment_matches_the_v2_stack() -> None:
    projection = _projection()
    config = HumanoidExperimentConfig()
    environment = DummyVecEnv(
        [lambda: make_humanoid_env(config, capture_substep_contacts=False) for _ in range(5)]
    )
    try:
        assert tuple(environment.seed(95_001)) == tuple(range(95_001, 95_006))
        training_module._validate_training_environment(environment, projection)
        environment.envs[0].unwrapped.model.opt.gravity[2] = -9.8
        with pytest.raises(ExperimentContractError, match="gravity"):
            training_module._validate_training_environment(environment, projection)
        environment.envs[0].unwrapped.model.opt.gravity[2] = -9.81
        environment._seeds[2] = 7
        with pytest.raises(ExperimentContractError, match="worker seeds"):
            training_module._validate_training_environment(environment, projection)
    finally:
        environment.close()


@pytest.mark.gym
def test_environment_preparation_binds_preflighted_monitor_before_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor, manifest = _preflighted_monitor_and_manifest(tmp_path, monkeypatch)
    monitor.sample_lifecycle("preflight")

    authority = training_module.prepare_tqc_training_environment_v2(_loaded(), manifest, monitor)
    try:
        assert authority.claimed_work_directory_identity == (
            monitor.claimed_work_directory_identity
        )
        assert authority.execution_manifest_sha256 == manifest.sha256
        monitor.sample_lifecycle("post_model_construction")
    finally:
        authority.environment.close()


def test_environment_preparation_does_not_invent_missing_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monitor, manifest = _preflighted_monitor_and_manifest(tmp_path, monkeypatch)

    with pytest.raises(ExperimentContractError, match="out of order"):
        training_module.prepare_tqc_training_environment_v2(_loaded(), manifest, monitor)


@pytest.mark.gym
@pytest.mark.slow
def test_real_tqc_mechanical_run_checks_every_small_schedule_point() -> None:
    model, environment = _small_model()
    resource = _MechanicalResourceMonitor()
    callback = training_module._make_training_callback(
        model=model,
        projection=_projection(),
        plan=_mechanical_plan(),
        resource_monitor=resource,
    )
    try:
        returned = model.learn(
            total_timesteps=30,
            callback=callback,
            log_interval=1_000,
            tb_log_name="TQC",
            reset_num_timesteps=True,
            progress_bar=False,
        )
        assert returned is model
        assert callback.training_complete is True
        assert callback.n_calls == 30
        assert callback.rollout_check_count == 30
        assert callback.parameter_check_count == 5
        assert callback.optimizer_check_count == 4
        assert callback.training_scalar_check_count == 25
        assert resource.lifecycle == [
            "preflight",
            "post_model_construction",
            "pre_learn",
            "post_learn",
        ]
        assert resource.vector_samples == [(step, step) for step in range(1, 31)]
        training_module._validate_policy_modules(model, _projection())
        for digest in (
            callback.rollout_digest,
            callback.parameter_digest,
            callback.optimizer_digest,
            callback.training_scalar_digest,
        ):
            assert len(digest.hexdigest()) == 64
    finally:
        environment.close()


@pytest.mark.gym
def test_nonintegral_logged_update_counter_fails_closed() -> None:
    model, environment = _small_model()
    model.logger.name_to_value.update(
        {
            "train/actor_loss": 1.0,
            "train/critic_loss": 1.0,
            "train/ent_coef": 1.0,
            "train/ent_coef_loss": 1.0,
            "train/n_updates": 1.5,
        }
    )
    try:
        with pytest.raises(ExperimentContractError, match="count type"):
            training_module._training_scalar_receipt(
                model,
                _projection(),
                expected_update=1,
            )
    finally:
        environment.close()


@pytest.mark.gym
def test_callback_rejects_update_lag_before_current_train_call() -> None:
    model, environment = _small_model()
    resource = _MechanicalResourceMonitor()
    callback = training_module._make_training_callback(
        model=model,
        projection=_projection(),
        plan=_mechanical_plan(),
        resource_monitor=resource,
    )
    try:
        _prime_callback_start(model, environment)
        callback.init_callback(model)
        callback.on_training_start({}, {})
        model.num_timesteps = 1
        model._n_updates = 1
        callback.update_locals(_callback_locals())
        with pytest.raises(ExperimentContractError, match="update lag"):
            callback.on_step()
    finally:
        environment.close()


@pytest.mark.gym
def test_callback_rejects_nonfinite_rollout_before_replay_add() -> None:
    model, environment = _small_model()
    resource = _MechanicalResourceMonitor()
    callback = training_module._make_training_callback(
        model=model,
        projection=_projection(),
        plan=_mechanical_plan(),
        resource_monitor=resource,
    )
    locals_ = _callback_locals()
    locals_["new_obs"][0, 0] = np.nan
    try:
        _prime_callback_start(model, environment)
        callback.init_callback(model)
        callback.on_training_start({}, {})
        model.num_timesteps = 1
        callback.update_locals(locals_)
        with pytest.raises(ExperimentContractError, match="new_obs"):
            callback.on_step()
        assert model.replay_buffer.pos == 0
    finally:
        environment.close()


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("actions", 0.4001, "physical action"),
        ("buffer_actions", 1.0001, "normalized action"),
    ],
)
def test_rollout_action_envelopes_fail_closed(field: str, value: float, error: str) -> None:
    locals_ = _callback_locals()
    locals_[field][0, 0] = value

    with pytest.raises(ExperimentContractError, match=error):
        training_module._rollout_receipt(locals_, plan=_mechanical_plan(), vector_step=1)


@pytest.mark.gym
def test_parameter_and_optimizer_mutations_fail_closed() -> None:
    model, environment = _small_model()
    projection = _projection()
    try:
        initial = training_module._optimizer_state_receipt(model, projection, expected_update=None)
        assert initial["state_tensor_count"] == 0
        model.actor.optimizer.defaults["eps"] = 1e-8
        with pytest.raises(ExperimentContractError, match="defaults"):
            training_module._optimizer_state_receipt(model, projection, expected_update=None)
        model.actor.optimizer.defaults["eps"] = 1e-5
        with torch.no_grad():
            model.actor.mu.bias[0] = torch.nan
        with pytest.raises(ExperimentContractError, match="non-finite"):
            training_module._parameter_state_receipt(model, projection)
    finally:
        environment.close()


@pytest.mark.gym
def test_policy_module_graph_mutation_fails_closed() -> None:
    model, environment = _small_model()
    try:
        training_module._validate_policy_modules(model, _projection())
        model.actor.latent_pi[1] = torch.nn.Identity()
        with pytest.raises(ExperimentContractError, match="actor module graph"):
            training_module._validate_policy_modules(model, _projection())
    finally:
        environment.close()


def test_manifest_and_authority_boundaries_reject_plain_objects() -> None:
    loaded = _loaded()
    with pytest.raises(ExperimentContractError, match=r"execution.manifest|manifest"):
        training_module._require_execution_manifest(object(), loaded)

    for authority_class in (
        TQCTrainingEnvironmentAuthorityV2,
        TQCModelConstructionAuthorityV2,
        TQCTrainingCompletionAuthorityV2,
    ):
        with pytest.raises(ExperimentContractError, match="only be issued"):
            authority_class.__new__(authority_class).__post_init__(None)


def test_completion_authority_is_not_dataclass_replaceable() -> None:
    fields = {
        field.name: None
        for field in TQCTrainingCompletionAuthorityV2.__dataclass_fields__.values()
        if field.name != "_issuer"
    }
    instance = object.__new__(TQCTrainingCompletionAuthorityV2)
    for name, value in fields.items():
        object.__setattr__(instance, name, value)
    with pytest.raises(ExperimentContractError, match="only be issued"):
        replace(instance)


def _sealed_completion_shell() -> tuple[TQCTrainingCompletionAuthorityV2, object]:
    marker = object()
    instance = object.__new__(TQCTrainingCompletionAuthorityV2)
    string_values = {
        "integrity_id": training_module.TRAINING_INTEGRITY_ID,
        "attempt_id": resource_module.ATTEMPT_ID,
        "resource_monitor_id": resource_module.RESOURCE_MONITOR_ID,
        "target_polyak_update_count_source": (
            "pinned_TQC_train_source_plus_exact_train_and_update_counters/v1"
        ),
        "claim_boundary": "training_integrity_only_no_behavior_tracker_or_oracle_claim/v1",
    }
    ignored = {
        "_model_handle",
        "resource_monitor",
        "resource_authority",
        "manifest",
        "design",
        "_issuer",
    }
    for name in TQCTrainingCompletionAuthorityV2.__dataclass_fields__:
        if name in ignored:
            continue
        if name in string_values:
            value: object = string_values[name]
        elif name == "exact_final_checkpoint_only":
            value = True
        elif name.endswith("_sha256"):
            value = "a" * 64
        else:
            value = 1
        object.__setattr__(instance, name, value)
    object.__setattr__(instance, "resource_monitor", object())
    object.__setattr__(instance, "resource_authority", object())
    object.__setattr__(instance, "manifest", object())
    object.__setattr__(instance, "design", object())
    payload = instance.to_dict()
    payload["training_integrity_sha256"] = hashlib.sha256(
        training_module.canonical_json(
            {key: value for key, value in payload.items() if key != "training_integrity_sha256"}
        )
    ).hexdigest()
    object.__setattr__(instance, "training_integrity_sha256", payload["training_integrity_sha256"])
    handle = training_module._TQCLiveModelHandleV2(
        _model=marker,
        _phase="constructed",
        _creator_pid=os.getpid(),
        _issuer=training_module._MODEL_HANDLE_ISSUER,
    )
    handle.seal_completion(instance.to_dict())
    object.__setattr__(instance, "_model_handle", handle)
    return instance, marker


def test_model_handle_transfers_and_invalidates_construction() -> None:
    marker = object()
    handle = training_module._TQCLiveModelHandleV2(
        _model=marker,
        _phase="constructed",
        _creator_pid=os.getpid(),
        _issuer=training_module._MODEL_HANDLE_ISSUER,
    )

    assert handle.constructed_model() is marker
    handle.seal_completion({"sealed": True})
    with pytest.raises(ExperimentContractError, match="ownership was transferred"):
        handle.constructed_model()
    assert handle.completed_model({"sealed": True}) is marker


def test_consumption_is_one_shot_across_aliases() -> None:
    completion, marker = _sealed_completion_shell()
    alias = completion

    assert completion._model_handle.consume_completed_model(completion.to_dict()) is marker
    with pytest.raises(ExperimentContractError, match="completion seal differs"):
        alias._model_handle.consume_completed_model(alias.to_dict())


def test_original_replay_is_collectable_before_strict_reload() -> None:
    class ReplayOwner:
        pass

    model = ReplayOwner()
    reference = weakref.ref(model)
    handle = training_module._TQCLiveModelHandleV2(
        _model=model,
        _phase="constructed",
        _creator_pid=os.getpid(),
        _issuer=training_module._MODEL_HANDLE_ISSUER,
    )
    handle.seal_completion({"sealed": True})
    consumed = handle.consume_completed_model({"sealed": True})
    del model
    del consumed
    gc.collect()

    assert reference() is None


def test_recomputed_integrity_cannot_replace_callback_digests() -> None:
    completion, _marker = _sealed_completion_shell()
    clone = object.__new__(TQCTrainingCompletionAuthorityV2)
    for name in TQCTrainingCompletionAuthorityV2.__dataclass_fields__:
        if name != "_issuer":
            object.__setattr__(clone, name, getattr(completion, name))
    object.__setattr__(clone, "rollout_checks_sha256", "b" * 64)
    changed = clone.to_dict()
    changed["training_integrity_sha256"] = hashlib.sha256(
        training_module.canonical_json(
            {key: value for key, value in changed.items() if key != "training_integrity_sha256"}
        )
    ).hexdigest()
    object.__setattr__(clone, "training_integrity_sha256", changed["training_integrity_sha256"])

    with pytest.raises(ExperimentContractError, match="completion seal differs"):
        _ = clone.model


@pytest.mark.gym
@pytest.mark.parametrize("replay_position", (1, 63))
def test_callback_rejects_skipped_or_duplicate_pre_add_replay_counters(
    replay_position: int,
) -> None:
    model, environment = _small_model()
    resource = _MechanicalResourceMonitor()
    callback = training_module._make_training_callback(
        model=model,
        projection=_projection(),
        plan=_mechanical_plan(),
        resource_monitor=resource,
    )
    try:
        _prime_callback_start(model, environment)
        callback.init_callback(model)
        callback.on_training_start({}, {})
        model.num_timesteps = 1
        model.replay_buffer.pos = replay_position
        callback.update_locals(_callback_locals())
        with pytest.raises(ExperimentContractError, match="pre-add replay counter differs"):
            callback.on_step()
    finally:
        environment.close()
