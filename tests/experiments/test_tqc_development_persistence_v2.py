from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import weakref
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from sb3_contrib import TQC
from stable_baselines3.common.vec_env import DummyVecEnv

from oracle_composition.experiments import tqc_development_persistence_v2 as persistence
from oracle_composition.experiments import tqc_development_training_v2 as training_module
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_actor_equivalence_v2 import (
    admit_final_tqc_checkpoint_actor,
    revalidate_tqc_actor_equivalence_receipt_v2,
    verify_actor_equivalence_v2,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
V2_DESIGN_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "bootstrap_tqc_humanoid"
    / "configs"
    / "tqc_base_controller_dev_1m_v2.study.json"
)


class _Design:
    def __init__(self, projection: dict[str, object]) -> None:
        self._projection = projection

    def to_dict(self) -> dict[str, object]:
        return {"training_projection": self._projection}


class _Monitor:
    def __init__(self, work_directory: Path) -> None:
        self._descriptor = os.open(work_directory, os.O_RDONLY | os.O_DIRECTORY)
        self.lifecycle: list[str] = []
        self.disk: list[str] = []
        self.failed_reason: str | None = None

    def duplicate_work_directory_descriptor(self) -> int:
        if self.failed_reason is not None:
            raise ExperimentContractError("monitor is poisoned")
        return os.dup(self._descriptor)

    def sample_post_training_disk_gate(self, stage: str) -> None:
        self.disk.append(stage)

    def sample_lifecycle(self, stage: str) -> None:
        self.lifecycle.append(stage)

    def _poison(self, exc: BaseException) -> None:
        if self.failed_reason is None:
            self.failed_reason = f"{type(exc).__name__}:{exc}"

    def close(self) -> None:
        os.close(self._descriptor)


def _sealed_completion(
    model: TQC,
    monitor: _Monitor,
    projection: dict[str, object],
) -> training_module.TQCTrainingCompletionAuthorityV2:
    completion = object.__new__(training_module.TQCTrainingCompletionAuthorityV2)
    values: dict[str, object] = {
        "integrity_id": training_module.TRAINING_INTEGRITY_ID,
        "attempt_id": "dev1m-v2-seed-95001-attempt-01",
        "execution_manifest_sha256": "1" * 64,
        "claimed_work_directory_identity": "2" * 64,
        "resource_monitor_id": "tqc_dev_1m_v2_process_resource_monitor/v1",
        "worker_pid": os.getpid(),
        "design_file_sha256": "3" * 64,
        "design_semantic_sha256": "4" * 64,
        "training_projection_sha256": "5" * 64,
        "environment_steps": 1_000_000,
        "vector_steps": 200_000,
        "gradient_updates": 199_980,
        "replay_add_calls": 200_000,
        "actor_optimizer_steps": 199_980,
        "critic_optimizer_steps": 199_980,
        "entropy_optimizer_steps": 199_980,
        "target_polyak_updates": 199_980,
        "target_polyak_update_count_source": (
            "pinned_TQC_train_source_plus_exact_train_and_update_counters/v1"
        ),
        "rollout_check_count": 200_000,
        "parameter_check_count": 202,
        "optimizer_check_count": 201,
        "training_scalar_check_count": 199_980,
        "rollout_checks_sha256": "6" * 64,
        "parameter_checks_sha256": "7" * 64,
        "optimizer_checks_sha256": "8" * 64,
        "training_scalar_checks_sha256": "9" * 64,
        "initial_policy_state_sha256": "a" * 64,
        "final_policy_state_sha256": "b" * 64,
        "initial_entropy_state_sha256": "c" * 64,
        "final_entropy_state_sha256": "d" * 64,
        "final_optimizer_state_sha256": "e" * 64,
        "resource_prefix_event_sha256": "f" * 64,
        "training_integrity_sha256": "0" * 64,
        "exact_final_checkpoint_only": True,
        "claim_boundary": "training_integrity_only_no_behavior_tracker_or_oracle_claim/v1",
    }
    for name, value in values.items():
        object.__setattr__(completion, name, value)
    object.__setattr__(completion, "resource_monitor", monitor)
    object.__setattr__(completion, "resource_authority", object())
    object.__setattr__(completion, "manifest", object())
    object.__setattr__(completion, "design", _Design(projection))
    payload = completion.to_dict()
    payload["training_integrity_sha256"] = hashlib.sha256(
        training_module.canonical_json(
            {key: value for key, value in payload.items() if key != "training_integrity_sha256"}
        )
    ).hexdigest()
    object.__setattr__(
        completion,
        "training_integrity_sha256",
        payload["training_integrity_sha256"],
    )
    handle = training_module._TQCLiveModelHandleV2(
        _model=model,
        _phase="constructed",
        _creator_pid=os.getpid(),
        _issuer=training_module._MODEL_HANDLE_ISSUER,
    )
    handle.seal_completion(completion.to_dict())
    object.__setattr__(completion, "_model_handle", handle)
    return completion


def _initialize_optimizer_state(model: TQC) -> None:
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


def _small_model() -> tuple[TQC, DummyVecEnv]:
    environment = DummyVecEnv([lambda: gym.make("Humanoid-v5")])
    model = TQC(
        "MlpPolicy",
        environment,
        learning_rate=0.0003,
        buffer_size=12,
        learning_starts=100,
        batch_size=4,
        gradient_steps=1,
        policy_kwargs={
            "net_arch": [256, 256],
            "n_quantiles": 25,
            "n_critics": 2,
            "optimizer_kwargs": {"eps": 1e-5},
        },
        seed=95_001,
        device="cpu",
        verbose=0,
    )
    model.num_timesteps = 1_000_000
    model._n_updates = 199_980
    model.replay_buffer.pos = 0
    model.replay_buffer.full = True
    _initialize_optimizer_state(model)
    return model, environment


def _small_projection(model: TQC) -> dict[str, object]:
    replay = model.replay_buffer
    allocation = sum(int(getattr(replay, name).nbytes) for name in persistence.REPLAY_ARRAY_ORDER)
    return {
        "replay_buffer": {
            "requested_transition_capacity": replay.buffer_size * replay.n_envs,
            "n_envs": replay.n_envs,
            "internal_vector_slot_capacity": replay.buffer_size,
            "observation_shape": list(replay.observation_space.shape),
            "action_shape": list(replay.action_space.shape),
            "observation_dtype": replay.observations.dtype.str,
            "action_dtype": replay.actions.dtype.str,
            "reward_dtype": replay.rewards.dtype.str,
            "done_dtype": replay.dones.dtype.str,
            "timeout_dtype": replay.timeouts.dtype.str,
            "optimize_memory_usage": replay.optimize_memory_usage,
            "handle_timeout_termination": replay.handle_timeout_termination,
            "n_step_return": model.n_steps,
            "expected_allocation_bytes": allocation,
            "expected_final_position": 0,
            "expected_final_full": True,
        }
    }


@pytest.fixture
def completion_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    os.chmod(tmp_path, 0o700)
    model, environment = _small_model()
    projection = _small_projection(model)
    monitor = _Monitor(tmp_path)
    completion = _sealed_completion(model, monitor, projection)
    replay_bytes = projection["replay_buffer"]["expected_allocation_bytes"]

    monkeypatch.setattr(persistence, "TQCResourceMonitorV2", _Monitor)
    monkeypatch.setattr(persistence, "EXPECTED_REPLAY_ARRAY_PAYLOAD_BYTES", replay_bytes)
    monkeypatch.setattr(persistence, "_validate_persistence_projection", lambda _value: None)
    monkeypatch.setattr(
        persistence,
        "revalidate_tqc_training_completion_v2",
        lambda value: value.model,
    )
    monkeypatch.setattr(
        training_module,
        "revalidate_tqc_training_completion_v2",
        lambda value: value.model,
    )
    try:
        yield completion, tmp_path
    finally:
        retained_model = completion._model_handle._model
        if retained_model is not None:
            retained_model.replay_buffer = None
            retained_model.env = None
        environment.close()
        monitor.close()


def _persist_and_equivalence(
    completion: training_module.TQCTrainingCompletionAuthorityV2,
    work_directory: Path,
):
    authority = persistence.persist_tqc_training_completion_v2(
        completion,
        work_directory=work_directory,
    )
    actor_authority = admit_final_tqc_checkpoint_actor(authority)
    receipt = verify_actor_equivalence_v2(actor_authority)
    return authority, receipt


def test_chunked_array_hash_matches_the_declared_byte_rule() -> None:
    value = np.arange(35, dtype="<f4").reshape(5, 7)
    entry = persistence._array_entry("values", value)
    header = persistence.canonical_json({"dtype": value.dtype.str, "shape": list(value.shape)})
    expected = hashlib.sha256(
        len(header).to_bytes(8, "big")
        + header
        + value.nbytes.to_bytes(8, "big")
        + value.tobytes(order="C")
    ).hexdigest()

    assert entry == {
        "name": "values",
        "dtype": "<f4",
        "shape": [5, 7],
        "byte_count": value.nbytes,
        "array_sha256": expected,
    }


@pytest.mark.parametrize(
    "field_name",
    ["strict_reload_hash_pairs", "memory_safe_reload_lifecycle"],
)
def test_persistence_projection_rejects_strict_reload_protocol_drift(
    field_name: str,
) -> None:
    design = json.loads(V2_DESIGN_PATH.read_text(encoding="utf-8"))
    projection = design["training_projection"]
    persistence._validate_persistence_projection(projection)
    mutated = copy.deepcopy(projection)
    mutated["persistence"]["trusted_local_state"][field_name] = ["weakened"]

    with pytest.raises(ExperimentContractError, match="strict-reload contract"):
        persistence._validate_persistence_projection(mutated)


def test_chunked_array_hash_rejects_non_c_order_and_nonfinite() -> None:
    with pytest.raises(ExperimentContractError, match="C-order"):
        persistence._array_entry(
            "transposed",
            np.arange(12, dtype="<f4").reshape(3, 4).T,
        )
    value = np.zeros((4,), dtype="<f4")
    value[2] = np.nan
    with pytest.raises(ExperimentContractError, match="non-finite"):
        persistence._array_entry("nan", value)


def test_small_replay_hash_binds_arrays_and_exact_metadata() -> None:
    model, environment = _small_model()
    try:
        projection = _small_projection(model)
        first = persistence._replay_state_receipt(model, projection)
        model.replay_buffer.rewards[0, 0] = np.float32(1.0)
        second = persistence._replay_state_receipt(model, projection)

        assert (
            first["array_payload_bytes"] == projection["replay_buffer"]["expected_allocation_bytes"]
        )
        assert first["metadata"]["position"] == 0
        assert first["metadata"]["full"] is True
        assert first["arrays_sha256"] != second["arrays_sha256"]
        assert first["metadata_sha256"] == second["metadata_sha256"]
    finally:
        environment.close()


def test_replay_hash_rejects_boolean_counter_alias() -> None:
    model, environment = _small_model()
    try:
        projection = _small_projection(model)
        model.replay_buffer.pos = False

        with pytest.raises(ExperimentContractError, match="inexact primitive"):
            persistence._replay_state_receipt(model, projection)
    finally:
        environment.close()


def test_persistence_and_strict_reload_round_trip_without_two_replays(
    completion_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion, work_directory = completion_context
    authority, equivalence = _persist_and_equivalence(completion, work_directory)
    replay_reference = weakref.ref(completion.model.replay_buffer)
    empty_replay_reference = None
    load_model = persistence._load_bound_model
    load_replay = TQC.load_replay_buffer

    def guarded_load(artifact):
        nonlocal empty_replay_reference
        assert completion._model_handle._model is None
        assert replay_reference() is None
        loaded = load_model(artifact)
        empty_replay_reference = weakref.ref(loaded.replay_buffer)
        return loaded

    def guarded_replay_load(model, *args, **kwargs):
        assert empty_replay_reference is not None
        assert empty_replay_reference() is None
        return load_replay(model, *args, **kwargs)

    monkeypatch.setattr(persistence, "_load_bound_model", guarded_load)
    monkeypatch.setattr(TQC, "load_replay_buffer", guarded_replay_load)
    reloaded = persistence.strict_reload_tqc_persistence_v2(
        authority,
        actor_equivalence_receipt=equivalence,
    )

    assert reloaded.passed is True
    assert reloaded.behavioral_evidence is False
    assert reloaded.source_policy_state_sha256 == reloaded.loaded_policy_state_sha256
    assert reloaded.source_entropy_state_sha256 == reloaded.loaded_entropy_state_sha256
    assert (
        reloaded.source_actor_optimizer_state_sha256 == reloaded.loaded_actor_optimizer_state_sha256
    )
    assert (
        reloaded.source_critic_optimizer_state_sha256
        == reloaded.loaded_critic_optimizer_state_sha256
    )
    assert (
        reloaded.source_entropy_optimizer_state_sha256
        == reloaded.loaded_entropy_optimizer_state_sha256
    )
    assert reloaded.source_replay_arrays_sha256 == reloaded.loaded_replay_arrays_sha256
    assert reloaded.source_replay_metadata_sha256 == reloaded.loaded_replay_metadata_sha256
    assert authority.training_completion._model_handle._model is None
    assert authority.training_completion.resource_monitor.disk == ["pre_persistence"]
    assert authority.training_completion.resource_monitor.lifecycle == [
        "pre_model_save",
        "post_model_save",
        "pre_replay_save",
        "post_replay_save",
        "pre_actor_export",
        "post_actor_export",
    ]
    for artifact in (
        authority.model_artifact,
        authority.replay_artifact,
        authority.actor_artifact,
        authority.actor_manifest_artifact,
    ):
        assert stat.S_IMODE(artifact.path.stat().st_mode) == 0o600
        assert artifact.path.stat().st_size == artifact.byte_count
    assert revalidate_tqc_actor_equivalence_receipt_v2(equivalence) is equivalence
    assert persistence.revalidate_tqc_strict_reload_authority_v2(reloaded) is reloaded
    with pytest.raises(ExperimentContractError, match="only be issued"):
        replace(authority)
    with pytest.raises(ExperimentContractError, match="only be issued"):
        replace(reloaded)


def test_revalidation_rejects_forged_persistence_clone_with_recomputed_hash(
    completion_context,
) -> None:
    completion, work_directory = completion_context
    authority = persistence.persist_tqc_training_completion_v2(
        completion,
        work_directory=work_directory,
    )
    forged = copy.copy(authority)
    object.__setattr__(forged, "attempt_id", "dev1m-v2-seed-95001-attempt-02")
    unsigned = {
        key: value for key, value in forged.to_dict().items() if key != "persistence_sha256"
    }
    object.__setattr__(
        forged,
        "persistence_sha256",
        hashlib.sha256(persistence.canonical_json(unsigned)).hexdigest(),
    )

    with pytest.raises(ExperimentContractError, match="authority seal differs"):
        persistence.revalidate_tqc_persistence_authority_v2(forged)


def test_revalidation_rejects_forged_reload_clone_with_recomputed_hash(
    completion_context,
) -> None:
    completion, work_directory = completion_context
    authority, equivalence = _persist_and_equivalence(completion, work_directory)
    reloaded = persistence.strict_reload_tqc_persistence_v2(
        authority,
        actor_equivalence_receipt=equivalence,
    )
    forged = copy.copy(reloaded)
    object.__setattr__(forged, "loaded_environment_steps", 999_999)
    unsigned = {
        key: value for key, value in forged.to_dict().items() if key != "strict_reload_sha256"
    }
    object.__setattr__(
        forged,
        "strict_reload_sha256",
        hashlib.sha256(persistence.canonical_json(unsigned)).hexdigest(),
    )

    with pytest.raises(ExperimentContractError, match="authority seal differs"):
        persistence.revalidate_tqc_strict_reload_authority_v2(forged)


def test_strict_reload_consumes_all_persistence_aliases_once(
    completion_context,
) -> None:
    completion, work_directory = completion_context
    authority, equivalence = _persist_and_equivalence(completion, work_directory)
    alias = authority
    persistence.strict_reload_tqc_persistence_v2(
        authority,
        actor_equivalence_receipt=equivalence,
    )

    with pytest.raises(ExperimentContractError, match="authority seal differs"):
        persistence.strict_reload_tqc_persistence_v2(
            alias,
            actor_equivalence_receipt=equivalence,
        )


def test_strict_reload_requires_actor_equivalence_before_consuming_source(
    completion_context,
) -> None:
    completion, work_directory = completion_context
    authority = persistence.persist_tqc_training_completion_v2(
        completion,
        work_directory=work_directory,
    )

    with pytest.raises(ExperimentContractError, match="actor equivalence"):
        persistence.strict_reload_tqc_persistence_v2(
            authority,
            actor_equivalence_receipt=object(),
        )

    assert completion._model_handle._model is not None
    assert completion.resource_monitor.failed_reason is not None


def test_strict_reload_rejects_cross_manifest_actor_receipt_before_consuming_source(
    completion_context,
) -> None:
    completion, work_directory = completion_context
    authority, equivalence = _persist_and_equivalence(completion, work_directory)
    object.__setattr__(equivalence, "execution_manifest_sha256", "f" * 64)

    with pytest.raises(ExperimentContractError, match="receipt seal differs"):
        persistence.strict_reload_tqc_persistence_v2(
            authority,
            actor_equivalence_receipt=equivalence,
        )

    assert completion._model_handle._model is not None
    assert completion.resource_monitor.failed_reason is not None


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("attempt_id", "dev1m-v2-seed-95001-attempt-02"),
        ("execution_manifest_sha256", "a" * 64),
        ("design_file_sha256", "b" * 64),
        ("design_semantic_sha256", "c" * 64),
        ("training_projection_sha256", "d" * 64),
        ("claimed_work_directory_identity", "e" * 64),
    ],
)
def test_persistence_authority_rejects_completion_provenance_substitution(
    completion_context,
    field_name: str,
    replacement: str,
) -> None:
    completion, work_directory = completion_context
    authority = persistence.persist_tqc_training_completion_v2(
        completion,
        work_directory=work_directory,
    )
    object.__setattr__(completion, field_name, replacement)

    with pytest.raises(ExperimentContractError, match="authority is inconsistent"):
        persistence.revalidate_tqc_persistence_authority_v2(authority)


def test_strict_reload_rejects_replay_bytes_changed_after_persistence(
    completion_context,
) -> None:
    completion, work_directory = completion_context
    authority, equivalence = _persist_and_equivalence(completion, work_directory)
    replay_path = authority.replay_artifact.path
    with replay_path.open("r+b") as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first[0] ^ 0x01]))

    with pytest.raises(ExperimentContractError, match="SHA-256 differs"):
        persistence.strict_reload_tqc_persistence_v2(
            authority,
            actor_equivalence_receipt=equivalence,
        )

    assert completion._model_handle._model is None
    assert completion.resource_monitor.failed_reason is not None


def test_persistence_rejects_wrong_work_directory_binding(
    completion_context,
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    completion, _work_directory = completion_context
    other = tmp_path_factory.mktemp("other-work")
    os.chmod(other, 0o700)

    with pytest.raises(ExperimentContractError, match="retained work directory"):
        persistence.persist_tqc_training_completion_v2(
            completion,
            work_directory=other,
        )


def test_reservation_parent_must_match_retained_work_directory_descriptor(
    tmp_path: Path,
) -> None:
    retained = tmp_path / "retained"
    substituted = tmp_path / "substituted"
    retained.mkdir(mode=0o700)
    substituted.mkdir(mode=0o700)
    monitor = _Monitor(retained)
    reservation = persistence.reserve_streaming_artifact(
        substituted / "artifact.bin",
        max_bytes=16,
    )
    try:
        with pytest.raises(ExperimentContractError, match="reservation parent differs"):
            persistence._assert_reservation_parent(reservation, monitor)
    finally:
        reservation.abort()
        monitor.close()


def test_persistence_revalidation_rejects_replaced_work_directory(
    completion_context,
) -> None:
    completion, work_directory = completion_context
    authority = persistence.persist_tqc_training_completion_v2(
        completion,
        work_directory=work_directory,
    )
    displaced = work_directory.with_name(f"{work_directory.name}-displaced")
    work_directory.rename(displaced)
    work_directory.mkdir(mode=0o700)

    with pytest.raises(ExperimentContractError, match="retained work directory"):
        persistence.revalidate_tqc_persistence_authority_v2(authority)


def test_persistence_never_overwrites_an_existing_model(
    completion_context,
) -> None:
    completion, work_directory = completion_context
    model_path = work_directory / persistence.MODEL_FILENAME
    model_path.write_bytes(b"existing")
    os.chmod(model_path, 0o600)

    with pytest.raises(ExperimentContractError, match="refusing to overwrite"):
        persistence.persist_tqc_training_completion_v2(
            completion,
            work_directory=work_directory,
        )

    assert model_path.read_bytes() == b"existing"
    assert completion.resource_monitor.failed_reason is not None


def test_persistence_interrupt_poisoning_is_fail_closed(
    completion_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion, work_directory = completion_context

    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt("cancel persistence")

    monkeypatch.setattr(completion.model, "save", interrupt)

    with pytest.raises(KeyboardInterrupt, match="cancel persistence"):
        persistence.persist_tqc_training_completion_v2(
            completion,
            work_directory=work_directory,
        )

    assert completion.resource_monitor.failed_reason is not None
    assert completion._model_handle._model is not None


def test_late_reservation_collision_closes_every_earlier_stream(
    completion_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion, work_directory = completion_context
    manifest_path = work_directory / persistence.ACTOR_MANIFEST_FILENAME
    manifest_path.write_bytes(b"existing manifest")
    os.chmod(manifest_path, 0o600)
    observed = []
    reserve = persistence.reserve_streaming_artifact

    def track(*args, **kwargs):
        reservation = reserve(*args, **kwargs)
        observed.append(reservation)
        return reservation

    monkeypatch.setattr(persistence, "reserve_streaming_artifact", track)

    with pytest.raises(ExperimentContractError, match="refusing to overwrite"):
        persistence.persist_tqc_training_completion_v2(
            completion,
            work_directory=work_directory,
        )

    assert len(observed) == 3
    assert all(reservation._closed is True for reservation in observed)
    assert all(reservation.retained_path_after_failure is not None for reservation in observed)
    assert manifest_path.read_bytes() == b"existing manifest"
    assert completion.resource_monitor.failed_reason is not None


def test_strict_reload_interrupt_poisoning_invalidates_authority(
    completion_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion, work_directory = completion_context
    authority, equivalence = _persist_and_equivalence(completion, work_directory)

    def interrupt(_artifact):
        raise KeyboardInterrupt("cancel strict reload")

    monkeypatch.setattr(persistence, "_load_bound_model", interrupt)

    with pytest.raises(KeyboardInterrupt, match="cancel strict reload"):
        persistence.strict_reload_tqc_persistence_v2(
            authority,
            actor_equivalence_receipt=equivalence,
        )

    assert completion.resource_monitor.failed_reason is not None
    assert completion._model_handle._model is None
    assert authority._seal._phase == "failed"
    with pytest.raises(ExperimentContractError, match="authority seal differs"):
        persistence.revalidate_tqc_persistence_authority_v2(authority)


def test_persistence_revalidates_sealed_completion_immediately_before_issuance(
    completion_context,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completion, work_directory = completion_context
    calls = 0

    def revalidate(value):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ExperimentContractError("sealed completion changed")
        return value.model

    monkeypatch.setattr(persistence, "revalidate_tqc_training_completion_v2", revalidate)

    with pytest.raises(ExperimentContractError, match="sealed completion changed"):
        persistence.persist_tqc_training_completion_v2(
            completion,
            work_directory=work_directory,
        )

    assert calls == 2
    assert completion.resource_monitor.failed_reason is not None


def test_persistence_rejects_a_bare_model(completion_context) -> None:
    completion, work_directory = completion_context

    with pytest.raises(ExperimentContractError, match="training completion"):
        persistence.persist_tqc_training_completion_v2(
            completion.model,
            work_directory=work_directory,
        )
