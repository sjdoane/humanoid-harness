"""Exact v2 equivalence check between a TQC actor and its strict export."""

from __future__ import annotations

import hashlib
import io
import json
import os
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from pathlib import Path

import numpy as np

from .artifact_io import PublishedArtifact, read_verified_artifact_bytes
from .fixed_reference import ExperimentContractError
from .tqc_actor_npz import (
    ACTION_WIDTH,
    OBSERVATION_WIDTH,
    LoadedTQCActor,
    actor_state_sha256,
    validate_actor_arrays,
)
from .tqc_development_persistence_v2 import (
    TQCPersistenceAuthorityV2,
    revalidate_tqc_persistence_authority_v2,
)

EQUIVALENCE_ID = "strict_tqc_actor_mean_log_std_action_sample_equivalence/v2"
EQUIVALENCE_OBSERVATION_SHA256 = "0c6a81b06a88cab7eca0255e75f021008b60025c4ddc4d3719426e3647159ec6"
EQUIVALENCE_OBSERVATION_SHAPE = (4, OBSERVATION_WIDTH)
EQUIVALENCE_SAMPLING_SEED = 97_001
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
FINAL_ENVIRONMENT_STEP = 1_000_000
FINAL_UPDATE_COUNT = 199_980
MAX_TRUSTED_MODEL_ARCHIVE_BYTES = 100 * 1024 * 1024

_OUTPUT_NAMES = (
    "mean",
    "log_std",
    "deterministic_action",
    "seeded_sample",
)
_AUTHORITY_ISSUER = object()
_RECEIPT_ISSUER = object()
_AUTHORITY_SEAL_ISSUER = object()
_RECEIPT_SEAL_ISSUER = object()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExperimentContractError(f"value is not canonical JSON: {exc}") from exc


def canonical_array_sha256(value: np.ndarray) -> str:
    """Hash one C-order array with its exact dtype and shape."""

    if not isinstance(value, np.ndarray):
        raise ExperimentContractError("hashed value must be a NumPy array")
    if not value.flags.c_contiguous:
        raise ExperimentContractError("hashed array must use C-order storage")
    metadata = _canonical_json({"dtype": value.dtype.str, "shape": list(value.shape)})
    raw = value.tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _canonical_sha256(value: object, *, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"{field_name} must be a lowercase SHA-256")
    return value


def equivalence_observations() -> np.ndarray:
    """Build the frozen four-observation affine grid."""

    indices = np.arange(4 * OBSERVATION_WIDTH, dtype=np.int64).reshape(
        EQUIVALENCE_OBSERVATION_SHAPE
    )
    values = (((indices * 37 + 11) % 257) - 128).astype("<f4") / np.float32(64.0)
    result = np.ascontiguousarray(values, dtype="<f4")
    if canonical_array_sha256(result) != EQUIVALENCE_OBSERVATION_SHA256:
        raise ExperimentContractError("TQC equivalence observation batch differs")
    return result


def _actor_arrays(actor: object) -> dict[str, np.ndarray]:
    import gymnasium as gym
    import torch
    from sb3_contrib.tqc.policies import Actor
    from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
    from stable_baselines3.common.torch_layers import FlattenExtractor

    if type(actor) is not Actor:
        raise ExperimentContractError("trusted actor must be an exact SB3-Contrib TQC Actor")
    if getattr(actor, "device", None) != torch.device("cpu"):
        raise ExperimentContractError("trusted actor must reside on CPU")
    observation_space = getattr(actor, "observation_space", None)
    action_space = getattr(actor, "action_space", None)
    if (
        type(observation_space) is not gym.spaces.Box
        or observation_space.shape != (OBSERVATION_WIDTH,)
        or observation_space.dtype != np.dtype("<f8")
        or type(action_space) is not gym.spaces.Box
        or action_space.shape != (ACTION_WIDTH,)
        or action_space.dtype != np.dtype("<f4")
    ):
        raise ExperimentContractError("trusted actor spaces differ from Humanoid-v5")
    latent_modules = tuple(getattr(actor, "latent_pi", ()))
    if (
        type(getattr(actor, "features_extractor", None)) is not FlattenExtractor
        or getattr(actor, "activation_fn", None) is not torch.nn.ReLU
        or getattr(actor, "use_sde", None) is not False
        or getattr(actor, "use_expln", None) is not False
        or type(getattr(actor, "action_dist", None)) is not SquashedDiagGaussianDistribution
        or tuple(type(module) for module in latent_modules)
        != (torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear, torch.nn.ReLU)
        or tuple(
            (module.in_features, module.out_features)
            for module in (latent_modules[0], latent_modules[2])
        )
        != ((OBSERVATION_WIDTH, 256), (256, 256))
        or type(getattr(actor, "mu", None)) is not torch.nn.Linear
        or (actor.mu.in_features, actor.mu.out_features) != (256, ACTION_WIDTH)
        or type(getattr(actor, "log_std", None)) is not torch.nn.Linear
        or (actor.log_std.in_features, actor.log_std.out_features) != (256, ACTION_WIDTH)
    ):
        raise ExperimentContractError("trusted actor architecture differs")
    values = {
        name: np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4")
        for name, tensor in actor.state_dict().items()
    }
    values.update(
        {
            "action_low": np.ascontiguousarray(action_space.low, dtype="<f4"),
            "action_high": np.ascontiguousarray(action_space.high, dtype="<f4"),
            "format_version": np.asarray([1], dtype="<i8"),
        }
    )
    return validate_actor_arrays(values)


def _with_seeded_cpu_rng(computation: object, *, sampling_seed: int) -> object:
    import torch

    if not callable(computation):
        raise ExperimentContractError("seeded computation must be callable")
    before = torch.random.get_rng_state().clone()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(sampling_seed)
    try:
        torch.random.set_rng_state(generator.get_state())
        return computation()
    finally:
        torch.random.set_rng_state(before)


def _seeded_sample(mean: object, log_std: object, *, sampling_seed: int) -> object:
    import torch

    return _with_seeded_cpu_rng(
        lambda: torch.tanh(torch.distributions.Normal(mean, torch.exp(log_std)).rsample()),
        sampling_seed=sampling_seed,
    )


def _trusted_outputs(
    actor: object, observations: np.ndarray, *, sampling_seed: int
) -> dict[str, np.ndarray]:
    import torch

    value = torch.from_numpy(observations.copy(order="C"))
    with torch.inference_mode():
        mean, log_std, kwargs = actor.get_action_dist_params(value)
        if kwargs:
            raise ExperimentContractError("trusted actor unexpectedly uses distribution kwargs")
        clamped_log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        distribution = actor.action_dist.proba_distribution(mean, clamped_log_std)
        outputs = {
            "mean": mean,
            "log_std": clamped_log_std,
            "deterministic_action": distribution.mode(),
            "seeded_sample": _with_seeded_cpu_rng(
                distribution.sample,
                sampling_seed=sampling_seed,
            ),
        }
    return {
        name: np.ascontiguousarray(value.detach().cpu().numpy(), dtype="<f4")
        for name, value in outputs.items()
    }


def _loaded_outputs(
    arrays: Mapping[str, np.ndarray],
    observations: np.ndarray,
    *,
    sampling_seed: int,
) -> dict[str, np.ndarray]:
    import torch
    from torch.nn import functional

    parameters = {
        name: torch.from_numpy(arrays[name].copy(order="C"))
        for name in (
            "latent_pi.0.weight",
            "latent_pi.0.bias",
            "latent_pi.2.weight",
            "latent_pi.2.bias",
            "mu.weight",
            "mu.bias",
            "log_std.weight",
            "log_std.bias",
        )
    }
    value = torch.from_numpy(observations.copy(order="C"))
    with torch.inference_mode():
        latent = functional.relu(
            functional.linear(
                value,
                parameters["latent_pi.0.weight"],
                parameters["latent_pi.0.bias"],
            )
        )
        latent = functional.relu(
            functional.linear(
                latent,
                parameters["latent_pi.2.weight"],
                parameters["latent_pi.2.bias"],
            )
        )
        mean = functional.linear(
            latent,
            parameters["mu.weight"],
            parameters["mu.bias"],
        )
        log_std = torch.clamp(
            functional.linear(
                latent,
                parameters["log_std.weight"],
                parameters["log_std.bias"],
            ),
            min=LOG_STD_MIN,
            max=LOG_STD_MAX,
        )
        outputs = {
            "mean": mean,
            "log_std": log_std,
            "deterministic_action": torch.tanh(mean),
            "seeded_sample": _seeded_sample(mean, log_std, sampling_seed=sampling_seed),
        }
    return {
        name: np.ascontiguousarray(value.detach().cpu().numpy(), dtype="<f4")
        for name, value in outputs.items()
    }


def _validate_outputs(outputs: Mapping[str, np.ndarray], *, label: str) -> None:
    if tuple(outputs) != _OUTPUT_NAMES:
        raise ExperimentContractError(f"{label} actor outputs differ from the schema")
    for name, value in outputs.items():
        if (
            not isinstance(value, np.ndarray)
            or value.shape != (4, ACTION_WIDTH)
            or value.dtype != np.dtype("<f4")
            or not value.flags.c_contiguous
            or not np.isfinite(value).all()
        ):
            raise ExperimentContractError(f"{label} actor {name} output is invalid")


def _validate_optimizer(
    optimizer: object,
    *,
    expected_parameters: tuple[object, ...],
    label: str,
    expected_epsilon: float,
) -> None:
    import torch

    param_groups = getattr(optimizer, "param_groups", None)
    state = getattr(optimizer, "state", None)
    if (
        type(optimizer) is not torch.optim.Adam
        or type(param_groups) is not list
        or len(param_groups) != 1
        or not isinstance(state, dict)
    ):
        raise ExperimentContractError(f"{label} optimizer structure differs")
    group = param_groups[0]
    expected_group = {
        "lr": 0.0003,
        "betas": (0.9, 0.999),
        "eps": expected_epsilon,
        "weight_decay": 0,
        "amsgrad": False,
        "maximize": False,
        "foreach": None,
        "capturable": False,
        "differentiable": False,
        "fused": None,
        "decoupled_weight_decay": False,
    }
    if set(group) != {*expected_group, "params"} or any(
        group[name] != value for name, value in expected_group.items()
    ):
        raise ExperimentContractError(f"{label} optimizer parameter group differs")
    owned = tuple(parameter for group in param_groups for parameter in group.get("params", ()))
    if (
        len(owned) != len(expected_parameters)
        or len({id(parameter) for parameter in owned}) != len(owned)
        or any(left is not right for left, right in zip(owned, expected_parameters, strict=True))
        or set(state) != set(expected_parameters)
    ):
        raise ExperimentContractError(f"{label} optimizer ownership differs")
    for parameter in expected_parameters:
        values = state[parameter]
        if type(values) is not dict or set(values) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ExperimentContractError(f"{label} optimizer state fields differ")
        step = values["step"]
        if (
            not isinstance(step, torch.Tensor)
            or step.numel() != 1
            or not bool(torch.isfinite(step).all().item())
            or float(step.item()) != float(FINAL_UPDATE_COUNT)
        ):
            raise ExperimentContractError(f"{label} optimizer step differs")
        for field_name in ("exp_avg", "exp_avg_sq"):
            value = values[field_name]
            if (
                not isinstance(value, torch.Tensor)
                or value.shape != parameter.shape
                or value.dtype != parameter.dtype
                or value.device != parameter.device
                or not bool(torch.isfinite(value).all().item())
            ):
                raise ExperimentContractError(f"{label} optimizer {field_name} differs")


def _expected_policy_state_schema() -> tuple[tuple[str, tuple[int, ...]], ...]:
    schema: list[tuple[str, tuple[int, ...]]] = [
        ("actor.latent_pi.0.weight", (256, 348)),
        ("actor.latent_pi.0.bias", (256,)),
        ("actor.latent_pi.2.weight", (256, 256)),
        ("actor.latent_pi.2.bias", (256,)),
        ("actor.mu.weight", (17, 256)),
        ("actor.mu.bias", (17,)),
        ("actor.log_std.weight", (17, 256)),
        ("actor.log_std.bias", (17,)),
    ]
    for role in ("critic", "critic_target"):
        for critic_index in range(2):
            prefix = f"{role}.qf{critic_index}"
            schema.extend(
                (
                    (f"{prefix}.0.weight", (256, 365)),
                    (f"{prefix}.0.bias", (256,)),
                    (f"{prefix}.2.weight", (256, 256)),
                    (f"{prefix}.2.bias", (256,)),
                    (f"{prefix}.4.weight", (25, 256)),
                    (f"{prefix}.4.bias", (25,)),
                )
            )
    return tuple(schema)


def _validate_final_model_state(model: object) -> object:
    import torch
    from sb3_contrib import TQC
    from sb3_contrib.tqc.policies import TQCPolicy

    policy = getattr(model, "policy", None)
    actor = getattr(model, "actor", None)
    critic = getattr(model, "critic", None)
    if (
        type(model) is not TQC
        or type(policy) is not TQCPolicy
        or getattr(policy, "actor", None) is not actor
        or getattr(policy, "critic", None) is not critic
        or getattr(model, "critic_target", None) is not getattr(policy, "critic_target", None)
    ):
        raise ExperimentContractError("final checkpoint must be an exact TQC model")
    if (
        type(getattr(model, "num_timesteps", None)) is not int
        or model.num_timesteps != FINAL_ENVIRONMENT_STEP
        or type(getattr(model, "_n_updates", None)) is not int
        or model._n_updates != FINAL_UPDATE_COUNT
    ):
        raise ExperimentContractError("TQC model is not at the frozen final checkpoint")
    policy_state = policy.state_dict()
    observed_schema = tuple((name, tuple(value.shape)) for name, value in policy_state.items())
    if (
        observed_schema != _expected_policy_state_schema()
        or sum(value.numel() for value in policy_state.values()) != 827_526
        or any(value.dtype != torch.float32 for value in policy_state.values())
        or any(not bool(value.detach().isfinite().all().item()) for value in policy_state.values())
        or getattr(model, "learning_rate", None) != 0.0003
        or getattr(model, "target_entropy", None) != -17.0
        or getattr(model, "top_quantiles_to_drop_per_net", None) != 2
        or getattr(model, "gradient_steps", None) != 1
        or getattr(model, "n_steps", None) != 1
        or getattr(model, "use_sde", None) is not False
        or getattr(model, "action_noise", "missing") is not None
        or getattr(critic, "n_quantiles", None) != 25
        or getattr(critic, "n_critics", None) != 2
    ):
        raise ExperimentContractError("final TQC policy state differs or is non-finite")
    entropy = getattr(model, "log_ent_coef", None)
    if (
        not isinstance(entropy, torch.Tensor)
        or entropy.numel() != 1
        or not bool(entropy.detach().isfinite().all().item())
    ):
        raise ExperimentContractError("final TQC entropy state differs or is non-finite")
    _validate_optimizer(
        getattr(actor, "optimizer", None),
        expected_parameters=tuple(actor.parameters()),
        label="actor",
        expected_epsilon=1e-5,
    )
    _validate_optimizer(
        getattr(critic, "optimizer", None),
        expected_parameters=tuple(critic.parameters()),
        label="critic",
        expected_epsilon=1e-5,
    )
    _validate_optimizer(
        getattr(model, "ent_coef_optimizer", None),
        expected_parameters=(entropy,),
        label="entropy",
        expected_epsilon=1e-8,
    )
    return actor


def _exact_tree_equal(left: object, right: object) -> bool:
    import torch

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and tuple(left) == tuple(right)
            and all(_exact_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(
                _exact_tree_equal(left_value, right_value)
                for left_value, right_value in zip(left, right, strict=True)
            )
        )
    return type(left) is type(right) and left == right


def _validate_persisted_model(
    model: object,
    artifact: PublishedArtifact,
) -> None:
    from stable_baselines3.common.save_util import load_from_zip_file

    if type(artifact) is not PublishedArtifact:
        raise ExperimentContractError("trusted model requires a published artifact record")
    if (
        type(artifact.byte_count) is not int
        or not 0 < artifact.byte_count <= MAX_TRUSTED_MODEL_ARCHIVE_BYTES
    ):
        raise ExperimentContractError("trusted model artifact byte count differs")
    _canonical_sha256(artifact.sha256, field_name="trusted model artifact SHA-256")
    payload = read_verified_artifact_bytes(
        Path(artifact.path),
        expected_sha256=artifact.sha256,
        expected_size=artifact.byte_count,
        max_bytes=MAX_TRUSTED_MODEL_ARCHIVE_BYTES,
    )
    data, parameters, pytorch_variables = load_from_zip_file(
        io.BytesIO(payload),
        device="cpu",
        print_system_info=False,
    )
    if not isinstance(data, dict) or not isinstance(parameters, dict):
        raise ExperimentContractError("trusted model archive structure differs")
    if (
        type(data.get("num_timesteps")) is not int
        or data["num_timesteps"] != FINAL_ENVIRONMENT_STEP
        or type(data.get("_n_updates")) is not int
        or data["_n_updates"] != FINAL_UPDATE_COUNT
    ):
        raise ExperimentContractError("trusted model archive counters differ")
    expected_parameters = {
        "policy": model.policy.state_dict(),
        "actor.optimizer": model.actor.optimizer.state_dict(),
        "critic.optimizer": model.critic.optimizer.state_dict(),
        "ent_coef_optimizer": model.ent_coef_optimizer.state_dict(),
    }
    expected_variables = {"log_ent_coef": model.log_ent_coef}
    if not _exact_tree_equal(parameters, expected_parameters) or not _exact_tree_equal(
        pytorch_variables,
        expected_variables,
    ):
        raise ExperimentContractError("trusted model archive state differs from memory")


@dataclass(slots=True)
class _FinalTQCActorSealV2:
    """Bind one canonical actor authority to its persistence capability."""

    _payload: bytes = field(repr=False)
    _creator_pid: int
    _persistence_identity: int
    _actor_identity: int
    _loaded_actor_identity: int
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _AUTHORITY_SEAL_ISSUER:
            raise ExperimentContractError("final TQC actor seals are private")
        if (
            not self._payload
            or self._creator_pid != os.getpid()
            or self._persistence_identity <= 0
            or self._actor_identity <= 0
            or self._loaded_actor_identity <= 0
        ):
            raise ExperimentContractError("final TQC actor seal is invalid")

    def validate(
        self,
        payload: Mapping[str, object],
        *,
        persistence_authority: object,
        actor: object,
        loaded_actor: object,
    ) -> None:
        if (
            os.getpid() != self._creator_pid
            or id(persistence_authority) != self._persistence_identity
            or id(actor) != self._actor_identity
            or id(loaded_actor) != self._loaded_actor_identity
            or _canonical_json(dict(payload)) != self._payload
        ):
            raise ExperimentContractError("final TQC actor authority seal differs")


@dataclass(slots=True)
class _TQCActorEquivalenceSealV2:
    """Bind one immutable receipt payload without replaying private issuance."""

    _payload: bytes = field(repr=False)
    _creator_pid: int
    _authority_identity: int
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _RECEIPT_SEAL_ISSUER:
            raise ExperimentContractError("actor equivalence seals are private")
        if not self._payload or self._creator_pid != os.getpid() or self._authority_identity <= 0:
            raise ExperimentContractError("actor equivalence seal is invalid")

    def validate(
        self,
        payload: Mapping[str, object],
        *,
        actor_authority: object,
    ) -> None:
        if (
            os.getpid() != self._creator_pid
            or id(actor_authority) != self._authority_identity
            or _canonical_json(dict(payload)) != self._payload
        ):
            raise ExperimentContractError("actor equivalence receipt seal differs")


@dataclass(frozen=True, slots=True)
class FinalTQCActorAuthority:
    """Process-local capability for the exact persisted final TQC actor."""

    attempt_id: str
    execution_manifest_sha256: str
    claimed_work_directory_identity: str
    design_file_sha256: str
    design_semantic_sha256: str
    training_projection_sha256: str
    training_integrity_sha256: str
    persistence_sha256: str
    trusted_model_artifact_sha256: str
    trusted_model_artifact_byte_count: int
    strict_actor_artifact_sha256: str
    strict_actor_artifact_byte_count: int
    source_environment_step: int
    source_update_count: int
    actor_state_sha256: str
    actor: object = field(repr=False, compare=False)
    loaded_actor: LoadedTQCActor = field(repr=False, compare=False)
    persistence_authority: TQCPersistenceAuthorityV2 = field(repr=False, compare=False)
    _seal: _FinalTQCActorSealV2 = field(repr=False, compare=False)
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _AUTHORITY_ISSUER:
            raise ExperimentContractError(
                "final TQC actor authority may only be issued by checkpoint admission"
            )
        self._validate_sealed()

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name
            not in {
                "actor",
                "loaded_actor",
                "persistence_authority",
                "_seal",
                "_issuer",
            }
        }

    def _validate_sealed(self) -> None:
        if type(self._seal) is not _FinalTQCActorSealV2:
            raise ExperimentContractError("final TQC actor authority seal is unavailable")
        self._seal.validate(
            self.to_dict(),
            persistence_authority=self.persistence_authority,
            actor=self.actor,
            loaded_actor=self.loaded_actor,
        )
        persistence = self.persistence_authority
        if type(persistence) is not TQCPersistenceAuthorityV2:
            raise ExperimentContractError("final actor persistence authority differs")
        persistence._validate_lineage_seal()
        for name in (
            "execution_manifest_sha256",
            "claimed_work_directory_identity",
            "design_file_sha256",
            "design_semantic_sha256",
            "training_projection_sha256",
            "training_integrity_sha256",
            "persistence_sha256",
            "trusted_model_artifact_sha256",
            "strict_actor_artifact_sha256",
            "actor_state_sha256",
        ):
            _canonical_sha256(getattr(self, name), field_name=name)
        if (
            type(self.attempt_id) is not str
            or not self.attempt_id
            or self.attempt_id != persistence.attempt_id
            or self.execution_manifest_sha256 != persistence.execution_manifest_sha256
            or self.claimed_work_directory_identity != persistence.claimed_work_directory_identity
            or self.design_file_sha256 != persistence.design_file_sha256
            or self.design_semantic_sha256 != persistence.design_semantic_sha256
            or self.training_projection_sha256 != persistence.training_projection_sha256
            or self.training_integrity_sha256 != persistence.training_integrity_sha256
            or self.persistence_sha256 != persistence.persistence_sha256
            or self.trusted_model_artifact_sha256 != persistence.model_artifact.sha256
            or self.trusted_model_artifact_byte_count != persistence.model_artifact.byte_count
            or self.strict_actor_artifact_sha256 != persistence.actor_artifact.sha256
            or self.strict_actor_artifact_byte_count != persistence.actor_artifact.byte_count
            or self.loaded_actor is not persistence.loaded_actor
            or self.actor_state_sha256 != persistence.source_actor_state_sha256
            or self.actor_state_sha256 != self.loaded_actor.state_sha256
            or type(self.trusted_model_artifact_byte_count) is not int
            or not 0 < self.trusted_model_artifact_byte_count <= MAX_TRUSTED_MODEL_ARCHIVE_BYTES
            or type(self.strict_actor_artifact_byte_count) is not int
            or self.strict_actor_artifact_byte_count <= 0
            or type(self.source_environment_step) is not int
            or self.source_environment_step != FINAL_ENVIRONMENT_STEP
            or type(self.source_update_count) is not int
            or self.source_update_count != FINAL_UPDATE_COUNT
        ):
            raise ExperimentContractError("final TQC actor authority is inconsistent")
        if actor_state_sha256(_actor_arrays(self.actor)) != self.actor_state_sha256:
            raise ExperimentContractError("final TQC actor state differs from its authority")


def admit_final_tqc_checkpoint_actor(
    persistence_authority: TQCPersistenceAuthorityV2,
) -> FinalTQCActorAuthority:
    """Admit only the exact same-attempt persistence capability."""

    if type(persistence_authority) is not TQCPersistenceAuthorityV2:
        raise ExperimentContractError(
            "final actor admission requires exact TQC persistence authority"
        )
    model = revalidate_tqc_persistence_authority_v2(persistence_authority)
    actor = _validate_final_model_state(model)
    _validate_persisted_model(model, persistence_authority.model_artifact)
    arrays = _actor_arrays(actor)
    actor_state = actor_state_sha256(arrays)
    if (
        actor_state != persistence_authority.source_actor_state_sha256
        or actor_state != persistence_authority.loaded_actor.state_sha256
    ):
        raise ExperimentContractError("persisted TQC actor state binding differs")
    payload: dict[str, object] = {
        "attempt_id": persistence_authority.attempt_id,
        "execution_manifest_sha256": persistence_authority.execution_manifest_sha256,
        "claimed_work_directory_identity": (persistence_authority.claimed_work_directory_identity),
        "design_file_sha256": persistence_authority.design_file_sha256,
        "design_semantic_sha256": persistence_authority.design_semantic_sha256,
        "training_projection_sha256": persistence_authority.training_projection_sha256,
        "training_integrity_sha256": persistence_authority.training_integrity_sha256,
        "persistence_sha256": persistence_authority.persistence_sha256,
        "trusted_model_artifact_sha256": persistence_authority.model_artifact.sha256,
        "trusted_model_artifact_byte_count": persistence_authority.model_artifact.byte_count,
        "strict_actor_artifact_sha256": persistence_authority.actor_artifact.sha256,
        "strict_actor_artifact_byte_count": persistence_authority.actor_artifact.byte_count,
        "source_environment_step": model.num_timesteps,
        "source_update_count": model._n_updates,
        "actor_state_sha256": actor_state,
    }
    seal = _FinalTQCActorSealV2(
        _payload=_canonical_json(payload),
        _creator_pid=os.getpid(),
        _persistence_identity=id(persistence_authority),
        _actor_identity=id(actor),
        _loaded_actor_identity=id(persistence_authority.loaded_actor),
        _issuer=_AUTHORITY_SEAL_ISSUER,
    )
    return FinalTQCActorAuthority(
        **payload,
        actor=actor,
        loaded_actor=persistence_authority.loaded_actor,
        persistence_authority=persistence_authority,
        _seal=seal,
        _issuer=_AUTHORITY_ISSUER,
    )


@dataclass(frozen=True, slots=True)
class TQCActorEquivalenceReceipt:
    equivalence_id: str
    attempt_id: str
    execution_manifest_sha256: str
    claimed_work_directory_identity: str
    design_file_sha256: str
    design_semantic_sha256: str
    training_projection_sha256: str
    training_integrity_sha256: str
    persistence_sha256: str
    trusted_model_artifact_sha256: str
    trusted_model_artifact_byte_count: int
    source_environment_step: int
    source_update_count: int
    trusted_actor_state_sha256: str
    loaded_actor_state_sha256: str
    observation_sha256: str
    sampling_seed: int
    trusted_mean_sha256: str
    loaded_mean_sha256: str
    trusted_log_std_sha256: str
    loaded_log_std_sha256: str
    trusted_deterministic_action_sha256: str
    loaded_deterministic_action_sha256: str
    trusted_seeded_sample_sha256: str
    loaded_seeded_sample_sha256: str
    cpu_rng_state_before_sha256: str
    cpu_rng_state_after_sha256: str
    passed: bool
    actor_authority: FinalTQCActorAuthority = field(repr=False, compare=False)
    _seal: _TQCActorEquivalenceSealV2 = field(repr=False, compare=False)
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _RECEIPT_ISSUER:
            raise ExperimentContractError(
                "actor equivalence receipt may only be issued by the verifier"
            )
        self._validate_sealed()

    def to_dict(self) -> dict[str, object]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"actor_authority", "_seal", "_issuer"}
        }

    def _validate_sealed(self) -> None:
        if type(self._seal) is not _TQCActorEquivalenceSealV2:
            raise ExperimentContractError("actor equivalence receipt seal is unavailable")
        self._seal.validate(self.to_dict(), actor_authority=self.actor_authority)
        authority = self.actor_authority
        if type(authority) is not FinalTQCActorAuthority:
            raise ExperimentContractError("actor equivalence authority differs")
        authority._validate_sealed()
        for field_name in (
            "execution_manifest_sha256",
            "claimed_work_directory_identity",
            "design_file_sha256",
            "design_semantic_sha256",
            "training_projection_sha256",
            "training_integrity_sha256",
            "persistence_sha256",
            "trusted_model_artifact_sha256",
            "trusted_actor_state_sha256",
            "loaded_actor_state_sha256",
            "observation_sha256",
            "trusted_mean_sha256",
            "loaded_mean_sha256",
            "trusted_log_std_sha256",
            "loaded_log_std_sha256",
            "trusted_deterministic_action_sha256",
            "loaded_deterministic_action_sha256",
            "trusted_seeded_sample_sha256",
            "loaded_seeded_sample_sha256",
            "cpu_rng_state_before_sha256",
            "cpu_rng_state_after_sha256",
        ):
            _canonical_sha256(getattr(self, field_name), field_name=field_name)
        if (
            self.equivalence_id != EQUIVALENCE_ID
            or type(self.attempt_id) is not str
            or not self.attempt_id
            or self.attempt_id != authority.attempt_id
            or self.execution_manifest_sha256 != authority.execution_manifest_sha256
            or self.claimed_work_directory_identity != authority.claimed_work_directory_identity
            or self.design_file_sha256 != authority.design_file_sha256
            or self.design_semantic_sha256 != authority.design_semantic_sha256
            or self.training_projection_sha256 != authority.training_projection_sha256
            or self.training_integrity_sha256 != authority.training_integrity_sha256
            or self.persistence_sha256 != authority.persistence_sha256
            or self.sampling_seed != EQUIVALENCE_SAMPLING_SEED
            or self.source_environment_step != FINAL_ENVIRONMENT_STEP
            or self.source_update_count != FINAL_UPDATE_COUNT
            or type(self.trusted_model_artifact_byte_count) is not int
            or not 0 < self.trusted_model_artifact_byte_count <= MAX_TRUSTED_MODEL_ARCHIVE_BYTES
            or self.trusted_actor_state_sha256 != self.loaded_actor_state_sha256
            or self.trusted_mean_sha256 != self.loaded_mean_sha256
            or self.trusted_log_std_sha256 != self.loaded_log_std_sha256
            or self.trusted_deterministic_action_sha256 != self.loaded_deterministic_action_sha256
            or self.trusted_seeded_sample_sha256 != self.loaded_seeded_sample_sha256
            or self.cpu_rng_state_before_sha256 != self.cpu_rng_state_after_sha256
            or self.passed is not True
        ):
            raise ExperimentContractError("actor equivalence receipt is inconsistent")


def verify_actor_equivalence_v2(
    authority: FinalTQCActorAuthority,
    *,
    sampling_seed: int = EQUIVALENCE_SAMPLING_SEED,
) -> TQCActorEquivalenceReceipt:
    """Require byte-exact distribution outputs without changing CPU RNG state."""

    import torch

    if type(authority) is not FinalTQCActorAuthority:
        raise ExperimentContractError("trusted actor requires final checkpoint authority")
    authority._validate_sealed()
    loaded_actor = authority.loaded_actor
    if type(loaded_actor) is not LoadedTQCActor:
        raise ExperimentContractError("loaded actor must come from exact persistence")
    if type(sampling_seed) is not int or sampling_seed != EQUIVALENCE_SAMPLING_SEED:
        raise ExperimentContractError("actor equivalence sampling seed differs")
    trusted_actor = authority.actor
    trusted_arrays = _actor_arrays(trusted_actor)
    trusted_state_sha256 = actor_state_sha256(trusted_arrays)
    if loaded_actor.state_sha256 != actor_state_sha256(loaded_actor.arrays):
        raise ExperimentContractError("strict-loaded actor state changed after admission")
    if trusted_state_sha256 != loaded_actor.state_sha256:
        raise ExperimentContractError("trusted and strict-loaded actor states differ")
    if trusted_state_sha256 != authority.actor_state_sha256:
        raise ExperimentContractError("trusted actor changed after checkpoint admission")
    observations = equivalence_observations()
    rng_before = torch.random.get_rng_state().clone()
    trusted_outputs = _trusted_outputs(
        trusted_actor,
        observations,
        sampling_seed=sampling_seed,
    )
    loaded_outputs = _loaded_outputs(
        loaded_actor.arrays,
        observations,
        sampling_seed=sampling_seed,
    )
    rng_after = torch.random.get_rng_state().clone()
    _validate_outputs(trusted_outputs, label="trusted")
    _validate_outputs(loaded_outputs, label="loaded")
    hashes: dict[str, str] = {}
    for name in _OUTPUT_NAMES:
        trusted = trusted_outputs[name]
        loaded = loaded_outputs[name]
        if trusted.tobytes(order="C") != loaded.tobytes(order="C"):
            raise ExperimentContractError(f"trusted and strict-loaded actor {name} bytes differ")
        hashes[f"trusted_{name}_sha256"] = canonical_array_sha256(trusted)
        hashes[f"loaded_{name}_sha256"] = canonical_array_sha256(loaded)
    final_trusted_state_sha256 = actor_state_sha256(_actor_arrays(trusted_actor))
    if final_trusted_state_sha256 != trusted_state_sha256:
        raise ExperimentContractError("trusted actor changed during equivalence verification")
    rng_before_sha256 = hashlib.sha256(rng_before.numpy().tobytes(order="C")).hexdigest()
    rng_after_sha256 = hashlib.sha256(rng_after.numpy().tobytes(order="C")).hexdigest()
    if rng_before_sha256 != rng_after_sha256 or not torch.equal(rng_before, rng_after):
        raise ExperimentContractError("CPU RNG state changed during equivalence verification")
    payload: dict[str, object] = {
        "equivalence_id": EQUIVALENCE_ID,
        "attempt_id": authority.attempt_id,
        "execution_manifest_sha256": authority.execution_manifest_sha256,
        "claimed_work_directory_identity": authority.claimed_work_directory_identity,
        "design_file_sha256": authority.design_file_sha256,
        "design_semantic_sha256": authority.design_semantic_sha256,
        "training_projection_sha256": authority.training_projection_sha256,
        "training_integrity_sha256": authority.training_integrity_sha256,
        "persistence_sha256": authority.persistence_sha256,
        "trusted_model_artifact_sha256": authority.trusted_model_artifact_sha256,
        "trusted_model_artifact_byte_count": authority.trusted_model_artifact_byte_count,
        "source_environment_step": authority.source_environment_step,
        "source_update_count": authority.source_update_count,
        "trusted_actor_state_sha256": trusted_state_sha256,
        "loaded_actor_state_sha256": loaded_actor.state_sha256,
        "observation_sha256": canonical_array_sha256(observations),
        "sampling_seed": sampling_seed,
        "trusted_mean_sha256": hashes["trusted_mean_sha256"],
        "loaded_mean_sha256": hashes["loaded_mean_sha256"],
        "trusted_log_std_sha256": hashes["trusted_log_std_sha256"],
        "loaded_log_std_sha256": hashes["loaded_log_std_sha256"],
        "trusted_deterministic_action_sha256": hashes["trusted_deterministic_action_sha256"],
        "loaded_deterministic_action_sha256": hashes["loaded_deterministic_action_sha256"],
        "trusted_seeded_sample_sha256": hashes["trusted_seeded_sample_sha256"],
        "loaded_seeded_sample_sha256": hashes["loaded_seeded_sample_sha256"],
        "cpu_rng_state_before_sha256": rng_before_sha256,
        "cpu_rng_state_after_sha256": rng_after_sha256,
        "passed": True,
    }
    seal = _TQCActorEquivalenceSealV2(
        _payload=_canonical_json(payload),
        _creator_pid=os.getpid(),
        _authority_identity=id(authority),
        _issuer=_RECEIPT_SEAL_ISSUER,
    )
    return TQCActorEquivalenceReceipt(
        **payload,
        actor_authority=authority,
        _seal=seal,
        _issuer=_RECEIPT_ISSUER,
    )


def revalidate_tqc_actor_equivalence_receipt_v2(
    receipt: TQCActorEquivalenceReceipt,
) -> TQCActorEquivalenceReceipt:
    """Revalidate an issued receipt without access to its private issuer."""

    if type(receipt) is not TQCActorEquivalenceReceipt:
        raise ExperimentContractError("actor equivalence result must be the exact receipt")
    receipt._validate_sealed()
    return receipt


__all__ = [
    "EQUIVALENCE_ID",
    "EQUIVALENCE_OBSERVATION_SHA256",
    "EQUIVALENCE_SAMPLING_SEED",
    "FinalTQCActorAuthority",
    "TQCActorEquivalenceReceipt",
    "admit_final_tqc_checkpoint_actor",
    "canonical_array_sha256",
    "equivalence_observations",
    "revalidate_tqc_actor_equivalence_receipt_v2",
    "verify_actor_equivalence_v2",
]
