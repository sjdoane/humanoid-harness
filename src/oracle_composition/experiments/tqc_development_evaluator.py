"""Protected deterministic evaluator for the frozen TQC development screen."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np

from oracle_composition.envs import humanoid as humanoid_module
from oracle_composition.envs.humanoid import (
    CONTACT_CAPTURE_ID,
    HumanoidExperimentConfig,
    SubstepContactSample,
    make_humanoid_env,
)
from oracle_composition.traces.contract import (
    ArtifactBinding,
    EvidenceClass,
    MissingReason,
    MissingSignal,
    NumericSignalSpec,
    TraceRecorder,
    TraceRole,
    TrajectoryTrace,
)
from oracle_composition.tracking.humanoid_reference import (
    HumanoidActuatorABI,
    validate_humanoid_actuator_abi,
)

from . import tqc_development_contract as development_contract_module
from .fixed_reference import ExperimentContractError, sha256_file
from .protected_evaluator import FLOOR_GEOM_NAME, PERMITTED_STATIC_STAND_FLOOR_GEOMS, ContactFact
from .protected_runtime import direct_contact_facts, finite_direct_vector
from .runtime_identity import dependency_lock_path, module_sha256, source_tree_sha256, space_sha256
from .tqc_actor_npz import (
    ACTION_WIDTH,
    MAX_ARCHIVE_BYTES,
    OBSERVATION_WIDTH,
    LoadedTQCActor,
    actor_schema_sha256,
    actor_state_sha256,
    encode_actor_npz,
    validate_actor_arrays,
)
from .tqc_calibration_contract import canonical_json, sha256_json
from .tqc_development_contract import EVALUATION_SEEDS, LoadedTQCDevelopmentDesign
from .tqc_development_metrics import (
    CONTROL_PERIOD_SECONDS,
    EXPECTED_STEPS,
    MINIMUM_TORSO_UP_Z,
    TQCDevelopmentCohortMetrics,
    TQCDevelopmentEpisodeAccumulator,
    TQCDevelopmentEpisodeMetrics,
    TQCDevelopmentStepFacts,
    summarize_tqc_development_cohort,
)

EVALUATOR_ID = "protected_tqc_humanoid_development_evaluator/v1"
CLAIM_BOUNDARY = (
    "excluded_development_metrics_and_trace_only_no_checkpoint_provenance_or_tracker_claim/v1"
)
METRIC_SOURCE = "direct_MuJoCo_state_control_and_contact_readers/v1"
EXPECTED_WRAPPER_TYPES = (
    "gymnasium.wrappers.common.TimeLimit",
    "gymnasium.wrappers.common.OrderEnforcing",
    "gymnasium.wrappers.common.PassiveEnvChecker",
    "oracle_composition.envs.humanoid.SubstepContactHumanoidEnv",
)
EXPECTED_OBSERVATION_SPACE_SHA256 = (
    "07953989de29452db64aa5188067084bf881761072f9d9b1b9e45b442d38506c"
)
EXPECTED_ACTION_SPACE_SHA256 = "5a2389149db1f0253571a4f07f4f83285844f18c529355502101d5b50030cfb3"
EQUIVALENCE_OBSERVATION_SHA256 = "0c6a81b06a88cab7eca0255e75f021008b60025c4ddc4d3719426e3647159ec6"
EQUIVALENCE_OBSERVATION_SHAPE = (4, OBSERVATION_WIDTH)
EXPECTED_DEVELOPMENT_DESIGN_FILE_SHA256 = (
    "41c58694093070d7994aca5d1e8507832b40302fda088e4a1d42f26db7f5ca49"
)
EXPECTED_DEVELOPMENT_DESIGN_SEMANTIC_SHA256 = (
    "8a2f157dccde1d46f85ff8c30c4e7390379d4ea7854b9537c8d9a29a2e40344f"
)
PHYSICS_SUBSTEPS_PER_CONTROL = 5
CONTACT_EVENT_TYPE = "contact.active"
CONTACT_EVENT_SOURCE = "direct_mujoco_substep_geom_pair_and_normal_force/v1"
TERMINAL_EVENT_TYPE = "episode.truncated"
TERMINAL_EVENT_SOURCE = "Gymnasium_TimeLimit_exact_step_1000/v1"
EXPECTED_HUMANOID_GEOM_NAMES = (
    "floor",
    "torso1",
    "head",
    "uwaist",
    "lwaist",
    "butt",
    "right_thigh1",
    "right_shin1",
    "right_foot",
    "left_thigh1",
    "left_shin1",
    "left_foot",
    "right_uarm1",
    "right_larm",
    "right_hand",
    "left_uarm1",
    "left_larm",
    "left_hand",
)
_CONTACT_EVENT_KEYS = {
    "physics_substep_index",
    "contact_index_within_substep",
    "geom1_id",
    "geom2_id",
    "geom1_name",
    "geom2_name",
    "normal_force_n",
}

# Process-local constructor capabilities prevent accidental record forgery.
# They are not cryptographic boundaries against trusted in-process code.
_ACTOR_ISSUER = object()
_RUNTIME_ISSUER = object()
_SEED_EVALUATION_ISSUER = object()
_EVALUATION_ISSUER = object()


def _canonical_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"{field} must be a lowercase SHA-256")
    return value


def _array_sha256(value: np.ndarray) -> str:
    if not isinstance(value, np.ndarray) or not value.flags.c_contiguous:
        raise ExperimentContractError("hashed array must use C-order storage")
    metadata = canonical_json({"dtype": value.dtype.str, "shape": list(value.shape)})
    raw = value.tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, child in pairs:
        if key in value:
            raise ExperimentContractError(f"development design contains duplicate key {key!r}")
        value[key] = child
    return value


def _reject_nonfinite_constant(value: str) -> None:
    raise ExperimentContractError(f"development design contains non-finite number {value!r}")


def _validated_design_payload(design: object) -> dict[str, object]:
    """Revalidate bytes and both identities; the wrapper type is not authority."""

    if type(design) is not LoadedTQCDevelopmentDesign:
        raise ExperimentContractError("development design must already be validated")
    encoded = design.encoded_bytes
    if (
        type(encoded) is not bytes
        or not encoded
        or len(encoded) > development_contract_module.MAX_DESIGN_BYTES
    ):
        raise ExperimentContractError("development design bytes are absent or oversized")
    try:
        raw = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except ExperimentContractError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ExperimentContractError("development design bytes are invalid JSON") from exc
    if not isinstance(raw, dict):
        raise ExperimentContractError("development design root must be an object")
    development_contract_module._validate_design(raw)
    file_sha256 = hashlib.sha256(encoded).hexdigest()
    semantic_sha256 = sha256_json(raw)
    _canonical_sha256(design.file_sha256, field="development design file SHA-256")
    _canonical_sha256(design.semantic_sha256, field="development design semantic SHA-256")
    if design.file_sha256 != file_sha256 or design.semantic_sha256 != semantic_sha256:
        raise ExperimentContractError("development design identity differs from its exact bytes")
    if (
        file_sha256 != EXPECTED_DEVELOPMENT_DESIGN_FILE_SHA256
        or semantic_sha256 != EXPECTED_DEVELOPMENT_DESIGN_SEMANTIC_SHA256
    ):
        raise ExperimentContractError("development design is not the reviewed canonical artifact")
    return raw


def _equivalence_observations() -> np.ndarray:
    indices = np.arange(4 * OBSERVATION_WIDTH, dtype=np.int64).reshape(
        EQUIVALENCE_OBSERVATION_SHAPE
    )
    values = (((indices * 37 + 11) % 257) - 128).astype("<f4") / np.float32(64.0)
    result = np.ascontiguousarray(values, dtype="<f4")
    if _array_sha256(result) != EQUIVALENCE_OBSERVATION_SHA256:
        raise ExperimentContractError("TQC equivalence observation batch differs")
    return result


def _immutable_actor_arrays(values: Mapping[str, object]) -> Mapping[str, np.ndarray]:
    validated = validate_actor_arrays(values)
    immutable = {
        name: np.frombuffer(array.tobytes(order="C"), dtype=array.dtype).reshape(array.shape)
        for name, array in validated.items()
    }
    return MappingProxyType(immutable)


@dataclass(frozen=True, slots=True)
class DeterministicTQCAction:
    """One exact normalized action and its Humanoid physical transform."""

    normalized: np.ndarray = field(repr=False)
    physical: np.ndarray = field(repr=False)

    def __post_init__(self) -> None:
        normalized = np.asarray(self.normalized)
        physical = np.asarray(self.physical)
        if normalized.shape != (ACTION_WIDTH,) or normalized.dtype != np.dtype("<f4"):
            raise ExperimentContractError("normalized actor action must be float32 shape (17,)")
        if physical.shape != (ACTION_WIDTH,) or physical.dtype != np.dtype("<f4"):
            raise ExperimentContractError("physical actor action must be float32 shape (17,)")
        if not np.isfinite(normalized).all() or not np.isfinite(physical).all():
            raise ExperimentContractError("actor action contains a non-finite value")
        if np.any(normalized < -1.0) or np.any(normalized > 1.0):
            raise ExperimentContractError("normalized actor action lies outside [-1, 1]")
        normalized_copy = np.frombuffer(normalized.tobytes(), dtype="<f4")
        physical_copy = np.frombuffer(physical.tobytes(), dtype="<f4")
        object.__setattr__(self, "normalized", normalized_copy)
        object.__setattr__(self, "physical", physical_copy)


class _TorchTQCInference:
    """Trusted installed-runtime inference over validated data-only arrays."""

    _PARAMETER_NAMES = (
        "latent_pi.0.weight",
        "latent_pi.0.bias",
        "latent_pi.2.weight",
        "latent_pi.2.bias",
        "mu.weight",
        "mu.bias",
    )

    def __init__(self, arrays: Mapping[str, np.ndarray]) -> None:
        import torch

        self._parameters = {
            name: torch.from_numpy(arrays[name].copy(order="C")).requires_grad_(False)
            for name in self._PARAMETER_NAMES
        }

    def act(self, observation: np.ndarray) -> np.ndarray:
        import torch
        from torch.nn import functional

        value = torch.from_numpy(observation.copy(order="C")).reshape(1, OBSERVATION_WIDTH)
        with torch.inference_mode():
            latent = functional.relu(
                functional.linear(
                    value,
                    self._parameters["latent_pi.0.weight"],
                    self._parameters["latent_pi.0.bias"],
                )
            )
            latent = functional.relu(
                functional.linear(
                    latent,
                    self._parameters["latent_pi.2.weight"],
                    self._parameters["latent_pi.2.bias"],
                )
            )
            mean = functional.linear(
                latent,
                self._parameters["mu.weight"],
                self._parameters["mu.bias"],
            )
            normalized = torch.tanh(mean)
        return np.ascontiguousarray(normalized.numpy()[0], dtype="<f4")

    def assert_matches(self, arrays: Mapping[str, np.ndarray]) -> None:
        import torch

        for name, tensor in self._parameters.items():
            expected = torch.from_numpy(arrays[name].copy(order="C"))
            if not torch.equal(tensor, expected):
                raise ExperimentContractError("trusted TQC inference state changed")


def _require_exact_in_memory_tqc(model: object) -> object:
    import gymnasium as gym
    import torch
    from sb3_contrib import TQC
    from sb3_contrib.tqc.policies import Actor, TQCPolicy
    from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
    from stable_baselines3.common.torch_layers import FlattenExtractor

    policy = getattr(model, "policy", None)
    actor = getattr(model, "actor", None)
    if type(model) is not TQC or type(policy) is not TQCPolicy or type(actor) is not Actor:
        raise ExperimentContractError("trusted model must be exact TQC/TQCPolicy/Actor types")
    if (
        getattr(policy, "actor", None) is not actor
        or getattr(model, "policy_class", None) is not TQCPolicy
    ):
        raise ExperimentContractError("trusted TQC policy and model actor identities differ")
    if any(
        type(getattr(value, "device", None)) is not torch.device
        or getattr(value, "device", None) != torch.device("cpu")
        for value in (model, policy, actor)
    ):
        raise ExperimentContractError("trusted TQC model must reside on CPU")
    for field_name, space, expected_hash in (
        (
            "observation",
            getattr(model, "observation_space", None),
            EXPECTED_OBSERVATION_SPACE_SHA256,
        ),
        ("action", getattr(model, "action_space", None), EXPECTED_ACTION_SPACE_SHA256),
        (
            "policy observation",
            getattr(policy, "observation_space", None),
            EXPECTED_OBSERVATION_SPACE_SHA256,
        ),
        (
            "policy action",
            getattr(policy, "action_space", None),
            EXPECTED_ACTION_SPACE_SHA256,
        ),
        (
            "actor observation",
            getattr(actor, "observation_space", None),
            EXPECTED_OBSERVATION_SPACE_SHA256,
        ),
        ("actor action", getattr(actor, "action_space", None), EXPECTED_ACTION_SPACE_SHA256),
    ):
        if type(space) is not gym.spaces.Box or space_sha256(space) != expected_hash:
            raise ExperimentContractError(f"trusted TQC {field_name} Box differs")
    if (
        getattr(policy, "activation_fn", None) is not torch.nn.ReLU
        or type(getattr(policy, "net_arch", None)) is not list
        or policy.net_arch != [256, 256]
        or type(getattr(policy, "share_features_extractor", None)) is not bool
        or policy.share_features_extractor is not False
        or getattr(policy, "features_extractor", "missing") is not None
    ):
        raise ExperimentContractError("trusted TQC policy architecture differs")
    if (
        type(getattr(actor, "features_extractor", None)) is not FlattenExtractor
        or getattr(actor, "activation_fn", None) is not torch.nn.ReLU
        or type(getattr(actor, "net_arch", None)) is not list
        or actor.net_arch != [256, 256]
        or type(getattr(actor, "features_dim", None)) is not int
        or actor.features_dim != OBSERVATION_WIDTH
    ):
        raise ExperimentContractError("trusted TQC actor feature architecture differs")
    flatten = getattr(actor.features_extractor, "flatten", None)
    if type(flatten) is not torch.nn.Flatten or flatten.start_dim != 1 or flatten.end_dim != -1:
        raise ExperimentContractError("trusted TQC FlattenExtractor differs")
    exact_actor_attributes = {
        "use_sde": False,
        "use_expln": False,
        "full_std": True,
        "normalize_images": True,
        "squash_output": True,
    }
    for name, expected in exact_actor_attributes.items():
        observed = getattr(actor, name, None)
        if type(observed) is not bool or observed is not expected:
            raise ExperimentContractError(f"trusted TQC actor {name} differs")
    if type(getattr(actor, "log_std_init", None)) is not float or actor.log_std_init != -3.0:
        raise ExperimentContractError("trusted TQC actor log_std_init differs")
    if type(getattr(actor, "clip_mean", None)) is not float or actor.clip_mean != 2.0:
        raise ExperimentContractError("trusted TQC actor clip_mean differs")
    latent_modules = tuple(actor.latent_pi)
    if tuple(type(module) for module in latent_modules) != (
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.Linear,
        torch.nn.ReLU,
    ):
        raise ExperimentContractError("trusted TQC actor must use exact Linear/ReLU layers")
    dimensions = tuple(
        (module.in_features, module.out_features)
        for module in (latent_modules[0], latent_modules[2])
    )
    if dimensions != ((348, 256), (256, 256)):
        raise ExperimentContractError("trusted TQC actor hidden dimensions differ")
    if (
        type(actor.mu) is not torch.nn.Linear
        or (actor.mu.in_features, actor.mu.out_features) != (256, ACTION_WIDTH)
        or type(actor.log_std) is not torch.nn.Linear
        or (actor.log_std.in_features, actor.log_std.out_features) != (256, ACTION_WIDTH)
        or type(actor.action_dist) is not SquashedDiagGaussianDistribution
        or type(actor.action_dist.action_dim) is not int
        or actor.action_dist.action_dim != ACTION_WIDTH
    ):
        raise ExperimentContractError("trusted TQC actor output distribution differs")
    return actor


@dataclass(frozen=True, slots=True)
class ProtectedDeterministicTQCActor:
    """Locally issued snapshot; constructor guard is not a cryptographic boundary."""

    actor_id: str
    source_kind: str
    snapshot_state_sha256: str
    schema_sha256: str
    persisted_artifact_sha256: str | None
    persisted_artifact_byte_count: int | None
    equivalence_observation_sha256: str | None
    trusted_normalized_action_sha256: str | None
    protected_normalized_action_sha256: str | None
    trusted_physical_action_sha256: str | None
    protected_physical_action_sha256: str | None
    arrays: Mapping[str, np.ndarray] = field(repr=False)
    _inference: _TorchTQCInference = field(init=False, repr=False, compare=False)
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _ACTOR_ISSUER:
            raise ExperimentContractError(
                "protected actors may only be issued by an admission factory"
            )
        if self.source_kind not in {
            "already_strict_loaded_actor/v1",
            "trusted_in_memory_sb3_contrib_tqc_snapshot/v1",
        }:
            raise ExperimentContractError("protected actor source kind is unsupported")
        if type(self.source_kind) is not str:
            raise ExperimentContractError("protected actor source kind must be an exact string")
        if type(self.actor_id) is not str or not self.actor_id.startswith("tqc-actor-"):
            raise ExperimentContractError("protected actor id is invalid")
        _canonical_sha256(self.snapshot_state_sha256, field="actor snapshot state SHA-256")
        _canonical_sha256(self.schema_sha256, field="actor schema SHA-256")
        expected_state = actor_state_sha256(self.arrays)
        if self.snapshot_state_sha256 != expected_state:
            raise ExperimentContractError("protected actor state SHA-256 differs")
        if self.schema_sha256 != actor_schema_sha256():
            raise ExperimentContractError("protected actor schema SHA-256 differs")
        equivalence_fields = (
            self.equivalence_observation_sha256,
            self.trusted_normalized_action_sha256,
            self.protected_normalized_action_sha256,
            self.trusted_physical_action_sha256,
            self.protected_physical_action_sha256,
        )
        if self.source_kind == "already_strict_loaded_actor/v1":
            _canonical_sha256(
                self.persisted_artifact_sha256,
                field="persisted actor artifact SHA-256",
            )
            if (
                type(self.persisted_artifact_byte_count) is not int
                or not 0 < self.persisted_artifact_byte_count <= MAX_ARCHIVE_BYTES
            ):
                raise ExperimentContractError("persisted actor byte count is invalid")
            if any(value is not None for value in equivalence_fields):
                raise ExperimentContractError(
                    "strict-loaded actor has in-memory equivalence fields"
                )
        else:
            if (
                self.persisted_artifact_sha256 is not None
                or self.persisted_artifact_byte_count is not None
            ):
                raise ExperimentContractError("in-memory actor cannot claim a persisted artifact")
            for index, value in enumerate(equivalence_fields):
                _canonical_sha256(value, field=f"actor equivalence SHA-256 {index}")
            if self.equivalence_observation_sha256 != EQUIVALENCE_OBSERVATION_SHA256:
                raise ExperimentContractError("actor equivalence observation identity differs")
            if self.trusted_normalized_action_sha256 != self.protected_normalized_action_sha256:
                raise ExperimentContractError("normalized deterministic actor equivalence differs")
            if self.trusted_physical_action_sha256 != self.protected_physical_action_sha256:
                raise ExperimentContractError("physical deterministic actor equivalence differs")
        immutable = _immutable_actor_arrays(self.arrays)
        object.__setattr__(self, "arrays", immutable)
        object.__setattr__(self, "_inference", _TorchTQCInference(immutable))

    @classmethod
    def from_strict_loaded_actor(cls, loaded: LoadedTQCActor) -> ProtectedDeterministicTQCActor:
        """Snapshot an actor that was already admitted by the strict data-only loader."""

        if type(loaded) is not LoadedTQCActor:
            raise ExperimentContractError("expected an already-strict-loaded TQC actor")
        state_sha256 = actor_state_sha256(loaded.arrays)
        if loaded.state_sha256 != state_sha256:
            raise ExperimentContractError("strict-loaded actor state changed after admission")
        if loaded.schema_sha256 != actor_schema_sha256():
            raise ExperimentContractError("strict-loaded actor schema differs")
        canonical_payload = encode_actor_npz(loaded.arrays)
        canonical_sha256 = hashlib.sha256(canonical_payload).hexdigest()
        artifact_sha256 = _canonical_sha256(
            loaded.content_sha256,
            field="strict-loaded actor content SHA-256",
        )
        if canonical_sha256 != artifact_sha256:
            raise ExperimentContractError(
                "strict-loaded actor content is not canonical array bytes"
            )
        if type(loaded.byte_count) is not int or loaded.byte_count != len(canonical_payload):
            raise ExperimentContractError(
                "strict-loaded actor byte count differs from canonical bytes"
            )
        return cls(
            actor_id=f"tqc-actor-strict-{state_sha256[:16]}",
            source_kind="already_strict_loaded_actor/v1",
            snapshot_state_sha256=state_sha256,
            schema_sha256=loaded.schema_sha256,
            persisted_artifact_sha256=artifact_sha256,
            persisted_artifact_byte_count=loaded.byte_count,
            equivalence_observation_sha256=None,
            trusted_normalized_action_sha256=None,
            protected_normalized_action_sha256=None,
            trusted_physical_action_sha256=None,
            protected_physical_action_sha256=None,
            arrays=loaded.arrays,
            _issuer=_ACTOR_ISSUER,
        )

    @classmethod
    def from_trusted_in_memory_model(cls, model: object) -> ProtectedDeterministicTQCActor:
        """Snapshot only an exact, already-constructed CPU SB3-Contrib TQC model."""

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - dependency boundary
            raise ExperimentContractError("install the train extra for in-memory TQC") from exc
        actor = _require_exact_in_memory_tqc(model)
        state_dict = getattr(actor, "state_dict", None)
        if not callable(state_dict):
            raise ExperimentContractError("trusted TQC model has no actor state")
        arrays = {
            name: np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4")
            for name, tensor in state_dict().items()
        }
        arrays.update(
            {
                "action_low": np.full(ACTION_WIDTH, -0.4, dtype="<f4"),
                "action_high": np.full(ACTION_WIDTH, 0.4, dtype="<f4"),
                "format_version": np.asarray([1], dtype="<i8"),
            }
        )
        validated = validate_actor_arrays(arrays)
        state_sha256 = actor_state_sha256(validated)
        protected = cls(
            actor_id=f"tqc-actor-memory-{state_sha256[:16]}",
            source_kind="trusted_in_memory_sb3_contrib_tqc_snapshot/v1",
            snapshot_state_sha256=state_sha256,
            schema_sha256=actor_schema_sha256(),
            persisted_artifact_sha256=None,
            persisted_artifact_byte_count=None,
            equivalence_observation_sha256=EQUIVALENCE_OBSERVATION_SHA256,
            trusted_normalized_action_sha256="0" * 64,
            protected_normalized_action_sha256="0" * 64,
            trusted_physical_action_sha256="0" * 64,
            protected_physical_action_sha256="0" * 64,
            arrays=validated,
            _issuer=_ACTOR_ISSUER,
        )
        observations = _equivalence_observations()
        with torch.inference_mode():
            trusted_normalized = np.ascontiguousarray(
                [
                    actor(
                        torch.from_numpy(observation.copy(order="C")).reshape(1, OBSERVATION_WIDTH),
                        deterministic=True,
                    )
                    .detach()
                    .cpu()
                    .numpy()[0]
                    for observation in observations
                ],
                dtype="<f4",
            )
        trusted_physical = np.ascontiguousarray(
            [
                model.predict(observation.copy(order="C"), deterministic=True)[0]
                for observation in observations
            ],
            dtype="<f4",
        )
        protected_actions = tuple(protected.act(observation) for observation in observations)
        protected_normalized = np.ascontiguousarray(
            [action.normalized for action in protected_actions], dtype="<f4"
        )
        protected_physical = np.ascontiguousarray(
            [action.physical for action in protected_actions], dtype="<f4"
        )
        for name, trusted, observed in (
            ("normalized", trusted_normalized, protected_normalized),
            ("physical", trusted_physical, protected_physical),
        ):
            if (
                trusted.shape != (4, ACTION_WIDTH)
                or observed.shape != (4, ACTION_WIDTH)
                or trusted.dtype != np.dtype("<f4")
                or observed.dtype != np.dtype("<f4")
                or not trusted.flags.c_contiguous
                or not observed.flags.c_contiguous
                or not np.isfinite(trusted).all()
                or not np.isfinite(observed).all()
                or trusted.tobytes(order="C") != observed.tobytes(order="C")
            ):
                raise ExperimentContractError(
                    f"trusted and protected {name} deterministic action bytes differ"
                )
        final_state = {
            name: np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4")
            for name, tensor in actor.state_dict().items()
        }
        final_state.update(
            {
                "action_low": np.full(ACTION_WIDTH, -0.4, dtype="<f4"),
                "action_high": np.full(ACTION_WIDTH, 0.4, dtype="<f4"),
                "format_version": np.asarray([1], dtype="<i8"),
            }
        )
        if actor_state_sha256(final_state) != state_sha256:
            raise ExperimentContractError("trusted TQC actor changed during equivalence admission")
        return cls(
            actor_id=protected.actor_id,
            source_kind=protected.source_kind,
            snapshot_state_sha256=protected.snapshot_state_sha256,
            schema_sha256=protected.schema_sha256,
            persisted_artifact_sha256=None,
            persisted_artifact_byte_count=None,
            equivalence_observation_sha256=EQUIVALENCE_OBSERVATION_SHA256,
            trusted_normalized_action_sha256=_array_sha256(trusted_normalized),
            protected_normalized_action_sha256=_array_sha256(protected_normalized),
            trusted_physical_action_sha256=_array_sha256(trusted_physical),
            protected_physical_action_sha256=_array_sha256(protected_physical),
            arrays=protected.arrays,
            _issuer=_ACTOR_ISSUER,
        )

    def assert_integrity(self) -> None:
        if actor_state_sha256(self.arrays) != self.snapshot_state_sha256:
            raise ExperimentContractError("protected actor state changed during evaluation")
        if self.persisted_artifact_sha256 is not None:
            payload = encode_actor_npz(self.arrays)
            if (
                hashlib.sha256(payload).hexdigest() != self.persisted_artifact_sha256
                or len(payload) != self.persisted_artifact_byte_count
            ):
                raise ExperimentContractError("persisted actor binding changed during evaluation")
        self._inference.assert_matches(self.arrays)

    def act(self, observation: object) -> DeterministicTQCAction:
        """Run the fixed float32 ReLU/mean/tanh inference path."""

        try:
            value = np.asarray(observation, dtype="<f4")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ExperimentContractError("actor observation is not numeric") from exc
        if value.shape != (OBSERVATION_WIDTH,) or not np.isfinite(value).all():
            raise ExperimentContractError("actor observation must be finite shape (348,)")
        normalized = self._inference.act(value)
        low = self.arrays["action_low"]
        high = self.arrays["action_high"]
        physical = np.ascontiguousarray(
            low + (np.float32(0.5) * (normalized + np.float32(1.0)) * (high - low)),
            dtype="<f4",
        )
        return DeterministicTQCAction(normalized=normalized, physical=physical)


@dataclass(frozen=True, slots=True)
class TQCDevelopmentRuntimeReceipt:
    """Locally issued runtime facts; issuer guard is non-cryptographic."""

    canonical_bytes: bytes = field(repr=False)
    sha256: str
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _RUNTIME_ISSUER:
            raise ExperimentContractError("runtime receipts may only be issued by inspection")
        if type(self.canonical_bytes) is not bytes or not self.canonical_bytes:
            raise ExperimentContractError("runtime receipt bytes are absent")
        try:
            value = json.loads(
                self.canonical_bytes.decode("utf-8", errors="strict"),
                object_pairs_hook=_reject_duplicate_pairs,
                parse_constant=_reject_nonfinite_constant,
            )
        except ExperimentContractError:
            raise
        except (UnicodeError, ValueError, RecursionError) as exc:
            raise ExperimentContractError("runtime receipt bytes are invalid") from exc
        if not isinstance(value, dict):
            raise ExperimentContractError("runtime receipt root must be an object")
        if canonical_json(value) != self.canonical_bytes:
            raise ExperimentContractError("runtime receipt bytes are not canonical")
        _canonical_sha256(self.sha256, field="runtime receipt SHA-256")
        if hashlib.sha256(self.canonical_bytes).hexdigest() != self.sha256:
            raise ExperimentContractError("runtime receipt SHA-256 differs")

    def to_dict(self) -> dict[str, object]:
        return json.loads(self.canonical_bytes)


@dataclass(frozen=True, slots=True)
class TQCDevelopmentSeedEvaluation:
    """Locally issued metric/trace pair; issuer guard is non-cryptographic."""

    seed: int
    metrics: TQCDevelopmentEpisodeMetrics
    trace: TrajectoryTrace = field(repr=False)
    trace_sha256: str
    controller_observation_shape: tuple[int, int]
    controller_observation_sha256: str
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _SEED_EVALUATION_ISSUER:
            raise ExperimentContractError(
                "seed evaluations may only be issued by the protected evaluator"
            )
        if type(self.seed) is not int or self.seed not in EVALUATION_SEEDS:
            raise ExperimentContractError("seed evaluation uses an undeclared seed")
        if type(self.metrics) is not TQCDevelopmentEpisodeMetrics:
            raise ExperimentContractError("seed evaluation metrics were not issued by the core")
        if type(self.trace) is not TrajectoryTrace:
            raise ExperimentContractError("seed evaluation requires a canonical trace")
        if self.seed != self.metrics.seed:
            raise ExperimentContractError("seed evaluation and metric seed differ")
        if len(self.trace.samples) != EXPECTED_STEPS + 1:
            raise ExperimentContractError("development trace must cover all 1001 boundaries")
        if self.trace.trace_id != f"tqc-dev1m-seed-{self.seed}":
            raise ExperimentContractError("development trace id differs from its seed")
        if self.trace.evidence_class is not EvidenceClass.EXPLORATORY:
            raise ExperimentContractError("development trace must remain exploratory")
        _canonical_sha256(self.trace_sha256, field="trace SHA-256")
        if self.trace.sha256 != self.trace_sha256:
            raise ExperimentContractError("seed evaluation trace SHA-256 differs")
        if self.controller_observation_shape != (EXPECTED_STEPS + 1, OBSERVATION_WIDTH):
            raise ExperimentContractError("controller observation trace shape differs")
        if type(self.controller_observation_shape) is not tuple or any(
            type(value) is not int for value in self.controller_observation_shape
        ):
            raise ExperimentContractError("controller observation shape types differ")
        observations = _controller_observation_matrix(self.trace)
        _canonical_sha256(
            self.controller_observation_sha256,
            field="controller observation SHA-256",
        )
        if _array_sha256(observations) != self.controller_observation_sha256:
            raise ExperimentContractError("controller observation SHA-256 differs")
        if _metrics_from_trace(self.seed, self.trace) != self.metrics:
            raise ExperimentContractError("seed metrics differ from the canonical trace")


@dataclass(frozen=True, slots=True)
class TQCDevelopmentEvaluation:
    """Locally issued aggregate; issuer guard is non-cryptographic."""

    design_file_sha256: str
    design_semantic_sha256: str
    runtime: TQCDevelopmentRuntimeReceipt
    actor_id: str
    actor_source_kind: str
    actor_snapshot_state_sha256: str
    persisted_actor_artifact_sha256: str | None
    persisted_actor_artifact_byte_count: int | None
    actor_equivalence_observation_sha256: str | None
    actor_deterministic_normalized_action_sha256: str | None
    actor_deterministic_physical_action_sha256: str | None
    episodes: tuple[TQCDevelopmentSeedEvaluation, ...]
    cohort: TQCDevelopmentCohortMetrics
    reward_or_info_fields_read: bool = False
    checkpoint_or_episode_selection_performed: bool = False
    trace_evidence_class: EvidenceClass = EvidenceClass.EXPLORATORY
    formal_experiment_eligible: bool = False
    automatic_20m_authorization: bool = False
    automatic_tracker_admission: bool = False
    checkpoint_provenance_verified: bool = False
    claim_boundary: str = CLAIM_BOUNDARY
    record_issuer_boundary: str = "trusted_local_process_only_not_cryptographic/v1"
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _EVALUATION_ISSUER:
            raise ExperimentContractError(
                "development evaluations may only be issued by the protected evaluator"
            )
        for field_name in (
            "design_file_sha256",
            "design_semantic_sha256",
            "actor_snapshot_state_sha256",
        ):
            _canonical_sha256(getattr(self, field_name), field=field_name)
        if type(self.runtime) is not TQCDevelopmentRuntimeReceipt:
            raise ExperimentContractError("evaluation runtime receipt was not issued locally")
        if type(self.actor_id) is not str or not self.actor_id.startswith("tqc-actor-"):
            raise ExperimentContractError("evaluation actor id is invalid")
        if self.actor_source_kind not in {
            "already_strict_loaded_actor/v1",
            "trusted_in_memory_sb3_contrib_tqc_snapshot/v1",
        }:
            raise ExperimentContractError("evaluation actor source kind differs")
        if type(self.actor_source_kind) is not str:
            raise ExperimentContractError("evaluation actor source kind must be an exact string")
        if self.actor_source_kind == "already_strict_loaded_actor/v1":
            _canonical_sha256(
                self.persisted_actor_artifact_sha256,
                field="persisted actor artifact SHA-256",
            )
            if (
                type(self.persisted_actor_artifact_byte_count) is not int
                or not 0 < self.persisted_actor_artifact_byte_count <= MAX_ARCHIVE_BYTES
            ):
                raise ExperimentContractError("persisted actor artifact byte count is invalid")
            if any(
                value is not None
                for value in (
                    self.actor_equivalence_observation_sha256,
                    self.actor_deterministic_normalized_action_sha256,
                    self.actor_deterministic_physical_action_sha256,
                )
            ):
                raise ExperimentContractError(
                    "strict actor evaluation has memory equivalence fields"
                )
        else:
            if (
                self.persisted_actor_artifact_sha256 is not None
                or self.persisted_actor_artifact_byte_count is not None
            ):
                raise ExperimentContractError("in-memory actor evaluation claims persisted bytes")
            for field_name in (
                "actor_equivalence_observation_sha256",
                "actor_deterministic_normalized_action_sha256",
                "actor_deterministic_physical_action_sha256",
            ):
                _canonical_sha256(getattr(self, field_name), field=field_name)
        if type(self.episodes) is not tuple or len(self.episodes) != 20:
            raise ExperimentContractError("development evaluation requires exactly twenty episodes")
        if any(type(episode) is not TQCDevelopmentSeedEvaluation for episode in self.episodes):
            raise ExperimentContractError("evaluation episodes were not issued locally")
        observed_seeds = tuple(episode.seed for episode in self.episodes)
        if observed_seeds != EVALUATION_SEEDS:
            raise ExperimentContractError("evaluation episodes differ from fixed seed order")
        recomputed = summarize_tqc_development_cohort(
            tuple(episode.metrics for episode in self.episodes),
            expected_seeds=EVALUATION_SEEDS,
        )
        if type(self.cohort) is not TQCDevelopmentCohortMetrics or self.cohort != recomputed:
            raise ExperimentContractError("result episodes differ from the fixed cohort")
        boolean_fields = (
            "reward_or_info_fields_read",
            "checkpoint_or_episode_selection_performed",
            "formal_experiment_eligible",
            "automatic_20m_authorization",
            "automatic_tracker_admission",
            "checkpoint_provenance_verified",
        )
        if any(type(getattr(self, name)) is not bool for name in boolean_fields):
            raise ExperimentContractError("evaluation boundary flags must be exact booleans")
        if self.reward_or_info_fields_read or self.checkpoint_or_episode_selection_performed:
            raise ExperimentContractError("development evaluation crossed its protected boundary")
        if self.trace_evidence_class is not EvidenceClass.EXPLORATORY:
            raise ExperimentContractError("excluded development traces must remain exploratory")
        if (
            self.formal_experiment_eligible
            or self.automatic_20m_authorization
            or self.automatic_tracker_admission
            or self.checkpoint_provenance_verified
        ):
            raise ExperimentContractError("development result overclaims its authority")
        if self.claim_boundary != CLAIM_BOUNDARY:
            raise ExperimentContractError("development result claim boundary differs")
        if type(self.claim_boundary) is not str:
            raise ExperimentContractError("development claim boundary must be an exact string")
        if (
            type(self.record_issuer_boundary) is not str
            or self.record_issuer_boundary != "trusted_local_process_only_not_cryptographic/v1"
        ):
            raise ExperimentContractError("evaluation issuer boundary differs")
        for episode in self.episodes:
            bindings = {binding.role: binding for binding in episode.trace.artifact_bindings}
            expected_roles = {"runtime", "development_design", "evaluator", "actor_snapshot"}
            if self.persisted_actor_artifact_sha256 is not None:
                expected_roles.add("persisted_actor_artifact")
            if set(bindings) != expected_roles:
                raise ExperimentContractError("trace artifact binding roles differ")
            if (
                bindings["runtime"].artifact_id != "tqc-development-runtime"
                or bindings["runtime"].sha256 != self.runtime.sha256
                or bindings["development_design"].artifact_id != "tqc-development-design"
                or bindings["development_design"].sha256 != self.design_file_sha256
                or bindings["evaluator"].artifact_id != EVALUATOR_ID
                or bindings["evaluator"].sha256 != sha256_file(Path(__file__))
                or bindings["actor_snapshot"].artifact_id != self.actor_id
                or bindings["actor_snapshot"].sha256 != self.actor_snapshot_state_sha256
            ):
                raise ExperimentContractError("trace artifact identity differs from evaluation")
            if self.persisted_actor_artifact_sha256 is not None:
                persisted = bindings["persisted_actor_artifact"]
                expected_id = f"strict-tqc-actor-{self.persisted_actor_artifact_byte_count}-bytes"
                if (
                    persisted.sha256 != self.persisted_actor_artifact_sha256
                    or persisted.artifact_id != expected_id
                ):
                    raise ExperimentContractError("trace persisted actor binding differs")


def _wrapper_types(environment: object) -> tuple[str, ...]:
    observed: list[str] = []
    current = environment
    while True:
        observed.append(f"{type(current).__module__}.{type(current).__name__}")
        if not hasattr(current, "env"):
            return tuple(observed)
        current = current.env


def _host_hardware_receipt() -> dict[str, object]:
    cpu_model = platform.processor().strip()
    hardware_model: str | None = None
    if sys.platform == "darwin":
        for field_name, sysctl_name in (
            ("cpu_model", "machdep.cpu.brand_string"),
            ("hardware_model", "hw.model"),
        ):
            try:
                observed = subprocess.run(
                    ["/usr/sbin/sysctl", "-n", sysctl_name],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                observed = ""
            if field_name == "cpu_model" and observed:
                cpu_model = observed
            elif field_name == "hardware_model" and observed:
                hardware_model = observed
    logical_cpu_count = os.cpu_count()
    try:
        total_memory_bytes = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, TypeError, ValueError):
        total_memory_bytes = 0
    return {
        "cpu_model": cpu_model or None,
        "hardware_model": hardware_model,
        "logical_cpu_count": logical_cpu_count,
        "total_memory_bytes": total_memory_bytes or None,
    }


def _environment_config(design: LoadedTQCDevelopmentDesign) -> HumanoidExperimentConfig:
    raw = _validated_design_payload(design)["environment"]
    return HumanoidExperimentConfig(
        env_id=raw["environment_id"],
        terminate_when_unhealthy=raw["terminate_when_unhealthy"],
        reset_noise_scale=raw["reset_noise_scale"],
        exclude_current_positions_from_observation=(
            raw["exclude_current_positions_from_observation"]
        ),
        frame_skip=raw["frame_skip"],
    )


def _validate_episode_environment(environment: object, config: HumanoidExperimentConfig) -> None:
    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise ExperimentContractError("install the gym extra for TQC evaluation") from exc
    if type(environment) is not gym.wrappers.TimeLimit:
        raise ExperimentContractError("Humanoid TimeLimit must be the outer wrapper")
    if _wrapper_types(environment) != EXPECTED_WRAPPER_TYPES:
        raise ExperimentContractError("Humanoid evaluation wrapper stack differs")
    if (
        type(environment._max_episode_steps) is not int
        or environment._max_episode_steps != EXPECTED_STEPS
    ):
        raise ExperimentContractError("Humanoid TimeLimit must equal 1000 steps")
    physical = environment.unwrapped
    if getattr(physical, "contact_capture_id", None) != CONTACT_CAPTURE_ID:
        raise ExperimentContractError("all-substep contact capture is not active")
    expected_kwargs: dict[str, bool | float | int | None] = {
        "render_mode": None,
        **config.gym_kwargs(),
    }
    if dict(getattr(environment.spec, "kwargs", {})) != expected_kwargs:
        raise ExperimentContractError("Humanoid environment kwargs differ")
    if (
        type(physical._terminate_when_unhealthy) is not bool
        or type(physical._exclude_current_positions_from_observation) is not bool
        or type(physical._reset_noise_scale) is not float
        or type(physical.frame_skip) is not int
    ):
        raise ExperimentContractError("Humanoid configuration semantic types differ")
    direct = {
        "terminate_when_unhealthy": physical._terminate_when_unhealthy,
        "reset_noise_scale": physical._reset_noise_scale,
        "exclude_current_positions_from_observation": (
            physical._exclude_current_positions_from_observation
        ),
        "frame_skip": physical.frame_skip,
    }
    if direct != config.gym_kwargs():
        raise ExperimentContractError("Humanoid observed configuration differs")
    if type(physical.dt) is not float or not math.isclose(
        physical.dt, CONTROL_PERIOD_SECONDS, rel_tol=0.0, abs_tol=0.0
    ):
        raise ExperimentContractError("Humanoid control period differs")


def inspect_tqc_development_runtime(
    design: LoadedTQCDevelopmentDesign,
) -> TQCDevelopmentRuntimeReceipt:
    """Inspect and bind every pinned runtime fact used by this evaluator."""

    raw = _validated_design_payload(design)
    try:
        import gymnasium
        import mujoco
        import sb3_contrib
        import stable_baselines3
        import torch
        from gymnasium.envs.mujoco import humanoid_v5
        from sb3_contrib.common import utils as sb3_contrib_common_utils
        from sb3_contrib.tqc import policies as tqc_policies
        from sb3_contrib.tqc import tqc as tqc_algorithm
        from stable_baselines3.common import (
            base_class,
            buffers,
            distributions,
            noise,
            off_policy_algorithm,
            policies,
            save_util,
            torch_layers,
            utils,
        )
        from stable_baselines3.common import logger as sb3_logger
        from stable_baselines3.common.vec_env import base_vec_env, dummy_vec_env
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise ExperimentContractError("install the gym and train extras") from exc

    config = _environment_config(design)
    environment = make_humanoid_env(config, capture_substep_contacts=True)
    try:
        observation, _ignored_info = environment.reset(seed=design.evaluation_seeds[0])
        _validate_episode_environment(environment, config)
        physical = environment.unwrapped
        validate_humanoid_actuator_abi(physical)
        model_path = Path(str(getattr(physical, "fullpath", "")))
        if model_path.is_symlink() or not model_path.is_file():
            raise ExperimentContractError("Humanoid model XML must be one regular file")
        observation_vector = finite_direct_vector(
            observation,
            width=OBSERVATION_WIDTH,
            field="runtime reset observation",
        )
        direct_observation = finite_direct_vector(
            physical._get_obs(),
            width=OBSERVATION_WIDTH,
            field="runtime direct reset observation",
        )
        if not np.array_equal(observation_vector, direct_observation):
            raise ExperimentContractError("returned and direct reset observations differ")

        hardware = _host_hardware_receipt()
        observed: dict[str, object] = {
            "python_version": platform.python_version(),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "platform_release": platform.release(),
            "cpu_model": hardware["cpu_model"],
            "hardware_model": hardware["hardware_model"],
            "logical_cpu_count": hardware["logical_cpu_count"],
            "total_memory_bytes": hardware["total_memory_bytes"],
            "torch_intraop_thread_count": int(torch.get_num_threads()),
            "torch_interop_thread_count": int(torch.get_num_interop_threads()),
            "numpy_version": str(np.__version__),
            "torch_version": str(torch.__version__),
            "gymnasium_version": str(gymnasium.__version__),
            "mujoco_version": str(mujoco.__version__),
            "stable_baselines3_version": str(stable_baselines3.__version__),
            "sb3_contrib_version": str(sb3_contrib.__version__),
            "dependency_lock_sha256": sha256_file(dependency_lock_path()),
            "mujoco_model_sha256": sha256_file(model_path),
            "observation_space_sha256": space_sha256(environment.observation_space),
            "action_space_sha256": space_sha256(environment.action_space),
            "environment_source_sha256": module_sha256(humanoid_module),
            "gym_humanoid_source_sha256": module_sha256(humanoid_v5),
            "tqc_algorithm_source_sha256": module_sha256(tqc_algorithm),
            "tqc_policy_source_sha256": module_sha256(tqc_policies),
            "sb3_base_algorithm_source_sha256": module_sha256(base_class),
            "sb3_save_util_source_sha256": module_sha256(save_util),
            "sb3_off_policy_source_sha256": module_sha256(off_policy_algorithm),
            "sb3_replay_buffer_source_sha256": module_sha256(buffers),
            "sb3_dummy_vec_env_source_sha256": module_sha256(dummy_vec_env),
            "sb3_distributions_source_sha256": module_sha256(distributions),
            "sb3_policies_source_sha256": module_sha256(policies),
            "sb3_torch_layers_source_sha256": module_sha256(torch_layers),
            "sb3_utils_source_sha256": module_sha256(utils),
            "sb3_noise_source_sha256": module_sha256(noise),
            "sb3_logger_source_sha256": module_sha256(sb3_logger),
            "sb3_contrib_common_utils_source_sha256": module_sha256(sb3_contrib_common_utils),
            "sb3_base_vec_env_source_sha256": module_sha256(base_vec_env),
        }
        required = raw["runtime_requirements"]
        non_runtime_keys = {
            "execution_manifest_required_before_training",
            "execution_manifest_must_bind",
        }
        expected_runtime_keys = set(required) - non_runtime_keys
        if set(observed) != expected_runtime_keys:
            missing = sorted(expected_runtime_keys - set(observed))
            extra = sorted(set(observed) - expected_runtime_keys)
            raise ExperimentContractError(
                "development evaluator runtime coverage differs: "
                f"missing={missing!r}, extra={extra!r}"
            )
        mismatches = [
            name
            for name, value in observed.items()
            if type(value) is not type(required[name]) or value != required[name]
        ]
        if mismatches:
            raise ExperimentContractError(f"development evaluator runtime differs: {mismatches!r}")
        payload = {
            "runtime_id": "tqc_humanoid_development_evaluation_runtime/v1",
            **observed,
            "environment_id": str(environment.spec.id),
            "environment_kwargs": {"render_mode": None, **config.gym_kwargs()},
            "wrapper_types": list(_wrapper_types(environment)),
            "environment_max_episode_steps": int(environment._max_episode_steps),
            "observation_shape": list(environment.observation_space.shape),
            "observation_dtype": np.dtype(environment.observation_space.dtype).str,
            "action_shape": list(environment.action_space.shape),
            "action_dtype": np.dtype(environment.action_space.dtype).str,
            "qpos_shape": list(physical.data.qpos.shape),
            "qvel_shape": list(physical.data.qvel.shape),
            "control_period_seconds": float(physical.dt),
            "contact_capture_id": str(physical.contact_capture_id),
            "project_source_tree_sha256": source_tree_sha256(),
            "evaluator_source_sha256": sha256_file(Path(__file__)),
            "metric_core_source_sha256": sha256_file(
                Path(__file__).with_name("tqc_development_metrics.py")
            ),
            "actor_schema_sha256": actor_schema_sha256(),
            "reward_or_info_fields_read": False,
        }
        encoded = canonical_json(payload)
        return TQCDevelopmentRuntimeReceipt(
            canonical_bytes=encoded,
            sha256=sha256_json(payload),
            _issuer=_RUNTIME_ISSUER,
        )
    finally:
        environment.close()


def _numeric_signal_specs() -> tuple[NumericSignalSpec, ...]:
    return (
        NumericSignalSpec(
            "robot.qpos",
            TraceRole.ROBOT,
            (24,),
            "mixed_m_and_quaternion_and_rad",
            "MuJoCo_generalized_coordinates",
            "direct_mujoco_data.qpos_at_control_boundary/v1",
        ),
        NumericSignalSpec(
            "robot.qvel",
            TraceRole.ROBOT,
            (23,),
            "mixed_m_per_s_and_rad_per_s",
            "MuJoCo_generalized_velocities",
            "direct_mujoco_data.qvel_at_control_boundary/v1",
        ),
        NumericSignalSpec(
            "robot.root_position_world_m",
            TraceRole.ROBOT,
            (3,),
            "m",
            "MuJoCo_world",
            "direct_mujoco_data.qpos_0_3/v1",
        ),
        NumericSignalSpec(
            "evaluation.simulation_time_seconds",
            TraceRole.EVALUATION,
            (1,),
            "s",
            "MuJoCo_simulation_clock",
            "direct_mujoco_data.time_at_control_boundary/v1",
        ),
        NumericSignalSpec(
            "controller.observation",
            TraceRole.CONTROLLER,
            (OBSERVATION_WIDTH,),
            "Gymnasium_Humanoid-v5_observation",
            "Humanoid-v5_observation_order",
            "exact_returned_observation_at_control_boundary_float64/v1",
        ),
        NumericSignalSpec(
            "controller.action",
            TraceRole.CONTROLLER,
            (17,),
            "normalized",
            "Humanoid_actuator_order",
            "strict_tqc_actor_deterministic_tanh_mean/v1",
        ),
        NumericSignalSpec(
            "controller.physical_control",
            TraceRole.CONTROLLER,
            (17,),
            "MuJoCo_control",
            "Humanoid_actuator_order",
            "direct_mujoco_data.ctrl_at_boundary_reset_is_zero/v1",
        ),
        NumericSignalSpec(
            "controller.applied_torque",
            TraceRole.CONTROLLER,
            (17,),
            "N_m",
            "Humanoid_actuated_generalized_coordinates",
            "direct_mujoco_data.qfrc_actuator_at_actuator_dof_addresses/v1",
        ),
        NumericSignalSpec(
            "controller.saturation",
            TraceRole.CONTROLLER,
            (17,),
            "boolean_0_or_1",
            "Humanoid_actuator_order",
            "exact_abs_normalized_action_equals_one/v1",
        ),
        NumericSignalSpec(
            "contact.floor_normal_force_n",
            TraceRole.CONTACT,
            (1,),
            "N",
            "MuJoCo_contact_frame_normal",
            "maximum_direct_mj_contactForce_all_substeps_reset_zero_sentinel/v1",
        ),
        NumericSignalSpec(
            "contact.active_count_by_substep",
            TraceRole.CONTACT,
            (PHYSICS_SUBSTEPS_PER_CONTROL,),
            "contacts",
            "ordered_MuJoCo_physics_substeps",
            "count_of_ordered_contact.active_events_by_substep_reset_zero/v1",
        ),
        NumericSignalSpec(
            "task.progress",
            TraceRole.TASK,
            (1,),
            "m",
            "MuJoCo_world_x",
            "direct_mujoco_data.qpos_0_minus_reset_qpos_0/v1",
        ),
        NumericSignalSpec(
            "evaluation.root_height_m",
            TraceRole.EVALUATION,
            (1,),
            "m",
            "MuJoCo_world_z",
            "direct_mujoco_data.qpos_2/v1",
        ),
        NumericSignalSpec(
            "evaluation.torso_up_z",
            TraceRole.EVALUATION,
            (1,),
            "unitless",
            "MuJoCo_world_z",
            "normalized_direct_mujoco_data.qpos_3_7_wxyz_quaternion/v1",
        ),
        NumericSignalSpec(
            "evaluation.healthy",
            TraceRole.EVALUATION,
            (1,),
            "boolean_0_or_1",
            "MuJoCo_world_z",
            "direct_root_height_in_open_interval_1_2_m/v1",
        ),
        NumericSignalSpec(
            "evaluation.upright",
            TraceRole.EVALUATION,
            (1,),
            "boolean_0_or_1",
            "MuJoCo_world_z",
            "direct_torso_up_z_greater_than_or_equal_to_0.5/v1",
        ),
        NumericSignalSpec(
            "evaluation.non_foot_floor_contact",
            TraceRole.EVALUATION,
            (1,),
            "boolean_0_or_1",
            "all_MuJoCo_contact_substeps",
            "direct_nonfoot_floor_contact_all_substeps_reset_false_sentinel/v1",
        ),
    )


def _missing_signals() -> tuple[MissingSignal, ...]:
    no_reference = "Base-controller development screen has no reference trajectory."
    no_oracle = "Base-controller development screen has no oracle automaton."
    no_recovery = "Base-controller development screen injects no recovery disturbance."
    return (
        MissingSignal(
            "controller.motor_target",
            MissingReason.NOT_APPLICABLE,
            "TQC emits normalized torque controls, not motor-position targets.",
        ),
        MissingSignal(
            "controller.energy_j",
            MissingReason.NOT_IMPLEMENTED,
            "The frozen screen declares action and contact summaries, not an energy metric.",
        ),
        MissingSignal("reference.frame", MissingReason.NOT_APPLICABLE, no_reference),
        MissingSignal("reference.window_index", MissingReason.NOT_APPLICABLE, no_reference),
        MissingSignal("oracle.mode", MissingReason.NOT_APPLICABLE, no_oracle),
        MissingSignal("oracle.phase", MissingReason.NOT_APPLICABLE, no_oracle),
        MissingSignal(
            "oracle.transition_guard_margin",
            MissingReason.NOT_APPLICABLE,
            no_oracle,
        ),
        MissingSignal("recovery.disturbance", MissingReason.NOT_APPLICABLE, no_recovery),
        MissingSignal("recovery.rejoin_state", MissingReason.NOT_APPLICABLE, no_recovery),
    )


def _torso_up_z_from_qpos(qpos: object) -> float:
    values = finite_direct_vector(qpos, width=24, field="MuJoCo qpos")
    w, x, y, z = (float(value) for value in values[3:7])
    norm_squared = w * w + x * x + y * y + z * z
    if not math.isfinite(norm_squared) or norm_squared <= 0.0:
        raise ExperimentContractError("MuJoCo root quaternion has zero or non-finite norm")
    up_z = (w * w - x * x - y * y + z * z) / norm_squared
    if not math.isfinite(up_z) or not -1.0 <= up_z <= 1.0:
        raise ExperimentContractError("MuJoCo torso orientation is outside a rotation bound")
    return up_z


def _boundary_state(physical: object) -> dict[str, object]:
    qpos = finite_direct_vector(physical.data.qpos, width=24, field="MuJoCo qpos")
    qvel = finite_direct_vector(physical.data.qvel, width=23, field="MuJoCo qvel")
    finite_direct_vector(physical.data.qacc, width=23, field="MuJoCo qacc")
    simulation_time = float(physical.data.time)
    if not math.isfinite(simulation_time):
        raise ExperimentContractError("MuJoCo simulation time is non-finite")
    torso_up_z = _torso_up_z_from_qpos(qpos)
    root = qpos[:3].copy()
    return {
        "qpos": qpos,
        "qvel": qvel,
        "root_position_world_m": root,
        "root_height_m": float(root[2]),
        "torso_up_z": torso_up_z,
        "simulation_time_seconds": simulation_time,
    }


def _trace_sample(
    state: Mapping[str, object],
    *,
    controller_observation: np.ndarray,
    normalized_action: np.ndarray,
    physical_control: np.ndarray,
    applied_torque: np.ndarray,
    floor_normal_force_n: float,
    contact_count_by_substep: tuple[int, ...],
    non_foot_floor_contact: bool,
    initial_root_x_m: float,
) -> dict[str, object]:
    height = float(state["root_height_m"])
    up_z = float(state["torso_up_z"])
    return {
        "robot.qpos": state["qpos"],
        "robot.qvel": state["qvel"],
        "robot.root_position_world_m": state["root_position_world_m"],
        "evaluation.simulation_time_seconds": float(state["simulation_time_seconds"]),
        "controller.observation": controller_observation,
        "controller.action": normalized_action,
        "controller.physical_control": physical_control,
        "controller.applied_torque": applied_torque,
        "controller.saturation": (np.abs(normalized_action) == 1.0).astype(np.float64),
        "contact.floor_normal_force_n": float(floor_normal_force_n),
        "contact.active_count_by_substep": contact_count_by_substep,
        "task.progress": float(state["root_position_world_m"][0]) - initial_root_x_m,
        "evaluation.root_height_m": height,
        "evaluation.torso_up_z": up_z,
        "evaluation.healthy": float(1.0 < height < 2.0),
        "evaluation.upright": float(up_z >= MINIMUM_TORSO_UP_Z),
        "evaluation.non_foot_floor_contact": float(non_foot_floor_contact),
    }


def _direct_contact_event_values(
    environment: object,
) -> tuple[tuple[str, ...], tuple[int, ...], float, bool]:
    """Encode every ordered substep contact needed to recompute summaries."""

    physical = getattr(environment, "unwrapped", environment)
    samples = getattr(physical, "last_control_step_contact_samples", None)
    if type(samples) is not tuple:
        raise ExperimentContractError("ordered substep contact samples are unavailable")
    facts = direct_contact_facts(environment)
    if len(samples) != len(facts):
        raise ExperimentContractError("contact samples and classified facts differ")
    counts = [0] * PHYSICS_SUBSTEPS_PER_CONTROL
    encoded: list[str] = []
    previous_key: tuple[int, int] | None = None
    for flattened_index, (sample, fact) in enumerate(zip(samples, facts, strict=True)):
        if type(sample) is not SubstepContactSample or type(fact) is not ContactFact:
            raise ExperimentContractError("substep contact evidence has the wrong type")
        integer_values = (
            sample.physics_substep_index,
            sample.contact_index_within_substep,
            sample.geom1_id,
            sample.geom2_id,
        )
        if any(type(value) is not int for value in integer_values):
            raise ExperimentContractError("substep contact indices must be exact integers")
        substep = sample.physics_substep_index
        contact_index = sample.contact_index_within_substep
        if not 0 <= substep < PHYSICS_SUBSTEPS_PER_CONTROL:
            raise ExperimentContractError("contact substep is outside the fixed five substeps")
        if contact_index != counts[substep]:
            raise ExperimentContractError("contact indices are not contiguous within each substep")
        key = (substep, contact_index)
        if previous_key is not None and key <= previous_key:
            raise ExperimentContractError("substep contacts are not in simulator order")
        previous_key = key
        counts[substep] += 1
        if not 0 <= sample.geom1_id < len(EXPECTED_HUMANOID_GEOM_NAMES) or not 0 <= (
            sample.geom2_id
        ) < len(EXPECTED_HUMANOID_GEOM_NAMES):
            raise ExperimentContractError("contact geometry id is outside the Humanoid model")
        geom1_name = EXPECTED_HUMANOID_GEOM_NAMES[sample.geom1_id]
        geom2_name = EXPECTED_HUMANOID_GEOM_NAMES[sample.geom2_id]
        if (
            fact.contact_index != flattened_index
            or fact.geom1_name != geom1_name
            or fact.geom2_name != geom2_name
            or fact.normal_force_n != sample.normal_force_n
        ):
            raise ExperimentContractError("contact geometry or force differs from direct capture")
        encoded.append(
            canonical_json(
                {
                    "physics_substep_index": substep,
                    "contact_index_within_substep": contact_index,
                    "geom1_id": sample.geom1_id,
                    "geom2_id": sample.geom2_id,
                    "geom1_name": geom1_name,
                    "geom2_name": geom2_name,
                    "normal_force_n": fact.normal_force_n,
                }
            ).decode("utf-8")
        )
    floor_contacts = tuple(fact for fact in facts if fact.involves_floor)
    peak_force = max((fact.normal_force_n for fact in floor_contacts), default=0.0)
    forbidden = any(fact.forbidden_floor_contact for fact in facts)
    if not math.isfinite(peak_force) or peak_force < 0.0:
        raise ExperimentContractError("derived floor contact force is invalid")
    return tuple(encoded), tuple(counts), float(peak_force), bool(forbidden)


def _signal_index(trace: TrajectoryTrace, name: str) -> int:
    names = tuple(signal.name for signal in trace.numeric_signals)
    if names.count(name) != 1:
        raise ExperimentContractError(f"development trace signal {name!r} differs")
    return names.index(name)


def _controller_observation_matrix(trace: TrajectoryTrace) -> np.ndarray:
    index = _signal_index(trace, "controller.observation")
    signal = trace.numeric_signals[index]
    if (
        signal.role is not TraceRole.CONTROLLER
        or signal.shape != (OBSERVATION_WIDTH,)
        or signal.source != "exact_returned_observation_at_control_boundary_float64/v1"
    ):
        raise ExperimentContractError("controller observation trace schema differs")
    values = np.ascontiguousarray(
        [sample.numeric_values[index] for sample in trace.samples],
        dtype="<f8",
    )
    if values.shape != (EXPECTED_STEPS + 1, OBSERVATION_WIDTH):
        raise ExperimentContractError("controller observation trace shape differs")
    if not np.isfinite(values).all():
        raise ExperimentContractError("controller observation trace contains non-finite values")
    return values


def _decode_contact_event(event: object) -> tuple[int, int, ContactFact]:
    if (
        getattr(event, "event_type", None) != CONTACT_EVENT_TYPE
        or getattr(event, "source", None) != CONTACT_EVENT_SOURCE
        or type(getattr(event, "value", None)) is not str
    ):
        raise ExperimentContractError("contact trace event schema differs")
    try:
        raw = json.loads(
            event.value,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_constant,
        )
    except ExperimentContractError:
        raise
    except (TypeError, ValueError, RecursionError) as exc:
        raise ExperimentContractError("contact trace event is invalid JSON") from exc
    if type(raw) is not dict or set(raw) != _CONTACT_EVENT_KEYS:
        raise ExperimentContractError("contact trace event fields differ")
    if canonical_json(raw).decode("utf-8") != event.value:
        raise ExperimentContractError("contact trace event is not canonical JSON")
    integer_fields = (
        "physics_substep_index",
        "contact_index_within_substep",
        "geom1_id",
        "geom2_id",
    )
    if any(type(raw[field]) is not int for field in integer_fields):
        raise ExperimentContractError("contact trace indices must be exact integers")
    substep = raw["physics_substep_index"]
    contact_index = raw["contact_index_within_substep"]
    geom1_id = raw["geom1_id"]
    geom2_id = raw["geom2_id"]
    if not 0 <= substep < PHYSICS_SUBSTEPS_PER_CONTROL or contact_index < 0:
        raise ExperimentContractError("contact trace substep or index is invalid")
    if not 0 <= geom1_id < len(EXPECTED_HUMANOID_GEOM_NAMES) or not 0 <= geom2_id < len(
        EXPECTED_HUMANOID_GEOM_NAMES
    ):
        raise ExperimentContractError("contact trace geometry id is invalid")
    if (
        type(raw["geom1_name"]) is not str
        or raw["geom1_name"] != EXPECTED_HUMANOID_GEOM_NAMES[geom1_id]
        or type(raw["geom2_name"]) is not str
        or raw["geom2_name"] != EXPECTED_HUMANOID_GEOM_NAMES[geom2_id]
    ):
        raise ExperimentContractError("contact trace geometry name differs from its id")
    normal_force = raw["normal_force_n"]
    if type(normal_force) is not float or not math.isfinite(normal_force) or normal_force < 0.0:
        raise ExperimentContractError("contact trace normal force is invalid")
    involves_floor = FLOOR_GEOM_NAME in (raw["geom1_name"], raw["geom2_name"])
    other_geom = raw["geom2_name"] if raw["geom1_name"] == FLOOR_GEOM_NAME else raw["geom1_name"]
    forbidden = involves_floor and other_geom not in PERMITTED_STATIC_STAND_FLOOR_GEOMS
    return (
        substep,
        contact_index,
        ContactFact(
            contact_index=contact_index,
            geom1_name=raw["geom1_name"],
            geom2_name=raw["geom2_name"],
            involves_floor=involves_floor,
            forbidden_floor_contact=forbidden,
            normal_force_n=normal_force,
        ),
    )


def _validate_contact_trace(trace: TrajectoryTrace) -> None:
    """Recompute per-step contact summaries from ordered contact events."""

    if not trace.events:
        raise ExperimentContractError("development trace is missing its terminal event")
    terminal = trace.events[-1]
    if (
        terminal.sample_index != EXPECTED_STEPS
        or terminal.event_type != TERMINAL_EVENT_TYPE
        or terminal.value != "true"
        or terminal.source != TERMINAL_EVENT_SOURCE
    ):
        raise ExperimentContractError("development trace terminal event differs")
    contact_events = trace.events[:-1]
    counts = [[0] * PHYSICS_SUBSTEPS_PER_CONTROL for _ in range(EXPECTED_STEPS + 1)]
    peak_floor_force = [0.0] * (EXPECTED_STEPS + 1)
    forbidden_floor_contact = [False] * (EXPECTED_STEPS + 1)
    previous_key: tuple[int, int, int] | None = None
    for event in contact_events:
        if not 1 <= event.sample_index <= EXPECTED_STEPS:
            raise ExperimentContractError("contact event is outside a completed control step")
        substep, contact_index, fact = _decode_contact_event(event)
        key = (event.sample_index, substep, contact_index)
        if previous_key is not None and key <= previous_key:
            raise ExperimentContractError("contact events are not in simulator order")
        previous_key = key
        if contact_index != counts[event.sample_index][substep]:
            raise ExperimentContractError("contact event indices are not contiguous by substep")
        counts[event.sample_index][substep] += 1
        if fact.involves_floor:
            peak_floor_force[event.sample_index] = max(
                peak_floor_force[event.sample_index], fact.normal_force_n
            )
        forbidden_floor_contact[event.sample_index] = (
            forbidden_floor_contact[event.sample_index] or fact.forbidden_floor_contact
        )

    count_index = _signal_index(trace, "contact.active_count_by_substep")
    force_index = _signal_index(trace, "contact.floor_normal_force_n")
    forbidden_index = _signal_index(trace, "evaluation.non_foot_floor_contact")
    for sample_index, sample in enumerate(trace.samples):
        expected_counts = tuple(float(value) for value in counts[sample_index])
        if sample.numeric_values[count_index] != expected_counts:
            raise ExperimentContractError("contact count signal differs from ordered events")
        if sample.numeric_values[force_index] != (peak_floor_force[sample_index],):
            raise ExperimentContractError("floor-force signal differs from ordered events")
        if sample.numeric_values[forbidden_index] != (
            float(forbidden_floor_contact[sample_index]),
        ):
            raise ExperimentContractError("forbidden-contact signal differs from ordered events")


def _validate_trace_semantics(seed: int, trace: TrajectoryTrace) -> None:
    """Freeze the exact trace schema and every redundant numeric relationship."""

    if type(seed) is not int or seed not in EVALUATION_SEEDS:
        raise ExperimentContractError("trace metric seed is not in the frozen cohort")
    if type(trace) is not TrajectoryTrace or len(trace.samples) != EXPECTED_STEPS + 1:
        raise ExperimentContractError("metric source must be one complete canonical trace")
    if trace.trace_id != f"tqc-dev1m-seed-{seed}":
        raise ExperimentContractError("development trace id differs from its seed")
    if trace.evidence_class is not EvidenceClass.EXPLORATORY:
        raise ExperimentContractError("development trace must remain exploratory")
    if trace.control_period_seconds != CONTROL_PERIOD_SECONDS:
        raise ExperimentContractError("development trace control period differs")
    if trace.numeric_signals != _numeric_signal_specs():
        raise ExperimentContractError("development trace numeric schema differs")
    if trace.missing_signals != _missing_signals():
        raise ExperimentContractError("development trace missing-signal schema differs")
    binding_roles = tuple(binding.role for binding in trace.artifact_bindings)
    allowed_roles = (
        ("runtime", "development_design", "evaluator", "actor_snapshot"),
        (
            "runtime",
            "development_design",
            "evaluator",
            "actor_snapshot",
            "persisted_actor_artifact",
        ),
    )
    if binding_roles not in allowed_roles:
        raise ExperimentContractError("development trace artifact binding order differs")
    _validate_contact_trace(trace)

    names = tuple(signal.name for signal in trace.numeric_signals)
    indices = {name: names.index(name) for name in names}
    initial_root = trace.samples[0].numeric_values[indices["robot.root_position_world_m"]]
    previous_simulation_time: float | None = None
    low = np.full(ACTION_WIDTH, -0.4, dtype="<f4")
    high = np.full(ACTION_WIDTH, 0.4, dtype="<f4")
    for sample_index, sample in enumerate(trace.samples):
        if sample.time_seconds != sample_index * CONTROL_PERIOD_SECONDS:
            raise ExperimentContractError("trace nominal sample clock differs")
        qpos = sample.numeric_values[indices["robot.qpos"]]
        qvel = sample.numeric_values[indices["robot.qvel"]]
        root = sample.numeric_values[indices["robot.root_position_world_m"]]
        if qpos[:3] != root:
            raise ExperimentContractError("trace qpos root and root-position signal differ")
        observation = sample.numeric_values[indices["controller.observation"]]
        if observation[:22] != qpos[2:] or observation[22:45] != qvel:
            raise ExperimentContractError("trace observation differs from qpos or qvel")
        simulation_time = sample.numeric_values[indices["evaluation.simulation_time_seconds"]][0]
        expected_time = sample_index * CONTROL_PERIOD_SECONDS
        if not math.isclose(simulation_time, expected_time, rel_tol=0.0, abs_tol=1e-12):
            raise ExperimentContractError("direct MuJoCo clock differs from the nominal cadence")
        if sample_index == 0:
            if simulation_time != 0.0:
                raise ExperimentContractError("direct MuJoCo reset clock must equal zero")
        elif previous_simulation_time is None or simulation_time <= previous_simulation_time:
            raise ExperimentContractError("direct MuJoCo clock must increase every control step")
        previous_simulation_time = simulation_time

        action = sample.numeric_values[indices["controller.action"]]
        if any(abs(value) > 1.0 for value in action):
            raise ExperimentContractError("trace normalized action is outside [-1, 1]")
        action32 = np.asarray(action, dtype="<f4")
        expected_physical = np.ascontiguousarray(
            low + np.float32(0.5) * (action32 + np.float32(1.0)) * (high - low),
            dtype="<f4",
        )
        observed_physical = sample.numeric_values[indices["controller.physical_control"]]
        if observed_physical != tuple(float(value) for value in expected_physical):
            raise ExperimentContractError("trace physical control differs from actor transform")
        saturation = sample.numeric_values[indices["controller.saturation"]]
        if saturation != tuple(float(abs(value) == 1.0) for value in action):
            raise ExperimentContractError("trace saturation signal differs from actor action")
        applied_torque = sample.numeric_values[indices["controller.applied_torque"]]
        if observation[253:270] != applied_torque:
            raise ExperimentContractError("trace observation differs from applied torque")

        height = sample.numeric_values[indices["evaluation.root_height_m"]][0]
        up_z = sample.numeric_values[indices["evaluation.torso_up_z"]][0]
        healthy = sample.numeric_values[indices["evaluation.healthy"]][0]
        upright = sample.numeric_values[indices["evaluation.upright"]][0]
        progress = sample.numeric_values[indices["task.progress"]][0]
        if height != root[2]:
            raise ExperimentContractError("trace root-height signals differ")
        if up_z != _torso_up_z_from_qpos(qpos):
            raise ExperimentContractError("trace torso up-axis differs from root quaternion")
        if healthy not in (0.0, 1.0) or healthy != float(1.0 < height < 2.0):
            raise ExperimentContractError("trace healthy signal differs from root height")
        if upright not in (0.0, 1.0) or upright != float(up_z >= MINIMUM_TORSO_UP_Z):
            raise ExperimentContractError("trace upright signal differs from torso orientation")
        if progress != root[0] - initial_root[0]:
            raise ExperimentContractError("task progress differs from direct root-x displacement")


def _metrics_from_trace(seed: int, trace: TrajectoryTrace) -> TQCDevelopmentEpisodeMetrics:
    """Issue metrics only by replaying metric facts from the canonical trace."""

    _validate_trace_semantics(seed, trace)
    indices = {
        name: _signal_index(trace, name)
        for name in (
            "robot.root_position_world_m",
            "evaluation.simulation_time_seconds",
            "controller.action",
            "controller.saturation",
            "evaluation.root_height_m",
            "evaluation.torso_up_z",
            "evaluation.healthy",
            "evaluation.upright",
            "evaluation.non_foot_floor_contact",
        )
    }
    initial = trace.samples[0]
    initial_root = initial.numeric_values[indices["robot.root_position_world_m"]]
    initial_simulation_time = initial.numeric_values[indices["evaluation.simulation_time_seconds"]][
        0
    ]
    if initial_simulation_time != 0.0:
        raise ExperimentContractError("development trace must begin at simulation time zero")
    accumulator = TQCDevelopmentEpisodeAccumulator(
        seed=seed,
        initial_root_position_world_m=initial_root,
        initial_simulation_time_seconds=initial_simulation_time,
    )
    for expected_index, sample in enumerate(trace.samples[1:], start=1):
        root = sample.numeric_values[indices["robot.root_position_world_m"]]
        action = sample.numeric_values[indices["controller.action"]]
        saturation = sample.numeric_values[indices["controller.saturation"]]
        height = sample.numeric_values[indices["evaluation.root_height_m"]][0]
        up_z = sample.numeric_values[indices["evaluation.torso_up_z"]][0]
        healthy = sample.numeric_values[indices["evaluation.healthy"]][0]
        upright = sample.numeric_values[indices["evaluation.upright"]][0]
        forbidden = sample.numeric_values[indices["evaluation.non_foot_floor_contact"]][0]
        if height != root[2]:
            raise ExperimentContractError("trace root-height signals differ")
        if healthy not in (0.0, 1.0) or healthy != float(1.0 < height < 2.0):
            raise ExperimentContractError("trace healthy signal differs from root height")
        if upright not in (0.0, 1.0) or upright != float(up_z >= MINIMUM_TORSO_UP_Z):
            raise ExperimentContractError("trace upright signal differs from torso orientation")
        if forbidden not in (0.0, 1.0):
            raise ExperimentContractError("trace forbidden-contact signal is not boolean")
        if saturation != tuple(float(abs(value) == 1.0) for value in action):
            raise ExperimentContractError("trace saturation signal differs from actor action")
        accumulator.add(
            TQCDevelopmentStepFacts(
                step_index=expected_index,
                simulation_time_seconds=sample.numeric_values[
                    indices["evaluation.simulation_time_seconds"]
                ][0],
                root_position_world_m=root,
                torso_up_z=up_z,
                normalized_action=action,
                non_foot_floor_contact=bool(forbidden),
            )
        )
    return accumulator.finish()


def _evaluate_seed(
    *,
    design: LoadedTQCDevelopmentDesign,
    runtime: TQCDevelopmentRuntimeReceipt,
    actor: ProtectedDeterministicTQCActor,
    seed: int,
) -> TQCDevelopmentSeedEvaluation:
    _validated_design_payload(design)
    if type(runtime) is not TQCDevelopmentRuntimeReceipt:
        raise ExperimentContractError("seed evaluation requires an inspected runtime receipt")
    if type(actor) is not ProtectedDeterministicTQCActor:
        raise ExperimentContractError("seed evaluation requires a protected actor snapshot")
    if type(seed) is not int or seed not in EVALUATION_SEEDS:
        raise ExperimentContractError("seed evaluation uses an undeclared seed")
    actor.assert_integrity()
    config = _environment_config(design)
    environment = make_humanoid_env(config, capture_substep_contacts=True)
    try:
        _validate_episode_environment(environment, config)
        observation, _ignored_reset_info = environment.reset(seed=seed, options=None)
        if type(environment._elapsed_steps) is not int or environment._elapsed_steps != 0:
            raise ExperimentContractError("TimeLimit reset counter must equal zero")
        physical = environment.unwrapped
        abi: HumanoidActuatorABI = validate_humanoid_actuator_abi(physical)
        direct_observation = finite_direct_vector(
            physical._get_obs(), width=OBSERVATION_WIDTH, field="direct reset observation"
        )
        returned_observation = finite_direct_vector(
            observation, width=OBSERVATION_WIDTH, field="returned reset observation"
        )
        if not np.array_equal(returned_observation, direct_observation):
            raise ExperimentContractError("returned and direct reset observations differ")
        initial = _boundary_state(physical)
        if float(initial["simulation_time_seconds"]) != 0.0:
            raise ExperimentContractError("MuJoCo reset time must equal zero")
        bindings = [
            ArtifactBinding("runtime", "tqc-development-runtime", runtime.sha256),
            ArtifactBinding("development_design", "tqc-development-design", design.file_sha256),
            ArtifactBinding(
                "evaluator",
                EVALUATOR_ID,
                sha256_file(Path(__file__)),
            ),
            ArtifactBinding(
                "actor_snapshot",
                actor.actor_id,
                actor.snapshot_state_sha256,
            ),
        ]
        if actor.persisted_artifact_sha256 is not None:
            bindings.append(
                ArtifactBinding(
                    "persisted_actor_artifact",
                    f"strict-tqc-actor-{actor.persisted_artifact_byte_count}-bytes",
                    actor.persisted_artifact_sha256,
                )
            )
        recorder = TraceRecorder(
            trace_id=f"tqc-dev1m-seed-{seed}",
            evidence_class=EvidenceClass.EXPLORATORY,
            control_period_seconds=CONTROL_PERIOD_SECONDS,
            numeric_signals=_numeric_signal_specs(),
            missing_signals=_missing_signals(),
            artifact_bindings=tuple(bindings),
        )
        zeros = np.zeros(ACTION_WIDTH, dtype=np.float64)
        reset_control = finite_direct_vector(
            physical.data.ctrl,
            width=ACTION_WIDTH,
            field="direct reset MuJoCo physical control",
        )
        if not np.array_equal(reset_control, zeros):
            raise ExperimentContractError("MuJoCo reset control must be exactly zero")
        if (
            type(physical.data.ncon) is not int
            or physical.data.ncon != 0
            or type(physical.last_control_step_contact_samples) is not tuple
            or physical.last_control_step_contact_samples != ()
            or type(physical.last_control_step_substeps) is not int
            or physical.last_control_step_substeps != 0
        ):
            raise ExperimentContractError(
                "reset contact sentinels require zero active and prior-interval contacts"
            )
        reset_torque = finite_direct_vector(
            np.asarray(physical.data.qfrc_actuator)[list(abi.qvel_indices)],
            width=ACTION_WIDTH,
            field="direct reset generalized actuator torque",
        )
        recorder.add_sample(
            _trace_sample(
                initial,
                controller_observation=returned_observation,
                normalized_action=zeros,
                physical_control=reset_control,
                applied_torque=reset_torque,
                floor_normal_force_n=0.0,
                contact_count_by_substep=(0,) * PHYSICS_SUBSTEPS_PER_CONTROL,
                non_foot_floor_contact=False,
                initial_root_x_m=float(initial["root_position_world_m"][0]),
            )
        )

        current_observation = returned_observation
        for step_index in range(1, EXPECTED_STEPS + 1):
            action = actor.act(current_observation)
            if not environment.action_space.contains(action.physical):
                raise ExperimentContractError("actor physical action is outside Humanoid-v5")
            (
                next_observation,
                _ignored_reward,
                terminated,
                truncated,
                _ignored_info,
            ) = environment.step(action.physical.copy())
            if type(terminated) is not bool or type(truncated) is not bool:
                raise ExperimentContractError("Gymnasium termination flags must be exact booleans")
            if terminated:
                raise ExperimentContractError("unhealthy termination was not disabled")
            expected_truncated = step_index == EXPECTED_STEPS
            if truncated is not expected_truncated:
                raise ExperimentContractError("TimeLimit truncation did not occur exactly at 1000")
            if (
                type(environment._elapsed_steps) is not int
                or environment._elapsed_steps != step_index
            ):
                raise ExperimentContractError("TimeLimit counter differs from the control step")
            direct_next_observation = finite_direct_vector(
                physical._get_obs(),
                width=OBSERVATION_WIDTH,
                field="direct post-step observation",
            )
            returned_next_observation = finite_direct_vector(
                next_observation,
                width=OBSERVATION_WIDTH,
                field="returned post-step observation",
            )
            if not np.array_equal(returned_next_observation, direct_next_observation):
                raise ExperimentContractError("returned and direct post-step observations differ")

            state = _boundary_state(physical)
            expected_time = float(initial["simulation_time_seconds"]) + (
                step_index * CONTROL_PERIOD_SECONDS
            )
            if not math.isclose(
                float(state["simulation_time_seconds"]),
                expected_time,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ExperimentContractError("MuJoCo simulation clock differs")
            physical_control = finite_direct_vector(
                physical.data.ctrl,
                width=ACTION_WIDTH,
                field="direct MuJoCo physical control",
            )
            if not np.array_equal(
                physical_control,
                np.asarray(action.physical, dtype=np.float64),
            ):
                raise ExperimentContractError("MuJoCo control differs from the actor action")
            applied_torque = finite_direct_vector(
                np.asarray(physical.data.qfrc_actuator)[list(abi.qvel_indices)],
                width=ACTION_WIDTH,
                field="direct post-transmission generalized actuator torque",
            )
            (
                contact_event_values,
                contact_count_by_substep,
                peak_floor_force,
                forbidden_contact,
            ) = _direct_contact_event_values(environment)
            recorder.add_sample(
                _trace_sample(
                    state,
                    controller_observation=returned_next_observation,
                    normalized_action=np.asarray(action.normalized, dtype=np.float64),
                    physical_control=physical_control,
                    applied_torque=applied_torque,
                    floor_normal_force_n=peak_floor_force,
                    contact_count_by_substep=contact_count_by_substep,
                    non_foot_floor_contact=forbidden_contact,
                    initial_root_x_m=float(initial["root_position_world_m"][0]),
                )
            )
            for contact_event_value in contact_event_values:
                recorder.add_event(
                    sample_index=step_index,
                    event_type=CONTACT_EVENT_TYPE,
                    value=contact_event_value,
                    source=CONTACT_EVENT_SOURCE,
                )
            if truncated:
                recorder.add_event(
                    sample_index=step_index,
                    event_type=TERMINAL_EVENT_TYPE,
                    value="true",
                    source=TERMINAL_EVENT_SOURCE,
                )
            current_observation = returned_next_observation

        trace = recorder.finish()
        if trace.evidence_class is not EvidenceClass.EXPLORATORY:
            raise ExperimentContractError("development trace evidence class changed")
        metrics = _metrics_from_trace(seed, trace)
        observations = _controller_observation_matrix(trace)
        actor.assert_integrity()
        return TQCDevelopmentSeedEvaluation(
            seed=seed,
            metrics=metrics,
            trace=trace,
            trace_sha256=trace.sha256,
            controller_observation_shape=observations.shape,
            controller_observation_sha256=_array_sha256(observations),
            _issuer=_SEED_EVALUATION_ISSUER,
        )
    finally:
        environment.close()


def evaluate_tqc_development_actor(
    design: LoadedTQCDevelopmentDesign,
    actor: ProtectedDeterministicTQCActor,
) -> TQCDevelopmentEvaluation:
    """Evaluate one fixed actor once on every design-supplied seed, in order."""

    raw = _validated_design_payload(design)
    if type(actor) is not ProtectedDeterministicTQCActor:
        raise ExperimentContractError("evaluator accepts only a protected TQC actor snapshot")
    if raw["evaluation"]["metric_source"] != METRIC_SOURCE:
        raise ExperimentContractError("development metric source differs")
    if raw["evaluation"]["max_episode_steps"] != EXPECTED_STEPS:
        raise ExperimentContractError("development horizon differs")
    if raw["environment"]["terminate_when_unhealthy"] is not False:
        raise ExperimentContractError("development evaluation must measure through falls")
    actor.assert_integrity()
    runtime = inspect_tqc_development_runtime(design)
    episodes = tuple(
        _evaluate_seed(
            design=design,
            runtime=runtime,
            actor=actor,
            seed=seed,
        )
        for seed in EVALUATION_SEEDS
    )
    actor.assert_integrity()
    if inspect_tqc_development_runtime(design) != runtime:
        raise ExperimentContractError("development evaluator runtime changed during evaluation")
    cohort = summarize_tqc_development_cohort(
        tuple(episode.metrics for episode in episodes),
        expected_seeds=EVALUATION_SEEDS,
    )
    return TQCDevelopmentEvaluation(
        design_file_sha256=hashlib.sha256(design.encoded_bytes).hexdigest(),
        design_semantic_sha256=sha256_json(raw),
        runtime=runtime,
        actor_id=actor.actor_id,
        actor_source_kind=actor.source_kind,
        actor_snapshot_state_sha256=actor.snapshot_state_sha256,
        persisted_actor_artifact_sha256=actor.persisted_artifact_sha256,
        persisted_actor_artifact_byte_count=actor.persisted_artifact_byte_count,
        actor_equivalence_observation_sha256=actor.equivalence_observation_sha256,
        actor_deterministic_normalized_action_sha256=(actor.protected_normalized_action_sha256),
        actor_deterministic_physical_action_sha256=(actor.protected_physical_action_sha256),
        episodes=episodes,
        cohort=cohort,
        _issuer=_EVALUATION_ISSUER,
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "EVALUATOR_ID",
    "DeterministicTQCAction",
    "ProtectedDeterministicTQCActor",
    "TQCDevelopmentEvaluation",
    "TQCDevelopmentRuntimeReceipt",
    "TQCDevelopmentSeedEvaluation",
    "evaluate_tqc_development_actor",
    "inspect_tqc_development_runtime",
]
