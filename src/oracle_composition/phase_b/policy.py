"""Exact 708-D full-authority warm start and strict actor export."""

from __future__ import annotations

import hashlib
import io
import math
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import array_sha256
from oracle_composition.experiments.artifact_io import (
    PublishedArtifact,
    publish_bytes_without_overwrite,
)
from oracle_composition.experiments.external_tqc_initialization_identity import (
    EXPERT_ACTOR_NPZ_SHA256,
    EXPERT_ACTOR_SCHEMA_SHA256,
    EXPERT_ACTOR_STATE_SHA256,
)
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_actor_npz import (
    ACTION_WIDTH,
    OBSERVATION_WIDTH,
    LoadedTQCActor,
    load_actor_npz,
)

from .strict_npz import decode_strict_npz

try:
    import torch
    from torch import nn
    from torch.nn import functional
except ImportError as exc:  # pragma: no cover - the Phase B runtime requires the train extra
    raise ExperimentContractError("Torch is required for the Phase B policy") from exc

REFERENCE_HORIZON = 8
REFERENCE_WIDTH = 45
REFERENCE_INPUT_WIDTH = REFERENCE_HORIZON * REFERENCE_WIDTH
POLICY_INPUT_WIDTH = OBSERVATION_WIDTH + REFERENCE_INPUT_WIDTH
HIDDEN_WIDTH = 256
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
VALUE_ARCHITECTURE_ID = "independent_708_256_256_1_relu/v1"
POLICY_ARCHITECTURE_ID = "full_authority_708_256_256_17_state_dependent_std/v1"
INPUT_LAYOUT_ID = "state_float32_348_then_reference_row_major_float32_8x45/v1"
ACTION_ADAPTER_ID = "strict_float32_low_plus_half_u_plus_one_span/v1"
EXPORT_FORMAT_ID = "strict_full_authority_actor_npz_npy1_c_order_no_pickle/v1"
MAX_EXPORT_BYTES = 4 * 1024 * 1024
MAX_EXPORT_MEMBER_BYTES = 2 * 1024 * 1024
MAX_EXPORT_EXPANSION_BYTES = 8 * 1024 * 1024

_INPUT_ISSUER = object()
_FLOAT32 = np.dtype("<f4")
_INT64 = np.dtype("<i8")
_ACTOR_PARAMETER_NAMES = (
    "latent_pi.0.weight",
    "latent_pi.0.bias",
    "latent_pi.2.weight",
    "latent_pi.2.bias",
    "mu.weight",
    "mu.bias",
    "log_std.weight",
    "log_std.bias",
)
_PARAMETER_SHAPES = {
    "latent_pi.0.weight": (HIDDEN_WIDTH, POLICY_INPUT_WIDTH),
    "latent_pi.0.bias": (HIDDEN_WIDTH,),
    "latent_pi.2.weight": (HIDDEN_WIDTH, HIDDEN_WIDTH),
    "latent_pi.2.bias": (HIDDEN_WIDTH,),
    "mu.weight": (ACTION_WIDTH, HIDDEN_WIDTH),
    "mu.bias": (ACTION_WIDTH,),
    "log_std.weight": (ACTION_WIDTH, HIDDEN_WIDTH),
    "log_std.bias": (ACTION_WIDTH,),
}
_EXPORT_SCHEMA: Mapping[str, tuple[tuple[int, ...], np.dtype[object]]] = {
    **{name: (shape, _FLOAT32) for name, shape in _PARAMETER_SHAPES.items()},
    "action_low": ((ACTION_WIDTH,), _FLOAT32),
    "action_high": ((ACTION_WIDTH,), _FLOAT32),
    "input_layout": ((3,), _INT64),
    "log_std_bounds": ((2,), _FLOAT32),
    "format_version": ((1,), _INT64),
    "source_actor_sha256": ((1,), np.dtype("|S64")),
}


class StrictPolicyInput:
    """Capability-bearing input assembled only as state then row-major window."""

    __slots__ = ("_array", "layout_id")

    def __init__(self, value: np.ndarray, *, layout_id: str, _issuer: object = None) -> None:
        if _issuer is not _INPUT_ISSUER:
            raise ExperimentContractError("policy inputs must be issued by compose_policy_input")
        self._array = value
        self.layout_id = layout_id

    @property
    def array(self) -> np.ndarray:
        return self._array


def _strict_float32(value: object, *, shape_tail: tuple[int, ...], field: str) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype.str != "<f4"
        or value.shape[-len(shape_tail) :] != shape_tail
        or value.ndim not in {len(shape_tail), len(shape_tail) + 1}
        or not value.flags.c_contiguous
        or not np.isfinite(value).all()
    ):
        raise ExperimentContractError(
            f"{field} must be finite C-order little-endian float32 with tail {shape_tail}"
        )
    return value


def compose_policy_input(state: np.ndarray, reference_window: np.ndarray) -> StrictPolicyInput:
    """Build exact ``state[348] || reference[8,45]`` bytes without coercion."""

    checked_state = _strict_float32(state, shape_tail=(OBSERVATION_WIDTH,), field="state")
    checked_reference = _strict_float32(
        reference_window,
        shape_tail=(REFERENCE_HORIZON, REFERENCE_WIDTH),
        field="reference_window",
    )
    if checked_state.ndim == 1:
        if checked_reference.ndim != 2:
            raise ExperimentContractError("state and reference_window batch ranks differ")
        flat_reference = checked_reference.reshape(REFERENCE_INPUT_WIDTH)
        value = np.concatenate((checked_state, flat_reference))
    else:
        if checked_reference.ndim != 3 or checked_reference.shape[0] != checked_state.shape[0]:
            raise ExperimentContractError("state and reference_window batches differ")
        flat_reference = checked_reference.reshape(
            checked_reference.shape[0], REFERENCE_INPUT_WIDTH
        )
        value = np.concatenate((checked_state, flat_reference), axis=1)
    result = np.ascontiguousarray(value, dtype="<f4")
    return StrictPolicyInput(result, layout_id=INPUT_LAYOUT_ID, _issuer=_INPUT_ISSUER)


def _validate_policy_input(value: object) -> np.ndarray:
    if type(value) is not StrictPolicyInput or value.layout_id != INPUT_LAYOUT_ID:
        raise ExperimentContractError("policy input layout or issuing authority differs")
    result = value.array
    return _strict_float32(result, shape_tail=(POLICY_INPUT_WIDTH,), field="policy input")


@dataclass(frozen=True, slots=True)
class ActorDistribution:
    mean: np.ndarray
    log_std: np.ndarray


@dataclass(frozen=True, slots=True)
class FullAuthorityAction:
    mean: np.ndarray
    log_std: np.ndarray
    pre_tanh: np.ndarray
    normalized: np.ndarray
    physical: np.ndarray


def exact_physical_action(
    normalized: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
    *,
    adapter_id: str = ACTION_ADAPTER_ID,
) -> np.ndarray:
    """Apply the strict runtime's exact float32 operation order."""

    if adapter_id != ACTION_ADAPTER_ID:
        raise ExperimentContractError("generic or unknown action rescaling is forbidden")
    value = _strict_float32(normalized, shape_tail=(ACTION_WIDTH,), field="normalized action")
    low = _strict_float32(action_low, shape_tail=(ACTION_WIDTH,), field="action_low")
    high = _strict_float32(action_high, shape_tail=(ACTION_WIDTH,), field="action_high")
    if low.ndim != 1 or high.ndim != 1:
        raise ExperimentContractError("action bounds must be unbatched")
    if np.any(value < np.float32(-1.0)) or np.any(value > np.float32(1.0)):
        raise ExperimentContractError("normalized action is outside [-1, 1]")
    return np.ascontiguousarray(
        low + np.float32(0.5) * (value + np.float32(1.0)) * (high - low),
        dtype="<f4",
    )


def squashed_gaussian_log_likelihood(
    pre_tanh: torch.Tensor,
    mean: torch.Tensor,
    log_std: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Diagonal Normal likelihood with the required tanh Jacobian correction."""

    if (
        pre_tanh.dtype is not torch.float32
        or mean.dtype is not torch.float32
        or log_std.dtype is not torch.float32
        or pre_tanh.shape != mean.shape
        or mean.shape != log_std.shape
        or mean.shape[-1] != ACTION_WIDTH
    ):
        raise ExperimentContractError("likelihood tensors must be matching float32[...,17]")
    if not all(torch.isfinite(item).all() for item in (pre_tanh, mean, log_std)):
        raise ExperimentContractError("likelihood tensors must be finite")
    if torch.any(log_std < LOG_STD_MIN) or torch.any(log_std > LOG_STD_MAX):
        raise ExperimentContractError("likelihood log_std is outside the frozen clamp")
    gaussian = -0.5 * (
        ((pre_tanh - mean) / torch.exp(log_std)).square() + 2.0 * log_std + math.log(2.0 * math.pi)
    )
    squashed = torch.tanh(pre_tanh)
    correction = torch.log(1.0 - squashed.square() + epsilon)
    return (gaussian - correction).sum(dim=-1)


class FullAuthorityActor(nn.Module):
    """Complete-action actor initialized exactly from the admitted expert."""

    ortho_init = False
    input_layout_id = INPUT_LAYOUT_ID
    architecture_id = POLICY_ARCHITECTURE_ID
    log_std_min = LOG_STD_MIN
    log_std_max = LOG_STD_MAX

    def __init__(self, parameters: Mapping[str, np.ndarray], *, source_actor_sha256: str) -> None:
        super().__init__()
        if source_actor_sha256 != EXPERT_ACTOR_NPZ_SHA256:
            raise ExperimentContractError("full-authority warm start requires the exact expert")
        arrays = _validate_parameter_arrays(parameters)
        self.latent_0 = nn.Linear(POLICY_INPUT_WIDTH, HIDDEN_WIDTH, bias=True)
        self.latent_2 = nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH, bias=True)
        self.mu = nn.Linear(HIDDEN_WIDTH, ACTION_WIDTH, bias=True)
        self.log_std = nn.Linear(HIDDEN_WIDTH, ACTION_WIDTH, bias=True)
        self.source_actor_sha256 = source_actor_sha256
        self.register_buffer(
            "action_low",
            torch.from_numpy(np.full(ACTION_WIDTH, -0.4, dtype="<f4")),
        )
        self.register_buffer(
            "action_high",
            torch.from_numpy(np.full(ACTION_WIDTH, 0.4, dtype="<f4")),
        )
        module_parameters = {
            "latent_pi.0.weight": self.latent_0.weight,
            "latent_pi.0.bias": self.latent_0.bias,
            "latent_pi.2.weight": self.latent_2.weight,
            "latent_pi.2.bias": self.latent_2.bias,
            "mu.weight": self.mu.weight,
            "mu.bias": self.mu.bias,
            "log_std.weight": self.log_std.weight,
            "log_std.bias": self.log_std.bias,
        }
        with torch.no_grad():
            for name, parameter in module_parameters.items():
                parameter.copy_(torch.from_numpy(arrays[name]))
        verify_actor_warm_start(self, arrays)

    @classmethod
    def from_expert(cls, actor: LoadedTQCActor) -> FullAuthorityActor:
        if (
            type(actor) is not LoadedTQCActor
            or actor.content_sha256 != EXPERT_ACTOR_NPZ_SHA256
            or actor.state_sha256 != EXPERT_ACTOR_STATE_SHA256
            or actor.schema_sha256 != EXPERT_ACTOR_SCHEMA_SHA256
        ):
            raise ExperimentContractError("strict expert identity differs")
        base = actor.arrays
        expanded = {
            name: (
                np.ascontiguousarray(
                    np.concatenate(
                        (
                            base[name],
                            np.zeros((HIDDEN_WIDTH, REFERENCE_INPUT_WIDTH), dtype="<f4"),
                        ),
                        axis=1,
                    ),
                    dtype="<f4",
                )
                if name == "latent_pi.0.weight"
                else np.array(base[name], dtype="<f4", order="C", copy=True)
            )
            for name in _ACTOR_PARAMETER_NAMES
        }
        return cls(expanded, source_actor_sha256=actor.content_sha256)

    def parameter_arrays(self) -> dict[str, np.ndarray]:
        mapping = {
            "latent_pi.0.weight": self.latent_0.weight,
            "latent_pi.0.bias": self.latent_0.bias,
            "latent_pi.2.weight": self.latent_2.weight,
            "latent_pi.2.bias": self.latent_2.bias,
            "mu.weight": self.mu.weight,
            "mu.bias": self.mu.bias,
            "log_std.weight": self.log_std.weight,
            "log_std.bias": self.log_std.bias,
        }
        return {
            name: np.array(
                value.detach().cpu().numpy(),
                dtype="<f4",
                order="C",
                copy=True,
            )
            for name, value in mapping.items()
        }

    def _distribution_tensors(self, policy_input: StrictPolicyInput) -> tuple[torch.Tensor, ...]:
        value = _validate_policy_input(policy_input)
        tensor = torch.from_numpy(value.copy(order="C"))
        state = tensor[..., :OBSERVATION_WIDTH]
        reference = tensor[..., OBSERVATION_WIDTH:]
        first = functional.linear(
            state,
            self.latent_0.weight[:, :OBSERVATION_WIDTH],
            self.latent_0.bias,
        ) + functional.linear(
            reference,
            self.latent_0.weight[:, OBSERVATION_WIDTH:],
            None,
        )
        latent = functional.relu(first)
        latent = functional.relu(self.latent_2(latent))
        mean = self.mu(latent)
        log_std = torch.clamp(self.log_std(latent), min=LOG_STD_MIN, max=LOG_STD_MAX)
        return mean, log_std

    def distribution(self, policy_input: StrictPolicyInput) -> ActorDistribution:
        with torch.inference_mode():
            mean, log_std = self._distribution_tensors(policy_input)
        return ActorDistribution(
            mean=np.ascontiguousarray(mean.detach().cpu().numpy(), dtype="<f4"),
            log_std=np.ascontiguousarray(log_std.detach().cpu().numpy(), dtype="<f4"),
        )

    def act(
        self,
        policy_input: StrictPolicyInput,
        *,
        epsilon: np.ndarray | None = None,
    ) -> FullAuthorityAction:
        with torch.inference_mode():
            mean, log_std = self._distribution_tensors(policy_input)
            if epsilon is None:
                pre_tanh = mean
            else:
                noise = _strict_float32(epsilon, shape_tail=(ACTION_WIDTH,), field="epsilon")
                if tuple(noise.shape) != tuple(mean.shape):
                    raise ExperimentContractError("epsilon shape differs from actor output")
                pre_tanh = mean + torch.exp(log_std) * torch.from_numpy(noise.copy(order="C"))
            normalized_tensor = torch.tanh(pre_tanh)
        arrays = {
            "mean": mean,
            "log_std": log_std,
            "pre_tanh": pre_tanh,
            "normalized": normalized_tensor,
        }
        converted = {
            name: np.ascontiguousarray(value.detach().cpu().numpy(), dtype="<f4")
            for name, value in arrays.items()
        }
        low = np.ascontiguousarray(self.action_low.detach().cpu().numpy(), dtype="<f4")
        high = np.ascontiguousarray(self.action_high.detach().cpu().numpy(), dtype="<f4")
        return FullAuthorityAction(
            **converted,
            physical=exact_physical_action(converted["normalized"], low, high),
        )

    def log_likelihood(
        self,
        policy_input: StrictPolicyInput,
        pre_tanh: np.ndarray,
    ) -> np.ndarray:
        checked = _strict_float32(pre_tanh, shape_tail=(ACTION_WIDTH,), field="pre_tanh")
        mean, log_std = self._distribution_tensors(policy_input)
        if tuple(checked.shape) != tuple(mean.shape):
            raise ExperimentContractError("pre_tanh shape differs from actor output")
        values = squashed_gaussian_log_likelihood(
            torch.from_numpy(checked.copy(order="C")),
            mean,
            log_std,
        )
        return np.ascontiguousarray(values.detach().cpu().numpy(), dtype="<f4")


class FullAuthorityPolicy(nn.Module):
    """Separate actor and freshly seeded value function; no shared features."""

    ortho_init = False

    def __init__(self, actor: FullAuthorityActor, *, value_seed: int) -> None:
        super().__init__()
        if type(actor) is not FullAuthorityActor:
            raise ExperimentContractError("policy requires the exact full-authority actor")
        if type(value_seed) is not int or value_seed <= 0:
            raise ExperimentContractError("value_seed must be a positive integer")
        self.actor = actor
        self.value_seed = value_seed
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(value_seed)
            self.value = nn.Sequential(
                nn.Linear(POLICY_INPUT_WIDTH, HIDDEN_WIDTH),
                nn.ReLU(),
                nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH),
                nn.ReLU(),
                nn.Linear(HIDDEN_WIDTH, 1),
            )

    def value_estimate(self, policy_input: StrictPolicyInput) -> torch.Tensor:
        value = _validate_policy_input(policy_input)
        return self.value(torch.from_numpy(value.copy(order="C")))


def build_full_authority_policy(
    actor_path: Path,
    *,
    expected_sha256: str = EXPERT_ACTOR_NPZ_SHA256,
    value_seed: int,
    normalizer: object = None,
) -> FullAuthorityPolicy:
    """Fail before policy construction on source identity or normalizer drift."""

    if normalizer is not None:
        raise ExperimentContractError("observation or reward normalizers are forbidden")
    if expected_sha256 != EXPERT_ACTOR_NPZ_SHA256:
        raise ExperimentContractError("expert NPZ hash differs from the frozen identity")
    loaded = load_actor_npz(Path(actor_path), expected_sha256=expected_sha256)
    actor = FullAuthorityActor.from_expert(loaded)
    return FullAuthorityPolicy(actor, value_seed=value_seed)


def verify_actor_warm_start(
    actor: FullAuthorityActor,
    expected_parameters: Mapping[str, np.ndarray],
) -> dict[str, object]:
    """Require copied bits, positive-zero reference columns, clamp, and both heads."""

    if type(actor) is not FullAuthorityActor:
        raise ExperimentContractError("actor type differs from the full-authority contract")
    if (
        actor.ortho_init is not False
        or actor.log_std_min != LOG_STD_MIN
        or actor.log_std_max != LOG_STD_MAX
        or type(actor.log_std) is not nn.Linear
    ):
        raise ExperimentContractError("actor initialization or state-dependent log-std differs")
    expected = _validate_parameter_arrays(expected_parameters)
    observed = actor.parameter_arrays()
    reference = observed["latent_pi.0.weight"][:, OBSERVATION_WIDTH:]
    positive_zero = np.zeros_like(reference)
    if reference.tobytes(order="C") != positive_zero.tobytes(order="C"):
        raise ExperimentContractError("reference columns are not exact positive zero")
    copied: dict[str, object] = {}
    for name in _ACTOR_PARAMETER_NAMES:
        left = expected[name]
        right = observed[name]
        if left.tobytes(order="C") != right.tobytes(order="C"):
            raise ExperimentContractError(f"copied actor parameter differs: {name}")
        copied[name] = {
            "bitwise_equal": True,
            "sha256": array_sha256(right),
        }
    return {
        "copied_parameters": copied,
        "every_copied_parameter_bitwise_equal": True,
        "reference_columns_exact_positive_zero": True,
        "reference_columns_sha256": array_sha256(np.ascontiguousarray(reference)),
        "state_dependent_log_std": True,
        "log_std_clamp": [LOG_STD_MIN, LOG_STD_MAX],
        "ortho_init": False,
    }


def verify_optimizer_authority(
    policy: FullAuthorityPolicy,
    optimizer: torch.optim.Optimizer,
    *,
    expected_learning_rate: float = 3e-4,
    require_empty_state: bool = False,
    stage: str = "construction",
) -> dict[str, object]:
    """Validate and receipt exact Adam ownership, uniqueness, groups, and state."""

    if type(policy) is not FullAuthorityPolicy or type(optimizer) is not torch.optim.Adam:
        raise ExperimentContractError("optimizer must be the exact Phase B Adam authority")
    if type(expected_learning_rate) is not float or expected_learning_rate <= 0.0:
        raise ExperimentContractError("optimizer expected learning rate is invalid")
    named = tuple(policy.named_parameters())
    expected_parameters = tuple(parameter for _name, parameter in named)
    expected_ids = tuple(id(parameter) for parameter in expected_parameters)
    if len(set(expected_ids)) != len(expected_ids):
        raise ExperimentContractError("Phase B policy exposes duplicate parameter identities")
    if len(optimizer.param_groups) != 1:
        raise ExperimentContractError("optimizer parameter-group count differs")
    group = optimizer.param_groups[0]
    expected_group_keys = {
        "amsgrad",
        "betas",
        "capturable",
        "decoupled_weight_decay",
        "differentiable",
        "eps",
        "foreach",
        "fused",
        "lr",
        "maximize",
        "params",
        "weight_decay",
    }
    if set(group) != expected_group_keys:
        raise ExperimentContractError("optimizer parameter-group fields differ")
    observed_parameters = tuple(group["params"])
    observed_ids = tuple(id(parameter) for parameter in observed_parameters)
    if len(set(observed_ids)) != len(observed_ids):
        raise ExperimentContractError("optimizer contains duplicate parameter membership")
    if observed_ids != expected_ids:
        raise ExperimentContractError("optimizer omits or adds Phase B policy parameters")
    expected_hyperparameters = {
        "amsgrad": False,
        "betas": (0.9, 0.999),
        "capturable": False,
        "decoupled_weight_decay": False,
        "differentiable": False,
        "eps": 1e-8,
        "foreach": None,
        "fused": None,
        "lr": expected_learning_rate,
        "maximize": False,
        "weight_decay": 0,
    }
    if any(group[name] != value for name, value in expected_hyperparameters.items()):
        raise ExperimentContractError("optimizer hyperparameters differ from the frozen PPO recipe")
    state_ids = {id(parameter) for parameter in optimizer.state}
    if not state_ids.issubset(set(expected_ids)):
        raise ExperimentContractError("optimizer state contains a foreign parameter")
    if require_empty_state and optimizer.state:
        raise ExperimentContractError("fresh optimizer state is not empty")
    parameter_names = [name for name, _parameter in named]
    membership_sha256 = hashlib.sha256("\0".join(parameter_names).encode("utf-8")).hexdigest()
    return {
        "group_count": 1,
        "groups": [
            {
                "hyperparameters": {
                    **expected_hyperparameters,
                    "betas": list(expected_hyperparameters["betas"]),
                },
                "parameter_names": parameter_names,
            }
        ],
        "membership_complete": True,
        "membership_sha256": membership_sha256,
        "optimizer_type": "torch.optim.Adam",
        "parameter_count": len(parameter_names),
        "parameter_membership_unique": True,
        "stage": stage,
        "state_empty": not optimizer.state,
        "state_entry_count": len(optimizer.state),
    }


def _validate_parameter_arrays(values: Mapping[str, object]) -> dict[str, np.ndarray]:
    if type(values) is not dict or set(values) != set(_ACTOR_PARAMETER_NAMES):
        raise ExperimentContractError("full-authority actor parameter keys differ")
    result: dict[str, np.ndarray] = {}
    for name in _ACTOR_PARAMETER_NAMES:
        value = values[name]
        if (
            type(value) is not np.ndarray
            or value.dtype.str != "<f4"
            or value.shape != _PARAMETER_SHAPES[name]
            or not value.flags.c_contiguous
            or not np.isfinite(value).all()
        ):
            raise ExperimentContractError(f"full-authority actor parameter differs: {name}")
        result[name] = np.array(value, dtype="<f4", order="C", copy=True)
    return result


def _export_arrays(actor: FullAuthorityActor) -> dict[str, np.ndarray]:
    result = actor.parameter_arrays()
    result.update(
        {
            "action_low": np.ascontiguousarray(
                actor.action_low.detach().cpu().numpy(), dtype="<f4"
            ),
            "action_high": np.ascontiguousarray(
                actor.action_high.detach().cpu().numpy(), dtype="<f4"
            ),
            "input_layout": np.asarray(
                [OBSERVATION_WIDTH, REFERENCE_HORIZON, REFERENCE_WIDTH], dtype="<i8"
            ),
            "log_std_bounds": np.asarray([LOG_STD_MIN, LOG_STD_MAX], dtype="<f4"),
            "format_version": np.asarray([1], dtype="<i8"),
            "source_actor_sha256": np.asarray(
                [actor.source_actor_sha256.encode("ascii")], dtype="|S64"
            ),
        }
    )
    return _validate_export_arrays(result)


def _validate_export_arrays(values: Mapping[str, object]) -> dict[str, np.ndarray]:
    if type(values) is not dict or set(values) != set(_EXPORT_SCHEMA):
        raise ExperimentContractError("strict actor export keys differ")
    result: dict[str, np.ndarray] = {}
    for name, (shape, dtype) in _EXPORT_SCHEMA.items():
        value = values[name]
        if (
            type(value) is not np.ndarray
            or value.dtype.str != dtype.str
            or value.shape != shape
            or not value.flags.c_contiguous
        ):
            raise ExperimentContractError(f"strict actor export member differs: {name}")
        if value.dtype.kind == "f" and not np.isfinite(value).all():
            raise ExperimentContractError(f"strict actor export member is non-finite: {name}")
        result[name] = np.array(value, dtype=dtype, order="C", copy=True)
    if not np.array_equal(
        result["input_layout"],
        np.asarray([OBSERVATION_WIDTH, REFERENCE_HORIZON, REFERENCE_WIDTH], dtype="<i8"),
    ):
        raise ExperimentContractError("strict actor export input layout differs")
    if not np.array_equal(
        result["log_std_bounds"], np.asarray([LOG_STD_MIN, LOG_STD_MAX], dtype="<f4")
    ):
        raise ExperimentContractError("strict actor export log-std clamp differs")
    if not np.array_equal(result["format_version"], np.asarray([1], dtype="<i8")):
        raise ExperimentContractError("strict actor export version differs")
    source = bytes(result["source_actor_sha256"][0]).decode("ascii")
    if source != EXPERT_ACTOR_NPZ_SHA256:
        raise ExperimentContractError("strict actor export source identity differs")
    if not np.array_equal(result["action_low"], np.full(ACTION_WIDTH, -0.4, dtype="<f4")):
        raise ExperimentContractError("strict actor export lower action bound differs")
    if not np.array_equal(result["action_high"], np.full(ACTION_WIDTH, 0.4, dtype="<f4")):
        raise ExperimentContractError("strict actor export upper action bound differs")
    _validate_parameter_arrays({name: result[name] for name in _ACTOR_PARAMETER_NAMES})
    return result


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, value, version=(1, 0), allow_pickle=False)
    return stream.getvalue()


def _encode_export_arrays(values: Mapping[str, object]) -> bytes:
    arrays = _validate_export_arrays(values)
    stream = io.BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name in _EXPORT_SCHEMA:
            member = ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o100600 << 16
            archive.writestr(member, _npy_bytes(arrays[name]), compresslevel=9)
    payload = stream.getvalue()
    if not 0 < len(payload) <= MAX_EXPORT_BYTES:
        raise ExperimentContractError("strict actor export exceeds its byte bound")
    return payload


def encode_full_authority_actor(actor: FullAuthorityActor) -> bytes:
    return _encode_export_arrays(_export_arrays(actor))


def export_full_authority_actor(path: Path, actor: FullAuthorityActor) -> PublishedArtifact:
    return publish_bytes_without_overwrite(Path(path), encode_full_authority_actor(actor))


def _read_export(path: Path, expected_sha256: str) -> bytes:
    candidate = Path(path)
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise ExperimentContractError("strict actor export is unavailable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ExperimentContractError("strict actor export must be a regular non-linked file")
    if not 0 < before.st_size <= MAX_EXPORT_BYTES:
        raise ExperimentContractError("strict actor export byte count differs")
    payload = candidate.read_bytes()
    after = candidate.lstat()
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity):
        raise ExperimentContractError("strict actor export changed while read")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ExperimentContractError("strict actor export SHA-256 differs")
    return payload


@dataclass(frozen=True, slots=True)
class LoadedFullAuthorityActor:
    actor: FullAuthorityActor
    content_sha256: str
    byte_count: int
    member_hashes: Mapping[str, str]


def load_full_authority_actor_arrays(
    path: Path,
    *,
    expected_sha256: str,
) -> Mapping[str, np.ndarray]:
    """Decode and authenticate an export without constructing a Torch module."""

    payload = _read_export(Path(path), expected_sha256)
    validated = _validate_export_arrays(
        decode_strict_npz(
            payload,
            schema=_EXPORT_SCHEMA,
            archive_label="strict actor export",
            maximum_member_bytes=MAX_EXPORT_MEMBER_BYTES,
            maximum_total_bytes=MAX_EXPORT_EXPANSION_BYTES,
        )
    )
    if _encode_export_arrays(validated) != payload:
        raise ExperimentContractError("strict actor export is not canonical")
    return MappingProxyType(validated)


def load_full_authority_actor(
    path: Path,
    *,
    expected_sha256: str,
) -> LoadedFullAuthorityActor:
    payload = _read_export(Path(path), expected_sha256)
    validated = load_full_authority_actor_arrays(path, expected_sha256=expected_sha256)
    actor = FullAuthorityActor(
        {name: validated[name] for name in _ACTOR_PARAMETER_NAMES},
        source_actor_sha256=bytes(validated["source_actor_sha256"][0]).decode("ascii"),
    )
    if encode_full_authority_actor(actor) != payload:
        raise ExperimentContractError("strict actor export model reconstruction differs")
    return LoadedFullAuthorityActor(
        actor=actor,
        content_sha256=expected_sha256,
        byte_count=len(payload),
        member_hashes=MappingProxyType(
            {name: array_sha256(value) for name, value in validated.items()}
        ),
    )


def numpy_log_likelihood(
    pre_tanh: np.ndarray,
    mean: np.ndarray,
    log_std: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Independent float64 audit implementation of the squashed likelihood."""

    values = [
        _strict_float32(item, shape_tail=(ACTION_WIDTH,), field=field)
        for item, field in (
            (pre_tanh, "pre_tanh"),
            (mean, "mean"),
            (log_std, "log_std"),
        )
    ]
    if len({item.shape for item in values}) != 1:
        raise ExperimentContractError("likelihood audit shapes differ")
    x, mu, sigma_log = (item.astype(np.float64) for item in values)
    gaussian = -0.5 * (
        np.square((x - mu) / np.exp(sigma_log)) + 2.0 * sigma_log + math.log(2.0 * math.pi)
    )
    correction = np.log(1.0 - np.square(np.tanh(x)) + epsilon)
    return np.ascontiguousarray(np.sum(gaussian - correction, axis=-1), dtype="<f8")


__all__ = [
    "ACTION_ADAPTER_ID",
    "EXPORT_FORMAT_ID",
    "INPUT_LAYOUT_ID",
    "LOG_STD_MAX",
    "LOG_STD_MIN",
    "POLICY_ARCHITECTURE_ID",
    "POLICY_INPUT_WIDTH",
    "REFERENCE_HORIZON",
    "REFERENCE_WIDTH",
    "ActorDistribution",
    "FullAuthorityAction",
    "FullAuthorityActor",
    "FullAuthorityPolicy",
    "LoadedFullAuthorityActor",
    "StrictPolicyInput",
    "build_full_authority_policy",
    "compose_policy_input",
    "encode_full_authority_actor",
    "exact_physical_action",
    "export_full_authority_actor",
    "load_full_authority_actor",
    "load_full_authority_actor_arrays",
    "numpy_log_likelihood",
    "squashed_gaussian_log_likelihood",
    "verify_actor_warm_start",
    "verify_optimizer_authority",
]
