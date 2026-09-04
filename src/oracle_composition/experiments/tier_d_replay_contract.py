"""Pure contracts for a one-step Humanoid-v5 E2 replay probe.

Passing this contract establishes interface correctness for one recorded
transition. It does not admit a motion, controller, or corpus as Tier D and is
not behavioral evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from dataclasses import InitVar, asdict, dataclass
from numbers import Real
from typing import Any

import numpy as np

REPLAY_SCHEMA_VERSION = 1
REPLAY_PROBE_ID = "humanoid_v5_integration_plus_cfrc_one_step_replay/v2"
EVIDENCE_PURPOSE = "interface_correctness_probe"
CLAIM_BOUNDARY = "one_step_replay_only_no_behavior_or_tier_d_corpus_claim/v1"
STATE_SPEC = "mjtState.mjSTATE_INTEGRATION"
STATE_FLAG = 16_383
STATE_DTYPE = np.dtype("<f8")
STATE_SHAPE = (195,)
OBSERVATION_DTYPE = np.dtype("<f8")
OBSERVATION_SHAPE = (348,)
ACTION_DTYPE = np.dtype("<f4")
ACTION_SHAPE = (17,)
CFRCE_DTYPE = np.dtype("<f8")
CFRCE_SHAPE = (13, 6)
CANONICALIZATION_ID = "mj_forward_plus_exact_cfrc_ext_observation_sidecar/v2"
EXPECTED_WRAPPER_TYPES = (
    "gymnasium.wrappers.common.TimeLimit",
    "gymnasium.wrappers.common.OrderEnforcing",
    "gymnasium.wrappers.common.PassiveEnvChecker",
    "oracle_composition.envs.humanoid.SubstepContactHumanoidEnv",
)
EXPECTED_ENVIRONMENT_KWARGS = (
    ("exclude_current_positions_from_observation", True),
    ("frame_skip", 5),
    ("render_mode", None),
    ("reset_noise_scale", 0.01),
    ("terminate_when_unhealthy", False),
)
EXPECTED_ENVIRONMENT_SEMANTICS = (
    ("contact_cost_range_lower", "negative_infinity"),
    ("contact_cost_range_upper", 10.0),
    ("contact_cost_weight", 0.0000005),
    ("ctrl_cost_weight", 0.1),
    ("forward_reward_weight", 1.25),
    ("healthy_reward", 5.0),
    ("healthy_z_range_lower", 1.0),
    ("healthy_z_range_upper", 2.0),
    ("include_cfrc_ext_in_observation", True),
    ("include_cinert_in_observation", True),
    ("include_cvel_in_observation", True),
    ("include_qfrc_actuator_in_observation", True),
)

EXPECTED_STATE_COMPONENT_SIZES = (
    ("time", 1),
    ("qpos", 24),
    ("qvel", 23),
    ("act", 0),
    ("warmstart", 23),
    ("ctrl", 17),
    ("qfrc_applied", 23),
    ("xfrc_applied", 84),
    ("eq_active", 0),
    ("mocap_pos", 0),
    ("mocap_quat", 0),
    ("userdata", 0),
    ("plugin", 0),
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TierDReplayContractError(ValueError):
    """A replay probe violates the exact one-step interface contract."""


class TierDReplayMismatchError(RuntimeError):
    """A restored transition differs from the recorded transition."""

    def __init__(self, mismatches: tuple[str, ...]) -> None:
        self.mismatches = mismatches
        super().__init__("E2 replay mismatch: " + ", ".join(mismatches))


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
        raise TierDReplayContractError(f"value is not canonical JSON: {exc}") from exc


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _same_exact(observed: object, expected: object) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(expected, dict):
        return observed.keys() == expected.keys() and all(
            _same_exact(observed[key], value) for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        return len(observed) == len(expected) and all(
            _same_exact(left, right) for left, right in zip(observed, expected, strict=True)
        )
    return observed == expected


def _sha256(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise TierDReplayContractError(f"{field} must be a lowercase SHA-256")
    return value


def _bounded_text(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise TierDReplayContractError(f"{field} must be nonempty bounded text")
    return value


def _frozen_array(
    value: object,
    *,
    field: str,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype != dtype:
        raise TierDReplayContractError(
            f"{field} dtype is {array.dtype.str!r}; expected {dtype.str!r}"
        )
    if array.shape != shape:
        raise TierDReplayContractError(f"{field} shape is {array.shape!r}; expected {shape!r}")
    if not np.isfinite(array).all():
        raise TierDReplayContractError(f"{field} contains non-finite values")
    payload = np.ascontiguousarray(array, dtype=dtype).tobytes(order="C")
    return np.frombuffer(payload, dtype=dtype).reshape(shape)


def array_sha256(array: np.ndarray) -> str:
    """Hash array shape, dtype, and exact contiguous bytes."""

    contiguous = np.ascontiguousarray(array)
    metadata = _canonical_json({"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)})
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    payload = contiguous.tobytes(order="C")
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
    return digest.hexdigest()


def _arrays_exact(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _float64_exact(left: float, right: float) -> bool:
    return struct.pack(">d", float(left)) == struct.pack(">d", float(right))


@dataclass(frozen=True, slots=True)
class ReplayRuntimeIdentity:
    """Runtime facts that must match before state restoration is attempted."""

    environment_id: str
    environment_kwargs: tuple[tuple[str, bool | float | int | None], ...]
    environment_semantics: tuple[tuple[str, bool | float | str], ...]
    wrapper_types: tuple[str, ...]
    time_limit_steps: int
    python_version: str
    platform_system: str
    platform_release: str
    platform_machine: str
    numpy_version: str
    gymnasium_version: str
    mujoco_version: str
    model_sha256: str
    dependency_lock_sha256: str
    project_source_tree_sha256: str
    environment_source_sha256: str
    gym_humanoid_source_sha256: str
    gym_common_wrappers_source_sha256: str
    gym_passive_checker_source_sha256: str
    gym_mujoco_env_source_sha256: str
    mujoco_module_source_sha256: str
    mujoco_functions_binary_sha256: str
    mujoco_structs_binary_sha256: str
    mujoco_enums_binary_sha256: str
    adapter_source_sha256: str
    contract_source_sha256: str
    integration_state_spec: str
    integration_state_flag: int
    integration_state_size: int
    integration_state_dtype: str
    integration_state_components: tuple[tuple[str, int], ...]
    observation_shape: tuple[int, ...]
    observation_dtype: str
    observation_space_sha256: str
    observation_reconstruction_id: str
    cfrc_ext_shape: tuple[int, ...]
    cfrc_ext_dtype: str
    action_shape: tuple[int, ...]
    action_dtype: str
    action_space_sha256: str
    timestep_seconds: float
    frame_skip: int
    control_period_seconds: float
    integrator: str
    solver: str
    solver_iterations: int
    contact_capture_id: str

    def __post_init__(self) -> None:
        _bounded_text(self.environment_id, field="environment_id")
        if self.environment_id != "Humanoid-v5":
            raise TierDReplayContractError("environment_id must be 'Humanoid-v5'")
        if not _same_exact(self.environment_kwargs, EXPECTED_ENVIRONMENT_KWARGS):
            raise TierDReplayContractError("environment_kwargs differ from the frozen runtime")
        if not _same_exact(self.environment_semantics, EXPECTED_ENVIRONMENT_SEMANTICS):
            raise TierDReplayContractError("environment semantics differ from the frozen runtime")
        if not _same_exact(self.wrapper_types, EXPECTED_WRAPPER_TYPES):
            raise TierDReplayContractError("wrapper_types differ from the frozen runtime")
        if type(self.time_limit_steps) is not int or self.time_limit_steps != 1000:
            raise TierDReplayContractError("time_limit_steps must equal 1000")
        for field in (
            "python_version",
            "platform_system",
            "platform_release",
            "platform_machine",
            "numpy_version",
            "gymnasium_version",
            "mujoco_version",
        ):
            _bounded_text(getattr(self, field), field=field)
        if self.numpy_version != "2.5.2":
            raise TierDReplayContractError("numpy_version must equal '2.5.2'")
        if self.gymnasium_version != "1.3.0":
            raise TierDReplayContractError("gymnasium_version must equal '1.3.0'")
        if self.mujoco_version != "3.12.0":
            raise TierDReplayContractError("mujoco_version must equal '3.12.0'")
        for field in (
            "model_sha256",
            "dependency_lock_sha256",
            "project_source_tree_sha256",
            "environment_source_sha256",
            "gym_humanoid_source_sha256",
            "gym_common_wrappers_source_sha256",
            "gym_passive_checker_source_sha256",
            "gym_mujoco_env_source_sha256",
            "mujoco_module_source_sha256",
            "mujoco_functions_binary_sha256",
            "mujoco_structs_binary_sha256",
            "mujoco_enums_binary_sha256",
            "adapter_source_sha256",
            "contract_source_sha256",
            "observation_space_sha256",
            "action_space_sha256",
        ):
            _sha256(getattr(self, field), field=field)
        if self.integration_state_spec != STATE_SPEC:
            raise TierDReplayContractError(f"integration_state_spec must be {STATE_SPEC!r}")
        if (
            type(self.integration_state_flag) is not int
            or self.integration_state_flag != STATE_FLAG
        ):
            raise TierDReplayContractError(f"integration_state_flag must be {STATE_FLAG}")
        if (
            type(self.integration_state_size) is not int
            or self.integration_state_size != STATE_SHAPE[0]
        ):
            raise TierDReplayContractError(f"integration_state_size must be {STATE_SHAPE[0]}")
        if self.integration_state_dtype != STATE_DTYPE.str:
            raise TierDReplayContractError(f"integration_state_dtype must be {STATE_DTYPE.str!r}")
        if not _same_exact(self.integration_state_components, EXPECTED_STATE_COMPONENT_SIZES):
            raise TierDReplayContractError("integration-state component sizes differ")
        if sum(size for _name, size in self.integration_state_components) != (
            self.integration_state_size
        ):
            raise TierDReplayContractError("integration-state component sizes do not sum")
        if not _same_exact(self.observation_shape, OBSERVATION_SHAPE):
            raise TierDReplayContractError(f"observation_shape must be {OBSERVATION_SHAPE!r}")
        if self.observation_dtype != OBSERVATION_DTYPE.str:
            raise TierDReplayContractError(f"observation_dtype must be {OBSERVATION_DTYPE.str!r}")
        if self.observation_reconstruction_id != CANONICALIZATION_ID:
            raise TierDReplayContractError("observation reconstruction differs")
        if not _same_exact(self.cfrc_ext_shape, CFRCE_SHAPE):
            raise TierDReplayContractError(f"cfrc_ext_shape must be {CFRCE_SHAPE!r}")
        if self.cfrc_ext_dtype != CFRCE_DTYPE.str:
            raise TierDReplayContractError(f"cfrc_ext_dtype must be {CFRCE_DTYPE.str!r}")
        if not _same_exact(self.action_shape, ACTION_SHAPE):
            raise TierDReplayContractError(f"action_shape must be {ACTION_SHAPE!r}")
        if self.action_dtype != ACTION_DTYPE.str:
            raise TierDReplayContractError(f"action_dtype must be {ACTION_DTYPE.str!r}")
        if type(self.timestep_seconds) is not float or self.timestep_seconds != 0.003:
            raise TierDReplayContractError("Humanoid timestep differs")
        if type(self.frame_skip) is not int or self.frame_skip != 5:
            raise TierDReplayContractError("Humanoid timestep or frame_skip differs")
        if (
            type(self.control_period_seconds) is not float
            or self.control_period_seconds != self.timestep_seconds * self.frame_skip
        ):
            raise TierDReplayContractError("control period differs from timestep * frame_skip")
        if self.integrator != "mjINT_RK4" or self.solver != "mjSOL_PGS":
            raise TierDReplayContractError("MuJoCo integrator or solver differs")
        if type(self.solver_iterations) is not int or self.solver_iterations != 50:
            raise TierDReplayContractError("MuJoCo solver_iterations must be 50")
        if self.contact_capture_id != "all_mujoco_substeps_after_mj_step/v1":
            raise TierDReplayContractError("contact capture implementation differs")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return _sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class ReplayWrapperState:
    """State held outside MuJoCo but used by Gymnasium transition semantics."""

    time_limit_elapsed_steps: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.time_limit_elapsed_steps, int)
            or isinstance(self.time_limit_elapsed_steps, bool)
            or self.time_limit_elapsed_steps < 0
        ):
            raise TierDReplayContractError(
                "time_limit_elapsed_steps must be a non-negative integer"
            )


@dataclass(frozen=True, slots=True)
class ReplaySnapshot:
    """Canonical simulator and wrapper state at one control boundary."""

    integration_state: np.ndarray
    cfrc_ext: np.ndarray
    observation: np.ndarray
    wrapper_state: ReplayWrapperState
    canonicalization_id: str = CANONICALIZATION_ID

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "integration_state",
            _frozen_array(
                self.integration_state,
                field="integration_state",
                dtype=STATE_DTYPE,
                shape=STATE_SHAPE,
            ),
        )
        object.__setattr__(
            self,
            "cfrc_ext",
            _frozen_array(
                self.cfrc_ext,
                field="cfrc_ext",
                dtype=CFRCE_DTYPE,
                shape=CFRCE_SHAPE,
            ),
        )
        object.__setattr__(
            self,
            "observation",
            _frozen_array(
                self.observation,
                field="observation",
                dtype=OBSERVATION_DTYPE,
                shape=OBSERVATION_SHAPE,
            ),
        )
        observation_cfrc_ext = self.observation[-math.prod(CFRCE_SHAPE) :].reshape(CFRCE_SHAPE)
        if not _arrays_exact(self.cfrc_ext, observation_cfrc_ext):
            raise TierDReplayContractError(
                "cfrc_ext sidecar must equal the exact observation suffix"
            )
        if not isinstance(self.wrapper_state, ReplayWrapperState):
            raise TierDReplayContractError("wrapper_state has the wrong type")
        if self.canonicalization_id != CANONICALIZATION_ID:
            raise TierDReplayContractError(f"canonicalization_id must be {CANONICALIZATION_ID!r}")

    @property
    def simulation_time_seconds(self) -> float:
        return float(self.integration_state[0])


@dataclass(frozen=True, slots=True)
class ReplayContact:
    """One active contact sampled after one of five MuJoCo substeps."""

    physics_substep_index: int
    contact_index_within_substep: int
    geom1_id: int
    geom2_id: int
    normal_force_n: float

    def __post_init__(self) -> None:
        integer_fields = (
            "physics_substep_index",
            "contact_index_within_substep",
            "geom1_id",
            "geom2_id",
        )
        if any(
            not isinstance(getattr(self, field), int) or isinstance(getattr(self, field), bool)
            for field in integer_fields
        ):
            raise TierDReplayContractError("contact index and geometry fields must be integers")
        if not 0 <= self.physics_substep_index < 5:
            raise TierDReplayContractError("contact substep index must be in [0, 5)")
        if self.contact_index_within_substep < 0 or self.geom1_id < 0 or self.geom2_id < 0:
            raise TierDReplayContractError("contact and geometry indices must be non-negative")
        if isinstance(self.normal_force_n, bool) or not isinstance(self.normal_force_n, Real):
            raise TierDReplayContractError("contact normal force must be numeric")
        force = float(self.normal_force_n)
        if not math.isfinite(force) or force < 0.0:
            raise TierDReplayContractError("contact normal force must be finite and non-negative")
        object.__setattr__(self, "normal_force_n", force)

    def exact_bytes(self) -> bytes:
        return struct.pack(
            ">qqqqd",
            self.physics_substep_index,
            self.contact_index_within_substep,
            self.geom1_id,
            self.geom2_id,
            self.normal_force_n,
        )


@dataclass(frozen=True, slots=True)
class ReplayTransition:
    """Recorded action and exact next-boundary outputs."""

    action: np.ndarray
    next_snapshot: ReplaySnapshot
    returned_observation: np.ndarray
    contacts: tuple[ReplayContact, ...]
    reward: float
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action",
            _frozen_array(
                self.action,
                field="action",
                dtype=ACTION_DTYPE,
                shape=ACTION_SHAPE,
            ),
        )
        if not isinstance(self.next_snapshot, ReplaySnapshot):
            raise TierDReplayContractError("next_snapshot has the wrong type")
        object.__setattr__(
            self,
            "returned_observation",
            _frozen_array(
                self.returned_observation,
                field="returned_observation",
                dtype=OBSERVATION_DTYPE,
                shape=OBSERVATION_SHAPE,
            ),
        )
        contacts = tuple(self.contacts)
        if any(not isinstance(contact, ReplayContact) for contact in contacts):
            raise TierDReplayContractError("contacts contain an invalid record")
        object.__setattr__(self, "contacts", contacts)
        if not isinstance(self.reward, (int, float)) or isinstance(self.reward, bool):
            raise TierDReplayContractError("reward must be numeric")
        if not math.isfinite(float(self.reward)):
            raise TierDReplayContractError("reward must be finite")
        object.__setattr__(self, "reward", float(self.reward))
        if not isinstance(self.terminated, bool) or not isinstance(self.truncated, bool):
            raise TierDReplayContractError("termination and truncation flags must be boolean")


@dataclass(frozen=True, slots=True)
class ReplayProbe:
    """One source transition whose exact replay may be checked elsewhere."""

    runtime: ReplayRuntimeIdentity
    anchor: ReplaySnapshot
    transition: ReplayTransition

    def __post_init__(self) -> None:
        if not isinstance(self.runtime, ReplayRuntimeIdentity):
            raise TierDReplayContractError("runtime has the wrong type")
        if not isinstance(self.anchor, ReplaySnapshot):
            raise TierDReplayContractError("anchor has the wrong type")
        if not isinstance(self.transition, ReplayTransition):
            raise TierDReplayContractError("transition has the wrong type")
        expected_elapsed = self.anchor.wrapper_state.time_limit_elapsed_steps + 1
        observed_elapsed = self.transition.next_snapshot.wrapper_state.time_limit_elapsed_steps
        if observed_elapsed != expected_elapsed:
            raise TierDReplayContractError("next TimeLimit counter must equal anchor counter + 1")
        if observed_elapsed > self.runtime.time_limit_steps:
            raise TierDReplayContractError("next TimeLimit counter exceeds the episode limit")

    @property
    def schema_version(self) -> int:
        return REPLAY_SCHEMA_VERSION

    @property
    def probe_id(self) -> str:
        return REPLAY_PROBE_ID

    @property
    def evidence_purpose(self) -> str:
        return EVIDENCE_PURPOSE

    @property
    def claim_boundary(self) -> str:
        return CLAIM_BOUNDARY


@dataclass(frozen=True, slots=True)
class ReplayComparison:
    """Exact comparison fields; any false field blocks a receipt."""

    runtime: bool
    restored_anchor_state: bool
    restored_anchor_cfrc_ext: bool
    restored_anchor_observation: bool
    restored_wrapper_counter: bool
    next_state: bool
    next_cfrc_ext: bool
    next_observation: bool
    next_canonical_observation: bool
    contacts: bool
    reward: bool
    terminated: bool
    truncated: bool
    next_wrapper_counter: bool

    def __post_init__(self) -> None:
        if any(type(getattr(self, field)) is not bool for field in self.__dataclass_fields__):
            raise TierDReplayContractError("replay comparisons must be exact booleans")

    @property
    def mismatches(self) -> tuple[str, ...]:
        return tuple(field for field in self.__dataclass_fields__ if not bool(getattr(self, field)))

    @property
    def passed(self) -> bool:
        return not self.mismatches


def _contacts_exact(
    expected: tuple[ReplayContact, ...], observed: tuple[ReplayContact, ...]
) -> bool:
    return len(expected) == len(observed) and all(
        left.exact_bytes() == right.exact_bytes()
        for left, right in zip(expected, observed, strict=True)
    )


def compare_replay(
    expected: ReplayProbe,
    *,
    observed_runtime: ReplayRuntimeIdentity,
    restored_anchor: ReplaySnapshot,
    observed_transition: ReplayTransition,
) -> ReplayComparison:
    """Compare every state and Gymnasium output using exact bytes."""

    return ReplayComparison(
        runtime=expected.runtime == observed_runtime,
        restored_anchor_state=_arrays_exact(
            expected.anchor.integration_state,
            restored_anchor.integration_state,
        ),
        restored_anchor_cfrc_ext=_arrays_exact(
            expected.anchor.cfrc_ext,
            restored_anchor.cfrc_ext,
        ),
        restored_anchor_observation=_arrays_exact(
            expected.anchor.observation,
            restored_anchor.observation,
        ),
        restored_wrapper_counter=(expected.anchor.wrapper_state == restored_anchor.wrapper_state),
        next_state=_arrays_exact(
            expected.transition.next_snapshot.integration_state,
            observed_transition.next_snapshot.integration_state,
        ),
        next_cfrc_ext=_arrays_exact(
            expected.transition.next_snapshot.cfrc_ext,
            observed_transition.next_snapshot.cfrc_ext,
        ),
        next_observation=_arrays_exact(
            expected.transition.returned_observation,
            observed_transition.returned_observation,
        ),
        next_canonical_observation=_arrays_exact(
            expected.transition.next_snapshot.observation,
            observed_transition.next_snapshot.observation,
        ),
        contacts=_contacts_exact(
            expected.transition.contacts,
            observed_transition.contacts,
        ),
        reward=_float64_exact(expected.transition.reward, observed_transition.reward),
        terminated=(expected.transition.terminated is observed_transition.terminated),
        truncated=(expected.transition.truncated is observed_transition.truncated),
        next_wrapper_counter=(
            expected.transition.next_snapshot.wrapper_state
            == observed_transition.next_snapshot.wrapper_state
        ),
    )


def contacts_sha256(contacts: tuple[ReplayContact, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(len(contacts).to_bytes(8, "big"))
    for contact in contacts:
        payload = contact.exact_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


_RECEIPT_ISSUE_TOKEN = object()


@dataclass(frozen=True, slots=True)
class ReplayVerificationReceipt:
    """Passing interface receipt; construction is unavailable for mismatches."""

    runtime_sha256: str
    observed_runtime_sha256: str
    anchor_state_sha256: str
    restored_anchor_state_sha256: str
    anchor_cfrc_ext_sha256: str
    restored_anchor_cfrc_ext_sha256: str
    anchor_observation_sha256: str
    restored_anchor_observation_sha256: str
    action_sha256: str
    next_state_sha256: str
    observed_next_state_sha256: str
    next_cfrc_ext_sha256: str
    observed_next_cfrc_ext_sha256: str
    next_observation_sha256: str
    observed_next_observation_sha256: str
    next_canonical_observation_sha256: str
    observed_next_canonical_observation_sha256: str
    contacts_sha256: str
    observed_contacts_sha256: str
    reward_ieee754_hex: str
    observed_reward_ieee754_hex: str
    expected_terminated: bool
    observed_terminated: bool
    expected_truncated: bool
    observed_truncated: bool
    anchor_time_limit_elapsed_steps: int
    restored_anchor_time_limit_elapsed_steps: int
    next_time_limit_elapsed_steps: int
    observed_next_time_limit_elapsed_steps: int
    comparisons: ReplayComparison
    _issue_token: InitVar[object] = None

    def __post_init__(self, _issue_token: object) -> None:
        if _issue_token is not _RECEIPT_ISSUE_TOKEN:
            raise TierDReplayContractError(
                "replay receipts may be issued only from observed replay values"
            )
        if not isinstance(self.comparisons, ReplayComparison):
            raise TierDReplayContractError("replay receipt comparisons have the wrong type")
        if not self.comparisons.passed:
            raise TierDReplayContractError("a replay receipt requires every exact comparison")
        for field in (
            "runtime_sha256",
            "observed_runtime_sha256",
            "anchor_state_sha256",
            "restored_anchor_state_sha256",
            "anchor_cfrc_ext_sha256",
            "restored_anchor_cfrc_ext_sha256",
            "anchor_observation_sha256",
            "restored_anchor_observation_sha256",
            "action_sha256",
            "next_state_sha256",
            "observed_next_state_sha256",
            "next_cfrc_ext_sha256",
            "observed_next_cfrc_ext_sha256",
            "next_observation_sha256",
            "observed_next_observation_sha256",
            "next_canonical_observation_sha256",
            "observed_next_canonical_observation_sha256",
            "contacts_sha256",
            "observed_contacts_sha256",
        ):
            _sha256(getattr(self, field), field=field)
        for field in ("reward_ieee754_hex", "observed_reward_ieee754_hex"):
            if not re.fullmatch(r"[0-9a-f]{16}", getattr(self, field)):
                raise TierDReplayContractError(f"{field} must encode one float64")
        matching_pairs = (
            ("runtime_sha256", "observed_runtime_sha256"),
            ("anchor_state_sha256", "restored_anchor_state_sha256"),
            ("anchor_cfrc_ext_sha256", "restored_anchor_cfrc_ext_sha256"),
            ("anchor_observation_sha256", "restored_anchor_observation_sha256"),
            ("next_state_sha256", "observed_next_state_sha256"),
            ("next_cfrc_ext_sha256", "observed_next_cfrc_ext_sha256"),
            ("next_observation_sha256", "observed_next_observation_sha256"),
            (
                "next_canonical_observation_sha256",
                "observed_next_canonical_observation_sha256",
            ),
            ("contacts_sha256", "observed_contacts_sha256"),
            ("reward_ieee754_hex", "observed_reward_ieee754_hex"),
            ("expected_terminated", "observed_terminated"),
            ("expected_truncated", "observed_truncated"),
            (
                "anchor_time_limit_elapsed_steps",
                "restored_anchor_time_limit_elapsed_steps",
            ),
            ("next_time_limit_elapsed_steps", "observed_next_time_limit_elapsed_steps"),
        )
        if any(
            not _same_exact(getattr(self, source), getattr(self, observed))
            for source, observed in matching_pairs
        ):
            raise TierDReplayContractError("receipt source and replay values differ")
        for field in (
            "expected_terminated",
            "observed_terminated",
            "expected_truncated",
            "observed_truncated",
        ):
            if type(getattr(self, field)) is not bool:
                raise TierDReplayContractError(f"{field} must be boolean")
        for field in (
            "anchor_time_limit_elapsed_steps",
            "restored_anchor_time_limit_elapsed_steps",
            "next_time_limit_elapsed_steps",
            "observed_next_time_limit_elapsed_steps",
        ):
            if type(getattr(self, field)) is not int or getattr(self, field) < 0:
                raise TierDReplayContractError(f"{field} must be a non-negative integer")

    @property
    def schema_version(self) -> int:
        return REPLAY_SCHEMA_VERSION

    @property
    def probe_id(self) -> str:
        return REPLAY_PROBE_ID

    @property
    def evidence_purpose(self) -> str:
        return EVIDENCE_PURPOSE

    @property
    def claim_boundary(self) -> str:
        return CLAIM_BOUNDARY

    @property
    def replay_passed(self) -> bool:
        return True

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "probe_id": self.probe_id,
            "evidence_purpose": self.evidence_purpose,
            "claim_boundary": self.claim_boundary,
            "replay_passed": self.replay_passed,
            **asdict(self),
        }
