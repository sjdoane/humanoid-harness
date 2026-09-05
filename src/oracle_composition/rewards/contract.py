"""Pure, fail-closed schema for target-speed reward programs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

CONTROL_PERIOD_SECONDS = 0.015
TARGET_SPEEDS_M_S = (0.5, 1.0, 1.5)
CANDIDATE_READ_SET_V1 = ("com_x_velocity_m_s", "target_speed_m_s")
SIGNED_TERM_NAMES_V1 = ("task_progress", "healthy", "control", "contact")
ACTUATOR_QVEL_INDICES_BY_ACTION_V1 = (
    7,
    6,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
)
GENERALIZED_ACTUATOR_TORQUE_CAPACITY_N_M = (
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    120.0,
    80.0,
    40.0,
    40.0,
    120.0,
    80.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
)
TASK_TERM_ABS_MAX = 1000.0
TOTAL_REWARD_ABS_MAX = 1024.0
TOTAL_REPRODUCTION_ABS_TOLERANCE = 1e-12
STOCK_FLOAT32_CONTROL_ABS_MAX = float(np.float32(0.4))


class RewardContractError(ValueError):
    """Raised when reward data or identity cannot satisfy the frozen contract."""


def _deep_freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze_json(child) for key, child in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_deep_freeze_json(child) for child in value)
    return value


REWARD_SCHEMA_V1: Mapping[str, object] = _deep_freeze_json(
    {
        "schema_version": 1,
        "schema_id": "family-b-target-speed/reward-contract/v1",
        "control_period_s": CONTROL_PERIOD_SECONDS,
        "candidate_read_set": list(CANDIDATE_READ_SET_V1),
        "trusted_step": {
            "qpos_after_f64": {
                "shape": [24],
                "dtype": "float64",
                "units": "m, unit quaternion, rad",
                "bounds": [None, None],
            },
            "qvel_after_f64": {
                "shape": [23],
                "dtype": "float64",
                "units": "m/s, rad/s",
                "bounds": [None, None],
            },
            "com_x_velocity_m_s": {
                "shape": [],
                "dtype": "float64",
                "units": "m/s",
                "bounds": [None, None],
            },
            "ctrl_f64": {
                "shape": [17],
                "dtype": "float64",
                "units": "MuJoCo control coordinate",
                "bounds": [-0.4, 0.4],
            },
            "generalized_actuator_torque_n_m_f64": {
                "shape": [17],
                "dtype": "float64",
                "units": "N*m",
                "order": "actuator action order",
                "qvel_indices": list(ACTUATOR_QVEL_INDICES_BY_ACTION_V1),
                "bounds": [
                    [-value for value in GENERALIZED_ACTUATOR_TORQUE_CAPACITY_N_M],
                    list(GENERALIZED_ACTUATOR_TORQUE_CAPACITY_N_M),
                ],
                "absolute_capacity_n_m": list(GENERALIZED_ACTUATOR_TORQUE_CAPACITY_N_M),
            },
            "external_contact_wrench_f64": {
                "shape": [14, 6],
                "dtype": "float64",
                "units": ["N", "N", "N", "N*m", "N*m", "N*m"],
                "bounds": [None, None],
                "includes_world_row": True,
            },
            "target_speed_m_s": {
                "shape": [],
                "dtype": "float64",
                "units": "m/s",
                "allowed": list(TARGET_SPEEDS_M_S),
            },
            "control_period_s": {
                "shape": [],
                "dtype": "float64",
                "units": "s",
                "value": CONTROL_PERIOD_SECONDS,
            },
        },
        "candidate_inputs": {
            "com_x_velocity_m_s": {
                "shape": [],
                "dtype": "float64",
                "units": "m/s",
                "bounds": [None, None],
            },
            "target_speed_m_s": {
                "shape": [],
                "dtype": "float64",
                "units": "m/s",
                "allowed": list(TARGET_SPEEDS_M_S),
            },
        },
        "result": {
            "total": {
                "shape": [],
                "dtype": "float64",
                "units": "reward unit/control step",
                "bounds": [-TOTAL_REWARD_ABS_MAX, TOTAL_REWARD_ABS_MAX],
            },
            "signed_terms": list(SIGNED_TERM_NAMES_V1),
            "task_term_abs_max": TASK_TERM_ABS_MAX,
            "total_abs_max": TOTAL_REWARD_ABS_MAX,
            "total_reproduction_abs_tolerance": TOTAL_REPRODUCTION_ABS_TOLERANCE,
        },
    }
)


def _json_value(value: object) -> object:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_value(value.to_dict())
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, child in value.items():
            if type(key) is not str or key in result:
                raise RewardContractError("canonical JSON object keys must be unique strings")
            result[key] = _json_value(child)
        return result
    if isinstance(value, tuple | list):
        return [_json_value(child) for child in value]
    return value


def canonical_json_bytes(value: object) -> bytes:
    """Encode finite JSON with one canonical byte representation."""

    try:
        return json.dumps(
            _json_value(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RewardContractError(f"value is not finite canonical JSON: {exc}") from exc


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RewardContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def decode_canonical_json_object(
    encoded: bytes, *, maximum_bytes: int = 64 * 1024
) -> dict[str, Any]:
    if type(encoded) is not bytes or not encoded or len(encoded) > maximum_bytes:
        raise RewardContractError("canonical JSON bytes are absent or too large")

    def reject_constant(value: str) -> None:
        raise RewardContractError(f"non-finite JSON constant {value!r}")

    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except RewardContractError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise RewardContractError("canonical JSON is invalid") from exc
    if type(value) is not dict or canonical_json_bytes(value) != encoded:
        raise RewardContractError("JSON does not use the canonical encoding")
    return value


def reward_schema_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(REWARD_SCHEMA_V1)).hexdigest()


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise RewardContractError(f"{field} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise RewardContractError(f"{field} must be finite")
    return result


def _exact_f64_array(value: object, *, shape: tuple[int, ...], field: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise RewardContractError(f"{field} must be a numpy.ndarray")
    if value.shape != shape:
        raise RewardContractError(f"{field} must have shape {shape}")
    if value.dtype != np.dtype(np.float64):
        raise RewardContractError(f"{field} must have dtype float64")
    if not np.isfinite(value).all():
        raise RewardContractError(f"{field} must contain only finite values")
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    result.flags.writeable = False
    return result


def _wire_f64_array(value: object, *, shape: tuple[int, ...], field: str) -> np.ndarray:
    if isinstance(value, np.ndarray):
        if value.shape != shape:
            raise RewardContractError(f"{field} must have shape {shape}")
        if value.dtype != np.dtype(np.float64):
            raise RewardContractError(f"{field} direct array must have dtype float64")
        return value

    def validate_json_array(node: object, remaining_shape: tuple[int, ...]) -> None:
        if not remaining_shape:
            if type(node) not in (int, float):
                raise RewardContractError(
                    f"{field} JSON array elements must be numbers, not booleans or strings"
                )
            try:
                finite = math.isfinite(float(node))
            except (OverflowError, ValueError) as exc:
                raise RewardContractError(f"{field} JSON array element is not finite") from exc
            if not finite:
                raise RewardContractError(f"{field} JSON array element is not finite")
            return
        if type(node) is not list or len(node) != remaining_shape[0]:
            raise RewardContractError(f"{field} JSON array must have shape {shape}")
        for child in node:
            validate_json_array(child, remaining_shape[1:])

    validate_json_array(value, shape)
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RewardContractError(f"{field} JSON array cannot be represented as float64") from exc


def _target_speed(value: object) -> float:
    result = _finite_float(value, field="target_speed_m_s")
    if result not in TARGET_SPEEDS_M_S:
        raise RewardContractError(f"target_speed_m_s must be one of {TARGET_SPEEDS_M_S!r}")
    return result


@dataclass(frozen=True, slots=True)
class CandidateTaskInputsV1:
    """The complete and exclusive view available to authored code."""

    com_x_velocity_m_s: float
    target_speed_m_s: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "com_x_velocity_m_s",
            _finite_float(self.com_x_velocity_m_s, field="com_x_velocity_m_s"),
        )
        object.__setattr__(self, "target_speed_m_s", _target_speed(self.target_speed_m_s))

    def to_dict(self) -> dict[str, float]:
        return {
            "com_x_velocity_m_s": self.com_x_velocity_m_s,
            "target_speed_m_s": self.target_speed_m_s,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> CandidateTaskInputsV1:
        if type(value) is not dict or set(value) != set(CANDIDATE_READ_SET_V1):
            raise RewardContractError("candidate input keys differ from the v1 read set")
        return cls(
            com_x_velocity_m_s=value["com_x_velocity_m_s"],  # type: ignore[arg-type]
            target_speed_m_s=value["target_speed_m_s"],  # type: ignore[arg-type]
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> CandidateTaskInputsV1:
        return cls.from_dict(decode_canonical_json_object(encoded))


@runtime_checkable
class TaskTermV1(Protocol):
    def task_term(self, x: CandidateTaskInputsV1) -> float:
        """Return the sole authorable scalar task term."""


@dataclass(frozen=True, slots=True)
class TrustedRewardStepV1:
    """Complete trusted post-step state used by the reward compositor."""

    qpos_after_f64: np.ndarray
    qvel_after_f64: np.ndarray
    com_x_velocity_m_s: float
    ctrl_f64: np.ndarray
    generalized_actuator_torque_n_m_f64: np.ndarray
    external_contact_wrench_f64: np.ndarray
    target_speed_m_s: float
    control_period_s: float = CONTROL_PERIOD_SECONDS

    def __post_init__(self) -> None:
        array_fields = (
            ("qpos_after_f64", (24,)),
            ("qvel_after_f64", (23,)),
            ("ctrl_f64", (17,)),
            ("generalized_actuator_torque_n_m_f64", (17,)),
            ("external_contact_wrench_f64", (14, 6)),
        )
        for field, shape in array_fields:
            object.__setattr__(
                self,
                field,
                _exact_f64_array(getattr(self, field), shape=shape, field=field),
            )
        object.__setattr__(
            self,
            "com_x_velocity_m_s",
            _finite_float(self.com_x_velocity_m_s, field="com_x_velocity_m_s"),
        )
        object.__setattr__(self, "target_speed_m_s", _target_speed(self.target_speed_m_s))
        control_period = _finite_float(self.control_period_s, field="control_period_s")
        if control_period != CONTROL_PERIOD_SECONDS:
            raise RewardContractError(f"control_period_s must equal {CONTROL_PERIOD_SECONDS!r}")
        object.__setattr__(self, "control_period_s", control_period)
        if np.any(self.ctrl_f64 < -STOCK_FLOAT32_CONTROL_ABS_MAX) or np.any(
            self.ctrl_f64 > STOCK_FLOAT32_CONTROL_ABS_MAX
        ):
            raise RewardContractError("ctrl_f64 must stay in the stock [-0.4, 0.4] bounds")
        if not math.isclose(
            float(np.linalg.norm(self.qpos_after_f64[3:7])),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RewardContractError("qpos_after_f64 root orientation must be a unit quaternion")
        capacities = np.asarray(GENERALIZED_ACTUATOR_TORQUE_CAPACITY_N_M, dtype=np.float64)
        if np.any(np.abs(self.generalized_actuator_torque_n_m_f64) > capacities * (1.0 + 1e-9)):
            raise RewardContractError(
                "generalized actuator torque exceeds the stock structural capacity"
            )
        if not np.allclose(
            self.generalized_actuator_torque_n_m_f64 / capacities,
            self.ctrl_f64 / 0.4,
            rtol=0.0,
            atol=1e-7,
        ):
            raise RewardContractError("generalized actuator torque and post-step control disagree")

    @property
    def candidate_inputs(self) -> CandidateTaskInputsV1:
        return CandidateTaskInputsV1(
            com_x_velocity_m_s=self.com_x_velocity_m_s,
            target_speed_m_s=self.target_speed_m_s,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "qpos_after_f64": self.qpos_after_f64.tolist(),
            "qvel_after_f64": self.qvel_after_f64.tolist(),
            "com_x_velocity_m_s": self.com_x_velocity_m_s,
            "ctrl_f64": self.ctrl_f64.tolist(),
            "generalized_actuator_torque_n_m_f64": (
                self.generalized_actuator_torque_n_m_f64.tolist()
            ),
            "external_contact_wrench_f64": self.external_contact_wrench_f64.tolist(),
            "target_speed_m_s": self.target_speed_m_s,
            "control_period_s": self.control_period_s,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TrustedRewardStepV1:
        expected = {
            "qpos_after_f64",
            "qvel_after_f64",
            "com_x_velocity_m_s",
            "ctrl_f64",
            "generalized_actuator_torque_n_m_f64",
            "external_contact_wrench_f64",
            "target_speed_m_s",
            "control_period_s",
        }
        if type(value) is not dict or set(value) != expected:
            raise RewardContractError("trusted reward step keys differ")
        return cls(
            qpos_after_f64=_wire_f64_array(
                value["qpos_after_f64"], shape=(24,), field="qpos_after_f64"
            ),
            qvel_after_f64=_wire_f64_array(
                value["qvel_after_f64"], shape=(23,), field="qvel_after_f64"
            ),
            com_x_velocity_m_s=value["com_x_velocity_m_s"],  # type: ignore[arg-type]
            ctrl_f64=_wire_f64_array(value["ctrl_f64"], shape=(17,), field="ctrl_f64"),
            generalized_actuator_torque_n_m_f64=_wire_f64_array(
                value["generalized_actuator_torque_n_m_f64"],
                shape=(17,),
                field="generalized_actuator_torque_n_m_f64",
            ),
            external_contact_wrench_f64=_wire_f64_array(
                value["external_contact_wrench_f64"],
                shape=(14, 6),
                field="external_contact_wrench_f64",
            ),
            target_speed_m_s=value["target_speed_m_s"],  # type: ignore[arg-type]
            control_period_s=value["control_period_s"],  # type: ignore[arg-type]
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> TrustedRewardStepV1:
        return cls.from_dict(decode_canonical_json_object(encoded))


@dataclass(frozen=True, slots=True)
class RewardResultV1:
    total: float
    signed_terms: Mapping[str, float]

    def __post_init__(self) -> None:
        total = _finite_float(self.total, field="total")
        if abs(total) > TOTAL_REWARD_ABS_MAX:
            raise RewardContractError("total reward exceeds the fail-closed envelope")
        if type(self.signed_terms) is not dict and not isinstance(self.signed_terms, Mapping):
            raise RewardContractError("signed_terms must be a mapping")
        if set(self.signed_terms) != set(SIGNED_TERM_NAMES_V1):
            raise RewardContractError("signed_terms must contain exactly four frozen names")
        terms = {
            name: _finite_float(self.signed_terms[name], field=f"signed_terms.{name}")
            for name in SIGNED_TERM_NAMES_V1
        }
        if abs(terms["task_progress"]) > TASK_TERM_ABS_MAX:
            raise RewardContractError("task term exceeds the fail-closed envelope")
        if terms["healthy"] not in (0.0, 5.0):
            raise RewardContractError("healthy signed term must be exactly 0 or 5")
        if terms["control"] > 0.0 or terms["contact"] > 0.0:
            raise RewardContractError("control and contact signed terms must be non-positive")
        reproduced = terms["task_progress"] + terms["healthy"] + terms["control"] + terms["contact"]
        if not math.isclose(
            reproduced,
            total,
            rel_tol=0.0,
            abs_tol=TOTAL_REPRODUCTION_ABS_TOLERANCE,
        ):
            raise RewardContractError("signed terms do not reproduce total within 1e-12")
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "signed_terms", MappingProxyType(terms))

    def to_dict(self) -> dict[str, object]:
        return {"total": self.total, "signed_terms": dict(self.signed_terms)}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RewardResultV1:
        if type(value) is not dict or set(value) != {"total", "signed_terms"}:
            raise RewardContractError("reward result keys differ")
        terms = value["signed_terms"]
        if not isinstance(terms, Mapping):
            raise RewardContractError("signed_terms must be a mapping")
        return cls(total=value["total"], signed_terms=terms)  # type: ignore[arg-type]

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> RewardResultV1:
        return cls.from_dict(decode_canonical_json_object(encoded))


def _require_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RewardContractError(f"{field} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class RewardArtifactIdentityV1:
    """Content identity for authored bytes and every trusted reward dependency."""

    candidate_source_sha256: str
    schema_sha256: str
    read_set: tuple[str, ...]
    target_speed_m_s: float
    affine_alpha: float
    affine_beta: float
    compositor_source_sha256: str
    gymnasium_source_sha256: str
    humanoid_xml_sha256: str
    dependency_hashes: Mapping[str, str]
    artifact_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "candidate_source_sha256",
            "schema_sha256",
            "compositor_source_sha256",
            "gymnasium_source_sha256",
            "humanoid_xml_sha256",
            "artifact_sha256",
        ):
            object.__setattr__(self, field, _require_sha256(getattr(self, field), field=field))
        if self.schema_sha256 != reward_schema_sha256():
            raise RewardContractError("reward artifact schema SHA-256 differs from v1")
        if tuple(self.read_set) != CANDIDATE_READ_SET_V1:
            raise RewardContractError("reward artifact read_set differs from v1")
        object.__setattr__(self, "read_set", CANDIDATE_READ_SET_V1)
        object.__setattr__(self, "target_speed_m_s", _target_speed(self.target_speed_m_s))
        for field in ("affine_alpha", "affine_beta"):
            object.__setattr__(self, field, _finite_float(getattr(self, field), field=field))
        if not 0.25 <= self.affine_alpha <= 4.0 or abs(self.affine_beta) > 10.0:
            raise RewardContractError("reward artifact affine parameters exceed their bounds")
        if type(self.dependency_hashes) is not dict and not isinstance(
            self.dependency_hashes, Mapping
        ):
            raise RewardContractError("dependency_hashes must be a mapping")
        if not self.dependency_hashes:
            raise RewardContractError("dependency_hashes cannot be empty")
        dependencies: dict[str, str] = {}
        for name, digest in self.dependency_hashes.items():
            if (
                type(name) is not str
                or not name
                or len(name) > 255
                or name.startswith("_")
                or "__" in name
                or "\0" in name
            ):
                raise RewardContractError("dependency hash names must be public strings")
            dependencies[name] = _require_sha256(digest, field=f"dependency_hashes.{name}")
        object.__setattr__(
            self, "dependency_hashes", MappingProxyType(dict(sorted(dependencies.items())))
        )
        expected = hashlib.sha256(canonical_json_bytes(self._identity_payload())).hexdigest()
        if self.artifact_sha256 != expected:
            raise RewardContractError("reward artifact SHA-256 does not match its identity fields")

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "candidate_source_sha256": self.candidate_source_sha256,
            "schema_sha256": self.schema_sha256,
            "read_set": list(self.read_set),
            "target_speed_m_s": self.target_speed_m_s,
            "affine_alpha": self.affine_alpha,
            "affine_beta": self.affine_beta,
            "compositor_source_sha256": self.compositor_source_sha256,
            "gymnasium_source_sha256": self.gymnasium_source_sha256,
            "humanoid_xml_sha256": self.humanoid_xml_sha256,
            "dependency_hashes": dict(self.dependency_hashes),
        }

    @classmethod
    def create(
        cls,
        *,
        candidate_source_bytes: bytes,
        target_speed_m_s: float,
        affine_alpha: float,
        affine_beta: float,
        compositor_source_sha256: str,
        gymnasium_source_sha256: str,
        humanoid_xml_sha256: str,
        dependency_hashes: Mapping[str, str],
    ) -> RewardArtifactIdentityV1:
        if (
            type(candidate_source_bytes) is not bytes
            or not candidate_source_bytes
            or len(candidate_source_bytes) > 16 * 1024
        ):
            raise RewardContractError(
                "candidate_source_bytes must be nonempty exact bytes at most 16 KiB"
            )
        try:
            candidate_source_bytes.decode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise RewardContractError("candidate_source_bytes must be strict UTF-8") from exc
        fields = {
            "candidate_source_sha256": hashlib.sha256(candidate_source_bytes).hexdigest(),
            "schema_sha256": reward_schema_sha256(),
            "read_set": CANDIDATE_READ_SET_V1,
            "target_speed_m_s": _target_speed(target_speed_m_s),
            "affine_alpha": _finite_float(affine_alpha, field="affine_alpha"),
            "affine_beta": _finite_float(affine_beta, field="affine_beta"),
            "compositor_source_sha256": _require_sha256(
                compositor_source_sha256, field="compositor_source_sha256"
            ),
            "gymnasium_source_sha256": _require_sha256(
                gymnasium_source_sha256, field="gymnasium_source_sha256"
            ),
            "humanoid_xml_sha256": _require_sha256(
                humanoid_xml_sha256, field="humanoid_xml_sha256"
            ),
            "dependency_hashes": dict(dependency_hashes),
        }
        payload = {
            "schema_version": 1,
            **fields,
            "read_set": list(CANDIDATE_READ_SET_V1),
        }
        return cls(
            **fields,
            artifact_sha256=hashlib.sha256(canonical_json_bytes(payload)).hexdigest(),
        )

    def to_dict(self) -> dict[str, object]:
        return {**self._identity_payload(), "artifact_sha256": self.artifact_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RewardArtifactIdentityV1:
        expected = {
            "schema_version",
            "candidate_source_sha256",
            "schema_sha256",
            "read_set",
            "target_speed_m_s",
            "affine_alpha",
            "affine_beta",
            "compositor_source_sha256",
            "gymnasium_source_sha256",
            "humanoid_xml_sha256",
            "dependency_hashes",
            "artifact_sha256",
        }
        if type(value) is not dict or set(value) != expected or value["schema_version"] != 1:
            raise RewardContractError("reward artifact identity keys or version differ")
        dependencies = value["dependency_hashes"]
        if not isinstance(dependencies, Mapping):
            raise RewardContractError("dependency_hashes must be an object")
        read_set = value["read_set"]
        if not isinstance(read_set, list) or any(type(item) is not str for item in read_set):
            raise RewardContractError("read_set must be a string list")
        return cls(
            candidate_source_sha256=value["candidate_source_sha256"],  # type: ignore[arg-type]
            schema_sha256=value["schema_sha256"],  # type: ignore[arg-type]
            read_set=tuple(read_set),
            target_speed_m_s=value["target_speed_m_s"],  # type: ignore[arg-type]
            affine_alpha=value["affine_alpha"],  # type: ignore[arg-type]
            affine_beta=value["affine_beta"],  # type: ignore[arg-type]
            compositor_source_sha256=value["compositor_source_sha256"],  # type: ignore[arg-type]
            gymnasium_source_sha256=value["gymnasium_source_sha256"],  # type: ignore[arg-type]
            humanoid_xml_sha256=value["humanoid_xml_sha256"],  # type: ignore[arg-type]
            dependency_hashes=dict(dependencies),  # type: ignore[arg-type]
            artifact_sha256=value["artifact_sha256"],  # type: ignore[arg-type]
        )

    @property
    def canonical_bytes(self) -> bytes:
        return canonical_json_bytes(self.to_dict())

    @classmethod
    def from_canonical_bytes(cls, encoded: bytes) -> RewardArtifactIdentityV1:
        return cls.from_dict(decode_canonical_json_object(encoded))


__all__ = [
    "ACTUATOR_QVEL_INDICES_BY_ACTION_V1",
    "CANDIDATE_READ_SET_V1",
    "CONTROL_PERIOD_SECONDS",
    "GENERALIZED_ACTUATOR_TORQUE_CAPACITY_N_M",
    "REWARD_SCHEMA_V1",
    "SIGNED_TERM_NAMES_V1",
    "TARGET_SPEEDS_M_S",
    "TASK_TERM_ABS_MAX",
    "TOTAL_REWARD_ABS_MAX",
    "CandidateTaskInputsV1",
    "RewardArtifactIdentityV1",
    "RewardContractError",
    "RewardResultV1",
    "TaskTermV1",
    "TrustedRewardStepV1",
    "canonical_json_bytes",
    "decode_canonical_json_object",
    "reward_schema_sha256",
]
