"""Pure design and receipt contracts for disposable TQC calibration."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .fixed_reference import ExperimentContractError, read_bounded_json_artifact

MAX_DESIGN_BYTES = 128 * 1024
CALIBRATION_ID = "tqc_humanoid_resource_calibration/v1"
SEED_ROLE = "permanently_excluded_disposable_calibration"
MODEL_DISPOSITION = "discarded_unserialized"


def canonical_json(value: object) -> bytes:
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


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    observed = set(value)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ExperimentContractError(f"{field} keys differ: missing={missing}, extra={extra}")


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ExperimentContractError(f"{field} must be an object")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ExperimentContractError(f"{field} must be an integer >= {minimum}")
    return value


def _finite_number(value: object, *, field: str, minimum: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ExperimentContractError(f"{field} must be numeric")
    observed = float(value)
    if not math.isfinite(observed) or (minimum is not None and observed < minimum):
        raise ExperimentContractError(f"{field} must be finite and >= {minimum}")
    return observed


def _boolean(value: object, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ExperimentContractError(f"{field} must be boolean")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ExperimentContractError(f"{field} must be nonempty bounded text")
    return value


@dataclass(frozen=True, slots=True)
class EnvironmentDesign:
    environment_id: str
    n_envs: int
    worker_seeds: tuple[int, ...]
    vectorization: str
    terminate_when_unhealthy: bool
    reset_noise_scale: float
    exclude_current_positions_from_observation: bool
    frame_skip: int
    normalization: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EnvironmentDesign:
        _require_exact_keys(raw, set(cls.__dataclass_fields__), field="environment")
        worker_seeds = raw["worker_seeds"]
        if not isinstance(worker_seeds, list) or not worker_seeds:
            raise ExperimentContractError("environment.worker_seeds must be a nonempty list")
        design = cls(
            environment_id=_text(raw["environment_id"], field="environment.environment_id"),
            n_envs=_integer(raw["n_envs"], field="environment.n_envs", minimum=1),
            worker_seeds=tuple(
                _integer(value, field="environment.worker_seeds[]") for value in worker_seeds
            ),
            vectorization=_text(raw["vectorization"], field="environment.vectorization"),
            terminate_when_unhealthy=_boolean(
                raw["terminate_when_unhealthy"],
                field="environment.terminate_when_unhealthy",
            ),
            reset_noise_scale=_finite_number(
                raw["reset_noise_scale"], field="environment.reset_noise_scale", minimum=0.0
            ),
            exclude_current_positions_from_observation=_boolean(
                raw["exclude_current_positions_from_observation"],
                field="environment.exclude_current_positions_from_observation",
            ),
            frame_skip=_integer(raw["frame_skip"], field="environment.frame_skip", minimum=1),
            normalization=_text(raw["normalization"], field="environment.normalization"),
        )
        expected = {
            "environment_id": "Humanoid-v5",
            "vectorization": "stable_baselines3.DummyVecEnv/v2.9",
            "terminate_when_unhealthy": False,
            "exclude_current_positions_from_observation": True,
            "frame_skip": 5,
            "normalization": "none/v1",
        }
        for field, required in expected.items():
            if getattr(design, field) != required:
                raise ExperimentContractError(f"environment.{field} must equal {required!r}")
        return design


@dataclass(frozen=True, slots=True)
class TQCDesign:
    algorithm_id: str
    required_stable_baselines3_version: str
    required_sb3_contrib_version: str
    policy: str
    device: str
    total_timesteps: int
    learning_rate: float
    buffer_size: int
    learning_starts: int
    batch_size: int
    tau: float
    gamma: float
    train_freq_steps: int
    gradient_steps: int
    n_steps: int
    ent_coef: str
    target_entropy: str
    top_quantiles_to_drop_per_net: int
    n_quantiles: int
    n_critics: int
    policy_hidden_layers: tuple[int, ...]
    use_sde: bool
    optimize_memory_usage: bool
    handle_timeout_termination: bool

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TQCDesign:
        _require_exact_keys(raw, set(cls.__dataclass_fields__), field="tqc")
        hidden = raw["policy_hidden_layers"]
        if not isinstance(hidden, list) or not hidden or len(hidden) > 8:
            raise ExperimentContractError("tqc.policy_hidden_layers must be a bounded list")
        design = cls(
            algorithm_id=_text(raw["algorithm_id"], field="tqc.algorithm_id"),
            required_stable_baselines3_version=_text(
                raw["required_stable_baselines3_version"],
                field="tqc.required_stable_baselines3_version",
            ),
            required_sb3_contrib_version=_text(
                raw["required_sb3_contrib_version"],
                field="tqc.required_sb3_contrib_version",
            ),
            policy=_text(raw["policy"], field="tqc.policy"),
            device=_text(raw["device"], field="tqc.device"),
            total_timesteps=_integer(
                raw["total_timesteps"], field="tqc.total_timesteps", minimum=1
            ),
            learning_rate=_finite_number(
                raw["learning_rate"], field="tqc.learning_rate", minimum=0.0
            ),
            buffer_size=_integer(raw["buffer_size"], field="tqc.buffer_size", minimum=1),
            learning_starts=_integer(raw["learning_starts"], field="tqc.learning_starts"),
            batch_size=_integer(raw["batch_size"], field="tqc.batch_size", minimum=1),
            tau=_finite_number(raw["tau"], field="tqc.tau", minimum=0.0),
            gamma=_finite_number(raw["gamma"], field="tqc.gamma", minimum=0.0),
            train_freq_steps=_integer(
                raw["train_freq_steps"], field="tqc.train_freq_steps", minimum=1
            ),
            gradient_steps=_integer(raw["gradient_steps"], field="tqc.gradient_steps", minimum=1),
            n_steps=_integer(raw["n_steps"], field="tqc.n_steps", minimum=1),
            ent_coef=_text(raw["ent_coef"], field="tqc.ent_coef"),
            target_entropy=_text(raw["target_entropy"], field="tqc.target_entropy"),
            top_quantiles_to_drop_per_net=_integer(
                raw["top_quantiles_to_drop_per_net"],
                field="tqc.top_quantiles_to_drop_per_net",
            ),
            n_quantiles=_integer(raw["n_quantiles"], field="tqc.n_quantiles", minimum=1),
            n_critics=_integer(raw["n_critics"], field="tqc.n_critics", minimum=1),
            policy_hidden_layers=tuple(
                _integer(value, field="tqc.policy_hidden_layers[]", minimum=1) for value in hidden
            ),
            use_sde=_boolean(raw["use_sde"], field="tqc.use_sde"),
            optimize_memory_usage=_boolean(
                raw["optimize_memory_usage"], field="tqc.optimize_memory_usage"
            ),
            handle_timeout_termination=_boolean(
                raw["handle_timeout_termination"],
                field="tqc.handle_timeout_termination",
            ),
        )
        fixed = {
            "algorithm_id": "sb3_contrib.TQC/v2.9",
            "required_stable_baselines3_version": "2.9.0",
            "required_sb3_contrib_version": "2.9.0",
            "policy": "MlpPolicy",
            "device": "cpu",
            "train_freq_steps": 1,
            "n_steps": 1,
            "ent_coef": "auto",
            "target_entropy": "auto",
            "use_sde": False,
            "optimize_memory_usage": False,
            "handle_timeout_termination": True,
        }
        for field, required in fixed.items():
            if getattr(design, field) != required:
                raise ExperimentContractError(f"tqc.{field} must equal {required!r}")
        if design.learning_rate <= 0.0:
            raise ExperimentContractError("tqc.learning_rate must be positive")
        if not 0.0 < design.tau <= 1.0 or not 0.0 < design.gamma <= 1.0:
            raise ExperimentContractError("tqc.tau and tqc.gamma must be in (0, 1]")
        if design.total_timesteps <= design.learning_starts:
            raise ExperimentContractError("tqc.total_timesteps must exceed learning_starts")
        if design.buffer_size < design.total_timesteps:
            raise ExperimentContractError("tqc.buffer_size must hold every calibration transition")
        if design.top_quantiles_to_drop_per_net >= design.n_quantiles * design.n_critics:
            raise ExperimentContractError("tqc quantile drop removes every target quantile")
        return design


@dataclass(frozen=True, slots=True)
class ResourceGates:
    sampled_peak_rss_failure_threshold_bytes: int
    minimum_free_disk_bytes: int
    minimum_environment_steps_per_second: float
    throughput_warmup_environment_steps: int
    throughput_window_environment_steps: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ResourceGates:
        _require_exact_keys(raw, set(cls.__dataclass_fields__), field="resource_gates")
        return cls(
            sampled_peak_rss_failure_threshold_bytes=_integer(
                raw["sampled_peak_rss_failure_threshold_bytes"],
                field="resource_gates.sampled_peak_rss_failure_threshold_bytes",
                minimum=1,
            ),
            minimum_free_disk_bytes=_integer(
                raw["minimum_free_disk_bytes"],
                field="resource_gates.minimum_free_disk_bytes",
                minimum=1,
            ),
            minimum_environment_steps_per_second=_finite_number(
                raw["minimum_environment_steps_per_second"],
                field="resource_gates.minimum_environment_steps_per_second",
                minimum=0.0,
            ),
            throughput_warmup_environment_steps=_integer(
                raw["throughput_warmup_environment_steps"],
                field="resource_gates.throughput_warmup_environment_steps",
                minimum=1,
            ),
            throughput_window_environment_steps=_integer(
                raw["throughput_window_environment_steps"],
                field="resource_gates.throughput_window_environment_steps",
                minimum=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class PersistenceDesign:
    checkpoint: bool
    replay_buffer: bool
    normalizer: bool
    model_disposition: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PersistenceDesign:
        _require_exact_keys(raw, set(cls.__dataclass_fields__), field="persistence")
        design = cls(
            checkpoint=_boolean(raw["checkpoint"], field="persistence.checkpoint"),
            replay_buffer=_boolean(raw["replay_buffer"], field="persistence.replay_buffer"),
            normalizer=_boolean(raw["normalizer"], field="persistence.normalizer"),
            model_disposition=_text(
                raw["model_disposition"], field="persistence.model_disposition"
            ),
        )
        if design != cls(False, False, False, MODEL_DISPOSITION):
            raise ExperimentContractError("calibration persistence must be fully disabled")
        return design


@dataclass(frozen=True, slots=True)
class TQCCalibrationDesign:
    schema_version: int
    calibration_id: str
    design_status: str
    evidence_purpose: str
    claim_ceiling: None
    seed: int
    seed_role: str
    environment: EnvironmentDesign
    tqc: TQCDesign
    resource_gates: ResourceGates
    persistence: PersistenceDesign

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> TQCCalibrationDesign:
        _require_exact_keys(raw, set(cls.__dataclass_fields__), field="design")
        design = cls(
            schema_version=_integer(raw["schema_version"], field="schema_version", minimum=1),
            calibration_id=_text(raw["calibration_id"], field="calibration_id"),
            design_status=_text(raw["design_status"], field="design_status"),
            evidence_purpose=_text(raw["evidence_purpose"], field="evidence_purpose"),
            claim_ceiling=raw["claim_ceiling"],
            seed=_integer(raw["seed"], field="seed"),
            seed_role=_text(raw["seed_role"], field="seed_role"),
            environment=EnvironmentDesign.from_dict(
                _mapping(raw["environment"], field="environment")
            ),
            tqc=TQCDesign.from_dict(_mapping(raw["tqc"], field="tqc")),
            resource_gates=ResourceGates.from_dict(
                _mapping(raw["resource_gates"], field="resource_gates")
            ),
            persistence=PersistenceDesign.from_dict(
                _mapping(raw["persistence"], field="persistence")
            ),
        )
        expected = {
            "schema_version": 1,
            "calibration_id": CALIBRATION_ID,
            "design_status": "predeclared_not_executed",
            "evidence_purpose": "resource_calibration",
            "claim_ceiling": None,
            "seed_role": SEED_ROLE,
        }
        for field, required in expected.items():
            if getattr(design, field) != required:
                raise ExperimentContractError(f"{field} must equal {required!r}")
        exact_fields = {
            "seed": 92001,
            "environment.n_envs": 5,
            "environment.worker_seeds": (92001, 92002, 92003, 92004, 92005),
            "environment.reset_noise_scale": 0.01,
            "tqc.total_timesteps": 100_000,
            "tqc.learning_rate": 0.0003,
            "tqc.buffer_size": 1_000_000,
            "tqc.learning_starts": 100,
            "tqc.batch_size": 256,
            "tqc.tau": 0.005,
            "tqc.gamma": 0.99,
            "tqc.gradient_steps": 1,
            "tqc.top_quantiles_to_drop_per_net": 2,
            "tqc.n_quantiles": 25,
            "tqc.n_critics": 2,
            "tqc.policy_hidden_layers": (256, 256),
            "resource_gates.sampled_peak_rss_failure_threshold_bytes": 12 * 1024**3,
            "resource_gates.minimum_free_disk_bytes": 50 * 1024**3,
            "resource_gates.minimum_environment_steps_per_second": 116.0,
            "resource_gates.throughput_warmup_environment_steps": 10_000,
            "resource_gates.throughput_window_environment_steps": 10_000,
        }
        observed_fields = {
            "seed": design.seed,
            "environment.n_envs": design.environment.n_envs,
            "environment.worker_seeds": design.environment.worker_seeds,
            "environment.reset_noise_scale": design.environment.reset_noise_scale,
            "tqc.total_timesteps": design.tqc.total_timesteps,
            "tqc.learning_rate": design.tqc.learning_rate,
            "tqc.buffer_size": design.tqc.buffer_size,
            "tqc.learning_starts": design.tqc.learning_starts,
            "tqc.batch_size": design.tqc.batch_size,
            "tqc.tau": design.tqc.tau,
            "tqc.gamma": design.tqc.gamma,
            "tqc.gradient_steps": design.tqc.gradient_steps,
            "tqc.top_quantiles_to_drop_per_net": design.tqc.top_quantiles_to_drop_per_net,
            "tqc.n_quantiles": design.tqc.n_quantiles,
            "tqc.n_critics": design.tqc.n_critics,
            "tqc.policy_hidden_layers": design.tqc.policy_hidden_layers,
            "resource_gates.sampled_peak_rss_failure_threshold_bytes": (
                design.resource_gates.sampled_peak_rss_failure_threshold_bytes
            ),
            "resource_gates.minimum_free_disk_bytes": design.resource_gates.minimum_free_disk_bytes,
            "resource_gates.minimum_environment_steps_per_second": (
                design.resource_gates.minimum_environment_steps_per_second
            ),
            "resource_gates.throughput_warmup_environment_steps": (
                design.resource_gates.throughput_warmup_environment_steps
            ),
            "resource_gates.throughput_window_environment_steps": (
                design.resource_gates.throughput_window_environment_steps
            ),
        }
        mismatches = [
            field for field, required in exact_fields.items() if observed_fields[field] != required
        ]
        if mismatches:
            raise ExperimentContractError(
                "calibration v1 differs from its frozen values: " + ", ".join(mismatches)
            )
        if (
            design.resource_gates.throughput_warmup_environment_steps
            + design.resource_gates.throughput_window_environment_steps
            > design.tqc.total_timesteps
        ):
            raise ExperimentContractError("throughput warmup and window exceed the run budget")
        for field in (
            "throughput_warmup_environment_steps",
            "throughput_window_environment_steps",
        ):
            if getattr(design.resource_gates, field) % design.environment.n_envs:
                raise ExperimentContractError(f"resource_gates.{field} must align to n_envs")
        divisors = {
            "tqc.total_timesteps": design.tqc.total_timesteps,
            "tqc.learning_starts": design.tqc.learning_starts,
            "tqc.buffer_size": design.tqc.buffer_size,
        }
        for field, value in divisors.items():
            if value % design.environment.n_envs:
                raise ExperimentContractError(f"{field} must be divisible by environment.n_envs")
        if len(design.environment.worker_seeds) != design.environment.n_envs:
            raise ExperimentContractError("environment.worker_seeds must match environment.n_envs")
        return design

    @property
    def expected_vector_steps(self) -> int:
        return self.tqc.total_timesteps // self.environment.n_envs

    @property
    def expected_updates(self) -> int:
        return (
            (self.tqc.total_timesteps - self.tqc.learning_starts)
            // self.environment.n_envs
            * self.tqc.gradient_steps
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["environment"]["worker_seeds"] = list(self.environment.worker_seeds)
        payload["tqc"]["policy_hidden_layers"] = list(self.tqc.policy_hidden_layers)
        return payload

    @property
    def semantic_sha256(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True, slots=True)
class LoadedCalibrationDesign:
    design: TQCCalibrationDesign
    artifact_sha256: str
    artifact_byte_count: int


def load_calibration_design(path: Path) -> LoadedCalibrationDesign:
    artifact = read_bounded_json_artifact(
        path,
        maximum_bytes=MAX_DESIGN_BYTES,
        artifact="TQC calibration design",
    )
    return LoadedCalibrationDesign(
        design=TQCCalibrationDesign.from_dict(artifact.value),
        artifact_sha256=artifact.sha256,
        artifact_byte_count=len(artifact.encoded_bytes),
    )


def validate_runtime_receipt(runtime: object) -> dict[str, Any]:
    if not isinstance(runtime, dict):
        raise ExperimentContractError("runtime inspector must return an object")
    payload = dict(runtime)
    claimed_sha256 = payload.pop("runtime_sha256", None)
    if (
        not isinstance(claimed_sha256, str)
        or len(claimed_sha256) != 64
        or any(character not in "0123456789abcdef" for character in claimed_sha256)
    ):
        raise ExperimentContractError("runtime receipt has no valid semantic hash")
    if sha256_json(payload) != claimed_sha256:
        raise ExperimentContractError("runtime receipt semantic hash is inconsistent")
    return dict(runtime)


__all__ = [
    "CALIBRATION_ID",
    "MODEL_DISPOSITION",
    "SEED_ROLE",
    "EnvironmentDesign",
    "LoadedCalibrationDesign",
    "PersistenceDesign",
    "ResourceGates",
    "TQCCalibrationDesign",
    "TQCDesign",
    "canonical_json",
    "load_calibration_design",
    "sha256_json",
    "validate_runtime_receipt",
]
