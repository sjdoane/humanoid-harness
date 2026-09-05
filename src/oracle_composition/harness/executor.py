"""Plain Humanoid-v5 controller-switching executor and objective episode facts."""

from __future__ import annotations

import hashlib
import math
import platform
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import (
    array_sha256,
    canonical_json_bytes,
    sha256_file,
)
from oracle_composition.envs import humanoid as humanoid_module
from oracle_composition.envs.humanoid import HumanoidExperimentConfig
from oracle_composition.experiments.runtime_identity import dependency_lock_path, module_sha256
from oracle_composition.sources import strict_tqc_actor_runtime as actor_runtime_module

from .contract import ALLOWED_SIGNALS, EVIDENCE_CLASS, OracleMachine, OracleProgram
from .inputs import TaskSpec

TRACE_SCHEMA_ID = "humanoid_controller_switching_trace/v1"
RUNTIME_FINGERPRINT_ID = "plain_humanoid_v5_controller_switching_runtime/v1"
EXPECTED_PLAIN_WRAPPER_TYPES = (
    "gymnasium.wrappers.common.TimeLimit",
    "gymnasium.wrappers.common.OrderEnforcing",
    "gymnasium.wrappers.common.PassiveEnvChecker",
    "gymnasium.envs.mujoco.humanoid_v5.HumanoidEnv",
)


class CompositionRuntimeError(RuntimeError):
    """Raised when the frozen simulator/executor path changes or cannot finish."""


@dataclass(frozen=True, slots=True)
class BoundaryFacts:
    root_x_m: float
    root_height_m: float
    torso_up_z: float

    @property
    def fallen(self) -> bool:
        return not 1.0 < self.root_height_m < 2.0 or self.torso_up_z < 0.5


@dataclass(frozen=True, slots=True)
class MetricSample:
    t: int
    target_speed_m_s: float
    forward_speed_m_s: float
    root_height_m: float
    torso_up_z: float
    task_reward: float
    active_behavior: str
    controller_switched: bool


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    seed: int
    observed_steps: int
    mean_absolute_speed_error_m_s: float
    fall: bool
    first_fall_step: int | None
    switch_count: int
    time_in_each_behavior_steps: Mapping[str, int]
    task_return: float

    def to_dict(self) -> dict[str, object]:
        return {
            "fall": self.fall,
            "first_fall_step": self.first_fall_step,
            "mean_absolute_speed_error_m_s": self.mean_absolute_speed_error_m_s,
            "observed_steps": self.observed_steps,
            "seed": self.seed,
            "switch_count": self.switch_count,
            "task_return": self.task_return,
            "time_in_each_behavior_steps": dict(self.time_in_each_behavior_steps),
        }


@dataclass(frozen=True, slots=True)
class EpisodeExecution:
    metrics: EpisodeMetrics
    initial_boundary: BoundaryFacts
    metric_samples: tuple[MetricSample, ...]
    trace_bytes: bytes
    trace_sha256: str


def _finite(value: object, *, field: str) -> float:
    if type(value) not in {int, float, np.float32, np.float64}:
        raise CompositionRuntimeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CompositionRuntimeError(f"{field} must be finite")
    return result


def _wrapper_types(environment: object) -> tuple[str, ...]:
    result: list[str] = []
    current = environment
    while True:
        result.append(f"{type(current).__module__}.{type(current).__name__}")
        if not hasattr(current, "env"):
            return tuple(result)
        current = current.env


def runtime_fingerprint(environment: object, task: TaskSpec) -> tuple[dict[str, object], str]:
    """Bind the registered plain runtime without consuming an evaluation reset."""

    try:
        import gymnasium
        import mujoco
        import torch
        from gymnasium.envs.mujoco import humanoid_v5, mujoco_env
        from gymnasium.utils import passive_env_checker
        from gymnasium.wrappers import common as wrapper_common
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise CompositionRuntimeError(
            "frozen Gymnasium/Torch/MuJoCo dependencies are unavailable"
        ) from exc
    wrappers = _wrapper_types(environment)
    if wrappers != EXPECTED_PLAIN_WRAPPER_TYPES:
        raise CompositionRuntimeError(
            "plain Humanoid-v5 wrapper stack differs from the corpus runtime"
        )
    physical = environment.unwrapped
    if str(getattr(environment.spec, "id", "")) != "Humanoid-v5":
        raise CompositionRuntimeError("environment id differs from Humanoid-v5")
    if int(getattr(environment, "_max_episode_steps", -1)) != 1000:
        raise CompositionRuntimeError("Humanoid-v5 TimeLimit differs from 1000")
    if int(getattr(physical, "frame_skip", -1)) != 5:
        raise CompositionRuntimeError("Humanoid-v5 frame_skip differs from 5")
    timestep = _finite(physical.model.opt.timestep, field="physics timestep")
    control_period = timestep * int(physical.frame_skip)
    if timestep != 0.003 or control_period != task.control_period_seconds:
        raise CompositionRuntimeError("Humanoid-v5 control cadence differs from the task")
    if bool(getattr(physical, "_terminate_when_unhealthy", True)):
        raise CompositionRuntimeError("terminate_when_unhealthy must be false")
    if tuple(int(value) for value in environment.observation_space.shape) != (348,):
        raise CompositionRuntimeError("Humanoid-v5 observation shape differs")
    if tuple(int(value) for value in environment.action_space.shape) != (17,):
        raise CompositionRuntimeError("Humanoid-v5 action shape differs")
    model_path = Path(str(getattr(physical, "fullpath", "")))
    if model_path.is_symlink() or not model_path.is_file():
        raise CompositionRuntimeError("Humanoid-v5 model bytes are unavailable")
    fingerprint: dict[str, object] = {
        "action_shape": [17],
        "control_period_seconds": control_period,
        "dependency_versions": {
            "gymnasium": gymnasium.__version__,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "torch": str(torch.__version__),
        },
        "environment_id": "Humanoid-v5",
        "environment_kwargs": HumanoidExperimentConfig().gym_kwargs(),
        "evidence_class": EVIDENCE_CLASS,
        "frame_skip": 5,
        "model_sha256": sha256_file(model_path),
        "observation_shape": [348],
        "platform_machine": platform.machine(),
        "runtime_fingerprint_id": RUNTIME_FINGERPRINT_ID,
        "source_sha256": {
            "gymnasium": module_sha256(gymnasium),
            "gymnasium_humanoid_v5": module_sha256(humanoid_v5),
            "gymnasium_mujoco_env": module_sha256(mujoco_env),
            "gymnasium_passive_checker": module_sha256(passive_env_checker),
            "gymnasium_wrapper_common": module_sha256(wrapper_common),
            "harness_executor": sha256_file(Path(__file__)),
            "humanoid_environment": module_sha256(humanoid_module),
            "strict_tqc_actor_runtime": module_sha256(actor_runtime_module),
            "uv_lock": sha256_file(dependency_lock_path()),
        },
        "terminate_when_unhealthy": False,
        "time_limit_steps": 1000,
        "timestep_seconds": timestep,
        "wrapper_types": list(wrappers),
    }
    encoded = canonical_json_bytes(fingerprint)
    return fingerprint, hashlib.sha256(encoded).hexdigest()


def _boundary_facts(environment: object) -> BoundaryFacts:
    qpos = np.asarray(environment.unwrapped.data.qpos, dtype=np.float64)
    if qpos.shape != (24,) or not np.isfinite(qpos).all():
        raise CompositionRuntimeError("MuJoCo qpos must be finite shape (24,)")
    w, x, y, z = (float(value) for value in qpos[3:7])
    norm_squared = w * w + x * x + y * y + z * z
    if not math.isfinite(norm_squared) or norm_squared <= 0.0:
        raise CompositionRuntimeError("MuJoCo root quaternion is invalid")
    torso_up = (w * w - x * x - y * y + z * z) / norm_squared
    if not math.isfinite(torso_up) or not -1.0 <= torso_up <= 1.0:
        raise CompositionRuntimeError("MuJoCo torso up-axis is invalid")
    return BoundaryFacts(
        root_x_m=float(qpos[0]),
        root_height_m=float(qpos[2]),
        torso_up_z=torso_up,
    )


def compute_episode_metrics(
    *,
    seed: int,
    initial_boundary: BoundaryFacts,
    samples: Sequence[MetricSample],
    behavior_names: Sequence[str],
) -> EpisodeMetrics:
    """Compute objective metrics from state/reward samples, never from a trace file."""

    if not samples:
        raise CompositionRuntimeError("an episode must contain at least one metric sample")
    errors = [abs(sample.forward_speed_m_s - sample.target_speed_m_s) for sample in samples]
    if not all(math.isfinite(value) for value in errors):
        raise CompositionRuntimeError("speed error is non-finite")
    first_fall: int | None = 0 if initial_boundary.fallen else None
    if first_fall is None:
        for sample in samples:
            if not 1.0 < sample.root_height_m < 2.0 or sample.torso_up_z < 0.5:
                first_fall = sample.t + 1
                break
    counts = Counter(sample.active_behavior for sample in samples)
    expected = set(behavior_names)
    if set(counts) - expected:
        raise CompositionRuntimeError("metric samples contain an undefined behavior")
    task_return = math.fsum(sample.task_reward for sample in samples)
    return EpisodeMetrics(
        seed=seed,
        observed_steps=len(samples),
        mean_absolute_speed_error_m_s=math.fsum(errors) / len(errors),
        fall=first_fall is not None,
        first_fall_step=first_fall,
        switch_count=sum(int(sample.controller_switched) for sample in samples),
        time_in_each_behavior_steps=MappingProxyType(
            {name: int(counts.get(name, 0)) for name in behavior_names}
        ),
        task_return=task_return,
    )


def _physical_action(actor: object, observation: np.ndarray) -> tuple[np.ndarray, str]:
    try:
        result = actor.act(observation)
        action = np.ascontiguousarray(result.physical_action, dtype="<f4")
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise CompositionRuntimeError("actor did not return a physical action") from exc
    if action.shape != (17,) or not np.isfinite(action).all():
        raise CompositionRuntimeError("actor physical action must be finite shape (17,)")
    return action, array_sha256(action)


def execute_episode(
    *,
    environment: object,
    actors: Mapping[str, object],
    program: OracleProgram,
    task: TaskSpec,
    seed: int,
    runtime_fingerprint_sha256: str,
) -> EpisodeExecution:
    """Run one deterministic episode and retain a metric source separate from trace bytes."""

    if not set(program.behaviors).issubset(actors):
        raise CompositionRuntimeError("an oracle behavior has no loaded actor")
    try:
        raw_observation, _reset_info = environment.reset(seed=seed)
    except Exception as exc:  # pragma: no cover - external runtime boundary
        raise CompositionRuntimeError("Humanoid-v5 reset failed") from exc
    observation = np.ascontiguousarray(raw_observation, dtype=np.float64)
    if observation.shape != (348,) or not np.isfinite(observation).all():
        raise CompositionRuntimeError("reset observation must be finite shape (348,)")
    initial = _boundary_facts(environment)
    previous = initial
    last_speed = 0.0
    machine = OracleMachine(program)
    metric_samples: list[MetricSample] = []
    trace_rows: list[dict[str, object]] = []
    for step in range(task.horizon_steps):
        target = task.target_at(step)
        signals: dict[str, object] = {
            "dwell": machine.dwell,
            "t": step,
            "torso_up": previous.torso_up_z,
            "v_target": target,
            "v_x": last_speed,
            "x_travelled": previous.root_x_m - initial.root_x_m,
            "z_root": previous.root_height_m,
        }
        if set(signals) != ALLOWED_SIGNALS:
            raise AssertionError("executor signal set drifted")
        decision = machine.decide(signals)
        action, action_sha256 = _physical_action(actors[decision.behavior], observation)
        try:
            raw_next, reward, terminated, truncated, info = environment.step(action)
        except Exception as exc:  # pragma: no cover - external runtime boundary
            raise CompositionRuntimeError(f"Humanoid-v5 step {step} failed") from exc
        next_observation = np.ascontiguousarray(raw_next, dtype=np.float64)
        if next_observation.shape != (348,) or not np.isfinite(next_observation).all():
            raise CompositionRuntimeError("step observation must be finite shape (348,)")
        reward_value = _finite(reward, field="stock reward")
        if not isinstance(info, Mapping) or "x_velocity" not in info:
            raise CompositionRuntimeError("stock Humanoid-v5 info.x_velocity is unavailable")
        stock_info_speed = _finite(info["x_velocity"], field="stock info.x_velocity")
        current = _boundary_facts(environment)
        speed = (current.root_x_m - previous.root_x_m) / task.control_period_seconds
        if not math.isfinite(speed):
            raise CompositionRuntimeError("root-x forward speed is non-finite")
        terminated_flag = bool(terminated)
        truncated_flag = bool(truncated)
        if terminated_flag or (truncated_flag and step + 1 != task.horizon_steps):
            raise CompositionRuntimeError("episode ended before the frozen evaluation horizon")
        metric_samples.append(
            MetricSample(
                t=step,
                target_speed_m_s=target,
                forward_speed_m_s=speed,
                root_height_m=current.root_height_m,
                torso_up_z=current.torso_up_z,
                task_reward=reward_value,
                active_behavior=decision.behavior,
                controller_switched=decision.controller_switched,
            )
        )
        trace_rows.append(
            {
                "active_behavior": decision.behavior,
                "active_state": decision.state,
                "physical_action_sha256": action_sha256,
                "post_boundary": {
                    "stock_info_x_velocity_m_s": stock_info_speed,
                    "task_reward": reward_value,
                    "torso_up": current.torso_up_z,
                    "truncated": truncated_flag,
                    "v_x": speed,
                    "x_travelled": current.root_x_m - initial.root_x_m,
                    "z_root": current.root_height_m,
                },
                "signals": signals,
                "switch_flags": {
                    "controller_switched": decision.controller_switched,
                    "recovery_entered": decision.recovery_entered,
                    "recovery_exited": decision.recovery_exited,
                    "state_transition": decision.state_transition,
                },
                "switch_reason": decision.reason,
                "t": step,
            }
        )
        machine.advance()
        observation = next_observation
        previous = current
        last_speed = speed
    metrics = compute_episode_metrics(
        seed=seed,
        initial_boundary=initial,
        samples=metric_samples,
        behavior_names=tuple(actors),
    )
    trace = {
        "evidence_class": EVIDENCE_CLASS,
        "oracle_id": program.oracle_id,
        "oracle_sha256": program.sha256,
        "runtime_fingerprint_sha256": runtime_fingerprint_sha256,
        "schema_version": 1,
        "seed": seed,
        "steps": trace_rows,
        "trace_schema_id": TRACE_SCHEMA_ID,
    }
    trace_bytes = canonical_json_bytes(trace)
    return EpisodeExecution(
        metrics=metrics,
        initial_boundary=initial,
        metric_samples=tuple(metric_samples),
        trace_bytes=trace_bytes,
        trace_sha256=hashlib.sha256(trace_bytes).hexdigest(),
    )


__all__ = [
    "EXPECTED_PLAIN_WRAPPER_TYPES",
    "BoundaryFacts",
    "CompositionRuntimeError",
    "EpisodeExecution",
    "EpisodeMetrics",
    "MetricSample",
    "compute_episode_metrics",
    "execute_episode",
    "runtime_fingerprint",
]
