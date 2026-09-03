"""Minimal adapter for inspecting the fixed reference-tracker runtime.

This module intentionally does not select hyperparameters, train a policy, or
write a frozen manifest.  A reviewed manifest must be supplied to a later
training command and must exactly match :func:`inspect_runtime`.
"""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import gymnasium as gym
import numpy as np

from oracle_composition.envs import humanoid as humanoid_module
from oracle_composition.envs import reference_tracking as reference_tracking_module
from oracle_composition.envs.humanoid import make_humanoid_env
from oracle_composition.envs.reference_tracking import FixedReferenceTrackingWrapper
from oracle_composition.tracking import (
    TrackingRewardConfig,
    make_static_stand_reference,
    validate_humanoid_actuator_abi,
)
from oracle_composition.tracking import humanoid_reference as humanoid_reference_module
from oracle_composition.tracking import reward as tracking_reward_module

from . import fixed_reference as fixed_reference_module
from .fixed_reference import (
    FixedReferenceStudyDesign,
    FrozenExecutionManifest,
    RuntimeFingerprint,
    sha256_file,
)
from .protected_evaluator import protected_evaluator_sha256
from .runtime_identity import (
    dependency_lock_path,
    module_sha256,
    source_tree_sha256,
    space_sha256,
)
from .squashed_policy import (
    ACTION_TRANSFORM_ID,
    POLICY_ID,
    squashed_policy_source_sha256,
)

ZERO_TASK_REWARD_SPEC = {
    "reward_id": "constant_zero_task_reward/v1",
    "value": 0.0,
}

OBSERVATION_NORMALIZER_ID = "none/v1"
REWARD_NORMALIZER_ID = "none/v1"


def zero_task_reward_sha256() -> str:
    """Bind the explicit absence of a task reward as a frozen component."""

    encoded = json.dumps(
        ZERO_TASK_REWARD_SPEC,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runner_source_sha256() -> str:
    """Hash the exact runtime inspection adapter source."""

    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def execution_source_sha256() -> str:
    """Hash the training/evaluation implementation used after manifest review."""

    return sha256_file(Path(__file__).with_name("execution.py"))


def study_summary_source_sha256() -> str:
    """Hash the exact five-seed decision implementation before training."""

    return sha256_file(Path(__file__).with_name("study_summary.py"))


def make_static_tracking_env(
    design: FixedReferenceStudyDesign,
) -> gym.Env:
    """Construct only the declared 45D static positive-control environment."""

    reference = make_static_stand_reference(n_frames=design.static_reference_frames)
    base_environment = make_humanoid_env(capture_substep_contacts=True)
    observed_horizon = getattr(base_environment, "_max_episode_steps", None)
    if observed_horizon != design.max_episode_steps:
        base_environment.close()
        raise RuntimeError(
            "declared max_episode_steps does not match the installed Gym TimeLimit: "
            f"declared={design.max_episode_steps}, observed={observed_horizon!r}"
        )
    tracker = FixedReferenceTrackingWrapper(
        base_environment,
        reference=reference,
        horizon_steps=design.horizon_steps,
        reward_config=TrackingRewardConfig(),
    )
    normalized_low = -np.ones(tracker.action_space.shape, dtype=tracker.action_space.dtype)
    normalized_high = np.ones(tracker.action_space.shape, dtype=tracker.action_space.dtype)
    return gym.wrappers.RescaleAction(
        tracker,
        min_action=normalized_low,
        max_action=normalized_high,
    )


def _tracker_and_time_limit(environment: gym.Env) -> tuple[FixedReferenceTrackingWrapper, object]:
    """Require the reviewed outer-rescale -> tracker -> TimeLimit topology."""

    if not isinstance(environment, gym.wrappers.RescaleAction):
        raise RuntimeError("outer environment is not the required RescaleAction wrapper")
    tracker = environment.env
    if not isinstance(tracker, FixedReferenceTrackingWrapper):
        raise RuntimeError("RescaleAction does not directly wrap the reference tracker")
    time_limit = tracker.env
    if not isinstance(time_limit, gym.wrappers.TimeLimit):
        raise RuntimeError("reference tracker does not directly wrap Gym TimeLimit")
    return tracker, time_limit


def _audit_action_rescale(environment: gym.Env) -> None:
    """Probe the sole normalized-to-physical mapping at all affine endpoints."""

    tracker, _time_limit = _tracker_and_time_limit(environment)
    normalized_space = environment.action_space
    physical_space = tracker.action_space
    if not isinstance(normalized_space, gym.spaces.Box) or not isinstance(
        physical_space,
        gym.spaces.Box,
    ):
        raise RuntimeError("action mapping requires normalized and physical Box spaces")
    expected_normalized_low = -np.ones(
        normalized_space.shape,
        dtype=normalized_space.dtype,
    )
    expected_normalized_high = np.ones(
        normalized_space.shape,
        dtype=normalized_space.dtype,
    )
    if not np.array_equal(normalized_space.low, expected_normalized_low) or not np.array_equal(
        normalized_space.high,
        expected_normalized_high,
    ):
        raise RuntimeError("outer action space is not the exact normalized [-1, 1] Box")

    physical_low = np.asarray(physical_space.low, dtype=np.float64)
    physical_high = np.asarray(physical_space.high, dtype=np.float64)
    for normalized_value, expected_control in (
        (-1.0, physical_low),
        (0.0, (physical_low + physical_high) / 2.0),
        (1.0, physical_high),
    ):
        environment.reset(seed=20260902)
        normalized_action = np.full(
            normalized_space.shape,
            normalized_value,
            dtype=normalized_space.dtype,
        )
        environment.step(normalized_action)
        observed_control = np.asarray(environment.unwrapped.data.ctrl, dtype=np.float64)
        if not np.array_equal(observed_control, expected_control):
            raise RuntimeError(
                "normalized action endpoint did not map exactly once to physical control"
            )


def inspect_runtime(design: FixedReferenceStudyDesign) -> RuntimeFingerprint:
    """Observe all runtime facts required by a frozen execution manifest."""

    try:
        import gymnasium
        import mujoco
        import stable_baselines3
        import torch
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise RuntimeError("install the gym and train extras before runtime inspection") from exc

    env = make_static_tracking_env(design)
    try:
        tracker, time_limit = _tracker_and_time_limit(env)
        _audit_action_rescale(env)
        observation, info = env.reset(seed=design.evaluation_seeds[0])
        unwrapped = env.unwrapped
        actuator_abi = validate_humanoid_actuator_abi(unwrapped)
        model_path = Path(str(getattr(unwrapped, "fullpath", "")))
        telemetry = info.get("reference_tracking")
        if not isinstance(telemetry, dict):
            raise RuntimeError("tracking wrapper did not emit reset provenance")
        reference = make_static_stand_reference(n_frames=design.static_reference_frames)
        reward = TrackingRewardConfig()
        if telemetry.get("reference_content_sha256") != reference.identity.content_sha256:
            raise RuntimeError("wrapper reset reported a different reference")
        if telemetry.get("tracking_reward_sha256") != reward.sha256:
            raise RuntimeError("wrapper reset reported a different tracking reward")
        if tuple(observation.shape) != tuple(env.observation_space.shape):
            raise RuntimeError("reset observation does not match the declared wrapper space")

        return RuntimeFingerprint(
            env_id=str(unwrapped.spec.id),
            gymnasium_version=str(gymnasium.__version__),
            mujoco_version=str(mujoco.__version__),
            numpy_version=str(np.__version__),
            stable_baselines3_version=str(stable_baselines3.__version__),
            torch_version=str(torch.__version__),
            python_version=platform.python_version(),
            platform_system=platform.system(),
            platform_machine=platform.machine(),
            model_sha256=sha256_file(model_path),
            dependency_lock_sha256=sha256_file(dependency_lock_path()),
            observation_space_sha256=space_sha256(env.observation_space),
            action_space_sha256=space_sha256(env.action_space),
            physical_action_space_sha256=space_sha256(tracker.action_space),
            observation_shape=tuple(int(value) for value in env.observation_space.shape),
            action_shape=tuple(int(value) for value in env.action_space.shape),
            qpos_shape=tuple(int(value) for value in unwrapped.data.qpos.shape),
            qvel_shape=tuple(int(value) for value in unwrapped.data.qvel.shape),
            actuator_gear_by_joint=actuator_abi.actuator_gear_by_joint,
            generalized_actuator_torque_capacity_n_m=(
                actuator_abi.generalized_actuator_torque_capacity_n_m
            ),
            environment_max_episode_steps=int(time_limit._max_episode_steps),
            control_period_seconds=float(unwrapped.dt),
            reference_content_sha256=reference.identity.content_sha256,
            reference_schema_sha256=reference.identity.schema_sha256,
            tracking_reward_sha256=reward.sha256,
            task_reward_sha256=zero_task_reward_sha256(),
            environment_source_sha256=module_sha256(humanoid_module),
            reference_abi_source_sha256=module_sha256(humanoid_reference_module),
            tracking_reward_source_sha256=module_sha256(tracking_reward_module),
            wrapper_source_sha256=module_sha256(reference_tracking_module),
            experiment_contract_source_sha256=module_sha256(fixed_reference_module),
            evaluator_source_sha256=protected_evaluator_sha256(),
            runner_source_sha256=runner_source_sha256(),
            execution_source_sha256=execution_source_sha256(),
            study_summary_source_sha256=study_summary_source_sha256(),
            policy_source_sha256=squashed_policy_source_sha256(),
            source_tree_sha256=source_tree_sha256(),
            policy_id=POLICY_ID,
            action_transform_id=ACTION_TRANSFORM_ID,
            observation_normalizer_id=OBSERVATION_NORMALIZER_ID,
            reward_normalizer_id=REWARD_NORMALIZER_ID,
        )
    finally:
        env.close()


def validate_before_run(
    *,
    design: FixedReferenceStudyDesign,
    manifest: FrozenExecutionManifest,
) -> RuntimeFingerprint:
    """Reinspect and reject drift before any calibration, training, or evaluation."""

    observed = inspect_runtime(design)
    manifest.validate(design=design, observed=observed)
    return observed
