"""Pinned Gymnasium Humanoid substrate and runtime inspection.

This module validates the simulator interface only. It does not provide a
reference-conditioned tracker and cannot admit an oracle-performance claim.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import gymnasium as gym


EXPECTED_ENV_ID = "Humanoid-v5"
EXPECTED_OBSERVATION_SHAPE = (348,)
EXPECTED_ACTION_SHAPE = (17,)
EXPECTED_QPOS_SHAPE = (24,)
EXPECTED_QVEL_SHAPE = (23,)
CONTACT_CAPTURE_ID = "all_mujoco_substeps_after_mj_step/v1"


class HumanoidContractError(RuntimeError):
    """Raised when the installed simulator does not match the frozen contract."""


@dataclass(frozen=True, slots=True)
class SubstepContactSample:
    """One exact active contact sampled after one MuJoCo physics step."""

    physics_substep_index: int
    contact_index_within_substep: int
    geom1_id: int
    geom2_id: int
    normal_force_n: float


_INSTRUMENTED_HUMANOID_CLASS: type[Any] | None = None


def _instrumented_humanoid_class() -> type[Any]:
    """Create the source-pinned Humanoid subclass only when Gym is installed."""

    global _INSTRUMENTED_HUMANOID_CLASS
    if _INSTRUMENTED_HUMANOID_CLASS is not None:
        return _INSTRUMENTED_HUMANOID_CLASS
    try:
        import mujoco
        from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise HumanoidContractError(
            "Gymnasium MuJoCo is unavailable; install the project with the 'gym' extra"
        ) from exc

    class SubstepContactHumanoidEnv(HumanoidEnv):
        """Humanoid-v5 dynamics with complete within-control-step contact capture."""

        contact_capture_id = CONTACT_CAPTURE_ID

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._last_control_step_contact_samples: tuple[SubstepContactSample, ...] = ()
            self._last_control_step_substeps = 0

        @property
        def last_control_step_contact_samples(self) -> tuple[SubstepContactSample, ...]:
            return self._last_control_step_contact_samples

        @property
        def last_control_step_substeps(self) -> int:
            return self._last_control_step_substeps

        def reset(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
            self._last_control_step_contact_samples = ()
            self._last_control_step_substeps = 0
            return super().reset(*args, **kwargs)

        def _step_mujoco_simulation(self, ctrl: object, n_frames: int) -> None:
            control = np.asarray(ctrl)
            if control.shape != (int(self.model.nu),):
                raise HumanoidContractError(
                    f"action dimension mismatch: {control.shape!r}; "
                    f"expected {(int(self.model.nu),)!r}"
                )
            if not isinstance(n_frames, int) or isinstance(n_frames, bool) or n_frames < 1:
                raise HumanoidContractError("MuJoCo substep count must be a positive integer")
            self._last_control_step_contact_samples = ()
            self._last_control_step_substeps = 0
            self.data.ctrl[:] = control
            samples: list[SubstepContactSample] = []
            for substep_index in range(n_frames):
                mujoco.mj_step(self.model, self.data, nstep=1)
                for contact_index in range(int(self.data.ncon)):
                    contact = self.data.contact[contact_index]
                    force_torque = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(
                        self.model,
                        self.data,
                        contact_index,
                        force_torque,
                    )
                    if not np.isfinite(force_torque).all():
                        raise HumanoidContractError(
                            "MuJoCo produced a non-finite substep contact force"
                        )
                    samples.append(
                        SubstepContactSample(
                            physics_substep_index=substep_index,
                            contact_index_within_substep=contact_index,
                            geom1_id=int(contact.geom1),
                            geom2_id=int(contact.geom2),
                            normal_force_n=abs(float(force_torque[0])),
                        )
                    )
            # Match Gymnasium's base implementation after the final substep.
            mujoco.mj_rnePostConstraint(self.model, self.data)
            self._last_control_step_contact_samples = tuple(samples)
            self._last_control_step_substeps = n_frames

    _INSTRUMENTED_HUMANOID_CLASS = SubstepContactHumanoidEnv
    return SubstepContactHumanoidEnv


@dataclass(frozen=True, slots=True)
class HumanoidExperimentConfig:
    """Environment choices that must remain fixed inside a comparison."""

    env_id: str = EXPECTED_ENV_ID
    terminate_when_unhealthy: bool = False
    reset_noise_scale: float = 0.01
    exclude_current_positions_from_observation: bool = True
    frame_skip: int = 5

    def __post_init__(self) -> None:
        if self.env_id != EXPECTED_ENV_ID:
            raise ValueError(f"env_id must be {EXPECTED_ENV_ID!r}")
        if not np.isfinite(self.reset_noise_scale) or self.reset_noise_scale < 0:
            raise ValueError("reset_noise_scale must be finite and non-negative")
        if self.frame_skip <= 0:
            raise ValueError("frame_skip must be positive")

    def gym_kwargs(self) -> dict[str, bool | float | int]:
        return {
            "terminate_when_unhealthy": self.terminate_when_unhealthy,
            "reset_noise_scale": self.reset_noise_scale,
            "exclude_current_positions_from_observation": (
                self.exclude_current_positions_from_observation
            ),
            "frame_skip": self.frame_skip,
        }


@dataclass(frozen=True, slots=True)
class HumanoidRuntimeReceipt:
    """Observed simulator facts from a real reset and step."""

    config: dict[str, Any]
    gymnasium_version: str
    mujoco_version: str
    model_sha256: str
    observation_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    qpos_shape: tuple[int, ...]
    qvel_shape: tuple[int, ...]
    timestep_seconds: float
    frame_skip: int
    control_period_seconds: float
    observation_finite: bool
    next_observation_finite: bool
    reward_finite: bool
    terminated_after_zero_action: bool
    truncated_after_zero_action: bool
    seed: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for field in ("observation_shape", "action_shape", "qpos_shape", "qvel_shape"):
            payload[field] = list(payload[field])
        return payload


def make_humanoid_env(
    config: HumanoidExperimentConfig | None = None,
    *,
    render_mode: str | None = None,
    capture_substep_contacts: bool = False,
) -> gym.Env:
    """Construct the declared Gymnasium environment without hidden fallbacks."""

    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - exercised without the gym extra
        raise HumanoidContractError(
            "Gymnasium is unavailable; install the project with the 'gym' extra"
        ) from exc

    selected = config or HumanoidExperimentConfig()
    if not isinstance(capture_substep_contacts, bool):
        raise HumanoidContractError("capture_substep_contacts must be boolean")
    if capture_substep_contacts:
        instrumented_class = _instrumented_humanoid_class()
        raw_environment = instrumented_class(
            render_mode=render_mode,
            **selected.gym_kwargs(),
        )
        # Reproduce Gymnasium's registered wrapper stack while retaining the
        # exact Humanoid-v5 EnvSpec. The runtime separately records the local
        # instrumentation identifier and source hash.
        raw_spec = copy.deepcopy(gym.spec(selected.env_id))
        raw_spec.max_episode_steps = None
        raw_spec.order_enforce = False
        raw_spec.disable_env_checker = True
        raw_spec.kwargs = {
            "render_mode": render_mode,
            **selected.gym_kwargs(),
        }
        raw_environment.spec = raw_spec
        checked = gym.wrappers.PassiveEnvChecker(raw_environment)
        ordered = gym.wrappers.OrderEnforcing(checked)
        registered_spec = gym.spec(selected.env_id)
        if registered_spec.max_episode_steps is None:
            raw_environment.close()
            raise HumanoidContractError("Humanoid-v5 has no registered episode limit")
        return gym.wrappers.TimeLimit(
            ordered,
            max_episode_steps=int(registered_spec.max_episode_steps),
        )
    return gym.make(
        selected.env_id,
        render_mode=render_mode,
        **selected.gym_kwargs(),
    )


def inspect_humanoid_runtime(
    config: HumanoidExperimentConfig | None = None,
    *,
    seed: int = 20260902,
) -> HumanoidRuntimeReceipt:
    """Reset, step, and fail if the observed substrate differs from the contract."""

    try:
        import gymnasium as gym
        import mujoco
    except ImportError as exc:  # pragma: no cover - exercised without the gym extra
        raise HumanoidContractError(
            "Gymnasium MuJoCo is unavailable; install the project with the 'gym' extra"
        ) from exc

    selected = config or HumanoidExperimentConfig()
    env = make_humanoid_env(selected)
    try:
        observation, _ = env.reset(seed=seed)
        action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        unwrapped = env.unwrapped
        model_path = Path(str(getattr(unwrapped, "fullpath", "")))
        if not model_path.is_file():
            raise HumanoidContractError("Gymnasium did not expose a readable MuJoCo model file")

        receipt = HumanoidRuntimeReceipt(
            config=asdict(selected),
            gymnasium_version=gym.__version__,
            mujoco_version=mujoco.__version__,
            model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
            observation_shape=tuple(int(value) for value in observation.shape),
            action_shape=tuple(int(value) for value in env.action_space.shape),
            qpos_shape=tuple(int(value) for value in unwrapped.data.qpos.shape),
            qvel_shape=tuple(int(value) for value in unwrapped.data.qvel.shape),
            timestep_seconds=float(unwrapped.model.opt.timestep),
            frame_skip=int(unwrapped.frame_skip),
            control_period_seconds=(
                float(unwrapped.model.opt.timestep) * int(unwrapped.frame_skip)
            ),
            observation_finite=bool(np.isfinite(observation).all()),
            next_observation_finite=bool(np.isfinite(next_observation).all()),
            reward_finite=bool(np.isfinite(reward)),
            terminated_after_zero_action=bool(terminated),
            truncated_after_zero_action=bool(truncated),
            seed=int(seed),
        )
        _validate_runtime_receipt(receipt)
        return receipt
    finally:
        env.close()


def _validate_runtime_receipt(receipt: HumanoidRuntimeReceipt) -> None:
    expected_shapes = {
        "observation_shape": EXPECTED_OBSERVATION_SHAPE,
        "action_shape": EXPECTED_ACTION_SHAPE,
        "qpos_shape": EXPECTED_QPOS_SHAPE,
        "qvel_shape": EXPECTED_QVEL_SHAPE,
    }
    mismatches = [
        f"{field}={getattr(receipt, field)!r}, expected={expected!r}"
        for field, expected in expected_shapes.items()
        if getattr(receipt, field) != expected
    ]
    if receipt.frame_skip != receipt.config["frame_skip"]:
        mismatches.append(
            f"frame_skip={receipt.frame_skip!r}, expected={receipt.config['frame_skip']!r}"
        )
    if not receipt.observation_finite:
        mismatches.append("reset observation contains non-finite values")
    if not receipt.next_observation_finite:
        mismatches.append("step observation contains non-finite values")
    if not receipt.reward_finite:
        mismatches.append("step reward is non-finite")
    if receipt.terminated_after_zero_action or receipt.truncated_after_zero_action:
        mismatches.append("zero-action smoke step ended the episode")
    if mismatches:
        raise HumanoidContractError("Humanoid runtime contract failed: " + "; ".join(mismatches))
