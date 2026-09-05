"""Full-force Humanoid instrumentation for the same-runtime reference corpus."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .humanoid import HumanoidContractError, HumanoidExperimentConfig

if TYPE_CHECKING:
    import gymnasium as gym

FULL_CONTACT_CAPTURE_ID = "all_five_substeps_geom_names_contact_frame_force6/v1"
EXPECTED_REFERENCE_WRAPPER_TYPES = (
    "gymnasium.wrappers.common.TimeLimit",
    "gymnasium.wrappers.common.OrderEnforcing",
    "gymnasium.wrappers.common.PassiveEnvChecker",
    "oracle_composition.envs.reference_corpus.FullContactHumanoidEnv",
)


@dataclass(frozen=True, slots=True)
class FullSubstepContactSample:
    """One active MuJoCo contact and its exact six-axis contact-frame force."""

    physics_substep_index: int
    contact_index_within_substep: int
    geom1_id: int
    geom2_id: int
    geom1_name: str
    geom2_name: str
    force_torque_contact_frame: tuple[float, float, float, float, float, float]

    def __post_init__(self) -> None:
        integers = (
            self.physics_substep_index,
            self.contact_index_within_substep,
            self.geom1_id,
            self.geom2_id,
        )
        if any(type(value) is not int or value < 0 for value in integers):
            raise HumanoidContractError("contact indices must be non-negative exact integers")
        if not 0 <= self.physics_substep_index < 5:
            raise HumanoidContractError("contact substep index must be in [0, 5)")
        if any(type(value) is not str or not value for value in (self.geom1_name, self.geom2_name)):
            raise HumanoidContractError("contact geometry names must be nonempty exact strings")
        if (
            type(self.force_torque_contact_frame) is not tuple
            or len(self.force_torque_contact_frame) != 6
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in self.force_torque_contact_frame
            )
        ):
            raise HumanoidContractError("contact force must be a finite six-float tuple")

    @property
    def normal_force_n(self) -> float:
        return abs(self.force_torque_contact_frame[0])


_FULL_CONTACT_CLASS: type[Any] | None = None


def _full_contact_class() -> type[Any]:
    global _FULL_CONTACT_CLASS
    if _FULL_CONTACT_CLASS is not None:
        return _FULL_CONTACT_CLASS
    try:
        import mujoco
        from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise HumanoidContractError("Gymnasium MuJoCo is unavailable") from exc

    class FullContactHumanoidEnv(HumanoidEnv):
        contact_capture_id = FULL_CONTACT_CAPTURE_ID

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            names: list[str] = []
            for geom_id in range(int(self.model.ngeom)):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
                if not name:
                    raise HumanoidContractError("Humanoid model contains an unnamed geometry")
                names.append(name)
            self._reference_corpus_geom_names = tuple(names)
            self._last_full_contact_samples: tuple[FullSubstepContactSample, ...] = ()
            self._last_full_contact_substeps = 0

        @property
        def last_full_contact_samples(self) -> tuple[FullSubstepContactSample, ...]:
            return self._last_full_contact_samples

        @property
        def last_full_contact_substeps(self) -> int:
            return self._last_full_contact_substeps

        @property
        def reference_corpus_geom_names(self) -> tuple[str, ...]:
            return self._reference_corpus_geom_names

        def reset(self, *args: Any, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
            self._last_full_contact_samples = ()
            self._last_full_contact_substeps = 0
            return super().reset(*args, **kwargs)

        def _step_mujoco_simulation(self, ctrl: object, n_frames: int) -> None:
            control = np.asarray(ctrl)
            if control.shape != (int(self.model.nu),):
                raise HumanoidContractError("action dimension differs from the model")
            if type(n_frames) is not int or n_frames != 5:
                raise HumanoidContractError("reference corpus requires exactly five substeps")
            self._last_full_contact_samples = ()
            self._last_full_contact_substeps = 0
            self.data.ctrl[:] = control
            samples: list[FullSubstepContactSample] = []
            for substep_index in range(n_frames):
                mujoco.mj_step(self.model, self.data, nstep=1)
                for contact_index in range(int(self.data.ncon)):
                    contact = self.data.contact[contact_index]
                    force = np.empty(6, dtype=np.float64)
                    mujoco.mj_contactForce(self.model, self.data, contact_index, force)
                    if not np.isfinite(force).all():
                        raise HumanoidContractError("MuJoCo produced a non-finite contact force")
                    geom1_id = int(contact.geom1)
                    geom2_id = int(contact.geom2)
                    samples.append(
                        FullSubstepContactSample(
                            physics_substep_index=substep_index,
                            contact_index_within_substep=contact_index,
                            geom1_id=geom1_id,
                            geom2_id=geom2_id,
                            geom1_name=self._reference_corpus_geom_names[geom1_id],
                            geom2_name=self._reference_corpus_geom_names[geom2_id],
                            force_torque_contact_frame=tuple(float(value) for value in force),
                        )
                    )
            mujoco.mj_rnePostConstraint(self.model, self.data)
            self._last_full_contact_samples = tuple(samples)
            self._last_full_contact_substeps = n_frames

    FullContactHumanoidEnv.__module__ = __name__
    _FULL_CONTACT_CLASS = FullContactHumanoidEnv
    return FullContactHumanoidEnv


def make_reference_corpus_env(
    config: HumanoidExperimentConfig | None = None,
) -> gym.Env:
    """Construct the exact instrumented Humanoid-v5 wrapper stack."""

    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise HumanoidContractError("Gymnasium is unavailable") from exc
    selected = config or HumanoidExperimentConfig()
    if selected != HumanoidExperimentConfig():
        raise HumanoidContractError("reference corpus environment configuration differs")
    raw_class = _full_contact_class()
    raw = raw_class(render_mode=None, **selected.gym_kwargs())
    raw_spec = copy.deepcopy(gym.spec(selected.env_id))
    raw_spec.max_episode_steps = None
    raw_spec.order_enforce = False
    raw_spec.disable_env_checker = True
    raw_spec.kwargs = {"render_mode": None, **selected.gym_kwargs()}
    raw.spec = raw_spec
    checked = gym.wrappers.PassiveEnvChecker(raw)
    ordered = gym.wrappers.OrderEnforcing(checked)
    registered_spec = gym.spec(selected.env_id)
    if registered_spec.max_episode_steps != 1000:
        raw.close()
        raise HumanoidContractError("Humanoid-v5 TimeLimit differs from 1000")
    return gym.wrappers.TimeLimit(ordered, max_episode_steps=1000)


def wrapper_types(environment: object) -> tuple[str, ...]:
    observed: list[str] = []
    current = environment
    while True:
        observed.append(f"{type(current).__module__}.{type(current).__name__}")
        if not hasattr(current, "env"):
            break
        current = current.env
    return tuple(observed)


def require_reference_wrapper_stack(environment: object) -> None:
    if wrapper_types(environment) != EXPECTED_REFERENCE_WRAPPER_TYPES:
        raise HumanoidContractError("reference corpus wrapper order or type differs")


__all__ = [
    "EXPECTED_REFERENCE_WRAPPER_TYPES",
    "FULL_CONTACT_CAPTURE_ID",
    "FullSubstepContactSample",
    "make_reference_corpus_env",
    "require_reference_wrapper_stack",
    "wrapper_types",
]
