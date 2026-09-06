"""Shared 50 Hz control boundary for the admitted GMT G1 deployment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .actor import load_actor
from .checkpoint import verify_upstream_root
from .contracts import (
    ACTION_DIM,
    OBSERVATION_DIM,
    PROPRIOCEPTION_DIM,
    RAW_ACTION_MAX,
    RAW_ACTION_MIN,
    REFERENCE_FRAME_DIM,
    REFERENCE_HORIZON,
    SIMULATION_DECIMATION,
    SIMULATION_DT_SECONDS,
    TORQUE_LIMITS,
)
from .deployment import ObservationHistory, apply_pd_control, build_proprioception, pd_torque
from .io import GMTAdmissionError
from .replay import ModelABI, _extract_model_abi, validate_model_abi

# This is a conservative development value, not an admitted study authority.
# At deployment scale it permits at most 0.125 rad of additive PD-target shift.
PROVISIONAL_RESIDUAL_RAW_SCALE = np.float32(0.25)
F32 = np.dtype("<f4")
F64 = np.dtype("<f8")


def _exact_array(
    value: object,
    *,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
    field: str,
) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != dtype
        or value.shape != shape
        or not value.flags.c_contiguous
        or not np.isfinite(value).all()
    ):
        raise GMTAdmissionError(f"{field} must be finite C-order {dtype.name}{list(shape)}")
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _nonnegative_integer(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise GMTAdmissionError(f"{field} must be a non-negative integer")
    return value


def _freeze_fields(
    instance: object,
    specifications: tuple[tuple[str, tuple[int, ...], np.dtype[Any]], ...],
) -> None:
    for name, shape, dtype in specifications:
        object.__setattr__(
            instance,
            name,
            _exact_array(getattr(instance, name), shape=shape, dtype=dtype, field=name),
        )


@dataclass(frozen=True, slots=True)
class ControlBoundary:
    """One state after the prior 20 substeps and before the next control."""

    qpos: np.ndarray
    qvel: np.ndarray
    orientation_wxyz: np.ndarray
    angular_velocity: np.ndarray
    control_step: int
    simulation_step: int

    def __post_init__(self) -> None:
        _freeze_fields(
            self,
            (
                ("qpos", (30,), F64),
                ("qvel", (29,), F64),
                ("orientation_wxyz", (4,), F32),
                ("angular_velocity", (3,), F32),
            ),
        )
        _nonnegative_integer(self.control_step, field="control_step")
        _nonnegative_integer(self.simulation_step, field="simulation_step")
        if self.simulation_step != self.control_step * SIMULATION_DECIMATION:
            raise GMTAdmissionError("control and simulation counters differ from the 20:1 cadence")


ContactPair = tuple[int, int]
SubstepContactPairs = tuple[ContactPair, ...]


def _contact_trace(value: object) -> tuple[SubstepContactPairs, ...]:
    if type(value) is not tuple or len(value) != SIMULATION_DECIMATION:
        raise GMTAdmissionError("contact trace must contain exactly 20 substeps")
    checked: list[SubstepContactPairs] = []
    for substep in value:
        if type(substep) is not tuple:
            raise GMTAdmissionError("each contact substep must be an immutable tuple")
        pairs: list[ContactPair] = []
        for pair in substep:
            if (
                type(pair) is not tuple
                or len(pair) != 2
                or any(type(item) is not int or item < 0 for item in pair)
                or pair[0] > pair[1]
            ):
                raise GMTAdmissionError("contact pairs must be ordered non-negative geom IDs")
            pairs.append(pair)
        canonical = tuple(sorted(set(pairs)))
        if tuple(pairs) != canonical:
            raise GMTAdmissionError("contact pairs must be unique and sorted")
        checked.append(canonical)
    return tuple(checked)


@dataclass(frozen=True, slots=True)
class ControlInterval:
    """Complete post-substep evidence for one held 50 Hz PD target."""

    composite_raw_action: np.ndarray
    clipped_action: np.ndarray
    pd_target: np.ndarray
    qpos_after_substep: np.ndarray
    qvel_after_substep: np.ndarray
    torque_by_substep: np.ndarray
    contact_pairs_after_substep: tuple[SubstepContactPairs, ...]
    start_control_step: int
    start_simulation_step: int

    def __post_init__(self) -> None:
        _freeze_fields(
            self,
            (
                ("composite_raw_action", (ACTION_DIM,), F32),
                ("clipped_action", (ACTION_DIM,), F32),
                ("pd_target", (ACTION_DIM,), F64),
                ("qpos_after_substep", (SIMULATION_DECIMATION, 30), F64),
                ("qvel_after_substep", (SIMULATION_DECIMATION, 29), F64),
                ("torque_by_substep", (SIMULATION_DECIMATION, ACTION_DIM), F64),
            ),
        )
        object.__setattr__(
            self,
            "contact_pairs_after_substep",
            _contact_trace(self.contact_pairs_after_substep),
        )
        _nonnegative_integer(self.start_control_step, field="start_control_step")
        _nonnegative_integer(self.start_simulation_step, field="start_simulation_step")
        if self.start_simulation_step != self.start_control_step * SIMULATION_DECIMATION:
            raise GMTAdmissionError("interval counters differ from the 20:1 cadence")

    @property
    def action_saturation_fraction(self) -> float:
        return float(
            np.mean(
                (self.composite_raw_action <= np.float32(RAW_ACTION_MIN))
                | (self.composite_raw_action >= np.float32(RAW_ACTION_MAX))
            )
        )

    @property
    def torque_saturation_fraction(self) -> float:
        limits = np.asarray(TORQUE_LIMITS, dtype="<f8")
        return float(np.mean(np.abs(self.torque_by_substep) >= limits))


def compose_residual_raw_action(
    base_raw_action: np.ndarray,
    normalized_residual_action: np.ndarray,
    *,
    residual_scale: np.float32 = PROVISIONAL_RESIDUAL_RAW_SCALE,
) -> np.ndarray:
    """Compose in GMT pre-clip raw-action units with exact float32 order."""

    base = _exact_array(
        base_raw_action,
        shape=(ACTION_DIM,),
        dtype=F32,
        field="base raw action",
    )
    residual = _exact_array(
        normalized_residual_action,
        shape=(ACTION_DIM,),
        dtype=F32,
        field="normalized residual action",
    )
    if type(residual_scale) is not np.float32 or not np.isfinite(residual_scale):
        raise GMTAdmissionError("residual scale must be one finite float32 scalar")
    if residual_scale < np.float32(0.0):
        raise GMTAdmissionError("residual scale cannot be negative")
    if np.any(residual < np.float32(-1.0)) or np.any(residual > np.float32(1.0)):
        raise GMTAdmissionError("normalized residual action lies outside [-1, 1]")
    return np.ascontiguousarray(
        np.add(
            base,
            np.multiply(residual_scale, residual, dtype=np.float32),
            dtype=np.float32,
        ),
        dtype="<f4",
    )


def _load_mujoco() -> Any:
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - exercised by an authorized real run
        raise RuntimeError("MuJoCo is required only when constructing the G1 runtime") from exc
    return mujoco


class G1ControlRuntime:
    """Pinned G1 plant with one exact 20-substep transition per call."""

    def __init__(self, upstream_root: Path) -> None:
        self.upstream_root = Path(upstream_root)
        self._support_files = verify_upstream_root(self.upstream_root)
        self._mujoco = _load_mujoco()
        xml_path = self.upstream_root / "assets/robots/g1/g1.xml"
        self._model = self._mujoco.MjModel.from_xml_path(str(xml_path))
        self.model_abi: ModelABI = _extract_model_abi(self._mujoco, self._model)
        validate_model_abi(self.model_abi)
        body_name = self._mujoco.mj_id2name
        self._geom_body_names = tuple(
            body_name(
                self._model,
                self._mujoco.mjtObj.mjOBJ_BODY,
                int(body_id),
            )
            or f"body:{int(body_id)}"
            for body_id in self._model.geom_bodyid
        )
        self._model.opt.timestep = SIMULATION_DT_SECONDS
        self._data = self._mujoco.MjData(self._model)
        self._control_step = 0
        self._simulation_step = 0
        self._ready = False

    @property
    def support_files(self) -> dict[str, str]:
        return dict(self._support_files)

    @property
    def geom_body_names(self) -> tuple[str, ...]:
        """Body identity for each geom ID retained in interval contact pairs."""

        return self._geom_body_names

    def _boundary(self) -> ControlBoundary:
        return ControlBoundary(
            qpos=np.ascontiguousarray(self._data.qpos, dtype="<f8"),
            qvel=np.ascontiguousarray(self._data.qvel, dtype="<f8"),
            orientation_wxyz=np.ascontiguousarray(
                self._data.sensor("orientation").data,
                dtype="<f4",
            ),
            angular_velocity=np.ascontiguousarray(
                self._data.sensor("angular-velocity").data,
                dtype="<f4",
            ),
            control_step=self._control_step,
            simulation_step=self._simulation_step,
        )

    def _contact_pairs(self) -> SubstepContactPairs:
        pairs: set[ContactPair] = set()
        for contact in self._data.contact[: int(self._data.ncon)]:
            first, second = int(contact.geom1), int(contact.geom2)
            if not 0 <= first < int(self._model.ngeom) or not 0 <= second < int(self._model.ngeom):
                raise GMTAdmissionError("MuJoCo returned a contact geom outside the model")
            pairs.add((min(first, second), max(first, second)))
        return tuple(sorted(pairs))

    def reset(self) -> ControlBoundary:
        self._mujoco.mj_resetDataKeyframe(self._model, self._data, 0)
        self._mujoco.mj_step(self._model, self._data)
        self._control_step = 0
        self._simulation_step = 0
        self._ready = True
        return self._boundary()

    def step(self, composite_raw_action: np.ndarray) -> tuple[ControlBoundary, ControlInterval]:
        if not self._ready:
            raise GMTAdmissionError("G1 control runtime must be reset before stepping")
        raw = _exact_array(
            composite_raw_action, shape=(ACTION_DIM,), dtype=F32, field="composite raw action"
        )
        dof_position = np.ascontiguousarray(self._data.qpos[-ACTION_DIM:], dtype="<f4")
        dof_velocity = np.ascontiguousarray(self._data.qvel[-ACTION_DIM:], dtype="<f4")
        control = apply_pd_control(
            raw_action=raw,
            dof_position=dof_position,
            dof_velocity=dof_velocity,
        )
        qpos = np.empty((SIMULATION_DECIMATION, 30), dtype="<f8")
        qvel = np.empty((SIMULATION_DECIMATION, 29), dtype="<f8")
        torques = np.empty((SIMULATION_DECIMATION, ACTION_DIM), dtype="<f8")
        contacts: list[SubstepContactPairs] = []
        start_control_step = self._control_step
        start_simulation_step = self._simulation_step
        for substep in range(SIMULATION_DECIMATION):
            dof_position = np.ascontiguousarray(self._data.qpos[-ACTION_DIM:], dtype="<f4")
            dof_velocity = np.ascontiguousarray(self._data.qvel[-ACTION_DIM:], dtype="<f4")
            torque = pd_torque(
                pd_target=control.pd_target,
                dof_position=dof_position,
                dof_velocity=dof_velocity,
            )
            self._data.ctrl[:] = torque
            self._mujoco.mj_step(self._model, self._data)
            qpos[substep] = self._data.qpos
            qvel[substep] = self._data.qvel
            torques[substep] = torque
            contacts.append(self._contact_pairs())
        self._control_step += 1
        self._simulation_step += SIMULATION_DECIMATION
        interval = ControlInterval(
            composite_raw_action=np.ascontiguousarray(raw, dtype="<f4"),
            clipped_action=np.ascontiguousarray(control.clipped_action, dtype="<f4"),
            pd_target=np.ascontiguousarray(control.pd_target, dtype="<f8"),
            qpos_after_substep=qpos,
            qvel_after_substep=qvel,
            torque_by_substep=torques,
            contact_pairs_after_substep=tuple(contacts),
            start_control_step=start_control_step,
            start_simulation_step=start_simulation_step,
        )
        return self._boundary(), interval


@dataclass(frozen=True, slots=True)
class PreparedControl:
    control_step: int
    proprio: np.ndarray
    obs: np.ndarray
    base_raw: np.ndarray

    def __post_init__(self) -> None:
        _nonnegative_integer(self.control_step, field="prepared control_step")
        _freeze_fields(
            self,
            (
                ("proprio", (PROPRIOCEPTION_DIM,), F64),
                ("obs", (OBSERVATION_DIM,), F32),
                ("base_raw", (ACTION_DIM,), F32),
            ),
        )


class GMTActorSession:
    """Episode-local base inference and exact executed-action history.

    Commit only after the plant accepts the composite action: it advances the
    pre-append history with the prepared boundary and that executed pre-clip action.
    """

    def __init__(self, weights_path: Path, *, expected_sha256: str) -> None:
        self._initialize(
            load_actor(Path(weights_path), expected_sha256=expected_sha256, freeze=True)
        )

    @classmethod
    def _from_actor_for_test(cls, actor: torch.nn.Module) -> GMTActorSession:
        instance = cls.__new__(cls)
        instance._initialize(actor)
        return instance

    def _initialize(self, actor: torch.nn.Module) -> None:
        if not isinstance(actor, torch.nn.Module):
            raise GMTAdmissionError("GMT actor session requires one reviewed Torch module")
        actor.eval()
        self._actor = actor
        self.reset()

    @property
    def last_raw_action(self) -> np.ndarray:
        return self._last_raw_action.copy()

    @property
    def history(self) -> np.ndarray:
        return self._history.values

    def reset(self) -> None:
        self._history = ObservationHistory()
        self._last_raw_action = np.zeros(ACTION_DIM, dtype="<f4")
        self._expected_control_step = 0
        self._pending: PreparedControl | None = None

    def prepare(
        self,
        boundary: ControlBoundary,
        reference_window: np.ndarray,
    ) -> PreparedControl:
        if type(boundary) is not ControlBoundary:
            raise GMTAdmissionError("actor preparation requires an exact ControlBoundary")
        if self._pending is not None:
            raise GMTAdmissionError("the prior prepared control must be committed first")
        if boundary.control_step != self._expected_control_step:
            raise GMTAdmissionError("actor and plant control counters differ")
        reference = _exact_array(
            reference_window,
            shape=(REFERENCE_HORIZON, REFERENCE_FRAME_DIM),
            dtype=F32,
            field="reference window",
        )
        proprioception = np.ascontiguousarray(
            build_proprioception(
                dof_position=np.ascontiguousarray(boundary.qpos[-ACTION_DIM:], dtype="<f4"),
                dof_velocity=np.ascontiguousarray(boundary.qvel[-ACTION_DIM:], dtype="<f4"),
                orientation_wxyz=boundary.orientation_wxyz,
                angular_velocity=boundary.angular_velocity,
                last_raw_action=self._last_raw_action,
            ),
            dtype="<f8",
        )
        observation = self._history.assemble(reference, proprioception)
        with torch.inference_mode():
            output = self._actor(torch.from_numpy(observation).unsqueeze(0))
        if (
            type(output) is not torch.Tensor
            or output.device.type != "cpu"
            or output.dtype is not torch.float32
            or tuple(output.shape) != (1, ACTION_DIM)
            or not bool(torch.isfinite(output).all())
        ):
            raise GMTAdmissionError("GMT actor returned a malformed base raw action")
        prepared = PreparedControl(
            control_step=boundary.control_step,
            proprio=proprioception,
            obs=np.ascontiguousarray(observation, dtype="<f4"),
            base_raw=np.ascontiguousarray(output[0].detach().numpy(), dtype="<f4"),
        )
        self._pending = prepared
        return prepared

    def commit(self, prepared: PreparedControl, composite_raw_action: np.ndarray) -> None:
        if self._pending is None or prepared is not self._pending:
            raise GMTAdmissionError("commit does not match this session's pending preparation")
        composite = _exact_array(
            composite_raw_action,
            shape=(ACTION_DIM,),
            dtype=F32,
            field="committed composite raw action",
        )
        self._history.append(prepared.proprio)
        self._last_raw_action = np.array(composite, dtype="<f4", order="C", copy=True)
        self._expected_control_step += 1
        self._pending = None


__all__ = [
    "PROVISIONAL_RESIDUAL_RAW_SCALE",
    "ControlBoundary",
    "ControlInterval",
    "G1ControlRuntime",
    "GMTActorSession",
    "PreparedControl",
    "compose_residual_raw_action",
]
