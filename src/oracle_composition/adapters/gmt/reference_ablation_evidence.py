"""Numeric evidence checks for the closed-loop GMT reference ablation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .contracts import (
    ACTION_DIM,
    OBSERVATION_DIM,
    PROPRIOCEPTION_DIM,
    REFERENCE_FRAME_DIM,
    REFERENCE_HORIZON,
    REFERENCE_SLICE,
)
from .control_runtime import PreparedControl
from .io import GMTAdmissionError
from .reference_ablation import (
    F32,
    JOINT_ANGLE_DIVERGENCE_RAD,
    ROOT_ORIENTATION_DIVERGENCE_RAD,
    ROOT_TRANSLATION_DIVERGENCE_M,
    ReferenceInputAudit,
    _array,
    _same_bytes,
)

F64 = np.dtype("<f8")


def _qpos_trajectory(value: object, *, field: str) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.dtype != F64
        or value.ndim != 2
        or value.shape[0] < 2
        or value.shape[1] != 30
        or not value.flags.c_contiguous
        or not np.isfinite(value).all()
    ):
        raise GMTAdmissionError(f"{field} must be finite C-order float64[N>=2,30]")
    return value


def _orientation_geodesic(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    if np.any(left_norm <= 0.0) or np.any(right_norm <= 0.0):
        raise GMTAdmissionError("root orientation evidence contains a zero quaternion")
    unit_left = left / left_norm[:, None]
    unit_right = right / right_norm[:, None]
    dot = np.clip(np.abs(np.sum(unit_left * unit_right, axis=1)), 0.0, 1.0)
    return 2.0 * np.arccos(dot)


def _divergence_window(
    translation: np.ndarray,
    joints: np.ndarray,
    orientation: np.ndarray,
    *,
    control_steps: int,
) -> dict[str, object]:
    translation = translation[:control_steps]
    joints = joints[:control_steps]
    orientation = orientation[:control_steps]
    crossed = (
        (translation >= ROOT_TRANSLATION_DIVERGENCE_M)
        | (joints >= JOINT_ANGLE_DIVERGENCE_RAD)
        | (orientation >= ROOT_ORIENTATION_DIVERGENCE_RAD)
    )
    indices = np.flatnonzero(crossed)
    return {
        "compared_control_steps": control_steps,
        "maximum_root_translation_m": float(np.max(translation)),
        "maximum_joint_angle_difference_rad": float(np.max(joints)),
        "maximum_root_orientation_geodesic_rad": float(np.max(orientation)),
        "practical_divergence_observed": bool(indices.size),
        "first_practical_divergence_control_step": (
            None if not indices.size else int(indices[0] + 1)
        ),
    }


def measure_practical_trajectory_divergence(
    exact_qpos: np.ndarray,
    treated_qpos: np.ndarray,
) -> dict[str, object]:
    """Compare only the common observed post-reset control boundaries."""

    exact = _qpos_trajectory(exact_qpos, field="exact qpos trajectory")
    treated = _qpos_trajectory(treated_qpos, field="treated qpos trajectory")
    common_steps = min(exact.shape[0], treated.shape[0]) - 1
    exact = exact[1 : common_steps + 1]
    treated = treated[1 : common_steps + 1]
    translation = np.linalg.norm(exact[:, :3] - treated[:, :3], axis=1)
    joints = np.max(np.abs(exact[:, -ACTION_DIM:] - treated[:, -ACTION_DIM:]), axis=1)
    orientation = _orientation_geodesic(exact[:, 3:7], treated[:, 3:7])
    return {
        "comparison": "treated_minus_exact_on_common_observed_control_boundaries",
        "first_50_control_steps": _divergence_window(
            translation,
            joints,
            orientation,
            control_steps=min(common_steps, 50),
        ),
        "full_common_observed_horizon": _divergence_window(
            translation,
            joints,
            orientation,
            control_steps=common_steps,
        ),
        "post_common_observed_horizon": "unavailable",
    }


@dataclass(frozen=True, slots=True)
class ZeroResidualActorEvidence:
    audit: ReferenceInputAudit
    actor_observation: np.ndarray
    prepared_proprio: np.ndarray
    base_raw_action: np.ndarray
    residual_action: np.ndarray
    composite_raw_action: np.ndarray

    def __post_init__(self) -> None:
        if type(self.audit) is not ReferenceInputAudit:
            raise GMTAdmissionError("actor evidence requires one reference-input audit")
        object.__setattr__(
            self,
            "actor_observation",
            _array(
                self.actor_observation,
                shape=(OBSERVATION_DIM,),
                dtype=F32,
                field="actor observation",
            ),
        )
        object.__setattr__(
            self,
            "prepared_proprio",
            _array(
                self.prepared_proprio,
                shape=(PROPRIOCEPTION_DIM,),
                dtype=F64,
                field="prepared proprioception",
            ),
        )
        for name in ("base_raw_action", "residual_action", "composite_raw_action"):
            object.__setattr__(
                self,
                name,
                _array(
                    getattr(self, name),
                    shape=(ACTION_DIM,),
                    dtype=F32,
                    field=name.replace("_", " "),
                ),
            )
        observed_window = self.actor_observation[REFERENCE_SLICE].reshape(
            REFERENCE_HORIZON, REFERENCE_FRAME_DIM
        )
        if not _same_bytes(observed_window, self.audit.actor_reference_window):
            raise GMTAdmissionError("actor observation does not contain the audited reference window")
        if not _same_bytes(self.residual_action, np.zeros(ACTION_DIM, dtype=F32)):
            raise GMTAdmissionError("closed-loop reference ablation requires literal zero residual")
        if not _same_bytes(self.composite_raw_action, self.base_raw_action):
            raise GMTAdmissionError("zero residual did not preserve the base raw action exactly")


def capture_zero_residual_actor_evidence(
    *,
    audit: ReferenceInputAudit,
    prepared: PreparedControl,
    residual_action: np.ndarray,
    composite_raw_action: np.ndarray,
) -> ZeroResidualActorEvidence:
    if type(prepared) is not PreparedControl or prepared.control_step != audit.control_step:
        raise GMTAdmissionError("prepared control and reference audit steps differ")
    return ZeroResidualActorEvidence(
        audit=audit,
        actor_observation=prepared.obs,
        prepared_proprio=prepared.proprio,
        base_raw_action=prepared.base_raw,
        residual_action=residual_action,
        composite_raw_action=composite_raw_action,
    )


def validate_recomputed_base_action(
    evidence: ZeroResidualActorEvidence,
    actor: torch.nn.Module,
) -> None:
    if not isinstance(actor, torch.nn.Module) or actor.training or any(
        parameter.requires_grad for parameter in actor.parameters()
    ):
        raise GMTAdmissionError("actor recomputation requires the frozen evaluation actor")
    observation = np.array(evidence.actor_observation, dtype=F32, order="C", copy=True)
    with torch.inference_mode():
        output = actor(torch.from_numpy(observation).unsqueeze(0))
    if (
        type(output) is not torch.Tensor
        or output.device.type != "cpu"
        or output.dtype is not torch.float32
        or tuple(output.shape) != (1, ACTION_DIM)
        or not bool(torch.isfinite(output).all())
    ):
        raise GMTAdmissionError("actor recomputation returned a malformed base action")
    recomputed = np.ascontiguousarray(output[0].detach().cpu().numpy(), dtype=F32)
    if not _same_bytes(recomputed, evidence.base_raw_action):
        raise GMTAdmissionError("recomputed base action differs from retained actor evidence")


__all__ = [
    "ZeroResidualActorEvidence",
    "capture_zero_residual_actor_evidence",
    "measure_practical_trajectory_divergence",
    "validate_recomputed_base_action",
]
