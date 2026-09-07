"""Fixed closed-loop interventions on the GMT actor reference input."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from oracle_composition.harness.contract import OracleMachine, OracleProgram

from .composition import ComposedReference, ReferenceCommand, ReferenceSegment
from .contracts import (
    CONTROL_DT_SECONDS,
    REFERENCE_FRAME_DIM,
    REFERENCE_HORIZON,
    REFERENCE_OFFSETS,
    REFERENCE_SLICE,
)
from .io import GMTAdmissionError

REFERENCE_ABLATION_CONTRACT_ID = "gmt_g1_closed_loop_actor_reference_ablation/v1"
REFERENCE_ABLATION_ARTIFACT = "gmt_g1_closed_loop_reference_ablation"
REFERENCE_ABLATION_EVIDENCE_CLASS = "development_closed_loop_reference_input_effect"
REFERENCE_ABLATION_ARMS = (
    "exact",
    "zero_reference",
    "current_frame_repeated",
    "shuffled_reference",
    "shifted_reference",
)
SHUFFLE_RNG = "numpy.random.PCG64"
SHUFFLE_SEED = 20260906
SHIFT_CONTROL_STEPS = 250
SHIFT_SECONDS = SHIFT_CONTROL_STEPS * CONTROL_DT_SECONDS
ROOT_TRANSLATION_DIVERGENCE_M = 0.001
JOINT_ANGLE_DIVERGENCE_RAD = 0.001
ROOT_ORIENTATION_DIVERGENCE_RAD = 0.001

F32 = np.dtype("<f4")
I64 = np.dtype("<i8")


def _array(
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


def _same_bytes(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.dtype == right.dtype
        and left.shape == right.shape
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _command_copy(value: ReferenceCommand) -> ReferenceCommand:
    if type(value) is not ReferenceCommand:
        raise GMTAdmissionError("reference ablation requires an exact ReferenceCommand")
    if (
        type(value.state) is not str
        or not value.state
        or type(value.behavior) is not str
        or not value.behavior
        or type(value.phase_seconds) is not float
        or type(value.phase_fraction) is not float
        or not np.isfinite([value.phase_seconds, value.phase_fraction]).all()
        or not 0.0 <= value.phase_fraction <= 1.0
        or type(value.segment_sha256) is not str
        or len(value.segment_sha256) != 64
        or any(character not in "0123456789abcdef" for character in value.segment_sha256)
        or (value.transition is not None and not isinstance(value.transition, Mapping))
    ):
        raise GMTAdmissionError("original reference command is malformed")
    transition = (
        None
        if value.transition is None
        else MappingProxyType(copy.deepcopy(dict(value.transition)))
    )
    return ReferenceCommand(
        state=value.state,
        behavior=value.behavior,
        current=_array(
            value.current,
            shape=(REFERENCE_FRAME_DIM,),
            dtype=F32,
            field="original command current",
        ),
        window=_array(
            value.window,
            shape=(REFERENCE_HORIZON, REFERENCE_FRAME_DIM),
            dtype=F32,
            field="original command window",
        ),
        phase_seconds=value.phase_seconds,
        phase_fraction=value.phase_fraction,
        segment_sha256=value.segment_sha256,
        transition=transition,
    )


def fixed_shuffle_permutation() -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(SHUFFLE_SEED))
    result = np.ascontiguousarray(
        generator.permutation(REFERENCE_HORIZON).astype(I64, copy=False)
    )
    result.setflags(write=False)
    return result


def reference_ablation_contract() -> dict[str, object]:
    return {
        "contract_id": REFERENCE_ABLATION_CONTRACT_ID,
        "artifact": REFERENCE_ABLATION_ARTIFACT,
        "evidence_class": REFERENCE_ABLATION_EVIDENCE_CLASS,
        "arms": list(REFERENCE_ABLATION_ARMS),
        "reference_slice": [REFERENCE_SLICE.start, REFERENCE_SLICE.stop],
        "shuffle": {
            "rng": SHUFFLE_RNG,
            "seed": SHUFFLE_SEED,
            "permutation": fixed_shuffle_permutation().tolist(),
        },
        "shift_control_steps": SHIFT_CONTROL_STEPS,
        "shift_seconds": SHIFT_SECONDS,
        "residual_action": "literal_positive_zero_float32_23",
        "objective_target": "unmodified_composed_reference_current_after_step",
        "episode": {
            "runtime": "fresh_plant_actor_oracle_per_arm",
            "stop": "task_horizon_or_first_fall",
        },
        "reported_reference_errors": {
            "objective": "original_poststep_target",
            "actor_window_first_row": "supplemental_offset_or_permuted_target",
        },
        "practical_trajectory_divergence": {
            "root_translation_m": ROOT_TRANSLATION_DIVERGENCE_M,
            "joint_angle_rad": JOINT_ANGLE_DIVERGENCE_RAD,
            "root_orientation_geodesic_rad": ROOT_ORIENTATION_DIVERGENCE_RAD,
            "report_windows_control_steps": [50, "full_common_observed_horizon"],
            "post_fall_comparison": "unavailable",
        },
    }


def actor_reference_window(
    arm: str,
    original: ReferenceCommand,
    segments: Mapping[str, ReferenceSegment],
) -> np.ndarray:
    """Return the only direct treatment applied to one actor input."""

    if arm not in REFERENCE_ABLATION_ARMS:
        raise GMTAdmissionError("reference ablation arm differs from the fixed contract")
    command = _command_copy(original)
    if arm == "exact":
        result = command.window.copy()
    elif arm == "zero_reference":
        result = np.zeros((REFERENCE_HORIZON, REFERENCE_FRAME_DIM), dtype=F32)
    elif arm == "current_frame_repeated":
        result = np.repeat(command.current.reshape(1, -1), REFERENCE_HORIZON, axis=0)
    elif arm == "shuffled_reference":
        result = np.ascontiguousarray(command.window[fixed_shuffle_permutation()], dtype=F32)
    else:
        segment = segments.get(command.behavior)
        if type(segment) is not ReferenceSegment or segment.sha256 != command.segment_sha256:
            raise GMTAdmissionError("active segment identity differs from the original command")
        phase = torch.tensor(command.phase_seconds, dtype=torch.float32)
        offsets = torch.tensor(REFERENCE_OFFSETS, dtype=torch.float32) * CONTROL_DT_SECONDS
        shifted = segment.features(phase + offsets + SHIFT_SECONDS).detach().cpu().numpy()
        result = np.ascontiguousarray(shifted, dtype=F32)
    return _array(
        result,
        shape=(REFERENCE_HORIZON, REFERENCE_FRAME_DIM),
        dtype=F32,
        field="actor reference window",
    )


@dataclass(frozen=True, slots=True)
class ReferenceInputAudit:
    arm: str
    control_step: int
    original_command: ReferenceCommand
    actor_reference_window: np.ndarray

    def __post_init__(self) -> None:
        if self.arm not in REFERENCE_ABLATION_ARMS:
            raise GMTAdmissionError("reference ablation arm differs from the fixed contract")
        if type(self.control_step) is not int or self.control_step < 0:
            raise GMTAdmissionError("reference audit control step must be a non-negative integer")
        object.__setattr__(self, "original_command", _command_copy(self.original_command))
        object.__setattr__(
            self,
            "actor_reference_window",
            _array(
                self.actor_reference_window,
                shape=(REFERENCE_HORIZON, REFERENCE_FRAME_DIM),
                dtype=F32,
                field="audited actor reference window",
            ),
        )


def validate_reference_input_audit(
    audit: ReferenceInputAudit,
    segments: Mapping[str, ReferenceSegment],
) -> None:
    expected = actor_reference_window(audit.arm, audit.original_command, segments)
    if not _same_bytes(expected, audit.actor_reference_window):
        raise GMTAdmissionError("audited actor reference window differs from its intervention")


class AblatedComposedReference:
    """Preserve oracle decisions while changing only the base actor's window."""

    def __init__(self, oracle: ComposedReference, *, arm: str) -> None:
        if type(oracle) is not ComposedReference:
            raise GMTAdmissionError("reference ablation requires the reviewed composed runtime")
        if arm not in REFERENCE_ABLATION_ARMS:
            raise GMTAdmissionError("reference ablation arm differs from the fixed contract")
        self._oracle = oracle
        self.arm = arm
        self._last_audit: ReferenceInputAudit | None = None

    @property
    def program(self) -> OracleProgram:
        return self._oracle.program

    @property
    def segments(self) -> Mapping[str, ReferenceSegment]:
        return self._oracle.segments

    @property
    def machine(self) -> OracleMachine:
        return self._oracle.machine

    @property
    def last_audit(self) -> ReferenceInputAudit:
        if self._last_audit is None:
            raise GMTAdmissionError("no reference command has been audited")
        return self._last_audit

    def reset(self) -> None:
        self._oracle.reset()
        self._last_audit = None

    def current_after_step(self, step: int) -> np.ndarray:
        return self._oracle.current_after_step(step)

    def command(
        self, *, step: int, signals: Mapping[str, float], robot_pose: np.ndarray
    ) -> ReferenceCommand:
        original = self._oracle.command(step=step, signals=signals, robot_pose=robot_pose)
        actor_window = actor_reference_window(self.arm, original, self.segments)
        audit = ReferenceInputAudit(self.arm, step, original, actor_window)
        validate_reference_input_audit(audit, self.segments)
        self._last_audit = audit
        return replace(original, window=actor_window)


__all__ = [
    "JOINT_ANGLE_DIVERGENCE_RAD",
    "REFERENCE_ABLATION_ARMS",
    "REFERENCE_ABLATION_ARTIFACT",
    "REFERENCE_ABLATION_CONTRACT_ID",
    "REFERENCE_ABLATION_EVIDENCE_CLASS",
    "ROOT_ORIENTATION_DIVERGENCE_RAD",
    "ROOT_TRANSLATION_DIVERGENCE_M",
    "SHIFT_CONTROL_STEPS",
    "SHIFT_SECONDS",
    "SHUFFLE_RNG",
    "SHUFFLE_SEED",
    "AblatedComposedReference",
    "ReferenceInputAudit",
    "actor_reference_window",
    "fixed_shuffle_permutation",
    "reference_ablation_contract",
    "validate_reference_input_audit",
]
