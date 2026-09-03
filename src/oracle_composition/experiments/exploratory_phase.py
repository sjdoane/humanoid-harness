"""Bounded phase and fallback primitives for non-admitted exploration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real

import numpy as np

from .fixed_reference import ExperimentContractError


def _finite(value: object, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ExperimentContractError(f"{field} must be numeric")
    try:
        resolved = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ExperimentContractError(f"{field} must be finite") from exc
    if not math.isfinite(resolved) or (minimum is not None and resolved < minimum):
        raise ExperimentContractError(f"{field} must be finite and at least {minimum}")
    return resolved


def _float64_array(value: object, *, field: str) -> np.ndarray:
    try:
        return np.asarray(value, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ExperimentContractError(f"{field} must be convertible to float64") from exc


def _positive_integer(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ExperimentContractError(f"{field} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class BoundedPhaseConfig:
    horizon_steps: int
    search_radius_frames: int
    anchor_penalty: float
    switch_margin: float
    match_indices: tuple[int, ...]
    max_phase_advance_frames: int = 2
    quaternion_indices: tuple[int, int, int, int] | None = None
    quaternion_angle_scale: float = 1.0
    quaternion_unit_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "horizon_steps",
            _positive_integer(self.horizon_steps, field="horizon_steps"),
        )
        if (
            not isinstance(self.search_radius_frames, int)
            or isinstance(self.search_radius_frames, bool)
            or self.search_radius_frames < 0
        ):
            raise ExperimentContractError("search_radius_frames must be a non-negative integer")
        object.__setattr__(
            self,
            "anchor_penalty",
            _finite(self.anchor_penalty, field="anchor_penalty", minimum=0.0),
        )
        object.__setattr__(
            self,
            "switch_margin",
            _finite(self.switch_margin, field="switch_margin", minimum=0.0),
        )
        indices = tuple(self.match_indices)
        if (
            not indices
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in indices
            )
            or len(indices) != len(set(indices))
        ):
            raise ExperimentContractError("match_indices must be unique non-negative integers")
        object.__setattr__(self, "match_indices", indices)
        object.__setattr__(
            self,
            "max_phase_advance_frames",
            _positive_integer(
                self.max_phase_advance_frames,
                field="max_phase_advance_frames",
            ),
        )
        object.__setattr__(
            self,
            "quaternion_angle_scale",
            _finite(
                self.quaternion_angle_scale,
                field="quaternion_angle_scale",
                minimum=0.0,
            ),
        )
        if self.quaternion_angle_scale <= 0.0:
            raise ExperimentContractError("quaternion_angle_scale must be positive")
        object.__setattr__(
            self,
            "quaternion_unit_tolerance",
            _finite(
                self.quaternion_unit_tolerance,
                field="quaternion_unit_tolerance",
                minimum=0.0,
            ),
        )
        if not 0.0 < self.quaternion_unit_tolerance <= 1e-3:
            raise ExperimentContractError(
                "quaternion_unit_tolerance must be positive and at most 1e-3"
            )
        if self.quaternion_indices is not None:
            quaternion_indices = tuple(self.quaternion_indices)
            if (
                len(quaternion_indices) != 4
                or len(set(quaternion_indices)) != 4
                or any(
                    not isinstance(value, int) or isinstance(value, bool) or value < 0
                    for value in quaternion_indices
                )
            ):
                raise ExperimentContractError(
                    "quaternion_indices must contain four unique non-negative integers"
                )
            if not set(quaternion_indices).issubset(indices):
                raise ExperimentContractError(
                    "quaternion_indices must be included in match_indices"
                )
            object.__setattr__(self, "quaternion_indices", quaternion_indices)


@dataclass(frozen=True, slots=True)
class PhaseDecision:
    nominal_phase: int
    previous_selected_phase: int | None
    baseline_phase: int
    selected_phase: int
    best_candidate_phase: int
    selected_pose_error: float
    best_candidate_pose_error: float
    score_gain: float
    candidate_phase_bounds: tuple[int, int]

    @property
    def applied_offset_frames(self) -> int:
        return self.selected_phase - self.nominal_phase

    @property
    def corrected(self) -> bool:
        return self.selected_phase != self.nominal_phase

    @property
    def phase_advance_frames(self) -> int | None:
        if self.previous_selected_phase is None:
            return None
        return self.selected_phase - self.previous_selected_phase


class BoundedPhaseMatcher:
    """Match state near elapsed time without recursive phase drift."""

    def __init__(
        self,
        *,
        reference: np.ndarray,
        scale: np.ndarray,
        config: BoundedPhaseConfig,
    ) -> None:
        if not isinstance(config, BoundedPhaseConfig):
            raise ExperimentContractError("config must be a BoundedPhaseConfig")
        values = _float64_array(reference, field="reference")
        scales = _float64_array(scale, field="scale")
        if values.ndim != 2 or values.shape[0] < config.horizon_steps:
            raise ExperimentContractError("reference must be a 2D array at least one horizon long")
        if scales.shape != (values.shape[1],):
            raise ExperimentContractError("scale must match the reference width")
        if not np.isfinite(values).all() or not np.isfinite(scales).all():
            raise ExperimentContractError("reference and scale must be finite")
        if np.any(scales <= 0.0):
            raise ExperimentContractError("scale entries must be positive")
        if max(config.match_indices) >= values.shape[1]:
            raise ExperimentContractError("match_indices exceed the reference width")
        quaternion_indices = config.quaternion_indices
        if quaternion_indices is not None:
            quaternion_columns = np.asarray(quaternion_indices, dtype=np.int64)
            quaternion_norms = np.linalg.norm(values[:, quaternion_columns], axis=1)
            if np.any(np.abs(quaternion_norms - 1.0) > config.quaternion_unit_tolerance):
                raise ExperimentContractError(
                    "reference quaternions must be unit length within quaternion_unit_tolerance"
                )
        self._reference = values.copy()
        self._scale = scales.copy()
        self._config = config
        # Every action may target the next reference frame. Near the end, the
        # controller window repeats the terminal frame instead of freezing an
        # earlier full-window start.
        self._last_start = max(
            0,
            values.shape[0] - (2 if config.horizon_steps >= 2 else 1),
        )
        quaternion_index_set = set(quaternion_indices or ())
        self._scalar_match_indices = np.asarray(
            [index for index in config.match_indices if index not in quaternion_index_set],
            dtype=np.int64,
        )
        self._quaternion_indices = (
            None if quaternion_indices is None else np.asarray(quaternion_indices, dtype=np.int64)
        )

    @property
    def config(self) -> BoundedPhaseConfig:
        return self._config

    @property
    def last_start(self) -> int:
        return self._last_start

    def select(
        self,
        current: np.ndarray,
        *,
        nominal_phase: int,
        previous_selected_phase: int | None = None,
    ) -> PhaseDecision:
        if not isinstance(nominal_phase, int) or isinstance(nominal_phase, bool):
            raise ExperimentContractError("nominal_phase must be an integer")
        if nominal_phase < 0:
            raise ExperimentContractError("nominal_phase must be non-negative")
        if previous_selected_phase is not None and (
            not isinstance(previous_selected_phase, int)
            or isinstance(previous_selected_phase, bool)
            or not 0 <= previous_selected_phase <= self._last_start
        ):
            raise ExperimentContractError(
                "previous_selected_phase must be a valid window-start index"
            )
        current_values = _float64_array(current, field="current state")
        if (
            current_values.shape != (self._reference.shape[1],)
            or not np.isfinite(current_values).all()
        ):
            raise ExperimentContractError("current state must be one finite reference-width row")
        if self._quaternion_indices is not None:
            quaternion_norm = float(np.linalg.norm(current_values[self._quaternion_indices]))
            if abs(quaternion_norm - 1.0) > self.config.quaternion_unit_tolerance:
                raise ExperimentContractError(
                    "current quaternion must be unit length within quaternion_unit_tolerance"
                )

        nominal = min(nominal_phase, self._last_start)
        radius = self.config.search_radius_frames
        candidate_start = max(0, nominal - radius)
        candidate_stop = min(self._last_start, nominal + radius)
        if previous_selected_phase is not None:
            candidate_start = max(candidate_start, previous_selected_phase)
            candidate_stop = min(
                candidate_stop,
                previous_selected_phase + self.config.max_phase_advance_frames,
            )
        if candidate_start > candidate_stop:
            raise ExperimentContractError(
                "nominal phase leash and previous-phase continuity bound do not overlap"
            )
        candidates = np.arange(
            candidate_start,
            candidate_stop + 1,
            dtype=np.int64,
        )
        component_count = len(self._scalar_match_indices)
        pose_error_sum = np.zeros(len(candidates), dtype=np.float64)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            if component_count:
                scalar_errors = (
                    self._reference[candidates][:, self._scalar_match_indices]
                    - current_values[self._scalar_match_indices]
                ) / self._scale[self._scalar_match_indices]
                pose_error_sum += np.square(scalar_errors).sum(axis=1)
            if self._quaternion_indices is not None:
                reference_quaternions = self._reference[candidates][:, self._quaternion_indices]
                reference_quaternions = reference_quaternions / np.linalg.norm(
                    reference_quaternions,
                    axis=1,
                    keepdims=True,
                )
                current_quaternion = current_values[self._quaternion_indices]
                current_quaternion = current_quaternion / np.linalg.norm(current_quaternion)
                dots = np.abs(reference_quaternions @ current_quaternion)
                angles = 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))
                pose_error_sum += np.square(angles / self.config.quaternion_angle_scale)
                component_count += 1
            pose_errors = pose_error_sum / component_count
            scores = pose_errors + self.config.anchor_penalty * np.square(candidates - nominal)
        if not np.isfinite(pose_errors).all() or not np.isfinite(scores).all():
            raise ExperimentContractError("derived phase errors and scores must be finite")
        best_position = int(np.argmin(scores))
        baseline_position = int(np.argmin(np.abs(candidates - nominal)))
        gain = float(scores[baseline_position] - scores[best_position])
        if not math.isfinite(gain):
            raise ExperimentContractError("derived phase score gain must be finite")
        selected_position = (
            best_position
            if best_position != baseline_position and gain > self.config.switch_margin
            else baseline_position
        )
        return PhaseDecision(
            nominal_phase=nominal,
            previous_selected_phase=previous_selected_phase,
            baseline_phase=int(candidates[baseline_position]),
            selected_phase=int(candidates[selected_position]),
            best_candidate_phase=int(candidates[best_position]),
            selected_pose_error=float(pose_errors[selected_position]),
            best_candidate_pose_error=float(pose_errors[best_position]),
            score_gain=gain,
            candidate_phase_bounds=(int(candidates[0]), int(candidates[-1])),
        )

    def window(self, phase: int) -> np.ndarray:
        if (
            not isinstance(phase, int)
            or isinstance(phase, bool)
            or not 0 <= phase <= self.last_start
        ):
            raise ExperimentContractError("phase is outside the valid window-start range")
        indices = np.minimum(
            np.arange(phase, phase + self.config.horizon_steps, dtype=np.int64),
            self._reference.shape[0] - 1,
        )
        return self._reference[indices].copy()


class RecoveryMode(StrEnum):
    TRACK = "track"
    RECOVER = "recover"


@dataclass(frozen=True, slots=True)
class RecoveryGateConfig:
    entry_height_min: float
    entry_height_max: float
    entry_up_z_min: float
    entry_pose_error_max: float
    exit_height_min: float
    exit_height_max: float
    exit_up_z_min: float
    exit_pose_error_max: float
    entry_pose_error_dwell_steps: int
    exit_stable_steps: int

    def __post_init__(self) -> None:
        fields = (
            "entry_height_min",
            "entry_height_max",
            "entry_up_z_min",
            "entry_pose_error_max",
            "exit_height_min",
            "exit_height_max",
            "exit_up_z_min",
            "exit_pose_error_max",
        )
        for field in fields:
            object.__setattr__(self, field, _finite(getattr(self, field), field=field))
        object.__setattr__(
            self,
            "entry_pose_error_dwell_steps",
            _positive_integer(
                self.entry_pose_error_dwell_steps,
                field="entry_pose_error_dwell_steps",
            ),
        )
        object.__setattr__(
            self,
            "exit_stable_steps",
            _positive_integer(self.exit_stable_steps, field="exit_stable_steps"),
        )
        if not self.entry_height_min < self.entry_height_max:
            raise ExperimentContractError("entry height bounds must be ordered")
        if not self.exit_height_min < self.exit_height_max:
            raise ExperimentContractError("exit height bounds must be ordered")
        if self.entry_height_min < 0.0 or self.exit_height_min < 0.0:
            raise ExperimentContractError("height lower bounds must be non-negative")
        if self.entry_pose_error_max < 0.0 or self.exit_pose_error_max < 0.0:
            raise ExperimentContractError("pose-error thresholds must be non-negative")
        if not -1.0 <= self.entry_up_z_min <= 1.0 or not -1.0 <= self.exit_up_z_min <= 1.0:
            raise ExperimentContractError("torso up-z thresholds must be in [-1, 1]")
        if not (
            self.entry_height_min < self.exit_height_min
            and self.exit_height_max < self.entry_height_max
            and self.entry_up_z_min < self.exit_up_z_min
            and self.exit_pose_error_max < self.entry_pose_error_max
        ):
            raise ExperimentContractError("exit thresholds must define recovery hysteresis")


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    mode: RecoveryMode
    residual_weight: float
    selected_phase: int
    selected_pose_error: float
    stable_steps: int
    pose_error_bad_steps: int
    entered: bool
    exited: bool
    entry_reasons: tuple[str, ...]


class RecoveryGate:
    """Apply explicit entry/exit hysteresis to one selected phase."""

    def __init__(self, config: RecoveryGateConfig) -> None:
        self._config = config
        self._mode = RecoveryMode.TRACK
        self._stable_steps = 0
        self._pose_error_bad_steps = 0

    def reset(self) -> None:
        self._mode = RecoveryMode.TRACK
        self._stable_steps = 0
        self._pose_error_bad_steps = 0

    def step(
        self,
        *,
        root_height_m: float,
        torso_up_z: float,
        phase_decision: PhaseDecision,
    ) -> RecoveryDecision:
        if not isinstance(phase_decision, PhaseDecision):
            raise ExperimentContractError("phase_decision must be a PhaseDecision")
        height = _finite(root_height_m, field="root_height_m")
        up_z = _finite(torso_up_z, field="torso_up_z")
        if not -1.0 <= up_z <= 1.0:
            raise ExperimentContractError("torso_up_z must be in [-1, 1]")
        pose_error = _finite(
            phase_decision.selected_pose_error,
            field="phase_decision.selected_pose_error",
            minimum=0.0,
        )
        hard_reasons: list[str] = []
        if height < self._config.entry_height_min or height > self._config.entry_height_max:
            hard_reasons.append("height")
        if up_z < self._config.entry_up_z_min:
            hard_reasons.append("torso_up_z")

        reasons: list[str] = []
        if self._mode is RecoveryMode.TRACK:
            self._pose_error_bad_steps = (
                self._pose_error_bad_steps + 1
                if pose_error > self._config.entry_pose_error_max
                else 0
            )
            reasons.extend(hard_reasons)
            if self._pose_error_bad_steps >= self._config.entry_pose_error_dwell_steps:
                reasons.append("pose_error")
        else:
            self._pose_error_bad_steps = 0

        entered = self._mode is RecoveryMode.TRACK and bool(reasons)
        exited = False
        if entered:
            self._mode = RecoveryMode.RECOVER
            self._stable_steps = 0
        if self._mode is RecoveryMode.RECOVER:
            stable = (
                self._config.exit_height_min <= height <= self._config.exit_height_max
                and up_z >= self._config.exit_up_z_min
                and pose_error <= self._config.exit_pose_error_max
            )
            self._stable_steps = self._stable_steps + 1 if stable else 0
            if self._stable_steps >= self._config.exit_stable_steps:
                self._mode = RecoveryMode.TRACK
                self._stable_steps = 0
                exited = True

        return RecoveryDecision(
            mode=self._mode,
            residual_weight=0.0 if self._mode is RecoveryMode.RECOVER else 1.0,
            selected_phase=phase_decision.selected_phase,
            selected_pose_error=pose_error,
            stable_steps=self._stable_steps,
            pose_error_bad_steps=self._pose_error_bad_steps,
            entered=entered,
            exited=exited,
            entry_reasons=tuple(reasons) if entered else (),
        )


__all__ = [
    "BoundedPhaseConfig",
    "BoundedPhaseMatcher",
    "PhaseDecision",
    "RecoveryDecision",
    "RecoveryGate",
    "RecoveryGateConfig",
    "RecoveryMode",
]
