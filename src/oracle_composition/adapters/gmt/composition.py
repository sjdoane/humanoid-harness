"""State-triggered, native-cadence reference segments for the fixed GMT actor."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import torch

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.harness.contract import OracleMachine, OracleProgram

from .contracts import CONTROL_DT_SECONDS, REFERENCE_OFFSETS
from .reference_runtime import ReferenceMotion

COMPOSITION_RUNTIME_ID = "gmt_state_triggered_segment_entry_and_boundary/v2"
# Pose matching is a transfer heuristic, not the independent task evaluator.
POSE_SCALES = np.asarray([0.15, 0.35, 0.35] + [0.35] * 23, dtype=np.float64)
POSE_COLUMNS = [0, 1, 2, *range(7, 30)]


@dataclass(frozen=True)
class ReferenceSegment:
    motion: ReferenceMotion
    parent_sha256: str
    start_seconds: float
    end_seconds: float
    entry_phase_end_seconds: float | None = None
    boundary: str = "wrap_within_segment"

    def __post_init__(self) -> None:
        if (
            len(self.parent_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.parent_sha256)
            or not np.isfinite([self.start_seconds, self.end_seconds]).all()
            or not 0 <= self.start_seconds < self.end_seconds <= float(self.motion.duration)
            or self.duration < CONTROL_DT_SECONDS
        ):
            raise ValueError("reference segment bounds or parent identity differ")
        if self.boundary not in {"wrap_within_segment", "hold_last_pose_zero_velocity"}:
            raise ValueError("segment boundary must declare wrap or stationary terminal hold")
        if self.entry_phase_end_seconds is not None and (
            type(self.entry_phase_end_seconds) is not float
            or not np.isfinite(self.entry_phase_end_seconds)
            or not 0.0 <= self.entry_phase_end_seconds < self.duration
        ):
            raise ValueError("entry phase must lie within the segment")

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds

    @property
    def identity(self) -> dict[str, object]:
        result = {
            "parent_motion_sha256": self.parent_sha256,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "cadence": "unchanged_native_fps",
            "boundary": self.boundary,
            "certification": "kinematic_candidate_not_dynamics_certified",
        }
        if self.entry_phase_end_seconds is not None:
            result["entry_phase_end_seconds"] = self.entry_phase_end_seconds
        return result

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.identity)).hexdigest()

    def features(self, phase_seconds: torch.Tensor) -> torch.Tensor:
        if self.boundary == "hold_last_pose_zero_velocity":
            # The endpoint is exclusive; do not wrap a full parent at its duration.
            last = float(np.nextafter(np.float32(self.end_seconds), np.float32(-np.inf)))
            times = torch.clamp(self.start_seconds + phase_seconds, self.start_seconds, last)
            features = self.motion.features(times)
            features[phase_seconds >= self.duration, 3:7] = 0.0
            return features
        if self.start_seconds == 0 and self.end_seconds == float(self.motion.duration):
            # Preserve the original full-clip interpolation's exact arithmetic.
            return self.motion.features(phase_seconds)
        times = self.start_seconds + torch.remainder(phase_seconds, self.duration)
        return self.motion.features(times)

    def nearest_phase(self, robot_pose: np.ndarray) -> tuple[float, float]:
        pose = np.asarray(robot_pose, dtype=np.float64)
        if pose.shape != (26,) or not np.isfinite(pose).all():
            raise ValueError("phase transfer requires finite height, roll/pitch and 23 joints")
        times = torch.arange(0, self.duration, 1.0 / float(self.motion.fps))
        if self.entry_phase_end_seconds is not None:
            times = times[times <= self.entry_phase_end_seconds]
        candidates = self.features(times).numpy()[:, POSE_COLUMNS].astype(np.float64)
        scores = np.mean(((candidates - pose) / POSE_SCALES) ** 2, axis=1)
        selected = int(np.argmin(scores))
        return float(times[selected]), float(scores[selected])

    def reported_phase(self, phase: torch.Tensor) -> float:
        if self.boundary == "hold_last_pose_zero_velocity":
            return float(phase.clamp(0, self.duration))
        return float(phase.remainder(self.duration))


@dataclass(frozen=True)
class ReferenceCommand:
    state: str
    behavior: str
    current: np.ndarray
    window: np.ndarray
    phase_seconds: float
    phase_fraction: float
    segment_sha256: str
    transition: Mapping[str, object] | None


class ComposedReference:
    """One state-machine decision per control boundary, with explicit phase transfer."""

    def __init__(self, program: OracleProgram, segments: Mapping[str, ReferenceSegment]) -> None:
        if set(program.behaviors) != set(segments):
            raise ValueError("oracle behaviors and admitted segments must agree exactly")
        self.program = program
        self.segments = dict(segments)
        self.reset()

    def reset(self) -> None:
        self.machine = OracleMachine(self.program)
        self._last_step = -1
        self._phase_origin_step = 0
        self._phase_origin_seconds = 0.0

    def _phase(self, step: int) -> torch.Tensor:
        return torch.tensor(
            self._phase_origin_seconds + (step - self._phase_origin_step) * CONTROL_DT_SECONDS,
            dtype=torch.float32,
        )

    def current_after_step(self, step: int) -> np.ndarray:
        """Score the executed interval before selecting a new oracle target."""
        if step != self._last_step + 1:
            raise ValueError("objective target must be the next control boundary")
        segment = self.segments[self.machine.behavior]
        return segment.features(self._phase(step).reshape(1))[0].numpy().copy()

    def command(
        self, *, step: int, signals: Mapping[str, float], robot_pose: np.ndarray
    ) -> ReferenceCommand:
        if type(step) is not int or step != self._last_step + 1:
            raise ValueError("oracle commands must be consecutive, beginning at zero")
        if "dwell" in signals:
            raise ValueError("dwell is owned by the deterministic oracle machine")
        previous_state, previous_behavior = self.machine.state, self.machine.behavior
        decision = self.machine.decide({**signals, "dwell": self.machine.dwell})
        segment = self.segments[decision.behavior]
        transition = None
        if decision.controller_switched:
            selected, score = segment.nearest_phase(robot_pose)
            self._phase_origin_seconds, self._phase_origin_step = selected, step
            transition = {
                "control_step": step,
                "from_state": previous_state,
                "to_state": decision.state,
                "from_behavior": previous_behavior,
                "to_behavior": decision.behavior,
                "selected_phase_seconds": selected,
                "normalized_pose_distance": score,
                "rule": "nearest_native_phase_first_tie",
                "segment_sha256": segment.sha256,
            }
        phase = self._phase(step)
        offsets = torch.tensor(REFERENCE_OFFSETS, dtype=torch.float32) * CONTROL_DT_SECONDS
        current = segment.features(phase.reshape(1))[0].numpy().copy()
        window = segment.features(phase + offsets).numpy().copy()
        self._last_step = step
        self.machine.advance()
        return ReferenceCommand(
            decision.state,
            decision.behavior,
            current,
            window,
            segment.reported_phase(phase),
            segment.reported_phase(phase) / segment.duration,
            segment.sha256,
            transition,
        )
