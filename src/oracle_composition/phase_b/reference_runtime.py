"""Switch-only phase transfer and composed 8 x 45 reference windows."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import (
    array_sha256,
    canonical_json_bytes,
)
from oracle_composition.experiments.reference_corpus_bundle import (
    load_bundle_manifest,
    verify_bound_artifacts,
)
from oracle_composition.experiments.reference_corpus_contract import decode_clip_payload
from oracle_composition.harness.contract import ALLOWED_SIGNALS, OracleMachine
from oracle_composition.tracking.humanoid_reference import HumanoidTrackingState
from oracle_composition.tracking.reward import TrackingRewardConfig, compute_tracking_reward

from .contracts import PHASE_POLICY, PhaseBContractError, PhaseBOracleProgram
from .reward import require_frozen_tracking_reward_config

ERROR_NAMES = (
    "root_height_abs_error_m",
    "root_orientation_error_rad",
    "root_linear_velocity_rmse_m_s",
    "root_angular_velocity_rmse_rad_s",
    "joint_position_rmse_rad",
    "joint_velocity_rmse_rad_s",
)
_ERROR_RESULT_FIELDS = ERROR_NAMES


@dataclass(frozen=True, slots=True)
class LoadedReferenceClip:
    block: int
    behavior: str
    reference_rows: np.ndarray
    boundary_observations: np.ndarray
    bundle_sha256: str
    payload_sha256: str
    reference_identity_sha256: str


@dataclass(frozen=True, slots=True)
class PhaseCandidate:
    phase: int
    normalized_errors: tuple[float, float, float, float, float, float]
    score: tuple[float, float, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "normalized_errors": {
                name: value for name, value in zip(ERROR_NAMES, self.normalized_errors, strict=True)
            },
            "phase": self.phase,
            "score": [self.score[0], self.score[1], self.score[2]],
        }


@dataclass(frozen=True, slots=True)
class PhaseTransferLog:
    source_behavior: str
    target_behavior: str
    task_step: int
    candidate_range: tuple[int, int]
    candidates: tuple[PhaseCandidate, ...]
    selected_phase: int
    selected_normalized_errors: tuple[float, float, float, float, float, float]
    selected_score: tuple[float, float, int]
    target_window_indices: tuple[int, ...]
    selection_reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_range_inclusive": list(self.candidate_range),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "selected_normalized_errors": {
                name: value
                for name, value in zip(ERROR_NAMES, self.selected_normalized_errors, strict=True)
            },
            "selected_phase": self.selected_phase,
            "selected_score": [
                self.selected_score[0],
                self.selected_score[1],
                self.selected_score[2],
            ],
            "selection_reason": self.selection_reason,
            "source_behavior": self.source_behavior,
            "target_behavior": self.target_behavior,
            "target_window_indices": list(self.target_window_indices),
            "task_step": self.task_step,
        }


@dataclass(frozen=True, slots=True)
class ComposedReferenceFrame:
    task_step: int
    oracle_state: str
    behavior: str
    phase: int
    policy_window: np.ndarray
    policy_window_indices: tuple[int, ...]
    policy_window_sha256: str
    hidden_reward_target: np.ndarray
    hidden_reward_target_sha256: str
    reward_target_index: int
    terminal_hold: bool
    machine_reason: str
    transfer: PhaseTransferLog | None


def tracking_state_from_reference_row(row: np.ndarray) -> HumanoidTrackingState:
    """Interpret one validated corpus row as an exact synthetic current state."""

    value = np.asarray(row)
    if (
        type(row) is not np.ndarray
        or row.dtype.str != "<f8"
        or row.shape != (45,)
        or not row.flags.c_contiguous
        or not np.isfinite(row).all()
    ):
        raise PhaseBContractError("reference state row must be finite C-order float64[45]")
    return HumanoidTrackingState(
        root_position_world_m=np.asarray([0.0, 0.0, value[0]], dtype="<f8"),
        root_height_m=float(value[0]),
        root_orientation_wxyz=np.ascontiguousarray(value[1:5], dtype="<f8"),
        root_linear_velocity_world_m_s=np.ascontiguousarray(value[5:8], dtype="<f8"),
        root_angular_velocity_body_rad_s=np.ascontiguousarray(value[8:11], dtype="<f8"),
        joint_positions_rad=np.ascontiguousarray(value[11:28], dtype="<f8"),
        joint_velocities_rad_s=np.ascontiguousarray(value[28:45], dtype="<f8"),
    )


def normalized_tracking_errors(
    state: HumanoidTrackingState,
    reference_row: np.ndarray,
    *,
    config: TrackingRewardConfig | None = None,
) -> tuple[float, float, float, float, float, float]:
    selected = require_frozen_tracking_reward_config(config)
    result = compute_tracking_reward(state=state, reference_frame=reference_row, config=selected)
    raw = result.error_components()
    scales = (
        selected.root_height_scale_m,
        selected.root_orientation_scale_rad,
        selected.root_linear_velocity_scale_m_s,
        selected.root_angular_velocity_scale_rad_s,
        selected.joint_position_scale_rad,
        selected.joint_velocity_scale_rad_s,
    )
    values = tuple(
        float(raw[name]) / float(scale)
        for name, scale in zip(_ERROR_RESULT_FIELDS, scales, strict=True)
    )
    if len(values) != 6 or not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise PhaseBContractError("normalized phase-transfer errors are invalid")
    return values  # type: ignore[return-value]


def select_nearest_phase(
    *,
    state: HumanoidTrackingState,
    target_rows: np.ndarray,
    task_step: int,
    source_behavior: str,
    target_behavior: str,
    reason: str,
    config: TrackingRewardConfig | None = None,
) -> PhaseTransferLog:
    """Select ``argmin(max(e), sum(e^2), j)`` over the exact ``j <= t`` range."""

    if type(task_step) is not int or not 0 <= task_step < len(target_rows):
        raise PhaseBContractError("phase transfer task_step is outside the reference")
    if (
        type(target_rows) is not np.ndarray
        or target_rows.dtype.str != "<f8"
        or target_rows.ndim != 2
        or target_rows.shape[1] != 45
        or not target_rows.flags.c_contiguous
        or not np.isfinite(target_rows).all()
    ):
        raise PhaseBContractError("target reference rows must be finite C-order float64[N,45]")
    candidates: list[PhaseCandidate] = []
    for phase in range(task_step + 1):
        errors = normalized_tracking_errors(state, target_rows[phase], config=config)
        score = (max(errors), sum(error * error for error in errors), phase)
        candidates.append(PhaseCandidate(phase, errors, score))
    selected = min(candidates, key=lambda item: item.score)
    indices = _window_indices(selected.phase, len(target_rows))
    return PhaseTransferLog(
        source_behavior=source_behavior,
        target_behavior=target_behavior,
        task_step=task_step,
        candidate_range=(0, task_step),
        candidates=tuple(candidates),
        selected_phase=selected.phase,
        selected_normalized_errors=selected.normalized_errors,
        selected_score=selected.score,
        target_window_indices=indices,
        selection_reason=reason,
    )


def _window_indices(phase: int, length: int) -> tuple[int, ...]:
    if not 0 <= phase < length:
        raise PhaseBContractError("reference phase lies outside the clip")
    return tuple(min(phase + offset, length - 1) for offset in range(8))


class ComposedReferenceRuntime:
    """Drive one oracle program without continuous phase rematching."""

    def __init__(
        self,
        program: PhaseBOracleProgram,
        references: Mapping[str, np.ndarray],
        *,
        reward_config: TrackingRewardConfig | None = None,
    ) -> None:
        if type(program) is not PhaseBOracleProgram:
            raise PhaseBContractError("runtime requires a Phase B oracle program")
        if type(references) is not dict or set(references) != set(program.program.behaviors):
            raise PhaseBContractError("reference library differs from oracle behaviors")
        checked: dict[str, np.ndarray] = {}
        lengths: set[int] = set()
        for behavior, value in references.items():
            if (
                type(value) is not np.ndarray
                or value.dtype.str != "<f8"
                or value.ndim != 2
                or value.shape[1] != 45
                or value.shape[0] < 8
                or not value.flags.c_contiguous
                or not np.isfinite(value).all()
            ):
                raise PhaseBContractError(f"reference {behavior} must be float64[N,45]")
            frozen = np.array(value, dtype="<f8", order="C", copy=True)
            frozen.setflags(write=False)
            checked[behavior] = frozen
            lengths.add(len(value))
        if len(lengths) != 1:
            raise PhaseBContractError("all behavior references must have one boundary count")
        self._program = program
        self._references = checked
        self._reward_config = require_frozen_tracking_reward_config(reward_config)
        self._machine = OracleMachine(program.program)
        self._phase = 0
        self._task_step = 0
        self._last_behavior = self._machine.behavior
        self._awaiting_advance = False
        self._transfer_logs: list[PhaseTransferLog] = []

    @property
    def phase(self) -> int:
        return self._phase

    @property
    def behavior(self) -> str:
        return self._machine.behavior

    @property
    def dwell(self) -> int:
        return self._machine.dwell

    @property
    def task_step(self) -> int:
        return self._task_step

    @property
    def transfer_logs(self) -> tuple[PhaseTransferLog, ...]:
        return tuple(self._transfer_logs)

    def frame(
        self,
        *,
        state: HumanoidTrackingState,
        signals: Mapping[str, object],
    ) -> ComposedReferenceFrame:
        if self._awaiting_advance:
            raise PhaseBContractError("advance must follow each composed reference frame")
        if set(signals) != ALLOWED_SIGNALS:
            raise PhaseBContractError("oracle signals differ from the Phase A grammar")
        if signals["t"] != self._task_step or signals["dwell"] != self._machine.dwell:
            raise PhaseBContractError("oracle task-step or dwell counter drifted")
        decision = self._machine.decide(signals)
        transfer: PhaseTransferLog | None = None
        selection_event = (
            decision.controller_switched or decision.recovery_entered or decision.recovery_exited
        )
        if selection_event:
            transfer = select_nearest_phase(
                state=state,
                target_rows=self._references[decision.behavior],
                task_step=self._task_step,
                source_behavior=self._last_behavior,
                target_behavior=decision.behavior,
                reason=decision.reason,
                config=self._reward_config,
            )
            self._phase = transfer.selected_phase
            self._transfer_logs.append(transfer)
        self._last_behavior = decision.behavior
        rows = self._references[decision.behavior]
        indices = _window_indices(self._phase, len(rows))
        window64 = np.ascontiguousarray(rows[list(indices)], dtype="<f8")
        window = np.ascontiguousarray(window64, dtype="<f4")
        target_index = min(self._phase + 1, len(rows) - 1)
        target = np.ascontiguousarray(rows[target_index], dtype="<f8")
        window.setflags(write=False)
        target.setflags(write=False)
        self._awaiting_advance = True
        return ComposedReferenceFrame(
            task_step=self._task_step,
            oracle_state=decision.state,
            behavior=decision.behavior,
            phase=self._phase,
            policy_window=window,
            policy_window_indices=indices,
            policy_window_sha256=array_sha256(window),
            hidden_reward_target=target,
            hidden_reward_target_sha256=array_sha256(target),
            reward_target_index=target_index,
            terminal_hold=self._phase == len(rows) - 1,
            machine_reason=decision.reason,
            transfer=transfer,
        )

    def advance(self) -> None:
        if not self._awaiting_advance:
            raise PhaseBContractError("a composed reference frame must precede advance")
        rows = self._references[self._machine.behavior]
        if self._phase < len(rows) - 1:
            self._phase += 1
        self._machine.advance()
        self._task_step += 1
        self._awaiting_advance = False


def load_v2_reference_clip(
    corpus_root: Path,
    *,
    block: int,
    behavior: str,
) -> LoadedReferenceClip:
    """Load one exact v2 bundle and its canonical numeric payload."""

    if type(block) is not int or block <= 0 or behavior not in {"expert", "medium", "simple"}:
        raise PhaseBContractError("reference clip identity is invalid")
    root = Path(corpus_root)
    index_path = root / "corpus_index_v2.json"
    if index_path.is_symlink() or not index_path.is_file():
        raise PhaseBContractError("v2 corpus index is unavailable")
    encoded = index_path.read_bytes()
    try:
        index = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise PhaseBContractError("v2 corpus index is invalid JSON") from exc
    if canonical_json_bytes(index) != encoded:
        raise PhaseBContractError("v2 corpus index is not canonical")
    clip_id = f"corpus-{block}-{behavior}"
    matches = [entry for entry in index.get("clips", ()) if entry.get("clip_id") == clip_id]
    if len(matches) != 1:
        raise PhaseBContractError(f"v2 corpus index does not bind exactly one {clip_id}")
    entry = matches[0]
    bundle_sha256 = entry.get("bundle_manifest_sha256")
    if type(bundle_sha256) is not str:
        raise PhaseBContractError("v2 bundle SHA-256 is missing")
    bundle_path = root / "clips" / clip_id / f"bundle-{bundle_sha256}.json"
    try:
        manifest = load_bundle_manifest(bundle_path, expected_sha256=bundle_sha256)
        verify_bound_artifacts(manifest, artifact_root=root)
    except (OSError, ValueError) as exc:
        raise PhaseBContractError(f"v2 reference bundle failed verification: {exc}") from exc
    core = manifest["core"]
    if core["seed"] != block or core["actor_variant"] != behavior or core["clip_id"] != clip_id:
        raise PhaseBContractError("v2 reference bundle identity differs")
    payload = core["payload"]
    payload_path = root / payload["object_path"]
    if payload_path.is_symlink() or not payload_path.is_file():
        raise PhaseBContractError("v2 reference payload is unavailable")
    payload_bytes = payload_path.read_bytes()
    if (
        len(payload_bytes) != payload["byte_count"]
        or hashlib.sha256(payload_bytes).hexdigest() != payload["sha256"]
    ):
        raise PhaseBContractError("v2 reference payload identity differs")
    try:
        arrays = decode_clip_payload(
            payload_bytes,
            steps=core["steps"],
            plain_comparison=True,
        )
    except ValueError as exc:
        raise PhaseBContractError(f"v2 reference payload failed validation: {exc}") from exc
    return LoadedReferenceClip(
        block=block,
        behavior=behavior,
        reference_rows=np.ascontiguousarray(arrays["reference_rows"], dtype="<f8"),
        boundary_observations=np.ascontiguousarray(arrays["boundary_observation"], dtype="<f8"),
        bundle_sha256=bundle_sha256,
        payload_sha256=payload["sha256"],
        reference_identity_sha256=manifest["reference_identity_sha256"],
    )


def phase_policy_sha256() -> str:
    return hashlib.sha256(canonical_json_bytes(dict(PHASE_POLICY))).hexdigest()


__all__ = [
    "ERROR_NAMES",
    "ComposedReferenceFrame",
    "ComposedReferenceRuntime",
    "LoadedReferenceClip",
    "PhaseCandidate",
    "PhaseTransferLog",
    "load_v2_reference_clip",
    "normalized_tracking_errors",
    "phase_policy_sha256",
    "select_nearest_phase",
    "tracking_state_from_reference_row",
]
