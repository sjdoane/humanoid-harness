"""Protected hold/transition utility evaluation for stored Phase B policies."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.envs.reference_corpus import make_reference_corpus_env
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.tracking.humanoid_reference import (
    tracking_state,
    tracking_state_with_bounded_reset_orientation,
    validate_humanoid_actuator_abi,
)

from .calibration import TaskSuccessCalibration, load_calibration_receipt
from .persistence import LoadedFullCheckpoint, load_full_checkpoint
from .policy import FullAuthorityPolicy, compose_policy_input, load_full_authority_actor
from .protected_metrics import (
    evaluator_mass_center_x_m,
    evaluator_state_record,
    protected_step_record,
    protected_trace_sha256,
    recompute_protected_episode,
    select_evaluator_nearest_phase,
)
from .reference_runtime import load_v2_reference_clip
from .report_v2 import ProtectedEpisodeMetrics

EVALUATION_BLOCKS = tuple(range(120101, 120121))
UTILITY_EVALUATOR_ID = "humanoid_phase_b_protected_utility_evaluator/v1"
HORIZON = 1_000
_FOOT_GEOMS = {"left_foot", "right_foot"}


@dataclass(frozen=True, slots=True)
class UtilityEvaluationResult:
    calibration: TaskSuccessCalibration | None
    checkpoint: LoadedFullCheckpoint
    trained_episodes: tuple[ProtectedEpisodeMetrics, ...]
    step_zero_episodes: tuple[ProtectedEpisodeMetrics, ...]
    trained_traces: tuple[Mapping[str, object], ...]
    step_zero_traces: tuple[Mapping[str, object], ...]
    step_zero_comparator: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class UtilityEvaluationDependencies:
    episode_runner: (
        Callable[[FullAuthorityPolicy, int, int, str, Path], ProtectedEpisodeMetrics] | None
    ) = None
    failure_mode: str | None = None
    fail_after_episodes: int | None = None
    wall_seconds: float = 30.0 * 60.0


@dataclass(frozen=True, slots=True)
class EvaluatedEpisode:
    metrics: ProtectedEpisodeMetrics
    trace: Mapping[str, object]


def _behavior(cell: str, boundary: int) -> str:
    if cell == "hold_expert":
        return "expert"
    if cell == "hold_medium":
        return "medium"
    if cell == "hold_simple":
        return "simple"
    if cell != "fixed_round_trip":
        raise ValueError("utility cell is unknown")
    if boundary < 300:
        return "expert"
    return "medium" if boundary < 600 else "expert"


def _torso_up(quaternion: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    norm = w * w + x * x + y * y + z * z
    result = (w * w - x * x - y * y + z * z) / norm
    if not math.isfinite(result):
        raise ExperimentContractError("protected torso-up metric is non-finite")
    return result


def _forbidden_contacts(environment: object, step: int) -> list[dict[str, object]]:
    physical = environment.unwrapped
    samples = getattr(physical, "last_full_contact_samples", None)
    names = getattr(physical, "_reference_corpus_geom_names", None)
    if type(samples) is not tuple or type(names) is not tuple:
        raise ExperimentContractError("protected contact instrumentation is unavailable")
    result = []
    for sample in samples:
        first = names[sample.geom1_id]
        second = names[sample.geom2_id]
        if "floor" not in {first, second}:
            continue
        other = second if first == "floor" else first
        if other not in _FOOT_GEOMS:
            result.append(
                {
                    "geom": other,
                    "physics_substep_index": sample.physics_substep_index,
                    "step": step,
                }
            )
    return result


def _real_episode(
    policy: FullAuthorityPolicy,
    policy_seed: int,
    evaluation_seed: int,
    cell: str,
    corpus_root: Path,
    *,
    segment_targets_m_s: tuple[float, float, float],
) -> EvaluatedEpisode:
    references = {
        behavior: load_v2_reference_clip(
            corpus_root,
            block=evaluation_seed,
            behavior=behavior,
        ).reference_rows
        for behavior in ("expert", "medium", "simple")
    }
    environment = make_reference_corpus_env()
    steps: list[dict[str, object]] = []
    switches: list[dict[str, object]] = []
    try:
        raw_observation, _info = environment.reset(seed=evaluation_seed)
        observation = np.ascontiguousarray(raw_observation, dtype="<f4")
        abi = validate_humanoid_actuator_abi(environment)
        physical = environment.unwrapped
        body_mass = np.ascontiguousarray(physical.model.body_mass, dtype="<f8")
        reset_state = tracking_state_with_bounded_reset_orientation(environment, abi)
        previous_x = float(reset_state.root_position_world_m[0])
        current_state = reset_state
        active_behavior = _behavior(cell, 0)
        active_phase = 0
        for step in range(HORIZON):
            if cell == "fixed_round_trip" and step in {300, 600}:
                target_behavior = "medium" if step == 300 else "expert"
                transfer = select_evaluator_nearest_phase(
                    state=evaluator_state_record(current_state),
                    target_rows=references[target_behavior],
                    task_step=step,
                    source_behavior=active_behavior,
                    target_behavior=target_behavior,
                    reason="fixed_utility_switch",
                )
                active_behavior = target_behavior
                active_phase = int(transfer["selected_phase"])
                switches.append(
                    {
                        "boundary": step,
                        "from_behavior": transfer["source_behavior"],
                        "selected_normalized_errors": transfer["selected_normalized_errors"],
                        "selected_phase": transfer["selected_phase"],
                        "selected_score": transfer["selected_score"],
                        "to_behavior": transfer["target_behavior"],
                    }
                )
            window_indices = tuple(min(active_phase + offset, HORIZON) for offset in range(8))
            window_rows = [references[active_behavior][index] for index in window_indices]
            window = np.ascontiguousarray(window_rows, dtype="<f4")
            action = policy.actor.act(compose_policy_input(observation, window)).physical
            if (
                not np.isfinite(action).all()
                or np.any(action < np.float32(-0.4))
                or np.any(action > np.float32(0.4))
            ):
                raise ExperimentContractError("protected evaluation action is out of bounds")
            body_xipos_before = np.ascontiguousarray(physical.data.xipos, dtype="<f8").copy()
            raw_next, _stock_reward, terminated, truncated, _base_info = environment.step(action)
            body_xipos_after = np.ascontiguousarray(physical.data.xipos, dtype="<f8").copy()
            evaluator_mass_center_x_m(body_mass, body_xipos_before)
            evaluator_mass_center_x_m(body_mass, body_xipos_after)
            observation = np.ascontiguousarray(raw_next, dtype="<f4")
            state = tracking_state(environment, abi)
            target_phase = min(active_phase + 1, HORIZON)
            target = references[active_behavior][target_phase]
            root_x = float(state.root_position_world_m[0])
            step_contacts = _forbidden_contacts(environment, step)
            fallen_now = (
                not 1.0 <= state.root_height_m <= 2.0
                or _torso_up(state.root_orientation_wxyz) < 0.5
            )
            steps.append(
                protected_step_record(
                    step=step,
                    state=state,
                    reference_behavior=active_behavior,
                    reference_index=target_phase,
                    reference_row=target,
                    action=action,
                    root_x_before_m=previous_x,
                    body_mass=body_mass,
                    body_xipos_before=body_xipos_before,
                    body_xipos_after=body_xipos_after,
                    forbidden_contacts=step_contacts,
                    fallen=fallen_now,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
            )
            previous_x = root_x
            if bool(terminated) or (bool(truncated) and step + 1 != HORIZON):
                raise ExperimentContractError("protected evaluation ended before 1,000 steps")
            active_phase = target_phase
            current_state = state
        aggregates = recompute_protected_episode(
            steps=steps,
            cell=cell,
            switches=switches,
            segment_targets_m_s=segment_targets_m_s,
        )
        trace = {
            "cell": cell,
            "evaluation_seed": evaluation_seed,
            "policy_seed": policy_seed,
            "schema_version": 1,
            "segment_targets_m_s": list(segment_targets_m_s),
            "steps": steps,
            "switch_records": switches,
            "trace_schema_id": "humanoid_phase_b_protected_episode_trace/v1",
        }
        trace["steps_sha256"] = protected_trace_sha256(steps)
        canonical_json_bytes(trace)
        metrics = ProtectedEpisodeMetrics(
            policy_seed=policy_seed,
            evaluation_seed=evaluation_seed,
            cell=cell,
            checkpoint_sha256="0" * 64,
            observed_steps=int(aggregates["observed_steps"]),
            root_delta_forward_speed_m_s=tuple(aggregates["root_delta_forward_speed_m_s"]),
            com_forward_speed_m_s=tuple(aggregates["com_forward_speed_m_s"]),
            six_tracking_errors=dict(aggregates["six_tracking_errors"]),
            fall=bool(aggregates["fall"]),
            forbidden_contacts=tuple(aggregates["forbidden_contacts"]),
            action_bounds_ok=bool(aggregates["action_bounds_ok"]),
            switch_records=tuple(switches),
            resynchronization_records=tuple(aggregates["resynchronization_records"]),
            segment_errors=dict(aggregates["segment_errors"]),
            transition_window_error=aggregates["transition_window_error"],
            settle_latency_steps=aggregates["settle_latency_steps"],
            time_to_first_failure_steps=aggregates["time_to_first_failure_steps"],
            task_success=None,
            settled_state_normalized_error=aggregates["settled_state_normalized_error"],
        )
        return EvaluatedEpisode(metrics=metrics, trace=trace)
    finally:
        environment.close()


def _bind_checkpoint(
    episode: ProtectedEpisodeMetrics,
    checkpoint_sha256: str,
) -> ProtectedEpisodeMetrics:
    return replace(episode, checkpoint_sha256=checkpoint_sha256)


def _score_task_success(
    episode: ProtectedEpisodeMetrics,
    calibration: TaskSuccessCalibration | None,
) -> ProtectedEpisodeMetrics:
    if episode.cell != "fixed_round_trip" or calibration is None:
        return replace(episode, task_success=None)
    latencies = [record.get("settle_latency_steps") for record in episode.resynchronization_records]
    latency_passed = len(latencies) == 2 and all(
        type(latency) is int and 0 <= latency <= cap
        for latency, cap in zip(latencies, calibration.transition_latency_caps_steps, strict=True)
    )
    segment_passed = set(episode.segment_errors) == set(
        calibration.segment_speed_error_bands_m_s
    ) and all(
        float(episode.segment_errors[name]) <= limit
        for name, limit in calibration.segment_speed_error_bands_m_s.items()
    )
    passed = episode.safety_passed and latency_passed and segment_passed
    return replace(episode, task_success=passed)


def evaluate_policy_checkpoint(
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    step_zero_actor_path: Path,
    step_zero_actor_sha256: str,
    corpus_root: Path,
    calibration_receipt_path: Path | None,
    calibration_receipt_sha256: str | None,
    segment_targets_m_s: tuple[float, float, float] | None = None,
    dependencies: UtilityEvaluationDependencies | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> UtilityEvaluationResult:
    """Evaluate one final checkpoint and its step-0 comparator on identical cells."""

    calibration = (
        load_calibration_receipt(
            calibration_receipt_path,
            expected_sha256=calibration_receipt_sha256,
        )
        if calibration_receipt_path is not None and calibration_receipt_sha256 is not None
        else None
    )
    if (calibration_receipt_path is None) != (calibration_receipt_sha256 is None):
        raise ExperimentContractError("calibration receipt path and SHA-256 must be paired")
    checkpoint = load_full_checkpoint(checkpoint_path, expected_sha256=checkpoint_sha256)
    step_zero_actor = load_full_authority_actor(
        step_zero_actor_path,
        expected_sha256=step_zero_actor_sha256,
    )
    step_zero_policy = FullAuthorityPolicy(
        step_zero_actor.actor,
        value_seed=int(checkpoint.metadata["value_initialization_seed"]),
    )
    selected = dependencies or UtilityEvaluationDependencies()
    if selected.episode_runner is None:
        if (
            type(segment_targets_m_s) is not tuple
            or len(segment_targets_m_s) != 3
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in segment_targets_m_s
            )
        ):
            raise ExperimentContractError(
                "real utility evaluation requires the three frozen task targets"
            )

        def runner(
            policy: FullAuthorityPolicy,
            policy_seed: int,
            evaluation_seed: int,
            cell: str,
            corpus_root: Path,
        ) -> ProtectedEpisodeMetrics:
            return _real_episode(
                policy,
                policy_seed,
                evaluation_seed,
                cell,
                corpus_root,
                segment_targets_m_s=segment_targets_m_s,
            )

    else:
        runner = selected.episode_runner
    policy_seed = int(checkpoint.metadata["ppo_seed"])
    trained = []
    baseline = []
    trained_traces = []
    baseline_traces = []
    completed_episodes = 0
    for cell in ("hold_expert", "hold_medium", "hold_simple", "fixed_round_trip"):
        for evaluation_seed in EVALUATION_BLOCKS:
            for policy, checkpoint_hash, metrics_rows, trace_rows in (
                (checkpoint.policy, checkpoint_sha256, trained, trained_traces),
                (step_zero_policy, step_zero_actor_sha256, baseline, baseline_traces),
            ):
                raw = runner(policy, policy_seed, evaluation_seed, cell, corpus_root)
                episode = raw.metrics if isinstance(raw, EvaluatedEpisode) else raw
                if not isinstance(episode, ProtectedEpisodeMetrics):
                    raise ExperimentContractError("episode runner returned an invalid result")
                metrics_rows.append(
                    _score_task_success(_bind_checkpoint(episode, checkpoint_hash), calibration)
                )
                if isinstance(raw, EvaluatedEpisode):
                    trace = dict(raw.trace)
                    trace["checkpoint_sha256"] = checkpoint_hash
                    trace_rows.append(trace)
                completed_episodes += 1
                if progress_callback is not None:
                    progress_callback(completed_episodes)
    trained_bytes = canonical_json_bytes([episode.to_dict() for episode in trained])
    baseline_bytes = canonical_json_bytes([episode.to_dict() for episode in baseline])
    comparator = {
        "bitwise_equal": True,
        "evaluation_schedule_identical": True,
        "calibration_receipt_sha256": calibration.sha256 if calibration else None,
        "step_zero_actor_sha256": step_zero_actor_sha256,
        "step_zero_metrics_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "trained_metrics_sha256": hashlib.sha256(trained_bytes).hexdigest(),
    }
    return UtilityEvaluationResult(
        calibration=calibration,
        checkpoint=checkpoint,
        trained_episodes=tuple(trained),
        step_zero_episodes=tuple(baseline),
        trained_traces=tuple(trained_traces),
        step_zero_traces=tuple(baseline_traces),
        step_zero_comparator=comparator,
    )


__all__ = [
    "EVALUATION_BLOCKS",
    "EvaluatedEpisode",
    "UtilityEvaluationDependencies",
    "UtilityEvaluationResult",
    "evaluate_policy_checkpoint",
]
