"""Protected hold/transition utility evaluation for stored Phase B policies."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.envs.reference_corpus import make_reference_corpus_env
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.rewards.stock_humanoid import (
    body_mass_weighted_com_x_velocity_m_s,
)
from oracle_composition.tracking.humanoid_reference import (
    tracking_state,
    tracking_state_with_bounded_reset_orientation,
    validate_humanoid_actuator_abi,
)
from oracle_composition.tracking.reward import compute_tracking_reward

from .calibration import TaskSuccessCalibration, load_calibration_receipt
from .persistence import LoadedFullCheckpoint, load_full_checkpoint
from .policy import FullAuthorityPolicy, compose_policy_input, load_full_authority_actor
from .reference_runtime import ERROR_NAMES, load_v2_reference_clip, select_nearest_phase
from .report_v2 import ProtectedEpisodeMetrics

EVALUATION_BLOCKS = tuple(range(120101, 120121))
UTILITY_EVALUATOR_ID = "humanoid_phase_b_protected_utility_evaluator/v1"
HORIZON = 1_000
_FOOT_GEOMS = {"left_foot", "right_foot"}


@dataclass(frozen=True, slots=True)
class UtilityEvaluationResult:
    calibration: TaskSuccessCalibration
    checkpoint: LoadedFullCheckpoint
    trained_episodes: tuple[ProtectedEpisodeMetrics, ...]
    step_zero_episodes: tuple[ProtectedEpisodeMetrics, ...]
    step_zero_comparator: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class UtilityEvaluationDependencies:
    episode_runner: (
        Callable[[FullAuthorityPolicy, int, int, str, Path], ProtectedEpisodeMetrics] | None
    ) = None


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


def _resynchronization_records(
    normalized_errors: Sequence[float],
    switches: Sequence[Mapping[str, object]],
) -> tuple[Mapping[str, object], ...]:
    records = []
    for switch in switches:
        boundary = int(switch["boundary"])
        start = boundary
        latest_start = min(len(normalized_errors) - 8, boundary + 56)
        latency: int | None = None
        for index in range(start, latest_start + 1):
            if all(value <= 1.0 for value in normalized_errors[index : index + 8]):
                latency = index + 8 - boundary
                break
        records.append(
            {
                "eight_consecutive_boundaries_at_or_below_one": latency is not None,
                "from_behavior": switch["from_behavior"],
                "settle_latency_steps": latency,
                "to_behavior": switch["to_behavior"],
            }
        )
    return tuple(records)


def _real_episode(
    policy: FullAuthorityPolicy,
    policy_seed: int,
    evaluation_seed: int,
    cell: str,
    corpus_root: Path,
    *,
    segment_targets_m_s: tuple[float, float, float],
) -> ProtectedEpisodeMetrics:
    references = {
        behavior: load_v2_reference_clip(
            corpus_root,
            block=evaluation_seed,
            behavior=behavior,
        ).reference_rows
        for behavior in ("expert", "medium", "simple")
    }
    environment = make_reference_corpus_env()
    root_delta_speeds: list[float] = []
    com_speeds: list[float] = []
    raw_errors: dict[str, list[float]] = {name: [] for name in ERROR_NAMES}
    normalized_max_errors: list[float] = []
    contacts: list[dict[str, object]] = []
    switches: list[dict[str, object]] = []
    fall = False
    first_failure: int | None = None
    observed_steps = 0
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
                transfer = select_nearest_phase(
                    state=current_state,
                    target_rows=references[target_behavior],
                    task_step=step,
                    source_behavior=active_behavior,
                    target_behavior=target_behavior,
                    reason="fixed_utility_switch",
                )
                active_behavior = target_behavior
                active_phase = transfer.selected_phase
                switches.append(
                    {
                        "boundary": step,
                        "from_behavior": transfer.source_behavior,
                        "selected_normalized_errors": {
                            name: value
                            for name, value in zip(
                                ERROR_NAMES,
                                transfer.selected_normalized_errors,
                                strict=True,
                            )
                        },
                        "selected_phase": transfer.selected_phase,
                        "selected_score": list(transfer.selected_score),
                        "to_behavior": transfer.target_behavior,
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
            com_forward_speed = body_mass_weighted_com_x_velocity_m_s(
                body_mass_f64=body_mass,
                body_xipos_before_f64=body_xipos_before,
                body_xipos_after_f64=np.ascontiguousarray(physical.data.xipos, dtype="<f8"),
                control_period_s=0.015,
            )
            observation = np.ascontiguousarray(raw_next, dtype="<f4")
            state = tracking_state(environment, abi)
            target_phase = min(active_phase + 1, HORIZON)
            target = references[active_behavior][target_phase]
            tracking = compute_tracking_reward(state=state, reference_frame=target)
            errors = tracking.error_components()
            for name in ERROR_NAMES:
                raw_errors[name].append(float(errors[name]))
            normalized_max_errors.append(
                max(
                    float(errors["root_height_abs_error_m"]) / 0.20,
                    float(errors["root_orientation_error_rad"]) / 0.50,
                    float(errors["root_linear_velocity_rmse_m_s"]) / 1.0,
                    float(errors["root_angular_velocity_rmse_rad_s"]) / 2.0,
                    float(errors["joint_position_rmse_rad"]) / 0.35,
                    float(errors["joint_velocity_rmse_rad_s"]) / 2.0,
                )
            )
            root_x = float(state.root_position_world_m[0])
            root_delta_speeds.append((root_x - previous_x) / 0.015)
            previous_x = root_x
            com_speeds.append(com_forward_speed)
            step_contacts = _forbidden_contacts(environment, step)
            contacts.extend(step_contacts)
            fallen_now = (
                not 1.0 <= state.root_height_m <= 2.0
                or _torso_up(state.root_orientation_wxyz) < 0.5
            )
            fall = fall or fallen_now
            if first_failure is None and (fallen_now or step_contacts):
                first_failure = step + 1
            observed_steps += 1
            if bool(terminated) or (bool(truncated) and step + 1 != HORIZON):
                raise ExperimentContractError("protected evaluation ended before 1,000 steps")
            active_phase = target_phase
            current_state = state
        summaries = {
            name: float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))
            for name, values in raw_errors.items()
        }
        resynchronization = _resynchronization_records(normalized_max_errors, switches)
        fast_target, slow_target, return_fast_target = segment_targets_m_s
        segment_errors = {
            "fast": float(np.mean(np.abs(np.asarray(com_speeds[:300]) - fast_target))),
            "slow": float(np.mean(np.abs(np.asarray(com_speeds[300:600]) - slow_target))),
            "return_fast": float(
                np.mean(np.abs(np.asarray(com_speeds[600:]) - return_fast_target))
            ),
        }
        transition_window = (
            float(
                np.mean(
                    [
                        *normalized_max_errors[300:364],
                        *normalized_max_errors[600:664],
                    ]
                )
            )
            if cell == "fixed_round_trip"
            else None
        )
        settled_state_error = (
            float(
                max(
                    np.mean(normalized_max_errors[332:364]),
                    np.mean(normalized_max_errors[632:664]),
                )
            )
            if cell == "fixed_round_trip"
            else None
        )
        latencies = [
            record["settle_latency_steps"]
            for record in resynchronization
            if record["settle_latency_steps"] is not None
        ]
        return ProtectedEpisodeMetrics(
            policy_seed=policy_seed,
            evaluation_seed=evaluation_seed,
            cell=cell,
            checkpoint_sha256="0" * 64,
            observed_steps=observed_steps,
            root_delta_forward_speed_m_s=tuple(root_delta_speeds),
            com_forward_speed_m_s=tuple(com_speeds),
            six_tracking_errors=summaries,
            fall=fall,
            forbidden_contacts=tuple(contacts),
            action_bounds_ok=True,
            switch_records=tuple(switches),
            resynchronization_records=resynchronization,
            segment_errors=segment_errors,
            transition_window_error=transition_window,
            settle_latency_steps=max(latencies) if latencies else None,
            time_to_first_failure_steps=first_failure,
            task_success=None,
            settled_state_normalized_error=settled_state_error,
        )
    finally:
        environment.close()


def _bind_checkpoint(
    episode: ProtectedEpisodeMetrics,
    checkpoint_sha256: str,
) -> ProtectedEpisodeMetrics:
    return replace(episode, checkpoint_sha256=checkpoint_sha256)


def _score_task_success(
    episode: ProtectedEpisodeMetrics,
    calibration: TaskSuccessCalibration,
) -> ProtectedEpisodeMetrics:
    if episode.cell != "fixed_round_trip":
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
    settled = episode.settled_state_normalized_error
    passed = (
        episode.safety_passed
        and latency_passed
        and segment_passed
        and settled is not None
        and float(settled) <= calibration.settled_state_normalized_error_band
    )
    return replace(episode, task_success=passed)


def evaluate_policy_checkpoint(
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    step_zero_actor_path: Path,
    step_zero_actor_sha256: str,
    corpus_root: Path,
    calibration_receipt_path: Path,
    calibration_receipt_sha256: str,
    segment_targets_m_s: tuple[float, float, float] | None = None,
    dependencies: UtilityEvaluationDependencies | None = None,
) -> UtilityEvaluationResult:
    """Evaluate one final checkpoint and its step-0 comparator on identical cells."""

    calibration = load_calibration_receipt(
        calibration_receipt_path,
        expected_sha256=calibration_receipt_sha256,
    )
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
    for cell in ("hold_expert", "hold_medium", "hold_simple", "fixed_round_trip"):
        for evaluation_seed in EVALUATION_BLOCKS:
            trained.append(
                _score_task_success(
                    _bind_checkpoint(
                        runner(checkpoint.policy, policy_seed, evaluation_seed, cell, corpus_root),
                        checkpoint_sha256,
                    ),
                    calibration,
                )
            )
            baseline.append(
                _score_task_success(
                    _bind_checkpoint(
                        runner(step_zero_policy, policy_seed, evaluation_seed, cell, corpus_root),
                        step_zero_actor_sha256,
                    ),
                    calibration,
                )
            )
    trained_bytes = canonical_json_bytes([episode.to_dict() for episode in trained])
    baseline_bytes = canonical_json_bytes([episode.to_dict() for episode in baseline])
    comparator = {
        "bitwise_equal": True,
        "evaluation_schedule_identical": True,
        "calibration_receipt_sha256": calibration.sha256,
        "step_zero_actor_sha256": step_zero_actor_sha256,
        "step_zero_metrics_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "trained_metrics_sha256": hashlib.sha256(trained_bytes).hexdigest(),
    }
    return UtilityEvaluationResult(
        calibration=calibration,
        checkpoint=checkpoint,
        trained_episodes=tuple(trained),
        step_zero_episodes=tuple(baseline),
        step_zero_comparator=comparator,
    )


__all__ = [
    "EVALUATION_BLOCKS",
    "UtilityEvaluationDependencies",
    "UtilityEvaluationResult",
    "evaluate_policy_checkpoint",
]
