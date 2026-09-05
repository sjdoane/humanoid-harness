"""Deterministic Phase B scientific receipts with separate telemetry."""

from __future__ import annotations

import hashlib
import math
import platform
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from types import MappingProxyType

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes, sha256_file
from oracle_composition.experiments.artifact_io import (
    PublishedArtifact,
    publish_bytes_without_overwrite,
)
from oracle_composition.harness.evidence import current_authority_identities

from .contracts import CLAIM_CEILING, REPORT_V2_SCHEMA_ID, validate_cycle_report
from .reference_runtime import ERROR_NAMES
from .training import COHORT_SEEDS

SCIENTIFIC_RECEIPT_SCHEMA_ID = "humanoid_phase_b_scientific_receipt/v2"
TELEMETRY_SCHEMA_ID = "humanoid_phase_b_telemetry/v1"
REPORT_WRITER_ID = "humanoid_phase_b_report_v2_writer/v1"
TASK_SUCCESS_ENDPOINT_ID = "phase_b_speed_profile_task_success/v1"
UTILITY_GATE_ID = "phase_b_minimum_utility_gate/v1"

UTILITY_CELLS = ("hold_expert", "hold_medium", "hold_simple", "fixed_round_trip")
TRACKING_SCALES = MappingProxyType(
    {
        "joint_position_rmse_rad": 0.35,
        "joint_velocity_rmse_rad_s": 2.0,
        "root_angular_velocity_rmse_rad_s": 2.0,
        "root_height_abs_error_m": 0.20,
        "root_linear_velocity_rmse_m_s": 1.0,
        "root_orientation_error_rad": 0.50,
    }
)


@dataclass(frozen=True, slots=True)
class ProtectedEpisodeMetrics:
    policy_seed: int
    evaluation_seed: int
    cell: str
    checkpoint_sha256: str
    observed_steps: int
    root_delta_forward_speed_m_s: tuple[float, ...]
    com_forward_speed_m_s: tuple[float, ...]
    six_tracking_errors: Mapping[str, float]
    fall: bool
    forbidden_contacts: tuple[Mapping[str, object], ...]
    action_bounds_ok: bool
    switch_records: tuple[Mapping[str, object], ...]
    resynchronization_records: tuple[Mapping[str, object], ...]
    segment_errors: Mapping[str, float]
    transition_window_error: float | None
    settle_latency_steps: int | None
    time_to_first_failure_steps: int | None
    task_success: bool | None

    def __post_init__(self) -> None:
        if (
            type(self.policy_seed) is not int
            or self.policy_seed <= 0
            or type(self.evaluation_seed) is not int
            or self.evaluation_seed <= 0
            or self.cell not in UTILITY_CELLS
            or type(self.observed_steps) is not int
            or not 0 <= self.observed_steps <= 1_000
        ):
            raise ValueError("protected episode identity or horizon differs")
        _require_sha(self.checkpoint_sha256, "checkpoint_sha256")
        if not self.root_delta_forward_speed_m_s or not self.com_forward_speed_m_s:
            raise ValueError("both protected forward-speed quantities are required")
        if (
            len(self.root_delta_forward_speed_m_s) != len(self.com_forward_speed_m_s)
            or len(self.root_delta_forward_speed_m_s) != self.observed_steps
        ):
            raise ValueError("protected forward-speed series lengths differ")
        _finite_sequence(self.root_delta_forward_speed_m_s, "root-delta speed")
        _finite_sequence(self.com_forward_speed_m_s, "COM speed")
        if set(self.six_tracking_errors) != set(ERROR_NAMES):
            raise ValueError("episode must report all six tracking errors")
        _finite_sequence(tuple(self.six_tracking_errors.values()), "tracking error")
        if any(float(value) < 0.0 for value in self.six_tracking_errors.values()):
            raise ValueError("tracking errors must be non-negative")
        if type(self.fall) is not bool or type(self.action_bounds_ok) is not bool:
            raise ValueError("episode safety flags must be booleans")
        if type(self.task_success) not in {bool, type(None)}:
            raise ValueError("task_success must be boolean or uncalibrated")
        if set(self.segment_errors) != {"fast", "slow", "return_fast"}:
            raise ValueError("episode must report all three task segment errors")
        for name, value in self.segment_errors.items():
            if type(name) is not str or not name or not math.isfinite(float(value)):
                raise ValueError("segment error is malformed")
        for contact in self.forbidden_contacts:
            if (
                not isinstance(contact, Mapping)
                or type(contact.get("geom")) is not str
                or not contact["geom"]
                or type(contact.get("step")) is not int
                or not 0 <= int(contact["step"]) < 1_000
                or type(contact.get("physics_substep_index")) is not int
                or int(contact["physics_substep_index"]) < 0
            ):
                raise ValueError("forbidden-contact record is malformed")
        for value, name in (
            (self.transition_window_error, "transition_window_error"),
            (self.settle_latency_steps, "settle_latency_steps"),
            (self.time_to_first_failure_steps, "time_to_first_failure_steps"),
        ):
            if value is not None and (
                type(value) not in {int, float} or not math.isfinite(float(value)) or value < 0
            ):
                raise ValueError(f"{name} is malformed")
        if self.task_success is True and not self.safety_passed:
            raise ValueError("an unsafe episode cannot be recorded as task-successful")

    @property
    def safety_passed(self) -> bool:
        return (
            self.observed_steps == 1_000
            and not self.fall
            and not self.forbidden_contacts
            and self.action_bounds_ok
        )

    @property
    def tracking_passed(self) -> bool:
        return all(
            float(self.six_tracking_errors[name]) <= TRACKING_SCALES[name]
            for name in TRACKING_SCALES
        )

    @property
    def resynchronization_passed(self) -> bool:
        if self.cell != "fixed_round_trip":
            return not self.switch_records and not self.resynchronization_records
        if len(self.switch_records) != 2 or len(self.resynchronization_records) != 2:
            return False
        expected = (
            (300, "expert", "medium"),
            (600, "medium", "expert"),
        )
        observed = tuple(
            (
                record.get("boundary"),
                record.get("from_behavior"),
                record.get("to_behavior"),
            )
            for record in self.switch_records
        )
        if observed != expected:
            return False
        return all(
            resynchronization.get("from_behavior") == switch[1]
            and resynchronization.get("to_behavior") == switch[2]
            and resynchronization.get("eight_consecutive_boundaries_at_or_below_one") is True
            and type(resynchronization.get("settle_latency_steps")) is int
            and 0 <= int(resynchronization["settle_latency_steps"]) <= 64
            for resynchronization, switch in zip(
                self.resynchronization_records,
                expected,
                strict=True,
            )
        )

    @property
    def utility_passed(self) -> bool:
        return self.safety_passed and self.tracking_passed and self.resynchronization_passed

    def to_dict(self) -> dict[str, object]:
        return {
            "action_bounds_ok": self.action_bounds_ok,
            "checkpoint_sha256": self.checkpoint_sha256,
            "com_forward_speed_m_s": list(self.com_forward_speed_m_s),
            "contacts": [dict(item) for item in self.forbidden_contacts],
            "evaluation_seed": self.evaluation_seed,
            "fall": self.fall,
            "observed_steps": self.observed_steps,
            "policy_seed": self.policy_seed,
            "resynchronization_records": [dict(item) for item in self.resynchronization_records],
            "root_delta_forward_speed_m_s": list(self.root_delta_forward_speed_m_s),
            "segment_errors": dict(sorted(self.segment_errors.items())),
            "settle_latency_steps": self.settle_latency_steps,
            "six_tracking_errors": dict(sorted(self.six_tracking_errors.items())),
            "switch_records": [dict(item) for item in self.switch_records],
            "task_success": self.task_success,
            "time_to_first_failure_steps": self.time_to_first_failure_steps,
            "transition_window_error": self.transition_window_error,
            "utility_passed": self.utility_passed,
            "cell": self.cell,
        }


@dataclass(frozen=True, slots=True)
class SeedReportFacts:
    ppo_seed: int
    checkpoint_sha256: str
    strict_export_sha256: str
    training: Mapping[str, object]
    step_zero_comparator: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.ppo_seed) is not int or self.ppo_seed <= 0:
            raise ValueError("seed report PPO seed is invalid")
        _require_sha(self.checkpoint_sha256, "checkpoint SHA-256")
        _require_sha(self.strict_export_sha256, "strict export SHA-256")
        if self.step_zero_comparator.get("bitwise_equal") is not True:
            raise ValueError("every checkpoint requires its step-0 E1 comparator")
        required_training = {
            "evidence_class",
            "execution_manifest_sha256",
            "losses",
            "observed_transitions",
            "optimizer_updates",
            "planned_transitions",
            "ppo_seed",
            "reward_totals",
            "rollouts",
            "rsi_ledger_sha256",
            "stream_counts",
            "unfreeze_receipt_sha256",
        }
        if not required_training.issubset(self.training):
            raise ValueError("seed report training facts are incomplete")
        if (
            self.training["ppo_seed"] != self.ppo_seed
            or self.training["planned_transitions"] != self.training["observed_transitions"]
        ):
            raise ValueError("seed report training counters differ")
        _require_sha(self.training["execution_manifest_sha256"], "training manifest SHA-256")
        _require_sha(self.training["rsi_ledger_sha256"], "RSI ledger SHA-256")
        _require_sha(self.training["unfreeze_receipt_sha256"], "unfreeze receipt SHA-256")


@dataclass(frozen=True, slots=True)
class ReportArtifacts:
    scientific_receipt: PublishedArtifact
    telemetry: PublishedArtifact
    markdown: PublishedArtifact


def _require_sha(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _finite_sequence(values: Sequence[object], field: str) -> None:
    if any(type(value) not in {int, float} or not math.isfinite(float(value)) for value in values):
        raise ValueError(f"{field} contains NaN or Inf")


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    return math.fsum(
        math.comb(n, index) * probability**index * (1.0 - probability) ** (n - index)
        for index in range(k + 1)
    )


def exact_binomial_interval(
    successes: int, total: int, *, alpha: float = 0.05
) -> tuple[float, float]:
    """Return a two-sided Clopper-Pearson interval without SciPy."""

    if (
        type(successes) is not int
        or type(total) is not int
        or not 0 <= successes <= total
        or total <= 0
        or type(alpha) is not float
        or not 0.0 < alpha < 1.0
    ):
        raise ValueError("exact binomial interval inputs are invalid")
    tail = alpha / 2.0
    if successes == 0:
        lower = 0.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            midpoint = (lo + hi) / 2.0
            upper_tail = 1.0 - _binomial_cdf(successes - 1, total, midpoint)
            if upper_tail < tail:
                lo = midpoint
            else:
                hi = midpoint
        lower = (lo + hi) / 2.0
    if successes == total:
        upper = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            midpoint = (lo + hi) / 2.0
            lower_tail = _binomial_cdf(successes, total, midpoint)
            if lower_tail > tail:
                lo = midpoint
            else:
                hi = midpoint
        upper = (lo + hi) / 2.0
    return lower, upper


def utility_gate(
    *,
    checkpoints: Sequence[SeedReportFacts],
    episodes: Sequence[ProtectedEpisodeMetrics],
) -> dict[str, object]:
    """Compute exact cell and five-checkpoint family booleans."""

    by_seed_cell: dict[tuple[int, str], list[ProtectedEpisodeMetrics]] = defaultdict(list)
    checkpoint_by_seed = {checkpoint.ppo_seed: checkpoint for checkpoint in checkpoints}
    if len(checkpoint_by_seed) != len(checkpoints) or any(
        episode.policy_seed not in checkpoint_by_seed for episode in episodes
    ):
        raise ValueError("utility episodes and checkpoint seed authority differ")
    for episode in episodes:
        by_seed_cell[(episode.policy_seed, episode.cell)].append(episode)
    checkpoint_rows = []
    for checkpoint in sorted(checkpoints, key=lambda item: item.ppo_seed):
        cells = {}
        for cell in UTILITY_CELLS:
            rows = by_seed_cell[(checkpoint.ppo_seed, cell)]
            passed = sum(item.utility_passed for item in rows)
            fixed_denominator = sorted(item.evaluation_seed for item in rows) == list(
                range(120101, 120121)
            ) and all(item.checkpoint_sha256 == checkpoint.checkpoint_sha256 for item in rows)
            cells[cell] = {
                "cell_passed": fixed_denominator and passed >= 16,
                "episode_count": len(rows),
                "episode_pass_count": passed,
                "fixed_denominator_without_replacement": fixed_denominator,
                "minimum_pass_count": 16,
            }
        checkpoint_rows.append(
            {
                "cells": cells,
                "checkpoint_passed": all(value["cell_passed"] for value in cells.values()),
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
                "ppo_seed": checkpoint.ppo_seed,
                "step_zero_comparator": dict(checkpoint.step_zero_comparator),
            }
        )
    passed_checkpoints = sum(bool(row["checkpoint_passed"]) for row in checkpoint_rows)
    return {
        "checkpoint_results": checkpoint_rows,
        "family_passed": (
            tuple(row["ppo_seed"] for row in checkpoint_rows) == COHORT_SEEDS
            and passed_checkpoints >= 4
        ),
        "family_rule": "all_5_checkpoints_present_and_at_least_4_pass_all_4_cells",
        "gate_id": UTILITY_GATE_ID,
        "passed_checkpoint_count": passed_checkpoints,
        "required_checkpoint_count": 5,
        "required_passed_checkpoint_count": 4,
    }


def task_success_endpoint(
    episodes: Sequence[ProtectedEpisodeMetrics],
    *,
    segment_tolerances: Mapping[str, float] | None = None,
) -> dict[str, object]:
    calibrated = [episode for episode in episodes if episode.task_success is not None]
    successes = sum(bool(episode.task_success) for episode in calibrated)
    interval = exact_binomial_interval(successes, len(calibrated)) if calibrated else None
    segment_names = sorted({name for episode in episodes for name in episode.segment_errors})
    secondary = {
        "equal_weight_mean_three_segment_errors": (
            fmean(
                fmean(float(episode.segment_errors[name]) for name in segment_names)
                for episode in episodes
                if len(episode.segment_errors) == 3
            )
            if any(len(episode.segment_errors) == 3 for episode in episodes)
            else None
        ),
        "mean_settle_latency_steps": (
            fmean(
                float(episode.settle_latency_steps)
                for episode in episodes
                if episode.settle_latency_steps is not None
            )
            if any(episode.settle_latency_steps is not None for episode in episodes)
            else None
        ),
        "mean_transition_window_error": (
            fmean(
                float(episode.transition_window_error)
                for episode in episodes
                if episode.transition_window_error is not None
            )
            if any(episode.transition_window_error is not None for episode in episodes)
            else None
        ),
        "time_to_first_failure_steps": [
            episode.time_to_first_failure_steps for episode in episodes
        ],
    }
    tolerances: dict[str, object]
    if segment_tolerances is None:
        tolerances = {
            "calibration_status": "placeholder_until_calibrated",
            "fast": None,
            "return_fast": None,
            "slow": None,
            "transition_latency": None,
        }
    else:
        required_tolerances = {"fast", "return_fast", "slow", "transition_latency"}
        if set(segment_tolerances) != required_tolerances or any(
            type(value) not in {int, float} or not math.isfinite(float(value)) or float(value) < 0.0
            for value in segment_tolerances.values()
        ):
            raise ValueError("calibrated segment tolerances differ")
        tolerances = {
            "calibration_status": "frozen_calibrated",
            **{name: float(value) for name, value in sorted(segment_tolerances.items())},
        }
    return {
        "endpoint_id": TASK_SUCCESS_ENDPOINT_ID,
        "ordering": [
            "safety_gate",
            "task_success_proportion",
            "segment_balanced_error",
            "transition_latency",
        ],
        "primary_endpoint": {
            "exact_binomial_95_interval": list(interval) if interval is not None else None,
            "proportion": successes / len(calibrated) if calibrated else None,
            "successes": successes,
            "total": len(calibrated),
        },
        "safety_gate": {
            "all_1000_steps_completed": all(item.observed_steps == 1_000 for item in episodes),
            "fall_count": sum(item.fall for item in episodes),
            "forbidden_contact_count": sum(len(item.forbidden_contacts) for item in episodes),
            "passed": bool(episodes) and all(item.safety_passed for item in episodes),
            "rule": "0_falls_no_forbidden_contact_all_1000_steps",
        },
        "secondary_endpoints": secondary,
        "segment_tolerances": tolerances,
        "unsafe_arm_may_outrank_safe_arm": False,
    }


def _source_identity() -> dict[str, object]:
    path = Path(__file__)
    sources = {"phase_b/report_v2.py": sha256_file(path)}
    core = {"identity_id": REPORT_WRITER_ID, "source_sha256": sources}
    return {**core, "sha256": hashlib.sha256(canonical_json_bytes(core)).hexdigest()}


def _episode_rows(episodes: Sequence[ProtectedEpisodeMetrics]) -> list[dict[str, object]]:
    return [
        episode.to_dict()
        for episode in sorted(
            episodes,
            key=lambda item: (item.policy_seed, item.cell, item.evaluation_seed),
        )
    ]


def build_scientific_receipt(
    *,
    cycle: int,
    inputs: Mapping[str, object],
    seeds: Sequence[SeedReportFacts],
    episodes: Sequence[ProtectedEpisodeMetrics],
    reference_records: Sequence[Mapping[str, object]],
    reward_totals: Mapping[str, object],
    trace_index_sha256: str,
    prior_scientific_receipt: Mapping[str, object],
) -> dict[str, object]:
    """Build canonical report-v2 content without host or wall-clock telemetry."""

    if type(cycle) is not int or cycle < 0 or not seeds:
        raise ValueError("report cycle and seed evidence are required")
    required_inputs = {
        "e1_receipt_sha256",
        "evaluator_sha256",
        "execution_manifest_sha256",
        "library_sha256",
        "oracle_canonical_sha256",
        "oracle_file_sha256",
        "reference_sha256",
        "reward_compositor_sha256",
        "reward_file_sha256",
        "reward_formula_id",
        "reward_formula_sha256",
        "reward_schema_id",
        "starting_expert_identity",
        "task_sha256",
        "training_design_sha256",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != required_inputs:
        raise ValueError("report input identity set differs")
    for name in required_inputs - {
        "reward_formula_id",
        "reward_schema_id",
        "starting_expert_identity",
    }:
        _require_sha(inputs[name], name)
    _require_sha(trace_index_sha256, "trace index SHA-256")
    training_facts = [dict(seed.training) for seed in sorted(seeds, key=lambda item: item.ppo_seed)]
    evidence_classes = {value.get("evidence_class") for value in training_facts}
    if len(evidence_classes) != 1:
        raise ValueError("per-seed evidence classes differ")
    planned = sum(int(value["planned_transitions"]) for value in training_facts)
    observed = sum(int(value["observed_transitions"]) for value in training_facts)
    rollouts = sum(int(value["rollouts"]) for value in training_facts)
    updates = sum(int(value["optimizer_updates"]) for value in training_facts)
    stream_counts = {
        name: sum(int(value["stream_counts"][name]) for value in training_facts)
        for name in ("composition", "rehearsal")
    }
    rsi_sha = hashlib.sha256(
        canonical_json_bytes([value["rsi_ledger_sha256"] for value in training_facts])
    ).hexdigest()
    unfreeze_sha = hashlib.sha256(
        canonical_json_bytes([value["unfreeze_receipt_sha256"] for value in training_facts])
    ).hexdigest()
    episode_rows = _episode_rows(episodes)
    utility = utility_gate(checkpoints=seeds, episodes=episodes)
    endpoint = task_success_endpoint(
        [episode for episode in episodes if episode.cell == "fixed_round_trip"]
    )
    policy_rows = [
        {
            "checkpoint_sha256": seed.checkpoint_sha256,
            "ppo_seed": seed.ppo_seed,
            "step_zero_comparator": dict(seed.step_zero_comparator),
            "strict_export_sha256": seed.strict_export_sha256,
        }
        for seed in sorted(seeds, key=lambda item: item.ppo_seed)
    ]
    identities = {
        **current_authority_identities(),
        "phase_b_report_writer": _source_identity(),
    }
    runtime_fingerprint = {
        "device": "cpu",
        "environment_count": 4,
        "normalization": {"observation": False, "reward": False},
        "report_writer_sha256": identities["phase_b_report_writer"]["sha256"],
    }
    runtime_sha = hashlib.sha256(canonical_json_bytes(runtime_fingerprint)).hexdigest()
    first = policy_rows[0]
    reward = {
        "ignored_stock_reward": float(reward_totals["ignored_stock_reward"]),
        "parameters": dict(reward_totals.get("parameters", {})),
        "r_task": float(reward_totals["r_task"]),
        "r_track": float(reward_totals["r_track"]),
        "r_train": float(reward_totals["r_train"]),
    }
    report = {
        "claim_ceiling": CLAIM_CEILING,
        "cycle": cycle,
        "determinism_check": {
            "checkpoint_reload_bitwise": True,
            "scientific_receipt_excludes_wall_time": True,
            "step_zero_comparator_count": len(policy_rows),
        },
        "evaluation": {
            "episode_steps": 1_000,
            "evaluation_blocks": list(range(120101, 120121)),
            "failure_denominator": "all_predeclared_episodes",
            "protected_metrics_grade_candidate_reward": False,
            "utility_cells": list(UTILITY_CELLS),
        },
        "evidence_class": str(training_facts[0]["evidence_class"]),
        "execution_manifest_sha256": inputs["execution_manifest_sha256"],
        "generated_utc": "recorded_in_separate_telemetry",
        "identities": identities,
        "inputs": {
            name: value
            for name, value in inputs.items()
            if name
            in {
                "evaluator_sha256",
                "library_sha256",
                "oracle_canonical_sha256",
                "oracle_file_sha256",
                "reference_sha256",
                "reward_compositor_sha256",
                "reward_file_sha256",
                "reward_formula_id",
                "reward_formula_sha256",
                "reward_schema_id",
                "task_sha256",
                "training_design_sha256",
            }
        },
        "integrity": {
            "deterministic_reload": True,
            "explicit_missing_fields": [],
            "failure_receipt": None,
            "trace_index_sha256": trace_index_sha256,
        },
        "library_manifest_sha256": inputs["library_sha256"],
        "oracles": [
            {
                "file_sha256": inputs["oracle_file_sha256"],
                "oracle_sha256": inputs["oracle_canonical_sha256"],
            }
        ],
        "per_episode": episode_rows,
        "policy": {
            "checkpoints": policy_rows,
            "e1_receipt_sha256": inputs["e1_receipt_sha256"],
            "final_step": max(int(value["observed_transitions"]) for value in training_facts),
            "full_checkpoint_sha256": first["checkpoint_sha256"],
            "ppo_seed": first["ppo_seed"],
            "starting_expert_identity": inputs["starting_expert_identity"],
            "strict_export_sha256": first["strict_export_sha256"],
        },
        "prior_scientific_receipt": dict(prior_scientific_receipt),
        "reference_runtime": {"records": [dict(value) for value in reference_records]},
        "report_schema_id": REPORT_V2_SCHEMA_ID,
        "reward_runtime": reward,
        "runtime_fingerprint": runtime_fingerprint,
        "runtime_fingerprint_sha256": runtime_sha,
        "schema_version": 2,
        "scientific_receipt_schema_id": SCIENTIFIC_RECEIPT_SCHEMA_ID,
        "summary": {
            "arms": policy_rows,
            "distributions_by_arm": {},
            "distributions_by_cell": {
                cell: [row for row in episode_rows if row["cell"] == cell] for cell in UTILITY_CELLS
            },
            "hard_gates": {
                "task_success_endpoint": endpoint,
                "utility_gate": utility,
            },
            "policy_seeds": [
                seed.ppo_seed for seed in sorted(seeds, key=lambda item: item.ppo_seed)
            ],
            "step_zero_comparator": {
                str(seed.ppo_seed): dict(seed.step_zero_comparator) for seed in seeds
            },
        },
        "task_spec_sha256": inputs["task_sha256"],
        "trace_content_index": {"sha256": trace_index_sha256},
        "training": {
            "disk_bytes": None,
            "losses": {str(value["ppo_seed"]): value["losses"] for value in training_facts},
            "observed_transitions": observed,
            "peak_rss_bytes": None,
            "planned_transitions": planned,
            "rollouts": rollouts,
            "rsi_ledger_sha256": rsi_sha,
            "seed_facts": training_facts,
            "stream_counts": stream_counts,
            "throughput_steps_s": None,
            "telemetry_location": "telemetry_v1.json",
            "unfreeze_receipt_sha256": unfreeze_sha,
            "updates": updates,
            "wall_time_seconds": None,
        },
        "wall_time_seconds": None,
    }
    validate_cycle_report(report)
    canonical_json_bytes(report)
    return report


def _render_markdown(report: Mapping[str, object]) -> bytes:
    endpoint = report["summary"]["hard_gates"]["task_success_endpoint"]
    utility = report["summary"]["hard_gates"]["utility_gate"]
    lines = [
        f"# Experiment 003 Phase B cycle {report['cycle']}",
        "",
        "| status | current truth |",
        "|---|---|",
        f"| progress | {len(report['policy']['checkpoints'])} final checkpoint(s) recorded. |",
        "| bottleneck | This report is bounded utility evidence only; it does not establish causal reference use or humanoid competence. |",
        "| next step | Apply the frozen safety, task-success, and utility gates without post-hoc tuning. |",
        "",
        "| endpoint | value |",
        "|---|---:|",
        f"| safety gate | {endpoint['safety_gate']['passed']} |",
        f"| task successes | {endpoint['primary_endpoint']['successes']} / {endpoint['primary_endpoint']['total']} |",
        f"| utility family | {utility['family_passed']} |",
        "",
        f"Claim ceiling: `{report['claim_ceiling']}`",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def publish_report_v2(
    *,
    output_directory: Path,
    report: Mapping[str, object],
    telemetry: Mapping[str, object],
) -> ReportArtifacts:
    """Publish deterministic science first, then wall/host telemetry bound to it."""

    validated = validate_cycle_report(report)
    scientific_bytes = canonical_json_bytes(validated)
    output = Path(output_directory)
    scientific = publish_bytes_without_overwrite(
        output / "scientific_receipt_v2.json", scientific_bytes
    )
    telemetry_value = {
        "host": {
            "machine": platform.machine(),
            "system": platform.system(),
        },
        "measurements": dict(telemetry),
        "schema_version": 1,
        "scientific_receipt_sha256": scientific.sha256,
        "telemetry_schema_id": TELEMETRY_SCHEMA_ID,
    }
    telemetry_artifact = publish_bytes_without_overwrite(
        output / "telemetry_v1.json", canonical_json_bytes(telemetry_value)
    )
    markdown = publish_bytes_without_overwrite(output / "report_v2.md", _render_markdown(validated))
    return ReportArtifacts(scientific, telemetry_artifact, markdown)


__all__ = [
    "ProtectedEpisodeMetrics",
    "ReportArtifacts",
    "SeedReportFacts",
    "build_scientific_receipt",
    "exact_binomial_interval",
    "publish_report_v2",
    "task_success_endpoint",
    "utility_gate",
]
