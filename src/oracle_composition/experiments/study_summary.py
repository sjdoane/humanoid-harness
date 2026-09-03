"""Fail-closed aggregation for the five-seed static positive-control study.

Evaluation seeds are paired repeated measures within a checkpoint. Training
seeds are the only independent units used for uncertainty summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from numbers import Real
from pathlib import Path
from typing import Any

from oracle_composition.tracking import TrackingRewardConfig
from oracle_composition.tracking import reward as tracking_reward_module

from . import execution as execution_module
from . import fixed_reference as fixed_reference_module
from . import protected_evaluator as protected_evaluator_module
from . import squashed_policy as squashed_policy_module
from .execution import (
    PROTECTED_METRICS_STATE_SOURCE,
    _validated_training_receipt,
    emit_json_without_overwrite,
    load_execution_manifest,
)
from .fixed_reference import (
    CANONICAL_EXECUTION_CALLABLE_AUTHORITY,
    ExperimentContractError,
    FixedReferenceStudyDesign,
    FrozenExecutionManifest,
    load_study_design,
    read_bounded_json_object,
    sha256_file,
)
from .protected_evaluator import STATION_KEEPING_ORIGIN_SOURCE, EpisodeMetrics

MAX_CRITERIA_BYTES = 256 * 1024
MAX_EVALUATION_RECEIPT_BYTES = 16 * 1024 * 1024

_REWARD_SCALE_OCCUPANCY_BOUND_SPEC = (
    (
        "root_height_rmse_m",
        "m",
        "TrackingRewardConfig.root_height_scale_m",
        "root_height_scale_m",
    ),
    (
        "root_orientation_rmse_rad",
        "rad",
        "TrackingRewardConfig.root_orientation_scale_rad",
        "root_orientation_scale_rad",
    ),
    (
        "root_linear_velocity_rmse_m_s",
        "m/s",
        "TrackingRewardConfig.root_linear_velocity_scale_m_s",
        "root_linear_velocity_scale_m_s",
    ),
    (
        "root_angular_velocity_rmse_rad_s",
        "rad/s",
        "TrackingRewardConfig.root_angular_velocity_scale_rad_s",
        "root_angular_velocity_scale_rad_s",
    ),
    (
        "joint_position_rmse_rad",
        "rad",
        "TrackingRewardConfig.joint_position_scale_rad",
        "joint_position_scale_rad",
    ),
    (
        "joint_velocity_rmse_rad_s",
        "rad/s",
        "TrackingRewardConfig.joint_velocity_scale_rad_s",
        "joint_velocity_scale_rad_s",
    ),
    (
        "joint_position_max_abs_rad",
        "rad",
        "TrackingRewardConfig.joint_position_scale_rad",
        "joint_position_scale_rad",
    ),
    (
        "joint_velocity_max_abs_rad_s",
        "rad/s",
        "TrackingRewardConfig.joint_velocity_scale_rad_s",
        "joint_velocity_scale_rad_s",
    ),
)

_CONTROL_PERIOD_SECONDS = 0.015
_MAX_NORMALIZED_ACTION_DELTA_RATE_PER_S = 2.0 / _CONTROL_PERIOD_SECONDS
_RAW_TORQUE_CAPACITY_RMS_N_M = 56.0461994303649
_RAW_TORQUE_CAPACITY_MAX_N_M = 120.0

# These are ABI/action bounds or an explicit prohibited-contact rule. They are
# not naturalness thresholds. Field order, units, basis, and maxima are frozen
# so an arbitrary post-data metric cannot complete the decision rule.
_BOUNDED_CONTROL_AND_PROHIBITED_FLOOR_CONTACT_THRESHOLD_SPEC = (
    (
        "normalized_policy_action_rms",
        1.0,
        "1",
        "normalized_action_box[-1,1]_rms_upper_bound",
    ),
    (
        "normalized_policy_action_max_abs",
        1.0,
        "1",
        "normalized_action_box[-1,1]_absolute_upper_bound",
    ),
    (
        "normalized_policy_action_delta_rate_rms_per_s",
        _MAX_NORMALIZED_ACTION_DELTA_RATE_PER_S,
        "s^-1",
        "two_normalized_action_units/control_period_seconds[0.015]",
    ),
    (
        "normalized_policy_action_delta_rate_max_abs_per_s",
        _MAX_NORMALIZED_ACTION_DELTA_RATE_PER_S,
        "s^-1",
        "two_normalized_action_units/control_period_seconds[0.015]",
    ),
    (
        "generalized_actuator_torque_rms_n_m",
        _RAW_TORQUE_CAPACITY_RMS_N_M,
        "N*m",
        "Humanoid-v5_rms_abs(gear*ctrl_limit)=56.0461994303649",
    ),
    (
        "generalized_actuator_torque_max_abs_n_m",
        _RAW_TORQUE_CAPACITY_MAX_N_M,
        "N*m",
        "Humanoid-v5_max_abs(gear*ctrl_limit)=120",
    ),
    (
        "generalized_actuator_torque_normalized_rms",
        1.0,
        "1",
        "per_joint_abs(qfrc_actuator)/abs(gear*ctrl_limit)_rms_upper_bound",
    ),
    (
        "generalized_actuator_torque_normalized_max_abs",
        1.0,
        "1",
        "per_joint_abs(qfrc_actuator)/abs(gear*ctrl_limit)_upper_bound",
    ),
    (
        "forbidden_floor_contact_count",
        0.0,
        "active_contact_samples",
        "static_stand_permits_floor_contact_only_for_left_foot_and_right_foot",
    ),
    (
        "forbidden_floor_contact_fraction",
        0.0,
        "1",
        "forbidden_floor_contact_samples/all_floor_contact_samples",
    ),
    (
        "forbidden_floor_contact_step_fraction",
        0.0,
        "1",
        "episode_steps_with_forbidden_floor_contact/observed_episode_steps",
    ),
    (
        "forbidden_floor_contact_peak_force_n",
        0.0,
        "N",
        "zero_force_required_when_forbidden_floor_contact_count_is_zero",
    ),
)

_REQUIRED_DESCRIPTIVE_METRIC_SPEC = (
    (
        "floor_contact_peak_normal_force_n",
        "N",
        "direct_MuJoCo_peak_normal_force_over_all_floor_contacts",
    ),
    (
        "joint_jerk_rms_rad_s3",
        "rad/s^3",
        "direct_qacc_finite_difference_over_control_period_episode_rms",
    ),
    (
        "joint_jerk_max_abs_rad_s3",
        "rad/s^3",
        "direct_qacc_finite_difference_over_control_period_episode_max_abs",
    ),
)

_NATURALNESS_CALIBRATION_STATUS = "not_calibrated_no_independent_neutral_controller"

_STATION_KEEPING_REQUIREMENT_STATUS = (
    "blocked_missing_independently_justified_prebehavioral_thresholds"
)
_UNTHRESHOLDED_STATION_KEEPING_SPEC = (
    (
        "root_horizontal_displacement_max_m",
        "m",
        "max_norm_xy_of_direct_root_qpos_minus_episode_initial_root_qpos",
        "not_defined_requires_predata_support_region_basis",
    ),
    (
        "root_linear_velocity_x_max_abs_m_s",
        "m/s",
        "episode_max_abs_direct_world_root_linear_velocity_x",
        "not_defined_requires_predata_axis_specific_basis",
    ),
    (
        "root_linear_velocity_y_max_abs_m_s",
        "m/s",
        "episode_max_abs_direct_world_root_linear_velocity_y",
        "not_defined_requires_predata_axis_specific_basis",
    ),
    (
        "root_linear_velocity_z_max_abs_m_s",
        "m/s",
        "episode_max_abs_direct_world_root_linear_velocity_z",
        "not_defined_requires_predata_axis_specific_basis",
    ),
)

_REQUIRED_RECEIPT_FIELDS = {
    "schema_version",
    "evidence_purpose",
    "completion_status",
    "claim_ceiling",
    "design_sha256",
    "design_file_sha256",
    "study_criteria_semantic_sha256",
    "study_criteria_file_sha256",
    "execution_manifest_sha256",
    "execution_manifest_file_sha256",
    "runtime_sha256",
    "execution_callable_authority",
    "actuator_gear_by_joint",
    "generalized_actuator_torque_capacity_n_m",
    "source_tree_sha256",
    "checkpoint_receipt_file_sha256",
    "checkpoint_sha256",
    "train_seed",
    "evaluation_reference_sha256",
    "evaluation_reference_schema_sha256",
    "evaluation_seeds_requested",
    "evaluation_seeds_observed",
    "max_episode_steps",
    "deterministic_actions",
    "protected_metrics_state_source",
    "station_keeping_origin_source",
    "reward_telemetry_used_for_objective_metrics",
    "tracking_return_role",
    "reference_timing_scope",
    "episodes",
    "complete",
    "automatic_promotion",
    "behavioral_claim",
}

_STRUCTURAL_EPISODE_FIELDS = {
    "evaluation_seed",
    "observed_steps",
    "requested_steps",
    "survived_full_horizon",
    "first_collapse_step",
    "collapse_fraction",
}
_BASE_DESCRIPTIVE_EPISODE_FIELDS = {"tracking_return"}


@dataclass(frozen=True, slots=True)
class MaximumThreshold:
    """One pre-data maximum applied to an episode-level protected metric."""

    field: str
    maximum: float
    unit: str
    basis: str

    def __post_init__(self) -> None:
        for name in ("field", "unit", "basis"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ExperimentContractError(f"threshold {name} must be non-empty")
        maximum = _finite_number(self.maximum, field=f"threshold {self.field}.maximum")
        if maximum < 0.0:
            raise ExperimentContractError(f"threshold {self.field}.maximum cannot be negative")
        object.__setattr__(self, "maximum", maximum)

    def to_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "maximum": self.maximum,
            "unit": self.unit,
            "basis": self.basis,
        }


@dataclass(frozen=True, slots=True)
class RequiredDescriptiveMetric:
    """A protected metric required for context but excluded from pass/fail."""

    field: str
    unit: str
    basis: str
    calibration_status: str

    def __post_init__(self) -> None:
        for name in ("field", "unit", "basis", "calibration_status"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ExperimentContractError(
                    f"required descriptive metric {name} must be non-empty"
                )

    def to_dict(self) -> dict[str, str]:
        return {
            "field": self.field,
            "unit": self.unit,
            "basis": self.basis,
            "calibration_status": self.calibration_status,
        }


@dataclass(frozen=True, slots=True)
class BlockedMetricRequirement:
    """A measured station-keeping field lacking a justified pass/fail threshold."""

    field: str
    unit: str
    basis: str
    threshold_status: str

    def __post_init__(self) -> None:
        for name in ("field", "unit", "basis", "threshold_status"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ExperimentContractError(
                    f"blocked metric requirement {name} must be non-empty"
                )

    def to_dict(self) -> dict[str, str]:
        return {
            "field": self.field,
            "unit": self.unit,
            "basis": self.basis,
            "threshold_status": self.threshold_status,
        }


@dataclass(frozen=True, slots=True)
class StaticStudyCriteria:
    """Frozen tracking/conformance rule with naturalness excluded."""

    schema_version: int
    criteria_id: str
    status: str
    experiment_id: str
    claim_ceiling: str
    reward_scale_occupancy_bounds: tuple[MaximumThreshold, ...]
    guardrail_status: str
    guardrail_thresholds: tuple[MaximumThreshold, ...]
    required_descriptive_metrics: tuple[RequiredDescriptiveMetric, ...]
    naturalness_calibration_status: str
    station_keeping_requirement_status: str
    unthresholded_station_keeping_metrics: tuple[BlockedMetricRequirement, ...]
    require_survived_full_horizon: bool
    maximum_collapse_fraction: float
    require_all_metric_thresholds: bool
    required_evaluation_episodes: int
    minimum_conforming_episodes: int
    required_independent_train_seeds: int
    minimum_conforming_checkpoints: int
    uncertainty_method: str
    confidence_level: float
    resample_size: int
    expected_resample_count: int
    uncertainty_interval: str
    automatic_promotion: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != 1
        ):
            raise ExperimentContractError("criteria schema_version must be 1")
        expected_strings = {
            "criteria_id": "static_positive_control_criteria/v1",
            "status": "frozen_prebehavioral_candidate",
            "experiment_id": "001_humanoid_fixed_reference",
            "claim_ceiling": "static_tracking_feasibility",
            "uncertainty_method": "exhaustive_nonparametric_seed_bootstrap/v1",
            "uncertainty_interval": "percentile_linear",
        }
        mismatches = [
            field
            for field, expected in expected_strings.items()
            if getattr(self, field) != expected
        ]
        if mismatches:
            raise ExperimentContractError(
                "criteria fixed fields mismatch: " + ", ".join(mismatches)
            )
        if self.guardrail_status != (
            "complete_bounded_control_and_prohibited_floor_contact_conformance_prebehavioral"
        ):
            raise ExperimentContractError(
                "guardrail status must identify complete bounded-control and "
                "prohibited-floor-contact conformance"
            )
        if self.naturalness_calibration_status != _NATURALNESS_CALIBRATION_STATUS:
            raise ExperimentContractError(
                "naturalness calibration status is not the frozen uncalibrated state"
            )
        if self.station_keeping_requirement_status != _STATION_KEEPING_REQUIREMENT_STATUS:
            raise ExperimentContractError(
                "station-keeping requirement status is not the frozen blocked state"
            )
        if self.require_survived_full_horizon is not True:
            raise ExperimentContractError("episode rule must require full-horizon survival")
        if self.require_all_metric_thresholds is not True:
            raise ExperimentContractError("episode rule must require every metric threshold")
        collapse = _finite_number(
            self.maximum_collapse_fraction,
            field="maximum_collapse_fraction",
        )
        if collapse != 0.0:
            raise ExperimentContractError("maximum_collapse_fraction must be exactly zero")
        object.__setattr__(self, "maximum_collapse_fraction", collapse)
        for field in (
            "required_evaluation_episodes",
            "minimum_conforming_episodes",
            "required_independent_train_seeds",
            "minimum_conforming_checkpoints",
            "resample_size",
            "expected_resample_count",
        ):
            _positive_int(getattr(self, field), field=field)
        exact_design = (
            self.required_evaluation_episodes,
            self.minimum_conforming_episodes,
            self.required_independent_train_seeds,
            self.minimum_conforming_checkpoints,
            self.resample_size,
            self.expected_resample_count,
        )
        if exact_design != (20, 19, 5, 4, 5, 3125):
            raise ExperimentContractError(
                "criteria must preserve the exact 20/19 evaluation and 5/4 study design"
            )
        if self.minimum_conforming_episodes > self.required_evaluation_episodes:
            raise ExperimentContractError("minimum conforming episodes exceeds required episodes")
        if self.minimum_conforming_checkpoints > self.required_independent_train_seeds:
            raise ExperimentContractError("minimum conforming checkpoints exceeds required seeds")
        confidence = _finite_number(self.confidence_level, field="confidence_level")
        if confidence != 0.95:
            raise ExperimentContractError("confidence_level must be exactly 0.95")
        object.__setattr__(self, "confidence_level", confidence)
        if self.resample_size != self.required_independent_train_seeds:
            raise ExperimentContractError(
                "bootstrap resample size must equal independent seed count"
            )
        if self.expected_resample_count != (
            self.required_independent_train_seeds**self.resample_size
        ):
            raise ExperimentContractError("bootstrap resample count is inconsistent")
        if self.automatic_promotion is not False:
            raise ExperimentContractError("criteria cannot enable automatic promotion")

        fields = [
            threshold.field
            for threshold in (
                *self.reward_scale_occupancy_bounds,
                *self.guardrail_thresholds,
            )
        ]
        fields.extend(metric.field for metric in self.required_descriptive_metrics)
        fields.extend(metric.field for metric in self.unthresholded_station_keeping_metrics)
        if len(fields) != len(set(fields)):
            raise ExperimentContractError("criteria metric fields must be unique")
        _validate_reward_scale_occupancy_bounds(self.reward_scale_occupancy_bounds)
        _validate_bounded_control_and_prohibited_floor_contact_thresholds(self.guardrail_thresholds)
        _validate_required_descriptive_metrics(self.required_descriptive_metrics)
        _validate_unthresholded_station_keeping_metrics(self.unthresholded_station_keeping_metrics)

    @property
    def bounded_control_and_prohibited_floor_contact_conformance_complete(self) -> bool:
        return self.guardrail_status == (
            "complete_bounded_control_and_prohibited_floor_contact_conformance_prebehavioral"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "criteria_id": self.criteria_id,
            "status": self.status,
            "experiment_id": self.experiment_id,
            "claim_ceiling": self.claim_ceiling,
            "reward_scale_derived_occupancy_bounds": [
                threshold.to_dict() for threshold in self.reward_scale_occupancy_bounds
            ],
            "guardrail_thresholds": {
                "status": self.guardrail_status,
                "thresholds": [threshold.to_dict() for threshold in self.guardrail_thresholds],
                "required_descriptive_metrics": [
                    metric.to_dict() for metric in self.required_descriptive_metrics
                ],
                "naturalness_calibration_status": self.naturalness_calibration_status,
            },
            "episode_rule": {
                "require_survived_full_horizon": self.require_survived_full_horizon,
                "maximum_collapse_fraction": self.maximum_collapse_fraction,
                "require_all_metric_thresholds": self.require_all_metric_thresholds,
            },
            "blocked_feasibility_requirements": {
                "status": self.station_keeping_requirement_status,
                "unthresholded_station_keeping_metrics": [
                    metric.to_dict() for metric in self.unthresholded_station_keeping_metrics
                ],
            },
            "checkpoint_rule": {
                "required_evaluation_episodes": self.required_evaluation_episodes,
                "minimum_conforming_episodes": self.minimum_conforming_episodes,
            },
            "study_rule": {
                "required_independent_train_seeds": self.required_independent_train_seeds,
                "minimum_conforming_checkpoints": self.minimum_conforming_checkpoints,
            },
            "uncertainty": {
                "method": self.uncertainty_method,
                "confidence_level": self.confidence_level,
                "resample_size": self.resample_size,
                "expected_resample_count": self.expected_resample_count,
                "interval": self.uncertainty_interval,
            },
            "automatic_promotion": self.automatic_promotion,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ExperimentContractError(f"{field} must be a finite number")
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExperimentContractError(f"{field} must be a finite number") from exc
    if not math.isfinite(resolved):
        raise ExperimentContractError(f"{field} must be a finite number")
    return resolved


def _positive_int(value: object, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ExperimentContractError(f"{field} must be a positive integer")
    return value


def _sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"{field} must be a lowercase SHA-256")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], *, field: str) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ExperimentContractError(
            f"{field} keys mismatch: missing={missing!r}, extra={extra!r}"
        )


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExperimentContractError(f"value is not finite canonical JSON: {exc}") from exc


def _read_json_object(path: Path, *, maximum_bytes: int, artifact: str) -> dict[str, Any]:
    return read_bounded_json_object(
        path,
        maximum_bytes=maximum_bytes,
        artifact=artifact,
    )


def _threshold_from_dict(value: object, *, field: str) -> MaximumThreshold:
    if not isinstance(value, dict):
        raise ExperimentContractError(f"{field} must be an object")
    _exact_keys(value, {"field", "maximum", "unit", "basis"}, field=field)
    return MaximumThreshold(
        field=value["field"],
        maximum=value["maximum"],
        unit=value["unit"],
        basis=value["basis"],
    )


def _required_metric_from_dict(
    value: object,
    *,
    field: str,
) -> RequiredDescriptiveMetric:
    if not isinstance(value, dict):
        raise ExperimentContractError(f"{field} must be an object")
    _exact_keys(
        value,
        {"field", "unit", "basis", "calibration_status"},
        field=field,
    )
    return RequiredDescriptiveMetric(
        field=value["field"],
        unit=value["unit"],
        basis=value["basis"],
        calibration_status=value["calibration_status"],
    )


def _blocked_metric_from_dict(
    value: object,
    *,
    field: str,
) -> BlockedMetricRequirement:
    if not isinstance(value, dict):
        raise ExperimentContractError(f"{field} must be an object")
    _exact_keys(
        value,
        {"field", "unit", "basis", "threshold_status"},
        field=field,
    )
    return BlockedMetricRequirement(
        field=value["field"],
        unit=value["unit"],
        basis=value["basis"],
        threshold_status=value["threshold_status"],
    )


def _validate_reward_scale_occupancy_bounds(
    thresholds: tuple[MaximumThreshold, ...],
) -> None:
    reward = TrackingRewardConfig()
    expected_fields = tuple(spec[0] for spec in _REWARD_SCALE_OCCUPANCY_BOUND_SPEC)
    if tuple(threshold.field for threshold in thresholds) != expected_fields:
        raise ExperimentContractError("reward-scale occupancy fields or order changed")
    for threshold, (_, unit, basis, reward_field) in zip(
        thresholds,
        _REWARD_SCALE_OCCUPANCY_BOUND_SPEC,
        strict=True,
    ):
        expected_maximum = float(getattr(reward, reward_field))
        if (
            threshold.unit != unit
            or threshold.basis != basis
            or threshold.maximum != expected_maximum
        ):
            raise ExperimentContractError(
                f"reward-scale occupancy bound {threshold.field!r} is not tied to its scale"
            )


def _validate_bounded_control_and_prohibited_floor_contact_thresholds(
    thresholds: tuple[MaximumThreshold, ...],
) -> None:
    observed = tuple(
        (threshold.field, threshold.maximum, threshold.unit, threshold.basis)
        for threshold in thresholds
    )
    if observed != _BOUNDED_CONTROL_AND_PROHIBITED_FLOOR_CONTACT_THRESHOLD_SPEC:
        raise ExperimentContractError(
            "bounded-control/prohibited-floor-contact thresholds must match the exact "
            "frozen field/unit/basis/maxima set"
        )


def _validate_required_descriptive_metrics(
    metrics: tuple[RequiredDescriptiveMetric, ...],
) -> None:
    observed = tuple((metric.field, metric.unit, metric.basis) for metric in metrics)
    if observed != _REQUIRED_DESCRIPTIVE_METRIC_SPEC:
        raise ExperimentContractError(
            "required descriptive metrics must match the exact impact/jerk field/unit/basis set"
        )
    if any(metric.calibration_status != _NATURALNESS_CALIBRATION_STATUS for metric in metrics):
        raise ExperimentContractError(
            "descriptive naturalness metrics must remain explicitly uncalibrated"
        )


def _validate_unthresholded_station_keeping_metrics(
    metrics: tuple[BlockedMetricRequirement, ...],
) -> None:
    observed = tuple(
        (metric.field, metric.unit, metric.basis, metric.threshold_status) for metric in metrics
    )
    if observed != _UNTHRESHOLDED_STATION_KEEPING_SPEC:
        raise ExperimentContractError(
            "unthresholded station-keeping metrics must match the exact frozen set"
        )


def load_static_study_criteria(path: Path) -> StaticStudyCriteria:
    """Load and validate the exact frozen prebehavioral-candidate criteria schema."""

    value = _read_json_object(
        path.resolve(),
        maximum_bytes=MAX_CRITERIA_BYTES,
        artifact="static study criteria",
    )
    _exact_keys(
        value,
        {
            "schema_version",
            "criteria_id",
            "status",
            "experiment_id",
            "claim_ceiling",
            "reward_scale_derived_occupancy_bounds",
            "guardrail_thresholds",
            "blocked_feasibility_requirements",
            "episode_rule",
            "checkpoint_rule",
            "study_rule",
            "uncertainty",
            "automatic_promotion",
        },
        field="static study criteria",
    )
    nested_specs = {
        "guardrail_thresholds": {
            "status",
            "thresholds",
            "required_descriptive_metrics",
            "naturalness_calibration_status",
        },
        "blocked_feasibility_requirements": {
            "status",
            "unthresholded_station_keeping_metrics",
        },
        "episode_rule": {
            "require_survived_full_horizon",
            "maximum_collapse_fraction",
            "require_all_metric_thresholds",
        },
        "checkpoint_rule": {
            "required_evaluation_episodes",
            "minimum_conforming_episodes",
        },
        "study_rule": {
            "required_independent_train_seeds",
            "minimum_conforming_checkpoints",
        },
        "uncertainty": {
            "method",
            "confidence_level",
            "resample_size",
            "expected_resample_count",
            "interval",
        },
    }
    for field, keys in nested_specs.items():
        if not isinstance(value[field], dict):
            raise ExperimentContractError(f"criteria {field} must be an object")
        _exact_keys(value[field], keys, field=f"criteria {field}")
    if not isinstance(value["reward_scale_derived_occupancy_bounds"], list):
        raise ExperimentContractError("reward_scale_derived_occupancy_bounds must be an array")
    guardrail = value["guardrail_thresholds"]
    if not isinstance(guardrail["thresholds"], list):
        raise ExperimentContractError("guardrail thresholds must be an array")
    if not isinstance(guardrail["required_descriptive_metrics"], list):
        raise ExperimentContractError("guardrail required_descriptive_metrics must be an array")
    blocked = value["blocked_feasibility_requirements"]
    if not isinstance(blocked["unthresholded_station_keeping_metrics"], list):
        raise ExperimentContractError(
            "blocked unthresholded_station_keeping_metrics must be an array"
        )
    reward_scale_bounds = tuple(
        _threshold_from_dict(
            item,
            field=f"reward_scale_derived_occupancy_bounds[{index}]",
        )
        for index, item in enumerate(value["reward_scale_derived_occupancy_bounds"])
    )
    guardrails = tuple(
        _threshold_from_dict(item, field=f"guardrail thresholds[{index}]")
        for index, item in enumerate(guardrail["thresholds"])
    )
    descriptive_metrics = tuple(
        _required_metric_from_dict(
            item,
            field=f"guardrail required_descriptive_metrics[{index}]",
        )
        for index, item in enumerate(guardrail["required_descriptive_metrics"])
    )
    unthresholded_station_keeping_metrics = tuple(
        _blocked_metric_from_dict(
            item,
            field=f"blocked unthresholded_station_keeping_metrics[{index}]",
        )
        for index, item in enumerate(blocked["unthresholded_station_keeping_metrics"])
    )
    episode = value["episode_rule"]
    checkpoint = value["checkpoint_rule"]
    study = value["study_rule"]
    uncertainty = value["uncertainty"]
    return StaticStudyCriteria(
        schema_version=value["schema_version"],
        criteria_id=value["criteria_id"],
        status=value["status"],
        experiment_id=value["experiment_id"],
        claim_ceiling=value["claim_ceiling"],
        reward_scale_occupancy_bounds=reward_scale_bounds,
        guardrail_status=guardrail["status"],
        guardrail_thresholds=guardrails,
        required_descriptive_metrics=descriptive_metrics,
        naturalness_calibration_status=guardrail["naturalness_calibration_status"],
        station_keeping_requirement_status=blocked["status"],
        unthresholded_station_keeping_metrics=(unthresholded_station_keeping_metrics),
        require_survived_full_horizon=episode["require_survived_full_horizon"],
        maximum_collapse_fraction=episode["maximum_collapse_fraction"],
        require_all_metric_thresholds=episode["require_all_metric_thresholds"],
        required_evaluation_episodes=checkpoint["required_evaluation_episodes"],
        minimum_conforming_episodes=checkpoint["minimum_conforming_episodes"],
        required_independent_train_seeds=study["required_independent_train_seeds"],
        minimum_conforming_checkpoints=study["minimum_conforming_checkpoints"],
        uncertainty_method=uncertainty["method"],
        confidence_level=uncertainty["confidence_level"],
        resample_size=uncertainty["resample_size"],
        expected_resample_count=uncertainty["expected_resample_count"],
        uncertainty_interval=uncertainty["interval"],
        automatic_promotion=value["automatic_promotion"],
    )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ExperimentContractError("cannot summarize an empty sequence")
    return math.fsum(values) / len(values)


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ExperimentContractError("cannot compute an empty quantile")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def exhaustive_seed_bootstrap(
    values: Sequence[float],
    *,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Enumerate every n^n nonparametric resample of independent seed values."""

    resolved = tuple(_finite_number(value, field="seed-level value") for value in values)
    if len(resolved) != 5:
        raise ExperimentContractError("static study bootstrap requires exactly five seed values")
    confidence = _finite_number(confidence_level, field="confidence_level")
    if confidence != 0.95:
        raise ExperimentContractError("static study bootstrap confidence must be 0.95")
    resampled = sorted(
        _mean(tuple(resolved[index] for index in indices))
        for indices in itertools.product(range(len(resolved)), repeat=len(resolved))
    )
    tail = (1.0 - confidence) / 2.0
    return {
        "method": "exhaustive_nonparametric_seed_bootstrap/v1",
        "independent_unit": "train_seed",
        "independent_n": len(resolved),
        "resample_size": len(resolved),
        "resample_count": len(resampled),
        "confidence_level": confidence,
        "interval": "percentile_linear",
        "point_estimate_seed_mean": _mean(resolved),
        "interval_lower": _linear_quantile(resampled, tail),
        "interval_upper": _linear_quantile(resampled, 1.0 - tail),
        "interpretation": "descriptive_seed_sensitivity_not_hypothesis_test",
    }


def _descriptives(values: Sequence[float]) -> dict[str, float]:
    resolved = tuple(_finite_number(value, field="descriptive value") for value in values)
    if not resolved:
        raise ExperimentContractError("cannot summarize an empty metric")
    return {
        "mean": _mean(resolved),
        "median": float(statistics.median(resolved)),
        "minimum": min(resolved),
        "maximum": max(resolved),
    }


def _validate_design_and_criteria(
    design: FixedReferenceStudyDesign,
    criteria: StaticStudyCriteria,
    *,
    criteria_path: Path,
) -> None:
    design.assert_locked_for_behavior()
    if design.experiment_id != criteria.experiment_id:
        raise ExperimentContractError("criteria do not apply to this experiment")
    if design.claim_ceiling.value != criteria.claim_ceiling:
        raise ExperimentContractError("criteria claim ceiling differs from the study design")
    if len(design.train_seeds) != criteria.required_independent_train_seeds:
        raise ExperimentContractError("design train-seed count differs from the criteria")
    if len(design.evaluation_seeds) != criteria.required_evaluation_episodes:
        raise ExperimentContractError("design evaluation-seed count differs from the criteria")
    if design.study_criteria_semantic_sha256 != criteria.sha256:
        raise ExperimentContractError(
            "criteria semantic SHA-256 does not match the pretraining study design"
        )
    if design.study_criteria_file_sha256 != sha256_file(criteria_path):
        raise ExperimentContractError(
            "criteria file SHA-256 does not match the pretraining study design"
        )


def _validate_aggregation_source_hashes(manifest: FrozenExecutionManifest) -> None:
    expected_modules = {
        "execution_source_sha256": execution_module,
        "experiment_contract_source_sha256": fixed_reference_module,
        "tracking_reward_source_sha256": tracking_reward_module,
        "evaluator_source_sha256": protected_evaluator_module,
        "policy_source_sha256": squashed_policy_module,
    }
    mismatches = []
    for runtime_field, module in expected_modules.items():
        source_path = Path(str(getattr(module, "__file__", "")))
        observed = _local_sha256_file(source_path)
        if observed != getattr(manifest.runtime, runtime_field):
            mismatches.append(runtime_field)
    observed_summary = _local_sha256_file(Path(__file__))
    if observed_summary != manifest.runtime.study_summary_source_sha256:
        mismatches.append("study_summary_source_sha256")
    if _local_source_tree_sha256() != manifest.runtime.source_tree_sha256:
        mismatches.append("source_tree_sha256")
    if mismatches:
        raise ExperimentContractError(
            "aggregation source differs from the frozen runtime: " + ", ".join(mismatches)
        )


def _local_sha256_file(path: Path) -> str:
    """Hash source bytes without trusting an imported hashing helper."""

    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise ExperimentContractError(f"cannot hash aggregation source {path}: {exc}") from exc


def _local_source_tree_sha256() -> str:
    """Hash the bounded on-disk package tree without trusting imported helper code."""

    package_root = Path(__file__).resolve().parents[1]
    if package_root.is_symlink() or not package_root.is_dir():
        raise ExperimentContractError("aggregation package source root must be a real directory")
    paths = sorted(
        package_root.rglob("*.py"), key=lambda path: path.relative_to(package_root).as_posix()
    )
    if not paths or len(paths) > 1024:
        raise ExperimentContractError(
            "aggregation package source tree has an invalid Python file count"
        )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ExperimentContractError("aggregation package source tree contains a linked file")
        try:
            with path.open("rb") as stream:
                encoded = stream.read(4 * 1024 * 1024 + 1)
        except OSError as exc:
            raise ExperimentContractError(f"cannot read aggregation source {path}: {exc}") from exc
        if len(encoded) > 4 * 1024 * 1024:
            raise ExperimentContractError("aggregation package source file exceeds size limit")
        total_bytes += len(encoded)
        if total_bytes > 32 * 1024 * 1024:
            raise ExperimentContractError("aggregation package source tree exceeds size limit")
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _validate_receipt_root(
    receipt: dict[str, Any],
    *,
    design: FixedReferenceStudyDesign,
    design_file_sha256: str,
    manifest: FrozenExecutionManifest,
    manifest_file_sha256: str,
) -> None:
    missing = sorted(_REQUIRED_RECEIPT_FIELDS - set(receipt))
    if missing:
        raise ExperimentContractError(f"evaluation receipt missing fields: {missing!r}")
    if not isinstance(receipt["schema_version"], int) or isinstance(
        receipt["schema_version"], bool
    ):
        raise ExperimentContractError("evaluation receipt schema_version must be integer 1")
    for field in (
        "deterministic_actions",
        "reward_telemetry_used_for_objective_metrics",
        "complete",
        "automatic_promotion",
    ):
        if not isinstance(receipt[field], bool):
            raise ExperimentContractError(f"evaluation receipt {field} must be boolean")
    if not isinstance(receipt["max_episode_steps"], int) or isinstance(
        receipt["max_episode_steps"], bool
    ):
        raise ExperimentContractError("evaluation receipt max_episode_steps must be an integer")
    for field in ("evaluation_seeds_requested", "evaluation_seeds_observed"):
        seeds = receipt[field]
        if not isinstance(seeds, list) or any(
            not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds
        ):
            raise ExperimentContractError(
                f"evaluation receipt {field} must contain only integer seeds"
            )
    expected = {
        "schema_version": 1,
        "evidence_purpose": "behavioral_evaluation",
        "completion_status": "complete_measurements_no_promotion",
        "claim_ceiling": design.claim_ceiling.value,
        "design_sha256": design.sha256,
        "design_file_sha256": design_file_sha256,
        "study_criteria_semantic_sha256": design.study_criteria_semantic_sha256,
        "study_criteria_file_sha256": design.study_criteria_file_sha256,
        "execution_manifest_sha256": manifest.sha256,
        "execution_manifest_file_sha256": manifest_file_sha256,
        "runtime_sha256": manifest.runtime.sha256,
        "execution_callable_authority": CANONICAL_EXECUTION_CALLABLE_AUTHORITY,
        "actuator_gear_by_joint": list(manifest.runtime.actuator_gear_by_joint),
        "generalized_actuator_torque_capacity_n_m": list(
            manifest.runtime.generalized_actuator_torque_capacity_n_m
        ),
        "source_tree_sha256": manifest.runtime.source_tree_sha256,
        "evaluation_reference_sha256": manifest.runtime.reference_content_sha256,
        "evaluation_reference_schema_sha256": manifest.runtime.reference_schema_sha256,
        "evaluation_seeds_requested": list(design.evaluation_seeds),
        "evaluation_seeds_observed": list(design.evaluation_seeds),
        "max_episode_steps": design.max_episode_steps,
        "deterministic_actions": True,
        "protected_metrics_state_source": PROTECTED_METRICS_STATE_SOURCE,
        "station_keeping_origin_source": STATION_KEEPING_ORIGIN_SOURCE,
        "reward_telemetry_used_for_objective_metrics": False,
        "tracking_return_role": "environment_reward_diagnostic_only",
        "reference_timing_scope": "static_reference_only",
        "complete": True,
        "automatic_promotion": False,
        "behavioral_claim": None,
    }
    mismatches = [field for field, value in expected.items() if receipt.get(field) != value]
    if mismatches:
        raise ExperimentContractError(
            "evaluation receipt does not match the frozen study: " + ", ".join(mismatches)
        )
    for field in (
        "checkpoint_receipt_file_sha256",
        "checkpoint_sha256",
    ):
        _sha256(receipt[field], field=field)


def _validate_episode(
    episode: dict[str, Any],
    *,
    evaluation_seed: int,
    max_episode_steps: int,
    control_period_seconds: float,
    reward_scale_fields: tuple[str, ...],
    required_guardrail_fields: tuple[str, ...],
) -> None:
    required = (
        _STRUCTURAL_EPISODE_FIELDS
        | _BASE_DESCRIPTIVE_EPISODE_FIELDS
        | set((*reward_scale_fields, *required_guardrail_fields))
    )
    missing = sorted(required - set(episode))
    if missing:
        raise ExperimentContractError(f"evaluation episode missing fields: {missing!r}")
    if episode["evaluation_seed"] != evaluation_seed:
        raise ExperimentContractError(
            "episode evaluation_seed is missing, duplicated, or reordered"
        )
    for field in ("observed_steps", "requested_steps"):
        if episode[field] != max_episode_steps:
            raise ExperimentContractError(f"episode {field} is incomplete")
    episode_metric_fields = tuple(field.name for field in dataclass_fields(EpisodeMetrics))
    missing_metric_fields = sorted(set(episode_metric_fields) - set(episode))
    if missing_metric_fields:
        raise ExperimentContractError(
            f"evaluation episode missing exact metric fields: {missing_metric_fields!r}"
        )
    try:
        validated_metrics = EpisodeMetrics(
            **{field: episode[field] for field in episode_metric_fields},
        )
    except TypeError as exc:  # defensive against dataclass contract drift
        raise ExperimentContractError(f"evaluation episode schema is invalid: {exc}") from exc
    if validated_metrics.control_period_seconds != control_period_seconds:
        raise ExperimentContractError(
            "episode control_period_seconds differs from the frozen runtime"
        )
    if not isinstance(episode["survived_full_horizon"], bool):
        raise ExperimentContractError("survived_full_horizon must be boolean")
    collapse = _finite_number(episode["collapse_fraction"], field="collapse_fraction")
    if not 0.0 <= collapse <= 1.0:
        raise ExperimentContractError("collapse_fraction must be in [0, 1]")
    first_collapse = episode["first_collapse_step"]
    if episode["survived_full_horizon"]:
        if first_collapse is not None or collapse != 0.0:
            raise ExperimentContractError("survival fields are contradictory")
    elif (
        not isinstance(first_collapse, int)
        or isinstance(first_collapse, bool)
        or not 1 <= first_collapse <= max_episode_steps
        or collapse <= 0.0
    ):
        raise ExperimentContractError("collapsed episode fields are contradictory")
    for field in (*reward_scale_fields, *required_guardrail_fields, "tracking_return"):
        value = _finite_number(episode[field], field=field)
        if field != "tracking_return" and value < 0.0:
            raise ExperimentContractError(f"{field} cannot be negative")


def _episode_passes(
    episode: dict[str, Any],
    *,
    thresholds: tuple[MaximumThreshold, ...],
    maximum_collapse_fraction: float,
) -> tuple[bool, tuple[str, ...]]:
    failures: list[str] = []
    if episode["survived_full_horizon"] is not True:
        failures.append("survived_full_horizon")
    if float(episode["collapse_fraction"]) > maximum_collapse_fraction:
        failures.append("collapse_fraction")
    for threshold in thresholds:
        if threshold.field not in episode:
            raise ExperimentContractError(
                f"preregistered threshold field {threshold.field!r} is absent"
            )
        value = _finite_number(episode[threshold.field], field=threshold.field)
        if value < 0.0:
            raise ExperimentContractError(f"{threshold.field} cannot be negative")
        if value > threshold.maximum:
            failures.append(threshold.field)
    return not failures, tuple(failures)


def _numeric_additional_fields(
    episodes_by_seed: dict[int, tuple[dict[str, Any], ...]],
    additional_fields: tuple[str, ...],
) -> tuple[str, ...]:
    numeric: list[str] = []
    for field in additional_fields:
        values = [episode[field] for episodes in episodes_by_seed.values() for episode in episodes]
        if all(isinstance(value, Real) and not isinstance(value, bool) for value in values):
            for value in values:
                _finite_number(value, field=field)
            numeric.append(field)
    return tuple(numeric)


def _summarize_metric_by_seed(
    episodes_by_seed: dict[int, tuple[dict[str, Any], ...]],
    *,
    field: str,
    train_seed_order: tuple[int, ...],
    confidence_level: float,
) -> dict[str, object]:
    seed_records: list[dict[str, object]] = []
    seed_means: list[float] = []
    for train_seed in train_seed_order:
        values = [
            _finite_number(episode[field], field=field) for episode in episodes_by_seed[train_seed]
        ]
        descriptives = _descriptives(values)
        seed_means.append(descriptives["mean"])
        seed_records.append({"train_seed": train_seed, **descriptives})
    return {
        "episode_reduction_within_seed": "arithmetic_mean",
        "seed_summaries": seed_records,
        "seed_mean_uncertainty": exhaustive_seed_bootstrap(
            seed_means,
            confidence_level=confidence_level,
        ),
    }


def aggregate_static_study(
    *,
    design_path: Path,
    manifest_path: Path,
    criteria_path: Path,
    checkpoint_receipt_paths: Sequence[Path],
    evaluation_receipt_paths: Sequence[Path],
    output_path: Path,
) -> dict[str, object]:
    """Validate exactly five complete receipts and emit one study summary."""

    design_path = design_path.resolve()
    manifest_path = manifest_path.resolve()
    criteria_path = criteria_path.resolve()
    output_path = output_path.resolve()
    design = load_study_design(design_path)
    manifest = load_execution_manifest(manifest_path)
    manifest.validate(design=design, observed=manifest.runtime)
    if manifest.execution_callable_authority != CANONICAL_EXECUTION_CALLABLE_AUTHORITY:
        raise ExperimentContractError(
            "study aggregation requires canonical production execution callables"
        )
    if manifest.runtime.control_period_seconds != _CONTROL_PERIOD_SECONDS:
        raise ExperimentContractError(
            "manifest control period does not match the preregistered 0.015 s rate basis"
        )
    _validate_aggregation_source_hashes(manifest)
    criteria = load_static_study_criteria(criteria_path)
    _validate_design_and_criteria(
        design,
        criteria,
        criteria_path=criteria_path,
    )
    checkpoint_paths = tuple(path.resolve() for path in checkpoint_receipt_paths)
    if len(checkpoint_paths) != criteria.required_independent_train_seeds:
        raise ExperimentContractError(
            "must provide exactly one checkpoint receipt per training seed"
        )
    if len(set(checkpoint_paths)) != len(checkpoint_paths):
        raise ExperimentContractError("checkpoint receipt paths must not contain duplicates")
    training_receipts_by_seed: dict[int, dict[str, Any]] = {}
    checkpoint_receipt_file_hashes: set[str] = set()
    checkpoint_file_hashes: set[str] = set()
    for checkpoint_receipt_path in checkpoint_paths:
        training_receipt, checkpoint_path = _validated_training_receipt(
            receipt_path=checkpoint_receipt_path,
            design_path=design_path,
            design=design,
            manifest_path=manifest_path,
            manifest=manifest,
            runtime=manifest.runtime,
        )
        train_seed = training_receipt["train_seed"]
        if train_seed in training_receipts_by_seed:
            raise ExperimentContractError("checkpoint receipt train seeds are duplicated")
        receipt_file_sha256 = sha256_file(checkpoint_receipt_path)
        checkpoint_file_sha256 = sha256_file(checkpoint_path)
        if receipt_file_sha256 in checkpoint_receipt_file_hashes:
            raise ExperimentContractError("duplicate checkpoint receipt bytes are not allowed")
        if checkpoint_file_sha256 in checkpoint_file_hashes:
            raise ExperimentContractError("duplicate checkpoint bytes are not allowed")
        checkpoint_receipt_file_hashes.add(receipt_file_sha256)
        checkpoint_file_hashes.add(checkpoint_file_sha256)
        training_receipts_by_seed[train_seed] = {
            **training_receipt,
            "_file_sha256": receipt_file_sha256,
            "_checkpoint_path": str(checkpoint_path),
        }
    if set(training_receipts_by_seed) != set(design.train_seeds):
        raise ExperimentContractError(
            "checkpoint receipts do not cover every predetermined training seed"
        )

    paths = tuple(path.resolve() for path in evaluation_receipt_paths)
    if len(paths) != criteria.required_independent_train_seeds:
        raise ExperimentContractError("must provide exactly one receipt per training seed")
    if len(set(paths)) != len(paths):
        raise ExperimentContractError("evaluation receipt paths must not contain duplicates")

    design_file_sha256 = sha256_file(design_path)
    manifest_file_sha256 = sha256_file(manifest_path)
    reward_scale_fields = tuple(
        threshold.field for threshold in criteria.reward_scale_occupancy_bounds
    )
    guardrail_threshold_fields = tuple(
        threshold.field for threshold in criteria.guardrail_thresholds
    )
    required_descriptive_fields = tuple(
        metric.field for metric in criteria.required_descriptive_metrics
    )
    station_keeping_fields = tuple(
        metric.field for metric in criteria.unthresholded_station_keeping_metrics
    )
    required_guardrail_fields = (
        *guardrail_threshold_fields,
        *required_descriptive_fields,
        *station_keeping_fields,
    )
    receipts_by_seed: dict[int, dict[str, Any]] = {}
    receipt_hashes: set[str] = set()
    episodes_by_seed: dict[int, tuple[dict[str, Any], ...]] = {}
    common_episode_fields: set[str] | None = None
    common_extra_root_fields: set[str] | None = None

    for path in paths:
        receipt = _read_json_object(
            path,
            maximum_bytes=MAX_EVALUATION_RECEIPT_BYTES,
            artifact="evaluation receipt",
        )
        receipt_sha256 = sha256_file(path)
        if receipt_sha256 in receipt_hashes:
            raise ExperimentContractError("duplicate evaluation receipt bytes are not allowed")
        receipt_hashes.add(receipt_sha256)
        _validate_receipt_root(
            receipt,
            design=design,
            design_file_sha256=design_file_sha256,
            manifest=manifest,
            manifest_file_sha256=manifest_file_sha256,
        )
        train_seed = receipt.get("train_seed")
        if (
            not isinstance(train_seed, int)
            or isinstance(train_seed, bool)
            or train_seed not in design.train_seeds
            or train_seed in receipts_by_seed
        ):
            raise ExperimentContractError(
                "train seeds are missing, duplicated, or not predetermined"
            )
        checkpoint_hash = receipt["checkpoint_sha256"]
        checkpoint_receipt_hash = receipt["checkpoint_receipt_file_sha256"]
        training_receipt = training_receipts_by_seed[train_seed]
        if checkpoint_hash != training_receipt["checkpoint_sha256"]:
            raise ExperimentContractError(
                "evaluation receipt checkpoint SHA-256 differs from its revalidated checkpoint"
            )
        if checkpoint_receipt_hash != training_receipt["_file_sha256"]:
            raise ExperimentContractError(
                "evaluation receipt does not bind its exact revalidated checkpoint receipt"
            )

        raw_episodes = receipt["episodes"]
        if not isinstance(raw_episodes, list) or len(raw_episodes) != len(design.evaluation_seeds):
            raise ExperimentContractError("evaluation receipt does not contain every episode")
        episodes: list[dict[str, Any]] = []
        for evaluation_seed, episode in zip(
            design.evaluation_seeds,
            raw_episodes,
            strict=True,
        ):
            if not isinstance(episode, dict):
                raise ExperimentContractError("each evaluation episode must be an object")
            _validate_episode(
                episode,
                evaluation_seed=evaluation_seed,
                max_episode_steps=design.max_episode_steps,
                control_period_seconds=manifest.runtime.control_period_seconds,
                reward_scale_fields=reward_scale_fields,
                required_guardrail_fields=required_guardrail_fields,
            )
            fields = set(episode)
            if common_episode_fields is None:
                common_episode_fields = fields
            elif fields != common_episode_fields:
                raise ExperimentContractError(
                    "episode metric schema differs across evaluation receipts"
                )
            episodes.append(episode)
        extra_root_fields = set(receipt) - _REQUIRED_RECEIPT_FIELDS
        if common_extra_root_fields is None:
            common_extra_root_fields = extra_root_fields
        elif extra_root_fields != common_extra_root_fields:
            raise ExperimentContractError("evaluation receipt extension schema differs by seed")
        receipt["_file_sha256"] = receipt_sha256
        receipts_by_seed[train_seed] = receipt
        episodes_by_seed[train_seed] = tuple(episodes)

    if set(receipts_by_seed) != set(design.train_seeds):
        raise ExperimentContractError("evaluation receipts do not cover every predetermined seed")
    if common_episode_fields is None:
        raise ExperimentContractError("evaluation receipts contain no episodes")
    additional_fields = tuple(
        sorted(common_episode_fields - _STRUCTURAL_EPISODE_FIELDS - set(reward_scale_fields))
    )
    numeric_additional_fields = _numeric_additional_fields(
        episodes_by_seed,
        additional_fields,
    )
    train_seed_order = tuple(design.train_seeds)
    checkpoint_summaries: list[dict[str, object]] = []
    reward_scale_checkpoint_indicators: list[float] = []
    overall_checkpoint_indicators: list[float] = []
    reward_scale_occupancy_rates: list[float] = []
    overall_conformance_rates: list[float] = []

    for train_seed in train_seed_order:
        episodes = episodes_by_seed[train_seed]
        reward_scale_results = [
            _episode_passes(
                episode,
                thresholds=criteria.reward_scale_occupancy_bounds,
                maximum_collapse_fraction=criteria.maximum_collapse_fraction,
            )
            for episode in episodes
        ]
        episode_rule_reward_scale_occupancy_count = sum(
            passed for passed, _ in reward_scale_results
        )
        episode_rule_reward_scale_occupancy_rate = episode_rule_reward_scale_occupancy_count / len(
            episodes
        )
        episode_rule_reward_scale_checkpoint_conformance = (
            episode_rule_reward_scale_occupancy_count >= criteria.minimum_conforming_episodes
        )
        failure_counts: dict[str, int] = {}
        for _, failures in reward_scale_results:
            for field in failures:
                failure_counts[field] = failure_counts.get(field, 0) + 1

        conformance_results = [
            _episode_passes(
                episode,
                thresholds=criteria.guardrail_thresholds,
                maximum_collapse_fraction=criteria.maximum_collapse_fraction,
            )
            for episode in episodes
        ]
        conformance_success_count = sum(passed for passed, _ in conformance_results)
        combined = [
            reward_scale[0] and conformance[0]
            for reward_scale, conformance in zip(
                reward_scale_results,
                conformance_results,
                strict=True,
            )
        ]
        combined_conformance_count = sum(combined)
        combined_conformance_rate = combined_conformance_count / len(episodes)
        combined_checkpoint_conformance = (
            combined_conformance_count >= criteria.minimum_conforming_episodes
        )
        overall_checkpoint_indicators.append(float(combined_checkpoint_conformance))
        overall_conformance_rates.append(combined_conformance_rate)

        reward_scale_checkpoint_indicators.append(
            float(episode_rule_reward_scale_checkpoint_conformance)
        )
        reward_scale_occupancy_rates.append(episode_rule_reward_scale_occupancy_rate)
        receipt = receipts_by_seed[train_seed]
        checkpoint_summaries.append(
            {
                "train_seed": train_seed,
                "evaluation_receipt_sha256": receipt["_file_sha256"],
                "checkpoint_receipt_file_sha256": training_receipts_by_seed[train_seed][
                    "_file_sha256"
                ],
                "checkpoint_sha256": receipt["checkpoint_sha256"],
                "evaluation_episode_count": len(episodes),
                "episode_rule_and_reward_scale_occupancy_episode_count": (
                    episode_rule_reward_scale_occupancy_count
                ),
                "episode_rule_and_reward_scale_occupancy_fraction": (
                    episode_rule_reward_scale_occupancy_rate
                ),
                "episode_rule_and_reward_scale_occupancy_checkpoint_conformance": (
                    episode_rule_reward_scale_checkpoint_conformance
                ),
                "episode_rule_and_reward_scale_deviation_counts_by_field": dict(
                    sorted(failure_counts.items())
                ),
                "guardrail_threshold_status": criteria.guardrail_status,
                "episode_rule_bounded_control_and_prohibited_floor_contact_conformance_"
                "episode_count": conformance_success_count,
                "episode_rule_reward_scale_occupancy_and_bounded_control_prohibited_floor_"
                "contact_conformance_episode_count": (combined_conformance_count),
                "episode_rule_reward_scale_occupancy_and_bounded_control_prohibited_floor_"
                "contact_conformance_fraction": combined_conformance_rate,
                "episode_rule_reward_scale_occupancy_and_bounded_control_prohibited_floor_"
                "contact_checkpoint_conformance": (combined_checkpoint_conformance),
                "reward_scale_metric_descriptives": {
                    field: _descriptives(
                        [_finite_number(episode[field], field=field) for episode in episodes]
                    )
                    for field in reward_scale_fields
                },
                "additional_numeric_metric_descriptives": {
                    field: _descriptives(
                        [_finite_number(episode[field], field=field) for episode in episodes]
                    )
                    for field in numeric_additional_fields
                },
            }
        )

    reward_scale_conforming_checkpoints = int(sum(reward_scale_checkpoint_indicators))
    reward_scale_study_conformance = (
        reward_scale_conforming_checkpoints >= criteria.minimum_conforming_checkpoints
    )
    conforming_checkpoints = int(sum(overall_checkpoint_indicators))
    coarse_conformance = conforming_checkpoints >= criteria.minimum_conforming_checkpoints
    decision_status = (
        "blocked_missing_independently_justified_station_keeping_and_naturalness_thresholds"
    )

    reward_scale_descriptives = {
        "checkpoint_conformance_fraction": exhaustive_seed_bootstrap(
            reward_scale_checkpoint_indicators,
            confidence_level=criteria.confidence_level,
        ),
        "episode_conformance_fraction": exhaustive_seed_bootstrap(
            reward_scale_occupancy_rates,
            confidence_level=criteria.confidence_level,
        ),
        "metrics": {
            field: _summarize_metric_by_seed(
                episodes_by_seed,
                field=field,
                train_seed_order=train_seed_order,
                confidence_level=criteria.confidence_level,
            )
            for field in reward_scale_fields
        },
    }
    overall_uncertainty: dict[str, object] = {
        "checkpoint_conformance_fraction": exhaustive_seed_bootstrap(
            overall_checkpoint_indicators,
            confidence_level=criteria.confidence_level,
        ),
        "episode_conformance_fraction": exhaustive_seed_bootstrap(
            overall_conformance_rates,
            confidence_level=criteria.confidence_level,
        ),
    }

    payload = {
        "schema_version": 1,
        "artifact_kind": "static_positive_control_study_summary",
        "completion_status": "complete_all_predetermined_seeds",
        "evidence_purpose": "behavioral_evaluation",
        "claim_ceiling": criteria.claim_ceiling,
        "design_sha256": design.sha256,
        "design_file_sha256": design_file_sha256,
        "execution_manifest_sha256": manifest.sha256,
        "execution_manifest_file_sha256": manifest_file_sha256,
        "runtime_sha256": manifest.runtime.sha256,
        "protected_evaluator_sha256": manifest.runtime.evaluator_source_sha256,
        "evaluation_reference_sha256": manifest.runtime.reference_content_sha256,
        "evaluation_reference_schema_sha256": manifest.runtime.reference_schema_sha256,
        "criteria_id": criteria.criteria_id,
        "criteria_semantic_sha256": criteria.sha256,
        "criteria_file_sha256": sha256_file(criteria_path),
        "study_summary_source_sha256": manifest.runtime.study_summary_source_sha256,
        "independent_unit": "train_seed",
        "independent_n": len(train_seed_order),
        "repeated_measure": "paired_evaluation_seed_within_checkpoint",
        "evaluation_seeds_per_checkpoint": len(design.evaluation_seeds),
        "train_seeds": list(train_seed_order),
        "evaluation_seeds": list(design.evaluation_seeds),
        "evaluation_receipts": [
            {
                "train_seed": summary["train_seed"],
                "evaluation_receipt_sha256": summary["evaluation_receipt_sha256"],
                "checkpoint_receipt_file_sha256": summary["checkpoint_receipt_file_sha256"],
                "checkpoint_sha256": summary["checkpoint_sha256"],
            }
            for summary in checkpoint_summaries
        ],
        "episode_metric_schema": sorted(common_episode_fields),
        "additional_receipt_fields": sorted(common_extra_root_fields or set()),
        "additional_episode_fields": list(additional_fields),
        "additional_numeric_metric_fields": list(numeric_additional_fields),
        "checkpoint_summaries": checkpoint_summaries,
        "coarse_episode_rule_reward_scale_occupancy_bounded_control_and_prohibited_floor_"
        "contact_conformance": {
            "conforming_checkpoints": conforming_checkpoints,
            "required_conforming_checkpoints": criteria.minimum_conforming_checkpoints,
            "conforms": coarse_conformance,
            "episode_rule_reward_scale_only_conforming_checkpoints": (
                reward_scale_conforming_checkpoints
            ),
            "episode_rule_reward_scale_only_conforms": reward_scale_study_conformance,
        },
        "study_decision": {
            "decision_status": decision_status,
            "decision_name": "static_stand_feasibility_pass",
            "guardrail_threshold_status": criteria.guardrail_status,
            "passing_checkpoints": None,
            "required_passing_checkpoints": None,
            "static_stand_feasibility_pass": None,
            "coarse_episode_rule_reward_scale_occupancy_bounded_control_and_prohibited_floor_"
            "contact_conformance": coarse_conformance,
            "study_pass": None,
            "station_keeping_assessed": False,
            "station_keeping_measurements_complete": True,
            "station_keeping_thresholds_defined": False,
            "station_keeping_requirement_status": (criteria.station_keeping_requirement_status),
            "unthresholded_station_keeping_metrics": [
                metric.field for metric in criteria.unthresholded_station_keeping_metrics
            ],
            "naturalness_assessed": False,
            "naturalness_calibration_status": (criteria.naturalness_calibration_status),
            "interpretation": (
                "episode_rule_reward_scale_occupancy_bounded_control_and_prohibited_floor_"
                "contact_conformance_only_not_objective_tracking_or_"
                "thresholded_station_keeping_naturalness_causal_reference_use_or_"
                "oracle_superiority"
            ),
        },
        "episode_rule_reward_scale_occupancy_seed_level_descriptives": (reward_scale_descriptives),
        "coarse_episode_rule_reward_scale_occupancy_bounded_control_and_prohibited_"
        "floor_contact_conformance_"
        "seed_level_uncertainty": overall_uncertainty,
        "station_keeping_descriptive_metrics": {
            field: _summarize_metric_by_seed(
                episodes_by_seed,
                field=field,
                train_seed_order=train_seed_order,
                confidence_level=criteria.confidence_level,
            )
            for field in station_keeping_fields
        },
        "additional_numeric_metrics": {
            field: _summarize_metric_by_seed(
                episodes_by_seed,
                field=field,
                train_seed_order=train_seed_order,
                confidence_level=criteria.confidence_level,
            )
            for field in numeric_additional_fields
        },
        "uncertainty_caveat": (
            "descriptive_seed_sensitivity_only; evaluation episodes are repeated measures; "
            "independent n remains 5"
        ),
        "automatic_promotion": False,
        "behavioral_claim": None,
    }
    emit_json_without_overwrite(output_path, payload)
    return {
        "output_path": str(output_path),
        "output_file_sha256": sha256_file(output_path),
        **payload,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="humanoid-static-study-summary",
        description="Validate and aggregate all five fixed-reference evaluation receipts.",
    )
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--criteria", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-receipt",
        type=Path,
        action="append",
        required=True,
        dest="checkpoint_receipts",
    )
    parser.add_argument(
        "--evaluation-receipt",
        type=Path,
        action="append",
        required=True,
        dest="evaluation_receipts",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = aggregate_static_study(
            design_path=args.design,
            manifest_path=args.manifest,
            criteria_path=args.criteria,
            checkpoint_receipt_paths=args.checkpoint_receipts,
            evaluation_receipt_paths=args.evaluation_receipts,
            output_path=args.output,
        )
    except (ExperimentContractError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BlockedMetricRequirement",
    "MaximumThreshold",
    "RequiredDescriptiveMetric",
    "StaticStudyCriteria",
    "aggregate_static_study",
    "exhaustive_seed_bootstrap",
    "load_static_study_criteria",
    "main",
]
