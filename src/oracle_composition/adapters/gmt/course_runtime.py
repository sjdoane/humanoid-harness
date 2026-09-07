"""Versioned runtime profiles for the bounded G1 course family."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from oracle_composition.harness.contract import OracleProgram

from .contracts import CONTROL_DT_SECONDS, OBSERVATION_DIM
from .course_task import TASK_FEATURE_NAMES

LEGACY_CONFIG_SCHEMA_VERSION = 1
LOOP_CONFIG_SCHEMA_VERSION = 2
FINITE_HORIZON_CONFIG_SCHEMA_VERSION = 3
AFTER_HEADING_FEEDBACK_CONFIG_SCHEMA_VERSION = 4
LOOP_RUNTIME_PROFILE_ID = "gmt_g1_four_state_loop_course/v1"
FINITE_HORIZON_RUNTIME_PROFILE_ID = "gmt_g1_three_state_finite_horizon_course/v1"
AFTER_HEADING_FEEDBACK_RUNTIME_PROFILE_ID = (
    "gmt_g1_four_state_loop_after_heading_feedback_course/v1"
)
LEGACY_COMPOSITION_RUNTIME_ID = "gmt_state_triggered_segment_entry_and_boundary/v2"
LOOP_COMPOSITION_RUNTIME_ID = "gmt_state_triggered_segment_entry_loop_boundary/v3"
LEGACY_GYM_RUNTIME_ID = "gmt_g1_residual_course_50hz/v1"
LOOP_GYM_RUNTIME_ID = "gmt_g1_residual_course_four_state_50hz/v2"
FINITE_HORIZON_GYM_RUNTIME_ID = "gmt_g1_residual_course_three_state_finite_horizon_50hz/v1"
AFTER_HEADING_FEEDBACK_GYM_RUNTIME_ID = (
    "gmt_g1_residual_course_four_state_after_heading_feedback_50hz/v1"
)
COURSE_RESIDUAL_RAW_SCALE = 0.25
LEGACY_STATE_SLOTS = ("before", "inside", "after")
LOOP_STATE_SLOTS = ("before", "inside", "rise", "after")
_REQUIRED_STATES = frozenset(LEGACY_STATE_SLOTS)
_LOOP_CONFIG_VALUE = {
    "schema_version": 1,
    "profile_id": LOOP_RUNTIME_PROFILE_ID,
}
_FINITE_HORIZON_CONFIG_VALUE = {
    "schema_version": 1,
    "profile_id": FINITE_HORIZON_RUNTIME_PROFILE_ID,
}
_AFTER_HEADING_FEEDBACK_CONFIG_VALUE = {
    "schema_version": 1,
    "profile_id": AFTER_HEADING_FEEDBACK_RUNTIME_PROFILE_ID,
}


@dataclass(frozen=True, slots=True)
class CourseRuntimeProfile:
    """One closed runtime vocabulary; not an authorable hyperparameter bundle."""

    config_schema_version: int
    profile_id: str | None
    composition_runtime_id: str
    gym_runtime_id: str
    state_slots: tuple[str, ...]
    permits_entry_loop: bool
    training_admitted: bool
    intrinsic_horizon_termination: bool
    after_heading_reference_feedback: bool

    @property
    def observation_dim(self) -> int:
        return OBSERVATION_DIM + len(TASK_FEATURE_NAMES) + 3 + len(self.state_slots)

    @property
    def config_value(self) -> dict[str, object] | None:
        if self.profile_id is None:
            return None
        if self.profile_id == LOOP_RUNTIME_PROFILE_ID:
            return dict(_LOOP_CONFIG_VALUE)
        if self.profile_id == FINITE_HORIZON_RUNTIME_PROFILE_ID:
            return dict(_FINITE_HORIZON_CONFIG_VALUE)
        if self.profile_id == AFTER_HEADING_FEEDBACK_RUNTIME_PROFILE_ID:
            return dict(_AFTER_HEADING_FEEDBACK_CONFIG_VALUE)
        raise ValueError("course runtime profile is not admitted")

    def validate_states(self, states: Mapping[str, object]) -> None:
        observed = set(states)
        if self.after_heading_reference_feedback:
            if observed != frozenset(LOOP_STATE_SLOTS):
                raise ValueError(
                    "heading-feedback runtime requires the exact four-state course"
                )
            return
        valid = observed == _REQUIRED_STATES
        if self.permits_entry_loop:
            valid = valid or observed == frozenset(LOOP_STATE_SLOTS)
        if not valid:
            family = (
                "three-state or four-state loop" if self.permits_entry_loop else "three-state"
            )
            raise ValueError(f"oracle states differ from the {family} runtime profile")

    def validate_program(self, program: OracleProgram) -> None:
        self.validate_states(program.states)
        if not self.permits_entry_loop:
            return
        if program.recovery is not None:
            raise ValueError("loop runtime does not admit recovery semantics")
        if any(
            program.states[transition.source].behavior == program.states[transition.target].behavior
            for transition in program.transitions
        ):
            raise ValueError("loop runtime requires behavior-changing state transitions")

    def manifest_contract(self) -> dict[str, object] | None:
        if self.profile_id is None:
            return None
        result: dict[str, object] = {
            "schema_version": 1,
            "profile_id": self.profile_id,
            "state_observation_slots": list(self.state_slots),
            "training_admitted": self.training_admitted,
        }
        if self.permits_entry_loop:
            result["loop_exit_gate"] = {
                "guard_sampling": "fresh_signals_only_at_first_50hz_boundary_crossing_loop_end",
                "float32_boundary_tolerance": "four_eps_times_max_source_end_or_one",
                "maximum_deferral": "one_loop_period_plus_one_control_interval",
                "control_interval_seconds": CONTROL_DT_SECONDS,
            }
        if self.intrinsic_horizon_termination:
            result["termination"] = {
                "fall": "terminated",
                "intrinsic_horizon": "terminated",
                "truncated": False,
                "remaining_time_observation": "task_features.remaining_horizon_fraction",
            }
        if self.after_heading_reference_feedback:
            from .heading_feedback import after_heading_feedback_contract

            result["after_heading_reference_feedback"] = after_heading_feedback_contract()
        return result


LEGACY_RUNTIME = CourseRuntimeProfile(
    config_schema_version=LEGACY_CONFIG_SCHEMA_VERSION,
    profile_id=None,
    composition_runtime_id=LEGACY_COMPOSITION_RUNTIME_ID,
    gym_runtime_id=LEGACY_GYM_RUNTIME_ID,
    state_slots=LEGACY_STATE_SLOTS,
    permits_entry_loop=False,
    training_admitted=True,
    intrinsic_horizon_termination=False,
    after_heading_reference_feedback=False,
)
LOOP_RUNTIME = CourseRuntimeProfile(
    config_schema_version=LOOP_CONFIG_SCHEMA_VERSION,
    profile_id=LOOP_RUNTIME_PROFILE_ID,
    composition_runtime_id=LOOP_COMPOSITION_RUNTIME_ID,
    gym_runtime_id=LOOP_GYM_RUNTIME_ID,
    state_slots=LOOP_STATE_SLOTS,
    permits_entry_loop=True,
    training_admitted=False,
    intrinsic_horizon_termination=False,
    after_heading_reference_feedback=False,
)
FINITE_HORIZON_RUNTIME = CourseRuntimeProfile(
    config_schema_version=FINITE_HORIZON_CONFIG_SCHEMA_VERSION,
    profile_id=FINITE_HORIZON_RUNTIME_PROFILE_ID,
    composition_runtime_id=LEGACY_COMPOSITION_RUNTIME_ID,
    gym_runtime_id=FINITE_HORIZON_GYM_RUNTIME_ID,
    state_slots=LEGACY_STATE_SLOTS,
    permits_entry_loop=False,
    training_admitted=True,
    intrinsic_horizon_termination=True,
    after_heading_reference_feedback=False,
)
AFTER_HEADING_FEEDBACK_RUNTIME = CourseRuntimeProfile(
    config_schema_version=AFTER_HEADING_FEEDBACK_CONFIG_SCHEMA_VERSION,
    profile_id=AFTER_HEADING_FEEDBACK_RUNTIME_PROFILE_ID,
    composition_runtime_id=LOOP_COMPOSITION_RUNTIME_ID,
    gym_runtime_id=AFTER_HEADING_FEEDBACK_GYM_RUNTIME_ID,
    state_slots=LOOP_STATE_SLOTS,
    permits_entry_loop=True,
    training_admitted=False,
    intrinsic_horizon_termination=False,
    after_heading_reference_feedback=True,
)


def runtime_profile_from_config(value: Mapping[str, object]) -> CourseRuntimeProfile:
    """Resolve only the implicit legacy profile or an exact opt-in profile."""

    schema_version = value.get("schema_version")
    if type(schema_version) is not int:
        raise ValueError("course run schema version differs")
    if schema_version == LEGACY_CONFIG_SCHEMA_VERSION:
        if "runtime" in value:
            raise ValueError("legacy course config cannot declare a runtime profile")
        return LEGACY_RUNTIME
    if schema_version == LOOP_CONFIG_SCHEMA_VERSION:
        runtime = value.get("runtime")
        if (
            type(runtime) is not dict
            or set(runtime) != set(_LOOP_CONFIG_VALUE)
            or type(runtime.get("schema_version")) is not int
            or runtime != _LOOP_CONFIG_VALUE
        ):
            raise ValueError("course loop runtime profile differs")
        return LOOP_RUNTIME
    if schema_version == FINITE_HORIZON_CONFIG_SCHEMA_VERSION:
        runtime = value.get("runtime")
        if (
            type(runtime) is not dict
            or set(runtime) != set(_FINITE_HORIZON_CONFIG_VALUE)
            or type(runtime.get("schema_version")) is not int
            or runtime != _FINITE_HORIZON_CONFIG_VALUE
        ):
            raise ValueError("course finite-horizon runtime profile differs")
        return FINITE_HORIZON_RUNTIME
    if schema_version == AFTER_HEADING_FEEDBACK_CONFIG_SCHEMA_VERSION:
        runtime = value.get("runtime")
        if (
            type(runtime) is not dict
            or set(runtime) != set(_AFTER_HEADING_FEEDBACK_CONFIG_VALUE)
            or type(runtime.get("schema_version")) is not int
            or runtime != _AFTER_HEADING_FEEDBACK_CONFIG_VALUE
        ):
            raise ValueError("course after-heading-feedback runtime profile differs")
        return AFTER_HEADING_FEEDBACK_RUNTIME
    raise ValueError("course run schema version differs")


def frozen_runtime_contract(
    profile: CourseRuntimeProfile,
    *,
    trainer: Mapping[str, object],
    residual_raw_scale: float,
) -> dict[str, object]:
    """Return the exact manifest payload while preserving legacy shape."""

    result: dict[str, object] = {
        "gym": profile.gym_runtime_id,
        "composition": profile.composition_runtime_id,
        "observation_dim": profile.observation_dim,
        "residual_raw_scale": residual_raw_scale,
        "trainer": dict(trainer),
    }
    runtime = profile.manifest_contract()
    if runtime is not None:
        result["course_runtime"] = runtime
    return result


__all__ = [
    "AFTER_HEADING_FEEDBACK_CONFIG_SCHEMA_VERSION",
    "AFTER_HEADING_FEEDBACK_GYM_RUNTIME_ID",
    "AFTER_HEADING_FEEDBACK_RUNTIME",
    "AFTER_HEADING_FEEDBACK_RUNTIME_PROFILE_ID",
    "COURSE_RESIDUAL_RAW_SCALE",
    "FINITE_HORIZON_CONFIG_SCHEMA_VERSION",
    "FINITE_HORIZON_GYM_RUNTIME_ID",
    "FINITE_HORIZON_RUNTIME",
    "FINITE_HORIZON_RUNTIME_PROFILE_ID",
    "LEGACY_COMPOSITION_RUNTIME_ID",
    "LEGACY_CONFIG_SCHEMA_VERSION",
    "LEGACY_GYM_RUNTIME_ID",
    "LEGACY_RUNTIME",
    "LEGACY_STATE_SLOTS",
    "LOOP_COMPOSITION_RUNTIME_ID",
    "LOOP_CONFIG_SCHEMA_VERSION",
    "LOOP_GYM_RUNTIME_ID",
    "LOOP_RUNTIME",
    "LOOP_RUNTIME_PROFILE_ID",
    "LOOP_STATE_SLOTS",
    "CourseRuntimeProfile",
    "frozen_runtime_contract",
    "runtime_profile_from_config",
]
