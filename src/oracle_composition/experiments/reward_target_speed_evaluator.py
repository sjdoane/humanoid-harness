"""Independent target-speed endpoint and physical guardrails."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

FAMILY_ID = "family-b-target-speed-v1"
HORIZON_STEPS = 1000
BURN_IN_STEPS = 200
SCORED_STEPS = 800
CONTROL_PERIOD_SECONDS = 0.015
PHYSICS_TIMESTEP_SECONDS = 0.003
PHYSICS_SUBSTEPS = 5
ENDPOINT_SIGMA_M_S = 0.25
TARGET_SPEEDS_M_S = (0.5, 1.0, 1.5)
TRAINING_SEEDS = (101, 202, 303, 404, 505)
EVALUATION_SEEDS = tuple(range(11001, 11021))
FLOOR_GEOM_NAME = "floor"
ALLOWED_FOOT_GEOM_NAMES = frozenset({"left_foot", "right_foot"})
ACTUATOR_QVEL_INDICES_BY_ACTION = (
    7,
    6,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
)
TORQUE_CAPACITY_N_M = (
    40.0,
    40.0,
    40.0,
    40.0,
    40.0,
    120.0,
    80.0,
    40.0,
    40.0,
    120.0,
    80.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
    10.0,
)
TORQUE_ACTION_ABS_TOLERANCE = 1e-7
TORQUE_CAPACITY_REL_TOLERANCE = 1e-9
EXPLORATORY_RELATIVE_MARGIN = 0.10


class TargetSpeedEvaluationError(ValueError):
    """Raised when protected inputs or complete-result lineage fail closed."""


def target_speed_evaluator_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TargetSpeedEvaluationError(f"{field} must be one finite scalar")
    result = float(value)
    if not math.isfinite(result):
        raise TargetSpeedEvaluationError(f"{field} must be finite")
    return result


def _exact_f64_array(value: object, *, shape: tuple[int, ...], field: str) -> np.ndarray:
    if (
        not isinstance(value, np.ndarray)
        or value.dtype != np.dtype(np.float64)
        or value.shape != shape
        or not np.isfinite(value).all()
    ):
        raise TargetSpeedEvaluationError(f"{field} must be a finite float64 array {shape}")
    result = np.array(value, dtype=np.float64, order="C", copy=True)
    result.flags.writeable = False
    return result


def _target(value: object) -> float:
    result = _finite_float(value, field="target_speed_m_s")
    if result not in TARGET_SPEEDS_M_S:
        raise TargetSpeedEvaluationError("target_speed_m_s is outside the frozen target set")
    return result


def _rms(values: np.ndarray) -> float:
    if values.size < 1 or not np.isfinite(values).all():
        raise TargetSpeedEvaluationError("RMS input must be finite and nonempty")
    maximum = float(np.max(np.abs(values)))
    if maximum == 0.0:
        return 0.0
    return maximum * math.sqrt(float(np.mean(np.square(values / maximum))))


def _com_xy(body_mass_f64: np.ndarray, body_xipos_f64: np.ndarray) -> np.ndarray:
    numerator = np.einsum("b,bj->j", body_mass_f64, body_xipos_f64)
    denominator = body_mass_f64.sum()
    value = (numerator / denominator)[0:2].copy()
    if not np.isfinite(value).all():
        raise TargetSpeedEvaluationError("derived center of mass is non-finite")
    return value


def recompute_com_x_velocity_m_s(
    *,
    body_mass_f64: np.ndarray,
    body_xipos_before_f64: np.ndarray,
    body_xipos_after_f64: np.ndarray,
) -> float:
    masses = _exact_f64_array(body_mass_f64, shape=(14,), field="body_mass_f64")
    before = _exact_f64_array(body_xipos_before_f64, shape=(14, 3), field="body_xipos_before_f64")
    after = _exact_f64_array(body_xipos_after_f64, shape=(14, 3), field="body_xipos_after_f64")
    if np.any(masses < 0.0) or float(masses.sum()) <= 0.0:
        raise TargetSpeedEvaluationError("body masses must be non-negative with positive sum")
    velocity = (_com_xy(masses, after) - _com_xy(masses, before)) / CONTROL_PERIOD_SECONDS
    result = float(velocity[0])
    if not math.isfinite(result):
        raise TargetSpeedEvaluationError("derived COM velocity is non-finite")
    return result


def target_speed_progress(value_m_s: object, target_speed_m_s: object) -> float:
    velocity = _finite_float(value_m_s, field="com_x_velocity_m_s")
    target = _target(target_speed_m_s)
    standardized_error = (velocity - target) / ENDPOINT_SIGMA_M_S
    result = math.exp(-0.5 * standardized_error * standardized_error)
    if not 0.0 <= result <= 1.0 or not math.isfinite(result):
        raise TargetSpeedEvaluationError("derived target-speed progress is invalid")
    return result


@dataclass(frozen=True, slots=True)
class TargetSpeedSubstepContactV1:
    physics_substep_index: int
    contact_index_within_substep: int
    geom1_name: str
    geom2_name: str
    normal_force_n: float

    def __post_init__(self) -> None:
        for field in ("physics_substep_index", "contact_index_within_substep"):
            value = getattr(self, field)
            if type(value) is not int or value < 0:
                raise TargetSpeedEvaluationError(f"{field} must be a non-negative integer")
        if self.physics_substep_index >= PHYSICS_SUBSTEPS:
            raise TargetSpeedEvaluationError("physics_substep_index exceeds the five substeps")
        for field in ("geom1_name", "geom2_name"):
            value = getattr(self, field)
            if type(value) is not str or not value or len(value) > 255:
                raise TargetSpeedEvaluationError(f"{field} must be a bounded nonempty string")
        force = _finite_float(self.normal_force_n, field="normal_force_n")
        if force < 0.0:
            raise TargetSpeedEvaluationError("normal_force_n cannot be negative")
        object.__setattr__(self, "normal_force_n", force)

    @property
    def floor_other_geom(self) -> str | None:
        occurrences = (self.geom1_name, self.geom2_name).count(FLOOR_GEOM_NAME)
        if occurrences > 1:
            raise TargetSpeedEvaluationError("one contact cannot contain floor twice")
        if occurrences == 0:
            return None
        return self.geom2_name if self.geom1_name == FLOOR_GEOM_NAME else self.geom1_name


@dataclass(frozen=True, slots=True)
class ProtectedTargetSpeedStepV1:
    transition_index: int
    body_mass_f64: np.ndarray
    body_xipos_before_f64: np.ndarray
    body_xipos_after_f64: np.ndarray
    qpos_after_f64: np.ndarray
    qvel_after_f64: np.ndarray
    normalized_action_f64: np.ndarray
    generalized_actuator_torque_n_m_f64: np.ndarray
    contacts: tuple[TargetSpeedSubstepContactV1, ...]
    captured_physics_substeps: int
    control_period_s: float
    physics_timestep_s: float
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        if (
            type(self.transition_index) is not int
            or not 1 <= self.transition_index <= HORIZON_STEPS
        ):
            raise TargetSpeedEvaluationError("transition_index is outside 1..1000")
        arrays = (
            ("body_mass_f64", (14,)),
            ("body_xipos_before_f64", (14, 3)),
            ("body_xipos_after_f64", (14, 3)),
            ("qpos_after_f64", (24,)),
            ("qvel_after_f64", (23,)),
            ("normalized_action_f64", (17,)),
            ("generalized_actuator_torque_n_m_f64", (17,)),
        )
        for field, shape in arrays:
            object.__setattr__(
                self,
                field,
                _exact_f64_array(getattr(self, field), shape=shape, field=field),
            )
        if np.any(self.body_mass_f64 < 0.0) or float(self.body_mass_f64.sum()) <= 0.0:
            raise TargetSpeedEvaluationError("body masses must be non-negative with positive sum")
        if np.any(np.abs(self.normalized_action_f64) > 1.0):
            raise TargetSpeedEvaluationError("normalized action exceeds structural [-1, 1]")
        if type(self.contacts) is not tuple or any(
            not isinstance(item, TargetSpeedSubstepContactV1) for item in self.contacts
        ):
            raise TargetSpeedEvaluationError("contacts must be an immutable contact tuple")
        for substep in range(PHYSICS_SUBSTEPS):
            indices = [
                item.contact_index_within_substep
                for item in self.contacts
                if item.physics_substep_index == substep
            ]
            if indices != list(range(len(indices))):
                raise TargetSpeedEvaluationError(
                    f"contact indices are incomplete at physics substep {substep}"
                )
        if self.captured_physics_substeps != PHYSICS_SUBSTEPS:
            raise TargetSpeedEvaluationError("contact capture must cover all five physics substeps")
        if self.control_period_s != CONTROL_PERIOD_SECONDS:
            raise TargetSpeedEvaluationError("control cadence differs from 0.015 s")
        if self.physics_timestep_s != PHYSICS_TIMESTEP_SECONDS:
            raise TargetSpeedEvaluationError("physics cadence differs from 0.003 s")
        if type(self.terminated) is not bool or type(self.truncated) is not bool:
            raise TargetSpeedEvaluationError("termination flags must be built-in booleans")
        capacities = np.asarray(TORQUE_CAPACITY_N_M, dtype=np.float64)
        normalized_torque = self.generalized_actuator_torque_n_m_f64 / capacities
        if np.any(np.abs(normalized_torque) > 1.0 + TORQUE_CAPACITY_REL_TOLERANCE):
            raise TargetSpeedEvaluationError("generalized torque exceeds a structural capacity")
        if not np.allclose(
            normalized_torque,
            self.normalized_action_f64,
            rtol=0.0,
            atol=TORQUE_ACTION_ABS_TOLERANCE,
        ):
            raise TargetSpeedEvaluationError(
                "capacity-normalized torque differs from the executed normalized action"
            )
        quaternion = self.qpos_after_f64[3:7]
        if not math.isclose(float(np.linalg.norm(quaternion)), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise TargetSpeedEvaluationError("root orientation must be a unit quaternion")
        for item in self.contacts:
            _ = item.floor_other_geom

    @property
    def com_x_velocity_m_s(self) -> float:
        return recompute_com_x_velocity_m_s(
            body_mass_f64=self.body_mass_f64,
            body_xipos_before_f64=self.body_xipos_before_f64,
            body_xipos_after_f64=self.body_xipos_after_f64,
        )

    @property
    def torso_up_z(self) -> float:
        _, x, y, _ = self.qpos_after_f64[3:7]
        return float(1.0 - 2.0 * (x * x + y * y))

    @property
    def collapsed(self) -> bool:
        height = float(self.qpos_after_f64[2])
        return bool(height < 1.0 or height > 2.0 or self.torso_up_z < 0.5)


@dataclass(frozen=True, slots=True)
class TargetSpeedEpisodeMetricsV1:
    evaluation_seed: int
    target_speed_m_s: float
    observed_steps: int
    requested_steps: int
    endpoint: float
    com_x_velocity_trace_sha256: str
    survived_full_horizon: bool
    native_terminated: bool
    truncated: bool
    first_failure_step: int | None
    collapse_step_count: int
    root_height_min_m: float
    root_height_max_m: float
    torso_up_z_min: float
    normalized_action_rms: float
    normalized_action_max_abs: float
    normalized_action_rate_rms_per_s: float
    normalized_action_rate_max_abs_per_s: float
    torque_rms_n_m: float
    torque_max_abs_n_m: float
    torque_capacity_normalized_max_abs: float
    energy_j: float
    mean_absolute_power_w: float
    non_foot_floor_contact_count: int
    non_foot_floor_contact_step_count: int
    non_foot_floor_contact_peak_force_n: float
    non_foot_floor_contact_identities: tuple[str, ...]
    allowed_foot_floor_contact_count: int
    allowed_foot_floor_peak_force_n: float
    allowed_foot_floor_peak_impulse_n_s: float
    allowed_foot_floor_total_impulse_n_s: float

    def __post_init__(self) -> None:
        if type(self.evaluation_seed) is not int or self.evaluation_seed < 0:
            raise TargetSpeedEvaluationError("evaluation_seed must be a non-negative integer")
        object.__setattr__(self, "target_speed_m_s", _target(self.target_speed_m_s))
        if (
            type(self.observed_steps) is not int
            or type(self.requested_steps) is not int
            or self.requested_steps != HORIZON_STEPS
            or not 0 < self.observed_steps <= self.requested_steps
        ):
            raise TargetSpeedEvaluationError("episode step counts differ from the frozen horizon")
        endpoint = _finite_float(self.endpoint, field="endpoint")
        if not 0.0 <= endpoint <= 1.0:
            raise TargetSpeedEvaluationError("endpoint must be in [0, 1]")
        if (
            type(self.com_x_velocity_trace_sha256) is not str
            or len(self.com_x_velocity_trace_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.com_x_velocity_trace_sha256
            )
        ):
            raise TargetSpeedEvaluationError("COM velocity trace hash is invalid")
        for field in ("survived_full_horizon", "native_terminated", "truncated"):
            if type(getattr(self, field)) is not bool:
                raise TargetSpeedEvaluationError(f"{field} must be boolean")
        if type(self.collapse_step_count) is not int or not 0 <= self.collapse_step_count <= (
            self.observed_steps
        ):
            raise TargetSpeedEvaluationError("collapse_step_count is invalid")
        if self.first_failure_step is not None and (
            type(self.first_failure_step) is not int
            or not 1 <= self.first_failure_step <= self.observed_steps
        ):
            raise TargetSpeedEvaluationError("first_failure_step is invalid")
        expected_survival = (
            self.observed_steps == HORIZON_STEPS
            and not self.native_terminated
            and self.collapse_step_count == 0
        )
        if self.survived_full_horizon is not expected_survival:
            raise TargetSpeedEvaluationError("survival fields are contradictory")
        if (self.first_failure_step is None) is not self.survived_full_horizon:
            raise TargetSpeedEvaluationError("first failure and survival are contradictory")
        if self.observed_steps < HORIZON_STEPS and not (self.native_terminated or self.truncated):
            raise TargetSpeedEvaluationError("an early episode requires an end flag")
        root_min = _finite_float(self.root_height_min_m, field="root_height_min_m")
        root_max = _finite_float(self.root_height_max_m, field="root_height_max_m")
        torso_min = _finite_float(self.torso_up_z_min, field="torso_up_z_min")
        if root_min > root_max:
            raise TargetSpeedEvaluationError("root height extrema are reversed")
        extrema_show_collapse = root_min < 1.0 or root_max > 2.0 or torso_min < 0.5
        if (self.collapse_step_count == 0) is not (not extrema_show_collapse):
            raise TargetSpeedEvaluationError("collapse count and protected extrema disagree")
        nonnegative = (
            "normalized_action_rms",
            "normalized_action_max_abs",
            "normalized_action_rate_rms_per_s",
            "normalized_action_rate_max_abs_per_s",
            "torque_rms_n_m",
            "torque_max_abs_n_m",
            "torque_capacity_normalized_max_abs",
            "energy_j",
            "mean_absolute_power_w",
            "non_foot_floor_contact_peak_force_n",
            "allowed_foot_floor_peak_force_n",
            "allowed_foot_floor_peak_impulse_n_s",
            "allowed_foot_floor_total_impulse_n_s",
        )
        for field in nonnegative:
            if _finite_float(getattr(self, field), field=field) < 0.0:
                raise TargetSpeedEvaluationError(f"{field} cannot be negative")
        if self.normalized_action_max_abs > 1.0:
            raise TargetSpeedEvaluationError("normalized action maximum exceeds one")
        if self.normalized_action_rate_max_abs_per_s > 2.0 / CONTROL_PERIOD_SECONDS:
            raise TargetSpeedEvaluationError("normalized action-rate maximum exceeds its bound")
        if self.torque_capacity_normalized_max_abs > 1.0 + TORQUE_CAPACITY_REL_TOLERANCE:
            raise TargetSpeedEvaluationError("normalized torque maximum exceeds its capacity")
        expected_power = self.energy_j / (CONTROL_PERIOD_SECONDS * self.observed_steps)
        if not math.isclose(
            self.mean_absolute_power_w,
            expected_power,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise TargetSpeedEvaluationError("energy and mean absolute power disagree")
        for field in (
            "non_foot_floor_contact_count",
            "non_foot_floor_contact_step_count",
            "allowed_foot_floor_contact_count",
        ):
            if type(getattr(self, field)) is not int or getattr(self, field) < 0:
                raise TargetSpeedEvaluationError(f"{field} must be a non-negative integer")
        if (
            not 0
            <= self.non_foot_floor_contact_step_count
            <= min(self.observed_steps, self.non_foot_floor_contact_count)
        ):
            raise TargetSpeedEvaluationError("non-foot contact step count is inconsistent")
        identities = self.non_foot_floor_contact_identities
        if (
            type(identities) is not tuple
            or any(type(item) is not str or not item for item in identities)
            or tuple(sorted(set(identities))) != identities
            or (self.non_foot_floor_contact_count == 0) is not (not identities)
        ):
            raise TargetSpeedEvaluationError("non-foot contact identities are inconsistent")
        expected_peak_impulse = self.allowed_foot_floor_peak_force_n * PHYSICS_TIMESTEP_SECONDS
        if (
            not math.isclose(
                self.allowed_foot_floor_peak_impulse_n_s,
                expected_peak_impulse,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or self.allowed_foot_floor_total_impulse_n_s < self.allowed_foot_floor_peak_impulse_n_s
        ):
            raise TargetSpeedEvaluationError("allowed-foot impact descriptives disagree")
        if self.allowed_foot_floor_contact_count == 0 and (
            self.allowed_foot_floor_peak_force_n != 0.0
            or self.allowed_foot_floor_total_impulse_n_s != 0.0
        ):
            raise TargetSpeedEvaluationError("empty allowed-foot contacts have nonzero impacts")

    @property
    def non_foot_floor_conforms(self) -> bool:
        return self.non_foot_floor_contact_count == 0


def _trace_sha256(values: Sequence[float]) -> str:
    array = np.ascontiguousarray(values, dtype=np.dtype(">f8"))
    digest = hashlib.sha256()
    digest.update(b"family-b-target-speed-v1/com-x-velocity/f64be")
    digest.update(len(values).to_bytes(8, "big"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class TargetSpeedEpisodeAccumulatorV1:
    """Accumulate one predetermined episode and zero-pad only after a real end."""

    def __init__(self, *, evaluation_seed: int, target_speed_m_s: float) -> None:
        if type(evaluation_seed) is not int or evaluation_seed < 0:
            raise TargetSpeedEvaluationError("evaluation_seed must be a non-negative integer")
        self._evaluation_seed = evaluation_seed
        self._target = _target(target_speed_m_s)
        self._steps: list[ProtectedTargetSpeedStepV1] = []
        self._sealed = False
        self._invalid = False

    def add(self, step: ProtectedTargetSpeedStepV1) -> None:
        if self._invalid:
            raise TargetSpeedEvaluationError("episode receipt was invalidated")
        try:
            if self._sealed:
                raise TargetSpeedEvaluationError("cannot add a transition after episode end")
            if not isinstance(step, ProtectedTargetSpeedStepV1):
                raise TargetSpeedEvaluationError("step must be ProtectedTargetSpeedStepV1")
            expected_index = len(self._steps) + 1
            if step.transition_index != expected_index:
                raise TargetSpeedEvaluationError("transition indices must be complete and ordered")
            if self._steps:
                previous = self._steps[-1]
                if not np.array_equal(previous.body_mass_f64, step.body_mass_f64):
                    raise TargetSpeedEvaluationError("body masses changed within the episode")
                if not np.array_equal(previous.body_xipos_after_f64, step.body_xipos_before_f64):
                    raise TargetSpeedEvaluationError("COM boundary state is discontinuous")
            previous_action = (
                np.zeros(17, dtype=np.float64)
                if not self._steps
                else self._steps[-1].normalized_action_f64
            )
            action_rate = (step.normalized_action_f64 - previous_action) / CONTROL_PERIOD_SECONDS
            if float(np.max(np.abs(action_rate))) > 2.0 / CONTROL_PERIOD_SECONDS:
                raise TargetSpeedEvaluationError(
                    "normalized action rate exceeds its structural bound"
                )
            self._steps.append(step)
            if step.terminated or step.truncated or len(self._steps) == HORIZON_STEPS:
                self._sealed = True
        except BaseException:
            self._invalid = True
            raise

    def finish(self) -> TargetSpeedEpisodeMetricsV1:
        if self._invalid:
            raise TargetSpeedEvaluationError("episode receipt was invalidated")
        if not self._steps:
            raise TargetSpeedEvaluationError("cannot finish an empty episode")
        if not self._sealed:
            raise TargetSpeedEvaluationError("partial episode without an end flag fails closed")
        steps = tuple(self._steps)
        velocities = [step.com_x_velocity_m_s for step in steps]
        progress_sum = sum(
            target_speed_progress(velocities[index - 1], self._target)
            for index in range(BURN_IN_STEPS + 1, len(steps) + 1)
        )
        endpoint = progress_sum / SCORED_STEPS
        actions = np.stack([step.normalized_action_f64 for step in steps])
        previous_actions = np.vstack((np.zeros((1, 17), dtype=np.float64), actions[:-1]))
        action_rates = (actions - previous_actions) / CONTROL_PERIOD_SECONDS
        torques = np.stack([step.generalized_actuator_torque_n_m_f64 for step in steps])
        capacities = np.asarray(TORQUE_CAPACITY_N_M, dtype=np.float64)
        joint_velocities = np.stack(
            [step.qvel_after_f64[list(ACTUATOR_QVEL_INDICES_BY_ACTION)] for step in steps]
        )
        absolute_power = np.abs(torques * joint_velocities)
        energy = CONTROL_PERIOD_SECONDS * float(np.sum(absolute_power))
        mean_absolute_power = energy / (CONTROL_PERIOD_SECONDS * len(steps))
        collapsed = [step.collapsed for step in steps]
        root_heights = [float(step.qpos_after_f64[2]) for step in steps]
        torso_up = [step.torso_up_z for step in steps]
        non_foot_contacts = []
        allowed_contacts = []
        non_foot_steps = 0
        for step in steps:
            step_has_non_foot = False
            for contact in step.contacts:
                other = contact.floor_other_geom
                if other is None:
                    continue
                if other in ALLOWED_FOOT_GEOM_NAMES:
                    allowed_contacts.append(contact)
                else:
                    non_foot_contacts.append(contact)
                    step_has_non_foot = True
            non_foot_steps += int(step_has_non_foot)
        non_foot_peak = max((contact.normal_force_n for contact in non_foot_contacts), default=0.0)
        allowed_peak = max((contact.normal_force_n for contact in allowed_contacts), default=0.0)
        allowed_impulses = [
            contact.normal_force_n * PHYSICS_TIMESTEP_SECONDS for contact in allowed_contacts
        ]
        final = steps[-1]
        first_failure = next(
            (
                step.transition_index
                for step in steps
                if step.collapsed
                or step.terminated
                or (step.truncated and step.transition_index < HORIZON_STEPS)
            ),
            None,
        )
        return TargetSpeedEpisodeMetricsV1(
            evaluation_seed=self._evaluation_seed,
            target_speed_m_s=self._target,
            observed_steps=len(steps),
            requested_steps=HORIZON_STEPS,
            endpoint=endpoint,
            com_x_velocity_trace_sha256=_trace_sha256(velocities),
            survived_full_horizon=(
                len(steps) == HORIZON_STEPS
                and not any(collapsed)
                and not any(step.terminated for step in steps)
            ),
            native_terminated=any(step.terminated for step in steps),
            truncated=final.truncated,
            first_failure_step=first_failure,
            collapse_step_count=sum(collapsed),
            root_height_min_m=min(root_heights),
            root_height_max_m=max(root_heights),
            torso_up_z_min=min(torso_up),
            normalized_action_rms=_rms(actions),
            normalized_action_max_abs=float(np.max(np.abs(actions))),
            normalized_action_rate_rms_per_s=_rms(action_rates),
            normalized_action_rate_max_abs_per_s=float(np.max(np.abs(action_rates))),
            torque_rms_n_m=_rms(torques),
            torque_max_abs_n_m=float(np.max(np.abs(torques))),
            torque_capacity_normalized_max_abs=float(np.max(np.abs(torques / capacities))),
            energy_j=energy,
            mean_absolute_power_w=mean_absolute_power,
            non_foot_floor_contact_count=len(non_foot_contacts),
            non_foot_floor_contact_step_count=non_foot_steps,
            non_foot_floor_contact_peak_force_n=non_foot_peak,
            non_foot_floor_contact_identities=tuple(
                sorted({str(contact.floor_other_geom) for contact in non_foot_contacts})
            ),
            allowed_foot_floor_contact_count=len(allowed_contacts),
            allowed_foot_floor_peak_force_n=allowed_peak,
            allowed_foot_floor_peak_impulse_n_s=max(allowed_impulses, default=0.0),
            allowed_foot_floor_total_impulse_n_s=sum(allowed_impulses),
        )


def evaluate_target_speed_episode(
    steps: Sequence[ProtectedTargetSpeedStepV1],
    *,
    evaluation_seed: int,
    target_speed_m_s: float,
) -> TargetSpeedEpisodeMetricsV1:
    if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
        raise TargetSpeedEvaluationError("steps must be a sequence")
    accumulator = TargetSpeedEpisodeAccumulatorV1(
        evaluation_seed=evaluation_seed,
        target_speed_m_s=target_speed_m_s,
    )
    for step in steps:
        accumulator.add(step)
    return accumulator.finish()


@dataclass(frozen=True, slots=True)
class ExploratoryGuardrailComparisonV1:
    paired_endpoint_differences: tuple[float, ...]
    relative_ratios_by_metric: Mapping[str, tuple[float, ...]]
    aggregate_relative_ratio_by_metric: Mapping[str, float]
    survival_checkpoint_passes: int
    non_foot_checkpoint_passes: int
    passed: bool
    margin: float = EXPLORATORY_RELATIVE_MARGIN
    label: str = "exploratory_relative_non_inferiority"


def _complete_seed_map(
    value: Mapping[int, Sequence[TargetSpeedEpisodeMetricsV1]], *, field: str
) -> dict[int, tuple[TargetSpeedEpisodeMetricsV1, ...]]:
    if set(value) != set(TRAINING_SEEDS):
        raise TargetSpeedEvaluationError(f"{field} is missing a predetermined training seed")
    result: dict[int, tuple[TargetSpeedEpisodeMetricsV1, ...]] = {}
    for seed in TRAINING_SEEDS:
        episodes = tuple(value[seed])
        if len(episodes) != len(EVALUATION_SEEDS):
            raise TargetSpeedEvaluationError(f"{field} seed {seed} is a partial result")
        observed = tuple(item.evaluation_seed for item in episodes)
        if observed != EVALUATION_SEEDS:
            raise TargetSpeedEvaluationError(f"{field} evaluation seeds differ or are reordered")
        result[seed] = episodes
    return result


def _median_for_gate(episodes: Sequence[TargetSpeedEpisodeMetricsV1], field: str) -> float:
    values = [getattr(item, field) if item.survived_full_horizon else math.inf for item in episodes]
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _relative_ratio(candidate: float, baseline: float) -> float:
    if not math.isfinite(candidate) or not math.isfinite(baseline):
        return math.inf
    if baseline > 0.0:
        return candidate / baseline
    if candidate == baseline == 0.0:
        return 1.0
    return math.inf


def compare_exploratory_guardrails(
    baseline: Mapping[int, Sequence[TargetSpeedEpisodeMetricsV1]],
    candidate: Mapping[int, Sequence[TargetSpeedEpisodeMetricsV1]],
) -> ExploratoryGuardrailComparisonV1:
    baseline_map = _complete_seed_map(baseline, field="baseline")
    candidate_map = _complete_seed_map(candidate, field="candidate")
    baseline_survival_checkpoints = sum(
        sum(item.survived_full_horizon for item in baseline_map[seed]) >= 19
        for seed in TRAINING_SEEDS
    )
    baseline_contact_checkpoints = sum(
        sum(item.non_foot_floor_conforms for item in baseline_map[seed]) >= 19
        for seed in TRAINING_SEEDS
    )
    if baseline_survival_checkpoints < 4 or baseline_contact_checkpoints < 4:
        raise TargetSpeedEvaluationError("stock baseline viability stop rule failed")
    fields = {
        "torque_exposure": "torque_rms_n_m",
        "energy": "energy_j",
        "action_rate": "normalized_action_rate_rms_per_s",
    }
    ratios: dict[str, tuple[float, ...]] = {}
    aggregate: dict[str, float] = {}
    relative_pass = True
    for label, field in fields.items():
        baseline_medians = [_median_for_gate(baseline_map[seed], field) for seed in TRAINING_SEEDS]
        candidate_medians = [
            _median_for_gate(candidate_map[seed], field) for seed in TRAINING_SEEDS
        ]
        metric_ratios = tuple(
            _relative_ratio(candidate_value, baseline_value)
            for candidate_value, baseline_value in zip(
                candidate_medians, baseline_medians, strict=True
            )
        )
        denominator = sum(baseline_medians)
        numerator = sum(candidate_medians)
        aggregate_ratio = _relative_ratio(numerator, denominator)
        ratios[label] = metric_ratios
        aggregate[label] = aggregate_ratio
        relative_pass = (
            relative_pass
            and sum(value <= 1.0 + EXPLORATORY_RELATIVE_MARGIN for value in metric_ratios) >= 4
            and aggregate_ratio <= 1.0 + EXPLORATORY_RELATIVE_MARGIN
        )

    survival_checkpoint_passes = 0
    contact_checkpoint_passes = 0
    for seed in TRAINING_SEEDS:
        baseline_survival = sum(item.survived_full_horizon for item in baseline_map[seed])
        candidate_survival = sum(item.survived_full_horizon for item in candidate_map[seed])
        if candidate_survival >= 19 and candidate_survival >= baseline_survival:
            survival_checkpoint_passes += 1
        baseline_contact = sum(item.non_foot_floor_conforms for item in baseline_map[seed])
        candidate_contact = sum(item.non_foot_floor_conforms for item in candidate_map[seed])
        if candidate_contact >= 19 and candidate_contact >= baseline_contact:
            contact_checkpoint_passes += 1
    endpoint_differences = tuple(
        float(np.mean([item.endpoint for item in candidate_map[seed]]))
        - float(np.mean([item.endpoint for item in baseline_map[seed]]))
        for seed in TRAINING_SEEDS
    )
    return ExploratoryGuardrailComparisonV1(
        paired_endpoint_differences=endpoint_differences,
        relative_ratios_by_metric=MappingProxyType(ratios),
        aggregate_relative_ratio_by_metric=MappingProxyType(aggregate),
        survival_checkpoint_passes=survival_checkpoint_passes,
        non_foot_checkpoint_passes=contact_checkpoint_passes,
        passed=(
            relative_pass and survival_checkpoint_passes >= 4 and contact_checkpoint_passes >= 4
        ),
    )


def capture_body_xipos_f64(environment: object) -> np.ndarray:
    physical = getattr(environment, "unwrapped", environment)
    data = getattr(physical, "data", None)
    if data is None:
        raise TargetSpeedEvaluationError("direct simulator state is unavailable")
    return _exact_f64_array(np.asarray(data.xipos), shape=(14, 3), field="body_xipos_f64")


def capture_protected_target_speed_step(
    environment: object,
    *,
    transition_index: int,
    body_xipos_before_f64: np.ndarray,
    normalized_action_f64: np.ndarray,
    terminated: bool,
    truncated: bool,
) -> ProtectedTargetSpeedStepV1:
    """Capture direct post-transition simulator facts from an instrumented adapter."""

    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise TargetSpeedEvaluationError("MuJoCo is required for protected capture") from exc
    physical = getattr(environment, "unwrapped", environment)
    model = getattr(physical, "model", None)
    data = getattr(physical, "data", None)
    samples = getattr(physical, "last_control_step_contact_samples", None)
    substeps = getattr(physical, "last_control_step_substeps", None)
    if model is None or data is None or type(samples) is not tuple:
        raise TargetSpeedEvaluationError("instrumented direct simulator state is unavailable")
    contacts = []
    for sample in samples:
        geom1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(sample.geom1_id))
        geom2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(sample.geom2_id))
        if not geom1 or not geom2:
            raise TargetSpeedEvaluationError("active contact contains an unnamed geometry")
        contacts.append(
            TargetSpeedSubstepContactV1(
                physics_substep_index=sample.physics_substep_index,
                contact_index_within_substep=sample.contact_index_within_substep,
                geom1_name=geom1,
                geom2_name=geom2,
                normal_force_n=sample.normal_force_n,
            )
        )
    return ProtectedTargetSpeedStepV1(
        transition_index=transition_index,
        body_mass_f64=np.asarray(model.body_mass),
        body_xipos_before_f64=body_xipos_before_f64,
        body_xipos_after_f64=np.asarray(data.xipos),
        qpos_after_f64=np.asarray(data.qpos),
        qvel_after_f64=np.asarray(data.qvel),
        normalized_action_f64=normalized_action_f64,
        generalized_actuator_torque_n_m_f64=np.asarray(data.qfrc_actuator)[
            list(ACTUATOR_QVEL_INDICES_BY_ACTION)
        ],
        contacts=tuple(contacts),
        captured_physics_substeps=substeps,
        control_period_s=float(physical.dt),
        physics_timestep_s=float(model.opt.timestep),
        terminated=terminated,
        truncated=truncated,
    )


__all__ = [
    "ACTUATOR_QVEL_INDICES_BY_ACTION",
    "ALLOWED_FOOT_GEOM_NAMES",
    "BURN_IN_STEPS",
    "CONTROL_PERIOD_SECONDS",
    "ENDPOINT_SIGMA_M_S",
    "EVALUATION_SEEDS",
    "EXPLORATORY_RELATIVE_MARGIN",
    "FAMILY_ID",
    "HORIZON_STEPS",
    "PHYSICS_SUBSTEPS",
    "PHYSICS_TIMESTEP_SECONDS",
    "SCORED_STEPS",
    "TARGET_SPEEDS_M_S",
    "TORQUE_CAPACITY_N_M",
    "TRAINING_SEEDS",
    "ExploratoryGuardrailComparisonV1",
    "ProtectedTargetSpeedStepV1",
    "TargetSpeedEpisodeAccumulatorV1",
    "TargetSpeedEpisodeMetricsV1",
    "TargetSpeedEvaluationError",
    "TargetSpeedSubstepContactV1",
    "capture_body_xipos_f64",
    "capture_protected_target_speed_step",
    "compare_exploratory_guardrails",
    "evaluate_target_speed_episode",
    "recompute_com_x_velocity_m_s",
    "target_speed_evaluator_source_sha256",
    "target_speed_progress",
]
