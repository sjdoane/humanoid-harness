"""Pure metrics for the frozen one-checkpoint TQC development screen."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import InitVar, asdict, dataclass
from numbers import Real
from statistics import median

import numpy as np

from .fixed_reference import ExperimentContractError
from .tqc_development_contract import EVALUATION_SEEDS

EXPECTED_STEPS = 1_000
CONTROL_PERIOD_SECONDS = 0.015
HEALTHY_HEIGHT_OPEN_INTERVAL_M = (1.0, 2.0)
MINIMUM_TORSO_UP_Z = 0.5
ACTION_WIDTH = 17

# Process-local constructor capability. It prevents accidental receipt forgery,
# but is not a cryptographic boundary against code already running in-process.
_METRIC_RECORD_ISSUER = object()


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ExperimentContractError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ExperimentContractError(f"{field} must be a finite number")
    return result


def _vector(value: object, *, width: int, field: str) -> tuple[float, ...]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExperimentContractError(f"{field} must be a finite ({width},) vector") from exc
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ExperimentContractError(f"{field} must be a finite ({width},) vector")
    return tuple(float(item) for item in array)


@dataclass(frozen=True, slots=True)
class TQCDevelopmentStepFacts:
    """Reward-independent facts for one completed control interval."""

    step_index: int
    simulation_time_seconds: float
    root_position_world_m: tuple[float, float, float]
    torso_up_z: float
    normalized_action: tuple[float, ...]
    non_foot_floor_contact: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.step_index, int)
            or isinstance(self.step_index, bool)
            or self.step_index < 1
        ):
            raise ExperimentContractError("step_index must be a positive integer")
        object.__setattr__(
            self,
            "simulation_time_seconds",
            _finite(self.simulation_time_seconds, field="simulation_time_seconds"),
        )
        root = _vector(self.root_position_world_m, width=3, field="root_position_world_m")
        object.__setattr__(self, "root_position_world_m", root)
        up_z = _finite(self.torso_up_z, field="torso_up_z")
        if not -1.0 <= up_z <= 1.0:
            raise ExperimentContractError("torso_up_z must be in [-1, 1]")
        object.__setattr__(self, "torso_up_z", up_z)
        action = _vector(self.normalized_action, width=ACTION_WIDTH, field="normalized_action")
        if any(abs(item) > 1.0 for item in action):
            raise ExperimentContractError("normalized_action must be in [-1, 1]")
        object.__setattr__(self, "normalized_action", action)
        if not isinstance(self.non_foot_floor_contact, bool):
            raise ExperimentContractError("non_foot_floor_contact must be boolean")


@dataclass(frozen=True, slots=True)
class TQCDevelopmentEpisodeMetrics:
    """The exact required metrics for one predetermined reset."""

    seed: int
    observed_steps: int
    finite_state_action_fraction: float
    healthy_step_fraction: float
    first_unhealthy_step: int | None
    upright_step_fraction: float
    first_not_upright_step: int | None
    root_height_min_m: float
    root_height_max_m: float
    net_forward_displacement_m: float
    time_average_forward_velocity_m_s: float
    root_lateral_displacement_max_abs_m: float
    normalized_action_rms: float
    normalized_action_saturation_fraction: float
    non_foot_floor_contact_step_fraction: float
    full_horizon_healthy: bool
    full_horizon_upright: bool
    initial_simulation_time_seconds: float
    final_simulation_time_seconds: float
    elapsed_simulation_time_seconds: float
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _METRIC_RECORD_ISSUER:
            raise ExperimentContractError(
                "episode metric records may only be issued by the protected metric core"
            )
        if type(self.seed) is not int or self.seed < 0:
            raise ExperimentContractError("seed must be a non-negative integer")
        if type(self.observed_steps) is not int or self.observed_steps != EXPECTED_STEPS:
            raise ExperimentContractError(f"observed_steps must equal {EXPECTED_STEPS}")
        optional_steps = (self.first_unhealthy_step, self.first_not_upright_step)
        if any(
            value is not None and (type(value) is not int or not 1 <= value <= self.observed_steps)
            for value in optional_steps
        ):
            raise ExperimentContractError("first-failure steps must be one-based or null")
        if (
            type(self.full_horizon_healthy) is not bool
            or type(self.full_horizon_upright) is not bool
        ):
            raise ExperimentContractError("full-horizon flags must be exact booleans")
        fraction_fields = (
            "finite_state_action_fraction",
            "healthy_step_fraction",
            "upright_step_fraction",
            "normalized_action_saturation_fraction",
            "non_foot_floor_contact_step_fraction",
        )
        for field in fraction_fields:
            raw = getattr(self, field)
            if type(raw) is not float:
                raise ExperimentContractError(f"{field} must be an exact float")
            value = _finite(raw, field=field)
            if not 0.0 <= value <= 1.0:
                raise ExperimentContractError(f"{field} must be in [0, 1]")
        scalar_fields = (
            "root_height_min_m",
            "root_height_max_m",
            "net_forward_displacement_m",
            "time_average_forward_velocity_m_s",
            "root_lateral_displacement_max_abs_m",
            "normalized_action_rms",
            "initial_simulation_time_seconds",
            "final_simulation_time_seconds",
            "elapsed_simulation_time_seconds",
        )
        for field in scalar_fields:
            raw = getattr(self, field)
            if type(raw) is not float:
                raise ExperimentContractError(f"{field} must be an exact float")
            _finite(raw, field=field)
        if self.root_height_min_m > self.root_height_max_m:
            raise ExperimentContractError("root-height range is inverted")
        if self.root_lateral_displacement_max_abs_m < 0.0:
            raise ExperimentContractError("lateral displacement cannot be negative")
        if not 0.0 <= self.normalized_action_rms <= 1.0:
            raise ExperimentContractError("normalized_action_rms must be in [0, 1]")
        expected_elapsed = self.final_simulation_time_seconds - (
            self.initial_simulation_time_seconds
        )
        if not math.isclose(
            self.elapsed_simulation_time_seconds,
            expected_elapsed,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ExperimentContractError("elapsed simulation time is inconsistent")
        if not math.isclose(
            self.elapsed_simulation_time_seconds,
            self.observed_steps * CONTROL_PERIOD_SECONDS,
            rel_tol=0.0,
            abs_tol=1e-10,
        ):
            raise ExperimentContractError("episode did not cover the exact simulation horizon")
        if self.finite_state_action_fraction != 1.0:
            raise ExperimentContractError("non-finite episode facts cannot produce a receipt")
        if self.full_horizon_healthy is not (
            self.first_unhealthy_step is None and self.healthy_step_fraction == 1.0
        ):
            raise ExperimentContractError("full_horizon_healthy is inconsistent")
        if (self.first_unhealthy_step is None) is not (self.healthy_step_fraction == 1.0):
            raise ExperimentContractError("first_unhealthy_step is inconsistent")
        if self.full_horizon_upright is not (
            self.first_not_upright_step is None and self.upright_step_fraction == 1.0
        ):
            raise ExperimentContractError("full_horizon_upright is inconsistent")
        if (self.first_not_upright_step is None) is not (self.upright_step_fraction == 1.0):
            raise ExperimentContractError("first_not_upright_step is inconsistent")
        if not math.isclose(
            self.time_average_forward_velocity_m_s,
            self.net_forward_displacement_m / self.elapsed_simulation_time_seconds,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ExperimentContractError("forward displacement and velocity are inconsistent")
        for field, denominator in (
            ("healthy_step_fraction", EXPECTED_STEPS),
            ("upright_step_fraction", EXPECTED_STEPS),
            ("normalized_action_saturation_fraction", EXPECTED_STEPS * ACTION_WIDTH),
            ("non_foot_floor_contact_step_fraction", EXPECTED_STEPS),
        ):
            count = getattr(self, field) * denominator
            if not math.isclose(count, round(count), rel_tol=0.0, abs_tol=1e-12):
                raise ExperimentContractError(f"{field} is not an exact sample fraction")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _initial_state_sha256(
    *,
    seed: int,
    root_position_world_m: tuple[float, float, float],
    simulation_time_seconds: float,
) -> str:
    payload = {
        "initial_root_position_world_m": list(root_position_world_m),
        "initial_simulation_time_seconds": simulation_time_seconds,
        "seed": seed,
    }
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True, init=False)
class TQCDevelopmentEpisodeAccumulator:
    """Accumulate all 1000 post-step facts without episode selection."""

    _seed: int
    _initial_root: tuple[float, float, float]
    _initial_time: float
    _initial_state_sha256: str
    _facts: list[TQCDevelopmentStepFacts]

    def __init__(
        self,
        *,
        seed: int,
        initial_root_position_world_m: object,
        initial_simulation_time_seconds: float,
    ) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ExperimentContractError("seed must be a non-negative integer")
        initial_root = _vector(
            initial_root_position_world_m,
            width=3,
            field="initial_root_position_world_m",
        )
        initial_time = _finite(
            initial_simulation_time_seconds,
            field="initial_simulation_time_seconds",
        )
        binding = _initial_state_sha256(
            seed=seed,
            root_position_world_m=initial_root,
            simulation_time_seconds=initial_time,
        )
        object.__setattr__(self, "_seed", seed)
        object.__setattr__(self, "_initial_root", initial_root)
        object.__setattr__(self, "_initial_time", initial_time)
        object.__setattr__(self, "_initial_state_sha256", binding)
        object.__setattr__(self, "_facts", [])

    def _validate_initial_state(self) -> None:
        if type(self._seed) is not int or self._seed < 0:
            raise ExperimentContractError("stored episode seed is invalid")
        if (
            type(self._initial_root) is not tuple
            or len(self._initial_root) != 3
            or any(type(value) is not float for value in self._initial_root)
            or not all(math.isfinite(value) for value in self._initial_root)
        ):
            raise ExperimentContractError("stored episode initial root is invalid")
        if type(self._initial_time) is not float or not math.isfinite(self._initial_time):
            raise ExperimentContractError("stored episode initial time is invalid")
        expected = _initial_state_sha256(
            seed=self._seed,
            root_position_world_m=self._initial_root,
            simulation_time_seconds=self._initial_time,
        )
        if type(self._initial_state_sha256) is not str or self._initial_state_sha256 != expected:
            raise ExperimentContractError("stored episode initial state binding is invalid")

    def add(self, facts: TQCDevelopmentStepFacts) -> None:
        self._validate_initial_state()
        if type(facts) is not TQCDevelopmentStepFacts:
            raise ExperimentContractError("episode facts have the wrong type")
        expected_index = len(self._facts) + 1
        if facts.step_index != expected_index:
            raise ExperimentContractError("step facts must be contiguous and one-based")
        expected_time = self._initial_time + expected_index * CONTROL_PERIOD_SECONDS
        if not math.isclose(
            facts.simulation_time_seconds,
            expected_time,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ExperimentContractError("step simulation time differs from the control clock")
        if len(self._facts) >= EXPECTED_STEPS:
            raise ExperimentContractError("episode contains more than 1000 steps")
        self._facts.append(facts)

    def finish(self) -> TQCDevelopmentEpisodeMetrics:
        self._validate_initial_state()
        if len(self._facts) != EXPECTED_STEPS:
            raise ExperimentContractError("episode must contain exactly 1000 steps")
        previous_time = self._initial_time
        for expected_index, fact in enumerate(self._facts, start=1):
            if type(fact) is not TQCDevelopmentStepFacts or fact.step_index != expected_index:
                raise ExperimentContractError("stored step facts are not contiguous and one-based")
            expected_time = self._initial_time + expected_index * CONTROL_PERIOD_SECONDS
            if fact.simulation_time_seconds <= previous_time or not math.isclose(
                fact.simulation_time_seconds,
                expected_time,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ExperimentContractError("stored step simulation time differs from the clock")
            previous_time = fact.simulation_time_seconds
        heights = np.asarray(
            [fact.root_position_world_m[2] for fact in self._facts],
            dtype=np.float64,
        )
        up_z = np.asarray([fact.torso_up_z for fact in self._facts], dtype=np.float64)
        actions = np.asarray([fact.normalized_action for fact in self._facts], dtype=np.float64)
        forbidden = np.asarray(
            [fact.non_foot_floor_contact for fact in self._facts],
            dtype=np.bool_,
        )
        healthy = (heights > HEALTHY_HEIGHT_OPEN_INTERVAL_M[0]) & (
            heights < HEALTHY_HEIGHT_OPEN_INTERVAL_M[1]
        )
        upright = up_z >= MINIMUM_TORSO_UP_Z
        first_unhealthy = np.flatnonzero(~healthy)
        first_not_upright = np.flatnonzero(~upright)
        final = self._facts[-1]
        elapsed = final.simulation_time_seconds - self._initial_time
        forward = final.root_position_world_m[0] - self._initial_root[0]
        lateral = max(
            abs(fact.root_position_world_m[1] - self._initial_root[1]) for fact in self._facts
        )
        return TQCDevelopmentEpisodeMetrics(
            seed=self._seed,
            observed_steps=len(self._facts),
            finite_state_action_fraction=1.0,
            healthy_step_fraction=float(np.mean(healthy)),
            first_unhealthy_step=(int(first_unhealthy[0]) + 1 if first_unhealthy.size else None),
            upright_step_fraction=float(np.mean(upright)),
            first_not_upright_step=(
                int(first_not_upright[0]) + 1 if first_not_upright.size else None
            ),
            root_height_min_m=float(np.min(heights)),
            root_height_max_m=float(np.max(heights)),
            net_forward_displacement_m=float(forward),
            time_average_forward_velocity_m_s=float(forward / elapsed),
            root_lateral_displacement_max_abs_m=float(lateral),
            normalized_action_rms=float(math.sqrt(float(np.mean(np.square(actions))))),
            normalized_action_saturation_fraction=float(np.mean(np.abs(actions) == 1.0)),
            non_foot_floor_contact_step_fraction=float(np.mean(forbidden)),
            full_horizon_healthy=bool(np.all(healthy)),
            full_horizon_upright=bool(np.all(upright)),
            initial_simulation_time_seconds=self._initial_time,
            final_simulation_time_seconds=final.simulation_time_seconds,
            elapsed_simulation_time_seconds=float(elapsed),
            _issuer=_METRIC_RECORD_ISSUER,
        )


def _derive_cohort_values(
    episodes: tuple[TQCDevelopmentEpisodeMetrics, ...],
) -> dict[str, int | float | bool]:
    healthy_count = sum(episode.full_horizon_healthy for episode in episodes)
    upright_count = sum(episode.full_horizon_upright for episode in episodes)
    median_velocity = float(
        median(episode.time_average_forward_velocity_m_s for episode in episodes)
    )
    forward_count = sum(episode.net_forward_displacement_m >= 5.0 for episode in episodes)
    return {
        "full_horizon_healthy_episode_count": healthy_count,
        "full_horizon_upright_episode_count": upright_count,
        "median_time_average_forward_velocity_m_s": median_velocity,
        "episode_count_with_net_forward_displacement_at_least_5_m": forward_count,
        "development_behavior_gate_passed": (
            healthy_count == 20
            and upright_count == 20
            and median_velocity >= 0.5
            and forward_count >= 18
        ),
    }


@dataclass(frozen=True, slots=True)
class TQCDevelopmentCohortMetrics:
    """Predeclared aggregate over all twenty fixed reset seeds."""

    evaluation_seeds: tuple[int, ...]
    episodes: tuple[TQCDevelopmentEpisodeMetrics, ...]
    full_horizon_healthy_episode_count: int
    full_horizon_upright_episode_count: int
    median_time_average_forward_velocity_m_s: float
    episode_count_with_net_forward_displacement_at_least_5_m: int
    development_behavior_gate_passed: bool
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _METRIC_RECORD_ISSUER:
            raise ExperimentContractError(
                "cohort records may only be issued by the protected metric core"
            )
        if type(self.evaluation_seeds) is not tuple or type(self.episodes) is not tuple:
            raise ExperimentContractError("cohort seeds and episodes must be exact tuples")
        if len(self.evaluation_seeds) != 20 or len(self.episodes) != 20:
            raise ExperimentContractError("development cohort must contain exactly twenty seeds")
        if self.evaluation_seeds != EVALUATION_SEEDS:
            raise ExperimentContractError("cohort seeds differ from the frozen evaluation order")
        if any(type(seed) is not int or seed < 0 for seed in self.evaluation_seeds):
            raise ExperimentContractError("cohort seeds must be non-negative exact integers")
        if len(set(self.evaluation_seeds)) != 20:
            raise ExperimentContractError("cohort seeds must be unique")
        if any(type(episode) is not TQCDevelopmentEpisodeMetrics for episode in self.episodes):
            raise ExperimentContractError("cohort episodes must be issued metric records")
        if tuple(episode.seed for episode in self.episodes) != self.evaluation_seeds:
            raise ExperimentContractError("cohort episodes differ from the declared seed order")
        for field in (
            "full_horizon_healthy_episode_count",
            "full_horizon_upright_episode_count",
            "episode_count_with_net_forward_displacement_at_least_5_m",
        ):
            value = getattr(self, field)
            if type(value) is not int or not 0 <= value <= 20:
                raise ExperimentContractError(f"{field} must be an exact count in [0, 20]")
        if type(self.median_time_average_forward_velocity_m_s) is not float or not math.isfinite(
            self.median_time_average_forward_velocity_m_s
        ):
            raise ExperimentContractError("cohort median velocity must be an exact finite float")
        if type(self.development_behavior_gate_passed) is not bool:
            raise ExperimentContractError("development behavior gate must be an exact boolean")
        derived = _derive_cohort_values(self.episodes)
        for field, expected in derived.items():
            if type(getattr(self, field)) is not type(expected) or getattr(self, field) != expected:
                raise ExperimentContractError(f"cohort field {field} differs from its episodes")

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "episodes": [episode.to_dict() for episode in self.episodes],
        }


def summarize_tqc_development_cohort(
    episodes: tuple[TQCDevelopmentEpisodeMetrics, ...],
    *,
    expected_seeds: tuple[int, ...],
) -> TQCDevelopmentCohortMetrics:
    """Apply the fixed 20/20, 20/20, median, and 18/20 gates."""

    if type(episodes) is not tuple or type(expected_seeds) is not tuple:
        raise ExperimentContractError("cohort inputs must be exact tuples")
    if expected_seeds != EVALUATION_SEEDS:
        raise ExperimentContractError("expected seeds differ from the frozen evaluation order")
    observed = episodes
    if tuple(episode.seed for episode in observed) != expected_seeds:
        raise ExperimentContractError("evaluation seeds are missing, duplicated, or out of order")
    if len(observed) != 20 or len(set(expected_seeds)) != 20:
        raise ExperimentContractError("development evaluation requires exactly twenty seeds")
    if any(type(episode) is not TQCDevelopmentEpisodeMetrics for episode in observed):
        raise ExperimentContractError("cohort inputs must be issued episode metric records")
    derived = _derive_cohort_values(observed)
    return TQCDevelopmentCohortMetrics(
        evaluation_seeds=expected_seeds,
        episodes=observed,
        **derived,
        _issuer=_METRIC_RECORD_ISSUER,
    )
