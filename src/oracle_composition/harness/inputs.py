"""Frozen library and task inputs for a composition cycle."""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .contract import EVIDENCE_CLASS, OracleContractError, read_json_object

LIBRARY_MANIFEST_ID = "humanoid_three_actor_library/v1"
TASK_SPEC_ID = "humanoid_speed_profile_t1/v1"
MAX_BOUND_ARTIFACT_BYTES = 128 * 1024 * 1024

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class InputContractError(OracleContractError):
    """Raised when a frozen library or task input is invalid."""


def _exact_dict(value: object, keys: set[str], *, field: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise InputContractError(f"{field} keys differ from the contract")
    return value


def _name(value: object, *, field: str) -> str:
    if type(value) is not str or _NAME.fullmatch(value) is None:
        raise InputContractError(f"{field} must be a lowercase snake-case identifier")
    return value


def _text(value: object, *, field: str, maximum: int = 4096) -> str:
    if type(value) is not str or not value.strip() or len(value) > maximum:
        raise InputContractError(f"{field} must be nonempty bounded text")
    return value


def _integer(value: object, *, field: str, minimum: int = 0, maximum: int = 1_000_000) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise InputContractError(f"{field} must be an integer in [{minimum}, {maximum}]")
    return value


def _number(value: object, *, field: str) -> float:
    if type(value) not in {int, float}:
        raise InputContractError(f"{field} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise InputContractError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise InputContractError(f"{field} must be finite")
    return result


def _sha256(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise InputContractError(f"{field} must be a lowercase SHA-256")
    return value


def _relative_path(value: object, *, field: str) -> str:
    text = _text(value, field=field, maximum=512)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or text != path.as_posix():
        raise InputContractError(f"{field} must be a normalized repository-relative path")
    return text


@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class SpeedStatistics:
    admitted_clip_count: int
    admitted_step_count: int
    q1_m_s: float
    median_m_s: float
    q3_m_s: float
    iqr_m_s: float


@dataclass(frozen=True, slots=True)
class FallStatistics:
    e3_episode_count: int
    e3_step_count: int
    fall_event_count: int
    falls_per_1000_steps: float
    fall_seeds: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class BehaviorManifestEntry:
    name: str
    strict_npz: ArtifactBinding
    import_receipt: ArtifactBinding
    equivalence_receipt: ArtifactBinding
    speed: SpeedStatistics
    falls: FallStatistics


@dataclass(frozen=True, slots=True)
class LibraryManifest:
    source_evidence: Mapping[str, ArtifactBinding]
    behaviors: tuple[BehaviorManifestEntry, ...]
    raw_sha256: str

    @property
    def behavior_names(self) -> tuple[str, ...]:
        return tuple(entry.name for entry in self.behaviors)

    def behavior(self, name: str) -> BehaviorManifestEntry:
        for entry in self.behaviors:
            if entry.name == name:
                return entry
        raise InputContractError(f"library behavior is undefined: {name}")


@dataclass(frozen=True, slots=True)
class ScheduleSegment:
    start: int
    stop: int
    target_m_s: float


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_text: str
    horizon_steps: int
    control_period_seconds: float
    schedule: tuple[ScheduleSegment, ...]
    seeds: tuple[int, ...]
    cycle_zero_oracle_ids: tuple[str, ...]
    trace_artifact_directory: str
    slow_behavior: str
    library_manifest_sha256: str
    raw_sha256: str

    def target_at(self, step: int) -> float:
        if type(step) is not int or not 0 <= step < self.horizon_steps:
            raise InputContractError("target step lies outside the task horizon")
        for segment in self.schedule:
            if segment.start <= step < segment.stop:
                return segment.target_m_s
        raise AssertionError("validated schedule does not cover the horizon")


def _artifact_binding(value: object, *, field: str) -> ArtifactBinding:
    item = _exact_dict(value, {"path", "sha256"}, field=field)
    return ArtifactBinding(
        path=_relative_path(item["path"], field=f"{field}.path"),
        sha256=_sha256(item["sha256"], field=f"{field}.sha256"),
    )


def _speed_statistics(value: object, *, field: str) -> SpeedStatistics:
    item = _exact_dict(
        value,
        {
            "admitted_clip_count",
            "admitted_step_count",
            "q1_m_s",
            "median_m_s",
            "q3_m_s",
            "iqr_m_s",
        },
        field=field,
    )
    result = SpeedStatistics(
        admitted_clip_count=_integer(
            item["admitted_clip_count"], field=f"{field}.admitted_clip_count", minimum=1
        ),
        admitted_step_count=_integer(
            item["admitted_step_count"], field=f"{field}.admitted_step_count", minimum=1
        ),
        q1_m_s=_number(item["q1_m_s"], field=f"{field}.q1_m_s"),
        median_m_s=_number(item["median_m_s"], field=f"{field}.median_m_s"),
        q3_m_s=_number(item["q3_m_s"], field=f"{field}.q3_m_s"),
        iqr_m_s=_number(item["iqr_m_s"], field=f"{field}.iqr_m_s"),
    )
    if not result.q1_m_s <= result.median_m_s <= result.q3_m_s:
        raise InputContractError(f"{field} quantiles are not ordered")
    if result.iqr_m_s < 0.0 or not math.isclose(
        result.iqr_m_s, result.q3_m_s - result.q1_m_s, rel_tol=0.0, abs_tol=1e-12
    ):
        raise InputContractError(f"{field}.iqr_m_s disagrees with q3 - q1")
    return result


def _fall_statistics(value: object, *, field: str) -> FallStatistics:
    item = _exact_dict(
        value,
        {
            "e3_episode_count",
            "e3_step_count",
            "fall_event_count",
            "falls_per_1000_steps",
            "fall_seeds",
        },
        field=field,
    )
    raw_seeds = item["fall_seeds"]
    if type(raw_seeds) is not list:
        raise InputContractError(f"{field}.fall_seeds must be an array")
    seeds = tuple(
        _integer(seed, field=f"{field}.fall_seeds", maximum=2_147_483_647) for seed in raw_seeds
    )
    if len(set(seeds)) != len(seeds):
        raise InputContractError(f"{field}.fall_seeds must be unique")
    result = FallStatistics(
        e3_episode_count=_integer(
            item["e3_episode_count"], field=f"{field}.e3_episode_count", minimum=1
        ),
        e3_step_count=_integer(item["e3_step_count"], field=f"{field}.e3_step_count", minimum=1),
        fall_event_count=_integer(item["fall_event_count"], field=f"{field}.fall_event_count"),
        falls_per_1000_steps=_number(
            item["falls_per_1000_steps"], field=f"{field}.falls_per_1000_steps"
        ),
        fall_seeds=seeds,
    )
    expected = 1000.0 * result.fall_event_count / result.e3_step_count
    if (
        result.fall_event_count != len(result.fall_seeds)
        or result.fall_event_count > result.e3_episode_count
        or not math.isclose(result.falls_per_1000_steps, expected, rel_tol=0.0, abs_tol=1e-15)
    ):
        raise InputContractError(f"{field} fall-rate fields disagree")
    return result


def load_library_manifest(path: Path) -> LibraryManifest:
    value, encoded = read_json_object(path)
    root = _exact_dict(
        value,
        {
            "schema_version",
            "manifest_id",
            "evidence_class",
            "source_evidence",
            "speed_measurement",
            "fall_measurement",
            "behaviors",
        },
        field="library manifest",
    )
    if root["schema_version"] != 1 or root["manifest_id"] != LIBRARY_MANIFEST_ID:
        raise InputContractError("library manifest identity differs")
    if root["evidence_class"] != EVIDENCE_CLASS:
        raise InputContractError("library manifest evidence class differs")
    _text(root["speed_measurement"], field="speed_measurement")
    _text(root["fall_measurement"], field="fall_measurement")
    sources = root["source_evidence"]
    if type(sources) is not dict or set(sources) != {
        "corpus_manifest_v2",
        "e3_certificate_v2",
        "validation_manifest_v2",
    }:
        raise InputContractError("source_evidence keys differ")
    source_evidence = MappingProxyType(
        {
            name: _artifact_binding(binding, field=f"source_evidence.{name}")
            for name, binding in sources.items()
        }
    )
    raw_behaviors = root["behaviors"]
    if type(raw_behaviors) is not list or not raw_behaviors:
        raise InputContractError("library behaviors must be a nonempty array")
    behaviors: list[BehaviorManifestEntry] = []
    for index, raw_behavior in enumerate(raw_behaviors):
        field = f"behaviors[{index}]"
        item = _exact_dict(
            raw_behavior,
            {
                "name",
                "strict_npz",
                "import_receipt",
                "equivalence_receipt",
                "speed_m_s",
                "e3_falls",
            },
            field=field,
        )
        behaviors.append(
            BehaviorManifestEntry(
                name=_name(item["name"], field=f"{field}.name"),
                strict_npz=_artifact_binding(item["strict_npz"], field=f"{field}.strict_npz"),
                import_receipt=_artifact_binding(
                    item["import_receipt"], field=f"{field}.import_receipt"
                ),
                equivalence_receipt=_artifact_binding(
                    item["equivalence_receipt"], field=f"{field}.equivalence_receipt"
                ),
                speed=_speed_statistics(item["speed_m_s"], field=f"{field}.speed_m_s"),
                falls=_fall_statistics(item["e3_falls"], field=f"{field}.e3_falls"),
            )
        )
    names = tuple(item.name for item in behaviors)
    if len(set(names)) != len(names):
        raise InputContractError("library behavior names must be unique")
    return LibraryManifest(
        source_evidence=source_evidence,
        behaviors=tuple(behaviors),
        raw_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _validate_runtime(value: object) -> tuple[int, float]:
    item = _exact_dict(
        value,
        {
            "environment_id",
            "terminate_when_unhealthy",
            "time_limit_steps",
            "frame_skip",
            "physics_timestep_seconds",
            "control_period_seconds",
            "observation",
            "stock_reward",
        },
        field="runtime",
    )
    expected_literals = {
        "environment_id": "Humanoid-v5",
        "terminate_when_unhealthy": False,
        "time_limit_steps": 1000,
        "frame_skip": 5,
        "observation": "stock_348d_excluding_root_xy",
        "stock_reward": "untouched_descriptive_task_return_only",
    }
    if any(item[key] != expected for key, expected in expected_literals.items()):
        raise InputContractError("runtime differs from the frozen Humanoid-v5 contract")
    timestep = _number(item["physics_timestep_seconds"], field="physics_timestep_seconds")
    control_period = _number(item["control_period_seconds"], field="control_period_seconds")
    if timestep != 0.003 or control_period != 0.015:
        raise InputContractError("runtime cadence differs from the frozen contract")
    return int(item["time_limit_steps"]), control_period


def load_task_spec(
    path: Path, *, library: LibraryManifest, expected_library_sha256: str | None = None
) -> TaskSpec:
    value, encoded = read_json_object(path)
    root = _exact_dict(
        value,
        {
            "schema_version",
            "task_spec_id",
            "evidence_class",
            "library_manifest_sha256",
            "task_text",
            "runtime",
            "target_schedule",
            "slow_selection",
            "evaluation",
            "trace_artifact_directory",
        },
        field="task spec",
    )
    if root["schema_version"] != 1 or root["task_spec_id"] != TASK_SPEC_ID:
        raise InputContractError("task spec identity differs")
    if root["evidence_class"] != EVIDENCE_CLASS:
        raise InputContractError("task spec evidence class differs")
    library_sha256 = _sha256(root["library_manifest_sha256"], field="library_manifest_sha256")
    expected_hash = expected_library_sha256 or library.raw_sha256
    if library_sha256 != expected_hash or library_sha256 != library.raw_sha256:
        raise InputContractError("task spec does not bind the loaded library manifest")
    _time_limit, control_period = _validate_runtime(root["runtime"])

    evaluation = _exact_dict(
        root["evaluation"],
        {
            "seeds",
            "episode_steps",
            "deterministic_actor_output",
            "speed_quantity",
            "primary_metric",
            "aggregation",
            "falls_excluded",
            "fall_definition",
            "first_fall_step",
            "task_return",
            "determinism_replay",
            "cycle_zero_oracle_ids",
        },
        field="evaluation",
    )
    expected_evaluation = {
        "deterministic_actor_output": "mean",
        "speed_quantity": "root_x_sidecar_delta_over_0.015_s",
        "primary_metric": "mean_absolute_speed_error_m_s",
        "aggregation": "median_over_predeclared_seeds",
        "falls_excluded": False,
        "fall_definition": "z_root_outside_open_1.0_2.0_or_torso_up_below_0.5",
        "first_fall_step": "boundary_index_0_reset_or_1_to_horizon_post_step",
        "task_return": "sum_stock_reward_descriptive_only",
        "determinism_replay": "repeat_first_oracle_all_seeds_and_require_equal_trace_hashes",
    }
    if any(evaluation[key] != expected for key, expected in expected_evaluation.items()):
        raise InputContractError("evaluation semantics differ from the frozen contract")
    horizon = _integer(
        evaluation["episode_steps"], field="evaluation.episode_steps", minimum=1, maximum=1000
    )
    raw_seeds = evaluation["seeds"]
    if type(raw_seeds) is not list or not raw_seeds:
        raise InputContractError("evaluation.seeds must be a nonempty array")
    seeds = tuple(
        _integer(seed, field="evaluation.seed", maximum=2_147_483_647) for seed in raw_seeds
    )
    if len(seeds) > 100 or len(set(seeds)) != len(seeds):
        raise InputContractError("evaluation seeds must be unique and bounded")
    raw_oracle_ids = evaluation["cycle_zero_oracle_ids"]
    if type(raw_oracle_ids) is not list:
        raise InputContractError("evaluation.cycle_zero_oracle_ids must be an array")
    cycle_zero_oracle_ids = tuple(
        _name(value, field="evaluation.cycle_zero_oracle_ids") for value in raw_oracle_ids
    )
    if cycle_zero_oracle_ids != (
        "single_fast",
        "single_slow",
        "playback",
        "handwritten",
    ):
        raise InputContractError("cycle-zero oracle ids differ from the four frozen arms")

    raw_schedule = root["target_schedule"]
    if type(raw_schedule) is not list or not raw_schedule:
        raise InputContractError("target_schedule must be a nonempty array")
    schedule: list[ScheduleSegment] = []
    expected_start = 0
    for index, raw_segment in enumerate(raw_schedule):
        field = f"target_schedule[{index}]"
        segment = _exact_dict(raw_segment, {"start", "stop", "target_m_s"}, field=field)
        start = _integer(segment["start"], field=f"{field}.start", maximum=1000)
        stop = _integer(segment["stop"], field=f"{field}.stop", minimum=1, maximum=1000)
        if start != expected_start or stop <= start:
            raise InputContractError("target schedule must be contiguous and increasing")
        schedule.append(
            ScheduleSegment(
                start=start,
                stop=stop,
                target_m_s=_number(segment["target_m_s"], field=f"{field}.target_m_s"),
            )
        )
        expected_start = stop
    if expected_start != horizon:
        raise InputContractError("target schedule does not cover the evaluation horizon")

    slow = _exact_dict(
        root["slow_selection"],
        {"rule", "expert_behavior", "chosen_behavior", "candidates"},
        field="slow_selection",
    )
    if slow["rule"] != "nonexpert_median_farthest_below_expert":
        raise InputContractError("slow-selection rule differs")
    expert_name = _name(slow["expert_behavior"], field="slow_selection.expert_behavior")
    chosen = _name(slow["chosen_behavior"], field="slow_selection.chosen_behavior")
    expert = library.behavior(expert_name)
    raw_candidates = slow["candidates"]
    if type(raw_candidates) is not list or len(raw_candidates) != len(library.behaviors) - 1:
        raise InputContractError("slow_selection must list every non-expert candidate")
    candidate_deltas: dict[str, float] = {}
    for index, raw_candidate in enumerate(raw_candidates):
        field = f"slow_selection.candidates[{index}]"
        candidate = _exact_dict(
            raw_candidate, {"behavior", "median_speed_m_s", "below_expert_m_s"}, field=field
        )
        name = _name(candidate["behavior"], field=f"{field}.behavior")
        if name == expert_name or name in candidate_deltas:
            raise InputContractError("slow-selection candidates are duplicate or expert")
        entry = library.behavior(name)
        median = _number(candidate["median_speed_m_s"], field=f"{field}.median_speed_m_s")
        below = _number(candidate["below_expert_m_s"], field=f"{field}.below_expert_m_s")
        expected_below = expert.speed.median_m_s - entry.speed.median_m_s
        if median != entry.speed.median_m_s or not math.isclose(
            below, expected_below, rel_tol=0.0, abs_tol=1e-12
        ):
            raise InputContractError("slow-selection candidate disagrees with library statistics")
        candidate_deltas[name] = below
    expected_chosen = max(candidate_deltas, key=candidate_deltas.__getitem__)
    if chosen != expected_chosen:
        raise InputContractError("chosen slow behavior is not farthest below expert")
    chosen_median = library.behavior(chosen).speed.median_m_s
    targets = tuple(segment.target_m_s for segment in schedule)
    if len(schedule) != 3 or targets != (
        expert.speed.median_m_s,
        chosen_median,
        expert.speed.median_m_s,
    ):
        raise InputContractError("speed profile does not use the frozen fast/slow/fast medians")

    return TaskSpec(
        task_text=_text(root["task_text"], field="task_text"),
        horizon_steps=horizon,
        control_period_seconds=control_period,
        schedule=tuple(schedule),
        seeds=seeds,
        cycle_zero_oracle_ids=cycle_zero_oracle_ids,
        trace_artifact_directory=_relative_path(
            root["trace_artifact_directory"], field="trace_artifact_directory"
        ),
        slow_behavior=chosen,
        library_manifest_sha256=library_sha256,
        raw_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _verify_artifact(repository_root: Path, binding: ArtifactBinding, *, field: str) -> None:
    candidate = repository_root / binding.path
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise InputContractError(f"{field} artifact is unavailable: {binding.path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise InputContractError(f"{field} artifact must be a regular non-linked file")
    if not 0 < before.st_size <= MAX_BOUND_ARTIFACT_BYTES:
        raise InputContractError(f"{field} artifact size is outside the bound")
    digest = hashlib.sha256()
    observed = 0
    try:
        with candidate.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                observed += len(chunk)
        after = candidate.lstat()
    except OSError as exc:
        raise InputContractError(f"{field} artifact cannot be read") from exc

    def identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)

    if identity(before) != identity(after) or observed != before.st_size:
        raise InputContractError(f"{field} artifact changed while it was read")
    if digest.hexdigest() != binding.sha256:
        raise InputContractError(f"{field} SHA-256 mismatch")


def verify_library_artifacts(repository_root: Path, library: LibraryManifest) -> None:
    """Verify every corpus receipt and actor byte binding before an environment reset."""

    root = Path(repository_root)
    for name, binding in library.source_evidence.items():
        _verify_artifact(root, binding, field=f"source_evidence.{name}")
    for entry in library.behaviors:
        _verify_artifact(root, entry.import_receipt, field=f"{entry.name}.import_receipt")
        _verify_artifact(root, entry.strict_npz, field=f"{entry.name}.strict_npz")
        _verify_artifact(root, entry.equivalence_receipt, field=f"{entry.name}.equivalence_receipt")


def load_frozen_inputs(experiment: Path) -> tuple[LibraryManifest, TaskSpec]:
    directory = Path(experiment)
    library = load_library_manifest(directory / "library_manifest_v1.json")
    task = load_task_spec(directory / "task_spec_v1.json", library=library)
    return library, task


__all__ = [
    "ArtifactBinding",
    "BehaviorManifestEntry",
    "FallStatistics",
    "InputContractError",
    "LibraryManifest",
    "ScheduleSegment",
    "SpeedStatistics",
    "TaskSpec",
    "load_frozen_inputs",
    "load_library_manifest",
    "load_task_spec",
    "verify_library_artifacts",
]
