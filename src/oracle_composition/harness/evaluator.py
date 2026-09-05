"""Independent cycle evaluator, content-addressed traces, and compact reports."""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.envs.humanoid import make_humanoid_env
from oracle_composition.sources.strict_tqc_actor_runtime import StrictTQCActorRuntime

from .contract import EVIDENCE_CLASS, OracleProgram, load_oracle_program, read_json_object
from .executor import EpisodeExecution, execute_episode, runtime_fingerprint
from .inputs import (
    BehaviorManifestEntry,
    LibraryManifest,
    TaskSpec,
    load_frozen_inputs,
    verify_library_artifacts,
)

REPORT_SCHEMA_ID = "humanoid_composition_cycle_report/v1"
TRACE_INDEX_SCHEMA_ID = "humanoid_composition_trace_index/v1"
CLAIM_CEILING = (
    "cycle_0_ran_on_the_frozen_plain_humanoid_v5_runtime_with_predeclared_controller_"
    "switching_arms_and_a_cycle_1_designer_prompt_only_no_oracle_quality_generalization_"
    "tracker_reward_or_humanoid_competence_claim"
)


class CycleEvaluationError(RuntimeError):
    """Raised when frozen cycle evaluation cannot proceed or stay deterministic."""


ActorLoader = Callable[[BehaviorManifestEntry, Path], object]
EnvironmentFactory = Callable[[], object]
FingerprintFactory = Callable[[object, TaskSpec], tuple[dict[str, object], str]]


def _default_actor_loader(entry: BehaviorManifestEntry, repository_root: Path) -> object:
    return StrictTQCActorRuntime.from_npz(
        repository_root / entry.strict_npz.path,
        expected_sha256=entry.strict_npz.sha256,
    )


@dataclass(frozen=True, slots=True)
class EvaluationDependencies:
    environment_factory: EnvironmentFactory = make_humanoid_env
    actor_loader: ActorLoader = _default_actor_loader
    fingerprint_factory: FingerprintFactory = runtime_fingerprint


@dataclass(frozen=True, slots=True)
class LoadedOracle:
    path: Path
    relative_path: str
    program: OracleProgram
    file_sha256: str


def _sha256_bytes(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write(path: Path, encoded: bytes, *, allow_identical: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if (
            allow_identical
            and path.is_file()
            and not path.is_symlink()
            and path.read_bytes() == encoded
        ):
            return
        raise CycleEvaluationError(f"refusing to overwrite existing artifact: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _safe_relative(path: Path, root: Path, *, field: str) -> str:
    try:
        return path.resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()
    except (OSError, ValueError) as exc:
        raise CycleEvaluationError(f"{field} must be inside the repository") from exc


def _validate_expected_inputs(
    *,
    experiment: Path,
    cycle: int,
    library: LibraryManifest,
    task: TaskSpec,
    loaded_oracles: Sequence[LoadedOracle],
) -> dict[str, object]:
    expected_path = experiment / "cycles" / f"cycle_{cycle}" / "expected_inputs.json"
    value, _encoded = read_json_object(expected_path)
    required = {
        "schema_version",
        "evidence_class",
        "cycle",
        "task_spec_sha256",
        "library_manifest_sha256",
        "designer_prompt_sha256",
        "prior_report_sha256",
        "oracle_policy",
        "predeclared_oracles",
    }
    if set(value) != required:
        raise CycleEvaluationError("expected_inputs.json keys differ")
    if (
        value["schema_version"] != 1
        or value["evidence_class"] != EVIDENCE_CLASS
        or value["cycle"] != cycle
        or value["task_spec_sha256"] != task.raw_sha256
        or value["library_manifest_sha256"] != library.raw_sha256
    ):
        raise CycleEvaluationError("expected cycle inputs differ from the loaded frozen inputs")
    prompt_path = experiment / "cycles" / f"cycle_{cycle}" / "designer_prompt.md"
    try:
        prompt_bytes = prompt_path.read_bytes()
    except OSError as exc:
        raise CycleEvaluationError("designer prompt is unavailable") from exc
    if value["designer_prompt_sha256"] != _sha256_bytes(prompt_bytes):
        raise CycleEvaluationError("designer prompt hash differs from expected_inputs.json")
    prior_hash = value["prior_report_sha256"]
    if cycle == 0:
        if prior_hash is not None:
            raise CycleEvaluationError("cycle 0 cannot bind a prior report")
    else:
        prior_path = experiment / "cycles" / f"cycle_{cycle - 1}" / f"report_{cycle - 1}.json"
        try:
            observed_prior = _sha256_bytes(prior_path.read_bytes())
        except OSError as exc:
            raise CycleEvaluationError("prior cycle report is unavailable") from exc
        if prior_hash != observed_prior:
            raise CycleEvaluationError("prior cycle report hash differs")
    policy = value["oracle_policy"]
    declarations = value["predeclared_oracles"]
    if type(declarations) is not list:
        raise CycleEvaluationError("predeclared_oracles must be an array")
    observed = [
        {
            "file_sha256": item.file_sha256,
            "oracle_id": item.program.oracle_id,
            "oracle_sha256": item.program.sha256,
            "path": item.relative_path,
        }
        for item in loaded_oracles
    ]
    if policy == "exact_predeclared_set":
        if declarations != observed or not observed:
            raise CycleEvaluationError("evaluated oracles differ from the predeclared set")
    elif policy == "one_designer_oracle":
        if declarations or len(observed) != 1:
            raise CycleEvaluationError("designer cycles require exactly one oracle")
    else:
        raise CycleEvaluationError("oracle policy is invalid")
    return value


def _write_trace(
    *, artifact_root: Path, cycle: int, execution: EpisodeExecution, oracle_id: str
) -> tuple[str, int]:
    relative = (
        Path(f"cycle_{cycle}")
        / "traces"
        / oracle_id
        / f"{execution.metrics.seed}-{execution.trace_sha256}.json"
    )
    destination = artifact_root / relative
    _atomic_write(destination, execution.trace_bytes, allow_identical=True)
    return relative.as_posix(), len(execution.trace_bytes)


def _summary_for_oracle(
    oracle: LoadedOracle,
    episode_rows: Sequence[Mapping[str, object]],
    behavior_names: Sequence[str],
) -> dict[str, object]:
    errors = np.asarray(
        [float(row["mean_absolute_speed_error_m_s"]) for row in episode_rows],
        dtype=np.float64,
    )
    returns = np.asarray([float(row["task_return"]) for row in episode_rows], dtype=np.float64)
    switches = np.asarray([int(row["switch_count"]) for row in episode_rows], dtype=np.int64)
    if not np.isfinite(errors).all() or not np.isfinite(returns).all():
        raise CycleEvaluationError("summary inputs are non-finite")
    q1, median, q3 = np.quantile(errors, [0.25, 0.5, 0.75])
    time_medians = {
        behavior: float(
            np.median([int(row["time_in_each_behavior_steps"][behavior]) for row in episode_rows])
        )
        for behavior in behavior_names
    }
    return {
        "episode_count": len(episode_rows),
        "fall_count": sum(bool(row["fall"]) for row in episode_rows),
        "median_mean_absolute_speed_error_m_s": float(median),
        "median_switch_count": float(np.median(switches)),
        "median_task_return": float(np.median(returns)),
        "median_time_in_each_behavior_steps": time_medians,
        "oracle_file_sha256": oracle.file_sha256,
        "oracle_id": oracle.program.oracle_id,
        "oracle_sha256": oracle.program.sha256,
        "q1_mean_absolute_speed_error_m_s": float(q1),
        "q3_mean_absolute_speed_error_m_s": float(q3),
    }


def _report_markdown(cycle: int, arms: Sequence[Mapping[str, object]]) -> bytes:
    lines = [
        f"# Composition cycle {cycle}",
        "",
        f"Evidence class: `{EVIDENCE_CLASS}`. Controller switching stands in for tracker following.",
        "",
        "| arm | episodes | median MAE (m/s) | falls | median switches | median task return |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in arms:
        lines.append(
            "| {oracle_id} | {episode_count} | {mae:.6f} | {fall_count} | "
            "{switches:.1f} | {task_return:.6f} |".format(
                oracle_id=arm["oracle_id"],
                episode_count=arm["episode_count"],
                mae=arm["median_mean_absolute_speed_error_m_s"],
                fall_count=arm["fall_count"],
                switches=arm["median_switch_count"],
                task_return=arm["median_task_return"],
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def evaluate_cycle(
    *,
    experiment: Path,
    cycle: int,
    oracle_paths: Sequence[Path],
    repository_root: Path,
    dependencies: EvaluationDependencies | None = None,
) -> tuple[Path, Path]:
    """Evaluate frozen oracles, repeat the first arm, and publish one cycle report."""

    if type(cycle) is not int or cycle < 0:
        raise CycleEvaluationError("cycle must be a non-negative integer")
    start = time.perf_counter()
    root = Path(repository_root).resolve(strict=True)
    experiment_path = Path(experiment).resolve(strict=True)
    output_directory = experiment_path / "cycles" / f"cycle_{cycle}"
    report_path = output_directory / f"report_{cycle}.json"
    markdown_path = output_directory / f"report_{cycle}.md"
    if report_path.exists() or markdown_path.exists():
        raise CycleEvaluationError(
            "cycle report already exists; refusing to overwrite frozen evidence"
        )
    library, task = load_frozen_inputs(experiment_path)
    verify_library_artifacts(root, library)
    loaded_oracles: list[LoadedOracle] = []
    for path in oracle_paths:
        candidate = Path(path).resolve(strict=True)
        relative = _safe_relative(candidate, experiment_path, field="oracle")
        program, file_sha256 = load_oracle_program(
            candidate, available_behaviors=library.behavior_names
        )
        loaded_oracles.append(
            LoadedOracle(
                path=candidate,
                relative_path=relative,
                program=program,
                file_sha256=file_sha256,
            )
        )
    if not loaded_oracles:
        raise CycleEvaluationError("at least one oracle is required")
    if len({item.program.oracle_id for item in loaded_oracles}) != len(loaded_oracles):
        raise CycleEvaluationError("oracle ids must be unique within a cycle")
    _validate_expected_inputs(
        experiment=experiment_path,
        cycle=cycle,
        library=library,
        task=task,
        loaded_oracles=loaded_oracles,
    )

    selected_dependencies = dependencies or EvaluationDependencies()
    actors = {
        entry.name: selected_dependencies.actor_loader(entry, root) for entry in library.behaviors
    }
    environment = selected_dependencies.environment_factory()
    try:
        fingerprint, fingerprint_sha256 = selected_dependencies.fingerprint_factory(
            environment, task
        )
        artifact_root = root / task.trace_artifact_directory
        episode_rows: list[dict[str, object]] = []
        index_entries: list[dict[str, object]] = []
        first_hashes: dict[int, str] = {}
        for oracle_index, oracle in enumerate(loaded_oracles):
            for seed in task.seeds:
                episode_start = time.perf_counter()
                execution = execute_episode(
                    environment=environment,
                    actors=actors,
                    program=oracle.program,
                    task=task,
                    seed=seed,
                    runtime_fingerprint_sha256=fingerprint_sha256,
                )
                trace_path, trace_bytes = _write_trace(
                    artifact_root=artifact_root,
                    cycle=cycle,
                    execution=execution,
                    oracle_id=oracle.program.oracle_id,
                )
                row = {
                    "episode_wall_time_seconds": time.perf_counter() - episode_start,
                    "oracle_file_sha256": oracle.file_sha256,
                    "oracle_id": oracle.program.oracle_id,
                    "oracle_sha256": oracle.program.sha256,
                    **execution.metrics.to_dict(),
                    "trace_byte_count": trace_bytes,
                    "trace_path": trace_path,
                    "trace_sha256": execution.trace_sha256,
                }
                episode_rows.append(row)
                index_entries.append(
                    {
                        "byte_count": trace_bytes,
                        "cycle": cycle,
                        "oracle_id": oracle.program.oracle_id,
                        "path": trace_path,
                        "seed": seed,
                        "sha256": execution.trace_sha256,
                    }
                )
                if oracle_index == 0:
                    first_hashes[seed] = execution.trace_sha256

        replay_oracle = loaded_oracles[0]
        replay_start = time.perf_counter()
        for seed in task.seeds:
            replay = execute_episode(
                environment=environment,
                actors=actors,
                program=replay_oracle.program,
                task=task,
                seed=seed,
                runtime_fingerprint_sha256=fingerprint_sha256,
            )
            if replay.trace_sha256 != first_hashes[seed]:
                raise CycleEvaluationError(
                    f"nondeterministic trace for {replay_oracle.program.oracle_id} seed {seed}"
                )
        determinism_wall_time = time.perf_counter() - replay_start
    finally:
        close = getattr(environment, "close", None)
        if callable(close):
            close()

    index = {
        "entries": sorted(index_entries, key=lambda item: (item["oracle_id"], item["seed"])),
        "entry_count": len(index_entries),
        "evidence_class": EVIDENCE_CLASS,
        "schema_version": 1,
        "trace_index_schema_id": TRACE_INDEX_SCHEMA_ID,
    }
    index_bytes = canonical_json_bytes(index)
    index_path = artifact_root / f"cycle_{cycle}" / "content_index.json"
    _atomic_write(index_path, index_bytes)
    index_relative = index_path.relative_to(root).as_posix()
    arms: list[dict[str, object]] = []
    for oracle in loaded_oracles:
        rows = [row for row in episode_rows if row["oracle_id"] == oracle.program.oracle_id]
        arms.append(_summary_for_oracle(oracle, rows, library.behavior_names))
    report = {
        "claim_ceiling": CLAIM_CEILING,
        "cycle": cycle,
        "determinism_check": {
            "all_trace_hashes_equal": True,
            "oracle_id": replay_oracle.program.oracle_id,
            "replayed_episode_count": len(task.seeds),
            "wall_time_seconds": determinism_wall_time,
        },
        "evidence_class": EVIDENCE_CLASS,
        "evaluation": {
            "episode_steps": task.horizon_steps,
            "fall_definition": "z_root outside (1.0, 2.0) or torso_up < 0.5",
            "falls_excluded": False,
            "forward_speed_quantity": "root-x boundary difference over 0.015 s",
            "primary_score": "median episode mean_absolute_speed_error_m_s",
            "seeds": list(task.seeds),
            "task_return": "sum of untouched stock Humanoid-v5 reward; descriptive only",
        },
        "generated_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "library_manifest_sha256": library.raw_sha256,
        "oracles": [
            {
                "file_sha256": oracle.file_sha256,
                "oracle_id": oracle.program.oracle_id,
                "oracle_sha256": oracle.program.sha256,
                "path": oracle.relative_path,
            }
            for oracle in loaded_oracles
        ],
        "per_episode": episode_rows,
        "report_schema_id": REPORT_SCHEMA_ID,
        "runtime_fingerprint": fingerprint,
        "runtime_fingerprint_sha256": fingerprint_sha256,
        "schema_version": 1,
        "summary": {"arms": arms},
        "task_spec_sha256": task.raw_sha256,
        "trace_content_index": {
            "entry_count": len(index_entries),
            "path": index_relative,
            "sha256": _sha256_bytes(index_bytes),
        },
        "wall_time_seconds": time.perf_counter() - start,
    }
    _atomic_write(report_path, canonical_json_bytes(report))
    _atomic_write(markdown_path, _report_markdown(cycle, arms))
    return report_path, markdown_path


__all__ = [
    "CLAIM_CEILING",
    "CycleEvaluationError",
    "EvaluationDependencies",
    "evaluate_cycle",
]
