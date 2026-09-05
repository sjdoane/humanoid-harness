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
    ScheduleSegment,
    TaskSpec,
    load_frozen_inputs,
    verify_library_artifacts,
)

REPORT_SCHEMA_ID = "humanoid_composition_cycle_report/v1"
TRACE_INDEX_SCHEMA_ID = "humanoid_composition_trace_index/v1"
CLAIM_CEILING = (
    "exploratory_controller_switching_cycle_on_the_frozen_plain_humanoid_v5_runtime_"
    "only_no_oracle_quality_generalization_tracker_reference_following_reward_naturalness_"
    "robustness_or_humanoid_competence_claim"
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


def _cycle_zero_comparison(
    *, experiment: Path, cycle: int, library: LibraryManifest, task: TaskSpec
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    if cycle == 0:
        return [], None
    report_path = experiment / "cycles/cycle_0/report_0.json"
    report, encoded = read_json_object(report_path)
    if (
        report.get("cycle") != 0
        or report.get("evidence_class") != EVIDENCE_CLASS
        or report.get("task_spec_sha256") != task.raw_sha256
        or report.get("library_manifest_sha256") != library.raw_sha256
    ):
        raise CycleEvaluationError("cycle-0 report does not bind the frozen task and library")
    summary = report.get("summary")
    if type(summary) is not dict or type(summary.get("arms")) is not list:
        raise CycleEvaluationError("cycle-0 report summary is malformed")
    by_id: dict[str, dict[str, object]] = {}
    for raw_arm in summary["arms"]:
        if type(raw_arm) is not dict or type(raw_arm.get("oracle_id")) is not str:
            raise CycleEvaluationError("cycle-0 report arm is malformed")
        oracle_id = raw_arm["oracle_id"]
        if oracle_id in by_id:
            raise CycleEvaluationError("cycle-0 report oracle ids are not unique")
        by_id[oracle_id] = dict(raw_arm)
    if set(by_id) != set(task.cycle_zero_oracle_ids):
        raise CycleEvaluationError("cycle-0 report differs from the frozen baseline arms")
    arms = [by_id[oracle_id] for oracle_id in task.cycle_zero_oracle_ids]
    binding = {
        "arm_count": len(arms),
        "cycle": 0,
        "path": report_path.relative_to(experiment).as_posix(),
        "sha256": _sha256_bytes(encoded),
    }
    return arms, binding


def _slow_third(task: TaskSpec) -> ScheduleSegment:
    minimum = min(segment.target_m_s for segment in task.schedule)
    candidates = [segment for segment in task.schedule if segment.target_m_s == minimum]
    if len(candidates) != 1:
        raise CycleEvaluationError("task must define one unique slow-third segment")
    return candidates[0]


def _episode_diagnostics(
    *,
    execution: EpisodeExecution,
    program: OracleProgram,
    task: TaskSpec,
    behavior_names: Sequence[str],
) -> tuple[list[dict[str, object]], dict[str, float]]:
    samples = execution.metric_samples
    if any(sample.t != index for index, sample in enumerate(samples)):
        raise CycleEvaluationError("metric samples are not a contiguous episode")
    first_fall_step = execution.metrics.first_fall_step
    previous_behavior = program.states[program.initial].behavior
    switches: list[dict[str, object]] = []
    for sample in samples:
        if sample.controller_switched:
            switch_speed = 0.0 if sample.t == 0 else samples[sample.t - 1].forward_speed_m_s
            switches.append(
                {
                    "fall_followed_within_100_steps": (
                        first_fall_step is not None and sample.t < first_fall_step <= sample.t + 100
                    ),
                    "from_behavior": previous_behavior,
                    "step": sample.t,
                    "to_behavior": sample.active_behavior,
                    "v_x_m_s": switch_speed,
                }
            )
        previous_behavior = sample.active_behavior
    if len(switches) != execution.metrics.switch_count:
        raise CycleEvaluationError("controller-switch diagnostics differ from episode metrics")

    slow = _slow_third(task)
    slow_samples = [sample for sample in samples if slow.start <= sample.t < slow.stop]
    denominator = slow.stop - slow.start
    if len(slow_samples) != denominator:
        raise CycleEvaluationError("metric samples do not cover the frozen slow third")
    fractions = {
        behavior: sum(sample.active_behavior == behavior for sample in slow_samples) / denominator
        for behavior in behavior_names
    }
    return switches, fractions


def _metric_outcome(delta: float) -> str:
    if delta < 0.0:
        return "improved"
    if delta > 0.0:
        return "worsened"
    return "matched"


def _comparison_outcomes(
    *, baseline_arms: Sequence[Mapping[str, object]], current_arms: Sequence[Mapping[str, object]]
) -> dict[str, object] | None:
    if not baseline_arms:
        return None
    if len(current_arms) != 1:
        raise CycleEvaluationError("designer cycles require one current arm for comparison")
    candidate = current_arms[0]
    candidate_falls = int(candidate["fall_count"])
    candidate_mae = float(candidate["median_mean_absolute_speed_error_m_s"])
    comparisons: list[dict[str, object]] = []
    for baseline in baseline_arms:
        fall_delta = candidate_falls - int(baseline["fall_count"])
        mae_delta = candidate_mae - float(baseline["median_mean_absolute_speed_error_m_s"])
        comparisons.append(
            {
                "baseline_oracle_id": baseline["oracle_id"],
                "fall_count_delta": fall_delta,
                "fall_count_outcome": _metric_outcome(float(fall_delta)),
                "median_mean_absolute_speed_error_delta_m_s": mae_delta,
                "median_mean_absolute_speed_error_outcome": _metric_outcome(mae_delta),
            }
        )
    return {
        "candidate_oracle_id": candidate["oracle_id"],
        "comparisons": comparisons,
        "never_fall_requirement": "passed" if candidate_falls == 0 else "failed",
        "no_combined_ranking": True,
    }


def _report_markdown(
    *,
    cycle: int,
    arms: Sequence[Mapping[str, object]],
    baseline_arm_count: int,
    episode_rows: Sequence[Mapping[str, object]],
    behavior_names: Sequence[str],
    slow_third: ScheduleSegment,
    comparison_outcomes: Mapping[str, object] | None,
) -> bytes:
    lines = [
        f"# Composition cycle {cycle}",
        "",
        f"Evidence class: `{EVIDENCE_CLASS}`. Controller switching stands in for tracker following.",
        "",
    ]
    if baseline_arm_count:
        lines.extend(
            [
                "The four cycle-0 rows are carried forward unchanged; only the cycle-1 candidate "
                f"was evaluated in cycle {cycle}.",
                "",
                "| source | arm | episodes | median MAE (m/s) | falls | median switches | median task return |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
    else:
        lines.extend(
            [
                "| arm | episodes | median MAE (m/s) | falls | median switches | median task return |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
    for index, arm in enumerate(arms):
        source = ""
        if baseline_arm_count:
            source = (
                "cycle 0 baseline" if index < baseline_arm_count else f"cycle {cycle} candidate"
            )
        lines.append(
            "| {source}{oracle_id} | {episode_count} | {mae:.6f} | {fall_count} | "
            "{switches:.1f} | {task_return:.6f} |".format(
                source=f"{source} | " if source else "",
                oracle_id=arm["oracle_id"],
                episode_count=arm["episode_count"],
                mae=arm["median_mean_absolute_speed_error_m_s"],
                fall_count=arm["fall_count"],
                switches=arm["median_switch_count"],
                task_return=arm["median_task_return"],
            )
        )
    if cycle > 0:
        if comparison_outcomes is None:
            raise CycleEvaluationError("cycle comparison outcomes are missing")
        comparisons = comparison_outcomes["comparisons"]
        if type(comparisons) is not list:
            raise CycleEvaluationError("cycle comparison outcomes are malformed")
        candidate_arm = arms[-1]
        fall_count = int(candidate_arm["fall_count"])
        episode_count = int(candidate_arm["episode_count"])
        requirement = comparison_outcomes["never_fall_requirement"]
        lines.extend(
            [
                "",
                "## Outcome",
                "",
                f"The candidate **{requirement}** the never-fall requirement: "
                f"{fall_count}/{episode_count} episodes fell.",
                "Metric outcomes are reported separately because no combined ranking was "
                "predeclared. Negative deltas favor the candidate.",
                "",
                "| cycle-0 baseline | fall-count delta | fall outcome | median-MAE delta (m/s) | MAE outcome |",
                "|---|---:|---|---:|---|",
            ]
        )
        for comparison in comparisons:
            if type(comparison) is not dict:
                raise CycleEvaluationError("cycle comparison row is malformed")
            lines.append(
                "| {baseline} | {falls:+d} | {fall_outcome} | {mae:+.6f} | {mae_outcome} |".format(
                    baseline=comparison["baseline_oracle_id"],
                    falls=int(comparison["fall_count_delta"]),
                    fall_outcome=comparison["fall_count_outcome"],
                    mae=float(comparison["median_mean_absolute_speed_error_delta_m_s"]),
                    mae_outcome=comparison["median_mean_absolute_speed_error_outcome"],
                )
            )
        lines.extend(
            [
                "",
                "## Candidate controller switches by episode",
                "",
                "A fall is marked only when the episode's first fall boundary occurred after the "
                "switch and no more than 100 control steps later.",
                "",
                "| seed | step | from behavior | to behavior | v_x at switch (m/s) | first fall within 100 steps |",
                "|---:|---:|---|---|---:|:---:|",
            ]
        )
        for row in episode_rows:
            raw_switches = row["controller_switches"]
            if type(raw_switches) is not list:
                raise CycleEvaluationError("controller-switch diagnostics are malformed")
            if not raw_switches:
                lines.append(f"| {row['seed']} | n/a | none | none | n/a | no |")
                continue
            for switch in raw_switches:
                if type(switch) is not dict:
                    raise CycleEvaluationError("controller-switch event is malformed")
                lines.append(
                    "| {seed} | {step} | {source} | {target} | {speed:.6f} | {fall} |".format(
                        seed=row["seed"],
                        step=switch["step"],
                        source=switch["from_behavior"],
                        target=switch["to_behavior"],
                        speed=float(switch["v_x_m_s"]),
                        fall="yes" if switch["fall_followed_within_100_steps"] else "no",
                    )
                )
        lines.extend(
            [
                "",
                "## Slow-third behavior fractions by episode",
                "",
                f"Slow-third control steps: `[{slow_third.start},{slow_third.stop})`.",
                "",
                "| seed | " + " | ".join(behavior_names) + " |",
                "|---:|" + "---:|" * len(behavior_names),
            ]
        )
        for row in episode_rows:
            raw_fractions = row["slow_third_behavior_fractions"]
            if type(raw_fractions) is not dict:
                raise CycleEvaluationError("slow-third diagnostics are malformed")
            values = " | ".join(f"{float(raw_fractions[name]):.6f}" for name in behavior_names)
            lines.append(f"| {row['seed']} | {values} |")
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
                switches, slow_fractions = _episode_diagnostics(
                    execution=execution,
                    program=oracle.program,
                    task=task,
                    behavior_names=library.behavior_names,
                )
                row["controller_switches"] = switches
                row["slow_third_behavior_fractions"] = slow_fractions
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
    baseline_arms, baseline_binding = _cycle_zero_comparison(
        experiment=experiment_path,
        cycle=cycle,
        library=library,
        task=task,
    )
    current_arms: list[dict[str, object]] = []
    for oracle in loaded_oracles:
        rows = [row for row in episode_rows if row["oracle_id"] == oracle.program.oracle_id]
        current_arms.append(_summary_for_oracle(oracle, rows, library.behavior_names))
    arms = [*baseline_arms, *current_arms]
    comparison_outcomes = _comparison_outcomes(
        baseline_arms=baseline_arms,
        current_arms=current_arms,
    )
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
    if baseline_binding is not None:
        report["cycle_zero_comparison"] = baseline_binding
    if comparison_outcomes is not None:
        report["comparison_to_cycle_zero"] = comparison_outcomes
    _atomic_write(report_path, canonical_json_bytes(report))
    _atomic_write(
        markdown_path,
        _report_markdown(
            cycle=cycle,
            arms=arms,
            baseline_arm_count=len(baseline_arms),
            episode_rows=episode_rows,
            behavior_names=library.behavior_names,
            slow_third=_slow_third(task),
            comparison_outcomes=comparison_outcomes,
        ),
    )
    return report_path, markdown_path


__all__ = [
    "CLAIM_CEILING",
    "CycleEvaluationError",
    "EvaluationDependencies",
    "evaluate_cycle",
]
