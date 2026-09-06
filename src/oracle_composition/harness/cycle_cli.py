"""Command line entry point for preparing and evaluating composition cycles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes

from .contract import ALLOWED_SIGNALS, EVIDENCE_CLASS, ORACLE_SCHEMA_ID, load_oracle_program
from .evaluator import EvaluationDependencies, evaluate_cycle
from .evidence import (
    EvidenceChainError,
    ValidatedScientificReceipt,
    current_authority_identities,
    validate_prior_report_chain,
)
from .inputs import LibraryManifest, TaskSpec, load_frozen_inputs

MAX_STEERING_CHARACTERS = 4000


@dataclass(frozen=True, slots=True)
class TrainingCliDependencies:
    """Explicit test seam; production defaults remain fail-closed and real."""

    runtime_kind: str = "real"
    test_only: bool = False
    allow_dirty: bool = False
    test_steps_per_environment: int = 2048
    test_batch_size: int = 512
    test_n_epochs: int = 10
    failure_mode: str | None = None
    resource_limits: object | None = None
    utility_dependencies: object | None = None


class CycleCliError(RuntimeError):
    """Raised when a cycle command would violate its frozen inputs."""


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _write_once(path: Path, encoded: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == encoded:
            return
        raise CycleCliError(f"refusing to replace an existing cycle input: {path}")
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


def _library_table(library: LibraryManifest) -> list[str]:
    lines = [
        "| behavior | median m/s | IQR m/s | admitted clips | E3 falls / 1,000 steps |",
        "|---|---:|---:|---:|---:|",
    ]
    for behavior in library.behaviors:
        lines.append(
            f"| {behavior.name} | {behavior.speed.median_m_s:.12f} | "
            f"{behavior.speed.iqr_m_s:.12f} | {behavior.speed.admitted_clip_count} | "
            f"{behavior.falls.falls_per_1000_steps:.12f} |"
        )
    return lines


def _prior_table(report: ValidatedScientificReceipt | None) -> list[str]:
    lines = [
        "| arm | episodes | median MAE m/s | falls | median switches | median task return |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if report is None:
        lines.append("| none | 0 | n/a | n/a | n/a | n/a |")
        return lines
    for arm in report.arms:
        raw_arm = arm.value
        lines.append(
            "| {oracle_id} | {episode_count} | {mae:.6f} | {falls} | {switches:.1f} | "
            "{task_return:.6f} |".format(
                oracle_id=raw_arm["oracle_id"],
                episode_count=raw_arm["episode_count"],
                mae=float(raw_arm["median_mean_absolute_speed_error_m_s"]),
                falls=raw_arm["fall_count"],
                switches=float(raw_arm["median_switch_count"]),
                task_return=float(raw_arm["median_task_return"]),
            )
        )
    return lines


def _designer_prompt(
    *,
    cycle: int,
    library: LibraryManifest,
    task: TaskSpec,
    prior_report: ValidatedScientificReceipt | None,
    steering: str,
) -> bytes:
    schedule = ", ".join(
        f"[{segment.start},{segment.stop}): {segment.target_m_s:.12f} m/s"
        for segment in task.schedule
    )
    schema_example = {
        "behaviors": list(library.behavior_names),
        "evidence_class": EVIDENCE_CLASS,
        "initial": "state_name",
        "oracle_id": f"cycle_{cycle}_candidate",
        "recovery": {
            "behavior": "expert",
            "guard": "z_root < 1.1",
            "max_duration": 64,
            "min_dwell": 8,
            "reentry_dwell": 8,
            "rejoin": "suspended_state_dwell_reset",
        },
        "schema_version": 1,
        "states": {
            "next_state": {"behavior": "expert", "min_dwell": 0},
            "state_name": {"behavior": "expert", "min_dwell": 1},
        },
        "transitions": [
            {"from": "state_name", "guard": "t >= 300", "priority": 0, "to": "next_state"}
        ],
    }
    lines = [
        f"# Oracle designer prompt: cycle {cycle}",
        "",
        f"Evidence label: `{EVIDENCE_CLASS}`",
        f"Task spec SHA-256: `{task.raw_sha256}`",
        f"Library manifest SHA-256: `{library.raw_sha256}`",
        "",
        "## Task",
        "",
        task.task_text,
        "",
        f"Schedule: {schedule}.",
        f"Horizon: {task.horizon_steps} steps at {task.control_period_seconds} seconds per step.",
        "The untouched stock reward is descriptive task_return only and must not be changed.",
        "",
        "## Frozen behavior library",
        "",
        *_library_table(library),
        "",
        "Speed uses root-x sidecar boundary differences over 0.015 seconds on the 28 E3-admitted blocks. "
        "Fall rates use all 36 predeclared E3 branches and are not performance guarantees.",
        "",
        "## Prior cycle report",
        "",
        *_prior_table(prior_report),
        "",
        "## Oracle JSON contract",
        "",
        f"Schema: `{ORACLE_SCHEMA_ID}`.",
        f"Allowed signals only: `{', '.join(sorted(ALLOWED_SIGNALS))}`.",
        "Guards allow numeric constants, comparisons, `and`, `or`, and `not`; calls, attributes, "
        "arithmetic, and unknown names are invalid.",
        "Transitions are checked in ascending priority after the current state's min_dwell is met. "
        "At most one controller switch occurs per step.",
        "Recovery is optional and highest priority. It must declare positive min_dwell and "
        "reentry_dwell, a bounded max_duration, and rejoin=suspended_state_dwell_reset. A "
        "still-active guard at max_duration fails closed.",
        "",
        "```json",
        json.dumps(schema_example, indent=2, sort_keys=True),
        "```",
        "",
        "## Rules",
        "",
        "- Return one JSON object only; never return or request Python.",
        "- Use only behavior names in the frozen library and only observable allowed signals.",
        "- Do not change the task, schedule, runtime, actors, reward, seeds, horizon, or metrics.",
        "- Every state needs a non-negative integer min_dwell; every transition needs a unique "
        "priority among transitions from its source.",
        "- Avoid unreachable states and zero-dwell cycles.",
        "- This executor switches controllers as a stand-in for tracker following. Do not claim "
        "tracker or oracle quality from this cycle.",
        "",
        "## Human steering",
        "",
        steering if steering else "No additional steering text was supplied.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _cycle_zero_declarations(
    experiment: Path, task: TaskSpec, library: LibraryManifest
) -> list[dict[str, object]]:
    arm_directory = experiment / "arms"
    by_id: dict[str, tuple[Path, object, str]] = {}
    for path in sorted(arm_directory.glob("*.json")):
        program, file_sha256 = load_oracle_program(path, available_behaviors=library.behavior_names)
        if program.oracle_id in by_id:
            raise CycleCliError("cycle-zero oracle ids must be unique")
        by_id[program.oracle_id] = (path, program, file_sha256)
    if set(by_id) != set(task.cycle_zero_oracle_ids):
        raise CycleCliError("cycle-zero arm files differ from the four frozen oracle ids")
    result: list[dict[str, object]] = []
    for oracle_id in task.cycle_zero_oracle_ids:
        path, raw_program, file_sha256 = by_id[oracle_id]
        program = raw_program
        result.append(
            {
                "file_sha256": file_sha256,
                "oracle_id": oracle_id,
                "oracle_sha256": program.sha256,
                "path": path.relative_to(experiment).as_posix(),
            }
        )
    return result


def prepare_cycle(
    *,
    experiment: Path,
    cycle: int,
    repository_root: Path,
    steering: str = "",
    dependencies: EvaluationDependencies | None = None,
) -> tuple[Path, Path]:
    if type(cycle) is not int or cycle < 0:
        raise CycleCliError("cycle must be a non-negative integer")
    if type(steering) is not str or len(steering) > MAX_STEERING_CHARACTERS:
        raise CycleCliError("steering text exceeds the bounded contract")
    experiment_path = Path(experiment).resolve(strict=True)
    root = Path(repository_root).resolve(strict=True)
    selected_dependencies = dependencies or EvaluationDependencies()
    try:
        sealed = selected_dependencies.execution_loader(experiment_path, root)
    except (EvidenceChainError, OSError, ValueError) as exc:
        raise CycleCliError(str(exc)) from exc
    library, task = sealed.library, sealed.task
    identities = current_authority_identities()
    metric_core_sha256 = identities["metric_core"]["sha256"]
    prior_report: ValidatedScientificReceipt | None = None
    prior_sha256: str | None = None
    if cycle > 0:
        try:
            prior_report = validate_prior_report_chain(
                experiment=experiment_path,
                repository_root=root,
                library=library,
                task=task,
                prior_cycle=cycle - 1,
                expected_metric_core_sha256=metric_core_sha256,
            )
        except EvidenceChainError as exc:
            raise CycleCliError(str(exc)) from exc
        prior_sha256 = prior_report.sha256
    prompt = _designer_prompt(
        cycle=cycle,
        library=library,
        task=task,
        prior_report=prior_report,
        steering=steering,
    )
    output = experiment_path / "cycles" / f"cycle_{cycle}"
    prompt_path = output / "designer_prompt.md"
    declarations = _cycle_zero_declarations(experiment_path, task, library) if cycle == 0 else []
    expected = {
        "cycle": cycle,
        "designer_prompt_sha256": _sha256(prompt),
        "evidence_class": EVIDENCE_CLASS,
        "execution_manifest_sha256": sealed.manifest_sha256,
        "library_manifest_sha256": library.raw_sha256,
        "metric_core_sha256": metric_core_sha256,
        "oracle_policy": "exact_predeclared_set" if cycle == 0 else "one_designer_oracle",
        "predeclared_oracles": declarations,
        "prior_report_sha256": prior_sha256,
        "prior_scientific_receipt_sha256": prior_sha256,
        "schema_version": 2,
        "task_spec_sha256": task.raw_sha256,
    }
    expected_path = output / "expected_inputs.json"
    _write_once(prompt_path, prompt)
    _write_once(expected_path, canonical_json_bytes(expected))
    return prompt_path, expected_path


def _expand_oracle_paths(values: Sequence[str]) -> list[Path]:
    result: list[Path] = []
    for value in values:
        path = Path(value)
        if path.is_dir():
            result.extend(sorted(path.glob("*.json")))
        else:
            result.append(path)
    return result


def _seed_list(value: str) -> tuple[int, ...]:
    try:
        seeds = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be a comma-separated integer list") from exc
    if not seeds or any(seed <= 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique positive integers")
    return seeds


def _canonical_mapping(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise CycleCliError(f"input must be a regular non-linked file: {path}")
    encoded = path.read_bytes()
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise CycleCliError(f"input is not valid JSON: {path}") from exc
    if type(value) is not dict or canonical_json_bytes(value) != encoded:
        raise CycleCliError(f"input is not one canonical JSON object: {path}")
    return value


def _resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _trace_record(path: Path, *, output: Path, role: str) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise CycleCliError(f"trace artifact is unavailable: {path}")
    encoded = path.read_bytes()
    return {
        "byte_count": len(encoded),
        "path": path.relative_to(output).as_posix(),
        "role": role,
        "sha256": _sha256(encoded),
    }


def _train_command(
    args: argparse.Namespace,
    *,
    root: Path,
    selected: TrainingCliDependencies,
) -> int:
    from oracle_composition.experiments.artifact_io import publish_bytes_without_overwrite
    from oracle_composition.phase_b.report_v2 import (
        build_scientific_receipt,
        publish_report_v2,
    )
    from oracle_composition.phase_b.supervision import (
        ResourceLimits,
        SupervisorStatus,
        supervise_training_job,
        validate_training_preflight,
    )

    experiment = _resolve(root, args.experiment)
    oracle = _resolve(root, args.oracle)
    reward = _resolve(root, args.reward)
    output = _resolve(root, args.output)
    if type(args.cycle) is not int or args.cycle < 0:
        raise CycleCliError("cycle must be a non-negative integer")
    preflight = validate_training_preflight(
        repository_root=root,
        experiment=experiment,
        oracle_path=oracle,
        reward_path=reward,
        allow_dirty=selected.allow_dirty or selected.test_only,
    )
    reservation = (
        _canonical_mapping(_resolve(root, args.reservation))
        if args.reservation is not None
        else None
    )
    limits = selected.resource_limits
    if limits is None:
        limits = ResourceLimits()
    if not isinstance(limits, ResourceLimits):
        raise CycleCliError("training resource-limit dependency differs")
    result = supervise_training_job(
        preflight=preflight,
        output_directory=output,
        seeds=args.seeds,
        transitions=args.transitions,
        smoke=args.smoke,
        reservation=reservation,
        limits=limits,
        runtime_kind=selected.runtime_kind,
        test_only=selected.test_only,
        test_steps_per_environment=selected.test_steps_per_environment,
        test_batch_size=selected.test_batch_size,
        test_n_epochs=selected.test_n_epochs,
        failure_mode=selected.failure_mode,
    )
    if result.status != SupervisorStatus.SUCCEEDED:
        print(
            json.dumps(
                {
                    "job_result": str(result.job_result.path),
                    "status": result.status.value,
                },
                sort_keys=True,
            )
        )
        return 2
    trace_entries = [
        _trace_record(
            result.execution_manifest.path,
            output=output,
            role="phase_b_execution_manifest",
        ),
        _trace_record(result.job_result.path, output=output, role="training_job_result"),
    ]
    if result.checkpoint_index is not None:
        trace_entries.append(
            _trace_record(
                result.checkpoint_index.path,
                output=output,
                role="five_checkpoint_index",
            )
        )
    for outcome in result.outcomes:
        seed_directory = output / f"seed_{outcome.seed}"
        trace_entries.append(
            _trace_record(outcome.receipt.path, output=output, role="seed_success_receipt")
        )
        persistence = outcome.persistence
        if persistence is None:
            raise CycleCliError("successful training result omitted persistence")
        trace_entries.extend(
            (
                _trace_record(
                    seed_directory / "training_facts_v1.json",
                    output=output,
                    role="seed_training_facts",
                ),
                _trace_record(
                    seed_directory / "rsi_ledger_v1.json",
                    output=output,
                    role="seed_rsi_ledger",
                ),
                _trace_record(
                    persistence.receipt.path,
                    output=output,
                    role="seed_persistence_receipt",
                ),
                _trace_record(
                    persistence.checkpoint.path,
                    output=output,
                    role="final_full_checkpoint",
                ),
                _trace_record(
                    persistence.strict_export.path,
                    output=output,
                    role="final_strict_actor_export",
                ),
            )
        )
    trace_entries.sort(key=lambda item: (str(item["path"]), str(item["role"])))
    trace_index_value = {
        "entries": trace_entries,
        "entry_count": len(trace_entries),
        "schema_version": 2,
        "trace_index_schema_id": "humanoid_phase_b_trace_index/v2",
    }
    trace_index = publish_bytes_without_overwrite(
        output / "trace_index_v2.json",
        canonical_json_bytes(trace_index_value),
    )
    report_inputs = dict(preflight.report_inputs)
    report_inputs["execution_manifest_sha256"] = result.execution_manifest.sha256
    seed_facts = [
        outcome.report_facts for outcome in result.outcomes if outcome.report_facts is not None
    ]
    reward_totals: dict[str, object] = {
        field: sum(float(seed.training["reward_totals"][field]) for seed in seed_facts)
        for field in ("ignored_stock_reward", "r_task", "r_track", "r_train")
    }
    reward_totals["parameters"] = dict(_canonical_mapping(reward).get("parameters", {}))
    report = build_scientific_receipt(
        cycle=args.cycle,
        inputs=report_inputs,
        seeds=seed_facts,
        episodes=[],
        step_zero_episodes=[],
        calibration=None,
        reference_records=[],
        reward_totals=reward_totals,
        trace_index_sha256=trace_index.sha256,
        prior_scientific_receipt={
            "cycle": 2,
            "path": "cycles/cycle_2/scientific_receipt_v2.json",
            "sha256": preflight.prior_scientific_receipt_sha256,
        },
    )
    telemetry_rows = [
        _canonical_mapping(output / f"seed_{outcome.seed}/telemetry_v1.json")
        for outcome in result.outcomes
    ]
    reports = publish_report_v2(
        output_directory=output,
        report=report,
        telemetry={"per_seed": telemetry_rows},
    )
    print(
        json.dumps(
            {
                "checkpoint_index": (
                    str(result.checkpoint_index.path) if result.checkpoint_index else None
                ),
                "job_result": str(result.job_result.path),
                "promotable": not args.smoke and not selected.test_only,
                "scientific_receipt": str(reports.scientific_receipt.path),
                "status": result.status.value,
                "telemetry": str(reports.telemetry.path),
            },
            sort_keys=True,
        )
    )
    return 0


def _checkpoint_hash(checkpoint: Path, explicit: str | None) -> str:
    if explicit is not None:
        if len(explicit) != 64 or any(
            character not in "0123456789abcdef" for character in explicit
        ):
            raise CycleCliError("checkpoint SHA-256 is invalid")
        return explicit
    receipt_path = checkpoint.parent / "persistence_seed_{}_v1.json".format(
        checkpoint.name.removeprefix("checkpoint_seed_").removesuffix("_final.npz")
    )
    receipt = _canonical_mapping(receipt_path)
    binding = receipt.get("checkpoint")
    if type(binding) is not dict or binding.get("filename") != checkpoint.name:
        raise CycleCliError("checkpoint requires its persistence receipt or an explicit hash")
    return str(binding["sha256"])


def _evaluate_policy_command(
    args: argparse.Namespace,
    *,
    root: Path,
    selected: TrainingCliDependencies,
) -> int:
    from oracle_composition.experiments.artifact_io import publish_bytes_without_overwrite
    from oracle_composition.phase_b.evaluation import (
        UtilityEvaluationDependencies,
        evaluate_policy_checkpoint,
    )
    from oracle_composition.phase_b.report_v2 import (
        SeedReportFacts,
        build_scientific_receipt,
        publish_report_v2,
    )
    from oracle_composition.phase_b.supervision import validate_training_preflight

    experiment = _resolve(root, args.experiment)
    oracle = _resolve(root, args.oracle)
    reward = _resolve(root, args.reward)
    checkpoint = _resolve(root, args.checkpoint)
    output = _resolve(root, args.output)
    preflight = validate_training_preflight(
        repository_root=root,
        experiment=experiment,
        oracle_path=oracle,
        reward_path=reward,
        allow_dirty=selected.allow_dirty or selected.test_only,
    )
    expected_checkpoint_sha = _checkpoint_hash(checkpoint, args.checkpoint_sha256)
    if output.exists() or output.is_symlink():
        raise CycleCliError("evaluate-policy output must be fresh and no-overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    dependencies = selected.utility_dependencies
    if dependencies is not None and not isinstance(dependencies, UtilityEvaluationDependencies):
        raise CycleCliError("utility evaluation dependency differs")
    evaluation = evaluate_policy_checkpoint(
        checkpoint_path=checkpoint,
        checkpoint_sha256=expected_checkpoint_sha,
        step_zero_actor_path=preflight.runtime_config.starting_actor_path,
        step_zero_actor_sha256=preflight.runtime_config.starting_actor_sha256,
        corpus_root=root / "artifacts/reference_corpus_v2",
        calibration_receipt_path=_resolve(root, args.calibration_receipt),
        calibration_receipt_sha256=args.calibration_receipt_sha256,
        segment_targets_m_s=tuple(
            float(segment.target_m_s) for segment in load_frozen_inputs(experiment)[1].schedule
        ),
        dependencies=dependencies,
    )
    trained_value = [episode.to_dict() for episode in evaluation.trained_episodes]
    baseline_value = [episode.to_dict() for episode in evaluation.step_zero_episodes]
    trained = publish_bytes_without_overwrite(
        output / "trained_policy_metrics_v1.json", canonical_json_bytes(trained_value)
    )
    baseline = publish_bytes_without_overwrite(
        output / "step_zero_metrics_v1.json", canonical_json_bytes(baseline_value)
    )
    trace_index_value = {
        "entries": [
            {
                "byte_count": trained.byte_count,
                "path": trained.path.name,
                "role": "trained_policy_metrics",
                "sha256": trained.sha256,
            },
            {
                "byte_count": baseline.byte_count,
                "path": baseline.path.name,
                "role": "step_zero_metrics",
                "sha256": baseline.sha256,
            },
        ],
        "entry_count": 2,
        "schema_version": 2,
        "trace_index_schema_id": "humanoid_phase_b_trace_index/v2",
    }
    trace_index = publish_bytes_without_overwrite(
        output / "trace_index_v2.json", canonical_json_bytes(trace_index_value)
    )
    metadata = evaluation.checkpoint.metadata
    training_facts_path = checkpoint.parent / "training_facts_v1.json"
    if not training_facts_path.is_file() or training_facts_path.is_symlink():
        raise CycleCliError("checkpoint training facts are unavailable")
    training = _canonical_mapping(training_facts_path)
    if (
        training.get("execution_manifest_sha256") != metadata["execution_manifest_sha256"]
        or training.get("ppo_seed") != metadata["ppo_seed"]
        or _sha256(training_facts_path.read_bytes()) != metadata["training_facts_sha256"]
    ):
        raise CycleCliError("checkpoint and training facts differ")
    persistence_receipt = _canonical_mapping(
        checkpoint.parent / "persistence_seed_{}_v1.json".format(metadata["ppo_seed"])
    )
    if persistence_receipt.get("training_facts_sha256") != metadata["training_facts_sha256"]:
        raise CycleCliError("persistence and training facts differ")
    worker_step_zero = training.get("step_zero_comparator")
    if type(worker_step_zero) is not dict or worker_step_zero.get("bitwise_equal") is not True:
        raise CycleCliError("stored checkpoint omitted its worker step-0 comparator")
    step_zero_comparator = {
        **worker_step_zero,
        **dict(evaluation.step_zero_comparator),
    }
    seed = SeedReportFacts(
        ppo_seed=int(metadata["ppo_seed"]),
        checkpoint_sha256=expected_checkpoint_sha,
        strict_export_sha256=(persistence_receipt["strict_export"]["sha256"]),
        training=training,
        step_zero_comparator=step_zero_comparator,
    )
    report_inputs = dict(preflight.report_inputs)
    report_inputs["execution_manifest_sha256"] = metadata["execution_manifest_sha256"]
    stored_reward_totals = training["reward_totals"]
    reward_totals: dict[str, object] = {
        field: float(stored_reward_totals[field])
        for field in ("ignored_stock_reward", "r_task", "r_track", "r_train")
    }
    reward_totals["parameters"] = dict(_canonical_mapping(reward).get("parameters", {}))
    report = build_scientific_receipt(
        cycle=args.cycle,
        inputs=report_inputs,
        seeds=[seed],
        episodes=evaluation.trained_episodes,
        step_zero_episodes=evaluation.step_zero_episodes,
        calibration=evaluation.calibration,
        reference_records=[],
        reward_totals=reward_totals,
        trace_index_sha256=trace_index.sha256,
        prior_scientific_receipt={
            "cycle": 2,
            "path": "cycles/cycle_2/scientific_receipt_v2.json",
            "sha256": preflight.prior_scientific_receipt_sha256,
        },
    )
    reports = publish_report_v2(output_directory=output, report=report, telemetry={})
    print(
        json.dumps(
            {
                "scientific_receipt": str(reports.scientific_receipt.path),
                "step_zero_metrics": str(baseline.path),
                "trained_metrics": str(trained.path),
            },
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m oracle_composition.harness.cycle_cli")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--experiment", type=Path, required=True)
    prepare.add_argument("--cycle", type=int, required=True)
    prepare.add_argument("--steer", default="")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--experiment", type=Path, required=True)
    evaluate.add_argument("--cycle", type=int, required=True)
    evaluate.add_argument("--oracle", action="append", required=True)
    train = commands.add_parser("train")
    train.add_argument("--experiment", type=Path, required=True)
    train.add_argument("--cycle", type=int, required=True)
    train.add_argument("--oracle", type=Path, required=True)
    train.add_argument("--reward", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--seeds", type=_seed_list, required=True)
    train.add_argument("--transitions", type=int, required=True)
    train.add_argument("--reservation", type=Path)
    train.add_argument("--smoke", action="store_true")
    train.add_argument("--promote", action="store_true")
    policy = commands.add_parser("evaluate-policy")
    policy.add_argument("--experiment", type=Path, required=True)
    policy.add_argument("--cycle", type=int, required=True)
    policy.add_argument("--oracle", type=Path, required=True)
    policy.add_argument("--reward", type=Path, required=True)
    policy.add_argument("--checkpoint", type=Path, required=True)
    policy.add_argument("--checkpoint-sha256")
    policy.add_argument("--calibration-receipt", type=Path, required=True)
    policy.add_argument("--calibration-receipt-sha256", required=True)
    policy.add_argument("--output", type=Path, required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    repository_root: Path | None = None,
    dependencies: EvaluationDependencies | None = None,
    training_dependencies: TrainingCliDependencies | None = None,
) -> int:
    args = _parser().parse_args(argv)
    root = (
        Path(repository_root).resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    experiment = args.experiment
    if not experiment.is_absolute():
        experiment = root / experiment
    if args.command == "prepare":
        prompt, expected = prepare_cycle(
            experiment=experiment,
            cycle=args.cycle,
            repository_root=root,
            steering=args.steer,
            dependencies=dependencies,
        )
        print(json.dumps({"designer_prompt": str(prompt), "expected_inputs": str(expected)}))
        return 0
    selected_training = training_dependencies or TrainingCliDependencies()
    if args.command == "train":
        if args.promote and args.smoke:
            raise CycleCliError("smoke mode is interface_check and refuses promotion")
        if args.promote and selected_training.test_only:
            raise CycleCliError("controlled fake-runtime output refuses promotion")
        return _train_command(args, root=root, selected=selected_training)
    if args.command == "evaluate-policy":
        return _evaluate_policy_command(args, root=root, selected=selected_training)
    oracle_paths = _expand_oracle_paths(args.oracle)
    report, markdown = evaluate_cycle(
        experiment=experiment,
        cycle=args.cycle,
        oracle_paths=oracle_paths,
        repository_root=root,
        dependencies=dependencies,
    )
    print(json.dumps({"report_json": str(report), "report_markdown": str(markdown)}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
