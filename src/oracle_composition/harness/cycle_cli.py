"""Command line entry point for preparing and evaluating composition cycles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
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
from .inputs import LibraryManifest, TaskSpec

MAX_STEERING_CHARACTERS = 4000


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
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    repository_root: Path | None = None,
    dependencies: EvaluationDependencies | None = None,
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
