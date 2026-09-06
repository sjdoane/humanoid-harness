#!/usr/bin/env python3
"""Request or supervise one fixed GMT parity or course development run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import run_gmt_probe as supervisor

from oracle_composition.adapters.gmt.checkpoint import verify_upstream_root
from oracle_composition.adapters.gmt.contracts import GMT_UPSTREAM_COMMIT, MOTION_SPECS
from oracle_composition.adapters.gmt.io import GMTAdmissionError, sha256_file
from oracle_composition.adapters.gmt.reference_sensitivity import validate_probe_identities
from oracle_composition.adapters.gmt.trace_admission import load_validated_replay
from oracle_composition.adapters.gmt.training_contract import (
    CourseTrainerSpec,
    effective_training_contract,
    training_reward_metadata,
)
from oracle_composition.adapters.gmt.training_telemetry import (
    MAX_TELEMETRY_BYTES,
    SCALED_TELEMETRY_FILENAME,
    TELEMETRY_FILENAME,
    validate_training_telemetry_descriptor,
)
from oracle_composition.harness.resource_slot import (
    ResourceSlotError,
    canonical_json_bytes,
    load_validated_reservation,
)

MAX_CONFIG_BYTES = 256 * 1024
MAX_MANIFEST_BYTES = 1 * 1024**2
MAX_LOG_BYTES = 4 * 1024**2
PARITY_FILENAME = "runtime_parity.json"
COURSE_MANIFEST_FILENAME = "course_run_manifest.json"
COURSE_MANIFEST_FIELDS = {
    "schema_version",
    "artifact",
    "status",
    "input_config_sha256",
    "outputs",
    "identities",
    "frozen_runtime",
    "training",
    "zero_residual",
    "final_policy",
    "runtime",
    "claims",
}
COURSE_PROBE_OUTPUTS = {
    "input_config.json",
    "zero_residual_frames.jsonl",
    "zero_residual_trajectory.npz",
    "zero_residual_evaluation.json",
}
COURSE_TRAIN_OUTPUTS = {
    *COURSE_PROBE_OUTPUTS,
    "initial_residual_policy.npz",
    "final_residual_policy.npz",
    "final_policy_frames.jsonl",
    "final_policy_trajectory.npz",
    "final_policy_evaluation.json",
}
COURSE_TRAIN_TELEMETRY_OUTPUTS = {*COURSE_TRAIN_OUTPUTS, TELEMETRY_FILENAME}
COURSE_TRAIN_SCALED_TELEMETRY_OUTPUTS = {
    *COURSE_TRAIN_OUTPUTS,
    SCALED_TELEMETRY_FILENAME,
}
LAUNCHER_SOURCES = ("scripts/run_gmt_probe.py", "scripts/run_gmt_development.py")
PARITY_CONFIG_FIELDS = {
    "manifest_path",
    "manifest_sha256",
    "motion_name",
    "motion_path",
    "motion_sha256",
    "trace_path",
    "trace_sha256",
    "upstream_root",
    "weights_path",
    "weights_sha256",
}
PARITY_COMPARISONS = {
    "applied_torque",
    "base_raw_action",
    "boundary_angular_velocity",
    "boundary_orientation_wxyz",
    "boundary_qpos",
    "boundary_qvel",
    "clipped_action",
    "interval_composite_raw_action",
    "pd_target",
    "post_substep_qpos",
    "post_substep_qvel",
    "prepared_obs",
    "prepared_proprio",
    "reference_window",
    "zero_residual_composite",
}


@dataclass(frozen=True)
class DevelopmentRequest:
    workload: str
    repository_root: Path
    venv_python: Path
    config_path: Path
    output_directory: Path
    owner: str


@dataclass(frozen=True)
class _LoadedWorkload:
    raw: dict[str, object]
    sha256: str
    assets: dict[str, object]
    primary_motion_name: str
    primary_motion_path: Path
    primary_motion_sha256: str
    upstream_root: Path
    weights_path: Path
    weights_sha256: str
    course_mode: str | None = None
    course_identities: dict[str, object] | None = None
    course_trainer: CourseTrainerSpec | None = None
    parity_receipt_inputs: dict[str, object] | None = None


@dataclass(frozen=True)
class _PlanMaterial:
    root: Path
    commit: str
    source_files: dict[str, str]
    venv: dict[str, object]
    loaded: _LoadedWorkload
    inputs: dict[str, object]
    canonical_argv: tuple[str, ...]
    limits: supervisor.ProbeLimits
    artifact_label: str
    evidence_class: str


def _sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise supervisor.ProbeError(f"{field} must be a lowercase SHA-256")
    return value


def _absolute_json_path(value: object, *, field: str, directory: bool = False) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        raise supervisor.ProbeError(f"{field} path is malformed")
    path = Path(value)
    if not path.is_absolute():
        raise supervisor.ProbeError(f"{field} path must be absolute")
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        metadata = absolute.lstat()
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise supervisor.ProbeError(f"{field} is unavailable") from exc
    expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode) or not expected or resolved != absolute:
        kind = "directory" if directory else "file"
        raise supervisor.ProbeError(f"{field} must be a direct absolute {kind}")
    return absolute


def _load_json_config(path: Path) -> tuple[Path, bytes, dict[str, object]]:
    config_path = supervisor._regular_file(path, name="development config")
    payload = supervisor._read_bounded(config_path, MAX_CONFIG_BYTES, "development config")
    return config_path, payload, supervisor._json_object(payload, "development config")


def _file_binding(path: Path, expected_sha256: str, *, field: str) -> dict[str, object]:
    direct = _absolute_json_path(str(path), field=field)
    expected = _sha256(expected_sha256, field=f"{field} SHA-256")
    observed = sha256_file(direct)
    if observed != expected:
        raise supervisor.ProbeError(f"{field} bytes differ from the config")
    return {"path": str(direct), "sha256": observed, "size": direct.stat().st_size}


def _repository_sources(root: Path) -> tuple[Path, str, dict[str, str]]:
    root = supervisor._absolute(root)
    if Path(supervisor._run_git(root, "rev-parse", "--show-toplevel")) != root:
        raise supervisor.ProbeError("repository root differs from Git's top level")
    commit = supervisor._run_git(root, "rev-parse", "HEAD")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise supervisor.ProbeError("Git HEAD is not a full lowercase commit")
    if supervisor._run_git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise supervisor.ProbeError("development launch requires a clean source worktree")
    tracked = supervisor._run_git(
        root,
        "ls-files",
        "-z",
        "--",
        "src/oracle_composition",
        *LAUNCHER_SOURCES,
        "uv.lock",
    ).split("\x00")
    selected = sorted(
        path
        for path in tracked
        if path == "uv.lock"
        or path in LAUNCHER_SOURCES
        or (path.startswith("src/oracle_composition/") and path.endswith(".py"))
    )
    required = {*LAUNCHER_SOURCES, "uv.lock"}
    if not required.issubset(selected):
        raise supervisor.ProbeError("tracked development launcher source set is incomplete")
    identities = {
        relative: sha256_file(supervisor._regular_file(root / relative, name=relative))
        for relative in selected
    }
    return root, commit, identities


def _source_tree_binding(source_files: Mapping[str, str]) -> dict[str, object]:
    """Compactly bind the complete path-to-digest ledger without omitting names."""

    return {
        "file_count": len(source_files),
        "canonical_tree_sha256": hashlib.sha256(
            canonical_json_bytes(dict(source_files))
        ).hexdigest(),
    }


def _load_parity(path: Path) -> _LoadedWorkload:
    _, payload, raw = _load_json_config(path)
    if set(raw) != PARITY_CONFIG_FIELDS:
        raise supervisor.ProbeError("parity config fields differ")
    upstream = _absolute_json_path(raw["upstream_root"], field="upstream root", directory=True)
    weights_path = _absolute_json_path(raw["weights_path"], field="actor weights")
    weights_sha256 = _sha256(raw["weights_sha256"], field="actor weights SHA-256")
    if weights_sha256 != supervisor.OFFICIAL_ACTOR_SHA256:
        raise supervisor.ProbeError("parity actor is not the admitted GMT baseline actor")
    motion_name = raw["motion_name"]
    if type(motion_name) is not str or motion_name not in MOTION_SPECS:
        raise supervisor.ProbeError("parity motion is outside the admitted GMT catalog")
    motion_path = _absolute_json_path(raw["motion_path"], field="motion")
    motion_sha256 = _sha256(raw["motion_sha256"], field="motion SHA-256")
    trace_path = _absolute_json_path(raw["trace_path"], field="retained trace")
    trace_sha256 = _sha256(raw["trace_sha256"], field="retained trace SHA-256")
    manifest_path = _absolute_json_path(raw["manifest_path"], field="replay manifest")
    manifest_sha256 = _sha256(raw["manifest_sha256"], field="replay manifest SHA-256")
    manifest, abi, _, support_files = load_validated_replay(
        trace_path=trace_path,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        upstream_root=upstream,
    )
    validate_probe_identities(
        manifest,
        trace_sha256=trace_sha256,
        weights_sha256=weights_sha256,
        motion_name=motion_name,
        motion_sha256=motion_sha256,
    )
    weights = _file_binding(weights_path, weights_sha256, field="actor weights")
    motion = _file_binding(motion_path, motion_sha256, field="motion")
    trace = _file_binding(trace_path, trace_sha256, field="retained trace")
    replay_manifest = _file_binding(
        manifest_path, manifest_sha256, field="replay manifest"
    )
    receipt_inputs = {
        "upstream_commit": GMT_UPSTREAM_COMMIT,
        "trace": {"path": trace_path.name, "sha256": trace_sha256},
        "replay_manifest": {"path": manifest_path.name, "sha256": manifest_sha256},
        "actor": {"path": weights_path.name, "sha256": weights_sha256},
        "motion": {
            "path": motion_path.name,
            "name": motion_name,
            "sha256": motion_sha256,
        },
        "support_files": support_files,
        "model_abi": json.loads(json.dumps(asdict(abi))),
    }
    return _LoadedWorkload(
        raw=raw,
        sha256=hashlib.sha256(payload).hexdigest(),
        assets={
            "upstream_root": {
                "path": str(upstream),
                "upstream_commit": GMT_UPSTREAM_COMMIT,
                "support_files": support_files,
            },
            "weights": weights,
            "motions": {motion_name: motion},
            "retained_trace": trace,
            "replay_manifest": replay_manifest,
        },
        primary_motion_name=motion_name,
        primary_motion_path=motion_path,
        primary_motion_sha256=motion_sha256,
        upstream_root=upstream,
        weights_path=weights_path,
        weights_sha256=weights_sha256,
        parity_receipt_inputs=receipt_inputs,
    )


def _load_course(path: Path) -> _LoadedWorkload:
    # Keep request and terminal validation data-only. Importing course_run here
    # would initialize the trainer stack before the supervised child exists.
    from oracle_composition.adapters.gmt.course_config import load_run_config

    config_path, payload, raw = _load_json_config(path)
    config = load_run_config(config_path)
    digest = hashlib.sha256(payload).hexdigest()
    if config.sha256 != digest or config.raw != raw:
        raise supervisor.ProbeError("course loader identity differs from exact config bytes")
    assets = raw["assets"]
    assert isinstance(assets, dict)
    upstream = _absolute_json_path(assets["upstream_root"], field="upstream root", directory=True)
    weights_raw = assets["weights"]
    motions_raw = assets["motions"]
    assert isinstance(weights_raw, dict) and isinstance(motions_raw, dict)
    weights_path = _absolute_json_path(weights_raw["path"], field="actor weights")
    weights_sha256 = _sha256(weights_raw["sha256"], field="actor weights SHA-256")
    if weights_sha256 != supervisor.OFFICIAL_ACTOR_SHA256:
        raise supervisor.ProbeError("course actor is not the admitted GMT baseline actor")
    weights = _file_binding(weights_path, weights_sha256, field="actor weights")
    motions: dict[str, object] = {}
    motion_paths: dict[str, Path] = {}
    for name in sorted(motions_raw):
        binding = motions_raw[name]
        assert isinstance(name, str) and isinstance(binding, dict)
        if name not in MOTION_SPECS:
            raise supervisor.ProbeError("course motion is outside the admitted GMT catalog")
        motion_path = _absolute_json_path(binding["path"], field=f"motion {name}")
        motion_sha256 = _sha256(binding["sha256"], field=f"motion {name} SHA-256")
        motions[name] = _file_binding(motion_path, motion_sha256, field=f"motion {name}")
        motion_paths[name] = motion_path
    if not motions:
        raise supervisor.ProbeError("course config has no admitted motions")
    primary_name = sorted(motions)[0]
    primary = motions[primary_name]
    assert isinstance(primary, dict)
    mode = raw["mode"]
    assert isinstance(mode, str)
    return _LoadedWorkload(
        raw=raw,
        sha256=digest,
        assets={
            "upstream_root": {
                "path": str(upstream),
                "upstream_commit": GMT_UPSTREAM_COMMIT,
                "support_files": verify_upstream_root(upstream),
            },
            "weights": weights,
            "motions": motions,
        },
        primary_motion_name=primary_name,
        primary_motion_path=motion_paths[primary_name],
        primary_motion_sha256=str(primary["sha256"]),
        upstream_root=upstream,
        weights_path=weights_path,
        weights_sha256=weights_sha256,
        course_mode=mode,
        course_trainer=config.trainer,
        course_identities={
            "task": config.task.sha256,
            "oracle": config.program.sha256,
            "reward": config.recipe.sha256,
            "segments": {
                name: segment.sha256 for name, segment in config.segments.items()
            },
        },
    )


def _load_workload(workload: str, path: Path) -> _LoadedWorkload:
    if workload == "parity":
        return _load_parity(path)
    if workload == "course":
        return _load_course(path)
    raise supervisor.ProbeError("development workload must be parity or course")


def _profile(workload: str, course_mode: str | None) -> tuple[supervisor.ProbeLimits, str, str]:
    common = {
        "rss_bytes": 8 * 1024**3,
        "free_disk_bytes": 20 * 1024**3,
        "output_bytes": 1 * 1024**3,
    }
    if workload == "parity":
        return (
            supervisor.ProbeLimits(wall_seconds=120, cpu_seconds=120, **common),
            "gmt_g1_runtime_parity_development_resource_receipt",
            "development_runtime_parity_only_not_task_success",
        )
    if course_mode not in {"probe", "train"}:
        raise supervisor.ProbeError("course config mode must be probe or train")
    return (
        supervisor.ProbeLimits(wall_seconds=1_200, cpu_seconds=1_200, **common),
        "gmt_g1_course_development_resource_receipt",
        f"development_course_{course_mode}_not_task_success",
    )


def _child_argv(request: DevelopmentRequest, root: Path) -> tuple[str, ...]:
    if request.workload == "parity":
        return (
            str(request.venv_python),
            str(root / "scripts/run_gmt_development.py"),
            "parity-child",
            "--config",
            str(request.config_path),
            "--output",
            str(request.output_directory),
        )
    return (
        str(request.venv_python),
        "-m",
        "oracle_composition.adapters.gmt.course_run",
        "--config",
        str(request.config_path),
        "--output",
        str(request.output_directory),
    )


def _plan_material(request: DevelopmentRequest) -> _PlanMaterial:
    root, commit, source_files = _repository_sources(request.repository_root)
    venv_python = supervisor._absolute(request.venv_python)
    venv = supervisor._venv_identity(venv_python)
    config_path = supervisor._regular_file(request.config_path, name="development config")
    loaded = _load_workload(request.workload, config_path)
    limits, artifact_label, evidence_class = _profile(request.workload, loaded.course_mode)
    normalized = replace(
        request,
        repository_root=root,
        venv_python=venv_python,
        config_path=config_path,
        output_directory=supervisor._absolute(request.output_directory),
    )
    inputs: dict[str, object] = {
        "artifact_contract": "gmt_g1_fixed_development_launcher/v1",
        "workload": request.workload,
        "course_mode": loaded.course_mode,
        "config": {
            "path": str(config_path),
            "sha256": loaded.sha256,
            "size": config_path.stat().st_size,
        },
        "repository_sources": _source_tree_binding(source_files),
        "venv": venv,
        "gmt_assets": loaded.assets,
    }
    return _PlanMaterial(
        root=root,
        commit=commit,
        source_files=source_files,
        venv=venv,
        loaded=loaded,
        inputs=inputs,
        canonical_argv=_child_argv(normalized, root),
        limits=limits,
        artifact_label=artifact_label,
        evidence_class=evidence_class,
    )


def build_development_plan(request: DevelopmentRequest) -> supervisor.ProbePlan:
    material = _plan_material(request)
    output = supervisor._absolute(request.output_directory)
    if output.exists():
        raise supervisor.ProbeError("development output must not already exist")
    if output.is_relative_to(material.root):
        raise supervisor.ProbeError("development output must be outside the source worktree")
    parent = output.parent.resolve(strict=True)
    if not parent.is_dir() or parent.is_symlink():
        raise supervisor.ProbeError("development output parent must be a real existing directory")
    loaded = material.loaded
    config = supervisor.ProbeConfig(
        repository_root=material.root,
        venv_python=supervisor._absolute(request.venv_python),
        upstream_root=loaded.upstream_root,
        weights_path=loaded.weights_path,
        weights_sha256=loaded.weights_sha256,
        motion_path=loaded.primary_motion_path,
        motion_sha256=loaded.primary_motion_sha256,
        motion_name=loaded.primary_motion_name,
        output_directory=output,
        owner=request.owner,
    )
    return supervisor.ProbePlan(
        config=config,
        commit=material.commit,
        canonical_argv=material.canonical_argv,
        inputs=material.inputs,
        limits=material.limits,
        artifact_label=material.artifact_label,
        evidence_class=material.evidence_class,
    )


def _request_from_plan(plan: supervisor.ProbePlan) -> DevelopmentRequest:
    workload = plan.inputs.get("workload")
    config = plan.inputs.get("config")
    if type(workload) is not str or not isinstance(config, Mapping):
        raise supervisor.ProbeError("development plan input ledger is malformed")
    config_path = config.get("path")
    if type(config_path) is not str:
        raise supervisor.ProbeError("development plan config binding is malformed")
    return DevelopmentRequest(
        workload=workload,
        repository_root=plan.config.repository_root,
        venv_python=plan.config.venv_python,
        config_path=Path(config_path),
        output_directory=plan.config.output_directory,
        owner=plan.config.owner,
    )


def validate_development_plan(plan: supervisor.ProbePlan) -> None:
    request = _request_from_plan(plan)
    material = _plan_material(request)
    expected = {
        "commit": material.commit,
        "canonical_argv": list(material.canonical_argv),
        "inputs": material.inputs,
        "limits": material.limits.as_dict(),
        "artifact_label": material.artifact_label,
        "evidence_class": material.evidence_class,
    }
    observed = {
        "commit": plan.commit,
        "canonical_argv": list(plan.canonical_argv),
        "inputs": plan.inputs,
        "limits": plan.limits.as_dict(),
        "artifact_label": plan.artifact_label,
        "evidence_class": plan.evidence_class,
    }
    if canonical_json_bytes(observed) != canonical_json_bytes(expected):
        raise supervisor.ProbeError("development plan or bound inputs changed")


def _output_names(output: Path) -> set[str]:
    try:
        return {path.name for path in output.iterdir()}
    except OSError as exc:
        raise supervisor.ProbeError("development output directory is unavailable") from exc


def _json_file(path: Path, *, maximum: int, label: str) -> tuple[dict[str, object], str]:
    payload = supervisor._read_bounded(path, maximum, label)
    return supervisor._json_object(payload, label), hashlib.sha256(payload).hexdigest()


def _validate_parity_measurement(receipt: Mapping[str, object]) -> None:
    measurement = receipt.get("measurement")
    contract = receipt.get("comparison_contract")
    if not isinstance(measurement, Mapping) or not isinstance(contract, Mapping):
        raise supervisor.ProbeError("runtime parity measurement contract is missing")
    expected_counts = {
        "control_steps": 500,
        "simulation_steps": 10_000,
        "matched_runtime_poststate_samples": 9_999,
        "unavailable_baseline_final_poststate_samples": 1,
    }
    if any(measurement.get(key) != value for key, value in expected_counts.items()):
        raise supervisor.ProbeError("runtime parity sample counts differ")
    comparisons = measurement.get("comparisons")
    if not isinstance(comparisons, Mapping) or set(comparisons) != PARITY_COMPARISONS:
        raise supervisor.ProbeError("runtime parity comparison set differs")
    for name, raw in comparisons.items():
        if not isinstance(raw, Mapping) or set(raw) != {
            "chunks_compared",
            "exact_chunks",
            "exact_values",
            "max_abs_difference",
            "values_compared",
        }:
            raise supervisor.ProbeError(f"runtime parity comparison is malformed: {name}")
        chunks = raw["chunks_compared"]
        values = raw["values_compared"]
        if (
            type(chunks) is not int
            or chunks <= 0
            or type(values) is not int
            or values <= 0
            or raw["exact_chunks"] != chunks
            or raw["exact_values"] != values
            or type(raw["max_abs_difference"]) not in {int, float}
            or raw["max_abs_difference"] != 0
        ):
            raise supervisor.ProbeError(f"runtime parity comparison is not exact: {name}")
    if contract.get("numeric_tolerance") != 0:
        raise supervisor.ProbeError("runtime parity tolerance is not zero")


def _verify_parity(plan: supervisor.ProbePlan) -> dict[str, object]:
    output = plan.config.output_directory
    expected_names = {PARITY_FILENAME, supervisor.STDOUT_FILENAME, supervisor.STDERR_FILENAME}
    if _output_names(output) != expected_names:
        raise supervisor.ProbeError("runtime parity output file set differs")
    receipt_path = output / PARITY_FILENAME
    receipt, receipt_sha256 = _json_file(
        receipt_path, maximum=MAX_MANIFEST_BYTES, label="runtime parity receipt"
    )
    loaded = _load_workload("parity", Path(plan.inputs["config"]["path"]))
    if (
        set(receipt)
        != {
            "schema_version",
            "artifact",
            "claim_status",
            "inputs",
            "comparison_contract",
            "measurement",
            "runtime",
            "limits",
        }
        or receipt.get("schema_version") != 1
        or receipt.get("artifact") != "gmt_g1_shared_runtime_zero_residual_parity"
        or receipt.get("claim_status")
        != "shared_runtime_exact_parity_to_retained_reconstructed_baseline"
        or receipt.get("inputs") != loaded.parity_receipt_inputs
    ):
        raise supervisor.ProbeError("runtime parity receipt identity or config binding differs")
    _validate_parity_measurement(receipt)
    limits = receipt.get("limits")
    if limits != {
        "original_jit_executed": False,
        "jit_equivalence_tested": False,
        "runtime_parity_only": True,
        "task_competence_tested": False,
        "reference_composition_tested": False,
        "learning_performed": False,
    }:
        raise supervisor.ProbeError("runtime parity claim limits differ")
    stdout, _ = _json_file(
        output / supervisor.STDOUT_FILENAME,
        maximum=MAX_LOG_BYTES,
        label="parity child stdout",
    )
    supervisor._read_bounded(
        output / supervisor.STDERR_FILENAME, MAX_LOG_BYTES, "parity child stderr"
    )
    if (
        stdout.get("artifact_path") != str(receipt_path)
        or stdout.get("artifact_sha256") != receipt_sha256
        or stdout.get("claim_status") != receipt["claim_status"]
        or stdout.get("measurement") != receipt["measurement"]
    ):
        raise supervisor.ProbeError("runtime parity child receipt binding differs")
    return {
        "runtime_parity": {
            "path": PARITY_FILENAME,
            "sha256": receipt_sha256,
            "size": receipt_path.stat().st_size,
        }
    }


def _safe_output_name(value: object) -> str:
    reserved = {
        ".",
        "..",
        COURSE_MANIFEST_FILENAME,
        supervisor.STDOUT_FILENAME,
        supervisor.STDERR_FILENAME,
        supervisor.RECEIPT_FILENAME,
    }
    if (
        type(value) is not str
        or not value
        or "\x00" in value
        or Path(value).name != value
        or value in reserved
    ):
        raise supervisor.ProbeError("course output filename is unsafe or reserved")
    return value


def _validate_course_summary(
    value: object,
    *,
    label: str,
    horizon_steps: int,
    task_sha256: str,
    zero_residual: bool,
) -> None:
    expected = {
        "objective_evaluation",
        "training_reward_sum_not_success_metric",
        "reset",
        "steps",
        "residual_rms",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise supervisor.ProbeError(f"course {label} summary fields differ")
    steps = value["steps"]
    residual_rms = value["residual_rms"]
    reward_sum = value["training_reward_sum_not_success_metric"]
    objective = value["objective_evaluation"]
    if (
        type(steps) is not int
        or not 1 <= steps <= horizon_steps
        or type(residual_rms) not in {int, float}
        or not 0 <= residual_rms < float("inf")
        or type(reward_sum) not in {int, float}
        or not -float("inf") < reward_sum < float("inf")
        or (zero_residual and residual_rms != 0)
        or not isinstance(objective, Mapping)
        or objective.get("frame_count") != steps
        or objective.get("task_sha256") != task_sha256
        or not isinstance(value["reset"], Mapping)
    ):
        raise supervisor.ProbeError(f"course {label} summary is malformed")


def _validate_training_record(
    value: object,
    *,
    mode: str,
    steps: int,
    telemetry_encoded: bytes | None,
    trainer: CourseTrainerSpec | None,
) -> None:
    if mode == "probe":
        if value is not None:
            raise supervisor.ProbeError("probe course run reports training")
        return
    expected = {
        "completed_transitions",
        "episodes",
        "falls",
        "policy_artifact",
        "frozen_base_state_before_sha256",
        "frozen_base_state_after_sha256",
    }
    if telemetry_encoded is not None:
        expected.add("telemetry")
    if trainer is not None:
        expected.add("reward_preconditioning")
    if not isinstance(value, Mapping) or set(value) != expected:
        raise supervisor.ProbeError("course training record fields differ")
    if (
        value["completed_transitions"] != steps
        or type(value["episodes"]) is not int
        or value["episodes"] < 0
        or type(value["falls"]) is not int
        or not 0 <= value["falls"] <= value["episodes"]
        or value["policy_artifact"] != "numeric_weights_not_optimizer_resume"
        or _sha256(
            value["frozen_base_state_before_sha256"], field="frozen base state before"
        )
        != _sha256(value["frozen_base_state_after_sha256"], field="frozen base state after")
    ):
        raise supervisor.ProbeError("course training record differs from the fixed budget")
    if telemetry_encoded is not None:
        try:
            validate_training_telemetry_descriptor(
                value["telemetry"],
                telemetry_encoded,
                steps,
                reward_scale=(
                    trainer.total_training_reward_scale if trainer is not None else None
                ),
            )
        except ValueError as exc:
            raise supervisor.ProbeError(str(exc)) from exc
    if trainer is not None and value["reward_preconditioning"] != training_reward_metadata(
        trainer
    ):
        raise supervisor.ProbeError("course reward preconditioning metadata differs")


def _verify_course(plan: supervisor.ProbePlan) -> dict[str, object]:
    output = plan.config.output_directory
    manifest_path = output / COURSE_MANIFEST_FILENAME
    manifest, manifest_sha256 = _json_file(
        manifest_path, maximum=MAX_MANIFEST_BYTES, label="course run manifest"
    )
    config_binding = plan.inputs.get("config")
    if not isinstance(config_binding, Mapping) or type(config_binding.get("path")) is not str:
        raise supervisor.ProbeError("course plan config binding is malformed")
    loaded = _load_workload("course", Path(config_binding["path"]))
    raw = loaded.raw
    mode = loaded.course_mode
    if mode not in {"probe", "train"}:
        raise supervisor.ProbeError("course mode binding is malformed")
    if (
        set(manifest) != COURSE_MANIFEST_FIELDS
        or manifest.get("schema_version") != 1
        or manifest.get("artifact") != "gmt_g1_course_development_run"
        or manifest.get("status") != "completed"
        or manifest.get("input_config_sha256") != config_binding.get("sha256")
    ):
        raise supervisor.ProbeError("course run manifest identity or config binding differs")
    outputs = manifest.get("outputs")
    output_names = set(outputs) if isinstance(outputs, Mapping) else set()
    valid_output_sets = (
        {frozenset(COURSE_PROBE_OUTPUTS)}
        if mode == "probe"
        else {frozenset(COURSE_TRAIN_SCALED_TELEMETRY_OUTPUTS)}
        if loaded.course_trainer is not None
        else {
            frozenset(COURSE_TRAIN_OUTPUTS),
            frozenset(COURSE_TRAIN_TELEMETRY_OUTPUTS),
        }
    )
    if not isinstance(outputs, Mapping) or frozenset(output_names) not in valid_output_sets:
        raise supervisor.ProbeError("course run output hash ledger differs from its mode")
    expected_names = {
        COURSE_MANIFEST_FILENAME,
        supervisor.STDOUT_FILENAME,
        supervisor.STDERR_FILENAME,
    }
    artifacts: dict[str, object] = {}
    telemetry_encoded = None
    for raw_name, raw_sha256 in outputs.items():
        name = _safe_output_name(raw_name)
        expected_sha256 = _sha256(raw_sha256, field=f"course output {name}")
        path = supervisor._regular_file(output / name, name=f"course output {name}")
        observed = sha256_file(path)
        if observed != expected_sha256:
            raise supervisor.ProbeError(f"course output hash differs: {name}")
        expected_names.add(name)
        artifacts[name] = {"path": name, "sha256": observed, "size": path.stat().st_size}
        if name in {TELEMETRY_FILENAME, SCALED_TELEMETRY_FILENAME}:
            telemetry_encoded = supervisor._read_bounded(
                path, MAX_TELEMETRY_BYTES, "course training telemetry"
            )
    if _output_names(output) != expected_names:
        raise supervisor.ProbeError("course run has unknown or missing output files")
    supervisor._read_bounded(
        output / supervisor.STDOUT_FILENAME, MAX_LOG_BYTES, "course child stdout"
    )
    supervisor._read_bounded(
        output / supervisor.STDERR_FILENAME, MAX_LOG_BYTES, "course child stderr"
    )
    retained_config = supervisor._read_bounded(
        output / "input_config.json", MAX_CONFIG_BYTES, "retained input config"
    )
    source_config = supervisor._read_bounded(
        Path(config_binding["path"]), MAX_CONFIG_BYTES, "admitted input config"
    )
    if retained_config != source_config:
        raise supervisor.ProbeError("retained course config differs from admitted input bytes")
    if plan.inputs.get("course_mode") != mode:
        raise supervisor.ProbeError("course mode differs from the accepted plan")
    task = raw.get("task")
    if not isinstance(task, Mapping) or type(task.get("horizon_steps")) is not int:
        raise supervisor.ProbeError("admitted course task binding is malformed")
    horizon_steps = task["horizon_steps"]
    assert isinstance(horizon_steps, int)
    if not isinstance(loaded.course_identities, Mapping):
        raise supervisor.ProbeError("course semantic identity binding is missing")
    task_sha256 = _sha256(loaded.course_identities.get("task"), field="course task")
    _validate_course_summary(
        manifest.get("zero_residual"),
        label="zero-residual",
        horizon_steps=horizon_steps,
        task_sha256=task_sha256,
        zero_residual=True,
    )
    if mode == "probe":
        if manifest.get("final_policy") is not None:
            raise supervisor.ProbeError("probe course run reports a final policy")
    else:
        _validate_course_summary(
            manifest.get("final_policy"),
            label="final-policy",
            horizon_steps=horizon_steps,
            task_sha256=task_sha256,
            zero_residual=False,
        )
    training_steps = raw.get("training_steps")
    if type(training_steps) is not int:
        raise supervisor.ProbeError("course training-step binding is malformed")
    _validate_training_record(
        manifest.get("training"),
        mode=mode,
        steps=training_steps,
        telemetry_encoded=telemetry_encoded,
        trainer=loaded.course_trainer,
    )
    if manifest.get("identities") != loaded.course_identities:
        raise supervisor.ProbeError("course semantic identities differ from admitted config")
    expected_claims = {
        "development_only": True,
        "heldout_generalization_tested": False,
        "physical_obstacle_scene": False,
        "training_performed": mode == "train",
        "full_llm_revision_loop_demonstrated": False,
    }
    if manifest.get("claims") != expected_claims:
        raise supervisor.ProbeError("course run claim limits differ")
    frozen_runtime = manifest.get("frozen_runtime")
    if (
        not isinstance(frozen_runtime, Mapping)
        or (
            loaded.course_trainer is not None
            and frozen_runtime.get("trainer")
            != effective_training_contract(loaded.course_trainer)
        )
        or not isinstance(manifest.get("runtime"), Mapping)
    ):
        raise supervisor.ProbeError("course runtime provenance is missing")
    return {
        "course_manifest": {
            "path": COURSE_MANIFEST_FILENAME,
            "sha256": manifest_sha256,
            "size": manifest_path.stat().st_size,
        },
        "outputs": artifacts,
    }


def verify_development_completed(plan: supervisor.ProbePlan) -> dict[str, object]:
    workload = plan.inputs.get("workload")
    if workload == "parity":
        return _verify_parity(plan)
    if workload == "course":
        return _verify_course(plan)
    raise supervisor.ProbeError("development workload binding is invalid")


def _dependencies() -> supervisor.ProbeDependencies:
    return replace(
        supervisor.ProbeDependencies(),
        validate_plan=validate_development_plan,
        verify_completed=verify_development_completed,
    )


def _run_parity_child(config_path: Path, output: Path) -> dict[str, object]:
    from oracle_composition.adapters.gmt.parity import RuntimeParityConfig, run_runtime_parity

    output = supervisor._absolute(output)
    if not output.is_dir() or output.is_symlink() or _output_names(output) != {
        supervisor.STDOUT_FILENAME,
        supervisor.STDERR_FILENAME,
    }:
        raise supervisor.ProbeError("parity child requires the fresh supervisor output directory")
    loaded = _load_parity(config_path)
    raw = loaded.raw
    return run_runtime_parity(
        RuntimeParityConfig(
            trace_path=Path(raw["trace_path"]),
            trace_sha256=str(raw["trace_sha256"]),
            manifest_path=Path(raw["manifest_path"]),
            manifest_sha256=str(raw["manifest_sha256"]),
            upstream_root=Path(raw["upstream_root"]),
            weights_path=Path(raw["weights_path"]),
            weights_sha256=str(raw["weights_sha256"]),
            motion_path=Path(raw["motion_path"]),
            motion_sha256=str(raw["motion_sha256"]),
            motion_name=str(raw["motion_name"]),
            output_path=output / PARITY_FILENAME,
        )
    )


def _shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workload", choices=("parity", "course"), required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--python", dest="venv_python", type=Path, required=True)
    parser.add_argument("--config", dest="config_path", type=Path, required=True)
    parser.add_argument("--output", dest="output_directory", type=Path, required=True)
    parser.add_argument("--owner", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    request = commands.add_parser("request")
    _shared_arguments(request)
    run = commands.add_parser("run")
    _shared_arguments(run)
    run.add_argument("--coordination-root", type=Path, required=True)
    run.add_argument("--reservation", type=Path, required=True)
    child = commands.add_parser("parity-child", help=argparse.SUPPRESS)
    child.add_argument("--config", type=Path, required=True)
    child.add_argument("--output", type=Path, required=True)
    return parser


def _request(arguments: argparse.Namespace) -> DevelopmentRequest:
    return DevelopmentRequest(
        workload=arguments.workload,
        repository_root=arguments.repository_root,
        venv_python=arguments.venv_python,
        config_path=arguments.config_path,
        output_directory=arguments.output_directory,
        owner=arguments.owner,
    )


def main() -> int:
    arguments = _parser().parse_args()
    try:
        if arguments.command == "parity-child":
            result = _run_parity_child(arguments.config, arguments.output)
        else:
            plan = build_development_plan(_request(arguments))
            if arguments.command == "request":
                result = {
                    "schema_version": 1,
                    "exact_probe_binding": plan.accepted_fields(),
                    "effective_limits": plan.limits.as_dict(),
                    "reservation_v2_dynamic_fields_required": [
                        "accepted_until_utc",
                        "acceptance_message",
                        "acknowledgment",
                        "proposal_id",
                        "required_authorizer",
                    ],
                }
            else:
                reservation = load_validated_reservation(
                    arguments.coordination_root, arguments.reservation
                )
                result = supervisor.supervise_probe(
                    plan,
                    coordination_root=arguments.coordination_root,
                    reservation=reservation,
                    reservation_path=arguments.reservation,
                    dependencies=_dependencies(),
                )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0 if result.get("ok", True) else 2
    except (
        GMTAdmissionError,
        supervisor.ProbeError,
        ResourceSlotError,
        OSError,
        ValueError,
    ) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
