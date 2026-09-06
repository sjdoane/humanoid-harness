"""Strict inputs and receipts for the T1 development reference ablation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import (
    array_sha256,
    canonical_json_bytes,
)
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.reference_corpus_bundle import load_bundle_manifest
from oracle_composition.experiments.reference_input_transforms import (
    CONDITION_IDS,
    HORIZON_STEPS,
    REFERENCE_WIDTH,
    SHIFT_FRAMES,
    SHUFFLE_SEED,
    float64_array_sha256,
    transform_reference_input,
)
from oracle_composition.phase_b.isolation import (
    SealedArtifact,
    sealed_input_lineage_value,
    verify_sealed_inputs,
)
from oracle_composition.phase_b.persistence import _valid_success_resource_controls
from oracle_composition.phase_b.report_v2 import SeedReportFacts, validate_training_facts
from oracle_composition.phase_b.supervision import (
    _TRAINING_FACT_KEYS,
    _valid_persistence_receipt,
    _valid_step_zero_comparator,
)
from oracle_composition.phase_b.training import (
    SMOKE_SEED,
    SMOKE_TRANSITIONS,
    TRAINING_BLOCKS,
)

PROTOCOL_ID: Final = "humanoid_phase_b_t1_development_reference_ablation/v1"
PROTOCOL_SCHEMA_VERSION: Final = 1
EVIDENCE_CLASS: Final = "development_exploratory_reference_use_only"
CLAIM_CEILING: Final = (
    "development_only_in_sample_reference_input_sensitivity_no_protected_utility_"
    "promotion_oracle_quality_generalization_naturalness_or_humanoid_competence_claim"
)
DEVELOPMENT_BLOCKS: Final = (120001, 120002, 120003, 120005)
SCHEDULE_BEHAVIORS: Final = ("expert", "simple", "expert")
SCHEDULE_BOUNDARIES: Final = (0, 300, 600, 1000)
TARGET_SPEEDS_M_S: Final = (
    5.520768616125457,
    0.8853599908576963,
    5.520768616125457,
)
PROTECTED_EVALUATION_BLOCKS: Final = frozenset(range(120101, 120121))
CALIBRATION_BLOCKS: Final = frozenset(range(120201, 120221))
MAX_JSON_BYTES: Final = 16 * 1024 * 1024
EXPECTED_ROLLOUTS: Final = 24
CORPUS_PREFIX: Final = Path("artifacts/reference_corpus_v2")

_REPORT_INPUT_KEYS: Final = frozenset(
    {
        "e1_receipt_sha256",
        "evaluator_sha256",
        "execution_manifest_sha256",
        "library_sha256",
        "oracle_canonical_sha256",
        "oracle_file_sha256",
        "reference_sha256",
        "reward_compositor_sha256",
        "reward_file_sha256",
        "reward_formula_id",
        "reward_formula_sha256",
        "reward_schema_id",
        "sealed_input_lineage_sha256",
        "starting_expert_identity",
        "task_sha256",
        "training_design_sha256",
    }
)
_EXECUTION_MANIFEST_KEYS: Final = frozenset(
    {
        "checkpoint_selection",
        "e003_execution_manifest_sha256",
        "evidence_class",
        "execution_manifest_schema_id",
        "ft1_run_manifest_sha256",
        "inputs",
        "prior_scientific_receipt_sha256",
        "reservation",
        "reservation_sha256",
        "resource_control_policy",
        "resource_limits",
        "runtime_source_snapshot",
        "runtime_source_snapshot_sha256",
        "schema_version",
        "sealed_input_lineage",
        "sealed_input_lineage_sha256",
        "seeds",
        "smoke",
        "test_only",
        "transitions_per_seed",
    }
)
_SUCCESS_RECEIPT_KEYS: Final = frozenset(
    {
        "artifacts",
        "evidence_class",
        "execution_manifest_sha256",
        "failure_receipt_present",
        "outcome",
        "planned_transitions",
        "ppo_seed",
        "promotable",
        "resource_controls",
        "schema_version",
        "smoke",
        "status",
        "success_receipt_id",
        "test_only",
        "worker_cleanup",
    }
)
_RUNTIME_SOURCE_REQUIRED: Final = frozenset(
    {
        "src/oracle_composition/envs/humanoid.py",
        "src/oracle_composition/phase_b/policy.py",
        "src/oracle_composition/phase_b/reference_runtime.py",
        "src/oracle_composition/phase_b/runtime.py",
        "src/oracle_composition/phase_b/training.py",
    }
)


class DevelopmentAblationError(ExperimentContractError):
    """The development-only protocol or input lineage is invalid."""


@dataclass(frozen=True, slots=True)
class DevelopmentProtocol:
    path: Path
    sha256: str
    byte_count: int
    development_blocks: tuple[int, ...]
    conditions: tuple[str, ...]
    target_speeds_m_s: tuple[float, float, float]
    value: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class SmokeExportLineage:
    run_directory: Path
    actor_path: Path
    actor_sha256: str
    actor_byte_count: int
    checkpoint_sha256: str
    execution_manifest_sha256: str
    training_facts_sha256: str
    report_inputs: Mapping[str, object]
    sealed_inputs: Mapping[str, Mapping[str, object]]
    bindings: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True, slots=True)
class PreparedReferenceTransform:
    condition_id: str
    transformed_full_sequence: np.ndarray
    index_map: tuple[int | None, ...]
    receipt: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class PreparedReferenceWindow:
    values: np.ndarray
    timeline_indices: tuple[int, ...]
    source_indices: tuple[int | None, ...]
    sha256: str


def _require_object(value: object, *, field: str) -> dict[str, object]:
    if type(value) is not dict:
        raise DevelopmentAblationError(f"{field} must be an object")
    return value


def _require_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DevelopmentAblationError(f"{field} must be a lowercase SHA-256")
    return value


def _read_canonical_json(
    path: Path,
    *,
    field: str,
    allow_source_newline: bool = False,
) -> tuple[object, bytes]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise DevelopmentAblationError(f"{field} is not a regular file")
    size = candidate.stat().st_size
    if not 0 < size <= MAX_JSON_BYTES:
        raise DevelopmentAblationError(f"{field} exceeds its JSON size bound")
    encoded = candidate.read_bytes()
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise DevelopmentAblationError(f"{field} is invalid JSON") from exc
    canonical = canonical_json_bytes(value)
    accepted = {canonical, canonical + b"\n"} if allow_source_newline else {canonical}
    if encoded not in accepted:
        raise DevelopmentAblationError(f"{field} is not canonical JSON")
    return value, encoded


def _artifact_binding(value: object, *, field: str) -> dict[str, object]:
    record = _require_object(value, field=field)
    if set(record) != {"byte_count", "filename", "sha256"}:
        raise DevelopmentAblationError(f"{field} binding fields differ")
    filename = record["filename"]
    if (
        type(filename) is not str
        or filename in {"", ".", ".."}
        or Path(filename).name != filename
        or type(record["byte_count"]) is not int
        or record["byte_count"] <= 0
    ):
        raise DevelopmentAblationError(f"{field} binding is malformed")
    _require_sha256(record["sha256"], field=f"{field} SHA-256")
    return record


def _verify_bound_file(directory: Path, record: object, *, field: str) -> Path:
    binding = _artifact_binding(record, field=field)
    path = directory / str(binding["filename"])
    if path.is_symlink() or not path.is_file():
        raise DevelopmentAblationError(f"{field} artifact is unavailable")
    encoded = path.read_bytes()
    if (
        len(encoded) != binding["byte_count"]
        or hashlib.sha256(encoded).hexdigest() != binding["sha256"]
    ):
        raise DevelopmentAblationError(f"{field} artifact binding differs")
    return path


def _sealed_input_map(value: object) -> Mapping[str, Mapping[str, object]]:
    lineage = _require_object(value, field="smoke sealed input lineage")
    artifacts = lineage.get("artifacts")
    if (
        set(lineage) != {"artifacts", "lineage_id", "schema_version"}
        or lineage.get("lineage_id") != "humanoid_phase_b_sealed_input_lineage/v1"
        or lineage.get("schema_version") != 1
        or type(artifacts) is not list
        or not artifacts
    ):
        raise DevelopmentAblationError("smoke sealed input lineage is malformed")
    try:
        sealed = tuple(
            SealedArtifact(
                relative_path=raw["path"],
                byte_count=raw["byte_count"],
                sha256=raw["sha256"],
                roles=tuple(raw["roles"]),
            )
            for raw in artifacts
            if type(raw) is dict and set(raw) == {"byte_count", "path", "roles", "sha256"}
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DevelopmentAblationError("smoke sealed input record is malformed") from exc
    if len(sealed) != len(artifacts) or sealed_input_lineage_value(sealed) != lineage:
        raise DevelopmentAblationError("smoke sealed input lineage is not canonical")
    return MappingProxyType(
        {artifact.relative_path: MappingProxyType(artifact.to_dict()) for artifact in sealed}
    )


def _validate_runtime_source_snapshot(value: Mapping[str, object]) -> None:
    if set(value) != {
        "authority_identities",
        "device",
        "git",
        "platform",
        "source_sha256",
        "versions",
    }:
        raise DevelopmentAblationError("smoke runtime source snapshot fields differ")
    git = value.get("git")
    platform = value.get("platform")
    versions = value.get("versions")
    sources = value.get("source_sha256")
    if (
        value.get("device") != "cpu"
        or type(value.get("authority_identities")) is not dict
        or type(git) is not dict
        or set(git) != {"clean", "commit"}
        or git.get("clean") is not True
        or type(git.get("commit")) is not str
        or len(git["commit"]) != 40
        or any(character not in "0123456789abcdef" for character in git["commit"])
        or type(platform) is not dict
        or set(platform) != {"machine", "python", "system"}
        or any(type(item) is not str or not item for item in platform.values())
        or type(versions) is not dict
        or set(versions) != {"gymnasium", "mujoco", "numpy", "stable_baselines3", "torch"}
        or any(type(item) is not str or not item for item in versions.values())
        or type(sources) is not dict
        or not set(sources) >= _RUNTIME_SOURCE_REQUIRED
    ):
        raise DevelopmentAblationError("smoke runtime source snapshot is malformed")
    for path, digest in sources.items():
        if (
            type(path) is not str
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
        ):
            raise DevelopmentAblationError("smoke runtime source path is malformed")
        _require_sha256(digest, field=f"smoke runtime source {path}")


def _verify_sealed_corpus_file(
    corpus_root: Path,
    sealed_inputs: Mapping[str, Mapping[str, object]],
    relative_path: Path,
    *,
    role: str,
) -> Mapping[str, object]:
    logical_path = (CORPUS_PREFIX / relative_path).as_posix()
    record = sealed_inputs.get(logical_path)
    if record is None or role not in record["roles"]:
        raise DevelopmentAblationError(f"smoke sealed inputs omit {role}")
    candidate = corpus_root / relative_path
    if candidate.is_symlink() or not candidate.is_file():
        raise DevelopmentAblationError(f"sealed corpus input is unavailable: {relative_path}")
    encoded = candidate.read_bytes()
    if (
        len(encoded) != record["byte_count"]
        or hashlib.sha256(encoded).hexdigest() != record["sha256"]
    ):
        raise DevelopmentAblationError(f"sealed corpus input differs: {relative_path}")
    return record


def validate_development_corpus(
    corpus_root: Path,
    *,
    lineage: SmokeExportLineage,
    protocol: DevelopmentProtocol,
) -> Mapping[str, Mapping[str, object]]:
    """Bind every selected development clip to the smoke's sealed corpus bytes."""

    root = Path(corpus_root).resolve(strict=True)
    repository_root = root.parent.parent
    if root != repository_root / CORPUS_PREFIX:
        raise DevelopmentAblationError("corpus root is not the production artifact location")
    sealed_artifacts = tuple(
        SealedArtifact(
            relative_path=path,
            byte_count=int(record["byte_count"]),
            sha256=str(record["sha256"]),
            roles=tuple(record["roles"]),
        )
        for path, record in lineage.sealed_inputs.items()
    )
    try:
        verify_sealed_inputs(
            repository_root,
            sealed_artifacts,
            expected_lineage_sha256=str(lineage.report_inputs["sealed_input_lineage_sha256"]),
        )
    except (ExperimentContractError, OSError, ValueError) as exc:
        raise DevelopmentAblationError(
            "smoke sealed inputs do not match the corpus checkout"
        ) from exc
    validated: dict[str, Mapping[str, object]] = {}
    manifest_record = _verify_sealed_corpus_file(
        root,
        lineage.sealed_inputs,
        Path("corpus_manifest_v2.json"),
        role="reference_corpus",
    )
    if manifest_record["sha256"] != lineage.report_inputs["reference_sha256"]:
        raise DevelopmentAblationError("corpus manifest differs from smoke reference authority")
    validated["corpus_manifest_v2.json"] = manifest_record
    index_record = _verify_sealed_corpus_file(
        root,
        lineage.sealed_inputs,
        Path("corpus_index_v2.json"),
        role="reference_corpus.index",
    )
    validated["corpus_index_v2.json"] = index_record
    manifest_raw, _ = _read_canonical_json(
        root / "corpus_manifest_v2.json", field="reference corpus manifest"
    )
    index_raw, _ = _read_canonical_json(
        root / "corpus_index_v2.json", field="reference corpus index"
    )
    manifest = _require_object(manifest_raw, field="reference corpus manifest")
    index = _require_object(index_raw, field="reference corpus index")
    manifest_entries = {
        entry.get("clip_id"): entry
        for entry in manifest.get("clips_in_reset_order", ())
        if type(entry) is dict
    }
    index_entries = {
        entry.get("clip_id"): entry for entry in index.get("clips", ()) if type(entry) is dict
    }
    for block in protocol.development_blocks:
        for behavior in ("expert", "simple"):
            clip_id = f"corpus-{block}-{behavior}"
            manifest_entry = manifest_entries.get(clip_id)
            index_entry = index_entries.get(clip_id)
            bundle_sha = index_entry.get("bundle_manifest_sha256") if index_entry else None
            if (
                type(manifest_entry) is not dict
                or type(index_entry) is not dict
                or manifest_entry.get("bundle_manifest_sha256") != bundle_sha
            ):
                raise DevelopmentAblationError(f"corpus authorities differ for {clip_id}")
            _require_sha256(bundle_sha, field=f"{clip_id} bundle SHA-256")
            bundle_relative = Path("clips") / clip_id / f"bundle-{bundle_sha}.json"
            bundle_record = _verify_sealed_corpus_file(
                root,
                lineage.sealed_inputs,
                bundle_relative,
                role=f"reference_corpus.bundle.{clip_id}",
            )
            validated[bundle_relative.as_posix()] = bundle_record
            try:
                bundle = load_bundle_manifest(root / bundle_relative, expected_sha256=bundle_sha)
            except (OSError, ValueError) as exc:
                raise DevelopmentAblationError(f"invalid sealed bundle for {clip_id}") from exc
            core = bundle["core"]
            nested = [*core["bound_artifacts"], core["payload"], core["rng_state"]["binding"]]
            for binding in nested:
                object_path = Path(str(binding["object_path"]))
                record = _verify_sealed_corpus_file(
                    root,
                    lineage.sealed_inputs,
                    object_path,
                    role=f"reference_corpus.{clip_id}.{binding['role']}",
                )
                if (
                    record["byte_count"] != binding["byte_count"]
                    or record["sha256"] != binding["sha256"]
                ):
                    raise DevelopmentAblationError(f"bundle and sealed input differ for {clip_id}")
                validated[object_path.as_posix()] = record
    return MappingProxyType(dict(sorted(validated.items())))


def load_development_protocol(path: Path) -> DevelopmentProtocol:
    """Load the one preregistered in-sample protocol and reject broader variants."""

    candidate = Path(path).resolve(strict=True)
    raw, encoded = _read_canonical_json(
        candidate,
        field="development protocol",
        allow_source_newline=True,
    )
    value = _require_object(raw, field="development protocol")
    expected_fields = {
        "actor_action",
        "actor_visible_factor",
        "calibration_inputs",
        "claim_ceiling",
        "conditions",
        "development_blocks",
        "evidence_class",
        "held_out_inputs_used",
        "horizon_steps",
        "policy_seed",
        "promotable",
        "protocol_id",
        "reference_schedule",
        "schema_version",
        "shuffle_seed",
        "source_split",
        "task_success_scoring",
        "time_shift_frames",
        "trace_requirement",
    }
    if set(value) != expected_fields:
        raise DevelopmentAblationError("development protocol fields differ")
    expected_scalars = {
        "actor_action": "deterministic_mean_epsilon_none",
        "actor_visible_factor": "reference_window_only",
        "calibration_inputs": "forbidden",
        "claim_ceiling": CLAIM_CEILING,
        "evidence_class": EVIDENCE_CLASS,
        "held_out_inputs_used": False,
        "horizon_steps": 1000,
        "policy_seed": SMOKE_SEED,
        "promotable": False,
        "protocol_id": PROTOCOL_ID,
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "shuffle_seed": SHUFFLE_SEED,
        "source_split": "phase_b_training_reference_blocks_reused_in_sample",
        "task_success_scoring": False,
        "time_shift_frames": SHIFT_FRAMES,
        "trace_requirement": (
            "all_four_closed_loop_arms_plus_matched_state_action_deltas_on_exact_arm"
        ),
    }
    if any(value.get(field) != expected for field, expected in expected_scalars.items()):
        raise DevelopmentAblationError("development protocol fixed authority differs")
    if type(value["development_blocks"]) is not list or type(value["conditions"]) is not list:
        raise DevelopmentAblationError("development protocol arms or blocks are malformed")
    blocks = tuple(value["development_blocks"])
    conditions = tuple(value["conditions"])
    if blocks != DEVELOPMENT_BLOCKS or conditions != CONDITION_IDS:
        raise DevelopmentAblationError("development protocol arms or blocks differ")
    if (
        not set(blocks) <= set(TRAINING_BLOCKS)
        or set(blocks) & PROTECTED_EVALUATION_BLOCKS
        or set(blocks) & CALIBRATION_BLOCKS
    ):
        raise DevelopmentAblationError("development protocol crosses a frozen data split")
    schedule = value["reference_schedule"]
    if type(schedule) is not list:
        raise DevelopmentAblationError("development reference schedule is malformed")
    expected_schedule = [
        {
            "behavior": behavior,
            "start": start,
            "stop": stop,
            "target_m_s": target,
        }
        for behavior, start, stop, target in zip(
            SCHEDULE_BEHAVIORS,
            SCHEDULE_BOUNDARIES[:-1],
            SCHEDULE_BOUNDARIES[1:],
            TARGET_SPEEDS_M_S,
            strict=True,
        )
    ]
    if schedule != expected_schedule:
        raise DevelopmentAblationError("development schedule must be direct expert-simple-expert")
    return DevelopmentProtocol(
        path=candidate,
        sha256=hashlib.sha256(encoded).hexdigest(),
        byte_count=len(encoded),
        development_blocks=blocks,
        conditions=conditions,
        target_speeds_m_s=TARGET_SPEEDS_M_S,
        value=MappingProxyType(value),
    )


def validate_smoke_export(run_directory: Path) -> SmokeExportLineage:
    """Validate the successful nonpromotable T1 smoke chain without cohort admission."""

    raw_root = Path(run_directory)
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise DevelopmentAblationError("smoke run must be a real directory")
    root = raw_root.resolve(strict=True)
    seed_directory = root / f"seed_{SMOKE_SEED}"
    if seed_directory.is_symlink() or not seed_directory.is_dir():
        raise DevelopmentAblationError("smoke seed directory is unavailable")

    manifest_raw, manifest_bytes = _read_canonical_json(
        root / "execution_manifest_v3.json", field="smoke execution manifest"
    )
    manifest = _require_object(manifest_raw, field="smoke execution manifest")
    execution_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if set(manifest) != _EXECUTION_MANIFEST_KEYS:
        raise DevelopmentAblationError("execution manifest fields differ from production")
    expected_manifest = {
        "checkpoint_selection": "final_transition_only",
        "evidence_class": "interface_check",
        "execution_manifest_schema_id": "humanoid_phase_b_execution_manifest/v3",
        "schema_version": 3,
        "seeds": [SMOKE_SEED],
        "smoke": True,
        "test_only": False,
        "transitions_per_seed": SMOKE_TRANSITIONS,
    }
    if any(manifest.get(field) != expected for field, expected in expected_manifest.items()):
        raise DevelopmentAblationError("execution manifest is not the real T1 smoke")
    inputs = _require_object(manifest.get("inputs"), field="smoke report inputs")
    if set(inputs) != _REPORT_INPUT_KEYS:
        raise DevelopmentAblationError("smoke report input identity set differs")
    for field in _REPORT_INPUT_KEYS - {
        "reward_formula_id",
        "reward_schema_id",
        "starting_expert_identity",
    }:
        _require_sha256(inputs.get(field), field=f"smoke {field}")
    lineage_value = manifest.get("sealed_input_lineage")
    lineage_sha256 = hashlib.sha256(canonical_json_bytes(lineage_value)).hexdigest()
    if (
        manifest.get("sealed_input_lineage_sha256") != lineage_sha256
        or inputs["sealed_input_lineage_sha256"] != lineage_sha256
    ):
        raise DevelopmentAblationError("smoke sealed input lineage digest differs")
    sealed_inputs = _sealed_input_map(lineage_value)
    source_snapshot = _require_object(
        manifest.get("runtime_source_snapshot"), field="smoke runtime source snapshot"
    )
    _validate_runtime_source_snapshot(source_snapshot)
    if (
        manifest.get("runtime_source_snapshot_sha256")
        != hashlib.sha256(canonical_json_bytes(source_snapshot)).hexdigest()
    ):
        raise DevelopmentAblationError("smoke runtime source snapshot digest differs")
    reservation = _require_object(manifest.get("reservation"), field="smoke reservation")
    if (
        manifest.get("reservation_sha256")
        != hashlib.sha256(canonical_json_bytes(reservation)).hexdigest()
    ):
        raise DevelopmentAblationError("smoke reservation digest differs")
    expected_control_policy = {
        "cpu_time": "os_rlimit_when_supported_otherwise_recorded_unsupported",
        "environment": "spawn_time_explicit_allowlist",
        "filesystem": "parent_observed_os_best_effort",
        "process_group_cleanup": "os_session_group_best_effort_with_fail_closed_receipt",
        "process_tree_rss": "parent_observed_os_best_effort",
    }
    if manifest.get("resource_control_policy") != expected_control_policy:
        raise DevelopmentAblationError("smoke resource control policy differs")
    for field in (
        "e003_execution_manifest_sha256",
        "ft1_run_manifest_sha256",
        "prior_scientific_receipt_sha256",
    ):
        _require_sha256(manifest.get(field), field=f"smoke {field}")

    job_raw, job_bytes = _read_canonical_json(root / "job_result_v1.json", field="smoke job result")
    job = _require_object(job_raw, field="smoke job result")
    outcomes = job.get("outcomes")
    if (
        set(job)
        != {
            "checkpoint_index",
            "execution_manifest_sha256",
            "job_result_schema_id",
            "outcomes",
            "schema_version",
            "status",
        }
        or job.get("job_result_schema_id") != "humanoid_phase_b_job_result/v1"
        or job.get("schema_version") != 1
        or job.get("status") != "succeeded"
        or job.get("checkpoint_index") is not None
        or job.get("execution_manifest_sha256") != execution_sha256
        or type(outcomes) is not list
        or len(outcomes) != 1
    ):
        raise DevelopmentAblationError("smoke job result is not a successful noncohort job")
    outcome = _require_object(outcomes[0], field="smoke job outcome")
    if (
        set(outcome) != {"receipt", "seed", "status"}
        or outcome.get("seed") != SMOKE_SEED
        or outcome.get("status") != "succeeded"
    ):
        raise DevelopmentAblationError("smoke seed outcome did not succeed")
    success_path = _verify_bound_file(
        seed_directory, outcome.get("receipt"), field="smoke success receipt"
    )
    if success_path.name != "success_receipt_v2.json":
        raise DevelopmentAblationError("smoke success receipt filename differs")
    success_raw, success_bytes = _read_canonical_json(success_path, field="smoke success receipt")
    success = _require_object(success_raw, field="smoke success receipt")
    if set(success) != _SUCCESS_RECEIPT_KEYS:
        raise DevelopmentAblationError("smoke success receipt fields differ")
    expected_success = {
        "evidence_class": "interface_check",
        "execution_manifest_sha256": execution_sha256,
        "failure_receipt_present": False,
        "outcome": "success",
        "planned_transitions": SMOKE_TRANSITIONS,
        "ppo_seed": SMOKE_SEED,
        "promotable": False,
        "schema_version": 2,
        "smoke": True,
        "status": "succeeded",
        "success_receipt_id": "humanoid_phase_b_seed_success/v2",
        "test_only": False,
    }
    if (
        any(success.get(field) != expected for field, expected in expected_success.items())
        or success.get("worker_cleanup") != {"attempted": True, "error": None, "succeeded": True}
        or not _valid_success_resource_controls(success.get("resource_controls"))
    ):
        raise DevelopmentAblationError("seed receipt is not the successful real T1 smoke")
    if (seed_directory / "failure_receipt_v2.json").exists():
        raise DevelopmentAblationError("smoke seed has a contradictory failure receipt")
    if (root / "checkpoint_index_v1.json").exists():
        raise DevelopmentAblationError("smoke must not have a promotable checkpoint index")

    artifacts = _require_object(success.get("artifacts"), field="smoke success artifacts")
    if set(artifacts) != {"persistence", "rsi_ledger", "training_facts"}:
        raise DevelopmentAblationError("smoke success artifact set differs")
    persistence_path = _verify_bound_file(
        seed_directory, artifacts["persistence"], field="smoke persistence receipt"
    )
    training_path = _verify_bound_file(
        seed_directory, artifacts["training_facts"], field="smoke training facts"
    )
    rsi_path = _verify_bound_file(seed_directory, artifacts["rsi_ledger"], field="smoke RSI ledger")
    if persistence_path.name != f"persistence_seed_{SMOKE_SEED}_v1.json":
        raise DevelopmentAblationError("smoke persistence receipt filename differs")
    if training_path.name != "training_facts_v1.json" or rsi_path.name != "rsi_ledger_v1.json":
        raise DevelopmentAblationError("smoke scientific artifact filename differs")

    training_raw, training_bytes = _read_canonical_json(training_path, field="smoke training facts")
    training = _require_object(training_raw, field="smoke training facts")
    training_sha256 = hashlib.sha256(training_bytes).hexdigest()
    if set(training) != _TRAINING_FACT_KEYS:
        raise DevelopmentAblationError("smoke training fact fields differ")
    unfreeze_rollouts = training.get("unfreeze_rollouts")
    expected_training = {
        "evidence_class": "interface_check",
        "execution_manifest_sha256": execution_sha256,
        "observed_transitions": SMOKE_TRANSITIONS,
        "planned_transitions": SMOKE_TRANSITIONS,
        "ppo_seed": SMOKE_SEED,
        "promotable": False,
        "rollouts": EXPECTED_ROLLOUTS,
        "smoke": True,
    }
    if (
        any(training.get(field) != expected for field, expected in expected_training.items())
        or type(unfreeze_rollouts) is not list
        or len(unfreeze_rollouts) != EXPECTED_ROLLOUTS
        or any(
            type(row) is not dict
            or row.get("rollout_index") != index
            or row.get("actor_stage") != ("reference_columns_only" if index < 8 else "full_actor")
            for index, row in enumerate(unfreeze_rollouts)
        )
    ):
        raise DevelopmentAblationError("smoke training facts are incomplete")
    rsi_raw, rsi_bytes = _read_canonical_json(rsi_path, field="smoke RSI ledger")
    if (
        type(rsi_raw) is not list
        or not rsi_raw
        or training.get("rsi_ledger_sha256") != hashlib.sha256(rsi_bytes).hexdigest()
    ):
        raise DevelopmentAblationError("smoke RSI ledger does not cover every rollout")

    persistence_raw, persistence_bytes = _read_canonical_json(
        persistence_path, field="smoke persistence receipt"
    )
    persistence = _require_object(persistence_raw, field="smoke persistence receipt")
    if not _valid_persistence_receipt(persistence):
        raise DevelopmentAblationError("smoke persistence receipt is not production-shaped")
    expected_persistence = {
        "checkpoint_reload_bitwise_deterministic": True,
        "checkpoint_to_export_bitwise_equivalent": True,
        "evidence_class": "interface_check",
        "execution_manifest_sha256": execution_sha256,
        "final_transition_only": True,
        "persistence_receipt_id": "humanoid_phase_b_final_persistence/v1",
        "planned_transitions": SMOKE_TRANSITIONS,
        "ppo_seed": SMOKE_SEED,
        "promotable": False,
        "schema_version": 1,
        "smoke": True,
        "test_only": False,
        "training_facts_sha256": training_sha256,
        "transitions": SMOKE_TRANSITIONS,
    }
    if any(persistence.get(field) != expected for field, expected in expected_persistence.items()):
        raise DevelopmentAblationError("smoke persistence lineage differs")
    actor_record = _artifact_binding(
        persistence.get("strict_export"), field="smoke strict actor export"
    )
    checkpoint_record = _artifact_binding(
        persistence.get("checkpoint"), field="smoke final checkpoint"
    )
    actor_path = _verify_bound_file(seed_directory, actor_record, field="smoke strict actor export")
    _verify_bound_file(seed_directory, checkpoint_record, field="smoke final checkpoint")
    if actor_path.name != f"actor_seed_{SMOKE_SEED}_final.npz":
        raise DevelopmentAblationError("smoke strict actor filename differs")
    step_zero = training.get("step_zero_comparator")
    if not _valid_step_zero_comparator(step_zero):
        raise DevelopmentAblationError("smoke step-zero comparator is invalid")
    try:
        validate_training_facts(
            SeedReportFacts(
                ppo_seed=SMOKE_SEED,
                checkpoint_sha256=str(checkpoint_record["sha256"]),
                strict_export_sha256=str(actor_record["sha256"]),
                training=training,
                step_zero_comparator=step_zero,
                execution_manifest_bytes=manifest_bytes,
                rsi_ledger_bytes=rsi_bytes,
            )
        )
    except ValueError as exc:
        raise DevelopmentAblationError(
            "smoke training facts differ from production authority"
        ) from exc

    bindings = {
        "execution_manifest": {
            "byte_count": len(manifest_bytes),
            "sha256": execution_sha256,
        },
        "job_result": {
            "byte_count": len(job_bytes),
            "sha256": hashlib.sha256(job_bytes).hexdigest(),
        },
        "persistence_receipt": {
            "byte_count": len(persistence_bytes),
            "sha256": hashlib.sha256(persistence_bytes).hexdigest(),
        },
        "success_receipt": {
            "byte_count": len(success_bytes),
            "sha256": hashlib.sha256(success_bytes).hexdigest(),
        },
        "training_facts": {
            "byte_count": len(training_bytes),
            "sha256": training_sha256,
        },
    }
    return SmokeExportLineage(
        run_directory=root,
        actor_path=actor_path,
        actor_sha256=str(actor_record["sha256"]),
        actor_byte_count=int(actor_record["byte_count"]),
        checkpoint_sha256=str(checkpoint_record["sha256"]),
        execution_manifest_sha256=execution_sha256,
        training_facts_sha256=training_sha256,
        report_inputs=MappingProxyType(inputs),
        sealed_inputs=sealed_inputs,
        bindings=MappingProxyType(
            {name: MappingProxyType(record) for name, record in bindings.items()}
        ),
    )


def load_bound_smoke_actor(
    lineage: SmokeExportLineage,
    *,
    loader: Callable[..., object] | None = None,
) -> object:
    """Construct only the strict actor bound by a validated smoke persistence receipt."""

    if type(lineage) is not SmokeExportLineage:
        raise DevelopmentAblationError("strict actor construction requires smoke lineage")
    if loader is None:
        from oracle_composition.phase_b.persistence import load_trained_full_authority_actor

        loader = load_trained_full_authority_actor
    return loader(lineage.actor_path, expected_sha256=lineage.actor_sha256)


def prepare_reference_transform(
    source_reference: np.ndarray,
    *,
    condition_id: str,
) -> PreparedReferenceTransform:
    """Prepare one full-sequence intervention once for repeated rollout windows."""

    result = transform_reference_input(
        source_reference,
        condition_id=condition_id,
        current_frame=0,
    )
    return PreparedReferenceTransform(
        condition_id=condition_id,
        transformed_full_sequence=result.transformed_full_sequence,
        index_map=result.receipt.index_map,
        receipt=MappingProxyType(result.receipt.to_dict()),
    )


def prepared_reference_window(
    prepared: PreparedReferenceTransform,
    *,
    current_frame: int,
) -> PreparedReferenceWindow:
    """Derive the exact terminal-hold window from a prepared full-sequence transform."""

    if type(prepared) is not PreparedReferenceTransform:
        raise DevelopmentAblationError("prepared reference transform authority differs")
    sequence = prepared.transformed_full_sequence
    if (
        type(current_frame) is not int
        or not 0 <= current_frame < sequence.shape[0]
        or sequence.ndim != 2
        or sequence.shape[1] != REFERENCE_WIDTH
    ):
        raise DevelopmentAblationError("prepared reference frame is invalid")
    timeline = tuple(
        min(current_frame + offset, sequence.shape[0] - 1) for offset in range(HORIZON_STEPS)
    )
    source_indices = tuple(prepared.index_map[index] for index in timeline)
    window = np.ascontiguousarray(sequence[np.asarray(timeline, dtype=np.int64)], dtype="<f8")
    window.setflags(write=False)
    return PreparedReferenceWindow(
        values=window,
        timeline_indices=timeline,
        source_indices=source_indices,
        sha256=float64_array_sha256(window),
    )


def matched_action_delta(
    exact_action: np.ndarray, compared_action: np.ndarray
) -> dict[str, object]:
    """Return deterministic action sensitivity at one identical observed state."""

    exact = np.asarray(exact_action)
    compared = np.asarray(compared_action)
    if (
        exact.dtype.str != "<f4"
        or compared.dtype.str != "<f4"
        or exact.shape != (17,)
        or compared.shape != (17,)
        or not exact.flags.c_contiguous
        or not compared.flags.c_contiguous
        or not np.isfinite(exact).all()
        or not np.isfinite(compared).all()
    ):
        raise DevelopmentAblationError("matched actions must be finite C-order float32[17]")
    difference = compared.astype(np.float64) - exact.astype(np.float64)
    return {
        "bitwise_equal": compared.tobytes(order="C") == exact.tobytes(order="C"),
        "compared_action_sha256": array_sha256(compared),
        "exact_action_sha256": array_sha256(exact),
        "l2_physical_action": float(np.linalg.vector_norm(difference)),
        "max_abs_physical_action": float(np.max(np.abs(difference))),
    }


def summarize_matched_action_deltas(
    rows: Sequence[Mapping[str, object]],
) -> Mapping[str, Mapping[str, object]]:
    """Aggregate matched-state action differences without selecting favorable steps."""

    if not rows:
        raise DevelopmentAblationError("matched-state action rows are empty")
    grouped: dict[str, list[Mapping[str, object]]] = {condition: [] for condition in CONDITION_IDS}
    for row in rows:
        condition = row.get("condition_id")
        if condition not in grouped:
            raise DevelopmentAblationError("matched-state condition differs")
        if (
            type(row.get("step")) is not int
            or type(row.get("bitwise_equal")) is not bool
            or type(row.get("l2_physical_action")) is not float
            or type(row.get("max_abs_physical_action")) is not float
        ):
            raise DevelopmentAblationError("matched-state action row is malformed")
        grouped[str(condition)].append(row)
    result: dict[str, Mapping[str, object]] = {}
    for condition in CONDITION_IDS:
        selected = grouped[condition]
        if not selected:
            raise DevelopmentAblationError("a matched-state condition has no rows")
        l2 = [float(row["l2_physical_action"]) for row in selected]
        maximum = [float(row["max_abs_physical_action"]) for row in selected]
        if any(not math.isfinite(value) or value < 0.0 for value in (*l2, *maximum)):
            raise DevelopmentAblationError("matched-state action delta is invalid")
        result[condition] = MappingProxyType(
            {
                "bitwise_changed_steps": sum(row["bitwise_equal"] is False for row in selected),
                "evaluated_steps": len(selected),
                "maximum_l2_physical_action": max(l2),
                "maximum_max_abs_physical_action": max(maximum),
                "mean_l2_physical_action": float(np.mean(l2, dtype=np.float64)),
                "mean_max_abs_physical_action": float(np.mean(maximum, dtype=np.float64)),
            }
        )
    return MappingProxyType(result)


__all__ = [
    "CALIBRATION_BLOCKS",
    "CLAIM_CEILING",
    "CONDITION_IDS",
    "DEVELOPMENT_BLOCKS",
    "EVIDENCE_CLASS",
    "PROTECTED_EVALUATION_BLOCKS",
    "PROTOCOL_ID",
    "SCHEDULE_BEHAVIORS",
    "SCHEDULE_BOUNDARIES",
    "DevelopmentAblationError",
    "DevelopmentProtocol",
    "PreparedReferenceTransform",
    "PreparedReferenceWindow",
    "SmokeExportLineage",
    "load_bound_smoke_actor",
    "load_development_protocol",
    "matched_action_delta",
    "prepare_reference_transform",
    "prepared_reference_window",
    "summarize_matched_action_deltas",
    "validate_smoke_export",
]
