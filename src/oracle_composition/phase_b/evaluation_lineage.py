"""Fail-closed checkpoint lineage and evaluator-source authority."""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.experiments.fixed_reference import ExperimentContractError

from .contracts import COHORT_TRANSITIONS, FINAL_CHECKPOINT_RULE
from .isolation import sealed_input_lineage_value
from .persistence import (
    MAX_ACTOR_EXPORT_BYTES,
    MAX_CHECKPOINT_BYTES,
    load_full_checkpoint_metadata,
)
from .report_v2 import SeedReportFacts, validate_training_facts
from .training import COHORT_SEEDS

MAX_LINEAGE_JSON_BYTES = 2 * 1024 * 1024
EVALUATION_LINEAGE_ID = "humanoid_phase_b_evaluation_lineage/v1"
RESOURCE_CONTROL_POLICY = {
    "cpu_time": "os_rlimit_when_supported_otherwise_recorded_unsupported",
    "environment": "spawn_time_explicit_allowlist",
    "filesystem": "parent_observed_os_best_effort",
    "process_group_cleanup": "os_session_group_best_effort_with_fail_closed_receipt",
    "process_tree_rss": "parent_observed_os_best_effort",
}


@dataclass(frozen=True, slots=True)
class EvaluationLineage:
    checkpoint_sha256: str
    checkpoint_metadata: Mapping[str, object]
    execution_manifest: Mapping[str, object]
    execution_manifest_sha256: str
    training_facts: Mapping[str, object]
    persistence_receipt: Mapping[str, object]
    success_receipt: Mapping[str, object]
    job_result: Mapping[str, object]
    checkpoint_index: Mapping[str, object]
    bindings: Mapping[str, Mapping[str, object]]
    execution_manifest_bytes: bytes
    rsi_ledger_bytes: bytes

    def manifest_record(self) -> dict[str, object]:
        return {
            "bindings": {name: dict(value) for name, value in sorted(self.bindings.items())},
            "checkpoint_metadata": dict(self.checkpoint_metadata),
            "checkpoint_sha256": self.checkpoint_sha256,
            "execution_manifest_sha256": self.execution_manifest_sha256,
            "lineage_id": EVALUATION_LINEAGE_ID,
        }


def _sha(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"evaluation lineage {field} is not a SHA-256")
    return value


def _looks_like_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_success_resource_controls(value: object) -> bool:
    if type(value) is not dict or set(value) != {
        "cpu_time",
        "environment",
        "executed_modules",
        "filesystem",
        "process_group_cleanup",
        "process_tree_rss",
    }:
        return False
    cpu = value["cpu_time"]
    environment = value["environment"]
    modules = value["executed_modules"]
    cleanup = value["process_group_cleanup"]
    rss = value["process_tree_rss"]
    if type(cpu) is not dict or cpu.get("resource") != "RLIMIT_CPU":
        return False
    if cpu.get("enforcement") == "os_enforced":
        if (
            set(cpu) != {"enforcement", "hard_limit_seconds", "limit_seconds", "resource"}
            or type(cpu["hard_limit_seconds"]) is not int
            or type(cpu["limit_seconds"]) is not int
            or not 0 < cpu["limit_seconds"] <= cpu["hard_limit_seconds"]
        ):
            return False
    elif cpu.get("enforcement") == "unsupported":
        keys = {"enforcement", "limit_seconds", "resource"}
        if "error" in cpu:
            keys.add("error")
        if (
            set(cpu) != keys
            or type(cpu["limit_seconds"]) is not int
            or cpu["limit_seconds"] <= 0
            or ("error" in cpu and type(cpu["error"]) is not str)
        ):
            return False
    else:
        return False
    environment_keys = environment.get("keys") if type(environment) is dict else None
    removed = environment.get("runtime_added_keys_removed") if type(environment) is dict else None
    return (
        type(environment) is dict
        and set(environment)
        == {
            "allowlist_enforced",
            "environment_sha256",
            "keys",
            "runtime_added_keys_removed",
            "unexpected_keys",
        }
        and environment.get("allowlist_enforced") is True
        and environment.get("unexpected_keys") == []
        and _looks_like_sha256(environment.get("environment_sha256"))
        and type(environment_keys) is list
        and type(removed) is list
        and all(type(item) is str for item in (*environment_keys, *removed))
        and environment_keys == sorted(set(environment_keys))
        and removed == sorted(set(removed))
        and type(modules) is dict
        and set(modules) == {"enforcement", "final_sha256", "start_sha256"}
        and modules.get("enforcement") == "checkout_realpath_and_recorded_digest_verified"
        and _looks_like_sha256(modules.get("start_sha256"))
        and _looks_like_sha256(modules.get("final_sha256"))
        and value["filesystem"] == {"enforcement": "parent_observed_os_best_effort"}
        and type(cleanup) is dict
        and set(cleanup) == {"enforcement", "succeeded"}
        and cleanup.get("enforcement") == "os_session_group_best_effort"
        and cleanup.get("succeeded") is True
        and type(rss) is dict
        and set(rss) == {"enforcement"}
        and rss.get("enforcement")
        in {"parent_observed_os_best_effort", "unsupported_in_current_os_sandbox"}
    )


def _canonical_file(
    path: Path, *, maximum: int = MAX_LINEAGE_JSON_BYTES
) -> tuple[dict[str, object], bytes]:
    candidate = Path(path)
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise ExperimentContractError(
            f"evaluation lineage artifact is unavailable: {path}"
        ) from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ExperimentContractError("evaluation lineage artifacts must be regular files")
    if not 0 < before.st_size <= maximum:
        raise ExperimentContractError("evaluation lineage artifact exceeds its byte bound")
    encoded = candidate.read_bytes()
    after = candidate.lstat()
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in identity):
        raise ExperimentContractError("evaluation lineage artifact changed while read")
    try:
        value = json.loads(encoded)
    except (UnicodeError, ValueError) as exc:
        raise ExperimentContractError("evaluation lineage artifact is not JSON") from exc
    if type(value) is not dict or canonical_json_bytes(value) != encoded:
        raise ExperimentContractError("evaluation lineage artifact is not canonical JSON")
    return value, encoded


def _canonical_list_file(
    path: Path, *, maximum: int = MAX_LINEAGE_JSON_BYTES
) -> tuple[list[object], bytes]:
    candidate = Path(path)
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise ExperimentContractError(
            f"evaluation lineage artifact is unavailable: {path}"
        ) from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ExperimentContractError("evaluation lineage artifacts must be regular files")
    if not 0 < before.st_size <= maximum:
        raise ExperimentContractError("evaluation lineage artifact exceeds its byte bound")
    encoded = candidate.read_bytes()
    after = candidate.lstat()
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in identity):
        raise ExperimentContractError("evaluation lineage artifact changed while read")
    try:
        value = json.loads(encoded)
    except (UnicodeError, ValueError) as exc:
        raise ExperimentContractError("evaluation lineage artifact is not JSON") from exc
    if type(value) is not list or canonical_json_bytes(value) != encoded:
        raise ExperimentContractError("evaluation lineage artifact is not a canonical list")
    return value, encoded


def _bounded_file(path: Path, *, maximum: int) -> bytes:
    candidate = Path(path)
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise ExperimentContractError(
            f"evaluation lineage artifact is unavailable: {path}"
        ) from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or not 0 < before.st_size <= maximum
    ):
        raise ExperimentContractError(
            "evaluation lineage binary artifact is not bounded regular data"
        )
    encoded = candidate.read_bytes()
    after = candidate.lstat()
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in identity):
        raise ExperimentContractError("evaluation lineage artifact changed while read")
    return encoded


def _binding(path: Path, encoded: bytes) -> dict[str, object]:
    return {
        "byte_count": len(encoded),
        "path": path.name,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _indexed_binding(
    *,
    relative_path: str,
    encoded: bytes,
) -> dict[str, object]:
    return {
        "byte_count": len(encoded),
        "path": relative_path,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _validate_execution_manifest(
    *,
    manifest: Mapping[str, object],
    cohort_directory: Path,
    preflight: object,
) -> None:
    required = {
        "checkpoint_selection",
        "e003_execution_manifest_sha256",
        "evidence_class",
        "execution_manifest_schema_id",
        "ft1_run_manifest_sha256",
        "inputs",
        "prior_scientific_receipt_sha256",
        "reservation",
        "reservation_sha256",
        "resource_limits",
        "resource_control_policy",
        "runtime_source_snapshot",
        "runtime_source_snapshot_sha256",
        "sealed_input_lineage",
        "sealed_input_lineage_sha256",
        "schema_version",
        "seeds",
        "smoke",
        "test_only",
        "transitions_per_seed",
    }
    expected_current = {
        "e003_execution_manifest_sha256": preflight.e003_execution_manifest_sha256,
        "ft1_run_manifest_sha256": preflight.template_manifest_sha256,
        "prior_scientific_receipt_sha256": preflight.prior_scientific_receipt_sha256,
        "runtime_source_snapshot_sha256": preflight.source_snapshot.sha256,
    }
    reservation = manifest.get("reservation")
    limits = manifest.get("resource_limits")
    expected_reservation_inputs = {
        **dict(preflight.report_inputs),
        **expected_current,
    }
    if (
        set(manifest) != required
        or manifest.get("execution_manifest_schema_id") != "humanoid_phase_b_execution_manifest/v3"
        or manifest.get("schema_version") != 3
        or manifest.get("checkpoint_selection") != FINAL_CHECKPOINT_RULE
        or manifest.get("evidence_class") != "exploratory_fine_tuning_cycle"
        or manifest.get("seeds") != list(COHORT_SEEDS)
        or manifest.get("transitions_per_seed") != COHORT_TRANSITIONS
        or manifest.get("smoke") is not False
        or manifest.get("test_only") is not False
        or manifest.get("inputs") != dict(preflight.report_inputs)
        or manifest.get("runtime_source_snapshot") != dict(preflight.source_snapshot.value)
        or manifest.get("sealed_input_lineage")
        != sealed_input_lineage_value(preflight.sealed_inputs)
        or manifest.get("sealed_input_lineage_sha256") != preflight.sealed_input_lineage_sha256
        or any(manifest.get(field) != expected for field, expected in expected_current.items())
        or type(reservation) is not dict
        or reservation.get("accepted") is not True
        or reservation.get("mode") != "cohort"
        or reservation.get("commit") != preflight.source_snapshot.value["git"]["commit"]
        or reservation.get("inputs") != expected_reservation_inputs
        or Path(str(reservation.get("output"))).resolve() != cohort_directory
        or manifest.get("reservation_sha256")
        != hashlib.sha256(canonical_json_bytes(reservation)).hexdigest()
        or type(limits) is not dict
        or set(limits)
        != {
            "cohort_wall_seconds",
            "cpu_time_seconds",
            "free_disk_bytes",
            "job_wall_seconds",
            "output_bytes",
            "per_seed_wall_seconds",
            "rss_bytes",
            "throughput_floor_steps_s",
            "throughput_warmup_transitions",
            "throughput_window_transitions",
        }
        or any(type(value) not in {int, float} or value <= 0 for value in limits.values())
        or float(limits["job_wall_seconds"]) > 7_200.0
        or float(limits["cohort_wall_seconds"]) > float(limits["job_wall_seconds"])
        or manifest.get("resource_control_policy") != RESOURCE_CONTROL_POLICY
    ):
        raise ExperimentContractError(
            "checkpoint execution manifest differs from current preflight"
        )


def _validate_indexed_cohort(
    *,
    cohort_directory: Path,
    index: Mapping[str, object],
    job: Mapping[str, object],
    manifest_bytes: bytes,
) -> None:
    """Reopen every indexed row and recompute its production authority."""

    execution_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    entries = index.get("entries")
    outcomes = job.get("outcomes")
    if type(entries) is not list or type(outcomes) is not list:
        raise ExperimentContractError("checkpoint index cohort rows are malformed")
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
        or job.get("checkpoint_index") is not None
        or job.get("job_result_schema_id") != "humanoid_phase_b_job_result/v1"
        or job.get("schema_version") != 1
        or job.get("status") != "succeeded"
        or job.get("execution_manifest_sha256") != execution_sha256
        or len(outcomes) != len(COHORT_SEEDS)
    ):
        raise ExperimentContractError("checkpoint job result cohort authority differs")
    outcome_by_seed: dict[int, Mapping[str, object]] = {}
    for expected_seed, outcome in zip(COHORT_SEEDS, outcomes, strict=True):
        if (
            type(outcome) is not dict
            or set(outcome) != {"receipt", "seed", "status"}
            or outcome.get("seed") != expected_seed
            or outcome.get("status") != "succeeded"
            or type(outcome.get("receipt")) is not dict
        ):
            raise ExperimentContractError("checkpoint job outcome differs")
        outcome_by_seed[expected_seed] = outcome

    for expected_seed, row in zip(COHORT_SEEDS, entries, strict=True):
        if (
            type(row) is not dict
            or set(row)
            != {
                "checkpoint",
                "persistence_receipt",
                "ppo_seed",
                "strict_export",
                "success_receipt",
            }
            or row.get("ppo_seed") != expected_seed
        ):
            raise ExperimentContractError("checkpoint index entry fields differ")
        seed_relative = f"seed_{expected_seed}"
        relatives = {
            "checkpoint": f"{seed_relative}/checkpoint_seed_{expected_seed}_final.npz",
            "persistence_receipt": f"{seed_relative}/persistence_seed_{expected_seed}_v1.json",
            "strict_export": f"{seed_relative}/actor_seed_{expected_seed}_final.npz",
            "success_receipt": f"{seed_relative}/success_receipt_v2.json",
            "training_facts": f"{seed_relative}/training_facts_v1.json",
            "rsi_ledger": f"{seed_relative}/rsi_ledger_v1.json",
        }
        checkpoint_bytes = _bounded_file(
            cohort_directory / relatives["checkpoint"], maximum=MAX_CHECKPOINT_BYTES
        )
        strict_export_bytes = _bounded_file(
            cohort_directory / relatives["strict_export"], maximum=MAX_ACTOR_EXPORT_BYTES
        )
        persistence, persistence_bytes = _canonical_file(
            cohort_directory / relatives["persistence_receipt"], maximum=64 * 1024
        )
        success, success_bytes = _canonical_file(
            cohort_directory / relatives["success_receipt"], maximum=64 * 1024
        )
        training, training_bytes = _canonical_file(cohort_directory / relatives["training_facts"])
        _ledger, ledger_bytes = _canonical_list_file(cohort_directory / relatives["rsi_ledger"])
        observed_index_bindings = {
            "checkpoint": _indexed_binding(
                relative_path=relatives["checkpoint"],
                encoded=checkpoint_bytes,
            ),
            "persistence_receipt": _indexed_binding(
                relative_path=relatives["persistence_receipt"],
                encoded=persistence_bytes,
            ),
            "strict_export": _indexed_binding(
                relative_path=relatives["strict_export"],
                encoded=strict_export_bytes,
            ),
            "success_receipt": _indexed_binding(
                relative_path=relatives["success_receipt"],
                encoded=success_bytes,
            ),
        }
        if any(row[name] != binding for name, binding in observed_index_bindings.items()):
            raise ExperimentContractError("checkpoint index artifact binding differs")
        checkpoint_sha256 = observed_index_bindings["checkpoint"]["sha256"]
        metadata = load_full_checkpoint_metadata(
            cohort_directory / relatives["checkpoint"],
            expected_sha256=str(checkpoint_sha256),
        )
        production = {
            "evidence_class": "exploratory_fine_tuning_cycle",
            "planned_transitions": COHORT_TRANSITIONS,
            "ppo_seed": expected_seed,
            "promotable": True,
            "smoke": False,
            "test_only": False,
            "transitions": COHORT_TRANSITIONS,
        }
        if (
            any(metadata.get(name) != value for name, value in production.items())
            or metadata.get("rollouts") != 128
            or metadata.get("execution_manifest_sha256") != execution_sha256
            or metadata.get("training_facts_sha256") != hashlib.sha256(training_bytes).hexdigest()
        ):
            raise ExperimentContractError("indexed checkpoint metadata is not a cohort outcome")
        if (
            set(persistence)
            != {
                "checkpoint",
                "checkpoint_reload_bitwise_deterministic",
                "checkpoint_to_export_bitwise_equivalent",
                "evidence_class",
                "execution_manifest_sha256",
                "final_transition_only",
                "fixture_action_sha256",
                "persistence_receipt_id",
                "planned_transitions",
                "ppo_seed",
                "promotable",
                "schema_version",
                "smoke",
                "strict_export",
                "test_only",
                "training_facts_sha256",
                "transitions",
            }
            or persistence.get("persistence_receipt_id") != "humanoid_phase_b_final_persistence/v1"
            or persistence.get("schema_version") != 1
            or any(persistence.get(name) != value for name, value in production.items())
            or persistence.get("execution_manifest_sha256") != execution_sha256
            or persistence.get("training_facts_sha256")
            != hashlib.sha256(training_bytes).hexdigest()
            or persistence.get("final_transition_only") is not True
            or persistence.get("checkpoint_reload_bitwise_deterministic") is not True
            or persistence.get("checkpoint_to_export_bitwise_equivalent") is not True
            or _sha(persistence.get("fixture_action_sha256"), "fixture action")
            != persistence.get("fixture_action_sha256")
            or persistence.get("checkpoint")
            != {
                "byte_count": len(checkpoint_bytes),
                "filename": Path(relatives["checkpoint"]).name,
                "sha256": checkpoint_sha256,
            }
            or persistence.get("strict_export")
            != {
                "byte_count": len(strict_export_bytes),
                "filename": Path(relatives["strict_export"]).name,
                "sha256": hashlib.sha256(strict_export_bytes).hexdigest(),
            }
        ):
            raise ExperimentContractError("indexed persistence receipt differs")
        artifacts = success.get("artifacts")
        expected_artifacts = {
            "persistence": {
                "byte_count": len(persistence_bytes),
                "filename": Path(relatives["persistence_receipt"]).name,
                "sha256": hashlib.sha256(persistence_bytes).hexdigest(),
            },
            "rsi_ledger": {
                "byte_count": len(ledger_bytes),
                "filename": Path(relatives["rsi_ledger"]).name,
                "sha256": hashlib.sha256(ledger_bytes).hexdigest(),
            },
            "training_facts": {
                "byte_count": len(training_bytes),
                "filename": Path(relatives["training_facts"]).name,
                "sha256": hashlib.sha256(training_bytes).hexdigest(),
            },
        }
        success_production = {
            name: value for name, value in production.items() if name != "transitions"
        }
        if (
            set(success)
            != {
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
            or success.get("success_receipt_id") != "humanoid_phase_b_seed_success/v2"
            or success.get("schema_version") != 2
            or success.get("outcome") != "success"
            or success.get("status") != "succeeded"
            or success.get("failure_receipt_present") is not False
            or any(success.get(name) != value for name, value in success_production.items())
            or success.get("execution_manifest_sha256") != execution_sha256
            or artifacts != expected_artifacts
            or not _valid_success_resource_controls(success.get("resource_controls"))
            or success.get("worker_cleanup")
            != {"attempted": True, "error": None, "succeeded": True}
        ):
            raise ExperimentContractError("indexed seed success authority differs")
        outcome_receipt = outcome_by_seed[expected_seed]["receipt"]
        if outcome_receipt != {
            "byte_count": len(success_bytes),
            "filename": Path(relatives["success_receipt"]).name,
            "sha256": hashlib.sha256(success_bytes).hexdigest(),
        }:
            raise ExperimentContractError("checkpoint job result receipt binding differs")
        step_zero = training.get("step_zero_comparator")
        if type(step_zero) is not dict or step_zero.get("bitwise_equal") is not True:
            raise ExperimentContractError("indexed training facts omit the step-zero comparator")
        facts = SeedReportFacts(
            ppo_seed=expected_seed,
            checkpoint_sha256=str(checkpoint_sha256),
            strict_export_sha256=hashlib.sha256(strict_export_bytes).hexdigest(),
            training=training,
            step_zero_comparator=step_zero,
            execution_manifest_bytes=manifest_bytes,
            rsi_ledger_bytes=ledger_bytes,
        )
        try:
            validate_training_facts(facts)
        except ValueError as exc:
            raise ExperimentContractError("indexed training facts differ") from exc


def validate_evaluation_lineage(
    *,
    checkpoint_path: Path,
    preflight: object,
) -> EvaluationLineage:
    """Validate a complete production checkpoint chain before model construction."""

    checkpoint = Path(checkpoint_path).resolve(strict=True)
    name = checkpoint.name
    prefix, suffix = "checkpoint_seed_", "_final.npz"
    if not name.startswith(prefix) or not name.endswith(suffix):
        raise ExperimentContractError("evaluation checkpoint filename is not canonical")
    seed_text = name[len(prefix) : -len(suffix)]
    if not seed_text.isdigit():
        raise ExperimentContractError("evaluation checkpoint seed is invalid")
    seed = int(seed_text)
    seed_directory = checkpoint.parent
    cohort_directory = seed_directory.parent
    paths = {
        "persistence_receipt": seed_directory / f"persistence_seed_{seed}_v1.json",
        "training_facts": seed_directory / "training_facts_v1.json",
        "success_receipt": seed_directory / "success_receipt_v2.json",
        "execution_manifest": cohort_directory / "execution_manifest_v3.json",
        "job_result": cohort_directory / "job_result_v1.json",
        "checkpoint_index": cohort_directory / "checkpoint_index_v1.json",
    }
    values: dict[str, dict[str, object]] = {}
    encodings: dict[str, bytes] = {}
    for key, path in paths.items():
        values[key], encodings[key] = _canonical_file(path)
    _rsi_ledger, rsi_ledger_bytes = _canonical_list_file(seed_directory / "rsi_ledger_v1.json")
    persistence = values["persistence_receipt"]
    checkpoint_binding = persistence.get("checkpoint")
    if type(checkpoint_binding) is not dict or checkpoint_binding.get("filename") != name:
        raise ExperimentContractError("checkpoint persistence binding differs")
    checkpoint_sha256 = _sha(checkpoint_binding.get("sha256"), "checkpoint")
    metadata = load_full_checkpoint_metadata(checkpoint, expected_sha256=checkpoint_sha256)
    execution_sha256 = hashlib.sha256(encodings["execution_manifest"]).hexdigest()
    training_sha256 = hashlib.sha256(encodings["training_facts"]).hexdigest()
    expected_production = {
        "evidence_class": "exploratory_fine_tuning_cycle",
        "planned_transitions": COHORT_TRANSITIONS,
        "ppo_seed": seed,
        "promotable": True,
        "smoke": False,
        "test_only": False,
        "transitions": COHORT_TRANSITIONS,
    }
    if any(metadata.get(field) != expected for field, expected in expected_production.items()):
        raise ExperimentContractError(
            "evaluation checkpoint is not a promotable full-budget outcome"
        )
    if (
        metadata.get("rollouts") != 128
        or metadata.get("execution_manifest_sha256") != execution_sha256
        or metadata.get("training_facts_sha256") != training_sha256
    ):
        raise ExperimentContractError("checkpoint metadata lineage differs")
    if any(persistence.get(field) != expected for field, expected in expected_production.items()):
        raise ExperimentContractError("persistence receipt is not a production outcome")
    if (
        persistence.get("execution_manifest_sha256") != execution_sha256
        or persistence.get("training_facts_sha256") != training_sha256
        or persistence.get("final_transition_only") is not True
        or persistence.get("checkpoint") != checkpoint_binding
    ):
        raise ExperimentContractError("persistence receipt lineage differs")

    training = values["training_facts"]
    if (
        training.get("ppo_seed") != seed
        or training.get("execution_manifest_sha256") != execution_sha256
        or training.get("planned_transitions") != COHORT_TRANSITIONS
        or training.get("observed_transitions") != COHORT_TRANSITIONS
        or training.get("rollouts") != 128
        or training.get("evidence_class") != "exploratory_fine_tuning_cycle"
        or training.get("promotable") is not True
        or training.get("smoke") is not False
    ):
        raise ExperimentContractError("training facts are incomplete or non-production")

    manifest = values["execution_manifest"]
    _validate_execution_manifest(
        manifest=manifest,
        cohort_directory=cohort_directory,
        preflight=preflight,
    )

    success = values["success_receipt"]
    success_binding = _binding(paths["success_receipt"], encodings["success_receipt"])
    if (
        set(success)
        != {
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
        or success.get("success_receipt_id") != "humanoid_phase_b_seed_success/v2"
        or success.get("schema_version") != 2
        or success.get("outcome") != "success"
        or success.get("status") != "succeeded"
        or success.get("ppo_seed") != seed
        or success.get("planned_transitions") != COHORT_TRANSITIONS
        or success.get("promotable") is not True
        or success.get("smoke") is not False
        or success.get("test_only") is not False
        or success.get("execution_manifest_sha256") != execution_sha256
        or not _valid_success_resource_controls(success.get("resource_controls"))
        or success.get("worker_cleanup") != {"attempted": True, "error": None, "succeeded": True}
    ):
        raise ExperimentContractError("seed success receipt is missing production authority")
    artifacts = success.get("artifacts")
    persistence_artifact = artifacts.get("persistence") if type(artifacts) is dict else None
    training_artifact = artifacts.get("training_facts") if type(artifacts) is dict else None
    if (
        type(artifacts) is not dict
        or set(artifacts) != {"persistence", "rsi_ledger", "training_facts"}
        or type(persistence_artifact) is not dict
        or type(training_artifact) is not dict
        or persistence_artifact.get("sha256")
        != hashlib.sha256(encodings["persistence_receipt"]).hexdigest()
        or training_artifact.get("sha256") != training_sha256
        or type(artifacts.get("rsi_ledger")) is not dict
        or artifacts["rsi_ledger"].get("sha256") != hashlib.sha256(rsi_ledger_bytes).hexdigest()
    ):
        raise ExperimentContractError("seed success receipt artifact chain differs")

    job = values["job_result"]
    job_outcomes = job.get("outcomes")
    if (
        job.get("job_result_schema_id") != "humanoid_phase_b_job_result/v1"
        or job.get("status") != "succeeded"
        or job.get("execution_manifest_sha256") != execution_sha256
        or type(job_outcomes) is not list
        or not any(
            type(row) is dict
            and row.get("seed") == seed
            and row.get("status") == "succeeded"
            and row.get("receipt") == success_binding
            for row in job_outcomes
        )
    ):
        raise ExperimentContractError("job result does not authorize this successful seed")

    index = values["checkpoint_index"]
    index_entries = index.get("entries")
    if (
        set(index)
        != {
            "cohort_seeds",
            "checkpoint_count",
            "checkpoint_index_schema_id",
            "entries",
            "execution_manifest_sha256",
            "job_result_sha256",
            "schema_version",
            "selection_rule",
            "summary",
        }
        or index.get("checkpoint_index_schema_id") != "humanoid_phase_b_checkpoint_index/v1"
        or index.get("schema_version") != 1
        or index.get("selection_rule") != "final_transition_only_no_replacement"
        or index.get("checkpoint_count") != 5
        or index.get("execution_manifest_sha256") != execution_sha256
        or index.get("job_result_sha256") != hashlib.sha256(encodings["job_result"]).hexdigest()
        or index.get("cohort_seeds") != list(COHORT_SEEDS)
        or type(index_entries) is not list
        or len(index_entries) != 5
        or [row.get("ppo_seed") for row in index_entries if type(row) is dict] != list(COHORT_SEEDS)
        or index.get("summary")
        != {"full_budget_success_count": 5, "promotable_checkpoint_count": 5}
    ):
        raise ExperimentContractError("checkpoint index cohort authority differs")
    expected_relative = checkpoint.relative_to(cohort_directory).as_posix()
    if not any(
        type(row) is dict
        and row.get("ppo_seed") == seed
        and type(row.get("checkpoint")) is dict
        and row["checkpoint"].get("path") == expected_relative
        and row["checkpoint"].get("sha256") == checkpoint_sha256
        and type(row.get("success_receipt")) is dict
        and row["success_receipt"].get("sha256") == success_binding["sha256"]
        for row in index.get("entries", [])
    ):
        raise ExperimentContractError("checkpoint index omits the selected checkpoint chain")

    _validate_indexed_cohort(
        cohort_directory=cohort_directory,
        index=index,
        job=job,
        manifest_bytes=encodings["execution_manifest"],
    )

    bindings = {
        name: MappingProxyType(_binding(path, encodings[name])) for name, path in paths.items()
    }
    bindings["rsi_ledger"] = MappingProxyType(
        _binding(seed_directory / "rsi_ledger_v1.json", rsi_ledger_bytes)
    )
    return EvaluationLineage(
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_metadata=MappingProxyType(dict(metadata)),
        execution_manifest=MappingProxyType(dict(manifest)),
        execution_manifest_sha256=execution_sha256,
        training_facts=MappingProxyType(dict(training)),
        persistence_receipt=MappingProxyType(dict(persistence)),
        success_receipt=MappingProxyType(dict(success)),
        job_result=MappingProxyType(dict(job)),
        checkpoint_index=MappingProxyType(dict(index)),
        bindings=MappingProxyType(bindings),
        execution_manifest_bytes=encodings["execution_manifest"],
        rsi_ledger_bytes=rsi_ledger_bytes,
    )


def evaluator_source_identity(repository_root: Path) -> dict[str, object]:
    """Bind actual imported evaluator modules and reject cross-checkout imports."""

    from . import (
        calibration,
        evaluation,
        evaluation_lineage,
        evaluation_supervision,
        persistence,
        policy,
        protected_metrics,
        reference_runtime,
        report_v2,
    )

    root = Path(repository_root).resolve(strict=True)
    source_root = (root / "src/oracle_composition").resolve(strict=True)
    modules = (
        calibration,
        evaluation,
        evaluation_lineage,
        evaluation_supervision,
        persistence,
        policy,
        protected_metrics,
        reference_runtime,
        report_v2,
    )
    records = []
    for module in modules:
        raw = getattr(module, "__file__", None)
        if type(raw) is not str:
            raise ExperimentContractError("an executed evaluator module has no source path")
        path = Path(raw).resolve(strict=True)
        try:
            relative = path.relative_to(source_root)
        except ValueError as exc:
            raise ExperimentContractError("cross-checkout evaluator import rejected") from exc
        if path.is_symlink() or not path.is_file():
            raise ExperimentContractError("evaluator source path is not a regular file")
        before = path.stat()
        encoded = path.read_bytes()
        after = path.stat()
        if any(
            getattr(before, field) != getattr(after, field)
            for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ):
            raise ExperimentContractError("evaluator source changed while hashed")
        records.append(
            {
                "module": module.__name__,
                "realpath": path.as_posix(),
                "repository_relative_path": f"src/oracle_composition/{relative.as_posix()}",
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    core = {
        "evaluator_source_id": "humanoid_phase_b_executed_evaluator_sources/v1",
        "modules": sorted(records, key=lambda row: str(row["module"])),
    }
    return {**core, "sha256": hashlib.sha256(canonical_json_bytes(core)).hexdigest()}


__all__ = [
    "EVALUATION_LINEAGE_ID",
    "EvaluationLineage",
    "evaluator_source_identity",
    "validate_evaluation_lineage",
]
