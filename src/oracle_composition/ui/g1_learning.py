"""Read-only summaries of explicitly registered, pinned G1 development runs."""

from __future__ import annotations

import hashlib
import re
import stat
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oracle_composition.adapters.gmt.training_contract import (
    TRAINING_REWARD_SCALE,
    CourseTrainerSpec,
    effective_training_contract,
)
from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.harness.contract import OracleContractError, read_json_object

REGISTRY_RELATIVE_PATH = Path("artifacts/gmt/g1_learning_registry.json")
REGISTRY_SCHEMA_VERSION = 1
REGISTRY_AUTHORITY = "registered_pinned_g1_development_runs"
MAX_REGISTERED_RUNS = 12
MAX_SINGLE_SOURCE_BYTES = 128 * 1024 * 1024
MAX_RUN_SOURCE_BYTES = 192 * 1024 * 1024
MAX_AGGREGATE_SOURCE_BYTES = 384 * 1024 * 1024

_ENTRY_FIELDS = {"run_id", "manifest_path", "manifest_sha256", "label"}
_LABELS = {"zero_residual", "final_policy"}
_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_OUTPUT_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,127}$")
_EVALUATION_FIELDS = (
    "evaluator_id",
    "claim_scope",
    "task_sha256",
    "development_gate_results",
    "development_gate_passed",
    "episode_success",
)
_LIMITATIONS = (
    "Fixed in-sample development evaluator; not protected or held-out task success.",
    "episode_success is intentionally unavailable; PASS means every development gate passed.",
    "The posture region is a flat-ground task constraint, not a physical obstacle.",
    "This view revalidates retained data; it does not rerun a policy, simulator, or trainer.",
)


class G1LearningError(ValueError):
    """A path-free refusal reason safe to expose through the local UI."""


@dataclass(frozen=True)
class _Registration:
    run_id: str
    manifest_path: Path
    manifest_sha256: str
    label: str


@dataclass(frozen=True)
class _Preflight:
    entry: _Registration
    manifest: dict[str, Any]
    encoded: bytes
    source_bytes: int


def _digest(value: object, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise G1LearningError(f"{field}_invalid")
    return value


def _read_json(path: Path, expected: str | None, field: str) -> tuple[dict[str, Any], bytes]:
    try:
        value, encoded = read_json_object(path)
    except (OSError, OracleContractError) as exc:
        raise G1LearningError(f"{field}_unavailable") from exc
    if expected is not None and hashlib.sha256(encoded).hexdigest() != expected:
        raise G1LearningError(f"{field}_sha256_mismatch")
    return value, encoded


def _load_registry(root: Path) -> tuple[list[_Registration], bytes] | None:
    resolved_root = root.resolve()
    path = resolved_root / REGISTRY_RELATIVE_PATH
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise G1LearningError("registry_unavailable") from exc
    try:
        if not path.resolve(strict=True).is_relative_to(resolved_root):
            raise G1LearningError("registry_path_escapes_project")
    except G1LearningError:
        raise
    except OSError as exc:
        raise G1LearningError("registry_unavailable") from exc
    registry, encoded = _read_json(path, None, "registry")
    if set(registry) != {"schema_version", "runs"}:
        raise G1LearningError("registry_fields_invalid")
    if type(registry["schema_version"]) is not int or registry["schema_version"] != 1:
        raise G1LearningError("registry_schema_version_invalid")
    rows = registry["runs"]
    if type(rows) is not list or len(rows) > MAX_REGISTERED_RUNS:
        raise G1LearningError("registry_run_count_invalid")
    entries = []
    for index, row in enumerate(rows):
        if type(row) is not dict or set(row) != _ENTRY_FIELDS:
            raise G1LearningError(f"registry_entry_{index}_fields_invalid")
        run_id, label, manifest_text = row["run_id"], row["label"], row["manifest_path"]
        if type(run_id) is not str or _RUN_ID.fullmatch(run_id) is None:
            raise G1LearningError(f"registry_entry_{index}_run_id_invalid")
        if type(label) is not str or label not in _LABELS:
            raise G1LearningError(f"registry_entry_{index}_label_invalid")
        if (
            type(manifest_text) is not str
            or not 1 <= len(manifest_text) <= 4096
            or "\0" in manifest_text
        ):
            raise G1LearningError(f"registry_entry_{index}_manifest_path_invalid")
        manifest_path = Path(manifest_text)
        if not manifest_path.is_absolute() or manifest_path.name != "course_run_manifest.json":
            raise G1LearningError(f"registry_entry_{index}_manifest_path_invalid")
        try:
            metadata = manifest_path.lstat()
            resolved_manifest = manifest_path.resolve(strict=True)
        except OSError as exc:
            raise G1LearningError(f"registry_entry_{index}_manifest_unavailable") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise G1LearningError(f"registry_entry_{index}_manifest_not_regular")
        entries.append(
            _Registration(
                run_id,
                resolved_manifest,
                _digest(row["manifest_sha256"], f"entry_{index}_manifest"),
                label,
            )
        )
    ids = [entry.run_id for entry in entries]
    identities = [
        (str(entry.manifest_path), entry.manifest_sha256, entry.label) for entry in entries
    ]
    if len(set(ids)) != len(ids) or len(set(identities)) != len(identities):
        raise G1LearningError("registry_duplicate_run_invalid")
    return entries, encoded


def _preflight(entry: _Registration) -> _Preflight:
    manifest, encoded = _read_json(entry.manifest_path, entry.manifest_sha256, "manifest")
    if (
        manifest.get("artifact") != "gmt_g1_course_development_run"
        or manifest.get("status") != "completed"
        or type(manifest.get("outputs")) is not dict
        or not manifest["outputs"]
    ):
        raise G1LearningError("manifest_identity_or_ledger_invalid")
    source_bytes = len(encoded)
    for name, digest in manifest["outputs"].items():
        if type(name) is not str or _OUTPUT_NAME.fullmatch(name) is None:
            raise G1LearningError("manifest_output_name_invalid")
        _digest(digest, "manifest_output")
        try:
            metadata = (entry.manifest_path.parent / name).lstat()
        except OSError as exc:
            raise G1LearningError("manifest_output_unavailable") from exc
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISREG(metadata.st_mode)
            or not 0 < metadata.st_size <= MAX_SINGLE_SOURCE_BYTES
        ):
            raise G1LearningError("manifest_output_size_or_type_invalid")
        source_bytes += metadata.st_size
    if source_bytes > MAX_RUN_SOURCE_BYTES:
        raise G1LearningError("run_source_bytes_exceed_limit")
    return _Preflight(entry, manifest, encoded, source_bytes)


def _rebuild_feedback(entry: _Registration) -> dict[str, Any]:
    try:
        # This existing builder is the single scientific validator. Lazy import
        # keeps an empty UI free of Torch and G1 asset inspection.
        from oracle_composition.feedback.g1_course import build_g1_course_feedback

        with tempfile.TemporaryDirectory(prefix="gmt-g1-ui-validation-") as temporary:
            output = Path(temporary).resolve(strict=True) / "feedback"
            result = build_g1_course_feedback(
                manifest_path=entry.manifest_path,
                expected_manifest_sha256=entry.manifest_sha256,
                label=entry.label,
                output=output,
            )
            record = result["feedback"]
            path = Path(record["path"])
            if path.parent != output or path.name != "feedback_v1.json":
                raise G1LearningError("validator_output_path_invalid")
            feedback, _ = _read_json(path, _digest(record["sha256"], "feedback"), "feedback")
            return feedback
    except G1LearningError:
        raise
    except (ImportError, KeyError, OSError, TypeError, ValueError) as exc:
        raise G1LearningError("source_validation_failed") from exc


def _summary(preflight: _Preflight) -> dict[str, object]:
    entry = preflight.entry
    feedback = _rebuild_feedback(entry)
    manifest, encoded = _read_json(entry.manifest_path, entry.manifest_sha256, "manifest")
    if encoded != preflight.encoded:
        raise G1LearningError("manifest_changed_during_validation")
    evaluation = feedback.get("evaluation")
    if (
        feedback.get("protected_evaluation") is not False
        or feedback.get("source_manifest_sha256") != entry.manifest_sha256
        or type(evaluation) is not dict
        or set(evaluation) != set(_EVALUATION_FIELDS)
    ):
        raise G1LearningError("validator_feedback_contract_invalid")

    outputs = manifest["outputs"]
    config_sha = _digest(manifest.get("input_config_sha256"), "input_config")
    if outputs.get("input_config.json") != config_sha:
        raise G1LearningError("input_config_ledger_invalid")
    config, _ = _read_json(entry.manifest_path.parent / "input_config.json", config_sha, "config")
    mode, seed, budget = config.get("mode"), config.get("seed"), config.get("training_steps")
    if mode not in {"probe", "train"} or type(seed) is not int or type(budget) is not int:
        raise G1LearningError("training_config_invalid")

    selected = manifest.get(entry.label)
    if type(selected) is not dict or type(selected.get("objective_evaluation")) is not dict:
        raise G1LearningError("selected_evaluation_invalid")
    objective = selected["objective_evaluation"]
    if any(objective.get(name) != evaluation[name] for name in _EVALUATION_FIELDS):
        raise G1LearningError("selected_evaluation_crosslink_invalid")
    gates = evaluation["development_gate_results"]
    if (
        evaluation["episode_success"] is not None
        or type(gates) is not dict
        or not gates
        or any(type(value) is not bool for value in gates.values())
        or evaluation["development_gate_passed"] is not all(gates.values())
    ):
        raise G1LearningError("development_gate_contract_invalid")

    completed = 0
    if mode == "train":
        training = manifest.get("training")
        if type(training) is not dict or training.get("completed_transitions") != budget:
            raise G1LearningError("completed_training_budget_invalid")
        completed = budget
    elif budget != 0 or manifest.get("training") is not None:
        raise G1LearningError("probe_training_record_invalid")

    frozen = manifest.get("frozen_runtime")
    trainer = frozen.get("trainer") if type(frozen) is dict else None
    raw_contract = effective_training_contract(None)
    scaled_contract = effective_training_contract(CourseTrainerSpec(TRAINING_REWARD_SCALE))
    if trainer == raw_contract:
        base_contract = trainer
        trainer_variant = "legacy_raw_training_reward"
        payload_identity = None
    elif trainer == scaled_contract:
        base_contract = trainer["base_ppo_contract"]
        trainer_variant = trainer["schema_id"]
        payload_identity = _digest(trainer["identity_sha256"], "trainer_payload_identity")
    else:
        raise G1LearningError("trainer_contract_invalid")
    trainer_bytes = canonical_json_bytes(trainer)
    if len(trainer_bytes) > 16 * 1024:
        raise G1LearningError("trainer_contract_size_invalid")
    identities = manifest.get("identities")
    if type(identities) is not dict:
        raise G1LearningError("run_identities_invalid")
    region, speed, tracking = (
        objective.get("region"),
        objective.get("speed"),
        objective.get("tracking"),
    )
    if any(type(group) is not dict for group in (region, speed, tracking)):
        raise G1LearningError("objective_metric_groups_invalid")
    return {
        "state": "available",
        "run_id": entry.run_id,
        "selected_label": entry.label,
        "full_task_development_gate_passed": evaluation["development_gate_passed"],
        "development_gates": {
            "passed": sum(gates.values()),
            "total": len(gates),
            "failed": sorted(name for name, passed in gates.items() if not passed),
        },
        "evaluator": {
            "id": evaluation["evaluator_id"],
            "claim_scope": evaluation["claim_scope"],
            "episode_success": None,
        },
        "metrics": {
            "duration_seconds": objective["duration_seconds"],
            "fall_count": objective["fall_count"],
            "finish_condition_observed": objective["finish_condition_observed"],
            "maximum_progress_m": objective["maximum_progress_m"],
            "posture_compliant_fraction": region["posture_compliant_fraction"],
            "minimum_root_height_m": region["minimum_root_height_m"],
            "mean_speed_error_m_s": speed["mean_absolute_error_m_s"],
            "inside_speed_target_deviation_m_s": speed["inside_mean_speed_target_deviation_m_s"],
            "maximum_lateral_error_m": objective["maximum_lateral_error_m"],
            "joint_position_rmse_p95_rad": tracking["joint_position_rmse_rad_p95"],
            "roll_pitch_rmse_p95_rad": tracking["roll_pitch_rmse_rad_p95"],
        },
        "training": {
            "mode": mode,
            "selected_policy": entry.label,
            "requested_transitions": budget,
            "completed_transitions": completed,
            "seed": seed,
            "algorithm": base_contract["algorithm"],
            "trainer_variant": trainer_variant,
            "producer_recorded_trainer": trainer,
            "full_trainer_contract_sha256": hashlib.sha256(trainer_bytes).hexdigest(),
            "trainer_payload_identity_sha256": payload_identity,
        },
        "receipts": {
            "manifest_sha256": entry.manifest_sha256,
            "input_config_sha256": config_sha,
            "selected_evaluation_sha256": _digest(
                outputs.get(f"{entry.label}_evaluation.json"), "selected_evaluation"
            ),
            **{
                f"{name}_sha256": _digest(identities.get(name), name)
                for name in ("task", "oracle", "reward")
            },
        },
        "limitations": list(_LIMITATIONS),
    }


def g1_learning_status(project_root: Path) -> dict[str, object]:
    """Validate registered runs sequentially and return one path-free snapshot."""

    try:
        loaded = _load_registry(project_root)
    except G1LearningError as exc:
        return {
            "state": "rejected",
            "authority": REGISTRY_AUTHORITY,
            "detail": str(exc),
            "runs": [],
        }
    if loaded is None:
        return {
            "state": "unavailable",
            "authority": REGISTRY_AUTHORITY,
            "detail": f"no local registry at {REGISTRY_RELATIVE_PATH.as_posix()}",
            "runs": [],
        }
    entries, registry_bytes = loaded
    registry = {
        "schema_version": 1,
        "sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "registered_runs": len(entries),
    }
    if not entries:
        return {
            "state": "empty",
            "authority": REGISTRY_AUTHORITY,
            "detail": "the local G1 registry contains no runs",
            "registry": registry,
            "runs": [],
        }
    planned: list[_Preflight | dict[str, object]] = []
    for entry in entries:
        try:
            planned.append(_preflight(entry))
        except G1LearningError as exc:
            planned.append(
                {
                    "state": "rejected",
                    "run_id": entry.run_id,
                    "selected_label": entry.label,
                    "detail": str(exc),
                }
            )
    aggregate_bytes = sum(item.source_bytes for item in planned if isinstance(item, _Preflight))
    if aggregate_bytes > MAX_AGGREGATE_SOURCE_BYTES:
        return {
            "state": "rejected",
            "authority": REGISTRY_AUTHORITY,
            "detail": "registry_aggregate_source_bytes_exceed_limit",
            "runs": [],
        }
    rows = []
    for item in planned:
        if isinstance(item, dict):
            rows.append(item)
            continue
        try:
            rows.append(_summary(item))
        except (G1LearningError, KeyError, TypeError, ValueError) as exc:
            reason = str(exc) if isinstance(exc, G1LearningError) else "validated_summary_invalid"
            rows.append(
                {
                    "state": "rejected",
                    "run_id": item.entry.run_id,
                    "selected_label": item.entry.label,
                    "detail": reason,
                }
            )
    accepted = sum(row["state"] == "available" for row in rows)
    rejected = len(rows) - accepted
    registry["aggregate_source_bytes"] = aggregate_bytes
    return {
        "state": "available" if not rejected else "partial" if accepted else "rejected",
        "authority": REGISTRY_AUTHORITY,
        "validation": "manual_snapshot_sequential_existing_feedback_evaluator",
        "validated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "registry": registry,
        "summary": {
            "accepted_runs": accepted,
            "rejected_runs": rejected,
            "full_task_development_gate_passes": sum(
                row.get("full_task_development_gate_passed") is True for row in rows
            ),
        },
        "runs": rows,
    }


__all__ = [
    "MAX_AGGREGATE_SOURCE_BYTES",
    "MAX_REGISTERED_RUNS",
    "REGISTRY_AUTHORITY",
    "REGISTRY_RELATIVE_PATH",
    "G1LearningError",
    "g1_learning_status",
]
