"""Validate development reference ablations and route bounded feedback."""

from __future__ import annotations

import hashlib
import json
import math
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import (
    array_sha256,
    canonical_json_bytes,
)
from oracle_composition.development.reference_ablation import (
    CLAIM_CEILING,
    DEVELOPMENT_BLOCKS,
    EVIDENCE_CLASS,
    PROTOCOL_ID,
    DevelopmentProtocol,
    PreparedReferenceTransform,
    SmokeExportLineage,
    load_development_protocol,
    prepare_reference_transform,
    prepared_reference_window,
    summarize_matched_action_deltas,
    validate_development_corpus,
    validate_smoke_export,
)
from oracle_composition.experiments.reference_input_transforms import CONDITION_IDS
from oracle_composition.phase_b.protected_metrics import (
    protected_trace_sha256,
    recompute_protected_episode,
    select_evaluator_nearest_phase,
)
from oracle_composition.phase_b.reference_runtime import load_v2_reference_clip

from .evidence import FeedbackEvidenceError

RESULT_SCHEMA_ID = "humanoid_phase_b_t1_development_reference_ablation_result/v1"
MANIFEST_SCHEMA_ID = "humanoid_phase_b_t1_development_reference_ablation_manifest/v1"
TRACE_SCHEMA_ID = "humanoid_phase_b_t1_development_reference_ablation_trace/v1"
MAX_RESULT_BYTES = 4 * 1024**2
MAX_MANIFEST_BYTES = 16 * 1024**2
MAX_TRACE_BYTES = 64 * 1024**2
MAX_TRACE_SET_BYTES = 512 * 1024**2
MAX_JSON_NODES = 2_000_000

_RESULT_KEYS = frozenset(
    {
        "actor_export_sha256",
        "arms",
        "claim_ceiling",
        "completed_arm_count",
        "development_only",
        "evidence_class",
        "held_out_inputs_used",
        "manifest",
        "matched_state_action_deltas",
        "planned_arm_count",
        "promotable",
        "result_schema_id",
        "runtime_source_snapshot_sha256",
        "schema_version",
        "status",
        "task_success",
        "trace_artifacts",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "actor_export",
        "actor_visible_factor",
        "calibration_inputs",
        "claim_ceiling",
        "conditions",
        "corpus_bindings",
        "development_blocks",
        "evidence_class",
        "held_out_inputs_used",
        "manifest_schema_id",
        "objective_reference",
        "package_import_path",
        "policy_seed",
        "promotable",
        "protocol",
        "runtime_source_snapshot",
        "runtime_source_snapshot_sha256",
        "schema_version",
        "smoke_lineage",
        "task_success_scoring",
    }
)
_TRACE_KEYS = frozenset(
    {
        "actor_export_sha256",
        "block",
        "claim_ceiling",
        "condition_id",
        "development_only",
        "evidence_class",
        "matched_state_action_deltas",
        "objective_metric_kernel",
        "objective_reference",
        "objective_steps",
        "objective_steps_sha256",
        "policy_inputs",
        "policy_inputs_sha256",
        "policy_reference_factor",
        "policy_seed",
        "promotable",
        "protocol_id",
        "reference_identities",
        "reference_transform_precomputations",
        "schema_version",
        "switch_records",
        "task_success",
        "trace_schema_id",
    }
)
_ARM_KEYS = frozenset(
    {
        "action_bounds_ok",
        "block",
        "condition_id",
        "fall",
        "forbidden_contact_count",
        "observed_steps",
        "resynchronization_records",
        "segment_errors",
        "settled_state_normalized_error",
        "settle_latency_steps",
        "six_tracking_errors",
        "task_success",
        "time_to_first_failure_steps",
        "trace",
        "transition_window_error",
    }
)
_POLICY_INPUT_KEYS = frozenset(
    {
        "action_sha256",
        "actor_input_sha256",
        "behavior",
        "condition_id",
        "observation_sha256",
        "phase",
        "policy_window_float32_sha256",
        "policy_window_float64_sha256",
        "step",
        "window_source_indices",
        "window_timeline_indices",
    }
)
_MATCHED_ROW_KEYS = frozenset(
    {
        "behavior",
        "bitwise_equal",
        "compared_action_sha256",
        "condition_id",
        "exact_action_sha256",
        "l2_physical_action",
        "max_abs_physical_action",
        "phase",
        "step",
    }
)


@dataclass(frozen=True, slots=True)
class ValidatedDevelopmentEvidence:
    """One fully cross-bound development result and its recomputed summaries."""

    result: Mapping[str, object]
    result_sha256: str
    manifest: Mapping[str, object]
    manifest_sha256: str
    trace_set_sha256: str
    arms: tuple[Mapping[str, object], ...]


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, *, field: str) -> str:
    if not _is_sha256(value):
        raise FeedbackEvidenceError(f"{field} must be a lowercase SHA-256")
    return str(value)


def _count_nodes(value: object) -> int:
    count = 0
    pending = [value]
    while pending:
        current = pending.pop()
        count += 1
        if count > MAX_JSON_NODES:
            raise FeedbackEvidenceError("development JSON exceeds its node bound")
        if type(current) is dict:
            pending.extend(current.keys())
            pending.extend(current.values())
        elif type(current) is list:
            pending.extend(current)
    return count


def _read_canonical_object(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> tuple[dict[str, object], bytes]:
    def reject_constant(token: str) -> None:
        raise FeedbackEvidenceError(f"{label} contains non-finite JSON constant {token}")

    candidate = Path(path)
    try:
        state = candidate.stat(follow_symlinks=False)
    except OSError as exc:
        raise FeedbackEvidenceError(f"cannot inspect {label}: {exc}") from exc
    if (
        candidate.is_symlink()
        or not stat.S_ISREG(state.st_mode)
        or state.st_nlink != 1
        or not 1 <= state.st_size <= maximum_bytes
    ):
        raise FeedbackEvidenceError(f"{label} must be one bounded regular file")
    try:
        encoded = candidate.read_bytes()
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise FeedbackEvidenceError(f"{label} is not strict JSON: {exc}") from exc
    if type(value) is not dict or canonical_json_bytes(value) != encoded:
        raise FeedbackEvidenceError(f"{label} must be one canonical JSON object")
    _count_nodes(value)
    return value, encoded


def _artifact_binding(value: object, *, field: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != {"byte_count", "filename", "sha256"}:
        raise FeedbackEvidenceError(f"{field} binding fields differ")
    filename = value.get("filename")
    byte_count = value.get("byte_count")
    if (
        type(filename) is not str
        or filename in {"", ".", ".."}
        or Path(filename).name != filename
        or type(byte_count) is not int
        or byte_count <= 0
    ):
        raise FeedbackEvidenceError(f"{field} binding is malformed")
    _require_sha256(value.get("sha256"), field=f"{field} SHA-256")
    return value


def _read_bound_object(
    directory: Path,
    binding: object,
    *,
    maximum_bytes: int,
    label: str,
) -> tuple[dict[str, object], bytes, Path]:
    record = _artifact_binding(binding, field=label)
    path = directory / str(record["filename"])
    value, encoded = _read_canonical_object(path, maximum_bytes=maximum_bytes, label=label)
    if len(encoded) != record["byte_count"] or _sha256(encoded) != record["sha256"]:
        raise FeedbackEvidenceError(f"{label} artifact binding differs")
    return value, encoded, path


def _same_json(left: object, right: object) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _validate_manifest(
    manifest: Mapping[str, object],
    *,
    result: Mapping[str, object],
    protocol: DevelopmentProtocol,
    smoke: SmokeExportLineage,
    corpus_bindings: Mapping[str, Mapping[str, object]],
) -> None:
    expected_scalars = {
        "actor_visible_factor": "reference_window_only",
        "calibration_inputs": None,
        "claim_ceiling": CLAIM_CEILING,
        "conditions": list(CONDITION_IDS),
        "development_blocks": list(DEVELOPMENT_BLOCKS),
        "evidence_class": EVIDENCE_CLASS,
        "held_out_inputs_used": False,
        "manifest_schema_id": MANIFEST_SCHEMA_ID,
        "objective_reference": "untransformed_exact_reference_rows",
        "policy_seed": 121901,
        "promotable": False,
        "schema_version": 1,
        "task_success_scoring": False,
    }
    if set(manifest) != _MANIFEST_KEYS or any(
        manifest.get(name) != expected for name, expected in expected_scalars.items()
    ):
        raise FeedbackEvidenceError("development manifest fields or fixed boundary differ")
    package_path = manifest.get("package_import_path")
    if (
        type(package_path) is not str
        or not Path(package_path).is_absolute()
        or Path(package_path).parts[-3:] != ("src", "oracle_composition", "__init__.py")
    ):
        raise FeedbackEvidenceError("development package import path is malformed")
    expected_actor = {"byte_count": smoke.actor_byte_count, "sha256": smoke.actor_sha256}
    if manifest.get("actor_export") != expected_actor:
        raise FeedbackEvidenceError("development actor identity differs from smoke export")
    expected_protocol = {
        "byte_count": protocol.byte_count,
        "path": str(protocol.path),
        "sha256": protocol.sha256,
    }
    if manifest.get("protocol") != expected_protocol:
        raise FeedbackEvidenceError("development protocol binding differs")
    expected_smoke = {
        "actor_sha256": smoke.actor_sha256,
        "bindings": {name: dict(value) for name, value in smoke.bindings.items()},
        "checkpoint_sha256": smoke.checkpoint_sha256,
        "execution_manifest_sha256": smoke.execution_manifest_sha256,
        "reference_sha256": smoke.report_inputs["reference_sha256"],
        "run_directory": str(smoke.run_directory),
        "sealed_input_lineage_sha256": smoke.report_inputs["sealed_input_lineage_sha256"],
        "training_facts_sha256": smoke.training_facts_sha256,
    }
    if not _same_json(manifest.get("smoke_lineage"), expected_smoke):
        raise FeedbackEvidenceError("development smoke lineage differs")
    expected_corpus = {name: dict(value) for name, value in sorted(corpus_bindings.items())}
    if not _same_json(manifest.get("corpus_bindings"), expected_corpus):
        raise FeedbackEvidenceError("development corpus bindings differ from sealed bytes")
    snapshot = manifest.get("runtime_source_snapshot")
    if type(snapshot) is not dict or set(snapshot) != {
        "authority_identities",
        "device",
        "git",
        "platform",
        "source_sha256",
        "versions",
    }:
        raise FeedbackEvidenceError("development runtime source snapshot is malformed")
    git = snapshot.get("git")
    sources = snapshot.get("source_sha256")
    platform = snapshot.get("platform")
    versions = snapshot.get("versions")
    if (
        snapshot.get("device") != "cpu"
        or type(snapshot.get("authority_identities")) is not dict
        or type(git) is not dict
        or set(git) != {"clean", "commit"}
        or git.get("clean") is not True
        or type(git.get("commit")) is not str
        or len(git["commit"]) != 40
        or any(character not in "0123456789abcdef" for character in git["commit"])
        or type(platform) is not dict
        or set(platform) != {"machine", "python", "system"}
        or any(type(value) is not str or not value for value in platform.values())
        or type(versions) is not dict
        or set(versions) != {"gymnasium", "mujoco", "numpy", "stable_baselines3", "torch"}
        or any(type(value) is not str or not value for value in versions.values())
        or type(sources) is not dict
        or "src/oracle_composition/__init__.py" not in sources
        or any(
            type(name) is not str
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not _is_sha256(digest)
            for name, digest in sources.items()
        )
    ):
        raise FeedbackEvidenceError("development runtime source identities are malformed")
    snapshot_sha = _sha256(canonical_json_bytes(snapshot))
    if (
        manifest.get("runtime_source_snapshot_sha256") != snapshot_sha
        or result.get("runtime_source_snapshot_sha256") != snapshot_sha
    ):
        raise FeedbackEvidenceError("development runtime source snapshot digest differs")


def _reference_inputs(
    corpus_root: Path,
    block: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, object]],
    dict[str, dict[str, PreparedReferenceTransform]],
]:
    references: dict[str, np.ndarray] = {}
    identities: dict[str, dict[str, object]] = {}
    transforms: dict[str, dict[str, PreparedReferenceTransform]] = {}
    for behavior in ("expert", "simple"):
        clip = load_v2_reference_clip(corpus_root, block=block, behavior=behavior)
        references[behavior] = clip.reference_rows
        identities[behavior] = {
            "bundle_sha256": clip.bundle_sha256,
            "payload_sha256": clip.payload_sha256,
            "reference_identity_sha256": clip.reference_identity_sha256,
        }
        transforms[behavior] = {
            condition: prepare_reference_transform(
                clip.reference_rows,
                condition_id=condition,
            )
            for condition in CONDITION_IDS
        }
    return references, identities, transforms


def _switches(
    trace: Mapping[str, object],
    references: Mapping[str, np.ndarray],
) -> tuple[dict[str, object], dict[str, object]]:
    rows = trace.get("switch_records")
    steps = trace.get("objective_steps")
    if type(rows) is not list or len(rows) != 2 or type(steps) is not list:
        raise FeedbackEvidenceError("development switch evidence is incomplete")
    expected_rows = []
    for boundary, source, target in (
        (300, "expert", "simple"),
        (600, "simple", "expert"),
    ):
        source_state = steps[boundary - 1]
        if type(source_state) is not dict or type(source_state.get("state")) is not dict:
            raise FeedbackEvidenceError("development switch source state is malformed")
        selected = select_evaluator_nearest_phase(
            state=source_state["state"],
            target_rows=references[target],
            task_step=boundary,
            source_behavior=source,
            target_behavior=target,
            reason="development_preregistered_direct_switch",
        )
        expected_rows.append(
            {
                "boundary": boundary,
                "from_behavior": source,
                "selected_normalized_errors": selected["selected_normalized_errors"],
                "selected_phase": selected["selected_phase"],
                "selected_score": selected["selected_score"],
                "to_behavior": target,
            }
        )
    if rows != expected_rows:
        raise FeedbackEvidenceError("development phase-transfer records do not recompute")
    return rows[0], rows[1]


def _validate_policy_and_reference_chain(
    trace: Mapping[str, object],
    *,
    condition: str,
    references: Mapping[str, np.ndarray],
    transforms: Mapping[str, Mapping[str, PreparedReferenceTransform]],
    switches: Sequence[Mapping[str, object]],
) -> tuple[tuple[str, int, str], ...]:
    policy_rows = trace.get("policy_inputs")
    objective_steps = trace.get("objective_steps")
    if (
        type(policy_rows) is not list
        or len(policy_rows) != 1_000
        or type(objective_steps) is not list
        or len(objective_steps) != 1_000
    ):
        raise FeedbackEvidenceError("development trace horizon differs")
    behavior = "expert"
    phase = 0
    switch_map = {int(row["boundary"]): row for row in switches}
    action_identities = []
    for step, (policy_row, objective_step) in enumerate(
        zip(policy_rows, objective_steps, strict=True)
    ):
        if step in switch_map:
            switched = switch_map[step]
            behavior = str(switched["to_behavior"])
            phase = int(switched["selected_phase"])
        if type(policy_row) is not dict or set(policy_row) != _POLICY_INPUT_KEYS:
            raise FeedbackEvidenceError("development policy-input fields differ")
        if (
            policy_row.get("step") != step
            or policy_row.get("condition_id") != condition
            or policy_row.get("behavior") != behavior
            or policy_row.get("phase") != phase
        ):
            raise FeedbackEvidenceError("development policy-input schedule differs")
        for name in (
            "action_sha256",
            "actor_input_sha256",
            "observation_sha256",
            "policy_window_float32_sha256",
            "policy_window_float64_sha256",
        ):
            _require_sha256(policy_row.get(name), field=f"development {name}")
        prepared = transforms[behavior][condition]
        window = prepared_reference_window(prepared, current_frame=phase)
        window_f32 = np.ascontiguousarray(window.values, dtype="<f4")
        if (
            policy_row.get("window_timeline_indices") != list(window.timeline_indices)
            or policy_row.get("window_source_indices") != list(window.source_indices)
            or policy_row.get("policy_window_float64_sha256") != window.sha256
            or policy_row.get("policy_window_float32_sha256") != array_sha256(window_f32)
        ):
            raise FeedbackEvidenceError("development policy reference window differs")
        target_phase = min(phase + 1, 1_000)
        if type(objective_step) is not dict or type(objective_step.get("reference")) is not dict:
            raise FeedbackEvidenceError("development objective reference is malformed")
        reference = objective_step["reference"]
        expected_reference = references[behavior][target_phase]
        if (
            reference.get("behavior") != behavior
            or reference.get("index") != target_phase
            or reference.get("sha256") != array_sha256(expected_reference)
        ):
            raise FeedbackEvidenceError("development objective reference differs from corpus")
        raw_action = objective_step.get("action")
        if type(raw_action) is not list or len(raw_action) != 17:
            raise FeedbackEvidenceError("development objective action is malformed")
        try:
            objective_action = np.ascontiguousarray(raw_action, dtype="<f4")
        except (TypeError, ValueError, OverflowError) as exc:
            raise FeedbackEvidenceError("development objective action is malformed") from exc
        if not np.isfinite(objective_action).all() or array_sha256(
            objective_action
        ) != policy_row.get("action_sha256"):
            raise FeedbackEvidenceError("development policy and objective actions differ")
        action_identities.append((behavior, phase, str(policy_row["action_sha256"])))
        phase = target_phase
    return tuple(action_identities)


def _validate_matched_rows(
    rows: object,
    *,
    exact_actions: Sequence[tuple[str, int, str]],
) -> Mapping[str, Mapping[str, object]]:
    if type(rows) is not list or len(rows) != 1_000 * len(CONDITION_IDS):
        raise FeedbackEvidenceError("development matched-state action rows are incomplete")
    for index, row in enumerate(rows):
        step, condition_index = divmod(index, len(CONDITION_IDS))
        condition = CONDITION_IDS[condition_index]
        behavior, phase, exact_sha = exact_actions[step]
        if type(row) is not dict or set(row) != _MATCHED_ROW_KEYS:
            raise FeedbackEvidenceError("development matched-state row fields differ")
        if (
            row.get("step") != step
            or row.get("condition_id") != condition
            or row.get("behavior") != behavior
            or row.get("phase") != phase
            or row.get("exact_action_sha256") != exact_sha
            or type(row.get("bitwise_equal")) is not bool
        ):
            raise FeedbackEvidenceError("development matched-state row identity differs")
        compared_sha = _require_sha256(
            row.get("compared_action_sha256"), field="development compared action"
        )
        _require_sha256(row.get("exact_action_sha256"), field="development exact action")
        l2 = row.get("l2_physical_action")
        maximum = row.get("max_abs_physical_action")
        if (
            type(l2) is not float
            or type(maximum) is not float
            or not math.isfinite(l2)
            or not math.isfinite(maximum)
            or l2 < 0.0
            or maximum < 0.0
        ):
            raise FeedbackEvidenceError("development matched-state action delta is malformed")
        equal = bool(row["bitwise_equal"])
        if condition == CONDITION_IDS[0] and not equal:
            raise FeedbackEvidenceError("development exact matched action is not deterministic")
        if equal != (compared_sha == exact_sha) or (equal and (l2 != 0.0 or maximum != 0.0)):
            raise FeedbackEvidenceError("development matched-state equality record is inconsistent")
        if not equal and (l2 <= 0.0 or maximum <= 0.0):
            raise FeedbackEvidenceError("development changed action has a zero recorded delta")
    return summarize_matched_action_deltas(rows)


def _arm_summary(
    aggregates: Mapping[str, object], *, block: int, condition: str
) -> dict[str, object]:
    return {
        "action_bounds_ok": aggregates["action_bounds_ok"],
        "block": block,
        "condition_id": condition,
        "fall": aggregates["fall"],
        "forbidden_contact_count": len(aggregates["forbidden_contacts"]),
        "observed_steps": aggregates["observed_steps"],
        "resynchronization_records": aggregates["resynchronization_records"],
        "segment_errors": aggregates["segment_errors"],
        "settled_state_normalized_error": aggregates["settled_state_normalized_error"],
        "settle_latency_steps": aggregates["settle_latency_steps"],
        "six_tracking_errors": aggregates["six_tracking_errors"],
        "task_success": None,
        "time_to_first_failure_steps": aggregates["time_to_first_failure_steps"],
        "transition_window_error": aggregates["transition_window_error"],
    }


def _validate_trace(
    trace: Mapping[str, object],
    *,
    block: int,
    condition: str,
    actor_sha256: str,
    protocol: DevelopmentProtocol,
    references: Mapping[str, np.ndarray],
    identities: Mapping[str, Mapping[str, object]],
    transforms: Mapping[str, Mapping[str, PreparedReferenceTransform]],
) -> tuple[dict[str, object], Mapping[str, Mapping[str, object]] | None]:
    expected_scalars = {
        "actor_export_sha256": actor_sha256,
        "block": block,
        "claim_ceiling": CLAIM_CEILING,
        "condition_id": condition,
        "development_only": True,
        "evidence_class": EVIDENCE_CLASS,
        "objective_metric_kernel": (
            "phase_b_protected_metric_math_reused_without_protected_split_or_gate"
        ),
        "objective_reference": "untransformed_exact_reference_rows",
        "policy_reference_factor": "actor_visible_8x45_window_only",
        "policy_seed": 121901,
        "promotable": False,
        "protocol_id": PROTOCOL_ID,
        "schema_version": 1,
        "task_success": None,
        "trace_schema_id": TRACE_SCHEMA_ID,
    }
    if set(trace) != _TRACE_KEYS or any(
        trace.get(name) != expected for name, expected in expected_scalars.items()
    ):
        raise FeedbackEvidenceError("development trace fields or fixed boundary differ")
    if trace.get("reference_identities") != identities:
        raise FeedbackEvidenceError("development trace reference identities differ")
    expected_transforms = {
        behavior: dict(transforms[behavior][condition].receipt) for behavior in ("expert", "simple")
    }
    if trace.get("reference_transform_precomputations") != expected_transforms:
        raise FeedbackEvidenceError("development reference transform receipt differs")
    objective_steps = trace.get("objective_steps")
    policy_inputs = trace.get("policy_inputs")
    if type(objective_steps) is not list or type(policy_inputs) is not list:
        raise FeedbackEvidenceError("development trace series are malformed")
    if trace.get("objective_steps_sha256") != protected_trace_sha256(objective_steps):
        raise FeedbackEvidenceError("development objective trace digest differs")
    if trace.get("policy_inputs_sha256") != _sha256(canonical_json_bytes(policy_inputs)):
        raise FeedbackEvidenceError("development policy-input digest differs")
    switches = _switches(trace, references)
    exact_actions = _validate_policy_and_reference_chain(
        trace,
        condition=condition,
        references=references,
        transforms=transforms,
        switches=switches,
    )
    aggregates = recompute_protected_episode(
        steps=objective_steps,
        cell="fixed_round_trip",
        switches=switches,
        segment_targets_m_s=protocol.target_speeds_m_s,
    )
    matched = trace.get("matched_state_action_deltas")
    if condition == CONDITION_IDS[0]:
        matched_summary = _validate_matched_rows(matched, exact_actions=exact_actions)
    else:
        if matched != []:
            raise FeedbackEvidenceError("only the exact arm may retain matched-state rows")
        matched_summary = None
    return _arm_summary(aggregates, block=block, condition=condition), matched_summary


def validate_development_result(
    result_path: Path,
    *,
    smoke_run: Path,
    corpus_root: Path,
    protocol_path: Path,
) -> ValidatedDevelopmentEvidence:
    """Reverify a complete in-sample result without opening held-out inputs."""

    result, result_bytes = _read_canonical_object(
        result_path,
        maximum_bytes=MAX_RESULT_BYTES,
        label="development result",
    )
    expected_count = len(DEVELOPMENT_BLOCKS) * len(CONDITION_IDS)
    expected_scalars = {
        "claim_ceiling": CLAIM_CEILING,
        "completed_arm_count": expected_count,
        "development_only": True,
        "evidence_class": EVIDENCE_CLASS,
        "held_out_inputs_used": False,
        "planned_arm_count": expected_count,
        "promotable": False,
        "result_schema_id": RESULT_SCHEMA_ID,
        "schema_version": 1,
        "status": "succeeded",
        "task_success": None,
    }
    if set(result) != _RESULT_KEYS or any(
        result.get(name) != expected for name, expected in expected_scalars.items()
    ):
        raise FeedbackEvidenceError("development result fields or claim boundary differ")
    actor_sha = _require_sha256(result.get("actor_export_sha256"), field="development actor")
    directory = Path(result_path).resolve(strict=True).parent
    if directory.is_symlink() or not directory.is_dir():
        raise FeedbackEvidenceError("development result directory must be real")
    protocol = load_development_protocol(protocol_path)
    smoke = validate_smoke_export(smoke_run)
    corpus_bindings = validate_development_corpus(
        corpus_root,
        lineage=smoke,
        protocol=protocol,
    )
    manifest, manifest_bytes, _manifest_path = _read_bound_object(
        directory,
        result.get("manifest"),
        maximum_bytes=MAX_MANIFEST_BYTES,
        label="development manifest",
    )
    _validate_manifest(
        manifest,
        result=result,
        protocol=protocol,
        smoke=smoke,
        corpus_bindings=corpus_bindings,
    )
    if actor_sha != smoke.actor_sha256:
        raise FeedbackEvidenceError("development result actor differs from smoke export")
    arms = result.get("arms")
    trace_records = result.get("trace_artifacts")
    if type(arms) is not list or type(trace_records) is not list:
        raise FeedbackEvidenceError("development result arms are malformed")
    if len(arms) != expected_count or len(trace_records) != expected_count:
        raise FeedbackEvidenceError("development result arm count differs")
    total_trace_bytes = 0
    validated_arms = []
    matched_by_block: dict[str, Mapping[str, Mapping[str, object]]] = {}
    reference_cache = {
        block: _reference_inputs(Path(corpus_root), block) for block in DEVELOPMENT_BLOCKS
    }
    expected_pairs = tuple(
        (block, condition) for block in DEVELOPMENT_BLOCKS for condition in CONDITION_IDS
    )
    for index, (block, condition) in enumerate(expected_pairs):
        arm = arms[index]
        if type(arm) is not dict or set(arm) != _ARM_KEYS:
            raise FeedbackEvidenceError("development arm fields differ")
        if arm.get("block") != block or arm.get("condition_id") != condition:
            raise FeedbackEvidenceError("development arm order or identity differs")
        if arm.get("trace") != trace_records[index]:
            raise FeedbackEvidenceError("development arm and trace index bindings differ")
        record = _artifact_binding(trace_records[index], field="development trace")
        expected_name = f"trace_block_{block}_{condition}.json"
        if record["filename"] != expected_name:
            raise FeedbackEvidenceError("development trace filename differs from bound arm")
        total_trace_bytes += int(record["byte_count"])
        if total_trace_bytes > MAX_TRACE_SET_BYTES:
            raise FeedbackEvidenceError("development trace set exceeds its byte bound")
        trace, _trace_bytes, _trace_path = _read_bound_object(
            directory,
            record,
            maximum_bytes=MAX_TRACE_BYTES,
            label="development trace",
        )
        summary, matched = _validate_trace(
            trace,
            block=block,
            condition=condition,
            actor_sha256=actor_sha,
            protocol=protocol,
            references=reference_cache[block][0],
            identities=reference_cache[block][1],
            transforms=reference_cache[block][2],
        )
        expected_arm = {**summary, "trace": record}
        if not _same_json(arm, expected_arm):
            raise FeedbackEvidenceError("development arm summary does not recompute")
        if matched is not None:
            matched_by_block[str(block)] = {name: dict(value) for name, value in matched.items()}
        validated_arms.append(expected_arm)
    if not _same_json(result.get("matched_state_action_deltas"), matched_by_block):
        raise FeedbackEvidenceError("development matched-state summary does not recompute")
    return ValidatedDevelopmentEvidence(
        result=result,
        result_sha256=_sha256(result_bytes),
        manifest=manifest,
        manifest_sha256=_sha256(manifest_bytes),
        trace_set_sha256=_sha256(canonical_json_bytes(trace_records)),
        arms=tuple(validated_arms),
    )


__all__ = [
    "ValidatedDevelopmentEvidence",
    "validate_development_result",
]
