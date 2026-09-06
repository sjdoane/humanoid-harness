"""Direct-state formulas; reward-telemetry separation proven by test."""

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
    E4_SCREEN_REFERENCE_SEEDS,
    array_sha256,
    canonical_json_bytes,
    validate_corpus_manifest,
)
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.harness.inputs import load_library_manifest
from oracle_composition.phase_b.protected_metrics import (
    CONTROL_PERIOD_SECONDS,
    ERROR_NAMES,
    ERROR_SCALES,
    evaluator_mass_center_x_m,
    evaluator_tracking_errors,
    reference_row_record,
    validate_protected_step,
)
from oracle_composition.phase_b.reference_runtime import load_v2_reference_clip

T2_EVALUATOR_ID = "t2_direct_state_reward_telemetry_separated_evaluator/v1"
T2_EVALUATOR_DESIGN_SCHEMA_ID = "t2_protected_evaluator_design/v1"
T2_TRACE_SCHEMA_ID = "t2_protected_episode_trace/v1"
T2_REPORT_SCHEMA_ID = "t2_reward_study_report/v1"
T2_HORIZON_STEPS = 1_000
T2_TARGET_SPEED_M_S = 3.0
T2_SPEED_BAND_M_S = (2.75, 3.25)
T2_EVALUATION_SEEDS = tuple(range(97001, 97021))
T2_LIBRARY_PATH = "experiments/003_composition_speed_profile/library_manifest_v1.json"
T2_CORPUS_PATH = "artifacts/reference_corpus_v2/corpus_manifest_v2.json"
T2_CORPUS_INDEX_PATH = "artifacts/reference_corpus_v2/corpus_index_v2.json"
_QUANTILE_PROBABILITIES = (0.05, 0.25, 0.5, 0.75, 0.95)


def _finite(value: object, *, field: str) -> float:
    if type(value) not in {int, float}:
        raise ExperimentContractError(f"T2 {field} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ExperimentContractError(f"T2 {field} must be finite") from exc
    if not math.isfinite(result):
        raise ExperimentContractError(f"T2 {field} must be finite")
    return result


def _sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"T2 {field} must be a lowercase SHA-256")
    return value


def _positive_seed(value: object, *, field: str) -> int:
    if type(value) is not int or not 0 < value <= 2_147_483_647:
        raise ExperimentContractError(f"T2 {field} is invalid")
    return value


def _canonical_regular_json(
    path: Path, *, field: str, maximum: int
) -> tuple[dict[str, object], bytes]:
    candidate = Path(path)
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise ExperimentContractError(f"T2 {field} is unavailable") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode) or before.st_size > maximum:
        raise ExperimentContractError(f"T2 {field} must be a bounded regular file")
    encoded = candidate.read_bytes()
    after = candidate.lstat()
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in identity):
        raise ExperimentContractError(f"T2 {field} changed while read")
    try:
        value = json.loads(encoded)
    except (UnicodeError, ValueError) as exc:
        raise ExperimentContractError(f"T2 {field} is not JSON") from exc
    if type(value) is not dict or canonical_json_bytes(value) != encoded:
        raise ExperimentContractError(f"T2 {field} is not canonical JSON")
    return value, encoded


def _direct_fall(state: Mapping[str, object]) -> bool:
    orientation = np.ascontiguousarray(state["root_orientation_wxyz"], dtype="<f8")
    if orientation.shape != (4,) or not np.isfinite(orientation).all():
        raise ExperimentContractError("T2 fall orientation is malformed")
    w, x, y, z = (float(value) for value in orientation)
    norm = w * w + x * x + y * y + z * z
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=2e-6):
        raise ExperimentContractError("T2 fall orientation is not normalized")
    torso_up = (w * w - x * x - y * y + z * z) / norm
    height = _finite(state["root_height_m"], field="root_height_m")
    return not 1.0 <= height <= 2.0 or torso_up < 0.5


def _longest_out_of_band_run(values: Sequence[float]) -> int:
    low, high = T2_SPEED_BAND_M_S
    longest = 0
    current = 0
    for value in values:
        if low <= value <= high:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


def summarize_t2_speeds(values: Sequence[float]) -> dict[str, object]:
    speeds = np.ascontiguousarray(values, dtype="<f8")
    if speeds.shape != (T2_HORIZON_STEPS,) or not np.isfinite(speeds).all():
        raise ExperimentContractError("T2 protected speed series is incomplete or malformed")
    absolute_errors = np.abs(speeds - T2_TARGET_SPEED_M_S)
    low, high = T2_SPEED_BAND_M_S
    quantiles = np.quantile(speeds, _QUANTILE_PROBABILITIES, method="linear")
    return {
        "fraction_steps_in_target_band": float(np.mean((speeds >= low) & (speeds <= high))),
        "longest_out_of_band_run_steps": _longest_out_of_band_run(speeds.tolist()),
        "mean_absolute_per_step_error_m_s": float(np.mean(absolute_errors, dtype=np.float64)),
        "protected_task_return": float(
            np.sum(1.0 - np.minimum(1.0, absolute_errors / T2_TARGET_SPEED_M_S))
        ),
        "speed_quantiles_m_s": {
            name: float(value)
            for name, value in zip(("p05", "p25", "p50", "p75", "p95"), quantiles, strict=True)
        },
    }


@dataclass(frozen=True, slots=True)
class T2VerifiedReference:
    rows: np.ndarray
    lineage: Mapping[str, object]
    lineage_sha256: str


def _evaluation_block(evaluation_seed: int) -> int:
    if evaluation_seed not in T2_EVALUATION_SEEDS:
        raise ExperimentContractError("T2 evaluation seed is outside the frozen set")
    return E4_SCREEN_REFERENCE_SEEDS[evaluation_seed - T2_EVALUATION_SEEDS[0]]


def load_t2_verified_reference(
    *, repository_root: Path, evaluation_seed: int
) -> T2VerifiedReference:
    """Resolve one expert clip through library, corpus, index, bundle, and payload."""

    root = Path(repository_root).resolve(strict=True)
    library_path = root / T2_LIBRARY_PATH
    corpus_path = root / T2_CORPUS_PATH
    index_path = root / T2_CORPUS_INDEX_PATH
    try:
        library = load_library_manifest(library_path)
    except (OSError, ValueError) as exc:
        raise ExperimentContractError(f"T2 library manifest failed verification: {exc}") from exc
    if set(library.behavior_names) != {"expert", "medium", "simple"}:
        raise ExperimentContractError("T2 library behavior identities differ")
    corpus_binding = library.source_evidence.get("corpus_manifest_v2")
    if corpus_binding is None or corpus_binding.path != T2_CORPUS_PATH:
        raise ExperimentContractError("T2 library does not bind the frozen corpus path")
    corpus, corpus_bytes = _canonical_regular_json(
        corpus_path, field="corpus manifest", maximum=256 * 1024
    )
    if hashlib.sha256(corpus_bytes).hexdigest() != corpus_binding.sha256:
        raise ExperimentContractError("T2 library corpus digest differs")
    try:
        validate_corpus_manifest(corpus)
    except ValueError as exc:
        raise ExperimentContractError(f"T2 corpus manifest failed verification: {exc}") from exc
    index, index_bytes = _canonical_regular_json(
        index_path, field="corpus index", maximum=256 * 1024
    )
    index_core = dict(index)
    index_content_sha256 = index_core.pop("index_content_sha256", None)
    corpus_index_binding = index.get("corpus_manifest")
    if (
        index_content_sha256 != hashlib.sha256(canonical_json_bytes(index_core)).hexdigest()
        or type(corpus_index_binding) is not dict
        or corpus_index_binding.get("sha256") != corpus_binding.sha256
    ):
        raise ExperimentContractError("T2 corpus index content or manifest binding differs")
    block = _evaluation_block(evaluation_seed)
    clip_id = f"corpus-{block}-expert"
    corpus_matches = [
        item for item in corpus["clips_in_reset_order"] if item.get("clip_id") == clip_id
    ]
    index_matches = [item for item in index.get("clips", ()) if item.get("clip_id") == clip_id]
    if len(corpus_matches) != 1 or len(index_matches) != 1:
        raise ExperimentContractError("T2 verified chain does not resolve exactly one expert clip")
    corpus_entry = corpus_matches[0]
    index_entry = index_matches[0]
    if any(
        corpus_entry[field] != index_entry[field]
        for field in (
            "bundle_manifest_sha256",
            "clip_id",
            "reference_identity_sha256",
        )
    ):
        raise ExperimentContractError("T2 corpus and index clip identities differ")
    try:
        clip = load_v2_reference_clip(
            root / "artifacts/reference_corpus_v2", block=block, behavior="expert"
        )
    except ValueError as exc:
        raise ExperimentContractError(f"T2 reference bundle failed verification: {exc}") from exc
    if (
        clip.bundle_sha256 != corpus_entry["bundle_manifest_sha256"]
        or clip.payload_sha256 != index_entry["payload_sha256"]
        or clip.reference_identity_sha256 != corpus_entry["reference_identity_sha256"]
        or clip.reference_rows.shape != (T2_HORIZON_STEPS + 1, 45)
    ):
        raise ExperimentContractError("T2 loaded reference identity differs from verified chain")
    rows = np.array(clip.reference_rows, dtype="<f8", order="C", copy=True)
    rows.setflags(write=False)
    lineage = {
        "behavior": "expert",
        "block": block,
        "bundle_manifest_sha256": clip.bundle_sha256,
        "clip_id": clip_id,
        "corpus_index": {
            "byte_count": len(index_bytes),
            "path": T2_CORPUS_INDEX_PATH,
            "sha256": hashlib.sha256(index_bytes).hexdigest(),
        },
        "library_manifest": {
            "byte_count": library_path.stat().st_size,
            "path": T2_LIBRARY_PATH,
            "sha256": library.raw_sha256,
        },
        "payload_sha256": clip.payload_sha256,
        "reference_corpus_manifest": {
            "byte_count": len(corpus_bytes),
            "path": T2_CORPUS_PATH,
            "sha256": corpus_binding.sha256,
        },
        "reference_identity_sha256": clip.reference_identity_sha256,
        "reference_rows": {
            "dtype": "<f8",
            "sha256": array_sha256(rows),
            "shape": [T2_HORIZON_STEPS + 1, 45],
        },
    }
    lineage_sha256 = hashlib.sha256(canonical_json_bytes(lineage)).hexdigest()
    return T2VerifiedReference(rows=rows, lineage=lineage, lineage_sha256=lineage_sha256)


@dataclass(frozen=True, slots=True)
class T2EpisodeMetrics:
    policy_seed: int
    evaluation_seed: int
    checkpoint_sha256: str
    trace_sha256: str
    reference_lineage_sha256: str
    observed_steps: int
    com_forward_speed_m_s: tuple[float, ...]
    mean_absolute_per_step_error_m_s: float | None
    fraction_steps_in_target_band: float | None
    protected_task_return: float | None
    speed_quantiles_m_s: Mapping[str, float] | None
    longest_out_of_band_run_steps: int | None
    first_fall_step: int | None
    fall_step_count: int
    forbidden_contact_count: int
    six_tracking_rmse: Mapping[str, float] | None
    complete_trace: bool
    action_bounds_ok: bool
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _positive_seed(self.policy_seed, field="policy_seed")
        if self.evaluation_seed not in T2_EVALUATION_SEEDS:
            raise ExperimentContractError("T2 evaluation seed is outside the frozen set")
        _sha256(self.checkpoint_sha256, field="checkpoint_sha256")
        _sha256(self.trace_sha256, field="trace_sha256")
        _sha256(self.reference_lineage_sha256, field="reference_lineage_sha256")
        if type(self.observed_steps) is not int or not 0 <= self.observed_steps <= T2_HORIZON_STEPS:
            raise ExperimentContractError("T2 observed step count is invalid")
        if len(self.com_forward_speed_m_s) != self.observed_steps:
            raise ExperimentContractError("T2 speed series length differs from observed steps")
        for value in self.com_forward_speed_m_s:
            _finite(value, field="COM forward speed")
        if type(self.complete_trace) is not bool or type(self.action_bounds_ok) is not bool:
            raise ExperimentContractError("T2 completeness and action flags must be booleans")
        if type(self.fall_step_count) is not int or self.fall_step_count < 0:
            raise ExperimentContractError("T2 fall step count is invalid")
        if type(self.forbidden_contact_count) is not int or self.forbidden_contact_count < 0:
            raise ExperimentContractError("T2 contact count is invalid")
        if self.first_fall_step is not None and (
            type(self.first_fall_step) is not int
            or not 1 <= self.first_fall_step <= self.observed_steps
        ):
            raise ExperimentContractError("T2 first-fall step is invalid")
        if (self.first_fall_step is None) is not (self.fall_step_count == 0):
            raise ExperimentContractError("T2 fall count and first-fall step differ")
        if (
            type(self.failure_reasons) is not tuple
            or len(set(self.failure_reasons)) != len(self.failure_reasons)
            or any(type(reason) is not str or not reason for reason in self.failure_reasons)
        ):
            raise ExperimentContractError("T2 failure reasons are malformed")
        complete_fields = (
            self.mean_absolute_per_step_error_m_s,
            self.fraction_steps_in_target_band,
            self.protected_task_return,
            self.speed_quantiles_m_s,
            self.longest_out_of_band_run_steps,
            self.six_tracking_rmse,
        )
        if self.complete_trace:
            if self.observed_steps != T2_HORIZON_STEPS or any(
                value is None for value in complete_fields
            ):
                raise ExperimentContractError("T2 complete trace lacks a whole-episode metric")
            expected = summarize_t2_speeds(self.com_forward_speed_m_s)
            observed = {
                "fraction_steps_in_target_band": self.fraction_steps_in_target_band,
                "longest_out_of_band_run_steps": self.longest_out_of_band_run_steps,
                "mean_absolute_per_step_error_m_s": self.mean_absolute_per_step_error_m_s,
                "protected_task_return": self.protected_task_return,
                "speed_quantiles_m_s": dict(self.speed_quantiles_m_s or {}),
            }
            if observed != expected:
                raise ExperimentContractError("T2 cached speed metrics differ from the series")
            if set(self.six_tracking_rmse or {}) != set(ERROR_NAMES):
                raise ExperimentContractError("T2 episode lacks all six tracking RMSEs")
            for value in (self.six_tracking_rmse or {}).values():
                if _finite(value, field="tracking RMSE") < 0.0:
                    raise ExperimentContractError("T2 tracking RMSE is negative")
        elif any(value is not None for value in complete_fields):
            raise ExperimentContractError("T2 incomplete trace cannot report whole-episode metrics")

    @property
    def safety_passed(self) -> bool:
        return (
            self.complete_trace
            and self.action_bounds_ok
            and self.fall_step_count == 0
            and self.forbidden_contact_count == 0
        )

    @property
    def tracking_passed(self) -> bool:
        return self.complete_trace and all(
            float((self.six_tracking_rmse or {})[name]) <= ERROR_SCALES[name]
            for name in ERROR_NAMES
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "action_bounds_ok": self.action_bounds_ok,
            "checkpoint_sha256": self.checkpoint_sha256,
            "com_forward_speed_m_s": list(self.com_forward_speed_m_s),
            "complete_trace": self.complete_trace,
            "evaluation_seed": self.evaluation_seed,
            "failure_reasons": list(self.failure_reasons),
            "first_fall_step": self.first_fall_step,
            "fraction_steps_in_target_band": self.fraction_steps_in_target_band,
            "fall_step_count": self.fall_step_count,
            "forbidden_contact_count": self.forbidden_contact_count,
            "longest_out_of_band_run_steps": self.longest_out_of_band_run_steps,
            "mean_absolute_per_step_error_m_s": self.mean_absolute_per_step_error_m_s,
            "observed_steps": self.observed_steps,
            "policy_seed": self.policy_seed,
            "protected_task_return": self.protected_task_return,
            "reference_lineage_sha256": self.reference_lineage_sha256,
            "safety_passed": self.safety_passed,
            "six_tracking_rmse": (
                None
                if self.six_tracking_rmse is None
                else dict(sorted(self.six_tracking_rmse.items()))
            ),
            "speed_quantiles_m_s": (
                None
                if self.speed_quantiles_m_s is None
                else dict(sorted(self.speed_quantiles_m_s.items()))
            ),
            "trace_sha256": self.trace_sha256,
            "tracking_passed": self.tracking_passed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> T2EpisodeMetrics:
        expected = {
            "action_bounds_ok",
            "checkpoint_sha256",
            "com_forward_speed_m_s",
            "complete_trace",
            "evaluation_seed",
            "failure_reasons",
            "first_fall_step",
            "fraction_steps_in_target_band",
            "fall_step_count",
            "forbidden_contact_count",
            "longest_out_of_band_run_steps",
            "mean_absolute_per_step_error_m_s",
            "observed_steps",
            "policy_seed",
            "protected_task_return",
            "reference_lineage_sha256",
            "safety_passed",
            "six_tracking_rmse",
            "speed_quantiles_m_s",
            "trace_sha256",
            "tracking_passed",
        }
        if type(value) is not dict or set(value) != expected:
            raise ExperimentContractError("T2 episode metric fields differ")
        if (
            type(value["com_forward_speed_m_s"]) is not list
            or type(value["failure_reasons"]) is not list
        ):
            raise ExperimentContractError("T2 episode metric arrays differ")
        result = cls(
            action_bounds_ok=value["action_bounds_ok"],
            checkpoint_sha256=value["checkpoint_sha256"],
            com_forward_speed_m_s=tuple(value["com_forward_speed_m_s"]),
            complete_trace=value["complete_trace"],
            evaluation_seed=value["evaluation_seed"],
            failure_reasons=tuple(value["failure_reasons"]),
            first_fall_step=value["first_fall_step"],
            fraction_steps_in_target_band=value["fraction_steps_in_target_band"],
            fall_step_count=value["fall_step_count"],
            forbidden_contact_count=value["forbidden_contact_count"],
            longest_out_of_band_run_steps=value["longest_out_of_band_run_steps"],
            mean_absolute_per_step_error_m_s=value["mean_absolute_per_step_error_m_s"],
            observed_steps=value["observed_steps"],
            policy_seed=value["policy_seed"],
            protected_task_return=value["protected_task_return"],
            reference_lineage_sha256=value["reference_lineage_sha256"],
            six_tracking_rmse=(
                None if value["six_tracking_rmse"] is None else dict(value["six_tracking_rmse"])
            ),
            speed_quantiles_m_s=(
                None if value["speed_quantiles_m_s"] is None else dict(value["speed_quantiles_m_s"])
            ),
            trace_sha256=value["trace_sha256"],
        )
        if value["safety_passed"] is not result.safety_passed:
            raise ExperimentContractError("T2 cached safety result differs")
        if value["tracking_passed"] is not result.tracking_passed:
            raise ExperimentContractError("T2 cached tracking result differs")
        return result


def t2_step_record(*, protected_step: Mapping[str, object]) -> dict[str, object]:
    """Return only evaluator-owned direct state; reward telemetry is separate."""

    result = validate_protected_step(dict(protected_step))
    canonical_json_bytes(result)
    return result


def build_t2_trace(
    *,
    policy_seed: int,
    evaluation_seed: int,
    checkpoint_sha256: str,
    repository_root: Path,
    steps: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build a trace against verified-chain expert reference rows 0 through 1000."""

    reference = load_t2_verified_reference(
        repository_root=repository_root, evaluation_seed=evaluation_seed
    )
    value = {
        "checkpoint_sha256": _sha256(checkpoint_sha256, field="checkpoint_sha256"),
        "deterministic_actions": True,
        "evaluation_seed": _positive_seed(evaluation_seed, field="evaluation_seed"),
        "expert_start": True,
        "policy_seed": _positive_seed(policy_seed, field="policy_seed"),
        "reference_lineage": dict(reference.lineage),
        "reference_lineage_sha256": reference.lineage_sha256,
        "reset_reference": reference_row_record(behavior="expert", index=0, row=reference.rows[0]),
        "schema_version": 1,
        "steps": [dict(step) for step in steps],
        "trace_schema_id": T2_TRACE_SCHEMA_ID,
    }
    value["steps_sha256"] = hashlib.sha256(canonical_json_bytes(value["steps"])).hexdigest()
    _validate_t2_trace_against_reference(value, reference=reference)
    return value


def _validate_t2_trace_against_reference(
    value: Mapping[str, object],
    *,
    reference: T2VerifiedReference,
) -> dict[str, object]:
    expected = {
        "checkpoint_sha256",
        "deterministic_actions",
        "evaluation_seed",
        "expert_start",
        "policy_seed",
        "reference_lineage",
        "reference_lineage_sha256",
        "reset_reference",
        "schema_version",
        "steps",
        "steps_sha256",
        "trace_schema_id",
    }
    if type(value) is not dict or set(value) != expected:
        raise ExperimentContractError("T2 protected trace fields differ")
    if (
        value["schema_version"] != 1
        or value["trace_schema_id"] != T2_TRACE_SCHEMA_ID
        or value["deterministic_actions"] is not True
        or value["expert_start"] is not True
    ):
        raise ExperimentContractError("T2 protected trace identity differs")
    _positive_seed(value["policy_seed"], field="policy_seed")
    evaluation_seed = value["evaluation_seed"]
    if evaluation_seed not in T2_EVALUATION_SEEDS:
        raise ExperimentContractError("T2 trace evaluation seed is outside the frozen set")
    _sha256(value["checkpoint_sha256"], field="checkpoint_sha256")
    if (
        reference.lineage != value["reference_lineage"]
        or reference.lineage_sha256 != value["reference_lineage_sha256"]
        or value["reference_lineage_sha256"]
        != hashlib.sha256(canonical_json_bytes(value["reference_lineage"])).hexdigest()
    ):
        raise ExperimentContractError("T2 trace verified reference lineage differs")
    reset = value["reset_reference"]
    if reset != reference_row_record(behavior="expert", index=0, row=reference.rows[0]):
        raise ExperimentContractError("T2 reset reference differs from verified-chain expert row 0")
    steps = value["steps"]
    if type(steps) is not list or len(steps) > T2_HORIZON_STEPS:
        raise ExperimentContractError("T2 trace step collection is malformed")
    for expected_step, raw_step in enumerate(steps):
        protected = validate_protected_step(dict(raw_step))
        row = protected["reference"]
        if (
            protected["step"] != expected_step
            or row["behavior"] != "expert"
            or row["index"] != expected_step + 1
            or row
            != reference_row_record(
                behavior="expert",
                index=expected_step + 1,
                row=reference.rows[expected_step + 1],
            )
        ):
            raise ExperimentContractError(
                "T2 trace reference differs from the verified-chain expert row"
            )
    expected_steps_sha256 = hashlib.sha256(canonical_json_bytes(steps)).hexdigest()
    if value["steps_sha256"] != expected_steps_sha256:
        raise ExperimentContractError("T2 protected trace step identity differs")
    canonical_json_bytes(dict(value))
    return dict(value)


def validate_t2_trace(
    trace: Mapping[str, object],
    *,
    repository_root: Path,
) -> dict[str, object]:
    """Resolve the verified chain before accepting any externally supplied trace."""

    evaluation_seed = trace.get("evaluation_seed") if type(trace) is dict else None
    reference = load_t2_verified_reference(
        repository_root=repository_root, evaluation_seed=evaluation_seed
    )
    return _validate_t2_trace_against_reference(trace, reference=reference)


def _evaluate_t2_trace_against_reference(
    trace: Mapping[str, object],
    *,
    reference: T2VerifiedReference,
) -> T2EpisodeMetrics:
    value = _validate_t2_trace_against_reference(trace, reference=reference)
    speeds: list[float] = []
    errors = {name: [] for name in ERROR_NAMES}
    first_fall = None
    fall_steps = 0
    contacts = 0
    action_bounds_ok = True
    early_termination = False
    early_truncation = False
    for index, step in enumerate(value["steps"]):
        mass_center = step["mass_center_state"]
        masses = np.ascontiguousarray(mass_center["body_mass_kg"], dtype="<f8")
        before = np.ascontiguousarray(mass_center["body_xipos_before_world_m"], dtype="<f8")
        after = np.ascontiguousarray(mass_center["body_xipos_after_world_m"], dtype="<f8")
        speed = (
            evaluator_mass_center_x_m(masses, after) - evaluator_mass_center_x_m(masses, before)
        ) / CONTROL_PERIOD_SECONDS
        speeds.append(speed)
        tracking = evaluator_tracking_errors(step["state"], step["reference"]["values"])
        for name in ERROR_NAMES:
            errors[name].append(tracking[name])
        fallen = _direct_fall(step["state"])
        if step["fallen"] is not fallen:
            raise ExperimentContractError("T2 recorded fall flag differs from direct state")
        if fallen:
            fall_steps += 1
            if first_fall is None:
                first_fall = index + 1
        contacts += len(step["forbidden_contacts"])
        action = np.ascontiguousarray(step["action"], dtype="<f4")
        action_bounds_ok = action_bounds_ok and bool(
            np.all(action >= np.float32(-0.4)) and np.all(action <= np.float32(0.4))
        )
        early_termination = early_termination or bool(step["plant_terminated"])
        early_truncation = early_truncation or (
            bool(step["plant_truncated"]) and index + 1 != T2_HORIZON_STEPS
        )
    complete = (
        len(value["steps"]) == T2_HORIZON_STEPS and not early_termination and not early_truncation
    )
    failure_reasons = []
    if not complete:
        failure_reasons.append("incomplete_trace")
    if early_termination:
        failure_reasons.append("plant_terminated")
    if early_truncation:
        failure_reasons.append("early_truncation")
    if not action_bounds_ok:
        failure_reasons.append("invalid_action")
    if fall_steps:
        failure_reasons.append("fall")
    if contacts:
        failure_reasons.append("forbidden_contact")
    summaries = summarize_t2_speeds(speeds) if complete else None
    tracking_rmse = (
        {
            name: math.sqrt(float(np.mean(np.square(values), dtype=np.float64)))
            for name, values in errors.items()
        }
        if complete
        else None
    )
    trace_sha256 = hashlib.sha256(canonical_json_bytes(value)).hexdigest()
    return T2EpisodeMetrics(
        action_bounds_ok=action_bounds_ok,
        checkpoint_sha256=value["checkpoint_sha256"],
        com_forward_speed_m_s=tuple(speeds),
        complete_trace=complete,
        evaluation_seed=value["evaluation_seed"],
        failure_reasons=tuple(failure_reasons),
        first_fall_step=first_fall,
        fraction_steps_in_target_band=(
            None if summaries is None else summaries["fraction_steps_in_target_band"]
        ),
        fall_step_count=fall_steps,
        forbidden_contact_count=contacts,
        longest_out_of_band_run_steps=(
            None if summaries is None else summaries["longest_out_of_band_run_steps"]
        ),
        mean_absolute_per_step_error_m_s=(
            None if summaries is None else summaries["mean_absolute_per_step_error_m_s"]
        ),
        observed_steps=len(value["steps"]),
        policy_seed=value["policy_seed"],
        protected_task_return=(None if summaries is None else summaries["protected_task_return"]),
        reference_lineage_sha256=value["reference_lineage_sha256"],
        six_tracking_rmse=tracking_rmse,
        speed_quantiles_m_s=(None if summaries is None else summaries["speed_quantiles_m_s"]),
        trace_sha256=trace_sha256,
    )


def evaluate_t2_trace(trace: Mapping[str, object], *, repository_root: Path) -> T2EpisodeMetrics:
    """Resolve the verified chain, then compute direct-state T2 endpoints."""

    evaluation_seed = trace.get("evaluation_seed") if type(trace) is dict else None
    reference = load_t2_verified_reference(
        repository_root=repository_root, evaluation_seed=evaluation_seed
    )
    return _evaluate_t2_trace_against_reference(trace, reference=reference)


def evaluator_design_contract_value(
    *,
    evaluator_source_sha256: str,
    protected_metrics_source_sha256: str,
    report_v2_source_sha256: str,
    report_writer_source_sha256: str,
) -> dict[str, object]:
    """Return the frozen, source-bound evaluator design artifact."""

    return {
        "calibration": "none",
        "deterministic_actions": True,
        "episode_steps": T2_HORIZON_STEPS,
        "evaluation_seeds": list(T2_EVALUATION_SEEDS),
        "evaluator_id": T2_EVALUATOR_ID,
        "evaluator_design_schema_id": T2_EVALUATOR_DESIGN_SCHEMA_ID,
        "evaluator_source_sha256": _sha256(
            evaluator_source_sha256, field="evaluator_source_sha256"
        ),
        "expert_start": True,
        "metric_definitions": {
            "action_bounds_ok": ("all_finite_raw_float32_actions_in_inclusive_-0.4_to_0.4"),
            "com_forward_speed_m_s": ("body_mass_weighted_com_x_after_minus_before_over_0.015_s"),
            "first_fall_step": "fall_only_one_based_post_step_or_null",
            "fraction_steps_in_target_band": "inclusive_2.75_to_3.25_m_s",
            "mean_absolute_per_step_error_m_s": ("mean_t_abs(com_speed_t_minus_3.0)"),
            "protected_task_return": ("sum_t_1_minus_min_1_abs_error_over_3.0"),
            "six_tracking_rmse": ("sqrt_mean_over_1000_steps_of_squared_direct_state_error"),
        },
        "protected_metrics_source_sha256": _sha256(
            protected_metrics_source_sha256,
            field="protected_metrics_source_sha256",
        ),
        "reference_chain": ("library_to_corpus_to_index_to_bundle_to_payload_to_row"),
        "report_schema_id": T2_REPORT_SCHEMA_ID,
        "report_v2_source_sha256": _sha256(
            report_v2_source_sha256, field="report_v2_source_sha256"
        ),
        "report_writer_source_sha256": _sha256(
            report_writer_source_sha256,
            field="report_writer_source_sha256",
        ),
        "reward_helpers_imported": False,
        "reward_telemetry_separation": (
            "protected_trace_and_scientific_receipt_exclude_reward_outputs"
        ),
        "schema_version": 1,
        "trace_schema_id": T2_TRACE_SCHEMA_ID,
        "tracking_scales": dict(sorted(ERROR_SCALES.items())),
    }


def load_evaluator_design(
    path: Path,
    *,
    evaluator_source_path: Path,
    protected_metrics_source_path: Path,
    report_v2_source_path: Path,
    report_writer_source_path: Path,
) -> tuple[dict[str, object], str]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file() or candidate.stat().st_size > 64 * 1024:
        raise ExperimentContractError("T2 evaluator design is unavailable or oversized")
    encoded = candidate.read_bytes()
    try:
        value = json.loads(encoded)
    except (UnicodeError, ValueError) as exc:
        raise ExperimentContractError("T2 evaluator design is not JSON") from exc
    expected = evaluator_design_contract_value(
        evaluator_source_sha256=hashlib.sha256(
            Path(evaluator_source_path).read_bytes()
        ).hexdigest(),
        protected_metrics_source_sha256=hashlib.sha256(
            Path(protected_metrics_source_path).read_bytes()
        ).hexdigest(),
        report_v2_source_sha256=hashlib.sha256(
            Path(report_v2_source_path).read_bytes()
        ).hexdigest(),
        report_writer_source_sha256=hashlib.sha256(
            Path(report_writer_source_path).read_bytes()
        ).hexdigest(),
    )
    if value != expected or canonical_json_bytes(value) != encoded:
        raise ExperimentContractError("T2 evaluator design bytes or source bindings differ")
    return value, hashlib.sha256(encoded).hexdigest()


__all__ = [
    "T2_CORPUS_INDEX_PATH",
    "T2_CORPUS_PATH",
    "T2_EVALUATION_SEEDS",
    "T2_EVALUATOR_DESIGN_SCHEMA_ID",
    "T2_EVALUATOR_ID",
    "T2_HORIZON_STEPS",
    "T2_LIBRARY_PATH",
    "T2_REPORT_SCHEMA_ID",
    "T2_SPEED_BAND_M_S",
    "T2_TARGET_SPEED_M_S",
    "T2_TRACE_SCHEMA_ID",
    "T2EpisodeMetrics",
    "T2VerifiedReference",
    "build_t2_trace",
    "evaluate_t2_trace",
    "evaluator_design_contract_value",
    "load_evaluator_design",
    "load_t2_verified_reference",
    "summarize_t2_speeds",
    "t2_step_record",
    "validate_t2_trace",
]
