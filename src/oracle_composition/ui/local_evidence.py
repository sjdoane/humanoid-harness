"""Validate the optional local oracle-exploration bundle."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

EXPLORATION_DIR = Path("artifacts/exploration/phase_oracle_holdout_v0")
_JSON_LIMIT = 1_000_000
_RUNS_LIMIT = 16_000_000
_MEDIA_LIMIT = 24_000_000
_SHA256_LENGTH = 64
_V0_MANIFEST_SHA256 = "39054d82a13aac694381f7fb8bad534ac7052f3218e08a47f137513aac35e67d"
_V0_RUNS_SHA256 = "0fd09731366db83b989761a31b7706dd6543a2c5b1e45a736a72af7a28fab0f6"
_V0_FILE_SHA256 = {
    "figure_data.csv": "de1edc314701c391e4f9d733ae507ccdcdcb0dea8533ea9bb7e80d28ac8b5fbc",
    "holdout_summary.alt.txt": "6f6f09832542745e92d2518cd03dc3ca5e7fbcf905b5acd413863f5a1b2125ab",
    "holdout_summary.figure_manifest.json": (
        "2882109bf22f275c997871f62def68e86a1ede03ef8cad3fcb13f4787de74939"
    ),
    "holdout_summary.png": "3ce6bccc7972efcab4717a23b3438649a88007e706c05b4985942222c4128fd9",
    "holdout_summary.svg": "51cd86d1d4afa31f8a2aa117945a1ab6a34b5c8a7503bb9c7240fd90e4519882",
    "lateral_seed3000_four_arm.video_receipt.json": (
        "11e9de10b4c19541d6a749b0a0fba124407771fcedf8abf4555732ab0f85d046"
    ),
    "lateral_seed3000_poster.png": (
        "de0763684930b3eaa3b3f16420f529f919035f4c8d79f25d5081cb069006a68d"
    ),
    "lateral_seed3000_four_arm.mp4": (
        "62106bbe3a773d5f757b742d8ee5a8df6036962b7f6a6dcd5a17f2b50b7b4701"
    ),
}
_MANIFEST_FIELDS = frozenset(
    {
        "arms",
        "bounded_phase_rule",
        "claim_ceiling",
        "frozen_runtime",
        "locked_before_holdout",
        "locked_expectations",
        "paired_evaluation",
        "question",
        "reason_not_admitted",
        "recovery_fallback_rule",
        "redistributable",
        "reference",
        "schema_version",
        "status",
        "visual_evidence",
    }
)
_SUMMARY_FIELDS = frozenset(
    {
        "completed_runs",
        "groups",
        "manifest_sha256",
        "runs_sha256",
        "scheduled_runs",
        "status",
    }
)
_FIGURE_MANIFEST_FIELDS = frozenset(
    {
        "audience",
        "outputs",
        "publisher_requirements",
        "source_manifest",
        "source_manifest_sha256",
        "source_runs",
        "source_runs_sha256",
        "status",
        "transformations",
        "uncertainty",
    }
)
_VIDEO_RECEIPT_FIELDS = frozenset(
    {
        "arms",
        "claim_ceiling",
        "condition",
        "manifest_sha256",
        "metrics",
        "perturb_action",
        "poster_sha256",
        "render_stride_actions",
        "runs_sha256",
        "seed",
        "selection",
        "status",
        "video_dimensions_px",
        "video_duration_seconds",
        "video_fps",
        "video_frames",
        "video_sha256",
    }
)
_V0_ARMS = ("T0_G0", "T0_G1", "T1_G0", "T1_G1", "base_context")
_V0_FACTORIAL_ARMS = ("T0_G0", "T0_G1", "T1_G0", "T1_G1")
_V0_CONDITIONS = ("nominal", "lateral_velocity", "pitch_velocity_falsifier")
_V0_SEEDS = tuple(range(3000, 3020))
_RUN_FIELDS = frozenset(
    {
        "action_saturation_count",
        "actions_executed",
        "arm",
        "collapsed_action_count",
        "condition",
        "environment_return",
        "final_recovered",
        "final_root_height_m",
        "final_torso_up_z",
        "first_collapse_action",
        "manifest_sha256",
        "no_collapse",
        "phase_correction_count",
        "recovery_entry_count",
        "recovery_exit_count",
        "recovery_gate_off_count",
        "root_x_displacement_m",
        "run_order",
        "seed",
    }
)
_GROUP_FIELDS = frozenset(
    {
        "final_recovered_count",
        "mean_collapsed_action_count",
        "mean_environment_return",
        "mean_first_collapse_or_1001",
        "mean_phase_correction_count",
        "mean_recovery_gate_off_count",
        "mean_root_x_displacement_m",
        "n",
        "no_collapse_count",
    }
)
_FIGURE_COLUMNS = (
    "arm",
    "condition",
    "n",
    "no_collapse_count",
    "mean_collapsed_action_count",
    "mean_environment_return",
    "mean_recovery_gate_off_fraction",
)
_LOCKED_EXPECTATIONS = {
    "nominal": "T1_G1 must retain 20/20 no-collapse outcomes",
    "lateral_velocity": "T1_G1 should exceed T0_G0 by at least two no-collapse outcomes",
    "pitch_velocity_falsifier": (
        "no improvement is assumed; failure identifies missing controller recovery capability"
    ),
    "attribution": ("report all four T x G arms; do not attribute a joint-arm gain to phase alone"),
}


@dataclass(frozen=True)
class LocalMedia:
    """One allowlisted local evidence asset."""

    filename: str
    content_type: str
    expected_sha256: str


class LocalEvidenceError(ValueError):
    """Raised when local exploratory evidence fails closed."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bounded_file(root: Path, relative: Path, *, limit: int) -> tuple[Path, bytes]:
    project_root = root.resolve()
    lexical = project_root / relative
    current = project_root
    for part in relative.parts:
        if part in {"", ".", ".."}:
            raise LocalEvidenceError("unsafe local evidence path")
        current /= part
        if current.is_symlink():
            raise LocalEvidenceError("local evidence symlinks are not allowed")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise LocalEvidenceError("local evidence bundle is incomplete") from exc
    if not resolved.is_relative_to(project_root):
        raise LocalEvidenceError("local evidence path is not a regular project file")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise LocalEvidenceError("local evidence file cannot be read") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise LocalEvidenceError("local evidence path is not a non-empty regular file")
        if before.st_size > limit:
            raise LocalEvidenceError("local evidence file exceeds its size limit")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            source = stream.read(limit + 1)
        after = os.fstat(descriptor)
        identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
            raise LocalEvidenceError("local evidence file changed while it was read")
        if len(source) != before.st_size:
            raise LocalEvidenceError("local evidence file changed while it was read")
        return resolved, source
    finally:
        os.close(descriptor)


def _reject_json_constant(value: str) -> None:
    raise LocalEvidenceError(f"non-finite JSON constant is forbidden: {value}")


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise LocalEvidenceError(f"duplicate local evidence JSON key: {key!r}")
        result[key] = value
    return result


def _json_object(root: Path, filename: str) -> tuple[dict[str, object], bytes]:
    _, source = _bounded_file(root, EXPLORATION_DIR / filename, limit=_JSON_LIMIT)
    try:
        value = json.loads(
            source,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_without_duplicates,
        )
    except LocalEvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise LocalEvidenceError("local evidence JSON is invalid") from exc
    if not isinstance(value, dict):
        raise LocalEvidenceError("local evidence JSON must be an object")
    return value, source


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise LocalEvidenceError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise LocalEvidenceError(f"{field} must be a non-negative integer")
    return value


def _number(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise LocalEvidenceError(f"{field} must be numeric")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise LocalEvidenceError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise LocalEvidenceError(f"{field} must be finite")
    return result


def _sha(value: object, field: str) -> str:
    digest = _string(value, field)
    if len(digest) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise LocalEvidenceError(f"{field} must be a lowercase SHA-256")
    return digest


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise LocalEvidenceError(f"{field} must be an object")
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise LocalEvidenceError(f"{field} must be boolean")
    return value


def _string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > 16:
        raise LocalEvidenceError(f"{field} must be a bounded non-empty list")
    return [_string(item, field) for item in value]


def _decode_object(source: bytes, field: str) -> dict[str, object]:
    try:
        value = json.loads(
            source,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_object_without_duplicates,
        )
    except LocalEvidenceError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise LocalEvidenceError(f"{field} JSON is invalid") from exc
    if not isinstance(value, dict):
        raise LocalEvidenceError(f"{field} must be a JSON object")
    return value


def _run_ledger(
    source: bytes,
    *,
    manifest: Mapping[str, object],
    manifest_sha256: str,
    scheduled: int,
    max_actions: int,
) -> dict[tuple[str, str], list[dict[str, object]]]:
    arms = _mapping(manifest.get("arms"), "manifest.arms")
    paired = _mapping(manifest.get("paired_evaluation"), "manifest.paired_evaluation")
    conditions = _mapping(paired.get("conditions"), "paired_evaluation.conditions")
    seed_values = paired.get("seeds")
    if not isinstance(seed_values, list) or not seed_values or len(seed_values) > 1_000:
        raise LocalEvidenceError("paired_evaluation.seeds must be a bounded non-empty list")
    seeds = tuple(_integer(value, "paired_evaluation seed") for value in seed_values)
    if len(seeds) != len(set(seeds)):
        raise LocalEvidenceError("paired_evaluation.seeds must be unique")
    if scheduled != len(arms) * len(conditions) * len(seeds):
        raise LocalEvidenceError("scheduled runs do not match the declared factorial design")

    lines = source.splitlines()
    if len(lines) != scheduled:
        raise LocalEvidenceError("run ledger row count does not match scheduled runs")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    observed_combinations: set[tuple[str, str, int]] = set()
    run_orders: set[int] = set()
    for index, line in enumerate(lines, start=1):
        if not line or len(line) > 64 * 1024:
            raise LocalEvidenceError("run ledger contains an empty or oversized row")
        row = _decode_object(line, f"run ledger row {index}")
        if set(row) != _RUN_FIELDS:
            raise LocalEvidenceError("run ledger row fields do not match the locked schema")
        if _sha(row.get("manifest_sha256"), "run manifest sha256") != manifest_sha256:
            raise LocalEvidenceError("run ledger row does not match the locked manifest")
        arm = _string(row.get("arm"), "run arm")
        condition = _string(row.get("condition"), "run condition")
        seed = _integer(row.get("seed"), "run seed")
        if arm not in arms or condition not in conditions or seed not in seeds:
            raise LocalEvidenceError("run ledger row is outside the declared factorial design")
        combination = (arm, condition, seed)
        if combination in observed_combinations:
            raise LocalEvidenceError("run ledger contains a duplicate arm/condition/seed")
        observed_combinations.add(combination)
        run_orders.add(_integer(row.get("run_order"), "run order"))

        actions = _integer(row.get("actions_executed"), "actions_executed")
        if actions != max_actions:
            raise LocalEvidenceError("run did not execute the locked action count")
        collapsed = _integer(row.get("collapsed_action_count"), "collapsed_action_count")
        phase_count = _integer(row.get("phase_correction_count"), "phase_correction_count")
        gate_count = _integer(row.get("recovery_gate_off_count"), "recovery_gate_off_count")
        if collapsed > actions or phase_count > actions or gate_count > actions:
            raise LocalEvidenceError("per-run counts exceed actions_executed")
        for field in (
            "action_saturation_count",
            "recovery_entry_count",
            "recovery_exit_count",
        ):
            _integer(row.get(field), field)
        for field in (
            "environment_return",
            "final_root_height_m",
            "final_torso_up_z",
            "root_x_displacement_m",
        ):
            _number(row.get(field), field)
        no_collapse = _boolean(row.get("no_collapse"), "no_collapse")
        _boolean(row.get("final_recovered"), "final_recovered")
        first_collapse = row.get("first_collapse_action")
        if first_collapse is None:
            if not no_collapse or collapsed != 0:
                raise LocalEvidenceError("no-collapse fields are inconsistent")
        else:
            first = _integer(first_collapse, "first_collapse_action")
            if no_collapse or collapsed == 0 or not 1 <= first <= actions:
                raise LocalEvidenceError("collapse fields are inconsistent")
        grouped.setdefault((arm, condition), []).append(row)

    expected_combinations = {
        (arm, condition, seed) for arm in arms for condition in conditions for seed in seeds
    }
    if observed_combinations != expected_combinations:
        raise LocalEvidenceError("run ledger does not cover the declared factorial design")
    if run_orders != set(range(1, scheduled + 1)):
        raise LocalEvidenceError("run_order must be a complete one-based permutation")
    return grouped


def _mean(rows: list[dict[str, object]], field: str) -> float:
    return sum(_number(row[field], field) for row in rows) / len(rows)


def _recomputed_group(
    rows: list[dict[str, object]],
    *,
    max_actions: int,
) -> dict[str, int | float]:
    return {
        "final_recovered_count": sum(bool(row["final_recovered"]) for row in rows),
        "mean_collapsed_action_count": _mean(rows, "collapsed_action_count"),
        "mean_environment_return": _mean(rows, "environment_return"),
        "mean_first_collapse_or_1001": sum(
            max_actions + 1
            if row["first_collapse_action"] is None
            else int(row["first_collapse_action"])
            for row in rows
        )
        / len(rows),
        "mean_phase_correction_count": _mean(rows, "phase_correction_count"),
        "mean_recovery_gate_off_count": _mean(rows, "recovery_gate_off_count"),
        "mean_root_x_displacement_m": _mean(rows, "root_x_displacement_m"),
        "n": len(rows),
        "no_collapse_count": sum(bool(row["no_collapse"]) for row in rows),
    }


def _validate_summary_groups(
    value: object,
    *,
    grouped_rows: Mapping[tuple[str, str], list[dict[str, object]]],
    max_actions: int,
) -> dict[str, Mapping[str, object]]:
    groups = _mapping(value, "summary.groups")
    expected_names = {f"{arm}/{condition}" for arm, condition in grouped_rows}
    if set(groups) != expected_names:
        raise LocalEvidenceError("summary groups do not match the run ledger")
    validated: dict[str, Mapping[str, object]] = {}
    for (arm, condition), rows in grouped_rows.items():
        name = f"{arm}/{condition}"
        observed = _mapping(groups[name], f"summary.groups.{name}")
        if set(observed) != _GROUP_FIELDS:
            raise LocalEvidenceError("summary group fields do not match the locked schema")
        expected = _recomputed_group(rows, max_actions=max_actions)
        for field, expected_value in expected.items():
            observed_value = observed.get(field)
            if isinstance(expected_value, int):
                if _integer(observed_value, f"{name}.{field}") != expected_value:
                    raise LocalEvidenceError("summary integer does not match the run ledger")
            elif not math.isclose(
                _number(observed_value, f"{name}.{field}"),
                expected_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise LocalEvidenceError("summary mean does not match the run ledger")
        validated[name] = observed
    return validated


def _group(groups: Mapping[str, object], arm: str, condition: str) -> Mapping[str, object]:
    return _mapping(groups.get(f"{arm}/{condition}"), f"groups.{arm}/{condition}")


def _media_specs(
    figure_manifest: Mapping[str, object], video_receipt: Mapping[str, object]
) -> dict[str, LocalMedia]:
    outputs = _mapping(figure_manifest.get("outputs"), "figure_manifest.outputs")
    return {
        "holdout-summary.png": LocalMedia(
            filename="holdout_summary.png",
            content_type="image/png",
            expected_sha256=_sha(outputs.get("holdout_summary.png"), "figure png sha256"),
        ),
        "lateral-seed3000-poster.png": LocalMedia(
            filename="lateral_seed3000_poster.png",
            content_type="image/png",
            expected_sha256=_sha(video_receipt.get("poster_sha256"), "poster sha256"),
        ),
        "lateral-seed3000-four-arm.mp4": LocalMedia(
            filename="lateral_seed3000_four_arm.mp4",
            content_type="video/mp4",
            expected_sha256=_sha(video_receipt.get("video_sha256"), "video sha256"),
        ),
    }


def _validate_figure_data(
    source: bytes,
    *,
    groups: Mapping[str, Mapping[str, object]],
    max_actions: int,
) -> None:
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LocalEvidenceError("figure data must be valid UTF-8") from exc
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != _FIGURE_COLUMNS:
        raise LocalEvidenceError("figure data columns do not match the locked schema")
    rows = list(reader)
    if len(rows) != len(groups):
        raise LocalEvidenceError("figure data row count does not match summary groups")
    observed_names: set[str] = set()
    for row in rows:
        name = f"{row['arm']}/{row['condition']}"
        if name in observed_names or name not in groups:
            raise LocalEvidenceError("figure data has an unknown or duplicate group")
        observed_names.add(name)
        group = groups[name]
        integer_fields = ("n", "no_collapse_count")
        for field in integer_fields:
            try:
                observed = int(row[field])
            except (OverflowError, TypeError, ValueError) as exc:
                raise LocalEvidenceError("figure data integer is invalid") from exc
            if observed != _integer(group[field], f"{name}.{field}"):
                raise LocalEvidenceError("figure data integer does not match the run ledger")
        expected_floats = {
            "mean_collapsed_action_count": _number(
                group["mean_collapsed_action_count"], f"{name}.mean_collapsed_action_count"
            ),
            "mean_environment_return": _number(
                group["mean_environment_return"], f"{name}.mean_environment_return"
            ),
            "mean_recovery_gate_off_fraction": _number(
                group["mean_recovery_gate_off_count"], f"{name}.mean_recovery_gate_off_count"
            )
            / max_actions,
        }
        for field, expected in expected_floats.items():
            try:
                observed = float(row[field])
            except (OverflowError, TypeError, ValueError) as exc:
                raise LocalEvidenceError("figure data number is invalid") from exc
            if not math.isfinite(observed) or not math.isclose(
                observed, expected, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise LocalEvidenceError("figure data does not match the run ledger")
    if observed_names != set(groups):
        raise LocalEvidenceError("figure data does not cover every summary group")


def _validate_video_metrics(
    video_receipt: Mapping[str, object],
    *,
    visual: Mapping[str, object],
    grouped_rows: Mapping[tuple[str, str], list[dict[str, object]]],
    first_seed: int,
) -> dict[str, object]:
    if video_receipt.get("claim_ceiling") != (
        "visible_closed_loop_execution_not_formal_oracle_evidence"
    ):
        raise LocalEvidenceError("video claim ceiling does not match the fixed v0 receipt")
    if _integer(video_receipt.get("perturb_action"), "video perturb_action") != 300:
        raise LocalEvidenceError("video perturbation does not match the fixed v0 receipt")
    if _integer(video_receipt.get("render_stride_actions"), "video render stride") != 2:
        raise LocalEvidenceError("video render stride does not match the fixed v0 receipt")
    if video_receipt.get("video_dimensions_px") != [960, 1008]:
        raise LocalEvidenceError("video dimensions do not match the fixed v0 receipt")
    if _integer(video_receipt.get("video_frames"), "video frames") != 501:
        raise LocalEvidenceError("video frame count does not match the fixed v0 receipt")
    if not math.isclose(
        _number(video_receipt.get("video_duration_seconds"), "video duration"),
        15.03,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise LocalEvidenceError("video duration does not match the fixed v0 receipt")
    if not math.isclose(
        _number(video_receipt.get("video_fps"), "video fps"),
        100 / 3,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise LocalEvidenceError("video frame rate does not match the fixed v0 receipt")
    seed = _integer(video_receipt.get("seed"), "video seed")
    condition = _string(video_receipt.get("condition"), "video condition")
    if seed != first_seed or seed != visual.get("seed"):
        raise LocalEvidenceError("video does not use the first receipt-declared evaluation seed")
    if condition != visual.get("condition"):
        raise LocalEvidenceError("video condition does not match the locked visual receipt")
    selection = _string(video_receipt.get("selection"), "video selection")
    selection_rule = _string(visual.get("selection_rule"), "visual selection_rule")
    if selection != selection_rule.split(";", maxsplit=1)[0]:
        raise LocalEvidenceError("video selection does not match the locked selection rule")
    arms = visual.get("arms")
    if not isinstance(arms, list) or not arms or len(arms) != len(set(arms)):
        raise LocalEvidenceError("visual arms must be a unique non-empty list")
    resolved_arms = [_string(arm, "visual arm") for arm in arms]
    if video_receipt.get("arms") != resolved_arms:
        raise LocalEvidenceError("video arms do not match the locked visual receipt")
    metrics = _mapping(video_receipt.get("metrics"), "video metrics")
    if set(metrics) != set(resolved_arms):
        raise LocalEvidenceError("video metrics do not cover the locked visual arms")
    for arm in resolved_arms:
        matches = [row for row in grouped_rows.get((arm, condition), []) if row["seed"] == seed]
        if len(matches) != 1:
            raise LocalEvidenceError("video selection does not identify exactly one run per arm")
        row = matches[0]
        observed = _mapping(metrics[arm], f"video metrics.{arm}")
        if set(observed) != {"collapsed", "first_collapse", "return"}:
            raise LocalEvidenceError("video metric fields do not match the locked schema")
        if _integer(observed["collapsed"], "video collapsed") != row["collapsed_action_count"]:
            raise LocalEvidenceError("video collapse count does not match the run ledger")
        if observed["first_collapse"] != row["first_collapse_action"]:
            raise LocalEvidenceError("video first-collapse value does not match the run ledger")
        if not math.isclose(
            _number(observed["return"], "video return"),
            _number(row["environment_return"], "run return"),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise LocalEvidenceError("video return does not match the run ledger")
    return {
        "seed": seed,
        "condition": condition,
        "selection": "first receipt-declared evaluation seed",
    }


def _validated_bundle(
    root: Path,
) -> tuple[dict[str, object], dict[str, tuple[bytes, str]]]:
    manifest, manifest_bytes = _json_object(root, "manifest.json")
    summary, _ = _json_object(root, "summary.json")
    figure_manifest, figure_manifest_bytes = _json_object(
        root, "holdout_summary.figure_manifest.json"
    )
    video_receipt, video_receipt_bytes = _json_object(
        root, "lateral_seed3000_four_arm.video_receipt.json"
    )
    _, runs_bytes = _bounded_file(root, EXPLORATION_DIR / "runs.jsonl", limit=_RUNS_LIMIT)

    if set(manifest) != _MANIFEST_FIELDS or manifest.get("schema_version") != 1:
        raise LocalEvidenceError("manifest does not match the fixed v0 schema")
    if set(summary) != _SUMMARY_FIELDS:
        raise LocalEvidenceError("summary does not match the fixed v0 schema")
    if set(figure_manifest) != _FIGURE_MANIFEST_FIELDS:
        raise LocalEvidenceError("figure receipt does not match the fixed v0 schema")
    if set(video_receipt) != _VIDEO_RECEIPT_FIELDS:
        raise LocalEvidenceError("video receipt does not match the fixed v0 schema")

    if (
        manifest.get("status") != "local_exploration_only"
        or manifest.get("locked_before_holdout") is not True
    ):
        raise LocalEvidenceError("local exploration lacks its self-attested pre-run lock flag")
    if manifest.get("redistributable") is not False:
        raise LocalEvidenceError("local exploration must retain its redistribution boundary")
    if summary.get("status") != "local_exploration_only":
        raise LocalEvidenceError("local exploration summary has the wrong status")
    if figure_manifest.get("status") != "local_exploration_only":
        raise LocalEvidenceError("local figure has the wrong status")
    if video_receipt.get("status") != "local_exploration_only":
        raise LocalEvidenceError("local video has the wrong status")

    manifest_sha = _sha256(manifest_bytes)
    runs_sha = _sha256(runs_bytes)
    if manifest_sha != _V0_MANIFEST_SHA256:
        raise LocalEvidenceError("manifest identity does not match the reviewed v0 bundle")
    if runs_sha != _V0_RUNS_SHA256:
        raise LocalEvidenceError("run-ledger identity does not match the reviewed v0 bundle")
    receipt_sources = {
        "holdout_summary.figure_manifest.json": figure_manifest_bytes,
        "lateral_seed3000_four_arm.video_receipt.json": video_receipt_bytes,
    }
    if any(
        _sha256(source) != _V0_FILE_SHA256[filename] for filename, source in receipt_sources.items()
    ):
        raise LocalEvidenceError("media receipt identity does not match the reviewed v0 bundle")
    if _sha(summary.get("manifest_sha256"), "summary manifest sha256") != manifest_sha:
        raise LocalEvidenceError("summary does not match the locked manifest")
    if _sha(summary.get("runs_sha256"), "summary runs sha256") != runs_sha:
        raise LocalEvidenceError("summary does not match the run ledger")
    if (
        _sha(figure_manifest.get("source_manifest_sha256"), "figure manifest sha256")
        != manifest_sha
    ):
        raise LocalEvidenceError("figure does not match the locked manifest")
    if _sha(figure_manifest.get("source_runs_sha256"), "figure runs sha256") != runs_sha:
        raise LocalEvidenceError("figure does not match the run ledger")
    if _sha(video_receipt.get("manifest_sha256"), "video manifest sha256") != manifest_sha:
        raise LocalEvidenceError("video does not match the locked manifest")
    if _sha(video_receipt.get("runs_sha256"), "video runs sha256") != runs_sha:
        raise LocalEvidenceError("video does not match the run ledger")

    completed = _integer(summary.get("completed_runs"), "completed_runs")
    scheduled = _integer(summary.get("scheduled_runs"), "scheduled_runs")
    if completed != 300 or scheduled != 300:
        raise LocalEvidenceError("local exploration run ledger is incomplete")

    reference = _mapping(manifest.get("reference"), "manifest.reference")
    if reference.get("admission") != "Tier-K_not_admitted":
        raise LocalEvidenceError("local reference must remain Tier-K and not admitted")
    visual = _mapping(manifest.get("visual_evidence"), "manifest.visual_evidence")
    arms = _mapping(manifest.get("arms"), "manifest.arms")
    paired = _mapping(manifest.get("paired_evaluation"), "manifest.paired_evaluation")
    conditions = _mapping(paired.get("conditions"), "paired_evaluation.conditions")
    if set(arms) != set(_V0_ARMS):
        raise LocalEvidenceError("manifest arms do not match the fixed v0 design")
    if set(conditions) != set(_V0_CONDITIONS):
        raise LocalEvidenceError("manifest conditions do not match the fixed v0 design")
    if paired.get("seeds") != list(_V0_SEEDS):
        raise LocalEvidenceError("manifest seeds do not match the fixed v0 design")
    if visual.get("arms") != list(_V0_FACTORIAL_ARMS):
        raise LocalEvidenceError("visual arms do not cover the four fixed v0 factorial arms")
    max_actions = _integer(
        _mapping(manifest.get("frozen_runtime"), "manifest.frozen_runtime").get("max_actions"),
        "max_actions",
    )
    if max_actions == 0:
        raise LocalEvidenceError("max_actions must be positive")
    grouped_rows = _run_ledger(
        runs_bytes,
        manifest=manifest,
        manifest_sha256=manifest_sha,
        scheduled=scheduled,
        max_actions=max_actions,
    )
    groups = _validate_summary_groups(
        summary.get("groups"),
        grouped_rows=grouped_rows,
        max_actions=max_actions,
    )

    outputs = _mapping(figure_manifest.get("outputs"), "figure_manifest.outputs")
    if figure_manifest.get("source_manifest") != "manifest.json":
        raise LocalEvidenceError("figure receipt names the wrong manifest")
    if figure_manifest.get("source_runs") != "runs.jsonl":
        raise LocalEvidenceError("figure receipt names the wrong run ledger")
    expected_outputs = {
        "figure_data.csv",
        "holdout_summary.alt.txt",
        "holdout_summary.png",
        "holdout_summary.svg",
    }
    if set(outputs) != expected_outputs:
        raise LocalEvidenceError("figure output receipt is incomplete")
    figure_sources: dict[str, bytes] = {}
    for filename in expected_outputs:
        limit = _MEDIA_LIMIT if filename.endswith((".png", ".svg")) else _JSON_LIMIT
        _, source = _bounded_file(root, EXPLORATION_DIR / filename, limit=limit)
        if _sha256(source) != _sha(outputs.get(filename), f"{filename} sha256"):
            raise LocalEvidenceError(f"{filename} does not match its receipt")
        if _sha256(source) != _V0_FILE_SHA256[filename]:
            raise LocalEvidenceError(f"{filename} identity does not match the reviewed v0 bundle")
        figure_sources[filename] = source
    if not figure_sources["holdout_summary.png"].startswith(b"\x89PNG\r\n\x1a\n"):
        raise LocalEvidenceError("holdout figure is not a PNG")
    svg_prefix = figure_sources["holdout_summary.svg"][:512].lstrip()
    if not (
        svg_prefix.startswith(b"<svg")
        or (svg_prefix.startswith(b"<?xml") and b"<svg" in svg_prefix)
    ):
        raise LocalEvidenceError("holdout vector figure is not an SVG")
    try:
        alt_text = figure_sources["holdout_summary.alt.txt"].decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise LocalEvidenceError("holdout alt text is not UTF-8") from exc
    if not alt_text:
        raise LocalEvidenceError("holdout alt text is empty")
    _validate_figure_data(
        figure_sources["figure_data.csv"],
        groups=groups,
        max_actions=max_actions,
    )

    seed_values = paired.get("seeds")
    if not isinstance(seed_values, list) or not seed_values:
        raise LocalEvidenceError("paired_evaluation.seeds must be non-empty")
    video_selection = _validate_video_metrics(
        video_receipt,
        visual=visual,
        grouped_rows=grouped_rows,
        first_seed=_integer(seed_values[0], "first evaluation seed"),
    )

    media_specs = _media_specs(figure_manifest, video_receipt)
    media_payloads: dict[str, tuple[bytes, str]] = {}
    for route_name, item in media_specs.items():
        if item.filename == "holdout_summary.png":
            source = figure_sources[item.filename]
        else:
            _, source = _bounded_file(
                root,
                EXPLORATION_DIR / item.filename,
                limit=_MEDIA_LIMIT,
            )
        if _sha256(source) != item.expected_sha256:
            raise LocalEvidenceError(f"{item.filename} does not match its receipt")
        if _sha256(source) != _V0_FILE_SHA256[item.filename]:
            raise LocalEvidenceError(
                f"{item.filename} identity does not match the reviewed v0 bundle"
            )
        if item.content_type == "image/png" and not source.startswith(b"\x89PNG\r\n\x1a\n"):
            raise LocalEvidenceError(f"{item.filename} is not a PNG")
        if item.content_type == "video/mp4" and (len(source) < 12 or source[4:8] != b"ftyp"):
            raise LocalEvidenceError(f"{item.filename} is not an MP4")
        media_payloads[route_name] = (source, item.content_type)

    comparisons: list[dict[str, object]] = []
    locked_expectations = _mapping(
        manifest.get("locked_expectations"), "manifest.locked_expectations"
    )
    if dict(locked_expectations) != _LOCKED_EXPECTATIONS:
        raise LocalEvidenceError("locked expectations do not match the v0 display contract")
    expectations = {
        "nominal": ("retain 20/20 no-collapse outcomes", 20),
        "lateral_velocity": ("gain at least 2 no-collapse outcomes", 2),
        "pitch_velocity_falsifier": ("descriptive falsifier; no gain assumed", None),
    }
    for condition, (expectation, target) in expectations.items():
        baseline = _group(groups, "T0_G0", condition)
        candidate = _group(groups, "T1_G1", condition)
        baseline_count = _integer(baseline.get("no_collapse_count"), "baseline no-collapse")
        candidate_count = _integer(candidate.get("no_collapse_count"), "candidate no-collapse")
        baseline_n = _integer(baseline.get("n"), "baseline group n")
        candidate_n = _integer(candidate.get("n"), "candidate group n")
        if baseline_n == 0 or baseline_n != candidate_n:
            raise LocalEvidenceError("comparison arms must have the same positive sample count")
        delta = candidate_count - baseline_count
        if condition == "nominal":
            passed: bool | None = candidate_count == target
        elif target is None:
            passed = None
        else:
            passed = delta >= target
        comparisons.append(
            {
                "condition": condition,
                "baseline_no_collapse": baseline_count,
                "candidate_no_collapse": candidate_count,
                "n": candidate_n,
                "delta": delta,
                "expectation": expectation,
                "passed": passed,
            }
        )

    nominal_candidate = _group(groups, "T1_G1", "nominal")
    return (
        {
            "state": "available",
            "authority": "local_exploration_only",
            "claim_ceiling": _string(manifest.get("claim_ceiling"), "claim_ceiling"),
            "question": _string(manifest.get("question"), "question"),
            "completed_runs": completed,
            "scheduled_runs": scheduled,
            "reference_admission": _string(reference.get("admission"), "reference admission"),
            "not_admitted_reasons": _string_list(
                manifest.get("reason_not_admitted"), "reason_not_admitted"
            ),
            "audit_limitations": [
                "The claimed pre-run lock chronology is self-attested; no earlier committed hash exists.",
                "The original runner and phase-oracle source were not retained or bound.",
                "The locked manifest does not bind a preserved evaluator source artifact.",
                "The referenced projection receipt is unavailable and differs from the current registered-import receipt.",
                "No canonical per-step trajectory trace was retained for this study.",
            ],
            "visual": video_selection,
            "comparisons": comparisons,
            "diagnostics": {
                "nominal_phase_correction_percent": round(
                    100
                    * _number(
                        nominal_candidate.get("mean_phase_correction_count"),
                        "nominal phase correction count",
                    )
                    / max_actions,
                    2,
                ),
                "nominal_fallback_percent": round(
                    100
                    * _number(
                        nominal_candidate.get("mean_recovery_gate_off_count"),
                        "nominal fallback count",
                    )
                    / max_actions,
                    2,
                ),
            },
            "alt_text": alt_text,
            "media": {
                "figure": "/local-evidence/holdout-summary.png",
                "poster": "/local-evidence/lateral-seed3000-poster.png",
                "video": "/local-evidence/lateral-seed3000-four-arm.mp4",
            },
            "receipts": {
                "manifest_sha256": manifest_sha,
                "runs_sha256": runs_sha,
                "video_sha256": media_specs["lateral-seed3000-four-arm.mp4"].expected_sha256,
            },
        },
        media_payloads,
    )


def local_exploration_status(project_root: Path) -> dict[str, object]:
    """Return a fail-closed view of the optional local exploration."""

    evidence_dir = project_root.resolve() / EXPLORATION_DIR
    if not evidence_dir.exists():
        return {
            "state": "unavailable",
            "authority": "local_exploration_only",
            "detail": "no local exploration bundle is present",
        }
    try:
        summary, _ = _validated_bundle(project_root)
    except LocalEvidenceError as exc:
        return {
            "state": "rejected",
            "authority": "local_exploration_only",
            "detail": str(exc),
        }
    return summary


def local_media(project_root: Path, route_name: str) -> tuple[bytes, str]:
    """Read one integrity-checked, allowlisted local media route."""

    if PurePosixPath(route_name).name != route_name:
        raise LocalEvidenceError("unknown local evidence asset")
    _, media = _validated_bundle(project_root)
    payload = media.get(route_name)
    if payload is None:
        raise LocalEvidenceError("unknown local evidence asset")
    return payload
