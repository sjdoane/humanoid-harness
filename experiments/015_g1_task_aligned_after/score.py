"""Fail-closed offline scorer for the Study 015 O7/O7b probe contrast."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean

import numpy as np

from oracle_composition.adapters.gmt.io import write_json_receipt
from oracle_composition.harness.contract import decode_json_object, read_json_object

BASE_SOURCE_COMMIT = "031a0660401bda7cf887fab68eea686980c9a4fd"
BASE_MANIFEST_SHA256 = "6141e860a018fcb0dfac89444234fe987351f188f29e14d3400d0a78db9fe34c"
BASE_RESOURCE_SHA256 = "6fa316fc8b15eccf29e42920ac48914cf1baf67f885af0e93e2b1916d06db466"
BASE_CONFIG_SHA256 = "430c6674485a4ec9ca7546389a71b485c7670f11d08a8b911b48758ee6a51ff4"
CANDIDATE_CONFIG_SHA256 = "ba45dba36ef88bdc522ed2115e46a4b3d876e00a627a089fd56ddf0de623e18a"
PREFIX_ACTION_COUNT = 263
PREFIX_STATE_COUNT = 264
_HEX = frozenset("0123456789abcdef")
_OUTPUTS = {
    "input_config.json",
    "zero_residual_evaluation.json",
    "zero_residual_frames.jsonl",
    "zero_residual_trajectory.npz",
}
_COMPARABLE_OUTPUTS = _OUTPUTS - {"input_config.json"}
_DEVELOPMENT_GATES = {
    "finish_reached",
    "full_horizon_without_fall",
    "inside_mean_speed_target",
    "inside_posture_compliance",
    "inside_posture_dip",
    "joint_position_rmse_p95",
    "maximum_lateral_error",
    "mean_speed_error",
    "minimum_inside_samples",
    "region_entry_and_exit_observed",
    "roll_pitch_rmse_p95",
}


@dataclass(frozen=True, slots=True)
class RunPins:
    root: Path
    manifest_sha256: str
    resource_sha256: str
    config_sha256: str
    source_commit: str
    authority_path: Path | None = None
    authority_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class StudyInputs:
    baseline: RunPins
    control: RunPins
    candidate: RunPins
    candidate_config_sha256: str


def _digest(value: object, *, length: int, field: str) -> str:
    if type(value) is not str or len(value) != length or any(c not in _HEX for c in value):
        raise ValueError(f"{field} must be one lowercase hexadecimal digest")
    return value


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _pinned_json(path: Path, expected_sha256: str, *, source: str) -> tuple[dict, bytes]:
    expected = _digest(expected_sha256, length=64, field=f"{source} SHA-256")
    value, encoded = read_json_object(path)
    if _sha256(encoded) != expected:
        raise ValueError(f"{source} bytes differ from their pinned SHA-256")
    return value, encoded


def _output_bytes(root: Path, outputs: dict) -> dict[str, bytes]:
    if set(outputs) != _OUTPUTS:
        raise ValueError("probe output ledger differs from the exact four-file contract")
    retained = {}
    for name in sorted(outputs):
        digest = _digest(outputs[name], length=64, field=f"output {name}")
        path = root / name
        encoded = path.read_bytes()
        if _sha256(encoded) != digest:
            raise ValueError(f"retained output bytes differ: {name}")
        retained[name] = encoded
    return retained


def _verify_resource(
    *,
    pins: RunPins,
    manifest: dict,
    manifest_size: int,
    resource: dict,
    authority: dict | None,
) -> dict:
    if (
        resource.get("schema_version") != 1
        or resource.get("artifact") != "gmt_g1_course_development_resource_receipt"
        or resource.get("status") != "succeeded"
        or resource.get("evidence_class") != "development_course_probe_not_task_success"
        or resource.get("commit") != pins.source_commit
    ):
        raise ValueError("resource receipt status, identity, or source commit differs")
    artifacts = resource.get("artifacts")
    inputs = resource.get("inputs")
    manifest_record = artifacts.get("course_manifest") if type(artifacts) is dict else None
    output_records = artifacts.get("outputs") if type(artifacts) is dict else None
    config_record = inputs.get("config") if type(inputs) is dict else None
    repository = inputs.get("repository_sources") if type(inputs) is dict else None
    outputs = manifest.get("outputs")
    if (
        type(inputs) is not dict
        or inputs.get("workload") != "course"
        or inputs.get("course_mode") != "probe"
        or inputs.get("artifact_contract") != "gmt_g1_fixed_development_launcher/v1"
        or type(config_record) is not dict
        or set(config_record) != {"path", "sha256", "size"}
        or type(config_record.get("path")) is not str
        or config_record.get("sha256") != pins.config_sha256
        or config_record.get("size") != (pins.root / "input_config.json").stat().st_size
        or type(manifest_record) is not dict
        or manifest_record
        != {
            "path": "course_run_manifest.json",
            "sha256": pins.manifest_sha256,
            "size": manifest_size,
        }
        or type(outputs) is not dict
        or type(output_records) is not dict
        or set(output_records) != set(outputs)
        or type(repository) is not dict
        or set(repository) != {"canonical_tree_sha256", "file_count"}
        or type(repository.get("file_count")) is not int
        or repository["file_count"] < 1
    ):
        raise ValueError("resource receipt does not bind the exact probe inputs and manifest")
    for name, digest in outputs.items():
        record = output_records[name]
        if type(record) is not dict or record != {
            "path": name,
            "sha256": digest,
            "size": (pins.root / name).stat().st_size,
        }:
            raise ValueError(f"resource output record differs: {name}")
    tree = _digest(
        repository.get("canonical_tree_sha256"),
        length=64,
        field="repository source tree",
    )
    argv = resource.get("canonical_argv")
    if (
        type(argv) is not list
        or len(argv) != 7
        or argv[1:4] != ["-m", "oracle_composition.adapters.gmt.course_run", "--config"]
        or argv[4] != config_record.get("path")
        or argv[5] != "--output"
        or type(argv[6]) is not str
        or Path(argv[6]).resolve() != pins.root
    ):
        raise ValueError("resource command does not bind the exact config and output")
    if authority is not None and (
        authority.get("commit") != pins.source_commit
        or authority.get("inputs") != inputs
        or authority.get("canonical_argv") != argv
        or authority.get("output") != str(pins.root)
        or authority.get("mode") != "smoke"
        or authority.get("required_authorizer") != "fable"
    ):
        raise ValueError("fresh authority does not exactly bind the successful resource run")
    return {
        "source_tree_sha256": tree,
        "fixed_inputs": {name: value for name, value in inputs.items() if name != "config"},
    }


def _feedback_builder(**kwargs):
    from oracle_composition.feedback.g1_course import build_g1_course_feedback

    return build_g1_course_feedback(**kwargs)


def _frames(encoded: bytes) -> list[dict]:
    rows = [
        decode_json_object(line, source=f"Study015 frame {index}")
        for index, line in enumerate(encoded.splitlines(), start=1)
    ]
    if not 1 <= len(rows) <= 1000:
        raise ValueError("Study015 frame count is outside the fixed probe horizon")
    return rows


def _trajectory(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "composite_raw_action",
            "current_reference",
            "qpos",
            "qvel",
            "residual_action",
        }:
            raise ValueError("Study015 trajectory fields differ")
        return {name: np.ascontiguousarray(archive[name]) for name in archive.files}


def _require_zero_residual(report: dict, trajectory: dict[str, np.ndarray]) -> None:
    residual_rms = report.get("residual_rms")
    if (
        type(residual_rms) is not float
        or residual_rms != 0.0
        or np.count_nonzero(trajectory["residual_action"]) != 0
    ):
        raise ValueError("Study015 requires an exact zero-residual probe")


def _verify_run(pins: RunPins, feedback_root: Path) -> dict:
    root = pins.root.resolve(strict=True)
    if not root.is_dir() or root != pins.root:
        raise ValueError("run directory must be an exact absolute directory")
    _digest(pins.source_commit, length=40, field="source commit")
    manifest_path = root / "course_run_manifest.json"
    manifest, manifest_bytes = _pinned_json(
        manifest_path, pins.manifest_sha256, source="course manifest"
    )
    resource, _ = _pinned_json(
        root / "gmt_probe_resource_receipt_v1.json",
        pins.resource_sha256,
        source="resource receipt",
    )
    config, config_bytes = _pinned_json(
        root / "input_config.json", pins.config_sha256, source="input config"
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("artifact") != "gmt_g1_course_development_run"
        or manifest.get("status") != "completed"
        or manifest.get("input_config_sha256") != pins.config_sha256
        or config.get("mode") != "probe"
        or config.get("seed") != 20260906
        or config.get("training_steps") != 0
    ):
        raise ValueError("run manifest or fixed probe config differs")
    outputs = manifest.get("outputs")
    if type(outputs) is not dict or outputs.get("input_config.json") != pins.config_sha256:
        raise ValueError("manifest output ledger does not bind its input config")
    authority = None
    if pins.authority_path is not None or pins.authority_sha256 is not None:
        if pins.authority_path is None or pins.authority_sha256 is None:
            raise ValueError("fresh run requires both authority path and SHA-256")
        authority, _ = _pinned_json(
            pins.authority_path,
            pins.authority_sha256,
            source="fresh authority",
        )
    resource_binding = _verify_resource(
        pins=pins,
        manifest=manifest,
        manifest_size=len(manifest_bytes),
        resource=resource,
        authority=authority,
    )
    feedback = _feedback_builder(
        manifest_path=manifest_path,
        expected_manifest_sha256=pins.manifest_sha256,
        label="zero_residual",
        output=feedback_root,
    )
    retained = _output_bytes(root, outputs)
    report = manifest.get("zero_residual")
    trajectory = _trajectory(root / "zero_residual_trajectory.npz")
    if type(report) is not dict:
        raise ValueError("zero-residual report is absent")
    _require_zero_residual(report, trajectory)
    objective = report.get("objective_evaluation")
    if type(objective) is not dict or objective.get("episode_success") is not None:
        raise ValueError("probe objective is absent or improperly promotes task success")
    return {
        "pins": pins,
        "config": config,
        "config_bytes": config_bytes,
        "outputs": retained,
        "frames": _frames(retained["zero_residual_frames.jsonl"]),
        "trajectory": trajectory,
        "objective": objective,
        "source_tree_sha256": resource_binding["source_tree_sha256"],
        "resource_fixed_inputs": resource_binding["fixed_inputs"],
        "feedback": {
            name: {
                "sha256": record["sha256"],
                "byte_count": record["byte_count"],
            }
            for name, record in feedback.items()
        },
    }


def verify_o7b_config(control: dict, candidate: dict) -> None:
    expected = deepcopy(control)
    oracle = expected["oracle"]
    segments = expected["segments"]
    if type(oracle) is not dict or type(segments) is not dict:
        raise ValueError("control oracle or segments are malformed")
    if (
        oracle.get("oracle_id") != "g1_course_o7_balanced_loop_rise"
        or oracle.get("behaviors") != ["walk", "crouch", "rise"]
        or set(segments) != {"walk", "crouch", "rise"}
    ):
        raise ValueError("control is not the pinned O7 oracle")
    oracle["oracle_id"] = "g1_course_o7b_balanced_loop_rise_after_crop"
    oracle["behaviors"] = ["walk", "crouch", "rise", "walk_after"]
    oracle["states"]["after"]["behavior"] = "walk_after"
    segments["walk_after"] = {
        "boundary": "wrap_within_segment",
        "end_seconds": 31.56,
        "motion_name": "basic_walk",
        "start_seconds": 30.62,
    }
    if candidate != expected:
        raise ValueError("candidate differs from the exact O7b after-crop change")


def _first_after_index(frames: list[dict]) -> int | None:
    return next(
        (index for index, row in enumerate(frames) if row.get("executed_mode") == "after"),
        None,
    )


def verify_prefix(control: dict, candidate: dict) -> dict[str, int]:
    control_frames = control["frames"]
    candidate_frames = candidate["frames"]
    if (
        _first_after_index(control_frames) != PREFIX_ACTION_COUNT
        or _first_after_index(candidate_frames) != PREFIX_ACTION_COUNT
    ):
        raise ValueError("first after command is not the preregistered action index 263")
    for index in range(PREFIX_ACTION_COUNT):
        left = {key: value for key, value in control_frames[index].items() if key != "reward"}
        right = {key: value for key, value in candidate_frames[index].items() if key != "reward"}
        if left != right:
            raise ValueError(f"nonreward trace prefix differs before after command: {index}")
    for name, count in (
        ("composite_raw_action", PREFIX_ACTION_COUNT),
        ("qpos", PREFIX_STATE_COUNT),
        ("qvel", PREFIX_STATE_COUNT),
    ):
        if not np.array_equal(
            control["trajectory"][name][:count], candidate["trajectory"][name][:count]
        ):
            raise ValueError(f"numeric prefix differs before the O7b after crop: {name}")
    return {"equal_action_count": PREFIX_ACTION_COUNT, "equal_state_row_count": PREFIX_STATE_COUNT}


def _yaw_wxyz(qpos: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(qpos[:, 3:7], dtype=np.float64)
    if quaternion.ndim != 2 or quaternion.shape[1] != 4 or not np.isfinite(quaternion).all():
        raise ValueError("trajectory orientations must be finite WXYZ quaternions")
    w, x, y, z = quaternion.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def candidate_measures(
    frames: list[dict], trajectory: dict[str, np.ndarray], objective: dict, task: dict
) -> dict:
    qpos = trajectory["qpos"]
    current_reference = trajectory["current_reference"]
    if qpos.shape != (len(frames) + 1, 30):
        raise ValueError("qpos rows must include one initial state plus every frame")
    if current_reference.shape != (len(frames), 30) or not np.isfinite(current_reference).all():
        raise ValueError("current references must cover every finite retained frame")
    entry = task.get("region_entry_distance_m")
    exit_ = task.get("region_exit_distance_m")
    if type(entry) is not float or type(exit_) is not float or not 0 < entry < exit_:
        raise ValueError("physical task region differs")
    progress = []
    for row in frames:
        value = row.get("metrics", {}).get("progress_m")
        if type(value) is not float or not math.isfinite(value):
            raise ValueError("frame progress must be finite")
        progress.append(value)
    inside = [entry <= value < exit_ for value in progress]
    first_entry = next((i for i, value in enumerate(inside) if value), None)
    first_exit = (
        next((i for i in range(first_entry + 1, len(frames)) if progress[i] >= exit_), None)
        if first_entry is not None
        else None
    )
    first_cross_count = (
        sum(inside[first_entry:first_exit])
        if first_entry is not None and first_exit is not None
        else sum(inside)
    )
    later_count = sum(inside[first_exit:]) if first_exit is not None else 0
    all_count = sum(inside)
    region = objective.get("region")
    if type(region) is not dict or region.get("inside_sample_count") != all_count:
        raise ValueError("all-visit physical-region count differs from the objective evaluator")
    after_index = _first_after_index(frames)
    rise_index = next(
        (index for index, row in enumerate(frames) if row.get("executed_mode") == "rise"),
        None,
    )
    unwrapped = np.unwrap(_yaw_wxyz(qpos))
    relative = unwrapped - unwrapped[0]
    after_heading = relative[after_index:] if after_index is not None else np.asarray([])
    after_excursion = (
        unwrapped[after_index:] - unwrapped[after_index]
        if after_index is not None
        else np.asarray([])
    )
    after_rows = frames[after_index:] if after_index is not None else []
    after_reference = (
        current_reference[after_index:, 6] if after_index is not None else np.asarray([])
    )
    after_progress = [row["metrics"]["progress_m"] for row in after_rows]
    return {
        "first_rise_action_index": rise_index,
        "first_after_action_index": after_index,
        "first_cross_inside_sample_count": first_cross_count,
        "later_revisit_inside_sample_count": later_count,
        "all_visit_inside_sample_count": all_count,
        "post_after_region_reentry_count": (
            sum(inside[after_index:]) if after_index is not None else None
        ),
        "after_unwrapped_heading_absolute_max_rad": (
            float(np.max(np.abs(after_heading))) if after_heading.size else None
        ),
        "after_entry_relative_heading_excursion_max_rad": (
            float(np.max(np.abs(after_excursion))) if after_excursion.size else None
        ),
        "after_unwrapped_heading_final_rad": (float(relative[-1]) if after_heading.size else None),
        "crouch_exit_boundary_heading_rad": (
            float(relative[rise_index]) if rise_index is not None else None
        ),
        "after_entry_boundary_heading_rad": (
            float(relative[after_index]) if after_index is not None else None
        ),
        "after_commanded_local_yaw_rate_mean_rad_s": (
            float(np.mean(after_reference, dtype=np.float64)) if after_reference.size else None
        ),
        "after_speed_mae_m_s": (
            fmean(row["metrics"]["speed_error_m_s"] for row in after_rows) if after_rows else None
        ),
        "after_progress_start_m": after_progress[0] if after_progress else None,
        "after_progress_final_m": after_progress[-1] if after_progress else None,
        "after_progress_max_m": max(after_progress) if after_progress else None,
    }


def mechanism_screen(objective: dict, measures: dict) -> dict[str, bool]:
    original = objective.get("development_gate_results")
    tracking = objective.get("tracking")
    region = objective.get("region")
    diagnostics = objective.get("oracle_diagnostics")
    if (
        type(original) is not dict
        or set(original) != _DEVELOPMENT_GATES
        or type(tracking) is not dict
        or type(region) is not dict
        or type(diagnostics) is not dict
    ):
        raise ValueError("objective evaluator fields or original eleven gates differ")
    heading = measures["after_unwrapped_heading_absolute_max_rad"]
    reentries = measures["post_after_region_reentry_count"]
    return {
        "full_20s_without_fall": objective.get("duration_seconds") == 20.0
        and objective.get("fall_count") == 0,
        "three_switches": diagnostics.get("observed_switch_count") == 3,
        "finish_reached_3p5m": type(objective.get("maximum_progress_m")) is float
        and objective["maximum_progress_m"] >= 3.5,
        "physical_region_entry_observed": region.get("entry_observed") is True,
        "rise_state_executed": measures["first_rise_action_index"] is not None,
        "after_state_executed": measures["first_after_action_index"] is not None,
        "no_post_after_region_reentry": reentries == 0,
        "after_unwrapped_heading_below_1rad": type(heading) is float and heading < 1.0,
        "lateral_below_2p5m": type(objective.get("maximum_lateral_error_m")) is float
        and objective["maximum_lateral_error_m"] < 2.5,
        "joint_tracking_p95_at_most_0p35rad": type(tracking.get("joint_position_rmse_rad_p95"))
        is float
        and tracking["joint_position_rmse_rad_p95"] <= 0.35,
        "roll_pitch_tracking_p95_at_most_0p25rad": type(tracking.get("roll_pitch_rmse_rad_p95"))
        is float
        and tracking["roll_pitch_rmse_rad_p95"] <= 0.25,
    }


RunVerifier = Callable[[RunPins, Path], dict]


def score_study(inputs: StudyInputs, *, run_verifier: RunVerifier = _verify_run) -> dict:
    if (
        inputs.baseline.manifest_sha256 != BASE_MANIFEST_SHA256
        or inputs.baseline.resource_sha256 != BASE_RESOURCE_SHA256
        or inputs.baseline.config_sha256 != BASE_CONFIG_SHA256
        or inputs.baseline.source_commit != BASE_SOURCE_COMMIT
        or inputs.control.config_sha256 != BASE_CONFIG_SHA256
        or inputs.candidate.config_sha256 != inputs.candidate_config_sha256
        or inputs.candidate_config_sha256 != CANDIDATE_CONFIG_SHA256
    ):
        raise ValueError("Study015 fixed baseline or candidate identities differ")
    if (
        inputs.control.source_commit != inputs.candidate.source_commit
        or inputs.control.authority_path is None
        or inputs.candidate.authority_path is None
    ):
        raise ValueError("fresh control and candidate require one common authorized source")
    with tempfile.TemporaryDirectory(prefix="study015-feedback-") as raw:
        feedback_root = Path(raw).resolve(strict=True)
        baseline = run_verifier(inputs.baseline, feedback_root / "baseline")
        control = run_verifier(inputs.control, feedback_root / "control")
        candidate = run_verifier(inputs.candidate, feedback_root / "candidate")
    if baseline["config_bytes"] != control["config_bytes"]:
        raise ValueError("fresh control config bytes differ from retained O7")
    for name in _COMPARABLE_OUTPUTS:
        if baseline["outputs"][name] != control["outputs"][name]:
            raise ValueError(f"fresh control does not byte-reproduce retained O7: {name}")
    if (
        control["source_tree_sha256"] != candidate["source_tree_sha256"]
        or control["resource_fixed_inputs"] != candidate["resource_fixed_inputs"]
        or control["pins"].source_commit != candidate["pins"].source_commit
    ):
        raise ValueError("fresh control and candidate runtime or executable source differs")
    verify_o7b_config(control["config"], candidate["config"])
    prefix = verify_prefix(control, candidate)
    measures = candidate_measures(
        candidate["frames"],
        candidate["trajectory"],
        candidate["objective"],
        candidate["config"]["task"],
    )
    screen = mechanism_screen(candidate["objective"], measures)
    return {
        "schema_version": 1,
        "study": "015",
        "artifact": "gmt_g1_o7_after_crop_development_score/v1",
        "comparison": "retained_o7_vs_fresh_o7_control_vs_o7b_after_crop",
        "source_commit": inputs.control.source_commit,
        "source_tree_sha256": control["source_tree_sha256"],
        "candidate_config_sha256": inputs.candidate_config_sha256,
        "fresh_control_byte_reproduced_three_outputs": True,
        "prefix": prefix,
        "candidate_measures": measures,
        "mechanism_screen": screen,
        "mechanism_screen_passed": all(screen.values()),
        "original_development_gate_results": candidate["objective"]["development_gate_results"],
        "original_objective": candidate["objective"],
        "run_bindings": {
            name: {
                "manifest_sha256": run["pins"].manifest_sha256,
                "resource_sha256": run["pins"].resource_sha256,
                "config_sha256": run["pins"].config_sha256,
                "authority_sha256": run["pins"].authority_sha256,
                "feedback": run["feedback"],
            }
            for name, run in (
                ("baseline", baseline),
                ("control", control),
                ("candidate", candidate),
            )
        },
        "claim_scope": "one_seed_zero_residual_development_screen_not_task_success",
        "claim_limits": [
            "all_visits_are_authoritative_for_task_region_metrics",
            "first_cross_and_later_revisit_counts_are_descriptive_partitions",
            "unchanged_first_cross_depth_and_lateral_behavior_was_not_predicted_to_improve",
            "no_heldout_generalization_or_dynamics_certification_claim",
        ],
    }


def score_and_publish(
    inputs: StudyInputs,
    output: Path,
    *,
    run_verifier: RunVerifier = _verify_run,
) -> tuple[dict, str]:
    result = score_study(inputs, run_verifier=run_verifier)
    result["scorer_sha256"] = _sha256(Path(__file__).read_bytes())
    digest = write_json_receipt(output, result)
    return result, digest


def _arguments(argv: list[str] | None = None) -> tuple[StudyInputs, Path]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--baseline-manifest-sha256", required=True)
    parser.add_argument("--baseline-resource-sha256", required=True)
    parser.add_argument("--baseline-config-sha256", required=True)
    for name in ("control", "candidate"):
        parser.add_argument(f"--{name}-dir", type=Path, required=True)
        parser.add_argument(f"--{name}-manifest-sha256", required=True)
        parser.add_argument(f"--{name}-resource-sha256", required=True)
        parser.add_argument(f"--{name}-authority", type=Path, required=True)
        parser.add_argument(f"--{name}-authority-sha256", required=True)
    parser.add_argument("--control-config-sha256", required=True)
    parser.add_argument("--candidate-config-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    baseline = RunPins(
        args.baseline_dir.resolve(),
        args.baseline_manifest_sha256,
        args.baseline_resource_sha256,
        args.baseline_config_sha256,
        BASE_SOURCE_COMMIT,
    )
    control = RunPins(
        args.control_dir.resolve(),
        args.control_manifest_sha256,
        args.control_resource_sha256,
        args.control_config_sha256,
        args.source_commit,
        args.control_authority.resolve(),
        args.control_authority_sha256,
    )
    candidate = RunPins(
        args.candidate_dir.resolve(),
        args.candidate_manifest_sha256,
        args.candidate_resource_sha256,
        args.candidate_config_sha256,
        args.source_commit,
        args.candidate_authority.resolve(),
        args.candidate_authority_sha256,
    )
    return StudyInputs(baseline, control, candidate, args.candidate_config_sha256), args.output


def main(argv: list[str] | None = None) -> None:
    inputs, output = _arguments(argv)
    result, digest = score_and_publish(inputs, output)
    print(
        json.dumps(
            {
                "path": str(output),
                "sha256": digest,
                "mechanism_screen": result["mechanism_screen"],
                "mechanism_screen_passed": result["mechanism_screen_passed"],
                "candidate_measures": result["candidate_measures"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
