"""Offline Study016 score; shared probe verification, fixed feedback contrast."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np

from oracle_composition.adapters.gmt.course_task import TaskFrame
from oracle_composition.adapters.gmt.io import write_json_receipt
from oracle_composition.harness.contract import read_json_object

_SHARED_PATH = Path(__file__).resolve().parents[1] / "015_g1_task_aligned_after/score.py"
_SPEC = importlib.util.spec_from_file_location("study016_shared_probe_verifier", _SHARED_PATH)
shared = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = shared
_SPEC.loader.exec_module(shared)

BASE_MANIFEST = "92e2a72f743bded8b2694760c7492f8d35f0829489415a05c3afff6ec459944a"
BASE_RESOURCE = "ea00072569e159b8f04b1b7c801296b19297559b0ed91f04ec4077551cb429b8"
BASE_CONFIG = "ba45dba36ef88bdc522ed2115e46a4b3d876e00a627a089fd56ddf0de623e18a"
BASE_SOURCE = "3cb3102d5cb2e3a8603663df13ed50b5d6cad0cb"
BASE_TREE = "3ba7fe0ab72374e0667402848460c43915024fbcd2fe44f9ce5de791f0ab8b8e"
FRESH_TREE = "c4f5a24e285c01bde06d0c5530631084284eacdf22106d614b6e5ad228d0174d"
FRESH_FILE_COUNT = 195
CANDIDATE_CONFIG = "8fe33d993c6226690e986e99de3d0d1ee436424239069c4cceb91c1a8fabfc8e"
PROFILE = "gmt_g1_four_state_loop_after_heading_feedback_course/v1"
GYM_PROFILE = "gmt_g1_residual_course_four_state_after_heading_feedback_50hz/v1"
TRACE_KEY = "after_heading_reference_feedback"


def verify_config(control: dict, candidate: dict) -> None:
    expected = deepcopy(control)
    if (
        expected.get("runtime")
        != {"schema_version": 1, "profile_id": "gmt_g1_four_state_loop_course/v1"}
        or expected.get("schema_version") != 2
    ):
        raise ValueError("control must use the retained four-state profile")
    expected["schema_version"] = 4
    expected["runtime"]["profile_id"] = PROFILE
    if candidate != expected:
        raise ValueError("candidate changes more than the declared feedback profile")


def verify_intervention(control: dict, candidate: dict) -> dict:
    prefix = shared.verify_prefix(control, candidate)
    if np.array_equal(
        control["trajectory"]["current_reference"][shared.PREFIX_ACTION_COUNT],
        candidate["trajectory"]["current_reference"][shared.PREFIX_ACTION_COUNT],
    ):
        raise ValueError("feedback must change the first after reference target")
    return {**prefix, "first_after_reference_changed": True}


def verify_reset(control: dict, candidate: dict) -> None:
    from oracle_composition.adapters.gmt.course_runtime import runtime_profile_from_config

    old = json.loads(control["outputs"]["zero_residual_evaluation.json"])["reset"]
    observed = json.loads(candidate["outputs"]["zero_residual_evaluation.json"])["reset"]
    expected = deepcopy(old)
    expected["runtime_id"] = GYM_PROFILE
    expected["course_runtime"] = runtime_profile_from_config(
        candidate["config"]
    ).manifest_contract()
    if observed != expected:
        raise ValueError("reset differs beyond the declared runtime metadata")


def verify_runtime_inputs(control: dict, candidate: dict) -> None:
    from oracle_composition.adapters.gmt.course_runtime import runtime_profile_from_config

    normalized = []
    for run in (control, candidate):
        record = deepcopy(run["resource_fixed_inputs"])
        declared = runtime_profile_from_config(run["config"]).manifest_contract()
        if record.get("course_runtime") != declared:
            raise ValueError("resource runtime metadata differs from its exact declared profile")
        del record["course_runtime"]
        normalized.append(record)
    if normalized[0] != normalized[1]:
        raise ValueError("fresh arms differ beyond declared runtime resource metadata")


def steering_measures(run: dict) -> dict:
    """Cross-check the held-rate law independently from its runtime helper."""
    frames, trajectory = run["frames"], run["trajectory"]
    qpos = trajectory["qpos"]
    task_frame = TaskFrame.initialize(qpos[0, :2], qpos[0, 3:7])
    corrections, issued_rates, native_rates, saturation, native_exceedance = [], [], [], [], []
    endpoint_window_mismatches = 0
    native_endpoint_window_mismatches = 0
    first_after = None
    for index, row in enumerate(frames):
        active = row["executed_mode"] == "after"
        if (TRACE_KEY in row) != active:
            raise ValueError("steering trace must exist exactly on executed after actions")
        if not active:
            continue
        if first_after is None:
            first_after = index
        trace = row[TRACE_KEY]
        projection = task_frame.project(qpos[index, :2], qpos[index, 3:7])
        target = max(-0.3, min(0.3, -0.3 * projection.lateral_m))
        error = target - projection.heading_error_rad
        correction = math.atan2(math.sin(error), math.cos(error))
        expected_state = {
            "lateral_m": projection.lateral_m,
            "heading_error_signed_rad": projection.heading_error_rad,
        }
        if (
            trace["pre_action_control_step"] != index
            or trace["observed_pre_action"] != expected_state
            or trace["target_heading_rad"] != target
            or trace["correction_yaw_rate_rad_s"] != correction
            or trace["applied_correction_yaw_rate_rad_s_float32"] != float(np.float32(correction))
        ):
            raise ValueError("steering law differs from the observed pre-action task frame")
        endpoint = trace["held_poststep_target"]
        native = endpoint["native_yaw_rate_rad_s"]
        expected_yaw = float(
            np.clip(np.float32(native) + np.float32(correction), np.float32(-0.3), np.float32(0.3))
        )
        if (
            endpoint["issued_yaw_rate_rad_s"] != expected_yaw
            or float(trajectory["current_reference"][index, 6]) != expected_yaw
        ):
            raise ValueError("poststep target did not hold the pre-action correction")
        if any(
            type(endpoint[key]) is not bool
            for key in ("matches_issued_window_first_row", "native_matches_window_first_row")
        ):
            raise ValueError("endpoint/window comparison must be an observed boolean")
        endpoint_window_mismatches += not endpoint["matches_issued_window_first_row"]
        native_endpoint_window_mismatches += not endpoint["native_matches_window_first_row"]
        corrections.append(correction)
        issued_rates.append(expected_yaw)
        native_rates.append(native)
        saturation.append(trace["window"]["total_rate_saturation_fraction"])
        native_exceedance.append(trace["window"]["native_rate_above_limit_fraction"])
    if first_after is None:
        return {"after_action_count": 0}
    # This diagnostic includes the entry boundary plus the first 100 resulting states.
    boundary_end = min(len(qpos), first_after + 101)
    heading = np.unwrap(shared._yaw_wxyz(qpos))
    heading -= heading[0]
    early = [
        task_frame.project(state[:2], state[3:7]).lateral_m
        for state in qpos[first_after:boundary_end]
    ]
    return {
        "after_action_count": len(corrections),
        "issued_endpoint_window_first_row_mismatch_count": endpoint_window_mismatches,
        "native_endpoint_window_first_row_mismatch_count": native_endpoint_window_mismatches,
        "first_correction_rad_s": corrections[0],
        "mean_correction_rad_s": float(np.mean(corrections)),
        "issued_local_yaw_integral_rad": float(np.sum(issued_rates) * 0.02),
        "native_local_yaw_integral_rad": float(np.sum(native_rates) * 0.02),
        "mean_window_total_saturation_fraction": float(np.mean(saturation)),
        "mean_window_native_limit_exceedance_fraction": float(np.mean(native_exceedance)),
        "first_two_seconds_boundary_count": boundary_end - first_after,
        "first_two_seconds_heading_abs_max_rad": float(
            np.max(np.abs(heading[first_after:boundary_end]))
        ),
        "first_two_seconds_entry_relative_heading_excursion_max_rad": float(
            np.max(np.abs(heading[first_after:boundary_end] - heading[first_after]))
        ),
        "first_two_seconds_lateral_abs_max_m": max(abs(value) for value in early),
        "series_authority": "pinned raw trajectory and per-action steering receipts",
        "manipulation": "pre_action_feedback_plus_total_yaw_rate_clipping",
    }


def mechanism_screen(objective: dict, measures: dict) -> dict[str, bool]:
    screen = shared.mechanism_screen(objective, measures)
    del screen["after_unwrapped_heading_below_1rad"]
    heading = measures["after_unwrapped_heading_absolute_max_rad"]
    screen["after_unwrapped_heading_below_0p5rad"] = type(heading) is float and heading < 0.5
    return screen


def _pins(value: dict, coordination_root: Path):
    required = {"root", "manifest_sha256", "resource_sha256", "config_sha256", "source_commit"}
    optional = {"reservation_path", "reservation_sha256"}
    if type(value) is not dict or not required <= set(value) or set(value) - required - optional:
        raise ValueError("run pin fields differ")
    if ("reservation_path" in value) != ("reservation_sha256" in value):
        raise ValueError("reservation requires path and canonical digest")
    return shared.RunPins(
        root=Path(value["root"]).resolve(strict=True),
        manifest_sha256=value["manifest_sha256"],
        resource_sha256=value["resource_sha256"],
        config_sha256=value["config_sha256"],
        source_commit=value["source_commit"],
        reservation_path=(
            Path(value["reservation_path"]).resolve(strict=True)
            if "reservation_path" in value
            else None
        ),
        reservation_sha256=value.get("reservation_sha256"),
        coordination_root=coordination_root if "reservation_path" in value else None,
    )


def score_study(inputs: dict) -> dict:
    if (
        type(inputs) is not dict
        or set(inputs)
        != {
            "schema_version",
            "baseline",
            "control",
            "candidate",
            "coordination_root",
            "fresh_source_tree_sha256",
            "fresh_source_file_count",
            "candidate_config_sha256",
        }
        or type(inputs["schema_version"]) is not int
        or inputs["schema_version"] != 1
    ):
        raise ValueError("Study016 input fields or schema differ")
    if (
        inputs["fresh_source_tree_sha256"] != FRESH_TREE
        or type(inputs["fresh_source_file_count"]) is not int
        or inputs["fresh_source_file_count"] != FRESH_FILE_COUNT
    ):
        raise ValueError("Study016 fresh source differs from the predata executable tree")
    coordination_root = Path(inputs["coordination_root"]).resolve(strict=True)
    pins = {
        name: _pins(inputs[name], coordination_root)
        for name in ("baseline", "control", "candidate")
    }
    baseline, control, candidate = (pins[name] for name in ("baseline", "control", "candidate"))
    if (
        baseline.manifest_sha256 != BASE_MANIFEST
        or baseline.resource_sha256 != BASE_RESOURCE
        or baseline.config_sha256 != BASE_CONFIG
        or baseline.source_commit != BASE_SOURCE
        or control.config_sha256 != BASE_CONFIG
        or candidate.config_sha256 != inputs["candidate_config_sha256"]
        or inputs["candidate_config_sha256"] != CANDIDATE_CONFIG
        or candidate.source_commit != control.source_commit
        or control.reservation_path is None
        or candidate.reservation_path is None
    ):
        raise ValueError("Study016 fixed baseline or fresh pair pins differ")
    with tempfile.TemporaryDirectory(prefix="study016-feedback-") as temporary:
        feedback_root = Path(temporary).resolve(strict=True)
        runs = {
            name: shared._verify_run(
                record,
                feedback_root / name,
                expected_source_tree_sha256=(
                    BASE_TREE if name == "baseline" else inputs["fresh_source_tree_sha256"]
                ),
                expected_source_file_count=(
                    194 if name == "baseline" else inputs["fresh_source_file_count"]
                ),
            )
            for name, record in pins.items()
        }
    baseline, control, candidate = (runs[name] for name in ("baseline", "control", "candidate"))
    if baseline["config_bytes"] != control["config_bytes"] or any(
        baseline["outputs"][name] != control["outputs"][name] for name in shared._COMPARABLE_OUTPUTS
    ):
        raise ValueError("fresh legacy control must byte-reproduce all three retained outputs")
    verify_config(control["config"], candidate["config"])
    verify_runtime_inputs(control, candidate)
    verify_reset(control, candidate)
    prefix = verify_intervention(control, candidate)
    measures = shared.candidate_measures(
        candidate["frames"],
        candidate["trajectory"],
        candidate["objective"],
        candidate["config"]["task"],
    )
    steering = steering_measures(candidate)
    screen = mechanism_screen(candidate["objective"], measures)
    return {
        "schema_version": 1,
        "study": "016",
        "artifact": "gmt_g1_after_feedback_development_score/v1",
        "source_commit": pins["control"].source_commit,
        "fresh_source_tree_sha256": inputs["fresh_source_tree_sha256"],
        "fresh_source_file_count": inputs["fresh_source_file_count"],
        "candidate_config_sha256": inputs["candidate_config_sha256"],
        "fresh_control_byte_reproduced_three_outputs": True,
        "prefix": prefix,
        "candidate_measures": measures,
        "steering_measures": steering,
        "mechanism_screen": screen,
        "mechanism_screen_passed": all(screen.values()),
        "original_objective": candidate["objective"],
        "original_development_gate_results": candidate["objective"]["development_gate_results"],
        "run_bindings": {
            name: {
                "manifest_sha256": record.manifest_sha256,
                "resource_sha256": record.resource_sha256,
                "reservation_sha256": record.reservation_sha256,
                "feedback": runs[name]["feedback"],
            }
            for name, record in pins.items()
        },
        "claim_scope": "one_seed_zero_residual_development_screen_not_task_success",
        "claim_limits": [
            "correction_and_clipping_are_one_manipulation",
            "all_physical_region_visits_remain_authoritative",
            "after_only_feedback_cannot_erase_earlier_task_failures",
            "no_training_or_heldout_generalization_claim",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs, encoded = read_json_object(args.inputs)
    if hashlib.sha256(encoded).hexdigest() != args.inputs_sha256:
        raise ValueError("study inputs differ from their pinned bytes")
    result = score_study(inputs)
    result["inputs_sha256"] = args.inputs_sha256
    result["scorer_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["shared_verifier_sha256"] = hashlib.sha256(_SHARED_PATH.read_bytes()).hexdigest()
    digest = write_json_receipt(args.output, result)
    print(
        json.dumps(
            {
                "sha256": digest,
                "path": str(args.output),
                "mechanism_screen": result["mechanism_screen"],
                "candidate_measures": result["candidate_measures"],
                "steering_measures": result["steering_measures"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
