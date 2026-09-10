"""Exercise the preregistered Study015 prefix and trajectory screen."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest


def _module():
    path = Path(__file__).resolve().parents[2] / "experiments/015_g1_task_aligned_after/score.py"
    spec = importlib.util.spec_from_file_location("study015_score", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def scorer():
    return _module()


def _frame(index: int, *, mode: str, progress: float = 2.5) -> dict:
    return {
        "executed_mode": mode,
        "metrics": {
            "control_step": index + 1,
            "progress_m": progress,
            "speed_error_m_s": 0.1,
        },
        "reward": {"total_reward": 1.0},
        "transition": (
            {"control_step": index, "from_state": "rise", "to_state": "after"}
            if index == 263
            else None
        ),
    }


def _prefix_run() -> dict:
    frames = [_frame(i, mode="rise" if i < 263 else "after") for i in range(265)]
    qpos = np.zeros((266, 30), dtype=np.float64)
    qpos[:, 3] = 1.0
    return {
        "frames": frames,
        "trajectory": {
            "composite_raw_action": np.zeros((265, 23), dtype=np.float32),
            "current_reference": np.zeros((265, 30), dtype=np.float32),
            "qpos": qpos,
            "qvel": np.zeros((266, 29), dtype=np.float64),
        },
    }


def _objective(*, inside_count: int = 0, entry: bool = True) -> dict:
    return {
        "duration_seconds": 20.0,
        "fall_count": 0,
        "maximum_progress_m": 3.5,
        "maximum_lateral_error_m": 2.49,
        "region": {
            "entry_observed": entry,
            "inside_sample_count": inside_count,
        },
        "oracle_diagnostics": {"observed_switch_count": 3},
        "tracking": {
            "joint_position_rmse_rad_p95": 0.35,
            "roll_pitch_rmse_rad_p95": 0.25,
        },
        "development_gate_results": {
            "finish_reached": True,
            "full_horizon_without_fall": True,
            "inside_mean_speed_target": False,
            "inside_posture_compliance": False,
            "inside_posture_dip": False,
            "joint_position_rmse_p95": True,
            "maximum_lateral_error": False,
            "mean_speed_error": False,
            "minimum_inside_samples": True,
            "region_entry_and_exit_observed": True,
            "roll_pitch_rmse_p95": True,
        },
    }


def _qpos(yaws: list[float]) -> np.ndarray:
    result = np.zeros((len(yaws), 30), dtype=np.float64)
    result[:, 2] = 0.75
    result[:, 3] = np.cos(np.asarray(yaws) / 2)
    result[:, 6] = np.sin(np.asarray(yaws) / 2)
    return result


def _write_json(path: Path, value: dict) -> str:
    encoded = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _native_reservation(scorer, tmp_path: Path, authoritative: dict):
    coordination_root = tmp_path / "mailbox"
    messages = coordination_root / "messages"
    acknowledgments = coordination_root / "acks"
    messages.mkdir(parents=True)
    acknowledgments.mkdir()
    message_id = "20260907T000100.000000Z-" + "1" * 32
    message = {
        "body": scorer.canonical_json_bytes(authoritative).decode(),
        "from": "fable",
        "id": message_id,
        "kind": "acceptance",
        "reply_to": authoritative["proposal_id"],
        "schema_version": 1,
        "sent_at": "2026-09-07T00:01:00.000000Z",
        "subject": "accept fixed Study015 probe",
        "to": "astra",
    }
    message_path = messages / f"{message_id}.json"
    message_sha256 = _write_json(message_path, message)
    acknowledgment = {
        "acknowledged_at": "2026-09-07T00:01:01.000000Z",
        "meaning": "read_not_agreement",
        "message_id": message_id,
        "recipient": "astra",
        "schema_version": 1,
    }
    acknowledgment_path = acknowledgments / f"{message_id}.json"
    acknowledgment_sha256 = _write_json(acknowledgment_path, acknowledgment)
    reservation = {
        "accepted": True,
        "acceptance_message": {
            "path": f"messages/{message_id}.json",
            "sha256": message_sha256,
        },
        "acknowledgment": {
            "path": f"acks/{message_id}.json",
            "sha256": acknowledgment_sha256,
        },
        **authoritative,
        "schema_version": 2,
    }
    reservation_path = tmp_path / "reservation.json"
    _write_json(reservation_path, reservation)
    reservation_sha256 = hashlib.sha256(scorer.canonical_json_bytes(reservation)).hexdigest()
    return coordination_root, reservation_path, reservation_sha256, reservation


def test_prefix_boundary_allows_first_difference_only_at_action263_and_state264(scorer) -> None:
    control = _prefix_run()
    candidate = copy.deepcopy(control)
    candidate["trajectory"]["composite_raw_action"][263] = 1.0
    candidate["trajectory"]["qpos"][264] = 1.0
    candidate["trajectory"]["qvel"][264] = 1.0

    result = scorer.verify_prefix(control, candidate)

    assert result == {"equal_action_count": 263, "equal_state_row_count": 264}


@pytest.mark.parametrize(
    "field,index",
    [("composite_raw_action", 262), ("qpos", 263), ("qvel", 263)],
)
def test_prefix_rejects_any_earlier_numeric_difference(scorer, field: str, index: int) -> None:
    control = _prefix_run()
    candidate = copy.deepcopy(control)
    candidate["trajectory"][field][index] = 1.0

    with pytest.raises(ValueError, match="numeric prefix"):
        scorer.verify_prefix(control, candidate)


def test_prefix_rejects_changed_reward_before_after_command(scorer) -> None:
    control = _prefix_run()
    candidate = copy.deepcopy(control)
    candidate["frames"][100]["reward"]["total_reward"] = 2.0

    with pytest.raises(ValueError, match="complete trace prefix"):
        scorer.verify_prefix(control, candidate)


def test_zero_residual_contract_rejects_nonzero_action(scorer) -> None:
    trajectory = _prefix_run()["trajectory"]
    trajectory["residual_action"] = np.zeros((265, 23), dtype=np.float32)
    scorer._require_zero_residual({"residual_rms": 0.0}, trajectory)

    trajectory["residual_action"][10, 2] = 1e-8
    with pytest.raises(ValueError, match="exact zero-residual"):
        scorer._require_zero_residual({"residual_rms": 0.0}, trajectory)


def test_signed_commanded_yaw_prediction_is_a_separate_required_manipulation(scorer) -> None:
    assert scorer.manipulation_check({"after_commanded_local_yaw_rate_mean_rad_s": 0.06}) == {
        "signed_after_commanded_yaw_mean_at_most_0p06_rad_s": True
    }
    assert not all(
        scorer.manipulation_check({"after_commanded_local_yaw_rate_mean_rad_s": 0.060001}).values()
    )


def test_unwrapped_full_yaw_rejects_wrapped_endpoint_uturn(scorer) -> None:
    yaws = np.linspace(0.0, 2 * math.pi, 9).tolist()
    frames = [_frame(i, mode="after", progress=2.5) for i in range(8)]
    measures = scorer.candidate_measures(
        frames,
        {"qpos": _qpos(yaws), "current_reference": np.zeros((8, 30))},
        _objective(),
        {"region_entry_distance_m": 1.0, "region_exit_distance_m": 2.0},
    )

    assert abs(yaws[-1] % (2 * math.pi)) < 1e-12
    assert measures["after_unwrapped_heading_absolute_max_rad"] == pytest.approx(2 * math.pi)
    assert (
        scorer.mechanism_screen(_objective(), measures)["after_unwrapped_heading_below_1rad"]
        is False
    )


def test_missing_after_and_missing_physical_entry_fail_screen(scorer) -> None:
    frames = [_frame(i, mode="rise", progress=0.5) for i in range(5)]
    objective = _objective(entry=False)
    measures = scorer.candidate_measures(
        frames,
        {"qpos": _qpos([0.0] * 6), "current_reference": np.zeros((5, 30))},
        objective,
        {"region_entry_distance_m": 1.0, "region_exit_distance_m": 2.0},
    )
    screen = scorer.mechanism_screen(objective, measures)

    assert measures["first_after_action_index"] is None
    assert screen["after_state_executed"] is False
    assert screen["physical_region_entry_observed"] is False
    assert not all(screen.values())


def test_later_physical_revisit_is_separate_but_all_visits_remain_authoritative(scorer) -> None:
    progress = [0.5, 1.2, 1.5, 2.1, 2.5, 1.8, 1.5, 2.2]
    frames = [
        _frame(i, mode="rise" if i < 3 else "after", progress=value)
        for i, value in enumerate(progress)
    ]
    objective = _objective(inside_count=4)
    measures = scorer.candidate_measures(
        frames,
        {"qpos": _qpos([0.0] * 9), "current_reference": np.zeros((8, 30))},
        objective,
        {"region_entry_distance_m": 1.0, "region_exit_distance_m": 2.0},
    )

    assert measures["first_cross_inside_sample_count"] == 2
    assert measures["later_revisit_inside_sample_count"] == 2
    assert measures["all_visit_inside_sample_count"] == 4
    assert measures["post_after_region_reentry_count"] == 2
    assert scorer.mechanism_screen(objective, measures)["no_post_after_region_reentry"] is False


def _o7_config() -> dict:
    return {
        "schema_version": 2,
        "mode": "probe",
        "assets": {"fixed": True},
        "task": {
            "finish_distance_m": 3.5,
            "horizon_steps": 1000,
            "posture_band_high_m": 0.6,
            "posture_band_low_m": 0.3,
            "region_entry_distance_m": 1.0,
            "region_exit_distance_m": 2.0,
            "schema_id": "gmt_g1_posture_course_task/v1",
            "schema_version": 1,
            "target_speed_inside_m_s": 0.65,
            "target_speed_outside_m_s": 0.7,
        },
        "reward": {"fixed": True},
        "runtime": {"profile_id": "gmt_g1_four_state_loop_course/v1", "schema_version": 1},
        "seed": 20260906,
        "training_steps": 0,
        "oracle": {
            "schema_version": 1,
            "evidence_class": "exploratory_oracle_cycle",
            "initial": "before",
            "oracle_id": "g1_course_o7_balanced_loop_rise",
            "behaviors": ["walk", "crouch", "rise"],
            "states": {
                "before": {"behavior": "walk", "min_dwell": 25},
                "inside": {"behavior": "crouch", "min_dwell": 25},
                "rise": {"behavior": "rise", "min_dwell": 24},
                "after": {"behavior": "walk", "min_dwell": 25},
            },
            "transitions": [{"fixed": True}],
        },
        "segments": {
            "walk": {"fixed": "walk"},
            "crouch": {"fixed": "crouch"},
            "rise": {"fixed": "rise"},
        },
    }


def _o7b_config() -> dict:
    candidate = copy.deepcopy(_o7_config())
    candidate["oracle"]["oracle_id"] = "g1_course_o7b_balanced_loop_rise_after_crop"
    candidate["oracle"]["behaviors"].append("walk_after")
    candidate["oracle"]["states"]["after"]["behavior"] = "walk_after"
    candidate["segments"]["walk_after"] = {
        "boundary": "wrap_within_segment",
        "end_seconds": 31.56,
        "motion_name": "basic_walk",
        "start_seconds": 30.62,
    }
    return candidate


def test_candidate_config_is_exact_single_after_crop_change(scorer) -> None:
    control = _o7_config()
    candidate = _o7b_config()

    scorer.verify_o7b_config(control, candidate)
    candidate["task"] = {"changed": True}
    with pytest.raises(ValueError, match="exact O7b"):
        scorer.verify_o7b_config(control, candidate)


def test_resource_and_native_reservation_bind_source_inputs_argv_and_output(
    scorer, tmp_path: Path
) -> None:
    root = tmp_path.resolve()
    outputs = {}
    output_records = {}
    for name in scorer._OUTPUTS:
        encoded = name.encode()
        (root / name).write_bytes(encoded)
        digest = hashlib.sha256(encoded).hexdigest()
        outputs[name] = digest
        output_records[name] = {"path": name, "sha256": digest, "size": len(encoded)}
    config_path = str(root / "source_config.json")
    argv = [
        "python",
        "-m",
        "oracle_composition.adapters.gmt.course_run",
        "--config",
        config_path,
        "--output",
        str(root),
    ]
    inputs = {
        "workload": "course",
        "course_mode": "probe",
        "artifact_contract": "gmt_g1_fixed_development_launcher/v1",
        "config": {
            "path": config_path,
            "sha256": "c" * 64,
            "size": (root / "input_config.json").stat().st_size,
        },
        "repository_sources": {
            "canonical_tree_sha256": scorer.FRESH_SOURCE_TREE_SHA256,
            "file_count": scorer.SOURCE_FILE_COUNT,
        },
    }
    authoritative = {
        "accepted_until_utc": "2026-09-07T00:10:00.000000Z",
        "canonical_argv": argv,
        "commit": "a" * 40,
        "conflict_check": "no_other_heavy_repository_job",
        "hard_wall_seconds": 1200,
        "inputs": inputs,
        "mode": "smoke",
        "output": str(root),
        "owner": "astra-system-lead-20260906",
        "proposal_id": "20260907T000000.000000Z-" + "0" * 32,
        "required_authorizer": "fable",
    }
    coordination_root, reservation_path, reservation_sha256, reservation = _native_reservation(
        scorer, tmp_path, authoritative
    )
    pins = scorer.RunPins(
        root,
        "m" * 64,
        "r" * 64,
        "c" * 64,
        "a" * 40,
        reservation_path,
        reservation_sha256,
        coordination_root,
    )
    manifest = {"outputs": outputs}
    resource = {
        "schema_version": 1,
        "artifact": "gmt_g1_course_development_resource_receipt",
        "status": "succeeded",
        "evidence_class": "development_course_probe_not_task_success",
        "commit": "a" * 40,
        "reservation_sha256": reservation_sha256,
        "inputs": inputs,
        "canonical_argv": argv,
        "artifacts": {
            "course_manifest": {
                "path": "course_run_manifest.json",
                "sha256": "m" * 64,
                "size": 10,
            },
            "outputs": output_records,
        },
    }

    assert scorer._validated_reservation(pins) == reservation
    assert scorer._verify_resource(
        pins=pins,
        manifest=manifest,
        manifest_size=10,
        resource=resource,
        reservation=reservation,
    ) == {
        "source_tree_sha256": scorer.FRESH_SOURCE_TREE_SHA256,
        "fixed_inputs": {name: value for name, value in inputs.items() if name != "config"},
    }

    wrong_tree = copy.deepcopy(resource)
    wrong_tree["inputs"]["repository_sources"]["canonical_tree_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="predeclared identity"):
        scorer._verify_resource(
            pins=pins,
            manifest=manifest,
            manifest_size=10,
            resource=wrong_tree,
            reservation=reservation,
        )

    # A later study supplies its predeclared source identity, not new global defaults.
    explicit = {
        "expected_source_tree_sha256": scorer.FRESH_SOURCE_TREE_SHA256,
        "expected_source_file_count": scorer.SOURCE_FILE_COUNT,
    }
    scorer._verify_resource(
        pins=pins,
        manifest=manifest,
        manifest_size=10,
        resource=resource,
        reservation=reservation,
        **explicit,
    )
    for overrides in (
        {"expected_source_tree_sha256": "0" * 64, "expected_source_file_count": 194},
        {"expected_source_tree_sha256": scorer.FRESH_SOURCE_TREE_SHA256},
        {"expected_source_file_count": 194},
        {**explicit, "expected_source_file_count": True},
        {**explicit, "expected_source_file_count": 0},
        {**explicit, "expected_source_file_count": 193},
        {**explicit, "expected_source_tree_sha256": "not-a-digest"},
    ):
        with pytest.raises(ValueError):
            scorer._verify_resource(
                pins=pins,
                manifest=manifest,
                manifest_size=10,
                resource=resource,
                reservation=reservation,
                **overrides,
            )

    wrong_count = copy.deepcopy(resource)
    wrong_count["inputs"]["repository_sources"]["file_count"] = 193
    with pytest.raises(ValueError, match="predeclared identity"):
        scorer._verify_resource(
            pins=pins,
            manifest=manifest,
            manifest_size=10,
            resource=wrong_count,
            reservation=reservation,
        )

    wrong_reservation = copy.deepcopy(resource)
    wrong_reservation["reservation_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="native reservation"):
        scorer._verify_resource(
            pins=pins,
            manifest=manifest,
            manifest_size=10,
            resource=wrong_reservation,
            reservation=reservation,
        )

    acknowledgment = coordination_root / reservation["acknowledgment"]["path"]
    acknowledgment.write_text("{}\n")
    with pytest.raises(ValueError, match="acceptance chain"):
        scorer._validated_reservation(pins)


def test_positive_verified_fixture_scores_without_promoting_task_success(
    scorer, tmp_path: Path
) -> None:
    coordination_root = tmp_path / "mailbox"
    baseline_pins = scorer.RunPins(
        tmp_path / "baseline",
        scorer.BASE_MANIFEST_SHA256,
        scorer.BASE_RESOURCE_SHA256,
        scorer.BASE_CONFIG_SHA256,
        scorer.BASE_SOURCE_COMMIT,
    )
    control_pins = scorer.RunPins(
        tmp_path / "control",
        "1" * 64,
        "2" * 64,
        scorer.BASE_CONFIG_SHA256,
        "a" * 40,
        tmp_path / "control-reservation.json",
        "3" * 64,
        coordination_root,
    )
    candidate_pins = scorer.RunPins(
        tmp_path / "candidate",
        "4" * 64,
        "5" * 64,
        scorer.CANDIDATE_CONFIG_SHA256,
        "a" * 40,
        tmp_path / "candidate-reservation.json",
        "6" * 64,
        coordination_root,
    )
    control_run = _prefix_run()
    control_run["frames"][0]["metrics"]["progress_m"] = 1.5
    candidate_run = copy.deepcopy(control_run)
    candidate_run["frames"][-1]["metrics"]["progress_m"] = 3.5
    candidate_run["trajectory"]["composite_raw_action"][263, 0] = 0.1
    candidate_run["trajectory"]["qpos"][264, 0] = 0.001
    candidate_run["trajectory"]["current_reference"][263:, 6] = 0.05
    objective = _objective(inside_count=1)
    comparable = {name: name.encode() for name in scorer._COMPARABLE_OUTPUTS}

    def record(pins, *, config, run, candidate: bool = False):
        return {
            "pins": pins,
            "config": config,
            "config_bytes": b"o7b" if candidate else b"o7",
            "outputs": (
                {name: b"candidate-" + value for name, value in comparable.items()}
                if candidate
                else comparable
            ),
            "frames": run["frames"],
            "trajectory": run["trajectory"],
            "objective": objective,
            "source_tree_sha256": (
                scorer.BASE_SOURCE_TREE_SHA256
                if pins is baseline_pins
                else scorer.FRESH_SOURCE_TREE_SHA256
            ),
            "resource_fixed_inputs": {"fixed": True},
            "feedback": {"feedback": {"sha256": "8" * 64, "byte_count": 1}},
        }

    records = {
        baseline_pins.root: record(baseline_pins, config=_o7_config(), run=control_run),
        control_pins.root: record(control_pins, config=_o7_config(), run=control_run),
        candidate_pins.root: record(
            candidate_pins, config=_o7b_config(), run=candidate_run, candidate=True
        ),
    }
    result = scorer.score_study(
        scorer.StudyInputs(
            baseline_pins,
            control_pins,
            candidate_pins,
            scorer.CANDIDATE_CONFIG_SHA256,
        ),
        run_verifier=lambda pins, _feedback: records[pins.root],
    )

    assert result["fresh_control_byte_reproduced_three_outputs"] is True
    assert result["mechanism_screen_passed"] is True
    assert all(result["required_manipulation_check"].values())
    assert result["candidate_measures"][
        "after_commanded_local_yaw_rate_mean_rad_s"
    ] == pytest.approx(0.05)
    assert result["claim_scope"].endswith("not_task_success")

    candidate_run["trajectory"]["current_reference"][263:, 6] = 0.061
    failed_manipulation = scorer.score_study(
        scorer.StudyInputs(
            baseline_pins,
            control_pins,
            candidate_pins,
            scorer.CANDIDATE_CONFIG_SHA256,
        ),
        run_verifier=lambda pins, _feedback: records[pins.root],
    )
    assert all(failed_manipulation["mechanism_screen"].values())
    assert failed_manipulation["mechanism_screen_passed"] is False


def test_candidate_verification_failure_cannot_publish_score(scorer, tmp_path: Path) -> None:
    coordination_root = tmp_path / "mailbox"
    baseline = scorer.RunPins(
        tmp_path / "baseline",
        scorer.BASE_MANIFEST_SHA256,
        scorer.BASE_RESOURCE_SHA256,
        scorer.BASE_CONFIG_SHA256,
        scorer.BASE_SOURCE_COMMIT,
    )
    control = scorer.RunPins(
        tmp_path / "control",
        "1" * 64,
        "2" * 64,
        scorer.BASE_CONFIG_SHA256,
        "a" * 40,
        tmp_path / "control-reservation.json",
        "3" * 64,
        coordination_root,
    )
    candidate = scorer.RunPins(
        tmp_path / "candidate",
        "4" * 64,
        "5" * 64,
        scorer.CANDIDATE_CONFIG_SHA256,
        "a" * 40,
        tmp_path / "candidate-reservation.json",
        "6" * 64,
        coordination_root,
    )
    inputs = scorer.StudyInputs(baseline, control, candidate, scorer.CANDIDATE_CONFIG_SHA256)

    def verifier(pins, _output):
        if pins is candidate:
            raise ValueError("candidate verification failed")
        return {"not_used": True}

    output = tmp_path / "score.json"
    with pytest.raises(ValueError, match="candidate verification failed"):
        scorer.score_and_publish(inputs, output, run_verifier=verifier)
    assert not output.exists()
