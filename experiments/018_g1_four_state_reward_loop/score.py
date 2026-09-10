"""Fail-closed scorer for the sequential Study018 reward comparison."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

import oracle_composition
from oracle_composition.adapters.gmt.course_config import load_run_config
from oracle_composition.adapters.gmt.course_evaluation import DEVELOPMENT_GATE_THRESHOLDS
from oracle_composition.adapters.gmt.course_proposal import apply_proposal
from oracle_composition.adapters.gmt.course_runtime import FOUR_STATE_FINITE_HORIZON_RUNTIME
from oracle_composition.adapters.gmt.course_task import CourseStepMetrics, reward
from oracle_composition.adapters.gmt.io import write_json_receipt
from oracle_composition.adapters.gmt.study_scoring import (
    CourseStudyExpectation,
    RunManifestBinding,
    _score_cell,
    score_course_study_pair,
)
from oracle_composition.adapters.gmt.training_normalizer import FIXED_NORMALIZER_STATE_SHA256
from oracle_composition.adapters.gmt.training_telemetry import telemetry_filename
from oracle_composition.harness.contract import decode_json_object, read_json_object
from oracle_composition.harness.resource_slot import (
    ResourceSlotError,
    canonical_json_bytes,
    load_validated_reservation,
)

_SCORER_PATH = Path(__file__).resolve()
_PROTOCOL_PATH = _SCORER_PATH.with_name("PROTOCOL.md")
_STUDY017_PATH = _SCORER_PATH.parents[1] / "017_g1_execution_derived_reference/score.py"
_CRITERIA_PATH = _SCORER_PATH.with_name("criteria.py")
_REPOSITORY_ROOT = _SCORER_PATH.parents[2]
_RUN_GMT_PROBE_PATH = _REPOSITORY_ROOT / "scripts/run_gmt_probe.py"
_RUN_GMT_DEVELOPMENT_PATH = _REPOSITORY_ROOT / "scripts/run_gmt_development.py"


def _load_local(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load Study018 dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load_local("run_gmt_probe", _RUN_GMT_PROBE_PATH)
development = _load_local("study018_run_gmt_development", _RUN_GMT_DEVELOPMENT_PATH)
study017 = _load_local("study018_study017_helpers", _STUDY017_PATH)
criteria = _load_local("study018_criteria", _CRITERIA_PATH)
shared = study017.shared

BASELINE_CONFIG_SHA256 = "a9e52ba706e4ddfd394c757cf525bf210d553906430c7f08e11e0495affdf289"
RETAINED_ROOT = Path(
    "/Users/samueldoane/Documents/ChatGPT/humanoid-harness-probe-runs/"
    "gmt_course_o7b_study015_candidate_20260907"
)
RETAINED_MANIFEST_SHA256 = "92e2a72f743bded8b2694760c7492f8d35f0829489415a05c3afff6ec459944a"
RETAINED_RESOURCE_SHA256 = "ea00072569e159b8f04b1b7c801296b19297559b0ed91f04ec4077551cb429b8"
RETAINED_CONFIG_SHA256 = "ba45dba36ef88bdc522ed2115e46a4b3d876e00a627a089fd56ddf0de623e18a"
RETAINED_SOURCE_COMMIT = "3cb3102d5cb2e3a8603663df13ed50b5d6cad0cb"
RETAINED_SOURCE_TREE_SHA256 = "3ba7fe0ab72374e0667402848460c43915024fbcd2fe44f9ce5de791f0ab8b8e"
RETAINED_SOURCE_FILE_COUNT = 194

SEED = 20260906
TRAINING_STEPS = 131_072
EXPECTED_TRAINER = {
    "schema_version": 3,
    "schema_id": "gmt_g1_scaled_fixed_normalizer_training_profile/v3",
    "learning_rate": 0.00003,
    "total_training_reward_scale": 0.015625,
    "observation_preconditioning": "gmt_g1_fixed_actor_observation_normalizer/v1",
}
LATERAL_WEIGHT_BOUNDS = (0.5, 4.0)
HEADING_WEIGHT_BOUNDS = (0.25, 2.0)
_HEX = frozenset("0123456789abcdef")
_MODES = {"before", "inside", "rise", "after"}
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
class CellPins:
    root: Path
    config_sha256: str
    manifest_sha256: str
    resource_sha256: str
    reservation_path: Path
    reservation_sha256: str

    def receipt(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "config_sha256": self.config_sha256,
            "manifest_sha256": self.manifest_sha256,
            "resource_sha256": self.resource_sha256,
            "reservation_path": str(self.reservation_path),
            "reservation_sha256": self.reservation_sha256,
        }


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: object, *, length: int, field: str) -> str:
    if type(value) is not str or len(value) != length or any(c not in _HEX for c in value):
        raise ValueError(f"{field} must be one lowercase hexadecimal digest")
    return value


def _pinned_json(path: Path, expected_sha256: str, *, source: str) -> tuple[dict, bytes]:
    value, encoded = read_json_object(path)
    expected = _digest(expected_sha256, length=64, field=f"{source} SHA-256")
    if _sha256(encoded) != expected:
        raise ValueError(f"{source} bytes differ from their pinned SHA-256")
    return value, encoded


def _cell_pins(value: object) -> CellPins:
    fields = {
        "root",
        "config_sha256",
        "manifest_sha256",
        "resource_sha256",
        "reservation_path",
        "reservation_sha256",
    }
    if type(value) is not dict or set(value) != fields:
        raise ValueError("Study018 cell pin fields differ")
    for name in fields - {"root", "reservation_path"}:
        _digest(value[name], length=64, field=f"cell {name}")
    raw_root = Path(value["root"])
    raw_reservation = Path(value["reservation_path"])
    if not raw_root.is_absolute() or not raw_reservation.is_absolute():
        raise ValueError("Study018 run and reservation paths must be absolute")
    root = raw_root.resolve(strict=True)
    reservation = raw_reservation.resolve(strict=True)
    if root != raw_root or reservation != raw_reservation or not root.is_dir():
        raise ValueError("Study018 run and reservation paths must be direct")
    return CellPins(
        root,
        value["config_sha256"],
        value["manifest_sha256"],
        value["resource_sha256"],
        reservation,
        value["reservation_sha256"],
    )


def _file_pin(value: object, *, field: str) -> tuple[Path, str]:
    if type(value) is not dict or set(value) != {"path", "sha256"}:
        raise ValueError(f"{field} binding fields differ")
    raw = Path(value["path"])
    if not raw.is_absolute():
        raise ValueError(f"{field} path must be absolute")
    path = raw.resolve(strict=True)
    if path != raw or not path.is_file():
        raise ValueError(f"{field} path must be one direct file")
    return path, _digest(value["sha256"], length=64, field=f"{field} SHA-256")


def _validate_local_source(common: dict[str, object]) -> dict[str, object]:
    root, commit, files = development._repository_sources(_REPOSITORY_ROOT)
    binding = development._source_tree_binding(files)
    package_root = Path(oracle_composition.__file__).resolve(strict=True).parents[2]
    if (
        root != _REPOSITORY_ROOT
        or package_root != root
        or commit != common["source_commit"]
        or binding
        != {
            "canonical_tree_sha256": common["source_tree_sha256"],
            "file_count": common["source_file_count"],
        }
    ):
        raise ValueError("local scoring source differs from the sealed run source")
    return {"root": str(root), "commit": commit, **binding}


def _validate_common(inputs: dict, *, extra_fields: set[str]) -> dict[str, object]:
    common_fields = {
        "schema_version",
        "scorer_sha256",
        "protocol_sha256",
        "source_commit",
        "source_tree_sha256",
        "source_file_count",
        "base_state_sha256",
        "coordination_root",
        "dependency_sha256",
    }
    if type(inputs) is not dict or set(inputs) != common_fields | extra_fields:
        raise ValueError("Study018 input fields differ")
    if type(inputs["schema_version"]) is not int or inputs["schema_version"] != 1:
        raise ValueError("Study018 input schema differs")
    scorer = _digest(inputs["scorer_sha256"], length=64, field="scorer SHA-256")
    protocol = _digest(inputs["protocol_sha256"], length=64, field="protocol SHA-256")
    if _sha256(_SCORER_PATH.read_bytes()) != scorer:
        raise ValueError("Study018 scorer bytes differ from their predata pin")
    if _sha256(_PROTOCOL_PATH.read_bytes()) != protocol:
        raise ValueError("Study018 protocol bytes differ from their predata pin")
    dependencies = inputs["dependency_sha256"]
    dependency_paths = {
        "criteria": _CRITERIA_PATH,
        "study017_scorer": _STUDY017_PATH,
        "study015_scorer": study017._SHARED_PATH,
    }
    if (
        type(dependencies) is not dict
        or set(dependencies) != set(dependency_paths)
        or any(
            _digest(dependencies[name], length=64, field=f"{name} SHA-256")
            != _sha256(path.read_bytes())
            for name, path in dependency_paths.items()
        )
    ):
        raise ValueError("Study018 scorer dependency bytes differ from their predata pins")
    source_commit = _digest(inputs["source_commit"], length=40, field="source commit")
    source_tree = _digest(inputs["source_tree_sha256"], length=64, field="source tree")
    base_state = _digest(inputs["base_state_sha256"], length=64, field="base state")
    source_count = inputs["source_file_count"]
    if type(source_count) is not int or source_count < 1:
        raise ValueError("source file count must be a positive integer")
    raw_coordination = Path(inputs["coordination_root"])
    if not raw_coordination.is_absolute():
        raise ValueError("coordination root must be absolute")
    coordination = raw_coordination.resolve(strict=True)
    if coordination != raw_coordination or not coordination.is_dir():
        raise ValueError("coordination root must be one direct directory")
    common = {
        "scorer_sha256": scorer,
        "protocol_sha256": protocol,
        "source_commit": source_commit,
        "source_tree_sha256": source_tree,
        "source_file_count": source_count,
        "base_state_sha256": base_state,
        "coordination_root": coordination,
        "dependency_sha256": dependencies,
    }
    _validate_local_source(common)
    return common


def _retrospective_reservation(
    pins: CellPins, *, common: dict[str, object]
) -> dict[str, object]:
    reservation, _ = read_json_object(pins.reservation_path)
    expected = _digest(
        pins.reservation_sha256,
        length=64,
        field="native reservation canonical SHA-256",
    )
    if _sha256(canonical_json_bytes(reservation)) != expected:
        raise ValueError("native reservation differs from its canonical SHA-256")
    expiry = reservation.get("accepted_until_utc")
    if type(expiry) is not str:
        raise ValueError("native reservation acceptance expiry is absent")
    try:
        retrospective_now = datetime.strptime(expiry, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=UTC
        ) - timedelta(microseconds=1)
        validated = load_validated_reservation(
            common["coordination_root"], pins.reservation_path, now=retrospective_now
        )
    except (ResourceSlotError, ValueError) as error:
        raise ValueError("native reservation acceptance chain is invalid") from error
    if validated != reservation:
        raise ValueError("native reservation changed during validation")
    return reservation


def _validate_run_authority(
    pins: CellPins, *, common: dict[str, object]
) -> dict[str, object]:
    resource, encoded = read_json_object(pins.root / "gmt_probe_resource_receipt_v1.json")
    if _sha256(encoded) != pins.resource_sha256:
        raise ValueError("resource receipt bytes differ from their pin")
    reservation = _retrospective_reservation(pins, common=common)
    inputs = resource.get("inputs")
    repository = inputs.get("repository_sources") if type(inputs) is dict else None
    supervision = inputs.get("process_supervision") if type(inputs) is dict else None
    limits = supervision.get("limits") if type(supervision) is dict else None
    config = inputs.get("config") if type(inputs) is dict else None
    if (
        resource.get("schema_version") != 1
        or resource.get("artifact") != "gmt_g1_course_development_resource_receipt"
        or resource.get("status") != "succeeded"
        or resource.get("evidence_class") != "development_course_train_not_task_success"
        or resource.get("commit") != common["source_commit"]
        or resource.get("reservation_sha256") != pins.reservation_sha256
        or type(inputs) is not dict
        or inputs.get("workload") != "course"
        or inputs.get("course_mode") != "train"
        or type(config) is not dict
        or config.get("sha256") != pins.config_sha256
        or repository
        != {
            "canonical_tree_sha256": common["source_tree_sha256"],
            "file_count": common["source_file_count"],
        }
        or type(limits) is not dict
        or limits.get("cpu_seconds") != 1_200
        or limits.get("wall_seconds") != 1_200
        or limits.get("rss_bytes") != 8 * 1024**3
        or reservation.get("commit") != common["source_commit"]
        or reservation.get("inputs") != inputs
        or reservation.get("canonical_argv") != resource.get("canonical_argv")
        or reservation.get("output") != str(pins.root)
        or reservation.get("mode") != "smoke"
        or reservation.get("hard_wall_seconds") != 1_200
        or reservation.get("required_authorizer") != "fable"
    ):
        raise ValueError("native reservation or resource binding differs")
    return {
        "reservation_sha256": pins.reservation_sha256,
        "resource_sha256": pins.resource_sha256,
        "source_tree_sha256": common["source_tree_sha256"],
        "source_file_count": common["source_file_count"],
    }


def _identities(config: Any) -> dict[str, object]:
    return {
        "task": config.task.sha256,
        "oracle": config.program.sha256,
        "reward": config.recipe.sha256,
        "segments": {name: segment.sha256 for name, segment in config.segments.items()},
    }


def _configuration(
    pins: CellPins, *, common: dict[str, object], baseline: bool
) -> tuple[Any, RunManifestBinding, CourseStudyExpectation]:
    config = load_run_config(pins.root / "input_config.json")
    if config.sha256 != pins.config_sha256:
        raise ValueError("retained config bytes differ from their pin")
    if baseline and config.sha256 != BASELINE_CONFIG_SHA256:
        raise ValueError("Study018 A config differs from the exact predata baseline")
    raw = config.raw
    if (
        raw.get("schema_version") != 5
        or raw.get("mode") != "train"
        or raw.get("seed") != SEED
        or raw.get("training_steps") != TRAINING_STEPS
        or raw.get("trainer") != EXPECTED_TRAINER
        or config.runtime != FOUR_STATE_FINITE_HORIZON_RUNTIME
        or config.runtime.observation_dim != 2_172
        or config.runtime.after_heading_reference_feedback
    ):
        raise ValueError("Study018 run differs from the frozen four-state training config")
    _validate_run_authority(pins, common=common)
    binding = RunManifestBinding(
        pins.root / "course_run_manifest.json",
        pins.manifest_sha256,
        pins.resource_sha256,
    )
    expectation = CourseStudyExpectation(
        cell_id="baseline" if baseline else "candidate",
        seed=SEED,
        training_steps=TRAINING_STEPS,
        identities=_identities(config),
        trainer=config.trainer,
        base_state_sha256=common["base_state_sha256"],
        source_commit=common["source_commit"],
        runtime=FOUR_STATE_FINITE_HORIZON_RUNTIME,
    )
    return config, binding, expectation


def verify_reward_revision(baseline: dict, candidate: dict) -> dict[str, object]:
    expected = deepcopy(baseline)
    expected["reward"] = deepcopy(candidate.get("reward"))
    if candidate != expected:
        raise ValueError("candidate changes more than the declared reward recipe")
    first = baseline.get("reward")
    second = candidate.get("reward")
    if type(first) is not dict or type(second) is not dict or set(first) != set(second):
        raise ValueError("reward recipe fields differ")
    changed = {name for name in first if first[name] != second[name]}
    if not changed or not changed <= {"lateral_weight", "heading_weight"}:
        raise ValueError("candidate must change only lateral and/or heading weight")
    lateral = second.get("lateral_weight")
    heading = second.get("heading_weight")
    if (
        type(lateral) is not float
        or not LATERAL_WEIGHT_BOUNDS[0] <= lateral <= LATERAL_WEIGHT_BOUNDS[1]
        or type(heading) is not float
        or not HEADING_WEIGHT_BOUNDS[0] <= heading <= HEADING_WEIGHT_BOUNDS[1]
    ):
        raise ValueError("candidate lateral or heading weight is outside its predata bound")
    return {
        "changed_weights": sorted(changed),
        "baseline": {
            "lateral_weight": first["lateral_weight"],
            "heading_weight": first["heading_weight"],
        },
        "candidate": {"lateral_weight": lateral, "heading_weight": heading},
        "absolute_bounds": {
            "lateral_weight": list(LATERAL_WEIGHT_BOUNDS),
            "heading_weight": list(HEADING_WEIGHT_BOUNDS),
        },
    }


def _frames(root: Path, label: str) -> list[dict]:
    rows = [
        decode_json_object(line, source=f"{label} frame {index}")
        for index, line in enumerate(
            (root / f"{label}_frames.jsonl").read_bytes().splitlines(), start=1
        )
    ]
    if not 1 <= len(rows) <= 1_000:
        raise ValueError(f"{label} frame count differs from the finite horizon")
    return rows


def _trajectory(root: Path, label: str) -> dict[str, np.ndarray]:
    return shared._trajectory(root / f"{label}_trajectory.npz")


def _reward_reconstruction(root: Path, config: Any, label: str) -> dict[str, object]:
    rows = _frames(root, label)
    totals = {"total": 0.0, "task": 0.0, "weighted_lateral": 0.0, "weighted_heading": 0.0}
    for row in rows:
        metrics = CourseStepMetrics(**row["metrics"])
        observed = row.get("reward")
        rebuilt = reward(spec=config.task, recipe=config.recipe, metrics=metrics).to_dict()
        if observed != rebuilt:
            raise ValueError(f"{label} reward differs from exact reconstruction")
        totals["total"] += rebuilt["total_reward"]
        totals["task"] += rebuilt["task_reward"]
        totals["weighted_lateral"] += (
            config.recipe.lateral_weight * rebuilt["lateral_component_reward"]
        )
        totals["weighted_heading"] += (
            config.recipe.heading_weight * rebuilt["heading_component_reward"]
        )
    return {
        "row_count": len(rows),
        "total_reward_sum": totals["total"],
        "task_reward_sum": totals["task"],
        "weighted_lateral_contribution_sum": totals["weighted_lateral"],
        "weighted_heading_contribution_sum": totals["weighted_heading"],
    }


def _training_diagnostics(root: Path, config: Any) -> dict[str, object]:
    name = telemetry_filename(
        reward_scale=config.trainer.total_training_reward_scale,
        fixed_normalizer_sha256=(
            FIXED_NORMALIZER_STATE_SHA256
            if config.trainer.uses_fixed_observation_normalizer
            else None
        ),
        runtime=config.runtime,
    )
    rows = [
        decode_json_object(line, source=f"Study018 telemetry row {index}")
        for index, line in enumerate((root / name).read_bytes().splitlines(), start=1)
    ]
    if len(rows) < 2 or rows[-1].get("event") != "final_update":
        raise ValueError("Study018 telemetry lacks its final update")
    rollout = rows[:-1]
    completed = sum(row["episodes"]["completed"] for row in rollout)
    falls = sum(row["episodes"]["falls"] for row in rollout)
    horizons = sum(row["episodes"]["horizons"] for row in rollout)
    return {
        "completed_episode_count": completed,
        "fall_count": falls,
        "horizon_completion_count": horizons,
        "final_update": rows[-1]["update"],
        "incomplete_episode_count": len(rows[-1]["incomplete_episodes"]),
    }


def verify_zero_evaluation(
    first: dict, second: dict, first_recipe_sha256: str, second_recipe_sha256: str
) -> None:
    first_reset = first.get("reset")
    second_reset = second.get("reset")
    if type(first_reset) is not dict or type(second_reset) is not dict:
        raise ValueError("zero evaluation reset is absent")
    if (
        first_reset.get("reward_sha256") != first_recipe_sha256
        or second_reset.get("reward_sha256") != second_recipe_sha256
    ):
        raise ValueError("zero evaluation reward identity differs")
    normalized_first = {k: v for k, v in first_reset.items() if k != "reward_sha256"}
    normalized_second = {k: v for k, v in second_reset.items() if k != "reward_sha256"}
    if normalized_first != normalized_second:
        raise ValueError("zero evaluation reset differs beyond reward identity")
    allowed = {"reset", "training_reward_sum_not_success_metric"}
    if {k: v for k, v in first.items() if k not in allowed} != {
        k: v for k, v in second.items() if k not in allowed
    }:
        raise ValueError("zero objective differs between reward cells")


def verify_zero_pair(first_root: Path, second_root: Path, first: Any, second: Any) -> dict:
    first_trajectory = _trajectory(first_root, "zero_residual")
    second_trajectory = _trajectory(second_root, "zero_residual")
    if set(first_trajectory) != set(second_trajectory) or any(
        not np.array_equal(first_trajectory[name], second_trajectory[name])
        for name in first_trajectory
    ):
        raise ValueError("paired zero-residual numeric trajectories differ")
    first_frames = _frames(first_root, "zero_residual")
    second_frames = _frames(second_root, "zero_residual")
    if len(first_frames) != len(second_frames):
        raise ValueError("paired zero-residual frame counts differ")
    reward_allowed = {"task_reward_recipe_sha256", "task_reward", "total_reward"}
    for left, right in zip(first_frames, second_frames, strict=True):
        if {k: v for k, v in left.items() if k != "reward"} != {
            k: v for k, v in right.items() if k != "reward"
        }:
            raise ValueError("paired zero trace differs outside reward")
        if {k: v for k, v in left["reward"].items() if k not in reward_allowed} != {
            k: v for k, v in right["reward"].items() if k not in reward_allowed
        }:
            raise ValueError("paired zero reward components differ outside weighted totals")
    first_evaluation = decode_json_object(
        (first_root / "zero_residual_evaluation.json").read_bytes(),
        source="baseline zero evaluation",
    )
    second_evaluation = decode_json_object(
        (second_root / "zero_residual_evaluation.json").read_bytes(),
        source="candidate zero evaluation",
    )
    verify_zero_evaluation(first_evaluation, second_evaluation, first.recipe.sha256, second.recipe.sha256)
    initial_first = (first_root / "initial_residual_policy.npz").read_bytes()
    initial_second = (second_root / "initial_residual_policy.npz").read_bytes()
    if initial_first != initial_second:
        raise ValueError("paired initial residual-policy bytes differ")
    return {
        "numeric_trajectory_equal": True,
        "nonreward_frame_fields_equal": True,
        "independent_zero_objective_equal": True,
        "initial_policy_byte_equal": True,
        "initial_policy_sha256": _sha256(initial_first),
    }


def verify_retained_zero(root: Path, retained: dict, config: Any) -> dict[str, object]:
    observed = _trajectory(root, "zero_residual")
    expected = retained["trajectory"]
    if set(observed) != set(expected) or any(
        not np.array_equal(observed[name], expected[name]) for name in expected
    ):
        raise ValueError("A zero-residual numeric trajectory differs from retained O7b")
    if _frames(root, "zero_residual") != retained["frames"]:
        raise ValueError("A zero-residual frame trace differs from retained O7b")
    actual = decode_json_object(
        (root / "zero_residual_evaluation.json").read_bytes(), source="A zero evaluation"
    )
    expected_evaluation = decode_json_object(
        retained["outputs"]["zero_residual_evaluation.json"],
        source="retained O7b zero evaluation",
    )
    expected_evaluation = deepcopy(expected_evaluation)
    expected_evaluation["reset"]["runtime_id"] = config.runtime.gym_runtime_id
    expected_evaluation["reset"]["course_runtime"] = config.runtime.manifest_contract()
    if actual != expected_evaluation:
        raise ValueError("A zero evaluation differs beyond declared runtime metadata")
    return {
        "retained_root": str(RETAINED_ROOT),
        "retained_manifest_sha256": RETAINED_MANIFEST_SHA256,
        "numeric_trajectory_and_actions_equal": True,
        "frame_trace_equal": True,
        "objective_equal": True,
        "only_runtime_reset_metadata_changed": True,
    }


def _validated_objective(objective: object) -> dict:
    if type(objective) is not dict:
        raise ValueError("objective is absent")
    gates = objective.get("development_gate_results")
    if (
        type(gates) is not dict
        or set(gates) != _DEVELOPMENT_GATES
        or any(type(value) is not bool for value in gates.values())
        or objective.get("development_gate_thresholds") != DEVELOPMENT_GATE_THRESHOLDS
        or objective.get("development_gate_passed") is not all(gates.values())
    ):
        raise ValueError("original eleven development gates or thresholds differ")
    return objective


def _cell_diagnostics(root: Path, config: Any, cell: dict) -> dict[str, object]:
    objective = _validated_objective(cell["objective"])
    frames = _frames(root, "final_policy")
    visits = study017.physical_region_visits(frames, objective, config.raw["task"])
    contacts = study017.recorded_ground_contacts(frames)
    headings = criteria.heading_diagnostics(frames)
    rewards = {
        label: _reward_reconstruction(root, config, label)
        for label in ("zero_residual", "final_policy")
    }
    telemetry = _training_diagnostics(root, config)
    if (
        telemetry["completed_episode_count"] != cell["training"]["episodes"]
        or telemetry["fall_count"] != cell["training"]["falls"]
    ):
        raise ValueError("training telemetry episode totals differ from scored cell")
    return {
        "frames": frames,
        "headings": headings,
        "nonfoot_contact_count": contacts["nonfoot_ground_contact_action_count"],
        "after_reentry_count": visits["after_reentry_sample_count"],
        "rewards": rewards,
        "training": telemetry,
        "physical_region_visits": visits,
        "recorded_ground_contacts": contacts,
    }


def _retained_run(temp_root: Path) -> dict:
    pins = shared.RunPins(
        root=RETAINED_ROOT.resolve(strict=True),
        manifest_sha256=RETAINED_MANIFEST_SHA256,
        resource_sha256=RETAINED_RESOURCE_SHA256,
        config_sha256=RETAINED_CONFIG_SHA256,
        source_commit=RETAINED_SOURCE_COMMIT,
    )
    return shared._verify_run(
        pins,
        temp_root,
        expected_source_tree_sha256=RETAINED_SOURCE_TREE_SHA256,
        expected_source_file_count=RETAINED_SOURCE_FILE_COUNT,
    )


def score_baseline(inputs: dict) -> dict[str, object]:
    common = _validate_common(inputs, extra_fields={"baseline"})
    pins = _cell_pins(inputs["baseline"])
    config, binding, expectation = _configuration(pins, common=common, baseline=True)
    with tempfile.TemporaryDirectory(prefix="study018-baseline-") as raw:
        temporary = Path(raw).resolve(strict=True)
        retained = _retained_run(temporary / "retained")
        cell = _score_cell(binding, expectation, temporary / "cell")
    parity = verify_retained_zero(pins.root, retained, config)
    diagnostics = _cell_diagnostics(pins.root, config, cell)
    gates = criteria.baseline_criteria(
        cell["objective"],
        nonfoot_contact_count=diagnostics["nonfoot_contact_count"],
        after_reentry_count=diagnostics["after_reentry_count"],
    )
    return {
        "schema_version": 1,
        "study": "018",
        "artifact": "gmt_g1_study018_baseline_admission/v1",
        "source_commit": common["source_commit"],
        "source_tree_sha256": common["source_tree_sha256"],
        "source_file_count": common["source_file_count"],
        "base_state_sha256": common["base_state_sha256"],
        "protocol_sha256": common["protocol_sha256"],
        "scorer_sha256": common["scorer_sha256"],
        "dependency_sha256": common["dependency_sha256"],
        "run_binding": pins.receipt(),
        "retained_zero_parity": parity,
        "baseline_criteria": gates,
        "admitted_to_reward_revision": all(gates.values()),
        "heading_diagnostics": diagnostics["headings"],
        "reward_reconstruction": diagnostics["rewards"],
        "training_diagnostics": diagnostics["training"],
        "feedback": cell["feedback"],
        "learning": cell["learning"],
        "original_development_gate_results": cell["objective"]["development_gate_results"],
        "original_objective": cell["objective"],
        "claim_scope": "one_seed_baseline_admission_not_reward_improvement_or_generalization",
    }


def _verify_baseline_admission(
    binding: object, *, common: dict[str, object], pins: CellPins
) -> dict:
    path, digest = _file_pin(binding, field="baseline admission")
    receipt, _ = _pinned_json(path, digest, source="baseline admission")
    if (
        receipt.get("artifact") != "gmt_g1_study018_baseline_admission/v1"
        or receipt.get("study") != "018"
        or receipt.get("source_commit") != common["source_commit"]
        or receipt.get("source_tree_sha256") != common["source_tree_sha256"]
        or receipt.get("source_file_count") != common["source_file_count"]
        or receipt.get("base_state_sha256") != common["base_state_sha256"]
        or receipt.get("protocol_sha256") != common["protocol_sha256"]
        or receipt.get("scorer_sha256") != common["scorer_sha256"]
        or receipt.get("dependency_sha256") != common["dependency_sha256"]
        or receipt.get("run_binding") != pins.receipt()
        or receipt.get("admitted_to_reward_revision") is not True
    ):
        raise ValueError("candidate scoring lacks the exact successful A admission")
    return receipt


def _verify_proposal(
    inputs: dict,
    *,
    baseline_config: Any,
    candidate_config: Any,
    baseline_manifest_sha256: str,
    rebuilt_feedback_sha256: str,
) -> dict[str, object]:
    proposal_path, proposal_sha = _file_pin(inputs["proposal"], field="candidate proposal")
    feedback_path, feedback_sha = _file_pin(inputs["baseline_feedback"], field="baseline feedback")
    proposal, _ = _pinned_json(proposal_path, proposal_sha, source="candidate proposal")
    feedback, feedback_bytes = _pinned_json(
        feedback_path, feedback_sha, source="baseline feedback"
    )
    if (
        proposal.get("factor") != "reward"
        or proposal.get("feedback_sha256") != feedback_sha
        or feedback_sha != rebuilt_feedback_sha256
        or feedback.get("source_manifest_sha256") != baseline_manifest_sha256
    ):
        raise ValueError("candidate proposal is not bound to rebuilt A feedback")
    generated = apply_proposal(baseline_config, proposal, feedback_bytes)
    if generated != candidate_config.raw:
        raise ValueError("candidate config differs from the sealed feedback-linked proposal")
    return {
        "proposal_path": str(proposal_path),
        "proposal_sha256": proposal_sha,
        "proposal_id": proposal["proposal_id"],
        "feedback_path": str(feedback_path),
        "feedback_sha256": feedback_sha,
        "hypothesis": proposal["hypothesis"],
    }


def score_pair(inputs: dict) -> dict[str, object]:
    extra = {"baseline", "candidate", "baseline_admission", "proposal", "baseline_feedback"}
    common = _validate_common(inputs, extra_fields=extra)
    baseline_pins = _cell_pins(inputs["baseline"])
    candidate_pins = _cell_pins(inputs["candidate"])
    admission = _verify_baseline_admission(
        inputs["baseline_admission"], common=common, pins=baseline_pins
    )
    baseline_config, baseline_binding, baseline_expected = _configuration(
        baseline_pins, common=common, baseline=True
    )
    candidate_config, candidate_binding, candidate_expected = _configuration(
        candidate_pins, common=common, baseline=False
    )
    reward_change = verify_reward_revision(baseline_config.raw, candidate_config.raw)
    pair = score_course_study_pair(
        first=baseline_binding,
        first_expected=baseline_expected,
        second=candidate_binding,
        second_expected=candidate_expected,
    )
    if (
        pair["pair"]["semantic_identity_differences"] != ["reward"]
        or pair["pair"]["course_runtime_profile_difference"]
    ):
        raise ValueError("Study018 pair differs beyond one reward identity")
    cells = pair["cells"]
    baseline_cell = cells["baseline"]
    candidate_cell = cells["candidate"]
    if any(
        cell["resource"]["source_tree_sha256"] != common["source_tree_sha256"]
        for cell in cells.values()
    ):
        raise ValueError("Study018 cells differ from the sealed executable tree")
    proposal = _verify_proposal(
        inputs,
        baseline_config=baseline_config,
        candidate_config=candidate_config,
        baseline_manifest_sha256=baseline_pins.manifest_sha256,
        rebuilt_feedback_sha256=baseline_cell["feedback"]["sha256"],
    )
    zero = verify_zero_pair(
        baseline_pins.root, candidate_pins.root, baseline_config, candidate_config
    )
    baseline_diagnostics = _cell_diagnostics(
        baseline_pins.root, baseline_config, baseline_cell
    )
    candidate_diagnostics = _cell_diagnostics(
        candidate_pins.root, candidate_config, candidate_cell
    )
    baseline_gates = criteria.baseline_criteria(
        baseline_cell["objective"],
        nonfoot_contact_count=baseline_diagnostics["nonfoot_contact_count"],
        after_reentry_count=baseline_diagnostics["after_reentry_count"],
    )
    if not all(baseline_gates.values()) or baseline_gates != admission.get("baseline_criteria"):
        raise ValueError("A admission criteria differ from exact pair-time reconstruction")
    pair_gates = criteria.pair_criteria(
        baseline_cell["objective"],
        candidate_cell["objective"],
        baseline_diagnostics["headings"],
        candidate_diagnostics["headings"],
        nonfoot_contact_count=candidate_diagnostics["nonfoot_contact_count"],
        after_reentry_count=candidate_diagnostics["after_reentry_count"],
    )
    return {
        "schema_version": 1,
        "study": "018",
        "artifact": "gmt_g1_study018_reward_pair_score/v1",
        "source_commit": common["source_commit"],
        "source_tree_sha256": common["source_tree_sha256"],
        "source_file_count": common["source_file_count"],
        "base_state_sha256": common["base_state_sha256"],
        "protocol_sha256": common["protocol_sha256"],
        "scorer_sha256": common["scorer_sha256"],
        "dependency_sha256": common["dependency_sha256"],
        "run_bindings": {
            "baseline": baseline_pins.receipt(),
            "candidate": candidate_pins.receipt(),
        },
        "baseline_admission_sha256": inputs["baseline_admission"]["sha256"],
        "proposal": proposal,
        "reward_change": reward_change,
        "zero_parity": zero,
        "pair_criteria": pair_gates,
        "study_screen_passed": all(pair_gates.values()),
        "heading_diagnostics": {
            "baseline": baseline_diagnostics["headings"],
            "candidate": candidate_diagnostics["headings"],
        },
        "reward_reconstruction": {
            "baseline": baseline_diagnostics["rewards"],
            "candidate": candidate_diagnostics["rewards"],
        },
        "training_diagnostics": {
            "baseline": baseline_diagnostics["training"],
            "candidate": candidate_diagnostics["training"],
        },
        "pair": pair,
        "original_development_gate_results": {
            name: cell["objective"]["development_gate_results"]
            for name, cell in cells.items()
        },
        "claim_scope": "one_seed_reward_revision_contrast_not_heldout_or_generalization",
        "claim_limits": [
            "A_is_a_substrate_observation_not_a_matched_old_runtime_effect",
            "depth_is_measured_not_a_predicted_improvement",
            "study_screen_is_not_full_task_qualification",
            "generated_reward_does_not_grade_itself",
        ],
    }


def _arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("baseline", "pair"):
        command = commands.add_parser(name)
        command.add_argument("--inputs", type=Path, required=True)
        command.add_argument("--inputs-sha256", required=True)
        command.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _arguments(argv)
    inputs, encoded = read_json_object(args.inputs)
    if _sha256(encoded) != _digest(
        args.inputs_sha256, length=64, field="Study018 inputs SHA-256"
    ):
        raise ValueError("Study018 inputs differ from their pinned bytes")
    result = score_baseline(inputs) if args.command == "baseline" else score_pair(inputs)
    result["inputs_sha256"] = args.inputs_sha256
    digest = write_json_receipt(args.output, result)
    summary = {
        "path": str(args.output),
        "sha256": digest,
        "study": result["study"],
        "artifact": result["artifact"],
    }
    if args.command == "baseline":
        summary.update(
            baseline_criteria=result["baseline_criteria"],
            admitted_to_reward_revision=result["admitted_to_reward_revision"],
        )
    else:
        summary.update(
            pair_criteria=result["pair_criteria"],
            study_screen_passed=result["study_screen_passed"],
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
