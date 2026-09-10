"""Focused Study018 scorer contracts; no simulator or training execution."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oracle_composition.adapters.gmt.course_evaluation import COURSE_EVALUATOR_ID
from oracle_composition.adapters.gmt.course_proposal import COURSE_FEEDBACK_EVIDENCE_CLASS
from oracle_composition.adapters.gmt.course_runtime import FOUR_STATE_FINITE_HORIZON_RUNTIME
from oracle_composition.adapters.gmt.training_contract import CourseTrainerSpec
from oracle_composition.harness.resource_slot import canonical_json_bytes

EXACT_BASELINE_CONFIG = Path(
    "/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/"
    "artifacts/gmt/course_configs/study018_native_o7b_baseline_20260907.json"
)


def _module():
    path = Path(__file__).resolve().parents[2] / "experiments/018_g1_four_state_reward_loop/score.py"
    name = "study018_score_test_module"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    return _module()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")


def _reward_config() -> dict:
    return {
        "frozen": {"same": True},
        "reward": {
            "schema_id": "gmt_g1_posture_course_reward/v1",
            "schema_version": 1,
            "speed_weight": 1.0,
            "posture_weight": 2.0,
            "lateral_weight": 1.0,
            "heading_weight": 0.5,
            "failure_weight": 1.0,
        },
    }


@pytest.mark.parametrize(
    ("lateral", "heading", "changed"),
    [
        (0.5, 0.5, ["lateral_weight"]),
        (1.0, 2.0, ["heading_weight"]),
        (4.0, 0.25, ["heading_weight", "lateral_weight"]),
    ],
)
def test_reward_revision_accepts_only_predeclared_bounds(
    module, lateral: float, heading: float, changed: list[str]
) -> None:
    baseline = _reward_config()
    candidate = deepcopy(baseline)
    candidate["reward"].update(lateral_weight=lateral, heading_weight=heading)

    result = module.verify_reward_revision(baseline, candidate)

    assert result["changed_weights"] == changed
    assert result["absolute_bounds"] == {
        "lateral_weight": [0.5, 4.0],
        "heading_weight": [0.25, 2.0],
    }


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda value: value, "must change"),
        (lambda value: value["reward"].__setitem__("lateral_weight", 0.49), "outside"),
        (lambda value: value["reward"].__setitem__("heading_weight", 2.01), "outside"),
        (lambda value: value["reward"].__setitem__("speed_weight", 1.1), "only"),
        (lambda value: value["frozen"].__setitem__("same", False), "more than"),
    ],
)
def test_reward_revision_rejects_noop_out_of_range_or_frozen_change(
    module, edit, match: str
) -> None:
    baseline = _reward_config()
    candidate = deepcopy(baseline)
    edit(candidate)
    with pytest.raises(ValueError, match=match):
        module.verify_reward_revision(baseline, candidate)


def test_zero_evaluation_allows_only_reward_identity_and_total(module) -> None:
    first = {
        "reset": {"reward_sha256": "a", "runtime_id": "same"},
        "training_reward_sum_not_success_metric": 10.0,
        "objective_evaluation": {"same": True},
    }
    second = {
        "reset": {"reward_sha256": "b", "runtime_id": "same"},
        "training_reward_sum_not_success_metric": 20.0,
        "objective_evaluation": {"same": True},
    }
    module.verify_zero_evaluation(first, second, "a", "b")

    second["objective_evaluation"] = {"same": False}
    with pytest.raises(ValueError, match="objective differs"):
        module.verify_zero_evaluation(first, second, "a", "b")


def test_retained_zero_allows_only_exact_runtime_reset_metadata(
    module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "cell"
    root.mkdir()
    arrays = {"qpos": np.zeros((2, 1))}
    frames = [{"executed_mode": "before", "metrics": {}}]
    retained_evaluation = {
        "reset": {"runtime_id": "old", "reward_sha256": "same"},
        "objective_evaluation": {"same": True},
    }
    actual = deepcopy(retained_evaluation)
    actual["reset"].update(
        runtime_id=FOUR_STATE_FINITE_HORIZON_RUNTIME.gym_runtime_id,
        course_runtime=FOUR_STATE_FINITE_HORIZON_RUNTIME.manifest_contract(),
    )
    _write_json(root / "zero_residual_evaluation.json", actual)
    retained = {
        "trajectory": arrays,
        "frames": frames,
        "outputs": {
            "zero_residual_evaluation.json": (
                json.dumps(retained_evaluation, sort_keys=True, separators=(",", ":")) + "\n"
            ).encode()
        },
    }
    monkeypatch.setattr(module, "_trajectory", lambda *_args: arrays)
    monkeypatch.setattr(module, "_frames", lambda *_args: frames)
    config = SimpleNamespace(runtime=FOUR_STATE_FINITE_HORIZON_RUNTIME)

    assert module.verify_retained_zero(root, retained, config)["frame_trace_equal"]

    monkeypatch.setattr(
        module,
        "_frames",
        lambda *_args: [{"executed_mode": "before", "metrics": {}, "runtime": "new"}],
    )
    with pytest.raises(ValueError, match="frame trace differs"):
        module.verify_retained_zero(root, retained, config)


def _common(module, root: Path) -> dict:
    dependencies = {
        "criteria": module._sha256(module._CRITERIA_PATH.read_bytes()),
        "study017_scorer": module._sha256(module._STUDY017_PATH.read_bytes()),
        "study015_scorer": module._sha256(module.study017._SHARED_PATH.read_bytes()),
    }
    return {
        "schema_version": 1,
        "scorer_sha256": module._sha256(module._SCORER_PATH.read_bytes()),
        "protocol_sha256": module._sha256(module._PROTOCOL_PATH.read_bytes()),
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_file_count": 196,
        "base_state_sha256": "c" * 64,
        "coordination_root": str(root),
        "dependency_sha256": dependencies,
        "baseline": {},
    }


def test_predata_scorer_and_dependency_integrity_fail_invalid(
    module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(module, "_validate_local_source", lambda _common: {})
    inputs = _common(module, tmp_path)
    module._validate_common(inputs, extra_fields={"baseline"})

    inputs["scorer_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="scorer bytes"):
        module._validate_common(inputs, extra_fields={"baseline"})
    inputs = _common(module, tmp_path)
    inputs["dependency_sha256"]["criteria"] = "0" * 64
    with pytest.raises(ValueError, match="dependency bytes"):
        module._validate_common(inputs, extra_fields={"baseline"})


def test_local_repository_tree_is_recomputed_and_mismatch_rejected(module) -> None:
    try:
        _root, commit, files = module.development._repository_sources(
            module._REPOSITORY_ROOT
        )
    except Exception as error:
        if "clean source worktree" in str(error):
            pytest.skip("local source binding requires the committed scorer slice")
        raise
    binding = module.development._source_tree_binding(files)
    common = {
        "source_commit": commit,
        "source_tree_sha256": binding["canonical_tree_sha256"],
        "source_file_count": binding["file_count"],
    }

    assert module._validate_local_source(common)["file_count"] == len(files)
    with pytest.raises(ValueError, match="local scoring source differs"):
        module._validate_local_source({**common, "source_tree_sha256": "0" * 64})
    with pytest.raises(ValueError, match="local scoring source differs"):
        module._validate_local_source({**common, "source_file_count": len(files) + 1})


def _authority_fixture(module, tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    reservation_path = tmp_path / "reservation.json"
    common = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_file_count": 196,
        "coordination_root": tmp_path,
    }
    inputs = {
        "workload": "course",
        "course_mode": "train",
        "config": {"sha256": "c" * 64},
        "repository_sources": {
            "canonical_tree_sha256": common["source_tree_sha256"],
            "file_count": common["source_file_count"],
        },
        "process_supervision": {
            "limits": {"cpu_seconds": 1_200, "wall_seconds": 1_200, "rss_bytes": 8 * 1024**3}
        },
    }
    argv = ["python", "-m", "course", "--output", str(run)]
    reservation = {
        "accepted_until_utc": "2099-01-01T00:00:00.000000Z",
        "canonical_argv": argv,
        "commit": common["source_commit"],
        "hard_wall_seconds": 1_200,
        "inputs": inputs,
        "mode": "smoke",
        "output": str(run),
        "required_authorizer": "fable",
    }
    _write_json(reservation_path, reservation)
    reservation_sha = hashlib.sha256(canonical_json_bytes(reservation)).hexdigest()
    resource = {
        "schema_version": 1,
        "artifact": "gmt_g1_course_development_resource_receipt",
        "status": "succeeded",
        "evidence_class": "development_course_train_not_task_success",
        "commit": common["source_commit"],
        "reservation_sha256": reservation_sha,
        "canonical_argv": argv,
        "inputs": inputs,
    }
    _write_json(run / "gmt_probe_resource_receipt_v1.json", resource)
    pins = module.CellPins(
        run,
        "c" * 64,
        "d" * 64,
        _sha(run / "gmt_probe_resource_receipt_v1.json"),
        reservation_path,
        reservation_sha,
    )
    return pins, common, reservation


def test_resource_and_native_reservation_binding_is_hard_integrity(
    module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pins, common, reservation = _authority_fixture(module, tmp_path)
    monkeypatch.setattr(module, "_retrospective_reservation", lambda *_args, **_kwargs: reservation)
    assert module._validate_run_authority(pins, common=common)["source_file_count"] == 196

    bad = module.CellPins(
        pins.root,
        pins.config_sha256,
        pins.manifest_sha256,
        "0" * 64,
        pins.reservation_path,
        pins.reservation_sha256,
    )
    with pytest.raises(ValueError, match="resource receipt bytes"):
        module._validate_run_authority(bad, common=common)


def test_exact_baseline_config_builds_frozen_study_expectation_without_running(
    module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not EXACT_BASELINE_CONFIG.is_file():
        pytest.skip("exact local Study018 baseline config is unavailable")
    root = tmp_path / "baseline"
    root.mkdir()
    (root / "input_config.json").write_bytes(EXACT_BASELINE_CONFIG.read_bytes())
    pins = module.CellPins(
        root,
        module.BASELINE_CONFIG_SHA256,
        "d" * 64,
        "e" * 64,
        tmp_path / "reservation.json",
        "f" * 64,
    )
    common = {
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_file_count": 196,
        "base_state_sha256": "c" * 64,
    }
    monkeypatch.setattr(module, "_validate_run_authority", lambda *_args, **_kwargs: {})

    config, _binding, expectation = module._configuration(
        pins, common=common, baseline=True
    )

    assert config.sha256 == module.BASELINE_CONFIG_SHA256
    assert config.trainer == CourseTrainerSpec(0.015625, profile_version=3)
    assert expectation.runtime == FOUR_STATE_FINITE_HORIZON_RUNTIME
    assert expectation.training_steps == 131_072


def test_feedback_linked_reward_proposal_reconstructs_exact_candidate_config(
    module, tmp_path: Path
) -> None:
    if not EXACT_BASELINE_CONFIG.is_file():
        pytest.skip("exact local Study018 baseline config is unavailable")
    baseline = module.load_run_config(EXACT_BASELINE_CONFIG)
    candidate_raw = deepcopy(baseline.raw)
    candidate_raw["reward"].update(lateral_weight=0.8, heading_weight=0.75)
    candidate_path = tmp_path / "candidate.json"
    _write_json(candidate_path, candidate_raw)
    candidate = module.load_run_config(candidate_path)
    manifest_sha = "f" * 64
    feedback = {
        "evidence_class": COURSE_FEEDBACK_EVIDENCE_CLASS,
        "protected_evaluation": False,
        "source_manifest_sha256": manifest_sha,
        "evaluation": {
            "evaluator_id": COURSE_EVALUATOR_ID,
            "claim_scope": "fixed_development_gate_only_not_heldout_or_universal",
            "task_sha256": baseline.task.sha256,
            "development_gate_results": {
                name: True for name in module._DEVELOPMENT_GATES
            },
            "development_gate_passed": True,
            "episode_success": None,
        },
        "diagnosis": "Predeclared Study018 reward-only revision.",
    }
    feedback_path = tmp_path / "feedback.json"
    _write_json(feedback_path, feedback)
    feedback_sha = _sha(feedback_path)
    proposal = {
        "schema_version": 1,
        "proposal_id": "study018_reward_revision_001",
        "parent_config_sha256": baseline.sha256,
        "feedback_sha256": feedback_sha,
        "factor": "reward",
        "hypothesis": "Reduce lateral error without degrading heading or compliance.",
        "replacement": {"reward": candidate.raw["reward"]},
    }
    proposal_path = tmp_path / "proposal.json"
    _write_json(proposal_path, proposal)
    inputs = {
        "proposal": {"path": str(proposal_path), "sha256": _sha(proposal_path)},
        "baseline_feedback": {"path": str(feedback_path), "sha256": feedback_sha},
    }

    result = module._verify_proposal(
        inputs,
        baseline_config=baseline,
        candidate_config=candidate,
        baseline_manifest_sha256=manifest_sha,
        rebuilt_feedback_sha256=feedback_sha,
    )

    assert result["proposal_id"] == "study018_reward_revision_001"
    with pytest.raises(ValueError, match="rebuilt A feedback"):
        module._verify_proposal(
            inputs,
            baseline_config=baseline,
            candidate_config=candidate,
            baseline_manifest_sha256=manifest_sha,
            rebuilt_feedback_sha256="0" * 64,
        )


def _npz(path: Path, offset: float = 0.0) -> None:
    np.savez(
        path,
        qpos=np.asarray([[offset]], dtype="<f8"),
        qvel=np.asarray([[0.0]], dtype="<f8"),
        residual_action=np.asarray([[0.0]], dtype="<f4"),
        current_reference=np.asarray([[0.0]], dtype="<f4"),
        composite_raw_action=np.asarray([[0.0]], dtype="<f4"),
    )


def _zero_cell(path: Path, recipe: str) -> None:
    path.mkdir()
    _npz(path / "zero_residual_trajectory.npz")
    row = {
        "executed_mode": "before",
        "metrics": {"same": True},
        "reward": {
            "task_reward_recipe_sha256": recipe,
            "task_reward": 1.0,
            "total_reward": 2.0,
            "lateral_component_reward": 0.8,
            "heading_component_reward": 0.7,
        },
    }
    _write_json(path / "zero_residual_frames.jsonl", row)
    _write_json(
        path / "zero_residual_evaluation.json",
        {
            "reset": {"reward_sha256": recipe, "runtime_id": "same"},
            "training_reward_sum_not_success_metric": 2.0,
            "objective_evaluation": {"same": True},
        },
    )
    (path / "initial_residual_policy.npz").write_bytes(b"same-policy")


def test_pair_zero_parity_checks_numeric_and_every_nonreward_field(
    module, tmp_path: Path
) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    _zero_cell(first, "a")
    _zero_cell(second, "b")
    configs = (
        SimpleNamespace(recipe=SimpleNamespace(sha256="a")),
        SimpleNamespace(recipe=SimpleNamespace(sha256="b")),
    )
    assert module.verify_zero_pair(first, second, *configs)["initial_policy_byte_equal"]

    row = json.loads((second / "zero_residual_frames.jsonl").read_text())
    row["runtime"] = "changed"
    _write_json(second / "zero_residual_frames.jsonl", row)
    with pytest.raises(ValueError, match="outside reward"):
        module.verify_zero_pair(first, second, *configs)


def test_training_diagnostics_reports_horizon_completions(
    module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "telemetry.jsonl"
    records = [
        {
            "event": "rollout_boundary",
            "episodes": {"completed": 3, "falls": 1, "horizons": 2},
        },
        {
            "event": "final_update",
            "update": {"metrics": {"explained_variance": 0.4}, "sb3_n_updates": 4},
            "incomplete_episodes": [{"length": 1}],
        },
    ]
    (tmp_path / name).write_text("".join(json.dumps(row) + "\n" for row in records))
    monkeypatch.setattr(module, "telemetry_filename", lambda **_kwargs: name)
    config = SimpleNamespace(
        trainer=CourseTrainerSpec(0.015625, profile_version=3),
        runtime=FOUR_STATE_FINITE_HORIZON_RUNTIME,
    )

    result = module._training_diagnostics(tmp_path, config)

    assert result["completed_episode_count"] == 3
    assert result["fall_count"] == 1
    assert result["horizon_completion_count"] == 2
    assert result["final_update"]["metrics"]["explained_variance"] == 0.4
