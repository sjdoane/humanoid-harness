"""Focused Study019 scorer tests; no simulator or training execution."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oracle_composition.adapters.gmt.course_runtime import LOOP_RUNTIME

RETAINED_FEEDBACK = Path(
    "/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/artifacts/gmt/feedback/"
    "study019_retained_o7b_parent_20260907/feedback_v1.json"
)


def _module():
    path = (
        Path(__file__).resolve().parents[2] / "experiments/019_g1_crouch_entry_alignment/score.py"
    )
    spec = importlib.util.spec_from_file_location("study019_score_test_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    return _module()


@pytest.fixture(scope="module")
def retained(module):
    if not module.RETAINED_ROOT.is_dir() or not RETAINED_FEEDBACK.is_file():
        pytest.skip("exact local Study015 retained assets are unavailable")
    temporary_root = Path(tempfile.gettempdir()).resolve()
    common = {"coordination_root": temporary_root, "source_commit": "unused"}
    pins = module._run_pins(
        module.RETAINED_BINDING,
        common,
        config_sha256=module.RETAINED_BINDING["config_sha256"],
        fresh=False,
    )
    with tempfile.TemporaryDirectory(prefix="study019-test-", dir=temporary_root) as raw:
        run = module._verify_run(
            pins,
            Path(raw) / "feedback",
            tree=module.RETAINED_SOURCE_TREE_SHA256,
            count=module.RETAINED_SOURCE_FILE_COUNT,
        )
    return run, module.load_run_config(pins.root / "input_config.json")


def _write_json(path: Path, value: object) -> str:
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _common(module, root: Path) -> dict:
    return {
        "schema_version": 1,
        "scorer_sha256": module._sha256(module._SCORER_PATH.read_bytes()),
        "protocol_sha256": module._sha256(module._PROTOCOL_PATH.read_bytes()),
        "dependency_sha256": {
            "study017_scorer": module._sha256(module._STUDY017_PATH.read_bytes()),
            "study015_scorer": module._sha256(module.study017._SHARED_PATH.read_bytes()),
            "study018_criteria": module._sha256(module._STUDY018_CRITERIA_PATH.read_bytes()),
        },
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_file_count": 196,
        "coordination_root": str(root.resolve()),
    }


def test_real_retained_1000_row_native_replay_uses_poststep_phase(module, retained) -> None:
    run, config = retained
    entry = module.study017.physical_region_visits(
        run["frames"], run["objective"], config.raw["task"]
    )["first_entry_action_index"]

    replay = module.replay_native_loop(config, run, diagnostic_indices={entry})

    assert replay["command_count"] == 1000
    assert replay["reconstructed_window_shape"] == [1000, 20, 30]
    assert [row["action_index"] for row in replay["transitions"]] == [92, 239, 263]
    assert replay["windows_reconstructed_not_directly_retained"] is True
    assert replay["entry_boundary_dispatch"]["first_guard_satisfaction"]["action_index"] == 92
    assert replay["entry_boundary_dispatch"]["actual_switch"]["action_index"] == 92
    assert (
        replay["inside_exit_boundary_deferral"]["first_guard_satisfaction"]["action_index"] == 227
    )
    assert replay["inside_exit_boundary_deferral"]["actual_switch"]["action_index"] == 239
    target = replay["diagnostic_poststep_targets"][str(entry)]
    assert run["frames"][entry]["executed_phase_seconds"] == pytest.approx(1.16)
    assert target["poststep_phase_seconds"] == pytest.approx(1.18)
    assert target["poststep_target_height_m"] == run["trajectory"]["current_reference"][entry, 0]


@pytest.mark.parametrize("corruption", ["phase", "transition", "reference"])
def test_native_replay_rejects_phase_transition_or_reference_corruption(
    module, retained, corruption: str
) -> None:
    original, config = retained
    run = {
        **original,
        "frames": list(original["frames"]),
        "trajectory": dict(original["trajectory"]),
    }
    if corruption == "phase":
        run["frames"][0] = {**run["frames"][0], "executed_phase_seconds": 0.02}
    elif corruption == "transition":
        row = deepcopy(run["frames"][92])
        row["transition"]["selected_phase_seconds"] = 0.02
        run["frames"][92] = row
    else:
        values = run["trajectory"]["current_reference"].copy()
        values[0, 0] += np.float32(0.01)
        run["trajectory"]["current_reference"] = values

    with pytest.raises(ValueError, match="native replay"):
        module.replay_native_loop(config, run)


def _candidate_config(control, *, guard: str = "x_travelled >= 0.40", new_id: bool = True):
    raw = deepcopy(control.raw)
    if new_id:
        raw["oracle"]["oracle_id"] = "test_only_study019_candidate"
    raw["oracle"]["transitions"][0]["guard"] = guard
    return SimpleNamespace(raw=raw, runtime=LOOP_RUNTIME)


def _control_config_fixture(module):
    return SimpleNamespace(
        sha256=module.RETAINED_BINDING["config_sha256"],
        runtime=LOOP_RUNTIME,
        raw={
            "schema_version": 2,
            "mode": "probe",
            "training_steps": 0,
            "seed": module.SEED,
            "runtime": LOOP_RUNTIME.config_value,
            "oracle": {
                "oracle_id": "g1_course_o7b_balanced_loop_rise_after_crop",
                "transitions": [
                    {"from": "before", "to": "inside", "guard": "x_travelled >= 0.65"},
                    {"from": "inside", "to": "rise", "guard": "x_travelled >= 2.05"},
                    {"from": "rise", "to": "after", "guard": "dwell >= 24"},
                ],
            },
            "task": {"frozen": True},
        },
    )


def test_config_allows_only_new_oracle_id_and_0p40_guard(module) -> None:
    control = _control_config_fixture(module)
    change = module.verify_config(control, _candidate_config(control))
    assert change["candidate_guard"] == "x_travelled >= 0.40"

    with pytest.raises(ValueError, match="guard or immutable"):
        module.verify_config(control, _candidate_config(control, guard="x_travelled >= 0.41"))
    with pytest.raises(ValueError, match="guard or immutable"):
        module.verify_config(control, _candidate_config(control, new_id=False))
    candidate = _candidate_config(control)
    candidate.raw["task"]["finish_distance_m"] = 3.6
    with pytest.raises(ValueError, match="changes more"):
        module.verify_config(control, candidate)


def _criteria_fixture(phase: float) -> dict:
    return {
        "objective": {
            "duration_seconds": 20.0,
            "fall_count": 0,
            "finish_condition_observed": True,
            "oracle_diagnostics": {
                "observed_switch_count": 3,
                "executed_mode_counts": {name: 1 for name in ("before", "inside", "rise", "after")},
            },
            "tracking": {
                "joint_position_rmse_rad_p95": 0.35,
                "roll_pitch_rmse_rad_p95": 0.25,
            },
            "region": {
                "posture_compliant_fraction": 0.85,
                "entry_observed": True,
                "exit_observed_after_entry": True,
            },
        },
        "visits": {"all_visit_sample_count": 25, "after_reentry_sample_count": 0},
        "contacts": {"nonfoot_ground_contact_action_count": 0},
        "measurements": {
            "first_physical_entry_executed_mode": "inside",
            "first_physical_entry_pre_action_phase_seconds": -100.0,
            "first_physical_entry_poststep_target": {
                "poststep_phase_seconds": phase,
                "poststep_target_height_m": 0.49,
            },
        },
    }


@pytest.mark.parametrize("phase", [1.45, 1.80])
def test_manipulation_accepts_poststep_phase_boundaries(module, phase: float) -> None:
    rows, checks = module.candidate_criteria(_criteria_fixture(phase))
    assert checks["entry_poststep_phase_in_1p45_to_1p80_s"] is True
    assert rows["manipulation"] is True


@pytest.mark.parametrize("phase", [1.43, 1.82])
def test_manipulation_rejects_one_control_step_outside_poststep_phase(module, phase: float) -> None:
    rows, checks = module.candidate_criteria(_criteria_fixture(phase))
    assert checks["entry_poststep_phase_in_1p45_to_1p80_s"] is False
    assert rows["manipulation"] is False


@pytest.mark.parametrize(
    "row",
    ["p1", "survival", "composition", "exposure", "recovery_guardrail", "tracking"],
)
def test_each_nonmanipulation_screen_row_has_a_negative(module, row: str) -> None:
    diagnostics = _criteria_fixture(1.50)
    if row == "p1":
        diagnostics["objective"]["region"]["posture_compliant_fraction"] = 0.849
    elif row == "survival":
        diagnostics["objective"]["fall_count"] = 1
        diagnostics["contacts"]["nonfoot_ground_contact_action_count"] = 1
    elif row == "composition":
        diagnostics["objective"]["oracle_diagnostics"]["observed_switch_count"] = 2
        del diagnostics["objective"]["oracle_diagnostics"]["executed_mode_counts"]["after"]
    elif row == "exposure":
        diagnostics["objective"]["finish_condition_observed"] = False
        diagnostics["visits"]["all_visit_sample_count"] = 24
    elif row == "recovery_guardrail":
        diagnostics["visits"]["after_reentry_sample_count"] = 1
    else:
        diagnostics["objective"]["tracking"]["joint_position_rmse_rad_p95"] = 0.351
        diagnostics["objective"]["tracking"]["roll_pitch_rmse_rad_p95"] = 0.251

    rows, _checks = module.candidate_criteria(diagnostics)

    assert rows[row] is False


def _prefix_fixture():
    frames = [{"row": index} for index in range(100)]
    trajectory = {
        "composite_raw_action": np.zeros((100, 1), dtype="<f4"),
        "residual_action": np.zeros((100, 1), dtype="<f4"),
        "current_reference": np.zeros((100, 1), dtype="<f4"),
        "qpos": np.zeros((101, 1), dtype="<f8"),
        "qvel": np.zeros((101, 1), dtype="<f8"),
    }
    control = {
        "frames": deepcopy(frames),
        "trajectory": {k: v.copy() for k, v in trajectory.items()},
    }
    candidate = {
        "frames": deepcopy(frames),
        "trajectory": {k: v.copy() for k, v in trajectory.items()},
    }
    candidate["frames"][71]["changed"] = True
    candidate["trajectory"]["current_reference"][71, 0] = 1.0

    def replay(index: int):
        transition = {
            "action_index": index,
            "to_state": "inside",
            "selected_phase_seconds": 0.0,
        }
        return {
            "transitions": [transition],
            "entry_boundary_dispatch": {
                "first_guard_satisfaction": {"action_index": index},
                "actual_switch": transition,
                "deferred_action_count": 0,
            },
        }

    replays = (replay(92), replay(71))
    return control, candidate, replays


def test_prefix_requires_71_complete_rows_and_72_state_boundaries(module) -> None:
    control, candidate, replays = _prefix_fixture()
    assert module.verify_prefix(control, candidate, *replays) == {
        "equal_action_and_frame_rows": 71,
        "equal_state_boundaries": 72,
        "first_changed_action_index": 71,
        "first_changed_command_number_1_based": 72,
    }
    candidate["trajectory"]["qpos"][71, 0] = 1.0
    with pytest.raises(ValueError, match="qpos"):
        module.verify_prefix(control, candidate, *replays)


def test_common_seals_scorer_protocol_dependencies_and_local_source(
    module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _common(module, tmp_path)
    inputs.update(retained={}, control={})
    monkeypatch.setattr(module, "_validate_local_source", lambda common: common)
    module._validate_common(inputs, extra={"retained", "control"})

    inputs["dependency_sha256"]["study017_scorer"] = "0" * 64
    with pytest.raises(ValueError, match="dependency bytes"):
        module._validate_common(inputs, extra={"retained", "control"})
    inputs = {**_common(module, tmp_path), "retained": {}, "control": {}}
    inputs["scorer_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="scorer or protocol"):
        module._validate_common(inputs, extra={"retained", "control"})


def test_control_verification_requires_all_three_byte_identical_outputs(
    module, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = {
        "scorer_sha256": "a",
        "protocol_sha256": "b",
        "dependency_sha256": {},
        "source_commit": "c",
        "source_tree_sha256": "d",
        "source_file_count": 1,
    }
    outputs = {name: name.encode() for name in module.shared._COMPARABLE_OUTPUTS}
    retained = {
        "config_bytes": b"config",
        "outputs": outputs,
        "feedback": {"feedback": {"sha256": module.PARENT_FEEDBACK_SHA256}},
    }
    control = deepcopy(retained)
    monkeypatch.setattr(module, "_validate_common", lambda *_args, **_kwargs: common)
    monkeypatch.setattr(module, "_retained", lambda *_args: "retained-pins")
    monkeypatch.setattr(module, "_fresh", lambda *_args: (None, control, object()))
    monkeypatch.setattr(module, "_verify_run", lambda *_args, **_kwargs: retained)
    monkeypatch.setattr(module.study017, "verify_loop_manifest", lambda _run: {"exact": True})
    monkeypatch.setattr(module, "replay_native_loop", lambda *_args: {"command_count": 1000})
    monkeypatch.setattr(module, "_objective", lambda _run: {})
    monkeypatch.setattr(
        module,
        "_feedback_parity",
        lambda *_args: {
            "content_equal_except_source_manifest_sha256": True,
            "retained_feedback_sha256": module.PARENT_FEEDBACK_SHA256,
            "fresh_control_feedback_sha256": "0" * 64,
            "manifest_bound_feedback_sha256_different": True,
        },
    )
    inputs = {"retained": module.RETAINED_BINDING, "control": {}}

    assert module.score_control(inputs)["verified_for_candidate_dispatch"] is True
    control["outputs"][next(iter(outputs))] = b"changed"
    with pytest.raises(ValueError, match="byte-reproduce"):
        module.score_control(inputs)


def test_real_manifest_bound_feedback_differs_only_by_provenance(module, retained) -> None:
    retained_run, _config = retained
    root = Path(
        "/Users/samueldoane/Documents/ChatGPT/humanoid-harness-probe-runs/"
        "gmt_course_o7b_study018_legacy_control_20260907"
    )
    if not root.is_dir():
        pytest.skip("exact local Study018 legacy control is unavailable")
    root = root.resolve(strict=True)
    pins = module.shared.RunPins(
        root=root,
        manifest_sha256="923236b692259d5e06161f76d54adf657ff1fc6aa42af9f43ed9495aae83a27c",
        resource_sha256="047483c6f096ce645c76c03f0face39827c9638fc735f20521bc5590b90f83ea",
        config_sha256=module.RETAINED_BINDING["config_sha256"],
        source_commit="aef3d7d0cdbbdc15169495cce8c2def4f6a43472",
    )
    with tempfile.TemporaryDirectory(
        prefix="study019-feedback-test-", dir=Path(tempfile.gettempdir()).resolve()
    ) as raw:
        fresh = module._verify_run(
            pins,
            Path(raw) / "feedback",
            tree="6010bf0ce11bb56c30fe71fdb8b368235f7c5cde867dba8179571dff0ba11f15",
            count=196,
        )
    parity = module._feedback_parity(retained_run, fresh)
    assert parity == {
        "content_equal_except_source_manifest_sha256": True,
        "retained_feedback_sha256": module.PARENT_FEEDBACK_SHA256,
        "fresh_control_feedback_sha256": "51d24e8242555cbf224476ef3afc1f96b470f0c57c7baf970976c5dc6bd493df",
        "manifest_bound_feedback_sha256_different": True,
    }
    fresh["rebuilt_feedback"]["diagnosis"] += " changed"
    with pytest.raises(ValueError, match="beyond its new manifest provenance"):
        module._feedback_parity(retained_run, fresh)


def test_proposal_uses_actual_firewall_and_requires_labeled_hypothesis(
    module, retained, tmp_path: Path
) -> None:
    run, control = retained
    candidate = _candidate_config(control)
    proposal = {
        "schema_version": 1,
        "proposal_id": "test-only-study019",
        "parent_config_sha256": control.sha256,
        "feedback_sha256": module.PARENT_FEEDBACK_SHA256,
        "factor": "oracle",
        "hypothesis": "Rationale: test. Prediction: test. Falsifier: test.",
        "replacement": {
            "oracle": candidate.raw["oracle"],
            "segments": candidate.raw["segments"],
        },
    }
    proposal_path = tmp_path.resolve() / "proposal.json"
    proposal_sha = _write_json(proposal_path, proposal)
    generated = module.apply_proposal(control, proposal, RETAINED_FEEDBACK.read_bytes())
    candidate = SimpleNamespace(raw=generated)
    inputs = {
        "proposal": {"path": str(proposal_path), "sha256": proposal_sha},
        "parent_feedback": {
            "path": str(RETAINED_FEEDBACK),
            "sha256": module.PARENT_FEEDBACK_SHA256,
        },
    }
    assert (
        module._proposal(inputs, control, candidate, run)["proposal_id"] == proposal["proposal_id"]
    )

    proposal["hypothesis"] = "unlabeled test"
    proposal_sha = _write_json(proposal_path, proposal)
    inputs["proposal"]["sha256"] = proposal_sha
    with pytest.raises(ValueError, match="labeled feedback-linked"):
        module._proposal(inputs, control, candidate, run)


def test_local_repository_tree_is_recomputed_and_import_is_local(module) -> None:
    try:
        _root, commit, files = module.development._repository_sources(module._REPOSITORY_ROOT)
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
    assert module._validate_local_source(common)["file_count"] == binding["file_count"]
    with pytest.raises(ValueError, match="local scoring source differs"):
        module._validate_local_source({**common, "source_tree_sha256": "0" * 64})


def test_cli_exposes_separate_control_and_pair_modes(module, tmp_path: Path) -> None:
    for command in ("control", "pair"):
        args = module._arguments(
            [
                command,
                "--inputs",
                str(tmp_path / "inputs.json"),
                "--inputs-sha256",
                "a" * 64,
                "--output",
                str(tmp_path / "output.json"),
            ]
        )
        assert args.command == command
