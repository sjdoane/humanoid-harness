"""Study021 scorer checks use retained arrays only; no plant or actor executes."""

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

from oracle_composition.adapters.gmt.course_config import load_run_config
from oracle_composition.adapters.gmt.course_runtime import LOOP_RUNTIME

RETAINED_FEEDBACK = Path(
    "/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/artifacts/gmt/feedback/"
    "study019_retained_o7b_parent_20260907/feedback_v1.json"
)


def _module():
    path = Path(__file__).resolve().parents[2] / "experiments/021_g1_immediate_crouch_exit/score.py"
    spec = importlib.util.spec_from_file_location("study021_score_test_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def module():
    return _module()


@pytest.fixture(scope="module")
def retained(module):
    root = Path(module.RETAINED_BINDING["root"])
    if not root.is_dir() or not RETAINED_FEEDBACK.is_file():
        pytest.skip("exact local Study015 retained assets are unavailable")
    common = {
        "coordination_root": Path(tempfile.gettempdir()).resolve(),
        "source_commit": "unused",
    }
    pins = module.study019._run_pins(
        module.RETAINED_BINDING,
        common,
        config_sha256=module.CONTROL_CONFIG_SHA256,
        fresh=False,
    )
    with tempfile.TemporaryDirectory(
        prefix="study021-test-", dir=Path(tempfile.gettempdir()).resolve()
    ) as raw:
        run = module._verify_run(
            pins,
            Path(raw) / "feedback",
            tree=module.RETAINED_SOURCE_TREE_SHA256,
            count=module.RETAINED_SOURCE_FILE_COUNT,
        )
    return run, load_run_config(pins.root / "input_config.json")


def _write_json(path: Path, value: object) -> str:
    encoded = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _candidate_config(control, path: Path):
    raw = deepcopy(control.raw)
    raw["oracle"]["oracle_id"] = "test_only_study021_immediate_exit"
    del raw["segments"]["crouch"]["exit_at_loop_boundary"]
    _write_json(path, raw)
    return load_run_config(path)


def _common(module, root: Path) -> dict:
    return {
        "schema_version": 1,
        "scorer_sha256": module._sha256(module._SCORER_PATH.read_bytes()),
        "protocol_sha256": module._sha256(module._PROTOCOL_PATH.read_bytes()),
        "dependency_sha256": {
            name: module._sha256(path.read_bytes())
            for name, path in module._DEPENDENCY_PATHS.items()
        },
        "source_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
        "source_file_count": 196,
        "coordination_root": str(root.resolve()),
    }


def test_config_allows_only_new_oracle_id_and_absent_exit_key(
    module, retained, tmp_path: Path
) -> None:
    _run, control = retained
    candidate = _candidate_config(control, tmp_path / "candidate.json")

    change = module.verify_config(control, candidate)

    assert change["control_exit_at_loop_boundary"] is True
    assert change["candidate_exit_at_loop_boundary"] is False
    assert change["control_crouch_segment_sha256"] == module.CONTROL_CROUCH_SEGMENT_SHA256
    assert change["candidate_crouch_segment_sha256"] == module.CANDIDATE_CROUCH_SEGMENT_SHA256


@pytest.mark.parametrize(
    "mutation",
    [
        "same-oracle-id",
        "explicit-false-exit",
        "exit-guard",
        "other-segment",
        "task",
        "reward",
        "segment-identity",
    ],
)
def test_config_rejects_every_change_beyond_the_declared_factor(
    module, retained, tmp_path: Path, mutation: str
) -> None:
    _run, control = retained
    candidate = _candidate_config(control, tmp_path / f"candidate-{mutation}.json")
    raw = deepcopy(candidate.raw)
    segments = dict(candidate.segments)
    program = candidate.program
    if mutation == "same-oracle-id":
        raw["oracle"]["oracle_id"] = control.raw["oracle"]["oracle_id"]
    elif mutation == "explicit-false-exit":
        raw["segments"]["crouch"]["exit_at_loop_boundary"] = False
    elif mutation == "exit-guard":
        raw["oracle"]["transitions"][1]["guard"] = "x_travelled >= 2.04"
    elif mutation == "other-segment":
        raw["segments"]["rise"]["start_seconds"] += 0.02
    elif mutation == "task":
        raw["task"]["finish_distance_m"] += 0.1
    elif mutation == "reward":
        raw["reward"]["speed_weight"] += 0.01
    else:
        segments["crouch"] = SimpleNamespace(sha256="0" * 64, exit_at_loop_boundary=False)
    changed = SimpleNamespace(
        raw=raw,
        runtime=LOOP_RUNTIME,
        segments=segments,
        program=program,
    )
    with pytest.raises(ValueError):
        module.verify_config(control, changed)


def _prefix_fixture(module):
    size = 300
    frames = [
        {
            "executed_mode": "before" if index < 92 else "inside",
            "executed_behavior": "walk" if index < 92 else "crouch",
            "transition": None,
            "marker": index,
        }
        for index in range(size)
    ]
    frames[92]["transition"] = {
        "from_state": "before",
        "to_state": "inside",
        "selected_phase_seconds": 0.0,
        "segment_sha256": module.CONTROL_CROUCH_SEGMENT_SHA256,
    }
    trajectory = {
        "composite_raw_action": np.zeros((size, 2), dtype="<f4"),
        "residual_action": np.zeros((size, 2), dtype="<f4"),
        "current_reference": np.zeros((size, 2), dtype="<f4"),
        "qpos": np.zeros((size + 1, 2), dtype="<f8"),
        "qvel": np.zeros((size + 1, 2), dtype="<f8"),
    }
    control = {
        "frames": deepcopy(frames),
        "trajectory": {name: value.copy() for name, value in trajectory.items()},
    }
    candidate = {
        "frames": deepcopy(frames),
        "trajectory": {name: value.copy() for name, value in trajectory.items()},
    }
    candidate["frames"][92]["transition"]["segment_sha256"] = module.CANDIDATE_CROUCH_SEGMENT_SHA256
    candidate["frames"][227] = {
        **candidate["frames"][227],
        "executed_mode": "rise",
        "executed_behavior": "rise",
        "transition": {
            "from_state": "inside",
            "to_state": "rise",
            "selected_phase_seconds": 0.0,
        },
    }
    candidate["trajectory"]["current_reference"][227, 0] = 1.0

    def replay(switch: int, deferred: int):
        transition = {"action_index": switch, "selected_phase_seconds": 0.0}
        return {
            "prefix_reconstruction": {
                "action_count": 227,
                "reconstructed_command_window_prefix_shape": [227, 20, 30],
                "reconstructed_command_current_prefix_shape": [227, 30],
                "reconstructed_poststep_target_prefix_shape": [227, 30],
                "reconstructed_command_window_prefix_sha256": "a" * 64,
                "reconstructed_command_current_prefix_sha256": "b" * 64,
                "reconstructed_poststep_target_prefix_sha256": "c" * 64,
            },
            "inside_exit_boundary_deferral": {
                "first_guard_satisfaction": {"action_index": 227},
                "actual_switch": transition,
                "deferred_action_count": deferred,
            },
        }

    configs = (
        SimpleNamespace(
            segments={"crouch": SimpleNamespace(sha256=module.CONTROL_CROUCH_SEGMENT_SHA256)}
        ),
        SimpleNamespace(
            segments={"crouch": SimpleNamespace(sha256=module.CANDIDATE_CROUCH_SEGMENT_SHA256)}
        ),
    )
    return control, candidate, replay(239, 12), replay(227, 0), configs


def test_prefix_allows_only_exact_action92_identity_and_changes_at_227(module) -> None:
    control, candidate, control_replay, candidate_replay, configs = _prefix_fixture(module)
    result = module.verify_prefix(control, candidate, control_replay, candidate_replay, *configs)
    assert result["equal_action_reference_rows"] == 227
    assert result["equal_state_boundaries"] == 228
    assert result["first_changed_action_index"] == 227
    assert result["allowed_frame_identity_exception"]["field"] == ("transition.segment_sha256")


@pytest.mark.parametrize(
    "corruption",
    [
        "wrong-identity",
        "other-frame-field",
        "reference-before",
        "state-boundary",
        "prefix-window",
        "off-by-one-switch",
        "missing-numeric-change",
        "early-end",
    ],
)
def test_prefix_rejects_wrong_identity_other_fields_and_off_by_one(module, corruption: str) -> None:
    control, candidate, control_replay, candidate_replay, configs = _prefix_fixture(module)
    if corruption == "wrong-identity":
        candidate["frames"][92]["transition"]["segment_sha256"] = "0" * 64
    elif corruption == "other-frame-field":
        candidate["frames"][92]["marker"] = -1
    elif corruption == "reference-before":
        candidate["trajectory"]["current_reference"][226, 0] = 1.0
    elif corruption == "state-boundary":
        candidate["trajectory"]["qpos"][227, 0] = 1.0
    elif corruption == "prefix-window":
        candidate_replay["prefix_reconstruction"]["reconstructed_command_window_prefix_sha256"] = (
            "0" * 64
        )
    elif corruption == "off-by-one-switch":
        candidate_replay["inside_exit_boundary_deferral"]["actual_switch"]["action_index"] = 228
    elif corruption == "missing-numeric-change":
        candidate["trajectory"]["current_reference"][227] = 0.0
    else:
        candidate["frames"] = candidate["frames"][:227]
    with pytest.raises(ValueError):
        module.verify_prefix(control, candidate, control_replay, candidate_replay, *configs)


def test_previous_source_phase_is_rounded_from_its_own_raw_clock(module) -> None:
    class Segment:
        def __init__(self) -> None:
            self.boundary_inputs = []

        def reported_phase(self, phase):
            return float(phase)

        def _entry_loop_boundary_index(self, phase):
            self.boundary_inputs.append(float(phase))
            return len(self.boundary_inputs) - 1

    segment = Segment()
    config = SimpleNamespace(segments={"crouch": segment})
    replay = {
        "transitions": [
            {
                "from_state": "before",
                "to_state": "inside",
                "action_index": 92,
                "selected_phase_seconds": 0.7,
            }
        ]
    }

    raw, phases = module._source_phase(config, replay, 239)
    expected_previous = float(
        module.torch.tensor(
            0.7 + (238 - 92) * module.CONTROL_DT_SECONDS,
            dtype=module.torch.float32,
        )
    )
    rounded_subtraction = float(
        raw - module.torch.tensor(module.CONTROL_DT_SECONDS, dtype=module.torch.float32)
    )

    assert phases["previous_source_clock_unwrapped_seconds"] == expected_previous
    assert segment.boundary_inputs == [expected_previous, float(raw)]
    assert expected_previous != rounded_subtraction


def test_fall_and_contact_times_use_observation_boundary_and_action_interval(module) -> None:
    frames = [{"executed_mode": "rise"} for _ in range(228)]

    fall = module._fall_event(frames, [227], 227)
    contact = module._contact_event(frames, [227], 227)

    assert fall == {
        "action_index": 227,
        "observed_boundary_index": 228,
        "executed_mode": "rise",
        "observed_boundary_seconds_relative_to_rise": pytest.approx(0.02),
    }
    assert contact == {
        "action_index": 227,
        "executed_mode": "rise",
        "action_interval_seconds_relative_to_rise": {
            "start": pytest.approx(0.0),
            "end": pytest.approx(0.02),
        },
    }


def _criteria_fixture() -> dict:
    modes = {name: 1 for name in ("before", "inside", "rise", "after")}
    return {
        "objective": {
            "duration_seconds": 20.0,
            "fall_count": 0,
            "finish_condition_observed": True,
            "oracle_diagnostics": {
                "observed_switch_count": 3,
                "executed_mode_counts": modes,
            },
            "region": {
                "entry_observed": True,
                "exit_observed_after_entry": True,
            },
            "tracking": {
                "joint_position_rmse_rad_p95": 0.35,
                "roll_pitch_rmse_rad_p95": 0.25,
            },
        },
        "visits": {"all_visit_sample_count": 25, "after_reentry_sample_count": 0},
        "contacts": {"nonfoot_ground_contact_action_count": 0},
        "measurements": {
            "inside_exit_dispatch": {
                "guard": {"action_index": 227},
                "switch": {
                    "action_index": 227,
                    "selected_destination_phase_seconds": 0.0,
                },
                "deferred_action_count": 0,
            },
            "ordered_transitions": [
                ["before", "inside"],
                ["inside", "rise"],
                ["rise", "after"],
            ],
        },
    }


def test_screen_has_no_old_p1_and_missing_rise_is_null_failed_manipulation(module) -> None:
    diagnostics = _criteria_fixture()
    rows, _ = module.candidate_criteria(diagnostics)
    assert set(rows) == {
        "manipulation",
        "survival",
        "composition",
        "exposure",
        "recovery_guardrail",
        "tracking",
    }
    diagnostics["measurements"]["inside_exit_dispatch"] = {
        "guard": {"action_index": 227},
        "switch": None,
        "deferred_action_count": None,
    }
    diagnostics["objective"]["oracle_diagnostics"] = {
        "observed_switch_count": 1,
        "executed_mode_counts": {"before": 1, "inside": 1},
    }
    diagnostics["measurements"]["ordered_transitions"] = [["before", "inside"]]
    rows, checks = module.candidate_criteria(diagnostics)
    assert checks["first_rise_switch_at_action_227"] is False
    assert rows["manipulation"] is False
    assert rows["composition"] is False


@pytest.mark.parametrize(
    ("case", "row"),
    [
        ("guard", "manipulation"),
        ("switch", "manipulation"),
        ("deferral", "manipulation"),
        ("selected-phase", "manipulation"),
        ("duration", "survival"),
        ("fall", "survival"),
        ("contact", "survival"),
        ("switch-count", "composition"),
        ("mode-set", "composition"),
        ("transition-order", "composition"),
        ("entry", "exposure"),
        ("exit", "exposure"),
        ("finish", "exposure"),
        ("region-samples", "exposure"),
        ("after-reentry", "recovery_guardrail"),
        ("joint-p95", "tracking"),
        ("roll-pitch-p95", "tracking"),
    ],
)
def test_every_screen_conjunct_has_a_negative(module, case: str, row: str) -> None:
    diagnostics = _criteria_fixture()
    if case == "guard":
        diagnostics["measurements"]["inside_exit_dispatch"]["guard"]["action_index"] = 226
    elif case == "switch":
        diagnostics["measurements"]["inside_exit_dispatch"]["switch"]["action_index"] = 228
    elif case == "deferral":
        diagnostics["measurements"]["inside_exit_dispatch"]["deferred_action_count"] = 1
    elif case == "selected-phase":
        diagnostics["measurements"]["inside_exit_dispatch"]["switch"][
            "selected_destination_phase_seconds"
        ] = 0.02
    elif case == "duration":
        diagnostics["objective"]["duration_seconds"] = 19.98
    elif case == "fall":
        diagnostics["objective"]["fall_count"] = 1
    elif case == "contact":
        diagnostics["contacts"]["nonfoot_ground_contact_action_count"] = 1
    elif case == "switch-count":
        diagnostics["objective"]["oracle_diagnostics"]["observed_switch_count"] = 2
    elif case == "mode-set":
        del diagnostics["objective"]["oracle_diagnostics"]["executed_mode_counts"]["after"]
    elif case == "transition-order":
        diagnostics["measurements"]["ordered_transitions"][1] = ["inside", "after"]
    elif case == "entry":
        diagnostics["objective"]["region"]["entry_observed"] = False
    elif case == "exit":
        diagnostics["objective"]["region"]["exit_observed_after_entry"] = False
    elif case == "finish":
        diagnostics["objective"]["finish_condition_observed"] = False
    elif case == "region-samples":
        diagnostics["visits"]["all_visit_sample_count"] = 24
    elif case == "after-reentry":
        diagnostics["visits"]["after_reentry_sample_count"] = 1
    elif case == "joint-p95":
        diagnostics["objective"]["tracking"]["joint_position_rmse_rad_p95"] = 0.351
    else:
        diagnostics["objective"]["tracking"]["roll_pitch_rmse_rad_p95"] = 0.251
    rows, _ = module.candidate_criteria(diagnostics)
    assert rows[row] is False


def test_real_retained_numeric_replay_and_boundary_diagnostics(module, retained) -> None:
    run, config = retained
    replay = module._replay_native_loop(config, run)
    prefix = module._reconstructed_prefix(config, run)
    dispatch = module._dispatch_measurements(config, run, replay)
    identities = module._command_segment_identities(config, run, replay)

    assert replay["command_count"] == 1000
    assert replay["reconstructed_window_shape"] == [1000, 20, 30]
    assert prefix["action_count"] == 227
    assert prefix["reconstructed_command_window_prefix_shape"] == [227, 20, 30]
    assert prefix["reconstructed_command_current_prefix_shape"] == [227, 30]
    assert prefix["reconstructed_poststep_target_prefix_shape"] == [227, 30]
    assert dispatch["guard"]["action_index"] == 227
    assert dispatch["guard"]["source_phase_reported_seconds"] == pytest.approx(2.70)
    assert dispatch["switch"]["action_index"] == 239
    assert dispatch["switch"]["source_clock_unwrapped_seconds"] == pytest.approx(2.94)
    assert dispatch["switch"]["source_phase_reported_seconds"] == pytest.approx(1.08)
    assert dispatch["switch"]["previous_entry_loop_boundary_index"] == -1
    assert dispatch["switch"]["current_entry_loop_boundary_index"] == 0
    assert dispatch["switch"]["robot_to_destination_normalized_pose_mse"] == pytest.approx(
        0.35051454377497515
    )
    assert identities["all_reconstructed_commands_checked"] is True
    assert identities["crouch_segment_sha256"] == module.CONTROL_CROUCH_SEGMENT_SHA256
    assert identities["first_crouch_action_index"] == 92
    assert identities["last_crouch_action_index"] == 238
    assert len(module._objective(run)["development_gate_results"]) == 11
    assert module._manifest_identity(run, config)["identities"]["oracle"] == config.program.sha256


def test_common_seals_self_protocol_helpers_and_local_source(
    module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = {**_common(module, tmp_path), "retained": {}, "control": {}}
    monkeypatch.setattr(module, "_validate_local_source", lambda common: common)
    module._validate_common(inputs, extra={"retained", "control"})
    boolean_schema = deepcopy(inputs)
    boolean_schema["schema_version"] = True
    with pytest.raises(ValueError, match="fields or schema"):
        module._validate_common(boolean_schema, extra={"retained", "control"})
    for field, key in (
        ("scorer_sha256", None),
        ("protocol_sha256", None),
        ("dependency_sha256", "study019_scorer"),
    ):
        changed = deepcopy(inputs)
        if key is None:
            changed[field] = "0" * 64
        else:
            changed[field][key] = "0" * 64
        with pytest.raises(ValueError):
            module._validate_common(changed, extra={"retained", "control"})


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
    monkeypatch.setattr(module, "_verify_loop_manifest", lambda _run: {"exact": True})
    monkeypatch.setattr(
        module,
        "_replay_native_loop",
        lambda *_args: {
            "inside_exit_boundary_deferral": {
                "first_guard_satisfaction": {"action_index": 227},
                "actual_switch": {"action_index": 239, "selected_phase_seconds": 0.0},
                "deferred_action_count": 12,
            }
        },
    )
    monkeypatch.setattr(module, "_objective", lambda _run: {})
    monkeypatch.setattr(module, "_manifest_identity", lambda *_args: {"exact": True})
    inputs = {"retained": module.RETAINED_BINDING, "control": {}}
    assert module.score_control(inputs)["verified_for_candidate_dispatch"] is True
    control["outputs"][next(iter(outputs))] = b"changed"
    with pytest.raises(ValueError, match="byte-reproduce"):
        module.score_control(inputs)


def test_pair_authorization_rejects_boolean_schema_and_foreign_common_pin(
    module, tmp_path: Path
) -> None:
    common = {
        "scorer_sha256": "a" * 64,
        "protocol_sha256": "b" * 64,
        "dependency_sha256": {"study019_scorer": "c" * 64},
        "source_commit": "d" * 40,
        "source_tree_sha256": "e" * 64,
        "source_file_count": 200,
    }
    inputs = {"retained": {"exact": "retained"}, "control": {"exact": "control"}}
    receipt = {
        "schema_version": 1,
        "study": "021",
        "artifact": "gmt_g1_study021_control_verification/v1",
        **common,
        "inputs_sha256": "f" * 64,
        "retained_binding": inputs["retained"],
        "control_binding": inputs["control"],
        "verified_for_candidate_dispatch": True,
        "retained_output_byte_parity": {name: True for name in module.shared._COMPARABLE_OUTPUTS},
        "proposal_parent_feedback_sha256": module.PARENT_FEEDBACK_SHA256,
        "fresh_feedback_parity": {
            "content_equal_except_source_manifest_sha256": True,
            "retained_feedback_sha256": module.PARENT_FEEDBACK_SHA256,
            "fresh_control_feedback_sha256": "0" * 64,
            "manifest_bound_feedback_sha256_different": True,
        },
        "inside_exit_boundary_deferral": {
            "actual_switch": {"action_index": module.CONTROL_RISE_ACTION_INDEX}
        },
    }
    path = tmp_path / "control-verification.json"
    digest = _write_json(path, receipt)
    binding = {"path": str(path.resolve()), "sha256": digest}
    assert module._control_receipt(binding, common, inputs)["schema_version"] == 1

    receipt["schema_version"] = True
    binding["sha256"] = _write_json(path, receipt)
    with pytest.raises(ValueError, match="lacks exact successful"):
        module._control_receipt(binding, common, inputs)

    receipt["schema_version"] = 1
    receipt["source_commit"] = "0" * 40
    binding["sha256"] = _write_json(path, receipt)
    with pytest.raises(ValueError, match="lacks exact successful"):
        module._control_receipt(binding, common, inputs)


def test_actual_proposal_firewall_requires_only_the_declared_replacement(
    module, retained, tmp_path: Path
) -> None:
    _run, control = retained
    candidate = _candidate_config(control, tmp_path / "candidate-proposal.json")
    proposal = {
        "schema_version": 1,
        "proposal_id": "test-only-study021",
        "parent_config_sha256": control.sha256,
        "feedback_sha256": module.PARENT_FEEDBACK_SHA256,
        "factor": "oracle",
        "hypothesis": "Rationale: test. Prediction: test. Falsifier: test.",
        "replacement": {
            "oracle": candidate.raw["oracle"],
            "segments": candidate.raw["segments"],
        },
    }
    proposal_path = tmp_path / "proposal.json"
    proposal_sha = _write_json(proposal_path, proposal)
    inputs = {
        "proposal": {"path": str(proposal_path), "sha256": proposal_sha},
        "parent_feedback": {
            "path": str(RETAINED_FEEDBACK),
            "sha256": module.PARENT_FEEDBACK_SHA256,
        },
    }
    result = module._proposal(inputs, control, candidate)
    assert result["proposal_id"] == proposal["proposal_id"]
    assert module.apply_proposal(control, proposal, RETAINED_FEEDBACK.read_bytes()) == candidate.raw

    proposal["hypothesis"] = "unlabeled"
    inputs["proposal"]["sha256"] = _write_json(proposal_path, proposal)
    with pytest.raises(ValueError, match="labeled feedback-linked"):
        module._proposal(inputs, control, candidate)


def test_local_repository_tree_and_import_are_recomputed(module) -> None:
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
