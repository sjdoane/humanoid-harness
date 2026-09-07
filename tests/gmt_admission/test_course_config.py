from __future__ import annotations

import copy
import json

import pytest

from oracle_composition.adapters.gmt import course_config as module
from oracle_composition.adapters.gmt.course_runtime import (
    AFTER_HEADING_FEEDBACK_RUNTIME,
    AFTER_HEADING_FEEDBACK_RUNTIME_PROFILE_ID,
    FINITE_HORIZON_RUNTIME,
    FINITE_HORIZON_RUNTIME_PROFILE_ID,
    LEGACY_RUNTIME,
    LOOP_RUNTIME,
    LOOP_RUNTIME_PROFILE_ID,
)
from oracle_composition.adapters.gmt.heading_feedback import after_heading_feedback_contract


@pytest.fixture
def admitted_config(tmp_path, monkeypatch):
    from tests.gmt_admission.test_composition import motion

    weights = tmp_path / "weights.npz"
    motion_asset = tmp_path / "motion.npz"
    weights.write_bytes(b"weight fixture")
    motion_asset.write_bytes(b"motion fixture")
    monkeypatch.setattr(module, "verify_upstream_root", lambda path: None)
    monkeypatch.setattr(
        module,
        "sha256_file",
        lambda path: (
            module.ACTOR_SHA256
            if path == weights
            else module.ADMITTED_CONVERTED_MOTION_SHA256["walk_stand"]
        ),
    )
    monkeypatch.setattr(
        module.ReferenceMotion, "from_converted", lambda *args, **kwargs: motion(0.8)
    )
    raw = {
        "schema_version": 1,
        "mode": "probe",
        "seed": 42,
        "training_steps": 0,
        "assets": {
            "upstream_root": str(tmp_path),
            "weights": {"path": str(weights), "sha256": module.ACTOR_SHA256},
            "motions": {
                "walk_stand": {
                    "path": str(motion_asset),
                    "sha256": module.ADMITTED_CONVERTED_MOTION_SHA256["walk_stand"],
                }
            },
        },
        "segments": {
            "walk": {"motion_name": "walk_stand", "start_seconds": 0.0, "end_seconds": 10.0}
        },
        "task": {
            "schema_id": "gmt_g1_posture_course_task/v1",
            "schema_version": 1,
            "region_entry_distance_m": 1.0,
            "region_exit_distance_m": 2.0,
            "finish_distance_m": 3.5,
            "target_speed_outside_m_s": 0.7,
            "target_speed_inside_m_s": 0.55,
            "posture_band_low_m": 0.45,
            "posture_band_high_m": 0.60,
            "horizon_steps": 1000,
        },
        "reward": {
            "schema_id": "gmt_g1_posture_course_reward/v1",
            "schema_version": 1,
            "speed_weight": 1.0,
            "posture_weight": 2.0,
            "lateral_weight": 1.0,
            "heading_weight": 0.5,
            "failure_weight": 1.0,
        },
        "oracle": {
            "schema_version": 1,
            "evidence_class": "exploratory_oracle_cycle",
            "oracle_id": "fixture",
            "behaviors": ["walk"],
            "initial": "before",
            "states": {
                name: {"behavior": "walk", "min_dwell": 25}
                for name in ("before", "inside", "after")
            },
            "transitions": [
                {"from": "before", "to": "inside", "priority": 0, "guard": "x_travelled >= 1"},
                {"from": "inside", "to": "after", "priority": 0, "guard": "x_travelled >= 2"},
            ],
        },
    }
    return raw, tmp_path / "config.json"


def test_exact_config_and_raw_byte_identity(admitted_config):
    raw, path = admitted_config
    path.write_text(json.dumps(raw))
    admitted = module.load_run_config(path)
    assert admitted.encoded == path.read_bytes()
    assert admitted.trainer is None and "trainer" not in admitted.raw
    assert admitted.runtime == LEGACY_RUNTIME
    assert admitted.program.initial == "before"
    assert admitted.task.horizon_steps == 1000
    assert admitted.segments["walk"].duration == 10.0
    original = admitted.sha256
    path.write_text(json.dumps(raw, indent=2))
    assert module.load_run_config(path).sha256 != original


def test_exact_opt_in_trainer_profile_is_admitted_without_rewriting_input(
    admitted_config,
):
    raw, path = admitted_config
    raw.update(mode="train", training_steps=512)
    raw["trainer"] = {
        "schema_id": module.CourseTrainerSpec(1.0 / 64.0).to_dict()["schema_id"],
        "schema_version": 1,
        "total_training_reward_scale": 0.015625,
    }
    encoded = json.dumps(raw, separators=(",", ":")).encode()
    path.write_bytes(encoded)

    admitted = module.load_run_config(path)

    assert admitted.encoded == encoded
    assert admitted.raw == raw
    assert admitted.trainer == module.CourseTrainerSpec(1.0 / 64.0)


def test_exact_low_rate_trainer_profile_is_admitted_without_rewriting_input(
    admitted_config,
):
    raw, path = admitted_config
    raw.update(mode="train", training_steps=512)
    trainer = module.CourseTrainerSpec(
        1.0 / 64.0,
        profile_version=2,
    )
    raw["trainer"] = trainer.to_dict()
    encoded = json.dumps(raw, separators=(",", ":")).encode()
    path.write_bytes(encoded)

    admitted = module.load_run_config(path)

    assert admitted.encoded == encoded
    assert admitted.raw == raw
    assert admitted.trainer == trainer


def test_exact_fixed_normalizer_trainer_is_admitted_without_rewriting_input(
    admitted_config,
):
    raw, path = admitted_config
    raw.update(mode="train", training_steps=512)
    trainer = module.CourseTrainerSpec(1.0 / 64.0, profile_version=3)
    raw["trainer"] = trainer.to_dict()
    encoded = json.dumps(raw, separators=(",", ":")).encode()
    path.write_bytes(encoded)

    admitted = module.load_run_config(path)

    assert admitted.encoded == encoded
    assert admitted.raw == raw
    assert admitted.trainer == trainer


@pytest.mark.parametrize("scale", [1.0, True, float("nan"), float("inf"), "0.015625"])
def test_unadmitted_training_reward_scales_fail_closed(admitted_config, scale):
    raw, path = admitted_config
    raw.update(mode="train", training_steps=512)
    raw["trainer"] = {
        "schema_id": "gmt_g1_total_training_reward_preconditioning/v1",
        "schema_version": 1,
        "total_training_reward_scale": scale,
    }
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match=r"reward scale|JSON"):
        module.load_run_config(path)


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_trainer_schema_version_requires_exact_integer(admitted_config, schema_version):
    raw, path = admitted_config
    raw.update(mode="train", training_steps=512)
    raw["trainer"] = {
        "schema_id": "gmt_g1_total_training_reward_preconditioning/v1",
        "schema_version": schema_version,
        "total_training_reward_scale": 0.015625,
    }
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="schema identity"):
        module.load_run_config(path)


def test_trainer_rejects_unknown_fields_and_probe_mode(admitted_config):
    raw, path = admitted_config
    raw["trainer"] = {
        "schema_id": "gmt_g1_total_training_reward_preconditioning/v1",
        "schema_version": 1,
        "total_training_reward_scale": 0.015625,
        "learning_rate": 1e-3,
    }
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="trainer fields"):
        module.load_run_config(path)
    raw["trainer"].pop("learning_rate")
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="train mode"):
        module.load_run_config(path)


def test_low_rate_trainer_rejects_rate_or_field_authoring(admitted_config):
    raw, path = admitted_config
    raw.update(mode="train", training_steps=512)
    trainer = module.CourseTrainerSpec(
        1.0 / 64.0,
        profile_version=2,
    ).to_dict()
    raw["trainer"] = trainer
    for update in (
        {"learning_rate": 1e-4},
        {"learning_rate": True},
        {"learning_rate_schedule": "linear"},
    ):
        raw["trainer"] = {**trainer, **update}
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="trainer"):
            module.load_run_config(path)


def test_course_config_admits_exact_reward_v2(admitted_config):
    raw, path = admitted_config
    reward_v2 = module.TaskRewardRecipe(
        1.0,
        2.0,
        1.0,
        0.5,
        1.0,
        recipe_version=2,
        depth_strength=0.75,
        ceiling_fraction=0.6,
    )
    raw["reward"] = reward_v2.to_dict()
    path.write_text(json.dumps(raw))

    admitted = module.load_run_config(path)

    assert admitted.recipe == reward_v2
    assert admitted.recipe.sha256 != module.TaskRewardRecipe(1.0, 2.0, 1.0, 0.5, 1.0).sha256


@pytest.mark.parametrize(
    "key,value",
    [
        ("training_steps", 1),
        ("training_steps", False),
        ("mode", "execute_python"),
        ("seed", -1),
        ("seed", True),
        ("schema_version", True),
        ("extra", "ignored"),
    ],
)
def test_invalid_top_level_contract_fails(admitted_config, key, value):
    raw, path = admitted_config
    raw[key] = value
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        module.load_run_config(path)


def test_changed_actor_and_undeclared_unused_motion_fail(admitted_config):
    raw, path = admitted_config
    original = copy.deepcopy(raw)
    raw["assets"]["weights"]["sha256"] = "a" * 64
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="digest"):
        module.load_run_config(path)
    original["assets"]["motions"]["crouch_walk_stand"] = original["assets"]["motions"]["walk_stand"]
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError):
        module.load_run_config(path)


def test_caller_cannot_self_admit_different_converted_motion(admitted_config):
    raw, path = admitted_config
    raw["assets"]["motions"]["walk_stand"]["sha256"] = "a" * 64
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="outside the admitted converted"):
        module.load_run_config(path)


@pytest.mark.parametrize("steps", [1, 513, 262656])
def test_training_budget_is_exact_rollout_multiple(admitted_config, steps):
    raw, path = admitted_config
    raw.update(mode="train", training_steps=steps)
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="training steps"):
        module.load_run_config(path)


def _enable_loop_runtime(raw: dict) -> None:
    raw["schema_version"] = 2
    raw["runtime"] = {
        "schema_version": 1,
        "profile_id": LOOP_RUNTIME_PROFILE_ID,
    }
    raw["segments"] = {
        "walk": {
            "motion_name": "walk_stand",
            "start_seconds": 0.0,
            "end_seconds": 10.0,
        },
        "crouch": {
            "motion_name": "walk_stand",
            "entry_phase_end_seconds": 0.15,
            "boundary": "entry_once_then_loop",
            "loop_start_seconds": 3.9,
            "exit_at_loop_boundary": True,
            "start_seconds": 2.7,
            "end_seconds": 4.86,
        },
    }
    raw["oracle"]["behaviors"] = ["walk", "crouch"]
    raw["oracle"]["states"]["inside"]["behavior"] = "crouch"


def _enable_finite_horizon_runtime(raw: dict) -> None:
    raw["schema_version"] = 3
    raw["runtime"] = {
        "schema_version": 1,
        "profile_id": FINITE_HORIZON_RUNTIME_PROFILE_ID,
    }


def _enable_after_heading_feedback_runtime(raw: dict) -> None:
    _enable_loop_runtime(raw)
    raw["schema_version"] = 4
    raw["runtime"] = {
        "schema_version": 1,
        "profile_id": AFTER_HEADING_FEEDBACK_RUNTIME_PROFILE_ID,
    }
    raw["segments"].update(
        {
            "rise": {
                "motion_name": "walk_stand",
                "start_seconds": 5.72,
                "end_seconds": 6.5,
                "entry_phase_end_seconds": 0.0,
                "boundary": "hold_last_pose_zero_velocity",
            },
            "walk_after": {
                "motion_name": "walk_stand",
                "start_seconds": 6.5,
                "end_seconds": 7.0,
            },
        }
    )
    raw["oracle"]["behaviors"] = ["walk", "crouch", "rise", "walk_after"]
    raw["oracle"]["states"] = {
        "before": {"behavior": "walk", "min_dwell": 25},
        "inside": {"behavior": "crouch", "min_dwell": 25},
        "rise": {"behavior": "rise", "min_dwell": 24},
        "after": {"behavior": "walk_after", "min_dwell": 25},
    }
    raw["oracle"]["transitions"] = [
        {"from": "before", "to": "inside", "priority": 0, "guard": "x_travelled >= 1"},
        {"from": "inside", "to": "rise", "priority": 0, "guard": "x_travelled >= 2.05"},
        {"from": "rise", "to": "after", "priority": 0, "guard": "dwell >= 24"},
    ]


def test_config_v4_admits_only_exact_four_state_probe_with_fixed_law(admitted_config):
    raw, path = admitted_config
    _enable_after_heading_feedback_runtime(raw)
    path.write_text(json.dumps(raw))

    admitted = module.load_run_config(path)

    assert admitted.runtime == AFTER_HEADING_FEEDBACK_RUNTIME
    assert admitted.runtime.observation_dim == 2_172
    assert set(admitted.program.states) == {"before", "inside", "rise", "after"}
    manifest = admitted.runtime.manifest_contract()
    assert manifest["after_heading_reference_feedback"] == after_heading_feedback_contract()
    assert manifest["training_admitted"] is False


def test_config_v4_is_probe_only_and_requires_four_states(admitted_config):
    raw, path = admitted_config
    _enable_after_heading_feedback_runtime(raw)
    raw.update(mode="train", training_steps=512)
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="probe-only"):
        module.load_run_config(path)

    raw.update(mode="probe", training_steps=0)
    del raw["oracle"]["states"]["rise"]
    raw["oracle"]["transitions"] = [
        raw["oracle"]["transitions"][0],
        {"from": "inside", "to": "after", "priority": 0, "guard": "x_travelled >= 2"},
    ]
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="exact four-state"):
        module.load_run_config(path)


@pytest.mark.parametrize(
    "runtime",
    [
        {"schema_version": 1, "profile_id": "unknown"},
        {"schema_version": True, "profile_id": AFTER_HEADING_FEEDBACK_RUNTIME_PROFILE_ID},
        {
            "schema_version": 1,
            "profile_id": AFTER_HEADING_FEEDBACK_RUNTIME_PROFILE_ID,
            "gain": 0.4,
        },
    ],
)
def test_config_v4_rejects_runtime_aliases_and_authorable_gains(
    admitted_config, runtime
):
    raw, path = admitted_config
    _enable_after_heading_feedback_runtime(raw)
    raw["runtime"] = runtime
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="runtime profile"):
        module.load_run_config(path)


def test_config_v3_admits_exact_three_state_finite_horizon_training(admitted_config):
    raw, path = admitted_config
    _enable_finite_horizon_runtime(raw)
    raw.update(mode="train", training_steps=512)
    trainer = module.CourseTrainerSpec(1.0 / 64.0, profile_version=3)
    raw["trainer"] = trainer.to_dict()
    encoded = json.dumps(raw, separators=(",", ":")).encode()
    path.write_bytes(encoded)

    admitted = module.load_run_config(path)

    assert admitted.encoded == encoded
    assert admitted.runtime == FINITE_HORIZON_RUNTIME
    assert admitted.runtime.observation_dim == 2_171
    assert admitted.trainer == trainer
    assert admitted.runtime.manifest_contract() == {
        "schema_version": 1,
        "profile_id": FINITE_HORIZON_RUNTIME_PROFILE_ID,
        "state_observation_slots": ["before", "inside", "after"],
        "training_admitted": True,
        "termination": {
            "fall": "terminated",
            "intrinsic_horizon": "terminated",
            "truncated": False,
            "remaining_time_observation": "task_features.remaining_horizon_fraction",
        },
    }


def test_config_v3_preserves_three_state_non_loop_runtime(admitted_config):
    raw, path = admitted_config
    _enable_finite_horizon_runtime(raw)
    path.write_text(json.dumps(raw))

    admitted = module.load_run_config(path)

    assert admitted.runtime == FINITE_HORIZON_RUNTIME
    assert admitted.runtime.composition_runtime_id == LEGACY_RUNTIME.composition_runtime_id
    assert admitted.runtime.state_slots == LEGACY_RUNTIME.state_slots
    assert admitted.runtime.permits_entry_loop is False


def test_config_v2_admits_loop_profile_and_three_state_control_subset(admitted_config):
    raw, path = admitted_config
    _enable_loop_runtime(raw)
    path.write_text(json.dumps(raw))

    admitted = module.load_run_config(path)

    assert admitted.runtime == LOOP_RUNTIME
    assert admitted.runtime.observation_dim == 2_172
    assert set(admitted.program.states) == {"before", "inside", "after"}
    assert admitted.segments["crouch"].loop_start_seconds == 3.9
    assert admitted.segments["crouch"].exit_at_loop_boundary is True


def test_config_v2_admits_full_four_state_oracle(admitted_config):
    raw, path = admitted_config
    _enable_loop_runtime(raw)
    raw["segments"]["rise"] = {
        "motion_name": "walk_stand",
        "start_seconds": 5.72,
        "end_seconds": 6.5,
        "entry_phase_end_seconds": 0.0,
        "boundary": "hold_last_pose_zero_velocity",
    }
    raw["oracle"]["behaviors"] = ["walk", "crouch", "rise"]
    raw["oracle"]["states"] = {
        "before": {"behavior": "walk", "min_dwell": 25},
        "inside": {"behavior": "crouch", "min_dwell": 25},
        "rise": {"behavior": "rise", "min_dwell": 24},
        "after": {"behavior": "walk", "min_dwell": 25},
    }
    raw["oracle"]["transitions"] = [
        {"from": "before", "to": "inside", "priority": 0, "guard": "x_travelled >= 1"},
        {"from": "inside", "to": "rise", "priority": 0, "guard": "x_travelled >= 2.05"},
        {
            "from": "rise",
            "to": "after",
            "priority": 0,
            "guard": "(dwell >= 24 and z_root >= 0.70) or dwell >= 40",
        },
    ]
    path.write_text(json.dumps(raw))

    admitted = module.load_run_config(path)

    assert tuple(admitted.runtime.state_slots) == ("before", "inside", "rise", "after")
    assert set(admitted.program.states) == set(admitted.runtime.state_slots)


def test_loop_profile_is_probe_only_and_legacy_profile_prohibits_loop(admitted_config):
    raw, path = admitted_config
    legacy = copy.deepcopy(raw)
    legacy["segments"]["walk"].update(
        {
            "entry_phase_end_seconds": 0.15,
            "boundary": "entry_once_then_loop",
            "loop_start_seconds": 3.9,
        }
    )
    path.write_text(json.dumps(legacy))
    with pytest.raises(ValueError, match="legacy runtime prohibits"):
        module.load_run_config(path)

    _enable_loop_runtime(raw)
    raw.update(mode="train", training_steps=512)
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="probe-only"):
        module.load_run_config(path)


def test_loop_profile_rejects_sub_control_repeat_window(admitted_config):
    raw, path = admitted_config
    _enable_loop_runtime(raw)
    raw["segments"]["crouch"]["loop_start_seconds"] = 4.859
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="at least one control interval"):
        module.load_run_config(path)


@pytest.mark.parametrize(
    "runtime",
    [
        {"schema_version": 1, "profile_id": "unknown"},
        {"schema_version": True, "profile_id": LOOP_RUNTIME_PROFILE_ID},
        {"schema_version": 1, "profile_id": LOOP_RUNTIME_PROFILE_ID, "extra": 1},
    ],
)
def test_config_v2_requires_exact_runtime_profile(admitted_config, runtime):
    raw, path = admitted_config
    raw["schema_version"] = 2
    raw["runtime"] = runtime
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="runtime profile"):
        module.load_run_config(path)


@pytest.mark.parametrize(
    "runtime",
    [
        {"schema_version": 1, "profile_id": "unknown"},
        {"schema_version": True, "profile_id": FINITE_HORIZON_RUNTIME_PROFILE_ID},
        {
            "schema_version": 1,
            "profile_id": FINITE_HORIZON_RUNTIME_PROFILE_ID,
            "termination": "caller_authored",
        },
    ],
)
def test_config_v3_requires_exact_finite_horizon_runtime_profile(
    admitted_config, runtime
):
    raw, path = admitted_config
    raw["schema_version"] = 3
    raw["runtime"] = runtime
    path.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="runtime profile"):
        module.load_run_config(path)


def test_existing_runtime_manifest_shapes_remain_exact() -> None:
    assert LEGACY_RUNTIME.manifest_contract() is None
    assert LOOP_RUNTIME.manifest_contract() == {
        "schema_version": 1,
        "profile_id": LOOP_RUNTIME_PROFILE_ID,
        "state_observation_slots": ["before", "inside", "rise", "after"],
        "training_admitted": False,
        "loop_exit_gate": {
            "guard_sampling": "fresh_signals_only_at_first_50hz_boundary_crossing_loop_end",
            "float32_boundary_tolerance": "four_eps_times_max_source_end_or_one",
            "maximum_deferral": "one_loop_period_plus_one_control_interval",
            "control_interval_seconds": 0.02,
        },
    }


def test_loop_profile_rejects_unknown_states_recovery_and_state_only_transitions(
    admitted_config,
):
    raw, path = admitted_config
    _enable_loop_runtime(raw)
    unknown = copy.deepcopy(raw)
    unknown["oracle"]["states"]["detour"] = unknown["oracle"]["states"].pop("after")
    unknown["oracle"]["transitions"][1]["to"] = "detour"
    path.write_text(json.dumps(unknown))
    with pytest.raises(ValueError, match="runtime profile"):
        module.load_run_config(path)

    recovery = copy.deepcopy(raw)
    recovery["oracle"]["recovery"] = {
        "behavior": "walk",
        "guard": "z_root < 0.3",
        "max_duration": 25,
        "min_dwell": 5,
        "reentry_dwell": 25,
        "rejoin": "suspended_state_dwell_reset",
    }
    path.write_text(json.dumps(recovery))
    with pytest.raises(ValueError, match="does not admit recovery"):
        module.load_run_config(path)

    state_only = copy.deepcopy(raw)
    state_only["oracle"]["states"]["after"]["behavior"] = "crouch"
    path.write_text(json.dumps(state_only))
    with pytest.raises(ValueError, match="behavior-changing"):
        module.load_run_config(path)
