from __future__ import annotations

import copy
import json

import pytest

from oracle_composition.adapters.gmt import course_config as module


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
    assert admitted.program.initial == "before"
    assert admitted.task.horizon_steps == 1000
    assert admitted.segments["walk"].duration == 10.0
    original = admitted.sha256
    path.write_text(json.dumps(raw, indent=2))
    assert module.load_run_config(path).sha256 != original


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
