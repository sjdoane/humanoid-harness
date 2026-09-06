from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from oracle_composition.ui import g1_learning as module


def _json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fixture(root: Path) -> tuple[Path, Path, dict[str, object]]:
    run = root / "runs" / "o2-seed7"
    run.mkdir(parents=True)
    config = {"mode": "train", "seed": 7, "training_steps": 512}
    config_bytes = _json(config)
    (run / "input_config.json").write_bytes(config_bytes)
    evaluation_bytes = _json({"retained": True})
    (run / "final_policy_evaluation.json").write_bytes(evaluation_bytes)
    task_sha = "1" * 64
    gates = {"finish_reached": True, "inside_posture_compliance": False}
    objective = {
        "evaluator_id": "gmt_g1_posture_course_development_evaluator/v1",
        "claim_scope": "fixed_development_gate_only_not_heldout_or_universal",
        "task_sha256": task_sha,
        "duration_seconds": 20.0,
        "fall_count": 0,
        "finish_condition_observed": True,
        "maximum_progress_m": 4.1,
        "region": {
            "posture_compliant_fraction": 0.5,
            "minimum_root_height_m": 0.48,
        },
        "speed": {
            "mean_absolute_error_m_s": 0.2,
            "inside_mean_speed_target_deviation_m_s": 0.08,
        },
        "maximum_lateral_error_m": 0.3,
        "tracking": {
            "joint_position_rmse_rad_p95": 0.19,
            "roll_pitch_rmse_rad_p95": 0.09,
        },
        "development_gate_results": gates,
        "development_gate_passed": False,
        "episode_success": None,
    }
    manifest = {
        "schema_version": 1,
        "artifact": "gmt_g1_course_development_run",
        "status": "completed",
        "input_config_sha256": _sha(config_bytes),
        "outputs": {
            "input_config.json": _sha(config_bytes),
            "final_policy_evaluation.json": _sha(evaluation_bytes),
        },
        "identities": {
            "task": task_sha,
            "oracle": "2" * 64,
            "reward": "3" * 64,
            "segments": {"walk": "4" * 64},
        },
        "frozen_runtime": {
            "trainer": {
                "algorithm": "stable_baselines3.PPO",
                "observation_normalization": "none",
                "reward_normalization": "none",
                "trainer": {
                    "schema_id": "static_total_reward_scale/v1",
                    "schema_version": 1,
                    "total_training_reward_scale": 0.015625,
                },
            }
        },
        "training": {"completed_transitions": 512},
        "zero_residual": None,
        "final_policy": {"objective_evaluation": objective},
        "runtime": {},
        "claims": {},
    }
    manifest_bytes = _json(manifest)
    manifest_path = run / "course_run_manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    registry = {
        "schema_version": 1,
        "runs": [
            {
                "run_id": "o2-seed7",
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha(manifest_bytes),
                "label": "final_policy",
            }
        ],
    }
    registry_path = root / module.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True)
    registry_path.write_bytes(_json(registry))
    return registry_path, manifest_path, objective


def _install_validator(
    monkeypatch: pytest.MonkeyPatch,
    objective: dict[str, object],
) -> list[object]:
    calls: list[object] = []

    def verified(entry):
        calls.append(entry)
        return {
            "evidence_class": "g1_course_development_feedback/v1",
            "protected_evaluation": False,
            "source_manifest_sha256": entry.manifest_sha256,
            "evaluation": {
                name: objective[name]
                for name in (
                    "evaluator_id",
                    "claim_scope",
                    "task_sha256",
                    "development_gate_results",
                    "development_gate_passed",
                    "episode_success",
                )
            },
            "diagnosis": "validated fixture",
        }

    monkeypatch.setattr(module, "_rebuild_feedback", verified)
    return calls


def test_registered_run_uses_existing_validator_and_returns_path_free_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry_path, manifest_path, objective = _fixture(tmp_path)
    calls = _install_validator(monkeypatch, objective)

    payload = module.g1_learning_status(tmp_path)

    assert payload["state"] == "available"
    assert payload["summary"] == {
        "accepted_runs": 1,
        "rejected_runs": 0,
        "full_task_development_gate_passes": 0,
    }
    assert payload["validation"] == "manual_snapshot_sequential_existing_feedback_evaluator"
    assert payload["validated_at"].endswith("+00:00")
    assert payload["registry"]["sha256"] == _sha(registry_path.read_bytes())
    assert len(calls) == 1
    assert calls[0].manifest_path == manifest_path
    run = payload["runs"][0]
    assert run["full_task_development_gate_passed"] is False
    assert run["development_gates"] == {
        "passed": 1,
        "total": 2,
        "failed": ["inside_posture_compliance"],
    }
    assert run["evaluator"]["episode_success"] is None
    assert run["training"]["requested_transitions"] == 512
    assert run["training"]["completed_transitions"] == 512
    assert run["training"]["seed"] == 7
    assert run["training"]["producer_recorded_trainer"]["trainer"] == {
        "schema_id": "static_total_reward_scale/v1",
        "schema_version": 1,
        "total_training_reward_scale": 0.015625,
    }
    serialized = json.dumps(payload)
    assert str(tmp_path) not in serialized
    assert str(manifest_path) not in serialized


def test_missing_or_empty_registry_does_not_search_for_runs(tmp_path: Path) -> None:
    missing = module.g1_learning_status(tmp_path)
    assert missing == {
        "state": "unavailable",
        "authority": module.REGISTRY_AUTHORITY,
        "detail": "no local registry at artifacts/gmt/g1_learning_registry.json",
        "runs": [],
    }

    registry = tmp_path / module.REGISTRY_RELATIVE_PATH
    registry.parent.mkdir(parents=True)
    registry.write_bytes(_json({"schema_version": 1, "runs": []}))
    empty = module.g1_learning_status(tmp_path)
    assert empty["state"] == "empty"
    assert empty["runs"] == []


def test_tampered_manifest_is_rejected_before_existing_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _registry, manifest, objective = _fixture(tmp_path)
    calls = _install_validator(monkeypatch, objective)
    manifest.write_bytes(manifest.read_bytes() + b" ")

    payload = module.g1_learning_status(tmp_path)

    assert payload["state"] == "rejected"
    assert payload["summary"]["rejected_runs"] == 1
    assert payload["runs"][0]["detail"] == "manifest_sha256_mismatch"
    assert calls == []


def test_registry_rejects_symlinks_duplicates_and_excess_source_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, _manifest, _objective = _fixture(tmp_path / "linked")
    target = tmp_path / "registry-target.json"
    target.write_bytes(registry.read_bytes())
    registry.unlink()
    registry.symlink_to(target)
    assert module.g1_learning_status(tmp_path / "linked")["detail"] == "registry_unavailable"

    duplicate_root = tmp_path / "duplicate"
    duplicate_registry, _manifest, _objective = _fixture(duplicate_root)
    value = json.loads(duplicate_registry.read_bytes())
    value["runs"].append(dict(value["runs"][0]))
    duplicate_registry.write_bytes(_json(value))
    assert module.g1_learning_status(duplicate_root)["detail"] == "registry_duplicate_run_invalid"

    bounded_root = tmp_path / "bounded"
    _registry, _manifest, _objective = _fixture(bounded_root)
    monkeypatch.setattr(module, "MAX_AGGREGATE_SOURCE_BYTES", 1)
    bounded = module.g1_learning_status(bounded_root)
    assert bounded["detail"] == "registry_aggregate_source_bytes_exceed_limit"


def test_budget_mismatch_rejects_only_the_registered_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, manifest, objective = _fixture(tmp_path)
    _install_validator(monkeypatch, objective)
    value = json.loads(manifest.read_bytes())
    value["training"]["completed_transitions"] = 511
    manifest_bytes = _json(value)
    manifest.write_bytes(manifest_bytes)
    registry_value = json.loads(registry.read_bytes())
    registry_value["runs"][0]["manifest_sha256"] = _sha(manifest_bytes)
    registry.write_bytes(_json(registry_value))

    payload = module.g1_learning_status(tmp_path)

    assert payload["state"] == "rejected"
    assert payload["summary"]["rejected_runs"] == 1
    assert payload["runs"] == [
        {
            "state": "rejected",
            "run_id": "o2-seed7",
            "selected_label": "final_policy",
            "detail": "completed_training_budget_invalid",
        }
    ]


def test_static_view_is_manual_and_separates_g1_from_historical_native() -> None:
    static = Path(module.__file__).with_name("static")
    html = (static / "index.html").read_text(encoding="utf-8")
    script = (static / "app.js").read_text(encoding="utf-8")
    server = Path(module.__file__).with_name("server.py").read_text(encoding="utf-8")

    assert 'data-view="g1-learning"' in html
    assert "full-task development gate" in html
    assert "Historical native Humanoid-v5" in html
    assert "Protected evaluation must not steer candidate authoring." in html
    assert "The protected evaluator feeds the next iteration." not in html
    assert 'loadJson("/api/g1-learning")' in script
    assert "setInterval" not in script
    assert 'request.path == "/api/g1-learning"' in server
    assert "g1_learning_validation_lock.acquire(blocking=False)" in server
