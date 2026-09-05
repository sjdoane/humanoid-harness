from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.harness.cycle_cli import main
from oracle_composition.harness.evaluator import EvaluationDependencies


class _Actor:
    def __init__(self, speed_m_s: float) -> None:
        self._delta = np.float32(speed_m_s * 0.015)

    def act(self, _observation: np.ndarray) -> SimpleNamespace:
        action = np.zeros(17, dtype="<f4")
        action[0] = self._delta
        return SimpleNamespace(physical_action=action)


class _Environment:
    def __init__(self) -> None:
        self.unwrapped = self
        self.data = SimpleNamespace(qpos=np.zeros(24, dtype=np.float64))
        self._step = 0

    def reset(self, *, seed: int) -> tuple[np.ndarray, dict[str, object]]:
        self._step = 0
        self.data.qpos[:] = 0.0
        self.data.qpos[0] = seed / 1000.0
        self.data.qpos[2] = 1.4
        self.data.qpos[3] = 1.0
        observation = np.zeros(348, dtype=np.float64)
        observation[0] = seed
        return observation, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        self._step += 1
        delta = float(action[0])
        self.data.qpos[0] += delta
        observation = np.zeros(348, dtype=np.float64)
        observation[0] = self.data.qpos[0]
        observation[1] = self._step
        return observation, 1.0, False, False, {"x_velocity": delta / 0.015}

    def close(self) -> None:
        pass


def _write_bound_file(root: Path, relative: str, content: bytes) -> dict[str, str]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {"path": relative, "sha256": hashlib.sha256(content).hexdigest()}


def _oracle(oracle_id: str, first: str, second: str | None = None) -> dict[str, object]:
    behaviors = [first] if second is None else [first, second]
    states: dict[str, object] = {"start": {"behavior": first, "min_dwell": 15}}
    transitions: list[dict[str, object]] = []
    if second is not None:
        states["middle"] = {"behavior": second, "min_dwell": 15}
        states["finish"] = {"behavior": first, "min_dwell": 0}
        transitions = [
            {"from": "start", "to": "middle", "guard": "t >= 15", "priority": 0},
            {"from": "middle", "to": "finish", "guard": "t >= 30", "priority": 0},
        ]
    return {
        "behaviors": behaviors,
        "evidence_class": "exploratory_oracle_cycle",
        "initial": "start",
        "oracle_id": oracle_id,
        "schema_version": 1,
        "states": states,
        "transitions": transitions,
    }


def _experiment(root: Path) -> Path:
    experiment = root / "experiments/003_composition_speed_profile"
    experiment.mkdir(parents=True)
    sources = {
        name: _write_bound_file(root, f"artifacts/sources/{name}.json", name.encode())
        for name in ("corpus_manifest_v2", "e3_certificate_v2", "validation_manifest_v2")
    }
    medians = {"expert": 5.0, "medium": 3.0, "simple": 1.0}
    behaviors: list[dict[str, object]] = []
    for name in ("expert", "medium", "simple"):
        strict = _write_bound_file(root, f"artifacts/actors/{name}.npz", f"{name}-npz".encode())
        imported = _write_bound_file(
            root, f"artifacts/actors/{name}-import.json", f"{name}-import".encode()
        )
        equivalence = _write_bound_file(
            root, f"artifacts/actors/{name}-equivalence.json", f"{name}-equivalence".encode()
        )
        behaviors.append(
            {
                "e3_falls": {
                    "e3_episode_count": 2,
                    "e3_step_count": 100,
                    "fall_event_count": 0,
                    "fall_seeds": [],
                    "falls_per_1000_steps": 0.0,
                },
                "equivalence_receipt": equivalence,
                "import_receipt": imported,
                "name": name,
                "speed_m_s": {
                    "admitted_clip_count": 2,
                    "admitted_step_count": 100,
                    "iqr_m_s": 0.2,
                    "median_m_s": medians[name],
                    "q1_m_s": medians[name] - 0.1,
                    "q3_m_s": medians[name] + 0.1,
                },
                "strict_npz": strict,
            }
        )
    manifest = {
        "behaviors": behaviors,
        "evidence_class": "exploratory_oracle_cycle",
        "fall_measurement": "test fall measurement",
        "manifest_id": "humanoid_three_actor_library/v1",
        "schema_version": 1,
        "source_evidence": sources,
        "speed_measurement": "test speed measurement",
    }
    manifest_bytes = canonical_json_bytes(manifest)
    (experiment / "library_manifest_v1.json").write_bytes(manifest_bytes)
    task = {
        "evaluation": {
            "aggregation": "median_over_predeclared_seeds",
            "cycle_zero_oracle_ids": [
                "single_fast",
                "single_slow",
                "playback",
                "handwritten",
            ],
            "determinism_replay": "repeat_first_oracle_all_seeds_and_require_equal_trace_hashes",
            "deterministic_actor_output": "mean",
            "episode_steps": 50,
            "fall_definition": "z_root_outside_open_1.0_2.0_or_torso_up_below_0.5",
            "falls_excluded": False,
            "first_fall_step": "boundary_index_0_reset_or_1_to_horizon_post_step",
            "primary_metric": "mean_absolute_speed_error_m_s",
            "seeds": [101, 102],
            "speed_quantity": "root_x_sidecar_delta_over_0.015_s",
            "task_return": "sum_stock_reward_descriptive_only",
        },
        "evidence_class": "exploratory_oracle_cycle",
        "library_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "runtime": {
            "control_period_seconds": 0.015,
            "environment_id": "Humanoid-v5",
            "frame_skip": 5,
            "observation": "stock_348d_excluding_root_xy",
            "physics_timestep_seconds": 0.003,
            "stock_reward": "untouched_descriptive_task_return_only",
            "terminate_when_unhealthy": False,
            "time_limit_steps": 1000,
        },
        "schema_version": 1,
        "slow_selection": {
            "candidates": [
                {"behavior": "medium", "below_expert_m_s": 2.0, "median_speed_m_s": 3.0},
                {"behavior": "simple", "below_expert_m_s": 4.0, "median_speed_m_s": 1.0},
            ],
            "chosen_behavior": "simple",
            "expert_behavior": "expert",
            "rule": "nonexpert_median_farthest_below_expert",
        },
        "target_schedule": [
            {"start": 0, "stop": 15, "target_m_s": 5.0},
            {"start": 15, "stop": 30, "target_m_s": 1.0},
            {"start": 30, "stop": 50, "target_m_s": 5.0},
        ],
        "task_spec_id": "humanoid_speed_profile_t1/v1",
        "task_text": "Run fast, slow, then fast without falling.",
        "trace_artifact_directory": "artifacts/experiments_003",
    }
    (experiment / "task_spec_v1.json").write_bytes(canonical_json_bytes(task))
    arms = experiment / "arms"
    arms.mkdir()
    values = (
        ("00_single_fast.json", _oracle("single_fast", "expert")),
        ("01_single_slow.json", _oracle("single_slow", "simple")),
        ("02_playback.json", _oracle("playback", "expert", "simple")),
        ("03_handwritten.json", _oracle("handwritten", "expert", "medium")),
    )
    for name, value in values:
        (arms / name).write_bytes(canonical_json_bytes(value))
    return experiment


def _fingerprint(_environment: object, _task: object) -> tuple[dict[str, object], str]:
    value = {"runtime": "fake_cli_smoke"}
    return value, hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def test_cli_smoke_two_episodes_by_fifty_steps_and_prepares_cycle_one(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    dependencies = EvaluationDependencies(
        environment_factory=_Environment,
        actor_loader=lambda entry, _root: _Actor(
            {"expert": 5.0, "medium": 3.0, "simple": 1.0}[entry.name]
        ),
        fingerprint_factory=_fingerprint,
    )
    assert (
        main(
            ["prepare", "--experiment", str(experiment), "--cycle", "0"],
            repository_root=tmp_path,
            dependencies=dependencies,
        )
        == 0
    )
    assert (
        main(
            [
                "evaluate",
                "--experiment",
                str(experiment),
                "--cycle",
                "0",
                "--oracle",
                str(experiment / "arms"),
            ],
            repository_root=tmp_path,
            dependencies=dependencies,
        )
        == 0
    )
    report_path = experiment / "cycles/cycle_0/report_0.json"
    report_bytes = report_path.read_bytes()
    report = json.loads(report_bytes)
    assert canonical_json_bytes(report) == report_bytes
    assert len(report["summary"]["arms"]) == 4
    assert len(report["per_episode"]) == 8
    assert report["determinism_check"]["replayed_episode_count"] == 2
    assert report["trace_content_index"]["entry_count"] == 8
    assert (
        main(
            [
                "prepare",
                "--experiment",
                str(experiment),
                "--cycle",
                "1",
                "--steer",
                "Prefer fewer recovery switches.",
            ],
            repository_root=tmp_path,
            dependencies=dependencies,
        )
        == 0
    )
    prompt = (experiment / "cycles/cycle_1/designer_prompt.md").read_text()
    assert "Prefer fewer recovery switches." in prompt
    assert "single_fast" in prompt
