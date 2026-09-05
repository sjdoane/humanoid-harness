from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.harness import evaluator
from oracle_composition.harness.cycle_cli import main
from oracle_composition.harness.evaluator import CycleEvaluationError, EvaluationDependencies
from oracle_composition.harness.inputs import load_frozen_inputs


class _Actor:
    def __init__(self, speed_m_s: float) -> None:
        self._delta = np.float32(speed_m_s * 0.015)

    def act(self, _observation: np.ndarray) -> SimpleNamespace:
        action = np.zeros(17, dtype="<f4")
        action[0] = self._delta
        return SimpleNamespace(physical_action=action)


class _Environment:
    def __init__(self, *, fall_at_boundary: int | None = None) -> None:
        self.unwrapped = self
        self.data = SimpleNamespace(qpos=np.zeros(24, dtype=np.float64))
        self._step = 0
        self._fall_at_boundary = fall_at_boundary

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
        if self._fall_at_boundary is not None and self._step >= self._fall_at_boundary:
            self.data.qpos[2] = 0.9
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


def test_cli_smoke_two_episodes_by_fifty_steps_runs_three_cycles(tmp_path: Path) -> None:
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

    candidate_path = experiment / "cycles/cycle_1/oracle_1.json"
    candidate_path.write_bytes(
        canonical_json_bytes(_oracle("cycle_1_candidate", "expert", "simple"))
    )
    cycle_one_dependencies = EvaluationDependencies(
        environment_factory=lambda: _Environment(fall_at_boundary=20),
        actor_loader=dependencies.actor_loader,
        fingerprint_factory=dependencies.fingerprint_factory,
    )
    assert (
        main(
            [
                "evaluate",
                "--experiment",
                str(experiment),
                "--cycle",
                "1",
                "--oracle",
                str(candidate_path),
            ],
            repository_root=tmp_path,
            dependencies=cycle_one_dependencies,
        )
        == 0
    )
    cycle_one = json.loads((experiment / "cycles/cycle_1/report_1.json").read_bytes())
    assert [arm["oracle_id"] for arm in cycle_one["summary"]["arms"]] == [
        "single_fast",
        "single_slow",
        "playback",
        "handwritten",
        "cycle_1_candidate",
    ]
    assert cycle_one["cycle_zero_comparison"]["arm_count"] == 4
    assert cycle_one["comparison_to_cycle_zero"]["never_fall_requirement"] == "failed"
    assert cycle_one["comparison_to_cycle_zero"]["no_combined_ranking"] is True
    assert {
        comparison["fall_count_outcome"]
        for comparison in cycle_one["comparison_to_cycle_zero"]["comparisons"]
    } == {"worsened"}
    assert len(cycle_one["per_episode"]) == 2
    first_episode = cycle_one["per_episode"][0]
    switches = first_episode["controller_switches"]
    assert [
        {key: value for key, value in switch.items() if key != "v_x_m_s"} for switch in switches
    ] == [
        {
            "fall_followed_within_100_steps": True,
            "from_behavior": "expert",
            "step": 15,
            "to_behavior": "simple",
        },
        {
            "fall_followed_within_100_steps": False,
            "from_behavior": "simple",
            "step": 30,
            "to_behavior": "expert",
        },
    ]
    np.testing.assert_allclose([switch["v_x_m_s"] for switch in switches], [5.0, 1.0])
    assert first_episode["slow_third_behavior_fractions"] == {
        "expert": 0.0,
        "medium": 0.0,
        "simple": 1.0,
    }
    markdown = (experiment / "cycles/cycle_1/report_1.md").read_text()
    assert "| cycle 0 baseline | single_fast |" in markdown
    assert "| cycle 1 candidate | cycle_1_candidate |" in markdown
    assert "The candidate **failed** the never-fall requirement" in markdown
    assert "| 101 | 15 | expert | simple | 5.000000 | yes |" in markdown
    assert "| 101 | 30 | simple | expert | 1.000000 | no |" in markdown
    assert "| 101 | 0.000000 | 0.000000 | 1.000000 |" in markdown

    assert (
        main(
            [
                "prepare",
                "--experiment",
                str(experiment),
                "--cycle",
                "2",
                "--steer",
                "No additional steering text was supplied.",
            ],
            repository_root=tmp_path,
            dependencies=dependencies,
        )
        == 0
    )
    cycle_two_prompt = (experiment / "cycles/cycle_2/designer_prompt.md").read_text()
    assert "cycle_1_candidate" in cycle_two_prompt
    assert "No additional steering text was supplied." in cycle_two_prompt

    cycle_two_candidate_path = experiment / "cycles/cycle_2/oracle_2.json"
    cycle_two_candidate_path.write_bytes(
        canonical_json_bytes(_oracle("cycle_2_candidate", "expert"))
    )
    assert (
        main(
            [
                "evaluate",
                "--experiment",
                str(experiment),
                "--cycle",
                "2",
                "--oracle",
                str(cycle_two_candidate_path),
            ],
            repository_root=tmp_path,
            dependencies=dependencies,
        )
        == 0
    )
    cycle_two = json.loads((experiment / "cycles/cycle_2/report_2.json").read_bytes())
    assert [arm["oracle_id"] for arm in cycle_two["summary"]["arms"]] == [
        "single_fast",
        "single_slow",
        "playback",
        "handwritten",
        "cycle_1_candidate",
        "cycle_2_candidate",
    ]
    assert cycle_two["prior_cycle_comparison"]["cycle"] == 1
    assert cycle_two["prior_cycle_comparison"]["arm_count"] == 5
    assert len(cycle_two["comparison_to_prior_arms"]["comparisons"]) == 5
    cycle_two_markdown = (experiment / "cycles/cycle_2/report_2.md").read_text()
    assert "The 5 rows from the cycle-1 report are carried forward unchanged" in cycle_two_markdown
    assert "| cycle 1 candidate | cycle_1_candidate |" in cycle_two_markdown
    assert "| cycle 2 candidate | cycle_2_candidate |" in cycle_two_markdown


def test_cycle_two_rejects_incomplete_prior_arm_history(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    library, task = load_frozen_inputs(experiment)
    report_path = experiment / "cycles/cycle_1/report_1.json"
    report_path.parent.mkdir(parents=True)
    report_path.write_bytes(
        canonical_json_bytes(
            {
                "cycle": 1,
                "evidence_class": "exploratory_oracle_cycle",
                "library_manifest_sha256": library.raw_sha256,
                "summary": {
                    "arms": [{"oracle_id": oracle_id} for oracle_id in task.cycle_zero_oracle_ids]
                },
                "task_spec_sha256": task.raw_sha256,
            }
        )
    )

    with pytest.raises(CycleEvaluationError, match="prior report differs"):
        evaluator._prior_cycle_comparison(
            experiment=experiment,
            cycle=2,
            library=library,
            task=task,
        )
