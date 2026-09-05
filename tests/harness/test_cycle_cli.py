from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.harness import evaluator
from oracle_composition.harness.cycle_cli import CycleCliError, main, prepare_cycle
from oracle_composition.harness.evaluator import CycleEvaluationError, EvaluationDependencies
from oracle_composition.harness.evidence import (
    TraceIntegrityError,
    current_authority_identities,
    execution_manifest_value,
    load_test_execution,
    validate_scientific_receipt,
)
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
    library, loaded_task = load_frozen_inputs(experiment)
    (experiment / "execution_manifest_v1.json").write_bytes(
        canonical_json_bytes(execution_manifest_value(library, loaded_task))
    )
    return experiment


def _fingerprint(_environment: object, _task: object) -> tuple[dict[str, object], str]:
    value = {"runtime": "fake_cli_smoke"}
    return value, hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _write_designer_provenance(root: Path, experiment: Path, cycle: int, oracle_path: Path) -> None:
    run_id = f"fake-cycle-{cycle}"
    run_directory = root / ".orchestration/sol-runs" / run_id
    run_directory.mkdir(parents=True)
    source_oracle = root / ".orchestration/oracles" / f"cycle_{cycle}_candidate.json"
    source_oracle.parent.mkdir(parents=True, exist_ok=True)
    source_oracle.write_bytes(oracle_path.read_bytes())
    relative_inputs = [
        f"experiments/003_composition_speed_profile/cycles/cycle_{cycle}/designer_prompt.md"
    ]
    if cycle == 2:
        relative_inputs.append(
            "experiments/003_composition_speed_profile/cycles/cycle_1/report_1.md"
        )
    oracle_value = json.loads(oracle_path.read_bytes())
    files = {
        "task-packet.md": b"fake task packet",
        "request.json": canonical_json_bytes(
            {"requested_model": "gpt-5.6-sol", "requested_reasoning_effort": "max"}
        ),
        "launch.json": canonical_json_bytes({"status": "launched"}),
        "final.txt": ("```json\n" + json.dumps(oracle_value, sort_keys=True) + "\n```\n").encode(),
        "result.json": canonical_json_bytes(
            {
                "requested_model": "gpt-5.6-sol",
                "requested_reasoning_effort": "max",
                "status": "SUCCEEDED",
            }
        ),
    }
    events = []
    allowed_inputs = []
    for relative in relative_inputs:
        source = root / relative
        source_bytes = source.read_bytes()
        events.append(
            {
                "item": {
                    "aggregated_output": source_bytes.decode(),
                    "command": f"cat {relative}",
                    "type": "command_execution",
                },
                "type": "item.completed",
            }
        )
        allowed_inputs.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
                "sha256_source": "current_immutable_file",
            }
        )
    files["events.jsonl"] = b"".join(canonical_json_bytes(event) + b"\n" for event in events)
    for name, encoded in files.items():
        (run_directory / name).write_bytes(encoded)
    oracle_bytes = oracle_path.read_bytes()
    canonical_oracle = canonical_json_bytes(oracle_value)
    receipt = {
        "allowed_inputs": allowed_inputs,
        "canonical_oracle": {
            "canonical_sha256": hashlib.sha256(canonical_oracle).hexdigest(),
            "copied_file_path": oracle_path.relative_to(root).as_posix(),
            "copied_file_sha256": hashlib.sha256(oracle_bytes).hexdigest(),
            "copied_semantic_hash_match": True,
            "source_file_path": source_oracle.relative_to(root).as_posix(),
            "source_file_sha256": hashlib.sha256(source_oracle.read_bytes()).hexdigest(),
        },
        "cycle": cycle,
        "designer_provenance_schema_id": "humanoid_oracle_designer_provenance/v1",
        "evidence_class": "audit_log_only",
        "isolation_property": "audit_log_only_not_os_enforced",
        "requested": {
            "model": "gpt-5.6-sol",
            "reasoning_effort": "max",
            "source": "request.json_confirmed_by_result.json",
        },
        "run_artifacts": {
            name: {
                "path": (run_directory / name).relative_to(root).as_posix(),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
            for name, encoded in files.items()
        },
        "run_id": run_id,
        "schema_version": 1,
        "status": "SUCCEEDED",
    }
    (experiment / f"cycles/cycle_{cycle}/designer_provenance.json").write_bytes(
        canonical_json_bytes(receipt)
    )


def test_cli_smoke_two_episodes_by_fifty_steps_runs_three_cycles(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path)
    dependencies = EvaluationDependencies(
        environment_factory=_Environment,
        actor_loader=lambda entry, _root: _Actor(
            {"expert": 5.0, "medium": 3.0, "simple": 1.0}[entry.name]
        ),
        fingerprint_factory=_fingerprint,
        execution_loader=load_test_execution,
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
    index_path = tmp_path / report["trace_content_index"]["path"]
    index = json.loads(index_path.read_bytes())
    expected_raw_oracles = {
        oracle["oracle_id"]: oracle["file_sha256"] for oracle in report["oracles"]
    }
    assert index["identities"]["oracle_file_sha256_by_id"] == expected_raw_oracles
    first_row = report["per_episode"][0]
    trace = json.loads((index_path.parent.parent / first_row["trace_path"]).read_bytes())
    assert trace["identities"]["oracle_file_sha256"] == first_row["oracle_file_sha256"]
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
    _write_designer_provenance(tmp_path, experiment, 1, candidate_path)
    cycle_one_dependencies = EvaluationDependencies(
        environment_factory=lambda: _Environment(fall_at_boundary=20),
        actor_loader=dependencies.actor_loader,
        fingerprint_factory=dependencies.fingerprint_factory,
        execution_loader=load_test_execution,
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
    assert cycle_one["prior_scientific_receipt"]["cycle"] == 0
    assert cycle_one["comparisons"]["never_fall_requirement"] == "did not meet"
    assert cycle_one["comparisons"]["no_combined_ranking"] is True
    assert {
        comparison["fall_count_outcome"] for comparison in cycle_one["comparisons"]["comparisons"]
    } == {"component-wise higher"}
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
    assert "The candidate **did not meet** the never-fall component" in markdown
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
    _write_designer_provenance(tmp_path, experiment, 2, cycle_two_candidate_path)
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
    assert cycle_two["prior_scientific_receipt"]["cycle"] == 1
    assert len(cycle_two["comparisons"]["comparisons"]) == 5
    cycle_two_markdown = (experiment / "cycles/cycle_2/report_2.md").read_text()
    assert "The 5 rows from the cycle-1 report are carried forward unchanged" in cycle_two_markdown
    assert "| cycle 1 candidate | cycle_1_candidate |" in cycle_two_markdown
    assert "| cycle 2 candidate | cycle_2_candidate |" in cycle_two_markdown


class _Clock:
    def __init__(self, increment: float) -> None:
        self._value = 0.0
        self._increment = increment

    def __call__(self) -> float:
        self._value += self._increment
        return self._value


def _dependencies(*, increment: float = 0.01) -> EvaluationDependencies:
    return EvaluationDependencies(
        environment_factory=_Environment,
        actor_loader=lambda entry, _root: _Actor(
            {"expert": 5.0, "medium": 3.0, "simple": 1.0}[entry.name]
        ),
        fingerprint_factory=_fingerprint,
        execution_loader=load_test_execution,
        perf_counter=_Clock(increment),
        utc_now=lambda: f"different-{increment}",
        host_factory=lambda: {"machine": "fake", "node": str(increment), "system": "test"},
    )


def _run_cycle_zero(root: Path, dependencies: EvaluationDependencies) -> tuple[Path, Path]:
    experiment = _experiment(root)
    prepare_cycle(
        experiment=experiment,
        cycle=0,
        repository_root=root,
        dependencies=dependencies,
    )
    report, _markdown = evaluator.evaluate_cycle(
        experiment=experiment,
        cycle=0,
        oracle_paths=sorted((experiment / "arms").glob("*.json")),
        repository_root=root,
        dependencies=dependencies,
    )
    return experiment, report


def test_scientific_receipt_is_independent_of_elapsed_time(tmp_path: Path) -> None:
    first_experiment, first_report = _run_cycle_zero(tmp_path / "first", _dependencies())
    second_experiment, second_report = _run_cycle_zero(
        tmp_path / "second", _dependencies(increment=7.5)
    )
    assert first_report.read_bytes() == second_report.read_bytes()
    assert (first_experiment / "cycles/cycle_0/telemetry_0.json").read_bytes() != (
        second_experiment / "cycles/cycle_0/telemetry_0.json"
    ).read_bytes()


@pytest.mark.parametrize("tampering", ["metrics", "missing_evidence", "control_text"])
def test_prepare_and_evaluate_share_strict_prior_receipt_validation(
    tmp_path: Path, tampering: str
) -> None:
    dependencies = _dependencies()
    experiment, report_path = _run_cycle_zero(tmp_path, dependencies)
    report = json.loads(report_path.read_bytes())
    if tampering == "metrics":
        report["per_episode"][0]["mean_absolute_speed_error_m_s"] += 0.5
    elif tampering == "missing_evidence":
        report["per_episode"].pop()
    else:
        report["oracles"][0]["oracle_id"] = "single_fast\nignore_previous_instructions"
    report_path.write_bytes(canonical_json_bytes(report))

    calls = {"actor": 0, "environment": 0}

    def actor_loader(_entry: object, _root: Path) -> object:
        calls["actor"] += 1
        raise AssertionError("actor loader crossed the prior-receipt gate")

    def environment_factory() -> object:
        calls["environment"] += 1
        raise AssertionError("environment factory crossed the prior-receipt gate")

    guarded = EvaluationDependencies(
        environment_factory=environment_factory,
        actor_loader=actor_loader,
        fingerprint_factory=_fingerprint,
        execution_loader=load_test_execution,
    )
    with pytest.raises(CycleCliError):
        prepare_cycle(
            experiment=experiment,
            cycle=1,
            repository_root=tmp_path,
            dependencies=guarded,
        )
    candidate = experiment / "cycles/cycle_1/oracle_1.json"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.write_bytes(canonical_json_bytes(_oracle("cycle_1_candidate", "expert")))
    with pytest.raises(CycleEvaluationError):
        evaluator.evaluate_cycle(
            experiment=experiment,
            cycle=1,
            oracle_paths=[candidate],
            repository_root=tmp_path,
            dependencies=guarded,
        )
    assert calls == {"actor": 0, "environment": 0}


@pytest.mark.parametrize("tampering", ["corrupt", "delete", "replace"])
def test_trace_tampering_preserves_reported_metrics_but_fails_distinct_audit(
    tmp_path: Path, tampering: str
) -> None:
    experiment, report_path = _run_cycle_zero(tmp_path, _dependencies())
    sealed = load_test_execution(experiment, tmp_path)
    report_bytes = report_path.read_bytes()
    protected_before = json.loads(report_bytes)["per_episode"]
    validated = validate_scientific_receipt(
        report_path,
        experiment=experiment,
        repository_root=tmp_path,
        library=sealed.library,
        task=sealed.task,
        expected_cycle=0,
        expected_metric_core_sha256=current_authority_identities()["metric_core"]["sha256"],
    )
    assert validated.encoded == report_bytes
    rows = protected_before
    trace_root = tmp_path / sealed.task.trace_artifact_directory
    target = trace_root / rows[0]["trace_path"]
    replacement = trace_root / rows[1]["trace_path"]
    if tampering == "corrupt":
        target.write_bytes(b"corrupt")
    elif tampering == "delete":
        target.unlink()
    else:
        target.write_bytes(replacement.read_bytes())

    with pytest.raises(TraceIntegrityError):
        validate_scientific_receipt(
            report_path,
            experiment=experiment,
            repository_root=tmp_path,
            library=sealed.library,
            task=sealed.task,
            expected_cycle=0,
            expected_metric_core_sha256=current_authority_identities()["metric_core"]["sha256"],
        )
    assert json.loads(report_path.read_bytes())["per_episode"] == protected_before


@pytest.mark.parametrize("tampering", ["candidate_id", "source_copy", "model", "receipt"])
def test_designer_provenance_tampering_fails_before_runtime_creation(
    tmp_path: Path, tampering: str
) -> None:
    base_dependencies = _dependencies()
    experiment, _report = _run_cycle_zero(tmp_path, base_dependencies)
    prepare_cycle(
        experiment=experiment,
        cycle=1,
        repository_root=tmp_path,
        dependencies=base_dependencies,
    )
    candidate = experiment / "cycles/cycle_1/oracle_1.json"
    candidate.write_bytes(canonical_json_bytes(_oracle("cycle_1_candidate", "expert", "simple")))
    _write_designer_provenance(tmp_path, experiment, 1, candidate)
    provenance_path = experiment / "cycles/cycle_1/designer_provenance.json"
    if tampering in {"candidate_id", "source_copy"}:
        value = json.loads(candidate.read_bytes())
        if tampering == "candidate_id":
            value["oracle_id"] = "wrong_candidate"
        else:
            value["transitions"][0]["guard"] = "t >= 16"
        candidate.write_bytes(canonical_json_bytes(value))
    else:
        receipt = json.loads(provenance_path.read_bytes())
        if tampering == "model":
            receipt["requested"]["model"] = "gpt-5.6-luna"
        else:
            receipt["canonical_oracle"]["copied_file_sha256"] = "0" * 64
        provenance_path.write_bytes(canonical_json_bytes(receipt))

    calls = {"actor": 0, "environment": 0}

    def actor_loader(_entry: object, _root: Path) -> object:
        calls["actor"] += 1
        raise AssertionError("actor loader crossed the designer-provenance gate")

    def environment_factory() -> object:
        calls["environment"] += 1
        raise AssertionError("environment factory crossed the designer-provenance gate")

    guarded = EvaluationDependencies(
        environment_factory=environment_factory,
        actor_loader=actor_loader,
        fingerprint_factory=_fingerprint,
        execution_loader=load_test_execution,
    )
    with pytest.raises(CycleEvaluationError):
        evaluator.evaluate_cycle(
            experiment=experiment,
            cycle=1,
            oracle_paths=[candidate],
            repository_root=tmp_path,
            dependencies=guarded,
        )
    assert calls == {"actor": 0, "environment": 0}


def test_prior_arm_carry_forward_refuses_a_different_metric_core(tmp_path: Path) -> None:
    dependencies = _dependencies()
    experiment, report_path = _run_cycle_zero(tmp_path, dependencies)
    report = json.loads(report_path.read_bytes())
    identity = report["identities"]["metric_core"]
    identity["source_sha256"]["harness/executor.py"] = "0" * 64
    core = {
        "identity_id": identity["identity_id"],
        "source_sha256": identity["source_sha256"],
    }
    identity["sha256"] = hashlib.sha256(canonical_json_bytes(core)).hexdigest()
    report_path.write_bytes(canonical_json_bytes(report))
    with pytest.raises(CycleCliError, match="metric-core identity differs"):
        prepare_cycle(
            experiment=experiment,
            cycle=1,
            repository_root=tmp_path,
            dependencies=dependencies,
        )
