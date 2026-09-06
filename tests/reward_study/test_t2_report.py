from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.phase_b.contracts import TargetSpeedRewardSpec
from oracle_composition.phase_b.protected_metrics import ERROR_NAMES, ERROR_SCALES
from oracle_composition.reward_study.t2_evaluator import (
    T2EpisodeMetrics,
    _evaluate_t2_trace_against_reference,
    build_t2_trace,
    load_t2_verified_reference,
    summarize_t2_speeds,
)
from oracle_composition.reward_study.t2_report import (
    T2_POLICY_SEEDS,
    arm_summary,
    build_t2_study_report,
    build_t2_trace_index,
    checkpoint_summary,
    paired_checkpoint_effect,
    publish_t2_study_report,
    validate_t2_study_report,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/004_t2_reward_study"


def _episode(
    policy_seed: int,
    evaluation_seed: int,
    *,
    speed: float,
    tracking_passed: bool = True,
) -> T2EpisodeMetrics:
    values = (speed,) * 1_000
    summaries = summarize_t2_speeds(values)
    errors = {name: (0.0 if tracking_passed else ERROR_SCALES[name] + 0.01) for name in ERROR_NAMES}
    return T2EpisodeMetrics(
        action_bounds_ok=True,
        checkpoint_sha256=hashlib.sha256(f"checkpoint:{policy_seed}".encode()).hexdigest(),
        com_forward_speed_m_s=values,
        complete_trace=True,
        evaluation_seed=evaluation_seed,
        failure_reasons=(),
        first_fall_step=None,
        fraction_steps_in_target_band=summaries["fraction_steps_in_target_band"],
        fall_step_count=0,
        forbidden_contact_count=0,
        longest_out_of_band_run_steps=summaries["longest_out_of_band_run_steps"],
        mean_absolute_per_step_error_m_s=summaries["mean_absolute_per_step_error_m_s"],
        observed_steps=1_000,
        policy_seed=policy_seed,
        protected_task_return=summaries["protected_task_return"],
        reference_lineage_sha256=hashlib.sha256(
            f"reference:{evaluation_seed}".encode()
        ).hexdigest(),
        six_tracking_rmse=errors,
        speed_quantiles_m_s=summaries["speed_quantiles_m_s"],
        trace_sha256=hashlib.sha256(
            f"trace:{policy_seed}:{evaluation_seed}:{speed}".encode()
        ).hexdigest(),
    )


def _arms() -> tuple[list[T2EpisodeMetrics], list[T2EpisodeMetrics]]:
    baseline = []
    candidate = []
    for index, policy_seed in enumerate(T2_POLICY_SEEDS):
        for evaluation_seed in range(97001, 97021):
            baseline.append(_episode(policy_seed, evaluation_seed, speed=3.6 + index * 0.01))
            candidate.append(_episode(policy_seed, evaluation_seed, speed=3.1 + index * 0.01))
    return baseline, candidate


def _fake_binding(name: str) -> dict[str, object]:
    return {
        "byte_count": 1,
        "path": f"{name}.json",
        "root": "run",
        "sha256": hashlib.sha256(name.encode()).hexdigest(),
    }


def _fake_inputs() -> dict[str, object]:
    names = (
        "baseline_reward",
        "candidate_reward",
        "evaluator_design",
        "execution_manifest",
        "integrated_pairing_receipt",
        "oracle",
        "study_manifest",
        "trace_index",
        "training_design",
    )
    return {
        "artifacts": {name: _fake_binding(name) for name in names},
        "study_pairing_sha256": "f" * 64,
    }


def _binding(path: Path, *, root: str, relative: str) -> dict[str, object]:
    encoded = path.read_bytes()
    return {
        "byte_count": len(encoded),
        "path": relative,
        "root": root,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


@pytest.fixture(scope="module")
def replay_fixture(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, object]:
    run_root = tmp_path_factory.mktemp("t2-report-run")
    trace_root = run_root / "traces"
    trace_root.mkdir()
    candidate_path = run_root / "candidate_reward.json"
    candidate_path.write_bytes(TargetSpeedRewardSpec(alpha=1.0, beta=0.0).canonical_bytes)

    references = {
        seed: load_t2_verified_reference(repository_root=ROOT, evaluation_seed=seed)
        for seed in range(97001, 97021)
    }
    templates = {
        seed: build_t2_trace(
            checkpoint_sha256="a" * 64,
            evaluation_seed=seed,
            policy_seed=121001,
            repository_root=ROOT,
            steps=[],
        )
        for seed in range(97001, 97021)
    }
    entries = []
    episodes = {"baseline": [], "candidate": []}
    for arm in ("baseline", "candidate"):
        for policy_seed in T2_POLICY_SEEDS:
            checkpoint = hashlib.sha256(f"{arm}:checkpoint:{policy_seed}".encode()).hexdigest()
            for evaluation_seed in range(97001, 97021):
                trace = copy.deepcopy(templates[evaluation_seed])
                trace["policy_seed"] = policy_seed
                trace["checkpoint_sha256"] = checkpoint
                encoded = canonical_json_bytes(trace)
                relative = f"traces/{arm}-{policy_seed}-{evaluation_seed}.json"
                path = run_root / relative
                path.write_bytes(encoded)
                entries.append(
                    {
                        "arm_label": arm,
                        "byte_count": len(encoded),
                        "evaluation_seed": evaluation_seed,
                        "path": relative,
                        "policy_seed": policy_seed,
                        "sha256": hashlib.sha256(encoded).hexdigest(),
                    }
                )
                episodes[arm].append(
                    _evaluate_t2_trace_against_reference(
                        trace, reference=references[evaluation_seed]
                    )
                )
    index = build_t2_trace_index(entries)
    index_path = run_root / "trace_index.json"
    index_path.write_bytes(canonical_json_bytes(index))
    study_path = EXPERIMENT / "t2_reward_study_expert_hold_v1.json"
    study = json.loads(study_path.read_bytes())
    artifacts = {
        "baseline_reward": _binding(
            ROOT / "experiments/003_composition_speed_profile/phase_b/tracking_only_v1.json",
            root="repository",
            relative=("experiments/003_composition_speed_profile/phase_b/tracking_only_v1.json"),
        ),
        "candidate_reward": _binding(candidate_path, root="run", relative="candidate_reward.json"),
        "evaluator_design": _binding(
            EXPERIMENT / "evaluator_design_t2_v1.json",
            root="repository",
            relative="experiments/004_t2_reward_study/evaluator_design_t2_v1.json",
        ),
        "execution_manifest": _binding(
            EXPERIMENT / "execution_manifest_t2_v1.json",
            root="repository",
            relative=("experiments/004_t2_reward_study/execution_manifest_t2_v1.json"),
        ),
        "integrated_pairing_receipt": _binding(
            ROOT / "artifacts/experiments_004/t2_pairing_adapter_receipt_v1.json",
            root="repository",
            relative=("artifacts/experiments_004/t2_pairing_adapter_receipt_v1.json"),
        ),
        "oracle": _binding(
            EXPERIMENT / "oracle_expert_hold_v1.json",
            root="repository",
            relative="experiments/004_t2_reward_study/oracle_expert_hold_v1.json",
        ),
        "study_manifest": _binding(
            study_path,
            root="repository",
            relative=("experiments/004_t2_reward_study/t2_reward_study_expert_hold_v1.json"),
        ),
        "trace_index": _binding(index_path, root="run", relative="trace_index.json"),
        "training_design": _binding(
            EXPERIMENT / "training_design_t2_v1.json",
            root="repository",
            relative="experiments/004_t2_reward_study/training_design_t2_v1.json",
        ),
    }
    inputs = {
        "artifacts": artifacts,
        "study_pairing_sha256": study["study_pairing_sha256"],
    }
    report = build_t2_study_report(
        inputs=inputs,
        baseline_episodes=episodes["baseline"],
        candidate_episodes=episodes["candidate"],
    )
    return {
        "candidate": episodes["candidate"],
        "index": index,
        "inputs": inputs,
        "baseline": episodes["baseline"],
        "report": report,
        "run_root": run_root,
    }


def test_paired_checkpoint_effect_has_nonzero_interval_and_exact_sign_result() -> None:
    differences = (0.3, 0.4, 0.5, 0.6, 0.7)
    baseline = {seed: 1.0 for seed in T2_POLICY_SEEDS}
    candidate = {
        seed: 1.0 - difference
        for seed, difference in zip(T2_POLICY_SEEDS, differences, strict=True)
    }
    result = paired_checkpoint_effect(baseline, candidate)
    assert result["policy_count"] == 5
    assert result["mean_difference_m_s"] == pytest.approx(0.5)
    assert result["standard_error_m_s"] > 0.0
    low, high = result["paired_95_percent_t_interval_m_s"]
    assert low < 0.5 < high
    assert result["exact_one_sided_sign_result"]["p_value"] == pytest.approx(1 / 32)
    assert result["task_improvement_rule"]["passed"] is True


def test_paired_checkpoint_effect_refuses_pooled_n_100_interval() -> None:
    pooled = {index: 1.0 for index in range(100)}
    with pytest.raises(ValueError, match="pooled n = 100 is forbidden"):
        paired_checkpoint_effect(pooled, pooled)


def test_tracking_gate_boundaries_are_15_vs_16_and_3_vs_4() -> None:
    policy_seed = T2_POLICY_SEEDS[0]
    fifteen = [
        _episode(
            policy_seed,
            seed,
            speed=3.0,
            tracking_passed=index < 15,
        )
        for index, seed in enumerate(range(97001, 97021))
    ]
    sixteen = [
        _episode(
            policy_seed,
            seed,
            speed=3.0,
            tracking_passed=index < 16,
        )
        for index, seed in enumerate(range(97001, 97021))
    ]
    assert checkpoint_summary(fifteen)["tracking_passed"] is False
    assert checkpoint_summary(sixteen)["tracking_passed"] is True

    def arm(checkpoint_pass_count: int) -> list[T2EpisodeMetrics]:
        result = []
        for index, seed in enumerate(T2_POLICY_SEEDS):
            result.extend(
                _episode(
                    seed,
                    evaluation_seed,
                    speed=3.0,
                    tracking_passed=index < checkpoint_pass_count,
                )
                for evaluation_seed in range(97001, 97021)
            )
        return result

    three = arm_summary(arm_label="baseline", reward_sha256="a" * 64, episodes=arm(3))
    four = arm_summary(arm_label="baseline", reward_sha256="a" * 64, episodes=arm(4))
    assert three["tracking_checkpoint_pass_count"] == 3
    assert three["tracking_passed"] is False
    assert four["tracking_checkpoint_pass_count"] == 4
    assert four["tracking_passed"] is True


def test_report_structure_contains_no_reward_telemetry() -> None:
    baseline, candidate = _arms()
    report = build_t2_study_report(
        inputs=_fake_inputs(),
        baseline_episodes=baseline,
        candidate_episodes=candidate,
    )
    encoded = canonical_json_bytes(report)
    assert b"stock_reward" not in encoded
    assert b"descriptive_stock_return" not in encoded
    assert b"reward_outputs" not in encoded
    assert report["evaluation"]["reward_telemetry_in_scientific_receipt"] is False
    assert report["summary"]["weighted_aggregate"] is None


def test_report_acceptance_resolves_inputs_and_replays_exact_index(
    replay_fixture: dict[str, object],
) -> None:
    assert (
        validate_t2_study_report(
            replay_fixture["report"],
            repository_root=ROOT,
            run_root=replay_fixture["run_root"],
        )
        == replay_fixture["report"]
    )


def _report_with_index(
    replay_fixture: dict[str, object],
    *,
    index: dict[str, object],
    name: str,
) -> dict[str, object]:
    run_root = replay_fixture["run_root"]
    path = run_root / name
    path.write_bytes(canonical_json_bytes(index))
    inputs = copy.deepcopy(replay_fixture["inputs"])
    inputs["artifacts"]["trace_index"] = _binding(path, root="run", relative=name)
    return build_t2_study_report(
        inputs=inputs,
        baseline_episodes=replay_fixture["baseline"],
        candidate_episodes=replay_fixture["candidate"],
    )


def test_report_refuses_forged_trace_index(
    replay_fixture: dict[str, object],
) -> None:
    index = copy.deepcopy(replay_fixture["index"])
    index["traces"][0]["evaluation_seed"] = 97002
    report = _report_with_index(replay_fixture, index=index, name="forged-trace-index.json")
    with pytest.raises(ValueError, match="coverage differs"):
        validate_t2_study_report(
            report,
            repository_root=ROOT,
            run_root=replay_fixture["run_root"],
        )


def test_report_refuses_missing_raw_trace(
    replay_fixture: dict[str, object],
) -> None:
    index = copy.deepcopy(replay_fixture["index"])
    index["traces"][0]["path"] = "traces/missing.json"
    report = _report_with_index(replay_fixture, index=index, name="missing-trace-index.json")
    with pytest.raises(ValueError, match="raw trace is unavailable"):
        validate_t2_study_report(
            report,
            repository_root=ROOT,
            run_root=replay_fixture["run_root"],
        )


def test_report_refuses_fabricated_episode_metrics(
    replay_fixture: dict[str, object],
) -> None:
    baseline = list(replay_fixture["baseline"])
    baseline[0] = replace(baseline[0], trace_sha256="0" * 64)
    report = build_t2_study_report(
        inputs=replay_fixture["inputs"],
        baseline_episodes=baseline,
        candidate_episodes=replay_fixture["candidate"],
    )
    with pytest.raises(ValueError, match="differ from raw-trace replay"):
        validate_t2_study_report(
            report,
            repository_root=ROOT,
            run_root=replay_fixture["run_root"],
        )


def test_reward_telemetry_missing_nonfinite_or_malformed_cannot_change_science(
    replay_fixture: dict[str, object],
    tmp_path: Path,
) -> None:
    artifacts = [
        publish_t2_study_report(
            output_directory=tmp_path / name,
            report=replay_fixture["report"],
            telemetry={"wall_time_seconds": index + 1.0},
            reward_telemetry=reward_telemetry,
            repository_root=ROOT,
            run_root=replay_fixture["run_root"],
        )
        for index, (name, reward_telemetry) in enumerate(
            (
                ("missing", None),
                ("nonfinite", {"stock_reward": float("nan")}),
                ("malformed", ["not", "a", "mapping"]),
            )
        )
    ]
    assert len({item.scientific_receipt.sha256 for item in artifacts}) == 1
    assert len({item.scientific_receipt.path.read_bytes() for item in artifacts}) == 1
    diagnostics = [json.loads(item.reward_diagnostics.path.read_bytes()) for item in artifacts]
    assert [item["status"] for item in diagnostics] == [
        "missing",
        "malformed",
        "malformed",
    ]
    assert all(
        item["scientific_receipt_sha256"] == artifacts[0].scientific_receipt.sha256
        for item in diagnostics
    )
