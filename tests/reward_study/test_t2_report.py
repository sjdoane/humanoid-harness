from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.phase_b.contracts import TargetSpeedRewardSpec
from oracle_composition.phase_b.protected_metrics import (
    CONTROL_PERIOD_SECONDS,
    ERROR_NAMES,
    ERROR_SCALES,
    protected_step_record,
)
from oracle_composition.phase_b.reference_runtime import tracking_state_from_reference_row
from oracle_composition.reward_study.execution_manifest import (
    t2_execution_manifest_contract_value,
)
from oracle_composition.reward_study.study_manifest import (
    CANDIDATE_REWARD_ID,
    STUDY_FINAL_READY_STATUS,
    study_pairing_sha256_from_arm,
)
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
TEST_EXECUTION_COMMIT = "a" * 40


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


def _fake_binding(name: str, *, reward_id: str | None = None) -> dict[str, object]:
    value = {
        "byte_count": 1,
        "path": f"{name}.json",
        "root": "run",
        "sha256": hashlib.sha256(name.encode()).hexdigest(),
    }
    if reward_id is not None:
        value["reward_id"] = reward_id
    return value


def _fake_inputs() -> dict[str, object]:
    names = (
        "baseline_reward",
        "candidate_reward",
        "evaluator_design",
        "execution_manifest",
        "integrated_pairing_receipt",
        "library",
        "oracle",
        "pairing_adapter",
        "reference_corpus",
        "starting_checkpoint",
        "study_manifest",
        "trace_index",
        "training_design",
    )
    artifacts = {name: _fake_binding(name) for name in names}
    artifacts["baseline_reward"] = _fake_binding(
        "baseline_reward",
        reward_id="tracking_only/v1",
    )
    artifacts["candidate_reward"] = _fake_binding(
        "candidate_reward",
        reward_id=CANDIDATE_REWARD_ID,
    )
    return {
        "artifacts": artifacts,
        "study_pairing_sha256": "f" * 64,
    }


def _binding(
    path: Path,
    *,
    root: str,
    relative: str,
    reward_id: str | None = None,
) -> dict[str, object]:
    encoded = path.read_bytes()
    value = {
        "byte_count": len(encoded),
        "path": relative,
        "root": root,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
    if reward_id is not None:
        value["reward_id"] = reward_id
    return value


def _one_step(reference: object) -> dict[str, object]:
    row = reference.rows[1]
    state = tracking_state_from_reference_row(row)
    evaluator_state = SimpleNamespace(
        root_position_world_m=state.root_position_world_m,
        root_height_m=state.root_height_m,
        root_orientation_wxyz=state.root_orientation_wxyz,
        root_linear_velocity_world_m_s=state.root_linear_velocity_world_m_s,
        root_angular_velocity_body_rad_s=state.root_angular_velocity_body_rad_s,
        joint_positions_rad=state.joint_positions_rad,
        joint_velocities_rad_s=state.joint_velocities_rad_s,
    )
    masses = np.asarray([1.0, 2.0], dtype="<f8")
    before = np.asarray([[0.0, 0.0, 1.4], [0.0, 0.0, 1.4]], dtype="<f8")
    after = before.copy()
    after[:, 0] = 3.0 * CONTROL_PERIOD_SECONDS
    return protected_step_record(
        action=np.zeros(17, dtype="<f4"),
        body_mass=masses,
        body_xipos_after=after,
        body_xipos_before=before,
        fallen=False,
        forbidden_contacts=(),
        reference_behavior="expert",
        reference_index=1,
        reference_row=row,
        root_x_before_m=0.0,
        state=evaluator_state,
        step=0,
        terminated=False,
        truncated=False,
    )


@pytest.fixture(autouse=True)
def _verified_execution_head(monkeypatch: pytest.MonkeyPatch) -> None:
    import oracle_composition.reward_study.execution_manifest as execution_manifest

    monkeypatch.setattr(
        execution_manifest,
        "_observed_clean_execution_commit",
        lambda _root: TEST_EXECUTION_COMMIT,
    )


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
            steps=(_one_step(references[seed]),) if seed == 97001 else (),
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
    pending_study_path = EXPERIMENT / "t2_reward_study_expert_hold_v1.json"
    pending_study = json.loads(pending_study_path.read_bytes())
    execution_relative = pending_study["arms"][0]["execution_manifest"]["path"]
    final_execution = canonical_json_bytes(
        t2_execution_manifest_contract_value(
            ROOT,
            execution_commit=TEST_EXECUTION_COMMIT,
        )
    )
    final_execution_path = run_root / execution_relative
    final_execution_path.parent.mkdir(parents=True, exist_ok=True)
    final_execution_path.write_bytes(final_execution)

    final_study = copy.deepcopy(pending_study)
    final_execution_binding = {
        "byte_count": len(final_execution),
        "path": execution_relative,
        "sha256": hashlib.sha256(final_execution).hexdigest(),
    }
    for arm in final_study["arms"]:
        arm["execution_manifest"] = final_execution_binding
    final_study["arms"][1]["reward"] = {
        "path": "candidate_reward.json",
        "reward_id": CANDIDATE_REWARD_ID,
        "sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
    }
    final_study["status"] = STUDY_FINAL_READY_STATUS
    final_study["study_pairing_sha256"] = study_pairing_sha256_from_arm(final_study["arms"][0])
    study_relative = "experiments/004_t2_reward_study/t2_reward_study_expert_hold_v1.json"
    final_study_path = run_root / study_relative
    final_study_path.write_bytes(canonical_json_bytes(final_study))
    common = final_study["arms"][0]
    artifacts = {
        "baseline_reward": _binding(
            ROOT / "experiments/003_composition_speed_profile/phase_b/tracking_only_v1.json",
            root="repository",
            relative=("experiments/003_composition_speed_profile/phase_b/tracking_only_v1.json"),
            reward_id="tracking_only/v1",
        ),
        "candidate_reward": _binding(
            candidate_path,
            root="run",
            relative="candidate_reward.json",
            reward_id=CANDIDATE_REWARD_ID,
        ),
        "evaluator_design": _binding(
            ROOT / common["evaluator"]["path"],
            root="repository",
            relative=common["evaluator"]["path"],
        ),
        "execution_manifest": _binding(
            final_execution_path,
            root="run",
            relative=execution_relative,
        ),
        "integrated_pairing_receipt": _binding(
            EXPERIMENT / "pairing_receipt_v1.json",
            root="repository",
            relative="experiments/004_t2_reward_study/pairing_receipt_v1.json",
        ),
        "library": _binding(
            ROOT / common["library"]["path"],
            root="repository",
            relative=common["library"]["path"],
        ),
        "oracle": _binding(
            ROOT / common["oracle"]["path"],
            root="repository",
            relative=common["oracle"]["path"],
        ),
        "pairing_adapter": _binding(
            ROOT / common["pairing_adapter"]["path"],
            root="repository",
            relative=common["pairing_adapter"]["path"],
        ),
        "reference_corpus": _binding(
            ROOT / common["reference_corpus"]["path"],
            root="repository",
            relative=common["reference_corpus"]["path"],
        ),
        "starting_checkpoint": _binding(
            ROOT / common["starting_checkpoint"]["path"],
            root="repository",
            relative=common["starting_checkpoint"]["path"],
        ),
        "study_manifest": _binding(
            final_study_path,
            root="run",
            relative=study_relative,
        ),
        "trace_index": _binding(index_path, root="run", relative="trace_index.json"),
        "training_design": _binding(
            ROOT / common["training_design"]["path"],
            root="repository",
            relative=common["training_design"]["path"],
        ),
    }
    inputs = {
        "artifacts": artifacts,
        "study_pairing_sha256": final_study["study_pairing_sha256"],
    }
    report = build_t2_study_report(
        inputs=inputs,
        baseline_episodes=episodes["baseline"],
        candidate_episodes=episodes["candidate"],
    )
    pending_inputs = copy.deepcopy(inputs)
    pending_inputs["artifacts"]["study_manifest"] = _binding(
        pending_study_path,
        root="repository",
        relative=study_relative,
    )
    pending_inputs["study_pairing_sha256"] = pending_study["study_pairing_sha256"]
    pending_report = build_t2_study_report(
        inputs=pending_inputs,
        baseline_episodes=episodes["baseline"],
        candidate_episodes=episodes["candidate"],
    )
    return {
        "candidate": episodes["candidate"],
        "final_study": final_study,
        "index": index,
        "inputs": inputs,
        "baseline": episodes["baseline"],
        "pending_report": pending_report,
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


def test_pending_canonical_study_report_fails_with_explicit_state_reason(
    replay_fixture: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match=r"^study_not_final_ready$"):
        validate_t2_study_report(
            replay_fixture["pending_report"],
            repository_root=ROOT,
            run_root=replay_fixture["run_root"],
        )


@pytest.mark.parametrize(
    "artifact_name",
    ["candidate_reward", "evaluator_design", "execution_manifest", "oracle", "training_design"],
)
def test_final_ready_gate_refuses_each_rebound_manifest_cross_link(
    replay_fixture: dict[str, object],
    artifact_name: str,
) -> None:
    inputs = copy.deepcopy(replay_fixture["inputs"])
    original = inputs["artifacts"][artifact_name]
    source_root = ROOT if original["root"] == "repository" else replay_fixture["run_root"]
    source = source_root / original["path"]
    relative = f"rebound/{artifact_name}.json"
    rebound = replay_fixture["run_root"] / relative
    rebound.parent.mkdir(parents=True, exist_ok=True)
    rebound.write_bytes(source.read_bytes())
    inputs["artifacts"][artifact_name] = _binding(
        rebound,
        root="run",
        relative=relative,
        reward_id=original.get("reward_id"),
    )
    report = build_t2_study_report(
        inputs=inputs,
        baseline_episodes=replay_fixture["baseline"],
        candidate_episodes=replay_fixture["candidate"],
    )
    with pytest.raises(
        ValueError,
        match=rf"{artifact_name} binding differs from study manifest",
    ):
        validate_t2_study_report(
            report,
            repository_root=ROOT,
            run_root=replay_fixture["run_root"],
        )


def test_final_ready_gate_refuses_pairing_cross_link_and_old_adapter_receipt(
    replay_fixture: dict[str, object],
) -> None:
    inputs = copy.deepcopy(replay_fixture["inputs"])
    old_relative = "artifacts/experiments_004/t2_pairing_adapter_receipt_v1.json"
    inputs["artifacts"]["integrated_pairing_receipt"] = _binding(
        ROOT / old_relative,
        root="repository",
        relative=old_relative,
    )
    report = build_t2_study_report(
        inputs=inputs,
        baseline_episodes=replay_fixture["baseline"],
        candidate_episodes=replay_fixture["candidate"],
    )
    with pytest.raises(
        ValueError,
        match="integrated pairing receipt differs from study manifest",
    ):
        validate_t2_study_report(
            report,
            repository_root=ROOT,
            run_root=replay_fixture["run_root"],
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


def test_report_refuses_deep_metric_tamper_with_unchanged_trace_and_index(
    replay_fixture: dict[str, object],
) -> None:
    baseline = list(replay_fixture["baseline"])
    assert baseline[0].observed_steps == 1
    assert baseline[0].com_forward_speed_m_s == pytest.approx((3.0,))
    original_trace_sha256 = baseline[0].trace_sha256
    baseline[0] = replace(baseline[0], com_forward_speed_m_s=(2.0,))
    assert baseline[0].trace_sha256 == original_trace_sha256
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
