from __future__ import annotations

import ast
import copy
import hashlib
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oracle_composition.contracts.reference_identity_v2 import array_sha256, canonical_json_bytes
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.phase_b.protected_metrics import (
    CONTROL_PERIOD_SECONDS,
    protected_step_record,
)
from oracle_composition.phase_b.reference_runtime import tracking_state_from_reference_row
from oracle_composition.reward_study import t2_evaluator
from oracle_composition.reward_study.t2_evaluator import (
    T2_HORIZON_STEPS,
    T2VerifiedReference,
    build_t2_trace,
    evaluate_t2_trace,
    load_t2_verified_reference,
    summarize_t2_speeds,
    t2_step_record,
)
from oracle_composition.reward_study.t2_report import checkpoint_summary

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def synthetic_reference(monkeypatch: pytest.MonkeyPatch) -> T2VerifiedReference:
    """Analytical rows for metric tests, never retained-corpus admission."""
    time = np.arange(T2_HORIZON_STEPS + 1) * CONTROL_PERIOD_SECONDS
    rows = np.zeros((T2_HORIZON_STEPS + 1, 45), dtype="<f8", order="C")
    rows[:, 0] = 1.4
    rows[:, 1] = 1.0
    rows[:, 5] = 3.0
    phases = time[:, None] + np.arange(17)[None, :] / 17
    rows[:, 11:28] = 0.1 * np.sin(phases)
    rows[:, 28:45] = 0.1 * np.cos(phases)
    rows.setflags(write=False)
    lineage = {
        "provenance": "synthetic_unit_test_only_not_corpus_admission",
        "clip_id": "synthetic-analytical-metric-reference",
        "payload_sha256": array_sha256(rows),
        "reference_rows": {
            "dtype": "<f8",
            "sha256": array_sha256(rows),
            "shape": [T2_HORIZON_STEPS + 1, 45],
        },
    }
    reference = T2VerifiedReference(
        rows=rows,
        lineage=lineage,
        lineage_sha256=hashlib.sha256(canonical_json_bytes(lineage)).hexdigest(),
    )

    def load_synthetic_reference(
        *, repository_root: Path, evaluation_seed: int
    ) -> T2VerifiedReference:
        assert repository_root == ROOT
        assert evaluation_seed == 97001
        return reference

    monkeypatch.setattr(t2_evaluator, "load_t2_verified_reference", load_synthetic_reference)
    return reference


def test_historical_t2_reference_chain_resolves_exact_expert_rows() -> None:
    reference = load_t2_verified_reference(repository_root=ROOT, evaluation_seed=97001)
    assert reference.lineage["block"] == 120101
    assert reference.lineage["clip_id"] == "corpus-120101-expert"
    assert reference.rows.shape == (T2_HORIZON_STEPS + 1, 45)
    assert reference.rows.dtype.str == "<f8"
    assert reference.rows.flags.c_contiguous
    assert not reference.rows.flags.writeable
    assert reference.lineage["reference_rows"]["sha256"] == array_sha256(reference.rows)
    assert (
        reference.lineage_sha256
        == hashlib.sha256(canonical_json_bytes(reference.lineage)).hexdigest()
    )


def _copy_public_library(repository_root: Path) -> None:
    library_path = repository_root / t2_evaluator.T2_LIBRARY_PATH
    library_path.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / t2_evaluator.T2_LIBRARY_PATH, library_path)


def test_t2_reference_loader_refuses_missing_corpus(tmp_path: Path) -> None:
    _copy_public_library(tmp_path)
    with pytest.raises(ExperimentContractError, match=r"^T2 corpus manifest is unavailable$"):
        load_t2_verified_reference(repository_root=tmp_path, evaluation_seed=97001)


def test_t2_reference_loader_refuses_wrong_canonical_corpus_digest(tmp_path: Path) -> None:
    _copy_public_library(tmp_path)
    corpus_path = tmp_path / t2_evaluator.T2_CORPUS_PATH
    corpus_path.parent.mkdir(parents=True)
    corpus_path.write_bytes(canonical_json_bytes({}))
    with pytest.raises(ExperimentContractError, match=r"^T2 library corpus digest differs$"):
        load_t2_verified_reference(repository_root=tmp_path, evaluation_seed=97001)


def _state(row: np.ndarray, *, fallen: bool) -> SimpleNamespace:
    state = tracking_state_from_reference_row(row)
    height = 0.8 if fallen else state.root_height_m
    position = np.array(state.root_position_world_m, dtype="<f8", copy=True)
    position[2] = height
    return SimpleNamespace(
        root_position_world_m=position,
        root_height_m=height,
        root_orientation_wxyz=state.root_orientation_wxyz,
        root_linear_velocity_world_m_s=state.root_linear_velocity_world_m_s,
        root_angular_velocity_body_rad_s=state.root_angular_velocity_body_rad_s,
        joint_positions_rad=state.joint_positions_rad,
        joint_velocities_rad_s=state.joint_velocities_rad_s,
    )


def _steps(
    reference: T2VerifiedReference,
    count: int,
    *,
    action_value: float = 0.0,
    fall_step: int | None = None,
    speeds: tuple[float, ...] | None = None,
) -> list[dict[str, object]]:
    result = []
    body_mass = np.asarray([1.0, 2.0], dtype="<f8")
    for step in range(count):
        speed = 3.0 if speeds is None else speeds[step]
        x_before = step * 0.01
        x_after = x_before + speed * CONTROL_PERIOD_SECONDS
        positions_before = np.asarray([[x_before, 0.0, 1.4], [x_before, 0.0, 1.4]], dtype="<f8")
        positions_after = np.asarray([[x_after, 0.0, 1.4], [x_after, 0.0, 1.4]], dtype="<f8")
        fallen = fall_step == step + 1
        protected = protected_step_record(
            action=np.full(17, action_value, dtype="<f4"),
            body_mass=body_mass,
            body_xipos_after=positions_after,
            body_xipos_before=positions_before,
            fallen=fallen,
            forbidden_contacts=(),
            reference_behavior="expert",
            reference_index=step + 1,
            reference_row=reference.rows[step + 1],
            root_x_before_m=x_before,
            state=_state(reference.rows[step + 1], fallen=fallen),
            step=step,
            terminated=False,
            truncated=step + 1 == T2_HORIZON_STEPS,
        )
        result.append(t2_step_record(protected_step=protected))
    return result


def _trace(
    reference: T2VerifiedReference,
    count: int = T2_HORIZON_STEPS,
    *,
    action_value: float = 0.0,
    fall_step: int | None = None,
) -> dict[str, object]:
    return build_t2_trace(
        checkpoint_sha256="a" * 64,
        evaluation_seed=97001,
        policy_seed=121001,
        repository_root=ROOT,
        steps=_steps(
            reference,
            count,
            action_value=action_value,
            fall_step=fall_step,
        ),
    )


def test_t2_evaluator_recomputes_direct_com_and_whole_episode_metrics(
    synthetic_reference: T2VerifiedReference,
) -> None:
    metrics = evaluate_t2_trace(
        _trace(synthetic_reference),
        repository_root=ROOT,
    )
    assert metrics.observed_steps == 1_000
    assert metrics.com_forward_speed_m_s == pytest.approx((3.0,) * 1_000)
    assert metrics.mean_absolute_per_step_error_m_s == pytest.approx(0.0, abs=1e-12)
    assert metrics.fraction_steps_in_target_band == 1.0
    assert metrics.protected_task_return == pytest.approx(1_000.0)
    assert metrics.first_fall_step is None
    assert metrics.fall_step_count == 0
    assert metrics.forbidden_contact_count == 0
    assert metrics.six_tracking_rmse == pytest.approx(
        {name: 0.0 for name in metrics.six_tracking_rmse}, abs=2e-8
    )
    assert metrics.reference_lineage_sha256 == synthetic_reference.lineage_sha256
    assert metrics.safety_passed is True
    assert metrics.tracking_passed is True


def test_t2_evaluator_reports_fall_only_first_fall_without_censoring(
    synthetic_reference: T2VerifiedReference,
) -> None:
    metrics = evaluate_t2_trace(
        _trace(synthetic_reference, fall_step=37),
        repository_root=ROOT,
    )
    assert metrics.complete_trace is True
    assert metrics.observed_steps == 1_000
    assert metrics.first_fall_step == 37
    assert metrics.fall_step_count == 1
    assert metrics.safety_passed is False
    assert "fall" in metrics.failure_reasons


def test_incomplete_t2_trace_fails_safety_and_withholds_whole_episode_endpoints(
    synthetic_reference: T2VerifiedReference,
) -> None:
    metrics = evaluate_t2_trace(
        _trace(synthetic_reference, 3),
        repository_root=ROOT,
    )
    assert metrics.complete_trace is False
    assert metrics.safety_passed is False
    assert metrics.mean_absolute_per_step_error_m_s is None
    assert metrics.fraction_steps_in_target_band is None
    assert metrics.protected_task_return is None
    assert metrics.six_tracking_rmse is None
    assert metrics.failure_reasons == ("incomplete_trace",)


def test_t2_trace_binds_synthetic_reference_and_refuses_row_or_lineage_forgery(
    synthetic_reference: T2VerifiedReference,
) -> None:
    trace = _trace(synthetic_reference, 3)
    assert b"reward" not in canonical_json_bytes(trace)
    assert trace["reference_lineage"]["provenance"] == (
        "synthetic_unit_test_only_not_corpus_admission"
    )
    assert trace["reference_lineage"]["clip_id"] == "synthetic-analytical-metric-reference"
    assert trace["reset_reference"]["values"] == synthetic_reference.rows[0].tolist()
    assert trace["steps"][2]["reference"]["values"] == synthetic_reference.rows[3].tolist()

    forged_row = copy.deepcopy(trace)
    forged_row["steps"][1]["reference"]["index"] = 9
    forged_row["steps_sha256"] = hashlib.sha256(
        canonical_json_bytes(forged_row["steps"])
    ).hexdigest()
    with pytest.raises(ExperimentContractError, match="verified-chain expert row"):
        evaluate_t2_trace(
            forged_row,
            repository_root=ROOT,
        )

    forged_lineage = copy.deepcopy(trace)
    forged_lineage["reference_lineage"]["payload_sha256"] = "0" * 64
    forged_lineage["reference_lineage_sha256"] = hashlib.sha256(
        canonical_json_bytes(forged_lineage["reference_lineage"])
    ).hexdigest()
    with pytest.raises(ExperimentContractError, match="verified reference lineage"):
        evaluate_t2_trace(
            forged_lineage,
            repository_root=ROOT,
        )


def test_action_boundaries_reach_safety_report_and_out_of_bounds_fails(
    synthetic_reference: T2VerifiedReference,
) -> None:
    reports = {}
    for value in (-0.4, 0.4, 0.4001):
        metrics = evaluate_t2_trace(
            _trace(synthetic_reference, action_value=value),
            repository_root=ROOT,
        )
        episodes = [
            replace(metrics, evaluation_seed=evaluation_seed)
            for evaluation_seed in range(97001, 97021)
        ]
        reports[value] = checkpoint_summary(episodes)
    assert reports[-0.4]["safety_passed"] is True
    assert reports[0.4]["safety_passed"] is True
    assert reports[0.4001]["safety_passed"] is False
    assert reports[0.4001]["invalid_action_episode_count"] == 20


def test_varying_speed_profile_exercises_per_step_error_denominator_clipping_and_band() -> None:
    values = (2.75,) * 250 + (3.25,) * 250 + (-3.0,) * 250 + (9.0,) * 250
    result = summarize_t2_speeds(values)
    assert sum(values) / len(values) == pytest.approx(3.0)
    assert result["mean_absolute_per_step_error_m_s"] == pytest.approx(3.125)
    assert result["fraction_steps_in_target_band"] == pytest.approx(0.5)
    assert result["protected_task_return"] == pytest.approx(458.3333333333333)
    assert result["longest_out_of_band_run_steps"] == 500


def test_t2_evaluator_source_does_not_import_reward_helpers() -> None:
    source_path = ROOT / "src/oracle_composition/reward_study/t2_evaluator.py"
    tree = ast.parse(source_path.read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.append(node.module)
    assert "oracle_composition.phase_b.reward" not in imported
    assert not any(name.startswith("oracle_composition.rewards") for name in imported)
