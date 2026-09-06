from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.phase_b.protected_metrics import (
    CONTROL_PERIOD_SECONDS,
    protected_step_record,
)
from oracle_composition.reward_study.t2_evaluator import (
    T2_HORIZON_STEPS,
    build_t2_trace,
    evaluate_t2_trace,
    t2_step_record,
)

ROOT = Path(__file__).resolve().parents[2]


def _reference_row() -> np.ndarray:
    row = np.zeros(45, dtype="<f8")
    row[0] = 1.4
    row[1] = 1.0
    row[5] = 3.0
    return row


def _state(*, x_position: float, fallen: bool) -> SimpleNamespace:
    return SimpleNamespace(
        root_position_world_m=np.asarray([x_position, 0.0, 0.8 if fallen else 1.4], dtype="<f8"),
        root_height_m=0.8 if fallen else 1.4,
        root_orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype="<f8"),
        root_linear_velocity_world_m_s=np.asarray([3.0, 0.0, 0.0], dtype="<f8"),
        root_angular_velocity_body_rad_s=np.zeros(3, dtype="<f8"),
        joint_positions_rad=np.zeros(17, dtype="<f8"),
        joint_velocities_rad_s=np.zeros(17, dtype="<f8"),
    )


def _steps(count: int, *, fall_step: int | None = None) -> list[dict[str, object]]:
    result = []
    reference = _reference_row()
    body_mass = np.asarray([1.0, 2.0], dtype="<f8")
    for step in range(count):
        x_before = step * 3.0 * CONTROL_PERIOD_SECONDS
        x_after = x_before + 3.0 * CONTROL_PERIOD_SECONDS
        positions_before = np.asarray(
            [[x_before, 0.0, 1.4], [x_before, 0.0, 1.4]],
            dtype="<f8",
        )
        positions_after = np.asarray(
            [[x_after, 0.0, 1.4], [x_after, 0.0, 1.4]],
            dtype="<f8",
        )
        fallen = fall_step == step + 1
        protected = protected_step_record(
            action=np.zeros(17, dtype="<f4"),
            body_mass=body_mass,
            body_xipos_after=positions_after,
            body_xipos_before=positions_before,
            fallen=fallen,
            forbidden_contacts=(),
            reference_behavior="expert",
            reference_index=step + 1,
            reference_row=reference,
            root_x_before_m=x_before,
            state=_state(x_position=x_after, fallen=fallen),
            step=step,
            terminated=False,
            truncated=step + 1 == T2_HORIZON_STEPS,
        )
        result.append(t2_step_record(protected_step=protected, stock_reward=2.0))
    return result


def _trace(count: int = T2_HORIZON_STEPS, *, fall_step: int | None = None) -> dict[str, object]:
    references = np.repeat(_reference_row()[None, :], 1_001, axis=0)
    return build_t2_trace(
        checkpoint_sha256="a" * 64,
        evaluation_seed=97001,
        policy_seed=121001,
        expert_reference_rows=references,
        steps=_steps(count, fall_step=fall_step),
    )


def test_t2_evaluator_recomputes_direct_com_and_whole_episode_metrics() -> None:
    references = np.repeat(_reference_row()[None, :], 1_001, axis=0)
    metrics = evaluate_t2_trace(_trace(), expert_reference_rows=references)
    assert metrics.observed_steps == 1_000
    assert metrics.com_forward_speed_m_s == pytest.approx((3.0,) * 1_000)
    assert metrics.mean_absolute_per_step_error_m_s == pytest.approx(0.0, abs=1e-12)
    assert metrics.fraction_steps_in_target_band == 1.0
    assert metrics.protected_task_return == pytest.approx(1_000.0)
    assert metrics.descriptive_stock_return == pytest.approx(2_000.0)
    assert metrics.first_fall_step is None
    assert metrics.fall_step_count == 0
    assert metrics.forbidden_contact_count == 0
    assert metrics.six_tracking_rmse == pytest.approx(
        {name: 0.0 for name in metrics.six_tracking_rmse}
    )
    assert metrics.safety_passed is True
    assert metrics.tracking_passed is True


def test_t2_evaluator_reports_fall_only_first_fall_without_censoring() -> None:
    references = np.repeat(_reference_row()[None, :], 1_001, axis=0)
    metrics = evaluate_t2_trace(_trace(fall_step=37), expert_reference_rows=references)
    assert metrics.complete_trace is True
    assert metrics.observed_steps == 1_000
    assert metrics.first_fall_step == 37
    assert metrics.fall_step_count == 1
    assert metrics.safety_passed is False
    assert "fall" in metrics.failure_reasons
    assert metrics.mean_absolute_per_step_error_m_s == pytest.approx(0.0, abs=1e-12)


def test_incomplete_t2_trace_fails_safety_and_withholds_whole_episode_endpoints() -> None:
    references = np.repeat(_reference_row()[None, :], 1_001, axis=0)
    metrics = evaluate_t2_trace(_trace(999), expert_reference_rows=references)
    assert metrics.complete_trace is False
    assert metrics.safety_passed is False
    assert metrics.mean_absolute_per_step_error_m_s is None
    assert metrics.fraction_steps_in_target_band is None
    assert metrics.protected_task_return is None
    assert metrics.six_tracking_rmse is None
    assert metrics.descriptive_stock_return is None
    assert metrics.failure_reasons == ("incomplete_trace",)


def test_t2_trace_refuses_reference_row_drift() -> None:
    trace = _trace(3)
    trace["steps"][1]["protected_step"]["reference"]["index"] = 9
    references = np.repeat(_reference_row()[None, :], 1_001, axis=0)
    with pytest.raises(ExperimentContractError, match="expert rows"):
        evaluate_t2_trace(trace, expert_reference_rows=references)


def test_t2_evaluator_refuses_self_consistent_but_nonauthoritative_reference_rows() -> None:
    trace = _trace(3)
    references = np.repeat(_reference_row()[None, :], 1_001, axis=0)
    references[2, 5] = 2.5
    with pytest.raises(ExperimentContractError, match="authoritative expert row"):
        evaluate_t2_trace(trace, expert_reference_rows=references)


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
