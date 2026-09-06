from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.phase_b.protected_metrics import (
    evaluator_tracking_errors,
    protected_step_record,
    protected_trace_sha256,
    recompute_protected_episode,
)
from oracle_composition.phase_b.reference_runtime import tracking_state_from_reference_row
from oracle_composition.tracking.reward import compute_tracking_reward

ROOT = Path(__file__).resolve().parents[2]


def _row() -> np.ndarray:
    row = np.zeros(45, dtype="<f8")
    row[0] = 1.4
    row[1] = 1.0
    return row


@pytest.fixture(scope="module")
def sufficient_steps() -> list[dict[str, object]]:
    reference = _row()
    state = tracking_state_from_reference_row(reference)
    masses = np.asarray([1.0, 3.0], dtype="<f8")
    action = np.zeros(17, dtype="<f4")
    steps = []
    for step in range(1_000):
        before = np.asarray(
            [[step * 0.001, 0.0, 0.0], [step * 0.001, 0.0, 0.0]],
            dtype="<f8",
        )
        after = before.copy()
        after[:, 0] += 0.0015
        steps.append(
            protected_step_record(
                step=step,
                state=state,
                reference_behavior="expert",
                reference_index=step + 1,
                reference_row=reference,
                action=action,
                root_x_before_m=0.0,
                body_mass=masses,
                body_xipos_before=before,
                body_xipos_after=after,
                forbidden_contacts=(),
                fallen=False,
                terminated=False,
                truncated=step == 999,
            )
        )
    return steps


def test_evaluator_tracking_errors_are_reward_independent_under_corruption(
    monkeypatch: pytest.MonkeyPatch,
    sufficient_steps: list[dict[str, object]],
) -> None:
    from oracle_composition.phase_b import reward as phase_b_reward
    from oracle_composition.rewards import stock_humanoid
    from oracle_composition.tracking import reward as tracking_reward

    reference = _row()
    state = tracking_state_from_reference_row(reference)
    expected = compute_tracking_reward(state=state, reference_frame=reference).error_components()
    protected = evaluator_tracking_errors(
        {
            "joint_positions_rad": state.joint_positions_rad.tolist(),
            "joint_velocities_rad_s": state.joint_velocities_rad_s.tolist(),
            "root_angular_velocity_body_rad_s": state.root_angular_velocity_body_rad_s.tolist(),
            "root_height_m": state.root_height_m,
            "root_linear_velocity_world_m_s": state.root_linear_velocity_world_m_s.tolist(),
            "root_orientation_wxyz": state.root_orientation_wxyz.tolist(),
            "root_position_world_m": state.root_position_world_m.tolist(),
        },
        reference,
    )
    episode_before = recompute_protected_episode(
        steps=sufficient_steps,
        cell="hold_expert",
        switches=(),
        segment_targets_m_s=(0.1, 0.1, 0.1),
    )
    assert protected == {name: expected[name] for name in protected}

    monkeypatch.setattr(tracking_reward, "compute_tracking_reward", lambda **_kwargs: -999.0)
    monkeypatch.setattr(phase_b_reward, "compute_tracking_reward", lambda **_kwargs: -888.0)
    monkeypatch.setattr(
        stock_humanoid,
        "body_mass_weighted_com_x_velocity_m_s",
        lambda **_kwargs: 777.0,
    )
    corrupted_reward_telemetry = {"r_track": -1e9, "r_task": 1e9, "stock_reward": np.nan}
    assert corrupted_reward_telemetry["r_track"] != 0.0
    assert (
        evaluator_tracking_errors(
            {
                "joint_positions_rad": state.joint_positions_rad.tolist(),
                "joint_velocities_rad_s": state.joint_velocities_rad_s.tolist(),
                "root_angular_velocity_body_rad_s": state.root_angular_velocity_body_rad_s.tolist(),
                "root_height_m": state.root_height_m,
                "root_linear_velocity_world_m_s": state.root_linear_velocity_world_m_s.tolist(),
                "root_orientation_wxyz": state.root_orientation_wxyz.tolist(),
                "root_position_world_m": state.root_position_world_m.tolist(),
            },
            reference,
        )
        == protected
    )
    assert (
        recompute_protected_episode(
            steps=sufficient_steps,
            cell="hold_expert",
            switches=(),
            segment_targets_m_s=(0.1, 0.1, 0.1),
        )
        == episode_before
    )


def test_protected_metric_module_has_no_reward_helper_imports() -> None:
    path = ROOT / "src/oracle_composition/phase_b/protected_metrics.py"
    tree = ast.parse(path.read_text())
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert not any("reward" in name for name in imports)


def test_all_protected_metrics_recompute_from_canonical_sufficient_trace(
    sufficient_steps: list[dict[str, object]],
) -> None:
    first = recompute_protected_episode(
        steps=sufficient_steps,
        cell="hold_expert",
        switches=(),
        segment_targets_m_s=(0.1, 0.1, 0.1),
    )
    second = recompute_protected_episode(
        steps=sufficient_steps,
        cell="hold_expert",
        switches=(),
        segment_targets_m_s=(0.1, 0.1, 0.1),
    )
    assert first == second
    assert first["com_forward_speed_m_s"] == pytest.approx([0.1] * 1_000)
    assert set(first["six_tracking_errors"]) == {
        "joint_position_rmse_rad",
        "joint_velocity_rmse_rad_s",
        "root_angular_velocity_rmse_rad_s",
        "root_height_abs_error_m",
        "root_linear_velocity_rmse_m_s",
        "root_orientation_error_rad",
    }
    assert len(protected_trace_sha256(sufficient_steps)) == 64


@pytest.mark.parametrize(
    ("action_value", "expected"),
    [(-0.4, True), (0.4, True), (0.4001, False)],
)
def test_protected_action_bounds_are_independently_reachable(
    sufficient_steps: list[dict[str, object]],
    action_value: float,
    expected: bool,
) -> None:
    steps = copy.deepcopy(sufficient_steps)
    steps[0]["action"] = np.full(17, action_value, dtype="<f4").tolist()
    result = recompute_protected_episode(
        steps=steps,
        cell="hold_expert",
        switches=(),
        segment_targets_m_s=(0.1, 0.1, 0.1),
    )
    assert result["action_bounds_ok"] is expected
