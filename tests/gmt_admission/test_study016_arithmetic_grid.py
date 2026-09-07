"""Cross-check the runtime law against the independent study scorer."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.adapters.gmt.course_task import TaskFrame
from oracle_composition.adapters.gmt.heading_feedback import (
    after_heading_feedback_trace,
    issue_held_poststep_target,
    plan_after_heading_feedback,
)


@pytest.fixture(scope="module")
def scorer():
    path = Path(__file__).resolve().parents[2] / "experiments/016_g1_after_feedback/score.py"
    spec = importlib.util.spec_from_file_location("study016_grid_score", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("lateral", [-100.0, -1.0, 0.0, 1.0, 100.0])
@pytest.mark.parametrize("heading", [-math.pi + 1e-7, -0.3, 0.0, 0.3, math.pi - 1e-7])
def test_exact_arithmetic_and_held_endpoint_on_wrap_clip_grid(scorer, lateral, heading):
    origin, yaw = np.asarray([2.0, -3.0]), 0.7
    qpos = np.zeros((267, 30), dtype=np.float64)
    qpos[:, :2] = origin
    qpos[:, 2] = 0.75
    qpos[:, 3], qpos[:, 6] = math.cos(yaw / 2), math.sin(yaw / 2)
    qpos[263:, :2] = origin + lateral * np.asarray([-math.sin(yaw), math.cos(yaw)])
    qpos[263:, 3] = math.cos((yaw + heading) / 2)
    qpos[263:, 6] = math.sin((yaw + heading) / 2)
    frame = TaskFrame.initialize(qpos[0, :2], qpos[0, 3:7])
    rows = [{"executed_mode": "before"} for _ in range(266)]
    current = np.zeros((266, 30), dtype="<f4")
    for index, native_yaw in zip(range(263, 266), (-0.36, 0.0, 0.36), strict=True):
        native = np.arange(600, dtype="<f4").reshape(20, 30) / np.float32(1000)
        native[:, 6] = np.linspace(-0.36, 0.36, 20, dtype="<f4")
        projection = frame.project(qpos[index, :2], qpos[index, 3:7])
        plan = plan_after_heading_feedback(
            native,
            lateral_m=projection.lateral_m,
            heading_error_signed_rad=projection.heading_error_rad,
        )
        # The native poststep endpoint may legitimately differ at a crop wrap.
        endpoint = native[0].copy()
        endpoint[6] = np.float32(native_yaw)
        endpoint[0] += np.float32(0.01)
        issued = issue_held_poststep_target(plan, endpoint)
        current[index] = issued
        rows[index] = {
            "executed_mode": "after",
            scorer.TRACE_KEY: after_heading_feedback_trace(
                control_step=index, plan=plan, source_current=endpoint, issued_current=issued
            ),
        }
    result = scorer.steering_measures(
        {"frames": rows, "trajectory": {"qpos": qpos, "current_reference": current}}
    )
    assert result["after_action_count"] == 3
    assert result["native_endpoint_window_first_row_mismatch_count"] == 3
    assert result["issued_endpoint_window_first_row_mismatch_count"] == 3
