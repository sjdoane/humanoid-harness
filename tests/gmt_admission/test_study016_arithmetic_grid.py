"""Cross-check the runtime law against the independent study scorer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
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


def _independent_hash(value):
    descriptor = json.dumps(
        {"dtype": value.dtype.str, "shape": list(value.shape)},
        sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()
    return hashlib.sha256(descriptor + b"\0" + value.tobytes(order="C")).hexdigest()


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
        target = max(-0.3, min(0.3, -0.3 * projection.lateral_m))
        angle = target - projection.heading_error_rad
        correction = np.float32(math.atan2(math.sin(angle), math.cos(angle)))
        unclipped = np.add(native[:, 6], correction, dtype=np.float32)
        expected = native.copy()
        expected[:, 6] = np.minimum(
            np.maximum(unclipped, np.float32(-0.3)), np.float32(0.3)
        )
        np.testing.assert_array_equal(plan.native_window, native)
        np.testing.assert_array_equal(plan.issued_window, expected)
        np.testing.assert_array_equal(
            np.delete(plan.issued_window, 6, axis=1), np.delete(native, 6, axis=1)
        )
        # The native poststep endpoint may legitimately differ at a crop wrap.
        endpoint = native[0].copy()
        endpoint[6] = np.float32(native_yaw)
        endpoint[0] += np.float32(0.01)
        issued = issue_held_poststep_target(plan, endpoint)
        expected_endpoint = endpoint.copy()
        expected_endpoint[6] = min(
            np.float32(0.3), max(np.float32(-0.3), endpoint[6] + correction)
        )
        np.testing.assert_array_equal(issued, expected_endpoint)
        current[index] = issued
        rows[index] = {
            "executed_mode": "after",
            scorer.TRACE_KEY: after_heading_feedback_trace(
                control_step=index, plan=plan, source_current=endpoint, issued_current=issued
            ),
        }
        trace = rows[index][scorer.TRACE_KEY]
        window = trace["window"]
        exceeded = int(np.count_nonzero(np.abs(native[:, 6]) > np.float32(0.3)))
        saturated = int(np.count_nonzero(np.abs(unclipped) >= np.float32(0.3)))
        changed = int(np.count_nonzero(native != expected))
        assert window["native_float32_c_sha256"] == _independent_hash(native)
        assert window["issued_float32_c_sha256"] == _independent_hash(expected)
        assert window["native_rate_above_limit_count"] == exceeded
        assert window["native_rate_above_limit_fraction"] == exceeded / 20
        assert window["total_rate_saturation_count"] == saturated
        assert window["total_rate_saturation_fraction"] == saturated / 20
        assert window["changed_row_count"] == changed
        assert window["changed_value_count"] == changed
        assert window["unclipped_total_yaw_rate_rad_s"] == {
            "minimum": float(np.min(unclipped)),
            "mean": float(np.mean(unclipped, dtype=np.float64)),
            "maximum": float(np.max(unclipped)),
        }
        endpoint_trace = trace["held_poststep_target"]
        assert endpoint_trace["native_float32_c_sha256"] == _independent_hash(endpoint)
        assert endpoint_trace["issued_float32_c_sha256"] == _independent_hash(expected_endpoint)
    result = scorer.steering_measures(
        {"frames": rows, "trajectory": {"qpos": qpos, "current_reference": current}}
    )
    assert result["after_action_count"] == 3
    assert result["native_endpoint_window_first_row_mismatch_count"] == 3
    assert result["issued_endpoint_window_first_row_mismatch_count"] == 3
