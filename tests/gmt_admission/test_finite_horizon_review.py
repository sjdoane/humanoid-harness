"""Pre-launch peer-requested coverage; no simulator or checkpoint loading."""

import json
from dataclasses import replace

import numpy as np
import pytest

from oracle_composition.adapters.gmt.course_config import load_run_config
from oracle_composition.adapters.gmt.course_runtime import FINITE_HORIZON_RUNTIME
from oracle_composition.adapters.gmt.training_telemetry import EpisodeAccumulator
from tests.gmt_admission.test_course_config import admitted_config as admitted_fixture
from tests.gmt_admission.test_gym_env import _env, _task


@pytest.fixture
def admitted_config(tmp_path, monkeypatch):
    return admitted_fixture.__wrapped__(tmp_path, monkeypatch)


@pytest.mark.parametrize("kind", ["loop_segment", "four_states"])
def test_finite_runtime_rejects_extra_composition_capability(admitted_config, kind):
    raw, path = admitted_config
    raw["schema_version"] = 3
    raw["runtime"] = FINITE_HORIZON_RUNTIME.config_value
    if kind == "loop_segment":
        raw["segments"]["walk"].update(
            boundary="entry_once_then_loop",
            loop_start_seconds=3.9,
            entry_phase_end_seconds=0.15,
            exit_at_loop_boundary=True,
        )
    else:
        raw["oracle"]["states"]["rise"] = {"behavior": "walk", "min_dwell": 25}
        raw["oracle"]["transitions"][1]["to"] = "rise"
        raw["oracle"]["transitions"].append(
            {"from": "rise", "to": "after", "priority": 0, "guard": "dwell >= 25"}
        )
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=r"loop|states"):
        load_run_config(path)


@pytest.mark.parametrize("fall_substep", [None, 11])
def test_finite_terminal_telemetry_counts_metrics_once(fall_substep):
    env, _, _ = _env(
        [0.0, 0.25],
        task=replace(_task(), horizon_steps=1),
        fall_substep=fall_substep,
        runtime=FINITE_HORIZON_RUNTIME,
    )
    env.reset(seed=7)
    _, reward, terminated, truncated, info = env.step(np.zeros(23, dtype=np.float32))
    assert terminated and not truncated
    accumulator = EpisodeAccumulator()
    accumulator.observe([reward], [terminated or truncated], [info])
    summary = accumulator.take_summary()
    assert summary["completed"] == 1
    assert summary["horizons"] == 1
    assert summary["falls"] == int(fall_substep is not None)
    assert accumulator.incomplete() == []
