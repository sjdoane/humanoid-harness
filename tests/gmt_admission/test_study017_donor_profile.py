from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def test_retained_donor_profile_uses_exact_half_open_boundaries():
    path = Path(__file__).parents[2] / "experiments/017_g1_execution_derived_reference/donor_profile.py"
    spec = importlib.util.spec_from_file_location("study017_donor_profile_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not module.DONOR_ROOT.is_dir():
        pytest.skip("retained Study012 donor fixture absent")
    result = module.donor_profile()
    rows = result["rows"]
    assert [row["donor_action_index"] for row in rows] == list(range(92, 197))
    assert rows[0]["start_phase_seconds"] == 0.0
    assert rows[-1]["end_phase_seconds"] == 2.1
    assert rows[-1]["poststep_progress_m"] == pytest.approx(2.0502309108945904, abs=1e-12)
    assert rows[-1]["poststep_root_height_m"] == pytest.approx(.5954433356039784, abs=1e-12)
    deepest = min(rows, key=lambda row: row["poststep_root_height_m"])
    assert deepest["donor_action_index"] == 179
    assert deepest["poststep_root_height_m"] == pytest.approx(.504942203119977, abs=1e-12)
