from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from oracle_composition.cli import main
from oracle_composition.effort import summarize_effort


def _record() -> dict:
    return {
        "schema_version": 1,
        "artifact": "gmt_g1_course_development_run",
        "status": "completed",
        "input_config_sha256": "a" * 64,
        "training": {"completed_transitions": 32768},
        "runtime": {"wall_seconds": 30.5},
        "claims": {"training_performed": True},
        "final_policy": {"all_development_gates_pass": False},
    }


def _write(path: Path, record: dict) -> tuple[Path, str]:
    encoded = json.dumps(record).encode()
    path.write_bytes(encoded)
    return path, hashlib.sha256(encoded).hexdigest()


def test_counts_completed_training_even_when_task_failed(tmp_path):
    first = _write(tmp_path / "first.json", _record())
    second = _record()
    second["input_config_sha256"] = "b" * 64
    second["training"]["completed_transitions"] = 131072
    result = summarize_effort([first, _write(tmp_path / "second.json", second)])
    assert result["selected_receipts"] == {
        "training_run_count": 2,
        "reported_training_transitions": 163840,
        "summed_run_wall_seconds": 61.0,
    }
    assert all(value is None for value in result["unmeasured"].values())
    assert len(result["runs"]) == 2


def test_probe_zero_is_explicit_not_imputed(tmp_path):
    record = _record()
    record["training"] = None
    record["claims"]["training_performed"] = False
    result = summarize_effort([_write(tmp_path / "probe.json", record)])
    assert result["selected_receipts"]["reported_training_transitions"] == 0
    assert result["unmeasured"]["total_simulator_steps_including_evaluation"] is None


@pytest.mark.parametrize("value", [True, -1, 1.5, "12", None, 0, 2**53])
def test_rejects_invalid_training_counts(tmp_path, value):
    record = _record()
    record["training"]["completed_transitions"] = value
    with pytest.raises(ValueError):
        summarize_effort([_write(tmp_path / "bad.json", record)])


@pytest.mark.parametrize("value", [True, -1, "12", None, float("nan"), float("inf")])
def test_rejects_invalid_duration(tmp_path, value):
    record = _record()
    record["runtime"]["wall_seconds"] = value
    with pytest.raises(ValueError):
        summarize_effort([_write(tmp_path / "bad.json", record)])


@pytest.mark.parametrize("field", ["training", "runtime", "claims", "input_config_sha256"])
def test_missing_evidence_is_not_zero(tmp_path, field):
    record = _record()
    del record[field]
    with pytest.raises(ValueError):
        summarize_effort([_write(tmp_path / "bad.json", record)])


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 2),
        ("status", "failed"),
        ("artifact", "unrelated"),
        ("claims", {"training_performed": False}),
        ("claims", {"training_performed": 1}),
    ],
)
def test_wrong_or_contradictory_receipts_refuse(tmp_path, field, value):
    record = _record()
    record[field] = value
    with pytest.raises(ValueError):
        summarize_effort([_write(tmp_path / "bad.json", record)])


def test_rejects_duplicate_aliases_and_copies(tmp_path):
    first = _write(tmp_path / "a.json", _record())
    copied = _write(tmp_path / "b.json", copy.deepcopy(_record()))
    for second in [first, copied]:
        with pytest.raises(ValueError, match="duplicate"):
            summarize_effort([first, second])


def test_rejects_links_missing_files_and_hash_changes(tmp_path):
    source, digest = _write(tmp_path / "a.json", _record())
    link = tmp_path / "link.json"
    link.symlink_to(source)
    for path, expected in [(link, digest), (source, "0" * 64), (tmp_path / "absent", digest)]:
        with pytest.raises(ValueError):
            summarize_effort([(path, expected)])


@pytest.mark.parametrize("metric", ["time", "transitions"])
def test_aggregate_overflow_refuses(tmp_path, metric):
    a = _record()
    if metric == "time":
        a["runtime"]["wall_seconds"] = 1e308
    else:
        a["training"]["completed_transitions"] = 2**53 - 1
    b = copy.deepcopy(a)
    b["input_config_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        summarize_effort([_write(tmp_path / "a.json", a), _write(tmp_path / "b.json", b)])


@pytest.mark.parametrize("receipts", [[], [(Path("not-read"), "a" * 64)] * 257])
def test_bounds_receipt_count(receipts):
    with pytest.raises(ValueError):
        summarize_effort(receipts)


def test_cli_is_read_only_and_reports_unknown_costs(tmp_path, capsys):
    path, digest = _write(tmp_path / "run.json", _record())
    before = path.read_bytes()
    assert main(["--json", "effort", "--run", str(path), digest]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["receipt_count"] == 1
    assert result["unmeasured"]["model_call_cost_usd"] is None
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]
    assert main(["effort", "--run", str(path), "0" * 64]) == 2
    assert "SHA-256 mismatch" in capsys.readouterr().err


def test_cli_does_not_import_training_or_simulator_libraries(tmp_path):
    path, digest = _write(tmp_path / "run.json", _record())
    script = """
import sys

class RefuseTrainingImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'gymnasium', 'mujoco', 'stable_baselines3'}:
            raise AssertionError(f'effort reporting imported {fullname}')

sys.meta_path.insert(0, RefuseTrainingImports())
from oracle_composition.cli import main
raise SystemExit(main(sys.argv[1:]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, "--json", "effort", "--run", str(path), digest],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["receipt_count"] == 1
