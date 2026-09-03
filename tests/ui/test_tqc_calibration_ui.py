from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from oracle_composition.ui import create_server
from oracle_composition.ui import local_tqc_calibration as tqc_ui

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_bound_file(root: Path, relative: Path, source: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(source)
    return path


def _write_e0_fixture(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    design_source = (PROJECT_ROOT / tqc_ui.TQC_CALIBRATION_DESIGN_PATH).read_bytes()
    design = json.loads(design_source)
    source = (PROJECT_ROOT / tqc_ui.TQC_CALIBRATION_SOURCE_PATH).read_bytes()
    _write_bound_file(root, tqc_ui.TQC_CALIBRATION_DESIGN_PATH, design_source)
    _write_bound_file(root, tqc_ui.TQC_CALIBRATION_SOURCE_PATH, source)

    runtime: dict[str, object] = {
        "action_shape": [17],
        "calibration_source_sha256": _sha256(source),
        "dependency_lock_sha256": "2" * 64,
        "environment_id": "Humanoid-v5",
        "observation_shape": [348],
        "sb3_contrib_version": "2.9.0",
        "source_tree_sha256": "1" * 64,
        "stable_baselines3_version": "2.9.0",
    }
    runtime["runtime_sha256"] = _sha256(_canonical_json(runtime))
    initial_parameters = {"structure_sha256": "3" * 64, "value_sha256": "4" * 64}
    observed_model: dict[str, object] = {
        "algorithm_class": "sb3_contrib.tqc.tqc.TQC",
        "device": "cpu",
        "expected_replay_allocation_bytes": 5_648_000_000,
        "initial_parameters": initial_parameters,
        "logger_directory": None,
        "logger_output_format_count": 0,
        "n_envs": 5,
        "pending_worker_seeds": [92001, 92002, 92003, 92004, 92005],
        "replay_allocation_bytes": 5_648_000_000,
        "vec_normalize_active": False,
    }
    observed_model["model_semantic_sha256"] = _sha256(_canonical_json(observed_model))
    windows = [
        {
            "start_environment_step": start,
            "end_environment_step": start + 10_000,
            "environment_steps_per_second": 600.0,
            "wall_seconds": 50 / 3,
        }
        for start in range(10_000, 100_000, 10_000)
    ]
    receipt = {
        "authoritative_execution": True,
        "automatic_promotion": False,
        "behavioral_claim": None,
        "calibration_gate_passed": True,
        "calibration_id": "tqc_humanoid_resource_calibration/v1",
        "checkpoint_emitted": False,
        "claim_boundary": "resource_integrity_only_no_behavior_or_controller_claim/v1",
        "cleanup_failures": [],
        "completion_status": "complete",
        "controller_artifact": None,
        "controller_claim": None,
        "design": design,
        "design_artifact_byte_count": len(design_source),
        "design_artifact_sha256": _sha256(design_source),
        "design_semantic_sha256": _sha256(_canonical_json(design)),
        "disk_measurement": "test",
        "disposable_resource_probe_acknowledged": True,
        "eligible_for_behavioral_evaluation": False,
        "eligible_for_controller_training": False,
        "environment_steps_per_second": 600.0,
        "evidence_purpose": "resource_calibration",
        "execution_callable_authority": "canonical_production_callables/v1",
        "expected_environment_steps": 100_000,
        "expected_gradient_updates": 19_980,
        "expected_vector_steps": 20_000,
        "failure_reason": None,
        "failure_stage": None,
        "final_parameters": {"structure_sha256": "3" * 64, "value_sha256": "5" * 64},
        "finite_parameter_checks": 2,
        "finite_rollout_signal_checks": 20_000,
        "finite_training_signals": {},
        "measured_workload_gates_passed": True,
        "minimum_free_disk_bytes": 100 * 1024**3,
        "model_disposition": "discarded_unserialized",
        "normalizer_emitted": False,
        "observed_environment_steps": 100_000,
        "observed_gradient_updates": 19_980,
        "observed_model": observed_model,
        "observed_vector_steps": 20_000,
        "oom_before_final_receipt_possible": True,
        "output_reservation": "test",
        "parameter_count": 827_526,
        "peak_rss_bytes": 4 * 1024**3,
        "replay_buffer_allocation_bytes": 5_648_000_000,
        "replay_buffer_emitted": False,
        "replay_buffer_page_write": {
            "claim_boundary": "page_addresses_written_no_residency_claim/v1",
            "method": "write_zero_every_os_page_plus_final_byte/v1",
            "total_array_bytes": 5_648_000_000,
        },
        "resource_checks": 1,
        "rss_measurement": "test",
        "rss_threshold_semantics": "sampled_failure_threshold_not_os_memory_cap/v1",
        "rss_transient_above_threshold_possible": True,
        "runtime": runtime,
        "schema_version": 1,
        "seed": 92001,
        "seed_role": "permanently_excluded_disposable_calibration",
        "sustained_throughput_windows": windows,
        "training_wall_seconds": 150.0,
        "wall_seconds_from_preflight_through_learning": 151.0,
    }
    receipt_path = _write_bound_file(
        root, tqc_ui.TQC_CALIBRATION_RECEIPT_PATH, _canonical_json(receipt)
    )
    figure = b"\x89PNG\r\n\x1a\nreviewed-test-figure"
    _write_bound_file(root, tqc_ui.TQC_CALIBRATION_FIGURE_PATH, figure)
    monkeypatch.setattr(tqc_ui, "_REVIEWED_RECEIPT_SHA256", _sha256(receipt_path.read_bytes()))
    monkeypatch.setattr(tqc_ui, "_REVIEWED_DESIGN_SHA256", _sha256(design_source))
    monkeypatch.setattr(tqc_ui, "_REVIEWED_SOURCE_SHA256", _sha256(source))
    monkeypatch.setattr(tqc_ui, "_REVIEWED_FIGURE_SHA256", _sha256(figure))
    return receipt_path


def _serve(root: Path) -> tuple[object, threading.Thread, str]:
    server = create_server(port=0, project_root=root, database=root / "missing.db")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    return server, thread, f"http://{host}:{port}"


def test_ui_serves_compact_reviewed_e0_receipt_and_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = _write_e0_fixture(tmp_path, monkeypatch)
    server, thread, base = _serve(tmp_path)
    try:
        with urlopen(f"{base}/api/experiments/e0-tqc", timeout=2) as response:
            payload = json.loads(response.read())
        with urlopen(f"{base}/local-evidence/e0-tqc-resource.png", timeout=2) as response:
            figure = response.read()
            assert response.headers["Content-Type"] == "image/png"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert payload["state"] == "available"
    assert payload["authority"] == "local_reviewed_resource_calibration_receipt"
    assert payload["evidence_class"] == "E0_resource_only"
    assert payload["gate_passed"] is True
    assert payload["observed_environment_steps"] == 100_000
    assert payload["observed_gradient_updates"] == 19_980
    assert payload["checkpoint_emitted"] is False
    assert payload["eligible_for_controller_training"] is False
    assert payload["eligible_for_behavioral_evaluation"] is False
    assert payload["media"] == {"figure": "/local-evidence/e0-tqc-resource.png"}
    assert payload["receipts"]["receipt_sha256"] == _sha256(receipt_path.read_bytes())
    assert figure.startswith(b"\x89PNG\r\n\x1a\n")
    serialized = json.dumps(payload)
    assert str(tmp_path) not in serialized
    assert '"design"' not in serialized
    assert '"runtime"' not in serialized
    assert '"observed_model"' not in serialized


def test_ui_marks_missing_e0_receipt_unavailable(tmp_path: Path) -> None:
    assert tqc_ui.local_tqc_calibration_status(tmp_path) == {
        "state": "unavailable",
        "authority": "local_reviewed_resource_calibration_receipt",
        "detail": "no reviewed local E0 TQC resource receipt is present",
    }


def test_ui_rejects_tampered_e0_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    receipt_path = _write_e0_fixture(tmp_path, monkeypatch)
    receipt = json.loads(receipt_path.read_bytes())
    receipt["peak_rss_bytes"] += 1
    receipt_path.write_bytes(_canonical_json(receipt))

    payload = tqc_ui.local_tqc_calibration_status(tmp_path)

    assert payload["state"] == "rejected"
    assert "receipt identity" in str(payload["detail"])


def test_ui_rejects_rebound_e0_controller_eligibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = _write_e0_fixture(tmp_path, monkeypatch)
    receipt = json.loads(receipt_path.read_bytes())
    receipt["eligible_for_controller_training"] = True
    receipt_path.write_bytes(_canonical_json(receipt))
    monkeypatch.setattr(tqc_ui, "_REVIEWED_RECEIPT_SHA256", _sha256(receipt_path.read_bytes()))

    payload = tqc_ui.local_tqc_calibration_status(tmp_path)

    assert payload["state"] == "rejected"
    assert "successful E0 gate" in str(payload["detail"])


def test_ui_rejects_e0_after_bound_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_e0_fixture(tmp_path, monkeypatch)
    (tmp_path / tqc_ui.TQC_CALIBRATION_SOURCE_PATH).write_bytes(b"changed after E0")

    payload = tqc_ui.local_tqc_calibration_status(tmp_path)

    assert payload["state"] == "rejected"
    assert "calibration source differs" in str(payload["detail"])


def test_ui_omits_tampered_optional_e0_figure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_e0_fixture(tmp_path, monkeypatch)
    (tmp_path / tqc_ui.TQC_CALIBRATION_FIGURE_PATH).write_bytes(b"not the reviewed PNG")
    payload = tqc_ui.local_tqc_calibration_status(tmp_path)
    server, thread, base = _serve(tmp_path)
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(f"{base}/local-evidence/e0-tqc-resource.png", timeout=2)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert payload["state"] == "available"
    assert payload["media"] is None
    assert error.value.code == 404


def test_e0_frontend_states_the_resource_only_boundary(tmp_path: Path) -> None:
    server, thread, base = _serve(tmp_path)
    try:
        with urlopen(base, timeout=2) as response:
            html = response.read().decode("utf-8")
        with urlopen(f"{base}/app.js", timeout=2) as response:
            script = response.read().decode("utf-8")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert "Tracker bootstrap" in html
    assert "Did the E0 workload fit this host?" in html
    assert "Measured workload checks; none evaluate humanoid behavior" in html
    assert "Resource evidence only" in html
    assert "Page addresses written; no residency claim" in html
    assert 'loadJson("/api/experiments/e0-tqc")' in script
    assert "payload.eligible_for_controller_training !== false" in script
    assert "payload.eligible_for_behavioral_evaluation !== false" in script
