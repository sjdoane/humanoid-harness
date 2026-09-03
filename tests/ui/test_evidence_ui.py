from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from oracle_composition.research import build_index
from oracle_composition.ui import create_server
from oracle_composition.ui import local_evidence as local_evidence_module


def _write_record(directory: Path, paper_id: str, title: str) -> None:
    record = {
        "paper": {
            "id": paper_id,
            "title": title,
            "year": 2026,
            "primary_url": f"https://example.test/{paper_id}",
            "source_version": "test-v1",
        },
        "decision_relevance": {
            "tier": "core",
            "knobs": ["reference_composition"],
            "why_admitted": "Tests state-aware recovery decisions.",
        },
        "mechanisms": [],
        "parameters": [],
        "failure_modes": [],
        "evaluation": [],
        "relations": [],
        "evidence": [],
    }
    (directory / f"{paper_id}.json").write_text(json.dumps(record), encoding="utf-8")


def _get_json(url: str) -> dict[str, object]:
    with urlopen(url, timeout=2) as response:
        assert response.status == 200
        return json.loads(response.read())


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_local_exploration(
    root: Path,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> Path:
    directory = root / "artifacts/exploration/phase_oracle_holdout_v0"
    directory.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "status": "local_exploration_only",
        "locked_before_holdout": True,
        "redistributable": False,
        "claim_ceiling": "tier_k_local_oracle_diagnostic_not_formal_experiment_evidence",
        "question": "Does bounded phase correction improve recovery?",
        "reason_not_admitted": ["source license absent", "no Tier-D certificate"],
        "frozen_runtime": {"max_actions": 1000},
        "reference": {"admission": "Tier-K_not_admitted"},
        "bounded_phase_rule": {},
        "recovery_fallback_rule": {},
        "arms": {
            "T0_G0": "elapsed-time phase",
            "T0_G1": "elapsed-time phase plus fallback",
            "T1_G0": "bounded phase",
            "T1_G1": "bounded phase plus fallback",
            "base_context": "controller without the residual tracker",
        },
        "paired_evaluation": {
            "seeds": list(range(3000, 3020)),
            "conditions": {
                "nominal": "none",
                "lateral_velocity": "lateral push",
                "pitch_velocity_falsifier": "pitch push",
            },
        },
        "visual_evidence": {
            "seed": 3000,
            "condition": "lateral_velocity",
            "arms": ["T0_G0", "T0_G1", "T1_G0", "T1_G1"],
            "selection_rule": "first predetermined holdout seed; never selected from outcome",
        },
        "locked_expectations": {
            "nominal": "T1_G1 must retain 20/20 no-collapse outcomes",
            "lateral_velocity": ("T1_G1 should exceed T0_G0 by at least two no-collapse outcomes"),
            "pitch_velocity_falsifier": (
                "no improvement is assumed; failure identifies missing controller recovery capability"
            ),
            "attribution": (
                "report all four T x G arms; do not attribute a joint-arm gain to phase alone"
            ),
        },
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    (directory / "manifest.json").write_bytes(manifest_bytes)
    manifest_sha = _sha256(manifest_bytes)
    arms = tuple(manifest["arms"])
    conditions = tuple(manifest["paired_evaluation"]["conditions"])
    seeds = tuple(manifest["paired_evaluation"]["seeds"])
    rows = []
    combinations = (
        (arm, condition, seed) for arm in arms for condition in conditions for seed in seeds
    )
    for run_order, (arm, condition, seed) in enumerate(combinations, start=1):
        no_collapse = not (
            (arm == "T0_G0" and condition == "lateral_velocity" and seed in {3000, 3001})
            or (arm == "T1_G1" and condition == "lateral_velocity" and seed == 3000)
            or condition == "pitch_velocity_falsifier"
        )
        rows.append(
            {
                "action_saturation_count": 0,
                "actions_executed": 1000,
                "arm": arm,
                "collapsed_action_count": 0 if no_collapse else 100,
                "condition": condition,
                "environment_return": float(10_000 + run_order),
                "final_recovered": no_collapse,
                "final_root_height_m": 1.3 if no_collapse else 0.8,
                "final_torso_up_z": 0.9 if no_collapse else 0.4,
                "first_collapse_action": None if no_collapse else 901,
                "manifest_sha256": manifest_sha,
                "no_collapse": no_collapse,
                "phase_correction_count": 700 if arm == "T1_G1" else 0,
                "recovery_entry_count": 1 if arm == "T1_G1" else 0,
                "recovery_exit_count": 1 if arm == "T1_G1" else 0,
                "recovery_gate_off_count": 300 if arm == "T1_G1" else 0,
                "root_x_displacement_m": float(70 + run_order),
                "run_order": run_order,
                "seed": seed,
            }
        )
    runs = b"".join((json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows)
    (directory / "runs.jsonl").write_bytes(runs)
    runs_sha = _sha256(runs)

    groups: dict[str, object] = {}
    for arm in arms:
        for condition in conditions:
            selected = [row for row in rows if row["arm"] == arm and row["condition"] == condition]
            count = len(selected)
            groups[f"{arm}/{condition}"] = {
                "final_recovered_count": sum(bool(row["final_recovered"]) for row in selected),
                "mean_collapsed_action_count": sum(
                    int(row["collapsed_action_count"]) for row in selected
                )
                / count,
                "mean_environment_return": sum(float(row["environment_return"]) for row in selected)
                / count,
                "mean_first_collapse_or_1001": sum(
                    int(row["first_collapse_action"])
                    if row["first_collapse_action"] is not None
                    else 1001
                    for row in selected
                )
                / count,
                "mean_phase_correction_count": sum(
                    int(row["phase_correction_count"]) for row in selected
                )
                / count,
                "mean_recovery_gate_off_count": sum(
                    int(row["recovery_gate_off_count"]) for row in selected
                )
                / count,
                "mean_root_x_displacement_m": sum(
                    float(row["root_x_displacement_m"]) for row in selected
                )
                / count,
                "n": count,
                "no_collapse_count": sum(bool(row["no_collapse"]) for row in selected),
            }
    summary = {
        "status": "local_exploration_only",
        "completed_runs": len(rows),
        "scheduled_runs": len(rows),
        "manifest_sha256": manifest_sha,
        "runs_sha256": runs_sha,
        "groups": groups,
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    figure_rows = [
        "arm,condition,n,no_collapse_count,mean_collapsed_action_count,"
        "mean_environment_return,mean_recovery_gate_off_fraction"
    ]
    for name, group in groups.items():
        arm, condition = name.split("/", maxsplit=1)
        figure_rows.append(
            f"{arm},{condition},{group['n']},{group['no_collapse_count']},"
            f"{group['mean_collapsed_action_count']},{group['mean_environment_return']},"
            f"{float(group['mean_recovery_gate_off_count']) / 1000}"
        )
    figure_data = ("\n".join(figure_rows) + "\n").encode()
    alt_text = b"Synthetic held-out result figure."
    figure = b"\x89PNG\r\n\x1a\nsynthetic figure"
    figure_svg = b"<svg xmlns='http://www.w3.org/2000/svg'></svg>"
    poster = b"\x89PNG\r\n\x1a\nsynthetic poster"
    video = b"\x00\x00\x00\x18ftypisomsynthetic-video"
    (directory / "figure_data.csv").write_bytes(figure_data)
    (directory / "holdout_summary.png").write_bytes(figure)
    (directory / "holdout_summary.svg").write_bytes(figure_svg)
    (directory / "lateral_seed3000_poster.png").write_bytes(poster)
    (directory / "lateral_seed3000_four_arm.mp4").write_bytes(video)
    (directory / "holdout_summary.alt.txt").write_bytes(alt_text)
    figure_receipt = {
        "status": "local_exploration_only",
        "audience": "test",
        "publisher_requirements": "not applicable",
        "source_manifest": "manifest.json",
        "source_manifest_sha256": manifest_sha,
        "source_runs": "runs.jsonl",
        "source_runs_sha256": runs_sha,
        "transformations": ["none"],
        "uncertainty": "not claimed",
        "outputs": {
            "figure_data.csv": _sha256(figure_data),
            "holdout_summary.alt.txt": _sha256(alt_text),
            "holdout_summary.png": _sha256(figure),
            "holdout_summary.svg": _sha256(figure_svg),
        },
    }
    (directory / "holdout_summary.figure_manifest.json").write_text(
        json.dumps(figure_receipt), encoding="utf-8"
    )
    video_receipt = {
        "status": "local_exploration_only",
        "manifest_sha256": manifest_sha,
        "runs_sha256": runs_sha,
        "seed": 3000,
        "condition": "lateral_velocity",
        "selection": "first predetermined holdout seed",
        "arms": ["T0_G0", "T0_G1", "T1_G0", "T1_G1"],
        "claim_ceiling": "visible_closed_loop_execution_not_formal_oracle_evidence",
        "perturb_action": 300,
        "render_stride_actions": 2,
        "video_dimensions_px": [960, 1008],
        "video_duration_seconds": 15.03,
        "video_fps": 100 / 3,
        "video_frames": 501,
        "metrics": {
            row["arm"]: {
                "collapsed": row["collapsed_action_count"],
                "first_collapse": row["first_collapse_action"],
                "return": row["environment_return"],
            }
            for row in rows
            if row["condition"] == "lateral_velocity"
            and row["seed"] == 3000
            and row["arm"] != "base_context"
        },
        "poster_sha256": _sha256(poster),
        "video_sha256": _sha256(video),
    }
    (directory / "lateral_seed3000_four_arm.video_receipt.json").write_text(
        json.dumps(video_receipt), encoding="utf-8"
    )
    if monkeypatch is not None:
        monkeypatch.setattr(local_evidence_module, "_V0_MANIFEST_SHA256", manifest_sha)
        monkeypatch.setattr(local_evidence_module, "_V0_RUNS_SHA256", runs_sha)
        pinned_files = (
            "figure_data.csv",
            "holdout_summary.alt.txt",
            "holdout_summary.figure_manifest.json",
            "holdout_summary.png",
            "holdout_summary.svg",
            "lateral_seed3000_four_arm.video_receipt.json",
            "lateral_seed3000_poster.png",
            "lateral_seed3000_four_arm.mp4",
        )
        monkeypatch.setattr(
            local_evidence_module,
            "_V0_FILE_SHA256",
            {name: _sha256((directory / name).read_bytes()) for name in pinned_files},
        )
    return directory


def _rewrite_json(path: Path, mutate: Callable[[dict[str, object]], None]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def _rebind_manifest(directory: Path, mutate: Callable[[dict[str, object]], None]) -> None:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha = _sha256(manifest_bytes)

    rows = [json.loads(line) for line in (directory / "runs.jsonl").read_bytes().splitlines()]
    for row in rows:
        row["manifest_sha256"] = manifest_sha
    runs = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    (directory / "runs.jsonl").write_bytes(runs)
    runs_sha = _sha256(runs)

    _rewrite_json(
        directory / "summary.json",
        lambda value: value.update(manifest_sha256=manifest_sha, runs_sha256=runs_sha),
    )
    _rewrite_json(
        directory / "holdout_summary.figure_manifest.json",
        lambda value: value.update(
            source_manifest_sha256=manifest_sha,
            source_runs_sha256=runs_sha,
        ),
    )
    _rewrite_json(
        directory / "lateral_seed3000_four_arm.video_receipt.json",
        lambda value: value.update(manifest_sha256=manifest_sha, runs_sha256=runs_sha),
    )


def test_ui_serves_static_shell_and_read_only_evidence_api(tmp_path: Path) -> None:
    extractions = tmp_path / "extractions"
    extractions.mkdir()
    _write_record(extractions, "2600.00001", "Phase recovery")
    database = tmp_path / "graph.db"
    build_index(extractions, database)

    server = create_server(
        port=0,
        project_root=tmp_path,
        database=database,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        with urlopen(base, timeout=2) as response:
            html = response.read().decode("utf-8")
            assert "What is true right now?" in html
            assert response.headers["Content-Security-Policy"]
            assert response.headers["Cross-Origin-Resource-Policy"] == "same-origin"
        health = _get_json(f"{base}/health")
        status = _get_json(f"{base}/api/status")
        query = _get_json(f"{base}/api/research/query?q=phase%20recovery")
        exploration = _get_json(f"{base}/api/exploration/latest")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert health == {"authority": "read_only", "status": "ok"}
    assert status["evidence_class"] == "none"
    assert status["measured_evidence"] == []
    assert status["evidence_receipts"] == []
    assert query["results"]
    assert exploration["state"] == "unavailable"


def test_ui_serves_only_integrity_checked_local_exploration(tmp_path: Path, monkeypatch) -> None:
    _write_local_exploration(tmp_path, monkeypatch)
    server = create_server(port=0, project_root=tmp_path, database=tmp_path / "missing.db")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        exploration = _get_json(f"{base}/api/exploration/latest")
        with urlopen(f"{base}/local-evidence/holdout-summary.png", timeout=2) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "image/png"
            assert response.read() == b"\x89PNG\r\n\x1a\nsynthetic figure"
        request = Request(
            f"{base}/local-evidence/lateral-seed3000-four-arm.mp4",
            headers={"Range": "bytes=4-8"},
        )
        with urlopen(request, timeout=2) as response:
            assert response.status == 206
            assert response.headers["Accept-Ranges"] == "bytes"
            assert response.headers["Content-Range"] == "bytes 4-8/27"
            assert response.read() == b"ftypi"
        head = Request(
            f"{base}/local-evidence/lateral-seed3000-four-arm.mp4",
            method="HEAD",
        )
        with urlopen(head, timeout=2) as response:
            assert response.status == 200
            assert response.headers["Content-Length"] == "27"
            assert response.read() == b""
        invalid_range = Request(
            f"{base}/local-evidence/lateral-seed3000-four-arm.mp4",
            headers={"Range": "bytes=1-2,4-5"},
        )
        try:
            urlopen(invalid_range, timeout=2)
        except HTTPError as exc:
            assert exc.code == 416
        else:
            raise AssertionError("multiple byte ranges must fail closed")
        hostile_host = Request(
            f"{base}/local-evidence/holdout-summary.png",
            headers={"Host": "attacker.example"},
        )
        try:
            urlopen(hostile_host, timeout=2)
        except HTTPError as exc:
            assert exc.code == 421
        else:
            raise AssertionError("unexpected Host headers must fail closed")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert exploration["state"] == "available"
    assert exploration["authority"] == "local_exploration_only"
    assert exploration["reference_admission"] == "Tier-K_not_admitted"
    assert exploration["completed_runs"] == 300
    assert exploration["comparisons"][1]["passed"] is False
    assert str(tmp_path) not in json.dumps(exploration)


def test_ui_rejects_tampered_local_exploration_and_media(tmp_path: Path, monkeypatch) -> None:
    directory = _write_local_exploration(tmp_path, monkeypatch)
    (directory / "runs.jsonl").write_bytes(b"tampered\n")
    server = create_server(port=0, project_root=tmp_path, database=tmp_path / "missing.db")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        exploration = _get_json(f"{base}/api/exploration/latest")
        try:
            urlopen(f"{base}/local-evidence/holdout-summary.png", timeout=2)
        except HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("tampered bundles must not expose media")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert exploration["state"] == "rejected"
    assert exploration["authority"] == "local_exploration_only"


def test_ui_normalizes_oversized_local_json_as_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "artifacts/exploration/phase_oracle_holdout_v0"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text('{"value":' + "9" * 5_000 + "}")

    status = local_evidence_module.local_exploration_status(tmp_path)

    assert status["state"] == "rejected"
    assert "JSON is invalid" in str(status["detail"])


def test_local_evidence_number_normalizes_float_overflow() -> None:
    with pytest.raises(local_evidence_module.LocalEvidenceError, match="must be finite"):
        local_evidence_module._number(10**400, "hostile")


def test_ui_rejects_rebound_run_ledger_outside_reviewed_identity(
    tmp_path: Path, monkeypatch
) -> None:
    directory = _write_local_exploration(tmp_path, monkeypatch)
    first_row = (directory / "runs.jsonl").read_bytes().splitlines(keepends=True)[0]
    (directory / "runs.jsonl").write_bytes(first_row)
    runs_sha = _sha256(first_row)
    _rewrite_json(
        directory / "summary.json",
        lambda value: value.__setitem__("runs_sha256", runs_sha),
    )
    _rewrite_json(
        directory / "holdout_summary.figure_manifest.json",
        lambda value: value.__setitem__("source_runs_sha256", runs_sha),
    )
    _rewrite_json(
        directory / "lateral_seed3000_four_arm.video_receipt.json",
        lambda value: value.__setitem__("runs_sha256", runs_sha),
    )

    server = create_server(port=0, project_root=tmp_path, database=tmp_path / "missing.db")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        exploration = _get_json(f"http://{host}:{port}/api/exploration/latest")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert exploration["state"] == "rejected"
    assert "run-ledger identity" in str(exploration["detail"])


def test_ui_rejects_smaller_design_even_when_all_hashes_are_rebound(
    tmp_path: Path, monkeypatch
) -> None:
    directory = _write_local_exploration(tmp_path, monkeypatch)
    _rebind_manifest(directory, lambda value: value["arms"].pop("T0_G1"))

    server = create_server(port=0, project_root=tmp_path, database=tmp_path / "missing.db")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        exploration = _get_json(f"http://{host}:{port}/api/exploration/latest")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert exploration["state"] == "rejected"
    assert "manifest identity" in str(exploration["detail"])


def test_ui_rejects_valid_signature_media_rebound_outside_reviewed_identity(
    tmp_path: Path, monkeypatch
) -> None:
    directory = _write_local_exploration(tmp_path, monkeypatch)
    replacement = b"\x00\x00\x00\x18ftypisomdifferent-valid-signature-video"
    (directory / "lateral_seed3000_four_arm.mp4").write_bytes(replacement)
    _rewrite_json(
        directory / "lateral_seed3000_four_arm.video_receipt.json",
        lambda value: value.__setitem__("video_sha256", _sha256(replacement)),
    )

    server = create_server(port=0, project_root=tmp_path, database=tmp_path / "missing.db")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        exploration = _get_json(f"http://{host}:{port}/api/exploration/latest")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert exploration["state"] == "rejected"
    assert "media receipt identity" in str(exploration["detail"])


def test_ui_rejects_symlinked_local_media(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    directory = _write_local_exploration(project, monkeypatch)
    target = tmp_path / "outside.png"
    target.write_bytes((directory / "holdout_summary.png").read_bytes())
    (directory / "holdout_summary.png").unlink()
    (directory / "holdout_summary.png").symlink_to(target)

    server = create_server(port=0, project_root=project, database=project / "missing.db")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        exploration = _get_json(f"http://{host}:{port}/api/exploration/latest")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert exploration["state"] == "rejected"


def test_local_evidence_reader_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "source.fifo"
    os.mkfifo(fifo)

    with pytest.raises(local_evidence_module.LocalEvidenceError, match="regular file"):
        local_evidence_module._bounded_file(tmp_path, Path("source.fifo"), limit=8)


def test_ui_rejects_non_loopback_bind(tmp_path: Path, monkeypatch) -> None:
    _write_local_exploration(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="loopback"):
        create_server(
            host="0.0.0.0",
            port=0,
            project_root=tmp_path,
            database=tmp_path / "missing.db",
        )


def test_ui_reports_corrupt_index_as_not_built(tmp_path: Path) -> None:
    database = tmp_path / "graph.db"
    database.write_bytes(b"not a sqlite database")
    server = create_server(port=0, project_root=tmp_path, database=database)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        stats = _get_json(f"http://{host}:{port}/api/research/stats")
        status = _get_json(f"http://{host}:{port}/api/status")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert stats["state"] == "not_built"
    assert "corrupt or incomplete" in stats["detail"]
    assert status["knowledge_graph"]["state"] == "not_built"
