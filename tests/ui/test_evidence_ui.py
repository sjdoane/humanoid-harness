from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.request import urlopen

from oracle_composition.research import build_index
from oracle_composition.ui import create_server


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
        health = _get_json(f"{base}/health")
        status = _get_json(f"{base}/api/status")
        query = _get_json(f"{base}/api/research/query?q=phase%20recovery")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert health == {"authority": "read_only", "status": "ok"}
    assert status["evidence_class"] == "none"
    assert status["measured_evidence"] == []
    assert status["evidence_receipts"] == []
    assert query["results"]


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
