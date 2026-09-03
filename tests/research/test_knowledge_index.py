from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from oracle_composition.research import (
    KnowledgeIndexError,
    build_index,
    index_stats,
    query_index,
)


def _record(
    paper_id: str,
    *,
    title: str,
    status: str = "active",
) -> dict[str, object]:
    return {
        "paper": {
            "id": paper_id,
            "title": title,
            "year": 2026,
            "primary_url": f"https://example.test/{paper_id}",
            "source_version": "test-v1",
            "publication_status": status,
        },
        "decision_relevance": {
            "tier": "core",
            "knobs": ["reference_composition"],
            "why_admitted": "Tests state-aware recovery decisions.",
        },
        "mechanisms": [
            {
                "id": f"{paper_id}:phase_guard",
                "name": "Phase recovery guard",
                "description": "Rejoins a reference after a state-based recovery condition.",
                "evidence_ids": ["E1"],
            }
        ],
        "parameters": [
            {
                "name": "minimum_dwell",
                "value": 4,
                "units": "steps",
                "context": "Prevents transition chatter.",
                "evidence_ids": ["E1"],
            }
        ],
        "failure_modes": [],
        "evaluation": [],
        "relations": [
            {
                "source": f"{paper_id}:phase_guard",
                "relation": "MITIGATES",
                "target": "oscillating recovery transition",
                "evidence_ids": ["E1"],
            }
        ],
        "evidence": [
            {
                "id": "E1",
                "source_url": f"https://example.test/{paper_id}#e1",
                "locator": "Section 2",
                "claim": "A state guard controls recovery and phase rejoin.",
                "evidence_type": "paper_text",
            }
        ],
    }


def _write_record(directory: Path, record: dict[str, object]) -> None:
    paper = record["paper"]
    assert isinstance(paper, dict)
    path = directory / f"{paper['id']}.json"
    path.write_text(json.dumps(record), encoding="utf-8")


def test_builds_queryable_graph_with_provenance_and_arbitrary_concepts(
    tmp_path: Path,
) -> None:
    extractions = tmp_path / "extractions"
    extractions.mkdir()
    _write_record(extractions, _record("2600.00001", title="Closed-loop phase recovery"))
    _write_record(
        extractions,
        _record("2600.00002", title="Withdrawn recovery note", status="withdrawn"),
    )
    database = tmp_path / "knowledge" / "graph.db"

    built = build_index(extractions, database)
    results = query_index("phase recovery", database)

    assert built == index_stats(database)
    assert built["papers"] == 2
    assert built["active_papers"] == 1
    assert built["evidence_items"] == 2
    assert any(item["paper_id"] == "2600.00001" for item in results)
    assert all(item["paper_id"] != "2600.00002" for item in results)

    connection = sqlite3.connect(database)
    relation = connection.execute(
        "SELECT relation, provenance FROM edges WHERE relation='MITIGATES'"
    ).fetchone()
    concept = connection.execute(
        "SELECT name FROM nodes WHERE type='concept' AND name=?",
        ("oscillating recovery transition",),
    ).fetchone()
    connection.close()
    assert relation == ("MITIGATES", "extraction:2600.00001.json")
    assert concept == ("oscillating recovery transition",)


def test_inactive_sources_are_returned_only_when_requested(tmp_path: Path) -> None:
    extractions = tmp_path / "extractions"
    extractions.mkdir()
    _write_record(
        extractions,
        _record("2600.00002", title="Withdrawn recovery note", status="withdrawn"),
    )
    database = tmp_path / "graph.db"
    build_index(extractions, database)

    assert query_index("withdrawn", database) == []
    assert query_index("withdrawn", database, include_inactive=True)
    active_results = query_index("oscillating recovery transition", database)
    inactive_results = query_index(
        "oscillating recovery transition",
        database,
        include_inactive=True,
    )
    assert all(
        item["title"] != "oscillating recovery transition"
        for item in active_results
    )
    assert any(
        item["title"] == "oscillating recovery transition"
        for item in inactive_results
    )


def test_already_qualified_evidence_ids_are_not_double_prefixed(tmp_path: Path) -> None:
    extractions = tmp_path / "extractions"
    extractions.mkdir()
    record = _record("2600.00001", title="Qualified evidence")
    record["evidence"][0]["id"] = "2600.00001:E1"
    record["mechanisms"][0]["evidence_ids"] = ["2600.00001:E1"]
    database = tmp_path / "graph.db"

    _write_record(extractions, record)
    build_index(extractions, database)

    connection = sqlite3.connect(database)
    evidence_ids = connection.execute("SELECT id FROM evidence").fetchall()
    edge_evidence = connection.execute(
        "SELECT evidence_ids_json FROM edges WHERE relation='REPORTS'"
    ).fetchall()
    connection.close()
    assert evidence_ids == [("2600.00001:E1",)]
    assert all(
        json.loads(item[0]) == ["2600.00001:E1"]
        for item in edge_evidence
    )


def test_missing_edge_evidence_fails_closed(tmp_path: Path) -> None:
    extractions = tmp_path / "extractions"
    extractions.mkdir()
    record = _record("2600.00001", title="Missing evidence")
    record["mechanisms"][0]["evidence_ids"] = ["E404"]
    _write_record(extractions, record)

    with pytest.raises(KnowledgeIndexError, match="references missing evidence"):
        build_index(extractions, tmp_path / "graph.db")


def test_missing_or_empty_corpus_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(KnowledgeIndexError, match="not found"):
        build_index(tmp_path / "missing", tmp_path / "graph.db")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(KnowledgeIndexError, match="no extraction"):
        build_index(empty, tmp_path / "graph.db")


def test_each_extraction_is_read_once_for_claims_and_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    extractions = tmp_path / "extractions"
    extractions.mkdir()
    _write_record(extractions, _record("2600.00001", title="Single read"))
    source = extractions / "2600.00001.json"
    original_read_bytes = Path.read_bytes
    source_reads = 0

    def counted_read_bytes(path: Path) -> bytes:
        nonlocal source_reads
        if path == source:
            source_reads += 1
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
    build_index(extractions, tmp_path / "graph.db")

    assert source_reads == 1


def test_query_rejects_empty_text_and_unbounded_limits(tmp_path: Path) -> None:
    extractions = tmp_path / "extractions"
    extractions.mkdir()
    _write_record(extractions, _record("2600.00001", title="Phase recovery"))
    database = tmp_path / "graph.db"
    build_index(extractions, database)

    with pytest.raises(KnowledgeIndexError, match="contain text"):
        query_index("   ", database)
    with pytest.raises(KnowledgeIndexError, match=r"\[1, 100\]"):
        query_index("phase", database, limit=101)


def test_corrupt_database_reports_rebuild_instead_of_sqlite_error(tmp_path: Path) -> None:
    database = tmp_path / "graph.db"
    database.write_bytes(b"not a sqlite database")

    with pytest.raises(KnowledgeIndexError, match="corrupt or incomplete"):
        index_stats(database)
    with pytest.raises(KnowledgeIndexError, match="corrupt or incomplete"):
        query_index("phase", database)
