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


def _source_record(
    record_id: str,
    *,
    record_type: str,
    evidence_status: str,
) -> dict[str, object]:
    sources: list[dict[str, object]] = [
        {
            "source_id": "audit",
            "provenance_class": "local_audit_document",
            "locator": "Complete audit",
            "version": "git:test",
            "sha256": "a" * 64,
            "url": None,
            "repository_relative_path": f"docs/{record_id}.md",
        },
        {
            "source_id": "public",
            "provenance_class": "primary_public_source",
            "locator": "Pinned public bytes",
            "version": "git:test",
            "sha256": "b" * 64,
            "url": "https://example.test/source",
            "repository_relative_path": None,
        },
    ]
    finding_type = "source_reported_fact"
    finding_source_ids = ["public"]
    if record_type == "local_numeric_measurement":
        sources.extend(
            [
                {
                    "source_id": "numeric",
                    "provenance_class": "local_admitted_numeric",
                    "locator": "Admitted numeric fixture",
                    "version": "sha:test",
                    "sha256": "c" * 64,
                    "url": None,
                    "repository_relative_path": "artifacts/test.npz",
                },
                {
                    "source_id": "analysis",
                    "provenance_class": "local_analysis_code",
                    "locator": "Data-only analysis fixture",
                    "version": "git:test",
                    "sha256": "d" * 64,
                    "url": None,
                    "repository_relative_path": "scripts/test_analysis.py",
                },
            ]
        )
        finding_type = "local_measurement"
        finding_source_ids = ["numeric", "analysis"]
    return {
        "schema_version": 1,
        "record_id": record_id,
        "record_type": record_type,
        "title": f"Typed {record_type}",
        "evidence_status": evidence_status,
        "summary": "Reference audit result with searchable low-gait yaw seam terms.",
        "audit_document": {
            "repository_relative_path": f"docs/{record_id}.md",
            "sha256": "a" * 64,
        },
        "sources": sources,
        "findings": [
            {
                "finding_id": "result",
                "finding_type": finding_type,
                "title": "Low-gait yaw seam finding",
                "statement": "A bounded finding tied to exact source bytes.",
                "source_ids": finding_source_ids,
                "locator": "Audit table 1",
                "not_evidence_for": ["policy success"],
            }
        ],
        "not_evidence_for": ["policy success"],
    }


def _write_source_record(directory: Path, record: dict[str, object]) -> Path:
    path = directory / f"{record['record_id']}.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


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
    assert "source_records" not in built
    assert "source_findings" not in built
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


def test_repository_reference_audits_are_queryable_with_claim_limits(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    extractions = tmp_path / "extractions"
    extractions.mkdir()
    _write_record(extractions, _record("2600.00001", title="Closed-loop recovery"))
    database = tmp_path / "graph.db"

    stats = build_index(
        extractions,
        database,
        supplemental_records=repository_root / "research/evidence/reference_audits/records",
    )
    external = query_index("external low gait", database)
    native = query_index("basic walk native window", database)

    assert stats["source_records"] == 2
    assert stats["public_source_audits"] == 1
    assert stats["local_numeric_measurements"] == 1
    assert stats["source_findings"] == 12
    assert any(
        item["record_id"] == "source_record:g1_external_reference_audit_20260907"
        and item["kind"] == "public_source_audit"
        and "NOT_ADMITTED" in str(item["description"])
        and "policy evaluation" in str(item["description"])
        for item in external
    )
    assert any(
        item["record_id"] == "source_record:gmt_basic_walk_window_audit_20260907"
        and item["kind"] == "local_numeric_measurement"
        and "EXPLORATORY_NUMERIC_FITNESS_NOT_DYNAMICS" in str(item["description"])
        and "learned-task success" in str(item["description"])
        for item in native
    )

    connection = sqlite3.connect(database)
    supplemental_papers = connection.execute(
        """
        SELECT count(*) FROM papers
        WHERE id IN (
          'g1_external_reference_audit_20260907',
          'gmt_basic_walk_window_audit_20260907'
        )
        """
    ).fetchone()
    informing_edges = connection.execute(
        "SELECT count(*) FROM edges WHERE source LIKE 'source_record:%' AND relation='INFORMS'"
    ).fetchone()
    reporting_edges = connection.execute(
        "SELECT count(*) FROM edges WHERE source LIKE 'source_record:%' AND relation='REPORTS'"
    ).fetchone()
    connection.close()
    assert supplemental_papers == (0,)
    assert informing_edges == (0,)
    assert reporting_edges == (12,)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("unknown_record_type", "record_type is unsupported"),
        ("unknown_provenance", "provenance_class is unsupported"),
        ("wrong_status", "evidence_status must be"),
        ("missing_source", "references missing source provenance"),
        ("reported_without_public", "source-reported fact lacks public provenance"),
        ("measurement_without_local", "local measurement lacks local provenance"),
    ),
)
def test_supplemental_records_fail_closed_on_type_status_and_provenance(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    extractions = tmp_path / "extractions"
    extractions.mkdir()
    _write_record(extractions, _record("2600.00001", title="Closed-loop recovery"))
    supplemental = tmp_path / "supplemental"
    supplemental.mkdir()
    record = _source_record(
        "source_audit",
        record_type="public_source_audit",
        evidence_status="not_admitted",
    )
    finding = record["findings"][0]
    assert isinstance(finding, dict)
    if case == "unknown_record_type":
        record["record_type"] = "paper"
    elif case == "unknown_provenance":
        sources = record["sources"]
        assert isinstance(sources, list)
        public_source = sources[1]
        assert isinstance(public_source, dict)
        public_source["provenance_class"] = "unreviewed_web"
    elif case == "wrong_status":
        record["evidence_status"] = "admitted"
    elif case == "missing_source":
        finding["source_ids"] = ["missing"]
    elif case == "reported_without_public":
        finding["source_ids"] = ["audit"]
    else:
        record = _source_record(
            "source_audit",
            record_type="local_numeric_measurement",
            evidence_status="exploratory_numeric_fitness_not_dynamics",
        )
        finding = record["findings"][0]
        assert isinstance(finding, dict)
        finding["source_ids"] = ["public"]
    _write_source_record(supplemental, record)

    with pytest.raises(KnowledgeIndexError, match=message):
        build_index(extractions, tmp_path / "graph.db", supplemental_records=supplemental)


def test_supplemental_record_symlink_is_rejected(tmp_path: Path) -> None:
    extractions = tmp_path / "extractions"
    extractions.mkdir()
    _write_record(extractions, _record("2600.00001", title="Closed-loop recovery"))
    supplemental = tmp_path / "supplemental"
    supplemental.mkdir()
    target = tmp_path / "target.json"
    _write_source_record(
        tmp_path,
        _source_record(
            "target",
            record_type="public_source_audit",
            evidence_status="not_admitted",
        ),
    )
    (supplemental / "target.json").symlink_to(target)

    with pytest.raises(KnowledgeIndexError, match="bounded regular file"):
        build_index(extractions, tmp_path / "graph.db", supplemental_records=supplemental)


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
    assert all(item["title"] != "oscillating recovery transition" for item in active_results)
    assert any(item["title"] == "oscillating recovery transition" for item in inactive_results)


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
    assert all(json.loads(item[0]) == ["2600.00001:E1"] for item in edge_evidence)


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
