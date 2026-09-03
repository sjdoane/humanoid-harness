"""Build and query a provenance-bearing local research graph."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_DATABASE = Path("artifacts/knowledge/graph.db")
DEFAULT_EXTRACTIONS = Path("research/evidence/legacy_policy_harness_kg/extractions")

PROJECT_NODES = {
    "system:agentic_harness": (
        "system",
        "Agentic design harness",
        "Produces auditable oracle and task-reward candidates from evidence.",
    ),
    "knob:reference_oracle": (
        "project_knob",
        "Reference oracle",
        "Selects, sequences, transitions, recovers, and rejoins reference behavior.",
    ),
    "knob:task_reward": (
        "project_knob",
        "Task reward",
        "Defines the executable task objective while tracking reward stays fixed.",
    ),
    "system:training_adapter": (
        "system",
        "Policy-training adapter",
        "Consumes exact oracle and reward artifacts under a frozen MDP contract.",
    ),
    "system:protected_evaluator": (
        "system",
        "Protected evaluator",
        "Computes task, tracking, transition, recovery, safety, and stability evidence.",
    ),
    "artifact:trajectory_trace": (
        "artifact",
        "Trajectory trace",
        "Canonical synchronized robot, controller, oracle, and task signals.",
    ),
    "artifact:decision_record": (
        "artifact",
        "Decision record",
        "Binds evidence, missing information, candidate patch, prediction, and falsifier.",
    ),
    "input:human_steering": (
        "input",
        "Human steering",
        "A compact correction translated into one declared research artifact.",
    ),
}

PROJECT_EDGES = (
    ("system:agentic_harness", "PRODUCES", "knob:reference_oracle"),
    ("system:agentic_harness", "PRODUCES", "knob:task_reward"),
    ("knob:reference_oracle", "CONFIGURES", "system:training_adapter"),
    ("knob:task_reward", "CONFIGURES", "system:training_adapter"),
    ("system:training_adapter", "EMITS", "artifact:trajectory_trace"),
    ("artifact:trajectory_trace", "EVALUATED_BY", "system:protected_evaluator"),
    ("system:protected_evaluator", "INFORMS", "artifact:decision_record"),
    ("artifact:decision_record", "FEEDBACK_TO", "system:agentic_harness"),
    ("input:human_steering", "STEERS", "system:agentic_harness"),
)

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,511}$")


class KnowledgeIndexError(ValueError):
    """The research corpus or generated index violates its contract."""


def _compact(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _required_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KnowledgeIndexError(f"{field} must be an object")
    return value


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KnowledgeIndexError(f"{field} must be non-empty text")
    return value.strip()


def _stable_id(value: object, *, field: str) -> str:
    resolved = _required_text(value, field=field)
    if not _SAFE_ID.fullmatch(resolved):
        raise KnowledgeIndexError(f"{field} must be a stable identifier")
    return resolved


def _relation_endpoint(value: object, *, paper_id: str, field: str) -> tuple[str, str]:
    label = _required_text(value, field=field)
    if label in PROJECT_NODES or label == paper_id or label.startswith(f"{paper_id}:"):
        return label, label
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()[:12]
    return f"concept:{paper_id}:{digest}", label


def _records(extractions_dir: Path) -> list[tuple[Path, bytes, Mapping[str, Any]]]:
    if not extractions_dir.is_dir():
        raise KnowledgeIndexError(f"extraction directory not found: {extractions_dir}")
    paths = sorted(path for path in extractions_dir.glob("*.json") if not path.name.startswith("_"))
    if not paths:
        raise KnowledgeIndexError(f"no extraction records found: {extractions_dir}")

    records: list[tuple[Path, bytes, Mapping[str, Any]]] = []
    paper_ids: set[str] = set()
    for path in paths:
        try:
            source_bytes = path.read_bytes()
            raw = json.loads(source_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise KnowledgeIndexError(f"cannot read extraction: {path}") from exc
        record = _required_mapping(raw, field=path.name)
        paper = _required_mapping(record.get("paper"), field=f"{path.name}.paper")
        paper_id = _stable_id(paper.get("id"), field=f"{path.name}.paper.id")
        if paper_id in paper_ids:
            raise KnowledgeIndexError(f"duplicate paper id: {paper_id}")
        paper_ids.add(paper_id)
        records.append((path, source_bytes, record))
    return records


def _corpus_sha256(
    records: Iterable[tuple[Path, bytes, Mapping[str, Any]]],
) -> str:
    digest = hashlib.sha256()
    for path, source_bytes, _ in records:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_bytes)
        digest.update(b"\0")
    return digest.hexdigest()


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE metadata (
          key TEXT PRIMARY KEY,
          value TEXT NOT NULL
        );
        CREATE TABLE papers (
          id TEXT PRIMARY KEY,
          title TEXT NOT NULL,
          year INTEGER NOT NULL,
          primary_url TEXT NOT NULL,
          source_version TEXT NOT NULL,
          publication_status TEXT NOT NULL,
          tier TEXT NOT NULL,
          knobs_json TEXT NOT NULL,
          why_admitted TEXT NOT NULL,
          source_sha256 TEXT NOT NULL
        );
        CREATE TABLE nodes (
          id TEXT PRIMARY KEY,
          type TEXT NOT NULL,
          paper_id TEXT REFERENCES papers(id),
          name TEXT NOT NULL,
          description TEXT NOT NULL,
          payload_json TEXT NOT NULL
        );
        CREATE TABLE edges (
          id INTEGER PRIMARY KEY,
          source TEXT NOT NULL REFERENCES nodes(id),
          relation TEXT NOT NULL,
          target TEXT NOT NULL REFERENCES nodes(id),
          evidence_ids_json TEXT NOT NULL,
          provenance TEXT NOT NULL
        );
        CREATE TABLE evidence (
          id TEXT PRIMARY KEY,
          paper_id TEXT NOT NULL REFERENCES papers(id),
          source_url TEXT NOT NULL,
          locator TEXT NOT NULL,
          claim TEXT NOT NULL,
          evidence_type TEXT NOT NULL
        );
        CREATE TABLE search_items (
          kind TEXT NOT NULL,
          record_id TEXT NOT NULL,
          paper_id TEXT,
          title TEXT NOT NULL,
          body TEXT NOT NULL,
          source_url TEXT
        );
        CREATE INDEX idx_nodes_type_paper ON nodes(type, paper_id);
        CREATE INDEX idx_edges_source_relation ON edges(source, relation);
        CREATE INDEX idx_edges_target_relation ON edges(target, relation);
        CREATE INDEX idx_search_kind_paper ON search_items(kind, paper_id);
        """
    )


def _insert_node(
    connection: sqlite3.Connection,
    *,
    node_id: str,
    kind: str,
    name: str,
    description: str,
    paper_id: str | None,
    payload: Mapping[str, Any],
    source_url: str | None = None,
) -> None:
    connection.execute(
        "INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?)",
        (node_id, kind, paper_id, name, description, _compact(payload)),
    )
    connection.execute(
        "INSERT INTO search_items VALUES (?, ?, ?, ?, ?, ?)",
        (kind, node_id, paper_id, name, description, source_url),
    )


def _paper_knob_id(knob: str) -> str | None:
    aliases = {
        "reference_composition": "knob:reference_oracle",
        "task_reward": "knob:task_reward",
    }
    return aliases.get(knob)


def _canonical_evidence_id(paper_id: str, value: object, *, field: str) -> str:
    identifier = _stable_id(value, field=field)
    prefix = f"{paper_id}:"
    return identifier if identifier.startswith(prefix) else f"{prefix}{identifier}"


def _evidence_ids(paper_id: str, values: object) -> str:
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise KnowledgeIndexError(f"{paper_id}.evidence_ids must be a text list")
    resolved = [
        _canonical_evidence_id(
            paper_id,
            value,
            field=f"{paper_id}.evidence_ids",
        )
        for value in values
    ]
    return _compact(sorted(resolved))


def _build_database(
    database: Path,
    records: list[tuple[Path, bytes, Mapping[str, Any]]],
    *,
    corpus_sha256: str,
) -> None:
    connection = sqlite3.connect(database)
    try:
        _create_schema(connection)
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)",
            (
                ("schema_version", str(SCHEMA_VERSION)),
                ("corpus_sha256", corpus_sha256),
                ("paper_count", str(len(records))),
            ),
        )

        for node_id, (kind, name, description) in PROJECT_NODES.items():
            _insert_node(
                connection,
                node_id=node_id,
                kind=kind,
                name=name,
                description=description,
                paper_id=None,
                payload={"provenance": "docs/PROJECT_CHARTER.md"},
            )

        pending_edges: list[tuple[str, str, str, str, str]] = [
            (source, relation, target, "[]", "docs/PROJECT_CHARTER.md")
            for source, relation, target in PROJECT_EDGES
        ]
        concept_labels: dict[str, str] = {}
        concept_papers: dict[str, str] = {}
        concept_sources: dict[str, str] = {}

        for path, source_bytes, record in records:
            paper = _required_mapping(record.get("paper"), field=f"{path.name}.paper")
            relevance = _required_mapping(
                record.get("decision_relevance"),
                field=f"{path.name}.decision_relevance",
            )
            paper_id = _stable_id(paper.get("id"), field=f"{path.name}.paper.id")
            title = _required_text(paper.get("title"), field=f"{paper_id}.title")
            primary_url = _required_text(paper.get("primary_url"), field=f"{paper_id}.primary_url")
            source_version = _required_text(
                paper.get("source_version"), field=f"{paper_id}.source_version"
            )
            why_admitted = _required_text(
                relevance.get("why_admitted"), field=f"{paper_id}.why_admitted"
            )
            year = paper.get("year")
            if not isinstance(year, int) or isinstance(year, bool):
                raise KnowledgeIndexError(f"{paper_id}.year must be an integer")
            status = str(paper.get("publication_status", "active"))
            tier = _required_text(relevance.get("tier"), field=f"{paper_id}.tier")
            knobs = relevance.get("knobs", [])
            if not isinstance(knobs, list) or not all(isinstance(item, str) for item in knobs):
                raise KnowledgeIndexError(f"{paper_id}.knobs must be a text list")
            source_sha256 = hashlib.sha256(source_bytes).hexdigest()
            connection.execute(
                "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    paper_id,
                    title,
                    year,
                    primary_url,
                    source_version,
                    status,
                    tier,
                    _compact(sorted(knobs)),
                    why_admitted,
                    source_sha256,
                ),
            )
            _insert_node(
                connection,
                node_id=paper_id,
                kind="paper",
                name=title,
                description=why_admitted,
                paper_id=paper_id,
                payload=dict(paper),
                source_url=primary_url,
            )

            for field, kind in (
                ("mechanisms", "mechanism"),
                ("failure_modes", "failure_mode"),
                ("evaluation", "evaluation"),
            ):
                items = record.get(field, [])
                if not isinstance(items, list):
                    raise KnowledgeIndexError(f"{paper_id}.{field} must be a list")
                for item_raw in items:
                    item = _required_mapping(item_raw, field=f"{paper_id}.{field}")
                    local_id = _stable_id(item.get("id"), field=f"{paper_id}.{field}.id")
                    node_id = local_id
                    _insert_node(
                        connection,
                        node_id=node_id,
                        kind=kind,
                        name=_required_text(item.get("name"), field=f"{node_id}.name"),
                        description=_required_text(
                            item.get("description"), field=f"{node_id}.description"
                        ),
                        paper_id=paper_id,
                        payload=dict(item),
                        source_url=primary_url,
                    )
                    pending_edges.append(
                        (
                            paper_id,
                            "REPORTS",
                            node_id,
                            _evidence_ids(paper_id, item.get("evidence_ids", [])),
                            f"extraction:{path.name}",
                        )
                    )

            for index, parameter_raw in enumerate(record.get("parameters", []), start=1):
                parameter = _required_mapping(
                    parameter_raw, field=f"{paper_id}.parameters[{index}]"
                )
                name = _required_text(parameter.get("name"), field=f"{paper_id}.parameter.name")
                node_id = f"{paper_id}:parameter:{index}"
                _insert_node(
                    connection,
                    node_id=node_id,
                    kind="parameter",
                    name=name,
                    description=_required_text(
                        parameter.get("context"), field=f"{node_id}.context"
                    ),
                    paper_id=paper_id,
                    payload=dict(parameter),
                    source_url=primary_url,
                )
                pending_edges.append(
                    (
                        paper_id,
                        "REPORTS",
                        node_id,
                        _evidence_ids(paper_id, parameter.get("evidence_ids", [])),
                        f"extraction:{path.name}",
                    )
                )

            for evidence_raw in record.get("evidence", []):
                evidence = _required_mapping(evidence_raw, field=f"{paper_id}.evidence")
                local_id = _stable_id(evidence.get("id"), field=f"{paper_id}.evidence.id")
                evidence_id = _canonical_evidence_id(
                    paper_id,
                    local_id,
                    field=f"{paper_id}.evidence.id",
                )
                source_url = _required_text(
                    evidence.get("source_url"), field=f"{evidence_id}.source_url"
                )
                locator = _required_text(evidence.get("locator"), field=f"{evidence_id}.locator")
                claim = _required_text(evidence.get("claim"), field=f"{evidence_id}.claim")
                evidence_type = _required_text(
                    evidence.get("evidence_type"), field=f"{evidence_id}.evidence_type"
                )
                connection.execute(
                    "INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?)",
                    (evidence_id, paper_id, source_url, locator, claim, evidence_type),
                )
                connection.execute(
                    "INSERT INTO search_items VALUES (?, ?, ?, ?, ?, ?)",
                    ("evidence", evidence_id, paper_id, locator, claim, source_url),
                )

            for knob in sorted(knobs):
                target = _paper_knob_id(knob)
                if target is not None:
                    relation = "INFORMS" if status == "active" else "QUARANTINED_FOR"
                    pending_edges.append(
                        (
                            paper_id,
                            relation,
                            target,
                            "[]",
                            f"curation:{path.name}",
                        )
                    )

            relations = record.get("relations", [])
            if not isinstance(relations, list):
                raise KnowledgeIndexError(f"{paper_id}.relations must be a list")
            for relation_raw in relations:
                relation = _required_mapping(relation_raw, field=f"{paper_id}.relations")
                source, source_label = _relation_endpoint(
                    relation.get("source"),
                    paper_id=paper_id,
                    field=f"{paper_id}.relation.source",
                )
                target, target_label = _relation_endpoint(
                    relation.get("target"),
                    paper_id=paper_id,
                    field=f"{paper_id}.relation.target",
                )
                for endpoint, label in ((source, source_label), (target, target_label)):
                    prior_owner = concept_papers.get(endpoint)
                    if (
                        prior_owner is not None
                        and prior_owner != paper_id
                        and endpoint not in PROJECT_NODES
                    ):
                        raise KnowledgeIndexError(f"concept provenance is ambiguous: {endpoint}")
                    concept_labels.setdefault(endpoint, label)
                    concept_papers.setdefault(endpoint, paper_id)
                    concept_sources.setdefault(endpoint, primary_url)
                pending_edges.append(
                    (
                        source,
                        _stable_id(relation.get("relation"), field=f"{paper_id}.relation.relation"),
                        target,
                        _evidence_ids(paper_id, relation.get("evidence_ids", [])),
                        f"extraction:{path.name}",
                    )
                )

        known_nodes = {row[0] for row in connection.execute("SELECT id FROM nodes").fetchall()}
        unresolved = sorted(
            {
                endpoint
                for source, _, target, _, _ in pending_edges
                for endpoint in (source, target)
                if endpoint not in known_nodes
            }
        )
        for node_id in unresolved:
            _insert_node(
                connection,
                node_id=node_id,
                kind="concept",
                name=concept_labels.get(node_id, node_id.replace(":", " / ").replace("_", " ")),
                description="Concept endpoint declared by a source extraction.",
                paper_id=concept_papers.get(node_id),
                payload={"provenance": "source_extraction_relation"},
                source_url=concept_sources.get(node_id),
            )
            known_nodes.add(node_id)
        known_evidence = {
            row[0] for row in connection.execute("SELECT id FROM evidence").fetchall()
        }
        for source, relation, target, evidence_ids, provenance in pending_edges:
            if source not in known_nodes or target not in known_nodes:
                raise KnowledgeIndexError(f"edge endpoint is missing: {source} {relation} {target}")
            missing_evidence = sorted(set(json.loads(evidence_ids)) - known_evidence)
            if missing_evidence:
                raise KnowledgeIndexError(
                    "edge references missing evidence: "
                    f"{source} {relation} {target}: {missing_evidence!r}"
                )
            connection.execute(
                """
                INSERT INTO edges(source, relation, target, evidence_ids_json, provenance)
                VALUES (?, ?, ?, ?, ?)
                """,
                (source, relation, target, evidence_ids, provenance),
            )
        connection.commit()
    finally:
        connection.close()


def build_index(
    extractions_dir: Path = DEFAULT_EXTRACTIONS,
    database: Path = DEFAULT_DATABASE,
) -> dict[str, object]:
    """Build the index atomically and return its verified coverage."""

    extractions_dir = extractions_dir.resolve()
    database = database.resolve()
    records = _records(extractions_dir)
    corpus_sha256 = _corpus_sha256(records)
    database.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="knowledge-build-", dir=database.parent) as directory:
        staged = Path(directory) / "graph.db"
        try:
            _build_database(staged, records, corpus_sha256=corpus_sha256)
        except sqlite3.DatabaseError as exc:
            raise KnowledgeIndexError(f"research graph integrity failure: {exc}") from exc
        os.replace(staged, database)
    return index_stats(database)


def _connect(database: Path) -> sqlite3.Connection:
    database = database.resolve()
    if not database.is_file():
        raise KnowledgeIndexError(
            f"research index not found: {database}; run 'humanoid-harness research build'"
        )
    try:
        connection = sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise KnowledgeIndexError(f"cannot open research index: {database}") from exc
    connection.row_factory = sqlite3.Row
    try:
        version_row = connection.execute(
            "SELECT value FROM metadata WHERE key = 'schema_version'"
        ).fetchone()
        valid_version = version_row is not None and int(version_row["value"]) == SCHEMA_VERSION
    except (sqlite3.Error, TypeError, ValueError) as exc:
        connection.close()
        raise KnowledgeIndexError("research index is corrupt or incomplete; rebuild it") from exc
    if not valid_version:
        connection.close()
        raise KnowledgeIndexError("research index schema is unsupported; rebuild it")
    return connection


def index_stats(database: Path = DEFAULT_DATABASE) -> dict[str, object]:
    connection = _connect(database)
    try:
        try:
            counts = connection.execute(
                """
                SELECT
                  (SELECT count(*) FROM papers) AS papers,
                  (SELECT count(*) FROM papers WHERE publication_status='active') AS active_papers,
                  (SELECT count(*) FROM nodes) AS nodes,
                  (SELECT count(*) FROM edges) AS edges,
                  (SELECT count(*) FROM evidence) AS evidence_items,
                  (SELECT count(*) FROM nodes WHERE type='mechanism') AS mechanisms,
                  (SELECT count(*) FROM nodes WHERE type='failure_mode') AS failure_modes,
                  (SELECT count(*) FROM nodes WHERE type='evaluation') AS evaluations
                """
            ).fetchone()
            metadata = {
                row["key"]: row["value"]
                for row in connection.execute("SELECT key, value FROM metadata")
            }
            result = dict(counts) if counts is not None else {}
            result.update(
                {
                    "schema_version": int(metadata["schema_version"]),
                    "corpus_sha256": metadata["corpus_sha256"],
                    "database": str(database.resolve()),
                }
            )
            return result
        except (KeyError, sqlite3.Error, TypeError, ValueError) as exc:
            raise KnowledgeIndexError(
                "research index is corrupt or incomplete; rebuild it"
            ) from exc
    finally:
        connection.close()


def query_index(
    query: str,
    database: Path = DEFAULT_DATABASE,
    *,
    limit: int = 20,
    include_inactive: bool = False,
) -> list[dict[str, object]]:
    """Return ranked graph records with primary-source links."""

    if not isinstance(limit, int) or isinstance(limit, bool) or not (1 <= limit <= 100):
        raise KnowledgeIndexError("limit must be in [1, 100]")
    phrase = " ".join(query.lower().split())
    if not phrase:
        raise KnowledgeIndexError("query must contain text")
    tokens = tuple(dict.fromkeys(re.findall(r"[a-z0-9_.:/-]+", phrase)))
    connection = _connect(database)
    try:
        sql = """
          SELECT s.kind, s.record_id, s.paper_id, s.title, s.body, s.source_url,
                 p.publication_status
          FROM search_items s
          LEFT JOIN papers p ON p.id = s.paper_id
        """
        if not include_inactive:
            sql += " WHERE p.id IS NULL OR p.publication_status = 'active'"
        try:
            rows = connection.execute(sql).fetchall()
        except sqlite3.Error as exc:
            raise KnowledgeIndexError(
                "research index is corrupt or incomplete; rebuild it"
            ) from exc
    finally:
        connection.close()

    ranked: list[tuple[int, str, dict[str, object]]] = []
    for row in rows:
        haystack = f"{row['title']} {row['body']}".lower()
        token_hits = sum(token in haystack for token in tokens)
        if token_hits == 0:
            continue
        score = token_hits * 10
        if phrase in haystack:
            score += 25
        if phrase in str(row["title"]).lower():
            score += 15
        result = {
            "kind": row["kind"],
            "record_id": row["record_id"],
            "paper_id": row["paper_id"],
            "title": row["title"],
            "description": row["body"],
            "source_url": row["source_url"],
            "score": score,
        }
        ranked.append((score, str(row["record_id"]), result))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in ranked[:limit]]
