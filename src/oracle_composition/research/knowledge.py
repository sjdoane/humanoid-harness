"""Build and query a provenance-bearing local research graph."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

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
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_RECORD_TYPES = {
    "public_source_audit": "not_admitted",
    "local_numeric_measurement": "exploratory_numeric_fitness_not_dynamics",
}
_FINDING_TYPES = {
    "source_reported_fact",
    "local_measurement",
    "project_interpretation",
    "limitation",
}
_PUBLIC_PROVENANCE = {"primary_public_source", "official_public_code", "publisher_metadata"}
_LOCAL_PROVENANCE = {
    "local_admitted_numeric",
    "local_analysis_code",
    "local_audit_document",
}
_SOURCE_PROVENANCE = _PUBLIC_PROVENANCE | _LOCAL_PROVENANCE
_MAX_SOURCE_RECORDS = 32
_MAX_SOURCE_RECORD_BYTES = 256 * 1024
_MAX_SOURCES_PER_RECORD = 64
_MAX_FINDINGS_PER_RECORD = 64


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


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    observed = set(value)
    if observed != expected:
        raise KnowledgeIndexError(
            f"{field} keys differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _sha256_text(value: object, *, field: str) -> str:
    resolved = _required_text(value, field=field)
    if not _SHA256.fullmatch(resolved):
        raise KnowledgeIndexError(f"{field} must be a lowercase SHA-256")
    return resolved


def _text_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise KnowledgeIndexError(f"{field} must be a non-empty-text list")
    resolved = [item.strip() for item in value]
    if not resolved:
        raise KnowledgeIndexError(f"{field} cannot be empty")
    if len(set(resolved)) != len(resolved):
        raise KnowledgeIndexError(f"{field} cannot contain duplicates")
    return resolved


def _repository_path(value: object, *, field: str) -> str:
    resolved = _required_text(value, field=field)
    path = PurePosixPath(resolved)
    if (
        path.is_absolute()
        or "\\" in resolved
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise KnowledgeIndexError(f"{field} must be a safe repository-relative path")
    return resolved


def _https_url(value: object, *, field: str) -> str:
    resolved = _required_text(value, field=field)
    parsed = urlsplit(resolved)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise KnowledgeIndexError(f"{field} must be an HTTPS URL without credentials")
    return resolved


def _optional_source_location(value: object, *, field: str, kind: str) -> str | None:
    if value is None:
        return None
    if kind == "url":
        return _https_url(value, field=field)
    return _repository_path(value, field=field)


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


def _validate_source_record(record: Mapping[str, Any], *, path: Path) -> None:
    field = path.name
    _exact_keys(
        record,
        {
            "schema_version",
            "record_id",
            "record_type",
            "title",
            "evidence_status",
            "summary",
            "audit_document",
            "sources",
            "findings",
            "not_evidence_for",
        },
        field=field,
    )
    if record.get("schema_version") != 1 or isinstance(record.get("schema_version"), bool):
        raise KnowledgeIndexError(f"{field}.schema_version must equal 1")
    record_id = _stable_id(record.get("record_id"), field=f"{field}.record_id")
    if path.stem != record_id:
        raise KnowledgeIndexError(f"{field}.record_id must equal its filename stem")
    record_type = _required_text(record.get("record_type"), field=f"{field}.record_type")
    try:
        expected_status = _SOURCE_RECORD_TYPES[record_type]
    except KeyError as exc:
        raise KnowledgeIndexError(f"{field}.record_type is unsupported") from exc
    if record.get("evidence_status") != expected_status:
        raise KnowledgeIndexError(
            f"{field}.evidence_status must be {expected_status!r} for {record_type}"
        )
    _required_text(record.get("title"), field=f"{field}.title")
    _required_text(record.get("summary"), field=f"{field}.summary")
    _text_list(record.get("not_evidence_for"), field=f"{field}.not_evidence_for")

    audit_document = _required_mapping(
        record.get("audit_document"), field=f"{field}.audit_document"
    )
    _exact_keys(
        audit_document,
        {"repository_relative_path", "sha256"},
        field=f"{field}.audit_document",
    )
    audit_path = _repository_path(
        audit_document.get("repository_relative_path"),
        field=f"{field}.audit_document.repository_relative_path",
    )
    audit_sha256 = _sha256_text(
        audit_document.get("sha256"), field=f"{field}.audit_document.sha256"
    )

    sources_raw = record.get("sources")
    if (
        not isinstance(sources_raw, list)
        or not sources_raw
        or len(sources_raw) > _MAX_SOURCES_PER_RECORD
    ):
        raise KnowledgeIndexError(f"{field}.sources must contain 1-{_MAX_SOURCES_PER_RECORD} items")
    sources: dict[str, Mapping[str, Any]] = {}
    for index, source_raw in enumerate(sources_raw):
        source = _required_mapping(source_raw, field=f"{field}.sources[{index}]")
        _exact_keys(
            source,
            {
                "source_id",
                "provenance_class",
                "locator",
                "version",
                "sha256",
                "url",
                "repository_relative_path",
            },
            field=f"{field}.sources[{index}]",
        )
        source_id = _stable_id(
            source.get("source_id"), field=f"{field}.sources[{index}].source_id"
        )
        if source_id in sources:
            raise KnowledgeIndexError(f"{field} has duplicate source_id {source_id!r}")
        provenance_class = _required_text(
            source.get("provenance_class"),
            field=f"{field}.sources[{index}].provenance_class",
        )
        if provenance_class not in _SOURCE_PROVENANCE:
            raise KnowledgeIndexError(f"{field}.{source_id}.provenance_class is unsupported")
        url = _optional_source_location(
            source.get("url"), field=f"{field}.{source_id}.url", kind="url"
        )
        repository_path = _optional_source_location(
            source.get("repository_relative_path"),
            field=f"{field}.{source_id}.repository_relative_path",
            kind="path",
        )
        if url is None and repository_path is None:
            raise KnowledgeIndexError(f"{field}.{source_id} must provide a URL or repository path")
        if provenance_class in _PUBLIC_PROVENANCE and url is None:
            raise KnowledgeIndexError(f"{field}.{source_id} public provenance requires a URL")
        if provenance_class in _LOCAL_PROVENANCE and repository_path is None:
            raise KnowledgeIndexError(
                f"{field}.{source_id} local provenance requires a repository path"
            )
        _required_text(source.get("locator"), field=f"{field}.{source_id}.locator")
        _required_text(source.get("version"), field=f"{field}.{source_id}.version")
        _sha256_text(source.get("sha256"), field=f"{field}.{source_id}.sha256")
        sources[source_id] = source

    document_sources = [
        source
        for source in sources.values()
        if source.get("provenance_class") == "local_audit_document"
        and source.get("repository_relative_path") == audit_path
        and source.get("sha256") == audit_sha256
    ]
    if len(document_sources) != 1:
        raise KnowledgeIndexError(f"{field}.audit_document must match one local source pin")
    if record_type == "public_source_audit" and not any(
        source.get("provenance_class") in _PUBLIC_PROVENANCE for source in sources.values()
    ):
        raise KnowledgeIndexError(f"{field} public audit has no public-source provenance")
    if record_type == "local_numeric_measurement" and not {
        "local_admitted_numeric",
        "local_analysis_code",
    }.issubset({str(source.get("provenance_class")) for source in sources.values()}):
        raise KnowledgeIndexError(
            f"{field} numeric measurement requires admitted numeric and analysis-code provenance"
        )

    findings_raw = record.get("findings")
    if (
        not isinstance(findings_raw, list)
        or not findings_raw
        or len(findings_raw) > _MAX_FINDINGS_PER_RECORD
    ):
        raise KnowledgeIndexError(
            f"{field}.findings must contain 1-{_MAX_FINDINGS_PER_RECORD} items"
        )
    finding_ids: set[str] = set()
    for index, finding_raw in enumerate(findings_raw):
        finding = _required_mapping(finding_raw, field=f"{field}.findings[{index}]")
        _exact_keys(
            finding,
            {
                "finding_id",
                "finding_type",
                "title",
                "statement",
                "source_ids",
                "locator",
                "not_evidence_for",
            },
            field=f"{field}.findings[{index}]",
        )
        finding_id = _stable_id(
            finding.get("finding_id"), field=f"{field}.findings[{index}].finding_id"
        )
        if finding_id in finding_ids:
            raise KnowledgeIndexError(f"{field} has duplicate finding_id {finding_id!r}")
        finding_ids.add(finding_id)
        finding_type = _required_text(
            finding.get("finding_type"), field=f"{field}.{finding_id}.finding_type"
        )
        if finding_type not in _FINDING_TYPES:
            raise KnowledgeIndexError(f"{field}.{finding_id}.finding_type is unsupported")
        _required_text(finding.get("title"), field=f"{field}.{finding_id}.title")
        _required_text(finding.get("statement"), field=f"{field}.{finding_id}.statement")
        _required_text(finding.get("locator"), field=f"{field}.{finding_id}.locator")
        _text_list(
            finding.get("not_evidence_for"), field=f"{field}.{finding_id}.not_evidence_for"
        )
        source_ids = _text_list(
            finding.get("source_ids"), field=f"{field}.{finding_id}.source_ids"
        )
        missing = sorted(set(source_ids) - set(sources))
        if missing:
            raise KnowledgeIndexError(
                f"{field}.{finding_id} references missing source provenance: {missing}"
            )
        provenance = {str(sources[source_id].get("provenance_class")) for source_id in source_ids}
        if finding_type == "source_reported_fact" and not provenance.intersection(
            _PUBLIC_PROVENANCE
        ):
            raise KnowledgeIndexError(
                f"{field}.{finding_id} source-reported fact lacks public provenance"
            )
        if finding_type == "local_measurement" and not provenance.intersection(
            _LOCAL_PROVENANCE
        ):
            raise KnowledgeIndexError(
                f"{field}.{finding_id} local measurement lacks local provenance"
            )


def _source_records(directory: Path) -> list[tuple[Path, bytes, Mapping[str, Any]]]:
    if directory.is_symlink() or not directory.is_dir():
        raise KnowledgeIndexError(f"supplemental source-record directory not found: {directory}")
    paths = sorted(path for path in directory.glob("*.json") if not path.name.startswith("_"))
    if not paths or len(paths) > _MAX_SOURCE_RECORDS:
        raise KnowledgeIndexError(
            f"supplemental source records must contain 1-{_MAX_SOURCE_RECORDS} JSON files"
        )
    records: list[tuple[Path, bytes, Mapping[str, Any]]] = []
    record_ids: set[str] = set()
    for path in paths:
        try:
            metadata = path.lstat()
        except OSError as exc:
            raise KnowledgeIndexError(f"cannot inspect source record: {path}") from exc
        if not stat.S_ISREG(metadata.st_mode) or not (0 < metadata.st_size <= _MAX_SOURCE_RECORD_BYTES):
            raise KnowledgeIndexError(f"source record must be a bounded regular file: {path}")
        try:
            with path.open("rb") as handle:
                source_bytes = handle.read(_MAX_SOURCE_RECORD_BYTES + 1)
        except OSError as exc:
            raise KnowledgeIndexError(f"cannot read source record: {path}") from exc
        if len(source_bytes) != metadata.st_size:
            raise KnowledgeIndexError(f"source record changed while reading: {path}")
        try:
            record = _required_mapping(json.loads(source_bytes), field=path.name)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise KnowledgeIndexError(f"cannot read source record: {path}") from exc
        _validate_source_record(record, path=path)
        record_id = str(record["record_id"])
        if record_id in record_ids:
            raise KnowledgeIndexError(f"duplicate source record id: {record_id}")
        record_ids.add(record_id)
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
    source_records: list[tuple[Path, bytes, Mapping[str, Any]]] | None,
    source_records_sha256: str | None,
) -> None:
    connection = sqlite3.connect(database)
    try:
        _create_schema(connection)
        metadata = [
            ("schema_version", str(SCHEMA_VERSION)),
            ("corpus_sha256", corpus_sha256),
            ("paper_count", str(len(records))),
        ]
        if source_records is not None:
            if source_records_sha256 is None:
                raise KnowledgeIndexError("supplemental source-record digest is missing")
            metadata.extend(
                (
                    ("source_record_count", str(len(source_records))),
                    ("source_records_sha256", source_records_sha256),
                )
            )
        connection.executemany("INSERT INTO metadata VALUES (?, ?)", metadata)

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

        for path, source_bytes, record in source_records or ():
            record_id = str(record["record_id"])
            node_id = f"source_record:{record_id}"
            record_type = str(record["record_type"])
            evidence_status = str(record["evidence_status"])
            title = str(record["title"])
            summary = str(record["summary"])
            not_evidence_for = [str(value) for value in record["not_evidence_for"]]
            sources = {str(source["source_id"]): source for source in record["sources"]}
            primary_url = next(
                (str(source["url"]) for source in sources.values() if source["url"] is not None),
                None,
            )
            provenance = (
                f"supplemental_source_record:{path.name}:"
                f"{hashlib.sha256(source_bytes).hexdigest()}"
            )
            boundary = "; ".join(not_evidence_for)
            audit_document = record["audit_document"]
            description = (
                f"Evidence status: {evidence_status.upper()}. {summary} "
                f"Audit document: {audit_document['repository_relative_path']} "
                f"(SHA-256 {audit_document['sha256']}). "
                f"Not evidence for: {boundary}."
            )
            _insert_node(
                connection,
                node_id=node_id,
                kind=record_type,
                name=title,
                description=description,
                paper_id=None,
                payload={**dict(record), "record_sha256": hashlib.sha256(source_bytes).hexdigest()},
                source_url=primary_url,
            )
            for finding in record["findings"]:
                finding_id = str(finding["finding_id"])
                finding_node_id = f"{node_id}:{finding_id}"
                finding_sources = [sources[str(source_id)] for source_id in finding["source_ids"]]
                finding_url = next(
                    (
                        str(source["url"])
                        for source in finding_sources
                        if source["url"] is not None
                    ),
                    primary_url,
                )
                finding_boundary = "; ".join(str(value) for value in finding["not_evidence_for"])
                _insert_node(
                    connection,
                    node_id=finding_node_id,
                    kind=str(finding["finding_type"]),
                    name=str(finding["title"]),
                    description=(
                        f"Evidence status: {evidence_status.upper()}. "
                        f"{finding['statement']} Not evidence for: {finding_boundary}."
                    ),
                    paper_id=None,
                    payload={**dict(finding), "record_id": record_id, "provenance": provenance},
                    source_url=finding_url,
                )
                pending_edges.append((node_id, "REPORTS", finding_node_id, "[]", provenance))

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
    *,
    supplemental_records: Path | None = None,
) -> dict[str, object]:
    """Build the index atomically and return its verified coverage."""

    extractions_dir = extractions_dir.resolve()
    database = database.resolve()
    records = _records(extractions_dir)
    corpus_sha256 = _corpus_sha256(records)
    source_records = None
    source_records_sha256 = None
    if supplemental_records is not None:
        source_records = _source_records(supplemental_records.absolute())
        source_records_sha256 = _corpus_sha256(source_records)
    database.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="knowledge-build-", dir=database.parent) as directory:
        staged = Path(directory) / "graph.db"
        try:
            _build_database(
                staged,
                records,
                corpus_sha256=corpus_sha256,
                source_records=source_records,
                source_records_sha256=source_records_sha256,
            )
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
            if "source_record_count" in metadata:
                supplemental_counts = connection.execute(
                    """
                    SELECT
                      (SELECT count(*) FROM nodes
                         WHERE type IN ('public_source_audit', 'local_numeric_measurement'))
                        AS source_records,
                      (SELECT count(*) FROM nodes WHERE type='public_source_audit')
                        AS public_source_audits,
                      (SELECT count(*) FROM nodes WHERE type='local_numeric_measurement')
                        AS local_numeric_measurements,
                      (SELECT count(*) FROM nodes
                         WHERE type IN (
                           'source_reported_fact', 'local_measurement',
                           'project_interpretation', 'limitation'
                         )) AS source_findings
                    """
                ).fetchone()
                if supplemental_counts is None:
                    raise KnowledgeIndexError("supplemental source-record stats are missing")
                result.update(dict(supplemental_counts))
                result["source_records_sha256"] = metadata["source_records_sha256"]
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
