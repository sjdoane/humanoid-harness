#!/usr/bin/env python3
"""Build deterministic JSONL and SQLite views from paper extractions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
EXTRACTIONS = ROOT / "extractions"
GRAPH = ROOT / "graph"
KNOB_DEFINITIONS = {
    "reference_composition": "Select, sequence, guard, recover, or otherwise compose closed-loop reference behavior.",
    "task_reward": "Create or revise the executable task-specific reward while tracking reward remains fixed.",
    "human_steering": "Translate compact human feedback into one of the two permitted harness artifacts.",
    "evaluation": "Measure task, tracking, safety, stability, compute, and uncertainty independently of training return.",
    "fixed_trainer": "Evidence about the controller/trainer contract or its reachable-behavior boundary; not an allowed optimization knob.",
}
PROJECT_COMPONENTS = {
    "system:sam_harness": ("Sam's policy harness", "Iteratively proposes only an executable reference oracle and task reward."),
    "artifact:reference_dataset": ("Reference dataset", "Versioned input motions and metadata supplied at iteration zero."),
    "artifact:task_reward_0": ("Initial task reward", "Seed reward supplied as mathematics or executable code."),
    "artifact:oracle_k": ("Reference oracle k", "Executable reference composition, state machine, guards, transitions, and recovery logic."),
    "artifact:task_reward_k": ("Task reward k", "Executable task-specific reward program at iteration k."),
    "system:parameterized_mdp": ("Generic parameterized MDP trainer", "Fixed scene, observation/action spaces, dynamics, controller, tracking reward, algorithm, and budget."),
    "artifact:policy_k": ("Policy checkpoint k", "Policy produced by the fixed training system for one artifact pair."),
    "system:independent_evaluator": ("Independent evaluator", "Computes declared task, tracking, safety, stability, transition, compute, and uncertainty metrics."),
    "artifact:metrics_k": ("Metrics and failure traces k", "Evaluation evidence returned to the harness for the next decision."),
    "input:human_steering": ("Human steering", "Compact preference or correction translated by the harness, never sent directly to the low-level trainer."),
    "boundary:frozen_mdp": ("Frozen MDP boundary", "Components outside the two permitted harness artifacts for a controlled comparison."),
}
PROJECT_EDGES = (
    ("artifact:reference_dataset", "INPUT_TO", "system:sam_harness"),
    ("artifact:task_reward_0", "INPUT_TO", "system:sam_harness"),
    ("input:human_steering", "STEERS", "system:sam_harness"),
    ("system:sam_harness", "PRODUCES", "artifact:oracle_k"),
    ("system:sam_harness", "PRODUCES", "artifact:task_reward_k"),
    ("artifact:oracle_k", "CONFIGURES", "system:parameterized_mdp"),
    ("artifact:task_reward_k", "CONFIGURES", "system:parameterized_mdp"),
    ("boundary:frozen_mdp", "CONSTRAINS", "system:parameterized_mdp"),
    ("system:parameterized_mdp", "RETURNS", "artifact:policy_k"),
    ("artifact:policy_k", "EVALUATED_BY", "system:independent_evaluator"),
    ("system:independent_evaluator", "REPORTS", "artifact:metrics_k"),
    ("artifact:metrics_k", "FEEDBACK_TO", "system:sam_harness"),
)


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def evidence_keys(paper_id: str, evidence_ids: Iterable[str]) -> list[str]:
    """Convert paper-local evidence labels into globally unique graph keys."""
    return sorted(f"{paper_id}:{evidence_id}" for evidence_id in evidence_ids)


def load_extractions() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(EXTRACTIONS.glob("*.json")):
        if path.name.startswith("_"):
            continue
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        record["_source_file"] = path.name
        records.append(record)
    return sorted(records, key=lambda item: item["paper"]["id"])


def entity_node(paper_id: str, kind: str, entity: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": entity["id"],
        "type": kind,
        "paper_id": paper_id,
        "name": entity["name"],
        "description": entity["description"],
        "tags": sorted(entity.get("tags", [])),
        "evidence_ids": evidence_keys(paper_id, entity.get("evidence_ids", [])),
    }


def graph_rows(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {
        f"knob:{name}": {
            "id": f"knob:{name}",
            "type": "project_knob",
            "name": name,
            "description": description,
        }
        for name, description in KNOB_DEFINITIONS.items()
    }
    nodes.update(
        {
            component_id: {
                "id": component_id,
                "type": "project_component",
                "name": name,
                "description": description,
            }
            for component_id, (name, description) in PROJECT_COMPONENTS.items()
        }
    )
    edges: list[dict[str, Any]] = [
        {
            "source": source,
            "relation": relation,
            "target": target,
            "evidence_ids": [],
            "provenance": "sanitized_context:legacy_project_contract",
        }
        for source, relation, target in PROJECT_EDGES
    ]
    unresolved: set[str] = set()

    for record in records:
        paper = record["paper"]
        paper_id = paper["id"]
        nodes[paper_id] = {
            "id": paper_id,
            "type": "paper",
            "name": paper["title"],
            "description": record["decision_relevance"]["why_admitted"],
            "year": paper["year"],
            "authors": paper["authors"],
            "primary_url": paper["primary_url"],
            "tier": record["decision_relevance"]["tier"],
            "knobs": sorted(record["decision_relevance"]["knobs"]),
            "source_version": paper["source_version"],
            "publication_status": paper.get("publication_status", "active"),
        }

        for field, kind in (
            ("mechanisms", "mechanism"),
            ("failure_modes", "failure_mode"),
            ("evaluation", "evaluation"),
        ):
            for entity in record[field]:
                nodes[entity["id"]] = entity_node(paper_id, kind, entity)

        for index, parameter in enumerate(record["parameters"], start=1):
            parameter_id = f"{paper_id}:parameter:{index}:{parameter['name']}"
            nodes[parameter_id] = {
                "id": parameter_id,
                "type": "parameter",
                "paper_id": paper_id,
                "name": parameter["name"],
                "description": parameter["context"],
                "value": parameter["value"],
                "units": parameter["units"],
                "evidence_ids": evidence_keys(paper_id, parameter["evidence_ids"]),
            }
            edges.append(
                {
                    "source": paper_id,
                    "relation": "REPORTS",
                    "target": parameter_id,
                    "evidence_ids": evidence_keys(paper_id, parameter["evidence_ids"]),
                    "provenance": f"paper_extraction:{paper_id}",
                }
            )

        for relation in record["relations"]:
            edge = {
                "source": relation["source"],
                "relation": relation["relation"],
                "target": relation["target"],
                "evidence_ids": evidence_keys(paper_id, relation["evidence_ids"]),
                "provenance": f"paper_extraction:{paper_id}",
            }
            edges.append(edge)
            if edge["source"] not in nodes:
                unresolved.add(edge["source"])
            if edge["target"] not in nodes:
                unresolved.add(edge["target"])

        for knob in sorted(record["decision_relevance"]["knobs"]):
            edges.append(
                {
                    "source": paper_id,
                    "relation": "INFORMS" if paper.get("publication_status", "active") == "active" else "QUARANTINED_FOR",
                    "target": f"knob:{knob}",
                    "evidence_ids": [],
                    "provenance": "curatorial:decision_relevance.knobs",
                }
            )

    # Cross-paper concepts are explicit nodes, never silent dangling endpoints.
    for concept_id in sorted(unresolved - nodes.keys()):
        nodes[concept_id] = {
            "id": concept_id,
            "type": "concept",
            "name": concept_id.replace("_", " ").replace(":", " / "),
            "description": "Concept endpoint declared by a source extraction.",
        }

    ordered_nodes = [nodes[key] for key in sorted(nodes)]
    ordered_edges = sorted(
        edges,
        key=lambda item: (item["source"], item["relation"], item["target"], compact(item["evidence_ids"])),
    )
    return ordered_nodes, ordered_edges


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def bib_escape(value: str) -> str:
    replacements = {"&": r"\&", "%": r"\%", "#": r"\#", "_": r"\_"}
    return "".join(replacements.get(character, character) for character in value)


def write_bibliography(path: Path, records: Iterable[dict[str, Any]]) -> None:
    entries: list[str] = []
    for record in records:
        paper = record["paper"]
        key_suffix = re.sub(r"[^A-Za-z0-9]", "", paper["id"])
        fields = [
            ("title", "{" + bib_escape(paper["title"]) + "}"),
            ("author", bib_escape(" and ".join(paper["authors"]))),
            ("year", str(paper["year"])),
            ("url", bib_escape(paper["primary_url"])),
            ("note", "Publication status: " + paper.get("publication_status", "active") + "; source version: " + bib_escape(paper["source_version"])),
        ]
        if paper.get("doi"):
            fields.insert(3, ("doi", bib_escape(paper["doi"])))
        body = ",\n".join(f"  {name} = {{{value}}}" for name, value in fields)
        entries.append(f"@misc{{rlsculptor{key_suffix},\n{body}\n}}")
    path.write_text("\n\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")


def write_manifest(path: Path, records: list[dict[str, Any]], nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    inputs = []
    for record in records:
        source = EXTRACTIONS / record["_source_file"]
        inputs.append(
            {
                "paper_id": record["paper"]["id"],
                "path": f"extractions/{source.name}",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source_version": record["paper"]["source_version"],
            }
        )
    entity_counts: dict[str, int] = {}
    for node in nodes:
        entity_counts[node["type"]] = entity_counts.get(node["type"], 0) + 1
    manifest = {
        "schema_version": 1,
        "scope_date": "2026-08-31",
        "input_count": len(inputs),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_types": dict(sorted(entity_counts.items())),
        "inputs": inputs,
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_database(path: Path, records: list[dict[str, Any]], nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA foreign_keys = ON;
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE papers (
          id TEXT PRIMARY KEY,
          arxiv_id TEXT,
          doi TEXT,
          title TEXT NOT NULL,
          authors_json TEXT NOT NULL,
          year INTEGER NOT NULL,
          primary_url TEXT NOT NULL,
          source_version TEXT NOT NULL,
          publication_status TEXT NOT NULL,
          tier TEXT NOT NULL,
          knobs_json TEXT NOT NULL,
          why_admitted TEXT NOT NULL,
          not_evidence_for_json TEXT NOT NULL
        );
        CREATE TABLE entities (
          id TEXT PRIMARY KEY,
          paper_id TEXT NOT NULL REFERENCES papers(id),
          kind TEXT NOT NULL,
          name TEXT NOT NULL,
          description TEXT NOT NULL,
          tags_json TEXT NOT NULL,
          evidence_ids_json TEXT NOT NULL
        );
        CREATE TABLE parameters (
          id INTEGER PRIMARY KEY,
          paper_id TEXT NOT NULL REFERENCES papers(id),
          name TEXT NOT NULL,
          value_json TEXT NOT NULL,
          units TEXT,
          context TEXT NOT NULL,
          evidence_ids_json TEXT NOT NULL
        );
        CREATE TABLE evidence (
          id TEXT PRIMARY KEY,
          local_id TEXT NOT NULL,
          paper_id TEXT NOT NULL REFERENCES papers(id),
          source_url TEXT NOT NULL,
          locator TEXT NOT NULL,
          claim TEXT NOT NULL,
          evidence_type TEXT NOT NULL
        );
        CREATE TABLE nodes (
          id TEXT PRIMARY KEY,
          type TEXT NOT NULL,
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
        CREATE INDEX idx_entities_paper_kind ON entities(paper_id, kind);
        CREATE INDEX idx_parameters_paper_name ON parameters(paper_id, name);
        CREATE INDEX idx_evidence_paper ON evidence(paper_id);
        CREATE INDEX idx_edges_source_relation ON edges(source, relation);
        CREATE INDEX idx_edges_target_relation ON edges(target, relation);
        """
    )
    connection.execute("INSERT INTO metadata VALUES (?, ?)", ("scope_date", "2026-08-31"))

    for record in records:
        paper = record["paper"]
        relevance = record["decision_relevance"]
        connection.execute(
            "INSERT INTO papers VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                paper["id"], paper.get("arxiv_id"), paper.get("doi"), paper["title"],
                compact(paper["authors"]), paper["year"], paper["primary_url"], paper["source_version"],
                paper.get("publication_status", "active"),
                relevance["tier"], compact(sorted(relevance["knobs"])), relevance["why_admitted"],
                compact(relevance["not_evidence_for"]),
            ),
        )
        for field, kind in (("mechanisms", "mechanism"), ("failure_modes", "failure_mode"), ("evaluation", "evaluation")):
            for entity in record[field]:
                connection.execute(
                    "INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        entity["id"], paper["id"], kind, entity["name"], entity["description"],
                        compact(sorted(entity.get("tags", []))), compact(evidence_keys(paper["id"], entity.get("evidence_ids", []))),
                    ),
                )
        for parameter in record["parameters"]:
            connection.execute(
                "INSERT INTO parameters(paper_id, name, value_json, units, context, evidence_ids_json) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    paper["id"], parameter["name"], compact(parameter["value"]), parameter["units"],
                    parameter["context"], compact(evidence_keys(paper["id"], parameter["evidence_ids"])),
                ),
            )
        for evidence in record["evidence"]:
            connection.execute(
                "INSERT INTO evidence VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    f"{paper['id']}:{evidence['id']}", evidence["id"], paper["id"], evidence["source_url"], evidence["locator"],
                    evidence["claim"], evidence["evidence_type"],
                ),
            )

    for node in nodes:
        connection.execute("INSERT INTO nodes VALUES (?, ?, ?)", (node["id"], node["type"], compact(node)))
    for edge in edges:
        connection.execute(
            "INSERT INTO edges(source, relation, target, evidence_ids_json, provenance) VALUES (?, ?, ?, ?, ?)",
            (
                edge["source"], edge["relation"], edge["target"], compact(edge["evidence_ids"]),
                edge.get("provenance", "paper_extraction"),
            ),
        )

    try:
        connection.executescript(
            """
            CREATE VIRTUAL TABLE search USING fts5(kind, record_id UNINDEXED, paper_id UNINDEXED, title, body);
            """
        )
        for record in records:
            paper = record["paper"]
            connection.execute(
                "INSERT INTO search VALUES (?, ?, ?, ?, ?)",
                ("paper", paper["id"], paper["id"], paper["title"], record["decision_relevance"]["why_admitted"]),
            )
            for field, kind in (("mechanisms", "mechanism"), ("failure_modes", "failure_mode"), ("evaluation", "evaluation")):
                for entity in record[field]:
                    connection.execute(
                        "INSERT INTO search VALUES (?, ?, ?, ?, ?)",
                        (kind, entity["id"], paper["id"], entity["name"], entity["description"]),
                    )
            for evidence in record["evidence"]:
                connection.execute(
                    "INSERT INTO search VALUES (?, ?, ?, ?, ?)",
                    ("evidence", f"{paper['id']}:{evidence['id']}", paper["id"], evidence["locator"], evidence["claim"]),
                )
        for node in nodes:
            if node["type"] in {"project_component", "project_knob", "concept"}:
                connection.execute(
                    "INSERT INTO search VALUES (?, ?, ?, ?, ?)",
                    (node["type"], node["id"], node.get("paper_id", ""), node["name"], node["description"]),
                )
    except sqlite3.OperationalError:
        connection.execute("INSERT INTO metadata VALUES (?, ?)", ("fts5", "unavailable"))
    else:
        connection.execute("INSERT INTO metadata VALUES (?, ?)", ("fts5", "enabled"))

    connection.commit()
    connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Load and summarize without writing artifacts")
    args = parser.parse_args()

    records = load_extractions()
    nodes, edges = graph_rows(records)
    if args.check_only:
        print(compact({"papers": len(records), "nodes": len(nodes), "edges": len(edges)}))
        return 0

    GRAPH.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kg-build-", dir=GRAPH) as temporary:
        staging = Path(temporary)
        write_jsonl(staging / "nodes.jsonl", nodes)
        write_jsonl(staging / "edges.jsonl", edges)
        write_manifest(staging / "manifest.json", records, nodes, edges)
        write_bibliography(staging / "references.bib", records)
        create_database(staging / "graph.db", records, nodes, edges)
        for name in ("nodes.jsonl", "edges.jsonl", "manifest.json", "graph.db"):
            os.replace(staging / name, GRAPH / name)
        os.replace(staging / "references.bib", ROOT / "references.bib")

    print(compact({"papers": len(records), "nodes": len(nodes), "edges": len(edges), "database": str(GRAPH / "graph.db")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
