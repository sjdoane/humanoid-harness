#!/usr/bin/env python3
"""Query the policy-harness research knowledge graph."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


DATABASE = Path(__file__).resolve().parents[1] / "graph" / "graph.db"


def rows(connection: sqlite3.Connection, sql: str, parameters: tuple[object, ...] = ()) -> list[dict[str, object]]:
    return [dict(row) for row in connection.execute(sql, parameters).fetchall()]


def print_rows(items: list[dict[str, object]]) -> None:
    for item in items:
        print(json.dumps(item, ensure_ascii=False, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DATABASE)
    subparsers = parser.add_subparsers(dest="command", required=True)

    papers = subparsers.add_parser("papers", help="List papers, optionally filtered")
    papers.add_argument("--tag", help="Match a knob, tier, title, or admission rationale")
    papers.add_argument("--include-inactive", action="store_true", help="Include withdrawn, retracted, or superseded records")

    paper = subparsers.add_parser("paper", help="Show one paper with extraction counts")
    paper.add_argument("id")

    mechanisms = subparsers.add_parser("mechanisms", help="List mechanism nodes")
    mechanisms.add_argument("--knob", choices=["reference_composition", "task_reward", "human_steering", "evaluation", "fixed_trainer"])
    mechanisms.add_argument("--text")
    mechanisms.add_argument("--include-inactive", action="store_true")

    parameters = subparsers.add_parser("parameters", help="Find exact parameters")
    parameters.add_argument("--name")
    parameters.add_argument("--paper")
    parameters.add_argument("--include-inactive", action="store_true")

    failures = subparsers.add_parser("failures", help="Find failure modes")
    failures.add_argument("--text")
    failures.add_argument("--paper")
    failures.add_argument("--include-inactive", action="store_true")

    evidence = subparsers.add_parser("evidence", help="Show claim provenance")
    evidence.add_argument("--paper", required=True)

    sources = subparsers.add_parser("sources", help="List primary and official evidence sources")
    sources.add_argument("--type", dest="evidence_type")
    sources.add_argument("--paper")
    sources.add_argument("--include-inactive", action="store_true")

    subparsers.add_parser("stats", help="Show graph coverage and artifact counts")

    neighbors = subparsers.add_parser("neighbors", help="Traverse one hop from a node")
    neighbors.add_argument("node")
    neighbors.add_argument("--relation")

    search = subparsers.add_parser("search", help="Full-text search over papers, entities, and evidence")
    search.add_argument("query")
    search.add_argument("--limit", type=int, default=20)
    search.add_argument("--include-inactive", action="store_true")

    args = parser.parse_args()
    if not args.db.exists():
        parser.error(f"database not found: {args.db}; run scripts/build_graph.py")
    connection = sqlite3.connect(args.db)
    connection.row_factory = sqlite3.Row

    if args.command == "papers":
        sql = "SELECT id, title, year, publication_status, tier, knobs_json, primary_url FROM papers WHERE 1=1"
        values_list: list[object] = []
        if not args.include_inactive:
            sql += " AND publication_status = 'active'"
        if args.tag:
            sql += " AND lower(title || ' ' || tier || ' ' || knobs_json || ' ' || why_admitted) LIKE ?"
            values_list.append(f"%{args.tag.lower()}%")
        sql += " ORDER BY year DESC, id"
        result = rows(connection, sql, tuple(values_list))
    elif args.command == "paper":
        result = rows(
            connection,
            """
            SELECT p.*,
              (SELECT count(*) FROM entities e WHERE e.paper_id=p.id AND e.kind='mechanism') AS mechanism_count,
              (SELECT count(*) FROM entities e WHERE e.paper_id=p.id AND e.kind='failure_mode') AS failure_count,
              (SELECT count(*) FROM entities e WHERE e.paper_id=p.id AND e.kind='evaluation') AS evaluation_count,
              (SELECT count(*) FROM parameters x WHERE x.paper_id=p.id) AS parameter_count,
              (SELECT count(*) FROM evidence v WHERE v.paper_id=p.id) AS evidence_count
            FROM papers p WHERE p.id=?
            """,
            (args.id,),
        )
    elif args.command == "mechanisms":
        sql = """
          SELECT e.id, e.paper_id, e.name, e.description, e.tags_json
          FROM entities e JOIN papers p ON p.id = e.paper_id
          WHERE e.kind = 'mechanism'
        """
        conditions: list[str] = []
        values_list: list[object] = []
        if not args.include_inactive:
            conditions.append("p.publication_status = 'active'")
        if args.knob:
            conditions.append("p.knobs_json LIKE ?")
            values_list.append(f"%{args.knob}%")
        if args.text:
            conditions.append("lower(e.name || ' ' || e.description || ' ' || e.tags_json) LIKE ?")
            values_list.append(f"%{args.text.lower()}%")
        if conditions:
            sql += " AND " + " AND ".join(conditions)
        sql += " ORDER BY e.paper_id, e.name"
        result = rows(connection, sql, tuple(values_list))
    elif args.command == "parameters":
        sql = """
          SELECT x.paper_id, x.name, x.value_json, x.units, x.context, x.evidence_ids_json
          FROM parameters x JOIN papers p ON p.id = x.paper_id WHERE 1=1
        """
        values_list = []
        if not args.include_inactive:
            sql += " AND p.publication_status = 'active'"
        if args.name:
            sql += " AND lower(x.name || ' ' || x.context) LIKE ?"
            values_list.append(f"%{args.name.lower()}%")
        if args.paper:
            sql += " AND x.paper_id = ?"
            values_list.append(args.paper)
        sql += " ORDER BY x.paper_id, x.name"
        result = rows(connection, sql, tuple(values_list))
    elif args.command == "failures":
        sql = """
          SELECT e.id, e.paper_id, e.name, e.description, e.evidence_ids_json
          FROM entities e JOIN papers p ON p.id = e.paper_id
          WHERE e.kind = 'failure_mode'
        """
        values_list = []
        if not args.include_inactive:
            sql += " AND p.publication_status = 'active'"
        if args.text:
            sql += " AND lower(e.name || ' ' || e.description || ' ' || e.tags_json) LIKE ?"
            values_list.append(f"%{args.text.lower()}%")
        if args.paper:
            sql += " AND e.paper_id = ?"
            values_list.append(args.paper)
        sql += " ORDER BY e.paper_id, e.name"
        result = rows(connection, sql, tuple(values_list))
    elif args.command == "evidence":
        result = rows(
            connection,
            "SELECT id, local_id, source_url, locator, claim, evidence_type FROM evidence WHERE paper_id = ? ORDER BY local_id",
            (args.paper,),
        )
    elif args.command == "sources":
        sql = """
          SELECT v.paper_id, v.evidence_type, v.source_url, v.locator, v.claim
          FROM evidence v JOIN papers p ON p.id = v.paper_id WHERE 1=1
        """
        values_list = []
        if not args.include_inactive:
            sql += " AND p.publication_status = 'active'"
        if args.evidence_type:
            sql += " AND v.evidence_type = ?"
            values_list.append(args.evidence_type)
        if args.paper:
            sql += " AND v.paper_id = ?"
            values_list.append(args.paper)
        sql += " ORDER BY v.paper_id, v.evidence_type, v.source_url, v.locator"
        result = rows(connection, sql, tuple(values_list))
    elif args.command == "stats":
        result = rows(
            connection,
            """
            SELECT
              (SELECT count(*) FROM papers) AS papers,
              (SELECT count(*) FROM papers WHERE publication_status='active') AS active_papers,
              (SELECT count(*) FROM papers WHERE publication_status!='active') AS inactive_papers,
              (SELECT count(*) FROM entities WHERE kind='mechanism') AS mechanisms,
              (SELECT count(*) FROM parameters) AS parameters,
              (SELECT count(*) FROM entities WHERE kind='failure_mode') AS failure_modes,
              (SELECT count(*) FROM entities WHERE kind='evaluation') AS evaluations,
              (SELECT count(*) FROM evidence) AS evidence_items,
              (SELECT count(*) FROM nodes) AS nodes,
              (SELECT count(*) FROM edges) AS edges
            """,
        )
    elif args.command == "neighbors":
        sql = """
          SELECT source, relation, target, evidence_ids_json, provenance
          FROM edges WHERE (source = ? OR target = ?)
        """
        values_list = [args.node, args.node]
        if args.relation:
            sql += " AND relation = ?"
            values_list.append(args.relation)
        sql += " ORDER BY relation, source, target"
        result = rows(connection, sql, tuple(values_list))
    else:
        try:
            active_filter = "" if args.include_inactive else " AND (paper_id IS NULL OR paper_id IN (SELECT id FROM papers WHERE publication_status='active'))"
            result = rows(
                connection,
                "SELECT kind, record_id, paper_id, title, body FROM search WHERE search MATCH ?"
                + active_filter
                + " ORDER BY rank LIMIT ?",
                (args.query, args.limit),
            )
        except sqlite3.OperationalError as error:
            parser.error(f"full-text search unavailable: {error}")

    print_rows(result)
    connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
