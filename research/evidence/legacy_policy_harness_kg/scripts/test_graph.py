#!/usr/bin/env python3
"""Regression tests for deterministic graph construction and retrieval."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from build_graph import create_database, graph_rows, load_extractions


QUERY_SCRIPT = Path(__file__).resolve().with_name("query_graph.py")


class GraphBuildTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = load_extractions()
        cls.nodes, cls.edges = graph_rows(cls.records)

    def test_unique_nodes_and_resolved_edges(self) -> None:
        node_ids = [node["id"] for node in self.nodes]
        self.assertEqual(len(node_ids), len(set(node_ids)))
        endpoints = set(node_ids)
        for edge in self.edges:
            self.assertIn(edge["source"], endpoints)
            self.assertIn(edge["target"], endpoints)

    def test_evidence_keys_are_namespaced(self) -> None:
        for node in self.nodes:
            paper_id = node.get("paper_id")
            if not paper_id:
                continue
            for evidence_id in node.get("evidence_ids", []):
                self.assertTrue(evidence_id.startswith(f"{paper_id}:"), evidence_id)

    def test_database_foreign_keys_and_search(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kg-test-") as temporary:
            database = Path(temporary) / "graph.db"
            create_database(database, self.records, self.nodes, self.edges)
            connection = sqlite3.connect(database)
            self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
            paper_count = connection.execute("SELECT count(*) FROM papers").fetchone()[0]
            self.assertEqual(paper_count, len(self.records))
            fts = connection.execute("SELECT value FROM metadata WHERE key='fts5'").fetchone()[0]
            if fts == "enabled":
                connection.execute("SELECT count(*) FROM search WHERE search MATCH 'reward'").fetchone()
            connection.close()

    def test_inactive_papers_do_not_inform_project_knobs(self) -> None:
        status_by_paper = {
            record["paper"]["id"]: record["paper"].get("publication_status", "active")
            for record in self.records
        }
        for edge in self.edges:
            if edge["relation"] == "INFORMS" and edge["source"] in status_by_paper:
                self.assertEqual(status_by_paper[edge["source"]], "active")
            if edge["relation"] == "QUARANTINED_FOR":
                self.assertNotEqual(status_by_paper[edge["source"]], "active")

    def test_broad_queries_hide_inactive_papers_by_default(self) -> None:
        inactive = [
            record["paper"]["id"]
            for record in self.records
            if record["paper"].get("publication_status", "active") != "active"
        ]
        if not inactive:
            self.skipTest("corpus has no inactive paper")
        with tempfile.TemporaryDirectory(prefix="kg-query-test-") as temporary:
            database = Path(temporary) / "graph.db"
            create_database(database, self.records, self.nodes, self.edges)
            base = [sys.executable, str(QUERY_SCRIPT), "--db", str(database), "parameters", "--paper", inactive[0]]
            hidden = subprocess.run(base, check=True, capture_output=True, text=True)
            visible = subprocess.run(base + ["--include-inactive"], check=True, capture_output=True, text=True)
            self.assertEqual(hidden.stdout, "")
            self.assertNotEqual(visible.stdout, "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
