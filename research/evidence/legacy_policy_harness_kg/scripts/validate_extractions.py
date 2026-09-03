#!/usr/bin/env python3
"""Validate extraction shape, IDs, evidence links, and graph invariants."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
EXTRACTIONS = ROOT / "extractions"
TIERS = {"anchor", "core", "supporting", "baseline"}
KNOBS = {"reference_composition", "task_reward", "human_steering", "evaluation", "fixed_trainer"}
RELATIONS = {
    "INTRODUCES", "EXTENDS", "USES", "CONDITIONS_ON", "COMPOSES", "TRACKS", "OPTIMIZES",
    "ADDRESSES", "EVALUATES_ON", "REPORTS", "REQUIRES", "SUPPORTS", "CONTRASTS_WITH", "MOTIVATES",
}
ENTITY_FIELDS = ("mechanisms", "failure_modes", "evaluation")
REQUIRED_TOP = {
    "paper", "decision_relevance", "mechanisms", "parameters", "failure_modes", "evaluation", "relations", "evidence"
}
ARXIV_ID = re.compile(r"^[0-9]{4}\.[0-9]{4,5}$")


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def public_url(value: Any) -> bool:
    if not nonempty_string(value):
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    files = [path for path in sorted(EXTRACTIONS.glob("*.json")) if not path.name.startswith("_")]
    records: list[tuple[Path, dict[str, Any]]] = []
    paper_ids: list[str] = []
    global_entity_ids: list[str] = []

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            errors.append(f"{path.name}: cannot load JSON: {error}")
            continue
        records.append((path, data))
        prefix = path.name
        require(REQUIRED_TOP <= data.keys(), f"{prefix}: missing top-level fields {sorted(REQUIRED_TOP - data.keys())}", errors)
        if not REQUIRED_TOP <= data.keys():
            continue

        paper = data["paper"]
        for key in ("id", "title", "authors", "year", "primary_url", "source_version"):
            require(key in paper, f"{prefix}: paper.{key} missing", errors)
        paper_id = paper.get("id", "")
        require(nonempty_string(paper_id), f"{prefix}: paper.id must be non-empty", errors)
        if nonempty_string(paper_id):
            paper_ids.append(paper_id)
            require(bool(ARXIV_ID.fullmatch(paper_id)), f"{prefix}: paper.id must be a bare arXiv ID", errors)
            require(path.stem == paper_id, f"{prefix}: filename must match paper.id ({paper_id})", errors)
        require(nonempty_string(paper.get("title")), f"{prefix}: paper.title must be non-empty", errors)
        require(isinstance(paper.get("authors"), list) and all(nonempty_string(author) for author in paper.get("authors", [])), f"{prefix}: paper.authors invalid", errors)
        require(isinstance(paper.get("year"), int), f"{prefix}: paper.year must be an integer", errors)
        require(public_url(paper.get("primary_url")), f"{prefix}: paper.primary_url invalid", errors)
        require(nonempty_string(paper.get("source_version")), f"{prefix}: paper.source_version invalid", errors)
        require(paper.get("publication_status", "active") in {"active", "withdrawn", "retracted", "superseded"}, f"{prefix}: paper.publication_status invalid", errors)

        relevance = data["decision_relevance"]
        require(relevance.get("tier") in TIERS, f"{prefix}: invalid evidence tier", errors)
        knobs = relevance.get("knobs", [])
        require(isinstance(knobs, list) and bool(knobs) and set(knobs) <= KNOBS, f"{prefix}: invalid or empty knobs", errors)
        require(nonempty_string(relevance.get("why_admitted")), f"{prefix}: why_admitted is empty", errors)
        require(isinstance(relevance.get("not_evidence_for"), list), f"{prefix}: not_evidence_for must be a list", errors)

        evidence = data["evidence"]
        require(isinstance(evidence, list) and bool(evidence), f"{prefix}: evidence must be non-empty", errors)
        evidence_id_list = [item.get("id") for item in evidence if isinstance(item, dict)]
        evidence_ids = set(evidence_id_list)
        for evidence_id, count in Counter(evidence_id_list).items():
            if evidence_id and count > 1:
                errors.append(f"{prefix}: duplicate evidence ID {evidence_id}")
        for item in evidence:
            require(isinstance(item, dict), f"{prefix}: evidence item must be an object", errors)
            if not isinstance(item, dict):
                continue
            for key in ("id", "source_url", "locator", "claim", "evidence_type"):
                require(nonempty_string(item.get(key)), f"{prefix}: evidence.{key} must be non-empty", errors)
            require(public_url(item.get("source_url")), f"{prefix}: evidence.source_url invalid for {item.get('id')}", errors)

        referenced_evidence: list[tuple[str, str]] = []
        for field in ENTITY_FIELDS:
            require(isinstance(data[field], list), f"{prefix}: {field} must be a list", errors)
            for entity in data[field]:
                if not isinstance(entity, dict):
                    errors.append(f"{prefix}: {field} item must be an object")
                    continue
                entity_id = entity.get("id")
                require(nonempty_string(entity_id), f"{prefix}: {field}.id must be non-empty", errors)
                if nonempty_string(entity_id):
                    global_entity_ids.append(entity_id)
                for key in ("name", "description"):
                    require(nonempty_string(entity.get(key)), f"{prefix}: {field}.{key} must be non-empty", errors)
                require(isinstance(entity.get("tags"), list), f"{prefix}: {field}.tags must be a list", errors)
                entity_evidence = entity.get("evidence_ids")
                require(isinstance(entity_evidence, list) and bool(entity_evidence), f"{prefix}: {field}:{entity_id} must cite evidence", errors)
                for evidence_id in entity_evidence or []:
                    referenced_evidence.append((f"{field}:{entity_id}", evidence_id))

        require(isinstance(data["parameters"], list), f"{prefix}: parameters must be a list", errors)
        for parameter in data["parameters"]:
            if not isinstance(parameter, dict):
                errors.append(f"{prefix}: parameter must be an object")
                continue
            require(nonempty_string(parameter.get("name")), f"{prefix}: parameter.name must be non-empty", errors)
            require("value" in parameter, f"{prefix}: parameter.value missing", errors)
            require(nonempty_string(parameter.get("context")), f"{prefix}: parameter.context must be non-empty", errors)
            parameter_evidence = parameter.get("evidence_ids")
            require(isinstance(parameter_evidence, list) and bool(parameter_evidence), f"{prefix}: parameter:{parameter.get('name')} must cite evidence", errors)
            for evidence_id in parameter_evidence or []:
                referenced_evidence.append((f"parameter:{parameter.get('name')}", evidence_id))

        require(isinstance(data["relations"], list), f"{prefix}: relations must be a list", errors)
        for relation in data["relations"]:
            if not isinstance(relation, dict):
                errors.append(f"{prefix}: relation must be an object")
                continue
            require(nonempty_string(relation.get("source")), f"{prefix}: relation.source must be non-empty", errors)
            require(relation.get("relation") in RELATIONS, f"{prefix}: invalid relation {relation.get('relation')}", errors)
            require(nonempty_string(relation.get("target")), f"{prefix}: relation.target must be non-empty", errors)
            relation_evidence = relation.get("evidence_ids")
            require(isinstance(relation_evidence, list) and bool(relation_evidence), f"{prefix}: relation:{relation.get('source')}->{relation.get('target')} must cite evidence", errors)
            for evidence_id in relation_evidence or []:
                referenced_evidence.append((f"relation:{relation.get('source')}->{relation.get('target')}", evidence_id))

        for location, evidence_id in referenced_evidence:
            require(evidence_id in evidence_ids, f"{prefix}: {location} references unknown evidence {evidence_id}", errors)
        # Standalone ledger entries are valid: metadata and negative evidence can
        # remain directly searchable without being attached to a graph edge.

    for label, values in (("paper ID", paper_ids), ("entity ID", global_entity_ids)):
        for value, count in Counter(values).items():
            if count > 1:
                errors.append(f"duplicate {label}: {value} ({count} occurrences)")

    summary = {"files": len(files), "errors": len(errors), "warnings": len(warnings)}
    print(json.dumps(summary, sort_keys=True))
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
