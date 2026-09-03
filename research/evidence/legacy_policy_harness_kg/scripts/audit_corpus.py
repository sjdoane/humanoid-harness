#!/usr/bin/env python3
"""Audit corpus registry, one-paper-per-agent assignments, and artifact coverage."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus.yaml"
ALLOWED_STATUSES = {"queued", "assigned", "extracted", "quarantined_withdrawn"}


def main() -> int:
    records: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for raw_line in CORPUS.read_text(encoding="utf-8").splitlines():
        match = re.match(r'^  - id: "([0-9]{4}\.[0-9]{4,5})"$', raw_line)
        if match:
            current = {"id": match.group(1)}
            records.append(current)
            continue
        if current is None:
            continue
        match = re.match(r"^    (agent|status): ([A-Za-z0-9_]+)$", raw_line)
        if match:
            current[match.group(1)] = match.group(2)

    errors: list[str] = []
    for field in ("id", "agent", "status"):
        missing = [record.get("id", "unknown") for record in records if field not in record]
        if missing:
            errors.append(f"missing {field}: {', '.join(missing)}")
    for field in ("id", "agent"):
        for value, count in Counter(record.get(field) for record in records).items():
            if value and count > 1:
                errors.append(f"duplicate {field}: {value}")
    for record in records:
        if record.get("status") not in ALLOWED_STATUSES:
            errors.append(f"{record.get('id', 'unknown')}: unsupported status {record.get('status')}")

    corpus_ids = {record["id"] for record in records}
    json_ids = {path.stem for path in (ROOT / "extractions").glob("*.json") if not path.name.startswith("_")}
    card_ids = {path.stem for path in (ROOT / "cards").glob("*.md") if not path.name.startswith("_")}
    for paper_id in sorted((json_ids | card_ids) - corpus_ids):
        errors.append(f"artifact has no corpus record: {paper_id}")
    status_by_id = {record["id"]: record.get("status", "missing") for record in records}
    for paper_id in sorted(json_ids ^ card_ids):
        if status_by_id.get(paper_id) != "assigned":
            errors.append(f"paper must have both card and extraction: {paper_id}")
    complete_statuses = {"extracted", "quarantined_withdrawn"}
    for record in records:
        if record.get("status") == "queued" and (record["id"] in json_ids or record["id"] in card_ids):
            errors.append(f"{record['id']}: queued status already has an artifact")
        if record.get("status") in complete_statuses:
            if record["id"] not in json_ids or record["id"] not in card_ids:
                errors.append(f"{record['id']}: complete status without both artifacts")
        extraction_path = ROOT / "extractions" / f"{record['id']}.json"
        if extraction_path.exists():
            try:
                publication_status = json.loads(extraction_path.read_text(encoding="utf-8")).get("paper", {}).get("publication_status", "active")
            except (json.JSONDecodeError, OSError) as error:
                errors.append(f"{record['id']}: cannot inspect publication status: {error}")
                continue
            if record.get("status") == "quarantined_withdrawn" and publication_status != "withdrawn":
                errors.append(f"{record['id']}: quarantined corpus record must be withdrawn in extraction")
            if record.get("status") == "extracted" and publication_status != "active":
                errors.append(f"{record['id']}: extracted corpus record has non-active publication status {publication_status}")

    statuses = Counter(record.get("status", "missing") for record in records)
    summary = {
        "papers": len(records),
        "agents": len({record.get("agent") for record in records}),
        "cards": len(card_ids),
        "extractions": len(json_ids),
        "statuses": dict(sorted(statuses.items())),
        "errors": len(errors),
    }
    print(json.dumps(summary, sort_keys=True))
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
