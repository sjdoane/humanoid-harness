#!/usr/bin/env python3
"""Acquire or reverify bounded, version-pinned literature PDF bytes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
import urllib.request
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

MAX_PDF_BYTES = 64 * 1024 * 1024
PDF_PREFIX = b"%PDF-"
REVIEW_ID = "001_oracle_composition"
REGISTERED_STATUS = "registered_not_screened"
CITATION_STATUS = "not_citable_until_screened_and_extracted"
DOCUMENT_TYPE = "version-pinned primary-paper PDF"
CLAIM_BOUNDARY = (
    "Exact primary-paper bytes are locally registered but not screened, included, "
    "extracted, synthesized, or admissible as evidence."
)
ARXIV_CANDIDATE = re.compile(r"arxiv:(?P<id>\d{4}\.\d{4,5})v(?P<version>\d+)")
ARXIV_PDF_PATH = re.compile(r"/pdf/(?P<stem>\d{4}\.\d{4,5}v\d+)(?:\.pdf)?")
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "review_id",
        "registered_at_utc",
        "evidence_status",
        "claim_boundary",
        "max_pdf_bytes",
        "sources",
    }
)
SOURCE_FIELDS = frozenset(
    {
        "source_id",
        "candidate_id",
        "title",
        "source_url",
        "local_path",
        "bytes",
        "sha256",
        "status",
    }
)
INVENTORY_FIELDS = (
    "source_id",
    "file_name",
    "file_path",
    "document_type",
    "ingestion_status",
    "citation_status",
    "notes",
)
CANDIDATE_FIELDS = (
    "candidate_id",
    "database",
    "search_string_id",
    "title",
    "authors",
    "year",
    "venue",
    "doi",
    "url",
    "abstract_available",
    "full_text_available",
    "candidate_relevance",
    "workflow_stage_fit",
    "grounding_relevance",
    "conceptual_synthesis_relevance",
    "empirical_evaluation",
    "notes",
)
BATCH_FIELDS = (
    "candidate_id",
    "source_url",
    "local_path",
    "registration_status",
)


class SourceRegistrationError(ValueError):
    """Raised when a candidate or local source violates the registration contract."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _regular_pdf(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise SourceRegistrationError(f"source must be a regular non-symlink file: {path}")
    size = path.stat().st_size
    if size <= len(PDF_PREFIX) or size > MAX_PDF_BYTES:
        raise SourceRegistrationError(f"source size outside allowed bounds: {path}")
    raw = path.read_bytes()
    if len(raw) != size:
        raise SourceRegistrationError(f"source size changed while reading: {path}")
    if not raw.startswith(PDF_PREFIX):
        raise SourceRegistrationError(f"source does not have a PDF header: {path}")
    return raw


def _safe_local_path(repository_root: Path, path_text: str) -> Path:
    relative = PurePosixPath(path_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise SourceRegistrationError(f"unsafe source path: {path_text!r}")
    expected_prefix = PurePosixPath("research/raw_sources/oracle_composition")
    if relative.parent != expected_prefix or relative.suffix.lower() != ".pdf":
        raise SourceRegistrationError(
            "source path must be a PDF directly below research/raw_sources/oracle_composition"
        )
    path = repository_root
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise SourceRegistrationError(f"source path crosses a symlink: {path_text!r}")
    return path


def _validate_arxiv_binding(candidate_id: str, source_url: str, local_path: str) -> str:
    match = ARXIV_CANDIDATE.fullmatch(candidate_id)
    if match is None:
        raise SourceRegistrationError(
            f"batch supports versioned arXiv candidates only: {candidate_id}"
        )
    stem = f"{match.group('id')}v{match.group('version')}"
    expected_url = f"https://arxiv.org/pdf/{stem}"
    expected_path = f"research/raw_sources/oracle_composition/{stem}.pdf"
    if source_url != expected_url:
        raise SourceRegistrationError(f"source URL is not exact and version-pinned: {source_url}")
    if local_path != expected_path:
        raise SourceRegistrationError(f"local path disagrees with candidate ID: {local_path}")
    return stem


def _download_pdf(source_url: str, destination: Path) -> None:
    parsed = urlparse(source_url)
    requested_match = ARXIV_PDF_PATH.fullmatch(parsed.path)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "arxiv.org"
        or parsed.params
        or parsed.query
        or parsed.fragment
        or requested_match is None
    ):
        raise SourceRegistrationError(f"download host is not allowed: {source_url}")
    expected_stem = requested_match.group("stem")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.parent.is_symlink() or not destination.parent.is_dir():
        raise SourceRegistrationError("raw-source directory must be a real directory")
    if destination.is_symlink():
        raise SourceRegistrationError("download destination must not be a symlink")
    request = urllib.request.Request(
        source_url,
        headers={"User-Agent": "humanoid-harness-literature-review/0.1"},
    )
    temporary_path: Path | None = None
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            final_url = response.geturl()
            final = urlparse(final_url)
            final_match = ARXIV_PDF_PATH.fullmatch(final.path)
            if (
                final.scheme != "https"
                or final.netloc not in {"arxiv.org", "export.arxiv.org"}
                or final.params
                or final.query
                or final.fragment
                or final_match is None
                or final_match.group("stem") != expected_stem
            ):
                raise SourceRegistrationError(
                    f"redirect changed exact source identity: {source_url} -> {final_url}"
                )
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > MAX_PDF_BYTES:
                raise SourceRegistrationError("declared PDF size exceeds the registration limit")
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".literature-",
                suffix=".part",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_PDF_BYTES:
                        raise SourceRegistrationError(
                            "downloaded PDF exceeds the registration limit"
                        )
                    temporary.write(chunk)
        if temporary_path is None:
            raise SourceRegistrationError("download did not create a temporary source")
        incoming = _regular_pdf(temporary_path)
        if destination.exists():
            existing = _regular_pdf(destination)
            if _sha256(existing) != _sha256(incoming):
                raise SourceRegistrationError(
                    f"version-pinned source bytes changed at download boundary: {source_url}"
                )
            temporary_path.unlink()
        else:
            os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _load_candidates(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CANDIDATE_FIELDS:
            raise SourceRegistrationError("candidate registry has an unexpected schema")
        rows = list(reader)
    if not rows:
        raise SourceRegistrationError("candidate registry is empty")
    candidates: dict[str, dict[str, str]] = {}
    for row in rows:
        candidate_id = row["candidate_id"]
        if not candidate_id or not row["title"]:
            raise SourceRegistrationError("candidate registry has a blank ID or title")
        if candidate_id in candidates:
            raise SourceRegistrationError(f"duplicate candidate registry ID: {candidate_id}")
        candidates[candidate_id] = row
    return candidates


def _load_batch(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != BATCH_FIELDS:
            raise SourceRegistrationError("acquisition batch has an unexpected schema")
        rows = list(reader)
    if not rows:
        raise SourceRegistrationError("acquisition batch has an unexpected schema or is empty")
    seen: set[str] = set()
    for row in rows:
        candidate_id = row["candidate_id"]
        if candidate_id in seen:
            raise SourceRegistrationError(f"duplicate acquisition candidate: {candidate_id}")
        seen.add(candidate_id)
        if row["registration_status"] != REGISTERED_STATUS:
            raise SourceRegistrationError(f"unsupported registration status: {candidate_id}")
        _validate_arxiv_binding(candidate_id, row["source_url"], row["local_path"])
    return rows


def _load_inventory(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != INVENTORY_FIELDS:
            raise SourceRegistrationError("source inventory has an unexpected schema")
        rows = list(reader)
    inventory: dict[str, dict[str, str]] = {}
    for row in rows:
        source_id = row["source_id"]
        if source_id in inventory:
            raise SourceRegistrationError(f"duplicate source inventory ID: {source_id}")
        inventory[source_id] = row
    return inventory


def _validate_registered_at(value: Any) -> None:
    if not isinstance(value, str):
        raise SourceRegistrationError("registered_at_utc must be a UTC timestamp string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise SourceRegistrationError("registered_at_utc is not ISO 8601") from error
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() != UTC.utcoffset(None)
        or parsed.microsecond != 0
        or parsed.isoformat() != value
    ):
        raise SourceRegistrationError(
            "registered_at_utc must be canonical second-precision ISO 8601 UTC"
        )


def _expected_inventory_row(record: dict[str, Any]) -> dict[str, str]:
    return {
        "source_id": record["source_id"],
        "file_name": PurePosixPath(record["local_path"]).name,
        "file_path": record["local_path"],
        "document_type": DOCUMENT_TYPE,
        "ingestion_status": REGISTERED_STATUS,
        "citation_status": CITATION_STATUS,
        "notes": (f"sha256={record['sha256']}; exact bytes remain local and ignored by Git"),
    }


def register(
    *,
    repository_root: Path,
    candidates_path: Path,
    batch_path: Path,
    manifest_path: Path,
    inventory_path: Path,
    download: bool,
) -> dict[str, Any]:
    candidates = _load_candidates(candidates_path)
    batch = _load_batch(batch_path)
    records: list[dict[str, Any]] = []
    for item in batch:
        candidate_id = item["candidate_id"]
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise SourceRegistrationError(
                f"batch candidate is absent from registry: {candidate_id}"
            )
        if candidate["full_text_available"] != "yes":
            raise SourceRegistrationError(
                f"batch candidate is not marked full-text available: {candidate_id}"
            )
        path = _safe_local_path(repository_root, item["local_path"])
        if download:
            _download_pdf(item["source_url"], path)
        raw = _regular_pdf(path)
        records.append(
            {
                "source_id": candidate_id,
                "candidate_id": candidate_id,
                "title": candidate["title"],
                "source_url": item["source_url"],
                "local_path": item["local_path"],
                "bytes": len(raw),
                "sha256": _sha256(raw),
                "status": REGISTERED_STATUS,
            }
        )
    manifest = {
        "schema_version": 1,
        "review_id": REVIEW_ID,
        "registered_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "evidence_status": REGISTERED_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "max_pdf_bytes": MAX_PDF_BYTES,
        "sources": records,
    }
    manifest_path.write_text(_canonical_json(manifest), encoding="utf-8")
    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(INVENTORY_FIELDS)
        for record in records:
            expected = _expected_inventory_row(record)
            writer.writerow([expected[field] for field in INVENTORY_FIELDS])
    return manifest


def verify(
    *,
    repository_root: Path,
    candidates_path: Path,
    batch_path: Path,
    manifest_path: Path,
    inventory_path: Path,
) -> dict[str, Any]:
    candidates = _load_candidates(candidates_path)
    batch = _load_batch(batch_path)
    inventory = _load_inventory(inventory_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_FIELDS:
        raise SourceRegistrationError("source manifest has an unexpected schema")
    if (
        manifest["schema_version"] != 1
        or isinstance(manifest["schema_version"], bool)
        or manifest["review_id"] != REVIEW_ID
        or manifest["evidence_status"] != REGISTERED_STATUS
        or manifest["claim_boundary"] != CLAIM_BOUNDARY
    ):
        raise SourceRegistrationError("source manifest has an unsupported contract")
    _validate_registered_at(manifest["registered_at_utc"])
    if manifest["max_pdf_bytes"] != MAX_PDF_BYTES or isinstance(manifest["max_pdf_bytes"], bool):
        raise SourceRegistrationError("source manifest size bound disagrees with verifier")
    records = manifest["sources"]
    if not isinstance(records, list) or not records:
        raise SourceRegistrationError("source manifest must contain records")
    batch_by_id = {item["candidate_id"]: item for item in batch}
    record_ids = [
        record.get("source_id") if isinstance(record, dict) else None for record in records
    ]
    batch_ids = [item["candidate_id"] for item in batch]
    if record_ids != batch_ids:
        raise SourceRegistrationError(
            "source manifest IDs/order do not exactly match the acquisition batch"
        )
    if set(inventory) != set(batch_ids):
        raise SourceRegistrationError(
            "source inventory IDs do not exactly match the acquisition batch"
        )
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    total_bytes = 0
    for record in records:
        if not isinstance(record, dict) or set(record) != SOURCE_FIELDS:
            raise SourceRegistrationError("source record has an unexpected schema")
        source_id = record["source_id"]
        path_text = record["local_path"]
        if not isinstance(source_id, str) or not isinstance(path_text, str):
            raise SourceRegistrationError("source manifest ID/path must be strings")
        if source_id in seen_ids or path_text in seen_paths:
            raise SourceRegistrationError("source manifest contains a duplicate ID or path")
        seen_ids.add(source_id)
        seen_paths.add(path_text)
        if source_id != record["candidate_id"] or record["status"] != REGISTERED_STATUS:
            raise SourceRegistrationError(f"source identity/status mismatch: {source_id}")
        candidate = candidates.get(source_id)
        batch_item = batch_by_id.get(source_id)
        if candidate is None or batch_item is None:
            raise SourceRegistrationError(f"source is absent from registry/batch: {source_id}")
        if candidate["full_text_available"] != "yes":
            raise SourceRegistrationError(
                f"registered source is not marked full-text available: {source_id}"
            )
        if record["title"] != candidate["title"]:
            raise SourceRegistrationError(f"source title disagrees with registry: {source_id}")
        if (
            record["source_url"] != batch_item["source_url"]
            or record["local_path"] != batch_item["local_path"]
            or record["status"] != batch_item["registration_status"]
        ):
            raise SourceRegistrationError(f"source disagrees with acquisition batch: {source_id}")
        if (
            not isinstance(record["bytes"], int)
            or isinstance(record["bytes"], bool)
            or record["bytes"] <= len(PDF_PREFIX)
            or record["bytes"] > MAX_PDF_BYTES
        ):
            raise SourceRegistrationError(f"source has an invalid byte count: {source_id}")
        if not isinstance(record["sha256"], str) or SHA256_HEX.fullmatch(record["sha256"]) is None:
            raise SourceRegistrationError(f"source has an invalid SHA-256: {source_id}")
        if not isinstance(record["source_url"], str):
            raise SourceRegistrationError(f"source URL must be a string: {source_id}")
        _validate_arxiv_binding(source_id, record["source_url"], path_text)
        raw = _regular_pdf(_safe_local_path(repository_root, path_text))
        if len(raw) != record["bytes"] or _sha256(raw) != record["sha256"]:
            raise SourceRegistrationError(f"source bytes do not match manifest: {source_id}")
        if inventory[source_id] != _expected_inventory_row(record):
            raise SourceRegistrationError(f"source inventory disagrees with manifest: {source_id}")
        total_bytes += len(raw)
    return {"source_count": len(records), "total_bytes": total_bytes, "verified": True}


def main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    process_root = repository_root / "research/literature_review/01_process/001_oracle_composition"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=process_root / "matrices/candidate_papers.csv",
    )
    parser.add_argument("--batch", type=Path, default=process_root / "acquisition_batch_01.csv")
    parser.add_argument(
        "--manifest", type=Path, default=process_root / "source_acquisition_manifest.json"
    )
    parser.add_argument(
        "--inventory", type=Path, default=process_root / "matrices/source_inventory.csv"
    )
    args = parser.parse_args()
    if args.verify:
        result = verify(
            repository_root=repository_root,
            candidates_path=args.candidates.resolve(),
            batch_path=args.batch.resolve(),
            manifest_path=args.manifest.resolve(),
            inventory_path=args.inventory.resolve(),
        )
    else:
        result = register(
            repository_root=repository_root,
            candidates_path=args.candidates.resolve(),
            batch_path=args.batch.resolve(),
            manifest_path=args.manifest.resolve(),
            inventory_path=args.inventory.resolve(),
            download=args.download,
        )
        result = {
            "source_count": len(result["sources"]),
            "total_bytes": sum(item["bytes"] for item in result["sources"]),
            "registered": True,
        }
    print(_canonical_json(result), end="")


if __name__ == "__main__":
    main()
