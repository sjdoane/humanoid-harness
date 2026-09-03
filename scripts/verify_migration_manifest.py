#!/usr/bin/env python3
"""Verify byte-level migration receipts without modifying either repository."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

_REVIEW_ROOT = PurePosixPath("research/literature_review/01_process/001_oracle_composition")
_ARCHIVE_ROOT = PurePosixPath("archive/initial_review_scaffold")
_ARCHIVED_REVIEW_PATHS = frozenset(
    {
        (_REVIEW_ROOT / "README.md").as_posix(),
        *(
            (_REVIEW_ROOT / "matrices" / name).as_posix()
            for name in (
                "candidate_papers.csv",
                "citation_ledger.csv",
                "coding_taxonomy.csv",
                "extraction_table.csv",
                "reference_table.csv",
                "screening_table.csv",
                "search_log.csv",
                "selection_decisions.csv",
                "source_inventory.csv",
                "source_triage.csv",
                "synthesis_matrix.csv",
            )
        ),
    }
)
_COPIED_CLASSIFICATIONS = frozenset({"copied_verbatim", "sanitized_private_context_label"})
_AUTHORED_CLASSIFICATIONS = frozenset(
    {
        "new_empty_review_table",
        "new_migration_boundary",
        "new_process_pointer",
        "new_sanitized_context_boundary",
    }
)
_COUNT_KEYS = frozenset(
    {
        "copied_verbatim_files",
        "sanitized_migrated_files",
        "new_or_sanitized_files",
        "paper_cards_including_template",
        "structured_extractions_including_schema",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(
    manifest_path: Path,
    *,
    repository_root: Path,
    source_workspace: Path | None = None,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    verified = 0
    archived_snapshot_files = 0

    if not isinstance(manifest, dict):
        return _result(
            manifest_path,
            listed_repository_files=0,
            verified_repository_files=0,
            archived_snapshot_files=0,
            supplemental_verified=0,
            errors=["manifest root must be an object"],
        )

    schema_version = manifest.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version not in {1, 2}
    ):
        errors.append("unsupported migration manifest schema_version")

    copied_files = _required_entry_list(manifest, "copied_files", errors)
    authored_files = _required_entry_list(manifest, "authored_files", errors)
    supplemental_receipts = _optional_entry_list(manifest, "supplemental_source_receipts", errors)
    _validate_classifications(copied_files, authored_files, errors)
    _validate_counts(manifest, copied_files, authored_files, errors)
    _validate_redirect_contract(schema_version, copied_files, authored_files, errors)

    entries: list[tuple[str, dict[str, Any]]] = []
    entries.extend(("copied", item) for item in copied_files)
    entries.extend(("authored", item) for item in authored_files)
    logical_paths: dict[str, str] = {}
    verification_paths: dict[str, str] = {}
    for kind, item in entries:
        relative = item.get("destination_path") if kind == "copied" else item.get("path")
        if not isinstance(relative, str) or not relative:
            errors.append(f"{kind} entry has no path")
            continue
        _record_unique_path(relative, kind, "logical", logical_paths, errors)

        verification_relative = relative
        expected_verification = _expected_verification_path(relative)
        if (
            schema_version == 2
            and kind == "authored"
            and relative in _ARCHIVED_REVIEW_PATHS
            and item.get("verification_path") == expected_verification
        ):
            verification_relative = expected_verification
            archived_snapshot_files += 1

        _record_unique_path(
            verification_relative,
            relative,
            "verification",
            verification_paths,
            errors,
        )
        path = _bounded_repository_path(repository_root, verification_relative)
        if path is None:
            errors.append(f"{relative}: unsafe verification path {verification_relative!r}")
            continue
        error_count = len(errors)
        _verify_file(path, item, relative, errors)
        if path.is_file() and len(errors) == error_count:
            verified += 1

    supplemental_verified = 0
    if source_workspace is not None:
        for item in supplemental_receipts:
            relative = item.get("source_path")
            if not isinstance(relative, str) or not relative:
                errors.append("supplemental entry has no source_path")
                continue
            path = _bounded_repository_path(source_workspace, relative)
            if path is None:
                errors.append(f"{relative}: unsafe supplemental source path")
                continue
            error_count = len(errors)
            _verify_file(path, item, relative, errors)
            if path.is_file() and len(errors) == error_count:
                supplemental_verified += 1

    return _result(
        manifest_path,
        listed_repository_files=len(entries),
        verified_repository_files=verified,
        archived_snapshot_files=archived_snapshot_files,
        supplemental_verified=supplemental_verified,
        errors=errors,
    )


def _result(
    manifest_path: Path,
    *,
    listed_repository_files: int,
    verified_repository_files: int,
    archived_snapshot_files: int,
    supplemental_verified: int,
    errors: list[str],
) -> dict[str, Any]:
    return {
        "ok": not errors,
        "manifest": str(manifest_path),
        "listed_repository_files": listed_repository_files,
        "verified_repository_files": verified_repository_files,
        "archived_snapshot_files": archived_snapshot_files,
        "supplemental_verified": supplemental_verified,
        "errors": errors,
    }


def _required_entry_list(
    manifest: dict[str, Any],
    key: str,
    errors: list[str],
) -> list[dict[str, Any]]:
    value = manifest.get(key)
    if not isinstance(value, list) or not value:
        errors.append(f"{key} must be a nonempty list")
        return []
    valid: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"{key}[{index}] must be an object")
        else:
            valid.append(item)
    return valid


def _optional_entry_list(
    manifest: dict[str, Any],
    key: str,
    errors: list[str],
) -> list[dict[str, Any]]:
    value = manifest.get(key, [])
    if not isinstance(value, list):
        errors.append(f"{key} must be a list")
        return []
    valid: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            errors.append(f"{key}[{index}] must be an object")
        else:
            valid.append(item)
    return valid


def _validate_classifications(
    copied_files: list[dict[str, Any]],
    authored_files: list[dict[str, Any]],
    errors: list[str],
) -> None:
    for index, item in enumerate(copied_files):
        if item.get("classification") not in _COPIED_CLASSIFICATIONS:
            errors.append(f"copied_files[{index}]: unsupported classification")
    for index, item in enumerate(authored_files):
        if item.get("classification") not in _AUTHORED_CLASSIFICATIONS:
            errors.append(f"authored_files[{index}]: unsupported classification")


def _validate_counts(
    manifest: dict[str, Any],
    copied_files: list[dict[str, Any]],
    authored_files: list[dict[str, Any]],
    errors: list[str],
) -> None:
    counts = manifest.get("counts")
    if not isinstance(counts, dict):
        errors.append("counts must be an object")
        return
    if set(counts) != _COUNT_KEYS:
        errors.append("counts must contain exactly the schema count keys")

    classifications = Counter(item.get("classification") for item in copied_files)
    migration = manifest.get("migration")
    destination_root = migration.get("destination_root") if isinstance(migration, dict) else None
    if not isinstance(destination_root, str) or not destination_root:
        errors.append("migration.destination_root must be a nonempty string")
        destination_root = ""
    card_prefix = f"{destination_root}/cards/"
    extraction_prefix = f"{destination_root}/extractions/"
    copied_destinations = [item.get("destination_path") for item in copied_files]
    expected = {
        "copied_verbatim_files": classifications["copied_verbatim"],
        "sanitized_migrated_files": classifications["sanitized_private_context_label"],
        "new_or_sanitized_files": (
            len(authored_files) + classifications["sanitized_private_context_label"]
        ),
        "paper_cards_including_template": sum(
            isinstance(path, str) and path.startswith(card_prefix) and path.endswith(".md")
            for path in copied_destinations
        ),
        "structured_extractions_including_schema": sum(
            isinstance(path, str) and path.startswith(extraction_prefix) and path.endswith(".json")
            for path in copied_destinations
        ),
    }
    for key, expected_value in expected.items():
        value = counts.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value != expected_value:
            errors.append(f"counts.{key} must equal {expected_value}")


def _validate_redirect_contract(
    schema_version: object,
    copied_files: list[dict[str, Any]],
    authored_files: list[dict[str, Any]],
    errors: list[str],
) -> None:
    copied_redirects = [item for item in copied_files if "verification_path" in item]
    if copied_redirects:
        errors.append("copied_files must not declare verification_path")

    authored_redirect_paths = {
        relative
        for item in authored_files
        if "verification_path" in item and isinstance((relative := item.get("path")), str)
    }
    if schema_version == 1:
        if authored_redirect_paths:
            errors.append("schema_version 1 must not declare verification_path")
        return
    if schema_version != 2:
        return

    if authored_redirect_paths != _ARCHIVED_REVIEW_PATHS:
        errors.append("schema_version 2 must redirect exactly the initial review scaffold")
    for item in authored_files:
        relative = item.get("path")
        if relative not in _ARCHIVED_REVIEW_PATHS:
            continue
        expected = _expected_verification_path(relative)
        if item.get("verification_path") != expected:
            errors.append(f"{relative}: verification_path must equal {expected!r}")


def _expected_verification_path(relative: str) -> str:
    return (_ARCHIVE_ROOT / PurePosixPath(relative)).as_posix()


def _record_unique_path(
    relative: str,
    owner: str,
    description: str,
    seen: dict[str, str],
    errors: list[str],
) -> None:
    parsed = PurePosixPath(relative)
    canonical = parsed.as_posix()
    previous = seen.get(canonical)
    if previous is not None:
        errors.append(
            f"{owner}: duplicate {description} path {relative!r}; already used by {previous}"
        )
    else:
        seen[canonical] = owner


def _bounded_repository_path(root: Path, relative: str) -> Path | None:
    parsed = PurePosixPath(relative)
    if (
        parsed.is_absolute()
        or not parsed.parts
        or ".." in parsed.parts
        or parsed.as_posix() != relative
    ):
        return None
    path = root
    for part in parsed.parts:
        path /= part
        if path.is_symlink():
            return None
    return path


def _verify_file(
    path: Path,
    item: dict[str, Any],
    label: str,
    errors: list[str],
) -> None:
    if path.is_symlink() or not path.is_file():
        errors.append(f"{label}: missing")
        return
    expected_bytes = item.get("bytes")
    if path.stat().st_size != expected_bytes:
        errors.append(f"{label}: size mismatch")
    if _sha256(path) != item.get("sha256"):
        errors.append(f"{label}: SHA-256 mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("archive/MIGRATION_MANIFEST.json"),
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-workspace", type=Path)
    args = parser.parse_args()

    result = verify_manifest(
        args.manifest.resolve(),
        repository_root=args.repository_root.resolve(),
        source_workspace=(args.source_workspace.resolve() if args.source_workspace else None),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
