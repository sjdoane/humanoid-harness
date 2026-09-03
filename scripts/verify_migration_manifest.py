#!/usr/bin/env python3
"""Verify byte-level migration receipts without modifying either repository."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


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

    entries: list[tuple[str, dict[str, Any]]] = []
    entries.extend(("copied", item) for item in manifest.get("copied_files", []))
    entries.extend(("authored", item) for item in manifest.get("authored_files", []))
    for kind, item in entries:
        relative = item.get("destination_path") if kind == "copied" else item.get("path")
        if not isinstance(relative, str) or not relative:
            errors.append(f"{kind} entry has no path")
            continue
        path = repository_root / relative
        _verify_file(path, item, relative, errors)
        if path.is_file() and not any(error.startswith(f"{relative}:") for error in errors):
            verified += 1

    supplemental_verified = 0
    if source_workspace is not None:
        for item in manifest.get("supplemental_source_receipts", []):
            relative = item.get("source_path")
            if not isinstance(relative, str) or not relative:
                errors.append("supplemental entry has no source_path")
                continue
            path = source_workspace / relative
            _verify_file(path, item, relative, errors)
            if path.is_file() and not any(error.startswith(f"{relative}:") for error in errors):
                supplemental_verified += 1

    return {
        "ok": not errors,
        "manifest": str(manifest_path),
        "listed_repository_files": len(entries),
        "verified_repository_files": verified,
        "supplemental_verified": supplemental_verified,
        "errors": errors,
    }


def _verify_file(
    path: Path,
    item: dict[str, Any],
    label: str,
    errors: list[str],
) -> None:
    if not path.is_file():
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
