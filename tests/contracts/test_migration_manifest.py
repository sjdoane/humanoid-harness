from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
BOUNDARY_PATHS = ("research/README.md", "docs/meetings/PUBLIC_CONTEXT_BOUNDARY.md")


def _run_verifier(
    manifest: Path, repository_root: Path
) -> tuple[subprocess.CompletedProcess[str], dict]:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "verify_migration_manifest.py"),
            "--manifest",
            str(manifest),
            "--repository-root",
            str(repository_root),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed, json.loads(completed.stdout)


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _minimal_v1_manifest(repository_root: Path) -> dict:
    copied = b"copied"
    authored = b"authored"
    (repository_root / "copied.txt").write_bytes(copied)
    (repository_root / "authored.txt").write_bytes(authored)
    return {
        "schema_version": 1,
        "migration": {"destination_root": "legacy"},
        "counts": {
            "copied_verbatim_files": 1,
            "sanitized_migrated_files": 0,
            "new_or_sanitized_files": 1,
            "paper_cards_including_template": 0,
            "structured_extractions_including_schema": 0,
        },
        "copied_files": [
            {
                "classification": "copied_verbatim",
                "source_path": "source/copied.txt",
                "destination_path": "copied.txt",
                "bytes": len(copied),
                "sha256": hashlib.sha256(copied).hexdigest(),
            }
        ],
        "authored_files": [
            {
                "classification": "new_migration_boundary",
                "path": "authored.txt",
                "bytes": len(authored),
                "sha256": hashlib.sha256(authored).hexdigest(),
            }
        ],
    }


def _repository_manifest(schema_version: int = 3) -> dict:
    manifest = json.loads((ROOT / "archive/MIGRATION_MANIFEST.json").read_text(encoding="utf-8"))
    manifest["schema_version"] = schema_version
    if schema_version == 2:
        for item in manifest["authored_files"]:
            if item["path"] in BOUNDARY_PATHS:
                del item["verification_path"]
    return manifest


def _materialize_snapshot(repository_root: Path, manifest: dict) -> None:
    original = _repository_manifest()
    source_paths = {
        item["path"]: item.get("verification_path", item["path"])
        for item in original["authored_files"]
    }
    for item in [*manifest["copied_files"], *manifest["authored_files"]]:
        logical = item.get("path", item.get("destination_path"))
        target = repository_root / item.get("verification_path", logical)
        source = ROOT / source_paths.get(logical, logical)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def test_migration_manifest_matches_every_listed_repository_file() -> None:
    manifest_path = ROOT / "archive/MIGRATION_MANIFEST.json"
    completed, result = _run_verifier(manifest_path, ROOT)
    assert completed.returncode == 0
    assert result["ok"] is True
    assert result["listed_repository_files"] == 120
    assert result["verified_repository_files"] == 120
    assert result["archived_snapshot_files"] == 14
    assert result["errors"] == []

    manifest = _repository_manifest()
    assert manifest["schema_version"] == 3
    archived = [item for item in manifest["authored_files"] if "verification_path" in item]
    assert len(archived) == 14
    assert all(
        item["verification_path"]
        == (
            "archive/initial_migration_boundaries/"
            if item["path"] in BOUNDARY_PATHS
            else "archive/initial_review_scaffold/"
        )
        + item["path"]
        for item in archived
    )
    assert {
        item["path"]: (item["bytes"], item["sha256"])
        for item in archived
        if item["path"] in BOUNDARY_PATHS
    } == {
        "research/README.md": (
            3012,
            "f463c050f8478afe1a12b064f65a984653f420ef12bd7475d894c0d0ecb08e5a",
        ),
        "docs/meetings/PUBLIC_CONTEXT_BOUNDARY.md": (
            1781,
            "60cf53ffeba6ca4ff9635915b596c45ce9842b597341e8b26801b608827f4db2",
        ),
    }


def test_schema_v2_still_verifies_the_original_twelve_redirect_contract(tmp_path: Path) -> None:
    manifest = _repository_manifest(schema_version=2)
    _materialize_snapshot(tmp_path, manifest)

    completed, result = _run_verifier(_write_manifest(tmp_path, manifest), tmp_path)

    assert completed.returncode == 0
    assert result["verified_repository_files"] == 120
    assert result["archived_snapshot_files"] == 12


def test_schema_v3_allows_live_boundary_updates_without_rewriting_receipts(tmp_path: Path) -> None:
    manifest = _repository_manifest()
    _materialize_snapshot(tmp_path, manifest)
    for relative in BOUNDARY_PATHS:
        live = tmp_path / relative
        live.parent.mkdir(parents=True, exist_ok=True)
        live.write_text("Current guidance, not the migration snapshot.\n", encoding="utf-8")

    completed, result = _run_verifier(_write_manifest(tmp_path, manifest), tmp_path)

    assert completed.returncode == 0
    assert result["verified_repository_files"] == 120
    assert result["archived_snapshot_files"] == 14


@pytest.mark.parametrize("relative", BOUNDARY_PATHS)
@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_schema_v3_rejects_missing_or_changed_boundary_snapshots(
    tmp_path: Path, relative: str, damage: str
) -> None:
    manifest = _repository_manifest()
    _materialize_snapshot(tmp_path, manifest)
    entry = next(item for item in manifest["authored_files"] if item["path"] == relative)
    archived = tmp_path / entry["verification_path"]
    original = archived.read_bytes()
    live = tmp_path / relative
    live.parent.mkdir(parents=True, exist_ok=True)
    live.write_bytes(original)
    if damage == "missing":
        archived.unlink()
        expected_error = f"{relative}: missing"
    else:
        archived.write_bytes(b"!" + original[1:])
        expected_error = f"{relative}: SHA-256 mismatch"

    completed, result = _run_verifier(_write_manifest(tmp_path, manifest), tmp_path)

    assert completed.returncode == 1
    assert expected_error in result["errors"]


@pytest.mark.parametrize("relative", BOUNDARY_PATHS)
def test_schema_v3_rejects_remapped_boundary_redirects(tmp_path: Path, relative: str) -> None:
    manifest = _repository_manifest()
    entry = next(item for item in manifest["authored_files"] if item["path"] == relative)
    entry["verification_path"] = relative

    completed, result = _run_verifier(_write_manifest(tmp_path, manifest), ROOT)

    assert completed.returncode == 1
    assert any(f"{relative}: verification_path must equal" in error for error in result["errors"])


@pytest.mark.parametrize("relative", BOUNDARY_PATHS)
def test_schema_v3_requires_both_boundary_redirects(tmp_path: Path, relative: str) -> None:
    manifest = _repository_manifest()
    entry = next(item for item in manifest["authored_files"] if item["path"] == relative)
    del entry["verification_path"]

    completed, result = _run_verifier(_write_manifest(tmp_path, manifest), ROOT)

    assert completed.returncode == 1
    assert any("schema_version 3 must redirect exactly" in e for e in result["errors"])


def test_schema_v2_rejects_the_additional_boundary_redirects(tmp_path: Path) -> None:
    manifest = _repository_manifest()
    manifest["schema_version"] = 2

    completed, result = _run_verifier(_write_manifest(tmp_path, manifest), ROOT)

    assert completed.returncode == 1
    assert "schema_version 2 must redirect exactly the initial review scaffold" in result["errors"]


@pytest.mark.parametrize("relative", ["../outside.txt", "/tmp/outside.txt"])
def test_migration_verifier_rejects_unsafe_repository_paths(tmp_path: Path, relative: str) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    data = b"outside"
    (tmp_path / "outside.txt").write_bytes(data)
    manifest = _minimal_v1_manifest(repository_root)
    manifest["copied_files"][0].update(
        {
            "destination_path": relative,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    )

    completed, result = _run_verifier(_write_manifest(tmp_path, manifest), repository_root)
    assert completed.returncode == 1
    assert result["ok"] is False
    assert any("unsafe verification path" in error for error in result["errors"])


@pytest.mark.parametrize("ancestor_symlink", [False, True])
def test_migration_verifier_rejects_symlink_targets(tmp_path: Path, ancestor_symlink: bool) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    data = b"outside copied file"
    (outside / "copied.txt").write_bytes(data)
    manifest = _minimal_v1_manifest(repository_root)
    if ancestor_symlink:
        (repository_root / "linked").symlink_to(outside, target_is_directory=True)
        destination = "linked/copied.txt"
    else:
        (repository_root / "linked.txt").symlink_to(outside / "copied.txt")
        destination = "linked.txt"
    manifest["copied_files"][0].update(
        {
            "destination_path": destination,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    )

    completed, result = _run_verifier(_write_manifest(tmp_path, manifest), repository_root)
    assert completed.returncode == 1
    assert result["ok"] is False
    assert any("unsafe verification path" in error for error in result["errors"])


def test_schema_v1_rejects_redirects(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    manifest = _minimal_v1_manifest(repository_root)
    manifest["authored_files"][0]["verification_path"] = "authored.txt"

    completed, result = _run_verifier(_write_manifest(tmp_path, manifest), repository_root)
    assert completed.returncode == 1
    assert "schema_version 1 must not declare verification_path" in result["errors"]


@pytest.mark.parametrize("schema_version", [None, [], {}, "3", True])
def test_invalid_schema_versions_fail_with_a_structured_error(
    tmp_path: Path, schema_version: object
) -> None:
    manifest = _repository_manifest()
    manifest["schema_version"] = schema_version

    completed, result = _run_verifier(_write_manifest(tmp_path, manifest), ROOT)

    assert completed.returncode == 1
    assert "unsupported migration manifest schema_version" in result["errors"]


@pytest.mark.parametrize("schema_version", [2, 3])
def test_versioned_manifest_rejects_a_remapped_scaffold_redirect(
    tmp_path: Path, schema_version: int
) -> None:
    manifest = _repository_manifest(schema_version)
    entry = next(
        item for item in manifest["authored_files"] if item["path"].endswith("candidate_papers.csv")
    )
    entry["verification_path"] = entry["verification_path"].replace(
        "candidate_papers.csv", "search_log.csv"
    )

    completed, result = _run_verifier(_write_manifest(tmp_path, manifest), ROOT)
    assert completed.returncode == 1
    assert any("verification_path must equal" in error for error in result["errors"])


@pytest.mark.parametrize("schema_version", [2, 3])
def test_versioned_manifest_requires_the_exact_scaffold_redirect_set(
    tmp_path: Path, schema_version: int
) -> None:
    manifest = _repository_manifest(schema_version)
    entry = next(
        item for item in manifest["authored_files"] if item["path"].endswith("citation_ledger.csv")
    )
    del entry["verification_path"]

    completed, result = _run_verifier(_write_manifest(tmp_path, manifest), ROOT)
    assert completed.returncode == 1
    assert any(
        f"schema_version {schema_version} must redirect exactly" in e for e in result["errors"]
    )


@pytest.mark.parametrize("schema_version", [2, 3])
def test_versioned_manifest_rejects_copied_file_redirects(
    tmp_path: Path, schema_version: int
) -> None:
    manifest = _repository_manifest(schema_version)
    copied = manifest["copied_files"][0]
    copied["verification_path"] = copied["destination_path"]

    completed, result = _run_verifier(_write_manifest(tmp_path, manifest), ROOT)
    assert completed.returncode == 1
    assert "copied_files must not declare verification_path" in result["errors"]


def test_schema_v3_rejects_redirecting_another_authored_document(tmp_path: Path) -> None:
    manifest = _repository_manifest()
    manifest["authored_files"].append(
        {
            "classification": "new_migration_boundary",
            "path": "unapproved.md",
            "verification_path": "archive/initial_migration_boundaries/unapproved.md",
            "bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        }
    )
    manifest["counts"]["new_or_sanitized_files"] += 1

    completed, result = _run_verifier(_write_manifest(tmp_path, manifest), ROOT)

    assert completed.returncode == 1
    assert any("schema_version 3 must redirect exactly" in e for e in result["errors"])


def test_migration_verifier_rejects_duplicate_logical_and_verification_paths(
    tmp_path: Path,
) -> None:
    manifest = _repository_manifest()
    manifest["copied_files"][1] = dict(manifest["copied_files"][0])

    completed, result = _run_verifier(_write_manifest(tmp_path, manifest), ROOT)
    assert completed.returncode == 1
    assert any("duplicate logical path" in error for error in result["errors"])
    assert any("duplicate verification path" in error for error in result["errors"])


def test_migration_verifier_reconciles_declared_counts(tmp_path: Path) -> None:
    manifest = _repository_manifest()
    manifest["counts"]["copied_verbatim_files"] += 1

    completed, result = _run_verifier(_write_manifest(tmp_path, manifest), ROOT)
    assert completed.returncode == 1
    assert "counts.copied_verbatim_files must equal 104" in result["errors"]


@pytest.mark.parametrize("key", ["copied_files", "authored_files"])
def test_migration_verifier_requires_nonempty_entry_lists(tmp_path: Path, key: str) -> None:
    manifest = _repository_manifest()
    manifest[key] = []

    completed, result = _run_verifier(_write_manifest(tmp_path, manifest), ROOT)
    assert completed.returncode == 1
    assert f"{key} must be a nonempty list" in result["errors"]
