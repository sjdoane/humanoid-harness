from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "research/source_motions/deepmimic_humanoid3d"
MANIFEST_PATH = SOURCE_ROOT / "source_manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob_sha1(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()


def test_deepmimic_source_motion_manifest_reverifies_exact_bytes() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    assert set(manifest) == {
        "schema_version",
        "source_repository",
        "source_commit",
        "source_commit_date_utc",
        "registered_date",
        "license",
        "legacy_provenance",
        "files",
        "claim_boundary",
    }
    assert manifest["schema_version"] == 1
    source_commit = manifest["source_commit"]
    assert len(source_commit) == 40
    assert all(character in "0123456789abcdef" for character in source_commit)

    expected_paths = {
        "raw/humanoid3d_getup_facedown.txt",
        "raw/humanoid3d_getup_faceup.txt",
        "raw/humanoid3d_run.txt",
        "raw/humanoid3d_walk.txt",
    }
    records = manifest["files"]
    assert {record["path"] for record in records} == expected_paths
    for record in records:
        assert set(record) == {"path", "bytes", "sha256", "source_url"}
        relative = PurePosixPath(record["path"])
        assert not relative.is_absolute()
        assert ".." not in relative.parts
        source_path = SOURCE_ROOT.joinpath(*relative.parts)
        assert source_path.is_file()
        assert source_path.stat().st_size == record["bytes"]
        assert _sha256(source_path) == record["sha256"]
        assert f"/{source_commit}/" in record["source_url"]

    license_record = manifest["license"]
    license_path = ROOT / license_record["repository_path"]
    assert license_record["spdx_id"] == "MIT"
    assert license_path.is_file()
    assert _git_blob_sha1(license_path) == license_record["upstream_git_blob_sha1"]


def test_deepmimic_source_motion_manifest_preserves_nonadmission_boundary() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    boundary = manifest["claim_boundary"]
    for phrase in (
        "Source archive only",
        "no Gymnasium Humanoid joint mapping",
        "controller compatibility",
        "dynamics feasibility",
        "behavioral evidence",
    ):
        assert phrase in boundary
