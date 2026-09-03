from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_migration_manifest_matches_every_listed_repository_file() -> None:
    root = Path(__file__).parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "verify_migration_manifest.py"),
            "--repository-root",
            str(root),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(completed.stdout)
    assert result["ok"] is True
    assert result["listed_repository_files"] == 120
    assert result["verified_repository_files"] == 120
    assert result["errors"] == []
