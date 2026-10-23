"""Software-only source guards; no retained donors, simulator, or conversion runs."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PRODUCER = Path("scripts/convert_study012_reference_candidate.py")
BINDINGS = next(
    ast.literal_eval(node.value)
    for node in ast.parse((ROOT / PRODUCER).read_text()).body
    if isinstance(node, ast.Assign)
    and any(
        isinstance(target, ast.Name) and target.id == "SOURCE_BINDINGS" for target in node.targets
    )
)
PROBE = """
import importlib.util
import json
import sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('converter_under_test', sys.argv[1])
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)
mode = sys.argv[3]
module_name = 'oracle_composition.adapters.gmt.course_task'
if mode == 'missing':
    del sys.modules[module_name]
elif mode == 'outside':
    sys.modules[module_name].__file__ = str(Path(sys.argv[1]).parent / 'course_task.py')
elif mode == 'wrong_inside':
    sys.modules[module_name].__file__ = sys.modules['oracle_composition.adapters.gmt.contracts'].__file__
print(json.dumps(converter.source_identity(sys.argv[2]), sort_keys=True))
"""


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", "-C", str(root), *arguments), check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(root: Path) -> str:
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Software Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "-qm",
        "synthetic source fixture",
    )
    return _git(root, "rev-parse", "HEAD")


@pytest.fixture
def checkout(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "public-source-copy"
    for relative in [PRODUCER, *(Path("src") / path for path in BINDINGS)]:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    _git(root, "init", "-q")
    return root, _commit(root)


def _probe(checkout: tuple[Path, str], mode: str = "normal") -> subprocess.CompletedProcess[str]:
    root, commit = checkout
    return subprocess.run(
        (sys.executable, "-B", "-c", PROBE, str(root / PRODUCER), commit, mode),
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src"), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        check=False,
    )


def test_current_epoch_hashes_actual_imports_and_producer(checkout: tuple[Path, str]) -> None:
    root, commit = checkout
    result = _probe(checkout)
    assert result.returncode == 0, result.stderr
    source = json.loads(result.stdout)
    assert source["binding_version"] == "study012-converter-source/v2"
    assert len(source["imports"]) == 14
    expected = {
        relative: hashlib.sha256((root / "src" / relative).read_bytes()).hexdigest()
        for relative in BINDINGS
    }
    assert source["imports"] == expected
    producer_digest = hashlib.sha256((root / PRODUCER).read_bytes()).hexdigest()
    assert source["producer"] == {"path": str(PRODUCER), "sha256": producer_digest}
    executed = {**expected, str(PRODUCER): producer_digest}
    assert (
        source["executed_source_set_sha256"]
        == hashlib.sha256(
            json.dumps(executed, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )
    assert source["reviewed_checkout"]["commit"] == commit
    assert source["reviewed_checkout"]["checked_before_donor_reads_or_output_writes"] is True


@pytest.mark.parametrize(
    "mode,message",
    [
        ("missing", "expected source module was not imported"),
        ("outside", "project import escaped exact source root"),
        ("wrong_inside", "project import escaped exact source root"),
    ],
)
def test_import_identity_rejects_substitution(checkout, mode: str, message: str) -> None:
    result = _probe(checkout, mode)
    assert result.returncode != 0
    assert message in result.stderr


def test_committed_single_byte_dependency_change_is_rejected(checkout) -> None:
    root, _ = checkout
    path = root / "src/oracle_composition/adapters/gmt/course_task.py"
    path.write_bytes(path.read_bytes() + b"\n")
    changed_commit = _commit(root)
    result = _probe((root, changed_commit))
    assert result.returncode != 0
    assert "source digest mismatch: oracle_composition/adapters/gmt/course_task.py" in result.stderr


@pytest.mark.parametrize("change", ["unstaged", "staged", "untracked"])
def test_dirty_checkout_is_rejected_before_identity(checkout, change: str) -> None:
    root, _ = checkout
    if change == "untracked":
        (root / "unreviewed.txt").write_text("unreviewed\n")
    else:
        path = root / PRODUCER
        path.write_bytes(path.read_bytes() + b"\n")
        if change == "staged":
            _git(root, "add", str(PRODUCER))
    result = _probe(checkout)
    assert result.returncode != 0
    assert "reviewed source checkout is not clean" in result.stderr


@pytest.mark.parametrize(
    "commit,message",
    [
        ("c9ee2735829af75559fc066da6a0bd502e372667", "reviewed source commit differs"),
        ("c9ee273", "reviewed source commit must be one full lowercase Git SHA"),
    ],
)
def test_current_epoch_cannot_claim_historical_or_abbreviated_commit(
    checkout, commit, message
) -> None:
    result = _probe((checkout[0], commit))
    assert result.returncode != 0
    assert message in result.stderr
