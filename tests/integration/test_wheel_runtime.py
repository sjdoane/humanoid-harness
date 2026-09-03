from __future__ import annotations

import hashlib
import json
import os
import site
import subprocess
import sys
from pathlib import Path

import pytest

from oracle_composition.experiments.fixed_reference import load_study_design
from oracle_composition.experiments.fixed_reference_runner import inspect_runtime

ROOT = Path(__file__).resolve().parents[2]
DESIGN_PATH = (
    ROOT
    / "experiments"
    / "001_humanoid_fixed_reference"
    / "configs"
    / "static_stand_precalibration_v0.study.json"
)


@pytest.mark.gym
def test_installed_wheel_cli_and_runtime_use_packaged_dependency_lock(tmp_path: Path) -> None:
    """Prove the advertised CLI works without repository-relative package files."""

    expected_runtime_sha256 = inspect_runtime(load_study_design(DESIGN_PATH)).sha256
    distribution_dir = tmp_path / "distribution"
    build = subprocess.run(
        ["uv", "build", "--out-dir", str(distribution_dir)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    wheels = tuple(distribution_dir.glob("*.whl"))
    assert len(wheels) == 1

    environment_dir = tmp_path / "clean-wheel-environment"
    create_environment = subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(environment_dir)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert create_environment.returncode == 0, create_environment.stderr
    environment_python = environment_dir / "bin" / "python"
    install = subprocess.run(
        ["uv", "pip", "install", "--python", str(environment_python), "--no-deps", str(wheels[0])],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, install.stderr

    inherited_dependencies = os.pathsep.join(site.getsitepackages())
    environment = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": inherited_dependencies,
    }
    command = environment_dir / "bin" / "humanoid-fixed-reference"
    help_result = subprocess.run(
        [str(command), "--help"],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "fail-closed calibration" in help_result.stdout.lower()

    expected_lock_sha256 = hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest()
    probe = """
import hashlib
import json
import sys
from pathlib import Path

import oracle_composition
from oracle_composition.experiments.fixed_reference import load_study_design
from oracle_composition.experiments.fixed_reference_runner import (
    _dependency_lock_path,
    inspect_runtime,
)

environment_root = Path(sys.argv[1]).resolve()
package_path = Path(oracle_composition.__file__).resolve()
if environment_root not in package_path.parents:
    raise RuntimeError(f"oracle_composition did not load from wheel environment: {package_path}")
lock_path = _dependency_lock_path().resolve()
if environment_root not in lock_path.parents or lock_path.parts[-2:] != ("_runtime", "uv.lock"):
    raise RuntimeError(f"runtime lock did not load from wheel: {lock_path}")
lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
if lock_sha256 != sys.argv[3]:
    raise RuntimeError("packaged dependency lock differs from repository lock")
runtime = inspect_runtime(load_study_design(Path(sys.argv[2])))
print(json.dumps({
    "dependency_lock_sha256": runtime.dependency_lock_sha256,
    "package_path": str(package_path),
    "runtime_sha256": runtime.sha256,
}))
"""
    runtime_result = subprocess.run(
        [
            str(environment_python),
            "-c",
            probe,
            str(environment_dir),
            str(DESIGN_PATH),
            expected_lock_sha256,
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert runtime_result.returncode == 0, runtime_result.stderr
    observed = json.loads(runtime_result.stdout)
    assert observed["dependency_lock_sha256"] == expected_lock_sha256
    assert str(environment_dir.resolve()) in observed["package_path"]
    assert observed["runtime_sha256"] == expected_runtime_sha256
