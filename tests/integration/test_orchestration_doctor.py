from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCTOR = REPO_ROOT / "scripts" / "orchestration-doctor"


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _fake_claude(bin_dir: Path) -> None:
    _write_executable(
        bin_dir / "claude",
        """#!/bin/sh
case "$1" in
  --version)
    printf '%s\n' '2.1.260 (Claude Code)'
    ;;
  auth)
    printf '%s\n' '{"loggedIn":true}'
    ;;
  plugin)
    printf '%s\n' '[{"id":"codex@openai-codex","enabled":true}]'
    ;;
  *)
    exit 2
    ;;
esac
""",
    )


def _fake_codex(bin_dir: Path, *, commands_succeed: bool) -> None:
    exit_line = "exit 0" if commands_succeed else "exit 1"
    _write_executable(
        bin_dir / "codex",
        f"""#!/bin/sh
case "$1" in
  --version)
    printf '%s\n' 'codex-cli 0.153.2'
    {exit_line}
    ;;
  login)
    printf '%s\n' 'Logged in using ChatGPT'
    {exit_line}
    ;;
  debug)
    printf '%s\n' '{{"models":[{{"slug":"gpt-5.6-sol","supported_reasoning_levels":[{{"effort":"max"}}]}}]}}'
    {exit_line}
    ;;
  *)
    exit 2
    ;;
esac
""",
    )


def _doctor_environment(bin_dir: Path, marker: Path) -> dict[str, str]:
    jq_path = shutil.which("jq")
    if jq_path is not None:
        (bin_dir / "jq").symlink_to(jq_path)
    _write_executable(bin_dir / "screen", "#!/bin/sh\nexit 0\n")
    _write_executable(
        bin_dir / "rg",
        "#!/bin/sh\nprintf '%s\\n' called >\"$HUMANOID_RG_MARKER\"\nexit 99\n",
    )
    environment = os.environ.copy()
    environment["PATH"] = f"{bin_dir}:/usr/bin:/bin:/usr/sbin:/sbin"
    environment["HUMANOID_RG_MARKER"] = str(marker)
    return environment


@pytest.mark.skipif(shutil.which("jq") is None, reason="doctor requires jq")
def test_doctor_does_not_call_ripgrep(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "rg-called"
    _fake_claude(bin_dir)
    _fake_codex(bin_dir, commands_succeed=True)

    completed = subprocess.run(
        [str(DOCTOR)],
        cwd=REPO_ROOT,
        env=_doctor_environment(bin_dir, marker),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PASS | Orchestration preflight is ready" in completed.stdout
    assert not marker.exists()


@pytest.mark.skipif(shutil.which("jq") is None, reason="doctor requires jq")
def test_doctor_rejects_convincing_output_from_failed_codex_commands(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "rg-called"
    _fake_claude(bin_dir)
    _fake_codex(bin_dir, commands_succeed=False)

    completed = subprocess.run(
        [str(DOCTOR)],
        cwd=REPO_ROOT,
        env=_doctor_environment(bin_dir, marker),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "FAIL | Codex executable is present but its version check failed" in completed.stdout
    assert "FAIL | Codex ChatGPT login is unavailable" in completed.stdout
    assert "FAIL | Codex does not advertise gpt-5.6-sol at max reasoning" in completed.stdout
    assert "PASS | Codex uses the saved ChatGPT login" not in completed.stdout
    assert "PASS | Codex advertises gpt-5.6-sol at max reasoning" not in completed.stdout
    assert not marker.exists()


@pytest.mark.parametrize(
    ("config", "expected_pass"),
    [
        (
            'model = "gpt-5.6-sol"\n'
            'model_reasoning_effort = "max"\n\n'
            "[features]\n"
            "multi_agent = false\n",
            True,
        ),
        (
            "[wrong]\n"
            'model = "gpt-5.6-sol"\n'
            'model_reasoning_effort = "max"\n\n'
            "[features]\n"
            "multi_agent = false\n",
            False,
        ),
        (
            'model = "gpt-5.6-sol"\n'
            'model_reasoning_effort = "max"\n\n'
            "[features]\n"
            "multi_agent = false\n"
            "multi_agent = true\n",
            False,
        ),
    ],
)
def test_doctor_checks_codex_policy_in_effective_sections(
    tmp_path: Path, config: str, expected_pass: bool
) -> None:
    repo = tmp_path / "repo"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    doctor = scripts / "orchestration-doctor"
    shutil.copy2(DOCTOR, doctor)
    config_path = repo / ".codex" / "config.toml"
    config_path.parent.mkdir()
    config_path.write_text(config, encoding="utf-8")

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "rg-called"
    _write_executable(bin_dir / "claude", "#!/bin/sh\nexit 1\n")
    _write_executable(bin_dir / "codex", "#!/bin/sh\nexit 1\n")

    completed = subprocess.run(
        [str(doctor)],
        cwd=repo,
        env=_doctor_environment(bin_dir, marker),
        check=False,
        capture_output=True,
        text=True,
    )

    policy_passed = (
        "PASS | Project requests Sol workers at max reasoning without nested delegation"
        in completed.stdout
    )
    assert policy_passed is expected_pass
    assert not marker.exists()
