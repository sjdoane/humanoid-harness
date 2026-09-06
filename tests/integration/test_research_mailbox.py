from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MAILBOX = REPO_ROOT / "scripts" / "research-mailbox"
RESOURCE_SLOT = REPO_ROOT / "src" / "oracle_composition" / "harness" / "resource_slot.py"
UTC = timezone.utc  # noqa: UP017 -- CLI also targets macOS system Python 3.9.


def run_mailbox(
    root: Path | None, *arguments: str, script: Path = MAILBOX
) -> subprocess.CompletedProcess[str]:
    command = [str(script)]
    if root is not None:
        command.extend(("--root", str(root)))
    command.extend(arguments)
    return subprocess.run(command, check=False, capture_output=True, text=True)


def succeed(root: Path | None, *arguments: str, script: Path = MAILBOX) -> dict[str, object]:
    completed = run_mailbox(root, *arguments, script=script)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return json.loads(completed.stdout)


def send(
    root: Path | None,
    subject: str,
    *,
    kind: str = "progress",
    script: Path = MAILBOX,
) -> dict[str, object]:
    return succeed(
        root,
        "send",
        "--from",
        "astra",
        "--to",
        "fable",
        "--kind",
        kind,
        "--subject",
        subject,
        "--body",
        f"body for {subject}",
        script=script,
    )


def git(*arguments: str, cwd: Path, environment: dict[str, str] | None = None) -> None:
    completed = subprocess.run(
        ["git", *arguments], cwd=cwd, env=environment, check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_two_git_worktrees_share_the_git_common_directory(tmp_path: Path) -> None:
    assert os.access(MAILBOX, os.X_OK)
    primary = tmp_path / "primary"
    peer = tmp_path / "peer"
    (primary / "scripts").mkdir(parents=True)
    (primary / "src" / "oracle_composition" / "harness").mkdir(parents=True)
    shutil.copy2(MAILBOX, primary / "scripts" / "research-mailbox")
    shutil.copy2(
        RESOURCE_SLOT,
        primary / "src" / "oracle_composition" / "harness" / "resource_slot.py",
    )
    git("init", cwd=primary)
    git(
        "add",
        "scripts/research-mailbox",
        "src/oracle_composition/harness/resource_slot.py",
        cwd=primary,
    )
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_AUTHOR_NAME": "Mailbox Test",
            "GIT_AUTHOR_EMAIL": "mailbox@example.invalid",
            "GIT_COMMITTER_NAME": "Mailbox Test",
            "GIT_COMMITTER_EMAIL": "mailbox@example.invalid",
        }
    )
    git("commit", "-m", "fixture", cwd=primary, environment=environment)
    git("worktree", "add", "-b", "peer", str(peer), cwd=primary)

    message = send(None, "shared", script=primary / "scripts" / "research-mailbox")
    inbox = succeed(None, "inbox", "fable", script=peer / "scripts" / "research-mailbox")
    board = succeed(None, "board", script=peer / "scripts" / "research-mailbox")

    assert [item["id"] for item in inbox["messages"]] == [message["id"]]  # type: ignore[index]
    assert board["heavy_job_slot"] == {"state": "free"}
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=peer,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert (Path(common) / "harness-coordination" / "messages" / f"{message['id']}.json").is_file()


def test_concurrent_sends_are_complete_and_unique(tmp_path: Path) -> None:
    root = tmp_path / "mailbox"
    with ThreadPoolExecutor(max_workers=12) as pool:
        messages = list(pool.map(lambda number: send(root, f"message-{number}"), range(36)))

    ids = {message["id"] for message in messages}
    inbox = succeed(root, "inbox", "fable")
    assert len(ids) == 36
    assert {message["id"] for message in inbox["messages"]} == ids  # type: ignore[index]
    assert len(list((root / "messages").glob("*.json"))) == 36


def test_inbox_does_not_ack_and_ack_is_recipient_only_and_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "mailbox"
    message = send(root, "read semantics")
    first_inbox = succeed(root, "inbox", "fable")
    second_inbox = succeed(root, "inbox", "fable")
    assert first_inbox == second_inbox
    assert first_inbox["messages"][0]["read_acknowledgment"] is None  # type: ignore[index]
    assert not list((root / "acks").glob("*.json"))

    wrong = run_mailbox(root, "ack", "astra", str(message["id"]))
    assert wrong.returncode == 2
    assert "only recipient fable" in wrong.stderr
    first = succeed(root, "ack", "fable", str(message["id"]))
    second = succeed(root, "ack", "fable", str(message["id"]))
    assert first["created"] is True
    assert second["created"] is False
    assert first["acknowledgment"] == second["acknowledgment"]
    assert first["acknowledgment"]["meaning"] == "read_not_agreement"  # type: ignore[index]
    assert succeed(root, "inbox", "fable")["messages"] == []
    assert len(succeed(root, "inbox", "fable", "--all")["messages"]) == 1  # type: ignore[arg-type]

    (root / "acks" / f"{message['id']}.json").write_text("{}\n", encoding="utf-8")
    assert run_mailbox(root, "inbox", "fable", "--all").returncode == 2


def test_acceptance_is_an_explicit_reply_to_a_proposal(tmp_path: Path) -> None:
    root = tmp_path / "mailbox"
    original = send(root, "proposal", kind="proposal")
    succeed(root, "ack", "fable", str(original["id"]))
    reply = succeed(
        root,
        "send",
        "--from",
        "fable",
        "--to",
        "astra",
        "--kind",
        "acceptance",
        "--subject",
        "Agreed",
        "--body",
        "I agree.",
        "--reply-to",
        str(original["id"]),
    )
    assert reply["reply_to"] == original["id"]
    assert succeed(root, "inbox", "astra")["messages"][0]["id"] == reply["id"]  # type: ignore[index]


def test_decisions_require_an_existing_proposal_reply_target(tmp_path: Path) -> None:
    root = tmp_path / "mailbox"
    non_proposal = send(root, "ordinary progress")
    for kind in ("acceptance", "rejection", "counterproposal"):
        missing = run_mailbox(
            root,
            "send",
            "--from",
            "fable",
            "--to",
            "astra",
            "--kind",
            kind,
            "--subject",
            kind,
            "--body",
            kind,
        )
        wrong_kind = run_mailbox(
            root,
            "send",
            "--from",
            "fable",
            "--to",
            "astra",
            "--kind",
            kind,
            "--subject",
            kind,
            "--body",
            kind,
            "--reply-to",
            str(non_proposal["id"]),
        )
        assert missing.returncode == wrong_kind.returncode == 2
        assert "existing proposal" in missing.stderr
        assert "existing proposal" in wrong_kind.stderr


def test_invalid_recipient_and_traversal_id_fail_closed(tmp_path: Path) -> None:
    invalid = run_mailbox(
        tmp_path,
        "send",
        "--from",
        "astra",
        "--to",
        "other",
        "--kind",
        "question",
        "--subject",
        "x",
        "--body",
        "x",
    )
    traversal = run_mailbox(tmp_path, "ack", "fable", "../../outside")
    assert invalid.returncode == 2
    assert traversal.returncode == 2
    assert "invalid message ID" in traversal.stderr
    assert not (tmp_path.parent / "outside.json").exists()


def test_malformed_message_fails_the_inbox_and_board(tmp_path: Path) -> None:
    root = tmp_path / "mailbox"
    succeed(root, "board")
    malformed_id = "20260904T120000.000000Z-" + "0" * 32
    (root / "messages" / f"{malformed_id}.json").write_text('{"to":"fable"}\n', encoding="utf-8")
    assert run_mailbox(root, "inbox", "fable").returncode == 2
    assert run_mailbox(root, "board").returncode == 2


def test_body_is_bounded_by_utf8_bytes_for_inline_and_file_input(tmp_path: Path) -> None:
    root = tmp_path / "mailbox"
    oversized = "é" * (16 * 1024 + 1)
    inline = run_mailbox(
        root,
        "send",
        "--from",
        "astra",
        "--to",
        "fable",
        "--kind",
        "blocker",
        "--subject",
        "large",
        "--body",
        oversized,
    )
    body_file = tmp_path / "body.txt"
    body_file.write_bytes(b"x" * (32 * 1024 + 1))
    file_result = run_mailbox(
        root,
        "send",
        "--from",
        "astra",
        "--to",
        "fable",
        "--kind",
        "blocker",
        "--subject",
        "large",
        "--body-file",
        str(body_file),
    )
    assert inline.returncode == file_result.returncode == 2
    assert "exceeds 32768 UTF-8 bytes" in inline.stderr
    assert "exceeds 32768 UTF-8 bytes" in file_result.stderr
    assert not list((root / "messages").glob("*.json"))


def test_status_replaces_one_agent_snapshot_and_board_reports_mail(tmp_path: Path) -> None:
    root = tmp_path / "mailbox"
    send(root, "pending")
    succeed(
        root,
        "status",
        "astra",
        "--progress",
        "first",
        "--bottleneck",
        "one",
        "--next-step",
        "draft",
    )
    replacement = succeed(
        root,
        "status",
        "astra",
        "--progress",
        "second",
        "--bottleneck",
        "none",
        "--next-step",
        "review",
    )
    board = succeed(root, "board")
    assert board["statuses"]["astra"] == replacement  # type: ignore[index]
    assert board["statuses"]["fable"] is None  # type: ignore[index]
    assert board["mail"]["fable"] == {"total": 1, "unread": 1}  # type: ignore[index]
    assert board["heavy_job_slot"] == {"state": "free"}
    assert len(list((root / "status").iterdir())) == 1


def _stamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _message_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{uuid.uuid4().hex}"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reservation_fixture(root: Path, *, owner: str) -> Path:
    now = datetime.now(UTC)
    proposal_id = _message_id()
    acceptance_id = _message_id()
    (root / "messages").mkdir(parents=True, exist_ok=True)
    (root / "acks").mkdir(exist_ok=True)
    authoritative = {
        "accepted_until_utc": _stamp(now + timedelta(minutes=20)),
        "canonical_argv": ["python", "-m", "oracle_composition.harness.cycle_cli", "train"],
        "commit": "1" * 40,
        "conflict_check": "no_other_heavy_repository_job",
        "hard_wall_seconds": 1_200,
        "inputs": {"execution_manifest_sha256": "2" * 64},
        "mode": "smoke",
        "output": str(root / "output"),
        "owner": owner,
        "proposal_id": proposal_id,
        "required_authorizer": "fable",
    }
    body = json.dumps(authoritative, sort_keys=True, separators=(",", ":"))
    message_path = root / "messages" / f"{acceptance_id}.json"
    _write_json(
        message_path,
        {
            "body": body,
            "from": "fable",
            "id": acceptance_id,
            "kind": "acceptance",
            "reply_to": proposal_id,
            "schema_version": 1,
            "sent_at": _stamp(now - timedelta(minutes=2)),
            "subject": "Accepted bounded heavy job",
            "to": "astra",
        },
    )
    ack_path = root / "acks" / f"{acceptance_id}.json"
    _write_json(
        ack_path,
        {
            "acknowledged_at": _stamp(now - timedelta(minutes=1)),
            "meaning": "read_not_agreement",
            "message_id": acceptance_id,
            "recipient": "astra",
            "schema_version": 1,
        },
    )
    reservation = {
        "accepted": True,
        "acceptance_message": {
            "path": f"messages/{acceptance_id}.json",
            "sha256": hashlib.sha256(message_path.read_bytes()).hexdigest(),
        },
        "acknowledgment": {
            "path": f"acks/{acceptance_id}.json",
            "sha256": hashlib.sha256(ack_path.read_bytes()).hexdigest(),
        },
        "schema_version": 2,
        **authoritative,
    }
    path = root / "reservation.json"
    _write_json(path, reservation)
    return path


def test_cli_reserve_board_and_exact_release(tmp_path: Path) -> None:
    root = tmp_path / "mailbox"
    reservation = _reservation_fixture(root, owner="cli-owner")

    reserved = succeed(
        root,
        "reserve",
        "--owner",
        "cli-owner",
        "--reservation",
        str(reservation),
        "--expected-wall-seconds",
        "600",
    )
    token = reserved["slot"]
    board = succeed(root, "board")

    assert reserved["created"] is True
    assert board["heavy_job_slot"]["state"] == "held"  # type: ignore[index]
    assert board["heavy_job_slot"]["token"] == token  # type: ignore[index]

    wrong = run_mailbox(
        root,
        "release",
        "--owner",
        "cli-owner",
        "--token-id",
        "0" * 32,
    )
    assert wrong.returncode == 2
    assert json.loads(wrong.stderr)["ok"] is False
    assert len(wrong.stderr.encode("utf-8")) < 512
    assert succeed(root, "board")["heavy_job_slot"]["token"] == token  # type: ignore[index]

    released = succeed(
        root,
        "release",
        "--owner",
        "cli-owner",
        "--token-id",
        str(token["token_id"]),  # type: ignore[index]
    )
    assert released == {"released": True, "slot": token}
    assert succeed(root, "board")["heavy_job_slot"] == {"state": "free"}


def test_cli_slot_errors_are_json_and_bounded(tmp_path: Path) -> None:
    root = tmp_path / "mailbox"
    reservation = _reservation_fixture(root, owner="cli-owner")
    invalid_seconds = run_mailbox(
        root,
        "reserve",
        "--owner",
        "cli-owner",
        "--reservation",
        str(reservation),
        "--expected-wall-seconds",
        "0",
    )
    assert invalid_seconds.returncode == 2
    assert json.loads(invalid_seconds.stderr)["ok"] is False
    assert len(invalid_seconds.stderr.encode("utf-8")) < 512

    (root / "heavy-job.lock").write_text("{}", encoding="utf-8")
    malformed = run_mailbox(root, "board")
    assert malformed.returncode == 2
    assert json.loads(malformed.stderr)["ok"] is False
    assert "slot fields differ" in malformed.stderr
    assert len(malformed.stderr.encode("utf-8")) < 512


@pytest.mark.parametrize("command", ["reserve", "release", "board"])
def test_cli_rejects_symlink_root_before_preparing_target(tmp_path: Path, command: str) -> None:
    root = tmp_path / "real-root"
    reservation = _reservation_fixture(root, owner="cli-owner")
    (root / "heavy-job.lock").write_bytes(b"retained-token")
    (root / "heavy-job.guard").write_bytes(b"")
    alias = tmp_path / "root-link"
    alias.symlink_to(root, target_is_directory=True)
    before_names = set(root.iterdir())
    guard_inode = (root / "heavy-job.guard").stat().st_ino
    arguments = [command]
    if command == "reserve":
        arguments.extend(
            [
                "--owner",
                "cli-owner",
                "--reservation",
                str(reservation),
                "--expected-wall-seconds",
                "600",
            ]
        )
    elif command == "release":
        arguments.extend(["--owner", "cli-owner", "--token-id", "0" * 32])

    result = run_mailbox(alias, *arguments)

    assert result.returncode == 2
    assert json.loads(result.stderr)["ok"] is False
    assert "symlink" in result.stderr
    assert len(result.stderr.encode()) < 512
    assert set(root.iterdir()) == before_names
    assert (root / "heavy-job.lock").read_bytes() == b"retained-token"
    assert (root / "heavy-job.guard").read_bytes() == b""
    assert (root / "heavy-job.guard").stat().st_ino == guard_inode


@pytest.mark.parametrize(
    "malformation",
    ["nul-authority-path", "surrogate-owner", "surrogate-argv", "surrogate-body", "surrogate-key"],
)
def test_cli_malformed_encoded_values_are_bounded_json_without_slot_changes(
    tmp_path: Path, malformation: str
) -> None:
    root = tmp_path / "mailbox"
    reservation_path = _reservation_fixture(root, owner="cli-owner")
    reservation = json.loads(reservation_path.read_text())
    if malformation == "nul-authority-path":
        reservation["acceptance_message"]["path"] = "messages/x\x00.json"
    elif malformation == "surrogate-owner":
        reservation["owner"] = "\ud800"
    elif malformation == "surrogate-argv":
        reservation["canonical_argv"] = ["\ud800"]
    elif malformation == "surrogate-key":
        reservation["inputs"]["\ud800"] = 1
    else:
        message_path = root / reservation["acceptance_message"]["path"]
        message = json.loads(message_path.read_text())
        message["body"] = "\ud800"
        _write_json(message_path, message)
        reservation["acceptance_message"]["sha256"] = hashlib.sha256(
            message_path.read_bytes()
        ).hexdigest()
    _write_json(reservation_path, reservation)
    (root / "heavy-job.lock").write_bytes(b"retained-token")
    (root / "heavy-job.guard").write_bytes(b"")
    guard_inode = (root / "heavy-job.guard").stat().st_ino

    result = run_mailbox(
        root,
        "reserve",
        "--owner",
        "cli-owner",
        "--reservation",
        str(reservation_path),
        "--expected-wall-seconds",
        "600",
    )

    assert result.returncode == 2
    assert json.loads(result.stderr)["ok"] is False
    assert "Traceback" not in result.stderr
    assert len(result.stderr.encode()) < 512
    assert (root / "heavy-job.lock").read_bytes() == b"retained-token"
    assert (root / "heavy-job.guard").read_bytes() == b""
    assert (root / "heavy-job.guard").stat().st_ino == guard_inode
