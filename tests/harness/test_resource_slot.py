from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from oracle_composition.harness.resource_slot import (
    GUARD_FILENAME,
    MAX_RECORD_BYTES,
    SLOT_FILENAME,
    ReservationError,
    SlotBindingError,
    SlotExpiredError,
    SlotMalformedError,
    SlotMissingError,
    SlotOccupiedError,
    SlotOwnershipError,
    canonical_json_bytes,
    read_slot,
    release_slot,
    reserve_slot,
    slot_status,
    validate_slot_for_reservation,
)

UTC = timezone.utc  # noqa: UP017 -- contract also targets macOS system Python 3.9.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _stamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _message_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{uuid.uuid4().hex}"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reservation_fixture(
    root: Path,
    *,
    owner: str = "astra-heavy-job",
    clock: datetime | None = None,
    expiry: datetime | None = None,
) -> tuple[Path, dict[str, object], datetime]:
    current = clock or datetime.now(UTC)
    accepted_until = expiry or current + timedelta(minutes=20)
    (root / "messages").mkdir(parents=True, exist_ok=True)
    (root / "acks").mkdir(exist_ok=True)
    proposal_id = _message_id()
    acceptance_id = _message_id()
    authoritative = {
        "accepted_until_utc": _stamp(accepted_until),
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
    message_path = root / "messages" / f"{acceptance_id}.json"
    _write_json(
        message_path,
        {
            "body": canonical_json_bytes(authoritative).decode("utf-8"),
            "from": "fable",
            "id": acceptance_id,
            "kind": "acceptance",
            "reply_to": proposal_id,
            "schema_version": 1,
            "sent_at": _stamp(current - timedelta(minutes=2)),
            "subject": "Accepted bounded heavy job",
            "to": "astra",
        },
    )
    ack_path = root / "acks" / f"{acceptance_id}.json"
    _write_json(
        ack_path,
        {
            "acknowledged_at": _stamp(current - timedelta(minutes=1)),
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
    reservation_path = root / f"reservation-{uuid.uuid4().hex}.json"
    _write_json(reservation_path, reservation)
    return reservation_path, reservation, current


def _rewrite_reservation(path: Path, reservation: dict[str, object]) -> None:
    _write_json(path, reservation)


def test_reserve_read_validate_and_owner_release(tmp_path: Path) -> None:
    reservation_path, reservation, now = _reservation_fixture(tmp_path)

    token = reserve_slot(
        tmp_path,
        owner="astra-heavy-job",
        reservation_path=reservation_path,
        expected_wall_seconds=600,
        now=now,
    )

    assert read_slot(tmp_path) == token
    assert (
        validate_slot_for_reservation(
            tmp_path,
            reservation,
            expected_wall_seconds=600,
            now=now,
        )
        == token
    )
    assert slot_status(tmp_path, now=now) == {
        "expired": False,
        "state": "held",
        "token": token,
    }
    assert (
        release_slot(
            tmp_path,
            owner="astra-heavy-job",
            token_id=str(token["token_id"]),
        )
        == token
    )
    assert slot_status(tmp_path, now=now) == {"state": "free"}
    assert (tmp_path / GUARD_FILENAME).is_file()
    assert (tmp_path / GUARD_FILENAME).read_bytes() == b""


def test_two_real_processes_contend_and_loser_preserves_winner(tmp_path: Path) -> None:
    first_path, _first, _now = _reservation_fixture(tmp_path, owner="first-owner")
    second_path, _second, _now = _reservation_fixture(tmp_path, owner="second-owner")
    worker = """
import json
import sys
from pathlib import Path
from oracle_composition.harness.resource_slot import ResourceSlotError, reserve_slot
try:
    token = reserve_slot(
        Path(sys.argv[1]),
        owner=sys.argv[2],
        reservation_path=Path(sys.argv[3]),
        expected_wall_seconds=600,
    )
except ResourceSlotError as error:
    print(json.dumps({"error": type(error).__name__}))
    raise SystemExit(2)
print(json.dumps({"token": token}, sort_keys=True))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", worker, str(tmp_path), owner, str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
        for owner, path in (("first-owner", first_path), ("second-owner", second_path))
    ]
    results = [process.communicate(timeout=10) for process in processes]

    assert sorted(process.returncode for process in processes) == [0, 2]
    winner_index = next(index for index, process in enumerate(processes) if process.returncode == 0)
    loser_index = 1 - winner_index
    winner = json.loads(results[winner_index][0])["token"]
    loser = json.loads(results[loser_index][0])
    assert loser == {"error": "SlotOccupiedError"}
    assert read_slot(tmp_path) == winner
    assert winner["owner"] in {"first-owner", "second-owner"}
    release_slot(tmp_path, owner=winner["owner"], token_id=winner["token_id"])


def test_wrong_release_and_stale_release_after_reacquisition_preserve_slot(tmp_path: Path) -> None:
    reservation_path, _reservation, now = _reservation_fixture(tmp_path)
    first = reserve_slot(
        tmp_path,
        owner="astra-heavy-job",
        reservation_path=reservation_path,
        expected_wall_seconds=600,
        now=now,
    )

    with pytest.raises(SlotOwnershipError, match="does not match"):
        release_slot(tmp_path, owner="other-owner", token_id=str(first["token_id"]))
    with pytest.raises(SlotOwnershipError, match="does not match"):
        release_slot(tmp_path, owner="astra-heavy-job", token_id="0" * 32)
    assert read_slot(tmp_path) == first

    release_slot(tmp_path, owner="astra-heavy-job", token_id=str(first["token_id"]))
    second = reserve_slot(
        tmp_path,
        owner="astra-heavy-job",
        reservation_path=reservation_path,
        expected_wall_seconds=600,
        now=now,
    )
    assert second["token_id"] != first["token_id"]
    with pytest.raises(SlotOwnershipError, match="does not match"):
        release_slot(tmp_path, owner="astra-heavy-job", token_id=str(first["token_id"]))
    assert read_slot(tmp_path) == second
    release_slot(tmp_path, owner="astra-heavy-job", token_id=str(second["token_id"]))


def test_expiry_rejects_dispatch_but_remains_held_until_owner_cleanup(tmp_path: Path) -> None:
    now = datetime.now(UTC)
    first_path, first_reservation, _clock = _reservation_fixture(
        tmp_path,
        owner="first-owner",
        clock=now,
        expiry=now + timedelta(minutes=1),
    )
    token = reserve_slot(
        tmp_path,
        owner="first-owner",
        reservation_path=first_path,
        expected_wall_seconds=30,
        now=now,
    )
    after_expiry = now + timedelta(minutes=2)

    assert slot_status(tmp_path, now=after_expiry)["state"] == "held"
    assert slot_status(tmp_path, now=after_expiry)["expired"] is True
    with pytest.raises(SlotExpiredError, match="remains held"):
        validate_slot_for_reservation(
            tmp_path,
            first_reservation,
            expected_wall_seconds=30,
            now=after_expiry,
        )

    second_path, _second, _clock = _reservation_fixture(
        tmp_path,
        owner="second-owner",
        clock=after_expiry,
    )
    with pytest.raises(SlotOccupiedError, match="owner cleanup required"):
        reserve_slot(
            tmp_path,
            owner="second-owner",
            reservation_path=second_path,
            expected_wall_seconds=30,
            now=after_expiry,
        )
    assert read_slot(tmp_path) == token
    release_slot(tmp_path, owner="first-owner", token_id=str(token["token_id"]))


@pytest.mark.parametrize("field", ["canonical_argv", "commit", "inputs"])
def test_complete_pre_spawn_binding_rejects_command_commit_and_digest_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    reservation_path, reservation, now = _reservation_fixture(tmp_path)
    token = reserve_slot(
        tmp_path,
        owner="astra-heavy-job",
        reservation_path=reservation_path,
        expected_wall_seconds=600,
        now=now,
    )
    altered = copy.deepcopy(reservation)
    if field == "canonical_argv":
        altered[field] = ["different-command"]
    elif field == "commit":
        altered[field] = "3" * 40
    else:
        altered[field] = {"execution_manifest_sha256": "4" * 64}

    with pytest.raises(SlotBindingError, match="does not bind"):
        validate_slot_for_reservation(
            tmp_path,
            altered,
            expected_wall_seconds=600,
            now=now,
        )
    assert read_slot(tmp_path) == token


@pytest.mark.parametrize("field", ["canonical_argv", "commit"])
def test_reserve_rejects_reservation_that_differs_from_exact_acceptance_body(
    tmp_path: Path,
    field: str,
) -> None:
    reservation_path, reservation, now = _reservation_fixture(tmp_path)
    if field == "canonical_argv":
        reservation[field] = ["different-command"]
    else:
        reservation[field] = "3" * 40
    _rewrite_reservation(reservation_path, reservation)

    with pytest.raises(ReservationError, match="authoritative acceptance differ"):
        reserve_slot(
            tmp_path,
            owner="astra-heavy-job",
            reservation_path=reservation_path,
            expected_wall_seconds=600,
            now=now,
        )
    assert not (tmp_path / SLOT_FILENAME).exists()


def test_reserve_rejects_acceptance_digest_mismatch_without_publishing(tmp_path: Path) -> None:
    reservation_path, reservation, now = _reservation_fixture(tmp_path)
    reservation["acceptance_message"]["sha256"] = "0" * 64  # type: ignore[index]
    _rewrite_reservation(reservation_path, reservation)

    with pytest.raises(ReservationError, match="acceptance message digest differs"):
        reserve_slot(
            tmp_path,
            owner="astra-heavy-job",
            reservation_path=reservation_path,
            expected_wall_seconds=600,
            now=now,
        )
    assert not (tmp_path / SLOT_FILENAME).exists()


def test_strict_types_reject_bool_int_and_int_float_aliases(tmp_path: Path) -> None:
    reservation_path, reservation, now = _reservation_fixture(tmp_path)
    reservation["hard_wall_seconds"] = 1_200.0
    _rewrite_reservation(reservation_path, reservation)
    with pytest.raises(ReservationError, match="positive integer"):
        reserve_slot(
            tmp_path,
            owner="astra-heavy-job",
            reservation_path=reservation_path,
            expected_wall_seconds=600,
            now=now,
        )

    reservation_path, _reservation, now = _reservation_fixture(tmp_path)
    with pytest.raises(ReservationError, match="positive integer"):
        reserve_slot(
            tmp_path,
            owner="astra-heavy-job",
            reservation_path=reservation_path,
            expected_wall_seconds=True,
            now=now,
        )


def test_missing_and_malformed_slots_are_distinct(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    with pytest.raises(SlotMissingError, match="does not exist"):
        read_slot(tmp_path)

    for payload in (
        b"{}",
        b'{"schema_version":1,"schema_version":1}',
        b'{"value":NaN}',
        b"x" * (MAX_RECORD_BYTES + 1),
    ):
        (tmp_path / SLOT_FILENAME).write_bytes(payload)
        with pytest.raises(SlotMalformedError):
            read_slot(tmp_path)
        (tmp_path / SLOT_FILENAME).unlink()


def test_symlink_slot_is_malformed_and_target_is_unchanged(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    (tmp_path / SLOT_FILENAME).symlink_to(target)

    with pytest.raises(SlotMalformedError, match="without following links"):
        read_slot(tmp_path)
    assert target.read_text(encoding="utf-8") == "{}"


def test_symlink_root_rejects_read_reserve_and_release_without_changes(tmp_path: Path) -> None:
    root = tmp_path / "real-root"
    reservation_path, _reservation, now = _reservation_fixture(root)
    token = reserve_slot(
        root,
        owner="astra-heavy-job",
        reservation_path=reservation_path,
        expected_wall_seconds=600,
        now=now,
    )
    alias = tmp_path / "root-link"
    alias.symlink_to(root, target_is_directory=True)
    before = {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()}
    guard_inode = (root / GUARD_FILENAME).stat().st_ino

    with pytest.raises(SlotMalformedError, match="symlink"):
        read_slot(alias)
    with pytest.raises(SlotMalformedError, match="symlink"):
        reserve_slot(
            alias,
            owner="astra-heavy-job",
            reservation_path=reservation_path,
            expected_wall_seconds=600,
            now=now,
        )
    with pytest.raises(SlotMalformedError, match="symlink"):
        release_slot(alias, owner="astra-heavy-job", token_id=str(token["token_id"]))

    assert {path.name: path.read_bytes() for path in root.iterdir() if path.is_file()} == before
    assert (root / GUARD_FILENAME).stat().st_ino == guard_inode


@pytest.mark.parametrize("guard_kind", ["symlink", "nonempty", "directory"])
def test_invalid_guard_fails_closed_without_slot_publication(
    tmp_path: Path,
    guard_kind: str,
) -> None:
    reservation_path, _reservation, now = _reservation_fixture(tmp_path)
    guard = tmp_path / GUARD_FILENAME
    if guard_kind == "symlink":
        target = tmp_path / "guard-target"
        target.write_bytes(b"")
        guard.symlink_to(target)
    elif guard_kind == "nonempty":
        guard.write_bytes(b"occupied")
    else:
        guard.mkdir()

    with pytest.raises(SlotMalformedError, match="guard"):
        reserve_slot(
            tmp_path,
            owner="astra-heavy-job",
            reservation_path=reservation_path,
            expected_wall_seconds=600,
            now=now,
        )
    assert not (tmp_path / SLOT_FILENAME).exists()


def test_unsafe_authority_and_reservation_paths_fail_closed(tmp_path: Path) -> None:
    reservation_path, reservation, now = _reservation_fixture(tmp_path)
    reservation["acceptance_message"]["path"] = "messages/../acks/authority.json"  # type: ignore[index]
    _rewrite_reservation(reservation_path, reservation)
    with pytest.raises(ReservationError, match="direct native-mailbox"):
        reserve_slot(
            tmp_path,
            owner="astra-heavy-job",
            reservation_path=reservation_path,
            expected_wall_seconds=600,
            now=now,
        )

    real_path, _reservation, now = _reservation_fixture(tmp_path)
    symlink_path = tmp_path / "reservation-link.json"
    symlink_path.symlink_to(real_path)
    with pytest.raises(ReservationError, match="must not be a symlink"):
        reserve_slot(
            tmp_path,
            owner="astra-heavy-job",
            reservation_path=symlink_path,
            expected_wall_seconds=600,
            now=now,
        )

    oversized = tmp_path / "oversized-reservation.json"
    oversized.write_bytes(b"x" * (MAX_RECORD_BYTES + 1))
    with pytest.raises(ReservationError, match="bounded regular"):
        reserve_slot(
            tmp_path,
            owner="astra-heavy-job",
            reservation_path=oversized,
            expected_wall_seconds=600,
            now=now,
        )
