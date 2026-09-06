"""Cooperative exclusion token for one heavy repository job."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import stat
import tempfile
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised only on unsupported platforms
    fcntl = None  # type: ignore[assignment]

UTC = timezone.utc  # noqa: UP017 -- macOS system Python 3.9 has no datetime.UTC.
SLOT_FILENAME = "heavy-job.lock"
GUARD_FILENAME = "heavy-job.guard"
MAX_RECORD_BYTES = 64 * 1024
MAX_OWNER_BYTES = 256
MAX_ARG_BYTES = 16 * 1024
MESSAGE_ID = re.compile(r"^\d{8}T\d{6}\.\d{6}Z-[0-9a-f]{32}$")
TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
TOKEN_ID = re.compile(r"^[0-9a-f]{32}$")
MAILBOX_AGENTS = ("astra", "fable")


class ResourceSlotError(Exception):
    """Base class for bounded slot failures."""


class ReservationError(ResourceSlotError):
    """The reservation or its mailbox authority is invalid."""


class SlotMissingError(ResourceSlotError):
    """No slot token exists."""


class SlotMalformedError(ResourceSlotError):
    """The slot token or persistent guard is unsafe or malformed."""


class SlotOccupiedError(ResourceSlotError):
    """Another immutable token already occupies the slot."""


class SlotOwnershipError(ResourceSlotError):
    """A release does not match the current token identity."""


class SlotBindingError(ResourceSlotError):
    """The current token does not bind the supplied reservation."""


class SlotExpiredError(ResourceSlotError):
    """The current token is still held but cannot authorize a new dispatch."""


class LockingUnavailableError(ResourceSlotError):
    """The platform cannot provide the required no-follow flock protocol."""


def _error(error_type: type[ResourceSlotError], message: str) -> ResourceSlotError:
    return error_type(message)


def _utf8_bytes(value: str, *, label: str, error_type: type[ResourceSlotError]) -> bytes:
    try:
        return value.encode("utf-8")
    except UnicodeError as error:
        raise _error(error_type, f"{label} is not valid UTF-8 text") from error


def _validate_json_tree(
    value: object,
    *,
    label: str,
    error_type: type[ResourceSlotError],
    depth: int = 0,
) -> None:
    if depth > 64:
        raise _error(error_type, f"{label} exceeds the JSON nesting bound")
    if type(value) is dict:
        for key, child in value.items():  # type: ignore[union-attr]
            if type(key) is not str:
                raise _error(error_type, f"{label} has a non-string JSON key")
            _utf8_bytes(key, label=label, error_type=error_type)
            _validate_json_tree(
                child,
                label=label,
                error_type=error_type,
                depth=depth + 1,
            )
        return
    if type(value) is list:
        for child in value:  # type: ignore[union-attr]
            _validate_json_tree(
                child,
                label=label,
                error_type=error_type,
                depth=depth + 1,
            )
        return
    if type(value) is float and not math.isfinite(value):
        raise _error(error_type, f"{label} contains a non-finite JSON number")
    if type(value) is str:
        _utf8_bytes(value, label=label, error_type=error_type)
        return
    if value is None or type(value) in {bool, int, float}:
        return
    raise _error(error_type, f"{label} contains a non-JSON value")


def canonical_json_bytes(value: object) -> bytes:
    """Return the one compact, sorted encoding used for token bindings."""

    _validate_json_tree(value, label="value", error_type=ResourceSlotError)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise ResourceSlotError("value is not finite canonical JSON") from error


def _decode_json_object(
    payload: bytes,
    *,
    label: str,
    error_type: type[ResourceSlotError],
    require_canonical: bool = False,
) -> dict[str, object]:
    def reject_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise _error(error_type, f"{label} has duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise _error(error_type, f"{label} contains non-finite JSON constant {value!r}")

    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except ResourceSlotError:
        raise
    except (UnicodeError, ValueError, RecursionError) as error:
        raise _error(error_type, f"{label} is malformed JSON") from error
    if type(value) is not dict:
        raise _error(error_type, f"{label} must be a JSON object")
    _validate_json_tree(value, label=label, error_type=error_type)
    if require_canonical:
        try:
            encoded = canonical_json_bytes(value)
        except ResourceSlotError as error:
            raise _error(error_type, f"{label} is not canonical JSON") from error
        if not hmac.compare_digest(encoded, payload):
            raise _error(error_type, f"{label} is not canonical JSON")
    return value


def _require_platform() -> None:
    if fcntl is None or not hasattr(fcntl, "flock") or not hasattr(os, "O_NOFOLLOW"):
        raise LockingUnavailableError("safe flock resource slots are unavailable on this platform")


def _coordination_root(root: Path) -> Path:
    _require_platform()
    absolute = Path(os.path.abspath(os.fspath(root)))
    try:
        if stat.S_ISLNK(absolute.lstat().st_mode):
            raise SlotMalformedError("coordination root must not be a symlink")
        resolved = absolute.resolve(strict=True)
        metadata = resolved.lstat()
    except (OSError, ValueError) as error:
        raise SlotMalformedError("coordination root is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise SlotMalformedError("coordination root must resolve to a real directory")
    return resolved


def _safe_input_file(path: Path, *, label: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    try:
        metadata = absolute.lstat()
        resolved = absolute.resolve(strict=True)
    except (OSError, ValueError) as error:
        raise ReservationError(f"{label} is unavailable") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ReservationError(f"{label} must not be a symlink")
    return resolved


def _read_regular_file(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
    error_type: type[ResourceSlotError],
    missing_type: type[ResourceSlotError] | None = None,
) -> tuple[bytes, tuple[int, int, int, int, int]]:
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(os.fspath(path), flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= maximum_bytes:
            raise _error(error_type, f"{label} is not bounded regular data")
        remaining = maximum_bytes + 1
        chunks: list[bytes] = []
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        if before_identity != identity or len(payload) != before.st_size:
            raise _error(error_type, f"{label} changed while read")
        try:
            named = path.lstat()
        except OSError as error:
            raise _error(error_type, f"{label} changed while read") from error
        named_identity = (
            named.st_dev,
            named.st_ino,
            named.st_size,
            named.st_mtime_ns,
            named.st_ctime_ns,
        )
        if stat.S_ISLNK(named.st_mode) or named_identity != identity:
            raise _error(error_type, f"{label} changed while read")
        return payload, identity
    except FileNotFoundError as error:
        selected = missing_type or error_type
        raise _error(selected, f"{label} does not exist") from error
    except ResourceSlotError:
        raise
    except (OSError, ValueError) as error:
        raise _error(error_type, f"{label} is unreadable without following links") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _sha256(value: object, *, label: str) -> str:
    if type(value) is not str or SHA256.fullmatch(value) is None:
        raise ReservationError(f"{label} must be a lowercase SHA-256")
    return value


def _timestamp(
    value: object,
    *,
    label: str,
    error_type: type[ResourceSlotError],
) -> datetime:
    if type(value) is not str or TIMESTAMP.fullmatch(value) is None:
        raise _error(error_type, f"{label} is not an exact UTC mailbox timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise _error(error_type, f"{label} is malformed") from error
    if parsed.utcoffset() != UTC.utcoffset(parsed):
        raise _error(error_type, f"{label} is not UTC")
    return parsed


def _current_time(now: datetime | None) -> datetime:
    current = now or datetime.now(UTC)
    if type(current) is not datetime or current.tzinfo is None or current.utcoffset() is None:
        raise ResourceSlotError("current time must be timezone-aware")
    return current


def _owner(value: object, *, error_type: type[ResourceSlotError]) -> str:
    if (
        type(value) is not str
        or not value.strip()
        or "\x00" in value
        or len(_utf8_bytes(value, label="owner", error_type=error_type)) > MAX_OWNER_BYTES
    ):
        raise _error(error_type, "owner is empty or exceeds its bound")
    return value


def _argv(value: object, *, error_type: type[ResourceSlotError]) -> list[str]:
    if type(value) is not list or not value:
        raise _error(error_type, "canonical argv is malformed")
    total = 0
    for item in value:
        if type(item) is not str or not item or "\x00" in item:
            raise _error(error_type, "canonical argv is malformed")
        total += len(_utf8_bytes(item, label="canonical argv", error_type=error_type))
    if total > MAX_ARG_BYTES:
        raise _error(error_type, "canonical argv exceeds its bound")
    return list(value)


def _positive_seconds(value: object, *, label: str, error_type: type[ResourceSlotError]) -> int:
    if type(value) is not int or value <= 0:
        raise _error(error_type, f"{label} must be a positive integer")
    return value


def _validate_reservation_shape(reservation: object) -> dict[str, object]:
    required = {
        "accepted",
        "acceptance_message",
        "accepted_until_utc",
        "acknowledgment",
        "canonical_argv",
        "commit",
        "conflict_check",
        "hard_wall_seconds",
        "inputs",
        "mode",
        "output",
        "owner",
        "proposal_id",
        "required_authorizer",
        "schema_version",
    }
    if type(reservation) is not dict or set(reservation) != required:
        raise ReservationError("mailbox reservation v2 fields differ")
    if type(reservation["schema_version"]) is not int or reservation["schema_version"] != 2:
        raise ReservationError("mailbox reservation schema version differs")
    if reservation["accepted"] is not True:
        raise ReservationError("mailbox reservation is not accepted")
    _owner(reservation["owner"], error_type=ReservationError)
    _argv(reservation["canonical_argv"], error_type=ReservationError)
    if type(reservation["commit"]) is not str or COMMIT.fullmatch(reservation["commit"]) is None:
        raise ReservationError("mailbox reservation commit is not a full lowercase Git commit")
    if (
        type(reservation["proposal_id"]) is not str
        or MESSAGE_ID.fullmatch(reservation["proposal_id"]) is None
    ):
        raise ReservationError("mailbox reservation proposal identity is malformed")
    _timestamp(
        reservation["accepted_until_utc"],
        label="reservation acceptance expiry",
        error_type=ReservationError,
    )
    if reservation["conflict_check"] != "no_other_heavy_repository_job":
        raise ReservationError("mailbox reservation conflict declaration differs")
    mode = reservation["mode"]
    hard_wall = _positive_seconds(
        reservation["hard_wall_seconds"],
        label="hard wall seconds",
        error_type=ReservationError,
    )
    if type(mode) is not str or mode not in {"smoke", "cohort"}:
        raise ReservationError("mailbox reservation mode differs")
    if hard_wall != (1_200 if mode == "smoke" else 7_200):
        raise ReservationError("mailbox reservation hard deadline differs")
    if type(reservation["output"]) is not str or not reservation["output"]:
        raise ReservationError("mailbox reservation output is empty")
    if type(reservation["inputs"]) is not dict or not reservation["inputs"]:
        raise ReservationError("mailbox reservation input ledger is empty")
    if (
        type(reservation["required_authorizer"]) is not str
        or not reservation["required_authorizer"].strip()
    ):
        raise ReservationError("mailbox reservation required authorizer is empty")
    for field in ("acceptance_message", "acknowledgment"):
        binding = reservation[field]
        if type(binding) is not dict or set(binding) != {"path", "sha256"}:
            raise ReservationError(f"mailbox reservation {field} binding is malformed")
        if type(binding["path"]) is not str or not binding["path"]:
            raise ReservationError(f"mailbox reservation {field} path is malformed")
        _sha256(binding["sha256"], label=f"mailbox reservation {field} digest")
    try:
        canonical_json_bytes(reservation)
    except ResourceSlotError as error:
        raise ReservationError("mailbox reservation is not finite JSON") from error
    return dict(reservation)


def _authority_path(root: Path, binding: Mapping[str, object], directory: str) -> Path:
    raw = binding["path"]
    if type(raw) is not str or not raw or "\x00" in raw:
        raise ReservationError("mailbox authority path is malformed")
    raw_path = Path(raw)
    candidate = raw_path if raw_path.is_absolute() else root / raw_path
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise ReservationError("mailbox authority path escapes the coordination root") from error
    if len(relative.parts) != 2 or relative.parts[0] != directory or candidate.suffix != ".json":
        raise ReservationError("mailbox authority path is not a direct native-mailbox record")
    parent = root / directory
    try:
        parent_metadata = parent.lstat()
        parent_resolved = parent.resolve(strict=True)
    except OSError as error:
        raise ReservationError("mailbox authority directory is unavailable") from error
    if (
        parent_resolved != parent
        or stat.S_ISLNK(parent_metadata.st_mode)
        or not stat.S_ISDIR(parent_metadata.st_mode)
    ):
        raise ReservationError("mailbox authority directory is unsafe")
    return candidate


def _authority_record(
    root: Path,
    binding: Mapping[str, object],
    *,
    directory: str,
    label: str,
) -> tuple[Path, dict[str, object]]:
    path = _authority_path(root, binding, directory)
    payload, _identity = _read_regular_file(
        path,
        maximum_bytes=MAX_RECORD_BYTES,
        label=label,
        error_type=ReservationError,
    )
    expected = _sha256(binding["sha256"], label=f"{label} digest")
    if not hmac.compare_digest(hashlib.sha256(payload).hexdigest(), expected):
        raise ReservationError(f"{label} digest differs")
    return path, _decode_json_object(payload, label=label, error_type=ReservationError)


def load_validated_reservation(
    coordination_root: Path,
    reservation_path: Path,
    *,
    now: datetime | None = None,
) -> dict[str, object]:
    """Load v2 authority and verify its exact acceptance and read acknowledgment."""

    root = _coordination_root(Path(coordination_root))
    path = _safe_input_file(Path(reservation_path), label="reservation file")
    payload, _identity = _read_regular_file(
        path,
        maximum_bytes=MAX_RECORD_BYTES,
        label="reservation file",
        error_type=ReservationError,
    )
    reservation = _validate_reservation_shape(
        _decode_json_object(payload, label="reservation file", error_type=ReservationError)
    )
    acceptance_binding = reservation["acceptance_message"]
    acknowledgment_binding = reservation["acknowledgment"]
    assert type(acceptance_binding) is dict
    assert type(acknowledgment_binding) is dict
    message_path, message = _authority_record(
        root,
        acceptance_binding,
        directory="messages",
        label="reservation acceptance message",
    )
    ack_path, acknowledgment = _authority_record(
        root,
        acknowledgment_binding,
        directory="acks",
        label="reservation acceptance acknowledgment",
    )
    message_id = message_path.stem
    message_fields = {
        "body",
        "from",
        "id",
        "kind",
        "reply_to",
        "schema_version",
        "sent_at",
        "subject",
        "to",
    }
    valid_message = (
        set(message) == message_fields
        and type(message.get("schema_version")) is int
        and message["schema_version"] == 1
        and MESSAGE_ID.fullmatch(message_id) is not None
        and message.get("id") == message_id
        and message.get("kind") == "acceptance"
        and message.get("reply_to") == reservation["proposal_id"]
        and message.get("from") == reservation["required_authorizer"]
        and message.get("from") in MAILBOX_AGENTS
        and message.get("to") in MAILBOX_AGENTS
        and message.get("from") != message.get("to")
        and type(message.get("subject")) is str
        and bool(message["subject"])
        and type(message.get("body")) is str
        and len(message["body"].encode("utf-8")) <= 32 * 1024
        and type(message.get("sent_at")) is str
        and TIMESTAMP.fullmatch(message["sent_at"]) is not None
    )
    if not valid_message:
        raise ReservationError("reservation acceptance authorizer or message differs")
    authoritative = {
        field: reservation[field]
        for field in (
            "accepted_until_utc",
            "canonical_argv",
            "commit",
            "conflict_check",
            "hard_wall_seconds",
            "inputs",
            "mode",
            "output",
            "owner",
            "proposal_id",
            "required_authorizer",
        )
    }
    body_payload = message["body"].encode("utf-8")
    body = _decode_json_object(
        body_payload,
        label="reservation acceptance body",
        error_type=ReservationError,
        require_canonical=True,
    )
    if not hmac.compare_digest(canonical_json_bytes(body), canonical_json_bytes(authoritative)):
        raise ReservationError("reservation and authoritative acceptance differ")
    acknowledgment_fields = {
        "acknowledged_at",
        "meaning",
        "message_id",
        "recipient",
        "schema_version",
    }
    valid_ack = (
        set(acknowledgment) == acknowledgment_fields
        and type(acknowledgment.get("schema_version")) is int
        and acknowledgment["schema_version"] == 1
        and ack_path.name == f"{message_id}.json"
        and acknowledgment.get("message_id") == message_id
        and acknowledgment.get("recipient") == message.get("to")
        and acknowledgment.get("meaning") == "read_not_agreement"
        and type(acknowledgment.get("acknowledged_at")) is str
        and TIMESTAMP.fullmatch(acknowledgment["acknowledged_at"]) is not None
    )
    if not valid_ack:
        raise ReservationError("reservation acceptance acknowledgment differs")
    sent_at = _timestamp(
        message["sent_at"], label="acceptance sent_at", error_type=ReservationError
    )
    acknowledged_at = _timestamp(
        acknowledgment["acknowledged_at"],
        label="acceptance acknowledged_at",
        error_type=ReservationError,
    )
    expiry = _timestamp(
        reservation["accepted_until_utc"],
        label="reservation acceptance expiry",
        error_type=ReservationError,
    )
    current = _current_time(now)
    if not sent_at <= acknowledged_at <= current < expiry:
        raise ReservationError("mailbox reservation is stale")
    return reservation


def _acceptance_id(reservation: Mapping[str, object]) -> str:
    binding = reservation["acceptance_message"]
    if type(binding) is not dict or type(binding.get("path")) is not str:
        raise ReservationError("mailbox reservation acceptance binding is malformed")
    identity = Path(binding["path"]).stem
    if MESSAGE_ID.fullmatch(identity) is None:
        raise ReservationError("mailbox reservation acceptance identity is malformed")
    return identity


def _binding(
    reservation: Mapping[str, object],
    *,
    expected_wall_seconds: int,
) -> dict[str, object]:
    validated = _validate_reservation_shape(dict(reservation))
    expected = _positive_seconds(
        expected_wall_seconds,
        label="expected wall seconds",
        error_type=ReservationError,
    )
    hard = validated["hard_wall_seconds"]
    assert type(hard) is int
    if expected > hard:
        raise ReservationError("expected wall seconds exceed the accepted hard wall")
    return {
        "acceptance_id": _acceptance_id(validated),
        "accepted_until_utc": validated["accepted_until_utc"],
        "canonical_argv": validated["canonical_argv"],
        "commit": validated["commit"],
        "expected_wall_seconds": expected,
        "hard_wall_seconds": hard,
        "owner": validated["owner"],
        "proposal_id": validated["proposal_id"],
        "reservation_canonical_sha256": hashlib.sha256(canonical_json_bytes(validated)).hexdigest(),
    }


def _validate_token_shape(token: object) -> dict[str, object]:
    required = {
        "acceptance_id",
        "accepted_until_utc",
        "canonical_argv",
        "commit",
        "expected_wall_seconds",
        "hard_wall_seconds",
        "owner",
        "proposal_id",
        "reservation_canonical_sha256",
        "schema_version",
        "token_id",
    }
    if type(token) is not dict or set(token) != required:
        raise SlotMalformedError("heavy-job slot fields differ")
    if type(token["schema_version"]) is not int or token["schema_version"] != 1:
        raise SlotMalformedError("heavy-job slot schema version differs")
    if type(token["token_id"]) is not str or TOKEN_ID.fullmatch(token["token_id"]) is None:
        raise SlotMalformedError("heavy-job token identity is malformed")
    _owner(token["owner"], error_type=SlotMalformedError)
    _argv(token["canonical_argv"], error_type=SlotMalformedError)
    if type(token["commit"]) is not str or COMMIT.fullmatch(token["commit"]) is None:
        raise SlotMalformedError("heavy-job slot commit is malformed")
    for field in ("proposal_id", "acceptance_id"):
        if type(token[field]) is not str or MESSAGE_ID.fullmatch(token[field]) is None:
            raise SlotMalformedError(f"heavy-job slot {field} is malformed")
    if (
        type(token["reservation_canonical_sha256"]) is not str
        or SHA256.fullmatch(token["reservation_canonical_sha256"]) is None
    ):
        raise SlotMalformedError("heavy-job slot reservation digest is malformed")
    expected = _positive_seconds(
        token["expected_wall_seconds"],
        label="expected wall seconds",
        error_type=SlotMalformedError,
    )
    hard = _positive_seconds(
        token["hard_wall_seconds"],
        label="hard wall seconds",
        error_type=SlotMalformedError,
    )
    if expected > hard or hard not in {1_200, 7_200}:
        raise SlotMalformedError("heavy-job slot wall bounds contradict each other")
    _timestamp(
        token["accepted_until_utc"],
        label="heavy-job slot acceptance expiry",
        error_type=SlotMalformedError,
    )
    try:
        canonical_json_bytes(token)
    except ResourceSlotError as error:
        raise SlotMalformedError("heavy-job slot is not finite JSON") from error
    return dict(token)


def _read_slot_record(root: Path) -> tuple[dict[str, object], tuple[int, int, int, int, int]]:
    payload, identity = _read_regular_file(
        root / SLOT_FILENAME,
        maximum_bytes=MAX_RECORD_BYTES,
        label="heavy-job slot",
        error_type=SlotMalformedError,
        missing_type=SlotMissingError,
    )
    token = _decode_json_object(
        payload,
        label="heavy-job slot",
        error_type=SlotMalformedError,
        require_canonical=True,
    )
    return _validate_token_shape(token), identity


def read_slot(coordination_root: Path) -> dict[str, object]:
    """Read and validate the immutable token; missing and malformed are distinct."""

    root = _coordination_root(Path(coordination_root))
    token, _identity = _read_slot_record(root)
    return token


def slot_status(
    coordination_root: Path,
    *,
    now: datetime | None = None,
) -> dict[str, object]:
    """Return ``free`` or the validated held token, including expiry state."""

    try:
        token = read_slot(coordination_root)
    except SlotMissingError:
        return {"state": "free"}
    expiry = _timestamp(
        token["accepted_until_utc"],
        label="heavy-job slot acceptance expiry",
        error_type=SlotMalformedError,
    )
    return {"expired": _current_time(now) >= expiry, "state": "held", "token": token}


@contextmanager
def _serialized(root: Path) -> Iterator[None]:
    guard_path = root / GUARD_FILENAME
    flags = os.O_RDWR | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(guard_path, flags | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            descriptor = os.open(guard_path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size != 0:
            raise SlotMalformedError("heavy-job guard must be one empty regular file")
        assert fcntl is not None
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        held = os.fstat(descriptor)
        named = guard_path.lstat()
        if (
            stat.S_ISLNK(named.st_mode)
            or not stat.S_ISREG(named.st_mode)
            or (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino)
            or held.st_nlink != 1
            or held.st_size != 0
        ):
            raise SlotMalformedError("heavy-job guard changed or is unsafe")
        yield
    except ResourceSlotError:
        raise
    except OSError as error:
        raise SlotMalformedError(
            "heavy-job guard cannot provide safe flock serialization"
        ) from error
    finally:
        if descriptor is not None:
            if fcntl is not None:
                with suppress(OSError):
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _publish_token(root: Path, token: Mapping[str, object]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{SLOT_FILENAME}.", dir=root)
    temporary_path = Path(temporary)
    try:
        payload = canonical_json_bytes(dict(token))
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fchmod(handle.fileno(), 0o400)
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, root / SLOT_FILENAME)
        except FileExistsError as error:
            raise SlotOccupiedError("heavy-job slot became occupied") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        with suppress(FileNotFoundError):
            temporary_path.unlink()


def reserve_slot(
    coordination_root: Path,
    *,
    owner: str,
    reservation_path: Path,
    expected_wall_seconds: int,
    now: datetime | None = None,
) -> dict[str, object]:
    """Atomically reserve the one slot from verified v2 mailbox authority."""

    root = _coordination_root(Path(coordination_root))
    requested_owner = _owner(owner, error_type=ReservationError)
    reservation = load_validated_reservation(root, Path(reservation_path), now=now)
    if type(reservation["owner"]) is not str or reservation["owner"] != requested_owner:
        raise ReservationError("reserve owner differs from the accepted reservation")
    binding = _binding(reservation, expected_wall_seconds=expected_wall_seconds)
    token = {
        **binding,
        "schema_version": 1,
        "token_id": uuid.uuid4().hex,
    }
    _validate_token_shape(token)
    with _serialized(root):
        expiry = _timestamp(
            token["accepted_until_utc"],
            label="heavy-job slot acceptance expiry",
            error_type=ReservationError,
        )
        if _current_time(now) >= expiry:
            raise ReservationError("mailbox reservation expired before slot publication")
        try:
            occupied, _identity = _read_slot_record(root)
        except SlotMissingError:
            occupied = None
        if occupied is not None:
            occupied_expiry = _timestamp(
                occupied["accepted_until_utc"],
                label="heavy-job slot acceptance expiry",
                error_type=SlotMalformedError,
            )
            state = (
                "expired; owner cleanup required"
                if _current_time(now) >= occupied_expiry
                else "active"
            )
            raise SlotOccupiedError(
                f"heavy-job slot is occupied by owner {occupied['owner']!r} "
                f"token {occupied['token_id']} ({state})"
            )
        _publish_token(root, token)
    return token


def validate_slot_for_reservation(
    coordination_root: Path,
    validated_reservation: Mapping[str, object],
    *,
    expected_wall_seconds: int,
    now: datetime | None = None,
) -> dict[str, object]:
    """Validate the complete current-token binding immediately before spawn.

    ``validated_reservation`` must already have passed the caller's reservation
    admission. This helper is read-only and intentionally does not spawn work.
    """

    token = read_slot(coordination_root)
    expected = _binding(
        validated_reservation,
        expected_wall_seconds=expected_wall_seconds,
    )
    observed = {field: token[field] for field in expected}
    if not hmac.compare_digest(canonical_json_bytes(observed), canonical_json_bytes(expected)):
        raise SlotBindingError("heavy-job slot does not bind the validated reservation")
    expiry = _timestamp(
        token["accepted_until_utc"],
        label="heavy-job slot acceptance expiry",
        error_type=SlotMalformedError,
    )
    if _current_time(now) >= expiry:
        raise SlotExpiredError("heavy-job slot is expired and remains held pending owner cleanup")
    return token


def release_slot(
    coordination_root: Path,
    *,
    owner: str,
    token_id: str,
) -> dict[str, object]:
    """Release only the exact current owner/token pair, including after expiry."""

    root = _coordination_root(Path(coordination_root))
    requested_owner = _owner(owner, error_type=SlotOwnershipError)
    if type(token_id) is not str or TOKEN_ID.fullmatch(token_id) is None:
        raise SlotOwnershipError("release token identity is malformed")
    with _serialized(root):
        token, identity = _read_slot_record(root)
        if token["owner"] != requested_owner or not hmac.compare_digest(
            token["token_id"], token_id
        ):
            raise SlotOwnershipError("release owner or token identity does not match the held slot")
        path = root / SLOT_FILENAME
        try:
            current = path.lstat()
        except OSError as error:
            raise SlotMalformedError("heavy-job slot changed before release") from error
        current_identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if stat.S_ISLNK(current.st_mode) or current_identity != identity:
            raise SlotMalformedError("heavy-job slot changed before release")
        path.unlink()
    return token


__all__ = [
    "GUARD_FILENAME",
    "MAX_RECORD_BYTES",
    "SLOT_FILENAME",
    "LockingUnavailableError",
    "ReservationError",
    "ResourceSlotError",
    "SlotBindingError",
    "SlotExpiredError",
    "SlotMalformedError",
    "SlotMissingError",
    "SlotOccupiedError",
    "SlotOwnershipError",
    "canonical_json_bytes",
    "load_validated_reservation",
    "read_slot",
    "release_slot",
    "reserve_slot",
    "slot_status",
    "validate_slot_for_reservation",
]
