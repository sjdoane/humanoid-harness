"""Bounded, data-only DeepMimic ``humanoid3d`` source-format audit.

The parser accepts exact JSON bytes and checks the documented 44-value source
layout.  Quaternion normalization is used only on temporary values when
computing the sign-invariant diagnostic dot product.  The source values are
never rewritten, and a source-format pass is not a Gymnasium mapping,
retargeting, controller-compatibility, or dynamics-feasibility certificate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

AUDIT_SCHEMA_VERSION = 1
SOURCE_MANIFEST_SCHEMA_VERSION = 1
HUMANOID3D_FRAME_WIDTH = 44
MIN_FRAMES = 2
MAX_FRAMES = 100_000
MAX_SOURCE_BYTES = 8 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MAX_ABS_FRAME_VALUE = 1_000_000.0
MIN_NONTERMINAL_DURATION_SECONDS = 1e-6
MAX_FRAME_DURATION_SECONDS = 60.0
QUATERNION_NORM_TOLERANCE = 0.02
QUATERNION_ZERO_NORM_EPSILON = 1e-15
DEEPMIMIC_SOURCE_REPOSITORY = "https://github.com/xbpeng/DeepMimic"
DEEPMIMIC_RAW_HOST = "raw.githubusercontent.com"

_MOTION_FIELDS = frozenset({"Loop", "Frames", "RightJoints", "LeftJoints"})
_REQUIRED_MOTION_FIELDS = frozenset({"Loop", "Frames"})
_MANIFEST_FIELDS = frozenset(
    {
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
)
_MANIFEST_FILE_FIELDS = frozenset({"path", "bytes", "sha256", "source_url"})

# DeepMimic's documented humanoid3d layout uses wxyz quaternions at these
# offsets. Scalar knee/elbow values occupy the gaps between the slices.
QUATERNION_FIELDS: tuple[tuple[str, int, int], ...] = (
    ("root", 4, 8),
    ("chest", 8, 12),
    ("neck", 12, 16),
    ("right_hip", 16, 20),
    ("right_ankle", 21, 25),
    ("right_shoulder", 25, 29),
    ("left_hip", 30, 34),
    ("left_ankle", 35, 39),
    ("left_shoulder", 39, 43),
)


class DeepMimicFormatError(ValueError):
    """Raised when exact source bytes violate the bounded audit contract."""


@dataclass(frozen=True, slots=True)
class ParsedMotion:
    """Structurally validated source values; no retargeting or normalization."""

    loop: str
    frames: tuple[tuple[float, ...], ...]
    source_sha256: str
    source_bytes: int


def _reject_nonfinite_json_constant(value: str) -> None:
    raise DeepMimicFormatError(f"non-finite JSON constant is forbidden: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DeepMimicFormatError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _decode_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise DeepMimicFormatError(f"{label} must be valid UTF-8") from exc
    try:
        value = json.loads(
            text,
            parse_constant=_reject_nonfinite_json_constant,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except DeepMimicFormatError:
        raise
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise DeepMimicFormatError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise DeepMimicFormatError(f"{label} top-level value must be an object")
    return value


def _bounded_regular_file(path: Path, *, max_bytes: int, label: str) -> bytes:
    """Read one regular file once, rejecting links and oversized payloads."""

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DeepMimicFormatError(f"cannot open {label} as a regular file: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise DeepMimicFormatError(f"{label} must be a regular file: {path}")
        if metadata.st_size <= 0:
            raise DeepMimicFormatError(f"{label} is empty: {path}")
        if metadata.st_size > max_bytes:
            raise DeepMimicFormatError(
                f"{label} has {metadata.st_size} bytes; limit is {max_bytes}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise DeepMimicFormatError(f"{label} exceeds the {max_bytes}-byte limit")
        if len(raw) != metadata.st_size:
            raise DeepMimicFormatError(f"{label} changed while it was being read: {path}")
        return raw
    finally:
        os.close(descriptor)


def _exact_int(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DeepMimicFormatError(f"{field} must be an exact JSON integer")
    if minimum is not None and value < minimum:
        raise DeepMimicFormatError(f"{field} must be at least {minimum}")
    return value


def _required_text(value: Any, *, field: str, max_length: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeepMimicFormatError(f"{field} must be non-empty text")
    if len(value) > max_length:
        raise DeepMimicFormatError(f"{field} exceeds the {max_length}-character limit")
    if any(ord(character) < 32 for character in value):
        raise DeepMimicFormatError(f"{field} contains control characters")
    return value


def _sha256_hex(value: Any, *, field: str) -> str:
    text = _required_text(value, field=field, max_length=64)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise DeepMimicFormatError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _validate_optional_joint_indices(payload: dict[str, Any]) -> None:
    present = {field for field in ("RightJoints", "LeftJoints") if field in payload}
    if present and present != {"RightJoints", "LeftJoints"}:
        raise DeepMimicFormatError("RightJoints and LeftJoints must be supplied together")
    for field in sorted(present):
        values = payload[field]
        if not isinstance(values, list) or not values:
            raise DeepMimicFormatError(f"{field} must be a non-empty JSON list")
        if len(values) > HUMANOID3D_FRAME_WIDTH:
            raise DeepMimicFormatError(f"{field} exceeds the {HUMANOID3D_FRAME_WIDTH}-entry limit")
        indices = [
            _exact_int(value, field=f"{field}[{index}]", minimum=0)
            for index, value in enumerate(values)
        ]
        if any(index >= HUMANOID3D_FRAME_WIDTH for index in indices):
            raise DeepMimicFormatError(
                f"{field} entries must be less than {HUMANOID3D_FRAME_WIDTH}"
            )
        if len(indices) != len(set(indices)):
            raise DeepMimicFormatError(f"{field} entries must be unique")


def _validate_frames(rows: Any) -> tuple[tuple[float, ...], ...]:
    if not isinstance(rows, list):
        raise DeepMimicFormatError("Frames must be a JSON list")
    frame_count = len(rows)
    if not MIN_FRAMES <= frame_count <= MAX_FRAMES:
        raise DeepMimicFormatError(
            f"frame count {frame_count} is outside [{MIN_FRAMES}, {MAX_FRAMES}]"
        )

    frames: list[tuple[float, ...]] = []
    for frame_index, row in enumerate(rows):
        if not isinstance(row, list):
            raise DeepMimicFormatError(f"frame {frame_index} must be a JSON list")
        if len(row) != HUMANOID3D_FRAME_WIDTH:
            raise DeepMimicFormatError(
                f"frame {frame_index} width {len(row)} does not match "
                f"humanoid3d width {HUMANOID3D_FRAME_WIDTH}"
            )
        parsed_row: list[float] = []
        for column_index, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise DeepMimicFormatError(
                    f"frame {frame_index} column {column_index} must be a JSON number"
                )
            try:
                number = float(value)
            except OverflowError as exc:
                raise DeepMimicFormatError(
                    f"frame {frame_index} column {column_index} is outside finite float range"
                ) from exc
            if not math.isfinite(number):
                raise DeepMimicFormatError(
                    f"frame {frame_index} column {column_index} is non-finite"
                )
            if abs(number) > MAX_ABS_FRAME_VALUE:
                raise DeepMimicFormatError(
                    f"frame {frame_index} column {column_index} exceeds absolute bound "
                    f"{MAX_ABS_FRAME_VALUE:g}"
                )
            parsed_row.append(number)
        frames.append(tuple(parsed_row))

    durations = [frame[0] for frame in frames]
    for frame_index, duration in enumerate(durations[:-1]):
        if duration < MIN_NONTERMINAL_DURATION_SECONDS:
            raise DeepMimicFormatError(
                f"nonterminal frame {frame_index} duration must be at least "
                f"{MIN_NONTERMINAL_DURATION_SECONDS:g} s"
            )
    if durations[-1] < 0.0:
        raise DeepMimicFormatError("terminal frame duration must be nonnegative")
    for frame_index, duration in enumerate(durations):
        if duration > MAX_FRAME_DURATION_SECONDS:
            raise DeepMimicFormatError(
                f"frame {frame_index} duration exceeds {MAX_FRAME_DURATION_SECONDS:g} s"
            )
    total_duration = math.fsum(durations)
    if not math.isfinite(total_duration) or total_duration <= 0.0:
        raise DeepMimicFormatError("motion duration must be positive and finite")
    return tuple(frames)


def parse_motion_bytes(raw: bytes) -> ParsedMotion:
    """Validate exact source bytes without changing their numeric values."""

    if not isinstance(raw, bytes):
        raise DeepMimicFormatError("motion source must be exact bytes")
    if not raw:
        raise DeepMimicFormatError("motion source is empty")
    if len(raw) > MAX_SOURCE_BYTES:
        raise DeepMimicFormatError(
            f"motion source has {len(raw)} bytes; limit is {MAX_SOURCE_BYTES}"
        )
    payload = _decode_json_object(raw, label="DeepMimic motion")
    missing = sorted(_REQUIRED_MOTION_FIELDS - payload.keys())
    unknown = sorted(payload.keys() - _MOTION_FIELDS)
    if missing:
        raise DeepMimicFormatError(f"missing top-level fields: {missing!r}")
    if unknown:
        raise DeepMimicFormatError(f"unknown top-level fields: {unknown!r}")
    loop = payload["Loop"]
    if not isinstance(loop, str) or loop not in {"wrap", "none"}:
        raise DeepMimicFormatError("Loop must be exactly 'wrap' or 'none'")
    _validate_optional_joint_indices(payload)
    frames = _validate_frames(payload["Frames"])
    return ParsedMotion(
        loop=loop,
        frames=frames,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        source_bytes=len(raw),
    )


def _quaternion_diagnostic(
    frames: tuple[tuple[float, ...], ...], *, field: str, start: int, stop: int
) -> dict[str, Any]:
    quaternions = [frame[start:stop] for frame in frames]
    norms = [
        math.sqrt(math.fsum(component * component for component in value)) for value in quaternions
    ]
    errors = [abs(norm - 1.0) for norm in norms]
    maximum_error_frame = max(range(len(errors)), key=errors.__getitem__)
    minimum_norm_frame = min(range(len(norms)), key=norms.__getitem__)
    maximum_norm_frame = max(range(len(norms)), key=norms.__getitem__)
    zero_norm_frames = [
        frame_index
        for frame_index, norm in enumerate(norms)
        if norm <= QUATERNION_ZERO_NORM_EPSILON
    ]

    adjacent_dots: list[tuple[int, float]] = []
    if not zero_norm_frames:
        for frame_index in range(len(quaternions) - 1):
            left = quaternions[frame_index]
            right = quaternions[frame_index + 1]
            dot = math.fsum(a * b for a, b in zip(left, right, strict=True))
            dot /= norms[frame_index] * norms[frame_index + 1]
            # Round-off can move a mathematical cosine a few ulps outside the
            # closed interval. Clamping affects only this diagnostic scalar.
            adjacent_dots.append((frame_index, min(1.0, max(-1.0, dot))))

    negative_pairs = [
        [frame_index, frame_index + 1] for frame_index, dot in adjacent_dots if dot < 0.0
    ]
    if adjacent_dots:
        minimum_dot_frame, minimum_dot = min(adjacent_dots, key=lambda item: item[1])
        minimum_dot_pair: list[int] | None = [minimum_dot_frame, minimum_dot_frame + 1]
    else:
        minimum_dot = None
        minimum_dot_pair = None

    return {
        "field": field,
        "frame_slice": [start, stop],
        "quaternion_order": "wxyz",
        "maximum_absolute_norm_error": errors[maximum_error_frame],
        "maximum_absolute_norm_error_frame": maximum_error_frame,
        "minimum_norm": norms[minimum_norm_frame],
        "minimum_norm_frame": minimum_norm_frame,
        "maximum_norm": norms[maximum_norm_frame],
        "maximum_norm_frame": maximum_norm_frame,
        "norm_tolerance_pass": (
            not zero_norm_frames and errors[maximum_error_frame] <= QUATERNION_NORM_TOLERANCE
        ),
        "zero_norm_frames": zero_norm_frames,
        "minimum_adjacent_normalized_dot": minimum_dot,
        "minimum_adjacent_normalized_dot_pair": minimum_dot_pair,
        "negative_adjacent_normalized_dot_count": len(negative_pairs),
        "negative_adjacent_normalized_dot_pairs": negative_pairs,
        "sign_continuity_pass": not zero_norm_frames and not negative_pairs,
    }


def inspect_motion_bytes(raw: bytes) -> dict[str, Any]:
    """Return source-format metrics for validated exact bytes.

    A false ``source_format_pass`` is a diagnostic result rather than a parser
    exception. Structural, type, bound, width, and duration failures do raise.
    """

    motion = parse_motion_bytes(raw)
    quaternion_fields = [
        _quaternion_diagnostic(motion.frames, field=field, start=start, stop=stop)
        for field, start, stop in QUATERNION_FIELDS
    ]
    durations = [frame[0] for frame in motion.frames]
    quaternion_norm_pass = all(item["norm_tolerance_pass"] for item in quaternion_fields)
    quaternion_sign_continuity_pass = all(
        item["sign_continuity_pass"] for item in quaternion_fields
    )
    return {
        "source_sha256": motion.source_sha256,
        "source_bytes": motion.source_bytes,
        "loop": motion.loop,
        "frame_count": len(motion.frames),
        "frame_width": HUMANOID3D_FRAME_WIDTH,
        "duration": {
            "total_seconds": math.fsum(durations),
            "minimum_frame_seconds": min(durations),
            "maximum_frame_seconds": max(durations),
            "terminal_frame_seconds": durations[-1],
            "positive_nonterminal_frames": all(
                value >= MIN_NONTERMINAL_DURATION_SECONDS for value in durations[:-1]
            ),
        },
        "quaternion_fields": quaternion_fields,
        "checks": {
            "bounded_exact_json_pass": True,
            "frame_count_and_width_pass": True,
            "duration_pass": True,
            "quaternion_norm_pass": quaternion_norm_pass,
            "quaternion_sign_continuity_pass": quaternion_sign_continuity_pass,
        },
        "source_format_pass": quaternion_norm_pass and quaternion_sign_continuity_pass,
    }


def _validate_source_url(
    value: Any,
    *,
    field: str,
    source_commit: str,
    expected_filename: str,
) -> str:
    url = _required_text(value, field=field)
    parsed = urlsplit(url)
    expected_path = f"/xbpeng/DeepMimic/{source_commit}/data/motions/{expected_filename}"
    if (
        parsed.scheme != "https"
        or parsed.netloc != DEEPMIMIC_RAW_HOST
        or parsed.path != expected_path
        or parsed.query
        or parsed.fragment
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port is not None
    ):
        raise DeepMimicFormatError(f"{field} must be the exact pinned official DeepMimic raw URL")
    return url


def _validate_manifest(raw: bytes) -> dict[str, Any]:
    manifest = _decode_json_object(raw, label="source manifest")
    missing = sorted(_MANIFEST_FIELDS - manifest.keys())
    unknown = sorted(manifest.keys() - _MANIFEST_FIELDS)
    if missing:
        raise DeepMimicFormatError(f"source manifest missing fields: {missing!r}")
    if unknown:
        raise DeepMimicFormatError(f"source manifest has unknown fields: {unknown!r}")
    schema_version = _exact_int(manifest["schema_version"], field="schema_version", minimum=1)
    if schema_version != SOURCE_MANIFEST_SCHEMA_VERSION:
        raise DeepMimicFormatError(
            f"source manifest schema_version must be {SOURCE_MANIFEST_SCHEMA_VERSION}"
        )
    source_repository = _required_text(manifest["source_repository"], field="source_repository")
    if source_repository != DEEPMIMIC_SOURCE_REPOSITORY:
        raise DeepMimicFormatError(
            f"source_repository must be exactly {DEEPMIMIC_SOURCE_REPOSITORY!r}"
        )
    source_commit = _required_text(manifest["source_commit"], field="source_commit", max_length=40)
    if len(source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in source_commit
    ):
        raise DeepMimicFormatError("source_commit must be a lowercase 40-character Git digest")
    _required_text(manifest["source_commit_date_utc"], field="source_commit_date_utc")
    _required_text(manifest["registered_date"], field="registered_date")
    _required_text(manifest["claim_boundary"], field="claim_boundary", max_length=8192)
    if not isinstance(manifest["license"], dict):
        raise DeepMimicFormatError("license must be a JSON object")
    if not isinstance(manifest["legacy_provenance"], dict):
        raise DeepMimicFormatError("legacy_provenance must be a JSON object")

    records = manifest["files"]
    if not isinstance(records, list) or not records or len(records) > 100:
        raise DeepMimicFormatError("files must be a non-empty JSON list with at most 100 entries")
    seen_paths: set[str] = set()
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise DeepMimicFormatError(f"files[{record_index}] must be a JSON object")
        if set(record) != _MANIFEST_FILE_FIELDS:
            raise DeepMimicFormatError(
                f"files[{record_index}] must contain exactly {sorted(_MANIFEST_FILE_FIELDS)!r}"
            )
        path_text = _required_text(record["path"], field=f"files[{record_index}].path")
        relative = PurePosixPath(path_text)
        if relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != "raw":
            raise DeepMimicFormatError(
                f"files[{record_index}].path must be one direct file under raw/"
            )
        if any(part in {"", ".", ".."} for part in relative.parts):
            raise DeepMimicFormatError(f"files[{record_index}].path is not a safe relative path")
        if path_text in seen_paths:
            raise DeepMimicFormatError(f"duplicate source manifest path: {path_text}")
        seen_paths.add(path_text)
        byte_count = _exact_int(record["bytes"], field=f"files[{record_index}].bytes", minimum=1)
        if byte_count > MAX_SOURCE_BYTES:
            raise DeepMimicFormatError(
                f"files[{record_index}].bytes exceeds source limit {MAX_SOURCE_BYTES}"
            )
        _sha256_hex(record["sha256"], field=f"files[{record_index}].sha256")
        _validate_source_url(
            record["source_url"],
            field=f"files[{record_index}].source_url",
            source_commit=source_commit,
            expected_filename=relative.name,
        )
    return manifest


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _audit_payload_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def audit_source_tree(source_root: Path) -> dict[str, Any]:
    """Audit every manifest-bound motion below one source directory."""

    source_root = Path(source_root)
    raw_root = source_root / "raw"
    if source_root.is_symlink() or not source_root.is_dir():
        raise DeepMimicFormatError("source root must be a real directory")
    if raw_root.is_symlink() or not raw_root.is_dir():
        raise DeepMimicFormatError("source raw root must be a real directory")
    manifest_path = source_root / "source_manifest.json"
    manifest_raw = _bounded_regular_file(
        manifest_path, max_bytes=MAX_MANIFEST_BYTES, label="source manifest"
    )
    manifest = _validate_manifest(manifest_raw)
    motions: list[dict[str, Any]] = []
    for record in sorted(manifest["files"], key=lambda item: item["path"]):
        relative = PurePosixPath(record["path"])
        source_path = source_root.joinpath(*relative.parts)
        source_raw = _bounded_regular_file(
            source_path, max_bytes=MAX_SOURCE_BYTES, label=f"source motion {record['path']}"
        )
        actual_sha256 = hashlib.sha256(source_raw).hexdigest()
        if len(source_raw) != record["bytes"]:
            raise DeepMimicFormatError(
                f"{record['path']} byte count does not match source_manifest.json"
            )
        if actual_sha256 != record["sha256"]:
            raise DeepMimicFormatError(
                f"{record['path']} SHA-256 does not match source_manifest.json"
            )
        result = inspect_motion_bytes(source_raw)
        if result["source_sha256"] != actual_sha256:
            raise DeepMimicFormatError(f"{record['path']} changed during audit")
        motions.append(
            {
                "path": record["path"],
                "source_url": record["source_url"],
                "manifest_binding_pass": True,
                **result,
            }
        )

    failed_paths = [item["path"] for item in motions if not item["source_format_pass"]]
    payload: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "kind": "deepmimic_humanoid3d_source_format_audit",
        "generated_by": "oracle_composition.sources.deepmimic/v1",
        "source_manifest": {
            "path": "source_manifest.json",
            "sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "source_repository": manifest["source_repository"],
            "source_commit": manifest["source_commit"],
        },
        "policy": {
            "max_manifest_bytes": MAX_MANIFEST_BYTES,
            "max_source_bytes": MAX_SOURCE_BYTES,
            "minimum_frames": MIN_FRAMES,
            "maximum_frames": MAX_FRAMES,
            "frame_width": HUMANOID3D_FRAME_WIDTH,
            "maximum_absolute_frame_value": MAX_ABS_FRAME_VALUE,
            "minimum_nonterminal_duration_seconds": MIN_NONTERMINAL_DURATION_SECONDS,
            "maximum_frame_duration_seconds": MAX_FRAME_DURATION_SECONDS,
            "quaternion_order": "wxyz",
            "quaternion_norm_tolerance": QUATERNION_NORM_TOLERANCE,
            "quaternion_sign_discontinuity_rule": "adjacent normalized dot < 0",
            "normalization_scope": (
                "temporary diagnostic dot products only; source bytes and source values unchanged"
            ),
        },
        "motions": motions,
        "summary": {
            "motion_count": len(motions),
            "source_format_pass_count": len(motions) - len(failed_paths),
            "source_format_fail_count": len(failed_paths),
            "failed_paths": failed_paths,
        },
        "admission": {
            "source_bytes_transformed": False,
            "normalized_motion_emitted": False,
            "gymnasium_joint_mapping_established": False,
            "retargeting_performed": False,
            "controller_compatibility_established": False,
            "dynamics_feasibility_established": False,
            "training_admission_granted": False,
        },
        "claim_boundary": (
            "Source-format diagnostics only. A pass does not establish Gymnasium Humanoid "
            "joint mapping, root-frame conversion, cadence compatibility, retargeting, "
            "controller compatibility, kinematic quality, dynamics feasibility, or behavior."
        ),
        "audit_hash_scope": (
            "SHA-256 of canonical UTF-8 JSON for every top-level field except audit_payload_sha256"
        ),
    }
    return {**payload, "audit_payload_sha256": _audit_payload_sha256(payload)}


def verify_audit_payload_hash(audit: dict[str, Any]) -> bool:
    """Verify the self-described canonical payload digest."""

    if not isinstance(audit, dict):
        return False
    expected = audit.get("audit_payload_sha256")
    if not isinstance(expected, str):
        return False
    payload = {key: value for key, value in audit.items() if key != "audit_payload_sha256"}
    return expected == _audit_payload_sha256(payload)


def render_audit_json(audit: dict[str, Any]) -> str:
    """Render deterministic, reviewable JSON with a terminating newline."""

    if not verify_audit_payload_hash(audit):
        raise DeepMimicFormatError("audit_payload_sha256 does not match audit payload")
    return (
        json.dumps(
            audit,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "HUMANOID3D_FRAME_WIDTH",
    "MAX_FRAMES",
    "MAX_SOURCE_BYTES",
    "MIN_FRAMES",
    "QUATERNION_FIELDS",
    "QUATERNION_NORM_TOLERANCE",
    "DeepMimicFormatError",
    "ParsedMotion",
    "audit_source_tree",
    "inspect_motion_bytes",
    "parse_motion_bytes",
    "render_audit_json",
    "verify_audit_payload_hash",
]
