"""Immutable, content-addressed reference data.

The reference contract binds ordered feature names, units, root-frame
semantics, cadence, shape, and exact samples.  A runtime must compare the full
identity, not just a convenient clip name, before emitting a reference window.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Real

from .errors import OracleContractError

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_FEATURE_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
MAX_HORIZON_STEPS = 256


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise OracleContractError(f"{field} must be a numeric scalar")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise OracleContractError(f"{field} must be finite")
    return resolved


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise OracleContractError(
            f"reference cannot be content-addressed: {type(exc).__name__}: {exc}"
        ) from exc


@dataclass(frozen=True)
class ReferenceSchema:
    """Exact interpretation of columns in a reference matrix."""

    feature_names: tuple[str, ...]
    feature_units: tuple[str, ...]
    root_frame: str
    cadence_hz: float

    def __post_init__(self) -> None:
        if isinstance(self.feature_names, (str, bytes)) or not isinstance(
            self.feature_names, Sequence
        ):
            raise OracleContractError("feature_names must be an ordered sequence")
        if isinstance(self.feature_units, (str, bytes)) or not isinstance(
            self.feature_units, Sequence
        ):
            raise OracleContractError("feature_units must be an ordered sequence")
        names = tuple(self.feature_names)
        units = tuple(self.feature_units)
        object.__setattr__(self, "feature_names", names)
        object.__setattr__(self, "feature_units", units)
        object.__setattr__(
            self,
            "cadence_hz",
            _finite_number(self.cadence_hz, field="cadence_hz"),
        )

        if not names:
            raise OracleContractError("reference schema must contain features")
        if len(names) != len(units):
            raise OracleContractError("feature_names and feature_units must have equal lengths")
        if len(set(names)) != len(names):
            raise OracleContractError("feature_names must be unique and ordered")
        if any(not isinstance(name, str) or not _FEATURE_NAME.fullmatch(name) for name in names):
            raise OracleContractError("feature_names contain an invalid identifier")
        if any(not isinstance(unit, str) or not unit.strip() for unit in units):
            raise OracleContractError("feature_units must be non-empty strings")
        if not isinstance(self.root_frame, str) or not self.root_frame.strip():
            raise OracleContractError("root_frame must be an explicit non-empty name")
        if self.root_frame != self.root_frame.strip():
            raise OracleContractError("root_frame cannot contain surrounding whitespace")
        if self.cadence_hz <= 0.0:
            raise OracleContractError("cadence_hz must be positive")

    @property
    def width(self) -> int:
        return len(self.feature_names)

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "feature_units": list(self.feature_units),
            "root_frame": self.root_frame,
            "cadence_hz": self.cadence_hz,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()


def assert_exact_schema(
    expected: ReferenceSchema,
    observed: ReferenceSchema,
) -> None:
    """Reject any ordered-feature, units, frame, or cadence mismatch."""

    mismatches: list[str] = []
    if observed.feature_names != expected.feature_names:
        mismatches.append("ordered feature_names")
    if observed.feature_units != expected.feature_units:
        mismatches.append("ordered feature_units")
    if observed.root_frame != expected.root_frame:
        mismatches.append("root_frame")
    if observed.cadence_hz != expected.cadence_hz:
        mismatches.append("cadence_hz")
    if mismatches:
        raise OracleContractError("reference schema mismatch: " + ", ".join(mismatches))


@dataclass(frozen=True)
class ReferenceIdentity:
    """Stable name plus the exact semantic/content identity it resolves to."""

    artifact_id: str
    content_sha256: str
    schema_sha256: str
    n_frames: int
    n_features: int

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not _IDENTIFIER.fullmatch(self.artifact_id):
            raise OracleContractError("artifact_id must be a stable identifier")
        for field in ("content_sha256", "schema_sha256"):
            value = getattr(self, field)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise OracleContractError(f"{field} must be a lowercase SHA-256")
        for field in ("n_frames", "n_features"):
            value = getattr(self, field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise OracleContractError(f"{field} must be a positive integer")


@dataclass(frozen=True)
class ReferenceWindow:
    """A finite, immutable ``H x D`` policy-facing reference window."""

    frame_indices: tuple[int, ...]
    values: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        indices = tuple(self.frame_indices)
        rows = tuple(tuple(row) for row in self.values)
        object.__setattr__(self, "frame_indices", indices)
        object.__setattr__(self, "values", rows)
        if not indices or len(indices) != len(rows):
            raise OracleContractError(
                "reference window must have matching non-empty indices and rows"
            )
        width = len(rows[0])
        if width < 1 or any(len(row) != width for row in rows):
            raise OracleContractError("reference window must have finite H x D shape")
        if any(
            not isinstance(index, int) or isinstance(index, bool) or index < 0 for index in indices
        ):
            raise OracleContractError("reference window indices must be non-negative")
        for row_index, row in enumerate(rows):
            for column_index, value in enumerate(row):
                _finite_number(
                    value,
                    field=f"window[{row_index}][{column_index}]",
                )

    @property
    def horizon(self) -> int:
        return len(self.values)

    @property
    def width(self) -> int:
        return len(self.values[0])


def _canonical_rows(
    rows: Iterable[Sequence[float]],
    *,
    width: int,
) -> tuple[tuple[float, ...], ...]:
    if isinstance(rows, (str, bytes)):
        raise OracleContractError("reference rows must be a two-dimensional sequence")
    canonical: list[tuple[float, ...]] = []
    try:
        iterator = iter(rows)
    except TypeError as exc:
        raise OracleContractError("reference rows must be iterable") from exc
    for row_index, row in enumerate(iterator):
        if isinstance(row, (str, bytes)):
            raise OracleContractError(f"reference row {row_index} is not a sequence")
        try:
            row_values = tuple(row)
        except TypeError as exc:
            raise OracleContractError(f"reference row {row_index} is not a sequence") from exc
        if len(row_values) != width:
            raise OracleContractError(
                f"reference row {row_index} has width {len(row_values)}; expected {width}"
            )
        canonical.append(
            tuple(
                _finite_number(value, field=f"reference[{row_index}][{column_index}]")
                for column_index, value in enumerate(row_values)
            )
        )
    if not canonical:
        raise OracleContractError("reference must contain at least one frame")
    return tuple(canonical)


def reference_content_sha256(
    schema: ReferenceSchema,
    values: tuple[tuple[float, ...], ...],
) -> str:
    """Hash semantic schema, shape, and exact canonical samples."""

    payload = {
        "schema": schema.to_dict(),
        "shape": [len(values), schema.width],
        "values": [list(row) for row in values],
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclass(frozen=True)
class ReferenceArtifact:
    """Validated reference samples whose identity is re-verifiable."""

    identity: ReferenceIdentity
    schema: ReferenceSchema
    values: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        rows = _canonical_rows(self.values, width=self.schema.width)
        object.__setattr__(self, "values", rows)
        self.verify()

    @classmethod
    def create(
        cls,
        *,
        artifact_id: str,
        schema: ReferenceSchema,
        values: Iterable[Sequence[float]],
    ) -> ReferenceArtifact:
        rows = _canonical_rows(values, width=schema.width)
        identity = ReferenceIdentity(
            artifact_id=artifact_id,
            content_sha256=reference_content_sha256(schema, rows),
            schema_sha256=schema.sha256,
            n_frames=len(rows),
            n_features=schema.width,
        )
        return cls(identity=identity, schema=schema, values=rows)

    def verify(self) -> None:
        errors: list[str] = []
        if self.identity.schema_sha256 != self.schema.sha256:
            errors.append("schema hash")
        if self.identity.n_frames != len(self.values):
            errors.append("frame count")
        if self.identity.n_features != self.schema.width:
            errors.append("feature count")
        if self.identity.content_sha256 != reference_content_sha256(self.schema, self.values):
            errors.append("content hash")
        if errors:
            raise OracleContractError("reference identity mismatch: " + ", ".join(errors))

    def window(
        self,
        *,
        start_frame: int,
        horizon_steps: int,
        end_exclusive: int | None = None,
    ) -> ReferenceWindow:
        """Return an ``H x D`` window, holding the final admitted frame."""

        for field, value in (
            ("start_frame", start_frame),
            ("horizon_steps", horizon_steps),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise OracleContractError(f"{field} must be an integer")
        if not (1 <= horizon_steps <= MAX_HORIZON_STEPS):
            raise OracleContractError(f"horizon_steps must be in [1, {MAX_HORIZON_STEPS}]")
        stop = len(self.values) if end_exclusive is None else end_exclusive
        if not isinstance(stop, int) or isinstance(stop, bool):
            raise OracleContractError("end_exclusive must be an integer")
        if not (0 <= start_frame < stop <= len(self.values)):
            raise OracleContractError(
                "reference window bounds must satisfy 0 <= start_frame < end_exclusive <= n_frames"
            )
        indices = tuple(min(start_frame + offset, stop - 1) for offset in range(horizon_steps))
        return ReferenceWindow(
            frame_indices=indices,
            values=tuple(self.values[index] for index in indices),
        )
