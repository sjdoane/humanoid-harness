"""Bounded, content-addressed trajectory trace contract."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from numbers import Real
from pathlib import Path
from types import MappingProxyType

import numpy as np

TRACE_SCHEMA_VERSION = 1
TRACE_SEMANTICS = "state_at_control_boundary_action_from_previous_interval/v1"
MAX_SIGNALS = 256
MAX_SIGNAL_WIDTH = 16_384
MAX_SAMPLES = 1_000_000
MAX_EVENTS = 1_000_000
MAX_MISSING_SIGNALS = 256
MAX_ARTIFACT_BINDINGS = 64
MAX_TRACE_BYTES = 512 * 1024 * 1024
MAX_TEXT = 1_024

BEHAVIORAL_BINDING_ROLES = frozenset(
    {
        "runtime",
        "execution_manifest",
        "evaluator",
        "oracle",
        "policy_checkpoint",
        "reference",
        "task_reward",
        "tracker_checkpoint",
        "tracking_reward",
    }
)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TraceContractError(ValueError):
    """A trajectory trace violates its schema or evidence boundary."""


class EvidenceClass(StrEnum):
    INTERFACE_CHECK = "interface_check"
    EXPLORATORY = "exploratory"
    BEHAVIORAL_EVALUATION = "behavioral_evaluation"


class TraceRole(StrEnum):
    ROBOT = "robot"
    CONTROLLER = "controller"
    REFERENCE = "reference"
    ORACLE = "oracle"
    TASK = "task"
    CONTACT = "contact"
    RECOVERY = "recovery"
    EVALUATION = "evaluation"


class MissingReason(StrEnum):
    NOT_EXPOSED = "not_exposed"
    NOT_IMPLEMENTED = "not_implemented"
    NOT_APPLICABLE = "not_applicable"
    REDACTED = "redacted"


REQUIRED_DIAGNOSTIC_ROLES = MappingProxyType(
    {
        "robot.qpos": TraceRole.ROBOT,
        "robot.qvel": TraceRole.ROBOT,
        "robot.root_position_world_m": TraceRole.ROBOT,
        "controller.action": TraceRole.CONTROLLER,
        "controller.motor_target": TraceRole.CONTROLLER,
        "controller.applied_torque": TraceRole.CONTROLLER,
        "controller.energy_j": TraceRole.CONTROLLER,
        "controller.saturation": TraceRole.CONTROLLER,
        "reference.frame": TraceRole.REFERENCE,
        "reference.window_index": TraceRole.REFERENCE,
        "contact.floor_normal_force_n": TraceRole.CONTACT,
        "oracle.mode": TraceRole.ORACLE,
        "oracle.phase": TraceRole.ORACLE,
        "oracle.transition_guard_margin": TraceRole.ORACLE,
        "recovery.disturbance": TraceRole.RECOVERY,
        "recovery.rejoin_state": TraceRole.RECOVERY,
        "task.progress": TraceRole.TASK,
    }
)
REQUIRED_DIAGNOSTIC_SIGNALS = frozenset(REQUIRED_DIAGNOSTIC_ROLES)


def _identifier(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise TraceContractError(f"{field} must be a stable identifier")
    return value


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_TEXT:
        raise TraceContractError(f"{field} must be non-empty bounded text")
    if any(ord(character) < 32 and character not in "\t\n" for character in value):
        raise TraceContractError(f"{field} contains a control character")
    return value.strip()


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TraceContractError(f"{field} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise TraceContractError(f"{field} must be finite")
    return resolved


def _enum(value: object, enum_type: type[StrEnum], *, field: str) -> StrEnum:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise TraceContractError(
            f"{field} must be one of {[item.value for item in enum_type]!r}"
        ) from exc


def _exact_mapping(
    value: object,
    *,
    field: str,
    keys: frozenset[str],
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TraceContractError(f"{field} must be an object")
    actual = set(value)
    missing = sorted(keys - actual)
    extra = sorted(actual - keys)
    if missing or extra:
        raise TraceContractError(f"{field} fields mismatch: missing={missing!r}, extra={extra!r}")
    return value


def _array(value: object, *, field: str) -> list[object]:
    if not isinstance(value, list):
        raise TraceContractError(f"{field} must be a JSON array")
    return value


def _compact(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    role: str
    artifact_id: str
    sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _identifier(self.role, field="artifact role"))
        object.__setattr__(
            self,
            "artifact_id",
            _identifier(self.artifact_id, field="artifact id"),
        )
        if not isinstance(self.sha256, str) or not _SHA256.fullmatch(self.sha256):
            raise TraceContractError("artifact sha256 must be lowercase SHA-256")

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "artifact_id": self.artifact_id,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class NumericSignalSpec:
    name: str
    role: TraceRole
    shape: tuple[int, ...]
    unit: str
    frame: str
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, field="signal name"))
        object.__setattr__(self, "role", _enum(self.role, TraceRole, field="signal role"))
        if (
            not isinstance(self.shape, Sequence)
            or isinstance(self.shape, (str, bytes))
            or not 1 <= len(self.shape) <= 4
        ):
            raise TraceContractError("signal shape must have one to four dimensions")
        resolved_shape = tuple(self.shape)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in resolved_shape
        ):
            raise TraceContractError("signal shape dimensions must be positive integers")
        width = math.prod(resolved_shape)
        if width > MAX_SIGNAL_WIDTH:
            raise TraceContractError(f"signal width cannot exceed {MAX_SIGNAL_WIDTH}")
        object.__setattr__(self, "shape", resolved_shape)
        object.__setattr__(self, "unit", _text(self.unit, field="signal unit"))
        object.__setattr__(self, "frame", _text(self.frame, field="signal frame"))
        object.__setattr__(self, "source", _text(self.source, field="signal source"))

    @property
    def width(self) -> int:
        return math.prod(self.shape)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "role": self.role.value,
            "shape": list(self.shape),
            "unit": self.unit,
            "frame": self.frame,
            "source": self.source,
            "dtype": "float64",
        }


@dataclass(frozen=True, slots=True)
class MissingSignal:
    name: str
    reason: MissingReason
    detail: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, field="missing signal name"))
        object.__setattr__(
            self,
            "reason",
            _enum(self.reason, MissingReason, field="missing signal reason"),
        )
        object.__setattr__(self, "detail", _text(self.detail, field="missing signal detail"))

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "reason": self.reason.value,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class TraceEvent:
    sample_index: int
    event_type: str
    value: str
    source: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sample_index, int)
            or isinstance(self.sample_index, bool)
            or self.sample_index < 0
        ):
            raise TraceContractError("event sample_index must be an integer >= 0")
        object.__setattr__(
            self,
            "event_type",
            _identifier(self.event_type, field="event type"),
        )
        object.__setattr__(self, "value", _text(self.value, field="event value"))
        object.__setattr__(self, "source", _text(self.source, field="event source"))

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_index": self.sample_index,
            "event_type": self.event_type,
            "value": self.value,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class _TraceSample:
    sample_index: int
    time_seconds: float
    numeric_values: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sample_index, int)
            or isinstance(self.sample_index, bool)
            or self.sample_index < 0
        ):
            raise TraceContractError("sample_index must be an integer >= 0")
        object.__setattr__(
            self,
            "time_seconds",
            _finite(self.time_seconds, field="sample time_seconds"),
        )
        resolved_values: list[tuple[float, ...]] = []
        for signal_index, raw_values in enumerate(self.numeric_values):
            if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)):
                raise TraceContractError(f"sample signal {signal_index} values must be an array")
            resolved_values.append(
                tuple(_finite(value, field=f"sample signal {signal_index}") for value in raw_values)
            )
        object.__setattr__(self, "numeric_values", tuple(resolved_values))

    def to_dict(self, signals: tuple[NumericSignalSpec, ...]) -> dict[str, object]:
        return {
            "sample_index": self.sample_index,
            "time_seconds": self.time_seconds,
            "signals": {
                signal.name: list(values)
                for signal, values in zip(signals, self.numeric_values, strict=True)
            },
        }


@dataclass(frozen=True, slots=True)
class TrajectoryTrace:
    trace_id: str
    evidence_class: EvidenceClass
    control_period_seconds: float
    numeric_signals: tuple[NumericSignalSpec, ...]
    missing_signals: tuple[MissingSignal, ...]
    artifact_bindings: tuple[ArtifactBinding, ...]
    samples: tuple[_TraceSample, ...]
    events: tuple[TraceEvent, ...] = ()
    schema_version: int = TRACE_SCHEMA_VERSION
    sample_semantics: str = TRACE_SEMANTICS

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != TRACE_SCHEMA_VERSION
        ):
            raise TraceContractError(f"schema_version must be {TRACE_SCHEMA_VERSION}")
        if self.sample_semantics != TRACE_SEMANTICS:
            raise TraceContractError("sample_semantics is unsupported")
        object.__setattr__(self, "trace_id", _identifier(self.trace_id, field="trace id"))
        object.__setattr__(
            self,
            "evidence_class",
            _enum(self.evidence_class, EvidenceClass, field="evidence class"),
        )
        control_period = _finite(self.control_period_seconds, field="control_period_seconds")
        if control_period <= 0.0:
            raise TraceContractError("control_period_seconds must be positive")
        object.__setattr__(self, "control_period_seconds", control_period)

        signals = tuple(self.numeric_signals)
        missing = tuple(self.missing_signals)
        bindings = tuple(self.artifact_bindings)
        samples = tuple(self.samples)
        events = tuple(self.events)
        object.__setattr__(self, "numeric_signals", signals)
        object.__setattr__(self, "missing_signals", missing)
        object.__setattr__(self, "artifact_bindings", bindings)
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "events", events)

        if not 1 <= len(signals) <= MAX_SIGNALS:
            raise TraceContractError(f"numeric signal count must be in [1, {MAX_SIGNALS}]")
        if len(missing) > MAX_MISSING_SIGNALS:
            raise TraceContractError(f"missing signal count cannot exceed {MAX_MISSING_SIGNALS}")
        if len(bindings) > MAX_ARTIFACT_BINDINGS:
            raise TraceContractError(
                f"artifact binding count cannot exceed {MAX_ARTIFACT_BINDINGS}"
            )
        if not 1 <= len(samples) <= MAX_SAMPLES:
            raise TraceContractError(f"sample count must be in [1, {MAX_SAMPLES}]")
        if len(events) > MAX_EVENTS:
            raise TraceContractError(f"event count cannot exceed {MAX_EVENTS}")
        if any(not isinstance(item, NumericSignalSpec) for item in signals):
            raise TraceContractError("numeric_signals must contain NumericSignalSpec values")
        if any(not isinstance(item, MissingSignal) for item in missing):
            raise TraceContractError("missing_signals must contain MissingSignal values")
        if any(not isinstance(item, ArtifactBinding) for item in bindings):
            raise TraceContractError("artifact_bindings must contain ArtifactBinding values")
        if any(not isinstance(item, _TraceSample) for item in samples):
            raise TraceContractError("samples contain an unsupported value")
        if any(not isinstance(item, TraceEvent) for item in events):
            raise TraceContractError("events must contain TraceEvent values")

        signal_names = tuple(signal.name for signal in signals)
        missing_names = tuple(item.name for item in missing)
        if len(set(signal_names)) != len(signal_names):
            raise TraceContractError("numeric signal names must be unique")
        if len(set(missing_names)) != len(missing_names):
            raise TraceContractError("missing signal names must be unique")
        if set(signal_names) & set(missing_names):
            raise TraceContractError("a signal cannot be both recorded and missing")
        coverage = set(signal_names) | set(missing_names)
        uncovered = sorted(REQUIRED_DIAGNOSTIC_SIGNALS - coverage)
        if uncovered:
            raise TraceContractError(f"diagnostic signal coverage is missing: {uncovered!r}")
        role_mismatches = sorted(
            signal.name
            for signal in signals
            if signal.name in REQUIRED_DIAGNOSTIC_ROLES
            and signal.role is not REQUIRED_DIAGNOSTIC_ROLES[signal.name]
        )
        if role_mismatches:
            raise TraceContractError(f"diagnostic signal roles are invalid: {role_mismatches!r}")
        if (
            self.evidence_class is EvidenceClass.BEHAVIORAL_EVALUATION
            and REQUIRED_DIAGNOSTIC_SIGNALS & set(missing_names)
        ):
            raise TraceContractError(
                "behavioral evaluation must record every required diagnostic signal"
            )

        binding_roles = tuple(binding.role for binding in bindings)
        if len(set(binding_roles)) != len(binding_roles):
            raise TraceContractError("artifact binding roles must be unique")
        required_roles = (
            BEHAVIORAL_BINDING_ROLES
            if self.evidence_class is EvidenceClass.BEHAVIORAL_EVALUATION
            else frozenset({"runtime"})
        )
        missing_roles = sorted(required_roles - set(binding_roles))
        if missing_roles:
            raise TraceContractError(f"artifact bindings are missing roles: {missing_roles!r}")

        for expected_index, sample in enumerate(samples):
            if sample.sample_index != expected_index:
                raise TraceContractError("sample indices must be contiguous from zero")
            expected_time = expected_index * control_period
            if not math.isclose(
                sample.time_seconds,
                expected_time,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise TraceContractError("sample time does not match index and control period")
            if len(sample.numeric_values) != len(signals):
                raise TraceContractError("sample signal count does not match the schema")
            for signal, values in zip(signals, sample.numeric_values, strict=True):
                if len(values) != signal.width:
                    raise TraceContractError(
                        f"sample signal {signal.name!r} does not match shape {signal.shape!r}"
                    )
                if any(not math.isfinite(value) for value in values):
                    raise TraceContractError(
                        f"sample signal {signal.name!r} must contain only finite values"
                    )
        if any(event.sample_index >= len(samples) for event in events):
            raise TraceContractError("event sample_index is outside the trace")

        if "controller.action" in signal_names:
            action_index = signal_names.index("controller.action")
            if any(value != 0.0 for value in samples[0].numeric_values[action_index]):
                raise TraceContractError("controller.action must be zero at the reset-state sample")

        if (
            self.evidence_class is EvidenceClass.BEHAVIORAL_EVALUATION
            and "oracle.mode" in signal_names
        ):
            mode_index = signal_names.index("oracle.mode")
            transition_samples = {
                event.sample_index for event in events if event.event_type == "oracle.transition"
            }
            unexplained = [
                index
                for index in range(1, len(samples))
                if samples[index].numeric_values[mode_index]
                != samples[index - 1].numeric_values[mode_index]
                and index not in transition_samples
            ]
            if unexplained:
                raise TraceContractError(
                    f"oracle mode changes require oracle.transition events: {unexplained!r}"
                )

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    @property
    def canonical_bytes(self) -> bytes:
        return _compact(self.to_dict())

    @property
    def diagnostic_coverage(self) -> dict[str, list[str]]:
        return {
            "recorded": sorted(
                REQUIRED_DIAGNOSTIC_SIGNALS & {signal.name for signal in self.numeric_signals}
            ),
            "missing": sorted(
                REQUIRED_DIAGNOSTIC_SIGNALS & {signal.name for signal in self.missing_signals}
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "trace_id": self.trace_id,
            "evidence_class": self.evidence_class.value,
            "sample_semantics": self.sample_semantics,
            "control_period_seconds": self.control_period_seconds,
            "numeric_signals": [signal.to_dict() for signal in self.numeric_signals],
            "missing_signals": [signal.to_dict() for signal in self.missing_signals],
            "artifact_bindings": [binding.to_dict() for binding in self.artifact_bindings],
            "samples": [sample.to_dict(self.numeric_signals) for sample in self.samples],
            "events": [event.to_dict() for event in self.events],
        }

    def write(self, path: Path) -> Path:
        content = self.canonical_bytes
        if len(content) > MAX_TRACE_BYTES:
            raise TraceContractError(f"trajectory trace exceeds {MAX_TRACE_BYTES} bytes")
        resolved = path.resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        try:
            descriptor = os.open(resolved, flags, 0o600)
        except FileExistsError as exc:
            raise TraceContractError(f"trace output already exists: {resolved}") from exc
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        return resolved


class TraceRecorder:
    """Build one trace while enforcing schema order and finite values."""

    def __init__(
        self,
        *,
        trace_id: str,
        evidence_class: EvidenceClass,
        control_period_seconds: float,
        numeric_signals: Sequence[NumericSignalSpec],
        missing_signals: Sequence[MissingSignal],
        artifact_bindings: Sequence[ArtifactBinding],
    ) -> None:
        self._trace_id = trace_id
        self._evidence_class = evidence_class
        self._control_period_seconds = _finite(
            control_period_seconds, field="control_period_seconds"
        )
        if self._control_period_seconds <= 0.0:
            raise TraceContractError("control_period_seconds must be positive")
        self._numeric_signals = tuple(numeric_signals)
        self._missing_signals = tuple(missing_signals)
        self._artifact_bindings = tuple(artifact_bindings)
        self._samples: list[_TraceSample] = []
        self._events: list[TraceEvent] = []

    def add_sample(self, values: Mapping[str, object]) -> int:
        if len(self._samples) >= MAX_SAMPLES:
            raise TraceContractError(f"sample count cannot exceed {MAX_SAMPLES}")
        expected = tuple(signal.name for signal in self._numeric_signals)
        if set(values) != set(expected):
            raise TraceContractError(
                "sample signal keys mismatch: "
                f"missing={sorted(set(expected) - set(values))!r}, "
                f"extra={sorted(set(values) - set(expected))!r}"
            )
        resolved_values: list[tuple[float, ...]] = []
        for signal in self._numeric_signals:
            raw = values[signal.name]
            if isinstance(raw, np.ndarray):
                raw = raw.reshape(-1).tolist()
            elif isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
                raw = (raw,)
            flattened = tuple(
                _finite(value, field=f"sample signal {signal.name!r}") for value in raw
            )
            if len(flattened) != signal.width:
                raise TraceContractError(
                    f"sample signal {signal.name!r} must have {signal.width} values"
                )
            resolved_values.append(flattened)
        index = len(self._samples)
        self._samples.append(
            _TraceSample(
                sample_index=index,
                time_seconds=index * self._control_period_seconds,
                numeric_values=tuple(resolved_values),
            )
        )
        return index

    def add_event(
        self,
        *,
        sample_index: int,
        event_type: str,
        value: str,
        source: str,
    ) -> None:
        if len(self._events) >= MAX_EVENTS:
            raise TraceContractError(f"event count cannot exceed {MAX_EVENTS}")
        self._events.append(TraceEvent(sample_index, event_type, value, source))

    def finish(self) -> TrajectoryTrace:
        return TrajectoryTrace(
            trace_id=self._trace_id,
            evidence_class=self._evidence_class,
            control_period_seconds=self._control_period_seconds,
            numeric_signals=self._numeric_signals,
            missing_signals=self._missing_signals,
            artifact_bindings=self._artifact_bindings,
            samples=tuple(self._samples),
            events=tuple(self._events),
        )


def load_trace(path: Path) -> TrajectoryTrace:
    try:
        size = path.stat().st_size
        if size > MAX_TRACE_BYTES:
            raise TraceContractError(f"trajectory trace exceeds {MAX_TRACE_BYTES} bytes")
        source_bytes = path.read_bytes()
        if len(source_bytes) > MAX_TRACE_BYTES:
            raise TraceContractError(f"trajectory trace exceeds {MAX_TRACE_BYTES} bytes")
        raw = json.loads(source_bytes)
    except TraceContractError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TraceContractError(f"cannot read trajectory trace: {path}") from exc
    root = _exact_mapping(
        raw,
        field="trajectory trace",
        keys=frozenset(
            {
                "schema_version",
                "trace_id",
                "evidence_class",
                "sample_semantics",
                "control_period_seconds",
                "numeric_signals",
                "missing_signals",
                "artifact_bindings",
                "samples",
                "events",
            }
        ),
    )

    signal_values = _array(root["numeric_signals"], field="numeric_signals")
    if not 1 <= len(signal_values) <= MAX_SIGNALS:
        raise TraceContractError(f"numeric signal count must be in [1, {MAX_SIGNALS}]")
    signals: list[NumericSignalSpec] = []
    for index, raw_signal in enumerate(signal_values):
        signal = _exact_mapping(
            raw_signal,
            field=f"numeric_signals[{index}]",
            keys=frozenset({"name", "role", "shape", "unit", "frame", "source", "dtype"}),
        )
        if signal["dtype"] != "float64":
            raise TraceContractError(f"numeric_signals[{index}].dtype must be 'float64'")
        shape = _array(signal["shape"], field=f"numeric_signals[{index}].shape")
        signals.append(
            NumericSignalSpec(
                name=signal["name"],
                role=signal["role"],
                shape=tuple(shape),
                unit=signal["unit"],
                frame=signal["frame"],
                source=signal["source"],
            )
        )

    missing_values = _array(root["missing_signals"], field="missing_signals")
    if len(missing_values) > MAX_MISSING_SIGNALS:
        raise TraceContractError(f"missing signal count cannot exceed {MAX_MISSING_SIGNALS}")
    missing: list[MissingSignal] = []
    for index, raw_missing in enumerate(missing_values):
        item = _exact_mapping(
            raw_missing,
            field=f"missing_signals[{index}]",
            keys=frozenset({"name", "reason", "detail"}),
        )
        missing.append(MissingSignal(item["name"], item["reason"], item["detail"]))

    binding_values = _array(root["artifact_bindings"], field="artifact_bindings")
    if len(binding_values) > MAX_ARTIFACT_BINDINGS:
        raise TraceContractError(f"artifact binding count cannot exceed {MAX_ARTIFACT_BINDINGS}")
    bindings: list[ArtifactBinding] = []
    for index, raw_binding in enumerate(binding_values):
        item = _exact_mapping(
            raw_binding,
            field=f"artifact_bindings[{index}]",
            keys=frozenset({"role", "artifact_id", "sha256"}),
        )
        bindings.append(ArtifactBinding(item["role"], item["artifact_id"], item["sha256"]))

    signal_names = frozenset(signal.name for signal in signals)
    sample_values = _array(root["samples"], field="samples")
    if not 1 <= len(sample_values) <= MAX_SAMPLES:
        raise TraceContractError(f"sample count must be in [1, {MAX_SAMPLES}]")
    samples: list[_TraceSample] = []
    for index, raw_sample in enumerate(sample_values):
        item = _exact_mapping(
            raw_sample,
            field=f"samples[{index}]",
            keys=frozenset({"sample_index", "time_seconds", "signals"}),
        )
        signal_map = _exact_mapping(
            item["signals"],
            field=f"samples[{index}].signals",
            keys=signal_names,
        )
        numeric_values = tuple(
            tuple(
                _array(
                    signal_map[signal.name],
                    field=f"samples[{index}].signals.{signal.name}",
                )
            )
            for signal in signals
        )
        samples.append(
            _TraceSample(
                sample_index=item["sample_index"],
                time_seconds=item["time_seconds"],
                numeric_values=numeric_values,
            )
        )

    event_values = _array(root["events"], field="events")
    if len(event_values) > MAX_EVENTS:
        raise TraceContractError(f"event count cannot exceed {MAX_EVENTS}")
    events: list[TraceEvent] = []
    for index, raw_event in enumerate(event_values):
        item = _exact_mapping(
            raw_event,
            field=f"events[{index}]",
            keys=frozenset({"sample_index", "event_type", "value", "source"}),
        )
        events.append(
            TraceEvent(
                item["sample_index"],
                item["event_type"],
                item["value"],
                item["source"],
            )
        )

    trace = TrajectoryTrace(
        schema_version=root["schema_version"],
        trace_id=root["trace_id"],
        evidence_class=root["evidence_class"],
        sample_semantics=root["sample_semantics"],
        control_period_seconds=root["control_period_seconds"],
        numeric_signals=tuple(signals),
        missing_signals=tuple(missing),
        artifact_bindings=tuple(bindings),
        samples=tuple(samples),
        events=tuple(events),
    )
    if source_bytes != trace.canonical_bytes:
        raise TraceContractError("trajectory trace bytes are not canonical")
    return trace
