"""Canonical synchronized trajectory evidence."""

from .contract import (
    BEHAVIORAL_BINDING_ROLES,
    REQUIRED_DIAGNOSTIC_ROLES,
    REQUIRED_DIAGNOSTIC_SIGNALS,
    ArtifactBinding,
    EvidenceClass,
    MissingReason,
    MissingSignal,
    NumericSignalSpec,
    TraceContractError,
    TraceEvent,
    TraceRecorder,
    TraceRole,
    TrajectoryTrace,
    load_trace,
)

__all__ = [
    "BEHAVIORAL_BINDING_ROLES",
    "REQUIRED_DIAGNOSTIC_ROLES",
    "REQUIRED_DIAGNOSTIC_SIGNALS",
    "ArtifactBinding",
    "EvidenceClass",
    "MissingReason",
    "MissingSignal",
    "NumericSignalSpec",
    "TraceContractError",
    "TraceEvent",
    "TraceRecorder",
    "TraceRole",
    "TrajectoryTrace",
    "load_trace",
]
