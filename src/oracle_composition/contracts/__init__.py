"""Public immutable contracts for oracle composition."""

from .errors import OracleContractError
from .program import (
    GuardOperator,
    ModeKind,
    ModeSpec,
    OracleProgram,
    PhaseTransferKind,
    PhaseTransferSpec,
    ScalarGuard,
    SignalSpec,
    SignalVisibility,
    TransitionKind,
    TransitionRule,
    validate_program,
)
from .reference import (
    ReferenceArtifact,
    ReferenceIdentity,
    ReferenceSchema,
    ReferenceWindow,
    assert_exact_schema,
    reference_content_sha256,
)

__all__ = [
    "GuardOperator",
    "ModeKind",
    "ModeSpec",
    "OracleContractError",
    "OracleProgram",
    "PhaseTransferKind",
    "PhaseTransferSpec",
    "ReferenceArtifact",
    "ReferenceIdentity",
    "ReferenceSchema",
    "ReferenceWindow",
    "ScalarGuard",
    "SignalSpec",
    "SignalVisibility",
    "TransitionKind",
    "TransitionRule",
    "assert_exact_schema",
    "reference_content_sha256",
    "validate_program",
]
