"""Validated, data-only evidence-to-diagnosis feedback surface."""

from .context import CandidateContext, FeedbackContextError, build_candidate_context
from .evidence import (
    FeedbackDiagnosis,
    FeedbackEvidenceError,
    ObservedFact,
    SteeringInput,
    diagnose_feedback,
)

__all__ = [
    "CandidateContext",
    "FeedbackContextError",
    "FeedbackDiagnosis",
    "FeedbackEvidenceError",
    "ObservedFact",
    "SteeringInput",
    "build_candidate_context",
    "diagnose_feedback",
]
