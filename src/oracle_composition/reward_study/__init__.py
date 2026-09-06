"""Frozen artifacts and protected analysis for the T2 reward study."""

from .execution_manifest import (
    T2_EXECUTION_MANIFEST_SCHEMA_ID,
    load_t2_execution_manifest,
)
from .pairing import (
    PAIRING_DERIVATION_ID,
    derive_evaluation_stream_seeds,
    derive_training_stream_seeds,
)
from .study_manifest import (
    STUDY_ID,
    STUDY_MANIFEST_SCHEMA_ID,
    load_t2_study_manifest,
    validate_t2_study_manifest,
)
from .t2_evaluator import (
    T2_EVALUATION_SEEDS,
    T2_EVALUATOR_ID,
    T2_REPORT_SCHEMA_ID,
    T2EpisodeMetrics,
    evaluate_t2_trace,
)

__all__ = [
    "PAIRING_DERIVATION_ID",
    "STUDY_ID",
    "STUDY_MANIFEST_SCHEMA_ID",
    "T2_EVALUATION_SEEDS",
    "T2_EVALUATOR_ID",
    "T2_EXECUTION_MANIFEST_SCHEMA_ID",
    "T2_REPORT_SCHEMA_ID",
    "T2EpisodeMetrics",
    "derive_evaluation_stream_seeds",
    "derive_training_stream_seeds",
    "evaluate_t2_trace",
    "load_t2_execution_manifest",
    "load_t2_study_manifest",
    "validate_t2_study_manifest",
]
