"""Protected evaluation contracts."""

from oracle_composition.evaluation.admission import (
    AdmissionDecision,
    OracleTrainingEvidence,
    classify_oracle_training_evidence,
)

__all__ = [
    "AdmissionDecision",
    "OracleTrainingEvidence",
    "classify_oracle_training_evidence",
]
