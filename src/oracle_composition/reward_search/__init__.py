"""Portable, data-only task-reward proposal plumbing."""

from .contracts import (  # noqa: F401 - package API re-exports
    AggregateMeasurement,
    AuthorSurface,
    CandidateSnapshot,
    EvidenceProvenance,
    FrozenConfigurationHash,
    IterationRecord,
    MissingEvidence,
    ModelCallReceipt,
    PacketRecord,
    ProtectedEvidenceDossier,
    RewardProposal,
    SourceBundle,
    SourceCard,
)
from .loop import (  # noqa: F401 - package API re-exports
    IngestionOutcome,
    RewardSearchError,
    ingest_sol_run,
    load_model,
    model_sha256,
    prepare_initial_packet,
    prepare_revision_packet,
    publish_packet,
    render_packet,
    source_bundle_from_index,
)
