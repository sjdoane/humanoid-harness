"""Data-only source-format inspection utilities.

Nothing in this package retargets a motion, imports a simulator, or admits a
source artifact for controller training.
"""

from oracle_composition.sources.deepmimic import (
    DeepMimicFormatError,
    audit_source_tree,
    inspect_motion_bytes,
    render_audit_json,
    verify_audit_payload_hash,
)
from oracle_composition.sources.minari_humanoid import (
    HUMANOID_V5_OBSERVATION_TO_REFERENCE_INDICES,
    MinariHumanoidImportError,
    MinariHumanoidProjection,
    MinariHumanoidProjectionReceipt,
    import_minari_humanoid_episode,
    import_registered_minari_humanoid_episode,
)

__all__ = [
    "HUMANOID_V5_OBSERVATION_TO_REFERENCE_INDICES",
    "DeepMimicFormatError",
    "MinariHumanoidImportError",
    "MinariHumanoidProjection",
    "MinariHumanoidProjectionReceipt",
    "audit_source_tree",
    "import_minari_humanoid_episode",
    "import_registered_minari_humanoid_episode",
    "inspect_motion_bytes",
    "render_audit_json",
    "verify_audit_payload_hash",
]
