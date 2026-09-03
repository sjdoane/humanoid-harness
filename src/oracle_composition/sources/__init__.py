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

__all__ = [
    "DeepMimicFormatError",
    "audit_source_tree",
    "inspect_motion_bytes",
    "render_audit_json",
    "verify_audit_payload_hash",
]
