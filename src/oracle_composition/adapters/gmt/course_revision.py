"""Publish one lineage-bound, data-only G1 course revision candidate."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from pathlib import Path

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.experiments.artifact_io import (
    PublishedArtifact,
    finite_pretty_json,
    publish_bytes_without_overwrite,
    publish_json_without_overwrite,
)
from oracle_composition.feedback.g1_course import build_g1_course_feedback
from oracle_composition.harness.contract import read_json_object

from .course_config import load_run_config
from .course_proposal import apply_proposal

FEEDBACK_BUILD_RECEIPT_ARTIFACT = "gmt_g1_course_feedback_build_receipt/v1"
REVISION_RECEIPT_ARTIFACT = "gmt_g1_course_revision_receipt/v1"

_BUILD_RECEIPT_FIELDS = {
    "schema_version",
    "artifact",
    "status",
    "inputs",
    "output",
    "claim_limits",
}
_BUILD_INPUT_FIELDS = {
    "source_manifest_path",
    "source_manifest_sha256",
    "input_config_sha256",
    "label",
    "selected_outputs",
}
_BUILD_INPUT_FIELDS_WITH_RUNTIME = {*_BUILD_INPUT_FIELDS, "course_runtime"}
_BUILD_OUTPUT_FIELDS = {"path", "sha256", "byte_count"}
_LABELS = {"zero_residual", "final_policy"}


def _sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be one lowercase SHA-256")
    return value


def _read_exact_json(path: Path, expected_sha256: str, *, field: str) -> tuple[dict, bytes]:
    expected = _sha256(expected_sha256, field=f"{field} SHA-256")
    value, encoded = read_json_object(Path(path))
    if hashlib.sha256(encoded).hexdigest() != expected:
        raise ValueError(f"{field} SHA-256 mismatch")
    return value, encoded


def _fresh_output_path(output: Path) -> Path:
    destination = Path(os.path.abspath(output))
    if destination.name in {"", ".", ".."} or "\0" in destination.name:
        raise ValueError("revision output must name one fresh directory")
    try:
        destination.lstat()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise ValueError("revision output cannot be inspected") from exc
    else:
        raise ValueError("revision output must not already exist")
    try:
        parent = destination.parent.lstat()
    except OSError as exc:
        raise ValueError("revision output parent must already exist") from exc
    if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
        raise ValueError("revision output parent must be a real directory")
    if destination.parent.resolve(strict=True) != destination.parent:
        raise ValueError("revision output ancestors must not contain symbolic links")
    return destination


def _artifact_record(artifact: PublishedArtifact, *, source_path: Path | None = None) -> dict:
    record: dict[str, object] = {
        "path": artifact.path.name,
        "sha256": artifact.sha256,
        "byte_count": artifact.byte_count,
    }
    if source_path is not None:
        record["source_path"] = str(Path(os.path.abspath(source_path)))
    return record


def _verify_current_feedback_packet(
    *,
    manifest_path: Path,
    manifest_sha256: str,
    label: str,
    parent_sha256: str,
    supplied_feedback: bytes,
    supplied_feedback_sha256: str,
    expected_course_runtime: dict[str, object] | None,
) -> bytes:
    if label not in _LABELS:
        raise ValueError("source label must be zero_residual or final_policy")
    with tempfile.TemporaryDirectory(prefix="gmt-course-feedback-verification-") as temporary:
        # macOS may return /var, an alias of /private/var, for our owned workspace.
        rebuilt_root = Path(temporary).resolve(strict=True) / "packet"
        build_g1_course_feedback(
            manifest_path=manifest_path,
            expected_manifest_sha256=manifest_sha256,
            label=label,
            output=rebuilt_root,
        )
        _rebuilt_value, rebuilt_feedback = _read_exact_json(
            rebuilt_root / "feedback_v1.json",
            supplied_feedback_sha256,
            field="rebuilt feedback",
        )
        if rebuilt_feedback != supplied_feedback:
            raise ValueError("supplied feedback is not the exact current rebuilt packet")
        receipt, receipt_bytes = read_json_object(rebuilt_root / "feedback_receipt_v1.json")
        if (
            set(receipt) != _BUILD_RECEIPT_FIELDS
            or receipt.get("schema_version") != 1
            or receipt.get("artifact") != FEEDBACK_BUILD_RECEIPT_ARTIFACT
            or receipt.get("status") != "completed"
        ):
            raise ValueError("rebuilt feedback receipt version or fields differ")
        inputs = receipt.get("inputs")
        output = receipt.get("output")
        expected_input_fields = (
            _BUILD_INPUT_FIELDS_WITH_RUNTIME
            if expected_course_runtime is not None
            else _BUILD_INPUT_FIELDS
        )
        if (
            type(inputs) is not dict
            or set(inputs) != expected_input_fields
            or inputs.get("source_manifest_sha256") != manifest_sha256
            or inputs.get("input_config_sha256") != parent_sha256
            or inputs.get("label") != label
            or (
                expected_course_runtime is not None
                and inputs.get("course_runtime") != expected_course_runtime
            )
            or type(output) is not dict
            or set(output) != _BUILD_OUTPUT_FIELDS
            or output.get("path") != "feedback_v1.json"
            or output.get("sha256") != supplied_feedback_sha256
            or output.get("byte_count") != len(supplied_feedback)
        ):
            raise ValueError("rebuilt feedback lineage differs from the requested parent run")
        return receipt_bytes


def revise_g1_course(
    *,
    parent_config_path: Path,
    expected_parent_config_sha256: str,
    feedback_path: Path,
    expected_feedback_sha256: str,
    proposal_path: Path,
    expected_proposal_sha256: str,
    source_manifest_path: Path,
    expected_source_manifest_sha256: str,
    source_label: str,
    output: Path,
) -> dict[str, object]:
    """Verify one source run and publish one untrained revision candidate."""

    destination = _fresh_output_path(output)
    manifest_sha256 = _sha256(expected_source_manifest_sha256, field="source manifest SHA-256")
    feedback_sha256 = _sha256(expected_feedback_sha256, field="feedback SHA-256")
    parent_sha256 = _sha256(expected_parent_config_sha256, field="parent config SHA-256")
    proposal_sha256 = _sha256(expected_proposal_sha256, field="proposal SHA-256")

    _manifest, manifest_bytes = _read_exact_json(
        source_manifest_path, manifest_sha256, field="source manifest"
    )
    _parent_value, parent_bytes = _read_exact_json(
        parent_config_path, parent_sha256, field="parent config"
    )
    parent = load_run_config(parent_config_path)
    if parent.sha256 != parent_sha256 or parent.encoded != parent_bytes:
        raise ValueError("admitted parent config differs from the exact requested bytes")
    proposal, proposal_bytes = _read_exact_json(proposal_path, proposal_sha256, field="proposal")
    _feedback, feedback_bytes = _read_exact_json(feedback_path, feedback_sha256, field="feedback")
    feedback_build_receipt_bytes = _verify_current_feedback_packet(
        manifest_path=source_manifest_path,
        manifest_sha256=manifest_sha256,
        label=source_label,
        parent_sha256=parent_sha256,
        supplied_feedback=feedback_bytes,
        supplied_feedback_sha256=feedback_sha256,
        expected_course_runtime=parent.runtime.manifest_contract(),
    )
    # Re-read the manifest after the independent rebuild so the retained bytes
    # are the same bytes whose path remains present at publication time.
    _manifest_after, manifest_after = _read_exact_json(
        source_manifest_path, manifest_sha256, field="source manifest"
    )
    if manifest_after != manifest_bytes:
        raise ValueError("source manifest changed during revision verification")

    candidate = apply_proposal(parent, proposal, feedback_bytes)
    candidate_bytes = finite_pretty_json(candidate)
    factor = proposal.get("factor")
    if factor not in {"oracle", "reward"}:
        raise ValueError("proposal factor must be oracle or reward")
    allowed_changed_fields = ["oracle", "segments"] if factor == "oracle" else ["reward"]
    actual_changed_fields = sorted(
        field
        for field in parent.raw
        if canonical_json_bytes(candidate[field]) != canonical_json_bytes(parent.raw[field])
    )

    destination.mkdir(mode=0o700, parents=False, exist_ok=False)
    retained = {
        "parent_config": publish_bytes_without_overwrite(
            destination / "parent_config.input.json", parent_bytes
        ),
        "feedback": publish_bytes_without_overwrite(
            destination / "feedback.input.json", feedback_bytes
        ),
        "proposal": publish_bytes_without_overwrite(
            destination / "proposal.input.json", proposal_bytes
        ),
        "source_manifest": publish_bytes_without_overwrite(
            destination / "source_manifest.input.json", manifest_bytes
        ),
    }
    candidate_artifact = publish_bytes_without_overwrite(
        destination / "candidate_config.json", candidate_bytes
    )
    feedback_build_receipt_artifact = publish_bytes_without_overwrite(
        destination / "feedback_verification_receipt.json",
        feedback_build_receipt_bytes,
    )
    source_paths = {
        "parent_config": parent_config_path,
        "feedback": feedback_path,
        "proposal": proposal_path,
        "source_manifest": source_manifest_path,
    }
    receipt = {
        "schema_version": 1,
        "artifact": REVISION_RECEIPT_ARTIFACT,
        "status": "completed",
        "operation": "apply_one_authorable_g1_course_proposal",
        "proposal_id": proposal["proposal_id"],
        "factor": factor,
        "actual_changed_fields": actual_changed_fields,
        "allowed_changed_fields": allowed_changed_fields,
        "source_run": {
            "manifest_sha256": manifest_sha256,
            "label": source_label,
            "input_config_sha256": parent_sha256,
            "feedback_builder_packet": {
                "artifact": FEEDBACK_BUILD_RECEIPT_ARTIFACT,
                **_artifact_record(feedback_build_receipt_artifact),
            },
        },
        "retained_inputs": {
            name: _artifact_record(artifact, source_path=source_paths[name])
            for name, artifact in retained.items()
        },
        "candidate": _artifact_record(candidate_artifact),
        "claim_limits": [
            "data_only_candidate",
            "one_authorable_factor_only",
            "no_provider_call",
            "not_trained_or_evaluated",
            "not_protected_or_task_success_evidence",
        ],
    }
    receipt_artifact = publish_json_without_overwrite(
        destination / "revision_receipt_v1.json", receipt
    )
    return {
        "status": "completed",
        "factor": factor,
        "candidate": _artifact_record(candidate_artifact),
        "receipt": _artifact_record(receipt_artifact),
        "output": str(destination),
    }


__all__ = [
    "FEEDBACK_BUILD_RECEIPT_ARTIFACT",
    "REVISION_RECEIPT_ARTIFACT",
    "revise_g1_course",
]
