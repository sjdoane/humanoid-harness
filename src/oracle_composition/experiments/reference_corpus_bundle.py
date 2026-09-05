"""Content-addressed publication and admission of full-clip replay bundles."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from oracle_composition.contracts.reference_identity_v2 import (
    ArtifactBindingV2,
    ReferenceIdentityV2,
    array_sha256,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)
from oracle_composition.sources.strict_tqc_actor_runtime import INFERENCE_ID

from .artifact_io import publish_bytes_without_overwrite
from .reference_corpus_collector import CollectedClip
from .reference_corpus_contract import (
    CONTACT_SEQUENCE_ID,
    POLICY_INPUT_ID,
    WRAPPER_FLAG_NAMES,
    WRAPPER_STATE_ID,
    array_bindings,
    encode_clip_payload,
    plain_comparison_receipt,
    reference_schema_payload,
    validate_bundle_manifest,
)

BUNDLE_PUBLISHER_ID = "content_addressed_reference_replay_bundle_publisher/v2"


@dataclass(frozen=True, slots=True)
class PublishedClipBundle:
    clip_id: str
    manifest_path: Path
    manifest_sha256: str
    payload_path: Path
    payload_sha256: str
    reference_identity_sha256: str


def _safe_relative(path: Path, root: Path) -> str:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("published bundle artifact escaped its root") from exc
    if ".." in relative.parts:
        raise ValueError("published bundle artifact path is unsafe")
    return relative.as_posix()


def publish_object(
    source: Path,
    *,
    artifact_root: Path,
    role: str,
    logical_path: str | None = None,
) -> ArtifactBindingV2:
    """Copy one exact local file into the shared SHA-256 object store."""

    candidate = Path(source)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError(f"bound source is not a regular file: {candidate}")
    payload = candidate.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    destination = artifact_root / "objects" / "sha256" / digest[:2] / digest
    if destination.exists():
        if destination.is_symlink() or not destination.is_file():
            raise ValueError("content-addressed object path is not a regular file")
        if destination.stat().st_size != len(payload) or sha256_file(destination) != digest:
            raise ValueError("content-addressed object bytes differ")
    else:
        publish_bytes_without_overwrite(destination, payload)
    return ArtifactBindingV2(
        role=role,
        logical_path=logical_path or candidate.as_posix(),
        object_path=_safe_relative(destination, artifact_root),
        sha256=digest,
        byte_count=len(payload),
    )


def publish_object_bytes(
    payload: bytes,
    *,
    artifact_root: Path,
    role: str,
    logical_path: str,
) -> ArtifactBindingV2:
    if type(payload) is not bytes or not payload:
        raise ValueError("content-addressed object bytes must be nonempty")
    digest = hashlib.sha256(payload).hexdigest()
    destination = artifact_root / "objects" / "sha256" / digest[:2] / digest
    if destination.exists():
        if destination.is_symlink() or sha256_file(destination) != digest:
            raise ValueError("content-addressed object bytes differ")
    else:
        publish_bytes_without_overwrite(destination, payload)
    return ArtifactBindingV2(
        role=role,
        logical_path=logical_path,
        object_path=_safe_relative(destination, artifact_root),
        sha256=digest,
        byte_count=len(payload),
    )


def _binding_by_role(bindings: list[ArtifactBindingV2], role: str) -> ArtifactBindingV2:
    matches = [binding for binding in bindings if binding.role == role]
    if len(matches) != 1:
        raise ValueError(f"required artifact role {role!r} is missing or duplicated")
    return matches[0]


def publish_clip_bundle(
    clip: CollectedClip,
    *,
    artifact_root: Path,
    artifact_sources: Mapping[str, Path],
) -> PublishedClipBundle:
    """Publish one immutable payload, bundle core, and v2 reference identity."""

    required_roles = {
        "source_policy_bytes",
        "strict_actor_npz",
        "import_receipt",
        "equivalence_receipt",
        "mujoco_model_bytes",
        "dependency_lock",
        "collector_source",
        "certifier_source",
        "metric_source",
    }
    if not required_roles.issubset(artifact_sources):
        raise ValueError("artifact source map omits a required replay role")
    root = Path(artifact_root)
    clip_directory = root / "clips" / clip.clip_id
    payload = encode_clip_payload(
        clip.arrays,
        steps=clip.steps,
        plain_comparison=True,
    )
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    payload_path = clip_directory / f"payload-{payload_sha256}.npz"
    published_payload = publish_bytes_without_overwrite(payload_path, payload)
    if published_payload.sha256 != payload_sha256:
        raise ValueError("published clip payload hash differs")
    bound = [
        publish_object(
            path,
            artifact_root=root,
            role=role,
            logical_path=path.as_posix(),
        )
        for role, path in sorted(artifact_sources.items())
    ]
    rng_binding = publish_object_bytes(
        canonical_json_bytes(clip.rng_state),
        artifact_root=root,
        role="environment_rng_state",
        logical_path=f"{clip.clip_id}/initial_rng_state.json",
    )
    bound.append(rng_binding)
    actor_npz = _binding_by_role(bound, "strict_actor_npz")
    source_policy = _binding_by_role(bound, "source_policy_bytes")
    import_receipt = _binding_by_role(bound, "import_receipt")
    equivalence_receipt = _binding_by_role(bound, "equivalence_receipt")
    if actor_npz.sha256 != clip.actor_identity["npz_sha256"]:
        raise ValueError("bound actor NPZ differs from collected actor identity")
    if source_policy.sha256 != clip.actor_identity["source_policy_sha256"]:
        raise ValueError("bound source policy differs from actor identity")
    if import_receipt.sha256 != clip.actor_identity["import_receipt_sha256"]:
        raise ValueError("bound import receipt differs from actor identity")
    if equivalence_receipt.sha256 != clip.actor_identity["equivalence_receipt_sha256"]:
        raise ValueError("bound equivalence receipt differs from actor identity")
    runtime = clip.runtime_identity
    model_binding = _binding_by_role(bound, "mujoco_model_bytes")
    lock_binding = _binding_by_role(bound, "dependency_lock")
    if (
        model_binding.sha256 != runtime["model_sha256"]
        or model_binding.byte_count != runtime["model_byte_count"]
        or lock_binding.sha256 != runtime["uv_lock_sha256"]
    ):
        raise ValueError("bound model or dependency lock differs from runtime")
    bindings = array_bindings(clip.arrays)
    payload_binding = ArtifactBindingV2(
        role="clip_numeric_payload",
        logical_path=f"{clip.clip_id}/numeric_payload.npz",
        object_path=_safe_relative(payload_path, root),
        sha256=payload_sha256,
        byte_count=len(payload),
    )
    source_hashes = {
        role: _binding_by_role(bound, role).sha256
        for role in ("collector_source", "certifier_source", "metric_source")
    }
    expected_comparison = plain_comparison_receipt(clip.arrays, steps=clip.steps)
    if canonical_json_bytes(clip.plain_comparison) != canonical_json_bytes(expected_comparison):
        raise ValueError("plain comparison receipt differs from clip arrays")
    core: dict[str, object] = {
        "artifact_type": "same_runtime_full_integration_replay",
        "clip_id": clip.clip_id,
        "clip_kind": clip.clip_kind,
        "actor_variant": clip.actor_variant,
        "seed": clip.seed,
        "reset_order": clip.reset_order,
        "no_retry": True,
        "steps": clip.steps,
        "boundaries": clip.steps + 1,
        "collector_pid": clip.collector_pid,
        "runtime_identity": runtime,
        "runtime_identity_sha256": sha256_json(runtime),
        "source_actor": {
            **clip.actor_identity,
            "inference_id": INFERENCE_ID,
        },
        "bound_artifacts": [
            binding.to_dict() for binding in sorted(bound, key=lambda item: item.role)
        ],
        "payload": payload_binding.to_dict(),
        "array_bindings": bindings,
        "reference_contract": reference_schema_payload(),
        "wrapper_state_contract": {
            "id": WRAPPER_STATE_ID,
            "elapsed_counter": "TimeLimit._elapsed_steps",
            "flags_in_order": list(WRAPPER_FLAG_NAMES),
            "rng_state_bound_at_every_boundary": True,
            "unknown_fields_permitted": False,
        },
        "contact_contract": {
            "id": CONTACT_SEQUENCE_ID,
            "substeps": 5,
            "force_frame": "MuJoCo_contact_frame",
            "force_components": [
                "normal",
                "tangent1",
                "tangent2",
                "torque1",
                "torque2",
                "torsion",
            ],
            "ordered_by": ["transition", "physics_substep", "contact_index_within_substep"],
        },
        "policy_visible_input": {
            "id": POLICY_INPUT_ID,
            "fields": ["raw_observation_f32"],
            "shape": [348],
            "branch_metadata_permitted": False,
        },
        "controller_state": {
            "actor_recurrent_state": None,
            "normalizer_state": None,
            "residual_controller_state": None,
        },
        "rng_state": {
            "binding": rng_binding.to_dict(),
            "boundary_hash_array": "boundary_rng_state_sha256",
        },
        "plain_comparison": clip.plain_comparison,
        "source_hashes": source_hashes,
        "same_host_determinism_scope": "same_host_same_process_settings_bitwise/v1",
    }
    core_sha256 = sha256_json(core)
    reference_identity = ReferenceIdentityV2(
        robot_id="gymnasium/Humanoid-v5",
        reference_schema_id="humanoid_reference_45d_root_xy_sidecar/v2",
        reference_schema_sha256=sha256_json(reference_schema_payload()),
        reference_content_sha256=array_sha256(clip.arrays["reference_rows"]),
        root_xy_sidecar_sha256=array_sha256(clip.arrays["boundary_root_xy"]),
        replay_bundle_core_sha256=core_sha256,
        actor_npz_sha256=actor_npz.sha256,
        actor_state_sha256=str(clip.actor_identity["actor_state_sha256"]),
        import_receipt_sha256=import_receipt.sha256,
        equivalence_receipt_sha256=equivalence_receipt.sha256,
        generator_source_sha256=source_hashes["collector_source"],
        certifier_source_sha256=source_hashes["certifier_source"],
        runtime_identity_sha256=sha256_json(runtime),
        seed=clip.seed,
        n_boundaries=clip.steps + 1,
        cadence_seconds=0.015,
        derivation_kind="original",
        parent_reference_identity_sha256=None,
        parent_provenance={
            "actor_variant": clip.actor_variant,
            "source_policy_sha256": source_policy.sha256,
            "import_receipt_sha256": import_receipt.sha256,
            "equivalence_receipt_sha256": equivalence_receipt.sha256,
        },
    )
    manifest = {
        "bundle_id": "humanoid_full_clip_replay_bundle/v2",
        "schema_version": 2,
        "core": core,
        "core_sha256": core_sha256,
        "reference_identity": reference_identity.to_dict(),
        "reference_identity_sha256": reference_identity.sha256,
    }
    validate_bundle_manifest(manifest)
    encoded = canonical_json_bytes(manifest)
    manifest_sha256 = hashlib.sha256(encoded).hexdigest()
    manifest_path = clip_directory / f"bundle-{manifest_sha256}.json"
    published_manifest = publish_bytes_without_overwrite(manifest_path, encoded)
    if published_manifest.sha256 != manifest_sha256:
        raise ValueError("published bundle manifest hash differs")
    return PublishedClipBundle(
        clip_id=clip.clip_id,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        payload_path=payload_path,
        payload_sha256=payload_sha256,
        reference_identity_sha256=reference_identity.sha256,
    )


def load_bundle_manifest(path: Path, *, expected_sha256: str) -> dict[str, object]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("bundle manifest is missing or not a regular file")
    encoded = candidate.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise ValueError("bundle manifest hash differs")
    import json

    try:
        value = json.loads(encoded.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError("bundle manifest is invalid JSON") from exc
    if canonical_json_bytes(value) != encoded:
        raise ValueError("bundle manifest bytes are not canonical")
    return validate_bundle_manifest(value)


def verify_bound_artifacts(manifest: Mapping[str, object], *, artifact_root: Path) -> None:
    """Require every blob named by the bundle before any replay is attempted."""

    validated = validate_bundle_manifest(manifest)
    core = validated["core"]
    bindings = [*core["bound_artifacts"], core["payload"]]
    seen_paths: set[str] = set()
    for raw in bindings:
        if type(raw) is not dict:
            raise ValueError("bundle artifact binding is malformed")
        binding = ArtifactBindingV2(
            role=raw["role"],
            logical_path=raw["logical_path"],
            object_path=raw["object_path"],
            sha256=raw["sha256"],
            byte_count=raw["byte_count"],
        )
        if binding.object_path in seen_paths:
            # Identical content may serve multiple roles, but one path is hashed
            # once here and roles remain distinct in the manifest.
            continue
        seen_paths.add(binding.object_path)
        path = artifact_root / binding.object_path
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"MISSING_ARTIFACT: {binding.role}")
        if path.stat().st_size != binding.byte_count or sha256_file(path) != binding.sha256:
            raise ValueError(f"artifact hash differs: {binding.role}")


__all__ = [
    "BUNDLE_PUBLISHER_ID",
    "PublishedClipBundle",
    "load_bundle_manifest",
    "publish_clip_bundle",
    "publish_object",
    "verify_bound_artifacts",
]
