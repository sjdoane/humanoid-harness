"""Compound identity contracts for same-runtime Humanoid references."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np

REFERENCE_IDENTITY_ID = "humanoid_same_runtime_reference_identity/v2"
REPLAY_BUNDLE_ID = "humanoid_full_clip_replay_bundle/v1"
FULL_CLIP_CERTIFICATE_ID = "humanoid_full_clip_tier_d_certificate/v1"
CORPUS_MANIFEST_ID = "humanoid_reference_corpus_36x3/v1"
E3_MANIFEST_ID = "humanoid_same_state_fork_e3/v1"
ROBOT_ID = "gymnasium/Humanoid-v5"
REFERENCE_SCHEMA_ID = "humanoid_reference_45d_root_xy_sidecar/v2"
REFERENCE_WIDTH = 45
REFERENCE_HORIZON = 8
CONTROL_PERIOD_SECONDS = 0.015
CORPUS_ACTORS = ("expert", "medium", "simple")
TRAINING_REFERENCE_SEEDS = tuple(range(120001, 120013))
E4_SCREEN_REFERENCE_SEEDS = tuple(range(120101, 120121))
SEALED_E5_REFERENCE_SEEDS = tuple(range(120201, 120205))
CORPUS_SEEDS = (
    *TRAINING_REFERENCE_SEEDS,
    *E4_SCREEN_REFERENCE_SEEDS,
    *SEALED_E5_REFERENCE_SEEDS,
)
DEVELOPMENT_SCREEN_SEEDS = tuple(range(96001, 96021))
E3_THRESHOLDS = MappingProxyType(
    {
        "first_action_max_abs_physical": 0.02,
        "first_eight_actions_rms_physical": 0.01,
        "future_root_z_max_abs_m": 0.01,
        "future_quaternion_geodesic_max_rad": 0.02,
        "future_root_linear_velocity_rmse_m_s": 0.05,
        "future_root_angular_velocity_rmse_rad_s": 0.10,
        "future_joint_position_rmse_rad": 0.02,
        "future_joint_velocity_rmse_rad_s": 0.10,
    }
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ReferenceIdentityV2Error(ValueError):
    """A compound identity, manifest, or certificate is invalid."""


def canonical_json_bytes(value: object) -> bytes:
    """Encode one value with the repository's canonical JSON rules."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ReferenceIdentityV2Error(f"value is not canonical JSON: {exc}") from exc


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ReferenceIdentityV2Error(f"artifact is not a regular file: {candidate}")
    digest = hashlib.sha256()
    with candidate.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Hash exact array shape, dtype, and C-order bytes."""

    if not isinstance(value, np.ndarray):
        raise ReferenceIdentityV2Error("array hash input must be a NumPy array")
    contiguous = np.ascontiguousarray(value)
    metadata = canonical_json_bytes(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)}
    )
    raw = contiguous.tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def require_sha256(value: object, *, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ReferenceIdentityV2Error(f"{field} must be a lowercase SHA-256")
    return value


def require_exact_keys(value: Mapping[str, object], expected: set[str], *, field: str) -> None:
    if type(value) is not dict or set(value) != expected:
        raise ReferenceIdentityV2Error(f"{field} keys differ")


@dataclass(frozen=True, slots=True)
class ArtifactBindingV2:
    """One exact local blob used by a replay bundle."""

    role: str
    logical_path: str
    object_path: str
    sha256: str
    byte_count: int

    def __post_init__(self) -> None:
        for field in ("role", "logical_path", "object_path"):
            value = getattr(self, field)
            if type(value) is not str or not value or len(value) > 512:
                raise ReferenceIdentityV2Error(f"{field} must be nonempty bounded text")
        if Path(self.object_path).is_absolute() or ".." in Path(self.object_path).parts:
            raise ReferenceIdentityV2Error("object_path must be safe and relative")
        require_sha256(self.sha256, field=f"{self.role} SHA-256")
        if type(self.byte_count) is not int or not 0 < self.byte_count <= 128 * 1024 * 1024:
            raise ReferenceIdentityV2Error("artifact byte_count is invalid")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ArrayBindingV2:
    """One exact serialized numeric or fixed-width byte array."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name or len(self.name) > 128:
            raise ReferenceIdentityV2Error("array name is invalid")
        try:
            dtype = np.dtype(self.dtype)
        except TypeError as exc:
            raise ReferenceIdentityV2Error("array dtype is invalid") from exc
        if dtype.hasobject:
            raise ReferenceIdentityV2Error("object arrays are forbidden")
        if (
            type(self.shape) is not tuple
            or not self.shape
            or any(type(size) is not int or size < 0 for size in self.shape)
        ):
            raise ReferenceIdentityV2Error("array shape is invalid")
        require_sha256(self.sha256, field=f"array {self.name} SHA-256")

    def to_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["shape"] = list(self.shape)
        return value


@dataclass(frozen=True, slots=True)
class ReferenceIdentityV2:
    """Identity of a reference plus every artifact required to replay it."""

    robot_id: str
    reference_schema_id: str
    reference_schema_sha256: str
    reference_content_sha256: str
    root_xy_sidecar_sha256: str
    replay_bundle_core_sha256: str
    actor_npz_sha256: str
    actor_state_sha256: str
    import_receipt_sha256: str
    equivalence_receipt_sha256: str
    generator_source_sha256: str
    certifier_source_sha256: str
    runtime_identity_sha256: str
    seed: int
    n_boundaries: int
    cadence_seconds: float
    derivation_kind: str
    parent_reference_identity_sha256: str | None
    parent_provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.robot_id != ROBOT_ID:
            raise ReferenceIdentityV2Error("robot_id differs from Humanoid-v5")
        if self.reference_schema_id != REFERENCE_SCHEMA_ID:
            raise ReferenceIdentityV2Error("reference schema id differs")
        for field in (
            "reference_schema_sha256",
            "reference_content_sha256",
            "root_xy_sidecar_sha256",
            "replay_bundle_core_sha256",
            "actor_npz_sha256",
            "actor_state_sha256",
            "import_receipt_sha256",
            "equivalence_receipt_sha256",
            "generator_source_sha256",
            "certifier_source_sha256",
            "runtime_identity_sha256",
        ):
            require_sha256(getattr(self, field), field=field)
        if type(self.seed) is not int or isinstance(self.seed, bool) or self.seed < 0:
            raise ReferenceIdentityV2Error("seed must be a non-negative exact integer")
        if type(self.n_boundaries) is not int or self.n_boundaries < 2:
            raise ReferenceIdentityV2Error("n_boundaries must be at least two")
        if (
            type(self.cadence_seconds) is not float
            or not math.isfinite(self.cadence_seconds)
            or self.cadence_seconds <= 0.0
        ):
            raise ReferenceIdentityV2Error("cadence_seconds must be a positive exact float")
        if self.derivation_kind not in {"original", "crop", "retime"}:
            raise ReferenceIdentityV2Error("derivation_kind is unsupported")
        if self.derivation_kind == "original":
            if self.parent_reference_identity_sha256 is not None:
                raise ReferenceIdentityV2Error("original reference cannot name a parent identity")
            if self.cadence_seconds != CONTROL_PERIOD_SECONDS:
                raise ReferenceIdentityV2Error("original reference cadence differs")
        else:
            require_sha256(
                self.parent_reference_identity_sha256,
                field="parent reference identity SHA-256",
            )
        if type(self.parent_provenance) is not dict or not self.parent_provenance:
            raise ReferenceIdentityV2Error("parent_provenance must be a nonempty exact mapping")
        canonical_json_bytes(self.parent_provenance)

    def to_dict(self) -> dict[str, object]:
        return {
            "identity_id": REFERENCE_IDENTITY_ID,
            "schema_version": 2,
            **asdict(self),
        }

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())


def validate_reference_derivation(
    parent: ReferenceIdentityV2,
    child: ReferenceIdentityV2,
    *,
    parent_certificate_sha256: str,
    child_certificate_sha256: str,
) -> None:
    """Reject a crop or retime that reuses its parent's identity or certificate."""

    if child.derivation_kind not in {"crop", "retime"}:
        raise ReferenceIdentityV2Error("derived reference must declare crop or retime")
    if child.parent_reference_identity_sha256 != parent.sha256:
        raise ReferenceIdentityV2Error("derived reference parent identity differs")
    require_sha256(parent_certificate_sha256, field="parent certificate SHA-256")
    require_sha256(child_certificate_sha256, field="child certificate SHA-256")
    if child.sha256 == parent.sha256:
        raise ReferenceIdentityV2Error("derived reference retained its parent identity")
    if child_certificate_sha256 == parent_certificate_sha256:
        raise ReferenceIdentityV2Error("derived reference retained its parent certificate")
    if child.derivation_kind == "crop":
        if child.n_boundaries >= parent.n_boundaries:
            raise ReferenceIdentityV2Error("crop must contain fewer boundaries than its parent")
        if child.cadence_seconds != parent.cadence_seconds:
            raise ReferenceIdentityV2Error("crop must retain its parent cadence")
        if child.reference_content_sha256 == parent.reference_content_sha256:
            raise ReferenceIdentityV2Error("crop retained parent reference content")
    elif child.cadence_seconds == parent.cadence_seconds:
        raise ReferenceIdentityV2Error("retime must change cadence")


def transition_indices_sha256(steps: int) -> str:
    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ReferenceIdentityV2Error("steps must be in [1, 1000]")
    return sha256_json(list(range(steps)))


def validate_full_clip_certificate(value: Mapping[str, object]) -> dict[str, object]:
    """Validate an all-transitions certificate; fractions are not accepted."""

    expected_keys = {
        "certificate_id",
        "schema_version",
        "clip_id",
        "bundle_manifest_sha256",
        "reference_identity_sha256",
        "payload_sha256",
        "collector_pid",
        "certifier_pid",
        "steps_expected",
        "transitions_verified",
        "covered_transition_indices_sha256",
        "verification_digest_sha256",
        "all_hashes_verified",
        "all_transitions_passed",
        "first_failure",
        "evidence_class",
        "claim_ceiling",
    }
    require_exact_keys(value, expected_keys, field="full-clip certificate")
    if value["certificate_id"] != FULL_CLIP_CERTIFICATE_ID or value["schema_version"] != 1:
        raise ReferenceIdentityV2Error("full-clip certificate identity differs")
    steps = value["steps_expected"]
    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ReferenceIdentityV2Error("certificate steps_expected is invalid")
    if value["transitions_verified"] != steps:
        raise ReferenceIdentityV2Error("certificate covers fewer than all transitions")
    if value["covered_transition_indices_sha256"] != transition_indices_sha256(steps):
        raise ReferenceIdentityV2Error("certificate transition coverage differs")
    for field in (
        "bundle_manifest_sha256",
        "reference_identity_sha256",
        "payload_sha256",
        "verification_digest_sha256",
    ):
        require_sha256(value[field], field=field)
    for field in ("collector_pid", "certifier_pid"):
        if type(value[field]) is not int or value[field] <= 0:
            raise ReferenceIdentityV2Error(f"{field} must be a positive process id")
    if value["collector_pid"] == value["certifier_pid"]:
        raise ReferenceIdentityV2Error("certifier must run in a separate process")
    if value["all_hashes_verified"] is not True or value["all_transitions_passed"] is not True:
        raise ReferenceIdentityV2Error("certificate is not an all-transitions pass")
    if value["first_failure"] is not None:
        raise ReferenceIdentityV2Error("passing certificate cannot contain a failure")
    if value["evidence_class"] != "same_runtime_tier_d_replay":
        raise ReferenceIdentityV2Error("certificate evidence class differs")
    if value["claim_ceiling"] != "full_clip_same_host_replay_only":
        raise ReferenceIdentityV2Error("certificate claim ceiling differs")
    return dict(value)


def e3_manifest_payload() -> dict[str, object]:
    return {
        "manifest_id": E3_MANIFEST_ID,
        "schema_version": 1,
        "actors_in_order": list(CORPUS_ACTORS),
        "block_seeds_in_order": list(CORPUS_SEEDS),
        "reference_horizon": REFERENCE_HORIZON,
        "thresholds_strictly_greater_than": dict(E3_THRESHOLDS),
        "policy_visible_input": {
            "fields_in_order": [
                {"name": "raw_observation", "dtype": "float32", "shape": [348]},
                {
                    "name": "reference_window",
                    "dtype": "float32",
                    "shape": [REFERENCE_HORIZON, REFERENCE_WIDTH],
                },
            ],
            "branch_metadata_permitted": False,
            "root_xy_sidecar_permitted": False,
        },
        "block_pass_rule": "expert_vs_medium_and_expert_vs_simple_both_pass",
        "no_backfill": True,
        "self_reference_only_permitted": False,
    }


def validate_e3_manifest(value: Mapping[str, object]) -> dict[str, object]:
    expected = e3_manifest_payload()
    if type(value) is not dict or canonical_json_bytes(value) != canonical_json_bytes(expected):
        raise ReferenceIdentityV2Error("E3 manifest differs from the frozen design")
    return dict(value)


def corpus_manifest_payload(clips: list[Mapping[str, object]]) -> dict[str, object]:
    return {
        "manifest_id": CORPUS_MANIFEST_ID,
        "schema_version": 1,
        "actors_in_order": list(CORPUS_ACTORS),
        "training_reference_seeds": list(TRAINING_REFERENCE_SEEDS),
        "e4_screen_reference_seeds": list(E4_SCREEN_REFERENCE_SEEDS),
        "sealed_e5_reference_seeds": list(SEALED_E5_REFERENCE_SEEDS),
        "block_seeds_in_order": list(CORPUS_SEEDS),
        "clip_count": len(clips),
        "clips_in_reset_order": [dict(clip) for clip in clips],
        "replacement_seeds_permitted": False,
        "failed_attempts_retained": True,
    }


def validate_corpus_manifest(value: Mapping[str, object]) -> dict[str, object]:
    if type(value) is not dict:
        raise ReferenceIdentityV2Error("corpus manifest must be an exact mapping")
    expected_keys = set(corpus_manifest_payload([]))
    require_exact_keys(value, expected_keys, field="corpus manifest")
    if value["manifest_id"] != CORPUS_MANIFEST_ID or value["schema_version"] != 1:
        raise ReferenceIdentityV2Error("corpus manifest identity differs")
    if value["actors_in_order"] != list(CORPUS_ACTORS):
        raise ReferenceIdentityV2Error("corpus actor order differs")
    if value["training_reference_seeds"] != list(TRAINING_REFERENCE_SEEDS):
        raise ReferenceIdentityV2Error("training reference seeds differ")
    if value["e4_screen_reference_seeds"] != list(E4_SCREEN_REFERENCE_SEEDS):
        raise ReferenceIdentityV2Error("E4 reference seeds differ")
    if value["sealed_e5_reference_seeds"] != list(SEALED_E5_REFERENCE_SEEDS):
        raise ReferenceIdentityV2Error("sealed E5 reference seeds differ")
    if value["block_seeds_in_order"] != list(CORPUS_SEEDS):
        raise ReferenceIdentityV2Error("corpus block seeds differ")
    clips = value["clips_in_reset_order"]
    if type(clips) is not list or len(clips) != 108 or value["clip_count"] != 108:
        raise ReferenceIdentityV2Error("corpus must contain exactly 108 clips")
    expected_pairs = [(seed, actor) for seed in CORPUS_SEEDS for actor in CORPUS_ACTORS]
    observed_pairs: list[tuple[object, object]] = []
    seen_ids: set[object] = set()
    for reset_order, clip in enumerate(clips):
        if type(clip) is not dict or set(clip) != {
            "clip_id",
            "seed",
            "actor_variant",
            "reset_order",
            "bundle_manifest_sha256",
            "reference_identity_sha256",
        }:
            raise ReferenceIdentityV2Error("corpus clip entry keys differ")
        if (
            type(clip["reset_order"]) is not int
            or clip["reset_order"] != reset_order
            or type(clip["seed"]) is not int
            or type(clip["actor_variant"]) is not str
            or type(clip["clip_id"]) is not str
            or not clip["clip_id"]
        ):
            raise ReferenceIdentityV2Error("corpus reset order is not contiguous")
        if clip["clip_id"] in seen_ids:
            raise ReferenceIdentityV2Error("corpus contains a duplicate clip id")
        seen_ids.add(clip["clip_id"])
        observed_pairs.append((clip["seed"], clip["actor_variant"]))
        require_sha256(clip["bundle_manifest_sha256"], field="bundle manifest SHA-256")
        require_sha256(clip["reference_identity_sha256"], field="reference identity SHA-256")
    if observed_pairs != expected_pairs:
        raise ReferenceIdentityV2Error("corpus seeds or actor order differ")
    if value["replacement_seeds_permitted"] is not False:
        raise ReferenceIdentityV2Error("corpus permits replacement seeds")
    if value["failed_attempts_retained"] is not True:
        raise ReferenceIdentityV2Error("corpus does not retain failed attempts")
    return dict(value)
    return dict(value)


__all__ = [
    "CONTROL_PERIOD_SECONDS",
    "CORPUS_ACTORS",
    "CORPUS_SEEDS",
    "DEVELOPMENT_SCREEN_SEEDS",
    "E3_THRESHOLDS",
    "REFERENCE_HORIZON",
    "REFERENCE_IDENTITY_ID",
    "REFERENCE_SCHEMA_ID",
    "REFERENCE_WIDTH",
    "ArrayBindingV2",
    "ArtifactBindingV2",
    "ReferenceIdentityV2",
    "ReferenceIdentityV2Error",
    "array_sha256",
    "canonical_json_bytes",
    "corpus_manifest_payload",
    "e3_manifest_payload",
    "require_sha256",
    "sha256_file",
    "sha256_json",
    "transition_indices_sha256",
    "validate_corpus_manifest",
    "validate_e3_manifest",
    "validate_full_clip_certificate",
    "validate_reference_derivation",
]
