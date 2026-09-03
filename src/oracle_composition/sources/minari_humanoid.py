"""Strict Tier-K projection of pinned Minari Humanoid-v5 episodes."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, BinaryIO

import numpy as np

from oracle_composition.contracts import ReferenceArtifact
from oracle_composition.tracking.humanoid_reference import (
    HUMANOID_REFERENCE_SCHEMA,
    validate_humanoid_reference,
)

PROJECTION_RECEIPT_SCHEMA_VERSION = 2
MINARI_HUMANOID_DATASET_ID = "mujoco/humanoid/expert-v0"
MINARI_HUMANOID_SOURCE_REPOSITORY = "https://huggingface.co/datasets/farama-minari/mujoco"
MINARI_HUMANOID_GENERATION_REPOSITORY = (
    "https://github.com/Farama-Foundation/minari-dataset-generation-scripts"
)
MINARI_HUMANOID_ENVIRONMENT_ID = "Humanoid-v5"
MINARI_HUMANOID_ENTRY_POINT = "gymnasium.envs.mujoco.humanoid_v5:HumanoidEnv"
MINARI_HUMANOID_PROJECTION_ID = "gymnasium/Humanoid-v5/observation-348-to-reference-45/v1"
MINARI_HUMANOID_OBSERVATION_WIDTH = 348
MINARI_HUMANOID_ACTION_WIDTH = 17
MAX_HDF5_BYTES = 4 * 1024 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024
MAX_EPISODES = 100_000
MAX_TOTAL_STEPS = 100_000_000
MAX_EPISODE_STEPS = 1_000
REGISTERED_MINARI_HUMANOID_SOURCE_MANIFEST_SHA256 = (
    "974700591304a4d3be576d55bd294558ae0cd10cbfce5bef7bd73ca21632f899"
)
REGISTERED_MINARI_HUMANOID_SOURCE_MANIFEST_BYTES = 1_458

_REGISTERED_MINARI_HUMANOID_SOURCE_MANIFEST = (
    Path(__file__).parent / "manifests" / "minari_humanoid_expert_v0.json"
)
_REGISTERED_MINARI_HUMANOID_SOURCE_RECORD = {
    "schema_version": 1,
    "provenance_class": "registered_official_repository_revision",
    "dataset_id": MINARI_HUMANOID_DATASET_ID,
    "source_repository": MINARI_HUMANOID_SOURCE_REPOSITORY,
    "source_commit": "8e62dc7f7fcb4a19f8f869c65402d4bb60049117",
    "registered_date": "2026-09-03",
    "license_status": "no_dataset_license_declared_in_source_repository_at_registration",
    "metadata_declared_generation_code_url": MINARI_HUMANOID_GENERATION_REPOSITORY,
    "files": {
        "hdf5": {
            "repository_path": "humanoid/expert-v0/data/main_data.hdf5",
            "source_url": (
                "https://huggingface.co/datasets/farama-minari/mujoco/resolve/"
                "8e62dc7f7fcb4a19f8f869c65402d4bb60049117/"
                "humanoid/expert-v0/data/main_data.hdf5"
            ),
            "bytes": 2_946_805_796,
            "sha256": "8253be693f06aeeac3cb62eeb349ad02d4ca0bcad02685b4b8ed390798d9aa1e",
        },
        "metadata": {
            "repository_path": "humanoid/expert-v0/data/metadata.json",
            "source_url": (
                "https://huggingface.co/datasets/farama-minari/mujoco/resolve/"
                "8e62dc7f7fcb4a19f8f869c65402d4bb60049117/"
                "humanoid/expert-v0/data/metadata.json"
            ),
            "bytes": 9_305,
            "sha256": "2eb6e0ba388ceabef5eec1dea7e401e62391d856cf42b394c262db7c21366024",
        },
    },
    "claim_boundary": (
        "Exact source registration and data-only projection only. The dataset license is "
        "unresolved, root x/y evidence is absent, and no dynamics-feasibility or training "
        "admission follows."
    ),
}

# Humanoid-v5 stores qpos[2:] at observation[0:22] and qvel at
# observation[22:45]. The two swaps put abdomen_y before abdomen_z, matching
# the frozen actuator order used by the 45D controller reference ABI.
HUMANOID_V5_OBSERVATION_TO_REFERENCE_INDICES = (
    0,
    1,
    2,
    3,
    4,
    22,
    23,
    24,
    25,
    26,
    27,
    6,
    5,
    *range(7, 22),
    29,
    28,
    *range(30, 45),
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_METADATA_FIELDS = frozenset(
    {
        "total_episodes",
        "total_steps",
        "data_format",
        "observation_space",
        "action_space",
        "env_spec",
        "dataset_size",
        "dataset_id",
        "code_permalink",
        "author",
        "author_email",
        "algorithm_name",
        "description",
        "minari_version",
        "requirements",
    }
)
_SPACE_FIELDS = frozenset({"type", "dtype", "shape", "low", "high"})
_ENV_SPEC_FIELDS = frozenset(
    {
        "id",
        "entry_point",
        "reward_threshold",
        "nondeterministic",
        "max_episode_steps",
        "order_enforce",
        "disable_env_checker",
        "kwargs",
        "additional_wrappers",
        "vector_entry_point",
    }
)
_EPISODE_FIELDS = frozenset(
    {"actions", "infos", "observations", "rewards", "terminations", "truncations"}
)
_EPISODE_ATTRIBUTE_FIELDS = frozenset(
    {
        "id",
        "seed",
        "total_steps",
        "rewards_max",
        "rewards_mean",
        "rewards_min",
        "rewards_std",
        "rewards_sum",
    }
)


class MinariHumanoidImportError(ValueError):
    """A source file or selected episode violates the pinned projection contract."""


@dataclass(frozen=True, slots=True)
class MinariHumanoidProjectionReceipt:
    """Immutable provenance for one non-admitted 45D projection."""

    source_commit: str
    source_provenance_class: str
    source_origin_verified: bool
    source_record_sha256: str | None
    source_record_bytes: int | None
    hdf5_sha256: str
    hdf5_bytes: int
    metadata_sha256: str
    metadata_bytes: int
    minari_version: str
    declared_requirements: tuple[str, ...]
    episode_id: int
    episode_seed: int
    episode_total_steps: int
    observations_sha256: str
    actions_sha256: str
    rewards_sha256: str
    terminations_sha256: str
    truncations_sha256: str
    ignored_observation_fields_sha256: str
    projection_mapping_sha256: str
    reference_artifact_id: str
    reference_content_sha256: str
    reference_schema_sha256: str
    reference_frames: int

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": PROJECTION_RECEIPT_SCHEMA_VERSION,
            "evidence_tier": "Tier-K",
            "admission_status": "not_admitted",
            "claim_ceiling": "source_projection_and_numerical_candidate_only",
            "dynamics_feasibility_established": False,
            "source_provenance_class": self.source_provenance_class,
            "source_origin_verified": self.source_origin_verified,
            "asserted_source_repository": MINARI_HUMANOID_SOURCE_REPOSITORY,
            "asserted_source_commit": self.source_commit,
            "source_record_sha256": self.source_record_sha256,
            "source_record_bytes": self.source_record_bytes,
            "dataset_id": MINARI_HUMANOID_DATASET_ID,
            "metadata_declared_generation_code_url": MINARI_HUMANOID_GENERATION_REPOSITORY,
            "hdf5_sha256": self.hdf5_sha256,
            "hdf5_bytes": self.hdf5_bytes,
            "metadata_sha256": self.metadata_sha256,
            "metadata_bytes": self.metadata_bytes,
            "environment_id": MINARI_HUMANOID_ENVIRONMENT_ID,
            "environment_entry_point": MINARI_HUMANOID_ENTRY_POINT,
            "minari_version": self.minari_version,
            "declared_requirements": list(self.declared_requirements),
            "episode_id": self.episode_id,
            "episode_seed": self.episode_seed,
            "episode_total_steps": self.episode_total_steps,
            "episode_observation_shape": [self.episode_total_steps + 1, 348],
            "episode_action_shape": [self.episode_total_steps, 17],
            "observations_sha256": self.observations_sha256,
            "actions_sha256": self.actions_sha256,
            "rewards_sha256": self.rewards_sha256,
            "terminations_sha256": self.terminations_sha256,
            "truncations_sha256": self.truncations_sha256,
            "ignored_observation_fields_sha256": self.ignored_observation_fields_sha256,
            "root_position_xy_evidence": "absent_from_observation_and_empty_infos",
            "root_position_xy_reconstructed": False,
            "projection_id": MINARI_HUMANOID_PROJECTION_ID,
            "projection_indices": list(HUMANOID_V5_OBSERVATION_TO_REFERENCE_INDICES),
            "projection_mapping_sha256": self.projection_mapping_sha256,
            "reference_artifact_id": self.reference_artifact_id,
            "reference_content_sha256": self.reference_content_sha256,
            "reference_schema_sha256": self.reference_schema_sha256,
            "reference_shape": [self.reference_frames, HUMANOID_REFERENCE_SCHEMA.width],
            "reference_cadence_hz": HUMANOID_REFERENCE_SCHEMA.cadence_hz,
            "tier_d_certificate_sha256": None,
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()


@dataclass(frozen=True, slots=True)
class MinariHumanoidProjection:
    """Validated episode observations, projected reference, and Tier-K receipt."""

    episode_observations: np.ndarray
    reference: ReferenceArtifact
    receipt: MinariHumanoidProjectionReceipt

    def verify(self) -> None:
        """Reconcile the retained observation bytes, projection, and receipt."""

        if not isinstance(self.receipt, MinariHumanoidProjectionReceipt):
            raise MinariHumanoidImportError("projection receipt has the wrong type")
        if (
            not isinstance(self.receipt.episode_id, int)
            or isinstance(self.receipt.episode_id, bool)
            or not 0 <= self.receipt.episode_id < MAX_EPISODES
            or not isinstance(self.receipt.episode_total_steps, int)
            or isinstance(self.receipt.episode_total_steps, bool)
            or not 1 <= self.receipt.episode_total_steps <= MAX_EPISODE_STEPS
        ):
            raise MinariHumanoidImportError("projection receipt has invalid episode dimensions")
        observations = self.episode_observations
        expected_shape = (
            self.receipt.episode_total_steps + 1,
            MINARI_HUMANOID_OBSERVATION_WIDTH,
        )
        if (
            not isinstance(observations, np.ndarray)
            or observations.dtype.str != np.dtype("<f8").str
            or observations.shape != expected_shape
            or not observations.flags.c_contiguous
            or observations.flags.writeable
            or not np.isfinite(observations).all()
        ):
            raise MinariHumanoidImportError(
                "retained episode observations violate the immutable projection contract"
            )
        if _array_sha256(observations) != self.receipt.observations_sha256:
            raise MinariHumanoidImportError("retained episode observation hash mismatch")
        ignored = np.ascontiguousarray(observations[:, 45:])
        if _array_sha256(ignored) != self.receipt.ignored_observation_fields_sha256:
            raise MinariHumanoidImportError("ignored observation-field hash mismatch")

        self.reference.verify()
        identity = self.reference.identity
        expected_artifact_id = (
            f"minari/{MINARI_HUMANOID_DATASET_ID}/episode-"
            f"{self.receipt.episode_id}/projected-45d/v1"
        )
        expected_reference = observations[:, HUMANOID_V5_OBSERVATION_TO_REFERENCE_INDICES]
        observed_reference = np.asarray(self.reference.values, dtype=np.float64)
        if not np.array_equal(observed_reference, expected_reference):
            raise MinariHumanoidImportError(
                "reference values do not match the retained observation projection"
            )
        if (
            identity.artifact_id != expected_artifact_id
            or identity.artifact_id != self.receipt.reference_artifact_id
            or identity.content_sha256 != self.receipt.reference_content_sha256
            or identity.schema_sha256 != HUMANOID_REFERENCE_SCHEMA.sha256
            or identity.schema_sha256 != self.receipt.reference_schema_sha256
            or identity.n_frames != observations.shape[0]
            or identity.n_frames != self.receipt.reference_frames
            or self.receipt.projection_mapping_sha256 != _projection_mapping_sha256()
        ):
            raise MinariHumanoidImportError(
                "projection reference identity differs from its receipt"
            )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _exact_keys(value: Mapping[str, object], expected: frozenset[str], *, field: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise MinariHumanoidImportError(
            f"{field} fields mismatch: missing={missing!r}, extra={extra!r}"
        )


def _without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise MinariHumanoidImportError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _metadata_constant(value: str) -> float:
    if value == "Infinity":
        return math.inf
    if value == "-Infinity":
        return -math.inf
    raise MinariHumanoidImportError(f"unsupported JSON constant: {value}")


def _decode_json_object(raw: bytes, *, field: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_without_duplicate_keys,
            parse_constant=_metadata_constant,
        )
    except MinariHumanoidImportError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise MinariHumanoidImportError(f"invalid {field} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise MinariHumanoidImportError(f"{field} must be a JSON object")
    return value


def _registered_source_record() -> tuple[dict[str, Any], str, int]:
    with _verified_binary_file(
        _REGISTERED_MINARI_HUMANOID_SOURCE_MANIFEST,
        expected_sha256=REGISTERED_MINARI_HUMANOID_SOURCE_MANIFEST_SHA256,
        expected_bytes=REGISTERED_MINARI_HUMANOID_SOURCE_MANIFEST_BYTES,
        max_bytes=MAX_METADATA_BYTES,
        field="registered Minari source manifest",
    ) as (stream, byte_count):
        raw = stream.read(MAX_METADATA_BYTES + 1)
    record = _decode_json_object(raw, field="registered Minari source manifest")
    if record != _REGISTERED_MINARI_HUMANOID_SOURCE_RECORD:
        raise MinariHumanoidImportError("registered Minari source manifest content mismatch")
    return record, REGISTERED_MINARI_HUMANOID_SOURCE_MANIFEST_SHA256, byte_count


def _decode_nested_json(value: object, *, field: str) -> dict[str, Any]:
    if not isinstance(value, str) or not value:
        raise MinariHumanoidImportError(f"{field} must be non-empty serialized JSON")
    return _decode_json_object(value.encode("utf-8"), field=field)


def _sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise MinariHumanoidImportError(f"{field} must be a lowercase SHA-256")
    return value


def _source_commit(value: object) -> str:
    if not isinstance(value, str) or not _GIT_COMMIT.fullmatch(value):
        raise MinariHumanoidImportError(
            "source_commit must be a pinned lowercase 40-character Git commit"
        )
    return value


def _exact_int(value: object, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise MinariHumanoidImportError(f"{field} must be an integer")
    resolved = int(value)
    if not minimum <= resolved <= maximum:
        raise MinariHumanoidImportError(f"{field} must be in [{minimum}, {maximum}]")
    return resolved


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MinariHumanoidImportError(f"{field} must be numeric")
    try:
        resolved = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise MinariHumanoidImportError(f"{field} must be finite") from exc
    if not math.isfinite(resolved):
        raise MinariHumanoidImportError(f"{field} must be finite")
    return resolved


def _bounded_text(value: object, *, field: str, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise MinariHumanoidImportError(f"{field} must be non-empty bounded text")
    if any(ord(character) < 32 for character in value):
        raise MinariHumanoidImportError(f"{field} contains a control character")
    return value


def _bounded_text_sequence(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or len(value) > 64:
        raise MinariHumanoidImportError(f"{field} must be a non-empty bounded JSON list")
    return tuple(_bounded_text(item, field=f"{field}[{index}]") for index, item in enumerate(value))


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _hash_exact_stream(
    stream: BinaryIO,
    *,
    byte_count: int,
    field: str,
    phase: str,
) -> str:
    """Hash exactly the admitted byte count and reject short or growing inputs."""

    digest = hashlib.sha256()
    remaining = byte_count
    while remaining:
        chunk = stream.read(min(1024 * 1024, remaining))
        if not chunk:
            raise MinariHumanoidImportError(f"{field} became shorter during {phase}")
        digest.update(chunk)
        remaining -= len(chunk)
    if stream.read(1):
        raise MinariHumanoidImportError(f"{field} grew during {phase}")
    return digest.hexdigest()


@contextmanager
def _verified_binary_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None = None,
    max_bytes: int,
    field: str,
) -> Any:
    resolved = Path(path)
    if resolved.is_symlink():
        raise MinariHumanoidImportError(f"{field} must not be a symbolic link")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise MinariHumanoidImportError(
            f"cannot open {field} as a regular file: {resolved}"
        ) from exc
    stream: BinaryIO = os.fdopen(descriptor, "rb")
    try:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise MinariHumanoidImportError(f"{field} must be a regular file")
        if not 0 < before.st_size <= max_bytes:
            raise MinariHumanoidImportError(
                f"{field} byte size {before.st_size} is outside [1, {max_bytes}]"
            )
        if expected_bytes is not None and before.st_size != expected_bytes:
            raise MinariHumanoidImportError(
                f"{field} byte size mismatch: expected {expected_bytes}, observed {before.st_size}"
            )
        observed_sha256 = _hash_exact_stream(
            stream,
            byte_count=before.st_size,
            field=field,
            phase="initial verification",
        )
        after_hash = os.fstat(stream.fileno())
        if _stat_identity(after_hash) != _stat_identity(before):
            raise MinariHumanoidImportError(f"{field} changed while it was hashed")
        if observed_sha256 != expected_sha256:
            raise MinariHumanoidImportError(
                f"{field} SHA-256 mismatch: expected {expected_sha256}, observed {observed_sha256}"
            )
        stream.seek(0)
        yield stream, before.st_size
        after_read = os.fstat(stream.fileno())
        if _stat_identity(after_read) != _stat_identity(before):
            raise MinariHumanoidImportError(f"{field} changed while it was read")
        stream.seek(0)
        verification_sha256 = _hash_exact_stream(
            stream,
            byte_count=before.st_size,
            field=field,
            phase="final verification",
        )
        after_verification = os.fstat(stream.fileno())
        if _stat_identity(after_verification) != _stat_identity(before):
            raise MinariHumanoidImportError(f"{field} changed during final verification")
        if verification_sha256 != expected_sha256:
            raise MinariHumanoidImportError(f"{field} bytes changed while it was read")
    finally:
        stream.close()


def _load_h5py() -> ModuleType:
    try:
        return importlib.import_module("h5py")
    except ImportError as exc:
        raise MinariHumanoidImportError(
            "HDF5 projection requires the optional 'sources' extra with h5py"
        ) from exc


def _validate_box_space(
    value: object,
    *,
    field: str,
    dtype: str,
    width: int,
    bounded_action: bool,
) -> None:
    space = _decode_nested_json(value, field=field)
    _exact_keys(space, _SPACE_FIELDS, field=field)
    if space["type"] != "Box" or space["dtype"] != dtype or space["shape"] != [width]:
        raise MinariHumanoidImportError(f"{field} type, dtype, or shape mismatch")
    try:
        low = np.asarray(space["low"], dtype=np.float64)
        high = np.asarray(space["high"], dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise MinariHumanoidImportError(f"{field} bounds must be numeric") from exc
    if low.shape != (width,) or high.shape != (width,):
        raise MinariHumanoidImportError(f"{field} bounds shape mismatch")
    if bounded_action:
        expected_low = float(np.float32(-0.4))
        expected_high = float(np.float32(0.4))
        if not np.all(low == expected_low) or not np.all(high == expected_high):
            raise MinariHumanoidImportError(f"{field} bounds mismatch")
    elif not np.isneginf(low).all() or not np.isposinf(high).all():
        raise MinariHumanoidImportError(f"{field} must use unbounded observation limits")


def _validate_metadata(raw: bytes, *, episode_id: int) -> dict[str, Any]:
    metadata = _decode_json_object(raw, field="metadata")
    _exact_keys(metadata, _METADATA_FIELDS, field="metadata")
    total_episodes = _exact_int(
        metadata["total_episodes"], field="metadata.total_episodes", minimum=1, maximum=MAX_EPISODES
    )
    _exact_int(
        metadata["total_steps"], field="metadata.total_steps", minimum=1, maximum=MAX_TOTAL_STEPS
    )
    if episode_id >= total_episodes:
        raise MinariHumanoidImportError("episode_id is outside metadata.total_episodes")
    if metadata["data_format"] != "hdf5":
        raise MinariHumanoidImportError("metadata.data_format must be exactly 'hdf5'")
    if metadata["dataset_id"] != MINARI_HUMANOID_DATASET_ID:
        raise MinariHumanoidImportError("metadata.dataset_id mismatch")
    _validate_box_space(
        metadata["observation_space"],
        field="metadata.observation_space",
        dtype="float64",
        width=MINARI_HUMANOID_OBSERVATION_WIDTH,
        bounded_action=False,
    )
    _validate_box_space(
        metadata["action_space"],
        field="metadata.action_space",
        dtype="float32",
        width=MINARI_HUMANOID_ACTION_WIDTH,
        bounded_action=True,
    )
    env_spec = _decode_nested_json(metadata["env_spec"], field="metadata.env_spec")
    _exact_keys(env_spec, _ENV_SPEC_FIELDS, field="metadata.env_spec")
    expected_env_spec = {
        "id": MINARI_HUMANOID_ENVIRONMENT_ID,
        "entry_point": MINARI_HUMANOID_ENTRY_POINT,
        "reward_threshold": None,
        "nondeterministic": False,
        "max_episode_steps": MAX_EPISODE_STEPS,
        "order_enforce": True,
        "disable_env_checker": False,
        "kwargs": {},
        "additional_wrappers": [],
        "vector_entry_point": None,
    }
    if env_spec != expected_env_spec:
        raise MinariHumanoidImportError("metadata.env_spec mismatch")
    size = _finite_number(metadata["dataset_size"], field="metadata.dataset_size")
    if size <= 0.0:
        raise MinariHumanoidImportError("metadata.dataset_size must be finite and positive")
    if metadata["code_permalink"] != MINARI_HUMANOID_GENERATION_REPOSITORY:
        raise MinariHumanoidImportError("metadata.code_permalink mismatch")
    _bounded_text_sequence(metadata["author"], field="metadata.author")
    _bounded_text_sequence(metadata["author_email"], field="metadata.author_email")
    _bounded_text(metadata["algorithm_name"], field="metadata.algorithm_name")
    _bounded_text(metadata["description"], field="metadata.description", maximum=32_768)
    _bounded_text(metadata["minari_version"], field="metadata.minari_version")
    metadata["requirements"] = _bounded_text_sequence(
        metadata["requirements"], field="metadata.requirements"
    )
    return metadata


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    header = _canonical_json({"dtype": array.dtype.str, "shape": list(array.shape)})
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, byteorder="big"))
    digest.update(header)
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _require_hard_child(parent: Any, name: str, *, h5py: ModuleType, kind: type) -> Any:
    link = parent.get(name, getlink=True)
    if not isinstance(link, h5py.HardLink):
        raise MinariHumanoidImportError(f"HDF5 field {name!r} must be a hard link")
    child = parent[name]
    if not isinstance(child, kind):
        raise MinariHumanoidImportError(f"HDF5 field {name!r} has the wrong object type")
    return child


def _dataset_array(
    group: Any,
    name: str,
    *,
    h5py: ModuleType,
    shape: tuple[int, ...],
    dtype: np.dtype[Any],
) -> np.ndarray:
    dataset = _require_hard_child(group, name, h5py=h5py, kind=h5py.Dataset)
    if tuple(dataset.shape) != shape or dataset.dtype != dtype:
        raise MinariHumanoidImportError(
            f"episode.{name} shape/dtype mismatch: observed={dataset.shape}/{dataset.dtype}, "
            f"expected={shape}/{dtype}"
        )
    creation = dataset.id.get_create_plist()
    if dataset.is_virtual or creation.get_layout() == h5py.h5d.VIRTUAL:
        raise MinariHumanoidImportError(f"episode.{name} virtual storage is forbidden")
    if dataset.external is not None or creation.get_external_count() != 0:
        raise MinariHumanoidImportError(f"episode.{name} external storage is forbidden")
    if (
        creation.get_nfilters() != 0
        or dataset.compression is not None
        or dataset.shuffle
        or dataset.fletcher32
        or dataset.scaleoffset is not None
    ):
        raise MinariHumanoidImportError(f"episode.{name} HDF5 filters are forbidden")
    if set(dataset.attrs):
        raise MinariHumanoidImportError(f"episode.{name} attributes must be empty")
    value = np.asarray(dataset[()])
    if value.shape != shape or value.dtype != dtype:
        raise MinariHumanoidImportError(f"episode.{name} changed while it was read")
    return value


def _episode_arrays(
    store: Any,
    *,
    h5py: ModuleType,
    metadata: Mapping[str, object],
    episode_id: int,
    expected_seed: int,
    expected_total_steps: int,
    expected_final_terminated: bool,
    expected_final_truncated: bool,
) -> dict[str, np.ndarray]:
    expected_root_names = {f"episode_{index}" for index in range(int(metadata["total_episodes"]))}
    actual_root_names = set(store.keys())
    if actual_root_names != expected_root_names:
        raise MinariHumanoidImportError("HDF5 root episode groups do not match metadata")
    if set(store.attrs):
        raise MinariHumanoidImportError("HDF5 root attributes must be empty")
    group = _require_hard_child(store, f"episode_{episode_id}", h5py=h5py, kind=h5py.Group)
    _exact_keys(group, _EPISODE_FIELDS, field="episode")
    _exact_keys(group.attrs, _EPISODE_ATTRIBUTE_FIELDS, field="episode attributes")
    observed_id = _exact_int(
        group.attrs["id"], field="episode attribute id", minimum=0, maximum=MAX_EPISODES - 1
    )
    observed_seed = _exact_int(
        group.attrs["seed"], field="episode attribute seed", minimum=0, maximum=2**63 - 1
    )
    observed_steps = _exact_int(
        group.attrs["total_steps"],
        field="episode attribute total_steps",
        minimum=1,
        maximum=MAX_EPISODE_STEPS,
    )
    if observed_id != episode_id:
        raise MinariHumanoidImportError("episode attribute id mismatch")
    if observed_seed != expected_seed:
        raise MinariHumanoidImportError("episode attribute seed mismatch")
    if observed_steps != expected_total_steps:
        raise MinariHumanoidImportError("episode attribute total_steps mismatch")
    for field in ("rewards_max", "rewards_mean", "rewards_min", "rewards_std", "rewards_sum"):
        value = group.attrs[field]
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise MinariHumanoidImportError(f"episode attribute {field} must be numeric")
        if not math.isfinite(float(value)):
            raise MinariHumanoidImportError(f"episode attribute {field} must be finite")

    infos = _require_hard_child(group, "infos", h5py=h5py, kind=h5py.Group)
    if set(infos.keys()) or set(infos.attrs):
        raise MinariHumanoidImportError(
            "episode.infos must be empty; root x/y classification would otherwise change"
        )
    steps = expected_total_steps
    arrays = {
        "observations": _dataset_array(
            group,
            "observations",
            h5py=h5py,
            shape=(steps + 1, MINARI_HUMANOID_OBSERVATION_WIDTH),
            dtype=np.dtype("float64"),
        ),
        "actions": _dataset_array(
            group,
            "actions",
            h5py=h5py,
            shape=(steps, MINARI_HUMANOID_ACTION_WIDTH),
            dtype=np.dtype("float32"),
        ),
        "rewards": _dataset_array(
            group,
            "rewards",
            h5py=h5py,
            shape=(steps,),
            dtype=np.dtype("float64"),
        ),
        "terminations": _dataset_array(
            group,
            "terminations",
            h5py=h5py,
            shape=(steps,),
            dtype=np.dtype("bool"),
        ),
        "truncations": _dataset_array(
            group,
            "truncations",
            h5py=h5py,
            shape=(steps,),
            dtype=np.dtype("bool"),
        ),
    }
    if not np.isfinite(arrays["observations"]).all():
        raise MinariHumanoidImportError("episode.observations contains non-finite values")
    if not np.isfinite(arrays["actions"]).all():
        raise MinariHumanoidImportError("episode.actions contains non-finite values")
    if not np.isfinite(arrays["rewards"]).all():
        raise MinariHumanoidImportError("episode.rewards contains non-finite values")
    expected_reward_attributes = {
        "rewards_max": float(np.max(arrays["rewards"])),
        "rewards_mean": float(np.mean(arrays["rewards"])),
        "rewards_min": float(np.min(arrays["rewards"])),
        "rewards_std": float(np.std(arrays["rewards"])),
        "rewards_sum": float(np.sum(arrays["rewards"])),
    }
    mismatched_reward_attributes = [
        field
        for field, expected in expected_reward_attributes.items()
        if not math.isclose(float(group.attrs[field]), expected, rel_tol=1e-12, abs_tol=1e-12)
    ]
    if mismatched_reward_attributes:
        raise MinariHumanoidImportError(
            "episode reward summary attributes mismatch: " + ", ".join(mismatched_reward_attributes)
        )
    action_limit = np.float32(0.4)
    if np.any(arrays["actions"] < -action_limit) or np.any(arrays["actions"] > action_limit):
        raise MinariHumanoidImportError("episode.actions exceeds Humanoid-v5 bounds")
    done = arrays["terminations"] | arrays["truncations"]
    if np.any(done[:-1]) or not bool(done[-1]):
        raise MinariHumanoidImportError("episode must end exactly at its final transition")
    if bool(arrays["terminations"][-1]) is not expected_final_terminated:
        raise MinariHumanoidImportError("episode final termination flag mismatch")
    if bool(arrays["truncations"][-1]) is not expected_final_truncated:
        raise MinariHumanoidImportError("episode final truncation flag mismatch")
    return arrays


def _projection_mapping_sha256() -> str:
    return hashlib.sha256(
        _canonical_json(
            {
                "projection_id": MINARI_HUMANOID_PROJECTION_ID,
                "indices": list(HUMANOID_V5_OBSERVATION_TO_REFERENCE_INDICES),
                "source_width": MINARI_HUMANOID_OBSERVATION_WIDTH,
                "target_schema_sha256": HUMANOID_REFERENCE_SCHEMA.sha256,
            }
        )
    ).hexdigest()


def _import_minari_humanoid_episode(
    *,
    hdf5_path: Path,
    metadata_path: Path,
    expected_hdf5_sha256: str,
    expected_metadata_sha256: str,
    expected_hdf5_bytes: int | None,
    expected_metadata_bytes: int | None,
    source_commit: str,
    source_provenance_class: str,
    source_record_sha256: str | None,
    source_record_bytes: int | None,
    episode_id: int,
    expected_seed: int,
    expected_total_steps: int,
    expected_final_terminated: bool,
    expected_final_truncated: bool,
) -> MinariHumanoidProjection:

    expected_hdf5_sha256 = _sha256(expected_hdf5_sha256, field="expected_hdf5_sha256")
    expected_metadata_sha256 = _sha256(expected_metadata_sha256, field="expected_metadata_sha256")
    source_commit = _source_commit(source_commit)
    is_registered = source_provenance_class == "registered_official_repository_revision"
    if source_provenance_class not in {
        "caller_pinned_unverified",
        "registered_official_repository_revision",
    }:
        raise MinariHumanoidImportError("unsupported source provenance class")
    if is_registered:
        hdf5_record = _REGISTERED_MINARI_HUMANOID_SOURCE_RECORD["files"]["hdf5"]
        metadata_record = _REGISTERED_MINARI_HUMANOID_SOURCE_RECORD["files"]["metadata"]
        expected_registration = (
            _REGISTERED_MINARI_HUMANOID_SOURCE_RECORD["source_commit"],
            hdf5_record["sha256"],
            hdf5_record["bytes"],
            metadata_record["sha256"],
            metadata_record["bytes"],
            REGISTERED_MINARI_HUMANOID_SOURCE_MANIFEST_SHA256,
            REGISTERED_MINARI_HUMANOID_SOURCE_MANIFEST_BYTES,
        )
        observed_registration = (
            source_commit,
            expected_hdf5_sha256,
            expected_hdf5_bytes,
            expected_metadata_sha256,
            expected_metadata_bytes,
            source_record_sha256,
            source_record_bytes,
        )
        if observed_registration != expected_registration:
            raise MinariHumanoidImportError("registered Minari source identity mismatch")
    elif source_record_sha256 is not None or source_record_bytes is not None:
        raise MinariHumanoidImportError(
            "caller-pinned source cannot claim a registered source record"
        )
    episode_id = _exact_int(episode_id, field="episode_id", minimum=0, maximum=MAX_EPISODES - 1)
    expected_seed = _exact_int(expected_seed, field="expected_seed", minimum=0, maximum=2**63 - 1)
    expected_total_steps = _exact_int(
        expected_total_steps,
        field="expected_total_steps",
        minimum=1,
        maximum=MAX_EPISODE_STEPS,
    )
    if not isinstance(expected_final_terminated, bool) or not isinstance(
        expected_final_truncated, bool
    ):
        raise MinariHumanoidImportError("expected final flags must be booleans")
    if expected_final_terminated == expected_final_truncated:
        raise MinariHumanoidImportError("exactly one expected final flag must be true")

    with _verified_binary_file(
        Path(metadata_path),
        expected_sha256=expected_metadata_sha256,
        expected_bytes=expected_metadata_bytes,
        max_bytes=MAX_METADATA_BYTES,
        field="metadata file",
    ) as (metadata_stream, metadata_bytes):
        metadata_raw = metadata_stream.read(MAX_METADATA_BYTES + 1)
    metadata = _validate_metadata(metadata_raw, episode_id=episode_id)

    h5py = _load_h5py()
    try:
        with (
            _verified_binary_file(
                Path(hdf5_path),
                expected_sha256=expected_hdf5_sha256,
                expected_bytes=expected_hdf5_bytes,
                max_bytes=MAX_HDF5_BYTES,
                field="HDF5 file",
            ) as (hdf5_stream, hdf5_bytes),
            h5py.File(hdf5_stream, mode="r") as store,
        ):
            arrays = _episode_arrays(
                store,
                h5py=h5py,
                metadata=metadata,
                episode_id=episode_id,
                expected_seed=expected_seed,
                expected_total_steps=expected_total_steps,
                expected_final_terminated=expected_final_terminated,
                expected_final_truncated=expected_final_truncated,
            )
    except MinariHumanoidImportError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise MinariHumanoidImportError(f"cannot read the pinned HDF5 episode: {exc}") from exc

    observations = arrays["observations"]
    reference_values = observations[:, HUMANOID_V5_OBSERVATION_TO_REFERENCE_INDICES]
    if np.array_equal(reference_values[1:], reference_values[:-1]):
        raise MinariHumanoidImportError("selected episode is time-invariant")
    adjacent_quaternion_dots = np.sum(
        reference_values[:-1, 1:5] * reference_values[1:, 1:5], axis=1
    )
    if np.any(adjacent_quaternion_dots <= 0.0):
        raise MinariHumanoidImportError("root quaternion sequence has a sign discontinuity")

    artifact_id = f"minari/{MINARI_HUMANOID_DATASET_ID}/episode-{episode_id}/projected-45d/v1"
    reference = ReferenceArtifact.create(
        artifact_id=artifact_id,
        schema=HUMANOID_REFERENCE_SCHEMA,
        values=reference_values,
    )
    try:
        validate_humanoid_reference(reference)
    except ValueError as exc:
        raise MinariHumanoidImportError(f"projected reference is invalid: {exc}") from exc

    receipt = MinariHumanoidProjectionReceipt(
        source_commit=source_commit,
        source_provenance_class=source_provenance_class,
        source_origin_verified=is_registered,
        source_record_sha256=source_record_sha256,
        source_record_bytes=source_record_bytes,
        hdf5_sha256=expected_hdf5_sha256,
        hdf5_bytes=hdf5_bytes,
        metadata_sha256=expected_metadata_sha256,
        metadata_bytes=metadata_bytes,
        minari_version=str(metadata["minari_version"]),
        declared_requirements=tuple(metadata["requirements"]),
        episode_id=episode_id,
        episode_seed=expected_seed,
        episode_total_steps=expected_total_steps,
        observations_sha256=_array_sha256(observations),
        actions_sha256=_array_sha256(arrays["actions"]),
        rewards_sha256=_array_sha256(arrays["rewards"]),
        terminations_sha256=_array_sha256(arrays["terminations"]),
        truncations_sha256=_array_sha256(arrays["truncations"]),
        ignored_observation_fields_sha256=_array_sha256(observations[:, 45:]),
        projection_mapping_sha256=_projection_mapping_sha256(),
        reference_artifact_id=reference.identity.artifact_id,
        reference_content_sha256=reference.identity.content_sha256,
        reference_schema_sha256=reference.identity.schema_sha256,
        reference_frames=reference.identity.n_frames,
    )
    immutable_observations = np.frombuffer(
        observations.tobytes(order="C"),
        dtype=np.dtype("<f8"),
    ).reshape(observations.shape)
    return MinariHumanoidProjection(
        episode_observations=immutable_observations,
        reference=reference,
        receipt=receipt,
    )


def import_minari_humanoid_episode(
    *,
    hdf5_path: Path,
    metadata_path: Path,
    expected_hdf5_sha256: str,
    expected_metadata_sha256: str,
    source_commit: str,
    episode_id: int,
    expected_seed: int,
    expected_total_steps: int,
    expected_final_terminated: bool,
    expected_final_truncated: bool,
) -> MinariHumanoidProjection:
    """Project caller-pinned bytes while marking source origin unverified."""

    return _import_minari_humanoid_episode(
        hdf5_path=hdf5_path,
        metadata_path=metadata_path,
        expected_hdf5_sha256=expected_hdf5_sha256,
        expected_metadata_sha256=expected_metadata_sha256,
        expected_hdf5_bytes=None,
        expected_metadata_bytes=None,
        source_commit=source_commit,
        source_provenance_class="caller_pinned_unverified",
        source_record_sha256=None,
        source_record_bytes=None,
        episode_id=episode_id,
        expected_seed=expected_seed,
        expected_total_steps=expected_total_steps,
        expected_final_terminated=expected_final_terminated,
        expected_final_truncated=expected_final_truncated,
    )


def import_registered_minari_humanoid_episode(
    *,
    hdf5_path: Path,
    metadata_path: Path,
    episode_id: int,
    expected_seed: int,
    expected_total_steps: int,
    expected_final_terminated: bool,
    expected_final_truncated: bool,
) -> MinariHumanoidProjection:
    """Project bytes matching the package-registered official source record."""

    record, source_record_sha256, source_record_bytes = _registered_source_record()
    files = record["files"]
    hdf5 = files["hdf5"]
    metadata = files["metadata"]
    return _import_minari_humanoid_episode(
        hdf5_path=hdf5_path,
        metadata_path=metadata_path,
        expected_hdf5_sha256=hdf5["sha256"],
        expected_metadata_sha256=metadata["sha256"],
        expected_hdf5_bytes=hdf5["bytes"],
        expected_metadata_bytes=metadata["bytes"],
        source_commit=record["source_commit"],
        source_provenance_class=record["provenance_class"],
        source_record_sha256=source_record_sha256,
        source_record_bytes=source_record_bytes,
        episode_id=episode_id,
        expected_seed=expected_seed,
        expected_total_steps=expected_total_steps,
        expected_final_terminated=expected_final_terminated,
        expected_final_truncated=expected_final_truncated,
    )


__all__ = [
    "HUMANOID_V5_OBSERVATION_TO_REFERENCE_INDICES",
    "MINARI_HUMANOID_ACTION_WIDTH",
    "MINARI_HUMANOID_DATASET_ID",
    "MINARI_HUMANOID_GENERATION_REPOSITORY",
    "MINARI_HUMANOID_OBSERVATION_WIDTH",
    "MINARI_HUMANOID_PROJECTION_ID",
    "MINARI_HUMANOID_SOURCE_REPOSITORY",
    "REGISTERED_MINARI_HUMANOID_SOURCE_MANIFEST_BYTES",
    "REGISTERED_MINARI_HUMANOID_SOURCE_MANIFEST_SHA256",
    "MinariHumanoidImportError",
    "MinariHumanoidProjection",
    "MinariHumanoidProjectionReceipt",
    "import_minari_humanoid_episode",
    "import_registered_minari_humanoid_episode",
]
