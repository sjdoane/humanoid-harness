"""Exact probe-only admission for reviewed execution-derived GMT references."""

from __future__ import annotations

import hashlib
import io
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oracle_composition.harness.contract import decode_json_object

from .io import GMTAdmissionError, read_verified_bytes, validate_zip_members
from .reference_runtime import ReferenceMotion


@dataclass(frozen=True, slots=True)
class _ArrayContract:
    shape: tuple[int, ...]
    dtype: str
    member_size: int
    values_sha256: str


@dataclass(frozen=True, slots=True)
class DerivedReferenceSpec:
    """One exact reviewed candidate; this is not a dynamics certificate."""

    name: str
    candidate_id: str
    archive_sha256: str
    archive_size: int
    manifest_sha256: str
    manifest_size: int
    arrays: dict[str, _ArrayContract]


@dataclass(frozen=True, slots=True)
class DerivedReferenceAdmission:
    motion: ReferenceMotion
    path: Path
    resource_binding: dict[str, object]


STUDY012_INSIDE_PASSAGE = DerivedReferenceSpec(
    name="study012_inside_passage_v1",
    candidate_id="gmt_g1_study012_executed_inside_passage/v1",
    archive_sha256="885e4c1b324a9b41a4d176ec5fee9e4bc634226d5ade46634111cb1c3204c259",
    archive_size=13_650,
    manifest_sha256="1c7edb409579d757ef6657378ab605e6aa013cf60b552630beb914728bec602c",
    manifest_size=10_965,
    arrays={
        "dof_pos": _ArrayContract(
            (106, 23),
            "<f4",
            9_880,
            "16bc89a26a70cdc17890126c4b6b91fd6a00fee6101683c8f5e60456c2066f31",
        ),
        "fps": _ArrayContract(
            (1,),
            "<f8",
            136,
            "fe555d486cc8379252e92fe7c97b5e5b603a97a9f965105cfd169a1ab4327fb9",
        ),
        "root_pos": _ArrayContract(
            (106, 3),
            "<f4",
            1_400,
            "0225143c1e8f61efd0ec09efaf87696331376b9178f48c9d0f17605ac31646a8",
        ),
        "root_rot": _ArrayContract(
            (106, 4),
            "<f4",
            1_824,
            "734aa6c2a88ff29c9f7dc66b0871d388f0c42976e567125792f0b5e2131fb729",
        ),
    },
)
DERIVED_REFERENCE_SPECS = {STUDY012_INSIDE_PASSAGE.name: STUDY012_INSIDE_PASSAGE}

_MANIFEST_FIELDS = {
    "schema_version",
    "artifact",
    "candidate_id",
    "claim_ceiling",
    "donor",
    "measurements",
    "model",
    "output",
    "promotion",
    "source",
    "static_geometry",
    "transform",
}
_CLAIM_CEILING = (
    "hash_pinned_kinematic_candidate_only_not_dynamics_certified_admitted_or_retracked"
)


def is_derived_reference_name(name: object) -> bool:
    return type(name) is str and name in DERIVED_REFERENCE_SPECS


def _direct_file(value: object, *, field: str, suffix: str) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        raise GMTAdmissionError(f"{field} path is malformed")
    path = Path(value)
    if not path.is_absolute() or path.suffix != suffix:
        raise GMTAdmissionError(f"{field} must be an absolute {suffix} file")
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise GMTAdmissionError(f"{field} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or resolved != path:
        raise GMTAdmissionError(f"{field} must be a direct regular file")
    return path


def _manifest_array_contract(spec: DerivedReferenceSpec) -> dict[str, object]:
    return {
        name: {
            "c_order_values_sha256": contract.values_sha256,
            "dtype": contract.dtype,
            "shape": list(contract.shape),
        }
        for name, contract in spec.arrays.items()
    }


def _validate_manifest(encoded: bytes, spec: DerivedReferenceSpec) -> dict[str, Any]:
    manifest = decode_json_object(encoded, source="execution-derived reference manifest")
    output = manifest.get("output")
    source = manifest.get("source")
    geometry = manifest.get("static_geometry")
    measurements = manifest.get("measurements")
    transform = manifest.get("transform")
    donor = manifest.get("donor")
    if (
        set(manifest) != _MANIFEST_FIELDS
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or manifest.get("artifact") != "gmt_g1_execution_derived_reference_candidate"
        or manifest.get("candidate_id") != spec.candidate_id
        or manifest.get("claim_ceiling") != _CLAIM_CEILING
        or manifest.get("promotion")
        != "none; separate reviewed admission and re-tracking evidence required"
        or type(output) is not dict
        or output
        != {
            "arrays": _manifest_array_contract(spec),
            "format": "deterministic_numeric_only_npz",
            "path": f"{spec.name}.npz",
            "sha256": spec.archive_sha256,
            "size": spec.archive_size,
        }
        or type(source) is not dict
        or source.get("reviewed_checkout")
        != {
            "authority": "caller_supplied_exact_commit_from_external_completed_review",
            "checked_before_donor_reads_or_output_writes": True,
            "commit": "e889f0c157811384973ccd27571b1f83766fc020",
            "working_tree": "clean_including_staged_unstaged_and_untracked_files",
        }
        or type(geometry) is not dict
        or geometry.get("method") != "mujoco.mj_forward_only_with_qvel_zero_no_mj_step"
        or geometry.get("evaluated_pose_count") != 106
        or geometry.get("nonfoot_ground_contact_pose_count") != 0
        or geometry.get("reviewed_maximum_allowed_foot_penetration_m") != 0.01544
        or type(measurements) is not dict
        or measurements.get("frame_count") != 106
        or measurements.get("duration_seconds") != 2.1
        or type(transform) is not dict
        or transform.get("cadence_hz") != 50.0
        or transform.get("selection")
        != "qpos[92:198]; frame rows 92:197 are the 105 executed intervals"
        or type(donor) is not dict
        or donor.get("source_commit") != "9bca23bf62ea386cd7f0492aa63626b6c2a219dc"
        or type(donor.get("execution_contact_evidence")) is not dict
        or donor["execution_contact_evidence"].get("nonfoot_ground_contact_observed") is not False
    ):
        raise GMTAdmissionError("execution-derived reference provenance differs")
    return manifest


def _load_arrays(payload: bytes, spec: DerivedReferenceSpec) -> dict[str, np.ndarray]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise GMTAdmissionError("execution-derived reference is not a valid NPZ") from exc
    expected = {f"{name}.npy": contract for name, contract in spec.arrays.items()}
    arrays: dict[str, np.ndarray] = {}
    with archive:
        members = validate_zip_members(
            archive,
            expected_count=len(expected),
            expected_uncompressed_size=sum(item.member_size for item in expected.values()),
            maximum_member_size=max(item.member_size for item in expected.values()),
        )
        if set(members) != set(expected):
            raise GMTAdmissionError("execution-derived reference member set differs")
        for member_name, contract in expected.items():
            info = members[member_name]
            if info.file_size != contract.member_size or info.compress_type != zipfile.ZIP_STORED:
                raise GMTAdmissionError(f"execution-derived member envelope differs: {member_name}")
            member_payload = archive.read(member_name)
            stream = io.BytesIO(member_payload)
            try:
                array = np.lib.format.read_array(stream, allow_pickle=False, max_header_size=512)
            except (ValueError, EOFError) as exc:
                raise GMTAdmissionError(f"invalid execution-derived member: {member_name}") from exc
            if (
                stream.tell() != len(member_payload)
                or array.shape != contract.shape
                or array.dtype.str != contract.dtype
                or not array.flags.c_contiguous
                or not np.isfinite(array).all()
                or hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
                != contract.values_sha256
            ):
                raise GMTAdmissionError(f"execution-derived array contract differs: {member_name}")
            arrays[member_name.removesuffix(".npy")] = array
    if arrays["fps"].item() != 50.0 or not np.allclose(
        np.linalg.norm(arrays["root_rot"], axis=1), 1.0, rtol=0.0, atol=1e-6
    ):
        raise GMTAdmissionError("execution-derived cadence or quaternion contract differs")
    return arrays


def admit_derived_reference(
    name: object, asset: object, *, mode: object
) -> DerivedReferenceAdmission:
    """Admit the one reviewed candidate for a probe, preserving its evidence ceiling."""

    if not is_derived_reference_name(name):
        raise GMTAdmissionError("reference is outside the derived candidate registry")
    if mode != "probe":
        raise GMTAdmissionError("execution-derived reference is probe-only pending re-tracking")
    if type(asset) is not dict or set(asset) != {"path", "sha256", "manifest"}:
        raise GMTAdmissionError("execution-derived asset fields differ")
    manifest_binding = asset["manifest"]
    if type(manifest_binding) is not dict or set(manifest_binding) != {"path", "sha256"}:
        raise GMTAdmissionError("execution-derived manifest binding fields differ")
    spec = DERIVED_REFERENCE_SPECS[name]
    if asset["sha256"] != spec.archive_sha256 or manifest_binding["sha256"] != spec.manifest_sha256:
        raise GMTAdmissionError("execution-derived asset identity differs")
    archive_path = _direct_file(asset["path"], field="execution-derived reference", suffix=".npz")
    manifest_path = _direct_file(
        manifest_binding["path"], field="execution-derived manifest", suffix=".json"
    )
    if manifest_path != archive_path.with_suffix(f"{archive_path.suffix}.manifest.json"):
        raise GMTAdmissionError("execution-derived manifest must be the archive sibling")
    manifest_payload = read_verified_bytes(
        manifest_path,
        spec.manifest_sha256,
        expected_size=spec.manifest_size,
        maximum_size=spec.manifest_size,
    )
    _validate_manifest(manifest_payload, spec)
    archive_payload = read_verified_bytes(
        archive_path,
        spec.archive_sha256,
        expected_size=spec.archive_size,
        maximum_size=spec.archive_size,
    )
    motion = ReferenceMotion(_load_arrays(archive_payload, spec))
    return DerivedReferenceAdmission(
        motion=motion,
        path=archive_path,
        resource_binding={
            "path": str(archive_path),
            "sha256": spec.archive_sha256,
            "size": spec.archive_size,
            "manifest": {
                "path": str(manifest_path),
                "sha256": spec.manifest_sha256,
                "size": spec.manifest_size,
            },
            "provenance_class": "execution_derived_kinematic_candidate_not_dynamics_certificate",
            "candidate_id": spec.candidate_id,
            "training_admitted": False,
        },
    )


__all__ = [
    "DERIVED_REFERENCE_SPECS",
    "DerivedReferenceAdmission",
    "admit_derived_reference",
    "is_derived_reference_name",
]
