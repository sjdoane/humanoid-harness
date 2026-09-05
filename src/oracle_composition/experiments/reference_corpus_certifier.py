"""Independent-process all-transition Tier-D replay certifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import (
    ArtifactBindingV2,
    array_sha256,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
    transition_indices_sha256,
    validate_full_clip_certificate,
)
from oracle_composition.envs.reference_corpus import (
    FullSubstepContactSample,
    make_reference_corpus_env,
)
from oracle_composition.sources.strict_tqc_actor_runtime import StrictTQCActorRuntime

from .artifact_io import publish_bytes_without_overwrite
from .reference_corpus_bundle import (
    load_bundle_manifest,
    verify_bound_artifacts,
)
from .reference_corpus_collector import (
    _capture_boundary,
    _contact_record,
    inspect_reference_corpus_runtime,
    restore_rng_state,
    restore_wrapper_state,
)
from .reference_corpus_contract import (
    ReferenceCorpusContractError,
    array_sha256_from_binding,
    decode_clip_payload,
    derive_reference_row,
    require_same_runtime,
    validate_bundle_manifest,
)

CERTIFIER_ID = "separate_process_all_transition_tier_d_certifier/v1"


class FullClipReplayMismatch(RuntimeError):
    """One exact field differed while replaying a clip."""

    def __init__(self, clip_id: str, transition: int, field: str) -> None:
        self.clip_id = clip_id
        self.transition = transition
        self.field = field
        super().__init__(f"{clip_id} transition {transition}: {field} differs")


@dataclass(frozen=True, slots=True)
class PublishedTierDCertificate:
    clip_id: str
    path: Path
    sha256: str
    transitions_verified: int


def _binding_by_role(core: Mapping[str, object], role: str) -> ArtifactBindingV2:
    raw_bindings = core["bound_artifacts"]
    matches = [raw for raw in raw_bindings if raw.get("role") == role]
    if len(matches) != 1:
        raise ReferenceCorpusContractError(f"bundle role {role!r} is missing or duplicated")
    raw = matches[0]
    return ArtifactBindingV2(
        role=raw["role"],
        logical_path=raw["logical_path"],
        object_path=raw["object_path"],
        sha256=raw["sha256"],
        byte_count=raw["byte_count"],
    )


def _payload_binding(core: Mapping[str, object]) -> ArtifactBindingV2:
    raw = core["payload"]
    return ArtifactBindingV2(
        role=raw["role"],
        logical_path=raw["logical_path"],
        object_path=raw["object_path"],
        sha256=raw["sha256"],
        byte_count=raw["byte_count"],
    )


def _load_rng_state(core: Mapping[str, object], artifact_root: Path) -> dict[str, object]:
    raw = core["rng_state"]["binding"]
    binding = ArtifactBindingV2(
        role=raw["role"],
        logical_path=raw["logical_path"],
        object_path=raw["object_path"],
        sha256=raw["sha256"],
        byte_count=raw["byte_count"],
    )
    path = artifact_root / binding.object_path
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != binding.sha256:
        raise ReferenceCorpusContractError("environment RNG object hash differs")
    try:
        value = json.loads(encoded.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError) as exc:
        raise ReferenceCorpusContractError("environment RNG object is invalid") from exc
    if type(value) is not dict or canonical_json_bytes(value) != encoded:
        raise ReferenceCorpusContractError("environment RNG object is not canonical")
    return value


def _bytes_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def _float_equal(left: float, right: float) -> bool:
    return struct.pack(">d", float(left)) == struct.pack(">d", float(right))


def _require_array(
    observed: object,
    expected: np.ndarray,
    *,
    clip_id: str,
    transition: int,
    field: str,
    dtype: str = "<f8",
) -> None:
    value = np.ascontiguousarray(observed, dtype=dtype)
    if not _bytes_equal(value, expected):
        raise FullClipReplayMismatch(clip_id, transition, field)


def _verify_contacts(
    samples: list[FullSubstepContactSample],
    arrays: Mapping[str, np.ndarray],
    transition: int,
    *,
    clip_id: str,
) -> None:
    start = int(arrays["contact_offsets"][transition])
    stop = int(arrays["contact_offsets"][transition + 1])
    if len(samples) != stop - start:
        raise FullClipReplayMismatch(clip_id, transition, "contact count")
    for relative, sample in enumerate(samples):
        index = start + relative
        exact_integers = {
            "contact substep": (sample.physics_substep_index, arrays["contact_substep"][index]),
            "contact index": (
                sample.contact_index_within_substep,
                arrays["contact_index_within_substep"][index],
            ),
            "contact geom1 id": (sample.geom1_id, arrays["contact_geom1_id"][index]),
            "contact geom2 id": (sample.geom2_id, arrays["contact_geom2_id"][index]),
        }
        for field, (observed, expected) in exact_integers.items():
            if observed != int(expected):
                raise FullClipReplayMismatch(clip_id, transition, field)
        expected_name1 = bytes(arrays["contact_geom1_name"][index]).rstrip(b"\0").decode()
        expected_name2 = bytes(arrays["contact_geom2_name"][index]).rstrip(b"\0").decode()
        if sample.geom1_name != expected_name1 or sample.geom2_name != expected_name2:
            raise FullClipReplayMismatch(clip_id, transition, "contact geom name")
        _require_array(
            sample.force_torque_contact_frame,
            arrays["contact_force_torque"][index],
            clip_id=clip_id,
            transition=transition,
            field="contact force",
        )


def _verify_bundle_arrays(core: Mapping[str, object], arrays: Mapping[str, np.ndarray]) -> None:
    if len(core["array_bindings"]) != len(arrays):
        raise ReferenceCorpusContractError("bundle array binding count differs")
    for name, value in arrays.items():
        if array_sha256(value) != array_sha256_from_binding(core["array_bindings"], name):
            raise ReferenceCorpusContractError(f"bundle array binding differs: {name}")


def certify_bundle(
    *,
    artifact_root: Path,
    manifest_path: Path,
    manifest_sha256: str,
) -> PublishedTierDCertificate:
    """Replay every transition and publish exactly one pass certificate."""

    root = Path(artifact_root)
    manifest = load_bundle_manifest(manifest_path, expected_sha256=manifest_sha256)
    validate_bundle_manifest(manifest)
    verify_bound_artifacts(manifest, artifact_root=root)
    core = manifest["core"]
    if sha256_file(Path(__file__)) != core["source_hashes"]["certifier_source"]:
        raise ReferenceCorpusContractError("running certifier source differs from the bundle")
    clip_id = core["clip_id"]
    steps = core["steps"]
    screen_canary = core["clip_kind"] == "development_screen"
    payload_binding = _payload_binding(core)
    payload = (root / payload_binding.object_path).read_bytes()
    arrays = decode_clip_payload(payload, steps=steps, screen_canary=screen_canary)
    _verify_bundle_arrays(core, arrays)
    actor_binding = _binding_by_role(core, "strict_actor_npz")
    source_actor = core["source_actor"]
    expected_artifact_hashes = {
        "strict_actor_npz": "npz_sha256",
        "source_policy_bytes": "source_policy_sha256",
        "import_receipt": "import_receipt_sha256",
        "equivalence_receipt": "equivalence_receipt_sha256",
    }
    for role, field in expected_artifact_hashes.items():
        if _binding_by_role(core, role).sha256 != source_actor[field]:
            raise ReferenceCorpusContractError(f"source actor {field} differs from its object")
    actor = StrictTQCActorRuntime.from_npz(
        root / actor_binding.object_path,
        expected_sha256=actor_binding.sha256,
    )
    if actor.loaded_actor.state_sha256 != source_actor["actor_state_sha256"]:
        raise ReferenceCorpusContractError("source actor state hash differs")
    rng_state = _load_rng_state(core, root)
    rng_sha256 = sha256_json(rng_state)
    if any(bytes(value).decode() != rng_sha256 for value in arrays["boundary_rng_state_sha256"]):
        raise ReferenceCorpusContractError("boundary RNG binding differs")
    environment = make_reference_corpus_env()
    verification_digest = hashlib.sha256()
    try:
        environment.reset(seed=0)
        observed_runtime = inspect_reference_corpus_runtime(environment)
        require_same_runtime(core["runtime_identity"], observed_runtime)
        import mujoco

        physical = environment.unwrapped
        for transition in range(steps):
            mujoco.mj_setState(
                physical.model,
                physical.data,
                arrays["boundary_integration_state"][transition],
                mujoco.mjtState.mjSTATE_INTEGRATION,
            )
            restore_wrapper_state(
                environment,
                int(arrays["boundary_wrapper_elapsed"][transition]),
                arrays["boundary_wrapper_flags"][transition],
            )
            restore_rng_state(physical, rng_state)
            anchor = _capture_boundary(
                environment,
                terminated=bool(arrays["boundary_result_flags"][transition, 0]),
                truncated=bool(arrays["boundary_result_flags"][transition, 1]),
            )
            for field, array_name in (
                ("integration state", "boundary_integration_state"),
                ("observation", "boundary_observation"),
                ("cfrc_ext", "boundary_cfrc_ext"),
                ("root x/y", "boundary_root_xy"),
            ):
                _require_array(
                    anchor[array_name.removeprefix("boundary_")],
                    arrays[array_name][transition],
                    clip_id=clip_id,
                    transition=transition,
                    field=f"restored {field}",
                )
            if not _float_equal(
                anchor["simulation_time"], arrays["boundary_simulation_time"][transition]
            ):
                raise FullClipReplayMismatch(clip_id, transition, "restored simulation time")
            if anchor["wrapper_elapsed"] != int(
                arrays["boundary_wrapper_elapsed"][transition]
            ) or not np.array_equal(
                anchor["wrapper_flags"], arrays["boundary_wrapper_flags"][transition]
            ):
                raise FullClipReplayMismatch(clip_id, transition, "restored wrapper state")
            if not _float_equal(anchor["torso_up_z"], arrays["boundary_torso_up_z"][transition]):
                raise FullClipReplayMismatch(clip_id, transition, "restored torso up axis")
            previous_quaternion = (
                arrays["reference_rows"][transition - 1, 1:5] if transition > 0 else None
            )
            restored_row = derive_reference_row(
                physical.data.qpos,
                physical.data.qvel,
                previous_quaternion=previous_quaternion,
            )
            _require_array(
                restored_row,
                arrays["reference_rows"][transition],
                clip_id=clip_id,
                transition=transition,
                field="restored reference row",
            )
            action = actor.act(anchor["observation"])
            expected_hashes = {
                "actor input hash": (
                    action.actor_input_sha256,
                    arrays["transition_actor_input_sha256"][transition],
                ),
                "actor output hash": (
                    action.actor_output_sha256,
                    arrays["transition_actor_output_sha256"][transition],
                ),
                "physical action hash": (
                    action.physical_action_sha256,
                    arrays["transition_physical_action_sha256"][transition],
                ),
            }
            for field, (observed, expected) in expected_hashes.items():
                if observed != bytes(expected).decode():
                    raise FullClipReplayMismatch(clip_id, transition, field)
            _require_array(
                action.normalized_output,
                arrays["transition_normalized_action"][transition],
                clip_id=clip_id,
                transition=transition,
                field="actor normalized output",
                dtype="<f4",
            )
            _require_array(
                action.physical_action,
                arrays["transition_physical_action"][transition],
                clip_id=clip_id,
                transition=transition,
                field="physical action",
                dtype="<f4",
            )
            returned, reward, terminated, truncated, _info = environment.step(
                action.physical_action.copy()
            )
            samples, nonfoot = _contact_record(environment)
            _verify_contacts(samples, arrays, transition, clip_id=clip_id)
            if nonfoot is not bool(arrays["transition_nonfoot_floor_contact"][transition]):
                raise FullClipReplayMismatch(clip_id, transition, "non-foot floor contact")
            _require_array(
                returned,
                arrays["transition_returned_observation"][transition],
                clip_id=clip_id,
                transition=transition,
                field="returned observation",
            )
            if not _float_equal(reward, arrays["transition_reward"][transition]):
                raise FullClipReplayMismatch(clip_id, transition, "reward canary")
            expected_flags = arrays["transition_flags"][transition]
            if (bool(terminated), bool(truncated)) != (
                bool(expected_flags[0]),
                bool(expected_flags[1]),
            ):
                raise FullClipReplayMismatch(clip_id, transition, "termination flags")
            next_boundary = _capture_boundary(
                environment,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            next_index = transition + 1
            for field, array_name in (
                ("next integration state", "boundary_integration_state"),
                ("next observation", "boundary_observation"),
                ("next cfrc_ext", "boundary_cfrc_ext"),
                ("next root x/y", "boundary_root_xy"),
            ):
                _require_array(
                    next_boundary[array_name.removeprefix("boundary_")],
                    arrays[array_name][next_index],
                    clip_id=clip_id,
                    transition=transition,
                    field=field,
                )
            if next_boundary["wrapper_elapsed"] != int(
                arrays["transition_next_wrapper_elapsed"][transition]
            ) or not np.array_equal(
                next_boundary["wrapper_flags"],
                arrays["transition_next_wrapper_flags"][transition],
            ):
                raise FullClipReplayMismatch(clip_id, transition, "next wrapper state")
            if not _float_equal(
                next_boundary["simulation_time"],
                arrays["boundary_simulation_time"][next_index],
            ):
                raise FullClipReplayMismatch(clip_id, transition, "next simulation time")
            next_row = derive_reference_row(
                physical.data.qpos,
                physical.data.qvel,
                previous_quaternion=arrays["reference_rows"][transition, 1:5],
            )
            _require_array(
                next_row,
                arrays["reference_rows"][next_index],
                clip_id=clip_id,
                transition=transition,
                field="next reference row",
            )
            record = {
                "transition": transition,
                "boundary_record_sha256": bytes(
                    arrays["boundary_record_sha256"][transition]
                ).decode(),
                "transition_record_sha256": bytes(
                    arrays["transition_record_sha256"][transition]
                ).decode(),
                "next_boundary_record_sha256": bytes(
                    arrays["boundary_record_sha256"][next_index]
                ).decode(),
            }
            encoded_record = canonical_json_bytes(record)
            verification_digest.update(len(encoded_record).to_bytes(8, "big"))
            verification_digest.update(encoded_record)
    finally:
        environment.close()
    certificate = {
        "certificate_id": "humanoid_full_clip_tier_d_certificate/v1",
        "schema_version": 1,
        "clip_id": clip_id,
        "bundle_manifest_sha256": manifest_sha256,
        "reference_identity_sha256": manifest["reference_identity_sha256"],
        "payload_sha256": payload_binding.sha256,
        "collector_pid": core["collector_pid"],
        "certifier_pid": os.getpid(),
        "steps_expected": steps,
        "transitions_verified": steps,
        "covered_transition_indices_sha256": transition_indices_sha256(steps),
        "verification_digest_sha256": verification_digest.hexdigest(),
        "all_hashes_verified": True,
        "all_transitions_passed": True,
        "first_failure": None,
        "evidence_class": "same_runtime_tier_d_replay",
        "claim_ceiling": "full_clip_same_host_replay_only",
    }
    validate_full_clip_certificate(certificate)
    encoded = canonical_json_bytes(certificate)
    digest = hashlib.sha256(encoded).hexdigest()
    path = manifest_path.parent / f"tier-d-{digest}.json"
    published = publish_bytes_without_overwrite(path, encoded)
    if published.sha256 != digest:
        raise ReferenceCorpusContractError("published Tier-D certificate hash differs")
    return PublishedTierDCertificate(
        clip_id=clip_id,
        path=path,
        sha256=digest,
        transitions_verified=steps,
    )


def certify_request(path: Path) -> dict[str, object]:
    encoded = Path(path).read_bytes()
    try:
        request = json.loads(encoded.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError) as exc:
        raise ReferenceCorpusContractError("certifier request is invalid JSON") from exc
    if canonical_json_bytes(request) != encoded or set(request) != {
        "request_id",
        "artifact_root",
        "bundles",
    }:
        raise ReferenceCorpusContractError("certifier request differs")
    if request["request_id"] != "reference_corpus_tier_d_certifier_request/v1":
        raise ReferenceCorpusContractError("certifier request identity differs")
    root = Path(request["artifact_root"])
    if type(request["bundles"]) is not list or not request["bundles"]:
        raise ReferenceCorpusContractError("certifier request has no bundles")
    results: list[dict[str, object]] = []
    for item in request["bundles"]:
        if type(item) is not dict or set(item) != {"manifest_path", "manifest_sha256"}:
            raise ReferenceCorpusContractError("certifier bundle request is malformed")
        certificate = certify_bundle(
            artifact_root=root,
            manifest_path=Path(item["manifest_path"]),
            manifest_sha256=item["manifest_sha256"],
        )
        results.append(
            {
                "clip_id": certificate.clip_id,
                "certificate_path": certificate.path.as_posix(),
                "certificate_sha256": certificate.sha256,
                "transitions_verified": certificate.transitions_verified,
            }
        )
        print(
            f"certified {len(results)}/{len(request['bundles'])} {certificate.clip_id}",
            file=sys.stderr,
            flush=True,
        )
    return {
        "certifier_id": CERTIFIER_ID,
        "certifier_pid": os.getpid(),
        "certificate_count": len(results),
        "certificates": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    arguments = parser.parse_args(argv)
    result = certify_request(arguments.request)
    print(canonical_json_bytes(result).decode("utf-8"))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess
    raise SystemExit(main())


__all__ = [
    "CERTIFIER_ID",
    "FullClipReplayMismatch",
    "PublishedTierDCertificate",
    "certify_bundle",
    "certify_request",
    "main",
]
