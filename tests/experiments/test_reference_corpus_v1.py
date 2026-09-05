from __future__ import annotations

import copy
import hashlib
import json
import struct
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.contracts.reference_identity_v2 import (
    CONTROL_PERIOD_SECONDS,
    CORPUS_ACTORS,
    CORPUS_SEEDS,
    FULL_CLIP_CERTIFICATE_ID,
    REFERENCE_SCHEMA_ID,
    REPLAY_METHOD_VERSION,
    ROBOT_ID,
    ReferenceIdentityV2,
    ReferenceIdentityV2Error,
    array_sha256,
    canonical_json_bytes,
    corpus_manifest_payload,
    e3_manifest_payload,
    sha256_file,
    sha256_json,
    transition_indices_sha256,
    validate_corpus_manifest,
    validate_e3_manifest,
    validate_full_clip_certificate,
    validate_reference_derivation,
)
from oracle_composition.envs.humanoid import make_humanoid_env
from oracle_composition.envs.reference_corpus import (
    EXPECTED_REFERENCE_WRAPPER_TYPES,
    FULL_CONTACT_CAPTURE_ID,
    make_reference_corpus_env,
)
from oracle_composition.experiments import (
    reference_corpus_certifier,
    reference_corpus_collector,
    reference_corpus_runner,
    tqc_development_metrics,
)
from oracle_composition.experiments.e3_fork_certifier import (
    E3Branch,
    E3CertificationError,
    evaluate_e3_pair,
    require_distinct_actor_states,
    require_same_state_anchors,
)
from oracle_composition.experiments.expert_development_screen import (
    development_episode_from_arrays,
)
from oracle_composition.experiments.reference_corpus_bundle import (
    load_bundle_manifest,
    publish_clip_bundle,
)
from oracle_composition.experiments.reference_corpus_collector import (
    CollectedClip,
    collect_reference_clip,
    restore_wrapper_state,
)
from oracle_composition.experiments.reference_corpus_contract import (
    EXPECTED_STATE_COMPONENT_SIZES,
    ReferenceCorpusContractError,
    array_bindings,
    boundary_record_sha256,
    contact_sequence_sha256,
    decode_clip_payload,
    derive_reference_rows,
    encode_clip_payload,
    plain_comparison_boundary_sha256,
    plain_comparison_receipt,
    plain_comparison_transition_sha256,
    reference_schema_payload,
    reference_window_index_arrays,
    require_same_runtime,
    transition_record_sha256,
    validate_bundle_manifest,
    validate_clip_arrays,
    validate_runtime_identity,
)
from oracle_composition.experiments.runtime_identity import dependency_lock_path
from oracle_composition.experiments.tqc_actor_npz import (
    actor_schema,
    actor_schema_sha256,
    actor_state_sha256,
    validate_actor_arrays,
    write_actor_npz_exclusive,
)
from oracle_composition.sources.strict_tqc_actor_runtime import (
    INFERENCE_ID,
    StrictTQCActorRuntime,
)

_DIGEST = "a" * 64


def _synthetic_clip_arrays(*, steps: int = 8) -> dict[str, np.ndarray]:
    boundaries = steps + 1
    integration = np.zeros((boundaries, 195), dtype="<f8")
    times = np.arange(boundaries, dtype="<f8") * CONTROL_PERIOD_SECONDS
    integration[:, 0] = times
    integration[:, 1] = np.arange(boundaries, dtype="<f8") * 0.003
    integration[:, 3] = 1.4
    integration[:, 4] = 1.0
    integration[:, 25] = 0.2
    observation = np.zeros((boundaries, 348), dtype="<f8")
    cfrc = np.zeros((boundaries, 13, 6), dtype="<f8")
    wrapper_flags = np.zeros((boundaries, 5), dtype="|u1")
    wrapper_flags[:, 0:2] = 1
    wrapper_flags[1:, 2] = 1
    result_flags = np.zeros((boundaries, 2), dtype="|u1")
    reference = derive_reference_rows(
        np.ascontiguousarray(integration[:, 1:25]),
        np.ascontiguousarray(integration[:, 25:48]),
    )
    window_indices, terminal_hold = reference_window_index_arrays(boundaries)
    normalized = np.full((steps, 17), 0.1, dtype="<f4")
    physical = np.full((steps, 17), 0.04, dtype="<f4")
    offsets = np.ones(boundaries, dtype="<i8")
    offsets[0] = 0
    arrays: dict[str, np.ndarray] = {
        "boundary_integration_state": integration,
        "boundary_observation": observation,
        "boundary_cfrc_ext": cfrc,
        "boundary_root_xy": np.ascontiguousarray(integration[:, 1:3]),
        "boundary_simulation_time": times,
        "boundary_wrapper_elapsed": np.arange(boundaries, dtype="<i8"),
        "boundary_wrapper_flags": wrapper_flags,
        "boundary_result_flags": result_flags,
        "boundary_torso_up_z": np.ones(boundaries, dtype="<f8"),
        "reference_rows": reference,
        "reference_window_indices": window_indices,
        "reference_window_terminal_hold": terminal_hold,
        "boundary_rng_state_sha256": np.asarray([_DIGEST] * boundaries, dtype="|S64"),
        "transition_normalized_action": normalized,
        "transition_physical_action": physical,
        "transition_returned_observation": np.ascontiguousarray(observation[1:]),
        "transition_reward": np.zeros(steps, dtype="<f8"),
        "transition_flags": np.zeros((steps, 2), dtype="|u1"),
        "transition_next_wrapper_elapsed": np.arange(1, boundaries, dtype="<i8"),
        "transition_next_wrapper_flags": np.ascontiguousarray(wrapper_flags[1:]),
        "transition_actor_input_sha256": np.asarray(
            [array_sha256(np.ascontiguousarray(row, dtype="<f4")) for row in observation[:-1]],
            dtype="|S64",
        ),
        "transition_actor_output_sha256": np.asarray(
            [array_sha256(row) for row in normalized], dtype="|S64"
        ),
        "transition_physical_action_sha256": np.asarray(
            [array_sha256(row) for row in physical], dtype="|S64"
        ),
        "transition_returned_observation_sha256": np.asarray(
            [array_sha256(row) for row in observation[1:]], dtype="|S64"
        ),
        "transition_contact_substeps": np.full(steps, 5, dtype="<i8"),
        "transition_nonfoot_floor_contact": np.zeros(steps, dtype="|u1"),
        "contact_offsets": offsets,
        "contact_substep": np.asarray([0], dtype="<i8"),
        "contact_index_within_substep": np.asarray([0], dtype="<i8"),
        "contact_geom1_id": np.asarray([0], dtype="<i8"),
        "contact_geom2_id": np.asarray([5], dtype="<i8"),
        "contact_geom1_name": np.asarray([b"floor"], dtype="|S32"),
        "contact_geom2_name": np.asarray([b"left_foot"], dtype="|S32"),
        "contact_force_torque": np.asarray([[1, 2, 3, 4, 5, 6]], dtype="<f8"),
    }
    arrays["boundary_integration_sha256"] = np.asarray(
        [array_sha256(row) for row in integration], dtype="|S64"
    )
    arrays["boundary_observation_sha256"] = np.asarray(
        [array_sha256(row) for row in observation], dtype="|S64"
    )
    arrays["boundary_cfrc_ext_sha256"] = np.asarray(
        [array_sha256(row) for row in cfrc], dtype="|S64"
    )
    arrays["boundary_root_xy_sha256"] = np.asarray(
        [array_sha256(row) for row in arrays["boundary_root_xy"]], dtype="|S64"
    )
    arrays["boundary_reference_sha256"] = np.asarray(
        [array_sha256(row) for row in reference], dtype="|S64"
    )
    arrays["transition_contact_sequence_sha256"] = np.asarray(
        [contact_sequence_sha256(arrays, index) for index in range(steps)], dtype="|S64"
    )
    arrays["boundary_record_sha256"] = np.asarray(
        [boundary_record_sha256(arrays, index) for index in range(boundaries)], dtype="|S64"
    )
    arrays["transition_record_sha256"] = np.asarray(
        [transition_record_sha256(arrays, index) for index in range(steps)], dtype="|S64"
    )
    arrays.update(
        {
            "plain_boundary_integration_state": integration.copy(),
            "plain_boundary_observation": observation.copy(),
            "plain_boundary_cfrc_ext": cfrc.copy(),
            "plain_boundary_simulation_time": times.copy(),
            "plain_boundary_wrapper_elapsed": np.arange(boundaries, dtype="<i8"),
            "plain_boundary_wrapper_flags": wrapper_flags.copy(),
            "plain_boundary_result_flags": result_flags.copy(),
            "plain_transition_physical_action": physical.copy(),
            "plain_transition_reward": np.zeros(steps, dtype="<f8"),
        }
    )
    arrays["plain_comparison_boundary_sha256"] = np.asarray(
        [plain_comparison_boundary_sha256(arrays, index) for index in range(boundaries)],
        dtype="|S64",
    )
    arrays["plain_comparison_transition_sha256"] = np.asarray(
        [plain_comparison_transition_sha256(arrays, index) for index in range(steps)],
        dtype="|S64",
    )
    return validate_clip_arrays(arrays, steps=steps, plain_comparison=True)


def _runtime_identity(
    *,
    model_sha256: str = _DIGEST,
    model_byte_count: int = 1,
    lock_sha256: str = "b" * 64,
    collector_sha256: str = "c" * 64,
) -> dict[str, object]:
    return {
        "runtime_id": "humanoid_reference_corpus_runtime/v1",
        "environment_id": "Humanoid-v5",
        "environment_kwargs": {
            "exclude_current_positions_from_observation": True,
            "frame_skip": 5,
            "render_mode": None,
            "reset_noise_scale": 0.01,
            "terminate_when_unhealthy": False,
        },
        "environment_semantics": {
            "termination": "TimeLimit_only_at_1000",
            "healthy_z_range_open_m": [1.0, 2.0],
            "observation": "raw_Humanoid-v5_348d_excluding_root_xy",
            "action": "physical_17d_float32_in_closed_minus_0.4_to_0.4",
        },
        "wrapper_types": list(EXPECTED_REFERENCE_WRAPPER_TYPES),
        "time_limit_steps": 1000,
        "python_version": "3.13.11",
        "platform_system": "Darwin",
        "platform_release": "test",
        "platform_machine": "arm64",
        "numpy_version": "2.5.2",
        "torch_version": "2.14.0",
        "gymnasium_version": "1.3.0",
        "mujoco_version": "3.12.0",
        "model_sha256": model_sha256,
        "model_byte_count": model_byte_count,
        "uv_lock_sha256": lock_sha256,
        "observation_space_sha256": "d" * 64,
        "action_space_sha256": "e" * 64,
        "integration_state_spec": "mjtState.mjSTATE_INTEGRATION",
        "integration_state_flag": 16_383,
        "integration_state_size": 195,
        "integration_state_components": [list(item) for item in EXPECTED_STATE_COMPONENT_SIZES],
        "timestep_seconds": 0.003,
        "frame_skip": 5,
        "control_period_seconds": 0.015,
        "integrator": "mjINT_RK4",
        "solver": "mjSOL_PGS",
        "solver_iterations": 50,
        "contact_capture_id": FULL_CONTACT_CAPTURE_ID,
        "source_hashes": {"reference_corpus_collector": collector_sha256},
        "thread_settings": {"torch_num_threads": 1},
        "cpu_only": True,
        "deterministic_actor_mean": True,
        "same_host_determinism_scope": "same_host_same_process_settings_bitwise/v1",
    }


def _identity(**overrides: object) -> ReferenceIdentityV2:
    values: dict[str, object] = {
        "robot_id": ROBOT_ID,
        "reference_schema_id": REFERENCE_SCHEMA_ID,
        "reference_schema_sha256": "1" * 64,
        "reference_content_sha256": "2" * 64,
        "root_xy_sidecar_sha256": "3" * 64,
        "replay_bundle_core_sha256": "4" * 64,
        "actor_npz_sha256": "5" * 64,
        "actor_state_sha256": "6" * 64,
        "import_receipt_sha256": "7" * 64,
        "equivalence_receipt_sha256": "8" * 64,
        "generator_source_sha256": "9" * 64,
        "certifier_source_sha256": "a" * 64,
        "runtime_identity_sha256": "b" * 64,
        "seed": 120001,
        "n_boundaries": 1001,
        "cadence_seconds": CONTROL_PERIOD_SECONDS,
        "derivation_kind": "original",
        "parent_reference_identity_sha256": None,
        "parent_provenance": {"source": "unit-test"},
    }
    values.update(overrides)
    return ReferenceIdentityV2(**values)


def _certificate(*, steps: int = 8) -> dict[str, object]:
    return {
        "certificate_id": FULL_CLIP_CERTIFICATE_ID,
        "schema_version": 2,
        "clip_id": "synthetic",
        "bundle_manifest_sha256": "1" * 64,
        "reference_identity_sha256": "2" * 64,
        "payload_sha256": "3" * 64,
        "collector_pid": 10,
        "certifier_pid": 11,
        "steps_expected": steps,
        "transitions_verified": steps,
        "covered_transition_indices_sha256": transition_indices_sha256(steps),
        "verification_digest_sha256": "4" * 64,
        "replay_method_version": REPLAY_METHOD_VERSION,
        "all_hashes_verified": True,
        "all_transitions_passed": True,
        "first_failure": None,
        "evidence_class": "same_runtime_tier_d_replay",
        "claim_ceiling": "full_clip_same_host_replay_only",
    }


def _write_artifact_sources(directory: Path) -> dict[str, Path]:
    directory.mkdir(parents=True)
    roles = (
        "source_policy_bytes",
        "strict_actor_npz",
        "import_receipt",
        "equivalence_receipt",
        "mujoco_model_bytes",
        "dependency_lock",
        "collector_source",
        "certifier_source",
        "metric_source",
    )
    result = {}
    for role in roles:
        path = directory / f"{role}.bin"
        path.write_bytes(role.encode())
        result[role] = path
    return result


def _published_unit_bundle(tmp_path: Path) -> tuple[dict[str, object], Path]:
    sources = _write_artifact_sources(tmp_path / "sources")
    collector_sha = sha256_file(sources["collector_source"])
    runtime = _runtime_identity(
        model_sha256=sha256_file(sources["mujoco_model_bytes"]),
        model_byte_count=sources["mujoco_model_bytes"].stat().st_size,
        lock_sha256=sha256_file(sources["dependency_lock"]),
        collector_sha256=collector_sha,
    )
    actor_identity = {
        "variant": "expert",
        "npz_sha256": sha256_file(sources["strict_actor_npz"]),
        "actor_state_sha256": "5" * 64,
        "actor_schema_sha256": "6" * 64,
        "source_policy_sha256": sha256_file(sources["source_policy_bytes"]),
        "import_receipt_sha256": sha256_file(sources["import_receipt"]),
        "equivalence_receipt_sha256": sha256_file(sources["equivalence_receipt"]),
        "inference_id": INFERENCE_ID,
    }
    arrays = _synthetic_clip_arrays()
    clip = CollectedClip(
        clip_id="synthetic-8-expert",
        clip_kind="corpus",
        actor_variant="expert",
        seed=120001,
        reset_order=0,
        steps=8,
        arrays=arrays,
        runtime_identity=runtime,
        rng_state={"bit_generator": "PCG64", "state": {"state": 1, "inc": 3}},
        actor_identity=actor_identity,
        plain_comparison=plain_comparison_receipt(arrays, steps=8),
        collector_pid=10,
    )
    root = tmp_path / "published"
    published = publish_clip_bundle(clip, artifact_root=root, artifact_sources=sources)
    manifest = load_bundle_manifest(
        published.manifest_path,
        expected_sha256=published.manifest_sha256,
    )
    return manifest, root


def _rebind_core(manifest: dict[str, object]) -> dict[str, object]:
    manifest["core_sha256"] = sha256_json(manifest["core"])
    return manifest


def test_reference_identity_is_versioned_and_binds_every_required_domain() -> None:
    identity = _identity()

    payload = identity.to_dict()
    assert payload["identity_id"] == "humanoid_same_runtime_reference_identity/v2"
    assert payload["schema_version"] == 2
    assert len(identity.sha256) == 64
    assert reference_schema_payload()["root_xy_policy_visible"] is False
    assert reference_schema_payload()["horizon"] == 8


def test_runner_receipt_loader_requires_exactly_one_terminating_lf(tmp_path: Path) -> None:
    value = {"receipt_id": "synthetic"}
    canonical = canonical_json_bytes(value)
    cases = (
        ("one-lf.json", canonical + b"\n", True),
        ("no-lf.json", canonical, False),
        ("two-lfs.json", canonical + b"\n\n", False),
    )

    for filename, encoded, accepted in cases:
        path = tmp_path / filename
        path.write_bytes(encoded)
        expected_sha256 = hashlib.sha256(encoded).hexdigest()
        if accepted:
            assert (
                reference_corpus_runner._load_json(
                    path,
                    expected_sha256=expected_sha256,
                    require_terminating_lf=True,
                )
                == value
            )
        else:
            with pytest.raises(ValueError, match="JSON artifact is not canonical"):
                reference_corpus_runner._load_json(
                    path,
                    expected_sha256=expected_sha256,
                    require_terminating_lf=True,
                )


def test_crop_and_retime_cannot_retain_parent_identity_or_certificate() -> None:
    parent = _identity()
    crop = _identity(
        derivation_kind="crop",
        parent_reference_identity_sha256=parent.sha256,
        n_boundaries=500,
        reference_content_sha256="c" * 64,
    )
    with pytest.raises(ReferenceIdentityV2Error, match="parent certificate"):
        validate_reference_derivation(
            parent,
            crop,
            parent_certificate_sha256="d" * 64,
            child_certificate_sha256="d" * 64,
        )
    retime = _identity(
        derivation_kind="retime",
        parent_reference_identity_sha256=parent.sha256,
    )
    with pytest.raises(ReferenceIdentityV2Error, match="retime must change cadence"):
        validate_reference_derivation(
            parent,
            retime,
            parent_certificate_sha256="d" * 64,
            child_certificate_sha256="e" * 64,
        )


def test_frozen_corpus_and_e3_manifests_reject_replacement_or_omission() -> None:
    e3 = e3_manifest_payload()
    validate_e3_manifest(e3)
    changed_e3 = copy.deepcopy(e3)
    changed_e3["block_seeds_in_order"][0] += 1
    with pytest.raises(ReferenceIdentityV2Error, match="frozen design"):
        validate_e3_manifest(changed_e3)

    entries = [
        {
            "clip_id": f"corpus-{seed}-{actor}",
            "seed": seed,
            "actor_variant": actor,
            "reset_order": order,
            "bundle_manifest_sha256": "1" * 64,
            "reference_identity_sha256": "2" * 64,
        }
        for order, (seed, actor) in enumerate(
            pair for seed in CORPUS_SEEDS for pair in ((seed, actor) for actor in CORPUS_ACTORS)
        )
    ]
    manifest = corpus_manifest_payload(entries)
    validate_corpus_manifest(manifest)
    manifest["clips_in_reset_order"].pop()
    manifest["clip_count"] -= 1
    with pytest.raises(ReferenceIdentityV2Error, match="exactly 108"):
        validate_corpus_manifest(manifest)


@pytest.mark.parametrize(
    ("field", "index"),
    [
        ("boundary_root_xy", (0, 0)),
        ("boundary_integration_state", (0, 1)),
        ("boundary_observation", (0, 0)),
        ("boundary_cfrc_ext", (0, 0, 0)),
        ("transition_physical_action", (0, 0)),
        ("contact_force_torque", (0, 0)),
        ("contact_geom1_name", (0,)),
        ("boundary_wrapper_flags", (0, 0)),
        ("reference_rows", (0, 0)),
    ],
)
def test_exact_array_contract_rejects_independent_one_bit_flips(
    field: str,
    index: tuple[int, ...],
) -> None:
    arrays = {name: value.copy() for name, value in _synthetic_clip_arrays().items()}
    byte_view = (
        arrays[field].view(np.uint8).reshape((*arrays[field].shape, arrays[field].dtype.itemsize))
    )
    byte_view[(*index, 0)] ^= 1

    with pytest.raises(ReferenceCorpusContractError):
        validate_clip_arrays(arrays, steps=8, plain_comparison=True)


def test_reference_contract_rejects_reduced_state_and_abi_drift() -> None:
    with pytest.raises(ReferenceCorpusContractError):
        validate_clip_arrays(
            {
                "qpos": np.zeros((9, 24), dtype="<f8"),
                "qvel": np.zeros((9, 23), dtype="<f8"),
            },
            steps=8,
            plain_comparison=True,
        )

    mutations = []
    quaternion = {name: value.copy() for name, value in _synthetic_clip_arrays().items()}
    quaternion["reference_rows"][1, 1:5] *= -1.0
    mutations.append(quaternion)
    joint_order = {name: value.copy() for name, value in _synthetic_clip_arrays().items()}
    joint_order["reference_rows"][:, [11, 12]] = joint_order["reference_rows"][:, [12, 11]]
    joint_order["reference_rows"][0, 11] = 0.01
    mutations.append(joint_order)
    velocity_frame = {name: value.copy() for name, value in _synthetic_clip_arrays().items()}
    velocity_frame["reference_rows"][1, 8] = 0.01
    mutations.append(velocity_frame)
    cadence = {name: value.copy() for name, value in _synthetic_clip_arrays().items()}
    cadence["boundary_simulation_time"][1] += 0.001
    cadence["boundary_integration_state"][1, 0] += 0.001
    mutations.append(cadence)
    early_hold = {name: value.copy() for name, value in _synthetic_clip_arrays().items()}
    early_hold["reference_window_terminal_hold"][0, 0] = 1
    mutations.append(early_hold)

    for changed in mutations:
        with pytest.raises(ReferenceCorpusContractError):
            validate_clip_arrays(changed, steps=8, plain_comparison=True)

    qpos = np.ascontiguousarray(_synthetic_clip_arrays()["boundary_integration_state"][:, 1:25])
    qvel = np.ascontiguousarray(_synthetic_clip_arrays()["boundary_integration_state"][:, 25:48])
    with pytest.raises(ReferenceCorpusContractError, match="joint order"):
        derive_reference_rows(qpos, qvel, joint_order=("wrong",))
    with pytest.raises(ReferenceCorpusContractError, match="angular velocity frame"):
        derive_reference_rows(qpos, qvel, velocity_frame="world")
    with pytest.raises(ReferenceCorpusContractError, match="cadence"):
        derive_reference_rows(qpos, qvel, cadence_seconds=0.03)


def test_payload_round_trip_is_canonical_and_complete() -> None:
    arrays = _synthetic_clip_arrays()
    first = encode_clip_payload(arrays, steps=8, plain_comparison=True)
    second = encode_clip_payload(arrays, steps=8, plain_comparison=True)

    assert first == second
    restored = decode_clip_payload(first, steps=8, plain_comparison=True)
    assert set(restored) == set(arrays)
    assert array_bindings(restored) == array_bindings(arrays)


def test_strict_actor_runtime_leaves_process_global_torch_state_unchanged(
    tmp_path: Path,
) -> None:
    import torch

    actor_arrays = {
        member["name"]: np.zeros(member["shape"], dtype=member["dtype"])
        for member in actor_schema()["members_in_order"]
    }
    actor_arrays["action_low"][:] = -0.4
    actor_arrays["action_high"][:] = 0.4
    actor_arrays["format_version"][:] = 1
    actor_path = tmp_path / "zero_actor.npz"
    actor_sha = write_actor_npz_exclusive(actor_path, validate_actor_arrays(actor_arrays))
    before = (
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
        torch.are_deterministic_algorithms_enabled(),
    )

    StrictTQCActorRuntime.from_npz(actor_path, expected_sha256=actor_sha)

    after = (
        torch.get_num_threads(),
        torch.get_num_interop_threads(),
        torch.are_deterministic_algorithms_enabled(),
    )
    assert after == before


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("gymnasium_version", "1.3.1"),
        ("model_sha256", "f" * 64),
        ("timestep_seconds", 0.004),
        ("frame_skip", 4),
        ("wrapper_types", list(reversed(EXPECTED_REFERENCE_WRAPPER_TYPES))),
    ],
)
def test_runtime_contract_rejects_version_model_timing_and_wrapper_drift(
    field: str,
    changed: object,
) -> None:
    recorded = _runtime_identity()
    runtime = copy.deepcopy(recorded)
    runtime[field] = changed
    with pytest.raises(ReferenceCorpusContractError, match="runtime"):
        require_same_runtime(recorded, runtime)


def test_runtime_contract_rejects_termination_drift() -> None:
    runtime = _runtime_identity()
    runtime["environment_kwargs"]["terminate_when_unhealthy"] = True
    with pytest.raises(ReferenceCorpusContractError, match="runtime"):
        validate_runtime_identity(runtime)


def test_bundle_contract_rejects_minari_branch_metadata_missing_boundaries_and_actor_flip(
    tmp_path: Path,
) -> None:
    original, _root = _published_unit_bundle(tmp_path)
    validate_bundle_manifest(original)
    mutations = []
    minari = copy.deepcopy(original)
    minari["core"]["artifact_type"] = "minari_dataset"
    mutations.append(minari)
    branch = copy.deepcopy(original)
    branch["core"]["policy_visible_input"]["fields"].append("branch_id")
    mutations.append(branch)
    missing = copy.deepcopy(original)
    missing["core"]["boundaries"] -= 1
    mutations.append(missing)
    actor = copy.deepcopy(original)
    actor["core"]["source_actor"]["npz_sha256"] = "f" * 64
    mutations.append(actor)
    unknown_wrapper = copy.deepcopy(original)
    unknown_wrapper["core"]["wrapper_state_contract"]["unknown_fields_permitted"] = True
    mutations.append(unknown_wrapper)

    for changed in mutations:
        with pytest.raises(ReferenceCorpusContractError):
            validate_bundle_manifest(_rebind_core(changed))


def test_full_clip_certificate_rejects_partial_coverage_and_same_process() -> None:
    validate_full_clip_certificate(_certificate())
    partial = _certificate()
    partial["transitions_verified"] = 7
    with pytest.raises(ReferenceIdentityV2Error, match="fewer than all"):
        validate_full_clip_certificate(partial)
    same_process = _certificate()
    same_process["certifier_pid"] = same_process["collector_pid"]
    with pytest.raises(ReferenceIdentityV2Error, match="separate process"):
        validate_full_clip_certificate(same_process)


def _e3_branch(
    actor: str,
    actor_state: str,
    *,
    action_offset: float = 0.0,
    future_offset: float = 0.0,
) -> E3Branch:
    actions = np.full((8, 17), action_offset, dtype="<f4")
    rows = np.zeros((8, 45), dtype="<f8")
    rows[:, 1] = 1.0
    rows[1:, 0] = future_offset
    arrays: dict[str, np.ndarray] = {
        "transition_physical_action": actions,
        "reference_rows": rows,
        "boundary_integration_state": np.zeros((1, 195), dtype="<f8"),
        "boundary_observation": np.zeros((1, 348), dtype="<f8"),
        "boundary_cfrc_ext": np.zeros((1, 13, 6), dtype="<f8"),
        "boundary_root_xy": np.zeros((1, 2), dtype="<f8"),
        "boundary_simulation_time": np.zeros(1, dtype="<f8"),
        "boundary_wrapper_elapsed": np.zeros(1, dtype="<i8"),
        "boundary_wrapper_flags": np.zeros((1, 5), dtype="|u1"),
        "boundary_result_flags": np.zeros((1, 2), dtype="|u1"),
        "boundary_torso_up_z": np.ones(1, dtype="<f8"),
        "boundary_rng_state_sha256": np.asarray([_DIGEST], dtype="|S64"),
    }
    return E3Branch(
        actor_variant=actor,
        manifest_sha256=_DIGEST,
        manifest={
            "core": {
                "source_actor": {"actor_state_sha256": actor_state},
                "controller_state": {
                    "actor_recurrent_state": None,
                    "normalizer_state": None,
                    "residual_controller_state": None,
                },
            }
        },
        certificate_sha256=_DIGEST,
        certificate={},
        arrays=arrays,
    )


def test_e3_rejects_nonidentical_anchors_and_self_reference_only() -> None:
    expert = _e3_branch("expert", "1" * 64)
    medium = _e3_branch("medium", "2" * 64)
    simple = _e3_branch("simple", "3" * 64)
    medium.arrays["boundary_root_xy"][0, 0] = 1.0
    with pytest.raises(E3CertificationError, match="nonidentical"):
        require_same_state_anchors((expert, medium, simple))

    duplicate = _e3_branch("medium", "1" * 64)
    with pytest.raises(E3CertificationError, match="self-reference-only"):
        require_distinct_actor_states((expert, duplicate, simple))


def test_e3_rejects_identical_first_actions_and_weak_future_variation() -> None:
    statuses = {"expert": {"branch_passed": True}, "medium": {"branch_passed": True}}
    expert = _e3_branch("expert", "1" * 64)
    identical = _e3_branch("medium", "2" * 64)
    result = evaluate_e3_pair(expert, identical, statuses)
    assert result["pair_passed"] is False
    assert "identical_or_weak_first_action" in result["failure_reasons"]

    action_only = _e3_branch("medium", "2" * 64, action_offset=0.03)
    result = evaluate_e3_pair(expert, action_only, statuses)
    assert result["first_action_passed"] is True
    assert result["first_eight_actions_passed"] is True
    assert result["future_variation_passed"] is False
    assert "weak_future_reference_variation" in result["failure_reasons"]


class _RewardGuard(dict[str, np.ndarray]):
    def __getitem__(self, key: str) -> np.ndarray:
        if key == "transition_reward":
            raise AssertionError("development screen read reward")
        return super().__getitem__(key)


def test_development_screen_episode_metrics_never_read_reward() -> None:
    times = np.arange(1001, dtype="<f8") * CONTROL_PERIOD_SECONDS
    root_xy = np.zeros((1001, 2), dtype="<f8")
    root_xy[:, 0] = np.linspace(0.0, 10.0, 1001)
    rows = np.zeros((1001, 45), dtype="<f8")
    rows[:, 0] = 1.4
    arrays = _RewardGuard(
        {
            "boundary_root_xy": root_xy,
            "reference_rows": rows,
            "boundary_simulation_time": times,
            "boundary_torso_up_z": np.ones(1001, dtype="<f8"),
            "transition_normalized_action": np.zeros((1000, 17), dtype="<f4"),
            "transition_nonfoot_floor_contact": np.zeros(1000, dtype="|u1"),
            "transition_reward": np.zeros(1000, dtype="<f8"),
        }
    )

    episode = development_episode_from_arrays(seed=96001, arrays=arrays)
    assert episode.full_horizon_healthy is True
    assert episode.full_horizon_upright is True
    assert episode.net_forward_displacement_m == pytest.approx(10.0)


def _frozen_expert() -> tuple[Path, dict[str, object]]:
    identity = reference_corpus_runner._actor_identity("expert")
    path = (
        reference_corpus_runner.PROJECT_ROOT
        / reference_corpus_runner.ACTOR_REGISTRY["expert"]["npz"]
    )
    return path, identity


@pytest.mark.gym
def test_production_collector_path_is_plain_runtime_bitwise_for_1000_steps() -> None:
    actor_path, identity = _frozen_expert()

    clip = collect_reference_clip(
        clip_id="production-equivalence-96001-expert",
        clip_kind="corpus",
        actor_variant="expert",
        actor_npz_path=actor_path,
        actor_identity=identity,
        seed=96001,
        reset_order=0,
        steps=1000,
        compare_plain_runtime=True,
    )

    assert clip.plain_comparison["status"] == "passed"
    assert clip.plain_comparison["steps_compared"] == 1000
    assert clip.plain_comparison["boundaries_compared"] == 1001
    for hashes in clip.plain_comparison["field_hashes"].values():
        assert hashes["instrumented_sha256"] == hashes["plain_sha256"]
    assert (
        clip.arrays["boundary_observation"][1:].tobytes()
        == clip.arrays["transition_returned_observation"].tobytes()
    )


@pytest.mark.gym
def test_post_step_forward_seam_exposes_boundary_one_and_reward_two() -> None:
    import mujoco

    actor_path, identity = _frozen_expert()

    def inject_forward(environment: object, transition: int) -> None:
        if transition == 0:
            physical = environment.unwrapped
            mujoco.mj_forward(physical.model, physical.data)

    with pytest.raises(
        ReferenceCorpusContractError,
        match="boundary observation 1 differs from live observation cache",
    ):
        collect_reference_clip(
            clip_id="negative-forward-96001-expert",
            clip_kind="corpus",
            actor_variant="expert",
            actor_npz_path=actor_path,
            actor_identity=identity,
            seed=96001,
            reset_order=0,
            steps=2,
            compare_plain_runtime=True,
            _post_step_test_hook=inject_forward,
        )

    actor = StrictTQCActorRuntime.from_npz(
        actor_path,
        expected_sha256=identity["npz_sha256"],
    )
    plain = make_humanoid_env()
    instrumented = make_reference_corpus_env()
    try:
        _plain_observation, _ = plain.reset(seed=96001)
        instrumented_observation, _ = instrumented.reset(seed=96001)
        action0 = actor.act(instrumented_observation).physical_action
        plain_step1 = plain.step(action0.copy())
        instrumented_step1 = instrumented.step(action0.copy())
        assert np.array_equal(plain_step1[0], instrumented_step1[0])
        mujoco.mj_forward(instrumented.unwrapped.model, instrumented.unwrapped.data)
        assert not np.array_equal(
            instrumented_step1[0],
            instrumented.unwrapped._get_obs(),
        )
        action1 = actor.act(instrumented_step1[0]).physical_action
        plain_step2 = plain.step(action1.copy())
        instrumented_step2 = instrumented.step(action1.copy())
        assert struct.pack(">d", plain_step2[1]) != struct.pack(">d", instrumented_step2[1])
    finally:
        plain.close()
        instrumented.close()


def test_live_capture_modules_do_not_call_state_touching_mujoco_helpers() -> None:
    modules = (
        reference_corpus_collector,
        reference_corpus_certifier,
        __import__(
            "oracle_composition.experiments.expert_development_screen",
            fromlist=["unused"],
        ),
    )
    for module in modules:
        source = Path(str(module.__file__)).read_text(encoding="utf-8")
        for forbidden in ("mj_forward(", "mj_step1(", "mj_kinematics("):
            assert forbidden not in source


@pytest.mark.gym
def test_short_clip_collect_publish_and_separate_process_certify(tmp_path: Path) -> None:
    actor_arrays = {
        member["name"]: np.zeros(member["shape"], dtype=member["dtype"])
        for member in actor_schema()["members_in_order"]
    }
    actor_arrays["action_low"][:] = -0.4
    actor_arrays["action_high"][:] = 0.4
    actor_arrays["format_version"][:] = 1
    validated_actor = validate_actor_arrays(actor_arrays)
    actor_path = tmp_path / "zero_actor.npz"
    actor_sha = write_actor_npz_exclusive(actor_path, validated_actor)
    source_policy = tmp_path / "source_policy.pth"
    import_receipt = tmp_path / "import.json"
    equivalence_receipt = tmp_path / "equivalence.json"
    source_policy.write_bytes(b"synthetic-source-policy")
    import_receipt.write_bytes(b"synthetic-import-receipt")
    equivalence_receipt.write_bytes(b"synthetic-equivalence-receipt")
    actor_identity = {
        "variant": "expert",
        "npz_sha256": actor_sha,
        "actor_state_sha256": actor_state_sha256(validated_actor),
        "actor_schema_sha256": actor_schema_sha256(),
        "source_policy_sha256": sha256_file(source_policy),
        "import_receipt_sha256": sha256_file(import_receipt),
        "equivalence_receipt_sha256": sha256_file(equivalence_receipt),
        "inference_id": INFERENCE_ID,
    }
    environment = make_reference_corpus_env()
    try:
        model_path = Path(str(environment.unwrapped.fullpath))
    finally:
        environment.close()
    clip = collect_reference_clip(
        clip_id="short-synthetic-expert",
        clip_kind="corpus",
        actor_variant="expert",
        actor_npz_path=actor_path,
        actor_identity=actor_identity,
        seed=7,
        reset_order=0,
        steps=3,
    )
    root = tmp_path / "replay"
    sources = {
        "source_policy_bytes": source_policy,
        "strict_actor_npz": actor_path,
        "import_receipt": import_receipt,
        "equivalence_receipt": equivalence_receipt,
        "mujoco_model_bytes": model_path,
        "dependency_lock": dependency_lock_path(),
        "collector_source": Path(str(reference_corpus_collector.__file__)),
        "certifier_source": Path(str(reference_corpus_certifier.__file__)),
        "metric_source": Path(str(tqc_development_metrics.__file__)),
    }
    bundle = publish_clip_bundle(clip, artifact_root=root, artifact_sources=sources)
    request = {
        "request_id": "reference_corpus_tier_d_certifier_request/v2",
        "replay_method_version": REPLAY_METHOD_VERSION,
        "artifact_root": root.as_posix(),
        "bundles": [
            {
                "manifest_path": bundle.manifest_path.as_posix(),
                "manifest_sha256": bundle.manifest_sha256,
            }
        ],
    }
    request_path = root / "request.json"
    request_path.write_bytes(canonical_json_bytes(request))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "oracle_composition.experiments.reference_corpus_certifier",
            "--request",
            str(request_path),
        ],
        check=False,
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    assert response["certifier_pid"] != clip.collector_pid
    assert response["certificate_count"] == 1
    item = response["certificates"][0]
    certificate_bytes = Path(item["certificate_path"]).read_bytes()
    assert hashlib.sha256(certificate_bytes).hexdigest() == item["certificate_sha256"]
    certificate = validate_full_clip_certificate(json.loads(certificate_bytes))
    assert certificate["transitions_verified"] == 3
    assert certificate["replay_method_version"] == REPLAY_METHOD_VERSION


@pytest.mark.gym
def test_midclip_replay_requires_predecessor_transition_for_reward(
    tmp_path: Path,
) -> None:
    actor_path, identity = _frozen_expert()
    collected = collect_reference_clip(
        clip_id="midclip-replay-96001-expert",
        clip_kind="corpus",
        actor_variant="expert",
        actor_npz_path=actor_path,
        actor_identity=identity,
        seed=96001,
        reset_order=0,
        steps=5,
        compare_plain_runtime=True,
    )
    clip = replace(collected, collector_pid=collected.collector_pid + 100_000)
    model_path = reference_corpus_runner._model_path()
    root = tmp_path / "midclip-replay"
    bundle = publish_clip_bundle(
        clip,
        artifact_root=root,
        artifact_sources=reference_corpus_runner._artifact_sources(
            "expert",
            model_path=model_path,
        ),
    )

    certificate = reference_corpus_certifier.certify_bundle(
        artifact_root=root,
        manifest_path=bundle.manifest_path,
        manifest_sha256=bundle.manifest_sha256,
    )
    assert certificate.transitions_verified == 5

    with pytest.raises(
        reference_corpus_certifier.FullClipReplayMismatch,
        match="transition 3: reward canary differs",
    ):
        reference_corpus_certifier.certify_bundle(
            artifact_root=root,
            manifest_path=bundle.manifest_path,
            manifest_sha256=bundle.manifest_sha256,
            _skip_predecessor_transition_for_test=3,
        )


@pytest.mark.gym
def test_unknown_wrapper_state_is_not_restorable() -> None:
    environment = make_reference_corpus_env()
    try:
        environment.reset(seed=1)
        with pytest.raises(ReferenceCorpusContractError, match="unknown wrapper"):
            restore_wrapper_state(environment, 0, np.zeros(6, dtype="|u1"))
    finally:
        environment.close()
