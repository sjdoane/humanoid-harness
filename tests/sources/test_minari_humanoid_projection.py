from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from oracle_composition.sources import minari_humanoid as minari_source
from oracle_composition.sources.minari_humanoid import (
    HUMANOID_V5_OBSERVATION_TO_REFERENCE_INDICES,
    MINARI_HUMANOID_DATASET_ID,
    REGISTERED_MINARI_HUMANOID_SOURCE_MANIFEST_BYTES,
    REGISTERED_MINARI_HUMANOID_SOURCE_MANIFEST_SHA256,
    MinariHumanoidImportError,
    import_minari_humanoid_episode,
    import_registered_minari_humanoid_episode,
)

SOURCE_COMMIT = "a" * 40
EPISODE_STEPS = 3
EPISODE_SEED = 123


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _space(*, dtype: str, width: int, low: float, high: float) -> str:
    return json.dumps(
        {
            "type": "Box",
            "dtype": dtype,
            "shape": [width],
            "low": [low] * width,
            "high": [high] * width,
        }
    )


def _metadata() -> dict[str, object]:
    return {
        "total_episodes": 1,
        "total_steps": EPISODE_STEPS,
        "data_format": "hdf5",
        "observation_space": _space(dtype="float64", width=348, low=-math_inf(), high=math_inf()),
        "action_space": _space(
            dtype="float32",
            width=17,
            low=float(np.float32(-0.4)),
            high=float(np.float32(0.4)),
        ),
        "env_spec": json.dumps(
            {
                "id": "Humanoid-v5",
                "entry_point": "gymnasium.envs.mujoco.humanoid_v5:HumanoidEnv",
                "reward_threshold": None,
                "nondeterministic": False,
                "max_episode_steps": 1000,
                "order_enforce": True,
                "disable_env_checker": False,
                "kwargs": {},
                "additional_wrappers": [],
                "vector_entry_point": None,
            }
        ),
        "dataset_size": 1.0,
        "dataset_id": MINARI_HUMANOID_DATASET_ID,
        "code_permalink": (
            "https://github.com/Farama-Foundation/minari-dataset-generation-scripts"
        ),
        "author": ["Synthetic Fixture"],
        "author_email": ["fixture@example.invalid"],
        "algorithm_name": "fixture",
        "description": "Synthetic validation fixture; not research evidence.",
        "minari_version": "0.5.2",
        "requirements": ["mujoco==3.2.3", "gymnasium>=1.0.0"],
    }


def math_inf() -> float:
    return float("inf")


def _observations() -> np.ndarray:
    values = np.zeros((EPISODE_STEPS + 1, 348), dtype=np.float64)
    for frame in range(EPISODE_STEPS + 1):
        values[frame, 0] = 1.4 + 0.01 * frame
        values[frame, 1] = 1.0
        values[frame, 5:22] = 0.001 * frame + np.arange(17) * 0.01
        values[frame, 22:45] = 0.01 * frame + np.arange(23) * 0.001
        values[frame, 45:] = frame + np.arange(303) * 0.0001
    return values


def _write_hdf5(path: Path, *, mutate: Any = None) -> np.ndarray:
    observations = _observations()
    actions = np.zeros((EPISODE_STEPS, 17), dtype=np.float32)
    rewards = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
    terminations = np.zeros(EPISODE_STEPS, dtype=np.bool_)
    truncations = np.asarray([False, False, True], dtype=np.bool_)
    with h5py.File(path, "w") as store:
        episode = store.create_group("episode_0")
        episode.attrs.update(
            {
                "id": 0,
                "seed": EPISODE_SEED,
                "total_steps": EPISODE_STEPS,
                "rewards_max": float(np.max(rewards)),
                "rewards_mean": float(np.mean(rewards)),
                "rewards_min": float(np.min(rewards)),
                "rewards_std": float(np.std(rewards)),
                "rewards_sum": float(np.sum(rewards)),
            }
        )
        episode.create_dataset("observations", data=observations)
        episode.create_dataset("actions", data=actions)
        episode.create_dataset("rewards", data=rewards)
        episode.create_dataset("terminations", data=terminations)
        episode.create_dataset("truncations", data=truncations)
        episode.create_group("infos")
        if mutate is not None:
            mutate(episode)
    return observations


def _fixture(tmp_path: Path, *, hdf_mutate: Any = None, metadata_mutate: Any = None):
    hdf5_path = tmp_path / "main_data.hdf5"
    metadata_path = tmp_path / "metadata.json"
    observations = _write_hdf5(hdf5_path, mutate=hdf_mutate)
    metadata = _metadata()
    if metadata_mutate is not None:
        metadata_mutate(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    kwargs = {
        "hdf5_path": hdf5_path,
        "metadata_path": metadata_path,
        "expected_hdf5_sha256": _sha256(hdf5_path),
        "expected_metadata_sha256": _sha256(metadata_path),
        "source_commit": SOURCE_COMMIT,
        "episode_id": 0,
        "expected_seed": EPISODE_SEED,
        "expected_total_steps": EPISODE_STEPS,
        "expected_final_terminated": False,
        "expected_final_truncated": True,
    }
    return observations, kwargs


def test_projects_exact_45d_order_and_emits_non_admitted_tier_k_receipt(
    tmp_path: Path,
) -> None:
    observations, kwargs = _fixture(tmp_path)

    result = import_minari_humanoid_episode(**kwargs)
    result.verify()

    expected = observations[:, HUMANOID_V5_OBSERVATION_TO_REFERENCE_INDICES]
    assert np.array_equal(result.episode_observations, observations)
    assert result.episode_observations.dtype.str == "<f8"
    assert result.episode_observations.flags.c_contiguous
    assert not result.episode_observations.flags.writeable
    with pytest.raises(ValueError):
        result.episode_observations.setflags(write=True)
    assert np.array_equal(np.asarray(result.reference.values), expected)
    assert result.reference.identity.n_frames == EPISODE_STEPS + 1
    receipt = result.receipt.to_dict()
    assert receipt["evidence_tier"] == "Tier-K"
    assert receipt["admission_status"] == "not_admitted"
    assert receipt["dynamics_feasibility_established"] is False
    assert receipt["tier_d_certificate_sha256"] is None
    assert receipt["source_provenance_class"] == "caller_pinned_unverified"
    assert receipt["source_origin_verified"] is False
    assert receipt["asserted_source_commit"] == SOURCE_COMMIT
    assert receipt["source_record_sha256"] is None
    assert receipt["source_record_bytes"] is None
    assert "source_commit" not in receipt
    assert "source_repository" not in receipt
    assert "generation_code_permalink" not in receipt
    assert receipt["metadata_declared_generation_code_url"].endswith(
        "minari-dataset-generation-scripts"
    )
    assert receipt["root_position_xy_evidence"] == "absent_from_observation_and_empty_infos"
    assert receipt["root_position_xy_reconstructed"] is False
    assert receipt["projection_indices"] == list(HUMANOID_V5_OBSERVATION_TO_REFERENCE_INDICES)
    assert receipt["reference_content_sha256"] == result.reference.identity.content_sha256
    assert len(result.receipt.sha256) == 64


def test_projection_verify_rejects_mutable_or_receipt_mismatched_observations(
    tmp_path: Path,
) -> None:
    _observations_value, kwargs = _fixture(tmp_path)
    result = import_minari_humanoid_episode(**kwargs)

    with pytest.raises(MinariHumanoidImportError, match="immutable projection contract"):
        replace(result, episode_observations=result.episode_observations.copy()).verify()

    tampered_values = result.episode_observations.copy()
    tampered_values[0, 45] += 1.0
    tampered = np.frombuffer(tampered_values.tobytes(order="C"), dtype="<f8").reshape(
        tampered_values.shape
    )
    with pytest.raises(MinariHumanoidImportError, match="observation hash mismatch"):
        replace(result, episode_observations=tampered).verify()


def test_projection_verify_rejects_reference_identity_receipt_mismatch(tmp_path: Path) -> None:
    _observations_value, kwargs = _fixture(tmp_path)
    result = import_minari_humanoid_episode(**kwargs)
    mismatched_receipt = replace(result.receipt, reference_content_sha256="0" * 64)

    with pytest.raises(MinariHumanoidImportError, match="reference identity"):
        replace(result, receipt=mismatched_receipt).verify()


def test_mapping_swaps_only_abdomen_y_z_positions_and_velocities() -> None:
    mapping = HUMANOID_V5_OBSERVATION_TO_REFERENCE_INDICES

    assert len(mapping) == 45
    assert len(set(mapping)) == 45
    assert mapping[:11] == (0, 1, 2, 3, 4, 22, 23, 24, 25, 26, 27)
    assert mapping[11:14] == (6, 5, 7)
    assert mapping[28:31] == (29, 28, 30)


def test_registered_source_manifest_is_exact_and_content_addressed() -> None:
    manifest_path = (
        Path(minari_source.__file__).parent / "manifests" / "minari_humanoid_expert_v0.json"
    )
    raw = manifest_path.read_bytes()
    record = json.loads(raw)

    assert len(raw) == REGISTERED_MINARI_HUMANOID_SOURCE_MANIFEST_BYTES
    assert hashlib.sha256(raw).hexdigest() == REGISTERED_MINARI_HUMANOID_SOURCE_MANIFEST_SHA256
    assert record["source_commit"] == "8e62dc7f7fcb4a19f8f869c65402d4bb60049117"
    assert record["files"]["hdf5"] == {
        "repository_path": "humanoid/expert-v0/data/main_data.hdf5",
        "source_url": (
            "https://huggingface.co/datasets/farama-minari/mujoco/resolve/"
            "8e62dc7f7fcb4a19f8f869c65402d4bb60049117/"
            "humanoid/expert-v0/data/main_data.hdf5"
        ),
        "bytes": 2_946_805_796,
        "sha256": "8253be693f06aeeac3cb62eeb349ad02d4ca0bcad02685b4b8ed390798d9aa1e",
    }
    assert record["files"]["metadata"]["bytes"] == 9_305
    assert record["files"]["metadata"]["sha256"] == (
        "2eb6e0ba388ceabef5eec1dea7e401e62391d856cf42b394c262db7c21366024"
    )


def test_registered_import_rejects_files_outside_fixed_source_record(tmp_path: Path) -> None:
    _observations_value, kwargs = _fixture(tmp_path)

    with pytest.raises(MinariHumanoidImportError, match="byte size mismatch"):
        import_registered_minari_humanoid_episode(
            hdf5_path=kwargs["hdf5_path"],
            metadata_path=kwargs["metadata_path"],
            episode_id=kwargs["episode_id"],
            expected_seed=kwargs["expected_seed"],
            expected_total_steps=kwargs["expected_total_steps"],
            expected_final_terminated=kwargs["expected_final_terminated"],
            expected_final_truncated=kwargs["expected_final_truncated"],
        )


def test_registered_import_rejects_modified_package_source_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "source-record.json"
    registered_path = (
        Path(minari_source.__file__).parent / "manifests" / "minari_humanoid_expert_v0.json"
    )
    raw = registered_path.read_bytes()
    manifest_path.write_bytes(raw.replace(b'"schema_version": 1', b'"schema_version": 2', 1))
    monkeypatch.setattr(
        minari_source,
        "_REGISTERED_MINARI_HUMANOID_SOURCE_MANIFEST",
        manifest_path,
    )

    with pytest.raises(MinariHumanoidImportError, match="SHA-256 mismatch"):
        import_registered_minari_humanoid_episode(
            hdf5_path=tmp_path / "unused.hdf5",
            metadata_path=tmp_path / "unused.json",
            episode_id=0,
            expected_seed=123,
            expected_total_steps=1_000,
            expected_final_terminated=False,
            expected_final_truncated=True,
        )


@pytest.mark.parametrize("field", ["expected_hdf5_sha256", "expected_metadata_sha256"])
def test_rejects_any_caller_pinned_hash_mismatch(tmp_path: Path, field: str) -> None:
    _observations_value, kwargs = _fixture(tmp_path)
    kwargs[field] = "0" * 64

    with pytest.raises(MinariHumanoidImportError, match="SHA-256 mismatch"):
        import_minari_humanoid_episode(**kwargs)


@pytest.mark.parametrize("source_commit", ["a" * 39, "A" * 40, "main", "0" * 64])
def test_requires_exact_pinned_source_commit(tmp_path: Path, source_commit: str) -> None:
    _observations_value, kwargs = _fixture(tmp_path)
    kwargs["source_commit"] = source_commit

    with pytest.raises(MinariHumanoidImportError, match="pinned lowercase"):
        import_minari_humanoid_episode(**kwargs)


def test_rejects_metadata_environment_drift_even_when_hash_matches(tmp_path: Path) -> None:
    def mutate(metadata: dict[str, object]) -> None:
        env_spec = json.loads(str(metadata["env_spec"]))
        env_spec["max_episode_steps"] = 999
        metadata["env_spec"] = json.dumps(env_spec)

    _observations_value, kwargs = _fixture(tmp_path, metadata_mutate=mutate)

    with pytest.raises(MinariHumanoidImportError, match="env_spec mismatch"):
        import_minari_humanoid_episode(**kwargs)


def test_rejects_oversized_metadata_number_with_contract_error(tmp_path: Path) -> None:
    def mutate(metadata: dict[str, object]) -> None:
        metadata["dataset_size"] = 10**400

    _observations_value, kwargs = _fixture(tmp_path, metadata_mutate=mutate)

    with pytest.raises(MinariHumanoidImportError, match="dataset_size must be finite"):
        import_minari_humanoid_episode(**kwargs)


def test_rejects_oversized_nested_box_bound_with_contract_error(tmp_path: Path) -> None:
    def mutate(metadata: dict[str, object]) -> None:
        action_space = json.loads(str(metadata["action_space"]))
        action_space["low"][0] = 10**400
        metadata["action_space"] = json.dumps(action_space)

    _observations_value, kwargs = _fixture(tmp_path, metadata_mutate=mutate)

    with pytest.raises(MinariHumanoidImportError, match="bounds must be numeric"):
        import_minari_humanoid_episode(**kwargs)


def test_rejects_nonempty_infos_instead_of_claiming_root_xy_absent(tmp_path: Path) -> None:
    def mutate(episode: Any) -> None:
        episode["infos"].create_dataset("x_position", data=np.zeros(EPISODE_STEPS + 1))

    _observations_value, kwargs = _fixture(tmp_path, hdf_mutate=mutate)

    with pytest.raises(MinariHumanoidImportError, match="infos must be empty"):
        import_minari_humanoid_episode(**kwargs)


def test_rejects_episode_shape_or_dtype_drift(tmp_path: Path) -> None:
    def mutate(episode: Any) -> None:
        del episode["actions"]
        episode.create_dataset("actions", data=np.zeros((EPISODE_STEPS, 17), dtype=np.float64))

    _observations_value, kwargs = _fixture(tmp_path, hdf_mutate=mutate)

    with pytest.raises(MinariHumanoidImportError, match="actions shape/dtype mismatch"):
        import_minari_humanoid_episode(**kwargs)


def test_rejects_filter_bearing_hdf5_dataset_before_read(tmp_path: Path) -> None:
    def mutate(episode: Any) -> None:
        values = np.asarray(episode["actions"])
        del episode["actions"]
        episode.create_dataset("actions", data=values, compression="gzip")

    _observations_value, kwargs = _fixture(tmp_path, hdf_mutate=mutate)

    with pytest.raises(MinariHumanoidImportError, match="filters are forbidden"):
        import_minari_humanoid_episode(**kwargs)


def test_rejects_external_storage_hdf5_dataset_before_read(tmp_path: Path) -> None:
    external_path = tmp_path / "actions.raw"

    def mutate(episode: Any) -> None:
        values = np.asarray(episode["actions"])
        del episode["actions"]
        dataset = episode.create_dataset(
            "actions",
            shape=values.shape,
            dtype=values.dtype,
            external=[(str(external_path), 0, h5py.h5f.UNLIMITED)],
        )
        dataset[:] = values

    _observations_value, kwargs = _fixture(tmp_path, hdf_mutate=mutate)

    with pytest.raises(MinariHumanoidImportError, match="external storage is forbidden"):
        import_minari_humanoid_episode(**kwargs)


def test_rejects_virtual_hdf5_dataset_before_read(tmp_path: Path) -> None:
    source_path = tmp_path / "virtual-source.hdf5"
    with h5py.File(source_path, "w") as source:
        source.create_dataset(
            "actions",
            data=np.zeros((EPISODE_STEPS, 17), dtype=np.float32),
        )

    def mutate(episode: Any) -> None:
        del episode["actions"]
        layout = h5py.VirtualLayout(shape=(EPISODE_STEPS, 17), dtype=np.float32)
        layout[:] = h5py.VirtualSource(
            str(source_path),
            "actions",
            shape=(EPISODE_STEPS, 17),
        )
        episode.create_virtual_dataset("actions", layout)

    _observations_value, kwargs = _fixture(tmp_path, hdf_mutate=mutate)

    with pytest.raises(MinariHumanoidImportError, match="virtual storage is forbidden"):
        import_minari_humanoid_episode(**kwargs)


def test_rejects_action_outside_exact_humanoid_bounds(tmp_path: Path) -> None:
    def mutate(episode: Any) -> None:
        episode["actions"][0, 0] = np.float32(0.41)

    _observations_value, kwargs = _fixture(tmp_path, hdf_mutate=mutate)

    with pytest.raises(MinariHumanoidImportError, match="actions exceeds"):
        import_minari_humanoid_episode(**kwargs)


def test_rejects_time_invariant_episode(tmp_path: Path) -> None:
    def mutate(episode: Any) -> None:
        episode["observations"][1:] = episode["observations"][0]

    _observations_value, kwargs = _fixture(tmp_path, hdf_mutate=mutate)

    with pytest.raises(MinariHumanoidImportError, match="time-invariant"):
        import_minari_humanoid_episode(**kwargs)


def test_rejects_wrong_final_flag_contract(tmp_path: Path) -> None:
    _observations_value, kwargs = _fixture(tmp_path)
    kwargs["expected_final_terminated"] = True
    kwargs["expected_final_truncated"] = False

    with pytest.raises(MinariHumanoidImportError, match="termination flag mismatch"):
        import_minari_humanoid_episode(**kwargs)


def test_rejects_linked_source_files(tmp_path: Path) -> None:
    _observations_value, kwargs = _fixture(tmp_path)
    link = tmp_path / "linked.hdf5"
    link.symlink_to(kwargs["hdf5_path"])
    kwargs["hdf5_path"] = link

    with pytest.raises(MinariHumanoidImportError, match="symbolic link"):
        import_minari_humanoid_episode(**kwargs)


def test_verified_file_rejects_same_size_rewrite_with_restored_mtime(tmp_path: Path) -> None:
    path = tmp_path / "pinned.bin"
    path.write_bytes(b"original")
    before = path.stat()

    with (
        pytest.raises(MinariHumanoidImportError, match="changed"),
        minari_source._verified_binary_file(
            path,
            expected_sha256=_sha256(path),
            expected_bytes=8,
            max_bytes=8,
            field="test file",
        ) as (stream, _byte_count),
    ):
        assert stream.read() == b"original"
        path.write_bytes(b"modified")
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def test_verified_file_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "source.fifo"
    os.mkfifo(fifo)

    with (
        pytest.raises(MinariHumanoidImportError, match="regular file"),
        minari_source._verified_binary_file(
            fifo,
            expected_sha256="0" * 64,
            max_bytes=8,
            field="test file",
        ),
    ):
        pytest.fail("FIFO must be rejected before read")


def test_exact_stream_hash_rejects_short_and_growing_inputs() -> None:
    with pytest.raises(MinariHumanoidImportError, match="grew"):
        minari_source._hash_exact_stream(
            io.BytesIO(b"ab"),
            byte_count=1,
            field="test file",
            phase="test verification",
        )
    with pytest.raises(MinariHumanoidImportError, match="shorter"):
        minari_source._hash_exact_stream(
            io.BytesIO(b"a"),
            byte_count=2,
            field="test file",
            phase="test verification",
        )
    assert (
        minari_source._hash_exact_stream(
            io.BytesIO(b"ab"),
            byte_count=2,
            field="test file",
            phase="test verification",
        )
        == hashlib.sha256(b"ab").hexdigest()
    )
