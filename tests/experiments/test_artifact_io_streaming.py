from __future__ import annotations

import hashlib
import io
import os
import stat
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.experiments import artifact_io
from oracle_composition.experiments.artifact_io import (
    reserve_streaming_artifact,
    verified_artifact_reader,
)
from oracle_composition.experiments.fixed_reference import ExperimentContractError


def test_streaming_publication_is_bounded_hashed_and_readable(tmp_path: Path) -> None:
    destination = tmp_path.resolve() / "evidence" / "replay.pkl"
    payload = (b"first-block" * 2000) + (b"second-block" * 3000)
    reservation = reserve_streaming_artifact(
        destination,
        max_bytes=len(payload),
        buffer_size=257,
    )

    assert isinstance(reservation.writer, io.BufferedWriter)
    for offset in range(0, len(payload), 113):
        reservation.writer.write(payload[offset : offset + 113])
    published = reservation.finalize()

    expected_sha256 = hashlib.sha256(payload).hexdigest()
    assert destination.read_bytes() == payload
    assert published.path == destination
    assert published.byte_count == len(payload)
    assert published.sha256 == expected_sha256
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert not list(destination.parent.glob(".*.partial"))

    with verified_artifact_reader(
        destination,
        expected_sha256=expected_sha256,
        expected_size=len(payload),
        max_bytes=len(payload),
        buffer_size=311,
    ) as reader:
        assert reader.read() == payload


def test_streaming_publication_rejects_limit_and_retains_partial(tmp_path: Path) -> None:
    destination = tmp_path.resolve() / "replay.pkl"
    reservation = reserve_streaming_artifact(destination, max_bytes=4, buffer_size=1)

    with pytest.raises(ExperimentContractError, match="byte limit"):
        reservation.writer.write(b"12345")
    reservation.close()

    assert not destination.exists()
    assert reservation.partial_path.exists()
    assert reservation.retained_path_after_failure == reservation.partial_path
    assert reservation.partial_path.stat().st_size <= 4


def test_streaming_close_retains_interrupted_partial(tmp_path: Path) -> None:
    destination = tmp_path.resolve() / "replay.pkl"
    reservation = reserve_streaming_artifact(destination, max_bytes=100)
    reservation.writer.write(b"partial-evidence")

    reservation.close()

    assert not destination.exists()
    assert reservation.partial_path.read_bytes() == b"partial-evidence"


def test_streaming_publication_rejects_destination_race(tmp_path: Path) -> None:
    destination = tmp_path.resolve() / "replay.pkl"
    reservation = reserve_streaming_artifact(destination, max_bytes=100)
    reservation.writer.write(b"candidate")
    destination.write_bytes(b"other-run")

    with pytest.raises(ExperimentContractError, match="refusing to overwrite"):
        reservation.finalize()

    assert destination.read_bytes() == b"other-run"
    assert reservation.partial_path.read_bytes() == b"candidate"


def test_streaming_publication_rejects_partial_path_substitution(tmp_path: Path) -> None:
    destination = tmp_path.resolve() / "replay.pkl"
    reservation = reserve_streaming_artifact(destination, max_bytes=100)
    reservation.writer.write(b"candidate")
    reservation.writer.flush()
    reservation.partial_path.unlink()
    reservation.partial_path.write_bytes(b"attacker!")

    with pytest.raises(ExperimentContractError, match="changed before publication"):
        reservation.finalize()

    assert not destination.exists()
    assert reservation.partial_path.read_bytes() == b"attacker!"


def test_streaming_publication_does_not_claim_an_unlinked_partial(tmp_path: Path) -> None:
    destination = tmp_path.resolve() / "replay.pkl"
    reservation = reserve_streaming_artifact(destination, max_bytes=100)
    reservation.writer.write(b"candidate")
    reservation.writer.flush()
    reservation.partial_path.unlink()

    with pytest.raises(ExperimentContractError, match="no descriptor-matched retained pathname"):
        reservation.finalize()

    assert reservation.retained_path_after_failure is None
    assert not destination.exists()
    assert not reservation.partial_path.exists()


def test_streaming_publication_rejects_hardlinked_partial(tmp_path: Path) -> None:
    destination = tmp_path.resolve() / "replay.pkl"
    reservation = reserve_streaming_artifact(destination, max_bytes=100)
    reservation.writer.write(b"candidate")
    reservation.writer.flush()
    hardlink = tmp_path / "same-inode"
    os.link(reservation.partial_path, hardlink)

    with pytest.raises(ExperimentContractError, match="changed before publication"):
        reservation.finalize()

    assert not destination.exists()
    assert hardlink.read_bytes() == b"candidate"


def test_streaming_publication_rehashes_after_path_reconciliation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path.resolve() / "replay.pkl"
    payload = b"candidate"
    reservation = reserve_streaming_artifact(destination, max_bytes=100)
    reservation.writer.write(payload)
    original_open_parent = artifact_io._open_parent_directory
    calls = 0

    def open_then_mutate(path: Path, *, create_missing: bool = True) -> int:
        nonlocal calls
        descriptor = original_open_parent(path, create_missing=create_missing)
        calls += 1
        if calls == 2:
            with destination.open("r+b") as mutable:
                mutable.write(b"corrupted")
        return descriptor

    monkeypatch.setattr(artifact_io, "_open_parent_directory", open_then_mutate)

    with pytest.raises(ExperimentContractError, match="final reconciliation"):
        reservation.finalize()

    assert destination.read_bytes() == b"corrupted"
    assert reservation.retained_path_after_failure == destination


def test_streaming_retention_does_not_claim_a_path_after_parent_rename(
    tmp_path: Path,
) -> None:
    parent = tmp_path.resolve() / "evidence"
    destination = parent / "replay.pkl"
    moved_parent = tmp_path.resolve() / "moved-evidence"
    reservation = reserve_streaming_artifact(destination, max_bytes=100)
    reservation.writer.write(b"candidate")
    reservation.writer.flush()
    parent.rename(moved_parent)
    parent.mkdir()

    with pytest.raises(ExperimentContractError, match="no descriptor-matched retained pathname"):
        reservation.finalize()

    assert reservation.retained_path_after_failure is None
    assert not reservation.partial_path.exists()
    assert (moved_parent / reservation.partial_path.name).read_bytes() == b"candidate"


def test_verified_reader_rejects_content_mutation_during_use(tmp_path: Path) -> None:
    destination = tmp_path.resolve() / "replay.pkl"
    reservation = reserve_streaming_artifact(destination, max_bytes=100)
    reservation.writer.write(b"trusted")
    published = reservation.finalize()

    with (
        pytest.raises(ExperimentContractError, match="changed while in use"),
        verified_artifact_reader(
            destination,
            expected_sha256=published.sha256,
            expected_size=published.byte_count,
            max_bytes=100,
        ) as reader,
    ):
        assert reader.read() == b"trusted"
        with destination.open("r+b") as mutable:
            mutable.write(b"changed")


def test_verified_reader_hashes_the_bytes_delivered_to_the_consumer(tmp_path: Path) -> None:
    destination = tmp_path.resolve() / "replay.pkl"
    reservation = reserve_streaming_artifact(destination, max_bytes=100)
    reservation.writer.write(b"trusted")
    published = reservation.finalize()

    with (
        pytest.raises(ExperimentContractError, match="bytes consumed"),
        verified_artifact_reader(
            destination,
            expected_sha256=published.sha256,
            expected_size=published.byte_count,
            max_bytes=100,
            buffer_size=1,
        ) as reader,
    ):
        with destination.open("r+b") as mutable:
            mutable.write(b"evil!!!")
        assert reader.read() == b"evil!!!"
        with destination.open("r+b") as mutable:
            mutable.write(b"trusted")


def test_verified_reader_caps_each_kernel_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path.resolve() / "replay.pkl"
    payload = b"x" * (3 * 1024 * 1024 + 17)
    reservation = reserve_streaming_artifact(destination, max_bytes=len(payload))
    reservation.writer.write(payload)
    published = reservation.finalize()
    original_readv = artifact_io.os.readv
    requested_sizes: list[int] = []

    def recording_readv(descriptor: int, buffers: list[object]) -> int:
        requested_sizes.extend(len(value) for value in buffers)
        return original_readv(descriptor, buffers)

    monkeypatch.setattr(artifact_io.os, "readv", recording_readv)
    with verified_artifact_reader(
        destination,
        expected_sha256=published.sha256,
        expected_size=published.byte_count,
        max_bytes=len(payload),
        buffer_size=64 * 1024,
    ) as reader:
        assert reader.read() == payload

    assert requested_sizes
    assert max(requested_sizes) <= 1024 * 1024


def test_verified_reader_reconciles_leaf_after_reopened_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path.resolve() / "replay.pkl"
    reservation = reserve_streaming_artifact(destination, max_bytes=100)
    reservation.writer.write(b"trusted")
    published = reservation.finalize()
    original_open_parent = artifact_io._open_parent_directory
    calls = 0

    def open_then_replace(path: Path, *, create_missing: bool = True) -> int:
        nonlocal calls
        descriptor = original_open_parent(path, create_missing=create_missing)
        calls += 1
        if calls == 2:
            destination.unlink()
            destination.write_bytes(b"evil!!!")
            destination.chmod(0o600)
        return descriptor

    monkeypatch.setattr(artifact_io, "_open_parent_directory", open_then_replace)

    with (
        pytest.raises(ExperimentContractError, match="final reconciliation"),
        verified_artifact_reader(
            destination,
            expected_sha256=published.sha256,
            expected_size=published.byte_count,
            max_bytes=100,
        ) as reader,
    ):
        assert reader.read() == b"trusted"


def test_verified_reader_rejects_wrong_identity_and_permissions(tmp_path: Path) -> None:
    destination = tmp_path.resolve() / "replay.pkl"
    reservation = reserve_streaming_artifact(destination, max_bytes=100)
    reservation.writer.write(b"trusted")
    published = reservation.finalize()

    with (
        pytest.raises(ExperimentContractError, match="SHA-256 differs"),
        verified_artifact_reader(
            destination,
            expected_sha256="0" * 64,
            expected_size=published.byte_count,
            max_bytes=100,
        ),
    ):
        pass

    with (
        pytest.raises(ExperimentContractError, match="SHA-256 is invalid"),
        verified_artifact_reader(
            destination,
            expected_sha256=None,
            expected_size=published.byte_count,
            max_bytes=100,
        ),
    ):
        pass

    destination.chmod(0o644)
    with (
        pytest.raises(ExperimentContractError, match="metadata differs"),
        verified_artifact_reader(
            destination,
            expected_sha256=published.sha256,
            expected_size=published.byte_count,
            max_bytes=100,
        ),
    ):
        pass


def test_verified_reader_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path.resolve() / "replay.pkl"
    os.mkfifo(fifo, mode=0o600)

    with (
        pytest.raises(ExperimentContractError, match="metadata differs"),
        verified_artifact_reader(
            fifo,
            expected_sha256=hashlib.sha256(b"x").hexdigest(),
            expected_size=1,
            max_bytes=100,
        ),
    ):
        pass


def test_streaming_writer_and_reader_are_accepted_by_sb3(tmp_path: Path) -> None:
    save_util = pytest.importorskip("stable_baselines3.common.save_util")
    destination = tmp_path.resolve() / "small-replay.pkl"
    value = {"observations": np.arange(24, dtype=np.float64).reshape(3, 8)}
    reservation = reserve_streaming_artifact(destination, max_bytes=1024 * 1024)

    save_util.save_to_pkl(reservation.writer, value)
    published = reservation.finalize()

    with verified_artifact_reader(
        destination,
        expected_sha256=published.sha256,
        expected_size=published.byte_count,
        max_bytes=1024 * 1024,
    ) as reader:
        loaded = save_util.load_from_pkl(reader)
    np.testing.assert_array_equal(loaded["observations"], value["observations"])


@pytest.mark.gym
def test_streaming_round_trip_uses_tqc_replay_methods(tmp_path: Path) -> None:
    gym = pytest.importorskip("gymnasium")
    tqc_module = pytest.importorskip("sb3_contrib")
    vec_env_module = pytest.importorskip("stable_baselines3.common.vec_env")
    env = vec_env_module.DummyVecEnv([lambda: gym.make("Humanoid-v5") for _index in range(5)])
    model = tqc_module.TQC(
        "MlpPolicy",
        env,
        buffer_size=100,
        learning_starts=1_000,
        policy_kwargs={"net_arch": [16, 16]},
        seed=91_001,
        device="cpu",
        verbose=0,
    )
    try:
        observations = np.zeros((5, 348), dtype=np.float64)
        next_observations = np.full((5, 348), 0.25, dtype=np.float64)
        actions = np.zeros((5, 17), dtype=np.float32)
        rewards = np.arange(5, dtype=np.float32)
        dones = np.zeros(5, dtype=np.float32)
        model.replay_buffer.add(
            observations,
            next_observations,
            actions,
            rewards,
            dones,
            [{} for _index in range(5)],
        )
        expected_observations = model.replay_buffer.observations.copy()
        destination = tmp_path.resolve() / "tqc-replay.pkl"
        reservation = reserve_streaming_artifact(
            destination,
            max_bytes=16 * 1024 * 1024,
        )

        model.save_replay_buffer(reservation.writer)
        published = reservation.finalize()
        model.replay_buffer = None
        with verified_artifact_reader(
            destination,
            expected_sha256=published.sha256,
            expected_size=published.byte_count,
            max_bytes=16 * 1024 * 1024,
        ) as reader:
            model.load_replay_buffer(reader, truncate_last_traj=False)

        assert model.replay_buffer is not None
        assert model.replay_buffer.pos == 1
        assert model.replay_buffer.full is False
        np.testing.assert_array_equal(
            model.replay_buffer.observations,
            expected_observations,
        )
    finally:
        env.close()
