from __future__ import annotations

import hashlib
import io
import os
import stat
from pathlib import Path

import numpy as np
import pytest

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
