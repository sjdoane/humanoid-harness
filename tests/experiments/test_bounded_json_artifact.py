from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from oracle_composition.experiments import fixed_reference
from oracle_composition.experiments.fixed_reference import (
    ExperimentContractError,
    read_bounded_json_artifact,
    sha256_file,
)


def test_reader_binds_parsed_value_to_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    encoded = b'{"value":1}\n'
    path.write_bytes(encoded)

    artifact = read_bounded_json_artifact(path, maximum_bytes=64, artifact="test input")

    assert artifact.value == {"value": 1}
    assert artifact.encoded_bytes == encoded
    assert artifact.sha256 == hashlib.sha256(encoded).hexdigest()


def test_reader_rejects_symlink_and_fifo_without_blocking(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    linked = tmp_path / "linked.json"
    linked.symlink_to(target)
    fifo = tmp_path / "input.fifo"
    os.mkfifo(fifo)

    with pytest.raises(ExperimentContractError, match="symlink"):
        read_bounded_json_artifact(linked, maximum_bytes=64, artifact="test input")
    with pytest.raises(ExperimentContractError, match="regular file"):
        read_bounded_json_artifact(fifo, maximum_bytes=64, artifact="test input")


def test_reader_rejects_file_changed_during_read(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "input.json"
    path.write_bytes(b'{"value":1}')
    original_read = os.read
    mutated = False

    def read_and_mutate(descriptor: int, size: int) -> bytes:
        nonlocal mutated
        chunk = original_read(descriptor, size)
        if chunk and not mutated:
            mutated = True
            path.write_bytes(b'{"value":2}')
        return chunk

    monkeypatch.setattr(fixed_reference.os, "read", read_and_mutate)
    with pytest.raises(ExperimentContractError, match="changed while it was read"):
        read_bounded_json_artifact(path, maximum_bytes=64, artifact="test input")


def test_file_hasher_rejects_symlink_and_fifo_without_blocking(tmp_path: Path) -> None:
    target = tmp_path / "target.bin"
    target.write_bytes(b"evidence")
    linked = tmp_path / "linked.bin"
    linked.symlink_to(target)
    fifo = tmp_path / "input.fifo"
    os.mkfifo(fifo)

    assert sha256_file(target) == hashlib.sha256(b"evidence").hexdigest()
    with pytest.raises(ExperimentContractError, match="symlink"):
        sha256_file(linked)
    with pytest.raises(ExperimentContractError, match="regular file"):
        sha256_file(fifo)


def test_file_hasher_rejects_path_replacement_during_hash(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"first")
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"second")
    original_read = os.read
    replaced = False

    def read_and_replace(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        chunk = original_read(descriptor, size)
        if chunk and not replaced:
            replaced = True
            replacement.replace(path)
        return chunk

    monkeypatch.setattr(fixed_reference.os, "read", read_and_replace)
    with pytest.raises(ExperimentContractError, match="changed while it was hashed"):
        sha256_file(path)


def test_file_hasher_rejects_path_replacement_before_open(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"first")
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"other")
    original_open = os.open
    replaced = False

    def open_after_replace(target: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal replaced
        if Path(target) == path and not replaced:
            replaced = True
            replacement.replace(path)
        return original_open(target, flags, *args, **kwargs)

    monkeypatch.setattr(fixed_reference.os, "open", open_after_replace)
    with pytest.raises(ExperimentContractError, match="changed before it was hashed"):
        sha256_file(path)


def test_file_hasher_enforces_frozen_byte_bound(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "input.bin"
    path.write_bytes(b"a")
    with pytest.raises(ExperimentContractError, match="positive integer"):
        sha256_file(path, maximum_bytes=0)
    with pytest.raises(ExperimentContractError, match="bounded size limit"):
        path.write_bytes(b"ab")
        sha256_file(path, maximum_bytes=1)

    path.write_bytes(b"a")
    original_read = os.read
    grew = False

    def read_then_grow(descriptor: int, size: int) -> bytes:
        nonlocal grew
        chunk = original_read(descriptor, size)
        if chunk and not grew:
            grew = True
            with path.open("ab") as stream:
                stream.write(b"b")
        return chunk

    monkeypatch.setattr(fixed_reference.os, "read", read_then_grow)
    with pytest.raises(ExperimentContractError, match="grew while it was hashed"):
        sha256_file(path, maximum_bytes=1)
