from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import pytest

from oracle_composition.experiments import artifact_io
from oracle_composition.experiments.artifact_io import (
    finite_pretty_json,
    publish_bytes_without_overwrite,
)
from oracle_composition.experiments.fixed_reference import ExperimentContractError


def test_publication_retains_exact_descriptor_verified_bytes(tmp_path: Path) -> None:
    destination = tmp_path.resolve() / "evidence" / "receipt.json"
    encoded = b'{"safe":true}\n'

    published = publish_bytes_without_overwrite(destination, encoded)

    assert published.path == destination
    assert destination.read_bytes() == encoded
    assert published.sha256 == hashlib.sha256(encoded).hexdigest()
    assert published.byte_count == len(encoded)
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert not list(destination.parent.glob(".*.pending"))
    with pytest.raises(ExperimentContractError, match="refusing to overwrite"):
        publish_bytes_without_overwrite(destination, encoded)


def test_publication_rejects_linked_parent_and_final_component(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ExperimentContractError, match="ancestors"):
        publish_bytes_without_overwrite(linked_parent / "receipt.json", b"safe")
    assert not (outside / "receipt.json").exists()

    final_target = tmp_path / "must-not-exist"
    final_link = tmp_path / "receipt.json"
    final_link.symlink_to(final_target)
    with pytest.raises(ExperimentContractError, match="refusing to overwrite"):
        publish_bytes_without_overwrite(final_link, b"safe")
    assert final_link.is_symlink()
    assert not final_target.exists()


def test_publication_rejects_pending_path_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path.resolve() / "receipt.json"
    encoded = b"claimed-bytes"
    replacement = b"x" * len(encoded)
    original_rename = artifact_io._rename_without_overwrite

    def substitute_then_rename(
        parent_descriptor: int,
        source: str,
        target: str,
    ) -> None:
        os.unlink(source, dir_fd=parent_descriptor)
        descriptor = os.open(
            source,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent_descriptor,
        )
        try:
            assert os.write(descriptor, replacement) == len(replacement)
        finally:
            os.close(descriptor)
        original_rename(parent_descriptor, source, target)

    monkeypatch.setattr(artifact_io, "_rename_without_overwrite", substitute_then_rename)

    with pytest.raises(ExperimentContractError, match="path changed"):
        publish_bytes_without_overwrite(destination, encoded)

    assert destination.read_bytes() == replacement


def test_publication_rejects_same_inode_content_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path.resolve() / "receipt.json"
    encoded = b"claimed-bytes"
    replacement = b"x" * len(encoded)
    original_rename = artifact_io._rename_without_overwrite

    def rename_then_rewrite(
        parent_descriptor: int,
        source: str,
        target: str,
    ) -> None:
        original_rename(parent_descriptor, source, target)
        descriptor = os.open(target, os.O_WRONLY, dir_fd=parent_descriptor)
        try:
            assert os.write(descriptor, replacement) == len(replacement)
        finally:
            os.close(descriptor)

    monkeypatch.setattr(artifact_io, "_rename_without_overwrite", rename_then_rewrite)

    with pytest.raises(ExperimentContractError, match="bytes changed"):
        publish_bytes_without_overwrite(destination, encoded)

    assert destination.read_bytes() == replacement


def test_publication_reconciles_final_entry_after_last_descriptor_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path.resolve() / "receipt.json"
    encoded = b"claimed-bytes"
    replacement = b"x" * len(encoded)
    original_hash = artifact_io._descriptor_sha256
    calls = 0

    def hash_then_replace(descriptor: int, *, expected_size: int) -> str:
        nonlocal calls
        calls += 1
        digest = original_hash(descriptor, expected_size=expected_size)
        if calls == 2:
            destination.unlink()
            destination.write_bytes(replacement)
        return digest

    monkeypatch.setattr(artifact_io, "_descriptor_sha256", hash_then_replace)

    with pytest.raises(ExperimentContractError, match="not visible"):
        publish_bytes_without_overwrite(destination, encoded)

    assert destination.read_bytes() == replacement


def test_publication_reconciles_renamed_parent_after_last_descriptor_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path.resolve() / "evidence"
    parent.mkdir()
    destination = parent / "receipt.json"
    moved_parent = tmp_path.resolve() / "moved-evidence"
    encoded = b"claimed-bytes"
    replacement = b"x" * len(encoded)
    original_hash = artifact_io._descriptor_sha256
    calls = 0

    def hash_then_move_parent(descriptor: int, *, expected_size: int) -> str:
        nonlocal calls
        calls += 1
        digest = original_hash(descriptor, expected_size=expected_size)
        if calls == 2:
            parent.rename(moved_parent)
            parent.mkdir()
            destination.write_bytes(replacement)
        return digest

    monkeypatch.setattr(artifact_io, "_descriptor_sha256", hash_then_move_parent)

    with pytest.raises(ExperimentContractError, match="ancestors changed"):
        publish_bytes_without_overwrite(destination, encoded)

    assert destination.read_bytes() == replacement
    assert (moved_parent / "receipt.json").read_bytes() == encoded


def test_json_encoder_rejects_nonfinite_values() -> None:
    with pytest.raises(ExperimentContractError, match="finite JSON"):
        finite_pretty_json({"invalid": float("nan")})
