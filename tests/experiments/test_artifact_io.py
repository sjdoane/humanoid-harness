from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from oracle_composition.experiments import artifact_io
from oracle_composition.experiments.artifact_io import (
    finite_pretty_json,
    publish_bytes_without_overwrite,
    reserve_json_artifact,
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


def test_reserved_json_is_valid_then_atomically_finalized(tmp_path: Path) -> None:
    destination = tmp_path.resolve() / "receipt.json"
    reservation = reserve_json_artifact(destination, {"status": "in_progress"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "in_progress"}
    published = reservation.finalize({"status": "failed", "reason": "bounded stop"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "reason": "bounded stop",
        "status": "failed",
    }
    assert published.sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert not list(tmp_path.glob(".*.final"))


def test_reserved_json_rejects_final_path_substitution(tmp_path: Path) -> None:
    destination = tmp_path.resolve() / "receipt.json"
    reservation = reserve_json_artifact(destination, {"status": "in_progress"})
    destination.unlink()
    destination.write_text('{"attacker":true}\n', encoding="utf-8")

    with pytest.raises(ExperimentContractError, match="path changed"):
        reservation.finalize({"status": "complete"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"attacker": True}
    assert not list(tmp_path.glob(".*.final"))


def test_reserved_json_keeps_provisional_receipt_if_exchange_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path.resolve() / "receipt.json"
    reservation = reserve_json_artifact(destination, {"status": "in_progress"})

    def fail_exchange(_parent_descriptor: int, _left: str, _right: str) -> None:
        raise OSError("forced exchange failure")

    monkeypatch.setattr(artifact_io, "_exchange_entries", fail_exchange)

    with pytest.raises(ExperimentContractError, match="cannot finalize"):
        reservation.finalize({"status": "complete"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "in_progress"}
    assert not list(tmp_path.glob(".*.final"))


def test_reserved_json_removes_descriptor_matched_partial_reservation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path.resolve() / "receipt.json"

    def partial_write_then_fail(descriptor: int, _encoded: bytes) -> None:
        assert os.write(descriptor, b"{") == 1
        raise OSError("forced partial reservation write")

    monkeypatch.setattr(artifact_io, "_write_descriptor_bytes", partial_write_then_fail)

    with pytest.raises(ExperimentContractError, match="cannot reserve"):
        reserve_json_artifact(destination, {"status": "in_progress"})

    assert not destination.exists()


def test_reserved_json_restores_provisional_after_post_exchange_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path.resolve() / "receipt.json"
    reservation = reserve_json_artifact(destination, {"status": "in_progress"})
    original_exchange = artifact_io._exchange_entries
    calls = 0

    def exchange_then_corrupt(parent_descriptor: int, left: str, right: str) -> None:
        nonlocal calls
        calls += 1
        original_exchange(parent_descriptor, left, right)
        if calls == 1:
            descriptor = os.open(left, os.O_WRONLY | os.O_TRUNC, dir_fd=parent_descriptor)
            try:
                assert os.write(descriptor, b"corrupt") == len(b"corrupt")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    monkeypatch.setattr(artifact_io, "_exchange_entries", exchange_then_corrupt)

    with pytest.raises(ExperimentContractError, match="changed during finalization"):
        reservation.finalize({"status": "complete"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"status": "in_progress"}
    assert calls == 2
    assert not list(tmp_path.glob(".*.final"))


def test_reserved_json_retains_recovery_after_post_exchange_path_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path.resolve() / "receipt.json"
    reservation = reserve_json_artifact(destination, {"status": "in_progress"})
    original_exchange = artifact_io._exchange_entries

    def exchange_then_replace(parent_descriptor: int, left: str, right: str) -> None:
        original_exchange(parent_descriptor, left, right)
        os.unlink(left, dir_fd=parent_descriptor)
        descriptor = os.open(
            left,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=parent_descriptor,
        )
        try:
            assert os.write(descriptor, b'{"attacker":true}\n') == len(b'{"attacker":true}\n')
        finally:
            os.close(descriptor)

    monkeypatch.setattr(artifact_io, "_exchange_entries", exchange_then_replace)

    with pytest.raises(ExperimentContractError, match="unexpected identities"):
        reservation.finalize({"status": "complete"})

    recovery = tmp_path / ".receipt.json.in-progress-recovery"
    assert json.loads(destination.read_text(encoding="utf-8")) == {"attacker": True}
    assert json.loads(recovery.read_text(encoding="utf-8")) == {"status": "in_progress"}
    assert not list(tmp_path.glob(".*.final"))


def test_reserved_json_restores_provisional_inside_renamed_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tmp_path.resolve() / "evidence"
    parent.mkdir()
    destination = parent / "receipt.json"
    moved_parent = tmp_path.resolve() / "moved-evidence"
    reservation = reserve_json_artifact(destination, {"status": "in_progress"})
    original_exchange = artifact_io._exchange_entries
    calls = 0

    def exchange_then_move_parent(parent_descriptor: int, left: str, right: str) -> None:
        nonlocal calls
        calls += 1
        original_exchange(parent_descriptor, left, right)
        if calls == 1:
            parent.rename(moved_parent)
            parent.mkdir()
            destination.write_text('{"attacker":true}\n', encoding="utf-8")

    monkeypatch.setattr(artifact_io, "_exchange_entries", exchange_then_move_parent)

    with pytest.raises(ExperimentContractError, match="ancestors changed"):
        reservation.finalize({"status": "complete"})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"attacker": True}
    assert json.loads((moved_parent / "receipt.json").read_text(encoding="utf-8")) == {
        "status": "in_progress"
    }
    assert calls == 2
    assert not list(moved_parent.glob(".*.final"))
