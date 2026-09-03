from __future__ import annotations

from pathlib import Path

import pytest

from oracle_composition.experiments.runtime_identity import resolve_dependency_lock


def test_dependency_lock_resolver_rejects_absent_candidates(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="absent"):
        resolve_dependency_lock(
            package_lock=tmp_path / "package" / "uv.lock",
            checkout_lock=tmp_path / "checkout" / "uv.lock",
        )


def test_dependency_lock_resolver_rejects_disagreement(tmp_path: Path) -> None:
    package_lock = tmp_path / "package.lock"
    checkout_lock = tmp_path / "checkout.lock"
    package_lock.write_bytes(b"package-lock")
    checkout_lock.write_bytes(b"checkout-lock")

    with pytest.raises(RuntimeError, match="disagree"):
        resolve_dependency_lock(package_lock=package_lock, checkout_lock=checkout_lock)


@pytest.mark.parametrize("linked_candidate", ["package", "checkout"])
@pytest.mark.parametrize("target_exists", [True, False])
def test_dependency_lock_resolver_rejects_symlinks(
    tmp_path: Path,
    linked_candidate: str,
    target_exists: bool,
) -> None:
    target = tmp_path / "target.lock"
    if target_exists:
        target.write_bytes(b"exact-lock")
    linked_lock = tmp_path / "linked.lock"
    linked_lock.symlink_to(target)
    package_lock = (
        linked_lock if linked_candidate == "package" else tmp_path / "absent-package.lock"
    )
    checkout_lock = (
        linked_lock if linked_candidate == "checkout" else tmp_path / "absent-checkout.lock"
    )

    with pytest.raises(RuntimeError, match="symlink"):
        resolve_dependency_lock(
            package_lock=package_lock,
            checkout_lock=checkout_lock,
        )


def test_dependency_lock_resolver_prefers_matching_packaged_bytes(tmp_path: Path) -> None:
    package_lock = tmp_path / "package.lock"
    checkout_lock = tmp_path / "checkout.lock"
    package_lock.write_bytes(b"exact-lock")
    checkout_lock.write_bytes(b"exact-lock")

    assert (
        resolve_dependency_lock(package_lock=package_lock, checkout_lock=checkout_lock)
        == package_lock
    )
