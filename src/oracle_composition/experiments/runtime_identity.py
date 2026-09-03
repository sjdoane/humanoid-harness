"""Public, bounded runtime-identity helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .fixed_reference import sha256_file

MAX_SOURCE_TREE_FILES = 1024
MAX_SOURCE_FILE_BYTES = 4 * 1024 * 1024
MAX_SOURCE_TREE_BYTES = 32 * 1024 * 1024


def resolve_dependency_lock(*, package_lock: Path, checkout_lock: Path) -> Path:
    """Select one immutable dependency lock and reject ambiguity."""

    candidates = (package_lock, checkout_lock)
    if any(path.is_symlink() for path in candidates):
        raise RuntimeError("runtime dependency lock must be a regular file, not a symlink")
    present = [path for path in candidates if path.is_file()]
    if not present:
        raise RuntimeError("runtime dependency lock is absent from the checkout and package")
    if len(present) == 2 and sha256_file(present[0]) != sha256_file(present[1]):
        raise RuntimeError("packaged and checkout dependency locks disagree")
    return package_lock if package_lock in present else checkout_lock


def dependency_lock_path() -> Path:
    """Resolve the exact lock from a checkout or installed wheel."""

    return resolve_dependency_lock(
        package_lock=Path(__file__).resolve().parents[1] / "_runtime" / "uv.lock",
        checkout_lock=Path(__file__).resolve().parents[3] / "uv.lock",
    )


def module_sha256(module: object) -> str:
    """Hash the exact source file for one imported module."""

    source_path = Path(str(getattr(module, "__file__", "")))
    return sha256_file(source_path)


def source_tree_sha256(root: Path | None = None) -> str:
    """Hash sorted, bounded Python source bytes under a package tree."""

    if root is not None and Path(root).is_symlink():
        raise RuntimeError("source-tree root must be a regular directory, not a symlink")
    source_root = root.resolve() if root is not None else Path(__file__).resolve().parents[1]
    if not source_root.is_dir() or source_root.is_symlink():
        raise RuntimeError("source-tree root must be a regular directory, not a symlink")
    candidates = sorted(
        source_root.rglob("*.py"),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )
    if not candidates or len(candidates) > MAX_SOURCE_TREE_FILES:
        raise RuntimeError("source-tree Python file count is outside the bounded contract")
    digest = hashlib.sha256()
    total_bytes = 0
    for path in candidates:
        if path.is_symlink() or not path.is_file():
            raise RuntimeError("source-tree entries must be regular Python files")
        try:
            with path.open("rb") as stream:
                payload = stream.read(MAX_SOURCE_FILE_BYTES + 1)
        except OSError as exc:
            raise RuntimeError(f"cannot read source-tree file {path}: {exc}") from exc
        if len(payload) > MAX_SOURCE_FILE_BYTES:
            raise RuntimeError("source-tree Python file exceeds the bounded size limit")
        total_bytes += len(payload)
        if total_bytes > MAX_SOURCE_TREE_BYTES:
            raise RuntimeError("source-tree bytes exceed the bounded aggregate size limit")
        relative = path.relative_to(source_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def space_sha256(space: object) -> str:
    """Hash exact Box bounds, shape, and dtype."""

    low = np.asarray(getattr(space, "low", None))
    high = np.asarray(getattr(space, "high", None))
    shape = tuple(int(value) for value in getattr(space, "shape", ()))
    dtype = np.dtype(getattr(space, "dtype", low.dtype))
    if low.shape != shape or high.shape != shape or not shape:
        raise RuntimeError("runtime space does not expose exact Box bounds")
    digest = hashlib.sha256()
    metadata = json.dumps(
        {"dtype": dtype.str, "shape": list(shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    for label, payload in (
        (b"metadata", metadata),
        (b"low", np.ascontiguousarray(low, dtype=dtype).tobytes()),
        (b"high", np.ascontiguousarray(high, dtype=dtype).tobytes()),
    ):
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


__all__ = [
    "dependency_lock_path",
    "module_sha256",
    "resolve_dependency_lock",
    "source_tree_sha256",
    "space_sha256",
]
