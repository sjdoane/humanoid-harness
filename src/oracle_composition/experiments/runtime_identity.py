"""Public, bounded runtime-identity helpers."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import numpy as np

from .fixed_reference import sha256_file

MAX_SOURCE_TREE_FILES = 1024
MAX_SOURCE_FILE_BYTES = 4 * 1024 * 1024
MAX_SOURCE_TREE_BYTES = 32 * 1024 * 1024


def _open_directory_without_links(path: Path) -> tuple[Path, int]:
    absolute = Path(os.path.abspath(path))
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(absolute.anchor, flags)
        for component in absolute.parts[1:]:
            if component in {"", ".", ".."}:
                raise RuntimeError("source-tree path contains an unsafe component")
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return absolute, descriptor
    except RuntimeError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise RuntimeError(
            "source-tree root and ancestors must be real directories without symbolic links"
        ) from exc


def _file_state(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_uid),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_source_file(directory_descriptor: int, name: str) -> bytes:
    before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError("source-tree entries must be regular Python files")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
        opened = os.fstat(descriptor)
        if _file_state(opened) != _file_state(before):
            raise RuntimeError("source-tree file changed while it was opened")
        if not 0 <= opened.st_size <= MAX_SOURCE_FILE_BYTES:
            raise RuntimeError("source-tree Python file exceeds the bounded size limit")
        chunks: list[bytes] = []
        observed = 0
        while True:
            chunk = os.read(descriptor, min(64 * 1024, MAX_SOURCE_FILE_BYTES + 1 - observed))
            if not chunk:
                break
            chunks.append(chunk)
            observed += len(chunk)
            if observed > MAX_SOURCE_FILE_BYTES:
                raise RuntimeError("source-tree Python file exceeds the bounded size limit")
        after = os.fstat(descriptor)
        visible_after = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if (
            _file_state(after) != _file_state(opened)
            or _file_state(visible_after) != _file_state(opened)
            or observed != opened.st_size
        ):
            raise RuntimeError("source-tree file changed while it was read")
        return b"".join(chunks)
    except RuntimeError:
        raise
    except OSError as exc:
        raise RuntimeError(f"cannot read source-tree file {name}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _source_entries(
    directory_descriptor: int,
    *,
    relative_parts: tuple[str, ...] = (),
) -> list[tuple[str, bytes]]:
    result: list[tuple[str, bytes]] = []
    try:
        names = sorted(os.listdir(directory_descriptor))
    except OSError as exc:
        raise RuntimeError(f"cannot enumerate source-tree directory: {exc}") from exc
    directory_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    for name in names:
        if name in {"", ".", ".."} or "/" in name or "\0" in name:
            raise RuntimeError("source-tree entry name is unsafe")
        before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode):
            if name.endswith(".py"):
                raise RuntimeError("source-tree entries must be regular Python files")
            continue
        if stat.S_ISDIR(before.st_mode):
            child_descriptor: int | None = None
            try:
                child_descriptor = os.open(name, directory_flags, dir_fd=directory_descriptor)
                opened = os.fstat(child_descriptor)
                if _file_state(opened) != _file_state(before):
                    raise RuntimeError("source-tree directory changed while it was opened")
                result.extend(
                    _source_entries(
                        child_descriptor,
                        relative_parts=(*relative_parts, name),
                    )
                )
                visible_after = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if _file_state(os.fstat(child_descriptor)) != _file_state(opened) or _file_state(
                    visible_after
                ) != _file_state(opened):
                    raise RuntimeError("source-tree directory changed during traversal")
            except RuntimeError:
                raise
            except OSError as exc:
                raise RuntimeError(f"cannot inspect source-tree directory {name}: {exc}") from exc
            finally:
                if child_descriptor is not None:
                    os.close(child_descriptor)
        elif name.endswith(".py"):
            relative = "/".join((*relative_parts, name))
            result.append((relative, _read_source_file(directory_descriptor, name)))
    return result


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

    requested = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    source_root, descriptor = _open_directory_without_links(requested)
    initial_state = _file_state(os.fstat(descriptor))
    try:
        candidates = sorted(_source_entries(descriptor), key=lambda item: item[0])
        if not candidates or len(candidates) > MAX_SOURCE_TREE_FILES:
            raise RuntimeError("source-tree Python file count is outside the bounded contract")
        digest = hashlib.sha256()
        total_bytes = 0
        for relative_name, payload in candidates:
            total_bytes += len(payload)
            if total_bytes > MAX_SOURCE_TREE_BYTES:
                raise RuntimeError("source-tree bytes exceed the bounded aggregate size limit")
            relative = relative_name.encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        _verification_root, verification_descriptor = _open_directory_without_links(source_root)
        try:
            if (
                _file_state(os.fstat(descriptor)) != initial_state
                or _file_state(os.fstat(verification_descriptor)) != initial_state
            ):
                raise RuntimeError("source-tree root changed during traversal")
        finally:
            os.close(verification_descriptor)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


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
