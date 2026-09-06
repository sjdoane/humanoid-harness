"""Sealed worker inputs and checkout-local module identity."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.experiments.fixed_reference import ExperimentContractError

SEALED_INPUT_LINEAGE_ID = "humanoid_phase_b_sealed_input_lineage/v1"
EXECUTED_MODULE_IDENTITY_ID = "humanoid_phase_b_executed_module_identity/v1"


@dataclass(frozen=True, slots=True)
class SealedArtifact:
    relative_path: str
    byte_count: int
    sha256: str
    roles: tuple[str, ...]

    def __post_init__(self) -> None:
        relative = PurePosixPath(self.relative_path)
        if (
            not self.relative_path
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != self.relative_path
        ):
            raise ValueError("sealed artifact path must be canonical and relative")
        if type(self.byte_count) is not int or self.byte_count < 0:
            raise ValueError("sealed artifact byte count is invalid")
        if (
            type(self.sha256) is not str
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("sealed artifact SHA-256 is invalid")
        if (
            not self.roles
            or tuple(sorted(set(self.roles))) != self.roles
            or any(type(role) is not str or not role for role in self.roles)
        ):
            raise ValueError("sealed artifact roles are invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "byte_count": self.byte_count,
            "path": self.relative_path,
            "roles": list(self.roles),
            "sha256": self.sha256,
        }


def _regular_file_identity(path: Path) -> tuple[int, int, int, int, int]:
    value = path.lstat()
    if stat.S_ISLNK(value.st_mode) or not stat.S_ISREG(value.st_mode):
        raise ExperimentContractError(f"sealed artifact is not a regular file: {path}")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _checked_path(root: Path, relative_path: str) -> Path:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != relative_path:
        raise ExperimentContractError("sealed artifact path escapes the checkout")
    candidate = root.joinpath(*relative.parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ExperimentContractError(
            "sealed artifact does not resolve beneath the checkout"
        ) from exc
    if resolved != candidate:
        raise ExperimentContractError("sealed artifact path traverses a symbolic link")
    return candidate


def _digest_regular_file(path: Path) -> tuple[int, str]:
    before = _regular_file_identity(path)
    digest = hashlib.sha256()
    observed = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            observed += len(chunk)
    after = _regular_file_identity(path)
    if before != after or observed != before[2]:
        raise ExperimentContractError(f"sealed artifact changed while read: {path}")
    return observed, digest.hexdigest()


def seal_artifact(
    repository_root: Path,
    path: Path,
    *,
    roles: Sequence[str],
    expected_byte_count: int | None = None,
    expected_sha256: str | None = None,
) -> SealedArtifact:
    """Bind one checkout-local regular file to its exact bytes."""

    root = Path(repository_root).resolve(strict=True)
    raw = Path(os.path.abspath(path))
    try:
        relative = raw.relative_to(root).as_posix()
    except ValueError as exc:
        raise ExperimentContractError("sealed artifact lies outside the declared checkout") from exc
    candidate = _checked_path(root, relative)
    byte_count, digest = _digest_regular_file(candidate)
    if expected_byte_count is not None and byte_count != expected_byte_count:
        raise ExperimentContractError("sealed artifact byte count differs from its authority")
    if expected_sha256 is not None and digest != expected_sha256:
        raise ExperimentContractError("sealed artifact digest differs from its authority")
    return SealedArtifact(relative, byte_count, digest, tuple(sorted(set(roles))))


def sealed_input_lineage_value(artifacts: Sequence[SealedArtifact]) -> dict[str, object]:
    ordered = tuple(sorted(artifacts, key=lambda item: item.relative_path))
    paths = [item.relative_path for item in ordered]
    if len(paths) != len(set(paths)):
        raise ExperimentContractError("sealed input lineage contains duplicate paths")
    return {
        "artifacts": [item.to_dict() for item in ordered],
        "lineage_id": SEALED_INPUT_LINEAGE_ID,
        "schema_version": 1,
    }


def sealed_input_lineage_sha256(artifacts: Sequence[SealedArtifact]) -> str:
    return hashlib.sha256(canonical_json_bytes(sealed_input_lineage_value(artifacts))).hexdigest()


def verify_sealed_inputs(
    repository_root: Path,
    artifacts: Sequence[SealedArtifact],
    *,
    expected_lineage_sha256: str,
) -> None:
    """Reopen and verify every parent-bound input without following links."""

    root = Path(repository_root).resolve(strict=True)
    if sealed_input_lineage_sha256(artifacts) != expected_lineage_sha256:
        raise ExperimentContractError("sealed input lineage declaration differs")
    for artifact in artifacts:
        path = _checked_path(root, artifact.relative_path)
        byte_count, digest = _digest_regular_file(path)
        if byte_count != artifact.byte_count or digest != artifact.sha256:
            raise ExperimentContractError(
                f"sealed input changed after preflight: {artifact.relative_path}"
            )


def verify_one_sealed_input(
    repository_root: Path,
    artifacts: Sequence[SealedArtifact],
    path: Path,
) -> None:
    """Verify the parent binding again at the point an artifact is consumed."""

    root = Path(repository_root).resolve(strict=True)
    candidate = Path(os.path.abspath(path))
    try:
        relative = candidate.relative_to(root).as_posix()
    except ValueError as exc:
        raise ExperimentContractError(
            "consumed artifact lies outside the declared checkout"
        ) from exc
    matches = [artifact for artifact in artifacts if artifact.relative_path == relative]
    if len(matches) != 1:
        raise ExperimentContractError(
            f"consumed artifact is absent from sealed lineage: {relative}"
        )
    bound = matches[0]
    verified = _checked_path(root, bound.relative_path)
    byte_count, digest = _digest_regular_file(verified)
    if byte_count != bound.byte_count or digest != bound.sha256:
        raise ExperimentContractError(f"sealed input changed at use: {relative}")


def validate_executing_modules(
    repository_root: Path,
    source_sha256: Mapping[str, object],
    *,
    module_files: Mapping[str, str | Path] | None = None,
) -> dict[str, object]:
    """Bind every loaded project module to a recorded source under this checkout."""

    root = Path(repository_root).resolve(strict=True)
    if module_files is None:
        selected: dict[str, str | Path] = {}
        for name, module in tuple(sys.modules.items()):
            if name != "oracle_composition" and not name.startswith("oracle_composition."):
                continue
            module_path = getattr(module, "__file__", None)
            if module_path is not None:
                selected[name] = module_path
    else:
        selected = dict(module_files)
    if not selected:
        raise ExperimentContractError("no executing project modules were identified")
    rows = []
    for name, raw_path in sorted(selected.items()):
        if type(name) is not str or not name:
            raise ExperimentContractError("executing module name is malformed")
        path = Path(raw_path)
        if path.suffix in {".pyc", ".pyo"}:
            try:
                path = Path(importlib.util.source_from_cache(str(path)))
            except ValueError as exc:
                raise ExperimentContractError("executing bytecode has no source identity") from exc
        path = Path(os.path.abspath(path))
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ExperimentContractError(
                f"executing module resolved outside the declared checkout: {name}"
            ) from exc
        recorded = source_sha256.get(relative)
        if type(recorded) is not str:
            raise ExperimentContractError(
                f"executing module is absent from source snapshot: {name}"
            )
        candidate = _checked_path(root, relative)
        byte_count, digest = _digest_regular_file(candidate)
        if digest != recorded:
            raise ExperimentContractError(f"executing module digest differs: {name}")
        rows.append(
            {
                "byte_count": byte_count,
                "module": name,
                "path": relative,
                "sha256": digest,
            }
        )
    value = {
        "identity_id": EXECUTED_MODULE_IDENTITY_ID,
        "modules": rows,
        "schema_version": 1,
    }
    return {
        "identity": value,
        "sha256": hashlib.sha256(canonical_json_bytes(value)).hexdigest(),
    }


__all__ = [
    "EXECUTED_MODULE_IDENTITY_ID",
    "SEALED_INPUT_LINEAGE_ID",
    "SealedArtifact",
    "seal_artifact",
    "sealed_input_lineage_sha256",
    "sealed_input_lineage_value",
    "validate_executing_modules",
    "verify_one_sealed_input",
    "verify_sealed_inputs",
]
