"""Two-phase execution-manifest authority for the frozen TQC v2 screen."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import secrets
import stat
import time
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass
from pathlib import Path
from typing import Any

from .artifact_io import publish_bytes_without_overwrite
from .fixed_reference import ExperimentContractError
from .runtime_identity import (
    MAX_SOURCE_FILE_BYTES,
    MAX_SOURCE_TREE_BYTES,
    MAX_SOURCE_TREE_FILES,
)
from .tqc_calibration_contract import canonical_json
from .tqc_development_contract_v2 import (
    AUDIT_FILE_SHA256,
    AUDIT_SEMANTIC_SHA256,
    DESIGN_FILE_SHA256,
    DESIGN_ID,
    DESIGN_SEMANTIC_SHA256,
    E0_DESIGN_FILE_SHA256,
    E0_RECEIPT_FILE_SHA256,
    E0_RUNTIME_SHA256,
    TRAINING_PROJECTION_SHA256,
    V1_DESIGN_FILE_SHA256,
    LoadedTQCDevelopmentDesignV2,
    ValidatedV2E0ReuseEvidence,
)
from .tqc_development_review_v2 import ValidatedIndependentReviewReceiptV2
from .tqc_development_runtime_v2 import TQCHostRuntimeReceiptV2, _host_hardware_identity

PREFLIGHT_CONTRACT_ID = "tqc_dev_1m_v2_preflight_contract/v1"
ATTEMPT_RESERVATION_RECEIPT_ID = "tqc_dev_1m_v2_attempt_reservation/v1"
WORKER_BOOTSTRAP_RECEIPT_ID = "tqc_dev_1m_v2_worker_bootstrap_identity/v1"
WORKER_PREFLIGHT_SET_ID = "tqc_dev_1m_v2_worker_preflight_receipt_set/v1"
HOST_RUNTIME_RECEIPT_ID = "tqc_dev_1m_v2_host_runtime/v1"
INSTRUMENTATION_RECEIPT_ID = "tqc_dev_1m_v2_instrumentation_equivalence/v1"
EXECUTION_MANIFEST_ID = "tqc_dev_1m_v2_execution_manifest/v1"
ATTEMPT_ID = "dev1m-v2-seed-95001-attempt-01"

MAX_PREFLIGHT_BYTES = 512 * 1024
MAX_WORKER_RECEIPT_BYTES = 512 * 1024
MAX_EXECUTION_MANIFEST_BYTES = 1024 * 1024
MAX_EXECUTABLE_BYTES = 64 * 1024 * 1024

PREFLIGHT_SOURCE_ROLES = (
    "v2_contract_source_sha256",
    "worker_entrypoint_source_sha256",
    "runtime_reinspection_source_sha256",
    "environment_instrumentation_equivalence_source_sha256",
)
EXECUTION_SOURCE_ROLES = (
    "training_adapter_source_sha256",
    "attempt_supervisor_source_sha256",
    "resource_monitor_source_sha256",
    "protected_locomotion_evaluator_source_sha256",
    "metric_core_source_sha256",
    "actor_export_source_sha256",
    "visual_capture_and_encoder_source_sha256",
)
SOURCE_ROLE_RELATIVE_PATHS = {
    "v2_contract_source_sha256": "experiments/tqc_development_contract_v2.py",
    "worker_entrypoint_source_sha256": "experiments/tqc_development_worker_v2.py",
    "runtime_reinspection_source_sha256": "experiments/tqc_development_runtime_v2.py",
    "environment_instrumentation_equivalence_source_sha256": (
        "experiments/tqc_instrumentation_equivalence_v2.py"
    ),
    "training_adapter_source_sha256": "experiments/tqc_development_training_v2.py",
    "attempt_supervisor_source_sha256": "experiments/tqc_development_supervisor_v2.py",
    "resource_monitor_source_sha256": "experiments/tqc_development_resource_v2.py",
    "protected_locomotion_evaluator_source_sha256": ("experiments/tqc_development_evaluator_v2.py"),
    "metric_core_source_sha256": "experiments/tqc_development_metrics.py",
    "actor_export_source_sha256": "experiments/tqc_actor_equivalence_v2.py",
    "visual_capture_and_encoder_source_sha256": "experiments/tqc_visual_evidence_v2.py",
}

_PREFLIGHT_ISSUER = object()
_RESERVATION_ISSUER = object()
_WORKER_RECEIPTS_ISSUER = object()
_MANIFEST_ISSUER = object()


def _expected_project_source_root() -> Path:
    """Return the installed package tree whose bytes the manifest may authorize."""

    return Path(__file__).resolve(strict=True).parents[1]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"{field} must be a lowercase SHA-256")
    return value


def _require_exact(value: object, expected: object, *, field: str) -> None:
    if type(value) is not type(expected):
        raise ExperimentContractError(f"{field} differs")
    if isinstance(expected, dict):
        if set(value) != set(expected):  # type: ignore[arg-type]
            raise ExperimentContractError(f"{field} keys differ")
        for key, child in expected.items():
            _require_exact(value[key], child, field=f"{field}.{key}")  # type: ignore[index]
        return
    if isinstance(expected, list):
        if len(value) != len(expected):  # type: ignore[arg-type]
            raise ExperimentContractError(f"{field} length differs")
        for index, child in enumerate(expected):
            _require_exact(value[index], child, field=f"{field}[{index}]")  # type: ignore[index]
        return
    if value != expected:
        raise ExperimentContractError(f"{field} differs")


def _require_keys(value: object, expected: set[str], *, field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ExperimentContractError(f"{field} must be an object")
    if set(value) != expected:
        raise ExperimentContractError(
            f"{field} keys differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExperimentContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ExperimentContractError(f"non-finite JSON value {value!r}")


def _decode_canonical_object(
    encoded: bytes, *, maximum_bytes: int, artifact: str
) -> dict[str, Any]:
    if type(encoded) is not bytes or not encoded or len(encoded) > maximum_bytes:
        raise ExperimentContractError(f"{artifact} bytes are absent or exceed the bound")
    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except ExperimentContractError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ExperimentContractError(f"cannot decode {artifact}: {exc}") from exc
    if type(value) is not dict:
        raise ExperimentContractError(f"{artifact} root must be an object")
    if canonical_json(value) != encoded:
        raise ExperimentContractError(f"{artifact} bytes are not canonical JSON UTF-8")
    return value


def _absolute_real_path(path: Path, *, artifact: str) -> Path:
    absolute = Path(os.path.abspath(path))
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise ExperimentContractError(f"cannot resolve {artifact}: {exc}") from exc
    if absolute != resolved:
        raise ExperimentContractError(f"{artifact} path or an ancestor is a symbolic link")
    return absolute


def _capture_regular_file(
    path: Path,
    *,
    artifact: str,
    maximum_bytes: int = MAX_SOURCE_FILE_BYTES,
) -> dict[str, Any]:
    absolute = _absolute_real_path(path, artifact=artifact)
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(absolute, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ExperimentContractError(f"{artifact} must be one regular, unlinked file")
        if before.st_size < 1 or before.st_size > maximum_bytes:
            raise ExperimentContractError(f"{artifact} size is outside its bound")
        digest = hashlib.sha256()
        observed = 0
        while observed <= maximum_bytes:
            chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - observed))
            if not chunk:
                break
            digest.update(chunk)
            observed += len(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_mode,
            before.st_nlink,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_mode,
            after.st_nlink,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if observed != before.st_size or identity_before != identity_after:
            raise ExperimentContractError(f"{artifact} changed while it was read")
        return {
            "absolute_path": str(absolute),
            "byte_count": observed,
            "sha256": digest.hexdigest(),
            "device": before.st_dev,
            "inode": before.st_ino,
            "mode_octal": f"{stat.S_IMODE(before.st_mode):04o}",
            "mtime_ns": before.st_mtime_ns,
            "ctime_ns": before.st_ctime_ns,
        }
    except ExperimentContractError:
        raise
    except OSError as exc:
        raise ExperimentContractError(f"cannot inspect {artifact}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _capture_work_directory(
    path: Path,
    *,
    reservation_receipt_path: Path,
    reservation_nonce: str,
    preflight_contract_path: Path,
    execution_manifest_path: Path,
) -> dict[str, Any]:
    _require_sha256(reservation_nonce, field="attempt reservation nonce")
    absolute = _absolute_real_path(path, artifact="claimed work directory")
    try:
        before = absolute.stat()
    except OSError as exc:
        raise ExperimentContractError(f"cannot inspect claimed work directory: {exc}") from exc
    if not stat.S_ISDIR(before.st_mode):
        raise ExperimentContractError("claimed work directory must be a directory")
    if stat.S_IMODE(before.st_mode) != 0o700:
        raise ExperimentContractError("claimed work directory mode must be 0700")
    if before.st_uid != os.geteuid():
        raise ExperimentContractError("claimed work directory must be owned by this process user")
    absolute_receipt = Path(os.path.abspath(reservation_receipt_path))
    if absolute_receipt.parent != absolute:
        raise ExperimentContractError("attempt reservation receipt must be in the work directory")
    absolute_preflight = Path(os.path.abspath(preflight_contract_path))
    if absolute_preflight.parent != absolute or absolute_preflight == absolute_receipt:
        raise ExperimentContractError("reserved preflight path must be a distinct work child")
    absolute_manifest = Path(os.path.abspath(execution_manifest_path))
    if absolute_manifest.parent != absolute or absolute_manifest in {
        absolute_receipt,
        absolute_preflight,
    }:
        raise ExperimentContractError("reserved manifest path must be a distinct work child")
    receipt = _capture_regular_file(
        absolute_receipt,
        artifact="attempt reservation receipt",
        maximum_bytes=MAX_WORKER_RECEIPT_BYTES,
    )
    if receipt["mode_octal"] != "0600":
        raise ExperimentContractError("attempt reservation receipt mode must be 0600")
    receipt_bytes = _read_regular_file_bytes(absolute_receipt, record=receipt)
    receipt_value = _decode_canonical_object(
        receipt_bytes,
        maximum_bytes=MAX_WORKER_RECEIPT_BYTES,
        artifact="attempt reservation receipt",
    )
    _require_exact(
        receipt_value,
        {
            "schema_version": 1,
            "receipt_id": ATTEMPT_RESERVATION_RECEIPT_ID,
            "attempt_id": ATTEMPT_ID,
            "reservation_nonce": reservation_nonce,
            "work_directory_absolute_path": str(absolute),
            "work_directory_mode_octal": "0700",
            "preflight_contract_absolute_path": str(absolute_preflight),
            "execution_manifest_absolute_path": str(absolute_manifest),
            "status": "reserved_pre_worker",
            "authorizes_model_construction": False,
            "behavioral_evidence": False,
        },
        field="attempt reservation receipt",
    )
    identity = {
        "absolute_path": str(absolute),
        "device": before.st_dev,
        "inode": before.st_ino,
        "owner_uid": before.st_uid,
        "owner_gid": before.st_gid,
        "mode_octal": "0700",
        "fresh_exclusive_reservation": True,
        "reservation_nonce": reservation_nonce,
        "preflight_contract_absolute_path": str(absolute_preflight),
        "execution_manifest_absolute_path": str(absolute_manifest),
        "reservation_receipt": receipt,
    }
    identity["identity_sha256"] = _sha256(canonical_json(identity))
    return identity


@dataclass(frozen=True, slots=True)
class ReservedTQCAttemptDirectoryV2:
    """Process-local proof that the attempt directory was created exclusively."""

    path: Path
    reservation_receipt_path: Path
    preflight_contract_path: Path
    execution_manifest_path: Path
    reservation_nonce: str
    identity_sha256: str
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _RESERVATION_ISSUER:
            raise ExperimentContractError(
                "v2 attempt reservations may only be issued by exclusive creation"
            )
        work = _capture_work_directory(
            self.path,
            reservation_receipt_path=self.reservation_receipt_path,
            reservation_nonce=self.reservation_nonce,
            preflight_contract_path=self.preflight_contract_path,
            execution_manifest_path=self.execution_manifest_path,
        )
        _require_exact(work["identity_sha256"], self.identity_sha256, field="reservation identity")


def reserve_tqc_attempt_directory_v2(
    work_directory: Path,
    *,
    reservation_receipt_name: str = "attempt.in-progress.json",
    preflight_contract_name: str = "preflight.contract.json",
    execution_manifest_name: str = "execution.manifest.json",
) -> ReservedTQCAttemptDirectoryV2:
    """Create the one private work directory and its non-authoritative receipt."""

    requested = Path(work_directory)
    if requested.name in {"", ".", ".."} or "\0" in requested.name:
        raise ExperimentContractError("attempt directory must have one safe final component")
    if (
        reservation_receipt_name in {"", ".", ".."}
        or Path(reservation_receipt_name).name != reservation_receipt_name
        or "\0" in reservation_receipt_name
    ):
        raise ExperimentContractError("attempt reservation receipt name is unsafe")
    if (
        preflight_contract_name in {"", ".", ".."}
        or Path(preflight_contract_name).name != preflight_contract_name
        or "\0" in preflight_contract_name
        or preflight_contract_name == reservation_receipt_name
    ):
        raise ExperimentContractError("preflight contract name is unsafe")
    if (
        execution_manifest_name in {"", ".", ".."}
        or Path(execution_manifest_name).name != execution_manifest_name
        or "\0" in execution_manifest_name
        or execution_manifest_name in {reservation_receipt_name, preflight_contract_name}
    ):
        raise ExperimentContractError("execution manifest name is unsafe")
    absolute = Path(os.path.abspath(requested))
    parent = _absolute_real_path(absolute.parent, artifact="attempt directory parent")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    parent_descriptor: int | None = None
    child_descriptor: int | None = None
    created = False
    try:
        parent_descriptor = os.open(parent, flags)
        parent_state = os.fstat(parent_descriptor)
        current_parent_state = parent.stat()
        if (
            parent_state.st_dev != current_parent_state.st_dev
            or parent_state.st_ino != current_parent_state.st_ino
            or not stat.S_ISDIR(parent_state.st_mode)
        ):
            raise ExperimentContractError("attempt directory parent changed during reservation")
        os.mkdir(absolute.name, mode=0o700, dir_fd=parent_descriptor)
        created = True
        child_descriptor = os.open(absolute.name, flags, dir_fd=parent_descriptor)
        state = os.fstat(child_descriptor)
        if not stat.S_ISDIR(state.st_mode) or state.st_uid != os.geteuid():
            raise ExperimentContractError("exclusive attempt directory identity differs")
        os.fchmod(child_descriptor, 0o700)
        os.fsync(parent_descriptor)
    except FileExistsError as exc:
        raise ExperimentContractError("attempt directory already exists") from exc
    except ExperimentContractError:
        raise
    except OSError as exc:
        raise ExperimentContractError(f"cannot reserve attempt directory: {exc}") from exc
    finally:
        if child_descriptor is not None:
            os.close(child_descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)
    if not created:
        raise ExperimentContractError("attempt directory was not reserved")
    nonce = secrets.token_hex(32)
    receipt_path = absolute / reservation_receipt_name
    preflight_path = absolute / preflight_contract_name
    manifest_path = absolute / execution_manifest_name
    receipt = {
        "schema_version": 1,
        "receipt_id": ATTEMPT_RESERVATION_RECEIPT_ID,
        "attempt_id": ATTEMPT_ID,
        "reservation_nonce": nonce,
        "work_directory_absolute_path": str(absolute),
        "work_directory_mode_octal": "0700",
        "preflight_contract_absolute_path": str(preflight_path),
        "execution_manifest_absolute_path": str(manifest_path),
        "status": "reserved_pre_worker",
        "authorizes_model_construction": False,
        "behavioral_evidence": False,
    }
    publish_bytes_without_overwrite(receipt_path, canonical_json(receipt))
    work = _capture_work_directory(
        absolute,
        reservation_receipt_path=receipt_path,
        reservation_nonce=nonce,
        preflight_contract_path=preflight_path,
        execution_manifest_path=manifest_path,
    )
    if work["device"] != state.st_dev or work["inode"] != state.st_ino:
        raise ExperimentContractError("attempt directory path changed during reservation")
    return ReservedTQCAttemptDirectoryV2(
        path=absolute,
        reservation_receipt_path=receipt_path,
        preflight_contract_path=preflight_path,
        execution_manifest_path=manifest_path,
        reservation_nonce=nonce,
        identity_sha256=work["identity_sha256"],
        _issuer=_RESERVATION_ISSUER,
    )


def _capture_source_tree(root: Path) -> dict[str, Any]:
    absolute = _absolute_real_path(root, artifact="project Python source tree")
    if not absolute.is_dir():
        raise ExperimentContractError("project Python source tree must be a directory")
    candidates = sorted(
        absolute.rglob("*.py"),
        key=lambda candidate: candidate.relative_to(absolute).as_posix(),
    )
    if not candidates or len(candidates) > MAX_SOURCE_TREE_FILES:
        raise ExperimentContractError("project Python source file count is outside the bound")
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    total_bytes = 0
    for candidate in candidates:
        record = _capture_regular_file(candidate, artifact="project Python source file")
        relative = candidate.relative_to(absolute).as_posix()
        total_bytes += record["byte_count"]
        if total_bytes > MAX_SOURCE_TREE_BYTES:
            raise ExperimentContractError("project Python source bytes exceed the aggregate bound")
        relative_bytes = relative.encode("utf-8")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(record["byte_count"].to_bytes(8, "big"))
        file_bytes = _read_regular_file_bytes(candidate, record=record)
        digest.update(file_bytes)
        records.append(
            {
                "relative_path": relative,
                "byte_count": record["byte_count"],
                "sha256": record["sha256"],
                "device": record["device"],
                "inode": record["inode"],
                "mode_octal": record["mode_octal"],
                "mtime_ns": record["mtime_ns"],
                "ctime_ns": record["ctime_ns"],
            }
        )
    second_candidates = sorted(
        candidate.relative_to(absolute).as_posix() for candidate in absolute.rglob("*.py")
    )
    if second_candidates != [record["relative_path"] for record in records]:
        raise ExperimentContractError("project Python source membership changed during inspection")
    return {
        "absolute_path": str(absolute),
        "file_count": len(records),
        "total_byte_count": total_bytes,
        "sha256": digest.hexdigest(),
        "files": records,
    }


def _read_regular_file_bytes(path: Path, *, record: Mapping[str, Any]) -> bytes:
    maximum = int(record["byte_count"])
    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        flags |= getattr(os, "O_NONBLOCK", 0)
        descriptor = os.open(path, flags)
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        encoded = b"".join(chunks)
        state = os.fstat(descriptor)
        if (
            len(encoded) != maximum
            or _sha256(encoded) != record["sha256"]
            or state.st_dev != record["device"]
            or state.st_ino != record["inode"]
            or state.st_mtime_ns != record["mtime_ns"]
            or state.st_ctime_ns != record["ctime_ns"]
        ):
            raise ExperimentContractError("project Python source changed during tree hashing")
        return encoded
    except ExperimentContractError:
        raise
    except OSError as exc:
        raise ExperimentContractError(f"cannot hash project Python source: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _require_paths_inside_tree(
    records: Mapping[str, dict[str, Any]],
    *,
    tree: Mapping[str, Any],
) -> None:
    root = Path(tree["absolute_path"])
    tree_by_path = {str(root / record["relative_path"]): record for record in tree["files"]}
    for role, record in records.items():
        expected_relative = SOURCE_ROLE_RELATIVE_PATHS.get(role)
        if expected_relative is None:
            raise ExperimentContractError(f"{role} has no canonical source path")
        try:
            observed_relative = Path(record["absolute_path"]).relative_to(root).as_posix()
        except ValueError as exc:
            raise ExperimentContractError(
                f"{role} is outside the bound project source tree"
            ) from exc
        _require_exact(
            observed_relative,
            expected_relative,
            field=f"{role} canonical relative path",
        )
        tree_record = tree_by_path.get(record["absolute_path"])
        if tree_record is None:
            raise ExperimentContractError(f"{role} is outside the bound project source tree")
        for field in (
            "byte_count",
            "sha256",
            "device",
            "inode",
            "mode_octal",
            "mtime_ns",
            "ctime_ns",
        ):
            _require_exact(record[field], tree_record[field], field=f"{role}.{field}")


def _runtime_requirements(design: LoadedTQCDevelopmentDesignV2) -> dict[str, Any]:
    return design.to_dict()["training_projection"]["runtime_requirements"]


def _observe_preflight_runtime_identity() -> dict[str, Any]:
    """Measure the host facts available before any simulator import."""

    return {
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "platform_release": platform.release(),
        **_host_hardware_identity(),
    }


def _expected_runtime_environment(
    design: LoadedTQCDevelopmentDesignV2,
    *,
    reset_observation_sha256: object,
) -> dict[str, Any]:
    _require_sha256(reset_observation_sha256, field="runtime reset observation")
    environment = design.to_dict()["training_projection"]["environment"]
    return {
        "environment_id": environment["environment_id"],
        "plain_wrapper_types": environment["wrapper_order_outer_to_inner"],
        "instrumented_wrapper_types": [
            *environment["wrapper_order_outer_to_inner"][:-1],
            "oracle_composition.envs.humanoid.SubstepContactHumanoidEnv",
        ],
        "max_episode_steps": environment["max_episode_steps"],
        "terminate_when_unhealthy": environment["terminate_when_unhealthy"],
        "reset_noise_scale": environment["reset_noise_scale"],
        "exclude_current_positions_from_observation": environment[
            "exclude_current_positions_from_observation"
        ],
        "frame_skip": environment["frame_skip"],
        "control_period_seconds": environment["control_period_seconds"],
        "observation_shape": environment["observation_shape"],
        "observation_dtype": environment["observation_dtype"],
        "action_shape": environment["action_shape"],
        "action_dtype": environment["action_dtype"],
        "render_mode": environment["render_mode"],
        "reset_observation_sha256": reset_observation_sha256,
    }


def _validate_primary_authorities(
    design: LoadedTQCDevelopmentDesignV2,
    e0: ValidatedV2E0ReuseEvidence,
    review: ValidatedIndependentReviewReceiptV2,
) -> None:
    if type(design) is not LoadedTQCDevelopmentDesignV2:
        raise ExperimentContractError("manifest admission requires the strict-loaded v2 design")
    if type(e0) is not ValidatedV2E0ReuseEvidence:
        raise ExperimentContractError("manifest admission requires validated v2 E0 reuse evidence")
    if type(review) is not ValidatedIndependentReviewReceiptV2:
        raise ExperimentContractError("manifest admission requires the exact independent review")
    _require_exact(design.file_sha256, DESIGN_FILE_SHA256, field="design file SHA-256")
    _require_exact(design.semantic_sha256, DESIGN_SEMANTIC_SHA256, field="design semantic SHA-256")
    _require_exact(e0.training_projection_sha256, TRAINING_PROJECTION_SHA256, field="E0 projection")
    _require_exact(
        e0.e0_design_file_sha256,
        E0_DESIGN_FILE_SHA256,
        field="E0 design file",
    )
    _require_exact(
        e0.e0_receipt_file_sha256,
        E0_RECEIPT_FILE_SHA256,
        field="E0 receipt file",
    )
    _require_exact(
        e0.e0_runtime_sha256,
        E0_RUNTIME_SHA256,
        field="E0 runtime",
    )
    _require_exact(
        e0.v1_design_file_sha256,
        V1_DESIGN_FILE_SHA256,
        field="superseded v1 design",
    )
    _require_exact(e0.resource_prior_accepted, True, field="E0 resource prior")
    _require_exact(e0.authorizes_training, False, field="E0 authority ceiling")
    _require_exact(review.reviewed_design_file_sha256, DESIGN_FILE_SHA256, field="review design")
    _require_exact(
        review.reviewed_design_semantic_sha256,
        DESIGN_SEMANTIC_SHA256,
        field="review design semantics",
    )
    _require_exact(
        review.reviewed_e0_audit_file_sha256,
        AUDIT_FILE_SHA256,
        field="review E0 audit",
    )
    _require_exact(
        review.reviewed_e0_audit_semantic_sha256,
        AUDIT_SEMANTIC_SHA256,
        field="review E0 audit semantics",
    )
    _require_exact(
        review.reviewed_training_projection_sha256,
        TRAINING_PROJECTION_SHA256,
        field="review projection",
    )
    _require_exact(review.contract_review_go, True, field="independent review GO")
    _require_exact(review.authorizes_training, False, field="review authority ceiling")


def _exact_path_mapping(
    paths: Mapping[str, Path],
    roles: Sequence[str],
    *,
    field: str,
) -> dict[str, Path]:
    if not isinstance(paths, Mapping) or set(paths) != set(roles):
        observed = set(paths) if isinstance(paths, Mapping) else set()
        raise ExperimentContractError(
            f"{field} keys differ: missing={sorted(set(roles) - observed)}, "
            f"extra={sorted(observed - set(roles))}"
        )
    return {role: Path(paths[role]) for role in roles}


def _preflight_payload(
    design: LoadedTQCDevelopmentDesignV2,
    e0: ValidatedV2E0ReuseEvidence,
    review: ValidatedIndependentReviewReceiptV2,
    *,
    output_path: Path,
    source_paths: Mapping[str, Path],
    project_source_root: Path,
    dependency_lock_path: Path,
    python_executable_path: Path,
    reservation: ReservedTQCAttemptDirectoryV2,
) -> dict[str, Any]:
    _validate_primary_authorities(design, e0, review)
    if type(reservation) is not ReservedTQCAttemptDirectoryV2:
        raise ExperimentContractError(
            "preflight publication requires an exclusive attempt-directory reservation"
        )
    source_paths = _exact_path_mapping(
        source_paths,
        PREFLIGHT_SOURCE_ROLES,
        field="preflight source paths",
    )
    work = _capture_work_directory(
        reservation.path,
        reservation_receipt_path=reservation.reservation_receipt_path,
        reservation_nonce=reservation.reservation_nonce,
        preflight_contract_path=reservation.preflight_contract_path,
        execution_manifest_path=reservation.execution_manifest_path,
    )
    _require_exact(
        work["identity_sha256"],
        reservation.identity_sha256,
        field="reserved work-directory identity",
    )
    absolute_output = Path(os.path.abspath(output_path))
    if absolute_output != reservation.preflight_contract_path:
        raise ExperimentContractError("preflight contract path differs from its reservation")
    expected_source_root = _expected_project_source_root()
    observed_source_root = _absolute_real_path(
        project_source_root,
        artifact="project Python source tree",
    )
    if observed_source_root != expected_source_root:
        raise ExperimentContractError(
            "project Python source tree is not the imported oracle_composition source root"
        )
    tree = _capture_source_tree(project_source_root)
    source_records = {
        role: _capture_regular_file(path, artifact=role) for role, path in source_paths.items()
    }
    _require_paths_inside_tree(source_records, tree=tree)
    dependency = _capture_regular_file(dependency_lock_path, artifact="dependency lock")
    executable_realpath = Path(python_executable_path).resolve(strict=True)
    executable = _capture_regular_file(
        executable_realpath,
        artifact="Python executable",
        maximum_bytes=MAX_EXECUTABLE_BYTES,
    )
    runtime = _runtime_requirements(design)
    observed_runtime_identity = _observe_preflight_runtime_identity()
    for field, observed in observed_runtime_identity.items():
        _require_exact(observed, runtime[field], field=f"observed preflight {field}")
    _require_exact(dependency["sha256"], runtime["dependency_lock_sha256"], field="dependency lock")
    _require_exact(
        source_records["v2_contract_source_sha256"]["sha256"],
        review.reviewed_v2_contract_source_sha256,
        field="reviewed v2 contract source",
    )
    observed_platform = {
        key: observed_runtime_identity[key]
        for key in (
            "python_version",
            "platform_system",
            "platform_machine",
            "platform_release",
        )
    }
    host_hardware = {
        key: observed_runtime_identity[key]
        for key in ("cpu_model", "hardware_model", "logical_cpu_count", "total_memory_bytes")
    }
    bindings: dict[str, Any] = {
        "design_file_sha256": DESIGN_FILE_SHA256,
        "design_semantic_sha256": DESIGN_SEMANTIC_SHA256,
        "v2_contract_source_sha256": source_records["v2_contract_source_sha256"]["sha256"],
        "e0_reuse_audit_file_sha256": AUDIT_FILE_SHA256,
        "e0_v2_training_projection_sha256": TRAINING_PROJECTION_SHA256,
        "dependency_lock_sha256": dependency["sha256"],
        "project_python_source_tree_sha256": tree["sha256"],
        "worker_entrypoint_source_sha256": source_records["worker_entrypoint_source_sha256"][
            "sha256"
        ],
        "runtime_reinspection_source_sha256": source_records["runtime_reinspection_source_sha256"][
            "sha256"
        ],
        "environment_instrumentation_equivalence_source_sha256": source_records[
            "environment_instrumentation_equivalence_source_sha256"
        ]["sha256"],
        "python_executable_realpath": executable["absolute_path"],
        "python_executable_file_sha256": executable["sha256"],
        **observed_platform,
        "host_hardware_identity": host_hardware,
        "independent_review_receipt_content_sha256": review.content_sha256,
        "claimed_work_directory_identity": work["identity_sha256"],
    }
    expected_roles = set(
        design.to_dict()["execution_manifest"]["two_phase_preflight_protocol"][
            "preflight_contract_must_bind"
        ]
    )
    _require_keys(bindings, expected_roles, field="preflight bindings")
    return {
        "schema_version": 1,
        "contract_id": PREFLIGHT_CONTRACT_ID,
        "attempt_id": ATTEMPT_ID,
        "attempt_nonce": reservation.reservation_nonce,
        "contract_absolute_path": str(absolute_output),
        "requested": {
            "design_id": DESIGN_ID,
            "attempt_id": ATTEMPT_ID,
            "runtime_requirements": runtime,
            "claimed_work_directory_absolute_path": work["absolute_path"],
        },
        "observed": {
            "pre_worker_runtime_identity": observed_runtime_identity,
            "source_files": source_records,
            "dependency_lock": dependency,
            "python_executable": executable,
            "project_python_source_tree": tree,
            "claimed_work_directory": work,
        },
        "bindings": bindings,
        "review_authority": {
            "contract_review_go": True,
            "authorizes_training": False,
            "behavioral_evidence": False,
        },
        "e0_authority": {
            "resource_prior_accepted": True,
            "authorizes_training": False,
        },
    }


def _validate_preflight_payload(value: dict[str, Any]) -> None:
    _require_keys(
        value,
        {
            "schema_version",
            "contract_id",
            "attempt_id",
            "attempt_nonce",
            "contract_absolute_path",
            "requested",
            "observed",
            "bindings",
            "review_authority",
            "e0_authority",
        },
        field="preflight contract",
    )
    _require_exact(value["schema_version"], 1, field="preflight schema version")
    _require_exact(value["contract_id"], PREFLIGHT_CONTRACT_ID, field="preflight contract id")
    _require_exact(value["attempt_id"], ATTEMPT_ID, field="preflight attempt id")
    _require_sha256(value["attempt_nonce"], field="preflight attempt nonce")
    _require_keys(
        value["requested"],
        {
            "design_id",
            "attempt_id",
            "runtime_requirements",
            "claimed_work_directory_absolute_path",
        },
        field="preflight requested values",
    )
    _require_keys(
        value["observed"],
        {
            "pre_worker_runtime_identity",
            "source_files",
            "dependency_lock",
            "python_executable",
            "project_python_source_tree",
            "claimed_work_directory",
        },
        field="preflight observed values",
    )
    _require_exact(
        value["review_authority"],
        {"contract_review_go": True, "authorizes_training": False, "behavioral_evidence": False},
        field="preflight review authority",
    )
    _require_exact(
        value["e0_authority"],
        {"resource_prior_accepted": True, "authorizes_training": False},
        field="preflight E0 authority",
    )


@dataclass(frozen=True, slots=True)
class ValidatedTQCPreflightContractV2:
    """Final pre-worker contract; it does not authorize model construction."""

    canonical_bytes: bytes
    sha256: str
    byte_count: int
    attempt_id: str
    attempt_nonce: str
    design_file_sha256: str
    design_semantic_sha256: str
    training_projection_sha256: str
    project_python_source_tree_sha256: str
    dependency_lock_sha256: str
    independent_review_receipt_content_sha256: str
    claimed_work_directory_identity: str
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _PREFLIGHT_ISSUER:
            raise ExperimentContractError("v2 preflight authority may only be issued by validation")
        value = _decode_canonical_object(
            self.canonical_bytes,
            maximum_bytes=MAX_PREFLIGHT_BYTES,
            artifact="validated v2 preflight contract",
        )
        _validate_preflight_payload(value)
        _require_exact(_sha256(self.canonical_bytes), self.sha256, field="preflight SHA-256")
        _require_exact(len(self.canonical_bytes), self.byte_count, field="preflight byte count")
        for field, expected in (
            ("attempt_id", value["attempt_id"]),
            ("attempt_nonce", value["attempt_nonce"]),
            ("design_file_sha256", value["bindings"]["design_file_sha256"]),
            ("design_semantic_sha256", value["bindings"]["design_semantic_sha256"]),
            (
                "training_projection_sha256",
                value["bindings"]["e0_v2_training_projection_sha256"],
            ),
            (
                "project_python_source_tree_sha256",
                value["bindings"]["project_python_source_tree_sha256"],
            ),
            ("dependency_lock_sha256", value["bindings"]["dependency_lock_sha256"]),
            (
                "independent_review_receipt_content_sha256",
                value["bindings"]["independent_review_receipt_content_sha256"],
            ),
            (
                "claimed_work_directory_identity",
                value["bindings"]["claimed_work_directory_identity"],
            ),
        ):
            _require_exact(getattr(self, field), expected, field=f"preflight authority {field}")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)


def _issue_preflight(encoded: bytes) -> ValidatedTQCPreflightContractV2:
    value = _decode_canonical_object(
        encoded,
        maximum_bytes=MAX_PREFLIGHT_BYTES,
        artifact="v2 preflight contract",
    )
    _validate_preflight_payload(value)
    bindings = value["bindings"]
    return ValidatedTQCPreflightContractV2(
        canonical_bytes=encoded,
        sha256=_sha256(encoded),
        byte_count=len(encoded),
        attempt_id=value["attempt_id"],
        attempt_nonce=value["attempt_nonce"],
        design_file_sha256=bindings["design_file_sha256"],
        design_semantic_sha256=bindings["design_semantic_sha256"],
        training_projection_sha256=bindings["e0_v2_training_projection_sha256"],
        project_python_source_tree_sha256=bindings["project_python_source_tree_sha256"],
        dependency_lock_sha256=bindings["dependency_lock_sha256"],
        independent_review_receipt_content_sha256=bindings[
            "independent_review_receipt_content_sha256"
        ],
        claimed_work_directory_identity=bindings["claimed_work_directory_identity"],
        _issuer=_PREFLIGHT_ISSUER,
    )


def _read_bound_canonical_file(path: Path, *, maximum_bytes: int, artifact: str) -> bytes:
    record = _capture_regular_file(path, artifact=artifact, maximum_bytes=maximum_bytes)
    if record["mode_octal"] != "0600":
        raise ExperimentContractError(f"{artifact} mode must be 0600")
    return _read_regular_file_bytes(path, record=record)


def publish_tqc_preflight_contract_v2(
    output_path: Path,
    design: LoadedTQCDevelopmentDesignV2,
    e0: ValidatedV2E0ReuseEvidence,
    review: ValidatedIndependentReviewReceiptV2,
    *,
    source_paths: Mapping[str, Path],
    project_source_root: Path,
    dependency_lock_path: Path,
    python_executable_path: Path,
    reservation: ReservedTQCAttemptDirectoryV2,
) -> ValidatedTQCPreflightContractV2:
    """Publish one exclusive preflight contract before any worker starts."""

    payload = _preflight_payload(
        design,
        e0,
        review,
        output_path=output_path,
        source_paths=source_paths,
        project_source_root=project_source_root,
        dependency_lock_path=dependency_lock_path,
        python_executable_path=python_executable_path,
        reservation=reservation,
    )
    encoded = canonical_json(payload)
    if len(encoded) > MAX_PREFLIGHT_BYTES:
        raise ExperimentContractError("v2 preflight contract exceeds its byte bound")
    published = publish_bytes_without_overwrite(output_path, encoded)
    persisted = _read_bound_canonical_file(
        published.path,
        maximum_bytes=MAX_PREFLIGHT_BYTES,
        artifact="persisted v2 preflight contract",
    )
    if published.sha256 != _sha256(persisted) or published.byte_count != len(persisted):
        raise ExperimentContractError("persisted v2 preflight contract differs after publication")
    expected = _preflight_payload(
        design,
        e0,
        review,
        output_path=output_path,
        source_paths=source_paths,
        project_source_root=project_source_root,
        dependency_lock_path=dependency_lock_path,
        python_executable_path=python_executable_path,
        reservation=reservation,
    )
    _require_exact(persisted, canonical_json(expected), field="revalidated preflight bytes")
    return _issue_preflight(persisted)


def validate_persisted_tqc_preflight_contract_v2(
    preflight_path: Path,
    design: LoadedTQCDevelopmentDesignV2,
    e0: ValidatedV2E0ReuseEvidence,
    review: ValidatedIndependentReviewReceiptV2,
    *,
    source_paths: Mapping[str, Path],
    project_source_root: Path,
    dependency_lock_path: Path,
    python_executable_path: Path,
    reservation: ReservedTQCAttemptDirectoryV2,
) -> ValidatedTQCPreflightContractV2:
    """Reload and reverify a finalized preflight contract without repairing it."""

    persisted = _read_bound_canonical_file(
        preflight_path,
        maximum_bytes=MAX_PREFLIGHT_BYTES,
        artifact="persisted v2 preflight contract",
    )
    _decode_canonical_object(
        persisted,
        maximum_bytes=MAX_PREFLIGHT_BYTES,
        artifact="persisted v2 preflight contract",
    )
    expected = _preflight_payload(
        design,
        e0,
        review,
        output_path=preflight_path,
        source_paths=source_paths,
        project_source_root=project_source_root,
        dependency_lock_path=dependency_lock_path,
        python_executable_path=python_executable_path,
        reservation=reservation,
    )
    _require_exact(persisted, canonical_json(expected), field="persisted preflight contract")
    return _issue_preflight(persisted)


def revalidate_tqc_preflight_contract_for_worker_v2(
    preflight_path: Path,
    design: LoadedTQCDevelopmentDesignV2,
    e0: ValidatedV2E0ReuseEvidence,
    review: ValidatedIndependentReviewReceiptV2,
    *,
    expected_preflight_sha256: str,
    expected_design_file_sha256: str,
    expected_design_semantic_sha256: str,
    expected_e0_audit_file_sha256: str,
    expected_training_projection_sha256: str,
    expected_claimed_work_directory_absolute_path: Path,
) -> ValidatedTQCPreflightContractV2:
    """Re-admit parent preflight bytes inside the fresh worker process."""

    _require_sha256(expected_preflight_sha256, field="expected preflight SHA-256")
    for value, expected, field in (
        (expected_design_file_sha256, DESIGN_FILE_SHA256, "bootstrap design file"),
        (
            expected_design_semantic_sha256,
            DESIGN_SEMANTIC_SHA256,
            "bootstrap design semantics",
        ),
        (expected_e0_audit_file_sha256, AUDIT_FILE_SHA256, "bootstrap E0 audit"),
        (
            expected_training_projection_sha256,
            TRAINING_PROJECTION_SHA256,
            "bootstrap training projection",
        ),
    ):
        _require_exact(value, expected, field=field)
    persisted = _read_bound_canonical_file(
        preflight_path,
        maximum_bytes=MAX_PREFLIGHT_BYTES,
        artifact="worker-read v2 preflight contract",
    )
    _require_exact(
        _sha256(persisted),
        expected_preflight_sha256,
        field="worker preflight SHA-256",
    )
    value = _decode_canonical_object(
        persisted,
        maximum_bytes=MAX_PREFLIGHT_BYTES,
        artifact="worker-read v2 preflight contract",
    )
    _validate_preflight_payload(value)
    observed = value["observed"]
    recorded_work = observed["claimed_work_directory"]
    expected_work = _absolute_real_path(
        expected_claimed_work_directory_absolute_path,
        artifact="bootstrap claimed work directory",
    )
    _require_exact(
        recorded_work["absolute_path"],
        str(expected_work),
        field="worker preflight claimed work directory",
    )
    _require_exact(
        value["contract_absolute_path"],
        str(Path(os.path.abspath(preflight_path))),
        field="worker preflight path",
    )
    reservation = ReservedTQCAttemptDirectoryV2(
        path=expected_work,
        reservation_receipt_path=Path(recorded_work["reservation_receipt"]["absolute_path"]),
        preflight_contract_path=Path(recorded_work["preflight_contract_absolute_path"]),
        execution_manifest_path=Path(recorded_work["execution_manifest_absolute_path"]),
        reservation_nonce=recorded_work["reservation_nonce"],
        identity_sha256=recorded_work["identity_sha256"],
        _issuer=_RESERVATION_ISSUER,
    )
    source_paths = {
        role: Path(record["absolute_path"]) for role, record in observed["source_files"].items()
    }
    revalidated = validate_persisted_tqc_preflight_contract_v2(
        preflight_path,
        design,
        e0,
        review,
        source_paths=source_paths,
        project_source_root=Path(observed["project_python_source_tree"]["absolute_path"]),
        dependency_lock_path=Path(observed["dependency_lock"]["absolute_path"]),
        python_executable_path=Path(observed["python_executable"]["absolute_path"]),
        reservation=reservation,
    )
    _require_exact(
        revalidated.sha256,
        expected_preflight_sha256,
        field="worker-issued preflight authority",
    )
    return revalidated


def worker_bootstrap_receipt_bytes_v2(
    preflight: ValidatedTQCPreflightContractV2,
    *,
    worker_process_start_monotonic_seconds: float,
) -> bytes:
    """Frame this worker's OS identity; validation alone grants no authority."""

    if type(preflight) is not ValidatedTQCPreflightContractV2:
        raise ExperimentContractError("worker bootstrap requires local preflight authority")
    if (
        type(worker_process_start_monotonic_seconds) is not float
        or not math.isfinite(worker_process_start_monotonic_seconds)
        or worker_process_start_monotonic_seconds < 0.0
        or worker_process_start_monotonic_seconds > time.perf_counter()
    ):
        raise ExperimentContractError("worker start monotonic time is invalid")
    worker_pid, worker_pgid, worker_sid = _current_worker_process_identity()
    value = preflight.to_dict()
    worker_source = value["bindings"]["worker_entrypoint_source_sha256"]
    return canonical_json(
        {
            "schema_version": 1,
            "receipt_id": WORKER_BOOTSTRAP_RECEIPT_ID,
            "attempt_id": preflight.attempt_id,
            "attempt_nonce": preflight.attempt_nonce,
            "claimed_work_directory_identity": preflight.claimed_work_directory_identity,
            "worker_pid": worker_pid,
            "worker_pgid": worker_pgid,
            "worker_sid": worker_sid,
            "worker_pid_equals_pgid_equals_sid": worker_pid == worker_pgid == worker_sid,
            "worker_process_start_monotonic_seconds": worker_process_start_monotonic_seconds,
            "worker_entrypoint_source_sha256": worker_source,
            "preflight_contract_sha256": preflight.sha256,
        }
    )


def _current_worker_process_identity() -> tuple[int, int, int]:
    worker_pid = os.getpid()
    try:
        worker_pgid = os.getpgid(worker_pid)
        worker_sid = os.getsid(worker_pid)
    except OSError as exc:
        raise ExperimentContractError("worker cannot inspect its process identity") from exc
    return worker_pid, worker_pgid, worker_sid


def _worker_preflight_delivery_type() -> type:
    try:
        from .tqc_development_supervisor_v2 import (
            ValidatedTQCWorkerPreflightDeliveryV2,
        )
    except (ImportError, AttributeError) as exc:
        raise ExperimentContractError(
            "the supervisor-issued v2 worker preflight delivery authority is unavailable"
        ) from exc
    return ValidatedTQCWorkerPreflightDeliveryV2


def _instrumentation_receipt_type() -> type:
    try:
        from .tqc_instrumentation_equivalence_v2 import (
            TQCInstrumentationEquivalenceReceiptV2,
        )
    except (ImportError, AttributeError) as exc:
        raise ExperimentContractError(
            "the exact v2 instrumentation-equivalence receipt authority is unavailable"
        ) from exc
    return TQCInstrumentationEquivalenceReceiptV2


def _runtime_receipt_validator():
    try:
        from .tqc_development_runtime_v2 import validate_tqc_development_runtime_receipt_v2
    except (ImportError, AttributeError) as exc:
        raise ExperimentContractError(
            "the exact v2 host-runtime byte re-admission validator is unavailable"
        ) from exc
    return validate_tqc_development_runtime_receipt_v2


def _instrumentation_receipt_validator():
    try:
        from .tqc_instrumentation_equivalence_v2 import (
            validate_tqc_instrumentation_equivalence_receipt_v2,
        )
    except (ImportError, AttributeError) as exc:
        raise ExperimentContractError(
            "the exact v2 instrumentation byte re-admission validator is unavailable"
        ) from exc
    return validate_tqc_instrumentation_equivalence_receipt_v2


def _worker_receipt_binding(
    preflight: ValidatedTQCPreflightContractV2,
    *,
    worker_pid: object,
    worker_process_start_monotonic_seconds: object,
) -> dict[str, Any]:
    return {
        "attempt_id": preflight.attempt_id,
        "attempt_nonce": preflight.attempt_nonce,
        "worker_pid": worker_pid,
        "worker_process_start_monotonic_seconds": (worker_process_start_monotonic_seconds),
        "preflight_contract_sha256": preflight.sha256,
        "claimed_work_directory_identity": preflight.claimed_work_directory_identity,
        "project_python_source_tree_sha256": (preflight.project_python_source_tree_sha256),
    }


@dataclass(frozen=True, slots=True)
class ValidatedTQCWorkerPreflightReceiptsV2:
    """Three same-attempt worker receipts admitted before model construction."""

    attempt_id: str
    attempt_nonce: str
    preflight_contract_sha256: str
    claimed_work_directory_identity: str
    project_python_source_tree_sha256: str
    worker_bootstrap_identity_receipt_sha256: str
    host_runtime_receipt_sha256: str
    instrumentation_equivalence_receipt_sha256: str
    receipt_set_sha256: str
    worker_pid: int
    worker_pgid: int
    worker_sid: int
    worker_process_start_monotonic_seconds: float
    preflight_receipts_message_sequence_index: int
    channel_transcript_frame_count: int
    channel_transcript_sha256: str
    channel_parent_send_sequence: int
    channel_parent_receive_sequence: int
    worker_bootstrap_canonical_bytes: bytes
    host_runtime_canonical_bytes: bytes
    instrumentation_canonical_bytes: bytes
    receipt_set_canonical_bytes: bytes
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _WORKER_RECEIPTS_ISSUER:
            raise ExperimentContractError(
                "worker preflight receipts may only be issued by validation"
            )
        for field in (
            "preflight_contract_sha256",
            "claimed_work_directory_identity",
            "project_python_source_tree_sha256",
            "worker_bootstrap_identity_receipt_sha256",
            "host_runtime_receipt_sha256",
            "instrumentation_equivalence_receipt_sha256",
            "receipt_set_sha256",
            "channel_transcript_sha256",
        ):
            _require_sha256(getattr(self, field), field=field)
        if not (
            type(self.worker_pid) is int
            and type(self.worker_pgid) is int
            and type(self.worker_sid) is int
            and self.worker_pid == self.worker_pgid == self.worker_sid
            and self.worker_pid > 1
        ):
            raise ExperimentContractError("validated worker process identity differs")
        if (
            type(self.worker_process_start_monotonic_seconds) is not float
            or not math.isfinite(self.worker_process_start_monotonic_seconds)
            or self.worker_process_start_monotonic_seconds < 0.0
        ):
            raise ExperimentContractError("validated worker start time is invalid")
        if (
            type(self.preflight_receipts_message_sequence_index) is not int
            or self.preflight_receipts_message_sequence_index < 2
            or type(self.channel_transcript_frame_count) is not int
            or type(self.channel_parent_send_sequence) is not int
            or type(self.channel_parent_receive_sequence) is not int
            or self.channel_parent_send_sequence != 0
            or self.channel_parent_receive_sequence
            != self.preflight_receipts_message_sequence_index + 1
            or self.channel_transcript_frame_count != self.channel_parent_receive_sequence
        ):
            raise ExperimentContractError("validated channel transcript prefix differs")
        _require_exact(
            _sha256(self.worker_bootstrap_canonical_bytes),
            self.worker_bootstrap_identity_receipt_sha256,
            field="worker bootstrap receipt binding",
        )
        _require_exact(
            _sha256(self.host_runtime_canonical_bytes),
            self.host_runtime_receipt_sha256,
            field="host runtime receipt binding",
        )
        _require_exact(
            _sha256(self.instrumentation_canonical_bytes),
            self.instrumentation_equivalence_receipt_sha256,
            field="instrumentation receipt binding",
        )
        _require_exact(
            _sha256(self.receipt_set_canonical_bytes),
            self.receipt_set_sha256,
            field="worker preflight receipt-set binding",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lineage": {
                "attempt_id": self.attempt_id,
                "attempt_nonce": self.attempt_nonce,
                "preflight_contract_sha256": self.preflight_contract_sha256,
                "claimed_work_directory_identity": self.claimed_work_directory_identity,
                "project_python_source_tree_sha256": (self.project_python_source_tree_sha256),
            },
            "supervised_delivery": {
                "worker_pid": self.worker_pid,
                "worker_pgid": self.worker_pgid,
                "worker_sid": self.worker_sid,
                "worker_process_start_monotonic_seconds": self.worker_process_start_monotonic_seconds,
                "preflight_receipts_message_sequence_index": (
                    self.preflight_receipts_message_sequence_index
                ),
                "channel_transcript_frame_count": self.channel_transcript_frame_count,
                "channel_transcript_sha256": self.channel_transcript_sha256,
                "channel_parent_send_sequence": self.channel_parent_send_sequence,
                "channel_parent_receive_sequence": self.channel_parent_receive_sequence,
                "authorizes_model_construction": False,
                "behavioral_evidence": False,
            },
            "receipt_set": json.loads(self.receipt_set_canonical_bytes),
            "worker_bootstrap_identity": json.loads(self.worker_bootstrap_canonical_bytes),
            "host_runtime": json.loads(self.host_runtime_canonical_bytes),
            "instrumentation_equivalence": json.loads(self.instrumentation_canonical_bytes),
        }


def worker_preflight_receipt_set_bytes_v2(
    preflight: ValidatedTQCPreflightContractV2,
    *,
    worker_bootstrap_receipt_bytes: bytes,
    host_runtime_receipt_bytes: bytes,
    instrumentation_receipt_bytes: bytes,
) -> bytes:
    """Bind this session-leader worker's three observations before delivery."""

    if type(preflight) is not ValidatedTQCPreflightContractV2:
        raise ExperimentContractError("worker receipt-set framing requires preflight authority")
    for field, encoded in (
        ("worker bootstrap", worker_bootstrap_receipt_bytes),
        ("host runtime", host_runtime_receipt_bytes),
        ("instrumentation", instrumentation_receipt_bytes),
    ):
        if type(encoded) is not bytes or not encoded or len(encoded) > MAX_WORKER_RECEIPT_BYTES:
            raise ExperimentContractError(f"{field} receipt bytes are absent or oversized")
    bootstrap = _decode_canonical_object(
        worker_bootstrap_receipt_bytes,
        maximum_bytes=MAX_WORKER_RECEIPT_BYTES,
        artifact="worker bootstrap identity receipt",
    )
    worker_pid, worker_pgid, worker_sid = _current_worker_process_identity()
    for field, expected in (
        ("attempt_id", preflight.attempt_id),
        ("attempt_nonce", preflight.attempt_nonce),
        ("preflight_contract_sha256", preflight.sha256),
        ("claimed_work_directory_identity", preflight.claimed_work_directory_identity),
        ("worker_pid", worker_pid),
        ("worker_pgid", worker_pgid),
        ("worker_sid", worker_sid),
        ("worker_pid_equals_pgid_equals_sid", True),
    ):
        _require_exact(bootstrap.get(field), expected, field=f"receipt-set bootstrap {field}")
    worker_start = bootstrap.get("worker_process_start_monotonic_seconds")
    if type(worker_start) is not float or not math.isfinite(worker_start) or worker_start < 0.0:
        raise ExperimentContractError("receipt-set worker start monotonic time is invalid")
    if worker_pid <= 1 or not worker_pid == worker_pgid == worker_sid:
        raise ExperimentContractError("receipt-set worker is not the fresh session leader")
    return canonical_json(
        {
            "schema_version": 1,
            "receipt_id": WORKER_PREFLIGHT_SET_ID,
            "attempt_id": preflight.attempt_id,
            "attempt_nonce": preflight.attempt_nonce,
            "preflight_contract_sha256": preflight.sha256,
            "claimed_work_directory_identity": preflight.claimed_work_directory_identity,
            "project_python_source_tree_sha256": (preflight.project_python_source_tree_sha256),
            "worker_pid": worker_pid,
            "worker_pgid": worker_pgid,
            "worker_sid": worker_sid,
            "worker_process_start_monotonic_seconds": worker_start,
            "worker_bootstrap_identity_receipt_sha256": _sha256(worker_bootstrap_receipt_bytes),
            "host_runtime_receipt_sha256": _sha256(host_runtime_receipt_bytes),
            "instrumentation_equivalence_receipt_sha256": _sha256(instrumentation_receipt_bytes),
        }
    )


def validate_tqc_worker_preflight_receipts_v2(
    design: LoadedTQCDevelopmentDesignV2,
    preflight: ValidatedTQCPreflightContractV2,
    *,
    supervised_delivery: object,
    worker_bootstrap_receipt_bytes: bytes,
    host_runtime_receipt_bytes: bytes,
    instrumentation_receipt_bytes: bytes,
    worker_receipt_set_bytes: bytes,
) -> ValidatedTQCWorkerPreflightReceiptsV2:
    """Admit exact receipts delivered by the one supervised worker channel."""

    if type(preflight) is not ValidatedTQCPreflightContractV2:
        raise ExperimentContractError(
            "worker receipt validation requires local preflight authority"
        )
    if type(supervised_delivery) is not _worker_preflight_delivery_type():
        raise ExperimentContractError(
            "worker receipt validation requires exact supervised delivery authority"
        )
    bootstrap = _decode_canonical_object(
        worker_bootstrap_receipt_bytes,
        maximum_bytes=MAX_WORKER_RECEIPT_BYTES,
        artifact="worker bootstrap identity receipt",
    )
    if type(design) is not LoadedTQCDevelopmentDesignV2:
        raise ExperimentContractError("worker receipt validation requires the strict-loaded design")
    _require_exact(design.file_sha256, preflight.design_file_sha256, field="worker design file")
    _require_exact(
        design.semantic_sha256,
        preflight.design_semantic_sha256,
        field="worker design semantics",
    )
    _require_exact(
        design.training_projection_sha256,
        preflight.training_projection_sha256,
        field="worker training projection",
    )
    receipt_binding = _worker_receipt_binding(
        preflight,
        worker_pid=getattr(supervised_delivery, "worker_pid", None),
        worker_process_start_monotonic_seconds=getattr(
            supervised_delivery,
            "worker_process_start_monotonic_seconds",
            None,
        ),
    )
    host_runtime_receipt = _runtime_receipt_validator()(
        host_runtime_receipt_bytes,
        design,
        expected_binding=receipt_binding,
    )
    if type(host_runtime_receipt) is not TQCHostRuntimeReceiptV2:
        raise ExperimentContractError("runtime validator did not issue the exact authority type")
    instrumentation_receipt = _instrumentation_receipt_validator()(
        instrumentation_receipt_bytes,
        design,
        expected_binding=_worker_receipt_binding(
            preflight,
            worker_pid=getattr(supervised_delivery, "worker_pid", None),
            worker_process_start_monotonic_seconds=getattr(
                supervised_delivery,
                "worker_process_start_monotonic_seconds",
                None,
            ),
        ),
    )
    if type(instrumentation_receipt) is not _instrumentation_receipt_type():
        raise ExperimentContractError(
            "instrumentation validator did not issue the exact authority type"
        )
    _require_exact(
        host_runtime_receipt.canonical_bytes,
        host_runtime_receipt_bytes,
        field="readmitted host runtime bytes",
    )
    runtime = _decode_canonical_object(
        host_runtime_receipt_bytes,
        maximum_bytes=MAX_WORKER_RECEIPT_BYTES,
        artifact="host runtime receipt",
    )
    _require_exact(
        getattr(instrumentation_receipt, "canonical_bytes", None),
        instrumentation_receipt_bytes,
        field="readmitted instrumentation bytes",
    )
    instrumentation = _decode_canonical_object(
        instrumentation_receipt_bytes,
        maximum_bytes=MAX_WORKER_RECEIPT_BYTES,
        artifact="instrumentation equivalence receipt",
    )
    receipt_set = _decode_canonical_object(
        worker_receipt_set_bytes,
        maximum_bytes=MAX_WORKER_RECEIPT_BYTES,
        artifact="worker preflight receipt set",
    )
    _require_keys(
        bootstrap,
        {
            "schema_version",
            "receipt_id",
            "attempt_id",
            "attempt_nonce",
            "claimed_work_directory_identity",
            "worker_pid",
            "worker_pgid",
            "worker_sid",
            "worker_pid_equals_pgid_equals_sid",
            "worker_process_start_monotonic_seconds",
            "worker_entrypoint_source_sha256",
            "preflight_contract_sha256",
        },
        field="worker bootstrap receipt",
    )
    _require_keys(
        runtime,
        {
            "schema_version",
            "runtime_receipt_id",
            "design_file_sha256",
            "design_semantic_sha256",
            "training_projection_sha256",
            "attempt_id",
            "attempt_nonce",
            "worker_pid",
            "worker_process_start_monotonic_seconds",
            "preflight_contract_sha256",
            "claimed_work_directory_identity",
            "observed_runtime_requirements",
            "observed_simulator",
            "observed_environment",
            "python_executable_realpath",
            "python_executable_file_sha256",
            "project_python_source_tree_sha256",
            "runtime_reinspection_source_sha256",
            "reward_or_info_fields_read",
            "behavioral_evidence",
        },
        field="host runtime receipt",
    )
    _require_keys(
        instrumentation,
        {
            "schema_version",
            "receipt_id",
            "design_file_sha256",
            "design_semantic_sha256",
            "training_projection_sha256",
            "attempt_id",
            "attempt_nonce",
            "worker_pid",
            "worker_process_start_monotonic_seconds",
            "preflight_contract_sha256",
            "claimed_work_directory_identity",
            "project_python_source_tree_sha256",
            "environment_instrumentation_equivalence_source_sha256",
            "seed",
            "horizon_steps",
            "action_shape",
            "action_sha256",
            "reset_comparison_count",
            "per_step_comparison_count",
            "comparison_fields",
            "structural_source_proof_sha256",
            "contact_trace_sha256",
            "contact_sample_count",
            "plain_shared_trace_sha256",
            "instrumented_shared_trace_sha256",
            "mismatch_count",
            "first_mismatch",
            "all_comparisons_passed",
            "scope",
            "claim_ceiling",
            "behavioral_evidence",
        },
        field="instrumentation equivalence receipt",
    )
    _require_keys(
        receipt_set,
        {
            "schema_version",
            "receipt_id",
            "attempt_id",
            "attempt_nonce",
            "preflight_contract_sha256",
            "claimed_work_directory_identity",
            "project_python_source_tree_sha256",
            "worker_pid",
            "worker_pgid",
            "worker_sid",
            "worker_process_start_monotonic_seconds",
            "worker_bootstrap_identity_receipt_sha256",
            "host_runtime_receipt_sha256",
            "instrumentation_equivalence_receipt_sha256",
        },
        field="worker preflight receipt set",
    )
    preflight_value = preflight.to_dict()
    bindings = preflight_value["bindings"]
    _require_exact(bootstrap["schema_version"], 1, field="worker bootstrap schema")
    _require_exact(bootstrap["attempt_id"], preflight.attempt_id, field="worker bootstrap attempt")
    _require_exact(
        bootstrap["attempt_nonce"], preflight.attempt_nonce, field="worker bootstrap nonce"
    )
    _require_exact(
        bootstrap["preflight_contract_sha256"],
        preflight.sha256,
        field="worker bootstrap preflight binding",
    )
    _require_exact(
        bootstrap["claimed_work_directory_identity"],
        preflight.claimed_work_directory_identity,
        field="worker bootstrap work-directory binding",
    )
    _require_exact(bootstrap["receipt_id"], WORKER_BOOTSTRAP_RECEIPT_ID, field="bootstrap id")
    for field in ("worker_pid", "worker_pgid", "worker_sid"):
        expected = getattr(supervised_delivery, field, None)
        if type(expected) is not int or expected <= 1:
            raise ExperimentContractError(f"supervised-delivery {field} is invalid")
        _require_exact(bootstrap[field], expected, field=f"bootstrap {field}")
    worker_pid = supervised_delivery.worker_pid
    worker_pgid = supervised_delivery.worker_pgid
    worker_sid = supervised_delivery.worker_sid
    _require_exact(
        worker_pid == worker_pgid == worker_sid,
        True,
        field="supervisor-observed worker process group",
    )
    _require_exact(
        bootstrap["worker_pid_equals_pgid_equals_sid"],
        True,
        field="worker process-group self-check",
    )
    expected_worker_start_monotonic_seconds = getattr(
        supervised_delivery,
        "worker_process_start_monotonic_seconds",
        None,
    )
    if (
        type(expected_worker_start_monotonic_seconds) is not float
        or not math.isfinite(expected_worker_start_monotonic_seconds)
        or expected_worker_start_monotonic_seconds < 0.0
    ):
        raise ExperimentContractError("supervised worker start monotonic time is invalid")
    _require_exact(
        bootstrap["worker_process_start_monotonic_seconds"],
        expected_worker_start_monotonic_seconds,
        field="worker start monotonic time",
    )
    _require_exact(
        bootstrap["worker_entrypoint_source_sha256"],
        bindings["worker_entrypoint_source_sha256"],
        field="worker entrypoint source",
    )
    _require_exact(runtime["schema_version"], 1, field="runtime receipt schema")
    _require_exact(
        runtime["runtime_receipt_id"], HOST_RUNTIME_RECEIPT_ID, field="runtime receipt id"
    )
    _require_exact(
        _sha256(host_runtime_receipt_bytes),
        host_runtime_receipt.sha256,
        field="host runtime authority binding",
    )
    from .tqc_instrumentation_equivalence_v2 import (
        EXPECTED_CONTACT_SAMPLE_COUNT,
        EXPECTED_CONTACT_TRACE_SHA256,
        EXPECTED_STRUCTURAL_SOURCE_PROOF_SHA256,
    )

    for field, expected in (
        ("design_file_sha256", preflight.design_file_sha256),
        ("design_semantic_sha256", preflight.design_semantic_sha256),
        ("training_projection_sha256", preflight.training_projection_sha256),
        (
            "project_python_source_tree_sha256",
            preflight.project_python_source_tree_sha256,
        ),
        ("python_executable_realpath", bindings["python_executable_realpath"]),
        ("python_executable_file_sha256", bindings["python_executable_file_sha256"]),
    ):
        _require_exact(runtime[field], expected, field=f"host runtime {field}")
    _require_exact(
        runtime["runtime_reinspection_source_sha256"],
        bindings["runtime_reinspection_source_sha256"],
        field="runtime reinspection source",
    )
    _require_exact(
        runtime["observed_runtime_requirements"],
        preflight_value["requested"]["runtime_requirements"],
        field="worker observed runtime",
    )
    _require_exact(
        runtime["observed_simulator"],
        design.to_dict()["training_projection"]["simulator"],
        field="worker observed simulator",
    )
    observed_environment = runtime["observed_environment"]
    if type(observed_environment) is not dict:
        raise ExperimentContractError("worker observed environment must be an object")
    _require_exact(
        observed_environment,
        _expected_runtime_environment(
            design,
            reset_observation_sha256=observed_environment.get("reset_observation_sha256"),
        ),
        field="worker observed environment",
    )
    _require_exact(runtime["reward_or_info_fields_read"], False, field="runtime reward access")
    _require_exact(runtime["behavioral_evidence"], False, field="runtime behavior evidence")
    _require_exact(
        getattr(instrumentation_receipt, "sha256", None),
        _sha256(instrumentation_receipt_bytes),
        field="instrumentation authority binding",
    )
    _require_exact(getattr(instrumentation_receipt, "seed", None), 98_001, field="probe seed")
    _require_exact(
        getattr(instrumentation_receipt, "horizon_steps", None),
        1_000,
        field="probe horizon",
    )
    _require_exact(
        getattr(instrumentation_receipt, "action_sha256", None),
        "8cbe0bc01c56134fdab90ba6506ebd74622f4929eb3216d332c05e459fa63af4",
        field="probe actions",
    )
    _require_exact(
        getattr(instrumentation_receipt, "all_comparisons_passed", None),
        True,
        field="instrumentation comparisons",
    )
    _require_exact(
        instrumentation["environment_instrumentation_equivalence_source_sha256"],
        bindings["environment_instrumentation_equivalence_source_sha256"],
        field="instrumentation source",
    )
    for field, expected in (
        ("schema_version", 1),
        ("receipt_id", INSTRUMENTATION_RECEIPT_ID),
        ("design_file_sha256", preflight.design_file_sha256),
        ("design_semantic_sha256", preflight.design_semantic_sha256),
        ("training_projection_sha256", preflight.training_projection_sha256),
        ("seed", 98_001),
        ("horizon_steps", 1_000),
        ("action_shape", [1_000, 17]),
        (
            "action_sha256",
            "8cbe0bc01c56134fdab90ba6506ebd74622f4929eb3216d332c05e459fa63af4",
        ),
        ("reset_comparison_count", 1),
        ("per_step_comparison_count", 1_000),
        (
            "comparison_fields",
            [
                "returned_observation",
                "reward",
                "terminated",
                "truncated",
                "mjSTATE_INTEGRATION",
                "qacc",
                "qfrc_actuator",
                "simulation_time",
                "TimeLimit_elapsed_steps",
            ],
        ),
        ("mismatch_count", 0),
        ("structural_source_proof_sha256", EXPECTED_STRUCTURAL_SOURCE_PROOF_SHA256),
        ("contact_trace_sha256", EXPECTED_CONTACT_TRACE_SHA256),
        ("contact_sample_count", EXPECTED_CONTACT_SAMPLE_COUNT),
        ("all_comparisons_passed", True),
        ("scope", "preflight_canary_only_not_complete_dynamics_equivalence_evidence"),
        ("claim_ceiling", "instrumentation_equivalence_only_no_controller_or_behavior_claim/v1"),
        ("behavioral_evidence", False),
    ):
        _require_exact(instrumentation.get(field), expected, field=f"instrumentation {field}")
    _require_exact(instrumentation.get("first_mismatch"), None, field="probe first mismatch")
    _require_sha256(
        instrumentation["structural_source_proof_sha256"], field="structural source proof"
    )
    for field in ("plain_shared_trace_sha256", "instrumented_shared_trace_sha256"):
        _require_sha256(instrumentation[field], field=field)
    _require_exact(
        instrumentation["plain_shared_trace_sha256"],
        instrumentation["instrumented_shared_trace_sha256"],
        field="instrumentation shared trace",
    )
    expected_receipt_set = {
        "schema_version": 1,
        "receipt_id": WORKER_PREFLIGHT_SET_ID,
        "attempt_id": preflight.attempt_id,
        "attempt_nonce": preflight.attempt_nonce,
        "preflight_contract_sha256": preflight.sha256,
        "claimed_work_directory_identity": preflight.claimed_work_directory_identity,
        "project_python_source_tree_sha256": (preflight.project_python_source_tree_sha256),
        "worker_pid": worker_pid,
        "worker_pgid": worker_pgid,
        "worker_sid": worker_sid,
        "worker_process_start_monotonic_seconds": expected_worker_start_monotonic_seconds,
        "worker_bootstrap_identity_receipt_sha256": _sha256(worker_bootstrap_receipt_bytes),
        "host_runtime_receipt_sha256": _sha256(host_runtime_receipt_bytes),
        "instrumentation_equivalence_receipt_sha256": _sha256(instrumentation_receipt_bytes),
    }
    _require_exact(receipt_set, expected_receipt_set, field="worker preflight receipt set")
    ordered_receipt_sha256 = (
        _sha256(worker_bootstrap_receipt_bytes),
        _sha256(host_runtime_receipt_bytes),
        _sha256(instrumentation_receipt_bytes),
        _sha256(worker_receipt_set_bytes),
    )
    for field, expected in (
        ("attempt_id", preflight.attempt_id),
        ("attempt_nonce", preflight.attempt_nonce),
        ("preflight_contract_sha256", preflight.sha256),
        ("claimed_work_directory_identity", preflight.claimed_work_directory_identity),
        ("ordered_receipt_sha256", ordered_receipt_sha256),
        ("authorizes_model_construction", False),
        ("behavioral_evidence", False),
    ):
        _require_exact(
            getattr(supervised_delivery, field, None),
            expected,
            field=f"supervised delivery {field}",
        )
    channel_transcript_sha256 = _require_sha256(
        getattr(supervised_delivery, "channel_transcript_sha256", None),
        field="supervised delivery channel transcript",
    )
    preflight_message_index = getattr(
        supervised_delivery,
        "preflight_receipts_message_sequence_index",
        None,
    )
    transcript_frame_count = getattr(
        supervised_delivery,
        "channel_transcript_frame_count",
        None,
    )
    parent_send_sequence = getattr(
        supervised_delivery,
        "channel_parent_send_sequence",
        None,
    )
    parent_receive_sequence = getattr(
        supervised_delivery,
        "channel_parent_receive_sequence",
        None,
    )
    if (
        type(preflight_message_index) is not int
        or preflight_message_index < 2
        or type(transcript_frame_count) is not int
        or type(parent_send_sequence) is not int
        or type(parent_receive_sequence) is not int
        or parent_send_sequence != 0
        or parent_receive_sequence != preflight_message_index + 1
        or transcript_frame_count != parent_receive_sequence
    ):
        raise ExperimentContractError("supervised delivery channel transcript boundary differs")
    return ValidatedTQCWorkerPreflightReceiptsV2(
        attempt_id=preflight.attempt_id,
        attempt_nonce=preflight.attempt_nonce,
        preflight_contract_sha256=preflight.sha256,
        claimed_work_directory_identity=preflight.claimed_work_directory_identity,
        project_python_source_tree_sha256=(preflight.project_python_source_tree_sha256),
        worker_bootstrap_identity_receipt_sha256=_sha256(worker_bootstrap_receipt_bytes),
        host_runtime_receipt_sha256=_sha256(host_runtime_receipt_bytes),
        instrumentation_equivalence_receipt_sha256=_sha256(instrumentation_receipt_bytes),
        receipt_set_sha256=_sha256(worker_receipt_set_bytes),
        worker_pid=worker_pid,
        worker_pgid=worker_pgid,
        worker_sid=worker_sid,
        worker_process_start_monotonic_seconds=expected_worker_start_monotonic_seconds,
        preflight_receipts_message_sequence_index=preflight_message_index,
        channel_transcript_frame_count=transcript_frame_count,
        channel_transcript_sha256=channel_transcript_sha256,
        channel_parent_send_sequence=parent_send_sequence,
        channel_parent_receive_sequence=parent_receive_sequence,
        worker_bootstrap_canonical_bytes=worker_bootstrap_receipt_bytes,
        host_runtime_canonical_bytes=host_runtime_receipt_bytes,
        instrumentation_canonical_bytes=instrumentation_receipt_bytes,
        receipt_set_canonical_bytes=worker_receipt_set_bytes,
        _issuer=_WORKER_RECEIPTS_ISSUER,
    )


def _revalidate_preflight_environment(preflight: ValidatedTQCPreflightContractV2) -> None:
    value = preflight.to_dict()
    observed = value["observed"]
    recorded_work = observed["claimed_work_directory"]
    work = _capture_work_directory(
        Path(recorded_work["absolute_path"]),
        reservation_receipt_path=Path(recorded_work["reservation_receipt"]["absolute_path"]),
        reservation_nonce=recorded_work["reservation_nonce"],
        preflight_contract_path=Path(recorded_work["preflight_contract_absolute_path"]),
        execution_manifest_path=Path(recorded_work["execution_manifest_absolute_path"]),
    )
    _require_exact(work, recorded_work, field="current work-directory identity")
    source_root = Path(observed["project_python_source_tree"]["absolute_path"])
    _require_exact(
        _absolute_real_path(source_root, artifact="current project Python source tree"),
        _expected_project_source_root(),
        field="current imported project source root",
    )
    tree = _capture_source_tree(source_root)
    _require_exact(
        tree, observed["project_python_source_tree"], field="current project source tree"
    )
    for role, record in observed["source_files"].items():
        current = _capture_regular_file(Path(record["absolute_path"]), artifact=role)
        _require_exact(current, record, field=f"current {role}")
    dependency = _capture_regular_file(
        Path(observed["dependency_lock"]["absolute_path"]),
        artifact="dependency lock",
    )
    _require_exact(dependency, observed["dependency_lock"], field="current dependency lock")
    executable = _capture_regular_file(
        Path(observed["python_executable"]["absolute_path"]),
        artifact="Python executable",
        maximum_bytes=MAX_EXECUTABLE_BYTES,
    )
    _require_exact(executable, observed["python_executable"], field="current Python executable")


def _execution_payload(
    design: LoadedTQCDevelopmentDesignV2,
    e0: ValidatedV2E0ReuseEvidence,
    review: ValidatedIndependentReviewReceiptV2,
    preflight: ValidatedTQCPreflightContractV2,
    worker_receipts: ValidatedTQCWorkerPreflightReceiptsV2,
    *,
    output_path: Path,
    execution_source_paths: Mapping[str, Path],
) -> dict[str, Any]:
    _validate_primary_authorities(design, e0, review)
    if type(preflight) is not ValidatedTQCPreflightContractV2:
        raise ExperimentContractError("execution manifest requires local preflight authority")
    if type(worker_receipts) is not ValidatedTQCWorkerPreflightReceiptsV2:
        raise ExperimentContractError("execution manifest requires validated worker receipts")
    for field in (
        "attempt_id",
        "attempt_nonce",
        "claimed_work_directory_identity",
        "project_python_source_tree_sha256",
    ):
        _require_exact(
            getattr(worker_receipts, field),
            getattr(preflight, field),
            field=f"worker/preflight {field}",
        )
    _require_exact(
        worker_receipts.preflight_contract_sha256,
        preflight.sha256,
        field="worker/preflight contract binding",
    )
    for observed, expected, field in (
        (preflight.design_file_sha256, design.file_sha256, "preflight/design file"),
        (
            preflight.design_semantic_sha256,
            design.semantic_sha256,
            "preflight/design semantics",
        ),
        (
            preflight.training_projection_sha256,
            e0.training_projection_sha256,
            "preflight/E0 training projection",
        ),
        (
            preflight.independent_review_receipt_content_sha256,
            review.content_sha256,
            "preflight/review receipt content",
        ),
    ):
        _require_exact(observed, expected, field=field)
    _revalidate_preflight_environment(preflight)
    preflight_value = preflight.to_dict()
    work = preflight_value["observed"]["claimed_work_directory"]
    absolute_output = Path(os.path.abspath(output_path))
    if absolute_output != Path(work["execution_manifest_absolute_path"]):
        raise ExperimentContractError("execution manifest path differs from its reservation")
    execution_paths = _exact_path_mapping(
        execution_source_paths,
        EXECUTION_SOURCE_ROLES,
        field="execution source paths",
    )
    execution_records = {
        role: _capture_regular_file(path, artifact=role) for role, path in execution_paths.items()
    }
    tree = preflight_value["observed"]["project_python_source_tree"]
    _require_paths_inside_tree(execution_records, tree=tree)
    preflight_sources = preflight_value["observed"]["source_files"]
    bindings = {
        "design_file_sha256": DESIGN_FILE_SHA256,
        "design_semantic_sha256": DESIGN_SEMANTIC_SHA256,
        "v2_contract_source_sha256": preflight_sources["v2_contract_source_sha256"]["sha256"],
        "e0_reuse_audit_file_sha256": AUDIT_FILE_SHA256,
        "e0_reuse_audit_semantic_sha256": AUDIT_SEMANTIC_SHA256,
        "e0_v2_training_projection_sha256": TRAINING_PROJECTION_SHA256,
        "e0_design_artifact_sha256": e0.e0_design_file_sha256,
        "e0_receipt_content_sha256": e0.e0_receipt_file_sha256,
        "e0_runtime_sha256": e0.e0_runtime_sha256,
        "project_python_source_tree_sha256": preflight.project_python_source_tree_sha256,
        "dependency_lock_sha256": preflight.dependency_lock_sha256,
        **{role: record["sha256"] for role, record in execution_records.items()},
        "environment_instrumentation_equivalence_source_sha256": preflight_sources[
            "environment_instrumentation_equivalence_source_sha256"
        ]["sha256"],
        "instrumentation_equivalence_receipt_sha256": worker_receipts.instrumentation_equivalence_receipt_sha256,
        "independent_review_receipt_content_sha256": review.content_sha256,
        "preflight_contract_sha256": preflight.sha256,
        "worker_bootstrap_identity_receipt_sha256": worker_receipts.worker_bootstrap_identity_receipt_sha256,
        "host_runtime_receipt_sha256": worker_receipts.host_runtime_receipt_sha256,
        "claimed_work_directory_identity": preflight.claimed_work_directory_identity,
    }
    expected_roles = set(design.to_dict()["execution_manifest"]["must_bind"])
    _require_keys(bindings, expected_roles, field="execution manifest bindings")
    return {
        "schema_version": 1,
        "manifest_id": EXECUTION_MANIFEST_ID,
        "attempt_id": preflight.attempt_id,
        "attempt_nonce": preflight.attempt_nonce,
        "manifest_absolute_path": str(absolute_output),
        "requested": {
            "design_id": DESIGN_ID,
            "attempt_id": ATTEMPT_ID,
            "design_file_sha256": DESIGN_FILE_SHA256,
            "design_semantic_sha256": DESIGN_SEMANTIC_SHA256,
            "training_projection_sha256": TRAINING_PROJECTION_SHA256,
            "runtime_requirements": preflight_value["requested"]["runtime_requirements"],
            "claimed_work_directory_absolute_path": work["absolute_path"],
        },
        "observed": {
            "runtime_requirements": worker_receipts.to_dict()["host_runtime"][
                "observed_runtime_requirements"
            ],
            "claimed_work_directory": work,
            "project_python_source_tree": tree,
            "preflight_source_files": preflight_sources,
            "execution_source_files": execution_records,
            "worker_receipts": worker_receipts.to_dict(),
        },
        "bindings": bindings,
        "admission": {
            "independent_review_go": True,
            "e0_resource_prior_only": True,
            "worker_runtime_reinspection_passed": True,
            "instrumentation_equivalence_canary_passed": True,
            "model_construction_allowed": True,
            "behavioral_evidence": False,
        },
    }


def _validate_execution_payload(
    value: dict[str, Any], *, design: LoadedTQCDevelopmentDesignV2
) -> None:
    _require_keys(
        value,
        {
            "schema_version",
            "manifest_id",
            "attempt_id",
            "attempt_nonce",
            "manifest_absolute_path",
            "requested",
            "observed",
            "bindings",
            "admission",
        },
        field="v2 execution manifest",
    )
    _require_exact(value["schema_version"], 1, field="execution manifest schema")
    _require_exact(value["manifest_id"], EXECUTION_MANIFEST_ID, field="execution manifest id")
    _require_exact(value["attempt_id"], ATTEMPT_ID, field="execution manifest attempt")
    _require_sha256(value["attempt_nonce"], field="execution manifest nonce")
    requested = _require_keys(
        value["requested"],
        {
            "design_id",
            "attempt_id",
            "design_file_sha256",
            "design_semantic_sha256",
            "training_projection_sha256",
            "runtime_requirements",
            "claimed_work_directory_absolute_path",
        },
        field="execution manifest requested values",
    )
    _require_keys(
        value["observed"],
        {
            "runtime_requirements",
            "claimed_work_directory",
            "project_python_source_tree",
            "preflight_source_files",
            "execution_source_files",
            "worker_receipts",
        },
        field="execution manifest observed values",
    )
    for field, expected in (
        ("design_id", DESIGN_ID),
        ("attempt_id", ATTEMPT_ID),
        ("design_file_sha256", DESIGN_FILE_SHA256),
        ("design_semantic_sha256", DESIGN_SEMANTIC_SHA256),
        ("training_projection_sha256", TRAINING_PROJECTION_SHA256),
        ("runtime_requirements", _runtime_requirements(design)),
    ):
        _require_exact(requested[field], expected, field=f"execution requested {field}")
    _require_keys(
        value["bindings"],
        set(design.to_dict()["execution_manifest"]["must_bind"]),
        field="execution manifest bindings",
    )
    for role, digest in value["bindings"].items():
        _require_sha256(digest, field=f"execution manifest binding {role}")
    for role, expected in (
        ("design_file_sha256", DESIGN_FILE_SHA256),
        ("design_semantic_sha256", DESIGN_SEMANTIC_SHA256),
        ("e0_reuse_audit_file_sha256", AUDIT_FILE_SHA256),
        ("e0_reuse_audit_semantic_sha256", AUDIT_SEMANTIC_SHA256),
        ("e0_v2_training_projection_sha256", TRAINING_PROJECTION_SHA256),
        ("e0_design_artifact_sha256", E0_DESIGN_FILE_SHA256),
        ("e0_receipt_content_sha256", E0_RECEIPT_FILE_SHA256),
        ("e0_runtime_sha256", E0_RUNTIME_SHA256),
    ):
        _require_exact(value["bindings"][role], expected, field=f"execution binding {role}")
    work = value["observed"]["claimed_work_directory"]
    if type(work) is not dict:
        raise ExperimentContractError("execution observed work directory must be an object")
    _require_exact(
        requested["claimed_work_directory_absolute_path"],
        work.get("absolute_path"),
        field="execution requested/observed work directory",
    )
    _require_exact(
        value["manifest_absolute_path"],
        work.get("execution_manifest_absolute_path"),
        field="execution manifest reserved path",
    )
    _require_exact(
        value["admission"],
        {
            "independent_review_go": True,
            "e0_resource_prior_only": True,
            "worker_runtime_reinspection_passed": True,
            "instrumentation_equivalence_canary_passed": True,
            "model_construction_allowed": True,
            "behavioral_evidence": False,
        },
        field="execution manifest admission",
    )


@dataclass(frozen=True, slots=True)
class ValidatedTQCExecutionManifestV2:
    """Immutable same-attempt authority required before TQC construction."""

    canonical_bytes: bytes
    sha256: str
    byte_count: int
    attempt_id: str
    design_file_sha256: str
    design_semantic_sha256: str
    training_projection_sha256: str
    preflight_contract_sha256: str
    claimed_work_directory_identity: str
    model_construction_allowed: bool
    authorizes_training: bool
    behavioral_evidence: bool
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _MANIFEST_ISSUER:
            raise ExperimentContractError("v2 execution manifests may only be issued by validation")
        value = _decode_canonical_object(
            self.canonical_bytes,
            maximum_bytes=MAX_EXECUTION_MANIFEST_BYTES,
            artifact="validated v2 execution manifest",
        )
        _require_exact(_sha256(self.canonical_bytes), self.sha256, field="manifest SHA-256")
        _require_exact(len(self.canonical_bytes), self.byte_count, field="manifest byte count")
        bindings = value["bindings"]
        for field, expected in (
            ("attempt_id", value["attempt_id"]),
            ("design_file_sha256", bindings["design_file_sha256"]),
            ("design_semantic_sha256", bindings["design_semantic_sha256"]),
            ("training_projection_sha256", bindings["e0_v2_training_projection_sha256"]),
            ("preflight_contract_sha256", bindings["preflight_contract_sha256"]),
            ("claimed_work_directory_identity", bindings["claimed_work_directory_identity"]),
            ("model_construction_allowed", value["admission"]["model_construction_allowed"]),
        ):
            _require_exact(getattr(self, field), expected, field=f"manifest authority {field}")
        _require_exact(self.authorizes_training, True, field="manifest training authority")
        _require_exact(self.behavioral_evidence, False, field="manifest behavior evidence")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)


def _issue_execution_manifest(
    encoded: bytes,
    *,
    design: LoadedTQCDevelopmentDesignV2,
    e0: ValidatedV2E0ReuseEvidence,
    review: ValidatedIndependentReviewReceiptV2,
    preflight: ValidatedTQCPreflightContractV2,
    worker_receipts: ValidatedTQCWorkerPreflightReceiptsV2,
    manifest_path: Path,
    execution_source_paths: Mapping[str, Path],
) -> ValidatedTQCExecutionManifestV2:
    expected = canonical_json(
        _execution_payload(
            design,
            e0,
            review,
            preflight,
            worker_receipts,
            output_path=manifest_path,
            execution_source_paths=execution_source_paths,
        )
    )
    _require_exact(encoded, expected, field="execution manifest issuance evidence")
    value = _decode_canonical_object(
        encoded,
        maximum_bytes=MAX_EXECUTION_MANIFEST_BYTES,
        artifact="v2 execution manifest",
    )
    _validate_execution_payload(value, design=design)
    bindings = value["bindings"]
    return ValidatedTQCExecutionManifestV2(
        canonical_bytes=encoded,
        sha256=_sha256(encoded),
        byte_count=len(encoded),
        attempt_id=value["attempt_id"],
        design_file_sha256=bindings["design_file_sha256"],
        design_semantic_sha256=bindings["design_semantic_sha256"],
        training_projection_sha256=bindings["e0_v2_training_projection_sha256"],
        preflight_contract_sha256=bindings["preflight_contract_sha256"],
        claimed_work_directory_identity=bindings["claimed_work_directory_identity"],
        model_construction_allowed=True,
        authorizes_training=True,
        behavioral_evidence=False,
        _issuer=_MANIFEST_ISSUER,
    )


def publish_tqc_execution_manifest_v2(
    output_path: Path,
    design: LoadedTQCDevelopmentDesignV2,
    e0: ValidatedV2E0ReuseEvidence,
    review: ValidatedIndependentReviewReceiptV2,
    preflight: ValidatedTQCPreflightContractV2,
    worker_receipts: ValidatedTQCWorkerPreflightReceiptsV2,
    *,
    execution_source_paths: Mapping[str, Path],
) -> ValidatedTQCExecutionManifestV2:
    """Publish and reverify one immutable final manifest before model construction."""

    payload = _execution_payload(
        design,
        e0,
        review,
        preflight,
        worker_receipts,
        output_path=output_path,
        execution_source_paths=execution_source_paths,
    )
    encoded = canonical_json(payload)
    if len(encoded) > MAX_EXECUTION_MANIFEST_BYTES:
        raise ExperimentContractError("v2 execution manifest exceeds its byte bound")
    published = publish_bytes_without_overwrite(output_path, encoded)
    persisted = _read_bound_canonical_file(
        published.path,
        maximum_bytes=MAX_EXECUTION_MANIFEST_BYTES,
        artifact="persisted v2 execution manifest",
    )
    if published.sha256 != _sha256(persisted) or published.byte_count != len(persisted):
        raise ExperimentContractError("persisted v2 execution manifest differs after publication")
    expected = canonical_json(
        _execution_payload(
            design,
            e0,
            review,
            preflight,
            worker_receipts,
            output_path=output_path,
            execution_source_paths=execution_source_paths,
        )
    )
    _require_exact(persisted, expected, field="revalidated execution manifest bytes")
    return _issue_execution_manifest(
        persisted,
        design=design,
        e0=e0,
        review=review,
        preflight=preflight,
        worker_receipts=worker_receipts,
        manifest_path=output_path,
        execution_source_paths=execution_source_paths,
    )


def revalidate_tqc_execution_manifest_v2(
    manifest_path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
    design: LoadedTQCDevelopmentDesignV2,
    e0: ValidatedV2E0ReuseEvidence,
    review: ValidatedIndependentReviewReceiptV2,
    preflight: ValidatedTQCPreflightContractV2,
    worker_receipts: ValidatedTQCWorkerPreflightReceiptsV2,
    execution_source_paths: Mapping[str, Path],
) -> ValidatedTQCExecutionManifestV2:
    """Parent-side exact-byte reinspection against retained authorities."""

    _require_sha256(expected_sha256, field="expected execution manifest SHA-256")
    if type(expected_byte_count) is not int or expected_byte_count < 1:
        raise ExperimentContractError("expected execution manifest byte count is invalid")
    persisted = _read_bound_canonical_file(
        manifest_path,
        maximum_bytes=MAX_EXECUTION_MANIFEST_BYTES,
        artifact="persisted v2 execution manifest",
    )
    _require_exact(
        _sha256(persisted), expected_sha256, field="delivered execution manifest SHA-256"
    )
    _require_exact(len(persisted), expected_byte_count, field="delivered execution manifest bytes")
    expected = canonical_json(
        _execution_payload(
            design,
            e0,
            review,
            preflight,
            worker_receipts,
            output_path=manifest_path,
            execution_source_paths=execution_source_paths,
        )
    )
    _require_exact(persisted, expected, field="worker-reverified execution manifest")
    return _issue_execution_manifest(
        persisted,
        design=design,
        e0=e0,
        review=review,
        preflight=preflight,
        worker_receipts=worker_receipts,
        manifest_path=manifest_path,
        execution_source_paths=execution_source_paths,
    )


def revalidate_tqc_execution_manifest_for_worker_v2(
    manifest_path: Path,
    *,
    expected_sha256: str,
    expected_byte_count: int,
    design: LoadedTQCDevelopmentDesignV2,
    e0: ValidatedV2E0ReuseEvidence,
    review: ValidatedIndependentReviewReceiptV2,
    preflight: ValidatedTQCPreflightContractV2,
    worker_bootstrap_receipt_bytes: bytes,
    host_runtime_receipt_bytes: bytes,
    instrumentation_receipt_bytes: bytes,
    worker_receipt_set_bytes: bytes,
    execution_source_paths: Mapping[str, Path],
    preflight_receipts_message_sequence_index: int,
    channel_transcript_frame_count: int,
    channel_transcript_sha256: str,
) -> ValidatedTQCExecutionManifestV2:
    """Re-admit the parent's final manifest using worker-local receipt bytes."""

    _require_sha256(expected_sha256, field="expected execution manifest SHA-256")
    _require_sha256(channel_transcript_sha256, field="worker channel transcript")
    if type(expected_byte_count) is not int or expected_byte_count < 1:
        raise ExperimentContractError("expected execution manifest byte count is invalid")
    if (
        type(preflight_receipts_message_sequence_index) is not int
        or preflight_receipts_message_sequence_index < 2
        or type(channel_transcript_frame_count) is not int
        or channel_transcript_frame_count != preflight_receipts_message_sequence_index + 1
    ):
        raise ExperimentContractError("worker channel transcript boundary differs")
    if type(preflight) is not ValidatedTQCPreflightContractV2:
        raise ExperimentContractError("worker manifest revalidation requires preflight authority")
    if type(design) is not LoadedTQCDevelopmentDesignV2:
        raise ExperimentContractError("worker manifest revalidation requires strict-loaded design")
    _require_exact(design.file_sha256, preflight.design_file_sha256, field="worker design file")
    _require_exact(
        design.semantic_sha256,
        preflight.design_semantic_sha256,
        field="worker design semantics",
    )
    bootstrap = _decode_canonical_object(
        worker_bootstrap_receipt_bytes,
        maximum_bytes=MAX_WORKER_RECEIPT_BYTES,
        artifact="worker-local bootstrap receipt",
    )
    _require_keys(
        bootstrap,
        {
            "schema_version",
            "receipt_id",
            "attempt_id",
            "attempt_nonce",
            "claimed_work_directory_identity",
            "worker_pid",
            "worker_pgid",
            "worker_sid",
            "worker_pid_equals_pgid_equals_sid",
            "worker_process_start_monotonic_seconds",
            "worker_entrypoint_source_sha256",
            "preflight_contract_sha256",
        },
        field="worker-local bootstrap receipt",
    )
    current_pid, current_pgid, current_sid = _current_worker_process_identity()
    preflight_value = preflight.to_dict()
    for field, expected in (
        ("schema_version", 1),
        ("receipt_id", WORKER_BOOTSTRAP_RECEIPT_ID),
        ("attempt_id", preflight.attempt_id),
        ("attempt_nonce", preflight.attempt_nonce),
        ("claimed_work_directory_identity", preflight.claimed_work_directory_identity),
        ("worker_pid", current_pid),
        ("worker_pgid", current_pgid),
        ("worker_sid", current_sid),
        ("worker_pid_equals_pgid_equals_sid", True),
        (
            "worker_entrypoint_source_sha256",
            preflight_value["bindings"]["worker_entrypoint_source_sha256"],
        ),
        ("preflight_contract_sha256", preflight.sha256),
    ):
        _require_exact(bootstrap[field], expected, field=f"worker-local bootstrap {field}")
    if current_pid <= 1 or not current_pid == current_pgid == current_sid:
        raise ExperimentContractError("worker-local process is not the fresh session leader")
    worker_start = bootstrap["worker_process_start_monotonic_seconds"]
    if type(worker_start) is not float or not math.isfinite(worker_start) or worker_start < 0.0:
        raise ExperimentContractError("worker-local start monotonic time is invalid")
    receipt_binding = _worker_receipt_binding(
        preflight,
        worker_pid=current_pid,
        worker_process_start_monotonic_seconds=worker_start,
    )
    runtime_authority = _runtime_receipt_validator()(
        host_runtime_receipt_bytes,
        design,
        expected_binding=receipt_binding,
    )
    if type(runtime_authority) is not TQCHostRuntimeReceiptV2:
        raise ExperimentContractError("runtime validator did not issue the exact authority type")
    instrumentation_authority = _instrumentation_receipt_validator()(
        instrumentation_receipt_bytes,
        design,
        expected_binding=_worker_receipt_binding(
            preflight,
            worker_pid=current_pid,
            worker_process_start_monotonic_seconds=worker_start,
        ),
    )
    if type(instrumentation_authority) is not _instrumentation_receipt_type():
        raise ExperimentContractError(
            "instrumentation validator did not issue the exact authority type"
        )
    runtime = runtime_authority.to_dict()
    instrumentation = instrumentation_authority.to_dict()
    for field, expected in (
        (
            "project_python_source_tree_sha256",
            preflight.project_python_source_tree_sha256,
        ),
        ("python_executable_realpath", preflight_value["bindings"]["python_executable_realpath"]),
        (
            "python_executable_file_sha256",
            preflight_value["bindings"]["python_executable_file_sha256"],
        ),
        (
            "runtime_reinspection_source_sha256",
            preflight_value["bindings"]["runtime_reinspection_source_sha256"],
        ),
    ):
        _require_exact(runtime[field], expected, field=f"worker-local runtime {field}")
    _require_exact(
        runtime["observed_runtime_requirements"],
        preflight_value["requested"]["runtime_requirements"],
        field="worker-local runtime requirements",
    )
    _require_exact(
        runtime["observed_simulator"],
        design.to_dict()["training_projection"]["simulator"],
        field="worker-local simulator",
    )
    worker_environment = runtime["observed_environment"]
    if type(worker_environment) is not dict:
        raise ExperimentContractError("worker-local environment must be an object")
    _require_exact(
        worker_environment,
        _expected_runtime_environment(
            design,
            reset_observation_sha256=worker_environment.get("reset_observation_sha256"),
        ),
        field="worker-local environment",
    )
    _require_exact(
        instrumentation["environment_instrumentation_equivalence_source_sha256"],
        preflight_value["bindings"]["environment_instrumentation_equivalence_source_sha256"],
        field="worker-local instrumentation source",
    )
    _require_exact(
        instrumentation_authority.all_comparisons_passed,
        True,
        field="worker-local instrumentation comparisons",
    )
    receipt_set = _decode_canonical_object(
        worker_receipt_set_bytes,
        maximum_bytes=MAX_WORKER_RECEIPT_BYTES,
        artifact="worker-local preflight receipt set",
    )
    expected_receipt_set = {
        "schema_version": 1,
        "receipt_id": WORKER_PREFLIGHT_SET_ID,
        "attempt_id": preflight.attempt_id,
        "attempt_nonce": preflight.attempt_nonce,
        "preflight_contract_sha256": preflight.sha256,
        "claimed_work_directory_identity": preflight.claimed_work_directory_identity,
        "project_python_source_tree_sha256": (preflight.project_python_source_tree_sha256),
        "worker_pid": current_pid,
        "worker_pgid": current_pgid,
        "worker_sid": current_sid,
        "worker_process_start_monotonic_seconds": worker_start,
        "worker_bootstrap_identity_receipt_sha256": _sha256(worker_bootstrap_receipt_bytes),
        "host_runtime_receipt_sha256": _sha256(host_runtime_receipt_bytes),
        "instrumentation_equivalence_receipt_sha256": _sha256(instrumentation_receipt_bytes),
    }
    _require_exact(receipt_set, expected_receipt_set, field="worker-local receipt set")
    worker_receipts = ValidatedTQCWorkerPreflightReceiptsV2(
        attempt_id=preflight.attempt_id,
        attempt_nonce=preflight.attempt_nonce,
        preflight_contract_sha256=preflight.sha256,
        claimed_work_directory_identity=preflight.claimed_work_directory_identity,
        project_python_source_tree_sha256=(preflight.project_python_source_tree_sha256),
        worker_bootstrap_identity_receipt_sha256=_sha256(worker_bootstrap_receipt_bytes),
        host_runtime_receipt_sha256=_sha256(host_runtime_receipt_bytes),
        instrumentation_equivalence_receipt_sha256=_sha256(instrumentation_receipt_bytes),
        receipt_set_sha256=_sha256(worker_receipt_set_bytes),
        worker_pid=current_pid,
        worker_pgid=current_pgid,
        worker_sid=current_sid,
        worker_process_start_monotonic_seconds=worker_start,
        preflight_receipts_message_sequence_index=preflight_receipts_message_sequence_index,
        channel_transcript_frame_count=channel_transcript_frame_count,
        channel_transcript_sha256=channel_transcript_sha256,
        channel_parent_send_sequence=0,
        channel_parent_receive_sequence=channel_transcript_frame_count,
        worker_bootstrap_canonical_bytes=worker_bootstrap_receipt_bytes,
        host_runtime_canonical_bytes=host_runtime_receipt_bytes,
        instrumentation_canonical_bytes=instrumentation_receipt_bytes,
        receipt_set_canonical_bytes=worker_receipt_set_bytes,
        _issuer=_WORKER_RECEIPTS_ISSUER,
    )
    persisted = _read_bound_canonical_file(
        manifest_path,
        maximum_bytes=MAX_EXECUTION_MANIFEST_BYTES,
        artifact="worker-read v2 execution manifest",
    )
    _require_exact(
        _sha256(persisted),
        expected_sha256,
        field="worker-read execution manifest SHA-256",
    )
    _require_exact(
        len(persisted),
        expected_byte_count,
        field="worker-read execution manifest byte count",
    )
    expected = canonical_json(
        _execution_payload(
            design,
            e0,
            review,
            preflight,
            worker_receipts,
            output_path=manifest_path,
            execution_source_paths=execution_source_paths,
        )
    )
    _require_exact(persisted, expected, field="worker-local execution manifest")
    return _issue_execution_manifest(
        persisted,
        design=design,
        e0=e0,
        review=review,
        preflight=preflight,
        worker_receipts=worker_receipts,
        manifest_path=manifest_path,
        execution_source_paths=execution_source_paths,
    )


__all__ = [
    "ATTEMPT_ID",
    "EXECUTION_SOURCE_ROLES",
    "PREFLIGHT_SOURCE_ROLES",
    "ReservedTQCAttemptDirectoryV2",
    "ValidatedTQCExecutionManifestV2",
    "ValidatedTQCPreflightContractV2",
    "ValidatedTQCWorkerPreflightReceiptsV2",
    "publish_tqc_execution_manifest_v2",
    "publish_tqc_preflight_contract_v2",
    "reserve_tqc_attempt_directory_v2",
    "revalidate_tqc_execution_manifest_for_worker_v2",
    "revalidate_tqc_execution_manifest_v2",
    "revalidate_tqc_preflight_contract_for_worker_v2",
    "validate_persisted_tqc_preflight_contract_v2",
    "validate_tqc_worker_preflight_receipts_v2",
    "worker_bootstrap_receipt_bytes_v2",
    "worker_preflight_receipt_set_bytes_v2",
]
