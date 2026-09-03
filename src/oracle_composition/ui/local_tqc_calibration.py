"""Fail-closed local view of the reviewed TQC resource receipt."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path

TQC_CALIBRATION_RECEIPT_PATH = Path(
    "artifacts/bootstrap_tqc_humanoid/resource_calibration_seed_92001.json"
)
TQC_CALIBRATION_DESIGN_PATH = Path(
    "experiments/bootstrap_tqc_humanoid/configs/tqc_resource_calibration_v0.study.json"
)
TQC_CALIBRATION_SOURCE_PATH = Path("src/oracle_composition/experiments/tqc_calibration.py")
TQC_CALIBRATION_FIGURE_PATH = Path(
    "artifacts/bootstrap_tqc_humanoid/resource_calibration_seed_92001.png"
)

_AUTHORITY = "local_reviewed_resource_calibration_receipt"
_RECEIPT_LIMIT = 64 * 1024
_DESIGN_LIMIT = 128 * 1024
_SOURCE_LIMIT = 2 * 1024 * 1024
_REVIEWED_RECEIPT_SHA256 = "2436c2e93cac8b5ed357acdcb19cca0b6222792a1c9c0d65f06577509d68a587"
_REVIEWED_DESIGN_SHA256 = "5e911653e6be94bed73a60322565ba6d017a95e88f12b04b2eb43578d7b04b7d"
_REVIEWED_SOURCE_SHA256 = "118032a1ab9488678b907f535396da239b9190bbff7697ca5c967ed6ca77007c"
_REVIEWED_FIGURE_SHA256 = "ce6a2b690a1156a86d80660233ff8a5387dabb5d4b72bec0b49af01f9314d188"
_CALIBRATION_ID = "tqc_humanoid_resource_calibration/v1"
_CLAIM_BOUNDARY = "resource_integrity_only_no_behavior_or_controller_claim/v1"
_SEED_ROLE = "permanently_excluded_disposable_calibration"
_MODEL_DISPOSITION = "discarded_unserialized"
_TOP_LEVEL_FIELDS = frozenset(
    {
        "authoritative_execution",
        "automatic_promotion",
        "behavioral_claim",
        "calibration_gate_passed",
        "calibration_id",
        "checkpoint_emitted",
        "claim_boundary",
        "cleanup_failures",
        "completion_status",
        "controller_artifact",
        "controller_claim",
        "design",
        "design_artifact_byte_count",
        "design_artifact_sha256",
        "design_semantic_sha256",
        "disk_measurement",
        "disposable_resource_probe_acknowledged",
        "eligible_for_behavioral_evaluation",
        "eligible_for_controller_training",
        "environment_steps_per_second",
        "evidence_purpose",
        "execution_callable_authority",
        "expected_environment_steps",
        "expected_gradient_updates",
        "expected_vector_steps",
        "failure_reason",
        "failure_stage",
        "final_parameters",
        "finite_parameter_checks",
        "finite_rollout_signal_checks",
        "finite_training_signals",
        "measured_workload_gates_passed",
        "minimum_free_disk_bytes",
        "model_disposition",
        "normalizer_emitted",
        "observed_environment_steps",
        "observed_gradient_updates",
        "observed_model",
        "observed_vector_steps",
        "oom_before_final_receipt_possible",
        "output_reservation",
        "parameter_count",
        "peak_rss_bytes",
        "replay_buffer_allocation_bytes",
        "replay_buffer_emitted",
        "replay_buffer_page_write",
        "resource_checks",
        "rss_measurement",
        "rss_threshold_semantics",
        "rss_transient_above_threshold_possible",
        "runtime",
        "schema_version",
        "seed",
        "seed_role",
        "sustained_throughput_windows",
        "training_wall_seconds",
        "wall_seconds_from_preflight_through_learning",
    }
)


class LocalTQCCalibrationError(ValueError):
    """Raised when the local E0 receipt fails the reviewed contract."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LocalTQCCalibrationError("receipt contains a non-canonical JSON value") from exc


def _read_project_file(root: Path, relative: Path, *, limit: int) -> bytes:
    if relative.is_absolute() or not relative.parts:
        raise LocalTQCCalibrationError("unsafe local receipt path")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise LocalTQCCalibrationError("unsafe local receipt path")

    descriptors: list[int] = []
    try:
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        current = os.open(root.resolve(strict=True), directory_flags)
        descriptors.append(current)
        for part in relative.parts[:-1]:
            current = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(current)
        file_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(relative.name, file_flags, dir_fd=current)
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise LocalTQCCalibrationError("local receipt is not a non-empty regular file")
        if before.st_size > limit:
            raise LocalTQCCalibrationError("local receipt exceeds its size limit")
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise LocalTQCCalibrationError("local receipt changed while it was read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in identity):
            raise LocalTQCCalibrationError("local receipt changed while it was read")
        return b"".join(chunks)
    except FileNotFoundError:
        raise
    except LocalTQCCalibrationError:
        raise
    except OSError as exc:
        raise LocalTQCCalibrationError("local receipt cannot be read safely") from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _reject_constant(value: str) -> None:
    raise LocalTQCCalibrationError(f"non-finite JSON constant is forbidden: {value}")


def _without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise LocalTQCCalibrationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _decode(source: bytes, *, artifact: str) -> dict[str, object]:
    try:
        value = json.loads(
            source.decode("utf-8", errors="strict"),
            object_pairs_hook=_without_duplicates,
            parse_constant=_reject_constant,
        )
    except LocalTQCCalibrationError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise LocalTQCCalibrationError(f"{artifact} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise LocalTQCCalibrationError(f"{artifact} root must be an object")
    return value


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise LocalTQCCalibrationError(f"{field} must be an object")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise LocalTQCCalibrationError(f"{field} must be an integer >= {minimum}")
    return value


def _number(value: object, field: str, *, minimum: float = 0.0) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise LocalTQCCalibrationError(f"{field} must be numeric")
    observed = float(value)
    if not math.isfinite(observed) or observed < minimum:
        raise LocalTQCCalibrationError(f"{field} is outside its finite range")
    return observed


def _digest(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise LocalTQCCalibrationError(f"{field} must be a lowercase SHA-256")
    return value


def _validate_semantic_hash(value: dict[str, object], field: str, label: str) -> None:
    payload = dict(value)
    claimed = _digest(payload.pop(field, None), f"{label} {field}")
    if _sha256(_canonical_json(payload)) != claimed:
        raise LocalTQCCalibrationError(f"{label} semantic SHA-256 is inconsistent")


def _validate_design(
    receipt: dict[str, object],
    design: dict[str, object],
    design_source: bytes,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    if _sha256(design_source) != _REVIEWED_DESIGN_SHA256:
        raise LocalTQCCalibrationError("design identity differs from the reviewed artifact")
    if receipt.get("design") != design:
        raise LocalTQCCalibrationError("embedded design differs from the reviewed artifact")
    if receipt.get("design_artifact_byte_count") != len(design_source):
        raise LocalTQCCalibrationError("design byte count is inconsistent")
    if receipt.get("design_artifact_sha256") != _REVIEWED_DESIGN_SHA256:
        raise LocalTQCCalibrationError("design SHA-256 is inconsistent")
    if receipt.get("design_semantic_sha256") != _sha256(_canonical_json(design)):
        raise LocalTQCCalibrationError("design semantic SHA-256 is inconsistent")

    environment = _mapping(design.get("environment"), "design environment")
    tqc = _mapping(design.get("tqc"), "design TQC")
    gates = _mapping(design.get("resource_gates"), "design resource gates")
    expected = {
        "schema_version": 1,
        "calibration_id": _CALIBRATION_ID,
        "evidence_purpose": "resource_calibration",
        "seed": 92001,
        "seed_role": _SEED_ROLE,
        "claim_ceiling": None,
        "design_status": "predeclared_not_executed",
    }
    if any(design.get(field) != value for field, value in expected.items()):
        raise LocalTQCCalibrationError("design identity or claim ceiling is inconsistent")
    if (
        environment.get("environment_id") != "Humanoid-v5"
        or environment.get("n_envs") != 5
        or environment.get("worker_seeds") != [92001, 92002, 92003, 92004, 92005]
        or environment.get("terminate_when_unhealthy") is not False
        or environment.get("normalization") != "none/v1"
    ):
        raise LocalTQCCalibrationError("environment design differs from reviewed E0")
    if (
        tqc.get("algorithm_id") != "sb3_contrib.TQC/v2.9"
        or tqc.get("device") != "cpu"
        or tqc.get("total_timesteps") != 100000
        or tqc.get("required_stable_baselines3_version") != "2.9.0"
        or tqc.get("required_sb3_contrib_version") != "2.9.0"
    ):
        raise LocalTQCCalibrationError("TQC design differs from reviewed E0")
    return environment, tqc, gates


def _validate_success(receipt: dict[str, object]) -> None:
    expected = {
        "schema_version": 1,
        "calibration_id": _CALIBRATION_ID,
        "evidence_purpose": "resource_calibration",
        "seed": 92001,
        "seed_role": _SEED_ROLE,
        "completion_status": "complete",
        "claim_boundary": _CLAIM_BOUNDARY,
        "behavioral_claim": None,
        "controller_claim": None,
        "controller_artifact": None,
        "eligible_for_controller_training": False,
        "eligible_for_behavioral_evaluation": False,
        "automatic_promotion": False,
        "checkpoint_emitted": False,
        "replay_buffer_emitted": False,
        "normalizer_emitted": False,
        "model_disposition": _MODEL_DISPOSITION,
        "disposable_resource_probe_acknowledged": True,
        "authoritative_execution": True,
        "measured_workload_gates_passed": True,
        "calibration_gate_passed": True,
        "failure_stage": None,
        "failure_reason": None,
        "cleanup_failures": [],
        "expected_environment_steps": 100000,
        "observed_environment_steps": 100000,
        "expected_vector_steps": 20000,
        "observed_vector_steps": 20000,
        "expected_gradient_updates": 19980,
        "observed_gradient_updates": 19980,
        "rss_threshold_semantics": "sampled_failure_threshold_not_os_memory_cap/v1",
        "rss_transient_above_threshold_possible": True,
        "oom_before_final_receipt_possible": True,
    }
    if any(receipt.get(field) != value for field, value in expected.items()):
        raise LocalTQCCalibrationError("receipt does not represent the reviewed successful E0 gate")


def _validate_resource_measurements(
    receipt: dict[str, object], gates: dict[str, object]
) -> tuple[float, int, int]:
    throughput = _number(
        receipt.get("environment_steps_per_second"), "environment throughput", minimum=0.0
    )
    peak_rss = _integer(receipt.get("peak_rss_bytes"), "peak RSS", minimum=1)
    free_disk = _integer(receipt.get("minimum_free_disk_bytes"), "free disk", minimum=1)
    throughput_gate = _number(
        gates.get("minimum_environment_steps_per_second"), "throughput gate", minimum=0.0
    )
    rss_gate = _integer(
        gates.get("sampled_peak_rss_failure_threshold_bytes"),
        "sampled RSS threshold",
        minimum=1,
    )
    disk_gate = _integer(gates.get("minimum_free_disk_bytes"), "disk threshold", minimum=1)
    if throughput < throughput_gate or peak_rss > rss_gate or free_disk < disk_gate:
        raise LocalTQCCalibrationError("resource measurements do not satisfy the reviewed gates")

    windows = receipt.get("sustained_throughput_windows")
    if not isinstance(windows, list) or len(windows) != 9:
        raise LocalTQCCalibrationError("sustained throughput coverage is incomplete")
    for index, value in enumerate(windows, start=1):
        window = _mapping(value, "throughput window")
        if (
            window.get("start_environment_step") != index * 10000
            or window.get("end_environment_step") != (index + 1) * 10000
            or _number(
                window.get("environment_steps_per_second"),
                "window throughput",
                minimum=0.0,
            )
            < throughput_gate
        ):
            raise LocalTQCCalibrationError("sustained throughput window failed its fixed gate")
    return throughput, peak_rss, free_disk


def _validate_runtime(receipt: dict[str, object], source: bytes) -> dict[str, object]:
    if _sha256(source) != _REVIEWED_SOURCE_SHA256:
        raise LocalTQCCalibrationError("calibration source differs from the reviewed execution")
    runtime = _mapping(receipt.get("runtime"), "runtime")
    _validate_semantic_hash(runtime, "runtime_sha256", "runtime")
    if (
        runtime.get("calibration_source_sha256") != _REVIEWED_SOURCE_SHA256
        or runtime.get("environment_id") != "Humanoid-v5"
        or runtime.get("stable_baselines3_version") != "2.9.0"
        or runtime.get("sb3_contrib_version") != "2.9.0"
        or runtime.get("observation_shape") != [348]
        or runtime.get("action_shape") != [17]
    ):
        raise LocalTQCCalibrationError("runtime identity differs from the reviewed execution")
    _digest(runtime.get("source_tree_sha256"), "run-time source tree SHA-256")
    _digest(runtime.get("dependency_lock_sha256"), "dependency lock SHA-256")
    return runtime


def _validate_model_and_buffer(receipt: dict[str, object]) -> None:
    model = _mapping(receipt.get("observed_model"), "observed model")
    _validate_semantic_hash(model, "model_semantic_sha256", "observed model")
    if (
        model.get("algorithm_class") != "sb3_contrib.tqc.tqc.TQC"
        or model.get("device") != "cpu"
        or model.get("n_envs") != 5
        or model.get("vec_normalize_active") is not False
        or model.get("logger_directory") is not None
        or model.get("logger_output_format_count") != 0
        or model.get("pending_worker_seeds") != [92001, 92002, 92003, 92004, 92005]
    ):
        raise LocalTQCCalibrationError("observed model differs from the reviewed E0 model")
    initial = _mapping(model.get("initial_parameters"), "initial parameters")
    final = _mapping(receipt.get("final_parameters"), "final parameters")
    if initial.get("structure_sha256") != final.get("structure_sha256"):
        raise LocalTQCCalibrationError("parameter structure changed during E0")
    if initial.get("value_sha256") == final.get("value_sha256"):
        raise LocalTQCCalibrationError("parameter values did not change during E0")
    for group in (initial, final):
        _digest(group.get("structure_sha256"), "parameter structure SHA-256")
        _digest(group.get("value_sha256"), "parameter value SHA-256")

    page_write = _mapping(receipt.get("replay_buffer_page_write"), "replay page write")
    allocation = _integer(
        receipt.get("replay_buffer_allocation_bytes"), "replay allocation", minimum=1
    )
    if (
        page_write.get("claim_boundary") != "page_addresses_written_no_residency_claim/v1"
        or page_write.get("method") != "write_zero_every_os_page_plus_final_byte/v1"
        or page_write.get("total_array_bytes") != allocation
        or model.get("replay_allocation_bytes") != allocation
        or model.get("expected_replay_allocation_bytes") != allocation
    ):
        raise LocalTQCCalibrationError("replay allocation receipt is inconsistent")


def _validated_status(
    receipt: dict[str, object],
    receipt_source: bytes,
    design: dict[str, object],
    design_source: bytes,
    calibration_source: bytes,
) -> dict[str, object]:
    if set(receipt) != _TOP_LEVEL_FIELDS:
        raise LocalTQCCalibrationError("receipt fields differ from schema version 1")
    if _sha256(receipt_source) != _REVIEWED_RECEIPT_SHA256:
        raise LocalTQCCalibrationError("receipt identity differs from the reviewed E0 artifact")
    environment, _tqc, gates = _validate_design(receipt, design, design_source)
    _validate_success(receipt)
    throughput, peak_rss, free_disk = _validate_resource_measurements(receipt, gates)
    runtime = _validate_runtime(receipt, calibration_source)
    _validate_model_and_buffer(receipt)

    return {
        "state": "available",
        "authority": _AUTHORITY,
        "evidence_class": "E0_resource_only",
        "gate_passed": True,
        "claim": (
            "The disposable 100,000-step TQC workload completed within its "
            "predeclared sampled resource and integrity gates."
        ),
        "environment_id": "Humanoid-v5",
        "seed": 92001,
        "seed_role": _SEED_ROLE,
        "worker_seeds": list(environment["worker_seeds"]),
        "observed_environment_steps": receipt["observed_environment_steps"],
        "observed_gradient_updates": receipt["observed_gradient_updates"],
        "environment_steps_per_second": throughput,
        "minimum_environment_steps_per_second": gates["minimum_environment_steps_per_second"],
        "sustained_throughput_window_count": 9,
        "training_wall_seconds": _number(
            receipt.get("training_wall_seconds"), "training wall seconds", minimum=0.0
        ),
        "peak_rss_bytes": peak_rss,
        "sampled_peak_rss_failure_threshold_bytes": gates[
            "sampled_peak_rss_failure_threshold_bytes"
        ],
        "minimum_free_disk_bytes": free_disk,
        "minimum_free_disk_gate_bytes": gates["minimum_free_disk_bytes"],
        "replay_buffer_allocation_bytes": receipt["replay_buffer_allocation_bytes"],
        "checkpoint_emitted": False,
        "model_disposition": _MODEL_DISPOSITION,
        "eligible_for_controller_training": False,
        "eligible_for_behavioral_evaluation": False,
        "limitations": [
            "No checkpoint, replay buffer, or normalizer was retained.",
            "The trained model was discarded without serialization.",
            (
                "No balance, locomotion, reference tracking, oracle, task-reward, "
                "or controller capability was evaluated."
            ),
            (
                "RSS is a sampled process high-water measurement; the threshold was "
                "not an operating-system memory cap."
            ),
            "Replay page addresses were written; physical residency was not established.",
            "The calibration seed and five worker seeds are excluded from formal studies.",
        ],
        "next_gate": "E1 initialization identity and successful checkpoint-load proof",
        "receipts": {
            "receipt_sha256": _REVIEWED_RECEIPT_SHA256,
            "design_sha256": _REVIEWED_DESIGN_SHA256,
            "calibration_source_sha256": _REVIEWED_SOURCE_SHA256,
            "run_source_tree_sha256": runtime["source_tree_sha256"],
            "runtime_sha256": runtime["runtime_sha256"],
            "dependency_lock_sha256": runtime["dependency_lock_sha256"],
        },
    }


def local_tqc_calibration_figure(project_root: Path) -> bytes:
    """Return the exact reviewed E0 figure bytes."""

    source = _read_project_file(Path(project_root), TQC_CALIBRATION_FIGURE_PATH, limit=512 * 1024)
    if _sha256(source) != _REVIEWED_FIGURE_SHA256:
        raise LocalTQCCalibrationError("E0 figure identity differs from the reviewed artifact")
    if not source.startswith(b"\x89PNG\r\n\x1a\n"):
        raise LocalTQCCalibrationError("E0 figure is not a PNG")
    return source


def local_tqc_calibration_status(project_root: Path) -> dict[str, object]:
    """Return a compact view only when every reviewed E0 binding still holds."""

    root = Path(project_root)
    try:
        receipt_source = _read_project_file(
            root, TQC_CALIBRATION_RECEIPT_PATH, limit=_RECEIPT_LIMIT
        )
    except FileNotFoundError:
        return {
            "state": "unavailable",
            "authority": _AUTHORITY,
            "detail": "no reviewed local E0 TQC resource receipt is present",
        }
    except LocalTQCCalibrationError as exc:
        return {"state": "rejected", "authority": _AUTHORITY, "detail": str(exc)}

    try:
        design_source = _read_project_file(root, TQC_CALIBRATION_DESIGN_PATH, limit=_DESIGN_LIMIT)
        calibration_source = _read_project_file(
            root, TQC_CALIBRATION_SOURCE_PATH, limit=_SOURCE_LIMIT
        )
        result = _validated_status(
            _decode(receipt_source, artifact="E0 receipt"),
            receipt_source,
            _decode(design_source, artifact="E0 design"),
            design_source,
            calibration_source,
        )
    except (FileNotFoundError, LocalTQCCalibrationError) as exc:
        detail = (
            "a reviewed E0 binding is unavailable"
            if isinstance(exc, FileNotFoundError)
            else str(exc)
        )
        return {"state": "rejected", "authority": _AUTHORITY, "detail": detail}

    try:
        local_tqc_calibration_figure(root)
    except (FileNotFoundError, LocalTQCCalibrationError):
        result["media"] = None
    else:
        result["media"] = {"figure": "/local-evidence/e0-tqc-resource.png"}
        receipts = _mapping(result["receipts"], "compact receipt bindings")
        receipts["figure_sha256"] = _REVIEWED_FIGURE_SHA256
    return result


__all__ = [
    "TQC_CALIBRATION_DESIGN_PATH",
    "TQC_CALIBRATION_FIGURE_PATH",
    "TQC_CALIBRATION_RECEIPT_PATH",
    "TQC_CALIBRATION_SOURCE_PATH",
    "LocalTQCCalibrationError",
    "local_tqc_calibration_figure",
    "local_tqc_calibration_status",
]
