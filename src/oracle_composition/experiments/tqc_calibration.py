"""Run the disposable TQC resource calibration."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact_io import PublishedArtifact, reserve_json_artifact
from .execution import _execution_callable_authority
from .fixed_reference import (
    CANONICAL_EXECUTION_CALLABLE_AUTHORITY,
    ExperimentContractError,
)
from .tqc_calibration_adapter import (
    HUMANOID_ACTION_HIGH,
    HUMANOID_ACTION_LOW,
    NORMALIZED_ACTION_HIGH,
    NORMALIZED_ACTION_LOW,
    REQUIRED_TRAINING_SIGNALS,
    _finite_array,
    _host_hardware_receipt,
    _nearest_existing_directory,
    _replay_arrays,
    calibration_source_sha256,
    fault_replay_buffer_pages,
    free_disk_bytes,
    inspect_tqc_model,
    inspect_tqc_runtime,
    make_tqc_model,
    make_tqc_vector_env,
    process_peak_rss_bytes,
    replay_buffer_allocation_bytes,
)
from .tqc_calibration_adapter import (
    assert_finite_parameters as _assert_finite_parameters,
)
from .tqc_calibration_adapter import (
    make_finite_resource_callback as _make_finite_resource_callback,
)
from .tqc_calibration_adapter import (
    parameter_receipt as _parameter_receipt,
)
from .tqc_calibration_adapter import (
    training_signal_snapshot as _training_signal_snapshot,
)
from .tqc_calibration_contract import (
    CALIBRATION_ID,
    MAX_DESIGN_BYTES,
    MODEL_DISPOSITION,
    SEED_ROLE,
    EnvironmentDesign,
    LoadedCalibrationDesign,
    PersistenceDesign,
    ResourceGates,
    TQCCalibrationDesign,
    TQCDesign,
    _boolean,
    _finite_number,
    _integer,
    _mapping,
    _require_exact_keys,
    _text,
    load_calibration_design,
)
from .tqc_calibration_contract import (
    canonical_json as _canonical_json,
)
from .tqc_calibration_contract import (
    sha256_json as _sha256_json,
)
from .tqc_calibration_contract import (
    validate_runtime_receipt as _validate_runtime_receipt,
)

CALIBRATION_RECEIPT_SCHEMA_VERSION = 1
RECEIPT_CLAIM = "resource_integrity_only_no_behavior_or_controller_claim/v1"


@dataclass(frozen=True, slots=True)
class CalibrationRunResult:
    published: PublishedArtifact
    receipt: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": str(self.published.path),
            "output_sha256": self.published.sha256,
            "output_byte_count": self.published.byte_count,
            "calibration_gate_passed": self.receipt["calibration_gate_passed"],
            "completion_status": self.receipt["completion_status"],
            "failure_stage": self.receipt["failure_stage"],
            "failure_reason": self.receipt["failure_reason"],
            "claim_boundary": self.receipt["claim_boundary"],
        }


def _verify_runtime_integrity(runtime: Mapping[str, Any], design: TQCCalibrationDesign) -> None:
    if dict(runtime) != inspect_tqc_runtime(design):
        raise ExperimentContractError("runtime integrity changed during calibration")


def _failure_text(exc: Exception) -> str:
    message = " ".join(str(exc).split())[:512]
    return f"{type(exc).__name__}: {message or 'no message'}"


def _receipt_base(
    *,
    loaded: LoadedCalibrationDesign,
    runtime: Mapping[str, Any] | None,
    callable_authority: str,
) -> dict[str, Any]:
    return {
        "schema_version": CALIBRATION_RECEIPT_SCHEMA_VERSION,
        "calibration_id": loaded.design.calibration_id,
        "evidence_purpose": loaded.design.evidence_purpose,
        "seed": loaded.design.seed,
        "seed_role": loaded.design.seed_role,
        "design_artifact_sha256": loaded.artifact_sha256,
        "design_artifact_byte_count": loaded.artifact_byte_count,
        "design_semantic_sha256": loaded.design.semantic_sha256,
        "design": loaded.design.to_dict(),
        "runtime": None if runtime is None else dict(runtime),
        "execution_callable_authority": callable_authority,
        "authoritative_execution": (callable_authority == CANONICAL_EXECUTION_CALLABLE_AUTHORITY),
        "claim_boundary": RECEIPT_CLAIM,
        "behavioral_claim": None,
        "controller_claim": None,
        "controller_artifact": None,
        "eligible_for_controller_training": False,
        "eligible_for_behavioral_evaluation": False,
        "automatic_promotion": False,
        "output_reservation": (
            "exclusive_descriptor_bound_provisional_json_then_atomic_exchange/v1"
        ),
        "checkpoint_emitted": False,
        "replay_buffer_emitted": False,
        "normalizer_emitted": False,
        "model_disposition": MODEL_DISPOSITION,
        "rss_measurement": (
            "getrusage_RUSAGE_SELF_process_lifetime_high_water_bytes/v1"
            if callable_authority == CANONICAL_EXECUTION_CALLABLE_AUTHORITY
            else "injected_test_reader_non_authoritative"
        ),
        "rss_threshold_semantics": "sampled_failure_threshold_not_os_memory_cap/v1",
        "rss_transient_above_threshold_possible": True,
        "oom_before_final_receipt_possible": True,
        "disk_measurement": (
            "shutil_disk_usage_nearest_existing_output_ancestor/v1"
            if callable_authority == CANONICAL_EXECUTION_CALLABLE_AUTHORITY
            else "injected_test_reader_non_authoritative"
        ),
    }


def run_tqc_calibration(
    *,
    design_path: Path,
    output_path: Path,
    confirm_disposable_resource_probe: bool = False,
    runtime_inspector: Callable[[TQCCalibrationDesign], dict[str, Any]] = (inspect_tqc_runtime),
    vector_env_factory: Callable[[TQCCalibrationDesign], object] = make_tqc_vector_env,
    model_factory: Callable[[TQCCalibrationDesign, object], object] = make_tqc_model,
    clock: Callable[[], float] = time.perf_counter,
    peak_rss_reader: Callable[[], int] = process_peak_rss_bytes,
    free_disk_reader: Callable[[Path], int] = free_disk_bytes,
) -> CalibrationRunResult:
    output_path = Path(output_path)
    if output_path.exists() or output_path.is_symlink():
        raise ExperimentContractError(f"refusing to overwrite existing file: {output_path}")
    loaded = load_calibration_design(design_path)
    if confirm_disposable_resource_probe is not True:
        raise ExperimentContractError(
            "explicit --confirm-disposable-resource-probe acknowledgement is required"
        )
    design = loaded.design
    callable_authority = _execution_callable_authority(
        (runtime_inspector, inspect_tqc_runtime),
        (vector_env_factory, make_tqc_vector_env),
        (model_factory, make_tqc_model),
        (clock, time.perf_counter),
        (peak_rss_reader, process_peak_rss_bytes),
        (free_disk_reader, free_disk_bytes),
    )
    receipt = _receipt_base(
        loaded=loaded,
        runtime=None,
        callable_authority=callable_authority,
    )
    receipt["disposable_resource_probe_acknowledged"] = True
    receipt["completion_status"] = "in_progress"
    reservation = reserve_json_artifact(output_path, receipt)
    try:
        runtime: dict[str, Any] | None = None
        environment: object | None = None
        model: object | None = None
        callback: object | None = None
        observed_timesteps: int | None = None
        observed_updates: int | None = None
        allocation_bytes: int | None = None
        model_runtime: dict[str, Any] | None = None
        final_parameters: dict[str, Any] | None = None
        page_write: dict[str, Any] | None = None
        wall_seconds: float | None = None
        training_seconds: float | None = None
        throughput: float | None = None
        training_started: float | None = None
        peak_rss_before_model_creation: int | None = None
        peak_rss_before_page_write: int | None = None
        failure: str | None = None
        cleanup_failures: list[str] = []
        stage = "runtime_inspection"
        started = clock()
        try:
            runtime = _validate_runtime_receipt(runtime_inspector(design))
            receipt["runtime"] = runtime
            stage = "preflight"
            callback = _make_finite_resource_callback(
                design=design,
                output_parent=output_path.parent,
                peak_rss_reader=peak_rss_reader,
                free_disk_reader=free_disk_reader,
                clock=clock,
            )
            callback.check_resources(include_disk=True)
            peak_rss_before_model_creation = callback.maximum_peak_rss_bytes
            stage = "vector_environment_creation"
            environment = vector_env_factory(design)
            stage = "model_creation"
            model = model_factory(design, environment)
            allocation_bytes = replay_buffer_allocation_bytes(model)
            if callable_authority == CANONICAL_EXECUTION_CALLABLE_AUTHORITY:
                model_runtime = inspect_tqc_model(model, design)
                if (
                    model_runtime["observation_space_sha256"] != runtime["observation_space_sha256"]
                    or model_runtime["action_space_sha256"] != runtime["action_space_sha256"]
                ):
                    raise ExperimentContractError("TQC model spaces differ from inspected runtime")
            else:
                model_runtime = {
                    "authority": "injected_test_model_non_authoritative",
                    "replay_allocation_bytes": allocation_bytes,
                    "initial_parameters": _parameter_receipt(model),
                }
            callback.model = model
            callback.check_resources(include_disk=True)
            callback.check_parameters()
            peak_rss_before_page_write = callback.maximum_peak_rss_bytes
            stage = "replay_buffer_page_write"
            page_write = fault_replay_buffer_pages(
                model,
                resource_check=lambda: callback.check_resources(include_disk=False),
            )
            callback.check_resources(include_disk=True)
            peak_rss_after_write = callback.maximum_peak_rss_bytes
            page_write["claim_boundary"] = "page_addresses_written_no_residency_claim/v1"
            page_write["peak_rss_high_water_bytes_before_model"] = peak_rss_before_model_creation
            page_write["peak_rss_high_water_bytes_before_write"] = peak_rss_before_page_write
            page_write["peak_rss_high_water_bytes_after_write"] = peak_rss_after_write
            page_write["peak_rss_delta_from_before_model_bytes"] = (
                peak_rss_after_write - peak_rss_before_model_creation
            )
            page_write["peak_rss_delta_from_before_write_bytes"] = (
                peak_rss_after_write - peak_rss_before_page_write
            )
            page_write["minimum_free_disk_bytes_through_write"] = callback.minimum_free_disk_bytes
            training_started = clock()
            stage = "learning"
            model.learn(
                total_timesteps=design.tqc.total_timesteps,
                callback=callback,
                log_interval=1000,
                reset_num_timesteps=True,
                progress_bar=False,
            )
            observed_timesteps = int(getattr(model, "num_timesteps", -1))
            observed_updates = int(getattr(model, "_n_updates", -1))
            callback.training_signals.update(_training_signal_snapshot(model))
            callback.check_parameters()
            final_parameters = _parameter_receipt(model)
            if (
                model_runtime["initial_parameters"]["structure_sha256"]
                != final_parameters["structure_sha256"]
            ):
                raise ExperimentContractError("TQC parameter structure changed during calibration")
            callback.check_resources(include_disk=True)
            if observed_timesteps != design.tqc.total_timesteps:
                raise ExperimentContractError(
                    "TQC did not complete the exact environment-step budget"
                )
            if int(getattr(callback, "n_calls", -1)) != design.expected_vector_steps:
                raise ExperimentContractError(
                    "TQC callback count did not complete the vector-step budget"
                )
            if observed_updates != design.expected_updates:
                raise ExperimentContractError("TQC update count did not match the frozen schedule")
            missing = REQUIRED_TRAINING_SIGNALS - set(callback.training_signals)
            if missing:
                raise ExperimentContractError(
                    f"TQC did not expose every required finite training signal: {sorted(missing)}"
                )
            training_ended = clock()
            training_seconds = training_ended - training_started
            if not math.isfinite(training_seconds) or training_seconds <= 0.0:
                raise ExperimentContractError("calibration training time is invalid")
            throughput = observed_timesteps / training_seconds
            stage = "integrity_recheck"
            if callable_authority == CANONICAL_EXECUTION_CALLABLE_AUTHORITY:
                _verify_runtime_integrity(runtime, design)
                callback.check_resources(include_disk=True)
            wall_seconds = training_ended - started
            if not math.isfinite(wall_seconds) or wall_seconds <= 0.0:
                raise ExperimentContractError("calibration elapsed time is invalid")
            stage = "throughput_gate"
            if throughput < design.resource_gates.minimum_environment_steps_per_second:
                raise ExperimentContractError(
                    "environment-step throughput missed the calibration gate"
                )
        except Exception as exc:
            failure = _failure_text(exc)
        finally:
            if model is not None:
                if observed_timesteps is None:
                    value = getattr(model, "num_timesteps", None)
                    if isinstance(value, int) and not isinstance(value, bool):
                        observed_timesteps = value
                if observed_updates is None:
                    value = getattr(model, "_n_updates", None)
                    if isinstance(value, int) and not isinstance(value, bool):
                        observed_updates = value
            if environment is not None:
                close = getattr(environment, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:
                        cleanup_failure = _failure_text(exc)
                        cleanup_failures.append(cleanup_failure)
                        if failure is None:
                            stage = "environment_close"
                            failure = cleanup_failure
            if callback is not None:
                callback.model = None
                callback.locals = {}
                callback.globals = {}
            model = None
            environment = None
            gc.collect()

        if wall_seconds is None:
            ended = clock()
            observed_wall = ended - started
            if math.isfinite(observed_wall) and observed_wall > 0.0:
                wall_seconds = observed_wall
            if training_started is not None:
                observed_training = ended - training_started
                if math.isfinite(observed_training) and observed_training > 0.0:
                    training_seconds = observed_training
                    if observed_timesteps is not None and observed_timesteps > 0:
                        throughput = observed_timesteps / training_seconds

        measured_gates_passed = failure is None
        passed = measured_gates_passed and (
            callable_authority == CANONICAL_EXECUTION_CALLABLE_AUTHORITY
        )
        receipt.update(
            {
                "completion_status": "complete" if failure is None else "failed",
                "measured_workload_gates_passed": measured_gates_passed,
                "calibration_gate_passed": passed,
                "failure_stage": None if failure is None else stage,
                "failure_reason": failure,
                "cleanup_failures": cleanup_failures,
                "expected_environment_steps": design.tqc.total_timesteps,
                "observed_environment_steps": observed_timesteps,
                "expected_vector_steps": design.expected_vector_steps,
                "observed_vector_steps": (
                    None if callback is None else int(getattr(callback, "n_calls", -1))
                ),
                "expected_gradient_updates": design.expected_updates,
                "observed_gradient_updates": observed_updates,
                "wall_seconds_from_preflight_through_learning": wall_seconds,
                "training_wall_seconds": training_seconds,
                "environment_steps_per_second": throughput,
                "replay_buffer_allocation_bytes": allocation_bytes,
                "observed_model": model_runtime,
                "final_parameters": final_parameters,
                "replay_buffer_page_write": page_write,
                "peak_rss_bytes": (
                    None
                    if callback is None
                    else int(getattr(callback, "maximum_peak_rss_bytes", 0))
                ),
                "minimum_free_disk_bytes": (
                    None if callback is None else getattr(callback, "minimum_free_disk_bytes", None)
                ),
                "finite_rollout_signal_checks": (
                    0 if callback is None else int(getattr(callback, "signal_checks", 0))
                ),
                "finite_parameter_checks": (
                    0 if callback is None else int(getattr(callback, "parameter_checks", 0))
                ),
                "resource_checks": (
                    0 if callback is None else int(getattr(callback, "resource_checks", 0))
                ),
                "parameter_count": (
                    None if callback is None else int(getattr(callback, "parameter_count", 0))
                ),
                "finite_training_signals": (
                    {} if callback is None else dict(getattr(callback, "training_signals", {}))
                ),
                "sustained_throughput_windows": (
                    [] if callback is None else list(getattr(callback, "throughput_windows", []))
                ),
            }
        )
        published = reservation.finalize(receipt)
        return CalibrationRunResult(published=published, receipt=receipt)
    finally:
        reservation.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confirm-disposable-resource-probe",
        action="store_true",
        help="confirm a 5.26-GiB replay probe with a sampled 12-GiB RSS failure threshold",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_tqc_calibration(
            design_path=args.design,
            output_path=args.output,
            confirm_disposable_resource_probe=args.confirm_disposable_resource_probe,
        )
    except (ExperimentContractError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), sort_keys=True, ensure_ascii=False))
    return 0 if result.receipt["calibration_gate_passed"] else 2


__all__ = [
    "CALIBRATION_ID",
    "CALIBRATION_RECEIPT_SCHEMA_VERSION",
    "HUMANOID_ACTION_HIGH",
    "HUMANOID_ACTION_LOW",
    "MAX_DESIGN_BYTES",
    "MODEL_DISPOSITION",
    "NORMALIZED_ACTION_HIGH",
    "NORMALIZED_ACTION_LOW",
    "RECEIPT_CLAIM",
    "REQUIRED_TRAINING_SIGNALS",
    "SEED_ROLE",
    "CalibrationRunResult",
    "EnvironmentDesign",
    "LoadedCalibrationDesign",
    "PersistenceDesign",
    "ResourceGates",
    "TQCCalibrationDesign",
    "TQCDesign",
    "_assert_finite_parameters",
    "_boolean",
    "_canonical_json",
    "_failure_text",
    "_finite_array",
    "_finite_number",
    "_host_hardware_receipt",
    "_integer",
    "_make_finite_resource_callback",
    "_mapping",
    "_nearest_existing_directory",
    "_parameter_receipt",
    "_parser",
    "_receipt_base",
    "_replay_arrays",
    "_require_exact_keys",
    "_sha256_json",
    "_text",
    "_training_signal_snapshot",
    "_validate_runtime_receipt",
    "_verify_runtime_integrity",
    "calibration_source_sha256",
    "fault_replay_buffer_pages",
    "free_disk_bytes",
    "inspect_tqc_model",
    "inspect_tqc_runtime",
    "load_calibration_design",
    "main",
    "make_tqc_model",
    "make_tqc_vector_env",
    "process_peak_rss_bytes",
    "replay_buffer_allocation_bytes",
    "run_tqc_calibration",
]


if __name__ == "__main__":
    raise SystemExit(main())
