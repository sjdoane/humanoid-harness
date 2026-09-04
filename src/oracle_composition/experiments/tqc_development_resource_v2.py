"""Process-local resource gates for the frozen TQC development attempt."""

from __future__ import annotations

import hashlib
import math
import os
import platform
import resource
import stat
import time
from dataclasses import InitVar, asdict, dataclass
from pathlib import Path

from .fixed_reference import ExperimentContractError
from .tqc_calibration_contract import canonical_json
from .tqc_development_contract_v2 import (
    EXPECTED_DISK_CHECKS,
    EXPECTED_VECTOR_STEPS,
    TOTAL_ENVIRONMENT_STEPS,
)

RESOURCE_MONITOR_ID = "tqc_dev_1m_v2_process_resource_monitor/v1"
ATTEMPT_ID = "dev1m-v2-seed-95001-attempt-01"
ENVIRONMENT_STEPS_PER_VECTOR_STEP = 5
PEAK_RSS_LIMIT_BYTES = 12_884_901_888
FREE_DISK_MINIMUM_BYTES = 53_687_091_200
THROUGHPUT_MINIMUM_STEPS_PER_SECOND = 116.0
THROUGHPUT_WARMUP_ENVIRONMENT_STEPS = 10_000
THROUGHPUT_WINDOW_ENVIRONMENT_STEPS = 10_000
EXPECTED_THROUGHPUT_WINDOWS = 99
WALL_TIME_LIMIT_SECONDS = 10_800.0
EXPECTED_TRAINING_PREFIX_DISK_CHECKS = 2_001

LIFECYCLE_STAGES = (
    "preflight",
    "post_model_construction",
    "pre_learn",
    "post_learn",
    "pre_model_save",
    "post_model_save",
    "pre_replay_save",
    "post_replay_save",
    "pre_actor_export",
    "post_actor_export",
    "pre_evaluation",
    "post_evaluation",
    "pre_visual_encoding",
    "post_visual_encoding",
    "pre_finalization",
    "post_finalization",
)
TRAINING_PREFIX_STAGES = LIFECYCLE_STAGES[:4]
POST_TRAINING_DISK_STAGES = (
    "pre_persistence",
    "pre_evaluation",
    "finalization",
)

_TRAINING_AUTHORITY_ISSUER = object()
_FINAL_RECEIPT_ISSUER = object()
_PERF_COUNTER = time.perf_counter
_GETRUSAGE = resource.getrusage
_FSTATVFS = os.fstatvfs


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"{field} must be a lowercase SHA-256")
    return value


def _open_work_directory(path: Path) -> tuple[Path, int, os.stat_result]:
    """Open one absolute directory without following any path-component link."""

    requested = Path(path)
    if not requested.is_absolute():
        raise ExperimentContractError("resource output path must be absolute")
    absolute = Path(os.path.abspath(requested))
    if requested != absolute:
        raise ExperimentContractError("claimed work directory path must already be normalized")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(absolute.anchor, flags)
        for component in absolute.parts[1:]:
            if component in {"", ".", ".."}:
                raise ExperimentContractError("claimed work directory path is unsafe")
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(observed.st_mode)
            or stat.S_IMODE(observed.st_mode) != 0o700
            or observed.st_uid != os.geteuid()
        ):
            raise ExperimentContractError(
                "claimed work directory must be an owned 0700 real directory"
            )
        return absolute, descriptor, observed
    except ExperimentContractError:
        if descriptor is not None:
            os.close(descriptor)
        raise
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise ExperimentContractError(
            "claimed work directory and its ancestors must be real directories"
        ) from exc


def _directory_state(value: os.stat_result) -> tuple[int, ...]:
    # APFS changes a directory's link count when regular children are published.
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_uid),
        int(value.st_gid),
    )


def _identity_from_open_directory(path: Path, observed: os.stat_result) -> str:
    payload = {
        "absolute_path": str(path),
        "device": int(observed.st_dev),
        "inode": int(observed.st_ino),
        "owner_uid": int(observed.st_uid),
        "owner_gid": int(observed.st_gid),
        "mode_octal": "0700",
    }
    return _sha256(canonical_json(payload))


def claimed_work_directory_identity(path: Path) -> str:
    """Hash the exact existing 0700 directory used by one attempt."""

    absolute, descriptor, observed = _open_work_directory(Path(path))
    try:
        return _identity_from_open_directory(absolute, observed)
    finally:
        os.close(descriptor)


def _peak_rss_bytes() -> int:
    observed = _GETRUSAGE(resource.RUSAGE_SELF).ru_maxrss
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        raise ExperimentContractError("peak RSS observation is invalid")
    if platform.system() == "Darwin":
        value = int(observed)
    elif platform.system() == "Linux":
        value = int(observed) * 1024
    else:
        raise ExperimentContractError("peak RSS unit is unsupported on this platform")
    if value <= 0:
        raise ExperimentContractError("peak RSS observation must be positive")
    return value


def _free_disk_bytes(descriptor: int) -> int:
    observed = _FSTATVFS(descriptor)
    value = int(observed.f_bavail) * int(observed.f_frsize)
    if value < 0:
        raise ExperimentContractError("free disk observation is invalid")
    return value


def _finite_monotonic_seconds() -> float:
    value = float(_PERF_COUNTER())
    if not math.isfinite(value) or value < 0:
        raise ExperimentContractError("monotonic clock observation is invalid")
    return value


@dataclass(frozen=True, slots=True)
class TQCTrainingResourceAuthorityV2:
    """Capability proving the complete, passing training resource prefix."""

    monitor_id: str
    attempt_id: str
    worker_pid: int
    execution_manifest_sha256: str
    claimed_work_directory_identity: str
    vector_sample_count: int
    last_vector_step: int
    last_environment_step: int
    throughput_window_count: int
    disk_check_count: int
    lifecycle_stages: tuple[str, ...]
    peak_rss_bytes: int
    minimum_free_disk_bytes: int
    minimum_throughput_steps_per_second: float
    elapsed_seconds: float
    prefix_event_sha256: str
    all_training_resource_gates_passed: bool
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _TRAINING_AUTHORITY_ISSUER:
            raise ExperimentContractError(
                "training resource authority may only be issued by the resource monitor"
            )
        _require_sha256(self.prefix_event_sha256, field="training resource event SHA-256")
        _require_sha256(self.execution_manifest_sha256, field="execution manifest SHA-256")
        _require_sha256(
            self.claimed_work_directory_identity,
            field="claimed work directory identity",
        )
        if (
            self.monitor_id != RESOURCE_MONITOR_ID
            or self.attempt_id != ATTEMPT_ID
            or type(self.worker_pid) is not int
            or self.worker_pid <= 1
            or self.vector_sample_count != EXPECTED_VECTOR_STEPS
            or self.last_vector_step != EXPECTED_VECTOR_STEPS
            or self.last_environment_step != TOTAL_ENVIRONMENT_STEPS
            or self.throughput_window_count != EXPECTED_THROUGHPUT_WINDOWS
            or self.disk_check_count != EXPECTED_TRAINING_PREFIX_DISK_CHECKS
            or self.lifecycle_stages != TRAINING_PREFIX_STAGES
            or not 0 < self.peak_rss_bytes <= PEAK_RSS_LIMIT_BYTES
            or self.minimum_free_disk_bytes < FREE_DISK_MINIMUM_BYTES
            or self.minimum_throughput_steps_per_second < THROUGHPUT_MINIMUM_STEPS_PER_SECOND
            or not 0 <= self.elapsed_seconds <= WALL_TIME_LIMIT_SECONDS
            or self.all_training_resource_gates_passed is not True
        ):
            raise ExperimentContractError("training resource authority is inconsistent")


@dataclass(frozen=True, slots=True)
class TQCResourceReceiptV2:
    """Complete resource evidence for one process-local attempt."""

    monitor_id: str
    attempt_id: str
    worker_pid: int
    execution_manifest_sha256: str
    claimed_work_directory_identity: str
    vector_sample_count: int
    last_vector_step: int
    last_environment_step: int
    throughput_window_count: int
    disk_check_count: int
    lifecycle_stages: tuple[str, ...]
    post_training_disk_stages: tuple[str, ...]
    peak_rss_bytes: int
    minimum_free_disk_bytes: int
    minimum_throughput_steps_per_second: float
    elapsed_seconds: float
    training_prefix_event_sha256: str
    event_sha256: str
    lifecycle_samples: tuple[tuple[str, float, int], ...]
    throughput_windows: tuple[tuple[int, int, float], ...]
    post_training_disk_observations: tuple[tuple[str, int], ...]
    all_resource_gates_passed: bool
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _FINAL_RECEIPT_ISSUER:
            raise ExperimentContractError(
                "resource receipt may only be issued by the resource monitor"
            )
        _require_sha256(self.event_sha256, field="resource event SHA-256")
        _require_sha256(
            self.training_prefix_event_sha256,
            field="training resource prefix SHA-256",
        )
        _require_sha256(self.execution_manifest_sha256, field="execution manifest SHA-256")
        _require_sha256(
            self.claimed_work_directory_identity,
            field="claimed work directory identity",
        )
        if (
            self.monitor_id != RESOURCE_MONITOR_ID
            or self.attempt_id != ATTEMPT_ID
            or type(self.worker_pid) is not int
            or self.worker_pid <= 1
            or self.vector_sample_count != EXPECTED_VECTOR_STEPS
            or self.last_vector_step != EXPECTED_VECTOR_STEPS
            or self.last_environment_step != TOTAL_ENVIRONMENT_STEPS
            or self.throughput_window_count != EXPECTED_THROUGHPUT_WINDOWS
            or self.disk_check_count != EXPECTED_DISK_CHECKS
            or self.lifecycle_stages != LIFECYCLE_STAGES
            or self.post_training_disk_stages != POST_TRAINING_DISK_STAGES
            or tuple(sample[0] for sample in self.lifecycle_samples) != LIFECYCLE_STAGES
            or len(self.lifecycle_samples) != len(LIFECYCLE_STAGES)
            or len(self.throughput_windows) != EXPECTED_THROUGHPUT_WINDOWS
            or tuple(sample[0] for sample in self.throughput_windows)
            != tuple(range(1, EXPECTED_THROUGHPUT_WINDOWS + 1))
            or tuple(sample[1] for sample in self.throughput_windows)
            != tuple(range(20_000, TOTAL_ENVIRONMENT_STEPS + 1, 10_000))
            or tuple(sample[0] for sample in self.post_training_disk_observations)
            != POST_TRAINING_DISK_STAGES
            or not 0 < self.peak_rss_bytes <= PEAK_RSS_LIMIT_BYTES
            or self.minimum_free_disk_bytes < FREE_DISK_MINIMUM_BYTES
            or self.minimum_throughput_steps_per_second < THROUGHPUT_MINIMUM_STEPS_PER_SECOND
            or not 0 <= self.elapsed_seconds <= WALL_TIME_LIMIT_SECONDS
            or self.all_resource_gates_passed is not True
        ):
            raise ExperimentContractError("resource receipt is inconsistent")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class TQCResourceMonitorV2:
    """Hash every required sample while retaining only bounded summaries."""

    def __init__(self, output_path: Path, *, attempt_id: str = ATTEMPT_ID) -> None:
        self._output_path = Path(output_path)
        if attempt_id != ATTEMPT_ID or type(attempt_id) is not str:
            raise ExperimentContractError("resource attempt id differs")
        self._attempt_id = attempt_id
        self._creator_pid = os.getpid()
        if self._creator_pid <= 1:
            raise ExperimentContractError("resource monitor creator PID is invalid")
        absolute, descriptor, observed = _open_work_directory(self._output_path)
        self._output_path = absolute
        self._work_directory_descriptor = descriptor
        self._work_directory_state = _directory_state(observed)
        self._claimed_work_directory_identity = _identity_from_open_directory(absolute, observed)
        self._execution_manifest_sha256: str | None = None
        self._started = _finite_monotonic_seconds()
        self._last_monotonic = self._started
        self._training_started: float | None = None
        self._training_finished: float | None = None
        self._events = hashlib.sha256()
        self._vector_count = 0
        self._last_environment_step = 0
        self._lifecycle: list[str] = []
        self._post_training_disk: list[str] = []
        self._post_training_disk_observations: list[tuple[str, int]] = []
        self._disk_checks = 0
        self._peak_rss = 0
        self._minimum_free_disk: int | None = None
        self._throughput_windows: list[tuple[int, int, float]] = []
        self._lifecycle_samples: list[tuple[str, float, int]] = []
        self._throughput_anchor_steps: int | None = None
        self._throughput_anchor_seconds: float | None = None
        self._training_authority: TQCTrainingResourceAuthorityV2 | None = None
        self._finalized = False
        self._failed_reason: str | None = None

    @property
    def claimed_work_directory_identity(self) -> str:
        return self._claimed_work_directory_identity

    @property
    def failed_reason(self) -> str | None:
        return self._failed_reason

    def _assert_active(self) -> None:
        if self._failed_reason is not None:
            raise ExperimentContractError(
                f"resource monitor is permanently failed: {self._failed_reason}"
            )
        if self._finalized:
            raise ExperimentContractError("resource monitor is already finalized")
        if os.getpid() != self._creator_pid:
            self._failed_reason = "process_identity_changed"
            raise ExperimentContractError("resource monitor crossed its worker process boundary")
        try:
            held_state = os.fstat(self._work_directory_descriptor)
            absolute, visible_descriptor, visible_state = _open_work_directory(self._output_path)
        except Exception as exc:
            self._poison(exc)
            raise
        try:
            if (
                absolute != self._output_path
                or _directory_state(held_state) != self._work_directory_state
                or _directory_state(visible_state) != self._work_directory_state
            ):
                self._failed_reason = "claimed_work_directory_identity_changed"
                raise ExperimentContractError("resource monitor work directory identity changed")
        finally:
            os.close(visible_descriptor)

    def _poison(self, exc: BaseException) -> None:
        if self._failed_reason is None:
            self._failed_reason = f"{type(exc).__name__}:{exc}"

    def duplicate_work_directory_descriptor(self) -> int:
        """Return one close-on-exec duplicate bound to the admitted directory."""

        self._assert_active()
        duplicate = os.dup(self._work_directory_descriptor)
        os.set_inheritable(duplicate, False)
        if _directory_state(os.fstat(duplicate)) != self._work_directory_state:
            os.close(duplicate)
            raise ExperimentContractError("duplicated work directory identity differs")
        return duplicate

    def bind_execution_manifest(self, manifest: object) -> None:
        """Bind the worker-acknowledged manifest before model construction."""

        self._assert_active()
        try:
            from .tqc_development_manifest_v2 import ValidatedTQCExecutionManifestV2

            if type(manifest) is not ValidatedTQCExecutionManifestV2:
                raise ExperimentContractError(
                    "resource monitor requires an issued v2 execution manifest"
                )
            if (
                tuple(self._lifecycle) != ("preflight",)
                or self._vector_count != 0
                or self._post_training_disk
            ):
                raise ExperimentContractError("resource manifest binding occurred out of order")
            if (
                manifest.attempt_id != self._attempt_id
                or manifest.claimed_work_directory_identity != self._claimed_work_directory_identity
            ):
                raise ExperimentContractError("resource monitor manifest binding differs")
            if self._execution_manifest_sha256 is not None:
                raise ExperimentContractError("resource monitor manifest is already bound")
            self._execution_manifest_sha256 = _require_sha256(
                manifest.sha256,
                field="execution manifest SHA-256",
            )
            self._record(
                "execution_manifest_bound",
                {
                    "attempt_id": self._attempt_id,
                    "claimed_work_directory_identity": self._claimed_work_directory_identity,
                    "execution_manifest_sha256": self._execution_manifest_sha256,
                    "worker_pid": self._creator_pid,
                },
            )
        except BaseException as exc:
            self._poison(exc)
            raise

    def _record(self, kind: str, payload: dict[str, object]) -> None:
        encoded = canonical_json({"kind": kind, **payload})
        self._events.update(len(encoded).to_bytes(8, "big"))
        self._events.update(encoded)

    def _observe(self) -> tuple[float, int]:
        now = _finite_monotonic_seconds()
        if now < self._last_monotonic:
            raise ExperimentContractError("monotonic clock moved backward")
        rss = _peak_rss_bytes()
        training_elapsed = (
            now - self._training_started
            if self._training_started is not None and self._training_finished is None
            else 0.0
        )
        if rss > PEAK_RSS_LIMIT_BYTES:
            raise ExperimentContractError("sampled peak RSS exceeds the v2 failure threshold")
        if training_elapsed > WALL_TIME_LIMIT_SECONDS:
            raise ExperimentContractError("sampled wall time exceeds the worker failure threshold")
        self._last_monotonic = now
        self._peak_rss = max(self._peak_rss, rss)
        return now, rss

    def _disk_gate(self, *, reason: str) -> int:
        free = _free_disk_bytes(self._work_directory_descriptor)
        if free < FREE_DISK_MINIMUM_BYTES:
            raise ExperimentContractError("free disk is below the v2 failure threshold")
        self._disk_checks += 1
        self._minimum_free_disk = (
            free if self._minimum_free_disk is None else min(self._minimum_free_disk, free)
        )
        self._record(
            "disk_gate",
            {"check_index": self._disk_checks, "free_bytes": free, "reason": reason},
        )
        return free

    def sample_lifecycle(self, stage: str) -> None:
        """Sample one exact lifecycle stage in frozen order."""

        self._assert_active()
        try:
            expected_index = len(self._lifecycle)
            if expected_index >= len(LIFECYCLE_STAGES) or stage != LIFECYCLE_STAGES[expected_index]:
                raise ExperimentContractError(
                    "resource lifecycle stage is missing, repeated, or reordered"
                )
            if expected_index > 0 and self._execution_manifest_sha256 is None:
                raise ExperimentContractError(
                    "resource monitor requires the execution manifest before model construction"
                )
            now, rss = self._observe()
            self._lifecycle.append(stage)
            if stage == "pre_learn":
                self._training_started = now
            elif stage == "post_learn":
                if self._training_started is None:
                    raise ExperimentContractError("training wall-time anchor is missing")
                self._training_finished = now
            lifecycle_sample = (stage, now - self._started, rss)
            self._lifecycle_samples.append(lifecycle_sample)
            self._record(
                "lifecycle",
                {
                    "elapsed_seconds": lifecycle_sample[1],
                    "peak_rss_bytes": rss,
                    "stage": stage,
                    "stage_index": expected_index,
                },
            )
            if stage == "preflight":
                self._disk_gate(reason="preflight")
        except BaseException as exc:
            self._poison(exc)
            raise

    def sample_vector_step(self, vector_step: int, environment_steps: int) -> None:
        """Observe one post-step, pre-replay-add callback point."""

        self._assert_active()
        try:
            if tuple(self._lifecycle) != LIFECYCLE_STAGES[:3]:
                raise ExperimentContractError(
                    "vector monitoring requires the pre-learn lifecycle prefix"
                )
            expected_vector_step = self._vector_count + 1
            if (
                type(vector_step) is not int
                or vector_step != expected_vector_step
                or vector_step > EXPECTED_VECTOR_STEPS
                or type(environment_steps) is not int
                or environment_steps != vector_step * ENVIRONMENT_STEPS_PER_VECTOR_STEP
            ):
                raise ExperimentContractError(
                    "resource vector counter differs from the frozen schedule"
                )
            now, rss = self._observe()
            self._vector_count = vector_step
            self._last_environment_step = environment_steps
            self._record(
                "vector_sample",
                {
                    "elapsed_seconds": now - self._started,
                    "environment_steps": environment_steps,
                    "peak_rss_bytes": rss,
                    "vector_step": vector_step,
                },
            )
            if vector_step % 100 == 0:
                self._disk_gate(reason=f"vector_step_{vector_step}")

            if environment_steps == THROUGHPUT_WARMUP_ENVIRONMENT_STEPS:
                self._throughput_anchor_steps = environment_steps
                self._throughput_anchor_seconds = now
            elif (
                environment_steps > THROUGHPUT_WARMUP_ENVIRONMENT_STEPS
                and environment_steps % THROUGHPUT_WINDOW_ENVIRONMENT_STEPS == 0
            ):
                if self._throughput_anchor_steps is None or self._throughput_anchor_seconds is None:
                    raise ExperimentContractError("throughput window anchor is missing")
                elapsed = now - self._throughput_anchor_seconds
                step_delta = environment_steps - self._throughput_anchor_steps
                if elapsed <= 0 or step_delta != THROUGHPUT_WINDOW_ENVIRONMENT_STEPS:
                    raise ExperimentContractError("throughput window timing is invalid")
                rate = step_delta / elapsed
                if not math.isfinite(rate) or rate < THROUGHPUT_MINIMUM_STEPS_PER_SECOND:
                    raise ExperimentContractError(
                        "training throughput is below the v2 failure threshold"
                    )
                window = (len(self._throughput_windows) + 1, environment_steps, rate)
                self._throughput_windows.append(window)
                self._record(
                    "throughput_window",
                    {
                        "end_environment_step": environment_steps,
                        "rate_environment_steps_per_second": rate,
                        "window_index": window[0],
                    },
                )
                self._throughput_anchor_steps = environment_steps
                self._throughput_anchor_seconds = now
        except BaseException as exc:
            self._poison(exc)
            raise

    def sample_post_training_disk_gate(self, stage: str) -> None:
        """Apply one exact post-training free-disk gate."""

        self._assert_active()
        try:
            expected_index = len(self._post_training_disk)
            if (
                expected_index >= len(POST_TRAINING_DISK_STAGES)
                or stage != POST_TRAINING_DISK_STAGES[expected_index]
            ):
                raise ExperimentContractError("post-training disk stage is missing or reordered")
            required_lifecycle_counts = {
                "pre_persistence": 4,
                "pre_evaluation": 10,
                "finalization": 14,
            }
            if len(self._lifecycle) != required_lifecycle_counts[stage]:
                raise ExperimentContractError(
                    "post-training disk stage is not aligned to its lifecycle point"
                )
            free = self._disk_gate(reason=stage)
            self._post_training_disk.append(stage)
            self._post_training_disk_observations.append((stage, free))
        except BaseException as exc:
            self._poison(exc)
            raise

    def validate_training_prefix(self) -> TQCTrainingResourceAuthorityV2:
        """Issue the training-only resource capability exactly once."""

        self._assert_active()
        try:
            if self._training_authority is not None:
                return self._training_authority
            if (
                self._execution_manifest_sha256 is None
                or tuple(self._lifecycle) != TRAINING_PREFIX_STAGES
                or self._vector_count != EXPECTED_VECTOR_STEPS
                or self._last_environment_step != TOTAL_ENVIRONMENT_STEPS
                or len(self._throughput_windows) != EXPECTED_THROUGHPUT_WINDOWS
                or self._disk_checks != EXPECTED_TRAINING_PREFIX_DISK_CHECKS
                or self._minimum_free_disk is None
            ):
                raise ExperimentContractError("training resource schedule is incomplete")
            if self._training_started is None or self._training_finished is None:
                raise ExperimentContractError("training wall-time observations are incomplete")
            elapsed = self._training_finished - self._training_started
            self._training_authority = TQCTrainingResourceAuthorityV2(
                monitor_id=RESOURCE_MONITOR_ID,
                attempt_id=self._attempt_id,
                worker_pid=self._creator_pid,
                execution_manifest_sha256=self._execution_manifest_sha256,
                claimed_work_directory_identity=self._claimed_work_directory_identity,
                vector_sample_count=self._vector_count,
                last_vector_step=self._vector_count,
                last_environment_step=self._last_environment_step,
                throughput_window_count=len(self._throughput_windows),
                disk_check_count=self._disk_checks,
                lifecycle_stages=tuple(self._lifecycle),
                peak_rss_bytes=self._peak_rss,
                minimum_free_disk_bytes=self._minimum_free_disk,
                minimum_throughput_steps_per_second=min(
                    window[2] for window in self._throughput_windows
                ),
                elapsed_seconds=elapsed,
                prefix_event_sha256=self._events.hexdigest(),
                all_training_resource_gates_passed=True,
                _issuer=_TRAINING_AUTHORITY_ISSUER,
            )
            return self._training_authority
        except BaseException as exc:
            self._poison(exc)
            raise

    def finalize(self) -> TQCResourceReceiptV2:
        """Issue complete evidence after every lifecycle and disk sample."""

        self._assert_active()
        try:
            training = self.validate_training_prefix()
            if (
                tuple(self._lifecycle) != LIFECYCLE_STAGES
                or tuple(self._post_training_disk) != POST_TRAINING_DISK_STAGES
                or self._disk_checks != EXPECTED_DISK_CHECKS
                or self._minimum_free_disk is None
                or self._execution_manifest_sha256 is None
                or self._training_started is None
                or self._training_finished is None
            ):
                raise ExperimentContractError("resource monitoring schedule is incomplete")
            receipt = TQCResourceReceiptV2(
                monitor_id=RESOURCE_MONITOR_ID,
                attempt_id=self._attempt_id,
                worker_pid=self._creator_pid,
                execution_manifest_sha256=self._execution_manifest_sha256,
                claimed_work_directory_identity=self._claimed_work_directory_identity,
                vector_sample_count=self._vector_count,
                last_vector_step=self._vector_count,
                last_environment_step=self._last_environment_step,
                throughput_window_count=len(self._throughput_windows),
                disk_check_count=self._disk_checks,
                lifecycle_stages=tuple(self._lifecycle),
                post_training_disk_stages=tuple(self._post_training_disk),
                peak_rss_bytes=self._peak_rss,
                minimum_free_disk_bytes=self._minimum_free_disk,
                minimum_throughput_steps_per_second=min(
                    window[2] for window in self._throughput_windows
                ),
                elapsed_seconds=self._training_finished - self._training_started,
                training_prefix_event_sha256=training.prefix_event_sha256,
                event_sha256=self._events.hexdigest(),
                lifecycle_samples=tuple(self._lifecycle_samples),
                throughput_windows=tuple(self._throughput_windows),
                post_training_disk_observations=tuple(self._post_training_disk_observations),
                all_resource_gates_passed=True,
                _issuer=_FINAL_RECEIPT_ISSUER,
            )
            self._finalized = True
            os.close(self._work_directory_descriptor)
            return receipt
        except BaseException as exc:
            self._poison(exc)
            raise


__all__ = [
    "ATTEMPT_ID",
    "EXPECTED_THROUGHPUT_WINDOWS",
    "FREE_DISK_MINIMUM_BYTES",
    "LIFECYCLE_STAGES",
    "PEAK_RSS_LIMIT_BYTES",
    "POST_TRAINING_DISK_STAGES",
    "RESOURCE_MONITOR_ID",
    "THROUGHPUT_MINIMUM_STEPS_PER_SECOND",
    "TQCResourceMonitorV2",
    "TQCResourceReceiptV2",
    "TQCTrainingResourceAuthorityV2",
    "claimed_work_directory_identity",
]
