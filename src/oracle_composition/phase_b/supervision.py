"""Parent/worker supervision, resource gates, and per-seed receipts."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import platform
import re
import resource
import shutil
import signal
import stat
import subprocess
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from multiprocessing.connection import Connection
from pathlib import Path
from types import MappingProxyType

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes, sha256_file
from oracle_composition.experiments.artifact_io import (
    PublishedArtifact,
    publish_bytes_without_overwrite,
)
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.harness.evidence import (
    current_authority_identities,
    load_e003_execution,
    validate_scientific_receipt,
)

from .contracts import (
    REWARD_COMPOSITOR_SHA256,
    FineTuningRunManifest,
    RewardRegistry,
    load_fine_tuning_run_manifest,
    load_phase_b_oracle,
)
from .persistence import (
    PersistenceResult,
    load_full_checkpoint,
    load_trained_full_authority_actor,
    publish_checkpoint_index,
    publish_final_persistence,
)
from .report_v2 import SeedReportFacts
from .runtime import (
    RealRuntimeConfig,
    fake_environment_factories,
    fake_policy_factory,
    real_environment_factories,
    real_policy_factory,
    worker_step_zero_e1_audit,
)
from .training import (
    COHORT_SEEDS,
    FULL_TRANSITIONS_PER_SEED,
    SMOKE_SEED,
    SMOKE_TRANSITIONS,
    PhaseBTrainingError,
    TrainingFailureStatus,
    TrainingPlan,
    run_ppo_training,
)

SUPERVISOR_ID = "humanoid_phase_b_training_supervisor/v2"
EXECUTION_MANIFEST_SCHEMA_ID = "humanoid_phase_b_execution_manifest/v2"
SEED_SUCCESS_RECEIPT_ID = "humanoid_phase_b_seed_success/v1"
SEED_FAILURE_RECEIPT_ID = "humanoid_phase_b_seed_failure/v1"
JOB_RESULT_SCHEMA_ID = "humanoid_phase_b_job_result/v1"
WORKER_ACK_TIMEOUT_SECONDS = 30.0
POLL_SECONDS = 0.05
TERMINATION_GRACE_SECONDS = 2.0
KILL_GRACE_SECONDS = 2.0
MAILBOX_MESSAGE_ID = re.compile(r"^\d{8}T\d{6}\.\d{6}Z-[0-9a-f]{32}$")
MAILBOX_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
MAILBOX_AGENTS = frozenset(("astra", "fable"))


class SupervisorStatus(StrEnum):
    SUCCEEDED = "succeeded"
    PRECONDITION_FAILURE = "precondition_failure"
    TIMEOUT = "timeout"
    CRASH = "crash"
    NON_FINITE = "non_finite"
    LIKELIHOOD_FAILURE = "likelihood_failure"
    COUNTER_DRIFT = "counter_drift"
    ACTION_BOUND_VIOLATION = "action_bound_violation"
    PHASE_SELECTION_FAILURE = "phase_selection_failure"
    SOURCE_MUTATION = "source_mutation"
    RESOURCE_BREACH = "resource_breach"
    CLEANUP_FAILURE = "cleanup_failure"
    NOT_STARTED = "not_started_due_to_cohort_failure"


_TRAINING_TO_SUPERVISOR = {
    TrainingFailureStatus.NON_FINITE: SupervisorStatus.NON_FINITE,
    TrainingFailureStatus.LIKELIHOOD_FAILURE: SupervisorStatus.LIKELIHOOD_FAILURE,
    TrainingFailureStatus.COUNTER_DRIFT: SupervisorStatus.COUNTER_DRIFT,
    TrainingFailureStatus.ACTION_BOUND_VIOLATION: SupervisorStatus.ACTION_BOUND_VIOLATION,
    TrainingFailureStatus.PHASE_SELECTION_FAILURE: SupervisorStatus.PHASE_SELECTION_FAILURE,
}


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    per_seed_wall_seconds: float = 22.0 * 60.0
    cohort_wall_seconds: float = 95.0 * 60.0
    job_wall_seconds: float = 120.0 * 60.0
    rss_bytes: int = 8 * 1024**3
    free_disk_bytes: int = 20 * 1024**3
    output_bytes: int = 8 * 1024**3
    throughput_floor_steps_s: float = 800.0
    throughput_warmup_transitions: int = 65_536
    throughput_window_transitions: int = 65_536

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(type(value) not in {int, float} or value <= 0 for value in values.values()):
            raise ValueError("resource limits must be positive")
        if self.cohort_wall_seconds > self.job_wall_seconds:
            raise ValueError("cohort wall cannot exceed job wall")

    def to_dict(self) -> dict[str, object]:
        return dict(asdict(self))


@dataclass(frozen=True, slots=True)
class RuntimeSourceSnapshot:
    value: Mapping[str, object]
    sha256: str


@dataclass(frozen=True, slots=True)
class TrainingPreflight:
    repository_root: Path
    experiment: Path
    oracle_path: Path
    reward_path: Path
    template_manifest: FineTuningRunManifest
    template_manifest_sha256: str
    e003_execution_manifest_sha256: str
    prior_scientific_receipt_sha256: str
    source_snapshot: RuntimeSourceSnapshot
    report_inputs: Mapping[str, object]
    runtime_config: RealRuntimeConfig


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    plan: TrainingPlan
    output_directory: str
    execution_manifest_bytes: bytes
    execution_manifest_sha256: str
    e1_receipt_sha256: str
    source_snapshot_sha256: str
    repository_root: str
    runtime_kind: str
    runtime_config: RealRuntimeConfig | None
    failure_mode: str | None
    rss_limit_bytes: int


@dataclass(frozen=True, slots=True)
class SeedOutcome:
    seed: int
    status: SupervisorStatus
    receipt: PublishedArtifact
    receipt_value: Mapping[str, object]
    report_facts: SeedReportFacts | None
    persistence: PersistenceResult | None


@dataclass(frozen=True, slots=True)
class SupervisionResult:
    status: SupervisorStatus
    outcomes: tuple[SeedOutcome, ...]
    job_result: PublishedArtifact
    checkpoint_index: PublishedArtifact | None
    execution_manifest: PublishedArtifact


class _ReportedResourceBreach(RuntimeError):
    pass


def _self_peak_rss_bytes() -> int:
    observed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    value = int(observed) if platform.system() == "Darwin" else int(observed) * 1024
    if value <= 0:
        raise ExperimentContractError("worker peak RSS observation is invalid")
    return value


def _source_files(root: Path) -> tuple[Path, ...]:
    package_sources = tuple(sorted((root / "src/oracle_composition").rglob("*.py")))
    return (root / "pyproject.toml", root / "uv.lock", *package_sources)


def _git_state(root: Path, source_files: Sequence[Path]) -> tuple[str, bool]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    relative_sources = [path.relative_to(root).as_posix() for path in source_files]
    source_files_tracked = (
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", *relative_sources],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        ).returncode
        == 0
    )
    return head, dirty or not source_files_tracked


def inspect_runtime_sources(
    repository_root: Path,
    *,
    allow_dirty: bool = False,
) -> RuntimeSourceSnapshot:
    """Hash every training authority and bind the installed CPU runtime."""

    root = Path(repository_root).resolve(strict=True)
    files = _source_files(root)
    if any(path.is_symlink() or not path.is_file() for path in files):
        raise ExperimentContractError("a Phase B source authority is unavailable")
    head, dirty = _git_state(root, files)
    if dirty and not allow_dirty:
        raise ExperimentContractError("Phase B training requires clean committed sources")
    try:
        import gymnasium
        import mujoco
        import stable_baselines3
        import torch
    except ImportError as exc:  # pragma: no cover - guarded by project extras
        raise ExperimentContractError("Phase B train and gym extras are required") from exc
    value = {
        "authority_identities": current_authority_identities(),
        "device": "cpu",
        "git": {"clean": not dirty, "commit": head},
        "platform": {
            "machine": platform.machine(),
            "python": platform.python_version(),
            "system": platform.system(),
        },
        "source_sha256": {path.relative_to(root).as_posix(): sha256_file(path) for path in files},
        "versions": {
            "gymnasium": gymnasium.__version__,
            "mujoco": mujoco.__version__,
            "numpy": __import__("numpy").__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
        },
    }
    encoded = canonical_json_bytes(value)
    return RuntimeSourceSnapshot(MappingProxyType(value), hashlib.sha256(encoded).hexdigest())


def _binding(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ExperimentContractError(f"bound input is unavailable: {path}")
    return {
        "byte_count": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_training_preflight(
    *,
    repository_root: Path,
    experiment: Path,
    oracle_path: Path,
    reward_path: Path,
    allow_dirty: bool = False,
) -> TrainingPreflight:
    """Validate FT1 plus E003R1 evidence before any policy/environment construction."""

    root = Path(repository_root).resolve(strict=True)
    experiment_path = Path(experiment).resolve(strict=True)
    oracle = Path(oracle_path).resolve(strict=True)
    reward = Path(reward_path).resolve(strict=True)
    sealed = load_e003_execution(experiment_path, root)
    prior_path = experiment_path / "cycles/cycle_2/scientific_receipt_v2.json"
    prior = validate_scientific_receipt(
        prior_path,
        experiment=experiment_path,
        repository_root=root,
        library=sealed.library,
        task=sealed.task,
        expected_cycle=2,
        expected_metric_core_sha256=None,
    )
    template_path = experiment_path / "phase_b/run_manifest_training_admission_v2.json"
    template, template_sha = load_fine_tuning_run_manifest(template_path, repository_root=root)
    loaded_oracle, oracle_sha = load_phase_b_oracle(
        oracle,
        available_behaviors=("expert", "medium", "simple"),
    )
    reward_spec, reward_sha = RewardRegistry().load(reward)
    if (
        oracle_sha != template.value["oracle"]["sha256"]
        or reward_sha != template.value["reward"]["sha256"]
        or reward_spec.to_dict()["compositor_sha256"] != REWARD_COMPOSITOR_SHA256
    ):
        raise ExperimentContractError("CLI oracle or reward differs from the FT1 run manifest")
    source_snapshot = inspect_runtime_sources(root, allow_dirty=allow_dirty)
    starting_binding = template.value["starting_checkpoint"]
    starting_path = root / str(starting_binding["path"])
    starting = json.loads(starting_path.read_bytes())
    actor_binding = starting["strict_actor_export"]
    e1_binding = starting["e1_receipt"]
    input_bindings = {
        name: template.value[name]
        for name in ("evaluator", "library", "reference_corpus", "task", "training_design")
    }
    report_inputs = {
        "e1_receipt_sha256": e1_binding["sha256"],
        "evaluator_sha256": input_bindings["evaluator"]["sha256"],
        "execution_manifest_sha256": sealed.manifest_sha256,
        "library_sha256": input_bindings["library"]["sha256"],
        "oracle_canonical_sha256": loaded_oracle.sha256,
        "oracle_file_sha256": oracle_sha,
        "reference_sha256": input_bindings["reference_corpus"]["sha256"],
        "reward_compositor_sha256": REWARD_COMPOSITOR_SHA256,
        "reward_file_sha256": reward_sha,
        "reward_formula_id": reward_spec.to_dict()["formula_id"],
        "reward_formula_sha256": reward_spec.to_dict()["formula_sha256"],
        "reward_schema_id": reward_spec.to_dict()["reward_schema_id"],
        "starting_expert_identity": starting["source_expert"],
        "task_sha256": input_bindings["task"]["sha256"],
        "training_design_sha256": input_bindings["training_design"]["sha256"],
    }
    runtime_config = RealRuntimeConfig(
        repository_root=root,
        experiment=experiment_path,
        oracle_path=oracle,
        reward_path=reward,
        starting_actor_path=root / actor_binding["path"],
        starting_actor_sha256=actor_binding["sha256"],
        value_seed=int(starting["value_initialization_seed"]),
    )
    return TrainingPreflight(
        repository_root=root,
        experiment=experiment_path,
        oracle_path=oracle,
        reward_path=reward,
        template_manifest=template,
        template_manifest_sha256=template_sha,
        e003_execution_manifest_sha256=sealed.manifest_sha256,
        prior_scientific_receipt_sha256=prior.sha256,
        source_snapshot=source_snapshot,
        report_inputs=MappingProxyType(report_inputs),
        runtime_config=runtime_config,
    )


def validate_reservation(
    reservation: Mapping[str, object] | None,
    *,
    smoke: bool,
    output_directory: Path,
    test_only: bool,
    repository_root: Path | None = None,
    canonical_argv: Sequence[str] | None = None,
    current_commit: str | None = None,
    expected_inputs: Mapping[str, object] | None = None,
    now: datetime | None = None,
    mailbox_root: Path | None = None,
) -> dict[str, object]:
    """Bind a reservation to its authoritative mailbox acceptance and acknowledgment."""

    if test_only:
        if reservation is None:
            return {"accepted": True, "proposal_id": "controlled-fake-runtime"}
        return dict(reservation)
    if type(reservation) is not dict:
        raise ExperimentContractError("training requires an accepted mailbox reservation")
    required = {
        "accepted",
        "acceptance_message",
        "accepted_until_utc",
        "acknowledgment",
        "canonical_argv",
        "commit",
        "conflict_check",
        "hard_wall_seconds",
        "inputs",
        "mode",
        "output",
        "owner",
        "proposal_id",
        "required_authorizer",
        "schema_version",
    }
    if (
        set(reservation) != required
        or reservation["schema_version"] != 2
        or reservation["accepted"] is not True
        or type(reservation["accepted_until_utc"]) is not str
        or MAILBOX_TIMESTAMP.fullmatch(reservation["accepted_until_utc"]) is None
    ):
        raise ExperimentContractError("mailbox reservation is missing or not accepted")
    if Path(str(reservation["output"])).resolve() != Path(output_directory).resolve():
        raise ExperimentContractError("mailbox reservation output differs")
    if reservation["mode"] != ("smoke" if smoke else "cohort"):
        raise ExperimentContractError("mailbox reservation mode differs")
    if reservation["hard_wall_seconds"] != (1_200 if smoke else 7_200):
        raise ExperimentContractError("mailbox reservation hard deadline differs")
    if reservation["conflict_check"] != "no_other_heavy_repository_job":
        raise ExperimentContractError("mailbox reservation conflict declaration differs")
    for name in ("owner", "commit", "proposal_id", "required_authorizer"):
        if type(reservation[name]) is not str or not reservation[name].strip():
            raise ExperimentContractError(f"mailbox reservation {name} is empty")
    inputs = reservation["inputs"]
    if type(inputs) is not dict or not inputs:
        raise ExperimentContractError("mailbox reservation input ledger is empty")
    argv = reservation["canonical_argv"]
    if (
        type(argv) is not list
        or not argv
        or any(type(item) is not str or not item for item in argv)
    ):
        raise ExperimentContractError("mailbox reservation canonical argv is malformed")
    if canonical_argv is not None and argv != list(canonical_argv):
        raise ExperimentContractError("mailbox reservation command differs")
    if current_commit is not None and reservation["commit"] != current_commit:
        raise ExperimentContractError("mailbox reservation commit differs")
    if expected_inputs is not None and inputs != dict(expected_inputs):
        raise ExperimentContractError("mailbox reservation input ledger differs")

    root = (
        Path(repository_root).resolve(strict=True)
        if repository_root is not None
        else Path(__file__).resolve().parents[3]
    )
    if mailbox_root is None:
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        mailbox = Path(common).resolve(strict=True) / "harness-coordination"
    else:
        mailbox = Path(mailbox_root).resolve(strict=True)

    def authority_file(record: object, directory: str) -> tuple[Path, dict[str, object], bytes]:
        if type(record) is not dict or set(record) != {"path", "sha256"}:
            raise ExperimentContractError("mailbox authority binding is malformed")
        raw_path = Path(str(record["path"]))
        path = raw_path if raw_path.is_absolute() else mailbox / raw_path
        path = Path(os.path.abspath(path))
        try:
            relative = path.relative_to(mailbox)
            before = path.lstat()
        except (OSError, ValueError) as exc:
            raise ExperimentContractError("mailbox authority artifact is unavailable") from exc
        if relative.parent.as_posix() != directory or path.suffix != ".json":
            raise ExperimentContractError("mailbox authority path differs")
        if (
            stat.S_ISLNK(before.st_mode)
            or not stat.S_ISREG(before.st_mode)
            or before.st_size > 64 * 1024
        ):
            raise ExperimentContractError("mailbox authority artifact is not bounded regular data")
        encoded = path.read_bytes()
        after = path.lstat()
        if any(
            getattr(before, field) != getattr(after, field)
            for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        ):
            raise ExperimentContractError("mailbox authority artifact changed while read")
        if hashlib.sha256(encoded).hexdigest() != record["sha256"]:
            raise ExperimentContractError("mailbox authority artifact digest differs")
        try:
            value = json.loads(encoded)
        except (UnicodeError, ValueError) as exc:
            raise ExperimentContractError("mailbox authority artifact is malformed") from exc
        if type(value) is not dict:
            raise ExperimentContractError("mailbox authority artifact is malformed")
        return path, value, encoded

    message_path, message, _message_bytes = authority_file(
        reservation["acceptance_message"], "messages"
    )
    ack_path, acknowledgment, _ack_bytes = authority_file(reservation["acknowledgment"], "acks")
    message_id = message_path.stem
    message_fields = {
        "body",
        "from",
        "id",
        "kind",
        "reply_to",
        "schema_version",
        "sent_at",
        "subject",
        "to",
    }
    if (
        set(message) != message_fields
        or message.get("schema_version") != 1
        or MAILBOX_MESSAGE_ID.fullmatch(message_id) is None
        or MAILBOX_MESSAGE_ID.fullmatch(str(reservation["proposal_id"])) is None
        or message.get("id") != message_id
        or message.get("kind") != "acceptance"
        or message.get("reply_to") != reservation["proposal_id"]
        or message.get("from") != reservation["required_authorizer"]
        or message.get("from") not in MAILBOX_AGENTS
        or message.get("to") not in MAILBOX_AGENTS
        or message.get("from") == message.get("to")
        or type(message.get("subject")) is not str
        or not message["subject"]
        or type(message.get("body")) is not str
        or len(message["body"].encode("utf-8")) > 32 * 1024
        or type(message.get("sent_at")) is not str
        or MAILBOX_TIMESTAMP.fullmatch(message["sent_at"]) is None
    ):
        raise ExperimentContractError("reservation acceptance authorizer or message differs")
    authoritative = {
        field: reservation[field]
        for field in (
            "accepted_until_utc",
            "canonical_argv",
            "commit",
            "conflict_check",
            "hard_wall_seconds",
            "inputs",
            "mode",
            "output",
            "owner",
            "proposal_id",
            "required_authorizer",
        )
    }
    try:
        body_value = json.loads(message["body"])
    except (TypeError, ValueError) as exc:
        raise ExperimentContractError("reservation acceptance body is not JSON") from exc
    if (
        type(body_value) is not dict
        or body_value != authoritative
        or canonical_json_bytes(body_value).decode("utf-8") != message["body"]
    ):
        raise ExperimentContractError("reservation and authoritative acceptance differ")
    if (
        set(acknowledgment)
        != {"acknowledged_at", "meaning", "message_id", "recipient", "schema_version"}
        or acknowledgment.get("schema_version") != 1
        or ack_path.name != f"{message_id}.json"
        or acknowledgment.get("message_id") != message_id
        or acknowledgment.get("recipient") != message.get("to")
        or acknowledgment.get("meaning") != "read_not_agreement"
        or type(acknowledgment.get("acknowledged_at")) is not str
        or MAILBOX_TIMESTAMP.fullmatch(acknowledgment["acknowledged_at"]) is None
    ):
        raise ExperimentContractError("reservation acknowledgment differs")
    try:
        expiry = datetime.fromisoformat(
            str(reservation["accepted_until_utc"]).replace("Z", "+00:00")
        )
        sent_at = datetime.fromisoformat(message["sent_at"].replace("Z", "+00:00"))
        acknowledged_at = datetime.fromisoformat(
            acknowledgment["acknowledged_at"].replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ExperimentContractError("reservation deadline is malformed") from exc
    current = now or datetime.now(UTC)
    if (
        expiry.tzinfo is None
        or current.tzinfo is None
        or not sent_at <= acknowledged_at <= current < expiry
        or expiry <= sent_at
    ):
        raise ExperimentContractError("mailbox reservation is stale")
    return dict(reservation)


def limits_bound_by_reservation(
    limits: ResourceLimits,
    reservation: Mapping[str, object],
) -> ResourceLimits:
    """Derive all wall limits from the authoritative reservation hard deadline."""

    hard = reservation.get("hard_wall_seconds")
    if type(hard) not in {int, float} or hard <= 0:
        raise ExperimentContractError("reservation hard deadline is invalid")
    deadline = float(hard)
    return replace(
        limits,
        per_seed_wall_seconds=min(limits.per_seed_wall_seconds, deadline),
        cohort_wall_seconds=min(limits.cohort_wall_seconds, deadline),
        job_wall_seconds=min(limits.job_wall_seconds, deadline),
    )


def _assert_no_conflicting_training_process() -> None:
    result = subprocess.run(
        ["ps", "-axo", "pid=,command="],
        check=True,
        capture_output=True,
        text=True,
    )
    own_pid = os.getpid()
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if len(fields) != 2 or not fields[0].isdigit() or int(fields[0]) == own_pid:
            continue
        command = fields[1]
        if "oracle_composition.harness.cycle_cli" in command and " train " in f" {command} ":
            raise ExperimentContractError("another heavy Phase B training process is active")


def _fresh_output(path: Path) -> Path:
    absolute = Path(os.path.abspath(path))
    if absolute.exists() or absolute.is_symlink():
        raise ExperimentContractError("training output must be fresh and no-overwrite")
    parent = absolute.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ExperimentContractError("training output parent must be a real directory")
    absolute.mkdir(mode=0o700)
    return absolute


def _execution_manifest_value(
    *,
    preflight: TrainingPreflight,
    seeds: Sequence[int],
    transitions: int,
    smoke: bool,
    reservation: Mapping[str, object],
    limits: ResourceLimits,
    test_only: bool,
) -> dict[str, object]:
    return {
        "checkpoint_selection": "final_transition_only",
        "e003_execution_manifest_sha256": preflight.e003_execution_manifest_sha256,
        "evidence_class": (
            "interface_check" if smoke or test_only else "exploratory_fine_tuning_cycle"
        ),
        "execution_manifest_schema_id": EXECUTION_MANIFEST_SCHEMA_ID,
        "ft1_run_manifest_sha256": preflight.template_manifest_sha256,
        "inputs": dict(preflight.report_inputs),
        "prior_scientific_receipt_sha256": preflight.prior_scientific_receipt_sha256,
        "reservation": dict(reservation),
        "reservation_sha256": hashlib.sha256(canonical_json_bytes(dict(reservation))).hexdigest(),
        "resource_limits": limits.to_dict(),
        "runtime_source_snapshot": dict(preflight.source_snapshot.value),
        "runtime_source_snapshot_sha256": preflight.source_snapshot.sha256,
        "schema_version": 2,
        "seeds": list(seeds),
        "smoke": smoke,
        "test_only": test_only,
        "transitions_per_seed": transitions,
    }


def _send(connection: Connection, message_type: str, payload: Mapping[str, object]) -> None:
    connection.send({"message_type": message_type, "payload": dict(payload)})


def _artifact_record(artifact: PublishedArtifact) -> dict[str, object]:
    return {
        "byte_count": artifact.byte_count,
        "filename": artifact.path.name,
        "sha256": artifact.sha256,
    }


def _worker_execute(connection: Connection, request: WorkerRequest) -> None:
    plan = request.plan
    if request.failure_mode == "crash":
        os._exit(17)
    if request.failure_mode == "hang":
        while True:
            time.sleep(0.25)
    if request.failure_mode == "likelihood_failure":
        raise PhaseBTrainingError(
            TrainingFailureStatus.LIKELIHOOD_FAILURE,
            "controlled likelihood failure",
        )
    if request.failure_mode == "action_bound_violation":
        raise PhaseBTrainingError(
            TrainingFailureStatus.ACTION_BOUND_VIOLATION,
            "controlled action-bound violation",
        )
    if request.failure_mode == "resource_breach":
        _send(connection, "resource_breach", {"reason": "controlled resource breach"})
        return
    if request.failure_mode == "source_mutation":
        _send(connection, "source_mutation", {"reason": "controlled source mutation"})
        return
    if request.runtime_kind == "fake":
        policy_factory = fake_policy_factory
        step_zero_audit = None
        environment_factories = fake_environment_factories(
            plan=plan,
            failure_mode=request.failure_mode,
        )
    elif request.runtime_kind == "real" and request.runtime_config is not None:
        policy_factory = real_policy_factory(request.runtime_config)

        def step_zero_audit(policy: object) -> Mapping[str, object]:
            return worker_step_zero_e1_audit(
                policy,
                repository_root=Path(request.repository_root),
                receipt_path=(
                    request.runtime_config.experiment
                    / "phase_b/receipts/e1_full_authority_warm_start_v1.json"
                ),
                expected_receipt_sha256=request.e1_receipt_sha256,
            )

        environment_factories = real_environment_factories(
            plan=plan,
            config=request.runtime_config,
        )
    else:
        raise ExperimentContractError("worker runtime kind is invalid")

    def progress(value: Mapping[str, object]) -> None:
        rss = _self_peak_rss_bytes()
        _send(connection, "progress", {**dict(value), "worker_peak_rss_bytes": rss})
        if rss > request.rss_limit_bytes:
            _send(connection, "resource_breach", {"reason": "worker RSS limit exceeded"})
            raise _ReportedResourceBreach("worker RSS limit exceeded")

    result = run_ppo_training(
        plan=plan,
        policy_factory=policy_factory,
        environment_factories=environment_factories,
        progress_callback=progress,
        step_zero_audit=step_zero_audit,
    )
    if request.failure_mode == "source_mutation_after_training":
        _send(connection, "source_mutation", {"reason": "controlled source mutation"})
        return
    output = Path(request.output_directory)
    training_facts = publish_bytes_without_overwrite(
        output / "training_facts_v1.json",
        canonical_json_bytes(dict(result.scientific_facts)),
    )
    rsi_ledger = publish_bytes_without_overwrite(
        output / "rsi_ledger_v1.json",
        canonical_json_bytes([dict(value) for value in result.rsi_ledger]),
    )
    persistence = publish_final_persistence(output_directory=output, result=result, plan=plan)
    after = inspect_runtime_sources(
        Path(request.repository_root),
        allow_dirty=plan.test_only,
    )
    if after.sha256 != request.source_snapshot_sha256:
        _send(connection, "source_mutation", {"reason": "source snapshot changed"})
        return
    payload = {
        "persistence": {
            "checkpoint": _artifact_record(persistence.checkpoint),
            "receipt": _artifact_record(persistence.receipt),
            "receipt_value": dict(persistence.receipt_value),
            "strict_export": _artifact_record(persistence.strict_export),
        },
        "rsi_ledger": _artifact_record(rsi_ledger),
        "step_zero_comparator": dict(result.scientific_facts["step_zero_comparator"]),
        "training_facts": _artifact_record(training_facts),
        "training_facts_value": dict(result.scientific_facts),
        "worker_peak_rss_bytes": _self_peak_rss_bytes(),
    }
    _send(connection, "completed", payload)


def _worker_session_entry(connection: Connection, request: WorkerRequest) -> None:
    os.setsid()
    try:
        snapshot = inspect_runtime_sources(
            Path(request.repository_root),
            allow_dirty=request.plan.test_only,
        )
        _send(
            connection,
            "worker_started",
            {
                "pgid": os.getpgrp(),
                "pid": os.getpid(),
                "sid": os.getsid(0),
                "source_snapshot_sha256": snapshot.sha256,
            },
        )
        if not connection.poll(WORKER_ACK_TIMEOUT_SECONDS):
            raise ExperimentContractError("worker did not receive execution admission")
        message = connection.recv()
        if (
            type(message) is not dict
            or message.get("message_type") != "admit_execution"
            or message.get("manifest_sha256") != request.execution_manifest_sha256
            or message.get("manifest_bytes") != request.execution_manifest_bytes
            or hashlib.sha256(request.execution_manifest_bytes).hexdigest()
            != request.execution_manifest_sha256
        ):
            raise ExperimentContractError("worker execution admission differs")
        # This acknowledgement is the last operation before model/environment construction.
        _send(
            connection,
            "execution_acknowledged",
            {
                "manifest_sha256": request.execution_manifest_sha256,
                "model_or_environment_constructed": False,
                "source_snapshot_sha256": snapshot.sha256,
            },
        )
        if not connection.poll(WORKER_ACK_TIMEOUT_SECONDS):
            raise ExperimentContractError("worker did not receive post-ACK construction admission")
        construction = connection.recv()
        if (
            type(construction) is not dict
            or construction.get("message_type") != "begin_construction"
            or construction.get("manifest_sha256") != request.execution_manifest_sha256
        ):
            raise ExperimentContractError("post-ACK construction admission differs")
        _worker_execute(connection, request)
    except _ReportedResourceBreach:
        pass
    except PhaseBTrainingError as exc:
        _send(
            connection,
            "failed",
            {"reason": str(exc), "status": _TRAINING_TO_SUPERVISOR[exc.status].value},
        )
    except BaseException as exc:
        with suppress(BrokenPipeError, EOFError, OSError):
            _send(
                connection,
                "failed",
                {"reason": f"{type(exc).__name__}: {exc}", "status": "crash"},
            )
        raise
    finally:
        connection.close()


def _spawn_worker(request: WorkerRequest) -> tuple[multiprocessing.Process, Connection]:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    process = context.Process(target=_worker_session_entry, args=(child, request), daemon=False)
    process.start()
    child.close()
    return process, parent


def _directory_bytes(path: Path) -> int:
    total = 0
    for root, directories, files in os.walk(path, followlinks=False):
        directories[:] = [name for name in directories if not (Path(root) / name).is_symlink()]
        for name in files:
            candidate = Path(root) / name
            if candidate.is_symlink():
                raise ExperimentContractError("run output contains a symbolic link")
            total += candidate.stat().st_size
    return total


def cleanup_worker_process(process: multiprocessing.Process, *, group_validated: bool) -> None:
    """Terminate then kill the exact process group, with bounded verification."""

    pid = process.pid
    if pid is None:
        return

    def group_exists() -> bool:
        if not group_validated:
            return process.is_alive()
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError as exc:
            raise ExperimentContractError("cannot inspect worker process group") from exc
        return True

    if process.is_alive() or group_exists():
        if group_validated:
            with suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGTERM)
        else:
            process.terminate()
        process.join(timeout=TERMINATION_GRACE_SECONDS)
    if process.is_alive() or group_exists():
        if group_validated:
            with suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGKILL)
        else:
            process.kill()
        process.join(timeout=KILL_GRACE_SECONDS)
    if process.is_alive() or group_exists():
        raise ExperimentContractError("worker process group survived cleanup")
    process.join(timeout=0.0)


def _failure_receipt(
    *,
    directory: Path,
    plan: TrainingPlan,
    status: SupervisorStatus,
    reason: str,
    last_stage: str,
    cleanup_succeeded: bool,
) -> tuple[PublishedArtifact, dict[str, object]]:
    value = {
        "cleanup_succeeded": cleanup_succeeded,
        "evidence_class": plan.evidence_class,
        "execution_manifest_sha256": plan.manifest_sha256,
        "failure_receipt_id": SEED_FAILURE_RECEIPT_ID,
        "last_acknowledged_stage": last_stage,
        "outcome": "failure",
        "planned_transitions": plan.transitions,
        "ppo_seed": plan.seed,
        "reason": reason[:1000],
        "schema_version": 1,
        "smoke": plan.smoke,
        "status": status.value,
        "success_receipt_present": False,
    }
    artifact = publish_bytes_without_overwrite(
        directory / "failure_receipt_v1.json",
        canonical_json_bytes(value),
    )
    return artifact, value


def _success_receipt(
    *,
    directory: Path,
    plan: TrainingPlan,
    payload: Mapping[str, object],
) -> tuple[PublishedArtifact, dict[str, object]]:
    value = {
        "artifacts": {
            "persistence": payload["persistence"]["receipt"],
            "rsi_ledger": payload["rsi_ledger"],
            "training_facts": payload["training_facts"],
        },
        "evidence_class": plan.evidence_class,
        "execution_manifest_sha256": plan.manifest_sha256,
        "failure_receipt_present": False,
        "outcome": "success",
        "planned_transitions": plan.transitions,
        "ppo_seed": plan.seed,
        "promotable": plan.promotable,
        "schema_version": 1,
        "smoke": plan.smoke,
        "status": SupervisorStatus.SUCCEEDED.value,
        "success_receipt_id": SEED_SUCCESS_RECEIPT_ID,
        "test_only": plan.test_only,
    }
    artifact = publish_bytes_without_overwrite(
        directory / "success_receipt_v1.json",
        canonical_json_bytes(value),
    )
    return artifact, value


def _persistence_from_payload(
    directory: Path,
    payload: Mapping[str, object],
    *,
    plan: TrainingPlan,
) -> PersistenceResult:
    raw = payload["persistence"]
    if type(raw) is not dict or type(raw.get("receipt_value")) is not dict:
        raise ExperimentContractError("worker persistence payload is malformed")

    def artifact(record: Mapping[str, object]) -> PublishedArtifact:
        path = directory / str(record["filename"])
        if _binding(path) != {
            "byte_count": record["byte_count"],
            "sha256": record["sha256"],
        }:
            raise ExperimentContractError("worker artifact binding differs")
        return PublishedArtifact(path, str(record["sha256"]), int(record["byte_count"]))

    checkpoint = artifact(raw["checkpoint"])
    strict_export = artifact(raw["strict_export"])
    receipt = artifact(raw["receipt"])
    receipt_value = dict(raw["receipt_value"])
    if receipt.path.read_bytes() != canonical_json_bytes(receipt_value):
        raise ExperimentContractError("worker persistence receipt content differs")
    training_facts = artifact(payload["training_facts"])
    rsi_ledger = artifact(payload["rsi_ledger"])
    training_value = payload.get("training_facts_value")
    if type(training_value) is not dict or training_facts.path.read_bytes() != canonical_json_bytes(
        training_value
    ):
        raise ExperimentContractError("worker training-facts content differs")
    if (
        payload.get("step_zero_comparator") != training_value.get("step_zero_comparator")
        or rsi_ledger.sha256 != training_value.get("rsi_ledger_sha256")
        or receipt_value.get("training_facts_sha256") != training_facts.sha256
        or receipt_value.get("execution_manifest_sha256") != plan.manifest_sha256
        or receipt_value.get("ppo_seed") != plan.seed
        or receipt_value.get("transitions") != plan.transitions
        or receipt_value.get("checkpoint") != raw["checkpoint"]
        or receipt_value.get("strict_export") != raw["strict_export"]
    ):
        raise ExperimentContractError("worker persistence evidence chain differs")
    loaded_checkpoint = load_full_checkpoint(
        checkpoint.path,
        expected_sha256=checkpoint.sha256,
    )
    loaded_export = load_trained_full_authority_actor(
        strict_export.path,
        expected_sha256=strict_export.sha256,
    )
    checkpoint_parameters = loaded_checkpoint.policy.actor.parameter_arrays()
    export_parameters = loaded_export.actor.parameter_arrays()
    if (
        loaded_checkpoint.metadata.get("training_facts_sha256") != training_facts.sha256
        or loaded_checkpoint.metadata.get("execution_manifest_sha256") != plan.manifest_sha256
        or loaded_checkpoint.metadata.get("ppo_seed") != plan.seed
        or any(
            checkpoint_parameters[name].tobytes(order="C")
            != export_parameters[name].tobytes(order="C")
            for name in checkpoint_parameters
        )
    ):
        raise ExperimentContractError("checkpoint, export, and training facts differ")
    return PersistenceResult(
        checkpoint=checkpoint,
        strict_export=strict_export,
        receipt=receipt,
        receipt_value=MappingProxyType(receipt_value),
    )


def _supervise_seed(
    *,
    plan: TrainingPlan,
    seed_directory: Path,
    execution_manifest_bytes: bytes,
    execution_manifest_sha256: str,
    preflight: TrainingPreflight,
    limits: ResourceLimits,
    cohort_job_deadline: float,
    runtime_kind: str,
    failure_mode: str | None,
) -> SeedOutcome:
    seed_directory.mkdir(mode=0o700)
    request = WorkerRequest(
        plan=plan,
        output_directory=str(seed_directory),
        execution_manifest_bytes=execution_manifest_bytes,
        execution_manifest_sha256=execution_manifest_sha256,
        e1_receipt_sha256=str(preflight.report_inputs["e1_receipt_sha256"]),
        source_snapshot_sha256=preflight.source_snapshot.sha256,
        repository_root=str(preflight.repository_root),
        runtime_kind=runtime_kind,
        runtime_config=preflight.runtime_config if runtime_kind == "real" else None,
        failure_mode=failure_mode,
        rss_limit_bytes=limits.rss_bytes,
    )
    process, connection = _spawn_worker(request)
    started = time.perf_counter()
    last_stage = "spawned"
    group_validated = False
    terminal_payload: Mapping[str, object] | None = None
    terminal_status: SupervisorStatus | None = None
    reason = ""
    progress_anchor_steps = 0
    progress_anchor_time = started
    last_progress_steps = 0
    peak_rss = 0
    minimum_free_disk_bytes = shutil.disk_usage(seed_directory).free
    maximum_output_bytes = _directory_bytes(seed_directory)
    throughput_windows: list[dict[str, object]] = []
    try:
        while True:
            now = time.perf_counter()
            elapsed = now - started
            if elapsed > limits.per_seed_wall_seconds:
                terminal_status = SupervisorStatus.TIMEOUT
                reason = "per-seed wall limit exceeded"
                break
            if now > cohort_job_deadline:
                terminal_status = SupervisorStatus.TIMEOUT
                reason = "cohort or job wall limit exceeded"
                break
            if last_stage in {"spawned", "worker_started"} and elapsed > min(
                WORKER_ACK_TIMEOUT_SECONDS, limits.per_seed_wall_seconds
            ):
                terminal_status = SupervisorStatus.TIMEOUT
                reason = "worker acknowledgement wall limit exceeded"
                break
            free_disk_bytes = shutil.disk_usage(seed_directory).free
            output_bytes = _directory_bytes(seed_directory)
            minimum_free_disk_bytes = min(minimum_free_disk_bytes, free_disk_bytes)
            maximum_output_bytes = max(maximum_output_bytes, output_bytes)
            if free_disk_bytes < limits.free_disk_bytes:
                terminal_status = SupervisorStatus.RESOURCE_BREACH
                reason = "free disk fell below the hard minimum"
                break
            if output_bytes > limits.output_bytes:
                terminal_status = SupervisorStatus.RESOURCE_BREACH
                reason = "worker output exceeded the hard cap"
                break
            if connection.poll(POLL_SECONDS):
                message = connection.recv()
                if type(message) is not dict or type(message.get("payload")) is not dict:
                    terminal_status = SupervisorStatus.CRASH
                    reason = "worker sent a malformed message"
                    break
                message_type = message.get("message_type")
                payload = message["payload"]
                if message_type == "worker_started":
                    if (
                        payload.get("pid") != process.pid
                        or payload.get("pgid") != process.pid
                        or payload.get("sid") != process.pid
                        or payload.get("source_snapshot_sha256") != preflight.source_snapshot.sha256
                    ):
                        terminal_status = SupervisorStatus.PRECONDITION_FAILURE
                        reason = "worker start or source identity differs"
                        break
                    group_validated = True
                    connection.send(
                        {
                            "message_type": "admit_execution",
                            "manifest_bytes": execution_manifest_bytes,
                            "manifest_sha256": execution_manifest_sha256,
                        }
                    )
                    last_stage = "worker_started"
                elif message_type == "execution_acknowledged":
                    if (
                        last_stage != "worker_started"
                        or payload.get("manifest_sha256") != execution_manifest_sha256
                        or payload.get("model_or_environment_constructed") is not False
                    ):
                        terminal_status = SupervisorStatus.PRECONDITION_FAILURE
                        reason = "worker acknowledgement differs"
                        break
                    connection.send(
                        {
                            "message_type": "begin_construction",
                            "manifest_sha256": execution_manifest_sha256,
                        }
                    )
                    last_stage = "execution_acknowledged"
                elif message_type == "progress":
                    if last_stage not in {"execution_acknowledged", "training"}:
                        terminal_status = SupervisorStatus.COUNTER_DRIFT
                        reason = "worker progress preceded acknowledgement"
                        break
                    steps = payload.get("observed_transitions")
                    rss = payload.get("worker_peak_rss_bytes")
                    if type(steps) is not int or steps <= last_progress_steps:
                        terminal_status = SupervisorStatus.COUNTER_DRIFT
                        reason = "worker progress counter drifted"
                        break
                    last_progress_steps = steps
                    if type(rss) is not int or rss <= 0:
                        terminal_status = SupervisorStatus.RESOURCE_BREACH
                        reason = "worker RSS sample is missing or invalid"
                        break
                    peak_rss = max(peak_rss, rss)
                    if rss > limits.rss_bytes:
                        terminal_status = SupervisorStatus.RESOURCE_BREACH
                        reason = "worker RSS limit exceeded"
                        break
                    now = time.perf_counter()
                    if (
                        steps > limits.throughput_warmup_transitions
                        and steps % limits.throughput_window_transitions == 0
                    ):
                        throughput = (steps - progress_anchor_steps) / (now - progress_anchor_time)
                        throughput_windows.append(
                            {
                                "elapsed_seconds": now - progress_anchor_time,
                                "end_transition": steps,
                                "start_transition": progress_anchor_steps,
                                "steps_per_second": throughput,
                            }
                        )
                        if throughput < limits.throughput_floor_steps_s:
                            terminal_status = SupervisorStatus.RESOURCE_BREACH
                            reason = "post-warm-up throughput floor was breached"
                            break
                        progress_anchor_steps = steps
                        progress_anchor_time = now
                    elif steps == limits.throughput_warmup_transitions:
                        progress_anchor_steps = steps
                        progress_anchor_time = now
                    last_stage = "training"
                elif message_type == "completed":
                    if last_stage not in {"execution_acknowledged", "training"}:
                        terminal_status = SupervisorStatus.COUNTER_DRIFT
                        reason = "worker completed before acknowledgement"
                    else:
                        rss = payload.get("worker_peak_rss_bytes")
                        if type(rss) is not int or rss <= 0 or rss > limits.rss_bytes:
                            terminal_status = SupervisorStatus.RESOURCE_BREACH
                            reason = "final worker RSS sample breached its contract"
                            break
                        peak_rss = max(peak_rss, rss)
                        terminal_payload = payload
                        last_stage = "completed"
                    break
                elif message_type in {"resource_breach", "source_mutation"}:
                    terminal_status = (
                        SupervisorStatus.RESOURCE_BREACH
                        if message_type == "resource_breach"
                        else SupervisorStatus.SOURCE_MUTATION
                    )
                    reason = str(payload.get("reason", message_type))
                    break
                elif message_type == "failed":
                    try:
                        terminal_status = SupervisorStatus(str(payload.get("status")))
                    except ValueError:
                        terminal_status = SupervisorStatus.CRASH
                    reason = str(payload.get("reason", "worker failed"))
                    break
                else:
                    terminal_status = SupervisorStatus.CRASH
                    reason = "worker message type is unknown"
                    break
            elif not process.is_alive():
                terminal_status = SupervisorStatus.CRASH
                reason = f"worker exited with code {process.exitcode} before completion"
                break
    except (EOFError, OSError, ExperimentContractError) as exc:
        terminal_status = SupervisorStatus.CRASH
        reason = f"supervisor channel/resource failure: {exc}"
    cleanup_succeeded = True
    try:
        cleanup_worker_process(process, group_validated=group_validated)
        if failure_mode == "cleanup_failure":
            raise ExperimentContractError("controlled cleanup failure")
    except BaseException as exc:
        cleanup_succeeded = False
        terminal_payload = None
        terminal_status = SupervisorStatus.CLEANUP_FAILURE
        reason = f"worker cleanup failed: {exc}"
    finally:
        connection.close()
    final_free_disk_bytes = shutil.disk_usage(seed_directory).free
    final_output_bytes = _directory_bytes(seed_directory)
    minimum_free_disk_bytes = min(minimum_free_disk_bytes, final_free_disk_bytes)
    maximum_output_bytes = max(maximum_output_bytes, final_output_bytes)
    if terminal_payload is not None and (
        final_output_bytes > limits.output_bytes or final_free_disk_bytes < limits.free_disk_bytes
    ):
        terminal_payload = None
        terminal_status = SupervisorStatus.RESOURCE_BREACH
        reason = "post-worker disk or output resource gate was breached"
    telemetry_value = {
        "cleanup_succeeded": cleanup_succeeded,
        "last_acknowledged_stage": last_stage,
        "maximum_output_bytes": maximum_output_bytes,
        "minimum_free_disk_bytes": minimum_free_disk_bytes,
        "peak_rss_bytes": peak_rss,
        "resource_limits": limits.to_dict(),
        "seed_wall_seconds": time.perf_counter() - started,
        "status": (
            SupervisorStatus.SUCCEEDED.value
            if terminal_payload is not None
            else (terminal_status or SupervisorStatus.CRASH).value
        ),
        "throughput_windows": throughput_windows,
    }
    telemetry_bytes = canonical_json_bytes(telemetry_value)
    if (
        terminal_payload is not None
        and final_output_bytes + len(telemetry_bytes) > limits.output_bytes
    ):
        terminal_payload = None
        terminal_status = SupervisorStatus.RESOURCE_BREACH
        reason = "telemetry publication would exceed the output resource cap"
        telemetry_value["status"] = SupervisorStatus.RESOURCE_BREACH.value
        telemetry_bytes = canonical_json_bytes(telemetry_value)
    publish_bytes_without_overwrite(
        seed_directory / "telemetry_v1.json",
        telemetry_bytes,
    )
    if terminal_payload is None:
        status = terminal_status or SupervisorStatus.CRASH
        receipt, value = _failure_receipt(
            directory=seed_directory,
            plan=plan,
            status=status,
            reason=reason or "worker produced no completion",
            last_stage=last_stage,
            cleanup_succeeded=cleanup_succeeded,
        )
        return SeedOutcome(plan.seed, status, receipt, MappingProxyType(value), None, None)
    try:
        persistence = _persistence_from_payload(
            seed_directory,
            terminal_payload,
            plan=plan,
        )
    except (ExperimentContractError, KeyError, TypeError, ValueError) as exc:
        receipt, value = _failure_receipt(
            directory=seed_directory,
            plan=plan,
            status=SupervisorStatus.CRASH,
            reason=f"worker persistence validation failed: {exc}",
            last_stage=last_stage,
            cleanup_succeeded=cleanup_succeeded,
        )
        return SeedOutcome(
            plan.seed,
            SupervisorStatus.CRASH,
            receipt,
            MappingProxyType(value),
            None,
            None,
        )
    success, success_value = _success_receipt(
        directory=seed_directory,
        plan=plan,
        payload=terminal_payload,
    )
    facts = SeedReportFacts(
        ppo_seed=plan.seed,
        checkpoint_sha256=persistence.checkpoint.sha256,
        strict_export_sha256=persistence.strict_export.sha256,
        training=MappingProxyType(dict(terminal_payload["training_facts_value"])),
        step_zero_comparator=MappingProxyType(dict(terminal_payload["step_zero_comparator"])),
        execution_manifest_bytes=execution_manifest_bytes,
        rsi_ledger_bytes=(seed_directory / "rsi_ledger_v1.json").read_bytes(),
    )
    return SeedOutcome(
        plan.seed,
        SupervisorStatus.SUCCEEDED,
        success,
        MappingProxyType(success_value),
        facts,
        persistence,
    )


def _not_started_outcome(
    *,
    seed: int,
    directory: Path,
    plan: TrainingPlan,
    failed_seed: int,
) -> SeedOutcome:
    directory.mkdir(mode=0o700)
    receipt, value = _failure_receipt(
        directory=directory,
        plan=plan,
        status=SupervisorStatus.NOT_STARTED,
        reason=f"cohort stopped after seed {failed_seed}",
        last_stage="not_started",
        cleanup_succeeded=True,
    )
    return SeedOutcome(
        seed, SupervisorStatus.NOT_STARTED, receipt, MappingProxyType(value), None, None
    )


def supervise_training_job(
    *,
    preflight: TrainingPreflight,
    output_directory: Path,
    seeds: Sequence[int],
    transitions: int,
    smoke: bool,
    reservation: Mapping[str, object] | None,
    limits: ResourceLimits | None = None,
    runtime_kind: str = "real",
    test_only: bool = False,
    test_steps_per_environment: int = 2_048,
    test_batch_size: int = 512,
    test_n_epochs: int = 10,
    failure_mode: str | None = None,
    canonical_argv: Sequence[str] | None = None,
) -> SupervisionResult:
    """Supervise serial seeds and account for every declared unit on failure."""

    if type(preflight) is not TrainingPreflight:
        raise ExperimentContractError("supervision requires validated preflight authority")
    selected_limits = limits or ResourceLimits()
    if type(selected_limits) is not ResourceLimits:
        raise ExperimentContractError("supervision resource-limit authority differs")
    declared = tuple(seeds)
    if smoke:
        if not test_only and (declared != (SMOKE_SEED,) or transitions != SMOKE_TRANSITIONS):
            raise ExperimentContractError("smoke seed or transition count differs")
    elif not test_only and (declared != COHORT_SEEDS or transitions != FULL_TRANSITIONS_PER_SEED):
        raise ExperimentContractError("cohort seed set or transition budget differs")
    if len(set(declared)) != len(declared) or not declared:
        raise ExperimentContractError("training seed list is empty or duplicated")
    if runtime_kind not in {"fake", "real"} or (runtime_kind == "fake" and not test_only):
        raise ExperimentContractError("controlled fake runtime is test-only")
    from .training import PPORecipe

    recipe = (
        PPORecipe(batch_size=test_batch_size, n_epochs=test_n_epochs) if test_only else PPORecipe()
    )
    plans = tuple(
        TrainingPlan(
            seed=seed,
            transitions=transitions,
            manifest_sha256="0" * 64,
            evidence_class=(
                "interface_check" if smoke or test_only else "exploratory_fine_tuning_cycle"
            ),
            promotable=not (smoke or test_only),
            smoke=smoke,
            steps_per_environment=(test_steps_per_environment if test_only else 2_048),
            recipe=recipe,
            test_only=test_only,
        )
        for seed in declared
    )
    output_path = Path(os.path.abspath(output_directory))
    expected_reservation_inputs = {
        **dict(preflight.report_inputs),
        "e003_execution_manifest_sha256": preflight.e003_execution_manifest_sha256,
        "ft1_run_manifest_sha256": preflight.template_manifest_sha256,
        "prior_scientific_receipt_sha256": preflight.prior_scientific_receipt_sha256,
        "runtime_source_snapshot_sha256": preflight.source_snapshot.sha256,
    }
    git_value = preflight.source_snapshot.value["git"]
    if not test_only and canonical_argv is None:
        raise ExperimentContractError("production training requires canonical argv authority")
    accepted = validate_reservation(
        reservation,
        smoke=smoke,
        output_directory=output_path,
        test_only=test_only,
        repository_root=preflight.repository_root,
        canonical_argv=canonical_argv,
        current_commit=str(git_value["commit"]),
        expected_inputs=expected_reservation_inputs,
    )
    if not test_only:
        selected_limits = limits_bound_by_reservation(selected_limits, accepted)
    if shutil.disk_usage(output_path.parent).free < selected_limits.free_disk_bytes:
        raise ExperimentContractError("free disk is below the 20 GiB preflight gate")
    output = _fresh_output(output_path)
    manifest_value = _execution_manifest_value(
        preflight=preflight,
        seeds=declared,
        transitions=transitions,
        smoke=smoke,
        reservation=accepted,
        limits=selected_limits,
        test_only=test_only,
    )
    manifest_bytes = canonical_json_bytes(manifest_value)
    manifest = publish_bytes_without_overwrite(
        output / "execution_manifest_v2.json", manifest_bytes
    )
    started = time.perf_counter()
    cohort_job_deadline = started + min(
        selected_limits.cohort_wall_seconds,
        selected_limits.job_wall_seconds,
    )
    outcomes: list[SeedOutcome] = []
    failed_seed: int | None = None
    for plan_template in plans:
        seed = plan_template.seed
        plan = TrainingPlan(
            seed=seed,
            transitions=transitions,
            manifest_sha256=manifest.sha256,
            evidence_class=plan_template.evidence_class,
            promotable=plan_template.promotable,
            smoke=smoke,
            steps_per_environment=plan_template.steps_per_environment,
            recipe=recipe,
            test_only=test_only,
        )
        if failed_seed is not None:
            outcomes.append(
                _not_started_outcome(
                    seed=seed,
                    directory=output / f"seed_{seed}",
                    plan=plan,
                    failed_seed=failed_seed,
                )
            )
            continue
        if time.perf_counter() > cohort_job_deadline:
            failed_seed = seed
            directory = output / f"seed_{seed}"
            directory.mkdir(mode=0o700)
            receipt, value = _failure_receipt(
                directory=directory,
                plan=plan,
                status=SupervisorStatus.TIMEOUT,
                reason="cohort or job wall limit exceeded before seed start",
                last_stage="not_started",
                cleanup_succeeded=True,
            )
            outcomes.append(
                SeedOutcome(
                    seed, SupervisorStatus.TIMEOUT, receipt, MappingProxyType(value), None, None
                )
            )
            continue
        if not test_only:
            validate_reservation(
                accepted,
                smoke=smoke,
                output_directory=output_path,
                test_only=False,
                repository_root=preflight.repository_root,
                canonical_argv=canonical_argv,
                current_commit=str(git_value["commit"]),
                expected_inputs=expected_reservation_inputs,
            )
            _assert_no_conflicting_training_process()
        outcome = _supervise_seed(
            plan=plan,
            seed_directory=output / f"seed_{seed}",
            execution_manifest_bytes=manifest_bytes,
            execution_manifest_sha256=manifest.sha256,
            preflight=preflight,
            limits=selected_limits,
            cohort_job_deadline=cohort_job_deadline,
            runtime_kind=runtime_kind,
            failure_mode=failure_mode,
        )
        outcomes.append(outcome)
        if outcome.status != SupervisorStatus.SUCCEEDED:
            failed_seed = seed
    overall = (
        SupervisorStatus.SUCCEEDED
        if all(outcome.status == SupervisorStatus.SUCCEEDED for outcome in outcomes)
        else next(
            outcome.status
            for outcome in outcomes
            if outcome.status not in {SupervisorStatus.SUCCEEDED, SupervisorStatus.NOT_STARTED}
        )
    )
    persistence = [outcome.persistence for outcome in outcomes if outcome.persistence is not None]
    job_value = {
        "checkpoint_index": None,
        "execution_manifest_sha256": manifest.sha256,
        "job_result_schema_id": JOB_RESULT_SCHEMA_ID,
        "outcomes": [
            {
                "receipt": _artifact_record(outcome.receipt),
                "seed": outcome.seed,
                "status": outcome.status.value,
            }
            for outcome in outcomes
        ],
        "schema_version": 1,
        "status": overall.value,
    }
    job_result = publish_bytes_without_overwrite(
        output / "job_result_v1.json",
        canonical_json_bytes(job_value),
    )
    checkpoint_index = (
        publish_checkpoint_index(
            output_directory=output,
            entries=persistence,
            success_receipts=[outcome.receipt for outcome in outcomes],
            execution_manifest=manifest,
            job_result=job_result,
        )
        if overall == SupervisorStatus.SUCCEEDED and not smoke and not test_only
        else None
    )
    return SupervisionResult(overall, tuple(outcomes), job_result, checkpoint_index, manifest)


__all__ = [
    "ResourceLimits",
    "RuntimeSourceSnapshot",
    "SeedOutcome",
    "SupervisionResult",
    "SupervisorStatus",
    "TrainingPreflight",
    "cleanup_worker_process",
    "inspect_runtime_sources",
    "limits_bound_by_reservation",
    "supervise_training_job",
    "validate_reservation",
    "validate_training_preflight",
]
