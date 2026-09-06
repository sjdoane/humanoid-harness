"""Parent/worker supervision, resource gates, and per-seed receipts."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import platform
import re
import resource
import shutil
import signal
import stat
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
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
from oracle_composition.experiments.reference_corpus_bundle import load_bundle_manifest
from oracle_composition.harness.evidence import (
    current_authority_identities,
    load_e003_execution,
    validate_scientific_receipt,
)
from oracle_composition.harness.resource_slot import (
    ResourceSlotError,
    release_slot,
    validate_slot_for_reservation,
)

from .contracts import (
    REWARD_COMPOSITOR_SHA256,
    FineTuningRunManifest,
    RewardRegistry,
    load_fine_tuning_run_manifest,
    load_phase_b_oracle,
)
from .isolation import (
    SealedArtifact,
    seal_artifact,
    sealed_input_lineage_sha256,
    sealed_input_lineage_value,
    validate_executing_modules,
    verify_sealed_inputs,
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
    BEHAVIORS,
    COHORT_SEEDS,
    FULL_TRANSITIONS_PER_SEED,
    SMOKE_SEED,
    SMOKE_TRANSITIONS,
    TRAINING_BLOCKS,
    PhaseBTrainingError,
    TrainingFailureStatus,
    TrainingPlan,
    run_ppo_training,
)

SUPERVISOR_ID = "humanoid_phase_b_training_supervisor/v2"
EXECUTION_MANIFEST_SCHEMA_ID = "humanoid_phase_b_execution_manifest/v3"
SEED_SUCCESS_RECEIPT_ID = "humanoid_phase_b_seed_success/v2"
SEED_FAILURE_RECEIPT_ID = "humanoid_phase_b_seed_failure/v2"
JOB_RESULT_SCHEMA_ID = "humanoid_phase_b_job_result/v1"
WORKER_ACK_TIMEOUT_SECONDS = 30.0
POLL_SECONDS = 0.05
TERMINATION_GRACE_SECONDS = 2.0
KILL_GRACE_SECONDS = 2.0
MAX_IPC_FRAME_BYTES = 4 * 1024**2
IPC_FRAME_SCHEMA_ID = "humanoid_phase_b_bounded_ipc_frame/v1"
MAILBOX_MESSAGE_ID = re.compile(r"^\d{8}T\d{6}\.\d{6}Z-[0-9a-f]{32}$")
MAILBOX_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")
MAILBOX_AGENTS = frozenset(("astra", "fable"))
WORKER_ENV_ALLOWLIST = frozenset(
    ("LANG", "LC_ALL", "PATH", "SSL_CERT_DIR", "SSL_CERT_FILE", "TMPDIR")
)
WORKER_ENV_FIXED = {
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "PYTHONHASHSEED": "0",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_SPAWN_ENVIRONMENT_LOCK = threading.Lock()


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
    MALFORMED_FRAME = "malformed_frame"
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
    cpu_time_seconds: float = 22.0 * 60.0
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
    sealed_inputs: tuple[SealedArtifact, ...]
    sealed_input_lineage_sha256: str
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
    sealed_inputs: tuple[SealedArtifact, ...]
    sealed_input_lineage_sha256: str
    repository_root: str
    runtime_kind: str
    runtime_config: RealRuntimeConfig | None
    failure_mode: str | None
    rss_limit_bytes: int
    cpu_time_limit_seconds: float
    worker_environment: tuple[tuple[str, str], ...]


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


class MalformedFrameError(ExperimentContractError):
    pass


class SourceMutationError(ExperimentContractError):
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


def _sealed_training_inputs(
    *,
    repository_root: Path,
    template_manifest: FineTuningRunManifest,
) -> tuple[SealedArtifact, ...]:
    """Enumerate every file the real worker may consume for the frozen run."""

    root = Path(repository_root).resolve(strict=True)
    records: dict[str, dict[str, object]] = {}

    def add(
        path: Path,
        *,
        role: str,
        expected: Mapping[str, object] | None = None,
    ) -> None:
        expected_size = None if expected is None else expected.get("byte_count")
        expected_sha = None if expected is None else expected.get("sha256")
        if expected_size is not None and type(expected_size) is not int:
            raise ExperimentContractError(f"{role} byte-count authority is malformed")
        if expected_sha is not None and type(expected_sha) is not str:
            raise ExperimentContractError(f"{role} digest authority is malformed")
        artifact = seal_artifact(
            root,
            path,
            roles=(role,),
            expected_byte_count=expected_size,
            expected_sha256=expected_sha,
        )
        existing = records.get(artifact.relative_path)
        if existing is None:
            records[artifact.relative_path] = artifact.to_dict()
            return
        if existing["byte_count"] != artifact.byte_count or existing["sha256"] != artifact.sha256:
            raise ExperimentContractError("one sealed path has contradictory identities")
        existing["roles"] = sorted({*existing["roles"], role})

    manifest = template_manifest.value
    for name in (
        "evaluator",
        "library",
        "oracle",
        "reference_corpus",
        "reward",
        "starting_checkpoint",
        "task",
        "training_design",
    ):
        binding = manifest[name]
        add(root / str(binding["path"]), role=name, expected=binding)

    starting_path = root / str(manifest["starting_checkpoint"]["path"])
    starting = json.loads(starting_path.read_bytes())
    for name in ("e1_receipt", "source_expert", "strict_actor_export"):
        binding = starting.get(name)
        if type(binding) is not dict:
            raise ExperimentContractError(f"starting checkpoint {name} binding is malformed")
        add(root / str(binding["path"]), role=f"starting_checkpoint.{name}", expected=binding)

    e1_receipt_path = root / str(starting["e1_receipt"]["path"])
    e1_receipt = json.loads(e1_receipt_path.read_bytes())
    fixture_batch = e1_receipt.get("fixture_batch")
    if (
        type(fixture_batch) is not dict
        or type(fixture_batch.get("synthetic_design_path")) is not str
    ):
        raise ExperimentContractError("E1 synthetic-design binding is malformed")
    add(
        root / fixture_batch["synthetic_design_path"],
        role="starting_checkpoint.e1_synthetic_design",
    )

    corpus_root = root / "artifacts/reference_corpus_v2"
    corpus_manifest_path = root / str(manifest["reference_corpus"]["path"])
    corpus_manifest = json.loads(corpus_manifest_path.read_bytes())
    index_path = corpus_root / "corpus_index_v2.json"
    add(index_path, role="reference_corpus.index")
    index = json.loads(index_path.read_bytes())
    if canonical_json_bytes(index) != index_path.read_bytes():
        raise ExperimentContractError("reference-corpus index is not canonical")
    manifest_entries = {
        entry.get("clip_id"): entry for entry in corpus_manifest.get("clips_in_reset_order", ())
    }
    index_entries = {entry.get("clip_id"): entry for entry in index.get("clips", ())}
    for block in TRAINING_BLOCKS:
        for behavior in BEHAVIORS:
            clip_id = f"corpus-{block}-{behavior}"
            manifest_entry = manifest_entries.get(clip_id)
            index_entry = index_entries.get(clip_id)
            if (
                type(manifest_entry) is not dict
                or type(index_entry) is not dict
                or manifest_entry.get("bundle_manifest_sha256")
                != index_entry.get("bundle_manifest_sha256")
            ):
                raise ExperimentContractError(f"reference-corpus authorities differ for {clip_id}")
            bundle_sha = index_entry.get("bundle_manifest_sha256")
            if type(bundle_sha) is not str:
                raise ExperimentContractError("reference-corpus bundle digest is malformed")
            bundle_path = corpus_root / "clips" / clip_id / f"bundle-{bundle_sha}.json"
            bundle_binding = {
                "byte_count": bundle_path.stat().st_size,
                "sha256": bundle_sha,
            }
            add(bundle_path, role=f"reference_corpus.bundle.{clip_id}", expected=bundle_binding)
            bundle = load_bundle_manifest(bundle_path, expected_sha256=bundle_sha)
            core = bundle["core"]
            nested = [*core["bound_artifacts"], core["payload"], core["rng_state"]["binding"]]
            for nested_binding in nested:
                add(
                    corpus_root / str(nested_binding["object_path"]),
                    role=f"reference_corpus.{clip_id}.{nested_binding['role']}",
                    expected=nested_binding,
                )

    return tuple(
        SealedArtifact(
            relative_path=path,
            byte_count=int(value["byte_count"]),
            sha256=str(value["sha256"]),
            roles=tuple(value["roles"]),
        )
        for path, value in sorted(records.items())
    )


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
    validate_executing_modules(root, source_snapshot.value["source_sha256"])
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
    sealed_inputs = _sealed_training_inputs(
        repository_root=root,
        template_manifest=template,
    )
    lineage_sha256 = sealed_input_lineage_sha256(sealed_inputs)
    report_inputs["sealed_input_lineage_sha256"] = lineage_sha256
    runtime_config = RealRuntimeConfig(
        repository_root=root,
        experiment=experiment_path,
        oracle_path=oracle,
        reward_path=reward,
        starting_actor_path=root / actor_binding["path"],
        starting_actor_sha256=actor_binding["sha256"],
        value_seed=int(starting["value_initialization_seed"]),
        sealed_inputs=sealed_inputs,
        sealed_input_lineage_sha256=lineage_sha256,
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
        sealed_inputs=sealed_inputs,
        sealed_input_lineage_sha256=lineage_sha256,
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
    mailbox = shared_coordination_root(root, supplied=mailbox_root)

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


def shared_coordination_root(repository_root: Path, *, supplied: Path | None = None) -> Path:
    if supplied is not None:
        return Path(supplied).resolve(strict=True)
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return (Path(common).resolve(strict=True) / "harness-coordination").resolve(strict=True)


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
        cpu_time_seconds=min(limits.cpu_time_seconds, deadline),
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
        "sealed_input_lineage": sealed_input_lineage_value(preflight.sealed_inputs),
        "sealed_input_lineage_sha256": preflight.sealed_input_lineage_sha256,
        "prior_scientific_receipt_sha256": preflight.prior_scientific_receipt_sha256,
        "reservation": dict(reservation),
        "reservation_sha256": hashlib.sha256(canonical_json_bytes(dict(reservation))).hexdigest(),
        "resource_limits": limits.to_dict(),
        "resource_control_policy": {
            "cpu_time": "os_rlimit_when_supported_otherwise_recorded_unsupported",
            "environment": "spawn_time_explicit_allowlist",
            "filesystem": "parent_observed_os_best_effort",
            "process_group_cleanup": "os_session_group_best_effort_with_fail_closed_receipt",
            "process_tree_rss": "parent_observed_os_best_effort",
        },
        "runtime_source_snapshot": dict(preflight.source_snapshot.value),
        "runtime_source_snapshot_sha256": preflight.source_snapshot.sha256,
        "schema_version": 3,
        "seeds": list(seeds),
        "smoke": smoke,
        "test_only": test_only,
        "transitions_per_seed": transitions,
    }


_FRAME_PAYLOAD_KEYS = {
    "admit_execution": frozenset(("manifest", "manifest_sha256")),
    "begin_construction": frozenset(("manifest_sha256",)),
    "completed": frozenset(
        (
            "executed_module_identity_sha256",
            "persistence",
            "rsi_ledger",
            "step_zero_comparator",
            "training_facts",
            "training_facts_value",
            "worker_cleanup",
            "worker_peak_rss_bytes",
        )
    ),
    "execution_acknowledged": frozenset(
        (
            "executed_module_identity_sha256",
            "manifest_sha256",
            "model_or_environment_constructed",
            "sealed_input_lineage_sha256",
            "source_snapshot_sha256",
        )
    ),
    "failed": frozenset(("reason", "status", "worker_cleanup")),
    "progress": frozenset(
        ("observed_transitions", "rollout_count", "stream_counts", "worker_peak_rss_bytes")
    ),
    "resource_breach": frozenset(("reason",)),
    "source_mutation": frozenset(("reason",)),
    "worker_started": frozenset(
        (
            "cpu_time_control",
            "environment_control",
            "executed_module_identity_sha256",
            "pgid",
            "pid",
            "sealed_input_lineage_sha256",
            "sid",
            "source_snapshot_sha256",
        )
    ),
}

_ARTIFACT_RECORD_KEYS = frozenset(("byte_count", "filename", "sha256"))
_PERSISTENCE_RECEIPT_KEYS = frozenset(
    (
        "checkpoint",
        "checkpoint_reload_bitwise_deterministic",
        "checkpoint_to_export_bitwise_equivalent",
        "evidence_class",
        "execution_manifest_sha256",
        "final_transition_only",
        "fixture_action_sha256",
        "persistence_receipt_id",
        "planned_transitions",
        "ppo_seed",
        "promotable",
        "schema_version",
        "smoke",
        "strict_export",
        "test_only",
        "training_facts_sha256",
        "transitions",
    )
)
_TRAINING_FACT_KEYS = frozenset(
    (
        "device",
        "evidence_class",
        "execution_manifest_sha256",
        "likelihood_audit",
        "losses",
        "normalization",
        "observed_transitions",
        "optimizer_initialization",
        "optimizer_updates",
        "planned_transitions",
        "ppo_recipe",
        "ppo_recipe_id",
        "ppo_seed",
        "promotable",
        "reward_totals",
        "rng_substreams",
        "rollouts",
        "rsi_ledger_sha256",
        "smoke",
        "step_zero_comparator",
        "stream_counts",
        "thread_counts",
        "time_limit_bootstrap_count",
        "training_worker_id",
        "unfreeze_receipt_sha256",
        "unfreeze_rollouts",
    )
)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _valid_artifact_record(value: object) -> bool:
    if type(value) is not dict or set(value) != _ARTIFACT_RECORD_KEYS:
        return False
    filename = value["filename"]
    return (
        type(value["byte_count"]) is int
        and value["byte_count"] > 0
        and type(filename) is str
        and filename not in {"", ".", ".."}
        and Path(filename).name == filename
        and _is_sha256(value["sha256"])
    )


def _valid_environment_control(value: object) -> bool:
    if type(value) is not dict or set(value) != {
        "allowlist_enforced",
        "environment_sha256",
        "keys",
        "runtime_added_keys_removed",
        "unexpected_keys",
    }:
        return False
    keys = value["keys"]
    removed = value["runtime_added_keys_removed"]
    return (
        value["allowlist_enforced"] is True
        and _is_sha256(value["environment_sha256"])
        and type(keys) is list
        and type(removed) is list
        and all(type(item) is str for item in (*keys, *removed))
        and keys == sorted(set(keys))
        and removed == sorted(set(removed))
        and value["unexpected_keys"] == []
    )


def _valid_cpu_time_control(value: object) -> bool:
    if type(value) is not dict or value.get("resource") != "RLIMIT_CPU":
        return False
    enforcement = value.get("enforcement")
    if enforcement == "os_enforced":
        return (
            set(value) == {"enforcement", "hard_limit_seconds", "limit_seconds", "resource"}
            and type(value["limit_seconds"]) is int
            and type(value["hard_limit_seconds"]) is int
            and 0 < value["limit_seconds"] <= value["hard_limit_seconds"]
        )
    if enforcement == "unsupported":
        expected = {"enforcement", "limit_seconds", "resource"}
        if "error" in value:
            expected.add("error")
        return (
            set(value) == expected
            and type(value["limit_seconds"]) is int
            and value["limit_seconds"] > 0
            and ("error" not in value or type(value["error"]) is str)
        )
    return False


def _valid_step_zero_comparator(value: object) -> bool:
    if type(value) is not dict or value.get("bitwise_equal") is not True:
        return False
    if set(value) == {"bitwise_equal", "evidence_class", "worker_path"}:
        return all(
            type(value[name]) is str and value[name] for name in ("evidence_class", "worker_path")
        )
    if set(value) == {
        "action_sha256",
        "bitwise_equal",
        "e1_receipt_sha256",
        "fixture_count",
        "reported_beside_checkpoint",
        "worker_path",
    }:
        return (
            _is_sha256(value["action_sha256"])
            and _is_sha256(value["e1_receipt_sha256"])
            and type(value["fixture_count"]) is int
            and value["fixture_count"] > 0
            and value["reported_beside_checkpoint"] is True
            and type(value["worker_path"]) is str
            and bool(value["worker_path"])
        )
    return False


def _valid_persistence_receipt(value: object) -> bool:
    if type(value) is not dict or set(value) != _PERSISTENCE_RECEIPT_KEYS:
        return False
    return (
        _valid_artifact_record(value["checkpoint"])
        and _valid_artifact_record(value["strict_export"])
        and value["checkpoint_reload_bitwise_deterministic"] is True
        and value["checkpoint_to_export_bitwise_equivalent"] is True
        and type(value["evidence_class"]) is str
        and _is_sha256(value["execution_manifest_sha256"])
        and value["final_transition_only"] is True
        and _is_sha256(value["fixture_action_sha256"])
        and value["persistence_receipt_id"] == "humanoid_phase_b_final_persistence/v1"
        and type(value["planned_transitions"]) is int
        and value["planned_transitions"] > 0
        and type(value["ppo_seed"]) is int
        and value["ppo_seed"] > 0
        and type(value["promotable"]) is bool
        and value["schema_version"] == 1
        and type(value["smoke"]) is bool
        and type(value["test_only"]) is bool
        and _is_sha256(value["training_facts_sha256"])
        and value["transitions"] == value["planned_transitions"]
    )


def _valid_completed_payload(payload: Mapping[str, object]) -> bool:
    persistence = payload["persistence"]
    training = payload["training_facts_value"]
    if (
        not _is_sha256(payload["executed_module_identity_sha256"])
        or payload["worker_cleanup"] != {"attempted": True, "error": None, "succeeded": True}
        or type(persistence) is not dict
        or set(persistence) != {"checkpoint", "receipt", "receipt_value", "strict_export"}
        or not all(
            _valid_artifact_record(persistence[name])
            for name in ("checkpoint", "receipt", "strict_export")
        )
        or not _valid_persistence_receipt(persistence["receipt_value"])
        or not _valid_artifact_record(payload["rsi_ledger"])
        or not _valid_artifact_record(payload["training_facts"])
        or not _valid_step_zero_comparator(payload["step_zero_comparator"])
        or type(training) is not dict
        or set(training) != _TRAINING_FACT_KEYS
        or training.get("step_zero_comparator") != payload["step_zero_comparator"]
    ):
        return False
    integer_fields = (
        "observed_transitions",
        "optimizer_updates",
        "planned_transitions",
        "ppo_seed",
        "rollouts",
        "time_limit_bootstrap_count",
    )
    return (
        all(type(training[name]) is int and training[name] >= 0 for name in integer_fields)
        and training["observed_transitions"] == training["planned_transitions"]
        and type(training["promotable"]) is bool
        and type(training["smoke"]) is bool
        and _is_sha256(training["execution_manifest_sha256"])
        and _is_sha256(training["rsi_ledger_sha256"])
        and _is_sha256(training["unfreeze_receipt_sha256"])
    )


def _validate_json_value(value: object, *, depth: int = 0) -> None:
    if depth > 16:
        raise MalformedFrameError("IPC frame nesting exceeds the schema bound")
    if value is None or type(value) in {bool, str}:
        if type(value) is str and len(value.encode("utf-8")) > 128 * 1024:
            raise MalformedFrameError("IPC frame string exceeds the schema bound")
        return
    if type(value) is int:
        if not -(2**63) <= value < 2**64:
            raise MalformedFrameError("IPC frame integer exceeds the schema bound")
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise MalformedFrameError("IPC frame contains a non-finite number")
        return
    if type(value) is list:
        if len(value) > 65_536:
            raise MalformedFrameError("IPC frame list exceeds the schema bound")
        for item in value:
            _validate_json_value(item, depth=depth + 1)
        return
    if type(value) is dict:
        if len(value) > 4_096 or any(
            type(key) is not str or len(key.encode("utf-8")) > 1_024 for key in value
        ):
            raise MalformedFrameError("IPC frame mapping exceeds the schema bound")
        for item in value.values():
            _validate_json_value(item, depth=depth + 1)
        return
    raise MalformedFrameError("IPC frame contains a non-JSON value")


def _validate_frame_payload(message_type: str, payload: object) -> dict[str, object]:
    required = _FRAME_PAYLOAD_KEYS.get(message_type)
    if required is None or type(payload) is not dict or set(payload) != required:
        raise MalformedFrameError("IPC frame type or exact payload schema differs")
    _validate_json_value(payload)
    if message_type == "admit_execution" and (
        type(payload["manifest"]) is not dict
        or type(payload["manifest_sha256"]) is not str
        or hashlib.sha256(canonical_json_bytes(payload["manifest"])).hexdigest()
        != payload["manifest_sha256"]
    ):
        raise MalformedFrameError("IPC execution-admission payload is malformed")
    if message_type == "begin_construction" and not _is_sha256(payload["manifest_sha256"]):
        raise MalformedFrameError("IPC construction-admission payload is malformed")
    if message_type in {"worker_started", "execution_acknowledged"}:
        sha_fields = {
            "executed_module_identity_sha256",
            "sealed_input_lineage_sha256",
            "source_snapshot_sha256",
        }
        if any(
            type(payload[field]) is not str
            or len(payload[field]) != 64
            or any(character not in "0123456789abcdef" for character in payload[field])
            for field in sha_fields
        ):
            raise MalformedFrameError("IPC worker identity payload is malformed")
    if message_type == "worker_started" and (
        any(
            type(payload[field]) is not int or payload[field] <= 0
            for field in ("pgid", "pid", "sid")
        )
        or not _valid_cpu_time_control(payload["cpu_time_control"])
        or not _valid_environment_control(payload["environment_control"])
    ):
        raise MalformedFrameError("IPC worker-start payload is malformed")
    if message_type == "execution_acknowledged" and (
        not _is_sha256(payload["manifest_sha256"])
        or payload["model_or_environment_constructed"] is not False
    ):
        raise MalformedFrameError("IPC execution acknowledgment is malformed")
    if message_type in {"resource_breach", "source_mutation"} and (
        type(payload["reason"]) is not str or not payload["reason"]
    ):
        raise MalformedFrameError("IPC failure reason is malformed")
    if message_type == "failed" and (
        type(payload["reason"]) is not str
        or not payload["reason"]
        or payload["status"]
        not in {
            status.value
            for status in SupervisorStatus
            if status not in {SupervisorStatus.SUCCEEDED, SupervisorStatus.NOT_STARTED}
        }
        or not _valid_worker_cleanup(payload["worker_cleanup"])
    ):
        raise MalformedFrameError("IPC worker-failure payload is malformed")
    if message_type == "progress":
        streams = payload["stream_counts"]
        if (
            type(payload["observed_transitions"]) is not int
            or type(payload["rollout_count"]) is not int
            or type(payload["worker_peak_rss_bytes"]) is not int
            or type(streams) is not dict
            or set(streams) != {"composition", "rehearsal"}
            or any(type(value) is not int or value < 0 for value in streams.values())
        ):
            raise MalformedFrameError("IPC progress payload is malformed")
    if message_type == "completed" and (
        not _valid_completed_payload(payload)
        or type(payload["worker_peak_rss_bytes"]) is not int
        or payload["worker_peak_rss_bytes"] <= 0
        or not _valid_worker_cleanup(payload["worker_cleanup"])
    ):
        raise MalformedFrameError("IPC completion payload is malformed")
    return dict(payload)


def _valid_worker_cleanup(value: object) -> bool:
    return (
        type(value) is dict
        and set(value) == {"attempted", "error", "succeeded"}
        and type(value["attempted"]) is bool
        and type(value["succeeded"]) is bool
        and (value["error"] is None or type(value["error"]) is str)
        and (value["succeeded"] is (value["error"] is None))
    )


def _frame_bytes(message_type: str, payload: Mapping[str, object]) -> bytes:
    validated = _validate_frame_payload(message_type, dict(payload))
    encoded = canonical_json_bytes(
        {
            "frame_schema_id": IPC_FRAME_SCHEMA_ID,
            "message_type": message_type,
            "payload": validated,
            "schema_version": 1,
        }
    )
    if not encoded or len(encoded) > MAX_IPC_FRAME_BYTES:
        raise MalformedFrameError("IPC frame exceeds the byte bound")
    return encoded


def _decode_frame(encoded: bytes) -> tuple[str, dict[str, object]]:
    if type(encoded) is not bytes or not encoded or len(encoded) > MAX_IPC_FRAME_BYTES:
        raise MalformedFrameError("IPC frame is empty or exceeds the byte bound")
    try:
        value = json.loads(encoded.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError) as exc:
        raise MalformedFrameError("IPC frame is truncated or invalid JSON") from exc
    _validate_json_value(value)
    if (
        type(value) is not dict
        or set(value) != {"frame_schema_id", "message_type", "payload", "schema_version"}
        or value.get("frame_schema_id") != IPC_FRAME_SCHEMA_ID
        or value.get("schema_version") != 1
        or type(value.get("message_type")) is not str
        or canonical_json_bytes(value) != encoded
    ):
        raise MalformedFrameError("IPC frame envelope or canonical encoding differs")
    message_type = value["message_type"]
    return message_type, _validate_frame_payload(message_type, value["payload"])


def _send_frame(connection: Connection, message_type: str, payload: Mapping[str, object]) -> None:
    connection.send_bytes(_frame_bytes(message_type, payload))


def _receive_frame(connection: Connection) -> tuple[str, dict[str, object]]:
    try:
        encoded = connection.recv_bytes(MAX_IPC_FRAME_BYTES)
    except EOFError:
        raise
    except OSError as exc:
        raise MalformedFrameError("IPC frame is truncated or exceeds the byte bound") from exc
    return _decode_frame(encoded)


def _artifact_record(artifact: PublishedArtifact) -> dict[str, object]:
    return {
        "byte_count": artifact.byte_count,
        "filename": artifact.path.name,
        "sha256": artifact.sha256,
    }


def minimal_worker_environment(source: Mapping[str, str]) -> dict[str, str]:
    """Return the explicit spawn-time environment allowlist plus fixed thread controls."""

    selected = {
        name: value
        for name, value in source.items()
        if name in WORKER_ENV_ALLOWLIST and type(value) is str
    }
    selected.update(WORKER_ENV_FIXED)
    return dict(sorted(selected.items()))


@contextmanager
def _spawn_environment(environment: Mapping[str, str]) -> Iterator[None]:
    with _SPAWN_ENVIRONMENT_LOCK:
        original = dict(os.environ)
        os.environ.clear()
        os.environ.update(environment)
        try:
            yield
        finally:
            os.environ.clear()
            os.environ.update(original)


def _apply_cpu_time_limit(seconds: float) -> dict[str, object]:
    requested = max(1, math.ceil(seconds))
    if not hasattr(resource, "RLIMIT_CPU"):
        return {
            "enforcement": "unsupported",
            "limit_seconds": requested,
            "resource": "RLIMIT_CPU",
        }
    try:
        _old_soft, old_hard = resource.getrlimit(resource.RLIMIT_CPU)
        hard = requested + 1
        if old_hard != resource.RLIM_INFINITY:
            hard = min(hard, int(old_hard))
        soft = min(requested, hard)
        resource.setrlimit(resource.RLIMIT_CPU, (soft, hard))
    except (OSError, ValueError) as exc:
        return {
            "enforcement": "unsupported",
            "error": f"{type(exc).__name__}: {exc}"[:500],
            "limit_seconds": requested,
            "resource": "RLIMIT_CPU",
        }
    return {
        "enforcement": "os_enforced",
        "hard_limit_seconds": hard,
        "limit_seconds": soft,
        "resource": "RLIMIT_CPU",
    }


def _environment_control(request: WorkerRequest) -> dict[str, object]:
    expected = dict(request.worker_environment)
    initial = dict(os.environ)
    removed = sorted(set(initial) - set(expected))
    for name in removed:
        os.environ.pop(name, None)
    observed = dict(os.environ)
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        changed = sorted(
            name for name in set(expected) & set(observed) if expected[name] != observed[name]
        )
        raise ExperimentContractError(
            f"worker environment differs from the allowlist (missing={missing}, changed={changed})"
        )
    return {
        "allowlist_enforced": True,
        "environment_sha256": hashlib.sha256(canonical_json_bytes(observed)).hexdigest(),
        "keys": sorted(observed),
        "runtime_added_keys_removed": removed,
        "unexpected_keys": [],
    }


def _worker_authorities(request: WorkerRequest) -> tuple[RuntimeSourceSnapshot, dict[str, object]]:
    root = Path(request.repository_root)
    try:
        verify_sealed_inputs(
            root,
            request.sealed_inputs,
            expected_lineage_sha256=request.sealed_input_lineage_sha256,
        )
        snapshot = inspect_runtime_sources(root, allow_dirty=request.plan.test_only)
        if snapshot.sha256 != request.source_snapshot_sha256:
            raise ExperimentContractError("worker source snapshot differs from parent preflight")
        modules = validate_executing_modules(root, snapshot.value["source_sha256"])
    except ExperimentContractError as exc:
        raise SourceMutationError(str(exc)) from exc
    return snapshot, modules


def _process_tree_rss_bytes(root_pid: int) -> int | None:
    """Observe aggregate RSS for the worker and its descendants from the parent."""

    try:
        result = subprocess.run(
            ["ps", "-axo", "pid=,ppid=,rss="],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    rows: dict[int, tuple[int, int]] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[0].isdigit() and fields[1].isdigit() and fields[2].isdigit():
            rows[int(fields[0])] = (int(fields[1]), int(fields[2]))
    members = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent_pid, _rss_kib) in rows.items():
            if pid not in members and parent_pid in members:
                members.add(pid)
                changed = True
    return sum(rows.get(pid, (0, 0))[1] for pid in members) * 1024


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
                sealed_inputs=request.sealed_inputs,
                sealed_input_lineage_sha256=request.sealed_input_lineage_sha256,
            )

        environment_factories = real_environment_factories(
            plan=plan,
            config=request.runtime_config,
        )
    else:
        raise ExperimentContractError("worker runtime kind is invalid")

    def progress(value: Mapping[str, object]) -> None:
        rss = _self_peak_rss_bytes()
        _send_frame(connection, "progress", {**dict(value), "worker_peak_rss_bytes": rss})
        if rss > request.rss_limit_bytes:
            _send_frame(connection, "resource_breach", {"reason": "worker RSS limit exceeded"})
            raise _ReportedResourceBreach("worker RSS limit exceeded")

    result = run_ppo_training(
        plan=plan,
        policy_factory=policy_factory,
        environment_factories=environment_factories,
        progress_callback=progress,
        step_zero_audit=step_zero_audit,
    )
    _worker_authorities(request)
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
    _after, final_modules = _worker_authorities(request)
    payload = {
        "executed_module_identity_sha256": final_modules["sha256"],
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
        "worker_cleanup": dict(result.worker_cleanup),
        "worker_peak_rss_bytes": _self_peak_rss_bytes(),
    }
    _send_frame(connection, "completed", payload)


def _worker_session_entry(connection: Connection, request: WorkerRequest) -> None:
    os.setsid()
    worker_cleanup = {"attempted": False, "error": None, "succeeded": True}
    try:
        environment_control = _environment_control(request)
        cpu_time_control = _apply_cpu_time_limit(request.cpu_time_limit_seconds)
        snapshot, modules = _worker_authorities(request)
        _send_frame(
            connection,
            "worker_started",
            {
                "cpu_time_control": cpu_time_control,
                "environment_control": environment_control,
                "executed_module_identity_sha256": modules["sha256"],
                "pgid": os.getpgrp(),
                "pid": os.getpid(),
                "sealed_input_lineage_sha256": request.sealed_input_lineage_sha256,
                "sid": os.getsid(0),
                "source_snapshot_sha256": snapshot.sha256,
            },
        )
        if not connection.poll(WORKER_ACK_TIMEOUT_SECONDS):
            raise ExperimentContractError("worker did not receive execution admission")
        message_type, message = _receive_frame(connection)
        manifest = json.loads(request.execution_manifest_bytes)
        if (
            message_type != "admit_execution"
            or message.get("manifest_sha256") != request.execution_manifest_sha256
            or message.get("manifest") != manifest
            or canonical_json_bytes(message["manifest"]) != request.execution_manifest_bytes
            or hashlib.sha256(request.execution_manifest_bytes).hexdigest()
            != request.execution_manifest_sha256
        ):
            raise ExperimentContractError("worker execution admission differs")
        # This acknowledgement is the last operation before model/environment construction.
        snapshot, modules = _worker_authorities(request)
        _send_frame(
            connection,
            "execution_acknowledged",
            {
                "executed_module_identity_sha256": modules["sha256"],
                "manifest_sha256": request.execution_manifest_sha256,
                "model_or_environment_constructed": False,
                "sealed_input_lineage_sha256": request.sealed_input_lineage_sha256,
                "source_snapshot_sha256": snapshot.sha256,
            },
        )
        if not connection.poll(WORKER_ACK_TIMEOUT_SECONDS):
            raise ExperimentContractError("worker did not receive post-ACK construction admission")
        construction_type, construction = _receive_frame(connection)
        if (
            construction_type != "begin_construction"
            or construction.get("manifest_sha256") != request.execution_manifest_sha256
        ):
            raise ExperimentContractError("post-ACK construction admission differs")
        _worker_authorities(request)
        _worker_execute(connection, request)
    except _ReportedResourceBreach:
        pass
    except PhaseBTrainingError as exc:
        worker_cleanup = {
            "attempted": True,
            "error": exc.cleanup_error,
            "succeeded": exc.cleanup_error is None,
        }
        _send_frame(
            connection,
            "failed",
            {
                "reason": str(exc),
                "status": _TRAINING_TO_SUPERVISOR[exc.status].value,
                "worker_cleanup": worker_cleanup,
            },
        )
    except BaseException as exc:
        with suppress(BrokenPipeError, EOFError, OSError):
            status = (
                SupervisorStatus.MALFORMED_FRAME.value
                if isinstance(exc, MalformedFrameError)
                else SupervisorStatus.SOURCE_MUTATION.value
                if isinstance(exc, SourceMutationError)
                else SupervisorStatus.PRECONDITION_FAILURE.value
                if isinstance(exc, ExperimentContractError)
                else SupervisorStatus.CRASH.value
            )
            _send_frame(
                connection,
                "failed",
                {
                    "reason": f"{type(exc).__name__}: {exc}",
                    "status": status,
                    "worker_cleanup": worker_cleanup,
                },
            )
        raise
    finally:
        connection.close()


def _spawn_worker(request: WorkerRequest) -> tuple[multiprocessing.Process, Connection]:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=True)
    process = context.Process(target=_worker_session_entry, args=(child, request), daemon=False)
    with _spawn_environment(dict(request.worker_environment)):
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


@dataclass(frozen=True, slots=True)
class SupervisionDependencies:
    """Injectable observers used to test real supervisor detectors without resources."""

    clock: Callable[[], float] = time.perf_counter
    spawn_worker: Callable[[WorkerRequest], tuple[object, object]] = _spawn_worker
    process_tree_rss_bytes: Callable[[int], int | None] = _process_tree_rss_bytes
    free_disk_bytes: Callable[[Path], int] = lambda path: shutil.disk_usage(path).free
    directory_bytes: Callable[[Path], int] = _directory_bytes
    cleanup_worker: Callable[..., None] = cleanup_worker_process
    validate_heavy_job_slot: Callable[..., Mapping[str, object]] = validate_slot_for_reservation
    release_heavy_job_slot: Callable[..., Mapping[str, object]] = release_slot


class HeavyJobSlotSession:
    """Retain one validated token identity until all worker cleanup finishes."""

    def __init__(self, dependencies: SupervisionDependencies, *, required: bool) -> None:
        self._dependencies = dependencies
        self._required = required
        self._coordination_root: Path | None = None
        self._reservation: Mapping[str, object] | None = None
        self._expected_wall_seconds: int | None = None
        self._owner: str | None = None
        self._token_id: str | None = None

    def configure(
        self,
        coordination_root: Path,
        reservation: Mapping[str, object],
        *,
        expected_wall_seconds: int,
    ) -> None:
        if not self._required or self._coordination_root is not None:
            raise ExperimentContractError("heavy-job slot session configuration differs")
        self._coordination_root = coordination_root
        self._reservation = dict(reservation)
        self._expected_wall_seconds = expected_wall_seconds

    def validate_before_spawn(self) -> None:
        if not self._required:
            return
        if (
            self._coordination_root is None
            or self._reservation is None
            or self._expected_wall_seconds is None
        ):
            raise ExperimentContractError("heavy-job slot session is not configured")
        try:
            token = self._dependencies.validate_heavy_job_slot(
                self._coordination_root,
                self._reservation,
                expected_wall_seconds=self._expected_wall_seconds,
            )
        except ResourceSlotError as exc:
            raise ExperimentContractError(f"heavy-job slot refused dispatch: {exc}") from exc
        owner = token.get("owner") if type(token) is dict else None
        token_id = token.get("token_id") if type(token) is dict else None
        if type(owner) is not str or type(token_id) is not str:
            raise ExperimentContractError("heavy-job slot returned an invalid token identity")
        if self._owner is None and self._token_id is None:
            self._owner = owner
            self._token_id = token_id
        elif (owner, token_id) != (self._owner, self._token_id):
            raise ExperimentContractError("heavy-job slot identity changed between workers")

    def release(self) -> None:
        if self._owner is None or self._token_id is None:
            return
        assert self._coordination_root is not None
        owner, token_id = self._owner, self._token_id
        try:
            released = self._dependencies.release_heavy_job_slot(
                self._coordination_root,
                owner=owner,
                token_id=token_id,
            )
        except ResourceSlotError as exc:
            raise ExperimentContractError(f"heavy-job slot cleanup failed: {exc}") from exc
        if (
            type(released) is not dict
            or released.get("owner") != owner
            or released.get("token_id") != token_id
        ):
            raise ExperimentContractError("heavy-job slot cleanup identity differs")
        self._owner = None
        self._token_id = None


def _failure_receipt(
    *,
    directory: Path,
    plan: TrainingPlan,
    status: SupervisorStatus,
    reason: str,
    last_stage: str,
    cleanup_succeeded: bool,
    worker_cleanup: Mapping[str, object] | None = None,
    supervisor_cleanup_error: str | None = None,
    resource_controls: Mapping[str, object] | None = None,
) -> tuple[PublishedArtifact, dict[str, object]]:
    primary_reason = reason[:1000]
    value = {
        "cleanup_succeeded": cleanup_succeeded,
        "cleanup_outcome": {
            "supervisor_process_group": {
                "error": supervisor_cleanup_error,
                "succeeded": cleanup_succeeded,
            },
            "worker_environment": dict(
                worker_cleanup or {"attempted": False, "error": None, "succeeded": True}
            ),
        },
        "evidence_class": plan.evidence_class,
        "execution_manifest_sha256": plan.manifest_sha256,
        "failure_receipt_id": SEED_FAILURE_RECEIPT_ID,
        "last_acknowledged_stage": last_stage,
        "outcome": "failure",
        "planned_transitions": plan.transitions,
        "ppo_seed": plan.seed,
        "primary_failure": {"reason": primary_reason, "status": status.value},
        "reason": primary_reason,
        "resource_controls": dict(resource_controls or {}),
        "schema_version": 2,
        "smoke": plan.smoke,
        "status": status.value,
        "success_receipt_present": False,
    }
    artifact = publish_bytes_without_overwrite(
        directory / "failure_receipt_v2.json",
        canonical_json_bytes(value),
    )
    return artifact, value


def _success_receipt(
    *,
    directory: Path,
    plan: TrainingPlan,
    payload: Mapping[str, object],
    resource_controls: Mapping[str, object],
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
        "resource_controls": dict(resource_controls),
        "schema_version": 2,
        "smoke": plan.smoke,
        "status": SupervisorStatus.SUCCEEDED.value,
        "success_receipt_id": SEED_SUCCESS_RECEIPT_ID,
        "test_only": plan.test_only,
        "worker_cleanup": dict(payload["worker_cleanup"]),
    }
    artifact = publish_bytes_without_overwrite(
        directory / "success_receipt_v2.json",
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
    job_output_directory: Path,
    dependencies: SupervisionDependencies,
    validate_slot_before_spawn: Callable[[], None],
) -> SeedOutcome:
    seed_directory.mkdir(mode=0o700)
    request = WorkerRequest(
        plan=plan,
        output_directory=str(seed_directory),
        execution_manifest_bytes=execution_manifest_bytes,
        execution_manifest_sha256=execution_manifest_sha256,
        e1_receipt_sha256=str(preflight.report_inputs["e1_receipt_sha256"]),
        source_snapshot_sha256=preflight.source_snapshot.sha256,
        sealed_inputs=preflight.sealed_inputs,
        sealed_input_lineage_sha256=preflight.sealed_input_lineage_sha256,
        repository_root=str(preflight.repository_root),
        runtime_kind=runtime_kind,
        runtime_config=preflight.runtime_config if runtime_kind == "real" else None,
        failure_mode=failure_mode,
        rss_limit_bytes=limits.rss_bytes,
        cpu_time_limit_seconds=limits.cpu_time_seconds,
        worker_environment=tuple(minimal_worker_environment(os.environ).items()),
    )
    validate_slot_before_spawn()
    process, connection = dependencies.spawn_worker(request)
    started = dependencies.clock()
    last_stage = "spawned"
    group_validated = False
    terminal_payload: Mapping[str, object] | None = None
    terminal_status: SupervisorStatus | None = None
    reason = ""
    progress_anchor_steps = 0
    progress_anchor_time = started
    last_progress_steps = 0
    peak_rss = 0
    child_reported_peak_rss = 0
    minimum_free_disk_bytes = dependencies.free_disk_bytes(job_output_directory)
    maximum_output_bytes = dependencies.directory_bytes(seed_directory)
    maximum_job_output_bytes = dependencies.directory_bytes(job_output_directory)
    throughput_windows: list[dict[str, object]] = []
    resource_controls: dict[str, object] = {
        "cpu_time": {"enforcement": "not_observed"},
        "environment": {"allowlist_enforced": False},
        "executed_modules": {"enforcement": "not_observed"},
        "filesystem": {"enforcement": "parent_observed_os_best_effort"},
        "process_group_cleanup": {"enforcement": "pending"},
        "process_tree_rss": {"enforcement": "parent_observed_os_best_effort"},
    }
    worker_cleanup: Mapping[str, object] = {
        "attempted": False,
        "error": None,
        "succeeded": True,
    }
    supervisor_cleanup_error: str | None = None
    acknowledged_module_identity_sha256: str | None = None
    try:
        while True:
            now = dependencies.clock()
            elapsed = now - started
            if elapsed >= limits.per_seed_wall_seconds:
                terminal_status = SupervisorStatus.TIMEOUT
                reason = "per-seed wall limit exceeded"
                break
            if now >= cohort_job_deadline:
                terminal_status = SupervisorStatus.TIMEOUT
                reason = "cohort or job wall limit exceeded"
                break
            if last_stage in {"spawned", "worker_started"} and elapsed > min(
                WORKER_ACK_TIMEOUT_SECONDS, limits.per_seed_wall_seconds
            ):
                terminal_status = SupervisorStatus.TIMEOUT
                reason = "worker acknowledgement wall limit exceeded"
                break
            free_disk_bytes = dependencies.free_disk_bytes(job_output_directory)
            output_bytes = dependencies.directory_bytes(seed_directory)
            job_output_bytes = dependencies.directory_bytes(job_output_directory)
            minimum_free_disk_bytes = min(minimum_free_disk_bytes, free_disk_bytes)
            maximum_output_bytes = max(maximum_output_bytes, output_bytes)
            maximum_job_output_bytes = max(maximum_job_output_bytes, job_output_bytes)
            if free_disk_bytes < limits.free_disk_bytes:
                terminal_status = SupervisorStatus.RESOURCE_BREACH
                reason = "free disk fell below the hard minimum"
                break
            if job_output_bytes > limits.output_bytes:
                terminal_status = SupervisorStatus.RESOURCE_BREACH
                reason = "aggregate job output exceeded the hard cap"
                break
            pid = process.pid
            if type(pid) is int and pid > 0:
                observed_rss = dependencies.process_tree_rss_bytes(pid)
                if observed_rss is None:
                    resource_controls["process_tree_rss"] = {
                        "enforcement": "unsupported_in_current_os_sandbox"
                    }
                elif type(observed_rss) is not int or observed_rss < 0:
                    terminal_status = SupervisorStatus.RESOURCE_BREACH
                    reason = "parent process-tree RSS observation is invalid"
                    break
                else:
                    peak_rss = max(peak_rss, observed_rss)
                if observed_rss is not None and observed_rss > limits.rss_bytes:
                    terminal_status = SupervisorStatus.RESOURCE_BREACH
                    reason = "parent-observed process-tree RSS limit exceeded"
                    break
            if connection.poll(POLL_SECONDS):
                message_type, payload = _receive_frame(connection)
                if message_type == "worker_started":
                    if (
                        payload.get("pid") != process.pid
                        or payload.get("pgid") != process.pid
                        or payload.get("sid") != process.pid
                        or payload.get("source_snapshot_sha256") != preflight.source_snapshot.sha256
                        or payload.get("sealed_input_lineage_sha256")
                        != preflight.sealed_input_lineage_sha256
                        or type(payload.get("executed_module_identity_sha256")) is not str
                    ):
                        terminal_status = SupervisorStatus.PRECONDITION_FAILURE
                        reason = "worker start or source identity differs"
                        break
                    resource_controls["cpu_time"] = payload["cpu_time_control"]
                    resource_controls["environment"] = payload["environment_control"]
                    acknowledged_module_identity_sha256 = str(
                        payload["executed_module_identity_sha256"]
                    )
                    resource_controls["executed_modules"] = {
                        "enforcement": "checkout_realpath_and_recorded_digest_verified",
                        "start_sha256": acknowledged_module_identity_sha256,
                    }
                    group_validated = True
                    _send_frame(
                        connection,
                        "admit_execution",
                        {
                            "manifest": json.loads(execution_manifest_bytes),
                            "manifest_sha256": execution_manifest_sha256,
                        },
                    )
                    last_stage = "worker_started"
                elif message_type == "execution_acknowledged":
                    if (
                        last_stage != "worker_started"
                        or payload.get("manifest_sha256") != execution_manifest_sha256
                        or payload.get("model_or_environment_constructed") is not False
                        or payload.get("source_snapshot_sha256") != preflight.source_snapshot.sha256
                        or payload.get("sealed_input_lineage_sha256")
                        != preflight.sealed_input_lineage_sha256
                        or payload.get("executed_module_identity_sha256")
                        != acknowledged_module_identity_sha256
                    ):
                        terminal_status = SupervisorStatus.PRECONDITION_FAILURE
                        reason = "worker acknowledgement differs"
                        break
                    _send_frame(
                        connection,
                        "begin_construction",
                        {"manifest_sha256": execution_manifest_sha256},
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
                    child_reported_peak_rss = max(child_reported_peak_rss, rss)
                    now = dependencies.clock()
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
                        if type(rss) is not int or rss <= 0:
                            terminal_status = SupervisorStatus.RESOURCE_BREACH
                            reason = "final worker RSS sample is invalid"
                            break
                        child_reported_peak_rss = max(child_reported_peak_rss, rss)
                        resource_controls["executed_modules"]["final_sha256"] = payload[
                            "executed_module_identity_sha256"
                        ]
                        worker_cleanup = payload["worker_cleanup"]
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
                    worker_cleanup = payload["worker_cleanup"]
                    break
                else:
                    terminal_status = SupervisorStatus.CRASH
                    reason = "worker message type is unknown"
                    break
            elif not process.is_alive():
                cpu_signal = getattr(signal, "SIGXCPU", None)
                if cpu_signal is not None and process.exitcode == -int(cpu_signal):
                    terminal_status = SupervisorStatus.RESOURCE_BREACH
                    reason = "worker exceeded the OS CPU-time limit"
                else:
                    terminal_status = SupervisorStatus.CRASH
                    reason = f"worker exited with code {process.exitcode} before completion"
                break
    except MalformedFrameError as exc:
        terminal_status = SupervisorStatus.MALFORMED_FRAME
        reason = f"malformed worker frame: {exc}"
    except (EOFError, OSError, ExperimentContractError) as exc:
        terminal_status = SupervisorStatus.CRASH
        reason = f"supervisor channel/resource failure: {exc}"
    cleanup_succeeded = True
    try:
        dependencies.cleanup_worker(process, group_validated=group_validated)
        resource_controls["process_group_cleanup"] = {
            "enforcement": "os_session_group_best_effort",
            "succeeded": True,
        }
    except BaseException as exc:
        cleanup_succeeded = False
        terminal_payload = None
        supervisor_cleanup_error = f"{type(exc).__name__}: {exc}"[:1000]
        resource_controls["process_group_cleanup"] = {
            "enforcement": "os_session_group_best_effort",
            "error": supervisor_cleanup_error,
            "succeeded": False,
        }
        if terminal_status is None:
            terminal_status = SupervisorStatus.CLEANUP_FAILURE
            reason = f"worker cleanup failed: {exc}"
    finally:
        connection.close()
    if terminal_payload is not None:
        try:
            verify_sealed_inputs(
                preflight.repository_root,
                preflight.sealed_inputs,
                expected_lineage_sha256=preflight.sealed_input_lineage_sha256,
            )
        except ExperimentContractError as exc:
            terminal_payload = None
            terminal_status = SupervisorStatus.SOURCE_MUTATION
            reason = f"post-worker sealed input verification failed: {exc}"
    final_free_disk_bytes = dependencies.free_disk_bytes(job_output_directory)
    final_output_bytes = dependencies.directory_bytes(seed_directory)
    final_job_output_bytes = dependencies.directory_bytes(job_output_directory)
    minimum_free_disk_bytes = min(minimum_free_disk_bytes, final_free_disk_bytes)
    maximum_output_bytes = max(maximum_output_bytes, final_output_bytes)
    maximum_job_output_bytes = max(maximum_job_output_bytes, final_job_output_bytes)
    if terminal_payload is not None and (
        final_job_output_bytes > limits.output_bytes
        or final_free_disk_bytes < limits.free_disk_bytes
    ):
        terminal_payload = None
        terminal_status = SupervisorStatus.RESOURCE_BREACH
        reason = "post-worker disk or output resource gate was breached"
    telemetry_value = {
        "child_reported_peak_rss_bytes": child_reported_peak_rss,
        "cleanup_succeeded": cleanup_succeeded,
        "last_acknowledged_stage": last_stage,
        "maximum_output_bytes": maximum_output_bytes,
        "maximum_job_output_bytes": maximum_job_output_bytes,
        "minimum_free_disk_bytes": minimum_free_disk_bytes,
        "peak_rss_bytes": peak_rss,
        "resource_limits": limits.to_dict(),
        "resource_controls": resource_controls,
        "seed_wall_seconds": dependencies.clock() - started,
        "supervisor_cleanup_error": supervisor_cleanup_error,
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
        and final_job_output_bytes + len(telemetry_bytes) > limits.output_bytes
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
            worker_cleanup=worker_cleanup,
            supervisor_cleanup_error=supervisor_cleanup_error,
            resource_controls=resource_controls,
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
            worker_cleanup=worker_cleanup,
            supervisor_cleanup_error=supervisor_cleanup_error,
            resource_controls=resource_controls,
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
        resource_controls=resource_controls,
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


def _supervise_training_job(
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
    expected_wall_seconds: int | None = None,
    dependencies: SupervisionDependencies | None = None,
    coordination_root: Path | None = None,
    slot_session: HeavyJobSlotSession,
) -> SupervisionResult:
    """Supervise serial seeds and account for every declared unit on failure."""

    if type(preflight) is not TrainingPreflight:
        raise ExperimentContractError("supervision requires validated preflight authority")
    selected_limits = limits or ResourceLimits()
    selected_dependencies = dependencies or SupervisionDependencies()
    if type(selected_limits) is not ResourceLimits:
        raise ExperimentContractError("supervision resource-limit authority differs")
    if type(selected_dependencies) is not SupervisionDependencies:
        raise ExperimentContractError("supervision dependency authority differs")
    if type(slot_session) is not HeavyJobSlotSession:
        raise ExperimentContractError("heavy-job slot session authority differs")
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
    coordination_directory = (
        None
        if test_only
        else shared_coordination_root(
            preflight.repository_root,
            supplied=coordination_root,
        )
    )
    accepted = validate_reservation(
        reservation,
        smoke=smoke,
        output_directory=output_path,
        test_only=test_only,
        repository_root=preflight.repository_root,
        canonical_argv=canonical_argv,
        current_commit=str(git_value["commit"]),
        expected_inputs=expected_reservation_inputs,
        mailbox_root=coordination_directory,
    )
    if not test_only:
        if type(expected_wall_seconds) is not int or expected_wall_seconds <= 0:
            raise ExperimentContractError(
                "production training requires a positive expected wall time"
            )
        if expected_wall_seconds > int(accepted["hard_wall_seconds"]):
            raise ExperimentContractError(
                "training expected wall time exceeds the accepted hard deadline"
            )
        selected_limits = limits_bound_by_reservation(selected_limits, accepted)
        assert coordination_directory is not None
        slot_session.configure(
            coordination_directory,
            accepted,
            expected_wall_seconds=expected_wall_seconds,
        )
        slot_session.validate_before_spawn()
    verify_sealed_inputs(
        preflight.repository_root,
        preflight.sealed_inputs,
        expected_lineage_sha256=preflight.sealed_input_lineage_sha256,
    )
    validate_executing_modules(
        preflight.repository_root,
        preflight.source_snapshot.value["source_sha256"],
    )
    if selected_dependencies.free_disk_bytes(output_path.parent) < selected_limits.free_disk_bytes:
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
        output / "execution_manifest_v3.json", manifest_bytes
    )
    started = selected_dependencies.clock()
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
        if selected_dependencies.clock() >= cohort_job_deadline:
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
                mailbox_root=coordination_directory,
            )
            _assert_no_conflicting_training_process()
        verify_sealed_inputs(
            preflight.repository_root,
            preflight.sealed_inputs,
            expected_lineage_sha256=preflight.sealed_input_lineage_sha256,
        )
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
            job_output_directory=output,
            dependencies=selected_dependencies,
            validate_slot_before_spawn=slot_session.validate_before_spawn,
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
    job_bytes = canonical_json_bytes(job_value)
    if (
        selected_dependencies.directory_bytes(output) + len(job_bytes)
        > selected_limits.output_bytes
    ):
        overall = SupervisorStatus.RESOURCE_BREACH
        job_value["status"] = overall.value
        job_bytes = canonical_json_bytes(job_value)
    job_result = publish_bytes_without_overwrite(
        output / "job_result_v1.json",
        job_bytes,
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
    expected_wall_seconds: int | None = None,
    dependencies: SupervisionDependencies | None = None,
    coordination_root: Path | None = None,
) -> SupervisionResult:
    """Supervise a job while retaining one exact heavy-slot token through cleanup."""

    selected_dependencies = dependencies or SupervisionDependencies()
    slot_session = HeavyJobSlotSession(selected_dependencies, required=not test_only)
    try:
        return _supervise_training_job(
            preflight=preflight,
            output_directory=output_directory,
            seeds=seeds,
            transitions=transitions,
            smoke=smoke,
            reservation=reservation,
            limits=limits,
            runtime_kind=runtime_kind,
            test_only=test_only,
            test_steps_per_environment=test_steps_per_environment,
            test_batch_size=test_batch_size,
            test_n_epochs=test_n_epochs,
            failure_mode=failure_mode,
            canonical_argv=canonical_argv,
            expected_wall_seconds=expected_wall_seconds,
            dependencies=selected_dependencies,
            coordination_root=coordination_root,
            slot_session=slot_session,
        )
    finally:
        slot_session.release()


__all__ = [
    "MAX_IPC_FRAME_BYTES",
    "HeavyJobSlotSession",
    "MalformedFrameError",
    "ResourceLimits",
    "RuntimeSourceSnapshot",
    "SeedOutcome",
    "SourceMutationError",
    "SupervisionDependencies",
    "SupervisionResult",
    "SupervisorStatus",
    "TrainingPreflight",
    "cleanup_worker_process",
    "inspect_runtime_sources",
    "limits_bound_by_reservation",
    "minimal_worker_environment",
    "shared_coordination_root",
    "supervise_training_job",
    "validate_reservation",
    "validate_training_preflight",
]
