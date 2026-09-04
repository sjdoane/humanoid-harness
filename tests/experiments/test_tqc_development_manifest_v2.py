from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from oracle_composition.experiments import tqc_development_contract_v2 as contract
from oracle_composition.experiments import tqc_development_manifest_v2 as manifest_module
from oracle_composition.experiments import tqc_development_runtime_v2 as runtime_module
from oracle_composition.experiments import tqc_instrumentation_equivalence_v2 as instrument_module
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_calibration_contract import canonical_json
from oracle_composition.experiments.tqc_development_contract_v2 import (
    ValidatedV2E0ReuseEvidence,
    load_tqc_development_design_v2,
)
from oracle_composition.experiments.tqc_development_manifest_v2 import (
    EXECUTION_SOURCE_ROLES,
    PREFLIGHT_SOURCE_ROLES,
    ReservedTQCAttemptDirectoryV2,
    ValidatedTQCExecutionManifestV2,
    ValidatedTQCPreflightContractV2,
    ValidatedTQCWorkerPreflightReceiptsV2,
    publish_tqc_execution_manifest_v2,
    publish_tqc_preflight_contract_v2,
    reserve_tqc_attempt_directory_v2,
    revalidate_tqc_execution_manifest_for_worker_v2,
    revalidate_tqc_execution_manifest_v2,
    revalidate_tqc_preflight_contract_for_worker_v2,
    validate_persisted_tqc_preflight_contract_v2,
    validate_tqc_worker_preflight_receipts_v2,
    worker_preflight_receipt_set_bytes_v2,
)
from oracle_composition.experiments.tqc_development_review_v2 import (
    validate_independent_review_receipt_v2,
)

ROOT = Path(__file__).parents[2]
PACKAGE_ROOT = ROOT / "src/oracle_composition"
CONFIG_ROOT = ROOT / "experiments/bootstrap_tqc_humanoid/configs"
REVIEW_ROOT = ROOT / "experiments/bootstrap_tqc_humanoid/reviews"
DESIGN_PATH = CONFIG_ROOT / "tqc_base_controller_dev_1m_v2.study.json"
AUDIT_PATH = CONFIG_ROOT / "tqc_e0_reuse_dev_1m_v2.audit.json"
CONTRACT_SOURCE_PATH = PACKAGE_ROOT / "experiments/tqc_development_contract_v2.py"
RUNTIME_SOURCE_PATH = PACKAGE_ROOT / "experiments/tqc_development_runtime_v2.py"
INSTRUMENT_SOURCE_PATH = PACKAGE_ROOT / "experiments/tqc_instrumentation_equivalence_v2.py"
CONTRACT_TEST_PATH = ROOT / "tests/experiments/test_tqc_development_contract_v2.py"
PROTOCOL_DOC_PATH = ROOT / "experiments/bootstrap_tqc_humanoid/DEV_1M_BASE_CONTROLLER_V2.md"
LOCK_PATH = ROOT / "uv.lock"
RECEIPT_PATH = REVIEW_ROOT / "tqc_dev_1m_v2_independent_design_review.review.json"
CHECK_PATHS = (
    REVIEW_ROOT / "tqc_dev_1m_v2_reviewed_identity.check.json",
    REVIEW_ROOT / "tqc_dev_1m_v2_focused_contract_tests.check.json",
    REVIEW_ROOT / "tqc_dev_1m_v2_repository_tests.check.json",
)

_EXPECTED_SOURCE_ROOT: list[Path | None] = [None]
_LIVE_WORKER: dict[str, Any] = {}


@dataclass(frozen=True, slots=True)
class FakeValidatedTQCWorkerPreflightDeliveryV2:
    attempt_id: str
    attempt_nonce: str
    preflight_contract_sha256: str
    claimed_work_directory_identity: str
    worker_pid: int
    worker_pgid: int
    worker_sid: int
    worker_process_start_monotonic_seconds: float
    ordered_receipt_sha256: tuple[str, str, str, str]
    preflight_receipts_message_sequence_index: int
    channel_transcript_frame_count: int
    channel_transcript_sha256: str
    channel_parent_send_sequence: int
    channel_parent_receive_sequence: int
    authorizes_model_construction: bool
    behavioral_evidence: bool


@dataclass(slots=True)
class Context:
    design: Any
    e0: Any
    review: Any
    reservation: ReservedTQCAttemptDirectoryV2
    work: Path
    source_root: Path
    dependency_lock: Path
    executable: Path
    runtime: dict[str, Any]
    preflight_sources: dict[str, Path]
    execution_sources: dict[str, Path]
    preflight_path: Path
    preflight: ValidatedTQCPreflightContractV2
    bootstrap_bytes: bytes
    runtime_bytes: bytes
    instrumentation_bytes: bytes
    receipt_set_bytes: bytes
    delivery: FakeValidatedTQCWorkerPreflightDeliveryV2
    worker_receipts: ValidatedTQCWorkerPreflightReceiptsV2


@pytest.fixture(scope="module", autouse=True)
def _live_session_leader() -> Any:
    start = time.perf_counter()
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)"],
        start_new_session=True,
    )
    deadline = time.perf_counter() + 2.0
    try:
        while True:
            try:
                pgid = os.getpgid(process.pid)
                sid = os.getsid(process.pid)
            except ProcessLookupError:
                if time.perf_counter() >= deadline:
                    raise
                time.sleep(0.01)
                continue
            if process.pid == pgid == sid:
                break
            if time.perf_counter() >= deadline:
                raise AssertionError("test worker did not become a session leader")
            time.sleep(0.01)
        _LIVE_WORKER.update(process=process, pid=process.pid, start=start)
        yield
    finally:
        _LIVE_WORKER.clear()
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.fixture(autouse=True)
def _install_future_delivery_type(monkeypatch: pytest.MonkeyPatch) -> Any:
    _EXPECTED_SOURCE_ROOT[0] = None
    monkeypatch.setattr(
        manifest_module,
        "_worker_preflight_delivery_type",
        lambda: FakeValidatedTQCWorkerPreflightDeliveryV2,
    )
    monkeypatch.setattr(
        manifest_module,
        "_expected_project_source_root",
        lambda: _EXPECTED_SOURCE_ROOT[0] or PACKAGE_ROOT.resolve(strict=True),
    )
    monkeypatch.setattr(
        manifest_module,
        "_current_worker_process_identity",
        lambda: (_LIVE_WORKER["pid"], _LIVE_WORKER["pid"], _LIVE_WORKER["pid"]),
    )
    yield
    _EXPECTED_SOURCE_ROOT[0] = None


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _review(design: Any) -> Any:
    return validate_independent_review_receipt_v2(
        design,
        review_receipt_path=RECEIPT_PATH,
        observed_check_receipt_paths=CHECK_PATHS,
        contract_source_path=CONTRACT_SOURCE_PATH,
        contract_test_path=CONTRACT_TEST_PATH,
        protocol_doc_path=PROTOCOL_DOC_PATH,
    )


def _e0() -> ValidatedV2E0ReuseEvidence:
    return ValidatedV2E0ReuseEvidence(
        e0_design_file_sha256=contract.E0_DESIGN_FILE_SHA256,
        e0_receipt_file_sha256=contract.E0_RECEIPT_FILE_SHA256,
        e0_runtime_sha256=contract.E0_RUNTIME_SHA256,
        v1_design_file_sha256=contract.V1_DESIGN_FILE_SHA256,
        training_projection_sha256=contract.TRAINING_PROJECTION_SHA256,
        resource_prior_accepted=True,
        authorizes_training=False,
        _issuer=contract._E0_PREFLIGHT_ISSUER,
    )


def _runtime_receipt_bytes(
    preflight: ValidatedTQCPreflightContractV2,
    design: Any,
    *,
    reset_observation_sha256: str = runtime_module.EXPECTED_RESET_OBSERVATION_SHA256,
) -> bytes:
    projection = design.to_dict()["training_projection"]
    environment = projection["environment"]
    observed_environment = {
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
    preflight_value = preflight.to_dict()
    return canonical_json(
        {
            "schema_version": 1,
            "runtime_receipt_id": runtime_module.RUNTIME_RECEIPT_ID,
            "design_file_sha256": preflight.design_file_sha256,
            "design_semantic_sha256": preflight.design_semantic_sha256,
            "training_projection_sha256": preflight.training_projection_sha256,
            "attempt_id": preflight.attempt_id,
            "attempt_nonce": preflight.attempt_nonce,
            "worker_pid": _LIVE_WORKER["pid"],
            "worker_process_start_monotonic_seconds": _LIVE_WORKER["start"],
            "preflight_contract_sha256": preflight.sha256,
            "claimed_work_directory_identity": preflight.claimed_work_directory_identity,
            "observed_runtime_requirements": projection["runtime_requirements"],
            "observed_simulator": projection["simulator"],
            "observed_environment": observed_environment,
            "python_executable_realpath": preflight_value["bindings"]["python_executable_realpath"],
            "python_executable_file_sha256": preflight_value["bindings"][
                "python_executable_file_sha256"
            ],
            "project_python_source_tree_sha256": (preflight.project_python_source_tree_sha256),
            "runtime_reinspection_source_sha256": preflight_value["bindings"][
                "runtime_reinspection_source_sha256"
            ],
            "reward_or_info_fields_read": False,
            "behavioral_evidence": False,
        }
    )


def _instrumentation_receipt_bytes(preflight: ValidatedTQCPreflightContractV2) -> bytes:
    preflight_value = preflight.to_dict()
    return canonical_json(
        {
            "schema_version": 1,
            "receipt_id": instrument_module.INSTRUMENTATION_RECEIPT_ID,
            "design_file_sha256": preflight.design_file_sha256,
            "design_semantic_sha256": preflight.design_semantic_sha256,
            "training_projection_sha256": preflight.training_projection_sha256,
            "attempt_id": preflight.attempt_id,
            "attempt_nonce": preflight.attempt_nonce,
            "worker_pid": _LIVE_WORKER["pid"],
            "worker_process_start_monotonic_seconds": _LIVE_WORKER["start"],
            "preflight_contract_sha256": preflight.sha256,
            "claimed_work_directory_identity": preflight.claimed_work_directory_identity,
            "project_python_source_tree_sha256": (preflight.project_python_source_tree_sha256),
            "environment_instrumentation_equivalence_source_sha256": preflight_value["bindings"][
                "environment_instrumentation_equivalence_source_sha256"
            ],
            "seed": instrument_module.INSTRUMENTATION_SEED,
            "horizon_steps": instrument_module.INSTRUMENTATION_HORIZON,
            "action_shape": [instrument_module.INSTRUMENTATION_HORIZON, 17],
            "action_sha256": instrument_module.EXPECTED_ACTION_SHA256,
            "reset_comparison_count": 1,
            "per_step_comparison_count": instrument_module.INSTRUMENTATION_HORIZON,
            "comparison_fields": list(instrument_module.COMPARISON_FIELDS),
            "structural_source_proof_sha256": instrument_module._structural_source_proof(
                instrument_module._instrumented_humanoid_class()
            ),
            "contact_trace_sha256": instrument_module.EXPECTED_CONTACT_TRACE_SHA256,
            "contact_sample_count": instrument_module.EXPECTED_CONTACT_SAMPLE_COUNT,
            "plain_shared_trace_sha256": instrument_module.EXPECTED_SHARED_TRACE_SHA256,
            "instrumented_shared_trace_sha256": (instrument_module.EXPECTED_SHARED_TRACE_SHA256),
            "mismatch_count": 0,
            "first_mismatch": None,
            "all_comparisons_passed": True,
            "scope": instrument_module.INSTRUMENTATION_SCOPE,
            "claim_ceiling": instrument_module.CLAIM_CEILING,
            "behavioral_evidence": False,
        }
    )


def _bootstrap_receipt_bytes(preflight: ValidatedTQCPreflightContractV2) -> bytes:
    pid = _LIVE_WORKER["pid"]
    return canonical_json(
        {
            "schema_version": 1,
            "receipt_id": manifest_module.WORKER_BOOTSTRAP_RECEIPT_ID,
            "attempt_id": preflight.attempt_id,
            "attempt_nonce": preflight.attempt_nonce,
            "claimed_work_directory_identity": preflight.claimed_work_directory_identity,
            "worker_pid": pid,
            "worker_pgid": os.getpgid(pid),
            "worker_sid": os.getsid(pid),
            "worker_pid_equals_pgid_equals_sid": True,
            "worker_process_start_monotonic_seconds": _LIVE_WORKER["start"],
            "worker_entrypoint_source_sha256": preflight.to_dict()["bindings"][
                "worker_entrypoint_source_sha256"
            ],
            "preflight_contract_sha256": preflight.sha256,
        }
    )


def _delivery(
    preflight: ValidatedTQCPreflightContractV2,
    bootstrap_bytes: bytes,
    runtime_bytes: bytes,
    instrumentation_bytes: bytes,
    receipt_set_bytes: bytes,
) -> FakeValidatedTQCWorkerPreflightDeliveryV2:
    pid = _LIVE_WORKER["pid"]
    transcript_sha256 = _sha256(
        canonical_json(
            {
                "attempt_nonce": preflight.attempt_nonce,
                "worker_pid": pid,
                "receipt_set_sha256": _sha256(receipt_set_bytes),
            }
        )
    )
    return FakeValidatedTQCWorkerPreflightDeliveryV2(
        attempt_id=preflight.attempt_id,
        attempt_nonce=preflight.attempt_nonce,
        preflight_contract_sha256=preflight.sha256,
        claimed_work_directory_identity=preflight.claimed_work_directory_identity,
        worker_pid=pid,
        worker_pgid=os.getpgid(pid),
        worker_sid=os.getsid(pid),
        worker_process_start_monotonic_seconds=_LIVE_WORKER["start"],
        ordered_receipt_sha256=(
            _sha256(bootstrap_bytes),
            _sha256(runtime_bytes),
            _sha256(instrumentation_bytes),
            _sha256(receipt_set_bytes),
        ),
        preflight_receipts_message_sequence_index=2,
        channel_transcript_frame_count=3,
        channel_transcript_sha256=transcript_sha256,
        channel_parent_send_sequence=0,
        channel_parent_receive_sequence=3,
        authorizes_model_construction=False,
        behavioral_evidence=False,
    )


def _make_context(tmp_path: Path) -> Context:
    tmp_path.mkdir(parents=True, exist_ok=True)
    design = load_tqc_development_design_v2(DESIGN_PATH, AUDIT_PATH)
    e0 = _e0()
    review = _review(design)
    reservation = reserve_tqc_attempt_directory_v2(tmp_path / "attempt")
    source_root = tmp_path / "source"
    source_root.mkdir()
    _EXPECTED_SOURCE_ROOT[0] = source_root.resolve(strict=True)
    all_roles = (*PREFLIGHT_SOURCE_ROLES, *EXECUTION_SOURCE_ROLES)
    role_paths: dict[str, Path] = {}
    copied_sources = {
        "v2_contract_source_sha256": CONTRACT_SOURCE_PATH,
        "runtime_reinspection_source_sha256": RUNTIME_SOURCE_PATH,
        "environment_instrumentation_equivalence_source_sha256": INSTRUMENT_SOURCE_PATH,
    }
    for role in all_roles:
        path = source_root / manifest_module.SOURCE_ROLE_RELATIVE_PATHS[role]
        path.parent.mkdir(parents=True, exist_ok=True)
        source = copied_sources.get(role)
        if source is None:
            path.write_text(f"ROLE = {role!r}\n", encoding="utf-8")
        else:
            shutil.copyfile(source, path)
        role_paths[role] = path
    dependency_lock = tmp_path / "uv.lock"
    shutil.copyfile(LOCK_PATH, dependency_lock)
    executable = tmp_path / "python3.13"
    executable.write_bytes(b"synthetic exact executable identity\n")
    os.chmod(executable, 0o755)
    runtime = design.to_dict()["training_projection"]["runtime_requirements"]
    preflight_sources = {role: role_paths[role] for role in PREFLIGHT_SOURCE_ROLES}
    execution_sources = {role: role_paths[role] for role in EXECUTION_SOURCE_ROLES}
    preflight = publish_tqc_preflight_contract_v2(
        reservation.preflight_contract_path,
        design,
        e0,
        review,
        source_paths=preflight_sources,
        project_source_root=source_root,
        dependency_lock_path=dependency_lock,
        python_executable_path=executable,
        reservation=reservation,
    )
    bootstrap_bytes = _bootstrap_receipt_bytes(preflight)
    runtime_bytes = _runtime_receipt_bytes(preflight, design)
    instrumentation_bytes = _instrumentation_receipt_bytes(preflight)
    receipt_set_bytes = worker_preflight_receipt_set_bytes_v2(
        preflight,
        worker_bootstrap_receipt_bytes=bootstrap_bytes,
        host_runtime_receipt_bytes=runtime_bytes,
        instrumentation_receipt_bytes=instrumentation_bytes,
    )
    delivery = _delivery(
        preflight,
        bootstrap_bytes,
        runtime_bytes,
        instrumentation_bytes,
        receipt_set_bytes,
    )
    worker_receipts = validate_tqc_worker_preflight_receipts_v2(
        design,
        preflight,
        supervised_delivery=delivery,
        worker_bootstrap_receipt_bytes=bootstrap_bytes,
        host_runtime_receipt_bytes=runtime_bytes,
        instrumentation_receipt_bytes=instrumentation_bytes,
        worker_receipt_set_bytes=receipt_set_bytes,
    )
    return Context(
        design=design,
        e0=e0,
        review=review,
        reservation=reservation,
        work=reservation.path,
        source_root=source_root,
        dependency_lock=dependency_lock,
        executable=executable,
        runtime=runtime,
        preflight_sources=preflight_sources,
        execution_sources=execution_sources,
        preflight_path=reservation.preflight_contract_path,
        preflight=preflight,
        bootstrap_bytes=bootstrap_bytes,
        runtime_bytes=runtime_bytes,
        instrumentation_bytes=instrumentation_bytes,
        receipt_set_bytes=receipt_set_bytes,
        delivery=delivery,
        worker_receipts=worker_receipts,
    )


def _receipt_kwargs(context: Context) -> dict[str, Any]:
    return {
        "supervised_delivery": context.delivery,
        "worker_bootstrap_receipt_bytes": context.bootstrap_bytes,
        "host_runtime_receipt_bytes": context.runtime_bytes,
        "instrumentation_receipt_bytes": context.instrumentation_bytes,
        "worker_receipt_set_bytes": context.receipt_set_bytes,
    }


def _publish_manifest(context: Context) -> ValidatedTQCExecutionManifestV2:
    return publish_tqc_execution_manifest_v2(
        context.reservation.execution_manifest_path,
        context.design,
        context.e0,
        context.review,
        context.preflight,
        context.worker_receipts,
        execution_source_paths=context.execution_sources,
    )


def test_two_phase_manifest_binds_every_role_and_separates_observations(
    tmp_path: Path,
) -> None:
    context = _make_context(tmp_path)
    issued = _publish_manifest(context)
    payload = issued.to_dict()

    assert set(payload["bindings"]) == set(
        context.design.to_dict()["execution_manifest"]["must_bind"]
    )
    assert payload["requested"]["runtime_requirements"] == context.runtime
    assert payload["observed"]["runtime_requirements"] == context.runtime
    assert payload["requested"] is not payload["observed"]
    assert (
        payload["observed"]["worker_receipts"]["supervised_delivery"]["channel_transcript_sha256"]
        == context.delivery.channel_transcript_sha256
    )
    assert issued.attempt_id == manifest_module.ATTEMPT_ID
    assert issued.design_file_sha256 == contract.DESIGN_FILE_SHA256
    assert issued.design_semantic_sha256 == contract.DESIGN_SEMANTIC_SHA256
    assert issued.training_projection_sha256 == contract.TRAINING_PROJECTION_SHA256
    assert issued.model_construction_allowed is True
    assert issued.authorizes_training is True
    assert issued.behavioral_evidence is False
    assert context.reservation.execution_manifest_path.stat().st_mode & 0o777 == 0o600

    reloaded = revalidate_tqc_execution_manifest_v2(
        context.reservation.execution_manifest_path,
        expected_sha256=issued.sha256,
        expected_byte_count=issued.byte_count,
        design=context.design,
        e0=context.e0,
        review=context.review,
        preflight=context.preflight,
        worker_receipts=context.worker_receipts,
        execution_source_paths=context.execution_sources,
    )
    assert reloaded.canonical_bytes == issued.canonical_bytes
    worker_preflight = revalidate_tqc_preflight_contract_for_worker_v2(
        context.preflight_path,
        context.design,
        context.e0,
        context.review,
        expected_preflight_sha256=context.preflight.sha256,
        expected_design_file_sha256=contract.DESIGN_FILE_SHA256,
        expected_design_semantic_sha256=contract.DESIGN_SEMANTIC_SHA256,
        expected_e0_audit_file_sha256=contract.AUDIT_FILE_SHA256,
        expected_training_projection_sha256=contract.TRAINING_PROJECTION_SHA256,
        expected_claimed_work_directory_absolute_path=context.work,
    )
    worker_manifest = revalidate_tqc_execution_manifest_for_worker_v2(
        context.reservation.execution_manifest_path,
        expected_sha256=issued.sha256,
        expected_byte_count=issued.byte_count,
        design=context.design,
        e0=context.e0,
        review=context.review,
        preflight=worker_preflight,
        worker_bootstrap_receipt_bytes=context.bootstrap_bytes,
        host_runtime_receipt_bytes=context.runtime_bytes,
        instrumentation_receipt_bytes=context.instrumentation_bytes,
        worker_receipt_set_bytes=context.receipt_set_bytes,
        execution_source_paths=context.execution_sources,
        preflight_receipts_message_sequence_index=(
            context.delivery.preflight_receipts_message_sequence_index
        ),
        channel_transcript_frame_count=context.delivery.channel_transcript_frame_count,
        channel_transcript_sha256=context.delivery.channel_transcript_sha256,
    )
    assert worker_manifest.sha256 == issued.sha256


def test_reservation_is_fresh_private_exact_and_non_reusable(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(ExperimentContractError, match="already exists"):
        reserve_tqc_attempt_directory_v2(existing)

    reservation = reserve_tqc_attempt_directory_v2(tmp_path / "new")
    assert reservation.path.stat().st_mode & 0o777 == 0o700
    assert reservation.reservation_receipt_path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ExperimentContractError, match="already exists"):
        reserve_tqc_attempt_directory_v2(reservation.path)
    with pytest.raises(ExperimentContractError, match="exclusive creation"):
        ReservedTQCAttemptDirectoryV2(**dataclasses.asdict(reservation))


def test_preflight_is_canonical_exclusive_and_reloads_live_paths(tmp_path: Path) -> None:
    context = _make_context(tmp_path)
    assert canonical_json(context.preflight.to_dict()) == context.preflight_path.read_bytes()
    assert context.preflight_path.stat().st_mode & 0o777 == 0o600
    reloaded = validate_persisted_tqc_preflight_contract_v2(
        context.preflight_path,
        context.design,
        context.e0,
        context.review,
        source_paths=context.preflight_sources,
        project_source_root=context.source_root,
        dependency_lock_path=context.dependency_lock,
        python_executable_path=context.executable,
        reservation=context.reservation,
    )
    assert reloaded.sha256 == context.preflight.sha256
    with pytest.raises(ExperimentContractError, match="overwrite"):
        publish_tqc_preflight_contract_v2(
            context.preflight_path,
            context.design,
            context.e0,
            context.review,
            source_paths=context.preflight_sources,
            project_source_root=context.source_root,
            dependency_lock_path=context.dependency_lock,
            python_executable_path=context.executable,
            reservation=context.reservation,
        )


def test_preflight_rejects_unreserved_path_and_detached_source_tree(tmp_path: Path) -> None:
    context = _make_context(tmp_path / "context")
    other = reserve_tqc_attempt_directory_v2(tmp_path / "other")
    with pytest.raises(ExperimentContractError, match="path differs from its reservation"):
        publish_tqc_preflight_contract_v2(
            other.path / "other.contract.json",
            context.design,
            context.e0,
            context.review,
            source_paths=context.preflight_sources,
            project_source_root=context.source_root,
            dependency_lock_path=context.dependency_lock,
            python_executable_path=context.executable,
            reservation=other,
        )

    detached = reserve_tqc_attempt_directory_v2(tmp_path / "detached")
    _EXPECTED_SOURCE_ROOT[0] = PACKAGE_ROOT.resolve(strict=True)
    with pytest.raises(ExperimentContractError, match="not the imported"):
        publish_tqc_preflight_contract_v2(
            detached.preflight_contract_path,
            context.design,
            context.e0,
            context.review,
            source_paths=context.preflight_sources,
            project_source_root=context.source_root,
            dependency_lock_path=context.dependency_lock,
            python_executable_path=context.executable,
            reservation=detached,
        )


def test_worker_bootstrap_rejects_unknown_missing_noncanonical_and_bool_pid(
    tmp_path: Path,
) -> None:
    context = _make_context(tmp_path)
    original = json.loads(context.bootstrap_bytes)
    mutations = []
    mutations.append((canonical_json(dict(original, unknown=True)), "keys differ"))
    missing = dict(original)
    del missing["worker_sid"]
    mutations.append((canonical_json(missing), "keys differ"))
    mutations.append((context.bootstrap_bytes + b"\n", "not canonical"))
    mutations.append((canonical_json(dict(original, worker_pid=True)), "worker_pid"))
    for encoded, message in mutations:
        kwargs = _receipt_kwargs(context)
        kwargs["worker_bootstrap_receipt_bytes"] = encoded
        with pytest.raises(ExperimentContractError, match=message):
            validate_tqc_worker_preflight_receipts_v2(
                context.design,
                context.preflight,
                **kwargs,
            )


def test_worker_receipts_require_exact_supervised_delivery_authority(tmp_path: Path) -> None:
    context = _make_context(tmp_path)
    kwargs = _receipt_kwargs(context)
    kwargs["supervised_delivery"] = object()
    with pytest.raises(ExperimentContractError, match="supervised delivery authority"):
        validate_tqc_worker_preflight_receipts_v2(
            context.design,
            context.preflight,
            **kwargs,
        )

    kwargs["supervised_delivery"] = dataclasses.replace(
        context.delivery,
        worker_pid=context.delivery.worker_pid + 1,
    )
    with pytest.raises(ExperimentContractError, match="worker_pid"):
        validate_tqc_worker_preflight_receipts_v2(
            context.design,
            context.preflight,
            **kwargs,
        )


def test_supervised_delivery_rejects_changed_ordered_receipt_binding(tmp_path: Path) -> None:
    context = _make_context(tmp_path)
    changed_order = (
        *context.delivery.ordered_receipt_sha256[:3],
        "d" * 64,
    )
    kwargs = _receipt_kwargs(context)
    kwargs["supervised_delivery"] = dataclasses.replace(
        context.delivery,
        ordered_receipt_sha256=changed_order,
    )
    with pytest.raises(ExperimentContractError, match="ordered_receipt_sha256"):
        validate_tqc_worker_preflight_receipts_v2(
            context.design,
            context.preflight,
            **kwargs,
        )


def test_worker_receipts_reject_another_attempt_replay(tmp_path: Path) -> None:
    first = _make_context(tmp_path / "first")
    second = _make_context(tmp_path / "second")
    with pytest.raises(ExperimentContractError, match=r"nonce|preflight binding"):
        validate_tqc_worker_preflight_receipts_v2(
            second.design,
            second.preflight,
            **_receipt_kwargs(first),
        )


def test_runtime_and_instrumentation_bytes_are_strictly_readmitted(tmp_path: Path) -> None:
    context = _make_context(tmp_path)
    kwargs = _receipt_kwargs(context)
    kwargs["host_runtime_receipt_bytes"] = context.runtime_bytes + b"\n"
    with pytest.raises(ExperimentContractError, match="not canonical"):
        validate_tqc_worker_preflight_receipts_v2(
            context.design,
            context.preflight,
            **kwargs,
        )

    instrument = json.loads(context.instrumentation_bytes)
    instrument["mismatch_count"] = 1
    instrument["first_mismatch"] = {"boundary_index": 4, "field": "reward"}
    instrument["all_comparisons_passed"] = False
    kwargs = _receipt_kwargs(context)
    kwargs["instrumentation_receipt_bytes"] = canonical_json(instrument)
    with pytest.raises(ExperimentContractError, match="instrumentation comparisons"):
        validate_tqc_worker_preflight_receipts_v2(
            context.design,
            context.preflight,
            **kwargs,
        )


def test_changed_source_tree_and_restored_bytes_both_block_manifest(tmp_path: Path) -> None:
    changed_context = _make_context(tmp_path / "changed")
    changed = changed_context.execution_sources["training_adapter_source_sha256"]
    changed.write_text("ROLE = 'changed'\n", encoding="utf-8")
    with pytest.raises(ExperimentContractError, match="current project source tree"):
        _publish_manifest(changed_context)

    restored_context = _make_context(tmp_path / "restored")
    restored = restored_context.preflight_sources["worker_entrypoint_source_sha256"]
    original = restored.read_bytes()
    restored.write_bytes(original + b"# transient\n")
    restored.write_bytes(original)
    with pytest.raises(ExperimentContractError, match="current project source tree"):
        _publish_manifest(restored_context)


def test_replaced_work_directory_or_reservation_receipt_blocks_manifest(tmp_path: Path) -> None:
    replaced = _make_context(tmp_path / "replaced")
    moved = tmp_path / "replaced-original"
    replaced.work.rename(moved)
    replaced.work.mkdir(mode=0o700)
    os.chmod(replaced.work, 0o700)
    with pytest.raises(ExperimentContractError, match=r"reservation receipt|work-directory"):
        _publish_manifest(replaced)

    tampered = _make_context(tmp_path / "tampered")
    os.chmod(tampered.reservation.reservation_receipt_path, 0o644)
    with pytest.raises(ExperimentContractError, match="receipt mode must be 0600"):
        _publish_manifest(tampered)


def test_source_roles_require_exact_canonical_paths(tmp_path: Path) -> None:
    context = _make_context(tmp_path / "context")
    reservation = reserve_tqc_attempt_directory_v2(tmp_path / "second")
    paths = dict(context.preflight_sources)
    paths["worker_entrypoint_source_sha256"] = paths["runtime_reinspection_source_sha256"]
    with pytest.raises(ExperimentContractError, match="canonical relative path"):
        publish_tqc_preflight_contract_v2(
            reservation.preflight_contract_path,
            context.design,
            context.e0,
            context.review,
            source_paths=paths,
            project_source_root=context.source_root,
            dependency_lock_path=context.dependency_lock,
            python_executable_path=context.executable,
            reservation=reservation,
        )


def test_missing_or_outside_execution_source_role_fails_closed(tmp_path: Path) -> None:
    context = _make_context(tmp_path)
    missing = dict(context.execution_sources)
    del missing["resource_monitor_source_sha256"]
    with pytest.raises(ExperimentContractError, match="execution source paths keys differ"):
        publish_tqc_execution_manifest_v2(
            context.reservation.execution_manifest_path,
            context.design,
            context.e0,
            context.review,
            context.preflight,
            context.worker_receipts,
            execution_source_paths=missing,
        )

    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 1\n", encoding="utf-8")
    paths = dict(context.execution_sources)
    paths["resource_monitor_source_sha256"] = outside
    with pytest.raises(ExperimentContractError, match="outside the bound project source tree"):
        publish_tqc_execution_manifest_v2(
            context.reservation.execution_manifest_path,
            context.design,
            context.e0,
            context.review,
            context.preflight,
            context.worker_receipts,
            execution_source_paths=paths,
        )


def test_manifest_wrong_path_hash_and_mode_fail_reverification(tmp_path: Path) -> None:
    context = _make_context(tmp_path)
    with pytest.raises(ExperimentContractError, match="path differs from its reservation"):
        publish_tqc_execution_manifest_v2(
            context.work / "other.manifest.json",
            context.design,
            context.e0,
            context.review,
            context.preflight,
            context.worker_receipts,
            execution_source_paths=context.execution_sources,
        )
    issued = _publish_manifest(context)
    path = context.reservation.execution_manifest_path
    with pytest.raises(ExperimentContractError, match="delivered execution manifest SHA-256"):
        revalidate_tqc_execution_manifest_v2(
            path,
            expected_sha256="f" * 64,
            expected_byte_count=issued.byte_count,
            design=context.design,
            e0=context.e0,
            review=context.review,
            preflight=context.preflight,
            worker_receipts=context.worker_receipts,
            execution_source_paths=context.execution_sources,
        )
    os.chmod(path, 0o644)
    with pytest.raises(ExperimentContractError, match="mode must be 0600"):
        revalidate_tqc_execution_manifest_v2(
            path,
            expected_sha256=issued.sha256,
            expected_byte_count=issued.byte_count,
            design=context.design,
            e0=context.e0,
            review=context.review,
            preflight=context.preflight,
            worker_receipts=context.worker_receipts,
            execution_source_paths=context.execution_sources,
        )


def test_capability_classes_cannot_be_constructed_or_replaced_publicly(tmp_path: Path) -> None:
    context = _make_context(tmp_path)
    with pytest.raises(ExperimentContractError, match="issued by validation"):
        ValidatedTQCPreflightContractV2(**dataclasses.asdict(context.preflight))
    with pytest.raises(ExperimentContractError, match="issued by validation"):
        dataclasses.replace(context.preflight)
    with pytest.raises(ExperimentContractError, match="issued by validation"):
        ValidatedTQCWorkerPreflightReceiptsV2(**dataclasses.asdict(context.worker_receipts))
    issued = _publish_manifest(context)
    with pytest.raises(ExperimentContractError, match="issued by validation"):
        ValidatedTQCExecutionManifestV2(**dataclasses.asdict(issued))


def test_preflight_rejects_wrong_host_observation_and_nonprivate_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _make_context(tmp_path / "context")
    observed = manifest_module._observe_preflight_runtime_identity()
    changed_observed = dict(observed, python_version="0.0.0")
    monkeypatch.setattr(
        manifest_module,
        "_observe_preflight_runtime_identity",
        lambda: changed_observed,
    )
    wrong = reserve_tqc_attempt_directory_v2(tmp_path / "wrong-runtime")
    with pytest.raises(ExperimentContractError, match="observed preflight python_version"):
        publish_tqc_preflight_contract_v2(
            wrong.preflight_contract_path,
            context.design,
            context.e0,
            context.review,
            source_paths=context.preflight_sources,
            project_source_root=context.source_root,
            dependency_lock_path=context.dependency_lock,
            python_executable_path=context.executable,
            reservation=wrong,
        )

    private = reserve_tqc_attempt_directory_v2(tmp_path / "mode")
    os.chmod(private.path, 0o755)
    with pytest.raises(ExperimentContractError, match="mode must be 0700"):
        publish_tqc_preflight_contract_v2(
            private.preflight_contract_path,
            context.design,
            context.e0,
            context.review,
            source_paths=context.preflight_sources,
            project_source_root=context.source_root,
            dependency_lock_path=context.dependency_lock,
            python_executable_path=context.executable,
            reservation=private,
        )
