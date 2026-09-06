from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.phase_b import supervision as supervision_module
from oracle_composition.phase_b.supervision import (
    ResourceLimits,
    SupervisorStatus,
    limits_bound_by_reservation,
    supervise_training_job,
    validate_reservation,
    validate_training_preflight,
)

ROOT = Path(__file__).resolve().parents[2]
PHASE_B = ROOT / "experiments/003_composition_speed_profile/phase_b"


@pytest.fixture(scope="module")
def preflight() -> object:
    return validate_training_preflight(
        repository_root=ROOT,
        experiment=ROOT / "experiments/003_composition_speed_profile",
        oracle_path=PHASE_B / "oracle_cycle_1_reference_v1.json",
        reward_path=PHASE_B / "tracking_only_v1.json",
        allow_dirty=True,
    )


def _limits(*, seed_wall: float = 20.0) -> ResourceLimits:
    return ResourceLimits(
        per_seed_wall_seconds=seed_wall,
        cohort_wall_seconds=30.0,
        job_wall_seconds=30.0,
        rss_bytes=8 * 1024**3,
        free_disk_bytes=1,
        output_bytes=64 * 1024**2,
        throughput_floor_steps_s=1.0,
        throughput_warmup_transitions=65_536,
        throughput_window_transitions=65_536,
    )


def _run(
    preflight: object,
    output: Path,
    *,
    failure_mode: str | None = None,
    seed_wall: float = 20.0,
) -> object:
    return supervise_training_job(
        preflight=preflight,
        output_directory=output.resolve(),
        seeds=(11, 13),
        transitions=16,
        smoke=False,
        reservation=None,
        limits=_limits(seed_wall=seed_wall),
        runtime_kind="fake",
        test_only=True,
        test_steps_per_environment=4,
        test_batch_size=16,
        test_n_epochs=1,
        failure_mode=failure_mode,
    )


def test_fake_supervisor_two_seed_science_is_byte_deterministic(
    tmp_path: Path, preflight: object
) -> None:
    first = _run(preflight, tmp_path / "first")
    second = _run(preflight, tmp_path / "second")
    assert first.status == second.status == SupervisorStatus.SUCCEEDED
    assert [outcome.seed for outcome in first.outcomes] == [11, 13]
    assert all(outcome.persistence is not None for outcome in first.outcomes)

    deterministic_paths = ["execution_manifest_v2.json", "job_result_v1.json"]
    for seed in (11, 13):
        deterministic_paths.extend(
            (
                f"seed_{seed}/actor_seed_{seed}_final.npz",
                f"seed_{seed}/checkpoint_seed_{seed}_final.npz",
                f"seed_{seed}/persistence_seed_{seed}_v1.json",
                f"seed_{seed}/rsi_ledger_v1.json",
                f"seed_{seed}/success_receipt_v1.json",
                f"seed_{seed}/training_facts_v1.json",
            )
        )
    for relative in deterministic_paths:
        assert (tmp_path / "first" / relative).read_bytes() == (
            tmp_path / "second" / relative
        ).read_bytes()
    assert (tmp_path / "first/seed_11/telemetry_v1.json").read_bytes() != (
        tmp_path / "second/seed_11/telemetry_v1.json"
    ).read_bytes()
    telemetry = json.loads((tmp_path / "first/seed_11/telemetry_v1.json").read_bytes())
    assert telemetry["status"] == "succeeded"
    assert telemetry["minimum_free_disk_bytes"] > 0
    assert telemetry["maximum_output_bytes"] > 0
    assert telemetry["resource_limits"] == _limits().to_dict()


@pytest.mark.parametrize(
    ("failure_mode", "status", "seed_wall"),
    [
        ("hang", SupervisorStatus.TIMEOUT, 0.25),
        ("crash", SupervisorStatus.CRASH, 20.0),
        ("non_finite", SupervisorStatus.NON_FINITE, 20.0),
        ("likelihood_failure", SupervisorStatus.LIKELIHOOD_FAILURE, 20.0),
        ("counter_drift", SupervisorStatus.COUNTER_DRIFT, 20.0),
        ("action_bound_violation", SupervisorStatus.ACTION_BOUND_VIOLATION, 20.0),
        ("phase_selection", SupervisorStatus.PHASE_SELECTION_FAILURE, 20.0),
        ("source_mutation", SupervisorStatus.SOURCE_MUTATION, 20.0),
        ("resource_breach", SupervisorStatus.RESOURCE_BREACH, 20.0),
        ("cleanup_failure", SupervisorStatus.CLEANUP_FAILURE, 20.0),
    ],
)
def test_controlled_failure_status_has_no_success_and_accounts_for_all_seeds(
    tmp_path: Path,
    preflight: object,
    failure_mode: str,
    status: SupervisorStatus,
    seed_wall: float,
) -> None:
    output = tmp_path / failure_mode
    result = _run(preflight, output, failure_mode=failure_mode, seed_wall=seed_wall)
    assert result.status == status
    assert [(item.seed, item.status) for item in result.outcomes] == [
        (11, status),
        (13, SupervisorStatus.NOT_STARTED),
    ]
    for seed in (11, 13):
        seed_directory = output / f"seed_{seed}"
        assert not (seed_directory / "success_receipt_v1.json").exists()
        assert (seed_directory / "failure_receipt_v1.json").is_file()
        assert len(list(seed_directory.glob("*success_receipt*"))) == 0
        assert len(list(seed_directory.glob("*failure_receipt*"))) == 1


def test_output_is_fresh_and_no_overwrite(tmp_path: Path, preflight: object) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    with pytest.raises(ExperimentContractError, match="fresh and no-overwrite"):
        _run(preflight, output)


def test_mailbox_reservation_requires_the_complete_frozen_declaration(tmp_path: Path) -> None:
    output = (tmp_path / "reserved-output").resolve()
    mailbox = tmp_path / "mailbox"
    (mailbox / "messages").mkdir(parents=True)
    (mailbox / "acks").mkdir()
    proposal_id = "20260906T000000.000000Z-" + "1" * 32
    message_id = "20260906T000001.000000Z-" + "2" * 32
    accepted_until = (datetime.now(UTC) + timedelta(minutes=20)).isoformat().replace("+00:00", "Z")
    authoritative = {
        "accepted_until_utc": accepted_until,
        "canonical_argv": ["python", "-m", "oracle_composition.harness.cycle_cli", "train"],
        "commit": "1" * 40,
        "conflict_check": "no_other_heavy_repository_job",
        "hard_wall_seconds": 1_200,
        "inputs": {"oracle_file_sha256": "2" * 64},
        "mode": "smoke",
        "output": str(output),
        "owner": "phase-b-smoke-owner",
        "proposal_id": proposal_id,
        "required_authorizer": "fable",
    }
    message_path = mailbox / "messages" / f"{message_id}.json"
    message_path.write_text(
        json.dumps(
            {
                "body": canonical_json_bytes(authoritative).decode(),
                "from": "fable",
                "id": message_id,
                "kind": "acceptance",
                "reply_to": proposal_id,
                "schema_version": 1,
                "sent_at": "2026-09-06T00:00:01.000000Z",
                "subject": "Phase B smoke reservation",
                "to": "astra",
            },
            sort_keys=True,
        )
    )
    ack_path = mailbox / "acks" / f"{message_id}.json"
    ack_path.write_text(
        json.dumps(
            {
                "acknowledged_at": "2026-09-06T00:00:02.000000Z",
                "meaning": "read_not_agreement",
                "message_id": message_id,
                "recipient": "astra",
                "schema_version": 1,
            },
            sort_keys=True,
        )
    )
    reservation = {
        "accepted": True,
        "acceptance_message": {
            "path": f"messages/{message_id}.json",
            "sha256": hashlib.sha256(message_path.read_bytes()).hexdigest(),
        },
        "acknowledgment": {
            "path": f"acks/{message_id}.json",
            "sha256": hashlib.sha256(ack_path.read_bytes()).hexdigest(),
        },
        "schema_version": 2,
        **authoritative,
    }
    assert (
        validate_reservation(
            reservation,
            smoke=True,
            output_directory=output,
            test_only=False,
            mailbox_root=mailbox,
        )
        == reservation
    )
    incomplete = dict(reservation)
    incomplete.pop("acceptance_message")
    with pytest.raises(ExperimentContractError, match="missing or not accepted"):
        validate_reservation(
            incomplete,
            smoke=True,
            output_directory=output,
            test_only=False,
            mailbox_root=mailbox,
        )
    with pytest.raises(ExperimentContractError, match="command differs"):
        validate_reservation(
            reservation,
            smoke=True,
            output_directory=output,
            test_only=False,
            canonical_argv=["different"],
            mailbox_root=mailbox,
        )
    with pytest.raises(ExperimentContractError, match="stale"):
        validate_reservation(
            reservation,
            smoke=True,
            output_directory=output,
            test_only=False,
            now=datetime.now(UTC) + timedelta(hours=1),
            mailbox_root=mailbox,
        )
    wrong_authorizer = dict(reservation)
    wrong_authorizer["required_authorizer"] = "astra"
    with pytest.raises(ExperimentContractError, match=r"authorizer|acceptance"):
        validate_reservation(
            wrong_authorizer,
            smoke=True,
            output_directory=output,
            test_only=False,
            mailbox_root=mailbox,
        )
    unrecognized = dict(reservation)
    unrecognized["acceptance_message"] = {
        "path": "messages/20260906T000003.000000Z-" + "3" * 32 + ".json",
        "sha256": "3" * 64,
    }
    with pytest.raises(ExperimentContractError, match="unavailable"):
        validate_reservation(
            unrecognized,
            smoke=True,
            output_directory=output,
            test_only=False,
            mailbox_root=mailbox,
        )
    assert not output.exists()
    bounded = limits_bound_by_reservation(ResourceLimits(), reservation)
    assert bounded.per_seed_wall_seconds == 20 * 60
    assert bounded.cohort_wall_seconds == 20 * 60
    assert bounded.job_wall_seconds == 20 * 60


def test_invalid_production_reservation_precedes_output_and_worker_spawn(
    tmp_path: Path,
    preflight: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "never-created"

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("output creation or worker spawn preceded reservation admission")

    monkeypatch.setattr(supervision_module, "_fresh_output", forbidden)
    monkeypatch.setattr(supervision_module, "_spawn_worker", forbidden)
    with pytest.raises(ExperimentContractError, match="missing or not accepted"):
        supervise_training_job(
            preflight=preflight,
            output_directory=output,
            seeds=(121901,),
            transitions=196_608,
            smoke=True,
            reservation={},
            limits=ResourceLimits(),
            runtime_kind="real",
            test_only=False,
            canonical_argv=[
                "python",
                "-m",
                "oracle_composition.harness.cycle_cli",
                "train",
            ],
        )
    assert not output.exists()
