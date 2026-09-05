from __future__ import annotations

import json
from pathlib import Path

import pytest

from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.phase_b.supervision import (
    ResourceLimits,
    SupervisorStatus,
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
    reservation = {
        "accepted": True,
        "authorization": "peer-acceptance-17",
        "command": "python -m oracle_composition.harness.cycle_cli train ...",
        "commit": "1" * 40,
        "conflict_check": "no other heavy repository job",
        "disk_free_preflight": "20GiB",
        "expected_wall": "3m",
        "hard_wall": "20m",
        "inputs": {"oracle_file_sha256": "2" * 64},
        "output": str(output),
        "output_cap": "8GiB",
        "owner": "phase-b-smoke-owner",
        "per_seed_wall": "22m",
        "proposal_id": "proposal-17",
        "rss_hard": "8GiB",
        "schema_version": 1,
        "throughput_floor": "800 steps/s after 65,536 transitions",
        "work": "1 seed; 196,608 counted transitions; 4 DummyVecEnv",
    }
    assert (
        validate_reservation(
            reservation,
            smoke=True,
            output_directory=output,
            test_only=False,
        )
        == reservation
    )
    incomplete = dict(reservation)
    incomplete.pop("authorization")
    with pytest.raises(ExperimentContractError, match="missing or not accepted"):
        validate_reservation(
            incomplete,
            smoke=True,
            output_directory=output,
            test_only=False,
        )
