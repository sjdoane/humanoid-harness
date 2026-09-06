from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/run_gmt_probe.py"
SPEC = importlib.util.spec_from_file_location("run_gmt_probe_test_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROBE
SPEC.loader.exec_module(PROBE)


class FakeProcess:
    pid = 4242

    def __init__(self) -> None:
        self.returncode: int | None = None
        self._polls = 0

    def poll(self) -> int | None:
        self._polls += 1
        if self._polls >= 2:
            self.returncode = 0
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


def _plan(tmp_path: Path) -> Any:
    config = PROBE.ProbeConfig(
        repository_root=tmp_path / "repo",
        venv_python=tmp_path / ".venv/bin/python",
        upstream_root=tmp_path / "upstream",
        weights_path=tmp_path / "weights.npz",
        weights_sha256=PROBE.OFFICIAL_ACTOR_SHA256,
        motion_path=tmp_path / "walk_stand.npz",
        motion_sha256="a" * 64,
        motion_name="walk_stand",
        output_directory=tmp_path / "output",
        owner="astra-gmt-baseline",
    )
    inputs = {
        "artifact_contract": "gmt_reconstructed_actor_baseline_no_learning/v1",
        "duration_seconds": 10,
        "motion": {"name": "walk_stand", "path": str(config.motion_path), "sha256": "a" * 64},
        "repository_sources": {"source.py": "b" * 64},
        "support_files": {"model.xml": "c" * 64},
        "upstream_commit": PROBE.GMT_UPSTREAM_COMMIT,
        "venv": {"path": str(config.venv_python)},
        "weights": {"path": str(config.weights_path), "sha256": config.weights_sha256},
    }
    return PROBE.ProbePlan(
        config=config,
        commit="d" * 40,
        canonical_argv=(str(config.venv_python), "-m", "trusted.replay"),
        inputs=inputs,
    )


def _reservation(plan: Any) -> dict[str, object]:
    return {
        **plan.accepted_fields(),
        "accepted": True,
        "accepted_until_utc": "2099-01-01T00:00:00.000000Z",
        "acceptance_message": {"path": "messages/id.json", "sha256": "e" * 64},
        "acknowledgment": {"path": "acks/id.json", "sha256": "f" * 64},
        "proposal_id": "proposal",
        "required_authorizer": "fable",
        "schema_version": 2,
    }


def _development_plan(tmp_path: Path) -> Any:
    return replace(
        _plan(tmp_path),
        limits=PROBE.ProbeLimits(
            wall_seconds=5,
            cpu_seconds=9,
            rss_bytes=2_048,
            free_disk_bytes=PROBE.MIN_FREE_DISK_BYTES,
            output_bytes=1 * 1024**2,
        ),
        artifact_label="g1_gym_development_pilot_resource_receipt",
        evidence_class="development_single_process_not_task_success",
    )


def test_real_mujoco_import_under_child_environment_and_limits(tmp_path: Path) -> None:
    pytest.importorskip("mujoco")
    plan = _plan(tmp_path)
    plan = replace(
        plan,
        config=replace(
            plan.config,
            repository_root=SCRIPT.parents[1],
            venv_python=Path(sys.executable),
        ),
    )
    environment = PROBE._child_environment(plan)
    assert environment["PATH"].split(":") == [
        str(Path(sys.executable).parent),
        "/usr/bin",
        "/bin",
        "/usr/sbin",
    ]
    result = subprocess.run(
        [sys.executable, "-c", "import mujoco; print(mujoco.__version__)"],
        env=environment,
        cwd=plan.config.repository_root,
        preexec_fn=PROBE._apply_child_limits,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def _write_fake_success(plan: Any, stdout: Any, stderr: Any) -> None:
    del stderr
    output = plan.config.output_directory
    trace = output / PROBE.TRACE_FILENAME
    trace.write_bytes(b"small numeric trace fixture")
    trace_sha256 = hashlib.sha256(trace.read_bytes()).hexdigest()
    manifest_path = trace.with_suffix(".npz.manifest.json")
    manifest = {
        "artifact": "gmt_g1_reconstructed_actor_headless_replay",
        "claim_status": "reconstructed_plain_actor_not_jit_equivalent",
        "inputs": {
            "upstream_commit": PROBE.GMT_UPSTREAM_COMMIT,
            "weights_sha256": plan.config.weights_sha256,
            "motion_name": plan.config.motion_name,
            "motion_sha256": plan.config.motion_sha256,
            "support_files": plan.inputs["support_files"],
        },
        "limits": {"original_jit_executed": False, "jit_equivalence_tested": False},
        "metrics": {"simulated_seconds": 10},
        "trace": {
            "path": trace.name,
            "sha256": trace_sha256,
            "size": trace.stat().st_size,
        },
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    stdout.write(
        json.dumps(
            {
                "trace_path": str(trace),
                "trace_sha256": trace_sha256,
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_sha256,
                "claim_status": "reconstructed_plain_actor_not_jit_equivalent",
            }
        ).encode()
    )
    stdout.flush()


def _dependencies(plan: Any, events: list[str], *, cleanup_fails: bool = False) -> Any:
    token = {"owner": plan.config.owner, "token_id": "1" * 32}

    def reserve(*args: Any, **kwargs: Any) -> dict[str, object]:
        del args, kwargs
        events.append("reserve")
        return token

    def validate_slot(*args: Any, **kwargs: Any) -> dict[str, object]:
        del args, kwargs
        events.append("validate_slot")
        return token

    def validate_plan(value: Any) -> None:
        assert value is plan
        events.append("validate_plan")

    def spawn(*args: Any, **kwargs: Any) -> FakeProcess:
        del args
        events.append("spawn")
        _write_fake_success(plan, kwargs["stdout"], kwargs["stderr"])
        return FakeProcess()

    def cleanup(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        events.append("cleanup")
        if cleanup_fails:
            raise RuntimeError("controlled cleanup failure")

    def release(*args: Any, **kwargs: Any) -> dict[str, object]:
        del args, kwargs
        events.append("release")
        return token

    return PROBE.ProbeDependencies(
        clock=lambda: 1.0,
        sleep=lambda _: None,
        rss_bytes=lambda _: 1024,
        free_disk_bytes=lambda _: 30 * 1024**3,
        directory_bytes=lambda path: sum(item.stat().st_size for item in path.iterdir()),
        spawn=spawn,
        cleanup=cleanup,
        reserve=reserve,
        validate_slot=validate_slot,
        release=release,
        validate_plan=validate_plan,
        validate_group=lambda _: True,
    )


def test_success_releases_exact_slot_only_after_verified_cleanup(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    events: list[str] = []

    result = PROBE.supervise_probe(
        plan,
        coordination_root=tmp_path,
        reservation=_reservation(plan),
        reservation_path=tmp_path / "reservation.json",
        dependencies=_dependencies(plan, events),
    )

    assert result["ok"] is True
    assert events[:4] == ["reserve", "validate_slot", "validate_plan", "spawn"]
    assert events.index("cleanup") < events.index("release")
    assert events[-1] == "release"
    receipt = json.loads((plan.config.output_directory / PROBE.RECEIPT_FILENAME).read_text())
    assert receipt["status"] == "succeeded"
    assert receipt["slot"] == {
        "owner": plan.config.owner,
        "released": True,
        "retained": False,
        "token_id": "1" * 32,
    }
    assert receipt["cleanup"] == {
        "process_group_validated": True,
        "verified_before_release": True,
    }
    assert receipt["resources"]["effective_outer_wall_seconds"] == 120
    assert receipt["resources"]["accepted_schema_wall_seconds"] == 1_200


def test_baseline_plan_keeps_exact_historical_defaults(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    assert plan.limits == PROBE.ProbeLimits()
    assert plan.limits.as_dict() == {
        "wall_seconds": 120,
        "cpu_seconds": 120,
        "rss_bytes": 8 * 1024**3,
        "free_disk_bytes": 20 * 1024**3,
        "output_bytes": 128 * 1024**2,
        "child_output_bytes": 128 * 1024**2 - 512 * 1024,
    }
    accepted = plan.accepted_fields()
    assert accepted["inputs"] is plan.inputs
    assert "process_supervision" not in accepted["inputs"]
    assert accepted["hard_wall_seconds"] == 1_200
    assert accepted["mode"] == "smoke"


def test_development_limits_and_labels_are_bound_into_reservation(tmp_path: Path) -> None:
    plan = _development_plan(tmp_path)
    supervision = plan.accepted_fields()["inputs"]["process_supervision"]

    assert supervision == {
        "artifact_label": "g1_gym_development_pilot_resource_receipt",
        "evidence_class": "development_single_process_not_task_success",
        "limits": plan.limits.as_dict(),
    }
    altered = copy.deepcopy(_reservation(plan))
    altered["inputs"]["process_supervision"]["limits"]["wall_seconds"] = 6
    with pytest.raises(PROBE.ProbeError, match="exact GMT probe plan"):
        PROBE.validate_reservation_binding(plan, altered)


def test_nondefault_limits_cannot_keep_baseline_evidence_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nonbaseline evidence labels"):
        replace(_plan(tmp_path), limits=PROBE.ProbeLimits(wall_seconds=121))


def test_development_ceiling_allows_exact_reviewed_maximum() -> None:
    limits = PROBE.ProbeLimits(
        wall_seconds=1_200,
        cpu_seconds=1_200,
        rss_bytes=8 * 1024**3,
        free_disk_bytes=20 * 1024**3,
        output_bytes=1 * 1024**3,
    )

    assert limits.wall_seconds == 1_200
    assert limits.cpu_seconds == 1_200
    assert limits.output_bytes == 1 * 1024**3


@pytest.mark.parametrize(
    "changes",
    [
        {"wall_seconds": 1_201},
        {"cpu_seconds": 1_201},
        {"rss_bytes": 8 * 1024**3 + 1},
        {"free_disk_bytes": 20 * 1024**3 - 1},
        {"output_bytes": 1 * 1024**3 + 1},
    ],
)
def test_development_limits_reject_unreviewed_relaxations(changes: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="probe"):
        PROBE.ProbeLimits(**changes)


def test_nondefault_cpu_and_file_limits_reach_child_rlimits(monkeypatch: Any) -> None:
    limits = PROBE.ProbeLimits(
        wall_seconds=5,
        cpu_seconds=9,
        rss_bytes=2_048,
        free_disk_bytes=20 * 1024**3,
        output_bytes=1 * 1024**2,
    )
    observed: list[tuple[int, tuple[int, int]]] = []
    monkeypatch.setattr(
        PROBE.resource,
        "setrlimit",
        lambda key, value: observed.append((key, value)),
    )

    PROBE._apply_child_limits(limits)

    assert observed == [
        (PROBE.resource.RLIMIT_CPU, (9, 9)),
        (PROBE.resource.RLIMIT_FSIZE, (1 * 1024**2, 1 * 1024**2)),
    ]


def test_spawn_closes_over_the_plan_limits(tmp_path: Path, monkeypatch: Any) -> None:
    plan = _development_plan(tmp_path)
    observed: list[Any] = []

    def apply_child_limits(limits: Any) -> None:
        observed.append(limits)

    def spawn(*args: Any, **kwargs: Any) -> FakeProcess:
        del args
        kwargs["preexec_fn"]()
        return FakeProcess()

    monkeypatch.setattr(PROBE, "_apply_child_limits", apply_child_limits)
    dependencies = replace(PROBE.ProbeDependencies(), spawn=spawn)

    PROBE._spawn_child(
        plan,
        stdout=io.BytesIO(),
        stderr=io.BytesIO(),
        dependencies=dependencies,
    )

    assert observed == [plan.limits]


@pytest.mark.parametrize("breach", ["wall", "rss", "output"])
def test_nondefault_supervisor_limits_are_enforced_before_release(
    tmp_path: Path, breach: str
) -> None:
    plan = _development_plan(tmp_path)
    events: list[str] = []
    dependencies = _dependencies(plan, events)
    expected_status = "resource_breach"
    if breach == "wall":
        times = iter((0.0, 0.0, 0.0, 6.0, 6.0))
        dependencies = replace(dependencies, clock=lambda: next(times))
        expected_status = "timeout"
    elif breach == "rss":
        dependencies = replace(dependencies, rss_bytes=lambda _: plan.limits.rss_bytes + 1)
    else:
        dependencies = replace(
            dependencies,
            directory_bytes=lambda _: plan.limits.child_output_bytes + 1,
        )

    result = PROBE.supervise_probe(
        plan,
        coordination_root=tmp_path,
        reservation=_reservation(plan),
        reservation_path=tmp_path / "reservation.json",
        dependencies=dependencies,
    )

    assert result["status"] == expected_status
    assert events.index("cleanup") < events.index("release")
    receipt = json.loads((plan.config.output_directory / PROBE.RECEIPT_FILENAME).read_text())
    assert receipt["resources"]["effective_outer_wall_seconds"] == 5
    assert receipt["resources"]["cpu_time_seconds"] == 9
    assert receipt["resources"]["rss_bytes"] == 2_048
    assert receipt["resources"]["output_bytes"] == 1 * 1024**2


def test_trusted_terminal_verifier_is_injectable_but_not_cli_data(tmp_path: Path) -> None:
    plan = _development_plan(tmp_path)
    events: list[str] = []
    dependencies = _dependencies(plan, events)

    def verify_completed(value: Any) -> dict[str, object]:
        assert value is plan
        events.append("verify_completed")
        return {"artifact": "strict_development_fixture"}

    dependencies = replace(dependencies, verify_completed=verify_completed)
    result = PROBE.supervise_probe(
        plan,
        coordination_root=tmp_path,
        reservation=_reservation(plan),
        reservation_path=tmp_path / "reservation.json",
        dependencies=dependencies,
    )

    assert result["ok"] is True
    assert events.index("cleanup") < events.index("verify_completed") < events.index("release")
    receipt = json.loads((plan.config.output_directory / PROBE.RECEIPT_FILENAME).read_text())
    assert receipt["artifact"] == plan.artifact_label
    assert receipt["evidence_class"] == plan.evidence_class
    assert receipt["artifacts"] == {"artifact": "strict_development_fixture"}


def test_cleanup_failure_retains_slot_and_records_terminal_receipt(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    events: list[str] = []

    result = PROBE.supervise_probe(
        plan,
        coordination_root=tmp_path,
        reservation=_reservation(plan),
        reservation_path=tmp_path / "reservation.json",
        dependencies=_dependencies(plan, events, cleanup_fails=True),
    )

    assert result["status"] == "cleanup_failure"
    assert result["slot"] == {"released": False, "retained": True}
    assert "release" not in events
    receipt = json.loads((plan.config.output_directory / PROBE.RECEIPT_FILENAME).read_text())
    assert receipt["cleanup"]["verified_before_release"] is False
    assert receipt["slot"]["retained"] is True


def test_reservation_binding_rejects_argv_or_input_drift(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    reservation = _reservation(plan)
    reservation["canonical_argv"] = ["different", "command"]
    with pytest.raises(PROBE.ProbeError, match="exact GMT probe plan"):
        PROBE.validate_reservation_binding(plan, reservation)

    reservation = _reservation(plan)
    reservation["inputs"] = {"weights": "different"}
    with pytest.raises(PROBE.ProbeError, match="exact GMT probe plan"):
        PROBE.validate_reservation_binding(plan, reservation)


@pytest.mark.parametrize("breach", ["timeout", "rss", "rss_unavailable", "output"])
def test_resource_breach_stops_and_cleans_before_release(tmp_path: Path, breach: str) -> None:
    plan = _plan(tmp_path)
    events: list[str] = []
    dependencies = _dependencies(plan, events)
    expected_status = "resource_breach"
    if breach == "timeout":
        times = iter((0.0, 0.0, 0.0, 121.0, 121.0))
        dependencies = replace(dependencies, clock=lambda: next(times))
        expected_status = "timeout"
    elif breach == "rss":
        dependencies = replace(dependencies, rss_bytes=lambda _: PROBE.RSS_BYTES + 1)
    elif breach == "rss_unavailable":
        dependencies = replace(dependencies, rss_bytes=lambda _: None)
    else:
        dependencies = replace(dependencies, directory_bytes=lambda _: PROBE.CHILD_OUTPUT_BYTES + 1)

    result = PROBE.supervise_probe(
        plan,
        coordination_root=tmp_path,
        reservation=_reservation(plan),
        reservation_path=tmp_path / "reservation.json",
        dependencies=dependencies,
    )

    assert result["status"] == expected_status
    assert result["ok"] is False
    assert events.index("cleanup") < events.index("release")
    assert result["slot"] == {"released": True, "retained": False}
