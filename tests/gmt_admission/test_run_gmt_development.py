from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).parents[2] / "scripts/run_gmt_development.py"
SCRIPT_DIR = str(SCRIPT.parent)
sys.path.insert(0, SCRIPT_DIR)
SPEC = importlib.util.spec_from_file_location("run_gmt_development_test_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DEVELOPMENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DEVELOPMENT
SPEC.loader.exec_module(DEVELOPMENT)
sys.path.remove(SCRIPT_DIR)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> str:
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8")
    return _digest(path)


def _loaded(tmp_path: Path, mode: str = "probe") -> Any:
    config_path = tmp_path / "config.json"
    raw = {
        "mode": mode,
        "training_steps": 0 if mode == "probe" else 512,
        "task": {"horizon_steps": 50},
    }
    config_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return DEVELOPMENT._LoadedWorkload(
        raw=raw,
        sha256=_digest(config_path),
        assets={"fixed": True},
        primary_motion_name="walk_stand",
        primary_motion_path=tmp_path / "motion.npz",
        primary_motion_sha256="a" * 64,
        upstream_root=tmp_path / "upstream",
        weights_path=tmp_path / "weights.npz",
        weights_sha256=DEVELOPMENT.supervisor.OFFICIAL_ACTOR_SHA256,
        course_mode=mode,
        course_identities={
            "task": "1" * 64,
            "oracle": "2" * 64,
            "reward": "3" * 64,
            "segments": {"walk": "4" * 64},
        },
    )


def _plan(tmp_path: Path, workload: str = "course", mode: str = "probe") -> tuple[Any, Any]:
    loaded = _loaded(tmp_path, mode)
    output = tmp_path / "output"
    output.mkdir()
    config_path = tmp_path / "config.json"
    config = DEVELOPMENT.supervisor.ProbeConfig(
        repository_root=tmp_path / "repo",
        venv_python=tmp_path / ".venv/bin/python",
        upstream_root=loaded.upstream_root,
        weights_path=loaded.weights_path,
        weights_sha256=loaded.weights_sha256,
        motion_path=loaded.primary_motion_path,
        motion_sha256=loaded.primary_motion_sha256,
        motion_name=loaded.primary_motion_name,
        output_directory=output,
        owner="astra-development",
    )
    limits, artifact_label, evidence_class = DEVELOPMENT._profile(
        workload, loaded.course_mode
    )
    plan = DEVELOPMENT.supervisor.ProbePlan(
        config=config,
        commit="d" * 40,
        canonical_argv=(str(config.venv_python), "fixed-child"),
        inputs={
            "artifact_contract": "gmt_g1_fixed_development_launcher/v1",
            "workload": workload,
            "course_mode": loaded.course_mode,
            "config": {
                "path": str(config_path),
                "sha256": loaded.sha256,
                "size": config_path.stat().st_size,
            },
            "repository_sources": {"source.py": "b" * 64},
            "venv": {"path": str(config.venv_python)},
            "gmt_assets": loaded.assets,
        },
        limits=limits,
        artifact_label=artifact_label,
        evidence_class=evidence_class,
    )
    return plan, loaded


def _summary(*, residual_rms: float) -> dict[str, object]:
    return {
        "objective_evaluation": {"status": "measured"},
        "training_reward_sum_not_success_metric": 1.5,
        "reset": {"seed": 7},
        "steps": 50,
        "residual_rms": residual_rms,
    }


def _write_course_result(plan: Any, loaded: Any) -> dict[str, object]:
    output = plan.config.output_directory
    (output / DEVELOPMENT.supervisor.STDOUT_FILENAME).write_text("{}\n", encoding="utf-8")
    (output / DEVELOPMENT.supervisor.STDERR_FILENAME).write_text("", encoding="utf-8")
    expected = (
        DEVELOPMENT.COURSE_PROBE_OUTPUTS
        if loaded.course_mode == "probe"
        else DEVELOPMENT.COURSE_TRAIN_OUTPUTS
    )
    outputs = {}
    for name in sorted(expected):
        path = output / name
        if name == "input_config.json":
            path.write_bytes((Path(plan.inputs["config"]["path"])).read_bytes())
        else:
            path.write_bytes(f"fixture:{name}".encode())
        outputs[name] = _digest(path)
    training = None
    final_policy = None
    if loaded.course_mode == "train":
        training = {
            "completed_transitions": 512,
            "episodes": 2,
            "falls": 1,
            "policy_artifact": "numeric_weights_not_optimizer_resume",
            "frozen_base_state_before_sha256": "5" * 64,
            "frozen_base_state_after_sha256": "5" * 64,
        }
        final_policy = _summary(residual_rms=0.02)
    manifest = {
        "schema_version": 1,
        "artifact": "gmt_g1_course_development_run",
        "status": "completed",
        "input_config_sha256": loaded.sha256,
        "outputs": outputs,
        "identities": loaded.course_identities,
        "frozen_runtime": {"gym": "fixed"},
        "training": training,
        "zero_residual": _summary(residual_rms=0.0),
        "final_policy": final_policy,
        "runtime": {"wall_seconds": 1.0},
        "claims": {
            "development_only": True,
            "heldout_generalization_tested": False,
            "physical_obstacle_scene": False,
            "training_performed": loaded.course_mode == "train",
            "full_llm_revision_loop_demonstrated": False,
        },
    }
    _write_json(output / DEVELOPMENT.COURSE_MANIFEST_FILENAME, manifest)
    return manifest


@pytest.mark.parametrize("mode", ["probe", "train"])
def test_course_verifier_accepts_only_mode_bound_complete_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    plan, loaded = _plan(tmp_path, mode=mode)
    _write_course_result(plan, loaded)
    monkeypatch.setattr(DEVELOPMENT, "_load_workload", lambda *_: loaded)

    artifacts = DEVELOPMENT.verify_development_completed(plan)

    assert set(artifacts["outputs"]) == (
        DEVELOPMENT.COURSE_PROBE_OUTPUTS
        if mode == "probe"
        else DEVELOPMENT.COURSE_TRAIN_OUTPUTS
    )


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("extra", "unknown or missing"),
        ("hash", "output hash differs"),
        ("status", "identity or config binding"),
        ("budget", "fixed budget"),
        ("config", "retained course config differs"),
    ],
)
def test_course_verifier_rejects_unbound_or_incomplete_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    error: str,
) -> None:
    plan, loaded = _plan(tmp_path, mode="train")
    manifest = _write_course_result(plan, loaded)
    output = plan.config.output_directory
    if mutation == "extra":
        (output / "unreported.bin").write_bytes(b"untrusted")
    elif mutation == "hash":
        (output / "final_residual_policy.npz").write_bytes(b"changed")
    elif mutation == "status":
        manifest["status"] = "almost"
        _write_json(output / DEVELOPMENT.COURSE_MANIFEST_FILENAME, manifest)
    elif mutation == "budget":
        manifest["training"]["completed_transitions"] = 511
        _write_json(output / DEVELOPMENT.COURSE_MANIFEST_FILENAME, manifest)
    else:
        (output / "input_config.json").write_bytes(b"{}")
        manifest["outputs"]["input_config.json"] = _digest(output / "input_config.json")
        _write_json(output / DEVELOPMENT.COURSE_MANIFEST_FILENAME, manifest)
    monkeypatch.setattr(DEVELOPMENT, "_load_workload", lambda *_: loaded)

    with pytest.raises(DEVELOPMENT.supervisor.ProbeError, match=error):
        DEVELOPMENT.verify_development_completed(plan)


def _parity_receipt(inputs: dict[str, object]) -> dict[str, object]:
    comparison = {
        "chunks_compared": 1,
        "values_compared": 2,
        "exact_chunks": 1,
        "exact_values": 2,
        "max_abs_difference": 0.0,
    }
    return {
        "schema_version": 1,
        "artifact": "gmt_g1_shared_runtime_zero_residual_parity",
        "claim_status": "shared_runtime_exact_parity_to_retained_reconstructed_baseline",
        "inputs": inputs,
        "comparison_contract": {"numeric_tolerance": 0},
        "measurement": {
            "control_steps": 500,
            "simulation_steps": 10_000,
            "matched_runtime_poststate_samples": 9_999,
            "unavailable_baseline_final_poststate_samples": 1,
            "comparisons": {
                name: copy.deepcopy(comparison) for name in DEVELOPMENT.PARITY_COMPARISONS
            },
        },
        "runtime": {"python": "fixture"},
        "limits": {
            "original_jit_executed": False,
            "jit_equivalence_tested": False,
            "runtime_parity_only": True,
            "task_competence_tested": False,
            "reference_composition_tested": False,
            "learning_performed": False,
        },
    }


def test_parity_verifier_binds_exact_zero_delta_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, loaded = _plan(tmp_path, workload="parity")
    loaded = replace(loaded, course_mode=None, parity_receipt_inputs={"trace": "bound"})
    receipt = _parity_receipt(loaded.parity_receipt_inputs)
    output = plan.config.output_directory
    receipt_path = output / DEVELOPMENT.PARITY_FILENAME
    receipt_sha = _write_json(receipt_path, receipt)
    _write_json(
        output / DEVELOPMENT.supervisor.STDOUT_FILENAME,
        {
            "artifact_path": str(receipt_path),
            "artifact_sha256": receipt_sha,
            "claim_status": receipt["claim_status"],
            "measurement": receipt["measurement"],
        },
    )
    (output / DEVELOPMENT.supervisor.STDERR_FILENAME).write_bytes(b"")
    monkeypatch.setattr(DEVELOPMENT, "_load_workload", lambda *_: loaded)

    DEVELOPMENT.verify_development_completed(plan)
    receipt["measurement"]["comparisons"]["prepared_obs"]["max_abs_difference"] = 1e-9
    _write_json(receipt_path, receipt)

    with pytest.raises(DEVELOPMENT.supervisor.ProbeError, match="not exact"):
        DEVELOPMENT.verify_development_completed(plan)


def test_profiles_and_commands_are_fixed(tmp_path: Path) -> None:
    request = DEVELOPMENT.DevelopmentRequest(
        workload="course",
        repository_root=tmp_path / "repo",
        venv_python=tmp_path / ".venv/bin/python",
        config_path=tmp_path / "config.json",
        output_directory=tmp_path / "output",
        owner="owner",
    )
    course_argv = DEVELOPMENT._child_argv(request, request.repository_root)
    parity_argv = DEVELOPMENT._child_argv(replace(request, workload="parity"), request.repository_root)

    assert course_argv == (
        str(request.venv_python),
        "-m",
        "oracle_composition.adapters.gmt.course_run",
        "--config",
        str(request.config_path),
        "--output",
        str(request.output_directory),
    )
    assert "parity-child" in parity_argv
    assert DEVELOPMENT._profile("parity", None)[0].wall_seconds == 120
    assert DEVELOPMENT._profile("course", "train")[0].wall_seconds == 1_200


def test_nondefault_limits_are_part_of_the_accepted_reservation(tmp_path: Path) -> None:
    plan, _ = _plan(tmp_path)
    accepted = plan.accepted_fields()
    assert accepted["inputs"]["process_supervision"]["limits"] == plan.limits.as_dict()
    reservation = {
        **accepted,
        "accepted": True,
        "accepted_until_utc": "2099-01-01T00:00:00Z",
        "schema_version": 2,
    }
    DEVELOPMENT.supervisor.validate_reservation_binding(plan, reservation)
    reservation["inputs"]["process_supervision"]["limits"]["wall_seconds"] = 121
    with pytest.raises(DEVELOPMENT.supervisor.ProbeError, match="exact GMT probe plan"):
        DEVELOPMENT.supervisor.validate_reservation_binding(plan, reservation)


def test_repository_source_set_binds_all_tracked_oracle_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [
        "src/oracle_composition/a.py",
        "src/oracle_composition/nested/b.py",
        "src/oracle_composition/ignored.txt",
        *DEVELOPMENT.LAUNCHER_SOURCES,
        "uv.lock",
    ]
    for relative in paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")

    def fake_git(root: Path, *arguments: str) -> str:
        assert root == tmp_path
        if arguments == ("rev-parse", "--show-toplevel"):
            return str(tmp_path)
        if arguments == ("rev-parse", "HEAD"):
            return "c" * 40
        if arguments[:2] == ("status", "--porcelain=v1"):
            return ""
        if arguments[:2] == ("ls-files", "-z"):
            return "\x00".join(paths)
        raise AssertionError(arguments)

    monkeypatch.setattr(DEVELOPMENT.supervisor, "_run_git", fake_git)
    _, _, sources = DEVELOPMENT._repository_sources(tmp_path)

    assert set(sources) == {
        "src/oracle_composition/a.py",
        "src/oracle_composition/nested/b.py",
        *DEVELOPMENT.LAUNCHER_SOURCES,
        "uv.lock",
    }
