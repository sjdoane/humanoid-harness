from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from oracle_composition.adapters.gmt.io import write_deterministic_npz, write_json_receipt
from oracle_composition.adapters.gmt.saved_policy_diagnostic_config import (
    ACTION_DIMENSION,
    CONFIG_ARTIFACT,
    HORIZON_STEPS,
    NOISE_FILENAME,
    NOISE_GENERATOR_ID,
    NOISE_SEEDS,
    PARITY_ARTIFACT,
    POLICY_LOG_STD_SHA256,
    POLICY_STATE_DICT_KEY_COUNT,
    POLICY_STATE_SCHEMA_SHA256,
    RETAINED_DETERMINISTIC_OUTPUTS,
    RUN_ARTIFACT,
    RUN_CLAIMS,
    RUN_MANIFEST_FILENAME,
    RUN_RUNTIME,
    diagnostic_config_payload,
    diagnostic_episode_schedule,
    expected_diagnostic_outputs,
    load_saved_policy_diagnostic_config,
    verify_saved_policy_diagnostic_outputs,
)
from oracle_composition.adapters.gmt.training_normalizer import (
    FIXED_NORMALIZER_STATE_SHA256,
)

REPOSITORY = Path(__file__).resolve().parents[2]
PROTOCOL = REPOSITORY / "experiments/020_g1_saved_policy_diagnostic/PROTOCOL.md"
RETAINED = Path(
    "/Users/samueldoane/Documents/ChatGPT/"
    "humanoid-harness-probe-runs/gmt_course_study018_baseline_20260907"
)
SCRIPT = REPOSITORY / "scripts/run_gmt_development.py"


def _development_module() -> Any:
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec = importlib.util.spec_from_file_location("study020_development_test_module", SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPT.parent))


DEVELOPMENT = _development_module()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _retained() -> Path:
    required = {
        RETAINED / "course_run_manifest.json",
        RETAINED / "gmt_probe_resource_receipt_v1.json",
        RETAINED / "input_config.json",
        RETAINED / "initial_residual_policy.npz",
        RETAINED / "final_residual_policy.npz",
        RETAINED / "zero_residual_frames.jsonl",
        RETAINED / "zero_residual_trajectory.npz",
        RETAINED / "zero_residual_evaluation.json",
        RETAINED / "final_policy_frames.jsonl",
        RETAINED / "final_policy_trajectory.npz",
        RETAINED / "final_policy_evaluation.json",
    }
    if not all(path.is_file() for path in required):
        pytest.skip("exact local Study018 retained run is unavailable")
    return RETAINED


def _write_config(tmp_path: Path) -> tuple[Path, Any]:
    noise_path = tmp_path / "source_noise.npz"
    write_deterministic_npz(
        noise_path,
        {
            "seeds": np.asarray(NOISE_SEEDS, dtype="<i8"),
            "standard_normal": np.zeros(
                (len(NOISE_SEEDS), HORIZON_STEPS, ACTION_DIMENSION), dtype="<f4"
            ),
        },
    )
    payload = diagnostic_config_payload(
        retained_root=_retained(), protocol_path=PROTOCOL, noise_path=noise_path
    )
    config_path = tmp_path / "diagnostic.json"
    write_json_receipt(config_path, payload)
    return config_path, load_saved_policy_diagnostic_config(config_path)


def _summary(policy: str) -> dict[str, Any]:
    source = (
        "zero_residual_evaluation.json" if policy == "initial" else "final_policy_evaluation.json"
    )
    return json.loads((_retained() / source).read_text(encoding="utf-8"))


def _complete_output(tmp_path: Path, config: Any) -> Path:
    output = tmp_path / "output"
    output.mkdir()
    (output / "child_stdout.json").write_text("{}\n", encoding="utf-8")
    (output / "child_stderr.log").write_bytes(b"")
    (output / "input_diagnostic_config.json").write_bytes(config.encoded)
    shutil.copyfile(config.course_config.path, output / "input_course_config.json")
    shutil.copyfile(config.noise.path, output / NOISE_FILENAME)

    ledger: dict[str, str] = {
        "input_diagnostic_config.json": config.sha256,
        "input_course_config.json": config.course_config.sha256,
        NOISE_FILENAME: config.noise.sha256,
    }
    episodes = []
    for episode in diagnostic_episode_schedule():
        summary = _summary(episode.policy)
        if episode.kind == "deterministic_parity":
            source_prefix = "zero_residual" if episode.policy == "initial" else "final_policy"
            for suffix, extension in (
                ("frames", ".jsonl"),
                ("trajectory", ".npz"),
                ("evaluation", ".json"),
            ):
                name = f"{episode.label}_{suffix}{extension}"
                shutil.copyfile(_retained() / f"{source_prefix}_{suffix}{extension}", output / name)
                ledger[name] = _sha256(output / name)
        else:
            frames = output / f"{episode.label}_frames.jsonl"
            trajectory = output / f"{episode.label}_trajectory.npz"
            evaluation = output / f"{episode.label}_evaluation.json"
            frames.write_text("sample fixture\n", encoding="utf-8")
            trajectory.write_bytes(b"sample fixture")
            write_json_receipt(evaluation, summary)
            ledger[frames.name] = _sha256(frames)
            ledger[trajectory.name] = _sha256(trajectory)
            ledger[evaluation.name] = _sha256(evaluation)
        sampling = None
        if episode.noise_seed is not None:
            total = summary["steps"] * ACTION_DIMENSION
            sampling = {
                "noise_seed": episode.noise_seed,
                "noise_rows_available": HORIZON_STEPS,
                "noise_rows_consumed": summary["steps"],
                "action_dimension": ACTION_DIMENSION,
                "clipped_component_count": 0,
                "total_component_count": total,
                "max_abs_unclipped_action": 0.0,
            }
        episodes.append(
            {
                "sequence_index": episode.sequence_index,
                "kind": episode.kind,
                "label": episode.label,
                "policy": episode.policy,
                "noise_seed": episode.noise_seed,
                "summary": summary,
                "sampling": sampling,
            }
        )

    parity_bindings = {}
    for policy in ("initial", "final"):
        name = f"{policy}_deterministic_parity.json"
        receipt = {
            "schema_version": 1,
            "artifact": PARITY_ARTIFACT,
            "policy": policy,
            "label": f"{policy}_deterministic",
            "expected": RETAINED_DETERMINISTIC_OUTPUTS[policy],
            "observed": RETAINED_DETERMINISTIC_OUTPUTS[policy],
            "passed": True,
        }
        digest = write_json_receipt(output / name, receipt)
        ledger[name] = digest
        parity_bindings[policy] = {
            "path": name,
            "sha256": digest,
            "size": (output / name).stat().st_size,
        }

    policy_admission = {
        policy: {
            "archive": config.policies[policy].receipt(),
            "state_dict_key_count": POLICY_STATE_DICT_KEY_COUNT,
            "state_dict_schema_sha256": POLICY_STATE_SCHEMA_SHA256,
            "log_std_sha256": POLICY_LOG_STD_SHA256[policy],
            "normalizer_state_sha256": FIXED_NORMALIZER_STATE_SHA256,
            "strict_state_dict_loaded": True,
            "state_dict_readback_exact": True,
        }
        for policy in ("initial", "final")
    }
    assert set(ledger) == set(expected_diagnostic_outputs())
    write_json_receipt(
        output / RUN_MANIFEST_FILENAME,
        {
            "schema_version": 1,
            "artifact": RUN_ARTIFACT,
            "status": "completed",
            "input_config_sha256": config.sha256,
            "inputs": config.resource_binding(),
            "outputs": ledger,
            "policy_admission": policy_admission,
            "deterministic_parity": {
                "sampled_started_after_both_receipts": True,
                "receipts": parity_bindings,
            },
            "episodes": episodes,
            "runtime": RUN_RUNTIME,
            "claims": RUN_CLAIMS,
        },
    )
    return output


def test_exact_config_and_declared_pair_order_are_admitted(tmp_path: Path) -> None:
    _, config = _write_config(tmp_path)
    schedule = diagnostic_episode_schedule()
    assert config.raw["artifact"] == CONFIG_ARTIFACT
    assert config.raw["sampling"]["generator_id"] == NOISE_GENERATOR_ID
    assert len(schedule) == 34
    assert [episode.label for episode in schedule[:4]] == [
        "initial_deterministic",
        "final_deterministic",
        "sample_20260920_initial",
        "sample_20260920_final",
    ]
    assert [episode.label for episode in schedule[-2:]] == [
        "sample_20260935_final",
        "sample_20260935_initial",
    ]


@pytest.mark.parametrize("mutation", ["unknown-field", "noise-hash", "missing-retained"])
def test_config_rejects_unknown_tampered_or_missing_inputs(tmp_path: Path, mutation: str) -> None:
    config_path, config = _write_config(tmp_path)
    raw = copy.deepcopy(config.raw)
    if mutation == "unknown-field":
        raw["extra"] = True
    elif mutation == "noise-hash":
        raw["sampling"]["noise"]["sha256"] = "0" * 64
    else:
        raw["retained_run_root"] = str(tmp_path / "missing")
    changed = tmp_path / f"{mutation}.json"
    write_json_receipt(changed, raw)
    with pytest.raises(ValueError):
        load_saved_policy_diagnostic_config(changed)
    assert config_path.is_file()


def test_output_validator_and_native_terminal_route_share_exact_contract(tmp_path: Path) -> None:
    config_path, config = _write_config(tmp_path)
    output = _complete_output(tmp_path, config)
    verified = verify_saved_policy_diagnostic_outputs(config, output)
    assert verified["manifest"]["path"] == RUN_MANIFEST_FILENAME
    assert set(verified["outputs"]) == set(expected_diagnostic_outputs())

    plan = DEVELOPMENT.supervisor.ProbePlan(
        config=DEVELOPMENT.supervisor.ProbeConfig(
            repository_root=REPOSITORY,
            venv_python=REPOSITORY / ".venv/bin/python",
            upstream_root=Path(config.raw["retained_run_root"]),
            weights_path=Path(config.raw["retained_run_root"]),
            weights_sha256=DEVELOPMENT.supervisor.OFFICIAL_ACTOR_SHA256,
            motion_path=Path(config.raw["retained_run_root"]),
            motion_sha256="a" * 64,
            motion_name="walk_stand",
            output_directory=output,
            owner="astra-study020-test",
        ),
        commit="a" * 40,
        canonical_argv=("python",),
        inputs={
            "workload": "saved-policy-diagnostic",
            "config": {"path": str(config_path), "sha256": config.sha256},
            "saved_policy_diagnostic": config.resource_binding(),
        },
        limits=DEVELOPMENT._profile("saved-policy-diagnostic", "evaluation-only")[0],
        artifact_label="gmt_g1_saved_policy_diagnostic_resource_receipt",
        evidence_class="evaluation_only_paired_saved_policy_diagnostic_not_task_success",
    )
    terminal = DEVELOPMENT._verify_saved_policy_diagnostic(plan)
    assert set(terminal) == {"saved_policy_diagnostic_manifest", "outputs"}

    (output / "gmt_probe_resource_receipt_v1.json").write_text("{}\n", encoding="utf-8")
    assert (
        verify_saved_policy_diagnostic_outputs(config, output)["manifest"] == verified["manifest"]
    )


@pytest.mark.parametrize("mutation", ["failed-parity", "reordered-episode", "extra-file"])
def test_output_validator_rejects_claim_invalidating_evidence_changes(
    tmp_path: Path, mutation: str
) -> None:
    _, config = _write_config(tmp_path)
    output = _complete_output(tmp_path, config)
    manifest_path = output / RUN_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "failed-parity":
        parity_path = output / "initial_deterministic_parity.json"
        parity = json.loads(parity_path.read_text(encoding="utf-8"))
        parity["passed"] = False
        write_json_receipt(parity_path.with_suffix(".replacement"), parity)
        parity_path.write_bytes(parity_path.with_suffix(".replacement").read_bytes())
        parity_path.with_suffix(".replacement").unlink()
        digest = _sha256(parity_path)
        manifest["outputs"][parity_path.name] = digest
        manifest["deterministic_parity"]["receipts"]["initial"].update(
            {"sha256": digest, "size": parity_path.stat().st_size}
        )
    elif mutation == "reordered-episode":
        manifest["episodes"][2], manifest["episodes"][3] = (
            manifest["episodes"][3],
            manifest["episodes"][2],
        )
    else:
        (output / "undeclared.bin").write_bytes(b"unexpected")
    manifest_path.unlink()
    write_json_receipt(manifest_path, manifest)
    with pytest.raises(ValueError):
        verify_saved_policy_diagnostic_outputs(config, output)


def test_plan_is_evaluation_only_and_does_not_import_worker_or_mujoco(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, _ = _write_config(tmp_path)
    output = tmp_path / "fresh-output"
    sys.modules.pop("mujoco", None)
    sys.modules.pop("oracle_composition.adapters.gmt.saved_policy_diagnostic_run", None)
    monkeypatch.setattr(
        DEVELOPMENT,
        "_repository_sources",
        lambda root: (
            REPOSITORY,
            "b" * 40,
            {
                "scripts/run_gmt_probe.py": "1" * 64,
                "scripts/run_gmt_development.py": "2" * 64,
                "uv.lock": "3" * 64,
            },
        ),
    )
    monkeypatch.setattr(
        DEVELOPMENT.supervisor,
        "_venv_identity",
        lambda path: {"path": str(path), "test": True},
    )
    plan = DEVELOPMENT.build_development_plan(
        DEVELOPMENT.DevelopmentRequest(
            workload="saved-policy-diagnostic",
            repository_root=REPOSITORY,
            venv_python=REPOSITORY / ".venv/bin/python",
            config_path=config_path,
            output_directory=output,
            owner="astra-study020-test",
        )
    )
    assert plan.inputs["course_mode"] == "evaluation-only"
    assert plan.inputs["artifact_contract"] == "gmt_g1_saved_policy_diagnostic_launcher/v1"
    assert plan.canonical_argv[1:4] == (
        "-m",
        "oracle_composition.adapters.gmt.saved_policy_diagnostic_run",
        "run",
    )
    assert plan.limits.wall_seconds == plan.limits.cpu_seconds == 1_200
    assert "mujoco" not in sys.modules
    assert "oracle_composition.adapters.gmt.saved_policy_diagnostic_run" not in sys.modules
