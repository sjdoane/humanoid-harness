from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oracle_composition.adapters.gmt import study_scoring as module
from oracle_composition.adapters.gmt.course_evaluation import evaluate_episode
from oracle_composition.adapters.gmt.course_runtime import (
    COURSE_RESIDUAL_RAW_SCALE,
    LEGACY_RUNTIME,
    frozen_runtime_contract,
)
from oracle_composition.adapters.gmt.training_contract import (
    TRAINING_REWARD_SCALE,
    CourseTrainerSpec,
    effective_training_contract,
)
from oracle_composition.adapters.gmt.training_telemetry import (
    FIXED_NORMALIZER_TELEMETRY_FILENAME,
    SCALED_TELEMETRY_FILENAME,
    TELEMETRY_FILENAME,
    TrainingTelemetry,
    validate_training_telemetry,
)
from oracle_composition.feedback import g1_course
from tests.feedback.test_g1_course_feedback import (
    _fixed_state,
    _run_fixture,
    _write_frames,
    _write_json,
)

SOURCE_COMMIT = "c" * 40
BASE_STATE = "b" * 64
TREE_SHA256 = "d" * 64
TRAINING_STEPS = 8_192


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _replace_budget_and_telemetry(
    run: Path, config: SimpleNamespace, *, scaled: bool
) -> SimpleNamespace:
    raw = dict(config.raw)
    raw["training_steps"] = TRAINING_STEPS
    encoded = (json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n").encode()
    updated = SimpleNamespace(**vars(config))
    updated.raw = raw
    updated.encoded = encoded
    updated.sha256 = hashlib.sha256(encoded).hexdigest()
    (run / "input_config.json").write_bytes(encoded)

    filename = SCALED_TELEMETRY_FILENAME if scaled else TELEMETRY_FILENAME
    telemetry_path = run / filename
    telemetry_path.unlink()
    reward_scale = TRAINING_REWARD_SCALE if scaled else None
    with TrainingTelemetry(telemetry_path, reward_scale=reward_scale) as telemetry:
        for index in range(1, 17):
            logger = (
                {}
                if index == 1
                else {
                    "train/explained_variance": 0.40 + 0.02 * (index - 2),
                    "train/n_updates": 4 * (index - 1),
                }
            )
            telemetry.rollout_boundary(
                index * 512,
                np.zeros((1, 2171), dtype=np.float32),
                logger,
                4 * (index - 1),
            )
        telemetry.final_update({"train/explained_variance": 0.70, "train/n_updates": 64}, 64)
        descriptor = telemetry.descriptor(TRAINING_STEPS)

    manifest_path = run / "course_run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["input_config_sha256"] = updated.sha256
    manifest["outputs"]["input_config.json"] = updated.sha256
    manifest["outputs"][filename] = _sha(telemetry_path)
    manifest["training"].update(
        {
            "completed_transitions": TRAINING_STEPS,
            "episodes": 0,
            "falls": 0,
            "policy_artifact": "numeric_weights_not_optimizer_resume",
            "frozen_base_state_before_sha256": BASE_STATE,
            "frozen_base_state_after_sha256": BASE_STATE,
            "telemetry": descriptor,
        }
    )
    manifest["frozen_runtime"] = frozen_runtime_contract(
        LEGACY_RUNTIME,
        trainer=effective_training_contract(updated.trainer),
        residual_raw_scale=COURSE_RESIDUAL_RAW_SCALE,
    )

    frames_path = run / "final_policy_frames.jsonl"
    frames = [json.loads(line) for line in frames_path.read_text().splitlines()]
    frames[-1]["executed_mode"] = "after"
    manifest["outputs"][frames_path.name] = _write_frames(frames_path, frames)
    evaluation_path = run / "final_policy_evaluation.json"
    evaluation = json.loads(evaluation_path.read_text())
    evaluation["objective_evaluation"] = evaluate_episode(spec=updated.task, frames=frames)
    manifest["final_policy"] = evaluation
    manifest["outputs"][evaluation_path.name] = _write_json(evaluation_path, evaluation)
    _write_json(manifest_path, manifest)
    return updated


def _write_resource_receipt(run: Path, manifest_sha256: str, *, scaled: bool) -> str:
    manifest = json.loads((run / "course_run_manifest.json").read_text())
    config_path = run / "input_config.json"
    inputs = {
        "artifact_contract": "gmt_g1_fixed_development_launcher/v1",
        "config": {
            "path": str(config_path),
            "sha256": manifest["input_config_sha256"],
            "size": config_path.stat().st_size,
        },
        "course_mode": "train",
        "repository_sources": {
            "canonical_tree_sha256": TREE_SHA256,
            "file_count": 10,
        },
        "workload": "course",
    }
    if scaled:
        inputs["trainer"] = effective_training_contract(CourseTrainerSpec(TRAINING_REWARD_SCALE))
    outputs = {
        name: {"path": name, "sha256": digest, "size": (run / name).stat().st_size}
        for name, digest in manifest["outputs"].items()
    }
    receipt = {
        "schema_version": 1,
        "artifact": "gmt_g1_course_development_resource_receipt",
        "status": "succeeded",
        "commit": SOURCE_COMMIT,
        "canonical_argv": [
            "/fixed/python",
            "-m",
            "oracle_composition.adapters.gmt.course_run",
            "--config",
            str(config_path),
            "--output",
            str(run),
        ],
        "inputs": inputs,
        "artifacts": {
            "course_manifest": {
                "path": "course_run_manifest.json",
                "sha256": manifest_sha256,
                "size": (run / "course_run_manifest.json").stat().st_size,
            },
            "outputs": outputs,
        },
    }
    return _write_json(run / module.RESOURCE_RECEIPT_FILENAME, receipt)


def _pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    records = {}
    for cell_id, scaled in (("raw", False), ("scaled", True)):
        run = tmp_path / cell_id
        manifest_path, _, config = _run_fixture(run, monkeypatch, telemetry=True, scaled=scaled)
        config = _replace_budget_and_telemetry(run, config, scaled=scaled)
        manifest_sha256 = _sha(manifest_path)
        receipt_sha256 = _write_resource_receipt(run, manifest_sha256, scaled=scaled)
        manifest = json.loads(manifest_path.read_text())
        records[cell_id] = (
            run,
            config,
            module.RunManifestBinding(manifest_path, manifest_sha256, receipt_sha256),
            module.CourseStudyExpectation(
                cell_id=cell_id,
                seed=7,
                training_steps=TRAINING_STEPS,
                identities=manifest["identities"],
                trainer=config.trainer,
                base_state_sha256=BASE_STATE,
                source_commit=SOURCE_COMMIT,
            ),
        )

    configs = {run.resolve(): config for run, config, _, _ in records.values()}

    def load(path: Path):
        return configs[Path(path).resolve().parent]

    monkeypatch.setattr(module, "load_run_config", load)
    monkeypatch.setattr(g1_course, "load_run_config", load)
    return records


def _score(records):
    _, _, raw, raw_expected = records["raw"]
    _, _, scaled, scaled_expected = records["scaled"]
    return module.score_course_study_pair(
        first=raw,
        first_expected=raw_expected,
        second=scaled,
        second_expected=scaled_expected,
    )


def test_scores_both_cells_with_exact_raw_and_scaled_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _pair(tmp_path, monkeypatch)
    before = {
        name: sorted(path.name for path in record[0].iterdir()) for name, record in records.items()
    }

    result = _score(records)

    assert result["claim_scope"] == "fixed_development_pair_not_heldout_or_universal"
    assert result["pair"] == {
        "seed": 7,
        "training_steps": TRAINING_STEPS,
        "task_sha256": records["raw"][3].identities["task"],
        "base_state_sha256": BASE_STATE,
        "semantic_identity_differences": [],
    }
    assert set(result["cells"]) == {"raw", "scaled"}
    assert result["cells"]["raw"]["effective_trainer"] == effective_training_contract(None)
    assert result["cells"]["scaled"]["effective_trainer"] == effective_training_contract(
        CourseTrainerSpec(TRAINING_REWARD_SCALE)
    )
    for cell in result["cells"].values():
        assert cell["learning"]["update_count"] == 16
        assert cell["learning"]["completed_episode_count"] == 0
        assert cell["learning"]["fall_count"] == 0
        assert cell["learning"]["explained_variance_last16_mean"] == pytest.approx(0.55)
        assert cell["phase_measures"]["actual_task_region"]["sample_count"] == 2
        assert cell["phase_measures"]["executed_after_state"]["sample_count"] == 1
        assert cell["feedback"]["protected_evaluation"] is False
        assert cell["gates"]["development_all_passed"] is False
    after = {
        name: sorted(path.name for path in record[0].iterdir()) for name, record in records.items()
    }
    assert after == before


def test_failed_short_cell_is_retained_and_future_lateral_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _score(_pair(tmp_path, monkeypatch))

    raw = result["cells"]["raw"]
    checkpoints = raw["phase_measures"]["executed_after_state"]["lateral_checkpoints"]
    assert raw["objective"]["development_gate_passed"] is False
    assert checkpoints["10_seconds"] == {
        "control_step": 500,
        "available": False,
        "reason": "trace_ended_before_checkpoint",
        "executed_mode": None,
        "lateral_error_m": None,
    }
    assert checkpoints["20_seconds"]["control_step"] == 1000
    assert checkpoints["20_seconds"]["available"] is False


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("seed", 8, "predeclared study cell"),
        ("training_steps", 16_384, "predeclared study cell"),
        ("base_state_sha256", "e" * 64, "frozen base state"),
        ("source_commit", "f" * 40, "source commit"),
    ],
)
def test_rejects_wrong_predeclared_runtime_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
    match: str,
) -> None:
    records = _pair(tmp_path, monkeypatch)
    run, config, binding, expected = records["scaled"]
    values = {
        "cell_id": expected.cell_id,
        "seed": expected.seed,
        "training_steps": expected.training_steps,
        "identities": expected.identities,
        "trainer": expected.trainer,
        "base_state_sha256": expected.base_state_sha256,
        "source_commit": expected.source_commit,
    }
    values[field] = replacement
    records["scaled"] = (run, config, binding, module.CourseStudyExpectation(**values))
    if field in {"seed", "training_steps", "base_state_sha256"}:
        raw_run, raw_config, raw_binding, raw_expected = records["raw"]
        raw_values = {
            "cell_id": raw_expected.cell_id,
            "seed": raw_expected.seed,
            "training_steps": raw_expected.training_steps,
            "identities": raw_expected.identities,
            "trainer": raw_expected.trainer,
            "base_state_sha256": raw_expected.base_state_sha256,
            "source_commit": raw_expected.source_commit,
        }
        raw_values[field] = replacement
        records["raw"] = (
            raw_run,
            raw_config,
            raw_binding,
            module.CourseStudyExpectation(**raw_values),
        )

    with pytest.raises(ValueError, match=match):
        _score(records)


def test_rejects_wrong_semantic_identity_and_raw_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _pair(tmp_path, monkeypatch)
    run, config, binding, expected = records["raw"]
    identities = dict(expected.identities)
    identities["reward"] = "e" * 64
    records["raw"] = (
        run,
        config,
        binding,
        module.CourseStudyExpectation(
            cell_id="raw",
            seed=7,
            training_steps=TRAINING_STEPS,
            identities=identities,
            trainer=CourseTrainerSpec(TRAINING_REWARD_SCALE),
            base_state_sha256=BASE_STATE,
            source_commit=SOURCE_COMMIT,
        ),
    )

    with pytest.raises(ValueError, match="predeclared study cell"):
        _score(records)


def test_rejects_resource_receipt_and_manifest_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _pair(tmp_path, monkeypatch)
    run, config, binding, expected = records["scaled"]
    receipt_path = run / module.RESOURCE_RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_text())
    receipt["inputs"]["config"]["sha256"] = "e" * 64
    receipt_sha256 = _write_json(receipt_path, receipt)
    records["scaled"] = (
        run,
        config,
        module.RunManifestBinding(binding.path, binding.sha256, receipt_sha256),
        expected,
    )
    with pytest.raises(ValueError, match="bind the retained run"):
        _score(records)

    frames_path = records["raw"][0] / "final_policy_frames.jsonl"
    frames_path.write_bytes(frames_path.read_bytes() + b"{}\n")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _score(records)


def test_rejects_manifest_binding_and_same_commit_source_tree_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _pair(tmp_path, monkeypatch)
    run, config, binding, expected = records["scaled"]
    records["scaled"] = (
        run,
        config,
        module.RunManifestBinding(binding.path, "0" * 64, binding.resource_receipt_sha256),
        expected,
    )
    with pytest.raises(ValueError, match="manifest bytes differ"):
        _score(records)

    receipt_path = run / module.RESOURCE_RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_text())
    receipt["inputs"]["repository_sources"]["canonical_tree_sha256"] = "e" * 64
    receipt_sha256 = _write_json(receipt_path, receipt)
    records["scaled"] = (
        run,
        config,
        module.RunManifestBinding(binding.path, binding.sha256, receipt_sha256),
        expected,
    )
    with pytest.raises(ValueError, match="different source trees"):
        _score(records)


def test_resource_receipt_digest_is_required_and_binds_altered_tree_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _pair(tmp_path, monkeypatch)
    run, _, binding, _ = records["scaled"]
    with pytest.raises(TypeError):
        module.RunManifestBinding(binding.path, binding.sha256)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="resource receipt SHA-256"):
        module.RunManifestBinding(binding.path, binding.sha256, None)  # type: ignore[arg-type]

    receipt_path = run / module.RESOURCE_RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_text())
    receipt["inputs"]["repository_sources"]["canonical_tree_sha256"] = "e" * 64
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="resource receipt bytes differ"):
        _score(records)


def test_rejects_manifest_episode_and_fall_totals_not_in_validated_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _pair(tmp_path, monkeypatch)
    run, config, binding, expected = records["scaled"]
    manifest = json.loads(binding.path.read_text())
    manifest["training"]["episodes"] = 999
    manifest["training"]["falls"] = 999
    manifest_sha256 = _write_json(binding.path, manifest)

    receipt_path = run / module.RESOURCE_RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_text())
    receipt["artifacts"]["course_manifest"].update(
        {
            "sha256": manifest_sha256,
            "size": binding.path.stat().st_size,
        }
    )
    receipt_sha256 = _write_json(receipt_path, receipt)
    records["scaled"] = (
        run,
        config,
        module.RunManifestBinding(binding.path, manifest_sha256, receipt_sha256),
        expected,
    )

    with pytest.raises(ValueError, match="episode/fall totals differ"):
        _score(records)


def test_missing_explained_variance_is_unavailable_not_a_numeric_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _pair(tmp_path, monkeypatch)
    run, _, _, expected = records["raw"]
    telemetry_path = run / TELEMETRY_FILENAME
    rows = [json.loads(line) for line in telemetry_path.read_text().splitlines()]
    final_update = rows[-1]["update"]
    final_update["metrics"]["explained_variance"] = None
    final_update["metric_unavailable_reasons"]["explained_variance"] = "undefined_nonfinite"
    encoded = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows
    )
    telemetry_path.write_bytes(encoded)

    validate_training_telemetry(encoded, TRAINING_STEPS)
    learning = module._updates(
        run,
        {"outputs": {TELEMETRY_FILENAME: hashlib.sha256(encoded).hexdigest()}},
        expected.trainer,
    )

    assert learning["explained_variance_last16_mean"] is None
    assert learning["explained_variance_last16_available"] is False
    assert learning["explained_variance_last16_at_least_0_5"] is False


def test_updates_select_exact_v3_fixed_normalizer_telemetry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _fixed_state(monkeypatch)
    monkeypatch.setattr(module, "FIXED_NORMALIZER_STATE_SHA256", state.sha256)
    path = tmp_path / FIXED_NORMALIZER_TELEMETRY_FILENAME
    with TrainingTelemetry(
        path,
        reward_scale=TRAINING_REWARD_SCALE,
        fixed_normalizer=state,
    ) as telemetry:
        telemetry.rollout_boundary(
            512, np.zeros((1, 2171), dtype=np.float32), {}, None
        )
        telemetry.final_update({}, None)

    learning = module._updates(
        tmp_path,
        {"outputs": {path.name: _sha(path)}},
        CourseTrainerSpec(TRAINING_REWARD_SCALE, profile_version=3),
    )

    assert learning["telemetry_path"] == FIXED_NORMALIZER_TELEMETRY_FILENAME
    assert learning["update_count"] == 1


def test_pair_rejects_mismatched_seed_before_reading_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = _pair(tmp_path, monkeypatch)
    run, config, binding, expected = records["scaled"]
    records["scaled"] = (
        run,
        config,
        binding,
        module.CourseStudyExpectation(
            cell_id="scaled",
            seed=8,
            training_steps=TRAINING_STEPS,
            identities=expected.identities,
            trainer=expected.trainer,
            base_state_sha256=BASE_STATE,
            source_commit=SOURCE_COMMIT,
        ),
    )

    with pytest.raises(ValueError, match="paired study cells differ"):
        _score(records)
