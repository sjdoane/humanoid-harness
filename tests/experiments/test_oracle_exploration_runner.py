from __future__ import annotations

import hashlib
import inspect
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces

from oracle_composition.experiments import oracle_exploration_runner as runner_module
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.oracle_exploration_design import (
    EXPECTED_PRE_EXECUTION_BINDINGS,
    load_oracle_exploration_design,
)
from oracle_composition.experiments.oracle_exploration_runner import (
    MANIFEST_CANDIDATE_STATUS,
    MANIFEST_REVIEWED_STATUS,
    REVIEW_CONFIRMATION,
    PreparedOracleExploration,
    compose_controller_action,
    diagnostic_stock_environment_return,
    emit_manifest_candidate,
    finalize_reviewed_manifest,
    inspect_oracle_exploration_runtime,
    load_oracle_exploration_manifest,
    manifest_from_prepared,
    materialize_executed_action,
    run_oracle_exploration,
    validate_prepared_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
DESIGN_PATH = (
    ROOT / "experiments" / "exploratory_phase_oracle_v1" / "configs" / "local_v1.study.json"
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


class _Receipt:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def to_dict(self) -> dict[str, object]:
        return dict(self.payload)


def _prepared() -> PreparedOracleExploration:
    design = load_oracle_exploration_design(DESIGN_PATH)
    design_file_bytes = DESIGN_PATH.read_bytes()
    bindings = {role: _sha(role) for role in EXPECTED_PRE_EXECUTION_BINDINGS}
    bindings["oracle_exploration_design_sha256"] = design.sha256
    return PreparedOracleExploration(
        design_path=DESIGN_PATH,
        design_file_bytes=design_file_bytes,
        design_file_sha256=hashlib.sha256(design_file_bytes).hexdigest(),
        design=design,
        projection=None,  # type: ignore[arg-type]
        base_infer=lambda observation: np.zeros(17),
        residual_infer=lambda observation, window: np.zeros(17),
        base_receipt=None,  # type: ignore[arg-type]
        residual_receipt=None,  # type: ignore[arg-type]
        runtime_receipt=None,  # type: ignore[arg-type]
        bindings=bindings,
    )


def test_manifest_requires_explicit_review_and_exact_current_bindings(tmp_path: Path) -> None:
    prepared = _prepared()
    candidate_path = tmp_path / "manifest.candidate.json"
    reviewed_path = tmp_path / "manifest.reviewed.json"
    candidate = emit_manifest_candidate(candidate_path, prepared)

    assert candidate.manifest_status == MANIFEST_CANDIDATE_STATUS
    assert load_oracle_exploration_manifest(candidate_path).to_dict() == candidate.to_dict()
    with pytest.raises(ExperimentContractError, match="reviewed frozen"):
        validate_prepared_manifest(prepared, candidate)
    with pytest.raises(ExperimentContractError, match="exact human review"):
        finalize_reviewed_manifest(
            candidate_path=candidate_path,
            output_path=reviewed_path,
            review_confirmation="yes",
        )

    reviewed = finalize_reviewed_manifest(
        candidate_path=candidate_path,
        output_path=reviewed_path,
        review_confirmation=REVIEW_CONFIRMATION,
    )
    assert reviewed.manifest_status == MANIFEST_REVIEWED_STATUS
    validate_prepared_manifest(prepared, reviewed)

    drifted = replace(
        prepared,
        bindings={
            **prepared.bindings,
            "runner_source_sha256": "f" * 64,
        },
    )
    with pytest.raises(ExperimentContractError, match="current pre-execution facts"):
        validate_prepared_manifest(drifted, reviewed)
    with pytest.raises(ExperimentContractError, match="refusing to overwrite"):
        emit_manifest_candidate(candidate_path, prepared)


def test_unreviewed_manifest_blocks_before_environment_construction(tmp_path: Path) -> None:
    prepared = _prepared()
    candidate_path = tmp_path / "manifest.candidate.json"
    emit_manifest_candidate(candidate_path, prepared)
    with pytest.raises(ExperimentContractError, match="reviewed frozen"):
        run_oracle_exploration(
            design_path=tmp_path / "must-not-be-read.study.json",
            hdf5_path=tmp_path / "must-not-be-read.hdf5",
            metadata_path=tmp_path / "must-not-be-read.metadata.json",
            base_controller_path=tmp_path / "must-not-be-read.bc.npz",
            residual_controller_path=tmp_path / "must-not-be-read.residual.npz",
            manifest_path=candidate_path,
            runs_dir=tmp_path / "runs",
        )
    assert not (tmp_path / "runs").exists()


def test_action_composition_matches_frozen_raw_and_normalized_convention() -> None:
    design = load_oracle_exploration_design(DESIGN_PATH)
    base = np.linspace(-0.4, 0.4, 17)
    residual = np.linspace(1.0, -1.0, 17)

    physical, normalized = compose_controller_action(
        design=design,
        base_raw_control=base,
        residual_unscaled=residual,
        gate_weight=1.0,
    )
    base_f32 = base.astype("<f4")
    residual_f32 = residual.astype("<f4")
    expected = np.clip(
        np.add(
            base_f32,
            np.multiply(np.float32(0.08), residual_f32, dtype=np.float32),
            dtype=np.float32,
        ),
        np.float32(-0.4),
        np.float32(0.4),
    ).astype("<f4", copy=False)
    np.testing.assert_array_equal(physical, expected)
    np.testing.assert_array_equal(
        normalized,
        np.clip(expected.astype(np.float64) / 0.4, -1.0, 1.0),
    )

    base_only, base_only_normalized = compose_controller_action(
        design=design,
        base_raw_control=base,
        residual_unscaled=residual,
        gate_weight=0.0,
    )
    np.testing.assert_array_equal(base_only, base_f32)
    np.testing.assert_array_equal(
        base_only_normalized,
        np.clip(base_f32.astype(np.float64) / 0.4, -1.0, 1.0),
    )


def test_executed_action_views_derive_from_exact_float32_bytes() -> None:
    design = load_oracle_exploration_design(DESIGN_PATH)
    value = 0.123456789123
    raw = np.full(17, value, dtype=np.float64)
    action_space = spaces.Box(-0.4, 0.4, shape=(17,), dtype=np.float32)

    typed, physical, normalized = materialize_executed_action(
        design=design,
        physical_control=raw,
        action_space=action_space,
    )

    assert typed.dtype == np.float32
    assert physical[0] == float(np.float32(value))
    assert physical[0] != value
    np.testing.assert_array_equal(normalized, np.clip(physical / 0.4, -1.0, 1.0))

    endpoint = np.full(17, np.float32(0.4), dtype=np.float32)
    _, submitted_endpoint, normalized_endpoint = materialize_executed_action(
        design=design,
        physical_control=endpoint,
        action_space=action_space,
    )
    assert submitted_endpoint[0] == 0.4000000059604645
    np.testing.assert_array_equal(normalized_endpoint, np.ones(17))

    float64_space = spaces.Box(
        low=np.full(17, float(np.float32(-0.4)), dtype=np.float64),
        high=np.full(17, float(np.float32(0.4)), dtype=np.float64),
        dtype=np.float64,
    )
    with pytest.raises(ExperimentContractError, match="action space changed"):
        materialize_executed_action(
            design=design,
            physical_control=endpoint,
            action_space=float64_space,
        )


def test_stock_environment_return_is_finite_exact_count_and_diagnostic_only() -> None:
    rule = "python_math_fsum_in_action_order_over_exactly_1000_values/v1"
    rewards = [1e16, 1.0, -1e16] + [0.0] * 997

    assert (
        diagnostic_stock_environment_return(
            rewards,
            expected_count=1000,
            summation_rule=rule,
        )
        == 1.0
    )
    with pytest.raises(ExperimentContractError, match="missing step rewards"):
        diagnostic_stock_environment_return(
            rewards[:-1],
            expected_count=1000,
            summation_rule=rule,
        )
    with pytest.raises(ExperimentContractError, match="must be finite"):
        diagnostic_stock_environment_return(
            [*rewards[:-1], float("nan")],
            expected_count=1000,
            summation_rule=rule,
        )
    with pytest.raises(ExperimentContractError, match="summation rule"):
        diagnostic_stock_environment_return(
            rewards,
            expected_count=1000,
            summation_rule="numpy_sum/v0",
        )


@pytest.mark.parametrize(
    ("base", "residual", "weight", "message"),
    [
        (np.full(17, 0.401), np.zeros(17), 1.0, "base action"),
        (np.zeros(17), np.full(17, 1.001), 1.0, "residual action"),
        (np.zeros(17), np.zeros(17), 0.5, "gate weight"),
        (np.full(17, np.nan), np.zeros(17), 1.0, "finite"),
    ],
)
def test_action_composition_rejects_contract_drift(
    base: np.ndarray,
    residual: np.ndarray,
    weight: float,
    message: str,
) -> None:
    design = load_oracle_exploration_design(DESIGN_PATH)
    with pytest.raises(ExperimentContractError, match=message):
        compose_controller_action(
            design=design,
            base_raw_control=base,
            residual_unscaled=residual,
            gate_weight=weight,
        )


def test_manifest_candidate_binds_canonical_design_and_complete_table() -> None:
    prepared = _prepared()
    manifest = manifest_from_prepared(prepared)

    assert manifest.bindings["oracle_exploration_design_sha256"] == prepared.design.sha256
    assert manifest.expected_run_count == 240
    assert manifest.run_table_sha256 == prepared.design.expected_run_table_sha256


def test_study_lineage_carries_exact_inputs_and_bound_receipts(tmp_path: Path) -> None:
    prepared = _prepared()
    runtime = _Receipt({"receipt": "runtime"})
    projection_receipt = _Receipt({"receipt": "projection"})
    base = _Receipt({"receipt": "base"})
    residual = _Receipt({"receipt": "residual"})
    source_record = (
        ROOT
        / "src"
        / "oracle_composition"
        / "sources"
        / "manifests"
        / "minari_humanoid_expert_v0.json"
    )
    bindings = dict(prepared.bindings)
    for role, payload in (
        ("observed_runtime_model_abi_receipt_sha256", runtime.to_dict()),
        ("registered_import_projection_receipt_sha256", projection_receipt.to_dict()),
        ("base_controller_load_receipt_sha256", base.to_dict()),
        ("reference_residual_load_receipt_sha256", residual.to_dict()),
    ):
        bindings[role] = runner_module._json_sha256(payload)
    bindings["registered_source_record_sha256"] = runner_module.sha256_file(source_record)
    prepared = replace(
        prepared,
        runtime_receipt=runtime,  # type: ignore[arg-type]
        projection=SimpleNamespace(receipt=projection_receipt),  # type: ignore[arg-type]
        base_receipt=base,  # type: ignore[arg-type]
        residual_receipt=residual,  # type: ignore[arg-type]
        bindings=bindings,
    )
    manifest = replace(
        manifest_from_prepared(prepared),
        manifest_status=MANIFEST_REVIEWED_STATUS,
    )
    manifest_bytes = runner_module._pretty_json(manifest.to_dict())
    staging = tmp_path / "study.pending"
    staging.mkdir()

    lineage_path, lineage_sha256 = runner_module._materialize_study_lineage(
        staging=staging,
        prepared=prepared,
        manifest=manifest,
        manifest_file_bytes=manifest_bytes,
        manifest_file_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )

    assert (staging / "lineage" / "design.study.json").read_bytes() == prepared.design_file_bytes
    assert (staging / "lineage" / "execution_manifest.reviewed.json").read_bytes() == manifest_bytes
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    assert lineage_sha256 == runner_module.sha256_file(lineage_path)
    assert set(lineage["files"]) == {
        "design.study.json",
        "execution_manifest.reviewed.json",
        "observed_runtime_model_abi_receipt.json",
        "registered_import_projection_receipt.json",
        "base_controller_load_receipt.json",
        "reference_residual_load_receipt.json",
        "run_table.json",
        "registered_source_record.json",
    }
    for filename, entry in lineage["files"].items():
        payload = (staging / "lineage" / filename).read_bytes()
        assert entry["bytes"] == len(payload)
        assert entry["sha256"] == hashlib.sha256(payload).hexdigest()
    assert lineage["controller_and_source_code_bytes_embedded"] is False


def test_snapshot_reader_rejects_links_and_nonregular_files(tmp_path: Path) -> None:
    regular = tmp_path / "manifest.json"
    regular.write_text('{"ok":true}', encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(regular)
    fifo = tmp_path / "manifest.fifo"
    fifo.parent.mkdir(parents=True, exist_ok=True)
    fifo_path = str(fifo)
    import os

    os.mkfifo(fifo_path)

    source = runner_module.read_bounded_json_artifact(
        regular,
        maximum_bytes=100,
        artifact="test manifest",
    )
    assert source.encoded_bytes == b'{"ok":true}'
    assert source.sha256 == hashlib.sha256(source.encoded_bytes).hexdigest()
    with pytest.raises(ExperimentContractError, match="must not be a symlink"):
        runner_module.read_bounded_json_artifact(
            link,
            maximum_bytes=100,
            artifact="test manifest",
        )
    with pytest.raises(ExperimentContractError, match="regular file"):
        runner_module.read_bounded_json_artifact(
            fifo,
            maximum_bytes=100,
            artifact="test manifest",
        )


def test_atomic_file_publication_rejects_a_final_component_symlink(tmp_path: Path) -> None:
    target = tmp_path / "must-not-be-created.json"
    destination = tmp_path / "receipt.json"
    destination.symlink_to(target)

    with pytest.raises(ExperimentContractError, match="refusing to overwrite"):
        runner_module.emit_bytes_without_overwrite(destination, b'{"safe":true}')

    assert destination.is_symlink()
    assert not target.exists()


def test_canonical_run_api_has_no_prepared_or_environment_injection() -> None:
    parameters = inspect.signature(run_oracle_exploration).parameters

    assert "prepared" not in parameters
    assert "environment_factory" not in parameters
    assert set(parameters) == {
        "design_path",
        "hdf5_path",
        "metadata_path",
        "base_controller_path",
        "residual_controller_path",
        "manifest_path",
        "runs_dir",
    }


def test_perturbation_boundary_does_not_rewrite_the_prior_sample() -> None:
    source = inspect.getsource(runner_module._run_one)

    assert "samples[-1]" not in source
    assert "disturbance=disturbance" in source
    assert source.index("disturbance = _apply_perturbation") < source.index(
        "observation, stock_reward, terminated, truncated"
    )


def test_recovery_transition_events_preserve_entry_and_exit_evidence() -> None:
    entered = runner_module.RecoveryDecision(
        mode=runner_module.RecoveryMode.RECOVER,
        residual_weight=0.0,
        selected_phase=10,
        selected_pose_error=3.0,
        stable_steps=0,
        pose_error_bad_steps=3,
        entered=True,
        exited=False,
        entry_reasons=("height", "pose_error"),
    )
    exited = runner_module.RecoveryDecision(
        mode=runner_module.RecoveryMode.TRACK,
        residual_weight=1.0,
        selected_phase=20,
        selected_pose_error=1.0,
        stable_steps=0,
        pose_error_bad_steps=0,
        entered=False,
        exited=True,
        entry_reasons=(),
    )

    assert (
        runner_module._recovery_transition_value(
            previous_mode=runner_module.RecoveryMode.TRACK,
            decision=entered,
            exit_stable_steps=20,
        )
        == "track->recover;entry_reasons=height,pose_error"
    )
    assert (
        runner_module._recovery_transition_value(
            previous_mode=runner_module.RecoveryMode.RECOVER,
            decision=exited,
            exit_stable_steps=20,
        )
        == "recover->track;exit_reason=all_exit_guards_stable_for_20_steps"
    )
    with pytest.raises(ExperimentContractError, match="lacks guard reasons"):
        runner_module._recovery_transition_value(
            previous_mode=runner_module.RecoveryMode.TRACK,
            decision=replace(entered, entry_reasons=()),
            exit_stable_steps=20,
        )


def test_atomic_publication_never_replaces_a_concurrent_final_directory(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    with (
        pytest.raises(ExperimentContractError, match="refusing to overwrite"),
        runner_module._atomic_study_directory(
            runs_dir,
            final_name="local-v1-race",
        ) as (staging, final),
    ):
        (staging / "retained.txt").write_text("attempt evidence", encoding="utf-8")
        final.mkdir()

    assert (runs_dir / "local-v1-race").is_dir()
    pending = list(runs_dir.glob(".local-v1-race.*.pending"))
    assert len(pending) == 1
    assert (pending[0] / "retained.txt").read_text(encoding="utf-8") == "attempt evidence"


def test_failed_attempt_is_retained_with_completed_and_missing_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    candidate_path = tmp_path / "manifest.candidate.json"
    reviewed_path = tmp_path / "manifest.reviewed.json"
    emit_manifest_candidate(candidate_path, prepared)
    reviewed = finalize_reviewed_manifest(
        candidate_path=candidate_path,
        output_path=reviewed_path,
        review_confirmation=REVIEW_CONFIRMATION,
    )
    monkeypatch.setattr(
        runner_module,
        "_prepare_oracle_exploration_from_design",
        lambda **_kwargs: prepared,
    )

    def fake_lineage(**kwargs: object) -> tuple[Path, str]:
        staging = kwargs["staging"]
        assert isinstance(staging, Path)
        lineage = staging / "lineage"
        lineage.mkdir()
        path = runner_module.emit_json_without_overwrite(
            lineage / "lineage_manifest.json",
            {"test_only": True},
        )
        return path, runner_module.sha256_file(path)

    monkeypatch.setattr(runner_module, "_materialize_study_lineage", fake_lineage)

    def fake_run_one(**kwargs: object) -> dict[str, object]:
        run = kwargs["run"]
        output_dir = kwargs["output_dir"]
        assert hasattr(run, "run_order")
        assert isinstance(output_dir, Path)
        if run.run_order == 2:  # type: ignore[union-attr]
            raise ValueError("sensitive synthetic detail must not be persisted")
        receipt = {"run": run.to_dict()}  # type: ignore[union-attr]
        runner_module.emit_json_without_overwrite(
            output_dir / "run_receipt.json",
            receipt,
        )
        return receipt

    monkeypatch.setattr(runner_module, "_run_one", fake_run_one)
    monkeypatch.setattr(
        runner_module,
        "_validate_completed_run_receipt",
        lambda **_kwargs: None,
    )
    runs_dir = tmp_path / "runs"
    call = {
        "design_path": DESIGN_PATH,
        "hdf5_path": tmp_path / "unused.hdf5",
        "metadata_path": tmp_path / "unused.metadata.json",
        "base_controller_path": tmp_path / "unused.bc.npz",
        "residual_controller_path": tmp_path / "unused.residual.npz",
        "manifest_path": reviewed_path,
        "runs_dir": runs_dir,
    }

    with pytest.raises(ValueError, match="sensitive synthetic detail"):
        run_oracle_exploration(**call)

    failed = list(runs_dir.glob(f"local-v1-{reviewed.sha256[:12]}-failed-*"))
    assert len(failed) == 1
    receipt = json.loads((failed[0] / "failure_receipt.json").read_text(encoding="utf-8"))
    assert receipt["completion_status"] == "failed_attempt_retained_no_comparison"
    assert receipt["completed_run_count"] == 1
    assert receipt["completed_scheduled_runs"] == [prepared.design.run_table[0].to_dict()]
    assert len(receipt["completed_run_receipt_file_sha256"]) == 1
    assert receipt["completed_run_receipt_file_sha256"][0] == runner_module.sha256_file(
        failed[0] / "runs" / "001" / "run_receipt.json"
    )
    assert receipt["failed_scheduled_run"] == prepared.design.run_table[1].to_dict()
    assert receipt["missing_run_count"] == 239
    assert receipt["missing_scheduled_runs"] == [
        run.to_dict() for run in prepared.design.run_table[1:]
    ]
    assert receipt["failure_class"] == "ValueError"
    assert receipt["failure_summary"] == "unexpected_execution_failure"
    declared = prepared.design.to_dict()["hard_gates"]
    assert receipt["declared_hard_gates"] == declared
    assert receipt["hard_gates_passed"] == declared[:2]
    assert receipt["hard_gates_failed"] == [declared[2]]
    assert receipt["hard_gates_not_reached"] == declared[3:]
    assert receipt["pre_execution_hard_gates_complete"] is True
    assert receipt["comparative_interpretation_allowed"] is False
    assert receipt["outcome_based_retry_allowed"] is False
    assert "sensitive synthetic detail" not in json.dumps(receipt)
    assert (failed[0] / "runs" / "001" / "run_receipt.json").is_file()
    assert (failed[0] / "runs" / "002").is_dir()
    assert (runs_dir / f".claim-local-v1-{reviewed.sha256[:12]}.json").is_file()

    with pytest.raises(ExperimentContractError, match="already has an attempt"):
        run_oracle_exploration(**call)


def test_post_claim_system_exit_gets_a_terminal_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    candidate_path = tmp_path / "manifest.candidate.json"
    reviewed_path = tmp_path / "manifest.reviewed.json"
    emit_manifest_candidate(candidate_path, prepared)
    reviewed = finalize_reviewed_manifest(
        candidate_path=candidate_path,
        output_path=reviewed_path,
        review_confirmation=REVIEW_CONFIRMATION,
    )

    @contextmanager
    def interrupted_before_staging(*_args: object, **_kwargs: object):
        raise SystemExit(7)
        yield  # pragma: no cover

    monkeypatch.setattr(
        runner_module,
        "_atomic_study_directory",
        interrupted_before_staging,
    )
    runs_dir = tmp_path / "runs"
    with pytest.raises(SystemExit, match="7"):
        run_oracle_exploration(
            design_path=DESIGN_PATH,
            hdf5_path=tmp_path / "unused.hdf5",
            metadata_path=tmp_path / "unused.metadata.json",
            base_controller_path=tmp_path / "unused.bc.npz",
            residual_controller_path=tmp_path / "unused.residual.npz",
            manifest_path=reviewed_path,
            runs_dir=runs_dir,
        )

    terminal = runs_dir / f".claim-local-v1-{reviewed.sha256[:12]}.terminal_failure.json"
    receipt = json.loads(terminal.read_text(encoding="utf-8"))
    assert receipt["failure_class"] == "SystemExit"
    assert receipt["failure_summary"] == "system_exit_interruption"
    assert receipt["completed_run_count"] == 0
    assert receipt["hard_gates_passed"] == []
    assert receipt["hard_gates_failed"] == []
    assert receipt["hard_gates_not_reached"] == prepared.design.to_dict()["hard_gates"]


def test_post_claim_runtime_mismatch_does_not_guess_a_failed_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared()
    candidate_path = tmp_path / "manifest.candidate.json"
    reviewed_path = tmp_path / "manifest.reviewed.json"
    emit_manifest_candidate(candidate_path, prepared)
    reviewed = finalize_reviewed_manifest(
        candidate_path=candidate_path,
        output_path=reviewed_path,
        review_confirmation=REVIEW_CONFIRMATION,
    )

    def fail_preparation(**_kwargs: object) -> PreparedOracleExploration:
        raise ExperimentContractError("observed runtime differs from requested fields")

    monkeypatch.setattr(
        runner_module,
        "_prepare_oracle_exploration_from_design",
        fail_preparation,
    )
    runs_dir = tmp_path / "runs"
    with pytest.raises(ExperimentContractError, match="observed runtime differs"):
        run_oracle_exploration(
            design_path=DESIGN_PATH,
            hdf5_path=tmp_path / "unused.hdf5",
            metadata_path=tmp_path / "unused.metadata.json",
            base_controller_path=tmp_path / "unused.bc.npz",
            residual_controller_path=tmp_path / "unused.residual.npz",
            manifest_path=reviewed_path,
            runs_dir=runs_dir,
        )

    failed = list(runs_dir.glob(f"local-v1-{reviewed.sha256[:12]}-failed-*"))
    assert len(failed) == 1
    receipt = json.loads((failed[0] / "failure_receipt.json").read_text(encoding="utf-8"))
    assert receipt["failure_stage"] == "pre_execution_validation"
    assert receipt["hard_gates_passed"] == []
    assert receipt["hard_gates_failed"] == []
    assert receipt["hard_gates_not_reached"] == prepared.design.to_dict()["hard_gates"]
    assert receipt["pre_execution_hard_gates_complete"] is False
    assert (failed[0] / "attempt_start_receipt.json").is_file()


@pytest.mark.gym
def test_runtime_inspection_observes_requested_model_abi_without_an_action() -> None:
    design = load_oracle_exploration_design(DESIGN_PATH)
    receipt = inspect_oracle_exploration_runtime(design, reset_probe_seed=20260903)

    assert receipt.environment_id == "Humanoid-v5"
    assert (
        receipt.model_xml_sha256 == design.to_dict()["requested_runtime"]["mujoco_model_xml_sha256"]
    )
    assert receipt.max_actions == 1000
    assert receipt.observation_shape == (348,)
    assert receipt.action_shape == (17,)
    assert receipt.qpos_shape == (24,)
    assert receipt.qvel_shape == (23,)
    assert len(receipt.actuator_joint_order) == 17
    assert receipt.contact_capture_id == "all_mujoco_substeps_after_mj_step/v1"

    float64_space = spaces.Box(-0.4, 0.4, shape=(17,), dtype=np.float64)
    with pytest.raises(ExperimentContractError, match="runtime action space changed"):
        runner_module._validate_runtime_against_design(
            replace(
                receipt,
                action_space_sha256=runner_module._space_sha256(float64_space),
            ),
            design,
        )
    changed_bounds = spaces.Box(-0.39, 0.4, shape=(17,), dtype=np.float32)
    with pytest.raises(ExperimentContractError, match="runtime action space changed"):
        runner_module._validate_runtime_against_design(
            replace(
                receipt,
                action_space_sha256=runner_module._space_sha256(changed_bounds),
            ),
            design,
        )
