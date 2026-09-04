from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from oracle_composition.experiments import tqc_development_contract as contract_module
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_development_contract import (
    EVALUATION_SEEDS,
    EXPECTED_GRADIENT_UPDATES,
    EXPECTED_VECTOR_STEPS,
    MODEL_SEED,
    VIDEO_SEEDS,
    WORKER_SEEDS,
    load_tqc_development_design,
    validate_required_e0_receipt,
)

ROOT = Path(__file__).parents[2]
DESIGN_PATH = ROOT / (
    "experiments/bootstrap_tqc_humanoid/configs/tqc_base_controller_dev_1m_v0.study.json"
)


def _design_payload() -> dict[str, object]:
    return json.loads(DESIGN_PATH.read_text(encoding="utf-8"))


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def test_exact_development_design_is_predeclared_and_excluded() -> None:
    loaded = load_tqc_development_design(DESIGN_PATH)
    value = loaded.to_dict()

    assert loaded.model_seed == MODEL_SEED == 95001
    assert loaded.worker_seeds == WORKER_SEEDS == tuple(range(95001, 95006))
    assert loaded.evaluation_seeds == EVALUATION_SEEDS == tuple(range(96001, 96021))
    assert VIDEO_SEEDS == (96001, 96010, 96020)
    assert not set(WORKER_SEEDS) & set(EVALUATION_SEEDS)
    assert set(VIDEO_SEEDS) <= set(EVALUATION_SEEDS)
    assert value["formal_experiment_eligible"] is False
    assert value["tqc"]["expected_vector_steps"] == EXPECTED_VECTOR_STEPS
    assert value["tqc"]["expected_gradient_updates"] == EXPECTED_GRADIENT_UPDATES
    assert value["decision_rule"]["automatic_20m_authorization"] is False
    assert value["decision_rule"]["automatic_tracker_admission"] is False
    assert value["persistence"]["actor_export"]["publication_allowed"] is False
    assert value["persistence"]["actor_export"]["equivalence_comparison"] == (
        "exact_shape_dtype_and_c_order_bytes/v1"
    )
    assert len(value["persistence"]["actor_export"]["equivalence_receipt_required_hashes"]) == 8
    assert value["persistence"]["actor_export"]["equivalence_observation_set"] == {
        "construction": "integer_affine_grid_float32/v1",
        "shape": [4, 348],
        "sha256": "0c6a81b06a88cab7eca0255e75f021008b60025c4ddc4d3719426e3647159ec6",
        "sampling_seed": 97001,
    }
    assert loaded.file_sha256 == hashlib.sha256(DESIGN_PATH.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "mutate, error",
    [
        (lambda value: value.update({"unknown": True}), "design keys differ"),
        (
            lambda value: value["seed_schedule"].update({"model_seed": 95002}),
            "model_seed",
        ),
        (
            lambda value: value["seed_schedule"]["evaluation_seeds"].__setitem__(0, 95001),
            "evaluation_seeds",
        ),
        (
            lambda value: value["tqc"].update({"total_environment_steps": 20_000_000}),
            "total_environment_steps",
        ),
        (
            lambda value: value["tqc"].update({"n_steps": 3}),
            "n_steps",
        ),
        (
            lambda value: value["tqc"]["policy_kwargs"].update(
                {"optimizer_class": "torch.optim.SGD"}
            ),
            "optimizer_class",
        ),
        (
            lambda value: value["tqc"]["replay_buffer_kwargs"].update(
                {"handle_timeout_termination": 1}
            ),
            "replay_buffer_kwargs",
        ),
        (
            lambda value: value["tqc"]["policy_kwargs"].update({"use_expln": 0}),
            "policy_kwargs",
        ),
        (
            lambda value: value["persistence"]["actor_export"][
                "equivalence_observation_set"
            ].update({"sampling_seed": 97001.0}),
            "equivalence_observation_set",
        ),
        (
            lambda value: value["persistence"]["actor_export"][
                "equivalence_observation_set"
            ].update({"shape": [4.0, 348.0]}),
            "equivalence_observation_set",
        ),
        (
            lambda value: value["evaluation"].update(
                {"healthy_root_height_open_interval_m": [1, 2]}
            ),
            "healthy_root_height_open_interval_m",
        ),
        (
            lambda value: value["resource_gates"].update(
                {"minimum_environment_steps_per_second": 116}
            ),
            "minimum_environment_steps_per_second",
        ),
        (
            lambda value: value["attempt"].update({"metric_based_checkpointing": True}),
            "metric_based_checkpointing",
        ),
        (
            lambda value: value["persistence"]["actor_export"].update(
                {"publication_allowed": True}
            ),
            "publication_allowed",
        ),
        (
            lambda value: value["evaluation"].update({"deterministic_actions": False}),
            "deterministic_actions",
        ),
        (
            lambda value: value["decision_rule"].update({"automatic_20m_authorization": True}),
            "automatic_20m_authorization",
        ),
        (
            lambda value: value["runtime_requirements"].update(
                {"stable_baselines3_version": "2.9.1"}
            ),
            "stable_baselines3_version",
        ),
    ],
)
def test_design_rejects_scientific_boundary_drift(
    tmp_path: Path,
    mutate: object,
    error: str,
) -> None:
    value = copy.deepcopy(_design_payload())
    mutate(value)
    path = tmp_path / "design.json"
    _write_json(path, value)

    with pytest.raises(ExperimentContractError, match=error):
        load_tqc_development_design(path)


def test_design_rejects_duplicate_keys_and_nonfinite_numbers(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ExperimentContractError, match="duplicate key"):
        load_tqc_development_design(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"schema_version":NaN}', encoding="utf-8")
    with pytest.raises(ExperimentContractError, match="non-finite"):
        load_tqc_development_design(nonfinite)


def _e0_receipt(design: dict[str, object]) -> dict[str, object]:
    required = design["required_e0"]
    return {
        "completion_status": required["completion_status"],
        "calibration_gate_passed": required["calibration_gate_passed"],
        "measured_workload_gates_passed": required["measured_workload_gates_passed"],
        "design_artifact_sha256": required["design_artifact_sha256"],
        "observed_environment_steps": required["observed_environment_steps"],
        "environment_steps_per_second": required["observed_environment_steps_per_second"],
        "training_wall_seconds": required["observed_training_wall_seconds"],
        "peak_rss_bytes": required["observed_peak_rss_bytes"],
        "replay_buffer_allocation_bytes": required["observed_replay_buffer_allocation_bytes"],
        "checkpoint_emitted": False,
        "replay_buffer_emitted": False,
        "normalizer_emitted": False,
        "eligible_for_controller_training": False,
        "eligible_for_behavioral_evaluation": False,
        "runtime": {"runtime_sha256": required["runtime_sha256"]},
    }


def _bind_fixture_design(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt: dict[str, object],
) -> tuple[object, Path, str]:
    receipt_path = tmp_path / "e0.json"
    _write_json(receipt_path, receipt)
    digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    monkeypatch.setattr(contract_module, "E0_RECEIPT_SHA256", digest)
    payload = _design_payload()
    payload["required_e0"]["receipt_content_sha256"] = digest
    design_path = tmp_path / "design.json"
    _write_json(design_path, payload)
    return load_tqc_development_design(design_path), receipt_path, digest


def test_required_e0_receipt_is_content_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical_design = load_tqc_development_design(DESIGN_PATH)
    receipt = _e0_receipt(canonical_design.to_dict())
    design, receipt_path, digest = _bind_fixture_design(tmp_path, monkeypatch, receipt)

    result = validate_required_e0_receipt(receipt_path, design)

    assert result["receipt_content_sha256"] == digest
    assert result["resource_gate_passed"] is True
    assert result["controller_bytes_emitted"] is False

    tampered = _e0_receipt(canonical_design.to_dict())
    tampered["checkpoint_emitted"] = True
    tampered_design, tampered_path, _digest = _bind_fixture_design(
        tmp_path,
        monkeypatch,
        tampered,
    )
    with pytest.raises(ExperimentContractError, match="checkpoint_emitted"):
        validate_required_e0_receipt(tampered_path, tampered_design)


def test_required_e0_receipt_must_match_both_compiled_and_design_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = load_tqc_development_design(DESIGN_PATH)
    receipt_path = tmp_path / "e0.json"
    _write_json(receipt_path, _e0_receipt(design.to_dict()))
    monkeypatch.setattr(
        contract_module,
        "E0_RECEIPT_SHA256",
        hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(ExperimentContractError, match="compiled and development-design"):
        validate_required_e0_receipt(receipt_path, design)


def test_required_e0_receipt_rejects_unregistered_bytes(tmp_path: Path) -> None:
    design = load_tqc_development_design(DESIGN_PATH)
    path = tmp_path / "wrong.json"
    _write_json(path, _e0_receipt(design.to_dict()))

    with pytest.raises(ExperimentContractError, match="receipt bytes differ"):
        validate_required_e0_receipt(path, design)
