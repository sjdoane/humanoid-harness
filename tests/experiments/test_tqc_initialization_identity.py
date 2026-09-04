from __future__ import annotations

import inspect
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oracle_composition.experiments import tqc_initialization_identity as identity_module
from oracle_composition.experiments import tqc_initialization_identity_adapter as adapter_module
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_calibration_contract import sha256_json
from oracle_composition.experiments.tqc_initialization_identity import (
    build_transfer_fixture_receipt,
    run_transfer_fixture,
)
from oracle_composition.experiments.tqc_initialization_identity_adapter import (
    fixture_observation_set,
    fixture_reference_set,
    inspect_fixture_runtime,
    run_synthetic_transfer_fixture,
    validate_e0_receipt,
)
from oracle_composition.experiments.tqc_initialization_identity_contract import (
    MISSING_CONTROLLER_BLOCKER,
    InitializationIdentityDesign,
    array_sha256,
    load_initialization_identity_design,
    verify_initialization_identity_arrays,
)
from oracle_composition.experiments.tqc_initialization_identity_v1 import (
    DESIGN_ARTIFACT_BYTE_COUNT,
    DESIGN_ARTIFACT_SHA256,
    E0_RECEIPT_BYTE_COUNT,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DESIGN_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "bootstrap_tqc_humanoid"
    / "configs"
    / "tqc_initialization_transfer_fixture_v0.study.json"
)


def _design() -> InitializationIdentityDesign:
    return load_initialization_identity_design(DESIGN_PATH).design


@pytest.fixture
def canonical_fixture_runtime() -> dict[str, object]:
    """Run positive receipt tests only on the exact reviewed Torch build."""

    design = _design()
    try:
        return inspect_fixture_runtime(design)
    except ExperimentContractError as exc:
        import torch

        expected = "fixture runtime torch_version differs from the design"
        public_version = str(torch.__version__).split("+", maxsplit=1)[0]
        if str(exc) == expected and public_version == design.runtime.required_torch_version:
            pytest.skip("fixture evidence excludes an unreviewed Torch local-build tag")
        raise


@pytest.fixture
def non_authoritative_candidate_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    """Exercise rejection paths without treating a CI wheel as evidence."""

    import torch

    design = _design()
    with monkeypatch.context() as temporary_patch:
        temporary_patch.setattr(torch, "__version__", design.runtime.required_torch_version)
        runtime = inspect_fixture_runtime(design)
    monkeypatch.setattr(
        identity_module,
        "inspect_fixture_runtime",
        lambda fixture_design: deepcopy(runtime),
    )
    return runtime


def _e0_payload(design: InitializationIdentityDesign) -> dict[str, object]:
    requirement = design.required_e0
    runtime: dict[str, object] = {
        "environment_id": requirement.environment_id,
        "observation_shape": list(requirement.observation_shape),
        "action_shape": list(requirement.action_shape),
        "observation_space_sha256": requirement.observation_space_sha256,
        "action_space_sha256": requirement.action_space_sha256,
        "dependency_lock_sha256": design.runtime.dependency_lock_sha256,
        "tqc_policy_source_sha256": design.runtime.tqc_policy_source_sha256,
        "numpy_version": design.runtime.required_numpy_version,
        "torch_version": design.runtime.required_torch_version,
        "stable_baselines3_version": design.runtime.required_stable_baselines3_version,
        "sb3_contrib_version": design.runtime.required_sb3_contrib_version,
    }
    runtime["runtime_sha256"] = requirement.runtime_sha256
    return {
        "schema_version": 1,
        "calibration_id": requirement.calibration_id,
        "completion_status": "complete",
        "calibration_gate_passed": True,
        "authoritative_execution": True,
        "measured_workload_gates_passed": True,
        "design_artifact_sha256": requirement.design_artifact_sha256,
        "seed": requirement.seed,
        "seed_role": requirement.seed_role,
        "evidence_purpose": "resource_calibration",
        "execution_callable_authority": "canonical_production_callables/v1",
        "claim_boundary": "resource_integrity_only_no_behavior_or_controller_claim/v1",
        "automatic_promotion": False,
        "expected_environment_steps": requirement.expected_environment_steps,
        "observed_environment_steps": requirement.expected_environment_steps,
        "expected_vector_steps": requirement.expected_vector_steps,
        "observed_vector_steps": requirement.expected_vector_steps,
        "expected_gradient_updates": requirement.expected_gradient_updates,
        "observed_gradient_updates": requirement.expected_gradient_updates,
        "model_disposition": "discarded_unserialized",
        "checkpoint_emitted": False,
        "replay_buffer_emitted": False,
        "normalizer_emitted": False,
        "controller_artifact": None,
        "controller_claim": None,
        "behavioral_claim": None,
        "eligible_for_controller_training": False,
        "eligible_for_behavioral_evaluation": False,
        "runtime": runtime,
    }


def _e0_binding(design: InitializationIdentityDesign) -> dict[str, object]:
    return {
        "receipt_sha256": design.required_e0.receipt_sha256,
        "receipt_byte_count": E0_RECEIPT_BYTE_COUNT,
        "calibration_id": design.required_e0.calibration_id,
        "design_artifact_sha256": design.required_e0.design_artifact_sha256,
        "runtime_sha256": design.required_e0.runtime_sha256,
        "calibration_gate_passed": True,
        "controller_bytes_emitted": False,
        "claim_boundary": "resource_integrity_only_no_behavior_or_controller_claim/v1",
    }


def _mock_exact_e0_artifact(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, object],
) -> None:
    design = _design()
    artifact = SimpleNamespace(
        value=payload,
        encoded_bytes=b"x" * E0_RECEIPT_BYTE_COUNT,
        sha256=design.required_e0.receipt_sha256,
    )
    monkeypatch.setattr(
        adapter_module, "read_bounded_json_artifact", lambda *args, **kwargs: artifact
    )
    monkeypatch.setattr(
        adapter_module,
        "validate_runtime_receipt",
        lambda runtime: dict(runtime),
    )


def _matrix(record: dict[str, object]) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(record["values"], dtype="<f4"))


def _verification_arguments(fixture: dict[str, object]) -> dict[str, np.ndarray]:
    base = fixture["base_distribution_and_actions"]
    expanded = fixture["expanded_distribution_and_actions"]
    alternate = fixture["alternate_reference_distribution_and_actions"]
    residual = fixture["synthetic_residual_distribution_and_composition"]
    assert isinstance(base, dict)
    assert isinstance(expanded, dict)
    assert isinstance(alternate, dict)
    assert isinstance(residual, dict)
    return {
        "base_mean": _matrix(base["mean"]),
        "base_log_std": _matrix(base["log_std"]),
        "base_sampling_noise": _matrix(base["sampling_noise"]),
        "base_deterministic": _matrix(base["deterministic_normalized"]),
        "base_sampled": _matrix(base["sampled_normalized"]),
        "base_physical": _matrix(base["deterministic_physical"]),
        "base_sampled_physical": _matrix(base["sampled_physical"]),
        "expanded_mean": _matrix(expanded["mean"]),
        "expanded_log_std": _matrix(expanded["log_std"]),
        "expanded_deterministic": _matrix(expanded["deterministic_normalized"]),
        "expanded_sampled": _matrix(expanded["sampled_normalized"]),
        "expanded_physical": _matrix(expanded["deterministic_physical"]),
        "expanded_sampled_physical": _matrix(expanded["sampled_physical"]),
        "alternate_reference_mean": _matrix(alternate["mean"]),
        "alternate_reference_log_std": _matrix(alternate["log_std"]),
        "alternate_reference_deterministic": _matrix(alternate["deterministic_normalized"]),
        "alternate_reference_sampled": _matrix(alternate["sampled_normalized"]),
        "alternate_reference_physical": _matrix(alternate["deterministic_physical"]),
        "alternate_reference_sampled_physical": _matrix(alternate["sampled_physical"]),
        "residual_mean": _matrix(residual["mean"]),
        "residual_log_std": _matrix(residual["log_std"]),
        "residual_sampling_noise": _matrix(residual["sampling_noise"]),
        "residual_deterministic": _matrix(residual["deterministic_normalized"]),
        "residual_sampled": _matrix(residual["sampled_normalized"]),
        "zero_residual_composed": _matrix(residual["zero_residual_composed_normalized"]),
        "zero_residual_physical": _matrix(residual["zero_residual_composed_physical"]),
        "sampled_residual_composed": _matrix(residual["sampled_residual_composed_normalized"]),
        "sampled_residual_physical": _matrix(residual["sampled_residual_composed_physical"]),
    }


def test_design_pins_e0_dimensions_sources_and_numeric_sets() -> None:
    loaded = load_initialization_identity_design(DESIGN_PATH)
    design = loaded.design

    assert design.fixture.base_observation_dim == 348
    assert design.fixture.reference_window_shape == (8, 45)
    assert design.fixture.expanded_observation_dim == 708
    assert design.fixture.action_dim == 17
    assert {
        design.fixture.actor_initialization_seed,
        design.fixture.expanded_scratch_initialization_seed,
        design.fixture.action_sampling_seed,
        design.fixture.residual_sampling_seed,
    } == {93001, 93002, 93003, 93004}
    assert design.required_e0.receipt_sha256 == (
        "2436c2e93cac8b5ed357acdcb19cca0b6222792a1c9c0d65f06577509d68a587"
    )
    assert design.required_e0.runtime_sha256 == (
        "710d524b2936ee29bbeb5be9336ed5bb0e12964919f7035d213d9f19348e0a77"
    )
    assert design.tolerance.absolute == 1e-6
    assert design.tolerance.relative == 1e-6
    assert design.fixture.residual_log_std == -3.0
    assert design.fixture.residual_scale == 0.08
    assert design.fixture.expanded_observation_space_sha256 == (
        "901c0e6ca77dbbb23e47299b90473a6f6ee84565c44a6156d9bc57d5c1320e51"
    )
    assert array_sha256(fixture_observation_set(design)) == (design.fixture.observation_set_sha256)
    assert array_sha256(fixture_reference_set(design)) == design.fixture.reference_set_sha256
    assert loaded.artifact_sha256 == DESIGN_ARTIFACT_SHA256
    assert loaded.artifact_byte_count == DESIGN_ARTIFACT_BYTE_COUNT


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        (("required_e0", "receipt_sha256"), "0" * 64, "required_e0.receipt_sha256"),
        (("required_e0", "runtime_sha256"), "1" * 64, "required_e0.runtime_sha256"),
        (("required_e0", "design_artifact_sha256"), "2" * 64, "design_artifact_sha256"),
        (("required_e0", "observation_space_sha256"), "3" * 64, "observation_space"),
        (("required_e0", "action_space_sha256"), "4" * 64, "action_space"),
        (("fixture", "actor_initialization_seed"), 123, "actor_initialization_seed"),
        (
            ("fixture", "expanded_scratch_initialization_seed"),
            123,
            "expanded_scratch_initialization_seed",
        ),
        (("fixture", "action_sampling_seed"), 123, "action_sampling_seed"),
        (("fixture", "residual_sampling_seed"), 123, "residual_sampling_seed"),
        (("fixture", "residual_log_std"), -1.0, "residual_log_std"),
        (("fixture", "residual_scale"), 0.5, "residual_scale"),
        (("fixture", "observation_set_sha256"), "5" * 64, "observation_set_sha256"),
        (("fixture", "reference_set_sha256"), "6" * 64, "reference_set_sha256"),
        (
            ("fixture", "expanded_observation_space_sha256"),
            "7" * 64,
            "expanded_observation_space_sha256",
        ),
        (("runtime", "dependency_lock_sha256"), "8" * 64, "dependency_lock_sha256"),
        (("runtime", "tqc_policy_source_sha256"), "9" * 64, "tqc_policy_source_sha256"),
        (
            ("runtime", "sb3_distribution_source_sha256"),
            "a" * 64,
            "sb3_distribution_source_sha256",
        ),
        (
            ("runtime", "sb3_base_policy_source_sha256"),
            "b" * 64,
            "sb3_base_policy_source_sha256",
        ),
        (
            ("runtime", "sb3_torch_layers_source_sha256"),
            "c" * 64,
            "sb3_torch_layers_source_sha256",
        ),
        (
            ("runtime", "fixture_adapter_source_sha256"),
            "d" * 64,
            "fixture_adapter_source_sha256",
        ),
        (
            ("runtime", "fixture_contract_source_sha256"),
            "e" * 64,
            "fixture_contract_source_sha256",
        ),
        (
            ("runtime", "fixture_runner_source_sha256"),
            "f" * 64,
            "fixture_runner_source_sha256",
        ),
        (("tolerance", "absolute"), 0.01, "tolerance"),
    ],
)
def test_v1_design_rejects_every_reviewed_identity_variant(
    path: tuple[str, str],
    value: object,
    error: str,
) -> None:
    payload = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    payload[path[0]][path[1]] = value

    with pytest.raises(ExperimentContractError, match=error):
        InitializationIdentityDesign.from_dict(payload)


def test_design_loader_rejects_reencoded_or_changed_v1_bytes(tmp_path: Path) -> None:
    path = tmp_path / "design.json"
    path.write_text(json.dumps(json.loads(DESIGN_PATH.read_text())) + "\n", encoding="utf-8")

    with pytest.raises(ExperimentContractError, match="compiled identity"):
        load_initialization_identity_design(path)


def test_e0_binding_accepts_only_completed_discarded_resource_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = _design()
    payload = _e0_payload(design)
    _mock_exact_e0_artifact(monkeypatch, payload)

    binding = validate_e0_receipt(tmp_path / "e0.json", design)

    assert binding["calibration_gate_passed"] is True
    assert binding["controller_bytes_emitted"] is False
    assert binding["receipt_sha256"] == design.required_e0.receipt_sha256
    assert binding["receipt_byte_count"] == E0_RECEIPT_BYTE_COUNT


def test_e0_binding_rejects_semantically_matching_unpinned_substitute(tmp_path: Path) -> None:
    design = _design()
    path = tmp_path / "e0.json"
    path.write_text(json.dumps(_e0_payload(design)) + "\n", encoding="utf-8")

    with pytest.raises(ExperimentContractError, match="receipt SHA-256"):
        validate_e0_receipt(path, design)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("calibration_gate_passed", 1),
        ("authoritative_execution", False),
        ("checkpoint_emitted", True),
        ("observed_environment_steps", 99_999),
    ],
)
def test_e0_binding_rejects_failed_or_controller_bearing_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    design = _design()
    payload = _e0_payload(design)
    payload[field] = value
    _mock_exact_e0_artifact(monkeypatch, payload)

    with pytest.raises(ExperimentContractError, match=field):
        validate_e0_receipt(tmp_path / "e0.json", design)


def test_e0_binding_rejects_a_forged_runtime_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = _design()
    payload = _e0_payload(design)
    payload["runtime"]["action_shape"] = [16]  # type: ignore[index]
    _mock_exact_e0_artifact(monkeypatch, payload)

    with pytest.raises(ExperimentContractError, match="action_shape"):
        validate_e0_receipt(tmp_path / "e0.json", design)


def test_e0_binding_rejects_unpinned_runtime_receipt(tmp_path: Path) -> None:
    design = _design()
    design = replace(
        design,
        required_e0=replace(design.required_e0, runtime_sha256="0" * 64),
    )

    with pytest.raises(ExperimentContractError, match=r"required_e0\.runtime_sha256"):
        validate_e0_receipt(tmp_path / "e0.json", design)


def test_runtime_inspection_binds_pinned_actor_distribution_and_lock_sources(
    canonical_fixture_runtime: dict[str, object],
) -> None:
    runtime = canonical_fixture_runtime

    assert runtime["runtime_kind"] == "data_only_synthetic_tqc_actor_fixture/v1"
    assert runtime["runtime_sha256"]
    assert runtime["fixture_trains_or_steps_environment"] is False


def test_runtime_inspection_rejects_unreviewed_torch_local_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    design = _design()
    monkeypatch.setattr(torch, "__version__", f"{design.runtime.required_torch_version}+unreviewed")

    with pytest.raises(ExperimentContractError, match="torch_version"):
        inspect_fixture_runtime(design)


def test_receipt_issuance_rejects_unreviewed_torch_local_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    design = _design()
    with monkeypatch.context() as temporary_patch:
        temporary_patch.setattr(torch, "__version__", design.runtime.required_torch_version)
        candidate_runtime = inspect_fixture_runtime(design)
    monkeypatch.setattr(torch, "__version__", f"{design.runtime.required_torch_version}+unreviewed")

    with pytest.raises(ExperimentContractError, match="torch_version"):
        build_transfer_fixture_receipt(
            design_path=DESIGN_PATH,
            e0_binding=_e0_binding(design),
            runtime=candidate_runtime,
            fixture=run_synthetic_transfer_fixture(design),
        )


def test_runtime_inspection_rejects_source_hash_drift() -> None:
    design = _design()
    drifted = replace(
        design,
        runtime=replace(design.runtime, tqc_policy_source_sha256="0" * 64),
    )

    with pytest.raises(ExperimentContractError, match="tqc_policy_source_sha256"):
        inspect_fixture_runtime(drifted)


def test_synthetic_fixture_preserves_expanded_distribution_and_zero_residual() -> None:
    design = _design()
    fixture = run_synthetic_transfer_fixture(design)

    assert fixture["fixture_identity_checks_passed"] is True
    assert fixture["training_steps"] == 0
    assert fixture["environment_steps"] == 0
    assert fixture["checkpoint_emitted"] is False
    transfer = fixture["transfer"]
    assert transfer["copied_state_columns_bitwise_equal"] is True
    assert transfer["reference_columns_exact_positive_zero"] is True
    assert transfer["reference_column_count"] == 256 * 8 * 45
    checks = fixture["checks"]
    assert checks["stochastic_residual_is_nonzero"] is True
    assert all(item["within_tolerance"] for item in checks["comparisons"].values())
    assert checks["comparisons"]["zero_residual_composition_vs_base"]["bitwise_equal"] is True


def test_numeric_verifier_rejects_expanded_mean_drift() -> None:
    design = _design()
    fixture = run_synthetic_transfer_fixture(design)
    arguments = _verification_arguments(fixture)
    arguments["expanded_mean"] = arguments["expanded_mean"].copy()
    arguments["expanded_mean"][0, 0] += np.float32(0.01)

    with pytest.raises(ExperimentContractError, match="exceeded tolerance"):
        verify_initialization_identity_arrays(design=design, **arguments)


def test_numeric_verifier_rejects_nonzero_deterministic_residual() -> None:
    design = _design()
    fixture = run_synthetic_transfer_fixture(design)
    arguments = _verification_arguments(fixture)
    arguments["residual_deterministic"] = arguments["residual_deterministic"].copy()
    arguments["residual_deterministic"][0, 0] = np.float32(0.1)

    with pytest.raises(ExperimentContractError, match="exact positive zero"):
        verify_initialization_identity_arrays(design=design, **arguments)


def test_receipt_keeps_fixture_pass_separate_from_actual_e1(
    canonical_fixture_runtime: dict[str, object],
) -> None:
    loaded = load_initialization_identity_design(DESIGN_PATH)
    design = loaded.design
    runtime = canonical_fixture_runtime
    fixture = run_synthetic_transfer_fixture(design)
    e0_binding = _e0_binding(design)

    receipt = build_transfer_fixture_receipt(
        design_path=DESIGN_PATH,
        e0_binding=e0_binding,
        runtime=runtime,
        fixture=fixture,
    )

    assert receipt["fixture_identity_checks_passed"] is True
    assert receipt["fixture_execution_authoritative"] is True
    assert "authoritative_execution" not in receipt
    assert receipt["actual_e1_gate_passed"] is False
    assert receipt["actual_e1_execution_blocked"] is True
    assert receipt["actual_e1_blocker"] == MISSING_CONTROLLER_BLOCKER
    assert len(receipt["actual_e1_blockers"]) == 2
    assert receipt["trained_controller_content_sha256"] is None
    assert receipt["stable_tracking_established"] is False
    assert receipt["behavioral_claim"] is None
    assert receipt["design_artifact_sha256"] == DESIGN_ARTIFACT_SHA256
    assert receipt["design_artifact_byte_count"] == DESIGN_ARTIFACT_BYTE_COUNT
    assert receipt["actor_parameter_hash_source"] == "canonical_recomputed_tensor_bytes/v1"


def test_receipt_builder_rejects_spoofed_nested_fixture_pass(
    non_authoritative_candidate_runtime: dict[str, object],
) -> None:
    loaded = load_initialization_identity_design(DESIGN_PATH)
    design = loaded.design
    runtime = non_authoritative_candidate_runtime
    fixture = deepcopy(run_synthetic_transfer_fixture(design))
    fixture["transfer"]["reference_columns_exact_positive_zero"] = False
    e0_binding = _e0_binding(design)

    with pytest.raises(ExperimentContractError, match="reference_columns_exact_positive_zero"):
        build_transfer_fixture_receipt(
            design_path=DESIGN_PATH,
            e0_binding=e0_binding,
            runtime=runtime,
            fixture=fixture,
        )


def test_receipt_builder_recomputes_numeric_checks_from_recorded_values(
    non_authoritative_candidate_runtime: dict[str, object],
) -> None:
    loaded = load_initialization_identity_design(DESIGN_PATH)
    design = loaded.design
    runtime = non_authoritative_candidate_runtime
    fixture = deepcopy(run_synthetic_transfer_fixture(design))
    mean_record = fixture["expanded_distribution_and_actions"]["mean"]
    mean = _matrix(mean_record)
    mean[0, 0] += np.float32(0.01)
    mean_record["values"] = mean.tolist()
    mean_record["sha256"] = array_sha256(mean)
    e0_binding = _e0_binding(design)

    with pytest.raises(ExperimentContractError, match="exceeded tolerance"):
        build_transfer_fixture_receipt(
            design_path=DESIGN_PATH,
            e0_binding=e0_binding,
            runtime=runtime,
            fixture=fixture,
        )


def test_receipt_builder_rejects_unbound_actor_value_hash(
    non_authoritative_candidate_runtime: dict[str, object],
) -> None:
    design = _design()
    runtime = non_authoritative_candidate_runtime
    fixture = deepcopy(run_synthetic_transfer_fixture(design))
    fixture["transfer"]["base_actor_parameters"]["value_sha256"] = "0" * 64

    with pytest.raises(ExperimentContractError, match="canonical recomputation"):
        build_transfer_fixture_receipt(
            design_path=DESIGN_PATH,
            e0_binding=_e0_binding(design),
            runtime=runtime,
            fixture=fixture,
        )


def test_receipt_builder_rejects_unshared_action_sampling_noise(
    non_authoritative_candidate_runtime: dict[str, object],
) -> None:
    design = _design()
    runtime = non_authoritative_candidate_runtime
    fixture = deepcopy(run_synthetic_transfer_fixture(design))
    noise_record = fixture["expanded_distribution_and_actions"]["sampling_noise"]
    noise = np.zeros_like(_matrix(noise_record))
    noise_record["values"] = noise.tolist()
    noise_record["sha256"] = array_sha256(noise)

    with pytest.raises(ExperimentContractError, match="sampling noise is not shared"):
        build_transfer_fixture_receipt(
            design_path=DESIGN_PATH,
            e0_binding=_e0_binding(design),
            runtime=runtime,
            fixture=fixture,
        )


def test_receipt_builder_rejects_false_saturation_count(
    non_authoritative_candidate_runtime: dict[str, object],
) -> None:
    design = _design()
    runtime = non_authoritative_candidate_runtime
    fixture = deepcopy(run_synthetic_transfer_fixture(design))
    residual = fixture["synthetic_residual_distribution_and_composition"]
    residual["sampled_saturated_component_count"] = 999

    with pytest.raises(ExperimentContractError, match="sampled_saturated_component_count"):
        build_transfer_fixture_receipt(
            design_path=DESIGN_PATH,
            e0_binding=_e0_binding(design),
            runtime=runtime,
            fixture=fixture,
        )


def test_receipt_builder_rejects_false_lost_authority_receipt(
    non_authoritative_candidate_runtime: dict[str, object],
) -> None:
    design = _design()
    runtime = non_authoritative_candidate_runtime
    fixture = deepcopy(run_synthetic_transfer_fixture(design))
    residual = fixture["synthetic_residual_distribution_and_composition"]
    record = residual["sampled_residual_lost_authority_normalized"]
    lost = _matrix(record)
    lost[0, 0] = np.float32(0.25)
    record["values"] = lost.tolist()
    record["sha256"] = array_sha256(lost)

    with pytest.raises(ExperimentContractError, match="lost authority receipt is inconsistent"):
        build_transfer_fixture_receipt(
            design_path=DESIGN_PATH,
            e0_binding=_e0_binding(design),
            runtime=runtime,
            fixture=fixture,
        )


def test_receipt_builder_recomputes_dynamic_runtime_identity(
    non_authoritative_candidate_runtime: dict[str, object],
) -> None:
    design = _design()
    runtime = dict(non_authoritative_candidate_runtime)
    runtime["source_tree_sha256"] = "0" * 64
    unsigned = dict(runtime)
    unsigned.pop("runtime_sha256")
    runtime["runtime_sha256"] = sha256_json(unsigned)

    with pytest.raises(
        ExperimentContractError,
        match=r"runtime receipt.*canonical recomputation",
    ):
        build_transfer_fixture_receipt(
            design_path=DESIGN_PATH,
            e0_binding=_e0_binding(design),
            runtime=runtime,
            fixture=run_synthetic_transfer_fixture(design),
        )


def test_receipt_builder_has_no_caller_supplied_design_identity() -> None:
    parameters = inspect.signature(build_transfer_fixture_receipt).parameters

    assert "design_artifact_sha256" not in parameters
    assert "design_artifact_byte_count" not in parameters
    assert "design_path" in parameters


def test_runner_publishes_exclusive_local_fixture_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    canonical_fixture_runtime: dict[str, object],
) -> None:
    assert canonical_fixture_runtime["runtime_kind"] == ("data_only_synthetic_tqc_actor_fixture/v1")
    e0 = tmp_path / "e0.json"
    output = tmp_path / "fixture.json"
    monkeypatch.setattr(
        identity_module,
        "validate_e0_receipt",
        lambda path, fixture_design: _e0_binding(fixture_design),
    )

    result = run_transfer_fixture(
        design_path=DESIGN_PATH,
        e0_receipt_path=e0,
        output_path=output,
    )

    assert result.receipt["fixture_identity_checks_passed"] is True
    assert result.receipt["actual_e1_gate_passed"] is False
    assert result.published.path == output.resolve()
    with pytest.raises(ExperimentContractError, match="refusing to overwrite"):
        run_transfer_fixture(
            design_path=DESIGN_PATH,
            e0_receipt_path=e0,
            output_path=output,
        )
