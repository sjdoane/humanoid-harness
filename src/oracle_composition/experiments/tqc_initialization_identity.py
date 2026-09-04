"""Run the synthetic TQC initialization-identity fixture."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifact_io import PublishedArtifact, publish_json_without_overwrite
from .fixed_reference import ExperimentContractError
from .tqc_calibration_contract import canonical_json, sha256_json, validate_runtime_receipt
from .tqc_initialization_identity_adapter import (
    inspect_fixture_runtime,
    run_synthetic_transfer_fixture,
    validate_e0_receipt,
)
from .tqc_initialization_identity_contract import (
    ACTUAL_E1_GATE_ID,
    FIXTURE_ID,
    MISSING_CONTROLLER_BLOCKER,
    UNEXECUTED_INITIALIZATIONS_BLOCKER,
    InitializationIdentityDesign,
    array_sha256,
    load_initialization_identity_design,
    verify_initialization_identity_arrays,
)
from .tqc_initialization_identity_v1 import E0_RECEIPT_BYTE_COUNT

RECEIPT_SCHEMA_VERSION = 1
EXECUTION_CALLABLE_AUTHORITY = "canonical_production_callables/v1"


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"{field} must be a lowercase SHA-256")
    return value


def _require_exact(value: object, expected: object, *, field: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ExperimentContractError(f"{field} must equal {expected!r}")


def _require_canonical_match(value: object, expected: object, *, field: str) -> None:
    if type(value) is not type(expected) or canonical_json(value) != canonical_json(expected):
        raise ExperimentContractError(f"{field} differs from canonical recomputation")


def _validated_e0_binding(
    binding: Mapping[str, Any],
    design: InitializationIdentityDesign,
) -> None:
    if type(binding) is not dict:
        raise ExperimentContractError("E0 binding must be an exact object")
    expected = {
        "receipt_sha256": design.required_e0.receipt_sha256,
        "runtime_sha256": design.required_e0.runtime_sha256,
        "calibration_id": design.required_e0.calibration_id,
        "design_artifact_sha256": design.required_e0.design_artifact_sha256,
        "calibration_gate_passed": True,
        "controller_bytes_emitted": False,
        "claim_boundary": "resource_integrity_only_no_behavior_or_controller_claim/v1",
        "receipt_byte_count": E0_RECEIPT_BYTE_COUNT,
    }
    if set(binding) != set(expected):
        raise ExperimentContractError("E0 binding keys differ from the canonical receipt")
    for field, value in expected.items():
        _require_exact(binding.get(field), value, field=f"E0 binding {field}")


def _validated_runtime_summary(
    runtime: Mapping[str, Any],
    design: InitializationIdentityDesign,
) -> None:
    required = design.runtime
    expected = {
        "runtime_kind": "data_only_synthetic_tqc_actor_fixture/v1",
        "numpy_version": required.required_numpy_version,
        "torch_version": required.required_torch_version,
        "stable_baselines3_version": required.required_stable_baselines3_version,
        "sb3_contrib_version": required.required_sb3_contrib_version,
        "dependency_lock_sha256": required.dependency_lock_sha256,
        "tqc_policy_source_sha256": required.tqc_policy_source_sha256,
        "sb3_distribution_source_sha256": required.sb3_distribution_source_sha256,
        "sb3_base_policy_source_sha256": required.sb3_base_policy_source_sha256,
        "sb3_torch_layers_source_sha256": required.sb3_torch_layers_source_sha256,
        "fixture_adapter_source_sha256": required.fixture_adapter_source_sha256,
        "fixture_contract_source_sha256": required.fixture_contract_source_sha256,
        "fixture_runner_source_sha256": required.fixture_runner_source_sha256,
        "torch_device": "cpu",
        "fixture_trains_or_steps_environment": False,
    }
    for field, value in expected.items():
        _require_exact(runtime.get(field), value, field=f"fixture runtime {field}")
    for field in (
        "runtime_sha256",
        "source_tree_sha256",
    ):
        _require_sha256(runtime.get(field), field=f"fixture runtime {field}")


def _receipt_matrix(
    section: Mapping[str, Any],
    field: str,
    *,
    shape: tuple[int, int],
) -> np.ndarray:
    record = section.get(field)
    if type(record) is not dict or set(record) != {"dtype", "shape", "sha256", "values"}:
        raise ExperimentContractError(f"synthetic fixture {field} array receipt is invalid")
    _require_exact(record["dtype"], "<f4", field=f"synthetic fixture {field} dtype")
    observed_shape = record["shape"]
    if (
        type(observed_shape) is not list
        or any(type(item) is not int for item in observed_shape)
        or observed_shape != list(shape)
    ):
        raise ExperimentContractError(f"synthetic fixture {field} array ABI is invalid")
    values = record["values"]
    if (
        type(values) is not list
        or len(values) != shape[0]
        or any(type(row) is not list or len(row) != shape[1] for row in values)
        or any(type(item) is not float for row in values for item in row)
    ):
        raise ExperimentContractError(
            f"synthetic fixture {field} values must be an exact float matrix"
        )
    value = np.ascontiguousarray(np.asarray(values, dtype="<f4"))
    if value.shape != shape or not np.isfinite(value).all():
        raise ExperimentContractError(f"synthetic fixture {field} values are invalid")
    claimed_sha256 = _require_sha256(
        record["sha256"], field=f"synthetic fixture {field} array SHA-256"
    )
    if array_sha256(value) != claimed_sha256:
        raise ExperimentContractError(f"synthetic fixture {field} array hash is invalid")
    return value


def _recompute_numeric_checks(
    fixture: Mapping[str, Any],
    design: InitializationIdentityDesign,
) -> dict[str, Any]:
    base = fixture.get("base_distribution_and_actions")
    expanded = fixture.get("expanded_distribution_and_actions")
    alternate = fixture.get("alternate_reference_distribution_and_actions")
    residual = fixture.get("synthetic_residual_distribution_and_composition")
    if not all(type(value) is dict for value in (base, expanded, alternate, residual)):
        raise ExperimentContractError("synthetic fixture omits distribution receipts")
    shape = (design.fixture.observation_batch_size, design.fixture.action_dim)
    base_arrays = {
        name: _receipt_matrix(base, name, shape=shape)
        for name in (
            "mean",
            "log_std",
            "sampling_noise",
            "deterministic_normalized",
            "sampled_normalized",
            "deterministic_physical",
            "sampled_physical",
        )
    }
    expanded_arrays = {
        name: _receipt_matrix(expanded, name, shape=shape)
        for name in (
            "mean",
            "log_std",
            "sampling_noise",
            "deterministic_normalized",
            "sampled_normalized",
            "deterministic_physical",
            "sampled_physical",
        )
    }
    alternate_arrays = {
        name: _receipt_matrix(alternate, name, shape=shape)
        for name in (
            "mean",
            "log_std",
            "sampling_noise",
            "deterministic_normalized",
            "sampled_normalized",
            "deterministic_physical",
            "sampled_physical",
        )
    }
    for name, noise in (
        ("expanded", expanded_arrays["sampling_noise"]),
        ("alternate-reference", alternate_arrays["sampling_noise"]),
    ):
        if noise.tobytes(order="C") != base_arrays["sampling_noise"].tobytes(order="C"):
            raise ExperimentContractError(f"synthetic fixture {name} sampling noise is not shared")
    residual_arrays = {
        name: _receipt_matrix(residual, name, shape=shape)
        for name in (
            "mean",
            "log_std",
            "sampling_noise",
            "deterministic_normalized",
            "sampled_normalized",
            "zero_residual_unclipped_normalized",
            "zero_residual_composed_normalized",
            "zero_residual_composed_physical",
            "sampled_residual_unclipped_normalized",
            "sampled_residual_composed_normalized",
            "sampled_residual_composed_physical",
            "sampled_residual_lost_authority_normalized",
        )
    }
    scale = np.float32(design.fixture.residual_scale)
    expected_zero_unclipped = np.ascontiguousarray(
        base_arrays["deterministic_normalized"]
        + scale * residual_arrays["deterministic_normalized"],
        dtype="<f4",
    )
    expected_sampled_unclipped = np.ascontiguousarray(
        base_arrays["deterministic_normalized"] + scale * residual_arrays["sampled_normalized"],
        dtype="<f4",
    )
    expected_lost_authority = np.ascontiguousarray(
        expected_sampled_unclipped - residual_arrays["sampled_residual_composed_normalized"],
        dtype="<f4",
    )
    sidecars = {
        "zero-residual unclipped": (
            residual_arrays["zero_residual_unclipped_normalized"],
            expected_zero_unclipped,
        ),
        "sampled-residual unclipped": (
            residual_arrays["sampled_residual_unclipped_normalized"],
            expected_sampled_unclipped,
        ),
        "sampled-residual lost authority": (
            residual_arrays["sampled_residual_lost_authority_normalized"],
            expected_lost_authority,
        ),
    }
    for name, (observed, expected) in sidecars.items():
        if observed.tobytes(order="C") != expected.tobytes(order="C"):
            raise ExperimentContractError(f"synthetic fixture {name} receipt is inconsistent")
    expected_saturation_count = int(np.count_nonzero(expected_lost_authority))
    _require_exact(
        residual.get("sampled_saturated_component_count"),
        expected_saturation_count,
        field="synthetic fixture sampled_saturated_component_count",
    )
    return verify_initialization_identity_arrays(
        design=design,
        base_mean=base_arrays["mean"],
        base_log_std=base_arrays["log_std"],
        base_sampling_noise=base_arrays["sampling_noise"],
        base_deterministic=base_arrays["deterministic_normalized"],
        base_sampled=base_arrays["sampled_normalized"],
        base_physical=base_arrays["deterministic_physical"],
        base_sampled_physical=base_arrays["sampled_physical"],
        expanded_mean=expanded_arrays["mean"],
        expanded_log_std=expanded_arrays["log_std"],
        expanded_deterministic=expanded_arrays["deterministic_normalized"],
        expanded_sampled=expanded_arrays["sampled_normalized"],
        expanded_physical=expanded_arrays["deterministic_physical"],
        expanded_sampled_physical=expanded_arrays["sampled_physical"],
        alternate_reference_mean=alternate_arrays["mean"],
        alternate_reference_log_std=alternate_arrays["log_std"],
        alternate_reference_deterministic=alternate_arrays["deterministic_normalized"],
        alternate_reference_sampled=alternate_arrays["sampled_normalized"],
        alternate_reference_physical=alternate_arrays["deterministic_physical"],
        alternate_reference_sampled_physical=alternate_arrays["sampled_physical"],
        residual_mean=residual_arrays["mean"],
        residual_log_std=residual_arrays["log_std"],
        residual_sampling_noise=residual_arrays["sampling_noise"],
        residual_deterministic=residual_arrays["deterministic_normalized"],
        residual_sampled=residual_arrays["sampled_normalized"],
        zero_residual_composed=residual_arrays["zero_residual_composed_normalized"],
        zero_residual_physical=residual_arrays["zero_residual_composed_physical"],
        sampled_residual_composed=residual_arrays["sampled_residual_composed_normalized"],
        sampled_residual_physical=residual_arrays["sampled_residual_composed_physical"],
    )


def _validated_fixture_summary(
    fixture: Mapping[str, Any],
    design: InitializationIdentityDesign,
) -> None:
    expected = {
        "fixture_identity_checks_passed": True,
        "training_steps": 0,
        "environment_steps": 0,
        "checkpoint_emitted": False,
        "base_observation_space_sha256": design.required_e0.observation_space_sha256,
        "expanded_observation_space_sha256": (design.fixture.expanded_observation_space_sha256),
        "action_space_sha256": design.required_e0.action_space_sha256,
    }
    for field, value in expected.items():
        _require_exact(fixture.get(field), value, field=f"synthetic fixture {field}")
    observation = fixture.get("observation_set")
    reference = fixture.get("reference_set")
    alternate_reference = fixture.get("alternate_reference_set")
    transfer = fixture.get("transfer")
    checks = fixture.get("checks")
    if not all(
        type(value) is dict
        for value in (observation, reference, alternate_reference, transfer, checks)
    ):
        raise ExperimentContractError("synthetic fixture omits a required receipt section")
    observation_values = _receipt_matrix(
        {"observation_set": observation},
        "observation_set",
        shape=(design.fixture.observation_batch_size, design.fixture.base_observation_dim),
    )
    reference_values = _receipt_matrix(
        {"reference_set": reference},
        "reference_set",
        shape=(design.fixture.observation_batch_size, design.fixture.reference_input_dim),
    )
    alternate_reference_values = _receipt_matrix(
        {"alternate_reference_set": alternate_reference},
        "alternate_reference_set",
        shape=(design.fixture.observation_batch_size, design.fixture.reference_input_dim),
    )
    if array_sha256(observation_values) != design.fixture.observation_set_sha256:
        raise ExperimentContractError("synthetic fixture observation hash is invalid")
    if array_sha256(reference_values) != design.fixture.reference_set_sha256:
        raise ExperimentContractError("synthetic fixture reference hash is invalid")
    if array_sha256(alternate_reference_values) == array_sha256(reference_values):
        raise ExperimentContractError("synthetic fixture alternate reference did not change")
    if not np.array_equal(alternate_reference_values, -reference_values):
        raise ExperimentContractError("synthetic fixture alternate reference is not exact negation")
    transfer_expected = {
        "copied_state_columns_bitwise_equal": True,
        "reference_columns_exact_positive_zero": True,
        "reference_column_count": (
            design.fixture.hidden_layers[0] * design.fixture.reference_input_dim
        ),
        "all_other_parameters_bitwise_equal": True,
        "critics_replay_and_optimizer_transferred": False,
    }
    for field, value in transfer_expected.items():
        _require_exact(
            transfer.get(field),
            value,
            field=f"synthetic fixture transfer {field}",
        )
    for actor_name in ("base_actor_parameters", "expanded_actor_parameters"):
        parameter_receipt = transfer.get(actor_name)
        if type(parameter_receipt) is not dict:
            raise ExperimentContractError(f"synthetic fixture {actor_name} is absent")
        structure = parameter_receipt.get("structure")
        if type(structure) is not list or not structure:
            raise ExperimentContractError(f"synthetic fixture {actor_name} structure is invalid")
        structure_sha256 = _require_sha256(
            parameter_receipt.get("structure_sha256"),
            field=f"synthetic fixture {actor_name} structure SHA-256",
        )
        if sha256_json(structure) != structure_sha256:
            raise ExperimentContractError(
                f"synthetic fixture {actor_name} structure hash is inconsistent"
            )
        _require_sha256(
            parameter_receipt.get("value_sha256"),
            field=f"synthetic fixture {actor_name} value SHA-256",
        )
        _require_exact(
            parameter_receipt.get("value_hash_source"),
            "ordered_name_dtype_shape_and_exact_tensor_bytes/v1",
            field=f"synthetic fixture {actor_name} value hash source",
        )
    _require_exact(
        checks.get("fixture_identity_checks_passed"),
        True,
        field="synthetic fixture numeric pass",
    )
    _require_exact(
        checks.get("stochastic_residual_is_nonzero"),
        True,
        field="synthetic fixture stochastic residual evidence",
    )
    recomputed_checks = _recompute_numeric_checks(fixture, design)
    if sha256_json(recomputed_checks) != sha256_json(checks):
        raise ExperimentContractError("synthetic fixture numeric check receipt is inconsistent")
    for section in ("formula_checks", "comparisons"):
        records = checks.get(section)
        if type(records) is not dict or not records:
            raise ExperimentContractError(f"synthetic fixture {section} is absent")
        for name, record in records.items():
            if type(record) is not dict or record.get("within_tolerance") is not True:
                raise ExperimentContractError(f"synthetic fixture {section}.{name} did not pass")
    for name in (
        "zero_residual_composition_vs_base",
        "zero_residual_physical_vs_base",
    ):
        record = checks["comparisons"].get(name)
        if type(record) is not dict or record.get("bitwise_equal") is not True:
            raise ExperimentContractError(f"synthetic fixture {name} is not bitwise exact")


@dataclass(frozen=True, slots=True)
class TransferFixtureResult:
    published: PublishedArtifact
    receipt: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.published.path),
            "file_sha256": self.published.sha256,
            "byte_count": self.published.byte_count,
            "fixture_identity_checks_passed": self.receipt["fixture_identity_checks_passed"],
            "actual_e1_gate_passed": self.receipt["actual_e1_gate_passed"],
            "actual_e1_blocker": self.receipt["actual_e1_blocker"],
            "claim_status": self.receipt["claim_status"],
        }


def build_transfer_fixture_receipt(
    *,
    design_path: Path,
    e0_binding: dict[str, Any],
    runtime: dict[str, Any],
    fixture: dict[str, Any],
) -> dict[str, Any]:
    loaded = load_initialization_identity_design(Path(design_path))
    design = loaded.design
    runtime = validate_runtime_receipt(runtime)
    _validated_e0_binding(e0_binding, design)
    _validated_runtime_summary(runtime, design)
    _validated_fixture_summary(fixture, design)
    canonical_runtime = inspect_fixture_runtime(design)
    _require_canonical_match(runtime, canonical_runtime, field="fixture runtime receipt")
    canonical_fixture = run_synthetic_transfer_fixture(design)
    _require_canonical_match(fixture, canonical_fixture, field="synthetic fixture receipt")
    analysis = {
        "design_semantic_sha256": design.semantic_sha256,
        "e0_receipt_sha256": e0_binding["receipt_sha256"],
        "runtime_sha256": runtime["runtime_sha256"],
        "observation_set_sha256": fixture["observation_set"]["sha256"],
        "reference_set_sha256": fixture["reference_set"]["sha256"],
        "base_actor_parameter_sha256": fixture["transfer"]["base_actor_parameters"]["value_sha256"],
        "expanded_actor_parameter_sha256": fixture["transfer"]["expanded_actor_parameters"][
            "value_sha256"
        ],
        "checks": fixture["checks"],
        "actor_parameter_hash_source": "canonical_recomputed_tensor_bytes/v1",
    }
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "fixture_id": FIXTURE_ID,
        "actual_gate_id": ACTUAL_E1_GATE_ID,
        "completion_status": "complete",
        "fixture_execution_authoritative": True,
        "execution_callable_authority": EXECUTION_CALLABLE_AUTHORITY,
        "evidence_class": "software_identity_fixture",
        "claim_status": "synthetic_transfer_implementation_only_not_controller_evidence",
        "fixture_identity_checks_passed": True,
        "actual_e1_gate_passed": False,
        "actual_e1_execution_blocked": True,
        "actual_e1_blocker": MISSING_CONTROLLER_BLOCKER,
        "actual_e1_blockers": [
            MISSING_CONTROLLER_BLOCKER,
            UNEXECUTED_INITIALIZATIONS_BLOCKER,
        ],
        "trained_controller_content_sha256": None,
        "trained_controller_loaded": False,
        "stable_tracking_established": False,
        "behavioral_claim": None,
        "oracle_claim": None,
        "eligible_for_tracker_admission": False,
        "eligible_for_controller_training": False,
        "automatic_promotion": False,
        "design_artifact_sha256": loaded.artifact_sha256,
        "design_artifact_byte_count": loaded.artifact_byte_count,
        "design_semantic_sha256": design.semantic_sha256,
        "design": design.to_dict(),
        "actor_parameter_hash_source": "canonical_recomputed_tensor_bytes/v1",
        "e0_binding": e0_binding,
        "runtime": runtime,
        "fixture": fixture,
        "analysis_payload_sha256": sha256_json(analysis),
        "next_gate": (
            "obtain_one_trusted_hash_pinned_trained_tqc_actor_then_run_the_same_"
            "identity_checks_without_transferring_critics_replay_or_optimizer"
        ),
    }


def run_transfer_fixture(
    *,
    design_path: Path,
    e0_receipt_path: Path,
    output_path: Path,
) -> TransferFixtureResult:
    loaded = load_initialization_identity_design(Path(design_path))
    design = loaded.design
    e0_binding = validate_e0_receipt(Path(e0_receipt_path), design)
    runtime = inspect_fixture_runtime(design)
    fixture = run_synthetic_transfer_fixture(design)
    receipt = build_transfer_fixture_receipt(
        design_path=Path(design_path),
        e0_binding=e0_binding,
        runtime=runtime,
        fixture=fixture,
    )
    published = publish_json_without_overwrite(Path(output_path), receipt)
    return TransferFixtureResult(published=published, receipt=receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--e0-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_transfer_fixture(
            design_path=args.design,
            e0_receipt_path=args.e0_receipt,
            output_path=args.output,
        )
    except (ExperimentContractError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), sort_keys=True, ensure_ascii=False))
    return 0


__all__ = [
    "EXECUTION_CALLABLE_AUTHORITY",
    "RECEIPT_SCHEMA_VERSION",
    "TransferFixtureResult",
    "build_transfer_fixture_receipt",
    "main",
    "run_transfer_fixture",
]


if __name__ == "__main__":
    raise SystemExit(main())
