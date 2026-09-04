"""SB3-Contrib adapter for the synthetic TQC transfer fixture."""

from __future__ import annotations

import hashlib
import platform
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .fixed_reference import ExperimentContractError, read_bounded_json_artifact, sha256_file
from .runtime_identity import (
    dependency_lock_path,
    module_sha256,
    source_tree_sha256,
    space_sha256,
)
from .tqc_calibration_contract import canonical_json, sha256_json, validate_runtime_receipt
from .tqc_initialization_identity_contract import (
    InitializationIdentityDesign,
    array_receipt,
    array_sha256,
    normalized_to_physical,
    require_canonical_initialization_identity_design,
    verify_initialization_identity_arrays,
)
from .tqc_initialization_identity_v1 import E0_RECEIPT_BYTE_COUNT

MAX_E0_RECEIPT_BYTES = 4 * 1024 * 1024


def _require_receipt_field(
    receipt: Mapping[str, Any],
    field: str,
    expected: object,
) -> None:
    observed = receipt.get(field)
    if type(observed) is not type(expected) or observed != expected:
        raise ExperimentContractError(f"E0 receipt {field} differs from the fixture design")


def validate_e0_receipt(path: Path, design: InitializationIdentityDesign) -> dict[str, Any]:
    """Bind a completed authoritative E0 receipt without loading a controller."""

    require_canonical_initialization_identity_design(design)
    artifact = read_bounded_json_artifact(
        path,
        maximum_bytes=MAX_E0_RECEIPT_BYTES,
        artifact="E0 TQC resource receipt",
    )
    receipt = artifact.value
    requirement = design.required_e0
    if artifact.sha256 != requirement.receipt_sha256:
        raise ExperimentContractError("E0 receipt SHA-256 differs from the fixture design")
    if len(artifact.encoded_bytes) != E0_RECEIPT_BYTE_COUNT:
        raise ExperimentContractError("E0 receipt byte count differs from the fixture design")
    fixed = {
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
    }
    for field, expected in fixed.items():
        _require_receipt_field(receipt, field, expected)

    runtime = validate_runtime_receipt(receipt.get("runtime"))
    if runtime["runtime_sha256"] != requirement.runtime_sha256:
        raise ExperimentContractError("E0 runtime SHA-256 differs from the fixture design")
    runtime_expected = {
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
    for field, expected in runtime_expected.items():
        observed = runtime.get(field)
        if type(observed) is not type(expected) or observed != expected:
            raise ExperimentContractError(f"E0 runtime {field} differs from the fixture design")
    return {
        "receipt_sha256": requirement.receipt_sha256,
        "receipt_byte_count": len(artifact.encoded_bytes),
        "calibration_id": receipt["calibration_id"],
        "design_artifact_sha256": receipt["design_artifact_sha256"],
        "runtime_sha256": requirement.runtime_sha256,
        "calibration_gate_passed": True,
        "controller_bytes_emitted": False,
        "claim_boundary": receipt.get("claim_boundary"),
    }


def inspect_fixture_runtime(design: InitializationIdentityDesign) -> dict[str, Any]:
    require_canonical_initialization_identity_design(design)
    try:
        import gymnasium
        import sb3_contrib
        import stable_baselines3
        import torch
        from sb3_contrib.tqc import policies as tqc_policies
        from stable_baselines3.common import distributions, policies, torch_layers
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise ExperimentContractError(
            "install the train extra for the TQC transfer fixture"
        ) from exc

    required = design.runtime
    versions = {
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "gymnasium_version": gymnasium.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "sb3_contrib_version": sb3_contrib.__version__,
    }
    expected_versions = {
        "numpy_version": required.required_numpy_version,
        "torch_version": required.required_torch_version,
        "stable_baselines3_version": required.required_stable_baselines3_version,
        "sb3_contrib_version": required.required_sb3_contrib_version,
    }
    for field, expected in expected_versions.items():
        if str(versions[field]) != expected:
            raise ExperimentContractError(f"fixture runtime {field} differs from the design")

    sources = {
        "tqc_policy_source_sha256": module_sha256(tqc_policies),
        "sb3_distribution_source_sha256": module_sha256(distributions),
        "sb3_base_policy_source_sha256": module_sha256(policies),
        "sb3_torch_layers_source_sha256": module_sha256(torch_layers),
        "fixture_adapter_source_sha256": sha256_file(Path(__file__)),
        "fixture_contract_source_sha256": sha256_file(
            Path(__file__).with_name("tqc_initialization_identity_contract.py")
        ),
        "fixture_runner_source_sha256": sha256_file(
            Path(__file__).with_name("tqc_initialization_identity.py")
        ),
    }
    expected_sources = {
        "tqc_policy_source_sha256": required.tqc_policy_source_sha256,
        "sb3_distribution_source_sha256": required.sb3_distribution_source_sha256,
        "sb3_base_policy_source_sha256": required.sb3_base_policy_source_sha256,
        "sb3_torch_layers_source_sha256": required.sb3_torch_layers_source_sha256,
        "fixture_adapter_source_sha256": required.fixture_adapter_source_sha256,
        "fixture_contract_source_sha256": required.fixture_contract_source_sha256,
        "fixture_runner_source_sha256": required.fixture_runner_source_sha256,
    }
    for field, expected in expected_sources.items():
        if sources[field] != expected:
            raise ExperimentContractError(f"fixture runtime {field} differs from the design")

    lock_sha256 = sha256_file(dependency_lock_path())
    if lock_sha256 != required.dependency_lock_sha256:
        raise ExperimentContractError("fixture dependency lock differs from the design")
    payload: dict[str, Any] = {
        "runtime_kind": "data_only_synthetic_tqc_actor_fixture/v1",
        **{field: str(value) for field, value in versions.items()},
        **sources,
        "dependency_lock_sha256": lock_sha256,
        "source_tree_sha256": source_tree_sha256(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "torch_device": "cpu",
        "torch_intraop_thread_count": int(torch.get_num_threads()),
        "torch_interop_thread_count": int(torch.get_num_interop_threads()),
        "torch_deterministic_algorithms_enabled": bool(
            torch.are_deterministic_algorithms_enabled()
        ),
        "fixture_trains_or_steps_environment": False,
    }
    payload["runtime_sha256"] = sha256_json(payload)
    return payload


def fixture_observation_set(design: InitializationIdentityDesign) -> np.ndarray:
    require_canonical_initialization_identity_design(design)
    fixture = design.fixture
    indices = np.arange(
        fixture.observation_batch_size * fixture.base_observation_dim,
        dtype=np.int64,
    ).reshape(fixture.observation_batch_size, fixture.base_observation_dim)
    values = (((indices * 37 + 11) % 257) - 128).astype("<f4") / np.float32(64.0)
    result = np.ascontiguousarray(values, dtype="<f4")
    if array_sha256(result) != fixture.observation_set_sha256:
        raise ExperimentContractError("synthetic observation set differs from its pinned hash")
    return result


def fixture_reference_set(design: InitializationIdentityDesign) -> np.ndarray:
    require_canonical_initialization_identity_design(design)
    fixture = design.fixture
    indices = np.arange(
        fixture.observation_batch_size * fixture.reference_input_dim,
        dtype=np.int64,
    ).reshape(fixture.observation_batch_size, fixture.reference_input_dim)
    values = (((indices * 53 + 7) % 251) - 125).astype("<f4") / np.float32(32.0)
    result = np.ascontiguousarray(values, dtype="<f4")
    if array_sha256(result) != fixture.reference_set_sha256:
        raise ExperimentContractError("synthetic reference set differs from its pinned hash")
    return result


def _make_actor(design: InitializationIdentityDesign, observation_dim: int) -> object:
    import torch
    from gymnasium import spaces
    from sb3_contrib.tqc.policies import Actor
    from stable_baselines3.common.torch_layers import FlattenExtractor

    observation_space = spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(observation_dim,),
        dtype=np.float64,
    )
    action_space = spaces.Box(
        low=np.full(design.fixture.action_dim, design.action_transform.physical_low, dtype="<f4"),
        high=np.full(
            design.fixture.action_dim,
            design.action_transform.physical_high,
            dtype="<f4",
        ),
        dtype=np.float32,
    )
    features = FlattenExtractor(observation_space)
    actor = Actor(
        observation_space=observation_space,
        action_space=action_space,
        net_arch=list(design.fixture.hidden_layers),
        features_extractor=features,
        features_dim=features.features_dim,
        activation_fn=torch.nn.ReLU,
        use_sde=design.fixture.use_sde,
    )
    actor.eval()
    return actor


def _parameter_receipt(actor: object) -> dict[str, Any]:
    state_dict = getattr(actor, "state_dict", None)
    if not callable(state_dict):
        raise ExperimentContractError("fixture actor has no state dict")
    structure: list[dict[str, Any]] = []
    content = hashlib.sha256()
    for name, tensor in state_dict().items():
        array = np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4")
        metadata = {"name": name, "dtype": array.dtype.str, "shape": list(array.shape)}
        structure.append(metadata)
        encoded = canonical_json(metadata)
        content.update(len(encoded).to_bytes(8, "big"))
        content.update(encoded)
        raw = array.tobytes(order="C")
        content.update(len(raw).to_bytes(8, "big"))
        content.update(raw)
    if not structure:
        raise ExperimentContractError("fixture actor state dict is empty")
    return {
        "structure": structure,
        "structure_sha256": sha256_json(structure),
        "value_sha256": content.hexdigest(),
        "value_hash_source": "ordered_name_dtype_shape_and_exact_tensor_bytes/v1",
    }


def transfer_expanded_actor(
    base_actor: object,
    expanded_actor: object,
    design: InitializationIdentityDesign,
) -> dict[str, Any]:
    """Copy the base actor and append exact-zero reference columns."""

    import torch

    require_canonical_initialization_identity_design(design)
    base_state = base_actor.state_dict()
    expanded_state = expanded_actor.state_dict()
    if set(base_state) != set(expanded_state):
        raise ExperimentContractError("base and expanded actor parameter names differ")
    first_weight = "latent_pi.0.weight"
    if first_weight not in base_state:
        raise ExperimentContractError("TQC actor lacks its expected first affine layer")
    base_dim = design.fixture.base_observation_dim
    expanded_dim = design.fixture.expanded_observation_dim
    expected_base_shape = (design.fixture.hidden_layers[0], base_dim)
    expected_expanded_shape = (design.fixture.hidden_layers[0], expanded_dim)
    if tuple(base_state[first_weight].shape) != expected_base_shape:
        raise ExperimentContractError("base TQC first-layer shape differs from the design")
    if tuple(expanded_state[first_weight].shape) != expected_expanded_shape:
        raise ExperimentContractError("expanded TQC first-layer shape differs from the design")

    with torch.no_grad():
        for name, source in base_state.items():
            destination = expanded_state[name]
            if name == first_weight:
                destination.zero_()
                destination[:, :base_dim].copy_(source)
            else:
                if destination.shape != source.shape or destination.dtype != source.dtype:
                    raise ExperimentContractError(
                        f"expanded TQC parameter {name} is not architecture-preserving"
                    )
                destination.copy_(source)
        expanded_actor.load_state_dict(expanded_state, strict=True)

    final_state = expanded_actor.state_dict()
    for name, source in base_state.items():
        target = final_state[name]
        if name == first_weight:
            if not torch.equal(target[:, :base_dim], source):
                raise ExperimentContractError("expanded TQC state columns differ from the base")
            reference_columns = target[:, base_dim:]
            if not torch.equal(reference_columns, torch.zeros_like(reference_columns)):
                raise ExperimentContractError("expanded TQC reference columns are not exact zero")
        elif not torch.equal(target, source):
            raise ExperimentContractError(f"expanded TQC parameter {name} differs from the base")

    reference_columns = np.ascontiguousarray(
        final_state[first_weight][:, base_dim:].detach().cpu().numpy(),
        dtype="<f4",
    )
    state_columns = np.ascontiguousarray(
        final_state[first_weight][:, :base_dim].detach().cpu().numpy(),
        dtype="<f4",
    )
    base_first = np.ascontiguousarray(base_state[first_weight].detach().cpu().numpy(), dtype="<f4")
    return {
        "method": "copy_base_actor_and_zero_appended_reference_columns/v1",
        "first_affine_parameter": first_weight,
        "base_observation_dim": base_dim,
        "reference_input_dim": design.fixture.reference_input_dim,
        "expanded_observation_dim": expanded_dim,
        "base_first_layer_shape": list(base_first.shape),
        "expanded_first_layer_shape": list(final_state[first_weight].shape),
        "copied_state_columns_sha256": array_sha256(state_columns),
        "base_first_layer_sha256": array_sha256(base_first),
        "copied_state_columns_bitwise_equal": (
            state_columns.tobytes(order="C") == base_first.tobytes(order="C")
        ),
        "reference_columns_sha256": array_sha256(reference_columns),
        "reference_columns_exact_positive_zero": (
            reference_columns.tobytes(order="C")
            == np.zeros_like(reference_columns).tobytes(order="C")
        ),
        "reference_column_count": int(reference_columns.size),
        "all_other_parameters_bitwise_equal": True,
        "base_actor_parameters": _parameter_receipt(base_actor),
        "expanded_actor_parameters": _parameter_receipt(expanded_actor),
        "critics_replay_and_optimizer_transferred": False,
    }


def _actor_outputs(actor: object, inputs: np.ndarray, *, sample_seed: int) -> dict[str, np.ndarray]:
    import torch

    tensor = torch.as_tensor(inputs, dtype=torch.float32, device="cpu")
    with torch.inference_mode():
        mean, log_std, distribution_kwargs = actor.get_action_dist_params(tensor)
        if distribution_kwargs:
            raise ExperimentContractError("fixture only supports unstructured TQC exploration")
        deterministic = actor(tensor, deterministic=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(sample_seed)
            sampled = actor(tensor, deterministic=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(sample_seed)
            noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device)
        explicit_sample = torch.tanh(mean + torch.exp(log_std) * noise)
        if not torch.equal(sampled, explicit_sample):
            raise ExperimentContractError("seeded TQC sample differs from its Gaussian formula")

    arrays = {
        "mean": mean,
        "log_std": log_std,
        "deterministic_normalized": deterministic,
        "sampled_normalized": sampled,
        "sampling_noise": noise,
    }
    result = {
        name: np.ascontiguousarray(value.detach().cpu().numpy(), dtype="<f4")
        for name, value in arrays.items()
    }
    for name, value in result.items():
        if not np.isfinite(value).all():
            raise ExperimentContractError(f"fixture actor {name} contains non-finite values")
    result["deterministic_physical"] = np.ascontiguousarray(
        actor.unscale_action(result["deterministic_normalized"]), dtype="<f4"
    )
    result["sampled_physical"] = np.ascontiguousarray(
        actor.unscale_action(result["sampled_normalized"]), dtype="<f4"
    )
    return result


def _residual_outputs(design: InitializationIdentityDesign) -> dict[str, np.ndarray]:
    import torch

    shape = (design.fixture.observation_batch_size, design.fixture.action_dim)
    mean = torch.zeros(shape, dtype=torch.float32)
    log_std = torch.full(shape, design.fixture.residual_log_std, dtype=torch.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(design.fixture.residual_sampling_seed)
        noise = torch.randn(shape, dtype=torch.float32)
    deterministic = torch.tanh(mean)
    sampled = torch.tanh(mean + torch.exp(log_std) * noise)
    return {
        name: np.ascontiguousarray(value.numpy(), dtype="<f4")
        for name, value in {
            "mean": mean,
            "log_std": log_std,
            "deterministic_normalized": deterministic,
            "sampled_normalized": sampled,
            "sampling_noise": noise,
        }.items()
    }


def _output_receipt(output: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {field: array_receipt(value) for field, value in output.items()}


def run_synthetic_transfer_fixture(
    design: InitializationIdentityDesign,
) -> dict[str, Any]:
    """Run the fixture in memory; no model or checkpoint is serialized."""

    import torch

    require_canonical_initialization_identity_design(design)
    fixture = design.fixture
    observations = fixture_observation_set(design)
    references = fixture_reference_set(design)
    alternate_references = np.ascontiguousarray(-references, dtype="<f4")
    if array_sha256(alternate_references) == array_sha256(references):
        raise ExperimentContractError("alternate reference fixture does not change bytes")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(fixture.actor_initialization_seed)
        base_actor = _make_actor(design, fixture.base_observation_dim)
        torch.manual_seed(fixture.expanded_scratch_initialization_seed)
        expanded_actor = _make_actor(design, fixture.expanded_observation_dim)
    transfer = transfer_expanded_actor(base_actor, expanded_actor, design)

    expanded_inputs = np.ascontiguousarray(
        np.concatenate((observations, references), axis=1), dtype="<f4"
    )
    alternate_inputs = np.ascontiguousarray(
        np.concatenate((observations, alternate_references), axis=1), dtype="<f4"
    )
    base = _actor_outputs(base_actor, observations, sample_seed=fixture.action_sampling_seed)
    expanded = _actor_outputs(
        expanded_actor, expanded_inputs, sample_seed=fixture.action_sampling_seed
    )
    alternate = _actor_outputs(
        expanded_actor, alternate_inputs, sample_seed=fixture.action_sampling_seed
    )
    residual = _residual_outputs(design)

    zero_unclipped = np.ascontiguousarray(
        base["deterministic_normalized"]
        + np.float32(fixture.residual_scale) * residual["deterministic_normalized"],
        dtype="<f4",
    )
    zero_composed = np.ascontiguousarray(np.clip(zero_unclipped, -1.0, 1.0), dtype="<f4")
    sampled_unclipped = np.ascontiguousarray(
        base["deterministic_normalized"]
        + np.float32(fixture.residual_scale) * residual["sampled_normalized"],
        dtype="<f4",
    )
    sampled_composed = np.ascontiguousarray(np.clip(sampled_unclipped, -1.0, 1.0), dtype="<f4")
    sampled_lost_authority = np.ascontiguousarray(sampled_unclipped - sampled_composed, dtype="<f4")
    zero_physical = normalized_to_physical(zero_composed, design.action_transform)
    sampled_physical = normalized_to_physical(sampled_composed, design.action_transform)

    checks = verify_initialization_identity_arrays(
        design=design,
        base_mean=base["mean"],
        base_log_std=base["log_std"],
        base_sampling_noise=base["sampling_noise"],
        base_deterministic=base["deterministic_normalized"],
        base_sampled=base["sampled_normalized"],
        base_physical=base["deterministic_physical"],
        base_sampled_physical=base["sampled_physical"],
        expanded_mean=expanded["mean"],
        expanded_log_std=expanded["log_std"],
        expanded_deterministic=expanded["deterministic_normalized"],
        expanded_sampled=expanded["sampled_normalized"],
        expanded_physical=expanded["deterministic_physical"],
        expanded_sampled_physical=expanded["sampled_physical"],
        alternate_reference_mean=alternate["mean"],
        alternate_reference_log_std=alternate["log_std"],
        alternate_reference_deterministic=alternate["deterministic_normalized"],
        alternate_reference_sampled=alternate["sampled_normalized"],
        alternate_reference_physical=alternate["deterministic_physical"],
        alternate_reference_sampled_physical=alternate["sampled_physical"],
        residual_mean=residual["mean"],
        residual_log_std=residual["log_std"],
        residual_sampling_noise=residual["sampling_noise"],
        residual_deterministic=residual["deterministic_normalized"],
        residual_sampled=residual["sampled_normalized"],
        zero_residual_composed=zero_composed,
        zero_residual_physical=zero_physical,
        sampled_residual_composed=sampled_composed,
        sampled_residual_physical=sampled_physical,
    )

    residual_composition = {
        **residual,
        "zero_residual_unclipped_normalized": zero_unclipped,
        "zero_residual_composed_normalized": zero_composed,
        "zero_residual_composed_physical": zero_physical,
        "sampled_residual_unclipped_normalized": sampled_unclipped,
        "sampled_residual_composed_normalized": sampled_composed,
        "sampled_residual_composed_physical": sampled_physical,
        "sampled_residual_lost_authority_normalized": sampled_lost_authority,
    }
    residual_composition_receipt = _output_receipt(residual_composition)
    residual_composition_receipt["sampled_saturated_component_count"] = int(
        np.count_nonzero(sampled_lost_authority)
    )
    base_space_sha256 = space_sha256(base_actor.observation_space)
    expanded_space_sha256 = space_sha256(expanded_actor.observation_space)
    action_space_sha256 = space_sha256(base_actor.action_space)
    expected_spaces = {
        "base observation": (
            base_space_sha256,
            design.required_e0.observation_space_sha256,
        ),
        "expanded observation": (
            expanded_space_sha256,
            design.fixture.expanded_observation_space_sha256,
        ),
        "action": (action_space_sha256, design.required_e0.action_space_sha256),
    }
    for name, (observed, expected) in expected_spaces.items():
        if observed != expected:
            raise ExperimentContractError(f"fixture {name} space SHA-256 drifted")
    return {
        "fixture_identity_checks_passed": True,
        "observation_set": array_receipt(observations),
        "reference_set": array_receipt(references),
        "alternate_reference_set": array_receipt(alternate_references),
        "base_observation_space_sha256": base_space_sha256,
        "expanded_observation_space_sha256": expanded_space_sha256,
        "action_space_sha256": action_space_sha256,
        "transfer": transfer,
        "base_distribution_and_actions": _output_receipt(base),
        "expanded_distribution_and_actions": _output_receipt(expanded),
        "alternate_reference_distribution_and_actions": _output_receipt(alternate),
        "synthetic_residual_distribution_and_composition": residual_composition_receipt,
        "checks": checks,
        "training_steps": 0,
        "environment_steps": 0,
        "checkpoint_emitted": False,
    }


__all__ = [
    "MAX_E0_RECEIPT_BYTES",
    "fixture_observation_set",
    "fixture_reference_set",
    "inspect_fixture_runtime",
    "run_synthetic_transfer_fixture",
    "transfer_expanded_actor",
    "validate_e0_receipt",
]
