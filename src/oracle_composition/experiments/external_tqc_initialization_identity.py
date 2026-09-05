"""Actual E1 initialization identity on the imported external expert actor."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oracle_composition.sources.external_sb3_actor import (
    AUTHORITY,
    POLICY_SHA256,
    validate_external_actor_import_receipt,
)

from .artifact_io import PublishedArtifact, publish_json_without_overwrite
from .external_tqc_actor_equivalence import validate_external_actor_equivalence_receipt
from .fixed_reference import ExperimentContractError, read_bounded_json_artifact
from .tqc_actor_equivalence_primitives import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    seeded_squashed_normal_sample,
)
from .tqc_actor_npz import ACTION_WIDTH, OBSERVATION_WIDTH, load_actor_npz, validate_actor_arrays
from .tqc_initialization_identity_adapter import fixture_observation_set, fixture_reference_set
from .tqc_initialization_identity_contract import (
    ACTUAL_E1_GATE_ID,
    InitializationIdentityDesign,
    array_receipt,
    array_sha256,
    load_initialization_identity_design,
    normalized_to_physical,
    require_canonical_initialization_identity_design,
)
from .tqc_initialization_identity_v1 import (
    DESIGN_ARTIFACT_BYTE_COUNT,
    DESIGN_ARTIFACT_SHA256,
)

E1_RECEIPT_ID = "external_tqc_initialization_identity/v1"
EVIDENCE_CLASS = "external_base_import"
EXPERT_ACTOR_LOGICAL_PATH = (
    "artifacts/bootstrap_tqc_humanoid/farama_minari_humanoid_v5_tqc_actor_v1.npz"
)
EXPERT_ACTOR_NPZ_SHA256 = "60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b"
EXPERT_ACTOR_NPZ_BYTE_COUNT = 618_674
EXPERT_ACTOR_STATE_SHA256 = "3fd39cc715a10126fd92b20f6ce213c380eb4d5df843a42315aac50cf116748a"
EXPERT_ACTOR_SCHEMA_SHA256 = "f72c9e52ac3771bf635e27edd8f08f69b82313d7dc33700162e42740228ce5c6"
MAX_CHAIN_RECEIPT_BYTES = 1024 * 1024
IMPLEMENTATION_LOGICAL_PATH = (
    "src/oracle_composition/experiments/external_tqc_initialization_identity.py"
)
_FIRST_WEIGHT = "latent_pi.0.weight"
_ACTOR_PARAMETER_NAMES = (
    "latent_pi.0.weight",
    "latent_pi.0.bias",
    "latent_pi.2.weight",
    "latent_pi.2.bias",
    "mu.weight",
    "mu.bias",
    "log_std.weight",
    "log_std.bias",
)
_EXPANDED_OUTPUT_NAMES = ("mean", "log_std", "deterministic_action", "seeded_sample")
_ZERO_OUTPUT_NAMES = ("normalized_control", "physical_control")
_EXPECTED_ZERO_OUTPUT_SHA256 = {
    "normalized_control": "b19f9bdbbeffd88b528cc574cf82cc5d71fe13a36361e9f5106375d7a138221e",
    "physical_control": "21db07eb6791ddb669dbb22bc6adc3b76ed6a14cbdb0473d3199135f34040a94",
}
_EXPECTED_EXPANDED_OUTPUT_SHA256 = {
    "mean": "502bebf62b38d3b85cbc653d23460a718edb48a9963f731108da8b3177eab0b8",
    "log_std": "a6ae6c3f96d413e2d08b78a31c8f6880a32bc9ff17c0f4d25e3a44c16c019543",
    "deterministic_action": ("b19f9bdbbeffd88b528cc574cf82cc5d71fe13a36361e9f5106375d7a138221e"),
    "seeded_sample": "423f8fd6d4bb1e3ea116e457e3ecf78b197544f03977f9b71de1b3590242cc49",
}


def _sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"{field} must be a lowercase SHA-256")
    return value


def _implementation_identity() -> dict[str, object]:
    try:
        payload = Path(__file__).resolve(strict=True).read_bytes()
    except OSError as exc:
        raise ExperimentContractError("actual E1 implementation source is unavailable") from exc
    if not 0 < len(payload) <= 1024 * 1024:
        raise ExperimentContractError("actual E1 implementation source size differs")
    return {
        "logical_path": IMPLEMENTATION_LOGICAL_PATH,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_count": len(payload),
    }


def _positive_zero(value: np.ndarray) -> bool:
    return value.tobytes(order="C") == np.zeros_like(value).tobytes(order="C")


def initialize_zero_residual_mean_parameters(
    design: InitializationIdentityDesign,
) -> dict[str, np.ndarray]:
    """Create the exact-zero deterministic mean head for the residual contender."""

    require_canonical_initialization_identity_design(design)
    hidden_width = design.fixture.hidden_layers[-1]
    return {
        "mean.weight": np.zeros((design.fixture.action_dim, hidden_width), dtype="<f4"),
        "mean.bias": np.zeros((design.fixture.action_dim,), dtype="<f4"),
    }


def verify_zero_residual_mean_parameters(
    parameters: Mapping[str, object],
    design: InitializationIdentityDesign,
) -> dict[str, object]:
    """Reject any nonzero, non-float32, or misshapen residual mean parameter."""

    require_canonical_initialization_identity_design(design)
    if type(parameters) is not dict or set(parameters) != {"mean.weight", "mean.bias"}:
        raise ExperimentContractError("zero-residual mean parameter schema differs")
    expected_shapes = {
        "mean.weight": (design.fixture.action_dim, design.fixture.hidden_layers[-1]),
        "mean.bias": (design.fixture.action_dim,),
    }
    receipt: dict[str, object] = {}
    for name, shape in expected_shapes.items():
        value = parameters[name]
        if (
            type(value) is not np.ndarray
            or value.dtype.str != "<f4"
            or value.shape != shape
            or not value.flags.c_contiguous
            or not np.isfinite(value).all()
        ):
            raise ExperimentContractError(f"zero-residual {name} differs")
        if not _positive_zero(value):
            raise ExperimentContractError(f"zero-residual {name} is not exact positive zero")
        receipt[name] = {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "sha256": array_sha256(value),
            "exact_positive_zero": True,
        }
    return receipt


def initialize_expanded_actor_parameters(
    base_parameters: Mapping[str, object],
    design: InitializationIdentityDesign,
) -> dict[str, np.ndarray]:
    """Copy the imported actor and append 8 x 45 exact-zero first-layer columns."""

    require_canonical_initialization_identity_design(design)
    base = validate_actor_arrays(base_parameters)
    reference_width = design.fixture.reference_input_dim
    expanded: dict[str, np.ndarray] = {}
    for name in _ACTOR_PARAMETER_NAMES:
        value = base[name]
        if name == _FIRST_WEIGHT:
            zeros = np.zeros((value.shape[0], reference_width), dtype="<f4")
            expanded[name] = np.ascontiguousarray(
                np.concatenate((value, zeros), axis=1), dtype="<f4"
            )
        else:
            expanded[name] = np.array(value, dtype="<f4", order="C", copy=True)
    verify_expanded_actor_parameters(base, expanded, design)
    return expanded


def verify_expanded_actor_parameters(
    base_parameters: Mapping[str, object],
    expanded_parameters: Mapping[str, object],
    design: InitializationIdentityDesign,
) -> dict[str, object]:
    """Require every copied bit and every appended positive-zero bit."""

    require_canonical_initialization_identity_design(design)
    base = validate_actor_arrays(base_parameters)
    if type(expanded_parameters) is not dict or set(expanded_parameters) != set(
        _ACTOR_PARAMETER_NAMES
    ):
        raise ExperimentContractError("expanded TQC parameter schema differs")
    copied: dict[str, object] = {}
    reference_columns_receipt: dict[str, object] | None = None
    expanded_first_sha256: str | None = None
    for name in _ACTOR_PARAMETER_NAMES:
        source = base[name]
        target = expanded_parameters[name]
        if (
            type(target) is not np.ndarray
            or target.dtype.str != "<f4"
            or not target.flags.c_contiguous
        ):
            raise ExperimentContractError(f"expanded TQC parameter {name} storage differs")
        if not np.isfinite(target).all():
            raise ExperimentContractError(f"expanded TQC parameter {name} is non-finite")
        if name == _FIRST_WEIGHT:
            expected_shape = (
                source.shape[0],
                design.fixture.base_observation_dim + design.fixture.reference_input_dim,
            )
            if target.shape != expected_shape:
                raise ExperimentContractError("expanded TQC first-layer shape differs")
            state_columns = np.ascontiguousarray(
                target[:, : design.fixture.base_observation_dim], dtype="<f4"
            )
            reference_columns = np.ascontiguousarray(
                target[:, design.fixture.base_observation_dim :], dtype="<f4"
            )
            if state_columns.tobytes(order="C") != source.tobytes(order="C"):
                raise ExperimentContractError("expanded TQC copied first-layer bits differ")
            if not _positive_zero(reference_columns):
                raise ExperimentContractError("expanded TQC reference column is nonzero")
            copied[name] = {
                "base_sha256": array_sha256(source),
                "expanded_copy_sha256": array_sha256(state_columns),
                "bitwise_equal": True,
            }
            reference_columns_receipt = {
                "dtype": reference_columns.dtype.str,
                "shape": list(reference_columns.shape),
                "sha256": array_sha256(reference_columns),
                "exact_positive_zero": True,
            }
            expanded_first_sha256 = array_sha256(target)
        else:
            if target.shape != source.shape or target.tobytes(order="C") != source.tobytes(
                order="C"
            ):
                raise ExperimentContractError(f"expanded TQC copied parameter {name} differs")
            copied[name] = {
                "base_sha256": array_sha256(source),
                "expanded_copy_sha256": array_sha256(target),
                "bitwise_equal": True,
            }
    if reference_columns_receipt is None or expanded_first_sha256 is None:
        raise ExperimentContractError("expanded TQC first-layer audit is missing")
    return {
        "first_affine_parameter": _FIRST_WEIGHT,
        "base_first_layer_shape": list(base[_FIRST_WEIGHT].shape),
        "expanded_first_layer_shape": list(expanded_parameters[_FIRST_WEIGHT].shape),
        "reference_window_shape": list(design.fixture.reference_window_shape),
        "reference_column_count": (
            base[_FIRST_WEIGHT].shape[0] * design.fixture.reference_input_dim
        ),
        "reference_columns": reference_columns_receipt,
        "expanded_first_layer_sha256": expanded_first_sha256,
        "copied_parameters": copied,
        "every_copied_parameter_bitwise_equal": True,
        "critics_replay_optimizer_or_rng_transferred": False,
    }


def _actor_outputs(
    parameters: Mapping[str, np.ndarray],
    inputs: np.ndarray,
    *,
    sampling_seed: int,
) -> dict[str, np.ndarray]:
    import torch
    from torch.nn import functional

    tensors = {
        name: torch.from_numpy(np.ascontiguousarray(parameters[name], dtype="<f4").copy())
        for name in _ACTOR_PARAMETER_NAMES
    }
    value = torch.from_numpy(np.ascontiguousarray(inputs, dtype="<f4").copy())
    with torch.inference_mode():
        latent = functional.relu(
            functional.linear(value, tensors["latent_pi.0.weight"], tensors["latent_pi.0.bias"])
        )
        latent = functional.relu(
            functional.linear(
                latent,
                tensors["latent_pi.2.weight"],
                tensors["latent_pi.2.bias"],
            )
        )
        mean = functional.linear(latent, tensors["mu.weight"], tensors["mu.bias"])
        log_std = torch.clamp(
            functional.linear(latent, tensors["log_std.weight"], tensors["log_std.bias"]),
            min=LOG_STD_MIN,
            max=LOG_STD_MAX,
        )
        outputs = {
            "mean": mean,
            "log_std": log_std,
            "deterministic_action": torch.tanh(mean),
            "seeded_sample": seeded_squashed_normal_sample(
                mean,
                log_std,
                sampling_seed=sampling_seed,
            ),
        }
    result = {
        name: np.ascontiguousarray(output.detach().cpu().numpy(), dtype="<f4")
        for name, output in outputs.items()
    }
    for name, output in result.items():
        if output.shape != (4, ACTION_WIDTH) or not np.isfinite(output).all():
            raise ExperimentContractError(f"actual E1 actor {name} output differs")
    return result


def _paired_hashes(
    base: Mapping[str, np.ndarray],
    contender: Mapping[str, np.ndarray],
    names: Sequence[str],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in names:
        left = base[name]
        right = contender[name]
        if left.shape != right.shape or left.dtype.str != right.dtype.str:
            raise ExperimentContractError(f"actual E1 {name} shape or dtype differs")
        if left.tobytes(order="C") != right.tobytes(order="C"):
            raise ExperimentContractError(f"actual E1 {name} is not bitwise identical")
        result[name] = {
            "dtype": left.dtype.str,
            "shape": list(left.shape),
            "base_sha256": array_sha256(left),
            "contender_sha256": array_sha256(right),
            "bitwise_equal": True,
        }
    return result


def _load_chain_receipt(path: Path, *, artifact: str) -> tuple[dict[str, object], str, int]:
    loaded = read_bounded_json_artifact(
        Path(path),
        maximum_bytes=MAX_CHAIN_RECEIPT_BYTES,
        artifact=artifact,
    )
    if type(loaded.value) is not dict:
        raise ExperimentContractError(f"{artifact} must be a JSON object")
    return loaded.value, loaded.sha256, len(loaded.encoded_bytes)


def _build_receipt(
    *,
    design_path: Path,
    actor_npz_path: Path,
    import_receipt_path: Path,
    equivalence_receipt_path: Path,
) -> dict[str, object]:
    loaded_design = load_initialization_identity_design(Path(design_path))
    design = loaded_design.design
    observations = fixture_observation_set(design)
    references = fixture_reference_set(design)

    import_value, import_sha256, import_bytes = _load_chain_receipt(
        import_receipt_path,
        artifact="external expert import receipt",
    )
    validate_external_actor_import_receipt(import_value)
    equivalence_value, equivalence_sha256, equivalence_bytes = _load_chain_receipt(
        equivalence_receipt_path,
        artifact="external expert equivalence receipt",
    )
    validate_external_actor_equivalence_receipt(equivalence_value)
    expected_npz = {
        "logical_path": EXPERT_ACTOR_LOGICAL_PATH,
        "sha256": EXPERT_ACTOR_NPZ_SHA256,
        "byte_count": EXPERT_ACTOR_NPZ_BYTE_COUNT,
        "actor_state_sha256": EXPERT_ACTOR_STATE_SHA256,
        "schema_sha256": EXPERT_ACTOR_SCHEMA_SHA256,
    }
    import_npz = import_value["strict_actor_npz"]
    if (
        import_value["authority"] != AUTHORITY
        or import_value["source"]["policy_pth"]["sha256"] != POLICY_SHA256
        or {key: import_npz[key] for key in expected_npz} != expected_npz
        or equivalence_value["source_variant"] != "expert"
        or equivalence_value["source_policy_sha256"] != POLICY_SHA256
        or equivalence_value["import_receipt_sha256"] != import_sha256
        or equivalence_value["strict_actor_npz"]["sha256"] != EXPERT_ACTOR_NPZ_SHA256
    ):
        raise ExperimentContractError("actual E1 external expert lineage differs")

    actor = load_actor_npz(Path(actor_npz_path), expected_sha256=EXPERT_ACTOR_NPZ_SHA256)
    if (
        actor.byte_count != EXPERT_ACTOR_NPZ_BYTE_COUNT
        or actor.state_sha256 != EXPERT_ACTOR_STATE_SHA256
        or actor.schema_sha256 != EXPERT_ACTOR_SCHEMA_SHA256
    ):
        raise ExperimentContractError("actual E1 strict expert actor identity differs")
    base_parameters = validate_actor_arrays(actor.arrays)
    expanded_parameters = initialize_expanded_actor_parameters(base_parameters, design)
    expanded_audit = verify_expanded_actor_parameters(
        base_parameters,
        expanded_parameters,
        design,
    )
    expanded_inputs = np.ascontiguousarray(
        np.concatenate((observations, references), axis=1),
        dtype="<f4",
    )
    base_outputs = _actor_outputs(
        base_parameters,
        observations,
        sampling_seed=design.fixture.action_sampling_seed,
    )
    expanded_outputs = _actor_outputs(
        expanded_parameters,
        expanded_inputs,
        sampling_seed=design.fixture.action_sampling_seed,
    )
    expanded_pairs = _paired_hashes(base_outputs, expanded_outputs, _EXPANDED_OUTPUT_NAMES)

    residual_parameters = initialize_zero_residual_mean_parameters(design)
    residual_parameter_receipt = verify_zero_residual_mean_parameters(
        residual_parameters,
        design,
    )
    import torch

    residual_mean = torch.zeros((4, ACTION_WIDTH), dtype=torch.float32)
    residual_log_std = torch.full_like(residual_mean, design.fixture.residual_log_std)
    residual_sample = seeded_squashed_normal_sample(
        residual_mean,
        residual_log_std,
        sampling_seed=design.fixture.residual_sampling_seed,
    )
    residual_sample_array = np.ascontiguousarray(residual_sample.cpu().numpy(), dtype="<f4")
    if not np.any(residual_sample_array != np.float32(0.0)):
        raise ExperimentContractError("actual E1 seeded residual sample unexpectedly equals zero")
    zero_residual = np.zeros_like(base_outputs["deterministic_action"])
    composed_normalized = np.ascontiguousarray(
        np.clip(
            base_outputs["deterministic_action"]
            + np.float32(design.fixture.residual_scale) * zero_residual,
            np.float32(-1.0),
            np.float32(1.0),
        ),
        dtype="<f4",
    )
    base_physical = normalized_to_physical(
        base_outputs["deterministic_action"],
        design.action_transform,
    )
    composed_physical = normalized_to_physical(composed_normalized, design.action_transform)
    zero_pairs = _paired_hashes(
        {
            "normalized_control": base_outputs["deterministic_action"],
            "physical_control": base_physical,
        },
        {
            "normalized_control": composed_normalized,
            "physical_control": composed_physical,
        },
        _ZERO_OUTPUT_NAMES,
    )

    return {
        "schema_version": 1,
        "receipt_id": E1_RECEIPT_ID,
        "gate_id": ACTUAL_E1_GATE_ID,
        "authority": AUTHORITY,
        "artifact_class": "external_pretrained_artifact",
        "evidence_class": EVIDENCE_CLASS,
        "source_chain": {
            "source_variant": "expert",
            "source_policy_sha256": POLICY_SHA256,
            "import_receipt_sha256": import_sha256,
            "import_receipt_byte_count": import_bytes,
            "equivalence_receipt_sha256": equivalence_sha256,
            "equivalence_receipt_byte_count": equivalence_bytes,
        },
        "strict_actor_npz": expected_npz,
        "fixture_design": {
            "logical_path": (
                "experiments/bootstrap_tqc_humanoid/configs/"
                "tqc_initialization_transfer_fixture_v0.study.json"
            ),
            "sha256": loaded_design.artifact_sha256,
            "byte_count": loaded_design.artifact_byte_count,
            "semantic_sha256": design.semantic_sha256,
        },
        "observation_set": {
            "dtype": observations.dtype.str,
            "shape": list(observations.shape),
            "sha256": array_sha256(observations),
        },
        "reference_set": {
            "dtype": references.dtype.str,
            "shape": list(references.shape),
            "sha256": array_sha256(references),
            "window_shape": list(design.fixture.reference_window_shape),
        },
        "zero_residual_contender": {
            "method": "base_deterministic_plus_0.08_times_zero_residual_then_clip/v1",
            "residual_scale": design.fixture.residual_scale,
            "mean_parameters": residual_parameter_receipt,
            "deterministic_mean_exact_positive_zero": True,
            "log_std": design.fixture.residual_log_std,
            "sampling_seed": design.fixture.residual_sampling_seed,
            "seeded_residual_sample": array_receipt(residual_sample_array),
            "seeded_residual_sample_nonzero": True,
            "paired_output_sha256": zero_pairs,
            "bitwise_identity_passed": True,
        },
        "expanded_tqc_contender": {
            "method": "copy_actor_and_append_exact_zero_reference_columns/v1",
            "sampling_seed": design.fixture.action_sampling_seed,
            "parameter_audit": expanded_audit,
            "paired_output_sha256": expanded_pairs,
            "bitwise_identity_passed": True,
        },
        "implementation": _implementation_identity(),
        "passed": True,
        "training_steps": 0,
        "environment_steps": 0,
        "behavior_evaluated": False,
        "eligible_for_local_training": False,
        "eligible_for_tracker_admission": False,
        "claim": {
            "establishes": (
                "initialization identity for both ADR 0004 contenders on the exact imported "
                "external expert actor"
            ),
            "does_not_establish": [
                "local-training authority",
                "Humanoid behavior",
                "tracker admission",
                "E2",
                "E3",
                "oracle quality",
            ],
        },
    }


def _validate_pair_map(value: object, *, names: Sequence[str], field: str) -> None:
    if type(value) is not dict or set(value) != set(names):
        raise ExperimentContractError(f"{field} paired output fields differ")
    for name in names:
        pair = value[name]
        if (
            type(pair) is not dict
            or set(pair) != {"dtype", "shape", "base_sha256", "contender_sha256", "bitwise_equal"}
            or pair["dtype"] != "<f4"
            or pair["shape"] != [4, ACTION_WIDTH]
            or pair["bitwise_equal"] is not True
            or pair["base_sha256"] != pair["contender_sha256"]
        ):
            raise ExperimentContractError(f"{field} {name} identity differs")
        _sha256(pair["base_sha256"], field=f"{field} {name} output")


def validate_external_initialization_identity_receipt(value: object) -> dict[str, object]:
    """Validate an actual E1 receipt without granting any local-training authority."""

    required = {
        "schema_version",
        "receipt_id",
        "gate_id",
        "authority",
        "artifact_class",
        "evidence_class",
        "source_chain",
        "strict_actor_npz",
        "fixture_design",
        "observation_set",
        "reference_set",
        "zero_residual_contender",
        "expanded_tqc_contender",
        "implementation",
        "passed",
        "training_steps",
        "environment_steps",
        "behavior_evaluated",
        "eligible_for_local_training",
        "eligible_for_tracker_admission",
        "claim",
    }
    if type(value) is not dict or set(value) != required:
        raise ExperimentContractError("actual E1 receipt fields differ")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or value["receipt_id"] != E1_RECEIPT_ID
        or value["gate_id"] != ACTUAL_E1_GATE_ID
        or value["authority"] != AUTHORITY
        or value["artifact_class"] != "external_pretrained_artifact"
        or value["evidence_class"] != EVIDENCE_CLASS
        or value["passed"] is not True
        or type(value["training_steps"]) is not int
        or value["training_steps"] != 0
        or type(value["environment_steps"]) is not int
        or value["environment_steps"] != 0
        or value["behavior_evaluated"] is not False
        or value["eligible_for_local_training"] is not False
        or value["eligible_for_tracker_admission"] is not False
    ):
        raise ExperimentContractError("actual E1 receipt identity or authority differs")
    source = value["source_chain"]
    if (
        type(source) is not dict
        or set(source)
        != {
            "source_variant",
            "source_policy_sha256",
            "import_receipt_sha256",
            "import_receipt_byte_count",
            "equivalence_receipt_sha256",
            "equivalence_receipt_byte_count",
        }
        or source["source_variant"] != "expert"
        or source["source_policy_sha256"] != POLICY_SHA256
        or type(source["import_receipt_byte_count"]) is not int
        or source["import_receipt_byte_count"] <= 0
        or type(source["equivalence_receipt_byte_count"]) is not int
        or source["equivalence_receipt_byte_count"] <= 0
    ):
        raise ExperimentContractError("actual E1 source chain differs")
    _sha256(source["import_receipt_sha256"], field="actual E1 import receipt")
    _sha256(source["equivalence_receipt_sha256"], field="actual E1 equivalence receipt")
    expected_npz = {
        "logical_path": EXPERT_ACTOR_LOGICAL_PATH,
        "sha256": EXPERT_ACTOR_NPZ_SHA256,
        "byte_count": EXPERT_ACTOR_NPZ_BYTE_COUNT,
        "actor_state_sha256": EXPERT_ACTOR_STATE_SHA256,
        "schema_sha256": EXPERT_ACTOR_SCHEMA_SHA256,
    }
    if value["strict_actor_npz"] != expected_npz:
        raise ExperimentContractError("actual E1 NPZ binding differs")
    design = value["fixture_design"]
    if (
        type(design) is not dict
        or set(design) != {"logical_path", "sha256", "byte_count", "semantic_sha256"}
        or design["logical_path"]
        != (
            "experiments/bootstrap_tqc_humanoid/configs/"
            "tqc_initialization_transfer_fixture_v0.study.json"
        )
        or type(design["byte_count"]) is not int
        or design["byte_count"] != DESIGN_ARTIFACT_BYTE_COUNT
        or design["sha256"] != DESIGN_ARTIFACT_SHA256
        or design["semantic_sha256"]
        != "9a601545eacb2ebc55623084cf9be08b76ac858827eabb5e05423be02767e9a3"
    ):
        raise ExperimentContractError("actual E1 fixture design binding differs")
    _sha256(design["sha256"], field="actual E1 fixture design")
    _sha256(design["semantic_sha256"], field="actual E1 fixture design semantic hash")
    observation = value["observation_set"]
    if (
        type(observation) is not dict
        or set(observation) != {"dtype", "shape", "sha256"}
        or observation["dtype"] != "<f4"
        or observation["shape"] != [4, OBSERVATION_WIDTH]
        or observation["sha256"]
        != "0c6a81b06a88cab7eca0255e75f021008b60025c4ddc4d3719426e3647159ec6"
    ):
        raise ExperimentContractError("actual E1 observation-set binding differs")
    reference = value["reference_set"]
    if (
        type(reference) is not dict
        or set(reference) != {"dtype", "shape", "sha256", "window_shape"}
        or reference["dtype"] != "<f4"
        or reference["shape"] != [4, 360]
        or reference["window_shape"] != [8, 45]
        or reference["sha256"] != "ffce030803e735c4d763658df320fec57810fd7c42811797152977538e012d1a"
    ):
        raise ExperimentContractError("actual E1 reference-set binding differs")

    zero = value["zero_residual_contender"]
    if (
        type(zero) is not dict
        or set(zero)
        != {
            "method",
            "residual_scale",
            "mean_parameters",
            "deterministic_mean_exact_positive_zero",
            "log_std",
            "sampling_seed",
            "seeded_residual_sample",
            "seeded_residual_sample_nonzero",
            "paired_output_sha256",
            "bitwise_identity_passed",
        }
        or zero["method"] != "base_deterministic_plus_0.08_times_zero_residual_then_clip/v1"
        or type(zero["residual_scale"]) is not float
        or zero["residual_scale"] != 0.08
        or zero["deterministic_mean_exact_positive_zero"] is not True
        or type(zero["log_std"]) is not float
        or zero["log_std"] != -3.0
        or type(zero["sampling_seed"]) is not int
        or zero["sampling_seed"] != 93_004
        or zero["seeded_residual_sample_nonzero"] is not True
        or zero["bitwise_identity_passed"] is not True
    ):
        raise ExperimentContractError("actual E1 zero-residual record differs")
    mean_parameters = zero["mean_parameters"]
    if type(mean_parameters) is not dict or set(mean_parameters) != {"mean.weight", "mean.bias"}:
        raise ExperimentContractError("actual E1 residual parameter record differs")
    for name, shape in (("mean.weight", [17, 256]), ("mean.bias", [17])):
        item = mean_parameters[name]
        if (
            type(item) is not dict
            or set(item) != {"dtype", "shape", "sha256", "exact_positive_zero"}
            or item["dtype"] != "<f4"
            or item["shape"] != shape
            or item["exact_positive_zero"] is not True
        ):
            raise ExperimentContractError(f"actual E1 residual {name} receipt differs")
        _sha256(item["sha256"], field=f"actual E1 residual {name}")
        expected_zero = np.zeros(tuple(shape), dtype="<f4")
        if item["sha256"] != array_sha256(expected_zero):
            raise ExperimentContractError(f"actual E1 residual {name} zero hash differs")
    sample = zero["seeded_residual_sample"]
    if (
        type(sample) is not dict
        or set(sample) != {"dtype", "shape", "sha256", "values"}
        or sample["dtype"] != "<f4"
        or sample["shape"] != [4, ACTION_WIDTH]
    ):
        raise ExperimentContractError("actual E1 seeded residual sample differs")
    try:
        sample_array = np.ascontiguousarray(np.asarray(sample["values"], dtype="<f4"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExperimentContractError("actual E1 seeded residual values differ") from exc
    if (
        sample_array.shape != (4, ACTION_WIDTH)
        or not np.isfinite(sample_array).all()
        or not np.any(sample_array != np.float32(0.0))
        or array_sha256(sample_array) != sample["sha256"]
    ):
        raise ExperimentContractError("actual E1 seeded residual sample hash differs")
    _validate_pair_map(
        zero["paired_output_sha256"], names=_ZERO_OUTPUT_NAMES, field="zero residual"
    )
    for name, digest in _EXPECTED_ZERO_OUTPUT_SHA256.items():
        if zero["paired_output_sha256"][name]["base_sha256"] != digest:
            raise ExperimentContractError(f"actual E1 zero-residual {name} hash differs")

    expanded = value["expanded_tqc_contender"]
    if (
        type(expanded) is not dict
        or set(expanded)
        != {
            "method",
            "sampling_seed",
            "parameter_audit",
            "paired_output_sha256",
            "bitwise_identity_passed",
        }
        or expanded["method"] != "copy_actor_and_append_exact_zero_reference_columns/v1"
        or type(expanded["sampling_seed"]) is not int
        or expanded["sampling_seed"] != 93_003
        or expanded["bitwise_identity_passed"] is not True
    ):
        raise ExperimentContractError("actual E1 expanded-TQC record differs")
    audit = expanded["parameter_audit"]
    if (
        type(audit) is not dict
        or set(audit)
        != {
            "first_affine_parameter",
            "base_first_layer_shape",
            "expanded_first_layer_shape",
            "reference_window_shape",
            "reference_column_count",
            "reference_columns",
            "expanded_first_layer_sha256",
            "copied_parameters",
            "every_copied_parameter_bitwise_equal",
            "critics_replay_optimizer_or_rng_transferred",
        }
        or audit["first_affine_parameter"] != _FIRST_WEIGHT
        or audit["base_first_layer_shape"] != [256, 348]
        or audit["expanded_first_layer_shape"] != [256, 708]
        or audit["reference_window_shape"] != [8, 45]
        or type(audit["reference_column_count"]) is not int
        or audit["reference_column_count"] != 256 * 360
        or audit["every_copied_parameter_bitwise_equal"] is not True
        or audit["critics_replay_optimizer_or_rng_transferred"] is not False
    ):
        raise ExperimentContractError("actual E1 expanded parameter audit differs")
    reference_columns = audit["reference_columns"]
    if (
        type(reference_columns) is not dict
        or set(reference_columns) != {"dtype", "shape", "sha256", "exact_positive_zero"}
        or reference_columns["dtype"] != "<f4"
        or reference_columns["shape"] != [256, 360]
        or reference_columns["exact_positive_zero"] is not True
    ):
        raise ExperimentContractError("actual E1 appended reference columns differ")
    _sha256(reference_columns["sha256"], field="actual E1 reference columns")
    if reference_columns["sha256"] != array_sha256(np.zeros((256, 360), dtype="<f4")):
        raise ExperimentContractError("actual E1 appended reference-column zero hash differs")
    _sha256(audit["expanded_first_layer_sha256"], field="actual E1 expanded first layer")
    copied = audit["copied_parameters"]
    if type(copied) is not dict or set(copied) != set(_ACTOR_PARAMETER_NAMES):
        raise ExperimentContractError("actual E1 copied parameter fields differ")
    for name in _ACTOR_PARAMETER_NAMES:
        item = copied[name]
        if (
            type(item) is not dict
            or set(item) != {"base_sha256", "expanded_copy_sha256", "bitwise_equal"}
            or item["base_sha256"] != item["expanded_copy_sha256"]
            or item["bitwise_equal"] is not True
        ):
            raise ExperimentContractError(f"actual E1 copied parameter {name} differs")
        _sha256(item["base_sha256"], field=f"actual E1 copied parameter {name}")
    _validate_pair_map(
        expanded["paired_output_sha256"],
        names=_EXPANDED_OUTPUT_NAMES,
        field="expanded TQC",
    )
    for name, digest in _EXPECTED_EXPANDED_OUTPUT_SHA256.items():
        if expanded["paired_output_sha256"][name]["base_sha256"] != digest:
            raise ExperimentContractError(f"actual E1 expanded-TQC {name} hash differs")
    if value["implementation"] != _implementation_identity():
        raise ExperimentContractError("actual E1 implementation identity differs")
    expected_claim = {
        "establishes": (
            "initialization identity for both ADR 0004 contenders on the exact imported "
            "external expert actor"
        ),
        "does_not_establish": [
            "local-training authority",
            "Humanoid behavior",
            "tracker admission",
            "E2",
            "E3",
            "oracle quality",
        ],
    }
    if value["claim"] != expected_claim:
        raise ExperimentContractError("actual E1 claim ceiling differs")
    return value


@dataclass(frozen=True, slots=True)
class ExternalInitializationIdentityResult:
    published: PublishedArtifact
    receipt: dict[str, object]


def run_external_initialization_identity(
    *,
    design_path: Path,
    actor_npz_path: Path,
    import_receipt_path: Path,
    equivalence_receipt_path: Path,
    output_path: Path,
) -> ExternalInitializationIdentityResult:
    """Execute both no-training E1 initializers and publish one exclusive receipt."""

    receipt = _build_receipt(
        design_path=Path(design_path),
        actor_npz_path=Path(actor_npz_path),
        import_receipt_path=Path(import_receipt_path),
        equivalence_receipt_path=Path(equivalence_receipt_path),
    )
    validate_external_initialization_identity_receipt(receipt)
    published = publish_json_without_overwrite(Path(output_path), receipt)
    return ExternalInitializationIdentityResult(published=published, receipt=receipt)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--actor-npz", type=Path, required=True)
    parser.add_argument("--import-receipt", type=Path, required=True)
    parser.add_argument("--equivalence-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_external_initialization_identity(
            design_path=args.design,
            actor_npz_path=args.actor_npz,
            import_receipt_path=args.import_receipt,
            equivalence_receipt_path=args.equivalence_receipt,
            output_path=args.output,
        )
    except (ExperimentContractError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "path": str(result.published.path),
                "sha256": result.published.sha256,
                "byte_count": result.published.byte_count,
                "passed": result.receipt["passed"],
                "evidence_class": result.receipt["evidence_class"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "E1_RECEIPT_ID",
    "EXPERT_ACTOR_NPZ_SHA256",
    "ExternalInitializationIdentityResult",
    "initialize_expanded_actor_parameters",
    "initialize_zero_residual_mean_parameters",
    "main",
    "run_external_initialization_identity",
    "validate_external_initialization_identity_receipt",
    "verify_expanded_actor_parameters",
    "verify_zero_residual_mean_parameters",
]


if __name__ == "__main__":
    raise SystemExit(main())
