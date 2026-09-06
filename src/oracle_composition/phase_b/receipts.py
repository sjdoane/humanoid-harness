"""No-learning E1 and static phase-transfer receipt generators."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import (
    array_sha256,
    canonical_json_bytes,
    sha256_file,
)
from oracle_composition.experiments.artifact_io import (
    PublishedArtifact,
    publish_bytes_without_overwrite,
)
from oracle_composition.experiments.external_tqc_initialization_identity import (
    EXPERT_ACTOR_NPZ_BYTE_COUNT,
    EXPERT_ACTOR_NPZ_SHA256,
    EXPERT_ACTOR_SCHEMA_SHA256,
    EXPERT_ACTOR_STATE_SHA256,
)
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_actor_npz import load_actor_npz
from oracle_composition.experiments.tqc_initialization_identity_adapter import (
    fixture_observation_set,
    fixture_reference_set,
)
from oracle_composition.experiments.tqc_initialization_identity_contract import (
    load_initialization_identity_design,
)

from .contracts import EVIDENCE_CLASS
from .policy import (
    ACTION_ADAPTER_ID,
    LOG_STD_MAX,
    LOG_STD_MIN,
    FullAuthorityActor,
    build_full_authority_policy,
    compose_policy_input,
    encode_full_authority_actor,
    exact_physical_action,
    export_full_authority_actor,
    load_full_authority_actor,
    load_full_authority_actor_arrays,
    verify_actor_warm_start,
    verify_optimizer_authority,
)
from .reference_runtime import (
    load_v2_reference_clip,
    normalized_tracking_errors,
    select_nearest_phase,
    tracking_state_from_reference_row,
)

E1_RECEIPT_ID = "full_authority_warm_start_e1/v1"
PHASE_TRANSFER_RECEIPT_ID = "switch_only_phase_transfer_static_check/v1"
E1_SAMPLING_SEED = 20_260_905
E1_VALUE_SEED = 20_260_905
ADMITTED_TRAINING_BLOCKS = (
    120001,
    120002,
    120003,
    120005,
    120007,
    120008,
    120009,
    120011,
    120012,
)
SYNTHETIC_DESIGN_PATH = (
    "experiments/bootstrap_tqc_humanoid/configs/tqc_initialization_transfer_fixture_v0.study.json"
)
EXPERT_ACTOR_PATH = "artifacts/bootstrap_tqc_humanoid/farama_minari_humanoid_v5_tqc_actor_v1.npz"
CORPUS_PATH = "artifacts/reference_corpus_v2"


def sha_ranked_real_fixtures(
    *,
    blocks: Sequence[int] = ADMITTED_TRAINING_BLOCKS,
    sampling_seed: int = E1_SAMPLING_SEED,
    count: int = 64,
) -> tuple[tuple[int, int, str], ...]:
    if tuple(blocks) != ADMITTED_TRAINING_BLOCKS:
        raise ExperimentContractError("E1 admitted training block list differs")
    if sampling_seed != E1_SAMPLING_SEED or count != 64:
        raise ExperimentContractError("E1 fixture sampling design differs")
    candidates: list[tuple[str, int, int]] = []
    for block in blocks:
        for boundary in range(1001):
            key = f"full_authority_e1_fixture/v1\0{sampling_seed}\0{block}\0{boundary}"
            digest = hashlib.sha256(key.encode("ascii")).hexdigest()
            candidates.append((digest, block, boundary))
    ranked = sorted(candidates)[:count]
    return tuple((block, boundary, digest) for digest, block, boundary in ranked)


def _input_batch(repository_root: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    design = load_initialization_identity_design(repository_root / SYNTHETIC_DESIGN_PATH).design
    synthetic_states = fixture_observation_set(design)
    synthetic_windows = fixture_reference_set(design).reshape(4, 8, 45)
    selected = sha_ranked_real_fixtures()
    clips = {
        block: load_v2_reference_clip(
            repository_root / CORPUS_PATH,
            block=block,
            behavior="expert",
        )
        for block in ADMITTED_TRAINING_BLOCKS
    }
    states: list[np.ndarray] = [row for row in synthetic_states]
    windows: list[np.ndarray] = [row for row in synthetic_windows]
    fixture_records: list[dict[str, object]] = []
    for block, boundary, rank_sha256 in selected:
        clip = clips[block]
        state = np.ascontiguousarray(clip.boundary_observations[boundary], dtype="<f4")
        indices = tuple(min(boundary + offset, 1000) for offset in range(8))
        window = np.ascontiguousarray(clip.reference_rows[list(indices)], dtype="<f4")
        states.append(state)
        windows.append(window)
        fixture_records.append(
            {
                "block": block,
                "boundary": boundary,
                "bundle_sha256": clip.bundle_sha256,
                "observation_sha256": array_sha256(state),
                "rank_sha256": rank_sha256,
                "reference_window_indices": list(indices),
                "reference_window_sha256": array_sha256(window),
            }
        )
    state_batch = np.ascontiguousarray(np.stack(states), dtype="<f4")
    window_batch = np.ascontiguousarray(np.stack(windows), dtype="<f4")
    if state_batch.shape != (68, 348) or window_batch.shape != (68, 8, 45):
        raise ExperimentContractError("E1 fixture batch shape differs")
    return state_batch, window_batch, fixture_records


def _base_distribution(
    parameters: Mapping[str, np.ndarray],
    states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch.nn import functional

    tensors = {
        name: torch.from_numpy(np.ascontiguousarray(value, dtype="<f4").copy())
        for name, value in parameters.items()
        if name.startswith(("latent_pi", "mu", "log_std"))
    }
    value = torch.from_numpy(states.copy(order="C"))
    with torch.inference_mode():
        latent = functional.relu(
            functional.linear(
                value,
                tensors["latent_pi.0.weight"],
                tensors["latent_pi.0.bias"],
            )
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
            functional.linear(
                latent,
                tensors["log_std.weight"],
                tensors["log_std.bias"],
            ),
            min=LOG_STD_MIN,
            max=LOG_STD_MAX,
        )
    return tuple(
        np.ascontiguousarray(item.detach().cpu().numpy(), dtype="<f4") for item in (mean, log_std)
    )  # type: ignore[return-value]


def _epsilon(shape: tuple[int, ...]) -> np.ndarray:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(E1_SAMPLING_SEED)
    value = torch.randn(shape, dtype=torch.float32, generator=generator)
    return np.ascontiguousarray(value.numpy(), dtype="<f4")


def _require_bitwise(left: np.ndarray, right: np.ndarray, *, field: str) -> dict[str, object]:
    if (
        left.dtype.str != right.dtype.str
        or left.shape != right.shape
        or left.tobytes(order="C") != right.tobytes(order="C")
    ):
        delta = np.max(
            np.abs(left.astype(np.float64) - right.astype(np.float64)),
            initial=0.0,
        )
        raise ExperimentContractError(f"E1 {field} differs; max_abs={float(delta)!r}")
    return {
        "bitwise_equal": True,
        "dtype": left.dtype.str,
        "sha256": array_sha256(left),
        "shape": list(left.shape),
    }


def _expanded_expected(parameters: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {
        name: np.array(parameters[name], dtype="<f4", order="C", copy=True)
        for name in (
            "latent_pi.0.bias",
            "latent_pi.2.weight",
            "latent_pi.2.bias",
            "mu.weight",
            "mu.bias",
            "log_std.weight",
            "log_std.bias",
        )
    }
    result["latent_pi.0.weight"] = np.ascontiguousarray(
        np.concatenate(
            (
                parameters["latent_pi.0.weight"],
                np.zeros((256, 360), dtype="<f4"),
            ),
            axis=1,
        ),
        dtype="<f4",
    )
    return result


def _export_distribution(
    parameters: Mapping[str, np.ndarray],
    policy_input: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replay the actor math from authenticated arrays without constructing a model."""

    import torch
    from torch.nn import functional

    tensors = {
        name: torch.from_numpy(np.ascontiguousarray(value, dtype="<f4").copy())
        for name, value in parameters.items()
        if name.startswith(("latent_pi", "mu", "log_std"))
    }
    value = torch.from_numpy(policy_input.copy(order="C"))
    with torch.inference_mode():
        first = functional.linear(
            value[..., :348],
            tensors["latent_pi.0.weight"][:, :348],
            tensors["latent_pi.0.bias"],
        ) + functional.linear(
            value[..., 348:],
            tensors["latent_pi.0.weight"][:, 348:],
            None,
        )
        latent = functional.relu(first)
        latent = functional.relu(
            functional.linear(
                latent,
                tensors["latent_pi.2.weight"],
                tensors["latent_pi.2.bias"],
            )
        )
        mean = functional.linear(latent, tensors["mu.weight"], tensors["mu.bias"])
        log_std = torch.clamp(
            functional.linear(
                latent,
                tensors["log_std.weight"],
                tensors["log_std.bias"],
            ),
            min=LOG_STD_MIN,
            max=LOG_STD_MAX,
        )
    return tuple(np.ascontiguousarray(item.numpy(), dtype="<f4") for item in (mean, log_std))  # type: ignore[return-value]


def validate_e1_receipt_at_admission(
    *,
    repository_root: Path,
    receipt_path: Path,
    expected_receipt_sha256: str,
    export_path: Path,
    expected_export_sha256: str,
) -> dict[str, object]:
    """Bind live sources and replay all 68 E1 fixtures before model construction."""

    root = Path(repository_root)
    encoded = Path(receipt_path).read_bytes()
    if hashlib.sha256(encoded).hexdigest() != expected_receipt_sha256:
        raise ExperimentContractError("E1 admission receipt identity differs")
    try:
        receipt = json.loads(encoded)
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise ExperimentContractError("E1 admission receipt is invalid") from exc
    if type(receipt) is not dict or canonical_json_bytes(receipt) != encoded:
        raise ExperimentContractError("E1 admission receipt is not canonical")
    live_source_hashes = {
        "policy": sha256_file(Path(sys.modules[FullAuthorityActor.__module__].__file__)),
        "receipt_generator": sha256_file(Path(__file__)),
    }
    if receipt.get("source_hashes") != live_source_hashes:
        raise ExperimentContractError("E1 admission source hashes differ from live modules")
    source = receipt.get("source_expert")
    export = receipt.get("export")
    if (
        type(source) is not dict
        or source.get("sha256") != EXPERT_ACTOR_NPZ_SHA256
        or type(export) is not dict
        or export.get("sha256") != expected_export_sha256
        or receipt.get("passed") is not True
        or receipt.get("training_steps") != 0
    ):
        raise ExperimentContractError("E1 admission receipt bindings differ")
    optimizer = receipt.get("optimizer_admission")
    expected_names = [
        "actor.latent_0.weight",
        "actor.latent_0.bias",
        "actor.latent_2.weight",
        "actor.latent_2.bias",
        "actor.mu.weight",
        "actor.mu.bias",
        "actor.log_std.weight",
        "actor.log_std.bias",
        "value.0.weight",
        "value.0.bias",
        "value.2.weight",
        "value.2.bias",
        "value.4.weight",
        "value.4.bias",
    ]
    expected_optimizer = {
        "group_count": 1,
        "groups": [
            {
                "hyperparameters": {
                    "amsgrad": False,
                    "betas": [0.9, 0.999],
                    "capturable": False,
                    "decoupled_weight_decay": False,
                    "differentiable": False,
                    "eps": 1e-8,
                    "foreach": None,
                    "fused": None,
                    "lr": 0.0003,
                    "maximize": False,
                    "weight_decay": 0,
                },
                "parameter_names": expected_names,
            }
        ],
        "membership_complete": True,
        "membership_sha256": hashlib.sha256("\0".join(expected_names).encode()).hexdigest(),
        "optimizer_type": "torch.optim.Adam",
        "parameter_count": len(expected_names),
        "parameter_membership_unique": True,
        "stage": "e1_fresh_before_first_update",
        "state_empty": True,
        "state_entry_count": 0,
    }
    if optimizer != expected_optimizer:
        raise ExperimentContractError("E1 fresh optimizer admission differs")
    states, windows, real_fixtures = _input_batch(root)
    policy_input = compose_policy_input(states, windows).array
    fixture = receipt.get("fixture_batch")
    if (
        type(fixture) is not dict
        or fixture.get("count") != 68
        or fixture.get("real_fixtures") != real_fixtures
        or fixture.get("state_sha256") != array_sha256(states)
        or fixture.get("window_sha256") != array_sha256(windows)
        or fixture.get("input_sha256") != array_sha256(policy_input)
    ):
        raise ExperimentContractError("E1 admission fixture batch differs")
    expert = load_actor_npz(root / EXPERT_ACTOR_PATH, expected_sha256=EXPERT_ACTOR_NPZ_SHA256)
    exported = load_full_authority_actor_arrays(
        export_path,
        expected_sha256=expected_export_sha256,
    )
    parameters = {name: exported[name] for name in _expanded_expected(expert.arrays)}
    expected_parameters = _expanded_expected(expert.arrays)
    if any(
        parameters[name].tobytes(order="C") != expected_parameters[name].tobytes(order="C")
        for name in expected_parameters
    ):
        raise ExperimentContractError("E1 admission exported parameters differ")
    base_mean, base_log_std = _base_distribution(expert.arrays, states)
    mean, log_std = _export_distribution(parameters, policy_input)
    if mean.tobytes(order="C") != base_mean.tobytes(order="C") or log_std.tobytes(
        order="C"
    ) != base_log_std.tobytes(order="C"):
        raise ExperimentContractError("E1 admission distribution replay differs")
    epsilon = _epsilon((68, 17))
    import torch

    with torch.inference_mode():
        deterministic = np.ascontiguousarray(torch.tanh(torch.from_numpy(mean.copy())).numpy())
        pre_tanh = torch.from_numpy(mean.copy()) + torch.exp(torch.from_numpy(log_std.copy())) * (
            torch.from_numpy(epsilon.copy())
        )
        stochastic = np.ascontiguousarray(torch.tanh(pre_tanh).numpy(), dtype="<f4")
    low = np.ascontiguousarray(expert.arrays["action_low"], dtype="<f4")
    high = np.ascontiguousarray(expert.arrays["action_high"], dtype="<f4")
    replay = {
        "deterministic_normalized_action": deterministic,
        "deterministic_physical_action": exact_physical_action(deterministic, low, high),
        "log_std": log_std,
        "mean": mean,
        "seeded_stochastic_normalized_action": stochastic,
        "seeded_stochastic_physical_action": exact_physical_action(stochastic, low, high),
    }
    identity = receipt.get("identity_checks")
    if type(identity) is not dict or any(
        type(identity.get(name)) is not dict
        or identity[name].get("sha256") != array_sha256(value)
        or identity[name].get("bitwise_equal") is not True
        for name, value in replay.items()
    ):
        raise ExperimentContractError("E1 admission action replay differs")
    return {
        "fixture_count": 68,
        "receipt_sha256": expected_receipt_sha256,
        "replayed": True,
        "source_hashes": live_source_hashes,
    }


def build_e1_receipt(
    *,
    repository_root: Path,
    export_path: Path,
) -> dict[str, object]:
    """Execute the complete 68-fixture no-learning E1 identity audit."""

    root = Path(repository_root)
    actor_path = root / EXPERT_ACTOR_PATH
    if sha256_file(actor_path) != EXPERT_ACTOR_NPZ_SHA256:
        raise ExperimentContractError("expert NPZ hash mismatch")
    expert = load_actor_npz(actor_path, expected_sha256=EXPERT_ACTOR_NPZ_SHA256)
    if (
        expert.byte_count != EXPERT_ACTOR_NPZ_BYTE_COUNT
        or expert.state_sha256 != EXPERT_ACTOR_STATE_SHA256
        or expert.schema_sha256 != EXPERT_ACTOR_SCHEMA_SHA256
    ):
        raise ExperimentContractError("expert NPZ identity mismatch")
    policy = build_full_authority_policy(actor_path, value_seed=E1_VALUE_SEED)
    import torch

    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    optimizer_admission = verify_optimizer_authority(
        policy,
        optimizer,
        expected_learning_rate=3e-4,
        require_empty_state=True,
        stage="e1_fresh_before_first_update",
    )
    expected_parameters = _expanded_expected(expert.arrays)
    parameter_audit = verify_actor_warm_start(policy.actor, expected_parameters)
    states, windows, real_fixtures = _input_batch(root)
    policy_input = compose_policy_input(states, windows)
    base_mean, base_log_std = _base_distribution(expert.arrays, states)
    contender_distribution = policy.actor.distribution(policy_input)
    equality: dict[str, object] = {
        "mean": _require_bitwise(
            base_mean,
            contender_distribution.mean,
            field="mean",
        ),
        "log_std": _require_bitwise(
            base_log_std,
            contender_distribution.log_std,
            field="log_std",
        ),
    }
    with torch.inference_mode():
        base_deterministic = np.ascontiguousarray(
            torch.tanh(torch.from_numpy(base_mean.copy(order="C"))).numpy(),
            dtype="<f4",
        )
    low = np.ascontiguousarray(expert.arrays["action_low"], dtype="<f4")
    high = np.ascontiguousarray(expert.arrays["action_high"], dtype="<f4")
    base_deterministic_physical = exact_physical_action(base_deterministic, low, high)
    contender_deterministic = policy.actor.act(policy_input)
    equality["deterministic_normalized_action"] = _require_bitwise(
        base_deterministic,
        contender_deterministic.normalized,
        field="deterministic normalized action",
    )
    equality["deterministic_physical_action"] = _require_bitwise(
        base_deterministic_physical,
        contender_deterministic.physical,
        field="deterministic physical action",
    )
    epsilon = _epsilon((68, 17))
    with torch.inference_mode():
        base_pre_tanh_tensor = torch.from_numpy(base_mean.copy(order="C")) + torch.exp(
            torch.from_numpy(base_log_std.copy(order="C"))
        ) * torch.from_numpy(epsilon.copy(order="C"))
        base_stochastic = np.ascontiguousarray(
            torch.tanh(base_pre_tanh_tensor).numpy(),
            dtype="<f4",
        )
    base_stochastic_physical = exact_physical_action(base_stochastic, low, high)
    contender_stochastic = policy.actor.act(policy_input, epsilon=epsilon)
    equality["seeded_stochastic_normalized_action"] = _require_bitwise(
        base_stochastic,
        contender_stochastic.normalized,
        field="seeded stochastic normalized action",
    )
    equality["seeded_stochastic_physical_action"] = _require_bitwise(
        base_stochastic_physical,
        contender_stochastic.physical,
        field="seeded stochastic physical action",
    )
    torch_likelihood = policy.actor.log_likelihood(
        policy_input,
        contender_stochastic.pre_tanh,
    )
    with torch.inference_mode():
        audit_pre_tanh = torch.from_numpy(contender_stochastic.pre_tanh.copy(order="C"))
        audit_mean = torch.from_numpy(contender_stochastic.mean.copy(order="C"))
        audit_log_std = torch.from_numpy(contender_stochastic.log_std.copy(order="C"))
        audit_squashed = torch.tanh(audit_pre_tanh)
        independently_recomputed = torch.distributions.Normal(
            audit_mean, torch.exp(audit_log_std)
        ).log_prob(audit_pre_tanh).sum(dim=-1) - torch.log(
            1.0 - audit_squashed.square() + 1e-6
        ).sum(dim=-1)
    recomputed_likelihood = np.ascontiguousarray(independently_recomputed.numpy(), dtype="<f4")
    likelihood_delta = float(
        np.max(
            np.abs(torch_likelihood.astype(np.float64) - recomputed_likelihood.astype(np.float64)),
            initial=0.0,
        )
    )
    if not math.isfinite(likelihood_delta) or likelihood_delta > 1e-5:
        raise ExperimentContractError(
            f"PPO likelihood recomputation exceeded 1e-5: {likelihood_delta!r}"
        )
    export_payload = encode_full_authority_actor(policy.actor)
    if Path(export_path).exists():
        if Path(export_path).is_symlink() or Path(export_path).read_bytes() != export_payload:
            raise ExperimentContractError("sealed E1 export differs from the regenerated bytes")
        published_export = PublishedArtifact(
            Path(export_path),
            hashlib.sha256(export_payload).hexdigest(),
            len(export_payload),
        )
    else:
        published_export = export_full_authority_actor(export_path, policy.actor)
    reloaded = load_full_authority_actor(
        published_export.path,
        expected_sha256=published_export.sha256,
    )
    reload_distribution = reloaded.actor.distribution(policy_input)
    reload_deterministic = reloaded.actor.act(policy_input)
    reload_stochastic = reloaded.actor.act(policy_input, epsilon=epsilon)
    reloaded_parameters = reloaded.actor.parameter_arrays()
    reload_parameter_checks = {
        name: _require_bitwise(
            policy.actor.parameter_arrays()[name],
            reloaded_parameters[name],
            field=f"reload parameter {name}",
        )
        for name in policy.actor.parameter_arrays()
    }
    reload_equality = {
        "mean": _require_bitwise(
            contender_distribution.mean,
            reload_distribution.mean,
            field="reload mean",
        ),
        "log_std": _require_bitwise(
            contender_distribution.log_std,
            reload_distribution.log_std,
            field="reload log_std",
        ),
        "deterministic_normalized_action": _require_bitwise(
            contender_deterministic.normalized,
            reload_deterministic.normalized,
            field="reload deterministic normalized action",
        ),
        "deterministic_physical_action": _require_bitwise(
            contender_deterministic.physical,
            reload_deterministic.physical,
            field="reload deterministic physical action",
        ),
        "seeded_stochastic_normalized_action": _require_bitwise(
            contender_stochastic.normalized,
            reload_stochastic.normalized,
            field="reload stochastic normalized action",
        ),
        "seeded_stochastic_physical_action": _require_bitwise(
            contender_stochastic.physical,
            reload_stochastic.physical,
            field="reload stochastic physical action",
        ),
    }
    source_hashes = {
        name: sha256_file(Path(module.__file__))
        for name, module in (
            ("policy", sys.modules[FullAuthorityActor.__module__]),
            ("receipt_generator", sys.modules[__name__]),
        )
    }
    return {
        "action_adapter_id": ACTION_ADAPTER_ID,
        "claim_ceiling": "interface identity only; no behavioral evidence",
        "copied_parameter_audit": parameter_audit,
        "e1_sampling_seed": E1_SAMPLING_SEED,
        "e1_sampling_epsilon_sha256": array_sha256(epsilon),
        "evidence_class": EVIDENCE_CLASS,
        "export": {
            "byte_count": published_export.byte_count,
            "format_id": "strict_full_authority_actor_npz_npy1_c_order_no_pickle/v1",
            "path": Path(published_export.path).relative_to(root).as_posix(),
            "sha256": published_export.sha256,
        },
        "fixture_batch": {
            "count": 68,
            "input_sha256": array_sha256(policy_input.array),
            "real_count": 64,
            "real_fixtures": real_fixtures,
            "sampling_method": "ascending_sha256_full_authority_e1_fixture/v1",
            "state_sha256": array_sha256(states),
            "synthetic_count": 4,
            "synthetic_design_path": SYNTHETIC_DESIGN_PATH,
            "window_sha256": array_sha256(windows),
        },
        "identity_checks": equality,
        "likelihood_audit": {
            "maximum_absolute_recomputation_difference": likelihood_delta,
            "passed": True,
            "tolerance": 1e-5,
        },
        "optimizer_admission": optimizer_admission,
        "passed": True,
        "receipt_id": E1_RECEIPT_ID,
        "reload_identity_checks": reload_equality,
        "reload_parameter_checks": reload_parameter_checks,
        "schema_version": 1,
        "source_expert": {
            "actor_schema_sha256": expert.schema_sha256,
            "actor_state_sha256": expert.state_sha256,
            "byte_count": expert.byte_count,
            "path": EXPERT_ACTOR_PATH,
            "sha256": expert.content_sha256,
        },
        "source_hashes": source_hashes,
        "training_steps": 0,
        "value_network": {
            "architecture_id": "independent_708_256_256_1_relu/v1",
            "parameter_count": sum(parameter.numel() for parameter in policy.value.parameters()),
            "seed": policy.value_seed,
            "shared_actor_parameters": False,
        },
    }


def publish_e1_receipt(
    *,
    repository_root: Path,
    export_path: Path,
    output_path: Path,
) -> PublishedArtifact:
    receipt = build_e1_receipt(repository_root=repository_root, export_path=export_path)
    return publish_bytes_without_overwrite(output_path, canonical_json_bytes(receipt))


def _same_phase_error(source_row: np.ndarray, target_row: np.ndarray) -> float:
    state = tracking_state_from_reference_row(np.ascontiguousarray(source_row, dtype="<f8"))
    return max(
        normalized_tracking_errors(
            state,
            np.ascontiguousarray(target_row, dtype="<f8"),
        )
    )


def build_phase_transfer_receipt(*, repository_root: Path) -> dict[str, object]:
    """Reproduce both preregistered reference-row splice comparisons."""

    root = Path(repository_root)
    rows: list[dict[str, object]] = []
    for block in ADMITTED_TRAINING_BLOCKS:
        expert = load_v2_reference_clip(
            root / CORPUS_PATH,
            block=block,
            behavior="expert",
        )
        medium = load_v2_reference_clip(
            root / CORPUS_PATH,
            block=block,
            behavior="medium",
        )
        first_state = tracking_state_from_reference_row(
            np.ascontiguousarray(expert.reference_rows[300], dtype="<f8")
        )
        first = select_nearest_phase(
            state=first_state,
            target_rows=medium.reference_rows,
            task_step=300,
            source_behavior="expert",
            target_behavior="medium",
            reason="static_expert_boundary_300_to_medium",
        )
        first_same = _same_phase_error(expert.reference_rows[300], medium.reference_rows[300])
        medium_phase_after_300 = first.selected_phase + 300
        if medium_phase_after_300 > 1000:
            raise ExperimentContractError("static medium phase reached an early terminal hold")
        second_state = tracking_state_from_reference_row(
            np.ascontiguousarray(
                medium.reference_rows[medium_phase_after_300],
                dtype="<f8",
            )
        )
        second = select_nearest_phase(
            state=second_state,
            target_rows=expert.reference_rows,
            task_step=600,
            source_behavior="medium",
            target_behavior="expert",
            reason="static_medium_after_300_steps_to_expert",
        )
        second_same = _same_phase_error(
            medium.reference_rows[medium_phase_after_300],
            expert.reference_rows[600],
        )
        if (
            first.selected_score[0] > first_same
            or second.selected_score[0] > second_same
            or first.selected_phase > 300
            or second.selected_phase > 600
        ):
            raise ExperimentContractError("nearest-phase static check failed")
        rows.append(
            {
                "block": block,
                "expert_bundle_sha256": expert.bundle_sha256,
                "expert_to_medium": {
                    "candidate_range_inclusive": [0, 300],
                    "nearest_maximum_normalized_error": first.selected_score[0],
                    "same_index_maximum_normalized_error": first_same,
                    "selected_phase": first.selected_phase,
                    "selected_six_normalized_errors": list(first.selected_normalized_errors),
                    "selected_sum_squared_error": first.selected_score[1],
                },
                "medium_after_300_to_expert": {
                    "candidate_range_inclusive": [0, 600],
                    "medium_source_phase": medium_phase_after_300,
                    "nearest_maximum_normalized_error": second.selected_score[0],
                    "same_index_maximum_normalized_error": second_same,
                    "selected_phase": second.selected_phase,
                    "selected_six_normalized_errors": list(second.selected_normalized_errors),
                    "selected_sum_squared_error": second.selected_score[1],
                },
                "medium_bundle_sha256": medium.bundle_sha256,
            }
        )
    first_same_values = [
        row["expert_to_medium"]["same_index_maximum_normalized_error"] for row in rows
    ]
    first_nearest_values = [
        row["expert_to_medium"]["nearest_maximum_normalized_error"] for row in rows
    ]
    first_phases = [row["expert_to_medium"]["selected_phase"] for row in rows]
    second_same_values = [
        row["medium_after_300_to_expert"]["same_index_maximum_normalized_error"] for row in rows
    ]
    second_nearest_values = [
        row["medium_after_300_to_expert"]["nearest_maximum_normalized_error"] for row in rows
    ]
    second_phases = [row["medium_after_300_to_expert"]["selected_phase"] for row in rows]
    return {
        "admitted_training_blocks": list(ADMITTED_TRAINING_BLOCKS),
        "claim_ceiling": "static reference-row interface check; not simulator evidence",
        "evidence_class": EVIDENCE_CLASS,
        "passed": True,
        "receipt_id": PHASE_TRANSFER_RECEIPT_ID,
        "rows": rows,
        "schema_version": 1,
        "summary": {
            "expert_to_medium": {
                "nearest_error_range": [min(first_nearest_values), max(first_nearest_values)],
                "same_index_error_range": [min(first_same_values), max(first_same_values)],
                "selected_phase_range": [min(first_phases), max(first_phases)],
            },
            "medium_after_300_to_expert": {
                "nearest_error_range": [min(second_nearest_values), max(second_nearest_values)],
                "same_index_error_range": [min(second_same_values), max(second_same_values)],
                "selected_phase_range": [min(second_phases), max(second_phases)],
            },
        },
    }


def publish_phase_transfer_receipt(
    *,
    repository_root: Path,
    output_path: Path,
) -> PublishedArtifact:
    receipt = build_phase_transfer_receipt(repository_root=repository_root)
    return publish_bytes_without_overwrite(output_path, canonical_json_bytes(receipt))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    e1 = subparsers.add_parser("e1")
    e1.add_argument("--repository-root", type=Path, required=True)
    e1.add_argument("--export", type=Path, required=True)
    e1.add_argument("--output", type=Path, required=True)
    transfer = subparsers.add_parser("phase-transfer")
    transfer.add_argument("--repository-root", type=Path, required=True)
    transfer.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "e1":
            published = publish_e1_receipt(
                repository_root=args.repository_root,
                export_path=args.export,
                output_path=args.output,
            )
        else:
            published = publish_phase_transfer_receipt(
                repository_root=args.repository_root,
                output_path=args.output,
            )
    except (ExperimentContractError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "byte_count": published.byte_count,
                "path": str(published.path),
                "sha256": published.sha256,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ADMITTED_TRAINING_BLOCKS",
    "E1_RECEIPT_ID",
    "E1_SAMPLING_SEED",
    "PHASE_TRANSFER_RECEIPT_ID",
    "build_e1_receipt",
    "build_phase_transfer_receipt",
    "main",
    "publish_e1_receipt",
    "publish_phase_transfer_receipt",
    "sha_ranked_real_fixtures",
    "validate_e1_receipt_at_admission",
]


if __name__ == "__main__":
    raise SystemExit(main())
