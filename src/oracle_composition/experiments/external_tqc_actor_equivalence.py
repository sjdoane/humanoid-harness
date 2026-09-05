"""Fixed-batch equivalence for an external actor and its strict NPZ."""

from __future__ import annotations

import hashlib
import importlib.metadata as importlib_metadata
import json
import os
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from oracle_composition.sources.external_sb3_actor import (
    AUTHORITY,
    EXPECTED_LOCAL_VERSIONS,
    POLICY_SHA256,
    ExternalPretrainedActorAuthority,
    revalidate_external_pretrained_actor_authority,
)
from oracle_composition.sources.farama_tqc_sibling_registrations import (
    SIBLING_REGISTRATION_SPECS,
)

from .artifact_io import publish_bytes_without_overwrite
from .fixed_reference import ExperimentContractError
from .tqc_actor_equivalence_primitives import (
    EQUIVALENCE_OBSERVATION_SHA256,
    EQUIVALENCE_SAMPLING_SEED,
    LOG_STD_MAX,
    LOG_STD_MIN,
    canonical_array_sha256,
    equivalence_observations,
    seeded_squashed_normal_sample,
)
from .tqc_actor_npz import ACTION_WIDTH, actor_state_sha256, validate_actor_arrays

EXTERNAL_EQUIVALENCE_ID = "external_tqc_actor_strict_npz_equivalence/v2"
PURE_ACTOR_ARCHITECTURE_ID = "pure_torch_tqc_actor_348_256_256_17_relu_state_dependent_std/v1"
VERIFIER_SOURCE_LOGICAL_PATH = (
    "src/oracle_composition/experiments/external_tqc_actor_equivalence.py"
)
MAX_VERIFIER_SOURCE_BYTES = 1024 * 1024
_OUTPUT_NAMES = ("mean", "log_std", "deterministic_action", "seeded_sample")
_RECEIPT_ISSUER = object()
SOURCE_POLICY_SHA256_BY_VARIANT = {
    "expert": POLICY_SHA256,
    **{variant: spec.policy_sha256 for variant, spec in SIBLING_REGISTRATION_SPECS.items()},
}


def _exact_json_match(value: object, expected: object) -> bool:
    try:
        return _exact_json_match_inner(value, expected)
    except (RecursionError, TypeError, ValueError):
        return False


def _exact_json_match_inner(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        if set(value) != set(expected):
            return False
        return all(_exact_json_match_inner(value[key], expected[key]) for key in expected)
    if type(expected) is list:
        return len(value) == len(expected) and all(
            _exact_json_match_inner(observed, wanted)
            for observed, wanted in zip(value, expected, strict=True)
        )
    return value == expected


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExperimentContractError("external equivalence value is not canonical JSON") from exc


def _canonical_sha256(value: object, *, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"{field_name} must be a lowercase SHA-256")
    return value


def _verifier_source_identity() -> dict[str, object]:
    try:
        payload = Path(__file__).resolve(strict=True).read_bytes()
    except OSError as exc:
        raise ExperimentContractError(
            "external equivalence verifier source is unavailable"
        ) from exc
    if not 0 < len(payload) <= MAX_VERIFIER_SOURCE_BYTES:
        raise ExperimentContractError("external equivalence verifier source size differs")
    return {
        "logical_path": VERIFIER_SOURCE_LOGICAL_PATH,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_count": len(payload),
    }


def _distribution_identity(name: str) -> dict[str, str]:
    try:
        distribution = importlib_metadata.distribution(name)
        observed_name = distribution.metadata["Name"]
        version = distribution.version
    except (KeyError, OSError, ValueError, importlib_metadata.PackageNotFoundError) as exc:
        raise ExperimentContractError(f"{name} distribution identity is unavailable") from exc
    if type(observed_name) is not str or type(version) is not str:
        raise ExperimentContractError(f"{name} distribution identity is invalid")
    return {"name": observed_name, "version": version}


def _verifier_runtime_identity() -> dict[str, object]:
    import torch

    identity = {
        "torch": {
            "runtime_version": str(torch.__version__),
            "distribution": _distribution_identity("torch"),
        },
        "numpy": {
            "runtime_version": np.__version__,
            "distribution": _distribution_identity("numpy"),
        },
    }
    expected = {
        "torch": {
            "runtime_version": EXPECTED_LOCAL_VERSIONS["torch"],
            "distribution": {
                "name": "torch",
                "version": EXPECTED_LOCAL_VERSIONS["torch"],
            },
        },
        "numpy": {
            "runtime_version": EXPECTED_LOCAL_VERSIONS["numpy"],
            "distribution": {
                "name": "numpy",
                "version": EXPECTED_LOCAL_VERSIONS["numpy"],
            },
        },
    }
    if not _exact_json_match(identity, expected):
        raise ExperimentContractError("external equivalence verifier runtime differs")
    return identity


def _actor_outputs(
    arrays: Mapping[str, np.ndarray],
    observations: np.ndarray,
    *,
    sampling_seed: int,
) -> dict[str, np.ndarray]:
    import torch
    from torch.nn import functional

    validated = validate_actor_arrays(arrays)
    parameters = {
        name: torch.from_numpy(validated[name].copy(order="C"))
        for name in (
            "latent_pi.0.weight",
            "latent_pi.0.bias",
            "latent_pi.2.weight",
            "latent_pi.2.bias",
            "mu.weight",
            "mu.bias",
            "log_std.weight",
            "log_std.bias",
        )
    }
    value = torch.from_numpy(observations.copy(order="C"))
    with torch.inference_mode():
        latent = functional.relu(
            functional.linear(
                value,
                parameters["latent_pi.0.weight"],
                parameters["latent_pi.0.bias"],
            )
        )
        latent = functional.relu(
            functional.linear(
                latent,
                parameters["latent_pi.2.weight"],
                parameters["latent_pi.2.bias"],
            )
        )
        mean = functional.linear(latent, parameters["mu.weight"], parameters["mu.bias"])
        log_std = torch.clamp(
            functional.linear(
                latent,
                parameters["log_std.weight"],
                parameters["log_std.bias"],
            ),
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
        if (
            output.shape != (4, ACTION_WIDTH)
            or output.dtype != np.dtype("<f4")
            or not output.flags.c_contiguous
            or not np.isfinite(output).all()
        ):
            raise ExperimentContractError(f"external actor {name} output is invalid")
    return result


def _receipt_payload(
    authority: ExternalPretrainedActorAuthority,
    *,
    hashes: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "equivalence_id": EXTERNAL_EQUIVALENCE_ID,
        "authority": AUTHORITY,
        "evidence_class": "external_base_import",
        "evidence_level": "interface_check",
        "source_variant": authority.source_variant,
        "import_receipt_sha256": authority.receipt_sha256,
        "source_policy_sha256": SOURCE_POLICY_SHA256_BY_VARIANT[authority.source_variant],
        "strict_actor_npz": {
            "sha256": authority.loaded_actor.content_sha256,
            "byte_count": authority.loaded_actor.byte_count,
            "actor_state_sha256": authority.loaded_actor.state_sha256,
            "schema_sha256": authority.loaded_actor.schema_sha256,
        },
        "actor_state": {
            "source_sha256": actor_state_sha256(authority.source_arrays),
            "strict_npz_sha256": authority.loaded_actor.state_sha256,
        },
        "architecture": {
            "id": PURE_ACTOR_ARCHITECTURE_ID,
            "implementation": "pure torch functional linear-ReLU-linear-ReLU heads",
            "policy_kwargs": {"use_sde": False},
            "layers": [348, 256, 256, 17],
            "log_std_clamp": [LOG_STD_MIN, LOG_STD_MAX],
            "deterministic_action": "tanh(mean)",
            "seeded_sample": "tanh(Normal(mean, exp(clamped_log_std)).rsample())",
        },
        "fixed_batch": {
            "shape": [4, 348],
            "sha256": EQUIVALENCE_OBSERVATION_SHA256,
        },
        "sampling_seed": EQUIVALENCE_SAMPLING_SEED,
        "paired_output_sha256": {name: dict(hashes[name]) for name in _OUTPUT_NAMES},
        "verifier": {
            "source": _verifier_source_identity(),
            "runtime": _verifier_runtime_identity(),
        },
        "cpu_rng_state": {
            "preserved": True,
            "ambient_values_recorded": False,
        },
        "passed": True,
        "claim": {
            "establishes": "exact output equivalence on one fixed 4 x 348 batch",
            "does_not_establish": [
                "E1",
                "tracker admission",
                "reference use",
                "Humanoid behavior",
            ],
        },
    }


def validate_external_actor_equivalence_receipt(value: object) -> dict[str, object]:
    """Reject incomplete, unpaired, or over-claimed external equivalence receipts."""

    required = {
        "schema_version",
        "equivalence_id",
        "authority",
        "evidence_class",
        "evidence_level",
        "source_variant",
        "import_receipt_sha256",
        "source_policy_sha256",
        "strict_actor_npz",
        "actor_state",
        "architecture",
        "fixed_batch",
        "sampling_seed",
        "paired_output_sha256",
        "verifier",
        "cpu_rng_state",
        "passed",
        "claim",
    }
    if type(value) is not dict or set(value) != required:
        raise ExperimentContractError("external actor equivalence receipt fields differ")
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 2
        or type(value["equivalence_id"]) is not str
        or value["equivalence_id"] != EXTERNAL_EQUIVALENCE_ID
        or type(value["authority"]) is not str
        or value["authority"] != AUTHORITY
        or type(value["evidence_class"]) is not str
        or value["evidence_class"] != "external_base_import"
        or type(value["evidence_level"]) is not str
        or value["evidence_level"] != "interface_check"
        or type(value["source_variant"]) is not str
        or value["source_variant"] not in SOURCE_POLICY_SHA256_BY_VARIANT
        or type(value["source_policy_sha256"]) is not str
        or value["source_policy_sha256"] != SOURCE_POLICY_SHA256_BY_VARIANT[value["source_variant"]]
        or type(value["sampling_seed"]) is not int
        or value["sampling_seed"] != EQUIVALENCE_SAMPLING_SEED
        or value["passed"] is not True
    ):
        raise ExperimentContractError("external actor equivalence identity differs")
    _canonical_sha256(value["import_receipt_sha256"], field_name="import receipt SHA-256")
    fixed_batch = value["fixed_batch"]
    if not _exact_json_match(
        fixed_batch,
        {"shape": [4, 348], "sha256": EQUIVALENCE_OBSERVATION_SHA256},
    ):
        raise ExperimentContractError("external actor equivalence batch differs")
    if not _exact_json_match(
        value["architecture"],
        {
            "id": PURE_ACTOR_ARCHITECTURE_ID,
            "implementation": "pure torch functional linear-ReLU-linear-ReLU heads",
            "policy_kwargs": {"use_sde": False},
            "layers": [348, 256, 256, 17],
            "log_std_clamp": [LOG_STD_MIN, LOG_STD_MAX],
            "deterministic_action": "tanh(mean)",
            "seeded_sample": "tanh(Normal(mean, exp(clamped_log_std)).rsample())",
        },
    ):
        raise ExperimentContractError("external actor equivalence architecture differs")
    strict_actor = value["strict_actor_npz"]
    if type(strict_actor) is not dict or set(strict_actor) != {
        "sha256",
        "byte_count",
        "actor_state_sha256",
        "schema_sha256",
    }:
        raise ExperimentContractError("external actor equivalence NPZ binding differs")
    for field_name in ("sha256", "actor_state_sha256", "schema_sha256"):
        _canonical_sha256(strict_actor[field_name], field_name=f"strict actor {field_name}")
    if type(strict_actor["byte_count"]) is not int or strict_actor["byte_count"] <= 0:
        raise ExperimentContractError("external actor equivalence NPZ byte count differs")
    actor_state = value["actor_state"]
    if (
        type(actor_state) is not dict
        or set(actor_state) != {"source_sha256", "strict_npz_sha256"}
        or actor_state["source_sha256"] != actor_state["strict_npz_sha256"]
        or actor_state["strict_npz_sha256"] != strict_actor["actor_state_sha256"]
    ):
        raise ExperimentContractError("external actor state fingerprints differ")
    for digest in actor_state.values():
        _canonical_sha256(digest, field_name="actor state fingerprint")
    hashes = value["paired_output_sha256"]
    if type(hashes) is not dict or set(hashes) != set(_OUTPUT_NAMES):
        raise ExperimentContractError("external actor output hash schema differs")
    for name in _OUTPUT_NAMES:
        pair = hashes[name]
        if (
            type(pair) is not dict
            or set(pair) != {"source", "strict_npz"}
            or pair["source"] != pair["strict_npz"]
        ):
            raise ExperimentContractError(f"external actor {name} hashes differ")
        _canonical_sha256(pair["source"], field_name=f"external actor {name} SHA-256")
    verifier = value["verifier"]
    expected_verifier = {
        "source": _verifier_source_identity(),
        "runtime": _verifier_runtime_identity(),
    }
    if not _exact_json_match(verifier, expected_verifier):
        raise ExperimentContractError("external actor equivalence verifier identity differs")
    rng = value["cpu_rng_state"]
    if not _exact_json_match(
        rng,
        {"preserved": True, "ambient_values_recorded": False},
    ):
        raise ExperimentContractError("external actor CPU RNG preservation differs")
    if not _exact_json_match(
        value["claim"],
        {
            "establishes": "exact output equivalence on one fixed 4 x 348 batch",
            "does_not_establish": [
                "E1",
                "tracker admission",
                "reference use",
                "Humanoid behavior",
            ],
        },
    ):
        raise ExperimentContractError("external actor equivalence claim ceiling differs")
    return value


def _reject_duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExperimentContractError("equivalence receipt contains a duplicate key")
        result[key] = value
    return result


@dataclass(frozen=True, slots=True)
class ExternalActorEquivalenceReceipt:
    """Sealed exact-output receipt tied to one external import authority."""

    receipt_sha256: str
    receipt_byte_count: int
    receipt_bytes: bytes = field(repr=False, compare=False)
    actor_authority: ExternalPretrainedActorAuthority = field(repr=False, compare=False)
    _creator_pid: int = field(repr=False, compare=False)
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _RECEIPT_ISSUER:
            raise ExperimentContractError(
                "external actor equivalence receipt may only be issued by the verifier"
            )
        self._validate_sealed()

    def to_dict(self) -> dict[str, object]:
        try:
            value = json.loads(
                self.receipt_bytes.decode("utf-8", errors="strict"),
                object_pairs_hook=_reject_duplicate_key,
                parse_constant=lambda _value: (_ for _ in ()).throw(
                    ExperimentContractError("equivalence receipt contains a non-finite constant")
                ),
            )
        except ExperimentContractError:
            raise
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ExperimentContractError("sealed equivalence receipt is invalid") from exc
        return validate_external_actor_equivalence_receipt(value)

    def _validate_sealed(self) -> None:
        authority = revalidate_external_pretrained_actor_authority(self.actor_authority)
        _canonical_sha256(self.receipt_sha256, field_name="equivalence receipt SHA-256")
        if (
            self._creator_pid != os.getpid()
            or type(self.receipt_byte_count) is not int
            or self.receipt_byte_count != len(self.receipt_bytes)
            or hashlib.sha256(self.receipt_bytes).hexdigest() != self.receipt_sha256
        ):
            raise ExperimentContractError("external actor equivalence receipt seal differs")
        value = self.to_dict()
        if (
            value["source_variant"] != authority.source_variant
            or value["source_policy_sha256"]
            != authority.to_receipt_dict()["source"]["policy_pth"]["sha256"]
            or value["import_receipt_sha256"] != authority.receipt_sha256
            or value["strict_actor_npz"]["sha256"] != authority.loaded_actor.content_sha256
            or value["actor_state"]["source_sha256"] != actor_state_sha256(authority.source_arrays)
        ):
            raise ExperimentContractError("external actor equivalence lineage differs")


def verify_external_actor_equivalence(
    authority: ExternalPretrainedActorAuthority,
    *,
    receipt_output_path: Path,
    sampling_seed: int = EQUIVALENCE_SAMPLING_SEED,
) -> ExternalActorEquivalenceReceipt:
    """Prove exact source-to-NPZ outputs without invoking a local checkpoint loader."""

    import torch

    authority = revalidate_external_pretrained_actor_authority(authority)
    if type(sampling_seed) is not int or sampling_seed != EQUIVALENCE_SAMPLING_SEED:
        raise ExperimentContractError("external actor equivalence sampling seed differs")
    observations = equivalence_observations()
    rng_before = torch.random.get_rng_state().clone()
    source_outputs = _actor_outputs(
        authority.source_arrays,
        observations,
        sampling_seed=sampling_seed,
    )
    strict_outputs = _actor_outputs(
        authority.loaded_actor.arrays,
        observations,
        sampling_seed=sampling_seed,
    )
    rng_after = torch.random.get_rng_state().clone()
    hashes: dict[str, dict[str, str]] = {}
    for name in _OUTPUT_NAMES:
        source = source_outputs[name]
        strict = strict_outputs[name]
        if source.tobytes(order="C") != strict.tobytes(order="C"):
            raise ExperimentContractError(f"external source and strict NPZ {name} bytes differ")
        hashes[name] = {
            "source": canonical_array_sha256(source),
            "strict_npz": canonical_array_sha256(strict),
        }
    if not torch.equal(rng_before, rng_after):
        raise ExperimentContractError("CPU RNG state changed during external equivalence")
    payload = _receipt_payload(
        authority,
        hashes=hashes,
    )
    validate_external_actor_equivalence_receipt(payload)
    receipt_bytes = _canonical_json(payload) + b"\n"
    published = publish_bytes_without_overwrite(Path(receipt_output_path), receipt_bytes)
    return ExternalActorEquivalenceReceipt(
        receipt_sha256=published.sha256,
        receipt_byte_count=published.byte_count,
        receipt_bytes=receipt_bytes,
        actor_authority=authority,
        _creator_pid=os.getpid(),
        _issuer=_RECEIPT_ISSUER,
    )


def revalidate_external_actor_equivalence_receipt(
    receipt: ExternalActorEquivalenceReceipt,
) -> ExternalActorEquivalenceReceipt:
    """Revalidate the exact sealed external-equivalence result."""

    if type(receipt) is not ExternalActorEquivalenceReceipt:
        raise ExperimentContractError("external equivalence result must be the exact receipt")
    receipt._validate_sealed()
    return receipt


__all__ = [
    "EXTERNAL_EQUIVALENCE_ID",
    "PURE_ACTOR_ARCHITECTURE_ID",
    "ExternalActorEquivalenceReceipt",
    "revalidate_external_actor_equivalence_receipt",
    "validate_external_actor_equivalence_receipt",
    "verify_external_actor_equivalence",
]
