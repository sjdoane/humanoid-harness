"""Frozen, disjoint calibration contract for the Phase B task-success endpoint."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes, sha256_file
from oracle_composition.experiments.artifact_io import (
    PublishedArtifact,
    publish_bytes_without_overwrite,
)
from oracle_composition.experiments.fixed_reference import ExperimentContractError

CALIBRATION_RECEIPT_SCHEMA_ID = "phase_b_task_success_calibration_receipt/v1"
CALIBRATION_BLOCKS = tuple(range(120201, 120221))
CALIBRATION_POLICY_SEEDS = (122001, 122101, 122201, 122301, 122401)
EVALUATION_BLOCKS = frozenset(range(120101, 120121))
TRAINING_POLICY_SEEDS = frozenset((121001, 121101, 121201, 121301, 121401, 121901))
SEGMENTS = ("fast", "return_fast", "slow")
QUANTILE = 0.95
SEGMENT_MARGIN = 1.1
SETTLED_MARGIN = 1.1
LATENCY_MARGIN_STEPS = 4
LATENCY_CENSOR_STEPS = 65
MAXIMUM_LATENCY_CAP_STEPS = 64
MAX_RECEIPT_BYTES = 2 * 1024 * 1024
MAX_SIGNED_32 = 2_147_483_647


@dataclass(frozen=True, slots=True)
class TaskSuccessCalibration:
    sha256: str
    segment_speed_error_bands_m_s: Mapping[str, float]
    transition_latency_caps_steps: tuple[int, int]
    settled_state_normalized_error_band: float
    censoring_latency_steps: int


def _finite_nonnegative(value: object, *, field: str) -> float:
    if type(value) not in {int, float}:
        raise ExperimentContractError(f"calibration {field} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ExperimentContractError(f"calibration {field} must be finite") from exc
    if not math.isfinite(result) or result < 0.0:
        raise ExperimentContractError(f"calibration {field} must be finite and non-negative")
    return result


def _integer(value: object, *, field: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ExperimentContractError(f"calibration {field} lies outside its integer bound")
    return value


def _sha(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"calibration {field} must be a lowercase SHA-256")
    return value


def _quantile_higher(values: Sequence[float]) -> float:
    if not values:
        raise ExperimentContractError("calibration statistic has no samples")
    ordered = sorted(values)
    return ordered[math.ceil(QUANTILE * len(ordered)) - 1]


def _checked_sample(value: object) -> dict[str, object]:
    expected = {
        "block_id",
        "fall",
        "first_transition_latency_steps",
        "policy_seed_id",
        "second_transition_latency_steps",
        "segment_speed_errors_m_s",
        "settled_state_normalized_error",
        "source_checkpoint_sha256",
    }
    if type(value) is not dict or set(value) != expected:
        raise ExperimentContractError("calibration sample schema differs")
    block = _integer(value["block_id"], field="block_id", minimum=1, maximum=MAX_SIGNED_32)
    seed = _integer(
        value["policy_seed_id"], field="policy_seed_id", minimum=1, maximum=MAX_SIGNED_32
    )
    if block not in CALIBRATION_BLOCKS or seed not in CALIBRATION_POLICY_SEEDS:
        raise ExperimentContractError("calibration sample lies outside the frozen split")
    if block in EVALUATION_BLOCKS or seed in TRAINING_POLICY_SEEDS:
        raise ExperimentContractError("calibration and evaluation/training identities overlap")
    if type(value["fall"]) is not bool:
        raise ExperimentContractError("calibration fall must be boolean")
    segments = value["segment_speed_errors_m_s"]
    if type(segments) is not dict or set(segments) != set(SEGMENTS):
        raise ExperimentContractError("calibration segment statistic schema differs")
    checked_segments = {
        name: _finite_nonnegative(segments[name], field=f"segment {name}") for name in SEGMENTS
    }
    latencies: list[int | None] = []
    for name in ("first_transition_latency_steps", "second_transition_latency_steps"):
        raw = value[name]
        if raw is None:
            latencies.append(None)
        else:
            latencies.append(
                _integer(raw, field=name, minimum=0, maximum=MAXIMUM_LATENCY_CAP_STEPS)
            )
    return {
        "block_id": block,
        "fall": value["fall"],
        "first_transition_latency_steps": latencies[0],
        "policy_seed_id": seed,
        "second_transition_latency_steps": latencies[1],
        "segment_speed_errors_m_s": checked_segments,
        "settled_state_normalized_error": _finite_nonnegative(
            value["settled_state_normalized_error"], field="settled-state error"
        ),
        "source_checkpoint_sha256": _sha(
            value["source_checkpoint_sha256"], field="source checkpoint"
        ),
    }


def calibration_receipt_value(samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Compute the reviewed statistics over the exact disjoint 5 x 20 split."""

    checked = [_checked_sample(dict(sample)) for sample in samples]
    expected_pairs = {
        (seed, block) for seed in CALIBRATION_POLICY_SEEDS for block in CALIBRATION_BLOCKS
    }
    observed_pairs = {(row["policy_seed_id"], row["block_id"]) for row in checked}
    if observed_pairs != expected_pairs or len(checked) != len(expected_pairs):
        raise ExperimentContractError("calibration samples do not cover the exact disjoint split")
    checkpoint_hashes_by_seed: dict[int, set[str]] = {
        seed: set() for seed in CALIBRATION_POLICY_SEEDS
    }
    for row in checked:
        checkpoint_hashes_by_seed[int(row["policy_seed_id"])].add(
            str(row["source_checkpoint_sha256"])
        )
    if any(len(hashes) != 1 for hashes in checkpoint_hashes_by_seed.values()):
        raise ExperimentContractError(
            "calibration policy seed does not identify one source checkpoint"
        )
    checked.sort(key=lambda row: (row["policy_seed_id"], row["block_id"]))
    segment_bands = {
        name: _quantile_higher([float(row["segment_speed_errors_m_s"][name]) for row in checked])
        * SEGMENT_MARGIN
        for name in SEGMENTS
    }
    latency_caps = []
    for field in ("first_transition_latency_steps", "second_transition_latency_steps"):
        censored = [
            LATENCY_CENSOR_STEPS if row["fall"] or row[field] is None else int(row[field])
            for row in checked
        ]
        latency_caps.append(
            min(
                MAXIMUM_LATENCY_CAP_STEPS,
                math.ceil(_quantile_higher(censored)) + LATENCY_MARGIN_STEPS,
            )
        )
    settled_band = (
        _quantile_higher([float(row["settled_state_normalized_error"]) for row in checked])
        * SETTLED_MARGIN
    )
    sample_bytes = canonical_json_bytes(checked)
    return {
        "calibration_receipt_schema_id": CALIBRATION_RECEIPT_SCHEMA_ID,
        "evidence_class": "exploratory_calibration_only",
        "procedure": {
            "censoring": {
                "fall_or_never_settled_latency_steps": LATENCY_CENSOR_STEPS,
                "maximum_admissible_latency_cap_steps": MAXIMUM_LATENCY_CAP_STEPS,
            },
            "empirical_quantile": QUANTILE,
            "quantile_method": "higher",
            "segment_multiplicative_safety_margin": SEGMENT_MARGIN,
            "settled_state_multiplicative_safety_margin": SETTLED_MARGIN,
            "transition_latency_additive_safety_margin_steps": LATENCY_MARGIN_STEPS,
        },
        "sample_count": len(checked),
        "samples": checked,
        "samples_sha256": hashlib.sha256(sample_bytes).hexdigest(),
        "schema_version": 1,
        "split": {
            "block_ids": list(CALIBRATION_BLOCKS),
            "disjoint_from_evaluation_and_training": True,
            "policy_seed_ids": list(CALIBRATION_POLICY_SEEDS),
        },
        "thresholds": {
            "censoring_latency_steps": LATENCY_CENSOR_STEPS,
            "segment_speed_error_bands_m_s": segment_bands,
            "settled_state_normalized_error_band": settled_band,
            "transition_latency_caps_steps": latency_caps,
        },
        "source_sha256": sha256_file(Path(__file__)),
    }


def publish_calibration_receipt(
    path: Path, samples: Sequence[Mapping[str, object]]
) -> PublishedArtifact:
    return publish_bytes_without_overwrite(
        path, canonical_json_bytes(calibration_receipt_value(samples))
    )


def load_calibration_receipt(path: Path, *, expected_sha256: str) -> TaskSuccessCalibration:
    """Load and recompute a bound receipt before a policy or environment exists."""

    _sha(expected_sha256, field="receipt SHA")
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ExperimentContractError("calibration receipt is unavailable")
    encoded = candidate.read_bytes()
    if (
        not encoded
        or len(encoded) > MAX_RECEIPT_BYTES
        or hashlib.sha256(encoded).hexdigest() != expected_sha256
    ):
        raise ExperimentContractError("calibration receipt byte identity differs")
    try:
        value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError, MemoryError, RecursionError) as exc:
        raise ExperimentContractError("calibration receipt is invalid JSON") from exc
    if type(value) is not dict or encoded != canonical_json_bytes(value):
        raise ExperimentContractError("calibration receipt is not canonical JSON")
    expected = calibration_receipt_value(value.get("samples", []))
    if value != expected:
        raise ExperimentContractError("calibration receipt does not recompute")
    thresholds = value["thresholds"]
    return TaskSuccessCalibration(
        sha256=expected_sha256,
        segment_speed_error_bands_m_s=MappingProxyType(
            dict(thresholds["segment_speed_error_bands_m_s"])
        ),
        transition_latency_caps_steps=tuple(thresholds["transition_latency_caps_steps"]),
        settled_state_normalized_error_band=float(
            thresholds["settled_state_normalized_error_band"]
        ),
        censoring_latency_steps=int(thresholds["censoring_latency_steps"]),
    )


__all__ = [
    "CALIBRATION_BLOCKS",
    "CALIBRATION_POLICY_SEEDS",
    "TaskSuccessCalibration",
    "calibration_receipt_value",
    "load_calibration_receipt",
    "publish_calibration_receipt",
]
