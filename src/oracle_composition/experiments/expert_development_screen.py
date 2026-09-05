"""Reward-independent metrics for the imported expert development screen."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import (
    DEVELOPMENT_SCREEN_SEEDS,
    sha256_file,
    sha256_json,
    validate_full_clip_certificate,
)

from . import tqc_development_metrics as metric_module
from .reference_corpus_contract import ReferenceCorpusContractError
from .tqc_development_metrics import (
    TQCDevelopmentEpisodeAccumulator,
    TQCDevelopmentEpisodeMetrics,
    TQCDevelopmentStepFacts,
    summarize_tqc_development_cohort,
)

SCREEN_ID = "public_expert_same_runtime_development_screen/v1"


@dataclass(frozen=True, slots=True)
class DevelopmentScreenClip:
    seed: int
    bundle_manifest_sha256: str
    reference_identity_sha256: str
    certificate_sha256: str
    certificate: Mapping[str, object]
    arrays: Mapping[str, np.ndarray]
    plain_comparison: Mapping[str, object]
    source_hashes: Mapping[str, str]


def _required_array(
    arrays: Mapping[str, np.ndarray],
    name: str,
    *,
    shape: tuple[int, ...],
    dtype: str,
) -> np.ndarray:
    try:
        value = arrays[name]
    except KeyError as exc:
        raise ReferenceCorpusContractError(f"screen array {name} is missing") from exc
    if (
        not isinstance(value, np.ndarray)
        or value.shape != shape
        or value.dtype != np.dtype(dtype)
        or value.dtype.str != np.dtype(dtype).str
        or not value.flags.c_contiguous
        or not np.isfinite(value).all()
    ):
        raise ReferenceCorpusContractError(f"screen array {name} differs")
    return value


def development_episode_from_arrays(
    *,
    seed: int,
    arrays: Mapping[str, np.ndarray],
) -> TQCDevelopmentEpisodeMetrics:
    """Derive one screen episode from the six reward-independent arrays only."""

    root_xy = _required_array(
        arrays,
        "boundary_root_xy",
        shape=(1001, 2),
        dtype="<f8",
    )
    reference = _required_array(
        arrays,
        "reference_rows",
        shape=(1001, 45),
        dtype="<f8",
    )
    times = _required_array(
        arrays,
        "boundary_simulation_time",
        shape=(1001,),
        dtype="<f8",
    )
    torso_up = _required_array(
        arrays,
        "boundary_torso_up_z",
        shape=(1001,),
        dtype="<f8",
    )
    normalized_actions = _required_array(
        arrays,
        "transition_normalized_action",
        shape=(1000, 17),
        dtype="<f4",
    )
    nonfoot = _required_array(
        arrays,
        "transition_nonfoot_floor_contact",
        shape=(1000,),
        dtype="|u1",
    )
    initial_root = (float(root_xy[0, 0]), float(root_xy[0, 1]), float(reference[0, 0]))
    accumulator = TQCDevelopmentEpisodeAccumulator(
        seed=seed,
        initial_root_position_world_m=initial_root,
        initial_simulation_time_seconds=float(times[0]),
    )
    for index in range(1000):
        boundary = index + 1
        accumulator.add(
            TQCDevelopmentStepFacts(
                step_index=boundary,
                simulation_time_seconds=float(times[boundary]),
                root_position_world_m=(
                    float(root_xy[boundary, 0]),
                    float(root_xy[boundary, 1]),
                    float(reference[boundary, 0]),
                ),
                torso_up_z=float(torso_up[boundary]),
                normalized_action=tuple(float(value) for value in normalized_actions[index]),
                non_foot_floor_contact=bool(nonfoot[index]),
            )
        )
    return accumulator.finish()


def evaluate_public_expert_development_screen(
    clips: tuple[DevelopmentScreenClip, ...],
) -> dict[str, object]:
    """Apply the four predeclared gates without reading a reward field."""

    if type(clips) is not tuple or tuple(clip.seed for clip in clips) != DEVELOPMENT_SCREEN_SEEDS:
        raise ReferenceCorpusContractError("development screen seeds are missing or out of order")
    episodes = []
    clip_bindings: list[dict[str, object]] = []
    expected_source_hashes = {
        "screen_evaluator_source": sha256_file(Path(__file__)),
        "metric_source": sha256_file(Path(str(metric_module.__file__))),
    }
    for clip in clips:
        certificate = validate_full_clip_certificate(clip.certificate)
        if (
            certificate["bundle_manifest_sha256"] != clip.bundle_manifest_sha256
            or certificate["reference_identity_sha256"] != clip.reference_identity_sha256
            or certificate["steps_expected"] != 1000
        ):
            raise ReferenceCorpusContractError("screen Tier-D certificate binding differs")
        if dict(clip.source_hashes) != expected_source_hashes:
            raise ReferenceCorpusContractError("screen evaluator or metric source differs")
        comparison = clip.plain_comparison
        if (
            type(comparison) is not dict
            or comparison.get("status") != "passed"
            or comparison.get("steps_compared") != 1000
            or comparison.get("boundaries_compared") != 1001
            or comparison.get("field_hashes", {}).get("reward", {}).get("instrumented_sha256")
            != comparison.get("field_hashes", {}).get("reward", {}).get("plain_sha256")
        ):
            raise ReferenceCorpusContractError("screen plain-runtime comparison differs")
        episode = development_episode_from_arrays(seed=clip.seed, arrays=clip.arrays)
        episodes.append(episode)
        clip_bindings.append(
            {
                "seed": clip.seed,
                "bundle_manifest_sha256": clip.bundle_manifest_sha256,
                "reference_identity_sha256": clip.reference_identity_sha256,
                "tier_d_certificate_sha256": clip.certificate_sha256,
                "plain_comparison_sha256": sha256_json(dict(comparison)),
            }
        )
    cohort = summarize_tqc_development_cohort(
        tuple(episodes),
        expected_seeds=DEVELOPMENT_SCREEN_SEEDS,
    )
    payload = {
        "screen_id": SCREEN_ID,
        "schema_version": 1,
        "evidence_class": "external_base_import",
        "evidence_level": "development_screen",
        "actor_variant": "expert",
        "evaluation_seeds": list(DEVELOPMENT_SCREEN_SEEDS),
        "steps_per_seed": 1000,
        "source_hashes": expected_source_hashes,
        "clip_bindings": clip_bindings,
        "cohort": cohort.to_dict(),
        "gates": {
            "healthy_episodes": {
                "observed": cohort.full_horizon_healthy_episode_count,
                "required": 20,
                "passed": cohort.full_horizon_healthy_episode_count == 20,
                "definition": "root_height_open_interval_(1.0,2.0)_at_all_1000_poststep_boundaries",
            },
            "upright_episodes": {
                "observed": cohort.full_horizon_upright_episode_count,
                "required": 20,
                "passed": cohort.full_horizon_upright_episode_count == 20,
                "definition": "torso_up_axis_world_z_at_least_0.5_at_all_1000_poststep_boundaries",
            },
            "median_forward_velocity_m_s": {
                "observed": cohort.median_time_average_forward_velocity_m_s,
                "required_minimum": 0.5,
                "passed": cohort.median_time_average_forward_velocity_m_s >= 0.5,
            },
            "episodes_net_forward_displacement_at_least_5_m": {
                "observed": (cohort.episode_count_with_net_forward_displacement_at_least_5_m),
                "required_minimum": 18,
                "passed": (cohort.episode_count_with_net_forward_displacement_at_least_5_m >= 18),
            },
        },
        "screen_passed": cohort.development_behavior_gate_passed,
        "reward_use": {
            "plain_vs_instrumented_canary_only": True,
            "locomotion_metrics_read_reward": False,
        },
        "visual_capture": "disabled_by_external_screen_design/v1",
        "claim_ceiling": (
            "exact_imported_expert_passed_or_failed_predeclared_local_development_screen_only"
        ),
        "does_not_establish": [
            "tracker",
            "E4",
            "E5",
            "oracle",
            "naturalness",
            "robustness",
        ],
    }
    payload["screen_result_sha256"] = sha256_json(payload)
    return payload


__all__ = [
    "SCREEN_ID",
    "DevelopmentScreenClip",
    "development_episode_from_arrays",
    "evaluate_public_expert_development_screen",
]
