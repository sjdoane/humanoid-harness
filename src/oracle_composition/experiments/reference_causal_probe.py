"""Matched-observation numeric-reference sensitivity probe for Experiment 002A."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from oracle_composition.sources import minari_humanoid as minari_module
from oracle_composition.sources.minari_humanoid import (
    MinariHumanoidProjection,
    import_registered_minari_humanoid_episode,
)

from . import local_controllers as controller_module
from . import reference_input_transforms as transform_module
from .artifact_io import publish_json_without_overwrite
from .fixed_reference import ExperimentContractError, sha256_file
from .local_controllers import (
    ACTION_WIDTH,
    BASE_ACTION_SCALE,
    LOCAL_BEHAVIOR_CLONING_SHA256,
    LOCAL_REFERENCE_RESIDUAL_SHA256,
    OBSERVATION_WIDTH,
    REFERENCE_CONDITIONED_WIDTH,
    RESIDUAL_SCALE,
    LocalControllerReceipt,
    ReferenceResidualInferenceFacts,
    load_local_behavior_cloning_controller,
    load_local_reference_residual_controller,
)
from .reference_input_transforms import C_EXACT, CONDITION_IDS, transform_reference_input

PROTOCOL_ID = "experiment_002a_matched_observation_numeric_reference_probe/v1"
EVIDENCE_CLASS = "exploratory"
CLAIM_STATUS = "numeric_reference_input_sensitivity_only_not_tracking_or_oracle_quality"
SNAPSHOT_FRAMES = (0, 125, 250, 375, 500, 625, 750, 875)
EXPECTED_EPISODE_ID = 0
EXPECTED_EPISODE_SEED = 123
EXPECTED_EPISODE_STEPS = 1000
REPORT_SCHEMA_VERSION = 1

_FLOAT32 = np.dtype("<f4")
_FLOAT64 = np.dtype("<f8")


class ResidualController(Protocol):
    """Minimum policy surface consumed by the probe."""

    def assert_integrity(self) -> str: ...

    def inference_facts(
        self,
        observation: np.ndarray,
        reference_window: np.ndarray,
    ) -> ReferenceResidualInferenceFacts: ...


@dataclass(frozen=True, slots=True)
class ReferenceCausalProbeResult:
    """Published exploratory report identity and gate result."""

    path: Path
    file_sha256: str
    analysis_payload_sha256: str
    mechanistic_gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "file_sha256": self.file_sha256,
            "analysis_payload_sha256": self.analysis_payload_sha256,
            "mechanistic_gate_passed": self.mechanistic_gate_passed,
            "claim_status": CLAIM_STATUS,
        }


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExperimentContractError(f"probe value is not canonical JSON: {exc}") from exc


def _raw_sha256(value: np.ndarray) -> str:
    if not value.flags.c_contiguous:
        raise ExperimentContractError("hashed probe array must use C-order storage")
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"{field} must be a lowercase SHA-256")
    return value


def _module_sha256(module: object) -> str:
    source = getattr(module, "__file__", None)
    if not isinstance(source, str) or not source:
        raise ExperimentContractError("bound probe module has no source file")
    return sha256_file(Path(source), maximum_bytes=2 * 1024 * 1024)


def _validated_observations(projection: MinariHumanoidProjection) -> np.ndarray:
    observations = projection.episode_observations
    reference_frames = projection.reference.identity.n_frames
    if (
        not isinstance(observations, np.ndarray)
        or observations.dtype.str != _FLOAT64.str
        or not observations.flags.c_contiguous
        or observations.flags.writeable
        or observations.shape != (EXPECTED_EPISODE_STEPS + 1, OBSERVATION_WIDTH)
        or observations.shape != (reference_frames, OBSERVATION_WIDTH)
        or not np.isfinite(observations).all()
    ):
        raise ExperimentContractError(
            "source observations violate the matched-observation contract"
        )
    if max(SNAPSHOT_FRAMES) >= observations.shape[0]:
        raise ExperimentContractError("source episode is too short for the frozen snapshots")
    return observations


def _validated_base_action(value: object) -> np.ndarray:
    if not isinstance(value, np.ndarray) or value.dtype.str != _FLOAT32.str:
        raise ExperimentContractError("base controller must return exact float32 bytes")
    if value.shape != (ACTION_WIDTH,) or not value.flags.c_contiguous:
        raise ExperimentContractError("base controller action must be one C-order 17D vector")
    if (
        not np.isfinite(value).all()
        or np.any(value < -BASE_ACTION_SCALE)
        or np.any(value > BASE_ACTION_SCALE)
    ):
        raise ExperimentContractError("base controller action exceeds the Humanoid action ABI")
    return value


def _residual_action(facts: ReferenceResidualInferenceFacts) -> np.ndarray:
    action = np.ascontiguousarray(np.asarray(facts.deterministic_action, dtype="<f4"))
    if (
        facts.actor_output_shape != (ACTION_WIDTH,)
        or facts.actor_output_dtype != _FLOAT32.str
        or action.shape != (ACTION_WIDTH,)
        or not np.isfinite(action).all()
        or np.any(action < -1.0)
        or np.any(action > 1.0)
        or _raw_sha256(action) != facts.actor_output_sha256
    ):
        raise ExperimentContractError("residual actor output receipt is invalid")
    return action


def _compose_action(base_action: np.ndarray, residual_action: np.ndarray) -> np.ndarray:
    composed = np.add(
        base_action,
        np.multiply(np.float32(RESIDUAL_SCALE), residual_action, dtype=np.float32),
        dtype=np.float32,
    )
    return np.clip(
        composed,
        np.float32(-BASE_ACTION_SCALE),
        np.float32(BASE_ACTION_SCALE),
    ).astype("<f4", copy=False)


def _facts_dict(facts: ReferenceResidualInferenceFacts) -> dict[str, object]:
    return {
        "policy_input_sha256": facts.policy_input_sha256,
        "policy_input_shape": list(facts.policy_input_shape),
        "policy_input_dtype": facts.policy_input_dtype,
        "actor_input_sha256": facts.actor_input_sha256,
        "critic_input_sha256": facts.critic_input_sha256,
        "controller_state_sha256": facts.controller_state_sha256,
        "actor_output_sha256": facts.actor_output_sha256,
        "actor_output_shape": list(facts.actor_output_shape),
        "actor_output_dtype": facts.actor_output_dtype,
        "actor_output": list(facts.deterministic_action),
        "critic_output_sha256": facts.critic_output_sha256,
        "critic_output_shape": list(facts.critic_output_shape),
        "critic_output_dtype": facts.critic_output_dtype,
        "critic_value": facts.critic_value,
    }


def _validate_consumption(facts: ReferenceResidualInferenceFacts) -> None:
    for field in (
        "policy_input_sha256",
        "actor_input_sha256",
        "critic_input_sha256",
        "controller_state_sha256",
        "actor_output_sha256",
        "critic_output_sha256",
    ):
        _require_sha256(getattr(facts, field), field=field)
    if (
        facts.policy_input_shape != (REFERENCE_CONDITIONED_WIDTH,)
        or facts.policy_input_dtype != _FLOAT32.str
        or facts.actor_input_sha256 != facts.policy_input_sha256
        or facts.critic_input_sha256 != facts.policy_input_sha256
    ):
        raise ExperimentContractError("actor and critic did not consume the declared policy input")
    if facts.critic_output_shape != (1, 1) or facts.critic_output_dtype != _FLOAT32.str:
        raise ExperimentContractError("critic output receipt is invalid")
    if isinstance(facts.critic_value, bool) or not math.isfinite(facts.critic_value):
        raise ExperimentContractError("critic output is non-finite")
    critic_output = np.asarray([[facts.critic_value]], dtype="<f4")
    if (
        not np.isfinite(critic_output).all()
        or float(critic_output[0, 0]) != facts.critic_value
        or _raw_sha256(critic_output) != facts.critic_output_sha256
    ):
        raise ExperimentContractError("critic output SHA-256 does not match its float32 value")


def _difference_summary(records: list[dict[str, object]]) -> dict[str, object]:
    by_key = {
        (int(record["snapshot_frame"]), str(record["condition_id"])): record for record in records
    }
    conditions: dict[str, object] = {}
    all_corrupt_actions_changed = True
    for condition_id in CONDITION_IDS:
        input_changes = 0
        actor_changes = 0
        composed_changes = 0
        critic_changes = 0
        squared_action_differences: list[float] = []
        maximum_action_difference = 0.0
        critic_differences: list[float] = []
        for frame in SNAPSHOT_FRAMES:
            exact = by_key[(frame, C_EXACT)]
            observed = by_key[(frame, condition_id)]
            input_changes += int(observed["policy_input_sha256"] != exact["policy_input_sha256"])
            actor_changes += int(observed["actor_output_sha256"] != exact["actor_output_sha256"])
            composed_changes += int(
                observed["composed_action_sha256"] != exact["composed_action_sha256"]
            )
            critic_changes += int(observed["critic_output_sha256"] != exact["critic_output_sha256"])
            observed_action = np.asarray(observed["composed_action"], dtype=np.float64)
            exact_action = np.asarray(exact["composed_action"], dtype=np.float64)
            difference = observed_action - exact_action
            squared_action_differences.extend(float(value * value) for value in difference)
            maximum_action_difference = max(
                maximum_action_difference,
                float(np.max(np.abs(difference))),
            )
            critic_differences.append(
                abs(float(observed["critic_value"]) - float(exact["critic_value"]))
            )
        count = len(SNAPSHOT_FRAMES)
        conditions[condition_id] = {
            "snapshot_count": count,
            "policy_input_changed_count_vs_exact": input_changes,
            "actor_output_changed_count_vs_exact": actor_changes,
            "composed_action_changed_count_vs_exact": composed_changes,
            "critic_output_changed_count_vs_exact": critic_changes,
            "composed_action_rms_delta_vs_exact": math.sqrt(
                math.fsum(squared_action_differences) / len(squared_action_differences)
            ),
            "composed_action_max_abs_delta_vs_exact": maximum_action_difference,
            "critic_mean_abs_delta_vs_exact": math.fsum(critic_differences) / count,
        }
        if condition_id != C_EXACT:
            all_corrupt_actions_changed &= (
                input_changes == count and actor_changes == count and composed_changes == count
            )
    return {
        "mechanistic_gate_passed": all_corrupt_actions_changed,
        "mechanistic_gate_rule": (
            "every_corrupted_arm_changes_policy_input_actor_output_and_composed_"
            "float32_control_at_all_8_predeclared_observations/v1"
        ),
        "conditions": conditions,
    }


def build_reference_causal_probe_report(
    *,
    projection: MinariHumanoidProjection,
    base_controller: Callable[[np.ndarray], np.ndarray],
    residual_controller: ResidualController,
    base_receipt: LocalControllerReceipt,
    residual_receipt: LocalControllerReceipt,
) -> dict[str, object]:
    """Evaluate the four reference-input arms at exact common source observations."""

    projection.verify()
    if (
        projection.receipt.episode_id != EXPECTED_EPISODE_ID
        or projection.receipt.episode_seed != EXPECTED_EPISODE_SEED
        or projection.receipt.episode_total_steps != EXPECTED_EPISODE_STEPS
    ):
        raise ExperimentContractError("source projection differs from the frozen episode contract")
    observations = _validated_observations(projection)
    reference = np.ascontiguousarray(np.asarray(projection.reference.values, dtype="<f8"))
    projection.reference.verify()
    initial_controller_state = residual_controller.assert_integrity()
    _require_sha256(initial_controller_state, field="initial controller state SHA-256")
    records: list[dict[str, object]] = []
    snapshots: list[dict[str, object]] = []

    for frame in SNAPSHOT_FRAMES:
        observation = observations[frame]
        observation_sha256 = transform_module.float64_array_sha256(observation)
        exact_target = transform_reference_input(
            reference,
            condition_id=C_EXACT,
            current_frame=frame,
        )
        base_action = _validated_base_action(base_controller(observation))
        base_action_sha256 = _raw_sha256(base_action)
        snapshots.append(
            {
                "snapshot_frame": frame,
                "observation_sha256": observation_sha256,
                "ground_truth_window_sha256": exact_target.receipt.output_window_sha256,
                "base_action_sha256": base_action_sha256,
            }
        )
        policy_input_hashes: set[str] = set()
        for condition_id in CONDITION_IDS:
            transformed = transform_reference_input(
                reference,
                condition_id=condition_id,
                current_frame=frame,
            )
            facts = residual_controller.inference_facts(observation, transformed.window)
            repeated = residual_controller.inference_facts(observation, transformed.window)
            if facts != repeated:
                raise ExperimentContractError("residual inference is not bitwise repeatable")
            _validate_consumption(facts)
            if facts.controller_state_sha256 != initial_controller_state:
                raise ExperimentContractError("controller state changed during the probe")
            if facts.policy_input_sha256 in policy_input_hashes:
                raise ExperimentContractError(
                    "reference conditions produced duplicate policy-input bytes"
                )
            policy_input_hashes.add(facts.policy_input_sha256)
            residual_action = _residual_action(facts)
            composed_action = _compose_action(base_action, residual_action)
            records.append(
                {
                    "snapshot_frame": frame,
                    "condition_id": condition_id,
                    "observation_sha256": observation_sha256,
                    "ground_truth_window_sha256": exact_target.receipt.output_window_sha256,
                    "transformed_window_sha256": transformed.receipt.output_window_sha256,
                    "transform_receipt_sha256": transformed.receipt.sha256,
                    "transform_receipt": transformed.receipt.to_dict(),
                    "base_action_sha256": base_action_sha256,
                    "base_action": [float(value) for value in base_action],
                    **_facts_dict(facts),
                    "composed_action_sha256": _raw_sha256(composed_action),
                    "composed_action_dtype": composed_action.dtype.str,
                    "composed_action_shape": list(composed_action.shape),
                    "composed_action": [float(value) for value in composed_action],
                }
            )

    snapshot_set_sha256 = hashlib.sha256(_canonical_json(snapshots)).hexdigest()
    summary = _difference_summary(records)
    bindings = {
        "minari_projection_receipt_sha256": projection.receipt.sha256,
        "source_observations_sha256": projection.receipt.observations_sha256,
        "reference_content_sha256": projection.reference.identity.content_sha256,
        "reference_schema_sha256": projection.reference.identity.schema_sha256,
        "base_controller_content_sha256": base_receipt.content_sha256,
        "base_controller_load_receipt_sha256": base_receipt.sha256(),
        "residual_controller_content_sha256": residual_receipt.content_sha256,
        "residual_controller_load_receipt_sha256": residual_receipt.sha256(),
        "residual_controller_state_sha256": initial_controller_state,
        "minari_importer_source_sha256": _module_sha256(minari_module),
        "local_controller_loader_source_sha256": _module_sha256(controller_module),
        "reference_transform_source_sha256": _module_sha256(transform_module),
        "probe_runner_source_sha256": _module_sha256(sys.modules[__name__]),
        "matched_observation_set_sha256": snapshot_set_sha256,
    }
    design = {
        "protocol_id": PROTOCOL_ID,
        "snapshot_source": "registered_Minari_Humanoid-v5_expert_episode_0_observations/v1",
        "snapshot_frames": list(SNAPSHOT_FRAMES),
        "condition_ids": list(CONDITION_IDS),
        "shuffle_seed": transform_module.SHUFFLE_SEED,
        "shift_frames": transform_module.SHIFT_FRAMES,
        "reference_horizon_steps": transform_module.HORIZON_STEPS,
        "base_action_scale": BASE_ACTION_SCALE,
        "residual_scale": RESIDUAL_SCALE,
        "composition": "clip_float32(base_float32 + float32(0.08) * residual_float32)",
        "independent_unit": "one_existing_local_checkpoint",
        "repeated_measures": "8_predeclared_matched_source_observations_x_4_input_arms",
        "human_review_required_for_execution": False,
        "reason_review_not_required": (
            "offline_local_exploratory_input_sensitivity_probe_with_no_training_or_formal_seed_use"
        ),
    }
    design_sha256 = hashlib.sha256(_canonical_json(design)).hexdigest()
    analysis_payload = {
        "design_sha256": design_sha256,
        "bindings": bindings,
        "snapshots": snapshots,
        "records": records,
        "summary": summary,
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "experiment_id": "experiment_002a",
        "protocol_id": PROTOCOL_ID,
        "evidence_class": EVIDENCE_CLASS,
        "claim_status": CLAIM_STATUS,
        "formal_oracle_comparison_authorized": False,
        "tracker_behavior_or_stability_established": False,
        "oracle_quality_established": False,
        "dataset_license_status": "unresolved_no_redistribution_or_formal_admission",
        "design": design,
        "design_sha256": design_sha256,
        "bindings": bindings,
        "source_projection_receipt": projection.receipt.to_dict(),
        "base_controller_receipt": base_receipt.to_dict(),
        "residual_controller_receipt": residual_receipt.to_dict(),
        "snapshots": snapshots,
        "records": records,
        "summary": summary,
        "analysis_payload_sha256": hashlib.sha256(_canonical_json(analysis_payload)).hexdigest(),
        "next_gate": (
            "train_and_admit_a_stable_time_varying_reference_tracker_then_run_"
            "expected_direction_behavioral_ablation_002b"
        ),
    }


def run_reference_causal_probe(
    *,
    hdf5_path: Path,
    metadata_path: Path,
    base_controller_path: Path,
    residual_controller_path: Path,
    output_path: Path,
) -> ReferenceCausalProbeResult:
    """Load exact local assets, run the offline probe, and publish one report."""

    projection = import_registered_minari_humanoid_episode(
        hdf5_path=Path(hdf5_path),
        metadata_path=Path(metadata_path),
        episode_id=EXPECTED_EPISODE_ID,
        expected_seed=EXPECTED_EPISODE_SEED,
        expected_total_steps=EXPECTED_EPISODE_STEPS,
        expected_final_terminated=False,
        expected_final_truncated=True,
    )
    base_controller, base_receipt = load_local_behavior_cloning_controller(
        Path(base_controller_path),
        expected_sha256=LOCAL_BEHAVIOR_CLONING_SHA256,
    )
    residual_controller, residual_receipt = load_local_reference_residual_controller(
        Path(residual_controller_path),
        expected_sha256=LOCAL_REFERENCE_RESIDUAL_SHA256,
    )
    report = build_reference_causal_probe_report(
        projection=projection,
        base_controller=base_controller,
        residual_controller=residual_controller,
        base_receipt=base_receipt,
        residual_receipt=residual_receipt,
    )
    summary = report.get("summary")
    if not isinstance(summary, dict) or not isinstance(
        summary.get("mechanistic_gate_passed"), bool
    ):
        raise ExperimentContractError("probe report is missing its mechanistic gate")
    published = publish_json_without_overwrite(Path(output_path), report)
    return ReferenceCausalProbeResult(
        path=published.path,
        file_sha256=published.sha256,
        analysis_payload_sha256=str(report["analysis_payload_sha256"]),
        mechanistic_gate_passed=summary["mechanistic_gate_passed"],
    )


def _default_cache_root() -> Path:
    cache = os.environ.get("XDG_CACHE_HOME")
    return Path(cache) if cache else Path.home() / ".cache"


def _parser() -> argparse.ArgumentParser:
    cache = _default_cache_root() / "humanoid-harness"
    data = cache / "minari" / "mujoco" / "humanoid" / "expert-v0" / "data"
    models = cache / "local_models"
    parser = argparse.ArgumentParser(
        prog="humanoid-reference-probe",
        description="Run the local matched-observation numeric-reference sensitivity probe.",
    )
    parser.add_argument("--hdf5", type=Path, default=data / "main_data.hdf5")
    parser.add_argument("--metadata", type=Path, default=data / "metadata.json")
    parser.add_argument("--base-controller", type=Path, default=models / "minari_bc_v0.npz")
    parser.add_argument(
        "--residual-controller",
        type=Path,
        default=models / "reference_residual_ppo_v0.npz",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_reference_causal_probe(
            hdf5_path=args.hdf5,
            metadata_path=args.metadata,
            base_controller_path=args.base_controller,
            residual_controller_path=args.residual_controller,
            output_path=args.output,
        )
    except (ExperimentContractError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "CLAIM_STATUS",
    "EVIDENCE_CLASS",
    "EXPECTED_EPISODE_ID",
    "EXPECTED_EPISODE_SEED",
    "EXPECTED_EPISODE_STEPS",
    "PROTOCOL_ID",
    "REPORT_SCHEMA_VERSION",
    "SNAPSHOT_FRAMES",
    "ReferenceCausalProbeResult",
    "build_reference_causal_probe_report",
    "main",
    "run_reference_causal_probe",
]


if __name__ == "__main__":
    raise SystemExit(main())
