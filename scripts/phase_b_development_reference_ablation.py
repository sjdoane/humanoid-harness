#!/usr/bin/env python3
"""Run the preregistered T1 development-only reference-input ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

import oracle_composition
from oracle_composition.contracts.reference_identity_v2 import (
    array_sha256,
    canonical_json_bytes,
)
from oracle_composition.development.reference_ablation import (
    CLAIM_CEILING,
    EVIDENCE_CLASS,
    PROTOCOL_ID,
    DevelopmentAblationError,
    DevelopmentProtocol,
    PreparedReferenceTransform,
    SmokeExportLineage,
    load_bound_smoke_actor,
    load_development_protocol,
    matched_action_delta,
    prepare_reference_transform,
    prepared_reference_window,
    summarize_matched_action_deltas,
    validate_development_corpus,
    validate_smoke_export,
)
from oracle_composition.envs.reference_corpus import make_reference_corpus_env
from oracle_composition.experiments.artifact_io import (
    PublishedArtifact,
    publish_bytes_without_overwrite,
)
from oracle_composition.experiments.reference_input_transforms import CONDITION_IDS
from oracle_composition.phase_b.isolation import validate_executing_modules
from oracle_composition.phase_b.policy import compose_policy_input
from oracle_composition.phase_b.protected_metrics import (
    evaluator_mass_center_x_m,
    evaluator_state_record,
    protected_step_record,
    protected_trace_sha256,
    recompute_protected_episode,
    select_evaluator_nearest_phase,
)
from oracle_composition.phase_b.reference_runtime import load_v2_reference_clip
from oracle_composition.phase_b.supervision import RuntimeSourceSnapshot, inspect_runtime_sources
from oracle_composition.tracking.humanoid_reference import (
    tracking_state,
    tracking_state_with_bounded_reset_orientation,
    validate_humanoid_actuator_abi,
)

TRACE_SCHEMA_ID = "humanoid_phase_b_t1_development_reference_ablation_trace/v1"
MANIFEST_SCHEMA_ID = "humanoid_phase_b_t1_development_reference_ablation_manifest/v1"
RESULT_SCHEMA_ID = "humanoid_phase_b_t1_development_reference_ablation_result/v1"
HORIZON = 1_000
SWITCHES = {300: "simple", 600: "expert"}
FOOT_GEOMS = frozenset(("left_foot", "right_foot"))


def _resolve(root: Path, path: Path) -> Path:
    return (
        (root / path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True)
    )


def _fresh_output(path: Path) -> Path:
    requested = Path(os.path.abspath(path))
    if requested.exists() or requested.is_symlink():
        raise DevelopmentAblationError("development output must be fresh and no-overwrite")
    if requested.parent.is_symlink() or not requested.parent.is_dir():
        raise DevelopmentAblationError("development output parent must be a real directory")
    requested.mkdir(mode=0o700)
    return requested


def _artifact_record(artifact: PublishedArtifact) -> dict[str, object]:
    return {
        "byte_count": artifact.byte_count,
        "filename": artifact.path.name,
        "sha256": artifact.sha256,
    }


def _assert_checkout_import(repository_root: Path) -> str:
    package_file = Path(oracle_composition.__file__).resolve(strict=True)
    expected_root = (repository_root / "src/oracle_composition").resolve(strict=True)
    if not package_file.is_relative_to(expected_root):
        raise DevelopmentAblationError(
            "oracle_composition import does not resolve inside the selected checkout"
        )
    return str(package_file)


def _torso_up(quaternion: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    norm = w * w + x * x + y * y + z * z
    if norm <= 0.0:
        raise DevelopmentAblationError("development torso quaternion has zero norm")
    result = (w * w - x * x - y * y + z * z) / norm
    if not math.isfinite(result):
        raise DevelopmentAblationError("development torso-up metric is non-finite")
    return result


def _forbidden_contacts(environment: object, step: int) -> list[dict[str, object]]:
    physical = environment.unwrapped
    samples = getattr(physical, "last_full_contact_samples", None)
    names = getattr(physical, "_reference_corpus_geom_names", None)
    if type(samples) is not tuple or type(names) is not tuple:
        raise DevelopmentAblationError("development contact instrumentation is unavailable")
    result = []
    for sample in samples:
        first = names[sample.geom1_id]
        second = names[sample.geom2_id]
        if "floor" not in {first, second}:
            continue
        other = second if first == "floor" else first
        if other not in FOOT_GEOMS:
            result.append(
                {
                    "geom": other,
                    "physics_substep_index": sample.physics_substep_index,
                    "step": step,
                }
            )
    return result


def _prepare_references(
    corpus_root: Path,
    *,
    block: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, PreparedReferenceTransform]],
    dict[str, dict[str, object]],
]:
    references = {}
    transforms = {}
    identities = {}
    for behavior in ("expert", "simple"):
        clip = load_v2_reference_clip(corpus_root, block=block, behavior=behavior)
        references[behavior] = clip.reference_rows
        identities[behavior] = {
            "bundle_sha256": clip.bundle_sha256,
            "payload_sha256": clip.payload_sha256,
            "reference_identity_sha256": clip.reference_identity_sha256,
        }
        transforms[behavior] = {
            condition: prepare_reference_transform(
                clip.reference_rows,
                condition_id=condition,
            )
            for condition in CONDITION_IDS
        }
    return references, transforms, identities


def _matched_rows(
    actor: object,
    observation: np.ndarray,
    *,
    exact_action: np.ndarray,
    transforms: Mapping[str, PreparedReferenceTransform],
    behavior: str,
    phase: int,
    step: int,
) -> list[dict[str, object]]:
    rows = []
    for condition in CONDITION_IDS:
        window = prepared_reference_window(transforms[condition], current_frame=phase)
        policy_input = compose_policy_input(
            observation,
            np.ascontiguousarray(window.values, dtype="<f4"),
        )
        compared = np.ascontiguousarray(actor.act(policy_input, epsilon=None).physical, dtype="<f4")
        rows.append(
            {
                "behavior": behavior,
                "condition_id": condition,
                "phase": phase,
                "step": step,
                **matched_action_delta(exact_action, compared),
            }
        )
    return rows


def _run_arm(
    actor: object,
    *,
    actor_sha256: str,
    block: int,
    condition: str,
    references: Mapping[str, np.ndarray],
    transforms: Mapping[str, Mapping[str, PreparedReferenceTransform]],
    reference_identities: Mapping[str, Mapping[str, object]],
    protocol: DevelopmentProtocol,
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    environment = make_reference_corpus_env()
    objective_steps: list[dict[str, object]] = []
    policy_inputs: list[dict[str, object]] = []
    switches: list[dict[str, object]] = []
    matched_rows: list[dict[str, object]] = []
    try:
        raw_observation, _reset_info = environment.reset(seed=block)
        observation = np.ascontiguousarray(raw_observation, dtype="<f4")
        abi = validate_humanoid_actuator_abi(environment)
        physical = environment.unwrapped
        body_mass = np.ascontiguousarray(physical.model.body_mass, dtype="<f8")
        current_state = tracking_state_with_bounded_reset_orientation(environment, abi)
        previous_x = float(current_state.root_position_world_m[0])
        active_behavior = "expert"
        active_phase = 0
        for step in range(HORIZON):
            if step in SWITCHES:
                target_behavior = SWITCHES[step]
                transfer = select_evaluator_nearest_phase(
                    state=evaluator_state_record(current_state),
                    target_rows=references[target_behavior],
                    task_step=step,
                    source_behavior=active_behavior,
                    target_behavior=target_behavior,
                    reason="development_preregistered_direct_switch",
                )
                active_behavior = target_behavior
                active_phase = int(transfer["selected_phase"])
                switches.append(
                    {
                        "boundary": step,
                        "from_behavior": transfer["source_behavior"],
                        "selected_normalized_errors": transfer["selected_normalized_errors"],
                        "selected_phase": transfer["selected_phase"],
                        "selected_score": transfer["selected_score"],
                        "to_behavior": transfer["target_behavior"],
                    }
                )

            prepared = transforms[active_behavior][condition]
            window = prepared_reference_window(prepared, current_frame=active_phase)
            window_f32 = np.ascontiguousarray(window.values, dtype="<f4")
            policy_input = compose_policy_input(observation, window_f32)
            action = np.ascontiguousarray(
                actor.act(policy_input, epsilon=None).physical,
                dtype="<f4",
            )
            if (
                not np.isfinite(action).all()
                or np.any(action < np.float32(-0.4))
                or np.any(action > np.float32(0.4))
            ):
                raise DevelopmentAblationError("development actor action is out of bounds")
            if condition == CONDITION_IDS[0]:
                matched_rows.extend(
                    _matched_rows(
                        actor,
                        observation,
                        exact_action=action,
                        transforms=transforms[active_behavior],
                        behavior=active_behavior,
                        phase=active_phase,
                        step=step,
                    )
                )

            policy_inputs.append(
                {
                    "action_sha256": array_sha256(action),
                    "actor_input_sha256": array_sha256(policy_input.array),
                    "behavior": active_behavior,
                    "condition_id": condition,
                    "observation_sha256": array_sha256(observation),
                    "phase": active_phase,
                    "policy_window_float32_sha256": array_sha256(window_f32),
                    "policy_window_float64_sha256": window.sha256,
                    "step": step,
                    "window_source_indices": list(window.source_indices),
                    "window_timeline_indices": list(window.timeline_indices),
                }
            )

            body_xipos_before = np.ascontiguousarray(physical.data.xipos, dtype="<f8").copy()
            raw_next, _stock_reward, terminated, truncated, _step_info = environment.step(action)
            body_xipos_after = np.ascontiguousarray(physical.data.xipos, dtype="<f8").copy()
            evaluator_mass_center_x_m(body_mass, body_xipos_before)
            evaluator_mass_center_x_m(body_mass, body_xipos_after)
            observation = np.ascontiguousarray(raw_next, dtype="<f4")
            state = tracking_state(environment, abi)
            target_phase = min(active_phase + 1, HORIZON)
            exact_target = references[active_behavior][target_phase]
            contacts = _forbidden_contacts(environment, step)
            fallen = (
                not 1.0 <= state.root_height_m <= 2.0
                or _torso_up(state.root_orientation_wxyz) < 0.5
            )
            objective_steps.append(
                protected_step_record(
                    step=step,
                    state=state,
                    reference_behavior=active_behavior,
                    reference_index=target_phase,
                    reference_row=exact_target,
                    action=action,
                    root_x_before_m=previous_x,
                    body_mass=body_mass,
                    body_xipos_before=body_xipos_before,
                    body_xipos_after=body_xipos_after,
                    forbidden_contacts=contacts,
                    fallen=fallen,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )
            )
            previous_x = float(state.root_position_world_m[0])
            if bool(terminated) or (bool(truncated) and step + 1 != HORIZON):
                raise DevelopmentAblationError(
                    "development rollout ended before the declared 1,000 steps"
                )
            active_phase = target_phase
            current_state = state

        aggregates = recompute_protected_episode(
            steps=objective_steps,
            cell="fixed_round_trip",
            switches=switches,
            segment_targets_m_s=protocol.target_speeds_m_s,
        )
        trace = {
            "actor_export_sha256": actor_sha256,
            "block": block,
            "claim_ceiling": CLAIM_CEILING,
            "condition_id": condition,
            "development_only": True,
            "evidence_class": EVIDENCE_CLASS,
            "matched_state_action_deltas": matched_rows,
            "objective_metric_kernel": (
                "phase_b_protected_metric_math_reused_without_protected_split_or_gate"
            ),
            "objective_reference": "untransformed_exact_reference_rows",
            "objective_steps": objective_steps,
            "policy_inputs": policy_inputs,
            "policy_reference_factor": "actor_visible_8x45_window_only",
            "policy_seed": 121901,
            "promotable": False,
            "protocol_id": PROTOCOL_ID,
            "reference_identities": {
                name: dict(value) for name, value in reference_identities.items()
            },
            "reference_transform_precomputations": {
                behavior: dict(transforms[behavior][condition].receipt)
                for behavior in ("expert", "simple")
            },
            "schema_version": 1,
            "switch_records": switches,
            "task_success": None,
            "trace_schema_id": TRACE_SCHEMA_ID,
        }
        trace["objective_steps_sha256"] = protected_trace_sha256(objective_steps)
        trace["policy_inputs_sha256"] = hashlib.sha256(
            canonical_json_bytes(policy_inputs)
        ).hexdigest()
        canonical_json_bytes(trace)
        summary = {
            "action_bounds_ok": aggregates["action_bounds_ok"],
            "block": block,
            "condition_id": condition,
            "fall": aggregates["fall"],
            "forbidden_contact_count": len(aggregates["forbidden_contacts"]),
            "observed_steps": aggregates["observed_steps"],
            "resynchronization_records": aggregates["resynchronization_records"],
            "segment_errors": aggregates["segment_errors"],
            "settled_state_normalized_error": aggregates["settled_state_normalized_error"],
            "settle_latency_steps": aggregates["settle_latency_steps"],
            "six_tracking_errors": aggregates["six_tracking_errors"],
            "task_success": None,
            "time_to_first_failure_steps": aggregates["time_to_first_failure_steps"],
            "transition_window_error": aggregates["transition_window_error"],
        }
        return trace, summary, matched_rows
    finally:
        environment.close()


def _manifest_value(
    *,
    package_file: str,
    protocol: DevelopmentProtocol,
    lineage: SmokeExportLineage,
    corpus_bindings: Mapping[str, Mapping[str, object]],
    source_snapshot: RuntimeSourceSnapshot,
) -> dict[str, object]:
    return {
        "actor_export": {
            "byte_count": lineage.actor_byte_count,
            "sha256": lineage.actor_sha256,
        },
        "actor_visible_factor": "reference_window_only",
        "calibration_inputs": None,
        "claim_ceiling": CLAIM_CEILING,
        "conditions": list(protocol.conditions),
        "corpus_bindings": {name: dict(value) for name, value in sorted(corpus_bindings.items())},
        "development_blocks": list(protocol.development_blocks),
        "evidence_class": EVIDENCE_CLASS,
        "held_out_inputs_used": False,
        "manifest_schema_id": MANIFEST_SCHEMA_ID,
        "objective_reference": "untransformed_exact_reference_rows",
        "package_import_path": package_file,
        "policy_seed": 121901,
        "promotable": False,
        "protocol": {
            "byte_count": protocol.byte_count,
            "path": str(protocol.path),
            "sha256": protocol.sha256,
        },
        "schema_version": 1,
        "smoke_lineage": {
            "actor_sha256": lineage.actor_sha256,
            "bindings": {name: dict(value) for name, value in lineage.bindings.items()},
            "checkpoint_sha256": lineage.checkpoint_sha256,
            "execution_manifest_sha256": lineage.execution_manifest_sha256,
            "reference_sha256": lineage.report_inputs["reference_sha256"],
            "run_directory": str(lineage.run_directory),
            "sealed_input_lineage_sha256": lineage.report_inputs["sealed_input_lineage_sha256"],
            "training_facts_sha256": lineage.training_facts_sha256,
        },
        "runtime_source_snapshot": dict(source_snapshot.value),
        "runtime_source_snapshot_sha256": source_snapshot.sha256,
        "task_success_scoring": False,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    repository_root = Path(args.repository_root).resolve(strict=True)
    package_file = _assert_checkout_import(repository_root)
    protocol_path = _resolve(repository_root, args.protocol)
    corpus_root = _resolve(repository_root, args.corpus_root)
    smoke_run = _resolve(repository_root, args.smoke_run)
    protocol = load_development_protocol(protocol_path)
    lineage = validate_smoke_export(smoke_run)
    corpus_bindings = validate_development_corpus(
        corpus_root,
        lineage=lineage,
        protocol=protocol,
    )
    source_snapshot = inspect_runtime_sources(repository_root)
    validate_executing_modules(repository_root, source_snapshot.value["source_sha256"])
    output_path = args.output if args.output.is_absolute() else repository_root / args.output
    output = _fresh_output(output_path)
    manifest_value = _manifest_value(
        package_file=package_file,
        protocol=protocol,
        lineage=lineage,
        corpus_bindings=corpus_bindings,
        source_snapshot=source_snapshot,
    )
    manifest = publish_bytes_without_overwrite(
        output / "development_manifest_v1.json",
        canonical_json_bytes(manifest_value),
    )

    loaded = load_bound_smoke_actor(lineage)
    actor = getattr(loaded, "actor", None)
    if actor is None or getattr(loaded, "content_sha256", None) != lineage.actor_sha256:
        raise DevelopmentAblationError("strict actor loader result differs from its binding")

    arms = []
    trace_artifacts = []
    matched_by_block = {}
    for block in protocol.development_blocks:
        references, transforms, reference_identities = _prepare_references(
            corpus_root,
            block=block,
        )
        for condition in protocol.conditions:
            trace, arm, matched_rows = _run_arm(
                actor,
                actor_sha256=lineage.actor_sha256,
                block=block,
                condition=condition,
                references=references,
                transforms=transforms,
                reference_identities=reference_identities,
                protocol=protocol,
            )
            trace_artifact = publish_bytes_without_overwrite(
                output / f"trace_block_{block}_{condition}.json",
                canonical_json_bytes(trace),
            )
            arm["trace"] = _artifact_record(trace_artifact)
            arms.append(arm)
            trace_artifacts.append(_artifact_record(trace_artifact))
            if condition == CONDITION_IDS[0]:
                matched_by_block[str(block)] = {
                    name: dict(value)
                    for name, value in summarize_matched_action_deltas(matched_rows).items()
                }

    final_source_snapshot = inspect_runtime_sources(repository_root)
    if final_source_snapshot.sha256 != source_snapshot.sha256:
        raise DevelopmentAblationError(
            "runtime source snapshot changed during development ablation"
        )
    validate_executing_modules(repository_root, final_source_snapshot.value["source_sha256"])
    result = {
        "actor_export_sha256": lineage.actor_sha256,
        "arms": arms,
        "claim_ceiling": CLAIM_CEILING,
        "completed_arm_count": len(arms),
        "development_only": True,
        "evidence_class": EVIDENCE_CLASS,
        "held_out_inputs_used": False,
        "manifest": _artifact_record(manifest),
        "matched_state_action_deltas": matched_by_block,
        "planned_arm_count": len(protocol.development_blocks) * len(protocol.conditions),
        "promotable": False,
        "result_schema_id": RESULT_SCHEMA_ID,
        "runtime_source_snapshot_sha256": source_snapshot.sha256,
        "schema_version": 1,
        "status": "succeeded",
        "task_success": None,
        "trace_artifacts": trace_artifacts,
    }
    published = publish_bytes_without_overwrite(
        output / "development_result_v1.json",
        canonical_json_bytes(result),
    )
    return {**result, "result_artifact": _artifact_record(published)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only in-sample reference-input sensitivity; never a protected "
            "utility or promotion path."
        )
    )
    repository_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--repository-root", type=Path, default=repository_root)
    parser.add_argument("--smoke-run", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(
            "experiments/003_composition_speed_profile/phase_b/"
            "development_reference_ablation_v1.json"
        ),
    )
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    result = run(_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
