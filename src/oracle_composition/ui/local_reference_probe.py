"""Fail-closed local view of the Experiment 002A report."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import struct
from functools import lru_cache
from pathlib import Path, PurePath

REFERENCE_PROBE_PATH = Path("artifacts/experiment_002a/reference_causal_probe.json")

_REPORT_LIMIT = 1_000_000
_SOURCE_LIMIT = 2 * 1024 * 1024
_REVIEWED_REPORT_SHA256 = "0af686b506ac227b5ffcc2196dbe74dd2cbfef941d25139d0cb0210182a5150a"
_PROTOCOL_ID = "experiment_002a_matched_observation_numeric_reference_probe/v1"
_CLAIM_STATUS = "numeric_reference_input_sensitivity_only_not_tracking_or_oracle_quality"
_AUTHORITY = "local_exploratory_mechanistic_probe"
_SNAPSHOT_FRAMES = (0, 125, 250, 375, 500, 625, 750, 875)
_CONDITION_IDS = ("C_exact", "C_zero_input", "C_shuffle_input", "C_shift_input")
_CONDITION_LABELS = {
    "C_exact": "Exact reference",
    "C_zero_input": "Zero input",
    "C_shuffle_input": "Deterministic shuffle",
    "C_shift_input": "+250-frame shift",
}
_GATE_RULE = (
    "every_corrupted_arm_changes_policy_input_actor_output_and_composed_"
    "float32_control_at_all_8_predeclared_observations/v1"
)
_NEXT_GATE = (
    "train_and_admit_a_stable_time_varying_reference_tracker_then_run_"
    "expected_direction_behavioral_ablation_002b"
)
_SOURCE_FILES = {
    "minari_importer_source_sha256": Path("src/oracle_composition/sources/minari_humanoid.py"),
    "local_controller_loader_source_sha256": Path(
        "src/oracle_composition/experiments/local_controllers.py"
    ),
    "reference_transform_source_sha256": Path(
        "src/oracle_composition/experiments/reference_input_transforms.py"
    ),
    "probe_runner_source_sha256": Path(
        "src/oracle_composition/experiments/reference_causal_probe.py"
    ),
}

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "experiment_id",
        "protocol_id",
        "evidence_class",
        "claim_status",
        "formal_oracle_comparison_authorized",
        "tracker_behavior_or_stability_established",
        "oracle_quality_established",
        "dataset_license_status",
        "design",
        "design_sha256",
        "bindings",
        "source_projection_receipt",
        "base_controller_receipt",
        "residual_controller_receipt",
        "snapshots",
        "records",
        "summary",
        "analysis_payload_sha256",
        "next_gate",
    }
)
_DESIGN_FIELDS = frozenset(
    {
        "protocol_id",
        "snapshot_source",
        "snapshot_frames",
        "condition_ids",
        "shuffle_seed",
        "shift_frames",
        "reference_horizon_steps",
        "base_action_scale",
        "residual_scale",
        "composition",
        "independent_unit",
        "repeated_measures",
        "human_review_required_for_execution",
        "reason_review_not_required",
    }
)
_BINDING_FIELDS = frozenset(
    {
        "minari_projection_receipt_sha256",
        "source_observations_sha256",
        "reference_content_sha256",
        "reference_schema_sha256",
        "base_controller_content_sha256",
        "base_controller_load_receipt_sha256",
        "residual_controller_content_sha256",
        "residual_controller_load_receipt_sha256",
        "residual_controller_state_sha256",
        "minari_importer_source_sha256",
        "local_controller_loader_source_sha256",
        "reference_transform_source_sha256",
        "probe_runner_source_sha256",
        "matched_observation_set_sha256",
    }
)
_SNAPSHOT_FIELDS = frozenset(
    {
        "snapshot_frame",
        "observation_sha256",
        "ground_truth_window_sha256",
        "base_action_sha256",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "snapshot_frame",
        "condition_id",
        "observation_sha256",
        "ground_truth_window_sha256",
        "transformed_window_sha256",
        "transform_receipt_sha256",
        "transform_receipt",
        "base_action_sha256",
        "base_action",
        "policy_input_sha256",
        "policy_input_shape",
        "policy_input_dtype",
        "actor_input_sha256",
        "critic_input_sha256",
        "controller_state_sha256",
        "actor_output_sha256",
        "actor_output_shape",
        "actor_output_dtype",
        "actor_output",
        "critic_output_sha256",
        "critic_output_shape",
        "critic_output_dtype",
        "critic_value",
        "composed_action_sha256",
        "composed_action_dtype",
        "composed_action_shape",
        "composed_action",
    }
)
_TRANSFORM_FIELDS = frozenset(
    {
        "schema_version",
        "algorithm_id",
        "array_hash_algorithm_id",
        "index_map_hash_algorithm_id",
        "condition_id",
        "dtype",
        "order",
        "source_shape",
        "source_sha256",
        "transformed_full_sequence_sha256",
        "parameters",
        "index_map",
        "index_map_sha256",
        "current_frame",
        "window_timeline_indices",
        "window_source_indices",
        "output_window_shape",
        "output_window_sha256",
    }
)
_SUMMARY_FIELDS = frozenset({"mechanistic_gate_passed", "mechanistic_gate_rule", "conditions"})
_CONDITION_SUMMARY_FIELDS = frozenset(
    {
        "snapshot_count",
        "policy_input_changed_count_vs_exact",
        "actor_output_changed_count_vs_exact",
        "composed_action_changed_count_vs_exact",
        "critic_output_changed_count_vs_exact",
        "composed_action_rms_delta_vs_exact",
        "composed_action_max_abs_delta_vs_exact",
        "critic_mean_abs_delta_vs_exact",
    }
)
_CONTROLLER_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "format_id",
        "architecture_id",
        "claim_status",
        "filename",
        "file_size_bytes",
        "expected_content_sha256",
        "content_sha256",
        "registered_content_match",
        "loader_source_sha256",
        "numpy_version",
        "torch_version",
        "gymnasium_version",
        "stable_baselines3_version",
        "observation_width",
        "action_width",
        "reference_width",
        "reference_horizon_steps",
        "residual_scale",
        "inference_output_contract",
    }
)
_PROJECTION_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "evidence_tier",
        "admission_status",
        "claim_ceiling",
        "dynamics_feasibility_established",
        "source_provenance_class",
        "source_origin_verified",
        "asserted_source_repository",
        "asserted_source_commit",
        "source_record_sha256",
        "source_record_bytes",
        "dataset_id",
        "metadata_declared_generation_code_url",
        "hdf5_sha256",
        "hdf5_bytes",
        "metadata_sha256",
        "metadata_bytes",
        "environment_id",
        "environment_entry_point",
        "minari_version",
        "declared_requirements",
        "episode_id",
        "episode_seed",
        "episode_total_steps",
        "episode_observation_shape",
        "episode_action_shape",
        "observations_sha256",
        "actions_sha256",
        "rewards_sha256",
        "terminations_sha256",
        "truncations_sha256",
        "ignored_observation_fields_sha256",
        "root_position_xy_evidence",
        "root_position_xy_reconstructed",
        "projection_id",
        "projection_indices",
        "projection_mapping_sha256",
        "reference_artifact_id",
        "reference_content_sha256",
        "reference_schema_sha256",
        "reference_shape",
        "reference_cadence_hz",
        "tier_d_certificate_sha256",
    }
)


class LocalReferenceProbeError(ValueError):
    """Raised when a local Experiment 002A report fails validation."""


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
        raise LocalReferenceProbeError("report contains a non-canonical JSON value") from exc


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_project_file(root: Path, relative: Path, *, limit: int) -> bytes:
    if relative.is_absolute() or not relative.parts:
        raise LocalReferenceProbeError("unsafe local report path")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise LocalReferenceProbeError("unsafe local report path")

    descriptors: list[int] = []
    try:
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        current = os.open(root.resolve(strict=True), directory_flags)
        descriptors.append(current)
        for part in relative.parts[:-1]:
            current = os.open(part, directory_flags, dir_fd=current)
            descriptors.append(current)
        file_flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(relative.name, file_flags, dir_fd=current)
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise LocalReferenceProbeError("local report path is not a non-empty regular file")
        if before.st_size > limit:
            raise LocalReferenceProbeError("local report exceeds its size limit")

        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 64 * 1024))
            if not chunk:
                raise LocalReferenceProbeError("local report changed while it was read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise LocalReferenceProbeError("local report changed while it was read")
        after = os.fstat(descriptor)
        identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in identity):
            raise LocalReferenceProbeError("local report changed while it was read")
        return b"".join(chunks)
    except FileNotFoundError:
        raise
    except LocalReferenceProbeError:
        raise
    except OSError as exc:
        raise LocalReferenceProbeError("local report could not be read safely") from exc
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _reject_constant(value: str) -> None:
    raise LocalReferenceProbeError(f"non-finite JSON constant is forbidden: {value}")


def _without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise LocalReferenceProbeError(f"duplicate report JSON key: {key!r}")
        result[key] = value
    return result


def _decode_report(source: bytes) -> dict[str, object]:
    try:
        value = json.loads(
            source,
            parse_constant=_reject_constant,
            object_pairs_hook=_without_duplicates,
        )
    except LocalReferenceProbeError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise LocalReferenceProbeError("local report JSON is invalid") from exc
    if not isinstance(value, dict):
        raise LocalReferenceProbeError("local report must be a JSON object")
    return value


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise LocalReferenceProbeError(f"{field} must be an object")
    return value


def _exact_fields(value: dict[str, object], expected: frozenset[str], field: str) -> None:
    if set(value) != expected:
        raise LocalReferenceProbeError(f"{field} fields do not match the fixed schema")


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise LocalReferenceProbeError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise LocalReferenceProbeError(
            f"{field} must be an integer greater than or equal to {minimum}"
        )
    return value


def _number(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise LocalReferenceProbeError(f"{field} must be numeric")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise LocalReferenceProbeError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise LocalReferenceProbeError(f"{field} must be finite")
    return result


def _digest(value: object, field: str) -> str:
    result = _string(value, field)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise LocalReferenceProbeError(f"{field} must be a lowercase SHA-256")
    return result


def _vector(
    value: object, field: str, *, length: int, lower: float, upper: float
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise LocalReferenceProbeError(f"{field} must be a {length}-element list")
    result = tuple(_number(item, field) for item in value)
    if any(item < lower or item > upper for item in result):
        raise LocalReferenceProbeError(f"{field} exceeds its declared bounds")
    return result


def _float32_bytes(values: tuple[float, ...], field: str) -> bytes:
    try:
        return struct.pack(f"<{len(values)}f", *values)
    except (OverflowError, struct.error) as exc:
        raise LocalReferenceProbeError(f"{field} cannot be represented as float32") from exc


def _f32(value: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0]
    except (OverflowError, struct.error) as exc:
        raise LocalReferenceProbeError("composed action is not representable as float32") from exc


def _expected_design() -> dict[str, object]:
    return {
        "protocol_id": _PROTOCOL_ID,
        "snapshot_source": "registered_Minari_Humanoid-v5_expert_episode_0_observations/v1",
        "snapshot_frames": list(_SNAPSHOT_FRAMES),
        "condition_ids": list(_CONDITION_IDS),
        "shuffle_seed": 260_907,
        "shift_frames": 250,
        "reference_horizon_steps": 8,
        "base_action_scale": 0.4,
        "residual_scale": 0.08,
        "composition": "clip_float32(base_float32 + float32(0.08) * residual_float32)",
        "independent_unit": "one_existing_local_checkpoint",
        "repeated_measures": "8_predeclared_matched_source_observations_x_4_input_arms",
        "human_review_required_for_execution": False,
        "reason_review_not_required": (
            "offline_local_exploratory_input_sensitivity_probe_with_no_training_or_formal_seed_use"
        ),
    }


@lru_cache(maxsize=1)
def _shuffle_index_map() -> tuple[int, ...]:
    def rank(frame_index: int) -> tuple[bytes, int]:
        payload = _canonical_json(
            {
                "algorithm_id": "sha256_ranked_full_frame_indices/v1",
                "frame_index": frame_index,
                "n_frames": 1001,
                "seed": 260_907,
            }
        )
        return hashlib.sha256(payload).digest(), frame_index

    return tuple(sorted(range(1001), key=rank))


def _expected_index_map(condition_id: str) -> tuple[int | None, ...]:
    if condition_id == "C_exact":
        return tuple(range(1001))
    if condition_id == "C_zero_input":
        return (None,) * 1001
    if condition_id == "C_shuffle_input":
        return _shuffle_index_map()
    return tuple((index + 250) % 1001 for index in range(1001))


def _expected_transform_parameters(condition_id: str) -> dict[str, object]:
    common: dict[str, object] = {"horizon_steps": 8, "reference_width": 45}
    if condition_id == "C_exact":
        return {**common, "operation": "identity"}
    if condition_id == "C_zero_input":
        return {
            **common,
            "fill_float64_bits_hex": "0000000000000000",
            "operation": "positive_zero_fill",
        }
    if condition_id == "C_shuffle_input":
        return {
            **common,
            "operation": "output_frame_i_reads_source_frame_at_rank_i",
            "ranking_algorithm_id": "sha256_ranked_full_frame_indices/v1",
            "seed": 260_907,
        }
    return {
        **common,
        "operation": "output_frame_i_reads_source_frame_i_plus_shift_mod_T",
        "shift_algorithm_id": "cyclic_source_index_plus_250/v1",
        "shift_frames": 250,
    }


def _validate_transform_receipt(
    value: object,
    *,
    frame: int,
    condition_id: str,
) -> dict[str, object]:
    receipt = _mapping(value, "transform receipt")
    _exact_fields(receipt, _TRANSFORM_FIELDS, "transform receipt")
    fixed = {
        "schema_version": 1,
        "algorithm_id": "experiment_002a_numeric_reference_input/v1",
        "array_hash_algorithm_id": "framed_c_order_little_endian_float64_array/v1",
        "index_map_hash_algorithm_id": "canonical_json_index_map/v1",
        "condition_id": condition_id,
        "dtype": "<f8",
        "order": "C",
        "source_shape": [1001, 45],
        "current_frame": frame,
        "output_window_shape": [8, 45],
    }
    if any(receipt.get(key) != expected for key, expected in fixed.items()):
        raise LocalReferenceProbeError("transform receipt differs from the fixed design")
    if receipt.get("parameters") != _expected_transform_parameters(condition_id):
        raise LocalReferenceProbeError("transform parameters differ from the fixed design")

    expected_map = _expected_index_map(condition_id)
    if receipt.get("index_map") != list(expected_map):
        raise LocalReferenceProbeError("transform index map differs from the fixed design")
    expected_index_hash = _sha256(
        _canonical_json(
            {
                "algorithm_id": "canonical_json_index_map/v1",
                "index_map": list(expected_map),
            }
        )
    )
    if (
        _digest(receipt.get("index_map_sha256"), "transform index-map SHA-256")
        != expected_index_hash
    ):
        raise LocalReferenceProbeError("transform index-map SHA-256 is inconsistent")

    timeline = tuple(min(frame + offset, 1000) for offset in range(8))
    sources = tuple(expected_map[index] for index in timeline)
    if receipt.get("window_timeline_indices") != list(timeline):
        raise LocalReferenceProbeError("transform timeline differs from the fixed design")
    if receipt.get("window_source_indices") != list(sources):
        raise LocalReferenceProbeError("transform source indices differ from the fixed design")
    for field in (
        "source_sha256",
        "transformed_full_sequence_sha256",
        "output_window_sha256",
    ):
        _digest(receipt.get(field), f"transform {field}")
    if condition_id == "C_exact" and (
        receipt["transformed_full_sequence_sha256"] != receipt["source_sha256"]
    ):
        raise LocalReferenceProbeError("exact transform changed the full reference sequence")
    return receipt


def _validate_controller_receipt(
    value: object,
    *,
    role: str,
    bindings: dict[str, object],
) -> None:
    receipt = _mapping(value, f"{role} controller receipt")
    _exact_fields(receipt, _CONTROLLER_RECEIPT_FIELDS, f"{role} controller receipt")
    if receipt.get("schema_version") != 1:
        raise LocalReferenceProbeError("controller receipt has the wrong schema version")
    if receipt.get("format_id") != "strict_npz_npy1_c_order_no_pickle/v1":
        raise LocalReferenceProbeError("controller receipt has the wrong data-only format")
    if receipt.get("claim_status") != "local_exploration_only_not_admitted":
        raise LocalReferenceProbeError("controller receipt exceeds the local claim boundary")
    filename = _string(receipt.get("filename"), "controller filename")
    if PurePath(filename).name != filename:
        raise LocalReferenceProbeError("controller receipt contains a path")
    _integer(receipt.get("file_size_bytes"), "controller file size", minimum=1)
    for field in (
        "expected_content_sha256",
        "content_sha256",
        "loader_source_sha256",
    ):
        _digest(receipt.get(field), f"controller {field}")
    if (
        receipt["expected_content_sha256"] != receipt["content_sha256"]
        or receipt.get("registered_content_match") is not True
        or receipt["loader_source_sha256"] != bindings["local_controller_loader_source_sha256"]
    ):
        raise LocalReferenceProbeError("controller receipt does not match its registered bytes")
    for field in (
        "numpy_version",
        "torch_version",
        "gymnasium_version",
        "stable_baselines3_version",
        "architecture_id",
        "inference_output_contract",
    ):
        _string(receipt.get(field), f"controller {field}")
    if receipt.get("observation_width") != 348 or receipt.get("action_width") != 17:
        raise LocalReferenceProbeError("controller receipt has the wrong Humanoid ABI")

    if role == "base":
        expected = {
            "artifact_kind": "minari_humanoid_behavior_cloning_controller",
            "architecture_id": "torch_mlp_348_512_512_512_17_relu_action_0.4tanh/v1",
            "filename": "minari_bc_v0.npz",
            "content_sha256": ("ce2aa3a1358609f09509d7f352475a7b517c6d11858ff76419b18a187cb3adf3"),
            "reference_width": None,
            "reference_horizon_steps": None,
            "residual_scale": None,
            "inference_output_contract": (
                "float32_raw_Humanoid-v5_control_with_action_space_endpoint_bytes"
            ),
        }
        binding_key = "base_controller_content_sha256"
        receipt_binding_key = "base_controller_load_receipt_sha256"
    else:
        expected = {
            "artifact_kind": "minari_reference_residual_actor",
            "architecture_id": "sb3_actor_critic_pi256x256_vf256x256_relu_deterministic/v1",
            "filename": "reference_residual_ppo_v0.npz",
            "content_sha256": ("6916bf6778dd3044bca5feae22897b7d582e7389871a728549f791112d90fc22"),
            "reference_width": 45,
            "reference_horizon_steps": 8,
            "residual_scale": 0.08,
            "inference_output_contract": (
                "unscaled_dimensionless_action_in_closed_interval_-1_1;"
                "caller_multiplies_by_residual_scale_before_composition"
            ),
        }
        binding_key = "residual_controller_content_sha256"
        receipt_binding_key = "residual_controller_load_receipt_sha256"
    if any(receipt.get(key) != expected_value for key, expected_value in expected.items()):
        raise LocalReferenceProbeError(f"{role} controller receipt differs from its fixed role")
    if receipt["content_sha256"] != bindings[binding_key]:
        raise LocalReferenceProbeError(f"{role} controller content binding is inconsistent")
    if _sha256(_canonical_json(receipt)) != bindings[receipt_binding_key]:
        raise LocalReferenceProbeError(f"{role} controller receipt binding is inconsistent")


def _validate_projection_receipt(value: object, bindings: dict[str, object]) -> None:
    receipt = _mapping(value, "source projection receipt")
    _exact_fields(receipt, _PROJECTION_RECEIPT_FIELDS, "source projection receipt")
    expected = {
        "schema_version": 2,
        "evidence_tier": "Tier-K",
        "admission_status": "not_admitted",
        "claim_ceiling": "source_projection_and_numerical_candidate_only",
        "dynamics_feasibility_established": False,
        "source_provenance_class": "registered_official_repository_revision",
        "asserted_source_repository": "https://huggingface.co/datasets/farama-minari/mujoco",
        "asserted_source_commit": "8e62dc7f7fcb4a19f8f869c65402d4bb60049117",
        "source_record_sha256": (
            "974700591304a4d3be576d55bd294558ae0cd10cbfce5bef7bd73ca21632f899"
        ),
        "source_record_bytes": 1_458,
        "dataset_id": "mujoco/humanoid/expert-v0",
        "metadata_declared_generation_code_url": (
            "https://github.com/Farama-Foundation/minari-dataset-generation-scripts"
        ),
        "hdf5_sha256": "8253be693f06aeeac3cb62eeb349ad02d4ca0bcad02685b4b8ed390798d9aa1e",
        "hdf5_bytes": 2_946_805_796,
        "metadata_sha256": ("2eb6e0ba388ceabef5eec1dea7e401e62391d856cf42b394c262db7c21366024"),
        "metadata_bytes": 9_305,
        "environment_id": "Humanoid-v5",
        "environment_entry_point": "gymnasium.envs.mujoco.humanoid_v5:HumanoidEnv",
        "episode_id": 0,
        "episode_seed": 123,
        "episode_total_steps": 1000,
        "episode_observation_shape": [1001, 348],
        "episode_action_shape": [1000, 17],
        "observations_sha256": ("bd7ce25125e322c5be1d6ca2de23eaef35317ebb6b24b023ceafdf63eb8d2387"),
        "actions_sha256": "4e7d9cef2a80eb2d05f065f5de4ccb4d04222750da8d1a4d0d9c61ad3fa8c30e",
        "rewards_sha256": "785f49e8552f9038e8507d53d0cc6b84c33e5b7ffe12c5c77e7b606c79a6392b",
        "terminations_sha256": ("b165fd211be467fe0761fd70eacdd28c4419fcfebe24f747afda4236c023e44d"),
        "truncations_sha256": ("1bf9590007e7df7e8f0552ba20389393a0e915cf2feb1fd79d023724e41d21b6"),
        "ignored_observation_fields_sha256": (
            "e015df720194e3b35e55f4750802aa875c8dccfa7b3a08d9958f1bdb063bd333"
        ),
        "root_position_xy_evidence": "absent_from_observation_and_empty_infos",
        "root_position_xy_reconstructed": False,
        "projection_id": "gymnasium/Humanoid-v5/observation-348-to-reference-45/v1",
        "projection_mapping_sha256": (
            "85e2eb4dc0771b63ec77f343943d683771ad59ee2972148bbb77c74d25027be1"
        ),
        "reference_artifact_id": ("minari/mujoco/humanoid/expert-v0/episode-0/projected-45d/v1"),
        "reference_shape": [1001, 45],
        "reference_cadence_hz": 200 / 3,
        "reference_content_sha256": (
            "ef7557643ec87a31e4feb98c30faad98a6139dd646babf2d9e3be57b18f6936c"
        ),
        "reference_schema_sha256": (
            "35d5e7cc86c0054d40362a40ff552ddf9252c3067d02a943cbc2f294c24b9771"
        ),
        "tier_d_certificate_sha256": None,
    }
    if any(receipt.get(key) != expected_value for key, expected_value in expected.items()):
        raise LocalReferenceProbeError("source projection receipt differs from the fixed contract")
    if receipt.get("source_origin_verified") is not True:
        raise LocalReferenceProbeError("source projection origin was not verified")
    commit = _string(receipt.get("asserted_source_commit"), "source commit")
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise LocalReferenceProbeError("source commit must be a lowercase Git commit")
    _string(receipt.get("source_provenance_class"), "source provenance class")
    _string(receipt.get("asserted_source_repository"), "source repository")
    _string(receipt.get("metadata_declared_generation_code_url"), "generation source")
    _string(receipt.get("minari_version"), "Minari version")
    requirements = receipt.get("declared_requirements")
    if (
        not isinstance(requirements, list)
        or not requirements
        or not all(isinstance(item, str) and item for item in requirements)
    ):
        raise LocalReferenceProbeError("declared requirements must be a non-empty string list")
    for field in ("source_record_bytes", "hdf5_bytes", "metadata_bytes"):
        _integer(receipt.get(field), field, minimum=1)
    for field in (
        "source_record_sha256",
        "hdf5_sha256",
        "metadata_sha256",
        "observations_sha256",
        "actions_sha256",
        "rewards_sha256",
        "terminations_sha256",
        "truncations_sha256",
        "ignored_observation_fields_sha256",
        "projection_mapping_sha256",
        "reference_content_sha256",
        "reference_schema_sha256",
    ):
        _digest(receipt.get(field), f"projection {field}")
    indices = (0, 1, 2, 3, 4, *range(22, 28), 6, 5, *range(7, 22), 29, 28, *range(30, 45))
    if receipt.get("projection_indices") != list(indices):
        raise LocalReferenceProbeError("projection indices differ from the fixed Humanoid mapping")
    mapping_hash = _sha256(
        _canonical_json(
            {
                "projection_id": receipt["projection_id"],
                "indices": list(indices),
                "source_width": 348,
                "target_schema_sha256": receipt["reference_schema_sha256"],
            }
        )
    )
    if receipt["projection_mapping_sha256"] != mapping_hash:
        raise LocalReferenceProbeError("projection mapping SHA-256 is inconsistent")
    if (
        receipt["observations_sha256"] != bindings["source_observations_sha256"]
        or receipt["reference_content_sha256"] != bindings["reference_content_sha256"]
        or receipt["reference_schema_sha256"] != bindings["reference_schema_sha256"]
        or _sha256(_canonical_json(receipt)) != bindings["minari_projection_receipt_sha256"]
    ):
        raise LocalReferenceProbeError("source projection receipt binding is inconsistent")


def _validate_bindings(root: Path, value: object, snapshots: list[object]) -> dict[str, object]:
    bindings = _mapping(value, "bindings")
    _exact_fields(bindings, _BINDING_FIELDS, "bindings")
    for field in _BINDING_FIELDS:
        _digest(bindings.get(field), f"binding {field}")
    expected_snapshot_hash = _sha256(_canonical_json(snapshots))
    if bindings["matched_observation_set_sha256"] != expected_snapshot_hash:
        raise LocalReferenceProbeError("matched-observation-set binding is inconsistent")
    for field, relative in _SOURCE_FILES.items():
        try:
            source = _read_project_file(root, relative, limit=_SOURCE_LIMIT)
        except FileNotFoundError as exc:
            raise LocalReferenceProbeError("a bound source file is unavailable") from exc
        if _sha256(source) != bindings[field]:
            raise LocalReferenceProbeError("a bound source file changed after the probe")
    return bindings


def _validate_snapshots(value: object) -> list[object]:
    if not isinstance(value, list) or len(value) != len(_SNAPSHOT_FRAMES):
        raise LocalReferenceProbeError("snapshot ledger does not cover the fixed observations")
    for observed_frame, item in zip(_SNAPSHOT_FRAMES, value, strict=True):
        snapshot = _mapping(item, "snapshot")
        _exact_fields(snapshot, _SNAPSHOT_FIELDS, "snapshot")
        if snapshot.get("snapshot_frame") != observed_frame:
            raise LocalReferenceProbeError("snapshot ledger order differs from the fixed design")
        for field in _SNAPSHOT_FIELDS - {"snapshot_frame"}:
            _digest(snapshot.get(field), f"snapshot {field}")
    return value


def _validate_record(
    value: object,
    *,
    frame: int,
    condition_id: str,
    snapshot: dict[str, object],
    bindings: dict[str, object],
) -> dict[str, object]:
    record = _mapping(value, "probe record")
    _exact_fields(record, _RECORD_FIELDS, "probe record")
    if record.get("snapshot_frame") != frame or record.get("condition_id") != condition_id:
        raise LocalReferenceProbeError("probe record order differs from the fixed design")
    for field in (
        "observation_sha256",
        "ground_truth_window_sha256",
        "transformed_window_sha256",
        "transform_receipt_sha256",
        "base_action_sha256",
        "policy_input_sha256",
        "actor_input_sha256",
        "critic_input_sha256",
        "controller_state_sha256",
        "actor_output_sha256",
        "critic_output_sha256",
        "composed_action_sha256",
    ):
        _digest(record.get(field), f"record {field}")
    if any(
        record[field] != snapshot[field]
        for field in ("observation_sha256", "ground_truth_window_sha256", "base_action_sha256")
    ):
        raise LocalReferenceProbeError("probe record does not match its frozen snapshot")
    if (
        record.get("policy_input_shape") != [708]
        or record.get("policy_input_dtype") != "<f4"
        or record.get("actor_input_sha256") != record.get("policy_input_sha256")
        or record.get("critic_input_sha256") != record.get("policy_input_sha256")
        or record.get("controller_state_sha256") != bindings["residual_controller_state_sha256"]
    ):
        raise LocalReferenceProbeError("actor and critic input receipts are inconsistent")

    base_action = _vector(
        record.get("base_action"),
        "base action",
        length=17,
        lower=_f32(-0.4),
        upper=_f32(0.4),
    )
    actor_output = _vector(
        record.get("actor_output"), "actor output", length=17, lower=-1.0, upper=1.0
    )
    composed_action = _vector(
        record.get("composed_action"),
        "composed action",
        length=17,
        lower=_f32(-0.4),
        upper=_f32(0.4),
    )
    if record.get("actor_output_shape") != [17] or record.get("actor_output_dtype") != "<f4":
        raise LocalReferenceProbeError("actor output ABI is inconsistent")
    if record.get("composed_action_shape") != [17] or record.get("composed_action_dtype") != "<f4":
        raise LocalReferenceProbeError("composed action ABI is inconsistent")
    if _sha256(_float32_bytes(base_action, "base action")) != record["base_action_sha256"]:
        raise LocalReferenceProbeError("base action SHA-256 is inconsistent")
    if _sha256(_float32_bytes(actor_output, "actor output")) != record["actor_output_sha256"]:
        raise LocalReferenceProbeError("actor output SHA-256 is inconsistent")
    if (
        _sha256(_float32_bytes(composed_action, "composed action"))
        != record["composed_action_sha256"]
    ):
        raise LocalReferenceProbeError("composed action SHA-256 is inconsistent")
    expected_composed = tuple(
        max(-0.4, min(0.4, _f32(_f32(base) + _f32(_f32(0.08) * _f32(residual)))))
        for base, residual in zip(base_action, actor_output, strict=True)
    )
    if _float32_bytes(expected_composed, "expected composed action") != _float32_bytes(
        composed_action, "composed action"
    ):
        raise LocalReferenceProbeError("composed action does not follow the fixed float32 rule")

    critic_value = _number(record.get("critic_value"), "critic value")
    if record.get("critic_output_shape") != [1, 1] or record.get("critic_output_dtype") != "<f4":
        raise LocalReferenceProbeError("critic output ABI is inconsistent")
    critic_bytes = _float32_bytes((critic_value,), "critic value")
    if struct.unpack("<f", critic_bytes)[0] != critic_value:
        raise LocalReferenceProbeError("critic value is not exact float32")
    if _sha256(critic_bytes) != record["critic_output_sha256"]:
        raise LocalReferenceProbeError("critic output SHA-256 is inconsistent")

    transform = _validate_transform_receipt(
        record.get("transform_receipt"),
        frame=frame,
        condition_id=condition_id,
    )
    if _sha256(_canonical_json(transform)) != record["transform_receipt_sha256"]:
        raise LocalReferenceProbeError("transform receipt SHA-256 is inconsistent")
    if transform["output_window_sha256"] != record["transformed_window_sha256"]:
        raise LocalReferenceProbeError("transformed-window binding is inconsistent")
    if condition_id == "C_exact" and (
        record["transformed_window_sha256"] != record["ground_truth_window_sha256"]
    ):
        raise LocalReferenceProbeError("exact reference window differs from ground truth")
    return record


def _recompute_summary(records: list[dict[str, object]]) -> dict[str, object]:
    by_key = {(record["snapshot_frame"], record["condition_id"]): record for record in records}
    conditions: dict[str, object] = {}
    gate_passed = True
    for condition_id in _CONDITION_IDS:
        input_changes = actor_changes = composed_changes = critic_changes = 0
        squared_differences: list[float] = []
        max_difference = 0.0
        critic_differences: list[float] = []
        for frame in _SNAPSHOT_FRAMES:
            exact = by_key[(frame, "C_exact")]
            observed = by_key[(frame, condition_id)]
            input_changes += int(observed["policy_input_sha256"] != exact["policy_input_sha256"])
            actor_changes += int(observed["actor_output_sha256"] != exact["actor_output_sha256"])
            composed_changes += int(
                observed["composed_action_sha256"] != exact["composed_action_sha256"]
            )
            critic_changes += int(observed["critic_output_sha256"] != exact["critic_output_sha256"])
            observed_action = tuple(float(item) for item in observed["composed_action"])
            exact_action = tuple(float(item) for item in exact["composed_action"])
            for observed_value, exact_value in zip(observed_action, exact_action, strict=True):
                difference = observed_value - exact_value
                squared_differences.append(difference * difference)
                max_difference = max(max_difference, abs(difference))
            critic_differences.append(
                abs(float(observed["critic_value"]) - float(exact["critic_value"]))
            )
        conditions[condition_id] = {
            "snapshot_count": 8,
            "policy_input_changed_count_vs_exact": input_changes,
            "actor_output_changed_count_vs_exact": actor_changes,
            "composed_action_changed_count_vs_exact": composed_changes,
            "critic_output_changed_count_vs_exact": critic_changes,
            "composed_action_rms_delta_vs_exact": math.sqrt(
                math.fsum(squared_differences) / len(squared_differences)
            ),
            "composed_action_max_abs_delta_vs_exact": max_difference,
            "critic_mean_abs_delta_vs_exact": math.fsum(critic_differences) / 8,
        }
        if condition_id != "C_exact":
            gate_passed &= input_changes == actor_changes == composed_changes == 8
    return {
        "mechanistic_gate_passed": gate_passed,
        "mechanistic_gate_rule": _GATE_RULE,
        "conditions": conditions,
    }


def _validate_summary(value: object, recomputed: dict[str, object]) -> None:
    summary = _mapping(value, "summary")
    _exact_fields(summary, _SUMMARY_FIELDS, "summary")
    if summary.get("mechanistic_gate_passed") is not recomputed["mechanistic_gate_passed"]:
        raise LocalReferenceProbeError("mechanistic gate result is inconsistent")
    if summary.get("mechanistic_gate_rule") != _GATE_RULE:
        raise LocalReferenceProbeError("mechanistic gate rule differs from the fixed protocol")
    observed_conditions = _mapping(summary.get("conditions"), "summary conditions")
    if set(observed_conditions) != set(_CONDITION_IDS):
        raise LocalReferenceProbeError("summary conditions differ from the fixed protocol")
    expected_conditions = _mapping(recomputed["conditions"], "recomputed conditions")
    for condition_id in _CONDITION_IDS:
        observed = _mapping(observed_conditions[condition_id], f"summary {condition_id}")
        _exact_fields(observed, _CONDITION_SUMMARY_FIELDS, f"summary {condition_id}")
        expected = _mapping(expected_conditions[condition_id], f"recomputed {condition_id}")
        for field in _CONDITION_SUMMARY_FIELDS:
            if field.endswith(
                ("rms_delta_vs_exact", "max_abs_delta_vs_exact", "mean_abs_delta_vs_exact")
            ):
                observed_number = _number(observed.get(field), f"summary {field}")
                if not math.isclose(
                    observed_number, float(expected[field]), rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise LocalReferenceProbeError("summary numeric result is inconsistent")
            elif observed.get(field) != expected[field]:
                raise LocalReferenceProbeError("summary count is inconsistent")


def _validated_report(root: Path, report: dict[str, object], source: bytes) -> dict[str, object]:
    _exact_fields(report, _TOP_LEVEL_FIELDS, "report")
    fixed = {
        "schema_version": 1,
        "experiment_id": "experiment_002a",
        "protocol_id": _PROTOCOL_ID,
        "evidence_class": "exploratory",
        "claim_status": _CLAIM_STATUS,
        "formal_oracle_comparison_authorized": False,
        "tracker_behavior_or_stability_established": False,
        "oracle_quality_established": False,
        "dataset_license_status": "unresolved_no_redistribution_or_formal_admission",
        "next_gate": _NEXT_GATE,
    }
    if any(report.get(key) != expected for key, expected in fixed.items()):
        raise LocalReferenceProbeError("report exceeds or differs from the fixed claim boundary")

    design = _mapping(report.get("design"), "design")
    _exact_fields(design, _DESIGN_FIELDS, "design")
    if design != _expected_design():
        raise LocalReferenceProbeError("report design differs from the fixed protocol")
    design_sha256 = _sha256(_canonical_json(design))
    if _digest(report.get("design_sha256"), "design SHA-256") != design_sha256:
        raise LocalReferenceProbeError("design SHA-256 is inconsistent")

    snapshots = _validate_snapshots(report.get("snapshots"))
    bindings = _validate_bindings(root, report.get("bindings"), snapshots)
    _validate_projection_receipt(report.get("source_projection_receipt"), bindings)
    _validate_controller_receipt(
        report.get("base_controller_receipt"), role="base", bindings=bindings
    )
    _validate_controller_receipt(
        report.get("residual_controller_receipt"), role="residual", bindings=bindings
    )

    values = report.get("records")
    if not isinstance(values, list) or len(values) != len(_SNAPSHOT_FRAMES) * len(_CONDITION_IDS):
        raise LocalReferenceProbeError("record ledger does not cover the fixed factorial probe")
    records: list[dict[str, object]] = []
    source_hashes: set[str] = set()
    transformed_hashes: dict[str, set[str]] = {
        condition_id: set() for condition_id in _CONDITION_IDS
    }
    expected_pairs = [
        (frame, condition_id) for frame in _SNAPSHOT_FRAMES for condition_id in _CONDITION_IDS
    ]
    for index, (frame, condition_id) in enumerate(expected_pairs):
        snapshot = _mapping(snapshots[_SNAPSHOT_FRAMES.index(frame)], "snapshot")
        record = _validate_record(
            values[index],
            frame=frame,
            condition_id=condition_id,
            snapshot=snapshot,
            bindings=bindings,
        )
        transform = _mapping(record["transform_receipt"], "transform receipt")
        source_hashes.add(str(transform["source_sha256"]))
        transformed_hashes[condition_id].add(str(transform["transformed_full_sequence_sha256"]))
        records.append(record)
    if len(source_hashes) != 1 or any(len(values) != 1 for values in transformed_hashes.values()):
        raise LocalReferenceProbeError("transform sequence identities changed across snapshots")
    source_identity = next(iter(source_hashes))
    transformed_identities = {next(iter(values)) for values in transformed_hashes.values()}
    if len(transformed_identities) != len(_CONDITION_IDS) or any(
        next(iter(transformed_hashes[condition_id])) == source_identity
        for condition_id in _CONDITION_IDS[1:]
    ):
        raise LocalReferenceProbeError(
            "corrupted transforms do not have distinct sequence identities"
        )
    for frame in _SNAPSHOT_FRAMES:
        block = [record for record in records if record["snapshot_frame"] == frame]
        if len({record["policy_input_sha256"] for record in block}) != len(_CONDITION_IDS):
            raise LocalReferenceProbeError("reference conditions produced duplicate policy inputs")
        if len({record["transformed_window_sha256"] for record in block}) != len(_CONDITION_IDS):
            raise LocalReferenceProbeError(
                "reference conditions produced duplicate window identities"
            )

    recomputed_summary = _recompute_summary(records)
    _validate_summary(report.get("summary"), recomputed_summary)
    analysis_payload = {
        "design_sha256": design_sha256,
        "bindings": bindings,
        "snapshots": snapshots,
        "records": records,
        "summary": report["summary"],
    }
    analysis_sha256 = _sha256(_canonical_json(analysis_payload))
    if (
        _digest(report.get("analysis_payload_sha256"), "analysis payload SHA-256")
        != analysis_sha256
    ):
        raise LocalReferenceProbeError("analysis payload SHA-256 is inconsistent")
    report_sha256 = _sha256(source)
    if report_sha256 != _REVIEWED_REPORT_SHA256:
        raise LocalReferenceProbeError("report identity does not match the reviewed local artifact")

    conditions = _mapping(recomputed_summary["conditions"], "recomputed conditions")
    compact_conditions = []
    for condition_id in _CONDITION_IDS:
        condition = _mapping(conditions[condition_id], f"condition {condition_id}")
        compact_conditions.append(
            {
                "condition_id": condition_id,
                "label": _CONDITION_LABELS[condition_id],
                **condition,
            }
        )
    gate_passed = bool(recomputed_summary["mechanistic_gate_passed"])
    claim = (
        "This checkpoint's composed control changed when only its numeric 8 x 45 reference "
        "input changed at eight predeclared matched observations."
        if gate_passed
        else "This checkpoint has not shown numeric reference-input sensitivity."
    )
    return {
        "state": "available",
        "authority": _AUTHORITY,
        "experiment_id": "experiment_002a",
        "protocol_id": _PROTOCOL_ID,
        "evidence_class": "exploratory",
        "claim_status": _CLAIM_STATUS,
        "claim": claim,
        "mechanistic_gate_passed": gate_passed,
        "mechanistic_gate_rule": _GATE_RULE,
        "snapshot_count": 8,
        "condition_count": 4,
        "independent_unit": "one_existing_local_checkpoint",
        "formal_oracle_comparison_authorized": False,
        "tracker_behavior_or_stability_established": False,
        "oracle_quality_established": False,
        "dataset_license_status": "unresolved_no_redistribution_or_formal_admission",
        "conditions": compact_conditions,
        "limitations": [
            "Offline matched-observation test only.",
            "No simulator state was restored.",
            "No action was submitted to Humanoid-v5.",
            (
                "This does not establish tracking stability, behavioral usefulness, oracle "
                "quality, or authorization for the formal oracle comparison."
            ),
        ],
        "next_gate": _NEXT_GATE,
        "receipts": {
            "report_sha256": report_sha256,
            "analysis_payload_sha256": analysis_sha256,
            "design_sha256": design_sha256,
            "matched_observation_set_sha256": bindings["matched_observation_set_sha256"],
            "source_projection_receipt_sha256": bindings["minari_projection_receipt_sha256"],
            "residual_controller_content_sha256": bindings["residual_controller_content_sha256"],
        },
    }


def local_reference_probe_status(project_root: Path) -> dict[str, object]:
    """Return a compact, fail-closed view of the optional local report."""

    try:
        source = _read_project_file(
            Path(project_root),
            REFERENCE_PROBE_PATH,
            limit=_REPORT_LIMIT,
        )
    except FileNotFoundError:
        return {
            "state": "unavailable",
            "authority": _AUTHORITY,
            "detail": "no local Experiment 002A report is present",
        }
    except LocalReferenceProbeError as exc:
        return {"state": "rejected", "authority": _AUTHORITY, "detail": str(exc)}
    try:
        return _validated_report(Path(project_root), _decode_report(source), source)
    except LocalReferenceProbeError as exc:
        return {"state": "rejected", "authority": _AUTHORITY, "detail": str(exc)}


__all__ = [
    "REFERENCE_PROBE_PATH",
    "LocalReferenceProbeError",
    "local_reference_probe_status",
]
