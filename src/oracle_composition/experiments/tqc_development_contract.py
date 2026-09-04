"""Frozen contract for the excluded-seed one-million-step TQC screen."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fixed_reference import ExperimentContractError, read_bounded_json_artifact
from .tqc_actor_npz import ARCHITECTURE_ID, FORMAT_ID, actor_schema_sha256
from .tqc_calibration_contract import canonical_json

DESIGN_ID = "tqc_humanoid_base_controller_dev_1m/v1"
MAX_DESIGN_BYTES = 256 * 1024
MAX_E0_RECEIPT_BYTES = 1024 * 1024
MODEL_SEED = 95001
WORKER_SEEDS = (95001, 95002, 95003, 95004, 95005)
EVALUATION_SEEDS = tuple(range(96001, 96021))
VIDEO_SEEDS = (96001, 96010, 96020)
TOTAL_ENVIRONMENT_STEPS = 1_000_000
EXPECTED_VECTOR_STEPS = 200_000
EXPECTED_GRADIENT_UPDATES = 199_980
E0_RECEIPT_SHA256 = "2436c2e93cac8b5ed357acdcb19cca0b6222792a1c9c0d65f06577509d68a587"
E0_DESIGN_SHA256 = "5e911653e6be94bed73a60322565ba6d017a95e88f12b04b2eb43578d7b04b7d"
E0_RUNTIME_SHA256 = "710d524b2936ee29bbeb5be9336ed5bb0e12964919f7035d213d9f19348e0a77"

_TOP_LEVEL_KEYS = {
    "schema_version",
    "design_id",
    "design_status",
    "evidence_purpose",
    "claim_ceiling",
    "formal_experiment_eligible",
    "required_e0",
    "seed_schedule",
    "environment",
    "tqc",
    "attempt",
    "resource_gates",
    "persistence",
    "evaluation",
    "decision_rule",
    "runtime_requirements",
}


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ExperimentContractError(f"{field} must be an object")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ExperimentContractError(f"{field} must be a list")
    return value


def _require_keys(value: Mapping[str, Any], expected: set[str], *, field: str) -> None:
    if set(value) != expected:
        raise ExperimentContractError(
            f"{field} keys differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _require_exact(value: object, expected: object, *, field: str) -> None:
    def matches_exact(observed: object, required: object) -> bool:
        if type(observed) is not type(required):
            return False
        if isinstance(required, dict):
            if observed.keys() != required.keys():
                return False
            return all(matches_exact(observed[name], item) for name, item in required.items())
        if isinstance(required, (list, tuple)):
            return len(observed) == len(required) and all(
                matches_exact(observed_item, required_item)
                for observed_item, required_item in zip(observed, required, strict=True)
            )
        return observed == required

    if not matches_exact(value, expected):
        raise ExperimentContractError(f"{field} must equal {expected!r}")


def _require_exact_mapping(
    value: object,
    expected: Mapping[str, object],
    *,
    field: str,
) -> Mapping[str, Any]:
    observed = _mapping(value, field=field)
    _require_keys(observed, set(expected), field=field)
    for name, required in expected.items():
        _require_exact(observed[name], required, field=f"{field}.{name}")
    return observed


def _require_unique_integer_seeds(value: object, expected: tuple[int, ...], *, field: str) -> None:
    observed = _sequence(value, field=field)
    if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in observed):
        raise ExperimentContractError(f"{field} must contain non-negative integer seeds")
    _require_exact(tuple(observed), expected, field=field)
    if len(set(observed)) != len(observed):
        raise ExperimentContractError(f"{field} contains duplicate seeds")


def _validate_design(raw: Mapping[str, Any]) -> None:
    _require_keys(raw, _TOP_LEVEL_KEYS, field="design")
    fixed_top = {
        "schema_version": 1,
        "design_id": DESIGN_ID,
        "design_status": "predeclared_not_executed",
        "evidence_purpose": "permanently_excluded_base_controller_development_screen",
        "claim_ceiling": (
            "one_checkpoint_one_training_seed_twenty_fixed_reset_development_behavior"
        ),
        "formal_experiment_eligible": False,
    }
    for name, expected in fixed_top.items():
        _require_exact(raw[name], expected, field=name)

    _require_exact_mapping(
        raw["required_e0"],
        {
            "receipt_content_sha256": E0_RECEIPT_SHA256,
            "design_artifact_sha256": E0_DESIGN_SHA256,
            "runtime_sha256": E0_RUNTIME_SHA256,
            "completion_status": "complete",
            "calibration_gate_passed": True,
            "measured_workload_gates_passed": True,
            "observed_environment_steps": 100_000,
            "observed_environment_steps_per_second": 619.6620593242255,
            "observed_training_wall_seconds": 161.37828433300456,
            "observed_peak_rss_bytes": 4_253_122_560,
            "observed_replay_buffer_allocation_bytes": 5_648_000_000,
        },
        field="required_e0",
    )

    seeds = _mapping(raw["seed_schedule"], field="seed_schedule")
    _require_keys(
        seeds,
        {
            "model_seed",
            "worker_seeds",
            "evaluation_seeds",
            "video_seeds",
            "role",
            "independent_training_units",
            "evaluation_unit",
        },
        field="seed_schedule",
    )
    _require_exact(seeds["model_seed"], MODEL_SEED, field="seed_schedule.model_seed")
    _require_unique_integer_seeds(
        seeds["worker_seeds"], WORKER_SEEDS, field="seed_schedule.worker_seeds"
    )
    _require_unique_integer_seeds(
        seeds["evaluation_seeds"], EVALUATION_SEEDS, field="seed_schedule.evaluation_seeds"
    )
    _require_unique_integer_seeds(
        seeds["video_seeds"], VIDEO_SEEDS, field="seed_schedule.video_seeds"
    )
    if set(WORKER_SEEDS) & set(EVALUATION_SEEDS):
        raise ExperimentContractError("training and evaluation seeds overlap")
    if not set(VIDEO_SEEDS) <= set(EVALUATION_SEEDS):
        raise ExperimentContractError("video seeds must be predetermined evaluation seeds")
    _require_exact(
        seeds["role"],
        "permanently_excluded_from_formal_oracle_and_reward_studies",
        field="seed_schedule.role",
    )
    _require_exact(
        seeds["independent_training_units"], 1, field="seed_schedule.independent_training_units"
    )
    _require_exact(
        seeds["evaluation_unit"],
        "repeated_seeded_reset_within_one_fixed_checkpoint",
        field="seed_schedule.evaluation_unit",
    )

    _require_exact_mapping(
        raw["environment"],
        {
            "environment_id": "Humanoid-v5",
            "n_envs": 5,
            "vectorization": "stable_baselines3.DummyVecEnv/v2.9",
            "terminate_when_unhealthy": False,
            "reset_noise_scale": 0.01,
            "exclude_current_positions_from_observation": True,
            "frame_skip": 5,
            "max_episode_steps": 1000,
            "control_period_seconds": 0.015,
            "observation_shape": [348],
            "action_shape": [17],
            "physical_action_low": -0.4,
            "physical_action_high": 0.4,
            "observation_normalizer": "none/v1",
            "reward_normalizer": "none/v1",
            "training_reward": "Gymnasium_Humanoid-v5_default_reward/v1",
        },
        field="environment",
    )
    _require_exact_mapping(
        raw["tqc"],
        {
            "algorithm_id": "sb3_contrib.TQC/v2.9",
            "required_stable_baselines3_version": "2.9.0",
            "required_sb3_contrib_version": "2.9.0",
            "policy": "MlpPolicy",
            "policy_class": "sb3_contrib.tqc.policies.TQCPolicy",
            "device": "cpu",
            "seed": MODEL_SEED,
            "init_setup_model": True,
            "total_environment_steps": TOTAL_ENVIRONMENT_STEPS,
            "expected_vector_steps": EXPECTED_VECTOR_STEPS,
            "expected_gradient_updates": EXPECTED_GRADIENT_UPDATES,
            "learning_rate": 0.0003,
            "buffer_size": 1_000_000,
            "learning_starts": 100,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq_steps": 1,
            "gradient_steps": 1,
            "n_steps": 1,
            "ent_coef": "auto",
            "initial_entropy_coefficient": 1.0,
            "target_entropy": "auto",
            "expected_resolved_target_entropy": -17.0,
            "target_update_interval": 1,
            "top_quantiles_to_drop_per_net": 2,
            "action_noise": None,
            "replay_buffer_class": "stable_baselines3.common.buffers.ReplayBuffer",
            "replay_buffer_kwargs": {"handle_timeout_termination": True},
            "use_sde": False,
            "sde_sample_freq": -1,
            "use_sde_at_warmup": False,
            "optimize_memory_usage": False,
            "stats_window_size": 100,
            "tensorboard_log": None,
            "verbose": 0,
            "logger_class": "stable_baselines3.common.logger.Logger",
            "logger_folder": None,
            "logger_output_format_count": 0,
            "learning_rate_schedule": "constant_0.0003/v1",
            "policy_kwargs": {
                "net_arch": [256, 256],
                "activation_fn": "torch.nn.ReLU",
                "log_std_init": -3.0,
                "use_expln": False,
                "clip_mean": 2.0,
                "features_extractor_class": (
                    "stable_baselines3.common.torch_layers.FlattenExtractor"
                ),
                "features_extractor_kwargs": None,
                "normalize_images": True,
                "optimizer_class": "torch.optim.Adam",
                "optimizer_kwargs": {"eps": 0.00001},
                "n_quantiles": 25,
                "n_critics": 2,
                "share_features_extractor": False,
            },
        },
        field="tqc",
    )
    _require_exact_mapping(
        raw["attempt"],
        {
            "attempt_id": "dev1m-seed-95001-attempt-01",
            "maximum_attempts": 1,
            "reset_num_timesteps": True,
            "early_stopping": False,
            "metric_based_checkpointing": False,
            "automatic_retry": False,
            "interrupted_run_disposition": (
                "failed_no_checkpoint_substitution_new_attempt_requires_review"
            ),
        },
        field="attempt",
    )

    resource = _require_exact_mapping(
        raw["resource_gates"],
        {
            "sampled_peak_rss_failure_threshold_bytes": 12_884_901_888,
            "minimum_free_disk_before_training_bytes": 53_687_091_200,
            "minimum_environment_steps_per_second": 116.0,
            "throughput_warmup_environment_steps": 10_000,
            "throughput_window_environment_steps": 10_000,
            "sampled_wall_time_failure_threshold_seconds": 10_800.0,
            "e0_rate_projected_training_seconds": 1613.7828433300454,
            "replay_buffer_allocation_bytes": 5_648_000_000,
            "maximum_trusted_model_archive_bytes": 104_857_600,
            "maximum_trusted_replay_archive_bytes": 8_589_934_592,
            "maximum_actor_export_bytes": 2_097_152,
        },
        field="resource_gates",
    )
    projected = TOTAL_ENVIRONMENT_STEPS / 619.6620593242255
    if not math.isclose(
        float(resource["e0_rate_projected_training_seconds"]),
        projected,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ExperimentContractError("resource projection is inconsistent with E0")

    persistence = _mapping(raw["persistence"], field="persistence")
    _require_keys(
        persistence, {"checkpoint_rule", "trusted_local_state", "actor_export"}, field="persistence"
    )
    checkpoint_rule = "exact_final_1000000_environment_step_checkpoint_only"
    _require_exact(
        persistence["checkpoint_rule"], checkpoint_rule, field="persistence.checkpoint_rule"
    )
    _require_exact_mapping(
        persistence["trusted_local_state"],
        {
            "model_filename": "tqc_model_step_1000000.zip",
            "model_format": "SB3_zip_with_torch_and_cloudpickle_trusted_local_only/v2.9",
            "replay_filename": "tqc_replay_step_1000000.pkl",
            "replay_format": "SB3_pickle_trusted_local_only/v2.9",
            "normalizer_filename": None,
            "load_policy": "same_run_locally_created_paths_only_never_uploaded",
            "continuation_claim": ("operational_continuation_only_not_bitwise_equivalent_resume"),
            "public_distribution_allowed": False,
        },
        field="persistence.trusted_local_state",
    )
    _require_exact_mapping(
        persistence["actor_export"],
        {
            "filename": "tqc_actor_step_1000000.npz",
            "manifest_filename": "tqc_actor_step_1000000.manifest.json",
            "format_id": FORMAT_ID,
            "architecture_id": ARCHITECTURE_ID,
            "schema_sha256": actor_schema_sha256(),
            "source_rule": "exact_final_checkpoint_actor_only_no_metric_selection",
            "equivalence_rule": (
                "pinned_observation_batch_mean_log_std_deterministic_and_seeded_sample"
            ),
            "equivalence_comparison": "exact_shape_dtype_and_c_order_bytes/v1",
            "equivalence_receipt_required_hashes": [
                "trusted_mean_sha256",
                "loaded_mean_sha256",
                "trusted_log_std_sha256",
                "loaded_log_std_sha256",
                "trusted_deterministic_action_sha256",
                "loaded_deterministic_action_sha256",
                "trusted_seeded_sample_sha256",
                "loaded_seeded_sample_sha256",
            ],
            "equivalence_observation_set": {
                "construction": "integer_affine_grid_float32/v1",
                "shape": [4, 348],
                "sha256": "0c6a81b06a88cab7eca0255e75f021008b60025c4ddc4d3719426e3647159ec6",
                "sampling_seed": 97_001,
            },
            "contains_executable_code": False,
            "contains_critic": False,
            "contains_optimizer": False,
            "contains_replay": False,
            "contains_rng_state": False,
            "exact_training_resume": False,
            "publication_allowed": False,
            "license_status": "blocked_no_project_or_trained_weight_license_selected",
        },
        field="persistence.actor_export",
    )

    required_metrics = (
        "observed_steps",
        "finite_state_action_fraction",
        "healthy_step_fraction",
        "first_unhealthy_step",
        "upright_step_fraction",
        "first_not_upright_step",
        "root_height_min_m",
        "root_height_max_m",
        "net_forward_displacement_m",
        "time_average_forward_velocity_m_s",
        "root_lateral_displacement_max_abs_m",
        "normalized_action_rms",
        "normalized_action_saturation_fraction",
        "non_foot_floor_contact_step_fraction",
    )
    evaluation = _mapping(raw["evaluation"], field="evaluation")
    _require_keys(
        evaluation,
        {
            "checkpoint_rule",
            "deterministic_actions",
            "episodes_per_seed",
            "max_episode_steps",
            "reset_options",
            "reward_or_info_reward_fields_used_for_metrics",
            "metric_source",
            "healthy_root_height_open_interval_m",
            "minimum_torso_up_z",
            "required_metrics",
            "trace_rule",
            "video_rule",
        },
        field="evaluation",
    )
    expected_evaluation = {
        "checkpoint_rule": checkpoint_rule,
        "deterministic_actions": True,
        "episodes_per_seed": 1,
        "max_episode_steps": 1000,
        "reset_options": None,
        "reward_or_info_reward_fields_used_for_metrics": False,
        "metric_source": "direct_MuJoCo_state_control_and_contact_readers/v1",
        "healthy_root_height_open_interval_m": [1.0, 2.0],
        "minimum_torso_up_z": 0.5,
        "required_metrics": list(required_metrics),
        "trace_rule": "one_complete_canonical_trace_per_evaluation_seed",
        "video_rule": ("record_full_horizon_only_for_the_three_predetermined_video_seeds"),
    }
    for name, expected in expected_evaluation.items():
        _require_exact(evaluation[name], expected, field=f"evaluation.{name}")

    decision = _mapping(raw["decision_rule"], field="decision_rule")
    _require_keys(
        decision,
        {
            "integrity_hard_gates",
            "development_behavior_gate",
            "selection",
            "if_gate_passes",
            "if_gate_fails",
            "automatic_20m_authorization",
            "automatic_tracker_admission",
        },
        field="decision_rule",
    )
    _require_exact(
        decision["integrity_hard_gates"],
        [
            "exact_training_budget_and_update_count",
            "all_training_and_evaluation_values_finite",
            "exact_final_checkpoint_only",
            "actor_export_strict_load_and_trusted_actor_output_equivalence",
            "all_twenty_evaluation_seeds_observed_once_in_declared_order",
        ],
        field="decision_rule.integrity_hard_gates",
    )
    _require_exact_mapping(
        decision["development_behavior_gate"],
        {
            "full_horizon_healthy_episode_count": 20,
            "full_horizon_upright_episode_count": 20,
            "minimum_median_time_average_forward_velocity_m_s": 0.5,
            "minimum_episode_count_with_net_forward_displacement_at_least_5_m": 18,
        },
        field="decision_rule.development_behavior_gate",
    )
    for name, expected in {
        "selection": "single_final_checkpoint_no_best_seed_episode_or_checkpoint_selection",
        "if_gate_passes": "eligible_for_actual_E1_identity_test_not_tracker_admission",
        "if_gate_fails": (
            "retain_failed_development_evidence_and_diagnose_before_any_20M_proposal"
        ),
        "automatic_20m_authorization": False,
        "automatic_tracker_admission": False,
    }.items():
        _require_exact(decision[name], expected, field=f"decision_rule.{name}")

    runtime = _mapping(raw["runtime_requirements"], field="runtime_requirements")
    runtime_fixed = {
        "python_version": "3.13.15",
        "platform_system": "Darwin",
        "platform_machine": "arm64",
        "platform_release": "25.6.0",
        "cpu_model": "Apple M5 Max",
        "hardware_model": "Mac17,6",
        "logical_cpu_count": 18,
        "total_memory_bytes": 38_654_705_664,
        "torch_intraop_thread_count": 6,
        "torch_interop_thread_count": 18,
        "numpy_version": "2.5.2",
        "torch_version": "2.14.0",
        "gymnasium_version": "1.3.0",
        "mujoco_version": "3.12.0",
        "stable_baselines3_version": "2.9.0",
        "sb3_contrib_version": "2.9.0",
        "dependency_lock_sha256": (
            "81b92d15dd2da62f27cd770322db78008d5387b530dc71e053f0d56b327f0b40"
        ),
        "mujoco_model_sha256": ("85816f372c826d2094b4a598918233bd9c5843b2439119eece2733bdc2e0d073"),
        "observation_space_sha256": (
            "07953989de29452db64aa5188067084bf881761072f9d9b1b9e45b442d38506c"
        ),
        "action_space_sha256": ("5a2389149db1f0253571a4f07f4f83285844f18c529355502101d5b50030cfb3"),
        "environment_source_sha256": (
            "41fd30a944faf39da15414527a6f70d1e69c4b0bfa4531560bfbc89da2b364e9"
        ),
        "gym_humanoid_source_sha256": (
            "079e98c842d76ff53a28ac5e35f8a12d1aa2d916ca0adb1cb36abdb0d38a13cf"
        ),
        "tqc_algorithm_source_sha256": (
            "28f42ad961d2f903166967ca25cef7f241a3e6805b1252b5f341a612ef26805c"
        ),
        "tqc_policy_source_sha256": (
            "7cddca54a7cfad31ad7473244ae7db4412037e4d677e4f3ae156443db52c0a55"
        ),
        "sb3_base_algorithm_source_sha256": (
            "8e142f3efc4397a94dbe8bbb0ee1998cfe9e809b2fb34a9297f737ed4d81ab63"
        ),
        "sb3_save_util_source_sha256": (
            "5dc9afe3aa245830fbdf8784e7ce77df8132944ab48e570eefd6b073ae0e83a1"
        ),
        "sb3_off_policy_source_sha256": (
            "ec895fa4b7ce96550b0acdfea3a73561db568f863b24c0b77d106f3f67e540a7"
        ),
        "sb3_replay_buffer_source_sha256": (
            "21db27f000048615f5848facf18af9afc22e6e2a1c26148911c2cba542d1988d"
        ),
        "sb3_dummy_vec_env_source_sha256": (
            "e9086bccdee89800a03e32fad09b2baa4195bb72ada97a12f381afe211c7c32b"
        ),
        "sb3_distributions_source_sha256": (
            "416b40a1bc3bf6435f45e79553fc553d796b58f95d28e49d3aa4437dcfe6e251"
        ),
        "sb3_policies_source_sha256": (
            "40bfc915633e216d9d620724e3d80029a6e2a2a9c7607ddc75782b2c8cc281e1"
        ),
        "sb3_torch_layers_source_sha256": (
            "5b3293b1139e5175fa77d65434d153a406ea470d75212b6a6d1bdc6e5fa08a40"
        ),
        "sb3_utils_source_sha256": (
            "c95c6d88f810e48ac32b488511cd63d87b83ef7b459b8a0a9e5e955a9856d809"
        ),
        "sb3_noise_source_sha256": (
            "335d9e322c8b5bf1b43703ac284951a01bb7980822cb30c8c2b94b0f48dc4e1f"
        ),
        "sb3_logger_source_sha256": (
            "d2533d0fe929aa533ecb373c040a5e7f1c88852667e367f0a8d923e1efe5fa5e"
        ),
        "sb3_contrib_common_utils_source_sha256": (
            "a260575056fddea745a1fcad0add714292ad706cb1ba0e47c839f06bfd406d29"
        ),
        "sb3_base_vec_env_source_sha256": (
            "1e7da85077c947fc1ec81941de793cabcc039a8ff8e0172b6b7078bd492ddf80"
        ),
        "execution_manifest_required_before_training": True,
        "execution_manifest_must_bind": [
            "project_source_tree_sha256",
            "training_adapter_source_sha256",
            "protected_locomotion_evaluator_source_sha256",
            "actor_export_source_sha256",
            "design_file_sha256",
            "e0_receipt_content_sha256",
        ],
    }
    _require_exact_mapping(runtime, runtime_fixed, field="runtime_requirements")


@dataclass(frozen=True, slots=True)
class LoadedTQCDevelopmentDesign:
    """Validated immutable design bytes and their two distinct identities."""

    encoded_bytes: bytes
    file_sha256: str
    semantic_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.encoded_bytes)

    @property
    def model_seed(self) -> int:
        return MODEL_SEED

    @property
    def worker_seeds(self) -> tuple[int, ...]:
        return WORKER_SEEDS

    @property
    def evaluation_seeds(self) -> tuple[int, ...]:
        return EVALUATION_SEEDS


def load_tqc_development_design(path: Path) -> LoadedTQCDevelopmentDesign:
    """Load one exact design without filling defaults or repairing drift."""

    artifact = read_bounded_json_artifact(
        Path(path), maximum_bytes=MAX_DESIGN_BYTES, artifact="TQC development design"
    )
    _validate_design(artifact.value)
    return LoadedTQCDevelopmentDesign(
        encoded_bytes=artifact.encoded_bytes,
        file_sha256=artifact.sha256,
        semantic_sha256=hashlib.sha256(canonical_json(artifact.value)).hexdigest(),
    )


def validate_required_e0_receipt(
    path: Path,
    design: LoadedTQCDevelopmentDesign,
) -> dict[str, object]:
    """Require the exact successful resource receipt before a training adapter runs."""

    artifact = read_bounded_json_artifact(
        Path(path), maximum_bytes=MAX_E0_RECEIPT_BYTES, artifact="required E0 receipt"
    )
    receipt = artifact.value
    required = design.to_dict()["required_e0"]
    if (
        artifact.sha256 != E0_RECEIPT_SHA256
        or artifact.sha256 != required["receipt_content_sha256"]
    ):
        raise ExperimentContractError(
            "E0 receipt bytes differ from the compiled and development-design identities"
        )
    checks = {
        "completion_status": required["completion_status"],
        "calibration_gate_passed": required["calibration_gate_passed"],
        "measured_workload_gates_passed": required["measured_workload_gates_passed"],
        "design_artifact_sha256": required["design_artifact_sha256"],
        "observed_environment_steps": required["observed_environment_steps"],
        "environment_steps_per_second": required["observed_environment_steps_per_second"],
        "training_wall_seconds": required["observed_training_wall_seconds"],
        "peak_rss_bytes": required["observed_peak_rss_bytes"],
        "replay_buffer_allocation_bytes": required["observed_replay_buffer_allocation_bytes"],
        "checkpoint_emitted": False,
        "replay_buffer_emitted": False,
        "normalizer_emitted": False,
        "eligible_for_controller_training": False,
        "eligible_for_behavioral_evaluation": False,
    }
    for field, expected in checks.items():
        _require_exact(receipt.get(field), expected, field=f"E0 receipt.{field}")
    runtime = _mapping(receipt.get("runtime"), field="E0 receipt.runtime")
    _require_exact(
        runtime.get("runtime_sha256"), required["runtime_sha256"], field="E0 runtime SHA-256"
    )
    return {
        "receipt_content_sha256": artifact.sha256,
        "receipt_byte_count": len(artifact.encoded_bytes),
        "runtime_sha256": runtime["runtime_sha256"],
        "resource_gate_passed": True,
        "controller_bytes_emitted": False,
    }


__all__ = [
    "DESIGN_ID",
    "E0_RECEIPT_SHA256",
    "EVALUATION_SEEDS",
    "EXPECTED_GRADIENT_UPDATES",
    "EXPECTED_VECTOR_STEPS",
    "MODEL_SEED",
    "TOTAL_ENVIRONMENT_STEPS",
    "VIDEO_SEEDS",
    "WORKER_SEEDS",
    "LoadedTQCDevelopmentDesign",
    "load_tqc_development_design",
    "validate_required_e0_receipt",
]
