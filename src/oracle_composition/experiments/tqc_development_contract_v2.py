"""Frozen contract for the unexecuted v2 TQC development screen."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import InitVar, dataclass
from pathlib import Path
from typing import Any

from .fixed_reference import ExperimentContractError, read_bounded_json_artifact
from .tqc_calibration_contract import canonical_json

DESIGN_ID = "tqc_humanoid_base_controller_dev_1m/v2"
AUDIT_ID = "tqc_e0_reuse_dev_1m_v2/v1"
DESIGN_FILE_SHA256 = "e20b0f0b69b008e875e0dbcfca80362aee64025b8c21e881f2424cc0d2b3cb68"
DESIGN_SEMANTIC_SHA256 = "7b54dde8fc6820dab44490abc78e40f7d995ed7a81f572ac91fcad5fe5e03ea2"
AUDIT_FILE_SHA256 = "9246735cdd2f72ae675446d2f706c698303fb2913cb02c318d61f033ecd94106"
AUDIT_SEMANTIC_SHA256 = "743291a47428a1f4d80c52e69f48fb847fd1877e677dc457098d8fc9879cef86"
TRAINING_PROJECTION_SHA256 = "1120f73fc458989a14a8f72f487fe4a9fa28922e1b55a9954744e271fb11425e"
E0_DESIGN_FILE_SHA256 = "5e911653e6be94bed73a60322565ba6d017a95e88f12b04b2eb43578d7b04b7d"
E0_DESIGN_SEMANTIC_SHA256 = "3a82a5191d52458523d45e267c0a703fba5d8b798751b6b12227861190cce72e"
E0_RECEIPT_FILE_SHA256 = "2436c2e93cac8b5ed357acdcb19cca0b6222792a1c9c0d65f06577509d68a587"
E0_RECEIPT_SEMANTIC_SHA256 = "ba33e80661d62a67c212666dabf72e7964654c7d840ed13128cda3a431ae35cc"
E0_RUNTIME_SHA256 = "710d524b2936ee29bbeb5be9336ed5bb0e12964919f7035d213d9f19348e0a77"
V1_DESIGN_FILE_SHA256 = "41c58694093070d7994aca5d1e8507832b40302fda088e4a1d42f26db7f5ca49"
V1_DESIGN_SEMANTIC_SHA256 = "8a2f157dccde1d46f85ff8c30c4e7390379d4ea7854b9537c8d9a29a2e40344f"
MODEL_SEED = 95_001
WORKER_SEEDS = tuple(range(95_001, 95_006))
EVALUATION_SEEDS = tuple(range(96_001, 96_021))
VIDEO_SEEDS = (96_001, 96_010, 96_020)
TOTAL_ENVIRONMENT_STEPS = 1_000_000
EXPECTED_VECTOR_STEPS = 200_000
EXPECTED_GRADIENT_UPDATES = 199_980
EXPECTED_PARAMETER_CHECKS = 202
EXPECTED_DISK_CHECKS = 2_004
CAPTURE_INDICES = tuple(range(0, 1_001, 2))
MAX_DESIGN_BYTES = 512 * 1024
MAX_AUDIT_BYTES = 512 * 1024
MAX_E0_RECEIPT_BYTES = 1024 * 1024

_LOADED_ISSUER = object()
_E0_PREFLIGHT_ISSUER = object()
_DESIGN_KEYS = {
    "schema_version",
    "design_id",
    "design_status",
    "supersession",
    "evidence_purpose",
    "claim_ceiling",
    "formal_experiment_eligible",
    "required_e0",
    "training_projection",
    "evaluation",
    "visual_evidence",
    "decision_rule",
    "execution_manifest",
}
_AUDIT_KEYS = {
    "schema_version",
    "audit_id",
    "audit_status",
    "evidence_purpose",
    "claim_ceiling",
    "authorizes_training",
    "calibration_evidence",
    "default_resolution_evidence",
    "shared_projection",
    "v2_training_projection_sha256",
    "v2_training_projection",
    "declared_deltas",
    "declared_protocol_additions",
    "unmeasured_v2_requirements",
    "reuse_decision",
}


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _semantic_sha256(value: object) -> str:
    return _sha256(canonical_json(value))


def _require_exact(value: object, expected: object, *, field: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise ExperimentContractError(f"{field} must equal {expected!r}")


def _require_keys(value: object, expected: set[str], *, field: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ExperimentContractError(f"{field} must be an object")
    if set(value) != expected:
        raise ExperimentContractError(
            f"{field} keys differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )
    return value


def _at(value: object, path: str, *, field: str) -> object:
    current = value
    for part in path.split("."):
        if type(current) is not dict or part not in current:
            raise ExperimentContractError(f"{field} path {path!r} is missing")
        current = current[part]
    return current


def _assert_sha(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"{field} must be a lowercase SHA-256")
    return value


def _contains_key(value: object, key: str) -> bool:
    if type(value) is dict:
        return key in value or any(_contains_key(child, key) for child in value.values())
    if type(value) is list:
        return any(_contains_key(child, key) for child in value)
    return False


def _artifact(
    path: Path,
    *,
    maximum_bytes: int,
    artifact: str,
    file_sha256: str,
    semantic_sha256: str,
) -> tuple[bytes, dict[str, Any]]:
    loaded = read_bounded_json_artifact(Path(path), maximum_bytes=maximum_bytes, artifact=artifact)
    value = loaded.value
    if type(value) is not dict:
        raise ExperimentContractError(f"{artifact} must be an object")
    if loaded.sha256 != file_sha256:
        raise ExperimentContractError(f"{artifact} file SHA-256 differs")
    if _semantic_sha256(value) != semantic_sha256:
        raise ExperimentContractError(f"{artifact} semantic SHA-256 differs")
    return loaded.encoded_bytes, value


def _validate_parameter_schema(training: dict[str, Any]) -> None:
    policy = training["tqc_policy"]
    schema = policy["ordered_parameter_schema"]
    if type(schema) is not list or len(schema) != 33:
        raise ExperimentContractError("ordered parameter schema must contain 33 tensors")
    role_counts = {role: 0 for role in ("actor", "critic", "critic_target", "entropy_coefficient")}
    total_scalars = 0
    optimizer_scalars = 0
    optimizer_tensors = 0
    names: list[str] = []
    for item in schema:
        row = _require_keys(
            item,
            {"role", "name", "shape", "dtype", "optimizer_owned"},
            field="ordered parameter row",
        )
        role = row["role"]
        if type(role) is not str or role not in role_counts:
            raise ExperimentContractError("parameter role differs")
        if type(row["name"]) is not str or row["name"] in names:
            raise ExperimentContractError("parameter names must be unique exact strings")
        names.append(row["name"])
        shape = row["shape"]
        if (
            type(shape) is not list
            or not shape
            or any(type(value) is not int or value < 1 for value in shape)
        ):
            raise ExperimentContractError("parameter shape differs")
        _require_exact(row["dtype"], "<f4", field="parameter dtype")
        if type(row["optimizer_owned"]) is not bool:
            raise ExperimentContractError("parameter optimizer ownership must be boolean")
        scalars = math.prod(shape)
        total_scalars += scalars
        role_counts[role] += 1
        if row["optimizer_owned"]:
            optimizer_scalars += scalars
            optimizer_tensors += 1
    _require_exact(
        role_counts,
        {"actor": 8, "critic": 12, "critic_target": 12, "entropy_coefficient": 1},
        field="parameter role counts",
    )
    _require_exact(total_scalars, 827_527, field="finite-state scalar count")
    _require_exact(optimizer_scalars, 495_701, field="optimizer-owned scalar count")
    _require_exact(optimizer_tensors, 21, field="optimizer-owned tensor count")
    _require_exact(policy["expected_policy_state_tensor_count"], 32, field="policy tensor count")
    _require_exact(policy["expected_finite_state_tensor_count"], 33, field="finite tensor count")
    _require_exact(
        policy["expected_policy_parameter_scalar_count"], 827_526, field="policy scalar count"
    )
    _require_exact(
        policy["expected_finite_state_scalar_count"], total_scalars, field="finite scalar count"
    )
    _require_exact(
        _semantic_sha256(schema),
        policy["ordered_parameter_schema_sha256"],
        field="parameter structure SHA-256",
    )
    _require_exact(
        policy["expected_observed_model_parameter_structure_sha256"],
        "0d9f1faecbf9dff3eadb350e12d2339aa76b853a07a08729347dc754beb41335",
        field="observed model parameter structure SHA-256",
    )


def _validate_training_projection(training: dict[str, Any]) -> None:
    seeds = training["seed_schedule"]
    _require_exact(tuple(seeds["worker_seeds"]), WORKER_SEEDS, field="worker seeds")
    _require_exact(tuple(seeds["evaluation_seeds"]), EVALUATION_SEEDS, field="evaluation seeds")
    _require_exact(tuple(seeds["video_seeds"]), VIDEO_SEEDS, field="video seeds")
    _require_exact(seeds["model_seed"], MODEL_SEED, field="model seed")
    _require_exact(seeds["instrumentation_equivalence_seed"], 98_001, field="instrumentation seed")
    if len(set(WORKER_SEEDS + EVALUATION_SEEDS)) != len(WORKER_SEEDS + EVALUATION_SEEDS):
        raise ExperimentContractError("training and evaluation seeds overlap")

    environment = training["environment"]
    simulator = training["simulator"]
    tqc = training["tqc"]
    counters = training["counters"]
    monitoring = training["monitoring"]
    replay = training["replay_buffer"]
    resource = training["resource_gates"]
    _require_exact(environment["environment_id"], "Humanoid-v5", field="environment id")
    _require_exact(
        environment["environment_class"],
        "gymnasium.envs.mujoco.humanoid_v5.HumanoidEnv",
        field="training environment class",
    )
    _require_exact(
        environment["wrapper_order_outer_to_inner"],
        [
            "gymnasium.wrappers.common.TimeLimit",
            "gymnasium.wrappers.common.OrderEnforcing",
            "gymnasium.wrappers.common.PassiveEnvChecker",
            "gymnasium.envs.mujoco.humanoid_v5.HumanoidEnv",
        ],
        field="training environment wrapper stack",
    )
    _require_exact(
        simulator["contact_capture_id"],
        "none_in_plain_training_environment/v1",
        field="training contact capture",
    )
    _require_exact(simulator["ccd_iterations"], 35, field="MuJoCo CCD iterations")
    _require_exact(simulator["ccd_tolerance"], 0.000001, field="MuJoCo CCD tolerance")
    _require_exact(simulator["disable_actuator"], 0, field="MuJoCo disabled actuators")
    _require_exact(
        simulator["override_friction"],
        [1.0, 1.0, 0.005, 0.0001, 0.0001],
        field="MuJoCo override friction",
    )
    _require_exact(simulator["sleep_tolerance"], 0.001, field="MuJoCo sleep tolerance")
    _require_exact(environment["render_mode"], None, field="training render mode")
    _require_exact(environment["mujoco_env_max_geom"], 1_000, field="Gym max_geom")
    if not math.isclose(
        simulator["physics_timestep_seconds"] * environment["frame_skip"],
        environment["control_period_seconds"],
        rel_tol=0.0,
        abs_tol=0.0,
    ):
        raise ExperimentContractError("physics and control periods are inconsistent")
    _require_exact(tqc["total_environment_steps"], TOTAL_ENVIRONMENT_STEPS, field="budget")
    expected_vectors = TOTAL_ENVIRONMENT_STEPS // environment["n_envs"]
    _require_exact(expected_vectors, EXPECTED_VECTOR_STEPS, field="vector-step arithmetic")
    expected_updates = expected_vectors - tqc["learning_starts"] // environment["n_envs"]
    _require_exact(expected_updates, EXPECTED_GRADIENT_UPDATES, field="update arithmetic")
    for field in ("expected_vector_steps", "expected_replay_add_calls"):
        _require_exact(counters[field], expected_vectors, field=f"counters.{field}")
    for field in (
        "expected_gradient_updates",
        "expected_train_calls",
        "expected_actor_optimizer_steps",
        "expected_critic_optimizer_steps",
        "expected_entropy_optimizer_steps",
        "expected_target_polyak_updates",
        "expected_final_n_updates",
    ):
        _require_exact(counters[field], expected_updates, field=f"counters.{field}")
    _require_exact(replay["internal_vector_slot_capacity"], expected_vectors, field="replay slots")
    _require_exact(replay["expected_final_position"], 0, field="replay final position")
    _require_exact(replay["expected_final_full"], True, field="replay final full flag")
    parameter_schedule = monitoring["parameter_finite_schedule"]
    parameter_checks = (
        parameter_schedule["pre_learning_checks"]
        + parameter_schedule["vector_step_checks"]
        + parameter_schedule["post_learning_checks"]
    )
    _require_exact(parameter_checks, EXPECTED_PARAMETER_CHECKS, field="parameter check count")
    _require_exact(
        parameter_schedule["vector_step_checks"],
        expected_vectors // parameter_schedule["vector_step_interval"],
        field="parameter vector check count",
    )
    optimizer_finite = monitoring["optimizer_state_finite_schedule"]
    _require_exact(
        optimizer_finite["expected_state_tensor_count_after_first_update"],
        63,
        field="optimizer state tensor count",
    )
    _require_exact(
        optimizer_finite["expected_state_scalar_count_after_first_update"],
        991_423,
        field="optimizer state scalar count",
    )
    _require_exact(
        optimizer_finite["expected_total_nonempty_optimizer_state_checks"],
        201,
        field="optimizer finite checks",
    )
    stage_finite = monitoring["stage_finite_schedule"]
    _require_exact(
        stage_finite["fixed_stage_checks"]
        + stage_finite["evaluation_seed_checks"]
        + stage_finite["visual_seed_checks"]
        + stage_finite["finalization_checks"],
        stage_finite["expected_total_checks"],
        field="stage finite checks",
    )
    _require_exact(stage_finite["expected_total_checks"], 29, field="stage finite check count")
    _require_exact(
        monitoring["expected_lifecycle_resource_samples"],
        len(monitoring["lifecycle_resource_samples"]),
        field="lifecycle resource sample count",
    )
    _require_exact(
        monitoring["lifecycle_resource_samples"][-4:],
        ["pre_visual_encoding", "post_visual_encoding", "pre_finalization", "post_finalization"],
        field="late lifecycle resource samples",
    )
    disk = monitoring["disk_gate_schedule"]
    disk_checks = (
        disk["preflight_checks"] + disk["vector_step_checks"] + disk["post_training_checks"]
    )
    _require_exact(disk_checks, EXPECTED_DISK_CHECKS, field="disk check count")
    _require_exact(
        disk["vector_step_checks"],
        expected_vectors // disk["vector_step_interval"],
        field="disk vector check count",
    )
    _require_exact(training["learn_call"]["log_interval"], 1_000, field="learn log interval")
    _require_exact(training["logger"]["learn_log_interval"], 1_000, field="logger interval")
    lag = monitoring["training_scalar_callback_lag"]
    _require_exact(
        monitoring["throughput_callback_sample_phase"],
        "after_vector_environment_step_before_replay_add_and_current_train_call/v1",
        field="throughput sample phase",
    )
    _require_exact(lag["callback_call_21_num_timesteps"], 105, field="callback lag steps")
    _require_exact(lag["callback_call_21_observed_n_updates"], 0, field="callback lag update")
    _require_exact(lag["callback_first_validation_call"], 22, field="first scalar check")
    _require_exact(
        lag["callback_last_validation_call"], expected_vectors, field="last scalar check"
    )
    _require_exact(
        lag["callback_validation_checks"],
        expected_updates - 1,
        field="callback scalar checks",
    )
    _require_exact(lag["post_learn_validation_checks"], 1, field="post-learn scalar checks")
    _require_exact(
        lag["expected_total_update_scalar_validations"],
        expected_updates,
        field="total scalar validations",
    )
    final_optimizer = training["adam_optimizers"]["final_optimizer_state"]
    _require_exact(
        final_optimizer["required_step_for_every_owned_parameter"],
        expected_updates,
        field="optimizer final step",
    )
    optimizer_resolution = training["adam_optimizers"]["model_construction_resolution"]
    for role in ("actor", "critic", "entropy"):
        _require_exact(
            optimizer_resolution[f"required_observed_{role}_foreach"],
            False,
            field=f"{role} resolved foreach",
        )
        _require_exact(
            optimizer_resolution[f"required_observed_{role}_fused"],
            False,
            field=f"{role} resolved fused",
        )
    _require_exact(
        optimizer_resolution["required_dispatch"],
        "single_tensor_Adam_CPU",
        field="Adam dispatch",
    )
    _validate_parameter_schema(training)
    projected = TOTAL_ENVIRONMENT_STEPS / resource["e0_observed_environment_steps_per_second"]
    if not math.isclose(
        resource["e0_rate_projected_training_seconds"], projected, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ExperimentContractError("E0 rate projection differs")
    floor_projection = TOTAL_ENVIRONMENT_STEPS / resource["minimum_environment_steps_per_second"]
    if not math.isclose(
        resource["throughput_floor_projected_training_seconds"],
        floor_projection,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ExperimentContractError("throughput-floor projection differs")
    supervisor = training["supervisor"]
    _require_exact(
        supervisor["implementation_status_at_predeclaration"],
        "required_not_implemented",
        field="supervisor predeclaration status",
    )
    _require_exact(supervisor["run_directory_mode_octal"], "0700", field="run-directory mode")
    _require_exact(supervisor["in_progress_receipt_mode_octal"], "0600", field="receipt mode")
    _require_exact(
        supervisor["attempt_owner_process_role"], "parent_supervisor", field="attempt owner"
    )
    _require_exact(supervisor["worker_process_count"], 1, field="worker process count")
    _require_exact(supervisor["worker_process_start_count"], 1, field="worker start count")
    _require_exact(supervisor["worker_process_fresh"], True, field="fresh worker")
    _require_exact(supervisor["worker_start_new_session"], True, field="worker session")
    _require_exact(
        supervisor["worker_child_process_creation_allowed"],
        False,
        field="worker child process policy",
    )
    _require_exact(supervisor["multiprocessing_start_method"], "spawn", field="worker start method")
    _require_exact(
        supervisor["parent_worker_message_frame"],
        "uint64be_payload_length_then_canonical_JSON_utf8",
        field="worker message framing",
    )
    _require_exact(
        supervisor["parent_worker_channel"],
        "full_duplex_two_dedicated_inherited_os_pipes_length_prefixed_canonical_JSON_messages",
        field="worker channel",
    )
    _require_exact(
        supervisor["parent_worker_sequence_rule"],
        "independent_contiguous_integer_counter_per_direction_starting_at_zero_no_duplicates",
        field="worker message sequence",
    )
    _require_exact(
        supervisor["parent_to_worker_message_types"],
        ["admit_execution_manifest", "request_shutdown"],
        field="parent-to-worker messages",
    )
    _require_exact(
        supervisor["model_construction_requires_matching_manifest_ack"],
        True,
        field="manifest acknowledgement",
    )
    _require_exact(
        supervisor["worker_process_group_identity_rule"],
        "parent_observation_and_worker_self_report_must_match_and_pid_equals_pgid_equals_sid",
        field="worker process-group identity",
    )
    _require_exact(supervisor["worker_started_timeout_seconds"], 30.0, field="worker start timeout")
    _require_exact(
        supervisor["worker_started_timeout_origin"],
        "parent_time_perf_counter_immediately_before_invoking_Process_start",
        field="worker start timeout origin",
    )
    _require_exact(
        supervisor["worker_start_failure_shutdown_sequence"][-1],
        "require_child_and_validated_process_group_absent",
        field="worker start cleanup",
    )
    _require_exact(
        supervisor["parent_signal_shutdown_sequence"],
        [
            "record_received_signal",
            "stop_accepting_nonterminal_worker_messages",
            "if_group_identity_valid_send_SIGTERM_to_exact_worker_pgid_else_terminate_exact_child_pid",
            "wait_up_to_10_seconds",
            "if_group_identity_valid_send_SIGKILL_to_exact_worker_pgid_else_kill_exact_child_pid_if_still_alive",
            "wait_and_join_up_to_30_seconds",
            "close_both_pipes",
            "require_child_and_validated_process_group_absent_before_failure_receipt",
        ],
        field="signal cleanup",
    )
    _require_exact(
        supervisor["parent_exception_shutdown_sequence"][-1],
        "require_child_and_validated_process_group_absent_before_failure_receipt",
        field="parent exception cleanup",
    )
    _require_exact(
        supervisor["all_parent_termination_paths_use_bounded_cleanup"],
        True,
        field="bounded parent cleanup",
    )
    _require_exact(
        supervisor["parent_worker_message_maximum_bytes"], 65_536, field="worker message bound"
    )
    _require_exact(
        supervisor["stage_timeout_enforcement"],
        "parent_supervisor_monotonic_deadline_per_named_worker_stage",
        field="stage timeout enforcement",
    )
    _require_exact(
        supervisor["allowed_state_order"],
        [
            "preflight",
            "model_constructed",
            "training",
            "training_complete",
            "persistence",
            "strict_reload",
            "evaluation",
            "visual_encoding",
            "finalized",
        ],
        field="supervisor state order",
    )
    if any(
        type(value) is not float or value <= 0.0
        for value in supervisor["stage_timeouts_seconds"].values()
    ):
        raise ExperimentContractError("supervisor stage timeouts must be positive exact floats")
    _require_exact(
        supervisor["training_worker_wall_gate_seconds"],
        resource["sampled_wall_time_failure_threshold_seconds"],
        field="worker wall gate",
    )
    _require_exact(
        supervisor["stage_timeouts_seconds"]["training"],
        supervisor["training_worker_wall_gate_seconds"]
        + supervisor["training_parent_deadline_grace_seconds"],
        field="parent training timeout",
    )
    _require_exact(
        supervisor["timeout_shutdown_sequence"][-1],
        "require_worker_process_group_absent_before_final_receipt",
        field="timeout shutdown completion",
    )
    trusted = training["persistence"]["trusted_local_state"]
    _require_exact(
        training["persistence"]["execution_manifest_binding_required"],
        True,
        field="persistence manifest binding",
    )
    _require_exact(
        trusted["replay_expected_array_payload_bytes"],
        resource["replay_buffer_allocation_bytes"],
        field="streamed replay byte count",
    )
    _require_exact(
        trusted["replay_serialization"],
        "streamed_exclusive_0600_partial_then_atomic_promote/v1",
        field="replay serialization",
    )
    _require_exact(trusted["replay_no_overwrite"], True, field="replay overwrite policy")
    _require_exact(
        trusted["replay_partial_artifact_eligible"],
        False,
        field="partial replay eligibility",
    )
    model_load = trusted["strict_model_load_call"]
    _require_exact(
        model_load["path_argument"], "exact_bound_exclusive_binary_reader", field="model load path"
    )
    _require_exact(model_load["env_argument"], None, field="model load environment")
    _require_exact(model_load["device"], "cpu", field="model load device")
    _require_exact(model_load["print_system_info"], False, field="model system-info flag")
    _require_exact(model_load["force_reset"], True, field="model force-reset flag")
    replay_load = trusted["strict_replay_load_call"]
    _require_exact(
        replay_load["path_argument"],
        "exact_bound_exclusive_binary_reader",
        field="replay load path",
    )
    _require_exact(replay_load["truncate_last_traj"], False, field="replay truncation")
    _require_exact(
        trusted["simultaneous_full_replay_buffers_allowed"],
        1,
        field="simultaneous replay buffers",
    )
    _require_exact(
        len(trusted["strict_reload_hash_pairs"]),
        7,
        field="strict reload hash pair count",
    )
    _require_exact(
        trusted["optimizer_state_hash_fields_per_owned_parameter"],
        ["step", "exp_avg", "exp_avg_sq"],
        field="strict optimizer-state hash fields",
    )
    _require_exact(
        trusted["optimizer_param_group_fields"],
        [
            "lr",
            "betas",
            "eps",
            "weight_decay",
            "amsgrad",
            "maximize",
            "foreach",
            "capturable",
            "differentiable",
            "fused",
            "decoupled_weight_decay",
        ],
        field="strict optimizer parameter-group fields",
    )
    _require_exact(
        trusted["replay_metadata_hash_fields"],
        [
            "class_id",
            "requested_transition_capacity",
            "internal_vector_slot_capacity",
            "n_envs",
            "position",
            "full",
            "optimize_memory_usage",
            "handle_timeout_termination",
            "n_step_return",
            "device_type",
            "observation_space_sha256",
            "action_space_sha256",
        ],
        field="strict replay metadata hash fields",
    )
    _require_exact(
        trusted["strict_reload_byte_exact_state_hash_equality_required"],
        True,
        field="strict reload state equality",
    )
    actor_export = training["persistence"]["actor_export"]
    _require_exact(
        actor_export["v2_equivalence_verifier_status_at_predeclaration"],
        "required_not_implemented",
        field="actor equivalence predeclaration status",
    )
    for output in actor_export["equivalence_output_schema"].values():
        _require_exact(output["shape"], [4, 17], field="actor equivalence shape")
        _require_exact(output["dtype"], "<f4", field="actor equivalence dtype")
        _require_exact(output["memory_order"], "C", field="actor equivalence memory order")
    _require_exact(
        actor_export["equivalence_array_hash_rule"],
        "sha256_uint64be_header_length_then_canonical_dtype_shape_header_then_uint64be_raw_length_then_c_order_bytes/v1",
        field="actor equivalence array hash rule",
    )
    _require_exact(
        actor_export["all_eight_equivalence_hashes_use_one_array_rule"],
        True,
        field="actor equivalence hash coverage",
    )
    _require_exact(
        actor_export["equivalence_receipt_required_state_hashes"],
        ["trusted_actor_state_sha256", "loaded_actor_state_sha256"],
        field="actor state receipt hashes",
    )
    _require_exact(
        actor_export["actor_state_hash_equality_required"],
        True,
        field="actor state hash equality",
    )
    _require_exact(
        actor_export["log_std_rule"],
        "actor_linear_log_std_output_clamped_elementwise_to_closed_interval_negative20_2",
        field="actor log_std rule",
    )
    _require_exact(
        actor_export["seeded_sample_rng_rule"],
        "save_cpu_rng_state_manual_seed_97001_compute_restore_then_repeat_independently_for_loaded_actor",
        field="actor sample RNG rule",
    )
    _require_exact(
        actor_export["seeded_sample_rule"],
        "torch_distributions_Normal_mean_exp_clamped_log_std_rsample_then_tanh",
        field="actor sample rule",
    )
    _require_exact(
        training["runtime_requirements"]["torch_optimizer_base_source_sha256"],
        "ebc2511e0a07b18bbfba44f61667ddaeee47f23cdc856a43234d89007010fe86",
        field="Torch optimizer source SHA-256",
    )
    attempt = training["attempt"]
    _require_exact(
        attempt["execution_manifest_required_before_worker_start"],
        False,
        field="pre-worker manifest",
    )
    _require_exact(
        attempt["preflight_contract_required_before_worker_start"],
        True,
        field="pre-worker contract",
    )
    _require_exact(
        attempt["worker_reverifies_execution_manifest_before_model_construction"],
        True,
        field="pre-model manifest",
    )
    _require_exact(
        attempt["same_execution_manifest_required_for_training_persistence_evaluation_and_visuals"],
        True,
        field="one-manifest execution",
    )


def _validate_visual(design: dict[str, Any]) -> None:
    visual = design["visual_evidence"]
    training = design["training_projection"]
    _require_exact(tuple(visual["seeds"]), VIDEO_SEEDS, field="visual seeds")
    _require_exact(
        visual["capture_phase"],
        "after_live_trace_boundary_before_next_action_inference_when_step_less_than_1000_terminal_boundary_has_no_next_inference",
        field="visual capture phase",
    )
    _require_exact(CAPTURE_INDICES[0], visual["capture_step_start"], field="capture start")
    _require_exact(CAPTURE_INDICES[-1], visual["capture_step_stop_inclusive"], field="capture stop")
    _require_exact(visual["capture_step_stride"], 2, field="capture stride")
    _require_exact(visual["expected_frame_count"], len(CAPTURE_INDICES), field="frame count")
    _require_exact(
        visual["camera"]["camera_record_count"], len(CAPTURE_INDICES), field="camera count"
    )
    _require_exact(
        visual["noninterference"]["expected_check_count_per_seed"],
        len(CAPTURE_INDICES),
        field="noninterference count",
    )
    encoding = visual["encoding"]
    expected_duration = len(CAPTURE_INDICES) * encoding["duration_ms_per_frame"]
    _require_exact(encoding["duration_list_length"], len(CAPTURE_INDICES), field="duration count")
    _require_exact(encoding["encoded_duration_ms"], expected_duration, field="encoded duration")
    _require_exact(encoding["simulated_boundary_span_ms"], 15_000, field="boundary span")
    _require_exact(
        encoding["encoded_minus_simulated_span_ms"],
        encoding["duration_ms_per_frame"],
        field="uniform final-frame duration",
    )
    _require_exact(
        encoding["final_boundary_standard_frame_duration_ms"],
        encoding["duration_ms_per_frame"],
        field="final-frame duration",
    )
    _require_exact(encoding["additional_terminal_hold"], False, field="additional terminal hold")
    _require_exact(encoding["terminal_hold_frame_count"], 0, field="terminal hold frames")
    _require_exact(encoding["terminal_hold_duration_ms"], 0, field="terminal hold duration")
    _require_exact(encoding["background_rgba"], [0, 0, 0, 0], field="WebP background")
    _require_exact(encoding["allow_mixed"], False, field="WebP allow_mixed")
    _require_exact(encoding["lossless"], True, field="WebP lossless")
    _require_exact(visual["renderer"]["environment_render_mode"], None, field="render mode")
    _require_exact(visual["renderer"]["max_geom"], 10_000, field="direct renderer max_geom")
    if visual["renderer"]["max_geom"] == training["environment"]["mujoco_env_max_geom"]:
        raise ExperimentContractError("Gym and direct-renderer max_geom choices were conflated")
    _require_exact(
        _semantic_sha256(visual["renderer"]["scene_option"]),
        visual["renderer"]["scene_option_sha256"],
        field="MuJoCo scene option SHA-256",
    )
    hashing = visual["hash_canonicalization"]
    _require_exact(
        hashing["array_rule"],
        "sha256_uint64be_header_length_then_header_then_uint64be_raw_length_then_c_order_bytes/v1",
        field="visual array hash rule",
    )
    _require_exact(hashing["per_frame_array_shape"], [480, 480, 3], field="frame hash shape")
    _require_exact(
        hashing["sequence_array_shape"],
        [len(CAPTURE_INDICES), 480, 480, 3],
        field="sequence hash shape",
    )
    _require_exact(hashing["array_dtype_str"], "|u1", field="visual hash dtype")
    _require_exact(
        hashing["record_rule"],
        "sha256_of_canonical_json_utf8_ordered_record_array/v1",
        field="visual record hash rule",
    )
    _require_exact(hashing["camera_record_domain_separator"], None, field="camera hash domain")
    _require_exact(
        hashing["noninterference_record_domain_separator"],
        None,
        field="noninterference hash domain",
    )
    _require_exact(
        hashing["float_encoding"],
        "finite_Python_3.13_json_binary64_shortest_round_trip_number",
        field="visual float encoding",
    )
    _require_exact(len(hashing["camera_record_schema"]), 15, field="camera record width")
    _require_exact(
        visual["per_frame_record_fields"],
        [item["name"] for item in hashing["camera_record_schema"]],
        field="camera record field order",
    )
    _require_exact(
        len(hashing["noninterference_record_schema"]),
        16,
        field="noninterference record width",
    )
    decode = visual["decode_verification"]
    _require_exact(decode["frame_count"], len(CAPTURE_INDICES), field="decoded frame count")
    _require_exact(decode["duration_ms_total"], expected_duration, field="decoded duration")
    _require_exact(
        decode["frame_load_before_duration_read_required"],
        True,
        field="decoded frame load before duration read",
    )
    _require_exact(decode["exact_pixel_equality_to_each_raw_frame"], True, field="pixel equality")
    _require_exact(
        decode["adjacent_raw_frame_hashes_must_all_differ"],
        True,
        field="adjacent frame hashes",
    )
    _require_exact(
        decode["adjacent_identical_frame_disposition"],
        "visual_artifact_ineligible_codec_constraint_not_behavior_gate_failure_no_rerun",
        field="adjacent frame disposition",
    )
    for field in (
        "rerun_allowed",
        "replay_allowed",
        "replacement_rollout_allowed",
        "metric_input_allowed",
    ):
        _require_exact(visual[field], False, field=f"visual.{field}")
    for field in (
        "observed_decoded_frame_count",
        "observed_decoded_frame_durations_ms",
        "observed_decoded_duration_ms_total",
        "observed_decoded_loop",
        "observed_decoded_mode",
        "observed_decoded_frame_shape",
        "observed_decoded_frame_dtype",
        "observed_decoded_memory_order",
        "observed_exact_pixel_equality",
    ):
        if field not in visual["receipt_must_bind"]:
            raise ExperimentContractError(f"visual receipt omits {field}")


def _validate_design_and_audit(design: dict[str, Any], audit: dict[str, Any]) -> None:
    _require_keys(design, _DESIGN_KEYS, field="v2 design")
    _require_keys(audit, _AUDIT_KEYS, field="E0 reuse audit")
    _require_exact(design["schema_version"], 2, field="schema_version")
    _require_exact(design["design_id"], DESIGN_ID, field="design_id")
    _require_exact(design["design_status"], "predeclared_not_executed", field="design_status")
    _require_exact(design["formal_experiment_eligible"], False, field="formal eligibility")
    _require_exact(audit["audit_id"], AUDIT_ID, field="audit_id")
    _require_exact(audit["authorizes_training"], False, field="audit training authority")
    if '"implementation_status"' in json.dumps(design, sort_keys=True):
        raise ExperimentContractError("plain implementation_status is temporally ambiguous")
    supersession = design["supersession"]
    _require_exact(
        supersession["scope"],
        "repository_authorized_future_executions_only",
        field="supersession scope",
    )
    _require_exact(
        supersession["superseded_design_file_sha256"],
        V1_DESIGN_FILE_SHA256,
        field="v1 file SHA-256",
    )
    _require_exact(
        supersession["superseded_design_semantic_sha256"],
        V1_DESIGN_SEMANTIC_SHA256,
        field="v1 semantic SHA-256",
    )
    _require_exact(supersession["universal_v1_execution_claim"], False, field="universal v1 claim")
    required_e0 = design["required_e0"]
    calibration = audit["calibration_evidence"]
    for design_field, audit_field in (
        ("design_artifact_sha256", "design_file_sha256"),
        ("design_semantic_sha256", "design_semantic_sha256"),
        ("receipt_content_sha256", "receipt_file_sha256"),
        ("receipt_byte_count", "receipt_byte_count"),
        ("runtime_sha256", "runtime_sha256"),
    ):
        _require_exact(
            required_e0[design_field], calibration[audit_field], field=f"required_e0.{design_field}"
        )
    _require_exact(
        required_e0["reuse_audit_file_sha256"], AUDIT_FILE_SHA256, field="audit file binding"
    )
    _require_exact(
        required_e0["reuse_audit_semantic_sha256"],
        AUDIT_SEMANTIC_SHA256,
        field="audit semantic binding",
    )
    _require_exact(
        required_e0["v2_training_projection_sha256"],
        TRAINING_PROJECTION_SHA256,
        field="training projection binding",
    )
    projection = design["training_projection"]
    if audit["v2_training_projection"] != projection:
        raise ExperimentContractError("E0 audit full v2 training projection differs from design")
    if _semantic_sha256(projection) != TRAINING_PROJECTION_SHA256:
        raise ExperimentContractError("full v2 training projection SHA-256 differs")
    _require_exact(
        audit["v2_training_projection_sha256"],
        TRAINING_PROJECTION_SHA256,
        field="audit projection SHA-256",
    )
    shared = audit["shared_projection"]
    mapping = shared["mapping_spec"]
    values = shared["projected_values"]
    _require_exact(shared["mapping_count"], 37, field="E0 mapping count")
    _require_exact(shared["distinct_concept_count"], 36, field="E0 concept count")
    _require_exact(len(mapping), 37, field="E0 mapping length")
    _require_exact(len(values), 37, field="E0 projected-value length")
    _require_exact(
        _semantic_sha256(mapping), shared["mapping_spec_sha256"], field="mapping SHA-256"
    )
    _require_exact(
        _semantic_sha256(values),
        shared["e0_projected_values_sha256"],
        field="E0 projection SHA-256",
    )
    _require_exact(
        shared["e0_projected_values_sha256"],
        shared["v2_projected_values_sha256"],
        field="shared projection equality",
    )
    for mapping_row, value_row in zip(mapping, values, strict=True):
        _require_exact(value_row["e0_path"], mapping_row["e0_path"], field="projected E0 path")
        target_path = mapping_row["v2_training_projection_path"]
        _require_exact(
            value_row["v2_training_projection_path"], target_path, field="projected v2 path"
        )
        _require_exact(
            value_row["value"],
            _at(projection, target_path, field="v2 projection"),
            field=target_path,
        )
    delta_classes = [item["class"] for item in audit["declared_deltas"]]
    _require_exact(
        delta_classes,
        [
            "seed_schedule",
            "training_horizon",
            "persistence_export",
            "protected_evaluation",
            "same_rollout_visual_evidence",
        ],
        field="E0 delta classes",
    )
    protocol_additions = audit["declared_protocol_additions"]
    _require_exact(
        [item["class"] for item in protocol_additions],
        ["effective_default_declarations", "supervisor_and_integrity_hardening"],
        field="E0 protocol addition classes",
    )
    for addition in protocol_additions:
        _require_exact(
            addition["scientific_or_mapped_training_workload_change"],
            False,
            field="protocol addition workload change",
        )
        _require_exact(
            addition["mapped_training_workload_value_change_allowed"],
            False,
            field="protocol addition mapped-value authority",
        )
        _require_exact(addition["evidence_granted"], [], field="protocol addition evidence")
        _require_exact(
            addition["exact_e0_hidden_default_invariance_claim"],
            False,
            field="protocol hidden-default invariance claim",
        )
    default_evidence = audit["default_resolution_evidence"]
    _require_exact(
        default_evidence["exact_hidden_default_invariance_claim"],
        False,
        field="E0 hidden-default invariance",
    )
    _require_exact(
        default_evidence["e0_observed_model_parameter_structure_sha256"],
        default_evidence["v2_required_model_parameter_structure_sha256"],
        field="E0 parameter structure evidence",
    )
    for forbidden_key in (
        "v2_design_file_sha256",
        "v2_design_semantic_sha256",
        "execution_manifest_sha256",
    ):
        if _contains_key(audit, forbidden_key):
            raise ExperimentContractError("E0 audit contains a forbidden future-artifact binding")
    _require_exact(audit["reuse_decision"]["audit_passed"], True, field="static E0 audit")
    _require_exact(audit["reuse_decision"]["behavioral_evidence"], False, field="E0 behavior")
    _validate_training_projection(projection)
    evaluation = design["evaluation"]
    _require_exact(
        evaluation["implementation_status_at_predeclaration"],
        "required_not_implemented_for_v2",
        field="evaluation predeclaration status",
    )
    _require_exact(
        evaluation["environment_class"],
        "oracle_composition.envs.humanoid.SubstepContactHumanoidEnv",
        field="evaluation environment class",
    )
    _require_exact(
        evaluation["base_environment_class"],
        "gymnasium.envs.mujoco.humanoid_v5.HumanoidEnv",
        field="evaluation base environment class",
    )
    _require_exact(
        evaluation["wrapper_order_outer_to_inner"],
        [
            "gymnasium.wrappers.common.TimeLimit",
            "gymnasium.wrappers.common.OrderEnforcing",
            "gymnasium.wrappers.common.PassiveEnvChecker",
            "oracle_composition.envs.humanoid.SubstepContactHumanoidEnv",
        ],
        field="evaluation environment wrapper stack",
    )
    _require_exact(evaluation["capture_substep_contacts"], True, field="evaluation contacts")
    _require_exact(
        evaluation["training_stack_reuse_allowed"], False, field="evaluation stack reuse"
    )
    _require_exact(
        evaluation["contact_capture_id"],
        "all_mujoco_substeps_after_mj_step/v1",
        field="evaluation contact capture",
    )
    equivalence = evaluation["instrumentation_equivalence_gate"]
    _require_exact(
        equivalence["implementation_status_at_predeclaration"],
        "required_not_implemented",
        field="instrumentation equivalence predeclaration status",
    )
    _require_exact(
        equivalence["required_before_protected_evaluation"], True, field="instrumentation gate"
    )
    _require_exact(equivalence["seed"], 98_001, field="instrumentation probe seed")
    _require_exact(equivalence["horizon_steps"], 1_000, field="instrumentation horizon")
    _require_exact(
        equivalence["scope"],
        "preflight_canary_only_not_complete_dynamics_equivalence_evidence",
        field="instrumentation canary scope",
    )
    _require_exact(
        equivalence["action_sha256"],
        "8cbe0bc01c56134fdab90ba6506ebd74622f4929eb3216d332c05e459fa63af4",
        field="instrumentation actions",
    )
    shadow = evaluation["actor_driven_shadow_equivalence"]
    _require_exact(evaluation["environment_instances_per_seed"], 2, field="evaluation env count")
    _require_exact(shadow["required_for_each_evaluation_seed"], True, field="shadow requirement")
    _require_exact(shadow["horizon_steps"], 1_000, field="shadow horizon")
    _require_exact(
        shadow["same_physical_action_bytes_applied_to_both_environments"],
        True,
        field="shadow actions",
    )
    _require_exact(shadow["plain_shadow_used_for_metrics"], False, field="shadow metric use")
    _require_exact(shadow["plain_shadow_used_for_visuals"], False, field="shadow visual use")
    if (
        "twenty_ordered_actor_driven_shadow_equivalence_receipt_sha256_values"
        not in evaluation["evaluation_receipt_must_bind"]
    ):
        raise ExperimentContractError("evaluation receipt omits actor-driven shadow equivalence")
    _require_exact(tuple(evaluation["seed_order"]), EVALUATION_SEEDS, field="evaluation seeds")
    _require_exact(evaluation["episodes_per_seed"], 1, field="evaluation episodes per seed")
    _require_exact(evaluation["step_count_per_seed"], 1_000, field="evaluation steps")
    _require_exact(evaluation["expected_final_terminated"], False, field="final terminated")
    _require_exact(evaluation["expected_final_truncated"], True, field="final truncated")
    trace = evaluation["canonical_trace"]
    _require_exact(trace["sample_count"], 1_001, field="trace sample count")
    _require_exact(
        [signal["name"] for signal in trace["numeric_signals"]],
        [
            "robot.qpos",
            "robot.qvel",
            "robot.root_position_world_m",
            "evaluation.simulation_time_seconds",
            "controller.observation",
            "controller.action",
            "controller.physical_control",
            "controller.applied_torque",
            "controller.saturation",
            "contact.floor_normal_force_n",
            "contact.active_count_by_substep",
            "task.progress",
            "evaluation.root_height_m",
            "evaluation.torso_up_z",
            "evaluation.healthy",
            "evaluation.upright",
            "evaluation.non_foot_floor_contact",
        ],
        field="trace numeric signal order",
    )
    _require_exact(
        [signal["name"] for signal in trace["explicit_missing_signals"]],
        [
            "controller.motor_target",
            "controller.energy_j",
            "reference.frame",
            "reference.window_index",
            "oracle.mode",
            "oracle.phase",
            "oracle.transition_guard_margin",
            "recovery.disturbance",
            "recovery.rejoin_state",
        ],
        field="trace missing signals",
    )
    _require_exact(
        trace["metrics_elapsed_time_source"],
        "evaluation.simulation_time_seconds_direct_signal_not_nominal_sample_time",
        field="trace metric clock",
    )
    _require_exact(
        trace["action_observation_alignment"],
        "sample_0_action_is_zero_reset_sentinel_and_for_each_sample_k_1_through_1000_action_k_is_actor_of_controller_observation_k_minus_1_applied_during_transition_to_boundary_k",
        field="trace action alignment",
    )
    if "execution_manifest" not in trace["artifact_bindings"]:
        raise ExperimentContractError("trace does not bind the execution manifest")
    _require_exact(trace["contact_event"]["event_type"], "contact.active", field="contact event")
    _require_exact(
        trace["terminal_event"],
        {
            "event_type": "episode.truncated",
            "source": "Gymnasium_TimeLimit_exact_step_1000/v1",
            "sample_index": 1_000,
            "value": "true",
            "expected_count": 1,
        },
        field="terminal event",
    )
    _require_exact(
        design["visual_evidence"]["implementation_status_at_predeclaration"],
        "required_not_implemented",
        field="visual predeclaration status",
    )
    _require_exact(
        design["execution_manifest"]["implementation_status_at_predeclaration"],
        "required_not_implemented",
        field="manifest predeclaration status",
    )
    preflight_protocol = design["execution_manifest"]["two_phase_preflight_protocol"]
    _require_exact(
        preflight_protocol["preflight_contract_finalized_before_worker_start"],
        True,
        field="preflight contract order",
    )
    _require_exact(
        preflight_protocol["worker_actions_before_execution_manifest"],
        ["runtime_reinspection", "environment_instrumentation_equivalence_probe"],
        field="pre-manifest worker actions",
    )
    _require_exact(
        preflight_protocol["worker_actions_forbidden_before_execution_manifest"][0],
        "model_construction",
        field="pre-manifest model prohibition",
    )
    for field in (
        "v2_contract_source_sha256",
        "dependency_lock_sha256",
        "project_python_source_tree_sha256",
        "runtime_reinspection_source_sha256",
        "python_executable_file_sha256",
        "host_hardware_identity",
    ):
        if field not in preflight_protocol["preflight_contract_must_bind"]:
            raise ExperimentContractError(f"preflight contract omits {field}")
    manifest = design["execution_manifest"]
    _require_exact(
        manifest["independent_review_receipt_required"], True, field="review receipt requirement"
    )
    _require_exact(
        manifest["independent_review_receipt_bound_later_by_manifest"],
        True,
        field="review receipt late binding",
    )
    _require_exact(
        manifest["design_contains_future_review_receipt_placeholder"],
        False,
        field="future review placeholder",
    )
    review = _require_keys(
        manifest["independent_review_receipt_contract"],
        {
            "schema_version",
            "receipt_id",
            "canonical_encoding",
            "exact_keys",
            "reviewer_role_required",
            "reviewer_identity_rule",
            "review_completed_at_utc_rule",
            "verdict_required",
            "required_reviewed_hash_bindings",
            "reviewed_source_paths",
            "hash_binding_rule",
            "finding_count_rule",
            "resolved_finding_count_rule",
            "unresolved_counts_required",
            "observed_check_receipts_rule",
            "completed_before_preflight_contract",
            "receipt_content_sha256_bound_by_preflight_and_execution_manifest",
            "no_placeholder_hash_in_design",
            "trust_boundary",
        },
        field="review receipt contract",
    )
    _require_exact(review["schema_version"], 1, field="review receipt schema version")
    _require_exact(
        review["receipt_id"],
        "tqc_dev_1m_v2_independent_design_review/v1",
        field="review receipt id",
    )
    _require_exact(review["canonical_encoding"], "canonical_JSON_utf8/v1", field="review encoding")
    _require_exact(
        review["exact_keys"],
        [
            "schema_version",
            "receipt_id",
            "reviewer_role",
            "reviewer_identity",
            "review_completed_at_utc",
            "verdict",
            "reviewed_design_file_sha256",
            "reviewed_design_semantic_sha256",
            "reviewed_e0_audit_file_sha256",
            "reviewed_e0_audit_semantic_sha256",
            "reviewed_v2_contract_source_sha256",
            "reviewed_v2_contract_test_file_sha256",
            "finding_count",
            "resolved_finding_count",
            "unresolved_p0_count",
            "unresolved_p1_count",
            "unresolved_p2_count",
            "observed_check_receipts",
        ],
        field="review receipt keys",
    )
    _require_exact(review["reviewer_role_required"], "independent_read_only", field="review role")
    _require_exact(
        review["reviewer_identity_rule"],
        "nonempty_string_distinct_from_design_author_and_implementation_owner",
        field="reviewer identity",
    )
    _require_exact(
        review["review_completed_at_utc_rule"],
        "RFC3339_UTC_Z_seconds_precision",
        field="review completion timestamp",
    )
    _require_exact(review["verdict_required"], "GO", field="review verdict")
    _require_exact(
        review["required_reviewed_hash_bindings"],
        [
            "reviewed_design_file_sha256",
            "reviewed_design_semantic_sha256",
            "reviewed_e0_audit_file_sha256",
            "reviewed_e0_audit_semantic_sha256",
            "reviewed_v2_contract_source_sha256",
            "reviewed_v2_contract_test_file_sha256",
        ],
        field="review hash bindings",
    )
    _require_exact(
        review["reviewed_source_paths"],
        {
            "reviewed_v2_contract_source_sha256": (
                "src/oracle_composition/experiments/tqc_development_contract_v2.py"
            ),
            "reviewed_v2_contract_test_file_sha256": (
                "tests/experiments/test_tqc_development_contract_v2.py"
            ),
        },
        field="review source paths",
    )
    _require_exact(
        review["hash_binding_rule"],
        "receipt_values_must_equal_the_exact_final_reviewed_bytes_and_semantics",
        field="review hash rule",
    )
    _require_exact(
        review["finding_count_rule"],
        "exact_nonnegative_integer",
        field="review finding count",
    )
    _require_exact(
        review["resolved_finding_count_rule"],
        "exact_nonnegative_integer_equal_to_finding_count",
        field="review resolved findings",
    )
    _require_exact(
        review["unresolved_counts_required"],
        {"p0": 0, "p1": 0, "p2": 0},
        field="review unresolved findings",
    )
    _require_exact(
        review["observed_check_receipts_rule"],
        "nonempty_ordered_array_of_content_sha256_values",
        field="review observed checks",
    )
    _require_exact(
        review["completed_before_preflight_contract"],
        True,
        field="review completion order",
    )
    _require_exact(
        review["receipt_content_sha256_bound_by_preflight_and_execution_manifest"],
        True,
        field="review receipt binding",
    )
    _require_exact(review["no_placeholder_hash_in_design"], True, field="review placeholder")
    _require_exact(
        review["trust_boundary"],
        "reviewer_attestation_and_process_local_file_identity_not_a_cryptographic_signature",
        field="review trust boundary",
    )
    _validate_visual(design)


@dataclass(frozen=True, slots=True)
class LoadedTQCDevelopmentDesignV2:
    """Exact design and non-circular E0 audit bytes."""

    encoded_bytes: bytes
    file_sha256: str
    semantic_sha256: str
    e0_audit_encoded_bytes: bytes
    e0_audit_file_sha256: str
    e0_audit_semantic_sha256: str
    training_projection_sha256: str
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _LOADED_ISSUER:
            raise ExperimentContractError("v2 designs may only be issued by the strict loader")
        design = json.loads(self.encoded_bytes)
        audit = json.loads(self.e0_audit_encoded_bytes)
        _require_exact(_sha256(self.encoded_bytes), self.file_sha256, field="design byte binding")
        _require_exact(
            _semantic_sha256(design), self.semantic_sha256, field="design semantic binding"
        )
        _require_exact(
            _sha256(self.e0_audit_encoded_bytes),
            self.e0_audit_file_sha256,
            field="audit byte binding",
        )
        _require_exact(
            _semantic_sha256(audit), self.e0_audit_semantic_sha256, field="audit semantic binding"
        )
        _require_exact(self.file_sha256, DESIGN_FILE_SHA256, field="compiled design file SHA-256")
        _require_exact(
            self.semantic_sha256, DESIGN_SEMANTIC_SHA256, field="compiled design semantic SHA-256"
        )
        _require_exact(
            self.e0_audit_file_sha256, AUDIT_FILE_SHA256, field="compiled audit file SHA-256"
        )
        _require_exact(
            self.e0_audit_semantic_sha256,
            AUDIT_SEMANTIC_SHA256,
            field="compiled audit semantic SHA-256",
        )
        _require_exact(
            self.training_projection_sha256,
            TRAINING_PROJECTION_SHA256,
            field="compiled training projection SHA-256",
        )
        _validate_design_and_audit(design, audit)

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.encoded_bytes)

    def e0_audit_to_dict(self) -> dict[str, Any]:
        return json.loads(self.e0_audit_encoded_bytes)


@dataclass(frozen=True, slots=True)
class ValidatedV2E0ReuseEvidence:
    """Resource-only evidence binding; never a training authorization."""

    e0_design_file_sha256: str
    e0_receipt_file_sha256: str
    e0_runtime_sha256: str
    v1_design_file_sha256: str
    training_projection_sha256: str
    resource_prior_accepted: bool
    authorizes_training: bool
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _E0_PREFLIGHT_ISSUER:
            raise ExperimentContractError("E0 reuse evidence may only be issued by validation")
        for field in (
            "e0_design_file_sha256",
            "e0_receipt_file_sha256",
            "e0_runtime_sha256",
            "v1_design_file_sha256",
            "training_projection_sha256",
        ):
            _assert_sha(getattr(self, field), field=field)
        _require_exact(self.resource_prior_accepted, True, field="resource prior")
        _require_exact(self.authorizes_training, False, field="E0 training authority")


def load_tqc_development_design_v2(
    design_path: Path,
    e0_reuse_audit_path: Path,
) -> LoadedTQCDevelopmentDesignV2:
    """Load the exact v2 design plus its prior, non-circular E0 audit."""

    design_bytes, design = _artifact(
        design_path,
        maximum_bytes=MAX_DESIGN_BYTES,
        artifact="TQC v2 development design",
        file_sha256=DESIGN_FILE_SHA256,
        semantic_sha256=DESIGN_SEMANTIC_SHA256,
    )
    audit_bytes, audit = _artifact(
        e0_reuse_audit_path,
        maximum_bytes=MAX_AUDIT_BYTES,
        artifact="TQC v2 E0 reuse audit",
        file_sha256=AUDIT_FILE_SHA256,
        semantic_sha256=AUDIT_SEMANTIC_SHA256,
    )
    _validate_design_and_audit(design, audit)
    return LoadedTQCDevelopmentDesignV2(
        encoded_bytes=design_bytes,
        file_sha256=DESIGN_FILE_SHA256,
        semantic_sha256=DESIGN_SEMANTIC_SHA256,
        e0_audit_encoded_bytes=audit_bytes,
        e0_audit_file_sha256=AUDIT_FILE_SHA256,
        e0_audit_semantic_sha256=AUDIT_SEMANTIC_SHA256,
        training_projection_sha256=TRAINING_PROJECTION_SHA256,
        _issuer=_LOADED_ISSUER,
    )


def validate_v2_e0_reuse_artifacts(
    loaded: LoadedTQCDevelopmentDesignV2,
    *,
    calibration_design_path: Path,
    calibration_receipt_path: Path,
    superseded_v1_design_path: Path,
) -> ValidatedV2E0ReuseEvidence:
    """Reverify E0 and v1 bytes while preserving E0's resource-only ceiling."""

    if type(loaded) is not LoadedTQCDevelopmentDesignV2:
        raise ExperimentContractError("v2 E0 validation requires a strict-loaded design")
    _calibration_bytes, calibration_design = _artifact(
        calibration_design_path,
        maximum_bytes=MAX_DESIGN_BYTES,
        artifact="E0 calibration design",
        file_sha256=E0_DESIGN_FILE_SHA256,
        semantic_sha256=E0_DESIGN_SEMANTIC_SHA256,
    )
    _receipt_bytes, receipt = _artifact(
        calibration_receipt_path,
        maximum_bytes=MAX_E0_RECEIPT_BYTES,
        artifact="E0 calibration receipt",
        file_sha256=E0_RECEIPT_FILE_SHA256,
        semantic_sha256=E0_RECEIPT_SEMANTIC_SHA256,
    )
    _v1_bytes, _v1 = _artifact(
        superseded_v1_design_path,
        maximum_bytes=MAX_DESIGN_BYTES,
        artifact="superseded v1 development design",
        file_sha256=V1_DESIGN_FILE_SHA256,
        semantic_sha256=V1_DESIGN_SEMANTIC_SHA256,
    )
    checks = {
        "completion_status": "complete",
        "calibration_gate_passed": True,
        "measured_workload_gates_passed": True,
        "checkpoint_emitted": False,
        "replay_buffer_emitted": False,
        "normalizer_emitted": False,
        "controller_artifact": None,
        "eligible_for_controller_training": False,
        "eligible_for_behavioral_evaluation": False,
        "design_artifact_sha256": E0_DESIGN_FILE_SHA256,
        "design_semantic_sha256": E0_DESIGN_SEMANTIC_SHA256,
        "observed_environment_steps": 100_000,
        "environment_steps_per_second": 619.6620593242255,
        "training_wall_seconds": 161.37828433300456,
        "peak_rss_bytes": 4_253_122_560,
        "replay_buffer_allocation_bytes": 5_648_000_000,
    }
    for field, expected in checks.items():
        _require_exact(receipt.get(field), expected, field=f"E0 receipt.{field}")
    _require_exact(receipt.get("design"), calibration_design, field="E0 embedded design")
    _require_exact(
        _at(receipt, "runtime.runtime_sha256", field="E0 receipt"),
        E0_RUNTIME_SHA256,
        field="E0 runtime SHA-256",
    )
    _require_exact(
        _at(receipt, "observed_model.initial_parameters.structure_sha256", field="E0 receipt"),
        "0d9f1faecbf9dff3eadb350e12d2339aa76b853a07a08729347dc754beb41335",
        field="E0 observed parameter structure",
    )
    audit = loaded.e0_audit_to_dict()
    for row in audit["shared_projection"]["mapping_spec"]:
        e0_value = _at(calibration_design, row["e0_path"], field="E0 design")
        v2_value = _at(
            loaded.to_dict()["training_projection"],
            row["v2_training_projection_path"],
            field="v2 training projection",
        )
        _require_exact(v2_value, e0_value, field=row["v2_training_projection_path"])
    return ValidatedV2E0ReuseEvidence(
        e0_design_file_sha256=E0_DESIGN_FILE_SHA256,
        e0_receipt_file_sha256=E0_RECEIPT_FILE_SHA256,
        e0_runtime_sha256=E0_RUNTIME_SHA256,
        v1_design_file_sha256=V1_DESIGN_FILE_SHA256,
        training_projection_sha256=TRAINING_PROJECTION_SHA256,
        resource_prior_accepted=True,
        authorizes_training=False,
        _issuer=_E0_PREFLIGHT_ISSUER,
    )


__all__ = [
    "AUDIT_FILE_SHA256",
    "AUDIT_ID",
    "CAPTURE_INDICES",
    "DESIGN_FILE_SHA256",
    "DESIGN_ID",
    "EVALUATION_SEEDS",
    "EXPECTED_DISK_CHECKS",
    "EXPECTED_GRADIENT_UPDATES",
    "EXPECTED_PARAMETER_CHECKS",
    "EXPECTED_VECTOR_STEPS",
    "MODEL_SEED",
    "TOTAL_ENVIRONMENT_STEPS",
    "TRAINING_PROJECTION_SHA256",
    "VIDEO_SEEDS",
    "WORKER_SEEDS",
    "LoadedTQCDevelopmentDesignV2",
    "ValidatedV2E0ReuseEvidence",
    "load_tqc_development_design_v2",
    "validate_v2_e0_reuse_artifacts",
]
