from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.envs.humanoid import CONTACT_CAPTURE_ID
from oracle_composition.experiments import tqc_development_contract_v2 as contract
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_development_contract_v2 import (
    CAPTURE_INDICES,
    EVALUATION_SEEDS,
    EXPECTED_DISK_CHECKS,
    EXPECTED_GRADIENT_UPDATES,
    EXPECTED_PARAMETER_CHECKS,
    EXPECTED_VECTOR_STEPS,
    LoadedTQCDevelopmentDesignV2,
    ValidatedV2E0ReuseEvidence,
    load_tqc_development_design_v2,
    validate_v2_e0_reuse_artifacts,
)

ROOT = Path(__file__).parents[2]
CONFIG_ROOT = ROOT / "experiments/bootstrap_tqc_humanoid/configs"
DESIGN_PATH = CONFIG_ROOT / "tqc_base_controller_dev_1m_v2.study.json"
AUDIT_PATH = CONFIG_ROOT / "tqc_e0_reuse_dev_1m_v2.audit.json"
E0_DESIGN_PATH = CONFIG_ROOT / "tqc_resource_calibration_v0.study.json"
E0_RECEIPT_PATH = ROOT / "artifacts/bootstrap_tqc_humanoid/resource_calibration_seed_92001.json"
V1_DESIGN_PATH = CONFIG_ROOT / "tqc_base_controller_dev_1m_v0.study.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _training() -> dict[str, object]:
    return copy.deepcopy(_read(DESIGN_PATH)["training_projection"])


def _wrapper_stack(environment: object) -> list[str]:
    result: list[str] = []
    current = environment
    while True:
        result.append(f"{type(current).__module__}.{type(current).__name__}")
        if current is current.unwrapped:
            return result
        current = current.env


def test_exact_v2_design_and_non_circular_audit_load() -> None:
    loaded = load_tqc_development_design_v2(DESIGN_PATH, AUDIT_PATH)
    design = loaded.to_dict()
    audit = loaded.e0_audit_to_dict()

    assert design["schema_version"] == 2
    assert design["design_id"].endswith("/v2")
    assert design["design_status"] == "predeclared_not_executed"
    assert design["formal_experiment_eligible"] is False
    assert audit["authorizes_training"] is False
    assert audit["v2_training_projection"] == design["training_projection"]
    assert "v2_design_file_sha256" not in audit
    assert "execution_manifest_sha256" not in audit
    assert loaded.file_sha256 == hashlib.sha256(DESIGN_PATH.read_bytes()).hexdigest()
    assert loaded.e0_audit_file_sha256 == hashlib.sha256(AUDIT_PATH.read_bytes()).hexdigest()


def test_v1_supersession_is_scoped_and_v1_bytes_are_unchanged() -> None:
    design = _read(DESIGN_PATH)
    supersession = design["supersession"]

    assert supersession["scope"] == "repository_authorized_future_executions_only"
    assert supersession["observed_v1_repository_evidence"] == (
        "no_reviewed_v1_runner_or_attempt_receipt"
    )
    assert supersession["universal_v1_execution_claim"] is False
    assert hashlib.sha256(V1_DESIGN_PATH.read_bytes()).hexdigest() == (
        "41c58694093070d7994aca5d1e8507832b40302fda088e4a1d42f26db7f5ca49"
    )


def test_temporal_status_fields_remain_truthful_after_future_implementation() -> None:
    design = _read(DESIGN_PATH)

    def walk(value: object) -> None:
        if isinstance(value, dict):
            assert "implementation_status" not in value
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(design)
    assert (
        design["training_projection"]["supervisor"]["implementation_status_at_predeclaration"]
        == "required_not_implemented"
    )
    assert design["evaluation"]["implementation_status_at_predeclaration"] == (
        "required_not_implemented_for_v2"
    )
    assert design["visual_evidence"]["implementation_status_at_predeclaration"] == (
        "required_not_implemented"
    )
    assert design["execution_manifest"]["implementation_status_at_predeclaration"] == (
        "required_not_implemented"
    )


def test_training_and_evaluation_environment_stacks_are_distinct() -> None:
    design = _read(DESIGN_PATH)
    training = design["training_projection"]
    evaluation = design["evaluation"]

    assert training["environment"]["wrapper_order_outer_to_inner"][-1] == (
        "gymnasium.envs.mujoco.humanoid_v5.HumanoidEnv"
    )
    assert training["simulator"]["contact_capture_id"] == ("none_in_plain_training_environment/v1")
    assert evaluation["wrapper_order_outer_to_inner"][-1] == (
        "oracle_composition.envs.humanoid.SubstepContactHumanoidEnv"
    )
    assert evaluation["contact_capture_id"] == CONTACT_CAPTURE_ID
    assert evaluation["capture_substep_contacts"] is True
    assert evaluation["training_stack_reuse_allowed"] is False
    assert evaluation["instrumentation_equivalence_gate"]["seed"] == 98001
    assert evaluation["environment_instances_per_seed"] == 2
    assert (
        evaluation["actor_driven_shadow_equivalence"]["required_for_each_evaluation_seed"] is True
    )


def test_instrumentation_equivalence_actions_are_exact_and_reproducible() -> None:
    gate = _read(DESIGN_PATH)["evaluation"]["instrumentation_equivalence_gate"]
    random = np.random.Generator(np.random.PCG64(98001))
    actions = np.ascontiguousarray(
        random.uniform(-0.4, 0.4, size=(1000, 17)),
        dtype="<f4",
    )
    header = contract.canonical_json({"dtype": actions.dtype.str, "shape": list(actions.shape)})
    raw = actions.tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)

    assert actions.shape == tuple(gate["action_shape"])
    assert digest.hexdigest() == gate["action_sha256"]


@pytest.mark.gym
def test_live_environment_constructors_match_both_frozen_stacks() -> None:
    import mujoco

    from oracle_composition.envs.humanoid import HumanoidExperimentConfig, make_humanoid_env

    design = _read(DESIGN_PATH)
    training = make_humanoid_env(HumanoidExperimentConfig(), capture_substep_contacts=False)
    evaluation = make_humanoid_env(HumanoidExperimentConfig(), capture_substep_contacts=True)
    try:
        assert (
            _wrapper_stack(training)
            == design["training_projection"]["environment"]["wrapper_order_outer_to_inner"]
        )
        assert _wrapper_stack(evaluation) == design["evaluation"]["wrapper_order_outer_to_inner"]
        assert evaluation.unwrapped.contact_capture_id == CONTACT_CAPTURE_ID
        option = training.unwrapped.model.opt
        declared = design["training_projection"]["simulator"]
        assert option.ccd_iterations == declared["ccd_iterations"]
        assert option.ccd_tolerance == declared["ccd_tolerance"]
        assert option.disableactuator == declared["disable_actuator"]
        assert option.o_friction.tolist() == declared["override_friction"]
        assert option.sleep_tolerance == declared["sleep_tolerance"]
        visual_option = mujoco.MjvOption()
        observed_visual_option = {
            "class": "mujoco.MjvOption",
            "flags": visual_option.flags.tolist(),
            "geomgroup": visual_option.geomgroup.tolist(),
            "sitegroup": visual_option.sitegroup.tolist(),
            "jointgroup": visual_option.jointgroup.tolist(),
            "tendongroup": visual_option.tendongroup.tolist(),
            "actuatorgroup": visual_option.actuatorgroup.tolist(),
            "flexgroup": visual_option.flexgroup.tolist(),
            "skingroup": visual_option.skingroup.tolist(),
            "frame": visual_option.frame,
            "label": visual_option.label,
            "bvh_depth": visual_option.bvh_depth,
            "flex_layer": visual_option.flex_layer,
        }
        assert observed_visual_option == design["visual_evidence"]["renderer"]["scene_option"]
    finally:
        training.close()
        evaluation.close()


def test_exact_budget_monitoring_supervision_and_reload_contract() -> None:
    training = _training()
    monitoring = training["monitoring"]
    supervisor = training["supervisor"]
    trusted = training["persistence"]["trusted_local_state"]

    assert training["tqc"]["total_environment_steps"] == 1_000_000
    assert training["counters"]["expected_vector_steps"] == EXPECTED_VECTOR_STEPS
    assert training["counters"]["expected_gradient_updates"] == EXPECTED_GRADIENT_UPDATES
    assert training["learn_call"]["log_interval"] == 1_000
    assert training["logger"]["learn_log_interval"] == 1_000
    assert monitoring["parameter_finite_schedule"]["expected_total_checks"] == (
        EXPECTED_PARAMETER_CHECKS
    )
    assert monitoring["disk_gate_schedule"]["expected_total_checks"] == EXPECTED_DISK_CHECKS
    assert monitoring["throughput_callback_sample_phase"] == (
        "after_vector_environment_step_before_replay_add_and_current_train_call/v1"
    )
    assert monitoring["training_scalar_callback_lag"] == {
        "callback_observes_completed_updates_from_previous_vector_call": True,
        "callback_call_21_num_timesteps": 105,
        "callback_call_21_observed_n_updates": 0,
        "callback_first_validation_call": 22,
        "callback_last_validation_call": 200000,
        "callback_validation_checks": 199979,
        "callback_first_expected_n_updates": 1,
        "callback_last_expected_n_updates": 199979,
        "post_learn_validation_checks": 1,
        "post_learn_expected_n_updates": 199980,
        "expected_total_update_scalar_validations": 199980,
    }
    assert supervisor["attempt_owner_process_role"] == "parent_supervisor"
    assert supervisor["worker_process_count"] == 1
    assert supervisor["worker_process_start_count"] == 1
    assert supervisor["worker_start_new_session"] is True
    assert supervisor["worker_child_process_creation_allowed"] is False
    assert supervisor["multiprocessing_start_method"] == "spawn"
    assert supervisor["parent_to_worker_message_types"] == [
        "admit_execution_manifest",
        "request_shutdown",
    ]
    assert supervisor["parent_worker_sequence_rule"].startswith(
        "independent_contiguous_integer_counter_per_direction"
    )
    assert supervisor["model_construction_requires_matching_manifest_ack"] is True
    assert supervisor["worker_started_timeout_origin"].endswith("Process_start")
    assert supervisor["parent_signal_shutdown_sequence"][2].startswith(
        "if_group_identity_valid_send_SIGTERM"
    )
    assert supervisor["parent_signal_shutdown_sequence"][4].startswith(
        "if_group_identity_valid_send_SIGKILL"
    )
    assert supervisor["allowed_state_order"][-3:] == [
        "evaluation",
        "visual_encoding",
        "finalized",
    ]
    assert trusted["strict_model_load_call"]["env_argument"] is None
    assert trusted["strict_model_load_call"]["force_reset"] is True
    assert trusted["strict_replay_load_call"]["truncate_last_traj"] is False
    assert trusted["simultaneous_full_replay_buffers_allowed"] == 1
    assert len(trusted["strict_reload_hash_pairs"]) == 7
    assert trusted["strict_reload_byte_exact_state_hash_equality_required"] is True
    assert "decoupled_weight_decay" in trusted["optimizer_param_group_fields"]
    assert trusted["replay_metadata_hash_fields"][-2:] == [
        "observation_space_sha256",
        "action_space_sha256",
    ]
    assert monitoring["expected_lifecycle_resource_samples"] == 16
    assert monitoring["lifecycle_resource_samples"][-2:] == [
        "pre_finalization",
        "post_finalization",
    ]
    assert supervisor["stage_timeouts_seconds"]["training"] == 11100.0
    assert supervisor["training_worker_wall_gate_seconds"] == 10800.0


@pytest.mark.parametrize(
    "mutation, error",
    [
        (lambda value: value["learn_call"].update({"log_interval": 4}), "learn log interval"),
        (
            lambda value: value["monitoring"]["parameter_finite_schedule"].update(
                {"vector_step_checks": 199}
            ),
            "parameter check count",
        ),
        (
            lambda value: value["monitoring"]["optimizer_state_finite_schedule"].update(
                {"expected_state_tensor_count_after_first_update": 42}
            ),
            "optimizer state tensor count",
        ),
        (
            lambda value: value["monitoring"]["stage_finite_schedule"].update(
                {"expected_total_checks": 28}
            ),
            "stage finite checks",
        ),
        (
            lambda value: value["monitoring"]["disk_gate_schedule"].update(
                {"vector_step_checks": 1999}
            ),
            "disk check count",
        ),
        (
            lambda value: value["monitoring"]["training_scalar_callback_lag"].update(
                {"callback_first_validation_call": 21}
            ),
            "first scalar check",
        ),
        (
            lambda value: value["monitoring"].update(
                {"throughput_callback_sample_phase": "after_train"}
            ),
            "throughput sample phase",
        ),
        (
            lambda value: value["runtime_requirements"].update(
                {"torch_optimizer_base_source_sha256": "0" * 64}
            ),
            "Torch optimizer source",
        ),
        (
            lambda value: value["adam_optimizers"]["model_construction_resolution"].update(
                {"required_observed_actor_foreach": True}
            ),
            "actor resolved foreach",
        ),
        (
            lambda value: value["supervisor"].update({"run_directory_mode_octal": "0755"}),
            "run-directory mode",
        ),
        (
            lambda value: value["supervisor"].update({"worker_process_count": 2}),
            "worker process count",
        ),
        (
            lambda value: value["supervisor"].update({"multiprocessing_start_method": "fork"}),
            "worker start method",
        ),
        (
            lambda value: value["supervisor"].update(
                {"parent_worker_sequence_rule": "ambiguous_global_counter"}
            ),
            "worker message sequence",
        ),
        (
            lambda value: value["supervisor"].update(
                {"model_construction_requires_matching_manifest_ack": False}
            ),
            "manifest acknowledgement",
        ),
        (
            lambda value: value["supervisor"].update(
                {"all_parent_termination_paths_use_bounded_cleanup": False}
            ),
            "bounded parent cleanup",
        ),
        (
            lambda value: value["supervisor"]["parent_signal_shutdown_sequence"].__setitem__(
                2, "send_SIGTERM_to_unvalidated_process_group"
            ),
            "signal cleanup",
        ),
        (
            lambda value: value["supervisor"]["allowed_state_order"].remove("visual_encoding"),
            "supervisor state order",
        ),
        (
            lambda value: value["supervisor"]["stage_timeouts_seconds"].update(
                {"training": 10800.0}
            ),
            "parent training timeout",
        ),
        (
            lambda value: value["persistence"]["trusted_local_state"].update(
                {"replay_partial_artifact_eligible": True}
            ),
            "partial replay eligibility",
        ),
        (
            lambda value: value["persistence"]["trusted_local_state"].update(
                {"strict_reload_byte_exact_state_hash_equality_required": False}
            ),
            "strict reload state equality",
        ),
        (
            lambda value: value["persistence"]["trusted_local_state"].update(
                {"replay_metadata_hash_fields": ["position", "full"]}
            ),
            "replay metadata hash fields",
        ),
        (
            lambda value: value["persistence"].update(
                {"execution_manifest_binding_required": False}
            ),
            "persistence manifest binding",
        ),
        (
            lambda value: value["persistence"]["trusted_local_state"][
                "strict_model_load_call"
            ].update({"env_argument": "Humanoid-v5"}),
            "model load environment",
        ),
        (
            lambda value: value["persistence"]["actor_export"].update(
                {"log_std_rule": "unclamped"}
            ),
            "actor log_std rule",
        ),
        (
            lambda value: value["persistence"]["actor_export"].update(
                {"all_eight_equivalence_hashes_use_one_array_rule": False}
            ),
            "actor equivalence hash coverage",
        ),
        (
            lambda value: value["persistence"]["actor_export"].update(
                {"actor_state_hash_equality_required": False}
            ),
            "actor state hash equality",
        ),
    ],
)
def test_training_contract_relationships_fail_closed(mutation: object, error: str) -> None:
    training = _training()
    mutation(training)

    with pytest.raises(ExperimentContractError, match=error):
        contract._validate_training_projection(training)


def test_parameter_schema_covers_all_state_and_optimizer_tensors() -> None:
    policy = _training()["tqc_policy"]
    schema = policy["ordered_parameter_schema"]

    assert len(schema) == 33
    assert [row["role"] for row in schema].count("actor") == 8
    assert [row["role"] for row in schema].count("critic") == 12
    assert [row["role"] for row in schema].count("critic_target") == 12
    assert [row["role"] for row in schema].count("entropy_coefficient") == 1
    assert sum(row["optimizer_owned"] for row in schema) == 21
    assert policy["expected_finite_state_scalar_count"] == 827_527
    assert policy["expected_optimizer_owned_scalar_count"] == 495_701


@pytest.mark.gym
def test_real_tqc_construction_matches_parameter_and_optimizer_contract() -> None:
    pytest.importorskip("sb3_contrib")
    torch = pytest.importorskip("torch")
    from sb3_contrib import TQC
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.torch_layers import FlattenExtractor
    from torch.optim.optimizer import _default_to_fused_or_foreach

    from oracle_composition.envs.humanoid import HumanoidExperimentConfig, make_humanoid_env

    training = _training()
    policy = training["tqc_policy"]
    environment = make_humanoid_env(HumanoidExperimentConfig(), capture_substep_contacts=False)
    try:
        model = TQC(
            "MlpPolicy",
            environment,
            learning_rate=0.0003,
            buffer_size=2,
            learning_starts=100,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            action_noise=None,
            replay_buffer_class=ReplayBuffer,
            replay_buffer_kwargs={"handle_timeout_termination": True},
            optimize_memory_usage=False,
            n_steps=1,
            ent_coef="auto",
            target_update_interval=1,
            target_entropy="auto",
            top_quantiles_to_drop_per_net=2,
            use_sde=False,
            sde_sample_freq=-1,
            use_sde_at_warmup=False,
            stats_window_size=100,
            tensorboard_log=None,
            policy_kwargs={
                "net_arch": [256, 256],
                "activation_fn": torch.nn.ReLU,
                "use_sde": False,
                "log_std_init": -3.0,
                "use_expln": False,
                "clip_mean": 2.0,
                "features_extractor_class": FlattenExtractor,
                "features_extractor_kwargs": None,
                "normalize_images": True,
                "optimizer_class": torch.optim.Adam,
                "optimizer_kwargs": {"eps": 0.00001},
                "n_quantiles": 25,
                "n_critics": 2,
                "share_features_extractor": False,
            },
            verbose=0,
            seed=95001,
            device="cpu",
            _init_setup_model=True,
        )
        observed = [
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": "<f4" if parameter.dtype is torch.float32 else str(parameter.dtype),
            }
            for name, parameter in model.policy.named_parameters()
        ]
        observed.append(
            {
                "name": "algorithm.log_ent_coef",
                "shape": list(model.log_ent_coef.shape),
                "dtype": "<f4",
            }
        )
        assert observed == [
            {"name": row["name"], "shape": row["shape"], "dtype": row["dtype"]}
            for row in policy["ordered_parameter_schema"]
        ]
        optimizers = training["adam_optimizers"]
        for observed_defaults, declared in (
            (model.actor.optimizer.defaults, optimizers["actor"]),
            (model.critic.optimizer.defaults, optimizers["critic"]),
            (model.ent_coef_optimizer.defaults, optimizers["entropy_coefficient"]),
        ):
            assert observed_defaults["lr"] == declared["lr"]
            assert list(observed_defaults["betas"]) == declared["betas"]
            for key in (
                "eps",
                "weight_decay",
                "amsgrad",
                "foreach",
                "maximize",
                "capturable",
                "differentiable",
                "fused",
                "decoupled_weight_decay",
            ):
                assert observed_defaults[key] == declared[key]
        assert model.actor.full_std is True
        assert model.actor.use_expln is False
        assert model.actor.clip_mean == 2.0
        assert type(model.actor.action_dist).__name__ == "SquashedDiagGaussianDistribution"
        resolution = training["adam_optimizers"]["model_construction_resolution"]
        for role, optimizer in (
            ("actor", model.actor.optimizer),
            ("critic", model.critic.optimizer),
            ("entropy", model.ent_coef_optimizer),
        ):
            parameters = [item for group in optimizer.param_groups for item in group["params"]]
            fused, foreach = _default_to_fused_or_foreach(
                parameters,
                differentiable=False,
                use_fused=False,
            )
            assert fused is resolution[f"required_observed_{role}_fused"]
            assert foreach is resolution[f"required_observed_{role}_foreach"]
            assert optimizer.state == {}
        assert torch.get_default_dtype() is torch.float32
        assert torch.are_deterministic_algorithms_enabled() is False
    finally:
        environment.close()


def test_visual_contract_is_same_rollout_lossless_and_noninterfering() -> None:
    visual = _read(DESIGN_PATH)["visual_evidence"]
    encoding = visual["encoding"]
    decode = visual["decode_verification"]

    assert tuple(visual["seeds"]) == (96001, 96010, 96020)
    assert tuple(range(0, 1001, 2)) == CAPTURE_INDICES
    assert visual["expected_frame_count"] == 501
    assert visual["capture_phase"].endswith("terminal_boundary_has_no_next_inference")
    assert visual["source"] == "same_environment_same_seed_same_evaluator_loop_no_replay"
    assert [
        visual[key] for key in ("rerun_allowed", "replay_allowed", "replacement_rollout_allowed")
    ] == [False] * 3
    assert visual["frame_shape"] == [480, 480, 3]
    assert encoding["duration_ms_per_frame"] == 30
    assert encoding["encoded_duration_ms"] == 15030
    assert encoding["terminal_hold_frame_count"] == 0
    assert encoding["background_rgba"] == [0, 0, 0, 0]
    assert decode["frame_load_before_duration_read_required"] is True
    assert decode["exact_pixel_equality_to_each_raw_frame"] is True
    assert decode["collapsed_identical_frames_allowed"] is False
    assert decode["adjacent_raw_frame_hashes_must_all_differ"] is True
    assert decode["adjacent_identical_frame_disposition"] == (
        "visual_artifact_ineligible_codec_constraint_not_behavior_gate_failure_no_rerun"
    )
    assert encoding["final_boundary_standard_frame_duration_ms"] == 30
    assert encoding["additional_terminal_hold"] is False
    assert visual["renderer"]["max_geom"] == 10000
    assert _read(DESIGN_PATH)["training_projection"]["environment"]["mujoco_env_max_geom"] == 1000
    assert visual["hash_canonicalization"]["array_rule"].startswith("sha256_uint64be")


@pytest.mark.parametrize(
    "mutation, error",
    [
        (lambda value: value.update({"expected_frame_count": 251}), "frame count"),
        (lambda value: value["encoding"].update({"terminal_hold_frame_count": 1}), "terminal hold"),
        (lambda value: value["encoding"].update({"background_rgba": [0, 0, 0, 255]}), "background"),
        (lambda value: value["renderer"].update({"max_geom": 1000}), "max_geom"),
        (
            lambda value: value.update({"capture_phase": "before_next_action_at_all_boundaries"}),
            "capture phase",
        ),
        (
            lambda value: value["renderer"]["scene_option"]["flags"].__setitem__(1, 0),
            "scene option SHA-256",
        ),
        (
            lambda value: value["decode_verification"].update(
                {"frame_load_before_duration_read_required": False}
            ),
            "frame load",
        ),
        (
            lambda value: value["hash_canonicalization"].update({"array_rule": "raw_only"}),
            "array hash rule",
        ),
        (
            lambda value: value["per_frame_record_fields"].pop(),
            "camera record field order",
        ),
        (
            lambda value: value["receipt_must_bind"].remove("observed_decoded_frame_count"),
            "visual receipt omits",
        ),
        (
            lambda value: value["decode_verification"].update(
                {"adjacent_raw_frame_hashes_must_all_differ": False}
            ),
            "adjacent frame hashes",
        ),
        (
            lambda value: value["decode_verification"].update(
                {"adjacent_identical_frame_disposition": "behavior_gate_failure"}
            ),
            "adjacent frame disposition",
        ),
    ],
)
def test_visual_contract_relationships_fail_closed(mutation: object, error: str) -> None:
    design = _read(DESIGN_PATH)
    mutation(design["visual_evidence"])

    with pytest.raises(ExperimentContractError, match=error):
        contract._validate_visual(design)


@pytest.mark.skipif(not E0_RECEIPT_PATH.exists(), reason="local exact E0 receipt is untracked")
def test_e0_audit_revalidates_real_resource_only_artifacts() -> None:
    loaded = load_tqc_development_design_v2(DESIGN_PATH, AUDIT_PATH)
    result = validate_v2_e0_reuse_artifacts(
        loaded,
        calibration_design_path=E0_DESIGN_PATH,
        calibration_receipt_path=E0_RECEIPT_PATH,
        superseded_v1_design_path=V1_DESIGN_PATH,
    )

    assert result.resource_prior_accepted is True
    assert result.authorizes_training is False
    assert result.training_projection_sha256 == contract.TRAINING_PROJECTION_SHA256


def test_e0_reuse_validation_is_portable_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = load_tqc_development_design_v2(DESIGN_PATH, AUDIT_PATH)
    e0_design = _read(E0_DESIGN_PATH)
    receipt = {
        "completion_status": "complete",
        "calibration_gate_passed": True,
        "measured_workload_gates_passed": True,
        "checkpoint_emitted": False,
        "replay_buffer_emitted": False,
        "normalizer_emitted": False,
        "controller_artifact": None,
        "eligible_for_controller_training": False,
        "eligible_for_behavioral_evaluation": False,
        "design_artifact_sha256": contract.E0_DESIGN_FILE_SHA256,
        "design_semantic_sha256": contract.E0_DESIGN_SEMANTIC_SHA256,
        "observed_environment_steps": 100000,
        "environment_steps_per_second": 619.6620593242255,
        "training_wall_seconds": 161.37828433300456,
        "peak_rss_bytes": 4253122560,
        "replay_buffer_allocation_bytes": 5648000000,
        "runtime": {"runtime_sha256": contract.E0_RUNTIME_SHA256},
        "observed_model": {
            "initial_parameters": {
                "structure_sha256": (
                    "0d9f1faecbf9dff3eadb350e12d2339aa76b853a07a08729347dc754beb41335"
                )
            }
        },
        "design": e0_design,
    }
    receipt_path = tmp_path / "synthetic-e0-receipt.json"
    receipt_path.write_text(json.dumps(receipt, allow_nan=False), encoding="utf-8")
    monkeypatch.setattr(
        contract,
        "E0_RECEIPT_FILE_SHA256",
        hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        contract,
        "E0_RECEIPT_SEMANTIC_SHA256",
        contract._semantic_sha256(receipt),
    )

    accepted = validate_v2_e0_reuse_artifacts(
        loaded,
        calibration_design_path=E0_DESIGN_PATH,
        calibration_receipt_path=receipt_path,
        superseded_v1_design_path=V1_DESIGN_PATH,
    )
    assert accepted.resource_prior_accepted is True
    assert accepted.authorizes_training is False

    receipt["eligible_for_controller_training"] = True
    receipt_path.write_text(json.dumps(receipt, allow_nan=False), encoding="utf-8")
    monkeypatch.setattr(
        contract,
        "E0_RECEIPT_FILE_SHA256",
        hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        contract,
        "E0_RECEIPT_SEMANTIC_SHA256",
        contract._semantic_sha256(receipt),
    )
    with pytest.raises(ExperimentContractError, match="eligible_for_controller_training"):
        validate_v2_e0_reuse_artifacts(
            loaded,
            calibration_design_path=E0_DESIGN_PATH,
            calibration_receipt_path=receipt_path,
            superseded_v1_design_path=V1_DESIGN_PATH,
        )


def test_loader_issued_records_cannot_be_publicly_forged() -> None:
    with pytest.raises(ExperimentContractError, match="strict loader"):
        LoadedTQCDevelopmentDesignV2(
            encoded_bytes=DESIGN_PATH.read_bytes(),
            file_sha256=contract.DESIGN_FILE_SHA256,
            semantic_sha256=contract.DESIGN_SEMANTIC_SHA256,
            e0_audit_encoded_bytes=AUDIT_PATH.read_bytes(),
            e0_audit_file_sha256=contract.AUDIT_FILE_SHA256,
            e0_audit_semantic_sha256=contract.AUDIT_SEMANTIC_SHA256,
            training_projection_sha256=contract.TRAINING_PROJECTION_SHA256,
        )
    with pytest.raises(ExperimentContractError, match="issued by validation"):
        ValidatedV2E0ReuseEvidence(
            e0_design_file_sha256="0" * 64,
            e0_receipt_file_sha256="0" * 64,
            e0_runtime_sha256="0" * 64,
            v1_design_file_sha256="0" * 64,
            training_projection_sha256="0" * 64,
            resource_prior_accepted=True,
            authorizes_training=False,
        )


def test_loader_rejects_changed_duplicate_nonfinite_and_oversized_artifacts(tmp_path: Path) -> None:
    changed = tmp_path / "changed.json"
    changed.write_text(DESIGN_PATH.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ExperimentContractError, match="file SHA-256 differs"):
        load_tqc_development_design_v2(changed, AUDIT_PATH)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":2,"schema_version":2}', encoding="utf-8")
    with pytest.raises(ExperimentContractError, match="duplicate key"):
        load_tqc_development_design_v2(duplicate, AUDIT_PATH)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"schema_version":NaN}', encoding="utf-8")
    with pytest.raises(ExperimentContractError, match="non-finite"):
        load_tqc_development_design_v2(nonfinite, AUDIT_PATH)

    oversized = tmp_path / "oversized.json"
    oversized.write_bytes(b" " * (contract.MAX_DESIGN_BYTES + 1))
    with pytest.raises(ExperimentContractError, match="bounded size limit"):
        load_tqc_development_design_v2(oversized, AUDIT_PATH)


def test_audit_projection_and_authority_mutations_fail_closed() -> None:
    design = _read(DESIGN_PATH)
    audit = _read(AUDIT_PATH)
    audit["authorizes_training"] = True
    with pytest.raises(ExperimentContractError, match="audit training authority"):
        contract._validate_design_and_audit(design, audit)

    audit = _read(AUDIT_PATH)
    audit["v2_training_projection"]["tqc"]["batch_size"] = 128
    with pytest.raises(ExperimentContractError, match="full v2 training projection"):
        contract._validate_design_and_audit(design, audit)

    audit = _read(AUDIT_PATH)
    audit["reuse_decision"]["execution_manifest_sha256"] = "0" * 64
    with pytest.raises(ExperimentContractError, match="forbidden future-artifact"):
        contract._validate_design_and_audit(design, audit)

    audit = _read(AUDIT_PATH)
    audit["declared_protocol_additions"][0]["mapped_training_workload_value_change_allowed"] = True
    with pytest.raises(ExperimentContractError, match="mapped-value authority"):
        contract._validate_design_and_audit(design, audit)


def test_independent_review_receipt_contract_is_complete_and_fails_closed() -> None:
    design = _read(DESIGN_PATH)
    audit = _read(AUDIT_PATH)
    review = design["execution_manifest"]["independent_review_receipt_contract"]

    assert review["verdict_required"] == "GO"
    assert review["unresolved_counts_required"] == {"p0": 0, "p1": 0, "p2": 0}
    assert review["completed_before_preflight_contract"] is True
    assert review["no_placeholder_hash_in_design"] is True
    assert len(review["required_reviewed_hash_bindings"]) == 6
    assert review["reviewed_source_paths"]["reviewed_v2_contract_test_file_sha256"] == (
        "tests/experiments/test_tqc_development_contract_v2.py"
    )

    changed = copy.deepcopy(design)
    changed["execution_manifest"]["independent_review_receipt_contract"]["verdict_required"] = (
        "NO-GO"
    )
    with pytest.raises(ExperimentContractError, match="review verdict"):
        contract._validate_design_and_audit(changed, audit)

    changed = copy.deepcopy(design)
    del changed["execution_manifest"]["independent_review_receipt_contract"]["trust_boundary"]
    with pytest.raises(ExperimentContractError, match="review receipt contract keys"):
        contract._validate_design_and_audit(changed, audit)


def test_fixed_evaluation_has_exact_one_checkpoint_seed_order_and_claim_ceiling() -> None:
    design = _read(DESIGN_PATH)
    evaluation = design["evaluation"]
    decision = design["decision_rule"]

    assert tuple(evaluation["seed_order"]) == EVALUATION_SEEDS
    assert evaluation["episodes_per_seed"] == 1
    assert evaluation["step_count_per_seed"] == 1000
    assert evaluation["episode_or_checkpoint_selection"] is False
    assert evaluation["retry_or_replay_for_metrics"] is False
    assert evaluation["reward_or_info_reward_fields_used_for_metrics"] is False
    trace = evaluation["canonical_trace"]
    assert trace["sample_count"] == 1001
    assert len(trace["numeric_signals"]) == 17
    assert "task.progress" in [signal["name"] for signal in trace["numeric_signals"]]
    assert len(trace["explicit_missing_signals"]) == 9
    assert trace["metrics_elapsed_time_source"].startswith(
        "evaluation.simulation_time_seconds_direct"
    )
    assert "k_minus_1" in trace["action_observation_alignment"]
    assert "execution_manifest" in trace["artifact_bindings"]
    assert decision["automatic_20m_authorization"] is False
    assert decision["automatic_tracker_admission"] is False
    assert design["claim_ceiling"] == (
        "one_checkpoint_one_training_seed_twenty_fixed_reset_development_behavior"
    )
