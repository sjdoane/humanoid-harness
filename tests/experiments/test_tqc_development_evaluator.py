from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.envs.humanoid import HumanoidExperimentConfig, make_humanoid_env
from oracle_composition.experiments import tqc_development_evaluator as evaluator_module
from oracle_composition.experiments import tqc_development_metrics as metrics_module
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_actor_npz import (
    LoadedTQCActor,
    load_actor_npz,
    validate_actor_arrays,
    write_actor_npz_exclusive,
)
from oracle_composition.experiments.tqc_development_contract import (
    EVALUATION_SEEDS,
    LoadedTQCDevelopmentDesign,
    load_tqc_development_design,
)
from oracle_composition.experiments.tqc_development_evaluator import (
    CLAIM_BOUNDARY,
    ProtectedDeterministicTQCActor,
    TQCDevelopmentEvaluation,
    TQCDevelopmentRuntimeReceipt,
    evaluate_tqc_development_actor,
    inspect_tqc_development_runtime,
)
from oracle_composition.experiments.tqc_development_metrics import (
    CONTROL_PERIOD_SECONDS,
    EXPECTED_STEPS,
    TQCDevelopmentEpisodeAccumulator,
    TQCDevelopmentStepFacts,
    summarize_tqc_development_cohort,
)
from oracle_composition.traces.contract import EvidenceClass, TrajectoryTrace

ROOT = Path(__file__).parents[2]
DESIGN_PATH = ROOT / (
    "experiments/bootstrap_tqc_humanoid/configs/tqc_base_controller_dev_1m_v0.study.json"
)


def _zero_actor_arrays() -> dict[str, np.ndarray]:
    shapes = {
        "latent_pi.0.weight": (256, 348),
        "latent_pi.0.bias": (256,),
        "latent_pi.2.weight": (256, 256),
        "latent_pi.2.bias": (256,),
        "mu.weight": (17, 256),
        "mu.bias": (17,),
        "log_std.weight": (17, 256),
        "log_std.bias": (17,),
    }
    arrays = {name: np.zeros(shape, dtype="<f4") for name, shape in shapes.items()}
    arrays.update(
        {
            "action_low": np.full(17, -0.4, dtype="<f4"),
            "action_high": np.full(17, 0.4, dtype="<f4"),
            "format_version": np.asarray([1], dtype="<i8"),
        }
    )
    return validate_actor_arrays(arrays)


def _is_reviewed_host() -> bool:
    return (
        platform.system() == "Darwin"
        and platform.machine() == "arm64"
        and platform.release() == "25.6.0"
        and evaluator_module._host_hardware_receipt()
        == {
            "cpu_model": "Apple M5 Max",
            "hardware_model": "Mac17,6",
            "logical_cpu_count": 18,
            "total_memory_bytes": 38_654_705_664,
        }
    )


@pytest.fixture(scope="module")
def design() -> LoadedTQCDevelopmentDesign:
    return load_tqc_development_design(DESIGN_PATH)


@pytest.fixture(scope="module")
def loaded_zero_actor(tmp_path_factory: pytest.TempPathFactory) -> LoadedTQCActor:
    path = tmp_path_factory.mktemp("strict-tqc-actor") / "zero-actor.npz"
    digest = write_actor_npz_exclusive(path, _zero_actor_arrays())
    return load_actor_npz(path, expected_sha256=digest)


@pytest.fixture(scope="module")
def zero_actor(loaded_zero_actor: LoadedTQCActor) -> ProtectedDeterministicTQCActor:
    return ProtectedDeterministicTQCActor.from_strict_loaded_actor(loaded_zero_actor)


@pytest.fixture(scope="module")
def canonical_runtime(design: LoadedTQCDevelopmentDesign) -> TQCDevelopmentRuntimeReceipt:
    return inspect_tqc_development_runtime(design)


@pytest.fixture(scope="module")
def runtime() -> TQCDevelopmentRuntimeReceipt:
    """Portable test capability; never authority for a research result."""

    payload = {
        "runtime_id": "test_only_tqc_development_runtime/v1",
        "authority": "non_authoritative_test_fixture",
        "behavioral_claim_eligible": False,
    }
    encoded = evaluator_module.canonical_json(payload)
    return TQCDevelopmentRuntimeReceipt(
        canonical_bytes=encoded,
        sha256=hashlib.sha256(encoded).hexdigest(),
        _issuer=evaluator_module._RUNTIME_ISSUER,
    )


@pytest.fixture(scope="module")
def full_result(
    design: LoadedTQCDevelopmentDesign,
    zero_actor: ProtectedDeterministicTQCActor,
    runtime: TQCDevelopmentRuntimeReceipt,
) -> TQCDevelopmentEvaluation:
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            evaluator_module,
            "inspect_tqc_development_runtime",
            lambda _design: runtime,
        )
        return evaluate_tqc_development_actor(design, zero_actor)


@pytest.mark.gym
@pytest.mark.skipif(not _is_reviewed_host(), reason="requires the exact reviewed Mac runtime")
def test_runtime_is_exact_and_reward_fields_are_outside_its_receipt(
    canonical_runtime: TQCDevelopmentRuntimeReceipt,
) -> None:
    value = canonical_runtime.to_dict()

    assert value["runtime_id"] == "tqc_humanoid_development_evaluation_runtime/v1"
    assert value["environment_id"] == "Humanoid-v5"
    assert value["environment_max_episode_steps"] == EXPECTED_STEPS
    assert value["environment_kwargs"]["terminate_when_unhealthy"] is False
    assert value["control_period_seconds"] == CONTROL_PERIOD_SECONDS
    assert value["observation_shape"] == [348]
    assert value["action_shape"] == [17]
    assert value["qpos_shape"] == [24]
    assert value["qvel_shape"] == [23]
    assert value["contact_capture_id"] == "all_mujoco_substeps_after_mj_step/v1"
    assert len(value["sb3_contrib_common_utils_source_sha256"]) == 64
    assert len(value["sb3_base_vec_env_source_sha256"]) == 64
    assert value["reward_or_info_fields_read"] is False
    assert len(canonical_runtime.sha256) == 64


def test_portable_runtime_receipt_is_explicitly_non_authoritative(
    runtime: TQCDevelopmentRuntimeReceipt,
) -> None:
    assert runtime.to_dict() == {
        "authority": "non_authoritative_test_fixture",
        "behavioral_claim_eligible": False,
        "runtime_id": "test_only_tqc_development_runtime/v1",
    }


def test_strict_loaded_actor_snapshot_is_code_free_and_immutable(
    zero_actor: ProtectedDeterministicTQCActor,
    loaded_zero_actor: LoadedTQCActor,
) -> None:
    observation = np.linspace(-2.0, 2.0, 348, dtype=np.float64)
    action = zero_actor.act(observation)

    assert zero_actor.source_kind == "already_strict_loaded_actor/v1"
    assert zero_actor.persisted_artifact_sha256 == loaded_zero_actor.content_sha256
    assert zero_actor.persisted_artifact_byte_count == loaded_zero_actor.byte_count
    assert zero_actor.snapshot_state_sha256 == loaded_zero_actor.state_sha256
    assert np.array_equal(action.normalized, np.zeros(17, dtype="<f4"))
    assert np.array_equal(action.physical, np.zeros(17, dtype="<f4"))
    assert not action.normalized.flags.writeable
    assert not action.physical.flags.writeable
    with pytest.raises(ValueError):
        zero_actor.arrays["mu.bias"][0] = np.float32(1.0)


def test_trusted_in_memory_tqc_is_snapshotted_without_loading_files() -> None:
    import torch
    from sb3_contrib import TQC
    from stable_baselines3.common.torch_layers import FlattenExtractor

    environment = make_humanoid_env(HumanoidExperimentConfig())
    try:
        model = TQC(
            "MlpPolicy",
            environment,
            buffer_size=2,
            learning_starts=1,
            batch_size=1,
            policy_kwargs={
                "net_arch": [256, 256],
                "activation_fn": torch.nn.ReLU,
                "log_std_init": -3.0,
                "use_expln": False,
                "clip_mean": 2.0,
                "features_extractor_class": FlattenExtractor,
                "features_extractor_kwargs": None,
                "normalize_images": True,
                "n_quantiles": 25,
                "n_critics": 2,
                "share_features_extractor": False,
            },
            device="cpu",
            seed=95001,
            verbose=0,
        )
        observation, _info = environment.reset(seed=96001)
        expected_physical, _state = model.predict(observation, deterministic=True)
        protected = ProtectedDeterministicTQCActor.from_trusted_in_memory_model(model)
        observed = protected.act(observation)
        model.actor.net_arch = (256, 256)
        with pytest.raises(ExperimentContractError, match="actor feature architecture"):
            ProtectedDeterministicTQCActor.from_trusted_in_memory_model(model)
    finally:
        environment.close()

    assert protected.source_kind == "trusted_in_memory_sb3_contrib_tqc_snapshot/v1"
    assert protected.persisted_artifact_sha256 is None
    assert protected.persisted_artifact_byte_count is None
    assert protected.equivalence_observation_sha256 == (
        evaluator_module.EQUIVALENCE_OBSERVATION_SHA256
    )
    assert protected.trusted_normalized_action_sha256 == (
        protected.protected_normalized_action_sha256
    )
    assert protected.trusted_physical_action_sha256 == (protected.protected_physical_action_sha256)
    assert np.array_equal(observed.physical, expected_physical)


def _successful_episode(
    seed: int,
    *,
    forward_displacement_m: float = 7.5,
    elapsed_simulation_time_seconds: float | None = None,
):
    accumulator = TQCDevelopmentEpisodeAccumulator(
        seed=seed,
        initial_root_position_world_m=(0.0, 0.0, 1.4),
        initial_simulation_time_seconds=0.0,
    )
    for step in range(1, EXPECTED_STEPS + 1):
        simulation_time = (
            step * CONTROL_PERIOD_SECONDS
            if elapsed_simulation_time_seconds is None
            else elapsed_simulation_time_seconds * (step / EXPECTED_STEPS)
        )
        accumulator.add(
            TQCDevelopmentStepFacts(
                step_index=step,
                simulation_time_seconds=simulation_time,
                root_position_world_m=(
                    forward_displacement_m * step / EXPECTED_STEPS,
                    0.0,
                    1.4,
                ),
                torso_up_z=0.5,
                normalized_action=(0.0,) * 17,
                non_foot_floor_contact=False,
            )
        )
    return accumulator.finish()


def _replace_trace_signal(
    trace: TrajectoryTrace,
    *,
    sample_index: int,
    signal_name: str,
    values: tuple[float, ...],
) -> TrajectoryTrace:
    signal_index = tuple(signal.name for signal in trace.numeric_signals).index(signal_name)
    samples = list(trace.samples)
    numeric_values = list(samples[sample_index].numeric_values)
    numeric_values[signal_index] = values
    samples[sample_index] = replace(samples[sample_index], numeric_values=tuple(numeric_values))
    return replace(trace, samples=tuple(samples))


def test_pure_metric_core_uses_exact_thresholds_and_fixed_seed_order() -> None:
    episodes = tuple(_successful_episode(seed) for seed in EVALUATION_SEEDS)
    cohort = summarize_tqc_development_cohort(episodes, expected_seeds=EVALUATION_SEEDS)

    assert cohort.full_horizon_healthy_episode_count == 20
    assert cohort.full_horizon_upright_episode_count == 20
    assert cohort.median_time_average_forward_velocity_m_s == pytest.approx(0.5)
    assert cohort.episode_count_with_net_forward_displacement_at_least_5_m == 20
    assert cohort.development_behavior_gate_passed is True
    with pytest.raises(ExperimentContractError, match="missing, duplicated, or out of order"):
        summarize_tqc_development_cohort(
            tuple(reversed(episodes)),
            expected_seeds=EVALUATION_SEEDS,
        )


def test_direct_mujoco_elapsed_time_keeps_the_7_5m_boundary_below_threshold() -> None:
    observed_elapsed = 15.000000000000457
    episodes = tuple(
        _successful_episode(
            seed,
            forward_displacement_m=7.5,
            elapsed_simulation_time_seconds=observed_elapsed,
        )
        for seed in EVALUATION_SEEDS
    )
    cohort = summarize_tqc_development_cohort(episodes, expected_seeds=EVALUATION_SEEDS)

    assert episodes[0].final_simulation_time_seconds == observed_elapsed
    assert episodes[0].time_average_forward_velocity_m_s == 7.5 / observed_elapsed
    assert episodes[0].time_average_forward_velocity_m_s < 0.5
    assert cohort.median_time_average_forward_velocity_m_s < 0.5
    assert cohort.development_behavior_gate_passed is False


def test_pure_metric_boundaries_are_open_height_inclusive_upright_and_exact_saturation() -> None:
    accumulator = TQCDevelopmentEpisodeAccumulator(
        seed=96001,
        initial_root_position_world_m=(0.0, 0.0, 1.4),
        initial_simulation_time_seconds=0.0,
    )
    almost_one = float(np.nextafter(1.0, 0.0))
    for step in range(1, EXPECTED_STEPS + 1):
        action = [0.0] * 17
        if step == 1:
            action[0] = 1.0
        elif step == 2:
            action[0] = almost_one
        accumulator.add(
            TQCDevelopmentStepFacts(
                step_index=step,
                simulation_time_seconds=step * CONTROL_PERIOD_SECONDS,
                root_position_world_m=(0.0, 0.0, 1.0 if step == 1 else 1.4),
                torso_up_z=0.5 if step != 3 else float(np.nextafter(0.5, 0.0)),
                normalized_action=tuple(action),
                non_foot_floor_contact=step == 2,
            )
        )
    metrics = accumulator.finish()

    assert metrics.first_unhealthy_step == 1
    assert metrics.healthy_step_fraction == 0.999
    assert metrics.first_not_upright_step == 3
    assert metrics.upright_step_fraction == 0.999
    assert metrics.normalized_action_saturation_fraction == 1.0 / (EXPECTED_STEPS * 17)
    assert metrics.non_foot_floor_contact_step_fraction == 1.0 / EXPECTED_STEPS


@pytest.mark.parametrize(
    "facts, error",
    [
        (
            TQCDevelopmentStepFacts,
            "simulation_time_seconds must be a finite number",
        ),
        (None, "episode must contain exactly 1000 steps"),
    ],
)
def test_metric_core_fails_closed_on_nonfinite_or_incomplete_input(
    facts: object,
    error: str,
) -> None:
    accumulator = TQCDevelopmentEpisodeAccumulator(
        seed=96001,
        initial_root_position_world_m=(0.0, 0.0, 1.4),
        initial_simulation_time_seconds=0.0,
    )
    if facts is TQCDevelopmentStepFacts:
        with pytest.raises(ExperimentContractError, match=error):
            TQCDevelopmentStepFacts(
                step_index=1,
                simulation_time_seconds=float("nan"),
                root_position_world_m=(0.0, 0.0, 1.4),
                torso_up_z=1.0,
                normalized_action=(0.0,) * 17,
                non_foot_floor_contact=False,
            )
    else:
        with pytest.raises(ExperimentContractError, match=error):
            accumulator.finish()


def test_metric_core_revalidates_mutated_private_fact_storage() -> None:
    accumulator = TQCDevelopmentEpisodeAccumulator(
        seed=96001,
        initial_root_position_world_m=(0.0, 0.0, 1.4),
        initial_simulation_time_seconds=0.0,
    )
    repeated = TQCDevelopmentStepFacts(
        step_index=777,
        simulation_time_seconds=15.0,
        root_position_world_m=(7.5, 0.0, 1.4),
        torso_up_z=0.5,
        normalized_action=(0.0,) * 17,
        non_foot_floor_contact=False,
    )
    accumulator._facts[:] = [repeated] * EXPECTED_STEPS

    with pytest.raises(ExperimentContractError, match="not contiguous"):
        accumulator.finish()


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("_seed", 96002),
        ("_initial_root", (-7.5, 0.0, 1.4)),
        ("_initial_time", -15.0),
        ("_facts", []),
    ),
)
def test_metric_core_seals_constructor_state_and_fact_storage_reference(
    field: str,
    replacement: object,
) -> None:
    accumulator = TQCDevelopmentEpisodeAccumulator(
        seed=96001,
        initial_root_position_world_m=(0.0, 0.0, 1.4),
        initial_simulation_time_seconds=0.0,
    )

    with pytest.raises(AttributeError):
        setattr(accumulator, field, replacement)


def test_metric_core_rechecks_bound_constructor_state_before_issuance() -> None:
    accumulator = TQCDevelopmentEpisodeAccumulator(
        seed=96001,
        initial_root_position_world_m=(0.0, 0.0, 1.4),
        initial_simulation_time_seconds=0.0,
    )
    for step in range(1, EXPECTED_STEPS + 1):
        accumulator.add(
            TQCDevelopmentStepFacts(
                step_index=step,
                simulation_time_seconds=step * CONTROL_PERIOD_SECONDS,
                root_position_world_m=(0.0, 0.0, 1.4),
                torso_up_z=1.0,
                normalized_action=(0.0,) * 17,
                non_foot_floor_contact=False,
            )
        )
    object.__setattr__(accumulator, "_initial_root", (-7.5, 0.0, 1.4))

    with pytest.raises(ExperimentContractError, match="initial state binding"):
        accumulator.finish()


def test_one_real_humanoid_trace_is_complete_and_reward_info_independent(
    monkeypatch: pytest.MonkeyPatch,
    design: LoadedTQCDevelopmentDesign,
    runtime: TQCDevelopmentRuntimeReceipt,
    zero_actor: ProtectedDeterministicTQCActor,
) -> None:
    original_factory = make_humanoid_env

    def poisoned_reward_factory(*args: object, **kwargs: object):
        environment = original_factory(*args, **kwargs)
        original_step = environment.step

        def poisoned_step(action: object):
            observation, _reward, terminated, truncated, _info = original_step(action)
            return (
                observation,
                float("nan"),
                terminated,
                truncated,
                {"reward_forward": float("nan"), "reward_ctrl": object()},
            )

        environment.step = poisoned_step
        return environment

    monkeypatch.setattr(evaluator_module, "make_humanoid_env", poisoned_reward_factory)
    episode = evaluator_module._evaluate_seed(
        design=design,
        runtime=runtime,
        actor=zero_actor,
        seed=EVALUATION_SEEDS[0],
    )

    assert episode.metrics.observed_steps == EXPECTED_STEPS
    assert len(episode.trace.samples) == EXPECTED_STEPS + 1
    assert episode.controller_observation_shape == (EXPECTED_STEPS + 1, 348)
    assert len(episode.controller_observation_sha256) == 64
    assert episode.trace_sha256 == episode.trace.sha256
    assert episode.trace.evidence_class is EvidenceClass.EXPLORATORY
    assert episode.trace.events[-1].to_dict() == {
        "sample_index": EXPECTED_STEPS,
        "event_type": "episode.truncated",
        "value": "true",
        "source": "Gymnasium_TimeLimit_exact_step_1000/v1",
    }


def test_full_real_humanoid_evaluation_recomputes_metrics_from_each_trace(
    full_result: TQCDevelopmentEvaluation,
) -> None:
    result = full_result

    assert tuple(episode.seed for episode in result.episodes) == EVALUATION_SEEDS
    assert result.reward_or_info_fields_read is False
    assert result.checkpoint_or_episode_selection_performed is False
    assert result.trace_evidence_class is EvidenceClass.EXPLORATORY
    assert result.formal_experiment_eligible is False
    assert result.automatic_20m_authorization is False
    assert result.automatic_tracker_admission is False
    assert result.checkpoint_provenance_verified is False
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert result.cohort.development_behavior_gate_passed is False
    assert (
        result.actor_snapshot_state_sha256 == result.episodes[0].trace.artifact_bindings[3].sha256
    )
    assert result.persisted_actor_artifact_sha256 is not None
    assert result.persisted_actor_artifact_byte_count is not None

    required_missing = {
        "controller.motor_target",
        "controller.energy_j",
        "reference.frame",
        "reference.window_index",
        "oracle.mode",
        "oracle.phase",
        "oracle.transition_guard_margin",
        "recovery.disturbance",
        "recovery.rejoin_state",
    }
    for episode in result.episodes:
        trace = episode.trace
        assert len(trace.samples) == EXPECTED_STEPS + 1
        assert {signal.name for signal in trace.missing_signals} == required_missing
        assert all("reward" not in signal.name for signal in trace.numeric_signals)
        assert len(trace.sha256) == 64
        assert episode.trace_sha256 == trace.sha256
        assert episode.controller_observation_shape == (EXPECTED_STEPS + 1, 348)
        assert len(episode.controller_observation_sha256) == 64
        assert {binding.role for binding in trace.artifact_bindings} == {
            "runtime",
            "development_design",
            "evaluator",
            "actor_snapshot",
            "persisted_actor_artifact",
        }
        payload = trace.to_dict()
        samples = payload["samples"]
        contact_events = trace.events[:-1]
        assert all(
            event.event_type == evaluator_module.CONTACT_EVENT_TYPE for event in contact_events
        )
        assert all(
            event.source == evaluator_module.CONTACT_EVENT_SOURCE for event in contact_events
        )
        assert sum(
            int(value)
            for sample in samples
            for value in sample["signals"]["contact.active_count_by_substep"]
        ) == len(contact_events)
        contact_order = [
            (
                event.sample_index,
                json.loads(event.value)["physics_substep_index"],
                json.loads(event.value)["contact_index_within_substep"],
            )
            for event in contact_events
        ]
        assert contact_order == sorted(contact_order)
        observations = np.asarray(
            [sample["signals"]["controller.observation"] for sample in samples],
            dtype=np.float64,
        )
        assert observations.shape == (EXPECTED_STEPS + 1, 348)
        assert np.isfinite(observations).all()
        post_step = [sample["signals"] for sample in samples[1:]]
        roots = np.asarray(
            [sample["robot.root_position_world_m"] for sample in post_step],
            dtype=np.float64,
        )
        actions = np.asarray(
            [sample["controller.action"] for sample in post_step],
            dtype=np.float64,
        )
        simulation_times = np.asarray(
            [sample["signals"]["evaluation.simulation_time_seconds"][0] for sample in samples],
            dtype=np.float64,
        )
        progress = np.asarray(
            [sample["signals"]["task.progress"][0] for sample in samples],
            dtype=np.float64,
        )
        forbidden = np.asarray(
            [sample["evaluation.non_foot_floor_contact"][0] for sample in post_step],
            dtype=np.float64,
        )
        initial_root = np.asarray(
            samples[0]["signals"]["robot.root_position_world_m"], dtype=np.float64
        )
        metrics = episode.metrics
        assert metrics.root_height_min_m == float(np.min(roots[:, 2]))
        assert metrics.root_height_max_m == float(np.max(roots[:, 2]))
        assert metrics.net_forward_displacement_m == roots[-1, 0] - initial_root[0]
        assert metrics.initial_simulation_time_seconds == simulation_times[0]
        assert metrics.final_simulation_time_seconds == simulation_times[-1]
        assert metrics.elapsed_simulation_time_seconds == simulation_times[-1] - simulation_times[0]
        assert metrics.time_average_forward_velocity_m_s == (
            metrics.net_forward_displacement_m / metrics.elapsed_simulation_time_seconds
        )
        assert np.array_equal(progress, np.concatenate(([0.0], roots[:, 0] - initial_root[0])))
        assert metrics.root_lateral_displacement_max_abs_m == float(
            np.max(np.abs(roots[:, 1] - initial_root[1]))
        )
        assert metrics.normalized_action_rms == math.sqrt(float(np.mean(actions**2)))
        assert metrics.normalized_action_saturation_fraction == float(
            np.mean(np.abs(actions) == 1.0)
        )
        assert metrics.non_foot_floor_contact_step_fraction == float(np.mean(forbidden))


@pytest.mark.parametrize(
    ("signal_name", "error"),
    [
        ("robot.qpos", "qpos root"),
        ("robot.qvel", "observation differs"),
        ("controller.physical_control", "physical control"),
        ("controller.applied_torque", "observation differs"),
        ("task.progress", "task progress"),
        ("evaluation.torso_up_z", "root quaternion"),
    ],
)
def test_seed_receipt_rejects_redundant_trace_signal_tampering(
    full_result: TQCDevelopmentEvaluation,
    signal_name: str,
    error: str,
) -> None:
    episode = full_result.episodes[0]
    signal_index = tuple(signal.name for signal in episode.trace.numeric_signals).index(signal_name)
    values = list(episode.trace.samples[1].numeric_values[signal_index])
    values[0] += 0.125
    tampered = _replace_trace_signal(
        episode.trace,
        sample_index=1,
        signal_name=signal_name,
        values=tuple(values),
    )

    with pytest.raises(ExperimentContractError, match=error):
        replace(
            episode,
            trace=tampered,
            trace_sha256=tampered.sha256,
            _issuer=evaluator_module._SEED_EVALUATION_ISSUER,
        )


def test_seed_receipt_rejects_contact_event_and_terminal_event_tampering(
    full_result: TQCDevelopmentEvaluation,
) -> None:
    episode = full_result.episodes[0]
    contact_event_index = next(
        index
        for index, event in enumerate(episode.trace.events)
        if event.event_type == evaluator_module.CONTACT_EVENT_TYPE
    )
    contact_event = episode.trace.events[contact_event_index]
    contact_payload = json.loads(contact_event.value)
    contact_payload["geom1_name"] = "changed"
    events = list(episode.trace.events)
    events[contact_event_index] = replace(
        contact_event,
        value=evaluator_module.canonical_json(contact_payload).decode("utf-8"),
    )
    tampered_contact = replace(episode.trace, events=tuple(events))
    with pytest.raises(ExperimentContractError, match="geometry name"):
        replace(
            episode,
            trace=tampered_contact,
            trace_sha256=tampered_contact.sha256,
            _issuer=evaluator_module._SEED_EVALUATION_ISSUER,
        )

    missing_terminal = replace(episode.trace, events=episode.trace.events[:-1])
    with pytest.raises(ExperimentContractError, match="terminal event"):
        replace(
            episode,
            trace=missing_terminal,
            trace_sha256=missing_terminal.sha256,
            _issuer=evaluator_module._SEED_EVALUATION_ISSUER,
        )


def test_seed_receipt_rejects_trace_schema_drift(
    full_result: TQCDevelopmentEvaluation,
) -> None:
    episode = full_result.episodes[0]
    signals = list(episode.trace.numeric_signals)
    signals[0] = replace(signals[0], source="changed/v1")
    tampered = replace(episode.trace, numeric_signals=tuple(signals))

    with pytest.raises(ExperimentContractError, match="numeric schema"):
        replace(
            episode,
            trace=tampered,
            trace_sha256=tampered.sha256,
            _issuer=evaluator_module._SEED_EVALUATION_ISSUER,
        )


def test_evaluator_rejects_unprotected_actor_and_real_runtime_drift(
    monkeypatch: pytest.MonkeyPatch,
    design: LoadedTQCDevelopmentDesign,
) -> None:
    with pytest.raises(ExperimentContractError, match="protected TQC actor snapshot"):
        evaluate_tqc_development_actor(design, object())

    monkeypatch.setattr(evaluator_module, "space_sha256", lambda _space: "0" * 64)
    with pytest.raises(ExperimentContractError, match="runtime differs"):
        inspect_tqc_development_runtime(design)


def test_loaded_actor_and_protected_actor_wrappers_cannot_forge_identity(
    zero_actor: ProtectedDeterministicTQCActor,
    loaded_zero_actor: LoadedTQCActor,
) -> None:
    with pytest.raises(ExperimentContractError, match="only be issued by load_actor_npz"):
        replace(loaded_zero_actor, content_sha256="0" * 64)
    with pytest.raises(ExperimentContractError, match="only be issued by load_actor_npz"):
        replace(loaded_zero_actor, byte_count=loaded_zero_actor.byte_count + 1)
    with pytest.raises(ExperimentContractError, match="admission factory"):
        replace(zero_actor)


@pytest.mark.parametrize("field", ["file_sha256", "semantic_sha256"])
def test_loaded_design_wrapper_hashes_are_recomputed_from_exact_bytes(
    design: LoadedTQCDevelopmentDesign,
    field: str,
) -> None:
    forged = replace(design, **{field: "0" * 64})
    with pytest.raises(ExperimentContractError, match="exact bytes"):
        inspect_tqc_development_runtime(forged)


def test_semantically_equal_but_unreviewed_design_encoding_is_rejected(
    design: LoadedTQCDevelopmentDesign,
) -> None:
    reencoded = evaluator_module.canonical_json(design.to_dict())
    assert reencoded != design.encoded_bytes
    forged = replace(
        design,
        encoded_bytes=reencoded,
        file_sha256=hashlib.sha256(reencoded).hexdigest(),
    )
    with pytest.raises(ExperimentContractError, match="reviewed canonical artifact"):
        inspect_tqc_development_runtime(forged)


def test_public_receipt_constructors_and_inconsistent_algebra_fail_closed(
    runtime: TQCDevelopmentRuntimeReceipt,
    zero_actor: ProtectedDeterministicTQCActor,
    full_result: TQCDevelopmentEvaluation,
) -> None:
    episode = full_result.episodes[0]
    with pytest.raises(ExperimentContractError, match="issued by inspection"):
        replace(runtime)
    with pytest.raises(ExperimentContractError, match="admission factory"):
        replace(zero_actor)
    with pytest.raises(ExperimentContractError, match="protected metric core"):
        replace(episode.metrics)
    with pytest.raises(ExperimentContractError, match="protected evaluator"):
        replace(episode)
    with pytest.raises(ExperimentContractError, match="protected metric core"):
        replace(full_result.cohort)
    with pytest.raises(ExperimentContractError, match="protected evaluator"):
        replace(full_result)

    with pytest.raises(ExperimentContractError, match="velocity are inconsistent"):
        replace(
            episode.metrics,
            time_average_forward_velocity_m_s=(
                episode.metrics.time_average_forward_velocity_m_s + 1.0
            ),
            _issuer=metrics_module._METRIC_RECORD_ISSUER,
        )
    with pytest.raises(ExperimentContractError, match="differs from its episodes"):
        replace(
            full_result.cohort,
            full_horizon_healthy_episode_count=(
                20 - full_result.cohort.full_horizon_healthy_episode_count
            ),
            _issuer=metrics_module._METRIC_RECORD_ISSUER,
        )
    with pytest.raises(ExperimentContractError, match="canonical trace"):
        replace(
            episode,
            metrics=_successful_episode(episode.seed),
            _issuer=evaluator_module._SEED_EVALUATION_ISSUER,
        )
    with pytest.raises(ExperimentContractError, match="fixed seed order"):
        replace(
            full_result,
            episodes=tuple(reversed(full_result.episodes)),
            _issuer=evaluator_module._EVALUATION_ISSUER,
        )


def test_real_humanoid_rejects_non_boolean_termination_flags(
    monkeypatch: pytest.MonkeyPatch,
    design: LoadedTQCDevelopmentDesign,
    runtime: TQCDevelopmentRuntimeReceipt,
    zero_actor: ProtectedDeterministicTQCActor,
) -> None:
    original_factory = make_humanoid_env

    def non_boolean_factory(*args: object, **kwargs: object):
        environment = original_factory(*args, **kwargs)
        original_step = environment.step

        def non_boolean_step(action: object):
            observation, reward, terminated, truncated, info = original_step(action)
            return observation, reward, np.bool_(terminated), truncated, info

        environment.step = non_boolean_step
        return environment

    monkeypatch.setattr(evaluator_module, "make_humanoid_env", non_boolean_factory)
    with pytest.raises(ExperimentContractError, match="exact booleans"):
        evaluator_module._evaluate_seed(
            design=design,
            runtime=runtime,
            actor=zero_actor,
            seed=EVALUATION_SEEDS[0],
        )


def test_real_humanoid_rejects_environment_semantic_type_drift(
    monkeypatch: pytest.MonkeyPatch,
    design: LoadedTQCDevelopmentDesign,
    runtime: TQCDevelopmentRuntimeReceipt,
    zero_actor: ProtectedDeterministicTQCActor,
) -> None:
    original_factory = make_humanoid_env

    def type_drift_factory(*args: object, **kwargs: object):
        environment = original_factory(*args, **kwargs)
        environment.unwrapped._terminate_when_unhealthy = 0
        return environment

    monkeypatch.setattr(evaluator_module, "make_humanoid_env", type_drift_factory)
    with pytest.raises(ExperimentContractError, match="semantic types differ"):
        evaluator_module._evaluate_seed(
            design=design,
            runtime=runtime,
            actor=zero_actor,
            seed=EVALUATION_SEEDS[0],
        )


def test_real_humanoid_horizon_drift_fails_before_rollout(
    monkeypatch: pytest.MonkeyPatch,
    design: LoadedTQCDevelopmentDesign,
    runtime: TQCDevelopmentRuntimeReceipt,
    zero_actor: ProtectedDeterministicTQCActor,
) -> None:
    original_factory = make_humanoid_env

    def shortened_factory(*args: object, **kwargs: object):
        environment = original_factory(*args, **kwargs)
        environment._max_episode_steps = EXPECTED_STEPS - 1
        return environment

    monkeypatch.setattr(evaluator_module, "make_humanoid_env", shortened_factory)
    with pytest.raises(ExperimentContractError, match="must equal 1000"):
        evaluator_module._evaluate_seed(
            design=design,
            runtime=runtime,
            actor=zero_actor,
            seed=EVALUATION_SEEDS[0],
        )
