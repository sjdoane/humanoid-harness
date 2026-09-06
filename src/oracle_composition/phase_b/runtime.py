"""Real and controlled-fake environment adapters for the Phase B worker."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

import gymnasium as gym
import numpy as np

from oracle_composition.contracts.reference_identity_v2 import (
    array_sha256,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)
from oracle_composition.envs.reference_corpus import make_reference_corpus_env
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.reference_corpus_bundle import (
    load_bundle_manifest,
    verify_bound_artifacts,
)
from oracle_composition.experiments.reference_corpus_collector import (
    capture_wrapper_state,
    restore_rng_state,
    restore_wrapper_state,
)
from oracle_composition.experiments.reference_corpus_contract import decode_clip_payload
from oracle_composition.harness.contract import ALLOWED_SIGNALS
from oracle_composition.harness.inputs import load_frozen_inputs
from oracle_composition.rewards.stock_humanoid import (
    body_mass_weighted_com_x_velocity_m_s,
)
from oracle_composition.tracking.humanoid_reference import (
    HumanoidTrackingState,
    tracking_state,
    tracking_state_with_bounded_reset_orientation,
    validate_humanoid_actuator_abi,
)

from .contracts import RewardRegistry, load_phase_b_oracle
from .isolation import SealedArtifact, verify_one_sealed_input, verify_sealed_inputs
from .policy import (
    ACTION_WIDTH,
    OBSERVATION_WIDTH,
    POLICY_INPUT_WIDTH,
    REFERENCE_HORIZON,
    REFERENCE_WIDTH,
    FullAuthorityActor,
    FullAuthorityPolicy,
    compose_policy_input,
    load_full_authority_actor,
)
from .reference_runtime import ComposedReferenceRuntime, load_v2_reference_clip
from .reward import compose_registered_reward
from .task_input_admission import MEASUREMENT_ORIGIN, admit_task_inputs_v2
from .training import (
    BEHAVIORS,
    STREAM_BY_ENVIRONMENT,
    BalancedRSIScheduler,
    PhaseBTrainingError,
    RSIAssignment,
    RSIRestorationReceipt,
    TrainingFailureStatus,
    TrainingPlan,
    domain_separated_seed,
    plan_domain_separated_seed,
    validate_rsi_restoration_receipt,
)


@dataclass(frozen=True, slots=True)
class RealRuntimeConfig:
    repository_root: Path
    experiment: Path
    oracle_path: Path
    reward_path: Path
    starting_actor_path: Path
    starting_actor_sha256: str
    value_seed: int
    sealed_inputs: tuple[SealedArtifact, ...] = ()
    sealed_input_lineage_sha256: str = ""


def _verify_runtime_input(config: RealRuntimeConfig, path: Path) -> None:
    verify_one_sealed_input(config.repository_root, config.sealed_inputs, path)


def _verify_clip_lineage(config: RealRuntimeConfig, *, block: int, behavior: str) -> None:
    corpus_root = config.repository_root / "artifacts/reference_corpus_v2"
    index_path = corpus_root / "corpus_index_v2.json"
    _verify_runtime_input(config, index_path)
    index = _canonical_object(index_path)
    clip_id = f"corpus-{block}-{behavior}"
    matches = [entry for entry in index.get("clips", ()) if entry.get("clip_id") == clip_id]
    if len(matches) != 1:
        raise ExperimentContractError(f"sealed corpus index omits {clip_id}")
    bundle_sha256 = matches[0].get("bundle_manifest_sha256")
    if type(bundle_sha256) is not str:
        raise ExperimentContractError("sealed corpus bundle digest is malformed")
    bundle_path = corpus_root / "clips" / clip_id / f"bundle-{bundle_sha256}.json"
    _verify_runtime_input(config, bundle_path)
    manifest = load_bundle_manifest(bundle_path, expected_sha256=bundle_sha256)
    core = manifest["core"]
    bindings = [*core["bound_artifacts"], core["payload"], core["rng_state"]["binding"]]
    for binding in bindings:
        _verify_runtime_input(config, corpus_root / str(binding["object_path"]))


@dataclass(frozen=True, slots=True)
class _RSIMaterial:
    block: int
    behavior: str
    bundle_sha256: str
    arrays: Mapping[str, np.ndarray]
    rng_state: Mapping[str, object]


def _canonical_object(path: Path) -> dict[str, object]:
    encoded = Path(path).read_bytes()
    value = json.loads(encoded.decode("utf-8"))
    if type(value) is not dict or canonical_json_bytes(value) != encoded:
        raise ValueError(f"artifact is not one canonical JSON object: {path}")
    return value


def _load_rsi_material(corpus_root: Path, *, block: int, behavior: str) -> _RSIMaterial:
    root = Path(corpus_root)
    index = _canonical_object(root / "corpus_index_v2.json")
    clip_id = f"corpus-{block}-{behavior}"
    matches = [entry for entry in index.get("clips", ()) if entry.get("clip_id") == clip_id]
    if len(matches) != 1:
        raise ValueError(f"corpus index does not bind one RSI clip: {clip_id}")
    bundle_sha256 = matches[0]["bundle_manifest_sha256"]
    bundle_path = root / "clips" / clip_id / f"bundle-{bundle_sha256}.json"
    manifest = load_bundle_manifest(bundle_path, expected_sha256=bundle_sha256)
    verify_bound_artifacts(manifest, artifact_root=root)
    core = manifest["core"]
    payload_binding = core["payload"]
    payload_path = root / payload_binding["object_path"]
    payload = payload_path.read_bytes()
    if (
        len(payload) != payload_binding["byte_count"]
        or hashlib.sha256(payload).hexdigest() != payload_binding["sha256"]
    ):
        raise ValueError("RSI payload identity differs")
    arrays = decode_clip_payload(payload, steps=core["steps"], plain_comparison=True)
    rng_binding = core["rng_state"]["binding"]
    rng_path = root / rng_binding["object_path"]
    rng_bytes = rng_path.read_bytes()
    if (
        len(rng_bytes) != rng_binding["byte_count"]
        or hashlib.sha256(rng_bytes).hexdigest() != rng_binding["sha256"]
    ):
        raise ValueError("RSI RNG identity differs")
    rng_state = json.loads(rng_bytes.decode("utf-8"))
    if type(rng_state) is not dict or canonical_json_bytes(rng_state) != rng_bytes:
        raise ValueError("RSI RNG state is not canonical")
    return _RSIMaterial(
        block=block,
        behavior=behavior,
        bundle_sha256=bundle_sha256,
        arrays=MappingProxyType(arrays),
        rng_state=MappingProxyType(rng_state),
    )


def _integration_state(environment: object, width: int) -> np.ndarray:
    import mujoco

    physical = environment.unwrapped
    value = np.empty(width, dtype="<f8")
    mujoco.mj_getState(
        physical.model,
        physical.data,
        value,
        mujoco.mjtState.mjSTATE_INTEGRATION,
    )
    return np.ascontiguousarray(value, dtype="<f8")


def _rng_state_sha256(physical: object) -> str:
    generator = getattr(physical, "np_random", None)
    bit_generator = getattr(generator, "bit_generator", None)
    state = getattr(bit_generator, "state", None)
    if type(state) is not dict:
        raise ValueError("environment RNG state is unavailable")
    return sha256_json(state)


@dataclass(frozen=True, slots=True)
class RSIRestoreDependencies:
    set_integration_state: Callable[[object, np.ndarray], None]
    restore_wrapper: Callable[[object, int, np.ndarray], None]
    restore_rng: Callable[[object, Mapping[str, object]], None]
    step: Callable[[object, np.ndarray], tuple[object, float, bool, bool, object]]
    capture_wrapper: Callable[[object], tuple[int, np.ndarray]]
    integration_state: Callable[[object, int], np.ndarray]
    rng_state_sha256: Callable[[object], str]
    counted_transition_total: Callable[[object], int | None]


def _default_rsi_restore_dependencies() -> RSIRestoreDependencies:
    def set_integration_state(environment: object, state: np.ndarray) -> None:
        import mujoco

        physical = environment.unwrapped
        mujoco.mj_setState(
            physical.model,
            physical.data,
            state,
            mujoco.mjtState.mjSTATE_INTEGRATION,
        )

    return RSIRestoreDependencies(
        set_integration_state=set_integration_state,
        restore_wrapper=restore_wrapper_state,
        restore_rng=lambda environment, state: restore_rng_state(
            environment.unwrapped, dict(state)
        ),
        step=lambda environment, action: environment.step(action),
        capture_wrapper=capture_wrapper_state,
        integration_state=_integration_state,
        rng_state_sha256=lambda environment: _rng_state_sha256(environment.unwrapped),
        counted_transition_total=lambda environment: getattr(
            environment, "counted_transitions", None
        ),
    )


def restore_predecessor_rsi(
    *,
    environment: object,
    material: _RSIMaterial,
    start_boundary: int,
    counted_transitions: int,
    dependencies: RSIRestoreDependencies | None = None,
) -> tuple[np.ndarray, RSIRestorationReceipt]:
    """Restore at ``s-1``, execute once, and verify the authentic boundary ``s``."""

    arrays = material.arrays
    selected = dependencies or _default_rsi_restore_dependencies()
    if type(start_boundary) is not int or not 0 <= start_boundary <= 488:
        raise ValueError("RSI start boundary lies outside [0,488]")
    raw_observation, _info = environment.reset(seed=material.block)
    if start_boundary == 0:
        observation = np.ascontiguousarray(raw_observation, dtype="<f8")
        elapsed, flags = selected.capture_wrapper(environment)
        physical = environment.unwrapped
        integration = selected.integration_state(
            environment, arrays["boundary_integration_state"].shape[1]
        )
        rng_sha256 = selected.rng_state_sha256(environment)
        if (
            observation.tobytes(order="C") != arrays["boundary_observation"][0].tobytes(order="C")
            or integration.tobytes(order="C")
            != arrays["boundary_integration_state"][0].tobytes(order="C")
            or elapsed != int(arrays["boundary_wrapper_elapsed"][0])
            or flags.tobytes(order="C") != arrays["boundary_wrapper_flags"][0].tobytes(order="C")
            or float(physical.data.time) != float(arrays["boundary_simulation_time"][0])
            or np.ascontiguousarray(physical.data.qpos[:2], dtype="<f8").tobytes(order="C")
            != arrays["boundary_root_xy"][0].tobytes(order="C")
            or rng_sha256 != bytes(arrays["boundary_rng_state_sha256"][0]).decode("ascii")
        ):
            raise PhaseBTrainingError(
                TrainingFailureStatus.COUNTER_DRIFT,
                "boundary-zero certified reset differs",
            )
        receipt = RSIRestorationReceipt(
            start_boundary=0,
            predecessor_boundary=None,
            predecessor_executed=False,
            predecessor_counted=False,
            counted_transitions_before=counted_transitions,
            counted_transitions_after=counted_transitions,
            wrapper_elapsed_after=elapsed,
            observation_sha256=array_sha256(observation),
            integration_state_sha256=array_sha256(integration),
            rng_state_sha256=rng_sha256,
        )
        validate_rsi_restoration_receipt(receipt)
        return observation, receipt

    predecessor = start_boundary - 1
    physical = environment.unwrapped
    selected.set_integration_state(
        environment,
        arrays["boundary_integration_state"][predecessor],
    )
    selected.restore_wrapper(
        environment,
        int(arrays["boundary_wrapper_elapsed"][predecessor]),
        arrays["boundary_wrapper_flags"][predecessor],
    )
    selected.restore_rng(environment, material.rng_state)
    counter_before = selected.counted_transition_total(environment)
    observation, _discarded_reward, terminated, truncated, _discarded_info = selected.step(
        environment,
        arrays["transition_physical_action"][predecessor].copy(),
    )
    counter_after = selected.counted_transition_total(environment)
    if counter_before is not None and counter_after != counter_before:
        raise PhaseBTrainingError(
            TrainingFailureStatus.COUNTER_DRIFT,
            "RSI predecessor reconstruction was counted as training",
        )
    observation = np.ascontiguousarray(observation, dtype="<f8")
    elapsed, flags = selected.capture_wrapper(environment)
    integration = selected.integration_state(
        environment, arrays["boundary_integration_state"].shape[1]
    )
    rng_sha256 = selected.rng_state_sha256(environment)
    expected_flags = arrays["boundary_result_flags"][start_boundary]
    if (
        observation.tobytes(order="C")
        != arrays["boundary_observation"][start_boundary].tobytes(order="C")
        or integration.tobytes(order="C")
        != arrays["boundary_integration_state"][start_boundary].tobytes(order="C")
        or elapsed != int(arrays["boundary_wrapper_elapsed"][start_boundary])
        or flags.tobytes(order="C")
        != arrays["boundary_wrapper_flags"][start_boundary].tobytes(order="C")
        or (bool(terminated), bool(truncated)) != (bool(expected_flags[0]), bool(expected_flags[1]))
        or float(physical.data.time) != float(arrays["boundary_simulation_time"][start_boundary])
        or np.ascontiguousarray(physical.data.qpos[:2], dtype="<f8").tobytes(order="C")
        != arrays["boundary_root_xy"][start_boundary].tobytes(order="C")
        or rng_sha256 != bytes(arrays["boundary_rng_state_sha256"][start_boundary]).decode("ascii")
    ):
        raise PhaseBTrainingError(
            TrainingFailureStatus.COUNTER_DRIFT,
            "predecessor RSI reconstruction differs from certified boundary",
        )
    receipt = RSIRestorationReceipt(
        start_boundary=start_boundary,
        predecessor_boundary=predecessor,
        predecessor_executed=True,
        predecessor_counted=False,
        counted_transitions_before=counted_transitions,
        counted_transitions_after=counted_transitions,
        wrapper_elapsed_after=elapsed,
        observation_sha256=array_sha256(observation),
        integration_state_sha256=array_sha256(integration),
        rng_state_sha256=rng_sha256,
    )
    validate_rsi_restoration_receipt(receipt)
    return observation, receipt


class _SharedResetAssignments:
    def __init__(self, scheduler: BalancedRSIScheduler) -> None:
        self.scheduler = scheduler
        self._rehearsal_episode = 0
        self._composition_episode = 0
        self._paired_rehearsal_episodes = {2: 0, 3: 0}
        self._paired_composition_episodes = {0: 0, 1: 0}
        self._block_order = (
            ()
            if scheduler.pairing_declared
            else tuple(
                sorted(
                    scheduler._cells[::3],
                    key=lambda item: domain_separated_seed(
                        scheduler.manifest_sha256,
                        scheduler.ppo_seed,
                        f"composition-block:{item[0]}",
                    ),
                )
            )
        )

    def rehearsal(self, environment_index: int) -> RSIAssignment:
        if self.scheduler.pairing_declared:
            index = self._paired_rehearsal_episodes[environment_index]
            self._paired_rehearsal_episodes[environment_index] += 1
            return self.scheduler.assignment(
                global_episode_index=index,
                environment_index=environment_index,
            )
        index = self._rehearsal_episode
        self._rehearsal_episode += 1
        return self.scheduler.assignment(
            global_episode_index=index,
            environment_index=environment_index,
        )

    def composition_block(self, environment_index: int) -> int:
        if self.scheduler.pairing_declared:
            index = self._paired_composition_episodes[environment_index]
            self._paired_composition_episodes[environment_index] += 1
            return self.scheduler.composition_block(
                global_episode_index=index,
                environment_index=environment_index,
            )
        block = self._block_order[self._composition_episode % len(self._block_order)][0]
        self._composition_episode += 1
        return block


def _torso_up(state: HumanoidTrackingState) -> float:
    w, x, y, z = (float(value) for value in state.root_orientation_wxyz)
    norm_squared = w * w + x * x + y * y + z * z
    value = (w * w - x * x - y * y + z * z) / norm_squared
    if not math.isfinite(value):
        raise ValueError("torso-up signal is non-finite")
    return value


def _policy_observation(state_observation: np.ndarray, window: np.ndarray) -> np.ndarray:
    state = np.ascontiguousarray(state_observation, dtype="<f4")
    reference = np.ascontiguousarray(window, dtype="<f4")
    if state.shape != (OBSERVATION_WIDTH,) or reference.shape != (
        REFERENCE_HORIZON,
        REFERENCE_WIDTH,
    ):
        raise ValueError("policy observation source shape differs")
    result = np.ascontiguousarray(
        np.concatenate((state, reference.reshape(-1))),
        dtype="<f4",
    )
    if result.shape != (POLICY_INPUT_WIDTH,) or not np.isfinite(result).all():
        raise ValueError("policy observation is malformed")
    return result


class RealPhaseBTrainingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gym-compatible wrapper for composition and certified RSI rehearsal."""

    metadata: ClassVar[dict[str, object]] = {}

    def __init__(
        self,
        *,
        environment_index: int,
        plan: TrainingPlan,
        assignments: _SharedResetAssignments,
        config: RealRuntimeConfig,
    ) -> None:
        if environment_index not in range(4):
            raise ValueError("environment index lies outside the four-vector contract")
        self.environment_index = environment_index
        self.stream = STREAM_BY_ENVIRONMENT[environment_index]
        self.plan = plan
        self.assignments = assignments
        self.config = config
        _verify_runtime_input(config, config.oracle_path)
        self.oracle, oracle_sha256 = load_phase_b_oracle(
            config.oracle_path,
            available_behaviors=BEHAVIORS,
        )
        _verify_runtime_input(config, config.reward_path)
        self.reward_spec, reward_sha256 = RewardRegistry().load(config.reward_path)
        _verify_runtime_input(config, config.experiment / "library_manifest_v1.json")
        _verify_runtime_input(config, config.experiment / "task_spec_v1.json")
        self.library, self.task = load_frozen_inputs(config.experiment)
        expected = {item.relative_path: item for item in config.sealed_inputs}
        oracle_relative = config.oracle_path.relative_to(config.repository_root).as_posix()
        reward_relative = config.reward_path.relative_to(config.repository_root).as_posix()
        if (
            oracle_sha256 != expected[oracle_relative].sha256
            or reward_sha256 != expected[reward_relative].sha256
            or self.library.raw_sha256
            != expected[
                (config.experiment / "library_manifest_v1.json")
                .relative_to(config.repository_root)
                .as_posix()
            ].sha256
            or self.task.raw_sha256
            != expected[
                (config.experiment / "task_spec_v1.json")
                .relative_to(config.repository_root)
                .as_posix()
            ].sha256
        ):
            raise ExperimentContractError("runtime input identity differs from parent seal")
        self.base = make_reference_corpus_env()
        self.action_space = gym.spaces.Box(
            low=np.full(ACTION_WIDTH, -0.4, dtype="<f4"),
            high=np.full(ACTION_WIDTH, 0.4, dtype="<f4"),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(POLICY_INPUT_WIDTH,),
            dtype=np.float32,
        )
        self.abi = validate_humanoid_actuator_abi(self.base)
        self.corpus_root = config.repository_root / "artifacts/reference_corpus_v2"
        self.counted_transitions = 0
        self.rsi_ledger: list[dict[str, object]] = []
        self.rsi_reset_count = 0
        self._frame: object | None = None
        self._runtime: ComposedReferenceRuntime | None = None
        self._assignment: RSIAssignment | None = None
        self._references: dict[str, np.ndarray] = {}
        self._boundary = 0
        self._initial_x = 0.0
        self._last_x = 0.0
        self._last_speed = 0.0

    @property
    def unwrapped(self) -> object:
        return self.base.unwrapped

    def _load_references(self, block: int) -> dict[str, np.ndarray]:
        for behavior in BEHAVIORS:
            _verify_clip_lineage(self.config, block=block, behavior=behavior)
        return {
            behavior: load_v2_reference_clip(
                self.corpus_root,
                block=block,
                behavior=behavior,
            ).reference_rows
            for behavior in BEHAVIORS
        }

    def _signals(self, state: HumanoidTrackingState) -> dict[str, object]:
        dwell = self._runtime.dwell if self._runtime is not None else self._boundary
        signals: dict[str, object] = {
            "dwell": dwell,
            "t": self._boundary,
            "torso_up": _torso_up(state),
            "v_target": self.task.target_at(min(self._boundary, 999)),
            "v_x": self._last_speed,
            "x_travelled": float(state.root_position_world_m[0]) - self._initial_x,
            "z_root": state.root_height_m,
        }
        if set(signals) != ALLOWED_SIGNALS:
            raise AssertionError("Phase B signal set drifted")
        return signals

    def _scheduled_window(self, boundary: int) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
        assignment = self._assignment
        if assignment is None:
            raise AssertionError("rehearsal assignment is unavailable")
        rows = []
        identities = []
        for offset in range(REFERENCE_HORIZON):
            absolute = min(boundary + offset, 1_000)
            behavior = assignment.behavior_at(absolute)
            rows.append(self._references[behavior][absolute])
            identities.append({"behavior": behavior, "boundary": absolute})
        target_boundary = min(boundary + 1, 1_000)
        target_behavior = assignment.behavior_at(target_boundary)
        target = np.ascontiguousarray(
            self._references[target_behavior][target_boundary],
            dtype="<f8",
        )
        window = np.ascontiguousarray(rows, dtype="<f4")
        return (
            window,
            target,
            {
                "behavior": assignment.behavior_at(boundary),
                "identities": identities,
                "policy_window_sha256": array_sha256(window),
                "reward_target_sha256": array_sha256(target),
            },
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del seed, options
        if self.stream == "composition":
            block = self.assignments.composition_block(self.environment_index)
            observation, _info = self.base.reset(seed=block)
            self._references = self._load_references(block)
            self._runtime = ComposedReferenceRuntime(self.oracle, self._references)
            self._assignment = None
            self._boundary = 0
            state = tracking_state_with_bounded_reset_orientation(self.base, self.abi)
            self._initial_x = self._last_x = float(state.root_position_world_m[0])
            self._last_speed = 0.0
            self._frame = self._runtime.frame(state=state, signals=self._signals(state))
            policy_observation = _policy_observation(observation, self._frame.policy_window)
            return policy_observation, {"block": block, "stream": self.stream}

        assignment = self.assignments.rehearsal(self.environment_index)
        self.rsi_reset_count += 1
        _verify_clip_lineage(
            self.config,
            block=assignment.block,
            behavior=assignment.origin_behavior,
        )
        material = _load_rsi_material(
            self.corpus_root,
            block=assignment.block,
            behavior=assignment.origin_behavior,
        )
        observation, receipt = restore_predecessor_rsi(
            environment=self.base,
            material=material,
            start_boundary=assignment.start_boundary,
            counted_transitions=self.counted_transitions,
        )
        self._assignment = assignment
        self._runtime = None
        self._references = self._load_references(assignment.block)
        self._boundary = assignment.start_boundary
        state = tracking_state(self.base, self.abi)
        self._initial_x = self._last_x = float(state.root_position_world_m[0])
        self._last_speed = 0.0
        window, _target, schedule = self._scheduled_window(self._boundary)
        entry = {
            **assignment.to_dict(),
            "environment_index": self.environment_index,
            "predecessor_receipt": receipt.to_dict(),
            "scheduled_window": schedule,
        }
        self.rsi_ledger.append(entry)
        return _policy_observation(observation, window), {
            "rsi": entry,
            "stream": self.stream,
        }

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        physical = np.ascontiguousarray(action, dtype="<f4")
        if (
            physical.shape != (ACTION_WIDTH,)
            or not np.isfinite(physical).all()
            or np.any(physical < np.float32(-0.4))
            or np.any(physical > np.float32(0.4))
        ):
            raise ValueError("Phase B physical action lies outside the exact bounds")
        if self.stream == "composition":
            if self._frame is None or self._runtime is None:
                raise AssertionError("composition frame is unavailable")
            target = self._frame.hidden_reward_target
        else:
            _window, target, _schedule = self._scheduled_window(self._boundary)
        physical_environment = self.base.unwrapped
        body_mass = np.ascontiguousarray(physical_environment.model.body_mass, dtype="<f8")
        body_xipos_before = np.ascontiguousarray(
            physical_environment.data.xipos, dtype="<f8"
        ).copy()
        observation, stock_reward, terminated, truncated, base_info = self.base.step(physical)
        com_forward_speed = body_mass_weighted_com_x_velocity_m_s(
            body_mass_f64=body_mass,
            body_xipos_before_f64=body_xipos_before,
            body_xipos_after_f64=np.ascontiguousarray(physical_environment.data.xipos, dtype="<f8"),
            control_period_s=0.015,
        )
        state = tracking_state(self.base, self.abi)
        reward = compose_registered_reward(
            state=state,
            hidden_reference_target=target,
            ignored_stock_reward=float(stock_reward),
            task_inputs=admit_task_inputs_v2(
                com_x_velocity_m_s=com_forward_speed,
                measurement_origin=MEASUREMENT_ORIGIN,
                cadence_seconds=0.015,
            ),
            specification=self.reward_spec,
        )
        root_x = float(state.root_position_world_m[0])
        root_delta_speed = (root_x - self._last_x) / 0.015
        self._last_x = root_x
        self._last_speed = root_delta_speed
        self._boundary += 1
        self.counted_transitions += 1
        done = bool(terminated or truncated)
        if not done:
            if self.stream == "composition":
                self._runtime.advance()
                self._frame = self._runtime.frame(state=state, signals=self._signals(state))
                window = self._frame.policy_window
            else:
                window, _next_target, _schedule = self._scheduled_window(self._boundary)
            next_observation = _policy_observation(observation, window)
        else:
            next_observation = _policy_observation(
                observation,
                np.repeat(np.ascontiguousarray(target[None, :], dtype="<f4"), 8, axis=0),
            )
        info = dict(base_info)
        info["phase_b"] = {
            "com_forward_speed_m_s": com_forward_speed,
            "counted_transition_delta": 1,
            "ignored_stock_reward": reward.ignored_stock_reward,
            "phase_selection_ok": True,
            "r_task": reward.r_task,
            "r_track": reward.r_track,
            "r_train": reward.r_train,
            "root_delta_forward_speed_m_s": root_delta_speed,
            "stream": self.stream,
            "tracking_errors": reward.tracking.error_components(),
        }
        return next_observation, reward.r_train, bool(terminated), bool(truncated), info

    def close(self) -> None:
        self.base.close()


class FakePhaseBTrainingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Small deterministic environment used only by fake-runtime contract tests."""

    metadata: ClassVar[dict[str, object]] = {}

    def __init__(
        self,
        *,
        environment_index: int,
        plan: TrainingPlan,
        assignments: _SharedResetAssignments,
        failure_mode: str | None = None,
        episode_steps: int = 13,
    ) -> None:
        self.environment_index = environment_index
        self.plan = plan
        self.stream = STREAM_BY_ENVIRONMENT[environment_index]
        self.assignments = assignments
        self.scheduler = assignments.scheduler
        self.failure_mode = failure_mode
        self.episode_steps = episode_steps
        self.action_space = gym.spaces.Box(
            low=np.full(ACTION_WIDTH, -0.4, dtype="<f4"),
            high=np.full(ACTION_WIDTH, 0.4, dtype="<f4"),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(POLICY_INPUT_WIDTH,),
            dtype=np.float32,
        )
        self._episode = 0
        self._step = 0
        self.counted_transitions = 0
        self.reset_ledger: list[dict[str, object]] = []
        self._rsi_ledger: list[dict[str, object]] = []
        self.rsi_reset_count = 0

    @property
    def rsi_ledger(self) -> list[dict[str, object]] | None:
        if self.failure_mode == "rsi_get_attr_failure" and self.environment_index == 2:
            raise AttributeError("controlled missing RSI ledger API")
        if self.failure_mode == "rsi_none" and self.environment_index == 2:
            return None
        return self._rsi_ledger

    def _observation(self) -> np.ndarray:
        state = np.linspace(-0.2, 0.2, OBSERVATION_WIDTH, dtype="<f4")
        state = state + np.float32(self.environment_index * 0.01 + self._step * 0.001)
        reference = np.linspace(
            -0.1,
            0.1,
            REFERENCE_HORIZON * REFERENCE_WIDTH,
            dtype="<f4",
        )
        reference = reference + np.float32(self.environment_index * 0.02 + self._step * 0.002)
        return np.ascontiguousarray(np.concatenate((state, reference)), dtype="<f4")

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del seed, options
        self._step = 0
        if self.stream == "composition":
            self.reset_ledger.append(
                {
                    "block": self.assignments.composition_block(self.environment_index),
                    "environment_index": self.environment_index,
                    "global_episode_index": self._episode,
                    "stream": self.stream,
                }
            )
        else:
            self.rsi_reset_count += 1
            assignment = self.assignments.rehearsal(self.environment_index)
            self.reset_ledger.append({**assignment.to_dict(), "stream": self.stream})
            receipt = RSIRestorationReceipt(
                start_boundary=assignment.start_boundary,
                predecessor_boundary=(
                    None if assignment.start_boundary == 0 else assignment.start_boundary - 1
                ),
                predecessor_executed=assignment.start_boundary > 0,
                predecessor_counted=False,
                counted_transitions_before=self.counted_transitions,
                counted_transitions_after=self.counted_transitions,
                wrapper_elapsed_after=assignment.start_boundary,
                observation_sha256="1" * 64,
                integration_state_sha256="2" * 64,
                rng_state_sha256="3" * 64,
            )
            validate_rsi_restoration_receipt(receipt)
            if self.failure_mode != "rsi_empty" and not (
                self.failure_mode == "rsi_omitted_reset"
                and self.environment_index == 2
                and self.rsi_reset_count == 1
            ):
                self._rsi_ledger.append(
                    {
                        **assignment.to_dict(),
                        "environment_index": self.environment_index,
                        "hidden_target_sha256": "4" * 64,
                        "policy_window_sha256": "5" * 64,
                        "predecessor_receipt": receipt.to_dict(),
                    }
                )
        self._episode += 1
        return self._observation(), {"stream": self.stream}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self._step += 1
        self.counted_transitions += 1
        action_value = np.ascontiguousarray(action, dtype="<f4")
        target = np.float32(0.05 * (self.environment_index + 1))
        r_track = float(1.0 - np.mean(np.square(action_value - target), dtype=np.float64))
        if (
            self.failure_mode in {"non_finite", "non_finite_close_failure"}
            and self.counted_transitions == 2
        ):
            r_track = float("nan")
        r_task = 0.0
        r_train = r_track + r_task
        observation = self._observation()
        if self.failure_mode == "non_finite_observation" and self.counted_transitions == 2:
            observation[0] = np.nan
        phase_selection_ok = not (
            self.failure_mode == "phase_selection" and self.counted_transitions == 2
        )
        delta = 0 if self.failure_mode == "counter_drift" and self.counted_transitions == 2 else 1
        truncated = self._step >= self.episode_steps
        return (
            observation,
            r_train,
            False,
            truncated,
            {
                "phase_b": {
                    "counted_transition_delta": delta,
                    "ignored_stock_reward": 0.0,
                    "phase_selection_ok": phase_selection_ok,
                    "r_task": r_task,
                    "r_track": r_track,
                    "r_train": r_train,
                    "stream": self.stream,
                }
            },
        )

    def close(self) -> None:
        if self.failure_mode == "non_finite_close_failure":
            raise RuntimeError("controlled vector environment close failure")
        return None


def fake_policy_factory(plan: TrainingPlan) -> FullAuthorityPolicy:
    """Build a deterministic full-sized FT1 actor without reading external payloads."""

    generator = np.random.Generator(
        np.random.PCG64(plan_domain_separated_seed(plan, "fake-policy"))
    )
    first = np.ascontiguousarray(
        generator.normal(0.0, 0.01, size=(256, POLICY_INPUT_WIDTH)).astype("<f4")
    )
    first[:, OBSERVATION_WIDTH:] = np.float32(0.0)
    parameters = {
        "latent_pi.0.weight": first,
        "latent_pi.0.bias": np.zeros(256, dtype="<f4"),
        "latent_pi.2.weight": np.ascontiguousarray(
            generator.normal(0.0, 0.01, size=(256, 256)).astype("<f4")
        ),
        "latent_pi.2.bias": np.zeros(256, dtype="<f4"),
        "mu.weight": np.ascontiguousarray(
            generator.normal(0.0, 0.01, size=(ACTION_WIDTH, 256)).astype("<f4")
        ),
        "mu.bias": np.zeros(ACTION_WIDTH, dtype="<f4"),
        "log_std.weight": np.ascontiguousarray(
            generator.normal(0.0, 0.005, size=(ACTION_WIDTH, 256)).astype("<f4")
        ),
        "log_std.bias": np.full(ACTION_WIDTH, -1.0, dtype="<f4"),
    }
    actor = FullAuthorityActor(
        parameters,
        source_actor_sha256=("60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b"),
    )
    return FullAuthorityPolicy(actor, value_seed=20260905)


def real_policy_factory(config: RealRuntimeConfig) -> Callable[[TrainingPlan], FullAuthorityPolicy]:
    def build(_plan: TrainingPlan) -> FullAuthorityPolicy:
        _verify_runtime_input(config, config.starting_actor_path)
        loaded = load_full_authority_actor(
            config.starting_actor_path,
            expected_sha256=config.starting_actor_sha256,
        )
        return FullAuthorityPolicy(loaded.actor, value_seed=config.value_seed)

    return build


def worker_step_zero_e1_audit(
    policy: FullAuthorityPolicy,
    *,
    repository_root: Path,
    receipt_path: Path,
    expected_receipt_sha256: str,
    sealed_inputs: tuple[SealedArtifact, ...] | None = None,
    sealed_input_lineage_sha256: str | None = None,
) -> dict[str, object]:
    """Re-run the worker action path on the unchanged FT1 fixture authority."""

    from .receipts import _input_batch

    if (sealed_inputs is None) != (sealed_input_lineage_sha256 is None):
        raise ExperimentContractError("worker E1 seal declaration is incomplete")
    if sealed_inputs is not None and sealed_input_lineage_sha256 is not None:
        verify_sealed_inputs(
            repository_root,
            sealed_inputs,
            expected_lineage_sha256=sealed_input_lineage_sha256,
        )
    receipt_file = Path(receipt_path)
    if sha256_file(receipt_file) != expected_receipt_sha256:
        raise ExperimentContractError("worker E1 receipt identity differs")
    receipt = _canonical_object(receipt_file)
    states, windows, _records = _input_batch(Path(repository_root))
    fixture = receipt.get("fixture_batch")
    expected = receipt.get("identity_checks")
    if (
        type(fixture) is not dict
        or type(expected) is not dict
        or fixture.get("state_sha256") != array_sha256(states)
        or fixture.get("window_sha256") != array_sha256(windows)
    ):
        raise ExperimentContractError("worker E1 fixture identity differs")
    action = policy.actor.act(compose_policy_input(states, windows)).physical
    action_identity = expected.get("deterministic_physical_action")
    if (
        type(action_identity) is not dict
        or action_identity.get("bitwise_equal") is not True
        or action_identity.get("sha256") != array_sha256(action)
    ):
        raise PhaseBTrainingError(
            TrainingFailureStatus.ACTION_BOUND_VIOLATION,
            "worker step-0 action path differs from E1",
        )
    return {
        "action_sha256": array_sha256(action),
        "bitwise_equal": True,
        "e1_receipt_sha256": expected_receipt_sha256,
        "fixture_count": len(action),
        "reported_beside_checkpoint": True,
        "worker_path": "compose_policy_input_then_full_authority_actor_then_exact_physical_action",
    }


def fake_environment_factories(
    *,
    plan: TrainingPlan,
    failure_mode: str | None = None,
    episode_steps: int = 13,
) -> tuple[Callable[[], object], ...]:
    scheduler = BalancedRSIScheduler.from_plan(plan)
    assignments = _SharedResetAssignments(scheduler)
    return tuple(
        lambda index=index: FakePhaseBTrainingEnv(
            environment_index=index,
            plan=plan,
            assignments=assignments,
            failure_mode=failure_mode,
            episode_steps=episode_steps,
        )
        for index in range(4)
    )


def real_environment_factories(
    *,
    plan: TrainingPlan,
    config: RealRuntimeConfig,
) -> tuple[Callable[[], object], ...]:
    scheduler = BalancedRSIScheduler.from_plan(plan)
    assignments = _SharedResetAssignments(scheduler)
    return tuple(
        lambda index=index: RealPhaseBTrainingEnv(
            environment_index=index,
            plan=plan,
            assignments=assignments,
            config=config,
        )
        for index in range(4)
    )


__all__ = [
    "FakePhaseBTrainingEnv",
    "RSIRestoreDependencies",
    "RealPhaseBTrainingEnv",
    "RealRuntimeConfig",
    "fake_environment_factories",
    "fake_policy_factory",
    "real_environment_factories",
    "real_policy_factory",
    "restore_predecessor_rsi",
    "worker_step_zero_e1_audit",
]
