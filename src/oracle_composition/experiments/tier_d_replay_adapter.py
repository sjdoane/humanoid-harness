"""Exact Gymnasium/MuJoCo adapter for the one-step E2 replay probe."""

from __future__ import annotations

import hashlib
import platform
import struct
from pathlib import Path
from typing import Any

import numpy as np

from oracle_composition.envs import humanoid as humanoid_module
from oracle_composition.envs.humanoid import (
    CONTACT_CAPTURE_ID,
    HumanoidExperimentConfig,
    make_humanoid_env,
)

from .runtime_identity import dependency_lock_path, module_sha256, source_tree_sha256, space_sha256
from .tier_d_replay_contract import (
    _RECEIPT_ISSUE_TOKEN,
    CANONICALIZATION_ID,
    EXPECTED_STATE_COMPONENT_SIZES,
    EXPECTED_WRAPPER_TYPES,
    STATE_DTYPE,
    STATE_FLAG,
    STATE_SHAPE,
    STATE_SPEC,
    ReplayContact,
    ReplayProbe,
    ReplayRuntimeIdentity,
    ReplaySnapshot,
    ReplayTransition,
    ReplayVerificationReceipt,
    ReplayWrapperState,
    TierDReplayContractError,
    TierDReplayMismatchError,
    array_sha256,
    compare_replay,
    contacts_sha256,
)


def _sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise TierDReplayContractError(f"runtime source is not a regular file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wrapper_types(environment: Any) -> tuple[str, ...]:
    observed: list[str] = []
    current = environment
    while True:
        observed.append(f"{type(current).__module__}.{type(current).__name__}")
        if not hasattr(current, "env"):
            break
        current = current.env
    return tuple(observed)


def _time_limit(environment: Any) -> Any:
    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - guarded by the gym extra
        raise TierDReplayContractError("Gymnasium is unavailable") from exc
    if not isinstance(environment, gym.wrappers.TimeLimit):
        raise TierDReplayContractError("replay requires TimeLimit as the outer wrapper")
    elapsed = getattr(environment, "_elapsed_steps", None)
    if not isinstance(elapsed, int) or isinstance(elapsed, bool) or elapsed < 0:
        raise TierDReplayContractError("environment must be reset before replay capture")
    return environment


def _state_components(model: Any, mujoco: Any) -> tuple[tuple[str, int], ...]:
    flags = (
        ("time", mujoco.mjtState.mjSTATE_TIME),
        ("qpos", mujoco.mjtState.mjSTATE_QPOS),
        ("qvel", mujoco.mjtState.mjSTATE_QVEL),
        ("act", mujoco.mjtState.mjSTATE_ACT),
        ("warmstart", mujoco.mjtState.mjSTATE_WARMSTART),
        ("ctrl", mujoco.mjtState.mjSTATE_CTRL),
        ("qfrc_applied", mujoco.mjtState.mjSTATE_QFRC_APPLIED),
        ("xfrc_applied", mujoco.mjtState.mjSTATE_XFRC_APPLIED),
        ("eq_active", mujoco.mjtState.mjSTATE_EQ_ACTIVE),
        ("mocap_pos", mujoco.mjtState.mjSTATE_MOCAP_POS),
        ("mocap_quat", mujoco.mjtState.mjSTATE_MOCAP_QUAT),
        ("userdata", mujoco.mjtState.mjSTATE_USERDATA),
        ("plugin", mujoco.mjtState.mjSTATE_PLUGIN),
    )
    return tuple((name, int(mujoco.mj_stateSize(model, flag))) for name, flag in flags)


def inspect_replay_runtime(
    environment: Any,
    config: HumanoidExperimentConfig,
) -> ReplayRuntimeIdentity:
    """Bind the exact runtime before source capture or restoration."""

    try:
        import gymnasium as gym
        import mujoco
        from gymnasium.envs.mujoco import humanoid_v5, mujoco_env
        from gymnasium.utils import passive_env_checker
        from gymnasium.wrappers import common as gym_common_wrappers
    except ImportError as exc:  # pragma: no cover - guarded by the gym extra
        raise TierDReplayContractError("Gymnasium MuJoCo is unavailable") from exc

    time_limit = _time_limit(environment)
    wrappers = _wrapper_types(environment)
    if wrappers != EXPECTED_WRAPPER_TYPES:
        raise TierDReplayContractError(
            f"wrapper stack differs: observed={wrappers!r}, expected={EXPECTED_WRAPPER_TYPES!r}"
        )
    physical = environment.unwrapped
    if getattr(physical, "contact_capture_id", None) != CONTACT_CAPTURE_ID:
        raise TierDReplayContractError("all-substep contact capture is not active")
    expected_kwargs: dict[str, bool | float | int | None] = {
        "render_mode": None,
        **config.gym_kwargs(),
    }
    observed_kwargs = dict(getattr(environment.spec, "kwargs", {}))
    if observed_kwargs != expected_kwargs:
        raise TierDReplayContractError(f"environment kwargs differ: observed={observed_kwargs!r}")
    physical_kwargs = {
        "exclude_current_positions_from_observation": getattr(
            physical,
            "_exclude_current_positions_from_observation",
            None,
        ),
        "frame_skip": getattr(physical, "frame_skip", None),
        "reset_noise_scale": getattr(physical, "_reset_noise_scale", None),
        "terminate_when_unhealthy": getattr(physical, "_terminate_when_unhealthy", None),
    }
    expected_physical_kwargs = config.gym_kwargs()
    if any(
        type(physical_kwargs[name]) is not type(expected) or physical_kwargs[name] != expected
        for name, expected in expected_physical_kwargs.items()
    ):
        raise TierDReplayContractError(
            f"physical environment kwargs differ: observed={physical_kwargs!r}"
        )
    contact_range = getattr(physical, "_contact_cost_range", None)
    if not isinstance(contact_range, tuple) or len(contact_range) != 2:
        raise TierDReplayContractError("physical contact-cost range is invalid")
    lower = contact_range[0]
    lower_label = (
        "negative_infinity"
        if type(lower) is float and np.isneginf(lower)
        else "not_negative_infinity"
    )
    environment_semantics = (
        ("contact_cost_range_lower", lower_label),
        ("contact_cost_range_upper", contact_range[1]),
        ("contact_cost_weight", getattr(physical, "_contact_cost_weight", None)),
        ("ctrl_cost_weight", getattr(physical, "_ctrl_cost_weight", None)),
        ("forward_reward_weight", getattr(physical, "_forward_reward_weight", None)),
        ("healthy_reward", getattr(physical, "_healthy_reward", None)),
        ("healthy_z_range_lower", getattr(physical, "_healthy_z_range", (None, None))[0]),
        ("healthy_z_range_upper", getattr(physical, "_healthy_z_range", (None, None))[1]),
        (
            "include_cfrc_ext_in_observation",
            getattr(physical, "_include_cfrc_ext_in_observation", None),
        ),
        (
            "include_cinert_in_observation",
            getattr(physical, "_include_cinert_in_observation", None),
        ),
        (
            "include_cvel_in_observation",
            getattr(physical, "_include_cvel_in_observation", None),
        ),
        (
            "include_qfrc_actuator_in_observation",
            getattr(physical, "_include_qfrc_actuator_in_observation", None),
        ),
    )

    state_flag = mujoco.mjtState.mjSTATE_INTEGRATION
    state_size = int(mujoco.mj_stateSize(physical.model, state_flag))
    components = _state_components(physical.model, mujoco)
    if int(state_flag) != STATE_FLAG or state_size != STATE_SHAPE[0]:
        raise TierDReplayContractError("mjSTATE_INTEGRATION ABI differs")
    if components != EXPECTED_STATE_COMPONENT_SIZES:
        raise TierDReplayContractError("mjSTATE_INTEGRATION component layout differs")

    model_path = Path(str(getattr(physical, "fullpath", "")))
    adapter_path = Path(__file__)
    environment_path = Path(str(getattr(humanoid_module, "__file__", "")))
    return ReplayRuntimeIdentity(
        environment_id=str(environment.spec.id),
        environment_kwargs=tuple(sorted(expected_kwargs.items())),
        environment_semantics=environment_semantics,
        wrapper_types=wrappers,
        time_limit_steps=int(time_limit._max_episode_steps),
        python_version=platform.python_version(),
        platform_system=platform.system(),
        platform_release=platform.release(),
        platform_machine=platform.machine(),
        numpy_version=np.__version__,
        gymnasium_version=gym.__version__,
        mujoco_version=mujoco.__version__,
        model_sha256=_sha256_file(model_path),
        dependency_lock_sha256=_sha256_file(dependency_lock_path()),
        project_source_tree_sha256=source_tree_sha256(),
        environment_source_sha256=_sha256_file(environment_path),
        gym_humanoid_source_sha256=module_sha256(humanoid_v5),
        gym_common_wrappers_source_sha256=module_sha256(gym_common_wrappers),
        gym_passive_checker_source_sha256=module_sha256(passive_env_checker),
        gym_mujoco_env_source_sha256=module_sha256(mujoco_env),
        mujoco_module_source_sha256=module_sha256(mujoco),
        mujoco_functions_binary_sha256=module_sha256(mujoco._functions),
        mujoco_structs_binary_sha256=module_sha256(mujoco._structs),
        mujoco_enums_binary_sha256=module_sha256(mujoco._enums),
        adapter_source_sha256=_sha256_file(adapter_path),
        contract_source_sha256=_sha256_file(adapter_path.with_name("tier_d_replay_contract.py")),
        integration_state_spec=STATE_SPEC,
        integration_state_flag=int(state_flag),
        integration_state_size=state_size,
        integration_state_dtype=STATE_DTYPE.str,
        integration_state_components=components,
        observation_shape=tuple(int(value) for value in environment.observation_space.shape),
        observation_dtype=np.dtype(environment.observation_space.dtype).str,
        observation_space_sha256=space_sha256(environment.observation_space),
        observation_reconstruction_id=CANONICALIZATION_ID,
        cfrc_ext_shape=tuple(int(value) for value in physical.data.cfrc_ext[1:].shape),
        cfrc_ext_dtype=np.dtype(physical.data.cfrc_ext.dtype).str,
        action_shape=tuple(int(value) for value in environment.action_space.shape),
        action_dtype=np.dtype(environment.action_space.dtype).str,
        action_space_sha256=space_sha256(environment.action_space),
        timestep_seconds=float(physical.model.opt.timestep),
        frame_skip=int(physical.frame_skip),
        control_period_seconds=(float(physical.model.opt.timestep) * int(physical.frame_skip)),
        integrator=mujoco.mjtIntegrator(physical.model.opt.integrator).name,
        solver=mujoco.mjtSolver(physical.model.opt.solver).name,
        solver_iterations=int(physical.model.opt.iterations),
        contact_capture_id=physical.contact_capture_id,
    )


def _capture_snapshot(environment: Any) -> ReplaySnapshot:
    import mujoco

    time_limit = _time_limit(environment)
    physical = environment.unwrapped
    mujoco.mj_forward(physical.model, physical.data)
    state = np.empty(STATE_SHAPE, dtype=STATE_DTYPE)
    mujoco.mj_getState(
        physical.model,
        physical.data,
        state,
        mujoco.mjtState.mjSTATE_INTEGRATION,
    )
    observation = np.asarray(physical._get_obs(), dtype=STATE_DTYPE)
    return ReplaySnapshot(
        integration_state=state,
        cfrc_ext=np.asarray(physical.data.cfrc_ext[1:], dtype=STATE_DTYPE),
        observation=observation,
        wrapper_state=ReplayWrapperState(time_limit_elapsed_steps=int(time_limit._elapsed_steps)),
    )


def _capture_contacts(environment: Any) -> tuple[ReplayContact, ...]:
    physical = environment.unwrapped
    if physical.last_control_step_substeps != physical.frame_skip:
        raise TierDReplayContractError("contact capture omitted one or more physics substeps")
    return tuple(
        ReplayContact(
            physics_substep_index=sample.physics_substep_index,
            contact_index_within_substep=sample.contact_index_within_substep,
            geom1_id=sample.geom1_id,
            geom2_id=sample.geom2_id,
            normal_force_n=sample.normal_force_n,
        )
        for sample in physical.last_control_step_contact_samples
    )


def _step_and_capture(environment: Any, action: np.ndarray) -> ReplayTransition:
    if not environment.action_space.contains(action):
        raise TierDReplayContractError("action is outside the exact Humanoid-v5 action space")
    next_observation, reward, terminated, truncated, _info = environment.step(action.copy())
    contacts = _capture_contacts(environment)
    next_snapshot = _capture_snapshot(environment)
    return ReplayTransition(
        action=action,
        next_snapshot=next_snapshot,
        returned_observation=np.asarray(next_observation),
        contacts=contacts,
        reward=float(reward),
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


def capture_one_step_replay_probe(
    environment: Any,
    *,
    config: HumanoidExperimentConfig,
    action: np.ndarray,
) -> ReplayProbe:
    """Capture one canonical source transition from an already reset environment."""

    runtime = inspect_replay_runtime(environment, config)
    resolved_action = np.asarray(action)
    anchor = _capture_snapshot(environment)
    transition = _step_and_capture(environment, resolved_action)
    return ReplayProbe(runtime=runtime, anchor=anchor, transition=transition)


def _config_from_runtime(runtime: ReplayRuntimeIdentity) -> HumanoidExperimentConfig:
    kwargs = dict(runtime.environment_kwargs)
    if kwargs.pop("render_mode") is not None:
        raise TierDReplayContractError("replay probe requires render_mode=None")
    return HumanoidExperimentConfig(
        env_id=runtime.environment_id,
        terminate_when_unhealthy=bool(kwargs["terminate_when_unhealthy"]),
        reset_noise_scale=float(kwargs["reset_noise_scale"]),
        exclude_current_positions_from_observation=bool(
            kwargs["exclude_current_positions_from_observation"]
        ),
        frame_skip=int(kwargs["frame_skip"]),
    )


def verify_one_step_replay_probe(probe: ReplayProbe) -> ReplayVerificationReceipt:
    """Restore in a fresh environment and issue a receipt only for an exact replay."""

    import mujoco

    if not isinstance(probe, ReplayProbe):
        raise TierDReplayContractError("probe has the wrong type")
    config = _config_from_runtime(probe.runtime)
    environment = make_humanoid_env(config, capture_substep_contacts=True)
    try:
        # Reset initializes OrderEnforcing, PassiveEnvChecker, TimeLimit, and
        # MuJoCo-owned data before the exact captured state replaces it.
        environment.reset(seed=0)
        observed_runtime = inspect_replay_runtime(environment, config)
        if observed_runtime != probe.runtime:
            raise TierDReplayMismatchError(("runtime",))

        time_limit = _time_limit(environment)
        physical = environment.unwrapped
        mujoco.mj_setState(
            physical.model,
            physical.data,
            probe.anchor.integration_state,
            mujoco.mjtState.mjSTATE_INTEGRATION,
        )
        time_limit._elapsed_steps = probe.anchor.wrapper_state.time_limit_elapsed_steps
        physical.data.cfrc_ext[1:] = probe.anchor.cfrc_ext
        restored_anchor = _capture_snapshot(environment)
        observed_transition = _step_and_capture(environment, probe.transition.action)
        comparison = compare_replay(
            probe,
            observed_runtime=observed_runtime,
            restored_anchor=restored_anchor,
            observed_transition=observed_transition,
        )
        if not comparison.passed:
            raise TierDReplayMismatchError(comparison.mismatches)
        return ReplayVerificationReceipt(
            runtime_sha256=probe.runtime.sha256,
            observed_runtime_sha256=observed_runtime.sha256,
            anchor_state_sha256=array_sha256(probe.anchor.integration_state),
            restored_anchor_state_sha256=array_sha256(restored_anchor.integration_state),
            anchor_cfrc_ext_sha256=array_sha256(probe.anchor.cfrc_ext),
            restored_anchor_cfrc_ext_sha256=array_sha256(restored_anchor.cfrc_ext),
            anchor_observation_sha256=array_sha256(probe.anchor.observation),
            restored_anchor_observation_sha256=array_sha256(restored_anchor.observation),
            action_sha256=array_sha256(probe.transition.action),
            next_state_sha256=array_sha256(probe.transition.next_snapshot.integration_state),
            observed_next_state_sha256=array_sha256(
                observed_transition.next_snapshot.integration_state
            ),
            next_cfrc_ext_sha256=array_sha256(probe.transition.next_snapshot.cfrc_ext),
            observed_next_cfrc_ext_sha256=array_sha256(observed_transition.next_snapshot.cfrc_ext),
            next_observation_sha256=array_sha256(probe.transition.returned_observation),
            observed_next_observation_sha256=array_sha256(observed_transition.returned_observation),
            next_canonical_observation_sha256=array_sha256(
                probe.transition.next_snapshot.observation
            ),
            observed_next_canonical_observation_sha256=array_sha256(
                observed_transition.next_snapshot.observation
            ),
            contacts_sha256=contacts_sha256(probe.transition.contacts),
            observed_contacts_sha256=contacts_sha256(observed_transition.contacts),
            reward_ieee754_hex=struct.pack(">d", probe.transition.reward).hex(),
            observed_reward_ieee754_hex=struct.pack(">d", observed_transition.reward).hex(),
            expected_terminated=probe.transition.terminated,
            observed_terminated=observed_transition.terminated,
            expected_truncated=probe.transition.truncated,
            observed_truncated=observed_transition.truncated,
            anchor_time_limit_elapsed_steps=(probe.anchor.wrapper_state.time_limit_elapsed_steps),
            restored_anchor_time_limit_elapsed_steps=(
                restored_anchor.wrapper_state.time_limit_elapsed_steps
            ),
            next_time_limit_elapsed_steps=(
                probe.transition.next_snapshot.wrapper_state.time_limit_elapsed_steps
            ),
            observed_next_time_limit_elapsed_steps=(
                observed_transition.next_snapshot.wrapper_state.time_limit_elapsed_steps
            ),
            comparisons=comparison,
            _issue_token=_RECEIPT_ISSUE_TOKEN,
        )
    finally:
        environment.close()


__all__ = [
    "EXPECTED_WRAPPER_TYPES",
    "capture_one_step_replay_probe",
    "inspect_replay_runtime",
    "verify_one_step_replay_probe",
]
