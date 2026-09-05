"""Collector for complete same-runtime Humanoid actor continuations."""

from __future__ import annotations

import copy
import hashlib
import math
import os
import platform
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import (
    array_sha256,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)
from oracle_composition.envs import reference_corpus as corpus_env_module
from oracle_composition.envs.humanoid import HumanoidExperimentConfig, make_humanoid_env
from oracle_composition.envs.reference_corpus import (
    FullSubstepContactSample,
    make_reference_corpus_env,
    require_reference_wrapper_stack,
)
from oracle_composition.sources import strict_tqc_actor_runtime as actor_runtime_module
from oracle_composition.sources.strict_tqc_actor_runtime import (
    INFERENCE_ID,
    StrictTQCActorRuntime,
)

from . import reference_corpus_contract as contract_module
from .reference_corpus_contract import (
    GEOM_NAME_BYTES,
    OBSERVATION_WIDTH,
    STATE_DTYPE,
    STATE_SIZE,
    WRAPPER_FLAG_NAMES,
    ReferenceCorpusContractError,
    array_bindings,
    boundary_record_sha256,
    contact_sequence_sha256,
    derive_reference_rows,
    reference_window_index_arrays,
    transition_record_sha256,
    validate_clip_arrays,
    validate_runtime_identity,
)
from .runtime_identity import dependency_lock_path, module_sha256, space_sha256

COLLECTOR_ID = "same_runtime_strict_npz_full_clip_collector/v1"
FLOOR_GEOM_NAME = "floor"
FOOT_GEOM_NAMES = frozenset({"left_foot", "right_foot"})


@dataclass(frozen=True, slots=True)
class CollectedClip:
    """In-memory result awaiting deterministic bundle publication."""

    clip_id: str
    clip_kind: str
    actor_variant: str
    seed: int
    reset_order: int
    steps: int
    arrays: dict[str, np.ndarray]
    runtime_identity: dict[str, object]
    rng_state: dict[str, object]
    actor_identity: dict[str, object]
    reward_canary: dict[str, object]
    collector_pid: int


def _source_hashes() -> dict[str, str]:
    import gymnasium
    import mujoco
    from gymnasium.envs.mujoco import humanoid_v5, mujoco_env
    from gymnasium.utils import passive_env_checker
    from gymnasium.wrappers import common as wrapper_common

    source_paths = {
        "reference_corpus_environment": Path(str(corpus_env_module.__file__)),
        "reference_corpus_contract": Path(str(contract_module.__file__)),
        "reference_corpus_collector": Path(__file__),
        "strict_tqc_actor_runtime": Path(str(actor_runtime_module.__file__)),
    }
    values = {name: sha256_file(path) for name, path in source_paths.items()}
    values.update(
        {
            "gymnasium": module_sha256(gymnasium),
            "gymnasium_humanoid_v5": module_sha256(humanoid_v5),
            "gymnasium_mujoco_env": module_sha256(mujoco_env),
            "gymnasium_wrapper_common": module_sha256(wrapper_common),
            "gymnasium_passive_checker": module_sha256(passive_env_checker),
            "mujoco": module_sha256(mujoco),
            "mujoco_functions": module_sha256(mujoco._functions),
            "mujoco_structs": module_sha256(mujoco._structs),
            "mujoco_enums": module_sha256(mujoco._enums),
        }
    )
    return values


def _state_components(model: object) -> list[list[object]]:
    import mujoco

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
    return [[name, int(mujoco.mj_stateSize(model, flag))] for name, flag in flags]


def inspect_reference_corpus_runtime(environment: object) -> dict[str, object]:
    """Inspect every frozen runtime fact before collection or verification."""

    import gymnasium
    import mujoco
    import torch

    require_reference_wrapper_stack(environment)
    physical = environment.unwrapped
    model_path = Path(str(getattr(physical, "fullpath", "")))
    if model_path.is_symlink() or not model_path.is_file():
        raise ReferenceCorpusContractError("MuJoCo model bytes are unavailable")
    model_bytes = model_path.read_bytes()
    config = HumanoidExperimentConfig()
    runtime: dict[str, object] = {
        "runtime_id": "humanoid_reference_corpus_runtime/v1",
        "environment_id": str(environment.spec.id),
        "environment_kwargs": {"render_mode": None, **config.gym_kwargs()},
        "environment_semantics": {
            "termination": "TimeLimit_only_at_1000",
            "healthy_z_range_open_m": [1.0, 2.0],
            "observation": "raw_Humanoid-v5_348d_excluding_root_xy",
            "action": "physical_17d_float32_in_closed_minus_0.4_to_0.4",
        },
        "wrapper_types": list(corpus_env_module.EXPECTED_REFERENCE_WRAPPER_TYPES),
        "time_limit_steps": int(environment._max_episode_steps),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "gymnasium_version": gymnasium.__version__,
        "mujoco_version": mujoco.__version__,
        "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
        "model_byte_count": len(model_bytes),
        "uv_lock_sha256": sha256_file(dependency_lock_path()),
        "observation_space_sha256": space_sha256(environment.observation_space),
        "action_space_sha256": space_sha256(environment.action_space),
        "integration_state_spec": "mjtState.mjSTATE_INTEGRATION",
        "integration_state_flag": int(mujoco.mjtState.mjSTATE_INTEGRATION),
        "integration_state_size": int(
            mujoco.mj_stateSize(physical.model, mujoco.mjtState.mjSTATE_INTEGRATION)
        ),
        "integration_state_components": _state_components(physical.model),
        "timestep_seconds": float(physical.model.opt.timestep),
        "frame_skip": int(physical.frame_skip),
        "control_period_seconds": float(physical.dt),
        "integrator": mujoco.mjtIntegrator(physical.model.opt.integrator).name,
        "solver": mujoco.mjtSolver(physical.model.opt.solver).name,
        "solver_iterations": int(physical.model.opt.iterations),
        "contact_capture_id": physical.contact_capture_id,
        "source_hashes": _source_hashes(),
        "thread_settings": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        },
        "cpu_only": True,
        "deterministic_actor_mean": True,
        "same_host_determinism_scope": "same_host_same_process_settings_bitwise/v1",
    }
    return validate_runtime_identity(runtime)


def _wrapper_layers(environment: object) -> tuple[object, object, object, object]:
    require_reference_wrapper_stack(environment)
    time_limit = environment
    order_enforcing = time_limit.env
    passive_checker = order_enforcing.env
    physical = passive_checker.env
    return time_limit, order_enforcing, passive_checker, physical


def capture_wrapper_state(environment: object) -> tuple[int, np.ndarray]:
    time_limit, order, checker, _physical = _wrapper_layers(environment)
    elapsed = getattr(time_limit, "_elapsed_steps", None)
    if type(elapsed) is not int or elapsed < 0:
        raise ReferenceCorpusContractError("TimeLimit counter is unavailable")
    raw_flags = (
        getattr(order, "_has_reset", None),
        getattr(checker, "checked_reset", None),
        getattr(checker, "checked_step", None),
        getattr(checker, "checked_render", None),
        getattr(checker, "close_called", None),
    )
    if any(type(value) is not bool for value in raw_flags):
        raise ReferenceCorpusContractError("unknown or missing wrapper-state flag")
    flags = np.asarray([int(value) for value in raw_flags], dtype="|u1")
    if flags.shape != (len(WRAPPER_FLAG_NAMES),):
        raise ReferenceCorpusContractError("wrapper-state flag width differs")
    return elapsed, flags


def restore_wrapper_state(environment: object, elapsed: int, flags: np.ndarray) -> None:
    time_limit, order, checker, _physical = _wrapper_layers(environment)
    value = np.asarray(flags)
    if (
        type(elapsed) is not int
        or not 0 <= elapsed <= 1000
        or value.dtype != np.dtype("|u1")
        or value.shape != (len(WRAPPER_FLAG_NAMES),)
        or np.any(value > 1)
    ):
        raise ReferenceCorpusContractError("unknown wrapper restoration state")
    time_limit._elapsed_steps = elapsed
    order._has_reset = bool(value[0])
    checker.checked_reset = bool(value[1])
    checker.checked_step = bool(value[2])
    checker.checked_render = bool(value[3])
    checker.close_called = bool(value[4])


def _rng_state(physical: object) -> dict[str, object]:
    generator = getattr(physical, "np_random", None)
    bit_generator = getattr(generator, "bit_generator", None)
    state = getattr(bit_generator, "state", None)
    if type(state) is not dict:
        raise ReferenceCorpusContractError("environment RNG state is unavailable")
    copied = copy.deepcopy(state)
    canonical_json_bytes(copied)
    return copied


def restore_rng_state(physical: object, state: Mapping[str, object]) -> None:
    if type(state) is not dict:
        raise ReferenceCorpusContractError("environment RNG restoration state is invalid")
    canonical_json_bytes(state)
    generator = getattr(physical, "np_random", None)
    bit_generator = getattr(generator, "bit_generator", None)
    if bit_generator is None:
        raise ReferenceCorpusContractError("environment RNG cannot be restored")
    try:
        bit_generator.state = copy.deepcopy(state)
    except (TypeError, ValueError) as exc:
        raise ReferenceCorpusContractError("environment RNG state is unknown") from exc


def _torso_up_z(physical: object) -> float:
    import mujoco

    torso_id = mujoco.mj_name2id(physical.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    if torso_id < 0:
        raise ReferenceCorpusContractError("Humanoid torso body is absent")
    value = float(np.asarray(physical.data.xmat[torso_id], dtype="<f8").reshape(3, 3)[2, 2])
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ReferenceCorpusContractError("torso up-axis value is invalid")
    return value


def _capture_boundary(
    environment: object,
    *,
    terminated: bool,
    truncated: bool,
) -> dict[str, object]:
    import mujoco

    _time_limit, _order, _checker, physical = _wrapper_layers(environment)
    mujoco.mj_forward(physical.model, physical.data)
    state = np.empty(STATE_SIZE, dtype=STATE_DTYPE)
    mujoco.mj_getState(
        physical.model,
        physical.data,
        state,
        mujoco.mjtState.mjSTATE_INTEGRATION,
    )
    elapsed, wrapper_flags = capture_wrapper_state(environment)
    observation = np.ascontiguousarray(physical._get_obs(), dtype="<f8").copy(order="C")
    cfrc_ext = np.ascontiguousarray(physical.data.cfrc_ext[1:], dtype="<f8").copy(order="C")
    if observation.shape != (OBSERVATION_WIDTH,) or cfrc_ext.shape != (13, 6):
        raise ReferenceCorpusContractError("boundary observation or force shape differs")
    if not np.array_equal(observation[-78:].reshape(13, 6), cfrc_ext):
        raise ReferenceCorpusContractError("boundary force sidecar differs from observation")
    return {
        "integration_state": state,
        "observation": observation,
        "cfrc_ext": cfrc_ext,
        "root_xy": np.ascontiguousarray(physical.data.qpos[0:2], dtype="<f8").copy(order="C"),
        "simulation_time": float(physical.data.time),
        "wrapper_elapsed": elapsed,
        "wrapper_flags": wrapper_flags,
        "result_flags": np.asarray([int(terminated), int(truncated)], dtype="|u1"),
        "torso_up_z": _torso_up_z(physical),
        "rng_state": _rng_state(physical),
    }


def _contact_record(environment: object) -> tuple[list[FullSubstepContactSample], bool]:
    physical = environment.unwrapped
    if physical.last_full_contact_substeps != 5:
        raise ReferenceCorpusContractError("collector omitted a MuJoCo physics substep")
    samples = list(physical.last_full_contact_samples)
    previous = (-1, -1)
    next_index = [0] * 5
    nonfoot = False
    for sample in samples:
        if type(sample) is not FullSubstepContactSample:
            raise ReferenceCorpusContractError("contact record type differs")
        key = (sample.physics_substep_index, sample.contact_index_within_substep)
        if (
            key <= previous
            or sample.contact_index_within_substep != next_index[sample.physics_substep_index]
        ):
            raise ReferenceCorpusContractError("contact sequence order differs")
        previous = key
        next_index[sample.physics_substep_index] += 1
        if (
            len(sample.geom1_name.encode()) > GEOM_NAME_BYTES
            or len(sample.geom2_name.encode()) > GEOM_NAME_BYTES
        ):
            raise ReferenceCorpusContractError("contact geometry name exceeds storage width")
        pair = (sample.geom1_name, sample.geom2_name)
        if FLOOR_GEOM_NAME in pair:
            other = pair[1] if pair[0] == FLOOR_GEOM_NAME else pair[0]
            if other not in FOOT_GEOM_NAMES:
                nonfoot = True
    return samples, nonfoot


def _arrays_from_records(
    boundaries: list[dict[str, object]],
    transitions: list[dict[str, object]],
    contacts_by_transition: list[list[FullSubstepContactSample]],
    *,
    steps: int,
    screen_canary: bool,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {
        "boundary_integration_state": np.ascontiguousarray(
            [item["integration_state"] for item in boundaries], dtype="<f8"
        ),
        "boundary_observation": np.ascontiguousarray(
            [item["observation"] for item in boundaries], dtype="<f8"
        ),
        "boundary_cfrc_ext": np.ascontiguousarray(
            [item["cfrc_ext"] for item in boundaries], dtype="<f8"
        ),
        "boundary_root_xy": np.ascontiguousarray(
            [item["root_xy"] for item in boundaries], dtype="<f8"
        ),
        "boundary_simulation_time": np.asarray(
            [item["simulation_time"] for item in boundaries], dtype="<f8"
        ),
        "boundary_wrapper_elapsed": np.asarray(
            [item["wrapper_elapsed"] for item in boundaries], dtype="<i8"
        ),
        "boundary_wrapper_flags": np.ascontiguousarray(
            [item["wrapper_flags"] for item in boundaries], dtype="|u1"
        ),
        "boundary_result_flags": np.ascontiguousarray(
            [item["result_flags"] for item in boundaries], dtype="|u1"
        ),
        "boundary_torso_up_z": np.asarray([item["torso_up_z"] for item in boundaries], dtype="<f8"),
        "transition_normalized_action": np.ascontiguousarray(
            [item["normalized_action"] for item in transitions], dtype="<f4"
        ),
        "transition_physical_action": np.ascontiguousarray(
            [item["physical_action"] for item in transitions], dtype="<f4"
        ),
        "transition_returned_observation": np.ascontiguousarray(
            [item["returned_observation"] for item in transitions], dtype="<f8"
        ),
        "transition_reward": np.asarray([item["reward"] for item in transitions], dtype="<f8"),
        "transition_flags": np.ascontiguousarray(
            [item["flags"] for item in transitions], dtype="|u1"
        ),
        "transition_next_wrapper_elapsed": np.asarray(
            [item["next_wrapper_elapsed"] for item in transitions], dtype="<i8"
        ),
        "transition_next_wrapper_flags": np.ascontiguousarray(
            [item["next_wrapper_flags"] for item in transitions], dtype="|u1"
        ),
        "transition_actor_input_sha256": np.asarray(
            [item["actor_input_sha256"] for item in transitions], dtype="|S64"
        ),
        "transition_actor_output_sha256": np.asarray(
            [item["actor_output_sha256"] for item in transitions], dtype="|S64"
        ),
        "transition_physical_action_sha256": np.asarray(
            [item["physical_action_sha256"] for item in transitions], dtype="|S64"
        ),
        "transition_returned_observation_sha256": np.asarray(
            [item["returned_observation_sha256"] for item in transitions], dtype="|S64"
        ),
        "transition_contact_substeps": np.full(steps, 5, dtype="<i8"),
        "transition_nonfoot_floor_contact": np.asarray(
            [item["nonfoot_floor_contact"] for item in transitions], dtype="|u1"
        ),
    }
    if screen_canary:
        result["canary_plain_reward"] = np.asarray(
            [item["plain_reward"] for item in transitions], dtype="<f8"
        )
        result["canary_plain_returned_observation_sha256"] = np.asarray(
            [item["plain_observation_sha256"] for item in transitions], dtype="|S64"
        )
    qpos = np.ascontiguousarray(result["boundary_integration_state"][:, 1:25], dtype="<f8")
    qvel = np.ascontiguousarray(result["boundary_integration_state"][:, 25:48], dtype="<f8")
    result["reference_rows"] = derive_reference_rows(qpos, qvel)
    window_indices, terminal_hold = reference_window_index_arrays(steps + 1)
    result["reference_window_indices"] = window_indices
    result["reference_window_terminal_hold"] = terminal_hold
    offsets = np.zeros(steps + 1, dtype="<i8")
    for index, contacts in enumerate(contacts_by_transition):
        offsets[index + 1] = offsets[index] + len(contacts)
    result["contact_offsets"] = offsets
    flat_contacts = [sample for samples in contacts_by_transition for sample in samples]
    result["contact_substep"] = np.asarray(
        [sample.physics_substep_index for sample in flat_contacts], dtype="<i8"
    )
    result["contact_index_within_substep"] = np.asarray(
        [sample.contact_index_within_substep for sample in flat_contacts], dtype="<i8"
    )
    result["contact_geom1_id"] = np.asarray(
        [sample.geom1_id for sample in flat_contacts], dtype="<i8"
    )
    result["contact_geom2_id"] = np.asarray(
        [sample.geom2_id for sample in flat_contacts], dtype="<i8"
    )
    result["contact_geom1_name"] = np.asarray(
        [sample.geom1_name.encode() for sample in flat_contacts], dtype=f"|S{GEOM_NAME_BYTES}"
    )
    result["contact_geom2_name"] = np.asarray(
        [sample.geom2_name.encode() for sample in flat_contacts], dtype=f"|S{GEOM_NAME_BYTES}"
    )
    result["contact_force_torque"] = np.ascontiguousarray(
        [sample.force_torque_contact_frame for sample in flat_contacts], dtype="<f8"
    ).reshape(len(flat_contacts), 6)
    boundaries_count = steps + 1
    result["boundary_integration_sha256"] = np.asarray(
        [
            array_sha256(result["boundary_integration_state"][index])
            for index in range(boundaries_count)
        ],
        dtype="|S64",
    )
    result["boundary_observation_sha256"] = np.asarray(
        [array_sha256(result["boundary_observation"][index]) for index in range(boundaries_count)],
        dtype="|S64",
    )
    result["boundary_cfrc_ext_sha256"] = np.asarray(
        [array_sha256(result["boundary_cfrc_ext"][index]) for index in range(boundaries_count)],
        dtype="|S64",
    )
    result["boundary_root_xy_sha256"] = np.asarray(
        [array_sha256(result["boundary_root_xy"][index]) for index in range(boundaries_count)],
        dtype="|S64",
    )
    result["boundary_reference_sha256"] = np.asarray(
        [array_sha256(result["reference_rows"][index]) for index in range(boundaries_count)],
        dtype="|S64",
    )
    rng_hashes = [sha256_json(item["rng_state"]) for item in boundaries]
    result["boundary_rng_state_sha256"] = np.asarray(rng_hashes, dtype="|S64")
    result["transition_contact_sequence_sha256"] = np.asarray(
        [contact_sequence_sha256(result, index) for index in range(steps)], dtype="|S64"
    )
    result["boundary_record_sha256"] = np.asarray(
        [boundary_record_sha256(result, index) for index in range(boundaries_count)], dtype="|S64"
    )
    result["transition_record_sha256"] = np.asarray(
        [transition_record_sha256(result, index) for index in range(steps)], dtype="|S64"
    )
    return result


def _plain_boundary(environment: object) -> tuple[np.ndarray, np.ndarray, float]:
    physical = environment.unwrapped
    return (
        np.ascontiguousarray(physical.data.qpos, dtype="<f8").copy(order="C"),
        np.ascontiguousarray(physical.data.qvel, dtype="<f8").copy(order="C"),
        float(physical.data.time),
    )


def _plain_boundaries_equal(
    left: tuple[np.ndarray, np.ndarray, float],
    right: tuple[np.ndarray, np.ndarray, float],
) -> bool:
    return (
        np.array_equal(left[0], right[0])
        and np.array_equal(left[1], right[1])
        and struct.pack(">d", left[2]) == struct.pack(">d", right[2])
    )


def collect_reference_clip(
    *,
    clip_id: str,
    clip_kind: str,
    actor_variant: str,
    actor_npz_path: Path,
    actor_identity: Mapping[str, object],
    seed: int,
    reset_order: int,
    steps: int = 1000,
    compare_plain_rewards: bool = False,
) -> CollectedClip:
    """Roll out one frozen actor once; no retry or checkpoint loader exists here."""

    if type(clip_id) is not str or not clip_id or len(clip_id) > 128:
        raise ReferenceCorpusContractError("clip_id is invalid")
    if clip_kind not in {"corpus", "development_screen"}:
        raise ReferenceCorpusContractError("clip_kind is invalid")
    if actor_variant not in {"expert", "medium", "simple"}:
        raise ReferenceCorpusContractError("actor variant is invalid")
    if clip_kind == "development_screen" and actor_variant != "expert":
        raise ReferenceCorpusContractError("development screen requires the expert actor")
    if type(seed) is not int or seed < 0 or type(reset_order) is not int or reset_order < 0:
        raise ReferenceCorpusContractError("seed and reset order must be non-negative integers")
    if type(steps) is not int or not 1 <= steps <= 1000:
        raise ReferenceCorpusContractError("steps must be in [1, 1000]")
    if compare_plain_rewards is not (clip_kind == "development_screen"):
        raise ReferenceCorpusContractError("plain reward canary is required only for screen clips")
    identity = dict(actor_identity)
    if identity.get("variant") != actor_variant:
        raise ReferenceCorpusContractError("actor identity variant differs")
    if identity.get("inference_id") != INFERENCE_ID:
        raise ReferenceCorpusContractError("actor inference identity differs")
    npz_sha256 = identity.get("npz_sha256")
    if type(npz_sha256) is not str:
        raise ReferenceCorpusContractError("actor NPZ identity is absent")
    actor = StrictTQCActorRuntime.from_npz(actor_npz_path, expected_sha256=npz_sha256)
    if actor.loaded_actor.state_sha256 != identity.get("actor_state_sha256"):
        raise ReferenceCorpusContractError("actor state identity differs")
    if actor.loaded_actor.schema_sha256 != identity.get("actor_schema_sha256"):
        raise ReferenceCorpusContractError("actor schema identity differs")
    environment = make_reference_corpus_env()
    plain_environment = make_humanoid_env() if compare_plain_rewards else None
    boundaries: list[dict[str, object]] = []
    transitions: list[dict[str, object]] = []
    contacts_by_transition: list[list[FullSubstepContactSample]] = []
    try:
        reset_observation, _reset_info = environment.reset(seed=seed)
        if plain_environment is not None:
            plain_observation, _plain_info = plain_environment.reset(seed=seed)
            if not np.array_equal(reset_observation, plain_observation):
                raise ReferenceCorpusContractError("plain/instrumented reset observation differs")
            if not _plain_boundaries_equal(
                _plain_boundary(plain_environment), _plain_boundary(environment)
            ):
                raise ReferenceCorpusContractError("plain/instrumented reset state differs")
        runtime = inspect_reference_corpus_runtime(environment)
        boundaries.append(_capture_boundary(environment, terminated=False, truncated=False))
        if not np.array_equal(
            np.asarray(reset_observation, dtype="<f8"), boundaries[0]["observation"]
        ):
            raise ReferenceCorpusContractError("reset return differs from canonical observation")
        initial_rng = boundaries[0]["rng_state"]
        for index in range(steps):
            action = actor.act(boundaries[-1]["observation"])
            plain_reward: float | None = None
            plain_observation_hash: str | None = None
            if plain_environment is not None:
                before_plain = _plain_boundary(plain_environment)
                before_instrumented = _plain_boundary(environment)
                if not _plain_boundaries_equal(before_plain, before_instrumented):
                    raise ReferenceCorpusContractError("plain/instrumented pre-step state differs")
                plain_observation, raw_plain_reward, plain_terminated, plain_truncated, _ = (
                    plain_environment.step(action.physical_action.copy())
                )
                plain_reward = float(raw_plain_reward)
                plain_observation_hash = array_sha256(
                    np.ascontiguousarray(plain_observation, dtype="<f8")
                )
            returned_observation, reward, terminated, truncated, _info = environment.step(
                action.physical_action.copy()
            )
            contact_samples, nonfoot = _contact_record(environment)
            if plain_environment is not None:
                if (
                    not np.array_equal(returned_observation, plain_observation)
                    or struct.pack(">d", float(reward)) != struct.pack(">d", plain_reward)
                    or bool(terminated) is not bool(plain_terminated)
                    or bool(truncated) is not bool(plain_truncated)
                ):
                    raise ReferenceCorpusContractError("plain/instrumented reward canary differs")
                after_plain = _plain_boundary(plain_environment)
                after_instrumented = _plain_boundary(environment)
                if not _plain_boundaries_equal(after_plain, after_instrumented):
                    raise ReferenceCorpusContractError("plain/instrumented post-step state differs")
            next_boundary = _capture_boundary(
                environment,
                terminated=bool(terminated),
                truncated=bool(truncated),
            )
            if index < steps - 1 and (terminated or truncated):
                raise ReferenceCorpusContractError("clip ended before its declared horizon")
            if steps == 1000 and index == 999 and (terminated or not truncated):
                raise ReferenceCorpusContractError(
                    "full clip must truncate, not terminate, at 1000"
                )
            next_elapsed, next_flags = capture_wrapper_state(environment)
            transition: dict[str, object] = {
                "normalized_action": action.normalized_output,
                "physical_action": action.physical_action,
                "returned_observation": np.ascontiguousarray(returned_observation, dtype="<f8"),
                "reward": float(reward),
                "flags": np.asarray([int(terminated), int(truncated)], dtype="|u1"),
                "next_wrapper_elapsed": next_elapsed,
                "next_wrapper_flags": next_flags,
                "actor_input_sha256": action.actor_input_sha256,
                "actor_output_sha256": action.actor_output_sha256,
                "physical_action_sha256": action.physical_action_sha256,
                "returned_observation_sha256": array_sha256(
                    np.ascontiguousarray(returned_observation, dtype="<f8")
                ),
                "nonfoot_floor_contact": nonfoot,
            }
            if plain_environment is not None:
                transition["plain_reward"] = plain_reward
                transition["plain_observation_sha256"] = plain_observation_hash
            transitions.append(transition)
            contacts_by_transition.append(contact_samples)
            boundaries.append(next_boundary)
        if any(
            sha256_json(boundary["rng_state"]) != sha256_json(initial_rng)
            for boundary in boundaries
        ):
            raise ReferenceCorpusContractError(
                "environment RNG state changed during deterministic rollout"
            )
        arrays = _arrays_from_records(
            boundaries,
            transitions,
            contacts_by_transition,
            steps=steps,
            screen_canary=compare_plain_rewards,
        )
        validate_clip_arrays(arrays, steps=steps, screen_canary=compare_plain_rewards)
        reward_canary: dict[str, object]
        if compare_plain_rewards:
            reward_canary = {
                "status": "passed",
                "role": "plain_vs_instrumented_equivalence_only",
                "locomotion_metrics_may_read_reward": False,
                "instrumented_reward_sha256": array_sha256(arrays["transition_reward"]),
                "plain_reward_sha256": array_sha256(arrays["canary_plain_reward"]),
                "visual_capture": "disabled_by_external_screen_design/v1",
            }
        else:
            reward_canary = {
                "status": "not_applicable",
                "role": "replay_canary_only",
                "locomotion_metrics_may_read_reward": False,
            }
        return CollectedClip(
            clip_id=clip_id,
            clip_kind=clip_kind,
            actor_variant=actor_variant,
            seed=seed,
            reset_order=reset_order,
            steps=steps,
            arrays=arrays,
            runtime_identity=runtime,
            rng_state=copy.deepcopy(initial_rng),
            actor_identity=identity,
            reward_canary=reward_canary,
            collector_pid=os.getpid(),
        )
    finally:
        environment.close()
        if plain_environment is not None:
            plain_environment.close()


def clip_array_bindings(clip: CollectedClip) -> list[dict[str, object]]:
    return array_bindings(clip.arrays)


__all__ = [
    "COLLECTOR_ID",
    "CollectedClip",
    "ReferenceCorpusContractError",
    "capture_wrapper_state",
    "clip_array_bindings",
    "collect_reference_clip",
    "inspect_reference_corpus_runtime",
    "restore_rng_state",
    "restore_wrapper_state",
]
