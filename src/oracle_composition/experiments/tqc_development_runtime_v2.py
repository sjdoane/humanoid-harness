"""Exact host and simulator reinspection for the TQC v2 worker."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from pathlib import Path

import numpy as np

from .fixed_reference import ExperimentContractError, sha256_file
from .runtime_identity import (
    dependency_lock_path,
    module_sha256,
    source_tree_sha256,
    space_sha256,
)
from .tqc_calibration_contract import canonical_json
from .tqc_development_contract_v2 import LoadedTQCDevelopmentDesignV2

RUNTIME_RECEIPT_ID = "tqc_dev_1m_v2_host_runtime/v1"
MAX_RUNTIME_RECEIPT_BYTES = 512 * 1024
ATTEMPT_ID = "dev1m-v2-seed-95001-attempt-01"
EXPECTED_RESET_OBSERVATION_SHA256 = (
    "2957e81d460feb085acc20c02b94444ab74a6eb47890be67182b27e3c329f99d"
)
_RUNTIME_ISSUER = object()
_BINDING_KEYS = {
    "attempt_id",
    "attempt_nonce",
    "worker_pid",
    "worker_process_start_monotonic_seconds",
    "preflight_contract_sha256",
    "claimed_work_directory_identity",
    "project_python_source_tree_sha256",
}
_RUNTIME_RECEIPT_KEYS = {
    "schema_version",
    "runtime_receipt_id",
    "design_file_sha256",
    "design_semantic_sha256",
    "training_projection_sha256",
    "observed_runtime_requirements",
    "observed_simulator",
    "observed_environment",
    "python_executable_realpath",
    "python_executable_file_sha256",
    "project_python_source_tree_sha256",
    "runtime_reinspection_source_sha256",
    *_BINDING_KEYS,
    "reward_or_info_fields_read",
    "behavioral_evidence",
}


def _require_exact(value: object, expected: object, *, field: str) -> None:
    if type(value) is not type(expected):
        raise ExperimentContractError(f"runtime {field} differs")
    if isinstance(expected, dict):
        if set(value) != set(expected):  # type: ignore[arg-type]
            raise ExperimentContractError(f"runtime {field} keys differ")
        for key, child in expected.items():
            _require_exact(value[key], child, field=f"{field}.{key}")  # type: ignore[index]
        return
    if isinstance(expected, (list, tuple)):
        if len(value) != len(expected):  # type: ignore[arg-type]
            raise ExperimentContractError(f"runtime {field} length differs")
        for index, child in enumerate(expected):
            _require_exact(value[index], child, field=f"{field}[{index}]")  # type: ignore[index]
        return
    if value != expected:
        raise ExperimentContractError(f"runtime {field} differs")


def _require_sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"runtime {field} must be a lowercase SHA-256")
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ExperimentContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ExperimentContractError(f"non-finite JSON value {value!r}")


def _decode_runtime_receipt(encoded: bytes) -> dict[str, object]:
    if type(encoded) is not bytes or not encoded or len(encoded) > MAX_RUNTIME_RECEIPT_BYTES:
        raise ExperimentContractError("v2 runtime receipt bytes are absent or too large")
    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except ExperimentContractError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ExperimentContractError("v2 runtime receipt bytes are invalid") from exc
    if type(value) is not dict or canonical_json(value) != encoded:
        raise ExperimentContractError("v2 runtime receipt bytes are not canonical")
    return value


def _validate_binding(
    value: Mapping[str, object],
    *,
    expected: Mapping[str, object] | None,
    require_current_worker: bool,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != _BINDING_KEYS:
        raise ExperimentContractError("runtime receipt binding keys differ")
    _require_exact(value["attempt_id"], ATTEMPT_ID, field="attempt id")
    for field_name in (
        "attempt_nonce",
        "preflight_contract_sha256",
        "claimed_work_directory_identity",
        "project_python_source_tree_sha256",
    ):
        _require_sha256(value[field_name], field=field_name)
    worker_pid = value["worker_pid"]
    worker_start = value["worker_process_start_monotonic_seconds"]
    if type(worker_pid) is not int or worker_pid <= 1:
        raise ExperimentContractError("runtime worker PID is invalid")
    if type(worker_start) is not float or not math.isfinite(worker_start) or worker_start < 0:
        raise ExperimentContractError("runtime worker process start is invalid")
    if require_current_worker and worker_pid != os.getpid():
        raise ExperimentContractError("runtime receipt worker PID is not this process")
    if require_current_worker and worker_start > time.perf_counter():
        raise ExperimentContractError("runtime worker process start is in the future")
    if expected is not None:
        _require_exact(dict(value), dict(expected), field="preflight binding")
    return dict(value)


def _validate_runtime_payload(
    value: dict[str, object],
    design: LoadedTQCDevelopmentDesignV2 | None,
    expected_binding: Mapping[str, object] | None = None,
) -> None:
    if set(value) != _RUNTIME_RECEIPT_KEYS:
        raise ExperimentContractError("v2 runtime receipt keys differ")
    for field_name, expected in (
        ("schema_version", 1),
        ("runtime_receipt_id", RUNTIME_RECEIPT_ID),
        ("reward_or_info_fields_read", False),
        ("behavioral_evidence", False),
    ):
        _require_exact(value[field_name], expected, field=field_name)
    for field_name in (
        "design_file_sha256",
        "design_semantic_sha256",
        "training_projection_sha256",
        "python_executable_file_sha256",
        "project_python_source_tree_sha256",
        "runtime_reinspection_source_sha256",
    ):
        _require_sha256(value[field_name], field=field_name)
    _require_exact(
        value["runtime_reinspection_source_sha256"],
        sha256_file(Path(__file__)),
        field="reinspection source SHA-256",
    )
    _validate_binding(
        {field_name: value[field_name] for field_name in _BINDING_KEYS},
        expected=expected_binding,
        require_current_worker=False,
    )
    executable = value["python_executable_realpath"]
    if type(executable) is not str or not executable or not Path(executable).is_absolute():
        raise ExperimentContractError("runtime Python executable path is invalid")
    if type(value["observed_runtime_requirements"]) is not dict:
        raise ExperimentContractError("runtime requirement observations are invalid")
    if type(value["observed_simulator"]) is not dict:
        raise ExperimentContractError("runtime simulator observations are invalid")
    environment = value["observed_environment"]
    if type(environment) is not dict:
        raise ExperimentContractError("runtime environment observations are invalid")
    _require_exact(
        environment.get("reset_observation_sha256"),
        EXPECTED_RESET_OBSERVATION_SHA256,
        field="reset observation",
    )
    if design is None:
        return
    if type(design) is not LoadedTQCDevelopmentDesignV2:
        raise ExperimentContractError("runtime receipt requires a strict-loaded design")
    for field_name, expected in (
        ("design_file_sha256", design.file_sha256),
        ("design_semantic_sha256", design.semantic_sha256),
        ("training_projection_sha256", design.training_projection_sha256),
    ):
        _require_exact(value[field_name], expected, field=field_name)
    projection = design.to_dict()["training_projection"]
    _require_exact(
        value["observed_runtime_requirements"],
        projection["runtime_requirements"],
        field="observed requirements",
    )
    _require_exact(value["observed_simulator"], projection["simulator"], field="observed simulator")
    environment_contract = projection["environment"]
    expected_environment = {
        "environment_id": environment_contract["environment_id"],
        "plain_wrapper_types": environment_contract["wrapper_order_outer_to_inner"],
        "instrumented_wrapper_types": [
            *environment_contract["wrapper_order_outer_to_inner"][:-1],
            "oracle_composition.envs.humanoid.SubstepContactHumanoidEnv",
        ],
        "max_episode_steps": environment_contract["max_episode_steps"],
        "terminate_when_unhealthy": environment_contract["terminate_when_unhealthy"],
        "reset_noise_scale": environment_contract["reset_noise_scale"],
        "exclude_current_positions_from_observation": environment_contract[
            "exclude_current_positions_from_observation"
        ],
        "frame_skip": environment_contract["frame_skip"],
        "control_period_seconds": environment_contract["control_period_seconds"],
        "observation_shape": environment_contract["observation_shape"],
        "observation_dtype": environment_contract["observation_dtype"],
        "action_shape": environment_contract["action_shape"],
        "action_dtype": environment_contract["action_dtype"],
        "render_mode": environment_contract["render_mode"],
        "reset_observation_sha256": environment["reset_observation_sha256"],
    }
    _require_exact(environment, expected_environment, field="observed environment")


def _wrapper_types(environment: object) -> tuple[str, ...]:
    result: list[str] = []
    current = environment
    while True:
        result.append(f"{type(current).__module__}.{type(current).__name__}")
        if not hasattr(current, "env"):
            return tuple(result)
        current = current.env


def _host_hardware_identity() -> dict[str, object]:
    cpu_model = platform.processor().strip()
    hardware_model: str | None = None
    total_memory_bytes: int | None = None
    if sys.platform == "darwin":
        for field, key in (
            ("cpu", "machdep.cpu.brand_string"),
            ("hardware", "hw.model"),
            ("memory", "hw.memsize"),
        ):
            try:
                value = subprocess.run(
                    ["/usr/sbin/sysctl", "-n", key],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                value = ""
            if field == "cpu" and value:
                cpu_model = value
            elif field == "hardware" and value:
                hardware_model = value
            elif field == "memory" and value.isdigit():
                total_memory_bytes = int(value)
    if total_memory_bytes is None:
        try:
            total_memory_bytes = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
        except (OSError, TypeError, ValueError):
            total_memory_bytes = None
    return {
        "cpu_model": cpu_model or None,
        "hardware_model": hardware_model,
        "logical_cpu_count": os.cpu_count(),
        "total_memory_bytes": total_memory_bytes,
    }


def _mujoco_option_receipt(model: object, mujoco: object) -> dict[str, object]:
    option = model.opt
    disable_actuator = int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    return {
        "physics_timestep_seconds": float(option.timestep),
        "integrator": str(mujoco.mjtIntegrator(option.integrator).name),
        "cone": str(mujoco.mjtCone(option.cone).name),
        "jacobian": str(mujoco.mjtJacobian(option.jacobian).name),
        "solver": str(mujoco.mjtSolver(option.solver).name),
        "iterations": int(option.iterations),
        "line_search_iterations": int(option.ls_iterations),
        "noslip_iterations": int(option.noslip_iterations),
        "sdf_iterations": int(option.sdf_iterations),
        "sdf_initial_points": int(option.sdf_initpoints),
        "solver_tolerance": float(option.tolerance),
        "line_search_tolerance": float(option.ls_tolerance),
        "noslip_tolerance": float(option.noslip_tolerance),
        "gravity_m_s2": [float(value) for value in option.gravity],
        "wind_m_s": [float(value) for value in option.wind],
        "magnetic": [float(value) for value in option.magnetic],
        "density": float(option.density),
        "viscosity": float(option.viscosity),
        "impedance_ratio": float(option.impratio),
        "override_margin": float(option.o_margin),
        "override_solref": [float(value) for value in option.o_solref],
        "override_solimp": [float(value) for value in option.o_solimp],
        "override_friction": [float(value) for value in option.o_friction],
        "ccd_iterations": int(option.ccd_iterations),
        "ccd_tolerance": float(option.ccd_tolerance),
        "disable_actuator": int(bool(int(option.disableflags) & disable_actuator)),
        "sleep_tolerance": float(option.sleep_tolerance),
        "disable_flags": int(option.disableflags),
        "enable_flags": int(option.enableflags),
    }


@dataclass(frozen=True, slots=True)
class TQCHostRuntimeReceiptV2:
    """Validated runtime content; it never authorizes execution or behavior."""

    canonical_bytes: bytes = field(repr=False)
    sha256: str
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _RUNTIME_ISSUER:
            raise ExperimentContractError("v2 runtime receipts may only be issued by reinspection")
        value = _decode_runtime_receipt(self.canonical_bytes)
        _validate_runtime_payload(value, None)
        _require_sha256(self.sha256, field="receipt SHA-256")
        if hashlib.sha256(self.canonical_bytes).hexdigest() != self.sha256:
            raise ExperimentContractError("v2 runtime receipt SHA-256 differs")

    def to_dict(self) -> dict[str, object]:
        return _decode_runtime_receipt(self.canonical_bytes)

    @property
    def authorizes_execution(self) -> bool:
        return False


def validate_tqc_development_runtime_receipt_v2(
    encoded_bytes: bytes,
    design: LoadedTQCDevelopmentDesignV2,
    *,
    expected_binding: Mapping[str, object],
) -> TQCHostRuntimeReceiptV2:
    """Validate worker runtime content against an independently known binding."""

    if type(design) is not LoadedTQCDevelopmentDesignV2:
        raise ExperimentContractError("runtime receipt requires a strict-loaded design")
    value = _decode_runtime_receipt(encoded_bytes)
    _validate_runtime_payload(value, design, expected_binding)
    return TQCHostRuntimeReceiptV2(
        canonical_bytes=encoded_bytes,
        sha256=hashlib.sha256(encoded_bytes).hexdigest(),
        _issuer=_RUNTIME_ISSUER,
    )


def inspect_tqc_development_runtime_v2(
    design: LoadedTQCDevelopmentDesignV2,
    *,
    receipt_binding: Mapping[str, object],
) -> TQCHostRuntimeReceiptV2:
    """Reinspect every pinned dependency, environment, and simulator fact."""

    if type(design) is not LoadedTQCDevelopmentDesignV2:
        raise ExperimentContractError("v2 runtime inspection requires a strict-loaded design")
    binding = _validate_binding(
        receipt_binding,
        expected=None,
        require_current_worker=True,
    )
    raw = design.to_dict()["training_projection"]
    required = raw["runtime_requirements"]
    environment_contract = raw["environment"]
    simulator_contract = raw["simulator"]
    try:
        import gymnasium
        import mujoco
        import PIL
        import sb3_contrib
        import stable_baselines3
        import torch
        from gymnasium.envs import registration
        from gymnasium.envs.mujoco import humanoid_v5
        from gymnasium.wrappers import common as wrappers_common
        from sb3_contrib.common import utils as sb3_contrib_common_utils
        from sb3_contrib.tqc import policies as tqc_policies
        from sb3_contrib.tqc import tqc as tqc_algorithm
        from stable_baselines3.common import (
            base_class,
            buffers,
            distributions,
            noise,
            off_policy_algorithm,
            policies,
            save_util,
            torch_layers,
            type_aliases,
            utils,
        )
        from stable_baselines3.common import logger as sb3_logger
        from stable_baselines3.common.vec_env import base_vec_env, dummy_vec_env
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise ExperimentContractError(
            "install the gym and train extras for v2 reinspection"
        ) from exc

    from oracle_composition.envs import humanoid as humanoid_module
    from oracle_composition.envs.humanoid import HumanoidExperimentConfig, make_humanoid_env

    adam_module = importlib.import_module("torch.optim.adam")
    optimizer_module = importlib.import_module("torch.optim.optimizer")
    renderer_module = importlib.import_module("mujoco.rendering.classic.renderer")
    webp_plugin = importlib.import_module("PIL.WebPImagePlugin")
    webp_binary = importlib.import_module("PIL._webp")
    config = HumanoidExperimentConfig(
        env_id="Humanoid-v5",
        terminate_when_unhealthy=False,
        reset_noise_scale=0.01,
        exclude_current_positions_from_observation=True,
        frame_skip=5,
    )
    plain = gymnasium.make("Humanoid-v5", render_mode=None, **config.gym_kwargs())
    instrumented = make_humanoid_env(config, capture_substep_contacts=True)
    try:
        plain_observation, _plain_info = plain.reset(seed=98_001)
        instrumented_observation, _instrumented_info = instrumented.reset(seed=98_001)
        if not np.array_equal(plain_observation, instrumented_observation):
            raise ExperimentContractError("plain and instrumented reset observations differ")
        expected_plain_wrappers = tuple(environment_contract["wrapper_order_outer_to_inner"])
        expected_instrumented_wrappers = (
            *expected_plain_wrappers[:-1],
            ("oracle_composition.envs.humanoid.SubstepContactHumanoidEnv"),
        )
        if _wrapper_types(plain) != expected_plain_wrappers:
            raise ExperimentContractError("plain Humanoid wrapper stack differs")
        if _wrapper_types(instrumented) != expected_instrumented_wrappers:
            raise ExperimentContractError("instrumented Humanoid wrapper stack differs")
        if type(plain.unwrapped) is not humanoid_v5.HumanoidEnv:
            raise ExperimentContractError("plain Humanoid environment class differs")
        if not isinstance(instrumented.unwrapped, humanoid_v5.HumanoidEnv):
            raise ExperimentContractError("instrumented Humanoid base class differs")
        physical = plain.unwrapped
        model_path = Path(str(getattr(physical, "fullpath", "")))
        if model_path.is_symlink() or not model_path.is_file():
            raise ExperimentContractError("Humanoid model XML must be one regular file")

        versions_and_host: dict[str, object] = {
            "python_version": platform.python_version(),
            "platform_system": platform.system(),
            "platform_machine": platform.machine(),
            "platform_release": platform.release(),
            **_host_hardware_identity(),
            "torch_intraop_thread_count": int(torch.get_num_threads()),
            "torch_interop_thread_count": int(torch.get_num_interop_threads()),
            "numpy_version": str(np.__version__),
            "torch_version": str(torch.__version__),
            "gymnasium_version": str(gymnasium.__version__),
            "mujoco_version": str(mujoco.__version__),
            "stable_baselines3_version": str(stable_baselines3.__version__),
            "sb3_contrib_version": str(sb3_contrib.__version__),
            "pillow_version": str(PIL.__version__),
            "libwebp_version": str(webp_binary.webpdecoder_version),
        }
        sources: dict[str, object] = {
            "dependency_lock_sha256": sha256_file(dependency_lock_path()),
            "mujoco_model_sha256": sha256_file(model_path),
            "observation_space_sha256": space_sha256(plain.observation_space),
            "action_space_sha256": space_sha256(plain.action_space),
            "environment_source_sha256": module_sha256(humanoid_module),
            "gym_humanoid_source_sha256": module_sha256(humanoid_v5),
            "gym_mujoco_env_source_sha256": module_sha256(
                importlib.import_module("gymnasium.envs.mujoco.mujoco_env")
            ),
            "gym_registration_source_sha256": module_sha256(registration),
            "gym_wrappers_source_sha256": module_sha256(wrappers_common),
            "tqc_algorithm_source_sha256": module_sha256(tqc_algorithm),
            "tqc_policy_source_sha256": module_sha256(tqc_policies),
            "torch_adam_source_sha256": module_sha256(adam_module),
            "torch_optimizer_base_source_sha256": module_sha256(optimizer_module),
            "sb3_base_algorithm_source_sha256": module_sha256(base_class),
            "sb3_save_util_source_sha256": module_sha256(save_util),
            "sb3_off_policy_source_sha256": module_sha256(off_policy_algorithm),
            "sb3_replay_buffer_source_sha256": module_sha256(buffers),
            "sb3_dummy_vec_env_source_sha256": module_sha256(dummy_vec_env),
            "sb3_distributions_source_sha256": module_sha256(distributions),
            "sb3_policies_source_sha256": module_sha256(policies),
            "sb3_torch_layers_source_sha256": module_sha256(torch_layers),
            "sb3_utils_source_sha256": module_sha256(utils),
            "sb3_noise_source_sha256": module_sha256(noise),
            "sb3_logger_source_sha256": module_sha256(sb3_logger),
            "sb3_callbacks_source_sha256": module_sha256(
                importlib.import_module("stable_baselines3.common.callbacks")
            ),
            "sb3_type_aliases_source_sha256": module_sha256(type_aliases),
            "sb3_contrib_common_utils_source_sha256": module_sha256(sb3_contrib_common_utils),
            "sb3_base_vec_env_source_sha256": module_sha256(base_vec_env),
            "mujoco_renderer_source_sha256": module_sha256(renderer_module),
            "pillow_webp_plugin_source_sha256": module_sha256(webp_plugin),
            "pillow_webp_binary_sha256": sha256_file(Path(str(webp_binary.__file__))),
        }
        observed = {**versions_and_host, **sources}
        if set(observed) != set(required):
            raise ExperimentContractError("v2 runtime requirement coverage differs")
        for field, expected in required.items():
            _require_exact(observed[field], expected, field=field)

        simulator = {
            "mujoco_version": observed["mujoco_version"],
            "model_sha256": sources["mujoco_model_sha256"],
            "nq": int(physical.model.nq),
            "nv": int(physical.model.nv),
            "nu": int(physical.model.nu),
            "na": int(physical.model.na),
            "nbody": int(physical.model.nbody),
            "ngeom": int(physical.model.ngeom),
            **_mujoco_option_receipt(physical.model, mujoco),
            "physics_substeps_per_control": int(physical.frame_skip),
            "contact_capture_id": "none_in_plain_training_environment/v1",
        }
        if set(simulator) != set(simulator_contract):
            raise ExperimentContractError("v2 simulator option coverage differs")
        for field, expected in simulator_contract.items():
            _require_exact(simulator[field], expected, field=f"simulator.{field}")

        environment = {
            "environment_id": str(plain.spec.id),
            "plain_wrapper_types": list(_wrapper_types(plain)),
            "instrumented_wrapper_types": list(_wrapper_types(instrumented)),
            "max_episode_steps": int(plain._max_episode_steps),
            "terminate_when_unhealthy": bool(physical._terminate_when_unhealthy),
            "reset_noise_scale": float(physical._reset_noise_scale),
            "exclude_current_positions_from_observation": bool(
                physical._exclude_current_positions_from_observation
            ),
            "frame_skip": int(physical.frame_skip),
            "control_period_seconds": float(physical.dt),
            "observation_shape": list(plain.observation_space.shape),
            "observation_dtype": np.dtype(plain.observation_space.dtype).str,
            "action_shape": list(plain.action_space.shape),
            "action_dtype": np.dtype(plain.action_space.dtype).str,
            "render_mode": plain.render_mode,
            "reset_observation_sha256": hashlib.sha256(
                np.ascontiguousarray(plain_observation).tobytes(order="C")
            ).hexdigest(),
        }
        expected_environment = {
            "environment_id": environment_contract["environment_id"],
            "max_episode_steps": environment_contract["max_episode_steps"],
            "terminate_when_unhealthy": environment_contract["terminate_when_unhealthy"],
            "reset_noise_scale": environment_contract["reset_noise_scale"],
            "exclude_current_positions_from_observation": environment_contract[
                "exclude_current_positions_from_observation"
            ],
            "frame_skip": environment_contract["frame_skip"],
            "control_period_seconds": environment_contract["control_period_seconds"],
            "observation_shape": environment_contract["observation_shape"],
            "observation_dtype": environment_contract["observation_dtype"],
            "action_shape": environment_contract["action_shape"],
            "action_dtype": environment_contract["action_dtype"],
            "render_mode": environment_contract["render_mode"],
        }
        for field, expected in expected_environment.items():
            _require_exact(environment[field], expected, field=f"environment.{field}")

        executable = Path(sys.executable).resolve(strict=True)
        observed_source_tree_sha256 = source_tree_sha256()
        _require_exact(
            observed_source_tree_sha256,
            binding["project_python_source_tree_sha256"],
            field="project Python source tree SHA-256",
        )
        payload = {
            "schema_version": 1,
            "runtime_receipt_id": RUNTIME_RECEIPT_ID,
            "design_file_sha256": design.file_sha256,
            "design_semantic_sha256": design.semantic_sha256,
            "training_projection_sha256": design.training_projection_sha256,
            "observed_runtime_requirements": observed,
            "observed_simulator": simulator,
            "observed_environment": environment,
            "python_executable_realpath": str(executable),
            "python_executable_file_sha256": sha256_file(executable),
            "project_python_source_tree_sha256": observed_source_tree_sha256,
            "runtime_reinspection_source_sha256": sha256_file(Path(__file__)),
            **binding,
            "reward_or_info_fields_read": False,
            "behavioral_evidence": False,
        }
        encoded = canonical_json(payload)
        return TQCHostRuntimeReceiptV2(
            canonical_bytes=encoded,
            sha256=hashlib.sha256(encoded).hexdigest(),
            _issuer=_RUNTIME_ISSUER,
        )
    finally:
        plain.close()
        instrumented.close()


__all__ = [
    "ATTEMPT_ID",
    "EXPECTED_RESET_OBSERVATION_SHA256",
    "MAX_RUNTIME_RECEIPT_BYTES",
    "RUNTIME_RECEIPT_ID",
    "TQCHostRuntimeReceiptV2",
    "inspect_tqc_development_runtime_v2",
    "validate_tqc_development_runtime_receipt_v2",
]
