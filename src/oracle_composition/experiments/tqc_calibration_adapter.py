"""Gymnasium, SB3, and host-resource adapter for TQC calibration."""

from __future__ import annotations

import hashlib
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from oracle_composition.envs import humanoid as humanoid_module
from oracle_composition.envs.humanoid import HumanoidExperimentConfig, make_humanoid_env

from .fixed_reference import ExperimentContractError, sha256_file
from .runtime_identity import (
    dependency_lock_path,
    module_sha256,
    source_tree_sha256,
    space_sha256,
)
from .tqc_calibration_contract import TQCCalibrationDesign, canonical_json, sha256_json

REQUIRED_TRAINING_SIGNALS = frozenset(
    {
        "train/actor_loss",
        "train/critic_loss",
        "train/ent_coef",
        "train/ent_coef_loss",
        "train/n_updates",
    }
)
HUMANOID_ACTION_LOW = -0.4
HUMANOID_ACTION_HIGH = 0.4
NORMALIZED_ACTION_LOW = -1.0
NORMALIZED_ACTION_HIGH = 1.0


def _environment_config(design: TQCCalibrationDesign) -> HumanoidExperimentConfig:
    environment = design.environment
    return HumanoidExperimentConfig(
        env_id=environment.environment_id,
        terminate_when_unhealthy=environment.terminate_when_unhealthy,
        reset_noise_scale=environment.reset_noise_scale,
        exclude_current_positions_from_observation=(
            environment.exclude_current_positions_from_observation
        ),
        frame_skip=environment.frame_skip,
    )


def calibration_source_sha256() -> str:
    return sha256_file(Path(__file__).with_name("tqc_calibration.py"))


def _host_hardware_receipt() -> dict[str, Any]:
    cpu_model = platform.processor().strip()
    hardware_model: str | None = None
    if sys.platform == "darwin":
        for field, name in (
            ("cpu_model", "machdep.cpu.brand_string"),
            ("hardware_model", "hw.model"),
        ):
            try:
                observed = subprocess.run(
                    ["/usr/sbin/sysctl", "-n", name],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                observed = ""
            if field == "cpu_model" and observed:
                cpu_model = observed
            elif field == "hardware_model" and observed:
                hardware_model = observed
    elif not cpu_model:
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    cpu_model = line.partition(":")[2].strip()
                    break
        except OSError:
            pass
    logical_cpu_count = os.cpu_count()
    try:
        total_memory_bytes = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, TypeError, ValueError):
        total_memory_bytes = 0
    return {
        "cpu_model": cpu_model or None,
        "hardware_model": hardware_model,
        "logical_cpu_count": logical_cpu_count,
        "total_memory_bytes": total_memory_bytes or None,
    }


def inspect_tqc_runtime(design: TQCCalibrationDesign) -> dict[str, Any]:
    try:
        import gymnasium
        import mujoco
        import sb3_contrib
        import stable_baselines3
        import torch
        from gymnasium.envs.mujoco import humanoid_v5
        from sb3_contrib.tqc import policies as tqc_policies
        from sb3_contrib.tqc import tqc as tqc_algorithm
        from stable_baselines3.common import buffers, off_policy_algorithm
        from stable_baselines3.common import logger as sb3_logger
        from stable_baselines3.common.vec_env import base_vec_env, dummy_vec_env
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise ExperimentContractError("install the gym and train extras") from exc
    if str(sb3_contrib.__version__) != design.tqc.required_sb3_contrib_version:
        raise ExperimentContractError("installed sb3-contrib version differs from the design")
    if str(stable_baselines3.__version__) != design.tqc.required_stable_baselines3_version:
        raise ExperimentContractError("installed stable-baselines3 version differs from the design")

    config = _environment_config(design)
    environment = make_humanoid_env(config)
    try:
        observation, _info = environment.reset(seed=design.seed)
        unwrapped = environment.unwrapped
        model_path = Path(str(getattr(unwrapped, "fullpath", "")))
        if not model_path.is_file() or model_path.is_symlink():
            raise ExperimentContractError("Humanoid model must be a regular file")
        if not np.isfinite(observation).all():
            raise ExperimentContractError("Humanoid reset observation is non-finite")
        if tuple(observation.shape) != tuple(environment.observation_space.shape):
            raise ExperimentContractError("Humanoid reset observation shape drifted")
        expected_low = np.full((17,), HUMANOID_ACTION_LOW, dtype=environment.action_space.dtype)
        expected_high = np.full((17,), HUMANOID_ACTION_HIGH, dtype=environment.action_space.dtype)
        if not np.array_equal(environment.action_space.low, expected_low) or not np.array_equal(
            environment.action_space.high, expected_high
        ):
            raise ExperimentContractError("Humanoid action bounds differ from [-0.4, 0.4]")
        max_steps = getattr(environment, "_max_episode_steps", None)
        if max_steps != 1000:
            raise ExperimentContractError("Humanoid TimeLimit must be exactly 1000 steps")
        runtime = {
            "python_version": platform.python_version(),
            "platform_system": platform.system(),
            "platform_release": platform.release(),
            "platform_machine": platform.machine(),
            "host_hardware": _host_hardware_receipt(),
            "gymnasium_version": str(gymnasium.__version__),
            "mujoco_version": str(mujoco.__version__),
            "numpy_version": str(np.__version__),
            "stable_baselines3_version": str(stable_baselines3.__version__),
            "sb3_contrib_version": str(sb3_contrib.__version__),
            "torch_version": str(torch.__version__),
            "torch_intraop_thread_count": int(torch.get_num_threads()),
            "torch_interop_thread_count": int(torch.get_num_interop_threads()),
            "environment_id": str(unwrapped.spec.id),
            "environment_config": asdict(config),
            "environment_max_episode_steps": int(max_steps),
            "control_period_seconds": float(unwrapped.dt),
            "observation_shape": list(environment.observation_space.shape),
            "observation_dtype": np.dtype(environment.observation_space.dtype).str,
            "action_shape": list(environment.action_space.shape),
            "action_dtype": np.dtype(environment.action_space.dtype).str,
            "qpos_shape": list(unwrapped.data.qpos.shape),
            "qvel_shape": list(unwrapped.data.qvel.shape),
            "model_sha256": sha256_file(model_path),
            "dependency_lock_sha256": sha256_file(dependency_lock_path()),
            "source_tree_sha256": source_tree_sha256(),
            "observation_space_sha256": space_sha256(environment.observation_space),
            "action_space_sha256": space_sha256(environment.action_space),
            "environment_source_sha256": module_sha256(humanoid_module),
            "gym_humanoid_source_sha256": module_sha256(humanoid_v5),
            "tqc_algorithm_source_sha256": module_sha256(tqc_algorithm),
            "tqc_policy_source_sha256": module_sha256(tqc_policies),
            "sb3_off_policy_source_sha256": module_sha256(off_policy_algorithm),
            "sb3_replay_buffer_source_sha256": module_sha256(buffers),
            "sb3_base_vec_env_source_sha256": module_sha256(base_vec_env),
            "sb3_dummy_vec_env_source_sha256": module_sha256(dummy_vec_env),
            "sb3_logger_source_sha256": module_sha256(sb3_logger),
            "calibration_source_sha256": calibration_source_sha256(),
        }
        runtime["runtime_sha256"] = sha256_json(runtime)
        return runtime
    finally:
        environment.close()


def verify_runtime_integrity(runtime: Mapping[str, Any], design: TQCCalibrationDesign) -> None:
    if dict(runtime) != inspect_tqc_runtime(design):
        raise ExperimentContractError("runtime integrity changed during calibration")


def make_tqc_vector_env(design: TQCCalibrationDesign) -> object:
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise ExperimentContractError("install the train extra") from exc

    config = _environment_config(design)

    def factory() -> object:
        return make_humanoid_env(config)

    environment = DummyVecEnv([factory for _ in range(design.environment.n_envs)])
    observed_worker_seeds = tuple(environment.seed(design.seed))
    if observed_worker_seeds != design.environment.worker_seeds:
        environment.close()
        raise ExperimentContractError("DummyVecEnv worker seeds differ from the design")
    if environment.num_envs != design.environment.n_envs:
        environment.close()
        raise ExperimentContractError("DummyVecEnv worker count differs from the design")
    return environment


def make_tqc_model(design: TQCCalibrationDesign, environment: object) -> object:
    try:
        from sb3_contrib import TQC
        from stable_baselines3.common.logger import Logger
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise ExperimentContractError("install the train extra") from exc
    model = TQC(
        design.tqc.policy,
        environment,
        learning_rate=design.tqc.learning_rate,
        buffer_size=design.tqc.buffer_size,
        learning_starts=design.tqc.learning_starts,
        batch_size=design.tqc.batch_size,
        tau=design.tqc.tau,
        gamma=design.tqc.gamma,
        train_freq=(design.tqc.train_freq_steps, "step"),
        gradient_steps=design.tqc.gradient_steps,
        replay_buffer_kwargs={
            "handle_timeout_termination": design.tqc.handle_timeout_termination,
        },
        optimize_memory_usage=design.tqc.optimize_memory_usage,
        n_steps=design.tqc.n_steps,
        ent_coef=design.tqc.ent_coef,
        target_entropy=design.tqc.target_entropy,
        top_quantiles_to_drop_per_net=design.tqc.top_quantiles_to_drop_per_net,
        use_sde=design.tqc.use_sde,
        tensorboard_log=None,
        policy_kwargs={
            "net_arch": list(design.tqc.policy_hidden_layers),
            "n_quantiles": design.tqc.n_quantiles,
            "n_critics": design.tqc.n_critics,
        },
        verbose=0,
        seed=design.seed,
        device=design.tqc.device,
    )
    model.set_logger(Logger(folder=None, output_formats=[]))
    return model


def process_peak_rss_bytes() -> int:
    observed = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    multiplier = 1 if sys.platform == "darwin" else 1024
    return int(observed) * multiplier


def _nearest_existing_directory(path: Path) -> Path:
    candidate = Path(path).absolute()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise ExperimentContractError("cannot locate an existing output ancestor")
        candidate = candidate.parent
    if not candidate.is_dir() or candidate.is_symlink():
        raise ExperimentContractError("output ancestor must be a real directory")
    return candidate


def free_disk_bytes(path: Path) -> int:
    return int(shutil.disk_usage(_nearest_existing_directory(path)).free)


def _finite_array(value: object, *, field: str) -> None:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentContractError(f"{field} is not numeric") from exc
    if array.dtype.kind not in "biufc" or not np.isfinite(array).all():
        raise ExperimentContractError(f"{field} contains non-finite or non-numeric values")


def training_signal_snapshot(model: object) -> dict[str, float]:
    logger = getattr(model, "logger", None)
    values = getattr(logger, "name_to_value", {})
    if not isinstance(values, dict):
        raise ExperimentContractError("TQC logger signals are unavailable")
    snapshot: dict[str, float] = {}
    for key in sorted(REQUIRED_TRAINING_SIGNALS & set(values)):
        value = values[key]
        if not isinstance(value, (int, float, np.number)):
            raise ExperimentContractError(f"training signal {key} is not scalar")
        number = float(value)
        if not math.isfinite(number):
            raise ExperimentContractError(f"training signal {key} is non-finite")
        snapshot[key] = number
    return snapshot


def assert_finite_parameters(model: object) -> int:
    policy = getattr(model, "policy", None)
    named_parameters = getattr(policy, "named_parameters", None)
    if not callable(named_parameters):
        raise ExperimentContractError("TQC policy parameters are unavailable")
    count = 0
    for name, parameter in named_parameters():
        count += int(parameter.numel())
        if not bool(parameter.detach().isfinite().all().item()):
            raise ExperimentContractError(f"TQC parameter {name} is non-finite")
    entropy = getattr(model, "log_ent_coef", None)
    if entropy is not None and not bool(entropy.detach().isfinite().all().item()):
        raise ExperimentContractError("TQC entropy parameter is non-finite")
    if count < 1:
        raise ExperimentContractError("TQC policy has no parameters")
    return count


def parameter_receipt(model: object) -> dict[str, Any]:
    policy = getattr(model, "policy", None)
    named_parameters = getattr(policy, "named_parameters", None)
    if not callable(named_parameters):
        raise ExperimentContractError("TQC policy parameters are unavailable")
    structure: list[dict[str, Any]] = []
    content = hashlib.sha256()
    for name, parameter in named_parameters():
        array = parameter.detach().cpu().numpy()
        metadata = {
            "name": name,
            "shape": list(array.shape),
            "dtype": np.dtype(array.dtype).str,
        }
        encoded = canonical_json(metadata)
        structure.append(metadata)
        content.update(len(encoded).to_bytes(8, "big"))
        content.update(encoded)
        raw = np.ascontiguousarray(array).tobytes()
        content.update(len(raw).to_bytes(8, "big"))
        content.update(raw)
    entropy = getattr(model, "log_ent_coef", None)
    if entropy is not None:
        array = entropy.detach().cpu().numpy()
        metadata = {
            "name": "algorithm.log_ent_coef",
            "shape": list(array.shape),
            "dtype": np.dtype(array.dtype).str,
        }
        encoded = canonical_json(metadata)
        structure.append(metadata)
        content.update(len(encoded).to_bytes(8, "big"))
        content.update(encoded)
        raw = np.ascontiguousarray(array).tobytes()
        content.update(len(raw).to_bytes(8, "big"))
        content.update(raw)
    if not structure:
        raise ExperimentContractError("TQC parameter receipt is empty")
    return {
        "structure_sha256": sha256_json(structure),
        "value_sha256": content.hexdigest(),
    }


def _replay_arrays(model: object) -> list[tuple[str, np.ndarray]]:
    replay = getattr(model, "replay_buffer", None)
    if replay is None:
        raise ExperimentContractError("TQC replay buffer was not allocated")
    arrays: list[tuple[str, np.ndarray]] = []
    for field in (
        "observations",
        "next_observations",
        "actions",
        "rewards",
        "dones",
        "timeouts",
    ):
        value = getattr(replay, field, None)
        if not isinstance(value, np.ndarray):
            raise ExperimentContractError(f"TQC replay buffer field {field} is unavailable")
        arrays.append((field, value))
    return arrays


def replay_buffer_allocation_bytes(model: object) -> int:
    total = sum(int(value.nbytes) for _field, value in _replay_arrays(model))
    if total < 1:
        raise ExperimentContractError("TQC replay buffer allocation is empty")
    return total


def inspect_tqc_model(model: object, design: TQCCalibrationDesign) -> dict[str, Any]:
    replay = getattr(model, "replay_buffer", None)
    policy = getattr(model, "policy", None)
    critic = getattr(model, "critic", None)
    train_frequency = getattr(model, "train_freq", None)
    get_environment = getattr(model, "get_env", None)
    model_environment = get_environment() if callable(get_environment) else None
    logger = getattr(model, "logger", None)
    unit = getattr(getattr(train_frequency, "unit", None), "value", None)
    observation_space = getattr(model, "observation_space", None)
    action_space = getattr(model, "action_space", None)
    capacity = design.tqc.buffer_size
    expected_replay_bytes = (
        2
        * capacity
        * math.prod(observation_space.shape)
        * np.dtype(observation_space.dtype).itemsize
        + capacity * math.prod(action_space.shape) * np.dtype(action_space.dtype).itemsize
        + capacity * 3 * np.dtype(np.float32).itemsize
    )
    receipt = {
        "algorithm_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "policy_class": f"{type(policy).__module__}.{type(policy).__qualname__}",
        "vector_environment_class": (
            f"{type(model_environment).__module__}.{type(model_environment).__qualname__}"
        ),
        "vec_normalize_active": getattr(model, "_vec_normalize_env", None) is not None,
        "custom_logger_active": getattr(model, "_custom_logger", None),
        "logger_directory": getattr(logger, "dir", None),
        "logger_output_format_count": len(getattr(logger, "output_formats", [])),
        "device": str(getattr(model, "device", "")),
        "n_envs": getattr(model, "n_envs", None),
        "buffer_size": getattr(model, "buffer_size", None),
        "learning_starts": getattr(model, "learning_starts", None),
        "batch_size": getattr(model, "batch_size", None),
        "tau": getattr(model, "tau", None),
        "gamma": getattr(model, "gamma", None),
        "gradient_steps": getattr(model, "gradient_steps", None),
        "n_steps": getattr(model, "n_steps", None),
        "train_frequency": getattr(train_frequency, "frequency", None),
        "train_frequency_unit": unit,
        "entropy_coefficient": getattr(model, "ent_coef", None),
        "resolved_target_entropy": float(getattr(model, "target_entropy", math.nan)),
        "top_quantiles_to_drop_per_net": getattr(model, "top_quantiles_to_drop_per_net", None),
        "use_sde": getattr(model, "use_sde", None),
        "action_noise": getattr(model, "action_noise", None),
        "policy_net_arch": list(getattr(policy, "net_arch", [])),
        "critic_n_quantiles": getattr(critic, "n_quantiles", None),
        "critic_n_critics": getattr(critic, "n_critics", None),
        "observation_space_sha256": space_sha256(observation_space),
        "action_space_sha256": space_sha256(action_space),
        "replay_buffer_size_per_environment": getattr(replay, "buffer_size", None),
        "replay_n_envs": getattr(replay, "n_envs", None),
        "replay_total_transition_slots": (
            getattr(replay, "buffer_size", -1) * getattr(replay, "n_envs", -1)
        ),
        "pending_worker_seeds": list(getattr(model_environment, "_seeds", [])),
        "replay_optimize_memory_usage": getattr(replay, "optimize_memory_usage", None),
        "replay_handle_timeout_termination": getattr(replay, "handle_timeout_termination", None),
        "replay_allocation_bytes": replay_buffer_allocation_bytes(model),
        "expected_replay_allocation_bytes": expected_replay_bytes,
        "initial_parameters": parameter_receipt(model),
    }
    expected = {
        "algorithm_class": "sb3_contrib.tqc.tqc.TQC",
        "policy_class": "sb3_contrib.tqc.policies.TQCPolicy",
        "vector_environment_class": ("stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv"),
        "vec_normalize_active": False,
        "custom_logger_active": True,
        "logger_directory": None,
        "logger_output_format_count": 0,
        "device": "cpu",
        "n_envs": design.environment.n_envs,
        "buffer_size": design.tqc.buffer_size,
        "learning_starts": design.tqc.learning_starts,
        "batch_size": design.tqc.batch_size,
        "tau": design.tqc.tau,
        "gamma": design.tqc.gamma,
        "gradient_steps": design.tqc.gradient_steps,
        "n_steps": design.tqc.n_steps,
        "train_frequency": design.tqc.train_freq_steps,
        "train_frequency_unit": "step",
        "entropy_coefficient": design.tqc.ent_coef,
        "resolved_target_entropy": -17.0,
        "top_quantiles_to_drop_per_net": design.tqc.top_quantiles_to_drop_per_net,
        "use_sde": design.tqc.use_sde,
        "action_noise": None,
        "policy_net_arch": list(design.tqc.policy_hidden_layers),
        "critic_n_quantiles": design.tqc.n_quantiles,
        "critic_n_critics": design.tqc.n_critics,
        "replay_buffer_size_per_environment": (design.tqc.buffer_size // design.environment.n_envs),
        "replay_n_envs": design.environment.n_envs,
        "replay_total_transition_slots": design.tqc.buffer_size,
        "pending_worker_seeds": list(design.environment.worker_seeds),
        "replay_optimize_memory_usage": design.tqc.optimize_memory_usage,
        "replay_handle_timeout_termination": design.tqc.handle_timeout_termination,
        "replay_allocation_bytes": expected_replay_bytes,
        "expected_replay_allocation_bytes": expected_replay_bytes,
    }
    mismatches = [field for field, value in expected.items() if receipt[field] != value]
    if mismatches:
        raise ExperimentContractError(
            "observed TQC model differs from design: " + ", ".join(mismatches)
        )
    receipt["model_semantic_sha256"] = sha256_json(receipt)
    return receipt


def fault_replay_buffer_pages(
    model: object,
    *,
    resource_check: Callable[[], None],
) -> dict[str, Any]:
    replay = getattr(model, "replay_buffer", None)
    before_position = getattr(replay, "pos", None)
    before_full = getattr(replay, "full", None)
    if before_position != 0 or before_full is not False:
        raise ExperimentContractError("replay buffer must be empty before the residency fault")
    page_size = int(resource.getpagesize())
    if page_size < 1:
        raise ExperimentContractError("operating-system page size is invalid")
    array_receipts: list[dict[str, Any]] = []
    for field, array in _replay_arrays(model):
        if not array.flags.c_contiguous or not array.flags.writeable:
            raise ExperimentContractError(f"replay array {field} must be writable C-contiguous")
        byte_view = array.view(np.uint8).reshape(-1)
        address = int(array.__array_interface__["data"][0])
        pages = (address % page_size + array.nbytes + page_size - 1) // page_size
        check_stride = page_size * 4096
        for chunk_start in range(0, array.nbytes, check_stride):
            chunk_end = min(array.nbytes, chunk_start + check_stride)
            byte_view[chunk_start:chunk_end:page_size] = 0
            resource_check()
        if array.nbytes:
            byte_view[-1] = 0
        resource_check()
        array_receipts.append(
            {
                "field": field,
                "byte_count": int(array.nbytes),
                "backing_page_count": int(pages),
            }
        )
    if (
        getattr(replay, "pos", None) != before_position
        or getattr(replay, "full", None) is not False
    ):
        raise ExperimentContractError("replay residency fault changed active buffer state")
    return {
        "method": "write_zero_every_os_page_plus_final_byte/v1",
        "os_page_size_bytes": page_size,
        "resource_check_interval_pages": 4096,
        "buffer_position_before": before_position,
        "buffer_position_after": getattr(replay, "pos", None),
        "buffer_full_before": before_full,
        "buffer_full_after": getattr(replay, "full", None),
        "arrays": array_receipts,
        "total_array_bytes": sum(item["byte_count"] for item in array_receipts),
    }


def make_finite_resource_callback(
    *,
    design: TQCCalibrationDesign,
    output_parent: Path,
    peak_rss_reader: Callable[[], int],
    free_disk_reader: Callable[[Path], int],
    clock: Callable[[], float],
) -> object:
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise ExperimentContractError("install the train extra") from exc

    class FiniteResourceCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.signal_checks = 0
            self.parameter_checks = 0
            self.resource_checks = 0
            self.parameter_count = 0
            self.maximum_peak_rss_bytes = 0
            self.minimum_free_disk_bytes: int | None = None
            self.training_signals: dict[str, float] = {}
            self.throughput_windows: list[dict[str, float | int]] = []
            self.throughput_window_start_time: float | None = None
            self.throughput_window_start_step: int | None = None

        def check_resources(self, *, include_disk: bool) -> None:
            peak = peak_rss_reader()
            if not isinstance(peak, int) or isinstance(peak, bool) or peak < 0:
                raise ExperimentContractError("peak RSS reader returned invalid bytes")
            self.maximum_peak_rss_bytes = max(self.maximum_peak_rss_bytes, peak)
            if peak > design.resource_gates.sampled_peak_rss_failure_threshold_bytes:
                raise ExperimentContractError(
                    "sampled peak RSS exceeded the calibration failure threshold"
                )
            if include_disk:
                disk = free_disk_reader(output_parent)
                if not isinstance(disk, int) or isinstance(disk, bool) or disk < 0:
                    raise ExperimentContractError("free-disk reader returned invalid bytes")
                self.minimum_free_disk_bytes = (
                    disk
                    if self.minimum_free_disk_bytes is None
                    else min(self.minimum_free_disk_bytes, disk)
                )
                if disk < design.resource_gates.minimum_free_disk_bytes:
                    raise ExperimentContractError("free disk fell below the calibration hard stop")
            self.resource_checks += 1

        def check_parameters(self) -> None:
            self.parameter_count = assert_finite_parameters(self.model)
            self.parameter_checks += 1

        def _on_step(self) -> bool:
            for field in ("new_obs", "rewards", "dones"):
                if field not in self.locals:
                    raise ExperimentContractError(f"TQC callback omitted {field}")
                _finite_array(self.locals[field], field=f"rollout.{field}")
            expected_shape = (design.environment.n_envs, 17)
            for field, low, high in (
                ("actions", HUMANOID_ACTION_LOW, HUMANOID_ACTION_HIGH),
                ("buffer_actions", NORMALIZED_ACTION_LOW, NORMALIZED_ACTION_HIGH),
            ):
                if field not in self.locals:
                    raise ExperimentContractError(f"TQC callback omitted {field}")
                array = np.asarray(self.locals[field])
                _finite_array(array, field=f"rollout.{field}")
                if array.shape != expected_shape or np.any(array < low) or np.any(array > high):
                    raise ExperimentContractError(
                        f"rollout.{field} escaped its exact [{low}, {high}] envelope"
                    )
            self.signal_checks += 1
            self.training_signals.update(training_signal_snapshot(self.model))
            include_disk = self.n_calls == 1 or self.n_calls % 100 == 0
            self.check_resources(include_disk=include_disk)
            if self.n_calls == 1 or self.n_calls % 1000 == 0:
                self.check_parameters()
            environment_steps = int(self.model.num_timesteps)
            warmup = design.resource_gates.throughput_warmup_environment_steps
            window = design.resource_gates.throughput_window_environment_steps
            if environment_steps == warmup:
                self.throughput_window_start_step = environment_steps
                self.throughput_window_start_time = clock()
            elif (
                self.throughput_window_start_step is not None
                and self.throughput_window_start_time is not None
                and environment_steps - self.throughput_window_start_step == window
            ):
                ended = clock()
                elapsed = ended - self.throughput_window_start_time
                if not math.isfinite(elapsed) or elapsed <= 0.0:
                    raise ExperimentContractError("throughput-window elapsed time is invalid")
                rate = window / elapsed
                self.throughput_windows.append(
                    {
                        "start_environment_step": self.throughput_window_start_step,
                        "end_environment_step": environment_steps,
                        "wall_seconds": elapsed,
                        "environment_steps_per_second": rate,
                    }
                )
                if rate < design.resource_gates.minimum_environment_steps_per_second:
                    raise ExperimentContractError(
                        "sustained environment-step throughput missed the calibration hard stop"
                    )
                self.throughput_window_start_step = environment_steps
                self.throughput_window_start_time = ended
            return True

    return FiniteResourceCallback()


__all__ = [
    "HUMANOID_ACTION_HIGH",
    "HUMANOID_ACTION_LOW",
    "NORMALIZED_ACTION_HIGH",
    "NORMALIZED_ACTION_LOW",
    "REQUIRED_TRAINING_SIGNALS",
    "assert_finite_parameters",
    "calibration_source_sha256",
    "fault_replay_buffer_pages",
    "free_disk_bytes",
    "inspect_tqc_model",
    "inspect_tqc_runtime",
    "make_finite_resource_callback",
    "make_tqc_model",
    "make_tqc_vector_env",
    "parameter_receipt",
    "process_peak_rss_bytes",
    "replay_buffer_allocation_bytes",
    "training_signal_snapshot",
    "verify_runtime_integrity",
]
