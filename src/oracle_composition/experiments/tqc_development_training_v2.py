"""Exact TQC construction and training checks for the frozen v2 screen."""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from typing import Any

import numpy as np

from oracle_composition.envs.humanoid import HumanoidExperimentConfig, make_humanoid_env

from .fixed_reference import ExperimentContractError, sha256_file
from .runtime_identity import space_sha256
from .tqc_calibration_contract import canonical_json
from .tqc_development_contract_v2 import (
    DESIGN_FILE_SHA256,
    DESIGN_SEMANTIC_SHA256,
    EXPECTED_GRADIENT_UPDATES,
    EXPECTED_PARAMETER_CHECKS,
    EXPECTED_VECTOR_STEPS,
    MODEL_SEED,
    TOTAL_ENVIRONMENT_STEPS,
    TRAINING_PROJECTION_SHA256,
    WORKER_SEEDS,
    LoadedTQCDevelopmentDesignV2,
)
from .tqc_development_resource_v2 import (
    EXPECTED_THROUGHPUT_WINDOWS,
    TQCResourceMonitorV2,
    TQCTrainingResourceAuthorityV2,
)

TRAINING_INTEGRITY_ID = "tqc_dev_1m_v2_training_integrity/v1"
EXPECTED_ROLLOUT_CHECKS = EXPECTED_VECTOR_STEPS
EXPECTED_OPTIMIZER_CHECKS = 201
EXPECTED_TRAINING_SCALAR_CHECKS = EXPECTED_GRADIENT_UPDATES
EXPECTED_REPLAY_ALLOCATION_BYTES = 5_648_000_000

_ENVIRONMENT_ISSUER = object()
_CONSTRUCTION_ISSUER = object()
_COMPLETION_ISSUER = object()
_CALLBACK_ISSUER = object()
_MODEL_HANDLE_ISSUER = object()


@dataclass(slots=True)
class _TQCLiveModelHandleV2:
    """Transfer one live model from construction through strict persistence."""

    _model: object | None = field(repr=False)
    _phase: str
    _creator_pid: int
    _completion_seal: bytes | None = field(default=None, repr=False)
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _MODEL_HANDLE_ISSUER:
            raise ExperimentContractError("TQC model handles may only be issued by construction")
        if (
            self._model is None
            or self._phase != "constructed"
            or self._creator_pid != os.getpid()
            or self._completion_seal is not None
        ):
            raise ExperimentContractError("initial TQC model handle state is invalid")

    def _assert_process(self) -> None:
        if os.getpid() != self._creator_pid:
            raise ExperimentContractError("TQC model handle crossed its worker process boundary")

    def constructed_model(self) -> object:
        self._assert_process()
        if self._phase != "constructed" or self._model is None:
            raise ExperimentContractError("constructed TQC model ownership was transferred")
        return self._model

    def seal_completion(self, payload: Mapping[str, object]) -> None:
        self._assert_process()
        if self._phase != "constructed" or self._model is None:
            raise ExperimentContractError("TQC model completion transfer is not available")
        encoded = canonical_json(dict(payload))
        if not encoded:
            raise ExperimentContractError("TQC training completion seal is empty")
        self._completion_seal = encoded
        self._phase = "completed"

    def completed_model(self, payload: Mapping[str, object]) -> object:
        self._assert_process()
        if (
            self._phase != "completed"
            or self._model is None
            or self._completion_seal != canonical_json(dict(payload))
        ):
            raise ExperimentContractError("TQC training completion seal differs")
        return self._model

    def consume_completed_model(self, payload: Mapping[str, object]) -> object:
        model = self.completed_model(payload)
        self._model = None
        self._phase = "consumed"
        return model


def _require_sha256(value: object, *, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"{field_name} must be a lowercase SHA-256")
    return value


def _require_loaded_design(value: object) -> LoadedTQCDevelopmentDesignV2:
    if type(value) is not LoadedTQCDevelopmentDesignV2:
        raise ExperimentContractError("training requires the strict-loaded v2 design")
    return value


def _training_projection(loaded: LoadedTQCDevelopmentDesignV2) -> dict[str, Any]:
    design = _require_loaded_design(loaded).to_dict()
    projection = design.get("training_projection")
    if type(projection) is not dict:
        raise ExperimentContractError("v2 training projection is unavailable")
    return projection


def _require_execution_manifest(
    manifest: object,
    loaded: LoadedTQCDevelopmentDesignV2,
) -> object:
    try:
        from .tqc_development_manifest_v2 import ValidatedTQCExecutionManifestV2
    except ImportError as exc:  # pragma: no cover - implementation ordering boundary
        raise ExperimentContractError("v2 execution-manifest authority is unavailable") from exc

    if type(manifest) is not ValidatedTQCExecutionManifestV2:
        raise ExperimentContractError(
            "model construction requires a validated v2 execution manifest"
        )
    if (
        manifest.model_construction_allowed is not True
        or manifest.authorizes_training is not True
        or manifest.behavioral_evidence is not False
    ):
        raise ExperimentContractError("execution manifest admission flags differ")
    expected = {
        "design_file_sha256": loaded.file_sha256,
        "design_semantic_sha256": loaded.semantic_sha256,
        "training_projection_sha256": loaded.training_projection_sha256,
    }
    for name, value in expected.items():
        if getattr(manifest, name, None) != value:
            raise ExperimentContractError(f"execution manifest {name} differs")
    _require_sha256(getattr(manifest, "sha256", None), field_name="execution manifest SHA-256")
    attempt_id = getattr(manifest, "attempt_id", None)
    if type(attempt_id) is not str or not attempt_id.strip() or len(attempt_id) > 128:
        raise ExperimentContractError("execution manifest attempt ID is invalid")
    return manifest


def _qualified_type(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _wrapper_stack(environment: object) -> tuple[str, ...]:
    result: list[str] = []
    current = environment
    while True:
        result.append(_qualified_type(current))
        unwrapped = getattr(current, "unwrapped", None)
        if current is unwrapped:
            return tuple(result)
        if not hasattr(current, "env"):
            raise ExperimentContractError("Humanoid wrapper chain is incomplete")
        current = current.env


def _require_exact_numeric_array(
    observed: object,
    expected: object,
    *,
    field_name: str,
) -> None:
    value = np.asarray(observed)
    reference = np.asarray(expected)
    if value.shape != reference.shape or not np.array_equal(value, reference):
        raise ExperimentContractError(f"{field_name} differs")


def _validate_humanoid_constructor(
    worker: object,
    unwrapped: object,
    environment_design: Mapping[str, Any],
) -> None:
    spec = getattr(worker, "spec", None)
    expected_spec_kwargs = {
        "render_mode": environment_design["render_mode"],
        "terminate_when_unhealthy": environment_design["terminate_when_unhealthy"],
        "reset_noise_scale": environment_design["reset_noise_scale"],
        "exclude_current_positions_from_observation": environment_design[
            "exclude_current_positions_from_observation"
        ],
        "frame_skip": environment_design["frame_skip"],
    }
    if (
        spec is None
        or spec.id != environment_design["environment_id"]
        or spec.max_episode_steps != environment_design["max_episode_steps"]
        or spec.order_enforce is not True
        or spec.disable_env_checker is not environment_design["disable_env_checker"]
        or spec.kwargs != expected_spec_kwargs
        or getattr(worker.env, "_disable_render_order_enforcing", None)
        is not environment_design["order_enforcing_disable_render_order_enforcing"]
    ):
        raise ExperimentContractError("registered Humanoid construction differs")

    args = getattr(unwrapped, "_ezpickle_args", None)
    kwargs = getattr(unwrapped, "_ezpickle_kwargs", None)
    if type(args) is not tuple or len(args) != 16 or kwargs != {"render_mode": None}:
        raise ExperimentContractError("Humanoid constructor receipt differs")
    expected_args = (
        environment_design["xml_file"],
        environment_design["frame_skip"],
        environment_design["default_camera_config"],
        environment_design["forward_reward_weight"],
        environment_design["ctrl_cost_weight"],
        environment_design["contact_cost_weight"],
        (-math.inf, environment_design["contact_cost_range"][1]),
        environment_design["healthy_reward"],
        environment_design["terminate_when_unhealthy"],
        tuple(environment_design["healthy_z_range"]),
        environment_design["reset_noise_scale"],
        environment_design["exclude_current_positions_from_observation"],
        environment_design["include_cinert_in_observation"],
        environment_design["include_cvel_in_observation"],
        environment_design["include_qfrc_actuator_in_observation"],
        environment_design["include_cfrc_ext_in_observation"],
    )
    for index, (observed, expected) in enumerate(zip(args, expected_args, strict=True)):
        if index == 2:
            if type(observed) is not dict or set(observed) != set(expected):
                raise ExperimentContractError("Humanoid default camera construction differs")
            for name, expected_value in expected.items():
                if name == "lookat":
                    _require_exact_numeric_array(
                        observed[name],
                        expected_value,
                        field_name="Humanoid default camera lookat",
                    )
                elif observed[name] != expected_value:
                    raise ExperimentContractError(f"Humanoid default camera {name} differs")
        elif observed != expected:
            raise ExperimentContractError(f"Humanoid constructor argument {index} differs")

    renderer = getattr(unwrapped, "mujoco_renderer", None)
    if (
        renderer is None
        or renderer.model is not unwrapped.model
        or renderer.data is not unwrapped.data
        or renderer.width != environment_design["mujoco_env_width"]
        or renderer.height != environment_design["mujoco_env_height"]
        or renderer.max_geom != environment_design["mujoco_env_max_geom"]
        or renderer._vopt != {}
    ):
        raise ExperimentContractError("Humanoid renderer construction differs")


def _validate_mujoco_model(
    unwrapped: object,
    simulator_design: Mapping[str, Any],
) -> None:
    import mujoco

    if mujoco.__version__ != simulator_design["mujoco_version"]:
        raise ExperimentContractError("MuJoCo version differs")
    model = unwrapped.model
    option = model.opt
    observed_scalars = {
        "ccd_iterations": option.ccd_iterations,
        "ccd_tolerance": option.ccd_tolerance,
        "cone": mujoco.mjtCone(option.cone).name,
        "density": option.density,
        "disable_actuator": option.disableactuator,
        "disable_flags": option.disableflags,
        "enable_flags": option.enableflags,
        "impedance_ratio": option.impratio,
        "integrator": mujoco.mjtIntegrator(option.integrator).name,
        "iterations": option.iterations,
        "jacobian": mujoco.mjtJacobian(option.jacobian).name,
        "line_search_iterations": option.ls_iterations,
        "line_search_tolerance": option.ls_tolerance,
        "na": model.na,
        "nbody": model.nbody,
        "ngeom": model.ngeom,
        "noslip_iterations": option.noslip_iterations,
        "noslip_tolerance": option.noslip_tolerance,
        "nq": model.nq,
        "nu": model.nu,
        "nv": model.nv,
        "override_margin": option.o_margin,
        "physics_substeps_per_control": unwrapped.frame_skip,
        "physics_timestep_seconds": option.timestep,
        "sdf_initial_points": option.sdf_initpoints,
        "sdf_iterations": option.sdf_iterations,
        "sleep_tolerance": option.sleep_tolerance,
        "solver": mujoco.mjtSolver(option.solver).name,
        "solver_tolerance": option.tolerance,
        "viscosity": option.viscosity,
    }
    for name, observed in observed_scalars.items():
        if observed != simulator_design[name]:
            raise ExperimentContractError(f"MuJoCo {name} differs")
    for name, observed in (
        ("gravity_m_s2", option.gravity),
        ("magnetic", option.magnetic),
        ("override_friction", option.o_friction),
        ("override_solimp", option.o_solimp),
        ("override_solref", option.o_solref),
        ("wind_m_s", option.wind),
    ):
        _require_exact_numeric_array(
            observed,
            simulator_design[name],
            field_name=f"MuJoCo {name}",
        )


def _validate_training_environment(environment: object, projection: Mapping[str, Any]) -> None:
    try:
        import gymnasium as gym
        from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise ExperimentContractError("install the gym and train extras") from exc

    environment_design = projection["environment"]
    simulator_design = projection["simulator"]
    runtime = projection["runtime_requirements"]
    if type(environment) is not DummyVecEnv:
        raise ExperimentContractError("training environment must be an exact DummyVecEnv")
    if (
        type(environment.num_envs) is not int
        or environment.num_envs != environment_design["n_envs"]
        or len(environment.envs) != environment_design["n_envs"]
    ):
        raise ExperimentContractError("training worker count differs")
    expected_stack = tuple(environment_design["wrapper_order_outer_to_inner"])
    for worker_index, worker in enumerate(environment.envs):
        if _wrapper_stack(worker) != expected_stack or type(worker.unwrapped) is not HumanoidEnv:
            raise ExperimentContractError(f"training worker {worker_index} wrapper stack differs")
        if (
            space_sha256(worker.observation_space) != runtime["observation_space_sha256"]
            or space_sha256(worker.action_space) != runtime["action_space_sha256"]
        ):
            raise ExperimentContractError(f"training worker {worker_index} spaces differ")
        _validate_humanoid_constructor(worker, worker.unwrapped, environment_design)
        if getattr(worker, "_max_episode_steps", None) != environment_design["max_episode_steps"]:
            raise ExperimentContractError(f"training worker {worker_index} horizon differs")
        unwrapped = worker.unwrapped
        observed_environment_values = {
            "render_mode": getattr(unwrapped, "render_mode", None),
            "frame_skip": getattr(unwrapped, "frame_skip", None),
            "reset_noise_scale": getattr(unwrapped, "_reset_noise_scale", None),
            "terminate_when_unhealthy": getattr(unwrapped, "_terminate_when_unhealthy", None),
            "exclude_current_positions_from_observation": getattr(
                unwrapped, "_exclude_current_positions_from_observation", None
            ),
            "include_cinert_in_observation": getattr(
                unwrapped, "_include_cinert_in_observation", None
            ),
            "include_cvel_in_observation": getattr(unwrapped, "_include_cvel_in_observation", None),
            "include_qfrc_actuator_in_observation": getattr(
                unwrapped, "_include_qfrc_actuator_in_observation", None
            ),
            "include_cfrc_ext_in_observation": getattr(
                unwrapped, "_include_cfrc_ext_in_observation", None
            ),
            "forward_reward_weight": getattr(unwrapped, "_forward_reward_weight", None),
            "ctrl_cost_weight": getattr(unwrapped, "_ctrl_cost_weight", None),
            "contact_cost_weight": getattr(unwrapped, "_contact_cost_weight", None),
            "healthy_reward": getattr(unwrapped, "_healthy_reward", None),
        }
        for name, observed in observed_environment_values.items():
            if observed != environment_design[name]:
                raise ExperimentContractError(f"training worker {worker_index} {name} differs")
        if (
            tuple(getattr(unwrapped, "_healthy_z_range", ()))
            != tuple(environment_design["healthy_z_range"])
            or tuple(getattr(unwrapped, "_contact_cost_range", ()))
            != (-math.inf, environment_design["contact_cost_range"][1])
            or getattr(unwrapped, "width", None) != environment_design["mujoco_env_width"]
            or getattr(unwrapped, "height", None) != environment_design["mujoco_env_height"]
            or getattr(unwrapped, "camera_id", None) is not None
            or getattr(unwrapped, "camera_name", None) is not None
            or sha256_file(unwrapped.fullpath) != simulator_design["model_sha256"]
        ):
            raise ExperimentContractError(
                f"training worker {worker_index} Humanoid defaults differ"
            )
        _validate_mujoco_model(unwrapped, simulator_design)
        if not math.isclose(
            float(unwrapped.dt),
            environment_design["control_period_seconds"],
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ExperimentContractError(f"training worker {worker_index} control period differs")
    if (
        type(environment.observation_space) is not gym.spaces.Box
        or type(environment.action_space) is not gym.spaces.Box
    ):
        raise ExperimentContractError("training spaces must be exact Box spaces")
    if (
        environment.observation_space.shape != tuple(environment_design["observation_shape"])
        or environment.observation_space.dtype != np.dtype(environment_design["observation_dtype"])
        or environment.action_space.shape != tuple(environment_design["action_shape"])
        or environment.action_space.dtype != np.dtype(environment_design["action_dtype"])
        or not np.array_equal(
            environment.action_space.low,
            np.full(
                environment_design["action_shape"],
                environment_design["physical_action_low"],
                dtype=environment_design["action_dtype"],
            ),
        )
        or not np.array_equal(
            environment.action_space.high,
            np.full(
                environment_design["action_shape"],
                environment_design["physical_action_high"],
                dtype=environment_design["action_dtype"],
            ),
        )
    ):
        raise ExperimentContractError("training observation or action space differs")
    if (
        space_sha256(environment.observation_space) != runtime["observation_space_sha256"]
        or space_sha256(environment.action_space) != runtime["action_space_sha256"]
    ):
        raise ExperimentContractError("training space identity differs")
    pending_seeds = tuple(getattr(environment, "_seeds", ()))
    if pending_seeds != WORKER_SEEDS:
        raise ExperimentContractError("pending DummyVecEnv worker seeds differ")


def _array_sha256(value: np.ndarray) -> str:
    if not isinstance(value, np.ndarray) or not value.flags.c_contiguous:
        raise ExperimentContractError("hashed training array must be C-contiguous")
    header = canonical_json({"dtype": value.dtype.str, "shape": list(value.shape)})
    raw = value.tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _tensor_sha256(name: str, tensor: object) -> str:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise ExperimentContractError(f"training tensor {name} is unavailable")
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    metadata = canonical_json({"dtype": array.dtype.str, "name": name, "shape": list(array.shape)})
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    raw = array.tobytes(order="C")
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def _parameter_state_receipt(
    model: object,
    projection: Mapping[str, Any],
) -> dict[str, object]:
    import torch

    policy = getattr(model, "policy", None)
    named_parameters = getattr(policy, "named_parameters", None)
    if not callable(named_parameters):
        raise ExperimentContractError("TQC policy parameters are unavailable")
    parameters = tuple(named_parameters())
    expected_rows = tuple(projection["tqc_policy"]["ordered_parameter_schema"])
    expected_policy_rows = expected_rows[:-1]
    if len(parameters) != len(expected_policy_rows):
        raise ExperimentContractError("TQC policy parameter count differs")
    policy_digest = hashlib.sha256()
    policy_scalars = 0
    for (name, parameter), row in zip(parameters, expected_policy_rows, strict=True):
        if (
            name != row["name"]
            or tuple(parameter.shape) != tuple(row["shape"])
            or parameter.dtype is not torch.float32
            or parameter.device.type != "cpu"
            or not bool(parameter.detach().isfinite().all().item())
        ):
            raise ExperimentContractError(f"TQC parameter {name} differs or is non-finite")
        value_sha256 = _tensor_sha256(name, parameter)
        encoded = canonical_json(
            {
                "dtype": row["dtype"],
                "name": name,
                "shape": row["shape"],
                "value_sha256": value_sha256,
            }
        )
        policy_digest.update(len(encoded).to_bytes(8, "big"))
        policy_digest.update(encoded)
        policy_scalars += int(parameter.numel())
    entropy = getattr(model, "log_ent_coef", None)
    entropy_row = expected_rows[-1]
    if (
        not isinstance(entropy, torch.Tensor)
        or entropy_row["name"] != "algorithm.log_ent_coef"
        or tuple(entropy.shape) != tuple(entropy_row["shape"])
        or entropy.dtype is not torch.float32
        or entropy.device.type != "cpu"
        or not bool(entropy.detach().isfinite().all().item())
    ):
        raise ExperimentContractError("TQC entropy parameter differs or is non-finite")
    if policy_scalars != projection["tqc_policy"]["expected_policy_parameter_scalar_count"]:
        raise ExperimentContractError("TQC policy scalar count differs")
    return {
        "policy_state_sha256": policy_digest.hexdigest(),
        "entropy_state_sha256": _tensor_sha256("algorithm.log_ent_coef", entropy),
        "policy_tensor_count": len(parameters),
        "policy_scalar_count": policy_scalars,
        "finite_state_tensor_count": len(parameters) + 1,
        "finite_state_scalar_count": policy_scalars + int(entropy.numel()),
    }


def _optimizer_specs(model: object) -> tuple[tuple[str, object, tuple[object, ...], float], ...]:
    return (
        ("actor", model.actor.optimizer, tuple(model.actor.parameters()), 1e-5),
        ("critic", model.critic.optimizer, tuple(model.critic.parameters()), 1e-5),
        ("entropy_coefficient", model.ent_coef_optimizer, (model.log_ent_coef,), 1e-8),
    )


def _optimizer_state_receipt(
    model: object,
    projection: Mapping[str, Any],
    *,
    expected_update: int | None,
) -> dict[str, object]:
    import torch
    from torch.optim.optimizer import _default_to_fused_or_foreach

    optimizers = projection["adam_optimizers"]
    digest = hashlib.sha256()
    state_tensor_count = 0
    state_scalar_count = 0
    owned_parameter_count = 0
    for role, optimizer, expected_parameters, expected_epsilon in _optimizer_specs(model):
        if type(optimizer) is not torch.optim.Adam:
            raise ExperimentContractError(f"TQC {role} optimizer class differs")
        declared = optimizers[role]
        expected_defaults = {
            "lr": declared["lr"],
            "betas": tuple(declared["betas"]),
            "eps": expected_epsilon,
            "weight_decay": declared["weight_decay"],
            "amsgrad": declared["amsgrad"],
            "maximize": declared["maximize"],
            "foreach": declared["foreach"],
            "capturable": declared["capturable"],
            "differentiable": declared["differentiable"],
            "fused": declared["fused"],
            "decoupled_weight_decay": declared["decoupled_weight_decay"],
        }
        if set(optimizer.defaults) != set(expected_defaults) or any(
            optimizer.defaults[name] != value for name, value in expected_defaults.items()
        ):
            raise ExperimentContractError(f"TQC {role} optimizer defaults differ")
        if len(optimizer.param_groups) != 1:
            raise ExperimentContractError(f"TQC {role} optimizer group count differs")
        group = optimizer.param_groups[0]
        if set(group) != {*expected_defaults, "params"} or any(
            group[name] != value for name, value in expected_defaults.items()
        ):
            raise ExperimentContractError(f"TQC {role} optimizer group differs")
        owned = tuple(group["params"])
        if (
            len(owned) != len(expected_parameters)
            or len({id(parameter) for parameter in owned}) != len(owned)
            or any(
                actual is not expected
                for actual, expected in zip(owned, expected_parameters, strict=True)
            )
        ):
            raise ExperimentContractError(f"TQC {role} optimizer ownership differs")
        fused, foreach = _default_to_fused_or_foreach(
            list(owned), differentiable=False, use_fused=False
        )
        resolution_role = "entropy" if role == "entropy_coefficient" else role
        resolution = optimizers["model_construction_resolution"]
        if (
            fused is not resolution[f"required_observed_{resolution_role}_fused"]
            or foreach is not resolution[f"required_observed_{resolution_role}_foreach"]
        ):
            raise ExperimentContractError(f"TQC {role} Adam dispatch differs")
        state = optimizer.state
        if expected_update is None:
            if len(state) != 0:
                raise ExperimentContractError(f"TQC {role} optimizer state must start empty")
            continue
        if set(state) != set(expected_parameters):
            raise ExperimentContractError(f"TQC {role} optimizer state ownership differs")
        for parameter_index, parameter in enumerate(expected_parameters):
            values = state[parameter]
            if type(values) is not dict or set(values) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ExperimentContractError(f"TQC {role} optimizer state fields differ")
            step = values["step"]
            if (
                not isinstance(step, torch.Tensor)
                or step.shape != ()
                or step.dtype is not torch.float32
                or step.device.type != "cpu"
                or not bool(step.isfinite().item())
                or float(step.item()) != float(expected_update)
            ):
                raise ExperimentContractError(f"TQC {role} optimizer step differs")
            fields: dict[str, str] = {"step": _tensor_sha256("step", step)}
            state_tensor_count += 1
            state_scalar_count += 1
            for field_name in ("exp_avg", "exp_avg_sq"):
                value = values[field_name]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != parameter.shape
                    or value.dtype is not parameter.dtype
                    or value.device != parameter.device
                    or not bool(value.isfinite().all().item())
                ):
                    raise ExperimentContractError(
                        f"TQC {role} optimizer {field_name} differs or is non-finite"
                    )
                fields[field_name] = _tensor_sha256(field_name, value)
                state_tensor_count += 1
                state_scalar_count += int(value.numel())
            entry = canonical_json(
                {
                    "fields": fields,
                    "parameter_index": parameter_index,
                    "parameter_shape": list(parameter.shape),
                    "role": role,
                }
            )
            digest.update(len(entry).to_bytes(8, "big"))
            digest.update(entry)
            owned_parameter_count += 1
    if expected_update is None:
        return {
            "optimizer_state_sha256": digest.hexdigest(),
            "owned_parameter_count": 0,
            "state_tensor_count": 0,
            "state_scalar_count": 0,
        }
    expected_schedule = projection["monitoring"]["optimizer_state_finite_schedule"]
    if (
        owned_parameter_count != projection["tqc_policy"]["expected_optimizer_owned_tensor_count"]
        or state_tensor_count != expected_schedule["expected_state_tensor_count_after_first_update"]
        or state_scalar_count != expected_schedule["expected_state_scalar_count_after_first_update"]
    ):
        raise ExperimentContractError("TQC optimizer state size differs")
    return {
        "optimizer_state_sha256": digest.hexdigest(),
        "owned_parameter_count": owned_parameter_count,
        "state_tensor_count": state_tensor_count,
        "state_scalar_count": state_scalar_count,
    }


def _replay_arrays(model: object) -> tuple[tuple[str, np.ndarray], ...]:
    replay = getattr(model, "replay_buffer", None)
    if replay is None:
        raise ExperimentContractError("TQC replay buffer is unavailable")
    values: list[tuple[str, np.ndarray]] = []
    for name in ("observations", "next_observations", "actions", "rewards", "dones", "timeouts"):
        value = getattr(replay, name, None)
        if not isinstance(value, np.ndarray):
            raise ExperimentContractError(f"TQC replay array {name} is unavailable")
        values.append((name, value))
    return tuple(values)


def _validate_replay_structure(
    model: object,
    projection: Mapping[str, Any],
    *,
    final: bool,
) -> int:
    import torch
    from stable_baselines3.common.buffers import ReplayBuffer

    replay = model.replay_buffer
    replay_design = projection["replay_buffer"]
    if type(replay) is not ReplayBuffer:
        raise ExperimentContractError("TQC replay-buffer class differs")
    expected_metadata = {
        "buffer_size": replay_design["internal_vector_slot_capacity"],
        "n_envs": replay_design["n_envs"],
        "obs_shape": tuple(replay_design["observation_shape"]),
        "action_dim": replay_design["action_shape"][0],
        "optimize_memory_usage": replay_design["optimize_memory_usage"],
        "handle_timeout_termination": replay_design["handle_timeout_termination"],
    }
    for name, expected in expected_metadata.items():
        if getattr(replay, name, None) != expected:
            raise ExperimentContractError(f"TQC replay {name} differs")
    if getattr(replay, "device", None) != torch.device("cpu"):
        raise ExperimentContractError("TQC replay device differs")
    slot_count = replay_design["internal_vector_slot_capacity"]
    n_envs = replay_design["n_envs"]
    expected_arrays = {
        "observations": ((slot_count, n_envs, 348), np.dtype("<f8")),
        "next_observations": ((slot_count, n_envs, 348), np.dtype("<f8")),
        "actions": ((slot_count, n_envs, 17), np.dtype("<f4")),
        "rewards": ((slot_count, n_envs), np.dtype("<f4")),
        "dones": ((slot_count, n_envs), np.dtype("<f4")),
        "timeouts": ((slot_count, n_envs), np.dtype("<f4")),
    }
    allocation = 0
    for name, value in _replay_arrays(model):
        shape, dtype = expected_arrays[name]
        if value.shape != shape or value.dtype != dtype or not value.flags.c_contiguous:
            raise ExperimentContractError(f"TQC replay array {name} differs")
        allocation += int(value.nbytes)
    if allocation != replay_design["expected_allocation_bytes"]:
        raise ExperimentContractError("TQC replay allocation differs")
    expected_position = replay_design["expected_final_position"] if final else 0
    expected_full = replay_design["expected_final_full"] if final else False
    if replay.pos != expected_position or replay.full is not expected_full:
        raise ExperimentContractError("TQC replay position or full flag differs")
    return allocation


def _linear_signature(module: object) -> tuple[int, int, bool]:
    import torch

    if type(module) is not torch.nn.Linear:
        raise ExperimentContractError("TQC network layer class differs")
    return module.in_features, module.out_features, module.bias is not None


def _validate_policy_modules(model: object, projection: Mapping[str, Any]) -> None:
    import torch
    from stable_baselines3.common.torch_layers import FlattenExtractor

    policy_design = projection["tqc_policy"]
    actor = model.actor
    critic = model.critic
    target = model.critic_target
    expected_policy_kwargs = _constructor_kwargs(projection)["policy_kwargs"]
    if (
        model.policy_kwargs != expected_policy_kwargs
        or model.policy.net_arch != policy_design["net_arch"]
        or model.policy.activation_fn is not torch.nn.ReLU
        or model.policy.features_extractor_class is not FlattenExtractor
        or model.policy.features_extractor_kwargs != {}
        or model.policy.optimizer_class is not torch.optim.Adam
        or model.policy.optimizer_kwargs != policy_design["optimizer_kwargs"]
        or model.policy.features_extractor is not None
        or model.policy.training is not True
    ):
        raise ExperimentContractError("TQC policy construction differs")
    feature_extractors = (
        actor.features_extractor,
        critic.features_extractor,
        target.features_extractor,
    )
    if (
        any(type(value) is not FlattenExtractor for value in feature_extractors)
        or len({id(value) for value in feature_extractors}) != 3
        or any(
            type(value.flatten) is not torch.nn.Flatten
            or value.flatten.start_dim != 1
            or value.flatten.end_dim != -1
            for value in feature_extractors
        )
    ):
        raise ExperimentContractError("TQC feature-extractor construction differs")
    actor_layers = tuple(actor.latent_pi)
    if (
        type(actor.latent_pi) is not torch.nn.Sequential
        or len(actor_layers) != 4
        or _linear_signature(actor_layers[0]) != (348, 256, True)
        or type(actor_layers[1]) is not torch.nn.ReLU
        or _linear_signature(actor_layers[2]) != (256, 256, True)
        or type(actor_layers[3]) is not torch.nn.ReLU
        or _linear_signature(actor.mu) != (256, 17, True)
        or _linear_signature(actor.log_std) != (256, 17, True)
        or actor.training is not True
    ):
        raise ExperimentContractError("TQC actor module graph differs")
    for role, value, expected_training in (
        ("critic", critic, True),
        ("critic_target", target, False),
    ):
        networks = value.q_networks
        if (
            type(networks) is not list
            or len(networks) != policy_design["n_critics"]
            or value.training is not expected_training
        ):
            raise ExperimentContractError(f"TQC {role} module graph differs")
        for index, network in enumerate(networks):
            layers = tuple(network)
            if (
                type(network) is not torch.nn.Sequential
                or len(layers) != 5
                or _linear_signature(layers[0]) != (365, 256, True)
                or type(layers[1]) is not torch.nn.ReLU
                or _linear_signature(layers[2]) != (256, 256, True)
                or type(layers[3]) is not torch.nn.ReLU
                or _linear_signature(layers[4]) != (256, 25, True)
                or getattr(value, f"qf{index}", None) is not network
            ):
                raise ExperimentContractError(f"TQC {role} network {index} differs")
    if (
        getattr(target, "optimizer", None) is not None
        or any(not parameter.requires_grad for parameter in actor.parameters())
        or any(not parameter.requires_grad for parameter in critic.parameters())
    ):
        raise ExperimentContractError("TQC trainable parameter ownership differs")


def _validate_model_configuration(
    model: object,
    environment: object,
    projection: Mapping[str, Any],
    *,
    initial: bool,
) -> dict[str, object]:
    import torch
    from sb3_contrib import TQC
    from sb3_contrib.tqc.policies import TQCPolicy
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution
    from stable_baselines3.common.logger import Logger
    from stable_baselines3.common.torch_layers import FlattenExtractor

    tqc = projection["tqc"]
    policy_design = projection["tqc_policy"]
    runtime = projection["runtime_requirements"]
    if initial:
        _validate_training_environment(environment, projection)
    if (
        type(model) is not TQC
        or type(model.policy) is not TQCPolicy
        or model.actor is not model.policy.actor
        or model.critic is not model.policy.critic
        or model.critic_target is not model.policy.critic_target
        or model.get_env() is not environment
    ):
        raise ExperimentContractError("TQC model or policy identity differs")
    spaces = (
        model.observation_space,
        model.policy.observation_space,
        model.actor.observation_space,
        model.replay_buffer.observation_space,
    )
    action_spaces = (
        model.action_space,
        model.policy.action_space,
        model.actor.action_space,
        model.replay_buffer.action_space,
    )
    if any(space_sha256(value) != runtime["observation_space_sha256"] for value in spaces) or any(
        space_sha256(value) != runtime["action_space_sha256"] for value in action_spaces
    ):
        raise ExperimentContractError("TQC model space identity differs")
    train_frequency = model.train_freq
    observed = {
        "device": str(model.device),
        "n_envs": model.n_envs,
        "buffer_size": model.buffer_size,
        "learning_starts": model.learning_starts,
        "batch_size": model.batch_size,
        "tau": model.tau,
        "gamma": model.gamma,
        "gradient_steps": model.gradient_steps,
        "n_steps": model.n_steps,
        "stats_window_size": model._stats_window_size,
        "ent_coef": model.ent_coef,
        "target_entropy": float(model.target_entropy),
        "target_update_interval": model.target_update_interval,
        "top_quantiles_to_drop_per_net": model.top_quantiles_to_drop_per_net,
        "action_noise": model.action_noise,
        "use_sde": model.use_sde,
        "sde_sample_freq": model.sde_sample_freq,
        "use_sde_at_warmup": model.use_sde_at_warmup,
        "tensorboard_log": model.tensorboard_log,
        "verbose": model.verbose,
        "seed": model.seed,
        "train_frequency": train_frequency.frequency,
        "train_frequency_unit": train_frequency.unit.value,
    }
    expected = {
        "device": tqc["device"],
        "n_envs": projection["environment"]["n_envs"],
        "buffer_size": tqc["buffer_size"],
        "learning_starts": tqc["learning_starts"],
        "batch_size": tqc["batch_size"],
        "tau": tqc["tau"],
        "gamma": tqc["gamma"],
        "gradient_steps": tqc["gradient_steps"],
        "n_steps": tqc["n_steps"],
        "stats_window_size": tqc["stats_window_size"],
        "ent_coef": tqc["ent_coef"],
        "target_entropy": tqc["expected_resolved_target_entropy"],
        "target_update_interval": tqc["target_update_interval"],
        "top_quantiles_to_drop_per_net": tqc["top_quantiles_to_drop_per_net"],
        "action_noise": tqc["action_noise"],
        "use_sde": tqc["use_sde"],
        "sde_sample_freq": tqc["sde_sample_freq"],
        "use_sde_at_warmup": tqc["use_sde_at_warmup"],
        "tensorboard_log": tqc["tensorboard_log"],
        "verbose": tqc["verbose"],
        "seed": tqc["seed"],
        "train_frequency": tqc["expected_train_frequency"],
        "train_frequency_unit": tqc["expected_train_frequency_unit"],
    }
    mismatches = [name for name, value in expected.items() if observed[name] != value]
    if mismatches:
        raise ExperimentContractError(
            "observed TQC configuration differs: " + ", ".join(mismatches)
        )
    if (
        type(model.logger) is not Logger
        or model.logger.dir is not None
        or model.logger.output_formats != []
        or model._custom_logger is not True
    ):
        raise ExperimentContractError("TQC logger configuration differs")
    actor = model.actor
    _validate_policy_modules(model, projection)
    if (
        tuple(actor.net_arch) != tuple(policy_design["actor_net_arch"])
        or actor.activation_fn is not torch.nn.ReLU
        or type(actor.features_extractor) is not FlattenExtractor
        or actor.use_sde is not False
        or actor.full_std is not True
        or actor.use_expln is not False
        or actor.log_std_init != policy_design["log_std_init"]
        or actor.clip_mean != policy_design["clip_mean"]
        or type(actor.action_dist) is not SquashedDiagGaussianDistribution
        or model.critic.n_quantiles != policy_design["n_quantiles"]
        or model.critic.n_critics != policy_design["n_critics"]
        or model.critic.quantiles_total != policy_design["quantiles_total"]
        or actor.features_dim != policy_design["features_dim"]
        or actor.squash_output is not policy_design["squash_output"]
        or actor.action_dist.epsilon != projection["action_distribution"]["epsilon"]
        or model.policy.share_features_extractor is not False
        or model.policy.normalize_images is not True
        or model._vec_normalize_env is not None
        or model.policy_class is not TQCPolicy
        or model.replay_buffer_class is not ReplayBuffer
        or model.replay_buffer_kwargs != tqc["replay_buffer_kwargs"]
    ):
        raise ExperimentContractError("TQC policy architecture differs")
    if (
        model.lr_schedule(1.0) != tqc["learning_rate"]
        or model.lr_schedule(0.0) != tqc["learning_rate"]
        or torch.get_default_dtype() is not torch.float32
        or torch.are_deterministic_algorithms_enabled() is not False
    ):
        raise ExperimentContractError("TQC learning-rate or Torch defaults differ")
    if initial:
        if (
            model.num_timesteps != 0
            or model._n_updates != 0
            or model._last_obs is not None
            or model._last_episode_starts is not None
            or model._last_original_obs is not None
            or model._episode_num != 0
            or model._total_timesteps != 0
            or model._current_progress_remaining != 1.0
        ):
            raise ExperimentContractError("TQC initial counters or reset state differ")
        for left, right in zip(
            model.critic.parameters(), model.critic_target.parameters(), strict=True
        ):
            if not torch.equal(left.detach(), right.detach()):
                raise ExperimentContractError("TQC initial target critic is not an exact copy")
        if not torch.equal(model.log_ent_coef.detach(), torch.zeros_like(model.log_ent_coef)):
            raise ExperimentContractError("TQC initial entropy coefficient differs")
    else:
        last_observation = getattr(model, "_last_obs", None)
        last_episode_starts = getattr(model, "_last_episode_starts", None)
        if (
            model.num_timesteps != TOTAL_ENVIRONMENT_STEPS
            or model._n_updates != EXPECTED_GRADIENT_UPDATES
            or model._total_timesteps != TOTAL_ENVIRONMENT_STEPS
            or model._current_progress_remaining != 0.0
            or not isinstance(last_observation, np.ndarray)
            or last_observation.shape
            != (
                projection["environment"]["n_envs"],
                *tuple(projection["environment"]["observation_shape"]),
            )
            or last_observation.dtype != np.dtype("<f8")
            or not last_observation.flags.c_contiguous
            or not np.isfinite(last_observation).all()
            or not isinstance(last_episode_starts, np.ndarray)
            or last_episode_starts.shape != (projection["environment"]["n_envs"],)
            or last_episode_starts.dtype != np.dtype("|b1")
            or not bool(last_episode_starts.all())
            or getattr(model, "_last_original_obs", None) is not None
            or model._episode_num
            != (TOTAL_ENVIRONMENT_STEPS // projection["environment"]["max_episode_steps"])
        ):
            raise ExperimentContractError("TQC final state or counters differ")
    parameters = _parameter_state_receipt(model, projection)
    optimizer = _optimizer_state_receipt(
        model,
        projection,
        expected_update=None if initial else int(model._n_updates),
    )
    replay_allocation = _validate_replay_structure(model, projection, final=not initial)
    return {**parameters, **optimizer, "replay_allocation_bytes": replay_allocation}


def _constructor_kwargs(projection: Mapping[str, Any]) -> dict[str, object]:
    import torch
    from stable_baselines3.common.buffers import ReplayBuffer
    from stable_baselines3.common.torch_layers import FlattenExtractor

    tqc = projection["tqc"]
    policy = projection["tqc_policy"]
    return {
        "learning_rate": tqc["learning_rate"],
        "buffer_size": tqc["buffer_size"],
        "learning_starts": tqc["learning_starts"],
        "batch_size": tqc["batch_size"],
        "tau": tqc["tau"],
        "gamma": tqc["gamma"],
        "train_freq": tqc["train_freq_argument"],
        "gradient_steps": tqc["gradient_steps"],
        "action_noise": None,
        "replay_buffer_class": ReplayBuffer,
        "replay_buffer_kwargs": dict(tqc["replay_buffer_kwargs"]),
        "optimize_memory_usage": tqc["optimize_memory_usage"],
        "n_steps": tqc["n_steps"],
        "ent_coef": tqc["ent_coef"],
        "target_update_interval": tqc["target_update_interval"],
        "target_entropy": tqc["target_entropy"],
        "top_quantiles_to_drop_per_net": tqc["top_quantiles_to_drop_per_net"],
        "use_sde": tqc["use_sde"],
        "sde_sample_freq": tqc["sde_sample_freq"],
        "use_sde_at_warmup": tqc["use_sde_at_warmup"],
        "stats_window_size": tqc["stats_window_size"],
        "tensorboard_log": tqc["tensorboard_log"],
        "policy_kwargs": {
            "net_arch": list(policy["net_arch"]),
            "activation_fn": torch.nn.ReLU,
            "use_sde": policy["use_sde"],
            "log_std_init": policy["log_std_init"],
            "use_expln": policy["use_expln"],
            "clip_mean": policy["clip_mean"],
            "features_extractor_class": FlattenExtractor,
            "features_extractor_kwargs": None,
            "normalize_images": policy["normalize_images"],
            "optimizer_class": torch.optim.Adam,
            "optimizer_kwargs": dict(policy["optimizer_kwargs"]),
            "n_quantiles": policy["n_quantiles"],
            "n_critics": policy["n_critics"],
            "share_features_extractor": policy["share_features_extractor"],
        },
        "verbose": tqc["verbose"],
        "seed": tqc["seed"],
        "device": tqc["device"],
        "_init_setup_model": tqc["init_setup_model"],
    }


@dataclass(frozen=True, slots=True)
class TQCTrainingEnvironmentAuthorityV2:
    """Process-local capability for the exact five-worker training environment."""

    attempt_id: str
    execution_manifest_sha256: str
    claimed_work_directory_identity: str
    design_file_sha256: str
    design_semantic_sha256: str
    training_projection_sha256: str
    environment: object = field(repr=False, compare=False)
    resource_monitor: TQCResourceMonitorV2 = field(repr=False, compare=False)
    manifest: object = field(repr=False, compare=False)
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _ENVIRONMENT_ISSUER:
            raise ExperimentContractError(
                "training environment authority may only be issued by preparation"
            )
        _require_sha256(self.execution_manifest_sha256, field_name="execution manifest SHA-256")
        _require_sha256(
            self.claimed_work_directory_identity,
            field_name="claimed work directory identity",
        )
        for value, expected, field_name in (
            (self.design_file_sha256, DESIGN_FILE_SHA256, "design file SHA-256"),
            (self.design_semantic_sha256, DESIGN_SEMANTIC_SHA256, "design semantic SHA-256"),
            (
                self.training_projection_sha256,
                TRAINING_PROJECTION_SHA256,
                "training projection SHA-256",
            ),
        ):
            if value != expected:
                raise ExperimentContractError(f"{field_name} differs")
        if type(self.attempt_id) is not str or not self.attempt_id:
            raise ExperimentContractError("training attempt ID is invalid")
        if type(self.resource_monitor) is not TQCResourceMonitorV2:
            raise ExperimentContractError("training resource monitor type differs")
        if (
            self.claimed_work_directory_identity
            != self.resource_monitor.claimed_work_directory_identity
        ):
            raise ExperimentContractError("training work directory identity differs")

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "execution_manifest_sha256": self.execution_manifest_sha256,
            "claimed_work_directory_identity": self.claimed_work_directory_identity,
            "design_file_sha256": self.design_file_sha256,
            "design_semantic_sha256": self.design_semantic_sha256,
            "training_projection_sha256": self.training_projection_sha256,
            "environment_class": _qualified_type(self.environment),
            "worker_count": getattr(self.environment, "num_envs", None),
        }


@dataclass(frozen=True, slots=True)
class TQCModelConstructionAuthorityV2:
    """Process-local capability for one exact initial TQC model."""

    attempt_id: str
    execution_manifest_sha256: str
    claimed_work_directory_identity: str
    design_file_sha256: str
    design_semantic_sha256: str
    training_projection_sha256: str
    initial_policy_state_sha256: str
    initial_entropy_state_sha256: str
    initial_optimizer_state_sha256: str
    cpu_rng_state_before_sha256: str
    cpu_rng_state_after_sha256: str
    replay_allocation_bytes: int
    _model_handle: _TQCLiveModelHandleV2 = field(repr=False, compare=False)
    environment_authority: TQCTrainingEnvironmentAuthorityV2 = field(repr=False, compare=False)
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _CONSTRUCTION_ISSUER:
            raise ExperimentContractError(
                "TQC construction authority may only be issued by exact construction"
            )
        for name in (
            "execution_manifest_sha256",
            "claimed_work_directory_identity",
            "design_file_sha256",
            "design_semantic_sha256",
            "training_projection_sha256",
            "initial_policy_state_sha256",
            "initial_entropy_state_sha256",
            "initial_optimizer_state_sha256",
            "cpu_rng_state_before_sha256",
            "cpu_rng_state_after_sha256",
        ):
            _require_sha256(getattr(self, name), field_name=name)
        if (
            self.design_file_sha256 != DESIGN_FILE_SHA256
            or self.design_semantic_sha256 != DESIGN_SEMANTIC_SHA256
            or self.training_projection_sha256 != TRAINING_PROJECTION_SHA256
            or self.replay_allocation_bytes != EXPECTED_REPLAY_ALLOCATION_BYTES
            or type(self._model_handle) is not _TQCLiveModelHandleV2
            or type(self.environment_authority) is not TQCTrainingEnvironmentAuthorityV2
            or self.attempt_id != self.environment_authority.attempt_id
            or self.execution_manifest_sha256
            != self.environment_authority.execution_manifest_sha256
            or self.claimed_work_directory_identity
            != self.environment_authority.claimed_work_directory_identity
        ):
            raise ExperimentContractError("TQC construction authority is inconsistent")

    @property
    def model(self) -> object:
        return self._model_handle.constructed_model()

    def to_dict(self) -> dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "execution_manifest_sha256": self.execution_manifest_sha256,
            "claimed_work_directory_identity": self.claimed_work_directory_identity,
            "design_file_sha256": self.design_file_sha256,
            "design_semantic_sha256": self.design_semantic_sha256,
            "training_projection_sha256": self.training_projection_sha256,
            "initial_policy_state_sha256": self.initial_policy_state_sha256,
            "initial_entropy_state_sha256": self.initial_entropy_state_sha256,
            "initial_optimizer_state_sha256": self.initial_optimizer_state_sha256,
            "cpu_rng_state_before_sha256": self.cpu_rng_state_before_sha256,
            "cpu_rng_state_after_sha256": self.cpu_rng_state_after_sha256,
            "replay_allocation_bytes": self.replay_allocation_bytes,
        }


@dataclass(frozen=True, slots=True)
class _CallbackPlan:
    n_envs: int
    expected_vector_steps: int
    expected_environment_steps: int
    learning_starts: int
    expected_updates: int
    replay_slot_capacity: int
    expected_final_replay_position: int
    expected_final_replay_full: bool
    parameter_interval: int
    expected_parameter_checks: int
    expected_optimizer_checks: int
    expected_scalar_checks: int
    first_scalar_callback: int
    observation_shape: tuple[int, ...]
    action_shape: tuple[int, ...]


def _canonical_callback_plan(projection: Mapping[str, Any]) -> _CallbackPlan:
    monitoring = projection["monitoring"]
    return _CallbackPlan(
        n_envs=projection["environment"]["n_envs"],
        expected_vector_steps=projection["counters"]["expected_vector_steps"],
        expected_environment_steps=projection["counters"]["expected_environment_steps"],
        learning_starts=projection["tqc"]["learning_starts"],
        expected_updates=projection["counters"]["expected_gradient_updates"],
        replay_slot_capacity=projection["replay_buffer"]["internal_vector_slot_capacity"],
        expected_final_replay_position=projection["replay_buffer"]["expected_final_position"],
        expected_final_replay_full=projection["replay_buffer"]["expected_final_full"],
        parameter_interval=monitoring["parameter_finite_schedule"]["vector_step_interval"],
        expected_parameter_checks=monitoring["parameter_finite_schedule"]["expected_total_checks"],
        expected_optimizer_checks=monitoring["optimizer_state_finite_schedule"][
            "expected_total_nonempty_optimizer_state_checks"
        ],
        expected_scalar_checks=monitoring["expected_training_scalar_finite_checks"],
        first_scalar_callback=monitoring["training_scalar_callback_lag"][
            "callback_first_validation_call"
        ],
        observation_shape=tuple(projection["environment"]["observation_shape"]),
        action_shape=tuple(projection["environment"]["action_shape"]),
    )


def _assert_finite_tree(value: object, *, field_name: str) -> None:
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biufc" or not np.isfinite(value).all():
            raise ExperimentContractError(f"{field_name} contains non-finite values")
        return
    if isinstance(value, np.number):
        if not np.isfinite(value):
            raise ExperimentContractError(f"{field_name} contains a non-finite value")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExperimentContractError(f"{field_name} contains a non-finite value")
        return
    if isinstance(value, Mapping):
        for name, child in value.items():
            _assert_finite_tree(child, field_name=f"{field_name}.{name}")
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite_tree(child, field_name=f"{field_name}[{index}]")


def _rollout_receipt(
    callback_locals: Mapping[str, Any],
    *,
    plan: _CallbackPlan,
    vector_step: int,
) -> dict[str, object]:
    expected_arrays = {
        "new_obs": ((plan.n_envs, *plan.observation_shape), np.dtype("<f8")),
        "rewards": ((plan.n_envs,), np.dtype("<f4")),
        "dones": ((plan.n_envs,), np.dtype("|b1")),
        "actions": ((plan.n_envs, *plan.action_shape), np.dtype("<f4")),
        "buffer_actions": ((plan.n_envs, *plan.action_shape), np.dtype("<f4")),
    }
    hashes: dict[str, str] = {}
    for name, (shape, dtype) in expected_arrays.items():
        value = callback_locals.get(name)
        if (
            not isinstance(value, np.ndarray)
            or value.shape != shape
            or value.dtype != dtype
            or not value.flags.c_contiguous
            or not np.isfinite(value).all()
        ):
            raise ExperimentContractError(f"TQC callback {name} differs or is non-finite")
        hashes[f"{name}_sha256"] = _array_sha256(value)
    if np.any(callback_locals["actions"] < -0.4) or np.any(callback_locals["actions"] > 0.4):
        raise ExperimentContractError("TQC physical action escaped [-0.4, 0.4]")
    if np.any(callback_locals["buffer_actions"] < -1.0) or np.any(
        callback_locals["buffer_actions"] > 1.0
    ):
        raise ExperimentContractError("TQC normalized action escaped [-1, 1]")
    infos = callback_locals.get("infos")
    if type(infos) is not list or len(infos) != plan.n_envs:
        raise ExperimentContractError("TQC callback info batch differs")
    _assert_finite_tree(infos, field_name="rollout.infos")
    if callback_locals.get("num_collected_steps") != 1:
        raise ExperimentContractError("TQC rollout frequency differs")
    return {"array_hashes": hashes, "vector_step": vector_step}


def _training_scalar_receipt(
    model: object,
    projection: Mapping[str, Any],
    *,
    expected_update: int,
) -> dict[str, float | int]:
    values = getattr(getattr(model, "logger", None), "name_to_value", None)
    if not isinstance(values, dict):
        raise ExperimentContractError("TQC logger scalar map is unavailable")
    result: dict[str, float | int] = {}
    for name in projection["monitoring"]["monitored_training_scalars"]:
        value = values.get(name)
        if name == "train/n_updates":
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ExperimentContractError("TQC logged update count type differs")
            result[name] = int(value)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ExperimentContractError(f"TQC training scalar {name} is unavailable")
        number = float(value)
        if not math.isfinite(number):
            raise ExperimentContractError(f"TQC training scalar {name} is non-finite")
        result[name] = number
    if result["train/n_updates"] != expected_update:
        raise ExperimentContractError("TQC logged update count differs")
    return result


def _update_digest(digest: Any, payload: object) -> None:
    encoded = canonical_json(payload)
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _make_training_callback(
    *,
    model: object,
    projection: Mapping[str, Any],
    plan: _CallbackPlan,
    resource_monitor: object,
) -> object:
    try:
        from stable_baselines3.common.callbacks import BaseCallback
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise ExperimentContractError("install the train extra") from exc

    class TQCTrainingIntegrityCallbackV2(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self._expected_model = model
            self._projection = projection
            self._plan = plan
            self._resource_monitor = resource_monitor
            self._issuer = _CALLBACK_ISSUER
            self.rollout_check_count = 0
            self.parameter_check_count = 0
            self.optimizer_check_count = 0
            self.training_scalar_check_count = 0
            self.rollout_digest = hashlib.sha256()
            self.parameter_digest = hashlib.sha256()
            self.optimizer_digest = hashlib.sha256()
            self.training_scalar_digest = hashlib.sha256()
            self.training_complete = False
            self.training_resource_authority: object | None = None

        def _check_parameters(self, *, expected_update: int | None) -> None:
            parameter = _parameter_state_receipt(self.model, self._projection)
            self.parameter_check_count += 1
            _update_digest(
                self.parameter_digest,
                {"check_index": self.parameter_check_count, **parameter},
            )
            optimizer = _optimizer_state_receipt(
                self.model,
                self._projection,
                expected_update=expected_update,
            )
            if expected_update is not None:
                self.optimizer_check_count += 1
                _update_digest(
                    self.optimizer_digest,
                    {"check_index": self.optimizer_check_count, **optimizer},
                )

        def _check_training_scalars(self, *, expected_update: int) -> None:
            values = _training_scalar_receipt(
                self.model, self._projection, expected_update=expected_update
            )
            self.training_scalar_check_count += 1
            _update_digest(
                self.training_scalar_digest,
                {"check_index": self.training_scalar_check_count, **values},
            )

        def _on_training_start(self) -> None:
            if (
                self.model is not self._expected_model
                or self.n_calls != 0
                or self.model.num_timesteps != 0
                or self.model._n_updates != 0
                or self.training_env is not self._expected_model.get_env()
                or self.model._total_timesteps != self._plan.expected_environment_steps
                or self.model._num_timesteps_at_start != 0
                or self.model._current_progress_remaining != 1.0
                or self.model._episode_num != 0
            ):
                raise ExperimentContractError("TQC training did not start from exact zero")
            observation = getattr(self.model, "_last_obs", None)
            episode_starts = getattr(self.model, "_last_episode_starts", None)
            pending_seeds = tuple(getattr(self.training_env, "_seeds", ()))
            reset_infos = getattr(self.training_env, "reset_infos", None)
            if (
                not isinstance(observation, np.ndarray)
                or observation.shape != (self._plan.n_envs, *self._plan.observation_shape)
                or observation.dtype != np.dtype("<f8")
                or not observation.flags.c_contiguous
                or not np.isfinite(observation).all()
                or not isinstance(episode_starts, np.ndarray)
                or episode_starts.shape != (self._plan.n_envs,)
                or episode_starts.dtype != np.dtype("|b1")
                or not bool(episode_starts.all())
                or pending_seeds != (None,) * self._plan.n_envs
                or type(reset_infos) is not list
                or len(reset_infos) != self._plan.n_envs
                or any(type(value) is not dict for value in reset_infos)
            ):
                raise ExperimentContractError("TQC initial reset state differs")
            _assert_finite_tree(reset_infos, field_name="reset.infos")
            self._resource_monitor.sample_lifecycle("pre_learn")
            self._check_parameters(expected_update=None)

        def _on_step(self) -> bool:
            vector_step = self.n_calls
            environment_steps = vector_step * self._plan.n_envs
            if (
                vector_step < 1
                or vector_step > self._plan.expected_vector_steps
                or self.model.num_timesteps != environment_steps
            ):
                raise ExperimentContractError("TQC callback counter differs")
            warmup_vectors = self._plan.learning_starts // self._plan.n_envs
            expected_previous_updates = max(0, vector_step - warmup_vectors - 1)
            if self.model._n_updates != expected_previous_updates:
                raise ExperimentContractError("TQC callback update lag differs")
            replay = self.model.replay_buffer
            completed_adds = vector_step - 1
            if (
                replay.pos != completed_adds % self._plan.replay_slot_capacity
                or replay.full is not (completed_adds >= self._plan.replay_slot_capacity)
            ):
                raise ExperimentContractError("TQC pre-add replay counter differs")
            rollout = _rollout_receipt(self.locals, plan=self._plan, vector_step=vector_step)
            self.rollout_check_count += 1
            _update_digest(
                self.rollout_digest,
                {"check_index": self.rollout_check_count, **rollout},
            )
            self._resource_monitor.sample_vector_step(vector_step, environment_steps)
            if vector_step >= self._plan.first_scalar_callback:
                self._check_training_scalars(expected_update=expected_previous_updates)
            if vector_step % self._plan.parameter_interval == 0:
                self._check_parameters(expected_update=expected_previous_updates)
            return True

        def _on_training_end(self) -> None:
            if (
                self.n_calls != self._plan.expected_vector_steps
                or self.model.num_timesteps != self._plan.expected_environment_steps
                or self.model._n_updates != self._plan.expected_updates
                or self.model.replay_buffer.pos != self._plan.expected_final_replay_position
                or self.model.replay_buffer.full is not self._plan.expected_final_replay_full
            ):
                raise ExperimentContractError("TQC final training counters differ")
            self._check_training_scalars(expected_update=self._plan.expected_updates)
            self._check_parameters(expected_update=self._plan.expected_updates)
            if (
                self.rollout_check_count != self._plan.expected_vector_steps
                or self.parameter_check_count != self._plan.expected_parameter_checks
                or self.optimizer_check_count != self._plan.expected_optimizer_checks
                or self.training_scalar_check_count != self._plan.expected_scalar_checks
            ):
                raise ExperimentContractError("TQC integrity-monitor schedule is incomplete")
            self._resource_monitor.sample_lifecycle("post_learn")
            self.training_resource_authority = self._resource_monitor.validate_training_prefix()
            self.training_complete = True

    return TQCTrainingIntegrityCallbackV2()


@dataclass(frozen=True, slots=True)
class TQCTrainingCompletionAuthorityV2:
    """Capability proving exact completion of the one frozen training budget."""

    integrity_id: str
    attempt_id: str
    execution_manifest_sha256: str
    claimed_work_directory_identity: str
    resource_monitor_id: str
    worker_pid: int
    design_file_sha256: str
    design_semantic_sha256: str
    training_projection_sha256: str
    environment_steps: int
    vector_steps: int
    gradient_updates: int
    replay_add_calls: int
    actor_optimizer_steps: int
    critic_optimizer_steps: int
    entropy_optimizer_steps: int
    target_polyak_updates: int
    target_polyak_update_count_source: str
    rollout_check_count: int
    parameter_check_count: int
    optimizer_check_count: int
    training_scalar_check_count: int
    rollout_checks_sha256: str
    parameter_checks_sha256: str
    optimizer_checks_sha256: str
    training_scalar_checks_sha256: str
    initial_policy_state_sha256: str
    final_policy_state_sha256: str
    initial_entropy_state_sha256: str
    final_entropy_state_sha256: str
    final_optimizer_state_sha256: str
    resource_prefix_event_sha256: str
    training_integrity_sha256: str
    exact_final_checkpoint_only: bool
    claim_boundary: str
    _model_handle: _TQCLiveModelHandleV2 = field(repr=False, compare=False)
    resource_monitor: TQCResourceMonitorV2 = field(repr=False, compare=False)
    resource_authority: TQCTrainingResourceAuthorityV2 = field(repr=False, compare=False)
    manifest: object = field(repr=False, compare=False)
    design: LoadedTQCDevelopmentDesignV2 = field(repr=False, compare=False)
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _COMPLETION_ISSUER:
            raise ExperimentContractError(
                "training completion authority may only be issued by exact training"
            )
        from .tqc_development_manifest_v2 import ValidatedTQCExecutionManifestV2

        for name in (
            "execution_manifest_sha256",
            "claimed_work_directory_identity",
            "design_file_sha256",
            "design_semantic_sha256",
            "training_projection_sha256",
            "rollout_checks_sha256",
            "parameter_checks_sha256",
            "optimizer_checks_sha256",
            "training_scalar_checks_sha256",
            "initial_policy_state_sha256",
            "final_policy_state_sha256",
            "initial_entropy_state_sha256",
            "final_entropy_state_sha256",
            "final_optimizer_state_sha256",
            "resource_prefix_event_sha256",
            "training_integrity_sha256",
        ):
            _require_sha256(getattr(self, name), field_name=name)
        counters = (
            self.environment_steps == TOTAL_ENVIRONMENT_STEPS
            and self.vector_steps == EXPECTED_VECTOR_STEPS
            and self.gradient_updates == EXPECTED_GRADIENT_UPDATES
            and self.replay_add_calls == EXPECTED_VECTOR_STEPS
            and self.actor_optimizer_steps == EXPECTED_GRADIENT_UPDATES
            and self.critic_optimizer_steps == EXPECTED_GRADIENT_UPDATES
            and self.entropy_optimizer_steps == EXPECTED_GRADIENT_UPDATES
            and self.target_polyak_updates == EXPECTED_GRADIENT_UPDATES
            and self.rollout_check_count == EXPECTED_ROLLOUT_CHECKS
            and self.parameter_check_count == EXPECTED_PARAMETER_CHECKS
            and self.optimizer_check_count == EXPECTED_OPTIMIZER_CHECKS
            and self.training_scalar_check_count == EXPECTED_TRAINING_SCALAR_CHECKS
        )
        if (
            self.integrity_id != TRAINING_INTEGRITY_ID
            or not self.attempt_id
            or self.design_file_sha256 != DESIGN_FILE_SHA256
            or self.design_semantic_sha256 != DESIGN_SEMANTIC_SHA256
            or self.training_projection_sha256 != TRAINING_PROJECTION_SHA256
            or not counters
            or self.exact_final_checkpoint_only is not True
            or self.target_polyak_update_count_source
            != "pinned_TQC_train_source_plus_exact_train_and_update_counters/v1"
            or self.claim_boundary
            != "training_integrity_only_no_behavior_tracker_or_oracle_claim/v1"
            or type(self.resource_monitor) is not TQCResourceMonitorV2
            or type(self._model_handle) is not _TQCLiveModelHandleV2
            or type(self.resource_authority) is not TQCTrainingResourceAuthorityV2
            or self.resource_prefix_event_sha256 != self.resource_authority.prefix_event_sha256
            or self.resource_monitor_id != self.resource_authority.monitor_id
            or self.worker_pid != self.resource_authority.worker_pid
            or self.worker_pid != os.getpid()
            or self.attempt_id != self.resource_authority.attempt_id
            or self.execution_manifest_sha256 != self.resource_authority.execution_manifest_sha256
            or self.claimed_work_directory_identity
            != self.resource_authority.claimed_work_directory_identity
            or self.claimed_work_directory_identity
            != self.resource_monitor.claimed_work_directory_identity
            or type(self.manifest) is not ValidatedTQCExecutionManifestV2
            or self.attempt_id != self.manifest.attempt_id
            or self.execution_manifest_sha256 != self.manifest.sha256
            or self.claimed_work_directory_identity != self.manifest.claimed_work_directory_identity
            or type(self.design) is not LoadedTQCDevelopmentDesignV2
            or self.design_file_sha256 != self.design.file_sha256
            or self.design_semantic_sha256 != self.design.semantic_sha256
            or self.training_projection_sha256 != self.design.training_projection_sha256
            or self.design_file_sha256 != self.manifest.design_file_sha256
            or self.design_semantic_sha256 != self.manifest.design_semantic_sha256
            or self.training_projection_sha256 != self.manifest.training_projection_sha256
        ):
            raise ExperimentContractError("training completion authority is inconsistent")
        expected_integrity_sha256 = hashlib.sha256(
            canonical_json(
                {
                    key: value
                    for key, value in self.to_dict().items()
                    if key != "training_integrity_sha256"
                }
            )
        ).hexdigest()
        if self.training_integrity_sha256 != expected_integrity_sha256:
            raise ExperimentContractError("training completion content hash differs")
        self._model_handle.completed_model(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "integrity_id": self.integrity_id,
            "attempt_id": self.attempt_id,
            "execution_manifest_sha256": self.execution_manifest_sha256,
            "claimed_work_directory_identity": self.claimed_work_directory_identity,
            "resource_monitor_id": self.resource_monitor_id,
            "worker_pid": self.worker_pid,
            "design_file_sha256": self.design_file_sha256,
            "design_semantic_sha256": self.design_semantic_sha256,
            "training_projection_sha256": self.training_projection_sha256,
            "environment_steps": self.environment_steps,
            "vector_steps": self.vector_steps,
            "gradient_updates": self.gradient_updates,
            "replay_add_calls": self.replay_add_calls,
            "actor_optimizer_steps": self.actor_optimizer_steps,
            "critic_optimizer_steps": self.critic_optimizer_steps,
            "entropy_optimizer_steps": self.entropy_optimizer_steps,
            "target_polyak_updates": self.target_polyak_updates,
            "target_polyak_update_count_source": self.target_polyak_update_count_source,
            "rollout_check_count": self.rollout_check_count,
            "parameter_check_count": self.parameter_check_count,
            "optimizer_check_count": self.optimizer_check_count,
            "training_scalar_check_count": self.training_scalar_check_count,
            "rollout_checks_sha256": self.rollout_checks_sha256,
            "parameter_checks_sha256": self.parameter_checks_sha256,
            "optimizer_checks_sha256": self.optimizer_checks_sha256,
            "training_scalar_checks_sha256": self.training_scalar_checks_sha256,
            "initial_policy_state_sha256": self.initial_policy_state_sha256,
            "final_policy_state_sha256": self.final_policy_state_sha256,
            "initial_entropy_state_sha256": self.initial_entropy_state_sha256,
            "final_entropy_state_sha256": self.final_entropy_state_sha256,
            "final_optimizer_state_sha256": self.final_optimizer_state_sha256,
            "resource_prefix_event_sha256": self.resource_prefix_event_sha256,
            "training_integrity_sha256": self.training_integrity_sha256,
            "exact_final_checkpoint_only": self.exact_final_checkpoint_only,
            "claim_boundary": self.claim_boundary,
        }

    @property
    def model(self) -> object:
        return self._model_handle.completed_model(self.to_dict())


def _validate_sealed_training_completion_v2(
    completion: TQCTrainingCompletionAuthorityV2,
) -> object:
    from .tqc_development_manifest_v2 import ValidatedTQCExecutionManifestV2

    payload = completion.to_dict()
    expected_integrity = hashlib.sha256(
        canonical_json(
            {key: value for key, value in payload.items() if key != "training_integrity_sha256"}
        )
    ).hexdigest()
    model = completion._model_handle.completed_model(payload)
    if (
        completion.training_integrity_sha256 != expected_integrity
        or type(completion.resource_monitor) is not TQCResourceMonitorV2
        or type(completion.resource_authority) is not TQCTrainingResourceAuthorityV2
        or type(completion.manifest) is not ValidatedTQCExecutionManifestV2
        or type(completion.design) is not LoadedTQCDevelopmentDesignV2
        or completion.worker_pid != os.getpid()
        or completion.worker_pid != completion.resource_authority.worker_pid
        or completion.resource_prefix_event_sha256
        != completion.resource_authority.prefix_event_sha256
        or completion.resource_monitor_id != completion.resource_authority.monitor_id
        or completion.attempt_id != completion.resource_authority.attempt_id
        or completion.execution_manifest_sha256
        != completion.resource_authority.execution_manifest_sha256
        or completion.claimed_work_directory_identity
        != completion.resource_authority.claimed_work_directory_identity
        or completion.claimed_work_directory_identity
        != completion.resource_monitor.claimed_work_directory_identity
        or completion.attempt_id != completion.manifest.attempt_id
        or completion.execution_manifest_sha256 != completion.manifest.sha256
        or completion.claimed_work_directory_identity
        != completion.manifest.claimed_work_directory_identity
        or completion.design_file_sha256 != completion.design.file_sha256
        or completion.design_semantic_sha256 != completion.design.semantic_sha256
        or completion.training_projection_sha256 != completion.design.training_projection_sha256
        or completion.design_file_sha256 != completion.manifest.design_file_sha256
        or completion.design_semantic_sha256 != completion.manifest.design_semantic_sha256
        or completion.training_projection_sha256 != completion.manifest.training_projection_sha256
    ):
        raise ExperimentContractError("sealed training completion authority is inconsistent")
    return model


def revalidate_tqc_training_completion_v2(
    completion: TQCTrainingCompletionAuthorityV2,
) -> object:
    """Recheck the live model and every completion binding before persistence."""

    if type(completion) is not TQCTrainingCompletionAuthorityV2:
        raise ExperimentContractError("training completion must be the exact issued authority")
    model = _validate_sealed_training_completion_v2(completion)
    projection = _training_projection(completion.design)
    current = _validate_model_configuration(
        model,
        model.get_env(),
        projection,
        initial=False,
    )
    for field_name, expected in (
        ("policy_state_sha256", completion.final_policy_state_sha256),
        ("entropy_state_sha256", completion.final_entropy_state_sha256),
        ("optimizer_state_sha256", completion.final_optimizer_state_sha256),
        ("replay_allocation_bytes", EXPECTED_REPLAY_ALLOCATION_BYTES),
    ):
        if current[field_name] != expected:
            raise ExperimentContractError(f"training completion live {field_name} differs")
    return model


def _consume_tqc_training_model_v2(
    completion: TQCTrainingCompletionAuthorityV2,
) -> object:
    """Consume the live model after persistence so only one full replay remains."""

    revalidate_tqc_training_completion_v2(completion)
    return completion._model_handle.consume_completed_model(completion.to_dict())


def prepare_tqc_training_environment_v2(
    loaded: LoadedTQCDevelopmentDesignV2,
    manifest: object,
    resource_monitor: TQCResourceMonitorV2,
) -> TQCTrainingEnvironmentAuthorityV2:
    """Bind the preflighted monitor, then construct the exact five-worker environment."""

    loaded = _require_loaded_design(loaded)
    manifest = _require_execution_manifest(manifest, loaded)
    if type(resource_monitor) is not TQCResourceMonitorV2:
        raise ExperimentContractError("training requires the exact v2 resource monitor")
    resource_monitor.bind_execution_manifest(manifest)
    projection = _training_projection(loaded)
    environment_design = projection["environment"]
    config = HumanoidExperimentConfig(
        env_id=environment_design["environment_id"],
        terminate_when_unhealthy=environment_design["terminate_when_unhealthy"],
        reset_noise_scale=environment_design["reset_noise_scale"],
        exclude_current_positions_from_observation=environment_design[
            "exclude_current_positions_from_observation"
        ],
        frame_skip=environment_design["frame_skip"],
    )
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise ExperimentContractError("install the train extra") from exc

    def factory() -> object:
        return make_humanoid_env(config, render_mode=None, capture_substep_contacts=False)

    environment = DummyVecEnv([factory for _ in WORKER_SEEDS])
    try:
        if tuple(environment.seed(MODEL_SEED)) != WORKER_SEEDS:
            raise ExperimentContractError("DummyVecEnv worker seed resolution differs")
        _validate_training_environment(environment, projection)
    except Exception:
        environment.close()
        raise
    return TQCTrainingEnvironmentAuthorityV2(
        attempt_id=manifest.attempt_id,
        execution_manifest_sha256=manifest.sha256,
        claimed_work_directory_identity=manifest.claimed_work_directory_identity,
        design_file_sha256=loaded.file_sha256,
        design_semantic_sha256=loaded.semantic_sha256,
        training_projection_sha256=loaded.training_projection_sha256,
        environment=environment,
        resource_monitor=resource_monitor,
        manifest=manifest,
        _issuer=_ENVIRONMENT_ISSUER,
    )


def construct_tqc_model_v2(
    loaded: LoadedTQCDevelopmentDesignV2,
    environment_authority: TQCTrainingEnvironmentAuthorityV2,
) -> TQCModelConstructionAuthorityV2:
    """Allocate and validate the exact TQC model after manifest admission."""

    loaded = _require_loaded_design(loaded)
    if type(environment_authority) is not TQCTrainingEnvironmentAuthorityV2:
        raise ExperimentContractError("TQC model requires an issued environment authority")
    manifest = _require_execution_manifest(environment_authority.manifest, loaded)
    if (
        environment_authority.attempt_id != manifest.attempt_id
        or environment_authority.execution_manifest_sha256 != manifest.sha256
        or environment_authority.claimed_work_directory_identity
        != manifest.claimed_work_directory_identity
        or environment_authority.design_file_sha256 != loaded.file_sha256
        or environment_authority.design_semantic_sha256 != loaded.semantic_sha256
        or environment_authority.training_projection_sha256 != loaded.training_projection_sha256
    ):
        raise ExperimentContractError("TQC environment authority binding differs")
    projection = _training_projection(loaded)
    environment = environment_authority.environment
    _validate_training_environment(environment, projection)
    try:
        import torch
        from sb3_contrib import TQC
        from stable_baselines3.common.logger import Logger
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise ExperimentContractError("install the train extra") from exc

    cpu_rng_before = torch.random.get_rng_state().clone()
    try:
        tqc = projection["tqc"]
        model = TQC(
            tqc["policy"],
            environment,
            **_constructor_kwargs(projection),
        )
        model.set_logger(Logger(folder=None, output_formats=[]))
        cpu_rng_after = torch.random.get_rng_state().clone()
        initial = _validate_model_configuration(model, environment, projection, initial=True)
        environment_authority.resource_monitor.sample_lifecycle("post_model_construction")
    except Exception:
        environment.close()
        raise
    return TQCModelConstructionAuthorityV2(
        attempt_id=manifest.attempt_id,
        execution_manifest_sha256=manifest.sha256,
        claimed_work_directory_identity=manifest.claimed_work_directory_identity,
        design_file_sha256=loaded.file_sha256,
        design_semantic_sha256=loaded.semantic_sha256,
        training_projection_sha256=loaded.training_projection_sha256,
        initial_policy_state_sha256=str(initial["policy_state_sha256"]),
        initial_entropy_state_sha256=str(initial["entropy_state_sha256"]),
        initial_optimizer_state_sha256=str(initial["optimizer_state_sha256"]),
        cpu_rng_state_before_sha256=hashlib.sha256(
            cpu_rng_before.numpy().tobytes(order="C")
        ).hexdigest(),
        cpu_rng_state_after_sha256=hashlib.sha256(
            cpu_rng_after.numpy().tobytes(order="C")
        ).hexdigest(),
        replay_allocation_bytes=int(initial["replay_allocation_bytes"]),
        _model_handle=_TQCLiveModelHandleV2(
            _model=model,
            _phase="constructed",
            _creator_pid=os.getpid(),
            _issuer=_MODEL_HANDLE_ISSUER,
        ),
        environment_authority=environment_authority,
        _issuer=_CONSTRUCTION_ISSUER,
    )


def _completion_payload(
    construction: TQCModelConstructionAuthorityV2,
    callback: object,
    final: Mapping[str, object],
    resource: TQCTrainingResourceAuthorityV2,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "integrity_id": TRAINING_INTEGRITY_ID,
        "attempt_id": construction.attempt_id,
        "execution_manifest_sha256": construction.execution_manifest_sha256,
        "claimed_work_directory_identity": construction.claimed_work_directory_identity,
        "resource_monitor_id": resource.monitor_id,
        "worker_pid": resource.worker_pid,
        "design_file_sha256": construction.design_file_sha256,
        "design_semantic_sha256": construction.design_semantic_sha256,
        "training_projection_sha256": construction.training_projection_sha256,
        "environment_steps": TOTAL_ENVIRONMENT_STEPS,
        "vector_steps": callback.n_calls,
        "gradient_updates": construction.model._n_updates,
        "replay_add_calls": callback.n_calls,
        "actor_optimizer_steps": construction.model._n_updates,
        "critic_optimizer_steps": construction.model._n_updates,
        "entropy_optimizer_steps": construction.model._n_updates,
        "target_polyak_updates": construction.model._n_updates,
        "target_polyak_update_count_source": (
            "pinned_TQC_train_source_plus_exact_train_and_update_counters/v1"
        ),
        "rollout_check_count": callback.rollout_check_count,
        "parameter_check_count": callback.parameter_check_count,
        "optimizer_check_count": callback.optimizer_check_count,
        "training_scalar_check_count": callback.training_scalar_check_count,
        "rollout_checks_sha256": callback.rollout_digest.hexdigest(),
        "parameter_checks_sha256": callback.parameter_digest.hexdigest(),
        "optimizer_checks_sha256": callback.optimizer_digest.hexdigest(),
        "training_scalar_checks_sha256": callback.training_scalar_digest.hexdigest(),
        "initial_policy_state_sha256": construction.initial_policy_state_sha256,
        "final_policy_state_sha256": final["policy_state_sha256"],
        "initial_entropy_state_sha256": construction.initial_entropy_state_sha256,
        "final_entropy_state_sha256": final["entropy_state_sha256"],
        "final_optimizer_state_sha256": final["optimizer_state_sha256"],
        "resource_prefix_event_sha256": resource.prefix_event_sha256,
        "exact_final_checkpoint_only": True,
        "claim_boundary": "training_integrity_only_no_behavior_tracker_or_oracle_claim/v1",
    }
    payload["training_integrity_sha256"] = hashlib.sha256(canonical_json(payload)).hexdigest()
    return payload


def run_tqc_training_v2(
    loaded: LoadedTQCDevelopmentDesignV2,
    construction: TQCModelConstructionAuthorityV2,
) -> TQCTrainingCompletionAuthorityV2:
    """Run exactly one full learn call and issue training-only completion authority."""

    loaded = _require_loaded_design(loaded)
    if type(construction) is not TQCModelConstructionAuthorityV2:
        raise ExperimentContractError("TQC training requires construction authority")
    environment_authority = construction.environment_authority
    manifest = _require_execution_manifest(environment_authority.manifest, loaded)
    if (
        construction.attempt_id != manifest.attempt_id
        or construction.execution_manifest_sha256 != manifest.sha256
        or construction.claimed_work_directory_identity != manifest.claimed_work_directory_identity
        or construction.design_file_sha256 != loaded.file_sha256
        or construction.design_semantic_sha256 != loaded.semantic_sha256
        or construction.training_projection_sha256 != loaded.training_projection_sha256
    ):
        raise ExperimentContractError("TQC construction binding differs before learning")
    projection = _training_projection(loaded)
    initial = _validate_model_configuration(
        construction.model,
        environment_authority.environment,
        projection,
        initial=True,
    )
    if (
        initial["policy_state_sha256"] != construction.initial_policy_state_sha256
        or initial["entropy_state_sha256"] != construction.initial_entropy_state_sha256
        or initial["optimizer_state_sha256"] != construction.initial_optimizer_state_sha256
    ):
        raise ExperimentContractError("TQC initial state changed before learning")
    plan = _canonical_callback_plan(projection)
    callback = _make_training_callback(
        model=construction.model,
        projection=projection,
        plan=plan,
        resource_monitor=environment_authority.resource_monitor,
    )
    learned = construction.model.learn(
        total_timesteps=projection["learn_call"]["total_timesteps"],
        callback=callback,
        log_interval=projection["learn_call"]["log_interval"],
        tb_log_name=projection["learn_call"]["tb_log_name"],
        reset_num_timesteps=projection["learn_call"]["reset_num_timesteps"],
        progress_bar=projection["learn_call"]["progress_bar"],
    )
    if learned is not construction.model or callback.training_complete is not True:
        raise ExperimentContractError("TQC learn call did not complete exactly once")
    resource = callback.training_resource_authority
    if type(resource) is not TQCTrainingResourceAuthorityV2:
        raise ExperimentContractError("TQC training resource authority is missing")
    disk_schedule = projection["monitoring"]["disk_gate_schedule"]
    expected_training_disk_checks = (
        disk_schedule["preflight_checks"] + disk_schedule["vector_step_checks"]
    )
    if (
        resource.attempt_id != manifest.attempt_id
        or resource.worker_pid != os.getpid()
        or resource.execution_manifest_sha256 != manifest.sha256
        or resource.claimed_work_directory_identity != manifest.claimed_work_directory_identity
        or resource.claimed_work_directory_identity
        != environment_authority.resource_monitor.claimed_work_directory_identity
        or resource.vector_sample_count != EXPECTED_VECTOR_STEPS
        or resource.last_vector_step != EXPECTED_VECTOR_STEPS
        or resource.last_environment_step != TOTAL_ENVIRONMENT_STEPS
        or resource.throughput_window_count != EXPECTED_THROUGHPUT_WINDOWS
        or resource.disk_check_count != expected_training_disk_checks
        or resource.lifecycle_stages
        != ("preflight", "post_model_construction", "pre_learn", "post_learn")
        or resource.all_training_resource_gates_passed is not True
    ):
        raise ExperimentContractError("TQC training resource authority differs")
    final = _validate_model_configuration(
        construction.model,
        environment_authority.environment,
        projection,
        initial=False,
    )
    payload = _completion_payload(construction, callback, final, resource)
    construction._model_handle.seal_completion(payload)
    return TQCTrainingCompletionAuthorityV2(
        **payload,
        _model_handle=construction._model_handle,
        resource_monitor=environment_authority.resource_monitor,
        resource_authority=resource,
        manifest=manifest,
        design=loaded,
        _issuer=_COMPLETION_ISSUER,
    )


__all__ = [
    "EXPECTED_OPTIMIZER_CHECKS",
    "EXPECTED_REPLAY_ALLOCATION_BYTES",
    "EXPECTED_ROLLOUT_CHECKS",
    "EXPECTED_TRAINING_SCALAR_CHECKS",
    "TRAINING_INTEGRITY_ID",
    "TQCModelConstructionAuthorityV2",
    "TQCTrainingCompletionAuthorityV2",
    "TQCTrainingEnvironmentAuthorityV2",
    "construct_tqc_model_v2",
    "prepare_tqc_training_environment_v2",
    "revalidate_tqc_training_completion_v2",
    "run_tqc_training_v2",
]
