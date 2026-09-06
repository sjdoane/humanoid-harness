"""Bounded propose/train/evaluate worker; no generated Python or uploaded policy loading."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from .composition import COMPOSITION_RUNTIME_ID, ComposedReference
from .control_runtime import G1ControlRuntime, GMTActorSession
from .course_config import CourseRunConfig, load_run_config
from .course_evaluation import evaluate_episode
from .gym_env import GYM_RUNTIME_ID, RESIDUAL_OBSERVATION_DIM, RESIDUAL_RAW_SCALE, GMTResidualEnv
from .io import sha256_file, write_deterministic_npz, write_json_receipt

TRAINING_CONTRACT = {
    "algorithm": "stable_baselines3.PPO",
    "device": "cpu",
    "n_steps": 512,
    "batch_size": 128,
    "n_epochs": 4,
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "target_kl": 0.02,
    "ent_coef": 0.0,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "net_arch": [128, 128],
    "log_std_init": -1.5,
    "observation_normalization": "none",
    "reward_normalization": "none",
    "initial_mean_action": "zero_output_layer",
    "checkpoint_selection": "final_fixed_budget_only",
}


def make_env(config: CourseRunConfig, *, record_trajectory: bool = False) -> GMTResidualEnv:
    return GMTResidualEnv(
        plant=G1ControlRuntime(Path(config.assets["upstream_root"])),
        actor=GMTActorSession(
            Path(config.assets["weights"]["path"]),
            expected_sha256=config.assets["weights"]["sha256"],
        ),
        oracle=ComposedReference(config.program, config.segments),
        task=config.task,
        recipe=config.recipe,
        record_trajectory=record_trajectory,
    )


def make_policy(env: GMTResidualEnv, seed: int) -> PPO:
    kwargs = {
        key: TRAINING_CONTRACT[key]
        for key in (
            "n_steps",
            "batch_size",
            "n_epochs",
            "learning_rate",
            "gamma",
            "gae_lambda",
            "clip_range",
            "target_kl",
            "ent_coef",
            "vf_coef",
            "max_grad_norm",
        )
    }
    model = PPO(
        "MlpPolicy",
        env,
        seed=seed,
        device="cpu",
        verbose=0,
        policy_kwargs={"net_arch": [128, 128], "log_std_init": -1.5},
        **kwargs,
    )
    # Deterministic initialization preserves the verified frozen-base behavior.
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.zero_()
    return model


def _numeric_policy(model: PPO, path: Path) -> str:
    arrays = {
        name: value.detach().cpu().numpy().copy()
        for name, value in model.policy.state_dict().items()
    }
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("learned policy contains non-finite state")
    return write_deterministic_npz(path, arrays)


def _frozen_actor_digest(env: GMTResidualEnv) -> str:
    actor = env.actor._actor
    if actor.training or any(parameter.requires_grad for parameter in actor.parameters()):
        raise ValueError("base actor weights must remain frozen and in evaluation mode")
    digest = hashlib.sha256()
    for name, tensor in sorted(actor.state_dict().items()):
        array = tensor.detach().cpu().numpy()
        digest.update(json.dumps([name, array.dtype.str, array.shape]).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _rollout(
    config: CourseRunConfig, env: GMTResidualEnv, model: PPO | None, output: Path, label: str
) -> tuple[dict, dict[str, str]]:
    observation, reset_info = env.reset(seed=config.raw["seed"])
    initial_qpos, initial_qvel = env._boundary.qpos.copy(), env._boundary.qvel.copy()
    frames, residuals = [], []
    total_reward = 0.0
    for _ in range(config.task.horizon_steps):
        action = (
            np.zeros(23, dtype=np.float32)
            if model is None
            else model.predict(observation, deterministic=True)[0].astype(np.float32)
        )
        observation, scalar_reward, terminated, truncated, info = env.step(action)
        if not np.isfinite(observation).all() or not np.isfinite(scalar_reward):
            raise ValueError("non-finite evaluation trajectory")
        frames.append(info)
        residuals.append(action)
        total_reward += scalar_reward
        if terminated or truncated:
            break
    score = evaluate_episode(spec=config.task, frames=frames)
    rows_path = output / f"{label}_frames.jsonl"
    with rows_path.open("x", encoding="utf-8") as handle:
        for row in frames:
            handle.write(
                json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n"
            )
    trace_path = output / f"{label}_trajectory.npz"
    trace_sha = write_deterministic_npz(
        trace_path,
        {
            "qpos": np.asarray(
                [initial_qpos, *[f["trajectory"]["qpos"] for f in frames]], dtype="<f8"
            ),
            "qvel": np.asarray(
                [initial_qvel, *[f["trajectory"]["qvel"] for f in frames]], dtype="<f8"
            ),
            "residual_action": np.asarray(residuals, dtype="<f4"),
            "current_reference": np.asarray(
                [f["trajectory"]["current_reference"] for f in frames], dtype="<f4"
            ),
            "composite_raw_action": np.asarray(
                [f["trajectory"]["composite_raw_action"] for f in frames], dtype="<f4"
            ),
        },
    )
    report = {
        "objective_evaluation": score,
        "training_reward_sum_not_success_metric": total_reward,
        "reset": reset_info,
        "steps": len(frames),
        "residual_rms": float(np.sqrt(np.mean(np.asarray(residuals, dtype=np.float64) ** 2))),
    }
    report_path = output / f"{label}_evaluation.json"
    report_sha = write_json_receipt(report_path, report)
    return report, {
        rows_path.name: sha256_file(rows_path),
        trace_path.name: trace_sha,
        report_path.name: report_sha,
    }


class _TrainingProgress(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.started = time.monotonic()
        self.episodes = 0
        self.falls = 0

    def _on_step(self) -> bool:
        if not all(
            np.isfinite(np.asarray(self.locals[key])).all() for key in ("rewards", "new_obs")
        ):
            raise ValueError("non-finite training transition")
        for done, info in zip(self.locals["dones"], self.locals["infos"], strict=True):
            if done:
                self.episodes += 1
                self.falls += int(info["metrics"]["fallen"])
        return True

    def _on_rollout_end(self) -> None:
        print(
            json.dumps(
                {
                    "event": "training_rollout",
                    "transitions": self.num_timesteps,
                    "episodes": self.episodes,
                    "falls": self.falls,
                    "wall_seconds": time.monotonic() - self.started,
                }
            ),
            flush=True,
        )


def run_course(config_path: Path, output: Path) -> dict:
    config = load_run_config(config_path)
    output.mkdir(exist_ok=True)
    if output.is_symlink() or any(
        p.name not in {"child_stdout.json", "child_stderr.log"} for p in output.iterdir()
    ):
        raise ValueError("course output must be fresh except for supervisor logs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    outputs = {}
    with (output / "input_config.json").open("xb") as handle:
        handle.write(config.encoded)
    outputs["input_config.json"] = config.sha256
    env = make_env(config, record_trajectory=True)
    initial, artifacts = _rollout(config, env, None, output, "zero_residual")
    outputs.update(artifacts)
    final = None
    learned_steps = 0
    training = None
    if config.raw["mode"] == "train":
        training_env = make_env(config)
        base_before = _frozen_actor_digest(training_env)
        model = make_policy(training_env, config.raw["seed"])
        outputs["initial_residual_policy.npz"] = _numeric_policy(
            model, output / "initial_residual_policy.npz"
        )
        callback = _TrainingProgress()
        model.learn(total_timesteps=config.raw["training_steps"], callback=callback)
        base_after = _frozen_actor_digest(training_env)
        if base_after != base_before:
            raise ValueError("training changed the frozen base actor state")
        learned_steps = int(model.num_timesteps)
        if learned_steps != config.raw["training_steps"]:
            raise ValueError("completed training steps differ from the frozen budget")
        outputs["final_residual_policy.npz"] = _numeric_policy(
            model, output / "final_residual_policy.npz"
        )
        final, artifacts = _rollout(config, env, model, output, "final_policy")
        outputs.update(artifacts)
        training = {
            "completed_transitions": learned_steps,
            "episodes": callback.episodes,
            "falls": callback.falls,
            "policy_artifact": "numeric_weights_not_optimizer_resume",
            "frozen_base_state_before_sha256": base_before,
            "frozen_base_state_after_sha256": base_after,
        }
        training_env.close()
    env.close()
    manifest = {
        "schema_version": 1,
        "artifact": "gmt_g1_course_development_run",
        "status": "completed",
        "input_config_sha256": config.sha256,
        "outputs": outputs,
        "identities": {
            "task": config.task.sha256,
            "oracle": config.program.sha256,
            "reward": config.recipe.sha256,
            "segments": {name: segment.sha256 for name, segment in config.segments.items()},
        },
        "frozen_runtime": {
            "gym": GYM_RUNTIME_ID,
            "composition": COMPOSITION_RUNTIME_ID,
            "observation_dim": RESIDUAL_OBSERVATION_DIM,
            "residual_raw_scale": float(RESIDUAL_RAW_SCALE),
            "trainer": TRAINING_CONTRACT,
        },
        "training": training,
        "zero_residual": initial,
        "final_policy": final,
        "runtime": {
            "wall_seconds": time.monotonic() - started,
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "claims": {
            "development_only": True,
            "heldout_generalization_tested": False,
            "physical_obstacle_scene": False,
            "training_performed": learned_steps > 0,
            "full_llm_revision_loop_demonstrated": False,
        },
    }
    digest = write_json_receipt(output / "course_run_manifest.json", manifest)
    return {
        "manifest_sha256": digest,
        "training": training,
        "zero_residual": initial,
        "final_policy": final,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_course(args.config, args.output), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
