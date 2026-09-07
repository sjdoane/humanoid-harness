"""Fixed residual-learning environment around GMT and a state-triggered oracle."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np

from .composition import ComposedReference
from .contracts import ACTION_DIM
from .control_runtime import (
    G1ControlRuntime,
    GMTActorSession,
    compose_residual_raw_action,
)
from .course_runtime import (
    AFTER_HEADING_FEEDBACK_RUNTIME,
    COURSE_RESIDUAL_RAW_SCALE,
    FINITE_HORIZON_RUNTIME,
    FOUR_STATE_FINITE_HORIZON_RUNTIME,
    LEGACY_GYM_RUNTIME_ID,
    LEGACY_RUNTIME,
    LOOP_RUNTIME,
    CourseRuntimeProfile,
)
from .course_task import (
    ROOT_HEIGHT_FAILURE_M,
    TORSO_UP_FAILURE_MIN,
    CourseTaskSpec,
    TaskFrame,
    TaskRewardRecipe,
    evaluate_step,
    reward,
)
from .heading_feedback import (
    AFTER_HEADING_FEEDBACK_TRACE_KEY,
    AfterHeadingFeedbackPlan,
    after_heading_feedback_trace,
    issue_held_poststep_target,
    oracle_boundary_inputs,
    plan_after_heading_feedback,
)

GYM_RUNTIME_ID = LEGACY_GYM_RUNTIME_ID
ORACLE_SIGNAL_CONTRACT_ID = "gmt_initial_heading_frame_boundary_signals/v1"
STATE_SLOTS = LEGACY_RUNTIME.state_slots
RESIDUAL_RAW_SCALE = np.float32(COURSE_RESIDUAL_RAW_SCALE)
RESIDUAL_OBSERVATION_DIM = LEGACY_RUNTIME.observation_dim
LOOP_RESIDUAL_OBSERVATION_DIM = LOOP_RUNTIME.observation_dim
FINITE_HORIZON_RESIDUAL_OBSERVATION_DIM = FINITE_HORIZON_RUNTIME.observation_dim
FOUR_STATE_FINITE_HORIZON_OBSERVATION_DIM = FOUR_STATE_FINITE_HORIZON_RUNTIME.observation_dim
AFTER_HEADING_FEEDBACK_OBSERVATION_DIM = AFTER_HEADING_FEEDBACK_RUNTIME.observation_dim


class GMTResidualEnv(gym.Env):
    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(
        self,
        *,
        plant: G1ControlRuntime,
        actor: GMTActorSession,
        oracle: ComposedReference,
        task: CourseTaskSpec,
        recipe: TaskRewardRecipe,
        record_trajectory: bool = False,
        runtime: CourseRuntimeProfile = LEGACY_RUNTIME,
    ) -> None:
        if runtime not in {
            LEGACY_RUNTIME,
            LOOP_RUNTIME,
            FINITE_HORIZON_RUNTIME,
            AFTER_HEADING_FEEDBACK_RUNTIME,
            FOUR_STATE_FINITE_HORIZON_RUNTIME,
        }:
            raise ValueError("course runtime profile is not admitted")
        runtime.validate_program(oracle.program)
        self.plant, self.actor, self.oracle = plant, actor, oracle
        self.task, self.recipe = task, recipe
        self.runtime = runtime
        self.record_trajectory = record_trajectory
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACTION_DIM,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (runtime.observation_dim,), dtype=np.float32
        )
        self._done = True

    def _prepare(self) -> np.ndarray:
        boundary = self._boundary
        self._heading_feedback_plan: AfterHeadingFeedbackPlan | None = None
        inputs = oracle_boundary_inputs(
            task=self.task,
            frame=self._frame,
            qpos=boundary.qpos,
            qvel=boundary.qvel,
            control_step=boundary.control_step,
        )
        native_command = self.oracle.command(
            step=boundary.control_step,
            signals=inputs.signals,
            robot_pose=inputs.robot_pose,
        )
        if self.runtime.after_heading_reference_feedback and native_command.state == "after":
            self._heading_feedback_plan = plan_after_heading_feedback(
                native_command.window,
                lateral_m=inputs.projection.lateral_m,
                heading_error_signed_rad=inputs.projection.heading_error_rad,
            )
            self._command = replace(
                native_command,
                window=self._heading_feedback_plan.issued_window,
            )
        else:
            self._command = native_command
        self._prepared = self.actor.prepare(boundary, self._command.window)
        return self._observation()

    def _observation(self) -> np.ndarray:
        phase = 2 * math.pi * self._command.phase_fraction
        oracle_features = np.asarray(
            [
                math.sin(phase),
                math.cos(phase),
                *[float(self._command.state == name) for name in self.runtime.state_slots],
                min(self.oracle.machine.dwell / self.task.horizon_steps, 1.0),
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            (
                self._prepared.obs,
                self._metrics.task_features(self.task),
                oracle_features,
            )
        ).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options:
            raise ValueError("this frozen family has no caller-selected reset options")
        self._boundary = self.plant.reset()
        self.actor.reset()
        self.oracle.reset()
        self._frame = TaskFrame.initialize(self._boundary.qpos[:2], self._boundary.qpos[3:7])
        segment = self.oracle.segments[self.oracle.machine.behavior]
        import torch

        current = segment.features(torch.zeros(1))[0].numpy()
        self._metrics = evaluate_step(
            spec=self.task,
            frame=self._frame,
            before_qpos=self._boundary.qpos,
            after_qpos=self._boundary.qpos,
            ground_contact_bodies=(),
            current_reference=current,
            control_step=0,
        )
        self._done = False
        observation = self._prepare()
        info = {
            "runtime_id": self.runtime.gym_runtime_id,
            "signal_contract_id": ORACLE_SIGNAL_CONTRACT_ID,
            "reset_distribution": "fixed_home_keyframe_one_warmup_step",
            "seed_effect": "policy_training_rng_only_no_reset_randomization",
            "task_sha256": self.task.sha256,
            "oracle_sha256": self.oracle.program.sha256,
            "reward_sha256": self.recipe.sha256,
        }
        runtime = self.runtime.manifest_contract()
        if runtime is not None:
            info["course_runtime"] = runtime
        return observation, info

    def step(self, action: np.ndarray):
        if self._done:
            raise RuntimeError("reset is required before stepping or after termination")
        action = np.asarray(action)
        if action.dtype != np.float32 or not self.action_space.contains(action):
            raise ValueError("residual must be finite float32[23] in [-1,1]")
        before, executed_command = self._boundary, self._command
        heading_feedback_plan = self._heading_feedback_plan
        composite = compose_residual_raw_action(
            self._prepared.base_raw,
            np.ascontiguousarray(action),
            residual_scale=RESIDUAL_RAW_SCALE,
        )
        boundary, interval = self.plant.step(composite)
        self.actor.commit(self._prepared, composite)
        native_reference = self.oracle.current_after_step(boundary.control_step)
        reference = (
            issue_held_poststep_target(heading_feedback_plan, native_reference)
            if heading_feedback_plan is not None
            else native_reference
        )
        ground_bodies = set()
        names = self.plant.geom_body_names
        for contacts in interval.contact_pairs_after_substep:
            for first, second in contacts:
                if names[first] in {"world", "body:0"}:
                    ground_bodies.add(names[second])
                if names[second] in {"world", "body:0"}:
                    ground_bodies.add(names[first])
        metrics = evaluate_step(
            spec=self.task,
            frame=self._frame,
            before_qpos=before.qpos,
            after_qpos=boundary.qpos,
            ground_contact_bodies=tuple(sorted(ground_bodies)),
            current_reference=reference,
            control_step=boundary.control_step,
        )
        # A brief mid-interval fall cannot disappear between 50 Hz boundaries.
        q = interval.qpos_after_substep
        reasons = list(metrics.failure_reasons)
        if np.any(q[:, 2] < ROOT_HEIGHT_FAILURE_M):
            reasons.append("substep_root_height_failure")
        if np.any(1 - 2 * (q[:, 4] ** 2 + q[:, 5] ** 2) < TORSO_UP_FAILURE_MIN):
            reasons.append("substep_uprightness_failure")
        if reasons:
            metrics = replace(
                metrics,
                fallen=True,
                posture_success=False,
                failure_reasons=tuple(sorted(set(reasons))),
            )
        breakdown = reward(spec=self.task, recipe=self.recipe, metrics=metrics)
        self._boundary, self._metrics = boundary, metrics
        if self.runtime.intrinsic_horizon_termination:
            terminated, truncated = metrics.fallen or metrics.horizon_reached, False
        else:
            terminated, truncated = metrics.fallen, metrics.horizon_reached
        self._done = terminated or truncated
        # Preserve the actual final state; legacy timeout bootstrapping consumes it.
        observation = self._prepare()
        info = {
            "metrics": metrics.to_dict(),
            "reward": breakdown.to_dict(),
            "executed_mode": executed_command.state,
            "executed_behavior": executed_command.behavior,
            "executed_phase_seconds": executed_command.phase_seconds,
            "transition": executed_command.transition,
            "action_saturation_fraction": interval.action_saturation_fraction,
            "torque_saturation_fraction": interval.torque_saturation_fraction,
        }
        if heading_feedback_plan is not None:
            info[AFTER_HEADING_FEEDBACK_TRACE_KEY] = after_heading_feedback_trace(
                control_step=before.control_step,
                plan=heading_feedback_plan,
                source_current=native_reference,
                issued_current=reference,
            )
        if self.record_trajectory:
            info["trajectory"] = {
                "qpos": boundary.qpos.tolist(),
                "qvel": boundary.qvel.tolist(),
                "current_reference": reference.tolist(),
                "composite_raw_action": composite.tolist(),
                "contact_pairs": interval.contact_pairs_after_substep,
                "geom_body_names": names,
            }
        return observation, breakdown.total_reward, terminated, truncated, info
