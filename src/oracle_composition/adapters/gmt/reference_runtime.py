"""Reference playback reconstructed from the pinned GMT motion runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from .contracts import (
    ACTION_DIM,
    CONTROL_DT_SECONDS,
    REFERENCE_FRAME_DIM,
    REFERENCE_HORIZON,
    REFERENCE_OFFSETS,
)
from .motions import load_converted_motion
from .reference_math import (
    box_smooth,
    inverse_rotate_xyzw,
    quaternion_difference_xyzw,
    quaternion_to_euler_xyzw,
    quaternion_to_exponential_map_xyzw,
    slerp_xyzw,
)


@dataclass(frozen=True)
class MotionFrame:
    root_position: Tensor
    root_rotation_xyzw: Tensor
    root_velocity: Tensor
    root_angular_velocity: Tensor
    dof_position: Tensor
    dof_velocity: Tensor


class ReferenceMotion:
    """One native-cadence GMT clip with the upstream wrap and interpolation rules."""

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        fps_array = np.asarray(arrays["fps"])
        if fps_array.size != 1 or not np.isfinite(fps_array).all() or fps_array.item() <= 0:
            raise ValueError("GMT motion fps must be one positive finite scalar")
        self.fps = torch.tensor(fps_array.item(), dtype=torch.float32)
        self.root_position = torch.tensor(arrays["root_pos"], dtype=torch.float32)
        self.root_rotation_xyzw = torch.tensor(arrays["root_rot"], dtype=torch.float32)
        self.dof_position = torch.tensor(arrays["dof_pos"], dtype=torch.float32)
        if self.root_position.ndim != 2 or self.root_position.shape[1] != 3:
            raise ValueError("GMT root positions must have shape [frames, 3]")
        self.frame_count = self.root_position.shape[0]
        if self.frame_count < 2:
            raise ValueError("GMT motion must contain at least two frames")
        if self.root_rotation_xyzw.shape != (self.frame_count, 4):
            raise ValueError("GMT root rotations must have shape [frames, 4]")
        if self.dof_position.shape != (self.frame_count, ACTION_DIM):
            raise ValueError(f"GMT joint positions must have shape [frames, {ACTION_DIM}]")
        self.duration = torch.tensor(
            (1.0 / fps_array.item()) * (self.frame_count - 1),
            dtype=torch.float32,
        )
        self.root_velocity = self._linear_velocity(self.root_position)
        angular_delta = quaternion_difference_xyzw(
            self.root_rotation_xyzw[:-1], self.root_rotation_xyzw[1:]
        )
        angular_velocity = torch.zeros_like(self.root_position)
        angular_velocity[:-1] = self.fps * quaternion_to_exponential_map_xyzw(angular_delta)
        angular_velocity[-1] = angular_velocity[-2]
        self.root_angular_velocity = box_smooth(angular_velocity)
        self.dof_velocity = self._linear_velocity(self.dof_position)
        derived = (
            self.root_velocity,
            self.root_angular_velocity,
            self.dof_velocity,
        )
        if not all(torch.isfinite(value).all() for value in derived):
            raise ValueError("GMT motion produces non-finite derived velocities")

    @classmethod
    def from_converted(cls, path: Path, *, name: str, expected_sha256: str) -> ReferenceMotion:
        return cls(load_converted_motion(path, name=name, expected_sha256=expected_sha256))

    def _linear_velocity(self, values: Tensor) -> Tensor:
        velocity = torch.zeros_like(values)
        velocity[:-1] = self.fps * (values[1:] - values[:-1])
        velocity[-1] = velocity[-2]
        return box_smooth(velocity)

    def sample(self, times: Tensor) -> MotionFrame:
        if times.ndim != 1:
            raise ValueError(
                f"motion times must have shape [samples], observed {tuple(times.shape)}"
            )
        times = times.to(dtype=torch.float32, device="cpu")
        if not torch.isfinite(times).all():
            raise ValueError("motion times must be finite")
        loop_number = torch.floor(times / self.duration)
        wrapped = times - loop_number * self.duration
        phase = torch.clamp(wrapped / self.duration, 0.0, 1.0)
        frame0 = (phase * (self.frame_count - 1)).long()
        frame1 = torch.minimum(frame0 + 1, torch.full_like(frame0, self.frame_count - 1))
        blend = phase * (self.frame_count - 1) - frame0.float()
        blend_column = blend.unsqueeze(-1)
        return MotionFrame(
            root_position=(1.0 - blend_column) * self.root_position[frame0]
            + blend_column * self.root_position[frame1],
            root_rotation_xyzw=slerp_xyzw(
                self.root_rotation_xyzw[frame0],
                self.root_rotation_xyzw[frame1],
                blend,
            ),
            root_velocity=self.root_velocity[frame0],
            root_angular_velocity=self.root_angular_velocity[frame0],
            dof_position=(1.0 - blend_column) * self.dof_position[frame0]
            + blend_column * self.dof_position[frame1],
            dof_velocity=self.dof_velocity[frame0],
        )

    def features(self, times: Tensor) -> Tensor:
        frame = self.sample(times)
        euler = quaternion_to_euler_xyzw(frame.root_rotation_xyzw)
        local_velocity = inverse_rotate_xyzw(frame.root_rotation_xyzw, frame.root_velocity)
        local_angular_velocity = inverse_rotate_xyzw(
            frame.root_rotation_xyzw, frame.root_angular_velocity
        )
        features = torch.cat(
            (
                frame.root_position[:, 2:3],
                euler[:, :2],
                local_velocity,
                local_angular_velocity[:, 2:3],
                frame.dof_position,
            ),
            dim=-1,
        )
        if features.shape != (times.shape[0], REFERENCE_FRAME_DIM):
            raise RuntimeError(f"unexpected GMT reference feature shape: {tuple(features.shape)}")
        if not torch.isfinite(features).all():
            raise RuntimeError("GMT reference features are non-finite")
        return features

    def window(self, control_step: int) -> Tensor:
        if control_step < 0:
            raise ValueError("control step cannot be negative")
        current_time = torch.tensor(control_step * CONTROL_DT_SECONDS, dtype=torch.float32)
        offsets = torch.tensor(REFERENCE_OFFSETS, dtype=torch.float32) * CONTROL_DT_SECONDS
        features = self.features(current_time + offsets)
        if features.shape != (REFERENCE_HORIZON, REFERENCE_FRAME_DIM):
            raise RuntimeError("GMT reference window contract failed")
        return features

    def current(self, control_step: int) -> Tensor:
        if control_step < 0:
            raise ValueError("control step cannot be negative")
        time = torch.tensor([control_step * CONTROL_DT_SECONDS], dtype=torch.float32)
        return self.features(time)[0]
