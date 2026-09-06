"""Reviewed GMT quaternion and motion math with upstream float32 semantics."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as torch_functional


def _require_last_dimension(value: Tensor, width: int, name: str) -> None:
    if value.ndim == 0 or value.shape[-1] != width:
        raise ValueError(f"{name} must end in {width} values, observed {tuple(value.shape)}")


def quaternion_multiply_xyzw(left: Tensor, right: Tensor) -> Tensor:
    if left.shape != right.shape:
        raise ValueError("quaternion operands must have identical shapes")
    _require_last_dimension(left, 4, "quaternion")
    shape = left.shape
    left = left.reshape(-1, 4)
    right = right.reshape(-1, 4)
    x1, y1, z1, w1 = left[:, 0], left[:, 1], left[:, 2], left[:, 3]
    x2, y2, z2, w2 = right[:, 0], right[:, 1], right[:, 2], right[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack((x, y, z, w), dim=-1).view(shape)


def quaternion_conjugate_xyzw(value: Tensor) -> Tensor:
    _require_last_dimension(value, 4, "quaternion")
    shape = value.shape
    flattened = value.reshape(-1, 4)
    return torch.cat((-flattened[:, :3], flattened[:, 3:]), dim=-1).view(shape)


def quaternion_difference_xyzw(start: Tensor, end: Tensor) -> Tensor:
    return quaternion_multiply_xyzw(end, quaternion_conjugate_xyzw(start))


def normalize_angle(value: Tensor) -> Tensor:
    return torch.atan2(torch.sin(value), torch.cos(value))


def quaternion_to_angle_axis_xyzw(value: Tensor) -> tuple[Tensor, Tensor]:
    _require_last_dimension(value, 4, "quaternion")
    sin_theta = torch.sqrt(1 - value[..., 3] * value[..., 3])
    angle = normalize_angle(2 * torch.acos(value[..., 3]))
    axis = value[..., :3] / sin_theta.unsqueeze(-1)
    mask = torch.abs(sin_theta) > 1.0e-5
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1
    angle = torch.where(mask, angle, torch.zeros_like(angle))
    axis = torch.where(mask.unsqueeze(-1), axis, default_axis)
    return angle, axis


def quaternion_to_exponential_map_xyzw(value: Tensor) -> Tensor:
    angle, axis = quaternion_to_angle_axis_xyzw(value)
    return angle.unsqueeze(-1) * axis


def slerp_xyzw(start: Tensor, end: Tensor, blend: Tensor) -> Tensor:
    if start.shape != end.shape:
        raise ValueError("SLERP quaternion operands must have identical shapes")
    _require_last_dimension(start, 4, "quaternion")
    if blend.shape != start.shape[:-1]:
        raise ValueError(
            f"SLERP blend must have shape {tuple(start.shape[:-1])}, observed {tuple(blend.shape)}"
        )
    cosine = torch.sum(start * end, dim=-1)
    end = torch.where((cosine < 0).unsqueeze(-1), -end, end)
    cosine = torch.abs(cosine).unsqueeze(-1)
    half_theta = torch.acos(cosine)
    sin_half_theta = torch.sqrt(1.0 - cosine * cosine)
    blend = blend.unsqueeze(-1)
    ratio_start = torch.sin((1 - blend) * half_theta) / sin_half_theta
    ratio_end = torch.sin(blend * half_theta) / sin_half_theta
    result = ratio_start * start + ratio_end * end
    result = torch.where(
        torch.abs(sin_half_theta) < 0.001,
        0.5 * start + 0.5 * end,
        result,
    )
    return torch.where(torch.abs(cosine) >= 1, start, result)


def inverse_rotate_xyzw(quaternion: Tensor, vector: Tensor) -> Tensor:
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError("quaternion must have shape [batch, 4]")
    if vector.shape != (quaternion.shape[0], 3):
        raise ValueError("vector must have shape [batch, 3]")
    scalar = quaternion[:, -1]
    xyz = quaternion[:, :3]
    first = vector * (2.0 * scalar**2 - 1.0).unsqueeze(-1)
    second = torch.cross(xyz, vector, dim=-1) * scalar.unsqueeze(-1) * 2.0
    third = xyz * torch.bmm(xyz.view(-1, 1, 3), vector.view(-1, 3, 1)).squeeze(-1) * 2.0
    return first - second + third


def quaternion_to_euler_xyzw(value: Tensor) -> Tensor:
    _require_last_dimension(value, 4, "quaternion")
    x, y, z, w = value.unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_argument = torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(pitch_argument)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return torch.stack((roll, pitch, yaw), dim=-1)


def quaternion_to_euler_wxyz(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.shape != (4,):
        raise ValueError(f"sensor quaternion must have shape (4,), observed {value.shape}")
    w, x, y, z = value
    result = np.zeros(3, dtype=np.float64)
    result[0] = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch_argument = 2 * (w * y - z * x)
    if np.abs(pitch_argument) >= 1:
        result[1] = np.copysign(np.pi / 2, pitch_argument)
    else:
        result[1] = np.arcsin(pitch_argument)
    result[2] = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return result


def box_smooth(values: Tensor, width: int = 19) -> Tensor:
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("motion values must have shape [frames, channels]")
    if width != 19:
        raise ValueError("the pinned GMT smoothing width is exactly 19")
    box = torch.ones(width, dtype=values.dtype, device=values.device) / width
    channels = values.shape[1]
    transposed = values.T.unsqueeze(0)
    smoothed = torch_functional.conv1d(
        transposed,
        box.view(1, 1, -1).expand(channels, 1, -1),
        groups=channels,
        padding="same",
    )
    return smoothed.squeeze(0).T
