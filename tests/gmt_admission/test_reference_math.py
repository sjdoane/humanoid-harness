from __future__ import annotations

import numpy as np
import pytest
import torch

from oracle_composition.adapters.gmt.reference_math import (
    box_smooth,
    inverse_rotate_xyzw,
    quaternion_multiply_xyzw,
    quaternion_to_euler_wxyz,
    quaternion_to_euler_xyzw,
    slerp_xyzw,
)


def test_xyzw_quaternion_math_and_wxyz_sensor_order_are_distinct() -> None:
    root_xyzw = torch.tensor([[0.0, 0.0, 2**-0.5, 2**-0.5]], dtype=torch.float32)
    identity_xyzw = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
    product = quaternion_multiply_xyzw(root_xyzw, identity_xyzw)
    reference_euler = quaternion_to_euler_xyzw(root_xyzw)
    sensor_euler = quaternion_to_euler_wxyz(
        np.asarray([2**-0.5, 0.0, 0.0, 2**-0.5], dtype=np.float32)
    )

    torch.testing.assert_close(product, root_xyzw)
    assert reference_euler[0, 2].item() == pytest.approx(np.pi / 2)
    assert sensor_euler[2] == pytest.approx(np.pi / 2)


def test_nonunit_quaternions_are_not_silently_normalized() -> None:
    quaternion = torch.tensor([[0.0, 0.0, 0.0, 0.95]], dtype=torch.float32)
    vector = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    rotated = inverse_rotate_xyzw(quaternion, vector)
    interpolated = slerp_xyzw(quaternion, quaternion, torch.tensor([0.5]))

    assert rotated[0, 0].item() == pytest.approx(0.805, abs=1.0e-6)
    assert torch.linalg.vector_norm(interpolated).item() != pytest.approx(1.0)


def test_box_smoothing_uses_zero_padding_and_width_19() -> None:
    values = torch.ones((20, 2), dtype=torch.float32)
    smoothed = box_smooth(values)

    assert smoothed[0, 0].item() == pytest.approx(10 / 19)
    assert smoothed[9, 0].item() == pytest.approx(1.0)
    assert smoothed[-1, 0].item() == pytest.approx(10 / 19)
    with pytest.raises(ValueError, match="exactly 19"):
        box_smooth(values, width=3)


def test_quaternion_math_rejects_shape_ambiguity() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        quaternion_multiply_xyzw(torch.zeros((1, 4)), torch.zeros((2, 4)))
    with pytest.raises(ValueError, match="blend must have shape"):
        slerp_xyzw(torch.zeros((2, 4)), torch.zeros((2, 4)), torch.zeros((2, 1)))
