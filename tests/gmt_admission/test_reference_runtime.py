from __future__ import annotations

import numpy as np
import pytest
import torch

from oracle_composition.adapters.gmt.reference_runtime import ReferenceMotion


def _motion_arrays() -> dict[str, np.ndarray]:
    return {
        "fps": np.asarray([2.0], dtype="<f8"),
        "root_pos": np.asarray(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 2.0], [2.0, 0.0, 3.0]],
            dtype="<f4",
        ),
        "root_rot": np.asarray(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]],
            dtype="<f4",
        ),
        "dof_pos": np.stack(
            (
                np.zeros(23, dtype="<f4"),
                np.ones(23, dtype="<f4"),
                np.full(23, 2.0, dtype="<f4"),
            )
        ),
    }


def test_reference_motion_interpolates_and_wraps_at_native_duration() -> None:
    motion = ReferenceMotion(_motion_arrays())
    sample = motion.sample(torch.tensor([0.0, 0.25, 0.5, 1.0, 1.25]))

    torch.testing.assert_close(sample.root_position[:, 2], torch.tensor([1.0, 1.5, 2.0, 1.0, 1.5]))
    torch.testing.assert_close(sample.dof_position[:, 0], torch.tensor([0.0, 0.5, 1.0, 0.0, 0.5]))


def test_reference_feature_and_window_abi() -> None:
    motion = ReferenceMotion(_motion_arrays())
    current = motion.current(0)
    window = motion.window(0)

    assert current.shape == (30,)
    assert window.shape == (20, 30)
    assert current[0].item() == pytest.approx(1.0)
    assert torch.isfinite(window).all()


def test_reference_motion_rejects_wrong_joint_width() -> None:
    arrays = _motion_arrays()
    arrays["dof_pos"] = np.zeros((3, 22), dtype="<f4")
    with pytest.raises(ValueError, match="joint positions"):
        ReferenceMotion(arrays)


def test_reference_motion_rejects_nonfinite_time() -> None:
    motion = ReferenceMotion(_motion_arrays())
    with pytest.raises(ValueError, match="times must be finite"):
        motion.sample(torch.tensor([float("nan")]))
