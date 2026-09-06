from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.adapters.gmt.contracts import (
    DEFAULT_DOF_POSITION,
    JOINT_NAMES,
    REFERENCE_FRAME_DIM,
)
from oracle_composition.adapters.gmt.deployment import (
    ObservationHistory,
    apply_pd_control,
    build_proprioception,
)
from oracle_composition.adapters.gmt.io import GMTAdmissionError
from oracle_composition.adapters.gmt.replay import (
    ModelABI,
    ReplayConfig,
    _validate_config,
    validate_model_abi,
)


def _valid_model_abi() -> ModelABI:
    return ModelABI(
        nq=30,
        nv=29,
        nu=23,
        joint_names=("pelvis", *(f"{name}_joint" for name in JOINT_NAMES)),
        joint_types=("free", *("hinge" for _ in JOINT_NAMES)),
        actuator_names=tuple(f"{name}_joint" for name in JOINT_NAMES),
        actuator_joint_ids=tuple(range(1, 24)),
        sensor_names=("orientation", "position", "angular-velocity"),
        sensor_types=("framequat", "framepos", "gyro"),
        sensor_dimensions=(4, 3, 3),
        keyframe_count=1,
        home_qpos=(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, *DEFAULT_DOF_POSITION),
        timestep=0.001,
    )


def test_observation_uses_preappend_history_and_zeroes_ankle_velocity() -> None:
    velocity = np.arange(23, dtype=np.float32)
    proprioception = build_proprioception(
        dof_position=np.asarray(DEFAULT_DOF_POSITION, dtype=np.float32),
        dof_velocity=velocity,
        orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        angular_velocity=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        last_raw_action=np.zeros(23, dtype=np.float32),
    )
    history = ObservationHistory()
    reference = np.zeros((20, REFERENCE_FRAME_DIM), dtype=np.float32)

    first = history.assemble(reference, proprioception)
    history.append(proprioception)
    second = history.assemble(reference, proprioception)

    np.testing.assert_array_equal(first[-20 * 74 :], 0.0)
    np.testing.assert_allclose(second[-74:], proprioception.astype(np.float32))
    for joint_index in (4, 5, 10, 11):
        assert proprioception[28 + joint_index] == 0.0


def test_pd_action_contract_preserves_raw_history_and_saturates() -> None:
    raw = np.asarray([-12.0, *([0.0] * 21), 12.0], dtype=np.float32)
    control = apply_pd_control(
        raw_action=raw,
        dof_position=np.asarray(DEFAULT_DOF_POSITION, dtype=np.float32),
        dof_velocity=np.zeros(23, dtype=np.float32),
    )

    np.testing.assert_array_equal(control.raw_history_action, raw)
    assert control.clipped_action[0] == -10.0
    assert control.clipped_action[-1] == 10.0
    assert np.max(np.abs(control.torque)) <= 139.0


def test_model_abi_fails_closed_on_sensor_or_joint_order() -> None:
    validate_model_abi(_valid_model_abi())
    with pytest.raises(GMTAdmissionError, match="sensor names/order"):
        validate_model_abi(
            replace(
                _valid_model_abi(),
                sensor_names=("position", "orientation", "angular-velocity"),
            )
        )
    with pytest.raises(GMTAdmissionError, match="joint names/order"):
        validate_model_abi(replace(_valid_model_abi(), joint_names=("pelvis",)))


def test_replay_config_is_bounded_before_mujoco_import(tmp_path: Path) -> None:
    config = ReplayConfig(
        upstream_root=tmp_path,
        weights_path=tmp_path / "weights.npz",
        weights_sha256="0" * 64,
        motion_path=tmp_path / "motion.npz",
        motion_sha256="1" * 64,
        motion_name="walk_stand",
        trace_path=tmp_path / "trace.npz",
    )
    assert _validate_config(config) == (10_000, 500)
    with pytest.raises(ValueError, match=r"\(0, 10\]"):
        _validate_config(replace(config, duration_seconds=10.001))
    with pytest.raises(ValueError, match="1 ms"):
        _validate_config(replace(config, duration_seconds=0.0015))
    with pytest.raises(ValueError, match=r"\.npz suffix"):
        _validate_config(replace(config, trace_path=tmp_path / "trace.json"))
