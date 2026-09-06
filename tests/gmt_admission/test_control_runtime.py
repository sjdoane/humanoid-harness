from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from oracle_composition.adapters.gmt import control_runtime
from oracle_composition.adapters.gmt.contracts import (
    ACTION_DIM,
    DEFAULT_DOF_POSITION,
    JOINT_NAMES,
    PROPRIOCEPTION_DIM,
    SIMULATION_DECIMATION,
)
from oracle_composition.adapters.gmt.control_runtime import (
    ControlBoundary,
    G1ControlRuntime,
    GMTActorSession,
    compose_residual_raw_action,
)
from oracle_composition.adapters.gmt.io import GMTAdmissionError

HOME_QPOS = np.asarray(
    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, *DEFAULT_DOF_POSITION],
    dtype="<f8",
)


class _FakeModel:
    def __init__(self) -> None:
        self.nq = 30
        self.nv = 29
        self.nu = ACTION_DIM
        self.njnt = ACTION_DIM + 1
        self.nsensor = 3
        self.nkey = 1
        self.ngeom = 100
        self.opt = SimpleNamespace(timestep=0.001)
        self.jnt_type = np.asarray([0, *([3] * ACTION_DIM)], dtype=np.int32)
        self.actuator_trnid = np.column_stack(
            (np.arange(1, ACTION_DIM + 1), np.full(ACTION_DIM, -1))
        )
        self.sensor_type = np.asarray([10, 11, 12], dtype=np.int32)
        self.sensor_dim = np.asarray([4, 3, 3], dtype=np.int32)
        self.key_qpos = HOME_QPOS.reshape(1, -1).copy()
        self.geom_bodyid = np.arange(self.ngeom, dtype=np.int32)


class _FakeData:
    def __init__(self, model: _FakeModel) -> None:
        self.qpos = model.key_qpos[0].copy()
        self.qvel = np.zeros(model.nv, dtype="<f8")
        self.ctrl = np.zeros(model.nu, dtype="<f8")
        self.contact: list[SimpleNamespace] = []
        self.ncon = 0
        self.steps_since_reset = 0
        self.controls: list[np.ndarray] = []

    def sensor(self, name: str) -> SimpleNamespace:
        if name == "orientation":
            value = self.qpos[3:7]
        elif name == "position":
            value = self.qpos[:3]
        elif name == "angular-velocity":
            value = self.qvel[3:6]
        else:  # pragma: no cover - runtime asks only for admitted sensors
            raise KeyError(name)
        return SimpleNamespace(data=value)


class _FakeMujoco:
    def __init__(self) -> None:
        self.mjtObj = SimpleNamespace(
            mjOBJ_JOINT=1,
            mjOBJ_ACTUATOR=2,
            mjOBJ_SENSOR=3,
            mjOBJ_BODY=4,
        )
        self.mjtJoint = SimpleNamespace(mjJNT_FREE=0, mjJNT_HINGE=3)
        self.mjtSensor = SimpleNamespace(
            mjSENS_FRAMEQUAT=10,
            mjSENS_FRAMEPOS=11,
            mjSENS_GYRO=12,
        )
        self.MjModel = SimpleNamespace(from_xml_path=self._from_xml_path)
        self.MjData = self._make_data
        self.model: _FakeModel | None = None
        self.data: _FakeData | None = None

    def _from_xml_path(self, _path: str) -> _FakeModel:
        self.model = _FakeModel()
        return self.model

    def _make_data(self, model: _FakeModel) -> _FakeData:
        self.data = _FakeData(model)
        return self.data

    def mj_id2name(self, _model: _FakeModel, object_type: int, index: int) -> str:
        if object_type == self.mjtObj.mjOBJ_JOINT:
            return "pelvis" if index == 0 else f"{JOINT_NAMES[index - 1]}_joint"
        if object_type == self.mjtObj.mjOBJ_ACTUATOR:
            return f"{JOINT_NAMES[index]}_joint"
        if object_type == self.mjtObj.mjOBJ_SENSOR:
            return ("orientation", "position", "angular-velocity")[index]
        if object_type == self.mjtObj.mjOBJ_BODY:
            return f"body-{index}"
        raise AssertionError(object_type)

    @staticmethod
    def mj_resetDataKeyframe(model: _FakeModel, data: _FakeData, keyframe: int) -> None:
        assert keyframe == 0
        data.qpos[:] = model.key_qpos[keyframe]
        data.qvel.fill(0.0)
        data.ctrl.fill(0.0)
        data.contact = []
        data.ncon = 0
        data.steps_since_reset = 0
        data.controls.clear()

    @staticmethod
    def mj_step(_model: _FakeModel, data: _FakeData) -> None:
        data.steps_since_reset += 1
        data.controls.append(data.ctrl.copy())
        # The first call is the reset warmup. Subsequent calls are controlled substeps.
        if data.steps_since_reset > 1:
            data.qvel[-ACTION_DIM:] += data.ctrl * 1.0e-6
            data.qpos[-ACTION_DIM:] += data.qvel[-ACTION_DIM:] * 0.001
            data.qpos[0] += 1.0e-5
        controlled_substep = data.steps_since_reset - 1
        if controlled_substep == 5:
            data.contact = [SimpleNamespace(geom1=7, geom2=2)]
        elif controlled_substep == SIMULATION_DECIMATION:
            data.contact = [SimpleNamespace(geom1=99, geom2=1)]
        else:
            data.contact = []
        data.ncon = len(data.contact)


class _HistoryEchoActor(torch.nn.Module):
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        # Current proprioception starts at 600; its final 23 values are action history.
        return observation[:, 651:674] + torch.linspace(-0.2, 0.2, ACTION_DIM)


def _boundary(control_step: int = 0) -> ControlBoundary:
    return ControlBoundary(
        qpos=HOME_QPOS.copy(),
        qvel=np.zeros(29, dtype="<f8"),
        orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype="<f4"),
        angular_velocity=np.zeros(3, dtype="<f4"),
        control_step=control_step,
        simulation_step=control_step * SIMULATION_DECIMATION,
    )


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path) -> tuple[G1ControlRuntime, _FakeMujoco]:
    fake = _FakeMujoco()
    monkeypatch.setattr(control_runtime, "_load_mujoco", lambda: fake)
    monkeypatch.setattr(
        control_runtime,
        "verify_upstream_root",
        lambda _root: {"assets/robots/g1/g1.xml": "0" * 64},
    )
    return G1ControlRuntime(tmp_path), fake


def test_runtime_reset_cadence_complete_contacts_and_immutable_trace(fake_runtime) -> None:
    runtime, fake = fake_runtime
    action = np.ones(ACTION_DIM, dtype="<f4")
    with pytest.raises(GMTAdmissionError, match="reset"):
        runtime.step(action)

    initial = runtime.reset()
    np.testing.assert_array_equal(initial.qpos, HOME_QPOS)
    assert (initial.control_step, initial.simulation_step) == (0, 0)
    assert fake.data is not None and fake.data.steps_since_reset == 1

    boundary, interval = runtime.step(action)
    assert (boundary.control_step, boundary.simulation_step) == (1, 20)
    assert interval.qpos_after_substep.shape == (20, 30)
    assert interval.qvel_after_substep.shape == (20, 29)
    assert interval.torque_by_substep.shape == (20, ACTION_DIM)
    assert interval.contact_pairs_after_substep[4] == ((2, 7),)
    assert interval.contact_pairs_after_substep[-1] == ((1, 99),)
    assert runtime.geom_body_names[7] == "body-7"
    assert len(fake.data.controls) == 21  # warmup plus all 20 held-target substeps
    np.testing.assert_array_equal(interval.torque_by_substep, fake.data.controls[1:])
    assert not np.array_equal(interval.torque_by_substep[0], interval.torque_by_substep[-1])
    with pytest.raises(ValueError):
        interval.qpos_after_substep[0, 0] = 0.0

    again = runtime.reset()
    np.testing.assert_array_equal(again.qpos, HOME_QPOS)
    assert (again.control_step, again.simulation_step) == (0, 0)
    assert fake.data.steps_since_reset == 1


def test_runtime_rejects_nonexact_action_arrays(fake_runtime) -> None:
    runtime, _fake = fake_runtime
    runtime.reset()
    with pytest.raises(GMTAdmissionError, match="float32"):
        runtime.step(np.zeros(ACTION_DIM, dtype="<f8"))
    malformed = np.zeros(ACTION_DIM, dtype="<f4")
    malformed[0] = np.nan
    with pytest.raises(GMTAdmissionError, match="finite"):
        runtime.step(malformed)


def test_zero_residual_is_exact_and_scale_is_provisional_raw_action_units() -> None:
    base = np.linspace(-2.0, 2.0, ACTION_DIM, dtype="<f4")
    zero = np.zeros(ACTION_DIM, dtype="<f4")
    np.testing.assert_array_equal(compose_residual_raw_action(base, zero), base)

    positive = np.ones(ACTION_DIM, dtype="<f4")
    composed = compose_residual_raw_action(base, positive)
    np.testing.assert_array_equal(composed, base + np.float32(0.25))
    with pytest.raises(GMTAdmissionError, match=r"\[-1, 1\]"):
        compose_residual_raw_action(base, positive * np.float32(1.01))
    with pytest.raises(GMTAdmissionError, match="float32 scalar"):
        compose_residual_raw_action(base, zero, residual_scale=0.25)  # type: ignore[arg-type]


def test_actor_session_commits_executed_composite_to_next_history() -> None:
    session = GMTActorSession._from_actor_for_test(_HistoryEchoActor())
    reference = np.zeros((20, 30), dtype="<f4")
    first = session.prepare(_boundary(), reference)
    expected_offset = np.linspace(-0.2, 0.2, ACTION_DIM, dtype="<f4")
    np.testing.assert_allclose(first.base_raw, expected_offset, atol=1.0e-7)
    np.testing.assert_array_equal(first.obs[-20 * PROPRIOCEPTION_DIM :], 0.0)

    zero = np.zeros(ACTION_DIM, dtype="<f4")
    np.testing.assert_array_equal(
        compose_residual_raw_action(first.base_raw, zero),
        first.base_raw,
    )
    composite = first.base_raw + np.float32(0.1)
    session.commit(first, np.ascontiguousarray(composite, dtype="<f4"))

    second = session.prepare(_boundary(1), reference)
    np.testing.assert_array_equal(second.proprio[-ACTION_DIM:], composite)
    np.testing.assert_allclose(second.base_raw, composite + expected_offset, atol=1.0e-7)
    np.testing.assert_array_equal(
        second.obs[-PROPRIOCEPTION_DIM:],
        first.proprio.astype("<f4"),
    )


def test_actor_session_enforces_prepare_commit_order_and_control_tick() -> None:
    session = GMTActorSession._from_actor_for_test(_HistoryEchoActor())
    reference = np.zeros((20, 30), dtype="<f4")
    first = session.prepare(_boundary(), reference)
    with pytest.raises(GMTAdmissionError, match="committed first"):
        session.prepare(_boundary(), reference)
    other = GMTActorSession._from_actor_for_test(_HistoryEchoActor()).prepare(
        _boundary(), reference
    )
    with pytest.raises(GMTAdmissionError, match="does not match"):
        session.commit(other, first.base_raw)
    session.commit(first, first.base_raw)
    with pytest.raises(GMTAdmissionError, match="does not match"):
        session.commit(first, first.base_raw)
    with pytest.raises(GMTAdmissionError, match="counters differ"):
        session.prepare(_boundary(2), reference)


def test_boundary_copies_arrays_and_rejects_wrong_cadence() -> None:
    source = HOME_QPOS.copy()
    boundary = ControlBoundary(
        qpos=source,
        qvel=np.zeros(29, dtype="<f8"),
        orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype="<f4"),
        angular_velocity=np.zeros(3, dtype="<f4"),
        control_step=0,
        simulation_step=0,
    )
    source[0] = 99.0
    assert boundary.qpos[0] == 0.0
    with pytest.raises(ValueError):
        boundary.qpos[0] = 1.0
    with pytest.raises(GMTAdmissionError, match="20:1"):
        ControlBoundary(
            qpos=HOME_QPOS.copy(),
            qvel=np.zeros(29, dtype="<f8"),
            orientation_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0], dtype="<f4"),
            angular_velocity=np.zeros(3, dtype="<f4"),
            control_step=1,
            simulation_step=19,
        )
