from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from oracle_composition.adapters.gmt import parity
from oracle_composition.adapters.gmt.contracts import (
    ACTION_DIM,
    OBSERVATION_DIM,
    PROPRIOCEPTION_DIM,
    REFERENCE_FRAME_DIM,
    REFERENCE_HORIZON,
    SIMULATION_DECIMATION,
)
from oracle_composition.adapters.gmt.control_runtime import (
    ControlBoundary,
    ControlInterval,
    PreparedControl,
)
from oracle_composition.adapters.gmt.io import GMTAdmissionError
from oracle_composition.adapters.gmt.parity import (
    RuntimeParityConfig,
    _ExactComparison,
    _measure_runtime_parity,
)


def _fixture(control_steps: int = 2) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    simulation_steps = control_steps * SIMULATION_DECIMATION
    qpos_truth = np.arange(simulation_steps + 1, dtype="<f8")[:, None]
    qpos_truth = np.ascontiguousarray(qpos_truth + np.arange(30, dtype="<f8") / 1000.0)
    qvel_truth = np.arange(simulation_steps + 1, dtype="<f8")[:, None] / 10.0
    qvel_truth = np.ascontiguousarray(qvel_truth + np.arange(29, dtype="<f8") / 1000.0)
    control_index = np.arange(control_steps, dtype="<f4")[:, None]
    raw = np.ascontiguousarray(
        control_index / np.float32(1000.0) + np.arange(ACTION_DIM, dtype="<f4") / np.float32(100.0)
    )
    reference = np.broadcast_to(
        control_index[:, None, :],
        (control_steps, REFERENCE_HORIZON, REFERENCE_FRAME_DIM),
    ).copy()
    proprio = np.broadcast_to(
        np.arange(control_steps, dtype="<f8")[:, None],
        (control_steps, PROPRIOCEPTION_DIM),
    ).copy()
    observation = np.broadcast_to(
        control_index,
        (control_steps, OBSERVATION_DIM),
    ).copy()
    simulation_index = np.arange(simulation_steps, dtype="<f8")[:, None]
    torque = np.ascontiguousarray(
        simulation_index / 1000.0 + np.arange(ACTION_DIM, dtype="<f8") / 100.0
    )
    orientation = np.zeros((control_steps, 4), dtype="<f4")
    orientation[:, 0] = 1.0
    trace = {
        "sim_qpos": qpos_truth[:-1].copy(),
        "sim_qvel": qvel_truth[:-1].copy(),
        "sim_torque": torque,
        "control_orientation_wxyz": orientation,
        "control_angular_velocity": np.zeros((control_steps, 3), dtype="<f4"),
        "control_reference_window": np.ascontiguousarray(reference, dtype="<f4"),
        "control_proprioception": np.ascontiguousarray(proprio, dtype="<f8"),
        "control_observation": np.ascontiguousarray(observation, dtype="<f4"),
        "control_raw_action": raw,
        "control_clipped_action": raw.copy(),
        "control_pd_target": raw.astype("<f8"),
    }
    truth = {"qpos": qpos_truth, "qvel": qvel_truth}
    return trace, truth


class _FakeRuntime:
    def __init__(
        self,
        trace: dict[str, np.ndarray],
        truth: dict[str, np.ndarray],
        *,
        poststate_off_by_one: bool = False,
        reverse_torque: bool = False,
    ) -> None:
        self.trace = trace
        self.truth = truth
        self.poststate_off_by_one = poststate_off_by_one
        self.reverse_torque = reverse_torque
        self.control_step = 0

    def _boundary(self) -> ControlBoundary:
        simulation_step = self.control_step * SIMULATION_DECIMATION
        row = min(self.control_step, self.trace["control_raw_action"].shape[0] - 1)
        return ControlBoundary(
            qpos=self.truth["qpos"][simulation_step].copy(),
            qvel=self.truth["qvel"][simulation_step].copy(),
            orientation_wxyz=self.trace["control_orientation_wxyz"][row].copy(),
            angular_velocity=self.trace["control_angular_velocity"][row].copy(),
            control_step=self.control_step,
            simulation_step=simulation_step,
        )

    def reset(self) -> ControlBoundary:
        self.control_step = 0
        return self._boundary()

    def step(self, composite: np.ndarray) -> tuple[ControlBoundary, ControlInterval]:
        row = self.control_step
        start = row * SIMULATION_DECIMATION
        state_start = start if self.poststate_off_by_one else start + 1
        qpos = self.truth["qpos"][state_start : state_start + SIMULATION_DECIMATION]
        qvel = self.truth["qvel"][state_start : state_start + SIMULATION_DECIMATION]
        torques = self.trace["sim_torque"][start : start + SIMULATION_DECIMATION]
        if self.reverse_torque:
            torques = torques[::-1].copy()
        interval = ControlInterval(
            composite_raw_action=composite,
            clipped_action=self.trace["control_clipped_action"][row],
            pd_target=self.trace["control_pd_target"][row],
            qpos_after_substep=np.ascontiguousarray(qpos, dtype="<f8"),
            qvel_after_substep=np.ascontiguousarray(qvel, dtype="<f8"),
            torque_by_substep=np.ascontiguousarray(torques, dtype="<f8"),
            contact_pairs_after_substep=tuple(() for _ in range(SIMULATION_DECIMATION)),
            start_control_step=row,
            start_simulation_step=start,
        )
        self.control_step += 1
        return self._boundary(), interval


class _FakeSession:
    def __init__(
        self,
        trace: dict[str, np.ndarray],
        *,
        wrong_action_at: int | None = None,
        wrong_history_at: int | None = None,
    ) -> None:
        self.trace = trace
        self.wrong_action_at = wrong_action_at
        self.wrong_history_at = wrong_history_at
        self.control_step = 0
        self.pending: PreparedControl | None = None

    def reset(self) -> None:
        self.control_step = 0
        self.pending = None

    def prepare(self, boundary: ControlBoundary, window: np.ndarray) -> PreparedControl:
        assert boundary.control_step == self.control_step
        np.testing.assert_array_equal(
            window, self.trace["control_reference_window"][self.control_step]
        )
        base = self.trace["control_raw_action"][self.control_step].copy()
        obs = self.trace["control_observation"][self.control_step].copy()
        if self.wrong_action_at == self.control_step:
            base[0] += np.float32(1.0)
        if self.wrong_history_at == self.control_step:
            obs[-1] += np.float32(1.0)
        self.pending = PreparedControl(
            control_step=self.control_step,
            proprio=self.trace["control_proprioception"][self.control_step],
            obs=obs,
            base_raw=base,
        )
        return self.pending

    def commit(self, prepared: PreparedControl, composite: np.ndarray) -> None:
        assert prepared is self.pending
        np.testing.assert_array_equal(composite, prepared.base_raw)
        self.pending = None
        self.control_step += 1


def _measure(
    trace: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    *,
    runtime: _FakeRuntime | None = None,
    session: _FakeSession | None = None,
) -> dict[str, object]:
    return _measure_runtime_parity(
        trace=trace,
        runtime=runtime or _FakeRuntime(trace, truth),
        session=session or _FakeSession(trace),
        reference_window=lambda step: trace["control_reference_window"][step].copy(),
    )


def test_exact_mapping_compares_shifted_poststates_and_all_applied_torques() -> None:
    trace, truth = _fixture()
    measurement = _measure(trace, truth)

    assert measurement["control_steps"] == 2
    assert measurement["simulation_steps"] == 40
    assert measurement["matched_runtime_poststate_samples"] == 39
    assert measurement["unavailable_baseline_final_poststate_samples"] == 1
    comparisons = measurement["comparisons"]
    assert comparisons["post_substep_qpos"]["values_compared"] == 39 * 30
    assert comparisons["post_substep_qvel"]["values_compared"] == 39 * 29
    assert comparisons["applied_torque"]["values_compared"] == 40 * ACTION_DIM
    assert all(item["max_abs_difference"] == 0.0 for item in comparisons.values())
    assert all(item["exact_values"] == item["values_compared"] for item in comparisons.values())


def test_exact_comparison_has_no_signed_zero_tolerance() -> None:
    comparison = _ExactComparison()
    with pytest.raises(GMTAdmissionError, match="max absolute difference=0"):
        comparison.compare(
            np.asarray([0.0], dtype="<f4"),
            np.asarray([-0.0], dtype="<f4"),
            name="signed zero",
        )


@pytest.mark.parametrize(
    ("runtime_kwargs", "message"),
    [
        ({"poststate_off_by_one": True}, "post-substep qpos shifted by \\+1"),
        ({"reverse_torque": True}, "applied torque substeps"),
    ],
)
def test_off_by_one_or_wrong_substep_mapping_fails_closed(runtime_kwargs, message) -> None:
    trace, truth = _fixture()
    with pytest.raises(GMTAdmissionError, match=message):
        _measure(trace, truth, runtime=_FakeRuntime(trace, truth, **runtime_kwargs))


@pytest.mark.parametrize(
    ("session_kwargs", "message"),
    [
        ({"wrong_action_at": 0}, "base raw action"),
        ({"wrong_history_at": 1}, "prepared obs"),
    ],
)
def test_wrong_actor_action_or_history_fails_closed(session_kwargs, message) -> None:
    trace, truth = _fixture()
    with pytest.raises(GMTAdmissionError, match=message):
        _measure(trace, truth, session=_FakeSession(trace, **session_kwargs))


@dataclass(frozen=True)
class _FakeABI:
    identity: str


class _FakeMotion:
    def __init__(self, trace: dict[str, np.ndarray]) -> None:
        self.trace = trace

    def window(self, step: int) -> torch.Tensor:
        return torch.from_numpy(self.trace["control_reference_window"][step].copy())


def test_run_publishes_small_no_overwrite_receipt_with_all_identities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    trace, truth = _fixture(parity.CONTROL_STEPS)
    support_files = {"reviewed.xml": "d" * 64}
    abi = _FakeABI("fake-reviewed-abi")
    manifest = {
        "trace": {"sha256": "a" * 64},
        "inputs": {
            "weights_sha256": "b" * 64,
            "motion_name": "walk_stand",
            "motion_sha256": "c" * 64,
        },
    }
    runtime = _FakeRuntime(trace, truth)
    runtime.model_abi = abi
    runtime.support_files = support_files
    monkeypatch.setattr(
        parity,
        "load_validated_replay",
        lambda **_kwargs: (manifest, abi, trace, support_files),
    )
    monkeypatch.setattr(parity, "G1ControlRuntime", lambda _root: runtime)
    monkeypatch.setattr(
        parity,
        "GMTActorSession",
        lambda _path, expected_sha256: _FakeSession(trace),
    )
    monkeypatch.setattr(
        parity,
        "ReferenceMotion",
        SimpleNamespace(from_converted=lambda *_args, **_kwargs: _FakeMotion(trace)),
    )
    output = tmp_path / "runtime-parity.json"
    config = RuntimeParityConfig(
        trace_path=tmp_path / "trace.npz",
        trace_sha256="a" * 64,
        manifest_path=tmp_path / "trace.manifest.json",
        manifest_sha256="e" * 64,
        upstream_root=tmp_path / "upstream",
        weights_path=tmp_path / "weights.npz",
        weights_sha256="b" * 64,
        motion_path=tmp_path / "motion.npz",
        motion_sha256="c" * 64,
        motion_name="walk_stand",
        output_path=output,
    )

    result = parity.run_runtime_parity(config)
    payload = json.loads(output.read_text())
    assert result["artifact_path"] == str(output)
    assert result["artifact_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert payload["inputs"]["support_files"] == support_files
    assert payload["inputs"]["replay_manifest"]["sha256"] == "e" * 64
    assert payload["measurement"]["matched_runtime_poststate_samples"] == 9_999
    assert payload["measurement"]["unavailable_baseline_final_poststate_samples"] == 1
    assert payload["limits"]["task_competence_tested"] is False
    with pytest.raises(GMTAdmissionError, match="overwrite"):
        parity.run_runtime_parity(config)
