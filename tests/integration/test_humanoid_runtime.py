from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.envs.humanoid import (
    EXPECTED_ACTION_SHAPE,
    EXPECTED_OBSERVATION_SHAPE,
    EXPECTED_QPOS_SHAPE,
    EXPECTED_QVEL_SHAPE,
    inspect_humanoid_runtime,
)


@pytest.mark.gym
def test_real_humanoid_reset_and_step_match_frozen_substrate() -> None:
    receipt = inspect_humanoid_runtime(seed=20260902)

    assert receipt.observation_shape == EXPECTED_OBSERVATION_SHAPE
    assert receipt.action_shape == EXPECTED_ACTION_SHAPE
    assert receipt.qpos_shape == EXPECTED_QPOS_SHAPE
    assert receipt.qvel_shape == EXPECTED_QVEL_SHAPE
    assert receipt.frame_skip == 5
    assert receipt.control_period_seconds == pytest.approx(0.015)
    assert len(receipt.model_sha256) == 64


@pytest.mark.gym
def test_committed_substrate_receipt_matches_observed_runtime() -> None:
    receipt_path = (
        Path(__file__).parents[2]
        / "experiments"
        / "001_humanoid_fixed_reference"
        / "receipts"
        / "2026-09-02_substrate_smoke.json"
    )
    committed = json.loads(receipt_path.read_text(encoding="utf-8"))
    observed = inspect_humanoid_runtime(seed=committed["seed"]).to_dict()

    for key, value in observed.items():
        assert committed[key] == value


@pytest.mark.gym
def test_instrumented_humanoid_captures_transient_contacts_in_all_substeps() -> None:
    from oracle_composition.envs.humanoid import CONTACT_CAPTURE_ID, make_humanoid_env

    environment = make_humanoid_env(capture_substep_contacts=True)
    try:
        environment.reset(seed=20260902)
        zero_action = np.zeros(17, dtype=environment.action_space.dtype)
        found_transient_contact = False
        for _ in range(100):
            environment.step(zero_action)
            physical = environment.unwrapped
            assert physical.contact_capture_id == CONTACT_CAPTURE_ID
            assert physical.last_control_step_substeps == physical.frame_skip == 5
            samples = physical.last_control_step_contact_samples
            if len(samples) > int(physical.data.ncon):
                found_transient_contact = True
                assert all(0 <= sample.physics_substep_index < 5 for sample in samples)
                assert all(np.isfinite(sample.normal_force_n) for sample in samples)
                break
        assert found_transient_contact, "probe never observed an intermediate-only contact sample"
    finally:
        environment.close()


@pytest.mark.gym
def test_contact_instrumentation_preserves_stock_humanoid_dynamics_bit_for_bit() -> None:
    """Instrumentation invariant only; this is not locomotion evidence."""

    from oracle_composition.envs.humanoid import make_humanoid_env

    stock = make_humanoid_env(capture_substep_contacts=False)
    instrumented = make_humanoid_env(capture_substep_contacts=True)
    action_rng = np.random.default_rng(20260902)
    try:
        stock_observation, _stock_info = stock.reset(seed=20260902)
        instrumented_observation, _instrumented_info = instrumented.reset(seed=20260902)
        np.testing.assert_array_equal(instrumented_observation, stock_observation)
        np.testing.assert_array_equal(instrumented.unwrapped.data.qpos, stock.unwrapped.data.qpos)
        np.testing.assert_array_equal(instrumented.unwrapped.data.qvel, stock.unwrapped.data.qvel)

        for _ in range(20):
            action = action_rng.uniform(
                low=stock.action_space.low,
                high=stock.action_space.high,
            ).astype(stock.action_space.dtype)
            stock_transition = stock.step(action.copy())
            instrumented_transition = instrumented.step(action.copy())

            np.testing.assert_array_equal(instrumented_transition[0], stock_transition[0])
            assert instrumented_transition[1:4] == stock_transition[1:4]
            np.testing.assert_array_equal(
                instrumented.unwrapped.data.qpos,
                stock.unwrapped.data.qpos,
            )
            np.testing.assert_array_equal(
                instrumented.unwrapped.data.qvel,
                stock.unwrapped.data.qvel,
            )
    finally:
        stock.close()
        instrumented.close()
