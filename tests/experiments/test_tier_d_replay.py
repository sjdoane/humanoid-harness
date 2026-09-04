from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from oracle_composition.envs.humanoid import HumanoidExperimentConfig, make_humanoid_env
from oracle_composition.experiments.tier_d_replay_adapter import (
    EXPECTED_WRAPPER_TYPES,
    capture_one_step_replay_probe,
    verify_one_step_replay_probe,
)
from oracle_composition.experiments.tier_d_replay_contract import (
    CLAIM_BOUNDARY,
    EVIDENCE_PURPOSE,
    EXPECTED_STATE_COMPONENT_SIZES,
    ReplayComparison,
    ReplayContact,
    ReplayProbe,
    ReplaySnapshot,
    ReplayVerificationReceipt,
    ReplayWrapperState,
    TierDReplayContractError,
    TierDReplayMismatchError,
)


@pytest.fixture(scope="module")
def replay_probe() -> ReplayProbe:
    """Numeric interface fixture only; its actions are not behavioral evidence."""

    config = HumanoidExperimentConfig()
    environment = make_humanoid_env(config, capture_substep_contacts=True)
    try:
        environment.reset(seed=3)
        action = np.linspace(
            -0.03,
            0.03,
            17,
            dtype=environment.action_space.dtype,
        )
        for _ in range(37):
            environment.step(action)
        return capture_one_step_replay_probe(
            environment,
            config=config,
            action=action,
        )
    finally:
        environment.close()


@pytest.mark.gym
def test_exact_one_step_replay_matches_every_required_field(
    replay_probe: ReplayProbe,
) -> None:
    receipt = verify_one_step_replay_probe(replay_probe)

    assert receipt.replay_passed is True
    assert receipt.comparisons.passed is True
    assert receipt.comparisons.mismatches == ()
    assert receipt.anchor_time_limit_elapsed_steps == 37
    assert receipt.next_time_limit_elapsed_steps == 38
    assert np.linalg.norm(replay_probe.anchor.cfrc_ext) > 0.0
    assert len(replay_probe.transition.contacts) > 0
    assert receipt.to_dict()["comparisons"] == {
        "runtime": True,
        "restored_anchor_state": True,
        "restored_anchor_cfrc_ext": True,
        "restored_anchor_observation": True,
        "restored_wrapper_counter": True,
        "next_state": True,
        "next_cfrc_ext": True,
        "next_observation": True,
        "next_canonical_observation": True,
        "contacts": True,
        "reward": True,
        "terminated": True,
        "truncated": True,
        "next_wrapper_counter": True,
    }


@pytest.mark.gym
def test_probe_binds_exact_runtime_and_nonbehavioral_claim_ceiling(
    replay_probe: ReplayProbe,
) -> None:
    runtime = replay_probe.runtime

    assert runtime.integration_state_flag == 16_383
    assert runtime.integration_state_size == 195
    assert runtime.integration_state_components == EXPECTED_STATE_COMPONENT_SIZES
    assert runtime.wrapper_types == EXPECTED_WRAPPER_TYPES
    assert runtime.timestep_seconds == 0.003
    assert runtime.frame_skip == 5
    assert runtime.control_period_seconds == 0.015
    for field in (
        "project_source_tree_sha256",
        "environment_source_sha256",
        "gym_humanoid_source_sha256",
        "gym_common_wrappers_source_sha256",
        "gym_passive_checker_source_sha256",
        "gym_mujoco_env_source_sha256",
        "mujoco_module_source_sha256",
        "mujoco_functions_binary_sha256",
        "mujoco_structs_binary_sha256",
        "mujoco_enums_binary_sha256",
        "adapter_source_sha256",
        "contract_source_sha256",
    ):
        assert len(getattr(runtime, field)) == 64
    assert replay_probe.evidence_purpose == EVIDENCE_PURPOSE
    assert replay_probe.claim_boundary == CLAIM_BOUNDARY
    assert replay_probe.anchor.integration_state.flags.writeable is False
    assert replay_probe.anchor.cfrc_ext.flags.writeable is False
    assert replay_probe.transition.action.flags.writeable is False
    with pytest.raises(ValueError):
        replay_probe.anchor.integration_state.setflags(write=True)
    with pytest.raises(ValueError):
        replay_probe.anchor.cfrc_ext.setflags(write=True)
    with pytest.raises(ValueError):
        replay_probe.transition.action.setflags(write=True)


def _changed(array: np.ndarray, index: int, delta: float) -> np.ndarray:
    changed = array.copy()
    changed.flat[index] += delta
    return changed


@pytest.mark.gym
@pytest.mark.parametrize(
    "field",
    [
        "runtime",
        "anchor_state",
        "anchor_observation",
        "action",
        "next_state",
        "next_observation",
        "next_canonical_observation",
        "contacts",
        "reward",
        "terminated",
        "truncated",
    ],
)
def test_verifier_fails_closed_for_tampered_expected_values(
    replay_probe: ReplayProbe,
    field: str,
) -> None:
    probe = replay_probe
    if field == "runtime":
        probe = replace(probe, runtime=replace(probe.runtime, model_sha256="0" * 64))
    elif field == "anchor_state":
        anchor = replace(
            probe.anchor,
            integration_state=_changed(probe.anchor.integration_state, 1, 1e-6),
        )
        probe = replace(probe, anchor=anchor)
    elif field == "anchor_observation":
        anchor = replace(
            probe.anchor,
            observation=_changed(probe.anchor.observation, 0, 1e-6),
        )
        probe = replace(probe, anchor=anchor)
    elif field == "action":
        transition = replace(
            probe.transition,
            action=_changed(probe.transition.action, 0, 1e-3),
        )
        probe = replace(probe, transition=transition)
    elif field == "next_state":
        snapshot = replace(
            probe.transition.next_snapshot,
            integration_state=_changed(
                probe.transition.next_snapshot.integration_state,
                -1,
                1e-6,
            ),
        )
        probe = replace(probe, transition=replace(probe.transition, next_snapshot=snapshot))
    elif field == "next_observation":
        transition = replace(
            probe.transition,
            returned_observation=_changed(
                probe.transition.returned_observation,
                0,
                1e-6,
            ),
        )
        probe = replace(probe, transition=transition)
    elif field == "next_canonical_observation":
        snapshot = replace(
            probe.transition.next_snapshot,
            observation=_changed(
                probe.transition.next_snapshot.observation,
                0,
                1e-6,
            ),
        )
        probe = replace(probe, transition=replace(probe.transition, next_snapshot=snapshot))
    elif field == "contacts":
        fake = ReplayContact(
            physics_substep_index=0,
            contact_index_within_substep=0,
            geom1_id=0,
            geom2_id=1,
            normal_force_n=1.0,
        )
        probe = replace(
            probe,
            transition=replace(
                probe.transition,
                contacts=(*probe.transition.contacts, fake),
            ),
        )
    elif field == "reward":
        probe = replace(
            probe,
            transition=replace(
                probe.transition,
                reward=probe.transition.reward + 1.0,
            ),
        )
    elif field == "terminated":
        probe = replace(
            probe,
            transition=replace(
                probe.transition,
                terminated=not probe.transition.terminated,
            ),
        )
    elif field == "truncated":
        probe = replace(
            probe,
            transition=replace(
                probe.transition,
                truncated=not probe.transition.truncated,
            ),
        )

    expected_mismatch = {
        "anchor_state": "restored_anchor_observation",
        "action": "next_state",
    }.get(field, field)
    with pytest.raises(TierDReplayMismatchError, match=expected_mismatch):
        verify_one_step_replay_probe(probe)


@pytest.mark.gym
def test_restored_time_limit_counter_controls_truncation(
    replay_probe: ReplayProbe,
) -> None:
    anchor = replace(
        replay_probe.anchor,
        wrapper_state=ReplayWrapperState(time_limit_elapsed_steps=999),
    )
    next_snapshot = replace(
        replay_probe.transition.next_snapshot,
        wrapper_state=ReplayWrapperState(time_limit_elapsed_steps=1000),
    )
    probe = replace(
        replay_probe,
        anchor=anchor,
        transition=replace(
            replay_probe.transition,
            next_snapshot=next_snapshot,
        ),
    )

    with pytest.raises(TierDReplayMismatchError, match="truncated"):
        verify_one_step_replay_probe(probe)


def test_contract_rejects_qpos_qvel_only_state() -> None:
    with pytest.raises(TierDReplayContractError, match="shape"):
        ReplaySnapshot(
            integration_state=np.zeros(47, dtype="<f8"),
            cfrc_ext=np.zeros((13, 6), dtype="<f8"),
            observation=np.zeros(348, dtype="<f8"),
            wrapper_state=ReplayWrapperState(time_limit_elapsed_steps=0),
        )


@pytest.mark.gym
def test_contract_rejects_inconsistent_wrapper_counters(
    replay_probe: ReplayProbe,
) -> None:
    next_snapshot = replace(
        replay_probe.transition.next_snapshot,
        wrapper_state=ReplayWrapperState(time_limit_elapsed_steps=11),
    )

    with pytest.raises(TierDReplayContractError, match="counter"):
        replace(
            replay_probe,
            transition=replace(
                replay_probe.transition,
                next_snapshot=next_snapshot,
            ),
        )


def test_public_comparison_and_direct_receipt_construction_fail_closed(
    replay_probe: ReplayProbe,
) -> None:
    fields = ReplayComparison.__dataclass_fields__
    with pytest.raises(TierDReplayContractError, match="exact booleans"):
        ReplayComparison(**dict.fromkeys(fields, 1))
    assert not hasattr(ReplayVerificationReceipt, "issue")
    assert not hasattr(ReplayVerificationReceipt, "_from_observed_replay")

    receipt = verify_one_step_replay_probe(replay_probe)
    constructor_values = {
        field: getattr(receipt, field)
        for field in ReplayVerificationReceipt.__dataclass_fields__
        if field != "_issue_token"
    }
    with pytest.raises(TierDReplayContractError, match="issued only"):
        ReplayVerificationReceipt(**constructor_values)
    with pytest.raises(TierDReplayContractError, match="issued only"):
        replace(receipt)


@pytest.mark.parametrize("snapshot_name", ["anchor", "next"])
def test_snapshot_rejects_cfrc_sidecar_observation_mismatch(
    replay_probe: ReplayProbe,
    snapshot_name: str,
) -> None:
    snapshot = (
        replay_probe.anchor if snapshot_name == "anchor" else replay_probe.transition.next_snapshot
    )
    with pytest.raises(TierDReplayContractError, match="observation suffix"):
        replace(
            snapshot,
            cfrc_ext=_changed(snapshot.cfrc_ext, 0, 1e-6),
        )


@pytest.mark.parametrize("value", [True, "1.0"])
def test_contact_rejects_nonnumeric_force(value: object) -> None:
    with pytest.raises(TierDReplayContractError, match="normal force must be numeric"):
        ReplayContact(
            physics_substep_index=0,
            contact_index_within_substep=0,
            geom1_id=0,
            geom2_id=1,
            normal_force_n=value,
        )


@pytest.mark.gym
def test_runtime_rejects_boolean_integer_drift() -> None:
    config = HumanoidExperimentConfig(terminate_when_unhealthy=0)
    environment = make_humanoid_env(config, capture_substep_contacts=True)
    try:
        environment.reset(seed=1)
        with pytest.raises(TierDReplayContractError, match="environment_kwargs"):
            capture_one_step_replay_probe(
                environment,
                config=config,
                action=np.zeros(17, dtype="<f4"),
            )
    finally:
        environment.close()


@pytest.mark.gym
@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("_reset_noise_scale", 999.0, "physical environment kwargs"),
        ("_terminate_when_unhealthy", True, "physical environment kwargs"),
        ("_forward_reward_weight", 999.0, "environment semantics"),
    ],
)
def test_runtime_rejects_mutated_physical_semantics(
    field: str,
    value: object,
    error: str,
) -> None:
    config = HumanoidExperimentConfig()
    environment = make_humanoid_env(config, capture_substep_contacts=True)
    try:
        environment.reset(seed=1)
        setattr(environment.unwrapped, field, value)
        with pytest.raises(TierDReplayContractError, match=error):
            capture_one_step_replay_probe(
                environment,
                config=config,
                action=np.zeros(17, dtype="<f4"),
            )
    finally:
        environment.close()
