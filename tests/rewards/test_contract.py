from __future__ import annotations

import dataclasses
import hashlib

import numpy as np
import pytest

from oracle_composition.rewards.contract import (
    CANDIDATE_READ_SET_V1,
    CONTROL_PERIOD_SECONDS,
    CandidateTaskInputsV1,
    RewardArtifactIdentityV1,
    RewardContractError,
    RewardResultV1,
    TrustedRewardStepV1,
    reward_schema_sha256,
)


def _step(**updates: object) -> TrustedRewardStepV1:
    qpos = np.zeros(24, dtype=np.float64)
    qpos[3] = 1.0
    values: dict[str, object] = {
        "qpos_after_f64": qpos,
        "qvel_after_f64": np.zeros(23, dtype=np.float64),
        "com_x_velocity_m_s": 1.0,
        "ctrl_f64": np.zeros(17, dtype=np.float64),
        "generalized_actuator_torque_n_m_f64": np.zeros(17, dtype=np.float64),
        "external_contact_wrench_f64": np.zeros((14, 6), dtype=np.float64),
        "target_speed_m_s": 1.0,
        "control_period_s": CONTROL_PERIOD_SECONDS,
    }
    values.update(updates)
    return TrustedRewardStepV1(**values)  # type: ignore[arg-type]


def test_schema_exposes_only_two_float_candidate_fields_and_round_trips() -> None:
    step = _step()
    assert dataclasses.fields(step.candidate_inputs)[0].name == CANDIDATE_READ_SET_V1[0]
    assert tuple(field.name for field in dataclasses.fields(step.candidate_inputs)) == (
        CANDIDATE_READ_SET_V1
    )
    assert not hasattr(step.candidate_inputs, "qpos_after_f64")
    assert (
        CandidateTaskInputsV1.from_canonical_bytes(step.candidate_inputs.canonical_bytes)
        == step.candidate_inputs
    )
    assert (
        TrustedRewardStepV1.from_canonical_bytes(step.canonical_bytes).to_dict() == step.to_dict()
    )

    result = RewardResultV1(
        total=6.25,
        signed_terms={"task_progress": 1.25, "healthy": 5.0, "control": -0.0, "contact": -0.0},
    )
    assert RewardResultV1.from_canonical_bytes(result.canonical_bytes) == result


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("qpos_after_f64", np.zeros(23, dtype=np.float64), "shape"),
        ("qvel_after_f64", np.zeros(23, dtype=np.float32), "dtype"),
        ("ctrl_f64", np.full(17, 0.4000001, dtype=np.float64), "bounds"),
        ("external_contact_wrench_f64", np.zeros((13, 6), dtype=np.float64), "shape"),
        ("control_period_s", 0.014, "0.015"),
        ("target_speed_m_s", 2.0, "one of"),
    ],
)
def test_shape_dtype_unit_or_cadence_drift_refuses(field: str, value: object, match: str) -> None:
    with pytest.raises(RewardContractError, match=match):
        _step(**{field: value})


def test_contract_copies_and_freezes_trusted_arrays() -> None:
    qpos = np.zeros(24, dtype=np.float64)
    qpos[3] = 1.0
    step = _step(qpos_after_f64=qpos)
    qpos[2] = 1.5
    assert step.qpos_after_f64[2] == 0.0
    with pytest.raises(ValueError):
        step.qpos_after_f64[2] = 1.5


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("qpos_after_f64", [False] * 24, "numbers, not booleans or strings"),
        ("qvel_after_f64", ["0.0"] * 23, "numbers, not booleans or strings"),
        ("external_contact_wrench_f64", [[0.0] * 6] * 13, "shape"),
        ("ctrl_f64", np.zeros(17, dtype=np.float32), "direct array.*float64"),
        ("qvel_after_f64", np.zeros(23, dtype=np.int64), "direct array.*float64"),
    ],
)
def test_wire_arrays_reject_type_structure_and_direct_dtype_repair(
    field: str,
    value: object,
    match: str,
) -> None:
    payload = _step().to_dict()
    payload[field] = value
    with pytest.raises(RewardContractError, match=match):
        TrustedRewardStepV1.from_dict(payload)


def test_wire_numeric_json_arrays_convert_only_after_exact_structure_check() -> None:
    payload = _step().to_dict()
    payload["qvel_after_f64"] = [0] * 23
    restored = TrustedRewardStepV1.from_dict(payload)
    assert restored.qvel_after_f64.dtype == np.dtype(np.float64)
    assert restored.qvel_after_f64.shape == (23,)


def test_reward_identity_hashes_every_declared_dependency_and_readmits() -> None:
    digest = "a" * 64
    source = b"def task_term(x):\n    return x.com_x_velocity_m_s\n"
    identity = RewardArtifactIdentityV1.create(
        candidate_source_bytes=source,
        target_speed_m_s=1.0,
        affine_alpha=1.0,
        affine_beta=0.0,
        compositor_source_sha256="b" * 64,
        gymnasium_source_sha256="c" * 64,
        humanoid_xml_sha256="d" * 64,
        dependency_hashes={"uv.lock": digest},
    )
    assert identity.candidate_source_sha256 == hashlib.sha256(source).hexdigest()
    assert identity.schema_sha256 == reward_schema_sha256()
    assert identity.read_set == CANDIDATE_READ_SET_V1
    assert RewardArtifactIdentityV1.from_canonical_bytes(identity.canonical_bytes) == identity

    changed = identity.to_dict()
    changed["affine_beta"] = 1.0
    with pytest.raises(RewardContractError, match="does not match"):
        RewardArtifactIdentityV1.from_dict(changed)


def test_result_envelope_is_fail_closed_and_never_repairs() -> None:
    with pytest.raises(RewardContractError, match="task term"):
        RewardResultV1(
            total=1001.0,
            signed_terms={
                "task_progress": 1001.0,
                "healthy": 0.0,
                "control": 0.0,
                "contact": 0.0,
            },
        )
    with pytest.raises(RewardContractError, match="total reward"):
        RewardResultV1(
            total=1025.0,
            signed_terms={
                "task_progress": 1000.0,
                "healthy": 5.0,
                "control": -0.0,
                "contact": -0.0,
            },
        )
    with pytest.raises(RewardContractError, match="reproduce"):
        RewardResultV1(
            total=1.0,
            signed_terms={
                "task_progress": 0.0,
                "healthy": 0.0,
                "control": 0.0,
                "contact": 0.0,
            },
        )


def test_candidate_input_requires_finite_built_in_compatible_scalars() -> None:
    assert CandidateTaskInputsV1(0.0, 1.0).canonical_bytes
    for value in (True, float("nan"), float("inf"), "1.0"):
        with pytest.raises(RewardContractError):
            CandidateTaskInputsV1(value, 1.0)  # type: ignore[arg-type]
