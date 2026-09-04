from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.envs.humanoid import _instrumented_humanoid_class
from oracle_composition.experiments import (
    tqc_instrumentation_equivalence_v2 as instrumentation_module,
)
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_calibration_contract import canonical_json
from oracle_composition.experiments.tqc_development_contract_v2 import (
    load_tqc_development_design_v2,
)
from oracle_composition.experiments.tqc_instrumentation_equivalence_v2 import (
    COMPARISON_FIELDS,
    EXPECTED_ACTION_SHA256,
    INSTRUMENTATION_HORIZON,
    INSTRUMENTATION_SEED,
    TQCInstrumentationEquivalenceReceiptV2,
    run_tqc_instrumentation_equivalence_v2,
    tqc_instrumentation_actions_v2,
    validate_tqc_instrumentation_equivalence_receipt_v2,
)

ROOT = Path(__file__).parents[2]
CONFIG_ROOT = ROOT / "experiments/bootstrap_tqc_humanoid/configs"


def _design():
    return load_tqc_development_design_v2(
        CONFIG_ROOT / "tqc_base_controller_dev_1m_v2.study.json",
        CONFIG_ROOT / "tqc_e0_reuse_dev_1m_v2.audit.json",
    )


def _binding() -> dict[str, object]:
    return {
        "attempt_id": instrumentation_module.ATTEMPT_ID,
        "attempt_nonce": "a" * 64,
        "worker_pid": os.getpid(),
        "worker_process_start_monotonic_seconds": 1.0,
        "preflight_contract_sha256": "b" * 64,
        "claimed_work_directory_identity": "c" * 64,
        "project_python_source_tree_sha256": instrumentation_module.source_tree_sha256(),
    }


def _run():
    return run_tqc_instrumentation_equivalence_v2(
        _design(),
        receipt_binding=_binding(),
    )


def _readmit(encoded: bytes):
    return validate_tqc_instrumentation_equivalence_receipt_v2(
        encoded,
        _design(),
        expected_binding=_binding(),
    )


def test_frozen_instrumentation_actions_match_predeclaration() -> None:
    actions = tqc_instrumentation_actions_v2()

    assert actions.shape == (INSTRUMENTATION_HORIZON, 17)
    assert actions.dtype == np.dtype("<f4")
    assert actions.flags.c_contiguous
    assert float(actions.min()) >= -0.4
    assert float(actions.max()) < 0.4
    gate = _design().to_dict()["evaluation"]["instrumentation_equivalence_gate"]
    assert gate["seed"] == INSTRUMENTATION_SEED
    assert gate["action_sha256"] == EXPECTED_ACTION_SHA256


@pytest.mark.gym
def test_exact_instrumentation_canary_passes_and_readmits() -> None:
    design = _design()
    receipt = run_tqc_instrumentation_equivalence_v2(
        design,
        receipt_binding=_binding(),
    )
    value = receipt.to_dict()

    assert receipt.all_comparisons_passed is True
    assert value["reset_comparison_count"] == 1
    assert value["per_step_comparison_count"] == INSTRUMENTATION_HORIZON
    assert value["comparison_fields"] == list(COMPARISON_FIELDS)
    assert value["plain_shared_trace_sha256"] == value["instrumented_shared_trace_sha256"]
    assert value["mismatch_count"] == 0
    assert value["first_mismatch"] is None
    assert value["behavioral_evidence"] is False
    assert {name: value[name] for name in _binding()} == _binding()
    assert receipt.authorizes_execution is False

    readmitted = validate_tqc_instrumentation_equivalence_receipt_v2(
        receipt.canonical_bytes,
        design,
        expected_binding=_binding(),
    )
    assert readmitted == receipt


@pytest.mark.gym
def test_instrumentation_canary_detects_changed_dynamics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instrumented_type = _instrumented_humanoid_class()
    original = instrumented_type._step_mujoco_simulation

    def changed_step(self: object, ctrl: object, n_frames: int) -> None:
        original(self, ctrl, n_frames)
        self.data.qpos[0] += 1e-12

    monkeypatch.setattr(instrumented_type, "_step_mujoco_simulation", changed_step)
    with pytest.raises(ExperimentContractError, match="live method implementation differs"):
        _run()


@pytest.mark.gym
def test_instrumentation_canary_rejects_base_dynamics_with_empty_contact_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv

    instrumented_type = _instrumented_humanoid_class()

    def hidden_capture_omission(self: object, ctrl: object, n_frames: int) -> None:
        HumanoidEnv._step_mujoco_simulation(self, ctrl, n_frames)
        self._last_control_step_contact_samples = ()
        self._last_control_step_substeps = n_frames

    monkeypatch.setattr(
        instrumented_type,
        "_step_mujoco_simulation",
        hidden_capture_omission,
    )
    with pytest.raises(ExperimentContractError, match="live method implementation differs"):
        _run()


@pytest.mark.gym
def test_instrumentation_canary_rejects_falsified_contact_force_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mujoco

    observed_calls = 0

    def zero_contact_force(
        _model: object,
        _data: object,
        _contact_index: int,
        result: np.ndarray,
    ) -> None:
        nonlocal observed_calls
        observed_calls += 1
        result[:] = 0.0

    monkeypatch.setattr(mujoco, "mj_contactForce", zero_contact_force)
    with pytest.raises(ExperimentContractError, match="contact trace differs"):
        _run()
    assert observed_calls > 0


@pytest.mark.gym
def test_instrumentation_receipt_cannot_be_publicly_forged() -> None:
    receipt = _run()

    with pytest.raises(ExperimentContractError, match="only be issued"):
        TQCInstrumentationEquivalenceReceiptV2(**dataclasses.asdict(receipt))


@pytest.mark.gym
def test_readmission_rejects_changed_source_binding() -> None:
    design = _design()
    receipt = run_tqc_instrumentation_equivalence_v2(
        design,
        receipt_binding=_binding(),
    )
    value = receipt.to_dict()
    value["environment_instrumentation_equivalence_source_sha256"] = "0" * 64

    with pytest.raises(ExperimentContractError, match="source SHA-256"):
        _readmit(canonical_json(value))


@pytest.mark.gym
def test_readmission_rejects_duplicate_json_key() -> None:
    receipt = _run()
    encoded = receipt.canonical_bytes
    payload = json.loads(encoded)
    first_key = next(iter(payload))
    duplicate = b"{" + json.dumps(first_key).encode() + b":null," + encoded[1:]

    with pytest.raises(ExperimentContractError, match="duplicate JSON key"):
        _readmit(duplicate)


@pytest.mark.gym
def test_readmission_rejects_false_receipt_with_zero_mismatches() -> None:
    design = _design()
    receipt = run_tqc_instrumentation_equivalence_v2(
        design,
        receipt_binding=_binding(),
    )
    value = receipt.to_dict()
    value["all_comparisons_passed"] = False

    with pytest.raises(ExperimentContractError, match="failed instrumentation receipt"):
        _readmit(canonical_json(value))


@pytest.mark.gym
def test_instrumentation_receipt_rejects_attempt_replay_and_fabricated_proofs() -> None:
    receipt = _run()
    replayed_binding = {**_binding(), "attempt_nonce": "d" * 64}
    with pytest.raises(ExperimentContractError, match="preflight binding"):
        validate_tqc_instrumentation_equivalence_receipt_v2(
            receipt.canonical_bytes,
            _design(),
            expected_binding=replayed_binding,
        )

    value = receipt.to_dict()
    value["structural_source_proof_sha256"] = "0" * 64
    value["plain_shared_trace_sha256"] = "1" * 64
    value["instrumented_shared_trace_sha256"] = "1" * 64
    with pytest.raises(ExperimentContractError, match="structural source proof"):
        _readmit(canonical_json(value))


@pytest.mark.gym
def test_instrumentation_receipt_rejects_nested_numeric_type_changes() -> None:
    receipt = _run()
    value = receipt.to_dict()
    value["action_shape"] = [float(INSTRUMENTATION_HORIZON), 17.0]

    with pytest.raises(ExperimentContractError, match=r"action_shape\[0\]"):
        _readmit(canonical_json(value))


@pytest.mark.gym
def test_structural_proof_rejects_an_unlisted_dunder_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instrumented_type = _instrumented_humanoid_class()
    monkeypatch.setattr(instrumented_type, "__repr__", object.__repr__, raising=False)

    with pytest.raises(ExperimentContractError, match="structural source proof"):
        _run()


@pytest.mark.gym
def test_instrumentation_probe_rejects_nonself_worker_binding() -> None:
    binding = {**_binding(), "worker_pid": os.getpid() + 100_000}

    with pytest.raises(ExperimentContractError, match="not this process"):
        run_tqc_instrumentation_equivalence_v2(
            _design(),
            receipt_binding=binding,
        )


@pytest.mark.gym
def test_instrumentation_probe_detects_source_tree_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding()
    monkeypatch.setattr(instrumentation_module, "source_tree_sha256", lambda: "0" * 64)

    with pytest.raises(ExperimentContractError, match="project Python source tree"):
        run_tqc_instrumentation_equivalence_v2(
            _design(),
            receipt_binding=binding,
        )
