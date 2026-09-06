from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.phase_b.contracts import (
    PhaseBContractError,
    load_phase_b_oracle,
    t2_training_design_contract_value,
    validate_training_design,
)
from oracle_composition.phase_b.reference_runtime import (
    ComposedReferenceRuntime,
    tracking_state_from_reference_row,
)
from oracle_composition.reward_study.pairing import (
    fake_runtime_stream_receipt,
    summarize_fake_runtime_stream_receipt,
    validate_pairing_receipt,
)
from oracle_composition.reward_study.study_manifest import (
    T2StudyManifestError,
    load_t2_study_manifest,
    validate_t2_study_manifest,
)
from oracle_composition.reward_study.t2_evaluator import load_evaluator_design

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/004_t2_reward_study"
PAIRING_RECEIPT = ROOT / "artifacts/experiments_004/t2_pairing_adapter_receipt_v1.json"


def _signals(step: int) -> dict[str, object]:
    return {
        "dwell": step,
        "t": step,
        "torso_up": 1.0,
        "v_target": 3.0,
        "v_x": 3.0,
        "x_travelled": step * 0.045,
        "z_root": 1.4,
    }


def test_expert_hold_oracle_is_canonical_and_yields_rows_zero_through_1000() -> None:
    path = EXPERIMENT / "oracle_expert_hold_v1.json"
    program, digest = load_phase_b_oracle(
        path,
        available_behaviors=("expert", "medium", "simple"),
    )
    assert digest == "489b82591cf65034d58d30f921442a84c93c9c98885cb2f5b22dcaf645a77010"
    assert program.program.oracle_id == "expert_hold_v1"
    assert program.program.recovery is None
    assert not program.program.transitions
    rows = np.zeros((1_001, 45), dtype="<f8")
    rows[:, 0] = 1.4
    rows[:, 1] = 1.0
    rows[:, 5] = np.arange(1_001, dtype="<f8")
    runtime = ComposedReferenceRuntime(program, {"expert": rows})
    observed = []
    for step in range(1_001):
        frame = runtime.frame(
            state=tracking_state_from_reference_row(rows[step]),
            signals=_signals(step),
        )
        observed.append(frame.policy_window_indices[0])
        assert frame.behavior == "expert"
        assert frame.transfer is None
        assert frame.phase == step
        assert frame.policy_window[0] == pytest.approx(rows[step].astype("<f4"))
        if step < 1_000:
            runtime.advance()
    assert observed == list(range(1_001))
    assert runtime.transfer_logs == ()


def test_t2_training_design_is_phase_b_validated_without_changing_t1_bytes() -> None:
    t2_bytes = (EXPERIMENT / "training_design_t2_v1.json").read_bytes()
    t2_value = json.loads(t2_bytes)
    assert t2_bytes == canonical_json_bytes(t2_value)
    assert validate_training_design(t2_value) == t2_training_design_contract_value()
    assert hashlib.sha256(t2_bytes).hexdigest() == (
        "84543f08265dae5076697549f25ce69b7e5e87947d4bb6e32b86a7700d24e67e"
    )
    t1_path = ROOT / "experiments/003_composition_speed_profile/phase_b/training_design_v1.json"
    assert hashlib.sha256(t1_path.read_bytes()).hexdigest() == (
        "1d104a52238eee4f5065b4fcc30c285662e26ab06edd847da7ca9e160364c69e"
    )


def test_t2_training_design_refuses_rsi_family_drift() -> None:
    value = copy.deepcopy(t2_training_design_contract_value())
    value["rsi"]["rsi_family"] = "multi_behavior_rsi/v1"
    with pytest.raises(PhaseBContractError, match="semantics differ"):
        validate_training_design(value)


def test_evaluator_design_is_canonical_and_source_bound() -> None:
    value, digest = load_evaluator_design(
        EXPERIMENT / "evaluator_design_t2_v1.json",
        evaluator_source_path=ROOT / "src/oracle_composition/reward_study/t2_evaluator.py",
        protected_metrics_source_path=(
            ROOT / "src/oracle_composition/phase_b/protected_metrics.py"
        ),
        report_v2_source_path=ROOT / "src/oracle_composition/phase_b/report_v2.py",
        report_writer_source_path=ROOT / "src/oracle_composition/reward_study/t2_report.py",
    )
    assert digest == "d8f54f80051e4e3e58a540fc1414e9a971c9e299d903c2a00162bfebaa64a04b"
    assert value["reward_helpers_imported"] is False
    assert value["calibration"] == "none"


def test_study_manifest_verifies_common_files_and_pairing_key() -> None:
    value, digest = load_t2_study_manifest(
        EXPERIMENT / "t2_reward_study_expert_hold_v1.json",
        repository_root=ROOT,
    )
    assert digest == "7aefaafcef6d319dfd991b6a16eccc5f5578651d6ac8ebb3573358cc307f0738"
    assert value["study_pairing_sha256"] == (
        "4af9c9539bb2c7f9720f3e98a3f195c2b297d6bfe6adc51a1d63e7a8e4ad2464"
    )
    assert value["arms"][0]["reward"]["sha256"].startswith("eea2b6a9")
    assert value["arms"][1]["reward"]["sha256"] == "TBD"


def test_study_manifest_refuses_common_field_drift_even_with_a_rehashed_key() -> None:
    value = json.loads((EXPERIMENT / "t2_reward_study_expert_hold_v1.json").read_bytes())
    value["arms"][1]["training"]["ppo_seeds"][0] = 121002
    value["study_pairing_sha256"] = "0" * 64
    with pytest.raises(T2StudyManifestError, match="common field"):
        validate_t2_study_manifest(value)


def test_pairing_adapter_receipt_is_canonical_and_not_a_runtime_receipt() -> None:
    encoded = PAIRING_RECEIPT.read_bytes()
    value = json.loads(encoded)
    assert encoded == canonical_json_bytes(value)
    assert hashlib.sha256(encoded).hexdigest() == (
        "01c591c566554c386fbfa38cdfc1019579e20af4a3c9a3e72dca958b09fed1f1"
    )
    assert validate_pairing_receipt(value) == value
    assert (
        value["adapter_source_sha256"]
        == hashlib.sha256(
            (ROOT / "src/oracle_composition/reward_study/pairing.py").read_bytes()
        ).hexdigest()
    )
    recomputed = summarize_fake_runtime_stream_receipt(
        fake_runtime_stream_receipt(
            study_pairing_sha256=value["study_pairing_sha256"],
            ppo_seeds=[121001, 121101, 121201, 121301, 121401],
            evaluation_seeds=list(range(97001, 97021)),
        )
    )
    assert value["baseline_stream_receipt"] == recomputed
    assert value["candidate_stream_receipt"] == recomputed
    assert value["integrated_runtime_receipt"] == "TBD_pending_astra_acceptance"
