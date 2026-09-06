from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.phase_b.contracts import (
    PhaseBContractError,
    TargetSpeedRewardSpec,
    load_phase_b_oracle,
    t2_training_design_contract_value,
    validate_training_design,
)
from oracle_composition.phase_b.reference_runtime import (
    ComposedReferenceRuntime,
    tracking_state_from_reference_row,
)
from oracle_composition.reward_study.execution_manifest import (
    load_t2_execution_manifest,
)
from oracle_composition.reward_study.pairing import (
    validate_pairing_receipt,
)
from oracle_composition.reward_study.study_manifest import (
    T2StudyManifestError,
    load_t2_study_manifest,
    resolve_t2_reward_binding,
    validate_t2_study_manifest,
)
from oracle_composition.reward_study.t2_evaluator import (
    load_evaluator_design,
    load_t2_verified_reference,
)

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
    verified = load_t2_verified_reference(repository_root=ROOT, evaluation_seed=97001)
    rows = verified.rows
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


def test_pre_seam_execution_manifest_is_canonical_but_requires_pairing_reseal() -> None:
    manifest_path = EXPERIMENT / "execution_manifest_t2_v1.json"
    encoded = manifest_path.read_bytes()
    value = json.loads(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    assert encoded == canonical_json_bytes(value)
    assert digest == "d675b1ac02ad8713d995bd795ce2130acf41bc7a6fcc0a164167b24e3684ab3e"
    study = json.loads((EXPERIMENT / "t2_reward_study_expert_hold_v1.json").read_bytes())
    expected_binding = {
        "byte_count": manifest_path.stat().st_size,
        "path": "experiments/004_t2_reward_study/execution_manifest_t2_v1.json",
        "sha256": digest,
    }
    assert study["arms"][0]["execution_manifest"] == expected_binding
    assert study["arms"][1]["execution_manifest"] == expected_binding
    assert value["execution_manifest_schema_id"] == "t2_execution_manifest_v1"
    assert value["repository"]["commit"] == ("ed9f1d38aba4f7b41a576b0fd9c8be4f6b8b47fe")
    assert value["tracker"]["external_tracker_checkpoint"] is None
    assert value["normalizers"] == {"observation": None, "reward": None}
    with pytest.raises(ExperimentContractError, match="semantics or bindings differ"):
        load_t2_execution_manifest(manifest_path, repository_root=ROOT)


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
    assert digest == "7ae812d43c524285941cd567ce1663ff023cb6307229d9472a6dfe6577ebf5b3"
    assert value["reward_helpers_imported"] is False
    assert value["calibration"] == "none"


def test_study_manifest_verifies_common_files_and_pairing_key() -> None:
    path = EXPERIMENT / "t2_reward_study_expert_hold_v1.json"
    value, digest = load_t2_study_manifest(path)
    assert digest == "4eb3440b943355b8eee96e7663a4d542833464a8720d4b8ac102c050d9623627"
    assert value["study_pairing_sha256"] == (
        "fd91156a949a4484b497a112327db864c2a4cbcbaf0cf1cf5db75064f6a5b3e0"
    )
    assert value["arms"][0]["reward"]["sha256"].startswith("eea2b6a9")
    assert value["arms"][1]["reward"]["sha256"] == "TBD"
    assert (
        value["arms"][0]["pairing_adapter"]["sha256"]
        != hashlib.sha256(
            (ROOT / "src/oracle_composition/reward_study/pairing.py").read_bytes()
        ).hexdigest()
    )
    with pytest.raises(T2StudyManifestError, match="pairing_adapter artifact byte count differs"):
        load_t2_study_manifest(path, repository_root=ROOT)


def test_study_manifest_refuses_common_field_drift_even_with_a_rehashed_key() -> None:
    value = json.loads((EXPERIMENT / "t2_reward_study_expert_hold_v1.json").read_bytes())
    value["arms"][1]["training"]["ppo_seeds"][0] = 121002
    value["study_pairing_sha256"] = "0" * 64
    with pytest.raises(T2StudyManifestError, match="common field"):
        validate_t2_study_manifest(value)


def test_candidate_reward_resolves_through_registry_and_refuses_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidate.json"
    spec = TargetSpeedRewardSpec(alpha=1.0, beta=0.0)
    path.write_bytes(spec.canonical_bytes)
    digest = hashlib.sha256(spec.canonical_bytes).hexdigest()
    binding = {
        "path": path.name,
        "reward_id": "target_speed_triangular_affine_t2_adapter/v1",
        "sha256": digest,
    }
    resolved, observed, resolved_path = resolve_t2_reward_binding(
        binding, artifact_root=tmp_path, candidate=True
    )
    assert resolved == spec
    assert observed == digest
    assert resolved_path == path

    changed_digest = copy.deepcopy(binding)
    changed_digest["sha256"] = "0" * 64
    with pytest.raises(T2StudyManifestError, match="artifact SHA-256 differs"):
        resolve_t2_reward_binding(changed_digest, artifact_root=tmp_path, candidate=True)

    path.write_bytes(canonical_json_bytes({"formula_id": "forged"}))
    changed_file = copy.deepcopy(binding)
    changed_file["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(T2StudyManifestError, match="registry refused"):
        resolve_t2_reward_binding(changed_file, artifact_root=tmp_path, candidate=True)

    path.write_bytes(spec.canonical_bytes)

    class TamperedRegistry:
        def load(self, candidate: Path) -> tuple[object, str]:
            return spec, "f" * 64

    monkeypatch.setattr(
        "oracle_composition.reward_study.study_manifest.RewardRegistry",
        TamperedRegistry,
    )
    with pytest.raises(T2StudyManifestError, match="registry digest differs"):
        resolve_t2_reward_binding(binding, artifact_root=tmp_path, candidate=True)


def test_pre_seam_pairing_adapter_receipt_is_canonical_but_superseded() -> None:
    encoded = PAIRING_RECEIPT.read_bytes()
    value = json.loads(encoded)
    assert encoded == canonical_json_bytes(value)
    assert hashlib.sha256(encoded).hexdigest() == (
        "6bd6f5ab3eb33ea563fff06c01828781d4c01864b7f0384108bfa5161c02540a"
    )
    with pytest.raises(ValueError, match="pairing receipt identity differs"):
        validate_pairing_receipt(value)
    assert (
        value["adapter_source_sha256"]
        != hashlib.sha256(
            (ROOT / "src/oracle_composition/reward_study/pairing.py").read_bytes()
        ).hexdigest()
    )
    assert value["integrated_runtime_receipt"] == "TBD_pending_astra_acceptance"
