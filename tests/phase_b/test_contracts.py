from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.phase_b.contracts import (
    PHASE_POLICY,
    CandidateTaskInputsV2,
    FineTuningRunManifest,
    PhaseBContractError,
    PhaseBOracleProgram,
    RewardRegistry,
    StartingCheckpointContract,
    TrackingOnlyRewardSpec,
    load_fine_tuning_run_manifest,
    load_phase_b_oracle,
    validate_cycle_report,
)

ROOT = Path(__file__).resolve().parents[2]
PHASE_B = ROOT / "experiments/003_composition_speed_profile/phase_b"


def test_canonical_phase_b_contract_artifacts_and_hashes() -> None:
    oracle, file_hash = load_phase_b_oracle(
        PHASE_B / "oracle_cycle_1_reference_v1.json",
        available_behaviors=("expert", "medium", "simple"),
    )
    assert oracle.program.oracle_id == "cycle_1_candidate"
    assert oracle.to_dict()["phase_policy"] == dict(PHASE_POLICY)
    assert file_hash == oracle.sha256

    reward_bytes = (PHASE_B / "tracking_only_v1.json").read_bytes()
    reward_raw = json.loads(reward_bytes)
    reward = RewardRegistry().resolve(reward_raw)
    assert reward.to_dict()["r_task"] == 0.0
    assert reward.to_dict()["tracking_reward_config_sha256"]
    assert len(reward.registry_key) == 5
    assert all(len(digest) == 64 for digest in reward.registry_key)
    assert reward_bytes == canonical_json_bytes(reward_raw) == reward.canonical_bytes

    starting_raw = json.loads((PHASE_B / "starting_checkpoint_v1.json").read_bytes())
    assert StartingCheckpointContract.from_dict(starting_raw).value_initialization_seed == 20260905
    manifest_raw = json.loads((PHASE_B / "run_manifest_interface_check_v1.json").read_bytes())
    assert FineTuningRunManifest.from_dict(manifest_raw).value["ppo_seed"] == 121901
    loaded_manifest, manifest_sha256 = load_fine_tuning_run_manifest(
        PHASE_B / "run_manifest_interface_check_v1.json",
        repository_root=ROOT,
    )
    assert loaded_manifest.value["ppo_seed"] == 121901
    assert manifest_sha256 == FineTuningRunManifest.from_dict(manifest_raw).sha256


def test_run_manifest_refuses_a_bound_hash_mismatch(tmp_path: Path) -> None:
    raw = json.loads((PHASE_B / "run_manifest_interface_check_v1.json").read_bytes())
    raw["library"]["sha256"] = "0" * 64
    path = tmp_path / "manifest.json"
    path.write_bytes(canonical_json_bytes(raw))
    with pytest.raises(PhaseBContractError, match="library artifact SHA-256 differs"):
        load_fine_tuning_run_manifest(path, repository_root=ROOT)


def test_oracle_refuses_wrong_schema() -> None:
    raw = json.loads((PHASE_B / "oracle_cycle_1_reference_v1.json").read_bytes())
    raw["oracle_schema_id"] = "humanoid_reference_composition_oracle/v2"
    with pytest.raises(PhaseBContractError, match="schema identity"):
        PhaseBOracleProgram.from_dict(
            raw,
            available_behaviors=("expert", "medium", "simple"),
        )


def test_reward_registry_refuses_unknown_formula_and_schema() -> None:
    baseline = TrackingOnlyRewardSpec().to_dict()
    wrong_formula = {**baseline, "formula_id": "target_speed/unknown"}
    with pytest.raises(PhaseBContractError, match="unknown reward formula_id"):
        RewardRegistry().resolve(wrong_formula)
    wrong_schema = {**baseline, "reward_schema_id": "reward_specification/unknown/v1"}
    with pytest.raises(PhaseBContractError, match="specification differs"):
        RewardRegistry().resolve(wrong_schema)


def test_candidate_inputs_are_float64_bounded_and_constant() -> None:
    previous = np.asarray([1.0, 2.0, 3.0], dtype="<f8")
    current = np.asarray([1.045, 2.0, 3.0], dtype="<f8")
    inputs = CandidateTaskInputsV2.from_mass_center_state(previous, current)
    assert inputs.com_x_velocity_m_s == pytest.approx(3.0)
    assert set(inputs.to_dict()) == {
        "cadence_seconds",
        "com_x_velocity_m_s",
        "input_schema_id",
        "target_speed_m_s",
    }
    with pytest.raises(PhaseBContractError, match=r"within \[-25, 25\]"):
        CandidateTaskInputsV2(com_x_velocity_m_s=np.float64(25.0001))
    with pytest.raises(PhaseBContractError, match="exact finite float64"):
        CandidateTaskInputsV2(
            com_x_velocity_m_s=np.float64(0.0),
            target_speed_m_s=np.float64("nan"),
        )


def _minimal_v2_report() -> dict[str, object]:
    v1 = json.loads(
        (
            ROOT / "experiments/003_composition_speed_profile/cycles/cycle_0/report_0.json"
        ).read_text()
    )
    v1.update(
        {
            "claim_ceiling": (
                "exploratory_reference_conditioned_fine_tuning_utility_only_"
                "no_causal_reference_use_oracle_improvement_reward_improvement_"
                "generalization_naturalness_or_humanoid_competence_claim"
            ),
            "inputs": {
                "evaluator_sha256": "0" * 64,
                "library_sha256": "0" * 64,
                "oracle_canonical_sha256": "0" * 64,
                "oracle_file_sha256": "0" * 64,
                "reference_sha256": "0" * 64,
                "reward_compositor_sha256": "0" * 64,
                "reward_file_sha256": "0" * 64,
                "reward_formula_id": "tracking_only/v1",
                "reward_formula_sha256": "0" * 64,
                "reward_schema_id": "reward_specification/tracking_only/v1",
                "task_sha256": "0" * 64,
                "training_design_sha256": "0" * 64,
            },
            "integrity": {
                "deterministic_reload": True,
                "explicit_missing_fields": [],
                "failure_receipt": None,
                "trace_index_sha256": "0" * 64,
            },
            "policy": {
                "e1_receipt_sha256": "0" * 64,
                "final_step": 0,
                "full_checkpoint_sha256": "0" * 64,
                "ppo_seed": 121901,
                "starting_expert_identity": {},
                "strict_export_sha256": "0" * 64,
            },
            "reference_runtime": {"records": []},
            "report_schema_id": "humanoid_composition_cycle_report/v2",
            "reward_runtime": {
                "ignored_stock_reward": 100.0,
                "parameters": {},
                "r_task": 0.0,
                "r_track": 0.75,
                "r_train": 0.75,
            },
            "schema_version": 2,
            "training": {
                "disk_bytes": 0,
                "losses": {},
                "observed_transitions": 0,
                "peak_rss_bytes": 0,
                "planned_transitions": 0,
                "rollouts": 0,
                "rsi_ledger_sha256": "0" * 64,
                "stream_counts": {},
                "throughput_steps_s": 0.0,
                "unfreeze_receipt_sha256": "0" * 64,
                "updates": 0,
                "wall_time_seconds": 0.0,
            },
        }
    )
    for episode in v1["per_episode"]:
        episode.update(
            {
                "com_forward_speed_m_s": [],
                "contacts": [],
                "fall": episode.get("fall", False),
                "resynchronization_records": [],
                "root_delta_forward_speed_m_s": [],
                "six_tracking_errors": [],
                "switch_records": [],
            }
        )
    v1["summary"].update(
        {
            "distributions_by_arm": {},
            "distributions_by_cell": {},
            "hard_gates": {},
            "policy_seeds": [121901],
            "step_zero_comparator": {},
        }
    )
    return v1


def test_report_v2_missing_field_refused_and_v1_still_validates() -> None:
    v1 = json.loads(
        (
            ROOT / "experiments/003_composition_speed_profile/cycles/cycle_0/report_0.json"
        ).read_text()
    )
    assert validate_cycle_report(v1)["report_schema_id"].endswith("/v1")
    v2 = _minimal_v2_report()
    assert validate_cycle_report(v2)["report_schema_id"].endswith("/v2")
    missing = copy.deepcopy(v2)
    del missing["policy"]["strict_export_sha256"]
    with pytest.raises(PhaseBContractError, match="policy is missing"):
        validate_cycle_report(missing)


def test_report_reward_streams_cannot_be_collapsed() -> None:
    value = _minimal_v2_report()
    value["reward_runtime"]["r_train"] = 100.75
    with pytest.raises(PhaseBContractError, match=r"r_track \+ r_task"):
        validate_cycle_report(value)


def test_report_reference_record_must_expose_runtime_evidence() -> None:
    value = _minimal_v2_report()
    value["reference_runtime"]["records"] = [{"behavior": "expert"}]
    with pytest.raises(PhaseBContractError, match="reference record is incomplete"):
        validate_cycle_report(value)
