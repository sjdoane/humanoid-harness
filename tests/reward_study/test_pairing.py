from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.phase_b.training import domain_separated_seed
from oracle_composition.reward_study.pairing import (
    PAIRING_ADAPTER_ID,
    PAIRING_DERIVATION_ID,
    TRAINING_STREAM_DOMAINS,
    derive_t2_rsi_assignment,
    derive_training_stream_seeds,
    fake_runtime_stream_receipt,
    summarize_fake_runtime_stream_receipt,
    validate_pairing_receipt,
)
from oracle_composition.reward_study.study_manifest import (
    STUDY_CLAIM_CEILING,
    STUDY_EVIDENCE_CLASS,
    T2StudyManifestError,
    study_pairing_sha256_from_arm,
    validate_t2_study_arm_pair,
)

ROOT = Path(__file__).resolve().parents[2]


def _binding(name: str) -> dict[str, object]:
    return {
        "byte_count": 1,
        "path": f"{name}.json",
        "sha256": hashlib.sha256(name.encode()).hexdigest(),
    }


def _arm(label: str) -> dict[str, object]:
    reward = (
        {
            "path": "experiments/003_composition_speed_profile/phase_b/tracking_only_v1.json",
            "reward_id": "tracking_only/v1",
            "sha256": "eea2b6a9893e6e4ca5aea5d6787580f12062084e758cc9c27db2f1376bcb1c5f",
        }
        if label == "baseline"
        else {
            "path": "TBD",
            "reward_id": "target_speed_triangular_affine_t2_adapter/v1",
            "sha256": "TBD",
        }
    )
    return {
        "arm_label": label,
        "calibration": "none",
        "claim_ceiling": STUDY_CLAIM_CEILING,
        "evaluation": {
            "deterministic_actions": True,
            "evaluation_seeds": list(range(97001, 97021)),
            "evaluator_id": "t2_direct_state_reward_telemetry_separated_evaluator/v1",
            "expert_start": True,
            "horizon_steps": 1_000,
            "report_schema_id": "t2_reward_study_report/v1",
        },
        "evaluator": _binding("a-evaluator"),
        "evidence_class": STUDY_EVIDENCE_CLASS,
        "execution_manifest": _binding("g-execution"),
        "library": _binding("b-library"),
        "oracle": _binding("c-oracle"),
        "output_path": "TBD",
        "pairing": {
            "adapter_id": PAIRING_ADAPTER_ID,
            "declared": True,
            "derivation_id": PAIRING_DERIVATION_ID,
        },
        "pairing_adapter": _binding("a-pairing"),
        "reference_corpus": _binding("d-corpus"),
        "reward": reward,
        "starting_checkpoint": _binding("e-checkpoint"),
        "task": {
            "environment_id": "Humanoid-v5",
            "expert_start": True,
            "horizon_steps": 1_000,
            "target_com_forward_speed_m_s": 3.0,
        },
        "timestamps": {"completed_utc": "TBD", "started_utc": "TBD"},
        "training": {
            "checkpoint_selection": "final_transition_only",
            "environment_count": 4,
            "ppo_seeds": [121001, 121101, 121201, 121301, 121401],
            "retries_or_seed_replacement": False,
            "transitions_per_seed": 1_048_576,
        },
        "training_design": _binding("f-training"),
    }


def test_reward_only_fake_runtime_arms_have_identical_stream_receipts() -> None:
    baseline = _arm("baseline")
    candidate = _arm("candidate")
    common = validate_t2_study_arm_pair(baseline, candidate)
    pairing = hashlib.sha256(canonical_json_bytes(common)).hexdigest()
    assert pairing == study_pairing_sha256_from_arm(baseline)
    assert pairing == study_pairing_sha256_from_arm(candidate)
    baseline_receipt = fake_runtime_stream_receipt(
        study_pairing_sha256=pairing,
        ppo_seeds=[121001, 121101, 121201, 121301, 121401],
        evaluation_seeds=list(range(97001, 97021)),
    )
    candidate_receipt = fake_runtime_stream_receipt(
        study_pairing_sha256=pairing,
        ppo_seeds=[121001, 121101, 121201, 121301, 121401],
        evaluation_seeds=list(range(97001, 97021)),
    )
    assert baseline_receipt == candidate_receipt


def test_changed_common_field_changes_pairing_key_and_streams() -> None:
    arm = _arm("baseline")
    pairing = study_pairing_sha256_from_arm(arm)
    changed = copy.deepcopy(arm)
    changed["task"]["target_com_forward_speed_m_s"] = 3.1
    changed_pairing = study_pairing_sha256_from_arm(changed)
    assert changed_pairing != pairing
    original = fake_runtime_stream_receipt(
        study_pairing_sha256=pairing,
        ppo_seeds=[121001],
        evaluation_seeds=[97001],
    )
    changed_streams = fake_runtime_stream_receipt(
        study_pairing_sha256=changed_pairing,
        ppo_seeds=[121001],
        evaluation_seeds=[97001],
    )
    assert original["streams_sha256"] != changed_streams["streams_sha256"]


def test_t2_rsi_assignment_derives_expert_block_order_class_and_start() -> None:
    pairing = study_pairing_sha256_from_arm(_arm("baseline"))
    first = derive_t2_rsi_assignment(
        study_pairing_sha256=pairing,
        ppo_seed=121001,
        global_episode_index=0,
        environment_index=2,
    )
    replay = derive_t2_rsi_assignment(
        study_pairing_sha256=pairing,
        ppo_seed=121001,
        global_episode_index=0,
        environment_index=2,
    )
    assert first == replay
    assert first.reference_behavior == "expert"
    assert first.schedule_class == "hold"
    assert first.block in {120001, 120002, 120003, 120005, 120007, 120008, 120009, 120011, 120012}
    assert 0 <= first.start_boundary < 489

    first_cycle = [
        derive_t2_rsi_assignment(
            study_pairing_sha256=pairing,
            ppo_seed=121001,
            global_episode_index=index,
            environment_index=2 + (index % 2),
        )
        for index in range(9)
    ]
    assert {item.block for item in first_cycle} == {
        120001,
        120002,
        120003,
        120005,
        120007,
        120008,
        120009,
        120011,
        120012,
    }
    assert {item.reference_behavior for item in first_cycle} == {"expert"}
    assert {item.schedule_class for item in first_cycle} == {"hold"}


def test_study_pair_validation_refuses_nonreward_arm_drift() -> None:
    baseline = _arm("baseline")
    candidate = _arm("candidate")
    candidate["training"]["transitions_per_seed"] = 1_048_575
    with pytest.raises(T2StudyManifestError, match="common field"):
        validate_t2_study_arm_pair(baseline, candidate)


def test_pairing_declaration_refuses_arm_specific_manifest_derivation() -> None:
    with pytest.raises(ValueError, match="refuses arm-specific manifest"):
        derive_training_stream_seeds(
            ppo_seed=121001,
            pairing_declared=True,
            study_pairing_sha256="a" * 64,
            manifest_sha256="b" * 64,
        )


def test_single_arm_adapter_preserves_manifest_hash_derivation() -> None:
    manifest = "a" * 64
    observed = derive_training_stream_seeds(
        ppo_seed=121001,
        pairing_declared=False,
        manifest_sha256=manifest,
    )
    assert observed == {
        domain: domain_separated_seed(manifest, 121001, domain)
        for domain in TRAINING_STREAM_DOMAINS
    }


def test_pairing_receipt_validator_separates_adapter_from_runtime_receipt() -> None:
    baseline = _arm("baseline")
    pairing = study_pairing_sha256_from_arm(baseline)
    changed = copy.deepcopy(baseline)
    changed["task"]["target_com_forward_speed_m_s"] = 3.1
    changed_pairing = study_pairing_sha256_from_arm(changed)
    stream = summarize_fake_runtime_stream_receipt(
        fake_runtime_stream_receipt(
            study_pairing_sha256=pairing,
            ppo_seeds=[121001, 121101, 121201, 121301, 121401],
            evaluation_seeds=list(range(97001, 97021)),
        )
    )
    value = {
        "adapter_id": PAIRING_ADAPTER_ID,
        "adapter_source_sha256": hashlib.sha256(
            (ROOT / "src/oracle_composition/reward_study/pairing.py").read_bytes()
        ).hexdigest(),
        "arms_differ_only_in_reward": True,
        "baseline_stream_receipt": stream,
        "candidate_stream_receipt": copy.deepcopy(stream),
        "changed_common_field": "task.target_com_forward_speed_m_s:3.0_to_3.1",
        "changed_common_field_changes_key": True,
        "changed_common_pairing_sha256": changed_pairing,
        "changed_common_stream_receipt": summarize_fake_runtime_stream_receipt(
            fake_runtime_stream_receipt(
                study_pairing_sha256=changed_pairing,
                ppo_seeds=[121001, 121101, 121201, 121301, 121401],
                evaluation_seeds=list(range(97001, 97021)),
            )
        ),
        "derivation_id": PAIRING_DERIVATION_ID,
        "evidence_class": "interface_check",
        "integrated_runtime_receipt": "TBD_pending_astra_acceptance",
        "schema_version": 1,
        "study_pairing_sha256": pairing,
    }
    assert validate_pairing_receipt(value) == value
    value["adapter_source_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="adapter source SHA-256 differs"):
        validate_pairing_receipt(value)
