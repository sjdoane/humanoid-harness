from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.development.reference_ablation import (
    CLAIM_CEILING,
    DEVELOPMENT_BLOCKS,
    DevelopmentAblationError,
    load_bound_smoke_actor,
    load_development_protocol,
    matched_action_delta,
    prepare_reference_transform,
    prepared_reference_window,
    summarize_matched_action_deltas,
    validate_smoke_export,
)
from oracle_composition.experiments.reference_input_transforms import (
    CONDITION_IDS,
    transform_reference_input,
)
from oracle_composition.phase_b.training import SMOKE_SEED, SMOKE_TRANSITIONS

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = (
    REPOSITORY_ROOT / "experiments/003_composition_speed_profile/phase_b/"
    "development_reference_ablation_v1.json"
)


def _write_json(path: Path, value: object) -> bytes:
    encoded = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return encoded


def _record(path: Path) -> dict[str, object]:
    encoded = path.read_bytes()
    return {
        "byte_count": len(encoded),
        "filename": path.name,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _smoke_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "smoke"
    seed = root / f"seed_{SMOKE_SEED}"
    seed.mkdir(parents=True)
    manifest_bytes = _write_json(
        root / "execution_manifest_v3.json",
        {
            "checkpoint_selection": "final_transition_only",
            "evidence_class": "interface_check",
            "execution_manifest_schema_id": "humanoid_phase_b_execution_manifest/v3",
            "schema_version": 3,
            "seeds": [SMOKE_SEED],
            "smoke": True,
            "test_only": False,
            "transitions_per_seed": SMOKE_TRANSITIONS,
        },
    )
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    rsi_bytes = _write_json(seed / "rsi_ledger_v1.json", [{"global_episode_index": 0}])
    training_bytes = _write_json(
        seed / "training_facts_v1.json",
        {
            "evidence_class": "interface_check",
            "execution_manifest_sha256": manifest_sha,
            "observed_transitions": SMOKE_TRANSITIONS,
            "planned_transitions": SMOKE_TRANSITIONS,
            "ppo_seed": SMOKE_SEED,
            "promotable": False,
            "rollouts": 24,
            "rsi_ledger_sha256": hashlib.sha256(rsi_bytes).hexdigest(),
            "smoke": True,
            "unfreeze_rollouts": [
                {
                    "actor_stage": ("reference_columns_only" if index < 8 else "full_actor"),
                    "rollout_index": index,
                }
                for index in range(24)
            ],
        },
    )
    training_sha = hashlib.sha256(training_bytes).hexdigest()
    actor = seed / f"actor_seed_{SMOKE_SEED}_final.npz"
    actor.write_bytes(b"strict actor fixture")
    checkpoint = seed / f"checkpoint_seed_{SMOKE_SEED}_final.npz"
    checkpoint.write_bytes(b"full checkpoint fixture")
    _write_json(
        seed / f"persistence_seed_{SMOKE_SEED}_v1.json",
        {
            "checkpoint": _record(checkpoint),
            "checkpoint_reload_bitwise_deterministic": True,
            "checkpoint_to_export_bitwise_equivalent": True,
            "evidence_class": "interface_check",
            "execution_manifest_sha256": manifest_sha,
            "final_transition_only": True,
            "persistence_receipt_id": "humanoid_phase_b_final_persistence/v1",
            "planned_transitions": SMOKE_TRANSITIONS,
            "ppo_seed": SMOKE_SEED,
            "promotable": False,
            "schema_version": 1,
            "smoke": True,
            "strict_export": _record(actor),
            "test_only": False,
            "training_facts_sha256": training_sha,
            "transitions": SMOKE_TRANSITIONS,
        },
    )
    success = seed / "success_receipt_v2.json"
    _write_json(
        success,
        {
            "artifacts": {
                "persistence": _record(seed / f"persistence_seed_{SMOKE_SEED}_v1.json"),
                "rsi_ledger": _record(seed / "rsi_ledger_v1.json"),
                "training_facts": _record(seed / "training_facts_v1.json"),
            },
            "evidence_class": "interface_check",
            "execution_manifest_sha256": manifest_sha,
            "failure_receipt_present": False,
            "outcome": "success",
            "planned_transitions": SMOKE_TRANSITIONS,
            "ppo_seed": SMOKE_SEED,
            "promotable": False,
            "schema_version": 2,
            "smoke": True,
            "status": "succeeded",
            "success_receipt_id": "humanoid_phase_b_seed_success/v2",
            "test_only": False,
        },
    )
    _write_json(
        root / "job_result_v1.json",
        {
            "checkpoint_index": None,
            "execution_manifest_sha256": manifest_sha,
            "job_result_schema_id": "humanoid_phase_b_job_result/v1",
            "outcomes": [
                {
                    "receipt": _record(success),
                    "seed": SMOKE_SEED,
                    "status": "succeeded",
                }
            ],
            "schema_version": 1,
            "status": "succeeded",
        },
    )
    return root


def _reference() -> np.ndarray:
    value = np.zeros((1001, 45), dtype="<f8")
    value[:, 0] = 1.4 + np.arange(1001, dtype=np.float64) / 10_000.0
    value[:, 1] = 1.0
    value[:, 11] = np.arange(1001, dtype=np.float64) / 1_000.0
    return value


def test_preregistered_protocol_is_direct_expert_simple_expert_only() -> None:
    protocol = load_development_protocol(PROTOCOL_PATH)

    assert protocol.development_blocks == DEVELOPMENT_BLOCKS
    assert protocol.conditions == CONDITION_IDS
    assert [row["behavior"] for row in protocol.value["reference_schedule"]] == [
        "expert",
        "simple",
        "expert",
    ]
    assert protocol.value["calibration_inputs"] == "forbidden"
    assert protocol.value["held_out_inputs_used"] is False
    assert protocol.value["promotable"] is False
    assert protocol.value["claim_ceiling"] == CLAIM_CEILING


def test_protocol_rejects_medium_or_a_protected_block(tmp_path: Path) -> None:
    value = json.loads(PROTOCOL_PATH.read_bytes())
    value["reference_schedule"][1]["behavior"] = "medium"
    path = tmp_path / "medium.json"
    _write_json(path, value)
    with pytest.raises(DevelopmentAblationError, match="expert-simple-expert"):
        load_development_protocol(path)

    value = json.loads(PROTOCOL_PATH.read_bytes())
    value["development_blocks"][-1] = 120101
    path = tmp_path / "protected.json"
    _write_json(path, value)
    with pytest.raises(DevelopmentAblationError, match="arms or blocks differ"):
        load_development_protocol(path)


def test_smoke_lineage_loads_only_the_receipt_bound_strict_actor(tmp_path: Path) -> None:
    lineage = validate_smoke_export(_smoke_fixture(tmp_path))
    calls = []
    sentinel = object()

    def loader(path: Path, *, expected_sha256: str) -> object:
        calls.append((path, expected_sha256))
        return sentinel

    assert load_bound_smoke_actor(lineage, loader=loader) is sentinel
    assert calls == [(lineage.actor_path, lineage.actor_sha256)]
    assert lineage.actor_path.name == f"actor_seed_{SMOKE_SEED}_final.npz"
    assert lineage.execution_manifest_sha256 == lineage.bindings["execution_manifest"]["sha256"]


def test_smoke_lineage_rejects_promotion_or_tampered_export(tmp_path: Path) -> None:
    root = _smoke_fixture(tmp_path / "promoted")
    success_path = root / f"seed_{SMOKE_SEED}/success_receipt_v2.json"
    success = json.loads(success_path.read_bytes())
    success["promotable"] = True
    _write_json(success_path, success)
    job_path = root / "job_result_v1.json"
    job = json.loads(job_path.read_bytes())
    job["outcomes"][0]["receipt"] = _record(success_path)
    _write_json(job_path, job)
    with pytest.raises(DevelopmentAblationError, match="successful real T1 smoke"):
        validate_smoke_export(root)

    root = _smoke_fixture(tmp_path / "tampered")
    (root / f"seed_{SMOKE_SEED}/actor_seed_{SMOKE_SEED}_final.npz").write_bytes(b"tampered actor")
    with pytest.raises(DevelopmentAblationError, match="artifact binding differs"):
        validate_smoke_export(root)


@pytest.mark.parametrize("condition", CONDITION_IDS)
@pytest.mark.parametrize("frame", (0, 300, 600, 1000))
def test_prepared_windows_match_the_frozen_transform(condition: str, frame: int) -> None:
    source = _reference()
    prepared = prepare_reference_transform(source, condition_id=condition)
    observed = prepared_reference_window(prepared, current_frame=frame)
    direct = transform_reference_input(
        source,
        condition_id=condition,
        current_frame=frame,
    )

    np.testing.assert_array_equal(observed.values, direct.window)
    assert observed.timeline_indices == direct.receipt.window_timeline_indices
    assert observed.source_indices == direct.receipt.window_source_indices
    assert observed.sha256 == direct.receipt.output_window_sha256
    assert observed.values.flags.writeable is False


def test_matched_state_action_deltas_retain_every_condition() -> None:
    exact = np.zeros(17, dtype="<f4")
    shifted = np.full(17, 0.1, dtype="<f4")
    rows = []
    for condition in CONDITION_IDS:
        delta = matched_action_delta(
            exact,
            exact if condition == CONDITION_IDS[0] else shifted,
        )
        rows.append({"condition_id": condition, "step": 0, **delta})

    summary = summarize_matched_action_deltas(rows)

    assert tuple(summary) == CONDITION_IDS
    assert summary[CONDITION_IDS[0]]["bitwise_changed_steps"] == 0
    for condition in CONDITION_IDS[1:]:
        assert summary[condition]["bitwise_changed_steps"] == 1
        assert summary[condition]["maximum_max_abs_physical_action"] == pytest.approx(0.1)


def test_matched_state_summary_fails_when_an_arm_is_missing() -> None:
    exact = np.zeros(17, dtype="<f4")
    delta = matched_action_delta(exact, exact)

    with pytest.raises(DevelopmentAblationError, match="condition has no rows"):
        summarize_matched_action_deltas([{"condition_id": CONDITION_IDS[0], "step": 0, **delta}])
