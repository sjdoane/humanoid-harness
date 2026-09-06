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
    validate_development_corpus,
    validate_smoke_export,
)
from oracle_composition.experiments.reference_input_transforms import (
    CONDITION_IDS,
    transform_reference_input,
)
from oracle_composition.phase_b.training import (
    SMOKE_SEED,
    SMOKE_TRANSITIONS,
    PPORecipe,
    domain_separated_seed,
)

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
    sealed_lineage = {
        "artifacts": [
            {
                "byte_count": 1,
                "path": "artifacts/reference_corpus_v2/corpus_manifest_v2.json",
                "roles": ["reference_corpus"],
                "sha256": "a" * 64,
            }
        ],
        "lineage_id": "humanoid_phase_b_sealed_input_lineage/v1",
        "schema_version": 1,
    }
    sealed_sha = hashlib.sha256(canonical_json_bytes(sealed_lineage)).hexdigest()
    inputs = {
        "e1_receipt_sha256": "1" * 64,
        "evaluator_sha256": "2" * 64,
        "execution_manifest_sha256": "3" * 64,
        "library_sha256": "4" * 64,
        "oracle_canonical_sha256": "5" * 64,
        "oracle_file_sha256": "6" * 64,
        "reference_sha256": "a" * 64,
        "reward_compositor_sha256": "7" * 64,
        "reward_file_sha256": "8" * 64,
        "reward_formula_id": "fixture",
        "reward_formula_sha256": "9" * 64,
        "reward_schema_id": "fixture",
        "sealed_input_lineage_sha256": sealed_sha,
        "starting_expert_identity": {"fixture": True},
        "task_sha256": "b" * 64,
        "training_design_sha256": "c" * 64,
    }
    runtime_snapshot = {
        "authority_identities": {},
        "device": "cpu",
        "git": {"clean": True, "commit": "d" * 40},
        "platform": {"machine": "fixture", "python": "3.12", "system": "fixture"},
        "source_sha256": {
            path: "e" * 64
            for path in (
                "src/oracle_composition/envs/humanoid.py",
                "src/oracle_composition/phase_b/policy.py",
                "src/oracle_composition/phase_b/reference_runtime.py",
                "src/oracle_composition/phase_b/runtime.py",
                "src/oracle_composition/phase_b/training.py",
            )
        },
        "versions": {
            "gymnasium": "fixture",
            "mujoco": "fixture",
            "numpy": "fixture",
            "stable_baselines3": "fixture",
            "torch": "fixture",
        },
    }
    reservation = {"fixture": True}
    manifest_bytes = _write_json(
        root / "execution_manifest_v3.json",
        {
            "checkpoint_selection": "final_transition_only",
            "e003_execution_manifest_sha256": "3" * 64,
            "evidence_class": "interface_check",
            "execution_manifest_schema_id": "humanoid_phase_b_execution_manifest/v3",
            "ft1_run_manifest_sha256": "f" * 64,
            "inputs": inputs,
            "prior_scientific_receipt_sha256": "0" * 64,
            "reservation": reservation,
            "reservation_sha256": hashlib.sha256(canonical_json_bytes(reservation)).hexdigest(),
            "resource_control_policy": {
                "cpu_time": "os_rlimit_when_supported_otherwise_recorded_unsupported",
                "environment": "spawn_time_explicit_allowlist",
                "filesystem": "parent_observed_os_best_effort",
                "process_group_cleanup": ("os_session_group_best_effort_with_fail_closed_receipt"),
                "process_tree_rss": "parent_observed_os_best_effort",
            },
            "resource_limits": {"fixture": True},
            "runtime_source_snapshot": runtime_snapshot,
            "runtime_source_snapshot_sha256": hashlib.sha256(
                canonical_json_bytes(runtime_snapshot)
            ).hexdigest(),
            "schema_version": 3,
            "sealed_input_lineage": sealed_lineage,
            "sealed_input_lineage_sha256": sealed_sha,
            "seeds": [SMOKE_SEED],
            "smoke": True,
            "test_only": False,
            "transitions_per_seed": SMOKE_TRANSITIONS,
        },
    )
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    rsi_bytes = _write_json(seed / "rsi_ledger_v1.json", [{"global_episode_index": 0}])
    recipe = PPORecipe().to_dict()
    audit_rows = [
        {
            "audit_id": "tanh_corrected_rollout_likelihood_audit/v2",
            "audit_stage": "before_any_update_for_rollout",
            "distribution_snapshot_sha256": "1" * 64,
            "maximum_absolute_difference": 0.0,
            "observation_sample_sha256": "2" * 64,
            "old_log_prob_sample_sha256": "3" * 64,
            "passed": True,
            "pre_tanh_sample_sha256": "4" * 64,
            "rollout_index": index,
            "sample_count": 64,
            "sample_indices_sha256": "5" * 64,
            "tolerance": 1e-5,
        }
        for index in range(24)
    ]
    unfreeze = [
        {
            "actor_stage": ("reference_columns_only" if index < 8 else "full_actor"),
            "rollout_index": index,
            "rollout_likelihood_audit": audit_rows[index],
            "state_columns_changed": index >= 8,
        }
        for index in range(24)
    ]
    step_zero = {
        "action_sha256": "6" * 64,
        "bitwise_equal": True,
        "e1_receipt_sha256": "1" * 64,
        "fixture_count": 1,
        "reported_beside_checkpoint": True,
        "worker_path": "fixture-worker",
    }
    training_bytes = _write_json(
        seed / "training_facts_v1.json",
        {
            "device": "cpu",
            "evidence_class": "interface_check",
            "execution_manifest_sha256": manifest_sha,
            "likelihood_audit": {
                "all_passed": True,
                "audit_id": "tanh_corrected_rollout_likelihood_audit/v2",
                "audit_stage": "before_any_update_for_each_rollout",
                "receipt_sha256": hashlib.sha256(canonical_json_bytes(audit_rows)).hexdigest(),
                "rollout_audits": audit_rows,
                "rollout_count": 24,
            },
            "losses": [
                {
                    "approximate_kl": 0.0,
                    "clip_fraction": 0.0,
                    "entropy_loss": 0.0,
                    "explained_variance": 0.0,
                    "policy_loss": 0.0,
                    "rollout_index": index,
                    "total_loss": 0.0,
                    "value_loss": 0.0,
                }
                for index in range(24)
            ],
            "normalization": {"observation": False, "reward": False},
            "observed_transitions": SMOKE_TRANSITIONS,
            "optimizer_initialization": {},
            "optimizer_updates": 24 * recipe["n_epochs"] * (8_192 // recipe["batch_size"]),
            "planned_transitions": SMOKE_TRANSITIONS,
            "ppo_recipe": recipe,
            "ppo_recipe_id": "humanoid_phase_b_ppo_recipe/v1",
            "ppo_seed": SMOKE_SEED,
            "promotable": False,
            "reward_totals": {
                "ignored_stock_reward": 0.0,
                "r_task": 0.0,
                "r_track": 0.0,
                "r_train": 0.0,
            },
            "rng_substreams": {
                "action_sampling": domain_separated_seed(manifest_sha, SMOKE_SEED, "actions"),
                "environment_order": [
                    domain_separated_seed(manifest_sha, SMOKE_SEED, "vector-environment", index)
                    for index in range(4)
                ],
                "minibatches": domain_separated_seed(manifest_sha, SMOKE_SEED, "minibatches"),
                "numpy_global": SMOKE_SEED,
                "python_global": SMOKE_SEED,
                "scheduler": domain_separated_seed(
                    manifest_sha, SMOKE_SEED, "scheduler-construction"
                ),
                "torch_global": SMOKE_SEED,
            },
            "rollouts": 24,
            "rsi_ledger_sha256": hashlib.sha256(rsi_bytes).hexdigest(),
            "smoke": True,
            "step_zero_comparator": step_zero,
            "stream_counts": {
                "composition": SMOKE_TRANSITIONS // 2,
                "rehearsal": SMOKE_TRANSITIONS // 2,
            },
            "thread_counts": {"torch_interop": 1, "torch_intraop": 1},
            "time_limit_bootstrap_count": 0,
            "training_worker_id": "fixture-worker",
            "unfreeze_receipt_sha256": hashlib.sha256(canonical_json_bytes(unfreeze)).hexdigest(),
            "unfreeze_rollouts": unfreeze,
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
            "fixture_action_sha256": "6" * 64,
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
            "resource_controls": {
                "cpu_time": {
                    "enforcement": "unsupported",
                    "limit_seconds": 1_200,
                    "resource": "RLIMIT_CPU",
                },
                "environment": {
                    "allowlist_enforced": True,
                    "environment_sha256": "7" * 64,
                    "keys": [],
                    "runtime_added_keys_removed": [],
                    "unexpected_keys": [],
                },
                "executed_modules": {
                    "enforcement": "checkout_realpath_and_recorded_digest_verified",
                    "final_sha256": "8" * 64,
                    "start_sha256": "8" * 64,
                },
                "filesystem": {"enforcement": "parent_observed_os_best_effort"},
                "process_group_cleanup": {
                    "enforcement": "os_session_group_best_effort",
                    "succeeded": True,
                },
                "process_tree_rss": {"enforcement": "parent_observed_os_best_effort"},
            },
            "schema_version": 2,
            "smoke": True,
            "status": "succeeded",
            "success_receipt_id": "humanoid_phase_b_seed_success/v2",
            "test_only": False,
            "worker_cleanup": {"attempted": True, "error": None, "succeeded": True},
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


def _rebind_success_in_job(root: Path) -> None:
    success_path = root / f"seed_{SMOKE_SEED}/success_receipt_v2.json"
    job_path = root / "job_result_v1.json"
    job = json.loads(job_path.read_bytes())
    job["outcomes"][0]["receipt"] = _record(success_path)
    _write_json(job_path, job)


def test_smoke_lineage_rejects_missing_production_authority(tmp_path: Path) -> None:
    root = _smoke_fixture(tmp_path / "manifest")
    manifest_path = root / "execution_manifest_v3.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest.pop("sealed_input_lineage")
    _write_json(manifest_path, manifest)
    with pytest.raises(DevelopmentAblationError, match="manifest fields differ"):
        validate_smoke_export(root)

    root = _smoke_fixture(tmp_path / "success")
    success_path = root / f"seed_{SMOKE_SEED}/success_receipt_v2.json"
    success = json.loads(success_path.read_bytes())
    success.pop("resource_controls")
    _write_json(success_path, success)
    _rebind_success_in_job(root)
    with pytest.raises(DevelopmentAblationError, match="success receipt fields differ"):
        validate_smoke_export(root)

    root = _smoke_fixture(tmp_path / "persistence")
    seed = root / f"seed_{SMOKE_SEED}"
    persistence_path = seed / f"persistence_seed_{SMOKE_SEED}_v1.json"
    persistence = json.loads(persistence_path.read_bytes())
    persistence.pop("fixture_action_sha256")
    _write_json(persistence_path, persistence)
    success_path = seed / "success_receipt_v2.json"
    success = json.loads(success_path.read_bytes())
    success["artifacts"]["persistence"] = _record(persistence_path)
    _write_json(success_path, success)
    _rebind_success_in_job(root)
    with pytest.raises(DevelopmentAblationError, match="not production-shaped"):
        validate_smoke_export(root)


def test_corpus_root_must_match_smoke_sealed_bytes(tmp_path: Path) -> None:
    lineage = validate_smoke_export(_smoke_fixture(tmp_path / "run"))
    corpus = tmp_path / "corpus_checkout/artifacts/reference_corpus_v2"
    _write_json(corpus / "corpus_manifest_v2.json", {})

    with pytest.raises(DevelopmentAblationError, match="sealed inputs do not match"):
        validate_development_corpus(
            corpus,
            lineage=lineage,
            protocol=load_development_protocol(PROTOCOL_PATH),
        )


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
