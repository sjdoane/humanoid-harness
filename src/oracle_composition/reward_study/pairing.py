"""Exact-byte T2 reward-pairing adapter and interface receipt."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import array_sha256, canonical_json_bytes
from oracle_composition.phase_b import contracts as phase_b_contracts
from oracle_composition.phase_b import runtime as phase_b_runtime
from oracle_composition.phase_b import training as phase_b_training
from oracle_composition.phase_b.contracts import (
    NON_PAIRED_ID,
    T2_REWARD_PAIRING_DERIVATION_ID,
    T2_REWARD_PAIRING_ID,
    T2RewardPairing,
)
from oracle_composition.phase_b.policy import ACTION_WIDTH
from oracle_composition.phase_b.training import (
    PPORecipe,
    TrainingPlan,
    domain_separated_seed,
    paired_action_noise,
    paired_domain_separated_seed,
    paired_minibatch_permutation,
)

PAIRING_ADAPTER_ID = "t2_reward_study_pairing_adapter/v1"
PAIRING_DERIVATION_ID = T2_REWARD_PAIRING_DERIVATION_ID
PAIRING_RECEIPT_ID = "pairing_receipt_v1"
TRAINING_STREAM_DOMAINS = ("actions", "minibatches", "rsi_order", "rsi_class", "rsi_start")
T2_RSI_CLASSES = ("hold",)
T2_RSI_START_BOUNDARY_MODULUS = 489
_FAKE_ACTION_ROLLOUTS = 3
_FAKE_ACTION_STEPS = 7
_FAKE_MINIBATCH_UPDATES = 6
_FAKE_MINIBATCH_SAMPLE_COUNT = 64
_FAKE_RESET_EPISODES_PER_SLOT = 18


def _positive_seed(value: object, *, field: str) -> int:
    if type(value) is not int or not 0 < value <= 2_147_483_647:
        raise ValueError(f"{field} must be a positive signed-32-bit integer")
    return value


def _canonical_study_bytes(encoded: bytes) -> dict[str, object]:
    if type(encoded) is not bytes or not encoded or len(encoded) > 256 * 1024:
        raise ValueError("study manifest bytes are unavailable or oversized")
    try:
        value = json.loads(encoded)
    except (UnicodeError, ValueError) as exc:
        raise ValueError("study manifest bytes are not JSON") from exc
    if type(value) is not dict or canonical_json_bytes(value) != encoded:
        raise ValueError("study manifest bytes are not canonical JSON")
    return value


def pairing_from_study_manifest_bytes(encoded: bytes) -> T2RewardPairing:
    """Validate T2 semantics, then construct exact-byte pairing authority."""

    from .study_manifest import validate_t2_study_manifest

    value = _canonical_study_bytes(encoded)
    validate_t2_study_manifest(value)
    return T2RewardPairing.from_study_manifest_bytes(encoded)


def load_t2_reward_pairing(path: Path) -> T2RewardPairing:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("study manifest must be a regular non-linked file")
    return pairing_from_study_manifest_bytes(candidate.read_bytes())


def build_t2_training_plan(
    *,
    pairing: T2RewardPairing,
    arm_label: str,
    ppo_seed: int,
    transitions: int,
    evidence_class: str,
    promotable: bool,
    smoke: bool,
    steps_per_environment: int = phase_b_training.PRODUCTION_STEPS_PER_ENVIRONMENT,
    recipe: PPORecipe | None = None,
    test_only: bool = False,
) -> TrainingPlan:
    """Build a paired plan from verified bytes; no digest-only entry point exists."""

    if type(pairing) is not T2RewardPairing:
        raise ValueError("T2 paired plans require exact verified study and arm bytes")
    return TrainingPlan(
        seed=ppo_seed,
        transitions=transitions,
        manifest_sha256=pairing.arm_manifest_sha256(arm_label),
        evidence_class=evidence_class,
        promotable=promotable,
        smoke=smoke,
        pairing_declared=True,
        study_pairing=pairing,
        steps_per_environment=steps_per_environment,
        recipe=recipe or PPORecipe(),
        test_only=test_only,
    )


def derive_study_stream_seed(
    pairing: T2RewardPairing,
    ppo_seed: int,
    domain: str,
    index: int = 0,
) -> int:
    return paired_domain_separated_seed(pairing, ppo_seed, domain, index)


def derive_training_stream_seeds(
    *,
    ppo_seed: int,
    pairing_declared: bool,
    study_pairing: T2RewardPairing | None = None,
    manifest_sha256: str | None = None,
) -> dict[str, int]:
    """Select verified pairing or the byte-for-byte legacy manifest derivation."""

    _positive_seed(ppo_seed, field="ppo_seed")
    if type(pairing_declared) is not bool:
        raise ValueError("pairing_declared must be a boolean")
    if pairing_declared:
        if manifest_sha256 is not None:
            raise ValueError("a paired study refuses arm-specific manifest RNG derivation")
        if type(study_pairing) is not T2RewardPairing:
            raise ValueError("a paired study requires exact verified study and arm bytes")
        return {
            domain: derive_study_stream_seed(study_pairing, ppo_seed, domain)
            for domain in TRAINING_STREAM_DOMAINS
        }
    if study_pairing is not None:
        raise ValueError("a non-paired run refuses study pairing authority")
    if manifest_sha256 is None:
        raise ValueError("a non-paired run requires manifest_sha256")
    return {
        domain: domain_separated_seed(manifest_sha256, ppo_seed, domain)
        for domain in TRAINING_STREAM_DOMAINS
    }


def derive_evaluation_seed_identities(
    *,
    pairing: T2RewardPairing,
    ppo_seed: int,
    evaluation_seeds: Sequence[int],
) -> list[dict[str, object]]:
    """Bind declared indices while retaining each declared seed as the reset seed."""

    if type(evaluation_seeds) not in {list, tuple} or not evaluation_seeds:
        raise ValueError("evaluation seeds must be a nonempty ordered sequence")
    if len(set(evaluation_seeds)) != len(evaluation_seeds):
        raise ValueError("evaluation seeds must be unique")
    result = []
    for declared_index, evaluation_seed in enumerate(evaluation_seeds):
        checked = _positive_seed(evaluation_seed, field="evaluation_seed")
        identity = {
            "declared_index": declared_index,
            "environment_reset_seed": checked,
            "evaluation_seed": checked,
            "pairing_id": T2_REWARD_PAIRING_ID,
            "ppo_seed": _positive_seed(ppo_seed, field="ppo_seed"),
            "study_pairing_sha256": pairing.study_pairing_sha256,
        }
        result.append(
            {
                **identity,
                "identity_sha256": hashlib.sha256(canonical_json_bytes(identity)).hexdigest(),
            }
        )
    return result


derive_evaluation_stream_seeds = derive_evaluation_seed_identities


@dataclass(frozen=True, slots=True)
class T2RSIAssignment:
    ppo_seed: int
    environment_index: int
    global_episode_index: int
    cycle: int
    block: int
    reference_behavior: str
    schedule_class: str
    start_boundary: int

    def to_dict(self) -> dict[str, object]:
        return {
            "block": self.block,
            "cycle": self.cycle,
            "environment_index": self.environment_index,
            "global_episode_index": self.global_episode_index,
            "ppo_seed": self.ppo_seed,
            "reference_behavior": self.reference_behavior,
            "schedule_class": self.schedule_class,
            "start_boundary": self.start_boundary,
        }


def derive_t2_rsi_assignment(
    *,
    pairing: T2RewardPairing,
    ppo_seed: int,
    global_episode_index: int,
    environment_index: int,
) -> T2RSIAssignment:
    scheduler = phase_b_training.BalancedRSIScheduler(
        manifest_sha256=pairing.baseline_arm_manifest_sha256,
        ppo_seed=ppo_seed,
        study_pairing=pairing,
    )
    assignment = scheduler.assignment(
        global_episode_index=global_episode_index,
        environment_index=environment_index,
    )
    return T2RSIAssignment(
        ppo_seed=assignment.ppo_seed,
        environment_index=assignment.environment_index,
        global_episode_index=assignment.global_episode_index,
        cycle=assignment.cycle,
        block=assignment.block,
        reference_behavior=assignment.origin_behavior,
        schedule_class=assignment.schedule_class,
        start_boundary=assignment.start_boundary,
    )


def fake_runtime_stream_receipt(
    *,
    pairing: T2RewardPairing,
    ppo_seeds: Sequence[int],
    evaluation_seeds: Sequence[int],
) -> dict[str, object]:
    streams = []
    for ppo_seed in ppo_seeds:
        checked = _positive_seed(ppo_seed, field="ppo_seed")
        streams.append(
            {
                "evaluation": derive_evaluation_seed_identities(
                    pairing=pairing,
                    ppo_seed=checked,
                    evaluation_seeds=evaluation_seeds,
                ),
                "ppo_seed": checked,
                "rsi_assignment_prefix": [
                    derive_t2_rsi_assignment(
                        pairing=pairing,
                        ppo_seed=checked,
                        global_episode_index=index,
                        environment_index=2 + (index % 2),
                    ).to_dict()
                    for index in range(18)
                ],
                "training": derive_training_stream_seeds(
                    ppo_seed=checked,
                    pairing_declared=True,
                    study_pairing=pairing,
                ),
            }
        )
    value = {
        "derivation_id": PAIRING_DERIVATION_ID,
        "pairing_id": T2_REWARD_PAIRING_ID,
        "streams": streams,
        "study_pairing_sha256": pairing.study_pairing_sha256,
    }
    value["streams_sha256"] = hashlib.sha256(canonical_json_bytes(streams)).hexdigest()
    return value


def summarize_fake_runtime_stream_receipt(value: Mapping[str, object]) -> dict[str, object]:
    streams = value.get("streams")
    if type(streams) is not list or not streams:
        raise ValueError("fake-runtime stream receipt is empty")
    expected = hashlib.sha256(canonical_json_bytes(streams)).hexdigest()
    if value.get("streams_sha256") != expected:
        raise ValueError("fake-runtime stream receipt hash differs")
    return {
        "derivation_id": value.get("derivation_id"),
        "policy_seed_count": len(streams),
        "streams_sha256": expected,
        "study_pairing_sha256": value.get("study_pairing_sha256"),
    }


def _fake_policy_sha256(plan: TrainingPlan) -> str:
    policy = phase_b_runtime.fake_policy_factory(plan)
    arrays = {
        name: array_sha256(np.ascontiguousarray(tensor.detach().cpu().numpy()))
        for name, tensor in policy.state_dict().items()
    }
    return hashlib.sha256(canonical_json_bytes(arrays)).hexdigest()


def _runtime_source_paths() -> dict[str, Path]:
    return {
        "phase_b_contracts": Path(phase_b_contracts.__file__).resolve(),
        "phase_b_runtime": Path(phase_b_runtime.__file__).resolve(),
        "phase_b_training": Path(phase_b_training.__file__).resolve(),
        "reward_study_pairing": Path(__file__).resolve(),
    }


def _primitive_streams(
    *,
    pairing: T2RewardPairing,
    arm_label: str,
    ppo_seeds: Sequence[int],
    evaluation_seeds: Sequence[int],
) -> dict[str, object]:
    seed_slots = []
    per_seed = []
    for ppo_seed in ppo_seeds:
        plan = build_t2_training_plan(
            pairing=pairing,
            arm_label=arm_label,
            ppo_seed=ppo_seed,
            transitions=128,
            evidence_class="interface_check",
            promotable=False,
            smoke=False,
            steps_per_environment=4,
            recipe=PPORecipe(batch_size=16, n_epochs=1),
            test_only=True,
        )
        scheduler = phase_b_training.BalancedRSIScheduler.from_plan(plan)
        for environment_slot in range(4):
            action_digests = []
            for rollout_index in range(_FAKE_ACTION_ROLLOUTS):
                noise = paired_action_noise(
                    plan,
                    rollout_index=rollout_index,
                    steps_per_environment=_FAKE_ACTION_STEPS,
                )[:, environment_slot, :]
                action_digests.append(array_sha256(noise))
            if environment_slot in {0, 1}:
                reset_schedule = [
                    {
                        "block": scheduler.composition_block(
                            global_episode_index=index,
                            environment_index=environment_slot,
                        ),
                        "global_episode_index": index,
                    }
                    for index in range(_FAKE_RESET_EPISODES_PER_SLOT)
                ]
            else:
                reset_schedule = [
                    scheduler.assignment(
                        global_episode_index=index,
                        environment_index=environment_slot,
                    ).to_dict()
                    for index in range(_FAKE_RESET_EPISODES_PER_SLOT)
                ]
            payload = {
                "action_noise_by_rollout_sha256": action_digests,
                "reset_schedule_sha256": hashlib.sha256(
                    canonical_json_bytes(reset_schedule)
                ).hexdigest(),
            }
            seed_slots.append(
                {
                    "environment_slot": environment_slot,
                    "ppo_seed": ppo_seed,
                    "primitive_stream_sha256": hashlib.sha256(
                        canonical_json_bytes(payload)
                    ).hexdigest(),
                }
            )
        permutations = [
            array_sha256(
                paired_minibatch_permutation(
                    plan,
                    update_index=index,
                    sample_count=_FAKE_MINIBATCH_SAMPLE_COUNT,
                )
            )
            for index in range(_FAKE_MINIBATCH_UPDATES)
        ]
        evaluation = derive_evaluation_seed_identities(
            pairing=pairing,
            ppo_seed=ppo_seed,
            evaluation_seeds=evaluation_seeds,
        )
        per_seed.append(
            {
                "evaluation_seed_identities_sha256": hashlib.sha256(
                    canonical_json_bytes(evaluation)
                ).hexdigest(),
                "fake_policy_initialization_sha256": _fake_policy_sha256(plan),
                "minibatch_permutations_sha256": hashlib.sha256(
                    canonical_json_bytes(permutations)
                ).hexdigest(),
                "ppo_seed": ppo_seed,
            }
        )
    return {"per_seed": per_seed, "per_seed_and_slot": seed_slots}


def _receipt_pairing(path: Path) -> tuple[T2RewardPairing, dict[str, object], str]:
    source_bytes = Path(path).read_bytes()
    source = _canonical_study_bytes(source_bytes)
    from .study_manifest import STUDY_FINAL_READY_STATUS, validate_t2_study_manifest

    validate_t2_study_manifest(source)
    fake = copy.deepcopy(source)
    candidate_reward = phase_b_contracts.TargetSpeedRewardSpec(alpha=1.0, beta=0.0)
    fake["arms"][1]["reward"] = {
        "path": "fake-runtime/candidate_reward.json",
        "reward_id": "target_speed_triangular_affine_t2_adapter/v1",
        "sha256": hashlib.sha256(candidate_reward.canonical_bytes).hexdigest(),
    }
    fake["status"] = STUDY_FINAL_READY_STATUS
    fake_bytes = canonical_json_bytes(fake)
    validate_t2_study_manifest(fake)
    return (
        pairing_from_study_manifest_bytes(fake_bytes),
        fake,
        hashlib.sha256(source_bytes).hexdigest(),
    )


def generate_pairing_receipt_v1(study_manifest_path: Path) -> dict[str, object]:
    pairing, study, source_study_sha256 = _receipt_pairing(study_manifest_path)
    ppo_seeds = study["arms"][0]["training"]["ppo_seeds"]
    evaluation_seeds = study["arms"][0]["evaluation"]["evaluation_seeds"]
    baseline = _primitive_streams(
        pairing=pairing,
        arm_label="baseline",
        ppo_seeds=ppo_seeds,
        evaluation_seeds=evaluation_seeds,
    )
    candidate = _primitive_streams(
        pairing=pairing,
        arm_label="candidate",
        ppo_seeds=ppo_seeds,
        evaluation_seeds=evaluation_seeds,
    )
    if baseline != candidate:
        raise AssertionError("paired fake-runtime primitive streams differ")
    slot_pairs = [
        {
            "baseline_sha256": left["primitive_stream_sha256"],
            "candidate_sha256": right["primitive_stream_sha256"],
            "environment_slot": left["environment_slot"],
            "identical": left == right,
            "ppo_seed": left["ppo_seed"],
        }
        for left, right in zip(
            baseline["per_seed_and_slot"],
            candidate["per_seed_and_slot"],
            strict=True,
        )
    ]
    seed_pairs = [
        {
            "baseline_evaluation_sha256": left["evaluation_seed_identities_sha256"],
            "baseline_fake_policy_sha256": left["fake_policy_initialization_sha256"],
            "baseline_minibatches_sha256": left["minibatch_permutations_sha256"],
            "candidate_evaluation_sha256": right["evaluation_seed_identities_sha256"],
            "candidate_fake_policy_sha256": right["fake_policy_initialization_sha256"],
            "candidate_minibatches_sha256": right["minibatch_permutations_sha256"],
            "identical": left == right,
            "ppo_seed": left["ppo_seed"],
        }
        for left, right in zip(baseline["per_seed"], candidate["per_seed"], strict=True)
    ]
    root = Path(__file__).resolve().parents[3]
    runtime_sources = {
        name: {
            "byte_count": source.stat().st_size,
            "path": source.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
        for name, source in _runtime_source_paths().items()
    }
    value = {
        "arm_execution_manifest_sha256": {
            "baseline": pairing.baseline_arm_manifest_sha256,
            "candidate": pairing.candidate_arm_manifest_sha256,
        },
        "arm_execution_manifests_are_distinct": True,
        "claim_ceiling": "interface_only_no_training_smoke_reward_effect_or_humanoid_behavior_claim",
        "derivation_id": PAIRING_DERIVATION_ID,
        "evidence_class": "interface_check",
        "evaluation_seed_semantics": "declared_seed_is_actual_environment_reset_seed",
        "fake_runtime_coverage": {
            "action_noise_rollouts": _FAKE_ACTION_ROLLOUTS,
            "action_noise_steps_per_rollout": _FAKE_ACTION_STEPS,
            "action_width": ACTION_WIDTH,
            "minibatch_permutation_sample_count": _FAKE_MINIBATCH_SAMPLE_COUNT,
            "minibatch_update_indices": _FAKE_MINIBATCH_UPDATES,
            "reset_episode_indices_per_environment_slot": _FAKE_RESET_EPISODES_PER_SLOT,
        },
        "pairing_id": T2_REWARD_PAIRING_ID,
        "per_seed": seed_pairs,
        "per_seed_and_environment_slot": slot_pairs,
        "receipt_id": PAIRING_RECEIPT_ID,
        "reward_sha256": {
            "baseline": study["arms"][0]["reward"]["sha256"],
            "candidate": study["arms"][1]["reward"]["sha256"],
        },
        "runtime_sources": runtime_sources,
        "schedule_indexing": "global_episode_index_is_independent_within_each_environment_slot",
        "schema_version": 1,
        "source_study_manifest_sha256": source_study_sha256,
        "study_manifest_sha256": pairing.study_manifest_sha256,
        "study_pairing_sha256": pairing.study_pairing_sha256,
    }
    return validate_pairing_receipt(value)


def validate_pairing_receipt(value: Mapping[str, object]) -> dict[str, object]:
    expected_fields = {
        "arm_execution_manifest_sha256",
        "arm_execution_manifests_are_distinct",
        "claim_ceiling",
        "derivation_id",
        "evaluation_seed_semantics",
        "evidence_class",
        "fake_runtime_coverage",
        "pairing_id",
        "per_seed",
        "per_seed_and_environment_slot",
        "receipt_id",
        "reward_sha256",
        "runtime_sources",
        "schedule_indexing",
        "schema_version",
        "source_study_manifest_sha256",
        "study_manifest_sha256",
        "study_pairing_sha256",
    }
    if (
        type(value) is not dict
        or set(value) != expected_fields
        or value.get("receipt_id") != PAIRING_RECEIPT_ID
    ):
        raise ValueError("pairing receipt identity differs")
    if (
        value.get("schema_version") != 1
        or value.get("pairing_id") != T2_REWARD_PAIRING_ID
        or value.get("derivation_id") != PAIRING_DERIVATION_ID
        or value.get("evidence_class") != "interface_check"
        or value.get("claim_ceiling")
        != "interface_only_no_training_smoke_reward_effect_or_humanoid_behavior_claim"
        or value.get("arm_execution_manifests_are_distinct") is not True
        or value.get("schedule_indexing")
        != "global_episode_index_is_independent_within_each_environment_slot"
        or value.get("evaluation_seed_semantics")
        != "declared_seed_is_actual_environment_reset_seed"
    ):
        raise ValueError("pairing receipt contract differs")
    arm_hashes = value.get("arm_execution_manifest_sha256")
    rewards = value.get("reward_sha256")
    if (
        type(arm_hashes) is not dict
        or set(arm_hashes) != {"baseline", "candidate"}
        or arm_hashes["baseline"] == arm_hashes["candidate"]
        or type(rewards) is not dict
        or set(rewards) != {"baseline", "candidate"}
        or rewards["baseline"] == rewards["candidate"]
    ):
        raise ValueError("paired arm execution or reward identities differ from the contract")
    for digest in (*arm_hashes.values(), *rewards.values()):
        phase_b_contracts._sha(digest, field="pairing receipt SHA-256")
    for field_name in (
        "source_study_manifest_sha256",
        "study_manifest_sha256",
        "study_pairing_sha256",
    ):
        phase_b_contracts._sha(value[field_name], field=field_name)
    if value.get("fake_runtime_coverage") != {
        "action_noise_rollouts": _FAKE_ACTION_ROLLOUTS,
        "action_noise_steps_per_rollout": _FAKE_ACTION_STEPS,
        "action_width": ACTION_WIDTH,
        "minibatch_permutation_sample_count": _FAKE_MINIBATCH_SAMPLE_COUNT,
        "minibatch_update_indices": _FAKE_MINIBATCH_UPDATES,
        "reset_episode_indices_per_environment_slot": _FAKE_RESET_EPISODES_PER_SLOT,
    }:
        raise ValueError("pairing receipt fake-runtime coverage differs")
    sources = value.get("runtime_sources")
    expected_source_names = {
        "phase_b_contracts",
        "phase_b_runtime",
        "phase_b_training",
        "reward_study_pairing",
    }
    if type(sources) is not dict or set(sources) != expected_source_names:
        raise ValueError("pairing receipt runtime-source fields differ")
    root = Path(__file__).resolve().parents[3]
    expected_source_paths = _runtime_source_paths()
    for name, source in sources.items():
        if type(source) is not dict or set(source) != {"byte_count", "path", "sha256"}:
            raise ValueError("pairing receipt runtime-source binding is malformed")
        expected_path = expected_source_paths[name]
        expected_relative = expected_path.relative_to(root).as_posix()
        if source["path"] != expected_relative:
            raise ValueError("pairing receipt runtime-source path differs")
        path = root / expected_relative
        encoded = path.read_bytes()
        if (
            type(source["byte_count"]) is not int
            or source["byte_count"] != len(encoded)
            or source["sha256"] != hashlib.sha256(encoded).hexdigest()
        ):
            raise ValueError("pairing receipt runtime-source binding differs")
    slots = value.get("per_seed_and_environment_slot")
    seeds = value.get("per_seed")
    if type(slots) is not list or len(slots) != 20 or type(seeds) is not list or len(seeds) != 5:
        raise ValueError("pairing receipt does not cover the five-seed four-slot grid")
    if any(type(row) is not dict for row in [*slots, *seeds]):
        raise ValueError("pairing receipt stream rows are malformed")
    if any(
        set(row)
        != {
            "baseline_sha256",
            "candidate_sha256",
            "environment_slot",
            "identical",
            "ppo_seed",
        }
        or row.get("environment_slot") not in range(4)
        or row.get("identical") is not True
        or row.get("baseline_sha256") != row.get("candidate_sha256")
        for row in slots
    ) or any(
        set(row)
        != {
            "baseline_evaluation_sha256",
            "baseline_fake_policy_sha256",
            "baseline_minibatches_sha256",
            "candidate_evaluation_sha256",
            "candidate_fake_policy_sha256",
            "candidate_minibatches_sha256",
            "identical",
            "ppo_seed",
        }
        or row.get("identical") is not True
        or row.get("baseline_evaluation_sha256") != row.get("candidate_evaluation_sha256")
        or row.get("baseline_fake_policy_sha256") != row.get("candidate_fake_policy_sha256")
        or row.get("baseline_minibatches_sha256") != row.get("candidate_minibatches_sha256")
        for row in seeds
    ):
        raise ValueError("paired primitive stream digests differ")
    for row in slots:
        phase_b_contracts._sha(row["baseline_sha256"], field="baseline primitive stream")
        phase_b_contracts._sha(row["candidate_sha256"], field="candidate primitive stream")
    for row in seeds:
        for field in (
            "baseline_evaluation_sha256",
            "baseline_fake_policy_sha256",
            "baseline_minibatches_sha256",
            "candidate_evaluation_sha256",
            "candidate_fake_policy_sha256",
            "candidate_minibatches_sha256",
        ):
            phase_b_contracts._sha(row[field], field=field)
    expected_seeds = set(phase_b_training.COHORT_SEEDS)
    if {row["ppo_seed"] for row in seeds} != expected_seeds or {
        (row["ppo_seed"], row["environment_slot"]) for row in slots
    } != {(seed, slot) for seed in expected_seeds for slot in range(4)}:
        raise ValueError("pairing receipt seed-slot coverage differs")
    canonical_json_bytes(dict(value))
    return dict(value)


def write_pairing_receipt_v1(*, study_manifest_path: Path, output_path: Path) -> str:
    value = generate_pairing_receipt_v1(study_manifest_path)
    encoded = canonical_json_bytes(value)
    Path(output_path).write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the T2 fake-runtime pairing receipt")
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        write_pairing_receipt_v1(study_manifest_path=args.study_manifest, output_path=args.output)
    )
    return 0


__all__ = [
    "NON_PAIRED_ID",
    "PAIRING_ADAPTER_ID",
    "PAIRING_DERIVATION_ID",
    "PAIRING_RECEIPT_ID",
    "T2_RSI_CLASSES",
    "T2_RSI_START_BOUNDARY_MODULUS",
    "TRAINING_STREAM_DOMAINS",
    "T2RSIAssignment",
    "build_t2_training_plan",
    "derive_evaluation_seed_identities",
    "derive_evaluation_stream_seeds",
    "derive_study_stream_seed",
    "derive_t2_rsi_assignment",
    "derive_training_stream_seeds",
    "fake_runtime_stream_receipt",
    "generate_pairing_receipt_v1",
    "load_t2_reward_pairing",
    "pairing_from_study_manifest_bytes",
    "summarize_fake_runtime_stream_receipt",
    "validate_pairing_receipt",
    "write_pairing_receipt_v1",
]


if __name__ == "__main__":  # pragma: no cover - exercised through the generator function
    raise SystemExit(main())
