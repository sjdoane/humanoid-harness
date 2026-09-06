"""Arm-invariant RNG derivation seam for the frozen T2 study."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.phase_b.contracts import T2_TRAINING_BLOCKS

PAIRING_DERIVATION_ID = "t2_reward_study_rng_substream/v1"
PAIRING_ADAPTER_ID = "t2_reward_study_pairing_adapter/v1"
TRAINING_STREAM_DOMAINS = (
    "actions",
    "minibatches",
    "rsi_order",
    "rsi_class",
    "rsi_start",
)
T2_RSI_CLASSES = ("hold",)
T2_RSI_START_BOUNDARY_MODULUS = 489


def _sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _positive_seed(value: object, *, field: str) -> int:
    if type(value) is not int or not 0 < value <= 2_147_483_647:
        raise ValueError(f"{field} must be a positive signed-32-bit integer")
    return value


def derive_study_stream_seed(
    study_pairing_sha256: str,
    ppo_seed: int,
    domain: str,
    index: int = 0,
) -> int:
    """Derive an unsigned 64-bit seed from only arm-common study identity."""

    pairing = _sha256(study_pairing_sha256, field="study_pairing_sha256")
    seed = _positive_seed(ppo_seed, field="ppo_seed")
    if type(domain) is not str or not domain or len(domain) > 128:
        raise ValueError("pairing stream domain must be nonempty bounded text")
    if type(index) is not int or index < 0:
        raise ValueError("pairing stream index must be non-negative")
    payload = canonical_json_bytes(
        {
            "domain": domain,
            "index": index,
            "ppo_seed": seed,
            "schema": PAIRING_DERIVATION_ID,
            "study_pairing_sha256": pairing,
        }
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def derive_training_stream_seeds(
    *,
    ppo_seed: int,
    pairing_declared: bool,
    study_pairing_sha256: str | None = None,
    manifest_sha256: str | None = None,
) -> dict[str, int]:
    """Select paired-study or legacy single-arm derivation, never both."""

    _positive_seed(ppo_seed, field="ppo_seed")
    if type(pairing_declared) is not bool:
        raise ValueError("pairing_declared must be a boolean")
    if pairing_declared:
        if manifest_sha256 is not None:
            raise ValueError("a paired study refuses arm-specific manifest RNG derivation")
        if study_pairing_sha256 is None:
            raise ValueError("a paired study requires study_pairing_sha256")
        return {
            domain: derive_study_stream_seed(study_pairing_sha256, ppo_seed, domain)
            for domain in TRAINING_STREAM_DOMAINS
        }
    if study_pairing_sha256 is not None:
        raise ValueError("a single-arm run refuses an undeclared study pairing key")
    if manifest_sha256 is None:
        raise ValueError("a single-arm run requires manifest_sha256")
    manifest = _sha256(manifest_sha256, field="manifest_sha256")
    from oracle_composition.phase_b.training import domain_separated_seed

    return {
        domain: domain_separated_seed(manifest, ppo_seed, domain)
        for domain in TRAINING_STREAM_DOMAINS
    }


def derive_evaluation_stream_seeds(
    *,
    study_pairing_sha256: str,
    ppo_seed: int,
    evaluation_seeds: Sequence[int],
) -> list[dict[str, int]]:
    """Bind every evaluator reset stream to the common pairing key."""

    if type(evaluation_seeds) not in {list, tuple} or not evaluation_seeds:
        raise ValueError("evaluation seeds must be a nonempty ordered sequence")
    if len(set(evaluation_seeds)) != len(evaluation_seeds):
        raise ValueError("evaluation seeds must be unique")
    result = []
    for index, evaluation_seed in enumerate(evaluation_seeds):
        checked = _positive_seed(evaluation_seed, field="evaluation_seed")
        result.append(
            {
                "environment_reset_seed": derive_study_stream_seed(
                    study_pairing_sha256,
                    ppo_seed,
                    "evaluation_environment_reset",
                    index=checked,
                ),
                "evaluation_seed": checked,
                "stream_order_seed": derive_study_stream_seed(
                    study_pairing_sha256,
                    ppo_seed,
                    "evaluation_stream_order",
                    index=index,
                ),
            }
        )
    return result


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
    order_seed: int
    class_seed: int
    start_seed: int

    def to_dict(self) -> dict[str, object]:
        return {
            "block": self.block,
            "class_seed": self.class_seed,
            "cycle": self.cycle,
            "environment_index": self.environment_index,
            "global_episode_index": self.global_episode_index,
            "order_seed": self.order_seed,
            "ppo_seed": self.ppo_seed,
            "reference_behavior": self.reference_behavior,
            "schedule_class": self.schedule_class,
            "start_boundary": self.start_boundary,
            "start_seed": self.start_seed,
        }


def derive_t2_rsi_assignment(
    *,
    study_pairing_sha256: str,
    ppo_seed: int,
    global_episode_index: int,
    environment_index: int,
) -> T2RSIAssignment:
    """Derive expert-only RSI block order, class, and start from the common key."""

    if type(global_episode_index) is not int or global_episode_index < 0:
        raise ValueError("global_episode_index must be non-negative")
    if environment_index not in {2, 3}:
        raise ValueError("T2 RSI belongs only to rehearsal environments 2 and 3")
    cycle, position = divmod(global_episode_index, len(T2_TRAINING_BLOCKS))
    order_seed = derive_study_stream_seed(
        study_pairing_sha256,
        ppo_seed,
        "rsi_order",
        index=cycle,
    )

    def block_key(block: int) -> bytes:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "block": block,
                    "cycle": cycle,
                    "order_seed": order_seed,
                    "purpose": "t2_expert_rsi_block_order/v1",
                }
            )
        ).digest()

    ranked = tuple(sorted(T2_TRAINING_BLOCKS, key=block_key))
    block = ranked[position]
    class_seed = derive_study_stream_seed(
        study_pairing_sha256,
        ppo_seed,
        "rsi_class",
        index=global_episode_index,
    )
    schedule_class = T2_RSI_CLASSES[class_seed % len(T2_RSI_CLASSES)]
    start_seed = derive_study_stream_seed(
        study_pairing_sha256,
        ppo_seed,
        "rsi_start",
        index=global_episode_index,
    )
    return T2RSIAssignment(
        ppo_seed=ppo_seed,
        environment_index=environment_index,
        global_episode_index=global_episode_index,
        cycle=cycle,
        block=block,
        reference_behavior="expert",
        schedule_class=schedule_class,
        start_boundary=start_seed % T2_RSI_START_BOUNDARY_MODULUS,
        order_seed=order_seed,
        class_seed=class_seed,
        start_seed=start_seed,
    )


def fake_runtime_stream_receipt(
    *,
    study_pairing_sha256: str,
    ppo_seeds: Sequence[int],
    evaluation_seeds: Sequence[int],
) -> dict[str, object]:
    """Return deterministic fake-runtime streams for an interface receipt."""

    if type(ppo_seeds) not in {list, tuple} or not ppo_seeds:
        raise ValueError("PPO seeds must be a nonempty ordered sequence")
    if len(set(ppo_seeds)) != len(ppo_seeds):
        raise ValueError("PPO seeds must be unique")
    streams = []
    for ppo_seed in ppo_seeds:
        checked = _positive_seed(ppo_seed, field="ppo_seed")
        streams.append(
            {
                "evaluation": derive_evaluation_stream_seeds(
                    study_pairing_sha256=study_pairing_sha256,
                    ppo_seed=checked,
                    evaluation_seeds=evaluation_seeds,
                ),
                "ppo_seed": checked,
                "rsi_assignment_prefix": [
                    derive_t2_rsi_assignment(
                        study_pairing_sha256=study_pairing_sha256,
                        ppo_seed=checked,
                        global_episode_index=index,
                        environment_index=2 + (index % 2),
                    ).to_dict()
                    for index in range(18)
                ],
                "training": derive_training_stream_seeds(
                    ppo_seed=checked,
                    pairing_declared=True,
                    study_pairing_sha256=study_pairing_sha256,
                ),
            }
        )
    value = {
        "adapter_id": PAIRING_ADAPTER_ID,
        "derivation_id": PAIRING_DERIVATION_ID,
        "streams": streams,
        "study_pairing_sha256": _sha256(
            study_pairing_sha256,
            field="study_pairing_sha256",
        ),
    }
    value["streams_sha256"] = hashlib.sha256(canonical_json_bytes(streams)).hexdigest()
    return value


def summarize_fake_runtime_stream_receipt(value: Mapping[str, object]) -> dict[str, object]:
    """Retain complete-stream identity without duplicating the full receipt artifact."""

    expected = {
        "adapter_id",
        "derivation_id",
        "streams",
        "streams_sha256",
        "study_pairing_sha256",
    }
    if type(value) is not dict or set(value) != expected:
        raise ValueError("fake-runtime stream receipt fields differ")
    if value["adapter_id"] != PAIRING_ADAPTER_ID or value["derivation_id"] != PAIRING_DERIVATION_ID:
        raise ValueError("fake-runtime stream receipt identity differs")
    _sha256(value["study_pairing_sha256"], field="study_pairing_sha256")
    streams = value["streams"]
    if type(streams) is not list or not streams:
        raise ValueError("fake-runtime stream receipt is empty")
    evaluation_counts = {len(item.get("evaluation", ())) for item in streams if type(item) is dict}
    if len(evaluation_counts) != 1 or len(streams) != len(
        {item.get("ppo_seed") for item in streams}
    ):
        raise ValueError("fake-runtime stream receipt seed coverage differs")
    expected_streams_sha256 = hashlib.sha256(canonical_json_bytes(streams)).hexdigest()
    if value["streams_sha256"] != expected_streams_sha256:
        raise ValueError("fake-runtime stream receipt hash differs")
    return {
        "adapter_id": value["adapter_id"],
        "derivation_id": value["derivation_id"],
        "evaluation_seed_count_per_policy": next(iter(evaluation_counts)),
        "policy_seed_count": len(streams),
        "streams_sha256": value["streams_sha256"],
        "study_pairing_sha256": value["study_pairing_sha256"],
    }


def validate_pairing_receipt(value: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "adapter_id",
        "adapter_source_sha256",
        "arms_differ_only_in_reward",
        "baseline_stream_receipt",
        "candidate_stream_receipt",
        "changed_common_field",
        "changed_common_field_changes_key",
        "changed_common_pairing_sha256",
        "changed_common_stream_receipt",
        "derivation_id",
        "evidence_class",
        "integrated_runtime_receipt",
        "schema_version",
        "study_pairing_sha256",
    }
    if type(value) is not dict or set(value) != expected:
        raise ValueError("pairing receipt fields differ")
    if (
        value["schema_version"] != 1
        or value["adapter_id"] != PAIRING_ADAPTER_ID
        or value["derivation_id"] != PAIRING_DERIVATION_ID
        or value["evidence_class"] != "interface_check"
        or value["arms_differ_only_in_reward"] is not True
        or value["changed_common_field"] != "task.target_com_forward_speed_m_s:3.0_to_3.1"
        or value["changed_common_field_changes_key"] is not True
        or value["integrated_runtime_receipt"] != "TBD_pending_astra_acceptance"
    ):
        raise ValueError("pairing receipt identity or evidence boundary differs")
    _sha256(value["adapter_source_sha256"], field="adapter_source_sha256")
    pairing = _sha256(value["study_pairing_sha256"], field="study_pairing_sha256")
    changed = _sha256(
        value["changed_common_pairing_sha256"],
        field="changed_common_pairing_sha256",
    )
    if pairing == changed:
        raise ValueError("changed common field did not change the pairing key")
    baseline = value["baseline_stream_receipt"]
    candidate = value["candidate_stream_receipt"]
    changed_receipt = value["changed_common_stream_receipt"]
    if not all(type(item) is dict for item in (baseline, candidate, changed_receipt)):
        raise ValueError("pairing stream receipts must be objects")
    nested_fields = {
        "adapter_id",
        "derivation_id",
        "evaluation_seed_count_per_policy",
        "policy_seed_count",
        "streams_sha256",
        "study_pairing_sha256",
    }
    if any(set(item) != nested_fields for item in (baseline, candidate, changed_receipt)):
        raise ValueError("pairing stream receipt summary fields differ")
    for item in (baseline, candidate, changed_receipt):
        if (
            item["adapter_id"] != PAIRING_ADAPTER_ID
            or item["derivation_id"] != PAIRING_DERIVATION_ID
            or type(item["policy_seed_count"]) is not int
            or item["policy_seed_count"] <= 0
            or type(item["evaluation_seed_count_per_policy"]) is not int
            or item["evaluation_seed_count_per_policy"] <= 0
        ):
            raise ValueError("pairing stream receipt summary identity differs")
        if item["policy_seed_count"] != 5 or item["evaluation_seed_count_per_policy"] != 20:
            raise ValueError("pairing receipt does not cover the frozen 5 by 20 seed grid")
        _sha256(item["streams_sha256"], field="streams_sha256")
    if baseline != candidate or baseline.get("study_pairing_sha256") != pairing:
        raise ValueError("reward-only arms do not have identical stream receipts")
    if changed_receipt.get("study_pairing_sha256") != changed or changed_receipt.get(
        "streams_sha256"
    ) == baseline.get("streams_sha256"):
        raise ValueError("changed common field did not change derived streams")
    canonical_json_bytes(dict(value))
    return dict(value)


__all__ = [
    "PAIRING_ADAPTER_ID",
    "PAIRING_DERIVATION_ID",
    "T2_RSI_CLASSES",
    "T2_RSI_START_BOUNDARY_MODULUS",
    "TRAINING_STREAM_DOMAINS",
    "T2RSIAssignment",
    "derive_evaluation_stream_seeds",
    "derive_study_stream_seed",
    "derive_t2_rsi_assignment",
    "derive_training_stream_seeds",
    "fake_runtime_stream_receipt",
    "summarize_fake_runtime_stream_receipt",
    "validate_pairing_receipt",
]
