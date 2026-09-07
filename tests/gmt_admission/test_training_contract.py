from __future__ import annotations

import hashlib

import pytest

from oracle_composition.adapters.gmt.training_contract import (
    BASE_LEARNING_RATE,
    EFFECTIVE_RATE_TRAINING_CONTRACT_ID,
    LOW_LEARNING_RATE,
    TRAINER_RATE_SCHEMA_ID,
    TRAINING_CONTRACT,
    TRAINING_REWARD_SCALE,
    CourseTrainerSpec,
    effective_training_contract,
    training_reward_metadata,
)
from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def test_legacy_raw_and_v1_scaled_identities_are_exact_regressions() -> None:
    v1 = CourseTrainerSpec(TRAINING_REWARD_SCALE)

    assert _sha(TRAINING_CONTRACT) == (
        "a143cbb6dd1b9f51f9420ace6fe064ca94c1b29b9558ce73ee412fa30dbd0854"
    )
    assert _sha(v1.to_dict()) == (
        "2b98769f8a5caa51d66a56cab5315652b33b5785634d5eae205914725805048f"
    )
    assert _sha(effective_training_contract(v1)) == (
        "56d8ed6a20e3fff99a5cd7faa6600edb63f172d051543111358d50f6bde27b44"
    )
    assert _sha(training_reward_metadata(v1)) == (
        "b771c0b5181aa598d89b0f5db7c72d56c6a0fbae5017f55f27fe37c0737d7b77"
    )
    assert v1.learning_rate == BASE_LEARNING_RATE


def test_v2_is_one_exact_low_rate_scaled_profile() -> None:
    spec = CourseTrainerSpec(TRAINING_REWARD_SCALE, profile_version=2)

    assert spec.to_dict() == {
        "schema_id": TRAINER_RATE_SCHEMA_ID,
        "schema_version": 2,
        "total_training_reward_scale": TRAINING_REWARD_SCALE,
        "learning_rate": LOW_LEARNING_RATE,
    }
    assert CourseTrainerSpec.from_dict(spec.to_dict()) == spec
    effective = effective_training_contract(spec)
    assert effective["schema_id"] == EFFECTIVE_RATE_TRAINING_CONTRACT_ID
    assert effective["schema_version"] == 3
    assert effective["base_ppo_contract"]["learning_rate"] == LOW_LEARNING_RATE
    assert TRAINING_CONTRACT["learning_rate"] == BASE_LEARNING_RATE


@pytest.mark.parametrize(
    "rate",
    [3e-4, 1e-5, 3e-5 + 1e-12, float("nan"), float("inf"), True, "0.00003"],
)
def test_v2_rejects_every_nonexact_learning_rate(rate: object) -> None:
    value = CourseTrainerSpec(TRAINING_REWARD_SCALE, profile_version=2).to_dict()
    value["learning_rate"] = rate

    with pytest.raises(ValueError, match="fixed finite 3e-5"):
        CourseTrainerSpec.from_dict(value)


def test_trainer_rejects_unadmitted_versions_and_fields() -> None:
    with pytest.raises(ValueError, match="version"):
        CourseTrainerSpec(TRAINING_REWARD_SCALE, profile_version=3)
    value = CourseTrainerSpec(TRAINING_REWARD_SCALE, profile_version=2).to_dict()
    value["schedule"] = "linear"
    with pytest.raises(ValueError, match="v2 profile"):
        CourseTrainerSpec.from_dict(value)
