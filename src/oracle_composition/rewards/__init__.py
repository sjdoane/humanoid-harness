"""Executable task-reward contracts for controlled Family-B studies."""

from .contract import (
    ACTUATOR_QVEL_INDICES_BY_ACTION_V1,
    CANDIDATE_READ_SET_V1,
    CONTROL_PERIOD_SECONDS,
    REWARD_SCHEMA_V1,
    CandidateTaskInputsV1,
    RewardArtifactIdentityV1,
    RewardContractError,
    RewardResultV1,
    TaskTermV1,
    TrustedRewardStepV1,
    canonical_json_bytes,
    reward_schema_sha256,
)

__all__ = [
    "ACTUATOR_QVEL_INDICES_BY_ACTION_V1",
    "CANDIDATE_READ_SET_V1",
    "CONTROL_PERIOD_SECONDS",
    "REWARD_SCHEMA_V1",
    "CandidateTaskInputsV1",
    "RewardArtifactIdentityV1",
    "RewardContractError",
    "RewardResultV1",
    "TaskTermV1",
    "TrustedRewardStepV1",
    "canonical_json_bytes",
    "reward_schema_sha256",
]
