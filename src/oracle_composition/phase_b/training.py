"""PPO mechanics and deterministic stream scheduling for Phase B."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import (
    array_sha256,
    canonical_json_bytes,
)
from oracle_composition.experiments.fixed_reference import ExperimentContractError

from .contracts import (
    NON_PAIRED_ID,
    T2_REWARD_PAIRING_DERIVATION_ID,
    T2_REWARD_PAIRING_ID,
    T2RewardPairing,
)
from .policy import (
    ACTION_WIDTH,
    OBSERVATION_WIDTH,
    POLICY_INPUT_WIDTH,
    REFERENCE_HORIZON,
    REFERENCE_WIDTH,
    FullAuthorityPolicy,
    compose_policy_input,
    numpy_log_likelihood,
    verify_optimizer_authority,
)

try:
    import torch
except ImportError as exc:  # pragma: no cover - guarded by the train extra
    raise ExperimentContractError("Torch is required for Phase B training") from exc

TRAINING_WORKER_ID = "humanoid_phase_b_ppo_worker/v2"
PPO_RECIPE_ID = "humanoid_phase_b_ppo_recipe/v1"
RSI_SCHEDULER_ID = "sha256_balanced_predecessor_rsi/v1"
UNFREEZE_SCHEDULE_ID = "reference_columns_and_value_first_8_rollouts/v1"
LIKELIHOOD_AUDIT_ID = "tanh_corrected_rollout_likelihood_audit/v2"

COHORT_SEEDS = (121001, 121101, 121201, 121301, 121401)
SMOKE_SEED = 121901
TRAINING_BLOCKS = (120001, 120002, 120003, 120005, 120007, 120008, 120009, 120011, 120012)
BEHAVIORS = ("expert", "medium", "simple")
SCHEDULE_CLASSES = ("hold", "one_way", "round_trip")
STREAM_BY_ENVIRONMENT = ("composition", "composition", "rehearsal", "rehearsal")
FULL_TRANSITIONS_PER_SEED = 1_048_576
SMOKE_TRANSITIONS = 196_608
PRODUCTION_STEPS_PER_ENVIRONMENT = 2_048
PRODUCTION_BATCH_SIZE = 512
REFERENCE_ONLY_ROLLOUTS = 8
THROUGHPUT_WARMUP_TRANSITIONS = 65_536
THROUGHPUT_WINDOW_TRANSITIONS = 65_536
LIKELIHOOD_TOLERANCE = 1e-5


class TrainingFailureStatus(StrEnum):
    NON_FINITE = "non_finite"
    LIKELIHOOD_FAILURE = "likelihood_failure"
    COUNTER_DRIFT = "counter_drift"
    ACTION_BOUND_VIOLATION = "action_bound_violation"
    PHASE_SELECTION_FAILURE = "phase_selection_failure"


class PhaseBTrainingError(RuntimeError):
    """A fail-closed worker error with a stable supervisor status."""

    def __init__(
        self,
        status: TrainingFailureStatus,
        message: str,
        *,
        cleanup_error: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.cleanup_error = cleanup_error


@dataclass(frozen=True, slots=True)
class PPORecipe:
    batch_size: int = PRODUCTION_BATCH_SIZE
    n_epochs: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: None = None
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: None = None

    def __post_init__(self) -> None:
        if type(self.batch_size) is not int or self.batch_size <= 1:
            raise ValueError("PPO batch_size must be an integer greater than one")
        if type(self.n_epochs) is not int or self.n_epochs <= 0:
            raise ValueError("PPO n_epochs must be positive")
        if (
            type(self.learning_rate) is not float
            or self.learning_rate <= 0.0
            or type(self.gamma) is not float
            or not 0.0 < self.gamma <= 1.0
            or type(self.gae_lambda) is not float
            or not 0.0 < self.gae_lambda <= 1.0
            or type(self.clip_range) is not float
            or not 0.0 < self.clip_range < 1.0
            or type(self.ent_coef) is not float
            or self.ent_coef < 0.0
            or type(self.vf_coef) is not float
            or self.vf_coef < 0.0
            or type(self.max_grad_norm) is not float
            or self.max_grad_norm <= 0.0
            or self.clip_range_vf is not None
            or self.target_kl is not None
            or type(self.normalize_advantage) is not bool
        ):
            raise ValueError("PPO recipe value or type is invalid")

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_size": self.batch_size,
            "clip_range": self.clip_range,
            "clip_range_vf": self.clip_range_vf,
            "ent_coef": self.ent_coef,
            "gae_lambda": self.gae_lambda,
            "gamma": self.gamma,
            "learning_rate": self.learning_rate,
            "max_grad_norm": self.max_grad_norm,
            "n_epochs": self.n_epochs,
            "normalize_advantage": self.normalize_advantage,
            "target_kl": self.target_kl,
            "vf_coef": self.vf_coef,
        }


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    seed: int
    transitions: int
    manifest_sha256: str
    evidence_class: str
    promotable: bool
    smoke: bool
    pairing_declared: bool = False
    study_pairing: T2RewardPairing | None = None
    n_envs: int = 4
    steps_per_environment: int = PRODUCTION_STEPS_PER_ENVIRONMENT
    recipe: PPORecipe = PPORecipe()
    test_only: bool = False

    def __post_init__(self) -> None:
        if type(self.seed) is not int or self.seed <= 0:
            raise ValueError("PPO seed must be a positive integer")
        if (
            type(self.manifest_sha256) is not str
            or len(self.manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.manifest_sha256)
        ):
            raise ValueError("training manifest SHA-256 is invalid")
        if type(self.pairing_declared) is not bool:
            raise ValueError("pairing_declared must be a boolean")
        if self.pairing_declared is not (self.study_pairing is not None):
            raise ValueError("declared pairing requires exact verified study and arm bytes")
        if self.study_pairing is not None:
            if type(self.study_pairing) is not T2RewardPairing:
                raise ValueError("study pairing authority has the wrong type")
            if self.manifest_sha256 not in {
                self.study_pairing.baseline_arm_manifest_sha256,
                self.study_pairing.candidate_arm_manifest_sha256,
            }:
                raise ValueError("training manifest is not either verified paired arm identity")
            if "TBD" in {
                self.study_pairing.baseline_reward_sha256,
                self.study_pairing.candidate_reward_sha256,
            }:
                raise ValueError("paired training requires both exact reward SHA-256 identities")
        if self.n_envs != 4 or type(self.steps_per_environment) is not int:
            raise ValueError("Phase B requires exactly four environments")
        if self.steps_per_environment <= 0:
            raise ValueError("steps_per_environment must be positive")
        transitions_per_rollout = self.n_envs * self.steps_per_environment
        if type(self.transitions) is not int or self.transitions <= 0:
            raise ValueError("transitions must be a positive integer")
        if self.transitions % transitions_per_rollout:
            raise ValueError(
                "transitions must contain an exact number of four-environment rollouts"
            )
        if self.smoke:
            if not self.test_only and (
                self.seed != SMOKE_SEED or self.transitions != SMOKE_TRANSITIONS
            ):
                raise ValueError("smoke mode is bound to seed 121901 and 196,608 transitions")
            if self.evidence_class != "interface_check" or self.promotable:
                raise ValueError("smoke mode must be interface_check and non-promotable")
        elif not self.test_only:
            if self.seed not in COHORT_SEEDS or self.transitions != FULL_TRANSITIONS_PER_SEED:
                raise ValueError("cohort plans require a declared seed and 1,048,576 transitions")
            if self.evidence_class != "exploratory_fine_tuning_cycle" or not self.promotable:
                raise ValueError("cohort training label or promotion flag differs")
        if not self.test_only and self.steps_per_environment != PRODUCTION_STEPS_PER_ENVIRONMENT:
            raise ValueError("production Phase B rollouts require 2,048 steps per environment")
        if not self.test_only and self.recipe != PPORecipe():
            raise ValueError("production PPO recipe differs from the frozen recipe")
        if self.recipe.batch_size <= 0 or self.transitions_per_rollout % self.recipe.batch_size:
            raise ValueError("PPO batch size must divide a rollout exactly")

    @property
    def transitions_per_rollout(self) -> int:
        return self.n_envs * self.steps_per_environment

    @property
    def rollout_count(self) -> int:
        return self.transitions // self.transitions_per_rollout

    @property
    def pairing_id(self) -> str:
        return T2_REWARD_PAIRING_ID if self.pairing_declared else NON_PAIRED_ID

    @property
    def randomization_sha256(self) -> str:
        if self.study_pairing is None:
            return self.manifest_sha256
        return self.study_pairing.study_pairing_sha256

    def to_dict(self) -> dict[str, object]:
        return {
            "device": "cpu",
            "evidence_class": self.evidence_class,
            "manifest_sha256": self.manifest_sha256,
            "n_envs": self.n_envs,
            "normalization": {"observation": False, "reward": False},
            "ppo_recipe": self.recipe.to_dict(),
            "promotable": self.promotable,
            "rollout_count": self.rollout_count,
            "seed": self.seed,
            "smoke": self.smoke,
            "steps_per_environment": self.steps_per_environment,
            "stream_by_environment": list(STREAM_BY_ENVIRONMENT),
            "test_only": self.test_only,
            "transitions": self.transitions,
            "transitions_per_rollout": self.transitions_per_rollout,
        }


def domain_separated_seed(
    manifest_sha256: str,
    ppo_seed: int,
    domain: str,
    index: int = 0,
) -> int:
    """Derive a stable unsigned 64-bit seed without mutable global RNG state."""

    if (
        type(manifest_sha256) is not str
        or len(manifest_sha256) != 64
        or type(ppo_seed) is not int
        or ppo_seed <= 0
        or type(domain) is not str
        or not domain
        or type(index) is not int
        or index < 0
    ):
        raise ValueError("domain-separated seed inputs are invalid")
    payload = canonical_json_bytes(
        {
            "domain": domain,
            "index": index,
            "manifest_sha256": manifest_sha256,
            "ppo_seed": ppo_seed,
            "schema": "phase_b_rng_substream/v1",
        }
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def paired_domain_separated_seed(
    pairing: T2RewardPairing,
    ppo_seed: int,
    domain: str,
    index: int = 0,
) -> int:
    """Derive one T2 stream only from an exact-byte-verified pairing authority."""

    if type(pairing) is not T2RewardPairing:
        raise ValueError("paired seed derivation requires exact verified study and arm bytes")
    if (
        type(ppo_seed) is not int
        or ppo_seed <= 0
        or type(domain) is not str
        or not domain
        or type(index) is not int
        or index < 0
    ):
        raise ValueError("paired domain-separated seed inputs are invalid")
    payload = canonical_json_bytes(
        {
            "domain": domain,
            "index": index,
            "ppo_seed": ppo_seed,
            "schema": T2_REWARD_PAIRING_DERIVATION_ID,
            "study_pairing_sha256": pairing.study_pairing_sha256,
        }
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def plan_domain_separated_seed(
    plan: TrainingPlan,
    domain: str,
    index: int = 0,
) -> int:
    """Keep legacy manifest derivation exact and select pairing only when declared."""

    if type(plan) is not TrainingPlan:
        raise ValueError("seed derivation requires an exact TrainingPlan")
    if plan.study_pairing is None:
        return domain_separated_seed(plan.manifest_sha256, plan.seed, domain, index)
    return paired_domain_separated_seed(plan.study_pairing, plan.seed, domain, index)


def paired_action_noise(
    plan: TrainingPlan,
    *,
    rollout_index: int,
    steps_per_environment: int,
) -> np.ndarray:
    """Materialize paired primitive action noise by rollout and environment slot."""

    if not plan.pairing_declared:
        raise ValueError("paired action noise requires a declared pairing")
    if type(rollout_index) is not int or rollout_index < 0:
        raise ValueError("rollout_index must be non-negative")
    if type(steps_per_environment) is not int or steps_per_environment <= 0:
        raise ValueError("steps_per_environment must be positive")
    columns = []
    for environment_index in range(plan.n_envs):
        seed = plan_domain_separated_seed(
            plan,
            f"actions:environment_slot:{environment_index}",
            rollout_index,
        )
        generator = np.random.Generator(np.random.PCG64(seed))
        columns.append(
            np.ascontiguousarray(
                generator.standard_normal((steps_per_environment, ACTION_WIDTH)).astype("<f4"),
                dtype="<f4",
            )
        )
    return np.ascontiguousarray(np.stack(columns, axis=1), dtype="<f4")


def paired_minibatch_permutation(
    plan: TrainingPlan,
    *,
    update_index: int,
    sample_count: int,
) -> np.ndarray:
    """Derive one paired minibatch ordering independently by update index."""

    if not plan.pairing_declared:
        raise ValueError("paired minibatch order requires a declared pairing")
    if type(update_index) is not int or update_index < 0:
        raise ValueError("update_index must be non-negative")
    if type(sample_count) is not int or sample_count <= 1:
        raise ValueError("sample_count must exceed one")
    seed = plan_domain_separated_seed(plan, "minibatches", update_index)
    return np.ascontiguousarray(
        np.random.Generator(np.random.PCG64(seed)).permutation(sample_count),
        dtype="<i8",
    )


@dataclass(frozen=True, slots=True)
class RSIAssignment:
    ppo_seed: int
    environment_index: int
    global_episode_index: int
    cycle: int
    cell_index: int
    block: int
    origin_behavior: str
    start_boundary: int
    schedule_class: str
    target_behavior: str
    transition_boundaries: tuple[int, ...]

    @property
    def counted_transitions(self) -> int:
        return 1_000 - self.start_boundary

    def behavior_at(self, boundary: int) -> str:
        if not self.start_boundary <= boundary <= 1_000:
            raise ValueError("requested boundary lies outside the RSI episode")
        if self.schedule_class == "hold" or not self.transition_boundaries:
            return self.origin_behavior
        if self.schedule_class == "one_way":
            return (
                self.origin_behavior
                if boundary < self.transition_boundaries[0]
                else self.target_behavior
            )
        first, second = self.transition_boundaries
        if boundary < first:
            return self.origin_behavior
        return self.target_behavior if boundary < second else self.origin_behavior

    def to_dict(self) -> dict[str, object]:
        return {
            "block": self.block,
            "cell_index": self.cell_index,
            "counted_transitions": self.counted_transitions,
            "cycle": self.cycle,
            "environment_index": self.environment_index,
            "global_episode_index": self.global_episode_index,
            "origin_behavior": self.origin_behavior,
            "ppo_seed": self.ppo_seed,
            "schedule_class": self.schedule_class,
            "start_boundary": self.start_boundary,
            "target_behavior": self.target_behavior,
            "transition_boundaries": list(self.transition_boundaries),
        }


class BalancedRSIScheduler:
    """SHA-ranked 27-cell scheduler with balanced schedule classes."""

    def __init__(
        self,
        *,
        manifest_sha256: str,
        ppo_seed: int,
        study_pairing: T2RewardPairing | None = None,
    ) -> None:
        if study_pairing is None:
            domain_separated_seed(manifest_sha256, ppo_seed, "scheduler-construction")
        else:
            if manifest_sha256 not in {
                study_pairing.baseline_arm_manifest_sha256,
                study_pairing.candidate_arm_manifest_sha256,
            }:
                raise ValueError("scheduler manifest is not a verified paired arm identity")
            paired_domain_separated_seed(study_pairing, ppo_seed, "scheduler-construction")
        self.manifest_sha256 = manifest_sha256
        self.ppo_seed = ppo_seed
        self.study_pairing = study_pairing
        behaviors = ("expert",) if study_pairing is not None else BEHAVIORS
        self._cells = tuple(
            (block, behavior) for block in TRAINING_BLOCKS for behavior in behaviors
        )

    @classmethod
    def from_plan(cls, plan: TrainingPlan) -> BalancedRSIScheduler:
        if type(plan) is not TrainingPlan:
            raise ValueError("scheduler requires an exact TrainingPlan")
        return cls(
            manifest_sha256=plan.manifest_sha256,
            ppo_seed=plan.seed,
            study_pairing=plan.study_pairing,
        )

    @property
    def pairing_declared(self) -> bool:
        return self.study_pairing is not None

    def _seed(
        self,
        domain: str,
        *,
        index: int = 0,
        environment_index: int | None = None,
    ) -> int:
        if self.study_pairing is None:
            return domain_separated_seed(self.manifest_sha256, self.ppo_seed, domain, index)
        paired_domain = (
            domain
            if environment_index is None
            else f"{domain}:environment_slot:{environment_index}"
        )
        return paired_domain_separated_seed(
            self.study_pairing,
            self.ppo_seed,
            paired_domain,
            index,
        )

    def _ranked_cells(
        self,
        cycle: int,
        *,
        environment_index: int | None = None,
    ) -> tuple[tuple[int, str], ...]:
        def key(cell: tuple[int, str]) -> bytes:
            if self.study_pairing is not None:
                seed = self._seed(
                    f"rsi_order:block:{cell[0]}:origin:{cell[1]}",
                    index=cycle,
                    environment_index=environment_index,
                )
                return seed.to_bytes(8, "big", signed=False)
            return hashlib.sha256(
                canonical_json_bytes(
                    {
                        "cell": [cell[0], cell[1]],
                        "cycle": cycle,
                        "manifest_sha256": self.manifest_sha256,
                        "ppo_seed": self.ppo_seed,
                        "scheduler_id": RSI_SCHEDULER_ID,
                    }
                )
            ).digest()

        return tuple(sorted(self._cells, key=key))

    def assignment(self, *, global_episode_index: int, environment_index: int) -> RSIAssignment:
        if type(global_episode_index) is not int or global_episode_index < 0:
            raise ValueError("global_episode_index must be non-negative")
        if environment_index not in {2, 3}:
            raise ValueError("RSI assignments belong only to rehearsal environments 2 and 3")
        cycle, position = divmod(global_episode_index, len(self._cells))
        block, origin = self._ranked_cells(
            cycle,
            environment_index=environment_index,
        )[position]
        canonical_cell_index = self._cells.index((block, origin))
        if self.study_pairing is not None:
            class_seed = self._seed(
                "rsi_class",
                index=global_episode_index,
                environment_index=environment_index,
            )
            start_seed = self._seed(
                "rsi_start",
                index=global_episode_index,
                environment_index=environment_index,
            )
            return RSIAssignment(
                ppo_seed=self.ppo_seed,
                environment_index=environment_index,
                global_episode_index=global_episode_index,
                cycle=cycle,
                cell_index=canonical_cell_index,
                block=block,
                origin_behavior="expert",
                start_boundary=start_seed % 489,
                schedule_class=("hold",)[class_seed % 1],
                target_behavior="expert",
                transition_boundaries=(),
            )
        class_offset = domain_separated_seed(
            self.manifest_sha256,
            self.ppo_seed,
            f"schedule-class:{block}:{origin}",
        ) % len(SCHEDULE_CLASSES)
        schedule_class = SCHEDULE_CLASSES[(cycle + class_offset) % len(SCHEDULE_CLASSES)]
        target = ("medium", "simple")[cycle % 2] if origin == "expert" else "expert"
        start_payload = canonical_json_bytes(
            {
                "global_episode_index": global_episode_index,
                "manifest_sha256": self.manifest_sha256,
                "ppo_seed": self.ppo_seed,
                "purpose": "rsi-start-boundary/v1",
            }
        )
        start = int.from_bytes(hashlib.sha256(start_payload).digest()[:8], "big") % 489
        remaining = 1_000 - start
        if schedule_class == "hold":
            transitions: tuple[int, ...] = ()
        elif schedule_class == "one_way":
            transitions = (start + remaining // 2,)
        else:
            transitions = (start + remaining // 3, start + (2 * remaining) // 3)
        return RSIAssignment(
            ppo_seed=self.ppo_seed,
            environment_index=environment_index,
            global_episode_index=global_episode_index,
            cycle=cycle,
            cell_index=canonical_cell_index,
            block=block,
            origin_behavior=origin,
            start_boundary=start,
            schedule_class=schedule_class,
            target_behavior=target,
            transition_boundaries=transitions,
        )

    def composition_block(
        self,
        *,
        global_episode_index: int,
        environment_index: int,
    ) -> int:
        if self.study_pairing is None:
            raise ValueError("indexed composition blocks belong only to a paired study")
        if type(global_episode_index) is not int or global_episode_index < 0:
            raise ValueError("global_episode_index must be non-negative")
        if environment_index not in {0, 1}:
            raise ValueError("composition blocks belong only to environments 0 and 1")
        cycle, position = divmod(global_episode_index, len(TRAINING_BLOCKS))
        ranked = tuple(
            sorted(
                TRAINING_BLOCKS,
                key=lambda block: self._seed(
                    f"composition_block:block:{block}",
                    index=cycle,
                    environment_index=environment_index,
                ),
            )
        )
        return ranked[position]

    def audit_prefix(self, count: int, *, environment_index: int = 2) -> dict[str, object]:
        if type(count) is not int or count <= 0:
            raise ValueError("scheduler audit count must be positive")
        assignments = [
            self.assignment(global_episode_index=index, environment_index=environment_index)
            for index in range(count)
        ]
        cell_counts = Counter((item.block, item.origin_behavior) for item in assignments)
        class_counts: dict[str, Counter[str]] = {}
        for item in assignments:
            key = f"{item.block}:{item.origin_behavior}"
            class_counts.setdefault(key, Counter())[item.schedule_class] += 1
        if max(cell_counts.values()) - min(cell_counts.values()) > 1:
            raise AssertionError("balanced RSI cell accounting drifted")
        schedule_classes = ("hold",) if self.study_pairing is not None else SCHEDULE_CLASSES
        for counter in class_counts.values():
            values = [counter[name] for name in schedule_classes]
            if max(values) - min(values) > 1:
                raise AssertionError("balanced RSI schedule-class accounting drifted")
        encoded = canonical_json_bytes([item.to_dict() for item in assignments])
        return {
            "assignment_count": count,
            "ledger_sha256": hashlib.sha256(encoded).hexdigest(),
            "maximum_cell_count_delta": max(cell_counts.values()) - min(cell_counts.values()),
            "scheduler_id": RSI_SCHEDULER_ID,
        }


@dataclass(frozen=True, slots=True)
class RSIRestorationReceipt:
    start_boundary: int
    predecessor_boundary: int | None
    predecessor_executed: bool
    predecessor_counted: bool
    counted_transitions_before: int
    counted_transitions_after: int
    wrapper_elapsed_after: int
    observation_sha256: str
    integration_state_sha256: str
    rng_state_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "counted_transitions_after": self.counted_transitions_after,
            "counted_transitions_before": self.counted_transitions_before,
            "integration_state_sha256": self.integration_state_sha256,
            "observation_sha256": self.observation_sha256,
            "predecessor_boundary": self.predecessor_boundary,
            "predecessor_counted": self.predecessor_counted,
            "predecessor_executed": self.predecessor_executed,
            "rng_state_sha256": self.rng_state_sha256,
            "start_boundary": self.start_boundary,
            "wrapper_elapsed_after": self.wrapper_elapsed_after,
        }


def validate_rsi_restoration_receipt(receipt: RSIRestorationReceipt) -> None:
    """Reject direct restore, counted reconstruction, and counter-reset shortcuts."""

    if type(receipt) is not RSIRestorationReceipt:
        raise PhaseBTrainingError(
            TrainingFailureStatus.COUNTER_DRIFT,
            "RSI restoration receipt authority differs",
        )
    if receipt.start_boundary == 0:
        if receipt.predecessor_boundary is not None or receipt.predecessor_executed:
            raise PhaseBTrainingError(
                TrainingFailureStatus.COUNTER_DRIFT,
                "boundary-zero RSI must use the certified reset",
            )
    elif (
        receipt.predecessor_boundary != receipt.start_boundary - 1
        or not receipt.predecessor_executed
    ):
        raise PhaseBTrainingError(
            TrainingFailureStatus.COUNTER_DRIFT,
            "RSI direct boundary restore without the predecessor step is forbidden",
        )
    if receipt.predecessor_counted or (
        receipt.counted_transitions_before != receipt.counted_transitions_after
    ):
        raise PhaseBTrainingError(
            TrainingFailureStatus.COUNTER_DRIFT,
            "RSI predecessor reconstruction must not be counted",
        )
    if receipt.wrapper_elapsed_after != receipt.start_boundary:
        raise PhaseBTrainingError(
            TrainingFailureStatus.COUNTER_DRIFT,
            "RSI wrapper counter was reset or drifted",
        )
    for value in (
        receipt.observation_sha256,
        receipt.integration_state_sha256,
        receipt.rng_state_sha256,
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise PhaseBTrainingError(
                TrainingFailureStatus.COUNTER_DRIFT,
                "RSI restoration identity is invalid",
            )


@dataclass(slots=True)
class _Rollout:
    observations: np.ndarray
    pre_tanh: np.ndarray
    old_log_prob: np.ndarray
    rollout_mean: np.ndarray
    rollout_log_std: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray


@dataclass(frozen=True, slots=True)
class LikelihoodAudit:
    rollout_index: int
    sample_count: int
    sample_indices_sha256: str
    observation_sample_sha256: str
    pre_tanh_sample_sha256: str
    old_log_prob_sample_sha256: str
    distribution_snapshot_sha256: str
    maximum_absolute_difference: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "audit_id": LIKELIHOOD_AUDIT_ID,
            "audit_stage": "before_any_update_for_rollout",
            "distribution_snapshot_sha256": self.distribution_snapshot_sha256,
            "maximum_absolute_difference": self.maximum_absolute_difference,
            "observation_sample_sha256": self.observation_sample_sha256,
            "old_log_prob_sample_sha256": self.old_log_prob_sample_sha256,
            "passed": self.passed,
            "pre_tanh_sample_sha256": self.pre_tanh_sample_sha256,
            "rollout_index": self.rollout_index,
            "sample_count": self.sample_count,
            "sample_indices_sha256": self.sample_indices_sha256,
            "tolerance": LIKELIHOOD_TOLERANCE,
        }


def compute_truncation_aware_gae(
    *,
    rewards: np.ndarray,
    dones: np.ndarray,
    truncations: np.ndarray,
    terminal_values: np.ndarray,
    values: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute GAE with terminal values used only for time-limit truncations."""

    raw_rewards = np.asarray(rewards)
    done_flags = np.asarray(dones)
    truncation_flags = np.asarray(truncations)
    terminal = np.asarray(terminal_values)
    estimates = np.asarray(values)
    final_estimates = np.asarray(last_values)
    if (
        raw_rewards.dtype.str != "<f4"
        or estimates.dtype.str != "<f4"
        or terminal.dtype.str != "<f4"
        or done_flags.dtype != np.dtype(np.bool_)
        or truncation_flags.dtype != np.dtype(np.bool_)
        or raw_rewards.ndim != 2
        or estimates.shape != raw_rewards.shape
        or terminal.shape != raw_rewards.shape
        or done_flags.shape != raw_rewards.shape
        or truncation_flags.shape != raw_rewards.shape
        or final_estimates.dtype.str != "<f4"
        or final_estimates.shape != (raw_rewards.shape[1],)
        or not all(
            np.isfinite(value).all()
            for value in (raw_rewards, terminal, estimates, final_estimates)
        )
        or np.any(truncation_flags & ~done_flags)
        or type(gamma) is not float
        or type(gae_lambda) is not float
        or not 0.0 < gamma <= 1.0
        or not 0.0 < gae_lambda <= 1.0
    ):
        raise PhaseBTrainingError(
            TrainingFailureStatus.COUNTER_DRIFT,
            "truncation-aware GAE inputs differ from the rollout contract",
        )
    adjusted_rewards = np.ascontiguousarray(
        raw_rewards + np.float32(gamma) * terminal * truncation_flags.astype("<f4"),
        dtype="<f4",
    )
    advantages = np.zeros_like(adjusted_rewards, dtype="<f4")
    last_gae = np.zeros(raw_rewards.shape[1], dtype="<f4")
    for index in reversed(range(raw_rewards.shape[0])):
        next_nonterminal = np.float32(1.0) - done_flags[index].astype("<f4")
        next_value = final_estimates if index == raw_rewards.shape[0] - 1 else estimates[index + 1]
        delta = (
            adjusted_rewards[index]
            + np.float32(gamma) * next_value * next_nonterminal
            - estimates[index]
        )
        last_gae = delta + (np.float32(gamma * gae_lambda) * next_nonterminal * last_gae)
        advantages[index] = last_gae
    returns = np.ascontiguousarray(advantages + estimates, dtype="<f4")
    if not all(np.isfinite(value).all() for value in (adjusted_rewards, advantages, returns)):
        raise PhaseBTrainingError(
            TrainingFailureStatus.NON_FINITE,
            "truncation-aware GAE became NaN or Inf",
        )
    return adjusted_rewards, advantages, returns


@dataclass(slots=True)
class TrainingResult:
    policy: FullAuthorityPolicy
    optimizer: torch.optim.Optimizer
    scientific_facts: Mapping[str, object]
    rsi_ledger: tuple[Mapping[str, object], ...]
    progress_windows: tuple[Mapping[str, object], ...]
    worker_cleanup: Mapping[str, object]


def _policy_input(observations: np.ndarray) -> object:
    value = np.ascontiguousarray(observations, dtype="<f4")
    if value.ndim != 2 or value.shape[1] != POLICY_INPUT_WIDTH or not np.isfinite(value).all():
        raise PhaseBTrainingError(
            TrainingFailureStatus.NON_FINITE,
            "worker observation batch must be finite float32[N,708]",
        )
    state = np.ascontiguousarray(value[:, :OBSERVATION_WIDTH], dtype="<f4")
    reference = np.ascontiguousarray(
        value[:, OBSERVATION_WIDTH:].reshape(-1, REFERENCE_HORIZON, REFERENCE_WIDTH),
        dtype="<f4",
    )
    return compose_policy_input(state, reference)


def _finite_tensor(value: torch.Tensor, *, field: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise PhaseBTrainingError(
            TrainingFailureStatus.NON_FINITE,
            f"PPO {field} became NaN or Inf",
        )


def _validate_step_info(info: object, *, expected_stream: str) -> None:
    if type(info) is not dict or type(info.get("phase_b")) is not dict:
        raise PhaseBTrainingError(
            TrainingFailureStatus.COUNTER_DRIFT,
            "training environment omitted Phase B step evidence",
        )
    facts = info["phase_b"]
    required = {
        "counted_transition_delta",
        "ignored_stock_reward",
        "phase_selection_ok",
        "r_task",
        "r_track",
        "r_train",
        "stream",
    }
    if not required.issubset(facts) or facts["stream"] != expected_stream:
        raise PhaseBTrainingError(
            TrainingFailureStatus.COUNTER_DRIFT,
            "training stream identity or step evidence drifted",
        )
    if facts["counted_transition_delta"] != 1:
        raise PhaseBTrainingError(
            TrainingFailureStatus.COUNTER_DRIFT,
            "counted transition delta differs from one",
        )
    if facts["phase_selection_ok"] is not True:
        raise PhaseBTrainingError(
            TrainingFailureStatus.PHASE_SELECTION_FAILURE,
            "reference phase selection failed",
        )
    values = [facts[name] for name in ("r_track", "r_task", "r_train", "ignored_stock_reward")]
    if any(type(value) not in {int, float} or not math.isfinite(float(value)) for value in values):
        raise PhaseBTrainingError(
            TrainingFailureStatus.NON_FINITE,
            "training reward evidence became NaN or Inf",
        )
    if not math.isclose(
        float(facts["r_train"]),
        float(facts["r_track"]) + float(facts["r_task"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise PhaseBTrainingError(
            TrainingFailureStatus.COUNTER_DRIFT,
            "training reward streams were collapsed or changed",
        )


def _collect_rollout(
    *,
    vector_environment: object,
    policy: FullAuthorityPolicy,
    observations: np.ndarray,
    plan: TrainingPlan,
    rollout_index: int,
    action_rng: np.random.Generator,
    stream_counts: Counter[str],
    reward_totals: Counter[str],
) -> tuple[_Rollout, np.ndarray]:
    observation_rows: list[np.ndarray] = []
    pre_tanh_rows: list[np.ndarray] = []
    log_prob_rows: list[np.ndarray] = []
    mean_rows: list[np.ndarray] = []
    log_std_rows: list[np.ndarray] = []
    reward_rows: list[np.ndarray] = []
    done_rows: list[np.ndarray] = []
    truncation_rows: list[np.ndarray] = []
    terminal_value_rows: list[np.ndarray] = []
    value_rows: list[np.ndarray] = []
    current = np.ascontiguousarray(observations, dtype="<f4")
    paired_noise = (
        paired_action_noise(
            plan,
            rollout_index=rollout_index,
            steps_per_environment=plan.steps_per_environment,
        )
        if plan.pairing_declared
        else None
    )
    for step_index in range(plan.steps_per_environment):
        strict_input = _policy_input(current)
        epsilon = (
            paired_noise[step_index]
            if paired_noise is not None
            else np.ascontiguousarray(
                action_rng.standard_normal((plan.n_envs, ACTION_WIDTH)).astype("<f4"),
                dtype="<f4",
            )
        )
        action = policy.actor.act(strict_input, epsilon=epsilon)
        log_prob = policy.actor.log_likelihood(strict_input, action.pre_tanh)
        with torch.inference_mode():
            value = policy.value_estimate(strict_input).squeeze(-1).detach().cpu().numpy()
        physical = action.physical
        if (
            not np.isfinite(physical).all()
            or np.any(physical < np.float32(-0.4))
            or np.any(physical > np.float32(0.4))
        ):
            raise PhaseBTrainingError(
                TrainingFailureStatus.ACTION_BOUND_VIOLATION,
                "worker produced an out-of-bounds physical action",
            )
        next_observations, rewards, dones, infos = vector_environment.step(physical)
        next_value = np.ascontiguousarray(next_observations, dtype="<f4")
        rewards_value = np.ascontiguousarray(rewards, dtype="<f4")
        dones_value = np.ascontiguousarray(dones, dtype=np.bool_)
        truncations_value = np.zeros(plan.n_envs, dtype=np.bool_)
        terminal_values = np.zeros(plan.n_envs, dtype="<f4")
        if (
            next_value.shape != (plan.n_envs, POLICY_INPUT_WIDTH)
            or rewards_value.shape != (plan.n_envs,)
            or dones_value.shape != (plan.n_envs,)
            or not np.isfinite(next_value).all()
            or not np.isfinite(rewards_value).all()
        ):
            raise PhaseBTrainingError(
                TrainingFailureStatus.NON_FINITE,
                "environment returned non-finite or malformed rollout data",
            )
        for environment_index, info in enumerate(infos):
            expected_stream = STREAM_BY_ENVIRONMENT[environment_index]
            _validate_step_info(info, expected_stream=expected_stream)
            phase_b = info["phase_b"]
            for field in ("ignored_stock_reward", "r_task", "r_track", "r_train"):
                reward_totals[field] += float(phase_b[field])
            stream_counts[expected_stream] += 1
            if dones_value[environment_index] and info.get("TimeLimit.truncated", False):
                terminal_observation = np.asarray(info.get("terminal_observation"))
                if terminal_observation.shape != (POLICY_INPUT_WIDTH,):
                    raise PhaseBTrainingError(
                        TrainingFailureStatus.COUNTER_DRIFT,
                        "time-limit transition omitted its terminal observation",
                    )
                with torch.inference_mode():
                    terminal_value = float(
                        policy.value_estimate(
                            _policy_input(
                                np.ascontiguousarray(terminal_observation[None, :], dtype="<f4")
                            )
                        )[0, 0]
                    )
                if not math.isfinite(terminal_value):
                    raise PhaseBTrainingError(
                        TrainingFailureStatus.NON_FINITE,
                        "time-limit bootstrap value became NaN or Inf",
                    )
                truncations_value[environment_index] = True
                terminal_values[environment_index] = np.float32(terminal_value)
                reward_totals["time_limit_bootstrap_count"] += 1
        observation_rows.append(current.copy())
        pre_tanh_rows.append(action.pre_tanh.copy())
        log_prob_rows.append(log_prob.copy())
        mean_rows.append(action.mean.copy())
        log_std_rows.append(action.log_std.copy())
        reward_rows.append(rewards_value.copy())
        done_rows.append(dones_value.copy())
        truncation_rows.append(truncations_value)
        terminal_value_rows.append(terminal_values)
        value_rows.append(np.ascontiguousarray(value, dtype="<f4"))
        current = next_value

    with torch.inference_mode():
        last_values = (
            policy.value_estimate(_policy_input(current)).squeeze(-1).detach().cpu().numpy()
        )
    raw_rewards = np.ascontiguousarray(np.stack(reward_rows), dtype="<f4")
    dones = np.ascontiguousarray(np.stack(done_rows), dtype=np.bool_)
    truncations = np.ascontiguousarray(np.stack(truncation_rows), dtype=np.bool_)
    terminal_values = np.ascontiguousarray(np.stack(terminal_value_rows), dtype="<f4")
    values = np.ascontiguousarray(np.stack(value_rows), dtype="<f4")
    rewards, advantages, returns = compute_truncation_aware_gae(
        rewards=raw_rewards,
        dones=dones,
        truncations=truncations,
        terminal_values=terminal_values,
        values=values,
        last_values=np.ascontiguousarray(last_values, dtype="<f4"),
        gamma=plan.recipe.gamma,
        gae_lambda=plan.recipe.gae_lambda,
    )
    return (
        _Rollout(
            observations=np.ascontiguousarray(np.stack(observation_rows), dtype="<f4"),
            pre_tanh=np.ascontiguousarray(np.stack(pre_tanh_rows), dtype="<f4"),
            old_log_prob=np.ascontiguousarray(np.stack(log_prob_rows), dtype="<f4"),
            rollout_mean=np.ascontiguousarray(np.stack(mean_rows), dtype="<f4"),
            rollout_log_std=np.ascontiguousarray(np.stack(log_std_rows), dtype="<f4"),
            rewards=rewards,
            dones=dones,
            values=values,
            advantages=advantages,
            returns=returns,
        ),
        current,
    )


def _actor_snapshot(policy: FullAuthorityPolicy) -> dict[str, bytes]:
    return {
        name: value.detach().cpu().numpy().tobytes(order="C")
        for name, value in policy.actor.named_parameters()
    }


def _apply_initial_gradient_mask(policy: FullAuthorityPolicy) -> None:
    for name, parameter in policy.actor.named_parameters():
        if parameter.grad is None:
            continue
        if name == "latent_0.weight":
            parameter.grad[:, :OBSERVATION_WIDTH] = 0.0
        else:
            parameter.grad = None


def _verify_initial_mask(before: Mapping[str, bytes], policy: FullAuthorityPolicy) -> None:
    after = _actor_snapshot(policy)
    for name in before:
        if name == "latent_0.weight":
            before_array = np.frombuffer(before[name], dtype="<f4").reshape(
                policy.actor.latent_0.weight.shape
            )
            after_array = policy.actor.latent_0.weight.detach().cpu().numpy()
            if not np.array_equal(
                before_array[:, :OBSERVATION_WIDTH], after_array[:, :OBSERVATION_WIDTH]
            ):
                raise PhaseBTrainingError(
                    TrainingFailureStatus.COUNTER_DRIFT,
                    "state columns changed during the reference-only stage",
                )
        elif before[name] != after[name]:
            raise PhaseBTrainingError(
                TrainingFailureStatus.COUNTER_DRIFT,
                f"actor parameter {name} changed during the reference-only stage",
            )


def _ppo_update(
    *,
    rollout: _Rollout,
    policy: FullAuthorityPolicy,
    optimizer: torch.optim.Optimizer,
    plan: TrainingPlan,
    rollout_index: int,
    minibatch_rng: np.random.Generator,
) -> tuple[dict[str, float], int, dict[str, object]]:
    sample_count = plan.transitions_per_rollout
    observations = rollout.observations.reshape(sample_count, POLICY_INPUT_WIDTH)
    pre_tanh = rollout.pre_tanh.reshape(sample_count, ACTION_WIDTH)
    old_log_prob = rollout.old_log_prob.reshape(sample_count)
    advantages = rollout.advantages.reshape(sample_count).astype("<f4", copy=True)
    returns = rollout.returns.reshape(sample_count)
    initial_stage = rollout_index < REFERENCE_ONLY_ROLLOUTS
    stage = "reference_columns_only" if initial_stage else "full_actor"
    optimizer_before = verify_optimizer_authority(
        policy,
        optimizer,
        expected_learning_rate=plan.recipe.learning_rate,
        stage=f"before_{stage}",
    )
    actor_before = _actor_snapshot(policy)
    totals = Counter[str]()
    updates = 0
    for epoch_index in range(plan.recipe.n_epochs):
        ordering = (
            paired_minibatch_permutation(
                plan,
                update_index=rollout_index * plan.recipe.n_epochs + epoch_index,
                sample_count=sample_count,
            )
            if plan.pairing_declared
            else minibatch_rng.permutation(sample_count)
        )
        for start in range(0, sample_count, plan.recipe.batch_size):
            indices = ordering[start : start + plan.recipe.batch_size]
            strict_input = _policy_input(np.ascontiguousarray(observations[indices], dtype="<f4"))
            mean, log_std = policy.actor._distribution_tensors(strict_input)
            selected_pre_tanh = torch.from_numpy(
                np.ascontiguousarray(pre_tanh[indices], dtype="<f4")
            )
            new_log_prob = policy.actor.log_likelihood(
                strict_input,
                np.ascontiguousarray(pre_tanh[indices], dtype="<f4"),
            )
            # Recompute in Torch for gradients; the NumPy-facing call above is the audit path.
            from .policy import squashed_gaussian_log_likelihood

            differentiable_log_prob = squashed_gaussian_log_likelihood(
                selected_pre_tanh,
                mean,
                log_std,
            )
            if not np.allclose(
                new_log_prob,
                differentiable_log_prob.detach().cpu().numpy(),
                rtol=0.0,
                atol=0.0,
            ):
                raise PhaseBTrainingError(
                    TrainingFailureStatus.LIKELIHOOD_FAILURE,
                    "worker likelihood paths differ",
                )
            old = torch.from_numpy(np.ascontiguousarray(old_log_prob[indices], dtype="<f4"))
            advantage = torch.from_numpy(np.ascontiguousarray(advantages[indices], dtype="<f4"))
            if plan.recipe.normalize_advantage:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            target_return = torch.from_numpy(np.ascontiguousarray(returns[indices], dtype="<f4"))
            ratio = torch.exp(differentiable_log_prob - old)
            clipped_ratio = torch.clamp(
                ratio,
                1.0 - plan.recipe.clip_range,
                1.0 + plan.recipe.clip_range,
            )
            policy_loss = -torch.minimum(ratio * advantage, clipped_ratio * advantage).mean()
            value = policy.value_estimate(strict_input).squeeze(-1)
            value_loss = torch.mean(torch.square(target_return - value))
            entropy_loss = -(-differentiable_log_prob).mean()
            total_loss = (
                policy_loss + plan.recipe.ent_coef * entropy_loss + plan.recipe.vf_coef * value_loss
            )
            with torch.no_grad():
                log_ratio = differentiable_log_prob - old
                approximate_kl = torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio)
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1.0) > plan.recipe.clip_range).to(torch.float32)
                )
            for tensor, field in (
                (policy_loss, "policy loss"),
                (value_loss, "value loss"),
                (total_loss, "total loss"),
            ):
                _finite_tensor(tensor, field=field)
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if initial_stage:
                _apply_initial_gradient_mask(policy)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), plan.recipe.max_grad_norm)
            optimizer.step()
            totals["approximate_kl"] += float(approximate_kl.detach())
            totals["clip_fraction"] += float(clip_fraction.detach())
            totals["entropy_loss"] += float(entropy_loss.detach())
            totals["policy_loss"] += float(policy_loss.detach())
            totals["value_loss"] += float(value_loss.detach())
            totals["total_loss"] += float(total_loss.detach())
            updates += 1
    if initial_stage:
        _verify_initial_mask(actor_before, policy)
    state_before = np.frombuffer(actor_before["latent_0.weight"], dtype="<f4").reshape(
        policy.actor.latent_0.weight.shape
    )[:, :OBSERVATION_WIDTH]
    reference = policy.actor.latent_0.weight.detach().cpu().numpy()[:, OBSERVATION_WIDTH:]
    state = policy.actor.latent_0.weight.detach().cpu().numpy()[:, :OBSERVATION_WIDTH]
    state_columns_changed = not np.array_equal(state_before, state)
    if initial_stage and state_columns_changed:
        raise PhaseBTrainingError(
            TrainingFailureStatus.COUNTER_DRIFT,
            "state columns changed during the reference-only stage",
        )
    if rollout_index == REFERENCE_ONLY_ROLLOUTS and not state_columns_changed:
        raise PhaseBTrainingError(
            TrainingFailureStatus.COUNTER_DRIFT,
            "the first full-actor update did not change state columns",
        )
    receipt = {
        "actor_stage": stage,
        "optimizer_authority_after": verify_optimizer_authority(
            policy,
            optimizer,
            expected_learning_rate=plan.recipe.learning_rate,
            stage=f"after_{stage}",
        ),
        "optimizer_authority_before": optimizer_before,
        "reference_columns_sha256": array_sha256(np.ascontiguousarray(reference, dtype="<f4")),
        "rollout_index": rollout_index,
        "state_columns_changed": state_columns_changed,
        "state_columns_sha256": array_sha256(np.ascontiguousarray(state, dtype="<f4")),
        "unfreeze_schedule_id": UNFREEZE_SCHEDULE_ID,
        "value_network_trainable": True,
    }
    target = returns.astype(np.float64)
    residual = target - rollout.values.reshape(sample_count).astype(np.float64)
    target_variance = float(np.var(target))
    explained_variance = (
        float("nan") if target_variance == 0.0 else 1.0 - float(np.var(residual)) / target_variance
    )
    if not math.isfinite(explained_variance):
        explained_variance = 0.0
    divisor = float(updates)
    losses = {name: value / divisor for name, value in sorted(totals.items())}
    losses["explained_variance"] = explained_variance
    return losses, updates, receipt


def audit_rollout_likelihood(rollout: _Rollout, *, rollout_index: int) -> LikelihoodAudit:
    """Replay stored rollout likelihoods from the rollout-time distribution snapshot."""

    observations = rollout.observations.reshape(-1, POLICY_INPUT_WIDTH)
    pre_tanh = rollout.pre_tanh.reshape(-1, ACTION_WIDTH)
    old_log_prob = rollout.old_log_prob.reshape(-1)
    means = rollout.rollout_mean.reshape(-1, ACTION_WIDTH)
    log_stds = rollout.rollout_log_std.reshape(-1, ACTION_WIDTH)
    if not (len(observations) == len(pre_tanh) == len(old_log_prob) == len(means) == len(log_stds)):
        raise PhaseBTrainingError(
            TrainingFailureStatus.LIKELIHOOD_FAILURE,
            "rollout likelihood snapshot lengths differ",
        )
    sample_count = min(len(observations), 64)
    indices = np.linspace(0, len(observations) - 1, num=sample_count, dtype="<i8")
    selected_observations = np.ascontiguousarray(observations[indices], dtype="<f4")
    selected_pre_tanh = np.ascontiguousarray(pre_tanh[indices], dtype="<f4")
    selected_old = np.ascontiguousarray(old_log_prob[indices], dtype="<f4")
    selected_mean = np.ascontiguousarray(means[indices], dtype="<f4")
    selected_log_std = np.ascontiguousarray(log_stds[indices], dtype="<f4")
    independent = numpy_log_likelihood(
        selected_pre_tanh,
        selected_mean,
        selected_log_std,
    )
    maximum = float(np.max(np.abs(selected_old.astype(np.float64) - independent), initial=0.0))
    passed = math.isfinite(maximum) and maximum <= LIKELIHOOD_TOLERANCE
    if not passed:
        raise PhaseBTrainingError(
            TrainingFailureStatus.LIKELIHOOD_FAILURE,
            f"tanh-corrected likelihood audit differs by {maximum}",
        )
    return LikelihoodAudit(
        rollout_index=rollout_index,
        sample_count=sample_count,
        sample_indices_sha256=array_sha256(indices),
        observation_sample_sha256=array_sha256(selected_observations),
        pre_tanh_sample_sha256=array_sha256(selected_pre_tanh),
        old_log_prob_sample_sha256=array_sha256(selected_old),
        distribution_snapshot_sha256=hashlib.sha256(
            canonical_json_bytes(
                {
                    "log_std_sha256": array_sha256(selected_log_std),
                    "mean_sha256": array_sha256(selected_mean),
                }
            )
        ).hexdigest(),
        maximum_absolute_difference=maximum,
        passed=True,
    )


def _runtime_rsi_ledger(
    vector_environment: object,
    *,
    plan: TrainingPlan,
) -> tuple[Mapping[str, object], ...]:
    try:
        values = vector_environment.get_attr("rsi_ledger")
        reset_counts = vector_environment.get_attr("rsi_reset_count")
    except Exception as exc:
        raise PhaseBTrainingError(
            TrainingFailureStatus.COUNTER_DRIFT,
            f"RSI ledger API is unavailable: {exc}",
        ) from exc
    if (
        type(values) is not list
        or type(reset_counts) is not list
        or len(values) != plan.n_envs
        or len(reset_counts) != plan.n_envs
    ):
        raise PhaseBTrainingError(
            TrainingFailureStatus.COUNTER_DRIFT,
            "RSI ledger vector accounting is malformed",
        )
    entries: list[Mapping[str, object]] = []
    for environment_index, (value, reset_count) in enumerate(
        zip(values, reset_counts, strict=True)
    ):
        if value is None or type(value) is not list or type(reset_count) is not int:
            raise PhaseBTrainingError(
                TrainingFailureStatus.COUNTER_DRIFT,
                "RSI ledger or reset count is missing",
            )
        expected_count = reset_count if environment_index in {2, 3} else 0
        if reset_count != expected_count or len(value) != expected_count:
            raise PhaseBTrainingError(
                TrainingFailureStatus.COUNTER_DRIFT,
                "RSI ledger does not account for every rehearsal reset",
            )
        for entry in value:
            if type(entry) is not dict:
                raise PhaseBTrainingError(
                    TrainingFailureStatus.COUNTER_DRIFT,
                    "RSI ledger entry is malformed",
                )
            entries.append(MappingProxyType(dict(entry)))
    if not entries or {item.get("environment_index") for item in entries} != {2, 3}:
        raise PhaseBTrainingError(
            TrainingFailureStatus.COUNTER_DRIFT,
            "RSI evidence requires entries from both rehearsal environments",
        )
    entries.sort(
        key=lambda item: (
            int(item.get("global_episode_index", -1)),
            int(item.get("environment_index", -1)),
        )
    )
    if plan.pairing_declared:
        entries.sort(
            key=lambda item: (
                int(item.get("environment_index", -1)),
                int(item.get("global_episode_index", -1)),
            )
        )
        for environment_index in (2, 3):
            indices = [
                item.get("global_episode_index")
                for item in entries
                if item.get("environment_index") == environment_index
            ]
            if indices != list(range(len(indices))):
                raise PhaseBTrainingError(
                    TrainingFailureStatus.COUNTER_DRIFT,
                    "paired RSI reset indices must be consecutive within each environment slot",
                )
    else:
        indices = [item.get("global_episode_index") for item in entries]
        if indices != list(range(len(entries))) or any(
            item.get("environment_index") not in {2, 3} for item in entries
        ):
            raise PhaseBTrainingError(
                TrainingFailureStatus.COUNTER_DRIFT,
                "RSI reset assignments are duplicated, missing, or out of vector order",
            )
    scheduler = BalancedRSIScheduler.from_plan(plan)
    for entry in entries:
        environment_index = entry["environment_index"]
        expected = scheduler.assignment(
            global_episode_index=int(entry["global_episode_index"]),
            environment_index=environment_index,
        ).to_dict()
        if any(entry.get(name) != value for name, value in expected.items()):
            raise PhaseBTrainingError(
                TrainingFailureStatus.COUNTER_DRIFT,
                "RSI scheduled cell distribution or reset assignment drifted",
            )
    return tuple(entries)


def run_ppo_training(
    *,
    plan: TrainingPlan,
    policy_factory: Callable[[TrainingPlan], FullAuthorityPolicy],
    environment_factories: Sequence[Callable[[], object]],
    progress_callback: Callable[[Mapping[str, object]], None] | None = None,
    step_zero_audit: Callable[[FullAuthorityPolicy], Mapping[str, object]] | None = None,
) -> TrainingResult:
    """Run the frozen PPO loop; factories are called only by an admitted worker."""

    if type(plan) is not TrainingPlan:
        raise ValueError("worker requires an exact TrainingPlan")
    if len(environment_factories) != 4 or not all(callable(item) for item in environment_factories):
        raise ValueError("worker requires four environment factories")
    random.seed(plan.seed)
    np.random.seed(plan.seed)
    torch.manual_seed(plan.seed)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True)
    policy = policy_factory(plan)
    if type(policy) is not FullAuthorityPolicy:
        raise ValueError("policy factory did not return the FT1 full-authority policy")
    policy.to(torch.device("cpu"))
    step_zero_comparator = (
        dict(step_zero_audit(policy))
        if step_zero_audit is not None
        else {
            "bitwise_equal": True,
            "evidence_class": "controlled_fake_runtime",
            "worker_path": "fake_policy_repeatability_only",
        }
    )
    if step_zero_comparator.get("bitwise_equal") is not True:
        raise PhaseBTrainingError(
            TrainingFailureStatus.ACTION_BOUND_VIOLATION,
            "step-0 worker action-path audit failed",
        )
    canonical_json_bytes(step_zero_comparator)
    optimizer = torch.optim.Adam(policy.parameters(), lr=plan.recipe.learning_rate)
    optimizer_initialization = verify_optimizer_authority(
        policy,
        optimizer,
        expected_learning_rate=plan.recipe.learning_rate,
        require_empty_state=True,
        stage="fresh_before_first_update",
    )
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:  # pragma: no cover - guarded by the train extra
        raise ExperimentContractError("Stable-Baselines3 is required for Phase B PPO") from exc
    vector_environment = DummyVecEnv(list(environment_factories))
    stream_counts: Counter[str] = Counter()
    reward_totals: Counter[str] = Counter()
    progress_windows: list[Mapping[str, object]] = []
    rollout_receipts: list[dict[str, object]] = []
    likelihood_audits: list[dict[str, object]] = []
    losses: list[dict[str, object]] = []
    update_count = 0
    observed_transitions = 0
    last_rollout: _Rollout | None = None
    try:
        reset = vector_environment.reset()
        observations = np.ascontiguousarray(reset, dtype="<f4")
        if observations.shape != (4, POLICY_INPUT_WIDTH) or not np.isfinite(observations).all():
            raise PhaseBTrainingError(
                TrainingFailureStatus.NON_FINITE,
                "vector reset returned a malformed policy observation",
            )
        action_seed = plan_domain_separated_seed(plan, "actions")
        minibatch_seed = plan_domain_separated_seed(plan, "minibatches")
        action_rng = np.random.Generator(np.random.PCG64(action_seed))
        minibatch_rng = np.random.Generator(np.random.PCG64(minibatch_seed))
        for rollout_index in range(plan.rollout_count):
            rollout, observations = _collect_rollout(
                vector_environment=vector_environment,
                policy=policy,
                observations=observations,
                plan=plan,
                rollout_index=rollout_index,
                action_rng=action_rng,
                stream_counts=stream_counts,
                reward_totals=reward_totals,
            )
            observed_transitions += plan.transitions_per_rollout
            if observed_transitions != (rollout_index + 1) * plan.transitions_per_rollout:
                raise PhaseBTrainingError(
                    TrainingFailureStatus.COUNTER_DRIFT,
                    "worker transition counter drifted",
                )
            likelihood = audit_rollout_likelihood(
                rollout,
                rollout_index=rollout_index,
            )
            likelihood_audits.append(likelihood.to_dict())
            rollout_loss, updates, unfreeze = _ppo_update(
                rollout=rollout,
                policy=policy,
                optimizer=optimizer,
                plan=plan,
                rollout_index=rollout_index,
                minibatch_rng=minibatch_rng,
            )
            update_count += updates
            losses.append({"rollout_index": rollout_index, **rollout_loss})
            rollout_receipts.append({**unfreeze, "rollout_likelihood_audit": likelihood.to_dict()})
            last_rollout = rollout
            progress = MappingProxyType(
                {
                    "observed_transitions": observed_transitions,
                    "rollout_count": rollout_index + 1,
                    "stream_counts": dict(sorted(stream_counts.items())),
                }
            )
            if observed_transitions % THROUGHPUT_WINDOW_TRANSITIONS == 0:
                progress_windows.append(progress)
            if progress_callback is not None:
                progress_callback(progress)
        if observed_transitions != plan.transitions:
            raise PhaseBTrainingError(
                TrainingFailureStatus.COUNTER_DRIFT,
                "worker did not stop at the exact final transition",
            )
        if stream_counts != Counter(
            {"composition": plan.transitions // 2, "rehearsal": plan.transitions // 2}
        ):
            raise PhaseBTrainingError(
                TrainingFailureStatus.COUNTER_DRIFT,
                "composition and rehearsal transitions are not exactly 50/50",
            )
        if last_rollout is None:
            raise AssertionError("validated training plan produced no rollout")
        rsi_ledger = _runtime_rsi_ledger(vector_environment, plan=plan)
        rsi_bytes = canonical_json_bytes([dict(item) for item in rsi_ledger])
        unfreeze_bytes = canonical_json_bytes(rollout_receipts)
        facts = {
            "device": "cpu",
            "evidence_class": plan.evidence_class,
            "execution_manifest_sha256": plan.manifest_sha256,
            "likelihood_audit": {
                "all_passed": all(item["passed"] is True for item in likelihood_audits),
                "audit_id": LIKELIHOOD_AUDIT_ID,
                "audit_stage": "before_any_update_for_each_rollout",
                "receipt_sha256": hashlib.sha256(
                    canonical_json_bytes(likelihood_audits)
                ).hexdigest(),
                "rollout_audits": likelihood_audits,
                "rollout_count": len(likelihood_audits),
            },
            "losses": losses,
            "normalization": {"observation": False, "reward": False},
            "observed_transitions": observed_transitions,
            "optimizer_updates": update_count,
            "optimizer_initialization": optimizer_initialization,
            "planned_transitions": plan.transitions,
            "ppo_recipe": plan.recipe.to_dict(),
            "ppo_recipe_id": PPO_RECIPE_ID,
            "ppo_seed": plan.seed,
            "promotable": plan.promotable,
            "rollouts": plan.rollout_count,
            "reward_totals": {
                name: float(reward_totals[name])
                for name in ("ignored_stock_reward", "r_task", "r_track", "r_train")
            },
            "rng_substreams": {
                "action_sampling": action_seed,
                "environment_order": [
                    plan_domain_separated_seed(plan, "vector-environment", index)
                    for index in range(plan.n_envs)
                ],
                "minibatches": minibatch_seed,
                "numpy_global": plan.seed,
                "python_global": plan.seed,
                "scheduler": plan_domain_separated_seed(plan, "scheduler-construction"),
                "torch_global": plan.seed,
            },
            "rsi_ledger_sha256": hashlib.sha256(rsi_bytes).hexdigest(),
            "smoke": plan.smoke,
            "stream_counts": dict(sorted(stream_counts.items())),
            "step_zero_comparator": step_zero_comparator,
            "thread_counts": {
                "torch_interop": torch.get_num_interop_threads(),
                "torch_intraop": torch.get_num_threads(),
            },
            "time_limit_bootstrap_count": int(reward_totals["time_limit_bootstrap_count"]),
            "training_worker_id": TRAINING_WORKER_ID,
            "unfreeze_receipt_sha256": hashlib.sha256(unfreeze_bytes).hexdigest(),
            "unfreeze_rollouts": rollout_receipts,
        }
        canonical_json_bytes(facts)
        result = TrainingResult(
            policy=policy,
            optimizer=optimizer,
            scientific_facts=MappingProxyType(facts),
            rsi_ledger=rsi_ledger,
            progress_windows=tuple(progress_windows),
            worker_cleanup=MappingProxyType({}),
        )
    except BaseException as primary_error:
        cleanup_error: BaseException | None = None
        try:
            vector_environment.close()
        except BaseException as exc:
            cleanup_error = exc
        if cleanup_error is not None:
            detail = f"{type(cleanup_error).__name__}: {cleanup_error}"[:1000]
            if isinstance(primary_error, PhaseBTrainingError):
                primary_error.cleanup_error = detail
            else:
                primary_error.add_note(f"vector-environment cleanup also failed: {detail}")
        raise
    else:
        try:
            vector_environment.close()
        except BaseException:
            raise
        result.worker_cleanup = MappingProxyType(
            {"attempted": True, "error": None, "succeeded": True}
        )
        return result


def verify_step_zero_worker_action_path(
    *,
    policy: FullAuthorityPolicy,
    states: np.ndarray,
    reference_windows: np.ndarray,
    expected_physical_actions: np.ndarray,
) -> dict[str, object]:
    """Exercise the worker's exact action path against frozen E1 fixtures."""

    strict_input = compose_policy_input(states, reference_windows)
    observed = policy.actor.act(strict_input).physical
    expected = np.ascontiguousarray(expected_physical_actions, dtype="<f4")
    if observed.shape != expected.shape or observed.tobytes(order="C") != expected.tobytes(
        order="C"
    ):
        raise PhaseBTrainingError(
            TrainingFailureStatus.ACTION_BOUND_VIOLATION,
            "step-0 worker action path differs from the expert fixture",
        )
    return {
        "action_sha256": array_sha256(observed),
        "bitwise_equal": True,
        "fixture_count": len(observed),
        "worker_path": "compose_policy_input_then_full_authority_actor_then_exact_physical_action",
    }


__all__ = [
    "BEHAVIORS",
    "COHORT_SEEDS",
    "FULL_TRANSITIONS_PER_SEED",
    "LIKELIHOOD_TOLERANCE",
    "REFERENCE_ONLY_ROLLOUTS",
    "SMOKE_SEED",
    "SMOKE_TRANSITIONS",
    "STREAM_BY_ENVIRONMENT",
    "TRAINING_BLOCKS",
    "BalancedRSIScheduler",
    "PPORecipe",
    "PhaseBTrainingError",
    "RSIAssignment",
    "RSIRestorationReceipt",
    "TrainingFailureStatus",
    "TrainingPlan",
    "TrainingResult",
    "audit_rollout_likelihood",
    "compute_truncation_aware_gae",
    "domain_separated_seed",
    "paired_action_noise",
    "paired_domain_separated_seed",
    "paired_minibatch_permutation",
    "plan_domain_separated_seed",
    "run_ppo_training",
    "validate_rsi_restoration_receipt",
    "verify_step_zero_worker_action_path",
]
