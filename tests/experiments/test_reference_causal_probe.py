from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from oracle_composition.contracts import ReferenceArtifact
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.local_controllers import ReferenceResidualInferenceFacts
from oracle_composition.experiments.reference_causal_probe import (
    SNAPSHOT_FRAMES,
    build_reference_causal_probe_report,
)
from oracle_composition.experiments.reference_input_transforms import CONDITION_IDS
from oracle_composition.sources import minari_humanoid as minari_source
from oracle_composition.sources.minari_humanoid import (
    HUMANOID_V5_OBSERVATION_TO_REFERENCE_INDICES,
    MinariHumanoidImportError,
    MinariHumanoidProjection,
    MinariHumanoidProjectionReceipt,
)
from oracle_composition.tracking.humanoid_reference import HUMANOID_REFERENCE_SCHEMA


class _Receipt:
    def __init__(self, identity: str) -> None:
        self.content_sha256 = hashlib.sha256(identity.encode()).hexdigest()

    def sha256(self) -> str:
        return hashlib.sha256(("receipt:" + self.content_sha256).encode()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {"content_sha256": self.content_sha256}


class _ResidualController:
    def __init__(self, *, ignore_window: bool = False, forge_critic_hash: bool = False) -> None:
        self.ignore_window = ignore_window
        self.forge_critic_hash = forge_critic_hash
        self.state_sha256 = "3" * 64

    def assert_integrity(self) -> str:
        return self.state_sha256

    def inference_facts(
        self,
        observation: np.ndarray,
        reference_window: np.ndarray,
    ) -> ReferenceResidualInferenceFacts:
        visible_window = np.zeros_like(reference_window) if self.ignore_window else reference_window
        policy_input = np.ascontiguousarray(
            np.concatenate((observation, visible_window.ravel())).astype("<f4")
        )
        input_sha256 = hashlib.sha256(policy_input.tobytes(order="C")).hexdigest()
        actor_output = np.ascontiguousarray(np.tanh(visible_window[0, :17]).astype("<f4"))
        critic_output = np.asarray(
            [[np.float32(np.mean(visible_window, dtype=np.float64))]],
            dtype="<f4",
        )
        return ReferenceResidualInferenceFacts(
            policy_input_sha256=input_sha256,
            policy_input_shape=(708,),
            policy_input_dtype="<f4",
            actor_input_sha256=input_sha256,
            critic_input_sha256=input_sha256,
            controller_state_sha256=self.state_sha256,
            deterministic_action=tuple(float(value) for value in actor_output),
            actor_output_sha256=hashlib.sha256(actor_output.tobytes(order="C")).hexdigest(),
            actor_output_shape=(17,),
            actor_output_dtype="<f4",
            critic_value=float(critic_output[0, 0]),
            critic_output_sha256=(
                "f" * 64
                if self.forge_critic_hash
                else hashlib.sha256(critic_output.tobytes(order="C")).hexdigest()
            ),
            critic_output_shape=(1, 1),
            critic_output_dtype="<f4",
        )


def _projection() -> MinariHumanoidProjection:
    timeline = np.arange(1001, dtype=np.float64)[:, None]
    features = np.arange(45, dtype=np.float64)[None, :]
    reference_values = np.ascontiguousarray(
        0.001 * timeline + 0.01 * features,
        dtype="<f8",
    )
    reference_values[:, 1:5] = np.asarray([1.0, 0.0, 0.0, 0.0])
    mutable_observations = np.zeros((1001, 348), dtype="<f8", order="C")
    mutable_observations[:, HUMANOID_V5_OBSERVATION_TO_REFERENCE_INDICES] = reference_values
    observations = mutable_observations
    observations[:, 45:] = np.arange(303, dtype=np.float64) * 0.0001
    reference = ReferenceArtifact.create(
        artifact_id="minari/mujoco/humanoid/expert-v0/episode-0/projected-45d/v1",
        schema=HUMANOID_REFERENCE_SCHEMA,
        values=reference_values,
    )
    observations = np.frombuffer(observations.tobytes(order="C"), dtype="<f8").reshape(
        observations.shape
    )
    receipt = MinariHumanoidProjectionReceipt(
        source_commit="a" * 40,
        source_provenance_class="synthetic_test_fixture",
        source_origin_verified=False,
        source_record_sha256=None,
        source_record_bytes=None,
        hdf5_sha256="4" * 64,
        hdf5_bytes=1,
        metadata_sha256="5" * 64,
        metadata_bytes=1,
        minari_version="fixture",
        declared_requirements=(),
        episode_id=0,
        episode_seed=123,
        episode_total_steps=1000,
        observations_sha256=minari_source._array_sha256(observations),
        actions_sha256="6" * 64,
        rewards_sha256="7" * 64,
        terminations_sha256="8" * 64,
        truncations_sha256="9" * 64,
        ignored_observation_fields_sha256=minari_source._array_sha256(observations[:, 45:]),
        projection_mapping_sha256=minari_source._projection_mapping_sha256(),
        reference_artifact_id=reference.identity.artifact_id,
        reference_content_sha256=reference.identity.content_sha256,
        reference_schema_sha256=reference.identity.schema_sha256,
        reference_frames=reference.identity.n_frames,
    )
    return MinariHumanoidProjection(
        episode_observations=observations,
        reference=reference,
        receipt=receipt,
    )


def _base_controller(_observation: np.ndarray) -> np.ndarray:
    return np.linspace(-0.2, 0.2, 17, dtype="<f4")


def test_probe_holds_observation_and_target_fixed_while_reference_input_changes() -> None:
    report = build_reference_causal_probe_report(
        projection=_projection(),  # type: ignore[arg-type]
        base_controller=_base_controller,
        residual_controller=_ResidualController(),
        base_receipt=_Receipt("base"),  # type: ignore[arg-type]
        residual_receipt=_Receipt("residual"),  # type: ignore[arg-type]
    )

    assert report["evidence_class"] == "exploratory"
    assert report["formal_oracle_comparison_authorized"] is False
    assert report["tracker_behavior_or_stability_established"] is False
    assert report["oracle_quality_established"] is False
    assert len(report["records"]) == len(SNAPSHOT_FRAMES) * len(CONDITION_IDS)
    assert report["summary"]["mechanistic_gate_passed"] is True  # type: ignore[index]

    for frame in SNAPSHOT_FRAMES:
        block = [record for record in report["records"] if record["snapshot_frame"] == frame]
        assert [record["condition_id"] for record in block] == list(CONDITION_IDS)
        assert len({record["observation_sha256"] for record in block}) == 1
        assert len({record["ground_truth_window_sha256"] for record in block}) == 1
        assert len({record["base_action_sha256"] for record in block}) == 1
        assert len({record["policy_input_sha256"] for record in block}) == 4
        assert all(
            record["actor_input_sha256"] == record["policy_input_sha256"] for record in block
        )
        assert all(
            record["critic_input_sha256"] == record["policy_input_sha256"] for record in block
        )

    conditions = report["summary"]["conditions"]  # type: ignore[index]
    for condition_id in CONDITION_IDS[1:]:
        assert conditions[condition_id]["policy_input_changed_count_vs_exact"] == 8
        assert conditions[condition_id]["actor_output_changed_count_vs_exact"] == 8
        assert conditions[condition_id]["composed_action_changed_count_vs_exact"] == 8


def test_probe_rejects_controller_that_ignores_reference_window() -> None:
    with pytest.raises(ExperimentContractError, match="duplicate policy-input"):
        build_reference_causal_probe_report(
            projection=_projection(),  # type: ignore[arg-type]
            base_controller=_base_controller,
            residual_controller=_ResidualController(ignore_window=True),
            base_receipt=_Receipt("base"),  # type: ignore[arg-type]
            residual_receipt=_Receipt("residual"),  # type: ignore[arg-type]
        )


def test_probe_rejects_non_float32_base_action() -> None:
    def wrong_dtype(_observation: np.ndarray) -> np.ndarray:
        return np.zeros(17, dtype="<f8")

    with pytest.raises(ExperimentContractError, match="exact float32"):
        build_reference_causal_probe_report(
            projection=_projection(),  # type: ignore[arg-type]
            base_controller=wrong_dtype,
            residual_controller=_ResidualController(),
            base_receipt=_Receipt("base"),  # type: ignore[arg-type]
            residual_receipt=_Receipt("residual"),  # type: ignore[arg-type]
        )


def test_probe_rejects_projection_receipt_that_does_not_bind_observations() -> None:
    projection = _projection()
    tampered = replace(
        projection,
        receipt=replace(projection.receipt, observations_sha256="0" * 64),
    )

    with pytest.raises(MinariHumanoidImportError, match="observation hash mismatch"):
        build_reference_causal_probe_report(
            projection=tampered,
            base_controller=_base_controller,
            residual_controller=_ResidualController(),
            base_receipt=_Receipt("base"),  # type: ignore[arg-type]
            residual_receipt=_Receipt("residual"),  # type: ignore[arg-type]
        )


def test_probe_rejects_forged_critic_output_hash() -> None:
    with pytest.raises(ExperimentContractError, match="critic output SHA-256"):
        build_reference_causal_probe_report(
            projection=_projection(),
            base_controller=_base_controller,
            residual_controller=_ResidualController(forge_critic_hash=True),
            base_receipt=_Receipt("base"),  # type: ignore[arg-type]
            residual_receipt=_Receipt("residual"),  # type: ignore[arg-type]
        )
