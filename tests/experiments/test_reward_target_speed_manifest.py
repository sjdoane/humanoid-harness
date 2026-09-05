from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from oracle_composition.experiments.reward_target_speed_manifest import (
    EVALUATION_SEEDS,
    EXPECTED_PACKAGE_VERSIONS,
    FROZEN_STUDY_CONFIG_V1,
    NO_LEARNING_SMOKE_STEPS,
    RUNTIME_SOURCE_ROLES,
    SMOKE_COMPARISON_FIELDS,
    TRAINING_SEEDS,
    FrozenSourceBindingV1,
    RewardTargetSpeedExecutionManifestV1,
    TargetSpeedManifestError,
    inspect_target_speed_runtime_v1,
    load_target_speed_study_design,
    validate_complete_result_schedule_v1,
)
from oracle_composition.rewards.contract import (
    ACTUATOR_QVEL_INDICES_BY_ACTION_V1,
    RewardArtifactIdentityV1,
)
from oracle_composition.rewards.static_validation import statically_validate_task_term_source

ROOT = Path(__file__).parents[2]
FAMILY = ROOT / "experiments" / "family_b_target_speed_v1"
CONFIG = FAMILY / "configs" / "family_b_target_speed_v1.study.json"


@pytest.fixture(scope="module")
def inspected_runtime():
    return inspect_target_speed_runtime_v1()


def _identity(source: bytes, runtime) -> RewardArtifactIdentityV1:
    return RewardArtifactIdentityV1.create(
        candidate_source_bytes=source,
        target_speed_m_s=1.0,
        affine_alpha=1.0,
        affine_beta=0.0,
        compositor_source_sha256=runtime.source_hashes["compositor"],
        gymnasium_source_sha256=runtime.gymnasium_source_sha256,
        humanoid_xml_sha256=runtime.model_sha256,
        dependency_hashes={"uv.lock": runtime.dependency_lock_sha256},
    )


def test_proposed_config_is_exact_immutable_and_candidate_fixtures_validate() -> None:
    design = load_target_speed_study_design(CONFIG)
    assert design.payload == FROZEN_STUDY_CONFIG_V1
    assert design.payload["design_status"] == "proposed"
    assert design.payload["evidence_label"] == "exploratory_reward_cycle"
    assert design.payload["compute_reservation"] == {
        "amount": 150,
        "unit": "CPU minute",
    }
    assert design.payload["b0"] == {
        "training_permitted": False,
        "behavioral_evaluation_permitted": False,
    }
    assert len(design.sha256) == 64
    with pytest.raises(TypeError):
        design.payload["design_status"] = "locked"  # type: ignore[index]

    stock = statically_validate_task_term_source(
        (FAMILY / "candidates" / "stock_r0.py").read_bytes()
    )
    manual = statically_validate_task_term_source(
        (FAMILY / "candidates" / "manual_target_speed_v1.py").read_bytes()
    )
    assert stock.receipt.metadata["CANDIDATE_ID"] == "stock_r0"
    assert manual.receipt.metadata["CANDIDATE_ID"] == "manual_target_speed_v1"
    for candidate in (stock, manual):
        assert candidate.receipt.static_accepted is True
        assert candidate.receipt.dynamic_validation_status == "not_performed"
        assert not hasattr(candidate, "task_term")


@pytest.mark.gym
def test_real_plain_and_instrumented_fixed_control_smoke_matches_every_field(
    inspected_runtime,
) -> None:
    runtime, smoke = inspected_runtime
    assert smoke.compared_steps == NO_LEARNING_SMOKE_STEPS
    assert smoke.comparison_fields == SMOKE_COMPARISON_FIELDS
    assert smoke.plain_trace_sha256 == smoke.instrumented_trace_sha256
    assert smoke.passed is True
    assert smoke.behavioral_evidence is False
    assert runtime.instrumentation_smoke_sha256 == smoke.sha256


@pytest.mark.gym
def test_runtime_fingerprint_matches_recipe_spaces_gears_and_source_closure(
    inspected_runtime,
) -> None:
    runtime, _smoke = inspected_runtime
    assert dict(runtime.package_versions) == dict(EXPECTED_PACKAGE_VERSIONS)
    assert runtime.observation_shape == (348,)
    assert runtime.action_shape == (17,)
    assert runtime.qpos_shape == (24,)
    assert runtime.qvel_shape == (23,)
    assert runtime.control_period_s == 0.015
    assert runtime.physics_timestep_s == 0.003
    assert runtime.actuator_qvel_indices_by_action == ACTUATOR_QVEL_INDICES_BY_ACTION_V1
    assert tuple(runtime.source_hashes) == tuple(sorted(RUNTIME_SOURCE_ROLES))
    assert all(len(value) == 64 for value in runtime.source_hashes.values())
    assert len(runtime.observation_space_sha256) == 64
    assert len(runtime.action_space_sha256) == 64
    assert len(runtime.actuator_gear_sha256) == 64

    changed = dict(runtime.source_hashes)
    changed.pop("protected_endpoint")
    with pytest.raises(TargetSpeedManifestError, match="source hash closure"):
        dataclasses.replace(runtime, source_hashes=changed)


@pytest.mark.gym
def test_recorded_no_learning_runtime_receipt_replays_exactly(inspected_runtime) -> None:
    runtime, smoke = inspected_runtime
    recorded = json.loads(
        (FAMILY / "receipts" / "builder_runtime_no_learning_smoke.json").read_text(encoding="utf-8")
    )
    assert recorded["training_run"] is False
    assert recorded["behavioral_evaluation"] is False
    assert recorded["runtime_fingerprint"] == runtime.to_dict()
    assert recorded["runtime_sha256"] == runtime.sha256
    assert recorded["no_learning_smoke"] == smoke.to_dict()
    assert recorded["no_learning_smoke_sha256"] == smoke.sha256


def test_config_symlink_duplicate_keys_and_design_drift_fail_closed(tmp_path: Path) -> None:
    link = tmp_path / "config-link.json"
    link.symlink_to(CONFIG)
    with pytest.raises(TargetSpeedManifestError, match="non-symlink"):
        load_target_speed_study_design(link)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(TargetSpeedManifestError, match="duplicate"):
        load_target_speed_study_design(duplicate)

    drifted = tmp_path / "drifted.json"
    drifted.write_text('{"schema_version":1}', encoding="utf-8")
    with pytest.raises(TargetSpeedManifestError, match="differs"):
        load_target_speed_study_design(drifted)


def test_frozen_binding_rejects_post_freeze_mutation_and_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "candidate.py"
    source.write_bytes(b"def task_term(x):\n    return 0.0\n")
    binding = FrozenSourceBindingV1.capture(role="candidate_source:stock_r0", path=source)
    binding.verify()
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(TargetSpeedManifestError, match="mutated"):
        binding.verify()

    linked = tmp_path / "candidate-link.py"
    linked.symlink_to(source)
    with pytest.raises(TargetSpeedManifestError, match="non-symlink"):
        FrozenSourceBindingV1.capture(role="candidate_source:stock_r0", path=linked)


@pytest.mark.gym
def test_manifest_binds_design_runtime_sources_and_has_no_launch_authority(
    tmp_path: Path,
    inspected_runtime,
) -> None:
    runtime, _smoke = inspected_runtime
    design = load_target_speed_study_design(CONFIG)
    source = b"def task_term(x):\n    return 1.25 * x.com_x_velocity_m_s\n"
    paths = {}
    for name in ("protocol", "decision_rule"):
        path = tmp_path / (name.replace(":", "-") + ".txt")
        path.write_bytes(source + name.encode("utf-8"))
        paths[name] = path
    paths["study_config"] = CONFIG
    candidate_path = tmp_path / "candidate.py"
    candidate_path.write_bytes(source)
    paths["candidate_source:stock_r0"] = candidate_path
    bindings = tuple(
        FrozenSourceBindingV1.capture(role=name, path=path) for name, path in paths.items()
    )
    manifest = RewardTargetSpeedExecutionManifestV1(
        status="frozen",
        design_sha256=design.sha256,
        runtime_sha256=runtime.sha256,
        reward_artifacts={"stock_r0": _identity(source, runtime)},
        source_bindings=bindings,
        training_seeds=TRAINING_SEEDS,
        evaluation_seeds=EVALUATION_SEEDS,
    )
    manifest.verify(design=design, runtime=runtime)
    assert manifest.automatic_authorization is False
    assert hashlib.sha256(manifest.canonical_bytes).hexdigest() == manifest.sha256

    with pytest.raises(TargetSpeedManifestError, match="runtime drifted"):
        manifest.verify(
            design=design,
            runtime=dataclasses.replace(runtime, model_sha256="f" * 64),
        )
    paths["protocol"].write_bytes(b"changed")
    with pytest.raises(TargetSpeedManifestError, match="mutated"):
        manifest.verify(design=design, runtime=runtime)

    with pytest.raises(TargetSpeedManifestError, match="source closure"):
        dataclasses.replace(
            manifest,
            source_bindings=tuple(
                binding for binding in bindings if binding.role != "candidate_source:stock_r0"
            ),
        )
    with pytest.raises(TargetSpeedManifestError, match="automatically authorize"):
        dataclasses.replace(manifest, automatic_authorization=True)


def test_partial_missing_duplicated_or_reordered_results_fail_closed() -> None:
    complete = {seed: EVALUATION_SEEDS for seed in TRAINING_SEEDS}
    validate_complete_result_schedule_v1(complete)
    with pytest.raises(TargetSpeedManifestError, match="missing"):
        validate_complete_result_schedule_v1(
            {seed: resets for seed, resets in complete.items() if seed != 505}
        )
    for changed in (
        (*EVALUATION_SEEDS[:-1],),
        (*EVALUATION_SEEDS[:-1], EVALUATION_SEEDS[-2]),
        tuple(reversed(EVALUATION_SEEDS)),
    ):
        partial = dict(complete)
        partial[101] = changed
        with pytest.raises(TargetSpeedManifestError, match="partial, duplicated, or reordered"):
            validate_complete_result_schedule_v1(partial)
