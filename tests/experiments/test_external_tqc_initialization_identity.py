from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.experiments import (
    tqc_actor_equivalence_v2,
    tqc_development_persistence_v2,
    tqc_development_training_v2,
)
from oracle_composition.experiments.external_tqc_initialization_identity import (
    EXPERT_ACTOR_NPZ_SHA256,
    initialize_expanded_actor_parameters,
    initialize_zero_residual_mean_parameters,
    run_external_initialization_identity,
    validate_external_initialization_identity_receipt,
    verify_expanded_actor_parameters,
    verify_zero_residual_mean_parameters,
)
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_actor_npz import load_actor_npz
from oracle_composition.experiments.tqc_initialization_identity_contract import (
    load_initialization_identity_design,
)

REPOSITORY = Path(__file__).resolve().parents[2]
DESIGN = (
    REPOSITORY / "experiments/bootstrap_tqc_humanoid/configs/"
    "tqc_initialization_transfer_fixture_v0.study.json"
)
ACTOR = REPOSITORY / "artifacts/bootstrap_tqc_humanoid/farama_minari_humanoid_v5_tqc_actor_v1.npz"
IMPORT_RECEIPT = (
    REPOSITORY / "artifacts/bootstrap_tqc_humanoid/external_actor_import_expert_03a3_v1.json"
)
EQUIVALENCE_RECEIPT = (
    REPOSITORY / "artifacts/bootstrap_tqc_humanoid/external_actor_equivalence_expert_03a3_v1.json"
)
RECORDED_E1 = (
    REPOSITORY / "artifacts/bootstrap_tqc_humanoid/e1_initialization_identity_external_v1.json"
)


@pytest.fixture(scope="module")
def actual_e1(tmp_path_factory: pytest.TempPathFactory) -> object:
    required = (ACTOR, IMPORT_RECEIPT, EQUIVALENCE_RECEIPT)
    if not all(path.is_file() for path in required):
        pytest.skip("ignored external expert E1 chain is not installed")
    output = tmp_path_factory.mktemp("actual-external-e1") / "e1.json"
    return run_external_initialization_identity(
        design_path=DESIGN,
        actor_npz_path=ACTOR,
        import_receipt_path=IMPORT_RECEIPT,
        equivalence_receipt_path=EQUIVALENCE_RECEIPT,
        output_path=output,
    )


def test_actual_e1_is_bitwise_on_external_expert_and_not_training_authority(
    actual_e1: object,
) -> None:
    receipt = validate_external_initialization_identity_receipt(actual_e1.receipt)

    assert receipt["authority"] == "external_pretrained_artifact"
    assert receipt["artifact_class"] == "external_pretrained_artifact"
    assert receipt["evidence_class"] == "external_base_import"
    assert receipt["strict_actor_npz"]["sha256"] == EXPERT_ACTOR_NPZ_SHA256
    assert receipt["fixture_design"]["sha256"] == (
        "fe0565f2986b70732d1e054fda22c013b80f439a87f8e95cdded6977d438e3e6"
    )
    assert receipt["observation_set"]["sha256"] == (
        "0c6a81b06a88cab7eca0255e75f021008b60025c4ddc4d3719426e3647159ec6"
    )
    assert receipt["zero_residual_contender"]["seeded_residual_sample_nonzero"] is True
    assert receipt["zero_residual_contender"]["bitwise_identity_passed"] is True
    assert receipt["expanded_tqc_contender"]["bitwise_identity_passed"] is True
    assert all(
        pair["base_sha256"] == pair["contender_sha256"] and pair["bitwise_equal"] is True
        for contender in ("zero_residual_contender", "expanded_tqc_contender")
        for pair in receipt[contender]["paired_output_sha256"].values()
    )
    assert (
        receipt["expanded_tqc_contender"]["parameter_audit"]["every_copied_parameter_bitwise_equal"]
        is True
    )
    assert receipt["eligible_for_local_training"] is False
    assert receipt["training_steps"] == 0
    assert receipt["environment_steps"] == 0
    assert receipt["behavior_evaluated"] is False
    if RECORDED_E1.is_file():
        assert actual_e1.published.path.read_bytes() == RECORDED_E1.read_bytes()

    local_apis = (
        tqc_actor_equivalence_v2.admit_final_tqc_checkpoint_actor,
        tqc_actor_equivalence_v2.revalidate_tqc_actor_equivalence_receipt_v2,
        tqc_development_persistence_v2.revalidate_tqc_persistence_authority_v2,
        tqc_development_training_v2.revalidate_tqc_training_completion_v2,
    )
    for candidate in (receipt, actual_e1):
        for local_api in local_apis:
            with pytest.raises(ExperimentContractError):
                local_api(candidate)


def test_nonzero_residual_mean_parameter_fails_identity() -> None:
    design = load_initialization_identity_design(DESIGN).design
    parameters = initialize_zero_residual_mean_parameters(design)
    parameters["mean.weight"][0, 0] = np.float32(1.0)

    with pytest.raises(ExperimentContractError, match="not exact positive zero"):
        verify_zero_residual_mean_parameters(parameters, design)


@pytest.mark.parametrize("mutation", ["reference_column", "copied_parameter"])
def test_expanded_actor_rejects_nonzero_column_or_changed_copy(mutation: str) -> None:
    if not ACTOR.is_file():
        pytest.skip("ignored strict external expert NPZ is not installed")
    design = load_initialization_identity_design(DESIGN).design
    actor = load_actor_npz(ACTOR, expected_sha256=EXPERT_ACTOR_NPZ_SHA256)
    expanded = initialize_expanded_actor_parameters(actor.arrays, design)
    if mutation == "reference_column":
        expanded["latent_pi.0.weight"][0, 348] = np.float32(1.0)
        message = "reference column is nonzero"
    else:
        expanded["mu.bias"][0] = np.nextafter(
            expanded["mu.bias"][0],
            np.float32(np.inf),
            dtype=np.float32,
        )
        message = "copied parameter mu.bias differs"

    with pytest.raises(ExperimentContractError, match=message):
        verify_expanded_actor_parameters(actor.arrays, expanded, design)


@pytest.mark.parametrize("missing", ["observation_set", "strict_actor_npz"])
def test_actual_e1_receipt_rejects_missing_observation_or_npz_hash(
    actual_e1: object,
    missing: str,
) -> None:
    receipt = copy.deepcopy(actual_e1.receipt)
    receipt[missing].pop("sha256")

    with pytest.raises(ExperimentContractError, match=r"observation-set|NPZ"):
        validate_external_initialization_identity_receipt(receipt)


def test_recorded_actual_e1_receipt_is_valid() -> None:
    if not RECORDED_E1.is_file():
        pytest.skip("ignored actual E1 receipt is not installed")
    receipt = json.loads(RECORDED_E1.read_text(encoding="utf-8"))
    assert validate_external_initialization_identity_receipt(receipt)["passed"] is True
