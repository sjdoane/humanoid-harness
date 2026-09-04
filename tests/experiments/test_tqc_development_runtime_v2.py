from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import pytest

from oracle_composition.experiments import tqc_development_runtime_v2 as runtime_module
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_development_contract_v2 import (
    load_tqc_development_design_v2,
)
from oracle_composition.experiments.tqc_development_runtime_v2 import (
    TQCHostRuntimeReceiptV2,
    inspect_tqc_development_runtime_v2,
    validate_tqc_development_runtime_receipt_v2,
)

ROOT = Path(__file__).parents[2]
CONFIG_ROOT = ROOT / "experiments/bootstrap_tqc_humanoid/configs"


def _design():
    return load_tqc_development_design_v2(
        CONFIG_ROOT / "tqc_base_controller_dev_1m_v2.study.json",
        CONFIG_ROOT / "tqc_e0_reuse_dev_1m_v2.audit.json",
    )


def _binding() -> dict[str, object]:
    return {
        "attempt_id": runtime_module.ATTEMPT_ID,
        "attempt_nonce": "a" * 64,
        "worker_pid": os.getpid(),
        "worker_process_start_monotonic_seconds": 1.0,
        "preflight_contract_sha256": "b" * 64,
        "claimed_work_directory_identity": "c" * 64,
        "project_python_source_tree_sha256": runtime_module.source_tree_sha256(),
    }


def _inspect():
    return inspect_tqc_development_runtime_v2(_design(), receipt_binding=_binding())


def _readmit(encoded: bytes, design: object | None = None):
    return validate_tqc_development_runtime_receipt_v2(
        encoded,
        _design() if design is None else design,
        expected_binding=_binding(),
    )


@pytest.mark.gym
def test_runtime_reinspection_matches_every_frozen_requirement() -> None:
    receipt = _inspect()
    value = receipt.to_dict()

    assert value["runtime_receipt_id"] == "tqc_dev_1m_v2_host_runtime/v1"
    assert (
        value["observed_runtime_requirements"]
        == _design().to_dict()["training_projection"]["runtime_requirements"]
    )
    assert value["observed_simulator"] == _design().to_dict()["training_projection"]["simulator"]
    assert value["reward_or_info_fields_read"] is False
    assert value["behavioral_evidence"] is False
    assert {name: value[name] for name in _binding()} == _binding()
    assert receipt.authorizes_execution is False


@pytest.mark.gym
def test_runtime_receipt_cannot_be_publicly_forged() -> None:
    receipt = _inspect()

    with pytest.raises(ExperimentContractError, match="only be issued"):
        TQCHostRuntimeReceiptV2(**dataclasses.asdict(receipt))


@pytest.mark.gym
def test_runtime_receipt_bytes_are_strictly_readmitted() -> None:
    design = _design()
    receipt = inspect_tqc_development_runtime_v2(design, receipt_binding=_binding())

    assert _readmit(receipt.canonical_bytes, design) == receipt


@pytest.mark.gym
def test_runtime_receipt_readmission_rejects_duplicate_json_key() -> None:
    design = _design()
    receipt = inspect_tqc_development_runtime_v2(design, receipt_binding=_binding())
    payload = json.loads(receipt.canonical_bytes)
    first_key = next(iter(payload))
    duplicate = (
        b"{" + json.dumps(first_key).encode("utf-8") + b":null," + receipt.canonical_bytes[1:]
    )

    with pytest.raises(ExperimentContractError, match="duplicate JSON key"):
        _readmit(duplicate, design)


@pytest.mark.gym
def test_runtime_receipt_readmission_rejects_changed_design_binding() -> None:
    design = _design()
    receipt = inspect_tqc_development_runtime_v2(design, receipt_binding=_binding())
    payload = receipt.to_dict()
    payload["design_file_sha256"] = "0" * 64

    with pytest.raises(ExperimentContractError, match="design_file_sha256"):
        _readmit(runtime_module.canonical_json(payload), design)


@pytest.mark.gym
def test_runtime_reinspection_rejects_one_changed_source_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = runtime_module.module_sha256

    def changed(module: object) -> str:
        value = original(module)
        if getattr(module, "__name__", "") == "gymnasium.envs.registration":
            return "0" * 64
        return value

    monkeypatch.setattr(runtime_module, "module_sha256", changed)

    with pytest.raises(ExperimentContractError, match="gym_registration_source_sha256"):
        _inspect()


@pytest.mark.gym
def test_runtime_receipt_rejects_attempt_replay_and_fabricated_reset_hash() -> None:
    receipt = _inspect()
    replayed_binding = {**_binding(), "attempt_nonce": "d" * 64}
    with pytest.raises(ExperimentContractError, match="preflight binding"):
        validate_tqc_development_runtime_receipt_v2(
            receipt.canonical_bytes,
            _design(),
            expected_binding=replayed_binding,
        )

    value = receipt.to_dict()
    value["observed_environment"]["reset_observation_sha256"] = "0" * 64
    with pytest.raises(ExperimentContractError, match="reset observation"):
        _readmit(runtime_module.canonical_json(value))


@pytest.mark.gym
def test_runtime_receipt_rejects_nested_numeric_type_changes() -> None:
    receipt = _inspect()
    value = receipt.to_dict()
    value["observed_simulator"]["nq"] = 24.0

    with pytest.raises(ExperimentContractError, match=r"observed simulator\.nq"):
        _readmit(runtime_module.canonical_json(value))


@pytest.mark.gym
def test_runtime_inspection_rejects_nonself_worker_binding() -> None:
    binding = {**_binding(), "worker_pid": os.getpid() + 100_000}

    with pytest.raises(ExperimentContractError, match="not this process"):
        inspect_tqc_development_runtime_v2(_design(), receipt_binding=binding)


@pytest.mark.gym
def test_runtime_inspection_detects_source_tree_change_during_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    binding = _binding()
    monkeypatch.setattr(runtime_module, "source_tree_sha256", lambda: "0" * 64)

    with pytest.raises(ExperimentContractError, match="project Python source tree"):
        inspect_tqc_development_runtime_v2(_design(), receipt_binding=binding)
