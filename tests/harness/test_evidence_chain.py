from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.harness import evidence as evidence_module
from oracle_composition.harness.evaluator import (
    CycleEvaluationError,
    EvaluationDependencies,
    evaluate_cycle,
    render_legacy_report_markdown,
)
from oracle_composition.harness.evidence import (
    EvidenceChainError,
    current_authority_identities,
    execution_manifest_value,
    legacy_phase_a_artifacts,
    load_e003_execution,
    validate_designer_provenance,
    validate_scientific_receipt,
)
from oracle_composition.harness.inputs import load_frozen_inputs

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/003_composition_speed_profile"

FROZEN_ARTIFACT_SHA256 = {
    "arms/00_single_fast.json": "f00e68986d90dc295ee34203b4fbb04624a8fb90082c5cd382b46d93f43fa23a",
    "arms/01_single_slow.json": "2cde9fb518d690c42fcd7aa221390518a8de147620e51986681efecad2913d12",
    "arms/02_playback.json": "fdc7f6ab607d5282fd42a87a2a6d08601bf6da821dd212bbe22d04fa0b3ea64d",
    "arms/03_handwritten.json": "4368848952e43949ed5613352992d0db2dd61026cdbb78a7616c4b241e30d5a5",
    "cycles/cycle_0/report_0.json": "3d32ddcc3869f9a05df9beb1bc1f7816996bf3bda166eb4ca26f3ec9556aa623",
    "cycles/cycle_1/oracle_1.json": "32bfc555ffc578d4cf8f75823acc1ba98db6621b2bcd0204f0725b0bf70af029",
    "cycles/cycle_1/report_1.json": "480d5b3521ce66593e0258f3f42904f969c72b83141b330bc7ff4759144c25f0",
    "cycles/cycle_2/oracle_2.json": "f1cbc2784783d4f34312b036a911c536f8406aa3da1ffcf52f35f4927808173a",
    "cycles/cycle_2/report_2.json": "e677b9f3c3daabdb13648a18981c39b47bc74fa2e6908e7fe1af9d73674e194c",
    "library_manifest_v1.json": "ad57578dc2ed4fe3707da74a4f86620e758b1aa30064016785d187a5e86d3207",
    "phase_b/oracle_cycle_1_reference_v1.json": (
        "4d24f22780360d7632235572d97b3fccfe7c376162e69e15082f77bd7afcccf1"
    ),
    "task_spec_v1.json": "edb2cffde9f2eb087667d182f3da199ef6956aeadc6d45e9e218649d6fa9cbd7",
}

TRACE_INDEX_SHA256 = {
    0: "041a81cac8d77d717aacb31e6e633fcae2a47e09457fc92eade79dc103e7f983",
    1: "4d2c419140b71d1cc13f04c6af802d9fb569aaf690621cc17cd99ab5f4057097",
    2: "a210e04509a1ac2266981c1ea0af1407ce5059d72d633d6ba62daf9970bdf775",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase_a_json_oracle_task_and_trace_bytes_remain_frozen() -> None:
    for relative, expected in FROZEN_ARTIFACT_SHA256.items():
        assert _sha256(EXPERIMENT / relative) == expected
    for cycle, expected in TRACE_INDEX_SHA256.items():
        index_path = ROOT / f"artifacts/experiments_003/cycle_{cycle}/content_index.json"
        assert _sha256(index_path) == expected
        index = json.loads(index_path.read_bytes())
        assert index["entry_count"] == len(index["entries"])
        for entry in index["entries"]:
            trace_path = index_path.parent.parent / entry["path"]
            trace_bytes = trace_path.read_bytes()
            assert len(trace_bytes) == entry["byte_count"]
            assert hashlib.sha256(trace_bytes).hexdigest() == entry["sha256"]


def test_phase_a_receipts_and_corrected_markdown_are_reproducible() -> None:
    generated = legacy_phase_a_artifacts(experiment=EXPERIMENT, repository_root=ROOT)
    for path, expected_bytes in generated.items():
        assert path.read_bytes() == expected_bytes
    library, task = load_frozen_inputs(EXPERIMENT)
    for cycle in range(3):
        report = json.loads((EXPERIMENT / f"cycles/cycle_{cycle}/report_{cycle}.json").read_bytes())
        rendered = render_legacy_report_markdown(
            report,
            task=task,
            behavior_names=library.behavior_names,
        )
        assert rendered == (EXPERIMENT / f"cycles/cycle_{cycle}/report_{cycle}.md").read_bytes()
        if cycle > 0 and all(
            row["slow_third_behavior_fractions"]["simple"] == 0.0 for row in report["per_episode"]
        ):
            assert b"task not completed: slow segment absent\n\n|" in rendered

    sealed = load_e003_execution(EXPERIMENT, ROOT)
    receipt = validate_scientific_receipt(
        EXPERIMENT / "cycles/cycle_2/scientific_receipt_v2.json",
        experiment=EXPERIMENT,
        repository_root=ROOT,
        library=sealed.library,
        task=sealed.task,
        expected_cycle=2,
        expected_metric_core_sha256=None,
    )
    assert len(receipt.arms) == 6


@pytest.mark.parametrize(
    ("source_suffix", "changed_authority"),
    [
        ("harness/executor.py", "metric_core"),
        ("harness/contract.py", "oracle_interpreter"),
        ("harness/cycle_cli.py", "report_writer"),
        ("harness/evidence.py", "report_writer"),
        ("harness/evaluator.py", "report_writer"),
        ("contracts/reference_identity_v2.py", "report_writer"),
        ("experiments/tqc_actor_npz.py", "archive_loader"),
        ("harness/inputs.py", "archive_loader"),
        ("sources/strict_tqc_actor_runtime.py", "archive_loader"),
    ],
)
def test_authority_source_changes_affect_only_their_declared_digest(
    monkeypatch: pytest.MonkeyPatch, source_suffix: str, changed_authority: str
) -> None:
    baseline = current_authority_identities()
    original = evidence_module.sha256_file

    def changed(path: Path) -> str:
        digest = original(path)
        return "0" * 64 if path.as_posix().endswith(source_suffix) else digest

    monkeypatch.setattr(evidence_module, "sha256_file", changed)
    observed = current_authority_identities()
    changed_keys = {
        name for name in baseline if baseline[name]["sha256"] != observed[name]["sha256"]
    }
    assert changed_keys == {changed_authority}


@pytest.mark.parametrize("cycle", [1, 2])
def test_designer_provenance_binds_sanitized_audit_and_copied_oracle(cycle: int) -> None:
    path = EXPERIMENT / f"cycles/cycle_{cycle}/designer_provenance.json"
    encoded = path.read_bytes()
    receipt = json.loads(encoded)
    assert encoded == canonical_json_bytes(receipt)
    assert receipt["evidence_class"] == "audit_log_only"
    assert receipt["isolation_property"] == "audit_log_only_not_os_enforced"
    assert receipt["requested"] == {
        "model": "gpt-5.6-sol",
        "reasoning_effort": "max",
        "source": "requested_identity_only",
    }
    assert receipt["status"] == "SUCCEEDED"
    copied = ROOT / receipt["canonical_oracle"]["copied_file_path"]
    copied_bytes = copied.read_bytes()
    assert (
        hashlib.sha256(copied_bytes).hexdigest()
        == receipt["canonical_oracle"]["copied_file_sha256"]
    )
    assert (
        hashlib.sha256(canonical_json_bytes(json.loads(copied_bytes))).hexdigest()
        == (receipt["canonical_oracle"]["canonical_sha256"])
    )
    assert receipt["canonical_oracle"]["copied_semantic_hash_match"] is True
    source = ROOT / receipt["canonical_oracle"]["source_file_path"]
    assert source.read_bytes() == copied_bytes
    assert _sha256(source) == receipt["canonical_oracle"]["source_file_sha256"]
    for artifact in receipt["run_artifacts"].values():
        source = ROOT / artifact["path"]
        if source.exists():
            assert _sha256(source) == artifact["sha256"]
    for allowed in receipt["allowed_inputs"]:
        if allowed["sha256_source"] == "current_immutable_file":
            assert _sha256(ROOT / allowed["path"]) == allowed["sha256"]


def test_designer_provenance_is_explicitly_unverifiable_without_raw_audit(
    tmp_path: Path,
) -> None:
    relative_experiment = Path("experiments/003_composition_speed_profile")
    portable_experiment = tmp_path / relative_experiment
    copied_relatives = (
        "cycles/cycle_1/designer_prompt.md",
        "cycles/cycle_1/designer_provenance.json",
        "cycles/cycle_1/oracle_1.json",
    )
    for relative in copied_relatives:
        source = EXPERIMENT / relative
        destination = portable_experiment / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    receipt = json.loads(
        (portable_experiment / "cycles/cycle_1/designer_provenance.json").read_bytes()
    )
    with pytest.raises(EvidenceChainError, match="unverifiable"):
        validate_designer_provenance(
            experiment=portable_experiment,
            repository_root=tmp_path,
            cycle=1,
            oracle_id="cycle_1_candidate",
            oracle_file_sha256=receipt["canonical_oracle"]["copied_file_sha256"],
            oracle_sha256=receipt["canonical_oracle"]["canonical_sha256"],
        )


def _mutated_experiment(tmp_path: Path, mutation: str) -> Path:
    experiment = tmp_path / "experiment"
    experiment.mkdir(parents=True)
    library = json.loads((EXPERIMENT / "library_manifest_v1.json").read_bytes())
    task = json.loads((EXPERIMENT / "task_spec_v1.json").read_bytes())
    if mutation == "behaviors":
        library["behaviors"][2]["name"] = "slow"
        task["slow_selection"]["chosen_behavior"] = "slow"
        task["slow_selection"]["candidates"][1]["behavior"] = "slow"
    elif mutation == "seeds":
        task["evaluation"]["seeds"][-1] += 1000
    elif mutation == "horizon":
        task["evaluation"]["episode_steps"] = 999
        task["target_schedule"][-1]["stop"] = 999
    elif mutation == "schedule":
        task["target_schedule"][0]["stop"] = 299
        task["target_schedule"][1]["start"] = 299
    elif mutation == "task_text":
        task["task_text"] += " Altered."
    else:
        library["behaviors"][0]["speed_m_s"]["q3_m_s"] += 0.01
        library["behaviors"][0]["speed_m_s"]["iqr_m_s"] += 0.01
    library_bytes = canonical_json_bytes(library)
    (experiment / "library_manifest_v1.json").write_bytes(library_bytes)
    task["library_manifest_sha256"] = hashlib.sha256(library_bytes).hexdigest()
    (experiment / "task_spec_v1.json").write_bytes(canonical_json_bytes(task))
    loaded_library, loaded_task = load_frozen_inputs(experiment)
    (experiment / "execution_manifest_v1.json").write_bytes(
        canonical_json_bytes(execution_manifest_value(loaded_library, loaded_task))
    )
    return experiment


@pytest.mark.parametrize(
    "mutation",
    ["behaviors", "seeds", "horizon", "schedule", "task_text", "library_statistic"],
)
def test_self_consistent_frozen_field_changes_fail_before_runtime_creation(
    tmp_path: Path, mutation: str
) -> None:
    experiment = _mutated_experiment(tmp_path, mutation)
    calls = {"actor": 0, "environment": 0}

    def actor_loader(_entry: object, _root: Path) -> object:
        calls["actor"] += 1
        raise AssertionError("actor loader crossed the execution seal")

    def environment_factory() -> object:
        calls["environment"] += 1
        raise AssertionError("environment factory crossed the execution seal")

    with pytest.raises(CycleEvaluationError, match="E003 execution seal"):
        evaluate_cycle(
            experiment=experiment,
            cycle=0,
            oracle_paths=[EXPERIMENT / "arms/00_single_fast.json"],
            repository_root=ROOT,
            dependencies=EvaluationDependencies(
                environment_factory=environment_factory,
                actor_loader=actor_loader,
            ),
        )
    assert calls == {"actor": 0, "environment": 0}
