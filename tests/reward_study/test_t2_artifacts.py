from __future__ import annotations

import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path

import pytest

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.phase_b.contracts import (
    PhaseBContractError,
    RewardRegistry,
    TargetSpeedRewardSpec,
    load_phase_b_oracle,
    t2_training_design_contract_value,
    validate_training_design,
)
from oracle_composition.phase_b.reference_runtime import (
    ComposedReferenceRuntime,
    tracking_state_from_reference_row,
)
from oracle_composition.reward_search.t2_model_contracts import T2PreDispatchSeal
from oracle_composition.reward_study.execution_manifest import (
    T2_EXECUTION_FINAL_READY_STATUS,
    T2_EXECUTION_LAUNCH_BASE_COMMIT,
    final_ready_t2_execution_manifest_contract_value,
    load_t2_execution_manifest,
    t2_runtime_execution_identity,
    validate_t2_execution_manifest,
)
from oracle_composition.reward_study.final_admission import (
    T2FinalReadyArtifacts,
    admit_t2_candidate_and_reseal,
    admit_t2_study,
)
from oracle_composition.reward_study.pairing import (
    validate_pairing_receipt,
)
from oracle_composition.reward_study.study_manifest import (
    INTEGRATED_PAIRING_RECEIPT_SHA256,
    STUDY_FINAL_READY_STATUS,
    STUDY_STATUS,
    T2StudyManifestError,
    load_t2_study_manifest,
    resolve_t2_reward_binding,
    study_pairing_sha256_from_arm,
    validate_t2_study_manifest,
)
from oracle_composition.reward_study.t2_evaluator import (
    load_evaluator_design,
    load_t2_verified_reference,
)
from oracle_composition.rewards.target_speed_formula import (
    parse_target_speed_formula_recipe,
)

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / "experiments/004_t2_reward_study"
F3_CALL = EXPERIMENT / "f3_call"
PAIRING_RECEIPT = ROOT / "artifacts/experiments_004/t2_pairing_adapter_receipt_v1.json"
T2C1_EXECUTION_COMMIT = "28067291bf43bb315e1702fc66c649310d4a3c97"


def test_protocol_source_hash_ledger_matches_exact_source_bytes() -> None:
    protocol = (EXPERIMENT / "PROTOCOL.md").read_text()
    documented = dict(
        re.findall(
            r"^\| `((?:reward_study|phase_b)/[^`]+\.py)` \| source \| `([0-9a-f]{64})` \|",
            protocol,
            flags=re.MULTILINE,
        )
    )
    expected = {
        "phase_b/protected_metrics.py",
        "phase_b/report_v2.py",
        "reward_study/pairing.py",
        "reward_study/t2_evaluator.py",
        "reward_study/t2_report.py",
    }
    assert set(documented) == expected
    for relative, digest in documented.items():
        source = ROOT / "src/oracle_composition" / relative
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


def _signals(step: int) -> dict[str, object]:
    return {
        "dwell": step,
        "t": step,
        "torso_up": 1.0,
        "v_target": 3.0,
        "v_x": 3.0,
        "x_travelled": step * 0.045,
        "z_root": 1.4,
    }


def test_expert_hold_oracle_is_canonical_and_yields_rows_zero_through_1000() -> None:
    path = EXPERIMENT / "oracle_expert_hold_v1.json"
    program, digest = load_phase_b_oracle(
        path,
        available_behaviors=("expert", "medium", "simple"),
    )
    assert digest == "489b82591cf65034d58d30f921442a84c93c9c98885cb2f5b22dcaf645a77010"
    assert program.program.oracle_id == "expert_hold_v1"
    assert program.program.recovery is None
    assert not program.program.transitions
    verified = load_t2_verified_reference(repository_root=ROOT, evaluation_seed=97001)
    rows = verified.rows
    runtime = ComposedReferenceRuntime(program, {"expert": rows})
    observed = []
    for step in range(1_001):
        frame = runtime.frame(
            state=tracking_state_from_reference_row(rows[step]),
            signals=_signals(step),
        )
        observed.append(frame.policy_window_indices[0])
        assert frame.behavior == "expert"
        assert frame.transfer is None
        assert frame.phase == step
        assert frame.policy_window[0] == pytest.approx(rows[step].astype("<f4"))
        if step < 1_000:
            runtime.advance()
    assert observed == list(range(1_001))
    assert runtime.transfer_logs == ()


def test_final_ready_execution_manifest_is_canonical_and_records_verified_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import oracle_composition.reward_study.execution_manifest as execution_manifest

    monkeypatch.setattr(
        execution_manifest,
        "_observed_clean_execution_commit",
        lambda _root: T2C1_EXECUTION_COMMIT,
    )
    manifest_path = EXPERIMENT / "execution_manifest_t2_v1.json"
    encoded = manifest_path.read_bytes()
    value = json.loads(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    assert encoded == canonical_json_bytes(value)
    assert digest == "1ce2a4751f4e4149012a1cb0bbc88ee5a2fdaa497b616653d11a6146e0bd5eba"
    study = json.loads((EXPERIMENT / "t2_reward_study_expert_hold_v1.json").read_bytes())
    expected_binding = {
        "byte_count": manifest_path.stat().st_size,
        "path": "experiments/004_t2_reward_study/execution_manifest_t2_v1.json",
        "sha256": digest,
    }
    assert study["arms"][0]["execution_manifest"] == expected_binding
    assert study["arms"][1]["execution_manifest"] == expected_binding
    assert value["execution_manifest_schema_id"] == "t2_execution_manifest_v1"
    assert value["repository"] == {
        "execution_commit": T2C1_EXECUTION_COMMIT,
        "execution_commit_verification": "verified_clean_head_at_final_admission",
        "execution_tree_clean_at_admission": True,
        "launch_base_commit": T2_EXECUTION_LAUNCH_BASE_COMMIT,
        "launch_base_semantics": "provenance_only_not_execution_commit",
    }
    assert value["status"] == T2_EXECUTION_FINAL_READY_STATUS
    assert value["tracker"]["external_tracker_checkpoint"] is None
    assert value["normalizers"] == {"observation": None, "reward": None}
    assert load_t2_execution_manifest(manifest_path, repository_root=ROOT)[0] == value
    with pytest.raises(ExperimentContractError, match="requires the v2 descendant-aware re-seal"):
        t2_runtime_execution_identity(value, repository_root=ROOT)


def test_t2_training_design_is_phase_b_validated_without_changing_t1_bytes() -> None:
    t2_bytes = (EXPERIMENT / "training_design_t2_v1.json").read_bytes()
    t2_value = json.loads(t2_bytes)
    assert t2_bytes == canonical_json_bytes(t2_value)
    assert validate_training_design(t2_value) == t2_training_design_contract_value()
    assert hashlib.sha256(t2_bytes).hexdigest() == (
        "84543f08265dae5076697549f25ce69b7e5e87947d4bb6e32b86a7700d24e67e"
    )
    t1_path = ROOT / "experiments/003_composition_speed_profile/phase_b/training_design_v1.json"
    assert hashlib.sha256(t1_path.read_bytes()).hexdigest() == (
        "1d104a52238eee4f5065b4fcc30c285662e26ab06edd847da7ca9e160364c69e"
    )


def test_t2_training_design_refuses_rsi_family_drift() -> None:
    value = copy.deepcopy(t2_training_design_contract_value())
    value["rsi"]["rsi_family"] = "multi_behavior_rsi/v1"
    with pytest.raises(PhaseBContractError, match="semantics differ"):
        validate_training_design(value)


def test_evaluator_design_is_canonical_and_source_bound() -> None:
    value, digest = load_evaluator_design(
        EXPERIMENT / "evaluator_design_t2_v1.json",
        evaluator_source_path=ROOT / "src/oracle_composition/reward_study/t2_evaluator.py",
        protected_metrics_source_path=(
            ROOT / "src/oracle_composition/phase_b/protected_metrics.py"
        ),
        report_v2_source_path=ROOT / "src/oracle_composition/phase_b/report_v2.py",
        report_writer_source_path=ROOT / "src/oracle_composition/reward_study/t2_report.py",
    )
    assert digest == "634ea93e975e5331f9c71c5123a75ca525cc366055312bf4b3a4652dba772708"
    assert value["reward_helpers_imported"] is False
    assert value["calibration"] == "none"


def test_study_manifest_verifies_final_ready_files_rewards_and_pairing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import oracle_composition.reward_study.execution_manifest as execution_manifest

    monkeypatch.setattr(
        execution_manifest,
        "_observed_clean_execution_commit",
        lambda _root: T2C1_EXECUTION_COMMIT,
    )
    path = EXPERIMENT / "t2_reward_study_expert_hold_v1.json"
    value, digest = load_t2_study_manifest(path, repository_root=ROOT)
    assert digest == "2fea955d94719e455920fe65eaf017e746313353fd2246a69190ea4db623f6e5"
    assert value["study_pairing_sha256"] == (
        "ccef84c7f05992556183c9ebdacb86945fe3e7832cd1e90a03eb9544096309a2"
    )
    assert value["integrated_pairing_receipt_sha256"] == INTEGRATED_PAIRING_RECEIPT_SHA256
    assert value["status"] == STUDY_FINAL_READY_STATUS
    assert value["arms"][0]["reward"]["sha256"].startswith("eea2b6a9")
    assert value["arms"][1]["reward"] == {
        "path": "experiments/004_t2_reward_study/candidate_target_speed_t2_v1.json",
        "reward_id": "target_speed_triangular_affine_t2_adapter/v1",
        "sha256": "c09e93dc27f129f0f65ed5be515b114a77673fef95027422f60ddade76dcc123",
    }
    assert (
        value["arms"][0]["pairing_adapter"]["sha256"]
        == hashlib.sha256(
            (ROOT / "src/oracle_composition/reward_study/pairing.py").read_bytes()
        ).hexdigest()
    )


def test_committed_f3_call_ledger_reproduces_and_registry_resolves_candidate() -> None:
    ledger = json.loads((F3_CALL / "ledger_v1.json").read_bytes())
    assert set(ledger) == {
        "call_result",
        "derived_candidate",
        "kind",
        "raw_run_bytes_committed",
        "retained_call_json",
        "schema_version",
    }
    assert ledger["schema_version"] == 1
    assert ledger["kind"] == "t2_f3_call_committed_ledger"
    assert ledger["raw_run_bytes_committed"] is False

    def exact_bytes(binding: dict[str, object]) -> bytes:
        path = ROOT / str(binding["path"])
        encoded = path.read_bytes()
        assert len(encoded) == binding["byte_count"]
        assert hashlib.sha256(encoded).hexdigest() == binding["sha256"]
        return encoded

    exact_bytes(ledger["call_result"])
    proposal_bytes = exact_bytes(ledger["retained_call_json"]["proposal"])
    recipe_bytes = exact_bytes(ledger["retained_call_json"]["recipe"])
    receipt_bytes = exact_bytes(ledger["retained_call_json"]["model_call_receipt"])
    candidate_bytes = exact_bytes(ledger["derived_candidate"])

    recipe = parse_target_speed_formula_recipe(recipe_bytes)
    assert recipe.canonical_bytes == recipe_bytes
    reproduced = TargetSpeedRewardSpec(alpha=recipe.alpha, beta=recipe.beta)
    assert reproduced.canonical_bytes == candidate_bytes
    resolved, candidate_sha256 = RewardRegistry().load(
        ROOT / str(ledger["derived_candidate"]["path"])
    )
    assert resolved == reproduced
    assert candidate_sha256 == ledger["derived_candidate"]["sha256"]

    proposal = json.loads(proposal_bytes)
    receipt = json.loads(receipt_bytes)
    assert proposal["parameters"] == {"alpha": recipe.alpha, "beta": recipe.beta}
    assert receipt["candidate_recipe_sha256"] == ledger["retained_call_json"]["recipe"]["sha256"]
    assert receipt["proposal_sha256"] == ledger["retained_call_json"]["proposal"]["sha256"]
    assert {path.name for path in F3_CALL.iterdir()} == {
        "ledger_v1.json",
        *(Path(str(binding["path"])).name for binding in ledger["retained_call_json"].values()),
    }


def test_final_t2_seal_binds_the_published_final_ready_set() -> None:
    seal_bytes = (EXPERIMENT / "t2_seal_v1.json").read_bytes()
    seal = T2PreDispatchSeal.model_validate_json(seal_bytes)
    assert hashlib.sha256(seal_bytes).hexdigest() == (
        "89aff467d05414553439ac5cfed6b5679d94ae08e455d1b687e3b9d685190624"
    )
    assert seal.dispatch_state == "candidate_admitted_no_further_initial_dispatch"
    assert seal.execution_manifest.sha256 == (
        "1ce2a4751f4e4149012a1cb0bbc88ee5a2fdaa497b616653d11a6146e0bd5eba"
    )
    assert seal.study_manifest.sha256 == (
        "2fea955d94719e455920fe65eaf017e746313353fd2246a69190ea4db623f6e5"
    )
    assert seal.study_pairing_sha256 == (
        "ccef84c7f05992556183c9ebdacb86945fe3e7832cd1e90a03eb9544096309a2"
    )


def test_final_execution_manifest_refuses_wrong_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import oracle_composition.reward_study.execution_manifest as execution_manifest

    expected_commit = "a" * 40
    value = execution_manifest.t2_execution_manifest_contract_value(
        ROOT,
        admission_commit=expected_commit,
    )
    monkeypatch.setattr(
        execution_manifest,
        "_observed_clean_execution_commit",
        lambda _root: "b" * 40,
    )
    monkeypatch.setattr(
        execution_manifest.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 1, b"", b""),
    )
    with pytest.raises(ExperimentContractError, match="not a descendant"):
        validate_t2_execution_manifest(value, repository_root=ROOT)
    with pytest.raises(ExperimentContractError, match="expected admission commit"):
        final_ready_t2_execution_manifest_contract_value(
            ROOT,
            expected_admission_commit=expected_commit,
        )


def test_execution_manifest_accepts_allowlisted_document_descendant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import oracle_composition.reward_study.execution_manifest as execution_manifest

    admission_commit = "a" * 40
    observed_commit = "b" * 40
    value = execution_manifest.t2_execution_manifest_contract_value(
        ROOT,
        admission_commit=admission_commit,
    )
    monkeypatch.setattr(
        execution_manifest,
        "_observed_clean_execution_commit",
        lambda _root: observed_commit,
    )
    monkeypatch.setattr(
        execution_manifest.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, b"", b""),
    )
    monkeypatch.setattr(
        execution_manifest,
        "_descendant_changed_paths",
        lambda *_args, **_kwargs: (
            "README.md",
            "docs/operations/dual-orchestration/T2C2_RESULT.md",
        ),
    )

    assert validate_t2_execution_manifest(value, repository_root=ROOT) == value
    assert t2_runtime_execution_identity(value, repository_root=ROOT) == {
        "admission_commit": admission_commit,
        "execution_commit_observed": observed_commit,
    }


@pytest.mark.parametrize(
    "changed_path",
    (
        "src/oracle_composition/phase_b/training.py",
        "tests/reward_study/test_t2_artifacts.py",
        "uv.lock",
        "experiments/004_t2_reward_study/payloads/episode.npz",
    ),
)
def test_execution_manifest_refuses_non_record_descendant_path(
    monkeypatch: pytest.MonkeyPatch,
    changed_path: str,
) -> None:
    import oracle_composition.reward_study.execution_manifest as execution_manifest

    admission_commit = "a" * 40
    value = execution_manifest.t2_execution_manifest_contract_value(
        ROOT,
        admission_commit=admission_commit,
    )
    monkeypatch.setattr(
        execution_manifest,
        "_observed_clean_execution_commit",
        lambda _root: "b" * 40,
    )
    monkeypatch.setattr(
        execution_manifest.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, b"", b""),
    )
    monkeypatch.setattr(
        execution_manifest,
        "_descendant_changed_paths",
        lambda *_args, **_kwargs: (changed_path,),
    )

    with pytest.raises(ExperimentContractError, match="non-admission path"):
        validate_t2_execution_manifest(value, repository_root=ROOT)


def test_execution_manifest_refuses_changed_binding_on_allowed_descendant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import oracle_composition.reward_study.execution_manifest as execution_manifest

    admission_commit = "a" * 40
    value = execution_manifest.t2_execution_manifest_contract_value(
        ROOT,
        admission_commit=admission_commit,
    )
    value["sources"]["trainer"]["sha256"] = "0" * 64
    monkeypatch.setattr(
        execution_manifest,
        "_observed_clean_execution_commit",
        lambda _root: "b" * 40,
    )
    monkeypatch.setattr(
        execution_manifest.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, b"", b""),
    )
    monkeypatch.setattr(
        execution_manifest,
        "_descendant_changed_paths",
        lambda *_args, **_kwargs: ("docs/record.md",),
    )

    with pytest.raises(ExperimentContractError, match="semantics or bindings differ"):
        validate_t2_execution_manifest(value, repository_root=ROOT)


def test_final_execution_manifest_refuses_dirty_tree_through_isolation_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse_dirty(_root: Path, *, allow_dirty: bool) -> object:
        assert allow_dirty is False
        raise ExperimentContractError("Phase B training requires clean committed sources")

    monkeypatch.setattr(
        "oracle_composition.phase_b.supervision.inspect_runtime_sources",
        refuse_dirty,
    )
    with pytest.raises(ExperimentContractError, match="requires clean committed sources"):
        final_ready_t2_execution_manifest_contract_value(
            ROOT,
            expected_admission_commit="a" * 40,
        )


def test_execution_manifest_refuses_untracked_dirty_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import oracle_composition.reward_study.execution_manifest as execution_manifest

    class CleanSourceSnapshot:
        def __init__(self) -> None:
            self.value = {"git": {"clean": True, "commit": "a" * 40}}

    monkeypatch.setattr(
        "oracle_composition.phase_b.supervision.inspect_runtime_sources",
        lambda _root, *, allow_dirty: CleanSourceSnapshot(),
    )
    monkeypatch.setattr(
        execution_manifest.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0, b"?? untracked-record\n", b""),
    )
    with pytest.raises(ExperimentContractError, match="tree is dirty"):
        execution_manifest._observed_clean_execution_commit(ROOT)


def test_candidate_admission_builds_one_final_ready_reseal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import oracle_composition.reward_study.execution_manifest as execution_manifest
    import oracle_composition.reward_study.final_admission as final_admission

    candidate_path = tmp_path / "candidate.json"
    candidate = TargetSpeedRewardSpec(alpha=1.0, beta=0.0)
    candidate_path.write_bytes(candidate.canonical_bytes)
    pending = json.loads((EXPERIMENT / "t2_reward_study_expert_hold_v1.json").read_bytes())
    pending["arms"][1]["reward"] = {
        "path": "TBD",
        "reward_id": "target_speed_triangular_affine_t2_adapter/v1",
        "sha256": "TBD",
    }
    pending["status"] = STUDY_STATUS
    pending["study_pairing_sha256"] = study_pairing_sha256_from_arm(pending["arms"][0])
    admission_commit = "a" * 40
    monkeypatch.setattr(
        execution_manifest,
        "_observed_clean_execution_commit",
        lambda _root: admission_commit,
    )
    monkeypatch.setattr(
        final_admission,
        "load_t2_study_manifest",
        lambda *_args, **_kwargs: (
            pending,
            hashlib.sha256(canonical_json_bytes(pending)).hexdigest(),
        ),
    )
    artifacts = admit_t2_candidate_and_reseal(
        repository_root=ROOT,
        pending_study_manifest_path=(EXPERIMENT / "t2_reward_study_expert_hold_v1.json"),
        candidate_artifact_root=tmp_path,
        candidate_reward_path=candidate_path,
        expected_admission_commit=admission_commit,
    )
    execution = json.loads(artifacts.execution_manifest_bytes)
    study = json.loads(artifacts.study_manifest_bytes)
    seal = json.loads(artifacts.t2_seal_bytes)
    assert artifacts.admission_commit == admission_commit
    assert execution["status"] == T2_EXECUTION_FINAL_READY_STATUS
    assert execution["repository"]["admission_commit"] == admission_commit
    assert execution["repository"]["execution_tree_clean_at_admission"] is True
    assert len(execution["source_snapshot_sha256"]) == 64
    assert t2_runtime_execution_identity(execution, repository_root=ROOT) == {
        "admission_commit": admission_commit,
        "execution_commit_observed": admission_commit,
    }
    assert study["status"] == STUDY_FINAL_READY_STATUS
    assert study["arms"][1]["reward"] == {
        "path": "candidate.json",
        "reward_id": "target_speed_triangular_affine_t2_adapter/v1",
        "sha256": hashlib.sha256(candidate.canonical_bytes).hexdigest(),
    }
    assert seal["dispatch_state"] == "candidate_admitted_no_further_initial_dispatch"
    assert seal["pairing_receipt"] == INTEGRATED_PAIRING_RECEIPT_SHA256
    assert (
        seal["execution_manifest"]["sha256"]
        == hashlib.sha256(artifacts.execution_manifest_bytes).hexdigest()
    )
    assert (
        seal["study_manifest"]["sha256"]
        == hashlib.sha256(artifacts.study_manifest_bytes).hexdigest()
    )


def test_clean_commit_reseal_function_replaces_only_three_canonical_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import oracle_composition.reward_study.final_admission as final_admission

    experiment = tmp_path / "experiments/004_t2_reward_study"
    experiment.mkdir(parents=True)
    names = (
        "execution_manifest_t2_v1.json",
        "t2_reward_study_expert_hold_v1.json",
        "t2_seal_v1.json",
    )
    for name in names:
        (experiment / name).write_bytes(b"old\n")
    expected = T2FinalReadyArtifacts(
        execution_manifest_bytes=b"execution\n",
        study_manifest_bytes=b"study\n",
        t2_seal_bytes=b"seal\n",
        admission_commit="a" * 40,
    )

    def build(**kwargs: object) -> T2FinalReadyArtifacts:
        assert kwargs == {
            "repository_root": tmp_path,
            "pending_study_manifest_path": experiment / "t2_reward_study_expert_hold_v1.json",
            "candidate_artifact_root": tmp_path,
            "candidate_reward_path": experiment / "candidate_target_speed_t2_v1.json",
            "expected_admission_commit": "a" * 40,
            "allow_final_ready_reseal": True,
        }
        return expected

    monkeypatch.setattr(final_admission, "admit_t2_candidate_and_reseal", build)
    assert (
        admit_t2_study(
            repository_root=tmp_path,
            experiment=Path("experiments/004_t2_reward_study"),
            expected_commit="a" * 40,
        )
        == expected
    )
    assert (experiment / names[0]).read_bytes() == b"execution\n"
    assert (experiment / names[1]).read_bytes() == b"study\n"
    assert (experiment / names[2]).read_bytes() == b"seal\n"
    assert {path.name for path in experiment.iterdir()} == set(names)


def test_study_manifest_refuses_common_field_drift_even_with_a_rehashed_key() -> None:
    value = json.loads((EXPERIMENT / "t2_reward_study_expert_hold_v1.json").read_bytes())
    value["arms"][1]["training"]["ppo_seeds"][0] = 121002
    value["study_pairing_sha256"] = "0" * 64
    with pytest.raises(T2StudyManifestError, match="common field"):
        validate_t2_study_manifest(value)


def test_candidate_reward_resolves_through_registry_and_refuses_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "candidate.json"
    spec = TargetSpeedRewardSpec(alpha=1.0, beta=0.0)
    path.write_bytes(spec.canonical_bytes)
    digest = hashlib.sha256(spec.canonical_bytes).hexdigest()
    binding = {
        "path": path.name,
        "reward_id": "target_speed_triangular_affine_t2_adapter/v1",
        "sha256": digest,
    }
    resolved, observed, resolved_path = resolve_t2_reward_binding(
        binding, artifact_root=tmp_path, candidate=True
    )
    assert resolved == spec
    assert observed == digest
    assert resolved_path == path

    changed_digest = copy.deepcopy(binding)
    changed_digest["sha256"] = "0" * 64
    with pytest.raises(T2StudyManifestError, match="artifact SHA-256 differs"):
        resolve_t2_reward_binding(changed_digest, artifact_root=tmp_path, candidate=True)

    path.write_bytes(canonical_json_bytes({"formula_id": "forged"}))
    changed_file = copy.deepcopy(binding)
    changed_file["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(T2StudyManifestError, match="registry refused"):
        resolve_t2_reward_binding(changed_file, artifact_root=tmp_path, candidate=True)

    path.write_bytes(spec.canonical_bytes)

    class TamperedRegistry:
        def load(self, candidate: Path) -> tuple[object, str]:
            return spec, "f" * 64

    monkeypatch.setattr(
        "oracle_composition.reward_study.study_manifest.RewardRegistry",
        TamperedRegistry,
    )
    with pytest.raises(T2StudyManifestError, match="registry digest differs"):
        resolve_t2_reward_binding(binding, artifact_root=tmp_path, candidate=True)


def test_pre_seam_pairing_adapter_receipt_is_canonical_but_superseded() -> None:
    encoded = PAIRING_RECEIPT.read_bytes()
    value = json.loads(encoded)
    assert encoded == canonical_json_bytes(value)
    assert hashlib.sha256(encoded).hexdigest() == (
        "6bd6f5ab3eb33ea563fff06c01828781d4c01864b7f0384108bfa5161c02540a"
    )
    with pytest.raises(ValueError, match="pairing receipt identity differs"):
        validate_pairing_receipt(value)
    assert (
        value["adapter_source_sha256"]
        != hashlib.sha256(
            (ROOT / "src/oracle_composition/reward_study/pairing.py").read_bytes()
        ).hexdigest()
    )
    assert value["integrated_runtime_receipt"] == "TBD_pending_astra_acceptance"
