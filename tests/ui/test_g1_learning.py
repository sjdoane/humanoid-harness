from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from oracle_composition.adapters.gmt.training_contract import (
    TRAINING_REWARD_SCALE,
    CourseTrainerSpec,
    effective_training_contract,
)
from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.ui import g1_learning as module


def _json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fixture(
    root: Path,
    *,
    scaled: bool = False,
    manifest_scaled: bool | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    run = root / "runs" / "o2-seed7"
    run.mkdir(parents=True)
    config = {"mode": "train", "seed": 7, "training_steps": 512}
    trainer = CourseTrainerSpec(TRAINING_REWARD_SCALE) if scaled else None
    if trainer is not None:
        config["trainer"] = trainer.to_dict()
    config_bytes = _json(config)
    (run / "input_config.json").write_bytes(config_bytes)
    evaluation_bytes = _json({"retained": True})
    (run / "final_policy_evaluation.json").write_bytes(evaluation_bytes)
    task_sha = "1" * 64
    gates = {"finish_reached": True, "inside_posture_compliance": False}
    objective = {
        "evaluator_id": "gmt_g1_posture_course_development_evaluator/v1",
        "claim_scope": "fixed_development_gate_only_not_heldout_or_universal",
        "task_sha256": task_sha,
        "duration_seconds": 20.0,
        "fall_count": 0,
        "finish_condition_observed": True,
        "maximum_progress_m": 4.1,
        "region": {
            "posture_compliant_fraction": 0.5,
            "minimum_root_height_m": 0.48,
        },
        "speed": {
            "mean_absolute_error_m_s": 0.2,
            "inside_mean_speed_target_deviation_m_s": 0.08,
        },
        "maximum_lateral_error_m": 0.3,
        "tracking": {
            "joint_position_rmse_rad_p95": 0.19,
            "roll_pitch_rmse_rad_p95": 0.09,
        },
        "development_gate_results": gates,
        "development_gate_passed": False,
        "episode_success": None,
    }
    manifest = {
        "schema_version": 1,
        "artifact": "gmt_g1_course_development_run",
        "status": "completed",
        "input_config_sha256": _sha(config_bytes),
        "outputs": {
            "input_config.json": _sha(config_bytes),
            "final_policy_evaluation.json": _sha(evaluation_bytes),
        },
        "identities": {
            "task": task_sha,
            "oracle": "2" * 64,
            "reward": "3" * 64,
            "segments": {"walk": "4" * 64},
        },
        "frozen_runtime": {
            "trainer": effective_training_contract(
                CourseTrainerSpec(TRAINING_REWARD_SCALE)
                if (scaled if manifest_scaled is None else manifest_scaled)
                else None
            )
        },
        "training": {"completed_transitions": 512},
        "zero_residual": None,
        "final_policy": {"objective_evaluation": objective},
        "runtime": {},
        "claims": {},
    }
    manifest_bytes = _json(manifest)
    manifest_path = run / "course_run_manifest.json"
    manifest_path.write_bytes(manifest_bytes)
    registry = {
        "schema_version": 1,
        "runs": [
            {
                "run_id": "o2-seed7",
                "manifest_path": str(manifest_path),
                "manifest_sha256": _sha(manifest_bytes),
                "label": "final_policy",
            }
        ],
    }
    registry_path = root / module.REGISTRY_RELATIVE_PATH
    registry_path.parent.mkdir(parents=True)
    registry_path.write_bytes(_json(registry))
    return registry_path, manifest_path, objective


def _install_validator(
    monkeypatch: pytest.MonkeyPatch,
    objective: dict[str, object],
) -> list[object]:
    calls: list[object] = []

    def verified(entry):
        calls.append(entry)
        return {
            "evidence_class": "g1_course_development_feedback/v1",
            "protected_evaluation": False,
            "source_manifest_sha256": entry.manifest_sha256,
            "evaluation": {
                name: objective[name]
                for name in (
                    "evaluator_id",
                    "claim_scope",
                    "task_sha256",
                    "development_gate_results",
                    "development_gate_passed",
                    "episode_success",
                )
            },
            "diagnosis": "validated fixture",
        }

    monkeypatch.setattr(module, "_rebuild_feedback", verified)
    return calls


@pytest.mark.parametrize("scaled", [False, True], ids=["raw", "scaled-v2"])
def test_registered_raw_and_v2_runs_use_authoritative_trainer_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scaled: bool
) -> None:
    registry_path, manifest_path, objective = _fixture(tmp_path, scaled=scaled)
    calls = _install_validator(monkeypatch, objective)

    payload = module.g1_learning_status(tmp_path)

    assert payload["state"] == "available"
    assert payload["summary"] == {
        "accepted_runs": 1,
        "rejected_runs": 0,
        "full_task_development_gate_passes": 0,
    }
    assert payload["validation"] == "manual_snapshot_sequential_existing_feedback_evaluator"
    assert payload["validated_at"].endswith("+00:00")
    assert payload["registry"]["sha256"] == _sha(registry_path.read_bytes())
    assert len(calls) == 1
    assert calls[0].manifest_path == manifest_path
    run = payload["runs"][0]
    assert run["full_task_development_gate_passed"] is False
    assert run["development_gates"] == {
        "passed": 1,
        "total": 2,
        "failed": ["inside_posture_compliance"],
    }
    assert run["evaluator"]["episode_success"] is None
    assert run["training"]["requested_transitions"] == 512
    assert run["training"]["completed_transitions"] == 512
    assert run["training"]["seed"] == 7
    expected = effective_training_contract(
        CourseTrainerSpec(TRAINING_REWARD_SCALE) if scaled else None
    )
    training = run["training"]
    assert training["algorithm"] == "stable_baselines3.PPO"
    assert training["producer_recorded_trainer"] == expected
    assert training["full_trainer_contract_sha256"] == _sha(canonical_json_bytes(expected))
    if scaled:
        assert training["trainer_variant"] == "gmt_g1_ppo_training_contract/v2"
        assert training["trainer_payload_identity_sha256"] == expected["identity_sha256"]
        assert training["full_trainer_contract_sha256"] != expected["identity_sha256"]
    else:
        assert training["trainer_variant"] == "legacy_raw_training_reward"
        assert training["trainer_payload_identity_sha256"] is None
    serialized = json.dumps(payload)
    assert str(tmp_path) not in serialized
    assert str(manifest_path) not in serialized


def test_missing_or_empty_registry_does_not_search_for_runs(tmp_path: Path) -> None:
    missing = module.g1_learning_status(tmp_path)
    assert missing == {
        "state": "unavailable",
        "authority": module.REGISTRY_AUTHORITY,
        "detail": "no local registry at artifacts/gmt/g1_learning_registry.json",
        "runs": [],
    }

    registry = tmp_path / module.REGISTRY_RELATIVE_PATH
    registry.parent.mkdir(parents=True)
    registry.write_bytes(_json({"schema_version": 1, "runs": []}))
    empty = module.g1_learning_status(tmp_path)
    assert empty["state"] == "empty"
    assert empty["runs"] == []


def test_tampered_manifest_is_rejected_before_existing_validator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _registry, manifest, objective = _fixture(tmp_path)
    calls = _install_validator(monkeypatch, objective)
    manifest.write_bytes(manifest.read_bytes() + b" ")

    payload = module.g1_learning_status(tmp_path)

    assert payload["state"] == "rejected"
    assert payload["summary"]["rejected_runs"] == 1
    assert payload["runs"][0]["detail"] == "manifest_sha256_mismatch"
    assert calls == []


def test_registry_rejects_symlinks_duplicates_and_excess_source_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, _manifest, _objective = _fixture(tmp_path / "linked")
    target = tmp_path / "registry-target.json"
    target.write_bytes(registry.read_bytes())
    registry.unlink()
    registry.symlink_to(target)
    assert (
        module.g1_learning_status(tmp_path / "linked")["detail"] == "registry_path_escapes_project"
    )

    escape_root = tmp_path / "escape"
    escape_root.mkdir()
    outside = tmp_path / "outside-artifacts" / "gmt"
    outside.mkdir(parents=True)
    (outside / "g1_learning_registry.json").write_bytes(_json({"schema_version": 1, "runs": []}))
    (escape_root / "artifacts").symlink_to(outside.parent, target_is_directory=True)
    assert module.g1_learning_status(escape_root)["detail"] == "registry_path_escapes_project"

    alias_root = tmp_path / "alias"
    alias_registry, manifest, _objective = _fixture(alias_root)
    alias_parent = alias_root / "same-run-alias"
    alias_parent.symlink_to(manifest.parent, target_is_directory=True)
    value = json.loads(alias_registry.read_bytes())
    alias_entry = dict(value["runs"][0])
    alias_entry["run_id"] = "o2-seed7-alias"
    alias_entry["manifest_path"] = str(alias_parent / manifest.name)
    value["runs"].append(alias_entry)
    alias_registry.write_bytes(_json(value))
    assert module.g1_learning_status(alias_root)["detail"] == "registry_duplicate_run_invalid"

    bounded_root = tmp_path / "bounded"
    _registry, _manifest, _objective = _fixture(bounded_root)
    monkeypatch.setattr(module, "MAX_AGGREGATE_SOURCE_BYTES", 1)
    bounded = module.g1_learning_status(bounded_root)
    assert bounded["detail"] == "registry_aggregate_source_bytes_exceed_limit"


def test_budget_mismatch_rejects_only_the_registered_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry, manifest, objective = _fixture(tmp_path)
    _install_validator(monkeypatch, objective)
    value = json.loads(manifest.read_bytes())
    value["training"]["completed_transitions"] = 511
    manifest_bytes = _json(value)
    manifest.write_bytes(manifest_bytes)
    registry_value = json.loads(registry.read_bytes())
    registry_value["runs"][0]["manifest_sha256"] = _sha(manifest_bytes)
    registry.write_bytes(_json(registry_value))

    payload = module.g1_learning_status(tmp_path)

    assert payload["state"] == "rejected"
    assert payload["summary"]["rejected_runs"] == 1
    assert payload["runs"] == [
        {
            "state": "rejected",
            "run_id": "o2-seed7",
            "selected_label": "final_policy",
            "detail": "completed_training_budget_invalid",
        }
    ]


@pytest.mark.parametrize(
    ("config_scaled", "manifest_scaled"),
    [(False, True), (True, False)],
    ids=["raw-config-scaled-manifest", "scaled-config-raw-manifest"],
)
def test_trainer_manifest_must_match_retained_config_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_scaled: bool,
    manifest_scaled: bool,
) -> None:
    _registry, _manifest, objective = _fixture(
        tmp_path,
        scaled=config_scaled,
        manifest_scaled=manifest_scaled,
    )
    _install_validator(monkeypatch, objective)

    payload = module.g1_learning_status(tmp_path)

    assert payload["state"] == "rejected"
    assert payload["runs"][0]["detail"] == "trainer_config_manifest_mismatch"


def test_static_view_is_manual_and_separates_g1_from_historical_native() -> None:
    static = Path(module.__file__).with_name("static")
    html = (static / "index.html").read_text(encoding="utf-8")
    script = (static / "app.js").read_text(encoding="utf-8")
    server = Path(module.__file__).with_name("server.py").read_text(encoding="utf-8")

    assert 'data-view="g1-learning"' in html
    assert "full-task development gate" in html
    assert "Historical native Humanoid-v5" in html
    assert "Protected evaluation must not steer candidate authoring." in html
    assert '<section class="view is-visible" id="g1-learning"' in html
    assert "Historical native Humanoid-v5 admission" in html
    assert "The protected evaluator feeds the next iteration." not in html
    assert 'loadJson("/api/g1-learning")' in script
    assert "row.training.algorithm" in script
    assert "contract.algorithm" not in script
    assert "full_contract_sha256" in script
    assert "payload_identity_sha256" in script
    assert 'appendG1Line(source, "manifest"' not in script
    assert "setInterval" not in script
    assert 'request.path == "/api/g1-learning"' in server
    assert "g1_learning_validation_lock.acquire(blocking=False)" in server


def test_busy_validation_preserves_last_snapshot_for_manual_retry() -> None:
    script = (Path(module.__file__).with_name("static") / "app.js").read_text(encoding="utf-8")
    busy_renderer = script.split("function renderG1LearningBusy(payload) {", 1)[1].split(
        "function renderG1Learning(payload) {", 1
    )[0]

    assert "const hasValidatedSnapshot = !results.hidden;" in busy_renderer
    assert "Showing the last validated snapshot." in busy_renderer
    assert "Choose Validate runs to retry." in busy_renderer
    assert "results.hidden =" not in busy_renderer
    assert 'if (payload.state === "busy") {' in script
    assert "return false;" in script
    assert "if (snapshotAccepted) g1LearningLoaded = true;" in script


def test_busy_validation_dom_state_and_manual_retry() -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    script = (Path(module.__file__).with_name("static") / "app.js").read_text(encoding="utf-8")
    g1_source = script.split("function g1Text(value, field) {", 1)[1].split(
        "function renderResults(payload) {", 1
    )[0]
    production = (
        "let g1LearningLoaded = false;\n"
        "let g1LearningLoading = false;\n"
        "function g1Text(value, field) {" + g1_source
    )
    harness = r"""
import assert from "node:assert/strict";
import vm from "node:vm";
const production = __PRODUCTION__;

function element() {
  return {
    children: [], listeners: {}, hidden: false, textContent: "", className: "",
    append(...values) { this.children.push(...values); },
    replaceChildren(...values) { this.children = [...values]; },
    addEventListener(name, listener) { this.listeners[name] = listener; },
  };
}

function contextFor(responses) {
  const elements = new Map();
  const document = {
    querySelector(selector) {
      if (!elements.has(selector)) elements.set(selector, element());
      return elements.get(selector);
    },
    createElement: element,
    createTextNode(value) { return { textContent: String(value), children: [] }; },
  };
  document.querySelector("#g1-learning-results").hidden = true;
  const queue = [...responses];
  const context = vm.createContext({
    document,
    loadJson: async (url) => {
      assert.equal(url, "/api/g1-learning");
      assert.ok(queue.length, "unexpected validation request");
      return queue.shift();
    },
  });
  vm.runInContext(production, context);
  vm.runInContext(`
    appendG1Run = (row) => {
      const rendered = document.createElement("tr");
      rendered.runId = row.run_id;
      const receipt = document.createElement("pre");
      receipt.textContent = row.receipt;
      rendered.append(receipt);
      document.querySelector("#g1-learning-body").append(rendered);
    };
  `, context);
  return { context, document };
}

function available(runId, receipt) {
  return {
    state: "available", authority: "test-authority", validated_at: "test-time",
    validation: "manual", registry: { sha256: "test-sha", aggregate_source_bytes: 1 },
    summary: { accepted_runs: 1, rejected_runs: 0, full_task_development_gate_passes: 1 },
    runs: [{ run_id: runId, receipt }],
  };
}
const busy = { state: "busy", detail: "validation is running", runs: [] };
const loaded = (context) => vm.runInContext("g1LearningLoaded", context);
const load = (context) => vm.runInContext("loadG1Learning()", context);
const retry = (document) => document.querySelector("#g1-learning-refresh").listeners.click();

{
  const { context, document } = contextFor([
    available("one", "receipt-one"), busy, available("two", "receipt-two"),
  ]);
  await load(context);
  const results = document.querySelector("#g1-learning-results");
  const body = document.querySelector("#g1-learning-body");
  const row = body.children[0];
  const receipt = row.children[0];
  await retry(document);
  assert.equal(results.hidden, false);
  assert.strictEqual(body.children[0], row);
  assert.strictEqual(body.children[0].children[0], receipt);
  assert.equal(receipt.textContent, "receipt-one");
  assert.equal(document.querySelector("#g1-learning-badge").textContent, "validation busy");
  assert.match(document.querySelector("#g1-learning-state").textContent, /Showing the last validated snapshot/);
  assert.equal(loaded(context), true);
  await retry(document);
  assert.equal(body.children[0].runId, "two");
  assert.equal(body.children[0].children[0].textContent, "receipt-two");
}

{
  const { context, document } = contextFor([busy, available("after-busy", "receipt-after")]);
  await load(context);
  assert.equal(loaded(context), false);
  assert.equal(document.querySelector("#g1-learning-results").hidden, true);
  assert.equal(document.querySelector("#g1-learning-body").children.length, 0);
  assert.match(document.querySelector("#g1-learning-state").textContent, /No validated snapshot is loaded/);
  await retry(document);
  assert.equal(loaded(context), true);
  assert.equal(document.querySelector("#g1-learning-results").hidden, false);
  assert.equal(document.querySelector("#g1-learning-body").children[0].runId, "after-busy");
}
""".replace("__PRODUCTION__", json.dumps(production))

    subprocess.run(
        [node, "--input-type=module", "-"],
        input=harness,
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
