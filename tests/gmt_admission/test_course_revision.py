from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from oracle_composition import cli
from oracle_composition.adapters.gmt import course_revision as module
from oracle_composition.experiments.artifact_io import finite_pretty_json


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, value: object) -> tuple[bytes, str]:
    encoded = finite_pretty_json(value)
    path.write_bytes(encoded)
    return encoded, _sha256(encoded)


def _inputs(root: Path) -> dict[str, object]:
    root.mkdir()
    parent_bytes, parent_sha256 = _write_json(root / "parent.json", {"parent": "fixed"})
    feedback_bytes, feedback_sha256 = _write_json(
        root / "feedback.json",
        {
            "evidence_class": "gmt_g1_course_development_feedback/v1",
            "source_manifest_sha256": "a" * 64,
        },
    )
    proposal_bytes, proposal_sha256 = _write_json(
        root / "proposal.json",
        {
            "schema_version": 1,
            "proposal_id": "revision-1",
            "factor": "reward",
            "replacement": {"reward": {"speed_weight": 1.5}},
        },
    )
    manifest_bytes, manifest_sha256 = _write_json(
        root / "course_run_manifest.json", {"artifact": "test-source-run"}
    )
    return {
        "parent_path": root / "parent.json",
        "parent_bytes": parent_bytes,
        "parent_sha256": parent_sha256,
        "feedback_path": root / "feedback.json",
        "feedback_bytes": feedback_bytes,
        "feedback_sha256": feedback_sha256,
        "proposal_path": root / "proposal.json",
        "proposal_bytes": proposal_bytes,
        "proposal_sha256": proposal_sha256,
        "manifest_path": root / "course_run_manifest.json",
        "manifest_bytes": manifest_bytes,
        "manifest_sha256": manifest_sha256,
    }


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    inputs: dict[str, object],
    *,
    rebuilt_feedback: bytes | None = None,
    source_config_sha256: str | None = None,
    packet_artifact: str = module.FEEDBACK_BUILD_RECEIPT_ARTIFACT,
    candidate: dict[str, object] | None = None,
) -> None:
    parent = SimpleNamespace(
        raw={"parent": "fixed"},
        encoded=inputs["parent_bytes"],
        sha256=inputs["parent_sha256"],
    )
    monkeypatch.setattr(module, "load_run_config", lambda path: parent)

    def fake_builder(
        *,
        manifest_path: Path,
        expected_manifest_sha256: str,
        label: str,
        output: Path,
    ) -> dict[str, object]:
        assert manifest_path == inputs["manifest_path"]
        assert expected_manifest_sha256 == inputs["manifest_sha256"]
        assert label == "final_policy"
        output.mkdir()
        feedback = rebuilt_feedback or inputs["feedback_bytes"]
        feedback_path = output / "feedback_v1.json"
        feedback_path.write_bytes(feedback)
        receipt = {
            "schema_version": 1,
            "artifact": packet_artifact,
            "status": "completed",
            "inputs": {
                "source_manifest_path": str(manifest_path),
                "source_manifest_sha256": expected_manifest_sha256,
                "input_config_sha256": source_config_sha256 or inputs["parent_sha256"],
                "label": label,
                "selected_outputs": {},
            },
            "output": {
                "path": "feedback_v1.json",
                "sha256": _sha256(feedback),
                "byte_count": len(feedback),
            },
            "claim_limits": ["development_only"],
        }
        receipt_path = output / "feedback_receipt_v1.json"
        receipt_path.write_bytes(finite_pretty_json(receipt))
        return {
            "feedback": {"path": str(feedback_path), "sha256": _sha256(feedback)},
            "receipt": {"path": str(receipt_path)},
        }

    monkeypatch.setattr(module, "build_g1_course_feedback", fake_builder)

    def fake_apply(parent_value, proposal, feedback):
        assert parent_value is parent
        assert proposal["proposal_id"] == "revision-1"
        assert feedback == inputs["feedback_bytes"]
        return candidate or {"schema_version": 1, "candidate": "reward-only"}

    monkeypatch.setattr(module, "apply_proposal", fake_apply)


def _run(inputs: dict[str, object], output: Path) -> dict[str, object]:
    return module.revise_g1_course(
        parent_config_path=inputs["parent_path"],
        expected_parent_config_sha256=inputs["parent_sha256"],
        feedback_path=inputs["feedback_path"],
        expected_feedback_sha256=inputs["feedback_sha256"],
        proposal_path=inputs["proposal_path"],
        expected_proposal_sha256=inputs["proposal_sha256"],
        source_manifest_path=inputs["manifest_path"],
        expected_source_manifest_sha256=inputs["manifest_sha256"],
        source_label="final_policy",
        output=output,
    )


def test_revision_retains_exact_inputs_candidate_and_lineage_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path / "inputs")
    _install_fakes(monkeypatch, inputs)
    output = tmp_path / "revision"

    result = _run(inputs, output)

    assert result["status"] == "completed"
    assert (output / "parent_config.input.json").read_bytes() == inputs["parent_bytes"]
    assert (output / "feedback.input.json").read_bytes() == inputs["feedback_bytes"]
    assert (output / "proposal.input.json").read_bytes() == inputs["proposal_bytes"]
    assert (output / "source_manifest.input.json").read_bytes() == inputs["manifest_bytes"]
    assert json.loads((output / "candidate_config.json").read_text()) == {
        "candidate": "reward-only",
        "schema_version": 1,
    }
    receipt = json.loads((output / "revision_receipt_v1.json").read_text())
    assert receipt["artifact"] == module.REVISION_RECEIPT_ARTIFACT
    assert receipt["factor"] == "reward"
    assert receipt["changed_fields"] == ["reward"]
    assert receipt["source_run"]["manifest_sha256"] == inputs["manifest_sha256"]
    assert receipt["source_run"]["label"] == "final_policy"
    assert receipt["source_run"]["input_config_sha256"] == inputs["parent_sha256"]
    packet = receipt["source_run"]["feedback_builder_packet"]
    assert packet["artifact"] == module.FEEDBACK_BUILD_RECEIPT_ARTIFACT
    assert len(packet["sha256"]) == 64
    assert packet["byte_count"] > 0
    for name, encoded in (
        ("parent_config", inputs["parent_bytes"]),
        ("feedback", inputs["feedback_bytes"]),
        ("proposal", inputs["proposal_bytes"]),
        ("source_manifest", inputs["manifest_bytes"]),
    ):
        assert receipt["retained_inputs"][name]["sha256"] == _sha256(encoded)
        assert receipt["retained_inputs"][name]["byte_count"] == len(encoded)
    assert "not_trained_or_evaluated" in receipt["claim_limits"]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"rebuilt_feedback": finite_pretty_json({"stale": True})}, "rebuilt feedback"),
        ({"source_config_sha256": "f" * 64}, "lineage"),
        ({"packet_artifact": "gmt_g1_course_feedback_build_receipt/v0"}, "version"),
    ],
)
def test_revision_rejects_stale_or_wrong_lineage_feedback_before_emission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, object],
    message: str,
) -> None:
    inputs = _inputs(tmp_path / "inputs")
    _install_fakes(monkeypatch, inputs, **override)
    output = tmp_path / "revision"

    with pytest.raises(ValueError, match=message):
        _run(inputs, output)

    assert not output.exists()


@pytest.mark.parametrize(
    "field",
    ["parent_sha256", "feedback_sha256", "proposal_sha256", "manifest_sha256"],
)
def test_revision_rejects_each_wrong_declared_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    inputs = _inputs(tmp_path / "inputs")
    _install_fakes(monkeypatch, inputs)
    inputs[field] = "0" * 64

    with pytest.raises(ValueError, match="SHA-256"):
        _run(inputs, tmp_path / "revision")

    assert not (tmp_path / "revision").exists()


@pytest.mark.parametrize("input_name", ["parent", "feedback", "proposal", "manifest"])
def test_revision_rejects_symlinked_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_name: str,
) -> None:
    inputs = _inputs(tmp_path / "inputs")
    _install_fakes(monkeypatch, inputs)
    original = inputs[f"{input_name}_path"]
    linked = tmp_path / f"linked-{input_name}.json"
    linked.symlink_to(original)
    inputs[f"{input_name}_path"] = linked

    with pytest.raises(ValueError, match="regular non-linked"):
        _run(inputs, tmp_path / "revision")

    assert not (tmp_path / "revision").exists()


def test_revision_refuses_existing_or_symlinked_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path / "inputs")
    _install_fakes(monkeypatch, inputs)
    existing = tmp_path / "existing"
    existing.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(existing)

    for output in (existing, linked):
        with pytest.raises(ValueError, match="must not already exist"):
            _run(inputs, output)


def test_authorability_failure_does_not_emit_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path / "inputs")
    _install_fakes(monkeypatch, inputs)

    def reject(*_args, **_kwargs):
        raise ValueError("proposal changed a frozen parent config field")

    monkeypatch.setattr(module, "apply_proposal", reject)
    output = tmp_path / "revision"
    with pytest.raises(ValueError, match="frozen parent"):
        _run(inputs, output)
    assert not output.exists()


def test_nonfinite_candidate_does_not_emit_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = _inputs(tmp_path / "inputs")
    _install_fakes(monkeypatch, inputs, candidate={"reward": float("nan")})
    output = tmp_path / "revision"

    with pytest.raises(ValueError, match="finite JSON"):
        _run(inputs, output)

    assert not output.exists()


def test_cli_routes_all_revision_identities(tmp_path: Path, monkeypatch, capsys) -> None:
    observed = {}

    def fake_revise(**kwargs):
        observed.update(kwargs)
        return {"status": "completed", "factor": "oracle"}

    monkeypatch.setattr(module, "revise_g1_course", fake_revise)
    paths = {name: tmp_path / f"{name}.json" for name in ("parent", "feedback", "proposal")}
    manifest = tmp_path / "manifest.json"
    output = tmp_path / "output"
    digest = "a" * 64

    exit_code = cli.main(
        [
            "--json",
            "g1",
            "revise",
            "--parent-config",
            str(paths["parent"]),
            "--parent-config-sha256",
            digest,
            "--feedback",
            str(paths["feedback"]),
            "--feedback-sha256",
            digest,
            "--proposal",
            str(paths["proposal"]),
            "--proposal-sha256",
            digest,
            "--source-manifest",
            str(manifest),
            "--source-manifest-sha256",
            digest,
            "--source-label",
            "final_policy",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert observed == {
        "parent_config_path": paths["parent"],
        "expected_parent_config_sha256": digest,
        "feedback_path": paths["feedback"],
        "expected_feedback_sha256": digest,
        "proposal_path": paths["proposal"],
        "expected_proposal_sha256": digest,
        "source_manifest_path": manifest,
        "expected_source_manifest_sha256": digest,
        "source_label": "final_policy",
        "output": output,
    }
    assert json.loads(capsys.readouterr().out) == {"factor": "oracle", "status": "completed"}
