from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.adapters.gmt import course_config
from oracle_composition.adapters.gmt import derived_reference as module
from oracle_composition.adapters.gmt.contracts import MOTION_SPECS
from oracle_composition.adapters.gmt.io import GMTAdmissionError

SCRIPT = Path(__file__).parents[2] / "scripts/run_gmt_development.py"
SCRIPT_DIR = str(SCRIPT.parent)
SPEC = importlib.util.spec_from_file_location("derived_reference_launcher_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DEVELOPMENT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DEVELOPMENT
sys.path.insert(0, SCRIPT_DIR)
SPEC.loader.exec_module(DEVELOPMENT)
sys.path.remove(SCRIPT_DIR)

ASTRA_ROOT = Path("/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra")
ARTIFACT_ROOT = ASTRA_ROOT / "artifacts/gmt/2a590de25a1eb08e"
DERIVED_ROOT = ASTRA_ROOT / "artifacts/gmt/derived_references"


def _derived_asset() -> dict[str, object]:
    reference = DERIVED_ROOT / "study012_inside_passage_v1.npz"
    manifest = reference.with_suffix(".npz.manifest.json")
    if not reference.is_file() or not manifest.is_file():
        pytest.skip("exact reviewed Study 012 derived reference is unavailable")
    return {
        "path": str(reference),
        "sha256": module.STUDY012_INSIDE_PASSAGE.archive_sha256,
        "manifest": {
            "path": str(manifest),
            "sha256": module.STUDY012_INSIDE_PASSAGE.manifest_sha256,
        },
    }


def _config() -> dict[str, object]:
    upstream = ARTIFACT_ROOT / "upstream"
    weights = ARTIFACT_ROOT / "numeric/gmt_g1_actor_weights.npz"
    if not upstream.is_dir() or not weights.is_file():
        pytest.skip("exact local GMT actor artifacts are unavailable")
    return {
        "schema_version": 1,
        "mode": "probe",
        "seed": 20260906,
        "training_steps": 0,
        "assets": {
            "upstream_root": str(upstream),
            "weights": {"path": str(weights), "sha256": course_config.ACTOR_SHA256},
            "motions": {module.STUDY012_INSIDE_PASSAGE.name: _derived_asset()},
        },
        "segments": {
            "passage": {
                "motion_name": module.STUDY012_INSIDE_PASSAGE.name,
                "start_seconds": 0.0,
                "end_seconds": 2.08,
            }
        },
        "task": {
            "schema_id": "gmt_g1_posture_course_task/v1",
            "schema_version": 1,
            "region_entry_distance_m": 1.0,
            "region_exit_distance_m": 2.0,
            "finish_distance_m": 3.5,
            "target_speed_outside_m_s": 0.7,
            "target_speed_inside_m_s": 0.65,
            "posture_band_low_m": 0.3,
            "posture_band_high_m": 0.6,
            "horizon_steps": 1000,
        },
        "reward": {
            "schema_id": "gmt_g1_posture_course_reward/v1",
            "schema_version": 1,
            "speed_weight": 1.0,
            "posture_weight": 2.0,
            "lateral_weight": 1.0,
            "heading_weight": 0.5,
            "failure_weight": 1.0,
        },
        "oracle": {
            "schema_version": 1,
            "evidence_class": "exploratory_oracle_cycle",
            "oracle_id": "study012_derived_reference_probe_fixture",
            "behaviors": ["passage"],
            "initial": "before",
            "states": {
                name: {"behavior": "passage", "min_dwell": 25}
                for name in ("before", "inside", "after")
            },
            "transitions": [
                {
                    "from": "before",
                    "to": "inside",
                    "priority": 0,
                    "guard": "x_travelled >= 1",
                },
                {
                    "from": "inside",
                    "to": "after",
                    "priority": 0,
                    "guard": "x_travelled >= 2",
                },
            ],
        },
    }


def _write_config(path: Path, raw: dict[str, object]) -> None:
    path.write_bytes(json.dumps(raw, sort_keys=True, separators=(",", ":")).encode())


def test_exact_execution_derived_pair_loads_as_probe_only_reference() -> None:
    asset = _derived_asset()
    imported_mujoco_before = "mujoco" in sys.modules
    admission = module.admit_derived_reference(
        module.STUDY012_INSIDE_PASSAGE.name, asset, mode="probe"
    )

    assert admission.motion.frame_count == 106
    assert float(admission.motion.fps) == 50.0
    assert admission.resource_binding == {
        "path": asset["path"],
        "sha256": module.STUDY012_INSIDE_PASSAGE.archive_sha256,
        "size": 13_650,
        "manifest": {
            "path": asset["manifest"]["path"],
            "sha256": module.STUDY012_INSIDE_PASSAGE.manifest_sha256,
            "size": 10_965,
        },
        "provenance_class": "execution_derived_kinematic_candidate_not_dynamics_certificate",
        "candidate_id": module.STUDY012_INSIDE_PASSAGE.candidate_id,
        "training_admitted": False,
    }
    assert len(MOTION_SPECS) == 8
    assert len(module.DERIVED_REFERENCE_SPECS) == 1
    assert module.STUDY012_INSIDE_PASSAGE.name not in MOTION_SPECS
    assert ("mujoco" in sys.modules) is imported_mujoco_before


@pytest.mark.parametrize("mutation", ["training", "missing_manifest", "extra", "stale"])
def test_derived_reference_rejects_training_or_unbound_provenance(mutation: str) -> None:
    asset = copy.deepcopy(_derived_asset())
    mode = "probe"
    if mutation == "training":
        mode = "train"
    elif mutation == "missing_manifest":
        asset.pop("manifest")
    elif mutation == "extra":
        asset["tier_d"] = True
    else:
        asset["manifest"]["sha256"] = "0" * 64

    with pytest.raises(GMTAdmissionError):
        module.admit_derived_reference(
            module.STUDY012_INSIDE_PASSAGE.name, asset, mode=mode
        )


def test_changed_archive_or_manifest_bytes_fail_closed(tmp_path: Path) -> None:
    source = _derived_asset()
    archive = tmp_path / "study012_inside_passage_v1.npz"
    manifest = archive.with_suffix(".npz.manifest.json")
    archive.write_bytes(Path(source["path"]).read_bytes())
    manifest.write_bytes(Path(source["manifest"]["path"]).read_bytes())
    asset = {
        "path": str(archive),
        "sha256": source["sha256"],
        "manifest": {"path": str(manifest), "sha256": source["manifest"]["sha256"]},
    }
    archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(GMTAdmissionError, match="size mismatch"):
        module.admit_derived_reference(
            module.STUDY012_INSIDE_PASSAGE.name, asset, mode="probe"
        )

    archive.write_bytes(Path(source["path"]).read_bytes())
    manifest.write_bytes(manifest.read_bytes() + b"changed")
    with pytest.raises(GMTAdmissionError, match="size mismatch"):
        module.admit_derived_reference(
            module.STUDY012_INSIDE_PASSAGE.name, asset, mode="probe"
        )


def test_unknown_or_missing_derived_reference_fails_closed(tmp_path: Path) -> None:
    asset = copy.deepcopy(_derived_asset())
    with pytest.raises(GMTAdmissionError, match="outside the derived candidate registry"):
        module.admit_derived_reference("study012_unknown", asset, mode="probe")

    missing = tmp_path / "study012_inside_passage_v1.npz"
    asset["path"] = str(missing)
    asset["manifest"]["path"] = str(missing.with_suffix(".npz.manifest.json"))
    with pytest.raises(GMTAdmissionError, match="unavailable"):
        module.admit_derived_reference(
            module.STUDY012_INSIDE_PASSAGE.name, asset, mode="probe"
        )


def _numeric_payload(arrays: dict[str, np.ndarray]) -> tuple[bytes, module.DerivedReferenceSpec]:
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    payload = stream.getvalue()
    contracts = {}
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        for name, array in arrays.items():
            info = archive.getinfo(f"{name}.npy")
            contracts[name] = module._ArrayContract(
                shape=(106, 3) if name == "root_pos" else tuple(array.shape),
                dtype=array.dtype.str,
                member_size=info.file_size,
                values_sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest(),
            )
    return payload, replace(module.STUDY012_INSIDE_PASSAGE, arrays=contracts)


@pytest.mark.parametrize("mutation", ["shape", "nonfinite"])
def test_numeric_loader_rejects_wrong_shape_and_nonfinite_before_runtime(mutation: str) -> None:
    arrays = {
        "fps": np.asarray([50.0], dtype="<f8"),
        "root_pos": np.zeros((105 if mutation == "shape" else 106, 3), dtype="<f4"),
        "root_rot": np.tile(np.asarray([0.0, 0.0, 0.0, 1.0], dtype="<f4"), (106, 1)),
        "dof_pos": np.zeros((106, 23), dtype="<f4"),
    }
    if mutation == "nonfinite":
        arrays["root_pos"][0, 0] = np.nan
    payload, spec = _numeric_payload(arrays)

    with pytest.raises(GMTAdmissionError, match="array contract"):
        module._load_arrays(payload, spec)


def test_course_config_and_supervised_plan_retain_exact_provenance_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = _config()
    config_path = tmp_path / "derived-probe.json"
    _write_config(config_path, raw)

    admitted = course_config.load_run_config(config_path)
    assert admitted.encoded == config_path.read_bytes()
    assert admitted.segments["passage"].motion.frame_count == 106

    loaded = DEVELOPMENT._load_course(config_path)
    expected = module.admit_derived_reference(
        module.STUDY012_INSIDE_PASSAGE.name,
        raw["assets"]["motions"][module.STUDY012_INSIDE_PASSAGE.name],
        mode="probe",
    ).resource_binding
    assert loaded.assets["motions"] == {
        module.STUDY012_INSIDE_PASSAGE.name: expected
    }

    root = Path(__file__).parents[2].resolve()
    monkeypatch.setattr(
        DEVELOPMENT,
        "_repository_sources",
        lambda _: (root, "a" * 40, {"src/oracle_composition/fixed.py": "b" * 64}),
    )
    monkeypatch.setattr(
        DEVELOPMENT.supervisor,
        "_venv_identity",
        lambda path: {"python": str(path)},
    )
    request = DEVELOPMENT.DevelopmentRequest(
        workload="course",
        repository_root=root,
        venv_python=root / ".venv/bin/python",
        config_path=config_path,
        output_directory=tmp_path / "not-created-output",
        owner="astra-derived-reference-test",
    )
    plan = DEVELOPMENT.build_development_plan(request)
    DEVELOPMENT.validate_development_plan(plan)
    assert plan.inputs["gmt_assets"]["motions"] == {
        module.STUDY012_INSIDE_PASSAGE.name: expected
    }
    plan.inputs["gmt_assets"]["motions"][module.STUDY012_INSIDE_PASSAGE.name][
        "manifest"
    ]["sha256"] = "0" * 64
    with pytest.raises(DEVELOPMENT.supervisor.ProbeError, match="bound inputs changed"):
        DEVELOPMENT.validate_development_plan(plan)


def test_course_config_rejects_derived_training_before_runtime_use(tmp_path: Path) -> None:
    raw = _config()
    raw.update(mode="train", training_steps=512)
    config_path = tmp_path / "derived-train.json"
    _write_config(config_path, raw)

    with pytest.raises(GMTAdmissionError, match="probe-only"):
        course_config.load_run_config(config_path)
