from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.adapters.gmt import trace_admission as TRACE
from oracle_composition.adapters.gmt.contracts import (
    DEFAULT_DOF_POSITION,
    GMT_G1_MESH_NAMES,
    GMT_G1_MESH_TREE_SHA256,
    GMT_SUPPORT_FILE_SHA256,
    GMT_UPSTREAM_COMMIT,
    JOINT_NAMES,
    REFERENCE_OFFSETS,
)
from oracle_composition.adapters.gmt.io import GMTAdmissionError, write_deterministic_npz
from oracle_composition.adapters.gmt.replay import ModelABI

SCRIPT = Path(__file__).parents[2] / "scripts/render_gmt_trace.py"
SPEC = importlib.util.spec_from_file_location("render_gmt_trace_test_module", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RENDER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RENDER
SPEC.loader.exec_module(RENDER)


def _support_files() -> dict[str, str]:
    support = dict(GMT_SUPPORT_FILE_SHA256)
    support.update({f"assets/robots/g1/meshes/{name}": "a" * 64 for name in GMT_G1_MESH_NAMES})
    support["assets/robots/g1/meshes@tree"] = GMT_G1_MESH_TREE_SHA256
    return support


def _abi() -> ModelABI:
    return ModelABI(
        nq=30,
        nv=29,
        nu=23,
        joint_names=("pelvis", *(f"{name}_joint" for name in JOINT_NAMES)),
        joint_types=("free", *("hinge" for _ in JOINT_NAMES)),
        actuator_names=tuple(f"{name}_joint" for name in JOINT_NAMES),
        actuator_joint_ids=tuple(range(1, 24)),
        sensor_names=("orientation", "position", "angular-velocity"),
        sensor_types=("framequat", "framepos", "gyro"),
        sensor_dimensions=(4, 3, 3),
        keyframe_count=1,
        home_qpos=(0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, *DEFAULT_DOF_POSITION),
        timestep=0.001,
    )


def _trace_arrays(simulation_steps: int = 20) -> dict[str, np.ndarray]:
    contract = TRACE.expected_trace_contract(simulation_steps)
    arrays = {key: np.zeros(shape, dtype=dtype) for key, (shape, dtype) in contract.items()}
    arrays["sim_time"] = np.arange(simulation_steps) * 0.001
    arrays["control_sim_step"] = np.arange(arrays["control_sim_step"].shape[0]) * 20
    arrays["control_time"] = arrays["control_sim_step"] * 0.001
    arrays["sim_qpos"][:, 2] = 1.0
    arrays["sim_qpos"][:, 3] = 1.0
    return arrays


def _manifest(path: Path, digest: str, arrays: dict[str, np.ndarray]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact": "gmt_g1_reconstructed_actor_headless_replay",
        "claim_status": "reconstructed_plain_actor_not_jit_equivalent",
        "inputs": {
            "upstream_commit": GMT_UPSTREAM_COMMIT,
            "weights_sha256": "b" * 64,
            "motion_name": "walk_stand",
            "motion_sha256": "c" * 64,
            "support_files": _support_files(),
        },
        "runtime": {
            "simulation_dt_seconds": 0.001,
            "control_dt_seconds": 0.02,
            "state_sampling": "pre_step",
        },
        "controller_abi": {
            "reference_quaternion_order": "xyzw",
            "reference_quaternions_normalized": False,
            "sensor_quaternion_order": "wxyz",
            "reference_offsets_control_steps": list(REFERENCE_OFFSETS),
            "history_frames": 20,
            "history_update": "append_current_proprioception_after_actor_call",
            "action_history": "raw_preclip_actor_output",
            "raw_action_clip": [-10.0, 10.0],
            "action_scale": 0.5,
            "pd_torque_recomputed_each_simulation_step": True,
        },
        "model_abi": json.loads(json.dumps(asdict(_abi()))),
        "trace": {
            "path": path.name,
            "sha256": digest,
            "size": path.stat().st_size,
            "arrays": {
                key: {"shape": list(value.shape), "dtype": value.dtype.str}
                for key, value in arrays.items()
            },
        },
        "metrics": {"simulated_seconds": arrays["sim_time"].shape[0] * 0.001},
        "limits": {"original_jit_executed": False, "jit_equivalence_tested": False},
    }


def test_trace_validation_is_data_only_and_checks_clocks(tmp_path: Path) -> None:
    assert "mujoco" not in sys.modules
    arrays = _trace_arrays()
    path = tmp_path / "trace.npz"
    digest = write_deterministic_npz(path, arrays)
    manifest = _manifest(path, digest, arrays)

    abi, contract = TRACE.validate_trace_manifest(
        manifest,
        trace_path=path,
        support_files=_support_files(),
    )
    loaded = TRACE.read_trace_arrays(path, expected_sha256=digest, contract=contract)

    assert abi == _abi()
    np.testing.assert_array_equal(loaded["sim_qpos"], arrays["sim_qpos"])
    assert "mujoco" not in sys.modules
    arrays["control_time"][0] = 0.01
    bad_path = tmp_path / "bad_trace.npz"
    bad_digest = write_deterministic_npz(bad_path, arrays)
    bad_manifest = _manifest(bad_path, bad_digest, arrays)
    _, bad_contract = TRACE.validate_trace_manifest(
        bad_manifest,
        trace_path=bad_path,
        support_files=_support_files(),
    )
    with pytest.raises(GMTAdmissionError, match="control-time clock"):
        TRACE.read_trace_arrays(
            bad_path,
            expected_sha256=bad_digest,
            contract=bad_contract,
        )


def test_manifest_rejects_mesh_identity_mismatch(tmp_path: Path) -> None:
    arrays = _trace_arrays()
    path = tmp_path / "trace.npz"
    digest = write_deterministic_npz(path, arrays)
    manifest = _manifest(path, digest, arrays)
    manifest["inputs"]["support_files"]["assets/robots/g1/meshes/pelvis.STL"] = "0" * 64

    with pytest.raises(GMTAdmissionError, match="model/mesh identities"):
        TRACE.validate_trace_manifest(
            manifest,
            trace_path=path,
            support_files=_support_files(),
        )


def test_manifest_reader_requires_exact_hash_and_rejects_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(GMTAdmissionError, match="duplicate manifest key"):
        TRACE.read_trace_manifest(path, expected_sha256=digest)
    with pytest.raises(GMTAdmissionError, match="SHA-256 mismatch"):
        TRACE.read_trace_manifest(path, expected_sha256="0" * 64)
