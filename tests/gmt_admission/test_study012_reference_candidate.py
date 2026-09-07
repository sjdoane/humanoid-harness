from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from oracle_composition.adapters.gmt.io import GMTAdmissionError

SCRIPT = Path(__file__).parents[2] / "scripts/convert_study012_reference_candidate.py"
SPEC = importlib.util.spec_from_file_location("study012_reference_candidate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
candidate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(candidate)


def _quaternion_wxyz(*, roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.asarray(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype="<f8",
    )


def _qpos() -> np.ndarray:
    value = np.zeros((1001, 30), dtype="<f8")
    yaw = 0.4
    forward = np.asarray([math.cos(yaw), math.sin(yaw)])
    lateral = np.asarray([-forward[1], forward[0]])
    for index in range(len(value)):
        value[index, :2] = forward * (index * 0.01) + lateral * (index * -0.001)
        value[index, 2] = 0.8 - index * 0.0001
        value[index, 3:7] = _quaternion_wxyz(
            roll=0.1 + index * 1e-5,
            pitch=-0.2 + index * 2e-5,
            yaw=yaw + index * 1e-5,
        )
        value[index, 7:] = np.linspace(-0.2, 0.3, 23) + index * 1e-4
    return value


def _frames(qpos: np.ndarray) -> list[dict]:
    frames = []
    for index in range(1000):
        mode = "before" if index < 92 else "inside" if index < 197 else "after"
        transition = None
        if index == 92:
            transition = {
                "from_state": "before",
                "to_state": "inside",
                "control_step": 92,
            }
        elif index == 197:
            transition = {
                "from_state": "inside",
                "to_state": "after",
                "control_step": 197,
            }
        frames.append(
            {
                "executed_mode": mode,
                "transition": transition,
                "metrics": {"control_step": index + 1},
                "trajectory": {
                    "qpos": qpos[index + 1].tolist(),
                    "geom_body_names": [
                        "world",
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                        "torso_link",
                    ],
                    "contact_pairs": [
                        [[0, 1]] if 92 <= index < 197 else []
                        for _ in range(20)
                    ],
                },
            }
        )
    return frames


def test_exact_slice_rebases_progress_and_preserves_pose_values() -> None:
    qpos = _qpos()
    arrays, measurements = candidate.build_candidate_arrays(qpos)

    assert set(arrays) == {"fps", "root_pos", "root_rot", "dof_pos"}
    assert arrays["fps"].shape == (1,)
    assert arrays["fps"].dtype.str == "<f8"
    assert arrays["fps"].item() == 50.0
    assert arrays["root_pos"].shape == (106, 3)
    assert arrays["root_pos"].dtype.str == "<f4"
    assert arrays["root_rot"].shape == (106, 4)
    assert arrays["dof_pos"].shape == (106, 23)
    assert np.array_equal(arrays["root_pos"][:, 1], np.zeros(106, dtype="<f4"))
    assert arrays["root_pos"][0, 0] == 0.0
    assert arrays["root_pos"][-1, 0] == np.float32(1.05)
    assert np.array_equal(arrays["root_pos"][:, 2], qpos[92:198, 2].astype("<f4"))
    assert np.array_equal(arrays["dof_pos"], qpos[92:198, 7:30].astype("<f4"))
    assert np.allclose(np.linalg.norm(arrays["root_rot"], axis=1), 1.0, atol=1e-6)
    roll, pitch = 0.10092, -0.19816
    expected_xyzw = np.asarray(
        [
            math.sin(roll / 2) * math.cos(pitch / 2),
            math.cos(roll / 2) * math.sin(pitch / 2),
            -math.sin(roll / 2) * math.sin(pitch / 2),
            math.cos(roll / 2) * math.cos(pitch / 2),
        ],
        dtype="<f4",
    )
    assert np.array_equal(arrays["root_rot"][0], expected_xyzw)
    assert measurements["source_lateral_delta_m"] == pytest.approx(-0.105)
    assert measurements["duration_seconds"] == 2.1


@pytest.mark.parametrize("mutation", ["shape", "nonfinite", "quaternion", "backtrack"])
def test_conversion_rejects_malformed_or_backtracking_source(mutation: str) -> None:
    qpos = _qpos()
    if mutation == "shape":
        qpos = qpos[:-1]
    elif mutation == "nonfinite":
        qpos[100, 2] = np.nan
    elif mutation == "quaternion":
        qpos[100, 3:7] *= 2
    else:
        qpos[100, :2] = qpos[99, :2]
    with pytest.raises(GMTAdmissionError):
        candidate.build_candidate_arrays(qpos)


def test_frame_chain_and_all_selected_substeps_are_validated() -> None:
    qpos = _qpos()
    result = candidate.validate_frames(_frames(qpos), qpos)
    assert result == {
        "frame_intervals_half_open": [92, 197],
        "interval_count": 105,
        "simulator_substep_count": 2100,
        "ground_contact_bodies": ["left_ankle_roll_link"],
        "nonfoot_ground_contact_observed": False,
    }


def test_frame_chain_rejects_nonfoot_contact_and_changed_qpos() -> None:
    qpos = _qpos()
    frames = _frames(qpos)
    frames[110]["trajectory"]["contact_pairs"][7] = [[0, 3]]
    with pytest.raises(GMTAdmissionError, match="non-foot"):
        candidate.validate_frames(frames, qpos)

    frames = _frames(qpos)
    frames[500]["trajectory"]["qpos"][2] += 0.1
    with pytest.raises(GMTAdmissionError, match="frame/trajectory"):
        candidate.validate_frames(frames, qpos)


def test_verified_json_rejects_changed_bytes(tmp_path: Path, monkeypatch) -> None:
    payload = b'{"status":"succeeded"}\n'
    path = tmp_path / "input_config.json"
    path.write_bytes(payload)
    monkeypatch.setitem(
        candidate.DONOR_FILES,
        "input_config.json",
        (candidate._sha256_bytes(payload), len(payload)),
    )
    assert candidate._verified_json(tmp_path, "input_config.json")["status"] == "succeeded"
    path.write_bytes(payload + b" ")
    with pytest.raises(GMTAdmissionError):
        candidate._verified_json(tmp_path, "input_config.json")


def test_source_imports_are_exact_and_output_overwrite_fails_before_reads(tmp_path: Path) -> None:
    source = candidate.source_identity()
    assert set(source["imports"]) == set(candidate.SOURCE_BINDINGS)
    output = tmp_path / "candidate.npz"
    output.write_bytes(b"retained")
    with pytest.raises(GMTAdmissionError, match="overwrite"):
        candidate.convert(donor_root=tmp_path, artifact_root=tmp_path, output=output)
    assert output.read_bytes() == b"retained"


@pytest.mark.skipif(
    not candidate.DONOR_ROOT.is_dir() or not candidate.ARTIFACT_ROOT.is_dir(),
    reason="exact retained Study 012 donor and GMT artifact root are unavailable",
)
def test_exact_retained_donor_and_mj_forward_geometry() -> None:
    pytest.importorskip("mujoco")
    qpos, donor = candidate.validate_donor(candidate.DONOR_ROOT)
    arrays, measurements = candidate.build_candidate_arrays(qpos)
    candidate.validate_pinned_measurements(measurements)
    xml_path, model = candidate.verify_model_assets(candidate.ARTIFACT_ROOT)
    geometry = candidate.validate_static_geometry(xml_path, arrays)

    assert donor["execution_contact_evidence"]["simulator_substep_count"] == 2100
    assert model["mesh_tree_sha256"] == candidate.MESH_TREE_SHA256
    assert geometry["evaluated_pose_count"] == 106
    assert geometry["nonfoot_ground_contact_pose_count"] == 0
    assert geometry["maximum_allowed_foot_penetration_m"] == pytest.approx(
        0.01543102558135373
    )
    assert measurements["source_root_height_m_start_min_end"] == pytest.approx(
        [0.7785659884189685, 0.504942203119977, 0.5954433356039784]
    )


def test_array_identity_hashes_exact_values_not_shape_only() -> None:
    arrays, _ = candidate.build_candidate_arrays(_qpos())
    before = candidate._array_sha256(arrays["dof_pos"])
    changed = arrays["dof_pos"].copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.float32(np.inf))
    assert candidate._array_sha256(changed) != before
    assert json.dumps(
        {
            key: candidate._array_sha256(value)
            for key, value in sorted(arrays.items())
        },
        sort_keys=True,
    )
