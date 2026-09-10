#!/usr/bin/env python3
"""Reproduce the bounded Study 012/014 residual and static-reference audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch

from oracle_composition.adapters.gmt import (
    checkpoint,
    contracts,
    control_runtime,
    motions,
    reference_math,
    reference_runtime,
    replay,
)
from oracle_composition.adapters.gmt import io as gmt_io

ARTIFACT_ROOT = Path(
    "/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/"
    "artifacts/gmt/2a590de25a1eb08e"
)
RUN_ROOT = Path("/Users/samueldoane/Documents/ChatGPT/humanoid-harness-probe-runs")
MOTION_SHA = "a67c364e7d0013c24f43096e194f5d8099b2d1ebcaa3db42767402a0924be964"
XML_SHA = "7013cd256c89796b2844613d24dda2a13410df7cd32ddd4d59b1741f85094304"
MESH_TREE_SHA = "d8366a1f0c1e64d47c3710dfe7fd01d136ee462fda77d3cb571ab7dcecc967f2"
SOURCE_HASHES = {
    "checkpoint.py": "02d30bdffbda5ffbd6b9bd907064537b7521ad8d4f592b3d2ac21e3abdfedfff",
    "contracts.py": "f574b2a8266d7b4529525f720ce901faa9924538a80288627c558f750ad92f31",
    "control_runtime.py": "271cbb52c8286d24a2fecca8815eb2d4e9b26e2049d8ad3d54153fa1b0444877",
    "io.py": "e24ba59fa255c8b47b1cc71831bd7c9b2784af4d0e83d4e24501ec13b139f8d3",
    "motions.py": "1482a73601162f91e4bba1c86561842c94de678871dad84c93b6c9076eb505be",
    "reference_math.py": "6f82ce49056fc9beacf1215c817f1f4f0e91733fc2eb056b16bb9fec04c49b0e",
    "reference_runtime.py": "19039aac4223a47c4131131304f266b2e87bcc90647947e933d5cc1ddb2b725f",
    "replay.py": "8927557818d20e26801d30f6504423172281619f0a32b6b9ca9dc1e46bd84b6b",
}
RUNS = (
    (
        "study012_finite_horizon_baseline",
        "gmt_course_intrinsic_horizon_candidate_20260907",
        "9bca23bf62ea386cd7f0492aa63626b6c2a219dc",
        {
            "gmt_probe_resource_receipt_v1.json": "75e67574fb797a576c49886192bc0f88753d44142e873dbb500d3ad96bb06f89",
            "course_run_manifest.json": "fa4fe2cf601eae318cce06b3704ae2206b92b1e99011e95d9dbd3a687a8ad502",
            "input_config.json": "0be730e48fc53c9e34e671138a49cb6cfd3f1a41b350cf2047d55c347a6600bf",
            "final_policy_frames.jsonl": "cbe402b87138793649ffb943e33e2b5c98869898515a98f22d5a482f2b8e9208",
            "final_policy_trajectory.npz": "cf6cfd44baf19a2702bffa82a6c8b8ffe594568bab1f5ba79e1aaddb54e2a496",
        },
    ),
    (
        "study014_finite_depth_reward_candidate",
        "gmt_course_finite_r4_candidate_20260907",
        "be359a2accd21b3a94727695a2f7837311a36aaa",
        {
            "gmt_probe_resource_receipt_v1.json": "cc5c3c26c943bf12490a8aa3447a18013f28935565b217bb539607f783b01959",
            "course_run_manifest.json": "c73c4e771a5bf9e3e1f4f9e28cb9c560467f5855b0323605ef512058deb0eae2",
            "input_config.json": "39dd6a045fe680676389a8b46bdfe146bd231ccdb6fc9811499d56a33ab2cb13",
            "final_policy_frames.jsonl": "a4769a49e2cbfb4b7b5729be4997cffc258ef40cb65385ee6fb17515141c35b7",
            "final_policy_trajectory.npz": "10cce8e82f45aa5d6a01bd0d614a1ce5ce1a2bf75650e5d9463d3c9fd95d2836",
        },
    ),
)
HIP_KNEE = (0, 1, 2, 3, 6, 7, 8, 9)
ALLOWED_FEET = {"left_ankle_roll_link", "right_ankle_roll_link"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_hash(path: Path, expected: str) -> None:
    if (observed := sha256(path)) != expected:
        raise ValueError(f"hash mismatch for {path}: {observed}")


def source_identity() -> dict[str, str]:
    modules = (
        checkpoint,
        contracts,
        control_runtime,
        gmt_io,
        motions,
        reference_math,
        reference_runtime,
        replay,
    )
    source_root = Path(__file__).resolve().parents[1] / "src"
    result = {}
    for module in modules:
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(source_root.resolve()):
            raise RuntimeError(f"module import escaped checkout: {path}")
        require_hash(path, SOURCE_HASHES[path.name])
        result[str(path.relative_to(source_root))] = SOURCE_HASHES[path.name]
    return result


def producer_identity(source_hashes: dict[str, str]) -> dict[str, str]:
    path = Path(__file__).resolve()
    repository_root = path.parents[1]
    producer_path = str(path.relative_to(repository_root))
    producer_sha = sha256(path)
    source_set = {**source_hashes, producer_path: producer_sha}
    source_set_sha = hashlib.sha256(
        json.dumps(source_set, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "path": producer_path,
        "sha256": producer_sha,
        "executed_source_set_sha256": source_set_sha,
    }


def read_run(root: Path, record: tuple[Any, ...]) -> dict[str, Any]:
    study, directory, commit, hashes = record
    path = root / directory
    for name, digest in hashes.items():
        require_hash(path / name, digest)
    receipt = json.loads((path / "gmt_probe_resource_receipt_v1.json").read_bytes())
    manifest = json.loads((path / "course_run_manifest.json").read_bytes())
    config = json.loads((path / "input_config.json").read_bytes())
    if receipt["status"] != "succeeded" or receipt["commit"] != commit:
        raise ValueError(f"{study}: receipt status/source mismatch")
    if manifest["input_config_sha256"] != hashes["input_config.json"]:
        raise ValueError(f"{study}: config binding mismatch")
    for name in ("final_policy_frames.jsonl", "final_policy_trajectory.npz"):
        if manifest["outputs"][name] != hashes[name]:
            raise ValueError(f"{study}: manifest output mismatch for {name}")
        if receipt["artifacts"]["outputs"][name]["sha256"] != hashes[name]:
            raise ValueError(f"{study}: receipt output mismatch for {name}")
    if config["assets"]["motions"]["crouchwalk_stand"]["sha256"] != MOTION_SHA:
        raise ValueError(f"{study}: motion binding mismatch")
    if manifest["frozen_runtime"]["residual_raw_scale"] != 0.25:
        raise ValueError(f"{study}: residual scale mismatch")
    frames = [
        json.loads(line)
        for line in (path / "final_policy_frames.jsonl").read_text().splitlines()
    ]
    with np.load(path / "final_policy_trajectory.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    expected = {
        "qpos": ((1001, 30), "<f8"),
        "qvel": ((1001, 29), "<f8"),
        "residual_action": ((1000, 23), "<f4"),
        "current_reference": ((1000, 30), "<f4"),
        "composite_raw_action": ((1000, 23), "<f4"),
    }
    if len(frames) != 1000 or set(arrays) != set(expected):
        raise ValueError(f"{study}: trajectory shape/member mismatch")
    for name, (shape, dtype) in expected.items():
        if arrays[name].shape != shape or arrays[name].dtype.str != dtype:
            raise ValueError(f"{study}: {name} shape/dtype mismatch")
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"{study}: nonfinite {name}")
    for index, frame in enumerate(frames):
        if frame["metrics"]["control_step"] != index + 1:
            raise ValueError(f"{study}: control sequence mismatch")
        for name in ("current_reference", "composite_raw_action"):
            if not np.array_equal(
                arrays[name][index], np.asarray(frame["trajectory"][name], dtype="<f4")
            ):
                raise ValueError(f"{study}: JSONL/NPZ mismatch at {index}: {name}")
    return {
        "study": study,
        "path": path,
        "commit": commit,
        "hashes": hashes,
        "frames": frames,
        "arrays": arrays,
        "config": config,
    }


def literal_qpos(frame: reference_runtime.MotionFrame) -> np.ndarray:
    xyzw = frame.root_rotation_xyzw[0].numpy().astype("<f8")
    return np.concatenate(
        (
            frame.root_position[0].numpy().astype("<f8"),
            xyzw[[3, 0, 1, 2]],
            frame.dof_position[0].numpy().astype("<f8"),
        )
    )


def feature_qpos(features: np.ndarray) -> np.ndarray:
    """Pose represented by consumed features: x/y/yaw ground-invariant and zero."""
    value = np.asarray(features, dtype="<f8")
    if value.shape != (30,):
        raise ValueError(f"expected one 30-D reference, observed {value.shape}")
    height, roll, pitch = value[:3]
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    quaternion_wxyz = np.asarray([cr * cp, sr * cp, cr * sp, -sr * sp])
    return np.concatenate(([0.0, 0.0, height], quaternion_wxyz, value[7:]))


def native_features(
    motion: reference_runtime.ReferenceMotion, euler: torch.Tensor, index: int
) -> np.ndarray:
    """Build only pose-bearing consumed features from one exact native row."""
    value = np.zeros(30, dtype="<f4")
    value[0] = motion.root_position[index, 2].item()
    value[1:3] = euler[index, :2].numpy()
    value[7:] = motion.dof_position[index].numpy()
    return value


def geometry(model: Any, data: Any, qpos: np.ndarray) -> dict[str, Any]:
    data.qpos[:] = qpos
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    planes = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).tolist()
    if planes != [0]:
        raise ValueError(f"expected sole ground plane geom 0, observed {planes}")
    grouped: dict[tuple[int, str], list[float]] = {}
    for contact in data.contact[: int(data.ncon)]:
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if 0 not in (geom1, geom2):
            continue
        geom = geom2 if geom1 == 0 else geom1
        body_id = int(model.geom_bodyid[geom])
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        grouped.setdefault((geom, body), []).append(float(contact.dist))
    rows = [
        [geom, body, "foot" if body in ALLOWED_FEET else "nonfoot", len(values), min(values)]
        for (geom, body), values in sorted(grouped.items())
    ]
    all_depths = [row[4] for row in rows]
    nonfoot = [row[4] for row in rows if row[2] == "nonfoot"]
    return {
        "root_height_m": float(qpos[2]),
        "quaternion_norm": float(np.linalg.norm(qpos[3:7])),
        "contact_columns": ["geom_id", "body", "class", "points", "minimum_dist_m"],
        "contacts": rows,
        "deepest_contact_dist_m": min(all_depths) if all_depths else None,
        "deepest_nonfoot_dist_m": min(nonfoot) if nonfoot else None,
        "nonfoot_bodies": sorted({row[1] for row in rows if row[2] == "nonfoot"}),
    }


def run_measurements(
    run: dict[str, Any], motion: reference_runtime.ReferenceMotion
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    frames, arrays = run["frames"], run["arrays"]
    progress = np.asarray([row["metrics"]["progress_m"] for row in frames])
    physical = (progress >= 1.0) & (progress < 2.0)
    retained_physical = np.asarray(
        [row["metrics"]["inside_posture_region"] for row in frames], dtype=bool
    )
    oracle = np.asarray([row["executed_mode"] == "inside" for row in frames])
    if not np.array_equal(physical, retained_physical) or not physical.any():
        raise ValueError(f"{run['study']}: physical-region evidence mismatch/missing")
    heights = np.asarray([row["metrics"]["root_height_m"] for row in frames])
    ref_heights = arrays["current_reference"][:, 0]
    physical_rows = np.flatnonzero(physical)
    actual_min = int(physical_rows[np.argmin(heights[physical])])
    reference_min = int(physical_rows[np.argmin(ref_heights[physical])])

    def sample(row: int) -> tuple[float, reference_runtime.MotionFrame, np.ndarray]:
        time = torch.tensor(
            [
                float(run["config"]["segments"]["crouch"]["start_seconds"])
                + float(frames[row]["executed_phase_seconds"])
                + float(contracts.CONTROL_DT_SECONDS)
            ],
            dtype=torch.float32,
        )
        features = motion.features(time)[0].numpy()
        if not np.array_equal(features, arrays["current_reference"][row]):
            raise ValueError(f"{run['study']}: reference does not reproduce at row {row}")
        return float(time[0]), motion.sample(time), features

    ref_time, ref_frame, ref_features = sample(reference_min)
    actual_time, actual_frame, actual_features = sample(actual_min)
    residual, composite = arrays["residual_action"], arrays["composite_raw_action"]
    raw_scale = float(control_runtime.PROVISIONAL_RESIDUAL_RAW_SCALE)
    base = (composite - np.float32(raw_scale) * residual).astype("<f4")
    offsets = residual.astype("<f8") * float(contracts.ACTION_SCALE) * raw_scale
    reconstructed = (
        np.clip(composite, contracts.RAW_ACTION_MIN, contracts.RAW_ACTION_MAX)
        - np.clip(base, contracts.RAW_ACTION_MIN, contracts.RAW_ACTION_MAX)
    ).astype("<f8") * float(contracts.ACTION_SCALE)
    joint_rows = []
    for index in HIP_KNEE:
        values = offsets[physical, index]
        joint_rows.append(
            [
                index,
                contracts.JOINT_NAMES[index],
                float(values.min()),
                float(values.max()),
                float(np.sqrt(np.mean(values**2))),
                float(offsets[actual_min, index]),
            ]
        )
    result = {
        "masks": {
            "oracle_inside_rows": [int(np.flatnonzero(oracle)[0]), int(np.flatnonzero(oracle)[-1] + 1)],
            "oracle_inside_count": int(oracle.sum()),
            "physical_region_rows": [int(physical_rows[0]), int(physical_rows[-1] + 1)],
            "physical_region_count": int(physical.sum()),
            "physical_outside_oracle_count": int(np.sum(physical & ~oracle)),
        },
        "timing": {
            "reference_min": [reference_min, reference_min + 1, ref_time, float(ref_heights[reference_min]), float(heights[reference_min])],
            "actual_min": [actual_min, actual_min + 1, actual_time, float(ref_heights[actual_min]), float(heights[actual_min])],
            "columns": ["row", "control_step", "source_time_s", "reference_z_m", "actual_z_m"],
        },
        "residual_authority": {
            "pd_offset_formula": "0.5 * 0.25 * residual_action radians",
            "saturation": "abs(residual_action) >= 0.99",
            "oracle_saturation_counts_23": np.sum(np.abs(residual[oracle]) >= 0.99, axis=0).tolist(),
            "physical_saturation_counts_23": np.sum(np.abs(residual[physical]) >= 0.99, axis=0).tolist(),
            "whole_max_abs_residual": float(np.max(np.abs(residual))),
            "physical_max_abs_residual": float(np.max(np.abs(residual[physical]))),
            "physical_all_joint_offset_rms_rad": float(np.sqrt(np.mean(offsets[physical] ** 2))),
            "base_raw_max_abs": float(np.max(np.abs(base))),
            "composite_raw_max_abs": float(np.max(np.abs(composite))),
            "formula_reconstruction_max_error_rad": float(np.max(np.abs(offsets - reconstructed))),
            "hip_knee_columns": ["index", "name", "physical_min_rad", "physical_max_rad", "physical_rms_rad", "at_actual_min_rad"],
            "hip_knee": joint_rows,
        },
        "actual_geometry": {
            "at_reference_min": None,
            "at_actual_min": None,
        },
    }
    poses = {
        "literal_at_reference_min": literal_qpos(ref_frame),
        "feature_at_reference_min": feature_qpos(ref_features),
        "literal_at_actual_min_time": literal_qpos(actual_frame),
        "feature_at_actual_min_time": feature_qpos(actual_features),
        "actual_at_reference_min": arrays["qpos"][reference_min + 1],
        "actual_at_actual_min": arrays["qpos"][actual_min + 1],
    }
    return result, poses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    args = parser.parse_args()
    source_hashes = source_identity()
    motion_path = args.artifact_root / "numeric/motions/crouchwalk_stand.npz"
    xml_path = args.artifact_root / "upstream/assets/robots/g1/g1.xml"
    require_hash(motion_path, MOTION_SHA)
    require_hash(xml_path, XML_SHA)
    support = checkpoint.verify_upstream_root(args.artifact_root / "upstream")
    if support["assets/robots/g1/meshes@tree"] != MESH_TREE_SHA:
        raise ValueError("mesh-tree identity mismatch")
    motion = reference_runtime.ReferenceMotion.from_converted(
        motion_path, name="crouchwalk_stand", expected_sha256=MOTION_SHA
    )
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    replay.validate_model_abi(replay._extract_model_abi(mujoco, model))
    data = mujoco.MjData(model)
    runs = [read_run(args.run_root, record) for record in RUNS]
    results: dict[str, Any] = {}
    shared: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        result, poses = run_measurements(run, motion)
        contacts = {name: geometry(model, data, qpos) for name, qpos in poses.items()}
        result["actual_geometry"]["at_reference_min"] = contacts.pop("actual_at_reference_min")
        result["actual_geometry"]["at_actual_min"] = contacts.pop("actual_at_actual_min")
        for name, contact in contacts.items():
            shared.setdefault(name, []).append(contact)
        results[run["study"]] = result
    if any(values[0] != values[1] for values in shared.values()):
        raise ValueError("runs resolve to different shared reference poses")

    native_euler = reference_math.quaternion_to_euler_xyzw(motion.root_rotation_xyzw)
    center = int(torch.argmin(motion.root_position[:, 2]))
    neighbors = []
    for index in range(center - 4, center + 5):
        frame = reference_runtime.MotionFrame(
            root_position=motion.root_position[index : index + 1],
            root_rotation_xyzw=motion.root_rotation_xyzw[index : index + 1],
            root_velocity=motion.root_velocity[index : index + 1],
            root_angular_velocity=motion.root_angular_velocity[index : index + 1],
            dof_position=motion.dof_position[index : index + 1],
            dof_velocity=motion.dof_velocity[index : index + 1],
        )
        literal = geometry(model, data, literal_qpos(frame))
        consumed = geometry(
            model, data, feature_qpos(native_features(motion, native_euler, index))
        )
        neighbors.append(
            [
                index,
                index / float(motion.fps),
                literal["deepest_contact_dist_m"],
                literal["deepest_nonfoot_dist_m"],
                consumed["deepest_contact_dist_m"],
                consumed["deepest_nonfoot_dist_m"],
            ]
        )
    if any(row[5] is None or row[5] >= 0 for row in neighbors):
        raise ValueError("consumed-feature nonfoot overlap did not persist across neighbors")

    low_rows = np.flatnonzero(motion.root_position[:, 2].numpy() <= 0.50).tolist()
    low_geometry = [
        geometry(model, data, feature_qpos(native_features(motion, native_euler, index)))
        for index in low_rows
    ]
    penetrations = [-float(item["deepest_contact_dist_m"]) for item in low_geometry]
    required_heights = [
        float(motion.root_position[index, 2]) + penetration
        for index, penetration in zip(low_rows, penetrations, strict=True)
    ]
    low_segment = {
        "selection": "exact native rows with source root_z <= 0.50 m",
        "native_frame_indices": low_rows,
        "sample_count": len(low_rows),
        "consumed_pose": "native root_z and exact joints; source quaternion-derived roll/pitch converted to unit wxyz at yaw 0",
        "floor_penetration_over_0_01_m_count": int(
            sum(value > 0.01 for value in penetrations)
        ),
        "floor_penetration_m_min_max": [min(penetrations), max(penetrations)],
        "nonfoot_overlap_count": int(sum(bool(item["nonfoot_bodies"]) for item in low_geometry)),
        "translation_only_required_root_height_m_min_max": [
            min(required_heights),
            max(required_heights),
        ],
        "translation_only_semantics": "source root_z minus deepest signed plane distance, preserving roll/pitch and joints; descriptive only, not a repair or feasibility bound",
    }

    residual_saturation = any(
        result["residual_authority"]["whole_max_abs_residual"] >= 0.99
        for result in results.values()
    )
    literal_overlap = any(
        shared[name][0]["deepest_nonfoot_dist_m"] is not None
        and shared[name][0]["deepest_nonfoot_dist_m"] < 0
        for name in ("literal_at_reference_min", "literal_at_actual_min_time")
    )
    feature_overlap = any(
        shared[name][0]["deepest_nonfoot_dist_m"] is not None
        and shared[name][0]["deepest_nonfoot_dist_m"] < 0
        for name in ("feature_at_reference_min", "feature_at_actual_min_time")
    )
    selected_actual = [
        result["actual_geometry"][name]
        for result in results.values()
        for name in ("at_reference_min", "at_actual_min")
    ]
    selected_actual_overlap = any(
        item["deepest_nonfoot_dist_m"] is not None
        and item["deepest_nonfoot_dist_m"] < 0
        for item in selected_actual
    )
    if residual_saturation:
        raise ValueError("residual saturation observation changed")

    output = {
        "schema_version": 1,
        "evidence_class": "gmt_g1_residual_reference_geometry_audit/v1",
        "claim_ceiling": "static_mj_forward_and_recorded_arrays_only; not_dynamics_feasibility_policy_intention_training_exposure_or_causal_attribution",
        "method": {
            "used": [
                "strict hash-bound numeric reads",
                "ReferenceMotion sample/features at retained source times",
                "literal pose: original xyz + original xyzw reordered wxyz + exact 23 joints",
                "consumed pose: feature height + feature roll/pitch as unit wxyz yaw0 + exact feature joints",
                "mj_forward with qvel zero",
                "algebraic base action and PD target-offset reconstruction",
            ],
            "not_used": ["mj_step", "actor load", "policy prediction", "training"],
            "contact_semantics": "negative distance is static overlap with model ground plane",
        },
        "inputs": {
            "motion": [str(motion_path), MOTION_SHA],
            "model_xml": [str(xml_path), XML_SHA],
            "mesh_tree_sha256": MESH_TREE_SHA,
            "source_sha256": source_hashes,
            "producer": producer_identity(source_hashes),
            "runs": [
                {
                    "study": run["study"],
                    "path": str(run["path"]),
                    "source_commit": run["commit"],
                    "sha256": run["hashes"],
                }
                for run in runs
            ],
        },
        "runtime": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "mujoco": mujoco.__version__,
            "motion_fps": float(motion.fps),
            "motion_frames": motion.frame_count,
            "literal_quaternion_normalized_before_mj_forward": False,
            "consumed_pose_yaw_and_xy": "zero; invariant for infinite horizontal plane",
        },
        "shared_reference_geometry": {name: values[0] for name, values in shared.items()},
        "neighbor_columns": ["native_frame", "source_time_s", "literal_deepest_m", "literal_nonfoot_m", "consumed_deepest_m", "consumed_nonfoot_m"],
        "neighboring_native_frames": neighbors,
        "native_low_root_segment": low_segment,
        "runs": results,
        "observations": {
            "residual_saturation_observed": residual_saturation,
            "nonfoot_overlap_literal_selected_reference_poses_observed": literal_overlap,
            "nonfoot_overlap_consumed_selected_reference_poses_observed": feature_overlap,
            "nonfoot_overlap_selected_matched_actual_qpos_observed": selected_actual_overlap,
            "selected_matched_actual_qpos_count": len(selected_actual),
            "interpretation": "The issued height, roll/pitch, and joint reference is statically ground-inconsistent near the retained minimum; this is a rival explanation for the height gap, not proof of dynamic or causal contribution.",
        },
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
