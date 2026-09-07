#!/usr/bin/env python3
"""Convert the exact Study 012 executed inside passage into a numeric motion candidate.

This is a bounded, hash-pinned kinematic conversion.  It never runs the actor,
steps MuJoCo, admits the result for use, or claims a dynamics certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from oracle_composition.adapters.gmt import contracts, course_task, reference_math
from oracle_composition.adapters.gmt import io as gmt_io

DONOR_ROOT = Path(
    "/Users/samueldoane/Documents/ChatGPT/humanoid-harness-probe-runs/"
    "gmt_course_intrinsic_horizon_candidate_20260907"
)
ARTIFACT_ROOT = Path(
    "/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/"
    "artifacts/gmt/2a590de25a1eb08e"
)
CANDIDATE_ID = "gmt_g1_study012_executed_inside_passage/v1"
DONOR_COMMIT = "9bca23bf62ea386cd7f0492aa63626b6c2a219dc"
DONOR_TREE_SHA256 = "3ba7fe0ab72374e0667402848460c43915024fbcd2fe44f9ce5de791f0ab8b8e"
DONOR_FILES = {
    "gmt_probe_resource_receipt_v1.json": (
        "75e67574fb797a576c49886192bc0f88753d44142e873dbb500d3ad96bb06f89",
        15_091,
    ),
    "course_run_manifest.json": (
        "fa4fe2cf601eae318cce06b3704ae2206b92b1e99011e95d9dbd3a687a8ad502",
        19_970,
    ),
    "input_config.json": (
        "0be730e48fc53c9e34e671138a49cb6cfd3f1a41b350cf2047d55c347a6600bf",
        2_412,
    ),
    "final_policy_frames.jsonl": (
        "cbe402b87138793649ffb943e33e2b5c98869898515a98f22d5a482f2b8e9208",
        5_855_730,
    ),
    "final_policy_trajectory.npz": (
        "cf6cfd44baf19a2702bffa82a6c8b8ffe594568bab1f5ba79e1aaddb54e2a496",
        777_674,
    ),
}
MODEL_XML_SHA256 = "7013cd256c89796b2844613d24dda2a13410df7cd32ddd4d59b1741f85094304"
MODEL_XML_SIZE = 26_860
MESH_TREE_SHA256 = "d8366a1f0c1e64d47c3710dfe7fd01d136ee462fda77d3cb571ab7dcecc967f2"
SOURCE_BINDINGS = {
    "oracle_composition/__init__.py": (
        "oracle_composition",
        "e0ebdc0fbf4c9e8a17ffd65b2c16a8c9bdc70b304ba344670a71febdf3fb0505",
    ),
    "oracle_composition/adapters/gmt/__init__.py": (
        "oracle_composition.adapters.gmt",
        "f53ab7fe48c1b5f1ff28183dce225a334068294b1698fa7f9731e595fc039426",
    ),
    "oracle_composition/adapters/gmt/checkpoint.py": (
        "oracle_composition.adapters.gmt.checkpoint",
        "02d30bdffbda5ffbd6b9bd907064537b7521ad8d4f592b3d2ac21e3abdfedfff",
    ),
    "oracle_composition/adapters/gmt/contracts.py": (
        "oracle_composition.adapters.gmt.contracts",
        "f574b2a8266d7b4529525f720ce901faa9924538a80288627c558f750ad92f31"
    ),
    "oracle_composition/adapters/gmt/course_task.py": (
        "oracle_composition.adapters.gmt.course_task",
        "9865fe86c54aeb4b27bc1256fbde160423feec2a44d639b036bb6db8469da66a"
    ),
    "oracle_composition/adapters/gmt/deployment.py": (
        "oracle_composition.adapters.gmt.deployment",
        "7bc3e173cf4922212a6a5e080fd8b60a3cfa49434ff11ba14855bf2a89db8b46"
    ),
    "oracle_composition/adapters/gmt/io.py": (
        "oracle_composition.adapters.gmt.io",
        "e24ba59fa255c8b47b1cc71831bd7c9b2784af4d0e83d4e24501ec13b139f8d3"
    ),
    "oracle_composition/adapters/gmt/motions.py": (
        "oracle_composition.adapters.gmt.motions",
        "1482a73601162f91e4bba1c86561842c94de678871dad84c93b6c9076eb505be",
    ),
    "oracle_composition/adapters/gmt/reference_math.py": (
        "oracle_composition.adapters.gmt.reference_math",
        "6f82ce49056fc9beacf1215c817f1f4f0e91733fc2eb056b16bb9fec04c49b0e"
    ),
    "oracle_composition/contracts/__init__.py": (
        "oracle_composition.contracts",
        "97ea46cca87aeb7678f9588d466b46d7102ddd0846a6315bfdb8908616825c18",
    ),
    "oracle_composition/contracts/errors.py": (
        "oracle_composition.contracts.errors",
        "001e66fb6c3ca9c4ecd38b9b602f2e9525938527784891b4ed86c76f88e8843a",
    ),
    "oracle_composition/contracts/program.py": (
        "oracle_composition.contracts.program",
        "3067751b380ade571ef567cd07b926f2560e0a5578a8c4e9630ca34623a9e490",
    ),
    "oracle_composition/contracts/reference.py": (
        "oracle_composition.contracts.reference",
        "ec7b64ffe7b5b275071186791d27c5b5eb9be39c0a46dfa15a9000c8fba63aeb",
    ),
    "oracle_composition/contracts/reference_identity_v2.py": (
        "oracle_composition.contracts.reference_identity_v2",
        "c64fbe08cad98da57977767214c00f4f4dcbda83420a6113703d2b254016e4c3"
    ),
}
TRAJECTORY_MEMBERS = {
    "composite_raw_action.npy": 92_128,
    "current_reference.npy": 120_128,
    "qpos.npy": 240_368,
    "qvel.npy": 232_360,
    "residual_action.npy": 92_128,
}
SOURCE_QPOS_HALF_OPEN = (92, 198)
SOURCE_FRAME_INTERVALS_HALF_OPEN = (92, 197)
FRAME_COUNT = 106
INTERVAL_COUNT = 105
FPS_HZ = 50.0
MAX_ALLOWED_FOOT_PENETRATION_M = 0.01544
ALLOWED_FEET = course_task.ALLOWED_GROUND_CONTACT_BODIES


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_sha256(value: object) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    )


def _array_sha256(value: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(value).tobytes())


def source_identity() -> dict[str, object]:
    """Fail if any relied-upon project module resolves outside or differs from this source."""

    source_root = Path(__file__).resolve().parents[1] / "src"
    observed: dict[str, str] = {}
    for relative, (module_name, expected_digest) in SOURCE_BINDINGS.items():
        module = sys.modules.get(module_name)
        if module is None or getattr(module, "__file__", None) is None:
            raise gmt_io.GMTAdmissionError(f"expected source module was not imported: {module_name}")
        path = Path(module.__file__).resolve()
        expected_path = (source_root / relative).resolve()
        if path != expected_path or not path.is_relative_to(source_root.resolve()):
            raise gmt_io.GMTAdmissionError(f"project import escaped exact source root: {path}")
        digest = gmt_io.sha256_file(path)
        if digest != expected_digest:
            raise gmt_io.GMTAdmissionError(f"source digest mismatch: {relative}")
        observed[relative] = digest
    producer = Path(__file__).resolve()
    repository_root = producer.parents[1]
    producer_relative = str(producer.relative_to(repository_root))
    producer_sha256 = gmt_io.sha256_file(producer)
    executed = {**observed, producer_relative: producer_sha256}
    return {
        "imports": observed,
        "producer": {"path": producer_relative, "sha256": producer_sha256},
        "executed_source_set_sha256": _canonical_sha256(executed),
        "import_containment": "all listed project imports resolved beneath producer checkout src",
    }


def _verified_json(root: Path, name: str) -> dict[str, Any]:
    expected_sha256, expected_size = DONOR_FILES[name]
    payload = gmt_io.read_verified_bytes(
        root / name,
        expected_sha256,
        expected_size=expected_size,
        maximum_size=max(expected_size, 16 * 1024),
    )
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise gmt_io.GMTAdmissionError(f"invalid pinned JSON: {name}") from exc
    if type(value) is not dict:
        raise gmt_io.GMTAdmissionError(f"pinned JSON must contain one object: {name}")
    return value


def _load_qpos(root: Path) -> np.ndarray:
    name = "final_policy_trajectory.npz"
    expected_sha256, expected_size = DONOR_FILES[name]
    payload = gmt_io.read_verified_bytes(
        root / name,
        expected_sha256,
        expected_size=expected_size,
        maximum_size=1024 * 1024,
    )
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise gmt_io.GMTAdmissionError("invalid pinned trajectory NPZ") from exc
    with archive:
        members = gmt_io.validate_zip_members(
            archive,
            expected_count=len(TRAJECTORY_MEMBERS),
            expected_uncompressed_size=sum(TRAJECTORY_MEMBERS.values()),
            maximum_member_size=256 * 1024,
        )
        if set(members) != set(TRAJECTORY_MEMBERS):
            raise gmt_io.GMTAdmissionError("trajectory member set differs")
        for name, size in TRAJECTORY_MEMBERS.items():
            if members[name].file_size != size or members[name].compress_type != zipfile.ZIP_STORED:
                raise gmt_io.GMTAdmissionError(f"trajectory member envelope differs: {name}")
        qpos_payload = archive.read("qpos.npy")
    stream = io.BytesIO(qpos_payload)
    try:
        qpos = np.lib.format.read_array(stream, allow_pickle=False, max_header_size=512)
    except (ValueError, EOFError) as exc:
        raise gmt_io.GMTAdmissionError("invalid trajectory qpos member") from exc
    if stream.tell() != len(qpos_payload):
        raise gmt_io.GMTAdmissionError("trailing bytes in trajectory qpos member")
    if qpos.shape != (1001, 30) or qpos.dtype.str != "<f8" or not qpos.flags.c_contiguous:
        raise gmt_io.GMTAdmissionError("trajectory qpos shape/dtype/layout differs")
    if not np.isfinite(qpos).all():
        raise gmt_io.GMTAdmissionError("trajectory qpos contains non-finite values")
    return qpos


def _load_frames(root: Path) -> list[dict[str, Any]]:
    name = "final_policy_frames.jsonl"
    expected_sha256, expected_size = DONOR_FILES[name]
    payload = gmt_io.read_verified_bytes(
        root / name,
        expected_sha256,
        expected_size=expected_size,
        maximum_size=6 * 1024 * 1024,
    )
    lines = payload.splitlines()
    if len(lines) != 1000 or any(len(line) > 8192 for line in lines):
        raise gmt_io.GMTAdmissionError("frame count or bounded line size differs")
    try:
        frames = [json.loads(line) for line in lines]
    except json.JSONDecodeError as exc:
        raise gmt_io.GMTAdmissionError("invalid frame JSONL") from exc
    if any(type(frame) is not dict for frame in frames):
        raise gmt_io.GMTAdmissionError("frame JSONL rows must be objects")
    return frames


def _ground_body_names(pairs: object, geom_body_names: object) -> tuple[str, ...]:
    if type(geom_body_names) is not list or not geom_body_names or any(
        type(name) is not str or not name for name in geom_body_names
    ):
        raise gmt_io.GMTAdmissionError("invalid trajectory geom-body names")
    if geom_body_names[0] != "world" or type(pairs) is not list:
        raise gmt_io.GMTAdmissionError("ground geom/body trace contract differs")
    bodies: set[str] = set()
    for pair in pairs:
        if (
            type(pair) is not list
            or len(pair) != 2
            or any(type(index) is not int for index in pair)
            or any(index < 0 or index >= len(geom_body_names) for index in pair)
            or pair[0] == pair[1]
        ):
            raise gmt_io.GMTAdmissionError("invalid trajectory contact pair")
        if 0 in pair:
            other = pair[1] if pair[0] == 0 else pair[0]
            bodies.add(geom_body_names[other])
    return tuple(sorted(bodies))


def validate_frames(frames: list[dict[str, Any]], qpos: np.ndarray) -> dict[str, object]:
    if len(frames) != 1000 or qpos.shape != (1001, 30):
        raise gmt_io.GMTAdmissionError("retained frame/trajectory length differs")
    for index, frame in enumerate(frames):
        try:
            control_step = frame["metrics"]["control_step"]
            frame_qpos = np.asarray(frame["trajectory"]["qpos"], dtype="<f8")
        except (KeyError, TypeError, ValueError) as exc:
            raise gmt_io.GMTAdmissionError(f"malformed retained frame row: {index}") from exc
        if control_step != index + 1 or frame_qpos.shape != (30,):
            raise gmt_io.GMTAdmissionError(f"retained frame sequence differs at row {index}")
        if not np.array_equal(frame_qpos, qpos[index + 1]):
            raise gmt_io.GMTAdmissionError(f"frame/trajectory qpos differs at row {index}")

    start, stop = SOURCE_FRAME_INTERVALS_HALF_OPEN
    if any(frame.get("executed_mode") != "inside" for frame in frames[start:stop]):
        raise gmt_io.GMTAdmissionError("selected donor intervals are not all executed inside mode")
    entry = frames[start].get("transition")
    exit_transition = frames[stop].get("transition")
    if (
        frames[start - 1].get("executed_mode") != "before"
        or type(entry) is not dict
        or (entry.get("from_state"), entry.get("to_state"), entry.get("control_step"))
        != ("before", "inside", start)
        or frames[stop].get("executed_mode") != "after"
        or type(exit_transition) is not dict
        or (
            exit_transition.get("from_state"),
            exit_transition.get("to_state"),
            exit_transition.get("control_step"),
        )
        != ("inside", "after", stop)
    ):
        raise gmt_io.GMTAdmissionError("selected donor transition boundaries differ")

    bodies: set[str] = set()
    substeps = 0
    for index in range(start, stop):
        trajectory = frames[index]["trajectory"]
        contact_rows = trajectory.get("contact_pairs")
        if type(contact_rows) is not list or len(contact_rows) != contracts.SIMULATION_DECIMATION:
            raise gmt_io.GMTAdmissionError(f"donor substep-contact count differs at row {index}")
        for pairs in contact_rows:
            bodies.update(_ground_body_names(pairs, trajectory.get("geom_body_names")))
            substeps += 1
    if not bodies.issubset(ALLOWED_FEET):
        raise gmt_io.GMTAdmissionError(f"donor interval has non-foot ground contact: {sorted(bodies)}")
    if substeps != INTERVAL_COUNT * contracts.SIMULATION_DECIMATION:
        raise gmt_io.GMTAdmissionError("donor substep count differs")
    return {
        "frame_intervals_half_open": list(SOURCE_FRAME_INTERVALS_HALF_OPEN),
        "interval_count": INTERVAL_COUNT,
        "simulator_substep_count": substeps,
        "ground_contact_bodies": sorted(bodies),
        "nonfoot_ground_contact_observed": False,
    }


def validate_donor(root: Path) -> tuple[np.ndarray, dict[str, object]]:
    receipt = _verified_json(root, "gmt_probe_resource_receipt_v1.json")
    manifest = _verified_json(root, "course_run_manifest.json")
    config = _verified_json(root, "input_config.json")
    if receipt.get("status") != "succeeded" or receipt.get("commit") != DONOR_COMMIT:
        raise gmt_io.GMTAdmissionError("donor resource receipt status/source differs")
    if receipt.get("inputs", {}).get("repository_sources", {}).get(
        "canonical_tree_sha256"
    ) != DONOR_TREE_SHA256:
        raise gmt_io.GMTAdmissionError("donor executable source-tree identity differs")
    artifacts = receipt.get("artifacts", {})
    if artifacts.get("course_manifest", {}).get("sha256") != DONOR_FILES[
        "course_run_manifest.json"
    ][0]:
        raise gmt_io.GMTAdmissionError("donor receipt does not bind the course manifest")
    for name in ("input_config.json", "final_policy_frames.jsonl", "final_policy_trajectory.npz"):
        expected_sha256, expected_size = DONOR_FILES[name]
        record = artifacts.get("outputs", {}).get(name, {})
        if record.get("sha256") != expected_sha256 or record.get("size") != expected_size:
            raise gmt_io.GMTAdmissionError(f"donor resource output binding differs: {name}")
    if (
        manifest.get("status") != "completed"
        or manifest.get("input_config_sha256") != DONOR_FILES["input_config.json"][0]
        or any(
            manifest.get("outputs", {}).get(name) != DONOR_FILES[name][0]
            for name in ("final_policy_frames.jsonl", "final_policy_trajectory.npz")
        )
    ):
        raise gmt_io.GMTAdmissionError("donor course manifest linkage differs")
    if (
        config.get("mode") != "train"
        or config.get("runtime", {}).get("profile_id")
        != "gmt_g1_three_state_finite_horizon_course/v1"
    ):
        raise gmt_io.GMTAdmissionError("donor config semantics differ")
    qpos = _load_qpos(root)
    contact = validate_frames(_load_frames(root), qpos)
    return qpos, {
        "source_commit": DONOR_COMMIT,
        "executable_tree_sha256": DONOR_TREE_SHA256,
        "files": {
            name: {"sha256": digest, "size": size}
            for name, (digest, size) in DONOR_FILES.items()
        },
        "execution_contact_evidence": contact,
    }


def build_candidate_arrays(qpos: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    value = np.asarray(qpos)
    if value.shape != (1001, 30) or not np.issubdtype(value.dtype, np.floating):
        raise gmt_io.GMTAdmissionError("source qpos must be floating point with shape (1001, 30)")
    if not np.isfinite(value).all():
        raise gmt_io.GMTAdmissionError("source qpos contains non-finite values")
    norms = np.linalg.norm(value[:, 3:7], axis=1)
    if not np.allclose(norms, 1.0, atol=1e-6, rtol=0):
        raise gmt_io.GMTAdmissionError("source qpos includes a non-unit world quaternion")

    selected = value[slice(*SOURCE_QPOS_HALF_OPEN)]
    frame = course_task.TaskFrame.initialize(value[0, :2], value[0, 3:7])
    source_progress = np.asarray(
        [frame.project(row[:2], row[3:7]).progress_m for row in selected], dtype="<f8"
    )
    source_lateral = np.asarray(
        [frame.project(row[:2], row[3:7]).lateral_m for row in selected], dtype="<f8"
    )
    progress = source_progress - source_progress[0]
    if np.any(np.diff(progress) <= 0):
        raise gmt_io.GMTAdmissionError("selected donor progress is not strictly increasing")

    rotations = []
    for row in selected:
        roll, pitch, _ = reference_math.quaternion_to_euler_wxyz(row[3:7])
        cr, sr = math.cos(float(roll) / 2), math.sin(float(roll) / 2)
        cp, sp = math.cos(float(pitch) / 2), math.sin(float(pitch) / 2)
        rotations.append((sr * cp, cr * sp, -sr * sp, cr * cp))
    arrays = {
        "fps": np.asarray([FPS_HZ], dtype="<f8"),
        "root_pos": np.ascontiguousarray(
            np.column_stack((progress, np.zeros(FRAME_COUNT), selected[:, 2])), dtype="<f4"
        ),
        "root_rot": np.ascontiguousarray(rotations, dtype="<f4"),
        "dof_pos": np.ascontiguousarray(selected[:, 7:30], dtype="<f4"),
    }
    expected = {
        "fps": ((1,), "<f8"),
        "root_pos": ((FRAME_COUNT, 3), "<f4"),
        "root_rot": ((FRAME_COUNT, 4), "<f4"),
        "dof_pos": ((FRAME_COUNT, contracts.ACTION_DIM), "<f4"),
    }
    for name, (shape, dtype) in expected.items():
        array = arrays[name]
        if array.shape != shape or array.dtype.str != dtype or not np.isfinite(array).all():
            raise gmt_io.GMTAdmissionError(f"candidate array contract failed: {name}")
    if not np.allclose(np.linalg.norm(arrays["root_rot"], axis=1), 1.0, atol=1e-6, rtol=0):
        raise gmt_io.GMTAdmissionError("candidate float32 orientation is not unit length")
    measurements = {
        "source_qpos_half_open": list(SOURCE_QPOS_HALF_OPEN),
        "frame_count": FRAME_COUNT,
        "interval_count": INTERVAL_COUNT,
        "duration_seconds": INTERVAL_COUNT / FPS_HZ,
        "source_root_height_m_start_min_end": [
            float(selected[0, 2]),
            float(selected[:, 2].min()),
            float(selected[-1, 2]),
        ],
        "source_progress_delta_m": float(progress[-1]),
        "source_lateral_delta_m": float(source_lateral[-1] - source_lateral[0]),
        "output_progress_delta_m_float32": float(arrays["root_pos"][-1, 0]),
        "source_quaternion_norm_min_max": [float(norms.min()), float(norms.max())],
    }
    return arrays, measurements


def validate_pinned_measurements(measurements: dict[str, object]) -> None:
    expected = {
        "source_root_height_m_start_min_end": (
            0.7785659884189685,
            0.504942203119977,
            0.5954433356039784,
        ),
        "source_progress_delta_m": 1.3944665392223432,
        "source_lateral_delta_m": -0.19141908312751366,
    }
    for name, target in expected.items():
        observed = np.asarray(measurements[name], dtype="<f8")
        if not np.allclose(observed, np.asarray(target), atol=1e-12, rtol=0):
            raise gmt_io.GMTAdmissionError(f"pinned donor measurement differs: {name}")


def verify_model_assets(artifact_root: Path) -> tuple[Path, dict[str, object]]:
    xml_path = artifact_root / "upstream/assets/robots/g1/g1.xml"
    if gmt_io.sha256_file(xml_path, expected_size=MODEL_XML_SIZE) != MODEL_XML_SHA256:
        raise gmt_io.GMTAdmissionError("pinned G1 model XML digest differs")
    mesh_root = artifact_root / "upstream/assets/robots/g1/meshes"
    mesh_hashes: dict[str, str] = {}
    tree = hashlib.sha256()
    for name in contracts.GMT_G1_MESH_NAMES:
        digest = gmt_io.sha256_file(mesh_root / name)
        mesh_hashes[name] = digest
        tree.update(f"{name}\0{digest}\n".encode())
    if tree.hexdigest() != MESH_TREE_SHA256:
        raise gmt_io.GMTAdmissionError("pinned G1 model mesh-tree digest differs")
    return xml_path, {
        "xml": {"path": "upstream/assets/robots/g1/g1.xml", "sha256": MODEL_XML_SHA256},
        "mesh_tree_sha256": MESH_TREE_SHA256,
        "mesh_sha256": mesh_hashes,
    }


def validate_static_geometry(xml_path: Path, arrays: dict[str, np.ndarray]) -> dict[str, object]:
    """Run mj_forward only and fail on any non-foot ground contact."""

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    expected_joints = ("pelvis", *(f"{name}_joint" for name in contracts.JOINT_NAMES))
    joints = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    )
    actuators = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    )
    if (
        (model.nq, model.nv, model.nu) != (30, 29, contracts.ACTION_DIM)
        or joints != expected_joints
        or actuators != tuple(f"{name}_joint" for name in contracts.JOINT_NAMES)
        or tuple(int(value) for value in model.actuator_trnid[:, 0])
        != tuple(range(1, contracts.ACTION_DIM + 1))
    ):
        raise gmt_io.GMTAdmissionError("pinned G1 model joint/actuator ABI differs")
    planes = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).tolist()
    if planes != [0]:
        raise gmt_io.GMTAdmissionError(f"expected sole ground plane geom 0, observed {planes}")
    geom_body_names = [
        mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_index])
        )
        for geom_index in range(model.ngeom)
    ]
    data = mujoco.MjData(model)
    contact_pose_count = 0
    allowed_contact_pose_count = 0
    nonfoot_contact_pose_count = 0
    deepest_allowed: float | None = None
    per_pose: list[dict[str, object]] = []
    for index in range(FRAME_COUNT):
        xyzw = arrays["root_rot"][index].astype("<f8")
        data.qpos[:] = np.concatenate(
            (
                arrays["root_pos"][index].astype("<f8"),
                xyzw[[3, 0, 1, 2]],
                arrays["dof_pos"][index].astype("<f8"),
            )
        )
        data.qvel[:] = 0
        mujoco.mj_forward(model, data)
        bodies: set[str] = set()
        distances: list[float] = []
        for contact in data.contact[: int(data.ncon)]:
            pair = [int(contact.geom1), int(contact.geom2)]
            if 0 not in pair:
                continue
            other = pair[1] if pair[0] == 0 else pair[0]
            bodies.add(geom_body_names[other])
            distances.append(float(contact.dist))
        nonfoot = sorted(bodies.difference(ALLOWED_FEET))
        if nonfoot:
            nonfoot_contact_pose_count += 1
        if bodies:
            contact_pose_count += 1
        if bodies.intersection(ALLOWED_FEET):
            allowed_contact_pose_count += 1
            minimum = min(distances)
            deepest_allowed = minimum if deepest_allowed is None else min(deepest_allowed, minimum)
        per_pose.append(
            {
                "index": index,
                "ground_contact_bodies": sorted(bodies),
                "minimum_signed_contact_distance_m": min(distances) if distances else None,
            }
        )
    if nonfoot_contact_pose_count:
        raise gmt_io.GMTAdmissionError(
            f"candidate has non-foot static ground contact in {nonfoot_contact_pose_count} poses"
        )
    penetration = max(0.0, -(deepest_allowed or 0.0))
    if penetration > MAX_ALLOWED_FOOT_PENETRATION_M:
        raise gmt_io.GMTAdmissionError(
            f"candidate allowed-foot penetration {penetration} exceeds reviewed bound"
        )
    return {
        "method": "mujoco.mj_forward_only_with_qvel_zero_no_mj_step",
        "mujoco_version": mujoco.__version__,
        "evaluated_pose_indices_half_open": [0, FRAME_COUNT],
        "evaluated_pose_count": FRAME_COUNT,
        "ground_contact_pose_count": contact_pose_count,
        "allowed_foot_contact_pose_count": allowed_contact_pose_count,
        "nonfoot_ground_contact_pose_count": nonfoot_contact_pose_count,
        "allowed_ground_contact_bodies": sorted(ALLOWED_FEET),
        "deepest_allowed_foot_signed_contact_distance_m": deepest_allowed,
        "maximum_allowed_foot_penetration_m": penetration,
        "reviewed_maximum_allowed_foot_penetration_m": MAX_ALLOWED_FOOT_PENETRATION_M,
        "per_pose_sha256": _canonical_sha256(per_pose),
        "interpretation": "allowed-foot overlap is reported; candidate is not penetration-free",
    }


def convert(*, donor_root: Path, artifact_root: Path, output: Path) -> dict[str, object]:
    output = Path(output)
    if output.suffix != ".npz":
        raise gmt_io.GMTAdmissionError("candidate output must use the .npz suffix")
    receipt_path = output.with_suffix(f"{output.suffix}.manifest.json")
    if output.exists() or receipt_path.exists():
        raise gmt_io.GMTAdmissionError("refusing to overwrite candidate output or receipt")
    sources = source_identity()
    qpos, donor = validate_donor(Path(donor_root))
    arrays, measurements = build_candidate_arrays(qpos)
    validate_pinned_measurements(measurements)
    xml_path, model = verify_model_assets(Path(artifact_root))
    geometry = validate_static_geometry(xml_path, arrays)
    output_sha256 = gmt_io.write_deterministic_npz(output, arrays)
    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact": "gmt_g1_execution_derived_reference_candidate",
        "candidate_id": CANDIDATE_ID,
        "claim_ceiling": (
            "hash_pinned_kinematic_candidate_only_not_dynamics_certified_admitted_or_retracked"
        ),
        "donor": donor,
        "transform": {
            "cadence_hz": FPS_HZ,
            "selection": "qpos[92:198]; frame rows 92:197 are the 105 executed intervals",
            "root_x": "initial-episode-task-frame progress rebased to selected-state zero",
            "root_y": "zero",
            "root_z": "exact selected donor qpos cast float32",
            "root_rotation_xyzw": (
                "donor world quaternion roll/pitch reconstructed as unit quaternion at yaw zero"
            ),
            "dof_position": "exact selected donor qpos[7:30] cast float32 in GMT joint order",
            "velocity_semantics": (
                "not stored; ReferenceMotion rederives finite-difference velocities and applies "
                "the pinned 19-frame smoothing kernel when later loaded"
            ),
        },
        "measurements": measurements,
        "model": model,
        "static_geometry": geometry,
        "output": {
            "path": output.name,
            "format": "deterministic_numeric_only_npz",
            "sha256": output_sha256,
            "size": output.stat().st_size,
            "arrays": {
                name: {
                    "shape": list(value.shape),
                    "dtype": value.dtype.str,
                    "c_order_values_sha256": _array_sha256(value),
                }
                for name, value in arrays.items()
            },
        },
        "source": sources,
        "promotion": "none; separate reviewed admission and re-tracking evidence required",
    }
    receipt_sha256 = gmt_io.write_json_receipt(receipt_path, receipt)
    return {
        **receipt,
        "receipt": {"path": receipt_path.name, "sha256": receipt_sha256},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--donor-root", type=Path, default=DONOR_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = convert(
        donor_root=args.donor_root,
        artifact_root=args.artifact_root,
        output=args.output,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
