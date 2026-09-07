#!/usr/bin/env python3
"""Recompute the fixed, data-only GMT basic-walk window audit."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from oracle_composition.adapters.gmt.composition import (
    POSE_COLUMNS,
    POSE_SCALES,
    ReferenceSegment,
)
from oracle_composition.adapters.gmt.contracts import GMT_UPSTREAM_COMMIT, MOTION_SPECS
from oracle_composition.adapters.gmt.motions import load_converted_motion
from oracle_composition.adapters.gmt.reference_math import (
    inverse_rotate_xyzw,
    quaternion_to_euler_xyzw,
)
from oracle_composition.adapters.gmt.reference_runtime import ReferenceMotion

SCHEMA_ID = "gmt_basic_walk_native_window_audit/v1"
BASIC_WALK_NPZ_SHA256 = "b6ee3143e61b308daebb2a1f07d8b420459a2d19cbde76ecc8f4d42affcdb7b1"
WALK_STAND_NPZ_SHA256 = "908ba0e4f6acf1ecf829b0ddb73ed7e649ba6e7ca9fa1a96b43a95e0fdc2e3b0"
MIN_INTERVALS = 18
MAX_INTERVALS = 74
TARGET_FORWARD_SPEED_MPS = 0.7
CURRENT_ENTRY_END_SECONDS = 15.0
STANDING_ANCHOR_SECONDS = 1.0


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    role: str
    start_frame: int
    end_frame: int


# Selection is frozen. The script verifies these three candidates against the
# already declared search space; it does not discover, rank, or emit alternatives.
CANDIDATES = (
    Candidate("balanced_cycle", "balanced source-boundary seams", 1118, 1151),
    Candidate("minimum_net_yaw_cycle", "minimum net yaw among the retained set", 1119, 1148),
    Candidate("current_entry_control", "preserves the current source phase zero", 0, 18),
)


def _load_motion(path: Path, *, name: str, expected_sha256: str) -> ReferenceMotion:
    arrays = load_converted_motion(path, name=name, expected_sha256=expected_sha256)
    return ReferenceMotion(arrays)


def _native_frame_features(motion: ReferenceMotion) -> np.ndarray:
    euler = quaternion_to_euler_xyzw(motion.root_rotation_xyzw)
    local_velocity = inverse_rotate_xyzw(motion.root_rotation_xyzw, motion.root_velocity)
    local_angular_velocity = inverse_rotate_xyzw(
        motion.root_rotation_xyzw, motion.root_angular_velocity
    )
    return (
        torch.cat(
            (
                motion.root_position[:, 2:3],
                euler[:, :2],
                local_velocity,
                local_angular_velocity[:, 2:3],
                motion.dof_position,
            ),
            dim=-1,
        )
        .numpy()
        .astype(np.float64)
    )


def _seam_metrics(
    start: np.ndarray,
    end: np.ndarray,
    *,
    start_joint_velocity: np.ndarray,
    end_joint_velocity: np.ndarray,
) -> dict[str, float]:
    delta = end - start
    return {
        "root_height_abs_m": float(abs(delta[0])),
        "roll_pitch_rms_rad": float(np.sqrt(np.mean(delta[1:3] ** 2))),
        "joint_pose_rms_rad": float(np.sqrt(np.mean(delta[7:30] ** 2))),
        "root_local_velocity_l2_mps": float(np.linalg.norm(delta[3:6])),
        "yaw_rate_abs_radps": float(abs(delta[6])),
        "joint_velocity_rms_radps": float(
            np.sqrt(np.mean((end_joint_velocity - start_joint_velocity) ** 2))
        ),
    }


def _window_metrics(
    features: np.ndarray,
    joint_velocity: np.ndarray,
    candidate: Candidate,
    *,
    supplied_fps: float,
    standing_height_m: float,
) -> dict[str, object]:
    start, end = candidate.start_frame, candidate.end_frame
    window = features[start:end]
    duration = (end - start) / supplied_fps
    yaw_rate_mean = float(window[:, 6].mean())
    return {
        "candidate_id": candidate.candidate_id,
        "role": candidate.role,
        "source_frame_interval": {"start_inclusive": start, "end_exclusive": end},
        "source_time_seconds_supplied_double_fps": {
            "start": start / supplied_fps,
            "end_exclusive": end / supplied_fps,
            "duration": duration,
        },
        "in_window": {
            "yaw_rate_mean_radps": yaw_rate_mean,
            "yaw_rate_abs_mean_radps": float(np.abs(window[:, 6]).mean()),
            "integrated_yaw_proxy_rad": yaw_rate_mean * duration,
            "integrated_yaw_proxy_deg": math.degrees(yaw_rate_mean * duration),
            "root_local_forward_velocity_mean_mps": float(window[:, 3].mean()),
            "root_local_forward_velocity_mae_to_0p7_mps": float(
                np.abs(window[:, 3] - TARGET_FORWARD_SPEED_MPS).mean()
            ),
            "root_local_lateral_velocity_abs_mean_mps": float(np.abs(window[:, 4]).mean()),
            "root_height_mean_m": float(window[:, 0].mean()),
            "root_height_std_m": float(window[:, 0].std()),
            "root_height_mean_abs_error_to_standing_anchor_m": float(
                abs(window[:, 0].mean() - standing_height_m)
            ),
        },
        "native_frame_boundary_seam": _seam_metrics(
            features[start],
            features[end],
            start_joint_velocity=joint_velocity[start],
            end_joint_velocity=joint_velocity[end],
        ),
    }


def _pose_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.mean(
            ((left[np.asarray(POSE_COLUMNS)] - right[np.asarray(POSE_COLUMNS)]) / POSE_SCALES) ** 2
        )
    )


def _entry_metrics(
    features: np.ndarray,
    joint_velocity: np.ndarray,
    candidate: Candidate,
    *,
    standing_features: np.ndarray,
    standing_joint_velocity: np.ndarray,
    supplied_fps: float,
) -> dict[str, object]:
    start = candidate.start_frame
    return {
        "source_start_inside_current_full_clip_0_to_15s_entry_pool": bool(
            start / supplied_fps <= CURRENT_ENTRY_END_SECONDS
        ),
        "normalized_pose_distance_to_walk_stand_phase_zero": _pose_distance(
            features[start], standing_features[0]
        ),
        "root_local_velocity_jump_to_walk_stand_phase_zero_l2_mps": float(
            np.linalg.norm(features[start, 3:6] - standing_features[0, 3:6])
        ),
        "yaw_rate_jump_to_walk_stand_phase_zero_radps": float(
            abs(features[start, 6] - standing_features[0, 6])
        ),
        "joint_velocity_jump_to_walk_stand_phase_zero_rms_radps": float(
            np.sqrt(np.mean((joint_velocity[start] - standing_joint_velocity[0]) ** 2))
        ),
    }


def _frame_coordinate(motion: ReferenceMotion, source_time_seconds: float) -> dict[str, object]:
    time = torch.tensor([source_time_seconds], dtype=torch.float32)
    loop = torch.floor(time / motion.duration)
    wrapped = time - loop * motion.duration
    coordinate = torch.clamp(wrapped / motion.duration, 0.0, 1.0) * (motion.frame_count - 1)
    frame_zero = coordinate.long()
    return {
        "float32_source_time_seconds": float(time.item()),
        "frame_coordinate": float(coordinate.item()),
        "frame_zero": int(frame_zero.item()),
        "blend": float((coordinate - frame_zero.float()).item()),
    }


def _runtime_endpoint_verification(
    motion: ReferenceMotion,
    native_features: np.ndarray,
    joint_velocity: np.ndarray,
    candidate: Candidate,
) -> dict[str, object]:
    runtime_fps = float(motion.fps)
    start_seconds = candidate.start_frame / runtime_fps
    end_seconds = candidate.end_frame / runtime_fps
    times = torch.tensor([start_seconds, end_seconds], dtype=torch.float32)
    runtime_features = motion.features(times).numpy().astype(np.float64)
    native_endpoints = native_features[[candidate.start_frame, candidate.end_frame]]
    native_seam = _seam_metrics(
        native_endpoints[0],
        native_endpoints[1],
        start_joint_velocity=joint_velocity[candidate.start_frame],
        end_joint_velocity=joint_velocity[candidate.end_frame],
    )
    runtime_seam = _seam_metrics(
        runtime_features[0],
        runtime_features[1],
        start_joint_velocity=joint_velocity[candidate.start_frame],
        end_joint_velocity=joint_velocity[candidate.end_frame],
    )
    seam_deltas = {name: abs(runtime_seam[name] - native_seam[name]) for name in native_seam}
    return {
        "proposed_config_bounds_seconds_runtime_float32_fps": {
            "start": start_seconds,
            "end_exclusive": end_seconds,
        },
        "runtime_start_mapping": _frame_coordinate(motion, start_seconds),
        "runtime_end_mapping": _frame_coordinate(motion, end_seconds),
        "runtime_feature_max_abs_difference_from_native_frame": {
            "start": float(np.max(np.abs(runtime_features[0] - native_endpoints[0]))),
            "end": float(np.max(np.abs(runtime_features[1] - native_endpoints[1]))),
        },
        "runtime_config_boundary_seam": runtime_seam,
        "runtime_vs_native_seam_abs_difference": seam_deltas,
        "runtime_vs_native_seam_max_abs_difference": max(seam_deltas.values()),
    }


def _objective_vector(metrics: dict[str, object]) -> np.ndarray:
    in_window = metrics["in_window"]
    seam = metrics["native_frame_boundary_seam"]
    assert isinstance(in_window, dict)
    assert isinstance(seam, dict)
    return np.asarray(
        [
            abs(in_window["yaw_rate_mean_radps"]),
            in_window["yaw_rate_abs_mean_radps"],
            in_window["root_local_forward_velocity_mae_to_0p7_mps"],
            in_window["root_local_lateral_velocity_abs_mean_mps"],
            in_window["root_height_mean_abs_error_to_standing_anchor_m"],
            in_window["root_height_std_m"],
            seam["root_height_abs_m"],
            seam["roll_pitch_rms_rad"],
            seam["joint_pose_rms_rad"],
            seam["root_local_velocity_l2_mps"],
            seam["yaw_rate_abs_radps"],
            seam["joint_velocity_rms_radps"],
        ],
        dtype=np.float64,
    )


def _verify_selected_are_pareto(
    features: np.ndarray,
    joint_velocity: np.ndarray,
    selected_metrics: list[dict[str, object]],
    *,
    supplied_fps: float,
    standing_height_m: float,
) -> dict[str, object]:
    selected_vectors = {
        str(metrics["candidate_id"]): _objective_vector(metrics) for metrics in selected_metrics
    }
    dominator_counts = {candidate_id: 0 for candidate_id in selected_vectors}
    windows_checked = 0
    for length in range(MIN_INTERVALS, MAX_INTERVALS + 1):
        for start in range(0, features.shape[0] - length):
            comparison = Candidate("comparison", "not emitted", start, start + length)
            vector = _objective_vector(
                _window_metrics(
                    features,
                    joint_velocity,
                    comparison,
                    supplied_fps=supplied_fps,
                    standing_height_m=standing_height_m,
                )
            )
            windows_checked += 1
            for candidate_id, selected in selected_vectors.items():
                if np.all(vector <= selected) and np.any(vector < selected):
                    dominator_counts[candidate_id] += 1
    if any(dominator_counts.values()):
        raise RuntimeError(f"fixed candidate lost Pareto status: {dominator_counts}")
    return {
        "frame_interval_lengths_inclusive": [MIN_INTERVALS, MAX_INTERVALS],
        "duration_range_seconds_supplied_double_fps": [
            MIN_INTERVALS / supplied_fps,
            MAX_INTERVALS / supplied_fps,
        ],
        "windows_checked": windows_checked,
        "objectives_minimized": [
            "abs_yaw_rate_mean_radps",
            "yaw_rate_abs_mean_radps",
            "root_local_forward_velocity_mae_to_0p7_mps",
            "root_local_lateral_velocity_abs_mean_mps",
            "root_height_mean_abs_error_to_standing_anchor_m",
            "root_height_std_m",
            "seam_root_height_abs_m",
            "seam_roll_pitch_rms_rad",
            "seam_joint_pose_rms_rad",
            "seam_root_local_velocity_l2_mps",
            "seam_yaw_rate_abs_radps",
            "seam_joint_velocity_rms_radps",
        ],
        "selected_candidate_dominator_counts": dominator_counts,
    }


def analyze(numeric_root: Path) -> dict[str, object]:
    basic_path = numeric_root / "basic_walk.npz"
    stand_path = numeric_root / "walk_stand.npz"
    basic = _load_motion(
        basic_path,
        name="basic_walk",
        expected_sha256=BASIC_WALK_NPZ_SHA256,
    )
    stand = _load_motion(
        stand_path,
        name="walk_stand",
        expected_sha256=WALK_STAND_NPZ_SHA256,
    )
    basic_features = _native_frame_features(basic)
    stand_features = _native_frame_features(stand)
    basic_joint_velocity = basic.dof_velocity.numpy().astype(np.float64)
    stand_joint_velocity = stand.dof_velocity.numpy().astype(np.float64)
    supplied_fps = MOTION_SPECS["basic_walk"].fps
    standing_anchor_frames = round(MOTION_SPECS["walk_stand"].fps * STANDING_ANCHOR_SECONDS)
    standing_height_m = float(stand_features[-standing_anchor_frames:, 0].mean())

    candidates = []
    for candidate in CANDIDATES:
        metrics = _window_metrics(
            basic_features,
            basic_joint_velocity,
            candidate,
            supplied_fps=supplied_fps,
            standing_height_m=standing_height_m,
        )
        metrics["entry_comparison"] = _entry_metrics(
            basic_features,
            basic_joint_velocity,
            candidate,
            standing_features=stand_features,
            standing_joint_velocity=stand_joint_velocity,
            supplied_fps=supplied_fps,
        )
        metrics["runtime_endpoint_verification"] = _runtime_endpoint_verification(
            basic,
            basic_features,
            basic_joint_velocity,
            candidate,
        )
        candidates.append(metrics)

    full_window = basic_features
    current_full_segment = ReferenceSegment(
        basic,
        BASIC_WALK_NPZ_SHA256,
        0.0,
        float(basic.duration),
        entry_phase_end_seconds=CURRENT_ENTRY_END_SECONDS,
    )
    walk_stand_phase_zero = stand.features(torch.tensor([0.0], dtype=torch.float32))[0].numpy()
    nearest_entry_phase, nearest_entry_score = current_full_segment.nearest_phase(
        walk_stand_phase_zero[np.asarray(POSE_COLUMNS)]
    )
    nearest_entry_frame = round(nearest_entry_phase * float(basic.fps))
    current_phase_zero_score = _pose_distance(
        basic.features(torch.tensor([0.0], dtype=torch.float32))[0].numpy(),
        walk_stand_phase_zero,
    )

    result = {
        "schema_id": SCHEMA_ID,
        "source": {
            "gmt_upstream_commit": GMT_UPSTREAM_COMMIT,
            "basic_walk": {
                "converted_npz_sha256": BASIC_WALK_NPZ_SHA256,
                "original_pinned_motion_sha256": MOTION_SPECS["basic_walk"].sha256,
                "frames": basic.frame_count,
                "supplied_double_fps": supplied_fps,
                "runtime_float32_fps": float(basic.fps),
                "duration_float32_seconds": float(basic.duration),
            },
            "walk_stand": {
                "converted_npz_sha256": WALK_STAND_NPZ_SHA256,
                "original_pinned_motion_sha256": MOTION_SPECS["walk_stand"].sha256,
                "frames": stand.frame_count,
                "supplied_double_fps": MOTION_SPECS["walk_stand"].fps,
                "runtime_float32_fps": float(stand.fps),
            },
        },
        "method": {
            "classification": "exploratory_reference_numeric_fitness_only",
            "selection": "three_fixed_candidates_no_policy_evaluation_ranking",
            "window_endpoint_convention": (
                "frames [start,end) contribute in-window metrics; the source-boundary seam "
                "compares frame end with frame start. wrap_within_segment treats end as "
                "exclusive, so exact phase==duration returns start while phase just below "
                "duration approaches end. A 50 Hz control sequence need not hit that endpoint."
            ),
            "derived_velocity": (
                "float32 first differences times runtime float32 FPS; last raw difference "
                "copied from the prior frame; 19-frame zero-padded box smoothing. Angular "
                "velocity uses xyzw quaternion delta/exponential-map before the same smoothing."
            ),
            "feature_scope": (
                "root height m, roll/pitch rad, quaternion-inverse-rotated root velocity m/s, "
                "local yaw rate rad/s, and 23 joint positions rad; absolute yaw and root x/y "
                "are absent"
            ),
            "standing_height_anchor": {
                "motion": "walk_stand",
                "terminal_frames": standing_anchor_frames,
                "height_mean_m": standing_height_m,
            },
        },
        "current_full_basic_walk": {
            "root_local_forward_velocity_mean_mps": float(full_window[:, 3].mean()),
            "root_local_forward_velocity_mae_to_0p7_mps": float(
                np.abs(full_window[:, 3] - TARGET_FORWARD_SPEED_MPS).mean()
            ),
            "root_local_lateral_velocity_abs_mean_mps": float(np.abs(full_window[:, 4]).mean()),
            "yaw_rate_mean_radps": float(full_window[:, 6].mean()),
            "yaw_rate_abs_mean_radps": float(np.abs(full_window[:, 6]).mean()),
            "root_height_mean_m": float(full_window[:, 0].mean()),
            "root_height_std_m": float(full_window[:, 0].std()),
        },
        "current_0_to_15s_entry_pose_comparison": {
            "rule_scope": "height_roll_pitch_and_23_joint_pose_only_not_velocity",
            "nearest_to_walk_stand_phase_zero": {
                "frame": nearest_entry_frame,
                "runtime_phase_seconds": nearest_entry_phase,
                "source_time_seconds_supplied_double_fps": nearest_entry_frame / supplied_fps,
                "normalized_pose_distance": nearest_entry_score,
            },
            "current_phase_zero_normalized_pose_distance": current_phase_zero_score,
        },
        "pareto_verification": _verify_selected_are_pareto(
            basic_features,
            basic_joint_velocity,
            candidates,
            supplied_fps=supplied_fps,
            standing_height_m=standing_height_m,
        ),
        "candidates": candidates,
        "claim_limits": [
            "No policy evaluation, simulator, contact, stability, or tracker behavior is used.",
            "Numeric fitness does not establish dynamics feasibility or trackability.",
            "Integrated yaw rate is a local-reference proxy, not an absolute world-path result.",
            "Candidate selection is exploratory and must not alter a locked study implicitly.",
        ],
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--numeric-root",
        type=Path,
        required=True,
        help="directory containing the exact admitted basic_walk.npz and walk_stand.npz",
    )
    args = parser.parse_args()
    print(json.dumps(analyze(args.numeric_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
