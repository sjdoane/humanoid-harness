from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import threading
from collections.abc import Callable
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from oracle_composition.research import build_index
from oracle_composition.ui import create_server
from oracle_composition.ui import local_evidence as local_evidence_module
from oracle_composition.ui import local_reference_probe as reference_probe_module


def _write_record(directory: Path, paper_id: str, title: str) -> None:
    record = {
        "paper": {
            "id": paper_id,
            "title": title,
            "year": 2026,
            "primary_url": f"https://example.test/{paper_id}",
            "source_version": "test-v1",
        },
        "decision_relevance": {
            "tier": "core",
            "knobs": ["reference_composition"],
            "why_admitted": "Tests state-aware recovery decisions.",
        },
        "mechanisms": [],
        "parameters": [],
        "failure_modes": [],
        "evaluation": [],
        "relations": [],
        "evidence": [],
    }
    (directory / f"{paper_id}.json").write_text(json.dumps(record), encoding="utf-8")


def _get_json(url: str) -> dict[str, object]:
    with urlopen(url, timeout=2) as response:
        assert response.status == 200
        return json.loads(response.read())


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_local_exploration(
    root: Path,
    monkeypatch: pytest.MonkeyPatch | None = None,
) -> Path:
    directory = root / "artifacts/exploration/phase_oracle_holdout_v0"
    directory.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "status": "local_exploration_only",
        "locked_before_holdout": True,
        "redistributable": False,
        "claim_ceiling": "tier_k_local_oracle_diagnostic_not_formal_experiment_evidence",
        "question": "Does bounded phase correction improve recovery?",
        "reason_not_admitted": ["source license absent", "no Tier-D certificate"],
        "frozen_runtime": {"max_actions": 1000},
        "reference": {"admission": "Tier-K_not_admitted"},
        "bounded_phase_rule": {},
        "recovery_fallback_rule": {},
        "arms": {
            "T0_G0": "elapsed-time phase",
            "T0_G1": "elapsed-time phase plus fallback",
            "T1_G0": "bounded phase",
            "T1_G1": "bounded phase plus fallback",
            "base_context": "controller without the residual tracker",
        },
        "paired_evaluation": {
            "seeds": list(range(3000, 3020)),
            "conditions": {
                "nominal": "none",
                "lateral_velocity": "lateral push",
                "pitch_velocity_falsifier": "pitch push",
            },
        },
        "visual_evidence": {
            "seed": 3000,
            "condition": "lateral_velocity",
            "arms": ["T0_G0", "T0_G1", "T1_G0", "T1_G1"],
            "selection_rule": "first predetermined holdout seed; never selected from outcome",
        },
        "locked_expectations": {
            "nominal": "T1_G1 must retain 20/20 no-collapse outcomes",
            "lateral_velocity": ("T1_G1 should exceed T0_G0 by at least two no-collapse outcomes"),
            "pitch_velocity_falsifier": (
                "no improvement is assumed; failure identifies missing controller recovery capability"
            ),
            "attribution": (
                "report all four T x G arms; do not attribute a joint-arm gain to phase alone"
            ),
        },
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    (directory / "manifest.json").write_bytes(manifest_bytes)
    manifest_sha = _sha256(manifest_bytes)
    arms = tuple(manifest["arms"])
    conditions = tuple(manifest["paired_evaluation"]["conditions"])
    seeds = tuple(manifest["paired_evaluation"]["seeds"])
    rows = []
    combinations = (
        (arm, condition, seed) for arm in arms for condition in conditions for seed in seeds
    )
    for run_order, (arm, condition, seed) in enumerate(combinations, start=1):
        no_collapse = not (
            (arm == "T0_G0" and condition == "lateral_velocity" and seed in {3000, 3001})
            or (arm == "T1_G1" and condition == "lateral_velocity" and seed == 3000)
            or condition == "pitch_velocity_falsifier"
        )
        rows.append(
            {
                "action_saturation_count": 0,
                "actions_executed": 1000,
                "arm": arm,
                "collapsed_action_count": 0 if no_collapse else 100,
                "condition": condition,
                "environment_return": float(10_000 + run_order),
                "final_recovered": no_collapse,
                "final_root_height_m": 1.3 if no_collapse else 0.8,
                "final_torso_up_z": 0.9 if no_collapse else 0.4,
                "first_collapse_action": None if no_collapse else 901,
                "manifest_sha256": manifest_sha,
                "no_collapse": no_collapse,
                "phase_correction_count": 700 if arm == "T1_G1" else 0,
                "recovery_entry_count": 1 if arm == "T1_G1" else 0,
                "recovery_exit_count": 1 if arm == "T1_G1" else 0,
                "recovery_gate_off_count": 300 if arm == "T1_G1" else 0,
                "root_x_displacement_m": float(70 + run_order),
                "run_order": run_order,
                "seed": seed,
            }
        )
    runs = b"".join((json.dumps(row, sort_keys=True) + "\n").encode("utf-8") for row in rows)
    (directory / "runs.jsonl").write_bytes(runs)
    runs_sha = _sha256(runs)

    groups: dict[str, object] = {}
    for arm in arms:
        for condition in conditions:
            selected = [row for row in rows if row["arm"] == arm and row["condition"] == condition]
            count = len(selected)
            groups[f"{arm}/{condition}"] = {
                "final_recovered_count": sum(bool(row["final_recovered"]) for row in selected),
                "mean_collapsed_action_count": sum(
                    int(row["collapsed_action_count"]) for row in selected
                )
                / count,
                "mean_environment_return": sum(float(row["environment_return"]) for row in selected)
                / count,
                "mean_first_collapse_or_1001": sum(
                    int(row["first_collapse_action"])
                    if row["first_collapse_action"] is not None
                    else 1001
                    for row in selected
                )
                / count,
                "mean_phase_correction_count": sum(
                    int(row["phase_correction_count"]) for row in selected
                )
                / count,
                "mean_recovery_gate_off_count": sum(
                    int(row["recovery_gate_off_count"]) for row in selected
                )
                / count,
                "mean_root_x_displacement_m": sum(
                    float(row["root_x_displacement_m"]) for row in selected
                )
                / count,
                "n": count,
                "no_collapse_count": sum(bool(row["no_collapse"]) for row in selected),
            }
    summary = {
        "status": "local_exploration_only",
        "completed_runs": len(rows),
        "scheduled_runs": len(rows),
        "manifest_sha256": manifest_sha,
        "runs_sha256": runs_sha,
        "groups": groups,
    }
    (directory / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    figure_rows = [
        "arm,condition,n,no_collapse_count,mean_collapsed_action_count,"
        "mean_environment_return,mean_recovery_gate_off_fraction"
    ]
    for name, group in groups.items():
        arm, condition = name.split("/", maxsplit=1)
        figure_rows.append(
            f"{arm},{condition},{group['n']},{group['no_collapse_count']},"
            f"{group['mean_collapsed_action_count']},{group['mean_environment_return']},"
            f"{float(group['mean_recovery_gate_off_count']) / 1000}"
        )
    figure_data = ("\n".join(figure_rows) + "\n").encode()
    alt_text = b"Synthetic held-out result figure."
    figure = b"\x89PNG\r\n\x1a\nsynthetic figure"
    figure_svg = b"<svg xmlns='http://www.w3.org/2000/svg'></svg>"
    poster = b"\x89PNG\r\n\x1a\nsynthetic poster"
    video = b"\x00\x00\x00\x18ftypisomsynthetic-video"
    (directory / "figure_data.csv").write_bytes(figure_data)
    (directory / "holdout_summary.png").write_bytes(figure)
    (directory / "holdout_summary.svg").write_bytes(figure_svg)
    (directory / "lateral_seed3000_poster.png").write_bytes(poster)
    (directory / "lateral_seed3000_four_arm.mp4").write_bytes(video)
    (directory / "holdout_summary.alt.txt").write_bytes(alt_text)
    figure_receipt = {
        "status": "local_exploration_only",
        "audience": "test",
        "publisher_requirements": "not applicable",
        "source_manifest": "manifest.json",
        "source_manifest_sha256": manifest_sha,
        "source_runs": "runs.jsonl",
        "source_runs_sha256": runs_sha,
        "transformations": ["none"],
        "uncertainty": "not claimed",
        "outputs": {
            "figure_data.csv": _sha256(figure_data),
            "holdout_summary.alt.txt": _sha256(alt_text),
            "holdout_summary.png": _sha256(figure),
            "holdout_summary.svg": _sha256(figure_svg),
        },
    }
    (directory / "holdout_summary.figure_manifest.json").write_text(
        json.dumps(figure_receipt), encoding="utf-8"
    )
    video_receipt = {
        "status": "local_exploration_only",
        "manifest_sha256": manifest_sha,
        "runs_sha256": runs_sha,
        "seed": 3000,
        "condition": "lateral_velocity",
        "selection": "first predetermined holdout seed",
        "arms": ["T0_G0", "T0_G1", "T1_G0", "T1_G1"],
        "claim_ceiling": "visible_closed_loop_execution_not_formal_oracle_evidence",
        "perturb_action": 300,
        "render_stride_actions": 2,
        "video_dimensions_px": [960, 1008],
        "video_duration_seconds": 15.03,
        "video_fps": 100 / 3,
        "video_frames": 501,
        "metrics": {
            row["arm"]: {
                "collapsed": row["collapsed_action_count"],
                "first_collapse": row["first_collapse_action"],
                "return": row["environment_return"],
            }
            for row in rows
            if row["condition"] == "lateral_velocity"
            and row["seed"] == 3000
            and row["arm"] != "base_context"
        },
        "poster_sha256": _sha256(poster),
        "video_sha256": _sha256(video),
    }
    (directory / "lateral_seed3000_four_arm.video_receipt.json").write_text(
        json.dumps(video_receipt), encoding="utf-8"
    )
    if monkeypatch is not None:
        monkeypatch.setattr(local_evidence_module, "_V0_MANIFEST_SHA256", manifest_sha)
        monkeypatch.setattr(local_evidence_module, "_V0_RUNS_SHA256", runs_sha)
        pinned_files = (
            "figure_data.csv",
            "holdout_summary.alt.txt",
            "holdout_summary.figure_manifest.json",
            "holdout_summary.png",
            "holdout_summary.svg",
            "lateral_seed3000_four_arm.video_receipt.json",
            "lateral_seed3000_poster.png",
            "lateral_seed3000_four_arm.mp4",
        )
        monkeypatch.setattr(
            local_evidence_module,
            "_V0_FILE_SHA256",
            {name: _sha256((directory / name).read_bytes()) for name in pinned_files},
        )
    return directory


def _rewrite_json(path: Path, mutate: Callable[[dict[str, object]], None]) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def _rebind_manifest(directory: Path, mutate: Callable[[dict[str, object]], None]) -> None:
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha = _sha256(manifest_bytes)

    rows = [json.loads(line) for line in (directory / "runs.jsonl").read_bytes().splitlines()]
    for row in rows:
        row["manifest_sha256"] = manifest_sha
    runs = b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)
    (directory / "runs.jsonl").write_bytes(runs)
    runs_sha = _sha256(runs)

    _rewrite_json(
        directory / "summary.json",
        lambda value: value.update(manifest_sha256=manifest_sha, runs_sha256=runs_sha),
    )
    _rewrite_json(
        directory / "holdout_summary.figure_manifest.json",
        lambda value: value.update(
            source_manifest_sha256=manifest_sha,
            source_runs_sha256=runs_sha,
        ),
    )
    _rewrite_json(
        directory / "lateral_seed3000_four_arm.video_receipt.json",
        lambda value: value.update(manifest_sha256=manifest_sha, runs_sha256=runs_sha),
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _float32_sha(values: list[float]) -> str:
    return _sha256(struct.pack(f"<{len(values)}f", *values))


def _probe_index_map(condition_id: str) -> list[int | None]:
    if condition_id == "C_exact":
        return list(range(1001))
    if condition_id == "C_zero_input":
        return [None] * 1001
    if condition_id == "C_shift_input":
        return [(index + 250) % 1001 for index in range(1001)]

    def rank(frame_index: int) -> tuple[bytes, int]:
        value = {
            "algorithm_id": "sha256_ranked_full_frame_indices/v1",
            "frame_index": frame_index,
            "n_frames": 1001,
            "seed": 260_907,
        }
        return hashlib.sha256(_canonical_json(value)).digest(), frame_index

    return sorted(range(1001), key=rank)


def _probe_transform_parameters(condition_id: str) -> dict[str, object]:
    common: dict[str, object] = {"horizon_steps": 8, "reference_width": 45}
    if condition_id == "C_exact":
        return {**common, "operation": "identity"}
    if condition_id == "C_zero_input":
        return {
            **common,
            "fill_float64_bits_hex": "0000000000000000",
            "operation": "positive_zero_fill",
        }
    if condition_id == "C_shuffle_input":
        return {
            **common,
            "operation": "output_frame_i_reads_source_frame_at_rank_i",
            "ranking_algorithm_id": "sha256_ranked_full_frame_indices/v1",
            "seed": 260_907,
        }
    return {
        **common,
        "operation": "output_frame_i_reads_source_frame_i_plus_shift_mod_T",
        "shift_algorithm_id": "cyclic_source_index_plus_250/v1",
        "shift_frames": 250,
    }


def _write_reference_probe(root: Path) -> Path:
    frames = (0, 125, 250, 375, 500, 625, 750, 875)
    conditions = ("C_exact", "C_zero_input", "C_shuffle_input", "C_shift_input")
    source_files = {
        "minari_importer_source_sha256": "src/oracle_composition/sources/minari_humanoid.py",
        "local_controller_loader_source_sha256": (
            "src/oracle_composition/experiments/local_controllers.py"
        ),
        "reference_transform_source_sha256": (
            "src/oracle_composition/experiments/reference_input_transforms.py"
        ),
        "probe_runner_source_sha256": (
            "src/oracle_composition/experiments/reference_causal_probe.py"
        ),
    }
    bindings: dict[str, object] = {
        key: _sha256(("fixture:" + key).encode()) for key in source_files
    }
    for key, relative in source_files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        source = ("fixture:" + key).encode()
        path.write_bytes(source)
        bindings[key] = _sha256(source)

    reference_schema_sha256 = "35d5e7cc86c0054d40362a40ff552ddf9252c3067d02a943cbc2f294c24b9771"
    projection_indices = [
        0,
        1,
        2,
        3,
        4,
        *range(22, 28),
        6,
        5,
        *range(7, 22),
        29,
        28,
        *range(30, 45),
    ]
    projection_id = "gymnasium/Humanoid-v5/observation-348-to-reference-45/v1"
    projection_mapping_sha256 = _sha256(
        _canonical_json(
            {
                "projection_id": projection_id,
                "indices": projection_indices,
                "source_width": 348,
                "target_schema_sha256": reference_schema_sha256,
            }
        )
    )
    projection_receipt = {
        "schema_version": 2,
        "evidence_tier": "Tier-K",
        "admission_status": "not_admitted",
        "claim_ceiling": "source_projection_and_numerical_candidate_only",
        "dynamics_feasibility_established": False,
        "source_provenance_class": "registered_official_repository_revision",
        "source_origin_verified": True,
        "asserted_source_repository": "https://huggingface.co/datasets/farama-minari/mujoco",
        "asserted_source_commit": "8e62dc7f7fcb4a19f8f869c65402d4bb60049117",
        "source_record_sha256": (
            "974700591304a4d3be576d55bd294558ae0cd10cbfce5bef7bd73ca21632f899"
        ),
        "source_record_bytes": 1_458,
        "dataset_id": "mujoco/humanoid/expert-v0",
        "metadata_declared_generation_code_url": (
            "https://github.com/Farama-Foundation/minari-dataset-generation-scripts"
        ),
        "hdf5_sha256": "8253be693f06aeeac3cb62eeb349ad02d4ca0bcad02685b4b8ed390798d9aa1e",
        "hdf5_bytes": 2_946_805_796,
        "metadata_sha256": ("2eb6e0ba388ceabef5eec1dea7e401e62391d856cf42b394c262db7c21366024"),
        "metadata_bytes": 9_305,
        "environment_id": "Humanoid-v5",
        "environment_entry_point": "gymnasium.envs.mujoco.humanoid_v5:HumanoidEnv",
        "minari_version": "test",
        "declared_requirements": ["gymnasium==1.3.0"],
        "episode_id": 0,
        "episode_seed": 123,
        "episode_total_steps": 1000,
        "episode_observation_shape": [1001, 348],
        "episode_action_shape": [1000, 17],
        "observations_sha256": ("bd7ce25125e322c5be1d6ca2de23eaef35317ebb6b24b023ceafdf63eb8d2387"),
        "actions_sha256": ("4e7d9cef2a80eb2d05f065f5de4ccb4d04222750da8d1a4d0d9c61ad3fa8c30e"),
        "rewards_sha256": ("785f49e8552f9038e8507d53d0cc6b84c33e5b7ffe12c5c77e7b606c79a6392b"),
        "terminations_sha256": ("b165fd211be467fe0761fd70eacdd28c4419fcfebe24f747afda4236c023e44d"),
        "truncations_sha256": ("1bf9590007e7df7e8f0552ba20389393a0e915cf2feb1fd79d023724e41d21b6"),
        "ignored_observation_fields_sha256": (
            "e015df720194e3b35e55f4750802aa875c8dccfa7b3a08d9958f1bdb063bd333"
        ),
        "root_position_xy_evidence": "absent_from_observation_and_empty_infos",
        "root_position_xy_reconstructed": False,
        "projection_id": projection_id,
        "projection_indices": projection_indices,
        "projection_mapping_sha256": projection_mapping_sha256,
        "reference_artifact_id": ("minari/mujoco/humanoid/expert-v0/episode-0/projected-45d/v1"),
        "reference_content_sha256": (
            "ef7557643ec87a31e4feb98c30faad98a6139dd646babf2d9e3be57b18f6936c"
        ),
        "reference_schema_sha256": reference_schema_sha256,
        "reference_shape": [1001, 45],
        "reference_cadence_hz": 200 / 3,
        "tier_d_certificate_sha256": None,
    }
    bindings.update(
        {
            "minari_projection_receipt_sha256": _sha256(_canonical_json(projection_receipt)),
            "source_observations_sha256": projection_receipt["observations_sha256"],
            "reference_content_sha256": projection_receipt["reference_content_sha256"],
            "reference_schema_sha256": reference_schema_sha256,
            "base_controller_content_sha256": (
                "ce2aa3a1358609f09509d7f352475a7b517c6d11858ff76419b18a187cb3adf3"
            ),
            "residual_controller_content_sha256": (
                "6916bf6778dd3044bca5feae22897b7d582e7389871a728549f791112d90fc22"
            ),
            "residual_controller_state_sha256": "e" * 64,
        }
    )

    def controller_receipt(role: str) -> dict[str, object]:
        is_base = role == "base"
        return {
            "schema_version": 1,
            "artifact_kind": (
                "minari_humanoid_behavior_cloning_controller"
                if is_base
                else "minari_reference_residual_actor"
            ),
            "format_id": "strict_npz_npy1_c_order_no_pickle/v1",
            "architecture_id": (
                "torch_mlp_348_512_512_512_17_relu_action_0.4tanh/v1"
                if is_base
                else "sb3_actor_critic_pi256x256_vf256x256_relu_deterministic/v1"
            ),
            "claim_status": "local_exploration_only_not_admitted",
            "filename": "minari_bc_v0.npz" if is_base else "reference_residual_ppo_v0.npz",
            "file_size_bytes": 100,
            "expected_content_sha256": bindings[
                "base_controller_content_sha256"
                if is_base
                else "residual_controller_content_sha256"
            ],
            "content_sha256": bindings[
                "base_controller_content_sha256"
                if is_base
                else "residual_controller_content_sha256"
            ],
            "registered_content_match": True,
            "loader_source_sha256": bindings["local_controller_loader_source_sha256"],
            "numpy_version": "test",
            "torch_version": "test",
            "gymnasium_version": "test",
            "stable_baselines3_version": "test",
            "observation_width": 348,
            "action_width": 17,
            "reference_width": None if is_base else 45,
            "reference_horizon_steps": None if is_base else 8,
            "residual_scale": None if is_base else 0.08,
            "inference_output_contract": (
                "float32_raw_Humanoid-v5_control_with_action_space_endpoint_bytes"
                if is_base
                else (
                    "unscaled_dimensionless_action_in_closed_interval_-1_1;"
                    "caller_multiplies_by_residual_scale_before_composition"
                )
            ),
        }

    base_receipt = controller_receipt("base")
    residual_receipt = controller_receipt("residual")
    bindings["base_controller_load_receipt_sha256"] = _sha256(_canonical_json(base_receipt))
    bindings["residual_controller_load_receipt_sha256"] = _sha256(_canonical_json(residual_receipt))

    base_action = [_float32(0.01 * (index - 8)) for index in range(17)]
    base_action_sha256 = _float32_sha(base_action)
    snapshots = [
        {
            "snapshot_frame": frame,
            "observation_sha256": hashlib.sha256(f"observation:{frame}".encode()).hexdigest(),
            "ground_truth_window_sha256": hashlib.sha256(f"target:{frame}".encode()).hexdigest(),
            "base_action_sha256": base_action_sha256,
        }
        for frame in frames
    ]
    bindings["matched_observation_set_sha256"] = _sha256(_canonical_json(snapshots))

    source_reference_sha256 = "f" * 64
    transformed_sequence_sha256 = {
        condition: source_reference_sha256
        if condition == "C_exact"
        else _sha256(condition.encode())
        for condition in conditions
    }
    records = []
    condition_values = {
        "C_exact": 0.0,
        "C_zero_input": 0.1,
        "C_shuffle_input": 0.2,
        "C_shift_input": 0.3,
    }
    for snapshot in snapshots:
        frame = int(snapshot["snapshot_frame"])
        for condition in conditions:
            index_map = _probe_index_map(condition)
            timeline = [min(frame + offset, 1000) for offset in range(8)]
            transform = {
                "schema_version": 1,
                "algorithm_id": "experiment_002a_numeric_reference_input/v1",
                "array_hash_algorithm_id": "framed_c_order_little_endian_float64_array/v1",
                "index_map_hash_algorithm_id": "canonical_json_index_map/v1",
                "condition_id": condition,
                "dtype": "<f8",
                "order": "C",
                "source_shape": [1001, 45],
                "source_sha256": source_reference_sha256,
                "transformed_full_sequence_sha256": transformed_sequence_sha256[condition],
                "parameters": _probe_transform_parameters(condition),
                "index_map": index_map,
                "index_map_sha256": _sha256(
                    _canonical_json(
                        {
                            "algorithm_id": "canonical_json_index_map/v1",
                            "index_map": index_map,
                        }
                    )
                ),
                "current_frame": frame,
                "window_timeline_indices": timeline,
                "window_source_indices": [index_map[index] for index in timeline],
                "output_window_shape": [8, 45],
                "output_window_sha256": (
                    snapshot["ground_truth_window_sha256"]
                    if condition == "C_exact"
                    else _sha256(f"window:{frame}:{condition}".encode())
                ),
            }
            residual_value = _float32(condition_values[condition])
            actor_output = [residual_value] * 17
            composed_action = [
                max(
                    -0.4,
                    min(
                        0.4,
                        _float32(
                            _float32(base) + _float32(_float32(0.08) * _float32(residual_value))
                        ),
                    ),
                )
                for base in base_action
            ]
            critic_value = _float32(condition_values[condition])
            policy_sha = _sha256(f"input:{frame}:{condition}".encode())
            records.append(
                {
                    "snapshot_frame": frame,
                    "condition_id": condition,
                    "observation_sha256": snapshot["observation_sha256"],
                    "ground_truth_window_sha256": snapshot["ground_truth_window_sha256"],
                    "transformed_window_sha256": transform["output_window_sha256"],
                    "transform_receipt_sha256": _sha256(_canonical_json(transform)),
                    "transform_receipt": transform,
                    "base_action_sha256": base_action_sha256,
                    "base_action": base_action,
                    "policy_input_sha256": policy_sha,
                    "policy_input_shape": [708],
                    "policy_input_dtype": "<f4",
                    "actor_input_sha256": policy_sha,
                    "critic_input_sha256": policy_sha,
                    "controller_state_sha256": bindings["residual_controller_state_sha256"],
                    "actor_output_sha256": _float32_sha(actor_output),
                    "actor_output_shape": [17],
                    "actor_output_dtype": "<f4",
                    "actor_output": actor_output,
                    "critic_output_sha256": _float32_sha([critic_value]),
                    "critic_output_shape": [1, 1],
                    "critic_output_dtype": "<f4",
                    "critic_value": critic_value,
                    "composed_action_sha256": _float32_sha(composed_action),
                    "composed_action_dtype": "<f4",
                    "composed_action_shape": [17],
                    "composed_action": composed_action,
                }
            )

    summary_conditions = {}
    for condition in conditions:
        changed = 0 if condition == "C_exact" else 8
        squared_differences = []
        maximum_difference = 0.0
        for frame in frames:
            exact = next(
                record
                for record in records
                if record["snapshot_frame"] == frame and record["condition_id"] == "C_exact"
            )
            observed = next(
                record
                for record in records
                if record["snapshot_frame"] == frame and record["condition_id"] == condition
            )
            for observed_value, exact_value in zip(
                observed["composed_action"], exact["composed_action"], strict=True
            ):
                difference = observed_value - exact_value
                squared_differences.append(difference * difference)
                maximum_difference = max(maximum_difference, abs(difference))
        summary_conditions[condition] = {
            "snapshot_count": 8,
            "policy_input_changed_count_vs_exact": changed,
            "actor_output_changed_count_vs_exact": changed,
            "composed_action_changed_count_vs_exact": changed,
            "critic_output_changed_count_vs_exact": changed,
            "composed_action_rms_delta_vs_exact": math.sqrt(
                math.fsum(squared_differences) / len(squared_differences)
            ),
            "composed_action_max_abs_delta_vs_exact": maximum_difference,
            "critic_mean_abs_delta_vs_exact": abs(_float32(condition_values[condition])),
        }
    summary = {
        "mechanistic_gate_passed": True,
        "mechanistic_gate_rule": (
            "every_corrupted_arm_changes_policy_input_actor_output_and_composed_"
            "float32_control_at_all_8_predeclared_observations/v1"
        ),
        "conditions": summary_conditions,
    }
    design = {
        "protocol_id": "experiment_002a_matched_observation_numeric_reference_probe/v1",
        "snapshot_source": "registered_Minari_Humanoid-v5_expert_episode_0_observations/v1",
        "snapshot_frames": list(frames),
        "condition_ids": list(conditions),
        "shuffle_seed": 260_907,
        "shift_frames": 250,
        "reference_horizon_steps": 8,
        "base_action_scale": 0.4,
        "residual_scale": 0.08,
        "composition": "clip_float32(base_float32 + float32(0.08) * residual_float32)",
        "independent_unit": "one_existing_local_checkpoint",
        "repeated_measures": "8_predeclared_matched_source_observations_x_4_input_arms",
        "human_review_required_for_execution": False,
        "reason_review_not_required": (
            "offline_local_exploratory_input_sensitivity_probe_with_no_training_or_formal_seed_use"
        ),
    }
    design_sha256 = _sha256(_canonical_json(design))
    analysis_payload = {
        "design_sha256": design_sha256,
        "bindings": bindings,
        "snapshots": snapshots,
        "records": records,
        "summary": summary,
    }
    report = {
        "schema_version": 1,
        "experiment_id": "experiment_002a",
        "protocol_id": "experiment_002a_matched_observation_numeric_reference_probe/v1",
        "evidence_class": "exploratory",
        "claim_status": "numeric_reference_input_sensitivity_only_not_tracking_or_oracle_quality",
        "formal_oracle_comparison_authorized": False,
        "tracker_behavior_or_stability_established": False,
        "oracle_quality_established": False,
        "dataset_license_status": "unresolved_no_redistribution_or_formal_admission",
        "design": design,
        "design_sha256": design_sha256,
        "bindings": bindings,
        "source_projection_receipt": projection_receipt,
        "base_controller_receipt": base_receipt,
        "residual_controller_receipt": residual_receipt,
        "snapshots": snapshots,
        "records": records,
        "summary": summary,
        "analysis_payload_sha256": _sha256(_canonical_json(analysis_payload)),
        "next_gate": (
            "train_and_admit_a_stable_time_varying_reference_tracker_then_run_"
            "expected_direction_behavioral_ablation_002b"
        ),
    }
    path = root / reference_probe_module.REFERENCE_PROBE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json(report))
    return path


def _pin_reference_probe(path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        reference_probe_module,
        "_REVIEWED_REPORT_SHA256",
        _sha256(path.read_bytes()),
    )


def test_ui_serves_static_shell_and_read_only_evidence_api(tmp_path: Path) -> None:
    extractions = tmp_path / "extractions"
    extractions.mkdir()
    _write_record(extractions, "2600.00001", "Phase recovery")
    database = tmp_path / "graph.db"
    build_index(extractions, database)

    server = create_server(
        port=0,
        project_root=tmp_path,
        database=database,
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        with urlopen(base, timeout=2) as response:
            html = response.read().decode("utf-8")
            assert "Native baseline gates" in html
            assert response.headers["Content-Security-Policy"]
            assert response.headers["Cross-Origin-Resource-Policy"] == "same-origin"
        health = _get_json(f"{base}/health")
        status = _get_json(f"{base}/api/status")
        query = _get_json(f"{base}/api/research/query?q=phase%20recovery")
        exploration = _get_json(f"{base}/api/exploration/latest")
        reference_probe = _get_json(f"{base}/api/experiments/002a")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert health == {"authority": "read_only", "status": "ok"}
    assert status["evidence_class"] == "none"
    assert status["measured_evidence"] == []
    assert status["evidence_receipts"] == []
    assert query["results"]
    assert exploration["state"] == "unavailable"
    assert reference_probe == {
        "authority": "local_exploratory_mechanistic_probe",
        "detail": "no local Experiment 002A report is present",
        "state": "unavailable",
    }


def test_ui_serves_compact_verified_reference_probe(tmp_path: Path, monkeypatch) -> None:
    report_path = _write_reference_probe(tmp_path)
    _pin_reference_probe(report_path, monkeypatch)
    server = create_server(port=0, project_root=tmp_path, database=tmp_path / "missing.db")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        probe = _get_json(f"http://{host}:{port}/api/experiments/002a")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert probe["state"] == "available"
    assert probe["authority"] == "local_exploratory_mechanistic_probe"
    assert probe["mechanistic_gate_passed"] is True
    assert probe["snapshot_count"] == 8
    assert probe["condition_count"] == 4
    assert [item["condition_id"] for item in probe["conditions"]] == [
        "C_exact",
        "C_zero_input",
        "C_shuffle_input",
        "C_shift_input",
    ]
    assert probe["formal_oracle_comparison_authorized"] is False
    assert probe["tracker_behavior_or_stability_established"] is False
    assert probe["oracle_quality_established"] is False
    assert probe["receipts"]["report_sha256"] == _sha256(report_path.read_bytes())
    serialized = json.dumps(probe)
    assert str(tmp_path) not in serialized
    assert '"records"' not in serialized
    assert '"source_projection_receipt"' not in serialized
    assert '"base_action"' not in serialized
    assert '"composed_action"' not in serialized


def test_ui_rejects_rebound_reference_probe_summary(tmp_path: Path) -> None:
    report_path = _write_reference_probe(tmp_path)
    report = json.loads(report_path.read_bytes())
    report["summary"]["conditions"]["C_zero_input"]["composed_action_changed_count_vs_exact"] = 0
    report_path.write_bytes(_canonical_json(report))

    probe = reference_probe_module.local_reference_probe_status(tmp_path)

    assert probe["state"] == "rejected"
    assert probe["authority"] == "local_exploratory_mechanistic_probe"
    assert "summary count" in str(probe["detail"])


def test_ui_rejects_reference_probe_with_forged_analysis_hash(tmp_path: Path) -> None:
    report_path = _write_reference_probe(tmp_path)
    report = json.loads(report_path.read_bytes())
    report["analysis_payload_sha256"] = "0" * 64
    report_path.write_bytes(_canonical_json(report))

    probe = reference_probe_module.local_reference_probe_status(tmp_path)

    assert probe["state"] == "rejected"
    assert "analysis payload" in str(probe["detail"])


def test_ui_rejects_fully_rebound_reference_schema_identity(tmp_path: Path) -> None:
    report_path = _write_reference_probe(tmp_path)
    report = json.loads(report_path.read_bytes())
    forged_schema_sha256 = "0" * 64
    projection = report["source_projection_receipt"]
    projection["reference_schema_sha256"] = forged_schema_sha256
    projection["projection_mapping_sha256"] = _sha256(
        _canonical_json(
            {
                "projection_id": projection["projection_id"],
                "indices": projection["projection_indices"],
                "source_width": 348,
                "target_schema_sha256": forged_schema_sha256,
            }
        )
    )
    report["bindings"]["reference_schema_sha256"] = forged_schema_sha256
    report["bindings"]["minari_projection_receipt_sha256"] = _sha256(_canonical_json(projection))
    analysis_payload = {
        "design_sha256": report["design_sha256"],
        "bindings": report["bindings"],
        "snapshots": report["snapshots"],
        "records": report["records"],
        "summary": report["summary"],
    }
    report["analysis_payload_sha256"] = _sha256(_canonical_json(analysis_payload))
    report_path.write_bytes(_canonical_json(report))

    probe = reference_probe_module.local_reference_probe_status(tmp_path)

    assert probe["state"] == "rejected"
    assert "source projection receipt differs" in str(probe["detail"])


def test_ui_rejects_fully_rebound_report_outside_reviewed_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report_path = _write_reference_probe(tmp_path)
    _pin_reference_probe(report_path, monkeypatch)
    report = json.loads(report_path.read_bytes())
    rebound_observation_sha256 = "0" * 64
    report["snapshots"][0]["observation_sha256"] = rebound_observation_sha256
    for record in report["records"]:
        if record["snapshot_frame"] == 0:
            record["observation_sha256"] = rebound_observation_sha256
    report["bindings"]["matched_observation_set_sha256"] = _sha256(
        _canonical_json(report["snapshots"])
    )
    analysis_payload = {
        "design_sha256": report["design_sha256"],
        "bindings": report["bindings"],
        "snapshots": report["snapshots"],
        "records": report["records"],
        "summary": report["summary"],
    }
    report["analysis_payload_sha256"] = _sha256(_canonical_json(analysis_payload))
    report_path.write_bytes(_canonical_json(report))

    probe = reference_probe_module.local_reference_probe_status(tmp_path)

    assert probe["state"] == "rejected"
    assert "report identity" in str(probe["detail"])


def test_ui_rejects_reference_probe_with_forged_actor_bytes(tmp_path: Path) -> None:
    report_path = _write_reference_probe(tmp_path)
    report = json.loads(report_path.read_bytes())
    report["records"][0]["actor_output"][0] = 0.25
    analysis_payload = {
        "design_sha256": report["design_sha256"],
        "bindings": report["bindings"],
        "snapshots": report["snapshots"],
        "records": report["records"],
        "summary": report["summary"],
    }
    report["analysis_payload_sha256"] = _sha256(_canonical_json(analysis_payload))
    report_path.write_bytes(_canonical_json(report))

    probe = reference_probe_module.local_reference_probe_status(tmp_path)

    assert probe["state"] == "rejected"
    assert "actor output SHA-256" in str(probe["detail"])


def test_ui_rejects_reference_probe_after_bound_source_changes(tmp_path: Path) -> None:
    _write_reference_probe(tmp_path)
    source = tmp_path / "src/oracle_composition/experiments/reference_causal_probe.py"
    source.write_bytes(b"changed after report")

    probe = reference_probe_module.local_reference_probe_status(tmp_path)

    assert probe["state"] == "rejected"
    assert "bound source file changed" in str(probe["detail"])


def test_ui_rejects_symlinked_reference_probe_parent(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    _write_reference_probe(outside)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "experiment_002a").symlink_to(
        outside / "artifacts/experiment_002a",
        target_is_directory=True,
    )

    probe = reference_probe_module.local_reference_probe_status(tmp_path)

    assert probe["state"] == "rejected"
    assert probe["authority"] == "local_exploratory_mechanistic_probe"


def test_ui_serves_only_integrity_checked_local_exploration(tmp_path: Path, monkeypatch) -> None:
    _write_local_exploration(tmp_path, monkeypatch)
    server = create_server(port=0, project_root=tmp_path, database=tmp_path / "missing.db")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        exploration = _get_json(f"{base}/api/exploration/latest")
        with urlopen(f"{base}/local-evidence/holdout-summary.png", timeout=2) as response:
            assert response.status == 200
            assert response.headers["Content-Type"] == "image/png"
            assert response.read() == b"\x89PNG\r\n\x1a\nsynthetic figure"
        request = Request(
            f"{base}/local-evidence/lateral-seed3000-four-arm.mp4",
            headers={"Range": "bytes=4-8"},
        )
        with urlopen(request, timeout=2) as response:
            assert response.status == 206
            assert response.headers["Accept-Ranges"] == "bytes"
            assert response.headers["Content-Range"] == "bytes 4-8/27"
            assert response.read() == b"ftypi"
        head = Request(
            f"{base}/local-evidence/lateral-seed3000-four-arm.mp4",
            method="HEAD",
        )
        with urlopen(head, timeout=2) as response:
            assert response.status == 200
            assert response.headers["Content-Length"] == "27"
            assert response.read() == b""
        invalid_range = Request(
            f"{base}/local-evidence/lateral-seed3000-four-arm.mp4",
            headers={"Range": "bytes=1-2,4-5"},
        )
        try:
            urlopen(invalid_range, timeout=2)
        except HTTPError as exc:
            assert exc.code == 416
        else:
            raise AssertionError("multiple byte ranges must fail closed")
        hostile_host = Request(
            f"{base}/local-evidence/holdout-summary.png",
            headers={"Host": "attacker.example"},
        )
        try:
            urlopen(hostile_host, timeout=2)
        except HTTPError as exc:
            assert exc.code == 421
        else:
            raise AssertionError("unexpected Host headers must fail closed")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert exploration["state"] == "available"
    assert exploration["authority"] == "local_exploration_only"
    assert exploration["reference_admission"] == "Tier-K_not_admitted"
    assert exploration["completed_runs"] == 300
    assert exploration["comparisons"][1]["passed"] is False
    assert str(tmp_path) not in json.dumps(exploration)


def test_ui_rejects_tampered_local_exploration_and_media(tmp_path: Path, monkeypatch) -> None:
    directory = _write_local_exploration(tmp_path, monkeypatch)
    (directory / "runs.jsonl").write_bytes(b"tampered\n")
    server = create_server(port=0, project_root=tmp_path, database=tmp_path / "missing.db")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    base = f"http://{host}:{port}"
    try:
        exploration = _get_json(f"{base}/api/exploration/latest")
        try:
            urlopen(f"{base}/local-evidence/holdout-summary.png", timeout=2)
        except HTTPError as exc:
            assert exc.code == 404
        else:
            raise AssertionError("tampered bundles must not expose media")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert exploration["state"] == "rejected"
    assert exploration["authority"] == "local_exploration_only"


def test_ui_normalizes_oversized_local_json_as_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "artifacts/exploration/phase_oracle_holdout_v0"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text('{"value":' + "9" * 5_000 + "}")

    status = local_evidence_module.local_exploration_status(tmp_path)

    assert status["state"] == "rejected"
    assert "JSON is invalid" in str(status["detail"])


def test_local_evidence_number_normalizes_float_overflow() -> None:
    with pytest.raises(local_evidence_module.LocalEvidenceError, match="must be finite"):
        local_evidence_module._number(10**400, "hostile")


def test_ui_rejects_rebound_run_ledger_outside_reviewed_identity(
    tmp_path: Path, monkeypatch
) -> None:
    directory = _write_local_exploration(tmp_path, monkeypatch)
    first_row = (directory / "runs.jsonl").read_bytes().splitlines(keepends=True)[0]
    (directory / "runs.jsonl").write_bytes(first_row)
    runs_sha = _sha256(first_row)
    _rewrite_json(
        directory / "summary.json",
        lambda value: value.__setitem__("runs_sha256", runs_sha),
    )
    _rewrite_json(
        directory / "holdout_summary.figure_manifest.json",
        lambda value: value.__setitem__("source_runs_sha256", runs_sha),
    )
    _rewrite_json(
        directory / "lateral_seed3000_four_arm.video_receipt.json",
        lambda value: value.__setitem__("runs_sha256", runs_sha),
    )

    server = create_server(port=0, project_root=tmp_path, database=tmp_path / "missing.db")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        exploration = _get_json(f"http://{host}:{port}/api/exploration/latest")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert exploration["state"] == "rejected"
    assert "run-ledger identity" in str(exploration["detail"])


def test_ui_rejects_smaller_design_even_when_all_hashes_are_rebound(
    tmp_path: Path, monkeypatch
) -> None:
    directory = _write_local_exploration(tmp_path, monkeypatch)
    _rebind_manifest(directory, lambda value: value["arms"].pop("T0_G1"))

    server = create_server(port=0, project_root=tmp_path, database=tmp_path / "missing.db")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        exploration = _get_json(f"http://{host}:{port}/api/exploration/latest")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert exploration["state"] == "rejected"
    assert "manifest identity" in str(exploration["detail"])


def test_ui_rejects_valid_signature_media_rebound_outside_reviewed_identity(
    tmp_path: Path, monkeypatch
) -> None:
    directory = _write_local_exploration(tmp_path, monkeypatch)
    replacement = b"\x00\x00\x00\x18ftypisomdifferent-valid-signature-video"
    (directory / "lateral_seed3000_four_arm.mp4").write_bytes(replacement)
    _rewrite_json(
        directory / "lateral_seed3000_four_arm.video_receipt.json",
        lambda value: value.__setitem__("video_sha256", _sha256(replacement)),
    )

    server = create_server(port=0, project_root=tmp_path, database=tmp_path / "missing.db")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        exploration = _get_json(f"http://{host}:{port}/api/exploration/latest")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert exploration["state"] == "rejected"
    assert "media receipt identity" in str(exploration["detail"])


def test_ui_rejects_symlinked_local_media(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "project"
    directory = _write_local_exploration(project, monkeypatch)
    target = tmp_path / "outside.png"
    target.write_bytes((directory / "holdout_summary.png").read_bytes())
    (directory / "holdout_summary.png").unlink()
    (directory / "holdout_summary.png").symlink_to(target)

    server = create_server(port=0, project_root=project, database=project / "missing.db")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        exploration = _get_json(f"http://{host}:{port}/api/exploration/latest")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert exploration["state"] == "rejected"


def test_local_evidence_reader_rejects_fifo_without_blocking(tmp_path: Path) -> None:
    fifo = tmp_path / "source.fifo"
    os.mkfifo(fifo)

    with pytest.raises(local_evidence_module.LocalEvidenceError, match="regular file"):
        local_evidence_module._bounded_file(tmp_path, Path("source.fifo"), limit=8)


def test_ui_rejects_non_loopback_bind(tmp_path: Path, monkeypatch) -> None:
    _write_local_exploration(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="loopback"):
        create_server(
            host="0.0.0.0",
            port=0,
            project_root=tmp_path,
            database=tmp_path / "missing.db",
        )


def test_ui_reports_corrupt_index_as_not_built(tmp_path: Path) -> None:
    database = tmp_path / "graph.db"
    database.write_bytes(b"not a sqlite database")
    server = create_server(port=0, project_root=tmp_path, database=database)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        stats = _get_json(f"http://{host}:{port}/api/research/stats")
        status = _get_json(f"http://{host}:{port}/api/status")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert stats["state"] == "not_built"
    assert "corrupt or incomplete" in stats["detail"]
    assert status["knowledge_graph"]["state"] == "not_built"
