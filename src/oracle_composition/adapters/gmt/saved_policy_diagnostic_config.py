"""Data-only admission for the fixed Study020 saved-policy diagnostic."""

from __future__ import annotations

import hashlib
import io
import math
import os
import stat
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oracle_composition.harness.contract import read_json_object

from .io import GMTAdmissionError, read_verified_bytes, sha256_file, validate_zip_members
from .training_normalizer import (
    FIXED_NORMALIZER_STATE_SHA256,
    MAX_NUMERIC_POLICY_BYTES,
    validate_policy_normalizer_archive,
)

CONFIG_ARTIFACT = "gmt_g1_saved_policy_diagnostic_config/v1"
RUN_ARTIFACT = "gmt_g1_saved_policy_diagnostic/v1"
RUN_MANIFEST_FILENAME = "saved_policy_diagnostic_manifest.json"
NOISE_FILENAME = "paired_standard_normal.npz"
NOISE_GENERATOR_ID = "torch_cpu_float32_rowwise_normal_1x23/v1"
RESET_SEED = 20_260_906
HORIZON_STEPS = 1_000
ACTION_DIMENSION = 23
NOISE_SEEDS = tuple(range(20_260_920, 20_260_936))
FIRST_POLICY_BY_SEED = {
    seed: (
        "initial"
        if seed
        in {
            20_260_920,
            20_260_921,
            20_260_924,
            20_260_925,
            20_260_926,
            20_260_927,
            20_260_932,
            20_260_934,
        }
        else "final"
    )
    for seed in NOISE_SEEDS
}
RETAINED_SOURCE_COMMIT = "e57f220a7694ab8b5a16758fcae1a5ee6a12d2d2"
RETAINED_HASHES = {
    "course_manifest_sha256": "33b5e25ea8dfa7a8e3a4f520e7386edce067b920d75201e8ffb81581bd96f705",
    "resource_receipt_sha256": "c06fbc88be3098171019d639ddab2ce3e3f7b378fa0e3da43cbea81e7dbd2d0e",
    "course_config_sha256": "a9e52ba706e4ddfd394c757cf525bf210d553906430c7f08e11e0495affdf289",
    "initial_policy_sha256": "afc5f1f20e0a616e399d705676dafff4ccd1699bb0b5f59e6e36a220284802ae",
    "final_policy_sha256": "01421e9af87edde042048574b995e1b553fd4ec04bd7e46d546c5586444d4729",
}
RETAINED_FILENAMES = {
    "course_manifest": "course_run_manifest.json",
    "resource_receipt": "gmt_probe_resource_receipt_v1.json",
    "course_config": "input_config.json",
    "initial_policy": "initial_residual_policy.npz",
    "final_policy": "final_residual_policy.npz",
}
RETAINED_DETERMINISTIC_OUTPUTS = {
    "initial": {
        "frames": "a8f0c4bbe35023898984e17f61fb6a7215149ee36a1a259e55bab0fcbf7fb23e",
        "trajectory": "adb9dec34d17e728f297548b8fd0b2e77016b47e11ec0487e1a961e7cdb143a3",
        "evaluation": "035335092381cd5220a6d07287fde4444e779d1b68eca0ff6c778aac4c340552",
    },
    "final": {
        "frames": "38729578d079db6105bc3b494a4fb557b4313ae962f38cad30b064f0d0d73947",
        "trajectory": "f8c03064f6c27bbf670ccb18aae41fafb9e4aaa8a9ab5e7d3046a86a9a41d9b9",
        "evaluation": "06d14923cf038401bc28bb46c4e6cb0e1ddc9aa90003dffe6d1c77b79fd955e2",
    },
}
POLICY_STATE_DICT_KEY_COUNT = 19
POLICY_STATE_SCHEMA_SHA256 = "485d1e2b4922882c29894ebe35c6edac2a29c7e61d6e5ca64fdd354271d4d05e"
POLICY_LOG_STD_SHA256 = {
    "initial": "95de3712796cd6dc12d9a20b67b871c071efadb3d5230165b442a8e3db30393f",
    "final": "6e42dbcf2bf33f6e18a8a58951a3e635a64ed86c02903fa7ee6f06e21fca1a7b",
}
PARITY_ARTIFACT = "gmt_g1_saved_policy_deterministic_parity/v1"
RUN_CLAIMS = {
    "evaluation_only": True,
    "training_performed": False,
    "gradient_steps": 0,
    "optimizer_steps": 0,
    "checkpoint_selection_performed": False,
    "heldout_generalization_tested": False,
    "automatic_adoption": False,
}
RUN_RUNTIME = {
    "python": "3.13.15",
    "numpy": "2.5.2",
    "torch": "2.14.0",
    "stable_baselines3": "2.9.0",
    "device": "cpu",
    "sampling_generator": NOISE_GENERATOR_ID,
}

_CONFIG_FIELDS = {
    "schema_version",
    "artifact",
    "retained_run_root",
    "retained",
    "protocol",
    "sampling",
}
_RETAINED_FIELDS = {
    "course_manifest_sha256",
    "resource_receipt_sha256",
    "course_config_sha256",
    "policies",
}
_SAMPLING_FIELDS = {
    "reset_seed",
    "horizon_steps",
    "action_dimension",
    "noise_seeds",
    "first_policy_by_seed",
    "generator_id",
    "noise",
}
_MAX_CONFIG_BYTES = 256 * 1024
_MAX_MANIFEST_BYTES = 1 * 1024**2
_MAX_RESOURCE_BYTES = 512 * 1024
_MAX_PROTOCOL_BYTES = 1 * 1024**2
_MAX_NOISE_BYTES = 2 * 1024**2
_MAX_OUTPUT_MEMBER_BYTES = 128 * 1024**2
_SUPERVISOR_FILES = {
    "child_stdout.json",
    "child_stderr.log",
}
_OPTIONAL_RESOURCE_RECEIPT = "gmt_probe_resource_receipt_v1.json"


@dataclass(frozen=True, slots=True)
class FileBinding:
    path: Path
    sha256: str
    size: int

    def receipt(self) -> dict[str, object]:
        return {"path": str(self.path), "sha256": self.sha256, "size": self.size}


@dataclass(frozen=True, slots=True)
class SavedPolicyDiagnosticConfig:
    raw: dict[str, Any]
    encoded: bytes
    sha256: str
    retained_root: Path
    course_manifest: FileBinding
    resource_receipt: FileBinding
    course_config: FileBinding
    policies: dict[str, FileBinding]
    protocol: FileBinding
    noise: FileBinding

    def resource_binding(self) -> dict[str, object]:
        return {
            "artifact": CONFIG_ARTIFACT,
            "config": {"sha256": self.sha256, "size": len(self.encoded)},
            "retained_run_root": str(self.retained_root),
            "retained": {
                "source_commit": RETAINED_SOURCE_COMMIT,
                "course_manifest": self.course_manifest.receipt(),
                "resource_receipt": self.resource_receipt.receipt(),
                "course_config": self.course_config.receipt(),
                "policies": {
                    name: binding.receipt() for name, binding in sorted(self.policies.items())
                },
            },
            "protocol": self.protocol.receipt(),
            "sampling": self.raw["sampling"],
        }


@dataclass(frozen=True, slots=True)
class DiagnosticEpisode:
    sequence_index: int
    kind: str
    label: str
    policy: str
    noise_seed: int | None


def _sha256(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be one lowercase SHA-256")
    return value


def _object(value: object, fields: set[str], *, field: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"{field} fields differ")
    return value


def _path(value: object, *, field: str) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError(f"{field} path is malformed")
    return Path(value)


def _direct_path(path: Path, *, directory: bool, field: str) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{field} path must be absolute")
    direct = Path(os.path.abspath(os.fspath(path)))
    try:
        metadata = direct.lstat()
        resolved = direct.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{field} is unavailable") from exc
    required = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode) or not required or resolved != direct:
        kind = "directory" if directory else "file"
        raise ValueError(f"{field} must be a direct absolute {kind}")
    return direct


def _binding(path: Path, expected_sha256: str, *, maximum: int, field: str) -> FileBinding:
    direct = _direct_path(path, directory=False, field=field)
    expected = _sha256(expected_sha256, field=f"{field} SHA-256")
    encoded = read_verified_bytes(direct, expected, maximum_size=maximum)
    return FileBinding(direct, expected, len(encoded))


def _json(binding: FileBinding, *, field: str) -> dict[str, Any]:
    raw, encoded = read_json_object(binding.path)
    if len(encoded) != binding.size or hashlib.sha256(encoded).hexdigest() != binding.sha256:
        raise ValueError(f"{field} differs from its admitted bytes")
    return raw


def _validate_retained_lineage(
    root: Path,
    retained: dict[str, Any],
) -> tuple[FileBinding, FileBinding, FileBinding, dict[str, FileBinding]]:
    expected_policies = {
        "initial": RETAINED_HASHES["initial_policy_sha256"],
        "final": RETAINED_HASHES["final_policy_sha256"],
    }
    policies = _object(retained["policies"], set(expected_policies), field="retained policies")
    if any(policies[name] != digest for name, digest in expected_policies.items()):
        raise ValueError("retained policy hashes differ from exact Study018")
    for key in _RETAINED_FIELDS - {"policies"}:
        if retained[key] != RETAINED_HASHES[key]:
            raise ValueError("retained hashes differ from exact Study018")

    manifest = _binding(
        root / RETAINED_FILENAMES["course_manifest"],
        retained["course_manifest_sha256"],
        maximum=_MAX_MANIFEST_BYTES,
        field="retained course manifest",
    )
    resource = _binding(
        root / RETAINED_FILENAMES["resource_receipt"],
        retained["resource_receipt_sha256"],
        maximum=_MAX_RESOURCE_BYTES,
        field="retained resource receipt",
    )
    course = _binding(
        root / RETAINED_FILENAMES["course_config"],
        retained["course_config_sha256"],
        maximum=_MAX_CONFIG_BYTES,
        field="retained course config",
    )
    policy_bindings = {
        name: _binding(
            root / RETAINED_FILENAMES[f"{name}_policy"],
            digest,
            maximum=MAX_NUMERIC_POLICY_BYTES,
            field=f"retained {name} policy",
        )
        for name, digest in expected_policies.items()
    }
    for binding in policy_bindings.values():
        validate_policy_normalizer_archive(
            read_verified_bytes(
                binding.path,
                binding.sha256,
                expected_size=binding.size,
                maximum_size=MAX_NUMERIC_POLICY_BYTES,
            )
        )

    manifest_raw = _json(manifest, field="retained course manifest")
    outputs = manifest_raw.get("outputs")
    if (
        manifest_raw.get("artifact") != "gmt_g1_course_development_run"
        or manifest_raw.get("status") != "completed"
        or manifest_raw.get("input_config_sha256") != course.sha256
        or type(outputs) is not dict
        or outputs.get("input_config.json") != course.sha256
        or outputs.get("initial_residual_policy.npz") != policy_bindings["initial"].sha256
        or outputs.get("final_residual_policy.npz") != policy_bindings["final"].sha256
        or outputs.get("zero_residual_frames.jsonl")
        != RETAINED_DETERMINISTIC_OUTPUTS["initial"]["frames"]
        or outputs.get("zero_residual_trajectory.npz")
        != RETAINED_DETERMINISTIC_OUTPUTS["initial"]["trajectory"]
        or outputs.get("zero_residual_evaluation.json")
        != RETAINED_DETERMINISTIC_OUTPUTS["initial"]["evaluation"]
        or outputs.get("final_policy_frames.jsonl")
        != RETAINED_DETERMINISTIC_OUTPUTS["final"]["frames"]
        or outputs.get("final_policy_trajectory.npz")
        != RETAINED_DETERMINISTIC_OUTPUTS["final"]["trajectory"]
        or outputs.get("final_policy_evaluation.json")
        != RETAINED_DETERMINISTIC_OUTPUTS["final"]["evaluation"]
    ):
        raise ValueError("retained Study018 manifest lineage differs")
    resource_raw = _json(resource, field="retained resource receipt")
    resource_artifacts = resource_raw.get("artifacts")
    if not isinstance(resource_artifacts, Mapping):
        raise ValueError("retained Study018 native resource artifacts are malformed")
    resource_manifest = resource_artifacts.get("course_manifest")
    resource_outputs = resource_artifacts.get("outputs")
    if (
        resource_raw.get("status") != "succeeded"
        or resource_raw.get("commit") != RETAINED_SOURCE_COMMIT
        or not isinstance(resource_manifest, Mapping)
        or not isinstance(resource_outputs, Mapping)
        or resource_manifest.get("sha256") != manifest.sha256
    ):
        raise ValueError("retained Study018 native resource lineage differs")
    for name, binding in {
        "input_config.json": course,
        "initial_residual_policy.npz": policy_bindings["initial"],
        "final_residual_policy.npz": policy_bindings["final"],
    }.items():
        item = resource_outputs.get(name)
        if not isinstance(item, Mapping) or item.get("sha256") != binding.sha256:
            raise ValueError("retained Study018 native resource lineage differs")
    course_raw = _json(course, field="retained course config")
    task = course_raw.get("task")
    if (
        course_raw.get("schema_version") != 5
        or course_raw.get("mode") != "train"
        or course_raw.get("seed") != RESET_SEED
        or course_raw.get("training_steps") != 131_072
        or not isinstance(task, Mapping)
        or task.get("horizon_steps") != HORIZON_STEPS
        or course_raw.get("runtime")
        != {"schema_version": 1, "profile_id": "gmt_g1_four_state_finite_horizon_course/v1"}
    ):
        raise ValueError("retained Study018 course contract differs")
    return manifest, resource, course, policy_bindings


def load_noise_archive(binding: FileBinding) -> np.ndarray:
    encoded = read_verified_bytes(
        binding.path,
        binding.sha256,
        expected_size=binding.size,
        maximum_size=_MAX_NOISE_BYTES,
    )
    try:
        with zipfile.ZipFile(io.BytesIO(encoded), "r") as archive:
            members = validate_zip_members(archive, expected_count=2, maximum_member_size=2**21)
            if set(members) != {"seeds.npy", "standard_normal.npy"}:
                raise ValueError("Study020 noise archive members differ")
        with np.load(io.BytesIO(encoded), allow_pickle=False) as archive:
            seeds = archive["seeds"]
            noise = archive["standard_normal"]
            if (
                seeds.shape != (len(NOISE_SEEDS),)
                or seeds.dtype.str != "<i8"
                or seeds.tolist() != list(NOISE_SEEDS)
                or noise.shape != (len(NOISE_SEEDS), HORIZON_STEPS, ACTION_DIMENSION)
                or noise.dtype.str != "<f4"
                or not noise.flags.c_contiguous
                or not np.isfinite(noise).all()
            ):
                raise ValueError("Study020 noise arrays differ")
            return np.ascontiguousarray(noise)
    except (GMTAdmissionError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError("Study020 noise archive is malformed") from exc


def load_saved_policy_diagnostic_config(path: Path) -> SavedPolicyDiagnosticConfig:
    config_path = _direct_path(path, directory=False, field="Study020 config")
    raw, encoded = read_json_object(config_path)
    if len(encoded) > _MAX_CONFIG_BYTES:
        raise ValueError("Study020 config exceeds its byte bound")
    root = _object(raw, _CONFIG_FIELDS, field="Study020 config")
    if root["schema_version"] != 1 or root["artifact"] != CONFIG_ARTIFACT:
        raise ValueError("Study020 config identity differs")
    retained_root = _direct_path(
        _path(root["retained_run_root"], field="retained run root"),
        directory=True,
        field="retained run root",
    )
    retained = _object(root["retained"], _RETAINED_FIELDS, field="retained binding")
    manifest, resource, course, policies = _validate_retained_lineage(retained_root, retained)

    protocol_raw = _object(root["protocol"], {"path", "sha256"}, field="protocol binding")
    protocol = _binding(
        _path(protocol_raw["path"], field="Study020 protocol"),
        protocol_raw["sha256"],
        maximum=_MAX_PROTOCOL_BYTES,
        field="Study020 protocol",
    )
    sampling = _object(root["sampling"], _SAMPLING_FIELDS, field="sampling contract")
    expected_first = {str(seed): FIRST_POLICY_BY_SEED[seed] for seed in NOISE_SEEDS}
    if (
        sampling["reset_seed"] != RESET_SEED
        or sampling["horizon_steps"] != HORIZON_STEPS
        or sampling["action_dimension"] != ACTION_DIMENSION
        or sampling["noise_seeds"] != list(NOISE_SEEDS)
        or sampling["first_policy_by_seed"] != expected_first
        or sampling["generator_id"] != NOISE_GENERATOR_ID
    ):
        raise ValueError("Study020 fixed sampling contract differs")
    noise_raw = _object(sampling["noise"], {"path", "sha256", "size"}, field="noise binding")
    if type(noise_raw["size"]) is not int or not 0 < noise_raw["size"] <= _MAX_NOISE_BYTES:
        raise ValueError("Study020 noise byte size is outside its bound")
    noise = _binding(
        _path(noise_raw["path"], field="Study020 paired noise"),
        noise_raw["sha256"],
        maximum=_MAX_NOISE_BYTES,
        field="Study020 paired noise",
    )
    if noise.size != noise_raw["size"]:
        raise ValueError("Study020 noise byte size differs")
    load_noise_archive(noise)
    return SavedPolicyDiagnosticConfig(
        raw=root,
        encoded=encoded,
        sha256=hashlib.sha256(encoded).hexdigest(),
        retained_root=retained_root,
        course_manifest=manifest,
        resource_receipt=resource,
        course_config=course,
        policies=policies,
        protocol=protocol,
        noise=noise,
    )


def diagnostic_config_payload(
    *, retained_root: Path, protocol_path: Path, noise_path: Path
) -> dict[str, object]:
    retained = _direct_path(retained_root, directory=True, field="retained run root")
    protocol = _direct_path(protocol_path, directory=False, field="Study020 protocol")
    noise = _direct_path(noise_path, directory=False, field="Study020 paired noise")
    noise_binding = FileBinding(noise, sha256_file(noise), noise.stat().st_size)
    load_noise_archive(noise_binding)
    return {
        "schema_version": 1,
        "artifact": CONFIG_ARTIFACT,
        "retained_run_root": str(retained),
        "retained": {
            "course_manifest_sha256": RETAINED_HASHES["course_manifest_sha256"],
            "resource_receipt_sha256": RETAINED_HASHES["resource_receipt_sha256"],
            "course_config_sha256": RETAINED_HASHES["course_config_sha256"],
            "policies": {
                "initial": RETAINED_HASHES["initial_policy_sha256"],
                "final": RETAINED_HASHES["final_policy_sha256"],
            },
        },
        "protocol": {"path": str(protocol), "sha256": sha256_file(protocol)},
        "sampling": {
            "reset_seed": RESET_SEED,
            "horizon_steps": HORIZON_STEPS,
            "action_dimension": ACTION_DIMENSION,
            "noise_seeds": list(NOISE_SEEDS),
            "first_policy_by_seed": {str(seed): FIRST_POLICY_BY_SEED[seed] for seed in NOISE_SEEDS},
            "generator_id": NOISE_GENERATOR_ID,
            "noise": noise_binding.receipt(),
        },
    }


def diagnostic_episode_schedule() -> tuple[DiagnosticEpisode, ...]:
    episodes = [
        DiagnosticEpisode(0, "deterministic_parity", "initial_deterministic", "initial", None),
        DiagnosticEpisode(1, "deterministic_parity", "final_deterministic", "final", None),
    ]
    for seed in NOISE_SEEDS:
        first = FIRST_POLICY_BY_SEED[seed]
        second = "final" if first == "initial" else "initial"
        for policy in (first, second):
            episodes.append(
                DiagnosticEpisode(
                    len(episodes),
                    "sampled",
                    f"sample_{seed}_{policy}",
                    policy,
                    seed,
                )
            )
    return tuple(episodes)


def expected_diagnostic_outputs() -> frozenset[str]:
    names = {
        "input_diagnostic_config.json",
        "input_course_config.json",
        NOISE_FILENAME,
        "initial_deterministic_parity.json",
        "final_deterministic_parity.json",
    }
    for episode in diagnostic_episode_schedule():
        if episode.noise_seed is not None:
            names.add(f"{episode.label}_sampling.npz")
        names.update(
            {
                f"{episode.label}_frames.jsonl",
                f"{episode.label}_trajectory.npz",
                f"{episode.label}_evaluation.json",
            }
        )
    return frozenset(names)


def _safe_output_name(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or "\x00" in value
        or Path(value).name != value
        or value in {".", "..", RUN_MANIFEST_FILENAME, *_SUPERVISOR_FILES}
    ):
        raise ValueError("Study020 output filename is unsafe or reserved")
    return value


def _finite(value: object, *, field: str, minimum: float | None = None) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{field} must be finite numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{field} must be finite numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite numeric")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} is below its minimum")
    return result


def _validate_summary(value: object, *, label: str) -> dict[str, Any]:
    fields = {
        "objective_evaluation",
        "training_reward_sum_not_success_metric",
        "reset",
        "steps",
        "residual_rms",
    }
    summary = _object(value, fields, field=f"episode {label} summary")
    steps = summary["steps"]
    objective = summary["objective_evaluation"]
    if (
        type(steps) is not int
        or not 1 <= steps <= HORIZON_STEPS
        or type(summary["reset"]) is not dict
        or type(objective) is not dict
        or objective.get("frame_count") != steps
    ):
        raise ValueError(f"episode {label} summary is malformed")
    _finite(
        summary["training_reward_sum_not_success_metric"],
        field=f"episode {label} reward",
    )
    _finite(summary["residual_rms"], field=f"episode {label} residual RMS", minimum=0.0)
    return summary


def _validate_policy_admission(
    value: object, config: SavedPolicyDiagnosticConfig
) -> dict[str, Any]:
    admissions = _object(value, {"initial", "final"}, field="policy admission")
    fields = {
        "archive",
        "state_dict_key_count",
        "state_dict_schema_sha256",
        "log_std_sha256",
        "normalizer_state_sha256",
        "strict_state_dict_loaded",
        "state_dict_readback_exact",
    }
    for policy in ("initial", "final"):
        item = _object(admissions[policy], fields, field=f"{policy} policy admission")
        if (
            item["archive"] != config.policies[policy].receipt()
            or item["state_dict_key_count"] != POLICY_STATE_DICT_KEY_COUNT
            or item["state_dict_schema_sha256"] != POLICY_STATE_SCHEMA_SHA256
            or item["log_std_sha256"] != POLICY_LOG_STD_SHA256[policy]
            or item["normalizer_state_sha256"] != FIXED_NORMALIZER_STATE_SHA256
            or item["strict_state_dict_loaded"] is not True
            or item["state_dict_readback_exact"] is not True
        ):
            raise ValueError(f"{policy} policy admission differs")
    return admissions


def _read_output_json(path: Path, expected_sha256: str, *, field: str) -> dict[str, Any]:
    binding = _binding(path, expected_sha256, maximum=_MAX_MANIFEST_BYTES, field=field)
    return _json(binding, field=field)


def _validate_parity(
    *,
    manifest_value: object,
    outputs: Mapping[str, object],
    output_root: Path,
) -> dict[str, Any]:
    parity = _object(
        manifest_value,
        {"sampled_started_after_both_receipts", "receipts"},
        field="deterministic parity",
    )
    if parity["sampled_started_after_both_receipts"] is not True:
        raise ValueError("sampled execution was not guarded by both parity receipts")
    receipts = _object(parity["receipts"], {"initial", "final"}, field="parity receipts")
    for policy in ("initial", "final"):
        name = f"{policy}_deterministic_parity.json"
        ledger_sha256 = _sha256(outputs[name], field=f"output {name}")
        binding = _object(
            receipts[policy], {"path", "sha256", "size"}, field=f"{policy} parity binding"
        )
        path = _direct_path(output_root / name, directory=False, field=f"{policy} parity")
        if (
            binding.get("path") != name
            or binding.get("sha256") != ledger_sha256
            or type(binding.get("size")) is not int
            or binding["size"] != path.stat().st_size
        ):
            raise ValueError(f"{policy} parity binding differs")
        receipt = _read_output_json(path, ledger_sha256, field=f"{policy} parity receipt")
        expected_hashes = RETAINED_DETERMINISTIC_OUTPUTS[policy]
        expected = {
            "schema_version",
            "artifact",
            "policy",
            "label",
            "expected",
            "observed",
            "passed",
        }
        if (
            set(receipt) != expected
            or receipt.get("schema_version") != 1
            or receipt.get("artifact") != PARITY_ARTIFACT
            or receipt.get("policy") != policy
            or receipt.get("label") != f"{policy}_deterministic"
            or receipt.get("expected") != expected_hashes
            or receipt.get("observed") != expected_hashes
            or receipt.get("passed") is not True
        ):
            raise ValueError(f"{policy} deterministic parity failed")
        for suffix, digest in expected_hashes.items():
            output_name = f"{policy}_deterministic_{suffix}"
            extension = (
                ".jsonl" if suffix == "frames" else ".npz" if suffix == "trajectory" else ".json"
            )
            if outputs.get(f"{output_name}{extension}") != digest:
                raise ValueError(f"{policy} deterministic output differs from retained Study018")
    return parity


def _validate_episodes(
    value: object, *, outputs: Mapping[str, object], output_root: Path
) -> list[dict[str, Any]]:
    schedule = diagnostic_episode_schedule()
    if type(value) is not list or len(value) != len(schedule):
        raise ValueError("Study020 episode schedule is incomplete")
    fields = {"sequence_index", "kind", "label", "policy", "noise_seed", "summary", "sampling"}
    sampling_fields = {
        "noise_seed",
        "noise_rows_available",
        "noise_rows_consumed",
        "action_dimension",
        "clipped_component_count",
        "total_component_count",
        "max_abs_unclipped_action",
    }
    checked: list[dict[str, Any]] = []
    for raw, expected in zip(value, schedule, strict=True):
        row = _object(raw, fields, field=f"episode {expected.sequence_index}")
        if any(
            row[name] != getattr(expected, name)
            for name in ("sequence_index", "kind", "label", "policy", "noise_seed")
        ):
            raise ValueError("Study020 episode order differs")
        summary = _validate_summary(row["summary"], label=expected.label)
        report_name = f"{expected.label}_evaluation.json"
        report_sha256 = _sha256(outputs[report_name], field=f"output {report_name}")
        if (
            _read_output_json(
                output_root / report_name,
                report_sha256,
                field=f"{expected.label} evaluation",
            )
            != summary
        ):
            raise ValueError(f"{expected.label} manifest summary differs from its output")
        if expected.kind == "deterministic_parity":
            if row["sampling"] is not None:
                raise ValueError("deterministic parity episode reports sampled noise")
        else:
            sampling = _object(row["sampling"], sampling_fields, field=f"{expected.label} sampling")
            steps = summary["steps"]
            total = steps * ACTION_DIMENSION
            clipped = sampling["clipped_component_count"]
            if (
                sampling["noise_seed"] != expected.noise_seed
                or sampling["noise_rows_available"] != HORIZON_STEPS
                or sampling["noise_rows_consumed"] != steps
                or sampling["action_dimension"] != ACTION_DIMENSION
                or sampling["total_component_count"] != total
                or type(clipped) is not int
                or not 0 <= clipped <= total
            ):
                raise ValueError(f"{expected.label} sampling counters differ")
            _finite(
                sampling["max_abs_unclipped_action"],
                field=f"{expected.label} maximum unclipped action",
                minimum=0.0,
            )
        checked.append(row)
    return checked


def verify_saved_policy_diagnostic_outputs(
    config: SavedPolicyDiagnosticConfig, output_root: Path
) -> dict[str, Any]:
    """Re-admit the exact Study020 ledger and pre-sampling parity prerequisites."""

    output = _direct_path(output_root, directory=True, field="Study020 output")
    manifest_path = _direct_path(
        output / RUN_MANIFEST_FILENAME, directory=False, field="Study020 run manifest"
    )
    manifest_raw, manifest_encoded = read_json_object(manifest_path)
    manifest = _object(
        manifest_raw,
        {
            "schema_version",
            "artifact",
            "status",
            "input_config_sha256",
            "inputs",
            "outputs",
            "policy_admission",
            "deterministic_parity",
            "episodes",
            "runtime",
            "claims",
        },
        field="Study020 run manifest",
    )
    if (
        manifest["schema_version"] != 1
        or manifest["artifact"] != RUN_ARTIFACT
        or manifest["status"] != "completed"
        or manifest["input_config_sha256"] != config.sha256
        or manifest["inputs"] != config.resource_binding()
        or manifest["runtime"] != RUN_RUNTIME
        or manifest["claims"] != RUN_CLAIMS
    ):
        raise ValueError("Study020 run manifest identity or claim boundary differs")
    outputs = _object(
        manifest["outputs"], set(expected_diagnostic_outputs()), field="Study020 output ledger"
    )
    expected_names = {
        RUN_MANIFEST_FILENAME,
        *_SUPERVISOR_FILES,
        *expected_diagnostic_outputs(),
    }
    observed_names = {path.name for path in output.iterdir()}
    if observed_names not in {
        frozenset(expected_names),
        frozenset({*expected_names, _OPTIONAL_RESOURCE_RECEIPT}),
    }:
        raise ValueError("Study020 output file set has unknown or missing files")

    artifacts: dict[str, dict[str, object]] = {}
    total_size = 0
    for raw_name, raw_sha256 in outputs.items():
        name = _safe_output_name(raw_name)
        digest = _sha256(raw_sha256, field=f"output {name}")
        path = _direct_path(output / name, directory=False, field=f"output {name}")
        size = path.stat().st_size
        if size > _MAX_OUTPUT_MEMBER_BYTES or sha256_file(path) != digest:
            raise ValueError(f"Study020 output bytes differ: {name}")
        total_size += size
        artifacts[name] = {"path": name, "sha256": digest, "size": size}
    if total_size > 1024**3:
        raise ValueError("Study020 output ledger exceeds its byte bound")

    exact_copies = {
        "input_diagnostic_config.json": (config.encoded, config.sha256),
        "input_course_config.json": (
            read_verified_bytes(
                config.course_config.path,
                config.course_config.sha256,
                expected_size=config.course_config.size,
                maximum_size=_MAX_CONFIG_BYTES,
            ),
            config.course_config.sha256,
        ),
        NOISE_FILENAME: (
            read_verified_bytes(
                config.noise.path,
                config.noise.sha256,
                expected_size=config.noise.size,
                maximum_size=_MAX_NOISE_BYTES,
            ),
            config.noise.sha256,
        ),
    }
    for name, (expected_bytes, expected_sha256) in exact_copies.items():
        if outputs[name] != expected_sha256 or (output / name).read_bytes() != expected_bytes:
            raise ValueError(f"Study020 retained input copy differs: {name}")

    admissions = _validate_policy_admission(manifest["policy_admission"], config)
    parity = _validate_parity(
        manifest_value=manifest["deterministic_parity"],
        outputs=outputs,
        output_root=output,
    )
    episodes = _validate_episodes(manifest["episodes"], outputs=outputs, output_root=output)
    return {
        "manifest": {
            "path": RUN_MANIFEST_FILENAME,
            "sha256": hashlib.sha256(manifest_encoded).hexdigest(),
            "size": len(manifest_encoded),
        },
        "outputs": artifacts,
        "policy_admission": admissions,
        "deterministic_parity": parity,
        "episodes": episodes,
    }


__all__ = [
    "ACTION_DIMENSION",
    "CONFIG_ARTIFACT",
    "FIRST_POLICY_BY_SEED",
    "HORIZON_STEPS",
    "NOISE_FILENAME",
    "NOISE_GENERATOR_ID",
    "NOISE_SEEDS",
    "PARITY_ARTIFACT",
    "POLICY_LOG_STD_SHA256",
    "POLICY_STATE_DICT_KEY_COUNT",
    "POLICY_STATE_SCHEMA_SHA256",
    "RESET_SEED",
    "RETAINED_DETERMINISTIC_OUTPUTS",
    "RETAINED_FILENAMES",
    "RETAINED_HASHES",
    "RETAINED_SOURCE_COMMIT",
    "RUN_ARTIFACT",
    "RUN_CLAIMS",
    "RUN_MANIFEST_FILENAME",
    "RUN_RUNTIME",
    "DiagnosticEpisode",
    "FileBinding",
    "SavedPolicyDiagnosticConfig",
    "diagnostic_config_payload",
    "diagnostic_episode_schedule",
    "expected_diagnostic_outputs",
    "load_noise_archive",
    "load_saved_policy_diagnostic_config",
    "verify_saved_policy_diagnostic_outputs",
]
