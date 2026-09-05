"""Pinned metadata-only registrations for the Farama medium and simple actors."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oracle_composition.experiments.fixed_reference import ExperimentContractError

FARAMA_SCRIPT_COMMIT = "e74f9d0524c6df014c5a9985d0804001b9ce40dc"
MAX_REGISTRATION_BYTES = 64 * 1024
_RIGHTS = {
    "hugging_face_license": "unspecified",
    "permitted_project_use": "local development only",
    "technical_import_approval": "Samuel approved 2026-09-04",
    "technical_approval_is_redistribution_grant": False,
    "payload_bytes_enter_git": False,
}


@dataclass(frozen=True, slots=True)
class SiblingRegistrationSpec:
    variant: str
    registration_id: str
    repository: str
    repository_commit: str
    registration_receipt_logical_path: str
    registration_receipt_sha256: str
    registration_receipt_byte_count: int
    registration_semantic_sha256: str
    captured_api_path: str
    captured_api_sha256: str
    captured_api_byte_count: int
    source_zip_sha256: str
    source_zip_byte_count: int
    policy_sha256: str
    policy_byte_count: int
    metadata_sha256: str
    metadata_byte_count: int
    total_timesteps: int
    num_timesteps: int
    n_updates: int
    mean_reward: str

    @property
    def source_prefix(self) -> str:
        return f"humanoid-v5-TQC-{self.variant}"

    @property
    def description(self) -> str:
        return (
            f"SB3 TQC {self.variant} actor; num_timesteps {self.num_timesteps} "
            f"of {self.total_timesteps}"
        )

    @property
    def remote_filenames(self) -> tuple[str, ...]:
        prefix = self.source_prefix
        return (
            ".gitattributes",
            "README.md",
            "config.json",
            f"{prefix}.zip",
            f"{prefix}/_stable_baselines3_version",
            f"{prefix}/actor.optimizer.pth",
            f"{prefix}/critic.optimizer.pth",
            f"{prefix}/data",
            f"{prefix}/ent_coef_optimizer.pth",
            f"{prefix}/policy.pth",
            f"{prefix}/pytorch_variables.pth",
            f"{prefix}/system_info.txt",
            "replay.mp4",
            "results.json",
        )

    @property
    def absent_filenames(self) -> frozenset[str]:
        prefix = self.source_prefix
        return frozenset(
            {
                ".gitattributes",
                f"{prefix}/actor.optimizer.pth",
                f"{prefix}/critic.optimizer.pth",
                f"{prefix}/ent_coef_optimizer.pth",
                "replay.mp4",
            }
        )


SIBLING_REGISTRATION_SPECS = {
    "medium": SiblingRegistrationSpec(
        variant="medium",
        registration_id="farama_minari_humanoid_v5_tqc_medium/v1",
        repository="farama-minari/Humanoid-v5-TQC-medium",
        repository_commit="949f7963c1a8964587dca73d48873ad021e168b5",
        registration_receipt_logical_path=(
            "research/source_controllers/farama_minari_humanoid_v5_tqc_medium/RECEIPT.json"
        ),
        registration_receipt_sha256=(
            "68344e980e42543ddd5ca3d164a9c8173f63950d47b4c8c9064e57283fd4e0a1"
        ),
        registration_receipt_byte_count=9_401,
        registration_semantic_sha256=(
            "d7a1fcbdf8bd812ba0592512464b53e82866a72e1dd0d8eae1252774e36d4f17"
        ),
        captured_api_path=(
            "artifacts/external/farama-minari-humanoid-v5-tqc-medium/hf_api_model_info.json"
        ),
        captured_api_sha256=("fb6b2075ab149d0bf8a42cb15190f1819f1dd5439696570f3e19d6122244afc6"),
        captured_api_byte_count=3_667,
        source_zip_sha256=("f47ae84f39c61416ddff1ce0a68dba446ea703a354469480515abd8570fa2779"),
        source_zip_byte_count=7_377_044,
        policy_sha256="d54c93dd82d97cd931caf85fcba5b4722b9c86075e9c679576b6ba45b7823c66",
        policy_byte_count=3_321_462,
        metadata_sha256="63194995ab0605aacd68c6f160619c3cc5717233cee1a5e6ff8038a92fa644ad",
        metadata_byte_count=68_621,
        total_timesteps=5_000_000,
        num_timesteps=4_950_000,
        n_updates=989_979,
        mean_reward="8021.95 +/- 912.19",
    ),
    "simple": SiblingRegistrationSpec(
        variant="simple",
        registration_id="farama_minari_humanoid_v5_tqc_simple/v1",
        repository="farama-minari/Humanoid-v5-TQC-simple",
        repository_commit="39e2954c193fc1352535935a7d71eca9e5745d9b",
        registration_receipt_logical_path=(
            "research/source_controllers/farama_minari_humanoid_v5_tqc_simple/RECEIPT.json"
        ),
        registration_receipt_sha256=(
            "745d82dea8fc4878f70ed58f4d6ff4cd506b0f94c85676dc70e3cffce8283ac5"
        ),
        registration_receipt_byte_count=9_401,
        registration_semantic_sha256=(
            "52c4ebc4c31847e356d84c98a87170d09f678c709e87d0434c5763b734e4a8ce"
        ),
        captured_api_path=(
            "artifacts/external/farama-minari-humanoid-v5-tqc-simple/hf_api_model_info.json"
        ),
        captured_api_sha256=("0755829dfa57723a4e4469381a27fea1505d8e49c90ea05d6dbae50aa60d38fd"),
        captured_api_byte_count=3_667,
        source_zip_sha256=("4814a61d04158a9535cec8d8173d4e2cb4f7be2e070b257b9d27215539d65735"),
        source_zip_byte_count=7_375_841,
        policy_sha256="ca9aff0dc359d6011ddde33f196fc8b612781cba80247fb8a9889eb1df74e58e",
        policy_byte_count=3_321_014,
        metadata_sha256="a689c4486127c8828a48ebc420493b36176b41f858cbffdc26517b052cfcbaea",
        metadata_byte_count=68_633,
        total_timesteps=2_000_000,
        num_timesteps=1_965_000,
        n_updates=392_979,
        mean_reward="5543.17 +/- 825.31",
    ),
}


def sibling_registration_spec(variant: str) -> SiblingRegistrationSpec:
    """Return one exact, compiled sibling registration profile."""

    if type(variant) is not str or variant not in SIBLING_REGISTRATION_SPECS:
        raise ExperimentContractError("Farama TQC sibling variant is not allowlisted")
    return SIBLING_REGISTRATION_SPECS[variant]


def _reject_duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExperimentContractError("registration JSON contains a duplicate key")
        result[key] = value
    return result


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ExperimentContractError("registration value is not canonical JSON") from exc


def _hex_digest(value: object, *, width: int, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != width
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"{field} is not a lowercase hexadecimal digest")
    return value


def validate_sibling_registration_receipt(
    value: object,
    *,
    variant: str,
) -> dict[str, object]:
    """Reject any sibling identity, digest-label, presence, or rights drift."""

    spec = sibling_registration_spec(variant)
    required = {
        "schema_version",
        "registration_id",
        "source",
        "captured_api",
        "remote_inventory",
        "local_inventory",
        "rights",
    }
    if type(value) is not dict or set(value) != required:
        raise ExperimentContractError("Farama TQC sibling registration fields differ")
    expected_source = {
        "repository": spec.repository,
        "repository_commit": spec.repository_commit,
        "farama_script_commit": FARAMA_SCRIPT_COMMIT,
        "description": spec.description,
        "model_card_metric": {
            "name": "mean_reward",
            "value": spec.mean_reward,
            "episodes": 1_000,
            "deterministic": True,
            "verified": False,
        },
    }
    expected_api = {
        "path": spec.captured_api_path,
        "sha256": spec.captured_api_sha256,
        "byte_count": spec.captured_api_byte_count,
    }
    if (
        type(value["schema_version"]) is not int
        or value["schema_version"] != 1
        or type(value["registration_id"]) is not str
        or value["registration_id"] != spec.registration_id
        or value["source"] != expected_source
        or value["captured_api"] != expected_api
        or value["rights"] != _RIGHTS
    ):
        raise ExperimentContractError("Farama TQC sibling registration identity differs")

    remote = value["remote_inventory"]
    local = value["local_inventory"]
    if type(remote) is not list or type(local) is not list or len(remote) != 14 or len(local) != 14:
        raise ExperimentContractError("Farama TQC sibling inventory length differs")
    if tuple(item.get("rfilename") for item in remote if type(item) is dict) != (
        spec.remote_filenames
    ) or tuple(item.get("rfilename") for item in local if type(item) is dict) != (
        spec.remote_filenames
    ):
        raise ExperimentContractError("Farama TQC sibling inventory names differ")

    for item in remote:
        if type(item) is not dict or set(item) != {"rfilename", "size", "digest", "api_record"}:
            raise ExperimentContractError("Farama TQC sibling remote inventory differs")
        if type(item["size"]) is not int or item["size"] <= 0:
            raise ExperimentContractError("Farama TQC sibling remote size differs")
        digest = item["digest"]
        api_record = item["api_record"]
        if type(digest) is not dict or type(api_record) is not dict:
            raise ExperimentContractError("Farama TQC sibling digest record differs")
        blob_id = _hex_digest(api_record.get("blobId"), width=40, field="API blobId")
        if "lfs" in api_record:
            lfs = api_record["lfs"]
            if (
                set(api_record) != {"blobId", "lfs"}
                or type(lfs) is not dict
                or set(lfs) != {"sha256", "size", "pointerSize"}
                or type(lfs["size"]) is not int
                or lfs["size"] != item["size"]
                or type(lfs["pointerSize"]) is not int
                or lfs["pointerSize"] <= 0
            ):
                raise ExperimentContractError("Farama TQC sibling LFS record differs")
            lfs_sha256 = _hex_digest(lfs["sha256"], width=64, field="LFS SHA-256")
            expected_digest = {
                "algorithm": "lfs.sha256",
                "value": lfs_sha256,
                "api_field": "lfs.sha256",
            }
        else:
            if set(api_record) != {"blobId"}:
                raise ExperimentContractError("Farama TQC sibling Git record differs")
            expected_digest = {
                "algorithm": "git-blob-sha1",
                "value": blob_id,
                "api_field": "blobId",
            }
        if digest != expected_digest:
            raise ExperimentContractError("Farama TQC sibling digest label differs")

    remote_sizes = {item["rfilename"]: item["size"] for item in remote}
    for item in local:
        if type(item) is not dict or type(item.get("rfilename")) is not str:
            raise ExperimentContractError("Farama TQC sibling local inventory differs")
        name = item["rfilename"]
        if name in spec.absent_filenames:
            if item != {"rfilename": name, "status": "absent"}:
                raise ExperimentContractError("Farama TQC sibling explicit absence differs")
        elif (
            set(item) != {"rfilename", "status", "byte_count", "sha256"}
            or item["status"] != "present"
            or type(item["byte_count"]) is not int
            or item["byte_count"] != remote_sizes[name]
        ):
            raise ExperimentContractError("Farama TQC sibling local presence differs")
        else:
            _hex_digest(item["sha256"], width=64, field="local SHA-256")

    if hashlib.sha256(_canonical_json(value)).hexdigest() != spec.registration_semantic_sha256:
        raise ExperimentContractError("Farama TQC sibling registration receipt differs")
    return value


def load_sibling_registration_receipt(
    path: Path,
    *,
    variant: str,
) -> dict[str, object]:
    """Read one exact metadata-only sibling registration receipt."""

    spec = sibling_registration_spec(variant)
    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise ExperimentContractError("cannot read Farama TQC sibling registration") from exc
    if (
        not 0 < len(payload) <= MAX_REGISTRATION_BYTES
        or len(payload) != spec.registration_receipt_byte_count
        or hashlib.sha256(payload).hexdigest() != spec.registration_receipt_sha256
    ):
        raise ExperimentContractError("Farama TQC sibling registration bytes differ")
    try:
        value = json.loads(
            payload.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_key,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ExperimentContractError("registration JSON contains a non-finite constant")
            ),
        )
    except ExperimentContractError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ExperimentContractError("Farama TQC sibling registration is invalid JSON") from exc
    return validate_sibling_registration_receipt(value, variant=variant)


__all__ = [
    "FARAMA_SCRIPT_COMMIT",
    "SIBLING_REGISTRATION_SPECS",
    "SiblingRegistrationSpec",
    "load_sibling_registration_receipt",
    "sibling_registration_spec",
    "validate_sibling_registration_receipt",
]
