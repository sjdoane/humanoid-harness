"""Pinned metadata-only registration for the Farama Humanoid TQC artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from oracle_composition.experiments.fixed_reference import ExperimentContractError

REGISTRATION_ID = "farama_minari_humanoid_v5_tqc_expert/v1"
SOURCE_REPOSITORY = "farama-minari/Humanoid-v5-TQC-expert"
SOURCE_COMMIT = "e5a86ffdb70e6f4750f39c0464ac026a8437001a"
FARAMA_SCRIPT_COMMIT = "e74f9d0524c6df014c5a9985d0804001b9ce40dc"
HF_API_SHA256 = "69ccb043f76918fd601c71420ec6511a301193dd768aab5847a4beba3ec652f6"
HF_API_BYTE_COUNT = 3_670
REGISTRATION_RECEIPT_SHA256 = "5ba0845e8b0cd9b6f39c956ddc46d0f46e8e08690d8bdf832941a1f914a8e0cf"
REGISTRATION_RECEIPT_BYTE_COUNT = 9_395
REGISTRATION_RECEIPT_LOGICAL_PATH = (
    "research/source_controllers/farama_minari_humanoid_v5_tqc_expert/RECEIPT.json"
)
MAX_REGISTRATION_BYTES = 64 * 1024

REMOTE_SIBLINGS = (
    (
        ".gitattributes",
        1_566,
        "289d0bf6e5324ded5971f0d32c096a068f7543d8",
        None,
        None,
    ),
    ("README.md", 775, "7cf665dd3bafb1d3225008b9b321bd786303286e", None, None),
    ("config.json", 68_117, "ff415ceb2f4ed5b9d6511b51cfd96a575437b538", None, None),
    (
        "humanoid-v5-TQC-expert.zip",
        7_377_061,
        "847326d03f5e227cb11d3f9af18eb0e14542c519",
        "c0675e01b4efd26d9c773de3e9b4defb40301b5fbdac9f0f31517156fac59fe3",
        132,
    ),
    (
        "humanoid-v5-TQC-expert/_stable_baselines3_version",
        5,
        "58073ef8d7f6ba5b761aca4a751a1c73d4508724",
        None,
        None,
    ),
    (
        "humanoid-v5-TQC-expert/actor.optimizer.pth",
        1_317_966,
        "f3b48fffdddd495697b0aaff472643204d5c52c2",
        "504e3ad3acdfdba8365e315b71457f50362caf47d39a01181c94d63eaa855f56",
        132,
    ),
    (
        "humanoid-v5-TQC-expert/critic.optimizer.pth",
        2_664_618,
        "5dd921c61fc20e4d73d3cd8d7dfc23bdcec1f773",
        "8ca67aab11371a1377e8c6deebe78b6b00cfb9acd041d1e9966345bf2f034298",
        132,
    ),
    (
        "humanoid-v5-TQC-expert/data",
        68_638,
        "075a96a06b48327e595a12549fb7865a6604ac74",
        None,
        None,
    ),
    (
        "humanoid-v5-TQC-expert/ent_coef_optimizer.pth",
        1_940,
        "4053d27e629c4d6236285de1c308a0d600aa1ba8",
        "f27e8b945d8b66edaea7abeb7520d7ad56e296f4aa821d8c03ba90c738c4a824",
        129,
    ),
    (
        "humanoid-v5-TQC-expert/policy.pth",
        3_321_462,
        "eb094d8e48a447fd2323fd130384d97545bd67ee",
        "1e64e56288155087089214548b6a634f332a41955a0d22629efd5ff2240e495c",
        132,
    ),
    (
        "humanoid-v5-TQC-expert/pytorch_variables.pth",
        1_180,
        "982670380df821a7c4e5ca75e3d274a75f6dd1f5",
        "2dd8b074d717a24a49de94053017c5590e5f78f3793d53287372b95bdbbcd5cc",
        129,
    ),
    (
        "humanoid-v5-TQC-expert/system_info.txt",
        248,
        "7d0a35c2255ce1ccd022e39f22eab6401c854d01",
        None,
        None,
    ),
    (
        "replay.mp4",
        991_004,
        "0e7aa8698c8d5f6ba836f93f1e1ff0a0c2be8b27",
        "13ef0e195ea037cc3e3c76bd829b477c60df035658f20046b7c1aea1a0a8e42e",
        131,
    ),
    (
        "results.json",
        167,
        "0407ce78b5b552b81890fc6a973598b14c8ca562",
        None,
        None,
    ),
)

LOCAL_FILES: dict[str, tuple[int, str] | None] = {
    ".gitattributes": None,
    "README.md": (
        775,
        "c843a154c8fd776b5db2e815076758693aa89af84fa4fbbbe230302d45d38081",
    ),
    "config.json": (
        68_117,
        "5093741a50e910c46db1e6b2037800668d83c754cf20d771381c866f381ae4b2",
    ),
    "humanoid-v5-TQC-expert.zip": (
        7_377_061,
        "c0675e01b4efd26d9c773de3e9b4defb40301b5fbdac9f0f31517156fac59fe3",
    ),
    "humanoid-v5-TQC-expert/_stable_baselines3_version": (
        5,
        "4e103ffd9a1e40c6d18d4ccc6d632df4aca6f1cd5b2ab88a343884bf7832ffb6",
    ),
    "humanoid-v5-TQC-expert/actor.optimizer.pth": None,
    "humanoid-v5-TQC-expert/critic.optimizer.pth": None,
    "humanoid-v5-TQC-expert/data": (
        68_638,
        "7466a42ba68e54135ead527b00ee2d23f7c2e1a25138ee9f7ba876a692a29e1c",
    ),
    "humanoid-v5-TQC-expert/ent_coef_optimizer.pth": None,
    "humanoid-v5-TQC-expert/policy.pth": (
        3_321_462,
        "1e64e56288155087089214548b6a634f332a41955a0d22629efd5ff2240e495c",
    ),
    "humanoid-v5-TQC-expert/pytorch_variables.pth": (
        1_180,
        "2dd8b074d717a24a49de94053017c5590e5f78f3793d53287372b95bdbbcd5cc",
    ),
    "humanoid-v5-TQC-expert/system_info.txt": (
        248,
        "a67a9c82f61f8c030e5405d83755427e94f99813195597b41358f7042ced6459",
    ),
    "replay.mp4": None,
    "results.json": (
        167,
        "51d032cb1a80444607ea008bd961e7b3fcd088df8a1f720a74331a003ffc9606",
    ),
}


def expected_registration_receipt() -> dict[str, object]:
    """Return the exact payload-free registration record."""

    remote = []
    for name, size, blob_id, lfs_sha256, pointer_size in REMOTE_SIBLINGS:
        api_record: dict[str, object] = {"blobId": blob_id}
        if lfs_sha256 is None:
            digest = {
                "algorithm": "git-blob-sha1",
                "value": blob_id,
                "api_field": "blobId",
            }
        else:
            api_record["lfs"] = {
                "sha256": lfs_sha256,
                "size": size,
                "pointerSize": pointer_size,
            }
            digest = {
                "algorithm": "lfs.sha256",
                "value": lfs_sha256,
                "api_field": "lfs.sha256",
            }
        remote.append(
            {
                "rfilename": name,
                "size": size,
                "digest": digest,
                "api_record": api_record,
            }
        )
    local = []
    for name, _, _, _, _ in REMOTE_SIBLINGS:
        item = LOCAL_FILES[name]
        if item is None:
            local.append({"rfilename": name, "status": "absent"})
        else:
            size, digest = item
            local.append(
                {
                    "rfilename": name,
                    "status": "present",
                    "byte_count": size,
                    "sha256": digest,
                }
            )
    return {
        "schema_version": 1,
        "registration_id": REGISTRATION_ID,
        "source": {
            "repository": SOURCE_REPOSITORY,
            "repository_commit": SOURCE_COMMIT,
            "farama_script_commit": FARAMA_SCRIPT_COMMIT,
            "description": "SB3 TQC, `20 x 10^6` steps, runs without falling",
            "model_card_metric": {
                "name": "mean_reward",
                "value": "10370.61 +/- 1542.02",
                "episodes": 1_000,
                "deterministic": True,
                "verified": False,
            },
        },
        "captured_api": {
            "path": (
                "artifacts/external/farama-minari-humanoid-v5-tqc-expert/hf_api_model_info.json"
            ),
            "sha256": HF_API_SHA256,
            "byte_count": HF_API_BYTE_COUNT,
        },
        "remote_inventory": remote,
        "local_inventory": local,
        "rights": {
            "hugging_face_license": "unspecified",
            "permitted_project_use": "local development only",
            "technical_import_approval": "Samuel approved 2026-09-04",
            "technical_approval_is_redistribution_grant": False,
            "payload_bytes_enter_git": False,
        },
    }


def _reject_duplicate_key(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExperimentContractError("registration JSON contains a duplicate key")
        result[key] = value
    return result


def _exact_json_match(value: object, expected: object) -> bool:
    try:
        return _exact_json_match_inner(value, expected)
    except (RecursionError, TypeError, ValueError):
        return False


def _exact_json_match_inner(value: object, expected: object) -> bool:
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        if set(value) != set(expected):
            return False
        return all(_exact_json_match_inner(value[key], expected[key]) for key in expected)
    if type(expected) is list:
        return len(value) == len(expected) and all(
            _exact_json_match_inner(observed, wanted)
            for observed, wanted in zip(value, expected, strict=True)
        )
    return value == expected


def validate_registration_receipt(value: object) -> dict[str, object]:
    """Reject any label, sibling, local-presence, or rights drift."""

    expected = expected_registration_receipt()
    if not _exact_json_match(value, expected):
        raise ExperimentContractError("Farama TQC registration receipt differs")
    return expected


def load_registration_receipt(path: Path) -> dict[str, object]:
    """Read and validate one bounded metadata-only registration receipt."""

    try:
        payload = Path(path).read_bytes()
    except OSError as exc:
        raise ExperimentContractError("cannot read Farama TQC registration receipt") from exc
    if not 0 < len(payload) <= MAX_REGISTRATION_BYTES:
        raise ExperimentContractError("Farama TQC registration receipt size is invalid")
    if (
        len(payload) != REGISTRATION_RECEIPT_BYTE_COUNT
        or hashlib.sha256(payload).hexdigest() != REGISTRATION_RECEIPT_SHA256
    ):
        raise ExperimentContractError("Farama TQC registration receipt bytes differ")
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
        raise ExperimentContractError("Farama TQC registration receipt is invalid JSON") from exc
    return validate_registration_receipt(value)


__all__ = [
    "FARAMA_SCRIPT_COMMIT",
    "HF_API_BYTE_COUNT",
    "HF_API_SHA256",
    "LOCAL_FILES",
    "REGISTRATION_ID",
    "REGISTRATION_RECEIPT_BYTE_COUNT",
    "REGISTRATION_RECEIPT_LOGICAL_PATH",
    "REGISTRATION_RECEIPT_SHA256",
    "REMOTE_SIBLINGS",
    "SOURCE_COMMIT",
    "SOURCE_REPOSITORY",
    "expected_registration_receipt",
    "load_registration_receipt",
    "validate_registration_receipt",
]
