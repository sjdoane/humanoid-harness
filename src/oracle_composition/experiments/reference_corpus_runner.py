"""One-shot generator for the frozen reference corpus and development screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from oracle_composition.contracts.reference_identity_v2 import (
    CORPUS_ACTORS,
    CORPUS_SEEDS,
    DEVELOPMENT_SCREEN_SEEDS,
    canonical_json_bytes,
    corpus_manifest_payload,
    sha256_file,
    sha256_json,
    validate_corpus_manifest,
    validate_e3_manifest,
    validate_full_clip_certificate,
)
from oracle_composition.envs.reference_corpus import make_reference_corpus_env
from oracle_composition.experiments.tqc_actor_npz import actor_schema_sha256

from .artifact_io import publish_bytes_without_overwrite
from .e3_fork_certifier import E3Branch, certify_e3_block, summarize_e3_block_results
from .expert_development_screen import (
    DevelopmentScreenClip,
    evaluate_public_expert_development_screen,
)
from .reference_corpus_bundle import (
    PublishedClipBundle,
    load_bundle_manifest,
    publish_clip_bundle,
)
from .reference_corpus_collector import collect_reference_clip
from .reference_corpus_contract import decode_clip_payload
from .runtime_identity import dependency_lock_path

RUNNER_ID = "reference_corpus_v1_one_shot_runner/v1"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "reference_corpus_v1"
DEFAULT_SCREEN_RECEIPT = (
    PROJECT_ROOT
    / "artifacts"
    / "bootstrap_tqc_humanoid"
    / "public_expert_development_screen_v1.json"
)
DEFAULT_E3_MANIFEST = PROJECT_ROOT / "experiments" / "reference_corpus_v1" / "e3_manifest_v1.json"
E1_RECEIPT = (
    PROJECT_ROOT
    / "artifacts"
    / "bootstrap_tqc_humanoid"
    / "e1_initialization_identity_external_v1.json"
)
E1_RECEIPT_SHA256 = "28725ecfba2ca89f4b4608e5e8d5b5024cfe2ed8df71383bc03a248891edf266"

ACTOR_REGISTRY = {
    "expert": {
        "npz": "artifacts/bootstrap_tqc_humanoid/farama_minari_humanoid_v5_tqc_actor_v1.npz",
        "npz_sha256": "60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b",
        "actor_state_sha256": "3fd39cc715a10126fd92b20f6ce213c380eb4d5df843a42315aac50cf116748a",
        "source_policy": "artifacts/external/farama-minari-humanoid-v5-tqc-expert/humanoid-v5-TQC-expert/policy.pth",
        "source_policy_sha256": "1e64e56288155087089214548b6a634f332a41955a0d22629efd5ff2240e495c",
        "import_receipt": "artifacts/bootstrap_tqc_humanoid/external_actor_import_expert_03a3_v1.json",
        "import_receipt_sha256": "b790f06ccb66ca45809eaa1b0c5cd6804e072c6fd9eb48c61838ee2e558bd9d7",
        "equivalence_receipt": "artifacts/bootstrap_tqc_humanoid/external_actor_equivalence_expert_03a3_v1.json",
        "equivalence_receipt_sha256": "65b2090b783e381e799e90e372308018d4e9a50fa6be2df7934af5f24e78a23e",
    },
    "medium": {
        "npz": "artifacts/bootstrap_tqc_humanoid/farama_minari_humanoid_v5_tqc_medium_actor_v1.npz",
        "npz_sha256": "2677ebb70cd20e0ba8f8a591814fc853e325277db65184ebafb5bf5e21198b04",
        "actor_state_sha256": "cffb5679e3ef10cc99980819013941cb55cdc9f2198ba4a91a306dfc5f73e00c",
        "source_policy": "artifacts/external/farama-minari-humanoid-v5-tqc-medium/humanoid-v5-TQC-medium/policy.pth",
        "source_policy_sha256": "d54c93dd82d97cd931caf85fcba5b4722b9c86075e9c679576b6ba45b7823c66",
        "import_receipt": "artifacts/bootstrap_tqc_humanoid/external_actor_import_medium_v1.json",
        "import_receipt_sha256": "b1c07c2f5070b48ff6bb28983b16ed16d1616e7ad18f557639dad89bb1f4f172",
        "equivalence_receipt": "artifacts/bootstrap_tqc_humanoid/external_actor_equivalence_medium_v1.json",
        "equivalence_receipt_sha256": "ede73be8db0ec3411dfb1a5ce2dbbe109d49432d1aeba97e6e4b9bf4fb44d487",
    },
    "simple": {
        "npz": "artifacts/bootstrap_tqc_humanoid/farama_minari_humanoid_v5_tqc_simple_actor_v1.npz",
        "npz_sha256": "b09aa921316640024e9703671d328d7ecbabe76f04cfed8c1d8fb86fbd95a917",
        "actor_state_sha256": "3068049a0d18b7924d117a96aafee7127e411a70eba344ff985d31e97da3312b",
        "source_policy": "artifacts/external/farama-minari-humanoid-v5-tqc-simple/humanoid-v5-TQC-simple/policy.pth",
        "source_policy_sha256": "ca9aff0dc359d6011ddde33f196fc8b612781cba80247fb8a9889eb1df74e58e",
        "import_receipt": "artifacts/bootstrap_tqc_humanoid/external_actor_import_simple_v1.json",
        "import_receipt_sha256": "bef2a2678d30f13a9e3350bf8c8f591661bb129b063276a15b8051f94ee313f7",
        "equivalence_receipt": "artifacts/bootstrap_tqc_humanoid/external_actor_equivalence_simple_v1.json",
        "equivalence_receipt_sha256": "604db637d316d3b8e46fb6f5d7a9b9004cfced267836394d89df183c87615268",
    },
}


def _publish_json(path: Path, value: object) -> tuple[Path, str]:
    encoded = canonical_json_bytes(value)
    published = publish_bytes_without_overwrite(path, encoded)
    return path, published.sha256


def _append_attempt(ledger: Path, value: object) -> None:
    encoded = canonical_json_bytes(value) + b"\n"
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(ledger, flags, 0o600)
    try:
        offset = 0
        while offset < len(encoded):
            written = os.write(descriptor, encoded[offset:])
            if written <= 0:
                raise OSError("attempt-ledger write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _load_json(
    path: Path,
    *,
    expected_sha256: str | None = None,
    require_terminating_lf: bool = False,
) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"JSON artifact is missing: {path}")
    encoded = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise ValueError(f"JSON artifact hash differs: {path}")
    try:
        value = json.loads(encoded.decode("utf-8", errors="strict"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"JSON artifact is invalid: {path}") from exc
    expected_encoding = canonical_json_bytes(value)
    if require_terminating_lf:
        expected_encoding += b"\n"
    if type(value) is not dict or expected_encoding != encoded:
        raise ValueError(f"JSON artifact is not canonical: {path}")
    return value


def _actor_identity(variant: str) -> dict[str, object]:
    record = ACTOR_REGISTRY[variant]
    checks = {
        "npz": record["npz_sha256"],
        "source_policy": record["source_policy_sha256"],
        "import_receipt": record["import_receipt_sha256"],
        "equivalence_receipt": record["equivalence_receipt_sha256"],
    }
    for path_key, digest in checks.items():
        path = PROJECT_ROOT / record[path_key]
        if sha256_file(path) != digest:
            raise ValueError(f"precondition hash differs for {variant} {path_key}")
    import_receipt = _load_json(
        PROJECT_ROOT / record["import_receipt"],
        expected_sha256=record["import_receipt_sha256"],
        require_terminating_lf=True,
    )
    equivalence = _load_json(
        PROJECT_ROOT / record["equivalence_receipt"],
        expected_sha256=record["equivalence_receipt_sha256"],
        require_terminating_lf=True,
    )
    if (
        import_receipt["strict_actor_npz"]["sha256"] != record["npz_sha256"]
        or import_receipt["strict_actor_npz"]["actor_state_sha256"] != record["actor_state_sha256"]
        or equivalence["strict_actor_npz"]["sha256"] != record["npz_sha256"]
        or equivalence["actor_state"]["strict_npz_sha256"] != record["actor_state_sha256"]
        or equivalence["source_variant"] != variant
        or equivalence["passed"] is not True
    ):
        raise ValueError(f"actor receipt chain differs for {variant}")
    return {
        "variant": variant,
        "npz_sha256": record["npz_sha256"],
        "actor_state_sha256": record["actor_state_sha256"],
        "actor_schema_sha256": actor_schema_sha256(),
        "source_policy_sha256": record["source_policy_sha256"],
        "import_receipt_sha256": record["import_receipt_sha256"],
        "equivalence_receipt_sha256": record["equivalence_receipt_sha256"],
        "inference_id": "strict_npz_tqc_deterministic_mean_cpu_float32/v1",
    }


def _model_path() -> Path:
    environment = make_reference_corpus_env()
    try:
        path = Path(str(environment.unwrapped.fullpath))
        if path.is_symlink() or not path.is_file():
            raise ValueError("Humanoid model path is unavailable")
        return path
    finally:
        environment.close()


def _artifact_sources(variant: str, *, model_path: Path) -> dict[str, Path]:
    record = ACTOR_REGISTRY[variant]
    experiment_directory = Path(__file__).resolve().parent
    return {
        "source_policy_bytes": PROJECT_ROOT / record["source_policy"],
        "strict_actor_npz": PROJECT_ROOT / record["npz"],
        "import_receipt": PROJECT_ROOT / record["import_receipt"],
        "equivalence_receipt": PROJECT_ROOT / record["equivalence_receipt"],
        "mujoco_model_bytes": model_path,
        "dependency_lock": dependency_lock_path(),
        "collector_source": experiment_directory / "reference_corpus_collector.py",
        "certifier_source": experiment_directory / "reference_corpus_certifier.py",
        "metric_source": experiment_directory / "tqc_development_metrics.py",
        "e3_metric_source": experiment_directory / "e3_fork_certifier.py",
        "e3_manifest_source": DEFAULT_E3_MANIFEST,
        "screen_evaluator_source": experiment_directory / "expert_development_screen.py",
        "bundle_source": experiment_directory / "reference_corpus_bundle.py",
        "contract_source": experiment_directory / "reference_corpus_contract.py",
        "runner_source": Path(__file__),
        "environment_source": PROJECT_ROOT / "src/oracle_composition/envs/reference_corpus.py",
        "actor_runtime_source": PROJECT_ROOT
        / "src/oracle_composition/sources/strict_tqc_actor_runtime.py",
        "identity_contract_source": PROJECT_ROOT
        / "src/oracle_composition/contracts/reference_identity_v2.py",
        "e1_initialization_receipt": E1_RECEIPT,
    }


def _load_arrays(root: Path, manifest: dict[str, object]) -> dict[str, Any]:
    core = manifest["core"]
    payload_path = root / core["payload"]["object_path"]
    payload = payload_path.read_bytes()
    return decode_clip_payload(
        payload,
        steps=core["steps"],
        screen_canary=core["clip_kind"] == "development_screen",
    )


def _artifact_hash_by_role(manifest: dict[str, object], role: str) -> str:
    matches = [
        item
        for item in manifest["core"]["bound_artifacts"]
        if type(item) is dict and item.get("role") == role
    ]
    if len(matches) != 1 or type(matches[0].get("sha256")) is not str:
        raise RuntimeError(f"bundle source role {role!r} is missing or duplicated")
    return matches[0]["sha256"]


def _run_certifier(
    root: Path,
    bundles: list[PublishedClipBundle],
) -> dict[str, object]:
    request = {
        "request_id": "reference_corpus_tier_d_certifier_request/v1",
        "artifact_root": root.as_posix(),
        "bundles": [
            {
                "manifest_path": bundle.manifest_path.as_posix(),
                "manifest_sha256": bundle.manifest_sha256,
            }
            for bundle in bundles
        ],
    }
    request_path, _request_sha256 = _publish_json(
        root / "tier_d_certifier_request_v1.json", request
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "oracle_composition.experiments.reference_corpus_certifier",
            "--request",
            str(request_path),
        ],
        cwd=PROJECT_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=None,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"separate Tier-D certifier failed with exit {result.returncode}")
    try:
        payload = json.loads(result.stdout)
    except ValueError as exc:
        raise RuntimeError("separate Tier-D certifier returned invalid JSON") from exc
    if (
        type(payload) is not dict
        or payload.get("certificate_count") != len(bundles)
        or len(payload.get("certificates", [])) != len(bundles)
    ):
        raise RuntimeError("separate Tier-D certifier omitted a clip")
    return payload


def generate_reference_corpus(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    screen_receipt_path: Path = DEFAULT_SCREEN_RECEIPT,
) -> dict[str, object]:
    """Execute the predeclared no-retry collection and certification once."""

    root = Path(artifact_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any((root / name).exists() for name in ("clips", "objects", "corpus_index_v1.json")):
        raise RuntimeError("reference corpus output already exists; no retry is permitted")
    ledger = root / "attempt_ledger.jsonl"
    if ledger.exists():
        raise RuntimeError("reference corpus attempt ledger already exists; no retry is permitted")
    e3_design = _load_json(DEFAULT_E3_MANIFEST)
    validate_e3_manifest(e3_design)
    if sha256_file(E1_RECEIPT) != E1_RECEIPT_SHA256:
        raise ValueError("actual E1 initialization receipt hash differs")
    identities = {variant: _actor_identity(variant) for variant in CORPUS_ACTORS}
    model_path = _model_path()
    bundles: list[PublishedClipBundle] = []
    entries: list[dict[str, object]] = []
    reset_order = 0
    try:
        for seed in CORPUS_SEEDS:
            for variant in CORPUS_ACTORS:
                clip_id = f"corpus-{seed}-{variant}"
                _append_attempt(
                    ledger,
                    {
                        "event": "started",
                        "clip_id": clip_id,
                        "seed": seed,
                        "actor_variant": variant,
                        "reset_order": reset_order,
                        "retry": False,
                    },
                )
                clip = collect_reference_clip(
                    clip_id=clip_id,
                    clip_kind="corpus",
                    actor_variant=variant,
                    actor_npz_path=PROJECT_ROOT / ACTOR_REGISTRY[variant]["npz"],
                    actor_identity=identities[variant],
                    seed=seed,
                    reset_order=reset_order,
                )
                bundle = publish_clip_bundle(
                    clip,
                    artifact_root=root,
                    artifact_sources=_artifact_sources(variant, model_path=model_path),
                )
                bundles.append(bundle)
                entries.append(
                    {
                        "clip_id": clip_id,
                        "seed": seed,
                        "actor_variant": variant,
                        "reset_order": reset_order,
                        "bundle_manifest_sha256": bundle.manifest_sha256,
                        "reference_identity_sha256": bundle.reference_identity_sha256,
                    }
                )
                _append_attempt(
                    ledger,
                    {
                        "event": "collected",
                        "clip_id": clip_id,
                        "bundle_manifest_sha256": bundle.manifest_sha256,
                        "payload_sha256": bundle.payload_sha256,
                    },
                )
                reset_order += 1
                print(f"collected {reset_order}/128 {clip_id}", flush=True)
        corpus_manifest = corpus_manifest_payload(entries)
        validate_corpus_manifest(corpus_manifest)
        corpus_manifest_path, corpus_manifest_sha256 = _publish_json(
            root / "corpus_manifest_v1.json", corpus_manifest
        )
        screen_bundles: list[PublishedClipBundle] = []
        for screen_index, seed in enumerate(DEVELOPMENT_SCREEN_SEEDS, start=1):
            clip_id = f"screen-{seed}-expert"
            _append_attempt(
                ledger,
                {
                    "event": "started",
                    "clip_id": clip_id,
                    "seed": seed,
                    "actor_variant": "expert",
                    "reset_order": reset_order,
                    "retry": False,
                },
            )
            clip = collect_reference_clip(
                clip_id=clip_id,
                clip_kind="development_screen",
                actor_variant="expert",
                actor_npz_path=PROJECT_ROOT / ACTOR_REGISTRY["expert"]["npz"],
                actor_identity=identities["expert"],
                seed=seed,
                reset_order=reset_order,
                compare_plain_rewards=True,
            )
            bundle = publish_clip_bundle(
                clip,
                artifact_root=root,
                artifact_sources=_artifact_sources("expert", model_path=model_path),
            )
            bundles.append(bundle)
            screen_bundles.append(bundle)
            _append_attempt(
                ledger,
                {
                    "event": "collected",
                    "clip_id": clip_id,
                    "bundle_manifest_sha256": bundle.manifest_sha256,
                    "payload_sha256": bundle.payload_sha256,
                },
            )
            reset_order += 1
            print(f"collected {108 + screen_index}/128 {clip_id}", flush=True)
        certifier_result = _run_certifier(root, bundles)
        certificates = {item["clip_id"]: item for item in certifier_result["certificates"]}
        if len(certificates) != 128:
            raise RuntimeError("Tier-D certificate identities are duplicated")
        certificate_records: dict[str, dict[str, object]] = {}
        for bundle in bundles:
            item = certificates[bundle.clip_id]
            certificate_path = Path(item["certificate_path"])
            certificate = _load_json(
                certificate_path,
                expected_sha256=item["certificate_sha256"],
            )
            validate_full_clip_certificate(certificate)
            certificate_records[bundle.clip_id] = certificate
        aggregate = {
            "certificate_id": "humanoid_reference_corpus_tier_d_aggregate/v1",
            "schema_version": 1,
            "corpus_manifest_sha256": corpus_manifest_sha256,
            "corpus_clip_count": 108,
            "development_screen_clip_count": 20,
            "certificate_count": 128,
            "passed_certificate_count": 128,
            "failed_certificate_count": 0,
            "certifier_pid": certifier_result["certifier_pid"],
            "all_transitions_rule": True,
            "certificates": [
                {
                    "clip_id": bundle.clip_id,
                    "bundle_manifest_sha256": bundle.manifest_sha256,
                    "reference_identity_sha256": bundle.reference_identity_sha256,
                    "tier_d_certificate_sha256": certificates[bundle.clip_id]["certificate_sha256"],
                }
                for bundle in bundles
            ],
            "claim_ceiling": "named_same_runtime_full_clip_replay_only",
        }
        aggregate["aggregate_result_sha256"] = sha256_json(aggregate)
        aggregate_path, aggregate_sha256 = _publish_json(
            root / "reference_corpus_tier_d_aggregate_v1.json", aggregate
        )
        bundle_by_id = {bundle.clip_id: bundle for bundle in bundles}
        e3_block_results = []
        for seed in CORPUS_SEEDS:
            branches = []
            for variant in CORPUS_ACTORS:
                clip_id = f"corpus-{seed}-{variant}"
                bundle = bundle_by_id[clip_id]
                manifest = load_bundle_manifest(
                    bundle.manifest_path,
                    expected_sha256=bundle.manifest_sha256,
                )
                certificate_item = certificates[clip_id]
                branches.append(
                    E3Branch(
                        actor_variant=variant,
                        manifest_sha256=bundle.manifest_sha256,
                        manifest=manifest,
                        certificate_sha256=certificate_item["certificate_sha256"],
                        certificate=certificate_records[clip_id],
                        arrays=_load_arrays(root, manifest),
                    )
                )
            e3_block_results.append(certify_e3_block(seed, tuple(branches)))
        e3_result = summarize_e3_block_results(tuple(e3_block_results), manifest=e3_design)
        e3_path, e3_sha256 = _publish_json(root / "e3_fork_certificate_v1.json", e3_result)
        screen_clips = []
        for bundle in screen_bundles:
            manifest = load_bundle_manifest(
                bundle.manifest_path,
                expected_sha256=bundle.manifest_sha256,
            )
            certificate_item = certificates[bundle.clip_id]
            screen_clips.append(
                DevelopmentScreenClip(
                    seed=manifest["core"]["seed"],
                    bundle_manifest_sha256=bundle.manifest_sha256,
                    reference_identity_sha256=bundle.reference_identity_sha256,
                    certificate_sha256=certificate_item["certificate_sha256"],
                    certificate=certificate_records[bundle.clip_id],
                    arrays=_load_arrays(root, manifest),
                    reward_canary=manifest["core"]["reward_canary"],
                    source_hashes={
                        role: _artifact_hash_by_role(manifest, role)
                        for role in ("screen_evaluator_source", "metric_source")
                    },
                )
            )
        screen_result = evaluate_public_expert_development_screen(tuple(screen_clips))
        screen_path, screen_sha256 = _publish_json(screen_receipt_path, screen_result)
        index = {
            "index_id": "reference_corpus_v1_content_index/v1",
            "schema_version": 1,
            "runner_id": RUNNER_ID,
            "corpus_manifest": {
                "path": corpus_manifest_path.as_posix(),
                "sha256": corpus_manifest_sha256,
            },
            "tier_d_aggregate": {
                "path": aggregate_path.as_posix(),
                "sha256": aggregate_sha256,
            },
            "e3_certificate": {"path": e3_path.as_posix(), "sha256": e3_sha256},
            "development_screen": {"path": screen_path.as_posix(), "sha256": screen_sha256},
            "attempt_ledger": {
                "path": ledger.as_posix(),
                "sha256": sha256_file(ledger),
            },
            "clip_count": len(bundles),
            "clips": [
                {
                    "clip_id": bundle.clip_id,
                    "bundle_manifest_sha256": bundle.manifest_sha256,
                    "payload_sha256": bundle.payload_sha256,
                    "reference_identity_sha256": bundle.reference_identity_sha256,
                    "tier_d_certificate_sha256": certificates[bundle.clip_id]["certificate_sha256"],
                }
                for bundle in bundles
            ],
            "object_count": sum(1 for path in (root / "objects").rglob("*") if path.is_file()),
            "screen_passed": screen_result["screen_passed"],
            "e3_qualifying_corpus": e3_result["qualifying_corpus"],
            "claim_ceiling": (
                "same_runtime_replay_e3_identifiability_and_imported_expert_development_screen_only"
            ),
        }
        index["index_content_sha256"] = sha256_json(index)
        index_path, index_sha256 = _publish_json(root / "corpus_index_v1.json", index)
        result = {
            "runner_id": RUNNER_ID,
            "index_path": index_path.as_posix(),
            "index_sha256": index_sha256,
            "corpus_manifest_sha256": corpus_manifest_sha256,
            "tier_d_aggregate_sha256": aggregate_sha256,
            "e3_certificate_sha256": e3_sha256,
            "e3_passed_blocks": e3_result["passed_block_count"],
            "e3_failed_blocks": e3_result["failed_block_count"],
            "screen_receipt_sha256": screen_sha256,
            "screen_passed": screen_result["screen_passed"],
        }
        result_path, result_sha256 = _publish_json(root / "run_result_v1.json", result)
        result["run_result_path"] = result_path.as_posix()
        result["run_result_sha256"] = result_sha256
        return result
    except BaseException as exc:
        _append_attempt(
            ledger,
            {
                "event": "stopped",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "no_retry": True,
            },
        )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--screen-receipt", type=Path, default=DEFAULT_SCREEN_RECEIPT)
    arguments = parser.parse_args(argv)
    result = generate_reference_corpus(
        artifact_root=arguments.artifact_root,
        screen_receipt_path=arguments.screen_receipt,
    )
    print(canonical_json_bytes(result).decode())
    return 0


if __name__ == "__main__":  # pragma: no cover - one-shot CLI
    raise SystemExit(main())


__all__ = [
    "ACTOR_REGISTRY",
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_E3_MANIFEST",
    "DEFAULT_SCREEN_RECEIPT",
    "RUNNER_ID",
    "generate_reference_corpus",
    "main",
]
