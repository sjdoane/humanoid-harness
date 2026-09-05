from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from oracle_composition.experiments.external_tqc_actor_equivalence import (
    verify_external_actor_equivalence,
)
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_actor_npz import validate_actor_arrays
from oracle_composition.sources import _external_sb3_actor_worker as worker
from oracle_composition.sources import external_sb3_actor as importer
from oracle_composition.sources.external_payload_policy import (
    FORBIDDEN_EXTERNAL_PAYLOAD_SHA256,
    TrackedBlob,
    validate_tracked_blobs,
)
from oracle_composition.sources.farama_tqc_sibling_registrations import (
    SIBLING_REGISTRATION_SPECS,
)

REPOSITORY = Path(__file__).resolve().parents[2]
EXPECTED_ACTORS = {
    "medium": {
        "npz_sha256": "2677ebb70cd20e0ba8f8a591814fc853e325277db65184ebafb5bf5e21198b04",
        "npz_bytes": 617_761,
        "state_sha256": "cffb5679e3ef10cc99980819013941cb55cdc9f2198ba4a91a306dfc5f73e00c",
    },
    "simple": {
        "npz_sha256": "b09aa921316640024e9703671d328d7ecbabe76f04cfed8c1d8fb86fbd95a917",
        "npz_bytes": 618_054,
        "state_sha256": "3068049a0d18b7924d117a96aafee7127e411a70eba344ff985d31e97da3312b",
    },
}


def _source_paths(variant: str) -> tuple[Path, Path, Path]:
    external = REPOSITORY / f"artifacts/external/farama-minari-humanoid-v5-tqc-{variant}"
    prefix = f"humanoid-v5-TQC-{variant}"
    registration = (
        REPOSITORY
        / f"research/source_controllers/farama_minari_humanoid_v5_tqc_{variant}/RECEIPT.json"
    )
    return registration, external / prefix / "policy.pth", external / prefix / "data"


@pytest.fixture(scope="module")
def imported_siblings(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    output = tmp_path_factory.mktemp("external-tqc-siblings")
    result: dict[str, object] = {}
    for variant in ("medium", "simple"):
        registration, policy, metadata = _source_paths(variant)
        if not policy.is_file() or not metadata.is_file():
            pytest.skip("ignored pinned sibling actor payload is not installed")
        actor_path = output / f"{variant}.npz"
        import_path = output / f"{variant}-import.json"
        equivalence_path = output / f"{variant}-equivalence.json"
        authority = importer.import_external_sb3_actor(
            source_variant=variant,
            registration_receipt_path=registration,
            policy_path=policy,
            metadata_path=metadata,
            dependency_lock_path=REPOSITORY / "uv.lock",
            actor_output_path=actor_path,
            receipt_output_path=import_path,
            actor_artifact_label=(
                f"artifacts/bootstrap_tqc_humanoid/"
                f"farama_minari_humanoid_v5_tqc_{variant}_actor_v1.npz"
            ),
        )
        equivalence = verify_external_actor_equivalence(
            authority,
            receipt_output_path=equivalence_path,
        )
        result[variant] = {
            "authority": authority,
            "equivalence": equivalence,
            "actor_path": actor_path,
            "import_path": import_path,
            "equivalence_path": equivalence_path,
        }
    return result


@pytest.mark.parametrize("variant", ["medium", "simple"])
def test_sibling_import_has_exact_stable_npz_and_equivalent_outputs(
    imported_siblings: dict[str, object],
    variant: str,
) -> None:
    record = imported_siblings[variant]
    authority = record["authority"]
    equivalence = record["equivalence"]
    expected = EXPECTED_ACTORS[variant]
    spec = SIBLING_REGISTRATION_SPECS[variant]
    import_receipt = authority.to_receipt_dict()
    equivalence_receipt = equivalence.to_dict()

    assert authority.source_variant == variant
    assert authority.loaded_actor.content_sha256 == expected["npz_sha256"]
    assert authority.loaded_actor.byte_count == expected["npz_bytes"]
    assert authority.loaded_actor.state_sha256 == expected["state_sha256"]
    expected_shapes = {
        name.removeprefix("actor."): list(shape) for name, shape in worker.ACTOR_STATE_SCHEMA
    }
    validated_arrays = validate_actor_arrays(authority.loaded_actor.arrays)
    assert {name: list(validated_arrays[name].shape) for name in expected_shapes} == (
        expected_shapes
    )
    assert import_receipt["source"]["repository_commit"] == spec.repository_commit
    assert import_receipt["source_metadata"]["scalars"]["num_timesteps"] == (spec.num_timesteps)
    assert import_receipt["source_metadata"]["scalars"]["_total_timesteps"] == (
        spec.total_timesteps
    )
    assert import_receipt["model_card"]["verified"] is False
    assert equivalence_receipt["source_variant"] == variant
    assert equivalence_receipt["source_policy_sha256"] == spec.policy_sha256
    assert all(
        pair["source"] == pair["strict_npz"]
        for pair in equivalence_receipt["paired_output_sha256"].values()
    )
    with pytest.raises(ExperimentContractError, match="forbidden external payload bytes"):
        validate_tracked_blobs(
            [
                TrackedBlob(
                    path="renamed-derived-controller.npz",
                    content=record["actor_path"].read_bytes(),
                )
            ],
            forbidden_sha256=FORBIDDEN_EXTERNAL_PAYLOAD_SHA256,
        )

    recorded_import = (
        REPOSITORY / f"artifacts/bootstrap_tqc_humanoid/external_actor_import_{variant}_v1.json"
    )
    recorded_actor = (
        REPOSITORY / f"artifacts/bootstrap_tqc_humanoid/"
        f"farama_minari_humanoid_v5_tqc_{variant}_actor_v1.npz"
    )
    recorded_equivalence = (
        REPOSITORY
        / f"artifacts/bootstrap_tqc_humanoid/external_actor_equivalence_{variant}_v1.json"
    )
    if recorded_import.is_file():
        assert record["import_path"].read_bytes() == recorded_import.read_bytes()
    if recorded_actor.is_file():
        assert record["actor_path"].read_bytes() == recorded_actor.read_bytes()
    if recorded_equivalence.is_file():
        assert record["equivalence_path"].read_bytes() == recorded_equivalence.read_bytes()


@pytest.mark.parametrize("variant", ["medium", "simple"])
@pytest.mark.parametrize("mutation", ["hash", "byte_count"])
def test_sibling_policy_hash_or_byte_count_refuses_before_load_or_export(
    variant: str,
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registration, policy, metadata = _source_paths(variant)
    if not policy.is_file() or not metadata.is_file():
        pytest.skip("ignored pinned sibling actor payload is not installed")
    payload = bytearray(policy.read_bytes())
    if mutation == "hash":
        payload[0] ^= 1
    else:
        payload.append(0)
    candidate = tmp_path / "policy.pth"
    candidate.write_bytes(payload)

    monkeypatch.setattr(
        importer,
        "_run_loader_subprocess",
        lambda _payload: pytest.fail("worker must not receive mismatched sibling bytes"),
    )
    actor_output = tmp_path / "actor.npz"
    receipt_output = tmp_path / "import.json"
    with pytest.raises(ExperimentContractError, match=r"SHA-256|byte count"):
        importer.import_external_sb3_actor(
            source_variant=variant,
            registration_receipt_path=registration,
            policy_path=candidate,
            metadata_path=metadata,
            dependency_lock_path=REPOSITORY / "uv.lock",
            actor_output_path=actor_output,
            receipt_output_path=receipt_output,
            actor_artifact_label=f"artifacts/bootstrap_tqc_humanoid/{variant}.npz",
        )
    assert not actor_output.exists()
    assert not receipt_output.exists()


def test_sibling_policy_key_schema_refuses_before_export(tmp_path: Path) -> None:
    state = {
        name: torch.zeros(shape, dtype=torch.float32) for name, shape in worker.POLICY_STATE_SCHEMA
    }
    state.pop(worker.POLICY_STATE_SCHEMA[0][0])
    with pytest.raises(worker.WorkerRefusal, match="policy state schema refused"):
        worker._validate_policy_state(state, torch)
    assert not (tmp_path / "actor.npz").exists()


@pytest.mark.parametrize("variant", ["medium", "simple"])
def test_sibling_pinned_policy_digest_is_recorded_with_algorithm(variant: str) -> None:
    _, policy, _ = _source_paths(variant)
    spec = SIBLING_REGISTRATION_SPECS[variant]
    if not policy.is_file():
        pytest.skip("ignored pinned sibling actor payload is not installed")
    payload = policy.read_bytes()
    assert len(payload) == spec.policy_byte_count
    assert hashlib.sha256(payload).hexdigest() == spec.policy_sha256
    with pytest.raises(ExperimentContractError, match="forbidden external payload bytes"):
        validate_tracked_blobs(
            [TrackedBlob(path="renamed-controller.bin", content=payload)],
            forbidden_sha256=FORBIDDEN_EXTERNAL_PAYLOAD_SHA256,
        )
