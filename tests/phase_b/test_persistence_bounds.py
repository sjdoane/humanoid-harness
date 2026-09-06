from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pytest

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.phase_b import persistence as persistence_module
from oracle_composition.phase_b.persistence import load_full_checkpoint, publish_final_persistence
from oracle_composition.phase_b.runtime import fake_environment_factories, fake_policy_factory
from oracle_composition.phase_b.training import PPORecipe, TrainingPlan, run_ppo_training


def _member(name: str) -> ZipInfo:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100600 << 16
    return info


@pytest.fixture(scope="module")
def checkpoint_bytes(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    output = tmp_path_factory.mktemp("bounded-checkpoint")
    plan = TrainingPlan(
        seed=11,
        transitions=16,
        manifest_sha256="d" * 64,
        evidence_class="interface_check",
        promotable=False,
        smoke=False,
        steps_per_environment=4,
        recipe=PPORecipe(batch_size=16, n_epochs=1),
        test_only=True,
    )
    result = run_ppo_training(
        plan=plan,
        policy_factory=fake_policy_factory,
        environment_factories=fake_environment_factories(plan=plan),
    )
    return publish_final_persistence(
        output_directory=output, result=result, plan=plan
    ).checkpoint.path.read_bytes()


def _parts(encoded: bytes) -> list[tuple[str, bytes]]:
    with ZipFile(io.BytesIO(encoded)) as archive:
        return [(info.filename, archive.read(info)) for info in archive.infolist()]


def _archive(parts: list[tuple[str, bytes]], path: Path) -> tuple[Path, str]:
    with ZipFile(path, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in parts:
            archive.writestr(_member(name), payload, compresslevel=9)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_version", "NPY 1.0"),
        ("hostile_header", "invalid NPY header"),
        ("wrong_dtype", "array schema"),
        ("wrong_order", "member order"),
        ("wrong_member_set", "member order"),
        ("over_count", "member count"),
        ("oversized_member", "member expansion"),
        ("oversized_manifest", "bound"),
    ],
)
def test_full_checkpoint_loader_rejects_archive_and_npy_adversaries(
    tmp_path: Path,
    checkpoint_bytes: bytes,
    mutation: str,
    message: str,
) -> None:
    parts = _parts(checkpoint_bytes)
    manifest = json.loads(parts[0][1])
    if mutation in {"wrong_version", "hostile_header", "wrong_dtype", "oversized_member"}:
        payload = bytearray(parts[1][1])
        if mutation == "wrong_version":
            payload[6:8] = b"\x02\x00"
        elif mutation == "wrong_dtype":
            payload[10:] = payload[10:].replace(b"<f4", b"<i4", 1)
        elif mutation == "hostile_header":
            header_length = int.from_bytes(payload[8:10], "little")
            hostile = b"{'descr': '<f4', 'fortran_order': False, 'shape': (__import__('os'),), }"
            payload[10 : 10 + header_length] = hostile.ljust(header_length - 1) + b"\n"
        else:
            payload.extend(b"x" * 2_048)
        parts[1] = (parts[1][0], bytes(payload))
        manifest["array_records"][0]["npy_sha256"] = hashlib.sha256(payload).hexdigest()
        parts[0] = ("manifest.json", canonical_json_bytes(manifest))
    elif mutation == "wrong_order":
        parts[1], parts[2] = parts[2], parts[1]
    elif mutation == "wrong_member_set":
        parts[1] = ("arrays/wrong.npy", parts[1][1])
    elif mutation == "over_count":
        parts.append(("arrays/9999.npy", parts[-1][1]))
    elif mutation == "oversized_manifest":
        parts[0] = ("manifest.json", b"x" * (256 * 1024 + 1))
    path, digest = _archive(parts, tmp_path / f"{mutation}.npz")
    with pytest.raises(ExperimentContractError, match=message):
        load_full_checkpoint(path, expected_sha256=digest)


def test_full_checkpoint_loader_enforces_total_expansion_before_member_read(
    tmp_path: Path,
    checkpoint_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "total-expansion.npz"
    path.write_bytes(checkpoint_bytes)
    digest = hashlib.sha256(checkpoint_bytes).hexdigest()
    monkeypatch.setattr(persistence_module, "MAX_CHECKPOINT_EXPANDED_BYTES", 1)

    def forbidden_read(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("member read occurred before aggregate expansion refusal")

    monkeypatch.setattr(persistence_module, "_bounded_zip_read", forbidden_read)
    with pytest.raises(ExperimentContractError, match="total expansion"):
        load_full_checkpoint(path, expected_sha256=digest)
