from __future__ import annotations

import ast
import base64
import copy
import hashlib
import io
import json
import os
import pickle
import stat
import subprocess
import sys
import warnings
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import cloudpickle
import pytest
import torch
from sb3_contrib import TQC
from stable_baselines3.common import save_util

from oracle_composition.experiments import (
    tqc_actor_equivalence_v2,
    tqc_development_persistence_v2,
    tqc_development_training_v2,
    tqc_development_worker_v2,
)
from oracle_composition.experiments.external_tqc_actor_equivalence import (
    validate_external_actor_equivalence_receipt,
    verify_external_actor_equivalence,
)
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.sources import _external_sb3_actor_worker as worker
from oracle_composition.sources import external_sb3_actor as importer

REPOSITORY = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = REPOSITORY / "artifacts/external/farama-minari-humanoid-v5-tqc-expert"
POLICY_PATH = EXTERNAL_ROOT / "humanoid-v5-TQC-expert/policy.pth"
METADATA_PATH = EXTERNAL_ROOT / "humanoid-v5-TQC-expert/data"
LOCK_PATH = REPOSITORY / "uv.lock"
RECORDED_IMPORT_RECEIPT = (
    REPOSITORY / "artifacts/bootstrap_tqc_humanoid/external_actor_import_v1.json"
)
EXPECTED_NPZ_SHA256 = "60987a4e054db2e04f9cb3ab73e13dfe8e2f3ec7dec46346d2b9d0277ad18d9b"
EXPECTED_ACTOR_STATE_SHA256 = "3fd39cc715a10126fd92b20f6ce213c380eb4d5df843a42315aac50cf116748a"
EXPECTED_SAFE_GLOBALS_COUNT = 75
EXPECTED_SAFE_GLOBALS_SHA256 = "7b70391289d8e8d285612f5ee4db68739af844eae8d945964d1174c83ce7d4b9"


def _policy_state() -> dict[str, torch.Tensor]:
    return {
        name: torch.full(shape, (index + 1) / 1000.0, dtype=torch.float32)
        for index, (name, shape) in enumerate(worker.POLICY_STATE_SCHEMA)
    }


@pytest.fixture(scope="module")
def synthetic_policy_bytes() -> bytes:
    stream = io.BytesIO()
    torch.save(_policy_state(), stream)
    payload = stream.getvalue()
    assert 0 < len(payload) <= importer.MAX_POLICY_BYTES
    return payload


def _zip_bytes(entries: list[tuple[ZipInfo | str, bytes]]) -> bytes:
    stream = io.BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
        for name, payload in entries:
            archive.writestr(name, payload)
    return stream.getvalue()


def _bomb(*_args: object, **_kwargs: object) -> object:
    raise AssertionError("forbidden deserializer was called")


def test_fresh_worker_records_the_fresh_torch_safe_globals(
    synthetic_policy_bytes: bytes,
) -> None:
    arrays, header = importer._run_loader_subprocess(synthetic_policy_bytes)
    probe = (
        "import hashlib,json,torch\n"
        "entries=torch.serialization.get_safe_globals()\n"
        "names=tuple(sorted(f'{x.__module__}.{x.__qualname__}' for x in entries))\n"
        "raw=json.dumps(names,ensure_ascii=True,separators=(',',':')).encode('ascii')\n"
        "print(json.dumps({'count':len(entries),'sha256':hashlib.sha256(raw).hexdigest()}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        capture_output=True,
        check=False,
        text=True,
        env={
            "LANG": "C",
            "LC_ALL": "C",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        },
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    observed = json.loads(completed.stdout)
    torch_load = header["torch_load"]

    assert torch_load["safe_globals_count"] == observed["count"]
    assert torch_load["safe_globals_sha256"] == observed["sha256"]
    assert torch_load["safe_globals_module_allowlist_passed"] is True
    assert observed == {
        "count": EXPECTED_SAFE_GLOBALS_COUNT,
        "sha256": EXPECTED_SAFE_GLOBALS_SHA256,
    }
    assert tuple(arrays) == (
        "latent_pi.0.weight",
        "latent_pi.0.bias",
        "latent_pi.2.weight",
        "latent_pi.2.bias",
        "mu.weight",
        "mu.bias",
        "log_std.weight",
        "log_std.bias",
        "action_low",
        "action_high",
        "format_version",
    )


def test_worker_refuses_one_registered_user_safe_global_before_load() -> None:
    worker_path = Path(worker.__file__).resolve()
    probe = f"""
import importlib.util
import torch
spec = importlib.util.spec_from_file_location("hh_external_worker", {os.fspath(worker_path)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
class UserRegisteredGlobal:
    pass
torch.serialization.add_safe_globals([UserRegisteredGlobal])
try:
    module._load_and_encode(b"not-a-checkpoint", {{}})
except module.WorkerRefusal as exc:
    print(str(exc))
else:
    raise SystemExit(3)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-c", probe],
        capture_output=True,
        check=False,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert completed.stdout.strip() == "process safe globals refused"


def test_non_allowlisted_safe_global_module_refuses() -> None:
    class ThirdPartyGlobal:
        pass

    ThirdPartyGlobal.__module__ = "third_party_checkpoint_extension"
    with pytest.raises(worker.WorkerRefusal, match="process safe globals refused"):
        worker._safe_globals_observation(
            [ThirdPartyGlobal],
            refusal_category="process safe globals refused",
        )


def test_safe_global_change_after_the_single_load_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations = iter(([ValueError], [ValueError, KeyError]))
    calls: list[tuple[object, dict[str, object]]] = []

    monkeypatch.setattr(torch.serialization, "get_safe_globals", lambda: next(observations))

    def fake_load(payload: object, **kwargs: object) -> object:
        calls.append((payload, kwargs))
        return _policy_state()

    monkeypatch.setattr(torch, "load", fake_load)
    with pytest.raises(worker.WorkerRefusal, match="process safe globals changed"):
        worker._load_and_encode(b"held", {})

    assert len(calls) == 1
    assert isinstance(calls[0][0], io.BytesIO)
    assert calls[0][1] == {"map_location": "cpu", "weights_only": True}


def test_worker_contains_exactly_one_weights_only_torch_load_and_no_registration() -> None:
    source = Path(worker.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "torch"
        and node.func.attr == "load"
    ]

    assert len(calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert isinstance(keywords["weights_only"], ast.Constant)
    assert keywords["weights_only"].value is True
    assert isinstance(keywords["map_location"], ast.Constant)
    assert keywords["map_location"].value == "cpu"
    assert "add_safe_globals" not in source
    assert "torch.serialization.safe_globals" not in source
    for forbidden in ("TQC.load", "load_from_zip_file", "cloudpickle", "pickle.loads"):
        assert forbidden not in source


@pytest.mark.parametrize(
    "name",
    ["../data.pkl", "/root/data.pkl", "root\\data.pkl", "root/../data.pkl"],
)
def test_torch_zip_preflight_rejects_unsafe_member_paths(name: str) -> None:
    with pytest.raises(ExperimentContractError, match="path is unsafe"):
        importer.preflight_torch_zip(_zip_bytes([(name, b"payload")]))


def test_torch_zip_preflight_rejects_duplicates_symlinks_and_nested_archives() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        duplicate = _zip_bytes([("root/data.pkl", b"one"), ("root/data.pkl", b"two")])
    with pytest.raises(ExperimentContractError, match="duplicate"):
        importer.preflight_torch_zip(duplicate)

    link = ZipInfo("root/data.pkl")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with pytest.raises(ExperimentContractError, match="metadata is unsafe"):
        importer.preflight_torch_zip(_zip_bytes([(link, b"target")]))

    with pytest.raises(ExperimentContractError, match="nested archive"):
        importer.preflight_torch_zip(_zip_bytes([("root/data.pkl", b"PK\x03\x04nested")]))


def test_torch_zip_preflight_rejects_entry_and_expansion_bombs() -> None:
    too_many = [
        (f"root/.data/member-{index}", b"x") for index in range(importer.MAX_TORCH_ZIP_ENTRIES + 1)
    ]
    with pytest.raises(ExperimentContractError, match="entry count"):
        importer.preflight_torch_zip(_zip_bytes(too_many))

    oversized = b"\0" * (importer.MAX_TORCH_ZIP_UNCOMPRESSED_BYTES + 1)
    with pytest.raises(ExperimentContractError, match="expansion"):
        importer.preflight_torch_zip(_zip_bytes([("root/data.pkl", oversized)]))


class _TensorSubclass(torch.Tensor):
    pass


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("renamed", "schema"),
        ("oversized_storage", "storage bound"),
        ("subclass", "tensor contract"),
        ("alias", "storage alias"),
        ("nonfinite", "non-finite"),
    ],
)
def test_policy_state_rejects_schema_and_tensor_attacks(
    mutation: str,
    error: str,
) -> None:
    state = _policy_state()
    if mutation == "renamed":
        value = state.pop("actor.mu.bias")
        state["actor.renamed.bias"] = value
    elif mutation == "oversized_storage":
        name, shape = worker.POLICY_STATE_SCHEMA[0]
        base = torch.zeros(torch.tensor(shape).prod().item() + 1, dtype=torch.float32)
        state[name] = base[:-1].reshape(shape)
    elif mutation == "subclass":
        name, _ = worker.POLICY_STATE_SCHEMA[0]
        state[name] = torch.Tensor._make_subclass(_TensorSubclass, state[name], False)
    elif mutation == "alias":
        state["actor.latent_pi.2.bias"] = state["actor.latent_pi.0.bias"]
    else:
        state["actor.mu.bias"][0] = torch.nan

    with pytest.raises(worker.WorkerRefusal, match=error):
        worker._validate_policy_state(state, torch)


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        (b'{"num_timesteps":1,"num_timesteps":2}', "duplicate key"),
        (b'{"num_timesteps":NaN}', "non-finite constant"),
        (
            json.dumps({"nested": [[[[[[[[[[[[[[[[[0]]]]]]]]]]]]]]]]]}).encode(),
            "nesting",
        ),
        (
            json.dumps({"oversized": "x" * (importer.MAX_METADATA_STRING_LENGTH + 1)}).encode(),
            "string exceeds",
        ),
    ],
)
def test_strict_metadata_rejects_ambiguous_or_unbounded_json(
    payload: bytes,
    error: str,
) -> None:
    with pytest.raises(ExperimentContractError, match=error):
        importer.parse_sb3_metadata_bytes(payload)


def test_serialized_metadata_marker_is_never_exposed() -> None:
    marker = "SERIALIZED_PAYLOAD_MARKER"
    payload = json.dumps(
        {"policy_class": {":type:": "cloudpickle", ":serialized:": marker}}
    ).encode()

    with pytest.raises(ExperimentContractError) as failure:
        importer.parse_sb3_metadata_bytes(payload)
    assert marker not in str(failure.value)


def test_actual_metadata_returns_only_allowlisted_scalars_and_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not METADATA_PATH.is_file():
        pytest.skip("ignored pinned metadata is not installed")
    monkeypatch.setattr(pickle, "loads", _bomb)
    monkeypatch.setattr(cloudpickle, "loads", _bomb)
    monkeypatch.setattr(base64, "b64decode", _bomb)

    result = importer.parse_sb3_metadata_bytes(METADATA_PATH.read_bytes())

    assert set(result) == {
        "scalars",
        "serialized_field_paths",
        "serialized_field_count",
    }
    assert result["scalars"] == importer.EXPECTED_METADATA
    assert result["serialized_field_paths"] == list(importer.EXPECTED_SERIALIZED_PATHS)
    assert result["serialized_field_count"] == 11
    assert len(json.dumps(result)) < 2_000


def test_single_open_reader_rejects_hash_count_and_symlink_then_keeps_held_bytes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "policy.pth"
    original = b"trusted-held-bytes"
    path.write_bytes(original)
    digest = hashlib.sha256(original).hexdigest()

    with pytest.raises(ExperimentContractError, match="byte count"):
        importer._read_regular_file_once(
            path,
            maximum_bytes=1024,
            label="fixture",
            expected_sha256=digest,
            expected_byte_count=len(original) + 1,
        )
    with pytest.raises(ExperimentContractError, match="SHA-256"):
        importer._read_regular_file_once(
            path,
            maximum_bytes=1024,
            label="fixture",
            expected_sha256="0" * 64,
            expected_byte_count=len(original),
        )

    link = tmp_path / "link.pth"
    link.symlink_to(path)
    with pytest.raises(ExperimentContractError, match="without following links"):
        importer._read_regular_file_once(
            link,
            maximum_bytes=1024,
            label="fixture",
            expected_sha256=digest,
            expected_byte_count=len(original),
        )

    held = importer._read_regular_file_once(
        path,
        maximum_bytes=1024,
        label="fixture",
        expected_sha256=digest,
        expected_byte_count=len(original),
    )
    path.write_bytes(b"path-was-swapped")
    assert held == original


def test_pinned_import_and_external_equivalence_are_exact_and_separate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not POLICY_PATH.is_file() or not METADATA_PATH.is_file():
        pytest.skip("ignored pinned Farama actor payload is not installed")
    monkeypatch.setattr(pickle, "loads", _bomb)
    monkeypatch.setattr(cloudpickle, "loads", _bomb)
    monkeypatch.setattr(base64, "b64decode", _bomb)
    monkeypatch.setattr(TQC, "load", _bomb)
    monkeypatch.setattr(save_util, "load_from_zip_file", _bomb)

    authority = importer.import_external_sb3_actor(
        policy_path=POLICY_PATH,
        metadata_path=METADATA_PATH,
        dependency_lock_path=LOCK_PATH,
        actor_output_path=tmp_path / "actor.npz",
        receipt_output_path=tmp_path / "import.json",
        actor_artifact_label=(
            "artifacts/bootstrap_tqc_humanoid/farama_minari_humanoid_v5_tqc_actor_v1.npz"
        ),
    )
    receipt = authority.to_receipt_dict()

    assert authority.loaded_actor.content_sha256 == EXPECTED_NPZ_SHA256
    assert authority.loaded_actor.state_sha256 == EXPECTED_ACTOR_STATE_SHA256
    assert receipt["authority"] == "external_pretrained_artifact"
    assert receipt["deserialization"]["call_count"] == 1
    assert receipt["deserialization"]["safe_globals_count"] == EXPECTED_SAFE_GLOBALS_COUNT
    assert receipt["deserialization"]["safe_globals_sha256"] == EXPECTED_SAFE_GLOBALS_SHA256
    assert receipt["deserialization"]["safe_globals_module_allowlist_passed"] is True
    assert receipt["source_metadata"]["timestep_counters_reconciled"] is False
    assert receipt["source_versions"]["sb3_contrib"] == "unknown"
    assert receipt["model_card"]["verified"] is False
    assert receipt["environment_termination"]["same_mdp_claimed"] is False
    assert receipt["strict_actor_npz"]["eligible_for_git"] is False

    if RECORDED_IMPORT_RECEIPT.is_file():
        assert (tmp_path / "import.json").read_bytes() == RECORDED_IMPORT_RECEIPT.read_bytes()

    equivalence = verify_external_actor_equivalence(
        authority,
        receipt_output_path=tmp_path / "equivalence.json",
    )
    result = equivalence.to_dict()
    assert result["actor_state"]["source_sha256"] == EXPECTED_ACTOR_STATE_SHA256
    assert result["actor_state"]["source_sha256"] == result["actor_state"]["strict_npz_sha256"]
    for pair in result["paired_output_sha256"].values():
        assert pair["source"] == pair["strict_npz"]

    missing_counter = copy.deepcopy(receipt)
    missing_counter["source_metadata"]["scalars"].pop("num_timesteps")
    with pytest.raises(ExperimentContractError, match="source metadata"):
        importer.validate_external_actor_import_receipt(missing_counter)
    missing_version = copy.deepcopy(receipt)
    missing_version["source_versions"].pop("sb3_contrib")
    with pytest.raises(ExperimentContractError, match="identity"):
        importer.validate_external_actor_import_receipt(missing_version)
    extra_deserializer_field = copy.deepcopy(receipt)
    extra_deserializer_field["deserialization"]["safe_global_names"] = []
    with pytest.raises(ExperimentContractError, match="deserialization contract"):
        importer.validate_external_actor_import_receipt(extra_deserializer_field)
    overclaimed_import = copy.deepcopy(receipt)
    overclaimed_import["claim"]["establishes"] = "stable Humanoid locomotion"
    with pytest.raises(ExperimentContractError, match="claim ceiling"):
        importer.validate_external_actor_import_receipt(overclaimed_import)

    mismatched_output = copy.deepcopy(result)
    mismatched_output["paired_output_sha256"]["mean"]["strict_npz"] = "0" * 64
    with pytest.raises(ExperimentContractError, match="mean hashes differ"):
        validate_external_actor_equivalence_receipt(mismatched_output)
    overclaimed_equivalence = copy.deepcopy(result)
    overclaimed_equivalence["claim"]["establishes"] = "E1"
    with pytest.raises(ExperimentContractError, match="claim ceiling"):
        validate_external_actor_equivalence_receipt(overclaimed_equivalence)

    local_rejections = (
        lambda: tqc_actor_equivalence_v2.admit_final_tqc_checkpoint_actor(authority),
        lambda: tqc_actor_equivalence_v2.verify_actor_equivalence_v2(authority),
        lambda: tqc_actor_equivalence_v2.revalidate_tqc_actor_equivalence_receipt_v2(equivalence),
        lambda: tqc_development_persistence_v2.revalidate_tqc_persistence_authority_v2(authority),
        lambda: tqc_development_persistence_v2.strict_reload_tqc_persistence_v2(
            authority,
            actor_equivalence_receipt=equivalence,
        ),
        lambda: tqc_development_persistence_v2.revalidate_tqc_strict_reload_authority_v2(
            equivalence
        ),
        lambda: tqc_development_training_v2.revalidate_tqc_training_completion_v2(equivalence),
        lambda: tqc_development_worker_v2.revalidate_acknowledged_tqc_execution_manifest_v2(
            equivalence
        ),
    )
    for local_api in local_rejections:
        with pytest.raises(ExperimentContractError):
            local_api()

    serialized_receipt_rejections = (
        tqc_actor_equivalence_v2.admit_final_tqc_checkpoint_actor,
        tqc_actor_equivalence_v2.revalidate_tqc_actor_equivalence_receipt_v2,
        tqc_development_persistence_v2.revalidate_tqc_persistence_authority_v2,
        tqc_development_persistence_v2.revalidate_tqc_strict_reload_authority_v2,
        tqc_development_training_v2.revalidate_tqc_training_completion_v2,
        tqc_development_worker_v2.revalidate_acknowledged_tqc_execution_manifest_v2,
    )
    for external_receipt in (receipt, result):
        for local_api in serialized_receipt_rejections:
            with pytest.raises(ExperimentContractError):
                local_api(external_receipt)
