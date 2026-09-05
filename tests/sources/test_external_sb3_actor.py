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
    REPOSITORY / "artifacts/bootstrap_tqc_humanoid/external_actor_import_expert_03a3_v1.json"
)
RECORDED_EQUIVALENCE_RECEIPT = (
    REPOSITORY / "artifacts/bootstrap_tqc_humanoid/external_actor_equivalence_expert_03a3_v1.json"
)
REGISTRATION_RECEIPT = (
    REPOSITORY / "research/source_controllers/farama_minari_humanoid_v5_tqc_expert/RECEIPT.json"
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


def _worker_resource_limits() -> dict[str, object]:
    return {
        "cpu": {
            "resource": "RLIMIT_CPU",
            "soft_seconds": worker.CPU_TIME_LIMIT_SECONDS,
            "hard_seconds": worker.CPU_TIME_LIMIT_SECONDS,
        },
        "address_space": {
            "resource": "RLIMIT_AS",
            "requested_bytes": worker.ADDRESS_SPACE_LIMIT_BYTES,
            "finite_enforced": False,
            "observed_soft": -1,
            "observed_hard": -1,
            "darwin_finite_limit_unavailable": True,
        },
    }


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


@pytest.mark.parametrize(
    "observations",
    [
        ([ValueError, KeyError], [KeyError, ValueError]),
        ([ValueError, KeyError], [ValueError, TypeError]),
        ([ValueError], [ValueError, KeyError]),
    ],
)
def test_safe_global_change_after_the_single_load_refuses(
    monkeypatch: pytest.MonkeyPatch,
    observations: tuple[list[type[BaseException]], list[type[BaseException]]],
) -> None:
    observations_iterator = iter(observations)
    calls: list[tuple[object, dict[str, object]]] = []

    monkeypatch.setattr(
        torch.serialization,
        "get_safe_globals",
        lambda: next(observations_iterator),
    )

    def fake_load(payload: object, **kwargs: object) -> object:
        calls.append((payload, kwargs))
        return _policy_state()

    monkeypatch.setattr(torch, "load", fake_load)
    with pytest.raises(worker.WorkerRefusal, match="process safe globals changed"):
        worker._load_and_encode(b"held", {})

    assert len(calls) == 1
    assert isinstance(calls[0][0], io.BytesIO)
    assert calls[0][1] == {"map_location": "cpu", "weights_only": True}


def test_safe_global_receipt_hash_is_order_independent_but_observation_is_not() -> None:
    forward_names, forward_hash = worker._safe_globals_observation(
        [ValueError, KeyError],
        refusal_category="process safe globals refused",
    )
    reverse_names, reverse_hash = worker._safe_globals_observation(
        [KeyError, ValueError],
        refusal_category="process safe globals refused",
    )

    assert forward_names != reverse_names
    assert forward_hash == reverse_hash


def test_worker_refuses_runtime_drift_before_torch_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_calls = 0

    def counted_load(*_args: object, **_kwargs: object) -> object:
        nonlocal load_calls
        load_calls += 1
        return _policy_state()

    monkeypatch.setattr(torch, "load", counted_load)
    monkeypatch.setattr(
        worker,
        "_installed_torch_distribution_identity",
        lambda: {"name": "torch", "version": "0.0.0"},
    )

    with pytest.raises(worker.WorkerRefusal) as failure:
        worker._load_and_encode(b"held", {})

    assert failure.value.outcome == worker.OUTCOME_RUNTIME_DRIFT
    assert load_calls == 0


def _dotted_name(node: ast.expr) -> str | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    return ".".join([node.id, *reversed(parts)])


def test_worker_and_importer_static_ast_forbid_alternative_decoders() -> None:
    torch_load_calls: list[ast.Call] = []
    forbidden_imports = {"pickle", "cloudpickle", "base64"}
    forbidden_calls = {
        "TQC.load",
        "load_from_zip_file",
        "pickle.loads",
        "cloudpickle.loads",
        "base64.b64decode",
        "add_safe_globals",
    }

    for source_path in (Path(worker.__file__), Path(importer.__file__)):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    alias.name.split(".")[0] not in forbidden_imports for alias in node.names
                )
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] not in forbidden_imports
            elif isinstance(node, ast.Call):
                name = _dotted_name(node.func)
                if name == "torch.load":
                    torch_load_calls.append(node)
                    keywords = {keyword.arg: keyword.value for keyword in node.keywords}
                    assert isinstance(keywords.get("weights_only"), ast.Constant)
                    assert keywords["weights_only"].value is True
                assert name is None or name.split(".", 1)[0] not in forbidden_imports
                assert name is None or all(
                    name != forbidden and not name.endswith(f".{forbidden}")
                    for forbidden in forbidden_calls
                )

    assert len(torch_load_calls) == 1


def test_worker_validation_in_process_uses_no_forbidden_decoder(
    synthetic_policy_bytes: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pickle, "loads", _bomb)
    monkeypatch.setattr(cloudpickle, "loads", _bomb)
    monkeypatch.setattr(base64, "b64decode", _bomb)
    monkeypatch.setattr(TQC, "load", _bomb)
    monkeypatch.setattr(save_util, "load_from_zip_file", _bomb)
    monkeypatch.setattr(torch.serialization, "add_safe_globals", _bomb)

    response = worker._load_and_encode(synthetic_policy_bytes, _worker_resource_limits())
    arrays, header = importer._parse_worker_response(response)

    assert tuple(arrays)[:8] == tuple(
        name.removeprefix("actor.") for name, _shape in worker.ACTOR_STATE_SCHEMA
    )
    assert header["torch_load"]["call_count"] == 1


def _stub_worker_command(source: str) -> tuple[str, ...]:
    return (sys.executable, "-I", "-c", source)


def _run_stub_worker(source: str, *, timeout_seconds: float = 2.0) -> object:
    return importer._run_worker_process(
        _stub_worker_command(source),
        held_bytes=b"held",
        environment={"LANG": "C", "LC_ALL": "C"},
        timeout_seconds=timeout_seconds,
    )


@pytest.mark.parametrize(
    ("source", "outcome"),
    [
        ("import time; time.sleep(10)", worker.OUTCOME_TIMEOUT),
        (
            "import os,signal; os.kill(os.getpid(), signal.SIGTERM)",
            worker.OUTCOME_CRASH_OR_SIGNAL,
        ),
        (
            "import os; os.write(1, b'x' * (2 * 1024 * 1024 + 65536))",
            worker.OUTCOME_MALFORMED_FRAME,
        ),
    ],
)
def test_worker_process_transport_reports_timeout_signal_and_output_bound(
    source: str,
    outcome: str,
) -> None:
    timeout = 0.1 if outcome == worker.OUTCOME_TIMEOUT else 2.0
    with pytest.raises(importer.ExternalActorWorkerError) as failure:
        _run_stub_worker(source, timeout_seconds=timeout)

    assert failure.value.outcome == outcome


@pytest.mark.parametrize(
    ("outcome", "detail"),
    [
        (worker.OUTCOME_RESOURCE_REFUSAL, "CPU resource limit refused"),
        (worker.OUTCOME_SAFE_GLOBALS_DRIFT, "process safe globals changed"),
        (worker.OUTCOME_SCHEMA_OR_TENSOR_REFUSAL, "tensor contract refused"),
        (worker.OUTCOME_NON_FINITE_VALUES, "non-finite tensor refused"),
    ],
)
def test_worker_process_transport_reports_framed_refusal(
    outcome: str,
    detail: str,
) -> None:
    frame = worker._error_frame(
        worker.WorkerRefusal(
            outcome,
            detail,
        )
    )
    source = f"import os; os.write(1, bytes.fromhex({frame.hex()!r}))"

    with pytest.raises(importer.ExternalActorWorkerError) as failure:
        _run_stub_worker(source)

    assert failure.value.outcome == outcome


@pytest.mark.parametrize("kind", ["truncated", "deep"])
def test_worker_process_transport_reports_malformed_frames(kind: str) -> None:
    if kind == "truncated":
        frame = worker.RESPONSE_MAGIC
    else:
        header = ("[" * 2_000 + "0" + "]" * 2_000).encode("ascii")
        frame = worker.RESPONSE_MAGIC + worker.FRAME_LENGTHS.pack(len(header), 0) + header
    source = f"import os; os.write(1, bytes.fromhex({frame.hex()!r}))"

    with pytest.raises(importer.ExternalActorWorkerError) as failure:
        _run_stub_worker(source)

    assert failure.value.outcome == worker.OUTCOME_MALFORMED_FRAME


def test_worker_process_transport_accepts_one_bounded_success_frame(
    synthetic_policy_bytes: bytes,
    tmp_path: Path,
) -> None:
    frame = worker._load_and_encode(synthetic_policy_bytes, _worker_resource_limits())
    frame_path = tmp_path / "success.frame"
    frame_path.write_bytes(frame)
    source = (
        "import os,pathlib; data=pathlib.Path("
        f"{os.fspath(frame_path)!r}).read_bytes(); os.write(1, data)"
    )

    arrays, header = _run_stub_worker(source)

    assert len(arrays) == 11
    assert header["outcome"] == worker.OUTCOME_SUCCESS


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


def test_policy_state_rejects_partially_overlapping_storage_intervals() -> None:
    state = _policy_state()
    backing = bytearray(257 * 4)
    state["actor.latent_pi.0.bias"] = torch.frombuffer(
        memoryview(backing)[: 256 * 4],
        dtype=torch.float32,
    )
    state["actor.latent_pi.2.bias"] = torch.frombuffer(
        memoryview(backing)[4 : 257 * 4],
        dtype=torch.float32,
    )

    with pytest.raises(worker.WorkerRefusal, match="storage alias"):
        worker._validate_policy_state(state, torch)


def test_policy_state_accepts_distinct_storage_intervals() -> None:
    details, actor_bytes = worker._validate_policy_state(_policy_state(), torch)

    assert len(details["policy_key_inventory"]) == len(worker.POLICY_STATE_SCHEMA)
    assert len(actor_bytes) > 0


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
    monkeypatch: pytest.MonkeyPatch,
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

    replacement = tmp_path / "replacement.pth"
    replacement.write_bytes(b"path-was-swapped")
    displaced = tmp_path / "original-held.pth"
    real_open = os.open
    policy_open_count = 0

    def swap_after_open(target: object, flags: int, *args: object, **kwargs: object) -> int:
        nonlocal policy_open_count
        descriptor = real_open(target, flags, *args, **kwargs)
        if os.fspath(target) == os.fspath(path):
            policy_open_count += 1
            path.replace(displaced)
            replacement.replace(path)
        return descriptor

    monkeypatch.setattr(importer.os, "open", swap_after_open)
    monkeypatch.setattr(importer, "POLICY_SHA256", digest)
    monkeypatch.setattr(importer, "POLICY_BYTE_COUNT", len(original))
    held = importer.read_pinned_policy_bytes(path)

    assert policy_open_count == 1
    assert held == original
    assert path.read_bytes() == b"path-was-swapped"


@pytest.mark.parametrize(
    "mutation",
    ["missing", "changed_rights", "changed_remote_digest", "same_id_different_bytes"],
)
def test_registration_receipt_refuses_before_policy_read_or_worker_launch(
    mutation: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "registration.json"
    if mutation != "missing":
        payload = REGISTRATION_RECEIPT.read_bytes()
        if mutation == "same_id_different_bytes":
            candidate.write_bytes(payload + b" ")
        else:
            value = json.loads(payload)
            if mutation == "changed_rights":
                value["rights"]["payload_bytes_enter_git"] = True
            else:
                value["remote_inventory"][0]["digest"]["value"] = "0" * 40
            candidate.write_text(json.dumps(value), encoding="utf-8")

    monkeypatch.setattr(importer, "read_pinned_policy_bytes", _bomb)
    monkeypatch.setattr(importer, "_run_loader_subprocess", _bomb)

    with pytest.raises(ExperimentContractError, match="registration receipt"):
        importer.import_external_sb3_actor(
            registration_receipt_path=candidate,
            policy_path=tmp_path / "policy.pth",
            metadata_path=tmp_path / "data",
            dependency_lock_path=LOCK_PATH,
            actor_output_path=tmp_path / "actor.npz",
            receipt_output_path=tmp_path / "import.json",
            actor_artifact_label="artifacts/bootstrap_tqc_humanoid/actor.npz",
        )


def _recorded_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _replace_nested(
    value: dict[str, object], path: tuple[object, ...], replacement: object
) -> None:
    current: object = value
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = replacement


@pytest.mark.parametrize("counter", ["num_timesteps", "_total_timesteps"])
def test_import_receipt_requires_both_timestep_counters(counter: str) -> None:
    receipt = _recorded_json(RECORDED_IMPORT_RECEIPT)
    receipt["source_metadata"]["scalars"].pop(counter)

    with pytest.raises(ExperimentContractError, match="source metadata"):
        importer.validate_external_actor_import_receipt(receipt)


@pytest.mark.parametrize("version", sorted(importer.SOURCE_VERSIONS))
def test_import_receipt_requires_every_source_version(version: str) -> None:
    receipt = _recorded_json(RECORDED_IMPORT_RECEIPT)
    receipt["source_versions"].pop(version)

    with pytest.raises(ExperimentContractError, match="identity"):
        importer.validate_external_actor_import_receipt(receipt)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("schema_version",), True),
        (("torch_zip_preflight", "member_count"), True),
        (("source_metadata", "scalars", "gamma"), 1),
        (("environment_termination", "same_mdp_claimed"), 0),
        (("source_metadata", "timestep_counter_note"), 7),
    ],
)
def test_import_receipt_rejects_type_confusable_json(
    path: tuple[object, ...],
    replacement: object,
) -> None:
    receipt = _recorded_json(RECORDED_IMPORT_RECEIPT)
    _replace_nested(receipt, path, replacement)

    with pytest.raises(ExperimentContractError):
        importer.validate_external_actor_import_receipt(receipt)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("schema_version",), True),
        (("sampling_seed",), 97_001.0),
        (("architecture", "log_std_clamp", 0), -20),
        (("cpu_rng_state", "preserved"), 1),
        (("equivalence_id",), 7),
    ],
)
def test_equivalence_receipt_rejects_type_confusable_json(
    path: tuple[object, ...],
    replacement: object,
) -> None:
    receipt = _recorded_json(RECORDED_EQUIVALENCE_RECEIPT)
    _replace_nested(receipt, path, replacement)

    with pytest.raises(ExperimentContractError):
        validate_external_actor_equivalence_receipt(receipt)


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
        registration_receipt_path=REGISTRATION_RECEIPT,
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
    assert receipt["deserialization"]["ordered_names_unchanged"] is True
    assert receipt["source_metadata"]["timestep_counters_reconciled"] is False
    assert receipt["source_versions"]["sb3_contrib"] == "unknown"
    registration_binding = receipt["source"]["registration_receipt"]
    assert (
        registration_binding["sha256"]
        == hashlib.sha256(REGISTRATION_RECEIPT.read_bytes()).hexdigest()
    )
    assert registration_binding["byte_count"] == len(REGISTRATION_RECEIPT.read_bytes())
    assert registration_binding["captured_api"]["sha256"] == (
        "69ccb043f76918fd601c71420ec6511a301193dd768aab5847a4beba3ec652f6"
    )
    assert registration_binding["rights"]["payload_bytes_enter_git"] is False
    assert set(receipt["implementation"]) == {"importer_source", "worker_source"}
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
    assert result["cpu_rng_state"] == {
        "ambient_values_recorded": False,
        "preserved": True,
    }
    assert set(result["verifier"]) == {"source", "runtime"}
    for pair in result["paired_output_sha256"].values():
        assert pair["source"] == pair["strict_npz"]
    if RECORDED_EQUIVALENCE_RECEIPT.is_file():
        assert (tmp_path / "equivalence.json").read_bytes() == (
            RECORDED_EQUIVALENCE_RECEIPT.read_bytes()
        )

    for counter in ("num_timesteps", "_total_timesteps"):
        missing_counter = copy.deepcopy(receipt)
        missing_counter["source_metadata"]["scalars"].pop(counter)
        with pytest.raises(ExperimentContractError, match="source metadata"):
            importer.validate_external_actor_import_receipt(missing_counter)
    for version in importer.SOURCE_VERSIONS:
        missing_version = copy.deepcopy(receipt)
        missing_version["source_versions"].pop(version)
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


def test_equivalence_receipt_is_reproducible_across_fresh_caller_rng_states(
    tmp_path: Path,
) -> None:
    if not POLICY_PATH.is_file() or not METADATA_PATH.is_file():
        pytest.skip("ignored pinned Farama actor payload is not installed")
    probe = """
import hashlib
import json
import sys
from pathlib import Path

import torch

from oracle_composition.experiments.external_tqc_actor_equivalence import (
    verify_external_actor_equivalence,
)
from oracle_composition.sources.external_sb3_actor import import_external_sb3_actor

repository = Path(sys.argv[1])
output = Path(sys.argv[2])
torch.manual_seed(int(sys.argv[3]))
before = torch.random.get_rng_state().clone()
authority = import_external_sb3_actor(
    registration_receipt_path=(
        repository
        / "research/source_controllers/farama_minari_humanoid_v5_tqc_expert/RECEIPT.json"
    ),
    policy_path=(
        repository
        / "artifacts/external/farama-minari-humanoid-v5-tqc-expert/"
        "humanoid-v5-TQC-expert/policy.pth"
    ),
    metadata_path=(
        repository
        / "artifacts/external/farama-minari-humanoid-v5-tqc-expert/"
        "humanoid-v5-TQC-expert/data"
    ),
    dependency_lock_path=repository / "uv.lock",
    actor_output_path=output / "actor.npz",
    receipt_output_path=output / "import.json",
    actor_artifact_label=(
        "artifacts/bootstrap_tqc_humanoid/farama_minari_humanoid_v5_tqc_actor_v1.npz"
    ),
)
receipt = verify_external_actor_equivalence(
    authority,
    receipt_output_path=output / "equivalence.json",
)
after = torch.random.get_rng_state().clone()
print(json.dumps({
    "caller_before_sha256": hashlib.sha256(before.numpy().tobytes()).hexdigest(),
    "caller_after_sha256": hashlib.sha256(after.numpy().tobytes()).hexdigest(),
    "caller_state_preserved": bool(torch.equal(before, after)),
    "equivalence_sha256": receipt.receipt_sha256,
    "paired_output_sha256": receipt.to_dict()["paired_output_sha256"],
}, sort_keys=True))
"""
    observations: list[dict[str, object]] = []
    receipt_bytes: list[bytes] = []
    import_bytes: list[bytes] = []
    for index, seed in enumerate((12_301, 98_707)):
        output = tmp_path / f"fresh-{index}"
        output.mkdir()
        completed = subprocess.run(
            [sys.executable, "-c", probe, os.fspath(REPOSITORY), os.fspath(output), str(seed)],
            cwd=REPOSITORY,
            capture_output=True,
            check=False,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stderr == ""
        observations.append(json.loads(completed.stdout))
        receipt_bytes.append((output / "equivalence.json").read_bytes())
        import_bytes.append((output / "import.json").read_bytes())

    assert observations[0]["caller_before_sha256"] != observations[1]["caller_before_sha256"]
    assert all(item["caller_state_preserved"] is True for item in observations)
    assert all(item["caller_before_sha256"] == item["caller_after_sha256"] for item in observations)
    assert observations[0]["paired_output_sha256"] == observations[1]["paired_output_sha256"]
    assert observations[0]["equivalence_sha256"] == observations[1]["equivalence_sha256"]
    assert receipt_bytes[0] == receipt_bytes[1] == RECORDED_EQUIVALENCE_RECEIPT.read_bytes()
    assert import_bytes[0] == import_bytes[1] == RECORDED_IMPORT_RECEIPT.read_bytes()
