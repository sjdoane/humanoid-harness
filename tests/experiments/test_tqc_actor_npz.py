from __future__ import annotations

import hashlib
import io
import os
from dataclasses import replace
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import gymnasium as gym
import numpy as np
import pytest
import torch
from sb3_contrib.tqc.policies import Actor
from stable_baselines3.common.torch_layers import FlattenExtractor

from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_actor_npz import (
    ACTION_WIDTH,
    MAX_ARCHIVE_BYTES,
    OBSERVATION_WIDTH,
    LoadedTQCActor,
    actor_schema,
    actor_schema_sha256,
    actor_state_sha256,
    encode_actor_npz,
    load_actor_npz,
    validate_actor_arrays,
    write_actor_npz_exclusive,
)


def _actor(seed: int) -> Actor:
    observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(OBSERVATION_WIDTH,),
        dtype=np.float64,
    )
    action_space = gym.spaces.Box(
        low=np.full(ACTION_WIDTH, -0.4, dtype=np.float32),
        high=np.full(ACTION_WIDTH, 0.4, dtype=np.float32),
        dtype=np.float32,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        extractor = FlattenExtractor(observation_space)
        actor = Actor(
            observation_space=observation_space,
            action_space=action_space,
            net_arch=[256, 256],
            features_extractor=extractor,
            features_dim=extractor.features_dim,
            activation_fn=torch.nn.ReLU,
            use_sde=False,
        )
    actor.eval()
    return actor


def _arrays(actor: Actor) -> dict[str, np.ndarray]:
    values = {
        name: np.ascontiguousarray(tensor.detach().cpu().numpy(), dtype="<f4")
        for name, tensor in actor.state_dict().items()
    }
    values.update(
        {
            "action_low": np.full(ACTION_WIDTH, -0.4, dtype="<f4"),
            "action_high": np.full(ACTION_WIDTH, 0.4, dtype="<f4"),
            "format_version": np.asarray([1], dtype="<i8"),
        }
    )
    return values


def _canonical_member(name: str) -> ZipInfo:
    member = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    member.compress_type = ZIP_DEFLATED
    member.create_system = 3
    member.external_attr = 0o100600 << 16
    return member


def test_schema_matches_the_pinned_sb3_tqc_actor() -> None:
    values = validate_actor_arrays(_arrays(_actor(95001)))

    assert tuple(values) == tuple(item["name"] for item in actor_schema()["members_in_order"])
    assert actor_schema()["contains_executable_code"] is False
    assert actor_schema()["exact_training_resume"] is False
    assert len(actor_schema_sha256()) == 64


def test_actor_archive_is_deterministic_and_preserves_outputs(tmp_path: Path) -> None:
    source = _actor(95001)
    arrays = _arrays(source)
    first = encode_actor_npz(arrays)
    second = encode_actor_npz(arrays)
    assert first == second
    assert len(first) <= MAX_ARCHIVE_BYTES

    path = tmp_path / "actor.npz"
    content_sha256 = write_actor_npz_exclusive(path, arrays)
    loaded = load_actor_npz(path, expected_sha256=content_sha256)
    assert loaded.content_sha256 == hashlib.sha256(first).hexdigest()
    assert loaded.state_sha256 == actor_state_sha256(arrays)
    assert loaded.schema_sha256 == actor_schema_sha256()
    with pytest.raises(TypeError):
        loaded.arrays["unexpected"] = np.zeros(1, dtype="<f4")
    with pytest.raises(ValueError):
        loaded.arrays["mu.bias"][0] = np.float32(1.0)
    with pytest.raises(ValueError):
        loaded.arrays["mu.bias"].setflags(write=True)

    restored = _actor(1)
    state = {
        name: torch.from_numpy(value.copy())
        for name, value in loaded.arrays.items()
        if name not in {"action_low", "action_high", "format_version"}
    }
    restored.load_state_dict(state, strict=True)
    observations = torch.linspace(
        -2.0,
        2.0,
        steps=4 * OBSERVATION_WIDTH,
        dtype=torch.float32,
    ).reshape(4, OBSERVATION_WIDTH)
    with torch.inference_mode():
        source_mean, source_log_std, _ = source.get_action_dist_params(observations)
        restored_mean, restored_log_std, _ = restored.get_action_dist_params(observations)
        source_action = source(observations, deterministic=True)
        restored_action = restored(observations, deterministic=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(96001)
            source_sample = source(observations, deterministic=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(96001)
            restored_sample = restored(observations, deterministic=False)
    assert torch.equal(source_mean, restored_mean)
    assert torch.equal(source_log_std, restored_log_std)
    assert torch.equal(source_action, restored_action)
    assert torch.equal(source_sample, restored_sample)

    with pytest.raises(ExperimentContractError, match="only be issued by load_actor_npz"):
        replace(loaded)


def test_loaded_actor_record_cannot_be_minted_from_arrays() -> None:
    arrays = validate_actor_arrays(_arrays(_actor(95001)))
    payload = encode_actor_npz(arrays)

    with pytest.raises(ExperimentContractError, match="only be issued by load_actor_npz"):
        LoadedTQCActor(
            content_sha256=hashlib.sha256(payload).hexdigest(),
            state_sha256=actor_state_sha256(arrays),
            schema_sha256=actor_schema_sha256(),
            byte_count=len(payload),
            arrays=arrays,
        )


@pytest.mark.parametrize(
    "mutation, error",
    [
        (lambda values: values.pop("mu.bias"), "keys differ"),
        (
            lambda values: values.update({"mu.bias": np.zeros((ACTION_WIDTH + 1,), dtype="<f4")}),
            "wrong shape",
        ),
        (
            lambda values: values.update({"mu.bias": np.zeros(ACTION_WIDTH, dtype="<f8")}),
            "wrong dtype",
        ),
        (
            lambda values: values["mu.bias"].__setitem__(0, np.nan),
            "non-finite",
        ),
        (
            lambda values: values["action_high"].__setitem__(0, np.float32(1.0)),
            "action bounds",
        ),
        (
            lambda values: values["format_version"].__setitem__(0, 2),
            "format_version",
        ),
    ],
)
def test_actor_schema_rejects_drift(mutation: object, error: str) -> None:
    values = _arrays(_actor(95001))
    mutation(values)

    with pytest.raises(ExperimentContractError, match=error):
        validate_actor_arrays(values)


def test_loader_rejects_hash_mismatch_trailing_bytes_and_overwrite(tmp_path: Path) -> None:
    values = _arrays(_actor(95001))
    path = tmp_path / "actor.npz"
    write_actor_npz_exclusive(path, values)

    with pytest.raises(ExperimentContractError, match="SHA-256"):
        load_actor_npz(path, expected_sha256="0" * 64)
    with pytest.raises(ExperimentContractError, match="overwrite"):
        write_actor_npz_exclusive(path, values)

    trailing = tmp_path / "trailing.npz"
    trailing.write_bytes(path.read_bytes() + b"not-archive")
    trailing_digest = hashlib.sha256(trailing.read_bytes()).hexdigest()
    with pytest.raises(ExperimentContractError, match="un-commented ZIP"):
        load_actor_npz(trailing, expected_sha256=trailing_digest)


def test_loader_rejects_noncanonical_local_header_bytes(tmp_path: Path) -> None:
    payload = bytearray(encode_actor_npz(_arrays(_actor(95001))))
    assert payload[:4] == b"PK\x03\x04"
    payload[10] ^= 1  # Local-header time; the central directory remains canonical.
    path = tmp_path / "local-header-drift.npz"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    with pytest.raises(ExperimentContractError, match="archive bytes are not canonical"):
        load_actor_npz(path, expected_sha256=digest)


def test_loader_rejects_reordered_members_and_pickle_dtype(tmp_path: Path) -> None:
    values = _arrays(_actor(95001))
    payload = encode_actor_npz(values)
    with ZipFile(io.BytesIO(payload), "r") as source:
        members = [(info.filename, source.read(info)) for info in source.infolist()]

    reordered = tmp_path / "reordered.npz"
    with ZipFile(reordered, "w", compression=ZIP_DEFLATED) as archive:
        for name, member in reversed(members):
            archive.writestr(_canonical_member(name), member)
    digest = hashlib.sha256(reordered.read_bytes()).hexdigest()
    with pytest.raises(ExperimentContractError, match="member order"):
        load_actor_npz(reordered, expected_sha256=digest)

    unsafe = tmp_path / "unsafe.npz"
    unsafe_members = list(members)
    object_stream = io.BytesIO()
    np.lib.format.write_array(
        object_stream,
        np.asarray([object()], dtype=object),
        version=(1, 0),
        allow_pickle=True,
    )
    unsafe_members[0] = (unsafe_members[0][0], object_stream.getvalue())
    with ZipFile(unsafe, "w", compression=ZIP_DEFLATED) as archive:
        for name, member in unsafe_members:
            archive.writestr(_canonical_member(name), member)
    digest = hashlib.sha256(unsafe.read_bytes()).hexdigest()
    with pytest.raises(ExperimentContractError, match="unsafe object dtype"):
        load_actor_npz(unsafe, expected_sha256=digest)


@pytest.mark.parametrize("metadata", ["extra", "comment", "mode", "timestamp", "compression"])
def test_loader_rejects_hidden_member_metadata(tmp_path: Path, metadata: str) -> None:
    payload = encode_actor_npz(_arrays(_actor(95001)))
    with ZipFile(io.BytesIO(payload), "r") as source:
        members = [(info.filename, source.read(info)) for info in source.infolist()]

    path = tmp_path / f"hidden-{metadata}.npz"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for index, (name, member_payload) in enumerate(members):
            info = _canonical_member(name)
            if index == 0:
                if metadata == "extra":
                    info.extra = b"\xfe\xca\x00\x00"
                elif metadata == "comment":
                    info.comment = b"hidden"
                elif metadata == "mode":
                    info.external_attr = 0o100644 << 16
                elif metadata == "timestamp":
                    info.date_time = (1981, 1, 1, 0, 0, 0)
                else:
                    info.compress_type = 0
            archive.writestr(info, member_payload)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    with pytest.raises(ExperimentContractError, match="noncanonical ZIP metadata"):
        load_actor_npz(path, expected_sha256=digest)


@pytest.mark.parametrize("mutation", ["npy_version", "fortran_order"])
def test_loader_rejects_noncanonical_npy_layout(tmp_path: Path, mutation: str) -> None:
    values = _arrays(_actor(95001))
    payload = encode_actor_npz(values)
    with ZipFile(io.BytesIO(payload), "r") as source:
        members = [(info.filename, source.read(info)) for info in source.infolist()]

    stream = io.BytesIO()
    first_name = next(iter(values))
    first_array = values[first_name]
    if mutation == "npy_version":
        np.lib.format.write_array(stream, first_array, version=(2, 0), allow_pickle=False)
    else:
        np.lib.format.write_array(
            stream,
            np.asfortranarray(first_array),
            version=(1, 0),
            allow_pickle=False,
        )
    members[0] = (members[0][0], stream.getvalue())

    path = tmp_path / f"{mutation}.npz"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, member_payload in members:
            archive.writestr(_canonical_member(name), member_payload)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    expected = "NPY 1.0" if mutation == "npy_version" else "C-order"
    with pytest.raises(ExperimentContractError, match=expected):
        load_actor_npz(path, expected_sha256=digest)


def test_loader_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.npz"
    digest = write_actor_npz_exclusive(target, _arrays(_actor(95001)))
    link = tmp_path / "link.npz"
    link.symlink_to(target)

    with pytest.raises(ExperimentContractError, match="symlink"):
        load_actor_npz(link, expected_sha256=digest)


def test_loader_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "actor.npz"
    digest = write_actor_npz_exclusive(target, _arrays(_actor(95001)))
    link = tmp_path / "linked"
    link.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ExperimentContractError, match="ancestors"):
        load_actor_npz(link / target.name, expected_sha256=digest)


def test_rejected_nonregular_actor_does_not_leak_descriptors(tmp_path: Path) -> None:
    descriptor_directory = Path("/dev/fd")
    if not descriptor_directory.is_dir():
        descriptor_directory = Path("/proc/self/fd")
    if not descriptor_directory.is_dir():
        pytest.skip("descriptor inventory is unavailable")
    before = len(os.listdir(descriptor_directory))

    for _ in range(50):
        with pytest.raises(ExperimentContractError, match="bounded regular file"):
            load_actor_npz(tmp_path, expected_sha256="0" * 64)

    assert len(os.listdir(descriptor_directory)) == before
