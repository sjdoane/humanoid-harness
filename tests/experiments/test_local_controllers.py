from __future__ import annotations

import hashlib
import io
import warnings
from dataclasses import FrozenInstanceError
from pathlib import Path
from zipfile import ZIP_BZIP2, ZIP_DEFLATED, ZipFile

import numpy as np
import pytest
import torch

from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.local_controllers import (
    ACTION_WIDTH,
    LOCAL_BEHAVIOR_CLONING_SHA256,
    LOCAL_CONTROLLER_CLAIM_STATUS,
    LOCAL_REFERENCE_RESIDUAL_SHA256,
    MAX_ARCHIVE_BYTES,
    MAX_MEMBER_BYTES,
    OBSERVATION_WIDTH,
    REFERENCE_HORIZON_STEPS,
    REFERENCE_WIDTH,
    RESIDUAL_SCALE,
    load_local_behavior_cloning_controller,
    load_local_reference_residual_controller,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bc_values() -> dict[str, np.ndarray]:
    values = {
        "obs_mean": np.zeros(348, dtype="<f4"),
        "obs_std": np.ones(348, dtype="<f4"),
        "net.0.weight": np.zeros((512, 348), dtype="<f4"),
        "net.0.bias": np.zeros(512, dtype="<f4"),
        "net.2.weight": np.zeros((512, 512), dtype="<f4"),
        "net.2.bias": np.zeros(512, dtype="<f4"),
        "net.4.weight": np.zeros((512, 512), dtype="<f4"),
        "net.4.bias": np.zeros(512, dtype="<f4"),
        "net.6.weight": np.zeros((17, 512), dtype="<f4"),
        "net.6.bias": np.linspace(-0.2, 0.2, 17, dtype="<f4"),
    }
    return values


def _residual_values() -> dict[str, np.ndarray]:
    conditioned_width = 348 + 8 * 45
    return {
        "policy::log_std": np.full(17, -4.0, dtype="<f4"),
        "policy::mlp_extractor.policy_net.0.weight": np.zeros(
            (256, conditioned_width), dtype="<f4"
        ),
        "policy::mlp_extractor.policy_net.0.bias": np.zeros(256, dtype="<f4"),
        "policy::mlp_extractor.policy_net.2.weight": np.zeros((256, 256), dtype="<f4"),
        "policy::mlp_extractor.policy_net.2.bias": np.zeros(256, dtype="<f4"),
        "policy::mlp_extractor.value_net.0.weight": np.zeros((256, conditioned_width), dtype="<f4"),
        "policy::mlp_extractor.value_net.0.bias": np.zeros(256, dtype="<f4"),
        "policy::mlp_extractor.value_net.2.weight": np.zeros((256, 256), dtype="<f4"),
        "policy::mlp_extractor.value_net.2.bias": np.zeros(256, dtype="<f4"),
        "policy::action_net.weight": np.zeros((17, 256), dtype="<f4"),
        "policy::action_net.bias": np.linspace(-0.1, 0.1, 17, dtype="<f4"),
        "policy::value_net.weight": np.zeros((1, 256), dtype="<f4"),
        "policy::value_net.bias": np.zeros(1, dtype="<f4"),
        "obs_rms_mean": np.zeros(conditioned_width, dtype="<f8"),
        "obs_rms_var": np.ones(conditioned_width, dtype="<f8"),
        "obs_rms_count": np.asarray([100.0], dtype="<f8"),
        "clip_obs": np.asarray([10.0], dtype="<f8"),
        "residual_scale": np.asarray([0.08], dtype="<f8"),
        "horizon": np.asarray([8], dtype="<i8"),
    }


def _write_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **values)


def _npy_bytes(value: np.ndarray, *, allow_pickle: bool = False) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, value, version=(1, 0), allow_pickle=allow_pickle)
    return stream.getvalue()


def _load_bc(path: Path):
    return load_local_behavior_cloning_controller(path, expected_sha256=_sha256(path))


def _load_residual(path: Path):
    return load_local_reference_residual_controller(path, expected_sha256=_sha256(path))


def test_loaders_reconstruct_deterministic_inference_and_immutable_receipts(
    tmp_path: Path,
) -> None:
    bc_path = tmp_path / "bc.npz"
    residual_path = tmp_path / "residual.npz"
    bc_values = _bc_values()
    residual_values = _residual_values()
    _write_npz(bc_path, bc_values)
    _write_npz(residual_path, residual_values)

    torch.manual_seed(817)
    rng_before = torch.random.get_rng_state().clone()
    bc_infer, bc_receipt = _load_bc(bc_path)
    residual_infer, residual_receipt = _load_residual(residual_path)
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before)

    observation = np.zeros(OBSERVATION_WIDTH, dtype=np.float64)
    window = np.zeros((REFERENCE_HORIZON_STEPS, REFERENCE_WIDTH), dtype=np.float64)
    expected_bc = (0.4 * torch.tanh(torch.from_numpy(bc_values["net.6.bias"]))).numpy()
    expected_residual = residual_values["policy::action_net.bias"].astype(np.float64)
    np.testing.assert_array_equal(bc_infer(observation), expected_bc)
    np.testing.assert_array_equal(bc_infer(observation), bc_infer(observation))
    np.testing.assert_array_equal(residual_infer(observation, window), expected_residual)
    np.testing.assert_array_equal(
        residual_infer(observation, window),
        residual_infer(observation, window),
    )

    assert bc_receipt.claim_status == LOCAL_CONTROLLER_CLAIM_STATUS
    assert bc_receipt.filename == "bc.npz"
    assert bc_receipt.content_sha256 == _sha256(bc_path)
    assert bc_receipt.file_size_bytes == bc_path.stat().st_size
    assert bc_receipt.registered_content_match is False
    assert bc_receipt.reference_horizon_steps is None
    assert bc_receipt.inference_output_contract == (
        "float32_raw_Humanoid-v5_control_with_action_space_endpoint_bytes"
    )
    assert residual_receipt.reference_horizon_steps == 8
    assert residual_receipt.reference_width == 45
    assert residual_receipt.residual_scale == 0.08
    assert residual_receipt.inference_output_contract.startswith("unscaled_dimensionless_action")
    assert len(residual_receipt.loader_source_sha256) == 64
    assert len(residual_receipt.sha256()) == 64
    assert residual_receipt.sha256() == residual_receipt.sha256()
    assert all(
        residual_receipt.to_dict()[field]
        for field in (
            "numpy_version",
            "torch_version",
            "gymnasium_version",
            "stable_baselines3_version",
        )
    )
    assert {"path", "file_device", "file_inode", "file_mtime_ns"}.isdisjoint(
        residual_receipt.to_dict()
    )
    with pytest.raises(FrozenInstanceError):
        bc_receipt.claim_status = "admitted"  # type: ignore[misc]


def test_bc_inference_uses_normalizer_relu_and_every_hidden_layer(tmp_path: Path) -> None:
    path = tmp_path / "sentinel-bc.npz"
    values = _bc_values()
    values["net.6.bias"].fill(0.0)
    values["obs_mean"][3] = -0.5
    values["obs_std"][3] = 2.0
    values["obs_mean"][4] = 0.5
    values["net.0.weight"][7, 3] = 0.5
    values["net.0.weight"][8, 4] = 0.5
    values["net.2.weight"][9, 7] = 0.5
    values["net.2.weight"][9, 8] = 0.5
    values["net.4.weight"][11, 9] = 0.5
    values["net.6.weight"][2, 11] = 0.5
    _write_npz(path, values)

    infer, _receipt = _load_bc(path)
    observed = infer(np.zeros(OBSERVATION_WIDTH, dtype=np.float64))
    expected_signal = float(0.4 * torch.tanh(torch.tensor(0.015625, dtype=torch.float32)))

    assert observed[2] == expected_signal
    np.testing.assert_array_equal(np.delete(observed, 2), np.zeros(ACTION_WIDTH - 1))


def test_residual_inference_uses_c_order_window_normalization_and_hidden_path(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sentinel-residual.npz"
    values = _residual_values()
    values["policy::action_net.bias"].fill(0.0)
    combined_index = OBSERVATION_WIDTH + 2 * REFERENCE_WIDTH + 4
    values["obs_rms_mean"][combined_index] = 0.5
    values["obs_rms_var"][combined_index] = 4.0
    values["policy::mlp_extractor.policy_net.0.weight"][7, combined_index] = 0.2
    values["policy::mlp_extractor.policy_net.0.bias"][7] = -0.1
    values["policy::mlp_extractor.policy_net.2.weight"][9, 7] = 0.5
    values["policy::action_net.weight"][2, 9] = 0.4
    values["policy::mlp_extractor.value_net.0.weight"][12, combined_index] = 0.3
    values["policy::mlp_extractor.value_net.0.bias"][12] = -0.05
    values["policy::mlp_extractor.value_net.2.weight"][13, 12] = 0.4
    values["policy::value_net.weight"][0, 13] = 0.5
    values["policy::value_net.bias"][0] = 0.1
    _write_npz(path, values)

    infer, receipt = _load_residual(path)
    observation = np.zeros(OBSERVATION_WIDTH, dtype=np.float64)
    window = np.zeros((REFERENCE_HORIZON_STEPS, REFERENCE_WIDTH), dtype=np.float64)
    window[2, 4] = 2.5
    observed = infer(observation, window)
    facts = infer.inference_facts(observation, window)
    normalized = np.float32((2.5 - 0.5) / np.sqrt(4.0 + 1e-8))
    first = max(np.float32(0.0), np.float32(0.2) * normalized + np.float32(-0.1))
    second = max(np.float32(0.0), np.float32(0.5) * first)
    expected_signal = np.float32(0.4) * second
    value_first = max(np.float32(0.0), np.float32(0.3) * normalized + np.float32(-0.05))
    value_second = max(np.float32(0.0), np.float32(0.4) * value_first)
    expected_value = np.float32(0.5) * value_second + np.float32(0.1)

    assert observed[2] == pytest.approx(float(expected_signal), rel=0.0, abs=1e-8)
    np.testing.assert_array_equal(np.delete(observed, 2), np.zeros(ACTION_WIDTH - 1))
    assert np.all(observed >= -1.0)
    assert np.all(observed <= 1.0)
    assert facts.critic_value == pytest.approx(float(expected_value), rel=0.0, abs=1e-8)
    assert facts.policy_input_shape == (708,)
    assert facts.policy_input_dtype == "<f4"
    assert facts.actor_input_sha256 == facts.policy_input_sha256
    assert facts.critic_input_sha256 == facts.policy_input_sha256
    assert facts.actor_output_shape == (ACTION_WIDTH,)
    assert facts.actor_output_dtype == "<f4"
    assert len(facts.actor_output_sha256) == 64
    assert facts.critic_output_shape == (1, 1)
    assert facts.critic_output_dtype == "<f4"
    assert len(facts.critic_output_sha256) == 64
    assert len(facts.controller_state_sha256) == 64
    assert facts == infer.inference_facts(observation, window)
    assert receipt.residual_scale == RESIDUAL_SCALE


def test_residual_policy_input_is_exact_contiguous_float32_and_window_sensitive(
    tmp_path: Path,
) -> None:
    path = tmp_path / "residual.npz"
    values = _residual_values()
    values["obs_rms_mean"][0] = 1.0
    values["obs_rms_var"][0] = 4.0
    values["clip_obs"][0] = 0.25
    _write_npz(path, values)

    infer, _receipt = _load_residual(path)
    observation = np.zeros(OBSERVATION_WIDTH, dtype=np.float64)
    window_a = np.zeros((REFERENCE_HORIZON_STEPS, REFERENCE_WIDTH), dtype=np.float64)
    window_b = window_a.copy()
    window_b[-1, -1] = 1.0

    policy_input = infer.policy_input(observation, window_a)
    facts_a = infer.inference_facts(observation, window_a)
    facts_b = infer.inference_facts(observation, window_b)

    assert policy_input.dtype.str == "<f4"
    assert policy_input.flags.c_contiguous
    assert policy_input.shape == (708,)
    assert policy_input[0] == np.float32(-0.25)
    assert np.max(np.abs(policy_input)) <= np.float32(0.25)
    assert facts_a.policy_input_sha256 != facts_b.policy_input_sha256


def test_residual_controller_detects_loaded_state_mutation(tmp_path: Path) -> None:
    path = tmp_path / "residual.npz"
    _write_npz(path, _residual_values())
    infer, _receipt = _load_residual(path)
    observation = np.zeros(OBSERVATION_WIDTH, dtype=np.float64)
    window = np.zeros((REFERENCE_HORIZON_STEPS, REFERENCE_WIDTH), dtype=np.float64)

    assert not infer._mean.flags.writeable
    with pytest.raises(ValueError):
        infer._mean.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        infer._clip = 0.01  # type: ignore[misc]

    with torch.no_grad():
        infer._policy.action_net.bias[0].add_(1.0)
    with pytest.raises(ExperimentContractError, match="state changed"):
        infer.inference_facts(observation, window)


def test_residual_callable_remains_actor_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "residual.npz"
    _write_npz(path, _residual_values())
    infer, _receipt = _load_residual(path)

    def critic_must_not_run(_observation: torch.Tensor) -> torch.Tensor:
        raise AssertionError("ordinary action inference invoked the critic")

    monkeypatch.setattr(infer._policy, "predict_values", critic_must_not_run)
    action = infer(
        np.zeros(OBSERVATION_WIDTH, dtype=np.float64),
        np.zeros((REFERENCE_HORIZON_STEPS, REFERENCE_WIDTH), dtype=np.float64),
    )
    assert action.shape == (ACTION_WIDTH,)


@pytest.mark.parametrize("expected", ["0" * 63, "A" * 64, "z" * 64, ""])
def test_loader_rejects_noncanonical_expected_hash(tmp_path: Path, expected: str) -> None:
    path = tmp_path / "controller.npz"
    _write_npz(path, _bc_values())
    with pytest.raises(ExperimentContractError, match="expected_sha256"):
        load_local_behavior_cloning_controller(path, expected_sha256=expected)


def test_loader_rejects_hash_mismatch_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "controller.npz"
    path.write_bytes(b"not an archive")
    with pytest.raises(ExperimentContractError, match="SHA-256"):
        load_local_behavior_cloning_controller(path, expected_sha256="0" * 64)


def test_registered_hash_is_the_default_trust_anchor(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.npz"
    _write_npz(path, _bc_values())
    with pytest.raises(ExperimentContractError, match="SHA-256"):
        load_local_behavior_cloning_controller(path)


def test_loader_rejects_trailing_archive_bytes(tmp_path: Path) -> None:
    path = tmp_path / "trailing.npz"
    _write_npz(path, _bc_values())
    path.write_bytes(path.read_bytes() + b"unbound trailer")
    with pytest.raises(ExperimentContractError, match="must end"):
        _load_bc(path)


def test_loader_rejects_symlink_directory_and_oversize_file(tmp_path: Path) -> None:
    target = tmp_path / "target.npz"
    _write_npz(target, _bc_values())
    linked = tmp_path / "linked.npz"
    linked.symlink_to(target)
    with pytest.raises(ExperimentContractError, match="symlink"):
        load_local_behavior_cloning_controller(linked, expected_sha256=_sha256(target))

    with pytest.raises(ExperimentContractError, match="regular file"):
        load_local_behavior_cloning_controller(tmp_path, expected_sha256="0" * 64)

    oversized = tmp_path / "oversized.npz"
    with oversized.open("wb") as stream:
        stream.truncate(MAX_ARCHIVE_BYTES + 1)
    with pytest.raises(ExperimentContractError, match="between 1"):
        load_local_behavior_cloning_controller(oversized, expected_sha256=_sha256(oversized))


def test_loader_rejects_traversal_member(tmp_path: Path) -> None:
    path = tmp_path / "traversal.npz"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("../obs_mean.npy", _npy_bytes(np.zeros(348, dtype="<f4")))
    with pytest.raises(ExperimentContractError, match="traversal"):
        _load_bc(path)


def test_loader_rejects_duplicate_member(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.npz"
    member = _npy_bytes(np.zeros(348, dtype="<f4"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("obs_mean.npy", member)
            archive.writestr("obs_mean.npy", member)
    with pytest.raises(ExperimentContractError, match="duplicate"):
        _load_bc(path)


def test_loader_rejects_encrypted_member_flag(tmp_path: Path) -> None:
    path = tmp_path / "encrypted.npz"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("obs_mean.npy", _npy_bytes(np.zeros(348, dtype="<f4")))
    payload = bytearray(path.read_bytes())
    central = payload.index(b"PK\x01\x02")
    flags = int.from_bytes(payload[central + 8 : central + 10], "little") | 0x1
    payload[central + 8 : central + 10] = flags.to_bytes(2, "little")
    path.write_bytes(payload)
    with pytest.raises(ExperimentContractError, match="encrypted"):
        _load_bc(path)


def test_loader_rejects_unsupported_compression(tmp_path: Path) -> None:
    path = tmp_path / "bzip2.npz"
    with ZipFile(path, "w", compression=ZIP_BZIP2) as archive:
        archive.writestr("obs_mean.npy", _npy_bytes(np.zeros(348, dtype="<f4")))
    with pytest.raises(ExperimentContractError, match="unsupported compression"):
        _load_bc(path)


def test_loader_rejects_oversized_member_before_expansion(tmp_path: Path) -> None:
    path = tmp_path / "large-member.npz"
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("obs_mean.npy", b"0" * (MAX_MEMBER_BYTES + 1))
    with pytest.raises(ExperimentContractError, match="size bound"):
        _load_bc(path)


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_loader_rejects_key_set_changes(tmp_path: Path, mutation: str) -> None:
    path = tmp_path / "keys.npz"
    values = _bc_values()
    if mutation == "missing":
        del values["net.6.bias"]
    else:
        values["unexpected"] = np.zeros(1, dtype="<f4")
    _write_npz(path, values)
    with pytest.raises(ExperimentContractError, match="keys differ"):
        _load_bc(path)


@pytest.mark.parametrize("mutation", ["shape", "dtype", "nonfinite", "fortran"])
def test_loader_rejects_invalid_array_contract(tmp_path: Path, mutation: str) -> None:
    path = tmp_path / "invalid-array.npz"
    values = _bc_values()
    if mutation == "shape":
        values["obs_mean"] = np.zeros(347, dtype="<f4")
    elif mutation == "dtype":
        values["obs_mean"] = np.zeros(348, dtype="<f8")
    elif mutation == "nonfinite":
        values["obs_mean"][0] = np.nan
    else:
        values["net.0.weight"] = np.asfortranarray(values["net.0.weight"])
    _write_npz(path, values)
    expected = {
        "shape": "wrong shape",
        "dtype": "wrong dtype",
        "nonfinite": "non-finite",
        "fortran": "C-order",
    }[mutation]
    with pytest.raises(ExperimentContractError, match=expected):
        _load_bc(path)


def test_loader_rejects_pickle_backed_array_without_unpickling(tmp_path: Path) -> None:
    path = tmp_path / "pickle.npz"
    values = _bc_values()
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for key, value in values.items():
            member = (
                _npy_bytes(np.asarray(["untrusted"] * 348, dtype=object), allow_pickle=True)
                if key == "obs_mean"
                else _npy_bytes(value)
            )
            archive.writestr(f"{key}.npy", member)
    with pytest.raises(ExperimentContractError, match="unsafe pickle"):
        _load_bc(path)


def test_bc_loader_requires_positive_standard_deviation(tmp_path: Path) -> None:
    path = tmp_path / "zero-std.npz"
    values = _bc_values()
    values["obs_std"][0] = 0.0
    _write_npz(path, values)
    with pytest.raises(ExperimentContractError, match="obs_std"):
        _load_bc(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("obs_rms_var", 0.0, "obs_rms_var"),
        ("obs_rms_count", 0.0, "obs_rms_count"),
        ("clip_obs", 0.0, "clip_obs"),
        ("horizon", 7, "horizon must equal 8"),
        ("residual_scale", 0.081, "residual_scale must equal 0.08"),
    ],
)
def test_residual_loader_rejects_semantic_contract_changes(
    tmp_path: Path,
    field: str,
    value: float,
    message: str,
) -> None:
    path = tmp_path / "invalid-residual.npz"
    values = _residual_values()
    values[field].flat[0] = value
    _write_npz(path, values)
    with pytest.raises(ExperimentContractError, match=message):
        _load_residual(path)


@pytest.mark.parametrize(
    ("kind", "value", "message"),
    [
        ("bc", np.zeros(347), "shape"),
        ("bc", np.full(348, np.inf), "finite"),
        ("residual_observation", np.zeros(347), "shape"),
        ("residual_window", np.zeros((7, 45)), "shape"),
        ("residual_window", np.full((8, 45), np.nan), "finite"),
    ],
)
def test_inference_rejects_invalid_runtime_inputs(
    tmp_path: Path,
    kind: str,
    value: np.ndarray,
    message: str,
) -> None:
    if kind == "bc":
        path = tmp_path / "bc.npz"
        _write_npz(path, _bc_values())
        infer, _receipt = _load_bc(path)
        with pytest.raises(ExperimentContractError, match=message):
            infer(value)
        return

    path = tmp_path / "residual.npz"
    _write_npz(path, _residual_values())
    infer, _receipt = _load_residual(path)
    observation = np.zeros(OBSERVATION_WIDTH)
    window = np.zeros((REFERENCE_HORIZON_STEPS, REFERENCE_WIDTH))
    with pytest.raises(ExperimentContractError, match=message):
        if kind == "residual_observation":
            infer(value, window)
        else:
            infer(observation, value)


def test_public_dimensions_and_scale_are_frozen() -> None:
    assert (
        LOCAL_BEHAVIOR_CLONING_SHA256
        == "ce2aa3a1358609f09509d7f352475a7b517c6d11858ff76419b18a187cb3adf3"
    )
    assert (
        LOCAL_REFERENCE_RESIDUAL_SHA256
        == "6916bf6778dd3044bca5feae22897b7d582e7389871a728549f791112d90fc22"
    )
    assert OBSERVATION_WIDTH == 348
    assert ACTION_WIDTH == 17
    assert REFERENCE_HORIZON_STEPS == 8
    assert REFERENCE_WIDTH == 45
    assert RESIDUAL_SCALE == 0.08
