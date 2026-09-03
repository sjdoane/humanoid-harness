"""Strict loaders for the non-admitted local exploration controllers."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile, ZipInfo

import gymnasium as gym
import numpy as np
import stable_baselines3
import torch
from stable_baselines3.common.policies import ActorCriticPolicy

from .fixed_reference import ExperimentContractError

LOCAL_BEHAVIOR_CLONING_SHA256 = "ce2aa3a1358609f09509d7f352475a7b517c6d11858ff76419b18a187cb3adf3"
LOCAL_REFERENCE_RESIDUAL_SHA256 = "6916bf6778dd3044bca5feae22897b7d582e7389871a728549f791112d90fc22"
LOCAL_CONTROLLER_CLAIM_STATUS = "local_exploration_only_not_admitted"

OBSERVATION_WIDTH = 348
ACTION_WIDTH = 17
REFERENCE_WIDTH = 45
REFERENCE_HORIZON_STEPS = 8
REFERENCE_CONDITIONED_WIDTH = OBSERVATION_WIDTH + REFERENCE_HORIZON_STEPS * REFERENCE_WIDTH
RESIDUAL_SCALE = 0.08
BASE_ACTION_SCALE = 0.4

MAX_ARCHIVE_BYTES = 8 * 1024 * 1024
MAX_MEMBER_BYTES = 2 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 8 * 1024 * 1024
MAX_MEMBERS = 32

_FLOAT32 = np.dtype("<f4")
_FLOAT64 = np.dtype("<f8")
_INT64 = np.dtype("<i8")
_ALLOWED_COMPRESSION = frozenset((ZIP_STORED, ZIP_DEFLATED))

_BC_SCHEMA: Mapping[str, tuple[tuple[int, ...], np.dtype[object]]] = {
    "obs_mean": ((OBSERVATION_WIDTH,), _FLOAT32),
    "obs_std": ((OBSERVATION_WIDTH,), _FLOAT32),
    "net.0.weight": ((512, OBSERVATION_WIDTH), _FLOAT32),
    "net.0.bias": ((512,), _FLOAT32),
    "net.2.weight": ((512, 512), _FLOAT32),
    "net.2.bias": ((512,), _FLOAT32),
    "net.4.weight": ((512, 512), _FLOAT32),
    "net.4.bias": ((512,), _FLOAT32),
    "net.6.weight": ((ACTION_WIDTH, 512), _FLOAT32),
    "net.6.bias": ((ACTION_WIDTH,), _FLOAT32),
}

_RESIDUAL_SCHEMA: Mapping[str, tuple[tuple[int, ...], np.dtype[object]]] = {
    "policy::log_std": ((ACTION_WIDTH,), _FLOAT32),
    "policy::mlp_extractor.policy_net.0.weight": ((256, REFERENCE_CONDITIONED_WIDTH), _FLOAT32),
    "policy::mlp_extractor.policy_net.0.bias": ((256,), _FLOAT32),
    "policy::mlp_extractor.policy_net.2.weight": ((256, 256), _FLOAT32),
    "policy::mlp_extractor.policy_net.2.bias": ((256,), _FLOAT32),
    "policy::mlp_extractor.value_net.0.weight": ((256, REFERENCE_CONDITIONED_WIDTH), _FLOAT32),
    "policy::mlp_extractor.value_net.0.bias": ((256,), _FLOAT32),
    "policy::mlp_extractor.value_net.2.weight": ((256, 256), _FLOAT32),
    "policy::mlp_extractor.value_net.2.bias": ((256,), _FLOAT32),
    "policy::action_net.weight": ((ACTION_WIDTH, 256), _FLOAT32),
    "policy::action_net.bias": ((ACTION_WIDTH,), _FLOAT32),
    "policy::value_net.weight": ((1, 256), _FLOAT32),
    "policy::value_net.bias": ((1,), _FLOAT32),
    "obs_rms_mean": ((REFERENCE_CONDITIONED_WIDTH,), _FLOAT64),
    "obs_rms_var": ((REFERENCE_CONDITIONED_WIDTH,), _FLOAT64),
    "obs_rms_count": ((1,), _FLOAT64),
    "clip_obs": ((1,), _FLOAT64),
    "residual_scale": ((1,), _FLOAT64),
    "horizon": ((1,), _INT64),
}


@dataclass(frozen=True, slots=True)
class LocalControllerReceipt:
    """Immutable identities and claim boundary for one loaded controller."""

    schema_version: int
    artifact_kind: str
    format_id: str
    architecture_id: str
    claim_status: str
    filename: str
    file_size_bytes: int
    expected_content_sha256: str
    content_sha256: str
    registered_content_match: bool
    loader_source_sha256: str
    numpy_version: str
    torch_version: str
    gymnasium_version: str
    stable_baselines3_version: str
    observation_width: int
    action_width: int
    reference_width: int | None
    reference_horizon_steps: int | None
    residual_scale: float | None
    inference_output_contract: str

    def to_dict(self) -> dict[str, object]:
        """Return the stable public receipt without machine-local file metadata."""

        return {
            "schema_version": self.schema_version,
            "artifact_kind": self.artifact_kind,
            "format_id": self.format_id,
            "architecture_id": self.architecture_id,
            "claim_status": self.claim_status,
            "filename": self.filename,
            "file_size_bytes": self.file_size_bytes,
            "expected_content_sha256": self.expected_content_sha256,
            "content_sha256": self.content_sha256,
            "registered_content_match": self.registered_content_match,
            "loader_source_sha256": self.loader_source_sha256,
            "numpy_version": self.numpy_version,
            "torch_version": self.torch_version,
            "gymnasium_version": self.gymnasium_version,
            "stable_baselines3_version": self.stable_baselines3_version,
            "observation_width": self.observation_width,
            "action_width": self.action_width,
            "reference_width": self.reference_width,
            "reference_horizon_steps": self.reference_horizon_steps,
            "residual_scale": self.residual_scale,
            "inference_output_contract": self.inference_output_contract,
        }

    def sha256(self) -> str:
        """Content-address the canonical public receipt."""

        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class LocalBehaviorCloningController(torch.nn.Module):
    """Repository-defined local BC architecture; weights remain data-only."""

    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(OBSERVATION_WIDTH, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, ACTION_WIDTH),
        )
        self.register_buffer("mean", torch.zeros(OBSERVATION_WIDTH, dtype=torch.float32))
        self.register_buffer("std", torch.ones(OBSERVATION_WIDTH, dtype=torch.float32))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = (observation - self.mean) / self.std
        return BASE_ACTION_SCALE * torch.tanh(self.net(normalized))


def _canonical_sha256(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError("expected_sha256 must be 64 lowercase hexadecimal characters")
    return value


def _read_bound_file(path: Path, *, expected_sha256: str) -> tuple[bytes, os.stat_result]:
    expected = _canonical_sha256(expected_sha256)
    candidate = Path(path)
    try:
        link_state = candidate.lstat()
    except OSError as exc:
        raise ExperimentContractError(f"cannot inspect local controller file: {exc}") from exc
    if stat.S_ISLNK(link_state.st_mode):
        raise ExperimentContractError("local controller path must not be a symlink")
    if not stat.S_ISREG(link_state.st_mode):
        raise ExperimentContractError("local controller path must be a regular file")

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise ExperimentContractError(f"cannot open local controller file: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ExperimentContractError("local controller path must be a regular file")
        if (link_state.st_dev, link_state.st_ino) != (before.st_dev, before.st_ino):
            raise ExperimentContractError("local controller file changed while it was opened")
        if not 0 < before.st_size <= MAX_ARCHIVE_BYTES:
            raise ExperimentContractError(
                f"local controller file must be between 1 and {MAX_ARCHIVE_BYTES} bytes"
            )
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            payload = stream.read(MAX_ARCHIVE_BYTES + 1)
        after = os.fstat(descriptor)
    except ExperimentContractError:
        raise
    except OSError as exc:
        raise ExperimentContractError(f"cannot read local controller file: {exc}") from exc
    finally:
        os.close(descriptor)

    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in identity_fields):
        raise ExperimentContractError("local controller file changed while it was read")
    if len(payload) != before.st_size:
        raise ExperimentContractError("local controller file length changed while it was read")
    observed = hashlib.sha256(payload).hexdigest()
    if not hmac.compare_digest(observed, expected):
        raise ExperimentContractError("local controller content SHA-256 does not match")
    return payload, before


def _validate_member(info: ZipInfo) -> None:
    name = info.filename
    if (
        not name
        or name.startswith(("/", "\\"))
        or "/" in name
        or "\\" in name
        or name in {".", ".."}
    ):
        raise ExperimentContractError("controller archive contains a traversal-capable member")
    if info.is_dir():
        raise ExperimentContractError("controller archive must not contain directories")
    member_mode = (info.external_attr >> 16) & 0xFFFF
    if member_mode and stat.S_ISLNK(member_mode):
        raise ExperimentContractError("controller archive must not contain symlinks")
    if info.flag_bits & 0x1:
        raise ExperimentContractError("controller archive must not contain encrypted members")
    if info.compress_type not in _ALLOWED_COMPRESSION:
        raise ExperimentContractError("controller archive uses unsupported compression")
    if not 0 < info.file_size <= MAX_MEMBER_BYTES:
        raise ExperimentContractError("controller archive member exceeds the size bound")


def _parse_npy(
    payload: bytes,
    *,
    key: str,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype[object],
) -> np.ndarray:
    stream = io.BytesIO(payload)
    try:
        version = np.lib.format.read_magic(stream)
        if version != (1, 0):
            raise ExperimentContractError(f"{key} must use the NPY 1.0 format")
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    except (EOFError, TypeError, ValueError) as exc:
        raise ExperimentContractError(f"{key} has an invalid NPY header") from exc
    if dtype.hasobject:
        raise ExperimentContractError(f"{key} contains an unsafe pickle-backed dtype")
    if tuple(shape) != expected_shape:
        raise ExperimentContractError(f"{key} has the wrong shape")
    if dtype != expected_dtype or dtype.str != expected_dtype.str:
        raise ExperimentContractError(f"{key} has the wrong dtype")
    if fortran_order:
        raise ExperimentContractError(f"{key} must use C-order storage")
    value_count = int(np.prod(expected_shape, dtype=np.int64))
    expected_bytes = value_count * expected_dtype.itemsize
    if len(payload) - stream.tell() != expected_bytes:
        raise ExperimentContractError(f"{key} payload length does not match its header")
    values = np.frombuffer(
        payload,
        dtype=expected_dtype,
        count=value_count,
        offset=stream.tell(),
    ).reshape(expected_shape)
    if not np.isfinite(values).all():
        raise ExperimentContractError(f"{key} contains non-finite values")
    return values.copy()


def _read_npz(
    payload: bytes,
    *,
    schema: Mapping[str, tuple[tuple[int, ...], np.dtype[object]]],
) -> dict[str, np.ndarray]:
    if not payload.startswith(b"PK\x03\x04"):
        raise ExperimentContractError("local controller file is not a canonical NPZ archive")
    if len(payload) < 22 or payload[-22:-18] != b"PK\x05\x06" or payload[-2:] != b"\x00\x00":
        raise ExperimentContractError("local controller NPZ must end at an un-commented ZIP record")
    try:
        with ZipFile(io.BytesIO(payload), "r") as archive:
            if archive.comment:
                raise ExperimentContractError("controller archive comments are not permitted")
            infos = archive.infolist()
            if not 0 < len(infos) <= MAX_MEMBERS:
                raise ExperimentContractError("controller archive has an invalid member count")
            names: set[str] = set()
            for info in infos:
                _validate_member(info)
                if info.filename in names:
                    raise ExperimentContractError("controller archive contains duplicate members")
                names.add(info.filename)
            expected_names = {f"{key}.npy" for key in schema}
            missing = sorted(expected_names - names)
            unknown = sorted(names - expected_names)
            if missing or unknown:
                raise ExperimentContractError(
                    f"controller archive keys differ from the schema: missing={missing!r}, "
                    f"unknown={unknown!r}"
                )
            if sum(info.file_size for info in infos) > MAX_UNCOMPRESSED_BYTES:
                raise ExperimentContractError("controller archive exceeds the expansion bound")

            values: dict[str, np.ndarray] = {}
            for info in infos:
                key = info.filename.removesuffix(".npy")
                with archive.open(info, "r") as stream:
                    member = stream.read(info.file_size + 1)
                if len(member) != info.file_size:
                    raise ExperimentContractError("controller archive member length changed")
                shape, dtype = schema[key]
                values[key] = _parse_npy(
                    member,
                    key=key,
                    expected_shape=shape,
                    expected_dtype=dtype,
                )
    except (BadZipFile, EOFError, OSError, RuntimeError) as exc:
        raise ExperimentContractError(f"local controller archive is invalid: {exc}") from exc
    return values


def _numeric_vector(value: object, *, shape: tuple[int, ...], field: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ExperimentContractError(f"{field} must be a numeric array") from exc
    if array.shape != shape:
        raise ExperimentContractError(f"{field} must have shape {shape}")
    if array.dtype.kind not in "fiu":
        raise ExperimentContractError(f"{field} must be a numeric array")
    try:
        converted = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExperimentContractError(f"{field} must be a numeric array") from exc
    if not np.isfinite(converted).all():
        raise ExperimentContractError(f"{field} must contain only finite values")
    return converted


def _loader_source_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _receipt(
    path: Path,
    state: os.stat_result,
    *,
    artifact_kind: str,
    architecture_id: str,
    expected_sha256: str,
    registered_sha256: str,
    reference_width: int | None,
    reference_horizon_steps: int | None,
    residual_scale: float | None,
    inference_output_contract: str,
) -> LocalControllerReceipt:
    return LocalControllerReceipt(
        schema_version=1,
        artifact_kind=artifact_kind,
        format_id="strict_npz_npy1_c_order_no_pickle/v1",
        architecture_id=architecture_id,
        claim_status=LOCAL_CONTROLLER_CLAIM_STATUS,
        filename=Path(path).name,
        file_size_bytes=state.st_size,
        expected_content_sha256=expected_sha256,
        content_sha256=expected_sha256,
        registered_content_match=hmac.compare_digest(expected_sha256, registered_sha256),
        loader_source_sha256=_loader_source_sha256(),
        numpy_version=np.__version__,
        torch_version=str(torch.__version__),
        gymnasium_version=gym.__version__,
        stable_baselines3_version=stable_baselines3.__version__,
        observation_width=OBSERVATION_WIDTH,
        action_width=ACTION_WIDTH,
        reference_width=reference_width,
        reference_horizon_steps=reference_horizon_steps,
        residual_scale=residual_scale,
        inference_output_contract=inference_output_contract,
    )


def load_local_behavior_cloning_controller(
    path: Path,
    *,
    expected_sha256: str = LOCAL_BEHAVIOR_CLONING_SHA256,
) -> tuple[Callable[[np.ndarray], np.ndarray], LocalControllerReceipt]:
    """Load the exact local BC format without importing executable checkpoint state."""

    expected = _canonical_sha256(expected_sha256)
    payload, file_state = _read_bound_file(Path(path), expected_sha256=expected)
    values = _read_npz(payload, schema=_BC_SCHEMA)
    if np.any(values["obs_std"] <= 0.0):
        raise ExperimentContractError("obs_std must be positive")

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        module = LocalBehaviorCloningController()
    state = {
        **{key: torch.from_numpy(value) for key, value in values.items() if key.startswith("net.")},
        "mean": torch.from_numpy(values["obs_mean"]),
        "std": torch.from_numpy(values["obs_std"]),
    }
    module.load_state_dict(state, strict=True)
    module.eval()

    def infer(observation: np.ndarray) -> np.ndarray:
        resolved = _numeric_vector(
            observation,
            shape=(OBSERVATION_WIDTH,),
            field="observation",
        ).astype(np.float32)
        with torch.inference_mode():
            action = module(torch.from_numpy(resolved).unsqueeze(0))
        observed = action.squeeze(0).cpu().numpy().copy()
        if observed.shape != (ACTION_WIDTH,) or not np.isfinite(observed).all():
            raise ExperimentContractError("BC inference returned an invalid action")
        return observed

    receipt = _receipt(
        Path(path),
        file_state,
        artifact_kind="minari_humanoid_behavior_cloning_controller",
        architecture_id="torch_mlp_348_512_512_512_17_relu_action_0.4tanh/v1",
        expected_sha256=expected,
        registered_sha256=LOCAL_BEHAVIOR_CLONING_SHA256,
        reference_width=None,
        reference_horizon_steps=None,
        residual_scale=None,
        inference_output_contract=(
            "float32_raw_Humanoid-v5_control_with_action_space_endpoint_bytes"
        ),
    )
    return infer, receipt


def _build_residual_policy(values: Mapping[str, np.ndarray]) -> ActorCriticPolicy:
    observation_space = gym.spaces.Box(
        -np.inf,
        np.inf,
        shape=(REFERENCE_CONDITIONED_WIDTH,),
        dtype=np.float64,
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(ACTION_WIDTH,), dtype=np.float32)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        policy = ActorCriticPolicy(
            observation_space,
            action_space,
            lambda _: 0.0,
            net_arch={"pi": [256, 256], "vf": [256, 256]},
            activation_fn=torch.nn.ReLU,
            log_std_init=-4.0,
            ortho_init=False,
        )
    policy_state = {
        key.removeprefix("policy::"): torch.from_numpy(value)
        for key, value in values.items()
        if key.startswith("policy::")
    }
    policy.load_state_dict(policy_state, strict=True)
    policy.set_training_mode(False)
    return policy


def load_local_reference_residual_controller(
    path: Path,
    *,
    expected_sha256: str = LOCAL_REFERENCE_RESIDUAL_SHA256,
) -> tuple[Callable[[np.ndarray, np.ndarray], np.ndarray], LocalControllerReceipt]:
    """Load the local residual actor and return its unscaled dimensionless action.

    The caller owns composition: multiply the returned ``[-1, 1]`` residual by
    the receipt's ``residual_scale`` before combining it with the base action.
    """

    expected = _canonical_sha256(expected_sha256)
    payload, file_state = _read_bound_file(Path(path), expected_sha256=expected)
    values = _read_npz(payload, schema=_RESIDUAL_SCHEMA)
    if np.any(values["obs_rms_var"] <= 0.0):
        raise ExperimentContractError("obs_rms_var must be positive")
    if values["obs_rms_count"][0] <= 0.0:
        raise ExperimentContractError("obs_rms_count must be positive")
    if values["clip_obs"][0] <= 0.0:
        raise ExperimentContractError("clip_obs must be positive")
    if not np.array_equal(values["horizon"], np.asarray([REFERENCE_HORIZON_STEPS], dtype="<i8")):
        raise ExperimentContractError("horizon must equal 8")
    if not np.array_equal(values["residual_scale"], np.asarray([RESIDUAL_SCALE], dtype="<f8")):
        raise ExperimentContractError("residual_scale must equal 0.08")

    policy = _build_residual_policy(values)
    mean = values["obs_rms_mean"]
    variance = values["obs_rms_var"]
    clip = float(values["clip_obs"][0])

    def infer(observation: np.ndarray, reference_window: np.ndarray) -> np.ndarray:
        resolved_observation = _numeric_vector(
            observation,
            shape=(OBSERVATION_WIDTH,),
            field="observation",
        )
        resolved_window = _numeric_vector(
            reference_window,
            shape=(REFERENCE_HORIZON_STEPS, REFERENCE_WIDTH),
            field="reference_window",
        )
        combined = np.concatenate((resolved_observation, resolved_window.ravel()))
        normalized = np.clip(
            (combined - mean) / np.sqrt(variance + 1e-8),
            -clip,
            clip,
        ).astype(np.float32)
        action, _state = policy.predict(normalized, deterministic=True)
        observed = np.asarray(action, dtype=np.float64)
        if (
            observed.shape != (ACTION_WIDTH,)
            or not np.isfinite(observed).all()
            or np.any(observed < -1.0)
            or np.any(observed > 1.0)
        ):
            raise ExperimentContractError("residual inference returned an invalid action")
        return observed.copy()

    receipt = _receipt(
        Path(path),
        file_state,
        artifact_kind="minari_reference_residual_actor",
        architecture_id="sb3_actor_critic_pi256x256_vf256x256_relu_deterministic/v1",
        expected_sha256=expected,
        registered_sha256=LOCAL_REFERENCE_RESIDUAL_SHA256,
        reference_width=REFERENCE_WIDTH,
        reference_horizon_steps=REFERENCE_HORIZON_STEPS,
        residual_scale=RESIDUAL_SCALE,
        inference_output_contract=(
            "unscaled_dimensionless_action_in_closed_interval_-1_1;"
            "caller_multiplies_by_residual_scale_before_composition"
        ),
    )
    return infer, receipt


__all__ = [
    "ACTION_WIDTH",
    "LOCAL_BEHAVIOR_CLONING_SHA256",
    "LOCAL_CONTROLLER_CLAIM_STATUS",
    "LOCAL_REFERENCE_RESIDUAL_SHA256",
    "OBSERVATION_WIDTH",
    "REFERENCE_HORIZON_STEPS",
    "REFERENCE_WIDTH",
    "RESIDUAL_SCALE",
    "LocalBehaviorCloningController",
    "LocalControllerReceipt",
    "load_local_behavior_cloning_controller",
    "load_local_reference_residual_controller",
]
