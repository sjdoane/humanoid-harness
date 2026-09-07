"""Fixed GMT actor-normalizer reuse for one opt-in residual trainer profile."""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .contracts import NORMALIZER_EPSILON, OBSERVATION_DIM
from .io import GMTAdmissionError, validate_zip_members

FIXED_NORMALIZER_ID = "gmt_g1_fixed_actor_observation_normalizer/v1"
FIXED_NORMALIZER_STATE_SHA256 = (
    "42934b5e9155536db8e36c27f75858d6c9f60c93d29b9352191fe6cd5f9ce250"
)
FIXED_NORMALIZER_MEAN_SHA256 = (
    "72b2c94ae9873c573d480bca87d5914fc6e56e4677f07836c00dceb265b97ee6"
)
FIXED_NORMALIZER_STD_SHA256 = (
    "bc90eb2f9426d5aa226e4ef563d74533d2a6f8a00bc3ef228b47528f29efc765"
)
_DIGEST_PREFIX = f"{FIXED_NORMALIZER_ID}\0".encode()
_POLICY_BUFFER_SUFFIXES = ("normalizer_mean", "normalizer_std")
_POLICY_BUFFER_PREFIXES = (
    "features_extractor.",
    "pi_features_extractor.",
    "vf_features_extractor.",
)
MAX_NUMERIC_POLICY_BYTES = 32 * 1024**2


def normalizer_state_sha256(mean: np.ndarray, standard_deviation: np.ndarray) -> str:
    """Hash exact validated float32 buffers under the versioned transform identity."""

    digest = hashlib.sha256(_DIGEST_PREFIX)
    digest.update(mean.tobytes(order="C"))
    digest.update(standard_deviation.tobytes(order="C"))
    return digest.hexdigest()


def _buffer(value: object, *, name: str, positive: bool = False) -> np.ndarray:
    if (
        type(value) is not np.ndarray
        or value.shape != (OBSERVATION_DIM,)
        or value.dtype != np.dtype("<f4")
        or not np.isfinite(value).all()
        or (positive and np.any(value <= 0))
    ):
        requirement = "finite positive" if positive else "finite"
        raise ValueError(
            f"fixed normalizer {name} must be exact {requirement} float32[{OBSERVATION_DIM}]"
        )
    result = np.array(value, dtype="<f4", order="C", copy=True)
    result.flags.writeable = False
    return result


@dataclass(frozen=True, slots=True)
class FixedNormalizerState:
    mean: np.ndarray
    standard_deviation: np.ndarray
    sha256: str

    @classmethod
    def from_arrays(
        cls,
        mean: object,
        standard_deviation: object,
        *,
        expected_sha256: str,
    ) -> FixedNormalizerState:
        if (
            type(expected_sha256) is not str
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError("fixed normalizer expected SHA-256 is malformed")
        admitted_mean = _buffer(mean, name="mean")
        admitted_std = _buffer(
            standard_deviation, name="standard deviation", positive=True
        )
        observed = normalizer_state_sha256(admitted_mean, admitted_std)
        if observed != expected_sha256:
            raise ValueError("fixed normalizer buffers differ from their exact identity")
        return cls(admitted_mean, admitted_std, observed)

    def require_pinned(self) -> None:
        if (
            self.sha256 != FIXED_NORMALIZER_STATE_SHA256
            or hashlib.sha256(self.mean.tobytes()).hexdigest()
            != FIXED_NORMALIZER_MEAN_SHA256
            or hashlib.sha256(self.standard_deviation.tobytes()).hexdigest()
            != FIXED_NORMALIZER_STD_SHA256
        ):
            raise ValueError("fixed normalizer does not match the pinned GMT actor buffers")


def pinned_normalizer_from_actor(actor: object) -> FixedNormalizerState:
    """Read only the already-admitted numeric actor buffers; never load another artifact."""

    mean = getattr(actor, "normalizer_mean", None)
    standard_deviation = getattr(actor, "normalizer_std", None)
    if not all(
        hasattr(value, "detach") and hasattr(value, "device")
        for value in (mean, standard_deviation)
    ):
        raise ValueError("pinned GMT actor lacks its fixed normalizer buffers")
    if mean.device.type != "cpu" or standard_deviation.device.type != "cpu":  # type: ignore[union-attr]
        raise ValueError("pinned GMT actor normalizer must remain on CPU")
    return FixedNormalizerState.from_arrays(
        mean.detach().numpy().copy(),  # type: ignore[union-attr]
        standard_deviation.detach().numpy().copy(),  # type: ignore[union-attr]
        expected_sha256=FIXED_NORMALIZER_STATE_SHA256,
    )


def fixed_normalizer_contract() -> dict[str, object]:
    """Describe one non-authorable, artifact-bound observation transform."""

    return {
        "schema_id": FIXED_NORMALIZER_ID,
        "schema_version": 1,
        "source_binding": "input_config.assets.weights.normalizer_mean_and_std",
        "normalizer_state_sha256": FIXED_NORMALIZER_STATE_SHA256,
        "buffers": {
            "normalizer_mean": {
                "sha256": FIXED_NORMALIZER_MEAN_SHA256,
                "dtype": "<f4",
                "shape": [OBSERVATION_DIM],
            },
            "normalizer_std": {
                "sha256": FIXED_NORMALIZER_STD_SHA256,
                "dtype": "<f4",
                "shape": [OBSERVATION_DIM],
            },
        },
        "normalized_prefix": {"start": 0, "stop": OBSERVATION_DIM},
        "tail_semantics": "identity",
        "formula": "float32_(observation_minus_mean)_divided_by_(std_plus_1e-4)",
        "epsilon": NORMALIZER_EPSILON,
        "clipping": "none",
        "online_statistics": False,
        "features_extractor": {
            "trainable_parameter_count": 0,
            "state": "nontrainable_registered_buffers",
            "output_width": "input_observation_width",
        },
    }


def normalized_base_max_abs(observations: object, state: FixedNormalizerState) -> float:
    """Measure the transform on raw rollout-buffer observations, never model calls."""

    state.require_pinned()
    values = np.asarray(observations)
    if (
        values.ndim < 2
        or values.shape[-1] <= OBSERVATION_DIM
        or not np.issubdtype(values.dtype, np.floating)
        or not np.isfinite(values).all()
    ):
        raise ValueError("fixed-normalizer telemetry requires finite flat rollout observations")
    raw = np.ascontiguousarray(
        values.reshape(-1, values.shape[-1])[:, :OBSERVATION_DIM], dtype="<f4"
    )
    normalized = np.divide(
        np.subtract(raw, state.mean, dtype=np.float32),
        np.add(state.standard_deviation, np.float32(NORMALIZER_EPSILON), dtype=np.float32),
        dtype=np.float32,
    )
    if not np.isfinite(normalized).all():
        raise ValueError("fixed-normalizer telemetry produced non-finite values")
    return float(np.max(np.abs(normalized)))


def policy_normalizer_metadata(
    state_dict: Mapping[str, object], state: FixedNormalizerState
) -> dict[str, object]:
    """Verify and describe every retained copy of the required policy buffers."""

    state.require_pinned()
    expected_keys = {
        f"{prefix}{suffix}"
        for prefix in _POLICY_BUFFER_PREFIXES
        for suffix in _POLICY_BUFFER_SUFFIXES
    }
    observed_keys = {
        name
        for name in state_dict
        if name.endswith(tuple(_POLICY_BUFFER_SUFFIXES))
    }
    if observed_keys != expected_keys:
        raise ValueError("fixed-normalizer policy state lacks required transform buffers")
    for suffix in _POLICY_BUFFER_SUFFIXES:
        expected = state.mean if suffix == "normalizer_mean" else state.standard_deviation
        for prefix in _POLICY_BUFFER_PREFIXES:
            name = f"{prefix}{suffix}"
            tensor = state_dict[name]
            if not hasattr(tensor, "detach") or not hasattr(tensor, "device"):
                raise ValueError("fixed-normalizer policy state contains changed buffers")
            if tensor.device.type != "cpu":  # type: ignore[union-attr]
                raise ValueError("fixed-normalizer policy state contains changed buffers")
            array = tensor.detach().numpy()  # type: ignore[union-attr]
            if (
                array.dtype != np.dtype("<f4")
                or not np.array_equal(array, expected)
            ):
                raise ValueError("fixed-normalizer policy state contains changed buffers")
    return fixed_normalizer_policy_metadata()


def fixed_normalizer_policy_metadata() -> dict[str, object]:
    return {
        **fixed_normalizer_contract(),
        "numeric_policy_required_buffer_suffixes": list(_POLICY_BUFFER_SUFFIXES),
        "policy_artifact_semantics": "numeric_weights_and_fixed_transform_not_optimizer_resume",
    }


def validate_policy_normalizer_archive(encoded: bytes) -> dict[str, object]:
    """Verify that a retained numeric policy carries the exact fixed transform buffers."""

    if type(encoded) is not bytes or not 0 < len(encoded) <= MAX_NUMERIC_POLICY_BYTES:
        raise ValueError("fixed-normalizer numeric policy bytes are outside their bound")
    try:
        with zipfile.ZipFile(io.BytesIO(encoded), "r") as archive:
            members = validate_zip_members(
                archive,
                maximum_member_size=16 * 1024**2,
            )
            if (
                not 1 <= len(members) <= 128
                or sum(member.file_size for member in members.values())
                > MAX_NUMERIC_POLICY_BYTES
                or any(not name.endswith(".npy") for name in members)
            ):
                raise ValueError("fixed-normalizer numeric policy archive is unbounded")
        with np.load(io.BytesIO(encoded), allow_pickle=False) as archive:
            names = set(archive.files)
            expected_keys = {
                f"{prefix}{suffix}"
                for prefix in _POLICY_BUFFER_PREFIXES
                for suffix in _POLICY_BUFFER_SUFFIXES
            }
            observed_keys = {
                name
                for name in names
                if name.endswith(tuple(_POLICY_BUFFER_SUFFIXES))
            }
            if observed_keys != expected_keys:
                raise ValueError("fixed-normalizer numeric policy lacks paired buffers")
            for prefix in _POLICY_BUFFER_PREFIXES:
                FixedNormalizerState.from_arrays(
                    archive[f"{prefix}normalizer_mean"],
                    archive[f"{prefix}normalizer_std"],
                    expected_sha256=FIXED_NORMALIZER_STATE_SHA256,
                )
    except (GMTAdmissionError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError("fixed-normalizer numeric policy archive is malformed") from exc
    return fixed_normalizer_policy_metadata()


__all__ = [
    "FIXED_NORMALIZER_ID",
    "FIXED_NORMALIZER_MEAN_SHA256",
    "FIXED_NORMALIZER_STATE_SHA256",
    "FIXED_NORMALIZER_STD_SHA256",
    "MAX_NUMERIC_POLICY_BYTES",
    "FixedNormalizerState",
    "fixed_normalizer_contract",
    "fixed_normalizer_policy_metadata",
    "normalized_base_max_abs",
    "normalizer_state_sha256",
    "pinned_normalizer_from_actor",
    "policy_normalizer_metadata",
    "validate_policy_normalizer_archive",
]
