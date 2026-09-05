"""Deterministic inference from the repository's strict data-only TQC NPZ."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import array_sha256
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.experiments.tqc_actor_npz import (
    ACTION_WIDTH,
    OBSERVATION_WIDTH,
    LoadedTQCActor,
    load_actor_npz,
)

INFERENCE_ID = "strict_npz_tqc_deterministic_mean_cpu_float32/v1"


@dataclass(frozen=True, slots=True)
class StrictTQCAction:
    """Exact actor input, normalized output, and physical action."""

    actor_input: np.ndarray
    normalized_output: np.ndarray
    physical_action: np.ndarray

    def __post_init__(self) -> None:
        expected = (
            ("actor_input", (OBSERVATION_WIDTH,)),
            ("normalized_output", (ACTION_WIDTH,)),
            ("physical_action", (ACTION_WIDTH,)),
        )
        for field, shape in expected:
            value = np.asarray(getattr(self, field))
            if value.dtype != np.dtype("<f4") or value.shape != shape:
                raise ExperimentContractError(f"{field} must be little-endian float32 {shape}")
            if not value.flags.c_contiguous or not np.isfinite(value).all():
                raise ExperimentContractError(f"{field} must be finite C-order data")
            immutable = np.frombuffer(value.tobytes(order="C"), dtype="<f4").reshape(shape)
            object.__setattr__(self, field, immutable)
        if np.any(np.abs(self.normalized_output) > 1.0):
            raise ExperimentContractError("normalized actor output lies outside [-1, 1]")

    @property
    def actor_input_sha256(self) -> str:
        return array_sha256(self.actor_input)

    @property
    def actor_output_sha256(self) -> str:
        return array_sha256(self.normalized_output)

    @property
    def physical_action_sha256(self) -> str:
        return array_sha256(self.physical_action)


class StrictTQCActorRuntime:
    """Pure feed-forward actor; no SB3 object or external checkpoint is accepted."""

    def __init__(self, loaded_actor: LoadedTQCActor) -> None:
        if type(loaded_actor) is not LoadedTQCActor:
            raise ExperimentContractError("strict actor runtime requires LoadedTQCActor authority")
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - project train extra is pinned
            raise ExperimentContractError("Torch is required for strict actor inference") from exc
        self.loaded_actor = loaded_actor
        self._torch = torch
        names = (
            "latent_pi.0.weight",
            "latent_pi.0.bias",
            "latent_pi.2.weight",
            "latent_pi.2.bias",
            "mu.weight",
            "mu.bias",
        )
        self._parameters = {
            name: torch.from_numpy(loaded_actor.arrays[name].copy(order="C")) for name in names
        }
        self._low = loaded_actor.arrays["action_low"]
        self._high = loaded_actor.arrays["action_high"]

    @classmethod
    def from_npz(cls, path: Path, *, expected_sha256: str) -> StrictTQCActorRuntime:
        """Admit exactly one strict NPZ through the existing code-free loader."""

        return cls(load_actor_npz(Path(path), expected_sha256=expected_sha256))

    def act(self, observation: object) -> StrictTQCAction:
        """Run deterministic mean inference and the frozen physical transform."""

        try:
            actor_input = np.ascontiguousarray(observation, dtype="<f4")
        except (TypeError, ValueError, OverflowError) as exc:
            raise ExperimentContractError("actor observation is not numeric") from exc
        if actor_input.shape != (OBSERVATION_WIDTH,) or not np.isfinite(actor_input).all():
            raise ExperimentContractError("actor observation must be finite shape (348,)")
        torch = self._torch
        functional = torch.nn.functional
        value = torch.from_numpy(actor_input.copy(order="C")).reshape(1, OBSERVATION_WIDTH)
        with torch.inference_mode():
            hidden = functional.relu(
                functional.linear(
                    value,
                    self._parameters["latent_pi.0.weight"],
                    self._parameters["latent_pi.0.bias"],
                )
            )
            hidden = functional.relu(
                functional.linear(
                    hidden,
                    self._parameters["latent_pi.2.weight"],
                    self._parameters["latent_pi.2.bias"],
                )
            )
            mean = functional.linear(
                hidden,
                self._parameters["mu.weight"],
                self._parameters["mu.bias"],
            )
            normalized = np.ascontiguousarray(torch.tanh(mean).numpy()[0], dtype="<f4")
        physical = np.ascontiguousarray(
            self._low + np.float32(0.5) * (normalized + np.float32(1.0)) * (self._high - self._low),
            dtype="<f4",
        )
        return StrictTQCAction(
            actor_input=actor_input,
            normalized_output=normalized,
            physical_action=physical,
        )


__all__ = ["INFERENCE_ID", "StrictTQCAction", "StrictTQCActorRuntime"]
