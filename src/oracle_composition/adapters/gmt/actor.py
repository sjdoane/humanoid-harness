"""Reviewed plain-PyTorch reconstruction of the pinned GMT G1 actor graph."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from .checkpoint import _validate_numeric_arrays, load_converted_arrays
from .contracts import (
    ACTION_DIM,
    ACTION_SCALE,
    CURRENT_PROPRIOCEPTION_SLICE,
    CURRENT_REFERENCE_SLICE,
    DEFAULT_DOF_POSITION,
    LAYER_NORM_EPSILON,
    NORMALIZER_EPSILON,
    OBSERVATION_DIM,
    PROPRIOCEPTION_HISTORY_SLICE,
    RAW_ACTION_MAX,
    RAW_ACTION_MIN,
    REFERENCE_FRAME_DIM,
    REFERENCE_HORIZON,
    REFERENCE_SLICE,
    TENSOR_SPECS,
)
from .io import GMTAdmissionError


class _TemporalEncoder(nn.Module):
    def __init__(
        self,
        *,
        frame_dim: int,
        projected_dim: int,
        conv1_dim: int,
        conv2_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.frame_dim = frame_dim
        self.input_linear = nn.Linear(frame_dim, projected_dim)
        self.conv1 = nn.Conv1d(projected_dim, conv1_dim, kernel_size=6, stride=2, padding=0)
        self.conv2 = nn.Conv1d(conv1_dim, conv2_dim, kernel_size=4, stride=2, padding=0)
        self.output_linear = nn.Linear(conv2_dim * 3, output_dim)
        self.activation = nn.SiLU()

    def forward(self, values: Tensor) -> Tensor:
        batch_size = values.shape[0]
        frames = values.reshape(batch_size * REFERENCE_HORIZON, self.frame_dim)
        projected = self.activation(self.input_linear(frames))
        temporal = projected.reshape(batch_size, REFERENCE_HORIZON, -1).permute(0, 2, 1)
        temporal = self.activation(self.conv1(temporal))
        temporal = self.activation(self.conv2(temporal))
        return self.output_linear(torch.flatten(temporal, start_dim=1))


class GMTActor(nn.Module):
    """The statically reviewed 2154D-to-23D GMT deployment actor."""

    def __init__(self) -> None:
        super().__init__()
        self.history_encoder = _TemporalEncoder(
            frame_dim=74,
            projected_dim=30,
            conv1_dim=20,
            conv2_dim=10,
            output_dim=64,
        )
        self.motion_encoder = _TemporalEncoder(
            frame_dim=REFERENCE_FRAME_DIM,
            projected_dim=60,
            conv1_dim=40,
            conv2_dim=20,
            output_dim=128,
        )
        self.actor_backbone = nn.Sequential(
            nn.Linear(296, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256, eps=LAYER_NORM_EPSILON),
            nn.SiLU(),
            nn.Linear(256, ACTION_DIM),
        )
        self.register_buffer("normalizer_count", torch.zeros(1, dtype=torch.int64))
        self.register_buffer("normalizer_mean", torch.zeros(OBSERVATION_DIM, dtype=torch.float32))
        self.register_buffer("normalizer_std", torch.ones(OBSERVATION_DIM, dtype=torch.float32))

    def forward(self, observation: Tensor) -> Tensor:
        if observation.ndim != 2 or observation.shape[1] != OBSERVATION_DIM:
            raise ValueError(
                f"GMT observation must have shape [batch, {OBSERVATION_DIM}], "
                f"observed {tuple(observation.shape)}"
            )
        normalized = (observation - self.normalizer_mean) / (
            self.normalizer_std + NORMALIZER_EPSILON
        )
        normalized = torch.clamp(normalized, min=-torch.inf, max=torch.inf).to(torch.float32)
        reference = normalized[:, REFERENCE_SLICE]
        current_reference = reference[:, CURRENT_REFERENCE_SLICE]
        current_proprioception = normalized[:, CURRENT_PROPRIOCEPTION_SLICE]
        proprioception_history = normalized[:, PROPRIOCEPTION_HISTORY_SLICE]
        history_latent = self.history_encoder(proprioception_history)
        motion_latent = self.motion_encoder(reference)
        actor_input = torch.cat(
            (
                current_proprioception,
                current_reference,
                motion_latent,
                history_latent,
            ),
            dim=1,
        )
        return self.actor_backbone(actor_input)


def actor_from_arrays(arrays: Mapping[str, np.ndarray], *, freeze: bool = True) -> GMTActor:
    _validate_numeric_arrays(arrays, TENSOR_SPECS)
    actor = GMTActor()
    state = {key: torch.from_numpy(np.asarray(value)) for key, value in arrays.items()}
    try:
        actor.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise GMTAdmissionError("converted tensors do not satisfy the reviewed actor") from exc
    actor.eval()
    actor.requires_grad_(not freeze)
    return actor


def load_actor(path: Path, *, expected_sha256: str, freeze: bool = True) -> GMTActor:
    """Load the deterministic converted artifact, never an upstream PT file."""

    arrays = load_converted_arrays(Path(path), expected_sha256=expected_sha256)
    return actor_from_arrays(arrays, freeze=freeze)


def apply_deployment_action(raw_action: Tensor) -> tuple[Tensor, Tensor]:
    """Return the pre-clip history value and the corresponding PD position target."""

    if raw_action.shape[-1] != ACTION_DIM:
        raise ValueError(f"GMT action must end in {ACTION_DIM} values")
    history_action = raw_action.clone()
    default = raw_action.new_tensor(DEFAULT_DOF_POSITION)
    target = raw_action.clamp(RAW_ACTION_MIN, RAW_ACTION_MAX) * ACTION_SCALE + default
    return history_action, target
