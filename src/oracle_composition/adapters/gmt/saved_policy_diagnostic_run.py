"""Evaluate two pinned numeric policies; never resumes an optimizer or trains."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from .io import read_verified_bytes, sha256_file, write_deterministic_npz, write_json_receipt
from .saved_policy_diagnostic_config import (
    ACTION_DIMENSION,
    HORIZON_STEPS,
    NOISE_FILENAME,
    NOISE_GENERATOR_ID,
    NOISE_SEEDS,
    PARITY_ARTIFACT,
    RUN_ARTIFACT,
    RUN_MANIFEST_FILENAME,
    FileBinding,
    diagnostic_config_payload,
    diagnostic_episode_schedule,
    load_noise_archive,
    load_saved_policy_diagnostic_config,
    verify_saved_policy_diagnostic_outputs,
)
from .training_normalizer import (
    FIXED_NORMALIZER_STATE_SHA256,
    MAX_NUMERIC_POLICY_BYTES,
    pinned_normalizer_from_actor,
    validate_policy_normalizer_archive,
)


def paired_noise() -> np.ndarray:
    """Match Torch's per-action 1x23 draws without changing its global generator."""
    noise = np.empty((len(NOISE_SEEDS), HORIZON_STEPS, ACTION_DIMENSION), dtype="<f4")
    for pair, seed in enumerate(NOISE_SEEDS):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for step in range(HORIZON_STEPS):
            noise[pair, step] = (
                torch.empty((1, ACTION_DIMENSION), dtype=torch.float32)
                .normal_(generator=generator)
                .numpy()[0]
            )
    return noise


def strict_load_policy(model, binding: FileBinding) -> dict:
    """Load bounded numeric tensors into the trusted, locally constructed policy."""
    encoded = read_verified_bytes(
        binding.path,
        binding.sha256,
        expected_size=binding.size,
        maximum_size=MAX_NUMERIC_POLICY_BYTES,
    )
    validate_policy_normalizer_archive(encoded)
    expected = model.policy.state_dict()
    with np.load(io.BytesIO(encoded), allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError("saved policy tensor keys differ from the trusted architecture")
        arrays = {name: archive[name].copy() for name in archive.files}
    for name, value in arrays.items():
        target = expected[name].detach().cpu().numpy()
        if (
            value.shape != target.shape
            or value.dtype != target.dtype
            or not np.isfinite(value).all()
        ):
            raise ValueError(f"saved policy tensor contract differs: {name}")
    model.policy.load_state_dict(
        {name: torch.from_numpy(value) for name, value in arrays.items()}, strict=True
    )
    model.policy.set_training_mode(False)
    model.policy.requires_grad_(False)
    readback = model.policy.state_dict()
    if any(
        not np.array_equal(value, readback[name].detach().cpu().numpy())
        for name, value in arrays.items()
    ):
        raise ValueError("saved policy state readback differs from the admitted archive")
    schema = [[name, arrays[name].dtype.str, list(arrays[name].shape)] for name in sorted(arrays)]
    return {
        "archive": binding.receipt(),
        "state_dict_key_count": len(arrays),
        "state_dict_schema_sha256": hashlib.sha256(
            json.dumps(schema, separators=(",", ":")).encode()
        ).hexdigest(),
        "log_std_sha256": hashlib.sha256(arrays["log_std"].tobytes(order="C")).hexdigest(),
        "normalizer_state_sha256": FIXED_NORMALIZER_STATE_SHA256,
        "strict_state_dict_loaded": True,
        "state_dict_readback_exact": True,
    }


def policy_state_digest(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.policy.state_dict().items()):
        array = tensor.detach().cpu().numpy()
        digest.update(json.dumps([name, array.dtype.str, array.shape]).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class RecordedNoisePredictor:
    """Adapt the existing deterministic trace writer to explicit, indexed action noise."""

    def __init__(self, model, noise: np.ndarray | None) -> None:
        if noise is not None and (
            noise.shape != (HORIZON_STEPS, ACTION_DIMENSION)
            or noise.dtype != np.dtype("<f4")
            or not np.isfinite(noise).all()
        ):
            raise ValueError("sampling requires finite float32[1000,23] noise")
        if (
            model.policy.device.type != "cpu"
            or model.policy.squash_output
            or model.action_space.shape != (ACTION_DIMENSION,)
            or not np.array_equal(model.action_space.low, np.full(ACTION_DIMENSION, -1.0))
            or not np.array_equal(model.action_space.high, np.ones(ACTION_DIMENSION))
        ):
            raise ValueError("saved-policy diagnostic requires the unsquashed CPU Box action path")
        self.model = model
        self.noise = noise
        self.index = 0
        self.clipped_components = 0
        self.max_abs_unclipped = 0.0
        self.sampled_means, self.sampled_stds, self.unclipped_actions = [], [], []
        model.policy.set_training_mode(False)

    def predict(self, observation: np.ndarray, *, deterministic: bool = True):
        if deterministic is not True or self.index >= HORIZON_STEPS:
            raise ValueError("diagnostic trace adapter called outside its fixed schedule")
        policy = self.model.policy
        with torch.no_grad():
            tensor, vectorized = policy.obs_to_tensor(observation)
            if vectorized:
                raise ValueError("diagnostic requires one unbatched observation")
            distribution = policy.get_distribution(tensor).distribution
            if not isinstance(distribution, torch.distributions.Normal):
                raise ValueError("saved policy must expose its diagonal normal distribution")
            epsilon = (
                torch.zeros((1, ACTION_DIMENSION), dtype=torch.float32)
                if self.noise is None
                else torch.from_numpy(self.noise[self.index : self.index + 1].copy())
            )
            raw = (distribution.loc + epsilon * distribution.scale).cpu().numpy().reshape(-1)
        if (
            raw.shape != (ACTION_DIMENSION,)
            or raw.dtype != np.dtype("<f4")
            or not np.isfinite(raw).all()
        ):
            raise ValueError("non-finite or incompatible policy sample")
        action = np.clip(raw, self.model.action_space.low, self.model.action_space.high)
        if self.noise is not None:
            self.sampled_means.append(distribution.loc.cpu().numpy()[0].copy())
            self.sampled_stds.append(distribution.scale.cpu().numpy()[0].copy())
            self.unclipped_actions.append(raw.copy())
        if self.noise is None:
            expected, _ = self.model.predict(observation, deterministic=True)
            if not np.array_equal(action, expected):
                raise ValueError(
                    "zero-noise action differs from installed SB3 deterministic prediction"
                )
        self.index += 1
        self.clipped_components += int(np.count_nonzero(raw != action))
        self.max_abs_unclipped = max(self.max_abs_unclipped, float(np.max(np.abs(raw))))
        return action, None

    def sampling_receipt(self, seed: int) -> dict:
        return {
            "noise_seed": seed,
            "noise_rows_available": HORIZON_STEPS,
            "noise_rows_consumed": self.index,
            "action_dimension": ACTION_DIMENSION,
            "clipped_component_count": self.clipped_components,
            "total_component_count": self.index * ACTION_DIMENSION,
            "max_abs_unclipped_action": self.max_abs_unclipped,
        }

    def sampled_action_evidence(self) -> dict[str, np.ndarray]:
        if self.noise is None or self.index == 0:
            raise ValueError("no sampled action evidence in this episode")
        return {
            "policy_mean": np.asarray(self.sampled_means, dtype="<f4"),
            "policy_std": np.asarray(self.sampled_stds, dtype="<f4"),
            "unclipped_action": np.asarray(self.unclipped_actions, dtype="<f4"),
        }


def write_parity_receipt(
    output: Path, policy: str, label: str, expected: dict, observed: dict
) -> dict:
    name = f"{policy}_deterministic_parity.json"
    digest = write_json_receipt(
        output / name,
        {
            "schema_version": 1,
            "artifact": PARITY_ARTIFACT,
            "policy": policy,
            "label": label,
            "expected": expected,
            "observed": observed,
            "passed": expected == observed,
        },
    )
    if expected != observed:
        raise ValueError(f"{policy} deterministic parity failed; no sampled evaluation admitted")
    return {"path": name, "sha256": digest, "size": (output / name).stat().st_size}


def verified_canonical_noise(binding: FileBinding) -> np.ndarray:
    noise = load_noise_archive(binding)
    if not np.array_equal(noise, paired_noise()):
        raise ValueError("retained noise differs from the declared rowwise Torch generator")
    return noise


def prepare(retained_root: Path, protocol: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    noise_path = output / NOISE_FILENAME
    write_deterministic_npz(
        noise_path,
        {
            "seeds": np.asarray(NOISE_SEEDS, dtype="<i8"),
            "standard_normal": paired_noise(),
        },
    )
    payload = diagnostic_config_payload(
        retained_root=retained_root, protocol_path=protocol, noise_path=noise_path
    )
    config_path = output / "diagnostic_config.json"
    digest = write_json_receipt(config_path, payload)
    load_saved_policy_diagnostic_config(config_path)
    return {"config": str(config_path), "sha256": digest, "noise_sha256": sha256_file(noise_path)}


def run_diagnostic(config_path: Path, output: Path) -> dict:
    # Simulator imports and construction belong only to the supervised run path.
    from .course_config import load_run_config
    from .course_run import _frozen_actor_digest, _rollout, make_env, make_policy

    config = load_saved_policy_diagnostic_config(config_path)
    noise = verified_canonical_noise(config.noise)
    course = load_run_config(config.course_config.path)
    output.mkdir(exist_ok=True)
    if output.is_symlink() or any(
        p.name not in {"child_stdout.json", "child_stderr.log"} for p in output.iterdir()
    ):
        raise ValueError("diagnostic output must be fresh except for supervisor logs")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    outputs = {}
    for name, binding in (
        (
            "input_diagnostic_config.json",
            FileBinding(config_path, config.sha256, len(config.encoded)),
        ),
        ("input_course_config.json", config.course_config),
        (NOISE_FILENAME, config.noise),
    ):
        payload = read_verified_bytes(binding.path, binding.sha256, expected_size=binding.size)
        with (output / name).open("xb") as handle:
            handle.write(payload)
        outputs[name] = binding.sha256
    env = make_env(course, record_trajectory=True)
    try:
        base_before = _frozen_actor_digest(env)
        normalizer = pinned_normalizer_from_actor(env.actor._actor)
        models, admissions, state_before = {}, {}, {}
        for policy in ("initial", "final"):
            models[policy] = make_policy(
                env, course.raw["seed"], course.trainer, fixed_normalizer=normalizer
            )
            admissions[policy] = strict_load_policy(models[policy], config.policies[policy])
            state_before[policy] = policy_state_digest(models[policy])
        retained = json.loads(
            read_verified_bytes(config.course_manifest.path, config.course_manifest.sha256)
        )
        episodes, parity_receipts = [], {}
        for episode in diagnostic_episode_schedule():
            if episode.noise_seed is not None and set(parity_receipts) != {"initial", "final"}:
                raise ValueError("sampling is blocked until both deterministic traces reproduce")
            epsilon = (
                None if episode.noise_seed is None else noise[NOISE_SEEDS.index(episode.noise_seed)]
            )
            predictor = RecordedNoisePredictor(models[episode.policy], epsilon)
            summary, artifacts = _rollout(course, env, predictor, output, episode.label)
            outputs.update(artifacts)
            if episode.noise_seed is not None:
                name = f"{episode.label}_sampling.npz"
                outputs[name] = write_deterministic_npz(
                    output / name, predictor.sampled_action_evidence()
                )
            sampling = (
                None
                if episode.noise_seed is None
                else predictor.sampling_receipt(episode.noise_seed)
            )
            episodes.append(
                {
                    "sequence_index": episode.sequence_index,
                    "kind": episode.kind,
                    "label": episode.label,
                    "policy": episode.policy,
                    "noise_seed": episode.noise_seed,
                    "summary": summary,
                    "sampling": sampling,
                }
            )
            if episode.noise_seed is None:
                old_label = "zero_residual" if episode.policy == "initial" else "final_policy"
                expected = {
                    kind: retained["outputs"][f"{old_label}_{suffix}"]
                    for kind, suffix in (
                        ("frames", "frames.jsonl"),
                        ("trajectory", "trajectory.npz"),
                        ("evaluation", "evaluation.json"),
                    )
                }
                observed = {
                    kind: artifacts[f"{episode.label}_{suffix}"]
                    for kind, suffix in (
                        ("frames", "frames.jsonl"),
                        ("trajectory", "trajectory.npz"),
                        ("evaluation", "evaluation.json"),
                    )
                }
                receipt = write_parity_receipt(
                    output, episode.policy, episode.label, expected, observed
                )
                outputs[receipt["path"]] = receipt["sha256"]
                parity_receipts[episode.policy] = receipt
            print(
                json.dumps(
                    {
                        "event": "diagnostic_episode",
                        "sequence_index": episode.sequence_index,
                        "label": episode.label,
                        "steps": summary["steps"],
                    }
                ),
                flush=True,
            )
        if _frozen_actor_digest(env) != base_before or any(
            policy_state_digest(model) != state_before[policy] for policy, model in models.items()
        ):
            raise ValueError("evaluation mutated an admitted policy or frozen tracker")
    finally:
        env.close()
    manifest = {
        "schema_version": 1,
        "artifact": RUN_ARTIFACT,
        "status": "completed",
        "input_config_sha256": config.sha256,
        "inputs": config.resource_binding(),
        "outputs": outputs,
        "policy_admission": admissions,
        "deterministic_parity": {
            "sampled_started_after_both_receipts": True,
            "receipts": parity_receipts,
        },
        "episodes": episodes,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "stable_baselines3": version("stable-baselines3"),
            "device": "cpu",
            "sampling_generator": NOISE_GENERATOR_ID,
        },
        "claims": {
            "evaluation_only": True,
            "training_performed": False,
            "gradient_steps": 0,
            "optimizer_steps": 0,
            "checkpoint_selection_performed": False,
            "heldout_generalization_tested": False,
            "automatic_adoption": False,
        },
    }
    write_json_receipt(output / RUN_MANIFEST_FILENAME, manifest)
    verify_saved_policy_diagnostic_outputs(config, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser(
        "prepare", help="Pin numeric inputs and paired noise; no simulator"
    )
    command.add_argument("--retained-run", type=Path, required=True)
    command.add_argument("--protocol", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = commands.add_parser("run", help="Evaluate only, under the native supervisor")
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(
            args.retained_run.resolve(), args.protocol.resolve(), args.output.resolve()
        )
    else:
        result = run_diagnostic(args.config.resolve(), args.output.resolve())
    print(json.dumps(result))


if __name__ == "__main__":
    main()
