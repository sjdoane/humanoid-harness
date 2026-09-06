"""Content-addressed frozen-runtime contract for the T2 reward study."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import json
import platform
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

from oracle_composition.contracts.reference_identity_v2 import canonical_json_bytes
from oracle_composition.experiments.fixed_reference import ExperimentContractError
from oracle_composition.phase_b.contracts import (
    FROZEN_TRACKING_REWARD_CONFIG,
    TRACKING_REWARD_CONFIG_SHA256,
    TRACKING_REWARD_ID,
)

T2_EXECUTION_MANIFEST_SCHEMA_ID = "t2_execution_manifest_v2"
T2_EXECUTION_LEGACY_MANIFEST_SCHEMA_ID = "t2_execution_manifest_v1"
T2_EXECUTION_MANIFEST_ID = "t2_reward_study_frozen_execution/v1"
T2_EXECUTION_LAUNCH_BASE_COMMIT = "ed9f1d38aba4f7b41a576b0fd9c8be4f6b8b47fe"
T2_EXECUTION_BASE_COMMIT = T2_EXECUTION_LAUNCH_BASE_COMMIT
T2_EXECUTION_PENDING_STATUS = "execution_no_go_candidate_pending"
T2_EXECUTION_FINAL_READY_STATUS = "final_ready"
T2_EXECUTION_STATUS = T2_EXECUTION_PENDING_STATUS

_SOURCE_PATHS = {
    "environment_adapter": "src/oracle_composition/envs/reference_corpus.py",
    "policy": "src/oracle_composition/phase_b/policy.py",
    "reward_compositor": "src/oracle_composition/phase_b/reward.py",
    "strict_checkpoint_loader": "src/oracle_composition/phase_b/strict_npz.py",
    "tracker_reward": "src/oracle_composition/tracking/reward.py",
    "trainer": "src/oracle_composition/phase_b/training.py",
}
_DEPENDENCIES = ("gymnasium", "mujoco", "numpy", "torch")
_ADMISSION_RECORD_PREFIXES = (
    ("docs",),
    ("experiments", "004_t2_reward_study"),
    ("experiments", "bootstrap_tqc_humanoid", "reviews"),
    ("experiments", "family_b_target_speed_v1", "receipts"),
)


def _regular_bytes(path: Path, *, field: str, maximum: int) -> bytes:
    candidate = Path(path)
    try:
        before = candidate.lstat()
    except OSError as exc:
        raise ExperimentContractError(f"T2 execution {field} is unavailable") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or not 0 < before.st_size <= maximum
    ):
        raise ExperimentContractError(f"T2 execution {field} is not bounded regular data")
    encoded = candidate.read_bytes()
    after = candidate.lstat()
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, name) != getattr(after, name) for name in identity):
        raise ExperimentContractError(f"T2 execution {field} changed while read")
    if len(encoded) != before.st_size:
        raise ExperimentContractError(f"T2 execution {field} byte count changed")
    return encoded


def _binding(root: Path, relative_path: str) -> dict[str, object]:
    encoded = _regular_bytes(root / relative_path, field=relative_path, maximum=2 * 1024**2)
    return {
        "byte_count": len(encoded),
        "path": relative_path,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _package_binding(package: str, resource: str) -> dict[str, object]:
    traversable = importlib.resources.files(package).joinpath(resource)
    path = Path(str(traversable))
    encoded = _regular_bytes(path, field=f"{package}:{resource}", maximum=2 * 1024**2)
    return {
        "byte_count": len(encoded),
        "package": package,
        "resource": resource,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def t2_host_fingerprint_value() -> dict[str, object]:
    """Record the exact local runtime identity without a host name."""

    core = {
        "dependency_versions": {name: importlib.metadata.version(name) for name in _DEPENDENCIES},
        "machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "system": platform.system(),
    }
    return {**core, "sha256": hashlib.sha256(canonical_json_bytes(core)).hexdigest()}


def _git_commit(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"{field} must be a lowercase Git object ID")
    return value


def _observed_clean_execution_commit(repository_root: Path) -> str:
    """Reuse Phase B source isolation to observe one clean execution checkout."""

    from oracle_composition.phase_b.supervision import inspect_runtime_sources

    snapshot = inspect_runtime_sources(repository_root, allow_dirty=False)
    git = snapshot.value.get("git")
    if type(git) is not dict or git.get("clean") is not True:
        raise ExperimentContractError("T2 execution tree is dirty at final admission")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    ).stdout
    if status:
        raise ExperimentContractError("T2 execution tree is dirty at final admission")
    return _git_commit(git.get("commit"), field="observed T2 execution commit")


def _source_snapshot_sha256(
    *,
    dependency_lock: Mapping[str, object],
    environment_source: Mapping[str, object],
    model: Mapping[str, object],
    sources: Mapping[str, object],
) -> str:
    sealed_source_set = {
        "dependency_lock": dict(dependency_lock),
        "environment_source": dict(environment_source),
        "model": dict(model),
        "sources": dict(sources),
    }
    return hashlib.sha256(canonical_json_bytes(sealed_source_set)).hexdigest()


def _runtime_bindings(repository_root: Path) -> dict[str, object]:
    root = Path(repository_root).resolve(strict=True)
    sources = {name: _binding(root, path) for name, path in sorted(_SOURCE_PATHS.items())}
    dependency_lock = _binding(root, "uv.lock")
    model = _package_binding("gymnasium.envs.mujoco", "assets/humanoid.xml")
    environment_source = _package_binding("gymnasium.envs.mujoco", "humanoid_v5.py")
    return {
        "dependency_lock": dependency_lock,
        "environment_source": environment_source,
        "model": model,
        "source_snapshot_sha256": _source_snapshot_sha256(
            dependency_lock=dependency_lock,
            environment_source=environment_source,
            model=model,
            sources=sources,
        ),
        "sources": sources,
    }


def _admission_record_path(relative_path: str) -> bool:
    path = PurePosixPath(relative_path)
    if (
        not relative_path
        or path.is_absolute()
        or "\\" in relative_path
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        return False
    if path.parts == ("README.md",):
        return True
    for prefix in _ADMISSION_RECORD_PREFIXES:
        if path.parts[: len(prefix)] != prefix:
            continue
        if (
            prefix == ("experiments", "004_t2_reward_study")
            and "payloads" in path.parts[len(prefix) :]
        ):
            return False
        return len(path.parts) > len(prefix)
    return False


def _descendant_changed_paths(
    repository_root: Path,
    *,
    admission_commit: str,
    execution_commit_observed: str,
) -> tuple[str, ...]:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--no-renames",
            "-z",
            f"{admission_commit}..{execution_commit_observed}",
            "--",
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    try:
        return tuple(
            item.decode("utf-8", errors="strict") for item in result.stdout.split(b"\0") if item
        )
    except UnicodeError as exc:
        raise ExperimentContractError("T2 execution descendant paths are not UTF-8") from exc


def _validate_runtime_commit(
    repository_root: Path,
    *,
    admission_commit: str,
) -> str:
    root = Path(repository_root).resolve(strict=True)
    observed = _observed_clean_execution_commit(root)
    if observed == admission_commit:
        return observed
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", admission_commit, observed],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if ancestry.returncode == 1:
        raise ExperimentContractError("T2 execution HEAD is not a descendant of admission")
    if ancestry.returncode != 0:
        raise ExperimentContractError("T2 execution ancestry cannot be verified")
    changed = _descendant_changed_paths(
        root,
        admission_commit=admission_commit,
        execution_commit_observed=observed,
    )
    refused = tuple(path for path in changed if not _admission_record_path(path))
    if refused:
        raise ExperimentContractError(
            f"T2 execution descendant changes non-admission path {refused[0]}"
        )
    return observed


def t2_execution_manifest_contract_value(
    repository_root: Path,
    *,
    admission_commit: str | None = None,
) -> dict[str, object]:
    """Record immutable sources and the clean commit observed at admission."""

    bindings = _runtime_bindings(repository_root)
    return {
        "action_abi": {
            "action_dtype": "<f4",
            "action_high": [0.4] * 17,
            "action_low": [-0.4] * 17,
            "action_shape": [17],
            "actor_input_layout": "state_float32_348_then_reference_row_major_float32_8x45/v1",
            "observation_shape": [348],
            "reference_window_shape": [8, 45],
        },
        "dependency_lock": bindings["dependency_lock"],
        "execution_manifest_id": T2_EXECUTION_MANIFEST_ID,
        "execution_manifest_schema_id": T2_EXECUTION_MANIFEST_SCHEMA_ID,
        "host_fingerprint": t2_host_fingerprint_value(),
        "mdp": {
            "control_period_seconds": 0.015,
            "environment_id": "Humanoid-v5",
            "environment_kwargs": {
                "exclude_current_positions_from_observation": True,
                "frame_skip": 5,
                "reset_noise_scale": 0.01,
                "terminate_when_unhealthy": False,
            },
            "environment_source": bindings["environment_source"],
            "model": bindings["model"],
            "time_limit_steps": 1_000,
        },
        "normalizers": {"observation": None, "reward": None},
        "repository": {
            "admission_commit": admission_commit,
            "admission_commit_verification": (
                "pending_final_admission"
                if admission_commit is None
                else "verified_clean_head_at_final_admission"
            ),
            "execution_tree_clean_at_admission": (None if admission_commit is None else True),
            "launch_base_commit": T2_EXECUTION_LAUNCH_BASE_COMMIT,
            "launch_base_semantics": "provenance_only_not_execution_commit",
        },
        "schema_version": 2,
        "source_snapshot_sha256": bindings["source_snapshot_sha256"],
        "sources": bindings["sources"],
        "status": (
            T2_EXECUTION_PENDING_STATUS
            if admission_commit is None
            else T2_EXECUTION_FINAL_READY_STATUS
        ),
        "study_id": "t2_reward_study_expert_hold/v1",
        "tracker": {
            "external_tracker_checkpoint": None,
            "tracking_reward_config": FROZEN_TRACKING_REWARD_CONFIG.to_dict(),
            "tracking_reward_config_sha256": TRACKING_REWARD_CONFIG_SHA256,
            "tracking_reward_id": TRACKING_REWARD_ID,
        },
    }


def final_ready_t2_execution_manifest_contract_value(
    repository_root: Path,
    *,
    expected_admission_commit: str,
) -> dict[str, object]:
    """Bind the actual clean HEAD as admission provenance, not a forever HEAD."""

    expected = _git_commit(expected_admission_commit, field="expected T2 admission commit")
    observed = _observed_clean_execution_commit(repository_root)
    if observed != expected:
        raise ExperimentContractError("T2 execution HEAD differs from expected admission commit")
    value = t2_execution_manifest_contract_value(
        repository_root,
        admission_commit=observed,
    )
    if _observed_clean_execution_commit(repository_root) != expected:
        raise ExperimentContractError("T2 execution tree changed while admission was sealed")
    return value


def _legacy_t2_execution_manifest_contract_value(
    repository_root: Path,
    *,
    execution_commit: str | None,
) -> dict[str, object]:
    """Recompute the retained T2C1 bytes until Fable performs the v2 re-seal."""

    bindings = _runtime_bindings(repository_root)
    return {
        "action_abi": {
            "action_dtype": "<f4",
            "action_high": [0.4] * 17,
            "action_low": [-0.4] * 17,
            "action_shape": [17],
            "actor_input_layout": "state_float32_348_then_reference_row_major_float32_8x45/v1",
            "observation_shape": [348],
            "reference_window_shape": [8, 45],
        },
        "dependency_lock": bindings["dependency_lock"],
        "execution_manifest_id": T2_EXECUTION_MANIFEST_ID,
        "execution_manifest_schema_id": T2_EXECUTION_LEGACY_MANIFEST_SCHEMA_ID,
        "host_fingerprint": t2_host_fingerprint_value(),
        "mdp": {
            "control_period_seconds": 0.015,
            "environment_id": "Humanoid-v5",
            "environment_kwargs": {
                "exclude_current_positions_from_observation": True,
                "frame_skip": 5,
                "reset_noise_scale": 0.01,
                "terminate_when_unhealthy": False,
            },
            "environment_source": bindings["environment_source"],
            "model": bindings["model"],
            "time_limit_steps": 1_000,
        },
        "normalizers": {"observation": None, "reward": None},
        "repository": {
            "execution_commit": execution_commit,
            "execution_commit_verification": (
                "pending_final_admission"
                if execution_commit is None
                else "verified_clean_head_at_final_admission"
            ),
            "execution_tree_clean_at_admission": (None if execution_commit is None else True),
            "launch_base_commit": T2_EXECUTION_LAUNCH_BASE_COMMIT,
            "launch_base_semantics": "provenance_only_not_execution_commit",
        },
        "schema_version": 1,
        "sources": bindings["sources"],
        "status": (
            T2_EXECUTION_PENDING_STATUS
            if execution_commit is None
            else T2_EXECUTION_FINAL_READY_STATUS
        ),
        "study_id": "t2_reward_study_expert_hold/v1",
        "tracker": {
            "external_tracker_checkpoint": None,
            "tracking_reward_config": FROZEN_TRACKING_REWARD_CONFIG.to_dict(),
            "tracking_reward_config_sha256": TRACKING_REWARD_CONFIG_SHA256,
            "tracking_reward_id": TRACKING_REWARD_ID,
        },
    }


def _validate_t2_execution_manifest_runtime(
    value: Mapping[str, object],
    *,
    repository_root: Path,
) -> tuple[dict[str, object], str | None, str | None]:
    repository = value.get("repository") if type(value) is dict else None
    schema_id = value.get("execution_manifest_schema_id") if type(value) is dict else None
    if schema_id == T2_EXECUTION_LEGACY_MANIFEST_SCHEMA_ID:
        execution_commit = (
            repository.get("execution_commit") if type(repository) is dict else object()
        )
        if execution_commit is None:
            expected = _legacy_t2_execution_manifest_contract_value(
                repository_root, execution_commit=None
            )
            admission_commit = None
            observed = None
        else:
            admission_commit = _git_commit(
                execution_commit, field="legacy bound T2 admission commit"
            )
            observed = _validate_runtime_commit(
                repository_root,
                admission_commit=admission_commit,
            )
            expected = _legacy_t2_execution_manifest_contract_value(
                repository_root,
                execution_commit=admission_commit,
            )
    else:
        admission_value = (
            repository.get("admission_commit") if type(repository) is dict else object()
        )
        if admission_value is None:
            expected = t2_execution_manifest_contract_value(repository_root)
            admission_commit = None
            observed = None
        else:
            admission_commit = _git_commit(admission_value, field="bound T2 admission commit")
            observed = _validate_runtime_commit(
                repository_root,
                admission_commit=admission_commit,
            )
            expected = t2_execution_manifest_contract_value(
                repository_root,
                admission_commit=admission_commit,
            )
    if type(value) is not dict or value != expected:
        raise ExperimentContractError("T2 execution manifest semantics or bindings differ")
    canonical_json_bytes(dict(value))
    return dict(value), admission_commit, observed


def validate_t2_execution_manifest(
    value: Mapping[str, object],
    *,
    repository_root: Path,
) -> dict[str, object]:
    """Recompute bindings and verify a final manifest against the live clean HEAD."""

    validated, _admission_commit, _observed = _validate_t2_execution_manifest_runtime(
        value,
        repository_root=repository_root,
    )
    return validated


def t2_runtime_execution_identity(
    value: Mapping[str, object],
    *,
    repository_root: Path,
) -> dict[str, str]:
    """Return the two commit identities a runtime receipt must record together."""

    if value.get("execution_manifest_schema_id") != T2_EXECUTION_MANIFEST_SCHEMA_ID:
        raise ExperimentContractError("T2 runtime requires the v2 descendant-aware re-seal")
    _validated, admission_commit, observed = _validate_t2_execution_manifest_runtime(
        value,
        repository_root=repository_root,
    )
    if admission_commit is None or observed is None:
        raise ExperimentContractError("pending T2 execution manifest cannot authorize runtime")
    return {
        "admission_commit": admission_commit,
        "execution_commit_observed": observed,
    }


def load_t2_execution_manifest(
    path: Path,
    *,
    repository_root: Path,
) -> tuple[dict[str, object], str]:
    candidate = Path(path)
    encoded = _regular_bytes(candidate, field="manifest", maximum=256 * 1024)
    try:
        value = json.loads(encoded)
    except (UnicodeError, ValueError) as exc:
        raise ExperimentContractError("T2 execution manifest is not JSON") from exc
    if type(value) is not dict or canonical_json_bytes(value) != encoded:
        raise ExperimentContractError("T2 execution manifest is not canonical JSON")
    validated = validate_t2_execution_manifest(value, repository_root=repository_root)
    return validated, hashlib.sha256(encoded).hexdigest()


__all__ = [
    "T2_EXECUTION_BASE_COMMIT",
    "T2_EXECUTION_FINAL_READY_STATUS",
    "T2_EXECUTION_LAUNCH_BASE_COMMIT",
    "T2_EXECUTION_LEGACY_MANIFEST_SCHEMA_ID",
    "T2_EXECUTION_MANIFEST_ID",
    "T2_EXECUTION_MANIFEST_SCHEMA_ID",
    "T2_EXECUTION_PENDING_STATUS",
    "T2_EXECUTION_STATUS",
    "final_ready_t2_execution_manifest_contract_value",
    "load_t2_execution_manifest",
    "t2_execution_manifest_contract_value",
    "t2_host_fingerprint_value",
    "t2_runtime_execution_identity",
    "validate_t2_execution_manifest",
]
