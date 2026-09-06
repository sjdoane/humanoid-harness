"""Data-only admission for one bounded G1 course development run."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oracle_composition.harness.contract import (
    OracleProgram,
    oracle_program_from_dict,
    read_json_object,
)

from .checkpoint import verify_upstream_root
from .composition import ReferenceSegment
from .contracts import MOTION_SPECS
from .course_task import CourseTaskSpec, TaskRewardRecipe
from .io import sha256_file
from .reference_runtime import ReferenceMotion

ACTOR_SHA256 = "bc444fbd56ba4a582d7c6367504f2093ccb081c6956fcee8f30f2e85ced28686"
CONFIG_KEYS = {
    "schema_version",
    "mode",
    "assets",
    "task",
    "oracle",
    "segments",
    "reward",
    "seed",
    "training_steps",
}


def _keys(value: object, expected: set[str], field: str) -> dict:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{field} fields differ")
    return value


def _asset(value: object) -> Path:
    asset = _keys(value, {"path", "sha256"}, "numeric asset")
    if type(asset["path"]) is not str or type(asset["sha256"]) is not str:
        raise ValueError("asset path and digest must be strings")
    path = Path(asset["path"])
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("numeric asset must be an absolute regular non-linked file")
    if path.suffix != ".npz" or sha256_file(path) != asset["sha256"]:
        raise ValueError("numeric asset digest or format differs")
    return path


@dataclass(frozen=True)
class CourseRunConfig:
    raw: dict[str, Any]
    encoded: bytes
    sha256: str
    assets: dict[str, Any]
    task: CourseTaskSpec
    recipe: TaskRewardRecipe
    program: OracleProgram
    segments: dict[str, ReferenceSegment]


def load_run_config(path: Path) -> CourseRunConfig:
    raw, encoded = read_json_object(path)
    _keys(raw, CONFIG_KEYS, "course run")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("course run schema version differs")
    if raw["mode"] not in {"probe", "train"}:
        raise ValueError("course mode must be probe or train")
    steps = raw["training_steps"]
    if type(steps) is not int or (
        (raw["mode"] == "probe" and steps != 0)
        or (raw["mode"] == "train" and not (512 <= steps <= 262_144 and steps % 512 == 0))
    ):
        raise ValueError("training steps must match the bounded mode and rollout size")
    if type(raw["seed"]) is not int or not 0 <= raw["seed"] < 2**31:
        raise ValueError("seed must be a nonnegative signed 32-bit integer")
    assets = _keys(raw["assets"], {"upstream_root", "weights", "motions"}, "assets")
    if type(assets["upstream_root"]) is not str or not Path(assets["upstream_root"]).is_absolute():
        raise ValueError("upstream_root must be an absolute path")
    verify_upstream_root(Path(assets["upstream_root"]))
    _asset(assets["weights"])
    if assets["weights"]["sha256"] != ACTOR_SHA256:
        raise ValueError("this development family freezes the admitted GMT actor")
    motions = assets["motions"]
    segments = raw["segments"]
    if type(motions) is not dict or not 1 <= len(motions) <= 8:
        raise ValueError("one to eight admitted motion records are required")
    if type(segments) is not dict or not 1 <= len(segments) <= 8:
        raise ValueError("one to eight segment records are required")
    loaded = {}
    for name, asset in motions.items():
        if name not in MOTION_SPECS:
            raise ValueError("motion name is outside the admitted GMT library")
        loaded[name] = ReferenceMotion.from_converted(
            _asset(asset), name=name, expected_sha256=asset["sha256"]
        )
    admitted = {}
    for behavior, value in segments.items():
        required = {"motion_name", "start_seconds", "end_seconds"}
        optional = {"entry_phase_end_seconds", "boundary"}
        if type(value) is not dict or not required <= set(value) <= required | optional:
            raise ValueError("segment fields differ")
        segment = value
        if segment["motion_name"] not in loaded:
            raise ValueError("segment refers to an unadmitted motion")
        if any(type(segment[key]) is not float for key in ("start_seconds", "end_seconds")):
            raise ValueError("segment bounds must be floats")
        name = segment["motion_name"]
        admitted[behavior] = ReferenceSegment(
            loaded[name],
            motions[name]["sha256"],
            segment["start_seconds"],
            segment["end_seconds"],
            entry_phase_end_seconds=segment.get("entry_phase_end_seconds"),
            boundary=segment.get("boundary", "wrap_within_segment"),
        )
    if {segment["motion_name"] for segment in segments.values()} != set(motions):
        raise ValueError("unused assets must not masquerade as consumed references")
    program = oracle_program_from_dict(raw["oracle"], available_behaviors=list(admitted))
    if set(program.states) != {"before", "inside", "after"}:
        raise ValueError("all course arms require the same three state slots")
    task = CourseTaskSpec.from_dict(raw["task"])
    if task.horizon_steps > 2_000:
        raise ValueError("development evaluation is bounded to forty simulated seconds")
    return CourseRunConfig(
        raw,
        encoded,
        hashlib.sha256(encoded).hexdigest(),
        assets,
        task,
        TaskRewardRecipe.from_dict(raw["reward"]),
        program,
        admitted,
    )
