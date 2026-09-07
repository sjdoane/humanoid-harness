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
from .course_runtime import (
    LEGACY_RUNTIME,
    CourseRuntimeProfile,
    runtime_profile_from_config,
)
from .course_task import CourseTaskSpec, TaskRewardRecipe
from .io import sha256_file
from .reference_runtime import ReferenceMotion
from .training_contract import CourseTrainerSpec

ACTOR_SHA256 = "bc444fbd56ba4a582d7c6367504f2093ccb081c6956fcee8f30f2e85ced28686"
ADMITTED_CONVERTED_MOTION_SHA256 = {
    "airkick_stand": "c619dba5b25730178f92570d9368e95030bc024b8068012dba0f1e04770b6bc8",
    "basic_walk": "b6ee3143e61b308daebb2a1f07d8b420459a2d19cbde76ecc8f4d42affcdb7b1",
    "crouchwalk_stand": "a67c364e7d0013c24f43096e194f5d8099b2d1ebcaa3db42767402a0924be964",
    "dance": "71d8e63d58ff41d433f406a19b15ca836d3f6977b9e73a67d719110af527b25f",
    "dance_waltz": "5d6b521099f6a5242eda7a5cb5f0dea87f97501cb445501b525c865dd18c16ef",
    "kick_walk": "b8a7e75a700af7a32d9dd3b51299a771a45bd4c603a7f973ba3d554a03273709",
    "squat": "6f5707a4a3b86ca8d476ce1c3aee9ca812cb9e4cb76c8f3a0cdaacf57ed9ad6b",
    "walk_stand": "908ba0e4f6acf1ecf829b0ddb73ed7e649ba6e7ca9fa1a96b43a95e0fdc2e3b0",
}
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
CONFIG_KEYS_WITH_TRAINER = {*CONFIG_KEYS, "trainer"}
CONFIG_KEYS_WITH_RUNTIME = {*CONFIG_KEYS, "runtime"}
CONFIG_KEYS_WITH_RUNTIME_AND_TRAINER = {*CONFIG_KEYS_WITH_RUNTIME, "trainer"}


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
    trainer: CourseTrainerSpec | None = None
    runtime: CourseRuntimeProfile = LEGACY_RUNTIME


def load_run_config(path: Path) -> CourseRunConfig:
    raw, encoded = read_json_object(path)
    if type(raw) is not dict:
        raise ValueError("course run fields differ")
    runtime = runtime_profile_from_config(raw)
    valid_fields = (
        {
            frozenset(CONFIG_KEYS_WITH_RUNTIME),
            frozenset(CONFIG_KEYS_WITH_RUNTIME_AND_TRAINER),
        }
        if runtime.config_value is not None
        else {frozenset(CONFIG_KEYS), frozenset(CONFIG_KEYS_WITH_TRAINER)}
    )
    if frozenset(raw) not in valid_fields:
        raise ValueError("course run fields differ")
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
    trainer = CourseTrainerSpec.from_dict(raw["trainer"]) if "trainer" in raw else None
    if trainer is not None and raw["mode"] != "train":
        raise ValueError("trainer preconditioning is valid only for train mode")
    if not runtime.training_admitted and raw["mode"] != "probe":
        raise ValueError("course runtime profile is probe-only until feasibility is measured")
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
        if type(asset) is not dict or asset.get("sha256") != ADMITTED_CONVERTED_MOTION_SHA256[name]:
            raise ValueError("motion bytes are outside the admitted converted GMT library")
        loaded[name] = ReferenceMotion.from_converted(
            _asset(asset),
            name=name,
            expected_sha256=ADMITTED_CONVERTED_MOTION_SHA256[name],
        )
    admitted = {}
    for behavior, value in segments.items():
        required = {"motion_name", "start_seconds", "end_seconds"}
        optional = {
            "entry_phase_end_seconds",
            "boundary",
            "loop_start_seconds",
            "exit_at_loop_boundary",
        }
        if type(value) is not dict or not required <= set(value) <= required | optional:
            raise ValueError("segment fields differ")
        segment = value
        if segment["motion_name"] not in loaded:
            raise ValueError("segment refers to an unadmitted motion")
        if any(type(segment[key]) is not float for key in ("start_seconds", "end_seconds")):
            raise ValueError("segment bounds must be floats")
        if "loop_start_seconds" in segment and type(segment["loop_start_seconds"]) is not float:
            raise ValueError("segment loop start must be a float")
        if "exit_at_loop_boundary" in segment and segment["exit_at_loop_boundary"] is not True:
            raise ValueError("segment loop-boundary exit must be exactly true when declared")
        if not runtime.permits_entry_loop and (
            "loop_start_seconds" in segment
            or "exit_at_loop_boundary" in segment
            or segment.get("boundary") == "entry_once_then_loop"
        ):
            raise ValueError("legacy runtime prohibits entry-loop segment semantics")
        name = segment["motion_name"]
        admitted[behavior] = ReferenceSegment(
            loaded[name],
            motions[name]["sha256"],
            segment["start_seconds"],
            segment["end_seconds"],
            entry_phase_end_seconds=segment.get("entry_phase_end_seconds"),
            boundary=segment.get("boundary", "wrap_within_segment"),
            loop_start_seconds=segment.get("loop_start_seconds"),
            exit_at_loop_boundary=segment.get("exit_at_loop_boundary", False),
        )
    if {segment["motion_name"] for segment in segments.values()} != set(motions):
        raise ValueError("unused assets must not masquerade as consumed references")
    program = oracle_program_from_dict(raw["oracle"], available_behaviors=list(admitted))
    runtime.validate_program(program)
    task = CourseTaskSpec.from_dict(raw["task"])
    if task.horizon_steps > 2_000:
        raise ValueError("development evaluation is bounded to forty simulated seconds")
    return CourseRunConfig(
        raw=raw,
        encoded=encoded,
        sha256=hashlib.sha256(encoded).hexdigest(),
        assets=assets,
        task=task,
        recipe=TaskRewardRecipe.from_dict(raw["reward"]),
        program=program,
        segments=admitted,
        trainer=trainer,
        runtime=runtime,
    )
