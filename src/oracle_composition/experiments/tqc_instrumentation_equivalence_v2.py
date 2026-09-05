"""Exact preflight dynamics-equivalence canary for TQC v2 evaluation.

The receipt proves only that the contact-instrumented evaluation environment
matched the plain training environment on one frozen action trace. It does not
exercise the reference-corpus collector or its boundary-capture helper. It is
not controller or locomotion evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time
from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from types import CodeType
from typing import Any

import numpy as np

from oracle_composition.envs import humanoid as humanoid_module
from oracle_composition.envs.humanoid import (
    CONTACT_CAPTURE_ID,
    HumanoidExperimentConfig,
    SubstepContactSample,
    _instrumented_humanoid_class,
    make_humanoid_env,
)

from .fixed_reference import ExperimentContractError, sha256_file
from .runtime_identity import module_sha256, source_tree_sha256
from .tqc_calibration_contract import canonical_json
from .tqc_development_contract_v2 import LoadedTQCDevelopmentDesignV2

INSTRUMENTATION_RECEIPT_ID = "tqc_dev_1m_v2_instrumentation_equivalence/v1"
INSTRUMENTATION_SEED = 98_001
INSTRUMENTATION_HORIZON = 1_000
EXPECTED_ACTION_SHA256 = "8cbe0bc01c56134fdab90ba6506ebd74622f4929eb3216d332c05e459fa63af4"
INSTRUMENTATION_SCOPE = "preflight_canary_only_not_complete_dynamics_equivalence_evidence"
CLAIM_CEILING = "instrumentation_equivalence_only_no_controller_or_behavior_claim/v1"
MAX_RECEIPT_BYTES = 64 * 1024
ATTEMPT_ID = "dev1m-v2-seed-95001-attempt-01"
EXPECTED_SHARED_TRACE_SHA256 = "ecbf4e76191a059e6464070cab7b1f9c9900584c377039cb836b1a05264c55be"
EXPECTED_STRUCTURAL_SOURCE_PROOF_SHA256 = (
    "d189f503572937bc78812c3f030c31598039fa096866b0fd8d7313c36bc5ffbf"
)
EXPECTED_CONTACT_TRACE_SHA256 = "0a2bdf2c4a4715e8ab2eb37b5297bceb9d38bb168ad44e8ce04aba8e512f9a5f"
EXPECTED_CONTACT_SAMPLE_COUNT = 13_411
COMPARISON_FIELDS = (
    "returned_observation",
    "reward",
    "terminated",
    "truncated",
    "mjSTATE_INTEGRATION",
    "qacc",
    "qfrc_actuator",
    "simulation_time",
    "TimeLimit_elapsed_steps",
)

_RECEIPT_ISSUER = object()
_BINDING_KEYS = {
    "attempt_id",
    "attempt_nonce",
    "worker_pid",
    "worker_process_start_monotonic_seconds",
    "preflight_contract_sha256",
    "claimed_work_directory_identity",
    "project_python_source_tree_sha256",
}
_RECEIPT_KEYS = {
    "schema_version",
    "receipt_id",
    "design_file_sha256",
    "design_semantic_sha256",
    "training_projection_sha256",
    "environment_instrumentation_equivalence_source_sha256",
    *_BINDING_KEYS,
    "seed",
    "horizon_steps",
    "action_shape",
    "action_sha256",
    "reset_comparison_count",
    "per_step_comparison_count",
    "comparison_fields",
    "structural_source_proof_sha256",
    "contact_trace_sha256",
    "contact_sample_count",
    "plain_shared_trace_sha256",
    "instrumented_shared_trace_sha256",
    "mismatch_count",
    "first_mismatch",
    "all_comparisons_passed",
    "scope",
    "claim_ceiling",
    "behavioral_evidence",
}


def _sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, *, field_name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ExperimentContractError(f"{field_name} must be a lowercase SHA-256")
    return value


def _require_exact(value: object, expected: object, *, field_name: str) -> None:
    if type(value) is not type(expected):
        raise ExperimentContractError(f"{field_name} differs")
    if isinstance(expected, dict):
        if set(value) != set(expected):  # type: ignore[arg-type]
            raise ExperimentContractError(f"{field_name} keys differ")
        for key, child in expected.items():
            _require_exact(value[key], child, field_name=f"{field_name}.{key}")  # type: ignore[index]
        return
    if isinstance(expected, (list, tuple)):
        if len(value) != len(expected):  # type: ignore[arg-type]
            raise ExperimentContractError(f"{field_name} length differs")
        for index, child in enumerate(expected):
            _require_exact(value[index], child, field_name=f"{field_name}[{index}]")  # type: ignore[index]
        return
    if value != expected:
        raise ExperimentContractError(f"{field_name} differs")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExperimentContractError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> None:
    raise ExperimentContractError(f"non-finite JSON value {value!r}")


def _decode_receipt(encoded: bytes) -> dict[str, Any]:
    if type(encoded) is not bytes or not encoded or len(encoded) > MAX_RECEIPT_BYTES:
        raise ExperimentContractError("instrumentation receipt bytes are absent or too large")
    try:
        value = json.loads(
            encoded.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except ExperimentContractError:
        raise
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ExperimentContractError("instrumentation receipt JSON is invalid") from exc
    if type(value) is not dict or canonical_json(value) != encoded:
        raise ExperimentContractError("instrumentation receipt is not canonical JSON")
    return value


def _validate_binding(
    value: Mapping[str, object],
    *,
    expected: Mapping[str, object] | None,
    require_current_worker: bool,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != _BINDING_KEYS:
        raise ExperimentContractError("instrumentation receipt binding keys differ")
    _require_exact(value["attempt_id"], ATTEMPT_ID, field_name="attempt_id")
    for field_name in (
        "attempt_nonce",
        "preflight_contract_sha256",
        "claimed_work_directory_identity",
        "project_python_source_tree_sha256",
    ):
        _require_sha256(value[field_name], field_name=field_name)
    worker_pid = value["worker_pid"]
    worker_start = value["worker_process_start_monotonic_seconds"]
    if type(worker_pid) is not int or worker_pid <= 1:
        raise ExperimentContractError("instrumentation worker PID is invalid")
    if type(worker_start) is not float or not math.isfinite(worker_start) or worker_start < 0:
        raise ExperimentContractError("instrumentation worker process start is invalid")
    if require_current_worker and worker_pid != os.getpid():
        raise ExperimentContractError("instrumentation worker PID is not this process")
    if require_current_worker and worker_start > time.perf_counter():
        raise ExperimentContractError("instrumentation worker process start is in the future")
    if expected is not None:
        _require_exact(dict(value), dict(expected), field_name="preflight binding")
    return dict(value)


def _canonical_array_sha256(value: np.ndarray) -> str:
    if not isinstance(value, np.ndarray) or not value.flags.c_contiguous:
        raise ExperimentContractError("instrumentation array must use C-order storage")
    metadata = canonical_json({"dtype": value.dtype.str, "shape": list(value.shape)})
    raw = value.tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(len(metadata).to_bytes(8, "big"))
    digest.update(metadata)
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)
    return digest.hexdigest()


def tqc_instrumentation_actions_v2() -> np.ndarray:
    """Return the frozen physical-action canary without consulting an actor."""

    generator = np.random.Generator(np.random.PCG64(INSTRUMENTATION_SEED))
    actions = np.ascontiguousarray(
        generator.uniform(-0.4, 0.4, size=(INSTRUMENTATION_HORIZON, 17)).astype("<f4")
    )
    if _canonical_array_sha256(actions) != EXPECTED_ACTION_SHA256:
        raise ExperimentContractError("frozen instrumentation actions differ")
    return actions


def _frame(digest: Any, label: str, payload: bytes) -> None:
    label_bytes = label.encode("utf-8")
    digest.update(len(label_bytes).to_bytes(8, "big"))
    digest.update(label_bytes)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _trace_value(digest: Any, *, label: str, value: object) -> None:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        header = canonical_json({"dtype": array.dtype.str, "shape": list(array.shape)})
        _frame(digest, f"{label}/array-header", header)
        _frame(digest, f"{label}/array-bytes", array.tobytes(order="C"))
        return
    if type(value) is bool:
        _frame(digest, f"{label}/bool", b"\x01" if value else b"\x00")
        return
    if type(value) is int:
        _frame(digest, f"{label}/int64be", int(value).to_bytes(8, "big", signed=True))
        return
    if type(value) is float:
        _frame(digest, f"{label}/float64be", struct.pack(">d", value))
        return
    raise ExperimentContractError(f"unsupported trace value for {label}")


def _exact_equal(left: object, right: object) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right)
        )
    return type(left) is type(right) and left == right


def _time_limit_counter(environment: object) -> int:
    import gymnasium as gym

    if type(environment) is not gym.wrappers.TimeLimit:
        raise ExperimentContractError("instrumentation environment lacks exact TimeLimit wrapper")
    value = getattr(environment, "_elapsed_steps", None)
    if type(value) is not int or value < 0:
        raise ExperimentContractError("TimeLimit elapsed-step counter is invalid")
    return value


def _integration_state(environment: object) -> np.ndarray:
    import mujoco

    physical = environment.unwrapped
    flag = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(int(mujoco.mj_stateSize(physical.model, flag)), dtype="<f8")
    mujoco.mj_getState(physical.model, physical.data, state, flag)
    return np.ascontiguousarray(state)


def _boundary(
    environment: object,
    *,
    observation: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
) -> dict[str, object]:
    if (
        not isinstance(observation, np.ndarray)
        or observation.shape != (348,)
        or observation.dtype != np.dtype("<f8")
        or not observation.flags.c_contiguous
        or not np.isfinite(observation).all()
        or isinstance(reward, (bool, np.bool_))
        or not isinstance(reward, (float, np.floating))
        or not math.isfinite(float(reward))
        or type(terminated) is not bool
        or type(truncated) is not bool
    ):
        raise ExperimentContractError("instrumentation step return contract differs")
    physical = environment.unwrapped
    return {
        "returned_observation": np.ascontiguousarray(observation),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "mjSTATE_INTEGRATION": _integration_state(environment),
        "qacc": np.ascontiguousarray(physical.data.qacc.copy()),
        "qfrc_actuator": np.ascontiguousarray(physical.data.qfrc_actuator.copy()),
        "simulation_time": float(physical.data.time),
        "TimeLimit_elapsed_steps": _time_limit_counter(environment),
    }


def _reset_boundary(environment: object, observation: np.ndarray) -> dict[str, object]:
    if (
        not isinstance(observation, np.ndarray)
        or observation.shape != (348,)
        or observation.dtype != np.dtype("<f8")
        or not observation.flags.c_contiguous
        or not np.isfinite(observation).all()
    ):
        raise ExperimentContractError("instrumentation reset observation contract differs")
    return {
        "returned_observation": np.ascontiguousarray(observation),
        "mjSTATE_INTEGRATION": _integration_state(environment),
        "TimeLimit_elapsed_steps": _time_limit_counter(environment),
    }


def _code_value(value: object) -> object:
    if isinstance(value, CodeType):
        return _code_receipt(value)
    if type(value) in {type(None), bool, int, float, str}:
        return value
    if type(value) is bytes:
        return {"bytes_hex": value.hex()}
    if type(value) is tuple:
        return [_code_value(child) for child in value]
    raise ExperimentContractError(
        f"instrumented method contains unsupported code constant {type(value).__name__}"
    )


def _code_receipt(code: CodeType) -> dict[str, object]:
    return {
        "argcount": code.co_argcount,
        "posonlyargcount": code.co_posonlyargcount,
        "kwonlyargcount": code.co_kwonlyargcount,
        "nlocals": code.co_nlocals,
        "stacksize": code.co_stacksize,
        "flags": code.co_flags,
        "bytecode_hex": code.co_code.hex(),
        "constants": [_code_value(value) for value in code.co_consts],
        "names": list(code.co_names),
        "varnames": list(code.co_varnames),
        "freevars": list(code.co_freevars),
        "cellvars": list(code.co_cellvars),
    }


def _member_code_sha256(member: object) -> str:
    function = member.fget if type(member) is property else member
    code = getattr(function, "__code__", None)
    if type(code) is not CodeType:
        raise ExperimentContractError("instrumented Humanoid member has no exact Python code")
    return _sha256(canonical_json(_code_receipt(code)))


def _structural_source_proof(subclass: type[object]) -> str:
    from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv

    expected_subclass = _instrumented_humanoid_class()
    metadata_members = {
        "__doc__",
        "__firstlineno__",
        "__module__",
        "__parameters__",
        "__static_attributes__",
    }
    authored_members = set(subclass.__dict__) - metadata_members
    expected_members = {
        "__init__",
        "contact_capture_id",
        "_step_mujoco_simulation",
        "last_control_step_contact_samples",
        "last_control_step_substeps",
        "reset",
    }
    overridden_base_members = {
        name
        for name in authored_members
        if any(name in ancestor.__dict__ for ancestor in subclass.__mro__[1:])
    }
    expected_overrides = {"__init__", "_step_mujoco_simulation", "reset"}
    if (
        subclass is not expected_subclass
        or subclass.__bases__ != (HumanoidEnv,)
        or authored_members != expected_members
        or overridden_base_members != expected_overrides
        or subclass.contact_capture_id != CONTACT_CAPTURE_ID
        or type(subclass.__dict__["last_control_step_contact_samples"]) is not property
        or type(subclass.__dict__["last_control_step_substeps"]) is not property
    ):
        raise ExperimentContractError("instrumented Humanoid structural source proof differs")
    payload = {
        "environment_source_sha256": sha256_file(Path(str(humanoid_module.__file__))),
        "gym_humanoid_source_sha256": module_sha256(
            __import__("gymnasium.envs.mujoco.humanoid_v5", fromlist=["HumanoidEnv"])
        ),
        "subclass_module": subclass.__module__,
        "subclass_qualname": subclass.__qualname__,
        "exact_base": f"{HumanoidEnv.__module__}.{HumanoidEnv.__name__}",
        "authored_members": sorted(authored_members),
        "overridden_base_members": sorted(overridden_base_members),
        "contact_capture_id": CONTACT_CAPTURE_ID,
        "member_code_sha256": {
            name: _member_code_sha256(subclass.__dict__[name])
            for name in (
                "__init__",
                "last_control_step_contact_samples",
                "last_control_step_substeps",
                "reset",
                "_step_mujoco_simulation",
            )
        },
    }
    observed = _sha256(canonical_json(payload))
    if observed != EXPECTED_STRUCTURAL_SOURCE_PROOF_SHA256:
        raise ExperimentContractError("instrumented Humanoid live method implementation differs")
    return observed


def _validated_contact_record(environment: object, *, step_index: int) -> dict[str, object]:
    physical = environment.unwrapped
    if type(step_index) is not int or step_index < 1:
        raise ExperimentContractError("instrumentation contact step index differs")
    if physical.last_control_step_substeps != physical.frame_skip:
        raise ExperimentContractError("instrumentation omitted a MuJoCo physics substep")
    samples = physical.last_control_step_contact_samples
    if type(samples) is not tuple:
        raise ExperimentContractError("instrumentation contact record container differs")
    previous = (-1, -1)
    expected_indices = [0] * physical.frame_skip
    records: list[dict[str, object]] = []
    for sample in samples:
        current = (sample.physics_substep_index, sample.contact_index_within_substep)
        if (
            type(sample) is not SubstepContactSample
            or type(sample.physics_substep_index) is not int
            or not 0 <= sample.physics_substep_index < physical.frame_skip
            or type(sample.contact_index_within_substep) is not int
            or sample.contact_index_within_substep < 0
            or sample.contact_index_within_substep != expected_indices[sample.physics_substep_index]
            or current <= previous
            or type(sample.geom1_id) is not int
            or not 0 <= sample.geom1_id < physical.model.ngeom
            or type(sample.geom2_id) is not int
            or not 0 <= sample.geom2_id < physical.model.ngeom
            or type(sample.normal_force_n) is not float
            or not math.isfinite(sample.normal_force_n)
            or sample.normal_force_n < 0.0
        ):
            raise ExperimentContractError("instrumented contact records are invalid or unordered")
        expected_indices[sample.physics_substep_index] += 1
        previous = current
        records.append(
            {
                "physics_substep_index": sample.physics_substep_index,
                "contact_index_within_substep": sample.contact_index_within_substep,
                "geom1_id": sample.geom1_id,
                "geom2_id": sample.geom2_id,
                "normal_force_n": sample.normal_force_n,
            }
        )
    return {
        "step_index": step_index,
        "captured_substeps": physical.last_control_step_substeps,
        "contact_count": len(records),
        "contacts": records,
    }


def _validate_payload(
    value: dict[str, Any],
    design: LoadedTQCDevelopmentDesignV2 | None,
    expected_binding: Mapping[str, object] | None = None,
) -> None:
    if set(value) != _RECEIPT_KEYS:
        raise ExperimentContractError("instrumentation receipt keys differ")
    expected = {
        "schema_version": 1,
        "receipt_id": INSTRUMENTATION_RECEIPT_ID,
        "seed": INSTRUMENTATION_SEED,
        "horizon_steps": INSTRUMENTATION_HORIZON,
        "action_shape": [INSTRUMENTATION_HORIZON, 17],
        "action_sha256": EXPECTED_ACTION_SHA256,
        "comparison_fields": list(COMPARISON_FIELDS),
        "structural_source_proof_sha256": value["structural_source_proof_sha256"],
        "contact_trace_sha256": EXPECTED_CONTACT_TRACE_SHA256,
        "contact_sample_count": EXPECTED_CONTACT_SAMPLE_COUNT,
        "scope": INSTRUMENTATION_SCOPE,
        "claim_ceiling": CLAIM_CEILING,
        "behavioral_evidence": False,
    }
    for field_name, expected_value in expected.items():
        _require_exact(value[field_name], expected_value, field_name=field_name)
    _validate_binding(
        {field_name: value[field_name] for field_name in _BINDING_KEYS},
        expected=expected_binding,
        require_current_worker=False,
    )
    for field_name in (
        "environment_instrumentation_equivalence_source_sha256",
        "design_file_sha256",
        "design_semantic_sha256",
        "training_projection_sha256",
        "structural_source_proof_sha256",
        "contact_trace_sha256",
        "plain_shared_trace_sha256",
        "instrumented_shared_trace_sha256",
    ):
        _require_sha256(value[field_name], field_name=field_name)
    _require_exact(
        value["environment_instrumentation_equivalence_source_sha256"],
        sha256_file(Path(__file__)),
        field_name="instrumentation source SHA-256",
    )
    _require_exact(
        value["structural_source_proof_sha256"],
        _structural_source_proof(_instrumented_humanoid_class()),
        field_name="structural source proof SHA-256",
    )
    if design is not None:
        if type(design) is not LoadedTQCDevelopmentDesignV2:
            raise ExperimentContractError("instrumentation receipt requires strict-loaded design")
        for field_name, expected_value in (
            ("design_file_sha256", design.file_sha256),
            ("design_semantic_sha256", design.semantic_sha256),
            ("training_projection_sha256", design.training_projection_sha256),
        ):
            _require_exact(value[field_name], expected_value, field_name=field_name)
    for field_name, maximum in (
        ("reset_comparison_count", 1),
        ("per_step_comparison_count", INSTRUMENTATION_HORIZON),
        ("mismatch_count", len(COMPARISON_FIELDS) * INSTRUMENTATION_HORIZON + 4),
    ):
        observed = value[field_name]
        if type(observed) is not int or not 0 <= observed <= maximum:
            raise ExperimentContractError(f"{field_name} is outside its bound")
    passed = value["all_comparisons_passed"]
    if type(passed) is not bool:
        raise ExperimentContractError("all_comparisons_passed must be boolean")
    if passed:
        if (
            value["reset_comparison_count"] != 1
            or value["per_step_comparison_count"] != INSTRUMENTATION_HORIZON
            or value["mismatch_count"] != 0
            or value["first_mismatch"] is not None
            or value["plain_shared_trace_sha256"] != value["instrumented_shared_trace_sha256"]
            or value["plain_shared_trace_sha256"] != EXPECTED_SHARED_TRACE_SHA256
        ):
            raise ExperimentContractError("passing instrumentation receipt is inconsistent")
    elif (
        value["mismatch_count"] < 1
        or type(value["first_mismatch"]) is not dict
        or set(value["first_mismatch"]) != {"boundary_index", "field"}
    ):
        raise ExperimentContractError("failed instrumentation receipt is inconsistent")
    elif (
        type(value["first_mismatch"]["boundary_index"]) is not int
        or not 0 <= value["first_mismatch"]["boundary_index"] <= INSTRUMENTATION_HORIZON
        or type(value["first_mismatch"]["field"]) is not str
        or value["first_mismatch"]["field"] not in {*COMPARISON_FIELDS, "terminal_contract"}
    ):
        raise ExperimentContractError("failed instrumentation mismatch record is invalid")


@dataclass(frozen=True, slots=True)
class TQCInstrumentationEquivalenceReceiptV2:
    """Validated canary content; it never authorizes execution or behavior."""

    canonical_bytes: bytes = field(repr=False)
    sha256: str
    seed: int
    horizon_steps: int
    action_sha256: str
    all_comparisons_passed: bool
    _issuer: InitVar[object] = None

    def __post_init__(self, _issuer: object) -> None:
        if _issuer is not _RECEIPT_ISSUER:
            raise ExperimentContractError(
                "instrumentation receipts may only be issued by exact validation"
            )
        value = _decode_receipt(self.canonical_bytes)
        _validate_payload(value, None)
        _require_exact(_sha256(self.canonical_bytes), self.sha256, field_name="receipt SHA-256")
        _require_exact(value["seed"], self.seed, field_name="receipt seed authority")
        _require_exact(value["horizon_steps"], self.horizon_steps, field_name="receipt horizon")
        _require_exact(value["action_sha256"], self.action_sha256, field_name="receipt action")
        _require_exact(
            value["all_comparisons_passed"],
            self.all_comparisons_passed,
            field_name="receipt outcome authority",
        )

    def to_dict(self) -> dict[str, Any]:
        return _decode_receipt(self.canonical_bytes)

    @property
    def authorizes_execution(self) -> bool:
        return False


def validate_tqc_instrumentation_equivalence_receipt_v2(
    encoded_bytes: bytes,
    design: LoadedTQCDevelopmentDesignV2,
    *,
    expected_binding: Mapping[str, object],
) -> TQCInstrumentationEquivalenceReceiptV2:
    """Validate worker canary content against an independently known binding."""

    if type(design) is not LoadedTQCDevelopmentDesignV2:
        raise ExperimentContractError("instrumentation receipt requires strict-loaded design")
    value = _decode_receipt(encoded_bytes)
    _validate_payload(value, design, expected_binding)
    return TQCInstrumentationEquivalenceReceiptV2(
        canonical_bytes=encoded_bytes,
        sha256=_sha256(encoded_bytes),
        seed=value["seed"],
        horizon_steps=value["horizon_steps"],
        action_sha256=value["action_sha256"],
        all_comparisons_passed=value["all_comparisons_passed"],
        _issuer=_RECEIPT_ISSUER,
    )


def run_tqc_instrumentation_equivalence_v2(
    design: LoadedTQCDevelopmentDesignV2,
    *,
    receipt_binding: Mapping[str, object],
) -> TQCInstrumentationEquivalenceReceiptV2:
    """Run the frozen 1,000-step plain-versus-instrumented canary."""

    if type(design) is not LoadedTQCDevelopmentDesignV2:
        raise ExperimentContractError("instrumentation probe requires strict-loaded design")
    binding = _validate_binding(
        receipt_binding,
        expected=None,
        require_current_worker=True,
    )
    _require_exact(
        source_tree_sha256(),
        binding["project_python_source_tree_sha256"],
        field_name="project Python source tree SHA-256",
    )
    raw = design.to_dict()
    gate = raw["evaluation"]["instrumentation_equivalence_gate"]
    frozen = {
        "seed": INSTRUMENTATION_SEED,
        "horizon_steps": INSTRUMENTATION_HORIZON,
        "action_shape": [INSTRUMENTATION_HORIZON, 17],
        "action_sha256": EXPECTED_ACTION_SHA256,
        "per_step_exact_comparisons": list(COMPARISON_FIELDS),
        "scope": INSTRUMENTATION_SCOPE,
    }
    for field_name, expected in frozen.items():
        _require_exact(gate[field_name], expected, field_name=f"design {field_name}")

    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - dependency boundary
        raise ExperimentContractError(
            "install the gym extra for instrumentation preflight"
        ) from exc

    config = HumanoidExperimentConfig(
        env_id="Humanoid-v5",
        terminate_when_unhealthy=False,
        reset_noise_scale=0.01,
        exclude_current_positions_from_observation=True,
        frame_skip=5,
    )
    plain = gym.make("Humanoid-v5", render_mode=None, **config.gym_kwargs())
    instrumented = make_humanoid_env(config, render_mode=None, capture_substep_contacts=True)
    plain_digest = hashlib.sha256()
    instrumented_digest = hashlib.sha256()
    mismatch_count = 0
    first_mismatch: dict[str, object] | None = None
    compared_steps = 0
    contact_digest = hashlib.sha256()
    contact_sample_count = 0

    def compare_boundary(
        left: dict[str, object], right: dict[str, object], *, boundary_index: int
    ) -> None:
        nonlocal mismatch_count, first_mismatch
        if tuple(left) != tuple(right):
            raise ExperimentContractError("shared instrumentation comparison fields differ")
        for field_name in left:
            _trace_value(
                plain_digest, label=f"{boundary_index}/{field_name}", value=left[field_name]
            )
            _trace_value(
                instrumented_digest,
                label=f"{boundary_index}/{field_name}",
                value=right[field_name],
            )
            if not _exact_equal(left[field_name], right[field_name]):
                mismatch_count += 1
                if first_mismatch is None:
                    first_mismatch = {
                        "boundary_index": boundary_index,
                        "field": field_name,
                    }

    try:
        structural_source_proof_sha256 = _structural_source_proof(type(instrumented.unwrapped))
        actions = tqc_instrumentation_actions_v2()
        plain_observation, _ = plain.reset(seed=INSTRUMENTATION_SEED)
        instrumented_observation, _ = instrumented.reset(seed=INSTRUMENTATION_SEED)
        compare_boundary(
            _reset_boundary(plain, plain_observation),
            _reset_boundary(instrumented, instrumented_observation),
            boundary_index=0,
        )
        for step_index, action in enumerate(actions, start=1):
            plain_action = action.copy(order="C")
            instrumented_action = action.copy(order="C")
            plain_step = plain.step(plain_action)
            instrumented_step = instrumented.step(instrumented_action)
            if not np.array_equal(plain_action, action) or not np.array_equal(
                instrumented_action, action
            ):
                raise ExperimentContractError("an environment mutated the frozen physical action")
            contact_record = _validated_contact_record(
                instrumented,
                step_index=step_index,
            )
            contact_record_bytes = canonical_json(contact_record)
            contact_digest.update(len(contact_record_bytes).to_bytes(8, "big"))
            contact_digest.update(contact_record_bytes)
            contact_sample_count += int(contact_record["contact_count"])
            compare_boundary(
                _boundary(
                    plain,
                    observation=plain_step[0],
                    reward=plain_step[1],
                    terminated=plain_step[2],
                    truncated=plain_step[3],
                ),
                _boundary(
                    instrumented,
                    observation=instrumented_step[0],
                    reward=instrumented_step[1],
                    terminated=instrumented_step[2],
                    truncated=instrumented_step[3],
                ),
                boundary_index=step_index,
            )
            compared_steps = step_index
            if step_index < INSTRUMENTATION_HORIZON and any(
                (plain_step[2], plain_step[3], instrumented_step[2], instrumented_step[3])
            ):
                mismatch_count += 1
                if first_mismatch is None:
                    first_mismatch = {
                        "boundary_index": step_index,
                        "field": "terminal_contract",
                    }
                break
        if compared_steps == INSTRUMENTATION_HORIZON and not (
            plain_step[2] is False
            and plain_step[3] is True
            and instrumented_step[2] is False
            and instrumented_step[3] is True
        ):
            mismatch_count += 1
            if first_mismatch is None:
                first_mismatch = {
                    "boundary_index": INSTRUMENTATION_HORIZON,
                    "field": "terminal_contract",
                }
    finally:
        plain.close()
        instrumented.close()

    plain_sha256 = plain_digest.hexdigest()
    instrumented_sha256 = instrumented_digest.hexdigest()
    contact_trace_sha256 = contact_digest.hexdigest()
    if (
        contact_trace_sha256 != EXPECTED_CONTACT_TRACE_SHA256
        or contact_sample_count != EXPECTED_CONTACT_SAMPLE_COUNT
    ):
        raise ExperimentContractError("instrumented contact trace differs from the frozen canary")
    passed = (
        mismatch_count == 0
        and compared_steps == INSTRUMENTATION_HORIZON
        and plain_sha256 == instrumented_sha256
    )
    payload = {
        "schema_version": 1,
        "receipt_id": INSTRUMENTATION_RECEIPT_ID,
        "design_file_sha256": design.file_sha256,
        "design_semantic_sha256": design.semantic_sha256,
        "training_projection_sha256": design.training_projection_sha256,
        "environment_instrumentation_equivalence_source_sha256": sha256_file(Path(__file__)),
        **binding,
        "seed": INSTRUMENTATION_SEED,
        "horizon_steps": INSTRUMENTATION_HORIZON,
        "action_shape": [INSTRUMENTATION_HORIZON, 17],
        "action_sha256": EXPECTED_ACTION_SHA256,
        "reset_comparison_count": 1,
        "per_step_comparison_count": compared_steps,
        "comparison_fields": list(COMPARISON_FIELDS),
        "structural_source_proof_sha256": structural_source_proof_sha256,
        "contact_trace_sha256": contact_trace_sha256,
        "contact_sample_count": contact_sample_count,
        "plain_shared_trace_sha256": plain_sha256,
        "instrumented_shared_trace_sha256": instrumented_sha256,
        "mismatch_count": mismatch_count,
        "first_mismatch": first_mismatch,
        "all_comparisons_passed": passed,
        "scope": INSTRUMENTATION_SCOPE,
        "claim_ceiling": CLAIM_CEILING,
        "behavioral_evidence": False,
    }
    encoded = canonical_json(payload)
    return validate_tqc_instrumentation_equivalence_receipt_v2(
        encoded,
        design,
        expected_binding=binding,
    )


__all__ = [
    "ATTEMPT_ID",
    "CLAIM_CEILING",
    "COMPARISON_FIELDS",
    "EXPECTED_ACTION_SHA256",
    "EXPECTED_CONTACT_SAMPLE_COUNT",
    "EXPECTED_CONTACT_TRACE_SHA256",
    "EXPECTED_SHARED_TRACE_SHA256",
    "EXPECTED_STRUCTURAL_SOURCE_PROOF_SHA256",
    "INSTRUMENTATION_HORIZON",
    "INSTRUMENTATION_RECEIPT_ID",
    "INSTRUMENTATION_SCOPE",
    "INSTRUMENTATION_SEED",
    "TQCInstrumentationEquivalenceReceiptV2",
    "run_tqc_instrumentation_equivalence_v2",
    "tqc_instrumentation_actions_v2",
    "validate_tqc_instrumentation_equivalence_receipt_v2",
]
