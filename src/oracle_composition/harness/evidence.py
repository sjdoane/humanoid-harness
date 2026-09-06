"""Fail-closed Experiment 003 identities, execution seal, and evidence receipts."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import stat
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from zipfile import BadZipFile, ZipFile

import numpy as np

from oracle_composition.contracts.reference_identity_v2 import (
    array_sha256,
    canonical_json_bytes,
    sha256_file,
)
from oracle_composition.experiments.reference_corpus_bundle import load_bundle_manifest

from .contract import (
    ALLOWED_SIGNALS,
    EVIDENCE_CLASS,
    OracleContractError,
    load_oracle_program,
    read_json_object,
)
from .inputs import LibraryManifest, TaskSpec, load_frozen_inputs, verify_library_artifacts

EXECUTION_MANIFEST_SCHEMA_ID = "humanoid_composition_execution_manifest/v1"
SCIENTIFIC_RECEIPT_SCHEMA_ID = "humanoid_composition_scientific_receipt/v2"
TELEMETRY_SCHEMA_ID = "humanoid_composition_telemetry/v1"
REPORT_SCHEMA_ID = "humanoid_controller_switching_cycle_report/v2"
LEGACY_REPORT_SCHEMA_ID = "humanoid_composition_cycle_report/v1"
TRACE_SCHEMA_ID = "humanoid_controller_switching_trace/v2"
TRACE_INDEX_SCHEMA_ID = "humanoid_composition_trace_index/v2"
ORACLE_CONTRACT_SCHEMA_ID = "humanoid_controller_switching_oracle/v1"
CLAIM_CEILING = (
    "exploratory_controller_switching_cycle_on_the_frozen_plain_humanoid_v5_runtime_"
    "only_no_oracle_quality_generalization_tracker_reference_following_reward_naturalness_"
    "robustness_or_humanoid_competence_claim"
)
_LEGACY_CYCLE_ZERO_CLAIM_CEILING = (
    "cycle_0_ran_on_the_frozen_plain_humanoid_v5_runtime_with_predeclared_controller_"
    "switching_arms_and_a_cycle_1_designer_prompt_only_no_oracle_quality_generalization_"
    "tracker_reward_or_humanoid_competence_claim"
)

E003_LIBRARY_SHA256 = "ad57578dc2ed4fe3707da74a4f86620e758b1aa30064016785d187a5e86d3207"
E003_TASK_SHA256 = "edb2cffde9f2eb087667d182f3da199ef6956aeadc6d45e9e218649d6fa9cbd7"
E003_BEHAVIORS = ("expert", "medium", "simple")
E003_SEEDS = tuple(range(97001, 97021))
E003_HORIZON_STEPS = 1000
E003_CONTROL_PERIOD_SECONDS = 0.015
E003_TASK_TEXT = (
    "Run at the fast gait, slow to the slow gait for the middle third, return to the "
    "fast gait, and never fall."
)
E003_SCHEDULE = (
    (0, 300, 5.520768616125457),
    (300, 600, 0.8853599908576963),
    (600, 1000, 5.520768616125457),
)
E003_CYCLE_ZERO_ORACLE_IDS = ("single_fast", "single_slow", "playback", "handwritten")
E003_TRACE_DIRECTORY = "artifacts/experiments_003"
E003_EXECUTION_MANIFEST_FILENAME = "execution_manifest_v1.json"

MAX_REPORT_BYTES = 2 * 1024 * 1024
MAX_TRACE_BYTES = 4 * 1024 * 1024
MAX_TRACE_INDEX_BYTES = 512 * 1024
MAX_CORPUS_ARRAY_MEMBER_BYTES = 64 * 1024
MAX_CYCLES = 32
MAX_EVIDENCE_JSON_DEPTH = 64
MAX_EVIDENCE_JSON_NODES = 500_000
MAX_SIGNED_32 = 2_147_483_647

SCHEMA_VERSIONS = MappingProxyType(
    {
        "execution_manifest": 1,
        "oracle_contract": 1,
        "report": 2,
        "scientific_receipt": 2,
        "telemetry": 1,
        "trace": 2,
        "trace_index": 2,
    }
)
LEGACY_PHASE_A_SCHEMA_VERSIONS = MappingProxyType(
    {
        **dict(SCHEMA_VERSIONS),
        "report": 1,
        "trace": 1,
        "trace_index": 1,
    }
)

_LEGACY_REPORT_SHA256 = (
    "3d32ddcc3869f9a05df9beb1bc1f7816996bf3bda166eb4ca26f3ec9556aa623",
    "480d5b3521ce66593e0258f3f42904f969c72b83141b330bc7ff4759144c25f0",
    "e677b9f3c3daabdb13648a18981c39b47bc74fa2e6908e7fe1af9d73674e194c",
)
_LEGACY_SOURCE_SHA256 = (
    {
        "contract": "cfd92e2a1ddb201bae7372227cee0d5089730cbb05e7afd72b19552c12097ab7",
        "executor": "03136c17d7f492556af8114dc612057a46c73a622eb2f07739ee335cd675310f",
        "evaluator": "de8370e419a981b3fa165ea1c408b2e08360b52cca7221d6431ac582d225a51e",
        "strict_loader": "40e199b899cd723f656813649dc03970a53da26634399f08e4e44603b07677aa",
        "archive_parser": "ca771ff1629d39753e6c04d88c4db14fa70de84926d7b85cdffa24d7190448d7",
        "inputs": "5ba96d442bec35334de53d421a55ac202d886a9a92bd4ccc2e1a9e67c1cd216b",
        "cycle_cli": "6c75e0bafbdb447f2279d37bf4ef15ec4269e121f88401a2ad52a37fc5177261",
        "serializer": "c64fbe08cad98da57977767214c00f4f4dcbda83420a6113703d2b254016e4c3",
    },
    {
        "contract": "cfd92e2a1ddb201bae7372227cee0d5089730cbb05e7afd72b19552c12097ab7",
        "executor": "03136c17d7f492556af8114dc612057a46c73a622eb2f07739ee335cd675310f",
        "evaluator": "dfbe9f9fe8a0d430bf9173f41cb6ac49d7d11eabbc80f6f42a4271e648e42438",
        "strict_loader": "40e199b899cd723f656813649dc03970a53da26634399f08e4e44603b07677aa",
        "archive_parser": "ca771ff1629d39753e6c04d88c4db14fa70de84926d7b85cdffa24d7190448d7",
        "inputs": "5ba96d442bec35334de53d421a55ac202d886a9a92bd4ccc2e1a9e67c1cd216b",
        "cycle_cli": "6c75e0bafbdb447f2279d37bf4ef15ec4269e121f88401a2ad52a37fc5177261",
        "serializer": "c64fbe08cad98da57977767214c00f4f4dcbda83420a6113703d2b254016e4c3",
    },
    {
        "contract": "cfd92e2a1ddb201bae7372227cee0d5089730cbb05e7afd72b19552c12097ab7",
        "executor": "03136c17d7f492556af8114dc612057a46c73a622eb2f07739ee335cd675310f",
        "evaluator": "f9ecaff3021f45d0306659434bc6613089a5f3c1602f0e4eb27b473fe5a9f4ce",
        "strict_loader": "40e199b899cd723f656813649dc03970a53da26634399f08e4e44603b07677aa",
        "archive_parser": "ca771ff1629d39753e6c04d88c4db14fa70de84926d7b85cdffa24d7190448d7",
        "inputs": "5ba96d442bec35334de53d421a55ac202d886a9a92bd4ccc2e1a9e67c1cd216b",
        "cycle_cli": "6c75e0bafbdb447f2279d37bf4ef15ec4269e121f88401a2ad52a37fc5177261",
        "serializer": "c64fbe08cad98da57977767214c00f4f4dcbda83420a6113703d2b254016e4c3",
    },
)


class EvidenceChainError(RuntimeError):
    """Raised before execution when evidence identities or history differ."""


class TraceIntegrityError(EvidenceChainError):
    """Raised when a persisted trace or trace index fails its detached audit."""


@dataclass(frozen=True, slots=True)
class SealedExecution:
    library: LibraryManifest
    task: TaskSpec
    manifest: Mapping[str, object]
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class ValidatedArmSummary:
    value: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ValidatedScientificReceipt:
    path: Path
    value: Mapping[str, object]
    encoded: bytes
    sha256: str
    arms: tuple[ValidatedArmSummary, ...]


@dataclass(frozen=True, slots=True)
class ValidatedDesignerProvenance:
    path: Path
    sha256: str


def _package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _source_identity(identity_id: str, sources: Mapping[str, str]) -> dict[str, object]:
    core = {"identity_id": identity_id, "source_sha256": dict(sorted(sources.items()))}
    return {**core, "sha256": hashlib.sha256(canonical_json_bytes(core)).hexdigest()}


def current_authority_identities() -> dict[str, object]:
    """Return separately change-sensitive source identities for every authority."""

    package = _package_root()

    def digest(relative: str) -> str:
        return sha256_file(package / relative)

    return {
        "archive_loader": _source_identity(
            "strict_tqc_data_only_archive_loader/v1",
            {
                "experiments/tqc_actor_npz.py": digest("experiments/tqc_actor_npz.py"),
                "harness/inputs.py": digest("harness/inputs.py"),
                "sources/strict_tqc_actor_runtime.py": digest(
                    "sources/strict_tqc_actor_runtime.py"
                ),
            },
        ),
        "metric_core": _source_identity(
            "humanoid_state_protected_endpoint/v1",
            {"harness/executor.py": digest("harness/executor.py")},
        ),
        "oracle_interpreter": _source_identity(
            "bounded_oracle_contract_and_guard_interpreter/v1",
            {"harness/contract.py": digest("harness/contract.py")},
        ),
        "report_writer": _source_identity(
            "composition_scientific_receipt_writer/v2",
            {
                "harness/cycle_cli.py": digest("harness/cycle_cli.py"),
                "contracts/reference_identity_v2.py": digest("contracts/reference_identity_v2.py"),
                "harness/evidence.py": digest("harness/evidence.py"),
                "harness/evaluator.py": digest("harness/evaluator.py"),
            },
        ),
    }


def _legacy_authority_identities(cycle: int, library: LibraryManifest) -> dict[str, object]:
    source = _LEGACY_SOURCE_SHA256[cycle]
    return {
        "actors": {entry.name: entry.strict_npz.sha256 for entry in library.behaviors},
        "archive_loader": _source_identity(
            "strict_tqc_data_only_archive_loader/v1",
            {
                "experiments/tqc_actor_npz.py": source["archive_parser"],
                "harness/inputs.py": source["inputs"],
                "sources/strict_tqc_actor_runtime.py": source["strict_loader"],
            },
        ),
        "metric_core": _source_identity(
            "humanoid_state_protected_endpoint/v1",
            {"harness/executor.py": source["executor"]},
        ),
        "oracle_interpreter": _source_identity(
            "bounded_oracle_contract_and_guard_interpreter/v1",
            {"harness/contract.py": source["contract"]},
        ),
        "report_writer": _source_identity(
            "composition_legacy_report_writer/v1",
            {
                "contracts/reference_identity_v2.py": source["serializer"],
                "harness/cycle_cli.py": source["cycle_cli"],
                "harness/evaluator.py": source["evaluator"],
            },
        ),
    }


def _speed_dict(entry: object) -> dict[str, object]:
    speed = entry.speed
    return {
        "admitted_clip_count": speed.admitted_clip_count,
        "admitted_step_count": speed.admitted_step_count,
        "iqr_m_s": speed.iqr_m_s,
        "median_m_s": speed.median_m_s,
        "q1_m_s": speed.q1_m_s,
        "q3_m_s": speed.q3_m_s,
    }


def _fall_dict(entry: object) -> dict[str, object]:
    falls = entry.falls
    return {
        "e3_episode_count": falls.e3_episode_count,
        "e3_step_count": falls.e3_step_count,
        "fall_event_count": falls.fall_event_count,
        "fall_seeds": list(falls.fall_seeds),
        "falls_per_1000_steps": falls.falls_per_1000_steps,
    }


def execution_manifest_value(library: LibraryManifest, task: TaskSpec) -> dict[str, object]:
    """Build the redundant pre-execution seal; callers still check fixed literals."""

    return {
        "evidence_class": EVIDENCE_CLASS,
        "execution_manifest_schema_id": EXECUTION_MANIFEST_SCHEMA_ID,
        "library": {
            "behaviors": [
                {
                    "e3_falls": _fall_dict(entry),
                    "name": entry.name,
                    "speed_m_s": _speed_dict(entry),
                    "strict_npz_sha256": entry.strict_npz.sha256,
                }
                for entry in library.behaviors
            ],
            "manifest_id": "humanoid_three_actor_library/v1",
            "raw_sha256": library.raw_sha256,
            "source_evidence_sha256": {
                name: binding.sha256 for name, binding in library.source_evidence.items()
            },
        },
        "schema_version": 1,
        "task": {
            "control_period_seconds": task.control_period_seconds,
            "cycle_zero_oracle_ids": list(task.cycle_zero_oracle_ids),
            "horizon_steps": task.horizon_steps,
            "raw_sha256": task.raw_sha256,
            "schedule": [
                {"start": item.start, "stop": item.stop, "target_m_s": item.target_m_s}
                for item in task.schedule
            ],
            "seeds": list(task.seeds),
            "task_spec_id": "humanoid_speed_profile_t1/v1",
            "task_text": task.task_text,
            "trace_artifact_directory": task.trace_artifact_directory,
        },
    }


def _require_e003_literals(library: LibraryManifest, task: TaskSpec) -> None:
    observed_schedule = tuple((item.start, item.stop, item.target_m_s) for item in task.schedule)
    if library.raw_sha256 != E003_LIBRARY_SHA256 or task.raw_sha256 != E003_TASK_SHA256:
        raise EvidenceChainError("task or library bytes differ from the E003 execution seal")
    if library.behavior_names != E003_BEHAVIORS:
        raise EvidenceChainError("E003 requires exactly expert, medium, and simple")
    if (
        task.task_text != E003_TASK_TEXT
        or task.horizon_steps != E003_HORIZON_STEPS
        or task.control_period_seconds != E003_CONTROL_PERIOD_SECONDS
        or task.seeds != E003_SEEDS
        or observed_schedule != E003_SCHEDULE
        or task.cycle_zero_oracle_ids != E003_CYCLE_ZERO_ORACLE_IDS
        or task.trace_artifact_directory != E003_TRACE_DIRECTORY
    ):
        raise EvidenceChainError("task fields differ from the exact E003 execution seal")


def _regular_bytes(path: Path, *, maximum: int, field: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise EvidenceChainError(f"{field} is unavailable") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise EvidenceChainError(f"{field} must be a regular non-linked file")
    if not 0 < metadata.st_size <= maximum:
        raise EvidenceChainError(f"{field} exceeds its byte bound")
    try:
        encoded = path.read_bytes()
    except OSError as exc:
        raise EvidenceChainError(f"{field} cannot be read") from exc
    if len(encoded) != metadata.st_size:
        raise EvidenceChainError(f"{field} changed while it was read")
    return encoded


def _optional_regular_bytes(path: Path, *, maximum: int, field: str) -> bytes | None:
    """Read a locally retained audit source, or return ``None`` when it is absent."""

    try:
        path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise EvidenceChainError(f"{field} is unavailable") from exc
    return _regular_bytes(path, maximum=maximum, field=field)


def _read_root_x(payload_path: Path, *, expected_array_sha256: str) -> np.ndarray:
    try:
        with ZipFile(payload_path, mode="r") as archive:
            info = archive.getinfo("boundary_root_xy.npy")
            if not 0 < info.file_size <= MAX_CORPUS_ARRAY_MEMBER_BYTES:
                raise EvidenceChainError("boundary_root_xy member exceeds its byte bound")
            with archive.open(info, mode="r") as member:
                encoded = member.read(MAX_CORPUS_ARRAY_MEMBER_BYTES + 1)
    except (BadZipFile, KeyError, OSError, ValueError) as exc:
        raise EvidenceChainError("bound corpus root-x member is invalid") from exc
    if len(encoded) > MAX_CORPUS_ARRAY_MEMBER_BYTES:
        raise EvidenceChainError("boundary_root_xy member exceeds its byte bound")
    try:
        value = np.lib.format.read_array(io.BytesIO(encoded), allow_pickle=False)
    except (EOFError, OSError, ValueError) as exc:
        raise EvidenceChainError("bound corpus root-x array is invalid") from exc
    array = np.ascontiguousarray(value)
    if array.dtype != np.dtype("<f8") or array.shape != (1001, 2):
        raise EvidenceChainError("bound corpus root-x array shape or dtype differs")
    if not np.isfinite(array).all() or array_sha256(array) != expected_array_sha256:
        raise EvidenceChainError("bound corpus root-x array identity differs")
    return array[:, 0]


def verify_library_statistics(repository_root: Path, library: LibraryManifest) -> None:
    """Recompute speed and fall fields from the bound corpus and E3 certificate."""

    root = Path(repository_root)
    corpus_binding = library.source_evidence["corpus_manifest_v2"]
    e3_binding = library.source_evidence["e3_certificate_v2"]
    try:
        corpus, _ = read_json_object(root / corpus_binding.path)
        certificate, _ = read_json_object(root / e3_binding.path)
    except OracleContractError as exc:
        raise EvidenceChainError(str(exc)) from exc
    clips = corpus.get("clips_in_reset_order")
    blocks = certificate.get("blocks")
    if (
        corpus.get("actors_in_order") != list(E003_BEHAVIORS)
        or type(clips) is not list
        or type(blocks) is not list
        or len(clips) != 108
        or len(blocks) != 36
    ):
        raise EvidenceChainError("bound corpus or E3 receipt shape differs")
    clip_map: dict[tuple[int, str], dict[str, object]] = {}
    for raw in clips:
        if type(raw) is not dict:
            raise EvidenceChainError("bound corpus clip row is malformed")
        key = (raw.get("seed"), raw.get("actor_variant"))
        if type(key[0]) is not int or key[1] not in E003_BEHAVIORS or key in clip_map:
            raise EvidenceChainError("bound corpus clip identities are malformed")
        clip_map[key] = raw

    speed_rows: dict[str, list[np.ndarray]] = defaultdict(list)
    fall_seeds: dict[str, list[int]] = defaultdict(list)
    corpus_root = (root / corpus_binding.path).parent
    seen_seeds: set[int] = set()
    for raw_block in blocks:
        if type(raw_block) is not dict or type(raw_block.get("seed")) is not int:
            raise EvidenceChainError("E3 block is malformed")
        seed = raw_block["seed"]
        if seed in seen_seeds or type(raw_block.get("block_passed")) is not bool:
            raise EvidenceChainError("E3 block identities are duplicate or malformed")
        seen_seeds.add(seed)
        branches = raw_block.get("branches")
        if type(branches) is not list or len(branches) != len(E003_BEHAVIORS):
            raise EvidenceChainError("E3 block branches differ from the library")
        branch_names: set[str] = set()
        for branch in branches:
            if type(branch) is not dict or branch.get("actor_variant") not in E003_BEHAVIORS:
                raise EvidenceChainError("E3 branch identity is malformed")
            name = branch["actor_variant"]
            if name in branch_names:
                raise EvidenceChainError("E3 branch identity is duplicated")
            branch_names.add(name)
            if branch.get("full_horizon") is not True:
                raise EvidenceChainError("E3 branch did not execute the frozen horizon")
            healthy = branch.get("healthy_boundary_count")
            upright = branch.get("upright_boundary_count")
            if type(healthy) is not int or type(upright) is not int:
                raise EvidenceChainError("E3 boundary counts are malformed")
            if healthy < E003_HORIZON_STEPS or upright < E003_HORIZON_STEPS:
                fall_seeds[name].append(seed)
            clip = clip_map.get((seed, name))
            if clip is None:
                raise EvidenceChainError("E3 branch has no bound corpus clip")
            if clip.get("bundle_manifest_sha256") != branch.get(
                "bundle_manifest_sha256"
            ) or clip.get("reference_identity_sha256") != branch.get("reference_identity_sha256"):
                raise EvidenceChainError("E3 branch and corpus clip identities differ")
            if raw_block["block_passed"]:
                clip_id = clip.get("clip_id")
                bundle_sha256 = clip.get("bundle_manifest_sha256")
                if type(clip_id) is not str or type(bundle_sha256) is not str:
                    raise EvidenceChainError("corpus bundle binding is malformed")
                bundle_path = corpus_root / "clips" / clip_id / f"bundle-{bundle_sha256}.json"
                try:
                    bundle = load_bundle_manifest(bundle_path, expected_sha256=bundle_sha256)
                except ValueError as exc:
                    raise EvidenceChainError("corpus bundle manifest identity differs") from exc
                core = bundle["core"]
                if (
                    core["seed"] != seed
                    or core["actor_variant"] != name
                    or core["clip_id"] != clip_id
                    or core["steps"] != E003_HORIZON_STEPS
                ):
                    raise EvidenceChainError("corpus bundle core differs from its E3 branch")
                bindings = core["array_bindings"]
                root_bindings = [
                    item for item in bindings if item.get("name") == "boundary_root_xy"
                ]
                if len(root_bindings) != 1:
                    raise EvidenceChainError("corpus bundle omits the root-x binding")
                payload = core["payload"]
                payload_path = corpus_root / payload["object_path"]
                root_x = _read_root_x(
                    payload_path,
                    expected_array_sha256=root_bindings[0]["sha256"],
                )
                speed_rows[name].append(np.diff(root_x) / E003_CONTROL_PERIOD_SECONDS)

    if seen_seeds != set(corpus.get("block_seeds_in_order", [])):
        raise EvidenceChainError("corpus and E3 seed sets differ")
    for entry in library.behaviors:
        values = np.concatenate(speed_rows[entry.name])
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
        expected_speed = {
            "admitted_clip_count": len(speed_rows[entry.name]),
            "admitted_step_count": int(values.size),
            "iqr_m_s": float(q3 - q1),
            "median_m_s": float(median),
            "q1_m_s": float(q1),
            "q3_m_s": float(q3),
        }
        observed_speed = _speed_dict(entry)
        for field, expected in expected_speed.items():
            observed = observed_speed[field]
            if type(expected) is int:
                matches = observed == expected
            else:
                matches = math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1e-12)
            if not matches:
                raise EvidenceChainError(f"{entry.name} library speed statistic {field} differs")
        expected_falls = tuple(sorted(fall_seeds[entry.name]))
        observed_falls = entry.falls
        if (
            observed_falls.e3_episode_count != len(blocks)
            or observed_falls.e3_step_count != len(blocks) * E003_HORIZON_STEPS
            or observed_falls.fall_event_count != len(expected_falls)
            or observed_falls.fall_seeds != expected_falls
            or not math.isclose(
                observed_falls.falls_per_1000_steps,
                len(expected_falls) / len(blocks),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise EvidenceChainError(f"{entry.name} library fall statistics differ")


def load_e003_execution(experiment: Path, repository_root: Path) -> SealedExecution:
    """Validate the exact E003 seal and evidence sources before runtime creation."""

    directory = Path(experiment)
    library, task = load_frozen_inputs(directory)
    _require_e003_literals(library, task)
    manifest_path = directory / E003_EXECUTION_MANIFEST_FILENAME
    try:
        manifest, encoded = read_json_object(manifest_path)
    except OracleContractError as exc:
        raise EvidenceChainError(str(exc)) from exc
    expected = execution_manifest_value(library, task)
    if manifest != expected or encoded != canonical_json_bytes(expected):
        raise EvidenceChainError("execution manifest differs from the exact E003 seal")
    verify_library_artifacts(Path(repository_root), library)
    verify_library_statistics(Path(repository_root), library)
    return SealedExecution(
        library=library,
        task=task,
        manifest=MappingProxyType(manifest),
        manifest_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def load_test_execution(experiment: Path, repository_root: Path) -> SealedExecution:
    """Test-only seam for deterministic fakes; never selected by the CLI default."""

    library, task = load_frozen_inputs(Path(experiment))
    verify_library_artifacts(Path(repository_root), library)
    value = execution_manifest_value(library, task)
    encoded = canonical_json_bytes(value)
    return SealedExecution(
        library=library,
        task=task,
        manifest=MappingProxyType(value),
        manifest_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _decode_bounded_object(encoded: bytes, *, maximum: int, source: str) -> dict[str, object]:
    if type(encoded) is not bytes or not encoded or len(encoded) > maximum:
        raise EvidenceChainError(f"{source} exceeds its byte bound")

    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise EvidenceChainError(f"{source} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise EvidenceChainError(f"{source} contains non-finite constant {value}")

    try:
        value = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=reject_constant,
        )
    except EvidenceChainError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        MemoryError,
        OverflowError,
        ValueError,
    ) as exc:
        raise EvidenceChainError(f"{source} is invalid bounded JSON") from exc
    if type(value) is not dict:
        raise EvidenceChainError(f"{source} must contain one JSON object")
    pending: list[tuple[object, int]] = [(value, 1)]
    nodes = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if depth > MAX_EVIDENCE_JSON_DEPTH:
            raise EvidenceChainError(f"{source} exceeds the JSON depth limit")
        if nodes > MAX_EVIDENCE_JSON_NODES:
            raise EvidenceChainError(f"{source} exceeds the JSON node limit")
        if type(item) is dict:
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
    return value


def _finite(value: object, *, field: str) -> float:
    if type(value) not in {int, float}:
        raise EvidenceChainError(f"{field} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise EvidenceChainError(f"{field} must be finite") from exc
    if not math.isfinite(result):
        raise EvidenceChainError(f"{field} must be finite")
    return result


def _identifier(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 64
        or not value.isascii()
        or not value.replace("_", "a").isalnum()
        or not value[0].islower()
    ):
        raise EvidenceChainError(f"{field} must be a bounded ASCII identifier")
    return value


def _sha(value: object, *, field: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EvidenceChainError(f"{field} must be a lowercase SHA-256")
    return value


def _safe_receipt_path(root: Path, value: object, *, field: str) -> Path:
    if type(value) is not str or not value or len(value) > 512:
        raise EvidenceChainError(f"{field} must be a bounded relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts or value != relative.as_posix():
        raise EvidenceChainError(f"{field} must be a normalized relative path")
    return root / relative


def validate_designer_provenance(
    *,
    experiment: Path,
    repository_root: Path,
    cycle: int,
    oracle_id: str,
    oracle_file_sha256: str,
    oracle_sha256: str,
) -> ValidatedDesignerProvenance:
    """Bind the audit log, declared reads, model request, final JSON, and copied oracle."""

    path = Path(experiment) / "cycles" / f"cycle_{cycle}" / "designer_provenance.json"
    encoded = _regular_bytes(path, maximum=MAX_REPORT_BYTES, field="designer provenance")
    receipt = _decode_bounded_object(
        encoded,
        maximum=MAX_REPORT_BYTES,
        source="designer provenance",
    )
    if encoded != canonical_json_bytes(receipt) or set(receipt) != {
        "allowed_inputs",
        "canonical_oracle",
        "cycle",
        "designer_provenance_schema_id",
        "evidence_class",
        "isolation_property",
        "launch_identity",
        "requested",
        "run_artifacts",
        "run_id",
        "schema_version",
        "status",
    }:
        raise EvidenceChainError("designer provenance schema differs")
    if (
        receipt["cycle"] != cycle
        or receipt["schema_version"] != 1
        or receipt["designer_provenance_schema_id"] != "humanoid_oracle_designer_provenance/v1"
        or receipt["evidence_class"] != "audit_log_only"
        or receipt["isolation_property"] != "audit_log_only_not_os_enforced"
        or receipt["status"] != "SUCCEEDED"
        or receipt["requested"]
        != {
            "model": "gpt-5.6-sol",
            "reasoning_effort": "max",
            "source": "requested_identity_only",
        }
    ):
        raise EvidenceChainError("designer provenance identity differs")
    receipt_launch_identity = receipt["launch_identity"]
    if type(receipt_launch_identity) is not dict or set(receipt_launch_identity) != {
        "runner_kind",
        "screen_name",
        "started_at_utc",
    }:
        raise EvidenceChainError("designer receipt launch identity differs")
    run_id = receipt["run_id"]
    if (
        type(run_id) is not str
        or not 1 <= len(run_id) <= 128
        or not run_id.isascii()
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
            for character in run_id
        )
    ):
        raise EvidenceChainError("designer run_id is malformed")
    oracle = receipt["canonical_oracle"]
    if type(oracle) is not dict or set(oracle) != {
        "canonical_sha256",
        "copied_file_path",
        "copied_file_sha256",
        "copied_semantic_hash_match",
        "source_file_path",
        "source_file_sha256",
    }:
        raise EvidenceChainError("designer canonical-oracle binding differs")
    copied_path = _safe_receipt_path(
        Path(repository_root), oracle["copied_file_path"], field="copied oracle path"
    )
    expected_copied_path = Path(experiment) / "cycles" / f"cycle_{cycle}" / f"oracle_{cycle}.json"
    if copied_path != expected_copied_path:
        raise EvidenceChainError("designer copied-oracle path differs from the cycle contract")
    copied_bytes = _regular_bytes(
        copied_path,
        maximum=MAX_REPORT_BYTES,
        field="copied oracle",
    )
    source_path = _safe_receipt_path(
        Path(repository_root),
        oracle["source_file_path"],
        field="designer source oracle path",
    )
    expected_source_path = (
        Path(repository_root) / ".orchestration" / "oracles" / f"cycle_{cycle}_candidate.json"
    )
    if source_path != expected_source_path:
        raise EvidenceChainError("designer source-oracle path differs from the cycle contract")
    source_oracle_bytes = _optional_regular_bytes(
        source_path,
        maximum=MAX_REPORT_BYTES,
        field="designer source oracle",
    )
    try:
        copied_value = _decode_bounded_object(
            copied_bytes,
            maximum=MAX_REPORT_BYTES,
            source="copied oracle",
        )
        copied_canonical_sha256 = hashlib.sha256(canonical_json_bytes(copied_value)).hexdigest()
    except (EvidenceChainError, ValueError) as exc:
        raise EvidenceChainError("copied oracle cannot be canonicalized") from exc
    if (
        oracle["copied_semantic_hash_match"] is not True
        or _sha(oracle["copied_file_sha256"], field="copied oracle file SHA") != oracle_file_sha256
        or hashlib.sha256(copied_bytes).hexdigest() != oracle_file_sha256
        or _sha(oracle["source_file_sha256"], field="source oracle file SHA") != oracle_file_sha256
        or (source_oracle_bytes is not None and source_oracle_bytes != copied_bytes)
        or _sha(oracle["canonical_sha256"], field="canonical oracle SHA") != oracle_sha256
        or copied_canonical_sha256 != oracle_sha256
        or copied_value.get("oracle_id") != oracle_id
    ):
        raise EvidenceChainError("designer final and copied oracle identities differ")

    allowed = receipt["allowed_inputs"]
    if type(allowed) is not list or not allowed:
        raise EvidenceChainError("designer allowed-input ledger is empty")
    allowed_by_path: dict[str, dict[str, object]] = {}
    for item in allowed:
        if type(item) is not dict or set(item) != {"path", "sha256", "sha256_source"}:
            raise EvidenceChainError("designer allowed-input binding differs")
        relative = item["path"]
        _safe_receipt_path(Path(repository_root), relative, field="designer allowed input")
        _sha(item["sha256"], field="designer allowed-input SHA")
        if item["sha256_source"] not in {
            "current_immutable_file",
            "captured_events_log_before_corrected_rerender",
        }:
            raise EvidenceChainError("designer allowed-input provenance differs")
        if relative in allowed_by_path:
            raise EvidenceChainError("designer allowed-input path is duplicated")
        allowed_by_path[relative] = item
        if item["sha256_source"] == "current_immutable_file":
            allowed_bytes = _regular_bytes(
                Path(repository_root) / relative,
                maximum=MAX_REPORT_BYTES,
                field="designer allowed input",
            )
            if hashlib.sha256(allowed_bytes).hexdigest() != item["sha256"]:
                raise EvidenceChainError("designer allowed-input bytes differ")
    expected_allowed_paths = {
        f"experiments/003_composition_speed_profile/cycles/cycle_{cycle}/designer_prompt.md"
    }
    if cycle == 2:
        expected_allowed_paths.add(
            "experiments/003_composition_speed_profile/cycles/cycle_1/report_1.md"
        )
    if set(allowed_by_path) != expected_allowed_paths:
        raise EvidenceChainError("designer allowed-input set differs from the cycle contract")

    artifacts = receipt["run_artifacts"]
    artifact_names = {
        "events.jsonl",
        "final.txt",
        "launch.json",
        "request.json",
        "result.json",
        "task-packet.md",
    }
    if type(artifacts) is not dict or set(artifacts) != artifact_names:
        raise EvidenceChainError("designer run-artifact ledger differs")
    artifact_bytes: dict[str, bytes] = {}
    absent_artifacts: set[str] = set()
    for name in sorted(artifact_names):
        binding = artifacts[name]
        if type(binding) is not dict or set(binding) != {"path", "sha256"}:
            raise EvidenceChainError("designer run-artifact binding differs")
        artifact_path = _safe_receipt_path(
            Path(repository_root), binding["path"], field=f"designer {name} path"
        )
        expected_artifact_path = (
            Path(repository_root) / ".orchestration" / "sol-runs" / run_id / name
        )
        if artifact_path != expected_artifact_path:
            raise EvidenceChainError(f"designer {name} path differs from the run contract")
        expected_sha256 = _sha(binding["sha256"], field=f"designer {name} SHA")
        raw = _optional_regular_bytes(
            artifact_path,
            maximum=MAX_REPORT_BYTES,
            field=f"designer {name}",
        )
        if raw is None:
            absent_artifacts.add(name)
            continue
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise EvidenceChainError(f"designer {name} bytes differ")
        artifact_bytes[name] = raw
    if absent_artifacts and artifact_bytes:
        raise EvidenceChainError("designer raw audit bundle is incomplete")
    if not artifact_bytes:
        raise EvidenceChainError("designer provenance is unverifiable without its raw audit bundle")
    if source_oracle_bytes is None:
        raise EvidenceChainError("designer raw audit bundle omits its source oracle")
    launch = _decode_bounded_object(
        artifact_bytes["launch.json"], maximum=MAX_REPORT_BYTES, source="designer launch"
    )
    request = _decode_bounded_object(
        artifact_bytes["request.json"], maximum=MAX_REPORT_BYTES, source="designer request"
    )
    result = _decode_bounded_object(
        artifact_bytes["result.json"], maximum=MAX_REPORT_BYTES, source="designer result"
    )
    launch_keys = {"runner_kind", "runner_pid", "schema_version", "screen_name", "started_at_utc"}
    request_keys = {
        "codex_bin",
        "codex_cli_version",
        "created_at_utc",
        "mode",
        "owner",
        "prompt_bytes",
        "prompt_sha256",
        "requested_model",
        "requested_reasoning_effort",
        "role",
        "runner_kind",
        "schema_version",
        "scope",
        "screen_name",
    }
    result_keys = {
        "codex_cli_version",
        "exit_code",
        "finished_at_utc",
        "lease_release",
        "requested_model",
        "requested_reasoning_effort",
        "role",
        "runner_kind",
        "schema_version",
        "scope",
        "screen_name",
        "status",
        "termination_escalated",
        "thread_id",
    }
    task_packet = artifact_bytes["task-packet.md"]
    if (
        set(launch) != launch_keys
        or set(request) != request_keys
        or set(result) != result_keys
        or {
            "runner_kind": launch["runner_kind"],
            "screen_name": launch["screen_name"],
            "started_at_utc": launch["started_at_utc"],
        }
        != receipt_launch_identity
        or launch["schema_version"] != 1
        or type(launch["runner_pid"]) is not int
        or not 1 <= launch["runner_pid"] <= MAX_SIGNED_32
        or request["schema_version"] != 1
        or request["mode"] != "read-only"
        or request["role"] != "designer"
        or result["schema_version"] != 1
        or result["role"] != request["role"]
        or result["scope"] != request["scope"]
        or result["runner_kind"] != request["runner_kind"]
        or result["screen_name"] != request["screen_name"]
        or launch["runner_kind"] != request["runner_kind"]
        or launch["screen_name"] != request["screen_name"]
        or launch["started_at_utc"] != request["created_at_utc"]
        or type(request["prompt_bytes"]) is not int
        or not 1 <= request["prompt_bytes"] <= MAX_REPORT_BYTES
        or request["prompt_bytes"] != len(task_packet)
        or _sha(request["prompt_sha256"], field="designer prompt SHA")
        != hashlib.sha256(task_packet).hexdigest()
        or type(result["exit_code"]) is not int
        or result["exit_code"] != 0
        or result["lease_release"] != "NOT_REQUIRED"
        or result["termination_escalated"] is not False
        or result["codex_cli_version"] != request["codex_cli_version"]
        or request.get("requested_model") != receipt["requested"]["model"]
        or request.get("requested_reasoning_effort") != receipt["requested"]["reasoning_effort"]
        or result.get("requested_model") != receipt["requested"]["model"]
        or result.get("requested_reasoning_effort") != receipt["requested"]["reasoning_effort"]
        or result.get("status") != receipt["status"]
    ):
        raise EvidenceChainError("designer launch, request, or result identity differs")
    final_text = artifact_bytes["final.txt"].decode("utf-8")
    match = re.search(r"```json\n(.*?)\n```", final_text, flags=re.DOTALL)
    if match is None:
        raise EvidenceChainError("designer final omits one fenced JSON object")
    final_value = _decode_bounded_object(
        match.group(1).encode("utf-8"),
        maximum=MAX_REPORT_BYTES,
        source="designer final oracle",
    )
    if hashlib.sha256(canonical_json_bytes(final_value)).hexdigest() != oracle_sha256:
        raise EvidenceChainError("designer final oracle differs from the copied oracle")

    events_text = artifact_bytes["events.jsonl"].decode("utf-8")
    captured: dict[str, str] = {}
    for line in events_text.splitlines():
        event = _decode_bounded_object(
            line.encode("utf-8"), maximum=MAX_REPORT_BYTES, source="designer event"
        )
        item = event.get("item")
        if (
            event.get("type") == "item.completed"
            and type(item) is dict
            and item.get("type") == "command_execution"
            and type(item.get("aggregated_output")) is str
            and type(item.get("command")) is str
        ):
            for relative in allowed_by_path:
                if relative in item["command"]:
                    captured[relative] = hashlib.sha256(
                        item["aggregated_output"].encode("utf-8")
                    ).hexdigest()
    if set(captured) != set(allowed_by_path) or any(
        captured[relative] != item["sha256"] for relative, item in allowed_by_path.items()
    ):
        raise EvidenceChainError("designer event log does not confirm the declared reads")
    return ValidatedDesignerProvenance(path=path, sha256=hashlib.sha256(encoded).hexdigest())


def _summary_from_rows(
    *,
    oracle: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    behavior_names: Sequence[str],
) -> dict[str, object]:
    errors = np.asarray(
        [_finite(row["mean_absolute_speed_error_m_s"], field="episode MAE") for row in rows],
        dtype=np.float64,
    )
    returns = np.asarray(
        [_finite(row["task_return"], field="episode task_return") for row in rows],
        dtype=np.float64,
    )
    switches = np.asarray([row["switch_count"] for row in rows], dtype=np.int64)
    q1, median, q3 = np.quantile(errors, [0.25, 0.5, 0.75], method="linear")
    return {
        "episode_count": len(rows),
        "fall_count": sum(row["fall"] for row in rows),
        "median_mean_absolute_speed_error_m_s": float(median),
        "median_switch_count": float(np.median(switches)),
        "median_task_return": float(np.median(returns)),
        "median_time_in_each_behavior_steps": {
            name: float(np.median([row["time_in_each_behavior_steps"][name] for row in rows]))
            for name in behavior_names
        },
        "oracle_file_sha256": oracle["file_sha256"],
        "oracle_id": oracle["oracle_id"],
        "oracle_sha256": oracle["oracle_sha256"],
        "q1_mean_absolute_speed_error_m_s": float(q1),
        "q3_mean_absolute_speed_error_m_s": float(q3),
    }


def _validate_episode_row(
    row: object,
    *,
    task: TaskSpec,
    behavior_names: Sequence[str],
    diagnostics: bool,
) -> dict[str, object]:
    base = {
        "fall",
        "first_fall_step",
        "mean_absolute_speed_error_m_s",
        "observed_steps",
        "oracle_file_sha256",
        "oracle_id",
        "oracle_sha256",
        "seed",
        "switch_count",
        "task_return",
        "time_in_each_behavior_steps",
        "trace_byte_count",
        "trace_path",
        "trace_sha256",
    }
    expected = base | (
        {"controller_switches", "slow_third_behavior_fractions"} if diagnostics else set()
    )
    if type(row) is not dict or set(row) != expected:
        raise EvidenceChainError("per-episode row keys differ from the receipt schema")
    if type(row["fall"]) is not bool:
        raise EvidenceChainError("episode fall must be boolean")
    first_fall = row["first_fall_step"]
    if first_fall is not None and (
        type(first_fall) is not int or not 0 <= first_fall <= task.horizon_steps
    ):
        raise EvidenceChainError("episode first-fall boundary is invalid")
    if row["fall"] != (first_fall is not None):
        raise EvidenceChainError("episode fall and first-fall boundary disagree")
    integer_bounds = {
        "observed_steps": task.horizon_steps,
        "seed": MAX_SIGNED_32,
        "switch_count": task.horizon_steps,
        "trace_byte_count": MAX_TRACE_BYTES,
    }
    for field, maximum in integer_bounds.items():
        if type(row[field]) is not int or not 0 <= row[field] <= maximum:
            raise EvidenceChainError(f"episode {field} lies outside its semantic integer bound")
    if row["observed_steps"] != task.horizon_steps or row["seed"] not in task.seeds:
        raise EvidenceChainError("episode horizon or seed differs from the task")
    _finite(row["mean_absolute_speed_error_m_s"], field="episode MAE")
    _finite(row["task_return"], field="episode task_return")
    _identifier(row["oracle_id"], field="episode oracle_id")
    _sha(row["oracle_file_sha256"], field="episode oracle_file_sha256")
    _sha(row["oracle_sha256"], field="episode oracle_sha256")
    _sha(row["trace_sha256"], field="episode trace_sha256")
    counts = row["time_in_each_behavior_steps"]
    if type(counts) is not dict or set(counts) != set(behavior_names):
        raise EvidenceChainError("episode behavior counts differ from the library")
    if any(type(counts[name]) is not int or counts[name] < 0 for name in behavior_names):
        raise EvidenceChainError("episode behavior counts are malformed")
    if sum(counts.values()) != task.horizon_steps:
        raise EvidenceChainError("episode behavior counts do not cover the horizon")
    if type(row["trace_path"]) is not str or len(row["trace_path"]) > 512:
        raise EvidenceChainError("episode trace path is malformed")
    if diagnostics:
        switches = row["controller_switches"]
        if type(switches) is not list or len(switches) != row["switch_count"]:
            raise EvidenceChainError("episode switch diagnostics are malformed")
        for switch in switches:
            if type(switch) is not dict or set(switch) != {
                "fall_followed_within_100_steps",
                "from_behavior",
                "step",
                "to_behavior",
                "v_x_m_s",
            }:
                raise EvidenceChainError("episode switch diagnostic schema differs")
            if (
                type(switch["fall_followed_within_100_steps"]) is not bool
                or type(switch["step"]) is not int
                or not 0 <= switch["step"] < task.horizon_steps
                or switch["from_behavior"] not in behavior_names
                or switch["to_behavior"] not in behavior_names
            ):
                raise EvidenceChainError("episode switch diagnostic value differs")
            _finite(switch["v_x_m_s"], field="switch speed")
        fractions = row["slow_third_behavior_fractions"]
        if type(fractions) is not dict or set(fractions) != set(behavior_names):
            raise EvidenceChainError("episode slow-segment diagnostics are malformed")
        for value in fractions.values():
            fraction = _finite(value, field="slow-segment fraction")
            if not 0.0 <= fraction <= 1.0:
                raise EvidenceChainError("slow-segment fraction lies outside [0, 1]")
        if not math.isclose(
            math.fsum(
                _finite(value, field="slow-segment fraction") for value in fractions.values()
            ),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise EvidenceChainError("slow-segment fractions do not sum to one")
    return row


def _trace_metrics(
    trace: Mapping[str, object],
    *,
    task: TaskSpec,
    behavior_names: Sequence[str],
    diagnostics: bool,
) -> dict[str, object]:
    """Parse exact trace steps and independently reconstruct every protected metric."""

    steps = trace["steps"]
    assert type(steps) is list  # checked by the trace-header validator
    step_keys = {
        "active_behavior",
        "active_state",
        "physical_action_sha256",
        "post_boundary",
        "signals",
        "switch_flags",
        "switch_reason",
        "t",
    }
    post_keys = {
        "stock_info_x_velocity_m_s",
        "task_reward",
        "torso_up",
        "truncated",
        "v_x",
        "x_travelled",
        "z_root",
    }
    flag_keys = {
        "controller_switched",
        "recovery_entered",
        "recovery_exited",
        "state_transition",
    }
    errors: list[float] = []
    rewards: list[float] = []
    counts = {name: 0 for name in behavior_names}
    switches: list[dict[str, object]] = []
    first_fall: int | None = None
    initial_fallen = False
    previous_behavior: str | None = None
    active_by_step: list[str] = []
    fallen_boundaries: list[int] = []
    for index, step in enumerate(steps):
        if type(step) is not dict or set(step) != step_keys or step["t"] != index:
            raise TraceIntegrityError("persisted trace step schema or order differs")
        behavior = step["active_behavior"]
        if behavior not in behavior_names:
            raise TraceIntegrityError("persisted trace step behavior is unknown")
        _identifier(step["active_state"], field="trace active state")
        _sha(step["physical_action_sha256"], field="trace physical action SHA")
        if type(step["switch_reason"]) is not str or not 1 <= len(step["switch_reason"]) <= 256:
            raise TraceIntegrityError("persisted trace switch reason is malformed")
        signals = step["signals"]
        post = step["post_boundary"]
        flags = step["switch_flags"]
        if (
            type(signals) is not dict
            or set(signals) != ALLOWED_SIGNALS
            or signals["t"] != index
            or type(signals["dwell"]) is not int
            or not 0 <= signals["dwell"] <= task.horizon_steps
            or type(post) is not dict
            or set(post) != post_keys
            or type(post["truncated"]) is not bool
            or type(flags) is not dict
            or set(flags) != flag_keys
            or any(type(value) is not bool for value in flags.values())
        ):
            raise TraceIntegrityError("persisted trace nested step schema differs")
        for name in ALLOWED_SIGNALS - {"t", "dwell"}:
            _finite(signals[name], field=f"trace signal {name}")
        for name in post_keys - {"truncated"}:
            _finite(post[name], field=f"trace post-boundary {name}")
        if index == 0:
            initial_fallen = (
                not 1.0 < float(signals["z_root"]) < 2.0 or float(signals["torso_up"]) < 0.5
            )
            if initial_fallen:
                first_fall = 0
            previous_behavior = behavior
        fallen = not 1.0 < float(post["z_root"]) < 2.0 or float(post["torso_up"]) < 0.5
        if fallen:
            fallen_boundaries.append(index + 1)
            if first_fall is None:
                first_fall = index + 1
        errors.append(abs(float(post["v_x"]) - float(signals["v_target"])))
        rewards.append(float(post["task_reward"]))
        counts[behavior] += 1
        active_by_step.append(behavior)
        if flags["controller_switched"]:
            switch_speed = 0.0 if index == 0 else float(steps[index - 1]["post_boundary"]["v_x"])
            switches.append(
                {
                    "fall_followed_within_100_steps": any(
                        index < boundary <= index + 100 for boundary in fallen_boundaries
                    ),
                    "from_behavior": previous_behavior,
                    "step": index,
                    "to_behavior": behavior,
                    "v_x_m_s": switch_speed,
                }
            )
        previous_behavior = behavior
    # The forward scan above cannot know about later falls when it records a switch.
    for switch in switches:
        switch["fall_followed_within_100_steps"] = first_fall is not None and (
            switch["step"] < first_fall <= switch["step"] + 100
        )
    slow = min(task.schedule, key=lambda segment: segment.target_m_s)
    if sum(segment.target_m_s == slow.target_m_s for segment in task.schedule) != 1:
        raise TraceIntegrityError("task does not define one unique slow segment")
    slow_steps = active_by_step[slow.start : slow.stop]
    denominator = slow.stop - slow.start
    result: dict[str, object] = {
        "fall": initial_fallen or bool(fallen_boundaries),
        "first_fall_step": first_fall,
        "mean_absolute_speed_error_m_s": math.fsum(errors) / len(errors),
        "observed_steps": len(steps),
        "switch_count": len(switches),
        "task_return": math.fsum(rewards),
        "time_in_each_behavior_steps": counts,
    }
    if diagnostics:
        result.update(
            {
                "controller_switches": switches,
                "slow_third_behavior_fractions": {
                    name: sum(value == name for value in slow_steps) / denominator
                    for name in behavior_names
                },
            }
        )
    return result


def _require_trace_metrics_match(
    row: Mapping[str, object], recomputed: Mapping[str, object]
) -> None:
    for name, expected in recomputed.items():
        observed = row[name]
        if type(expected) is float:
            matches = type(observed) in {int, float} and math.isclose(
                float(observed), expected, rel_tol=0.0, abs_tol=1e-12
            )
        elif (
            type(expected) is dict
            and expected
            and all(type(value) is float for value in expected.values())
        ):
            matches = set(observed) == set(expected) and all(
                math.isclose(float(observed[key]), value, rel_tol=0.0, abs_tol=1e-12)
                for key, value in expected.items()
            )
        else:
            matches = observed == expected
        if not matches:
            raise TraceIntegrityError(f"protected episode metric {name} differs from trace steps")


def _expected_evaluation(task: TaskSpec) -> dict[str, object]:
    return {
        "episode_steps": task.horizon_steps,
        "fall_definition": "z_root outside (1.0, 2.0) or torso_up < 0.5",
        "falls_excluded": False,
        "forward_speed_quantity": "root-x boundary difference over 0.015 s",
        "primary_score": "median episode mean_absolute_speed_error_m_s",
        "seeds": list(task.seeds),
        "task_return": "sum of untouched stock Humanoid-v5 reward; descriptive only",
    }


def _metric_outcome(delta: float) -> str:
    if delta < 0.0:
        return "component-wise lower"
    if delta > 0.0:
        return "component-wise higher"
    return "component-wise equal"


def _expected_comparisons(
    prior_arms: Sequence[Mapping[str, object]],
    current_arms: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    if not prior_arms:
        return None
    if len(current_arms) != 1:
        raise EvidenceChainError("designer receipt must contain one current arm")
    candidate = current_arms[0]
    candidate_falls = int(candidate["fall_count"])
    candidate_mae = _finite(
        candidate["median_mean_absolute_speed_error_m_s"], field="candidate median MAE"
    )
    comparisons: list[dict[str, object]] = []
    for baseline in prior_arms:
        fall_delta = candidate_falls - int(baseline["fall_count"])
        mae_delta = candidate_mae - _finite(
            baseline["median_mean_absolute_speed_error_m_s"], field="baseline median MAE"
        )
        comparisons.append(
            {
                "baseline_oracle_id": baseline["oracle_id"],
                "fall_count_delta": fall_delta,
                "fall_count_outcome": _metric_outcome(float(fall_delta)),
                "median_mean_absolute_speed_error_delta_m_s": mae_delta,
                "median_mean_absolute_speed_error_outcome": _metric_outcome(mae_delta),
            }
        )
    return {
        "candidate_oracle_id": candidate["oracle_id"],
        "comparisons": comparisons,
        "never_fall_requirement": "met" if candidate_falls == 0 else "did not meet",
        "no_combined_ranking": True,
    }


def _trace_index_entries(
    *,
    receipt: Mapping[str, object],
    repository_root: Path,
    rows: Sequence[Mapping[str, object]],
    task: TaskSpec,
    behavior_names: Sequence[str],
    diagnostics: bool,
) -> None:
    binding = receipt["trace_content_index"]
    if type(binding) is not dict or set(binding) != {"entry_count", "path", "sha256"}:
        raise TraceIntegrityError("trace-index binding keys differ")
    if type(binding["entry_count"]) is not int or binding["entry_count"] != len(rows):
        raise TraceIntegrityError("trace-index entry count differs")
    index_path = _safe_receipt_path(repository_root, binding["path"], field="trace index path")
    try:
        encoded = _regular_bytes(
            index_path, maximum=MAX_TRACE_INDEX_BYTES, field="trace content index"
        )
    except EvidenceChainError as exc:
        raise TraceIntegrityError(str(exc)) from exc
    if hashlib.sha256(encoded).hexdigest() != _sha(binding["sha256"], field="trace index SHA"):
        raise TraceIntegrityError("trace content-index SHA-256 differs")
    try:
        index = _decode_bounded_object(
            encoded, maximum=MAX_TRACE_INDEX_BYTES, source="trace content index"
        )
    except EvidenceChainError as exc:
        raise TraceIntegrityError(str(exc)) from exc
    if encoded != canonical_json_bytes(index):
        raise TraceIntegrityError("trace content index is not canonical JSON")
    schema_version = index.get("schema_version")
    if schema_version not in {1, 2}:
        raise TraceIntegrityError("trace content-index version differs")
    legacy = schema_version == 1
    expected_index_keys = {
        "entries",
        "entry_count",
        "evidence_class",
        "schema_version",
        "trace_index_schema_id",
    } | (set() if legacy else {"identities", "schema_versions"})
    if type(index) is not dict or set(index) != expected_index_keys:
        raise TraceIntegrityError("trace content-index schema differs")
    expected_index_schema_id = (
        "humanoid_composition_trace_index/v1" if legacy else TRACE_INDEX_SCHEMA_ID
    )
    if (
        index["trace_index_schema_id"] != expected_index_schema_id
        or index["evidence_class"] != EVIDENCE_CLASS
    ):
        raise TraceIntegrityError("trace content-index identity differs")
    if not legacy:
        expected_index_identities = {
            "actor_sha256_by_behavior": receipt["identities"]["actors"],
            "archive_loader_sha256": receipt["identities"]["archive_loader"]["sha256"],
            "execution_manifest_sha256": receipt["execution_manifest_sha256"],
            "library_manifest_sha256": receipt["library_manifest_sha256"],
            "metric_core_sha256": receipt["identities"]["metric_core"]["sha256"],
            "oracle_file_sha256_by_id": {
                oracle["oracle_id"]: oracle["file_sha256"] for oracle in receipt["oracles"]
            },
            "oracle_interpreter_sha256": receipt["identities"]["oracle_interpreter"]["sha256"],
            "oracle_sha256_by_id": {
                oracle["oracle_id"]: oracle["oracle_sha256"] for oracle in receipt["oracles"]
            },
            "report_writer_sha256": receipt["identities"]["report_writer"]["sha256"],
            "runtime_fingerprint_sha256": receipt["runtime"]["runtime_fingerprint_sha256"],
            "task_spec_sha256": receipt["task_spec_sha256"],
        }
        if index["identities"] != expected_index_identities or index["schema_versions"] != dict(
            SCHEMA_VERSIONS
        ):
            raise TraceIntegrityError("trace content-index identities differ")
    if index["entry_count"] != len(rows) or type(index["entries"]) is not list:
        raise TraceIntegrityError("trace content-index rows differ")
    by_key: dict[tuple[str, int], Mapping[str, object]] = {}
    for entry in index["entries"]:
        if type(entry) is not dict:
            raise TraceIntegrityError("trace index entry is malformed")
        required = {"byte_count", "cycle", "oracle_id", "path", "seed", "sha256"}
        if set(entry) != required:
            raise TraceIntegrityError("trace index entry keys differ")
        if (
            type(entry["byte_count"]) is not int
            or entry["byte_count"] <= 0
            or type(entry["cycle"]) is not int
            or type(entry["seed"]) is not int
            or entry["cycle"] != receipt["cycle"]
        ):
            raise TraceIntegrityError("trace index entry values differ")
        _identifier(entry["oracle_id"], field="trace index oracle_id")
        _sha(entry["sha256"], field="trace index SHA")
        _safe_receipt_path(Path("."), entry["path"], field="trace index entry path")
        key = (entry["oracle_id"], entry["seed"])
        if key in by_key:
            raise TraceIntegrityError("trace index entry is duplicated")
        by_key[key] = entry
    for row in rows:
        key = (row["oracle_id"], row["seed"])
        entry = by_key.get(key)
        expected = {
            "byte_count": row["trace_byte_count"],
            "cycle": receipt["cycle"],
            "oracle_id": row["oracle_id"],
            "path": row["trace_path"],
            "seed": row["seed"],
            "sha256": row["trace_sha256"],
        }
        if entry != expected:
            raise TraceIntegrityError("trace index and episode evidence differ")
        trace_root = index_path.parent.parent
        trace_path = _safe_receipt_path(trace_root, entry["path"], field="trace path")
        try:
            trace_bytes = _regular_bytes(trace_path, maximum=MAX_TRACE_BYTES, field="trace")
        except EvidenceChainError as exc:
            raise TraceIntegrityError(str(exc)) from exc
        if (
            len(trace_bytes) != entry["byte_count"]
            or hashlib.sha256(trace_bytes).hexdigest() != entry["sha256"]
        ):
            raise TraceIntegrityError("persisted trace bytes fail their index binding")
        try:
            trace = _decode_bounded_object(
                trace_bytes, maximum=MAX_TRACE_BYTES, source="persisted trace"
            )
        except EvidenceChainError as exc:
            raise TraceIntegrityError(str(exc)) from exc
        expected_trace_keys = {
            "evidence_class",
            "oracle_id",
            "oracle_sha256",
            "runtime_fingerprint_sha256",
            "schema_version",
            "seed",
            "steps",
            "trace_schema_id",
        } | (set() if legacy else {"identities", "schema_versions"})
        if set(trace) != expected_trace_keys:
            raise TraceIntegrityError("persisted trace header schema differs")
        expected_trace_schema_id = (
            "humanoid_controller_switching_trace/v1" if legacy else TRACE_SCHEMA_ID
        )
        if (
            trace["schema_version"] != schema_version
            or trace["trace_schema_id"] != expected_trace_schema_id
            or trace["evidence_class"] != EVIDENCE_CLASS
            or type(trace["steps"]) is not list
            or len(trace["steps"]) != row["observed_steps"]
            or trace["oracle_id"] != row["oracle_id"]
            or trace["oracle_sha256"] != row["oracle_sha256"]
            or trace["seed"] != row["seed"]
            or trace["runtime_fingerprint_sha256"]
            != receipt["runtime"]["runtime_fingerprint_sha256"]
        ):
            raise TraceIntegrityError("persisted trace header identity differs")
        if not legacy:
            expected_identities = {
                "actor_sha256_by_behavior": receipt["identities"]["actors"],
                "archive_loader_sha256": receipt["identities"]["archive_loader"]["sha256"],
                "execution_manifest_sha256": receipt["execution_manifest_sha256"],
                "library_manifest_sha256": receipt["library_manifest_sha256"],
                "metric_core_sha256": receipt["identities"]["metric_core"]["sha256"],
                "oracle_file_sha256": row["oracle_file_sha256"],
                "oracle_interpreter_sha256": receipt["identities"]["oracle_interpreter"]["sha256"],
                "oracle_sha256": row["oracle_sha256"],
                "report_writer_sha256": receipt["identities"]["report_writer"]["sha256"],
                "runtime_fingerprint_sha256": receipt["runtime"]["runtime_fingerprint_sha256"],
                "task_spec_sha256": receipt["task_spec_sha256"],
            }
            if trace["identities"] != expected_identities or trace["schema_versions"] != dict(
                SCHEMA_VERSIONS
            ):
                raise TraceIntegrityError("persisted trace authority identities differ")
        _require_trace_metrics_match(
            row,
            _trace_metrics(
                trace,
                task=task,
                behavior_names=behavior_names,
                diagnostics=diagnostics,
            ),
        )
    if len(by_key) != len(rows):
        raise TraceIntegrityError("trace index contains unreported evidence")


def validate_scientific_receipt(
    path: Path,
    *,
    experiment: Path,
    repository_root: Path,
    library: LibraryManifest,
    task: TaskSpec,
    expected_cycle: int,
    expected_metric_core_sha256: str | None,
    verify_traces: bool = True,
) -> ValidatedScientificReceipt:
    """Validate one typed receipt, recompute summaries, and audit its trace index."""

    if not 0 <= expected_cycle <= MAX_CYCLES:
        raise EvidenceChainError("cycle lies outside the receipt-chain bound")
    encoded = _regular_bytes(Path(path), maximum=MAX_REPORT_BYTES, field="scientific receipt")
    receipt = _decode_bounded_object(encoded, maximum=MAX_REPORT_BYTES, source="scientific receipt")
    if encoded != canonical_json_bytes(receipt):
        raise EvidenceChainError("scientific receipt is not canonical JSON")
    required = {
        "claim_ceiling",
        "comparisons",
        "cycle",
        "determinism_check",
        "designer_provenance",
        "evaluation",
        "evidence_class",
        "execution_manifest_sha256",
        "identities",
        "library_manifest_sha256",
        "oracles",
        "per_episode",
        "prior_scientific_receipt",
        "report_schema_id",
        "runtime",
        "schema_version",
        "schema_versions",
        "scientific_receipt_schema_id",
        "source_report",
        "summary",
        "task_spec_sha256",
        "trace_content_index",
    }
    if set(receipt) != required:
        raise EvidenceChainError("scientific receipt keys differ from the exact schema")
    legacy_source = receipt["source_report"] is not None
    if (
        receipt["schema_version"] != 2
        or receipt["scientific_receipt_schema_id"] != SCIENTIFIC_RECEIPT_SCHEMA_ID
        or receipt["report_schema_id"]
        != (LEGACY_REPORT_SCHEMA_ID if legacy_source else REPORT_SCHEMA_ID)
        or receipt["cycle"] != expected_cycle
        or receipt["evidence_class"] != EVIDENCE_CLASS
        or receipt["task_spec_sha256"] != task.raw_sha256
        or receipt["library_manifest_sha256"] != library.raw_sha256
        or receipt["schema_versions"]
        != dict(
            LEGACY_PHASE_A_SCHEMA_VERSIONS
            if receipt["source_report"] is not None
            else SCHEMA_VERSIONS
        )
    ):
        raise EvidenceChainError("scientific receipt identity differs")
    execution_manifest = _regular_bytes(
        Path(experiment) / E003_EXECUTION_MANIFEST_FILENAME,
        maximum=MAX_REPORT_BYTES,
        field="execution manifest",
    )
    if hashlib.sha256(execution_manifest).hexdigest() != _sha(
        receipt["execution_manifest_sha256"], field="execution manifest SHA"
    ):
        raise EvidenceChainError("scientific receipt execution-manifest identity differs")
    try:
        execution_manifest_value_read = _decode_bounded_object(
            execution_manifest,
            maximum=MAX_REPORT_BYTES,
            source="execution manifest",
        )
    except EvidenceChainError:
        raise
    if execution_manifest != canonical_json_bytes(execution_manifest_value_read) or (
        execution_manifest_value_read != execution_manifest_value(library, task)
    ):
        raise EvidenceChainError("execution manifest no longer seals the loaded task and library")
    identities = receipt["identities"]
    if type(identities) is not dict or set(identities) != {
        "actors",
        "archive_loader",
        "metric_core",
        "oracle_interpreter",
        "report_writer",
    }:
        raise EvidenceChainError("authority identity set differs")
    expected_actors = {entry.name: entry.strict_npz.sha256 for entry in library.behaviors}
    if identities["actors"] != expected_actors:
        raise EvidenceChainError("actor identities differ from the library")
    for name in ("archive_loader", "metric_core", "oracle_interpreter", "report_writer"):
        identity = identities[name]
        if type(identity) is not dict or set(identity) != {
            "identity_id",
            "sha256",
            "source_sha256",
        }:
            raise EvidenceChainError(f"{name} identity schema differs")
        core = {"identity_id": identity["identity_id"], "source_sha256": identity["source_sha256"]}
        if hashlib.sha256(canonical_json_bytes(core)).hexdigest() != identity["sha256"]:
            raise EvidenceChainError(f"{name} identity digest differs")
    metric_core = identities["metric_core"]["sha256"]
    if expected_metric_core_sha256 is not None and metric_core != expected_metric_core_sha256:
        raise EvidenceChainError("prior receipt metric-core identity differs")
    runtime = receipt["runtime"]
    if type(runtime) is not dict or set(runtime) != {
        "runtime_fingerprint",
        "runtime_fingerprint_sha256",
    }:
        raise EvidenceChainError("runtime identity schema differs")
    if hashlib.sha256(canonical_json_bytes(runtime["runtime_fingerprint"])).hexdigest() != _sha(
        runtime["runtime_fingerprint_sha256"], field="runtime fingerprint SHA"
    ):
        raise EvidenceChainError("runtime fingerprint digest differs")
    oracles = receipt["oracles"]
    rows = receipt["per_episode"]
    if type(oracles) is not list or not oracles or type(rows) is not list or not rows:
        raise EvidenceChainError("receipt oracle or episode evidence is missing")
    for oracle in oracles:
        if type(oracle) is not dict or set(oracle) != {
            "file_sha256",
            "oracle_id",
            "oracle_sha256",
            "path",
        }:
            raise EvidenceChainError("receipt oracle binding is malformed")
        _identifier(oracle["oracle_id"], field="oracle_id")
        _sha(oracle["file_sha256"], field="oracle file SHA")
        _sha(oracle["oracle_sha256"], field="oracle canonical SHA")
        oracle_path = _safe_receipt_path(Path(experiment), oracle["path"], field="oracle path")
        try:
            program, file_sha256 = load_oracle_program(
                oracle_path,
                available_behaviors=library.behavior_names,
            )
        except (OracleContractError, OSError) as exc:
            raise EvidenceChainError("bound oracle file is invalid") from exc
        if (
            file_sha256 != oracle["file_sha256"]
            or program.sha256 != oracle["oracle_sha256"]
            or program.oracle_id != oracle["oracle_id"]
        ):
            raise EvidenceChainError("bound oracle file identity differs")
    provenance = receipt["designer_provenance"]
    if expected_cycle == 0:
        if provenance is not None:
            raise EvidenceChainError("cycle zero cannot bind designer provenance")
    else:
        if type(provenance) is not dict or set(provenance) != {"path", "sha256"}:
            raise EvidenceChainError("designer-provenance binding differs")
        validated_provenance = validate_designer_provenance(
            experiment=experiment,
            repository_root=repository_root,
            cycle=expected_cycle,
            oracle_id=oracles[0]["oracle_id"],
            oracle_file_sha256=oracles[0]["file_sha256"],
            oracle_sha256=oracles[0]["oracle_sha256"],
        )
        if (
            provenance["path"] != validated_provenance.path.relative_to(experiment).as_posix()
            or provenance["sha256"] != validated_provenance.sha256
        ):
            raise EvidenceChainError("designer-provenance identity differs")
    source_report = receipt["source_report"]
    source_report_value: Mapping[str, object] | None = None
    if source_report is not None:
        if type(source_report) is not dict or set(source_report) != {
            "path",
            "report_schema_id",
            "schema_version",
            "sha256",
        }:
            raise EvidenceChainError("legacy source-report binding differs")
        source_path = _safe_receipt_path(
            Path(experiment), source_report["path"], field="legacy source report path"
        )
        source_bytes = _regular_bytes(
            source_path, maximum=MAX_REPORT_BYTES, field="legacy source report"
        )
        if hashlib.sha256(source_bytes).hexdigest() != _sha(
            source_report["sha256"], field="legacy source report SHA"
        ):
            raise EvidenceChainError("legacy source-report SHA-256 differs")
        if (
            source_report["schema_version"] != 1
            or source_report["report_schema_id"] != LEGACY_REPORT_SCHEMA_ID
            or source_report["sha256"] != _LEGACY_REPORT_SHA256[expected_cycle]
            or source_report["path"]
            != f"cycles/cycle_{expected_cycle}/report_{expected_cycle}.json"
        ):
            raise EvidenceChainError("legacy source-report schema differs")
        source_report_value = _decode_bounded_object(
            source_bytes, maximum=MAX_REPORT_BYTES, source="legacy source report"
        )
    expected_claim = (
        _LEGACY_CYCLE_ZERO_CLAIM_CEILING
        if source_report is not None and expected_cycle == 0
        else CLAIM_CEILING
    )
    if receipt["claim_ceiling"] != expected_claim:
        raise EvidenceChainError("scientific receipt claim ceiling differs")
    if receipt["evaluation"] != _expected_evaluation(task):
        raise EvidenceChainError("scientific receipt evaluation contract differs")
    expected_identity = (
        _legacy_authority_identities(expected_cycle, library)
        if source_report is not None
        else {**current_authority_identities(), "actors": expected_actors}
    )
    if identities != expected_identity:
        raise EvidenceChainError("scientific receipt authority identities differ")
    expected_determinism = {
        "all_trace_hashes_equal": True,
        "oracle_id": oracles[0]["oracle_id"],
        "replayed_episode_count": len(task.seeds),
    }
    if receipt["determinism_check"] != expected_determinism:
        raise EvidenceChainError("scientific receipt determinism evidence differs")
    diagnostics = source_report is None or expected_cycle > 0
    checked_rows = [
        _validate_episode_row(
            row,
            task=task,
            behavior_names=library.behavior_names,
            diagnostics=diagnostics,
        )
        for row in rows
    ]
    if source_report_value is not None:
        source_rows = source_report_value.get("per_episode")
        if type(source_rows) is not list:
            raise EvidenceChainError("legacy source report omits episode evidence")
        migrated_rows = []
        for item in source_rows:
            if type(item) is not dict or "episode_wall_time_seconds" not in item:
                raise EvidenceChainError("legacy source-report episode schema differs")
            migrated = dict(item)
            migrated.pop("episode_wall_time_seconds")
            migrated_rows.append(migrated)
        if migrated_rows != checked_rows:
            raise EvidenceChainError(
                "migrated episode evidence differs from the frozen source report"
            )
    oracle_ids = [oracle["oracle_id"] for oracle in oracles]
    if expected_cycle == 0:
        if tuple(oracle_ids) != task.cycle_zero_oracle_ids:
            raise EvidenceChainError("cycle-zero oracle history differs")
    elif oracle_ids != [f"cycle_{expected_cycle}_candidate"]:
        raise EvidenceChainError("designer-cycle oracle identity differs")
    expected_episode_keys = {(oracle_id, seed) for oracle_id in oracle_ids for seed in task.seeds}
    observed_episode_keys = {(row["oracle_id"], row["seed"]) for row in checked_rows}
    if observed_episode_keys != expected_episode_keys or len(checked_rows) != len(
        expected_episode_keys
    ):
        raise EvidenceChainError("per-episode evidence does not cover the exact arm and seed set")
    for row in checked_rows:
        oracle = next(item for item in oracles if item["oracle_id"] == row["oracle_id"])
        if (
            row["oracle_file_sha256"] != oracle["file_sha256"]
            or row["oracle_sha256"] != oracle["oracle_sha256"]
        ):
            raise EvidenceChainError("episode and oracle identities differ")
    summary = receipt["summary"]
    if type(summary) is not dict or set(summary) != {"arms"} or type(summary["arms"]) is not list:
        raise EvidenceChainError("receipt summary schema differs")
    arms = summary["arms"]
    prior = receipt["prior_scientific_receipt"]
    prior_arms: list[Mapping[str, object]] = []
    if expected_cycle == 0:
        if prior is not None:
            raise EvidenceChainError("cycle zero cannot bind prior scientific evidence")
    else:
        if type(prior) is not dict or set(prior) != {"cycle", "path", "sha256"}:
            raise EvidenceChainError("prior scientific-receipt binding differs")
        if prior["cycle"] != expected_cycle - 1:
            raise EvidenceChainError("prior scientific-receipt cycle differs")
        prior_path = _safe_receipt_path(
            Path(experiment), prior["path"], field="prior scientific receipt path"
        )
        validated_prior = validate_scientific_receipt(
            prior_path,
            experiment=experiment,
            repository_root=repository_root,
            library=library,
            task=task,
            expected_cycle=expected_cycle - 1,
            expected_metric_core_sha256=metric_core,
            verify_traces=verify_traces,
        )
        if validated_prior.sha256 != prior["sha256"]:
            raise EvidenceChainError("prior scientific-receipt SHA-256 differs")
        prior_arms = [item.value for item in validated_prior.arms]
    recomputed = []
    for oracle in oracles:
        oracle_rows = [row for row in checked_rows if row["oracle_id"] == oracle["oracle_id"]]
        recomputed.append(
            _summary_from_rows(
                oracle=oracle,
                rows=oracle_rows,
                behavior_names=library.behavior_names,
            )
        )
    expected_arms = [*prior_arms, *recomputed]
    if arms != expected_arms:
        raise EvidenceChainError("receipt summary does not recompute from per-episode evidence")
    expected_history = [*task.cycle_zero_oracle_ids]
    expected_history.extend(f"cycle_{index}_candidate" for index in range(1, expected_cycle + 1))
    if [arm.get("oracle_id") for arm in arms] != expected_history:
        raise EvidenceChainError("receipt arm history differs from the exact cycle chain")
    if receipt["comparisons"] != _expected_comparisons(prior_arms, recomputed):
        raise EvidenceChainError("receipt comparisons do not recompute from the arm summaries")
    if verify_traces:
        _trace_index_entries(
            receipt=receipt,
            repository_root=Path(repository_root),
            rows=checked_rows,
            task=task,
            behavior_names=library.behavior_names,
            diagnostics=diagnostics,
        )
    return ValidatedScientificReceipt(
        path=Path(path),
        value=MappingProxyType(receipt),
        encoded=encoded,
        sha256=hashlib.sha256(encoded).hexdigest(),
        arms=tuple(ValidatedArmSummary(MappingProxyType(dict(arm))) for arm in arms),
    )


def prior_receipt_path(experiment: Path, cycle: int) -> Path:
    directory = Path(experiment) / "cycles" / f"cycle_{cycle}"
    migrated = directory / "scientific_receipt_v2.json"
    return migrated if migrated.exists() else directory / f"report_{cycle}.json"


def validate_prior_report_chain(
    *,
    experiment: Path,
    repository_root: Path,
    library: LibraryManifest,
    task: TaskSpec,
    prior_cycle: int,
    expected_metric_core_sha256: str,
) -> ValidatedScientificReceipt:
    """The sole prior-history validator used by both prepare and evaluate."""

    return validate_scientific_receipt(
        prior_receipt_path(experiment, prior_cycle),
        experiment=experiment,
        repository_root=repository_root,
        library=library,
        task=task,
        expected_cycle=prior_cycle,
        expected_metric_core_sha256=expected_metric_core_sha256,
    )


def _component_labels(comparison: object) -> object:
    if comparison is None:
        return None
    if type(comparison) is not dict:
        raise EvidenceChainError("legacy comparison is malformed")
    result = json.loads(json.dumps(comparison, allow_nan=False))
    labels = {
        "improved": "component-wise lower",
        "matched": "component-wise equal",
        "worsened": "component-wise higher",
    }
    for row in result["comparisons"]:
        row["fall_count_outcome"] = labels[row["fall_count_outcome"]]
        row["median_mean_absolute_speed_error_outcome"] = labels[
            row["median_mean_absolute_speed_error_outcome"]
        ]
    result["never_fall_requirement"] = {
        "failed": "did not meet",
        "passed": "met",
    }[result["never_fall_requirement"]]
    return result


def legacy_phase_a_artifacts(*, experiment: Path, repository_root: Path) -> dict[Path, bytes]:
    """Derive immutable deterministic receipts and separate telemetry from v1 reports."""

    directory = Path(experiment)
    library, task = load_frozen_inputs(directory)
    _require_e003_literals(library, task)
    manifest_bytes = _regular_bytes(
        directory / E003_EXECUTION_MANIFEST_FILENAME,
        maximum=MAX_REPORT_BYTES,
        field="execution manifest",
    )
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    result: dict[Path, bytes] = {}
    prior_path: Path | None = None
    prior_sha256: str | None = None
    for cycle in range(3):
        cycle_directory = directory / "cycles" / f"cycle_{cycle}"
        source_path = cycle_directory / f"report_{cycle}.json"
        source_bytes = _regular_bytes(
            source_path, maximum=MAX_REPORT_BYTES, field=f"cycle-{cycle} source report"
        )
        if hashlib.sha256(source_bytes).hexdigest() != _LEGACY_REPORT_SHA256[cycle]:
            raise EvidenceChainError(f"cycle-{cycle} source report bytes changed")
        source = _decode_bounded_object(
            source_bytes, maximum=MAX_REPORT_BYTES, source=f"cycle-{cycle} source report"
        )
        per_episode = [
            {key: value for key, value in row.items() if key != "episode_wall_time_seconds"}
            for row in source["per_episode"]
        ]
        comparison_key = "comparison_to_cycle_zero" if cycle == 1 else "comparison_to_prior_arms"
        prior_binding = None
        if prior_path is not None and prior_sha256 is not None:
            prior_binding = {
                "cycle": cycle - 1,
                "path": prior_path.relative_to(directory).as_posix(),
                "sha256": prior_sha256,
            }
        receipt = {
            "claim_ceiling": source["claim_ceiling"],
            "comparisons": _component_labels(source.get(comparison_key)),
            "cycle": cycle,
            "determinism_check": {
                key: value
                for key, value in source["determinism_check"].items()
                if key != "wall_time_seconds"
            },
            "designer_provenance": (
                None
                if cycle == 0
                else {
                    "path": f"cycles/cycle_{cycle}/designer_provenance.json",
                    "sha256": hashlib.sha256(
                        _regular_bytes(
                            cycle_directory / "designer_provenance.json",
                            maximum=MAX_REPORT_BYTES,
                            field="designer provenance",
                        )
                    ).hexdigest(),
                }
            ),
            "evaluation": source["evaluation"],
            "evidence_class": EVIDENCE_CLASS,
            "execution_manifest_sha256": manifest_sha256,
            "identities": _legacy_authority_identities(cycle, library),
            "library_manifest_sha256": source["library_manifest_sha256"],
            "oracles": source["oracles"],
            "per_episode": per_episode,
            "prior_scientific_receipt": prior_binding,
            "report_schema_id": LEGACY_REPORT_SCHEMA_ID,
            "runtime": {
                "runtime_fingerprint": source["runtime_fingerprint"],
                "runtime_fingerprint_sha256": source["runtime_fingerprint_sha256"],
            },
            "schema_version": 2,
            "schema_versions": dict(LEGACY_PHASE_A_SCHEMA_VERSIONS),
            "scientific_receipt_schema_id": SCIENTIFIC_RECEIPT_SCHEMA_ID,
            "source_report": {
                "path": source_path.relative_to(directory).as_posix(),
                "report_schema_id": source["report_schema_id"],
                "schema_version": source["schema_version"],
                "sha256": _LEGACY_REPORT_SHA256[cycle],
            },
            "summary": source["summary"],
            "task_spec_sha256": source["task_spec_sha256"],
            "trace_content_index": source["trace_content_index"],
        }
        receipt_path = cycle_directory / "scientific_receipt_v2.json"
        receipt_bytes = canonical_json_bytes(receipt)
        result[receipt_path] = receipt_bytes
        prior_path = receipt_path
        prior_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
        telemetry = {
            "cycle": cycle,
            "determinism_replay_wall_time_seconds": source["determinism_check"][
                "wall_time_seconds"
            ],
            "episode_wall_times": [
                {
                    "episode_wall_time_seconds": row["episode_wall_time_seconds"],
                    "oracle_id": row["oracle_id"],
                    "seed": row["seed"],
                }
                for row in source["per_episode"]
            ],
            "evidence_class": EVIDENCE_CLASS,
            "generated_utc": source["generated_utc"],
            "host": {
                "machine": source["runtime_fingerprint"]["platform_machine"],
                "node": "not_recorded",
                "system": "not_recorded",
            },
            "schema_version": 1,
            "scientific_receipt_sha256": prior_sha256,
            "source_report_sha256": _LEGACY_REPORT_SHA256[cycle],
            "telemetry_schema_id": TELEMETRY_SCHEMA_ID,
            "total_wall_time_seconds": source["wall_time_seconds"],
        }
        result[cycle_directory / "telemetry_v1.json"] = canonical_json_bytes(telemetry)
    return result


__all__ = [
    "E003_EXECUTION_MANIFEST_FILENAME",
    "REPORT_SCHEMA_ID",
    "SCHEMA_VERSIONS",
    "SCIENTIFIC_RECEIPT_SCHEMA_ID",
    "TELEMETRY_SCHEMA_ID",
    "TRACE_INDEX_SCHEMA_ID",
    "TRACE_SCHEMA_ID",
    "EvidenceChainError",
    "SealedExecution",
    "TraceIntegrityError",
    "ValidatedScientificReceipt",
    "current_authority_identities",
    "execution_manifest_value",
    "legacy_phase_a_artifacts",
    "load_e003_execution",
    "load_test_execution",
    "prior_receipt_path",
    "validate_prior_report_chain",
    "validate_scientific_receipt",
    "verify_library_statistics",
]
