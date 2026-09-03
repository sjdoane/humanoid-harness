"""Honest program status shared by the CLI and UI."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import platform
import re
import sys
from collections.abc import Mapping
from pathlib import Path

from .research import KnowledgeIndexError, index_stats
from .research.knowledge import DEFAULT_DATABASE, DEFAULT_EXTRACTIONS

SUBSTRATE_RECEIPT = Path(
    "experiments/001_humanoid_fixed_reference/receipts/"
    "2026-09-02_substrate_smoke.json"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _substrate_evidence(root: Path) -> list[dict[str, str]]:
    path = root / SUBSTRATE_RECEIPT
    try:
        source_bytes = path.read_bytes()
        receipt = json.loads(source_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    if not isinstance(receipt, Mapping):
        return []
    config = receipt.get("config")
    if not isinstance(config, Mapping):
        return []
    valid = (
        receipt.get("schema_version") == 1
        and not isinstance(receipt.get("schema_version"), bool)
        and receipt.get("evidence_class") == "interface_check"
        and isinstance(receipt.get("claim"), str)
        and config.get("env_id") == "Humanoid-v5"
        and receipt.get("observation_shape") == [348]
        and receipt.get("action_shape") == [17]
        and receipt.get("qpos_shape") == [24]
        and receipt.get("qvel_shape") == [23]
        and receipt.get("observation_finite") is True
        and receipt.get("next_observation_finite") is True
        and receipt.get("reward_finite") is True
        and isinstance(receipt.get("model_sha256"), str)
        and _SHA256.fullmatch(receipt["model_sha256"])
    )
    if not valid:
        return []
    return [
        {
            "evidence_class": "interface_check",
            "claim": "Humanoid-v5 reset/step interface check",
            "claim_ceiling": "simulator interface only",
            "receipt": str(path.resolve()),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        }
    ]


def program_status(
    *,
    project_root: Path | None = None,
    database: Path | None = None,
) -> dict[str, object]:
    """Return claims separated by target, capability, and evidence."""

    root = (project_root or Path.cwd()).resolve()
    database_path = (database or root / DEFAULT_DATABASE).resolve()
    evidence_receipts = _substrate_evidence(root)
    graph: dict[str, object]
    try:
        graph = {"state": "built", **index_stats(database_path)}
    except KnowledgeIndexError:
        source = root / DEFAULT_EXTRACTIONS
        graph = {
            "state": "not_built",
            "database": str(database_path),
            "source_records": len(
                tuple(path for path in source.glob("*.json") if not path.name.startswith("_"))
            )
            if source.is_dir()
            else 0,
        }

    return {
        "program": "humanoid-harness",
        "research_target": (
            "Autonomous, steerable iteration over a state-aware reference oracle "
            "and executable task reward."
        ),
        "implemented_capability": [
            "immutable reference and oracle contracts",
            "deterministic guarded oracle runtime",
            "Gymnasium Humanoid development adapter",
            "protected mechanical evaluator",
            "provenance-bearing research index",
            "bounded canonical trajectory-trace contract",
            "read-only local evidence UI",
        ],
        "measured_evidence": [item["claim"] for item in evidence_receipts],
        "evidence_receipts": evidence_receipts,
        "not_demonstrated": [
            "trained reference-conditioned tracker",
            "causal use of numeric reference windows",
            "adapter-emitted canonical trajectory trace",
            "oracle improvement",
            "task-reward improvement",
            "autonomous evidence-to-candidate loop",
            "cross-MDP generality",
        ],
        "current_bottleneck": (
            "No qualifying trained tracker exists, so current falling-rollout evidence "
            "cannot test oracle quality."
        ),
        "next_gate": (
            "Train or admit one frozen reference-conditioned tracker, then pass exact, "
            "zero, shuffled, and time-shifted causal-use ablations."
        ),
        "knowledge_graph": graph,
        "evidence_class": "interface_check" if evidence_receipts else "none",
    }


def doctor_status(*, project_root: Path | None = None) -> dict[str, object]:
    """Inspect local requirements without mutating the environment."""

    root = (project_root or Path.cwd()).resolve()
    checks = [
        {
            "id": "python",
            "status": "pass" if (3, 12) <= sys.version_info[:2] < (3, 14) else "fail",
            "detail": platform.python_version(),
            "required": True,
        },
        {
            "id": "project_charter",
            "status": "pass" if (root / "docs/PROJECT_CHARTER.md").is_file() else "fail",
            "detail": str(root / "docs/PROJECT_CHARTER.md"),
            "required": True,
        },
        {
            "id": "research_corpus",
            "status": "pass" if (root / DEFAULT_EXTRACTIONS).is_dir() else "fail",
            "detail": str(root / DEFAULT_EXTRACTIONS),
            "required": True,
        },
        {
            "id": "gymnasium",
            "status": "pass" if importlib.util.find_spec("gymnasium") else "warn",
            "detail": "optional Gymnasium adapter",
            "required": False,
        },
        {
            "id": "stable_baselines3",
            "status": "pass" if importlib.util.find_spec("stable_baselines3") else "warn",
            "detail": "optional PPO development adapter",
            "required": False,
        },
    ]
    required_failures = [
        check["id"] for check in checks if check["required"] and check["status"] == "fail"
    ]
    return {
        "status": "fail" if required_failures else "pass",
        "required_failures": required_failures,
        "checks": checks,
    }
