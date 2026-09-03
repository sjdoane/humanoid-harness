"""Unified command line for research, evidence, and local inspection."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from .research import KnowledgeIndexError, build_index, index_stats, query_index
from .research.knowledge import DEFAULT_DATABASE, DEFAULT_EXTRACTIONS
from .status import doctor_status, program_status
from .traces import TraceContractError, load_trace


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="humanoid-harness",
        description="Inspect and operate the oracle-and-reward research harness.",
    )
    parser.add_argument("--json", action="store_true", help="emit stable JSON")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("status", help="show target, capability, evidence, and next gate")
    commands.add_parser("doctor", help="check required and optional local dependencies")

    research = commands.add_parser("research", help="build and query the research graph")
    research_commands = research.add_subparsers(dest="research_command", required=True)

    build = research_commands.add_parser("build", help="build the local graph atomically")
    build.add_argument("--extractions", type=Path, default=DEFAULT_EXTRACTIONS)
    build.add_argument("--database", type=Path, default=DEFAULT_DATABASE)

    query = research_commands.add_parser("query", help="search mechanisms and evidence")
    query.add_argument("query")
    query.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    query.add_argument("--limit", type=int, default=10)
    query.add_argument("--include-inactive", action="store_true")

    stats = research_commands.add_parser("stats", help="show graph coverage")
    stats.add_argument("--database", type=Path, default=DEFAULT_DATABASE)

    trace = commands.add_parser("trace", help="validate and inspect a trajectory trace")
    trace_commands = trace.add_subparsers(dest="trace_command", required=True)
    inspect_trace = trace_commands.add_parser(
        "inspect", help="show identity, coverage, and event counts"
    )
    inspect_trace.add_argument("path", type=Path)

    tracker = commands.add_parser("tracker", help="inspect reference-conditioned control")
    tracker_commands = tracker.add_subparsers(dest="tracker_command", required=True)
    probe = tracker_commands.add_parser(
        "probe-reference-use",
        help="run the local matched-observation numeric-reference sensitivity probe",
    )
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache = cache_root / "humanoid-harness"
    data = cache / "minari" / "mujoco" / "humanoid" / "expert-v0" / "data"
    models = cache / "local_models"
    probe.add_argument("--hdf5", type=Path, default=data / "main_data.hdf5")
    probe.add_argument("--metadata", type=Path, default=data / "metadata.json")
    probe.add_argument("--base-controller", type=Path, default=models / "minari_bc_v0.npz")
    probe.add_argument(
        "--residual-controller",
        type=Path,
        default=models / "reference_residual_ppo_v0.npz",
    )
    probe.add_argument("--output", type=Path, required=True)

    ui = commands.add_parser("ui", help="serve the read-only local evidence UI")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    ui.add_argument("--no-open", action="store_true", help="do not open a browser")
    return parser


def _print_human(result: object) -> None:
    if isinstance(result, list):
        if not result:
            print("No matching research records.")
            return
        for index, item in enumerate(result, start=1):
            if isinstance(item, dict):
                print(f"{index}. [{item.get('kind')}] {item.get('title')}")
                print(f"   {item.get('description')}")
                if item.get("source_url"):
                    print(f"   {item['source_url']}")
            else:
                print(item)
        return
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
        return
    print(result)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            result = program_status()
        elif args.command == "doctor":
            result = doctor_status()
        elif args.command == "research":
            if args.research_command == "build":
                result = build_index(args.extractions, args.database)
            elif args.research_command == "query":
                result = query_index(
                    args.query,
                    args.database,
                    limit=args.limit,
                    include_inactive=args.include_inactive,
                )
            else:
                result = index_stats(args.database)
        elif args.command == "trace":
            trace = load_trace(args.path)
            result = {
                "trace_id": trace.trace_id,
                "sha256": trace.sha256,
                "schema_version": trace.schema_version,
                "evidence_class": trace.evidence_class.value,
                "sample_count": len(trace.samples),
                "event_count": len(trace.events),
                "duration_seconds": (len(trace.samples) - 1) * trace.control_period_seconds,
                "diagnostic_coverage": trace.diagnostic_coverage,
                "artifact_bindings": [binding.to_dict() for binding in trace.artifact_bindings],
            }
        elif args.command == "tracker":
            from .experiments.reference_causal_probe import run_reference_causal_probe

            result = run_reference_causal_probe(
                hdf5_path=args.hdf5,
                metadata_path=args.metadata,
                base_controller_path=args.base_controller,
                residual_controller_path=args.residual_controller,
                output_path=args.output,
            ).to_dict()
        else:
            from .ui.server import serve

            serve(
                host=args.host,
                port=args.port,
                database=args.database,
                open_browser=not args.no_open,
            )
            return 0
    except (KnowledgeIndexError, TraceContractError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    else:
        _print_human(result)
    if args.command == "doctor" and isinstance(result, dict):
        return 0 if result["status"] == "pass" else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
