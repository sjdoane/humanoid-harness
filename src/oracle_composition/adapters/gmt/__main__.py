"""Command line entry point for pinned GMT artifact conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .checkpoint import convert_checkpoint, verify_upstream_root
from .motions import convert_motion_directory


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert pinned GMT artifacts without pickle execution"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    checkpoint = subparsers.add_parser("checkpoint")
    checkpoint.add_argument("source", type=Path)
    checkpoint.add_argument("output", type=Path)

    motions = subparsers.add_parser("motions")
    motions.add_argument("source_directory", type=Path)
    motions.add_argument("output_directory", type=Path)

    bundle = subparsers.add_parser("bundle")
    bundle.add_argument("upstream_root", type=Path)
    bundle.add_argument("output_directory", type=Path)

    replay = subparsers.add_parser("replay")
    replay.add_argument("--upstream-root", type=Path, required=True)
    replay.add_argument("--weights", type=Path, required=True)
    replay.add_argument("--weights-sha256", required=True)
    replay.add_argument("--motion", type=Path, required=True)
    replay.add_argument("--motion-sha256", required=True)
    replay.add_argument("--motion-name", required=True)
    replay.add_argument("--trace", type=Path, required=True)
    replay.add_argument("--duration-seconds", type=float, default=10.0)

    sensitivity = subparsers.add_parser(
        "reference-sensitivity",
        help="measure matched-state actor sensitivity to reference-only interventions",
    )
    sensitivity.add_argument("--trace", type=Path, required=True)
    sensitivity.add_argument("--trace-sha256", required=True)
    sensitivity.add_argument("--manifest", type=Path, required=True)
    sensitivity.add_argument("--manifest-sha256", required=True)
    sensitivity.add_argument("--upstream-root", type=Path, required=True)
    sensitivity.add_argument("--weights", type=Path, required=True)
    sensitivity.add_argument("--weights-sha256", required=True)
    sensitivity.add_argument("--motion", type=Path, required=True)
    sensitivity.add_argument("--motion-sha256", required=True)
    sensitivity.add_argument("--motion-name", required=True)
    sensitivity.add_argument("--shuffle-seed", type=int, required=True)
    sensitivity.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "checkpoint":
        result: object = convert_checkpoint(args.source, args.output)
    elif args.command == "motions":
        result = convert_motion_directory(args.source_directory, args.output_directory)
    elif args.command == "bundle":
        support_files = verify_upstream_root(args.upstream_root)
        checkpoint = convert_checkpoint(
            args.upstream_root / "assets/pretrained_checkpoints/pretrained.pt",
            args.output_directory / "gmt_g1_actor_weights.npz",
        )
        motions = convert_motion_directory(
            args.upstream_root / "assets/motions",
            args.output_directory / "motions",
        )
        result = {
            "checkpoint": checkpoint,
            "motions": motions,
            "support_files": support_files,
        }
    elif args.command == "replay":
        from .replay import ReplayConfig, run_headless_replay

        result = run_headless_replay(
            ReplayConfig(
                upstream_root=args.upstream_root,
                weights_path=args.weights,
                weights_sha256=args.weights_sha256,
                motion_path=args.motion,
                motion_sha256=args.motion_sha256,
                motion_name=args.motion_name,
                trace_path=args.trace,
                duration_seconds=args.duration_seconds,
            )
        )
    else:
        from .reference_sensitivity import (
            ReferenceSensitivityConfig,
            run_reference_sensitivity,
        )

        result = run_reference_sensitivity(
            ReferenceSensitivityConfig(
                trace_path=args.trace,
                trace_sha256=args.trace_sha256,
                manifest_path=args.manifest,
                manifest_sha256=args.manifest_sha256,
                upstream_root=args.upstream_root,
                weights_path=args.weights,
                weights_sha256=args.weights_sha256,
                motion_path=args.motion,
                motion_sha256=args.motion_sha256,
                motion_name=args.motion_name,
                shuffle_seed=args.shuffle_seed,
                output_path=args.output,
            )
        )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
