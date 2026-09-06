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
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "checkpoint":
        result: object = convert_checkpoint(args.source, args.output)
    elif args.command == "motions":
        result = convert_motion_directory(args.source_directory, args.output_directory)
    else:
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
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
