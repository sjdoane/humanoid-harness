"""Command-line interface for fixed-reference experiment 001."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from .execution import (
    DEFAULT_RUNS_DIR,
    calibrate_resources,
    emit_manifest_candidate,
    evaluate_checkpoint,
    finalize_reviewed_manifest,
    train_one_seed,
)
from .fixed_reference import ExperimentContractError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="humanoid-fixed-reference",
        description=(
            "Fail-closed calibration, manifest, one-seed training, and complete "
            "evaluation for experiment 001."
        ),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    calibrate = commands.add_parser(
        "calibrate",
        help="measure a proposed design and discard the unserialized model",
    )
    calibrate.add_argument("--design", type=Path, required=True)
    calibrate.add_argument("--seed", type=int, required=True)
    calibrate.add_argument(
        "--environment-steps",
        type=int,
        help="exact rollout multiple; defaults to one n_envs x n_steps rollout",
    )
    calibrate.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)

    manifest = commands.add_parser(
        "emit-manifest-candidate",
        help="inspect a locked design and emit a non-authorizing frozen-manifest candidate",
    )
    manifest.add_argument("--design", type=Path, required=True)
    manifest.add_argument("--calibrated-design", type=Path, required=True)
    manifest.add_argument("--calibration-receipt", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)

    finalize = commands.add_parser(
        "finalize-reviewed-manifest",
        help="record explicit human review and publish the frozen execution manifest",
    )
    finalize.add_argument("--design", type=Path, required=True)
    finalize.add_argument("--calibrated-design", type=Path, required=True)
    finalize.add_argument("--calibration-receipt", type=Path, required=True)
    finalize.add_argument("--candidate", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--reviewer-id", required=True)
    finalize.add_argument("--review-note", required=True)

    train = commands.add_parser(
        "train-one-seed",
        help="revalidate a frozen manifest and train exactly one predetermined seed",
    )
    train.add_argument("--design", type=Path, required=True)
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)

    evaluate = commands.add_parser(
        "evaluate-checkpoint",
        help="evaluate one exact checkpoint on every declared evaluation seed",
    )
    evaluate.add_argument("--design", type=Path, required=True)
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--checkpoint-receipt", type=Path, required=True)
    evaluate.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "calibrate":
            result = calibrate_resources(
                design_path=args.design,
                calibration_seed=args.seed,
                environment_steps=args.environment_steps,
                runs_dir=args.runs_dir,
            )
        elif args.command == "emit-manifest-candidate":
            result = emit_manifest_candidate(
                design_path=args.design,
                calibrated_design_path=args.calibrated_design,
                calibration_receipt_path=args.calibration_receipt,
                output_path=args.output,
            )
        elif args.command == "finalize-reviewed-manifest":
            result = finalize_reviewed_manifest(
                design_path=args.design,
                calibrated_design_path=args.calibrated_design,
                calibration_receipt_path=args.calibration_receipt,
                candidate_path=args.candidate,
                output_path=args.output,
                reviewer_id=args.reviewer_id,
                review_note=args.review_note,
            )
        elif args.command == "train-one-seed":
            result = train_one_seed(
                design_path=args.design,
                manifest_path=args.manifest,
                train_seed=args.seed,
                runs_dir=args.runs_dir,
            )
        else:
            result = evaluate_checkpoint(
                design_path=args.design,
                manifest_path=args.manifest,
                checkpoint_receipt_path=args.checkpoint_receipt,
                runs_dir=args.runs_dir,
            )
    except (ExperimentContractError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
