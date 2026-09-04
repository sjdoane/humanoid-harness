"""Command-line bridge for A1 reward proposal packets and receipts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .contracts import (
    IterationRecord,
    ModelCallReceipt,
    PacketRecord,
    ProtectedEvidenceDossier,
    RewardProposal,
    SourceBundle,
)
from .loop import (
    ingest_sol_run,
    load_model,
    prepare_initial_packet,
    prepare_revision_packet,
    publish_packet,
)
from .publication import finite_pretty_json


def _paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--packet-record", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m oracle_composition.reward_search")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="publish an initial deterministic packet")
    prepare.add_argument("--dossier", type=Path, required=True)
    prepare.add_argument("--source-bundle", type=Path, required=True)
    _paths(prepare)

    revision = commands.add_parser(
        "prepare-revision", help="publish a feedback-linked revision packet"
    )
    revision.add_argument("--prior-dossier", type=Path, required=True)
    revision.add_argument("--prior-proposal", type=Path, required=True)
    revision.add_argument("--prior-receipt", type=Path, required=True)
    revision.add_argument("--prior-packet-record", type=Path, required=True)
    revision.add_argument("--prior-iteration", type=Path, required=True)
    revision.add_argument("--feedback-dossier", type=Path, required=True)
    revision.add_argument("--source-bundle", type=Path, required=True)
    _paths(revision)

    ingest = commands.add_parser("ingest-sol", help="validate and retain one detached Sol result")
    ingest.add_argument("--packet-record", type=Path, required=True)
    ingest.add_argument("--run-directory", type=Path, required=True)
    ingest.add_argument("--records-directory", type=Path, required=True)
    return parser


def _write(value: object) -> None:
    sys.stdout.buffer.write(finite_pretty_json(value))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            record = prepare_initial_packet(
                load_model(args.dossier, ProtectedEvidenceDossier),
                load_model(args.source_bundle, SourceBundle),
            )
            published = publish_packet(record, args.packet, args.packet_record)
            _write(
                {
                    "packet": str(published.packet.path),
                    "packet_sha256": published.packet.sha256,
                    "packet_bytes": published.packet.byte_count,
                    "packet_record": str(published.record.path),
                }
            )
            return 0
        if args.command == "prepare-revision":
            record = prepare_revision_packet(
                load_model(args.prior_dossier, ProtectedEvidenceDossier),
                load_model(args.prior_proposal, RewardProposal),
                load_model(args.prior_iteration, IterationRecord),
                load_model(args.feedback_dossier, ProtectedEvidenceDossier),
                load_model(args.source_bundle, SourceBundle),
                prior_receipt=load_model(args.prior_receipt, ModelCallReceipt),
                prior_packet=load_model(args.prior_packet_record, PacketRecord),
            )
            published = publish_packet(record, args.packet, args.packet_record)
            _write(
                {
                    "packet": str(published.packet.path),
                    "packet_sha256": published.packet.sha256,
                    "packet_bytes": published.packet.byte_count,
                    "packet_record": str(published.record.path),
                }
            )
            return 0
        record = load_model(args.packet_record, PacketRecord)
        outcome = ingest_sol_run(record, args.run_directory, args.records_directory)
        _write(
            {
                "accepted": outcome.accepted,
                "proposal": None if outcome.proposal is None else str(outcome.proposal.path),
                "receipt": str(outcome.receipt.path),
                "iteration": str(outcome.iteration.path),
                "rejection_reason": outcome.rejection_reason,
            }
        )
        return 0 if outcome.accepted else 2
    except (OSError, ValueError) as exc:
        print(f"reward-search error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
