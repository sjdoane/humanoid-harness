#!/usr/bin/env python3
"""Regenerate or verify the committed DeepMimic source-format audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from oracle_composition.sources.deepmimic import audit_source_tree, render_audit_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT / "research/source_motions/deepmimic_humanoid3d"
DEFAULT_AUDIT_PATH = DEFAULT_SOURCE_ROOT / "source_format_audit.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Perform a bounded, data-only format audit of manifest-bound DeepMimic "
            "humanoid3d source bytes."
        )
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="fail unless the deterministic audit matches source_format_audit.json",
    )
    mode.add_argument(
        "--write",
        action="store_true",
        help="replace source_format_audit.json with the deterministic audit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_root = args.source_root.resolve()
    rendered = render_audit_json(audit_source_tree(source_root))
    audit_path = source_root / DEFAULT_AUDIT_PATH.name
    if args.check:
        if not audit_path.is_file() or audit_path.read_bytes() != rendered.encode("utf-8"):
            print(f"stale or missing audit: {audit_path}", file=sys.stderr)
            return 1
        print(f"verified {audit_path}")
        return 0
    if args.write:
        audit_path.write_text(rendered, encoding="utf-8", newline="")
        print(f"wrote {audit_path}")
        return 0
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
