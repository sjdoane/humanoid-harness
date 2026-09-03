# Oracle-composition review process

**State:** guiding question approved; candidate registry and acquisition batch
01 registered; formal search and screening have not started.

The authoritative human-facing protocol is
[docs/LITERATURE_REVIEW_PROTOCOL_DRAFT.md](../../../../docs/LITERATURE_REVIEW_PROTOCOL_DRAFT.md).
This directory supplies the traceability tables required to execute that
protocol. Candidate and pilot-search records may be populated before screening;
the evidence and decision tables remain empty.

Do not add inclusion decisions, source-triage classes, a coding taxonomy, or
synthesis claims until their corresponding human gate is approved. Candidate
discovery and metadata-only acquisition priorities may be recorded, but
candidate metadata is not citable evidence.

Current candidate registry: 80 deduplicated primary-paper records across five
pilot discovery routes plus one targeted web-follow-up route. One withdrawn
record is retained on administrative hold; every evaluative field remains
`not_assessed`.

Pending human-review surfaces:

- `SEARCH_STRATEGY_DRAFT.md`: sources, boundaries, and stopping rule.
- `FORMAL_SEARCH_QUERIES_DRAFT.md`: exact terms, request contracts, receipts,
  and citation chaining.
- `TRIAGE_PROPOSAL.md`: metadata-only follow-up acquisition order.
- `APPROVAL_LOG.md`: gate states; only the guiding question is approved.

## Reverification

```bash
uv run python scripts/build_candidate_registry.py \
  research/literature_review/01_process/001_oracle_composition/matrices/candidate_papers.csv
uv run python scripts/register_literature_sources.py --verify
```

The first command rebuilds metadata from exact version-pinned arXiv pages and
fails if the pinned page's identity, availability, or withdrawal state changes.
It waits at least three seconds between requests. The second command is local
and network-free: it reconciles the registry, batch, manifest, inventory,
evidence boundary, regular-file constraints, PDF header, size, and SHA-256 for
all sources in acquisition batch 01.
