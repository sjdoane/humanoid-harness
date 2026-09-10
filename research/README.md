# Research evidence

This directory keeps historical evidence, current review process, and later
synthesis separate.

Treat evidence/legacy_policy_harness_kg/ as read-only. Two project-context
labels were sanitized during migration; all paper cards and extractions remain
byte-identical to the old snapshot. Record corrections or new verification in
a current review run instead of rewriting the snapshot.

| Area | State | Authority |
| --- | --- | --- |
| evidence/legacy_policy_harness_kg/ | Paper evidence copied verbatim; two private-context labels sanitized | Historical research record; not the current project contract |
| evidence/engineering_methods/ | Registered process/orchestration sources | Engineering context; not direct oracle evidence |
| evidence/reference_audits/ | Optional typed G1 public-source and local numeric audit records | Discovery only; external motion is not admitted and numeric crop fitness is not dynamics evidence |
| literature_review/01_process/001_oracle_composition/ | New, empty review scaffold | Process only until the human review gates are approved |
| ../docs/meetings/PUBLIC_CONTEXT_BOUNDARY.md | New sanitized decision record | Current user direction; not literature evidence |
| ../archive/MIGRATION_MANIFEST.json | Generated provenance ledger | Byte-level receipt for this migration |

## Current scope

The new review is limited to reference-oracle composition for humanoid control:
mode selection, phase progression, transition guards, recovery, reference
compatibility, and the evidence needed to test them under a fixed experiment
family. Reward-only sources remain in the legacy snapshot for provenance; they
are not automatically in scope for the new review.

The attached PRAXIST paper is registered separately as an engineering-methods
source. It supports lineage and experiment-orchestration decisions; it does not
support a claim that any oracle mechanism improves humanoid behavior.

## Evidence states

- A legacy card or extraction is a migrated research artifact, not a newly
  verified paper claim.
- A search result is a candidate record, not citable evidence.
- A paper becomes a registered source only after its exact document is acquired
  and recorded in the review inventory.
- A final claim must trace through claim_id to evidence_id to source_id to the
  raw source.
- A paper mechanism never establishes that this repository implements or
  reproduces it.

## Human gates

The authoritative draft protocol is
[docs/LITERATURE_REVIEW_PROTOCOL_DRAFT.md](../docs/LITERATURE_REVIEW_PROTOCOL_DRAFT.md).
Its question, type, output, inclusion and exclusion criteria, source triage,
corpus adequacy, final included set, coding taxonomy, synthesis, and final
report require explicit human approval. The screening tables remain empty
until the first approval gate passes.

## Validation

From research/evidence/legacy_policy_harness_kg/, run:

    python3 scripts/audit_corpus.py
    python3 scripts/validate_extractions.py

The graph database, generated graph exports, generated summary, and generated
bibliography were intentionally not migrated. They can be regenerated in a
disposable workspace from the copied source artifacts when needed.

To include the two provenance-bearing reference audits in a fresh instance of
the existing SQLite index:

```bash
humanoid-harness research build \
  --extractions research/evidence/legacy_policy_harness_kg/extractions \
  --supplemental-records research/evidence/reference_audits/records \
  --database /tmp/humanoid-harness-research.db
```

The supplemental option is additive. Omitting it preserves the legacy-only
build. Supplemental results retain their explicit evidence status and are not
inserted into the paper table or connected to project knobs with `INFORMS`.
