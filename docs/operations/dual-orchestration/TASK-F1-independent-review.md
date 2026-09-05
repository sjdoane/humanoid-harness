# Review F1 data-only formula lineage and isolated T2 inputs

- One read-only Sol/max reviewer; no delegation, edits, commits, model calls,
  peer messages, live reward workers, OS probes, simulator, training or installs.
- Astra checkout only. Maximum 20 minutes; final at most 700 words.
- Existing source HEAD `402ff4f` plus nine uncommitted F1/T2 files. Verify every
  file hash in `.orchestration/f1-review-20260905T1937.json` before and after.
- Read AGENTS, dual README, `TASK-F1-formula-core.md`, `F1_RESULT.md`,
  `F1_PARENT_CHECKPOINT.md`, and the six core/test files plus config identified
  by the snapshot. Read the added isolated T2 module/test as a separate unit.
- Existing R1 and A1 code remain unchanged. M1 merge already independently
  accepted; do not repeat that review or audit Fable's dirty checkout.

## Required verdicts

1. `ACCEPT_F1_STATIC_ONLY` or `REJECT_F1`: fixed trusted math/strict input and
   recipe types; meaningful JSON/type/forgery negatives; new source-free schema;
   exact retained input/parent/feedback lineage and replay; bounded handling;
   honest supplied-byte origin; no reopened Python or total-reward admission.
2. `ACCEPT_T2_INPUT_ONLY` or `REJECT_T2`: fixed 3.0 COM input and cadence metadata,
   safe types/serialization, malformed/subclass/forged-slot refusal; old B0
   unchanged. It is not a simulator measurement or formula adapter.

Check real public entry paths, not only schema constructors. Parent concerns
to investigate, not assumed findings: rendered prompt requires its own packet
hash without plainly supplying it; raw response bytes may not be retained for
replay; model-frozen lists can still mutate and must revalidate coherently;
partial prior-lineage fields and synthetic/source provenance may disagree.
Classify findings by the actual static/synthetic-only claim ceiling. Do not
pretend this is an actual-model or training acceptance review.

You may run only the three new pure test files, each command bounded to 60
seconds, plus small in-memory regression probes. No files edited, no broad
suite. Use Astra's `.venv`; do not work around missing simulator dependencies.
Record exact commands/counts, up to three blocking findings with source lines
and a minimal negative, and the smallest repair or next integration boundary.
Do not consume the full budget on cosmetic issues or a second architecture.
