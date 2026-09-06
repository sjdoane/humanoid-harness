# Close the three F1 lineage findings

- One read-only Sol/max leaf, no delegation. Maximum 20 minutes; final 600 words.
- Astra checkout only; verify the ten-file snapshot
  `.orchestration/f1-closure-20260905T2048.json` before and after review.
- Read AGENTS, dual README, `F1_REVIEW_FINDINGS.md`, `F1_REPAIR_RESULT.md`, and
  `F1_REPAIR_PARENT_CHECKPOINT.md`. The original independent final is at
  `.orchestration/sol-runs/20260905T193917Z-382d4a3e-5e85-4229-9d82-a7958035231b/final.txt`.
- Compare current two formula-lineage modules/test with their exact originals
  under `.orchestration/f1-pre-repair-20260905/`. Formula math, config, T2, A1/R1
  are unchanged; do not repeat the broad project or integration review.

## Close or identify a concrete remaining defect

1. F1-01: exact accepted/rejected raw response bytes are retained, bound to
   receipt identity/count, rehashed on replay and reparsed during revision.
2. F1-02: responder can copy the needed semantic request identity from actual
   prompt bytes; final prompt bytes have a separate recomputed identity and
   both identities survive ingestion/revision without a self-reference.
3. F1-03: public entry points operate on fully revalidated canonical models;
   forged kind/schema, partial storage, unsafe nested values and list mutation
   cannot bypass checks or invoke user callbacks.
4. Parent follow-up: retained-response replay uses nonblocking open before
   checking file type; FIFO regression failed safely before fix and passes now.

Return ACCEPT_F1_STATIC_ONLY or REJECT_F1 with at most three actionable findings
and a minimal negative. Preserve the static/synthetic-only claim ceiling,
unverified supplied-byte origin, absent compositor and refused Python runtime.
Do not block this slice on future T2 adaptation, model calls or robot training.

No edits, source writes, model calls, reward workers, simulator, host sandbox
probes, network, installation or full suite. Read-only pytest cannot allocate
tmp_path on this host: do not repeat those known setup failures or seek a
permission bypass. Review the existing temporary-file tests and the parent's
exact snapshot-bound reproduction; optional in-memory/non-temp tests must be
bounded to 60 seconds and use Astra's `.venv`. Report what you actually ran.
