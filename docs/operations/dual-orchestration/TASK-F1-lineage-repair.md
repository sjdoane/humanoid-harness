# F1: repair only the three independent lineage findings

- One Sol/max writer, no delegation or model calls; maximum 20 minutes.
- Astra checkout only. Existing F1 files are uncommitted; preserve their work.
- Read AGENTS, dual README, `F1_REVIEW_FINDINGS.md`, `F1_RESULT.md`, and the
  review's exact final listed there. Verify current hashes against
  `.orchestration/f1-review-20260905T1937.json` for the seven F1 files.
- Source snapshots before repair are under `.orchestration/f1-pre-repair-20260905/`.
  Do not edit snapshots, old receipts or the historical builder result.

## Only these edits

1. `src/oracle_composition/reward_search/formula_contracts.py`
2. `src/oracle_composition/reward_search/formula_loop.py`
3. `tests/reward_search/test_formula_proposal_loop.py`
4. New `docs/operations/dual-orchestration/F1_REPAIR_RESULT.md`

No numerical formula/config, old A1/R1, accepted T2 source/test, shared runner,
compositor, CLI, dependencies, frozen experiment, other docs, or Git changes.
Use `apply_patch`; do not commit. If outside-scope changes are required, stop
and explain instead of widening the edit set.

## Required fixes

- F1-01: publish the exact bounded supplied response bytes without overwrite,
  including malformed/rejected responses within the declared size bound.
  Receipt records bind its retained artifact, SHA-256 and byte count. Replay
  must recompute these values from retained bytes, not trust a hash string.
  Tests cover accepted/rejected byte-for-byte retention, mutation/refusal and
  collision behavior. Do not execute or publish externally any response.
- F1-02: make the required request identity visible to a responder without
  asking it to solve a self-referential hash. Define canonical request-payload
  identity before rendering and expose it explicitly; bind the final rendered
  prompt SHA-256 separately. A retained envelope is acceptable if simpler.
  Use unambiguous field names/documentation. Keep both semantic request identity
  and exact prompt-byte identity reverified through ingestion and revision.
  Test a responder that copies only data actually present in the prompt;
  out-of-band access to the record is not a valid test of this repair.
- F1-03: strictly revalidate full canonical copies of records/bindings at all
  public consumption boundaries, then operate on the validated copies. Wrong
  kind/schema, partial lineage, forged nested entries and list mutation must
  fail without bypassing schema checks. Preserve safe type checks before any
  user-provided callbacks. Do not only add two special-case string checks.
- Keep the claims static/synthetic-only. Do not create model-call evidence,
  permit generated Python, add T2 adaptation or admit a total-reward runtime.

## Bounded checks and stop

- Run the updated formula-loop tests plus formula core/T2 tests, and existing
  reward-search tests if needed, using this checkout's `.venv` and writable
  test temp directories. Each command at most 60 seconds; no full suite.
- Run Ruff/format/compile on changed Python. Preserve every old negative.
- Record new regressions, actual counts and skips, exact diff paths, own import
  origin, lock hash and remaining limitations in F1_REPAIR_RESULT.md.
- No host probe, reward subprocess, simulator, training, network, install,
  paid API, model call, external write, push, or nested worker.
- End after the repair/result. Independent closure is the parent's next step.
