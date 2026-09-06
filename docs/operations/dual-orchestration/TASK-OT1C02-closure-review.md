# OT1C-02: independent closure of the settling-score repair

- Checkout: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`; branch `astra/reward-loop`.
- Read AGENTS.md and the dual-orchestration README, then
  `docs/operations/dual-orchestration/OT1_NARROW_RESULTS.md`.
- Exact source/commit bindings: `.orchestration/ot1-s2-20260906/snapshot.json`.
  Verify those hashes and review only the diff from `7f2298f` to its pinned
  `review_commit`. Parent holds source frozen during this review.
- Original finding: OT1C question 2, section in
  `.orchestration/sol-runs/20260906T050231Z-b95a7a98-2b7e-48db-a8c0-7f4f7078448b/final.txt`.
- Read-only Sol/max leaf. No nested agents, tests, Python imports, source/file
  edits, simulator, network, installs, training, model candidates or commits.
- Target final within 6 minutes; stop reading and report by 10 minutes.
  A 20-minute exact-run watchdog is attached. Return partial if needed.

## Only these checks

1. Both evaluator `_score_task_success` and report `_derived_task_success`
   now require present settling error at or below the calibrated band.
   Equality passes; one floating-point step above fails; `None` fails.
2. Missing calibration remains non-scoring; hold behavior, fixed safety,
   tracking/resynchronization scales and latency gates are unchanged.
3. Regression tests demonstrate failure before the fix and success after it.
   Read retained `red.txt` and `green.txt`; they are parent-executed results,
   not your reproduced tests. Check fixture expectations, not just counts.
4. The serialized-row test crosses real schema serialization and the public
   endpoint; cached true success does not control derived counts. Parent did
   not run full report-file reload. Follow the existing reload→endpoint→derived
   scorer call chain by source inspection and decide whether another narrowly
   bounded test is required for this repair, without reopening the full review.

## Final, maximum 600 words

- Three rows: progress, bottleneck, next step.
- Verdict `ACCEPT_OT1C02_SCORING_ONLY` or `REPAIR_REQUIRED`, with exact
  source/test locators and one concrete negative for each new blocking finding.
- State whether the original scoring omission is closed, which checks are
  inspection versus executed parent tests, and any narrow test gap.
- Do not accept calibration provenance, evaluator resources/manifest/cleanup,
  five-seed aggregation, other OT1 findings, or scientific/robot claims.
- Missing real calibration lineage and other remaining gates still block
  protected evaluation. This verdict grants no smoke/cohort authority.
