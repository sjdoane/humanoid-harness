# F3 targeted closure and one-call protocol review

- One independent Sol/max leaf; no delegation. Deadline: immutable request
  creation + 1,200 seconds. No source edits, model-candidate call or execution.
- Checkout `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`,
  branch `astra/reward-loop`; record exact HEAD and import root.
- Test-output-only writer: sole writable scope is the initially absent
  `.orchestration/f3-closure-20260906/`. Stop if it already exists; no deletion.
- No source/test/doc/snapshot edits, commit, push, network, install, simulator,
  full suite, peer checkout reads/writes, hook or sandbox change.

## Read and verify

1. Root `AGENTS.md`, dual-orchestration `README.md`.
2. `F3_REVIEW_FINDINGS.md`, `TASK-F3-repair-refresh.md`, `F3_REPAIR_RESULT.md`,
   `F3_REPAIR_PARENT_CHECKPOINT.md`, and `F3_ONE_CALL_PROTOCOL.md` here.
3. Repaired F3 modules/test and their differences from exact originals under
   `.orchestration/f3-pre-repair-20260906/`. Read surrounding code for each fix.
4. Verify every `.orchestration/f3-closure-snapshot-20260906.json` hash before
   and after review. A changed reviewed source is a stop, not a repair request.

## Execute only focused verification

- Use Astra `.venv`, `PYTHONDONTWRITEBYTECODE=1`,
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, `-s -p no:cacheprovider --assert=plain`.
  `TMPDIR` and every fresh non-existing `--basetemp` must be under the sole
  authorized output directory. Never reuse a populated basetemp.
- Independently run only `tests/reward_search/test_t2_model_protocol.py` and
  `tests/reward_search/test_formula_proposal_loop.py`; parent observed 112 passes.
- Add focused independent probes in your output directory if useful. Use actual
  readers/parsers/publishers; callback traps may only prove early refusal.
- If permissions deny a step, report it and continue with permitted static
  checks. No weakened settings or alternate out-of-scope writes.

## Targeted closure questions

- F3-01: extreme valid dates no longer overflow; negative/overlong durations
  produce retained rejection receipts; exact deadline remains allowed.
- F3-02: every partial identity subset fails parser validation in rejected
  bindings/receipts; ordinary accepted/rejected ingestion still works.
- F3-03: every public argument position is exact-type checked before all
  source/run reads, callbacks and publication.
- F3-04: fixed receipt metadata is explicitly `expected_*`; mismatched or
  malformed requests never fabricate observed facts or model attestation.
- Refreshed baseline exactly matches the 1,773-byte Git blob and hash above;
  old bytes fail. Its declared admission contract is not observed admission.
- Raw retention, immutable context, A1/F1 isolation, fixed formula and missing
  evidence remain intact. No newly reachable blocking defect from the repair.

## Conditional next-protocol review

- Separately review `F3_ONE_CALL_PROTOCOL.md` for a single bounded, read-only,
  subscription-authenticated hypothesis call with honest receipt ingestion.
- Check source/baseline/prompt pinning, dispatch-intent duplicate prevention,
  exact watchdog deadline, terminal failure retention, no retry/substitution,
  no runtime/training or shared-interface authority, and truthful claim ceiling.
- Approving this document does not launch a call. Parent must record your
  verdict and satisfy every preparation precondition first.

## Return

- Write only your output-directory `REVIEW.md`; final answer repeats verdicts.
- F3 verdict: `ACCEPT_F3_STATIC_ONLY` or `REJECT_F3` with each original finding
  closed/open and actual tests/reproductions, exact hashes and remaining limits.
- Protocol verdict: `APPROVE_F3_ONE_CALL_PROTOCOL`,
  `REJECT_F3_ONE_CALL_PROTOCOL`, or `NOT_REVIEWED`. Report it separately; missing
  approval leaves the call unauthorized even if code is accepted.
- Focus on actionable defects, not speculative expansion or behavioral claims.
- No lease alteration; the launcher releases your own lease on completion.
