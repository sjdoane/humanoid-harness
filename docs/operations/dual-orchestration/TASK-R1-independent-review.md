# R1 independent boundary review

- One read-only Sol/max leaf, no delegation; at most ten minutes.
- Checkout: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`.
- Review the uncommitted R1 diff against `67722402c58374cf7f22347f7055abb979c9436f`.
  Do not edit, commit, run another model, or contact the peer.
- Exact paths first:
  `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/AGENTS.md`,
  `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/docs/operations/dual-orchestration/README.md`,
  `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/docs/operations/dual-orchestration/ASTRA_HANDOFF.md`,
  `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/docs/operations/dual-orchestration/TASK-R1-parent-execution-boundary.md`,
  `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/docs/operations/dual-orchestration/R1_RESULT.md`.
- Verify `.orchestration/b0-repair/r1/review-snapshot.json` before and after
  inspection. If any named bytes change, stop and report the mismatch.

## Questions

1. Do all parent-facing validation, capture, drift, and scale/axiom paths avoid
   candidate installation/invocation? Trace the child-only installer and its
   call order after environment/limit setup. Do not execute a candidate.
2. Does the static object/receipt honestly refuse dynamic acceptance, and do
   public runtime routes fail closed? Identify callable/legacy receipt paths,
   fragile trust assumptions, and input validation regressions within R1.
3. Do depth-eight/nine statement, expression, and mixed cases enforce the
   original ceiling? Are numeric JSON types validated before coercion?
4. Are the tests substantive and safe? The builder blanket-skipped 33 sandbox
   tests and deferred three dynamic scale cases. Identify which inherited
   checks are actually static/data-only and should remain running. Do not
   lift the blanket skip or run the whole sandbox module dynamically.
5. The builder reports an out-of-scope stale validator import/call in
   `tests/experiments/test_reward_target_speed_manifest.py`. Locate all direct
   consumers of the removed API and give the smallest safe completion scope.
   Do not recommend an executable compatibility alias.
6. Are changes confined to R1, with reward/evaluator constants, historical
   receipts, A1 code, and external dependencies preserved? Does the result
   overstate any closure that actually depends on R2/R3?

## Checks and output

- Inspect tests before running any. Only safe AST/data tests and the 59 A1/
  mailbox tests are permitted. No candidate execution, reward child process,
  OS canary, simulator, model load, full suite, or network work.
- Final: progress/bottleneck/next step; ACCEPT, ACCEPT-WITH-REPAIRS, or REJECT
  **for the R1 static-only scope**; exact file/line findings; minimal fixes and
  negative tests; checks performed versus deferred. Do not rerun a broad B0
  scientific review or request that R1 prove the deliberately deferred R2/R3.
- Retain known defects as findings where material. Do not count an intentional
  dynamic refusal as proof that executable reward generation is complete.
