# R1 completion review: repairs required

| status | current truth |
|---|---|
| progress | Independent Sol/max review finished and reproduced 37 no-temp checks; parent verified all 15 reviewed file hashes unchanged. |
| bottleneck | REJECT for static-only R1: caller-controlled text/metadata methods can execute, and the accepted-source constructor can admit fabricated AST evidence. |
| next step | One bounded repair of exact-type ingress and source-envelope validation, followed by targeted independent re-review. |

- Run: `20260905T060349Z-d2435712-4e9a-4af1-a94b-fd183f40c96e`.
- Terminal: `SUCCEEDED`, 2026-09-05 06:18:34 UTC; no lease or escalation.
  Successful review execution is not acceptance of the implementation.
- Full verdict: `.orchestration/sol-runs/20260905T060349Z-d2435712-4e9a-4af1-a94b-fd183f40c96e/final.txt`.
- Watcher observed the terminal receipt; it did not terminate this review.

| finding | accepted repair scope |
|---|---|
| High: text subclass `.encode()` executes before AST parsing | Require exact built-in text or bytes; a hostile text sentinel must remain untouched. |
| High: proxy-wrapped custom mapping and envelope metadata equality dispatch callbacks | Accept exact plain JSON input containers before traversal; create immutable containers internally. Remove redundant envelope metadata input or validate it before any dispatch. |
| Medium: accepted-source envelope checks hash/length but not the AST-derived receipt | Derive or revalidate the complete receipt from exact source bytes; reject invalid source with a matching fabricated hash and valid source with altered AST counts. |

The scale/axiom routes currently revalidate the envelope and refuse dynamic
admission, so the forged-envelope finding is not an observed execution bypass
through those routes. It is a false static-acceptance claim that must be fixed.

## Checks and deferred work

- Reviewer: all frozen source/test paths inspected; 37 no-temp tests passed;
  donor fixtures and unchanged protected files verified.
- Parent: all 15 snapshot hashes matched at collection on this heartbeat.
- Parent's prior 139 passes/16 deferred remain separate from the reviewer's
  reproduction. Keep both receipts and the earlier timed-out review.
- No candidate execution, live OS canary, simulator, or behavior claim.
- R2 containment and R3 provenance are still separate gates. Do not expand
  this repair into those phases or rerun broad unrelated reviews.

## Repair checkpoint, 2026-09-05 07:11 UTC heartbeat

- Repair run `20260905T064008Z-229b5f64-f12b-46cf-a0f2-35cd1714a36b`
  finished 06:49:16 UTC, `SUCCEEDED`, lease released; watcher observed completion.
- Parent inspected the new constructors and sentinel tests, then reproduced
  `143 passed, 16 skipped in 2.42s`; same focused command as the completion
  checkpoint. Ruff/format on 15 Python files, compile, and diff checks passed.
- Compared with the historical 15-file snapshot, only the three authorized
  Python files changed: static validation, parent-boundary tests, and scale
  tests. The new builder result is `R1_INGRESS_REPAIR_RESULT.md`.
- A1, evaluator, stock compositor, envs, dependency lock, study config, and
  historical receipts have no diff from HEAD.
- These are parent checks, not independent closure. The next reviewer must
  explicitly close all three findings before R1 can be accepted.
