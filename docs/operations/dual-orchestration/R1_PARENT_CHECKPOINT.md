# R1 parent checks before independent review

Historical checkpoint. The review timed out without a verdict; the later
completion and current checks are in `R1_COMPLETION_CHECKPOINT.md`.

| status | current truth |
|---|---|
| progress | Parent reproduced 108 passing checks, with 36 explicit skips, using Astra's own environment. R1 source is frozen for independent review. |
| bottleneck | Safe-test coverage, a stale consumer, and static receipt version typing remain incomplete. No dynamic validation is claimed. |
| next step | Collect the exact independent review, then make one bounded completion patch before R1 acceptance. |

- Baseline: `67722402c58374cf7f22347f7055abb979c9436f` plus the uncommitted
  builder diff; 12 file hashes in `.orchestration/b0-repair/r1/review-snapshot.json`.
- Import path resolves inside `humanoid-harness-astra`.
- Parent ran the five R1 modules plus `tests/reward_search` and the mailbox
  module: `108 passed, 36 skipped in 2.39s`. Changed-path diff check passed.
- No candidate, child worker, OS canary, or simulator was executed.

## Findings for completion

- `tests/rewards/test_sandbox.py` has a module-wide skip. Several checks are
  pure: profile construction, byte framing, source drift, recorded JSON, and
  receipt validation. Restore safe checks individually; defer only genuinely
  dynamic operations. Do not read the skip count as containment coverage.
- `tests/experiments/test_reward_target_speed_manifest.py` still imports the
  removed executable validator and calls `.task_term`. Convert its fixture
  check to static acceptance/no-dynamic-claim assertions. Existing Gym tests
  and historical runtime fingerprint replay remain separately deferred.
- The three skipped scale negatives now contain only `assert source and
  expected_failure`. They are a case inventory, not working regression tests.
  They must fail explicitly if accidentally enabled until R2 supplies the
  actual worker-backed assertions; do not let unskipping create vacuous passes.
- Pure reproduction: `StaticAcceptanceReceiptV1.from_dict` accepts both
  `schema_version=True` and `schema_version=1.0`, then emits integer `1`.
  Require an exact integer version and test boolean/float/string rejection.
  This does not enable execution, but violates strict receipt typing.

These are parent observations, not the independent review's verdict. Keep the
builder and reviewer evidence separate; fix only after the reviewer stops.
