# Resource slot 01: independent bounded source review

## Scope

- Read-only Sol/max leaf; no nested agents. Work only in
  `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`, branch
  `astra/reward-loop`. Read AGENTS.md and the dual-orchestration README.
- Review the exact five uncommitted implementation files and their SHA-256
  table in `RESOURCE_SLOT_01_PARENT_RESULT.md`, relative to committed base
  `386b2ed`. Parent checkpoint/packet may be docs-only descendants.
- Verify all five hashes before and after review. Stop on drift. These are
  Astra's frozen bytes, not permission to inspect Fable's changing worktree.
- Read `TASK-RESOURCE-SLOT-01.md` for the accepted implementation contract,
  and `HEAVY_JOB_SLOT.md` for the promised boundary. Parent reproduced 28
  focused tests and static checks; do not present those as your execution.
- Read-only source access to `src/oracle_composition/phase_b/supervision.py`
  for existing `validate_reservation` and the planned pre-spawn integration.
  Do not assess unrelated tracker, reward, evaluation or cohort repairs.
- No source edits, commits, temp files, imports, tests, probes, simulator,
  network, API, real token acquisition, lease mutation or process control.
  Use bounded source reads, not recursive `.orchestration` scans or raw logs.

## Questions that determine acceptance

1. Does exclusive publication plus the persistent guard provide cooperative
   single-slot exclusion? Can an old owner/token release a newer same-owner
   slot? Does failure preserve an existing token? Expiry must not free it.
2. Does reserve verify native v2 acceptance and acknowledgment with exact
   canonical bytes, strict relevant types and pinned file digests? The helper
   is not responsible for validating actual runtime inputs or clean checkout;
   that remains the existing supervisor's preflight before token validation.
3. Does the read-only pre-spawn API compare the complete binding against an
   already validated reservation, not only owner/presence? State any caller
   obligations required to retain ownership until terminal cleanup.
4. Do bounded reads and path/error handling reject the promised malformed,
   symlink, oversized, duplicate-key and nonfinite inputs? Check invalid guard
   behavior and CLI JSON errors. Parent identified root normalization as a
   possible gap; independently determine whether a concrete repair is needed.
5. Are the CLI adapters thin, own-checkout loaded, Python 3.9 compatible and
   consistent with existing mailbox behavior? Are real contention and stale
   release tests exercising the intended mechanism, not only fabricated status?

Keep the review proportional: cooperative same-user exclusion is not an OS
sandbox or authentication system. Do not require protection from arbitrary
rogue same-user code. Identify missing required regressions, but do not demand
a broad test matrix for unrelated potential threats. No helper approval grants
supervisor integration, a heavy-job reservation or training authority.

## Return

- Target 10 minutes; stop analysis at 15 minutes and deliver a final verdict.
  Exact runner hard deadline is 20 minutes. Do not spend the final minutes
  repeating source reads or auditing your own logs.
- Start with progress/bottleneck/next-step rows.
- Verdict `ACCEPT_RESOURCE_SLOT_HELPER_ONLY` or `REPAIRS_RESOURCE_SLOT_01`.
- List only actionable findings: severity, exact file/function/line, concrete
  failure path, smallest repair and regression. Distinguish blocking contract
  failures from optional maintainability suggestions.
- State hashes reverified, checks actually run (static only), and remaining
  Fable-owned integration / separate compute-reservation boundary.
