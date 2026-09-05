# R1: remove parent execution of candidate rewards

## Assignment

- One Sol/max builder; no nested delegation. Own only the files below.
- Worktree: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`.
- Work for at most 20 minutes; retain partial work and report remaining items
  if the boundary cannot be completed. Do not start another task.
- This is a repair of transferred B0, not a new reward study or simulator.

Read these exact files first:

- `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/AGENTS.md`
- `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/docs/operations/dual-orchestration/README.md`
- `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/docs/operations/dual-orchestration/ASTRA_HANDOFF.md`
- `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/docs/operations/dual-orchestration/A3_RESULT.md`
- `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/docs/operations/dual-orchestration/B0_DONOR_SNAPSHOT.json`

## Required changes

1. Make parent validation and source capture/drift checks strictly data/AST-only.
   No candidate installation, invocation, determinism execution, or executable
   callback can remain in the parent-facing validated-source object. AST-only
   parsing is allowed. A parse pass must not claim dynamic validation passed.
2. Keep any compilation/installation/evaluation confined to the child worker's
   private implementation after the existing environment and resource-limit
   setup. Parent imports of trusted worker definitions must have no candidate
   side effects. Preserve source snapshot checks; no unsandboxed fallback.
3. Remove arbitrary duck-typed callables from axiom/scale evaluation. Prepare
   a narrow source-bound sandbox-worker route; while runtime admission remains
   unreviewed, dynamic calls must refuse clearly rather than execute locally.
   Do not manufacture determinism, axiom, scale, or runtime receipts. Separate
   static acceptance from dynamic acceptance in names and serialized evidence.
4. Enforce the existing conditional-depth ceiling for statement `if`, ternary
   `IfExp`, and mixed nesting. Add exact pass-at-eight/reject-at-nine tests.
5. Validate JSON array element types and structure before NumPy conversion.
   Reject booleans/strings and wrong-dtype direct arrays instead of repairing
   them. Preserve the established field shapes, units, and numeric semantics.

Use a small implementation and clear types. Do not add a general sandbox or
another orchestration layer. Resolve internal consumers in the allowed scope;
report any necessary out-of-scope change instead of editing it.

## Allowed edits

- `src/oracle_composition/rewards/{__init__,contract,static_validation,scale_calibration,sandbox,_sandbox_worker}.py`
- `src/oracle_composition/experiments/reward_target_speed_manifest.py` only for
  its static-validation consumer; do not change study constants or fingerprints.
- `tests/rewards/{test_contract,test_static_validation,test_scale_calibration,test_sandbox}.py`
- New focused tests under `tests/rewards/` if clearer than expanding old files.
- `docs/operations/dual-orchestration/R1_RESULT.md`.

Do not edit the evaluator, stock reward arithmetic, A1 reward_search, envs,
tracker/controller, dependency files, root exports/CLI, strategy, shared
protocol/ADR/config, donor receipts, mailbox, or Fable's checkout.

## Verification and stop conditions

- Use only this checkout's `.venv`; verify the imported package path first.
- No candidate source execution, even to reproduce the old bug. Do not run
  the old dynamic tests before inspecting/converting them to safe static
  checks or clearly deferred host tests. No subprocess reward worker, OS
  canary, model load, simulator, training, or full-suite run in this slice.
- Static tests must prove capture and validation never invoke the child
  installer or a supplied callable. Test an arbitrary callable with a sentinel
  that must remain untouched, and retain actionable refusal messages.
- Cover valid/invalid source, metadata, depth boundaries, type coercion, and
  stale receipt rejection. Never remove a negative test to obtain green status.
- Run the safe focused tests you wrote, then the existing 59 reward-search/
  mailbox tests, Ruff on changed files, compile checks, and `git diff --check`.
  A1 tests generate strings only; do not bind their proposals to B0.
- Record exact commands, passes/failures/skips, and every deferred dynamic
  proof. A clean exit or parse-only test is not closure of all B0 findings.
- Do not commit, push, regenerate historical receipts, or change leases except
  the launcher-managed writer renewal. Parent integrates after review.
- Final: progress, bottleneck, next step; changed paths; focused receipts;
  which findings are addressed versus remain open. Keep it brief.
