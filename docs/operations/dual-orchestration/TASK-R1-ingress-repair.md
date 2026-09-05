# R1: close the three independent static-boundary findings

- One Sol/max write builder; no nested agents, model calls, or peer messages.
- Worktree: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`.
- At most 20 minutes. Preserve partial work and return remaining items.
- Baseline HEAD: `67722402c58374cf7f22347f7055abb979c9436f` plus the reviewed,
  uncommitted R1 diff. Do not reset, overwrite unrelated files, or commit.

## Read first

Resolve all paths under the absolute worktree above:

1. `AGENTS.md`, `docs/operations/dual-orchestration/README.md`.
2. `docs/operations/dual-orchestration/R1_REVIEW_FINDINGS.md`.
3. `.orchestration/sol-runs/20260905T060349Z-d2435712-4e9a-4af1-a94b-fd183f40c96e/final.txt`.
4. `docs/operations/dual-orchestration/R1_COMPLETION_CHECKPOINT.md` and
   `.orchestration/b0-repair/r1review/review-snapshot.json`.

Verify the 15-file pre-repair snapshot before editing; report drift and stop
if it does not match. The snapshot becomes historical once your edit begins.

## Required repair

1. `_source_bytes` must reject text subclasses before calling any method.
   Test a hostile `str.encode` sentinel and preserve valid built-in str/bytes.
2. Close metadata callbacks in every public receipt/source constructor:
   reject arbitrary containers and externally supplied mapping proxies before
   traversal, equality, or serialization. Immutable proxies may be created
   internally only after plain JSON validation. A mapping proxy can wrap a
   hostile mapping; exact proxy type is not evidence of safe traversal.
   Remove the redundant source-envelope metadata argument if that simplifies
   the contract; preserve the public metadata view and deep immutability.
3. Accepted-source construction must derive or revalidate the complete
   AST-derived receipt from the exact source bytes. Hash/length alone cannot
   prove static acceptance. Test invalid source with matching fabricated
   hash/length and valid source with an incorrect `ast_nodes` value.

Prefer a small, self-validating data representation, not a second trust
registry or an untestable private-token convention. Update affected tests for
intentional constructor changes; retain serialized receipt v1 fields and its
static-only meaning. Exercise valid direct/factory/JSON round trips, stale
receipt rejection, nested metadata, and all hostile-mapping sentinels.

## Allowed edits

- `src/oracle_composition/rewards/static_validation.py`.
- `tests/rewards/test_parent_execution_boundary.py`, `test_static_validation.py`,
  and `test_scale_calibration.py` only for these findings or changed consumers.
- New result: `docs/operations/dual-orchestration/R1_INGRESS_REPAIR_RESULT.md`.

No worker, sandbox, scale implementation, evaluator, env, trainer, A1,
dependency, study config, historical receipt, shared strategy, hook, or other
checkout changes. Report necessary out-of-scope changes instead of making them.
Do not edit coordination handoff, active pointer, snapshot, or this packet.

## Verification

- Own `.venv`; verify `oracle_composition.__file__` resolves in this checkout.
- Read tests before running them. No candidate compile/exec/invocation,
  reward subprocess, live OS/network canary, simulator, model, training, or
  full suite. Repository-owned sentinel methods are test fixtures only.
- Safe focused command:

```text
.venv/bin/python -m pytest -q \
  tests/rewards/test_contract.py tests/rewards/test_static_validation.py \
  tests/rewards/test_scale_calibration.py tests/rewards/test_parent_execution_boundary.py \
  tests/rewards/test_sandbox.py tests/reward_search tests/integration/test_research_mailbox.py
```

- Prior parent: 139 passed/16 deferred. Keep the 13 host/subprocess sandbox
  and three dynamic scale cases deferred; do not enable dynamic admission.
- Ruff check/format, compile changed Python, `git diff --check`.
- No commit/push. Parent runs an independent targeted re-review after you stop.
- Final and result file: progress/bottleneck/next step, exact changes, checks,
  unresolved findings. If blocked or near deadline, stop with a partial result
  instead of improvising a broader scope. No claim that R1 or B0 is accepted.
