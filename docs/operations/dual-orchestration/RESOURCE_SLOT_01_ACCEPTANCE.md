# Resource slot 01: helper-only acceptance

| status | current truth |
|---|---|
| progress | Independent closure accepted the exact helper/CLI slice; source committed at `b7b67b7`. |
| bottleneck | Supervisor pre-spawn validation and terminal cleanup are not integrated; no heavy run is authorized. |
| next step | Fable integrates and reviews the spawn boundary, then returns a clean source handoff for the separate T1 smoke proposal. |

## Exact acceptance

- Source: `b7b67b7925650db2e999953e6b85e81f865c7d34`, five files only.
- Checkout: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`;
  branch `astra/reward-loop`. Clean worktree verified after source commit.
- Closure run: `.orchestration/sol-runs/20260906T152142Z-cb9379db-0519-4e6e-b86c-135382523d72`.
- Requested Sol/max; terminal `SUCCEEDED`, exit 0 at **15:29:43 UTC**.
- Verdict: **`ACCEPT_RESOURCE_SLOT_HELPER_ONLY`**. SLOT-01 and SLOT-02 closed;
  no remaining findings in this slice.
- Final SHA-256: `828a01796aa9976b2a7eaba32577b523afc02342ee23af4de86287757815ffc8`.
- Watcher: `terminal_observed`; no deadline signal was needed.
- Reviewer and collecting parent verified all five hashes in
  `RESOURCE_SLOT_01_REVIEW_RESULT.md`. Source commit preserves those bytes.
- Parent recheck: **37 passed in 2.69 s**; Ruff, format and diff checks pass.
  Reviewer performed static checks only, not these tests.
- `uv.lock` unchanged: `81b92d15dd2da62f27cd770322db78008d5387b530dc71e053f0d56b327f0b40`.

## Integration boundary

- Use the existing shared Git-common-directory mailbox root in both lanes.
  Import reviewed ancestry at a clean checkpoint; preserve Fable's own changes.
- Keep runtime-input/clean-commit preflight and native reservation validation.
  Immediately before spawn, call `validate_slot_for_reservation` with that
  validated reservation and the same expected wall time as the held token.
- Retain the returned exact owner/token identity until execution ends. Release
  only that pair after terminal cleanup, including spawn failure; expiry never
  transfers ownership or permits a force release.
- Test refusal before spawn for absent, malformed, expired or mismatched slots,
  and cleanup on failure/success, using isolated fixtures. Do not weaken the
  existing scientific/runtime admission to make token integration pass.
- Fable owns `phase_b/supervision.py`, integrated review and main promotion.
  Return its exact reviewed commit before Astra imports changing runtime code.

## Remaining research work

- No real token was acquired. No simulator, training, protected evaluation,
  full suite, dataset regeneration or push occurred in this slice.
- T1's admitted data and earlier passing preflight are retained. Refresh the
  preflight identity after the final reviewed integration, then propose the
  exact separate **1,200-second**, non-promotable training-only smoke.
- This acceptance is not resource agreement, a cohort permit, or evidence that
  the robot uses composed references. Causal-use and behavioral gates remain.
- Fable's committed `2806729` record reports one ingested F3 hypothesis,
  alpha 1.0 / beta 0.0, not yet admitted or measured. Astra read its committed
  ledger, not the private runtime bundle, and did not reproduce that call.
