# F3 static acceptance and reward-lane transfer

| status | current truth |
|---|---|
| progress | Independent review accepted repaired F3; parent reproduced 112 focused passes and verified all 20 pinned files. |
| bottleneck | No F3 model-candidate call, admission, protected evaluation, training or behavioral evidence. |
| next step | Transfer this exact slice to Fable under Samuel's lane swap; Astra does not dispatch the proposed reward call. |

## Decision

- Code: `ACCEPT_F3_STATIC_ONLY`.
- Separate protocol: `APPROVE_F3_ONE_CALL_PROTOCOL`, conditional on every
  preparation gate in the reviewed document.
- Samuel's subsequent lane assignment takes precedence over its dispatch
  instructions: execution remains stopped. Fable must review the transfer and
  explicitly reconcile checkout, owner and integration identities before use.
- Zero F3 candidate calls. No automatic retry, revision or model substitution.
- Earlier rejected source/review records remain historical and unchanged.

## Exact review

| item | receipt |
|---|---|
| review base | `7cd98c3d5befa49c4e52687652ff45ef70fc8a24` plus the five declared untracked F3 files |
| reviewer run | `20260906T022525Z-fb81129c-f170-4668-9fb2-63253d80a662` |
| requested settings | `gpt-5.6-sol`, `max`; not authenticated served-model evidence |
| terminal | `SUCCEEDED`, exit 0, `2026-09-06T02:38:18Z`; lease `RELEASED` |
| watchdog | `terminal_observed`; before the immutable `02:45:25Z` deadline |
| local report | `.orchestration/f3-closure-20260906/REVIEW.md` |
| report SHA-256 | `a13c2903e6efc99311e33b52cd72a623d0e21c725cb2aaf9697719519b901342` |
| snapshot SHA-256 | `f8ba2f77533ce9392784bcafd0ce95020cbbe26bd07ea8dfd9a31a813db75637` |

## Verification

- Independent reviewer: 68 F3 tests + 44 predecessor formula-loop tests passed;
  all four original findings closed; no new blocking defect reported.
- Parent after terminal: **112 passed in 0.88 s** using the same two test files.
- Parent import origin: this Astra checkout, not main.
- Changed Python paths: Ruff check and format check pass.
- Parent reverified every file in the 20-file closure snapshot before commit.

| accepted path | SHA-256 |
|---|---|
| `src/oracle_composition/reward_search/t2_model_contracts.py` | `4f1e7d36797f2d3d2e25050bbd980fa4c0c1f65949cacc931a73fc1dac208904` |
| `src/oracle_composition/reward_search/t2_model_protocol.py` | `e7390f16891f240435a1fe15d247bc58483a414e325da6db4fe2d268c74d1a94` |
| `tests/reward_search/test_t2_model_protocol.py` | `e59172944aa1e7500126236c357bbdb46f624555c165df707e59791216a40618` |
| `F3_ONE_CALL_PROTOCOL.md` | `193ddea8f3dbc363305157f8097daebdc5a186adcc605d50592bd853ffad89d8` |

## Claim ceiling

- Deterministic initial packet construction, bounded raw retention, strict
  local-envelope parsing and hypothesis/recipe format acceptance only.
- Tracking-only baseline: exact 1,773-byte blob from `87d2e39`, hash
  `eea2b6a9893e6e4ca5aea5d6787580f12062084e758cc9c27db2f1376bcb1c5f`.
  Its admission fields are declarations, not observed runtime certificates.
- Broader executable reward generation, feedback-driven revision and reward
  efficacy remain unproven. Existing Python execution gates remain unchanged.
