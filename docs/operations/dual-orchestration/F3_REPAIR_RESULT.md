# F3 repair and exact-baseline refresh

| status | current truth |
|---|---|
| progress | One builder under the existing Sol/max-requested lease repaired F3-01 through F3-04, refreshed the exact baseline, and passed 112 focused tests plus three-file Ruff and compile checks. |
| bottleneck | This builder result is not independent closure. No model service, adapter admission, execution, protected evaluation, measurement, training, or improvement is established. |
| next step | Independently review the changed F3 schema, protocol, and regressions before authorizing any candidate-model call. |

## Scope and starting state

| item | exact value |
|---|---|
| checkout / branch | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra` / `astra/reward-loop` |
| starting and final HEAD | `2fbac7782269abb7ecb4dd5dd616c60c71cc1ec2` |
| initial dirty paths | Untracked `F3_RESULT.md`, `t2_model_contracts.py`, `t2_model_protocol.py`, and `test_t2_model_protocol.py` only |
| writer lease | `astra-f3-repair-20260906`; requested `gpt-5.6-sol`, `max`, builder; acquired 2026-09-06 01:47:18 UTC |
| final audit / elapsed | 2026-09-06 01:55:31 UTC / 8 minutes 13 seconds from immutable lease request creation |
| `uv.lock` SHA-256 | `81b92d15dd2da62f27cd770322db78008d5387b530dc71e053f0d56b327f0b40` |
| import root | `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra/src/oracle_composition/__init__.py` |

The pre-edit check matched all 18 entries in
`.orchestration/f3-review-snapshot-20260906.json`. The four files under
`.orchestration/f3-pre-repair-20260906/` matched the snapshot and were not
edited. The original `F3_RESULT.md` remains unchanged at SHA-256
`28e86d5ee8341bd1519c95ec8cd831c5130af82f6756046872f853f4cd00afae`.

## Exact repair

| path | pre-repair SHA-256 | repaired SHA-256 | change |
|---|---|---|---|
| `src/oracle_composition/reward_search/t2_model_contracts.py` | `9cd79ea795acce2175e2fc34cfbcb8376f212b2b92a690bed1addbe064918bca` | `4f1e7d36797f2d3d2e25050bbd980fa4c0c1f65949cacc931a73fc1dac208904` | Enforced all-or-none identities and renamed fixed receipt metadata to `expected_*`. |
| `src/oracle_composition/reward_search/t2_model_protocol.py` | `0eae5d3b3d91dbb3412393ceb432ca13cbe5fe3a0222bdd82173b4e572f074f8` | `e7390f16891f240435a1fe15d247bc58483a414e325da6db4fe2d268c74d1a94` | Used bounded elapsed time, moved exact-type ingress checks ahead of work, clarified evidence wording, renamed receipt fields, and refreshed the baseline. |
| `tests/reward_search/test_t2_model_protocol.py` | `66389f92153eb81f31d3fc143e78ded927a8370837945c8341bdb82e793c55e1` | `e59172944aa1e7500126236c357bbdb46f624555c165df707e59791216a40618` | Added real parser, public ingress, timestamp, receipt-semantics, and exact-baseline regressions. |

### Findings addressed

- **F3-01:** Replaced `started + timedelta(...)` with `finished - started` and
  inclusive duration checks. Tests cover years 1 and 9999, a year boundary,
  negative duration, exactly 1,200 seconds, 1,201 seconds, and malformed
  timestamps. Every bounded fixture is retained and produces an outcome
  receipt; no wall-clock freshness or authenticated-time claim was added.
- **F3-02:** A retained run binding now requires filename, SHA-256, and byte
  count; every non-retained binding requires all three individually absent.
  Accepted receipts require both proposal and recipe hashes; rejected receipts
  require both absent. Real parser tests cover every presence subset and retain
  fixed artifact ordering and rejection reasons.
- **F3-03:** All bytes and native-`Path` arguments are exact-checked at each
  public entry before record reconstruction, trusted-source/run reads, or
  publication. Tests cover both path-taking APIs, every argument position,
  wrong types, subclasses, a preconstructed record, nested record mutations,
  early callback traps, and real success-path readers/publishers.
- **F3-04:** Receipt fields are now `expected_model`,
  `expected_reasoning_effort`, `expected_runner_kind`, `expected_owner`,
  `expected_role`, and `expected_scope`; metadata semantics is
  `expected_configuration_not_served_model_attestation`. Mismatched-model,
  malformed-request, and accepted fixtures verify that expected configuration
  is never represented as observed service. Raw retained request bytes remain
  the evidence for request contents. Launcher `requested_*` fields are unchanged.

## Refreshed exact baseline

| identity | value |
|---|---|
| Git object | `87d2e39c47b6747b505bc2657d505aeda2265b5e` |
| path | `experiments/003_composition_speed_profile/phase_b/tracking_only_v1.json` |
| byte count / ending | 1,773 / `}` with no newline |
| SHA-256 | `eea2b6a9893e6e4ca5aea5d6787580f12062084e758cc9c27db2f1376bcb1c5f` |

The bytes were read from that Astra Git object. The old exact 1,144-byte
baseline (`0986d4fc907e94185e14f58c9ef6eabf8ec26a8f336c68ee450bbacd5d8224d4`)
is now rejected at the public boundary. Its archived fixtures and receipts were
not changed. The refreshed artifact preserves tracking-only `r_task=0.0`, no
parent alpha/beta recipe, target 3.0 m/s, cadence 0.015 s, and accepted F2
formula/bounds/input identities. Its admission-contract fields are declarations,
not an observed adapter certificate or successful runtime load.

## Verification actually run

| check | result |
|---|---|
| Focused pytest: updated F3 plus `test_formula_proposal_loop.py` | `112 passed in 1.07s` |
| Ruff lint on three changed Python files | passed |
| Ruff format check on three changed Python files | passed; three already formatted |
| `py_compile` on three changed Python files | passed, with cache redirected outside the checkout |
| Accepted predecessor snapshot | all 13 hashes in `.orchestration/f2-review-20260905T2237.json` unchanged |
| Archive integrity | all four pre-repair archive hashes still match the F3 review snapshot |

Red evidence remains the parent's archived reproduction of all seven review
probes (`7 passed in 0.10s`, where pass meant the historical defect was
observed). Those defect-asserting probes were not rerun against repaired
production code. Green focused pytest first reached 110 passes, then 112 after
adding explicit malformed-timestamp cases.

The first Ruff pass found one `RUF043` regex-literal issue and one format-check
delta in the new tests. Both were corrected with `apply_patch`; final lint,
format, compile, and focused pytest all passed. No production test failed.

## Remaining gates and claim ceiling

- Independent static closure remains required; a successful builder is not
  independent acceptance.
- The prompt and dossier still declare missing observed adapter certificate,
  candidate admission, execution manifest, protected evaluator results, and
  candidate measurements. Only alpha and beta remain authorable; no feedback
  or revision API was invented.
- No candidate-model call, network, simulator, training, install, full suite,
  commit, push, peer checkout activity, or historical-record edit occurred.
- This repair establishes static packet/retention/schema behavior only. It does
  not establish model origin, runtime admission, successful loading, execution,
  evaluation, reward improvement, or humanoid competence.
