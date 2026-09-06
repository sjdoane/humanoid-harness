# F3 parent checkpoint

| status | current truth |
|---|---|
| progress | Builder finished; parent reproduced 79 focused tests and verified all 13 accepted predecessor hashes. Fable accepted exact F2 promotion and merged F1/F2. |
| bottleneck | F3 is unreviewed and binds a superseded baseline. No candidate call, adapter admission, training, or behavioral result is authorized by this checkpoint. |
| next step | Independent source/test verification, then a bounded repair and baseline refresh if required; a live call needs its own reviewed protocol. |

## Collected evidence

- Checkpoint: 2026-09-06, 01:05 UTC heartbeat.
- Checkout: `/Users/samueldoane/Documents/ChatGPT/humanoid-harness-astra`.
- Branch: `astra/reward-loop`; source base `f7f3244a910110ddb5a13ffc9ea583892c06d412`.
- Builder: `20260905T235645Z-ff8a9aaa-8a22-4b07-a3a5-4f6ebaf44bad`.
- Terminal: `SUCCEEDED`, exit 0, 00:13:01 UTC; lease `RELEASED`;
  watcher `terminal_observed`, no escalation.
- Exactly four untracked F3 files; no existing tracked/staged changes.
- `F3_RESULT.md` is the unchanged builder receipt, not acceptance.
- Import resolves to this checkout's `src/oracle_composition/__init__.py`.

| parent check | result |
|---|---|
| F3 protocol + F1 formula-loop tests | 79 passed in 0.57s |
| Ruff lint / format, three F3 Python files | passed / three already formatted |
| Three-file `py_compile` | passed |
| Accepted F2 snapshot | all 13 file hashes unchanged |
| `uv.lock` | `81b92d15dd2da62f27cd770322db78008d5387b530dc71e053f0d56b327f0b40` |
| Existing tracked/staged diff | empty before this documentation checkpoint |

No simulator, full suite, training, model-candidate call, or paid API was run.

## Exact peer agreement and baseline change

- Fable acceptance `20260906T002718.313618Z-cbaf7f3f6326473ab7226e0c183ea53a`
  replies to exact promotion proposal
  `20260905T231526.889802Z-539e60b2549f48459fda183b385842df`.
- Agreed: unchanged 137-byte canonical bounds, accepted parser/evaluator IDs
  and bytes, and adapter-owned rejection outside inclusive `[-25,25]` m/s,
  with verified stock-COM origin and `0.015` s cadence. No clipping or V2.1.
- Parent inspected committed Git object only:
  `87d2e39c47b6747b505bc2657d505aeda2265b5e` merges accepted F1/F2 into main.
- Its `experiments/003_composition_speed_profile/phase_b/tracking_only_v1.json`
  is 1,773 bytes, SHA-256
  `eea2b6a9893e6e4ca5aea5d6787580f12062084e758cc9c27db2f1376bcb1c5f`.
- The new artifact adds an admission contract/source/schema identity and changes
  its reward-schema hash. These are declared bindings, not observed execution.
- F3 still pins the old `a51ea9e` / `0986d4fc...224d4` baseline by its original
  packet. Review it as that historical static scope; do not make a current
  proposal until an explicitly reviewed refresh binds the new exact artifact.
- Fable's broader repairs and pending smoke are peer-owned; no heavy-resource
  reservation or training authorization is created here.

## Independent review scope

- Use one Sol/max leaf with a test-output-only writer lease. Source is immutable.
- Explicit temporary-output authority lets this new verification task run real
  fixture tests; earlier read-only reviewers could not create pytest temp files.
  Do not change a sandbox, hook, or permission after a denial.
- New files and accepted predecessors are pinned in
  `.orchestration/f3-review-snapshot-20260906.json` and checked before/after.
- Review partial identity fields on rejected receipts, expected-versus-observed
  model metadata on malformed envelopes, missing-evidence wording, exact-type
  ingress, retention/publication, and schema isolation. These are review
  questions, not pre-decided findings.
- Return `ACCEPT_F3_STATIC_ONLY` or `REJECT_F3`, with exact repros, scope,
  unresolved findings and tests actually executed. No edits to reviewed code.

The affine alpha/beta family remains a narrow first path, not general reward
program generation or evidence of better robot behavior.
