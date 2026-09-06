# F3: initial T2 proposal protocol

| status | current truth |
|---|---|
| progress | The bounded planner resolved a true-baseline, initial-only protocol; parent verified the baseline bytes and reusable bounded I/O. |
| bottleneck | No implementation or Fable-admitted candidate reward artifact exists. The exact F2 promotion reply remains pending. |
| next step | Build the two T2 protocol modules and focused tests, then review before any candidate call. |

- Planner: `20260905T231728Z-4f668197-b00f-4a3f-965f-e3acb817bcd2`.
- Completed 23:29:33 UTC; read-only Sol/max, no tests/writes/candidate calls;
  no escalation. Watcher observed completion.
- Keep A1 Python-source and F1 old-target/supplied-byte schemas unchanged.
- Reuse their bounded regular-file reads, immutable publication and detached
  request/result schema classes. Add distinct T2 packet/response/receipt types.
- Public record boundaries accept exact bytes, not mutable typed objects.
- Bind the actual tracking-only baseline, not a fabricated parent recipe.
- Initial evidence records no rollout measurements and explicitly missing
  admission/runtime/evaluator evidence. Initial format acceptance cannot
  authorize execution or become a claimed reward improvement.

## Verified baseline and one peer documentation mismatch

- Peer object: `a51ea9e68aa9e27a573e372f67e36d6fa4d494cd`.
- Blob: `experiments/003_composition_speed_profile/phase_b/tracking_only_v1.json`.
- Exactly 1,144 bytes, no newline; SHA-256
  `0986d4fc907e94185e14f58c9ef6eabf8ec26a8f336c68ee450bbacd5d8224d4`.
- Parent verified Git blob hash/size and its run-manifest binding. The
  `phase_b/DESIGN.md` table still names `aeb148...` at this same commit.
  Bind the actual blob and manifest, not the stale prose hash; Fable owns the fix.

## Parent corrections to the proposed plan

- Use the existing watcher deadline: immutable `request.json.created_at_utc`
  plus 1,200 seconds. Do not introduce a second deadline based on launch time.
- Every successfully bounded-read artifact is retained before parsing.
  Missing, empty, oversized or nonregular inputs need explicit read-failure
  states; do not claim unavailable/oversized bytes were retained or hashed.
- Add byte-type and nesting/error normalization around reused A1 parsing;
  its public parser alone does not reject callback-bearing byte subclasses.
- F3 is initially missing-evidence-only. Later protected-feedback revision is
  a separate slice; no placeholder evaluator digest or dormant runtime switch.
- No candidate call, retry or revision is authorized by this planning result.
