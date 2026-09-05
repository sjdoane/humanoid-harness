# A2R: one awake-host replacement canary

| status | protocol |
|---|---|
| progress | A2 failed without a proposal during repeated host sleep. Auth and current reported quota checks pass. |
| bottleneck | End-to-end live proposal/revision plumbing is still unproved. |
| next step | Reuse exact synthetic inputs with an idle-sleep assertion and a 600-second deadline for each of at most two calls. |

- This is a new attempt, not a continuation or relabeling of failed A2.
- Retain A2 and its rejection unchanged. Same A1 code and initial packet:
  `b07d7a87a916e39aad7c3a6172bfa9fd8b1bf5944626c30e7665e5fec85f0f71`, 5,925 bytes.
- New output directory: `.orchestration/a2-canary/awake-20260905/`.
- Exact original inputs stay under `.orchestration/a2-canary/20260904T2250/`.
- Maximum two new calls: initial and, only if admitted, feedback revision.
  Including failed A2, at most three live candidate calls in this sequence.
- Preserve `gpt-5.6-sol`, `max`, `read-only`, synthetic labels, no tools or
  nested delegation in the candidate task. No provider or safeguard changes.
- Run the existing detached launcher; use a local one-shot watcher for this
  attempt, not a new shared scheduler. Its deadline is 600 wall-clock seconds;
  on expiry, validate the exact runner identity and send TERM so the existing
  runner writes its own terminal receipt. Never target another lane/process.
- The watcher holds an idle-sleep assertion for at most 660 seconds and drops
  it on completion. It cannot ensure progress while the lid is closed.
- Keep initial/admitted/rejected/revision/watch records separately. Ingest
  with A1 unchanged. Do not edit model responses to make them pass.
- If call 1 is rejected or times out, do not launch call 2. If call 2 fails,
  retain the partial result. No third attempt while conditions are unchanged.
- Feedback is the exact predeclared `0.75` synthetic fixture; retain real
  evidence as missing. It is not a measurement of the generated candidate.
- Generated source remains nonexecuted text. No B0, simulator, training,
  evaluator, or robot-quality claim. Same acceptance as the original A2.
- Record outcomes in `A2R_RESULT.md`, with all attempts and limitations visible.
