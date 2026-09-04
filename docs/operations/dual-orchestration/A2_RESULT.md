# A2: first live call stopped without a proposal

| status | current truth |
|---|---|
| progress | Accepted A1 code prepared an immutable 5,925-byte packet and launched one real read-only Sol/max call. The failed attempt has retained receipt and iteration records. |
| bottleneck | No final response arrived. The CLI logged a transport idle timeout; the parent stopped this attempt. No proposal or revision was admitted. |
| next step | Diagnose the connection/host interruption read-only, then define a fresh bounded follow-up if warranted. Do not silently retry this canary or launch its dependent revision. |

## Outcome

- A1 commit: `76a0fe6e86f5c4e5b52231d3b69df3d0c9b317d2`.
- Final independent A1 review: `ACCEPT`; parent reproduced 59 focused tests,
  Ruff/format/diff checks. No A1 code changed during the canary.
- Inputs: synthetic two-float velocity task, source fixture, constant-zero
  parent, and predeclared synthetic feedback `0.75`. No real B0 or robot.
- Calls launched: **1 of a maximum 2**. No manual retry or model substitution.
- At `23:20:57Z`, CLI stderr recorded a disconnected stream and idle timeout
  waiting for the websocket; the CLI began an internal transport retry.
- Parent inspected that error at the next execution checkpoint and sent TERM
  to this exact runner at about `23:25Z`. Terminal receipt:
  `INTERRUPTED_TERM`, exit `143`, no escalation, finished `23:25:32Z`.
- The roughly 31-minute wall interval is not model compute time. Host suspension,
  network interruption, quota, and provider cause have not been established.
- A1 ingestion returned exit `2`, `accepted: false`, `proposal: null` because
  `final.txt` was missing. Rejection receipt and iteration were retained.
- Requested configuration is `gpt-5.6-sol`, reasoning `max`, `read-only`.
  Observed thread ID: `01a06ea1-0e34-7110-a591-ca70dd74a6b6`.
- No generated source was executed; no training, simulator, reward quality,
  reference-composition improvement, or robot-performance result exists here.

## Retained local evidence

Paths are relative to the Astra checkout; run artifacts remain ignored.

| artifact | location or SHA-256 |
|---|---|
| Frozen inputs and hash inventory | `.orchestration/a2-canary/20260904T2250/baseline.json` |
| Initial packet | `b07d7a87a916e39aad7c3a6172bfa9fd8b1bf5944626c30e7665e5fec85f0f71` |
| Actual run | `.orchestration/sol-runs/20260904T225403Z-f5fc865b-3d10-496b-8a5b-27e70aad70a4` |
| Transport log and terminal | `stderr.log`, `result.json` in that run |
| Rejection receipt | `initial-records/receipt-18e0f0a82d5d5b72627d21cadafed711bd844ae5eed8d37aba7cc7b469fac222.json` under frozen inputs |
| Rejected iteration | `initial-records/iteration-25f9fa64f9152e29c6fd3fdb5852cc6a23d255ff60e2fd285c036bae46eee6a4.json` under frozen inputs |

This is a recorded integration failure, not evidence that a generated reward
was poor. No generated reward was returned to evaluate.
