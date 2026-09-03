# PRAXIST integration boundary

PRAXIST is installed outside this repository as a pinned optional orchestration
tool. Its code is not vendored here.

## Ownership

| PRAXIST may own | This repository must own |
|---|---|
| Peer lifecycle and bounded variant directories | Scientific question and claims |
| Generation planning and evidence bookkeeping | Runnable baseline and fixed trainer |
| Process scheduling and replay | Protected evaluator and sealed tasks |
| Synthesis of task-owned results | Validity gates and final merge decision |

## Current installation receipt

| Item | Value |
|---|---|
| Version | `praxist==0.5.0` |
| Source pin | `sapientinc/PRAXIST@437292c7e1f414d87a4b742a23ae4ae3f66ef744` |
| Runtime | Isolated `uv` tool, Python 3.12 |
| Profile | Codex-native, saved ChatGPT login |
| Product-usage collection | Denied |
| User agreement | Accepted by Samuel on 2026-09-02 |
| Readiness | Codex-native doctor passes; task launch intentionally gated |

The PyPI `0.5.0` wheel blanked Codex Desktop's internal originator marker and
caused an invalid app-server handshake. The isolated tool is pinned to the
upstream one-commit fix while retaining the `0.5.0` SDK and dependency pins.
The unmodified doctor now confirms saved-login authentication and the selected
`gpt-5.6-luna` model. No environment-variable workaround is required.

## Two permitted uses

| Campaign | Earliest valid use | Claim ceiling |
|---|---|---|
| Harness engineering | A real baseline and test-owned evaluator canary both execute without PRAXIST | Candidate implementation or diagnostic only |
| Oracle research | The frozen tracker passes numeric-reference-use gates and the reference pool has same-simulator admission | The separately locked behavioral protocol only |

This split lets PRAXIST help build adapters, receipts, and oracle-evaluation
plumbing without treating its search process as scientific evidence. An
engineering campaign may optimize only an isolated candidate surface; the
repository's protected tests and review decide whether any code lands. It may
not tune a behavioral acceptance threshold or inspect sealed evaluation data.

## Common launch gates

1. The campaign-specific baseline and external evaluator run independently.
2. Resource limits derive from measured evaluator wall time and memory.
3. PRAXIST doctor, task resolution, evaluator canary, and lane routing pass.
4. The first run is CPU-only, foreground, one peer, one generation, and uses
   an absolute run directory.
5. At least one evaluator job is admitted; zero-job runs fail.
6. Candidate-editable files, frozen files, output ownership, and evidence
   maturity are explicit in the task manifest and audit rules.

An oracle-research campaign has additional hard gates: causal numeric-reference
use, controller-native time-varying reference admission, a frozen tracker and
trainer, a protected evaluator with pre-data thresholds, paired seeds, and
sealed tasks. Harness-engineering evidence cannot satisfy those gates.

Public PRAXIST-generated outputs must carry the attribution required by the
PRAXIST Fair Source license. Institutional license interpretation remains a
human/legal decision.
