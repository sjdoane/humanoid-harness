# ADR 0008: alignment pivot, retirement of the residual tracker family, and the failed development screen

| | |
|---|---|
| progress | Substrate verified: imported actors, plain-runtime corpus with certificates, E3 fork corpus, reward sandbox. The 2026-09-05 audit against the transcript found no harness cycle has run. |
| bottleneck | The residual tracker family cannot reproduce the library (`0/56,000` own-state steps within authority `0.08`; 90 percent coverage needs about `0.78`), and the imported expert failed its frozen development screen (`19/20`). |
| next step | Composition-loop cycle 0 and 1 with controller switching (packet `E003-C0`); fine-tuning runtime warm-started from the expert as the local policy-training block; E5 and hardening off the critical path. |

**Status:** accepted by the verified Fable strategy session on 2026-09-05 under its strategy authority; Samuel may reverse any item
**Date:** 2026-09-05
**Supersedes:** ADR 0005 slice B (residual reference-conditioned PPO) and its fallback row for a failed development gate; ADR 0004's E4/E5 placement on the critical path. ADR 0004's frozen MDP boundary and ADR 0005's import, corpus, E1, E2, and E3 rows remain in force.

## Evidence

| item | measurement | source |
|---|---|---|
| Residual authority gap | For the medium and simple actors on their own admitted-clip states, the expert's action differs by `L-inf` median `0.744` and `0.768`; steps within `0.08`: `0/28,000` each; within `0.40`: `148/28,000` and `5/28,000`; authority covering 90 percent of steps `0.784` and `0.794`; the gap is uniform over the whole clip | read-only analysis `sol-analysis-20260905-authgap` on the v2 corpus; expert-on-expert bitwise zero on `612,000` components |
| Development screen | healthy `19/20`, upright `19/20` (seed `96018` fell at step `860`), median velocity `5.00 m/s`, displacement `20/20`; gate `20/20` frozen in packet `TASK-20260904-03B` | run v2, commit `a09c804` |
| Base stability over `1,000` steps | expert fell in `2/36` corpus blocks and `1/20` screen resets (`3/56`) | v2 E3 certificate and screen receipt |
| Harness execution | zero oracle or reward cycles executed on any MDP in either lane as of 2026-09-05 | handoff, Astra status board |

## Decision

| item | decision |
|---|---|
| Residual family | Retired before training. A `0.08` residual, or any authority below the full range, cannot express the library's non-expert gaits; training it would test the wrong thing. |
| Local policy-training block | A fine-tuning runtime warm-started from the imported expert's actor weights, with a zero-initialized reference-window input (the predeclared expanded-TQC family), full authority, PPO or the collaborator's trainer when supplied. E1 identity holds by construction at initialization. Its gate is a utility smoke: follows each library gait and the stated transitions to a stated tolerance in a bounded number of episodes, receipted. No admission chain. |
| Composition executor for cycle 0 | Controller switching over the three actors, standing in for a frozen tracker following composed references; labeled `exploratory_oracle_cycle`; the oracle program contract (active behavior, guards, dwell, transition, recovery) is the same object that later drives the fine-tuning runtime's composed reference. |
| E4 and E5 | E4 becomes the utility smoke above. E5 causal-use ablations are optional science on the stand-in and leave the critical path; the sealed blocks were exposed on 2026-09-05 and must be redrawn before any confirmatory use. |
| Failed development screen | Recorded as failed; the gate is not weakened or rerun. The expert remains the development base because the harness's job is to compose and fine-tune imperfect controllers, and every downstream claim is gated by its own predeclared metrics. No local training fallback is requested; the fallback budget stays zero. |
| Hardening findings | The robustness and scientific review P1s that do not change a scientific conclusion move to `docs/operations/HARDENING_BACKLOG.md`; the ones that do (admission map and boolean rename, screen evidence label, process IDs out of scientific digests, E3 manifest wording) are folded into the next slice touching those files. |

## Consequences

- Packet `E003-C0` is the next writer; packet B is withdrawn and replaced by a fine-tuning-runtime packet after cycle 0 reports.
- The reward lane's first executed candidate targets the fine-tuning runtime with the target-speed task; the two lanes converge on one cycle runner and one report format.
- The preprint's claim of record becomes the loop demonstration, not tracker admission.
