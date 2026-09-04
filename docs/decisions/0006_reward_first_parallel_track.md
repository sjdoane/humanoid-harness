# ADR 0006: reward-first parallel track on stock Humanoid-v5

| | |
|---|---|
| progress | Samuel approved a Family-B track that needs no tracker: the authorable factor is an executable task reward on the stock `Humanoid-v5` MDP with a frozen stock trainer. |
| bottleneck | No executable reward contract, sandbox, predeclared task metric, or scored `r_0` baseline exists; the loop's design agent has no authorized API client. |
| next step | A read-only Sol design survey proposes the contract, task family, endpoint, trainer recipe, and reuse map from RL-Sculptor; a builder then lands the baseline slice when the writer slot is free. |

**Status:** accepted 2026-09-04 by the verified Fable session with Samuel's
answer `reward-first-track: approved`; design pending
**Date:** 2026-09-04
**Relation:** parallel to ADR 0005; it does not change the oracle track, the
tracker gates, or the charter's research questions. It relaxes the `LG-11`
ordering from one track to two.

## Decision

| item | decision |
|---|---|
| Family | B, reward. Frozen counterpart: a canonical null oracle and reference configuration (none), recorded in the manifest. The MDP is stock `Humanoid-v5` with its stock observation, action, dynamics, resets, and termination frozen in the manifest. |
| Authorable factor | Executable task reward `r_k`: an immutable Python program with declared inputs, units, bounds, and cadence, validated statically and in a sandbox before training. |
| Baseline `r_0` | The stock `Humanoid-v5` reward (forward, healthy, control cost, contact cost) re-expressed inside the same executable contract, so `r_0` and `r_k` share one interface and one validator. |
| Task family | Locomotion targets in the spirit of Lokesh's example (`LKS-A13`): reach and hold a predeclared forward speed, with survival, torque, energy, and contact guardrails. The exact endpoint is fixed by the locked protocol before any behavioral data. |
| Primary endpoint | One predeclared continuous task-progress score computed by the protected evaluator from simulator state, never from reward code. Training return is diagnostic only. |
| Trainer | One frozen stock recipe, seeds, budget, and final-checkpoint rule chosen by the design survey from the two local receipts: TQC at `619.66` steps/s and PPO at `1,623.85` steps/s. |
| Loop v0 | Evidence dossier from the protected evaluator, candidate `r_k` authored by a Sol worker acting as the design agent through a task packet, static and sandbox validation, matched training, protected evaluation, decision record. An API-driven design client is a later slice and needs its own paid-usage authorization. |
| Evidence label | The first cycles are `exploratory`. A confirmatory reward study with sealed tasks and predetermined seeds needs the charter's formal-study gate. |
| Reuse | The old RL-Sculptor reward-refinement loop, metric-trust pipeline, and sandbox are read-only source material. Only contracts that pass this repository's boundary are ported, with provenance. |

## Why

| option | science | cost | why not first |
|---|---|---|---|
| Wait for tracker admission before any reward work | none until S1 and S2 pass | zero | The reward track shares no frozen component with the tracker track and can test the whole harness loop now. |
| Reward track on the stock MDP (this ADR) | end-to-end test of the reward sub-loop (candidate generation, validation, matched training, protected evaluation, steering), not of the two-artifact harness | per cycle: the matched `r_0` arm, the candidate arm, and reward-scale and term-removal control arms, each about `27 min` TQC or `11 min` PPO per 1M steps per seed | chosen |
| Reward track on a tracker-based MDP | closest to the lab block | blocked by S1 | later, as Family B on the admitted tracker |

## Admission gates

Every claim follows `docs/SCIENTIFIC_BOUNDARY.md`, section "Reward-study
admission": exact `r_0` and `r_k` hashes; static checks; sandbox-only
execution; identical MDP, trainer, seeds, and budget across arms; primary
metrics computed outside reward code; reward-scale and term-removal controls.

## Consequences

- The writer slot stays with the oracle track's builder when both tracks have
  a slice ready; reward design work runs read-only in parallel.
- Compute per exploratory cycle is capped at `2 h` CPU unless the handoff
  records a larger budget.
- The evaluator gains locomotion task metrics that must be independent of any
  generated code and must stay fixed across the study.

## Risks retained

- Reward hacking and metric circularity; the design must keep the evaluator
  and the task endpoint outside the reward author's reach.
- Goodhart effects on the predeclared speed target; term-removal and
  reward-scale controls are mandatory before any improvement claim.
- Divergence between the stock MDP and the lab block; results here test the
  harness, not transfer.

## Sources

- Transcript anchor `LKS-A13` in `docs/strategy/SOURCE_MANIFEST.md`
- `docs/PROJECT_CHARTER.md`, Family B
- `docs/SCIENTIFIC_BOUNDARY.md`, reward-study admission
- Old RL-Sculptor workspace, read-only, path in `CLAUDE.local.md`
