# Humanoid Harness

An oracle-composition research harness for the 17-actuator Gymnasium MuJoCo
Humanoid.

## Research question

Under a frozen reference-conditioned tracker, tracking objective, task reward,
trainer, budget, and evaluator, can a state- and phase-conditioned reference
oracle improve multi-skill transitions and perturbation recovery over fixed or
open-loop reference playback?

```text
admitted reference motions
          |
          v
 state + task + horizon ---> reference oracle ---> mode + phase + H x D window
                                                       |
                                                       v
                         frozen reference-conditioned tracker
                                                       |
                                                       v
                     Gymnasium Humanoid-v5 / MuJoCo
                                                       |
                                                       v
                    independent transition evaluator
                                                       |
                                                       +----> evidence + failures
                                                                  |
                                                                  +----> next oracle candidate
```

| Surface | Oracle work may change | Frozen inside a comparison |
|---|---|---|
| Reference use | mode, phase, window, transition, recovery/rejoin | reference bytes and command ABI |
| Policy system | nothing | tracker, tracking reward, task reward, trainer |
| Environment | nothing | model, observations, actions, dynamics, reset, termination |
| Evaluation | nothing | seeds, budget, checkpoint rule, metrics, sealed tasks |

## Current status

| | Status |
|---|---|
| **research target** | Phase-aware, closed-loop reference-oracle composition |
| **implemented capability** | Immutable oracle/reference contracts, a real `Humanoid-v5` interface, and a 45D root-and-joint static-tracking candidate |
| **measured evidence** | Interface smoke and reward-regression evidence only; no trained tracker or new Humanoid oracle result |

The prior project’s seven Gym studies are retained only as negative evidence:
they appended reference-like commands to a newly initialized PPO policy, but no
frozen tracker or fixed tracking reward consumed those values. They therefore
tested interfaces, not oracle effectiveness.

## Repository map

```text
docs/                 charter, boundaries, protocols, decisions, failure audits
research/             versioned literature corpus and evidence records
src/oracle_composition/
  contracts/          immutable artifact and ABI validation
  runtime/            deterministic oracle execution
  envs/               Gymnasium adapters
  tracking/           controller-native state, reference, and reward candidate
  evaluation/         structural evidence admission
  experiments/        frozen runner, protected evaluator, and study aggregation
tests/                 contract, unit, integration, and negative tests
experiments/           declared protocols; run outputs remain untracked
praxist/               PRAXIST integration notes, never vendored PRAXIST code
archive/               provenance manifests for imported prior work
```

## Development setup

Requires Python 3.12 or 3.13 and `uv`. The complete test suite imports the
training stack, so install the exact locked development environment:

```bash
uv sync --locked --extra gym --extra train --extra dev
uv run pytest
uv run ruff check .
```

PRAXIST is an optional orchestration layer. It does not define the scientific
question, evaluator, or acceptance criteria. See [`praxist/README.md`](praxist/README.md).

Built wheels include the exact `uv.lock` bytes used in runtime fingerprints.
The installed entry point and runtime inspector are tested outside the source
checkout, including equality of the complete source and wheel runtime
fingerprints. The embedded lock binds intended dependency bytes; it does not by
itself prove that an installed environment was created from that lock. Study
designs, criteria, manifests, and run outputs remain explicit external
artifacts; they are not silently selected from package defaults. No package has
been published, and a successful build is not a behavioral result.

## Capability boundary

- Stock `Humanoid-v5` is a forward-locomotion task, not a reference-tracking
  benchmark.
- This repository must first establish and freeze a tracker that demonstrably
  consumes controller-native reference windows.
- Oracle variants will be admitted only after exact/zero/shuffle/time-shift
  reference ablations pass.
- "Works well" will mean held-out end-to-end task completion plus explicit
  transition, recovery, safety, and stability guardrails—not training reward.

This repository is currently private, and its project license is not yet
selected. Imported research records retain source-level attribution and
provenance.
