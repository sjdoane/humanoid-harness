# System architecture

- Target: automate the supplied VIBE/SONIC post-training process through
  reference guidance and task reward; not a new controller or training method.
- Current implementation is the GMT/G1 proxy. Its model calls, candidate
  selection and launch decisions are externally coordinated by agents.
- Current scope and task sequence: [post-training plan](strategy/astra/POST_TRAINING_PLAN_20260908.md).

## One loop

```text
                         optional human correction
                                   |
                                   v
task + R + r0 ---> evidence-grounded decision agent <---- research graph
                              ^               |
                              |               | candidate O_k, r_k
                              |               v
                     diagnostic dossier   validation gates
                              ^               |
                              |               v
                    development evaluator <--- policy-training adapter
                              ^                         |
                              +----- trajectory --------+
```

## Artifact flow

| Artifact | Producer | Consumer | Immutable identity |
|---|---|---|---|
| Task specification | User/upstream collaborator | Harness | Canonical JSON + SHA-256 |
| Reference set `R` | Motion source/upstream collaborator | Oracle | Bytes + schema + provenance hash |
| Oracle `O_k` | Harness | Training adapter | Typed program + parent IDs + SHA-256 |
| Task reward `r_k` | Harness | Training adapter | Validated source + interface + SHA-256 |
| Execution manifest | Harness and reviewer | Trainer/evaluator | All exact inputs + runtime fingerprint |
| Policy `pi_k` | Trainer | Evaluator | Checkpoint + load receipt + SHA-256 |
| Trajectory trace | Adapter | Evaluator/UI | Ordered samples + signal schema + SHA-256 |
| Development evaluation `E_k` | Independent evaluator | Agent/UI | Metrics + failures + uncertainty + SHA-256 |
| Decision record | Agent/human | Next iteration | Evidence IDs + patch + prediction + falsifier |

## Decision sequence

```text
observe evidence
      |
      v
name failure + missing signals
      |
      v
retrieve source-backed mechanisms
      |
      v
propose the smallest O or r change
      |
      v
validate -> matched run -> development evaluation
      |
      v
accept, reject, or ask one targeted question
```

The agent may not silently edit a frozen component, evaluator, or claim gate.
Final held-out evaluation is a separate readout, not feedback for another
revision. Its outcomes do not enter the development loop.

## Adapter interface

| Input | Output |
|---|---|
| `O_k`, `r_k`, fixed MDP manifest, seed, budget | Policy checkpoint |
| Evaluation seed and checkpoint | Canonical trajectory trace |
| Stop request | Explicit final state and partial receipt |

GMT/G1 through Gymnasium/PPO implements guarded reference windows, task-reward
recipes, residual training, synchronized traces, independent metrics and
feedback-linked candidate admission. Native Humanoid-v5 is an earlier family.
VIBE's actual reference/trainer ABI remains unconfirmed. The given trainer may
update its designated parameters; freeze that permitted set across comparisons.
Private integration does not require publishing the lab's source code.

## User surfaces

| Surface | Purpose | Authority |
|---|---|---|
| CLI | Build, validate, run, query, inspect | Calls application services |
| Evidence UI | Observe status, traces, comparisons, provenance | Read-only derived view |
| PRAXIST | Schedule bounded candidate experiments | Outer-loop orchestration only |
| Research graph | Retrieve mechanisms, failures, metrics, and provenance | Evidence aid; never a result generator |

## Implementation sequence

| Slice | State |
|---|---|
| Active research index and query command | Implemented |
| Clean read-only status UI | Implemented |
| `TrajectoryTrace/v1` contract | Implemented |
| Adapter trace recording and episode inspector | GMT implemented; not VIBE integration |
| Bounded oracle-candidate record and manual approval gate | GMT data-only admission implemented |
| Trained tracker with causal reference-use evidence | GMT supplied actor implemented; VIBE missing |
| First matched oracle comparison | GMT development results; no full-task pass |
| Separate reward contract and reward experiment family | GMT implemented; general reward execution gated |
| Full fixed/manual/harness post-training comparison | Missing; task admission and complete tuning costs required |
