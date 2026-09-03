# System architecture

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
                      protected evaluator <--- policy-training adapter
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
| Evaluation `E_k` | Protected evaluator | Agent/UI | Metrics + failures + uncertainty + SHA-256 |
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
validate -> matched run -> protected evaluation
      |
      v
accept, reject, or ask one targeted question
```

The agent may not silently edit a frozen component, evaluator, or claim gate.

## Adapter interface

| Input | Output |
|---|---|
| `O_k`, `r_k`, fixed MDP manifest, seed, budget | Policy checkpoint |
| Evaluation seed and checkpoint | Canonical trajectory trace |
| Stop request | Explicit final state and partial receipt |

Gymnasium/PPO is the first partial development adapter. It currently implements
the real reset/step substrate, static-reference wrapper, trainer path, and
mechanical evaluator. Canonical trace emission, executable task-reward input,
and full interface conformance remain pending. A later lab adapter should meet
the same complete contract without exposing or copying private system details.

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
| Adapter trace recording and episode inspector | Next |
| Bounded oracle-candidate record and manual approval gate | Pending |
| Trained tracker with causal reference-use evidence | Pending |
| First matched oracle comparison | Pending |
| Separate reward contract and reward experiment family | Pending |
