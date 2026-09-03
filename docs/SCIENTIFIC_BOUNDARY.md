# Scientific boundary

## Program truth

| Item | Current status | What advances it |
|---|---|---|
| Agentic design loop | Target; not implemented | One bounded evidence-to-candidate-to-evaluation cycle |
| Research knowledge graph | Deterministic build/query over 47 migrated paper extractions | Add approved current-review evidence, gap records, and decision links |
| Oracle contracts | Implemented and tested | Integration with a measured tracker and matched oracle study |
| Generated oracle | Not implemented | Bounded schema generation plus protected evaluation |
| Generated task reward | Not implemented | Executable reward contract, sandbox, and frozen-oracle study |
| Evidence UI | Read-only status and research-query views implemented | Add canonical trajectory and comparison views |
| Gymnasium adapter | Real `Humanoid-v5` reset/step interface check | Immutable runtime fingerprint in each run |
| Frozen reference tracker | No qualifying checkpoint | Trained checkpoint plus causal reference-use ablations |
| Protected evaluator | Mechanical metrics implemented | Task, transition, contact, recovery, and naturalness calibration |
| Canonical trajectory | Bounded trace schema and explicit missing-signal audit implemented | Wire Gymnasium evaluation and episode inspector to exact trace bytes |
| Humanoid improvement | No evidence | Held-out matched results across predetermined seeds |
| Cross-MDP generality | No evidence | Same harness contract on a second adapter |

## Program versus study

```text
program knobs:       oracle O_k       task reward r_k
                         |                   |
single-factor study: change O / freeze r   change r / freeze O
combined study:           declared O x r factorial
```

Within every study, freeze:

- robot, scene, observation/action spaces, dynamics, resets, and termination;
- tracker/controller, tracking reward, normalizers, and command ABI;
- trainer, hyperparameters, seeds, budget, and checkpoint rule;
- protected evaluator and task-outcome definitions; and
- admitted reference bytes and their embodiment, joints, units, frames, and
  cadence.

Changing one of these creates a separate experiment family.

## What stock Gymnasium provides

- A 17-actuator MuJoCo humanoid and stable environment API.
- `Humanoid-v5`, whose default task rewards forward locomotion.
- `HumanoidStandup-v5`, whose default task emphasizes rising height.

It does not provide a reference-conditioned tracker, motion library, phase
estimator, transition oracle, task-reward generator, or independent evaluator.

## Oracle-study admission

Label a run `oracle_training` only when its receipt proves:

1. Exact robot, ordered reference features, units, frame, cadence, horizon, and
   normalizers.
2. Controller-compatible reference bytes and simulator admission evidence.
3. Immutable tracker, tracking reward, task reward, trainer, and evaluator IDs.
4. Actor and critic receive the exact declared oracle observation.
5. Train and evaluation consume the same immutable execution manifest.
6. All oracle arms share seeds, budgets, reset distribution, reward, and
   checkpoint rule.
7. Exact, zero, shuffled, and time-shifted reference-use ablations pass.

Otherwise label the run `interface_check` or `exploratory`.

## Reward-study admission

Label a run `reward_training` only when its receipt proves:

1. Exact `r_0` and `r_k` source, validated inputs, units, bounds, and hashes.
2. Generated code passed static checks and executes only in the reward sandbox.
3. The oracle, references, tracker, training system, seeds, and budget are
   identical across reward arms.
4. Primary task metrics are computed outside generated reward code.
5. Reward-scale and term-removal controls rule out trivial magnitude effects.

No reward-study admission path is implemented yet.

## Meaning of “works well”

The sealed-test primary endpoint is end-to-end completion when the task has a
clear binary goal. Otherwise it is one predeclared continuous task-progress
score. The endpoint is fixed before results are inspected and cannot switch
afterward. Hard guardrails include:

- nominal tracking and transition success;
- falls, forbidden contacts, and time to failure;
- phase error, stalls, chattering, and loops;
- same-episode recovery and task resumption;
- contact impulse, jerk, action rate, torque, energy, and saturation;
- uncertainty across every predetermined seed; and
- unchanged protected evaluator results under receipt replay.

Training return is diagnostic only. It cannot establish success.
