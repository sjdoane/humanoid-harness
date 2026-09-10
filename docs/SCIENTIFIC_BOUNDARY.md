# Scientific boundary

## Program truth

| Item | Current status | What advances it |
|---|---|---|
| Agentic design loop | Real GMT cycles externally coordinated; integrated VIBE loop missing | Supplied trainer interface plus bounded model-to-run integration |
| Research knowledge graph | Deterministic build/query over migrated paper extractions; command reports the loaded count | Add approved current-review evidence, gap records, and decision links |
| Oracle contracts | Implemented and tested | Integration with a measured tracker and matched oracle study |
| Generated oracle | External LLM proposals admitted and exercised in GMT | Broader VIBE guidance contract and learning benefit |
| Generated task reward | Bounded GMT recipe revisions admitted and trained | Broader supplied task contract; no arbitrary Python admission implied |
| Evidence UI | Read-only status, research queries and registered local G1 evidence views | Portable artifact packaging and dynamic-task views |
| Gymnasium adapter | Active GMT/G1 proxy; native Humanoid-v5 is a separate earlier family | VIBE integration, not relabeling proxy results |
| Frozen reference tracker | GMT actor admitted; causal input tests recorded | Exact VIBE checkpoint/interface admission |
| Protected evaluator | Mechanical metrics implemented | Task, transition, contact, recovery, and naturalness calibration |
| Canonical trajectory | Synchronized GMT state/action/reference evidence and feedback implemented | Dynamic-task object/contact signals from the supplied adapter |
| Humanoid improvement | Local GMT factor effects; no full-task pass or demonstrated efficiency gain | Independent task success, tuning cost and held-out comparisons |
| Cross-MDP generality | No evidence | Same harness contract on a second adapter |

Current scope: [2026-09-08 plan](strategy/astra/POST_TRAINING_PLAN_20260908.md).
The native study gates below remain historical family requirements. They do not
turn a GMT proxy into VIBE evidence or require a new controller project.

## Program versus study

```text
program knobs:       oracle O_k       task reward r_k
                         |                   |
single-factor study: change O / freeze r   change r / freeze O
combined study:           declared O x r factorial
```

Within every study, freeze:

- robot, scene, observation/action spaces, dynamics, resets, and termination;
- initial tracker/controller, permitted trainable set, tracking reward,
  normalizers, and command ABI;
- trainer, hyperparameters, seeds, budget, and checkpoint rule;
- protected evaluator and task-outcome definitions; and
- admitted reference bytes and their embodiment, joints, units, frames, and
  cadence.

Changing one of these creates a separate experiment family.
The given trainer may update its declared policy parameters during post-training.

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

The native formal reward study remains gated. A distinct GMT data-only recipe
path is implemented; these are not interchangeable admission claims.

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
