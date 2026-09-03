# Agentic oracle-and-reward project charter

**Status:** current program contract

**Established:** 2026-09-02

**Revised:** 2026-09-02 after proposal and meeting review

## One-sentence goal

Build an auditable LLM-driven harness that improves state-aware reference
composition and executable task rewards around a fixed policy-training MDP.

## System contract

```text
task + reference set R + initial reward r0
                    |
                    v
     +-----------------------------------+
     | agentic design harness            |
     |                                   |
     | literature graph ----+            |
     | rollout diagnosis ---+--> decide  |<---- optional human steering
     | missing evidence ----+            |
     +-------------+---------------------+
                   | O_k + r_k
                   v
     parameterized policy-training MDP          <- adapter; fixed in a study
                   |
                   v
              policy pi_k
                   |
                   v
        independent evaluator
                   |
                   v
     trajectory + metrics + failures E_k
                   |
                   +----------------------------> next iteration
```

## Ownership boundary

| Harness owns | Adapter supplies and freezes |
|---|---|
| Versioned reference oracle `O_k` | Scene, robot, observations, actions, dynamics |
| Versioned executable task reward `r_k` | Foundation tracker/controller and tracking reward |
| Candidate generation and bounded patch decisions | Training algorithm and controller interface |
| Experiment design and immutable receipts | Policy checkpoint and raw trajectory signals |
| Independent evaluation and failure diagnosis | Adapter runtime fingerprint |
| Literature graph and decision memory | Later: lab VIBE/SONIC-compatible integration |
| Optional targeted human questions | Current: Gymnasium/PPO development integration |

The controller is not the research contribution. Gymnasium Humanoid is the
first public development adapter, not the identity of the complete system.

## Research questions

| ID | Question | Frozen counterpart |
|---|---|---|
| `RQ-O` | Does closed-loop, state- and phase-aware oracle composition improve held-out transitions and recovery? | Task reward |
| `RQ-R` | Does generated task-reward revision improve predeclared task success without violating protected guardrails? | Oracle |
| `RQ-OR` | When both change, which gains are attributable to oracle, reward, or their interaction? | Full factorial controls |
| `RQ-S` | Can compact human feedback steer either artifact predictably? | Trainer and evaluator |

## Experiment families

| Family | Authorable factor | Required controls |
|---|---|---|
| A: oracle | `O_k` | Fixed `r`, trainer, seeds, budget, evaluator, references |
| B: reward | `r_k` | Fixed `O`, trainer, seeds, budget, evaluator, references |
| C: attribution | Declared `O x r` factorial | Four locked arms: `(O0,r0)`, `(Ok,r0)`, `(O0,rk)`, `(Ok,rk)`; matched training seeds, budget, evaluator, and checkpoint rule; candidates locked before sealed evaluation |
| D: steering | One explicit human instruction targeting either `O` or `r` | Untargeted knob frozen; predeclared predicted artifact change and falsifier |

## Oracle contract

An oracle maps observable robot, world, and task state plus a requested horizon
to:

- active mode and local phase;
- a controller-native `H x D` reference window;
- the transition decision and reason; and
- explicit next oracle state, including recovery and rejoin.

Elapsed-time playback is a baseline. It is not sufficient evidence of a
state-aware oracle.

## Task-reward contract

A task reward is an immutable, executable program with:

- declared inputs, units, bounds, and update cadence;
- a source-linked rationale and predicted behavioral effect;
- static and sandbox validation before training;
- an independent task metric that the reward cannot edit; and
- a matched comparison against `r_0` with the oracle frozen.

Reward generation is a program target. It is not implemented yet.

## Diagnostic dossier

Every evaluated episode should eventually expose one synchronized trace:

| Surface | Minimum evidence |
|---|---|
| Task | completion/progress and task-state changes |
| Robot | `qpos`, `qvel`, root state, contacts, falls |
| Controller | action, target, torque, energy, saturation |
| Oracle | mode, phase, reference window ID, guard values, transition reason |
| Quality | tracking error, boundary error, jerk, contact agreement |
| Recovery | disturbance, detection, resynchronization, rejoin, task resumption |
| Agent | evidence read, missing information, candidate, rationale, prediction, falsifier |

Unavailable signals remain explicit. They are never filled by inference and
reported as measured.

## Evidence ladder

| Gate | Claim earned |
|---|---|
| Contract and negative tests | The artifact is well formed and fails closed |
| Real adapter smoke | The declared simulator path executes |
| Causal-use ablations | The policy behavior depends on the changed artifact |
| Matched single-factor study | The declared factor affects specified outcomes |
| Perturbation and held-out study | Recovery or task behavior generalizes within the adapter |
| Cross-adapter study | The harness interface transfers beyond one MDP |
| Full attribution study | Oracle, reward, and interaction effects are separable |

No higher claim is allowed when a lower gate is missing.

## Current admission experiment

Experiment 001 remains a valid prerequisite:

- adapter: Gymnasium `Humanoid-v5` plus PPO;
- authorable research factor: none;
- purpose: test static tracker feasibility under a frozen stand target and
  frozen reward;
- claim ceiling: static tracking feasibility only; a static target cannot prove
  causal use of a time-varying reference;
- current evidence: interface checks only.

Experiment 002 is the distinct time-varying reference installation and
causal-use study. The first Family-A oracle comparison is Experiment 003 and
remains blocked until Experiment 002 passes. The observed immediately falling
Humanoid does not demonstrate tracker or oracle quality.

## Deferred scope

- Editing the foundation controller, tracking reward, scene, or dynamics as an
  unreported way to improve a harness candidate.
- Private VIBE integration until a public adapter contract can be recorded.
- Vision-model research, raw motion generation, and hardware deployment.
- Autonomous external actions beyond bounded local research runs.

## Deliverables

- Convenient open-source CLI and evidence UI.
- Queryable, provenance-bearing research graph.
- Reproducible oracle and reward experiment families on at least two MDPs.
- Held-out baseline-to-final results with component attribution.
- Paper-ready figures, tables, examples, and a lead-author manuscript when the
  evidence supports the claims.
