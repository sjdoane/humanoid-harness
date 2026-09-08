# Automated post-training project charter

**Status:** current program contract

**Established:** 2026-09-02

**Revised:** 2026-09-08 from Samuel's updated proposal and sole-lead instruction

## One-sentence goal

Automate task-specific post-training in SONIC-based VIBE by revising reference
guidance and task reward from rollout feedback, testing whether this improves
task success with fewer training interactions and less manual tuning.

- Composition supports learning; transitions alone are not the research goal.
- The scene, sensing, controller family, training algorithm, initial checkpoint,
  permitted trainable components and evaluator come from the supplied system.
- That trainer may update its designated policy parameters. It cannot be
  replaced or silently retuned to make a guidance candidate win.
- Scaling and context-only control are empirical alternatives, not presumed
  failures. Check the actual checkpoint and interface before asserting a limit.

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
| Literature graph and decision memory | Semester target: lab VIBE/SONIC integration |
| Optional targeted human questions | Current proxy: GMT/G1 through Gymnasium/PPO |

The controller is not the research contribution. Native Gymnasium Humanoid is
an earlier adapter; GMT/G1 is the current proxy. Neither establishes VIBE
compatibility or success on the proposed dynamic tasks.

## Research questions

| ID | Question | Frozen counterpart |
|---|---|---|
| `RQ-O` | Does rollout-informed reference guidance improve post-training task success or effort? | Task reward |
| `RQ-R` | Does generated task-reward revision improve predeclared task success without violating protected guardrails? | Oracle |
| `RQ-OR` | When both change, which gains are attributable to oracle, reward, or their interaction? | Full factorial controls |
| `RQ-S` | Can compact human feedback steer either artifact predictably? | Trainer and evaluator |

## Semester comparison and tasks

- Planned task order: loaded-cart interception; changing obstacle course;
  floor hockey; volleyball. These are proposal commitments, not installed
  environments, validated feasibility, or novelty claims.
- First measure the initial policy. Compare post-training with guidance kept
  fixed, revised manually, and revised by the harness. Manual and automated
  tuning receive the same types of rollout feedback, including available video.
- Primary outcomes: independent task success; cumulative simulator steps and
  revision rounds; human edits/interventions and active tuning time. Also record
  compute time and model-call cost. Unknown costs are unknown, not zero.
- Keep task, initial policy and training budget matched. Count unsuccessful
  candidates and failed attempts in tuning cost; do not report only the winner.
- Final variations must not guide revision. Record training-seed replicates
  separately from repeated evaluations of a single policy.
- Numerical success thresholds, task assets, hardware and exact VIBE ABI are
  unresolved. Do not label the GMT posture region a physical obstacle.
- See the [current implementation plan](strategy/astra/POST_TRAINING_PLAN_20260908.md).

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

General reward-program generation remains the research target. The GMT/G1
adapter now executes bounded, data-only reward recipes and feedback-linked
LLM revisions. That implemented subset does not admit arbitrary generated
Python or establish successful task learning. See the
[current evidence](strategy/astra/G1_LEARNING_RESULTS_20260906.md).

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

## Native admission family (historical setup, separate from GMT/G1)

Experiment 001 remains a valid prerequisite:

- adapter: Gymnasium `Humanoid-v5` plus PPO;
- authorable research factor: none;
- purpose: test static tracker feasibility under a frozen stand target and
  frozen reward;
- claim ceiling: static tracking feasibility only; a static target cannot prove
  causal use of a time-varying reference;
- current evidence: interface checks only.

Experiment 002 is the distinct native time-varying reference installation and
causal-use study. Its admission requirements continue to govern the native
Experiment 003 family. The observed immediately falling Humanoid does not
demonstrate tracker or oracle quality.

The separately versioned GMT/G1 adapter now supplies a frozen reference tracker
for bounded composition/reward learning studies. Its implementation and results
are recorded in the [current G1 evidence](strategy/astra/G1_LEARNING_RESULTS_20260906.md).
This does not repair, certify, or substitute for a failed native experiment.
Both adapters remain subject to the same scientific boundaries above.

## Deferred scope

- Editing the foundation controller, tracking reward, scene, or dynamics as an
  unreported way to improve a harness candidate.
- VIBE execution until the actual private interface and permitted use are
  confirmed; no requirement to publish private lab code before local integration.
- New perception/controller research and raw motion generation.
- Hardware deployment until lab approval and readiness; proposal weeks 13–14
  are conditional sim-to-real work, not present authorization to move a robot.
- Autonomous external actions beyond bounded local research runs.

## Deliverables

- Convenient open-source CLI and evidence UI.
- Queryable, provenance-bearing research graph.
- Reproducible fixed/manual/harness post-training comparisons on the four
  planned VIBE tasks, with feasibility and unresolved dependencies explicit.
- Held-out results, full tuning-effort accounting, and component attribution
  where a separately controlled comparison supports it.
- Paper-ready figures, tables, examples, and a lead-author manuscript when the
  evidence supports the claims.
