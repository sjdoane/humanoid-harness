# Oracle-composition project charter

**Status:** current research contract  
**Established:** 2026-09-02

## One-sentence goal

Determine whether a generated, state- and phase-conditioned reference oracle
improves held-out transitions and recovery for a fixed Gymnasium MuJoCo
Humanoid policy-training system.

## System contract

```text
reference set R -----------------------------+
robot/world/task state ------------------+   |
requested horizon -----------------------+   |
                                           v v
                                    oracle O_k       <- authorable
                                           |
                              mode + phase + H x D reference
                                           |
                                           v
                               frozen tracker C      <- fixed
                                           |
                                           v
                                frozen Humanoid MDP  <- fixed
                                           |
                                           v
                               protected evaluator  <- fixed
                                           |
                                      evidence E_k
```

## Primary claim under test

With every other experimental component held fixed, closed-loop oracle
composition yields higher sealed-test end-to-end completion than fixed or
open-loop playback without degrading nominal tracking or safety guardrails.

## Rival explanations

| Rival | Discriminating control |
|---|---|
| Extra observation channels—not oracle logic—cause gains | Identical observation ABI across all oracle arms |
| More training exposure or favorable resets cause gains | Paired seeds, identical state distribution and step budget |
| The tracker ignores the reference | Exact/constant-frame/shuffled/time-shifted matched-state interventions |
| Reward or evaluator changes cause gains | Immutable reward and protected evaluator hashes |
| Switching exploits task labels or privileged state | Separate observable-state and privileged-state families |
| A best seed/checkpoint creates the effect | Predetermined checkpoint rule; all seeds reported |

## Initial oracle arms

| ID | Mechanism |
|---|---|
| `O_fixed` | One fixed linear reference playback |
| `O_random` | Equal-budget random/open-loop mode exposure |
| `O_manual_phase` | Hand-authored time/phase schedule |
| `O_manual_state` | Hand-authored observable-state guards with hysteresis and recovery |
| `O_generated` | Generated typed oracle restricted to the same fields and references |
| `O_graph` | Optional motion-matching or graph-search comparator |

## Evidence ladder

| Gate | Claim earned |
|---|---|
| Contract and negative tests | The artifact is well formed and fails closed |
| Real `Humanoid-v5` smoke | The declared simulator path executes |
| Tracker causal-use ablations | The policy behavior depends on numeric references |
| Matched transition study | Oracle choice affects specified transition outcomes |
| Perturbation/recovery study | Closed-loop recovery works in the same episode |
| Sealed composition study | Improvement generalizes beyond tuned orders and phases |

No higher claim is allowed when a lower gate is missing.

## Initial scenarios

1. Stand or gait tracking positive control.
2. Random phase, speed change, and dropped-window synchronization.
3. Stand → walk → run → stop with randomized switch times.
4. Mid-episode branch or heading change.
5. Lateral perturbation with recovery and rejoin.
6. Prone/supine → get up → stand → walk → turn → stop.
7. Held-out order, phase offset, and perturbation family.

## Deferred scope

- Generated task-reward research.
- Private VIBE or lab-runtime integration.
- Vision-conditioned world models.
- Hardware deployment.
- A user interface.

Any of these requires a new chartered experiment family.

## Deliverables

- Reproducible public repository with immutable run receipts.
- Source-traceable literature review and failure taxonomy.
- Frozen reference-tracker baseline and causal-use evidence.
- Matched manual/generated oracle studies on Gymnasium Humanoid.
- Paper-ready figures and tables if the evidence supports a claim.
