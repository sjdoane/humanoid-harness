# G1 course pilot — frozen before course data

| state | evidence |
|---|---|
| progress | Supplied walk and crouch replay; exact matched-state reference influence; course runtime implemented. |
| bottleneck | Shared-runtime parity, novel switches and learned task improvement remain unmeasured. |
| next step | Parity → zero-residual course probe → fixed-budget learning → objective diagnosis → one reviewed artifact revision. |

## Architecture

```text
INPUT: task + admitted numeric motion library + development feedback
                         ↓
              LLM proposes O or r_task
                         ↓
             strict data-only admission
                         ↓
robot state → O(state, phase) → H×D reference → GMT actor [FROZEN WEIGHTS]
     ↑             ↓                 ↓                    ↓
     │        phase/mode ───→ residual policy ← task       + bounded residual
     │                           ↑                        ↓
     └──────────── fixed G1 MuJoCo plant ← 50 Hz / PD at 1 kHz
                                 ↓
            independent trajectory metrics + recorded-state video
                                 ↓
                    OUTPUT: diagnosis / next LLM proposal
```

- Frozen: dynamics, reset, observations, action mapping, base weights, tracking
  reward, trainer, budget, seed, final-checkpoint rule and evaluator.
- Authored: reference segments/guards/phase transfer (`O`); bounded task-reward
  weights (`r_task`). Never generated simulator or evaluator code.
- The actor observes executed-action history. Frozen weights do **not** imply
  unchanged behavior. The ±0.125 rad bound applies to the immediate additive
  target, not the cumulative closed-loop effect.
- LLM reasoning runs between experiments. No LLM call controls a 50 Hz step.

## New family, not a retroactive MDP replacement

- GMT/G1 supplies a real reference-conditioned controller and compatible motions.
- Keep Gymnasium as the learning interface. Keep native Humanoid-v5 results
  separate; do not compare outcomes across robots as a treatment effect.
- Scene: pinned flat ground. A posture region is a task constraint, **not** a
  physical obstacle. No collision-avoidance or locomanipulation claim.
- Reset: one fixed home keyframe and warmup step. Seed changes policy RNG only.
- Base provenance: reviewed numeric reconstruction, not original JIT execution
  or a proven JIT-equivalent implementation.

## Pilot contract

| item | fixed value |
|---|---|
| posture region / finish | initial-heading progress [1, 2) m / 3.5 m |
| horizon | 1,000 control intervals; 20 s; finish does not terminate |
| target speed | outside 0.70 m/s; inside 0.65 m/s |
| posture constraint | root height [0.30, 0.60] m inside actual region |
| failure | root <0.30 m; torso-up <0.5; non-foot ground contact |
| initial O1 | full walk → crouch crop 3.5–5.8 s → full walk |
| initial switch guards | progress ≥0.65 m / ≥2.05 m; minimum dwell 25 ticks |
| phase transfer | nearest native pose; unchanged cadence; explicit crop wrap |
| initial task weights | speed 1; posture 2; lateral 1; heading 0.5; failure 1 |
| task error scales | speed 0.30 m/s; posture 0.20 m; lateral 1 m; heading 0.50 rad |
| residual | normalized ±1, multiplied by 0.25 in pre-clip GMT raw-action units |
| policy / optimizer | PPO; 128×128 MLP; zero initial mean; log std −1.5; CPU |
| rollout / batch / epochs | 512 / 128 / 4 |
| learning rate / discount | 0.0003 / 0.99; GAE 0.95 |
| clipping / KL | 0.2 / target 0.02 |
| normalization | advantages yes; observations and rewards no |
| first training budget | 32,768 transitions, seed 20260906, final checkpoint only |

- Crops are kinematic candidates until the real switch probe runs.
- All-positive components are gated off after a fall. Outside-region posture
  counts as a satisfied constraint; there is no reward for lingering inside.
- Safety is checked throughout each 20-substep interval, not only at boundaries.
- The critic receives the same reference/task/phase observation as the actor;
  PPO bootstraps the actual next observation on a time limit, not on a fall.

## Comparisons and claim limits

| arm | oracle | task reward | purpose |
|---|---|---|---|
| O0r0 | full walk in all three state slots | zero weights | matched tracking-only control |
| O1r0 | walk/crouch/walk | zero weights | composition contribution |
| O0r1 | full walk | proposed weights | task-reward contribution |
| O1r1 | walk/crouch/walk | proposed weights | joint system |

- First replay probe is not a trained control. Compare trained arms at identical
  initialization and budget before attributing learning to either knob.
- Objective gates: full horizon without fall; finish plus actual region entry
  and exit; ≥25 inside samples; ≥75% posture compliance; inside minimum height
  ≤0.50 m; inside mean speed within 0.10 m/s of target; total speed MAE ≤0.35 m/s;
  lateral error ≤0.75 m; joint RMSE p95 ≤0.35 rad; roll/pitch RMSE p95 ≤0.25 rad.
- Report continuous outcomes. The 0.50 m dip threshold is close to the supplied
  crouch baseline's 0.503 m minimum; millimetres are not a broad competence claim.
- Oracle switches are diagnostics, not objective success gates: O0 must remain
  eligible to succeed without using composition.
- Candidate reward hypothesis: close the crouch tracking speed/posture deficit.
  A speed gain with worse tracking does not support that hypothesis.
- One seed and one start establish only a development demonstration. Held-out
  inputs, other motions, perturbations and repeated seeds remain separate tests.

## Review provenance

- Fable task critique: `20260906T201618.012882Z-2b2ae6512ca948368a0da21bbbae5403`.
- Astra decision: `20260906T201915.702273Z-fc1fa5b3f2af4d9ba6044c3cc301671f`.
- Fable acknowledgment: `20260906T202031.095894Z-1fbd8a15bfc64063b6b4cb6a78ce9dd4`.
- These are pre-course-data decisions, not post-result changes to the gates.
