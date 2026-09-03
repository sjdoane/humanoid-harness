# ADR 0002: root-pose-gated static tracking reward candidate

**Status:** provisional; freeze before behavioral training  
**Date:** 2026-09-02

## Decision

Use a 45D Humanoid target for the static positive control:

```text
root z + world wxyz + world linear velocity + torso-local angular velocity
+ 17 actuator-ordered joint positions + 17 actuator-ordered joint velocities
```

World `x/y` root position is omitted. The numerical training reward multiplies
the joint/velocity detail score by a root height-and-orientation gate. Bounded
Cauchy root terms preserve a recovery signal far from the target; Gaussian
joint/velocity terms retain tight nominal tracking. The stock Gymnasium reward
is logged but contributes zero; declared task reward is zero.

At state `x[t]`, actor and critic see reference frames `r[t:t+H]` with
`H >= 2`. Action `a[t]` advances the plant to `x[t+1]`; that post-step state is
graded against the already-visible `r[t+1]`. This avoids a hidden one-step lag
when time-varying references are introduced.

## Why

A joint-only score assigns almost perfect reward to a fallen humanoid whose
limbs happen to match the reference. The root gate makes that shortcut low
value: at root height `0.2 m`, with every joint exactly on the stand target, the
measured score is below `0.03`. Moving the root toward the target strictly
increases it; a typical tilted fall scores lower still.

## Boundary

- The scale and weights are an authored candidate, not literature-established
  constants.
- The protected evaluator recomputes physical root, orientation, velocity,
  joint, and collapse metrics without trusting reward telemetry.
- The long-range Cauchy term is intended to avoid a nearly flat fallen-state
  reward, but recovery learnability remains unmeasured.
- Do not tune this reward after comparing oracle arms. Any later change creates
  a new tracker experiment family and checkpoint lineage.

## Revisit before

- training from fallen-state resets;
- certifying get-up/recovery references; or
- freezing the tracker used by oracle-composition comparisons.
