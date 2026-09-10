# Make the 20-second task ending terminal

| progress | Source audit found timeout bootstrapping at the task's declared ending. |
|---|---|
| bottleneck | The legacy runtime estimates continuation value beyond an intrinsic finite horizon. Its behavioral effect is unmeasured. |
| next step | Add an explicit finite-horizon runtime; verify old behavior is unchanged, then run one matched learning screen. |

## Decision and evidence

- The course requires 20 s survival, exposes remaining time, and defines no
  post-horizon continuation. Interpret that ending as intrinsic to this task.
- Current `gym_env.step` marks it truncated. SB3 then adds `gamma * V(final_obs)`
  to the final reward. This is intentional legacy behavior, not a wrapper race.
- [Gymnasium's time-limit guidance](https://gymnasium.farama.org/main/tutorials/handling_time_limits/)
  distinguishes intrinsic finite-horizon termination from an external cutoff.
- Correct the objective in a **new runtime identity**. Do not rewrite old runs,
  silently change defaults, or attribute any contrast to oracle/reward quality.
- The mismatch is a correctness concern independent of whether it explains
  poor learning. Its numerical/behavioral impact remains an empirical question.

## Implementation boundary

- New opt-in, three-state finite-horizon runtime; same 2,171 observations,
  23 actions, plant/reset/controller, segments, guards and tracking reward.
- Terminate on fall **or** horizon; do not truncate at the intrinsic ending.
- Preserve the actual final observation and all existing raw metrics.
- Config/runtime/reset/manifest/supervisor/feedback/UI must agree on semantics.
- Existing legacy and four-state loop profiles retain their exact behavior.
- Tests: horizon alone, fall alone, simultaneous fall/horizon, ordinary steps,
  malformed/contradictory runtime, and the real SB3 collector bootstrap branch.
  Prove no bootstrap for the finite task and retained bootstrap for legacy.

## One-factor learning screen

- Control: retained Study011 normalized O2/r1, seed `20260906`, 131,072
  transitions, profile 3, final checkpoint only.
- Candidate: same exact config except the explicit finite-horizon runtime.
- Runtime implementation must be independently reviewed before any launch.
- If implementation code changes, first reproduce the control's ten outputs
  byte-for-byte at the new source. Stop on any mismatch; then run candidate.
- Candidate zero-residual numeric trajectory and per-step metric/action records
  must match the control. Allow only explicitly reviewed runtime/reset metadata
  differences; do not mask numerical or evaluator changes.
- Both jobs: separately accepted source/resource reservations, one at a time,
  1,200 s / 8 GiB maximum each. No paid compute or held-out data.

## Reading and stop rule

- Required for a useful one-seed learning substrate: 20 s/no fall, two switches,
  region entry/exit, >=25 inside samples, compliance >=`0.6419753086419753`,
  minimum inside height <=`0.5133562284879697` m, inside speed deviation <=0.10
  m/s, joint p95 <=0.35 rad and roll-pitch p95 <=0.25 rad.
- Report every original full-task gate, lateral error, total speed MAE,
  residual RMS, episode falls and critic diagnostics. No scalar reward win
  substitutes for the task gates. No numerical conditioning claim follows.
- A failed behavioral screen does not make incorrect bootstrap semantics
  correct. Retain both results; revise the next training design explicitly.
- No reward revision, phase alignment, seed expansion, early checkpoint choice,
  evaluator change or further budget under this protocol.

## Deferred branches

- Depth-reward screen is paused until finite-horizon semantics are resolved.
- Online within-mode phase alignment is a separate zero-residual oracle study.
- A frozen structured GMT feature extractor is a later trainer hypothesis,
  not evidence of a defect in the current flattened baseline.
