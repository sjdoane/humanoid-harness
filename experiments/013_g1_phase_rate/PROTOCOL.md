# Within-mode phase-rate feedback: probe only

| progress | The current oracle selects modes from robot state but advances within-mode phase on a clock. |
|---|---|
| bottleneck | O5 loses forward progress during repeated crouch loops. Clock desynchronization is a hypothesis, not an established cause. |
| next step | Qualify one balanced reference-loop candidate, then review phase-rate feedback separately. |

## Pre-data amendment: qualify the reference first

- Amended after independent review of `0151e7c`, before any Study013 simulation.
  No old result or prediction is rewritten. Phase-rate core is still unwired.
- Stage A: one data-only O7 oracle proposal from O5 feedback. Change only
  crouch `loop_start_seconds: 3.90 -> 3.78` and `end_seconds: 4.86 -> 5.64`;
  a new oracle ID records lineage. All other O5 values remain exact.
- Retain four-state probe-only runtime, zero residual, seed `20260906`, all
  guards/entry/rise/walk/reward/reset/task/evaluator. No finite-horizon correction
  or phase-rate code is imported into this source before the probe finishes.
- Motivation: parent numeric-only audit of pinned motion `a67c364e...2be964`,
  50 Hz float32 offsets strictly below each window duration. Old/new local
  sideways velocity means are -0.139243 / +0.000066 m/s (48/93 samples);
  forward means 0.615116 / 0.608843 m/s. These are reference features, not world
  displacement, measured robot motion, contact labels or feasibility evidence.
- Stage A admission: 20 s/no fall, observed region exit, and actual rise/after
  execution. Report every original task gate; no training follows automatically.
- A failure ends this qualification; no extra crop or new seed in Stage A.
- Stage B may proceed only after a successful Stage A and a separate pinned
  matched protocol using O7 as the fixed reference in both clock/rate arms.
  The O5-only phase contrast below is superseded before execution; it remains
  a design record, not permission to select whichever baseline looks better.

## Frozen hypothesis

- Within a later Stage B, the qualified reference retains its exact clips,
  crops, guards, base controller, reset, reward,
  task and evaluator. Only the within-crouch phase scheduler changes.
- No training. No residual policy. No inference of original GMT equivalence.
- O5 is deliberately a failed development case; this is not held-out evidence.
- Default/disabled execution must remain the existing clock path exactly.

## Continuous scheduler

- Operate only in `crouch`; decision after each five completed intervals
  (0.10 s). On reset or behavior entry use rate 1 and reset the block counter.
- At current unwrapped phase `phi`, compare offsets `[0, -delta, +delta]`,
  `delta = 1 / float32(native_fps)`. Reject negative candidate source phases.
- Rate options: `1 + offset / 0.10`; admit only native cadences for which these
  rates are positive. The pinned crouch cadence gives approximately 0.667–1.333.
- Never jump phase to the selected candidate. Integrate the chosen positive
  rate for the NEXT five intervals. First-minimum tie order is the order above.
- Use the existing 30D reference feature order: height, roll/pitch, root-local
  linear velocity, local yaw rate, 23 joints. Robot features use the same frame.
- Frozen matching scales: `[.15, .35, .35, 1, 1, 1, 2] + [.35] * 23`.
  Score mean squared normalized error. Velocity features of each candidate are
  multiplied by its hypothesized rate; pose features are not.
- This score is a heuristic for selecting phase rate, never an objective gate.
- It matches a retimed plan, coupling phase and speed; it is not pure phase
  estimation. Pose/velocity partial costs and opposite-sign choices are useful
  diagnostics. Velocity discontinuities at the plan's rate boundary are a risk.

## Issued window and causal order

- Each issued window uses the chosen rate only through the remaining part of
  the five-step block, then nominal rate 1 beyond the unobserved next decision.
- Transform reference times through that piecewise clock. Multiply the sampled
  linear/yaw velocity features by its local rate; do not retime stored assets.
  These are new kinematic windows, not inherited dynamics-certified motions.
- Execute an interval; score its endpoint against the previously issued plan.
  Only then evaluate state/loop-boundary transitions and select the next rate.
- Loop crossings use consecutive unwrapped phases. No rewind, cross-segment
  phase search, stale guard latch, or decision made from a future robot state.
- Record rate/phase, block position, three candidate costs, chosen offset,
  loop crossings, and the plan used for the executed-interval target.
- Missing/non-finite/wrong-shaped state or inconsistent cadence fails closed.

## Verification and one probe pair

- Pure tests first: positive monotonic phase, exact clock when disabled, tie
  order, reset/entry, block boundaries, window continuity, velocity scaling,
  endpoint-before-decision semantics, and invalid inputs.
- New explicit probe-only runtime/evidence identity; no training admission.
- Source/parent/motion identities and independent readback required before run.
- First reproduce retained clock-only O5 numerical trajectory/frames exactly.
  Allow only predeclared runtime/reset metadata differences; stop on mismatch.
- Then one aligned O5 probe, same fixed start/seed. One local job at a time,
  existing 1,200 s / 8 GiB ceilings, exact peer resource reservation.

## Reading and stop

- Useful mechanism result requires 20 s without a fall, observed spatial exit,
  and both rise and after modes actually executed. Report all original gates.
- Report forward speed at actual loop crossings, heading/progress at step 300
  if reached, every phase correction, contacts, and tracking errors.
- Survival or lower tracking error without progress/required transitions fails
  the mechanism screen. A fall remains a failed candidate, not a partial win.
- One-sided rate selections are diagnostics, not proof of aliasing or stalling.
- Symmetric-pose aliasing, absent reference-contact labels, noisy velocities,
  changed command distribution and bounded but abrupt rate changes remain risks.
- No crop changes, further rate tuning, reward edits, training, or seed search
  follow automatically. This is not OGMP equivalence or generalization.
