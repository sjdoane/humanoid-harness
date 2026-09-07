# Repeatable crouch and explicit rise: zero-residual feasibility

| progress | A versioned four-state oracle runtime and full feedback-to-revision path are implemented. |
|---|---|
| bottleneck | Neither repeat-loop nor rise transition has dynamics evidence. |
| next step | Probe the vocabulary control, then one feedback-linked O5 oracle proposal. |

## Fixed order and boundary

1. O4b control: identical oracle/reward/reference bytes to retained O4b, with the
   explicit four-state observation vocabulary. Require byte-exact zero-residual
   trajectory, frame ledger and evaluation against the legacy O4b probe.
2. Rebuild independent control feedback. Fable returns one retained JSON oracle
   proposal; the full `revise_g1_course` path verifies and applies it.
3. O5 candidate: one descent, repeatable crouch, explicit rise, then walk.
   Exact accepted source/config/resource binding is required before its probe.

- Config v2, profile `gmt_g1_four_state_loop_course/v1`, 2172 observations.
  Both cells use slots `before/inside/rise/after`; O4b never activates `rise`.
  These are not legacy 2171-dimensional training comparisons.
- Fixed G1 model/dynamics/home reset, actor weights and history interface,
  50 Hz control, admitted motion bytes/native cadence, task, recipe r1,
  tracking reward, evaluator and 20 s horizon. Literal zero residual throughout.
- Only O5's oracle program and segments change. This is a composition-package
  feasibility contrast, not attribution among its descent/loop/rise edits.
- No training. This runtime profile stays immutable and probe-only; later
  training requires explicit admission under a separately versioned contract.
- One heavy local job; CPU=1, memory<=8 GiB, wall<=1,200 s per exact request.
- Stop on artifact/runtime mismatch. Retain behavioral failures; do not search
  seeds, change the evaluator, or start training to conceal a bad transition.

## Candidate specification, before its rollout

| state | native reference | entry and exit |
|---|---|---|
| before | `basic_walk`, full `0..39.09166717529297 s` | min dwell 25; enter crouch at `x>=.65` |
| inside | `crouchwalk_stand`, `2.7..4.86 s`; repeat `3.90..4.86 s` after first pass | initial phase search `0–0.15 s`; min dwell 25; fresh `x>=2.05` guard only at loop boundaries |
| rise | `crouchwalk_stand`, `5.72..6.50 s`; terminal pose hold with zero velocity fields | entry phase 0; min dwell 24; leave at `(dwell>=24 and z>=.70) or dwell>=40` |
| after | same full `basic_walk`; entry search within first 15 s | no further transition |

- `inside.exit_at_loop_boundary=true` is mandatory for this candidate.
- Guards are not latched. A guard that ceases to hold before the next loop
  boundary cannot trigger a rise there. Phase tolerance is part of the profile.
- Phase remains segment-local: descent runs once; repeated crouch maps between
  `1.20` and `2.16 s`. Do not reuse the older `3.8 s` terminal-hold detector.
- No recovery program or same-behavior state transitions in this first profile.
- No motion retiming, interpolation-policy change or external data admission.

## Predictions and readout

- Predicted O5 feasibility: 20 s without a fall; all three transitions;
  rise entered at `x=2.05..2.70 m`; walk re-entry pose distance<=.25.
- Predicted posture: at least 71 actual inside samples and compliance>=59/71
  (the exact retained O4b baseline, approximately 83.1%).
  **All visits** to the spatial posture region count, including backtracking.
  First contiguous crossing is a separately labeled diagnostic, not a substitute.
- Report every unchanged full-task gate, minimum height, inside speed, after
  behavior, falls, transition steps/guards/phase, and each loop-boundary seam.
- Verify actual actor reference windows and their segment/phase identities;
  missing evidence stays missing. The separate intervention study is needed
  before a closed-loop causal-use claim.
- Do not predict depth success from timing alone: retained O4b already requests
  `.425 m` inside the region while the robot remains about `.09 m` higher.
- Known after-walk risk: full `basic_walk` has substantial local yaw rate. O5
  deliberately leaves this unchanged; turning/backtracking is a task failure,
  not contaminated data. A later straight-window proposal is a separate revision.
- A feasible O5 transition is not full-task success, learned competence,
  held-out generalization, contact certification or a physical-obstacle result.

## Execution and feedback

```text
task + admitted reference library + O4b/r1
               ↓
 four-state oracle → frozen GMT actor → fixed G1 dynamics
               ↑                              ↓
       observable robot state       trajectory + independent metrics
                                              ↓
             next probe ← validated O5 ← LLM proposal + verified feedback
```

The proposal cannot edit the actor, MDP, task, r1, tracking reward or evaluator.
Source requests and output receipts bind the exact local bytes for each step.
