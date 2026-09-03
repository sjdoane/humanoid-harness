# Local phase-oracle exploration v1

|  |  |
|---|---|
| progress | The design, runner, protected evaluator, artifact identities, seed schedule, and run order are implemented and reviewed. No v1 result has been inspected. |
| bottleneck | The reference is Tier K, the local checkpoints are not redistributable, their training lineage is unavailable, and an excluded-seed pipeline probe still fell. |
| next step | Commit the pre-run slice, generate the exact manifest candidate, obtain byte-specific human review, then execute the frozen table once. |

## Evidence boundary

| Surface | Status |
|---|---|
| Research target | Diagnose bounded phase selection and explicit residual fallback around one fixed local controller bundle. |
| Implemented here | Strict design parsing, 240-run expansion, exact preflight, one-attempt execution, canonical traces, and protected evaluation. |
| Measured evidence | No v1 outcome. One excluded-prior-seed pipeline probe verified 1,000-action recording but fell; it is not formal study evidence. |

- Evidence class: `exploratory`.
- Distribution: `local_exploration_only`.
- Claim ceiling: exact-byte local replay and conditional four-arm diagnostics.
- No automatic promotion is possible.

## Frozen path

```text
registered Minari episode 0 -> exact 45D projection -> phase/fallback arm
                                                   -> fixed BC + residual
                                                   -> Humanoid-v5
Humanoid state + cached next reference frame ------> protected evaluator
Humanoid + controller + oracle facts --------------> canonical trace
```

- Configuration: [`configs/local_v1.study.json`](configs/local_v1.study.json).
- The configuration pins the current registered-import receipt, source record,
  source files, reference, controller files, rewards, requested runtime, oracle
  rules, and schedule. The projection receipt is recomputed from the exact
  registered bytes; a cached digest alone is not accepted.
- Local file paths are supplied at execution. Only filenames and SHA-256
  identities are public.
- Before execution, a reviewed manifest must bind the importer, reference
  schema/state extractor, design parser, runner, phase oracle, protected
  evaluator/readers, tracking reward, trace contract, and both controller-load
  receipts. A resolved path or pre-load file hash is not a load receipt.
- The manifest must bind `oracle_exploration_design_sha256`, recomputed from
  the parser's canonical design bytes. A path or filename is not the design
  identity.
- Frozen canonical design SHA-256:
  `8af8744fe4ded63181e9a50781de2cf3d9607cd954ac7c0601b9a04cd6902e9d`.
  A mismatch stops execution before environment or controller initialization.

### Requested configuration versus observed facts

| Surface | Requested and frozen now | Observed before execution |
|---|---|---|
| Packages | Gymnasium `1.3.0`; MuJoCo `3.12.0` | Exact imported versions |
| Model | Humanoid XML SHA-256 `85816f37…` | Hash of the file actually loaded |
| Environment | Reset, termination, frame skip, cadence, horizon | Constructed wrapper and `TimeLimit` facts |
| ABI | Exact Humanoid-v5 request | Spaces, little-endian float32 action dtype, state shapes, ordered joints and indices, gears, torque capacities |

- Requested values never count as observed execution facts.
- Inspection must emit one immutable runtime/model/ABI receipt.
- That receipt must match every requested field and contain every required
  observed field before its SHA-256 can enter the execution manifest.
- Gym reset noise perturbs all four root-quaternion components. At reset only,
  the runner requires `w` in `[0.99, 1.01]`, each vector component in
  `[-0.01, 0.01]`, and norm in `[0.98, 1.02]`; it L2-normalizes a copy for
  the action-zero phase matcher and recovery gate. Raw simulator state and the
  controller observation remain unchanged. Every post-step quaternion must
  already be unit length; no later projection is allowed.

## Design

### Two-by-two oracle factors

| Arm | Phase | Residual fallback |
|---|---|---|
| `T0_G0` | Elapsed-time phase | Never disabled |
| `T0_G1` | Elapsed-time phase | Explicit recovery gate |
| `T1_G0` | Bounded state-conditioned phase | Never disabled |
| `T1_G1` | Bounded state-conditioned phase | Explicit recovery gate |

- Task reward, controller, reference, simulator, evaluator, and action rule stay
  fixed across arms.
- The BC-only context shown in v0 is not a fifth factorial arm.

### Phase distance

- Quaternion columns are exactly `[1, 2, 3, 4]` in `wxyz` order.
- Orientation error is the shortest sign-invariant geodesic angle.
- The angle is scaled by `0.5 rad`; unit quaternions must be within `1e-6`.
- Pose error averages 17 squared standardized joint-position errors and one
  squared scaled geodesic-angle error.
- Quaternion columns never use per-component standard-deviation scoring. Their
  scale-array entries are inert validation fill values.
- Joint-position scales use population standard deviation (`ddof=0`) over all
  1,001 reference frames, at exact indices `11` through `27`, with a `0.1 rad`
  floor. Every other scale-array entry is exactly `1.0`.

### Action and evaluator

- Controllers share the exact 17-joint Humanoid actuator order in the design.
- Composition is float32: cast base/residual, multiply by the float32 `0.08`,
  add, then clip to the exact float32 action-space endpoint bytes around
  decimal `±0.4`. It occurs after oracle selection and before the environment
  step.
- The environment receives that exact float32 command. The physical-action
  trace records the same bytes promoted to float64 without changing value.
- MuJoCo clips the submitted command to its exact model control range
  `[-0.4, 0.4]` before actuation. The canonical normalized action and protected
  evaluator use `clip(command / 0.4, -1, 1)` and verify it against observed
  post-transmission torque.
- `W[t]` is cached before the action. The post-step state `x[t+1]` is graded
  against `W[t][1]` exactly once.
- Collapse means post-step root height outside the inclusive `[1.0, 2.0] m`
  interval or torso-up-z below the inclusive `0.5` bound.
- No-collapse is true only when all 1,000 required post-step samples are not
  collapsed. Direct simulator state, post-transmission torque, prior executed
  action/acceleration, and all contacts from all five substeps feed metrics.

### Repeated measures and blocking

- Predetermined seeds: `4000` through `4019`.
- Prior v0 seeds `3000` through `3019` are prohibited.
- Each seed receives all four arms in all three conditions.
- Each `(seed, condition)` pair is one complete four-arm block.
- SHA-256 ranking randomizes block order and arm order reproducibly.
- Evaluation seeds are repeated measures within one checkpoint bundle. They are
  not independent policy-training replicates.

### Perturbations

| Condition | Direct intervention | Magnitude |
|---|---|---:|
| `nominal` | None | `0` |
| `lateral_velocity` | World-frame root-y linear velocity | `±1.0 m/s` |
| `pitch_velocity_falsifier` | Torso-local root-y angular velocity | `±1.0 rad/s` |

- Perturbation actions are balanced over `200`, `325`, `450`, `575`, and `700`.
- At each action, the four assigned seeds contain all lateral/pitch sign pairs.
- The literal per-seed schedule is part of the configuration.
- Every intervention is applied at the same boundary for all four arms of its
  seed and condition.

## Runtime order

For action `t`:

1. Read observable state `x[t]`.
2. Apply the scheduled state intervention, if any, and record its exact vector.
3. Select and cache the untransformed reference window `W[t]`.
4. Compose the fixed BC action and gated residual action.
5. Step the fixed Humanoid environment once.
6. Grade `x[t+1]` against cached `W[t][1]` with the committed evaluator.
7. Record the boundary in the canonical trace.

- Never select an evaluator target from `x[t+1]` retroactively.
- One complete run has 1,000 actions and 1,001 boundary samples.
- An oracle contract fault is a hard failure, not permission to substitute a
  fallback implementation.

## Outcomes

- Primary diagnostic: no collapse over 1,000 actions.
- Report paired seed-level phase, fallback, and interaction contrasts.
- Also report collapse onset/duration, physical recovery and task-resumption
  latency, gate occupancy and sinks, phase offset/stall/chatter, tracking error,
  action rate and saturation, torque, jerk, and floor contacts.
- Stock Gymnasium return remains diagnostic only. Each step value must be
  finite; a completed run accumulates exactly 1,000 values in action order
  with Python `math.fsum`.
- Missing runs remain visible. Outcome-based retry is prohibited.
- Within a formal run attempt, before the first environment is created, the
  runner atomically claims the reviewed-manifest identity in the resolved
  `runs` directory. That identity
  may run once in that directory. A fault retains an immutable receipt with
  completed and missing rows. A local filesystem cannot prevent someone from
  copying inputs to another output root; the no-retry rule still applies and
  every receipt says so explicitly.

## Hard gates

- A reviewed execution manifest binds every required pre-execution role.
- The observed runtime/model/ABI receipt matches the requested runtime exactly.
- All 240 scheduled tuples appear exactly once.
- Every completed run has 1,000 actions and 1,001 trace samples.
- No oracle contract fault occurs.
- State, action, torque, and metric values remain finite.
- Contact capture covers all five physics substeps.
- Trace and receipt hashes reverify.

Failure of a hard gate prevents a comparative interpretation.

## Nonclaims

This exploration cannot establish:

- Tier-D reference feasibility or distributable reproduction;
- that seeds are unseen by checkpoint training;
- held-out motion, reference composition, or generated-oracle quality;
- causal use of reference, phase, mode, or clock metadata;
- task-reward improvement, policy-training replication, or cross-adapter
  generality; or
- paper-ready evidence.

The controller receives a numeric `348 + 8 × 45` input. It receives no explicit
mode, phase, or clock channel. A separate matched-state exact/zero/shuffled/shifted
intervention is required before causal-reference-use language.
