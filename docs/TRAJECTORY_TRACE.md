# TrajectoryTrace/v1

## Purpose

One immutable trace should let a researcher answer:

- What was the robot state?
- What did the controller receive and emit?
- Which reference, mode, and phase were active?
- Why did a transition occur?
- What happened at contacts and task boundaries?
- Which needed signal was unavailable?

## Time contract

```text
sample 0             sample i                     sample N
seeded reset state   state after i transitions   final control boundary
action = zero        prior-interval action        prior-interval action
t = 0                t = i * control_period       exact cadence
```

Samples are contiguous and finite. Events point to an exact sample index.

## Required diagnostic coverage

Each name must be recorded or listed explicitly with a missing reason:

| Surface | Signal |
|---|---|
| Robot | `robot.qpos`, `robot.qvel`, `robot.root_position_world_m` |
| Controller | `controller.action`, `controller.motor_target`, `controller.applied_torque`, `controller.energy_j`, `controller.saturation` |
| Reference | `reference.frame`, `reference.window_index` |
| Contact | `contact.floor_normal_force_n` |
| Oracle | `oracle.mode`, `oracle.phase`, `oracle.transition_guard_margin` |
| Recovery | `recovery.disturbance`, `recovery.rejoin_state` |
| Task | `task.progress` |

`not_exposed`, `not_implemented`, `not_applicable`, and `redacted` are distinct.
Missing values are never replaced with zero.

For `behavioral_evaluation`, every required signal must be recorded. Each oracle
mode change also requires an `oracle.transition` event carrying the reason at
the exact sample index.

## Evidence classes

| Class | Meaning |
|---|---|
| `interface_check` | Wiring or simulator behavior only |
| `exploratory` | Useful diagnosis outside a locked comparison |
| `behavioral_evaluation` | Full execution-manifest, runtime, evaluator, oracle, policy, reference, tracker, tracking-reward, and task-reward lineage required |

## Storage boundary

- Canonical JSON bytes are the first portable encoding; their SHA-256 is the
  trace identity. Unknown fields, alternate dtypes, and noncanonical encodings
  fail closed.
- Numeric signal order and shapes are fixed by the trace schema.
- Artifact bindings use exact SHA-256 identities.
- Writes never overwrite an existing trace.
- Rerun or another viewer is a derived projection, not the canonical evaluator
  input.
