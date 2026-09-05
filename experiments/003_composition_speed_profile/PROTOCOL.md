# Experiment 003 cycle-0 protocol

| status | frozen value |
|---|---|
| progress | Inputs and four controller-switching arms are frozen before evaluation. |
| bottleneck | Controller switching is only a stand-in for a tracker following composed references. |
| next step | Run all four arms on seeds `97001`-`97020`, require a deterministic full replay of the first arm, then prepare cycle 1. |

## Design

| item | value |
|---|---|
| evidence label | `exploratory_oracle_cycle` |
| runtime | Registered plain `Humanoid-v5`; stock wrapper stack; `terminate_when_unhealthy=false`; `TimeLimit=1000`; five `0.003 s` substeps per `0.015 s` control step |
| controller | Strict data-only NPZ actor, deterministic mean, CPU |
| authorable factor | Oracle JSON only; stock task reward and every runtime item remain fixed |
| task spec | `task_spec_v1.json`, SHA-256 `edb2cffde9f2eb087667d182f3da199ef6956aeadc6d45e9e218649d6fa9cbd7` |
| library | `library_manifest_v1.json`, SHA-256 `ad57578dc2ed4fe3707da74a4f86620e758b1aa30064016785d187a5e86d3207` |
| evaluation seeds | `97001`-`97020`, no replacement or selection |
| horizon | one `1,000`-step episode per seed and arm |
| tuning | none after evaluation begins; a wrong frozen spec stops the slice |

The library speed medians and quartiles pool root-x sidecar differences over
the `28` blocks with `block_passed=true` in the v2 E3 certificate. Fall rate
uses all `36` predeclared E3 branches per actor: one first-fall event whenever
root height left `(1.0, 2.0) m` or torso up-axis fell below `0.5`, divided by
`36,000` steps and scaled to `1,000` steps. This keeps the admitted speed sample
and the unfiltered stability denominator explicit.

## Metrics

| metric | rule |
|---|---|
| episode speed error | Mean over all `1,000` post-step root-x speeds of `abs(v_x - v_target)`; fallen steps remain included |
| primary score | Median episode mean absolute speed error over all `20` fixed seeds |
| fall | Any reset or post-step boundary with root height outside `(1.0, 2.0) m` or torso up-axis `z < 0.5` |
| first fall step | Boundary `0` for reset, otherwise the first post-step boundary `1`-`1000` |
| switching | Controller changes only; state/recovery events are separately retained in the trace |
| behavior time | Executed control steps per behavior |
| task return | Sum of untouched stock `Humanoid-v5` reward, descriptive only |
| determinism | Re-run the first oracle on all `20` seeds after the four arms; every trace SHA-256 must match |

Metrics are computed from copied simulator state, the stock step reward, and
the stock `info`; they are not read back from generated trace files. Forward
speed uses root-x boundary differences, the same quantity used in the library
manifest. Traces retain guard inputs, controller identity, switch flags,
post-step state quantities, stock `info.x_velocity`, reward, and exact action
hash.

## Cycle-0 arms

| arm | program | predeclared behavior |
|---|---|---|
| `single_fast` | `arms/00_single_fast.json` | expert for all steps |
| `single_slow` | `arms/01_single_slow.json` | simple for all steps |
| `playback` | `arms/02_playback.json` | expert/simple/expert with switches exactly before actions `300` and `600` |
| `handwritten` | `arms/03_handwritten.json` | expert to medium braking to simple, then medium acceleration to expert; velocity/time/dwell guards and expert recovery override |

The handwritten arm was designed and frozen in a single sub-`30`-minute
window before any Experiment 003 oracle evaluation. It was not changed in
response to cycle-0 simulator output.

## Claim ceiling

Passing supports only that composition-loop cycle 0 ran on the frozen runtime
with four predeclared arms, that its report exists, and that a designer prompt
for cycle 1 exists. Controller switching stands in for tracker following. No
oracle-quality, generalization, tracker, task-reward, or humanoid-competence
claim is allowed.
