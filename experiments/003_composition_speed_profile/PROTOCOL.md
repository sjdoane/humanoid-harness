# Experiment 003 protocol

| status | current truth |
|---|---|
| progress | Phase A is closed after three deterministic controller-switching cycles on seeds `97001`-`97020`. |
| bottleneck | None of the three tested step-300 running-speed handovers survived; alternative phases, timings, and state-conditioned handovers remain untested. The cycle-2 program avoids the transition and the slow third. |
| next step | Phase B uses the same T1 task with a tracker-following fine-tuning runtime warm-started at the expert. |

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

## Phase A result (controller switching)

| cycle | source | arm / program | median speed MAE (m/s) | falls | median switches | result |
|---:|---|---|---:|---:|---:|---|
| 0 | predeclared | `single_fast` (`expert` only) | `2.0741721755` | `0/20` | `0` | Safe; ignores the slow third |
| 0 | predeclared | `single_slow` (`simple` only) | `3.3071158305` | `0/20` | `0` | Safe; ignores both fast thirds |
| 0 | predeclared | `playback` (`expert -> simple -> expert`) | `3.1803425853` | `20/20` | `2` | Fell after the first switch at step `300` |
| 0 | builder | `handwritten` (`expert -> medium`; guarded) | `3.2062160613` | `20/20` | `2` | Fell after the first switch; never reached `simple` |
| 1 | audit-log-only designer run | staged through `medium`, dwell and velocity gates, recovery | `3.2215072393` | `20/20` | `3` | Every scheduled switch at step `300` preceded a fall; never reached `simple` |
| 2 | audit-log-only designer run | `expert` hold; no transitions | `2.0741721755` | `0/20` | `0` | Behavior and outcome fields matched `single_fast` exactly on `20/20` seeds |

None of the three tested step-300 running-speed handovers survived; alternative
phases, timings, and state-conditioned handovers remain untested. The tested
designs—`playback`, `handwritten`, and the cycle-1 candidate—fell in all `60/60`
episodes. Cycle 2 therefore selected the no-switch expert hold. It removed
the falls but did not execute the requested slow segment, so it is not evidence
of composition quality or task completion.

### Correction recorded 2026-09-05 (SCI-001)

The cycle-2 designer was steered from an inaccurate summary. The cycle-0
`playback` arm switched `expert` to `simple` and fell `20`-`25` steps later;
the cycle-0 `handwritten` arm switched `expert` to `medium` and fell `23`-`60`
steps later; only the cycle-1 candidate has the `22`-`57` range. The historical
steering text remains verbatim in the cycle-2 record, with this correction
attached rather than rewriting the record.

The cycle-2 designer stated that the missing library element is an admitted
deceleration behavior with entry coverage at expert running speeds and a
validated safe handoff to `medium` or `simple`.

### Evidence-chain repair recorded 2026-09-05

The Phase A JSON reports remain schema v1 with unchanged bytes. Corrected
Markdown is deterministically rendered from those reports. The execution seal
is `execution_manifest_v1.json` (`cccaf040140571a1907e2285b5633824cf0888c4492c2cb062e7c0c178fd0c00`).
Each cycle now has a deterministic v2 scientific receipt with separate
telemetry; the receipt chain is `e1f6466a35428ca37a4053271f869db58b07d3f458109ec44b3ece767596d812`,
`5a5a996c28423dd6c33de3b33c527de43825aeedccb485338d94ad033882cecc`,
then `c6dab6cb6e5a9b524b947ef11aec2e7ae64d4641f4e16828c9fd75b43a4eda7f`.
These are interface and integrity receipts, not new simulator evidence.

### Claim ceiling

Phase A is exploratory. Controller switching stood in for a tracker following
composed references. The reports establish deterministic execution and the
observed failure of these three switching designs under the frozen protocol;
they support no oracle-quality, generalization, frozen-tracker,
reference-following, task-reward, naturalness, robustness, or
humanoid-competence claim.

### Phase B

The next block keeps task T1 and replaces the stand-in with a tracker-following
executor: a fine-tuning runtime warm-started at the expert. The oracle supplies
a composed reference so that the runtime learns transitions from that reference
instead of executing transitions by switching controllers. No cycle-3 prompt
is prepared.
