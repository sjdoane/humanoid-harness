# One-million-step TQC development screen v2

| | |
|---|---|
| progress | V2 predeclares one excluded TQC training attempt, strict persistence, protected evaluation, and same-rollout visual evidence. Nothing in v2 has run. |
| bottleneck | The trainer, parent supervisor, v2 actor-equivalence verifier, v2 evaluator integration, visual writer, execution manifest, and independent review receipt do not exist. |
| next step | Independently review the immutable design and E0 audit; then implement each required module without changing the design bytes. |

**Status:** predeclared, not executed

**Evidence class:** excluded-checkpoint development screen only

**Training authorization:** no

## Purpose

- Produce one persisted TQC base-controller candidate.
- Test one exact final actor on `20` predetermined resets.
- Capture visual evidence from `3` of those same live rollouts.
- Keep all training and evaluation seeds outside formal oracle and task-reward studies.

Immutable inputs:

- [`configs/tqc_base_controller_dev_1m_v2.study.json`](configs/tqc_base_controller_dev_1m_v2.study.json)
- [`configs/tqc_e0_reuse_dev_1m_v2.audit.json`](configs/tqc_e0_reuse_dev_1m_v2.audit.json)

## Fixed flow

```text
E0 resource receipt + full v2 workload projection + independent review
                                  |
                                  v
 parent supervisor (one attempt) --> one fresh worker --> plain Humanoid-v5 training
                                  |                         1,000,000 env steps
                                  v
                    final model + streamed replay + actor NPZ
                                  |
                   strict reload + byte-exact actor outputs
                                  |
                                  v
       20 protected evaluations on instrumented Humanoid-v5
                  |                         |
            direct metrics           3 same-rollout RGB streams
                  |                         |
                  +------------+------------+
                               v
                one immutable execution receipt
```

Legend: inputs are the E0 receipt, design, audit, and review; frozen components
are the MDP, environment, TQC, seeds, budget, and evaluator gates; outputs are
the receipt, model, replay, actor, traces, and WebPs; there is no within-attempt
feedback, and failure returns only to a new reviewed design.

Frozen components stay fixed. Only future implementation code and its reviewed
hashes enter the execution manifest.

## V1 boundary

| item | v2 statement |
|---|---|
| Scope | V2 supersedes v1 only for future repository-authorized executions after review. |
| V1 bytes | Retained unchanged as historical predeclaration. |
| Observed repository evidence | No reviewed v1 runner or v1 attempt receipt. |
| Universal claim | V2 does not claim that v1 was never run elsewhere. |

## Frozen training workload

| item | exact value |
|---|---:|
| Model seed | `95001` |
| Worker seeds | `95001`–`95005`, fixed order |
| Instrumentation-equivalence seed | `98001`; fixed `1,000 x 17` PCG64 action array |
| Training unit | `1` checkpoint from `1` training seed |
| Environment | Plain registered `Humanoid-v5`; `TimeLimit -> OrderEnforcing -> PassiveEnvChecker -> HumanoidEnv` |
| Fall handling | Unhealthy termination off |
| Vectorization | `DummyVecEnv`, `5` workers |
| Budget | `1,000,000` environment steps; `200,000` vector calls |
| Updates | `199,980` actor, critic, entropy, and target updates |
| Replay | `1,000,000` transitions; `5,648,000,000` array bytes |
| TQC | SB3/SB3-Contrib `2.9.0`; CPU; `[256,256]` ReLU actor/critics; `25 x 2` quantiles |
| Normalization | None for observation and reward |
| Selection | Exact final checkpoint only |
| Attempts | One; no retry, resume, early stop, or checkpoint substitution |

The JSON repeats every effective Humanoid, MuJoCo, TQC, TQCPolicy, Adam,
distribution, replay, logger, seed, and `learn()` value. It does not inherit
unspecified values from v1.

## E0 reuse audit

| E0 fact | observed value | v2 use |
|---|---:|---|
| Workload | `100,000` environment steps | Resource prior only |
| Rate | `619.6620593242255` steps/s | Projects `1,613.7828433300454 s`; not a guarantee |
| Peak RSS | `4,253,122,560` bytes | Compared with sampled `12 GiB` failure threshold |
| Replay allocation | `5,648,000,000` bytes | Exact v2 replay-array expectation |
| Persisted controller | None | No controller or behavior evidence |
| Rendering | Not measured | No render or codec evidence |

- The audit compares `37` mapped paths representing `36` shared concepts.
- The audit embeds the complete v2 training projection and its hash.
- The five scientific/workload deltas are seeds, horizon, persistence, protected
  evaluation, and same-rollout visuals.
- Two non-scientific additions freeze v2 defaults and strengthen supervision
  and integrity. They cannot alter mapped workload values.
- E0 recorded pinned source/runtime identities and the observed model-parameter
  structure, but not every resolved hidden default. The audit makes no exact
  hidden-default invariance claim.
- The audit does not bind the later v2 design hash or execution manifest.
- It does not authorize training.

## Monitoring and counters

| check | exact schedule | expected count |
|---|---|---:|
| Rollout finite | Every vector call `1..200000` | `200,000` |
| Parameter finite | Before learning; calls `1000,2000,...,200000`; after learning | `202` |
| Parameter state | Actor `8`; critic `12`; target critic `12`; entropy `1` | `33` tensors / `827,527` scalars |
| Optimizer-owned | Actor `8`; critic `12`; entropy `1` | `21` tensors / `495,701` scalars |
| Optimizer state | `step`, `exp_avg`, `exp_avg_sq` for all `21` owned tensors after first update | `63` tensors / `991,423` scalars |
| Training scalars | Callback calls `22..200000`, then post-`learn` | `199,979 + 1` |
| Disk gate | Preflight; every `100` vector calls; persistence/evaluation/final | `2,004` |
| Logger dump | `log_interval=1000` episodes | Monitor input only |

SB3 callback call `21` sees `105` environment steps and zero completed updates.
Call `22` first sees update `1`; the post-`learn()` check binds update `199,980`.
Every optimizer-owned parameter must end with Adam state fields `step`,
`exp_avg`, and `exp_avg_sq`, with step `199,980`.
Model construction must also observe empty initial optimizer states, float32
default dtype, deterministic algorithms disabled, and CPU single-tensor Adam
dispatch with `foreach=false` and `fused=false` for all three optimizers.

## Supervisor and persistence

- Reserve a new run directory at mode `0700` and an exclusive in-progress
  receipt at `0600` before importing the simulator.
- Finalize a preflight contract before worker start. The worker may perform only
  runtime reinspection and instrumentation equivalence, then must pause.
- That contract binds the interpreter, dependency lock, project tree, runtime
  inspector, worker entrypoint, probe implementation, design, audit, and review.
- The parent binds those receipts into the final execution manifest. The worker
  receives it over the bounded reverse pipe, re-verifies it, and acknowledges
  the same bytes before model construction.
- One parent supervisor owns exactly one fresh worker.
- The worker uses multiprocessing `spawn`, a dedicated process group, and a
  bounded length-prefixed canonical-JSON stage channel.
- Each pipe direction has its own contiguous sequence counter beginning at
  zero; indices are never shared or arbitrated across directions.
- The parent enforces monotonic deadlines for every named stage.
- The worker applies its `10,800 s` training wall gate before the parent's
  `11,100 s` training deadline, leaving `300 s` for classification and shutdown.
- On timeout the parent terminates the owned process group, waits `10 s`, kills
  if needed, joins for at most `30 s`, and verifies no worker remains.
- Before process-group identity is validated, signal cleanup targets the exact
  child PID; it never signals an unvalidated process group.
- Signal, exception, observed `MemoryError`, unknown process disappearance or
  `SIGKILL`, timeout, and threshold failure remain distinct receipt outcomes.
- Success requires the ordered states `preflight`, `model_constructed`,
  `training`, `training_complete`, `persistence`, `strict_reload`, `evaluation`,
  `visual_encoding`, and `finalized` exactly once.
- Stream the `5.648 GB` replay through an exclusive `0600` partial file while
  hashing and counting bytes; atomically promote only a complete artifact.
- Never hold more than one full replay buffer. Release the original model and
  buffer before strict reload; release the empty reload buffer before loading
  the bound persisted replay.
- Strict model reload uses `TQC.load(reader, env=None, device="cpu",
  print_system_info=False, force_reset=True)`.
- Strict replay reload uses `load_replay_buffer(reader,
  truncate_last_traj=False)`.
- Pre-save and reloaded policy, entropy, all three optimizer moment states and
  Adam parameter groups, all replay arrays, and replay metadata must have byte-
  exact paired hashes.

## Actor export

- Export the exact final actor only; never select by metric.
- Use the code-free, loader-issued strict NPZ capability.
- Compare trusted and reloaded actor outputs on the fixed `[4,348]` observation
  batch.
- Compare mean, clamped log standard deviation, deterministic action, and one
  seeded sample as C-order `<f4` arrays of shape `[4,17]`.
- Seeded sampling is
  `Normal(mean, exp(clamped_log_std)).rsample()` followed by `tanh`.
- Save and restore CPU RNG state around separate `manual_seed(97001)` calls for
  trusted and strict-loaded actors.
- All eight paired output hashes must match exactly.
- Trusted and strict-loaded whole-actor state hashes must also match exactly.
- The new v2 equivalence verifier is required but not implemented.

## Protected evaluation

| item | exact rule |
|---|---|
| Seeds | `96001`–`96020`, fixed order, one reset each |
| Horizon | Exactly `1,000` steps; expected final `terminated=false`, `truncated=true` |
| Environment | Instrumented `SubstepContactHumanoidEnv` under the registered wrapper stack |
| Actions | Deterministic actor output |
| State | Direct MuJoCo position, qpos-derived torso up-axis, control, and exact simulation time |
| Contacts | Ordered geom pairs and normal forces from all `5` physics substeps |
| Reward | Reward and `info["reward_*"]` fields never enter metrics |
| Trace | One `1,001`-boundary canonical trace, including every `[348]` controller observation |

Trace sample `0` has reset sentinels. At each sample `k=1..1000`, the stored
action was inferred from observation `k-1` and applied during the transition to
boundary `k`; observation `k` is the returned post-step controller input.

Before protected evaluation, a run-bound `1,000`-step canary must show exact
state, observation, clock, wrapper-counter, reward, and termination equivalence
between the plain training environment and the contact-instrumented evaluator.
The fixed random control sequence is interface evidence, not behavior evidence.

During protected evaluation, each of the `20` actor-driven trajectories runs in
both stacks from the same reset with byte-identical physical actions. Every
boundary must match exactly; only the instrumented copy supplies metrics and
visuals.

Behavior gate:

- `20/20` full-horizon healthy episodes.
- `20/20` full-horizon upright episodes.
- Median direct-elapsed-time forward velocity at least `0.5 m/s`.
- At least `18/20` episodes with net forward displacement at least `5 m`.

Passing supports one excluded-checkpoint development statement. It does not
support tracker admission, oracle quality, naturalness, robustness, or training-
seed uncertainty.

## Same-rollout visual contract

| item | exact rule |
|---|---|
| Seeds | `96001`, `96010`, `96020` |
| Source | Same live evaluation environment and loop; no replay, rerun, or replacement |
| Boundaries | `0,2,...,1000` |
| Frames | `501`; RGB `uint8`; C-order; `480 x 480 x 3` |
| Renderer | Direct `mujoco.Renderer`; same model/data; `max_geom=10000`; env `render_mode=None` |
| Camera | Free; look at `[root_x, root_y, 1.2]`; distance `4`; azimuth `90`; elevation `-20` |
| Noninterference | Exact pre/post integration state, observation, qacc, actuator force, clock, wrapper counter, contacts, and actor binding |
| Scene options | Exact default `mujoco.MjvOption` record and semantic hash |
| Playback | Lossless WebP; `30 ms` each; `15.03 s`; loop `0`; no additional terminal hold |
| Codec | Pillow `12.3.0`; libwebp `1.6.0`; explicit lossless settings and empty metadata |
| Decode gate | Load every sought frame before reading duration; `501 x 30 ms`; exact raw-pixel equality |

Capture follows each live trace boundary. A next action is inferred only for
boundaries below `1000`; the terminal boundary has no next inference.

Array hashes bind the canonical dtype/shape header length, header, raw byte
length, and C-order bytes. Ordered camera and noninterference records use
canonical JSON. Collapsed identical WebP frames fail the artifact.
That codec constraint makes the visual artifact ineligible, not the behavior
gate; it cannot trigger a rerun.
The final boundary receives the same ordinary `30 ms` duration as every other
frame; this accounts for playback being `30 ms` longer than the simulated span.

## Execution gate

**GO:** immutable v2 design and E0 audit may enter independent review after all
tests pass.

**NO-GO:** training remains blocked until all of these exist and are hash-bound:

- parent supervisor and resource monitor;
- training/persistence adapter;
- v2 actor-equivalence verifier;
- protected evaluation integration;
- visual capture, encoder, and decoder verifier;
- current host/runtime receipt;
- complete execution manifest; and
- independent review receipt.

The future review receipt must bind the final design, audit, contract source,
and exact v2 contract test-file bytes; report a read-only independent reviewer,
`GO`, all findings resolved,
and zero unresolved P0/P1/P2 findings. The preflight contract and execution
manifest then bind its content hash. This is a reviewer attestation plus local
file identity, not a cryptographic signature. The design contains no future
receipt placeholder.

Experimental-design guidance shaped the fixed seed roles, single training unit,
single-checkpoint rule, predetermined visuals, and explicit evidence ceiling.
