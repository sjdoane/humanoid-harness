# Family B target-speed v1 protocol

| status | current truth |
|---|---|
| progress | B0FIX4 makes the `network` and `tracing` canaries exercise Seatbelt-governed operations. Profile `reward-worker-seatbelt/v3` and its SHA-256 are unchanged. |
| bottleneck | The builder sandbox denies nested Seatbelt bootstrap, so the 14 OS canaries remain `not_verified_in_builder_sandbox`; the current host receipt is the pre-B0FIX4 `18/20` diagnostic. |
| next step | Fable reruns all 14 OS canaries against unchanged profile v3 on the host and requires all 20 canaries to pass, with `memory_abuse` naming `timeout_kill`, before locking this proposed protocol or authorizing any calibration or training. |

**Status:** proposed, not locked  
**Family:** `family-b-target-speed-v1`  
**Evidence label:** `exploratory_reward_cycle`  
**Compute reservation:** `150 min` CPU for a later matched cycle  
**B0 authorization:** no training and no behavioral evaluation

## Single experimental factor

| component | frozen value |
|---|---|
| authorable field | Immutable source bytes defining `task_term(x)` |
| candidate read set | Exactly `com_x_velocity_m_s` and `target_speed_m_s`, both finite float64 scalars in `m/s` |
| targets | Separate fixed-target policies for `0.5`, `1.0`, and `1.5 m/s`; cycle 1 uses only `1.0 m/s` |
| MDP | Stock Gymnasium `Humanoid-v5`: XML, observations, actions, dynamics, reset, termination, and 1,000-step `TimeLimit` |
| counterpart artifacts | `null_oracle/v1`, `none/v1` reference, and `none/v1` tracker |
| trainer | Frozen PPO recipe and predetermined seed/checkpoint schedule below |
| protected measurement | Endpoint and guardrails in this protocol; no authored output is an objective input |

Changing anything except candidate source bytes creates another experiment
family. Widening the two-float read set requires reviewed version `v2`.

## Reward contract

At each `0.015 s` post-step boundary, the trusted compositor computes

\[
r = g + 5\mathbf 1[1<qpos_2<2]
    -0.1\sum_i ctrl_i^2
    -\min(10,5\times10^{-7}\sum_{b=1}^{14}\sum_{d=1}^{6}cfrc_{b,d}^2).
\]

- COM velocity is the body-mass-weighted displacement of direct MuJoCo
  `xipos`, divided by `0.015 s`.
- Health uses strict inequalities.
- Control uses post-step `data.ctrl`.
- Contact uses the complete float64 `(14, 6)` `data.cfrc_ext`, including the
  world row.
- Float64 summation, clipping, grouping, and subtraction order match installed
  Gymnasium `1.3.0`.
- The result contains exactly four signed finite terms: `task_progress`,
  `healthy`, `control`, and `contact`. They reproduce the total within `1e-12`.
- `|task_progress| <= 1000` and `|total| <= 1024`. A breach aborts the seed; it
  is never clipped, repaired, or replaced.

Cycle 1 `stock_r0` uses `g = 1.25 v`. This target-unaware baseline has trivial
headroom; cycle 1 tests loop mechanics, not a substantive speed-shaping claim.

The frozen `manual_target_speed_v1` baseline is

\[
g_{manual}=1.25\left(1-\min(1,|v-v^*|/v^*)\right).
\]

It is not trained in B0 and joins `stock_r0` and `candidate_rk` from cycle 2.
Its exact frozen formula uses trusted affine `(1, 0)`; candidate moment matching
does not redefine either baseline.

## Candidate admission

The static gate accepts strict UTF-8 of at most `16 KiB` and `512` AST nodes:
exactly one non-async `task_term(x)` plus literal metadata; numeric literals,
plain local assignments, returns, arithmetic, comparisons, Boolean expressions,
and conditionals of depth at most eight. Only the declared attributes and the
injected `abs`, `min`, `max`, `exp`, `sqrt`, and `clip` functions exist.

It rejects imports, dynamic calls or strings, private/dunder names, unknown
attributes, classes, lambdas, decorators, defaults, loops, comprehensions,
recursion, generators, async, exceptions, context managers, subscripting,
mutation, `global`, `nonlocal`, reflection, mutable state, wrong output types,
non-finite output, and envelope breaches. Repeated and interleaved probes must
be bitwise identical.

The target-speed axiom gate requires, on the frozen grid
`v = 1.0 + 0.25 j`, `j=-16..16`:

- a unique maximum at `v=1.0 m/s`;
- non-increasing values with increasing absolute target error on both sides;
- centered responses at each target in `{0.5, 1.0, 1.5}` greater than probes
  `0.25 m/s` below and above; and
- identical frozen-grid determinism digests.

For candidate `h`, the trusted scale is

\[
\alpha=2.9755951785595207 / sd(h),\qquad
\beta=1.25-\alpha mean(h),\qquad g=\alpha h+\beta.
\]

The stock grid population mean and SD are exactly `1.25` and
`2.9755951785595207`. Reject zero/non-finite raw SD, `alpha` outside
`[0.25, 4]`, or `|beta| > 10`. Coefficients are computed and hashed, never
authored.

## Process boundary

- One fresh persistent worker per training seed; each call contains exactly
  four environments.
- Parent snapshots and hashes one regular non-symlink source file, validates
  it, sends those bytes over a pipe, and verifies source bytes and file identity
  before and after use.
- Worker namespace is `__builtins__={}` plus the six numeric functions and the
  immutable two-float input record.
- IPC is length-framed fixed-schema binary over inherited pipes. Requests are
  at most `1 KiB`; responses are at most `64 B`; only finite numeric scalars are
  admitted. Pickle, object dtype, stdout, and arbitrary component names are not
  protocol channels.
- Worker cwd is a new empty private directory. Environment keys are exactly
  `PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, and `PYTHONHASHSEED=0`;
  no `HOME`, `PATH`, or inherited secret is passed.
- `/usr/bin/sandbox-exec` applies deny-default profile
  `reward-worker-seatbelt/v3`. The worker is executed through the unresolved
  repository path `.venv/bin/python`; it is never launched through the
  symlink's resolved uv-managed base interpreter. The profile grants
  `process-exec` to both paths and `file-read*` to the venv symlink itself.
  Receipts bind each absolute path and each executable-file SHA-256.
- Profile v3 retains every v2 rule unchanged and adds exactly three
  build-time-derived literal directory data reads:
  - `<repo>/src`, so Python can list the isolated environment's project source
    entry and discover the regular `oracle_composition` package;
  - `<repo>/src/oracle_composition`, so Python can list that regular package
    and discover its `rewards` subpackage; and
  - `<repo>/src/oracle_composition/rewards`, so Python can list the subpackage
    and discover the worker module and its imported reward modules.
  These are directory-listing literals, not recursive `subpath` grants.
  Profile v2's unrestricted path-metadata reads and single literal root
  directory data read remain unchanged. Other data reads remain limited to
  the interpreter/library trees, the venv, the three package directories, and
  exact allowlisted worker sources; file mutation, other repository data,
  network, process creation/fork, signals, and process inspection remain
  denied.
- The network canary creates a fresh IPv4 stream socket for each governed
  operation. It attempts `connect(("127.0.0.1", 9))` and, separately,
  `bind(("127.0.0.1", 0))`. Containment requires both operations to raise
  `PermissionError` or `OSError` with `EPERM`; connection refusal, successful
  bind, or any other result is not containment. Socket creation is setup and
  never contributes to the verdict because Seatbelt does not govern it as a
  `network*` operation.
- The tracing canary calls macOS `libproc` through `ctypes` and requests
  `PROC_PIDTBSDINFO` for the parent PID with `proc_pidinfo`. Containment
  requires zero returned bytes and `errno == EPERM`. The former `getpriority`
  probe was invalid because that syscall is not governed by
  `(deny process-info*)`.
- After exec, the worker attempts `RLIMIT_AS = 1 GiB`. Failure is fatal except
  when `sys.platform == "darwin"` and `setrlimit` raises `ValueError` or
  `OSError`; that exact case keeps the inherited limit and records
  `address_space_limit = {requested_bytes: 1073741824, enforced: false,
  reason: darwin_refuses_finite_rlimit_as}`.
- Cumulative CPU `900 s`, file size `0`, core size `0`, and open files `16`
  remain mandatory on every platform. Any failure applying one is categorical
  and fatal. Startup is `5 s`; each four-item batch is `50 ms`.
- The memory-abuse canary allocates without bound. It reports `rlimit_as` only
  when an enforced finite limit causes the worker to stop. With Darwin's
  recorded non-enforcement, it reports `timeout_kill` only after the parent's
  fixed `50 ms` deadline and observed process-group `SIGKILL`; that path
  invalidates the seed and has no fallback.
- Any failed canary, timeout, crash, malformed/extra frame, non-finite or
  out-of-envelope scalar, source drift, or limit failure invalidates the seed
  and arm without fallback.
- Failed OS-canary receipts distinguish a normal worker exit code from a
  terminating signal and retain at most the first sanitized `200` bytes of
  stderr. The frozen exit-code ledger states: `0` clean/operation denied; `2`
  command-line parsing failed; `4` canary operation not denied; `5` invalid
  canary invocation; `64` a required invocation file descriptor is missing;
  `65` environment sanitization failed; `66` a declared resource limit failed;
  `67` unexpected internal limit setup; `70` source bootstrap/static validation
  failed; `71` request/evaluation failed; and `86` the intentional crash
  canary. Exit `64` is an invocation error, not evidence that Seatbelt denied a
  hostile operation.

OS/runtime canaries deliberately bypass the AST gate. Production workers may
start only from a receipt where every canary verdict is `passed`.

### Canary receipts

| receipt | canonical SHA-256 | established fact |
|---|---|---|
| [`receipts/builder_sandbox_canaries.json`](receipts/builder_sandbox_canaries.json) | `3cfc64c0868c606cce9b400cf88493f2b9057d7c7dfe668cfd016b6114bf6238` | Profile `reward-worker-seatbelt/v3`: six protocol/identity self-canaries pass; all 14 OS canaries are `not_verified_in_builder_sandbox` with exact error `sandbox-exec: sandbox_apply: Operation not permitted`. It binds profile SHA-256 `f18216e5c08a0ac2eac8986f8338b032c84d2de24a231083ae28083bb061f44f`, launch/resolved interpreter paths and hashes, and the categorical exit-code ledger. The trusted limit probe records Darwin's refused finite `RLIMIT_AS`; this is not an OS-canary pass. |
| [`receipts/host_sandbox_canaries.json`](receipts/host_sandbox_canaries.json) | `097006bf8becd4a6b4bed5ffb4dbe56174516e9a878a76a71827ac3d34bf7a97` | Pre-B0FIX4 profile-v3 host diagnostic: `18/20` canaries pass, including `memory_abuse` through `timeout_kill`. Only `network` and `tracing` fail with exit `4` because their former socket-creation and `getpriority` probes were not governed by the profile. It does not verify the repaired probes. |

Fable's post-B0FIX4 host rerun must show all 14 OS canaries and all six
self-canaries as `passed`;
`memory_abuse` must name `timeout_kill` on Darwin. A `failed` verdict is never
accepted, and a builder-sandbox nonverification is never promoted to a pass.

## Protected endpoint

At every boundary, the protected evaluator independently recomputes

\[
v_t=(x^{COM}_t-x^{COM}_{t-1})/0.015,
\quad
p_t=\mathbf1[t\le N_e]\exp\left[-\tfrac12((v_t-v^*)/0.25)^2\right].
\]

The sole episode endpoint is

\[
P_e=\frac1{800}\sum_{t=201}^{1000}p_t,\qquad 0\le P_e\le1.
\]

Missing post-termination values are exactly zero. For each training seed,
`mu[a,s]` is the mean of all 20 reset-seed endpoints. Report each of the five
paired `mu[candidate,s] - mu[stock,s]` differences and their mean. Their sign
does not establish improvement.

The evaluator receives direct simulator state, normalized executed action,
termination flags, and all-substep contact captures. It has no reward, reward
component, candidate output, training-return, or diagnostic-telemetry input.

## Hard guardrails

| guardrail | per-episode quantity | proposed cycle rule |
|---|---|---|
| survival | 1,000 steps, no stock termination, height in closed `[1,2] m`, torso-up `>=0.5` | At least `19/20` episodes in at least `4/5` checkpoints; candidate count cannot be below paired `stock_r0` |
| structural action | `max |u| <= 1`; `max |(u_t-u_(t-1))/0.015| <= 2/0.015`, with `u_0=0` | Every observed step; any breach invalidates the receipt |
| structural torque | `|tau_j|/capacity_j <= 1+1e-9`; normalized torque equals executed normalized action within `1e-7` | Every observed step |
| torque exposure | `sqrt(sum tau^2 / (17 N_e)) N*m` | Relative rule below |
| energy | `0.015 sum |tau qdot| J`; also divide by `0.015 N_e` for W | Relative rule below |
| action rate | `sqrt(sum ((u_t-u_(t-1))/0.015)^2 / (17 N_e)) s^-1` | Relative rule below |
| non-foot floor contact | Count every floor contact from all five `0.003 s` substeps unless the other geom is `left_foot` or `right_foot` | Zero per episode; at least `19/20` episodes in `4/5` checkpoints; candidate count cannot be below paired `stock_r0` |
| allowed-foot impact | Peak force, peak impulse, and total impulse from every captured substep | Descriptive only; no naturalness threshold |

Torque capacities in actuator/action order are
`[40,40,40,40,40,120,80,40,40,120,80,10,10,10,10,10,10] N*m`.
The corresponding generalized-velocity/force indices are
`[7,6,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]`; the first two differ
from contiguous DoF order in the installed XML.

For torque exposure, energy, and action rate, a premature episode is `+inf` for
the gate while its partial measured value remains in the raw record. For metric
`m`, use the median `Q[m,a,s]` over 20 episodes and the relative ratio specified
in `DECISION_RULE.md`. The `10%` margin is exploratory, not a validated
naturalness threshold.

## Frozen later-run recipe

| field | value |
|---|---|
| runtime | SB3 `2.9.0`, Gymnasium `1.3.0`, MuJoCo `3.12.0`, NumPy `2.5.2`, Torch `2.14.0`, CPU |
| policy | `local.SquashedGaussianActorCriticPolicy/tanh_jacobian/v1`; one affine map from normalized `[-1,1]` to physical `[-0.4,0.4]` |
| seeds | training `101,202,303,404,505`; evaluation resets `11001..11020` for every final checkpoint |
| steps | `1,048,576` environment steps per arm/seed |
| rollout | `4` `DummyVecEnv` environments; `2,048` steps each; `8,192` transitions; exactly `128` rollouts |
| PPO | batch `512`; `16` minibatches/epoch; `10` epochs; learning rate `3e-4`; gamma `.99`; GAE lambda `.95`; clip `.2`; no value clip; entropy `0`; value coefficient `.5`; max gradient norm `.5`; normalized advantages |
| exploration | gSDE off; frequency `-1`; initial log SD `0`; full SD; `use_expln=false` |
| network | separate `[256,256]` actor/critic, Tanh, orthogonal init, shared `FlattenExtractor`, Adam epsilon `1e-5` |
| normalization | no observation or return normalization |
| learn/checkpoint | `reset_num_timesteps=true`, no progress bar, mandatory pre-update likelihood audit on all 128 rollouts; fresh initialization and final-timestep checkpoint only |
| evaluation | deterministic actions; all 20 reset seeds; no best-seed, episode, checkpoint, retry, resume, or substitution |

The matched cycle reservation is `150 min` CPU. A reward-enabled calibration
that projects beyond it stops before launch.

## Ten-step matched baseline plan

1. Freeze one execution manifest with endpoint, guardrails, target, trainer,
   seeds, budget, and exact runtime/source identities.
2. Train five fresh `stock_r0` policies, final checkpoint only.
3. Evaluate every final checkpoint on all 20 declared reset seeds.
4. Reopen and rehash all checkpoint, trace, evaluator, and manifest artifacts.
5. Apply the baseline viability stop rule.
6. Give the complete development dossier to the design agent; then lock
   candidate source, static receipt, axiom receipt, and affine receipt.
7. Train the candidate from fresh matched initializations and identical budget.
8. Evaluate through the identical protected endpoint and reset seeds.
9. Preserve every failed or missing seed; never retry or substitute.
10. Reuse `stock_r0` later only when every other execution identity is
    byte-identical.

None of these ten execution steps is authorized by B0.

## Runtime and evidence boundary

The runtime fingerprint binds package versions, Python/platform observations,
effective environment kwargs, state and space shapes/dtypes, both complete Box
spaces, timestep/frame skip, XML, installed Gym source, dependency lock, full
source tree, actuator gear/capacities, contact-capture implementation, reward
contract/compositor/validator/scale/sandbox sources, protected evaluator,
policy, trainer, and the no-learning equivalence receipt.

The refreshed
[`builder_runtime_no_learning_smoke.json`](receipts/builder_runtime_no_learning_smoke.json)
has canonical SHA-256
`69a20c2da8185f6c435be3249f887e557b997255eef5c4a570e345ce0d3d6ce3`.
It binds the v3 sandbox source while retaining the unchanged eight-step
equivalence digest `048cf64ca8b06f6bd9f00c1b591ba1d0b40fb87620114db6f3a3c367f4218bdc`;
it is an interface check, not behavioral evidence.

The execution manifest additionally binds the proposed config, this protocol,
the decision rule, every candidate source, reward artifact identity, and the
complete training/evaluation seed schedule. It grants no automatic launch or
claim authority.

B0 tests software contracts and fixed-control parity only. It supplies no
learned policy trace, endpoint measurement, guardrail result, or evidence of
humanoid competence or reward improvement.
