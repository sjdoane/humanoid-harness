# One-million-step TQC development screen

| | |
|---|---|
| progress | E0 passed; a single excluded-seed TQC run, final-checkpoint rule, fixed evaluation set, and code-free actor format are frozen. |
| bottleneck | No reviewed training adapter or protected locomotion evaluator is bound in an execution manifest. No trained controller exists. |
| next step | Implement and review those two adapters, freeze their hashes, then launch this design once. |

**Status:** predeclared, not executed

**Evidence class:** development only

**Automatic 20M authorization:** no

## Purpose

- Obtain the first persisted TQC base-controller candidate.
- Test whether one exact final actor shows healthy forward locomotion.
- Exercise safe actor export before considering a longer run.
- Keep all training and evaluation seeds out of formal oracle and reward studies.

The design is
[`configs/tqc_base_controller_dev_1m_v0.study.json`](configs/tqc_base_controller_dev_1m_v0.study.json).

## Fixed flow

```text
E0 receipt 2436c2e9... + frozen design + reviewed execution manifest
                                |
                                v
      Humanoid-v5 -- 1,000,000 steps --> exact final TQC checkpoint
                                               |
                       +-----------------------+---------------------+
                       |                                             |
                       v                                             v
       trusted local continuation state                  strict actor-only NPZ
       SB3 zip + replay pickle                            no code/pickle/critic
       never ingest from uploads                          output-equivalence check
                       |                                             |
                       +----------------------+----------------------+
                                              v
                          20 predetermined deterministic evaluations
                          3 predetermined full-horizon videos
                                              |
                                              v
                          protected gate + exact claim ceiling
```

## Frozen run

| item | value |
|---|---:|
| Model seed | `95001` |
| DummyVecEnv worker seeds | `95001`–`95005` |
| Training budget | `1,000,000` environment steps |
| Vector calls / updates | `200,000` / `199,980` |
| Replay capacity | `1,000,000` transitions; `5,648,000,000` allocated bytes |
| Training environment | `Humanoid-v5`; unhealthy termination off; no normalization |
| Checkpoint selection | Exact final step only |
| Attempts | One; no automatic retry or metric-based early stop |
| Evaluation seeds | `96001`–`96020`; one episode each; fixed order |
| Video seeds | `96001`, `96010`, `96020`; chosen before training |
| E0 observed rate | `619.6620593242255` environment steps/s |
| E0-rate projection | `1,613.7828433300454 s` (`26.90 min`) |
| Throughput floor projection | `8,620.69 s` (`2.39 h`) |
| Sampled run-time failure threshold | `10,800 s` (`3 h`) |

All `25` development streams are permanently excluded from formal oracle and
task-reward studies. The `20` evaluation resets are repeated measurements of
one checkpoint. They are not `n=20` independent training replicates.

Every exposed TQC, policy, replay-buffer, feature-extractor, optimizer,
exploration, logging, and update parameter is explicit in the design. The
future execution receipt must report the effective values observed on the
constructed model; matching library versions alone is insufficient.

## Protected evaluation

| gate | exact rule | role |
|---|---|---|
| Integrity | Exact budget/update count; finite training/evaluation; exact final checkpoint; strict actor reload and output equivalence; all seeds observed once in order | Hard |
| Healthy | All `20/20` episodes remain in root-height interval `(1.0, 2.0) m` for `1,000/1,000` steps | Hard for development behavior claim |
| Upright | All `20/20` episodes keep torso up-axis `z >= 0.5` for the full horizon | Hard for development behavior claim |
| Forward | Median time-average forward velocity `>= 0.5 m/s`; at least `18/20` episodes move forward `>= 5 m` | Hard for development behavior claim |
| Contacts/actions | Non-foot floor-contact step fraction, action RMS, and saturation fraction | Descriptive; no uncalibrated naturalness threshold |

Metric implementation requirements:

- Read root position, torso orientation, control, and contacts directly from
  MuJoCo.
- Compute mean forward velocity as net root-`x` displacement divided by exact
  elapsed simulation time.
- Ignore scalar reward and every `info["reward_*"]` field when grading.
- Keep unhealthy termination off so a fall remains measured rather than hidden
  by an early reset.
- Emit one complete trace for every evaluation seed.

Passing supports only this statement:

> The exact final actor from one excluded training seed remained healthy and
> upright on all 20 predetermined resets, reached the declared median forward-
> velocity threshold, and moved at least 5 m on at least 18 resets in the
> pinned local runtime.

It does not establish tracker admission, reference use, oracle quality,
naturalness, robustness outside those resets, or training-seed uncertainty.

## Persistence boundary

| artifact | contents | allowed use | prohibited use |
|---|---|---|---|
| Trusted local model zip | SB3 data plus Torch state | Same-run local continuation and forensic inspection | Upload ingestion or public controller format |
| Trusted local replay pickle | Full off-policy replay | Same-run local continuation | Upload ingestion or distribution |
| Strict actor NPZ | Eight little-endian float32 actor tensors, action bounds, format version | Data-only deterministic/stochastic actor reconstruction | Exact optimizer resume or claims about critic/replay state |
| Actor manifest JSON | Source checkpoint, design, runtime, schema, content, state, and paired trusted/reloaded hashes for mean, log standard deviation, deterministic action, and seeded sample | Provenance | Replacing missing actor bytes |

- Actor equivalence means exact shape, dtype, and C-order bytes for every paired output.
- A tolerance-based comparison does not satisfy the gate.

The trusted pair supports operational continuation. It does not preserve or
prove a bitwise-equivalent resume because environment and process RNG state are
outside the pair.

The actor archive:

- is at most `2 MiB`;
- has exactly `11` ordered NPY 1.0 members;
- permits only C-order little-endian float32/int64 arrays;
- rejects pickle-backed dtype, unknown keys, path-capable members, symlinks,
  archive comments, trailing bytes, and content-hash mismatch;
- contains no critic, optimizer, replay, RNG state, Torch archive, or executable
  object; and
- must reproduce mean, `log_std`, deterministic action, and one seeded sample
  on a separately pinned observation batch before evaluation.

## Stop rules

- Missing or changed E0 receipt: do not construct the model.
- Runtime, source, host, thread-count, or dependency-lock drift: do not train.
- Missing execution-manifest field: do not train.
- Resource threshold, non-finite value, wrong step/update count, persistence
  failure, or export mismatch: mark the attempt failed.
- Never substitute an earlier checkpoint, better episode, or successful video.
- Gate failure: retain the failed receipt and diagnose before proposing 20M.
- Gate success: run actual E1 initialization identity; do not admit a tracker.

## Remaining launch blockers

- Implement the training adapter without changing this design.
- Implement the reward-independent locomotion evaluator and trace writer.
- Bind both source hashes and the project source-tree hash in an immutable
  execution manifest.
- Select a project and trained-weight license before publishing actor bytes.
  The repository currently has no license; this design does not choose one.

## Method sources

- [Farama pinned Humanoid TQC runner](https://github.com/Farama-Foundation/minari-dataset-generation-scripts/blob/e74f9d0524c6df014c5a9985d0804001b9ce40dc/scripts/mujoco/train.py)
- [SB3-Contrib TQC actor `v2.9.0`](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/blob/v2.9.0/sb3_contrib/tqc/policies.py)
- [SB3 model save/load contract](https://stable-baselines3.readthedocs.io/en/v2.9.0/guide/save_format.html)
- [TQC paper](https://arxiv.org/abs/2005.04269v1)

The Stable-Baselines3 skill shaped the persistence split and actor-output
checks. Experimental-design guidance shaped the fixed reset/video schedule,
single-checkpoint rule, and explicit `n=1` training-unit boundary.
