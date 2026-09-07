# Study020: saved-policy sampling diagnostic

| progress | Study019 is rejected; Study018 retains exact initial/final policy tensors. |
|---|---|
| bottleneck | Deterministic degradation does not reveal the sampled checkpoint distributions. |
| next step | Reproduce both deterministic traces, then evaluate16 predeclared paired noise sequences. |

**Predata draft.** Implementation review, exact source seal and native resource
approval remain required. This file alone does not authorize a simulator run.

## Question and boundary

- For the exact018 initial/final checkpoints and fixed reset, how do sampled
  episode return, falls and task outcomes differ? Does the deterministic
  degradation persist under sampled control?
- Evaluation-only diagnostic, not a new oracle/reward comparison or a training
  experiment. Zero gradient/optimizer steps; no checkpoint selection.
- Freeze018 task, O7b oracle, reward, native four-state finite-horizon runtime,
  reset seed20260906, tracker, normalizer, action processing and20 s horizon.
- Load each checkpoint's own complete numeric state, including all23 log-std
  values and fixed normalizer buffers. Never replace final log-std with initial.
- Repeated episodes sample action noise around two fixed policies from **one
  training seed**. They are not16 independent training replications, held-out
  tasks, resets, motion datasets or a safety/generalization study.

```text
INPUT: pinned018 config + initial/final tensors + shared noise sequence
                         ↓
     frozen policy mean/std → training-equivalent clip → frozen G1 course
                         ↓
OUTPUT: raw states/actions/rewards + independent task gates + termination
                         ↓
      paired checkpoint diagnosis → next separately declared experiment
Frozen: oracle, task reward, tracker, normalizer, MDP/reset, evaluation gates.
No training, automatic adoption or altered018/019 verdict.
```

## Inputs

Retained run: sibling
`humanoid-harness-probe-runs/gmt_course_study018_baseline_20260907`.

| Input | SHA-256 |
|---|---|
| Manifest | `33b5e25ea8dfa7a8e3a4f520e7386edce067b920d75201e8ffb81581bd96f705` |
| Resource receipt | `c06fbc88be3098171019d639ddab2ce3e3f7b378fa0e3da43cbea81e7dbd2d0e` |
| Exact course config | `a9e52ba706e4ddfd394c757cf525bf210d553906430c7f08e11e0495affdf289` |
| Initial numeric policy | `afc5f1f20e0a616e399d705676dafff4ccd1699bb0b5f59e6e36a220284802ae` |
| Final numeric policy | `01421e9af87edde042048574b995e1b553fd4ec04bd7e46d546c5586444d4729` |

- SB3 2.9.0, Torch2.14.0, NumPy2.5.2 observed locally before data. Seal actual
  implementation/version identities. Numeric policies are not optimizer resumes.
- Verify retained lineage and bounded NPZ structure before strict tensor loading.
  No arbitrary policy code, pickle, reconstructed weights or online normalization.

## Evaluation sequence

1. Lock reviewed implementation, source tree, protocol, sampling schedule and
   input hashes. Use a fresh, exclusive native-supervised output directory.
2. Noise-disabled initial policy must reproduce018's retained zero-residual
   frames, trajectory and evaluation bytes. Noise-disabled final policy must
   reproduce its three retained final-policy outputs byte-for-byte.
3. Verify both parity receipts **before any sampled episode**. On failure,
   stop; diagnose the evaluation path without drawing behavioral conclusions.
4. Run all16 noise pairs in the fixed order below. Reset the entire environment,
   oracle and controller history for each episode. No outcome-selected repeats.
5. Retain every episode, including early falls; publish completion or explicit
   incomplete-run status. A missing episode is not a zero or a success.

| Noise seeds, ascending within each group | First policy |
|---|---|
|20260920,20260921,20260924,20260925,20260926,20260927,20260932,20260934 |Initial |
|20260922,20260923,20260928,20260929,20260930,20260931,20260933,20260935 |Final |

- Execute pairs in seed order20260920…20260935; run the other policy second.
  Balanced first-policy order uses Python3.13 `Random(20260907).shuffle` on
  `['initial','final'] * 8`; the explicit table is authoritative.
- Each seed generates1000 independent23-D float32 standard-normal rows on CPU,
  using an isolated Torch generator. Bind the exact generation algorithm and
  retained noise bytes before sampled execution. Both policies use the same row
  at the same control index; early termination does not shift another episode.
- Sample from the **unclipped** policy mean and its own standard deviation,
  then use the exact training action clipping/scaling path. Adding noise to
  already-clipped deterministic actions is invalid. No stage-wise noise reset.
- Test this sampling path against the installed SB3 Gaussian distribution and
  Box-action processing, including clipping tails and noise-disabled parity.

## Endpoints and interpretation

- Joint primary diagnostics: per-episode raw total reward sum and fall indicator;
  report all16 paired final-minus-initial returns and the four paired fall cells.
  Include raw task/tracking sums, duration and termination cause.
- Secondary: all11 unchanged task gates, full-horizon pass count, mode coverage,
  region compliance, speed, drift, tracking, residual RMS and clipping fraction.
- Handover table: actual transition indices and available pre-action height,
  heading, forward speed, lateral position, plus subsequent duration/fall mode.
  Any comparison to reference velocity must name its coordinate frame; no
  subtraction of world-frame motion from a local reference component.
- Report paired mean/median/range, not a single best episode.16 pairs is a fixed
  resource cap for diagnosis, **not a powered confirmatory sample-size claim**.
  Samples within episodes are correlated; no sample-level significance tests.
- Shorter failed episodes censor drift and later-mode exposure. Keep duration
  and failure alongside scores; do not label small failed-run maxima improvements.
- Initial sampled failures motivate an exploration rival for this exact setup.
  A worse sampled final policy motivates training/reward rivals. Better sampled
  survival with worse mean-policy metrics indicates a tradeoff, not an invalid
  deterministic evaluator. Handover/fall associations do not isolate a cause.
- No automatic follow-on if inconclusive, no retroactive018 admission, and no
  threshold or checkpoint-selection change. Choose the next single-factor test
  from the complete evidence; the two-knob composition demonstration remains the goal.

## Resource and review

- One heavy job; no training. Proposed cap:1200 s CPU/wall,8 GiB RSS,1 GiB
  output,20 GiB free disk. Exact native approval binds the completed plan.
- Reuse existing environment, policy constructor, trace writer and independent
  evaluator. Keep diagnostic configuration separate from training configuration;
  an input config saying `train` must not cause a training launch here.
- Freeze source from first parity through completed scoring. Independent review
  checks retained tensors/noise, action path, lineage, all pairs and negative cases.

## Procedural sources

- [SB3 2.9.0 rollout implementation](https://stable-baselines3.readthedocs.io/en/v2.9.0/_modules/stable_baselines3/common/on_policy_algorithm.html)
  and [policy implementation](https://stable-baselines3.readthedocs.io/en/v2.9.0/_modules/stable_baselines3/common/policies.html):
  sampling and postprocessing guidance, checked against the installed source.
- Experimental-design and stable-baselines3 skills informed pairing, replication
  limits and separate evaluation. Tooling credit: Kassis, T., Agarwal, V., He, Y.,
  Patel, D., and Brueckner, A. M. (2026).
  [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065), currentv2
  checked2026-09-07. This is not evidence of improved robot performance.
