# Study 020: sampled evaluation does not rescue the trained policy

| progress | Both retained checkpoints evaluated on all 16 paired action-noise sequences; independent scoring agrees exactly. |
|---|---|
| bottleneck | Falls increase from 5/16 to 11/16. Neither checkpoint passes the full task. |
| next step | Keep Study 018 B stopped. Declare a separate oracle-exit probe before more training. |

Local development report; AI-assisted draft. Numeric and artifact checks are
complete; accountable human scientific review remains pending. The source/claim
ledger is deliberately unverified for manuscript use. Numeric table and receipt
lines are not individually manuscript-tagged. No submission readiness is claimed.

## Fixed comparison

Evaluation only: two saved checkpoints, one training seed, one reset, 16 paired action-noise sequences, and no training or checkpoint selection. Each checkpoint retains its own learned action standard deviations. [claim:C001] [evidence:E001,E002,E003]

- Both deterministic episodes reproduce their retained Study 018 frame,
  trajectory and evaluation bytes before any sampled episode starts.
- Noise seeds: `20260920` through `20260935`; fixed execution order. The same
  noise vector is supplied at the same episode step, not the same robot state
  after trajectories diverge.
- MDP, task, reset, oracle, reward, controller, normalizer and evaluator remain
  fixed. The diagnostic evaluates each checkpoint's own distribution; it does
  not isolate policy-mean changes from learned standard-deviation changes.

## Measured result

The trained checkpoint falls in 11/16 episodes versus 5/16 initially; six pairs change from survival to a fall and none change in the reverse direction. Both checkpoints have zero full-task passes. [claim:C002] [evidence:E002]

| Measurement | Initial checkpoint | Trained checkpoint |
|---|---:|---:|
| Falls | 5/16 | 11/16 |
| Full-task passes | 0/16 | 0/16 |
| Mean raw episode return | 3,111.301 | 1,931.044 |
| Mean observed duration | 15.4625 s | 10.2875 s |
| Fall mode: rise / inside | 5 / 0 | 8 / 3 |
| Clipped action components / executed components | 2 / 284,510 | 25 / 189,290 |

- Paired trained-minus-initial return: mean **−1,180.257**, median **−451.102**,
  range **[−3,272.251, +237.368]**. All pairs appear in the figure and CSV.
- Paired fall cells: both survive **5**, initial only survives **6**,
  trained only survives **0**, both fall **5**.
- A fall shortens reward accumulation and later-phase observation. Smaller
  drift or total cost in a short episode is not an improvement claim.
- These are conditional action-noise outcomes from one training seed and one
  reset, not independent training replications or held-out generalization.
  No confidence interval, hypothesis test or universal safety claim.
- Result: **reject the trained policy for advancement; keep Study 018 B
  stopped**. Do not switch to stochastic evaluation to relabel the failure.

## What this narrows down

Thirteen falls occur in rise mode and three in inside mode before any rise. Rare clipping is not necessary for failure: all five initial fall episodes and seven of eleven trained fall episodes contain no clipped components. These are descriptive associations, not an isolated cause. [claim:C003] [evidence:E002]

| Rival explanation | Next discriminating intervention | Main risk |
|---|---|---|
| Oracle exit waits for an unhelpful loop boundary | One zero-residual probe changing only the crouch exit-boundary rule | Earlier exit can create a pose discontinuity or stall task progress |
| Action noise exposes fragile tracking | Separate fixed smaller-noise evaluation of both policies | It evaluates modified distributions, not the exact saved checkpoints |
| Training degrades the mean policy | Separate declared training contrast after substrate diagnosis | Changing trainer and reward together prevents attribution |

- Prefer the small **oracle-exit probe first**: it directly tests the reference
  composition design. Preserve the smaller-noise alternative.
- Fable's post-hoc review suggests delay between the progress guard and actual
  rise transition. Do not treat that explanation as causally established.
- Require a new predata protocol, exact control, single data-only candidate,
  handover pose/speed/height measurements and unchanged task gates. No next
  trial, candidate adoption, noise override or training is authorized by this report.

## Evidence and reproduction

The native run completes 34 episodes with 139 declared outputs. Independent scoring reproduces the complete diagnosis byte-for-byte and verifies recorded sampling arithmetic on all 32 sampled episodes. [claim:C004] [evidence:E002,E003,E004]

- Source: `615124f018e044a74b8ec564de0207237c7593fa`.
  Executable seal: 198 files,
  `348cc941ddae506b90692f67edeca35864f95254baaf07f6f23338c96dbb223a`.
- Legacy O7b parity completed before diagnostic approval/dispatch. Both exact
  saved-policy deterministic triplets then matched before sampled data.
- Scorer rechecks the exact output ledger, noise, sampled mean/std/raw-action
  arithmetic, clipping counts, rewards and independent boundary objectives.
  Recorded physics-substep failure flags are not independently resimulated;
  sidecars do not independently recompute neural means from observations.
- Focused runtime/scorer checks: 95 pass. Relevant GMT/feedback suite: 708 pass,
  2 clean-source-dependent skips; the clean old-study subset later passes all
  39 tests. No historical evaluator or gate changed.
- Source freeze ended after independent scoring. Native supervisor is terminal
  and released. New training transitions: **0**.

Run directory: sibling
`humanoid-harness-probe-runs/gmt_course_study020_saved_policy_diagnostic_20260907`.
Diagnosis: local `artifacts/gmt/course_configs/study020_diagnosis_20260907.json`.
Rescoring requires the pinned clean source, not a later checkout with its source
check disabled. Generated checkpoints, trajectories and figures remain local.

| Receipt | SHA-256 |
|---|---|
| Run manifest | `2c6962c519eea04d458e0f3f7dbd582ff535eb5e3bda63a92d90bcda1f90f289` |
| Native resource receipt | `3260b939b71bdce6ec895b057539f8acfa6d1bd23d68289bd9b1fb0f5f4e2559` |
| Root and independent diagnosis | `0a58b19ceb71fce8ee8dc301ab5e59a9170e4f312363c900ec8fb49cc24bece0` |
| Diagnostic config | `b7b33a0976bc29e61ca9bcbdd9265f9eb253fd49f7542ef96bcbca2a6e7412ab` |
| Paired noise | `61627836220ee63162812035555b3922a3e378ce4fc617d759c438e76ad9e9fe` |
| Clarified legacy parity receipt | `50eb61e85e1e07c9ad5dd260a72062912c2f430c38a3a05b4f27fd5650e8b0c2` |

The clarification separates reservation file bytes from the native canonical
reservation hash. It preserves the original receipt; parity results do not change.

## Visual evidence

- [Figure, caption and exact output hashes](FIGURE.md).
- [All paired outcomes](../../artifacts/gmt/course_configs/study020_figures_20260907/paired_outcomes_final.csv).
- This is a measured policy comparison, not a successful composition demo.

## Procedural guidance

Experimental-design guidance kept pairing and decision rules predata. Scientific
visualization guidance kept every failure visible beside episode duration.
Scientific-writing guidance preserved source/claim provenance and the human-review
boundary. Tooling attribution, not evidence of improved robot behavior:
Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner, A. M. (2026),
[Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).
