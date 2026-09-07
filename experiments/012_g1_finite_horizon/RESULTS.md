# Finite horizon: correct semantics, failed behavioral screen

| progress | Both 131,072-transition jobs completed; termination and numerical parity checks pass. |
|---|---|
| bottleneck | The finite candidate fails compliance non-regression and the inside-speed requirement. |
| next step | Keep the explicit finite semantics for new studies; test one reward-only depth proposal in Study014. |

## Matched result

- [Protocol](PROTOCOL.md) predates training. Seed `20260906`; final checkpoint.
- Source `9bca23bf62ea386cd7f0492aa63626b6c2a219dc`, frozen through both jobs.
- Only factor: intrinsic horizon termination instead of legacy timeout bootstrap.

| Measure | Legacy control | Finite candidate |
|---|---:|---:|
| Duration / falls / switches | 20 s / 0 / 2 | 20 s / 0 / 2 |
| Inside samples / compliant | 81 / 52 | 60 / 31 |
| All-visits compliance | 64.20% | 51.67% |
| Minimum inside root height | 0.513356 m | 0.504942 m |
| Inside mean-speed deviation | 0.032732 m/s | 0.183576 m/s |
| Overall speed MAE | 0.295033 m/s | 0.340857 m/s |
| Maximum lateral error | 3.419531 m | 2.226646 m |
| Joint / roll-pitch p95 | 0.229084 / 0.178847 rad | 0.221358 / 0.149644 rad |
| Residual RMS | 0.191946 | 0.137860 |
| Training episodes / falls | 151 / 29 | 141 / 15 |
| Last-16-update explained variance | 0.35574 | 0.69725 |
| Wall time | 333.64 s | 342.90 s |
| Original development gates passed | 8 / 11 | 7 / 11 |

- Candidate passes seven of nine substrate requirements. It is **not promoted
  on behavioral merit**. The original task still fails posture compliance,
  depth, inside speed and lateral error. No full-task or held-out success.
- The intrinsic ending is terminal regardless of comparative performance.
  New studies use that explicit profile; old run identities remain unchanged.
- Fewer training falls and higher critic explained variance are one-seed
  observations, not established general improvements. Equal transition budgets
  do not imply equal episode exposure.
- Exit-window fall timing cannot rule out semantics-mediated changes in the
  optimizer trajectory. Bootstrap changes propagate through value learning.

## Verification

- Control reproduces all ten retained Study011 normalized outputs byte-for-byte.
- Candidate zero trajectory, frame records and initial policy are byte-identical
  to control. Evaluation differs only in the declared reset/runtime metadata.
- Frozen base weights and all other config fields match. Candidate telemetry
  counts 126 horizon endings plus 15 falls = 141 completed episodes.
- Parent, Fable and independent Sol review agree on the actual task metrics.
  Fable's earlier control count of six passing gates was a reporting error;
  the retained evaluator reports eight. No gate was changed.
- Independent reviews: mailbox `20260907T052213.680287Z-c4654fd9e4344a73adc1935fd9eb99b6`
  and the subsequent exact-score acknowledgment at `20260907T052342.730740Z-51a95525099d44a2ba6f155108787214`.

## Receipts

Run directories are in sibling `humanoid-harness-probe-runs/`.

- Control: `gmt_course_intrinsic_horizon_control_20260907/`.
  - Manifest `2961ceb06ea719a2c9e5d54c75802724cbe74950084b71e59b8f3626f8889cc7`.
  - Resource `a7e263b650011679e263bf51107020267d66929e2f5835d34e221540df23a027`.
- Candidate: `gmt_course_intrinsic_horizon_candidate_20260907/`.
  - Manifest `fa4fe2cf601eae318cce06b3704ae2206b92b1e99011e95d9dbd3a687a8ad502`.
  - Resource `75e67574fb797a576c49886192bc0f88753d44142e873dbb500d3ad96bb06f89`.
  - Config `0be730e48fc53c9e34e671138a49cb6cfd3f1a41b350cf2047d55c347a6600bf`.
- Executable tree: 194 files,
  `3ba7fe0ab72374e0667402848460c43915024fbcd2fe44f9ce5de791f0ab8b8e`.
- Score: `artifacts/gmt/course_configs/study012_finite_horizon_score_20260907.json`,
  SHA-256 `6f91c0b6bf9cb9f08029e39af95a1112e80e7dff55e0c5252a7ca94df8f0812e`.
- Fresh candidate feedback: `gmt_course_intrinsic_horizon_feedback_20260907/feedback_v1.json`,
  SHA-256 `00addac31556f7b8c1ae3f9bc0c435bf655a67c58b818b3583ed1e94826de5c7`.
