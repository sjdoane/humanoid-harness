# Depth reward: reject this candidate

| progress | One real LLM reward revision completed 131,072 training transitions; all artifact and parity checks pass. |
|---|---|
| bottleneck | The predicted deeper crouch did not occur. Lateral drift and overall speed error worsened. |
| next step | Retain the negative result; test the task/reference geometry mismatch before another reward or trainer sweep. |

## Matched result

- [Protocol](PROTOCOL.md): seed `20260906`, final checkpoint, reward only.
- Candidate source `be359a2accd21b3a94727695a2f7837311a36aaa`.
  Its executable tree exactly matches the retained Study012 finite baseline.
- Fable supplied the actual proposal; the CLI reconstructed its source feedback,
  admitted the replacement, and preserved every non-reward config field.

| Measure | Finite r1 baseline | Depth r4 candidate |
|---|---:|---:|
| Duration / falls / switches | 20 s / 0 / 2 | 20 s / 0 / 2 |
| Inside samples / compliant | 60 / 31 | 68 / 38 |
| All-visits compliance | 51.67% | 55.88% |
| Minimum inside root height | 0.504942 m | **0.516207 m** |
| Inside mean-speed deviation | 0.183576 m/s | 0.087875 m/s |
| Overall speed MAE | 0.340857 m/s | 0.384267 m/s |
| Maximum lateral error | 2.226646 m | 3.152714 m |
| Joint / roll-pitch p95 | 0.221358 / 0.149644 rad | 0.223953 / 0.175050 rad |
| Residual RMS | 0.137860 | 0.137573 |
| Training episodes / falls | 141 / 15 | 142 / 16 |
| Last-16-update explained variance | 0.69725 | 0.71931 |
| Original gates passed | 7 / 11 | 7 / 11 |

- **P1 fails:** minimum height exceeds the predeclared `0.48494220311997693 m`
  target and is worse than baseline. All eight other screen conditions pass.
- Original task failures: compliance, dip, overall speed MAE and lateral error.
  Inside mean speed now passes; this does not rescue the failed depth hypothesis.
- The multiplier is active in exactly 58 zero-residual rows and 68 final-policy
  rows. Every saved reward component reconstructs exactly. These are evaluation
  traces, not per-step training activation telemetry.
- Zero numeric trajectory and initial policy match baseline byte-for-byte.
  All non-reward frame/evaluation fields match; only the declared recipe identity,
  posture component, task/total reward and reward summary differ.
- One seed does not explain why depth worsened. No stronger multiplier,
  automatic replication, checkpoint selection or default promotion follows.

## Receipts and cost

Sibling run directory: `humanoid-harness-probe-runs/gmt_course_finite_r4_candidate_20260907/`.

| Artifact | SHA-256 |
|---|---|
| Candidate config | `39dd6a045fe680676389a8b46bdfe146bd231ccdb6fc9811499d56a33ab2cb13` |
| Revision receipt | `26170f8aef7ee1974fb9b965896121f197325d3cb1940990bc5f8b7d822a223f` |
| Run manifest | `c73c4e771a5bf9e3e1f4f9e28cb9c560467f5855b0323605ef512058deb0eae2` |
| Resource receipt | `cc5c3c26c943bf12490a8aa3447a18013f28935565b217bb539607f783b01959` |
| Verified score | `89e7f5208e7c72660cbf915bb8c9de99567445a887f6845758bf03f0c43c542d` |
| Fresh feedback | `2d70389f97eb7bc8906d1c8dc4cd5448a5020e6ca5110904936330a74a224c4b` |

- Score: `artifacts/gmt/course_configs/study014_finite_depth_score_20260907.json`.
- Wall time 311.23 s; peak RSS 1,052,311,552 bytes. Worker successful/released.
- Project total: 37 completed training runs / 2,293,760 transitions;
  **zero full-task passes**. Probes are counted separately from training.

## Review repairs before data

- Independent Sol review found the missing recipe-specific reset exception.
  It also required fail-closed behavior under Python `-O` and exact zero-trace
  activation. Repairs committed before data; 12 focused tests pass.
- Old resource proposal at `85d117d` was withdrawn. Only fresh proposal
  `20260907T054511.561239Z-3575c34434a541c0a6888304898b3188` and acceptance
  `20260907T054551.626125Z-e12d1e979a204220bb8f4dcadc551502` authorized execution.
- Fable independently reconstructed the result in mailbox
  `20260907T055320.629355Z-db319af4f86548b0bd264170afc9a2a2` and verified
  exact digests/activation counts in `20260907T055411.783451Z-4c8fd2a2e8904b438a6fe864b5ea338c`.
  Its verdict matches: reject this depth intervention.
