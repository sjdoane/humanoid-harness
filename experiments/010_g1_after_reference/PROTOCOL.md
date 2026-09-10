# O6: change only the after-walk reference

| progress | O4b's finite crouch crossing is reproduced; reference-input dependence is verified in studies 009's two configurations. |
|---|---|
| bottleneck | Full basic_walk turns and backtracks; the previously tested repeat-crouch candidate failed before rising. |
| next step | Apply one feedback-linked after-only oracle proposal, then probe it without training. |

## Frozen comparison

- Parent: legacy three-state O4b, config SHA
  `bd22f6b5e1cd002e7dc2edd0bf5707cbf12d5852ff4e69d3008ad481f211727c`;
  retained manifest `ade71ffbb4c67ebf6237ca462aa8e4a10a935727138de70e5c90b99094031334`.
- Keep before-walk, finite crouch segment, both guards, dwell, task, reward r1,
  base actor, G1 dynamics/reset, observation/action contracts, evaluator,
  20 s horizon and seed 20260906 unchanged. Literal zero residual; no training.
- Only add a distinct `walk_after` behavior/segment and assign it to the
  existing after state. Use admitted basic_walk `30.62..31.56 s`, unchanged
  native cadence, wrap within segment, nearest entry phase over the whole
  segment. Do not retime, alter source features or admit external data.
- The legacy 2171-dimensional family is retained. Do not modify the separate
  immutable four-state probe-only profile to enable training.
- Current feedback was rebuilt from the full parent trace:
  `gmt_course_o4b_feedback_for_o6_20260907/feedback_v1.json`, SHA
  `4a07e57faf6685121aef778572063860aa85688db366b76140974b4a25d1b54f`.
  Fable returns one JSON oracle proposal; the actual revision CLI verifies and
  applies it before the exact probe request is accepted.

## Why this candidate

Four intervals were compared offline, without simulation: the two retained
native-audit crops and Fable's `30.62..31.56` / `30.66..31.56` suggestions.

| Native-feature diagnostic | Prior balanced crop | Selected 30.62..31.56 |
|---|---:|---:|
| Signed local lateral velocity | +0.05846 m/s | -0.00504 m/s |
| Mean absolute local yaw rate | 0.26848 rad/s | 0.18538 rad/s |
| Integrated local lateral proxy per loop | +0.06434 m | -0.00473 m |
| Normalized pose seam MSE | 0.04784 | 0.00665 |
| Entry pose cost at actual O4b decision 262 | 0.56485 | 0.52531 |
| Forward-speed MAE to 0.7 m/s | 0.02552 m/s | 0.25146 m/s |

- Tradeoff: smaller lateral bias and pose/velocity seams, but a faster motion
  poorly matched to the task's 0.7 m/s target. This is not a full-task prediction.
- Sampling: float32 runtime features at native frame times; left-rectangle
  quadrature with final interval clipped to the exclusive endpoint. A 50 Hz
  check preserves the qualitative tradeoff. Integrated local velocities are
  not world-frame displacement or measured robot motion.
- Short looping references always expose a seam within the 1.9 s actor
  lookahead. Good static seam metrics do not establish dynamics feasibility.

## Predictions and stop rules, before rollout

1. First 262 frame records and first 263 `qpos`/`qvel` rows match parent bytes.
   A mismatch stops interpretation; do not normalize away a difference.
2. Complete 20 s without falling, with both state transitions observed.
3. No spatial posture-region samples after the after-state transition.
4. Maximum after-state heading error <=1 rad.
5. Maximum lateral error is below parent's 7.9882757834790175 m.

- Report every unchanged full-task gate, region compliance/depth/speed,
  progress/backtracking, entry phase/pose cost and each after-loop seam.
- All region visits count. No best-seed selection, evaluator relaxation,
  phase-clock change, training retry or second crop after seeing this outcome.
- No prediction of depth or speed-gate success. If traversal stabilizes but
  speed remains wrong, a later reward/trainer study must address that explicitly.
- One accepted local job, <=1,200 s and 8 GiB. Artifact validation failure
  stops work on the candidate; behavioral failure is retained and diagnosed.
- A probe cannot demonstrate learned oracle/reward improvement. Successful
  feasibility only admits a later separately frozen matched training study.
