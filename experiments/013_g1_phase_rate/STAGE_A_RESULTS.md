# Stage A: balanced reference candidate

| progress | O7 executes walk → crouch → rise → walk and survives 20 s. |
|---|---|
| bottleneck | Full task fails five gates. No repeated crouch loop was executed. |
| next step | Retain the qualified probe substrate; review phase feedback separately. No training admission follows. |

## What changed

- Fable supplied one data-only oracle proposal from verified O5 feedback.
- Only crouch loop start/end changed: `3.90/4.86 → 3.78/5.64` s.
  New oracle ID records lineage. Task, reward, guards and controller stayed fixed.
- Execution source `031a0660401bda7cf887fab68eea686980c9a4fd`.
- Zero residual, seed `20260906`, fixed reset; 0 training transitions.

| Metric | O5 parent | O7 candidate |
|---|---:|---:|
| Duration / falls | 6.40 s / 1 | 20.00 s / 0 |
| Executed switches | 1 | 3 |
| Rise / after intervals | 0 / 0 | 24 / 737 |
| All-region posture compliance | 100/147 (68.03%) | 56/220 (25.45%) |
| Minimum inside height | 0.494438 m | 0.515773 m |
| Inside mean forward speed | 0.009208 m/s | 0.208818 m/s |
| Maximum lateral error | 2.369911 m, fall-censored | 5.659565 m |
| Full-task gates | 3/11 | 6/11 |

- All predeclared Stage A admission conditions pass: survival, spatial exit,
  actual rise and actual after execution. This is not full-task success.
- Crouch enters at command 92, exits at its first-pass boundary 239; rise
  exits at 263. No repeated crouch cycle or sustained loop balance is established.
- The longer first pass reaches the exit guard before any repeat. It includes
  the newly exposed source interval 4.86–5.64 s; loop-start balance was not tested.
- The first region crossing has 74 samples; later revisits add 146. All visits
  count. The one-way oracle stays in `after` during backtracking instead of
  re-entering crouch. This is a recovery-design limitation, not evaluator noise.
- Full-task failures: posture compliance, depth, inside speed, overall speed,
  lateral accuracy. Final progress is 1.916 m after a maximum of 5.408 m.
- Different observation durations prevent reading fall-censored whole-episode
  O5/O7 errors as matched-horizon improvements.

## Verification

- All 1,000 oracle commands and post-step targets reconstructed from saved states.
- Predicted unchanged prefix passes exactly: 106 qpos/qvel rows and 105 action rows.
- Hash/manifest/resource/feedback reconstruction passes; no new dynamics replay.
- Wall time 2.747 s; peak RSS 720,257,024 bytes. Resource slot released.
- Initial reservation was rejected before spawn because expiry lacked the
  required six fractional digits. Exact reacceptance corrected only that format.

| Artifact | SHA-256 |
|---|---|
| Candidate config | `430c6674485a4ec9ca7546389a71b485c7670f11d08a8b911b48758ee6a51ff4` |
| Revision receipt | `515bf99a03b1ab642b11fd45a80711684e7d02a7afdd7a0894302661eeac4f49` |
| Run manifest | `6141e860a018fcb0dfac89444234fe987351f188f29e14d3400d0a78db9fe34c` |
| Resource receipt | `6fa316fc8b15eccf29e42920ac48914cf1baf67f885af0e93e2b1916d06db466` |
| Verified score | `5ae07946f286ce79c41a50fa9a7a495c440c416497d452411f13db9cd3221569` |
| Feedback | `814d73e5ddd006c25f157ac5115f599fe584fb3633613c6b2d2bc78310ab6b59` |

- Local run: sibling `humanoid-harness-probe-runs/gmt_course_o7_probe_20260907`.
- Proposal: mailbox `20260907T035729.280347Z-92827778d8194d54a0721a236e27d5a5`.
- Raw proposal retained with terminal newline; no fabricated model authorship.
- Stage B still needs its own matched protocol. The pure scheduler is unwired.
