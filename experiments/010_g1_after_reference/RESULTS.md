# O6 rejected: the after-walk crop falls

| progress | The real feedback-to-proposal-to-probe cycle completed with exact unchanged-prefix parity. |
|---|---|
| bottleneck | O6 falls at 6.76 s; numerical seam improvements did not establish a stable transition. |
| next step | Retain the surviving reference baseline; test fixed observation normalization as a separate trainer factor. |

## Locked readout

| Prediction | Observed | Disposition |
|---|---|---|
| First 262 frames and first 263 state rows equal parent bytes | Exact frame, `qpos` and `qvel` parity | Pass |
| 20 s without falling; both switches | 6.76 s, non-foot ground contact; both switches | Fail |
| No after-state region revisits | Zero in 76 observed after samples | Censored at fall |
| After heading maximum <=1 rad | 0.58379 rad | Censored at fall |
| Lateral maximum below parent's 7.98828 m | 2.52693 m | Unequal horizons; not an improvement claim |

- Every one of 338 commands, transitions, reported phases and post-step
  numeric targets was reconstructed from the retained state and pinned config.
- After entry: command 262, phase 0.0667093 s, normalized pose cost 0.525306.
- One observed after-loop seam: control boundary 307; phase 0.926709 to
  0.006709 s; heading 0.520316 rad; forward speed 2.46141 m/s.
- Maximum observed backtrack from prior progress peak: 0.003472 m. The early
  fall prevents a full-horizon conclusion about straight traversal.
- Same spatial crouch crossing as parent: 71 samples, 59 compliant (83.10%),
  minimum height 0.515736 m, inside mean speed 0.702094 m/s.
- Full-task gate remains false: survival, depth, lateral error, overall speed
  error and roll/pitch tracking gates fail. Mean speed error is 0.639628 m/s;
  roll/pitch RMSE p95 is 0.321002 rad. All region visits count.
- Do not train or adopt O6 under this protocol. No second crop, retiming,
  alternative seed or evaluator change follows this outcome.

## What this establishes

- An actual Fable proposal was retained and passed through `g1 revise`, with
  exact parent config and reconstructed feedback identities.
- Only the after-walk oracle artifact changed. The unchanged prefix and
  reconstructed commands establish that the intended candidate was executed.
- The candidate fails its behavioral prediction. This does not isolate seam,
  velocity, entry matching or controller conditioning as the unique cause.
- No trained policy, upstream TorchScript equivalence, held-out success or
  generalization claim follows from this probe.

## Receipts

- Source: `1d60c464321aa387b0f928a964a3dde024048c16`; 192 runtime files;
  tree `17c52ac2b238bbe4dfecbe23f244c0fe027a6592ea77e251057763b0ea2743f8`.
- Candidate: `531bc1e80f4e8e0af5bda248fdb22372cbb9878badcb831a5a480c440711ce00`.
- Run: sibling `humanoid-harness-probe-runs/gmt_course_o6_probe_20260907/`.

| Artifact | SHA-256 |
|---|---|
| Run manifest | `43978e4ed74f93e9c6a58c80ae2326dd1476dc4734e110c2c879dedfb32eabae` |
| Resource receipt | `c85ffc42765aa7c35a7e7b4e062511fcff11c043e744f20032025ad59bab592d` |
| Reconstructed feedback | `3712579e7ac9ee6d51da96e79ed325a385bc4737dfcd2a0ed0bbf823ca8c8067` |
| Feedback receipt | `4a1de130b002033cfec69a928d56089ae258a50f07d4667f8eca17fc90561e94` |
| Local verified score | `c50cf162a399e11d2339c5f2e105a68eb2a7f47aa46356ac396d569db9b1a966` |

The local score is `artifacts/gmt/course_configs/study010_o6_verified_score_20260907.json`.
The accepted supervisor completed validation and released the heavy-job slot.
Training totals remain 32 runs / 1,638,400 transitions / zero full-task passes.
