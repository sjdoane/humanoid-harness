# Study018: trained baseline rejected; reward revision stopped

| progress | Baseline A completed 131,072 transitions; its final policy survives and executes all four stages. |
|---|---|
| bottleneck | Posture compliance and speed fail the predeclared admission checks. No full-task pass. |
| next step | Stop B. Test an earlier crouch transition separately before another training comparison. |

Local development report; AI-assisted draft. Numeric/receipt checks and independent
agent reviews are complete. Accountable human scientific review remains pending.
Manuscript lint is not complete: sources are unverified by a human, and this internal
report does not tag each numeric table/path line as a separate manuscript claim.
No target venue, submission approval, or formal reporting-guideline adherence is claimed.

## Locked decision

Study018 A is **not admitted to B**. Compliance is 47/71 = 0.661972 against
the unchanged 0.75 floor; speed MAE is 0.474331 m/s against 0.35 m/s.
The final policy passes 7/11 original task gates. No reward candidate B was
proposed or launched. [claim:C001] [evidence:E001,E002]

- The original protocol is unchanged. Its source freeze ended after verified
  scoring and the decision to stop B.
- Any next oracle or reward test requires a new declaration, not an amendment
  that retroactively admits this baseline.

## Measured comparison

One training seed (20260906), one deterministic final-policy episode and one
zero-residual episode; each has 1,000 correlated control steps, not 1,000
independent replicates. The final checkpoint was selected in advance.
[claim:C002] [evidence:E001,E002,E003]

| Measurement | Zero residual | Trained A |
|---|---:|---:|
| Survival / falls / switches | 20 s / 0 / 3 | 20 s / 0 / 3 |
| Region posture compliance | 56/74 (75.68%) | 47/71 (66.20%) |
| Minimum physical-region height | 0.515773 m | 0.512418 m |
| Overall speed MAE | 0.241548 m/s | 0.474331 m/s |
| Maximum lateral error | 10.939883 m | 15.368558 m |
| Joint / roll-pitch tracking p95 | 0.239778 / 0.183717 rad | 0.236567 / 0.205591 rad |
| Raw task reward sum | 2,947.331 | 2,583.940 |
| Raw total reward sum | 3,863.809 | 3,488.089 |
| Original task gates passed | 9/11 | 7/11 |

Table values come from reconstructed recorded traces. Training did not improve
these task outcomes or its own episode reward. This is not a matched estimate
of old-versus-new runtime effects or a reward-recipe contrast.
[claim:C003] [evidence:E002,E003,E004]

## Post-hoc diagnosis — hypotheses, not new gates

- On 15 of A's 24 failed posture samples, the retained **post-step tracking
  target** itself exceeds the 0.60 m ceiling. The other nine failures have
  targets at or below the ceiling. Mean actual-minus-target height across
  all 24 failures is +0.030593 m. This is not the entire pre-action reference
  window, and does not prove that the plant cannot reach the target.
  [claim:C004] [evidence:E004]
- Both zero and trained traces encounter high reference phases during the
  physical crouching region. Modes are state-gated, but phase advances on a
  within-mode clock. An earlier transition is a cheap, separate test of
  spatial/phase alignment; it is not continuous phase estimation.
  [claim:C005] [evidence:E003,E004]
- Training records 121 falls and 94 horizon completions among 215 completed
  episodes. Final critic explained variance is 0.898493. A well-fitted critic
  is not evidence of a better actor. Persistent exploration, cumulative policy
  drift, reward sensitivity, and seed variance remain rival explanations.
  [claim:C006] [evidence:E002,E003]
- After-mode lateral reward averages 0.00005925 for A versus 0.00735305 for
  zero residual. This supports investigating reward-tail sensitivity, but
  does not isolate reward saturation as the cause of training failure.
  A kernel change must remain separate from an oracle change.
  [claim:C007] [evidence:E004]

## Integrity and reproduction

- A used clean source `e57f220a7694ab8b5a16758fcae1a5ee6a12d2d2`;
  executable tree: 196 files,
  `6010bf0ce11bb56c30fe71fdb8b368235f7c5cde867dba8179571dff0ba11f15`.
- A fresh legacy probe reproduced all three retained Study015 outputs exactly
  before A launched. A's zero-residual trajectory, actions and frame trace also
  reproduce Study015; only declared runtime/reset metadata differs.
- Config, native resource approval, training telemetry, policy bytes, raw
  states, reward reconstruction and independent objectives verify. The scorer
  returned a valid failed-behavior result, not a scoring error.
- Verification and dispatch were separate successful calls. The resource
  supervisor reports success and release; no worker remains active.
- Exact rescoring requires the pinned clean source, scorer and dependency
  bytes, not a later checkout with a silently relaxed source check.
  [claim:C008] [evidence:E002,E003,E005]

## Recorded-state replay

The GIF shows recorded qpos/qvel through forward kinematics; no policy or
dynamics rerun. The blue region is a visual task marker, not a physical
obstacle. The evidence UI labels this run **FAIL, 7/11**.
[claim:C009] [evidence:E006]

- Run: sibling `humanoid-harness-probe-runs/gmt_course_study018_baseline_20260907`.
- Replay: sibling `humanoid-harness-probe-runs/gmt_course_study018_baseline_recorded_20260907.gif`.
- Score and diagnosis: local `artifacts/gmt/course_configs/study018_baseline_admission_20260907.json`
  and `study018_physical_phase_diagnostic_20260907.json`.
- Generated trajectories/checkpoints remain local and uncommitted.

| Receipt | SHA-256 |
|---|---|
| Protocol | `92672e183a357fa795b6f7be7275d00c794b6dfd90a2fd585bf3ec00ac8ec2c8` |
| Scorer | `564ad69031a54e1799114183e96384e3766be7741508b6310a7b60aa12ed309e` |
| Predata seal | `aaf9d636561f04878bac690f0006068a14d5f2d373f3f204a5d316e59a14e4cb` |
| Admission score | `ee15d6fd24703f5785e9a825db6a3557168e6feec5a5f2333f42f69e41d3d08f` |
| Run manifest | `33b5e25ea8dfa7a8e3a4f520e7386edce067b920d75201e8ffb81581bd96f705` |
| Resource receipt | `c06fbc88be3098171019d639ddab2ce3e3f7b378fa0e3da43cbea81e7dbd2d0e` |
| Physical/phase diagnosis | `ac640fd299070cbb2c00e1f2097d59189d736f3f1066aa1a6f159099b93bf15c` |
| Replay | `62702c702726a0cda2ae841e616c1a7e9af0243700038409db9b0e8cb5331b59` |

Source/claim registries retain locators and pending human-verification status.
