# G1: reference composition x depth-directed reward

| progress | O4b passes zero-residual gait admission; r4 is an admitted feedback-linked LLM proposal. |
|---|---|
| bottleneck | Neither improved whole-task behavior nor their combined trained effect is measured. |
| next step | One O4b/r1 pilot, then four fixed final-checkpoint training arms. |

## Fixed order, before new data

| order | oracle | reward | transitions | seed | purpose |
|---:|---|---|---:|---:|---|
| 0 | O4b | r1 | 32,768 | 20260906 | same-budget oracle pilot |
| 1 | O2 | r1 | 131,072 | 20260906 | larger-budget control |
| 2 | O4b | r1 | 131,072 | 20260906 | oracle factor |
| 3 | O2 | r4 | 131,072 | 20260906 | reward factor |
| 4 | O4b | r4 | 131,072 | 20260906 | combined factors |

- Local only; one shared heavy job, one CPU thread, 8 GiB, 1,200 s per job.
  Expected wall: approximately 75-100 s pilot and 300-400 s per larger job,
  estimates from recent runs, not deadlines or guaranteed throughput.
- No pilot-dependent parameter choices. All four larger arms run regardless
  of pilot behavior, unless a resource/runtime/identity failure invalidates
  execution. Preserve failed policies and every final checkpoint.
- Freeze G1 MDP, task, reset, action/observation interfaces, base weights,
  tracking reward, PPO parameters, 1/64 trainer scaling, evaluator and seed.
  The two budgets are separate strata; no cross-budget reward attribution.
- The reference universe is the same eight hash-admitted numeric GMT clips.
  O2 consumes walk_stand+crouchwalk_stand; O4b consumes basic_walk+crouchwalk_stand.
  The oracle factor jointly changes walking-clip selection and the exit guard;
  this does not isolate those two internal choices.

## Artifact derivation

- O2/r1 parent: scale64 seed06 config
  `f24182f42ee311a32a93bd194f661acb17bbeaa45344f2fbe8c6a19f7010e4d6`.
- r4: Fable mailbox `20260906T235910.758442Z-88ddf20c839f4710b375ca66fe1f76b0`.
  Raw proposal retained locally, including its original claims, under hash
  `2f24d0e33dd7b3d360d84eefc189dfc58b2043ca1fbb3ad89b0a4011754ad0a1`.
- Verified feedback `cda7bc397542fb1548e0d27ce37816eac1d0653dee5269dc021f23489f4fc82a`
  yields the admitted O2/r4 32k candidate
  `f59f26f849df714d3e8f31d52736485524a4f89e304210999bb4dd934edc8396`;
  revision receipt `21a2cc75491225639443d48fc91f23128e3278715e030fb30bde3fc7c6b52117`.
- r4 uses recipe v2, depth_strength=1.0, ceiling_fraction=0.6. All original
  weights remain fixed. The depth ceiling is 0.48 m; tracking is unchanged.
- O4b probe config `bd22f6b5e1cd002e7dc2edd0bf5707cbf12d5852ff4e69d3008ad481f211727c`
  becomes a train config by setting mode/train budget and adding the same trainer.
- O4b/r4 is an explicit experimental cross-product of the admitted O4b oracle
  and unchanged r4 recipe, not a second LLM proposal on an O4b feedback parent.
- The 131k configs change only training_steps from their declared 32k parent.
  Record each final config/source identity in its exact launch authority.

## Measurements and predictions

- All full-task gates stay unchanged. Report each cell, including falls.
  A fall prevents a full-horizon mean comparison; it is itself an outcome.
- Verify every output, exact trainer object, unchanged base weights, same-seed
  initial policy, and reward-independent zero-residual trajectory within each
  oracle. Reward-bearing frames compare only within the same recipe.
- Learning: retain all 256 updates; explained variance uses the last 16 of 256
  for every larger arm. Report >=0.5 as a diagnostic alongside behavior.
- Inside region: samples, compliance, minimum height, mean speed deviation,
  speed MAE and actual entry/exit. Do not confuse task region with oracle state.
- After oracle exit: speed MAE, lateral error at 10/20 s and final progress;
  missing times after a fall remain unavailable, never filled in.
- Oracle prediction at fixed recipe/budget: O4b compliance >= O2 and inside
  mean-speed deviation <= O2. Report each contrast even if it contradicts this.
- Reward predictions at each fixed oracle: r4 lowers minimum inside height
  by >=0.02 m and does not lower compliance versus r1. Report the 0.50 m gate
  separately; do not call a smaller but insufficient drop a pass.
- Guardrails for r4 relative to its same-oracle r1: 20 s/no fall/two switches,
  >=25 inside samples, mean inside speed no more than r1+0.05 m/s,
  compliance>=0.534, overall MAE<=0.35, lateral<=2.970430,
  joint p95<=0.35 and roll/pitch p95<=0.25 rad.
- The O4b/r1 learning prediction is overall speed MAE<0.835889 and lateral
  max<7.988276 versus its own zero-residual run. Other task gates still matter.
- Verify the depth multiplier from exact reward components and task state,
  not merely posture reward<2.0 (the old out-of-band penalty can also do that).

## Claims that are not accepted

- One seed is exploratory: report contrasts, not generalization or a robust
  statistical interaction. A combined pass is not a guarantee on new inputs.
- A null at maximum depth_strength does not eliminate weaker strengths or
  alternative reward forms. No additive-reward change follows automatically.
- Similar policy standard deviation does not show an unchanged policy mean.
- Better speed but unchanged depth does not uniquely identify reward limitation.
- A zero-residual gait pass is not a training-success or full-task certificate.
- The O4b height guard passed with roll/pitch p95 only 0.003 rad below its
  threshold. No safety guarantee follows from a closed guard.

## Prior evidence

- Trainer adoption and O4/O4b traces: `../005_g1_training_conditioning/RESULTS.md`.
- Existing raw/scaled families remain separate, with every failure retained.
- Candidate data: ignored `artifacts/gmt/course_configs/`; run outputs live
  outside the checkout in sibling `humanoid-harness-probe-runs/`.
