# One depth-reward revision on the finite-horizon baseline

| progress | A real Fable proposal uses the verified Study012 failure feedback. |
|---|---|
| bottleneck | The retained baseline survives but crouches too shallowly and traverses the region too quickly. |
| next step | Train one reward-only candidate; compare against exact retained baseline without changing task gates. |

## Hypothesis and authority

- Astra leads; Fable proposes and independently reviews. No separate Fable run.
- Use only superseding mailbox proposal
  `20260907T052435.978296Z-0ce70018bc6a406dbec0f4562ce65ca0`.
  Earlier `052342.683689` proposal is withdrawn, not an additional candidate.
- Retain its raw body plus newline at local
  `artifacts/gmt/course_configs/study014_fable_proposal_v2_20260907.json`;
  SHA-256 `a9ca12027d3f85157ac0b018cf5215bdcdfd7ebebe20d414ac19f12c622b0385`.
- `g1 revise` must rebuild the exact parent feedback and admit the data-only
  replacement. The proposal cannot change the evaluator or execute code.
- Mechanism: the old posture term is flat inside the allowed height band.
  The admitted v2 term multiplies it by a Cauchy factor above 0.48 m, scale
  0.10 m, strength 1.0. It leaves all other components and weights unchanged.
- This does not directly repair excessive speed or lateral drift. Coupled
  learned behavior can change those metrics anyway; measure all of them.
- The in/out-region reward discontinuity is a competing explanation, not a
  proven cause. Failure does not automatically authorize a stronger reward.

## Frozen comparison

- Baseline: Study012 finite candidate, config
  `0be730e48fc53c9e34e671138a49cb6cfd3f1a41b350cf2047d55c347a6600bf`;
  manifest `fa4fe2cf601eae318cce06b3704ae2206b92b1e99011e95d9dbd3a687a8ad502`;
  resource `75e67574fb797a576c49886192bc0f88753d44142e873dbb500d3ad96bb06f89`.
- Feedback `00addac31556f7b8c1ae3f9bc0c435bf655a67c58b818b3583ed1e94826de5c7`.
- Candidate changes **only** reward to v2: `depth_strength=1.0`,
  `ceiling_fraction=0.6`; speed/posture/lateral/heading/failure weights
  `1.0 / 2.0 / 1.0 / 0.5 / 1.0` remain unchanged.
- Fixed O2 references/oracle, task, base weights, tracking reward, finite
  runtime, 2,171 observations, residual actions, reset, trainer profile 3,
  seed `20260906`, 131,072 transitions, final checkpoint only.
- Retained baseline is a comparator, **not a successful substrate** or chosen
  winning seed. Its Study012 failures stay on record.
- Reuse it only if executable tree is exactly
  `3ba7fe0ab72374e0667402848460c43915024fbcd2fe44f9ce5de791f0ab8b8e`
  (194 files). A docs/tests-only commit may differ. Assert tree equality
  explicitly across commits; changed executable source requires exact control
  reproduction before the candidate.
- Verify zero-residual numeric trajectory and initial policy equality.
  Reward identities/components/totals may differ in frame/evaluation metadata;
  every non-reward field and independent objective must remain equal.
  Specifically, reset `reward_sha256` must equal each arm's exact recipe;
  all other reset fields remain equal. Evaluation reward sum may differ.
- One exact independently accepted resource reservation: CPU local,
  1,200 s / 8 GiB; one heavy job, source freeze through terminal.

## Pre-data predictions and stop rule

All spatial visits count, not only the first crossing. Use exact values below,
not the proposal's rounded prose.

| Requirement | Exact criterion |
|---|---|
| P1: meaningful deeper crouch | Minimum inside height <= `0.504942203119977 - 0.02` m |
| P2: compliance non-regression | Inside compliant fraction >= `31/60` |
| Survival and composition | 20 s, no fall, two switches, region entry and exit |
| Exposure | At least 25 inside samples |
| Speed regression guardrail | Inside mean-speed deviation <= `0.18357610804822067 + 0.05` m/s |
| Tracking guardrails | Joint p95 <=0.35 rad; roll-pitch p95 <=0.25 rad |

- Report all eleven original task gates, including the stricter 0.10 m/s
  inside-speed gate. The screen guardrail does not replace it.
- Report progress, lateral error, overall MAE, residual RMS, critic/optimizer
  telemetry, training falls and intrinsic-horizon completions.
- Reconstruct active reward terms from raw states. A failed P1 with active
  multiplier and intact guardrails rejects this depth intervention here.
  The matched zero trace must contain exactly 58 active multiplier rows,
  precomputed from the retained baseline. Final activation is descriptive;
  saved evaluation rows do not certify per-step training activation telemetry.
  Failed guardrails also reject it; no causal diagnosis follows from one seed.
- No automatic replication, stronger reward, oracle edit, extra budget,
  checkpoint selection or success claim. Record result, diagnose, and declare
  a new comparison before any further intervention.
