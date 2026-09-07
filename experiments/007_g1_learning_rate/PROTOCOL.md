# G1: one lower-learning-rate hypothesis

| progress | Study 006 separates a high explained variance from failed learned behavior. |
|---|---|
| bottleneck | Larger residuals and worse survival occur together; the cause is not established. |
| next step | Test learning rate `3e-5` against retained `3e-4` controls. |

## Locked order, before new data

| order | oracle/reward | learning rate | transitions | purpose |
|---:|---|---:|---:|---|
| 0 | O2/r1 | 3e-4 | 32,768 | exact legacy compatibility check after profile wiring |
| 1 | O2/r1 | 3e-5 | 131,072 | optimization hypothesis against study-006 O2/r1 |
| 2 | O4b/r4 | 3e-5 | 131,072 | separate matched rate contrast on the combined candidate |

- Seed `20260906`; final checkpoint only; same static total-reward scale `1/64`.
- The low-rate arms start fresh from the seeded initial policy. They are not
  continuations or optimizer resumes of the retained 32k paths.
- Fixed G1 dynamics/reset/observations/actions, base weights, reference/oracle,
  task recipe, tracking reward, other PPO parameters, evaluator and budget
  within each rate contrast. No repeatable-loop oracle in this study.
- The compatibility arm must reproduce retained numeric policies, trajectories,
  frame/reward ledgers, evaluations and telemetry. Identity/runtime failure
  blocks new training; behavioral failure does not suppress a locked arm.
- Local only: one shared heavy job, CPU=1, memory<=8 GiB, wall<=1,200 s per run.
  Exact committed source/config/resource authority must be accepted before launch.
- Compare the two new arms only to their own same-oracle/reward 131k controls.
  They do not isolate an oracle or reward effect from each other.

## Measurements and decision

- Validate source/config/trainer/receipt chains, initial-policy parity and
  same-oracle zero-residual parity before interpreting learned behavior.
- Retain all 256 updates, raw/scaled reward units, attempted epochs, KL,
  clip fraction, explained variance, episode/fall counts and episode lengths.
- Report falls and episode counts together in eight 32-rollout blocks. Compare
  first/last 64-rollout raw episode-return means descriptively; variable episode
  lengths and censoring prevent treating them as equal-exposure scores.
- Primary behavioral prediction for each arm: 20 s without falling, both oracle
  switches, and actual region exit. Preserve full-task gates unchanged.
- Secondary O2/r1 prediction: final residual RMS<=0.15 and no fall-rate increase
  from the first to last quarter. Compare height, compliance and speed as well;
  a small residual that merely restores zero-residual behavior is not task success.
- Report inside samples/mean speed/MAE/compliance/minimum height and after-state
  behavior separately. Missing 10/20 s checkpoints remain unavailable.
- Adoption requires the primary prediction and no regression against the retained
  control on complete-region compliance or minimum height. This is a local
  development decision, not evidence across seeds or a universal trainer choice.
- If the smaller rate fails, retain the result. Do not infer that reward design
  or optimizer design has been uniquely ruled in or out. Further changes require
  a new hypothesis and explicit experiment family.

## Rationale and limits

- Fable review `20260907T002541.941919Z-846c108b298d42b78381363c929da206` proposed
  the rate change after reviewing bootstrap, likelihood and history paths.
  No concrete code defect was found.
- KL early stopping is not a hard KL bound; one gradient step can overshoot.
  Smaller learning rate does not guarantee ten times less policy movement.
- Unchanged policy standard deviation says nothing sufficient about its mean.
  Base normalizer statistics alone do not establish raw observation conditioning.
- Baselines, failures and exact hashes: [study 006](../006_g1_composition_depth/RESULTS.md).
