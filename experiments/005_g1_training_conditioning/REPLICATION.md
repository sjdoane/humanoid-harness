# Scaled-trainer replication and reward contrast

| status | evidence |
|---|---|
| progress | Seed 20260906 passed the original trainer screen; no full task pass. |
| bottleneck | The changed trainer has not been replicated or tested with tracking-only reward. |
| next step | Run the five fixed jobs below, sequentially; retain every final checkpoint and outcome. |

## Fixed jobs, before data collection

| order | oracle | task reward | seed | transitions |
|---:|---|---|---:|---:|
| 1 | O2 | r1 | 20260907 | 32,768 |
| 2 | O2 | r1 | 20260908 | 32,768 |
| 3 | O2 | r0 | 20260906 | 32,768 |
| 4 | O2 | r0 | 20260907 | 32,768 |
| 5 | O2 | r0 | 20260908 | 32,768 |

- Trainer: exactly 1/64 static total-reward scaling; no other PPO changes.
- Freeze task, reset, observations/actions, base actor, tracking reward,
  references, composition runtime, evaluator and final-checkpoint selection.
- Five independent bounded requests; one shared heavy job; CPU one thread,
  8 GiB and 1,200 s per job. No paid compute or automatic budget extension.
- Data-only UI/report changes after `4ed558e` are not training changes. Verify
  the active GMT adapter and launcher bytes remain identical before using the
  retained raw seed controls; otherwise declare a new comparison.
- Check initial-policy and zero-residual parity within every matched seed/arm.
- No replacement seeds, best-checkpoint selection or task-gate changes.

## Interpretation

- Trainer replication: report the original screen criteria separately for each
  r1 seed, including the same 0.534 compliance floor. Do not pool away a failure.
- Compare each scaled r1 run with its same-seed raw r1 run. Keep full-horizon
  measures separate from measurements on early-fall rollouts.
- Reward contrast: compare scaled r0/r1 within each seed. Primary descriptive
  endpoints: 20 s survival and inside mean-speed target deviation. Report all
  posture, tracking and lateral gates; a speed gain cannot hide a fall.
- This establishes at most a three-seed development contrast at one fixed
  start. It does not replace held-out starts, reference sets or recovery tests.
- O0 controls under the new trainer are still needed before a renewed full
  oracle × reward attribution claim. Do not combine raw and scaled arms into
  one factorial table.
- A longer training budget or depth-reward revision is a subsequent declared
  experiment, not part of these five jobs.

## Pre-observation clarification

- Accepted at 23:45 UTC in mailbox message
  `20260906T234549.004670Z-9410e127b1a94948882fdec7986448cc`:
  adopt scaling for subsequent r1 development only if all three r1 seeds have
  last-16 explained variance >=0.5 and no final-rollout survival regression
  relative to their raw arms. Other screen rows remain reported per seed.
- Initial policy and reward-bearing zero-residual frames/evaluations compare
  against the raw same-seed, same-arm run. The reward-independent trajectory
  additionally matches the shared O2 zero-residual trajectory across arms.
- Seed 07/08 raw controls predate the source-only parity checks. Their reuse
  rests on the seed-06 exact reproduction chain and unchanged GMT adapter and
  launcher bytes through `22b5ee4`; this is not a new raw-seed reproduction.
