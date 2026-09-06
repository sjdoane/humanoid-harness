# G1 training telemetry v1

| status | evidence boundary |
|---|---|
| progress | The fixed course worker writes bounded, numeric PPO and episode measurements. |
| bottleneck | Telemetry describes optimization and development episodes; it does not establish convergence or task success. |
| next step | Review one fresh matched-control run before declaring a larger-budget experiment family. |

`training_telemetry_v1.jsonl` is measurement-only. It does not change the environment,
reward, observations, policy calls, PPO hyperparameters, or checkpoint selection. The callback
summarizes values already held by Stable-Baselines3 and its rollout buffer. It makes no RNG call
and performs no extra policy forward pass.

Each 512-transition rollout record contains:

- moments for the fixed base, task, and phase observation groups;
- complete episode return and length, fall/horizon counts, and terminal task metrics;
- the **previous** PPO update's available logger values. The first record therefore uses `null`.

A final record flushes the PPO update through the full transition budget. `sb3_n_updates` counts
attempted PPO epochs, including an epoch that stopped partway through because of the KL limit; it
is not labeled as optimizer minibatch steps. Unavailable logger values remain `null` with a
`missing` reason. Undefined non-finite explained variance is `null` with reason
`undefined_nonfinite`; other non-finite optimization statistics fail the run.

The run manifest binds the telemetry path, SHA-256, byte size, record counts, and boundary
semantics under `training.telemetry`. The output ledger must contain the matching filename and
digest. Historical manifests without either member retain their original validation path; a
descriptor or file appearing alone is rejected. The JSONL is capped at 4 MiB.

These records support diagnosis of optimization behavior and episode termination. They are not
held-out evidence, a universal convergence test, or authority to change the frozen task, MDP,
trainer family, or evaluation gates.
