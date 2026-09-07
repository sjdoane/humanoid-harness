# G1 training telemetry and reward units

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

## Opt-in training reward preconditioning

The legacy config omits `trainer`; that path, manifest contract, and v1 telemetry serialization
remain unchanged. The original opt-in profile is:

```json
{"schema_id":"gmt_g1_total_training_reward_preconditioning/v1","schema_version":1,"total_training_reward_scale":0.015625}
```

One versioned optimization hypothesis changes only PPO's fixed learning rate:

```json
{"learning_rate":0.00003,"schema_id":"gmt_g1_scaled_ppo_training_profile/v2","schema_version":2,"total_training_reward_scale":0.015625}
```

The v2 profile admits only `3e-5`; it is not a free-form sweep interface. Its matched control is
the v1 profile at `3e-4`, with reward scale `1/64`, policy initialization, task, oracle, base
controller, evaluator, budget, seed, and final-checkpoint rule held fixed. This profile is an
untested optimization hypothesis, not a bug fix or evidence of improved learning.

The training-only wrapper sends `float32(raw total reward / 64)` to PPO. Evaluation remains on
the unwrapped raw environment. Reward scaling does not change task-reward weights, reward
components, observations, actions, termination, the base actor, or checkpoint selection. The v1
profile keeps the original PPO hyperparameters; v2 differs only in its declared learning rate.
Timeout bootstrap values share the scaled PPO value units.

Opt-in runs emit `training_telemetry_v2.jsonl`. Complete-episode summaries use the literal fields
`scaled_environment_returns` and `raw_environment_returns`. Both exclude the later
PPO timeout bootstrap; the latter are accumulated from retained
unscaled step info, not reconstructed by multiplying quantized scaled values. Value loss is in
scaled optimization-reward units, so its magnitude is not comparable with raw-reward runs.
Both opt-in profiles emit the same scaled-reward telemetry schema. The config, effective trainer
identity, reward-unit metadata, v2 descriptor, and v2 output must appear as one exact validated
set. Proposals preserve the parent `trainer` field and cannot edit it.
