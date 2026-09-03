# Scientific boundary

## Truth table

| Item | Current status | What would advance it |
|---|---|---|
| Oracle contracts | Implemented and covered by positive/negative tests | Integration with a measured tracker and oracle study |
| Gymnasium substrate | Real `Humanoid-v5` reset/step receipt passes | Preserve the exact runtime fingerprint in each run |
| Controller-native references | Exact 45D ABI plus immutable static-stand candidate | Admit time-varying, dynamically feasible motion bytes |
| Frozen reference tracker | Wrapper/reward candidate and protected measurements exist; no checkpoint or objective success threshold | Independently justified task thresholds, then trained checkpoints with full provenance |
| Protected evaluator | Direct tracking, worst-joint, action/rate, torque/capacity, all-substep contact/impact, seeded-reset-relative horizontal displacement, per-axis world root speed, and jerk metrics implemented | Independently justify broader station-keeping and naturalness thresholds |
| Causal reference use | Unmeasured | Exact/zero/shuffle/time-shift ablations |
| Phase-aware composition | Research target | Matched manual oracle comparison |
| Generated oracle | Not implemented | Protected evaluator first, then bounded generation |
| PRAXIST integration | Installed externally; Codex-native doctor passes | Runnable baseline and protected evaluator canary |
| Humanoid improvement | No evidence | Held-out matched results across predetermined seeds |

## What stock Gymnasium provides

- A 17-actuator MuJoCo humanoid model and stable environment API.
- `Humanoid-v5`, whose default objective is forward locomotion.
- `HumanoidStandup-v5`, whose default objective emphasizes rising height.

Neither environment supplies a reference-conditioned tracker, motion library,
phase estimator, transition oracle, or independent oracle evaluator. Those are
research components that must be created and validated here.

## Admission rule

Label a run `oracle_training` only when its receipt proves:

1. Exact robot, ordered reference features, units, frame, cadence, horizon, and
   normalizers.
2. Controller-compatible reference bytes and simulator admission evidence.
3. Immutable tracker, tracking-reward, task-reward, trainer, and evaluator IDs.
4. Actor and critic receive the exact declared oracle observation.
5. Train and evaluation consume the same immutable execution manifest.
6. All oracle arms share seeds, budgets, reset distribution, reward, and
   checkpoint rule.
7. Causal reference-use ablations passed before composition claims are made.

Otherwise label the run `interface_check` or `exploratory`, with no oracle
quality claim.

The current receipt checker can issue only `interface_check` or
`structural_candidate`. It deliberately cannot promote a run to
`oracle_training`: trusted digest recomputation, run-ledger verification, and
causal-reference replay are not implemented yet.

## Meaning of "works well"

The primary endpoint is sealed-test end-to-end completion. The following remain
hard guardrails:

- nominal-mode success and tracking error;
- transition success and maximum boundary error;
- falls, forbidden contacts, and time to failure;
- phase error, resynchronization latency, stalls, chattering, and loops;
- same-episode recovery and task-resumption rate;
- contact impulse, jerk, torque/energy, and compute.

Training reward is diagnostic only. It cannot establish success.
