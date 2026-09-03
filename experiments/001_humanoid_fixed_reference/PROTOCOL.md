# Frozen reference-tracker baseline protocol

| | Status |
|---|---|
| **progress** | One disposable audited PPO rollout completed at `1,623.85` environment steps/s; no checkpoint or behavioral evidence was retained. |
| **bottleneck** | Peak memory was not measured. Horizontal station keeping and naturalness still lack justified pass/fail thresholds. |
| **next step** | Independently justify broader-study thresholds, review the calibrated budget, then issue a human-reviewed locked design before training. |

## Question and claim boundary

| Item | Frozen definition |
|---|---|
| Question | Can one predetermined PPO policy maintain and track the admitted static Humanoid stand target under the frozen tracker reward? |
| Independent replicate | One complete policy-training seed (`n = 5`) |
| Repeated measure | Evaluation reset seed (`20` episodes per checkpoint); these are not independent trained policies |
| Claim ceiling | Static tracking feasibility only |
| Excluded claim | Numeric reference use, transition quality, recovery, or oracle superiority |

The static command has zero joint position/velocity and constant root posture.
Shuffling or time-shifting it leaves the policy input unchanged. Therefore,
this experiment cannot pass the project’s causal-reference gate.

## Required path

```text
proposed study -> matched resource calibration -> locked study + execution manifest
                  |
                  v
five PPO seeds -> final checkpoint per seed -> 20 paired reset seeds
                                               |
                                               v
               protected evaluator -> five-seed fail-closed summary
```

The current `precalibration-v0` numbers below are proposals. They do not yet
authorize behavioral training. The study design binds both semantic and exact
file hashes for the decision criteria. The execution manifest binds those
criteria again, plus the MuJoCo model, reference, tracking reward, zero task
reward, wrapper, runner, evaluator, execution path, summary implementation,
package versions, and interface shapes. Any mismatch stops the run.

## Proposed pre-calibration design

Source: [`configs/static_stand_precalibration_v0.study.json`](configs/static_stand_precalibration_v0.study.json)

| Component | Value |
|---|---|
| Training seeds | `101, 202, 303, 404, 505` |
| Evaluation seeds | `11001` through `11020`, applied to every checkpoint |
| Training budget | `1,048,576` environment steps per training seed |
| PPO rollout | `4` environments x `2,048` steps; exact `128` rollouts |
| Checkpoint rule | Final-timestep checkpoint only; no best-seed or best-episode selection |
| Episode horizon | `1,000` control steps |
| Policy evaluation | Deterministic actions |
| Reference horizon | `4 x 45` values; actor and critic receive the same flat observation |

The training seed is the analysis unit. Evaluation episodes are paired blocks
within each trained checkpoint. Report all five checkpoints and every planned
evaluation seed, including failures.

## Protected outcomes

The evaluator recomputes metrics from simulator state and exact reference bytes;
it does not trust the generated reward’s component telemetry.

| Outcome | Definition |
|---|---|
| Root tracking | Height RMSE, quaternion geodesic RMSE, linear/angular velocity RMSE |
| Joint tracking | Position/velocity RMSE and worst absolute error |
| Survival | Full `1,000` steps with no collapse sample |
| Collapse | Root height outside `[1.0, 2.0] m` or torso-up alignment `< 0.5` |
| Timing | First collapse step and fraction of steps collapsed |
| Bounded control | Normalized action magnitude/rate and raw plus capacity-normalized actuator torque |
| Contact | All-substep contact counts/forces; prohibited floor-contact count, fraction, step fraction, and peak force |
| Station keeping, descriptive only | Maximum horizontal displacement from direct seeded-reset root qpos; maximum absolute world root velocity x/y/z |
| Naturalness, descriptive only | Allowed-floor peak force and joint jerk RMS/max |

The protected measurements come from direct simulator state, all physics
substep contacts, and the executed policy action. Generated reward telemetry is
not an objective source. Action/rate/torque capacity limits are structural
conformance checks, not naturalness evidence. Horizontal displacement and
per-axis speed thresholds remain absent. The current rule can return only the
named episode-rule/reward-scale-occupancy/control/contact conformance result in
`DECISION_RULE.md`; it is not an objective tracking-feasibility result.

## Resource calibration is not behavior

A short disposable calibration may measure steps/second and peak memory. Its
receipt is permanently labeled `resource_calibration`; its policy/checkpoint is
not eligible for behavioral evaluation or reuse. Before behavioral data exist,
calibration may inform a new version of the proposed architecture or budget.
Once a version is `locked_pre_behavioral`, any change requires a new design ID
or version; silent mutation is prohibited.

The recorded calibration executed `8,192` environment steps in
`5.0448041669988015` seconds on seed `911`, or
`1,623.8489600030387` environment steps/s. Its action-likelihood audit passed,
the model was discarded without serialization, and memory was not measured.
The exact receipt is
[`receipts/2026-09-02_resource_calibration_seed_911.json`](receipts/2026-09-02_resource_calibration_seed_911.json).
A linear throughput projection is about `10.8` minutes per proposed
`1,048,576`-step seed or `53.8` minutes for five serial seeds. This projection
does not include evaluation, startup, contention, memory limits, or nonlinear
slowdown and is planning evidence only.

## Future causal-use experiment (separate design)

A later admitted compound reference must contain meaningful temporal and/or
command variation. In the matched-state intervention, every condition pins one
common ground-truth oracle snapshot set plus the SHA-256 of its exact actor- and
critic-visible policy-input artifact and transformation receipt. Only those
visible command bytes may differ. Then compare these conditions using identical
checkpoints, saved states, policy recurrent state, evaluator, and deterministic
actions:

1. `C_exact`;
2. `C_constant_frame_input` from one declared frame in the same artifact;
3. `C_shuffle_input` from a seeded complete frame permutation; and
4. `C_shift_input` from a frozen nonzero cyclic time shift.

Before evaluation, the harness rejects time-invariant reference matrices. A
causal-use claim requires matched-state action/output sensitivity. Any
common-target behavioral degradation test uses an exogenous schedule. A later
closed-loop comparison instead retains each arm's untransformed arm-local
oracle-result trace after states diverge; it must not claim identical realized
targets. Passing static stand is not a substitute.

## Method provenance

The pre-data separation of randomization, independent replicates, repeated
measures, and blocked evaluation was informed by:

Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026).
*Scientific Agent Skills: A Library of Procedural Knowledge for Research
Agents*. arXiv:2609.00065. https://doi.org/10.48550/arXiv.2609.00065
