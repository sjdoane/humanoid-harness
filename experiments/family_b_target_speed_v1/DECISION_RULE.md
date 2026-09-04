# Family B target-speed v1 decision rule

| status | current truth |
|---|---|
| progress | The proposed decision rule fixes completeness, baseline viability, endpoint reporting, and hard guardrails before any Family-B behavior is observed. |
| bottleneck | No candidate has an outside-sandbox all-pass Seatbelt receipt, and no baseline or candidate has been trained or evaluated. |
| next step | Fable reviews and locks a byte-identical protocol/config/decision set only after every Seatbelt canary passes outside the builder sandbox. |

**Status:** proposed, not locked  
**Evidence label:** `exploratory_reward_cycle`  
**Margin status:** exploratory relative-harm guardrail, not a validated
naturalness threshold

## Eligibility gates

An arm is ineligible before training if any of these is absent or invalid:

1. exact candidate source bytes and static-validation receipt;
2. for `candidate_rk`, target-speed axiom and computed affine-scale receipts;
3. reward identity binding source, schema/read set, target, affine parameters,
   compositor, Gym source, XML, and dependency hashes;
4. an all-`passed` Seatbelt/runtime canary receipt from the execution host;
5. a frozen execution manifest whose config, protocol, decision rule, runtime,
   sources, spaces, seeds, budget, and checkpoint rule reverify; and
6. a reward-enabled resource calibration projecting within the `150 min` CPU
   reservation.

`stock_r0` and `manual_target_speed_v1` are exact frozen baselines, not
candidate scale-calibration outputs. Any static/runtime/envelope failure still
invalidates them. No failed arm receives fallback values or a substitute run.

## Baseline viability stop

Before any candidate training, the complete five-checkpoint `stock_r0` result
must satisfy both:

- at least `19/20` full-survival episodes in at least `4/5` checkpoints; and
- at least `19/20` zero-non-foot-floor-contact episodes in at least `4/5`
  checkpoints.

If either fails, stop the family before candidate training. The premise that
the stock locomotion MDP is already solved did not survive its declared check.

## Complete result set

A comparison requires final-timestep checkpoints for all training seeds
`101, 202, 303, 404, 505` and, for each checkpoint, ordered evaluation resets
`11001..11020`. Preserve failures and missing artifacts. Do not retry, impute,
drop, reorder, replace, or choose a better seed/checkpoint/episode.

## Endpoint report

For arm `a` and training seed `s`, compute the mean endpoint over its 20 paired
resets, `mu[a,s]`. Report the five paired differences
`mu[candidate,s] - mu[stock,s]` and their arithmetic mean. This descriptive
effect is not itself a pass gate and its sign is not an improvement claim.

The first cycle is target `1.0 m/s` only. `stock_r0` is target-unaware, so the
cycle is an exploratory loop-mechanics test. From cycle 2,
`manual_target_speed_v1` is a third arm evaluated with the same rules.

## Hard guardrail decision

For each training seed and each metric `m` in torque RMS, energy, and action-rate
RMS, let `Q[m,a,s]` be the median over all 20 episodes. A premature episode is
`+inf` only for this gate; retain its finite partial measurement in raw data.

\[
rho_{m,s}=\begin{cases}
Q_{m,k,s}/Q_{m,0,s} & Q_{m,0,s}>0\\
1 & Q_{m,k,s}=Q_{m,0,s}=0\\
+\infty & otherwise.
\end{cases}
\]

Every metric must satisfy both:

1. `rho[m,s] <= 1.10` for at least four of five paired training seeds; and
2. `sum_s Q[m,candidate,s] / sum_s Q[m,stock,s] <= 1.10` under the same
   zero-denominator rule.

The candidate must also satisfy:

- at least `19/20` full-survival episodes in at least `4/5` checkpoints, with
  its qualifying count never below paired `stock_r0`; and
- at least `19/20` zero-non-foot-floor-contact episodes in at least `4/5`
  checkpoints, with its qualifying count never below paired `stock_r0`.

Structural action, action-rate, torque-capacity, torque/action agreement,
contact-capture completeness, finite-value, lineage, and replay checks are hard
receipt-validity conditions. One breach invalidates the episode and comparison;
it cannot be traded against endpoint performance.

Allowed-foot peak force and impulse remain descriptive. This rule supplies no
undeclared gait-style or naturalness threshold.

## Outcomes

| outcome | exact meaning |
|---|---|
| `invalid` | An identity, completeness, structural, sandbox, or protected-data gate failed. Preserve the failure; do not score or substitute it. |
| `stop_family_baseline_not_viable` | The complete `stock_r0` survival or non-foot study rule failed before candidate training. |
| `fails_exploratory_guardrails` | Complete candidate evidence exists but one or more relative, survival, or non-foot gates failed. |
| `eligible_for_review` | Complete evidence exists and every hard guardrail passed. Report endpoint differences and all diagnostics for human review. |

`eligible_for_review` is not acceptance, deployment, scientific confirmation,
or evidence that the reward improved locomotion. Any confirmatory claim needs
the charter's sealed formal-study gate plus the required reward-scale and
term-removal controls.
