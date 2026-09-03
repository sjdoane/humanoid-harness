# Frozen prebehavioral-candidate static positive-control decision rule

| | Status |
|---|---|
| **progress** | The episode rule, reward-scale-derived occupancy bounds, bounded-control/prohibited-contact conformance, checkpoint counts, and uncertainty summary are frozen as a prebehavioral candidate; station-keeping measurements are protected and descriptive. |
| **bottleneck** | Station keeping and naturalness lack independently justified pre-data thresholds. Static-stand feasibility therefore cannot pass. |
| **next step** | Independently justify displacement, axis-speed, impact, and jerk thresholds before locking a broader study version. |

## Units of analysis

| Level | Role |
|---|---|
| Training seed | Independent replicate (`n = 5`) |
| Evaluation seed | Paired repeated measure (`20` per checkpoint) |
| Checkpoint | One predetermined final checkpoint per training seed |

All five checkpoint receipts and sibling checkpoint files are reopened and
rehashed. Every checkpoint must contain all 20 evaluation seeds in declared
order. Evaluation episodes never increase the independent sample size beyond
five.

## Coarse nonclaim conformance

An episode conforms only when it:

- completes all 1,000 steps with zero collapse samples;
- stays below the six reward-scale RMSE limits: `0.20 m`, `0.50 rad`,
  `1.0 m/s`, `2.0 rad/s`, `0.35 rad`, and `2.0 rad/s`; and
- keeps the worst joint-position and joint-velocity errors below `0.35 rad`
  and `2.0 rad/s`.

It must also meet exact structural-conformance limits:

- normalized action RMS/max `<= 1`;
- normalized action-rate RMS/max `<= 2 / 0.015 s`;
- raw Humanoid-v5 torque RMS/max `<= 56.0461994303649 N*m` and `120 N*m`;
- per-joint capacity-normalized torque RMS/max `<= 1`; and
- forbidden floor-contact count, contact fraction, step fraction, and peak
  force all equal zero. Only `left_foot` and `right_foot` may contact the floor.

The action, action-rate, and torque limits are theoretical runtime/ABI bounds.
They verify bounded-control conformance; they do not distinguish smooth,
efficient, or natural control. The prohibited-contact rule is behavior
discriminating.

A checkpoint conforms when at least 19 of 20 paired episodes meet every rule.
The machine-readable result
`coarse_episode_rule_reward_scale_occupancy_bounded_control_and_prohibited_floor_contact_conformance`
is true when at least four of five independent checkpoints conform. The
reward-derived scales are occupancy bounds chosen from the optimized reward;
they are not independent task-success thresholds and cannot establish
objective tracking feasibility.

## Broader decision remains blocked

The evaluator records maximum horizontal displacement from the direct seeded-
reset root position and maximum absolute world-frame root speed on each axis.
These remain descriptive because no independent support-region or axis-speed
basis yet justifies pass/fail thresholds. Allowed-foot peak force and joint
jerk are also recorded, but no independent neutral-controller calibration
supports naturalness thresholds.

Therefore every summary emits:

```text
decision_status = blocked_missing_independently_justified_station_keeping_and_naturalness_thresholds
static_stand_feasibility_pass = null
study_pass = null
automatic_promotion = false
```

The criteria file binds the measured displacement and per-axis speed fields
without inventing maxima. Impact and jerk remain required descriptive
measurements. Neither the explicitly limited conformance result nor a pipeline-
complete receipt means the controller “works well.”

## Uncertainty and interpretation

Evaluation episodes are reduced within each training seed. The aggregator then
enumerates all `5^5 = 3,125` nonparametric bootstrap resamples of the five
seed-level values and reports a descriptive 95% percentile interval. This is a
seed-sensitivity summary, not a hypothesis test. It does not repair the small
independent sample or tune thresholds.

No p-values, optional stopping, best-seed selection, automatic promotion,
station-keeping claim, naturalness claim, causal-reference-use claim,
transition claim, recovery claim, or oracle-superiority claim is permitted.
