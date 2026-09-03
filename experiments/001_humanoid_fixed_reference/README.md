# Experiment 001: frozen reference-tracker baseline

| | Status |
|---|---|
| **progress** | The pinned substrate, tracker, evaluator, and five-seed path execute fail closed; one disposable audited PPO rollout measured `1,623.85` environment steps/s. |
| **bottleneck** | Peak memory and independently justified station-keeping/naturalness thresholds remain missing; a static command cannot establish causal reference use. |
| **next step** | Review the calibrated budget, complete the broader pre-data criteria, freeze a human-reviewed execution manifest, and only then train predetermined seeds. |

## Purpose

Establish static-stand tracker feasibility under a frozen target and reward.
This is a tracker-admission prerequisite, not a causal-reference or oracle
study.

## Required sequence

1. Freeze the Humanoid model and environment manifest.
2. Define ordered reference features, units, frames, cadence, horizon, and
   normalizers.
3. Establish static-stand tracking feasibility; this is not a dynamics
   certificate for a motion library.
Experiment 002 separately admits a time-varying reference, freezes its tracker,
and runs exact, zero-input, shuffled, and time-shifted interventions.
Composition studies remain blocked until that distinct causal-use gate passes.

## Current evidence

[`receipts/2026-09-02_substrate_smoke.json`](receipts/2026-09-02_substrate_smoke.json)
records one real reset and zero-action step. This proves simulator availability
and interface dimensions only. It is not a tracker or oracle result.

[`receipts/2026-09-02_resource_calibration_seed_911.json`](receipts/2026-09-02_resource_calibration_seed_911.json)
records one complete proposed PPO rollout: `8,192` environment steps in
`5.0448041669988015` seconds (`1,623.8489600030387` steps/s). The action-likelihood
audit passed. The policy was discarded without serialization, memory was not
measured, and the receipt explicitly carries no behavioral claim or evaluation
eligibility.

The static wrapper also has a real-simulator negative test: lowering root height
to `0.2 m` while leaving every joint on target keeps the fixed tracking score
below `0.03`; moving back toward the target raises it. This closes the known
high-reward fallen-pose loophole without making the distant reward numerically
flat, but it does not show that PPO can learn standing or recovery.

`DECISION_RULE.md` defines the current claim boundary. Even a fully passing
coarse result leaves `static_stand_feasibility_pass` and generic `study_pass`
unset. Horizontal displacement and per-axis speed are protected descriptive
measurements, but their pass/fail thresholds—and impact/smoothness thresholds—
still need independent pre-data support.
