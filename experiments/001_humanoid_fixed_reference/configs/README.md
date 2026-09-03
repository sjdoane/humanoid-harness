# Configuration boundary

`static_stand_precalibration_v0.study.json` is a proposed pre-data design, not a
locked budget. One machine-matched rollout has been calibrated with its exact
network, optimizer, and PPO settings; the audited receipt is
[`../receipts/2026-09-02_resource_calibration_seed_911.json`](../receipts/2026-09-02_resource_calibration_seed_911.json).
Peak memory was not measured, and the result is resource-planning evidence
only. Human review is still required before the design is copied to a new
`locked_pre_behavioral` version. The proposed file deliberately does **not**
serve as an execution manifest.

Manifest-candidate emission and human finalization both require the exact
calibrated proposed-design path and its calibration receipt. The gate permits
only `design_version` and `design_status` to change, re-inspects the current
runtime, revalidates the receipt and action-likelihood audit, and binds semantic
and file SHA-256 identities for both inputs into the frozen manifest.

The proposed design already binds the semantic SHA-256 and exact file SHA-256
of `static_positive_control_criteria_v1.json`. Those criteria permit only the
named episode-rule/reward-scale-occupancy plus bounded-control/prohibited-floor-
contact conformance result. Reward-derived scales are not independent objective
tracking thresholds. Station-keeping measurements are descriptive;
station-keeping and naturalness decisions remain explicitly blocked by missing
independently justified thresholds. Any criteria change requires new design
bytes and review before behavioral data.

A run additionally requires a reviewed `FrozenExecutionManifest` containing
the exact criteria identities plus runtime, model, reference, reward, wrapper,
evaluator, runner, execution, and study-summary hashes. No execution manifest
is committed yet because the tracker slice has not been frozen. The runner must
fail closed until both a locked design and a matching execution manifest exist.

Resource calibration uses a separate `resource_calibration` receipt. Before
behavioral data, it may motivate a new design version with a different proposed
budget or architecture. After a version is locked, any such change requires a
new design version. Calibration never supports a claim about standing,
tracking, reference use, or oracle quality.
