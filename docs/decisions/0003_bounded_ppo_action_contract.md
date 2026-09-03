# ADR 0003: bounded PPO action contract

| | Status |
|---|---|
| **progress** | A local tanh-squashed Gaussian policy, one explicit normalized-to-physical affine map, and a mandatory pre-update rollout audit are implemented and source-hashed. |
| **bottleneck** | Behavioral quality remains unmeasured; this decision only closes an executed-versus-stored action mismatch. |
| **next step** | Recalibrate the final source tree, freeze the action contract in the reviewed manifest, then train the static positive control. |

**Status:** accepted before behavioral data  
**Date:** 2026-09-02

## Decision

PPO operates in an exact `[-1, 1]^17` policy action domain using a local
`SquashedDiagGaussianDistribution`. The sampled `tanh` action and its
Jacobian-corrected log probability are stored together. An outer Gymnasium
`RescaleAction` wrapper performs the sole affine map to Humanoid's physical
actuator bounds.

The runtime fingerprint binds:

- the normalized and physical action Boxes;
- the policy identifier and exact policy source bytes;
- the transform identifier, wrapper topology, dependency lock, and runner;
- endpoint probes `-1, 0, +1` against MuJoCo `data.ctrl`.

Every calibration and training rollout is checked before PPO updates its
parameters. The callback rejects non-finite or boundary-rounded stored actions,
recomputes their Jacobian-corrected log probabilities under the unchanged
policy, and requires every expected rollout to be audited. Receipts record the
rollout count, largest stored action magnitude, and largest recomputation
error. Evaluation refuses a checkpoint receipt with missing or inconsistent
audit evidence.

A real four-step Humanoid test additionally requires:

1. the rollout-buffer action to equal the normalized action received by the
   environment;
2. physical controls to equal the declared affine map;
3. the stored log probability to equal a pre-update recomputation; and
4. save/load to preserve the policy class, distribution, flag, and
   deterministic action.

## Rejected alternative

Native SB3 PPO samples an unbounded Gaussian and clips a copy before stepping a
Box environment while retaining the unclipped sample for the update. On the
initial Humanoid policy, an audit found most scalar samples outside the narrow
physical `[-0.4, 0.4]` bounds. Reducing initial standard deviation would lower
the rate temporarily but would not create an invariant as the learned mean and
standard deviation change.

## Claim boundary

This decision establishes coherent action and likelihood semantics. It does
not show that PPO learns balance, tracking, transitions, recovery, or natural
motion.
