# G1 trainer conditioning — development screen

| status | evidence |
|---|---|
| progress | The raw O2r1 control has exact policy/trajectory reproduction with measurement-only telemetry. |
| bottleneck | Value prediction explains almost none of the training-target variance; task gates remain unmet. |
| next step | Compare raw training reward with a fixed positive scale, then decide whether a longer budget is justified. |

## Factor and controls

- Factor: multiply the total scalar reward entering PPO by exactly `1/64`.
- No offset, clipping, running statistics, observation normalization or extra reward term.
- Keep the physical environment and raw task/tracking rewards unchanged.
- Scaling is a trainer setting. Oracle/reward proposals cannot author it.
- Freeze O2/r1, seed `20260906`, 32,768 transitions, final checkpoint only;
  references, base actor, residual limit, observations, dynamics, resets, PPO
  hyperparameters and independent evaluator remain unchanged.
- Raw and scaled arms use the same reviewed source. The raw arm must reproduce
  all nine original O2r1 output artifacts before interpreting the scaled arm.
- The scaled arm must retain byte-identical zero-residual trajectory, frames,
  evaluation and initial policy. Pin its exact factor and trainer identity in
  the manifest and exact-request receipt, not just a display label.
- One local worker at a time; each exact request retains its 1,200 s, 8 GiB limit.
- This is a development comparison, not held-out evaluation or a reward-knob study.

## Why this scale

- Raw reward is roughly 4 per control step. With discount `0.99`, a constant
  reward of that size has discounted sum near 400; `1/64` brings it near 6.25.
- A positive constant preserves the mathematical ranking of discounted reward
  objectives. It changes numerical conditioning, not the robot's task.
- Actor/critic MLPs are separate. Global gradient clipping couples parameters,
  but scalar loss ratios do not establish gradient dominance.
- Dynamic reward normalization is deferred to avoid adding policy-dependent
  running statistics to this single-factor test.
- Basis: [SB3 custom-environment guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
  and the installed SB3 PPO implementation, pinned by the runtime receipt.

## Predictions frozen before the scaled run

| question | decision rule |
|---|---|
| Does value prediction improve? | Mean explained variance over the last 16 updates is at least 0.5. Report the full curve. |
| Is the gait retained? | Full 20 s without falling, both state transitions, existing joint and roll/pitch gates pass. |
| Does task behavior regress? | Compliance at least 0.534; overall speed MAE at most 0.35 m/s; lateral max at most 2.970430 m. |
| Is there a task-level gain? | Report every independent task gate and whether a previously failing gate changes to pass; descriptive at this screen. |
| May this trainer proceed to a longer-budget comparison? | The mechanism, gait and no-regression rows pass. A gate flip is not required to test the remaining budget hypothesis. Otherwise retain the raw trainer while diagnosing the result. |

- Value-loss magnitudes have different units across arms. Smaller scaled loss
  is expected mechanically and is not an improvement criterion.
- KL, clipping fraction, policy standard deviation and attempted epochs are
  diagnostics, not proofs of gradient ownership or convergence.
- An attempted epoch can stop partway through. Do not label this counter as
  completed epochs or optimizer minibatch steps.
- The full task gate is unchanged. A trainer-screen pass is not task success.
- Proceed-rule clarification accepted from Fable's review before either new
  arm ran: improved value prediction with retained gait justifies testing a
  longer budget, even without an immediate task-gate flip.
- Failure of scaling does not uniquely identify capacity, learning rate, reward
  design or observations as the cause; retain competing explanations.

## Follow-on boundary

- A larger budget is a separately declared family, not an implicit retry.
- Predetermined replication seeds are `20260907` and `20260908`; no replacement
  seeds or intermediate-checkpoint selection.
- Re-establish oracle/reward contrasts under any accepted trainer before
  claiming that either authoring knob improved task behavior.
- Retain every request, candidate, final policy, trace, telemetry file and verdict.
