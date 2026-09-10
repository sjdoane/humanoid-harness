# Current research status

| progress | GMT/G1 exercises the oracle/reward revision loop; 38 retained training runs report 2,424,832 training transitions. |
|---|---|
| bottleneck | No full-task pass, VIBE integration or measured manual-versus-harness efficiency advantage. |
| next step | Confirm the supplied lab interface and loaded-cart assets, then bind task outcomes and complete tuning costs to one bounded comparison. |

**Reviewed September 10, 2026.** This update consolidates existing work; it adds
no new training or behavioral result.

## What exists

- **GMT/G1 adapter:** supplied reference tracker, guarded motion selection,
  learned residual actions, fixed tracking reward and separate task reward.
- **Revision loop:** externally coordinated LLM proposals, data-only candidate
  validation, immutable artifact identities, training/probes and independent
  scoring. Accepted inputs do not imply accepted behavior.
- **Feedback:** verified state/action/reference traces, boundary diagnostics and
  a read-only evidence UI. No VLM-based visual diagnosis is claimed implemented.
- **Effort command:** counts training transitions and run duration from selected
  completed receipts. Failed jobs, full search cost, human time and model cost
  are not yet fully accounted for.

## Measured evidence

| Record | Result | What it does not establish |
|---|---|---|
| [G1 development record](strategy/astra/G1_LEARNING_RESULTS_20260906.md) | 38 training runs; real reward and oracle revisions executed; no full-task pass | Success on the semester tasks or learning-efficiency gains |
| [Study 009](../experiments/009_g1_reference_input/RESULTS.md) | Two five-arm reference interventions alter actions and trajectories; exact controls reproduce | Good composition or generalization |
| [Study 014](../experiments/014_g1_finite_depth_reward/RESULTS.md) | LLM reward revision trained for 131,072 transitions; failed its depth prediction | Successful reward improvement |
| [Study 018](../experiments/018_g1_four_state_reward_loop/RESULTS.md) | Trained baseline survives 20 s; 7/11 gates pass; reward arm B not launched | Full-task success or a completed reward comparison |
| [Study 020](../experiments/020_g1_saved_policy_diagnostic/RESULTS.md) | Initial policy falls in 5/16 sampled episodes; trained policy in 11/16; neither completes the task | Generalization across training seeds: these are paired noise sequences at one reset |
| [Study 021](../experiments/021_g1_immediate_crouch_exit/PROTOCOL.md) | Immediate-exit scorer and protocol prepared; execution deferred | A candidate rollout or training result |

The 2,424,832 figure counts **training transitions**, not total simulator steps
or complete search cost. Later negative results qualify the earlier successful
survival videos. Survival, reference use and task completion are different claims.

## Semester target and evaluation

- Tasks: **volleyball, floor hockey, loaded-cart interception, basketball**.
  Basketball replaces the changing obstacle course; historical posture-course
  studies remain unchanged.
- Compare the initial policy, post-training with fixed guidance, manual
  revision and harness revision. Match training budgets and available feedback.
- Report independent task success, steps/revisions to a predefined target, and
  human interventions and active tuning time. Include failures and unknown costs.
- Sim-to-real work depends on lab approval, hardware and successful simulation
  checks. None is reported here as completed.

See the [charter](PROJECT_CHARTER.md) for task rationale and the
[implementation plan](strategy/astra/POST_TRAINING_PLAN_20260908.md) for dependencies.

## What a collaborator can reproduce today

- **From Git:** inspect code, study protocols/results, source records, the
  literature query index, and portable software tests.
- **With separately admitted local assets:** replay registered G1 evidence and
  run source-bound experiments. Reports contain historical host paths and
  hashes; those are provenance, not portable download instructions.
- **Not bundled:** raw training runs, motion/controller payloads, videos, private
  lab code or meeting transcripts. There is not yet a verified clean-clone G1
  training/demo recipe.
- **Test status:** the whole suite has unresolved host/artifact dependencies.
  See [current verification](operations/SHARE_READINESS_20260910.md); a passing
  portable subset is not a passing release or new robotics evidence.
