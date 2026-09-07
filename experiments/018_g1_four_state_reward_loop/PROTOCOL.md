# Four-state composition with one feedback-driven reward revision

| progress | Native O7b survives a four-stage zero-residual composition; Study017's derived replacement is rejected. |
|---|---|
| bottleneck | Native O7b has not been trained under the four-state finite-horizon runtime. |
| next step | Train one baseline A; admit one reward-only revision B only if A passes the substrate and headroom checks. |

## Question and frozen boundary

- Can one feedback-linked task-reward revision reduce lateral drift without
  degrading a trained, surviving walk → crouch → rise → walk composition?
- Astra leads. Fable reviews and proposes one data-only reward revision from
  A's verified feedback. No parallel Fable training or candidate sweep.
- Native O7b is a starting configuration, not an adopted successful policy.
  Preserve its exact assets, segments, oracle, task and original r1 recipe.
  Starting config: `ba45dba36ef88bdc522ed2115e46a4b3d876e00a627a089fd56ddf0de623e18a`.
- New study runtime: schema5 `gmt_g1_four_state_finite_horizon_course/v1`;
  exact before/inside/rise/after, 2,172 observations, intrinsic horizon termination,
  heading-reference feedback **off**. Old loop profile remains probe-only.
- Both A and B use seed20260906, 131,072 transitions, trainer profile3,
  learning rate0.00003, fixed base-observation normalizer, static reward scale1/64,
  same initialization and final checkpoint only. No checkpoint or seed selection.
- Freeze dynamics/reset, base tracker, action bounds, tracking reward, task,
  evaluation, references/oracle and trainer. B changes only the admitted reward
  recipe. Bind total observation width through frozen_runtime, not trainer ID alone.
- Telemetry v4 binds the runtime and base/task/phase widths2154/11/7. Existing
  episode/optimizer summaries are diagnostics. A per-phase height histogram is
  not currently available; no exposure-core import is required for this study.
- No derived-reference training admission, new reference continuation, preview
  mechanism, scene change or paid compute belongs to this comparison.

```text
INPUT: native O7b references/oracle + task + r1
              ↓
      train A → independent evaluator → verified feedback → one LLM reward rB
                                                               ↓
      train B with same frozen components and budget → evaluator → compare
OUTPUT: both final policies, synchronized traces, reward lineage, scores, replay
```

## Sequential gates — lock before A

| A admission to reward revision | Requirement |
|---|---|
| Survival | 20 s; no fall or recorded non-foot ground contact |
| Composition | Exactly three switches; all four modes execute |
| Physical task | Region entry/exit and finish observed; no after-mode region re-entry |
| Exposure / posture | At least25 physical-region samples; compliance≥0.75 |
| Tracking | Joint p95≤0.35 rad; roll-pitch p95≤0.25 rad |
| Speed | Overall MAE≤0.35 m/s; inside mean-speed deviation≤0.10 m/s |
| Lateral headroom | Maximum lateral error>0.75 m |

- Stop B if any row fails. Keep A and diagnose; do not relax gates after seeing A.
- A's zero-residual numeric trajectory/actions must reproduce retained O7b.
  Explicit runtime/reset termination metadata may differ; independent objective
  values must not. A failed parity check is not a behavioral experiment result.
- If A already meets the lateral task gate, this proposed reward target lacks
  headroom. Any subsequent depth or other revision needs a separately declared study.

## Candidate B — lock proposal before its data

- Rebuild A's feedback from immutable raw states. Retain the actual LLM proposal,
  rationale, prediction and falsifier; publish through `g1 revise`.
- Exactly one reward recipe, restricted to lateral/heading weight revision.
  Keep recipe version, all other weights and component formulas unchanged.
- P1: B whole-run maximum lateral error≤0.70×A's.
- P2: B maximum absolute unwrapped reset-relative heading during executed after
  mode≤A's. Unwrap the entire reset-relative heading sequence first, then select
  post-step samples whose executed mode is after; never reset unwrapping at entry.
- B must retain A's substrate gates, except the headroom requirement is not a
  B guardrail. Posture compliance must also be≥A's, without a rounding tolerance.
- Report every original task gate unchanged. Depth is measured, not a predicted
  improvement or a claim of physical impossibility. A study-screen pass is not
  full-task qualification, held-out performance or generalization evidence.
- Report after-entry phase/heading, training falls/horizon completions, critic
  and optimizer diagnostics. Coupled behavior may improve one error while
  worsening another; the generated reward never grades itself.

## Execution and integrity

- Before A: commit/review scorer and runtime repairs; seal exact source tree,
  scorer/protocol/config bytes, native assets and initial base state.
- Before B: separate successful A-verification receipt and exact proposal seal.
  Verification and launch occur in separate successful tool calls.
- One heavy local job at a time. Each run requires its own accepted native v2
  reservation, CPU/wall limit1,200 s, memory8 GiB, and immutable fresh output.
- A and B share the exact executable tree. If it must change, end the study
  before another launch and declare new authority; never silently mix runtimes.
- Verify equal initial residual-policy bytes, zero-residual numeric trajectory,
  every non-reward zero-trace field, and independent zero objective. Recipe hash
  and reconstructed lateral/heading reward contributions may differ.
- Both final traces must rebuild independent feedback and active reward totals.
  Resource/manifest/config/telemetry/policy linkage must verify before scoring.
- No automatic replication, stronger reward, extra budget or oracle patch after
  rejection. Retain all outcomes and choose the next bounded comparison explicitly.
