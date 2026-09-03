# Experiment 002: executable oracle contract — draft

| | Status |
|---|---|
| **progress** | The repository has a typed oracle runtime, an exact 45D Humanoid reference ABI, a frozen-tracker candidate, and a reward-independent evaluator. This document specifies the next integration boundary. |
| **bottleneck** | No time-varying reference has simulator admission evidence, the oracle is not installed in the Gymnasium observation/reward path, and causal reference use has not been measured. |
| **next step** | Approve or revise this draft, admit one time-varying compound reference, implement the wrapper and receipts, and pass the real-Gym gates before locking a behavioral design. |

**Status:** pre-data design draft; not approved, implemented, locked, or
authorizing. It records no behavioral result and raises no oracle-quality claim.

## Purpose and claim order

Experiment 002 should establish two prerequisites in this order:

1. the typed oracle executes with explicit timing and reaches both actor and
   critic through one immutable observation ABI; and
2. a frozen tracker changes its actions and protected tracking behavior when
   the numeric reference window changes.

This experiment does not compare a generated oracle with a manual oracle.
Generated-versus-manual composition belongs to a later matched experiment after
both prerequisites pass.

## Architecture and timing

```text
admitted compound reference R* ------------------------+
observable state adapter s[t] ----------------------+   |
                                                     v   v
                                      typed oracle O_k          future authorable
                              W_gt[t], mode, phase, clocks, masks
                                                     |
                                   causal-input transform A_j    evaluation only
                                                     |
                                                     v
Humanoid base obs x[t] --> frozen assembler --> actor + critic --> action a[t]
                              |                               |          |
                              |                               |          v
                              |                               |   frozen Humanoid MDP
                              |                               |          |
                              +-- exact installed-input receipt          v
                                                              state x[t+1]
                                                                       |
cached arm-local untransformed W_gt,j[t][1] -------------------> protected evaluator

Frozen feedback path: x[t+1] -> observable state adapter -> next oracle query.
The evaluator never reads a generated reward component or a perturbed policy target.
In a closed-loop comparison, different actions may produce different next states and
therefore different arm-local untransformed oracle targets. Common-target causal
interventions are performed only at matched saved states or with an explicitly
exogenous open-loop schedule; they are not inferred from divergent rollouts.
```

One control step has this exact order:

| Boundary | Required operation |
|---|---|
| Reset | Reset the plant and oracle; perform the `t = 0` query from reset-visible signals; cache `O[0]`; expose its policy-input form. |
| State `x[t]`, `t > 0` | Query the oracle exactly once from signals observable at `x[t]`; cache the untransformed result `O[t]` and reference window `W_gt[t]`. |
| Action | Assemble the policy observation from `x[t]` and the selected input-arm transform of `O[t]`; actor and critic receive the same bytes. |
| Post-step `x[t+1]` | Grade direct simulator state against cached `W_gt[t][1]`, which was visible before `a[t]`; do not query a new oracle before grading. |
| Next observation | Query `O[t+1]` from `x[t+1]`, including the final returned terminal/truncated observation, then assemble that observation. |

`H >= 2` is mandatory. Transition steps use the same rule: a post-step state is
never graded against a target selected retroactively from `x[t+1]`. End-of-span
and terminal holds must be explicit in the window mask and receipt.
Query index `0` belongs to reset. An episode with `N` executed actions emits
exactly `N + 1` oracle results, including the observation returned after the
final action. For the `1,000`-step `TimeLimit`, `max_queries` is exactly `1,001`.
An early plant termination emits its final query and then stops. No query occurs
after the terminal/truncated observation has been returned.

## Exact policy observation ABI

Use one finite, flat `float32` vector in this order:

| Slice | Shape | Meaning |
|---|---:|---|
| `humanoid_base` | `348` | Exact frozen `Humanoid-v5` policy observation. |
| `reference_window` | `H x 45` flattened row-major | Policy-input reference values after the declared causal-input transform. |
| `reference_hold_mask` | `H` | For window element `j`, `1` exactly when the unclamped requested offset exceeds the inclusive span end and clamping repeats its final frame; the actual final frame before overrun remains `0`. Explicit hold frames stored in the artifact are ordinary frames and remain `0`. |
| `mode_one_hot` | `M` | Current mode in one frozen, globally ordered vocabulary. A scalar ordinal mode ID is prohibited. |
| `guard_latch_mask` | `G` | Post-query hysteresis state in one frozen, globally ordered transition-rule vocabulary. |
| `local_phase` | `1` | Inclusive normalized phase in `[0, 1]`; sine/cosine alone is prohibited because it aliases the endpoints. |
| `reference_frame_index` | `1` | Exact current frame index. |
| `reference_offset_frames` | `1` | Exact offset within the current mode span. |
| `dwell_steps` | `1` | Exact control-step dwell count in the current mode. |
| `query_index` | `1` | Exact episode oracle-query count. |
| `transition_kind_one_hot` | `4` | Exactly one of `none`, `advance`, `recovery`, or `rejoin`. |

The total width is `348 + 46H + M + G + 9`. Before allocation, the compiler
must enforce `2 <= H <= 64`, `1 <= M <= 128`, `0 <= G <= 1,024`, at most `128`
declared signals, at most `1,000,000` reference frames, and total width at most
`65,536`. The manifest must freeze `H`, `M`, `G`, slice offsets, bounds, dtype,
and both ordered vocabularies. Mode kind is
not a separate slice because it is exactly derivable from the frozen mode
vocabulary and program; adding a redundant three-value mode-kind encoding would
change this ABI. Every integer clock must be bounded by `10,000,000`, below
`2^24`, so it is represented
exactly in `float32`; larger values fail closed rather than round silently.
The four frame/dwell/query clock slices use the values in the emitted
`OracleResult`/`ModePhase`, before the runtime increments its next-query state.

Mode, phase, and clock are justified because identical 45D samples can occur in
different modes, at a hold, or after recovery/rejoin. They are command state,
not proof of successful control. Guard latches are required because hysteresis
can otherwise give the same current signal and mode two different next
transitions based on hidden history. Experiment 002 admits only zero-latency
signals that are exact deterministic functions of the frozen `348`-value base
observation and whose producer identity is bound. A delayed signal, an
all-substep value not derivable from those bytes, or any other hidden producer
state requires a new ABI that exposes the signal and complete delay-buffer state
to both actor and critic. Experiment 002 gives actor and critic the same vector.
Privileged oracle signals or an asymmetric critic require a new experiment
family.

An installation receipt must bind the assembler source and schema hash and
record reset/step slice hashes. A policy-side hook must prove that both actor
and critic received the declared slices. Producing or hashing the vector before
the worker consumes it is insufficient.

## Reference admission gate

The current numerical/schema validator is a kinematic candidate check. It is
not enough to authorize Experiment 002 training. The current `OracleProgram`
binds one `ReferenceIdentity`; it does not dispatch among an implicit set.
Experiment 002 therefore uses one newly materialized compound reference artifact
whose mode spans index its ordered component trajectories. Before implementation,
`ReferenceIdentity` must gain an exact robot identifier so identity is scoped by
`(robot_id, content_sha256, schema_sha256, compound_manifest_sha256)`. The
compound artifact receives new bytes and a new identity; its content-addressed
manifest preserves every ordered parent/component identity, full-root evidence
sidecar, and boundary. The authorizing runtime separately binds the exact Tier-D
certificate and certification-procedure hashes. A future design may instead
extend each mode with a robot-scoped component identity, but must not claim that
behavior under the current program schema.

The compound reference must have an immutable Tier-D admission certificate
binding:

- robot-scoped artifact identity, exact bytes, component manifest, and parent
  chain;
- the ordered 45D schema, joint mapping, units, root-frame semantics, cadence,
  and frozen MuJoCo XML/model/runtime hashes;
- deterministic quaternion-sequence unwrapping and adjacent-frame continuity,
  not only an independent canonical sign check at each frame;
- XML joint limits plus declared velocity and acceleration checks, including
  position/velocity finite-difference consistency for all joints and
  quaternion-derived root angular velocity;
- full root `x/y/z` position and velocity evidence even though the policy ABI
  intentionally omits absolute root `x/y`; root motion, named support geometry,
  expected contact intervals, and contact
  consistency;
- complete simulator/tracker feasibility evidence at all five physics
  substeps, with the exact evidence procedure and controller identified;
- mode spans, transition boundaries, phase-transfer semantics, and explicit
  pre/post/terminal holds;
- every allowed edge's source-state/target-entry pair under its exact
  phase-transfer mapping, or one explicitly bounded certified compatibility
  envelope covering every reachable pair; component-level certificates alone
  do not certify a normalized-phase jump into an arbitrary target frame; and
- a fresh identity and fresh admission for every crop, retime, retarget, or
  composition. A certified clip is never retimed in place.

Admission must also impose bounded frame/file/component counts before loading.
Until an actual time-varying reference passes this gate, only an
`interface_check` is permitted. Begin with a small manually curated pool of
admitted components, then materialize and admit the exact compound bytes.
Generated logic may select or compose only those immutable components, while
the current runtime consumes the resulting compound artifact.

## Causal-use interventions and closed-loop arms

Use the same predetermined saved checkpoint in four conditions, but keep two
estimands separate.

1. **Matched-state intervention.** Restore the same saved simulator state and
   the same cached untransformed oracle result, then replace only the `H x 45`
   actor-and-critic-visible values. Mode, phase, clocks, latches, base
   observation, and policy recurrent state remain byte-identical. This supports
   an action/output-sensitivity claim at common states.
2. **Closed-loop rollout.** Pair reset seeds, initial states, program/reference
   identities, horizon, perturbation schedule, evaluator, deterministic action
   rule, and transform contract. Each arm queries the same frozen state-dependent
   oracle on its own realized state and is graded against its own cached,
   untransformed pre-action `W_gt,j[t][1]`. Realized targets and metadata may
   diverge after actions diverge; that divergence is an outcome. The protected
   end-to-end completion, transition, recovery, and failure measures remain
   comparable, while raw tracking errors must retain their arm-local target
   provenance.

For an initial common-target behavioral degradation test, use a separately
declared exogenous playback program whose target schedule cannot depend on the
realized state. Do not freeze the exact arm's state-conditioned oracle trace and
call the result closed loop.

| Arm | Policy-visible numeric window | Matched-state target / closed-loop target |
|---|---|---|
| `C_exact` | Exact oracle window | Common cached snapshot / arm-local untransformed `W_gt,exact[t][1]` |
| `C_constant_frame_input` | Repetition of one frozen frame index inside the same compound artifact; no stationary or standing meaning is implied | Common cached snapshot / arm-local untransformed `W_gt,constant[t][1]` |
| `C_shuffle_input` | Exact whole-artifact seeded permutation, then windowed | Common cached snapshot / arm-local untransformed `W_gt,shuffle[t][1]` |
| `C_shift_input` | Exact frozen nonzero cyclic frame shift, then windowed | Common cached snapshot / arm-local untransformed `W_gt,shift[t][1]` |

Changing both the visible command and evaluator target at one matched state
changes the task and cannot identify command use. Each intervention manifest
must therefore carry separate `ground_truth_reference_sha256`,
`policy_input_reference_sha256`, and transformation-receipt hashes. During a
closed-loop rollout, the receipt additionally commits to the complete ordered
per-step `OracleResult` trace: program/reference identity, query index, mode,
phase/frame/offset/dwell clocks, window and mask, transition, and latch state.
It proves byte equality between the untransformed result consumed by the wrapper
and the target consumed by the evaluator. This prevents downstream
state-dependent divergence from being mistaken for a common target. The
constant-frame transform is derived from a declared frame inside the same
compound ground-truth artifact, not an unrelated stand artifact. A future
`neutral` or `stand` label requires separately admitted stationary velocity,
support, and contact semantics.

Before any evaluation, reject the four-arm block unless:

- every arm ID appears exactly once and every non-exempt manifest field is
  byte-identical at a matched-state intervention; closed-loop manifests match
  their frozen contracts and initial conditions rather than post-divergence
  realized trace bytes;
- all transformed inputs derive from the same admitted ground-truth reference;
- constant-frame bytes are valid under the 45D schema;
- shuffle is a complete permutation, is not the identity, and changes temporal
  adjacency;
- shift is nonzero, changes bytes, and is not a symmetry of the sequence;
- the reference exceeds a frozen pre-data, unit-aware temporal-variation floor;
  a generic numerical epsilon is not a meaningful-motion threshold; and
- at least one admitted evaluation sequence is withheld from PPO optimization,
  or the claim ceiling is explicitly limited to in-distribution sensitivity.

The independent unit is the policy-training seed. Evaluation seeds and the
four arms are paired repeated measures within a checkpoint; they do not
increase independent `n`. Run order must be seeded and balanced within each
checkpoint. Report every predetermined seed and arm. Lock the seed counts,
minimum effect, uncertainty method, missing-run rule, and expected direction
before collecting behavioral data.

A numeric-reference-use claim requires both:

1. matched-state action/output sensitivity when only the numeric window is
   changed; and
2. a frozen expected-direction paired degradation from `C_exact` to the three
   corrupted-input arms under the exogenous common-target protocol, or a
   separately named closed-loop effect on protected task/transition outcomes
   with arm-local target provenance.

Passing these tests does not prove that mode, phase, or clock metadata is used.
That requires a separately locked metadata ablation. It also does not prove
that an oracle is better than another oracle.

## Frozen versus authorable surfaces

| Frozen for one experiment family | Authorable only after lower gates pass |
|---|---|
| Humanoid XML, Gymnasium class, reset/termination, `TimeLimit`, frame skip, and all-substep instrumentation | Typed dispatch rules over admitted references |
| Base observation, complete oracle observation ABI, masks, producer identities, units, visibility, and required zero latency | Guard thresholds and hysteresis inside the typed DSL |
| Normalized/physical action mapping, tracker architecture, checkpoint rule, normalizers, PPO settings, seeds, and budget | Phase-transfer choice from the admitted enum |
| Tracking reward, zero task reward, protected evaluator, outcomes, thresholds, perturbations, and analysis | Recovery and rejoin edges within the admitted program schema |
| Compound-reference bytes, ordered component/parent lineage, admission certificate, and cadence | Mode-span selection/composition inside that same admitted compound artifact |
| Query/reward timing, mode vocabulary, code/dependency/platform hashes, and receipt schemas | No model, reward, tracker, simulator, evaluator, or reference-byte edits |

Declared signal metadata is not runtime proof. The executable adapter must bind
each producer implementation, show that the value is policy-observable, and
measure its age at the query boundary. A source or latency mismatch stops the
run.

## Oracle liveness and transition outcomes

Topological reachability to a terminal mode does not prove that guards will
ever fire. In Experiment 002, entering a terminal oracle mode records terminal
reach and holds the final reference through the fixed Gymnasium `TimeLimit`; it
does not terminate or truncate the plant episode. Oracle-driven environment
termination would change the frozen MDP and requires a separate design.

The runtime and evaluator must fail closed on:

- maximum dwell without an eligible transition;
- zero-progress loops, excess per-mode or per-edge visits, or failure to improve
  one frozen progress measure within a frozen query bound; an intentional
  task -> recovery -> rejoin -> task cycle is not itself a failure;
- transition chattering and priority ambiguity;
- missing, stale, privileged, or out-of-range signals;
- phase-transfer error beyond the locked boundary;
- recovery without a rejoin; and
- rejoin without same-episode task resumption.

Protected outcomes must include transition success, maximum boundary error,
phase error, resynchronization latency, stalls, chatter, loops, time to
failure, recovery/rejoin rate, and post-rejoin task-resumption rate. Thresholds
must be fixed before behavioral data; absent thresholds remain an explicit
blocked state.

The static evaluator cannot be reused by name alone. A dynamic evaluator must
grade the same cached ground-truth frame as the wrapper, use reference-relative
velocity maxima in any RMSE/max algebra, and apply a frozen mode/phase-specific
contact contract. The static two-foot floor allowlist cannot represent a
get-up or recovery phase that intentionally uses a hand or knee.

## Minimum real-Gym implementation gates

No long training claim is needed for these gates. Use the real instrumented
`Humanoid-v5` path and require positive and negative cases.

1. **Dynamic timing:** use a ramp/impulse reference through a mode transition;
   prove that observation `W[t]`, cached reward/evaluator target `W_gt[t][1]`,
   and next query `W[t+1]` are distinct and correctly ordered.
2. **Actor/critic installation:** prove exact slice values and hashes reach both
   networks; a swapped, omitted, stale, or masked slice must fail.
3. **Admitted rollout:** reset and complete a 1,000-control-step rollout with a
   time-varying admitted clip, finite observations, exact action mapping, and
   complete five-substep contact capture. This is execution and full-horizon
   liveness evidence, not a tracking-success result.
4. **Four-arm derivation:** reproduce all transform bytes from their seeds;
   reject a static reference, identity shuffle, zero/equivalent shift, invalid
   constant-frame quaternion, or changed ground-truth target.
5. **Same-checkpoint replay:** load one saved PPO checkpoint across all four
   matched-state interventions; prove that only policy-input reference bytes
   differ and test action sensitivity. In a separate closed-loop replay, prove
   matched initial conditions and frozen contracts while retaining each arm's
   realized untransformed target trace.
6. **Transition semantics:** cover minimum/maximum dwell, hysteresis, priority,
   phase transfer, terminal reach, and deliberate stall/chatter/loop failures.
7. **Recovery semantics:** force recovery, verify bounded rejoin and task
   resumption, and reject missing or stale signal producers.
8. **Admission negatives:** reject quaternion sign discontinuity, cadence or
   retime mismatch, joint-order/limit mismatch, inconsistent contact/support,
   broken lineage, producer mismatch, and latency mismatch.
9. **Runtime integrity:** retain source-drift, exact action endpoint,
   rollout-log-probability, deterministic save/load, `TimeLimit`, and all-step
   receipt checks; prove reset plus `N` actions yields exactly `N + 1` oracle
   queries and reject either terminal off-by-one direction.

## Explicit nonclaims

This draft and its interface tests do not establish:

- reference dynamics feasibility before a Tier-D certificate exists;
- successful training, natural motion, safety, energy efficiency, or
  station-keeping;
- causal use of mode, phase, clock, or transition metadata;
- superior transitions, recovery, or held-out composition;
- generated-oracle quality or superiority over manual/open-loop comparators;
- robustness outside the frozen Gymnasium MuJoCo Humanoid configuration; or
- paper-ready evidence.

Human approval, a locked pre-data decision rule, complete provenance, and the
specified behavioral receipts are required before any of those claims can be
considered.
