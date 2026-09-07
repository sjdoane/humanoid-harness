# Study 021: leave crouch when the progress guard becomes true

| progress | Study 020 is closed; both saved policies fail the full task. |
|---|---|
| bottleneck | Crouch exit waits for a loop boundary; whether immediate exit is trackable is unknown. |
| next step | Review this predata protocol and scorer, then run one exact control and one oracle-only candidate. |

Status: predata draft; no Study 021 dynamics or training has run.

## Question and limits

- Can the fixed tracker survive the earlier, state-triggered crouch-to-rise
  handover when the loop-boundary exit restriction is removed?
- One deterministic, zero-residual episode per arm. This is a local feasibility
  screen, not the sampled-policy robustness question or a training comparison.
- Study 020 motivates the question: some falls occur near rise and three occur
  inside before rise. Speed, deferral, pose and controller history covary; no
  isolated cause is established. The unchanged guard may become true at an
  unsuitable pose. Earlier is not assumed to be better.

## Only intervention

| Item | Exact O7b control | Candidate |
|---|---|---|
| Crouch segment `exit_at_loop_boundary` | `true` | Key absent; admitted default `false` |
| Oracle ID | Existing ID | New immutable ID |
| All other fields | Retained bytes/semantics | Unchanged |

- Parent config SHA: `ba45dba36ef88bdc522ed2115e46a4b3d876e00a627a089fd56ddf0de623e18a`.
- Freeze clips, crops, parent bytes, native cadence, entry-phase matching,
  inside loop, all guards/priorities/dwell, rise/after logic, task reward,
  tracking reward, tracker weights, MDP, reset, observations/actions and evaluator.
- Both arms: existing 2172-D loop probe-only runtime; seed `20260906`, zero
  training, up to 1,000 actions / 20 s. No sampled residual, scale change,
  retiming, guard sweep or stronger controller.

```text
INPUT: fixed clips + task + reward + zero residual + measured robot state
                      ↓
  inside progress guard >= 2.05 m ── boundary restriction true / absent
                      ↓
  frozen oracle guards + phase transfer → reference window → fixed tracker
                      ↓
OUTPUT: raw states/actions/contacts → independent handover and task metrics
                      └────────────────→ feedback for a separate next study
Frozen: plant/reset, other oracle fields, controller, reward and evaluation.
```

## Predeclared measurements and screen

- Guard satisfaction uses **pre-action** boundary `qpos[i], qvel[i]` and the
  same reset task frame as the oracle. It is the first command while inside
  with `x_travelled >= 2.05`, not the preceding post-step row's index.
- Actual switch is the command carrying `inside → rise`. Report both zero-based
  action index and one-based command number. Deferral is switch minus guard.
- Retained control: guard at action 227, switch at 239. Candidate prediction:
  switch at 227, zero deferral, rise selected phase 0.0 s. Check against a fresh
  control before dispatch. These expected boundaries are not candidate results.

| Screen row | Candidate requirement |
|---|---|
| Manipulation | First rise switch at guard action 227; zero deferral; selected rise phase 0.0 s |
| Survival | 20 s; zero falls; zero recorded non-foot ground-contact actions |
| Composition | Exactly three switches; before/inside/rise/after all execute |
| Exposure | Physical-region entry/exit and finish observed; at least 25 region samples |
| Recovery guardrail | No physical-region revisit while executing after |
| Tracking | Episode joint p95 <= 0.35 rad; roll-pitch p95 <= 0.25 rad |

- Require all screen rows, but a pass **does not admit training** or modify any
  of the eleven original task gates. Report those unchanged, including failures.
- Report guard/switch progress, pre-action course-frame speed, heading, lateral
  position, robot/source-target/destination-target heights, source/selected
  phase and normalized robot-to-destination pose distance. Distinguish target
  discontinuity from robot tracking error and use the native frame convention.
- Report rise dwell, fall/contact mode and time relative to rise, after-entry
  heading/lateral, region compliance/depth, speed errors and maximum lateral error.
  Missing rise/after stays null, not a zero-error or zero-deferral result.
- Keep every attempted arm and failed measurement. A deterministic fall is
  evidence about this handover; survival does not establish sampled robustness.
  No confidence interval, significance test, generalization or safety certificate.

## Exact prefix and identity

- Numeric equality before changed command: 227 action/reference rows and 228
  `qpos/qvel` boundaries. Reconstruct oracle windows and all transitions from
  retained states; verify recorded post-step reference targets.
- Changing the exit rule also changes crouch segment identity. At the existing
  before-to-inside transition (action 92), only the exact, recomputed
  `transition.segment_sha256` may differ in the frame prefix. Check each
  against its own admitted segment hash, then compare every other frame field.
  Do not discard all metadata or permit an arbitrary hash difference.
- At action 227, require the expected inside-versus-rise decision and changed
  numeric reference. A different numeric prefix, changed guard/phase, missing
  transition, wrong identity or unexpected differing field is an integrity error.
- Reset metadata may differ only in its exact admitted oracle identity;
  initial plant/base state must be identical.

## Order, authorization and stopping

1. Review and commit protocol/scorer/tests; pin all imported scoring helpers,
   clean execution source and executable tree. No old study source is edited.
2. Fable supplies one data-only proposal through the existing `g1 revise`
   firewall after the lock, bound to the exact retained O7b feedback. Include
   labeled rationale, prediction and falsifier; no evaluator editing authority.
3. Obtain a separate native reservation for a fresh control. Require its three
   output files to byte-match retained Study 015 and its rebuilt feedback to
   differ only by exact new manifest provenance.
4. Complete control verification in a separate successful call. Only then obtain
   the exact candidate reservation and launch a fresh candidate output directory.
5. Independently score both at unchanged source; retain failed task gates.
   End source freeze only after scoring/review. No automatic follow-on run.

- Fixed order is a verification dependency, not random assignment. Fresh full
  reset and exact control replay check carryover; one pair cannot estimate drift
  or between-seed uncertainty. All control samples remain correlated.
- Per arm: at most 1,200 s wall/CPU, 8 GiB RSS, 1 GiB output; at least 20 GiB
  free disk. One atomic heavy-job slot. No retries or cap expansion by outcome.
- A runtime/scorer change before data needs review and resealing; changes after
  data require a new study. A failed screen ends this bounded probe.

## Procedural guidance

Experimental-design skill: fixed single factor, explicit unit/replication limit,
fresh control, declared order, unchanged task gates and predata stopping rule.
This uses existing deterministic design/verification code; no DOE dependencies
are added to the training environment. Tooling is not evidence of efficacy.
