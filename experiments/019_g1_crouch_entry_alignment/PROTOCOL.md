# Study019: earlier crouch entry, unchanged reference library

| progress | Study018 is closed: baseline admission failed; no B launched. |
|---|---|
| bottleneck | The robot reaches the physical crouching region during a high part of the inside reference. |
| next step | Lock and test one 0.40 m entry guard against a fresh exact O7b control. |

## Question and scope

- Does an earlier, observed-progress transition improve crouching-region
  compliance by shifting the reference phase at physical entry?
- Exploratory, single-seed, zero-residual **oracle** screen. No training,
  dynamics certificate, held-out result, continuous phase-estimation claim,
  or automatic admission to a later training study.
- Prior evidence: Study018 post-step target exceeds the ceiling on15/24
  failed samples. This motivates the test; it does not establish causality.
- Fable supplies one feedback-linked, data-only oracle proposal after the
  protocol/scorer lock. Astra validates and integrates it through `g1 revise`.
  Proposal authors cannot edit the evaluator or these criteria.

## Only intervention

| Item | Control | Candidate |
|---|---|---|
| Before → inside observed-progress guard | 0.65 m | 0.40 m |
| Oracle ID | Existing O7b | New immutable candidate ID |
| Everything else | Exact native O7b | Identical to control |

- Control config SHA: `ba45dba36ef88bdc522ed2115e46a4b3d876e00a627a089fd56ddf0de623e18a`.
- Freeze all segment crops, parent motion bytes, native cadence, guard priority,
  dwell, entry-phase matching, inside loop, rise/after logic, reward, task,
  tracker weights, robot/reset, observations/actions, evaluator and seed20260906.
- Both arms: existing2172-D loop **probe-only** profile; zero training; one
  deterministic20 s episode each,1000 actions. This is not018's training runtime.
- No guard sweep or repeat selected by outcome. No retiming, reward revision,
  heading feedback, new controller or stronger residual in this comparison.

```text
INPUT: fixed O7b clips + r1 + task + zero residual
       observed progress → [entry guard: 0.65 OR 0.40 m]
                                  ↓
       frozen state machine/phase clock → reference window → frozen tracker
                                  ↓
OUTPUT: robot trace → independent region/phase/task measurements
                                  ↓
                       feedback to next declared study
Frozen: plant/reset, all other oracle fields, reward, tracker, evaluation.
```

## Predeclared measurements and decision

Physical-region rows use measured progress in [1.0,2.0), not the oracle mode.
Use all visits for compliance. First physical entry is the first post-step row
in that interval. Read height from the retained **post-step** target and
reconstruct its reported local phase at boundary i+1. The row's
`executed_phase_seconds` instead describes the pre-action command at index i;
report it separately. Do not substitute a pre-action future-window row.

| Row | Candidate requirement |
|---|---|
| Manipulation | First physical entry executes inside; local phase in[1.45,1.80] s; post-step target height<0.50 m |
| P1 | All-visit posture compliance≥0.85, measured against the unchanged0.60 m ceiling |
| Survival |20 s,0 falls,0 recorded non-foot ground contacts |
| Composition |Exactly3 switches; before/inside/rise/after all execute |
| Exposure |Region entry/exit and finish observed;≥25 physical-region samples |
| Recovery guardrail |No physical-region revisit while executing after |
| Tracking |Joint p95≤0.35 rad; roll-pitch p95≤0.25 rad |

- Screen passes only if every row passes. P1 is a **new exploratory alignment
  screen**, not a replacement for any original full-task gate or018 admission.
- Report all11 original task gates unchanged, minimum physical height, inside
  mean-speed deviation, overall speed MAE, maximum lateral error, after-entry
  heading, rise handover phase/heights, and signed unwrapped after heading.
- If manipulation passes but compliance<0.80, reject the predicted large
  alignment benefit for this setting. If compliance is[0.80,0.85), P1 still
  fails. If manipulation fails, the intended phase shift was not established.
- No predicted depth or heading benefit. A positive result supports only this
  guard's effect on this deterministic trace; earlier entry also changes state
  and controller history. It does not isolate phase from every coupled effect.
- One paired episode, with correlated samples and no uncertainty estimate.
  A future learned pair needs a newly declared zero/trained control.
- Keep the existing loop-boundary exit rule: satisfying progress≥2.05 m does
  not immediately force rise. Report first guard satisfaction, actual switch
  command, elapsed deferral, source phase and handover heights. A further loop
  caused by the unchanged boundary rule is reported, not a new screen failure;
  all survival, exposure, re-entry and tracking checks still apply.
- Before candidate data, retained O7b first reaches0.40 m after control step71
  (progress0.407975 m; step70 is0.392494 m). Confirm dispatch semantics in the
  scorer: expected first change at the72nd command (zero-based action/control
  index71), with71 identical executed rows and72 identical state boundaries.
  A differing prefix is an integrity failure.

## Verification and launch order

1. Commit/review protocol and small scorer. Pin their bytes, imported helpers,
   exact executable tree and final clean source in a predata seal.
2. Retain one Fable proposal, rationale, prediction and falsifier; publish its
   exact candidate config through the existing feedback/proposal firewall.
3. Obtain exact native approval for a fresh control. Run it at the sealed
   source; require all three retained Study015 outputs byte-exact.
4. Verify that control in a **separate successful call**. Only then obtain
   exact candidate approval and launch its fresh output directory.
5. Verify manifests, reservations, raw states, contacts, reconstructed oracle
   windows/targets, independent objectives and all frozen config fields.
   Require exact prefix states/actions/frames before the first changed command.
6. Publish both results and a recorded-state replay; retain failures. End the
   source freeze after completed scoring. Do not edit runtime during either run.

- Both arms share clean source/tree, initial base state and reset metadata.
  Source changes invalidate dispatch until a new predata seal and control.
- Separate native v2 reservation per arm; one heavy job at a time; each capped
  at1200 s CPU/wall,8 GiB memory, with20 GiB free disk and1 GiB output limit.
- A protocol change before data requires review and resealing. A change after
  data is a new study; no threshold repair can manufacture a pass.
